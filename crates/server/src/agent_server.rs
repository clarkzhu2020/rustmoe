//! Standalone Dev Agent server that uses an OpenAI-compatible upstream model.

use crate::agent::{
    delete_project, export_project, list_projects_with_active, list_skill_infos, mark_interrupted_projects,
    read_skill, run_agent, AgentConfig, AgentEvent, AgentModel, HttpAgent,
};
use axum::{
    extract::{DefaultBodyLimit, Path as AxumPath, State},
    http::{header, HeaderMap, StatusCode},
    response::{
        sse::{Event, KeepAlive, Sse},
        Html, IntoResponse, Response,
    },
    routing::{delete, get, post},
    Json, Router,
};
use futures_util::Stream;
use serde::Deserialize;
use serde_json::{json, Value};
use std::convert::Infallible;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokio::sync::mpsc;
use tower_http::cors::CorsLayer;

#[derive(Clone)]
pub struct HttpAgentConfig {
    pub addr: std::net::SocketAddr,
    pub config: AgentConfig,
    pub upstream: String,
    pub model: String,
    pub upstream_api_key: Option<String>,
    pub api_key: Option<String>,
}

#[derive(Clone)]
struct AppState {
    config: Arc<AgentConfig>,
    upstream: Arc<str>,
    model: Arc<str>,
    upstream_key: Option<Arc<str>>,
    api_key: Option<Arc<str>>,
    running: Arc<AtomicBool>,
    cancel: Arc<AtomicBool>,
}

pub async fn serve(cfg: HttpAgentConfig) -> anyhow::Result<()> {
    std::fs::create_dir_all(&cfg.config.root)?;
    std::fs::create_dir_all(&cfg.config.skills_dir)?;
    let _ = mark_interrupted_projects(&cfg.config.root);
    let state = AppState {
        config: Arc::new(cfg.config),
        upstream: Arc::from(cfg.upstream),
        model: Arc::from(cfg.model),
        upstream_key: cfg.upstream_api_key.map(Arc::from),
        api_key: cfg.api_key.map(Arc::from),
        running: Arc::new(AtomicBool::new(false)),
        cancel: Arc::new(AtomicBool::new(false)),
    };
    let router = Router::new()
        .route(
            "/",
            get(|| async { axum::response::Redirect::temporary("/agent") }),
        )
        .route("/agent", get(agent_page))
        .route("/agent/", get(agent_page))
        .route("/health", get(health))
        .route("/v1/models", get(models))
        .route("/v1/agent/run", post(agent_run))
        .route("/v1/agent/cancel", post(agent_cancel))
        .route("/v1/agent/projects", get(projects))
        .route("/v1/agent/projects/{name}", delete(delete_project_handler))
        .route("/v1/agent/projects/{name}/export", get(export_project_handler))
        .route("/v1/agent/skills", get(skills))
        .route("/v1/agent/skills/{name}", get(skill))
        .route("/chat", get(upstream_chat))
        .route("/status", get(upstream_status))
        .layer(DefaultBodyLimit::max(8 * 1024 * 1024))
        .layer(CorsLayer::permissive())
        .with_state(state.clone());
    let listener = tokio::net::TcpListener::bind(cfg.addr).await?;
    tracing::info!(addr = %cfg.addr, upstream = %state.upstream, "RustMoE fast agent listening");
    axum::serve(listener, router).await?;
    Ok(())
}

async fn agent_page() -> Html<&'static str> {
    Html(include_str!("../static/agent.html"))
}

fn authorize(state: &AppState, headers: &HeaderMap) -> Result<(), (StatusCode, Json<Value>)> {
    let Some(key) = &state.api_key else { return Ok(()) };
    let ok = headers
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))
        .map(|v| v == key.as_ref())
        .unwrap_or(false);
    if ok {
        Ok(())
    } else {
        Err((
            StatusCode::UNAUTHORIZED,
            Json(
                json!({"error": {"message": "missing or invalid bearer token", "type": "authentication_error"}}),
            ),
        ))
    }
}

async fn health(State(state): State<AppState>) -> Json<Value> {
    Json(json!({
        "status": "ok",
        "service": "rustmoe-fast-agent",
        "model": state.model.to_string(),
        "upstream": state.upstream.to_string(),
        "agent_root": state.config.root,
        "auth_required": state.api_key.is_some(),
    }))
}

async fn models(State(state): State<AppState>) -> Json<Value> {
    Json(json!({
        "object": "list",
        "data": [{
            "id": state.model.to_string(),
            "object": "model",
            "owned_by": "strata",
            "context_length": 262144,
            "max_context_length": 262144
        }]
    }))
}

async fn upstream_chat(State(state): State<AppState>) -> Response {
    let target = format!("{}/", state.upstream.trim_end_matches('/'));
    axum::response::Redirect::temporary(&target).into_response()
}

async fn upstream_status(State(state): State<AppState>) -> Json<Value> {
    Json(
        json!({"service": "rustmoe-fast-agent", "upstream": state.upstream.to_string(), "model": state.model.to_string()}),
    )
}

async fn projects(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    authorize(&state, &headers)?;
    let root = state.config.root.clone();
    let active = state.running.load(Ordering::Relaxed);
    let projects = tokio::task::spawn_blocking(move || list_projects_with_active(&root, active))
        .await
        .map_err(internal)?
        .map_err(bad_request)?;
    Ok(Json(json!({ "projects": projects })))
}

async fn delete_project_handler(
    State(state): State<AppState>,
    headers: HeaderMap,
    AxumPath(name): AxumPath<String>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    authorize(&state, &headers)?;
    let root = state.config.root.clone();
    tokio::task::spawn_blocking(move || delete_project(&root, &name))
        .await
        .map_err(internal)?
        .map_err(bad_request)?;
    Ok(Json(json!({"deleted": true})))
}

async fn export_project_handler(
    State(state): State<AppState>,
    headers: HeaderMap,
    AxumPath(name): AxumPath<String>,
) -> Result<Response, (StatusCode, Json<Value>)> {
    authorize(&state, &headers)?;
    let root = state.config.root.clone();
    let file_name = name.clone();
    let bytes = tokio::task::spawn_blocking(move || export_project(&root, &name))
        .await
        .map_err(internal)?
        .map_err(bad_request)?;
    let mut response_headers = HeaderMap::new();
    response_headers.insert(
        header::CONTENT_TYPE,
        header::HeaderValue::from_static("application/zip"),
    );
    if let Ok(value) = header::HeaderValue::from_str(&format!("attachment; filename=\"{file_name}.zip\"")) {
        response_headers.insert(header::CONTENT_DISPOSITION, value);
    }
    Ok((response_headers, bytes).into_response())
}

async fn skills(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    authorize(&state, &headers)?;
    let dir = state.config.skills_dir.clone();
    let skills = tokio::task::spawn_blocking(move || list_skill_infos(&dir))
        .await
        .map_err(internal)?;
    Ok(Json(json!({ "skills": skills })))
}

async fn skill(
    State(state): State<AppState>,
    headers: HeaderMap,
    AxumPath(name): AxumPath<String>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    authorize(&state, &headers)?;
    let dir = state.config.skills_dir.clone();
    let requested = name.clone();
    let body = tokio::task::spawn_blocking(move || read_skill(&dir, &requested))
        .await
        .map_err(internal)?
        .map_err(bad_request)?;
    Ok(Json(json!({"name": name, "content": body})))
}

#[derive(Debug, Deserialize)]
struct AgentRunRequest {
    task: String,
    #[serde(default)]
    workspace: Option<String>,
    #[serde(default)]
    max_steps: Option<usize>,
    #[serde(default)]
    resume: bool,
}

async fn agent_run(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(request): Json<AgentRunRequest>,
) -> Result<Sse<impl Stream<Item = Result<Event, Infallible>>>, (StatusCode, Json<Value>)> {
    authorize(&state, &headers)?;
    let task = request.task.trim().to_string();
    if task.is_empty() && !request.resume {
        return Err(bad_request("task must not be empty"));
    }
    if state.running.load(Ordering::Relaxed) {
        return Err((
            StatusCode::CONFLICT,
            Json(json!({"error": {"message": "agent is already running"}})),
        ));
    }
    let workspace = request.workspace.unwrap_or_default();
    let max_steps = request.max_steps.unwrap_or(state.config.max_steps).clamp(1, 64);
    let (tx, mut rx) = mpsc::unbounded_channel::<AgentEvent>();
    state.running.store(true, Ordering::Relaxed);
    state.cancel.store(false, Ordering::Relaxed);
    let running = state.running.clone();
    let cancel = state.cancel.clone();
    let config = (*state.config).clone();
    let upstream = state.upstream.to_string();
    let model = state.model.to_string();
    let upstream_key = state.upstream_key.as_ref().map(|s| s.to_string());
    let resume = request.resume;
    tokio::task::spawn_blocking(move || {
        struct Guard(Arc<AtomicBool>);
        impl Drop for Guard {
            fn drop(&mut self) {
                self.0.store(false, Ordering::Relaxed);
            }
        }
        let _guard = Guard(running);
        let agent = HttpAgent::new(upstream, model, upstream_key);
        run_agent(
            AgentModel::Http(agent),
            config,
            workspace,
            task,
            max_steps,
            resume,
            cancel,
            tx,
        );
    });
    let stream = async_stream::stream! {
        while let Some(event) = rx.recv().await {
            let data = serde_json::to_string(&event).unwrap_or_else(|_| "{\"type\":\"error\",\"message\":\"serialize\"}".into());
            yield Ok(Event::default().event("agent").data(data));
        }
    };
    Ok(Sse::new(stream).keep_alive(KeepAlive::default()))
}

async fn agent_cancel(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    authorize(&state, &headers)?;
    state.cancel.store(true, Ordering::Relaxed);
    Ok(Json(json!({"cancelled": true})))
}

fn internal<E: std::fmt::Display>(error: E) -> (StatusCode, Json<Value>) {
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        Json(json!({"error": {"message": error.to_string(), "type": "server_error"}})),
    )
}

fn bad_request<E: std::fmt::Display>(error: E) -> (StatusCode, Json<Value>) {
    (
        StatusCode::BAD_REQUEST,
        Json(json!({"error": {"message": error.to_string(), "type": "invalid_request_error"}})),
    )
}
