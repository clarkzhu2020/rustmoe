//! OpenAI-compatible HTTP surface with SSE streaming and runtime metrics.

use crate::agent::{
    delete_project, export_project, list_projects_with_active, list_skill_infos, read_skill, run_agent,
    AgentConfig, AgentEvent,
};
use crate::runtime::{Runtime, RuntimeInfo};
use axum::{
    extract::{DefaultBodyLimit, Extension, Path as AxumPath, State},
    http::{header, HeaderMap, StatusCode, Uri},
    response::{
        sse::{Event, KeepAlive, Sse},
        Html, IntoResponse, Response,
    },
    routing::{delete, get, post},
    Json, Router,
};
use rustmoe_inference::{FinishReason, GenerationConfig, GenerationEvent, GenerationResult, InferenceError};
use rustmoe_memory::HardwareProfile;
use rustmoe_tokenizer::{format_chat_deepseek_v41, format_chat_simple, ChatMessage, ChatRole};
use serde::Deserialize;
use serde_json::{json, Value};
use std::convert::Infallible;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::mpsc;
use tower_http::{cors::CorsLayer, trace::TraceLayer};

pub type AppState = Arc<Runtime>;

pub fn build_router(runtime: AppState) -> Router {
    build_router_with_agent(runtime, AgentConfig::default())
}

pub fn build_router_with_agent(runtime: AppState, agent: AgentConfig) -> Router {
    Router::new()
        .route("/", get(root))
        .route("/health", get(health))
        .route("/status", get(status_page))
        .route("/status/", get(status_page))
        .route("/chat", get(chat_page))
        .route("/chat/", get(chat_page))
        .route("/agent", get(agent_page))
        .route("/agent/", get(agent_page))
        .route("/metrics", get(metrics))
        .route("/v1", get(v1_root))
        .route("/v1/", get(v1_root))
        .route("/v1/models", get(models))
        .route("/v1/runtime", get(runtime_info))
        .route("/v1/system", get(system_info))
        .route("/v1/telemetry", get(telemetry))
        .route("/v1/agent/run", post(agent_run))
        .route("/v1/agent/cancel", post(agent_cancel))
        .route("/v1/agent/projects", get(agent_projects))
        .route("/v1/agent/skills", get(agent_skills))
        .route("/v1/agent/skills/{name}", get(agent_skill))
        .route("/v1/agent/projects/{name}", delete(agent_delete_project))
        .route("/v1/agent/projects/{name}/export", get(agent_export_project))
        .route("/v1/tokenize", post(tokenize))
        .route("/v1/cache/reset", post(cache_reset))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/chat/completions/", post(chat_completions))
        .route("/v1/completions", post(completions))
        .route("/v1/completions/", post(completions))
        .route("/v1/models/", get(models))
        .route("/chat/completions", post(chat_completions))
        .route("/chat/completions/", post(chat_completions))
        .route("/completions", post(completions))
        .route("/completions/", post(completions))
        .route("/models", get(models))
        .fallback(not_found)
        .layer(DefaultBodyLimit::max(8 * 1024 * 1024))
        .layer(Extension(Arc::new(agent)))
        .layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http())
        .with_state(runtime)
}

async fn root(State(rt): State<AppState>) -> Json<Value> {
    Json(json!({
        "name": "RustMoE",
        "version": env!("CARGO_PKG_VERSION"),
        "backend": rt.info.backend,
        "simulation": rt.info.simulation,
        "chat_format": rt.info.chat_format,
        "endpoints": ["/chat", "/agent", "/status", "/health", "/v1/models", "/v1/chat/completions", "/v1/completions", "/chat/completions", "/completions", "/v1/system", "/v1/telemetry", "/v1/agent/run", "/v1/agent/cancel", "/v1/agent/projects", "/v1/runtime", "/metrics"]
    }))
}

async fn not_found(uri: Uri) -> (StatusCode, Json<Value>) {
    tracing::warn!("HTTP 404 for {}", uri.path());
    (
        StatusCode::NOT_FOUND,
        Json(json!({
            "error": {
                "message": format!(
                    "no route for {}; POST to /v1/chat/completions or /v1/completions",
                    uri.path()
                ),
                "type": "invalid_request_error",
                "code": "not_found"
            },
            "path": uri.path(),
            "endpoints": ["/chat", "/agent", "/status", "/health", "/v1/models", "/v1/chat/completions", "/v1/completions", "/chat/completions", "/completions", "/v1/system", "/v1/telemetry", "/v1/agent/run", "/v1/agent/cancel", "/v1/agent/projects", "/v1/runtime", "/metrics"]
        })),
    )
}

async fn status_page() -> Html<&'static str> {
    Html(include_str!("../static/status.html"))
}

async fn chat_page() -> Html<&'static str> {
    Html(include_str!("../static/chat.html"))
}

async fn agent_page() -> Html<&'static str> {
    Html(include_str!("../static/agent.html"))
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
    State(rt): State<AppState>,
    Extension(agent): Extension<Arc<AgentConfig>>,
    headers: HeaderMap,
    Json(request): Json<AgentRunRequest>,
) -> Result<Sse<impl futures_util::Stream<Item = Result<Event, Infallible>>>, ApiError> {
    authorize(&headers, &rt)?;
    let task = request.task.trim().to_string();
    let resume = request.resume;
    if task.is_empty() && !resume {
        return Err(ApiError::bad_request("invalid_task", "task must not be empty"));
    }
    let workspace = request.workspace.unwrap_or_default();
    let max_steps = request.max_steps.unwrap_or(agent.max_steps).clamp(1, 64);
    let (tx, mut rx) = mpsc::unbounded_channel::<AgentEvent>();
    let runtime = rt.clone();
    let config = (*agent).clone();
    let cancel = rt.agent_cancel.clone();
    cancel.store(false, std::sync::atomic::Ordering::Relaxed);
    rt.agent_running.store(true, std::sync::atomic::Ordering::Relaxed);
    let running = rt.agent_running.clone();
    tokio::task::spawn_blocking(move || {
        let _guard = RunningGuard(running);
        run_agent(runtime, config, workspace, task, max_steps, resume, cancel, tx);
    });
    let stream = async_stream::stream! {
        while let Some(event) = rx.recv().await {
            let data = serde_json::to_string(&event).unwrap_or_else(|_| "{\"type\":\"error\",\"message\":\"serialize\"}".into());
            yield Ok(Event::default().event("agent").data(data));
        }
    };
    Ok(Sse::new(stream).keep_alive(KeepAlive::default()))
}

struct RunningGuard(Arc<std::sync::atomic::AtomicBool>);

impl Drop for RunningGuard {
    fn drop(&mut self) {
        self.0.store(false, std::sync::atomic::Ordering::Relaxed);
    }
}

async fn agent_cancel(State(rt): State<AppState>, headers: HeaderMap) -> Result<Json<Value>, ApiError> {
    authorize(&headers, &rt)?;
    rt.agent_cancel.store(true, std::sync::atomic::Ordering::Relaxed);
    Ok(Json(json!({"cancelled": true})))
}

async fn agent_skills(
    State(rt): State<AppState>,
    Extension(agent): Extension<Arc<AgentConfig>>,
    headers: HeaderMap,
) -> Result<Json<Value>, ApiError> {
    authorize(&headers, &rt)?;
    let dir = agent.skills_dir.clone();
    let skills = tokio::task::spawn_blocking(move || list_skill_infos(&dir))
        .await
        .map_err(|e| ApiError::internal("agent_skills_task", e.to_string()))?;
    Ok(Json(json!({ "skills": skills })))
}

async fn agent_skill(
    State(rt): State<AppState>,
    Extension(agent): Extension<Arc<AgentConfig>>,
    headers: HeaderMap,
    AxumPath(name): AxumPath<String>,
) -> Result<Json<Value>, ApiError> {
    authorize(&headers, &rt)?;
    let dir = agent.skills_dir.clone();
    let requested = name.clone();
    let body = tokio::task::spawn_blocking(move || read_skill(&dir, &requested))
        .await
        .map_err(|e| ApiError::internal("agent_skill_task", e.to_string()))?
        .map_err(|e| ApiError::bad_request("agent_skill", e))?;
    Ok(Json(json!({ "name": name, "content": body })))
}

async fn agent_projects(
    State(rt): State<AppState>,
    Extension(agent): Extension<Arc<AgentConfig>>,
    headers: HeaderMap,
) -> Result<Json<Value>, ApiError> {
    authorize(&headers, &rt)?;
    let root = agent.root.clone();
    let active = rt.agent_running.load(std::sync::atomic::Ordering::Relaxed);
    let projects = tokio::task::spawn_blocking(move || list_projects_with_active(&root, active))
        .await
        .map_err(|e| ApiError::internal("agent_projects_task", e.to_string()))?
        .map_err(|e| ApiError::bad_request("agent_project", e))?;
    Ok(Json(json!({ "projects": projects })))
}

async fn agent_delete_project(
    State(rt): State<AppState>,
    Extension(agent): Extension<Arc<AgentConfig>>,
    headers: HeaderMap,
    AxumPath(name): AxumPath<String>,
) -> Result<Json<Value>, ApiError> {
    authorize(&headers, &rt)?;
    let root = agent.root.clone();
    tokio::task::spawn_blocking(move || delete_project(&root, &name))
        .await
        .map_err(|e| ApiError::internal("agent_delete_task", e.to_string()))?
        .map_err(|e| ApiError::bad_request("agent_project", e))?;
    Ok(Json(json!({ "deleted": true })))
}

async fn agent_export_project(
    State(rt): State<AppState>,
    Extension(agent): Extension<Arc<AgentConfig>>,
    headers: HeaderMap,
    AxumPath(name): AxumPath<String>,
) -> Result<Response, ApiError> {
    authorize(&headers, &rt)?;
    let root = agent.root.clone();
    let file_name = name.clone();
    let bytes = tokio::task::spawn_blocking(move || export_project(&root, &name))
        .await
        .map_err(|e| ApiError::internal("agent_export_task", e.to_string()))?
        .map_err(|e| ApiError::bad_request("agent_project", e))?;
    let mut response_headers = HeaderMap::new();
    response_headers.insert(
        header::CONTENT_TYPE,
        header::HeaderValue::from_static("application/zip"),
    );
    let disposition = format!("attachment; filename=\"{file_name}.zip\"");
    if let Ok(value) = header::HeaderValue::from_str(&disposition) {
        response_headers.insert(header::CONTENT_DISPOSITION, value);
    }
    Ok((response_headers, bytes).into_response())
}

async fn v1_root(State(rt): State<AppState>) -> Json<Value> {
    Json(json!({
        "object": "api",
        "backend": rt.info.backend,
        "model": rt.info.model_id,
        "endpoints": ["/chat", "/agent", "/status", "/v1/chat/completions", "/v1/completions", "/v1/models", "/v1/runtime", "/v1/system", "/v1/telemetry"]
    }))
}

async fn health(State(rt): State<AppState>) -> Json<Value> {
    Json(json!({
        "status": "ok",
        "model": rt.info.model_id,
        "backend": rt.info.backend,
        "simulation": rt.info.simulation,
        "auth_required": rt.api_key.is_some(),
        "uptime_secs": rt.uptime_secs(),
    }))
}

async fn metrics(State(rt): State<AppState>) -> Response {
    (
        [(header::CONTENT_TYPE, "text/plain; version=0.0.4")],
        rt.metrics.render_prometheus(),
    )
        .into_response()
}

async fn models(State(rt): State<AppState>) -> Json<Value> {
    Json(json!({
        "object": "list",
        "data": [{
            "id": rt.info.model_id,
            "object": "model",
            "created": rt.info.started_unix,
            "owned_by": "rustmoe",
            "context_length": rt.info.context,
            "max_context_length": rt.info.context_limit,
            "context_window": rt.info.context,
            "max_model_len": rt.info.context,
            "rustmoe": {
                "architecture": rt.info.architecture,
                "backend": rt.info.backend,
                "simulation": rt.info.simulation,
                "context_length": rt.info.context,
                "chat_format": rt.info.chat_format,
                "tensor_count": rt.info.tensor_count,
            }
        }]
    }))
}

async fn telemetry(State(rt): State<AppState>, headers: HeaderMap) -> Result<Json<Value>, ApiError> {
    authorize(&headers, &rt)?;
    let monitor = rt.monitor.clone();
    let system = tokio::task::spawn_blocking(move || monitor.sample_system())
        .await
        .map_err(|e| ApiError::internal("telemetry_task", e.to_string()))?;
    Ok(Json(json!({
        "timestamp": system.timestamp,
        "system": system,
        "activity": rt.monitor.activity(),
        "recent": rt.monitor.recent(),
        "metrics": rt.metrics.snapshot(),
    })))
}

async fn system_info(State(rt): State<AppState>, headers: HeaderMap) -> Result<Json<Value>, ApiError> {
    authorize(&headers, &rt)?;
    let hardware = tokio::task::spawn_blocking(|| HardwareProfile::detect(None))
        .await
        .map_err(|e| ApiError::internal("hardware_task", e.to_string()))?;
    Ok(Json(json!({ "hardware": hardware })))
}

async fn runtime_info(State(rt): State<AppState>, headers: HeaderMap) -> Result<Json<Value>, ApiError> {
    authorize(&headers, &rt)?;
    let metrics = rt.metrics.snapshot();
    let (scheduler, cache, backend_cache, busy) = match rt.engine.try_lock() {
        Ok(engine) => (
            json!(engine.scheduler().stats()),
            json!(engine.scheduler().cache_snapshot()),
            engine.backend_cache_stats(),
            false,
        ),
        Err(_) => (Value::Null, Value::Null, None, true),
    };
    let info: &RuntimeInfo = &rt.info;
    Ok(Json(json!({
        "runtime": info,
        "scheduler": scheduler,
        "cache": cache,
        "backend_cache": backend_cache,
        "busy": busy,
        "auth_required": rt.api_key.is_some(),
        "metrics": metrics,
        "uptime_secs": rt.uptime_secs(),
    })))
}

#[derive(Debug, Deserialize)]
struct TokenizeRequest {
    text: String,
    #[serde(default)]
    add_special: bool,
}

async fn tokenize(
    State(rt): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<TokenizeRequest>,
) -> Result<Json<Value>, ApiError> {
    authorize(&headers, &rt)?;
    let ids = if req.add_special {
        rt.tokenizer.encode(&req.text)
    } else {
        rt.tokenizer.encode_raw(&req.text)
    };
    let tokens: Vec<&str> = ids
        .iter()
        .map(|id| rt.tokenizer.token_text(*id).unwrap_or("<invalid>"))
        .collect();
    Ok(Json(json!({
        "ids": ids,
        "tokens": tokens,
        "decoded": rt.tokenizer.decode(&ids),
        "pre_tokenizer": rt.tokenizer.pre_tokenizer,
        "pre_tokenizer_exact": rt.tokenizer.pre_tokenizer_exact,
    })))
}

async fn cache_reset(State(rt): State<AppState>, headers: HeaderMap) -> Result<Json<Value>, ApiError> {
    authorize(&headers, &rt)?;
    let mut engine = rt.engine.lock().await;
    engine.scheduler_mut().reset_stats();
    let cache = engine.scheduler().cache_snapshot();
    drop(engine);
    Ok(Json(json!({
        "reset": true,
        "cache": cache,
    })))
}

#[derive(Debug, Deserialize)]
struct ChatRequest {
    model: Option<String>,
    messages: Vec<ChatMessageDto>,
    #[serde(flatten)]
    params: Params,
}

#[derive(Debug, Deserialize)]
struct ChatMessageDto {
    role: String,
    #[serde(default)]
    content: Value,
}

#[derive(Debug, Deserialize)]
struct CompletionRequest {
    model: Option<String>,
    prompt: Value,
    #[serde(flatten)]
    params: Params,
}

#[derive(Debug, Deserialize, Default)]
struct Params {
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    top_p: Option<f32>,
    #[serde(default)]
    top_k: Option<usize>,
    #[serde(default)]
    min_p: Option<f32>,
    #[serde(default)]
    repetition_penalty: Option<f32>,
    #[serde(default, alias = "max_completion_tokens")]
    max_tokens: Option<usize>,
    #[serde(default)]
    seed: Option<u64>,
    #[serde(default)]
    stop: Option<StopSpec>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum StopSpec {
    One(String),
    Many(Vec<String>),
}

async fn chat_completions(
    State(rt): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<ChatRequest>,
) -> Result<Response, ApiError> {
    authorize(&headers, &rt)?;
    check_model(req.model.as_deref(), &rt)?;
    let prompt = chat_prompt(&req.messages, &rt.info)?;
    let cfg = build_config(&rt, &req.params)?;
    if req.params.stream {
        let id = rt.next_id("chatcmpl");
        let created = now_unix();
        Ok(chat_stream(rt, prompt, cfg, id, created).into_response())
    } else {
        let result = run_generate(rt.clone(), prompt, cfg).await?;
        let body = json!({
            "id": rt.next_id("chatcmpl"),
            "object": "chat.completion",
            "created": now_unix(),
            "model": rt.info.model_id,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": result.text},
                "finish_reason": api_finish(result.finish_reason),
            }],
            "usage": usage(&result),
            "rustmoe": runtime_details(&result),
        });
        Ok(with_backend_header(
            Json(body).into_response(),
            rt.info.simulation,
            &rt.info.backend,
        ))
    }
}

async fn completions(
    State(rt): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<CompletionRequest>,
) -> Result<Response, ApiError> {
    authorize(&headers, &rt)?;
    check_model(req.model.as_deref(), &rt)?;
    let prompt = match req.prompt {
        Value::String(s) => s,
        _ => {
            return Err(ApiError::bad_request(
                "unsupported_prompt",
                "RustMoE v0.1 accepts a single string prompt",
            ))
        }
    };
    let cfg = build_config(&rt, &req.params)?;
    if req.params.stream {
        let id = rt.next_id("cmpl");
        let created = now_unix();
        Ok(completion_stream(rt, prompt, cfg, id, created).into_response())
    } else {
        let result = run_generate(rt.clone(), prompt, cfg).await?;
        let body = json!({
            "id": rt.next_id("cmpl"),
            "object": "text_completion",
            "created": now_unix(),
            "model": rt.info.model_id,
            "choices": [{
                "index": 0,
                "text": result.text,
                "finish_reason": api_finish(result.finish_reason),
                "logprobs": null,
            }],
            "usage": usage(&result),
            "rustmoe": runtime_details(&result),
        });
        Ok(with_backend_header(
            Json(body).into_response(),
            rt.info.simulation,
            &rt.info.backend,
        ))
    }
}

async fn run_generate(
    rt: AppState,
    prompt: String,
    cfg: GenerationConfig,
) -> Result<GenerationResult, ApiError> {
    let prompt_tokens = rt.tokenizer.encode(&prompt).len() as u64;
    let context_before = rt.metrics.snapshot().context_tokens;
    rt.monitor.start(prompt_tokens, context_before);
    let engine = rt.engine.clone();
    let monitor_cb = rt.monitor.clone();
    let joined = tokio::task::spawn_blocking(move || {
        let mut guard = engine.blocking_lock();
        guard.generate_with(&prompt, &cfg, |event| {
            match &event {
                GenerationEvent::Token(t) => monitor_cb.token(&t.text),
                GenerationEvent::Text { text } => monitor_cb.token(text),
                GenerationEvent::Finished { .. } => {}
            }
            true
        })
    })
    .await
    .map_err(|e| ApiError::internal("generation_task", e.to_string()))?;
    match joined {
        Ok(result) => {
            rt.monitor.finish(&result);
            Ok(result)
        }
        Err(error) => {
            rt.monitor.fail();
            Err(ApiError::from(error))
        }
    }
}

fn chat_prompt(messages: &[ChatMessageDto], info: &RuntimeInfo) -> Result<String, ApiError> {
    if messages.is_empty() {
        return Err(ApiError::bad_request(
            "invalid_messages",
            "messages must not be empty",
        ));
    }
    let mut out = Vec::with_capacity(messages.len());
    for message in messages {
        let role = match message.role.as_str() {
            "system" => ChatRole::System,
            "user" => ChatRole::User,
            "assistant" => ChatRole::Assistant,
            other => {
                return Err(ApiError::bad_request(
                    "unsupported_role",
                    format!("role {other:?} is not supported by the text simulation"),
                ))
            }
        };
        let content = content_text(&message.content).map_err(|e| {
            ApiError::bad_request(
                "invalid_content",
                format!("role {:?}: {}", message.role, e.message),
            )
        })?;
        out.push(ChatMessage { role, content });
    }
    if info.backend == "llama.cpp" {
        Ok(format_chat_deepseek_v41(&out, true))
    } else {
        Ok(format_chat_simple(&out, true))
    }
}

fn content_text(content: &Value) -> Result<String, ApiError> {
    match content {
        Value::String(s) => Ok(s.clone()),
        // OpenAI tool-call assistant messages may carry null content. We do not
        // execute tools, but accepting the empty text avoids a spurious 400 for
        // clients that replay such a message.
        Value::Null => Ok(String::new()),
        // A single text-part object is not the OpenAI wire shape, but some
        // clients send it. Accept it rather than rejecting the whole chat.
        Value::Object(part) => Ok(part
            .get("text")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string()),
        Value::Array(parts) => {
            let mut out = String::new();
            for part in parts {
                if let Some(text) = part.as_str() {
                    out.push_str(text);
                    continue;
                }
                let kind = part.get("type").and_then(Value::as_str).unwrap_or("text");
                if kind != "text" {
                    return Err(ApiError::bad_request(
                        "unsupported_content",
                        format!("content part {kind:?} is not supported by the text simulation"),
                    ));
                }
                if let Some(text) = part.get("text").and_then(Value::as_str) {
                    out.push_str(text);
                }
            }
            Ok(out)
        }
        other => Err(ApiError::bad_request(
            "invalid_content",
            format!(
                "message content must be a string or an array of text parts (got {})",
                json_kind(other)
            ),
        )),
    }
}

fn json_kind(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

fn build_config(rt: &Runtime, params: &Params) -> Result<GenerationConfig, ApiError> {
    let mut stop_token_ids = Vec::new();
    let stops: Vec<&str> = match &params.stop {
        None => Vec::new(),
        Some(StopSpec::One(s)) => vec![s.as_str()],
        Some(StopSpec::Many(v)) => v.iter().map(String::as_str).collect(),
    };
    for stop in stops {
        if stop.is_empty() {
            continue;
        }
        let ids = rt.tokenizer.encode_raw(stop);
        if ids.len() != 1 {
            return Err(ApiError::bad_request(
                "unsupported_stop",
                "stop strings must tokenize to exactly one token in v0.1",
            ));
        }
        stop_token_ids.push(ids[0]);
    }
    let cfg = GenerationConfig {
        max_tokens: params.max_tokens.unwrap_or(128),
        stop_token_ids,
        temperature: params.temperature.unwrap_or(1.0),
        top_p: params.top_p.unwrap_or(1.0),
        top_k: params.top_k.unwrap_or(0),
        min_p: params.min_p.unwrap_or(0.0),
        repetition_penalty: params.repetition_penalty.unwrap_or(1.0),
        repetition_last_n: 64,
        seed: params.seed.unwrap_or(0xD5E1_4A11_1017_2026),
    };
    cfg.validate().map_err(ApiError::from)?;
    Ok(cfg)
}

fn check_model(model: Option<&str>, rt: &Runtime) -> Result<(), ApiError> {
    const ALIASES: &[&str] = &[
        "default",
        "deepseek",
        "deepseek-chat",
        "deepseek-reasoner",
        "deepseek-v4.1",
        "deepseek-v4.1-flash",
        "deepseek-v4.1-flash-1017",
    ];
    if let Some(model) = model {
        let requested = model.to_ascii_lowercase();
        let served = rt.info.model_id.to_ascii_lowercase();
        if requested != served && !ALIASES.contains(&requested.as_str()) {
            return Err(ApiError {
                status: StatusCode::NOT_FOUND,
                kind: "model_not_found",
                message: format!(
                    "model {model:?} is not served; use {:?} (or deepseek-chat)",
                    rt.info.model_id
                ),
            });
        }
        if requested != served {
            tracing::debug!("model alias {model:?} maps to {:?}", rt.info.model_id);
        }
    }
    Ok(())
}

fn authorize(headers: &HeaderMap, rt: &Runtime) -> Result<(), ApiError> {
    let Some(expected) = &rt.api_key else {
        return Ok(());
    };
    let provided = headers
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    if provided.strip_prefix("Bearer ") == Some(expected.as_ref()) {
        Ok(())
    } else {
        Err(ApiError {
            status: StatusCode::UNAUTHORIZED,
            kind: "invalid_api_key",
            message: "missing or invalid bearer token".into(),
        })
    }
}

fn chat_stream(
    rt: AppState,
    prompt: String,
    cfg: GenerationConfig,
    id: String,
    created: u64,
) -> Sse<impl futures_util::Stream<Item = Result<Event, Infallible>>> {
    let (tx, mut rx) = mpsc::unbounded_channel::<Result<GenerationEvent, String>>();
    let prompt_tokens = rt.tokenizer.encode(&prompt).len() as u64;
    let context_before = rt.metrics.snapshot().context_tokens;
    rt.monitor.start(prompt_tokens, context_before);
    let monitor = rt.monitor.clone();
    let engine = rt.engine.clone();
    let tx_task = tx.clone();
    tokio::task::spawn_blocking(move || {
        let mut guard = engine.blocking_lock();
        let monitor_cb = monitor.clone();
        let result = guard.generate_with(&prompt, &cfg, |event| {
            match &event {
                GenerationEvent::Token(t) => monitor_cb.token(&t.text),
                GenerationEvent::Text { text } => monitor_cb.token(text),
                GenerationEvent::Finished { .. } => {}
            }
            tx_task.send(Ok(event)).is_ok()
        });
        match result {
            Ok(result) => monitor.finish(&result),
            Err(e) => {
                monitor.fail();
                let _ = tx_task.send(Err(e.to_string()));
            }
        }
    });
    drop(tx);
    let model = rt.info.model_id.clone();
    let stream = async_stream::stream! {
        let initial = json!({
            "id": id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": null}],
            "rustmoe": {"simulation": rt.info.simulation, "backend": rt.info.backend}
        });
        yield Ok(Event::default().data(initial.to_string()));
        let mut finish: Option<FinishReason> = None;
        while let Some(item) = rx.recv().await {
            match item {
                Ok(GenerationEvent::Token(t)) => {
                    if !t.text.is_empty() {
                        let chunk = json!({
                            "id": id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model,
                            "choices": [{"index": 0, "delta": {"content": t.text}, "finish_reason": null}]
                        });
                        yield Ok(Event::default().data(chunk.to_string()));
                    }
                }
                Ok(GenerationEvent::Text { text }) => {
                    if !text.is_empty() {
                        let chunk = json!({
                            "id": id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model,
                            "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": null}]
                        });
                        yield Ok(Event::default().data(chunk.to_string()));
                    }
                }
                Ok(GenerationEvent::Finished { reason, .. }) => {
                    finish = Some(reason);
                }
                Err(error) => {
                    let chunk = json!({"error": {"message": error, "type": "server_error"}});
                    yield Ok(Event::default().data(chunk.to_string()));
                    break;
                }
            }
        }
        let final_chunk = json!({
            "id": id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": api_finish(finish.unwrap_or(FinishReason::Stop))}]
        });
        yield Ok(Event::default().data(final_chunk.to_string()));
        yield Ok(Event::default().data("[DONE]"));
    };
    Sse::new(stream).keep_alive(KeepAlive::default())
}

fn completion_stream(
    rt: AppState,
    prompt: String,
    cfg: GenerationConfig,
    id: String,
    created: u64,
) -> Sse<impl futures_util::Stream<Item = Result<Event, Infallible>>> {
    let (tx, mut rx) = mpsc::unbounded_channel::<Result<GenerationEvent, String>>();
    let prompt_tokens = rt.tokenizer.encode(&prompt).len() as u64;
    let context_before = rt.metrics.snapshot().context_tokens;
    rt.monitor.start(prompt_tokens, context_before);
    let monitor = rt.monitor.clone();
    let engine = rt.engine.clone();
    let tx_task = tx.clone();
    tokio::task::spawn_blocking(move || {
        let mut guard = engine.blocking_lock();
        let monitor_cb = monitor.clone();
        let result = guard.generate_with(&prompt, &cfg, |event| {
            match &event {
                GenerationEvent::Token(t) => monitor_cb.token(&t.text),
                GenerationEvent::Text { text } => monitor_cb.token(text),
                GenerationEvent::Finished { .. } => {}
            }
            tx_task.send(Ok(event)).is_ok()
        });
        match result {
            Ok(result) => monitor.finish(&result),
            Err(e) => {
                monitor.fail();
                let _ = tx_task.send(Err(e.to_string()));
            }
        }
    });
    drop(tx);
    let model = rt.info.model_id.clone();
    let stream = async_stream::stream! {
        let mut finish: Option<FinishReason> = None;
        while let Some(item) = rx.recv().await {
            match item {
                Ok(GenerationEvent::Token(t)) => {
                    let chunk = json!({
                        "id": id,
                        "object": "text_completion",
                        "created": created,
                        "model": model,
                        "choices": [{"index": 0, "text": t.text, "finish_reason": null, "logprobs": null}]
                    });
                    yield Ok(Event::default().data(chunk.to_string()));
                }
                Ok(GenerationEvent::Text { text }) => {
                    let chunk = json!({
                        "id": id,
                        "object": "text_completion",
                        "created": created,
                        "model": model,
                        "choices": [{"index": 0, "text": text, "finish_reason": null, "logprobs": null}]
                    });
                    yield Ok(Event::default().data(chunk.to_string()));
                }
                Ok(GenerationEvent::Finished { reason, .. }) => {
                    finish = Some(reason);
                }
                Err(error) => {
                    let chunk = json!({"error": {"message": error, "type": "server_error"}});
                    yield Ok(Event::default().data(chunk.to_string()));
                    break;
                }
            }
        }
        let final_chunk = json!({
            "id": id,
            "object": "text_completion",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "text": "", "finish_reason": api_finish(finish.unwrap_or(FinishReason::Stop)), "logprobs": null}]
        });
        yield Ok(Event::default().data(final_chunk.to_string()));
        yield Ok(Event::default().data("[DONE]"));
    };
    Sse::new(stream).keep_alive(KeepAlive::default())
}

fn usage(result: &GenerationResult) -> Value {
    json!({
        "prompt_tokens": result.prompt_token_count,
        "completion_tokens": result.completion_token_count,
        "total_tokens": result.prompt_token_count + result.completion_token_count,
    })
}

fn runtime_details(result: &GenerationResult) -> Value {
    json!({
        "backend": result.backend,
        "simulation": result.simulation,
        "finish_reason": result.finish_reason,
        "ttft_ms": result.ttft_ms,
        "prefill_ms": result.prefill_ms,
        "decode_ms": result.decode_ms,
        "total_ms": result.total_ms,
        "backend_cache": result.backend_cache,
        "scheduler": result.scheduler,
    })
}

fn api_finish(reason: FinishReason) -> &'static str {
    match reason {
        FinishReason::Stop | FinishReason::Cancelled => "stop",
        FinishReason::Length => "length",
    }
}

fn with_backend_header(mut response: Response, simulation: bool, backend: &str) -> Response {
    if simulation {
        response
            .headers_mut()
            .insert("x-rustmoe-simulation", header::HeaderValue::from_static("true"));
    }
    if let Ok(value) = header::HeaderValue::from_str(backend) {
        response.headers_mut().insert("x-rustmoe-backend", value);
    }
    response
}

fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[derive(Debug)]
pub struct ApiError {
    status: StatusCode,
    kind: &'static str,
    message: String,
}

impl ApiError {
    fn bad_request(kind: &'static str, message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            kind,
            message: message.into(),
        }
    }

    fn internal(kind: &'static str, message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            kind,
            message: message.into(),
        }
    }
}

impl From<InferenceError> for ApiError {
    fn from(value: InferenceError) -> Self {
        let status = match &value {
            InferenceError::ContextLength { .. }
            | InferenceError::EmptyPrompt
            | InferenceError::NoTokens
            | InferenceError::Sampling(_) => StatusCode::BAD_REQUEST,
            _ => StatusCode::INTERNAL_SERVER_ERROR,
        };
        Self {
            status,
            kind: "inference_error",
            message: value.to_string(),
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (
            self.status,
            Json(json!({
                "error": {
                    "message": self.message,
                    "type": if self.status.is_client_error() { "invalid_request_error" } else { "server_error" },
                    "code": self.kind,
                }
            })),
        )
            .into_response()
    }
}

pub async fn serve(runtime: AppState, addr: std::net::SocketAddr, agent: AgentConfig) -> anyhow::Result<()> {
    let router = build_router_with_agent(runtime.clone(), agent);
    let listener = tokio::net::TcpListener::bind(addr).await?;
    let actual = listener.local_addr()?;
    tracing::info!(%actual, "RustMoE HTTP server listening");
    crate::progress::print_ready(actual, &runtime.info);
    axum::serve(listener, router)
        .with_graceful_shutdown(shutdown_signal())
        .await?;
    Ok(())
}

async fn shutdown_signal() {
    let _ = tokio::signal::ctrl_c().await;
    tracing::info!("shutdown signal received");
}
