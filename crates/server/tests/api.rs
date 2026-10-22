use axum::{
    body::Body,
    http::{header, Request, StatusCode},
};
use http_body_util::BodyExt;
use rustmoe_model::ModelDescriptor;
use rustmoe_server::{
    api::build_router,
    runtime::{Runtime, RuntimeOptions},
};
use serde_json::{json, Value};
use std::{path::PathBuf, sync::Arc};
use tower::ServiceExt;

fn fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/tiny-ds41/tiny-ds41-00001-of-00001.gguf")
}

fn runtime(api_key: Option<&str>) -> Arc<Runtime> {
    let mut rt = Runtime::load(
        &fixture(),
        RuntimeOptions {
            context: 64,
            gpu_cache_bytes: Some(128 * 1024),
            host_cache_bytes: Some(256 * 1024),
            prefill_chunk: 16,
            seed: 7,
            ..RuntimeOptions::default()
        },
    )
    .expect("load fixture runtime");
    rt = rt.with_api_key(api_key.map(str::to_string));
    Arc::new(rt)
}

async fn body_json(response: axum::response::Response) -> Value {
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    serde_json::from_slice(&bytes).unwrap()
}

#[test]
fn fixture_catalog_validates() {
    let model = ModelDescriptor::load(fixture()).expect("load fixture");
    let report = model.validate();
    assert!(
        report.is_ok(),
        "unexpected validation errors: {:?}",
        report.errors
    );
}

#[tokio::test]
async fn health_reports_simulation() {
    let app = build_router(runtime(None));
    let response = app
        .oneshot(Request::builder().uri("/health").body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let body = body_json(response).await;
    assert_eq!(body["status"], "ok");
    assert_eq!(body["simulation"], true);
}

#[tokio::test]
async fn chat_completion_is_openai_shaped() {
    let app = build_router(runtime(None));
    let request = json!({
        "model": "RustMoE Tiny DS41 Fixture",
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 4,
        "temperature": 0.0,
        "seed": 7
    });
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(request.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(response.headers().get("x-rustmoe-simulation").unwrap(), "true");
    let body = body_json(response).await;
    assert_eq!(body["object"], "chat.completion");
    assert_eq!(body["choices"][0]["finish_reason"], "length");
    assert_eq!(body["usage"]["completion_tokens"], 4);
    assert_eq!(body["rustmoe"]["simulation"], true);
}

#[tokio::test]
async fn chat_accepts_empty_assistant_content() {
    let app = build_router(runtime(None));
    let request = json!({
        "model": "RustMoE Tiny DS41 Fixture",
        "messages": [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": null},
            {"role": "user", "content": "Continue"}
        ],
        "max_tokens": 1
    });
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(request.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn chat_stream_terminates_with_done() {
    let app = build_router(runtime(None));
    let request = json!({
        "model": "RustMoE Tiny DS41 Fixture",
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 2,
        "stream": true,
        "temperature": 0.0
    });
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(request.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    let text = String::from_utf8_lossy(&bytes);
    assert!(text.contains("chat.completion.chunk"));
    assert!(text.contains("\"finish_reason\":\"length\""));
    assert!(text.contains("data: [DONE]"));
}

#[tokio::test]
async fn api_key_and_model_errors_are_typed() {
    let app = build_router(runtime(Some("secret")));
    let request = json!({
        "model": "wrong",
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 1
    });
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(request.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header(header::CONTENT_TYPE, "application/json")
                .header(header::AUTHORIZATION, "Bearer secret")
                .body(Body::from(request.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::NOT_FOUND);
    let body = body_json(response).await;
    assert_eq!(body["error"]["code"], "model_not_found");
}

#[tokio::test]
async fn status_page_is_html() {
    let app = build_router(runtime(None));
    let response = app
        .oneshot(Request::builder().uri("/status").body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let content_type = response
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    assert!(content_type.contains("text/html"));
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    let text = String::from_utf8_lossy(&bytes);
    assert!(text.contains("RustMoE"));
    assert!(text.contains("/v1/runtime"));
}

#[tokio::test]
async fn chat_page_is_html() {
    let app = build_router(runtime(None));
    let response = app
        .oneshot(Request::builder().uri("/chat").body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    let text = String::from_utf8_lossy(&bytes);
    assert!(text.contains("RustMoE Chat"));
    assert!(text.contains("/v1/chat/completions"));
}

#[tokio::test]
async fn agent_page_is_html() {
    let app = build_router(runtime(None));
    let response = app
        .oneshot(Request::builder().uri("/agent").body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    let text = String::from_utf8_lossy(&bytes);
    assert!(text.contains("RustMoE Dev Agent"));
    assert!(text.contains("/v1/agent/run"));
}

#[tokio::test]
async fn telemetry_reports_system_and_activity() {
    let app = build_router(runtime(None));
    let response = app
        .oneshot(
            Request::builder()
                .uri("/v1/telemetry")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let body = body_json(response).await;
    assert!(body["system"]["cpu"].is_object());
    assert!(body["system"]["memory"].is_object());
    assert_eq!(body["activity"]["phase"], "idle");
}

#[tokio::test]
async fn system_endpoint_reports_hardware() {
    let app = build_router(runtime(None));
    let response = app
        .oneshot(Request::builder().uri("/v1/system").body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let body = body_json(response).await;
    assert!(body["hardware"].is_object());
    assert!(body["hardware"]["gpus"].is_array());
}

#[tokio::test]
async fn metrics_are_prometheus_text() {
    let app = build_router(runtime(None));
    let response = app
        .oneshot(Request::builder().uri("/metrics").body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    let text = String::from_utf8_lossy(&bytes);
    assert!(text.contains("rustmoe_cache_gpu_hits_total"));
    assert!(text.contains("# TYPE rustmoe_ttft_ms histogram"));
}
