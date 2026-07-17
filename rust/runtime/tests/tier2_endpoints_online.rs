// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Real-HTTP tests for Tier-2 endpoint families and non-JSON lifecycles.

use std::collections::BTreeMap;
use std::net::SocketAddr;
use std::rc::Rc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use aiperf_runtime::clock::{Clock, RealClock};
use aiperf_runtime::dataset::{
    ComposeConfig, DatasetSource, LoadConfig, LoaderRegistry, TiktokenTokenizer,
};
use aiperf_runtime::endpoints::{
    EndpointConfig, EndpointId, EndpointRegistry, EndpointType, PreparedEndpoint,
    PreparedEndpointTable, RawEndpointConfig,
};
use aiperf_runtime::metrics_core::MetricTag;
use aiperf_runtime::multiturn::{
    ConversationSource, NativeDatasetConversationSource, PreparedEndpointReference,
    PreparedTurnEndpointResolver, ResolvedPreparedEndpoint,
};
use aiperf_runtime::rng::RngRoot;
use aiperf_runtime::scheduled::ScheduledRunReport;
use aiperf_runtime::transport::core::ErrorKind;
use aiperf_runtime::transport::http::config::ClientConfig;
use aiperf_runtime::transport::http::models::RequestConfig;
use aiperf_runtime::transport::http::transport::http_transport::HttpTransport;
use aiperf_runtime::transport::http::transport::polling::{
    JsonVideoPollingProtocol, PollingOptions, submit_and_poll,
};
use axum::body::Bytes;
use axum::extract::{OriginalUri, State};
use axum::http::{HeaderMap, StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use loadgen_core::collector::ReplayTerminalStatus;
use serde_json::{Value, json};

mod common;

const PNG: &[u8] = &[
    0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0x00, 0x00, 0x00, 0x0d, 0x49, 0x48, 0x44, 0x52,
    0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x08, 0x04, 0x00, 0x00, 0x00, 0xb5, 0x1c, 0x0c,
    0x02, 0x00, 0x00, 0x00, 0x0b, 0x49, 0x44, 0x41, 0x54, 0x78, 0xda, 0x63, 0x64, 0xf8, 0x0f, 0x00,
    0x01, 0x05, 0x01, 0x01, 0x27, 0x18, 0xe3, 0x66, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4e, 0x44,
    0xae, 0x42, 0x60, 0x82,
];
const PNG_DATA_URL: &str = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADElEQVR42mP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC";

#[derive(Clone, Debug)]
struct CapturedRequest {
    path: String,
    content_type: Option<String>,
    body: Bytes,
}

#[derive(Clone, Default)]
struct CaptureState(Arc<Mutex<Vec<CapturedRequest>>>);

impl CaptureState {
    fn record(&self, path: &str, headers: &HeaderMap, body: Bytes) {
        self.0.lock().unwrap().push(CapturedRequest {
            path: path.to_string(),
            content_type: headers
                .get(header::CONTENT_TYPE)
                .and_then(|value| value.to_str().ok())
                .map(str::to_string),
            body,
        });
    }

    fn by_path(&self) -> BTreeMap<String, CapturedRequest> {
        self.0
            .lock()
            .unwrap()
            .iter()
            .cloned()
            .map(|request| (request.path.clone(), request))
            .collect()
    }
}

async fn spawn(app: Router) -> (SocketAddr, tokio::task::JoinHandle<()>) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let task = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    (address, task)
}

fn normalize_endpoint_name(name: &str) -> String {
    name.trim().to_ascii_lowercase().replace(['-', '/'], "_")
}

/// Builtin dialects prepared for per-row `endpoint` overrides.
const OVERRIDE_DIALECTS: &[&str] = &[
    "chat",
    "nim_embeddings",
    "embeddings",
    "chat_embeddings",
    "nim_rankings",
    "cohere_rankings",
    "hf_tei_rankings",
    "huggingface_generate",
    "image_generation",
    "image_edit",
    "image_retrieval",
    "video_generation",
    "solido_rag",
];

/// Test-local multi-endpoint prepared resolver: resolves a per-turn endpoint
/// name override to a prepared dialect, or the authored default when absent.
struct MultiPreparedResolver {
    table: Rc<PreparedEndpointTable>,
    by_name: BTreeMap<String, PreparedEndpointReference>,
    default: PreparedEndpointReference,
}

impl std::fmt::Debug for MultiPreparedResolver {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("MultiPreparedResolver")
            .field("dialects", &self.by_name.keys().collect::<Vec<_>>())
            .finish()
    }
}

impl PreparedTurnEndpointResolver for MultiPreparedResolver {
    fn resolve(&self, name: Option<&str>) -> anyhow::Result<ResolvedPreparedEndpoint<'_>> {
        let reference = match name {
            None => self.default.clone(),
            Some(name) => self
                .by_name
                .get(&normalize_endpoint_name(name))
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("endpoint override {name:?} was not prepared"))?,
        };
        let endpoint = self.table.get(reference.key)?;
        Ok(ResolvedPreparedEndpoint {
            reference,
            endpoint,
        })
    }
}

fn prepare(id: &EndpointId, config: RawEndpointConfig) -> Box<dyn PreparedEndpoint> {
    EndpointRegistry::builtin()
        .unwrap()
        .prepare(id, config)
        .unwrap()
}

/// Build one prepared table holding the authored default endpoint plus every
/// override dialect, and a resolver over it. The same `Rc<table>` must be given
/// to the dispatching sink so dense keys resolve to identical endpoints.
fn dialect_table(
    endpoint_config: EndpointConfig,
) -> (
    Rc<PreparedEndpointTable>,
    Rc<dyn PreparedTurnEndpointResolver>,
) {
    let default_id = EndpointId::new(endpoint_config.endpoint_type.canonical_id()).unwrap();
    // Per-run polling, download, and response settings apply to every selected
    // dialect.
    let base = RawEndpointConfig::from(endpoint_config);
    let mut table = PreparedEndpointTable::new();
    let mut by_name = BTreeMap::new();
    let default_key = table.push(prepare(&default_id, base.clone())).unwrap();
    let default = PreparedEndpointReference {
        key: default_key,
        endpoint_id: default_id.clone(),
    };
    by_name.insert(
        normalize_endpoint_name(default_id.as_str()),
        default.clone(),
    );
    for name in OVERRIDE_DIALECTS {
        let normalized = normalize_endpoint_name(name);
        if normalized == normalize_endpoint_name(default_id.as_str()) {
            continue;
        }
        let id = EndpointId::new(name).unwrap();
        // Some dialects reject unrelated base fields; fall back to defaults.
        let prepared = EndpointRegistry::builtin()
            .unwrap()
            .prepare(&id, base.clone())
            .or_else(|_| {
                EndpointRegistry::builtin()
                    .unwrap()
                    .prepare(&id, RawEndpointConfig::default())
            })
            .unwrap();
        let key = table.push(prepared).unwrap();
        by_name.insert(
            normalized,
            PreparedEndpointReference {
                key,
                endpoint_id: id,
            },
        );
    }
    let table = Rc::new(table);
    let resolver: Rc<dyn PreparedTurnEndpointResolver> = Rc::new(MultiPreparedResolver {
        table: table.clone(),
        by_name,
        default,
    });
    (table, resolver)
}

async fn single_turn_source(
    rows: Value,
    endpoint_config: EndpointConfig,
) -> (Box<dyn ConversationSource>, Rc<PreparedEndpointTable>) {
    let dataset = LoaderRegistry::with_builtin_formats()
        .unwrap()
        .build_dataset(
            Some("single_turn"),
            &LoadConfig::new(DatasetSource::Inline(rows)),
            &ComposeConfig::new("fixture-model", RngRoot::new(Some(7))),
            &TiktokenTokenizer::builtin(),
        )
        .await
        .unwrap();
    let (table, resolver) = dialect_table(endpoint_config);
    let source = Box::new(
        NativeDatasetConversationSource::sequential_with_prepared_resolver(
            dataset,
            "fixture-model",
            16,
            resolver,
        )
        .unwrap(),
    );
    (source, table)
}

async fn raw_source(
    body: Bytes,
    endpoint_config: EndpointConfig,
) -> (Box<dyn ConversationSource>, Rc<PreparedEndpointTable>) {
    let dataset = LoaderRegistry::with_builtin_formats()
        .unwrap()
        .build_dataset(
            Some("raw_payload"),
            &LoadConfig::new(DatasetSource::Bytes(body)),
            &ComposeConfig::new("fixture-model", RngRoot::new(Some(8))),
            &TiktokenTokenizer::builtin(),
        )
        .await
        .unwrap();
    let (table, resolver) = dialect_table(endpoint_config);
    let source = Box::new(
        NativeDatasetConversationSource::sequential_with_prepared_resolver(
            dataset,
            "fixture-model",
            16,
            resolver,
        )
        .unwrap(),
    );
    (source, table)
}

async fn run(
    address: SocketAddr,
    source: Box<dyn ConversationSource>,
    table: Rc<PreparedEndpointTable>,
) -> ScheduledRunReport {
    tokio::task::LocalSet::new()
        .run_until(common::run_single_turn_dataset_online(
            format!("http://{address}"),
            "fixture-model".into(),
            source,
            1,
            false,
            Vec::new(),
            table,
        ))
        .await
        .unwrap()
}

fn assert_completed(report: &ScheduledRunReport, expected: usize) {
    assert_eq!(report.turns.len(), expected);
    let incomplete = report
        .turns
        .iter()
        .filter(|turn| {
            turn.terminal_status != Some(ReplayTerminalStatus::Completed)
                || turn.terminal_offset_ns.is_none()
        })
        .map(|turn| (&turn.conversation_id, turn.terminal_status))
        .collect::<Vec<_>>();
    assert!(incomplete.is_empty(), "incomplete turns: {incomplete:?}");
}

async fn decoded_endpoint(
    State(captured): State<CaptureState>,
    OriginalUri(uri): OriginalUri,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    captured.record(uri.path(), &headers, body);
    match uri.path() {
        "/v1/embeddings" => Json(json!({
            "data":[{"object":"embedding","embedding":[0.1,0.2]}],
            "usage":{"prompt_tokens":2,"total_tokens":2}
        }))
        .into_response(),
        "/v1/ranking" => Json(json!({"rankings":[{"index":0,"logit":0.9}]})).into_response(),
        "/v2/rerank" => {
            Json(json!({"results":[{"index":0,"relevance_score":0.8}]})).into_response()
        }
        "/rerank" => Json(json!([{"index":0,"score":0.7}])).into_response(),
        "/generate_stream" => (
            [(header::CONTENT_TYPE, "text/event-stream")],
            concat!(
                "data: {\"token\":{\"text\":\"hello\"}}\n\n",
                "data: {\"token\":{\"text\":\" world\"},\"generated_text\":\"hello world\"}\n\n"
            ),
        )
            .into_response(),
        "/v1/images/generations" => {
            Json(json!({"data":[{"b64_json":"AA=="}],"size":"1x1"})).into_response()
        }
        "/rag/api/prompt" => {
            Json(json!({"content":"grounded answer","sources":[{"id":"doc-1"}]})).into_response()
        }
        _ => StatusCode::NOT_FOUND.into_response(),
    }
}

#[tokio::test(flavor = "current_thread")]
async fn all_decoded_tier2_dialects_reach_their_real_paths_and_complete() {
    let captured = CaptureState::default();
    let app = Router::new()
        .route("/v1/embeddings", post(decoded_endpoint))
        .route("/v1/ranking", post(decoded_endpoint))
        .route("/v2/rerank", post(decoded_endpoint))
        .route("/rerank", post(decoded_endpoint))
        .route("/generate_stream", post(decoded_endpoint))
        .route("/v1/images/generations", post(decoded_endpoint))
        .route("/rag/api/prompt", post(decoded_endpoint))
        .with_state(captured.clone());
    let (address, server) = spawn(app).await;
    let (source, table) = single_turn_source(
        json!([
            {
                "session_id":"nim-embeddings",
                "endpoint":"nim_embeddings",
                "texts":["caption"],
                "images":[PNG_DATA_URL],
                "streaming":false
            },
            {
                "session_id":"nim-rankings",
                "endpoint":"nim_rankings",
                "texts":[
                    {"name":"query","contents":["query"]},
                    {"name":"passages","contents":["first","second"]}
                ],
                "streaming":false
            },
            {
                "session_id":"cohere-rankings",
                "endpoint":"cohere_rankings",
                "texts":[
                    {"name":"query","contents":["query"]},
                    {"name":"passages","contents":["first","second"]}
                ],
                "streaming":false
            },
            {
                "session_id":"tei-rankings",
                "endpoint":"hf_tei_rankings",
                "texts":[
                    {"name":"query","contents":["query"]},
                    {"name":"passages","contents":["first","second"]}
                ],
                "streaming":false
            },
            {
                "session_id":"tgi",
                "endpoint":"huggingface_generate",
                "text":"say hello",
                "output_length":2,
                "streaming":true
            },
            {
                "session_id":"image-generation",
                "endpoint":"image_generation",
                "text":"draw a fox",
                "streaming":false
            },
            {
                "session_id":"solido",
                "endpoint":"solido_rag",
                "texts":["what is grounded?"],
                "streaming":false
            }
        ]),
        EndpointConfig {
            streaming: true,
            use_server_token_count: true,
            ..EndpointConfig::default()
        },
    )
    .await;
    let report = run(address, source, table).await;
    server.abort();
    assert_completed(&report, 7);

    let requests = captured.by_path();
    assert_eq!(requests.len(), 7);
    assert_eq!(
        serde_json::from_slice::<Value>(&requests["/v1/embeddings"].body).unwrap()["input"][0],
        format!("caption {PNG_DATA_URL}")
    );
    assert_eq!(
        serde_json::from_slice::<Value>(&requests["/v1/ranking"].body).unwrap()["query"]["text"],
        "query"
    );
    assert_eq!(
        serde_json::from_slice::<Value>(&requests["/v2/rerank"].body).unwrap()["documents"],
        json!(["first", "second"])
    );
    assert_eq!(
        serde_json::from_slice::<Value>(&requests["/rerank"].body).unwrap()["texts"],
        json!(["first", "second"])
    );
    assert_eq!(
        serde_json::from_slice::<Value>(&requests["/generate_stream"].body).unwrap()["parameters"]
            ["max_new_tokens"],
        2
    );
    assert_eq!(
        serde_json::from_slice::<Value>(&requests["/v1/images/generations"].body).unwrap()["prompt"],
        "draw a fox"
    );
    assert_eq!(
        serde_json::from_slice::<Value>(&requests["/rag/api/prompt"].body).unwrap()["query"],
        json!(["what is grounded?"])
    );
}

async fn flexible_endpoint(
    State(captured): State<CaptureState>,
    OriginalUri(uri): OriginalUri,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    captured.record(uri.path(), &headers, body);
    Json(json!({"result":{"text":format!("{} answer", uri.path())}})).into_response()
}

#[tokio::test(flavor = "current_thread")]
async fn raw_and_template_run_through_generic_paths_and_jmespath_parsing() {
    let captured = CaptureState::default();
    let app = Router::new()
        .route("/raw", post(flexible_endpoint))
        .route("/template", post(flexible_endpoint))
        .with_state(captured.clone());
    let (address, server) = spawn(app).await;

    let (raw, raw_table) = raw_source(
        Bytes::from_static(
            br#"{"messages":[{"role":"user","content":"raw body"}],"stream":false}"#,
        ),
        EndpointConfig {
            endpoint_type: EndpointType::Raw,
            path: Some("/raw".into()),
            response_field: Some("result.text".into()),
            streaming: false,
            ..EndpointConfig::default()
        },
    )
    .await;
    assert_completed(&run(address, raw, raw_table).await, 1);

    let (template, template_table) = single_turn_source(
        json!([{
            "session_id":"template",
            "endpoint":"template",
            "text":"templated body",
            "streaming":false
        }]),
        EndpointConfig {
            endpoint_type: EndpointType::Template,
            path: Some("/template".into()),
            template: Some(r#"{"query":{{ text|tojson }},"model":{{ model|tojson }}}"#.into()),
            response_field: Some("result.text".into()),
            streaming: false,
            ..EndpointConfig::default()
        },
    )
    .await;
    assert_completed(&run(address, template, template_table).await, 1);
    server.abort();

    let requests = captured.by_path();
    assert_eq!(
        requests["/raw"].body,
        br#"{"messages":[{"role":"user","content":"raw body"}],"stream":false}"#[..]
    );
    assert_eq!(
        serde_json::from_slice::<Value>(&requests["/template"].body).unwrap(),
        json!({"query":"templated body","model":"fixture-model"})
    );
}

#[derive(Clone, Default)]
struct MediaState {
    captured: CaptureState,
    asset_hits: Arc<AtomicUsize>,
}

async fn media_asset(State(state): State<MediaState>) -> Response {
    state.asset_hits.fetch_add(1, Ordering::SeqCst);
    ([(header::CONTENT_TYPE, "image/png")], PNG).into_response()
}

async fn media_inference(
    State(state): State<MediaState>,
    OriginalUri(uri): OriginalUri,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    state.captured.record(uri.path(), &headers, body);
    match uri.path() {
        "/v1/images/edits" => Json(json!({"data":[{"b64_json":"AA=="}]})).into_response(),
        "/v1/infer" => Json(json!({"data":[{"index":0,"score":1.0}]})).into_response(),
        _ => StatusCode::NOT_FOUND.into_response(),
    }
}

#[tokio::test(flavor = "current_thread")]
async fn image_edit_is_multipart_and_retrieval_downloads_deduplicates_and_inlines_media() {
    let state = MediaState::default();
    let app = Router::new()
        .route("/asset.png", get(media_asset))
        .route("/v1/images/edits", post(media_inference))
        .route("/v1/infer", post(media_inference))
        .with_state(state.clone());
    let (address, server) = spawn(app).await;
    let asset_url = format!("http://{address}/asset.png");
    let (source, table) = single_turn_source(
        json!([
            {
                "session_id":"image-edit",
                "endpoint":"image_edit",
                "text":"remove the background",
                "image":PNG_DATA_URL,
                "streaming":false
            },
            {
                "session_id":"image-retrieval",
                "endpoint":"image_retrieval",
                "images":[asset_url, asset_url],
                "streaming":false
            }
        ]),
        EndpointConfig::default(),
    )
    .await;
    let report = run(address, source, table).await;
    server.abort();
    assert_completed(&report, 2);
    assert_eq!(state.asset_hits.load(Ordering::SeqCst), 1);
    assert_eq!(
        report.native_metrics.finite_value(MetricTag::NumImages),
        Some(2.0)
    );

    let requests = state.captured.by_path();
    let edit = &requests["/v1/images/edits"];
    assert!(
        edit.content_type
            .as_deref()
            .is_some_and(|value| value.starts_with("multipart/form-data; boundary=aiperf-"))
    );
    assert!(
        edit.body
            .windows(b"filename=\"image.png\"".len())
            .any(|window| window == b"filename=\"image.png\"")
    );
    assert!(
        edit.body
            .windows(8)
            .any(|window| window == b"\x89PNG\r\n\x1a\n")
    );

    let retrieval: Value = serde_json::from_slice(&requests["/v1/infer"].body).unwrap();
    let urls = retrieval["input"]
        .as_array()
        .unwrap()
        .iter()
        .map(|item| item["url"].as_str().unwrap())
        .collect::<Vec<_>>();
    assert_eq!(urls.len(), 2);
    assert_eq!(urls[0], urls[1]);
    assert!(urls[0].starts_with("data:image/png;base64,"));
}

#[derive(Clone, Default)]
struct VideoState {
    captured: CaptureState,
    polls: Arc<AtomicUsize>,
    downloads: Arc<AtomicUsize>,
}

async fn submit_video(
    State(state): State<VideoState>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    state.captured.record("/v1/videos", &headers, body);
    (
        StatusCode::CREATED,
        Json(json!({"id":"video-1","status":"queued"})),
    )
        .into_response()
}

async fn poll_video(State(state): State<VideoState>) -> Response {
    let poll = state.polls.fetch_add(1, Ordering::SeqCst);
    if poll == 0 {
        Json(json!({"id":"video-1","status":"in_progress","progress":50})).into_response()
    } else {
        Json(json!({
            "id":"video-1",
            "status":"completed",
            "progress":100,
            "url":"/video-content",
            "inference_time_s":0.25,
            "peak_memory_mb":512.0
        }))
        .into_response()
    }
}

async fn video_content(State(state): State<VideoState>) -> Response {
    state.downloads.fetch_add(1, Ordering::SeqCst);
    ([(header::CONTENT_TYPE, "video/mp4")], b"video-bytes").into_response()
}

#[tokio::test(flavor = "current_thread")]
async fn video_submission_polls_with_the_shared_clock_and_downloads_completed_content() {
    let state = VideoState::default();
    let app = Router::new()
        .route("/v1/videos", post(submit_video))
        .route("/v1/videos/{id}", get(poll_video))
        .route("/video-content", get(video_content))
        .with_state(state.clone());
    let (address, server) = spawn(app).await;
    let (source, table) = single_turn_source(
        json!([{
            "session_id":"video",
            "endpoint":"video_generation",
            "text":"a short orbit",
            "streaming":false
        }]),
        EndpointConfig {
            timeout_seconds: 2.0,
            polling_interval_seconds: 0.001,
            download_video_content: true,
            ..EndpointConfig::default()
        },
    )
    .await;
    let report = run(address, source, table).await;
    server.abort();
    assert_completed(&report, 1);
    assert_eq!(state.polls.load(Ordering::SeqCst), 2);
    assert_eq!(state.downloads.load(Ordering::SeqCst), 1);
    assert_eq!(
        report
            .native_metrics
            .finite_value(MetricTag::VideoInferenceTime),
        Some(250.0)
    );
    assert_eq!(
        report
            .native_metrics
            .finite_value(MetricTag::VideoPeakMemory),
        Some(512.0)
    );
    let request = &state.captured.by_path()["/v1/videos"];
    assert!(
        request
            .content_type
            .as_deref()
            .is_some_and(|value| value.starts_with("multipart/form-data; boundary=aiperf-"))
    );
    assert!(
        request
            .body
            .windows(b"a short orbit".len())
            .any(|window| window == b"a short orbit")
    );
}

async fn pending_video_submit() -> Response {
    Json(json!({"id":"video-pending","status":"queued"})).into_response()
}

async fn pending_video_poll(State(polls): State<Arc<AtomicUsize>>) -> Response {
    polls.fetch_add(1, Ordering::SeqCst);
    Json(json!({"id":"video-pending","status":"in_progress"})).into_response()
}

#[tokio::test(flavor = "current_thread")]
async fn video_polling_honors_the_original_post_send_cancellation_deadline() {
    let polls = Arc::new(AtomicUsize::new(0));
    let app = Router::new()
        .route("/v1/videos", post(pending_video_submit))
        .route("/v1/videos/{id}", get(pending_video_poll))
        .with_state(polls.clone());
    let (address, server) = spawn(app).await;
    let clock: Rc<dyn Clock> = RealClock::new();
    let transport = HttpTransport::new(clock.clone(), ClientConfig::default());
    let mut config = RequestConfig::new(format!("http://{address}/v1/videos"));
    config.cancel_after_ns = Some(100_000_000);
    let lifecycle = Box::pin(submit_and_poll(
        &transport,
        clock,
        &config,
        Bytes::from_static(br#"{"prompt":"orbit"}"#),
        PollingOptions {
            timeout_ns: 2_000_000_000,
            interval_ns: 1_000_000_000,
            download_content: false,
        },
        &JsonVideoPollingProtocol,
    ));
    let result = tokio::task::LocalSet::new().run_until(lifecycle).await;
    server.abort();
    assert_eq!(
        result.record.error.as_ref().map(|error| error.kind),
        Some(ErrorKind::Cancelled)
    );
    assert_eq!(result.record.status, Some(499));
    assert!(result.record.cancellation_ns.is_some());
    assert!(polls.load(Ordering::SeqCst) >= 1);
}
