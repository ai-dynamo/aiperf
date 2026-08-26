// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Axum handlers for every mock-server endpoint.

use std::convert::Infallible;
use std::sync::Arc;
use std::time::{Duration, Instant};

use aiperf_runtime::rng::RustRandomGenerator;
use aiperf_runtime::transport::core::EventStreamMessage;
use axum::Json;
use axum::body::Body;
use axum::extract::State;
use axum::http::{StatusCode, header};
use axum::response::{IntoResponse, Response};
use base64::Engine;
use blake2::{Blake2s256, Digest};
use bytes::Bytes;
use futures::stream::Stream;
use http_body_util::{BodyExt, Empty};
use hyper::{Request, Uri};
use serde::Serialize;
use serde_json::{Value, json};

use crate::latency::{LatencySimulator, wait_for_processing};
use crate::metrics::LLMLatencyInfo;
use crate::models::{
    ChatCompletionRequest, CohereRerankRequest, CompletionRequest, EmbeddingRequest,
    HFTEIRerankRequest, ImageGenerationRequest, ImageResponseFormat, ImageRetrievalRequest,
    Message, MessagesRequest, RankingRequest, ResponsesRequest, SolidoRAGRequest,
    TGIGenerateRequest, Usage, VllmGenerateRequest,
};
use crate::state::{AppState, ContentFetchClient};
use crate::tokens::{
    GenRequest, TokenizedText, tokenize_request, tokenize_request_with_fixed_output_tokens,
};

pub type AppResult<T> = Result<T, AppError>;

#[derive(Debug)]
pub struct AppError {
    pub status: StatusCode,
    pub message: String,
    /// `Retry-After` header value (seconds) to emit, for `429`/`503` backoff.
    /// `None` leaves the header off.
    pub retry_after: Option<u64>,
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let body = json!({ "detail": self.message });
        let mut response = (self.status, Json(body)).into_response();
        if let Some(secs) = self.retry_after
            && let Ok(value) = header::HeaderValue::from_str(&secs.to_string())
        {
            response.headers_mut().insert(header::RETRY_AFTER, value);
        }
        response
    }
}

fn internal_error<E: std::fmt::Display>(e: E) -> AppError {
    AppError {
        status: StatusCode::INTERNAL_SERVER_ERROR,
        message: format!("{e}"),
        retry_after: None,
    }
}

fn now_secs() -> i64 {
    chrono::Utc::now().timestamp()
}

/// A process-unique request-id number, cheaper than a v4 UUID on the hot path.
///
/// The response `id` only needs to be unique, not random. Each thread claims a
/// distinct high-order ordinal once, then increments a thread-local counter — no
/// per-request RNG and no cross-thread atomic contention. Unique for up to 2^40
/// requests across up to 2^24 threads, far beyond any run.
fn next_request_seq() -> u64 {
    use std::cell::Cell;
    use std::sync::atomic::{AtomicU64, Ordering};
    static THREAD_ORDINAL: AtomicU64 = AtomicU64::new(0);
    thread_local! {
        static SEQ: Cell<u64> = Cell::new(THREAD_ORDINAL.fetch_add(1, Ordering::Relaxed) << 40);
    }
    SEQ.with(|c| {
        let v = c.get();
        c.set(v.wrapping_add(1));
        v
    })
}

fn make_request_id(prefix: &str) -> String {
    format!("{prefix}-{:016x}", next_request_seq())
}

fn make_anthropic_message_id() -> String {
    format!("msg_{:016x}", next_request_seq())
}

fn maybe_inject_error(state: &AppState) -> Option<AppError> {
    let code = state.inject_error_status()?;
    let status = StatusCode::from_u16(code).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
    // A rate-limited (429) or overloaded (503) backend hands the client a
    // Retry-After backoff hint; other injected codes carry none.
    let retry_after = matches!(code, 429 | 503).then_some(state.config.error_retry_after);
    Some(AppError {
        status,
        message: format!("Simulated error (status {code})"),
        retry_after,
    })
}

/// Shared context for a tokenized LLM request.
///
/// HTTP and gRPC entry points share tokenization, usage, latency, and
/// prefix-cache preparation through this context.
pub(crate) struct RequestCtx {
    pub(crate) request_id: String,
    pub(crate) model: String,
    pub(crate) tokenized: TokenizedText,
    pub(crate) usage: Usage,
    pub(crate) latency_sim: LatencySimulator,
    pub(crate) start: Instant,
    /// Emits one `{"object": null}` SSE frame before `[DONE]`.
    pub(crate) null_object_chunk: bool,
    /// Function call selected by the seeded `--tool-call-rate` draw.
    pub(crate) tool_call: Option<ToolCallSpec>,
    /// Emits the canonical speculative-decoding fixture on this chat response.
    pub(crate) spec_decode_acceptance: bool,
    /// Emits cumulative usage on each streamed reasoning/content frame.
    pub(crate) continuous_usage_stats: bool,
    /// Number of output tokens carried by the first streamed content frame.
    pub(crate) first_chunk_tokens: usize,
}

/// Deterministic OpenAI-compatible function call. `arguments` is a
/// JSON-encoded *string* (not an object), and `id`/`type` identify the call.
pub(crate) struct ToolCallSpec {
    pub(crate) id: String,
    pub(crate) name: String,
    pub(crate) arguments: String,
}

impl ToolCallSpec {
    /// Returns the call and its deterministic `toolUsePromptTokenCount`.
    fn from_config(cfg: &crate::config::MockServerConfig) -> (Self, usize) {
        let name = cfg.tool_call_name.clone();
        let arguments = cfg.tool_call_arguments.clone();
        let tool_use_tokens = crate::tokens::tokenize(&format!("{name}{arguments}")).len();
        let spec = Self {
            // The runner keys streamed calls by `index`; the opaque ID need not
            // be stable.
            id: format!("call_{}", uuid::Uuid::new_v4()),
            name,
            arguments,
        };
        (spec, tool_use_tokens)
    }

    /// Splits arguments on a character boundary for streamed delta merging.
    fn split_arguments(&self) -> (&str, &str) {
        let mid = self.arguments.len() / 2;
        let boundary = (mid..=self.arguments.len())
            .find(|i| self.arguments.is_char_boundary(*i))
            .unwrap_or(self.arguments.len());
        self.arguments.split_at(boundary)
    }
}

impl RequestCtx {
    pub(crate) fn build(
        request_id_prefix: &str,
        req_gen: &GenRequest<'_>,
        _endpoint: &str,
        start: Instant,
        state: &AppState,
    ) -> Self {
        let mut tokenized =
            tokenize_request_with_fixed_output_tokens(req_gen, state.config.fixed_output_tokens);
        // Apply the accuracy decision here so every endpoint and streaming mode
        // serializes the same deterministic answer.
        let mut null_object_chunk = false;
        if let Some(ds) = &state.accuracy {
            if let Some(entry) = ds.lookup(&tokenized.text) {
                let decision = ds.decide(entry);
                state.accuracy_live.record(&decision, entry.task.as_deref());
                tokenized.tokens = crate::tokens::tokenize(&decision.content);
                tokenized.reasoning_content_tokens = decision
                    .reasoning_content
                    .as_deref()
                    .map(crate::tokens::tokenize)
                    .unwrap_or_default();
                tokenized.reasoning_tokens = tokenized.reasoning_content_tokens.len();
                tokenized.finish_reason = "stop";
                null_object_chunk = decision.null_object_chunk;
            } else {
                state.accuracy_live.record_unmatched();
            }
        }
        let mut usage = tokenized.usage();
        let model = req_gen.model().to_string();
        let request_id = make_request_id(request_id_prefix);
        // Cache hits are always reported, but only reduce TTFT when
        // `--prefix-cache-latency-aware` is enabled.
        let cached_tokens = match &state.prefix_cache {
            Some(pc) => pc.cached_tokens(&tokenized.text, usage.prompt_tokens, req_gen.priority()),
            None => 0,
        };
        // Always expose cache-read counts, including zero when caching is off.
        usage.prompt_tokens_details = Some(crate::models::PromptTokensDetails {
            cached_tokens,
            audio_tokens: None,
        });
        // Optional `--usage-*` fields stay absent unless explicitly configured.
        if state.config.usage_fields_enabled() {
            apply_usage_fields(&mut usage, &state.config);
        }
        let latency_cached = if state.config.prefix_cache_latency_aware {
            cached_tokens
        } else {
            0
        };
        // Include this request in concurrency-dependent latency.
        let active_inflight = (state.recorder.inflight_count().max(0) as usize) + 1;
        let latency_sim = LatencySimulator::new(
            state.clock_anchor,
            &state.config,
            usage.prompt_tokens,
            tokenized.count(),
            active_inflight,
            state.scheduler.clone(),
            &request_id,
            latency_cached,
        );
        let (continuous_usage_stats, first_chunk_tokens) = match req_gen {
            GenRequest::Chat(request) => (
                request.continuous_usage_stats(),
                request.first_chunk_tokens(),
            ),
            GenRequest::Completion(request) => (
                request.continuous_usage_stats(),
                request.first_chunk_tokens(),
            ),
            _ => (false, 1),
        };
        Self {
            request_id,
            latency_sim,
            model,
            tokenized,
            usage,
            start,
            null_object_chunk,
            tool_call: None,
            spec_decode_acceptance: state.config.spec_decode_acceptance,
            continuous_usage_stats,
            first_chunk_tokens,
        }
    }
}

fn spec_decode_acceptance_fixture() -> &'static Value {
    static FIXTURE: std::sync::OnceLock<Value> = std::sync::OnceLock::new();
    FIXTURE.get_or_init(|| {
        json!({
            "mean_acceptance_length": 3.25,
            "draft_acceptance_rate": 0.5625,
            "acceptance_histogram": {"0": 1, "1": 1, "2": 2, "3": 3, "4": 1},
            "num_accepted_draft_tokens": 18,
            "num_draft_tokens": 32,
            "num_spec_steps": 8,
            "num_spec_tokens": 4,
            "per_step_accepted": [2, 3, 1, 4, 2, 0, 3, 3],
            "per_step_drafted": [4, 4, 4, 4, 4, 4, 4, 4]
        })
    })
}

pub async fn root_info(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let cfg = serde_json::to_value(&state.config).unwrap_or(Value::Null);
    Json(json!({
        "message": "AIPerf Mock Server",
        "version": "2.0.0",
        "config": cfg,
    }))
}

pub async fn health(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let cfg = serde_json::to_value(&state.config).unwrap_or(Value::Null);
    Json(json!({
        "status": "healthy",
        "config": cfg,
    }))
}

/// Models advertised when `--models` is empty.
const DEFAULT_MODELS: &[&str] = &[
    "gpt-4",
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-3.5-turbo",
    "openai/gpt-oss-120b",
    "text-embedding-3-small",
    "text-embedding-ada-002",
    "Qwen/Qwen3-0.6B",
    "meta-llama/Llama-3-8B-Instruct",
    "black-forest-labs/FLUX.1-dev",
];

fn build_model_list(state: &AppState) -> Vec<String> {
    use std::collections::BTreeSet;
    let mut set: BTreeSet<String> = BTreeSet::new();
    if state.config.models.is_empty() {
        for m in DEFAULT_MODELS {
            set.insert((*m).to_string());
        }
    } else {
        for m in &state.config.models {
            let trimmed = m.trim();
            if !trimmed.is_empty() {
                set.insert(trimmed.to_string());
            }
        }
    }
    for m in state.recorder.seen_models() {
        set.insert(m);
    }
    set.into_iter().collect()
}

fn model_object(id: &str, created: i64) -> Value {
    json!({
        "id": id,
        "object": "model",
        "created": created,
        "owned_by": "aiperf-mock",
    })
}

fn server_start_epoch(state: &AppState) -> i64 {
    state
        .start_wallclock
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

pub async fn list_models(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let created = server_start_epoch(&state);
    let data: Vec<Value> = build_model_list(&state)
        .into_iter()
        .map(|id| model_object(&id, created))
        .collect();
    Json(json!({
        "object": "list",
        "data": data,
    }))
}

pub async fn get_model(
    State(state): State<Arc<AppState>>,
    axum::extract::Path(id): axum::extract::Path<String>,
) -> AppResult<Response> {
    let models = build_model_list(&state);
    if !models.iter().any(|m| m == &id) {
        return Err(AppError {
            status: StatusCode::NOT_FOUND,
            message: format!("Model '{id}' not found"),
            retry_after: None,
        });
    }
    let created = server_start_epoch(&state);
    Ok(Json(model_object(&id, created)).into_response())
}

pub async fn chat_completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatCompletionRequest>,
) -> AppResult<Response> {
    if let Some(e) = maybe_inject_error(&state) {
        return Err(e);
    }
    let endpoint = "/v1/chat/completions";
    let start = Instant::now();
    state.recorder.init_model_config(&req.model);
    let req_gen = GenRequest::Chat(&req);
    let mut ctx = RequestCtx::build("chatcmpl", &req_gen, endpoint, start, &state);
    if state.inject_tool_call() {
        let (spec, tool_use_tokens) = ToolCallSpec::from_config(&state.config);
        ctx.usage.tool_use_prompt_token_count = Some(tool_use_tokens);
        ctx.tool_call = Some(spec);
    }

    fetch_content_urls(&state, endpoint, &collect_content_urls(&req.messages)).await;

    if req.stream {
        state.recorder.record_streaming_start(endpoint, &ctx.model);
        let include_usage = req.include_usage();
        // Decide the mid-stream failure on the request thread (not inside the
        // async stream) so the seeded `mock.errors` draw order is deterministic.
        let midstream_error = state.inject_midstream();
        let body = chat_stream(
            state.clone(),
            ctx,
            endpoint.to_string(),
            include_usage,
            midstream_error,
        );
        Ok(sse_response(body))
    } else {
        // Resolve all labeled metric handles once, then drive the whole
        // request lifecycle through cached child handles (no per-metric label
        // hash/lookup on the hot path).
        let labeled = state.recorder.labeled(endpoint, &ctx.model);
        state.recorder.admit_fast(&labeled);
        let (prefill, _decode) = ctx.latency_sim.wait_for_tokens(ctx.tokenized.count()).await;
        let latency = start.elapsed();
        let info = LLMLatencyInfo {
            e2e: latency,
            prefill,
            decode: latency.saturating_sub(prefill),
        };
        let json_body = write_chat_response(&ctx);
        let resp_bytes = json_body.len() as u64;

        state.recorder.complete_fast(
            &labeled,
            latency.as_secs_f64(),
            &ctx.usage,
            &info,
            ctx.tokenized.text.len() as u64,
            resp_bytes,
        );

        Ok(Response::builder()
            .status(StatusCode::OK)
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(json_body))
            .map_err(internal_error)?)
    }
}

/// Authoritative `--fast` streaming metric sequence shared by the chat and text
/// completion renderers.
///
/// Both renderers must emit the *identical* ordered metric sequence for a
/// zero-latency streamed response; keeping it in one place removes the drift
/// hazard of two hand-copied sequences. The order is: streaming/request/inflight
/// start, then (for a non-empty response) a zero TTFT plus `total_tokens - 1`
/// zero ITLs and the streamed-token count, then — after `render_body` produces
/// the SSE bytes — the terminal `llm_success` / inflight-end / request-end. The
/// rendered body is returned to the caller.
fn record_streaming_fast(
    state: &AppState,
    endpoint: &str,
    ctx: &RequestCtx,
    labeled: &crate::metrics::LabeledMetrics,
    total_tokens: usize,
    start: Instant,
    render_body: impl FnOnce() -> Bytes,
) -> Bytes {
    state.recorder.record_streaming_start(endpoint, &ctx.model);
    state.recorder.record_request_start(endpoint, &ctx.model);
    state.recorder.record_llm_inflight_start(&ctx.model);
    if total_tokens > 0 {
        state
            .recorder
            .record_zero_ttft_and_itls(labeled, total_tokens - 1);
        state
            .recorder
            .record_streamed_tokens_fast(labeled, total_tokens as u64);
    }
    let body = render_body();
    let latency = start.elapsed();
    let info = LLMLatencyInfo {
        e2e: latency,
        prefill: std::time::Duration::ZERO,
        decode: latency,
    };
    state.recorder.record_llm_success(
        endpoint,
        &ctx.model,
        latency.as_secs_f64(),
        &ctx.usage,
        &info,
    );
    state.recorder.record_llm_inflight_end(&ctx.model);
    state.recorder.record_request_end(endpoint);
    body
}

/// Chat-completion generation for the hand-rolled `--blocking`/`--uring`
/// engines, without the tokio/axum request machinery. Handles both streaming
/// (SSE, one pre-rendered body) and non-streaming, under `--fast` semantics
/// (zero simulated latency, so no `wait_for_tokens` await). Returns the response
/// content-type and body. Records the same metrics the axum `--fast` path does.
///
/// Not handled here (these engines target raw throughput): error injection,
/// mid-stream failures, and null-object chunk injection.
pub(crate) fn render_chat_completion_fast(
    state: &AppState,
    req: &ChatCompletionRequest,
) -> (&'static str, Bytes) {
    let endpoint = "/v1/chat/completions";
    let start = Instant::now();
    state.recorder.init_model_config(&req.model);
    let req_gen = GenRequest::Chat(req);
    let mut ctx = RequestCtx::build("chatcmpl", &req_gen, endpoint, start, state);
    if state.inject_tool_call() {
        let (spec, tool_use_tokens) = ToolCallSpec::from_config(&state.config);
        ctx.usage.tool_use_prompt_token_count = Some(tool_use_tokens);
        ctx.tool_call = Some(spec);
    }
    let labeled = state.recorder.labeled(endpoint, &ctx.model);

    if req.stream {
        // Mirror the axum fast streaming path's metric sequence exactly, via
        // the shared authoritative `record_streaming_fast`.
        let include_usage = req.include_usage();
        let total_tokens =
            ctx.tokenized.reasoning_content_tokens.len() + ctx.tokenized.tokens.len();
        let body =
            record_streaming_fast(state, endpoint, &ctx, &labeled, total_tokens, start, || {
                render_chat_fast_body(&ctx, include_usage)
            });
        // `body` is already a single-owner `Bytes`; return it directly instead of
        // copying it out into an owned `Vec` (the caller only borrows it as
        // `&[u8]`). The SSE body scales with the output token count, so the copy
        // was per-request waste on the throughput engines.
        return ("text/event-stream", body);
    }

    state.recorder.admit_fast(&labeled);
    let latency = start.elapsed();
    let info = LLMLatencyInfo {
        e2e: latency,
        prefill: std::time::Duration::ZERO,
        decode: latency,
    };
    let json_body = write_chat_response(&ctx);
    let resp_bytes = json_body.len() as u64;
    state.recorder.complete_fast(
        &labeled,
        latency.as_secs_f64(),
        &ctx.usage,
        &info,
        ctx.tokenized.text.len() as u64,
        resp_bytes,
    );
    ("application/json", Bytes::from(json_body))
}

/// Text-completion (`/v1/completions`) generation for the hand-rolled engines,
/// streaming or not, under `--fast` semantics. Mirrors the axum path's metrics.
pub(crate) fn render_text_completion_fast(
    state: &AppState,
    req: &crate::models::CompletionRequest,
) -> (&'static str, Bytes) {
    let endpoint = "/v1/completions";
    let start = Instant::now();
    state.recorder.init_model_config(&req.model);
    let req_gen = GenRequest::Completion(req);
    let ctx = RequestCtx::build("cmpl", &req_gen, endpoint, start, state);
    let labeled = state.recorder.labeled(endpoint, &ctx.model);
    let latency_info = || {
        let latency = start.elapsed();
        (
            latency,
            LLMLatencyInfo {
                e2e: latency,
                prefill: std::time::Duration::ZERO,
                decode: latency,
            },
        )
    };

    if req.stream {
        let include_usage = req.include_usage();
        let total_tokens = ctx.tokenized.tokens.len();
        let body =
            record_streaming_fast(state, endpoint, &ctx, &labeled, total_tokens, start, || {
                render_text_fast_body(&ctx, include_usage)
            });
        // Return the single-owner SSE `Bytes` directly; see the chat path above.
        return ("text/event-stream", body);
    }

    state.recorder.record_request_start(endpoint, &ctx.model);
    state.recorder.record_llm_inflight_start(&ctx.model);
    let json_body =
        serde_json::to_vec(&build_completion_response(&ctx)).unwrap_or_else(|_| b"{}".to_vec());
    let (latency, info) = latency_info();
    state.recorder.record_request_bytes(
        endpoint,
        ctx.tokenized.text.len() as u64,
        json_body.len() as u64,
    );
    state.recorder.record_llm_success(
        endpoint,
        &ctx.model,
        latency.as_secs_f64(),
        &ctx.usage,
        &info,
    );
    state.recorder.record_llm_inflight_end(&ctx.model);
    state.recorder.record_request_end(endpoint);
    ("application/json", Bytes::from(json_body))
}

/// Embeddings (`/v1/embeddings`) generation for the hand-rolled engines. Always
/// non-streaming JSON; mirrors the axum path's metrics (`--fast` skips the
/// simulated per-input processing latency).
pub(crate) fn render_embeddings_fast(
    state: &AppState,
    req: &crate::models::EmbeddingRequest,
) -> Vec<u8> {
    let endpoint = "/v1/embeddings";
    let start = Instant::now();
    let req_gen = GenRequest::Embedding(req);
    let ctx = RequestCtx::build("emb", &req_gen, endpoint, start, state);
    let inputs = req.inputs();
    state.recorder.record_request_start(endpoint, &req.model);
    let data: Vec<Value> = inputs
        .iter()
        .enumerate()
        .map(|(i, text)| {
            json!({
                "object": "embedding",
                "index": i,
                "embedding": generate_embedding(text, 768),
            })
        })
        .collect();
    let body = json!({
        "object": "list",
        "model": req.model,
        "data": data,
        "usage": ctx.usage,
    });
    state.recorder.record_embedding_success(
        endpoint,
        &req.model,
        ctx.usage.prompt_tokens,
        inputs.len(),
        start.elapsed().as_secs_f64(),
    );
    state.recorder.record_request_end(endpoint);
    serde_json::to_vec(&body).unwrap_or_else(|_| b"{}".to_vec())
}

/// Typed mirror of the `json!()`-built response this replaced. Serializing a
/// typed struct directly skips building an intermediate `serde_json::Value`
/// tree (a `String`/`Map`/`Vec` allocation per field) before serializing that
/// tree to bytes — profiling showed `Value` construction/drop as a real
/// allocator-traffic cost in the hot path. `#[serde(skip_serializing_if)]`
/// on the optional fields reproduces the old behavior exactly: an absent
/// `reasoning_content`/`tool_calls` means the key doesn't appear in the JSON
/// at all, not `null`.
#[derive(Serialize)]
struct ChatResponseFunctionCall<'a> {
    name: &'a str,
    arguments: &'a str,
}

#[derive(Serialize)]
struct ChatResponseToolCall<'a> {
    id: &'a str,
    #[serde(rename = "type")]
    call_type: &'a str,
    function: ChatResponseFunctionCall<'a>,
}

#[derive(Serialize)]
struct ChatResponseMessage<'a> {
    role: &'a str,
    content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<[ChatResponseToolCall<'a>; 1]>,
}

#[derive(Serialize)]
struct ChatResponseChoice<'a> {
    index: u32,
    finish_reason: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    speculative_decoding_stats: Option<&'a Value>,
    message: ChatResponseMessage<'a>,
}

#[derive(Serialize)]
struct ChatResponse<'a> {
    id: &'a str,
    object: &'a str,
    created: i64,
    model: &'a str,
    choices: [ChatResponseChoice<'a>; 1],
    usage: &'a Usage,
}

/// Serializes as a single JSON string equal to the concatenation of `tokens`,
/// streamed straight into the output via `collect_str` — no intermediate
/// concatenated `String`. serde_json's `collect_str` applies the identical
/// string escaping it would to the joined `&str`, so output is byte-identical.
struct TokenJoin<'a>(&'a [String]);

impl std::fmt::Display for TokenJoin<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for t in self.0 {
            f.write_str(t)?;
        }
        Ok(())
    }
}

impl Serialize for TokenJoin<'_> {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.collect_str(self)
    }
}

/// Hand-assembled equivalent of `serde_json::to_vec(&build_chat_response(ctx))`.
///
/// Structural keys and literal values (`object`, `role`, braces, the
/// `assistant`/`function` constants) are written as raw bytes — serde would
/// escape-scan every one of them on the hot path. Only the variable values
/// (id, model, finish_reason, content, tool-call fields, usage) go through
/// `serde_json::to_writer`, so their escaping and number formatting stay
/// byte-identical to the derived-`Serialize` path. Verified by
/// `write_chat_response_matches_serde`.
fn write_chat_response(ctx: &RequestCtx) -> Vec<u8> {
    write_chat_response_bytes(
        &ctx.request_id,
        &ctx.model,
        now_secs(),
        ctx.tokenized.finish_reason,
        ctx.tool_call.as_ref(),
        ctx.spec_decode_acceptance,
        &ctx.tokenized.tokens,
        &ctx.tokenized.reasoning_content_tokens,
        &ctx.usage,
    )
}

#[allow(clippy::too_many_arguments)]
fn write_chat_response_bytes(
    id: &str,
    model: &str,
    created: i64,
    base_finish_reason: &str,
    tool_call: Option<&ToolCallSpec>,
    spec_decode_acceptance: bool,
    content_tokens: &[String],
    reasoning_tokens: &[String],
    usage: &Usage,
) -> Vec<u8> {
    use std::io::Write as _;
    let finish_reason = if tool_call.is_some() {
        "tool_calls"
    } else {
        base_finish_reason
    };
    let mut buf = Vec::with_capacity(256);
    buf.extend_from_slice(br#"{"id":"#);
    serde_json::to_writer(&mut buf, id).unwrap();
    buf.extend_from_slice(br#","object":"chat.completion","created":"#);
    let _ = write!(&mut buf, "{created}");
    buf.extend_from_slice(br#","model":"#);
    serde_json::to_writer(&mut buf, model).unwrap();
    buf.extend_from_slice(br#","choices":[{"index":0,"finish_reason":"#);
    serde_json::to_writer(&mut buf, finish_reason).unwrap();
    if spec_decode_acceptance {
        buf.extend_from_slice(br#","speculative_decoding_stats":"#);
        serde_json::to_writer(&mut buf, spec_decode_acceptance_fixture()).unwrap();
    }
    buf.extend_from_slice(br#","message":{"role":"assistant","content":"#);
    serde_json::to_writer(&mut buf, &TokenJoin(content_tokens)).unwrap();
    if !reasoning_tokens.is_empty() {
        buf.extend_from_slice(br#","reasoning_content":"#);
        serde_json::to_writer(&mut buf, &TokenJoin(reasoning_tokens)).unwrap();
    }
    if let Some(tc) = tool_call {
        buf.extend_from_slice(br#","tool_calls":[{"id":"#);
        serde_json::to_writer(&mut buf, &tc.id).unwrap();
        buf.extend_from_slice(br#","type":"function","function":{"name":"#);
        serde_json::to_writer(&mut buf, &tc.name).unwrap();
        buf.extend_from_slice(br#","arguments":"#);
        serde_json::to_writer(&mut buf, &tc.arguments).unwrap();
        buf.extend_from_slice(br#"}}]"#);
    }
    buf.extend_from_slice(br#"}}],"usage":"#);
    serde_json::to_writer(&mut buf, usage).unwrap();
    buf.push(b'}');
    buf
}

#[cfg(test)]
mod chat_response_serialize_tests {
    use super::*;
    use crate::models::{PromptTokensDetails, Usage};

    fn usage_sample() -> Usage {
        Usage {
            prompt_tokens: 2,
            completion_tokens: 3,
            total_tokens: 5,
            completion_tokens_details: None,
            prompt_tokens_details: Some(PromptTokensDetails {
                cached_tokens: 1,
                audio_tokens: None,
            }),
            cache_creation_input_tokens: None,
            prompt_cache_miss_tokens: None,
            tool_use_prompt_token_count: None,
            prompt_audio_seconds: None,
            cache_read_input_tokens: None,
        }
    }

    // Reference: the exact bytes the derived-Serialize path produced.
    fn reference(
        id: &str,
        model: &str,
        created: i64,
        base_fr: &str,
        tool_call: Option<&ToolCallSpec>,
        spec_decode_acceptance: bool,
        content_tokens: &[String],
        reasoning_tokens: &[String],
        usage: &Usage,
    ) -> Vec<u8> {
        let (finish_reason, tool_calls) = if let Some(tc) = tool_call {
            (
                "tool_calls",
                Some([ChatResponseToolCall {
                    id: &tc.id,
                    call_type: "function",
                    function: ChatResponseFunctionCall {
                        name: &tc.name,
                        arguments: &tc.arguments,
                    },
                }]),
            )
        } else {
            (base_fr, None)
        };
        let reasoning_content = if reasoning_tokens.is_empty() {
            None
        } else {
            Some(reasoning_tokens.concat())
        };
        let resp = ChatResponse {
            id,
            object: "chat.completion",
            created,
            model,
            choices: [ChatResponseChoice {
                index: 0,
                finish_reason,
                speculative_decoding_stats: spec_decode_acceptance
                    .then_some(spec_decode_acceptance_fixture()),
                message: ChatResponseMessage {
                    role: "assistant",
                    content: content_tokens.concat(),
                    reasoning_content,
                    tool_calls,
                },
            }],
            usage,
        };
        serde_json::to_vec(&resp).unwrap()
    }

    #[test]
    fn write_chat_response_matches_serde() {
        let usage = usage_sample();
        // Include characters that force JSON escaping (quote, backslash, newline)
        // in every user-influenced field, so escaping parity is exercised.
        let content = vec![" he\"llo".to_string(), " wo\\rld\n".to_string()];
        let reasoning = vec!["think ".to_string(), "hard".to_string()];
        let cases: Vec<(&str, Option<ToolCallSpec>, &[String])> =
            vec![("plain", None, &[]), ("reasoning", None, &reasoning)];
        for (label, tc, rsn) in cases {
            let got = write_chat_response_bytes(
                "chatcmpl-x\"1",
                "mo\"del",
                1234567890,
                "stop",
                tc.as_ref(),
                false,
                &content,
                rsn,
                &usage,
            );
            let exp = reference(
                "chatcmpl-x\"1",
                "mo\"del",
                1234567890,
                "stop",
                tc.as_ref(),
                false,
                &content,
                rsn,
                &usage,
            );
            assert_eq!(
                got,
                exp,
                "{label}:\n got={}\n exp={}",
                String::from_utf8_lossy(&got),
                String::from_utf8_lossy(&exp)
            );
        }
        // Tool-call case (overrides finish_reason, appends tool_calls).
        let tc = ToolCallSpec {
            id: "call_1".to_string(),
            name: "get\"weather".to_string(),
            arguments: "{\"city\":\"x\"}".to_string(),
        };
        let got = write_chat_response_bytes(
            "id1",
            "m",
            42,
            "stop",
            Some(&tc),
            false,
            &content,
            &[],
            &usage,
        );
        let exp = reference(
            "id1",
            "m",
            42,
            "stop",
            Some(&tc),
            false,
            &content,
            &[],
            &usage,
        );
        assert_eq!(
            got,
            exp,
            "tool_call:\n got={}\n exp={}",
            String::from_utf8_lossy(&got),
            String::from_utf8_lossy(&exp)
        );
    }
}

fn build_chat_response(ctx: &RequestCtx) -> ChatResponse<'_> {
    let (finish_reason, tool_calls) = if let Some(tc) = &ctx.tool_call {
        (
            "tool_calls",
            Some([ChatResponseToolCall {
                id: &tc.id,
                call_type: "function",
                function: ChatResponseFunctionCall {
                    name: &tc.name,
                    arguments: &tc.arguments,
                },
            }]),
        )
    } else {
        (ctx.tokenized.finish_reason, None)
    };
    ChatResponse {
        id: &ctx.request_id,
        object: "chat.completion",
        created: now_secs(),
        model: &ctx.model,
        choices: [ChatResponseChoice {
            index: 0,
            finish_reason,
            speculative_decoding_stats: ctx
                .spec_decode_acceptance
                .then_some(spec_decode_acceptance_fixture()),
            message: ChatResponseMessage {
                role: "assistant",
                content: ctx.tokenized.content(),
                reasoning_content: ctx.tokenized.reasoning_content(),
                tool_calls,
            },
        }],
        usage: &ctx.usage,
    }
}

#[cfg(test)]
mod spec_decode_acceptance_tests {
    use super::*;
    use crate::config::MockServerConfig;

    fn expected_stats() -> Value {
        json!({
            "mean_acceptance_length": 3.25,
            "draft_acceptance_rate": 0.5625,
            "acceptance_histogram": {"0": 1, "1": 1, "2": 2, "3": 3, "4": 1},
            "num_accepted_draft_tokens": 18,
            "num_draft_tokens": 32,
            "num_spec_steps": 8,
            "num_spec_tokens": 4,
            "per_step_accepted": [2, 3, 1, 4, 2, 0, 3, 3],
            "per_step_drafted": [4, 4, 4, 4, 4, 4, 4, 4]
        })
    }

    fn request(stream: bool) -> ChatCompletionRequest {
        serde_json::from_value(json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": stream,
            "stream_options": {"include_usage": true},
            "max_tokens": 2
        }))
        .unwrap()
    }

    fn enabled_state() -> Arc<AppState> {
        AppState::build(MockServerConfig {
            fast: true,
            no_tokenizer: true,
            fixed_output_tokens: Some(2),
            spec_decode_acceptance: true,
            ..Default::default()
        })
    }

    fn sse_values(body: &[u8]) -> Vec<Value> {
        String::from_utf8_lossy(body)
            .split("\n\n")
            .filter_map(|frame| frame.strip_prefix("data: "))
            .filter(|data| *data != "[DONE]")
            .map(|data| serde_json::from_str(data).unwrap())
            .collect()
    }

    async fn collect_sse_values(
        stream: impl Stream<Item = Result<Bytes, Infallible>>,
    ) -> Vec<Value> {
        let chunks = futures::StreamExt::collect::<Vec<_>>(stream).await;
        let mut body = Vec::new();
        for chunk in chunks {
            body.extend_from_slice(&chunk.unwrap());
        }
        sse_values(&body)
    }

    fn timed_state(fixed_output_tokens: Option<usize>) -> Arc<AppState> {
        AppState::build(MockServerConfig {
            no_tokenizer: true,
            fixed_output_tokens,
            ttft: 0.01,
            itl: 0.01,
            ..Default::default()
        })
    }

    #[tokio::test]
    async fn nonstream_choice_carries_opt_in_canonical_stats() {
        let state = enabled_state();
        let (content_type, body) = render_chat_completion_fast(&state, &request(false));
        assert_eq!(content_type, "application/json");
        let response: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(
            response["choices"][0]["speculative_decoding_stats"],
            expected_stats()
        );
    }

    #[tokio::test]
    async fn stream_emits_finish_only_stats_before_usage_only_chunk() {
        let state = enabled_state();
        let (content_type, body) = render_chat_completion_fast(&state, &request(true));
        assert_eq!(content_type, "text/event-stream");
        let frames = sse_values(&body);

        let (finish_index, finish) = frames
            .iter()
            .enumerate()
            .find(|(_, frame)| frame["choices"][0]["finish_reason"] == "length")
            .expect("finish-reason frame");
        assert_eq!(finish["choices"][0]["delta"], json!({}));
        assert_eq!(
            finish["choices"][0]["speculative_decoding_stats"],
            expected_stats()
        );
        assert!(finish.get("usage").is_none());

        let usage_index = frames
            .iter()
            .position(|frame| frame["choices"] == json!([]) && frame.get("usage").is_some())
            .expect("usage-only frame");
        assert!(finish_index < usage_index);
    }

    #[tokio::test]
    async fn stream_bundles_first_content_and_emits_cumulative_usage() {
        let state = AppState::build(MockServerConfig {
            fast: true,
            no_tokenizer: true,
            fixed_output_tokens: Some(6),
            ..Default::default()
        });
        let request: ChatCompletionRequest = serde_json::from_value(json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": true,
            "stream_options": {
                "include_usage": true,
                "continuous_usage_stats": true
            },
            "mock_first_chunk_tokens": 3,
            "max_tokens": 6
        }))
        .unwrap();
        let (content_type, body) = render_chat_completion_fast(&state, &request);
        assert_eq!(content_type, "text/event-stream");
        let frames = sse_values(&body);
        let content_frames = frames
            .iter()
            .filter(|frame| {
                frame["choices"][0]["delta"]["content"]
                    .as_str()
                    .is_some_and(|content| !content.is_empty())
            })
            .collect::<Vec<_>>();

        assert_eq!(content_frames.len(), 4);
        assert_eq!(
            content_frames
                .iter()
                .map(|frame| frame["usage"]["completion_tokens"].as_u64())
                .collect::<Vec<_>>(),
            vec![Some(3), Some(4), Some(5), Some(6)]
        );
        assert_eq!(
            content_frames[0]["choices"][0]["delta"]["role"],
            "assistant"
        );
        assert_eq!(content_frames[3]["choices"][0]["finish_reason"], "length");
        assert!(
            frames
                .iter()
                .any(|frame| frame["choices"] == json!([])
                    && frame["usage"]["completion_tokens"] == 6)
        );
    }

    #[tokio::test]
    async fn stream_keeps_terminal_and_continuous_usage_independent() {
        let state = AppState::build(MockServerConfig {
            fast: true,
            no_tokenizer: true,
            fixed_output_tokens: Some(3),
            ..Default::default()
        });

        for continuous_usage_stats in [None, Some(false)] {
            let stream_options = continuous_usage_stats.map_or_else(
                || json!({"include_usage": true}),
                |value| {
                    json!({
                        "include_usage": true,
                        "continuous_usage_stats": value
                    })
                },
            );
            let request: ChatCompletionRequest = serde_json::from_value(json!({
                "model": "test-model",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": true,
                "stream_options": stream_options,
                "max_tokens": 3
            }))
            .unwrap();
            let (_, body) = render_chat_completion_fast(&state, &request);
            let frames = sse_values(&body);
            let content_frames = frames.iter().filter(|frame| {
                frame["choices"][0]["delta"]["content"]
                    .as_str()
                    .is_some_and(|content| !content.is_empty())
            });
            assert!(
                content_frames
                    .into_iter()
                    .all(|frame| frame.get("usage").is_none())
            );
            assert_eq!(
                frames
                    .iter()
                    .filter(|frame| frame["choices"] == json!([]) && frame.get("usage").is_some())
                    .count(),
                1
            );
        }

        let continuous_without_terminal: ChatCompletionRequest = serde_json::from_value(json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": true,
            "stream_options": {
                "include_usage": false,
                "continuous_usage_stats": true
            },
            "max_tokens": 3
        }))
        .unwrap();
        let (_, body) = render_chat_completion_fast(&state, &continuous_without_terminal);
        let frames = sse_values(&body);
        let content_frames = frames
            .iter()
            .filter(|frame| {
                frame["choices"][0]["delta"]["content"]
                    .as_str()
                    .is_some_and(|content| !content.is_empty())
            })
            .collect::<Vec<_>>();
        assert!(
            content_frames
                .iter()
                .all(|frame| frame.get("usage").is_some())
        );
        assert!(
            frames
                .iter()
                .all(|frame| frame["choices"] != json!([]) || frame.get("usage").is_none())
        );
    }

    #[tokio::test]
    async fn reasoning_stream_emits_cumulative_usage_on_every_generated_chunk() {
        let state = AppState::build(MockServerConfig {
            fast: true,
            no_tokenizer: true,
            ..Default::default()
        });
        let request: ChatCompletionRequest = serde_json::from_value(json!({
            "model": "qwen-test",
            "messages": [{"role": "user", "content": "hello world"}],
            "stream": true,
            "stream_options": {
                "include_usage": true,
                "continuous_usage_stats": true
            },
            "reasoning_effort": "low",
            "max_tokens": 4
        }))
        .unwrap();
        let (_, body) = render_chat_completion_fast(&state, &request);
        let frames = sse_values(&body);
        let reasoning_frames = frames
            .iter()
            .filter(|frame| {
                frame["choices"][0]["delta"]["reasoning_content"]
                    .as_str()
                    .is_some_and(|content| !content.is_empty())
            })
            .collect::<Vec<_>>();

        assert_eq!(reasoning_frames.len(), 4);
        assert_eq!(
            reasoning_frames
                .iter()
                .map(|frame| {
                    (
                        frame["usage"]["prompt_tokens"].as_u64(),
                        frame["usage"]["completion_tokens"].as_u64(),
                        frame["usage"]["total_tokens"].as_u64(),
                    )
                })
                .collect::<Vec<_>>(),
            vec![
                (Some(3), Some(1), Some(4)),
                (Some(3), Some(2), Some(5)),
                (Some(3), Some(3), Some(6)),
                (Some(3), Some(4), Some(7)),
            ]
        );
    }

    #[tokio::test]
    async fn text_stream_bundles_first_content_and_emits_cumulative_usage() {
        let state = AppState::build(MockServerConfig {
            fast: true,
            no_tokenizer: true,
            fixed_output_tokens: Some(6),
            ..Default::default()
        });
        let request: CompletionRequest = serde_json::from_value(json!({
            "model": "test-model",
            "prompt": "hello",
            "stream": true,
            "stream_options": {
                "include_usage": true,
                "continuous_usage_stats": true
            },
            "mock_first_chunk_tokens": 3,
            "max_tokens": 6
        }))
        .unwrap();
        let (content_type, body) = render_text_completion_fast(&state, &request);
        assert_eq!(content_type, "text/event-stream");
        let frames = sse_values(&body);
        let content_frames = frames
            .iter()
            .filter(|frame| {
                frame["choices"][0]["text"]
                    .as_str()
                    .is_some_and(|content| !content.is_empty())
            })
            .collect::<Vec<_>>();

        assert_eq!(content_frames.len(), 4);
        assert_eq!(
            content_frames
                .iter()
                .map(|frame| frame["usage"]["completion_tokens"].as_u64())
                .collect::<Vec<_>>(),
            vec![Some(3), Some(4), Some(5), Some(6)]
        );
        assert_eq!(content_frames[3]["choices"][0]["finish_reason"], "length");
        assert!(
            frames
                .iter()
                .any(|frame| frame["choices"] == json!([])
                    && frame["usage"]["completion_tokens"] == 6)
        );
    }

    #[tokio::test]
    async fn text_stream_without_continuous_usage_keeps_usage_terminal_only() {
        let state = AppState::build(MockServerConfig {
            fast: true,
            no_tokenizer: true,
            fixed_output_tokens: Some(3),
            ..Default::default()
        });

        for continuous_usage_stats in [None, Some(false)] {
            let stream_options = continuous_usage_stats.map_or_else(
                || json!({"include_usage": true}),
                |value| {
                    json!({
                        "include_usage": true,
                        "continuous_usage_stats": value
                    })
                },
            );
            let request: CompletionRequest = serde_json::from_value(json!({
                "model": "test-model",
                "prompt": "hello",
                "stream": true,
                "stream_options": stream_options,
                "max_tokens": 3
            }))
            .unwrap();
            let (_, body) = render_text_completion_fast(&state, &request);
            let frames = sse_values(&body);
            assert!(frames.iter().all(|frame| {
                frame["choices"][0]["text"]
                    .as_str()
                    .is_none_or(|content| content.is_empty() || frame.get("usage").is_none())
            }));
            assert_eq!(
                frames
                    .iter()
                    .filter(|frame| frame["choices"] == json!([]) && frame.get("usage").is_some())
                    .count(),
                1
            );
        }
    }

    #[tokio::test]
    async fn timed_chat_and_text_streams_bundle_first_content_with_cumulative_usage() {
        let state = timed_state(Some(6));
        let chat_request: ChatCompletionRequest = serde_json::from_value(json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": true,
            "stream_options": {
                "include_usage": true,
                "continuous_usage_stats": true
            },
            "mock_first_chunk_tokens": 3,
            "max_tokens": 6
        }))
        .unwrap();
        let chat_gen = GenRequest::Chat(&chat_request);
        let chat_ctx = RequestCtx::build(
            "chatcmpl",
            &chat_gen,
            "/v1/chat/completions",
            Instant::now(),
            &state,
        );
        let chat_frames = collect_sse_values(chat_stream(
            state.clone(),
            chat_ctx,
            "/v1/chat/completions".to_string(),
            true,
            false,
        ))
        .await;
        let chat_usage = chat_frames
            .iter()
            .filter(|frame| {
                frame["choices"][0]["delta"]["content"]
                    .as_str()
                    .is_some_and(|content| !content.is_empty())
            })
            .map(|frame| frame["usage"]["completion_tokens"].as_u64())
            .collect::<Vec<_>>();
        assert_eq!(chat_usage, vec![Some(3), Some(4), Some(5), Some(6)]);

        let text_request: CompletionRequest = serde_json::from_value(json!({
            "model": "test-model",
            "prompt": "hello",
            "stream": true,
            "stream_options": {
                "include_usage": true,
                "continuous_usage_stats": true
            },
            "mock_first_chunk_tokens": 3,
            "max_tokens": 6
        }))
        .unwrap();
        let text_gen = GenRequest::Completion(&text_request);
        let text_ctx =
            RequestCtx::build("cmpl", &text_gen, "/v1/completions", Instant::now(), &state);
        let text_frames = collect_sse_values(text_stream(
            state,
            text_ctx,
            "/v1/completions".to_string(),
            true,
        ))
        .await;
        let text_usage = text_frames
            .iter()
            .filter(|frame| {
                frame["choices"][0]["text"]
                    .as_str()
                    .is_some_and(|content| !content.is_empty())
            })
            .map(|frame| frame["usage"]["completion_tokens"].as_u64())
            .collect::<Vec<_>>();
        assert_eq!(text_usage, vec![Some(3), Some(4), Some(5), Some(6)]);
    }

    #[tokio::test]
    async fn timed_streams_keep_absent_or_false_continuous_usage_terminal_only() {
        for continuous_usage_stats in [None, Some(false)] {
            let stream_options = continuous_usage_stats.map_or_else(
                || json!({"include_usage": true}),
                |value| {
                    json!({
                        "include_usage": true,
                        "continuous_usage_stats": value
                    })
                },
            );
            let state = timed_state(Some(3));
            let chat_request: ChatCompletionRequest = serde_json::from_value(json!({
                "model": "test-model",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": true,
                "stream_options": stream_options,
                "max_tokens": 3
            }))
            .unwrap();
            let chat_gen = GenRequest::Chat(&chat_request);
            let chat_ctx = RequestCtx::build(
                "chatcmpl",
                &chat_gen,
                "/v1/chat/completions",
                Instant::now(),
                &state,
            );
            let chat_frames = collect_sse_values(chat_stream(
                state.clone(),
                chat_ctx,
                "/v1/chat/completions".to_string(),
                true,
                false,
            ))
            .await;
            assert!(chat_frames.iter().all(|frame| {
                frame["choices"][0]["delta"]["content"]
                    .as_str()
                    .is_none_or(|content| content.is_empty() || frame.get("usage").is_none())
            }));
            assert_eq!(
                chat_frames
                    .iter()
                    .filter(|frame| frame["choices"] == json!([]) && frame.get("usage").is_some())
                    .count(),
                1
            );

            let text_request: CompletionRequest = serde_json::from_value(json!({
                "model": "test-model",
                "prompt": "hello",
                "stream": true,
                "stream_options": stream_options,
                "max_tokens": 3
            }))
            .unwrap();
            let text_gen = GenRequest::Completion(&text_request);
            let text_ctx =
                RequestCtx::build("cmpl", &text_gen, "/v1/completions", Instant::now(), &state);
            let text_frames = collect_sse_values(text_stream(
                state,
                text_ctx,
                "/v1/completions".to_string(),
                true,
            ))
            .await;
            assert!(text_frames.iter().all(|frame| {
                frame["choices"][0]["text"]
                    .as_str()
                    .is_none_or(|content| content.is_empty() || frame.get("usage").is_none())
            }));
            assert_eq!(
                text_frames
                    .iter()
                    .filter(|frame| frame["choices"] == json!([]) && frame.get("usage").is_some())
                    .count(),
                1
            );
        }
    }

    #[tokio::test]
    async fn timed_reasoning_stream_emits_cumulative_usage_on_every_chunk() {
        let state = timed_state(None);
        let request: ChatCompletionRequest = serde_json::from_value(json!({
            "model": "qwen-test",
            "messages": [{"role": "user", "content": "hello world"}],
            "stream": true,
            "stream_options": {"continuous_usage_stats": true},
            "reasoning_effort": "low",
            "max_tokens": 4
        }))
        .unwrap();
        let req_gen = GenRequest::Chat(&request);
        let ctx = RequestCtx::build(
            "chatcmpl",
            &req_gen,
            "/v1/chat/completions",
            Instant::now(),
            &state,
        );
        let frames = collect_sse_values(chat_stream(
            state,
            ctx,
            "/v1/chat/completions".to_string(),
            false,
            false,
        ))
        .await;
        let reasoning_usage = frames
            .iter()
            .filter(|frame| {
                frame["choices"][0]["delta"]["reasoning_content"]
                    .as_str()
                    .is_some_and(|content| !content.is_empty())
            })
            .map(|frame| frame["usage"]["completion_tokens"].as_u64())
            .collect::<Vec<_>>();
        assert_eq!(reasoning_usage, vec![Some(1), Some(2), Some(3), Some(4)]);
        assert!(
            frames
                .iter()
                .all(|frame| frame["choices"] != json!([]) || frame.get("usage").is_none())
        );
    }

    #[tokio::test]
    async fn zero_output_tool_call_defers_finish_to_stats_chunk() {
        let state = AppState::build(MockServerConfig {
            fast: true,
            no_tokenizer: true,
            fixed_output_tokens: Some(0),
            tool_call_rate: 1.0,
            spec_decode_acceptance: true,
            ..Default::default()
        });
        let (content_type, body) = render_chat_completion_fast(&state, &request(true));
        assert_eq!(content_type, "text/event-stream");
        let frames = sse_values(&body);

        let tool_frames: Vec<&Value> = frames
            .iter()
            .filter(|frame| frame["choices"][0]["delta"].get("tool_calls").is_some())
            .collect();
        assert_eq!(tool_frames.len(), 2);
        assert!(
            tool_frames
                .iter()
                .all(|frame| frame["choices"][0]["finish_reason"].is_null())
        );

        let (finish_index, finish) = frames
            .iter()
            .enumerate()
            .find(|(_, frame)| frame["choices"][0]["finish_reason"] == "tool_calls")
            .expect("finish-only stats frame");
        assert_eq!(finish["choices"][0]["delta"], json!({}));
        assert_eq!(
            finish["choices"][0]["speculative_decoding_stats"],
            expected_stats()
        );
        let usage_index = frames
            .iter()
            .position(|frame| frame["choices"] == json!([]) && frame.get("usage").is_some())
            .expect("usage-only frame");
        assert!(finish_index < usage_index);
    }
}

pub async fn messages(
    State(state): State<Arc<AppState>>,
    Json(req): Json<MessagesRequest>,
) -> AppResult<Response> {
    if let Some(e) = maybe_inject_error(&state) {
        return Err(e);
    }
    let endpoint = "/v1/messages";
    let start = Instant::now();
    state.recorder.init_model_config(&req.model);
    let req_gen = GenRequest::Messages(&req);
    let mut ctx = RequestCtx::build("msg", &req_gen, endpoint, start, &state);
    ctx.request_id = make_anthropic_message_id();

    if req.stream {
        state.recorder.record_streaming_start(endpoint, &ctx.model);
        Ok(sse_response(messages_stream(
            state,
            ctx,
            endpoint.to_string(),
        )))
    } else {
        state.recorder.record_request_start(endpoint, &ctx.model);
        state.recorder.record_llm_inflight_start(&ctx.model);
        let (prefill, _decode) = ctx.latency_sim.wait_for_tokens(ctx.tokenized.count()).await;
        let latency = start.elapsed();
        let info = LLMLatencyInfo {
            e2e: latency,
            prefill,
            decode: latency.saturating_sub(prefill),
        };
        let body = build_messages_response(&ctx);
        let json_body = serde_json::to_vec(&body).map_err(internal_error)?;
        state.recorder.record_request_bytes(
            endpoint,
            ctx.tokenized.text.len() as u64,
            json_body.len() as u64,
        );
        state.recorder.record_llm_success(
            endpoint,
            &ctx.model,
            latency.as_secs_f64(),
            &ctx.usage,
            &info,
        );
        state.recorder.record_llm_inflight_end(&ctx.model);
        state.recorder.record_request_end(endpoint);

        Ok(Response::builder()
            .status(StatusCode::OK)
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(json_body))
            .map_err(internal_error)?)
    }
}

/// Adds configured `--usage-*` fields, leaving zero-valued fields absent.
fn apply_usage_fields(usage: &mut Usage, cfg: &crate::config::MockServerConfig) {
    if cfg.usage_cache_write_tokens != 0 {
        usage.cache_creation_input_tokens = Some(cfg.usage_cache_write_tokens);
    }
    if cfg.usage_cache_miss_tokens != 0 {
        usage.prompt_cache_miss_tokens = Some(cfg.usage_cache_miss_tokens);
    }
    if cfg.usage_cache_read_tokens != 0 {
        usage.cache_read_input_tokens = Some(cfg.usage_cache_read_tokens);
    }
    if cfg.usage_tool_use_prompt_tokens != 0 {
        usage.tool_use_prompt_token_count = Some(cfg.usage_tool_use_prompt_tokens);
    }
    if cfg.usage_prompt_audio_seconds != 0.0 {
        usage.prompt_audio_seconds = Some(cfg.usage_prompt_audio_seconds);
    }
    if cfg.usage_prompt_audio_tokens != 0 {
        usage
            .prompt_tokens_details
            .get_or_insert_with(Default::default)
            .audio_tokens = Some(cfg.usage_prompt_audio_tokens);
    }
    if cfg.usage_completion_audio_tokens != 0
        || cfg.usage_accepted_prediction_tokens != 0
        || cfg.usage_rejected_prediction_tokens != 0
    {
        let details = usage
            .completion_tokens_details
            .get_or_insert_with(Default::default);
        if cfg.usage_completion_audio_tokens != 0 {
            details.audio_tokens = Some(cfg.usage_completion_audio_tokens);
        }
        if cfg.usage_accepted_prediction_tokens != 0 {
            details.accepted_prediction_tokens = Some(cfg.usage_accepted_prediction_tokens);
        }
        if cfg.usage_rejected_prediction_tokens != 0 {
            details.rejected_prediction_tokens = Some(cfg.usage_rejected_prediction_tokens);
        }
    }
}

/// Anthropic `messages` usage object. Adds the disjoint cache-read/write fields
/// that AIPerf re-totals when the corresponding `--usage-*` knobs are set.
fn anthropic_usage(usage: &Usage) -> Value {
    let mut obj = serde_json::Map::new();
    obj.insert("input_tokens".into(), json!(usage.prompt_tokens));
    obj.insert("output_tokens".into(), json!(usage.completion_tokens));
    if let Some(read) = usage.cache_read_input_tokens {
        obj.insert("cache_read_input_tokens".into(), json!(read));
    }
    if let Some(write) = usage.cache_creation_input_tokens {
        obj.insert("cache_creation_input_tokens".into(), json!(write));
    }
    Value::Object(obj)
}

fn build_messages_response(ctx: &RequestCtx) -> Value {
    json!({
        "id": ctx.request_id,
        "type": "message",
        "role": "assistant",
        "model": ctx.model,
        "content": [{"type": "text", "text": ctx.tokenized.content()}],
        "stop_reason": "end_turn",
        "stop_sequence": Value::Null,
        "usage": anthropic_usage(&ctx.usage),
    })
}

pub async fn text_completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CompletionRequest>,
) -> AppResult<Response> {
    if let Some(e) = maybe_inject_error(&state) {
        return Err(e);
    }
    let endpoint = "/v1/completions";
    let start = Instant::now();
    state.recorder.init_model_config(&req.model);
    let req_gen = GenRequest::Completion(&req);
    let ctx = RequestCtx::build("cmpl", &req_gen, endpoint, start, &state);

    if req.stream {
        state.recorder.record_streaming_start(endpoint, &ctx.model);
        let include_usage = req.include_usage();
        let body = text_stream(state.clone(), ctx, endpoint.to_string(), include_usage);
        Ok(sse_response(body))
    } else {
        state.recorder.record_request_start(endpoint, &ctx.model);
        state.recorder.record_llm_inflight_start(&ctx.model);
        let (prefill, _decode) = ctx.latency_sim.wait_for_tokens(ctx.tokenized.count()).await;
        let latency = start.elapsed();
        let info = LLMLatencyInfo {
            e2e: latency,
            prefill,
            decode: latency.saturating_sub(prefill),
        };
        let body = build_completion_response(&ctx);
        let json_body = serde_json::to_vec(&body).map_err(internal_error)?;
        state.recorder.record_request_bytes(
            endpoint,
            ctx.tokenized.text.len() as u64,
            json_body.len() as u64,
        );
        state.recorder.record_llm_success(
            endpoint,
            &ctx.model,
            latency.as_secs_f64(),
            &ctx.usage,
            &info,
        );
        state.recorder.record_llm_inflight_end(&ctx.model);
        state.recorder.record_request_end(endpoint);

        Ok(Response::builder()
            .status(StatusCode::OK)
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(json_body))
            .map_err(internal_error)?)
    }
}

fn build_completion_response(ctx: &RequestCtx) -> Value {
    json!({
        "id": ctx.request_id,
        "object": "text_completion",
        "created": now_secs(),
        "model": ctx.model,
        "choices": [{
            "index": 0,
            "finish_reason": ctx.tokenized.finish_reason,
            "text": ctx.tokenized.content(),
        }],
        "usage": ctx.usage,
    })
}

/// Token-native usage has no reasoning split.
fn token_native_usage(isl: usize, osl: usize) -> Usage {
    Usage {
        prompt_tokens: isl,
        completion_tokens: osl,
        total_tokens: isl + osl,
        completion_tokens_details: None,
        prompt_tokens_details: None,
        cache_creation_input_tokens: None,
        prompt_cache_miss_tokens: None,
        tool_use_prompt_token_count: None,
        prompt_audio_seconds: None,
        cache_read_input_tokens: None,
    }
}

/// vLLM/Dynamo token-in / token-out Generate. Consumes the request's raw
/// `token_ids` as the prompt (ISL = its length), derives the output length from
/// `sampling_params` with the shared budget logic, and returns integer
/// `choices[].token_ids`. The endpoint accepts only `stream: false`.
pub async fn vllm_generate(
    State(state): State<Arc<AppState>>,
    Json(req): Json<VllmGenerateRequest>,
) -> AppResult<Response> {
    if let Some(e) = maybe_inject_error(&state) {
        return Err(e);
    }
    let endpoint = "/inference/v1/generate";
    let start = Instant::now();
    state.recorder.init_model_config(&req.model);

    let isl = req.token_ids.len();
    let (out_ids, finish_reason) = crate::tokens::generate_output_token_ids(
        &req.token_ids,
        req.sampling_params.max_tokens,
        req.sampling_params.min_tokens,
        req.sampling_params.ignore_eos,
    );
    let osl = out_ids.len();
    let usage = token_native_usage(isl, osl);
    let request_id = req
        .request_id
        .clone()
        .unwrap_or_else(|| make_request_id("gen"));

    let active_inflight = (state.recorder.inflight_count().max(0) as usize) + 1;
    let latency_sim = LatencySimulator::new(
        state.clock_anchor,
        &state.config,
        isl,
        osl,
        active_inflight,
        state.scheduler.clone(),
        &request_id,
        0,
    );

    state.recorder.record_request_start(endpoint, &req.model);
    state.recorder.record_llm_inflight_start(&req.model);
    let (prefill, _decode) = latency_sim.wait_for_tokens(osl).await;
    let latency = start.elapsed();
    let info = LLMLatencyInfo {
        e2e: latency,
        prefill,
        decode: latency.saturating_sub(prefill),
    };

    let body = json!({
        "id": request_id,
        "object": "generate",
        "created": now_secs(),
        "model": req.model,
        "choices": [{
            "index": 0,
            "token_ids": out_ids,
            "finish_reason": finish_reason,
        }],
        "usage": usage,
    });
    let json_body = serde_json::to_vec(&body).map_err(internal_error)?;
    state
        .recorder
        .record_request_bytes(endpoint, isl as u64, json_body.len() as u64);
    state
        .recorder
        .record_llm_success(endpoint, &req.model, latency.as_secs_f64(), &usage, &info);
    state.recorder.record_llm_inflight_end(&req.model);
    state.recorder.record_request_end(endpoint);

    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(json_body))
        .map_err(internal_error)
}

/// Builds a `ChatCompletionRequest` from a SageMaker Runtime invocation body.
///
/// Accepts either an OpenAI-chat-shaped body (`messages` key present) or a
/// SageMaker JumpStart/DJL-shaped body (`inputs` key present, optional
/// `parameters`), detected by key presence. `endpoint_name` (the URL path
/// segment) always determines the target model, overriding any `model` field
/// in the body, matching real SageMaker Runtime semantics where the endpoint
/// name alone selects the deployed container.
fn sagemaker_request_to_chat(
    endpoint_name: &str,
    body: &Value,
    stream: bool,
) -> AppResult<ChatCompletionRequest> {
    if body.get("messages").is_some() {
        // The SageMaker wire body has no `model` field (`endpoint_name` alone
        // selects the target); seed a placeholder so it always deserializes,
        // then override it below regardless.
        let mut patched = body.clone();
        patched["model"] = Value::String(endpoint_name.to_string());
        let mut req: ChatCompletionRequest =
            serde_json::from_value(patched).map_err(|e| AppError {
                status: StatusCode::BAD_REQUEST,
                message: format!("invalid OpenAI-chat-shaped SageMaker request: {e}"),
                retry_after: None,
            })?;
        req.model = endpoint_name.to_string();
        req.stream = stream;
        Ok(req)
    } else if let Some(inputs) = body.get("inputs") {
        let content = match inputs {
            Value::String(s) => s.clone(),
            other => other.to_string(),
        };
        let params = body.get("parameters");
        let max_tokens = params
            .and_then(|p| p.get("max_new_tokens"))
            .and_then(Value::as_u64)
            .map(|v| v as usize);
        let min_tokens = params
            .and_then(|p| p.get("min_new_tokens"))
            .and_then(Value::as_u64)
            .map(|v| v as usize);
        Ok(ChatCompletionRequest {
            model: endpoint_name.to_string(),
            messages: vec![Message {
                role: "user".to_string(),
                content: Value::String(content),
            }],
            stream,
            stream_options: None,
            max_tokens,
            max_completion_tokens: None,
            ignore_eos: false,
            min_tokens,
            reasoning_effort: None,
            priority: None,
            mock_first_chunk_tokens: 1,
        })
    } else {
        Err(AppError {
            status: StatusCode::BAD_REQUEST,
            message: "SageMaker request body must contain either a `messages` (OpenAI-chat) \
                      or `inputs` (JumpStart/DJL) key"
                .to_string(),
            retry_after: None,
        })
    }
}

/// AWS SageMaker Runtime `InvokeEndpoint`: `POST
/// /endpoints/{endpoint_name}/invocations`. Non-streaming; always responds
/// OpenAI chat-completion shaped regardless of which request shape was sent.
pub async fn sagemaker_invoke(
    State(state): State<Arc<AppState>>,
    axum::extract::Path(endpoint_name): axum::extract::Path<String>,
    Json(body): Json<Value>,
) -> AppResult<Response> {
    if let Some(e) = maybe_inject_error(&state) {
        return Err(e);
    }
    let req = sagemaker_request_to_chat(&endpoint_name, &body, false)?;
    let endpoint = "/endpoints/{endpoint_name}/invocations";
    let start = Instant::now();
    state.recorder.init_model_config(&req.model);
    let req_gen = GenRequest::Chat(&req);
    let ctx = RequestCtx::build("chatcmpl", &req_gen, endpoint, start, &state);

    state.recorder.record_request_start(endpoint, &ctx.model);
    state.recorder.record_llm_inflight_start(&ctx.model);
    let (prefill, _decode) = ctx.latency_sim.wait_for_tokens(ctx.tokenized.count()).await;
    let latency = start.elapsed();
    let info = LLMLatencyInfo {
        e2e: latency,
        prefill,
        decode: latency.saturating_sub(prefill),
    };
    let body = build_chat_response(&ctx);
    let json_body = serde_json::to_vec(&body).map_err(internal_error)?;
    let resp_bytes = json_body.len() as u64;

    state
        .recorder
        .record_request_bytes(endpoint, ctx.tokenized.text.len() as u64, resp_bytes);
    state.recorder.record_llm_success(
        endpoint,
        &ctx.model,
        latency.as_secs_f64(),
        &ctx.usage,
        &info,
    );
    state.recorder.record_llm_inflight_end(&ctx.model);
    state.recorder.record_request_end(endpoint);

    Ok(Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(json_body))
        .map_err(internal_error)?)
}

/// Re-frames a `chat_stream` SSE byte stream (`data: <json>\n\n` frames,
/// terminated by `data: [DONE]\n\n`) as AWS eventstream binary frames, one
/// `PayloadPart` message per SSE `data:` line. The `[DONE]` sentinel is
/// dropped: AWS SageMaker eventstream responses have no terminal sentinel,
/// they end at HTTP body EOF.
///
/// The `data: ` prefix (and a trailing `\n`) is preserved in the
/// `PayloadPart.Bytes` payload rather than stripped: real SageMaker
/// containers (HF TGI/vLLM/LMI) emit the raw SSE-formatted line inside the
/// PayloadPart, and clients (boto3-based benchmarkers, AIPerf's SageMaker
/// transport) buffer/parse PayloadPart bytes as `data: {...}` lines. See
/// `~/nvidia/projects/aws-issue/sample-InferenceBenchmarker/factories/sagemakerai_realtime/factories_llm_textgeneration_stream.py`
/// for the reference client-side parsing this mirrors.
fn sse_to_eventstream<S>(sse: S) -> impl Stream<Item = Result<Bytes, Infallible>>
where
    S: Stream<Item = Result<Bytes, Infallible>>,
{
    async_stream::stream! {
        futures::pin_mut!(sse);
        while let Some(chunk) = futures::StreamExt::next(&mut sse).await {
            let Ok(chunk) = chunk;
            for piece in chunk.split(|&b| b == b'\n') {
                let Some(rest) = piece.strip_prefix(b"data: ") else { continue };
                if rest.is_empty() || rest == b"[DONE]" {
                    continue;
                }
                // The AWS `PayloadPart.Bytes` member carries the raw inner
                // chat-completion-chunk JSON directly (see
                // `EventStreamMessage::payload_part`) — not the SSE-framed
                // `data: {...}\n` line. Emit the stripped payload so the frame
                // matches the documented contract and real bare-JSON wire form.
                let frame =
                    EventStreamMessage::payload_part(Bytes::copy_from_slice(rest)).encode();
                yield Ok::<Bytes, Infallible>(frame);
            }
        }
    }
}

fn eventstream_response<S>(body: S) -> Response
where
    S: Stream<Item = Result<Bytes, Infallible>> + Send + 'static,
{
    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "application/vnd.amazon.eventstream")
        .header(header::CACHE_CONTROL, "no-cache")
        .body(Body::from_stream(body))
        .expect("body ok")
}

/// AWS SageMaker Runtime `InvokeEndpointWithResponseStream`: `POST
/// /endpoints/{endpoint_name}/invocations-response-stream`. Streams the same
/// OpenAI chat-completion-chunk payloads `chat_stream` produces, each wrapped
/// as a `PayloadPart` AWS eventstream binary frame.
pub async fn sagemaker_invoke_stream(
    State(state): State<Arc<AppState>>,
    axum::extract::Path(endpoint_name): axum::extract::Path<String>,
    Json(body): Json<Value>,
) -> AppResult<Response> {
    if let Some(e) = maybe_inject_error(&state) {
        return Err(e);
    }
    let req = sagemaker_request_to_chat(&endpoint_name, &body, true)?;
    let endpoint = "/endpoints/{endpoint_name}/invocations-response-stream";
    let start = Instant::now();
    state.recorder.init_model_config(&req.model);
    let req_gen = GenRequest::Chat(&req);
    let ctx = RequestCtx::build("chatcmpl", &req_gen, endpoint, start, &state);

    state.recorder.record_streaming_start(endpoint, &ctx.model);
    let include_usage = req.include_usage();
    let midstream_error = state.inject_midstream();
    let sse = chat_stream(
        state.clone(),
        ctx,
        endpoint.to_string(),
        include_usage,
        midstream_error,
    );
    Ok(eventstream_response(sse_to_eventstream(sse)))
}

/// Projects a Responses request onto the shared token and latency machinery.
fn responses_as_chat(req: &ResponsesRequest) -> ChatCompletionRequest {
    ChatCompletionRequest {
        model: req.model.clone(),
        messages: vec![Message {
            role: "user".into(),
            content: Value::String(req.prompt_text()),
        }],
        stream: false,
        stream_options: None,
        max_tokens: req.max_output_tokens,
        max_completion_tokens: None,
        ignore_eos: false,
        min_tokens: None,
        reasoning_effort: req.reasoning_effort.clone(),
        priority: None,
        mock_first_chunk_tokens: 1,
    }
}

/// OpenAI Responses API. Recovers the prompt from `input`/`instructions`, runs it
/// through the shared token/latency model, and emits the Responses wire shape the
/// runner consumes: non-streaming `{object:"response", status:"completed",
/// output:[{type:"message", content:[{type:"output_text", text}]}], usage}` or the streaming
/// `response.created` / `response.output_text.delta` / `response.completed`
/// event sequence.
pub async fn responses(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ResponsesRequest>,
) -> AppResult<Response> {
    if let Some(e) = maybe_inject_error(&state) {
        return Err(e);
    }
    let endpoint = "/v1/responses";
    let start = Instant::now();
    state.recorder.init_model_config(&req.model);
    let mock_chat = responses_as_chat(&req);
    let req_gen = GenRequest::Chat(&mock_chat);
    let mut ctx = RequestCtx::build("resp", &req_gen, endpoint, start, &state);
    ctx.request_id = make_request_id("resp");

    if req.stream {
        state.recorder.record_streaming_start(endpoint, &ctx.model);
        Ok(sse_response(responses_stream(
            state,
            ctx,
            endpoint.to_string(),
        )))
    } else {
        state.recorder.record_request_start(endpoint, &ctx.model);
        state.recorder.record_llm_inflight_start(&ctx.model);
        let (prefill, _decode) = ctx.latency_sim.wait_for_tokens(ctx.tokenized.count()).await;
        let latency = start.elapsed();
        let info = LLMLatencyInfo {
            e2e: latency,
            prefill,
            decode: latency.saturating_sub(prefill),
        };
        let body = build_responses_response(&ctx);
        let json_body = serde_json::to_vec(&body).map_err(internal_error)?;
        state.recorder.record_request_bytes(
            endpoint,
            ctx.tokenized.text.len() as u64,
            json_body.len() as u64,
        );
        state.recorder.record_llm_success(
            endpoint,
            &ctx.model,
            latency.as_secs_f64(),
            &ctx.usage,
            &info,
        );
        state.recorder.record_llm_inflight_end(&ctx.model);
        state.recorder.record_request_end(endpoint);

        Response::builder()
            .status(StatusCode::OK)
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(json_body))
            .map_err(internal_error)
    }
}

/// Responses uses `input_tokens` and `output_tokens` for prompt and completion.
fn responses_usage(ctx: &RequestCtx) -> Value {
    json!({
        "input_tokens": ctx.usage.prompt_tokens,
        "output_tokens": ctx.usage.completion_tokens,
        "total_tokens": ctx.usage.total_tokens,
    })
}

fn build_responses_response(ctx: &RequestCtx) -> Value {
    let mut output = Vec::new();
    if let Some(reasoning) = ctx.tokenized.reasoning_content() {
        output.push(json!({
            "type": "reasoning",
            "summary": [{"type": "summary_text", "text": reasoning}],
        }));
    }
    output.push(json!({
        "type": "message",
        "role": "assistant",
        "content": [{"type": "output_text", "text": ctx.tokenized.content()}],
    }));
    json!({
        "id": ctx.request_id,
        "object": "response",
        "status": "completed",
        "created_at": now_secs(),
        "model": ctx.model,
        "output": output,
        "usage": responses_usage(ctx),
    })
}

/// Stream the Responses event sequence with the shared per-token pacing so TTFT /
/// ITL reproduce the tuned mock. Only `response.output_text.delta` frames carry
/// generated content (the runner counts those toward OSL); the terminal
/// `response.completed` carries usage.
fn responses_stream(
    state: Arc<AppState>,
    ctx: RequestCtx,
    endpoint: String,
) -> impl Stream<Item = Result<Bytes, Infallible>> {
    let labeled = state.recorder.labeled(&endpoint, &ctx.model);
    async_stream::stream! {
        state.recorder.record_request_start(&endpoint, &ctx.model);
        state.recorder.record_llm_inflight_start(&ctx.model);

        let created = now_secs();
        yield Ok::<Bytes, Infallible>(sse_event("response.created", &json!({
            "type": "response.created",
            "response": {
                "id": ctx.request_id,
                "object": "response",
                "status": "in_progress",
                "created_at": created,
                "model": ctx.model,
            },
        })));

        let mut first_emit: Option<Instant> = None;
        let mut last_emit: Option<Instant> = None;
        let count = ctx.tokenized.tokens.len();

        if ctx.latency_sim.is_fast() {
            if count > 0 {
                state.recorder.record_zero_ttft_and_itls(&labeled, count - 1);
                state.recorder.record_streamed_tokens_fast(&labeled, count as u64);
            }
            for token in &ctx.tokenized.tokens {
                yield Ok::<Bytes, Infallible>(sse_event("response.output_text.delta", &json!({
                    "type": "response.output_text.delta",
                    "delta": token,
                })));
            }
        } else {
            for (index, token) in ctx.tokenized.tokens.iter().enumerate() {
                let emit_at = ctx.latency_sim.wait_for_index(index).await;
                if first_emit.is_none() {
                    first_emit = Some(emit_at);
                    state.recorder.record_ttft_fast(
                        &labeled,
                        emit_at.duration_since(ctx.start).as_secs_f64(),
                    );
                } else if let Some(last) = last_emit {
                    state.recorder.record_itl_fast(
                        &labeled,
                        emit_at.duration_since(last).as_secs_f64(),
                    );
                }
                last_emit = Some(emit_at);
                state.recorder.record_streamed_token_fast(&labeled);
                yield Ok::<Bytes, Infallible>(sse_event("response.output_text.delta", &json!({
                    "type": "response.output_text.delta",
                    "delta": token,
                })));
            }
        }

        yield Ok::<Bytes, Infallible>(sse_event("response.completed", &json!({
            "type": "response.completed",
            "response": {
                "id": ctx.request_id,
                "object": "response",
                "status": "completed",
                "model": ctx.model,
                "usage": responses_usage(&ctx),
            },
        })));

        let latency = ctx.start.elapsed();
        let prefill = first_emit
            .map(|at| at.duration_since(ctx.start))
            .unwrap_or(std::time::Duration::ZERO);
        let info = LLMLatencyInfo {
            e2e: latency,
            prefill,
            decode: latency.saturating_sub(prefill),
        };
        state.recorder.record_llm_success(
            &endpoint,
            &ctx.model,
            latency.as_secs_f64(),
            &ctx.usage,
            &info,
        );
        state.recorder.record_llm_inflight_end(&ctx.model);
        state.recorder.record_request_end(&endpoint);
    }
}

pub async fn embeddings(
    State(state): State<Arc<AppState>>,
    Json(req): Json<EmbeddingRequest>,
) -> AppResult<Response> {
    if let Some(e) = maybe_inject_error(&state) {
        return Err(e);
    }
    let endpoint = "/v1/embeddings";
    let start = Instant::now();
    let req_gen = GenRequest::Embedding(&req);
    let ctx = RequestCtx::build("emb", &req_gen, endpoint, start, &state);
    let inputs = req.inputs();

    state.recorder.record_request_start(endpoint, &req.model);
    wait_for_processing(
        state.config.embedding_base_latency,
        state.config.embedding_per_input_latency,
        inputs.len(),
    )
    .await;

    let data: Vec<Value> = inputs
        .iter()
        .enumerate()
        .map(|(i, text)| {
            json!({
                "object": "embedding",
                "index": i,
                "embedding": generate_embedding(text, 768),
            })
        })
        .collect();
    let body = json!({
        "object": "list",
        "model": req.model,
        "data": data,
        "usage": ctx.usage,
    });

    state.recorder.record_embedding_success(
        endpoint,
        &req.model,
        ctx.usage.prompt_tokens,
        inputs.len(),
        start.elapsed().as_secs_f64(),
    );
    state.recorder.record_request_end(endpoint);

    let buf = serde_json::to_vec(&body).map_err(internal_error)?;
    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(buf))
        .map_err(internal_error)
}

pub fn generate_embedding(text: &str, dim: usize) -> Vec<f64> {
    let mut hasher = Blake2s256::new();
    hasher.update(text.as_bytes());
    let digest = hasher.finalize();
    let seed: u64 = u64::from_be_bytes(digest[0..8].try_into().unwrap());
    let mut rng = RustRandomGenerator::from_seed(Some(seed));
    (0..dim).map(|_| rng.random() - 0.5).collect()
}

pub(crate) fn compute_mock_score(query: &str, passage: &str) -> f64 {
    let mut hasher = Blake2s256::new();
    hasher.update(query.as_bytes());
    hasher.update(b"|");
    hasher.update(passage.as_bytes());
    let digest = hasher.finalize();
    // Eight digest bytes are sufficient for the deterministic score.
    let prefix = u64::from_be_bytes(digest[0..8].try_into().unwrap());
    (prefix % 1000) as f64 / 1000.0
}

fn ranked_scores(query: &str, passages: &[&str]) -> Vec<(usize, f64)> {
    let mut scores: Vec<(usize, f64)> = passages
        .iter()
        .enumerate()
        .map(|(i, p)| (i, compute_mock_score(query, p)))
        .collect();
    scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scores
}

async fn handle_ranking_common(
    state: Arc<AppState>,
    endpoint: &str,
    model: &str,
    query: &str,
    passages: Vec<&str>,
    prompt_tokens: usize,
) -> (String, Vec<(usize, f64)>, Duration) {
    let start = Instant::now();
    state.recorder.record_request_start(endpoint, model);
    let scores = ranked_scores(query, &passages);
    wait_for_processing(
        state.config.ranking_base_latency,
        state.config.ranking_per_passage_latency,
        passages.len(),
    )
    .await;
    state.recorder.record_ranking_success(
        endpoint,
        model,
        prompt_tokens,
        passages.len(),
        start.elapsed().as_secs_f64(),
    );
    state.recorder.record_request_end(endpoint);
    (make_request_id("rank"), scores, start.elapsed())
}

pub async fn nim_ranking(
    State(state): State<Arc<AppState>>,
    Json(req): Json<RankingRequest>,
) -> AppResult<Response> {
    if let Some(e) = maybe_inject_error(&state) {
        return Err(e);
    }
    let endpoint = "/v1/ranking";
    let req_gen = GenRequest::Ranking(&req);
    let tokenized = tokenize_request(&req_gen);
    let passages: Vec<&str> = req.passage_texts();
    let (req_id, scores, _) = handle_ranking_common(
        state.clone(),
        endpoint,
        &req.model,
        req.query_text(),
        passages,
        tokenized.prompt_token_count,
    )
    .await;
    let rankings: Vec<Value> = scores
        .into_iter()
        .map(|(i, s)| json!({"index": i, "relevance_score": s}))
        .collect();
    let usage = tokenized.usage();
    Ok(Json(json!({
        "id": req_id,
        "object": "rankings",
        "model": req.model,
        "rankings": rankings,
        "usage": usage,
    }))
    .into_response())
}

/// Shared `{ "results": [ { "index": i, "<score_key>": score }, ... ] }` rerank
/// envelope for the HF TEI (`/rerank`, `score`) and Cohere (`/v2/rerank`,
/// `relevance_score`) endpoints, which are byte-identical apart from the
/// per-result score key. Runs the common ranking pipeline
/// ([`handle_ranking_common`]) then projects the sorted scores into the envelope.
async fn rerank_results_response(
    state: Arc<AppState>,
    endpoint: &str,
    model: &str,
    query: &str,
    passages: Vec<&str>,
    prompt_tokens: usize,
    score_key: &str,
) -> Response {
    let (_, scores, _) =
        handle_ranking_common(state, endpoint, model, query, passages, prompt_tokens).await;
    let results: Vec<Value> = scores
        .into_iter()
        .map(|(i, s)| {
            let mut obj = serde_json::Map::new();
            obj.insert("index".to_string(), json!(i));
            obj.insert(score_key.to_string(), json!(s));
            Value::Object(obj)
        })
        .collect();
    Json(json!({ "results": results })).into_response()
}

pub async fn hf_tei_rerank(
    State(state): State<Arc<AppState>>,
    Json(req): Json<HFTEIRerankRequest>,
) -> AppResult<Response> {
    if let Some(e) = maybe_inject_error(&state) {
        return Err(e);
    }
    let endpoint = "/rerank";
    let req_gen = GenRequest::HFTEIRerank(&req);
    let tokenized = tokenize_request(&req_gen);
    Ok(rerank_results_response(
        state.clone(),
        endpoint,
        &req.model,
        req.query_text(),
        req.passage_texts(),
        tokenized.prompt_token_count,
        "score",
    )
    .await)
}

pub async fn cohere_rerank(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CohereRerankRequest>,
) -> AppResult<Response> {
    if let Some(e) = maybe_inject_error(&state) {
        return Err(e);
    }
    let endpoint = "/v2/rerank";
    let req_gen = GenRequest::CohereRerank(&req);
    let tokenized = tokenize_request(&req_gen);
    Ok(rerank_results_response(
        state.clone(),
        endpoint,
        &req.model,
        req.query_text(),
        req.passage_texts(),
        tokenized.prompt_token_count,
        "relevance_score",
    )
    .await)
}

const BOUNDING_BOX_CATEGORIES: &[&str] = &["title", "table", "figure", "text", "header", "footer"];

fn generate_bounding_boxes(url: &str) -> serde_json::Map<String, Value> {
    let mut hasher = Blake2s256::new();
    hasher.update(url.as_bytes());
    let digest = hasher.finalize();
    let seed = u64::from_be_bytes(digest[0..8].try_into().unwrap());
    let mut rng = RustRandomGenerator::from_seed(Some(seed));
    let num_boxes: i64 = rng.randint(1, 5).expect("1..=5 is a valid inclusive range");
    let mut out: serde_json::Map<String, Value> = serde_json::Map::new();
    for _ in 0..num_boxes {
        let category = *rng
            .choice(BOUNDING_BOX_CATEGORIES)
            .expect("bounding-box categories are non-empty");
        let x_min: f64 = round_4(rng.uniform(0.0, 0.5));
        let y_min: f64 = round_4(rng.uniform(0.0, 0.5));
        let x_max: f64 = round_4(rng.uniform(x_min + 0.05, 1.0));
        let y_max: f64 = round_4(rng.uniform(y_min + 0.05, 1.0));
        let confidence: f64 = round_4(rng.uniform(0.7, 1.0));
        let box_json = json!({
            "x_min": x_min,
            "y_min": y_min,
            "x_max": x_max,
            "y_max": y_max,
            "confidence": confidence,
        });
        out.entry(category.to_string())
            .or_insert_with(|| Value::Array(Vec::new()))
            .as_array_mut()
            .unwrap()
            .push(box_json);
    }
    out
}

fn round_4(x: f64) -> f64 {
    (x * 10_000.0).round() / 10_000.0
}

/// Only `http(s)` URLs are fetched; `data:` URIs and other schemes are inline
/// or unroutable and are left untouched.
fn is_fetchable_url(url: &str) -> bool {
    url.starts_with("http://") || url.starts_with("https://")
}

/// Pull fetchable `image_url`/`video_url` targets out of OpenAI-style
/// multimodal chat content. Each part may carry either a bare string URL or an
/// object with a `url` field (`{"type": "image_url", "image_url": {"url": ...}}`).
fn collect_content_urls(messages: &[Message]) -> Vec<String> {
    let mut urls = Vec::new();
    for msg in messages {
        let Some(parts) = msg.content.as_array() else {
            continue;
        };
        for part in parts {
            for key in ["image_url", "video_url"] {
                let Some(field) = part.get(key) else {
                    continue;
                };
                let url = match field {
                    Value::String(s) => Some(s.as_str()),
                    Value::Object(_) => field.get("url").and_then(Value::as_str),
                    _ => None,
                };
                if let Some(u) = url
                    && is_fetchable_url(u)
                {
                    urls.push(u.to_string());
                }
            }
        }
    }
    urls
}

/// GET a single URL and drain its body, returning the byte count. Every failure
/// mode (bad URL, connect/transfer error, timeout) is logged and reported as `0`
/// bytes so a fetch never fails the mock response.
async fn fetch_one(client: &ContentFetchClient, url: &str, timeout: Duration) -> u64 {
    let uri = match url.parse::<Uri>() {
        Ok(u) => u,
        Err(e) => {
            tracing::warn!(url = %url, error = %e, "content url parse failed");
            return 0;
        }
    };
    let request = async {
        let req = Request::builder()
            .uri(uri)
            .body(Empty::<Bytes>::new())
            .map_err(Box::<dyn std::error::Error + Send + Sync>::from)?;
        let resp = client.request(req).await?;
        let status = resp.status();
        let bytes = resp.into_body().collect().await?.to_bytes().len() as u64;
        Ok::<_, Box<dyn std::error::Error + Send + Sync>>((status, bytes))
    };
    match tokio::time::timeout(timeout, request).await {
        Ok(Ok((status, bytes))) => {
            tracing::debug!(url = %url, %status, bytes, "content fetch ok");
            bytes
        }
        Ok(Err(e)) => {
            tracing::warn!(url = %url, error = %e, "content fetch failed");
            0
        }
        Err(_) => {
            tracing::warn!(url = %url, ?timeout, "content fetch timed out");
            0
        }
    }
}

/// Fetch every URL concurrently to exercise the remote content server. No-op
/// (and no allocation cost beyond the empty slice) when fetching is disabled.
async fn fetch_content_urls(state: &Arc<AppState>, endpoint: &str, urls: &[String]) {
    let Some(client) = state.content_fetch_client.as_ref() else {
        return;
    };
    if urls.is_empty() {
        return;
    }
    let timeout = Duration::from_secs_f64(state.config.content_fetch_timeout);
    let fetches = urls.iter().map(|url| fetch_one(client, url, timeout));
    let total: u64 = futures::future::join_all(fetches).await.into_iter().sum();
    if total > 0 {
        state.recorder.record_content_bytes_fetched(endpoint, total);
    }
}

pub async fn image_retrieval(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ImageRetrievalRequest>,
) -> AppResult<Response> {
    if let Some(e) = maybe_inject_error(&state) {
        return Err(e);
    }
    let endpoint = "/v1/image/infer";
    let start = Instant::now();
    let num_images = req.input.len();
    state
        .recorder
        .record_request_start(endpoint, "image-retrieval");
    wait_for_processing(
        state.config.image_retrieval_base_latency,
        state.config.image_retrieval_per_image_latency,
        num_images,
    )
    .await;

    let mut total_size_mb = 0.0;
    let mut fetched_total: u64 = 0;
    let mut data = Vec::new();
    for (i, img) in req.input.iter().enumerate() {
        let bounding = generate_bounding_boxes(&img.url);
        data.push(json!({
            "index": i,
            "bounding_boxes": bounding,
        }));
        // With fetching enabled, size the transfer by the bytes actually
        // downloaded from the URL; otherwise fall back to the inline base64
        // string-length proxy.
        let size_bytes = match state.content_fetch_client.as_ref() {
            Some(client) if is_fetchable_url(&img.url) => {
                let timeout = Duration::from_secs_f64(state.config.content_fetch_timeout);
                let b = fetch_one(client, &img.url, timeout).await;
                fetched_total += b;
                b as f64
            }
            _ => img.url.len() as f64 / 1.37,
        };
        total_size_mb += size_bytes / (1024.0 * 1024.0);
    }
    if fetched_total > 0 {
        state
            .recorder
            .record_content_bytes_fetched(endpoint, fetched_total);
    }

    state.recorder.record_image_retrieval_success(
        endpoint,
        num_images,
        start.elapsed().as_secs_f64(),
    );
    state.recorder.record_request_end(endpoint);

    Ok(Json(json!({
        "data": data,
        "usage": { "images_size_mb": round_4(total_size_mb) },
    }))
    .into_response())
}

// KServe HTTP and gRPC share tensor names and behavior selection. HTTP text
// responses are non-streaming; KServe streaming is available over gRPC.

const KSERVE_V2_TEXT_INPUT: &str = "text_input";
const KSERVE_V2_QUERY: &str = "query";
const KSERVE_V2_QUERIES: &str = "queries";
const KSERVE_V2_PASSAGES: &str = "passages";
const KSERVE_V2_PROMPT: &str = "prompt";

/// All string values of the first v2 input tensor named `name`.
fn v2_tensor_texts(body: &Value, name: &str) -> Vec<String> {
    body.get("inputs")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_object)
        .find(|tensor| tensor.get("name").and_then(Value::as_str) == Some(name))
        .and_then(|tensor| tensor.get("data").and_then(Value::as_array))
        .map(|data| {
            data.iter()
                .map(|value| match value {
                    Value::String(s) => s.clone(),
                    other => other.to_string(),
                })
                .collect()
        })
        .unwrap_or_default()
}

/// First integer value of the v2 input tensor named `name` (e.g. `max_tokens`).
fn v2_tensor_first_int(body: &Value, name: &str) -> Option<i64> {
    body.get("inputs")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_object)
        .find(|tensor| tensor.get("name").and_then(Value::as_str) == Some(name))
        .and_then(|tensor| tensor.get("data").and_then(Value::as_array))
        .and_then(|data| data.first())
        .and_then(|value| value.as_i64().or_else(|| value.as_str()?.parse().ok()))
}

/// True when a v2 input tensor named `name` is present.
fn v2_has_input(body: &Value, name: &str) -> bool {
    body.get("inputs")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_object)
        .any(|tensor| tensor.get("name").and_then(Value::as_str) == Some(name))
}

/// The single KServe text output for a generated turn: reasoning tokens (if the
/// model is a reasoning model) folded in front of the output tokens. KServe text
/// has no separate reasoning channel, and a reasoning model with a small
/// `max_tokens` budget can spend it all on reasoning, leaving `content()` empty —
/// folding keeps `text_output` non-empty, matching `crate::grpc::generated_tokens`.
fn kserve_text_output(tokenized: &TokenizedText) -> String {
    let mut text = tokenized.reasoning_content_tokens.concat();
    text.push_str(&tokenized.content());
    text
}

/// One KServe v2 output tensor as JSON.
fn v2_output(name: &str, datatype: &str, shape: Vec<usize>, data: Vec<Value>) -> Value {
    json!({"name": name, "datatype": datatype, "shape": shape, "data": data})
}

/// Wrap output tensors into a KServe v2 `ModelInferResponse` JSON body.
fn v2_response(model: &str, id: &str, outputs: Vec<Value>) -> Value {
    json!({"model_name": model, "id": id, "outputs": outputs})
}

/// Resolve the effective behavior for an HTTP v2 infer request from the config
/// override, falling back to auto-detection on the input tensor names.
fn resolve_v2_behavior(state: &AppState, body: &Value) -> crate::grpc::GrpcBehavior {
    use crate::grpc::GrpcBehavior;
    match state.config.grpc_behavior {
        GrpcBehavior::Auto => {
            if v2_has_input(body, KSERVE_V2_PASSAGES) {
                GrpcBehavior::Rankings
            } else if v2_has_input(body, KSERVE_V2_PROMPT)
                && !v2_has_input(body, KSERVE_V2_TEXT_INPUT)
            {
                GrpcBehavior::Images
            } else {
                GrpcBehavior::Text
            }
        }
        forced => forced,
    }
}

/// KServe v2 Open Inference Protocol: `POST /v2/models/{model}/infer`.
pub async fn kserve_v2_infer(
    State(state): State<Arc<AppState>>,
    axum::extract::Path(model): axum::extract::Path<String>,
    Json(body): Json<Value>,
) -> AppResult<Response> {
    if let Some(e) = maybe_inject_error(&state) {
        return Err(e);
    }
    let endpoint = "/v2/models/{model}/infer";
    let start = Instant::now();
    let req_id = make_request_id("kserve");

    match resolve_v2_behavior(&state, &body) {
        crate::grpc::GrpcBehavior::Rankings => {
            let query = v2_tensor_texts(&body, KSERVE_V2_QUERY)
                .into_iter()
                .next()
                .or_else(|| v2_tensor_texts(&body, KSERVE_V2_QUERIES).into_iter().next())
                .unwrap_or_default();
            let passages = v2_tensor_texts(&body, KSERVE_V2_PASSAGES);
            state.recorder.record_request_start(endpoint, &model);
            let scores: Vec<Value> = passages
                .iter()
                .map(|p| json!(compute_mock_score(&query, p)))
                .collect();
            let outputs = vec![v2_output("scores", "FP32", vec![scores.len()], scores)];
            state
                .recorder
                .record_basic_success(endpoint, start.elapsed().as_secs_f64());
            state.recorder.record_request_end(endpoint);
            Ok(Json(v2_response(&model, &req_id, outputs)).into_response())
        }
        crate::grpc::GrpcBehavior::Images => {
            let prompt = v2_tensor_texts(&body, KSERVE_V2_PROMPT)
                .into_iter()
                .next()
                .unwrap_or_default();
            state.recorder.record_request_start(endpoint, &model);
            let b64 = mock_jpeg_b64(&prompt, 0);
            let outputs = vec![v2_output(
                "generated_image",
                "BYTES",
                vec![1],
                vec![Value::String(b64)],
            )];
            state
                .recorder
                .record_basic_success(endpoint, start.elapsed().as_secs_f64());
            state.recorder.record_request_end(endpoint);
            Ok(Json(v2_response(&model, &req_id, outputs)).into_response())
        }
        // Image tensors do not affect the generated `text_output`.
        crate::grpc::GrpcBehavior::Text | crate::grpc::GrpcBehavior::Auto => {
            let prompt = v2_tensor_texts(&body, KSERVE_V2_TEXT_INPUT)
                .into_iter()
                .next()
                .unwrap_or_default();
            let max_tokens = v2_tensor_first_int(&body, "max_tokens")
                .filter(|value| *value > 0)
                .map(|value| value as usize);
            let mock_chat = ChatCompletionRequest {
                model: model.clone(),
                messages: vec![Message {
                    role: "user".into(),
                    content: Value::String(prompt),
                }],
                stream: false,
                stream_options: None,
                max_tokens,
                max_completion_tokens: None,
                ignore_eos: false,
                min_tokens: None,
                reasoning_effort: None,
                priority: None,
                mock_first_chunk_tokens: 1,
            };
            let req_gen = GenRequest::Chat(&mock_chat);
            let ctx = RequestCtx::build("kserve", &req_gen, endpoint, start, &state);

            state.recorder.init_model_config(&model);
            state.recorder.record_request_start(endpoint, &model);
            state.recorder.record_llm_inflight_start(&model);
            let (prefill, _decode) = ctx.latency_sim.wait_for_tokens(ctx.tokenized.count()).await;
            let latency = start.elapsed();
            let info = LLMLatencyInfo {
                e2e: latency,
                prefill,
                decode: latency.saturating_sub(prefill),
            };
            let text = kserve_text_output(&ctx.tokenized);
            let outputs = vec![v2_output(
                "text_output",
                "BYTES",
                vec![1],
                vec![Value::String(text)],
            )];
            state.recorder.record_llm_success(
                endpoint,
                &model,
                latency.as_secs_f64(),
                &ctx.usage,
                &info,
            );
            state.recorder.record_llm_inflight_end(&model);
            state.recorder.record_request_end(endpoint);
            Ok(Json(v2_response(&model, &ctx.request_id, outputs)).into_response())
        }
    }
}

/// KServe v1 inference: `POST /v1/models/{model}:predict`. The path parameter
/// arrives as `{model}:predict`; the `:predict` verb suffix is stripped. Reads
/// `instances[].text` (or the first string field), generates text, and returns
/// `{"predictions": [{"output": "<text>"}]}`.
pub async fn kserve_v1_predict(
    State(state): State<Arc<AppState>>,
    axum::extract::Path(model_verb): axum::extract::Path<String>,
    Json(body): Json<Value>,
) -> AppResult<Response> {
    if let Some(e) = maybe_inject_error(&state) {
        return Err(e);
    }
    let endpoint = "/v1/models/{model}:predict";
    let start = Instant::now();
    let model = model_verb
        .strip_suffix(":predict")
        .unwrap_or(&model_verb)
        .to_string();

    let text = body
        .get("instances")
        .and_then(Value::as_array)
        .and_then(|instances| instances.first())
        .and_then(|instance| {
            instance
                .as_object()
                .and_then(|object| {
                    object
                        .get("text")
                        .or_else(|| object.values().find(|value| value.is_string()))
                        .and_then(Value::as_str)
                        .map(ToString::to_string)
                })
                .or_else(|| instance.as_str().map(ToString::to_string))
        })
        .unwrap_or_default();

    let mock_chat = ChatCompletionRequest {
        model: model.clone(),
        messages: vec![Message {
            role: "user".into(),
            content: Value::String(text),
        }],
        stream: false,
        stream_options: None,
        max_tokens: None,
        max_completion_tokens: None,
        ignore_eos: false,
        min_tokens: None,
        reasoning_effort: None,
        priority: None,
        mock_first_chunk_tokens: 1,
    };
    let req_gen = GenRequest::Chat(&mock_chat);
    let ctx = RequestCtx::build("kserve-v1", &req_gen, endpoint, start, &state);

    state.recorder.init_model_config(&model);
    state.recorder.record_request_start(endpoint, &model);
    state.recorder.record_llm_inflight_start(&model);
    let (prefill, _decode) = ctx.latency_sim.wait_for_tokens(ctx.tokenized.count()).await;
    let latency = start.elapsed();
    let info = LLMLatencyInfo {
        e2e: latency,
        prefill,
        decode: latency.saturating_sub(prefill),
    };
    let output = kserve_text_output(&ctx.tokenized);
    state
        .recorder
        .record_llm_success(endpoint, &model, latency.as_secs_f64(), &ctx.usage, &info);
    state.recorder.record_llm_inflight_end(&model);
    state.recorder.record_request_end(endpoint);

    Ok(Json(json!({
        "predictions": [{"output": output}],
    }))
    .into_response())
}

/// KServe v2 model readiness: `GET /v2/models/{model}/ready` — the mock's model
/// is always ready.
pub async fn kserve_v2_model_ready() -> impl IntoResponse {
    Json(json!({"name": "", "ready": true}))
}

/// KServe v2 server readiness: `GET /v2/health/ready`.
pub async fn kserve_v2_health_ready() -> impl IntoResponse {
    Json(json!({"ready": true}))
}

pub async fn custom_multimodal(
    State(state): State<Arc<AppState>>,
    Json(req): Json<Value>,
) -> AppResult<Response> {
    if let Some(e) = maybe_inject_error(&state) {
        return Err(e);
    }
    let endpoint = "/v1/custom-multimodal";
    let start = Instant::now();

    let model_id = req
        .get("inference_params")
        .and_then(|p| p.get("model_id"))
        .and_then(Value::as_str)
        .unwrap_or("default-model")
        .to_string();

    let bundle = req.get("modality_bundle").cloned().unwrap_or(Value::Null);
    let text_fragments: Vec<String> = bundle
        .get("text_fragments")
        .and_then(Value::as_array)
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default();
    let visual_assets = bundle.get("visual_assets").cloned().unwrap_or(Value::Null);
    let images_len = visual_assets
        .get("images")
        .and_then(Value::as_array)
        .map(|v| v.len())
        .unwrap_or(0);
    let videos_len = visual_assets
        .get("videos")
        .and_then(Value::as_array)
        .map(|v| v.len())
        .unwrap_or(0);
    let audio_len = bundle
        .get("audio_streams")
        .and_then(Value::as_array)
        .map(|v| v.len())
        .unwrap_or(0);

    let text_content = if text_fragments.is_empty() {
        "default text".to_string()
    } else {
        text_fragments.join(" ")
    };
    let mock_req = ChatCompletionRequest {
        model: model_id.clone(),
        messages: vec![Message {
            role: "user".into(),
            content: Value::String(text_content),
        }],
        stream: false,
        stream_options: None,
        max_tokens: None,
        max_completion_tokens: None,
        ignore_eos: false,
        min_tokens: None,
        reasoning_effort: None,
        priority: None,
        mock_first_chunk_tokens: 1,
    };
    let req_gen = GenRequest::Chat(&mock_req);
    let ctx = RequestCtx::build("chatcmpl", &req_gen, endpoint, start, &state);

    state.recorder.record_request_start(endpoint, &model_id);
    state.recorder.record_llm_inflight_start(&model_id);
    let (prefill, _decode) = ctx.latency_sim.wait_for_tokens(ctx.tokenized.count()).await;
    let latency = start.elapsed();
    let info = LLMLatencyInfo {
        e2e: latency,
        prefill,
        decode: latency.saturating_sub(prefill),
    };
    state.recorder.record_llm_success(
        endpoint,
        &model_id,
        latency.as_secs_f64(),
        &ctx.usage,
        &info,
    );
    state.recorder.record_llm_inflight_end(&model_id);
    state.recorder.record_request_end(endpoint);

    let mut response_text = format!("Processed {} text fragments", text_fragments.len());
    if images_len > 0 {
        response_text.push_str(&format!(", {images_len} images"));
    }
    if videos_len > 0 {
        response_text.push_str(&format!(", {videos_len} videos"));
    }
    if audio_len > 0 {
        response_text.push_str(&format!(", {audio_len} audio streams"));
    }
    Ok(Json(json!({
        "text": response_text,
        "completion": {
            "generated_text": response_text,
            "metadata": {
                "tokens_used": {
                    "input": ctx.usage.prompt_tokens,
                    "output": ctx.usage.completion_tokens,
                    "total": ctx.usage.total_tokens,
                }
            }
        }
    }))
    .into_response())
}

pub async fn tgi_generate(
    State(state): State<Arc<AppState>>,
    Json(req): Json<TGIGenerateRequest>,
) -> AppResult<Response> {
    if let Some(e) = maybe_inject_error(&state) {
        return Err(e);
    }
    let endpoint = "/generate";
    let start = Instant::now();
    let req_gen = GenRequest::TGI(&req);
    let ctx = RequestCtx::build("cmpl", &req_gen, endpoint, start, &state);

    state.recorder.record_request_start(endpoint, &ctx.model);
    let _ = ctx.latency_sim.wait_for_tokens(ctx.tokenized.count()).await;
    let latency = start.elapsed();
    state
        .recorder
        .record_tgi_success(endpoint, &ctx.usage, latency.as_secs_f64());
    state.recorder.record_request_end(endpoint);

    Ok(Json(json!({ "generated_text": ctx.tokenized.content() })).into_response())
}

pub async fn tgi_generate_stream(
    State(state): State<Arc<AppState>>,
    Json(req): Json<TGIGenerateRequest>,
) -> AppResult<Response> {
    if let Some(e) = maybe_inject_error(&state) {
        return Err(e);
    }
    let endpoint = "/generate_stream";
    let start = Instant::now();
    let req_gen = GenRequest::TGI(&req);
    let ctx = RequestCtx::build("cmpl", &req_gen, endpoint, start, &state);

    state.recorder.record_streaming_start(endpoint, &ctx.model);
    let body = tgi_stream(state.clone(), ctx, endpoint.to_string());
    Ok(sse_response(body))
}

pub(crate) fn mock_jpeg_b64(prompt: &str, index: u32) -> String {
    let combined = format!("{prompt}|{index}");
    let mut hasher = Blake2s256::new();
    hasher.update(combined.as_bytes());
    let digest = hasher.finalize();

    // Clamp ranges because the digest is 32 bytes.
    let slice_safe = |start: usize, end: usize| -> &[u8] {
        let s = start.min(digest.len());
        let e = end.min(digest.len()).max(s);
        &digest[s..e]
    };

    let mut jpeg: Vec<u8> = Vec::new();
    jpeg.extend_from_slice(&[0xFF, 0xD8]);
    jpeg.extend_from_slice(b"\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00");
    jpeg.extend_from_slice(b"\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00");
    jpeg.extend_from_slice(
        b"\xff\xc4\x00\x14\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x09",
    );
    jpeg.extend_from_slice(
        b"\xff\xc4\x00\x14\x10\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00",
    );
    jpeg.extend_from_slice(&[0xFF, 0xDB, 0x00, 0x43, 0x00]);
    jpeg.extend_from_slice(slice_safe(0, 64));
    jpeg.extend_from_slice(&[0xFF, 0xDA, 0x00, 0x08, 0x01, 0x01, 0x00, 0x00, 0x3F, 0x00]);
    jpeg.extend_from_slice(slice_safe(64, 80));
    jpeg.extend_from_slice(&[0xFF, 0xD9]);
    base64::engine::general_purpose::STANDARD.encode(jpeg)
}

pub async fn image_generation(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ImageGenerationRequest>,
) -> AppResult<Response> {
    if let Some(e) = maybe_inject_error(&state) {
        return Err(e);
    }
    let endpoint = "/v1/images/generations";
    let start = Instant::now();
    let mock_chat = ChatCompletionRequest {
        model: req.model.clone(),
        messages: vec![Message {
            role: "user".into(),
            content: Value::String(req.prompt.clone()),
        }],
        stream: false,
        stream_options: None,
        max_tokens: None,
        max_completion_tokens: None,
        ignore_eos: false,
        min_tokens: None,
        reasoning_effort: None,
        priority: None,
        mock_first_chunk_tokens: 1,
    };
    let req_gen = GenRequest::Chat(&mock_chat);
    let ctx = RequestCtx::build("img", &req_gen, endpoint, start, &state);

    if req.stream {
        state.recorder.record_streaming_start(endpoint, &ctx.model);
        let req_clone = req.clone();
        let body = image_stream(state.clone(), ctx, req_clone, endpoint.to_string());
        Ok(sse_response(body))
    } else {
        state.recorder.record_request_start(endpoint, &req.model);
        state.recorder.record_llm_inflight_start(&req.model);
        let (prefill, _decode) = ctx.latency_sim.wait_for_tokens(ctx.tokenized.count()).await;
        let latency = start.elapsed();
        let info = LLMLatencyInfo {
            e2e: latency,
            prefill,
            decode: latency.saturating_sub(prefill),
        };

        let mut data = Vec::with_capacity(req.n as usize);
        for i in 0..req.n {
            let mut chunk = json!({
                "b64_json": mock_jpeg_b64(&req.prompt, i),
            });
            if matches!(req.response_format, ImageResponseFormat::Url) {
                chunk["url"] = Value::String(format!("https://mock.image.url/{i}"));
            }
            data.push(chunk);
        }
        let mut body = json!({
            "created": now_secs(),
            "data": data,
        });
        if let Some(size) = &req.size {
            body["size"] = Value::String(size.clone());
        }
        if let Some(quality) = &req.quality {
            body["quality"] = Value::String(quality.clone());
        }
        if let Some(style) = &req.style {
            body["style"] = Value::String(style.clone());
        }
        body["usage"] = serde_json::to_value(&ctx.usage).unwrap();

        state.recorder.record_llm_success(
            endpoint,
            &req.model,
            latency.as_secs_f64(),
            &ctx.usage,
            &info,
        );
        state.recorder.record_llm_inflight_end(&req.model);
        state.recorder.record_request_end(endpoint);

        Ok(Json(body).into_response())
    }
}

pub async fn solido_rag(
    State(state): State<Arc<AppState>>,
    Json(req): Json<SolidoRAGRequest>,
) -> AppResult<Response> {
    if let Some(e) = maybe_inject_error(&state) {
        return Err(e);
    }
    let endpoint = "/rag/api/prompt";
    let start = Instant::now();
    let query_text = req.query.join(" ");
    let mock_chat = ChatCompletionRequest {
        model: req.inference_model.clone(),
        messages: vec![Message {
            role: "user".into(),
            content: Value::String(query_text.clone()),
        }],
        stream: false,
        stream_options: None,
        max_tokens: None,
        max_completion_tokens: None,
        ignore_eos: false,
        min_tokens: None,
        reasoning_effort: None,
        priority: None,
        mock_first_chunk_tokens: 1,
    };
    let req_gen = GenRequest::Chat(&mock_chat);
    let ctx = RequestCtx::build("rag", &req_gen, endpoint, start, &state);

    state
        .recorder
        .record_request_start(endpoint, &req.inference_model);
    state
        .recorder
        .record_llm_inflight_start(&req.inference_model);
    let (prefill, _decode) = ctx.latency_sim.wait_for_tokens(ctx.tokenized.count()).await;
    let latency = start.elapsed();
    let info = LLMLatencyInfo {
        e2e: latency,
        prefill,
        decode: latency.saturating_sub(prefill),
    };

    let mut sources = Vec::new();
    let num_sources = 3.min(req.query.len());
    for i in 0..num_sources {
        let combined = format!("{}|source{}", query_text, i);
        let mut hasher = Blake2s256::new();
        hasher.update(combined.as_bytes());
        let digest = hasher.finalize();
        let hex: String = digest.iter().map(|b| format!("{b:02x}")).collect();
        let short = &hex[0..8];
        let preview: String = query_text.chars().take(50).collect();
        sources.push(json!({
            "id": format!("doc_{short}"),
            "title": format!("Document {}", i + 1),
            "score": 0.9 - (i as f64) * 0.1,
            "content": format!("Source content for query: {}...", preview),
        }));
    }

    state.recorder.record_llm_success(
        endpoint,
        &req.inference_model,
        latency.as_secs_f64(),
        &ctx.usage,
        &info,
    );
    state.recorder.record_llm_inflight_end(&req.inference_model);
    state.recorder.record_request_end(endpoint);

    Ok(Json(json!({
        "content": ctx.tokenized.content(),
        "sources": sources,
        "filters": req.filters,
        "inference_model": req.inference_model,
    }))
    .into_response())
}

fn sse_chunk(value: &Value) -> Bytes {
    let mut out = Vec::with_capacity(256);
    out.extend_from_slice(b"data: ");
    serde_json::to_writer(&mut out, value).unwrap();
    out.extend_from_slice(b"\n\n");
    Bytes::from(out)
}

/// Serializes directly into an SSE frame without an intermediate JSON map.
fn sse_chunk_ser<T: serde::Serialize>(value: &T) -> Bytes {
    let mut out = Vec::with_capacity(256);
    out.extend_from_slice(b"data: ");
    serde_json::to_writer(&mut out, value).expect("serialize");
    out.extend_from_slice(b"\n\n");
    Bytes::from(out)
}

fn write_sse_into<T: serde::Serialize>(buf: &mut Vec<u8>, value: &T) {
    buf.extend_from_slice(b"data: ");
    serde_json::to_writer(&mut *buf, value).expect("serialize");
    buf.extend_from_slice(b"\n\n");
}

#[derive(serde::Serialize)]
struct ChatChoiceDelta<'a> {
    index: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    finish_reason: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    speculative_decoding_stats: Option<&'a Value>,
    delta: ChatDelta<'a>,
}

#[derive(serde::Serialize)]
struct ChatDelta<'a> {
    #[serde(skip_serializing_if = "Option::is_none")]
    role: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_content: Option<&'a str>,
    /// Streamed function tool-call deltas (`--tool-call-rate`). Absent on every
    /// normal frame, so a non-tool-call stream is byte-identical. Each frame
    /// carries one delta; the runner merges them by `index` across frames,
    /// concatenating `function.arguments`.
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<[ToolCallDelta<'a>; 1]>,
}

/// One `delta.tool_calls[*]` entry. `id`/`type`/`function.name` are sent on the
/// first frame of a call; later frames omit them and carry only the next
/// `function.arguments` fragment (matching how real streamed tool calls arrive).
#[derive(serde::Serialize)]
struct ToolCallDelta<'a> {
    index: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    id: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none", rename = "type")]
    kind: Option<&'static str>,
    function: ToolCallFunctionDelta<'a>,
}

#[derive(serde::Serialize)]
struct ToolCallFunctionDelta<'a> {
    #[serde(skip_serializing_if = "Option::is_none")]
    name: Option<&'a str>,
    arguments: &'a str,
}

#[derive(serde::Serialize)]
struct ChatStreamChunk<'a> {
    id: &'a str,
    object: &'static str,
    created: i64,
    model: &'a str,
    choices: [ChatChoiceDelta<'a>; 1],
}

#[derive(serde::Serialize)]
struct PartialUsage {
    prompt_tokens: usize,
    completion_tokens: usize,
    total_tokens: usize,
}

impl PartialUsage {
    fn cumulative(ctx: &RequestCtx, completion_tokens: usize) -> Self {
        Self {
            prompt_tokens: ctx.usage.prompt_tokens,
            completion_tokens,
            total_tokens: ctx.usage.prompt_tokens + completion_tokens,
        }
    }
}

#[derive(serde::Serialize)]
struct ChatStreamContinuousChunk<'a> {
    id: &'a str,
    object: &'static str,
    created: i64,
    model: &'a str,
    choices: [ChatChoiceDelta<'a>; 1],
    usage: PartialUsage,
}

#[derive(serde::Serialize)]
struct ChatStreamUsageChunk<'a> {
    id: &'a str,
    object: &'static str,
    created: i64,
    model: &'a str,
    choices: [(); 0],
    usage: &'a crate::models::Usage,
}

#[derive(serde::Serialize)]
struct TextChoiceDelta<'a> {
    index: u32,
    text: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    finish_reason: Option<&'static str>,
}

#[derive(serde::Serialize)]
struct TextStreamChunk<'a> {
    id: &'a str,
    object: &'static str,
    created: i64,
    model: &'a str,
    choices: [TextChoiceDelta<'a>; 1],
}

#[derive(serde::Serialize)]
struct TextStreamContinuousChunk<'a> {
    id: &'a str,
    object: &'static str,
    created: i64,
    model: &'a str,
    choices: [TextChoiceDelta<'a>; 1],
    usage: PartialUsage,
}

#[derive(serde::Serialize)]
struct TextStreamUsageChunk<'a> {
    id: &'a str,
    object: &'static str,
    created: i64,
    model: &'a str,
    choices: [(); 0],
    usage: &'a crate::models::Usage,
}

#[derive(serde::Serialize)]
struct TgiStreamToken<'a> {
    id: usize,
    text: &'a str,
    logprob: f64,
    special: bool,
}

#[derive(serde::Serialize)]
struct TgiStreamChunk<'a> {
    index: usize,
    token: TgiStreamToken<'a>,
    #[serde(skip_serializing_if = "Option::is_none")]
    generated_text: Option<String>,
}

fn sse_done() -> Bytes {
    Bytes::from_static(b"data: [DONE]\n\n")
}

/// A terminal mid-stream SSE error frame: `event: error` with the message as an
/// SSE comment. The runner's SSE reader
/// (`aiperf_runtime::transport::http::sse::reader::read_sse`) classifies any frame whose
/// `event` field equals `error` as a transport `ErrorKind::Sse` (pseudo-status
/// 502, type `sse_error`) via `SseMessage::error_message`, aborting the stream
/// before `[DONE]`. Emitted after a few normal token frames so the record shows
/// partial content plus the error.
fn sse_error_frame(message: &str) -> Bytes {
    Bytes::from(format!("event: error\n: {message}\n\n"))
}

/// Number of normal token frames emitted before a mid-stream error fires, so the
/// captured record carries partial (truncated) content, not zero content.
const MIDSTREAM_TOKENS_BEFORE_ERROR: usize = 3;

fn anthropic_sse_event(event: &str, value: &Value) -> Bytes {
    let data = serde_json::to_string(value).expect("Anthropic event must serialize");
    Bytes::from(format!("event: {event}\ndata: {data}\n\n"))
}

/// A named SSE event frame (`event: <name>\ndata: <json>\n\n`) for the OpenAI
/// Responses streaming shape. The runner keys off the JSON `type` field, so
/// the OpenAI-compatible `event:` line does not affect parsing.
fn sse_event(event: &str, value: &Value) -> Bytes {
    let data = serde_json::to_string(value).expect("SSE event must serialize");
    Bytes::from(format!("event: {event}\ndata: {data}\n\n"))
}

fn sse_response<S>(body: S) -> Response
where
    S: Stream<Item = Result<Bytes, Infallible>> + Send + 'static,
{
    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "text/event-stream")
        .header(header::CACHE_CONTROL, "no-cache")
        .body(Body::from_stream(body))
        .expect("body ok")
}

fn chat_generated_sse(
    ctx: &RequestCtx,
    created: i64,
    choice: ChatChoiceDelta<'_>,
    completion_tokens: usize,
) -> Bytes {
    if ctx.continuous_usage_stats {
        return sse_chunk_ser(&ChatStreamContinuousChunk {
            id: &ctx.request_id,
            object: "chat.completion.chunk",
            created,
            model: &ctx.model,
            choices: [choice],
            usage: PartialUsage::cumulative(ctx, completion_tokens),
        });
    }
    sse_chunk_ser(&ChatStreamChunk {
        id: &ctx.request_id,
        object: "chat.completion.chunk",
        created,
        model: &ctx.model,
        choices: [choice],
    })
}

fn text_generated_sse(
    ctx: &RequestCtx,
    created: i64,
    choice: TextChoiceDelta<'_>,
    completion_tokens: usize,
) -> Bytes {
    if ctx.continuous_usage_stats {
        return sse_chunk_ser(&TextStreamContinuousChunk {
            id: &ctx.request_id,
            object: "text_completion",
            created,
            model: &ctx.model,
            choices: [choice],
            usage: PartialUsage::cumulative(ctx, completion_tokens),
        });
    }
    sse_chunk_ser(&TextStreamChunk {
        id: &ctx.request_id,
        object: "text_completion",
        created,
        model: &ctx.model,
        choices: [choice],
    })
}

fn render_chat_usage_body(ctx: &RequestCtx, include_usage: bool) -> Bytes {
    let created = now_secs();
    let has_reasoning = !ctx.tokenized.reasoning_content_tokens.is_empty();
    let mut buf = Vec::with_capacity(128 * (ctx.tokenized.count() + include_usage as usize + 1));
    let mut completion_tokens = 0;

    for token in &ctx.tokenized.reasoning_content_tokens {
        completion_tokens += 1;
        buf.extend_from_slice(&chat_generated_sse(
            ctx,
            created,
            ChatChoiceDelta {
                index: 0,
                finish_reason: None,
                speculative_decoding_stats: None,
                delta: ChatDelta {
                    role: Some("assistant"),
                    content: None,
                    reasoning_content: Some(token.as_str()),
                    tool_calls: None,
                },
            },
            completion_tokens,
        ));
    }

    let has_tool_call = ctx.tool_call.is_some();
    let num_tokens = ctx.tokenized.tokens.len();
    let first_group_end = ctx.first_chunk_tokens.min(num_tokens);
    let mut group_start = 0;
    while group_start < num_tokens {
        let group_end = if group_start == 0 {
            first_group_end
        } else {
            group_start + 1
        };
        let content = ctx.tokenized.tokens[group_start..group_end].concat();
        completion_tokens += group_end - group_start;
        buf.extend_from_slice(&chat_generated_sse(
            ctx,
            created,
            ChatChoiceDelta {
                index: 0,
                finish_reason: (group_end == num_tokens
                    && !has_tool_call
                    && !ctx.spec_decode_acceptance)
                    .then_some(ctx.tokenized.finish_reason),
                speculative_decoding_stats: None,
                delta: ChatDelta {
                    role: (group_start == 0 && !has_reasoning).then_some("assistant"),
                    content: Some(&content),
                    reasoning_content: None,
                    tool_calls: None,
                },
            },
            completion_tokens,
        ));
        group_start = group_end;
    }

    if let Some(tool_call) = &ctx.tool_call {
        let lead_role = !has_reasoning && num_tokens == 0;
        for chunk in tool_call_frames(
            ctx,
            created,
            tool_call,
            lead_role,
            !ctx.spec_decode_acceptance,
        ) {
            write_sse_into(&mut buf, &chunk);
        }
    }
    if ctx.spec_decode_acceptance {
        write_sse_into(&mut buf, &spec_decode_finish_chunk(ctx, created));
    }
    if include_usage {
        write_sse_into(
            &mut buf,
            &ChatStreamUsageChunk {
                id: &ctx.request_id,
                object: "chat.completion.chunk",
                created,
                model: &ctx.model,
                choices: [],
                usage: &ctx.usage,
            },
        );
    }
    buf.extend_from_slice(b"data: [DONE]\n\n");
    Bytes::from(buf)
}

fn render_text_usage_body(ctx: &RequestCtx, include_usage: bool) -> Bytes {
    let created = now_secs();
    let mut buf = Vec::with_capacity(128 * (ctx.tokenized.tokens.len() + 2));
    let num_tokens = ctx.tokenized.tokens.len();
    let first_group_end = ctx.first_chunk_tokens.min(num_tokens);
    let mut group_start = 0;
    while group_start < num_tokens {
        let group_end = if group_start == 0 {
            first_group_end
        } else {
            group_start + 1
        };
        let content = ctx.tokenized.tokens[group_start..group_end].concat();
        buf.extend_from_slice(&text_generated_sse(
            ctx,
            created,
            TextChoiceDelta {
                index: 0,
                text: &content,
                finish_reason: (group_end == num_tokens).then_some(ctx.tokenized.finish_reason),
            },
            group_end,
        ));
        group_start = group_end;
    }
    if include_usage {
        write_sse_into(
            &mut buf,
            &TextStreamUsageChunk {
                id: &ctx.request_id,
                object: "text_completion",
                created,
                model: &ctx.model,
                choices: [],
                usage: &ctx.usage,
            },
        );
    }
    buf.extend_from_slice(b"data: [DONE]\n\n");
    Bytes::from(buf)
}

/// Renders the complete fast-mode chat stream into one allocation.
fn render_chat_fast_body(ctx: &RequestCtx, include_usage: bool) -> Bytes {
    if ctx.continuous_usage_stats || ctx.first_chunk_tokens > 1 {
        return render_chat_usage_body(ctx, include_usage);
    }
    let created = now_secs();
    let has_reasoning = !ctx.tokenized.reasoning_content_tokens.is_empty();
    let est = 128
        * (ctx.tokenized.reasoning_content_tokens.len()
            + ctx.tokenized.tokens.len()
            + include_usage as usize
            + 1);
    let mut buf: Vec<u8> = Vec::with_capacity(est);

    for token in ctx.tokenized.reasoning_content_tokens.iter() {
        let chunk = ChatStreamChunk {
            id: &ctx.request_id,
            object: "chat.completion.chunk",
            created,
            model: &ctx.model,
            choices: [ChatChoiceDelta {
                index: 0,
                finish_reason: None,
                speculative_decoding_stats: None,
                delta: ChatDelta {
                    role: Some("assistant"),
                    content: None,
                    reasoning_content: Some(token.as_str()),
                    tool_calls: None,
                },
            }],
        };
        write_sse_into(&mut buf, &chunk);
    }

    // Only the final tool-call delta carries the terminal finish reason.
    let has_tool_call = ctx.tool_call.is_some();
    let num = ctx.tokenized.tokens.len();
    for (i, token) in ctx.tokenized.tokens.iter().enumerate() {
        let role = if i == 0 && !has_reasoning {
            Some("assistant")
        } else {
            None
        };
        let finish = if i + 1 == num && !has_tool_call && !ctx.spec_decode_acceptance {
            Some(ctx.tokenized.finish_reason)
        } else {
            None
        };
        let chunk = ChatStreamChunk {
            id: &ctx.request_id,
            object: "chat.completion.chunk",
            created,
            model: &ctx.model,
            choices: [ChatChoiceDelta {
                index: 0,
                finish_reason: finish,
                speculative_decoding_stats: None,
                delta: ChatDelta {
                    role,
                    content: Some(token.as_str()),
                    reasoning_content: None,
                    tool_calls: None,
                },
            }],
        };
        write_sse_into(&mut buf, &chunk);
    }

    if let Some(tc) = &ctx.tool_call {
        let lead_role = !has_reasoning && num == 0;
        for chunk in tool_call_frames(ctx, created, tc, lead_role, !ctx.spec_decode_acceptance) {
            write_sse_into(&mut buf, &chunk);
        }
    }

    if ctx.spec_decode_acceptance {
        write_sse_into(&mut buf, &spec_decode_finish_chunk(ctx, created));
    }

    if include_usage {
        let usage_chunk = ChatStreamUsageChunk {
            id: &ctx.request_id,
            object: "chat.completion.chunk",
            created,
            model: &ctx.model,
            choices: [],
            usage: &ctx.usage,
        };
        write_sse_into(&mut buf, &usage_chunk);
    }

    buf.extend_from_slice(b"data: [DONE]\n\n");
    Bytes::from(buf)
}

fn render_text_fast_body(ctx: &RequestCtx, include_usage: bool) -> Bytes {
    if ctx.continuous_usage_stats || ctx.first_chunk_tokens > 1 {
        return render_text_usage_body(ctx, include_usage);
    }
    let created = now_secs();
    let est = 128 * (ctx.tokenized.tokens.len() + include_usage as usize + 1);
    let mut buf: Vec<u8> = Vec::with_capacity(est);
    let num = ctx.tokenized.tokens.len();
    for (i, token) in ctx.tokenized.tokens.iter().enumerate() {
        let finish = if i + 1 == num {
            Some(ctx.tokenized.finish_reason)
        } else {
            None
        };
        let chunk = TextStreamChunk {
            id: &ctx.request_id,
            object: "text_completion",
            created,
            model: &ctx.model,
            choices: [TextChoiceDelta {
                index: 0,
                text: token.as_str(),
                finish_reason: finish,
            }],
        };
        write_sse_into(&mut buf, &chunk);
    }
    if include_usage {
        let usage_chunk = TextStreamUsageChunk {
            id: &ctx.request_id,
            object: "text_completion",
            created,
            model: &ctx.model,
            choices: [],
            usage: &ctx.usage,
        };
        write_sse_into(&mut buf, &usage_chunk);
    }
    buf.extend_from_slice(b"data: [DONE]\n\n");
    Bytes::from(buf)
}

fn render_tgi_fast_body(ctx: &RequestCtx) -> Bytes {
    let est = 128 * (ctx.tokenized.tokens.len() + 1);
    let mut buf: Vec<u8> = Vec::with_capacity(est);
    let num = ctx.tokenized.tokens.len();
    let content = if num > 0 {
        Some(ctx.tokenized.content())
    } else {
        None
    };
    for (i, token_text) in ctx.tokenized.tokens.iter().enumerate() {
        let generated = if i + 1 == num { content.clone() } else { None };
        let chunk = TgiStreamChunk {
            index: i,
            token: TgiStreamToken {
                id: i,
                text: token_text.as_str(),
                logprob: -0.1,
                special: false,
            },
            generated_text: generated,
        };
        write_sse_into(&mut buf, &chunk);
    }
    Bytes::from(buf)
}

/// Emits two deltas whose argument fragments concatenate to the configured
/// string. The second carries `finish_reason: "tool_calls"` unless a separate
/// finish-only speculative-decoding frame follows it.
fn tool_call_frames<'a>(
    ctx: &'a RequestCtx,
    created: i64,
    tc: &'a ToolCallSpec,
    lead_role: bool,
    emit_finish_reason: bool,
) -> [ChatStreamChunk<'a>; 2] {
    let (args_head, args_tail) = tc.split_arguments();
    let open = ChatStreamChunk {
        id: &ctx.request_id,
        object: "chat.completion.chunk",
        created,
        model: &ctx.model,
        choices: [ChatChoiceDelta {
            index: 0,
            finish_reason: None,
            speculative_decoding_stats: None,
            delta: ChatDelta {
                role: if lead_role { Some("assistant") } else { None },
                content: None,
                reasoning_content: None,
                tool_calls: Some([ToolCallDelta {
                    index: 0,
                    id: Some(&tc.id),
                    kind: Some("function"),
                    function: ToolCallFunctionDelta {
                        name: Some(&tc.name),
                        arguments: args_head,
                    },
                }]),
            },
        }],
    };
    let close = ChatStreamChunk {
        id: &ctx.request_id,
        object: "chat.completion.chunk",
        created,
        model: &ctx.model,
        choices: [ChatChoiceDelta {
            index: 0,
            finish_reason: emit_finish_reason.then_some("tool_calls"),
            speculative_decoding_stats: None,
            delta: ChatDelta {
                role: None,
                content: None,
                reasoning_content: None,
                tool_calls: Some([ToolCallDelta {
                    index: 0,
                    id: None,
                    kind: None,
                    function: ToolCallFunctionDelta {
                        name: None,
                        arguments: args_tail,
                    },
                }]),
            },
        }],
    };
    [open, close]
}

fn spec_decode_finish_chunk(ctx: &RequestCtx, created: i64) -> ChatStreamChunk<'_> {
    ChatStreamChunk {
        id: &ctx.request_id,
        object: "chat.completion.chunk",
        created,
        model: &ctx.model,
        choices: [ChatChoiceDelta {
            index: 0,
            finish_reason: Some(if ctx.tool_call.is_some() {
                "tool_calls"
            } else {
                ctx.tokenized.finish_reason
            }),
            speculative_decoding_stats: Some(spec_decode_acceptance_fixture()),
            delta: ChatDelta {
                role: None,
                content: None,
                reasoning_content: None,
                tool_calls: None,
            },
        }],
    }
}

fn chat_stream(
    state: Arc<AppState>,
    ctx: RequestCtx,
    endpoint: String,
    include_usage: bool,
    midstream_error: bool,
) -> impl Stream<Item = Result<Bytes, Infallible>> {
    let labeled = state.recorder.labeled(&endpoint, &ctx.model);
    async_stream::stream! {
        state.recorder.record_request_start(&endpoint, &ctx.model);
        state.recorder.record_llm_inflight_start(&ctx.model);

        // Mid-stream failures close after `event: error`, without usage or
        // `[DONE]`, after emitting up to three token frames.
        if midstream_error {
            let created = now_secs();
            let has_reasoning = !ctx.tokenized.reasoning_content_tokens.is_empty();
            let num = ctx.tokenized.tokens.len();
            let emit = num.min(MIDSTREAM_TOKENS_BEFORE_ERROR);
            for (i, token) in ctx.tokenized.tokens.iter().take(emit).enumerate() {
                if !ctx.latency_sim.is_fast() {
                    let _ = ctx.latency_sim.wait_for_index(i).await;
                }
                let role = if i == 0 && !has_reasoning { Some("assistant") } else { None };
                let chunk = ChatStreamChunk {
                    id: &ctx.request_id,
                    object: "chat.completion.chunk",
                    created,
                    model: &ctx.model,
                    choices: [ChatChoiceDelta {
                        index: 0,
                        finish_reason: None,
                        speculative_decoding_stats: None,
                        delta: ChatDelta {
                            role,
                            content: Some(token.as_str()),
                            reasoning_content: None,
                            tool_calls: None,
                        },
                    }],
                };
                yield Ok::<Bytes, Infallible>(sse_chunk_ser(&chunk));
            }
            yield Ok::<Bytes, Infallible>(sse_error_frame(
                "Simulated mid-stream error injected by aiperf-mock-server",
            ));
            state.recorder.record_error(&endpoint, "midstream_sse_error");
            state.recorder.record_llm_inflight_end(&ctx.model);
            state.recorder.record_request_end(&endpoint);
            return;
        }

        // Null-object injection cannot use the pre-rendered body because its
        // frame must precede `[DONE]`.
        if ctx.latency_sim.is_fast() && !ctx.null_object_chunk {
            let total_tokens = ctx.tokenized.reasoning_content_tokens.len()
                + ctx.tokenized.tokens.len();
            if total_tokens > 0 {
                state.recorder.record_zero_ttft_and_itls(&labeled, total_tokens - 1);
                state
                    .recorder
                    .record_streamed_tokens_fast(&labeled, total_tokens as u64);
            }
            let body = render_chat_fast_body(&ctx, include_usage);
            yield Ok::<Bytes, Infallible>(body);

            let latency = ctx.start.elapsed();
            let info = LLMLatencyInfo {
                e2e: latency,
                prefill: std::time::Duration::ZERO,
                decode: latency,
            };
            state.recorder.record_llm_success(
                &endpoint, &ctx.model, latency.as_secs_f64(), &ctx.usage, &info,
            );
            state.recorder.record_llm_inflight_end(&ctx.model);
            state.recorder.record_request_end(&endpoint);
            return;
        }

        let has_reasoning = !ctx.tokenized.reasoning_content_tokens.is_empty();

        // OpenAI requires a constant `created` value across one stream.
        let created = now_secs();

        let mut idx = 0usize;
        let mut first_emit: Option<Instant> = None;
        let mut last_emit: Option<Instant> = None;

        for token in ctx.tokenized.reasoning_content_tokens.iter() {
            let emit_at = ctx.latency_sim.wait_for_index(idx).await;
            if first_emit.is_none() {
                first_emit = Some(emit_at);
                let ttft = emit_at.duration_since(ctx.start);
                state.recorder.record_ttft_fast(&labeled, ttft.as_secs_f64());
            } else if let Some(last) = last_emit {
                let itl = emit_at.duration_since(last);
                state.recorder.record_itl_fast(&labeled, itl.as_secs_f64());
            }
            last_emit = Some(emit_at);
            idx += 1;
            state.recorder.record_streamed_token_fast(&labeled);
            yield Ok::<Bytes, Infallible>(chat_generated_sse(
                &ctx,
                created,
                ChatChoiceDelta {
                    index: 0,
                    finish_reason: None,
                    speculative_decoding_stats: None,
                    delta: ChatDelta {
                        role: Some("assistant"),
                        content: None,
                        reasoning_content: Some(token.as_str()),
                        tool_calls: None,
                    },
                },
                idx,
            ));
        }

        // A tool-call turn carries its finish reason on the final call delta.
        let has_tool_call = ctx.tool_call.is_some();
        let num = ctx.tokenized.tokens.len();
        if ctx.continuous_usage_stats || ctx.first_chunk_tokens > 1 {
            let mut group_start = 0;
            let first_group_end = ctx.first_chunk_tokens.min(num);
            while group_start < num {
                let group_end = if group_start == 0 {
                    first_group_end
                } else {
                    group_start + 1
                };
                for _ in group_start..group_end {
                    let emit_at = ctx.latency_sim.wait_for_index(idx).await;
                    if first_emit.is_none() {
                        first_emit = Some(emit_at);
                        let ttft = emit_at.duration_since(ctx.start);
                        state.recorder.record_ttft_fast(&labeled, ttft.as_secs_f64());
                    } else if let Some(last) = last_emit {
                        let itl = emit_at.duration_since(last);
                        state.recorder.record_itl_fast(&labeled, itl.as_secs_f64());
                    }
                    last_emit = Some(emit_at);
                    idx += 1;
                    state.recorder.record_streamed_token_fast(&labeled);
                }
                let content = ctx.tokenized.tokens[group_start..group_end].concat();
                yield Ok::<Bytes, Infallible>(chat_generated_sse(
                    &ctx,
                    created,
                    ChatChoiceDelta {
                        index: 0,
                        finish_reason: (group_end == num
                            && !has_tool_call
                            && !ctx.spec_decode_acceptance)
                            .then_some(ctx.tokenized.finish_reason),
                        speculative_decoding_stats: None,
                        delta: ChatDelta {
                            role: (group_start == 0 && !has_reasoning).then_some("assistant"),
                            content: Some(&content),
                            reasoning_content: None,
                            tool_calls: None,
                        },
                    },
                    idx,
                ));
                group_start = group_end;
            }
        } else {
            // Middle-token frames share a byte-identical envelope. Boundary frames
            // use full serialization because they carry role or finish metadata.
            let mid_prefix: Vec<u8> = {
                let mut p = Vec::with_capacity(96);
                p.extend_from_slice(b"data: {\"id\":");
                serde_json::to_writer(&mut p, &ctx.request_id).expect("serialize id");
                p.extend_from_slice(b",\"object\":\"chat.completion.chunk\",\"created\":");
                p.extend_from_slice(created.to_string().as_bytes());
                p.extend_from_slice(b",\"model\":");
                serde_json::to_writer(&mut p, &ctx.model).expect("serialize model");
                p.extend_from_slice(b",\"choices\":[{\"index\":0,\"delta\":{\"content\":");
                p
            };
            for (i, token) in ctx.tokenized.tokens.iter().enumerate() {
                let emit_at = ctx.latency_sim.wait_for_index(idx).await;
                if first_emit.is_none() {
                    first_emit = Some(emit_at);
                    let ttft = emit_at.duration_since(ctx.start);
                    state.recorder.record_ttft_fast(&labeled, ttft.as_secs_f64());
                } else if let Some(last) = last_emit {
                    let itl = emit_at.duration_since(last);
                    state.recorder.record_itl_fast(&labeled, itl.as_secs_f64());
                }
                last_emit = Some(emit_at);
                idx += 1;
                state.recorder.record_streamed_token_fast(&labeled);
                let role = if i == 0 && !has_reasoning { Some("assistant") } else { None };
                let finish = if i + 1 == num && !has_tool_call && !ctx.spec_decode_acceptance {
                    Some(ctx.tokenized.finish_reason)
                } else {
                    None
                };
                if role.is_none() && finish.is_none() {
                    let mut out = Vec::with_capacity(mid_prefix.len() + token.len() + 8);
                    out.extend_from_slice(&mid_prefix);
                    serde_json::to_writer(&mut out, token.as_str()).expect("serialize token");
                    out.extend_from_slice(b"}}]}\n\n");
                    yield Ok::<Bytes, Infallible>(Bytes::from(out));
                } else {
                    let chunk = ChatStreamChunk {
                        id: &ctx.request_id,
                        object: "chat.completion.chunk",
                        created,
                        model: &ctx.model,
                        choices: [ChatChoiceDelta {
                            index: 0,
                            finish_reason: finish,
                            speculative_decoding_stats: None,
                            delta: ChatDelta {
                                role,
                                content: Some(token.as_str()),
                                reasoning_content: None,
                                tool_calls: None,
                            },
                        }],
                    };
                    yield Ok::<Bytes, Infallible>(sse_chunk_ser(&chunk));
                }
            }
        }

        if let Some(tc) = &ctx.tool_call {
            let lead_role = !has_reasoning && num == 0;
            for chunk in tool_call_frames(
                &ctx,
                created,
                tc,
                lead_role,
                !ctx.spec_decode_acceptance,
            ) {
                yield Ok::<Bytes, Infallible>(sse_chunk_ser(&chunk));
            }
        }

        if ctx.spec_decode_acceptance {
            yield Ok::<Bytes, Infallible>(sse_chunk_ser(&spec_decode_finish_chunk(
                &ctx, created,
            )));
        }

        if include_usage {
            let usage_chunk = ChatStreamUsageChunk {
                id: &ctx.request_id,
                object: "chat.completion.chunk",
                created,
                model: &ctx.model,
                choices: [],
                usage: &ctx.usage,
            };
            yield Ok::<Bytes, Infallible>(sse_chunk_ser(&usage_chunk));
        }

        if ctx.null_object_chunk {
            // The standalone `object: null` frame must precede `[DONE]`.
            yield Ok::<Bytes, Infallible>(Bytes::from_static(
                b"data: {\"id\":\"adversarial-null\",\"object\":null,\"created\":0,\"choices\":[]}\n\n",
            ));
        }

        yield Ok::<Bytes, Infallible>(sse_done());

        let latency = ctx.start.elapsed();
        let prefill = first_emit
            .map(|t| t.duration_since(ctx.start))
            .unwrap_or(std::time::Duration::ZERO);
        let info = LLMLatencyInfo {
            e2e: latency,
            prefill,
            decode: latency.saturating_sub(prefill),
        };
        state.recorder.record_llm_success(
            &endpoint, &ctx.model, latency.as_secs_f64(), &ctx.usage, &info,
        );
        state.recorder.record_llm_inflight_end(&ctx.model);
        state.recorder.record_request_end(&endpoint);
    }
}

fn messages_stream(
    state: Arc<AppState>,
    ctx: RequestCtx,
    endpoint: String,
) -> impl Stream<Item = Result<Bytes, Infallible>> {
    let labeled = state.recorder.labeled(&endpoint, &ctx.model);
    async_stream::stream! {
        state.recorder.record_request_start(&endpoint, &ctx.model);
        state.recorder.record_llm_inflight_start(&ctx.model);

        let start_event = json!({
            "type": "message_start",
            "message": {
                "id": ctx.request_id,
                "type": "message",
                "role": "assistant",
                "model": ctx.model,
                "content": [],
                "stop_reason": Value::Null,
                "stop_sequence": Value::Null,
                "usage": {
                    "input_tokens": ctx.usage.prompt_tokens,
                    "output_tokens": 0,
                },
            },
        });
        yield Ok::<Bytes, Infallible>(anthropic_sse_event("message_start", &start_event));

        let block_start = json!({
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        });
        yield Ok::<Bytes, Infallible>(anthropic_sse_event("content_block_start", &block_start));

        let mut first_emit: Option<Instant> = None;
        let mut last_emit: Option<Instant> = None;
        let count = ctx.tokenized.tokens.len();
        if ctx.latency_sim.is_fast() {
            if count > 0 {
                state.recorder.record_zero_ttft_and_itls(&labeled, count - 1);
                state.recorder.record_streamed_tokens_fast(&labeled, count as u64);
            }
            for token in &ctx.tokenized.tokens {
                let event = json!({
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": token},
                });
                yield Ok::<Bytes, Infallible>(anthropic_sse_event("content_block_delta", &event));
            }
        } else {
            for (index, token) in ctx.tokenized.tokens.iter().enumerate() {
                let emit_at = ctx.latency_sim.wait_for_index(index).await;
                if first_emit.is_none() {
                    first_emit = Some(emit_at);
                    state.recorder.record_ttft_fast(
                        &labeled,
                        emit_at.duration_since(ctx.start).as_secs_f64(),
                    );
                } else if let Some(last) = last_emit {
                    state.recorder.record_itl_fast(
                        &labeled,
                        emit_at.duration_since(last).as_secs_f64(),
                    );
                }
                last_emit = Some(emit_at);
                state.recorder.record_streamed_token_fast(&labeled);
                let event = json!({
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": token},
                });
                yield Ok::<Bytes, Infallible>(anthropic_sse_event("content_block_delta", &event));
            }
        }

        yield Ok::<Bytes, Infallible>(anthropic_sse_event(
            "content_block_stop",
            &json!({"type": "content_block_stop", "index": 0}),
        ));
        yield Ok::<Bytes, Infallible>(anthropic_sse_event(
            "message_delta",
            &json!({
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": Value::Null},
                "usage": {"output_tokens": ctx.usage.completion_tokens},
            }),
        ));
        yield Ok::<Bytes, Infallible>(anthropic_sse_event(
            "message_stop",
            &json!({"type": "message_stop"}),
        ));

        let latency = ctx.start.elapsed();
        let prefill = first_emit
            .map(|at| at.duration_since(ctx.start))
            .unwrap_or(std::time::Duration::ZERO);
        let info = LLMLatencyInfo {
            e2e: latency,
            prefill,
            decode: latency.saturating_sub(prefill),
        };
        state.recorder.record_llm_success(
            &endpoint,
            &ctx.model,
            latency.as_secs_f64(),
            &ctx.usage,
            &info,
        );
        state.recorder.record_llm_inflight_end(&ctx.model);
        state.recorder.record_request_end(&endpoint);
    }
}

fn text_stream(
    state: Arc<AppState>,
    ctx: RequestCtx,
    endpoint: String,
    include_usage: bool,
) -> impl Stream<Item = Result<Bytes, Infallible>> {
    let labeled = state.recorder.labeled(&endpoint, &ctx.model);
    async_stream::stream! {
        state.recorder.record_request_start(&endpoint, &ctx.model);
        state.recorder.record_llm_inflight_start(&ctx.model);

        if ctx.latency_sim.is_fast() {
            let n = ctx.tokenized.tokens.len();
            if n > 0 {
                state.recorder.record_zero_ttft_and_itls(&labeled, n - 1);
                state.recorder.record_streamed_tokens_fast(&labeled, n as u64);
            }
            let body = render_text_fast_body(&ctx, include_usage);
            yield Ok::<Bytes, Infallible>(body);

            let latency = ctx.start.elapsed();
            let info = LLMLatencyInfo {
                e2e: latency,
                prefill: std::time::Duration::ZERO,
                decode: latency,
            };
            state.recorder.record_llm_success(
                &endpoint, &ctx.model, latency.as_secs_f64(), &ctx.usage, &info,
            );
            state.recorder.record_llm_inflight_end(&ctx.model);
            state.recorder.record_request_end(&endpoint);
            return;
        }

        let num = ctx.tokenized.tokens.len();
        // OpenAI requires a constant `created` value across one stream.
        let created = now_secs();
        let mut first_emit: Option<Instant> = None;
        let mut last_emit: Option<Instant> = None;
        if ctx.continuous_usage_stats || ctx.first_chunk_tokens > 1 {
            let mut group_start = 0;
            let first_group_end = ctx.first_chunk_tokens.min(num);
            while group_start < num {
                let group_end = if group_start == 0 {
                    first_group_end
                } else {
                    group_start + 1
                };
                for index in group_start..group_end {
                    let emit_at = ctx.latency_sim.wait_for_index(index).await;
                    if first_emit.is_none() {
                        first_emit = Some(emit_at);
                        let ttft = emit_at.duration_since(ctx.start);
                        state.recorder.record_ttft_fast(&labeled, ttft.as_secs_f64());
                    } else if let Some(last) = last_emit {
                        let itl = emit_at.duration_since(last);
                        state.recorder.record_itl_fast(&labeled, itl.as_secs_f64());
                    }
                    last_emit = Some(emit_at);
                    state.recorder.record_streamed_token_fast(&labeled);
                }
                let content = ctx.tokenized.tokens[group_start..group_end].concat();
                yield Ok::<Bytes, Infallible>(text_generated_sse(
                    &ctx,
                    created,
                    TextChoiceDelta {
                        index: 0,
                        text: &content,
                        finish_reason: (group_end == num).then_some(ctx.tokenized.finish_reason),
                    },
                    group_end,
                ));
                group_start = group_end;
            }
        } else {
            for (i, token) in ctx.tokenized.tokens.iter().enumerate() {
                let emit_at = ctx.latency_sim.wait_for_index(i).await;
                if first_emit.is_none() {
                    first_emit = Some(emit_at);
                    let ttft = emit_at.duration_since(ctx.start);
                    state.recorder.record_ttft_fast(&labeled, ttft.as_secs_f64());
                } else if let Some(last) = last_emit {
                    let itl = emit_at.duration_since(last);
                    state.recorder.record_itl_fast(&labeled, itl.as_secs_f64());
                }
                last_emit = Some(emit_at);
                state.recorder.record_streamed_token_fast(&labeled);
                let finish = if i + 1 == num { Some(ctx.tokenized.finish_reason) } else { None };
                let chunk = TextStreamChunk {
                    id: &ctx.request_id,
                    object: "text_completion",
                    created,
                    model: &ctx.model,
                    choices: [TextChoiceDelta {
                        index: 0,
                        text: token.as_str(),
                        finish_reason: finish,
                    }],
                };
                yield Ok::<Bytes, Infallible>(sse_chunk_ser(&chunk));
            }
        }
        if include_usage {
            let usage_chunk = TextStreamUsageChunk {
                id: &ctx.request_id,
                object: "text_completion",
                created,
                model: &ctx.model,
                choices: [],
                usage: &ctx.usage,
            };
            yield Ok::<Bytes, Infallible>(sse_chunk_ser(&usage_chunk));
        }
        yield Ok::<Bytes, Infallible>(sse_done());

        let latency = ctx.start.elapsed();
        let prefill = first_emit
            .map(|t| t.duration_since(ctx.start))
            .unwrap_or(std::time::Duration::ZERO);
        let info = LLMLatencyInfo {
            e2e: latency,
            prefill,
            decode: latency.saturating_sub(prefill),
        };
        state.recorder.record_llm_success(
            &endpoint, &ctx.model, latency.as_secs_f64(), &ctx.usage, &info,
        );
        state.recorder.record_llm_inflight_end(&ctx.model);
        state.recorder.record_request_end(&endpoint);
    }
}

fn tgi_stream(
    state: Arc<AppState>,
    ctx: RequestCtx,
    endpoint: String,
) -> impl Stream<Item = Result<Bytes, Infallible>> {
    let labeled = state.recorder.labeled(&endpoint, &ctx.model);
    async_stream::stream! {
        state.recorder.record_request_start(&endpoint, &ctx.model);

        if ctx.latency_sim.is_fast() {
            let n = ctx.tokenized.tokens.len();
            if n > 0 {
                state.recorder.record_zero_ttft_and_itls(&labeled, n - 1);
                state.recorder.record_streamed_tokens_fast(&labeled, n as u64);
            }
            let body = render_tgi_fast_body(&ctx);
            yield Ok::<Bytes, Infallible>(body);
            let latency = ctx.start.elapsed();
            state.recorder.record_tgi_success(&endpoint, &ctx.usage, latency.as_secs_f64());
            state.recorder.record_request_end(&endpoint);
            return;
        }

        let num = ctx.tokenized.tokens.len();
        let mut last_emit: Option<Instant> = None;
        for (i, token_text) in ctx.tokenized.tokens.iter().enumerate() {
            let emit_at = ctx.latency_sim.wait_for_index(i).await;
            if i == 0 {
                let ttft = emit_at.duration_since(ctx.start);
                state.recorder.record_ttft_fast(&labeled, ttft.as_secs_f64());
            } else if let Some(last) = last_emit {
                let itl = emit_at.duration_since(last);
                state.recorder.record_itl_fast(&labeled, itl.as_secs_f64());
            }
            last_emit = Some(emit_at);
            state.recorder.record_streamed_token_fast(&labeled);
            let generated = if i + 1 == num { Some(ctx.tokenized.content()) } else { None };
            let chunk = TgiStreamChunk {
                index: i,
                token: TgiStreamToken {
                    id: i,
                    text: token_text.as_str(),
                    logprob: -0.1,
                    special: false,
                },
                generated_text: generated,
            };
            yield Ok::<Bytes, Infallible>(sse_chunk_ser(&chunk));
        }
        let latency = ctx.start.elapsed();
        state.recorder.record_tgi_success(&endpoint, &ctx.usage, latency.as_secs_f64());
        state.recorder.record_request_end(&endpoint);
    }
}

fn image_stream(
    state: Arc<AppState>,
    ctx: RequestCtx,
    req: ImageGenerationRequest,
    endpoint: String,
) -> impl Stream<Item = Result<Bytes, Infallible>> {
    async_stream::stream! {
        state.recorder.record_request_start(&endpoint, &req.model);
        state.recorder.record_llm_inflight_start(&req.model);
        let per_image = (ctx.tokenized.count() / req.n.max(1) as usize).max(1);
        let mut first_emit: Option<Instant> = None;
        for i in 0..req.n {
            let (prefill_i, _decode_i) = ctx.latency_sim.wait_for_tokens(per_image).await;
            if first_emit.is_none() {
                first_emit = Some(ctx.start + prefill_i);
            }
            let mut chunk = json!({
                "b64_json": mock_jpeg_b64(&req.prompt, i),
                "partial_image_index": i,
            });
            if let Some(size) = &req.size {
                chunk["size"] = Value::String(size.clone());
            }
            if let Some(quality) = &req.quality {
                chunk["quality"] = Value::String(quality.clone());
            }
            yield Ok::<Bytes, Infallible>(sse_chunk(&chunk));
        }
        yield Ok::<Bytes, Infallible>(sse_done());

        let latency = ctx.start.elapsed();
        let prefill = first_emit
            .map(|t| t.duration_since(ctx.start))
            .unwrap_or(std::time::Duration::ZERO);
        let info = LLMLatencyInfo {
            e2e: latency,
            prefill,
            decode: latency.saturating_sub(prefill),
        };
        state.recorder.record_llm_success(
            &endpoint, &req.model, latency.as_secs_f64(), &ctx.usage, &info,
        );
        state.recorder.record_llm_inflight_end(&req.model);
        state.recorder.record_request_end(&endpoint);
    }
}

/// Render a Prometheus exposition body, honoring `--openmetrics`: when set, the
/// body is converted to OpenMetrics text (with `# EOF` and suffix-less counter
/// families) and served with the OpenMetrics content-type, matching the vLLM
/// Rust frontend; otherwise classic `text/plain; version=0.0.4`.
fn metrics_response(state: &AppState, mut body: Vec<u8>) -> Response {
    let content_type = if state.config.openmetrics {
        crate::prom::to_openmetrics(&mut body);
        crate::prom::OPENMETRICS_CONTENT_TYPE
    } else {
        "text/plain; version=0.0.4"
    };
    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, content_type)
        .body(Body::from(body))
        .expect("response")
}

pub async fn reset_prefix_cache(State(state): State<Arc<AppState>>) -> StatusCode {
    state.reset_prefix_cache();
    StatusCode::OK
}

pub async fn start_profile(State(state): State<Arc<AppState>>) -> StatusCode {
    state.profiler_state().note_start();
    StatusCode::OK
}

pub async fn stop_profile(State(state): State<Arc<AppState>>) -> StatusCode {
    state.profiler_state().note_stop();
    StatusCode::OK
}

pub async fn aiperf_mock_metrics(State(state): State<Arc<AppState>>) -> Response {
    state
        .recorder
        .metrics
        .aiperf
        .SERVER_UPTIME_SECONDS
        .set(state.uptime_secs());
    let mut body = crate::prom::encode(&state.recorder.metrics.aiperf.registry);
    // Accuracy metrics are not registered collectors, so append them to the
    // Prometheus exposition after encoding the registry.
    if state.accuracy.is_some() {
        crate::prom::append_accuracy_metrics(&mut body, &state.accuracy_live.snapshot());
    }
    metrics_response(&state, body)
}

/// `GET /accuracy` — the live accuracy tally for the current run: how many
/// prompt-matched requests the mock has actually answered, and how many of
/// those correctly (`correct / matched`). Returns `{"enabled": false}` when the
/// accuracy dataset mode is off.
pub async fn accuracy_status(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    match &state.accuracy {
        None => Json(json!({ "enabled": false })),
        Some(ds) => {
            let snap = state.accuracy_live.snapshot();
            Json(json!({
                "enabled": true,
                "config": {
                    "format": state.config.accuracy_format,
                    "correct_rate": state.config.accuracy_correct_rate,
                    "cot_rate": state.config.accuracy_cot_rate,
                    "adversarial_rate": state.config.accuracy_adversarial_rate,
                    "reasoning_field": state.config.accuracy_reasoning_field,
                    "dataset_rows": ds.len(),
                },
                "matched": snap.matched,
                "correct": snap.correct,
                "incorrect": snap.incorrect,
                "accuracy": snap.accuracy,
                "unmatched": snap.unmatched,
                "adversarial": snap.adversarial,
                "cot": snap.cot,
                "tasks": snap.tasks,
            }))
        }
    }
}

pub async fn vllm_metrics(State(state): State<Arc<AppState>>) -> Response {
    let body = crate::prom::encode(&state.recorder.metrics.vllm.registry);
    metrics_response(&state, body)
}

pub async fn sglang_metrics(State(state): State<Arc<AppState>>) -> Response {
    let body = crate::prom::encode(&state.recorder.metrics.sglang.registry);
    metrics_response(&state, body)
}

pub async fn trtllm_metrics(State(state): State<Arc<AppState>>) -> Response {
    let body = crate::prom::encode(&state.recorder.metrics.trtllm.registry);
    metrics_response(&state, body)
}

pub async fn dynamo_frontend_metrics(State(state): State<Arc<AppState>>) -> Response {
    let body = crate::prom::encode(&state.recorder.metrics.dynamo_frontend.registry);
    metrics_response(&state, body)
}

pub async fn dynamo_prefill_metrics(State(state): State<Arc<AppState>>) -> Response {
    let body = crate::prom::encode(&state.recorder.metrics.dynamo_prefill.registry);
    metrics_response(&state, body)
}

pub async fn dynamo_decode_metrics(State(state): State<Arc<AppState>>) -> Response {
    let body = crate::prom::encode(&state.recorder.metrics.dynamo_decode.registry);
    metrics_response(&state, body)
}

fn dcgm_response(state: &AppState, idx: usize) -> Result<Response, AppError> {
    let faker = state.dcgm.get(idx).ok_or(AppError {
        status: StatusCode::NOT_FOUND,
        message: "Invalid DCGM instance".to_string(),
        retry_after: None,
    })?;
    let body = faker.generate();
    Ok(Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "text/plain")
        .body(Body::from(body))
        .expect("resp"))
}

pub async fn dcgm_metrics_1(State(state): State<Arc<AppState>>) -> AppResult<Response> {
    dcgm_response(&state, 0)
}

pub async fn dcgm_metrics_2(State(state): State<Arc<AppState>>) -> AppResult<Response> {
    dcgm_response(&state, 1)
}

/// Accepts OpenAI-compatible multipart image edits and returns synthetic JPEGs.
pub async fn image_edit(
    State(state): State<Arc<AppState>>,
    mut multipart: axum::extract::Multipart,
) -> AppResult<Response> {
    if let Some(e) = maybe_inject_error(&state) {
        return Err(e);
    }
    let endpoint = "/v1/images/edits";
    let start = Instant::now();

    let mut prompt = String::from("edit");
    let mut model = String::from("mock-model");
    let mut n: u32 = 1;

    while let Ok(Some(field)) = multipart.next_field().await {
        match field.name() {
            Some("prompt") => {
                if let Ok(v) = field.text().await {
                    prompt = v;
                }
            }
            Some("model") => {
                if let Ok(v) = field.text().await {
                    model = v;
                }
            }
            Some("n") => {
                if let Ok(v) = field.text().await {
                    n = v.parse().unwrap_or(1);
                }
            }
            _ => {
                // Multipart fields must be consumed before advancing.
                let _ = field.bytes().await;
            }
        }
    }

    let mock_chat = ChatCompletionRequest {
        model: model.clone(),
        messages: vec![Message {
            role: "user".into(),
            content: Value::String(prompt.clone()),
        }],
        stream: false,
        stream_options: None,
        max_tokens: None,
        max_completion_tokens: None,
        ignore_eos: false,
        min_tokens: None,
        reasoning_effort: None,
        priority: None,
        mock_first_chunk_tokens: 1,
    };
    let req_gen = GenRequest::Chat(&mock_chat);
    let ctx = RequestCtx::build("img", &req_gen, endpoint, start, &state);

    state.recorder.record_request_start(endpoint, &model);
    state.recorder.record_llm_inflight_start(&model);
    let (prefill, _decode) = ctx.latency_sim.wait_for_tokens(ctx.tokenized.count()).await;
    let latency = start.elapsed();
    let info = LLMLatencyInfo {
        e2e: latency,
        prefill,
        decode: latency.saturating_sub(prefill),
    };

    let mut data: Vec<Value> = Vec::with_capacity(n as usize);
    for i in 0..n {
        data.push(json!({ "b64_json": mock_jpeg_b64(&prompt, i) }));
    }
    let mut body = json!({
        "created": now_secs(),
        "data": data,
    });
    body["usage"] = serde_json::to_value(&ctx.usage).unwrap();

    state
        .recorder
        .record_llm_success(endpoint, &model, latency.as_secs_f64(), &ctx.usage, &info);
    state.recorder.record_llm_inflight_end(&model);
    state.recorder.record_request_end(endpoint);

    Ok(Json(body).into_response())
}

/// Accept a transcription multipart request and return deterministic text.
pub async fn audio_transcription(
    State(state): State<Arc<AppState>>,
    mut multipart: axum::extract::Multipart,
) -> AppResult<Response> {
    if let Some(e) = maybe_inject_error(&state) {
        return Err(e);
    }
    let mut has_file = false;
    let mut language: Option<String> = None;
    let mut temperature: Option<f64> = None;
    loop {
        let Some(field) = multipart
            .next_field()
            .await
            .map_err(invalid_audio_multipart)?
        else {
            break;
        };
        match field.name() {
            Some("file") => {
                has_file = !field
                    .bytes()
                    .await
                    .map_err(invalid_audio_multipart)?
                    .is_empty();
            }
            Some("language") => {
                language = Some(field.text().await.map_err(invalid_audio_multipart)?);
            }
            Some("temperature") => {
                let value = field.text().await.map_err(invalid_audio_multipart)?;
                temperature = value.parse::<f64>().ok();
            }
            _ => {
                field.bytes().await.map_err(invalid_audio_multipart)?;
            }
        }
    }
    if !has_file {
        return Err(AppError {
            status: StatusCode::BAD_REQUEST,
            message: "file is required".into(),
            retry_after: None,
        });
    }
    let mut body = json!({
        "text": "mock transcription",
        "usage": {"input_tokens": 1}
    });
    if let Some(language) = language {
        body["language"] = Value::String(language);
    }
    if let Some(temperature) = temperature {
        body["temperature"] = json!(temperature);
    }
    Ok(Json(body).into_response())
}

fn invalid_audio_multipart(error: impl std::fmt::Display) -> AppError {
    AppError {
        status: StatusCode::BAD_REQUEST,
        message: format!("invalid audio multipart: {error}"),
        retry_after: None,
    }
}

#[cfg(test)]
mod content_url_tests {
    use super::*;

    fn user(content: Value) -> Message {
        Message {
            role: "user".into(),
            content,
        }
    }

    #[test]
    fn is_fetchable_url_only_accepts_http_schemes() {
        assert!(is_fetchable_url(
            "http://host:8090/content/images/img_1.png"
        ));
        assert!(is_fetchable_url("https://host/img.jpg"));
        assert!(!is_fetchable_url("data:image/png;base64,AAAA"));
        assert!(!is_fetchable_url("file:///tmp/img.png"));
        assert!(!is_fetchable_url(""));
    }

    #[test]
    fn collect_extracts_object_and_string_image_and_video_urls() {
        let messages = vec![user(json!([
            {"type": "text", "text": "describe"},
            {"type": "image_url", "image_url": {"url": "http://cs:8090/content/images/img_1.png"}},
            {"type": "video_url", "video_url": {"url": "https://cs/vid_1.mp4"}},
            {"type": "image_url", "image_url": "http://cs:8090/content/images/img_2.png"},
        ]))];
        let urls = collect_content_urls(&messages);
        assert_eq!(
            urls,
            vec![
                "http://cs:8090/content/images/img_1.png",
                "https://cs/vid_1.mp4",
                "http://cs:8090/content/images/img_2.png",
            ]
        );
    }

    #[test]
    fn collect_skips_data_uris_and_plain_string_content() {
        let messages = vec![
            user(Value::String("just text, no parts".into())),
            user(json!([
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
                {"type": "text", "text": "hi"},
            ])),
        ];
        assert!(collect_content_urls(&messages).is_empty());
    }
}

#[cfg(test)]
mod sagemaker_tests {
    use super::*;

    #[test]
    fn sniffs_openai_chat_shaped_body() {
        let body = json!({
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 16,
        });
        let req = sagemaker_request_to_chat("my-endpoint", &body, false).unwrap();
        assert_eq!(req.model, "my-endpoint");
        assert_eq!(req.messages.len(), 1);
        assert_eq!(req.messages[0].role, "user");
        assert_eq!(req.max_tokens, Some(16));
        assert!(!req.stream);
    }

    #[test]
    fn endpoint_name_overrides_body_model_field() {
        let body = json!({
            "model": "ignored-model",
            "messages": [{"role": "user", "content": "hi"}],
        });
        let req = sagemaker_request_to_chat("my-endpoint", &body, true).unwrap();
        assert_eq!(req.model, "my-endpoint");
        assert!(req.stream);
    }

    #[test]
    fn sniffs_jumpstart_djl_shaped_body() {
        let body = json!({
            "inputs": "tell me a story",
            "parameters": {"max_new_tokens": 32, "min_new_tokens": 4},
        });
        let req = sagemaker_request_to_chat("my-endpoint", &body, false).unwrap();
        assert_eq!(req.model, "my-endpoint");
        assert_eq!(req.messages.len(), 1);
        assert_eq!(req.messages[0].role, "user");
        assert_eq!(
            req.messages[0].content,
            Value::String("tell me a story".to_string())
        );
        assert_eq!(req.max_tokens, Some(32));
        assert_eq!(req.min_tokens, Some(4));
    }

    #[test]
    fn jumpstart_shape_without_parameters_has_no_token_caps() {
        let body = json!({"inputs": "hello"});
        let req = sagemaker_request_to_chat("my-endpoint", &body, false).unwrap();
        assert_eq!(req.max_tokens, None);
        assert_eq!(req.min_tokens, None);
    }

    #[test]
    fn rejects_body_missing_both_shapes() {
        let body = json!({"foo": "bar"});
        let err = sagemaker_request_to_chat("my-endpoint", &body, false).unwrap_err();
        assert_eq!(err.status, StatusCode::BAD_REQUEST);
    }

    #[test]
    fn sse_to_eventstream_drops_done_sentinel_and_wraps_frames() {
        let sse = futures::stream::iter(vec![
            Ok::<Bytes, Infallible>(Bytes::from_static(
                b"data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n",
            )),
            Ok::<Bytes, Infallible>(sse_done()),
        ]);
        let out: Vec<Bytes> = futures::executor::block_on(futures::StreamExt::collect::<Vec<_>>(
            sse_to_eventstream(sse),
        ))
        .into_iter()
        .map(|r| r.unwrap())
        .collect();

        assert_eq!(out.len(), 1);
        let mut decoder = aiperf_runtime::transport::core::EventStreamDecoder::new();
        decoder.push(&out[0]);
        let messages = decoder.drain_messages().unwrap();
        assert_eq!(messages.len(), 1);
        assert_eq!(
            &messages[0].payload[..],
            b"{\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}"
        );
    }
}

#[cfg(test)]
mod stream_frame_tests {
    use super::*;

    fn full_serde(id: &str, model: &str, created: i64, token: &str) -> Vec<u8> {
        let chunk = ChatStreamChunk {
            id,
            object: "chat.completion.chunk",
            created,
            model,
            choices: [ChatChoiceDelta {
                index: 0,
                finish_reason: None,
                speculative_decoding_stats: None,
                delta: ChatDelta {
                    role: None,
                    content: Some(token),
                    reasoning_content: None,
                    tool_calls: None,
                },
            }],
        };
        sse_chunk_ser(&chunk).to_vec()
    }

    fn templated(id: &str, model: &str, created: i64, token: &str) -> Vec<u8> {
        let mut p = Vec::new();
        p.extend_from_slice(b"data: {\"id\":");
        serde_json::to_writer(&mut p, &id).unwrap();
        p.extend_from_slice(b",\"object\":\"chat.completion.chunk\",\"created\":");
        p.extend_from_slice(created.to_string().as_bytes());
        p.extend_from_slice(b",\"model\":");
        serde_json::to_writer(&mut p, &model).unwrap();
        p.extend_from_slice(b",\"choices\":[{\"index\":0,\"delta\":{\"content\":");
        serde_json::to_writer(&mut p, &token).unwrap();
        p.extend_from_slice(b"}}]}\n\n");
        p
    }

    #[test]
    fn templated_frame_is_byte_identical_to_full_serialize() {
        let id = "chatcmpl-abc123";
        let model = "meta-llama/Llama-3.1-8B-Instruct";
        let created = 1_726_000_000_i64;
        for token in [
            "hello",
            " world",
            "tok42",
            "",
            "with\"quote",
            "back\\slash",
            "new\nline",
            "tab\there",
            "unicode-\u{00e9}\u{4e2d}\u{6587}",
            "emoji-\u{1F600}",
            "ctrl-\u{0001}\u{001f}",
            "\"",
            "\\",
            "\u{0000}",
        ] {
            assert_eq!(
                templated(id, model, created, token),
                full_serde(id, model, created, token),
                "templated frame diverged from full serialize for token {token:?}"
            );
        }
    }

    #[test]
    fn templated_frame_is_valid_sse_and_parses() {
        let out = templated("id1", "m", 7, "hi");
        assert!(out.starts_with(b"data: "));
        assert!(out.ends_with(b"}}]}\n\n"));
        let json = &out[b"data: ".len()..out.len() - 2];
        let v: serde_json::Value = serde_json::from_slice(json).unwrap();
        assert_eq!(v["object"], "chat.completion.chunk");
        assert_eq!(v["choices"][0]["delta"]["content"], "hi");
    }

    #[test]
    fn tool_call_arguments_split_reconstructs_and_respects_char_boundaries() {
        let spec = ToolCallSpec {
            id: "call_x".into(),
            name: "get_weather".into(),
            arguments: r#"{"location":"NYC"}"#.into(),
        };
        let (head, tail) = spec.split_arguments();
        assert!(!head.is_empty() && !tail.is_empty());
        assert_eq!(format!("{head}{tail}"), spec.arguments);

        let spec = ToolCallSpec {
            id: "call_y".into(),
            name: "f".into(),
            arguments: "\u{4e2d}\u{6587}\u{1F600}".into(),
        };
        let (head, tail) = spec.split_arguments();
        assert_eq!(format!("{head}{tail}"), spec.arguments);
    }
}

#[cfg(test)]
mod kserve_http_tests {
    use super::*;
    use crate::grpc::GrpcBehavior;

    fn fast_state() -> Arc<AppState> {
        let config = crate::config::MockServerConfig {
            fast: true,
            no_tokenizer: true,
            ..crate::config::MockServerConfig::default()
        }
        .apply_flags();
        AppState::build(config)
    }

    fn v2_body(inputs: Value) -> Value {
        json!({"inputs": inputs})
    }

    #[test]
    fn tensor_text_and_int_extraction() {
        let body = v2_body(json!([
            {"name": "text_input", "datatype": "BYTES", "shape": [1], "data": ["hello world"]},
            {"name": "max_tokens", "datatype": "INT32", "shape": [1], "data": [16]},
        ]));
        assert_eq!(v2_tensor_texts(&body, "text_input"), vec!["hello world"]);
        assert_eq!(v2_tensor_first_int(&body, "max_tokens"), Some(16));
        assert!(v2_has_input(&body, "text_input"));
        assert!(!v2_has_input(&body, "passages"));
    }

    #[tokio::test]
    async fn behavior_detection_from_input_tensors() {
        let state = fast_state();
        let rankings = v2_body(json!([
            {"name": "query", "datatype": "BYTES", "shape": [1], "data": ["q"]},
            {"name": "passages", "datatype": "BYTES", "shape": [2], "data": ["p0", "p1"]},
        ]));
        assert_eq!(
            resolve_v2_behavior(&state, &rankings),
            GrpcBehavior::Rankings
        );

        let images = v2_body(json!([
            {"name": "prompt", "datatype": "BYTES", "shape": [1], "data": ["draw"]},
        ]));
        assert_eq!(resolve_v2_behavior(&state, &images), GrpcBehavior::Images);

        let text = v2_body(json!([
            {"name": "text_input", "datatype": "BYTES", "shape": [1], "data": ["hi"]},
        ]));
        assert_eq!(resolve_v2_behavior(&state, &text), GrpcBehavior::Text);
    }

    #[tokio::test]
    async fn v2_infer_text_returns_text_output() {
        let state = fast_state();
        let body = v2_body(json!([
            {"name": "text_input", "datatype": "BYTES", "shape": [1], "data": ["generate some text here"]},
        ]));
        let resp = kserve_v2_infer(
            State(state),
            axum::extract::Path("m".to_string()),
            Json(body),
        )
        .await
        .expect("v2 infer ok");
        let value = response_json(resp).await;
        let outputs = value["outputs"].as_array().expect("outputs array");
        assert_eq!(outputs[0]["name"], "text_output");
        assert_eq!(outputs[0]["datatype"], "BYTES");
        let text = outputs[0]["data"][0].as_str().expect("text output");
        assert!(!text.is_empty());
    }

    #[tokio::test]
    async fn v2_infer_rankings_returns_scores() {
        let state = fast_state();
        let body = v2_body(json!([
            {"name": "query", "datatype": "BYTES", "shape": [1], "data": ["what is ai"]},
            {"name": "passages", "datatype": "BYTES", "shape": [3], "data": ["p0", "p1", "p2"]},
        ]));
        let resp = kserve_v2_infer(
            State(state),
            axum::extract::Path("reranker".to_string()),
            Json(body),
        )
        .await
        .expect("v2 rankings ok");
        let value = response_json(resp).await;
        let outputs = value["outputs"].as_array().expect("outputs array");
        assert_eq!(outputs[0]["name"], "scores");
        let data = outputs[0]["data"].as_array().expect("scores data");
        assert_eq!(data.len(), 3);
        for (passage, score) in ["p0", "p1", "p2"].iter().zip(data) {
            let expected = compute_mock_score("what is ai", passage);
            assert!((score.as_f64().unwrap() - expected).abs() < 1e-9);
        }
    }

    #[tokio::test]
    async fn v2_infer_images_returns_generated_image() {
        let state = fast_state();
        let body = v2_body(json!([
            {"name": "prompt", "datatype": "BYTES", "shape": [1], "data": ["a red bicycle"]},
        ]));
        let resp = kserve_v2_infer(
            State(state),
            axum::extract::Path("diffusion".to_string()),
            Json(body),
        )
        .await
        .expect("v2 images ok");
        let value = response_json(resp).await;
        let outputs = value["outputs"].as_array().expect("outputs array");
        assert_eq!(outputs[0]["name"], "generated_image");
        assert_eq!(outputs[0]["data"][0], mock_jpeg_b64("a red bicycle", 0));
    }

    #[tokio::test]
    async fn v1_predict_strips_verb_and_returns_predictions() {
        let state = fast_state();
        let body = json!({"instances": [{"text": "translate this sentence"}]});
        let resp = kserve_v1_predict(
            State(state),
            axum::extract::Path("mymodel:predict".to_string()),
            Json(body),
        )
        .await
        .expect("v1 predict ok");
        let value = response_json(resp).await;
        let predictions = value["predictions"].as_array().expect("predictions array");
        let output = predictions[0]["output"].as_str().expect("output text");
        assert!(!output.is_empty());
    }

    async fn response_json(resp: Response) -> Value {
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .expect("read body");
        serde_json::from_slice(&bytes).expect("json body")
    }
}
