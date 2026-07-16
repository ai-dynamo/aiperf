// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Axum handlers for every mock-server endpoint.

use std::convert::Infallible;
use std::sync::Arc;
use std::time::{Duration, Instant};

use aiperf::rng::RandomGenerator;
use axum::Json;
use axum::body::Body;
use axum::extract::State;
use axum::http::{StatusCode, header};
use axum::response::{IntoResponse, Response};
use base64::Engine;
use blake2::{Blake2s256, Digest};
use bytes::Bytes;
use futures::stream::Stream;
use serde_json::{Value, json};

use crate::latency::{LatencySimulator, wait_for_processing};
use crate::metrics::LLMLatencyInfo;
use crate::models::{
    ChatCompletionRequest, CohereRerankRequest, CompletionRequest, EmbeddingRequest,
    HFTEIRerankRequest, ImageGenerationRequest, ImageResponseFormat, ImageRetrievalRequest,
    Message, MessagesRequest, RankingRequest, SolidoRAGRequest, TGIGenerateRequest, Usage,
};
use crate::state::AppState;
use crate::tokens::{GenRequest, TokenizedText, tokenize_request};

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

fn make_request_id(prefix: &str) -> String {
    format!("{prefix}-{}", uuid::Uuid::new_v4())
}

fn make_anthropic_message_id() -> String {
    format!("msg_{}", uuid::Uuid::new_v4())
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
/// `pub(crate)` so alternative front doors (e.g. the KServe gRPC service in
/// [`crate::grpc`]) reuse the exact tokenize → usage → latency/prefix-cache head
/// the HTTP handlers use, rather than re-deriving it.
pub(crate) struct RequestCtx {
    pub(crate) request_id: String,
    pub(crate) model: String,
    pub(crate) tokenized: TokenizedText,
    pub(crate) usage: Usage,
    pub(crate) latency_sim: LatencySimulator,
    pub(crate) start: Instant,
    /// When true (accuracy adversarial `NullObjectChunk`), the streaming path
    /// emits one `{"object": null}` SSE frame before `[DONE]` (github #1010).
    pub(crate) null_object_chunk: bool,
    /// Present when the seeded `--tool-call-rate` draw fires for this request:
    /// the chat response answers with this function tool call. Set only on the
    /// chat endpoint (see [`chat_completions`]); every other front door leaves
    /// it `None`, so their payloads are unchanged.
    pub(crate) tool_call: Option<ToolCallSpec>,
}

/// A single deterministic function tool call the mock emits when
/// `--tool-call-rate` fires. Mirrors the OpenAI wire shape: `arguments` is a
/// JSON-encoded *string* (not an object), and `id`/`type` identify the call.
pub(crate) struct ToolCallSpec {
    pub(crate) id: String,
    pub(crate) name: String,
    pub(crate) arguments: String,
}

impl ToolCallSpec {
    /// Build the deterministic tool call from config, and report the count of
    /// tool-definition prompt tokens to attribute to it (`toolUsePromptTokenCount`
    /// in the emitted usage). The count is the mock-tokenized length of
    /// `name + arguments`, so it is deterministic in the configured knobs.
    fn from_config(cfg: &crate::config::MockServerConfig) -> (Self, usize) {
        let name = cfg.tool_call_name.clone();
        let arguments = cfg.tool_call_arguments.clone();
        let tool_use_tokens = crate::tokens::tokenize(&format!("{name}{arguments}")).len();
        let spec = Self {
            // Stable-per-request id derived off the request id at the callsite is
            // overkill; a fresh uuid matches real APIs and is opaque to the runner
            // (which keys tool calls by streamed `index`, not id).
            id: format!("call_{}", uuid::Uuid::new_v4()),
            name,
            arguments,
        };
        (spec, tool_use_tokens)
    }

    /// Split `arguments` into two contiguous halves on a char boundary, so the
    /// streaming path can emit the argument string across two `delta.tool_calls`
    /// frames and exercise the runner's argument-concatenation merge. Returns
    /// `(first, second)`; `second` is empty for a one-or-zero-char argument.
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
        let mut tokenized = tokenize_request(req_gen);
        // Ground-truth-aware override: when an accuracy dataset is loaded and the
        // request's user text matches a row, replace the corpus-generated tokens
        // with the (seeded) correct-or-wrong answer, formatted for the grader,
        // optionally as CoT or an adversarial parser-choke shape. All endpoints
        // and both streaming and non-streaming paths serialize
        // `tokenized.content()`, so this single seam covers every front door.
        let mut null_object_chunk = false;
        if let Some(ds) = &state.accuracy {
            if let Some(entry) = ds.lookup(&tokenized.text) {
                let decision = ds.decide(entry);
                // Live tally: count this real, prompt-matched response so
                // `correct / matched` reflects the run's actual accuracy.
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
        // KV-cache prefix reuse: cached prefix tokens are always REPORTED in
        // usage (AIPerf reads usage_prompt_cache_read_tokens). Whether they also
        // reduce prefill work / TTFT is gated by --prefix-cache-latency-aware,
        // because in a saturated queue-bound regime TTFT is contention-bound and
        // empirically independent of cache hits.
        let cached_tokens = match &state.prefix_cache {
            Some(pc) => pc.cached_tokens(&tokenized.text, usage.prompt_tokens, req_gen.priority()),
            None => 0,
        };
        // Always emit prompt_tokens_details so callers can observe cache-read
        // counts even when the prefix cache is disabled (cached_tokens == 0).
        // The Python mock server always includes this field.
        usage.prompt_tokens_details = Some(crate::models::PromptTokensDetails {
            cached_tokens,
            audio_tokens: None,
        });
        // Extended usage-accounting fields (deterministic, driven by the
        // `--usage-*` knobs). Skipped wholesale unless at least one is set, so a
        // normal run's usage payload is byte-identical. Every field maps to a
        // specific key AIPerf's `UsageView` reads (see models.rs doc comments).
        if state.config.usage_fields_enabled() {
            apply_usage_fields(&mut usage, &state.config);
        }
        let latency_cached = if state.config.prefix_cache_latency_aware {
            cached_tokens
        } else {
            0
        };
        // +1 counts this request alongside those already in flight. The
        // LatencySimulator derives effective TTFT/ITL from the ISL/OSL and
        // concurrency knobs, or routes through the scheduler when enabled.
        let active_inflight = (state.recorder.inflight_count().max(0) as usize) + 1;
        let latency_sim = LatencySimulator::new(
            state.clock_anchor,
            &state.config,
            usage.prompt_tokens,
            tokenized.count(),
            active_inflight,
            state.scheduler.clone(),
            request_id.clone(),
            latency_cached,
        );
        Self {
            request_id,
            latency_sim,
            model,
            tokenized,
            usage,
            start,
            null_object_chunk,
            tool_call: None,
        }
    }
}

// ============================================================================
// Root / health
// ============================================================================

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

// ============================================================================
// /v1/models — OpenAI-style model listing.
//
// Returns the configured `--models` list (or a builtin default set) unioned
// with every model name that has been observed via real traffic, so a client
// that just issued a request against `mymodel` will also see `mymodel` in the
// listing. `owned_by` is constant; `created` is the server start time so the
// output is stable across scrapes.
// ============================================================================

/// Default list advertised when no `--models` flag / env var was supplied.
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

// ============================================================================
// Chat completions
// ============================================================================

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
    // Seeded per-request tool-call decision (chat endpoint only). When it fires,
    // attach the deterministic tool call and report its tool-definition prompt
    // tokens as `toolUsePromptTokenCount`, which the runner reads into
    // `usage_tool_use_prompt_tokens`.
    if state.inject_tool_call() {
        let (spec, tool_use_tokens) = ToolCallSpec::from_config(&state.config);
        ctx.usage.tool_use_prompt_token_count = Some(tool_use_tokens);
        ctx.tool_call = Some(spec);
    }

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
}

fn build_chat_response(ctx: &RequestCtx) -> Value {
    let mut message = json!({
        "role": "assistant",
        "content": ctx.tokenized.content(),
    });
    if let Some(reasoning) = ctx.tokenized.reasoning_content() {
        message["reasoning_content"] = Value::String(reasoning);
    }
    // When a tool call fires, emit the OpenAI `message.tool_calls` array and
    // switch the finish reason to `tool_calls`. The generated content stays
    // alongside it (the mock's token/latency model is unchanged); the runner
    // parses `function.name` + `function.arguments` into the record.
    let finish_reason: Value = if let Some(tc) = &ctx.tool_call {
        message["tool_calls"] = json!([{
            "id": tc.id,
            "type": "function",
            "function": {
                "name": tc.name,
                "arguments": tc.arguments,
            },
        }]);
        Value::String("tool_calls".to_string())
    } else {
        Value::String(ctx.tokenized.finish_reason.to_string())
    };
    json!({
        "id": ctx.request_id,
        "object": "chat.completion",
        "created": now_secs(),
        "model": ctx.model,
        "choices": [{
            "index": 0,
            "finish_reason": finish_reason,
            "message": message,
        }],
        "usage": ctx.usage,
    })
}

// ============================================================================
// Anthropic Messages
// ============================================================================

/// Handle an Anthropic Messages request with the mock's shared token and latency model.
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

/// Inject the deterministic `--usage-*` extended-accounting fields into a usage
/// object. Each `0` (or `0.0`) knob leaves its field absent, so only explicitly
/// requested sub-fields appear. Nested details (`prompt_tokens_details`,
/// `completion_tokens_details`) are created on demand when only an extended
/// field needs them (e.g. a non-reasoning model with prediction tokens).
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
/// AIPerf's `UsageView` re-totals (`aiperf::endpoints::usage` lines 37-45,
/// 206-211) when the corresponding `--usage-*` knobs are set.
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

// ============================================================================
// Text completions
// ============================================================================

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

// ============================================================================
// Embeddings
// ============================================================================

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
    let mut rng = RandomGenerator::from_seed(Some(seed));
    (0..dim).map(|_| rng.random() - 0.5).collect()
}

// ============================================================================
// Rankings
// ============================================================================

fn compute_mock_score(query: &str, passage: &str) -> f64 {
    let mut hasher = Blake2s256::new();
    hasher.update(query.as_bytes());
    hasher.update(b"|");
    hasher.update(passage.as_bytes());
    let digest = hasher.finalize();
    // Mimic Python: int_digest = int.from_bytes(digest, byteorder='big'); (int_digest % 1000) / 1000.0
    // 8 bytes of prefix is enough for deterministic modulo.
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
    let passages = req.passage_texts();
    let (_, scores, _) = handle_ranking_common(
        state.clone(),
        endpoint,
        &req.model,
        req.query_text(),
        passages,
        tokenized.prompt_token_count,
    )
    .await;
    let results: Vec<Value> = scores
        .into_iter()
        .map(|(i, s)| json!({"index": i, "score": s}))
        .collect();
    Ok(Json(json!({ "results": results })).into_response())
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
    let passages = req.passage_texts();
    let (_, scores, _) = handle_ranking_common(
        state.clone(),
        endpoint,
        &req.model,
        req.query_text(),
        passages,
        tokenized.prompt_token_count,
    )
    .await;
    let results: Vec<Value> = scores
        .into_iter()
        .map(|(i, s)| json!({"index": i, "relevance_score": s}))
        .collect();
    Ok(Json(json!({ "results": results })).into_response())
}

// ============================================================================
// NIM Image Retrieval
// ============================================================================

const BOUNDING_BOX_CATEGORIES: &[&str] = &["title", "table", "figure", "text", "header", "footer"];

fn generate_bounding_boxes(url: &str) -> serde_json::Map<String, Value> {
    let mut hasher = Blake2s256::new();
    hasher.update(url.as_bytes());
    let digest = hasher.finalize();
    let seed = u64::from_be_bytes(digest[0..8].try_into().unwrap());
    let mut rng = RandomGenerator::from_seed(Some(seed));
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
    let mut data = Vec::new();
    for (i, img) in req.input.iter().enumerate() {
        let bounding = generate_bounding_boxes(&img.url);
        data.push(json!({
            "index": i,
            "bounding_boxes": bounding,
        }));
        total_size_mb += img.url.len() as f64 / (1024.0 * 1024.0 * 1.37);
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

// ============================================================================
// Custom multimodal
// ============================================================================

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

// ============================================================================
// TGI (HuggingFace)
// ============================================================================

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

// ============================================================================
// Image generation
// ============================================================================

fn mock_jpeg_b64(prompt: &str, index: u32) -> String {
    let combined = format!("{prompt}|{index}");
    let mut hasher = Blake2s256::new();
    hasher.update(combined.as_bytes());
    let digest = hasher.finalize();

    // Blake2s-256 produces 32 bytes. The Python source uses `digest[:64]` and
    // `digest[64:80]` which Python slices silently saturate/empty — we replicate
    // that by clamping to the digest length.
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

// ============================================================================
// SOLIDO RAG
// ============================================================================

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

// ============================================================================
// Streaming helpers
// ============================================================================

fn sse_chunk(value: &Value) -> Bytes {
    let mut out = Vec::with_capacity(256);
    out.extend_from_slice(b"data: ");
    serde_json::to_writer(&mut out, value).unwrap();
    out.extend_from_slice(b"\n\n");
    Bytes::from(out)
}

/// Serialize any serde-serializable value directly as an SSE chunk. Avoids the
/// intermediate serde_json::Value / HashMap allocation that `json!({...})` does.
fn sse_chunk_ser<T: serde::Serialize>(value: &T) -> Bytes {
    let mut out = Vec::with_capacity(256);
    out.extend_from_slice(b"data: ");
    serde_json::to_writer(&mut out, value).expect("serialize");
    out.extend_from_slice(b"\n\n");
    Bytes::from(out)
}

/// Append an SSE chunk (`data: {json}\n\n`) directly onto a shared buffer.
/// Used by the fast-mode pre-render path to build the whole response into
/// one allocation.
fn write_sse_into<T: serde::Serialize>(buf: &mut Vec<u8>, value: &T) {
    buf.extend_from_slice(b"data: ");
    serde_json::to_writer(&mut *buf, value).expect("serialize");
    buf.extend_from_slice(b"\n\n");
}

// ---------------------------------------------------------------------------
// Streaming chunk payloads — plain structs serialize faster than `json!({…})`
// because they skip the intermediate serde_json::Map allocation.
// ---------------------------------------------------------------------------

#[derive(serde::Serialize)]
struct ChatChoiceDelta<'a> {
    index: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    finish_reason: Option<&'static str>,
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
/// (`aiperf::transport_http::sse::reader::read_sse`) classifies any frame whose
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

/// Single-Bytes pre-rendered SSE body for fast mode. Renders all chat chunks,
/// the optional usage chunk, and the terminal `[DONE]` into one allocation.
fn render_chat_fast_body(ctx: &RequestCtx, include_usage: bool) -> Bytes {
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

    // When a tool call follows, the terminal `finish_reason` rides its final
    // frame, so content tokens never carry it (real APIs finish once, on the
    // last delta of the whole turn).
    let has_tool_call = ctx.tool_call.is_some();
    let num = ctx.tokenized.tokens.len();
    for (i, token) in ctx.tokenized.tokens.iter().enumerate() {
        let role = if i == 0 && !has_reasoning {
            Some("assistant")
        } else {
            None
        };
        let finish = if i + 1 == num && !has_tool_call {
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
        for chunk in tool_call_frames(ctx, created, tc, lead_role) {
            write_sse_into(&mut buf, &chunk);
        }
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

/// Build the two streamed tool-call frames for a fired `--tool-call-rate` chat
/// request. Frame 1 opens the call (`id`, `type`, `function.name`, and the first
/// half of the arguments); frame 2 carries the second half plus the terminal
/// `finish_reason: "tool_calls"`. Splitting the arguments across two frames
/// exercises the runner's argument-concatenation merge. `lead_role` stamps
/// `role: assistant` on the first frame when no content/reasoning token preceded
/// it (so a tool-call-only stream still opens the assistant turn).
fn tool_call_frames<'a>(
    ctx: &'a RequestCtx,
    created: i64,
    tc: &'a ToolCallSpec,
    lead_role: bool,
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
            finish_reason: Some("tool_calls"),
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

        // Mid-stream failure: emit a few real token frames, then a terminal
        // `event: error` SSE frame and close — no usage chunk, no `[DONE]`.
        // This is the only path that exercises the runner's mid-stream SSE
        // error classification (pre-stream injection fails at handler entry
        // before any bytes are sent). Runs even in fast mode, and never draws
        // the adversarial null-object path.
        if midstream_error {
            let created = now_secs();
            let has_reasoning = !ctx.tokenized.reasoning_content_tokens.is_empty();
            let num = ctx.tokenized.tokens.len();
            let emit = num.min(MIDSTREAM_TOKENS_BEFORE_ERROR);
            for (i, token) in ctx.tokenized.tokens.iter().take(emit).enumerate() {
                // Pace real (non-fast) streams so partial timing is realistic.
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
            // Count as a failed request so the mock's own metrics reflect it.
            state.recorder.record_error(&endpoint, "midstream_sse_error");
            state.recorder.record_llm_inflight_end(&ctx.model);
            state.recorder.record_request_end(&endpoint);
            return;
        }

        // Fast-mode short-circuit: ttft==itl==0. Pre-render the entire SSE
        // body into one Bytes and yield it in a single HTTP frame. The
        // adversarial null-object frame must land *before* `[DONE]`, which the
        // pre-rendered body already contains, so skip the short-circuit when it
        // is requested and run the (still instant) token loop instead.
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

        // OpenAI holds `created` constant across a stream (the fast path samples
        // once too); sample once here rather than per token.
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
            let chunk = ChatStreamChunk {
                id: &ctx.request_id,
                object: "chat.completion.chunk",
                created,
                model: &ctx.model,
                choices: [ChatChoiceDelta {
                    index: 0,
                    finish_reason: None,
                    delta: ChatDelta {
                        role: Some("assistant"),
                        content: None,
                        reasoning_content: Some(token.as_str()),
                        tool_calls: None,
                    },
                }],
            };
            yield Ok::<Bytes, Infallible>(sse_chunk_ser(&chunk));
        }

        // A pending tool call takes the terminal `finish_reason` on its own final
        // frame, so no content token carries it (real APIs finish once).
        let has_tool_call = ctx.tool_call.is_some();
        let num = ctx.tokenized.tokens.len();
        // Pre-serialize the constant per-request frame envelope once. Every
        // middle token's chunk is byte-for-byte `<prefix>"<escaped token>"<suffix>`
        // — only the token string varies — so we serialize just that string and
        // splice, instead of re-serializing the whole `ChatStreamChunk` struct
        // once per token (the profiled hot path: e.g. 254 serializes/request ×
        // millions of requests). Each token is still emitted as its own SSE
        // frame (one `yield` = one packet on the wire); this only removes the
        // redundant per-token struct traversal, byte-identical to `sse_chunk_ser`.
        // The first frame carries `role` and the last carries `finish_reason`, so
        // those two boundary frames fall back to the full serializer.
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
            let finish = if i + 1 == num && !has_tool_call { Some(ctx.tokenized.finish_reason) } else { None };
            if role.is_none() && finish.is_none() {
                // Common middle token: splice the pre-serialized envelope with the
                // token's own escaped JSON string. Still one frame per token.
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

        if let Some(tc) = &ctx.tool_call {
            let lead_role = !has_reasoning && num == 0;
            for chunk in tool_call_frames(&ctx, created, tc, lead_role) {
                yield Ok::<Bytes, Infallible>(sse_chunk_ser(&chunk));
            }
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
            // github #1010: a terminal chunk with `object: null` arriving before
            // `[DONE]`. A robust parser treats it as an end-of-stream marker; a
            // brittle one raises `Unsupported OpenAI object type: None`. Emitted
            // as a standalone frame so the run's parser is genuinely exercised.
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
        // OpenAI holds `created` constant across a stream (the fast path samples
        // once too); sample once here rather than per token.
        let created = now_secs();
        let mut first_emit: Option<Instant> = None;
        let mut last_emit: Option<Instant> = None;
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

// ============================================================================
// Metrics endpoints
// ============================================================================

fn prom_response(body: Vec<u8>) -> Response {
    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "text/plain; version=0.0.4")
        .body(Body::from(body))
        .expect("response")
}

pub async fn aiperf_mock_metrics(State(state): State<Arc<AppState>>) -> Response {
    // Update uptime on each scrape.
    state
        .recorder
        .metrics
        .aiperf
        .SERVER_UPTIME_SECONDS
        .set(state.uptime_secs());
    let mut body = crate::prom::encode(&state.recorder.metrics.aiperf.registry);
    // Append the live accuracy tally (computed from atomics at scrape time) when
    // the accuracy dataset mode is active. These names are not in the registry,
    // so appending them to the exposition text is valid.
    if state.accuracy.is_some() {
        crate::prom::append_accuracy_metrics(&mut body, &state.accuracy_live.snapshot());
    }
    prom_response(body)
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
    prom_response(crate::prom::encode(&state.recorder.metrics.vllm.registry))
}

pub async fn sglang_metrics(State(state): State<Arc<AppState>>) -> Response {
    prom_response(crate::prom::encode(&state.recorder.metrics.sglang.registry))
}

pub async fn trtllm_metrics(State(state): State<Arc<AppState>>) -> Response {
    prom_response(crate::prom::encode(&state.recorder.metrics.trtllm.registry))
}

pub async fn dynamo_frontend_metrics(State(state): State<Arc<AppState>>) -> Response {
    prom_response(crate::prom::encode(
        &state.recorder.metrics.dynamo_frontend.registry,
    ))
}

pub async fn dynamo_prefill_metrics(State(state): State<Arc<AppState>>) -> Response {
    prom_response(crate::prom::encode(
        &state.recorder.metrics.dynamo_prefill.registry,
    ))
}

pub async fn dynamo_decode_metrics(State(state): State<Arc<AppState>>) -> Response {
    prom_response(crate::prom::encode(
        &state.recorder.metrics.dynamo_decode.registry,
    ))
}

// ============================================================================
// DCGM
// ============================================================================

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

/// Mock `/v1/images/edits` — drain multipart body, return a synthetic JPEG.
///
/// Accepts multipart/form-data with optional `image` file and `prompt` text
/// fields (same surface as the Python mock). Mirrors `image_generation` in
/// response shape so the Rust runner receives a valid response.
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

    // Drain all multipart fields; capture the ones we care about.
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
                // Drain unknown / binary fields (e.g. image upload).
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

#[cfg(test)]
mod stream_frame_tests {
    use super::*;

    // The full-serialize form the streaming loop falls back to for boundary frames.
    fn full_serde(id: &str, model: &str, created: i64, token: &str) -> Vec<u8> {
        let chunk = ChatStreamChunk {
            id,
            object: "chat.completion.chunk",
            created,
            model,
            choices: [ChatChoiceDelta {
                index: 0,
                finish_reason: None,
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

    // The pre-serialized-envelope splice used for middle tokens.
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
        // ASCII arguments split into two non-empty contiguous halves.
        let spec = ToolCallSpec {
            id: "call_x".into(),
            name: "get_weather".into(),
            arguments: r#"{"location":"NYC"}"#.into(),
        };
        let (head, tail) = spec.split_arguments();
        assert!(!head.is_empty() && !tail.is_empty());
        assert_eq!(format!("{head}{tail}"), spec.arguments);

        // Multibyte arguments never split mid-codepoint.
        let spec = ToolCallSpec {
            id: "call_y".into(),
            name: "f".into(),
            arguments: "\u{4e2d}\u{6587}\u{1F600}".into(),
        };
        let (head, tail) = spec.split_arguments();
        assert_eq!(format!("{head}{tail}"), spec.arguments);
    }
}
