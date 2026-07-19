// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Axum handlers for every mock-server endpoint.

use std::convert::Infallible;
use std::sync::Arc;
use std::time::{Duration, Instant};

use aiperf_runtime::rng::RandomGenerator;
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
        let mut tokenized = tokenize_request(req_gen);
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
        request_id.clone(),
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
                let envelope = aiperf_runtime::transport::core::encode_payload_part(rest);
                let frame = EventStreamMessage::payload_part(envelope).encode();
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
    let mut rng = RandomGenerator::from_seed(Some(seed));
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

/// Renders the complete fast-mode chat stream into one allocation.
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

    // Only the final tool-call delta carries the terminal finish reason.
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

/// Emits two deltas whose argument fragments concatenate to the configured
/// string; the second carries `finish_reason: "tool_calls"`.
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

        // A tool-call turn carries its finish reason on the final call delta.
        let has_tool_call = ctx.tool_call.is_some();
        let num = ctx.tokenized.tokens.len();
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
            let finish = if i + 1 == num && !has_tool_call { Some(ctx.tokenized.finish_reason) } else { None };
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

fn prom_response(body: Vec<u8>) -> Response {
    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "text/plain; version=0.0.4")
        .body(Body::from(body))
        .expect("response")
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
        assert!(is_fetchable_url("http://host:8090/content/images/img_1.png"));
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
        let out: Vec<Bytes> = futures::executor::block_on(
            futures::StreamExt::collect::<Vec<_>>(sse_to_eventstream(sse)),
        )
        .into_iter()
        .map(|r| r.unwrap())
        .collect();

        assert_eq!(out.len(), 1);
        let mut decoder = aiperf_runtime::transport::core::EventStreamDecoder::new();
        decoder.push(&out[0]);
        let messages = decoder.drain_messages().unwrap();
        assert_eq!(messages.len(), 1);
        let inner =
            aiperf_runtime::transport::core::decode_payload_part(&messages[0].payload).unwrap();
        assert_eq!(
            &inner[..],
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
