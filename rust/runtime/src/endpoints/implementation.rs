// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Core endpoint implementations.

use std::collections::{BTreeMap, HashSet};
use std::sync::OnceLock;

use base64::Engine;
use base64::engine::general_purpose::STANDARD;
use bytes::Bytes;
use serde_json::{Map, Value, json};
use smallvec::SmallVec;

use crate::body_plan::{
    BodyPlan, JsonBodyMaterializer, PreparedWsMessage, PreparedWsMessageRole, PreparedWsOpcode,
    PreparedWsOperation,
};
use crate::dataset::materialize::Overrides;
use crate::dataset::segment::SegmentStore;
use crate::endpoints::config::{EndpointConfig, RawEndpointConfig};
use crate::endpoints::extraction::{PartTypes, extract_inputs};
use crate::endpoints::anthropic::apply_messages_auth_headers;
use crate::endpoints::metadata::{EndpointDescriptor, EndpointType, Modality};
use crate::endpoints::models::{
    AudioResponseData, CreditPhase, EndpointError, EndpointResult, ExtractedPayload, Media,
    ParsedResponse, RequestInfo, RequestRecord, ResponseData, ServerResponse, Turn,
};
use crate::endpoints::registry::{
    PreparedEndpointBehavior, PreparedReadinessRequest, PreparedRequest, ReadinessMethod,
    ReadinessPolicy, ReadinessSuccess, WebSocketCapabilities, WebSocketConnectionModel,
    WebSocketDialect, format_legacy_payload,
};

static FORCE_CONTENT_PARTS: OnceLock<bool> = OnceLock::new();

/// Capture endpoint wire-format policy once at runner bootstrap.
pub fn capture_endpoint_policy() {
    let _ = FORCE_CONTENT_PARTS.set(
        std::env::var("AIPERF_ENDPOINT_FORCE_CONTENT_PARTS").is_ok_and(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes"
            )
        }),
    );
}

/// Warmup prefix used by the completions endpoint.
pub const WARMUP_SYSTEM_MESSAGE_PREFIX: &str =
    "You are in warmup mode. This request is used to warm up the benchmark target.";

/// Endpoint adapter contract.
pub trait Endpoint: std::fmt::Debug + Send + Sync {
    /// Return the canonical open-ID descriptor registered with the runner.
    fn descriptor(&self) -> &'static EndpointDescriptor;
    /// Build a request-body plan the shared materializer splices into wire bytes.
    fn format_payload(&self, request_info: &RequestInfo) -> EndpointResult<BodyPlan>;
    /// Build endpoint-owned request headers before per-turn overrides.
    ///
    /// Authentication is a dialect property: OpenAI-compatible endpoints use
    /// bearer auth while Anthropic Messages uses `x-api-key`. Keeping the hook
    /// here lets every workload share the same transport path.
    fn format_headers(&self, config: &EndpointConfig) -> BTreeMap<String, String> {
        let mut headers = config.headers.clone();
        if let Some(api_key) = &config.api_key {
            apply_bearer_auth_header(&mut headers, api_key);
        }
        headers
    }
    /// Build the dialect-owned readiness policy for one effective model.
    ///
    /// Dialects must either return an exact request or explicitly decline
    /// readiness probing; orchestration must never substitute a chat payload.
    fn readiness_policy(
        &self,
        _config: &EndpointConfig,
        _model: &str,
    ) -> EndpointResult<ReadinessPolicy> {
        Ok(ReadinessPolicy::Unsupported {
            reason: "this endpoint dialect has not declared a readiness request",
        })
    }
    /// Parse a decoded server response.
    fn parse_response(&self, response: &ServerResponse) -> EndpointResult<Option<ParsedResponse>>;
    /// Parse with the effective per-request configuration when a dialect needs
    /// streaming or response-selector context. Stateless dialects inherit the
    /// ordinary parser.
    fn parse_response_with_config(
        &self,
        response: &ServerResponse,
        _config: &EndpointConfig,
    ) -> EndpointResult<Option<ParsedResponse>> {
        self.parse_response(response)
    }
    /// Extract tokenizable input and media counts from a built body.
    fn extract_payload_inputs(&self, body: &Value) -> ExtractedPayload {
        extract_inputs(body, &self.part_types())
    }
    /// Build an assistant turn for context replay.
    fn build_assistant_turn(&self, record: &RequestRecord) -> EndpointResult<Option<Turn>> {
        build_plain_assistant_turn(self, record)
    }
    /// Whether successful responses should be reconstructed for later turns.
    fn captures_assistant_turn(&self) -> bool {
        false
    }
    /// Return endpoint content-part names.
    fn part_types(&self) -> PartTypes {
        PartTypes::chat()
    }
    /// Parse every response in a record.
    fn extract_response_data(&self, record: &RequestRecord) -> EndpointResult<Vec<ParsedResponse>> {
        let mut out = Vec::new();
        for response in &record.responses {
            if let Some(parsed) = self.parse_response(response)? {
                out.push(parsed);
            }
        }
        Ok(out)
    }
    /// Parse every response with one effective endpoint configuration.
    fn extract_response_data_with_config(
        &self,
        record: &RequestRecord,
        config: &EndpointConfig,
    ) -> EndpointResult<Vec<ParsedResponse>> {
        let mut out = Vec::new();
        for response in &record.responses {
            if let Some(parsed) = self.parse_response_with_config(response, config)? {
                out.push(parsed);
            }
        }
        Ok(out)
    }
}

/// OpenAI Chat Completions endpoint.
#[derive(Debug, Clone, Copy, Default)]
pub struct ChatEndpoint;
/// OpenAI Responses endpoint.
#[derive(Debug, Clone, Copy, Default)]
pub struct ResponsesEndpoint;

/// OpenAI Realtime WebSocket endpoint.
#[derive(Debug, Clone, Copy, Default)]
pub struct RealtimeEndpoint;
/// OpenAI Completions endpoint.
#[derive(Debug, Clone, Copy, Default)]
pub struct CompletionsEndpoint;
/// OpenAI Embeddings endpoint.
#[derive(Debug, Clone, Copy, Default)]
pub struct EmbeddingsEndpoint;
/// Chat-shaped embeddings endpoint.
#[derive(Debug, Clone, Copy, Default)]
pub struct ChatEmbeddingsEndpoint;

const CHAT_DESCRIPTOR: EndpointDescriptor = EndpointDescriptor {
    id: "chat",
    aliases: &["chat_completions"],
    description: "OpenAI-compatible Chat Completions API",
    endpoint_path: Some("/v1/chat/completions"),
    streaming_path: None,
    supports_streaming: true,
    produces_tokens: true,
    tokenizes_input: true,
    requires_raw_token_ids: false,
    requires_form_data: false,
    requires_polling: false,
    requires_inline_media: false,
    input_modalities: &[
        Modality::Text,
        Modality::Image,
        Modality::Audio,
        Modality::Video,
    ],
    output_modalities: &[Modality::Tokens],
    metrics_title: "LLM Metrics",
    service_kind: "llm",
};

const RESPONSES_DESCRIPTOR: EndpointDescriptor = EndpointDescriptor {
    id: "responses",
    aliases: &[],
    description: "OpenAI Responses API",
    endpoint_path: Some("/v1/responses"),
    streaming_path: None,
    supports_streaming: true,
    produces_tokens: true,
    tokenizes_input: true,
    requires_raw_token_ids: false,
    requires_form_data: false,
    requires_polling: false,
    requires_inline_media: false,
    input_modalities: &[Modality::Text, Modality::Image, Modality::Audio],
    output_modalities: &[Modality::Tokens],
    metrics_title: "LLM Metrics",
    service_kind: "llm",
};

const REALTIME_DESCRIPTOR: EndpointDescriptor = EndpointDescriptor {
    id: "realtime",
    aliases: &[],
    description: "OpenAI Realtime WebSocket API",
    endpoint_path: Some("/v1/realtime"),
    streaming_path: None,
    supports_streaming: true,
    produces_tokens: true,
    tokenizes_input: true,
    requires_raw_token_ids: false,
    requires_form_data: false,
    requires_polling: false,
    requires_inline_media: true,
    input_modalities: &[Modality::Text, Modality::Audio],
    output_modalities: &[Modality::Tokens, Modality::Audio],
    metrics_title: "LLM Metrics",
    service_kind: "llm",
};

const COMPLETIONS_DESCRIPTOR: EndpointDescriptor = EndpointDescriptor {
    id: "completions",
    aliases: &[],
    description: "OpenAI-compatible Completions API",
    endpoint_path: Some("/v1/completions"),
    streaming_path: None,
    supports_streaming: true,
    produces_tokens: true,
    tokenizes_input: true,
    requires_raw_token_ids: false,
    requires_form_data: false,
    requires_polling: false,
    requires_inline_media: false,
    input_modalities: &[Modality::Text],
    output_modalities: &[Modality::Tokens],
    metrics_title: "LLM Metrics",
    service_kind: "llm",
};

const EMBEDDINGS_DESCRIPTOR: EndpointDescriptor = EndpointDescriptor {
    id: "embeddings",
    aliases: &[],
    description: "OpenAI-compatible Embeddings API",
    endpoint_path: Some("/v1/embeddings"),
    streaming_path: None,
    supports_streaming: false,
    produces_tokens: false,
    tokenizes_input: true,
    requires_raw_token_ids: false,
    requires_form_data: false,
    requires_polling: false,
    requires_inline_media: false,
    input_modalities: &[Modality::Text],
    output_modalities: &[Modality::Embeddings],
    metrics_title: "Embeddings Metrics",
    service_kind: "embeddings",
};

const CHAT_EMBEDDINGS_DESCRIPTOR: EndpointDescriptor = EndpointDescriptor {
    id: "chat_embeddings",
    aliases: &[],
    description: "Chat-shaped embeddings API",
    endpoint_path: Some("/v1/embeddings"),
    streaming_path: None,
    supports_streaming: false,
    produces_tokens: false,
    tokenizes_input: true,
    requires_raw_token_ids: false,
    requires_form_data: false,
    requires_polling: false,
    requires_inline_media: false,
    input_modalities: &[Modality::Text, Modality::Image],
    output_modalities: &[Modality::Embeddings],
    metrics_title: "Embeddings Metrics",
    service_kind: "embeddings",
};

impl Endpoint for ChatEndpoint {
    fn descriptor(&self) -> &'static EndpointDescriptor {
        &CHAT_DESCRIPTOR
    }

    fn format_payload(&self, request_info: &RequestInfo) -> EndpointResult<BodyPlan> {
        format_legacy_payload(self, request_info)
    }

    fn readiness_policy(
        &self,
        config: &EndpointConfig,
        model: &str,
    ) -> EndpointResult<ReadinessPolicy> {
        let headers = self.format_headers(config);
        let models = PreparedReadinessRequest {
            method: ReadinessMethod::Get,
            path: "/v1/models".into(),
            headers: headers.clone(),
            body: None,
            success: ReadinessSuccess::ModelListed(model.into()),
        };
        let inference = PreparedReadinessRequest {
            method: ReadinessMethod::Post,
            path: "/v1/chat/completions".into(),
            headers,
            body: Some(json!({
                "messages": [{"role": "user", "content": "Lo"}],
                "max_tokens": 1,
                "model": model,
            })),
            success: ReadinessSuccess::NonServerError,
        };
        match config.wait_for_model_mode.as_str() {
            "models" => Ok(ReadinessPolicy::Request(models)),
            "inference" => Ok(ReadinessPolicy::Request(inference)),
            "both" => Ok(ReadinessPolicy::Requests(vec![models, inference])),
            mode => Err(EndpointError::InvalidConfig(format!(
                "unsupported readiness mode {mode:?}"
            ))),
        }
    }

    fn parse_response(&self, response: &ServerResponse) -> EndpointResult<Option<ParsedResponse>> {
        let Some(obj) = response.json.as_ref().and_then(Value::as_object) else {
            return Ok(None);
        };
        let data = extract_chat_response_data(obj);
        let usage = non_empty_field(obj, "usage");
        Ok(
            (data.is_some() || usage.is_some()).then_some(ParsedResponse {
                perf_ns: response.perf_ns,
                data,
                usage,
                sources: None,
            }),
        )
    }

    fn build_assistant_turn(&self, record: &RequestRecord) -> EndpointResult<Option<Turn>> {
        let mut content_parts = Vec::new();
        let mut tool_calls_by_index: BTreeMap<i64, Map<String, Value>> = BTreeMap::new();
        for response in &record.responses {
            let Some(obj) = response.json.as_ref().and_then(Value::as_object) else {
                continue;
            };
            let Some(choice) = first_choice(obj) else {
                continue;
            };
            absorb_chat_choice(
                obj.get("object").and_then(Value::as_str),
                choice,
                &mut content_parts,
                &mut tool_calls_by_index,
            );
        }
        if tool_calls_by_index.is_empty() {
            return build_plain_assistant_turn(self, record);
        }
        let text = content_parts.concat();
        let mut message = Map::new();
        message.insert("role".into(), Value::String("assistant".into()));
        message.insert(
            "content".into(),
            if text.is_empty() {
                Value::Null
            } else {
                Value::String(text)
            },
        );
        message.insert(
            "tool_calls".into(),
            Value::Array(
                tool_calls_by_index
                    .into_values()
                    .map(Value::Object)
                    .collect(),
            ),
        );
        Ok(Some(Turn {
            role: Some("assistant".into()),
            raw_messages: Some(vec![Value::Object(message)]),
            ..Turn::default()
        }))
    }

    fn captures_assistant_turn(&self) -> bool {
        true
    }
}

impl PreparedEndpointBehavior for ChatEndpoint {
    fn format_prepared_payload(
        &self,
        request: &PreparedRequest<'_>,
        endpoint: &RawEndpointConfig,
    ) -> EndpointResult<BodyPlan> {
        let turns = require_prepared_turns(request, "Chat endpoint requires at least one turn")?;
        let message_wires = format_chat_message_wires(request, turns)?;
        let last = turns.last().expect("non-empty turns");
        let mut payload = Map::new();
        // Reserved slot: the value is discarded, the key fixes the field's
        // insertion position, and `fill_reserved` supplies the real wires.
        payload.insert("messages".into(), Value::Null);
        payload.insert(
            "model".into(),
            Value::String(
                last.model
                    .clone()
                    .unwrap_or_else(|| request.primary_model_name().to_string()),
            ),
        );
        payload.insert("stream".into(), Value::Bool(endpoint.streaming));
        if let Some(tools) = latest_turn_attr(turns, |turn| turn.raw_tools.as_ref()) {
            payload.insert("tools".into(), Value::Array(tools.clone()));
        }
        if let Some(max_tokens) = last.max_tokens {
            payload.insert(
                if endpoint.use_legacy_max_tokens {
                    "max_tokens"
                } else {
                    "max_completion_tokens"
                }
                .into(),
                json!(max_tokens),
            );
        }
        merge_extra(&mut payload, endpoint.extra.as_ref());
        merge_extra(&mut payload, last.extra_body.as_ref());
        ensure_openai_stream_usage(
            &mut payload,
            endpoint.per_chunk_usage && endpoint.use_server_token_count,
        );
        build_reserved_plan(&payload, "messages", message_wires)
    }

    fn renders_all_turns(&self) -> bool {
        true
    }

    fn splices_lowered_wires(&self) -> bool {
        true
    }
}

impl Endpoint for ResponsesEndpoint {
    fn descriptor(&self) -> &'static EndpointDescriptor {
        &RESPONSES_DESCRIPTOR
    }

    fn format_payload(&self, request_info: &RequestInfo) -> EndpointResult<BodyPlan> {
        format_legacy_payload(self, request_info)
    }

    fn parse_response(&self, response: &ServerResponse) -> EndpointResult<Option<ParsedResponse>> {
        let Some(obj) = response.json.as_ref().and_then(Value::as_object) else {
            return Ok(None);
        };
        if obj.contains_key("type") {
            return Ok(parse_responses_streaming_event(obj, response.perf_ns));
        }
        if obj.get("object").and_then(Value::as_str) == Some("response") {
            let data = extract_responses_full_content(obj);
            let usage = non_empty_field(obj, "usage");
            return Ok(
                (data.is_some() || usage.is_some()).then_some(ParsedResponse {
                    perf_ns: response.perf_ns,
                    data,
                    usage,
                    sources: None,
                }),
            );
        }
        Ok(None)
    }

    /// Parse every response, de-duplicating the streamed output text.
    ///
    /// A streaming Responses turn carries the assistant text twice: once as
    /// the chain of `response.output_text.delta` events and again, in full,
    /// as the terminal `response.output_text.done` event. Tokenizing both
    /// doubles client-side output tokens (OSL / output-token-throughput
    /// ~2x). Once a delta has carried text for an output/content part, that
    /// part's terminal `done` is treated as a structural envelope. Tracking
    /// is keyed per `(output_index, content_index)` so a done-only part (a
    /// server that emits only the `done` event for that item, or deltas
    /// dropped under load) is NOT suppressed by a sibling part that did
    /// stream — it stays the sole text carrier and is still counted exactly
    /// once. The same holds for the non-streaming convenience field, which
    /// emits no deltas at all.
    ///
    /// The single forward pass is correct because a part's `done` event
    /// always trails its deltas in SSE arrival order, which
    /// `record.responses` preserves. `parse_response` itself is left
    /// emitting the `done` text: per-event callers (first-token detection,
    /// request latency) treat it as a plain data-bearing event and neither
    /// sums tokens, so they see no behavioral change.
    fn extract_response_data(&self, record: &RequestRecord) -> EndpointResult<Vec<ParsedResponse>> {
        let mut parsed = Vec::new();
        let mut streamed_parts: HashSet<(Option<i64>, Option<i64>)> = HashSet::new();
        for response in &record.responses {
            let Some(obj) = response.json.as_ref().and_then(Value::as_object) else {
                continue;
            };
            let event_type = obj.get("type").and_then(Value::as_str);
            let part = (
                obj.get("output_index").and_then(Value::as_i64),
                obj.get("content_index").and_then(Value::as_i64),
            );
            if event_type == Some("response.output_text.delta")
                && obj
                    .get("delta")
                    .and_then(Value::as_str)
                    .is_some_and(|s| !s.is_empty())
            {
                streamed_parts.insert(part);
            } else if event_type == Some("response.output_text.done")
                && streamed_parts.contains(&part)
            {
                // This part's deltas already carried the text: drop the
                // duplicate TEXT but keep an empty-text response at the
                // done timestamp so content-timing metrics (request_latency
                // uses the last content response's perf_ns; inter_chunk_latency
                // walks the gaps) are unchanged from the pre-dedup behavior.
                // Empty text contributes zero output tokens.
                parsed.push(ParsedResponse {
                    perf_ns: response.perf_ns,
                    data: Some(ResponseData::Text {
                        text: String::new(),
                    }),
                    usage: None,
                    sources: None,
                });
                continue;
            }
            if let Some(result) = self.parse_response(response)? {
                parsed.push(result);
            }
        }
        Ok(parsed)
    }

    fn extract_payload_inputs(&self, body: &Value) -> ExtractedPayload {
        let mut result = extract_inputs(body, &PartTypes::responses());
        let Some(obj) = body.as_object() else {
            return result;
        };
        match obj.get("instructions") {
            Some(Value::String(text)) if !text.is_empty() => {
                result.texts.insert(0, text.clone());
                if let Some(messages) = &mut result.messages {
                    messages.insert(0, json!({"role":"system","content":text}));
                }
            }
            Some(Value::Array(parts)) => {
                let mut collected = Vec::new();
                for part in parts {
                    if let Some(text) = part
                        .as_object()
                        .and_then(|obj| obj.get("text"))
                        .and_then(Value::as_str)
                        .filter(|s| !s.is_empty())
                    {
                        collected.push(text.to_string());
                    } else if let Some(text) = part.as_str().filter(|s| !s.is_empty()) {
                        collected.push(text.to_string());
                    }
                }
                for text in collected.iter().rev() {
                    result.texts.insert(0, text.clone());
                }
                if let Some(messages) = &mut result.messages
                    && !collected.is_empty()
                {
                    messages.insert(0, json!({"role":"system","content":collected.concat()}));
                }
            }
            _ => {}
        }
        result
    }

    fn build_assistant_turn(&self, record: &RequestRecord) -> EndpointResult<Option<Turn>> {
        let mut items_by_key = Map::new();
        let mut done_items = Vec::new();
        for response in &record.responses {
            let Some(obj) = response.json.as_ref().and_then(Value::as_object) else {
                continue;
            };
            if is_responses_failure_event(obj) {
                return build_plain_assistant_turn(self, record);
            }
            collect_response_items(obj, &mut items_by_key, &mut done_items);
        }
        for item in done_items {
            merge_response_item(&mut items_by_key, item);
        }
        if items_by_key.is_empty() {
            return build_plain_assistant_turn(self, record);
        }
        Ok(Some(Turn {
            role: Some("assistant".into()),
            raw_messages: Some(items_by_key.into_values().collect()),
            ..Turn::default()
        }))
    }

    fn part_types(&self) -> PartTypes {
        PartTypes::responses()
    }

    fn captures_assistant_turn(&self) -> bool {
        true
    }
}

impl PreparedEndpointBehavior for ResponsesEndpoint {
    fn websocket_capabilities(&self) -> Option<WebSocketCapabilities> {
        Some(WebSocketCapabilities {
            dialect: WebSocketDialect::Responses,
            connection_model: WebSocketConnectionModel::TurnSerialized,
            application_opcode: PreparedWsOpcode::Text,
            has_affinity_state: true,
            supports_full_history_replay: true,
            supports_http_sse_fallback: true,
            supports_round_trip_metrics: true,
        })
    }

    fn format_prepared_payload(
        &self,
        request: &PreparedRequest<'_>,
        endpoint: &RawEndpointConfig,
    ) -> EndpointResult<BodyPlan> {
        let turns =
            require_prepared_turns(request, "Responses endpoint requires at least one turn")?;
        let last = turns.last().expect("non-empty turns");
        let input_wires = format_responses_input_wires(request, turns)?;
        let mut payload = Map::new();
        // Reserved slot: the value is discarded, the key fixes the field's
        // insertion position, and `fill_reserved` supplies the real wires.
        payload.insert("input".into(), Value::Null);
        payload.insert(
            "model".into(),
            Value::String(
                last.model
                    .clone()
                    .unwrap_or_else(|| request.primary_model_name().to_string()),
            ),
        );
        payload.insert("stream".into(), Value::Bool(endpoint.streaming));
        if let Some(system) = request.system_message().filter(|value| !value.is_empty()) {
            payload.insert("instructions".into(), Value::String(system.to_string()));
        }
        if let Some(max_tokens) = last.max_tokens {
            payload.insert("max_output_tokens".into(), json!(max_tokens));
        }
        if let Some(tools) = latest_turn_attr(turns, |turn| turn.raw_tools.as_ref()) {
            payload.insert("tools".into(), Value::Array(tools.clone()));
        }
        merge_extra(&mut payload, endpoint.extra.as_ref());
        merge_extra(&mut payload, last.extra_body.as_ref());
        if endpoint.streaming && endpoint.use_server_token_count {
            ensure_include_usage(&mut payload);
        }
        build_reserved_plan(&payload, "input", input_wires)
    }

    fn prepare_ws_operation(
        &self,
        _request: &PreparedRequest<'_>,
        _endpoint: &RawEndpointConfig,
        body: &BodyPlan,
        store: &dyn SegmentStore,
        overrides: &Overrides,
    ) -> EndpointResult<PreparedWsOperation> {
        if body.has_field("type") || overrides.fields().contains_key("type") {
            return Err(EndpointError::InvalidRequest(
                "Responses `type` is a reserved WebSocket event field".to_owned(),
            ));
        }
        let mut fallback_plan = body.clone();
        fallback_plan.merge_overrides(overrides);
        fallback_plan.set_literal("stream", Value::Bool(true));
        if fallback_plan.literal_field("stream") != Some(&Value::Bool(true)) {
            return Err(EndpointError::InvalidRequest(
                "Responses WebSocket fallback requires a named-field request body".to_owned(),
            ));
        }
        let fallback_body =
            JsonBodyMaterializer::materialize(&fallback_plan, store, &Overrides::new())
                .map_err(|error| EndpointError::Serialization(error.to_string()))?;
        let body = JsonBodyMaterializer::materialize(body, store, overrides)
            .map_err(|error| EndpointError::Serialization(error.to_string()))?;
        let fields = body
            .strip_prefix(b"{")
            .and_then(|body| body.strip_suffix(b"}"))
            .ok_or_else(|| {
                EndpointError::Serialization(
                    "Responses WebSocket request body must be a JSON object".to_owned(),
                )
            })?;
        let mut event = Vec::with_capacity(fields.len() + 27);
        event.extend_from_slice(b"{\"type\":\"response.create\"");
        if !fields.is_empty() {
            event.push(b',');
            event.extend_from_slice(fields);
        }
        event.push(b'}');
        Ok(PreparedWsOperation::new(
            [PreparedWsMessage::text(
                Bytes::from(event),
                PreparedWsMessageRole::MeasuredInput,
            )],
            Some(fallback_body),
        )
        .with_input_projection(body))
    }

    fn renders_all_turns(&self) -> bool {
        true
    }

    fn splices_lowered_wires(&self) -> bool {
        true
    }
}

impl Endpoint for RealtimeEndpoint {
    fn descriptor(&self) -> &'static EndpointDescriptor {
        &REALTIME_DESCRIPTOR
    }

    fn format_payload(&self, request_info: &RequestInfo) -> EndpointResult<BodyPlan> {
        ResponsesEndpoint.format_payload(request_info)
    }

    fn parse_response(&self, response: &ServerResponse) -> EndpointResult<Option<ParsedResponse>> {
        let Some(object) = response.json.as_ref().and_then(Value::as_object) else {
            return Ok(None);
        };
        let data = match object.get("type").and_then(Value::as_str) {
            Some("response.output_text.delta") => object
                .get("delta")
                .and_then(Value::as_str)
                .filter(|text| !text.is_empty())
                .map(|text| ResponseData::Text {
                    text: text.to_owned(),
                }),
            Some("response.output_audio.delta") => {
                let encoded = object
                    .get("delta")
                    .or_else(|| object.get("audio"))
                    .and_then(Value::as_str)
                    .ok_or_else(|| {
                        EndpointError::InvalidResponse(
                            "Realtime audio delta has no base64 payload".to_owned(),
                        )
                    })?;
                let audio_bytes = STANDARD.decode(encoded).map_err(|error| {
                    EndpointError::InvalidResponse(format!(
                        "Realtime audio delta is not valid base64: {error}"
                    ))
                })?;
                let duration_ms = u64::try_from(audio_bytes.len()).ok().map(|bytes| {
                    // Realtime's default PCM output is mono signed 16-bit at 24 kHz.
                    bytes as f64 * 1000.0 / (2.0 * 24_000.0)
                });
                Some(ResponseData::Audio(AudioResponseData {
                    audio_bytes,
                    sample_rate_hz: 24_000,
                    encoding: "pcm16".to_owned(),
                    duration_ms,
                }))
            }
            _ => None,
        };
        Ok(data.map(|data| ParsedResponse {
            perf_ns: response.perf_ns,
            data: Some(data),
            usage: None,
            sources: None,
        }))
    }

    fn extract_payload_inputs(&self, body: &Value) -> ExtractedPayload {
        ResponsesEndpoint.extract_payload_inputs(body)
    }

    fn part_types(&self) -> PartTypes {
        PartTypes::responses()
    }

    fn captures_assistant_turn(&self) -> bool {
        true
    }
}

impl PreparedEndpointBehavior for RealtimeEndpoint {
    fn websocket_capabilities(&self) -> Option<WebSocketCapabilities> {
        Some(WebSocketCapabilities {
            dialect: WebSocketDialect::Realtime,
            connection_model: WebSocketConnectionModel::Duplex,
            application_opcode: PreparedWsOpcode::Text,
            has_affinity_state: true,
            supports_full_history_replay: false,
            supports_http_sse_fallback: false,
            supports_round_trip_metrics: true,
        })
    }

    fn format_prepared_payload(
        &self,
        request: &PreparedRequest<'_>,
        endpoint: &RawEndpointConfig,
    ) -> EndpointResult<BodyPlan> {
        ResponsesEndpoint.format_prepared_payload(request, endpoint)
    }

    fn prepare_ws_operation(
        &self,
        _request: &PreparedRequest<'_>,
        _endpoint: &RawEndpointConfig,
        body: &BodyPlan,
        store: &dyn SegmentStore,
        overrides: &Overrides,
    ) -> EndpointResult<PreparedWsOperation> {
        let body = JsonBodyMaterializer::materialize(body, store, overrides)
            .map_err(|error| EndpointError::Serialization(error.to_string()))?;
        let mut request: Value = serde_json::from_slice(&body)
            .map_err(|error| EndpointError::Serialization(error.to_string()))?;
        let object = request.as_object_mut().ok_or_else(|| {
            EndpointError::Serialization("Realtime request body must be a JSON object".to_owned())
        })?;
        let input = object.remove("input").ok_or_else(|| {
            EndpointError::Serialization("Realtime request body has no input items".to_owned())
        })?;
        let items = input.as_array().ok_or_else(|| {
            EndpointError::Serialization("Realtime request input must be an array".to_owned())
        })?;
        let current_start = items
            .iter()
            .rposition(|item| item.get("role").and_then(Value::as_str) == Some("assistant"))
            .map_or(0, |index| index + 1);
        let mut messages = Vec::with_capacity(items.len().saturating_sub(current_start) + 2);
        for item in &items[current_start..] {
            let item = item.as_object().ok_or_else(|| {
                EndpointError::InvalidRequest(
                    "Realtime request input items must be JSON objects".to_owned(),
                )
            })?;
            let role = item.get("role").and_then(Value::as_str).unwrap_or("user");
            if !matches!(role, "user" | "system") {
                return Err(EndpointError::InvalidRequest(format!(
                    "Realtime live input role {role:?} is not supported"
                )));
            }
            let content = item.get("content").ok_or_else(|| {
                EndpointError::InvalidRequest(
                    "Realtime request input item has no content".to_owned(),
                )
            })?;
            let parts: Vec<&Value> = match content {
                Value::Array(parts) => parts.iter().collect(),
                Value::String(_) => vec![content],
                _ => {
                    return Err(EndpointError::InvalidRequest(
                        "Realtime request item content must be text or an array".to_owned(),
                    ));
                }
            };
            let mut text_parts = Vec::new();
            let mut has_item_audio = false;
            for part in parts {
                if let Some(text) = part
                    .as_str()
                    .or_else(|| part.get("text").and_then(Value::as_str))
                {
                    if !text.is_empty() {
                        text_parts.push(json!({"type":"input_text","text":text}));
                    }
                    continue;
                }
                let Some(audio) = part.get("input_audio") else {
                    return Err(EndpointError::InvalidRequest(
                        "Realtime input supports only text and input_audio content".to_owned(),
                    ));
                };
                if role != "user" {
                    return Err(EndpointError::InvalidRequest(
                        "Realtime live audio input requires the user role".to_owned(),
                    ));
                }
                let encoded = audio.get("data").and_then(Value::as_str).ok_or_else(|| {
                    EndpointError::InvalidRequest(
                        "Realtime input_audio content has no base64 data".to_owned(),
                    )
                })?;
                let decoded = STANDARD.decode(encoded).map_err(|error| {
                    EndpointError::InvalidRequest(format!(
                        "Realtime input_audio data is not valid base64: {error}"
                    ))
                })?;
                if decoded.is_empty() {
                    return Err(EndpointError::InvalidRequest(
                        "Realtime input_audio data must not be empty".to_owned(),
                    ));
                }
                let event = json!({
                    "type":"input_audio_buffer.append",
                    "audio":STANDARD.encode(decoded),
                });
                messages.push(PreparedWsMessage::text(
                    Bytes::from(
                        serde_json::to_vec(&event)
                            .map_err(|error| EndpointError::Serialization(error.to_string()))?,
                    ),
                    PreparedWsMessageRole::MeasuredInput,
                ));
                has_item_audio = true;
            }
            if has_item_audio && !text_parts.is_empty() {
                return Err(EndpointError::InvalidRequest(
                    "Realtime mixed text and audio content ordering is unsupported".to_owned(),
                ));
            }
            if !text_parts.is_empty() {
                let event = json!({
                    "type":"conversation.item.create",
                    "item":{
                        "type":"message",
                        "role":role,
                        "content":text_parts,
                    }
                });
                messages.push(PreparedWsMessage::text(
                    Bytes::from(
                        serde_json::to_vec(&event)
                            .map_err(|error| EndpointError::Serialization(error.to_string()))?,
                    ),
                    PreparedWsMessageRole::MeasuredInput,
                ));
            }
            if has_item_audio {
                messages.push(PreparedWsMessage::text(
                    Bytes::from_static(br#"{"type":"input_audio_buffer.commit"}"#),
                    PreparedWsMessageRole::MeasuredInput,
                ));
            }
        }
        if messages.is_empty() {
            return Err(EndpointError::InvalidRequest(
                "Realtime request contains no non-empty text or audio input".to_owned(),
            ));
        }
        messages.push(PreparedWsMessage::text(
            Bytes::from(
                serde_json::to_vec(
                    &json!({"type":"response.create","response":{"modalities":["text","audio"]}}),
                )
                .map_err(|error| EndpointError::Serialization(error.to_string()))?,
            ),
            PreparedWsMessageRole::Control,
        ));
        let operation = PreparedWsOperation::new(messages, None).with_input_projection(body);
        Ok(if current_start > 0 {
            operation.requiring_affinity_state()
        } else {
            operation
        })
    }

    fn renders_all_turns(&self) -> bool {
        true
    }

    fn splices_lowered_wires(&self) -> bool {
        true
    }
}

impl Endpoint for CompletionsEndpoint {
    fn descriptor(&self) -> &'static EndpointDescriptor {
        &COMPLETIONS_DESCRIPTOR
    }

    fn format_payload(&self, request_info: &RequestInfo) -> EndpointResult<BodyPlan> {
        format_legacy_payload(self, request_info)
    }

    fn parse_response(&self, response: &ServerResponse) -> EndpointResult<Option<ParsedResponse>> {
        let Some(obj) = response.json.as_ref().and_then(Value::as_object) else {
            return Ok(None);
        };
        let data = match obj.get("object").and_then(Value::as_str) {
            Some("completion" | "text_completion") => first_choice(obj)
                .and_then(|choice| choice.get("text"))
                .and_then(Value::as_str)
                .filter(|s| !s.is_empty())
                .map(|text| ResponseData::Text {
                    text: text.to_string(),
                }),
            _ => None,
        };
        let usage = non_empty_field(obj, "usage");
        Ok(
            (data.is_some() || usage.is_some()).then_some(ParsedResponse {
                perf_ns: response.perf_ns,
                data,
                usage,
                sources: None,
            }),
        )
    }
}

impl PreparedEndpointBehavior for CompletionsEndpoint {
    fn format_prepared_payload(
        &self,
        request: &PreparedRequest<'_>,
        endpoint: &RawEndpointConfig,
    ) -> EndpointResult<BodyPlan> {
        if request.turns().len() != 1 {
            return Err(EndpointError::InvalidRequest(
                "Completions endpoint only supports one turn".into(),
            ));
        }
        let turn = &request.turns()[0];
        let mut prompts = turn_texts(turn);
        if request.credit_phase() == CreditPhase::Warmup {
            prompts = prompts
                .into_iter()
                .map(|prompt| format!("{WARMUP_SYSTEM_MESSAGE_PREFIX}\n{prompt}"))
                .collect();
        }
        let mut payload = Map::new();
        payload.insert(
            "prompt".into(),
            Value::Array(prompts.into_iter().map(Value::String).collect()),
        );
        payload.insert(
            "model".into(),
            Value::String(
                turn.model
                    .clone()
                    .unwrap_or_else(|| request.primary_model_name().to_string()),
            ),
        );
        payload.insert("stream".into(), Value::Bool(endpoint.streaming));
        if let Some(max_tokens) = turn.max_tokens {
            payload.insert("max_tokens".into(), json!(max_tokens));
        }
        merge_extra(&mut payload, endpoint.extra.as_ref());
        merge_extra(&mut payload, turn.extra_body.as_ref());
        ensure_openai_stream_usage(&mut payload, false);
        Ok(BodyPlan::from_object(&payload)?)
    }
}

impl Endpoint for EmbeddingsEndpoint {
    fn descriptor(&self) -> &'static EndpointDescriptor {
        &EMBEDDINGS_DESCRIPTOR
    }

    fn format_payload(&self, request_info: &RequestInfo) -> EndpointResult<BodyPlan> {
        format_legacy_payload(self, request_info)
    }

    fn parse_response(&self, response: &ServerResponse) -> EndpointResult<Option<ParsedResponse>> {
        parse_embeddings_response(response, true)
    }
}

impl PreparedEndpointBehavior for EmbeddingsEndpoint {
    fn format_prepared_payload(
        &self,
        request: &PreparedRequest<'_>,
        endpoint: &RawEndpointConfig,
    ) -> EndpointResult<BodyPlan> {
        if request.turns().len() != 1 {
            return Err(EndpointError::InvalidRequest(
                "Embeddings endpoint only supports one turn".into(),
            ));
        }
        let turn = &request.turns()[0];
        let mut payload = Map::new();
        payload.insert(
            "model".into(),
            Value::String(
                turn.model
                    .clone()
                    .unwrap_or_else(|| request.primary_model_name().to_string()),
            ),
        );
        payload.insert(
            "input".into(),
            Value::Array(turn_texts(turn).into_iter().map(Value::String).collect()),
        );
        merge_extra(&mut payload, endpoint.extra.as_ref());
        merge_extra(&mut payload, turn.extra_body.as_ref());
        Ok(BodyPlan::from_object(&payload)?)
    }
}

impl Endpoint for ChatEmbeddingsEndpoint {
    fn descriptor(&self) -> &'static EndpointDescriptor {
        &CHAT_EMBEDDINGS_DESCRIPTOR
    }

    fn format_payload(&self, request_info: &RequestInfo) -> EndpointResult<BodyPlan> {
        format_legacy_payload(self, request_info)
    }
    fn parse_response(&self, response: &ServerResponse) -> EndpointResult<Option<ParsedResponse>> {
        parse_embeddings_response(response, false)
    }
}

impl PreparedEndpointBehavior for ChatEmbeddingsEndpoint {
    fn format_prepared_payload(
        &self,
        request: &PreparedRequest<'_>,
        config: &RawEndpointConfig,
    ) -> EndpointResult<BodyPlan> {
        ChatEndpoint.format_prepared_payload(request, config)
    }

    fn renders_all_turns(&self) -> bool {
        true
    }

    fn splices_lowered_wires(&self) -> bool {
        true
    }
}

/// Build a message-array body plan: reserve `field` at the position its payload
/// key holds, then fill it with the assembled wires.
///
/// Both halves are fallible on purpose. Reserving a name the payload never
/// declared, and filling a name the plan never reserved, are the two ways the
/// old empty-array-placeholder convention shipped a body with no message array
/// and no error; here each is an [`EndpointError::InvalidRequest`] naming the
/// field.
pub(crate) fn build_reserved_plan(
    payload: &Map<String, Value>,
    field: &str,
    wires: SmallVec<[Bytes; 1]>,
) -> EndpointResult<BodyPlan> {
    let build = || -> crate::dataset::error::Result<BodyPlan> {
        let mut plan = BodyPlan::from_object_reserving(payload, &[field])?;
        plan.fill_reserved(field, wires)?;
        Ok(plan)
    };
    build().map_err(|error| EndpointError::InvalidRequest(error.to_string()))
}

pub(crate) fn require_prepared_turns<'a>(
    request: &'a PreparedRequest<'_>,
    message: &str,
) -> EndpointResult<&'a [Turn]> {
    if request.turns().is_empty() {
        Err(EndpointError::InvalidRequest(message.into()))
    } else {
        Ok(request.turns())
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum PartShape {
    Chat,
    Responses,
    Messages,
}

/// One assembled message in serialized or value form.
enum RenderedMessage {
    Wire(Bytes),
    Value(Value),
}

/// Assemble message-array wires while preserving load-time bytes.
///
/// `render_first` forces the first turn to render to a mutable value even when it
/// has serialized bytes, allowing warmup to prepend the system prompt.
fn rendered_turn_messages(
    turns: &[Turn],
    shape: PartShape,
    render_first: bool,
) -> EndpointResult<Vec<RenderedMessage>> {
    let mut out = Vec::new();
    for (index, turn) in turns.iter().enumerate() {
        if let Some(lowered) = &turn.lowered
            && !(render_first && index == 0)
        {
            out.extend(lowered.iter().cloned().map(RenderedMessage::Wire));
            continue;
        }
        match shape {
            PartShape::Responses => {
                if let Some(raw_messages) = &turn.raw_messages {
                    for item in raw_messages {
                        if !item
                            .as_object()
                            .and_then(|obj| obj.get("type"))
                            .and_then(Value::as_str)
                            .is_some_and(is_replay_unsafe_output_item)
                        {
                            out.push(RenderedMessage::Value(item.clone()));
                        }
                    }
                } else {
                    let mut message = render_turn_message(turn, PartShape::Responses)?;
                    if let Value::Object(obj) = &mut message {
                        obj.insert("type".into(), Value::String("message".into()));
                    }
                    out.push(RenderedMessage::Value(message));
                }
            }
            PartShape::Chat | PartShape::Messages => {
                if let Some(raw_messages) = &turn.raw_messages {
                    out.extend(raw_messages.iter().cloned().map(RenderedMessage::Value));
                } else {
                    out.push(RenderedMessage::Value(render_turn_message(turn, shape)?));
                }
            }
        }
    }
    Ok(out)
}

/// Serialize an assembled message list to spliceable wires, cloning lowered wires
/// and serializing rendered values exactly once.
fn serialize_rendered_messages(
    messages: Vec<RenderedMessage>,
) -> EndpointResult<SmallVec<[Bytes; 1]>> {
    messages
        .into_iter()
        .map(|message| match message {
            RenderedMessage::Wire(wire) => Ok(wire),
            RenderedMessage::Value(value) => serde_json::to_vec(&value)
                .map(Bytes::from)
                .map_err(EndpointError::from),
        })
        .collect()
}

/// Whether Chat assembly's first non-empty message carries a `system` role.
/// Inspect lowered wires without parsing so a conversation prompt pays the
/// mutable-render cost only when it actually has to merge an authored system.
fn turns_first_is_system(turns: &[Turn]) -> bool {
    for turn in turns {
        if let Some(lowered) = turn.lowered.as_ref() {
            if let Some(wire) = lowered.first() {
                return wire.starts_with(br#"{"role":"system""#);
            }
            continue;
        }
        if let Some(raw_messages) = turn.raw_messages.as_ref() {
            if let Some(message) = raw_messages.first() {
                return message
                    .as_object()
                    .and_then(|object| object.get("role"))
                    .and_then(Value::as_str)
                    == Some("system");
            }
            continue;
        }
        return turn.role.as_deref() == Some("system");
    }
    false
}

/// Assemble Chat Completions `messages` wires with system and user-context prefixes.
fn format_chat_message_wires(
    request: &PreparedRequest<'_>,
    turns: &[Turn],
) -> EndpointResult<SmallVec<[Bytes; 1]>> {
    let system = request.system_message().filter(|value| !value.is_empty());
    let first_is_system = system.is_some() && turns_first_is_system(turns);
    // Warmup already resolves composed media and can re-render its first turn.
    // Profiling keeps lowered wires splice-only until an actual merge requires
    // parsing the one leading system wire below.
    let render_first = first_is_system && request.credit_phase() == CreditPhase::Warmup;
    let mut rendered = rendered_turn_messages(turns, PartShape::Chat, render_first)?;
    let mut out = Vec::new();
    if let Some(system) = system {
        if first_is_system {
            if let Some(first) = rendered.first_mut() {
                prepend_system_into_rendered(first, system)?;
            }
        } else {
            out.push(RenderedMessage::Value(
                json!({"role":"system","content":system}),
            ));
        }
    }
    if let Some(context) = request
        .user_context_message()
        .filter(|value| !value.is_empty())
    {
        out.push(RenderedMessage::Value(
            json!({"role":"user","content":context}),
        ));
    }
    out.extend(rendered);
    serialize_rendered_messages(out)
}

fn prepend_system_into_rendered(first: &mut RenderedMessage, system: &str) -> EndpointResult<()> {
    if let RenderedMessage::Wire(wire) = first {
        *first = RenderedMessage::Value(serde_json::from_slice(wire)?);
    }
    let RenderedMessage::Value(Value::Object(first)) = first else {
        return Err(EndpointError::InvalidRequest(
            "leading Chat system message must be a JSON object".into(),
        ));
    };
    prepend_system_into_object(first, system);
    Ok(())
}

/// Assemble Responses `input` wires with the user-context prefix.
fn format_responses_input_wires(
    request: &PreparedRequest<'_>,
    turns: &[Turn],
) -> EndpointResult<SmallVec<[Bytes; 1]>> {
    let mut out = Vec::new();
    if let Some(context) = request
        .user_context_message()
        .filter(|value| !value.is_empty())
    {
        out.push(RenderedMessage::Value(
            json!({"type": "message", "role": "user", "content": context}),
        ));
    }
    out.extend(rendered_turn_messages(turns, PartShape::Responses, false)?);
    serialize_rendered_messages(out)
}

/// Assemble Anthropic Messages `messages` wires with the user-context prefix.
pub(crate) fn format_messages_array_wires(
    request: &PreparedRequest<'_>,
    turns: &[Turn],
) -> EndpointResult<SmallVec<[Bytes; 1]>> {
    let mut out = Vec::new();
    if let Some(context) = request
        .user_context_message()
        .filter(|value| !value.is_empty())
    {
        out.push(RenderedMessage::Value(
            json!({"role":"user","content":context}),
        ));
    }
    out.extend(rendered_turn_messages(turns, PartShape::Messages, false)?);
    serialize_rendered_messages(out)
}

/// Load-time content-to-wire lowering seam.
///
/// Each result must equal the dispatch serializer's bytes, including Responses
/// discriminants and replay filters.
pub trait TurnMessageLowerer: Send + Sync {
    /// Render and serialize one turn's message wire(s) exactly as the dispatch
    /// message-array formatter would emit them for this turn in isolation.
    fn lower_turn(&self, turn: &Turn) -> EndpointResult<SmallVec<[Bytes; 1]>>;
}

/// The built-in [`TurnMessageLowerer`] over the three message-array part shapes.
#[derive(Debug, Clone, Copy)]
pub struct ShapeLowerer {
    shape: PartShape,
}

impl ShapeLowerer {
    /// Select the lowerer for a registered endpoint's **canonical** descriptor
    /// id, or `None` for dialects whose body is not a per-turn message array
    /// (embeddings, completions, rankings, media, …) and therefore is never
    /// lowered.
    ///
    /// Aliases are not matched: every caller passes `descriptor().id`, and an
    /// arm for one (`chat_completions`, an alias of `chat`) was unreachable.
    /// The dialects answering `Some` here are exactly those whose
    /// [`PreparedEndpoint::splices_lowered_wires`](crate::endpoints::PreparedEndpoint::splices_lowered_wires)
    /// is `true`; `lowerable_dialects_declare_that_they_splice_lowered_wires`
    /// holds the two in agreement.
    pub fn for_descriptor_id(id: &str) -> Option<Self> {
        let shape = match id {
            "chat" | "chat_embeddings" => PartShape::Chat,
            "realtime" | "responses" => PartShape::Responses,
            "messages" => PartShape::Messages,
            _ => return None,
        };
        Some(Self { shape })
    }
}

impl TurnMessageLowerer for ShapeLowerer {
    fn lower_turn(&self, turn: &Turn) -> EndpointResult<SmallVec<[Bytes; 1]>> {
        // Preformatted items are preserved except for replay-unsafe Responses
        // output; rendered Responses messages receive `type:"message"`.
        let values: Vec<Value> = match self.shape {
            PartShape::Responses => {
                if let Some(raw_messages) = &turn.raw_messages {
                    raw_messages
                        .iter()
                        .filter(|item| {
                            !item
                                .as_object()
                                .and_then(|obj| obj.get("type"))
                                .and_then(Value::as_str)
                                .is_some_and(is_replay_unsafe_output_item)
                        })
                        .cloned()
                        .collect()
                } else {
                    let mut message = render_turn_message(turn, PartShape::Responses)?;
                    if let Value::Object(obj) = &mut message {
                        obj.insert("type".into(), Value::String("message".into()));
                    }
                    vec![message]
                }
            }
            PartShape::Chat | PartShape::Messages => {
                if let Some(raw_messages) = &turn.raw_messages {
                    raw_messages.clone()
                } else {
                    vec![render_turn_message(turn, self.shape)?]
                }
            }
        };
        values
            .iter()
            .map(|value| serde_json::to_vec(value).map(Bytes::from))
            .collect::<std::result::Result<SmallVec<[Bytes; 1]>, _>>()
            .map_err(EndpointError::from)
    }
}

fn render_turn_message(turn: &Turn, shape: PartShape) -> EndpointResult<Value> {
    let role = turn
        .role
        .as_deref()
        .filter(|role| !role.is_empty())
        .unwrap_or("user");
    Ok(json!({"role": role, "content": render_turn_content(turn, shape)?}))
}

fn render_turn_content(turn: &Turn, shape: PartShape) -> EndpointResult<Value> {
    if !FORCE_CONTENT_PARTS.get().copied().unwrap_or(false)
        && turn.texts.len() == 1
        && turn.texts[0].contents.len() == 1
        && turn.images.is_empty()
        && turn.audios.is_empty()
        && turn.videos.is_empty()
    {
        return Ok(Value::String(turn.texts[0].contents[0].clone()));
    }
    let mut parts = Vec::new();
    extend_parts(&mut parts, &turn.texts, |content| {
        render_text_part(content, shape)
    });
    if matches!(shape, PartShape::Chat) {
        extend_chat_image_parts(&mut parts, &turn.images);
    } else {
        extend_parts(&mut parts, &turn.images, |content| {
            render_image_part(content, shape)
        });
    }
    for media in &turn.audios {
        for content in &media.contents {
            if !content.is_empty() {
                parts.push(render_audio_part(content, shape)?);
            }
        }
    }
    for media in &turn.videos {
        for content in &media.contents {
            if !content.is_empty() {
                parts.push(render_video_part(content, shape)?);
            }
        }
    }
    Ok(Value::Array(parts))
}

fn extend_parts<F>(parts: &mut Vec<Value>, media_items: &[Media], mut render: F)
where
    F: FnMut(&str) -> Value,
{
    for media in media_items {
        for content in &media.contents {
            if !content.is_empty() {
                parts.push(render(content));
            }
        }
    }
}

/// Append chat image parts, always passing through authored cache UUIDs.
///
/// A `Media` with `uuids` set always contributes one `image_url` part per
/// content/uuid pair, even when content is empty (a cache-only reference
/// the server resolves by UUID). `Media` without `uuids` keeps the generic
/// skip-empty behavior. Chat-endpoint-only: mirrors Python's
/// `ChatEndpoint._extend_image_parts` override.
fn extend_chat_image_parts(parts: &mut Vec<Value>, media_items: &[Media]) {
    for media in media_items {
        if media.uuids.is_empty() {
            for content in &media.contents {
                if !content.is_empty() {
                    parts.push(render_image_part(content, PartShape::Chat));
                }
            }
            continue;
        }
        for (content, uuid) in media.contents.iter().zip(&media.uuids) {
            parts.push(json!({
                "type": "image_url",
                "image_url": {"url": content},
                "uuid": uuid,
            }));
        }
    }
}

fn render_text_part(text: &str, shape: PartShape) -> Value {
    match shape {
        PartShape::Chat | PartShape::Messages => json!({"type":"text","text":text}),
        PartShape::Responses => json!({"type":"input_text","text":text}),
    }
}
fn render_image_part(url: &str, shape: PartShape) -> Value {
    match shape {
        PartShape::Chat => json!({"type":"image_url","image_url":{"url":url}}),
        PartShape::Responses => json!({"type":"input_image","image_url":url}),
        PartShape::Messages if url.starts_with("data:") => {
            let (header, data) = url.split_once(',').unwrap_or((url, ""));
            let media_type = header
                .strip_prefix("data:")
                .and_then(|value| value.split(';').next())
                .filter(|value| !value.is_empty())
                .unwrap_or("image/png");
            json!({"type":"image","source":{"type":"base64","media_type":media_type,"data":data}})
        }
        PartShape::Messages => json!({"type":"image","source":{"type":"url","url":url}}),
    }
}
fn render_audio_part(format_and_b64: &str, shape: PartShape) -> EndpointResult<Value> {
    if matches!(shape, PartShape::Messages) {
        return Err(EndpointError::InvalidRequest(
            "Anthropic Messages API does not support audio input. Use a different endpoint, or remove audio content from the turn."
                .into(),
        ));
    }
    let Some((left, b64)) = format_and_b64.split_once(',') else {
        return Err(EndpointError::InvalidRequest(
            "audio content must be in the format 'format,b64_audio'".into(),
        ));
    };
    let format = left
        .strip_prefix("data:audio/")
        .and_then(|rest| rest.split_once(';').map(|(fmt, _)| fmt))
        .filter(|fmt| !fmt.is_empty())
        .unwrap_or(left);
    Ok(json!({"type":"input_audio","input_audio":{"data":b64,"format":format}}))
}
fn render_video_part(url: &str, shape: PartShape) -> EndpointResult<Value> {
    match shape {
        PartShape::Chat => Ok(json!({"type":"video_url","video_url":{"url":url}})),
        PartShape::Responses => Err(EndpointError::InvalidRequest(
            "Responses API does not support video input".into(),
        )),
        PartShape::Messages => Err(EndpointError::InvalidRequest(
            "Anthropic Messages API does not support video input. Use a different endpoint, or remove video content from the turn."
                .into(),
        )),
    }
}

fn prepend_system_into_object(first: &mut Map<String, Value>, system: &str) {
    match first.get_mut("content") {
        Some(Value::String(content)) if content.is_empty() => *content = system.to_string(),
        Some(Value::String(content)) => *content = format!("{system}\n\n{content}"),
        Some(Value::Array(parts)) => parts.insert(0, json!({"type":"text","text":system})),
        Some(Value::Null) | None => {
            first.insert("content".into(), Value::String(system.to_string()));
        }
        Some(other) => *other = Value::String(system.to_string()),
    }
}

pub(crate) fn latest_turn_attr<'a, T, F>(turns: &'a [Turn], get: F) -> Option<&'a T>
where
    F: Fn(&'a Turn) -> Option<&'a T>,
{
    turns.iter().rev().find_map(get)
}

pub(crate) fn merge_extra(payload: &mut Map<String, Value>, extra: Option<&Map<String, Value>>) {
    if let Some(extra) = extra {
        for (key, value) in extra {
            payload.insert(key.clone(), value.clone());
        }
    }
}
fn ensure_include_usage(payload: &mut Map<String, Value>) {
    match payload.get_mut("stream_options") {
        Some(Value::Object(stream_options)) => {
            stream_options
                .entry("include_usage")
                .or_insert(Value::Bool(true));
        }
        Some(_) | None => {
            payload.insert("stream_options".into(), json!({"include_usage": true}));
        }
    }
}

fn ensure_openai_stream_usage(payload: &mut Map<String, Value>, continuous: bool) {
    if payload.get("stream") != Some(&Value::Bool(true)) {
        return;
    }
    match payload.get_mut("stream_options") {
        Some(Value::Object(stream_options)) => {
            stream_options
                .entry("include_usage")
                .or_insert(Value::Bool(true));
            if continuous {
                stream_options
                    .entry("continuous_usage_stats")
                    .or_insert(Value::Bool(true));
            }
        }
        Some(Value::Null) | None => {
            let mut stream_options =
                Map::from_iter([("include_usage".to_owned(), Value::Bool(true))]);
            if continuous {
                stream_options.insert("continuous_usage_stats".to_owned(), Value::Bool(true));
            }
            payload.insert("stream_options".into(), Value::Object(stream_options));
        }
        Some(_) => {}
    }
}

fn first_choice(obj: &Map<String, Value>) -> Option<&Map<String, Value>> {
    obj.get("choices")
        .and_then(Value::as_array)
        .and_then(|choices| choices.first())
        .and_then(Value::as_object)
}
pub(crate) fn non_empty_field(obj: &Map<String, Value>, field: &str) -> Option<Value> {
    match obj.get(field) {
        Some(Value::Null) | None => None,
        Some(Value::Object(map)) if map.is_empty() => None,
        Some(value) => Some(value.clone()),
    }
}

/// Visible to `chat_chunk`'s differential test, which asserts the typed
/// streaming fast path yields exactly this function's result.
pub(crate) fn extract_chat_response_data(obj: &Map<String, Value>) -> Option<ResponseData> {
    let key = match obj.get("object").and_then(Value::as_str) {
        Some("chat.completion") => "message",
        Some("chat.completion.chunk") => "delta",
        _ => return None,
    };
    let data = first_choice(obj)?.get(key)?.as_object()?;
    let content = data
        .get("content")
        .and_then(Value::as_str)
        .map(ToString::to_string);
    if let Some(reasoning) = data
        .get("reasoning_content")
        .or_else(|| data.get("reasoning"))
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
    {
        return Some(ResponseData::Reasoning {
            content,
            reasoning: reasoning.to_string(),
        });
    }
    let mut parts = Vec::new();
    if let Some(tool_calls) = data.get("tool_calls").and_then(Value::as_array) {
        for tc in tool_calls {
            let Some(function) = tc
                .as_object()
                .and_then(|tc| tc.get("function"))
                .and_then(Value::as_object)
            else {
                continue;
            };
            if let Some(name) = function
                .get("name")
                .and_then(Value::as_str)
                .filter(|s| !s.is_empty())
            {
                parts.push(name.to_string());
            }
            if let Some(arguments) = function
                .get("arguments")
                .and_then(Value::as_str)
                .filter(|s| !s.is_empty())
            {
                parts.push(arguments.to_string());
            }
        }
    }
    let tool_call_text = parts.concat();
    if !tool_call_text.is_empty() {
        return Some(ResponseData::ToolCall {
            tool_call_text,
            content: content.filter(|s| !s.is_empty()),
        });
    }
    content
        .filter(|s| !s.is_empty())
        .map(|text| ResponseData::Text { text })
}

fn absorb_chat_choice(
    object_type: Option<&str>,
    choice: &Map<String, Value>,
    content_parts: &mut Vec<String>,
    tool_calls_by_index: &mut BTreeMap<i64, Map<String, Value>>,
) {
    match object_type {
        Some("chat.completion") => {
            let Some(message) = choice.get("message").and_then(Value::as_object) else {
                return;
            };
            if let Some(content) = message.get("content").and_then(Value::as_str) {
                content_parts.push(content.to_string());
            }
            if let Some(tool_calls) = message.get("tool_calls").and_then(Value::as_array) {
                for tc in tool_calls {
                    let Some(tc_obj) = tc.as_object() else {
                        continue;
                    };
                    let idx = tc_obj
                        .get("index")
                        .and_then(Value::as_i64)
                        .unwrap_or(tool_calls_by_index.len() as i64);
                    let mut cloned = tc_obj.clone();
                    cloned.remove("index");
                    tool_calls_by_index.insert(idx, cloned);
                }
            }
            absorb_legacy_function_call(message.get("function_call"), tool_calls_by_index);
        }
        Some("chat.completion.chunk") => {
            let Some(delta) = choice.get("delta").and_then(Value::as_object) else {
                return;
            };
            if let Some(content) = delta.get("content").and_then(Value::as_str) {
                content_parts.push(content.to_string());
            }
            if let Some(tool_calls) = delta.get("tool_calls").and_then(Value::as_array) {
                for tc in tool_calls {
                    if let Some(tc) = tc.as_object() {
                        merge_tool_call_delta(tc, tool_calls_by_index);
                    }
                }
            }
            merge_legacy_function_call_delta(delta.get("function_call"), tool_calls_by_index);
        }
        _ => {}
    }
}

fn absorb_legacy_function_call(
    value: Option<&Value>,
    tool_calls_by_index: &mut BTreeMap<i64, Map<String, Value>>,
) {
    let Some(function_call) = value.and_then(Value::as_object) else {
        return;
    };
    let idx = tool_calls_by_index.len() as i64;
    tool_calls_by_index.insert(idx, json!({"type":"function","function":{"name":function_call.get("name").and_then(Value::as_str).unwrap_or(""),"arguments":function_call.get("arguments").and_then(Value::as_str).unwrap_or("")}}).as_object().unwrap().clone());
}
fn merge_legacy_function_call_delta(
    value: Option<&Value>,
    tool_calls_by_index: &mut BTreeMap<i64, Map<String, Value>>,
) {
    let Some(delta) = value.and_then(Value::as_object) else {
        return;
    };
    let existing = tool_calls_by_index.entry(0).or_insert_with(|| {
        json!({"type":"function","function":{}})
            .as_object()
            .unwrap()
            .clone()
    });
    existing
        .entry("type")
        .or_insert_with(|| Value::String("function".into()));
    let function = ensure_object_field(existing, "function");
    if let Some(name) = delta
        .get("name")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
    {
        concat_string_field(function, "name", name);
    }
    if let Some(arguments) = delta.get("arguments") {
        concat_string_field(function, "arguments", arguments.as_str().unwrap_or(""));
    }
}
fn merge_tool_call_delta(
    delta: &Map<String, Value>,
    tool_calls_by_index: &mut BTreeMap<i64, Map<String, Value>>,
) {
    let idx = delta
        .get("index")
        .and_then(Value::as_i64)
        .unwrap_or(tool_calls_by_index.len() as i64);
    let existing = tool_calls_by_index.entry(idx).or_default();
    if let Some(id) = delta
        .get("id")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
    {
        existing.insert("id".into(), Value::String(id.into()));
    }
    if let Some(ty) = delta
        .get("type")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
    {
        existing.insert("type".into(), Value::String(ty.into()));
    }
    let Some(fn_delta) = delta.get("function").and_then(Value::as_object) else {
        return;
    };
    let function = ensure_object_field(existing, "function");
    if let Some(name) = fn_delta
        .get("name")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
    {
        function.insert("name".into(), Value::String(name.into()));
    }
    if let Some(arguments) = fn_delta.get("arguments") {
        concat_string_field(function, "arguments", arguments.as_str().unwrap_or(""));
    }
}
fn ensure_object_field<'a>(
    map: &'a mut Map<String, Value>,
    field: &str,
) -> &'a mut Map<String, Value> {
    if !matches!(map.get(field), Some(Value::Object(_))) {
        map.insert(field.into(), Value::Object(Map::new()));
    }
    map.get_mut(field).and_then(Value::as_object_mut).unwrap()
}
fn concat_string_field(map: &mut Map<String, Value>, field: &str, suffix: &str) {
    let mut value = map
        .get(field)
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    value.push_str(suffix);
    map.insert(field.into(), Value::String(value));
}

fn parse_responses_streaming_event(
    obj: &Map<String, Value>,
    perf_ns: u64,
) -> Option<ParsedResponse> {
    let event_type = obj.get("type").and_then(Value::as_str)?;
    let data = match event_type {
        "response.output_text.delta" => obj
            .get("delta")
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
            .map(|text| ResponseData::Text { text: text.into() }),
        "response.reasoning_text.delta" => obj
            .get("delta")
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
            .map(|reasoning| ResponseData::Reasoning {
                content: None,
                reasoning: reasoning.into(),
            }),
        "response.output_text.done" => obj
            .get("text")
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
            .map(|text| ResponseData::Text { text: text.into() }),
        "response.function_call_arguments.delta" => obj
            .get("delta")
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
            .map(|tool_call_text| ResponseData::ToolCall {
                tool_call_text: tool_call_text.into(),
                content: None,
            }),
        _ => None,
    };
    if data.is_some() {
        return Some(ParsedResponse {
            perf_ns,
            data,
            usage: None,
            sources: None,
        });
    }
    if event_type == "response.completed" {
        let usage = obj
            .get("response")
            .and_then(Value::as_object)
            .and_then(|response| non_empty_field(response, "usage"));
        if usage.is_some() {
            return Some(ParsedResponse {
                perf_ns,
                data: None,
                usage,
                sources: None,
            });
        }
    }
    None
}

fn extract_responses_full_content(obj: &Map<String, Value>) -> Option<ResponseData> {
    if let Some(output) = obj.get("output").and_then(Value::as_array) {
        let mut text_parts = Vec::new();
        let mut reasoning_parts = Vec::new();
        let mut tool_parts = Vec::new();
        for item in output {
            let Some(item) = item.as_object() else {
                continue;
            };
            match item.get("type").and_then(Value::as_str) {
                Some("reasoning") => collect_reasoning_summary(item, &mut reasoning_parts),
                Some("message") => collect_message_content(item, &mut text_parts),
                Some("function_call") => collect_function_call(item, &mut tool_parts),
                _ => {}
            }
        }
        if !reasoning_parts.is_empty() {
            return Some(ResponseData::Reasoning {
                content: (!text_parts.is_empty()).then(|| text_parts.concat()),
                reasoning: reasoning_parts.concat(),
            });
        }
        if !text_parts.is_empty() {
            return Some(ResponseData::Text {
                text: text_parts.concat(),
            });
        }
        if !tool_parts.is_empty() {
            return Some(ResponseData::ToolCall {
                tool_call_text: tool_parts.concat(),
                content: None,
            });
        }
    }
    obj.get("output_text")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .map(|text| ResponseData::Text { text: text.into() })
}
fn collect_reasoning_summary(item: &Map<String, Value>, out: &mut Vec<String>) {
    if let Some(summary) = item.get("summary").and_then(Value::as_array) {
        for part in summary {
            if let Some(part) = part.as_object()
                && part.get("type").and_then(Value::as_str) == Some("summary_text")
                && let Some(text) = part
                    .get("text")
                    .and_then(Value::as_str)
                    .filter(|s| !s.is_empty())
            {
                out.push(text.into());
            }
        }
    }
}
fn collect_message_content(item: &Map<String, Value>, out: &mut Vec<String>) {
    if let Some(content) = item.get("content").and_then(Value::as_array) {
        for part in content {
            if let Some(part) = part.as_object()
                && part.get("type").and_then(Value::as_str) == Some("output_text")
                && let Some(text) = part
                    .get("text")
                    .and_then(Value::as_str)
                    .filter(|s| !s.is_empty())
            {
                out.push(text.into());
            }
        }
    }
}
fn collect_function_call(item: &Map<String, Value>, out: &mut Vec<String>) {
    if let Some(name) = item
        .get("name")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
    {
        out.push(name.into());
    }
    if let Some(arguments) = item
        .get("arguments")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
    {
        out.push(arguments.into());
    }
}

fn is_replay_unsafe_output_item(ty: &str) -> bool {
    matches!(
        ty,
        "web_search_call"
            | "file_search_call"
            | "image_generation_call"
            | "code_interpreter_call"
            | "computer_call"
            | "reasoning"
    )
}
fn is_responses_failure_event(obj: &Map<String, Value>) -> bool {
    matches!(
        obj.get("type").and_then(Value::as_str),
        Some("response.failed" | "response.incomplete" | "response.error" | "error")
    )
}
fn collect_response_items(
    obj: &Map<String, Value>,
    items_by_key: &mut Map<String, Value>,
    done_items: &mut Vec<Value>,
) {
    if obj.get("object").and_then(Value::as_str) == Some("response") {
        merge_output_list(items_by_key, obj.get("output"));
        return;
    }
    match obj.get("type").and_then(Value::as_str) {
        Some("response.completed") => merge_output_list(
            items_by_key,
            obj.get("response")
                .and_then(Value::as_object)
                .and_then(|response| response.get("output")),
        ),
        Some("response.output_item.done") => {
            if let Some(item) = obj.get("item").filter(|item| item.is_object()) {
                done_items.push(item.clone());
            }
        }
        _ => {}
    }
}
fn merge_output_list(items_by_key: &mut Map<String, Value>, output: Option<&Value>) {
    if let Some(output) = output.and_then(Value::as_array) {
        for item in output {
            if item.is_object() {
                merge_response_item(items_by_key, item.clone());
            }
        }
    }
}
fn merge_response_item(items_by_key: &mut Map<String, Value>, item: Value) {
    let Some(obj) = item.as_object() else {
        return;
    };
    let key = obj
        .get("id")
        .or_else(|| obj.get("call_id"))
        .or_else(|| obj.get("item_id"))
        .and_then(Value::as_str)
        .map(ToString::to_string)
        .unwrap_or_else(|| {
            format!(
                "{}::{}",
                obj.get("type").and_then(Value::as_str).unwrap_or("?"),
                serde_json::to_string(obj).unwrap_or_default()
            )
        });
    items_by_key.entry(key).or_insert(item);
}

pub(crate) fn turn_texts(turn: &Turn) -> Vec<String> {
    turn.texts
        .iter()
        .flat_map(|text| text.contents.iter())
        .filter(|content| !content.is_empty())
        .cloned()
        .collect()
}

pub(crate) fn joined_text(turn: &Turn) -> String {
    turn_texts(turn).join(" ")
}

/// Apply the OpenAI-compatible bearer credential rule to one header set.
///
/// Single source for the header spelling shared by [`Endpoint::format_headers`],
/// [`bearer_headers`], and out-of-band requests built by
/// [`auth_headers_for_endpoint`].
pub(crate) fn apply_bearer_auth_header(headers: &mut BTreeMap<String, String>, api_key: &str) {
    headers.insert("Authorization".to_string(), format!("Bearer {api_key}"));
}

/// Clone an endpoint's configured headers, adding a bearer `Authorization`
/// header when an API key is present. Shared by every dialect that prepares
/// from a [`RawEndpointConfig`] (KServe, Riva, vLLM-generate, DynoSim).
pub(crate) fn bearer_headers(config: &RawEndpointConfig) -> BTreeMap<String, String> {
    let mut headers = config.headers.clone();
    if let Some(api_key) = &config.api_key {
        apply_bearer_auth_header(&mut headers, api_key);
    }
    headers
}

/// Build the headers an out-of-band request to an endpoint's origin must carry.
///
/// Endpoint-local control-plane hooks POST to the same origin as inference, so
/// an authenticated endpoint rejects them unless they authenticate the same way:
/// the authored custom headers — which can carry proprietary auth, gateway
/// routing, or tracing metadata — plus the dialect's credential rule. Content
/// negotiation is deliberately absent; a bodyless control POST needs none, and
/// the shared HTTP transport supplies its own.
pub fn auth_headers_for_endpoint(
    endpoint_id: &str,
    config: &RawEndpointConfig,
) -> BTreeMap<String, String> {
    let mut headers = config.headers.clone();
    // An empty authored key is no key, matching the Messages dialect's filter.
    let api_key = config.api_key.as_deref().filter(|key| !key.is_empty());
    if matches!(
        EndpointType::from_canonical_id(endpoint_id),
        Some(EndpointType::Messages)
    ) {
        apply_messages_auth_headers(&mut headers, api_key);
    } else if let Some(api_key) = api_key {
        apply_bearer_auth_header(&mut headers, api_key);
    }
    headers
}

pub(crate) fn parse_embeddings_response(
    response: &ServerResponse,
    strict_invalid_data: bool,
) -> EndpointResult<Option<ParsedResponse>> {
    let Some(obj) = response.json.as_ref().and_then(Value::as_object) else {
        return Ok(None);
    };
    let Some(data) = obj.get("data") else {
        if !strict_invalid_data && let Some(embeddings) = try_extract_embeddings(obj) {
            return Ok(Some(ParsedResponse {
                perf_ns: response.perf_ns,
                data: Some(ResponseData::Embeddings { embeddings }),
                usage: None,
                sources: None,
            }));
        }
        return Ok(None);
    };
    let data_is_falsy = match data {
        Value::Null => true,
        Value::Bool(value) => !value,
        Value::Number(value) => value.as_f64().is_some_and(|value| value == 0.0),
        Value::String(value) => value.is_empty(),
        Value::Array(value) => value.is_empty(),
        Value::Object(value) => value.is_empty(),
    };
    if data_is_falsy {
        return Ok(None);
    }
    let Some(data_array) = data.as_array() else {
        return if strict_invalid_data {
            Err(EndpointError::InvalidResponse(format!(
                "received invalid list in response: {}",
                Value::Object(obj.clone())
            )))
        } else {
            Ok(None)
        };
    };
    if data_array.is_empty() {
        return Ok(None);
    }
    if data_array.iter().all(|item| {
        item.as_object()
            .and_then(|obj| obj.get("object"))
            .and_then(Value::as_str)
            == Some("embedding")
    }) {
        let embeddings: Vec<Vec<f64>> = data_array
            .iter()
            .filter_map(|item| item.as_object()?.get("embedding"))
            .filter_map(number_array)
            .collect();
        if embeddings.is_empty() {
            return Ok(None);
        }
        return Ok(Some(ParsedResponse {
            perf_ns: response.perf_ns,
            data: Some(ResponseData::Embeddings { embeddings }),
            usage: None,
            sources: None,
        }));
    }
    if strict_invalid_data {
        Err(EndpointError::InvalidResponse(format!(
            "received invalid list in response: {}",
            Value::Object(obj.clone())
        )))
    } else {
        Ok(None)
    }
}
pub(crate) fn try_extract_embeddings(obj: &Map<String, Value>) -> Option<Vec<Vec<f64>>> {
    for field in ["embeddings", "embedding"] {
        let Some(value) = obj.get(field) else {
            continue;
        };
        if let Some(numbers) = number_array(value) {
            return Some(vec![numbers]);
        }
        if let Some(arrays) = value.as_array() {
            let embeddings: Vec<Vec<f64>> = arrays.iter().filter_map(number_array).collect();
            if !embeddings.is_empty() {
                return Some(embeddings);
            }
        }
    }
    None
}
pub(crate) fn number_array(value: &Value) -> Option<Vec<f64>> {
    let array = value.as_array()?;
    if array.is_empty() {
        return None;
    }
    let mut out = Vec::with_capacity(array.len());
    for item in array {
        out.push(item.as_f64()?);
    }
    Some(out)
}

pub(crate) fn build_plain_assistant_turn<E: Endpoint + ?Sized>(
    endpoint: &E,
    record: &RequestRecord,
) -> EndpointResult<Option<Turn>> {
    let mut output_texts = Vec::new();
    for parsed in endpoint.extract_response_data(record)? {
        let Some(data) = parsed.data else {
            continue;
        };
        match data {
            ResponseData::Reasoning {
                content: Some(content),
                ..
            } => output_texts.push(content),
            ResponseData::Reasoning { reasoning, .. } if !reasoning.is_empty() => {
                output_texts.push(reasoning)
            }
            other => {
                let text = other.get_text();
                if !text.is_empty() {
                    output_texts.push(text);
                }
            }
        }
    }
    let text = output_texts.concat();
    Ok((!text.is_empty()).then_some(Turn {
        role: Some("assistant".into()),
        texts: vec![Media::new(vec![text])],
        ..Turn::default()
    }))
}

#[cfg(test)]
mod lowering_tests {
    use super::*;
    use crate::dataset::segment::SegmentPool;
    use crate::endpoints::registry::PreparedEndpointBehavior;

    fn text_turn() -> Turn {
        Turn {
            role: Some("user".into()),
            texts: vec![Media::new(vec!["hello".to_string()])],
            ..Turn::default()
        }
    }

    fn multimodal_turn(image: &str) -> Turn {
        Turn {
            role: Some("user".into()),
            texts: vec![Media::new(vec!["look".to_string()])],
            images: vec![Media::new(vec![image.to_string()])],
            ..Turn::default()
        }
    }

    fn openai_bodies(turn: Turn, endpoint: &RawEndpointConfig) -> [Value; 2] {
        let turns = [turn];
        let request = PreparedRequest::new(
            "gpt-test",
            &turns,
            None,
            None,
            CreditPhase::Profiling,
            None,
            None,
            None,
        );
        let materialize = |body: BodyPlan| {
            serde_json::from_slice(&body.materialize_standalone().unwrap()).unwrap()
        };
        [
            materialize(
                ChatEndpoint
                    .format_prepared_payload(&request, endpoint)
                    .unwrap(),
            ),
            materialize(
                CompletionsEndpoint
                    .format_prepared_payload(&request, endpoint)
                    .unwrap(),
            ),
        ]
    }

    #[test]
    fn streaming_chat_and_completions_request_usage_without_server_token_counting() {
        let endpoint = RawEndpointConfig {
            streaming: true,
            ..RawEndpointConfig::default()
        };

        for body in openai_bodies(text_turn(), &endpoint) {
            assert_eq!(body["stream_options"]["include_usage"], true);
        }
    }

    #[test]
    fn authored_stream_options_are_preserved_for_chat_and_completions() {
        let endpoint = RawEndpointConfig {
            streaming: true,
            extra: Some(Map::from_iter([(
                "stream_options".to_owned(),
                json!({"include_usage": false, "continuous_usage_stats": true}),
            )])),
            ..RawEndpointConfig::default()
        };

        for body in openai_bodies(text_turn(), &endpoint) {
            assert_eq!(body["stream_options"]["include_usage"], false);
            assert_eq!(body["stream_options"]["continuous_usage_stats"], true);
        }
    }

    #[test]
    fn effective_stream_value_controls_usage_negotiation() {
        let mut turn = text_turn();
        turn.extra_body = Some(Map::from_iter([
            ("stream".to_owned(), Value::Bool(false)),
            ("stream_options".to_owned(), Value::Null),
        ]));
        let endpoint = RawEndpointConfig {
            streaming: true,
            use_server_token_count: true,
            ..RawEndpointConfig::default()
        };

        for body in openai_bodies(turn, &endpoint) {
            assert_eq!(body["stream"], false);
            assert!(body["stream_options"].is_null());
        }
    }

    #[test]
    fn null_options_are_initialized_but_truthy_non_objects_are_preserved() {
        let mut endpoint = RawEndpointConfig {
            streaming: true,
            extra: Some(Map::from_iter([("stream_options".to_owned(), Value::Null)])),
            ..RawEndpointConfig::default()
        };

        for body in openai_bodies(text_turn(), &endpoint) {
            assert_eq!(body["stream_options"]["include_usage"], true);
        }

        endpoint.extra = Some(Map::from_iter([(
            "stream_options".to_owned(),
            Value::String("provider-owned".to_owned()),
        )]));
        for body in openai_bodies(text_turn(), &endpoint) {
            assert_eq!(body["stream_options"], "provider-owned");
        }
    }

    fn chat_payload(endpoint: &RawEndpointConfig) -> Value {
        let [chat, _completions] = openai_bodies(text_turn(), endpoint);
        chat
    }

    #[test]
    fn chat_per_chunk_usage_injects_only_when_opted_in() {
        let mut endpoint = RawEndpointConfig {
            streaming: true,
            use_server_token_count: true,
            per_chunk_usage: true,
            ..RawEndpointConfig::default()
        };
        let enabled = chat_payload(&endpoint);
        assert_eq!(
            enabled["stream_options"],
            json!({"include_usage": true, "continuous_usage_stats": true})
        );

        endpoint.per_chunk_usage = false;
        let disabled = chat_payload(&endpoint);
        assert_eq!(disabled["stream_options"], json!({"include_usage": true}));
    }

    #[test]
    fn chat_per_chunk_usage_preserves_authored_stream_options() {
        let endpoint = RawEndpointConfig {
            streaming: true,
            use_server_token_count: true,
            per_chunk_usage: true,
            extra: Some(Map::from_iter([(
                "stream_options".to_owned(),
                json!({
                    "include_usage": false,
                    "continuous_usage_stats": false,
                    "unrelated": "retained"
                }),
            )])),
            ..RawEndpointConfig::default()
        };
        assert_eq!(
            chat_payload(&endpoint)["stream_options"],
            json!({
                "include_usage": false,
                "continuous_usage_stats": false,
                "unrelated": "retained"
            })
        );
    }

    #[test]
    fn chat_per_chunk_usage_leaves_non_object_stream_options_unchanged() {
        let endpoint = RawEndpointConfig {
            streaming: true,
            use_server_token_count: true,
            per_chunk_usage: true,
            extra: Some(Map::from_iter([(
                "stream_options".to_owned(),
                Value::String("authored-invalid-shape".to_owned()),
            )])),
            ..RawEndpointConfig::default()
        };
        assert_eq!(
            chat_payload(&endpoint)["stream_options"],
            "authored-invalid-shape"
        );
    }

    #[test]
    fn lowered_wire_matches_rendered_dispatch_wire_text_only() {
        let turn = text_turn();
        let lowerer = ShapeLowerer::for_descriptor_id("chat").unwrap();
        let wires = lowerer.lower_turn(&turn).unwrap();
        assert_eq!(wires.len(), 1);
        let expected = Bytes::from(
            serde_json::to_vec(&render_turn_message(&turn, PartShape::Chat).unwrap()).unwrap(),
        );
        assert_eq!(wires[0], expected);
    }

    #[test]
    fn lowered_wire_matches_rendered_dispatch_wire_multimodal() {
        let turn = multimodal_turn("http://example/a.png");
        let lowerer = ShapeLowerer::for_descriptor_id("chat").unwrap();
        let wires = lowerer.lower_turn(&turn).unwrap();
        let expected = Bytes::from(
            serde_json::to_vec(&render_turn_message(&turn, PartShape::Chat).unwrap()).unwrap(),
        );
        assert_eq!(wires[0], expected);
    }

    #[test]
    fn lowered_wire_matches_rendered_dispatch_wire_responses() {
        let turn = text_turn();
        let lowerer = ShapeLowerer::for_descriptor_id("responses").unwrap();
        let wires = lowerer.lower_turn(&turn).unwrap();
        let mut expected_value = render_turn_message(&turn, PartShape::Responses).unwrap();
        if let Value::Object(obj) = &mut expected_value {
            obj.insert("type".into(), Value::String("message".into()));
        }
        assert_eq!(
            wires[0],
            Bytes::from(serde_json::to_vec(&expected_value).unwrap())
        );
        assert!(
            std::str::from_utf8(&wires[0])
                .unwrap()
                .contains(r#""type":"message""#)
        );
    }

    #[test]
    fn same_text_different_media_lowers_to_distinct_wires() {
        let lowerer = ShapeLowerer::for_descriptor_id("chat").unwrap();
        let a = lowerer
            .lower_turn(&multimodal_turn("http://example/a.png"))
            .unwrap();
        let b = lowerer
            .lower_turn(&multimodal_turn("http://example/b.png"))
            .unwrap();
        assert_ne!(a[0], b[0]);
    }

    #[test]
    fn non_message_array_dialects_have_no_lowerer() {
        assert!(ShapeLowerer::for_descriptor_id("embeddings").is_none());
        assert!(ShapeLowerer::for_descriptor_id("completions").is_none());
    }

    #[test]
    fn responses_websocket_lowering_emits_one_handle_free_create_event() {
        let turns = [text_turn()];
        let request = PreparedRequest::new(
            "gpt-test",
            &turns,
            None,
            None,
            CreditPhase::Profiling,
            None,
            None,
            None,
        );
        let store = SegmentPool::new().freeze();

        let body = ResponsesEndpoint
            .format_prepared_payload(&request, &RawEndpointConfig::default())
            .unwrap();
        let operation = ResponsesEndpoint
            .prepare_ws_operation(
                &request,
                &RawEndpointConfig::default(),
                &body,
                &store,
                &Overrides::new(),
            )
            .unwrap();

        assert_eq!(operation.messages().len(), 1);
        let message = &operation.messages()[0];
        assert_eq!(message.role(), PreparedWsMessageRole::MeasuredInput);
        assert_eq!(message.opcode(), crate::body_plan::PreparedWsOpcode::Text);
        let event: Value = serde_json::from_slice(message.payload()).unwrap();
        assert_eq!(event["type"], "response.create");
        assert_eq!(event["model"], "gpt-test");
        assert!(event["input"].is_array());
        let fallback: Value = serde_json::from_slice(
            operation
                .http_sse_fallback_body()
                .expect("Responses prepares its equivalent HTTP/SSE body"),
        )
        .unwrap();
        assert!(fallback.get("type").is_none());
        assert_eq!(fallback["model"], "gpt-test");
        assert_eq!(fallback["stream"], true);
    }

    #[test]
    fn responses_websocket_lowering_rejects_a_wire_backed_event_type() {
        let turns = [text_turn()];
        let request = PreparedRequest::new(
            "gpt-test",
            &turns,
            None,
            None,
            CreditPhase::Profiling,
            None,
            None,
            None,
        );
        let store = SegmentPool::new().freeze();
        let mut endpoint = RawEndpointConfig::default();
        endpoint.extra = Some(Map::from_iter([(
            "type".to_owned(),
            json!([{"value": "authored.type"}]),
        )]));
        let body = ResponsesEndpoint
            .format_prepared_payload(&request, &endpoint)
            .unwrap();

        let error = ResponsesEndpoint
            .prepare_ws_operation(&request, &endpoint, &body, &store, &Overrides::new())
            .unwrap_err();

        assert!(error.to_string().contains("reserved WebSocket event field"));
    }

    #[test]
    fn realtime_text_lowering_uses_conversation_item_without_empty_audio_commit() {
        let turns = [text_turn()];
        let request = PreparedRequest::new(
            "gpt-test",
            &turns,
            None,
            None,
            CreditPhase::Profiling,
            None,
            None,
            None,
        );
        let store = SegmentPool::new().freeze();
        let body = RealtimeEndpoint
            .format_prepared_payload(&request, &RawEndpointConfig::default())
            .unwrap();
        let operation = RealtimeEndpoint
            .prepare_ws_operation(
                &request,
                &RawEndpointConfig::default(),
                &body,
                &store,
                &Overrides::new(),
            )
            .unwrap();
        assert_eq!(operation.messages().len(), 2);
        assert!(!operation.requires_affinity_state());
        let input: Value = serde_json::from_slice(operation.messages()[0].payload()).unwrap();
        assert_eq!(input["type"], "conversation.item.create");
        assert_eq!(input["item"]["type"], "message");
        assert_eq!(input["item"]["role"], "user");
        assert_eq!(input["item"]["content"][0]["type"], "input_text");
        assert_eq!(
            operation.messages()[0].role(),
            PreparedWsMessageRole::MeasuredInput
        );
        let response: Value = serde_json::from_slice(operation.messages()[1].payload()).unwrap();
        assert_eq!(response["type"], "response.create");
        assert_eq!(response["response"]["modalities"], json!(["text", "audio"]));
        assert_eq!(
            operation.messages()[1].role(),
            PreparedWsMessageRole::Control
        );
        assert!(operation.http_sse_fallback_body().is_none());
    }

    #[test]
    fn realtime_audio_lowering_validates_and_appends_audio_before_commit() {
        let mut turn = text_turn();
        turn.texts.clear();
        turn.audios = vec![Media::new(vec!["data:audio/wav;base64,AAE=".to_owned()])];
        let turns = [turn];
        let request = PreparedRequest::new(
            "gpt-test",
            &turns,
            None,
            None,
            CreditPhase::Profiling,
            None,
            None,
            None,
        );
        let store = SegmentPool::new().freeze();
        let body = RealtimeEndpoint
            .format_prepared_payload(&request, &RawEndpointConfig::default())
            .unwrap();
        let operation = RealtimeEndpoint
            .prepare_ws_operation(
                &request,
                &RawEndpointConfig::default(),
                &body,
                &store,
                &Overrides::new(),
            )
            .unwrap();

        assert_eq!(operation.messages().len(), 3);
        let append: Value = serde_json::from_slice(operation.messages()[0].payload()).unwrap();
        assert_eq!(append["type"], "input_audio_buffer.append");
        assert_eq!(append["audio"], "AAE=");
        let commit: Value = serde_json::from_slice(operation.messages()[1].payload()).unwrap();
        assert_eq!(commit["type"], "input_audio_buffer.commit");
        let response: Value = serde_json::from_slice(operation.messages()[2].payload()).unwrap();
        assert_eq!(response["type"], "response.create");
    }

    #[test]
    fn realtime_audio_lowering_rejects_invalid_base64() {
        let mut turn = text_turn();
        turn.texts.clear();
        turn.audios = vec![Media::new(vec!["data:audio/wav;base64,%%%".to_owned()])];
        let turns = [turn];
        let request = PreparedRequest::new(
            "gpt-test",
            &turns,
            None,
            None,
            CreditPhase::Profiling,
            None,
            None,
            None,
        );
        let store = SegmentPool::new().freeze();
        let body = RealtimeEndpoint
            .format_prepared_payload(&request, &RawEndpointConfig::default())
            .unwrap();

        let error = RealtimeEndpoint
            .prepare_ws_operation(
                &request,
                &RawEndpointConfig::default(),
                &body,
                &store,
                &Overrides::new(),
            )
            .unwrap_err();
        assert!(error.to_string().contains("valid base64"));
    }

    #[test]
    fn realtime_mixed_text_and_audio_turn_fails_closed() {
        let mut turn = text_turn();
        turn.audios = vec![Media::new(vec!["data:audio/wav;base64,AAE=".to_owned()])];
        let turns = [turn];
        let request = PreparedRequest::new(
            "gpt-test",
            &turns,
            None,
            None,
            CreditPhase::Profiling,
            None,
            None,
            None,
        );
        let store = SegmentPool::new().freeze();
        let body = RealtimeEndpoint
            .format_prepared_payload(&request, &RawEndpointConfig::default())
            .unwrap();

        let error = RealtimeEndpoint
            .prepare_ws_operation(
                &request,
                &RawEndpointConfig::default(),
                &body,
                &store,
                &Overrides::new(),
            )
            .expect_err("mixed media ordering must fail closed");

        assert!(error.to_string().contains("mixed text and audio"));
    }

    #[test]
    fn realtime_history_sends_only_current_client_input_and_keeps_its_audio_commit_order() {
        let mut historical_user = text_turn();
        historical_user.texts.clear();
        historical_user.audios = vec![Media::new(vec!["data:audio/wav;base64,AAE=".to_owned()])];
        let assistant = Turn {
            role: Some("assistant".into()),
            texts: vec![Media::new(vec!["prior answer".to_owned()])],
            ..Turn::default()
        };
        let mut current_user = text_turn();
        current_user.texts.clear();
        current_user.audios = vec![Media::new(vec!["data:audio/wav;base64,AgM=".to_owned()])];
        let turns = [historical_user, assistant, current_user];
        let request = PreparedRequest::new(
            "gpt-test",
            &turns,
            None,
            None,
            CreditPhase::Profiling,
            None,
            None,
            None,
        );
        let store = SegmentPool::new().freeze();
        let body = RealtimeEndpoint
            .format_prepared_payload(&request, &RawEndpointConfig::default())
            .unwrap();

        let operation = RealtimeEndpoint
            .prepare_ws_operation(
                &request,
                &RawEndpointConfig::default(),
                &body,
                &store,
                &Overrides::new(),
            )
            .unwrap();
        let events = operation
            .messages()
            .iter()
            .map(|message| serde_json::from_slice::<Value>(message.payload()).unwrap())
            .collect::<Vec<_>>();

        assert!(operation.requires_affinity_state());
        assert_eq!(events.len(), 3);
        assert_eq!(events[0]["type"], "input_audio_buffer.append");
        assert_eq!(events[0]["audio"], "AgM=");
        assert_eq!(events[1]["type"], "input_audio_buffer.commit");
        assert_eq!(events[2]["type"], "response.create");
        assert!(
            events
                .iter()
                .all(|event| event["item"]["role"] != "assistant")
        );
    }

    #[test]
    fn realtime_audio_delta_decodes_to_audio_response_data() {
        let parsed = RealtimeEndpoint
            .parse_response(&ServerResponse::from_json(
                7,
                json!({"type":"response.output_audio.delta","delta":"AAE="}),
            ))
            .unwrap()
            .unwrap();
        let Some(ResponseData::Audio(audio)) = parsed.data else {
            panic!("expected Realtime audio response");
        };
        assert_eq!(audio.audio_bytes, vec![0, 1]);
        assert_eq!(audio.sample_rate_hz, 24_000);
        assert_eq!(audio.encoding, "pcm16");
    }

    #[test]
    fn websocket_dialects_register_closed_transport_capabilities() {
        let responses = ResponsesEndpoint
            .websocket_capabilities()
            .expect("Responses registers websocket capabilities");
        assert_eq!(responses.dialect, WebSocketDialect::Responses);
        assert_eq!(
            responses.connection_model,
            WebSocketConnectionModel::TurnSerialized
        );
        assert!(responses.supports_full_history_replay);
        assert!(responses.supports_http_sse_fallback);
        assert!(responses.has_affinity_state);

        let realtime = RealtimeEndpoint
            .websocket_capabilities()
            .expect("Realtime registers websocket capabilities");
        assert_eq!(realtime.dialect, WebSocketDialect::Realtime);
        assert_eq!(realtime.connection_model, WebSocketConnectionModel::Duplex);
        assert!(realtime.has_affinity_state);
        assert!(!realtime.supports_full_history_replay);
        assert!(!realtime.supports_http_sse_fallback);
    }

    /// The two halves of "this dialect splices lowered wires" must agree for
    /// every registered endpoint: the load-time predicate that *produces* the
    /// wires ([`ShapeLowerer`]) and the dispatch-time capability that consumes
    /// them. A dialect that lowers but does not declare it would re-resolve
    /// content the formatter discards; one that declares it but never lowers
    /// would skip resolving content it still needs.
    ///
    /// This is also the forcing function the old id list lacked: a new dialect
    /// that adds itself to one side alone fails here rather than silently
    /// falling out of an enumeration in another module.
    #[test]
    fn lowerable_dialects_declare_that_they_splice_lowered_wires() {
        let registry = crate::endpoints::EndpointRegistry::builtin().unwrap();
        let mut lowerable = Vec::new();
        for id in registry.canonical_ids() {
            // A dialect that cannot bind without authored configuration (the
            // template endpoint needs one) is not a message-array shape and has
            // no lowerer either way.
            let Ok(prepared) = registry.prepare(id, crate::endpoints::RawEndpointConfig::default())
            else {
                continue;
            };
            let descriptor_id = prepared.descriptor().id;
            let lowers = ShapeLowerer::for_descriptor_id(descriptor_id).is_some();
            assert_eq!(
                lowers,
                prepared.splices_lowered_wires(),
                "endpoint {descriptor_id} disagrees with its lowerer"
            );
            if lowers {
                lowerable.push(descriptor_id);
            }
        }
        lowerable.sort_unstable();
        assert_eq!(
            lowerable,
            [
                "chat",
                "chat_embeddings",
                "messages",
                "realtime",
                "responses"
            ]
        );
        // Rendering every turn is a different capability: these two compose their
        // own message parts and must never be treated as splicing lowered wires.
        for id in ["sagemaker", "kserve_chat"] {
            let prepared = registry
                .prepare(
                    &crate::endpoints::EndpointId::new(id).unwrap(),
                    crate::endpoints::RawEndpointConfig::default(),
                )
                .unwrap();
            assert!(prepared.renders_all_turns(), "{id} renders all turns");
            assert!(
                !prepared.splices_lowered_wires(),
                "{id} must not splice lowered wires"
            );
        }
    }

    #[test]
    fn reply_constructors_lower_to_value_dispatch_wire_all_shapes() {
        let text_reply = Turn {
            role: Some("assistant".into()),
            texts: vec![Media::new(vec!["live answer".to_string()])],
            ..Turn::default()
        };
        let raw_reply = Turn {
            raw_messages: Some(vec![json!({
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "why", "signature": "sig"},
                    {"type": "text", "text": "answer"},
                    // A replay-unsafe Responses item, to exercise the shared filter.
                    {"type": "reasoning", "id": "r-1"}
                ]
            })]),
            ..Turn::default()
        };
        for (id, shape) in [
            ("chat", PartShape::Chat),
            ("realtime", PartShape::Responses),
            ("responses", PartShape::Responses),
            ("messages", PartShape::Messages),
        ] {
            let lowerer = ShapeLowerer::for_descriptor_id(id).unwrap();
            for reply in [&text_reply, &raw_reply] {
                assert!(reply.lowered.is_none());
                let value_path = serialize_rendered_messages(
                    rendered_turn_messages(std::slice::from_ref(reply), shape, false).unwrap(),
                )
                .unwrap();
                let lowered = lowerer.lower_turn(reply).unwrap();
                assert_eq!(lowered, value_path, "shape {id} reply wire diverged");
            }
        }
    }
}
