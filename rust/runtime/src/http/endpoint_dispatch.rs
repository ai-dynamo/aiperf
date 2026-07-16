// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Runtime composition around HTTP-bound endpoint requests.
//!
//! Endpoint adapters own decoded semantics, while `aiperf-transport-http` owns
//! URL/body/lifecycle lowering and HTTP/SSE decoding through
//! [`HttpEndpointBinding`]. This module retains only runtime responsibilities:
//! endpoint parsing, observer emission, usage/response aggregation, and the
//! scheduled result shape.

use std::cell::Cell;
use std::task::{Context, Poll};

use anyhow::{Result, ensure};
use bytes::Bytes;
use serde_json::Value;

use crate::endpoints::{
    EndpointDescriptor, EndpointResult, ExtractedPayload, ParsedResponse, PreparedEndpoint,
    RequestRecord as EndpointRequestRecord, ResponseData, ServerResponse, Turn, UsageView,
};
use crate::metrics_core::RequestTrace;
use crate::transport_http::models::{ErrorDetails, ErrorKind, RequestRecord};
use crate::transport_http::transport::endpoint_binding::{
    HttpEndpointBinding, HttpEndpointBindingError, HttpEndpointRequest, HttpEndpointResponseFilter,
    MetadataHttpEndpointBinding, prepare_request,
};
use loadgen_core::collector::ReplayTerminalStatus;
use loadgen_core::sink::{
    ObservedEndpointMetrics, ObservedTokenKind, ObservedUsage, RequestObserver,
};

use crate::multiturn::TurnDataPolicy;
use crate::scheduled::{ModelResponseMetadata, TurnResponseObserver};

use super::{
    HttpCollectedDispatch, HttpDispatchResult, Request, TransportSink, absorb_transport_error,
    absorb_wire_response_metadata,
};

trait RuntimeEndpointAdapter {
    fn descriptor(&self) -> &'static EndpointDescriptor;
    fn streaming(&self) -> bool;
    fn extract_payload_inputs(&self, body: &Value) -> ExtractedPayload;
    fn parse_response(&self, response: &ServerResponse) -> EndpointResult<Option<ParsedResponse>>;
    fn build_assistant_turn(&self, record: &EndpointRequestRecord) -> EndpointResult<Option<Turn>>;
    fn captures_assistant_turn(&self) -> bool;
}

pub(super) struct EndpointDispatchHooks<'a> {
    observer: &'a dyn RequestObserver,
    on_first_token: &'a dyn Fn(i64),
    responses: Option<&'a dyn TurnResponseObserver>,
    data_policy: TurnDataPolicy,
}

impl<'a> EndpointDispatchHooks<'a> {
    pub(super) fn new(
        observer: &'a dyn RequestObserver,
        on_first_token: &'a dyn Fn(i64),
        responses: Option<&'a dyn TurnResponseObserver>,
        data_policy: TurnDataPolicy,
    ) -> Self {
        Self {
            observer,
            on_first_token,
            responses,
            data_policy,
        }
    }
}

struct EndpointResponseFilter<'a, A>
where
    A: RuntimeEndpointAdapter + ?Sized,
{
    endpoint: &'a A,
    responses: Option<&'a dyn TurnResponseObserver>,
    first_token_released: &'a Cell<bool>,
    on_first_token: &'a dyn Fn(i64),
}

impl<A> HttpEndpointResponseFilter for EndpointResponseFilter<'_, A>
where
    A: RuntimeEndpointAdapter + ?Sized,
{
    fn poll_ready(
        &mut self,
        context: &mut Context<'_>,
    ) -> Poll<Result<(), HttpEndpointBindingError>> {
        match self.responses {
            Some(responses) => responses.poll_ready(context).map(|result| {
                result.map_err(|_| {
                    HttpEndpointBindingError::from(ErrorDetails::other(
                        "normalized response consumer closed before terminal",
                    ))
                })
            }),
            None => Poll::Ready(Ok(())),
        }
    }

    fn start_send(
        &mut self,
        ttft_ns: i64,
        response: &ServerResponse,
    ) -> Result<bool, HttpEndpointBindingError> {
        let parsed = parse_endpoint_response(self.endpoint, response)
            .ok()
            .flatten();
        if let (Some(responses), Some(parsed)) = (self.responses, parsed.as_ref())
            && parsed.data.is_some()
        {
            responses.start_send(parsed.clone()).map_err(|_| {
                HttpEndpointBindingError::from(ErrorDetails::other(
                    "normalized response consumer closed before terminal",
                ))
            })?;
        }
        let meaningful = parsed
            .as_ref()
            .and_then(|parsed| parsed.data.as_ref())
            .is_some_and(|data| {
                self.endpoint.descriptor().produces_tokens && data.has_token_output()
            });
        if !meaningful {
            return Ok(false);
        }
        if !self.first_token_released.replace(true) {
            (self.on_first_token)(ttft_ns);
        }
        Ok(true)
    }
}

struct WorkerPreparedEndpointAdapter<'a>(&'a dyn PreparedEndpoint);

impl RuntimeEndpointAdapter for WorkerPreparedEndpointAdapter<'_> {
    fn descriptor(&self) -> &'static EndpointDescriptor {
        self.0.descriptor()
    }

    fn streaming(&self) -> bool {
        self.0.config().streaming()
    }

    fn extract_payload_inputs(&self, body: &Value) -> ExtractedPayload {
        self.0.extract_payload_inputs(body)
    }

    fn parse_response(&self, response: &ServerResponse) -> EndpointResult<Option<ParsedResponse>> {
        self.0.parse_response(response)
    }

    fn build_assistant_turn(&self, record: &EndpointRequestRecord) -> EndpointResult<Option<Turn>> {
        self.0.build_assistant_turn(record)
    }

    fn captures_assistant_turn(&self) -> bool {
        self.0.captures_assistant_turn()
    }
}

impl TransportSink {
    /// Dispatch through a worker-local prepared endpoint binding.
    pub(super) async fn dispatch_prepared_endpoint_collect_record_with_hooks(
        &self,
        req: Request,
        endpoint: &dyn PreparedEndpoint,
        model: &str,
        hooks: EndpointDispatchHooks<'_>,
    ) -> Result<HttpCollectedDispatch> {
        let binding = MetadataHttpEndpointBinding::from_prepared(endpoint, &self.base_urls, model);
        let endpoint = WorkerPreparedEndpointAdapter(endpoint);
        self.dispatch_runtime_endpoint_collect_record_with_hooks(req, &endpoint, &binding, hooks)
            .await
    }

    async fn dispatch_runtime_endpoint_collect_record_with_hooks<A, B>(
        &self,
        req: Request,
        endpoint: &A,
        binding: &B,
        hooks: EndpointDispatchHooks<'_>,
    ) -> Result<HttpCollectedDispatch>
    where
        A: RuntimeEndpointAdapter + ?Sized,
        B: HttpEndpointBinding,
    {
        let EndpointDispatchHooks {
            observer: obs,
            on_first_token,
            responses,
            data_policy,
        } = hooks;
        let Request {
            uuid,
            input_length,
            max_output_tokens,
            prompt_text,
            request_body,
            request_body_bytes,
            headers,
            parameters,
            endpoint_path,
            streaming,
            x_correlation_id,
            is_final_turn,
            cancel_after_ns,
            url_index,
            ..
        } = req;
        obs.on_admit(uuid, self.ms(self.clock.now_ns()), 0);

        ensure!(
            request_body.is_none() || request_body_bytes.is_none(),
            "an HTTP request cannot supply both JSON and serialized bodies"
        );
        let body = match request_body_bytes {
            Some(body) => body,
            None => {
                let payload = request_body.unwrap_or_else(|| {
                    let prompt = prompt_text.unwrap_or_default();
                    crate::endpoints::chat_request_body(
                        &self.model,
                        &[("user", prompt.as_str())],
                        max_output_tokens,
                    )
                });
                Bytes::from(serde_json::to_vec(&payload)?)
            }
        };
        let mut endpoint_metrics = ObservedEndpointMetrics {
            num_images: serde_json::from_slice::<Value>(&body)
                .ok()
                .map(|payload| endpoint.extract_payload_inputs(&payload).image_count as usize)
                .filter(|count| *count > 0),
            ..ObservedEndpointMetrics::default()
        };
        let prepared = prepare_request(
            binding,
            &self.transport,
            HttpEndpointRequest {
                body,
                headers,
                parameters,
                endpoint_path,
                streaming,
                correlation_id: x_correlation_id,
                request_id: Some(uuid.to_string()),
                is_final_turn,
                cancel_after_ns,
                url_index,
                reuse: self.connection_reuse,
            },
        )
        .await?;
        let request_payload = prepared.canonical_body().clone();
        let request_url = prepared.request_config().url.clone();

        let first_token_released = Cell::new(false);
        let record = if responses.is_some() {
            let mut first_response_filter = EndpointResponseFilter {
                endpoint,
                responses,
                first_token_released: &first_token_released,
                on_first_token,
            };
            prepared
                .dispatch_backpressured(
                    &self.transport,
                    self.clock.clone(),
                    binding,
                    &mut first_response_filter,
                )
                .await
        } else {
            let mut first_response_filter = |ttft_ns, response: &ServerResponse| {
                let meaningful = parse_endpoint_response(endpoint, response)
                    .ok()
                    .flatten()
                    .and_then(|parsed| parsed.data)
                    .is_some_and(|data| {
                        endpoint.descriptor().produces_tokens
                            && (data.raw_token_count().is_some_and(|count| count > 0)
                                || !data.get_text().is_empty())
                    });
                if meaningful && !first_token_released.replace(true) {
                    on_first_token(ttft_ns);
                }
                meaningful
            };
            prepared
                .dispatch(
                    &self.transport,
                    self.clock.clone(),
                    binding,
                    &mut first_response_filter,
                )
                .await
        };

        let mut parsed_any = false;
        let mut parsed_content = false;
        let mut parse_failed = false;
        let mut response_text = String::new();
        let mut model_response = ModelResponseMetadata::default();
        let mut observed_usage = ObservedUsage::default();
        for response in &record.responses {
            let Some(server_response) = binding.decode_response(response) else {
                continue;
            };
            if let Some(value) = &server_response.json {
                model_response.wire_responses.push(value.clone());
                absorb_wire_response_metadata(value, &mut model_response);
            }
            let parsed = match parse_endpoint_response(endpoint, &server_response) {
                Ok(parsed) => parsed,
                Err(error) => {
                    if data_policy.allow_content_diagnostics() {
                        tracing::warn!(
                            uuid = %uuid,
                            endpoint = endpoint.descriptor().id,
                            error = %error,
                            "endpoint response parsing failed"
                        );
                    } else {
                        tracing::warn!(
                            uuid = %uuid,
                            endpoint = endpoint.descriptor().id,
                            "restricted endpoint response parsing failed"
                        );
                    }
                    parse_failed = true;
                    continue;
                }
            };
            let Some(parsed) = parsed else { continue };
            parsed_any = true;
            absorb_usage(&parsed, &mut observed_usage);
            let Some(data) = parsed.data.as_ref() else {
                continue;
            };
            parsed_content = true;
            absorb_endpoint_metrics(data, &mut endpoint_metrics);
            let text = absorb_response_data(data, &mut model_response);
            response_text.push_str(&text);
            if endpoint.descriptor().produces_tokens {
                let at_ns = i64::try_from(parsed.perf_ns).unwrap_or(i64::MAX);
                if let ResponseData::TokenIds { token_ids } = data
                    && !token_ids.is_empty()
                {
                    if !first_token_released.replace(true) {
                        on_first_token(at_ns.saturating_sub(record.start_ns));
                    }
                    let timestamps = vec![self.ms(at_ns); token_ids.len()];
                    obs.on_output_tokens(uuid, &timestamps);
                } else if !text.is_empty() {
                    if !first_token_released.replace(true) {
                        on_first_token(at_ns.saturating_sub(record.start_ns));
                    }
                    obs.on_classified_token(uuid, self.ms(at_ns), token_kind(data));
                }
            }
        }

        if endpoint.descriptor().requires_raw_token_ids {
            observed_usage.prompt_tokens = Some(input_length);
            if observed_usage.completion_tokens.is_none() {
                observed_usage.completion_tokens =
                    model_response.output_token_ids.as_ref().map(Vec::len);
            }
            observed_usage.total_tokens = observed_usage
                .prompt_tokens
                .zip(observed_usage.completion_tokens)
                .and_then(|(prompt, completion)| prompt.checked_add(completion));
        }

        if endpoint.captures_assistant_turn() {
            let endpoint_record = EndpointRequestRecord {
                responses: record
                    .responses
                    .iter()
                    .filter_map(|response| binding.decode_response(response))
                    .collect(),
            };
            match endpoint.build_assistant_turn(&endpoint_record) {
                Ok(Some(turn)) => {
                    model_response.assistant_message = assistant_message(&turn);
                }
                Ok(None) => {}
                Err(error) => {
                    if data_policy.allow_content_diagnostics() {
                        tracing::warn!(
                            uuid = %uuid,
                            error = %error,
                            "endpoint assistant-message reconstruction failed"
                        );
                    } else {
                        tracing::warn!(
                            uuid = %uuid,
                            "restricted endpoint assistant-message reconstruction failed"
                        );
                    }
                    parse_failed = true;
                }
            }
        }

        let terminal = match record.error.as_ref().map(|error| error.kind) {
            Some(ErrorKind::Cancelled) => ReplayTerminalStatus::Canceled,
            Some(_) => ReplayTerminalStatus::Failed,
            None if record
                .status
                .is_some_and(|status| (200..300).contains(&status))
                && parsed_content
                && !parse_failed =>
            {
                ReplayTerminalStatus::Completed
            }
            None => ReplayTerminalStatus::Failed,
        };
        absorb_transport_error(
            record.error.as_ref(),
            terminal,
            record.status,
            &mut model_response,
        );
        if data_policy.allow_content_diagnostics() {
            tracing::debug!(
                uuid = %uuid,
                endpoint = endpoint.descriptor().id,
                url = %request_url,
                status = ?record.status,
                responses = record.responses.len(),
                error = ?record.error,
                parsed_any,
                parsed_content,
                parse_failed,
                terminal = ?terminal,
                "classified endpoint dispatch"
            );
        } else {
            tracing::debug!(
                uuid = %uuid,
                endpoint = endpoint.descriptor().id,
                status = ?record.status,
                responses = record.responses.len(),
                parsed_any,
                parsed_content,
                parse_failed,
                terminal = ?terminal,
                "classified restricted endpoint dispatch"
            );
        }
        obs.on_usage(uuid, observed_usage);
        obs.on_endpoint_metrics(uuid, endpoint_metrics);
        obs.on_terminal(uuid, terminal);

        let prompt_tokens = observed_usage
            .prompt_tokens
            .and_then(|value| u32::try_from(value).ok());
        let completion_tokens = observed_usage
            .completion_tokens
            .and_then(|value| u32::try_from(value).ok());

        let result = HttpDispatchResult {
            start_ns: record.start_ns,
            end_ns: record.end_ns.unwrap_or_else(|| self.clock.now_ns()),
            status: record.status,
            terminal,
            response_text,
            model_response,
            prompt_tokens,
            completion_tokens,
            http: http_trace(&record),
        };
        Ok(HttpCollectedDispatch {
            result,
            request_payload,
            record,
        })
    }
}

#[cfg(test)]
fn meaningful_endpoint_response<A: RuntimeEndpointAdapter + ?Sized>(
    endpoint: &A,
    response: &ServerResponse,
) -> bool {
    if !endpoint.descriptor().produces_tokens {
        return false;
    }
    parse_endpoint_response(endpoint, response)
        .ok()
        .flatten()
        .and_then(|parsed| parsed.data)
        .is_some_and(|data| data.has_token_output())
}

fn parse_endpoint_response<A: RuntimeEndpointAdapter + ?Sized>(
    endpoint: &A,
    response: &ServerResponse,
) -> EndpointResult<Option<ParsedResponse>> {
    let parsed = endpoint.parse_response(response)?;
    if parsed.is_some() || endpoint.descriptor().id != "chat" {
        return Ok(parsed);
    }
    let Some(mut object) = response.json.as_ref().and_then(Value::as_object).cloned() else {
        return Ok(None);
    };
    if object.contains_key("object") || !object.contains_key("choices") {
        return Ok(None);
    }

    // Older OpenAI-compatible mocks omitted `object` while retaining a valid
    // `choices` envelope. Keep the endpoint adapter itself source-strict and
    // normalize only this established wire-compatibility shape at dispatch.
    object.insert(
        "object".into(),
        Value::String(
            if endpoint.streaming() {
                "chat.completion.chunk"
            } else {
                "chat.completion"
            }
            .into(),
        ),
    );
    endpoint.parse_response(&ServerResponse {
        perf_ns: response.perf_ns,
        json: Some(Value::Object(object)),
        raw: response.raw.clone(),
    })
}

fn token_kind(data: &ResponseData) -> ObservedTokenKind {
    match data {
        ResponseData::Reasoning { reasoning, .. } if !reasoning.is_empty() => {
            ObservedTokenKind::Reasoning
        }
        _ => ObservedTokenKind::Output,
    }
}

fn absorb_response_data(data: &ResponseData, metadata: &mut ModelResponseMetadata) -> String {
    match data {
        ResponseData::Text { text } => {
            append_metadata_text(&mut metadata.content, text);
        }
        ResponseData::Reasoning { content, reasoning } => {
            metadata.content.get_or_insert_with(String::new);
            append_metadata_text(&mut metadata.reasoning, reasoning);
            if let Some(content) = content {
                append_metadata_text(&mut metadata.content, content);
            }
        }
        ResponseData::ToolCall {
            tool_call_text,
            content,
        } => {
            if let Some(content) = content {
                append_metadata_text(&mut metadata.content, content);
            }
            append_metadata_text(&mut metadata.content, tool_call_text);
        }
        ResponseData::TokenIds { token_ids } => {
            metadata
                .output_token_ids
                .get_or_insert_with(Vec::new)
                .extend_from_slice(token_ids);
        }
        ResponseData::Embeddings { .. }
        | ResponseData::Rankings { .. }
        | ResponseData::ImageRetrieval { .. }
        | ResponseData::Images(_)
        | ResponseData::Audio(_)
        | ResponseData::Video(_) => {}
    }
    data.get_text()
}

fn append_metadata_text(target: &mut Option<String>, text: &str) {
    target.get_or_insert_with(String::new).push_str(text);
}

fn assistant_message(turn: &Turn) -> Option<Value> {
    if let Some(message) = turn
        .raw_messages
        .as_ref()
        .and_then(|messages| messages.first())
    {
        return Some(message.clone());
    }
    let content = turn
        .texts
        .iter()
        .flat_map(|media| &media.contents)
        .cloned()
        .collect::<String>();
    (!content.is_empty()).then(|| {
        serde_json::json!({
            "role": turn.role.as_deref().unwrap_or("assistant"),
            "content": content,
        })
    })
}

fn absorb_endpoint_metrics(data: &ResponseData, metrics: &mut ObservedEndpointMetrics) {
    let ResponseData::Video(video) = data else {
        return;
    };
    metrics.video_inference_seconds = video
        .inference_time_s
        .filter(|value| value.is_finite())
        .or(metrics.video_inference_seconds);
    metrics.video_peak_memory_mb = video
        .peak_memory_mb
        .filter(|value| value.is_finite())
        .or(metrics.video_peak_memory_mb);
}

fn absorb_usage(parsed: &ParsedResponse, observed: &mut ObservedUsage) {
    let Some(usage) = parsed.usage.as_ref() else {
        return;
    };
    let Some(usage) = UsageView::from_value(usage) else {
        return;
    };
    observed.prompt_tokens = usage
        .prompt_tokens()
        .and_then(|value| usize::try_from(value).ok())
        .or(observed.prompt_tokens);
    observed.completion_tokens = usage
        .completion_tokens()
        .and_then(|value| usize::try_from(value).ok())
        .or(observed.completion_tokens);
    observed.total_tokens = usage
        .total_tokens()
        .and_then(|value| usize::try_from(value).ok())
        .or(observed.total_tokens);
    observed.reasoning_tokens = usage
        .reasoning_tokens()
        .and_then(|value| usize::try_from(value).ok())
        .or(observed.reasoning_tokens);
    observed.prompt_cache_read_tokens = usage
        .prompt_cache_read_tokens()
        .and_then(|value| usize::try_from(value).ok())
        .or(observed.prompt_cache_read_tokens);
    observed.prompt_cache_write_tokens = usage
        .prompt_cache_write_tokens()
        .and_then(|value| usize::try_from(value).ok())
        .or(observed.prompt_cache_write_tokens);
    observed.prompt_cache_miss_tokens = usage
        .prompt_cache_miss_tokens()
        .and_then(|value| usize::try_from(value).ok())
        .or(observed.prompt_cache_miss_tokens);
    observed.prompt_audio_tokens = usage
        .prompt_audio_tokens()
        .and_then(|value| usize::try_from(value).ok())
        .or(observed.prompt_audio_tokens);
    observed.completion_audio_tokens = usage
        .completion_audio_tokens()
        .and_then(|value| usize::try_from(value).ok())
        .or(observed.completion_audio_tokens);
    observed.accepted_prediction_tokens = usage
        .accepted_prediction_tokens()
        .and_then(|value| usize::try_from(value).ok())
        .or(observed.accepted_prediction_tokens);
    observed.rejected_prediction_tokens = usage
        .rejected_prediction_tokens()
        .and_then(|value| usize::try_from(value).ok())
        .or(observed.rejected_prediction_tokens);
    observed.tool_use_prompt_tokens = usage
        .tool_use_prompt_tokens()
        .and_then(|value| usize::try_from(value).ok())
        .or(observed.tool_use_prompt_tokens);
    observed.prompt_audio_seconds = usage
        .prompt_audio_seconds()
        .or(observed.prompt_audio_seconds);
}

fn http_trace(record: &RequestRecord) -> RequestTrace {
    let mut http = record
        .trace
        .as_ref()
        .map_or_else(RequestTrace::default, |trace| RequestTrace {
            blocked_ns: trace.blocked(),
            dns_lookup_ns: trace.dns_lookup(),
            connecting_ns: trace.connecting(),
            sending_ns: trace.sending(),
            waiting_ns: trace.waiting(),
            receiving_ns: trace.receiving(),
            duration_ns: trace.duration(),
            connection_reused: Some(trace.connection_reused_ns.is_some()),
            data_sent_bytes: Some(trace.request_bytes_total),
            data_received_bytes: Some(trace.response_bytes_total),
            chunks_sent: Some(u64::from(trace.request_chunks_count)),
            chunks_received: Some(u64::from(trace.response_chunks_count)),
            ..RequestTrace::default()
        });
    http.stream_setup_ns = record
        .recv_start_ns
        .map(|receive_start| receive_start.saturating_sub(record.start_ns));
    http
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::endpoints::PreparedEndpoint;
    use crate::transport_http::models::SseMessage;
    use crate::transport_http::transport::endpoint_binding::decode_sse_response;

    /// Prepare a builtin streaming endpoint by its open ID.
    fn prepared_streaming(endpoint_name: &str) -> Box<dyn PreparedEndpoint> {
        crate::endpoints::EndpointRegistry::builtin()
            .unwrap()
            .prepare(
                &crate::endpoints::EndpointId::new(endpoint_name).unwrap(),
                crate::endpoints::RawEndpointConfig {
                    streaming: true,
                    ..crate::endpoints::RawEndpointConfig::default()
                },
            )
            .unwrap()
    }

    #[test]
    fn endpoint_sse_filter_uses_the_selected_dialect() {
        let tgi = SseMessage::parse(r#"data: {"token":{"text":"hello"}}"#, 10);
        let tgi_endpoint = prepared_streaming("huggingface_generate");
        let tgi_adapter = WorkerPreparedEndpointAdapter(tgi_endpoint.as_ref());
        assert!(meaningful_endpoint_response(
            &tgi_adapter,
            &decode_sse_response(&tgi).unwrap()
        ));

        let image = SseMessage::parse(r#"data: {"b64_json":"AA=="}"#, 11);
        let image_endpoint = prepared_streaming("image_generation");
        let image_adapter = WorkerPreparedEndpointAdapter(image_endpoint.as_ref());
        assert!(!meaningful_endpoint_response(
            &image_adapter,
            &decode_sse_response(&image).unwrap()
        ));

        let legacy_chat =
            SseMessage::parse(r#"data: {"choices":[{"delta":{"content":"compat"}}]}"#, 12);
        let chat_endpoint = prepared_streaming("chat");
        let chat_adapter = WorkerPreparedEndpointAdapter(chat_endpoint.as_ref());
        assert!(meaningful_endpoint_response(
            &chat_adapter,
            &decode_sse_response(&legacy_chat).unwrap()
        ));
    }

    #[test]
    fn usage_aliases_and_second_conversion_are_bounded() {
        let parsed = ParsedResponse {
            perf_ns: 1,
            data: None,
            usage: Some(serde_json::json!({"input_tokens":3,"output_tokens":5})),
            sources: None,
        };
        let mut observed = ObservedUsage::default();
        absorb_usage(&parsed, &mut observed);
        assert_eq!(observed.prompt_tokens, Some(3));
        assert_eq!(observed.completion_tokens, Some(5));

        let parsed = ParsedResponse {
            perf_ns: 2,
            data: None,
            usage: Some(serde_json::json!({
                "input_tokens": 3,
                "cache_read_input_tokens": 7,
                "cache_creation_input_tokens": 2,
                "output_tokens": 5
            })),
            sources: None,
        };
        absorb_usage(&parsed, &mut observed);
        assert_eq!(observed.prompt_tokens, Some(12));
        assert_eq!(observed.prompt_cache_read_tokens, Some(7));
        assert_eq!(observed.prompt_cache_write_tokens, Some(2));

        let parsed = ParsedResponse {
            perf_ns: 3,
            data: None,
            usage: Some(serde_json::json!({
                "prompt_tokens_details": {"audio_tokens": 11},
                "completion_tokens_details": {
                    "audio_tokens": 13,
                    "accepted_prediction_tokens": 17,
                    "rejected_prediction_tokens": 19
                },
                "toolUsePromptTokenCount": 23,
                "prompt_audio_seconds": 2.5
            })),
            sources: None,
        };
        absorb_usage(&parsed, &mut observed);
        assert_eq!(observed.prompt_audio_tokens, Some(11));
        assert_eq!(observed.completion_audio_tokens, Some(13));
        assert_eq!(observed.accepted_prediction_tokens, Some(17));
        assert_eq!(observed.rejected_prediction_tokens, Some(19));
        assert_eq!(observed.tool_use_prompt_tokens, Some(23));
        assert_eq!(observed.prompt_audio_seconds, Some(2.5));
    }
}
