// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Runtime composition around HTTP-bound endpoint requests.
//!
//! Endpoint adapters own decoded semantics, while [`HttpEndpointBinding`] owns
//! URL/body/lifecycle lowering and HTTP/SSE decoding. This module retains:
//! endpoint parsing, observer emission, usage/response aggregation, and the
//! scheduled result shape.

use std::cell::Cell;
use std::task::{Context, Poll};

use anyhow::Result;
use bytes::Bytes;
use serde_json::Value;

use crate::dispatch::collector::ReplayTerminalStatus;
use crate::dispatch::sink::{ObservedEndpointMetrics, ObservedUsage, RequestObserver};
use crate::endpoints::{
    EndpointDescriptor, EndpointResult, ExtractedPayload, ParsedResponse, PreparedEndpoint,
    RequestRecord as EndpointRequestRecord, ServerResponse, Turn, extract_vllm_spec_decode_stats,
    parse_vllm_spec_decode_stats,
};
use crate::metrics_core::RequestTrace;
use crate::transport::core::{
    BoundedDecisionAdmission, BoundedDecisionMode, BoundedDecisionReader, ErrorDetails, ErrorKind,
    RequestRecord,
};
use crate::transport::http::transport::endpoint_binding::{
    HttpEndpointBinding, HttpEndpointBindingError, HttpEndpointRequest, HttpEndpointResponseFilter,
    MetadataHttpEndpointBinding, prepare_request,
};

use crate::multiturn::TurnDataPolicy;
use crate::scheduled::{ModelResponseMetadata, TurnResponseObserver};
use crate::transport::reduce::{
    EndpointReduceAccumulators, TokenEmitter, admit_parsed_decision, assistant_message,
    reduce_parsed_response, token_kind,
};

use super::{
    HttpCollectedDispatch, HttpDispatchResult, Request, TransportSink, absorb_transport_error,
    absorb_wire_response_metadata, normalize_finish_reason,
};
use crate::transport::core::Response;
use crate::transport::http::sse::ChatChunk;

/// Metadata absorption for a streamed chat chunk, equivalent to
/// [`absorb_wire_response_metadata`] on the same body but without a `Value`.
///
/// Deliberately narrow: content and reasoning deltas are `reduce_parsed_response`'s
/// job on this path, so appending them here too would emit every delta twice. Like
/// the generic reader, this takes `choices[0]` only.
///
/// `cached_prompt_tokens` is untouched because the caller only takes this path
/// for chunks with no `usage`, and the generic reader leaves it unchanged there.
fn absorb_chat_chunk_wire_metadata(chunk: &ChatChunk, metadata: &mut ModelResponseMetadata) {
    if let Some(response_id) = chunk
        .id
        .as_deref()
        .or(chunk.request_id.as_deref())
        .filter(|value| !value.is_empty())
    {
        // Streaming repeats one id per chunk; allocate only on a real change.
        if metadata.response_id.as_deref() != Some(response_id) {
            metadata.response_id = Some(response_id.to_string());
        }
    }
    if let Some(finish_reason) = chunk
        .choices
        .first()
        .and_then(|choice| choice.finish_reason.as_deref())
        .filter(|value| !value.is_empty())
    {
        metadata.finish_reason = Some(normalize_finish_reason(finish_reason));
    }
}

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

/// Owns the terminal lifecycle after the bounded path reports admission.
///
/// Every fallible preparation, send, response-admission, and cancellation path
/// then crosses one RAII boundary. It begins as `Canceled` while an in-flight
/// await remains unresolved; every completed result explicitly resolves its
/// final state before returning, so dropping an aborted future cannot flatten
/// cancellation into failure.
struct ArmedTerminalFence<'a> {
    observer: &'a dyn RequestObserver,
    uuid: uuid::Uuid,
    status: ReplayTerminalStatus,
}

impl<'a> ArmedTerminalFence<'a> {
    fn arm(observer: &'a dyn RequestObserver, uuid: uuid::Uuid) -> Self {
        Self {
            observer,
            uuid,
            status: ReplayTerminalStatus::Canceled,
        }
    }

    fn resolve(&mut self, status: ReplayTerminalStatus) {
        self.status = status;
    }
}

impl Drop for ArmedTerminalFence<'_> {
    fn drop(&mut self) {
        self.observer.on_terminal(self.uuid, self.status);
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

    /// Dispatch one selected endpoint through the no-record bounded-decision
    /// path.
    ///
    /// Response bytes are admitted while the SSE reader owns each decoded
    /// frame. This intentionally does not share the ordinary collect path,
    /// whose terminal `RequestRecord` is already an allocation boundary.
    pub(super) async fn dispatch_prepared_endpoint_bounded_decision(
        &self,
        req: Request,
        endpoint: &dyn PreparedEndpoint,
        model: &str,
        observer: &dyn RequestObserver,
        on_first_token: &dyn Fn(i64),
        mode: BoundedDecisionMode,
    ) -> Result<BoundedDecisionReader> {
        let endpoint = WorkerPreparedEndpointAdapter(endpoint);
        let binding =
            MetadataHttpEndpointBinding::from_prepared(endpoint.0, &self.base_urls, model);
        let Request {
            uuid,
            max_output_tokens,
            prompt_text,
            body,
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
        observer.on_admit(uuid, self.ms(self.clock.now_ns()), 0);
        let mut terminal_fence = ArmedTerminalFence::arm(observer, uuid);
        let dispatched: Result<(
            std::result::Result<u16, ErrorDetails>,
            BoundedDecisionAdmission,
        )> = async {
            let body = match body {
                Some(body) => body.into_wire()?,
                None => {
                    let prompt = prompt_text.unwrap_or_default();
                    let payload = crate::endpoints::chat_request_body(
                        model,
                        &[("user", prompt.as_str())],
                        max_output_tokens,
                    );
                    Bytes::from(serde_json::to_vec(&payload)?)
                }
            };
            let (body, _) = match self.content_server_base.as_deref() {
                Some(base) => super::tag_content_urls(
                    body,
                    base,
                    &uuid.to_string(),
                    crate::content_server::dispatch_wall_ns(),
                ),
                None => (body, None),
            };
            let prepared = prepare_request(
                &binding,
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
            let mut admission = BoundedDecisionAdmission::new(mode);
            let to_ms = |ns| self.ms(ns);
            let mut on_response = |perf_ns: i64, response: &ServerResponse| {
                let Some(parsed) =
                    parse_endpoint_response(&endpoint, response).map_err(|error| {
                        ErrorDetails::other(format!("bounded decision parse failed: {error}"))
                    })?
                else {
                    return Ok(false);
                };
                let Some(data) = parsed.data.as_ref() else {
                    return Ok(false);
                };
                admit_parsed_decision(&parsed, &mut admission)
                    .map_err(|error| ErrorDetails::other(error.to_string()))?;
                if data.has_token_output() {
                    observer.on_classified_token(uuid, to_ms(perf_ns), token_kind(data));
                    return Ok(true);
                }
                Ok(false)
            };
            let mut first = |ttft_ns| on_first_token(ttft_ns);
            let status = prepared
                .dispatch_bounded_sse(
                    &self.transport,
                    &binding,
                    mode.max_sse_frame_bytes(),
                    &mut first,
                    &mut on_response,
                )
                .await;
            drop(on_response);
            Ok((status, admission))
        }
        .await;
        let (result, terminal_status) = match dispatched {
            Ok((Ok(status), admission)) if (200..300).contains(&status) => {
                let result = admission.finish().map_err(anyhow::Error::from);
                let terminal_status = if result.is_ok() {
                    ReplayTerminalStatus::Completed
                } else {
                    ReplayTerminalStatus::Failed
                };
                (result, terminal_status)
            }
            Ok((Ok(status), _)) => (
                Err(anyhow::anyhow!(
                    "bounded decision dispatch returned HTTP status {status}"
                )),
                ReplayTerminalStatus::Failed,
            ),
            Ok((Err(error), _)) => {
                let terminal_status = if error.kind == ErrorKind::Cancelled {
                    ReplayTerminalStatus::Canceled
                } else {
                    ReplayTerminalStatus::Failed
                };
                (
                    Err(anyhow::anyhow!(
                        "bounded decision transport failed: {}",
                        error.message
                    )),
                    terminal_status,
                )
            }
            Err(error) => (Err(error), ReplayTerminalStatus::Failed),
        };
        terminal_fence.resolve(terminal_status);
        result
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
            body,
            headers,
            parameters,
            endpoint_path,
            streaming,
            x_correlation_id,
            is_final_turn,
            cancel_after_ns,
            url_index,
            image_count: known_image_count,
            ..
        } = req;
        obs.on_admit(uuid, self.ms(self.clock.now_ns()), 0);

        let body = match body {
            Some(body) => body.into_wire()?,
            None => {
                let prompt = prompt_text.unwrap_or_default();
                let payload = crate::endpoints::chat_request_body(
                    &self.model,
                    &[("user", prompt.as_str())],
                    max_output_tokens,
                );
                Bytes::from(serde_json::to_vec(&payload)?)
            }
        };
        // Tag content-server media URLs with rid/mi/dispatch-time so served
        // transfers correlate back to this request; external URLs and non-media
        // bodies are untouched. The parse is shared with image counting below.
        let (body, parsed) = match self.content_server_base.as_deref() {
            Some(base) => super::tag_content_urls(
                body,
                base,
                &uuid.to_string(),
                crate::content_server::dispatch_wall_ns(),
            ),
            None => (body, None),
        };
        // `num_images` is the only consumer of the parsed body here. When
        // composition already established the exact wire image count, trust it and
        // skip re-parsing the (potentially multi-MB multimodal) body on the hot
        // path — the full deserialize otherwise dominates dispatch for large image
        // batches. Fall back to parsing only when the count is unknown (raw
        // payloads, history-accumulating turns) or content-server tagging already
        // produced the value for free.
        let payload = match parsed {
            Some(parsed) => Some(parsed),
            // Parsing anyway would produce the same `num_images`, so the skip is
            // invisible to every artifact — see `BODY_PARSE_SKIPS`.
            None if known_image_count.is_some() => {
                #[cfg(test)]
                BODY_PARSE_SKIPS.with(|count| count.set(count.get() + 1));
                None
            }
            None => serde_json::from_slice::<Value>(&body).ok(),
        };
        let mut endpoint_metrics = ObservedEndpointMetrics {
            num_images: known_image_count
                .map(|count| count as usize)
                .or_else(|| {
                    payload.as_ref().map(|payload| {
                        endpoint.extract_payload_inputs(payload).image_count as usize
                    })
                })
                .filter(|count| *count > 0),
            ..ObservedEndpointMetrics::default()
        };
        let prepared = prepare_request(
            binding,
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
        // The canonical body is read back only by the raw artifact. Taking the
        // handle unconditionally promoted the assembled body —
        // `BytesMut::freeze()`-derived, so `len == capacity` and the first clone
        // heap-allocates a shared control block — on every dispatch of every
        // run, including the runs that export no raw artifact.
        let request_payload = if self.captures_request_payload() {
            prepared.canonical_body().clone()
        } else {
            Bytes::new()
        };
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
        let captures_spec_decode = matches!(endpoint.descriptor().id, "chat" | "completions");
        let mut spec_decode_stats = None;
        let to_ms = |ns| self.ms(ns);
        let emitter = TokenEmitter {
            uuid,
            produces_tokens: endpoint.descriptor().produces_tokens,
            start_ns: record.start_ns,
            obs,
            to_ms: &to_ms,
            first_token_released: &first_token_released,
            on_first_token,
        };
        // Retain the pass-1 decoded responses when this endpoint reconstructs an
        // assistant turn, so the capture below reuses them instead of decoding
        // every buffered response a second time. `filter_map(decode_response)`
        // in the old capture kept exactly the successfully-decoded responses
        // regardless of parse outcome, so we push right after a successful
        // decode (before any parse-failure `continue`) to preserve that set.
        // A reconstructed assistant turn is only ever read back as context for a
        // *later* turn of the same session (`multiturn` splices it into the next
        // request; `TurnDispatchOutcome::to_turn_response` forwards it to the
        // continuation hooks). It is not exported and no metric derives from it,
        // so on a session's final turn the reconstruction — parsing every buffered
        // response into `Value` maps and dropping them — is pure waste. Single-turn
        // workloads mark every request final, which is the common benchmark shape.
        let captures_turn = !is_final_turn && endpoint.captures_assistant_turn();
        // Streamed chat is the hot shape: at OSL 150 the generic path below
        // builds 150 `String` copies and 150 `serde_json::Value` trees per
        // request, and `preserve_order` makes every object an `IndexMap` whose
        // keys are individually allocated and SipHashed. A typed decode reads
        // the same fields with none of that. Only chunks this type fully models
        // take it; everything else falls through unchanged.
        let chat_fast_path = endpoint.descriptor().id == "chat" && !captures_turn;
        let mut decoded_responses: Vec<ServerResponse> = Vec::new();
        for response in &record.responses {
            if chat_fast_path
                && let Response::Sse(message) = response
                && !message.is_done()
                && let Some(data) = message.data()
                && let Ok(chunk) = serde_json::from_str::<ChatChunk>(data)
                // A chunk carrying usage still needs `ParsedResponse.usage` as a
                // `Value`, so hand those (one per request) to the generic path.
                && chunk.usage.is_none()
            {
                absorb_chat_chunk_wire_metadata(&chunk, &mut model_response);
                // Matches the generic parse: with no usage, a chunk yields a
                // ParsedResponse only when it carries response data, so
                // role-only frames leave `parsed_any` alone exactly as before.
                let data = chunk.into_stream_response_data();
                if data.is_some() {
                    parsed_any = true;
                    let parsed = ParsedResponse {
                        perf_ns: u64::try_from(message.perf_ns).unwrap_or_default(),
                        data,
                        usage: None,
                        sources: None,
                    };
                    parsed_content |= reduce_parsed_response(
                        &parsed,
                        &emitter,
                        EndpointReduceAccumulators {
                            response_text: &mut response_text,
                            model_response: &mut model_response,
                            endpoint_metrics: &mut endpoint_metrics,
                            observed_usage: &mut observed_usage,
                        },
                    );
                }
                continue;
            }
            let Some(decoded) = binding.decode_response(response) else {
                continue;
            };
            let server_response: &ServerResponse = if captures_turn {
                decoded_responses.push(decoded);
                decoded_responses
                    .last()
                    .expect("just pushed a decoded response")
            } else {
                &decoded
            };
            if let Some(value) = &server_response.json {
                if captures_spec_decode
                    && let Some(stats) = extract_vllm_spec_decode_stats(value)
                    && stats.as_object().is_some_and(|object| !object.is_empty())
                {
                    spec_decode_stats = Some(stats.clone());
                }
                absorb_wire_response_metadata(value, &mut model_response);
            }
            let parsed = match parse_endpoint_response(endpoint, server_response) {
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
            let carried_content = reduce_parsed_response(
                &parsed,
                &emitter,
                EndpointReduceAccumulators {
                    response_text: &mut response_text,
                    model_response: &mut model_response,
                    endpoint_metrics: &mut endpoint_metrics,
                    observed_usage: &mut observed_usage,
                },
            );
            parsed_content |= carried_content;
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

        if captures_turn {
            let endpoint_record = EndpointRequestRecord {
                responses: decoded_responses,
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
        if let Some(payload) = spec_decode_stats {
            let completion_tokens = observed_usage
                .completion_tokens
                .and_then(|value| u64::try_from(value).ok());
            match parse_vllm_spec_decode_stats(payload, completion_tokens) {
                Ok(acceptance) => obs.on_spec_decode_acceptance(uuid, &acceptance),
                Err(error) => tracing::warn!(
                    uuid = %uuid,
                    endpoint = endpoint.descriptor().id,
                    error = %error,
                    "ignoring malformed speculative-decoding statistics"
                ),
            }
        }
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
        // `http_trace` above already took the trace facts, and the only other
        // reader (RunCapture::record_http_exchange) drops these behind its own
        // raw-artifact guard. Release them on this worker so they are not freed
        // on whichever thread later consumes the record -- under GlobalHop that
        // is the single coordinator, which serializes every request's response
        // strings through one allocator.
        let mut record = record;
        if !self.retain_raw_responses.get() {
            record.responses = Vec::new();
        }
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

    // Accept the established OpenAI-compatible `choices` envelope when `object`
    // is absent, while keeping the endpoint adapter source-strict.
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

pub(super) fn http_trace(record: &RequestRecord) -> RequestTrace {
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
thread_local! {
    /// Dispatches on this thread that skipped the body re-parse because
    /// composition had already established the exact wire image count.
    ///
    /// Parsing anyway yields the same `num_images`, so the skip is invisible to
    /// every artifact and to every metric: a build that lost it would export
    /// byte-identical output and only pay a full `serde_json` deserialize of a
    /// possibly multi-MB multimodal body on each timed dispatch. Without a count
    /// there is no signal at all that the fast path is still taken.
    ///
    /// Thread-local rather than a global counter so concurrently running tests
    /// cannot perturb each other's reading.
    static BODY_PARSE_SKIPS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clock::Clock;
    use crate::endpoints::PreparedEndpoint;
    use crate::metrics::NativeMetricsObserver;
    use crate::metrics_core::{MetricsConfig, Phase};
    use crate::transport::core::SseMessage;
    use crate::transport::http::transport::endpoint_binding::decode_sse_response;
    use crate::transport::reduce::absorb_usage;
    use axum::{Router, http::header, response::IntoResponse, routing::post};
    use std::cell::{Cell, RefCell};
    use std::io::Read;
    use std::rc::Rc;
    use std::sync::Arc;

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

    /// Discards every observation; these tests assert only the returned payload.
    struct SilentObserver;

    impl RequestObserver for SilentObserver {
        fn on_arrival(&self, _: uuid::Uuid, _: f64, _: usize, _: usize) {}
        fn on_admit(&self, _: uuid::Uuid, _: f64, _: usize) {}
        fn on_token(&self, _: uuid::Uuid, _: f64) {}
        fn on_usage(&self, _: uuid::Uuid, _: ObservedUsage) {}
        fn on_terminal(&self, _: uuid::Uuid, _: ReplayTerminalStatus) {}
    }

    /// Captures the bounded path lifecycle without retaining a response record.
    struct TerminalObserver {
        admits: Cell<usize>,
        terminals: RefCell<Vec<ReplayTerminalStatus>>,
    }

    impl TerminalObserver {
        fn new() -> Self {
            Self {
                admits: Cell::new(0),
                terminals: RefCell::new(Vec::new()),
            }
        }
    }

    impl RequestObserver for TerminalObserver {
        fn on_arrival(&self, _: uuid::Uuid, _: f64, _: usize, _: usize) {}
        fn on_admit(&self, _: uuid::Uuid, _: f64, _: usize) {
            self.admits.set(self.admits.get() + 1);
        }
        fn on_token(&self, _: uuid::Uuid, _: f64) {}
        fn on_usage(&self, _: uuid::Uuid, _: ObservedUsage) {}
        fn on_terminal(&self, _: uuid::Uuid, status: ReplayTerminalStatus) {
            self.terminals.borrow_mut().push(status);
        }
    }

    fn bounded_request(
        body: Option<crate::body_plan::RequestBody>,
        cancel_after_ns: Option<i64>,
    ) -> Request {
        Request {
            uuid: uuid::Uuid::new_v4(),
            input_length: 2,
            max_output_tokens: 2,
            prompt_text: Some("bounded decision".to_string()),
            body,
            headers: std::collections::BTreeMap::new(),
            parameters: std::collections::BTreeMap::new(),
            endpoint_path: None,
            streaming: true,
            x_correlation_id: None,
            is_final_turn: true,
            cancel_after_ns,
            url_index: None,
            image_count: None,
            recorded_api_time_ns: None,
            recorded_ttft_ns: None,
        }
    }

    fn bounded_sink(base: &str) -> TransportSink {
        let clock = crate::clock::RealClock::new();
        TransportSink::new_multi_configured(
            clock.clone(),
            clock.now_ns(),
            std::slice::from_ref(&base.to_string()),
            "m",
            crate::transport::http::TransportSinkConfig::default(),
        )
        .unwrap()
    }

    async fn spec_decode_chat_handler() -> impl IntoResponse {
        let body = concat!(
            "data: {\"id\":\"x\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"a\"},\"finish_reason\":null}]}\n\n",
            "data: {\"id\":\"x\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
            "data: {\"id\":\"x\",\"object\":\"chat.completion.chunk\",\"choices\":[],\"usage\":{\"prompt_tokens\":3,\"completion_tokens\":2},\"metrics\":{\"speculative_decoding\":{\"mean_acceptance_length\":3.25,\"draft_acceptance_rate\":0.5625,\"acceptance_histogram\":[1,1,2,3,1],\"num_accepted_draft_tokens\":18,\"num_draft_tokens\":32,\"num_spec_steps\":8,\"num_spec_tokens\":4,\"per_step_accepted\":[2,3,1,4,2,0,3,3],\"per_step_drafted\":[4,4,4,4,4,4,4,4]}}}\n\n",
            "data: [DONE]\n\n",
        );
        ([(header::CONTENT_TYPE, "text/event-stream")], body)
    }

    async fn spawn_spec_decode_chat_mock() -> String {
        let app = Router::new().route("/v1/chat/completions", post(spec_decode_chat_handler));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        format!("http://{address}")
    }

    #[tokio::test]
    async fn trailing_usage_spec_decode_metrics_reach_the_terminal_record() {
        let local = tokio::task::LocalSet::new();
        local
            .run_until(async {
                let base = spawn_spec_decode_chat_mock().await;
                let clock = crate::clock::RealClock::new();
                let sink = TransportSink::new_multi_configured(
                    clock.clone(),
                    clock.now_ns(),
                    std::slice::from_ref(&base),
                    "m",
                    crate::transport::http::TransportSinkConfig::default(),
                )
                .unwrap();
                let endpoint = prepared_streaming("chat");
                let observer = NativeMetricsObserver::new(clock, 0, MetricsConfig::default());
                let uuid = uuid::Uuid::new_v4();
                let mut request = bounded_request(None, None);
                request.uuid = uuid;
                observer.register_metadata(
                    uuid,
                    crate::metrics::RequestMetricMetadata {
                        phase: Phase::Profiling,
                        ..crate::metrics::RequestMetricMetadata::default()
                    },
                );
                observer.on_arrival(uuid, 0.0, 2, 2);
                let on_first_token = |_: i64| {};

                let dispatch = sink
                    .dispatch_prepared_endpoint_collect_record_with_hooks(
                        request,
                        endpoint.as_ref(),
                        "m",
                        EndpointDispatchHooks::new(
                            &observer,
                            &on_first_token,
                            None,
                            TurnDataPolicy::ordinary(),
                        ),
                    )
                    .await
                    .unwrap();

                assert_eq!(dispatch.result.terminal, ReplayTerminalStatus::Completed);
                let record = observer.snapshot_record(uuid, 0).expect("terminal record");
                assert!(!record.errored);
                let acceptance = record
                    .spec_decode_acceptance
                    .expect("trailing usage metrics reach the observer record");
                assert_eq!(acceptance.num_spec_steps, 8);
                assert_eq!(acceptance.num_accepted_draft_tokens, 18);
                assert_eq!(acceptance.num_draft_tokens, 32);
                assert_eq!(acceptance.completion_tokens, Some(2));
            })
            .await;
    }

    /// Dispatch one streaming chat turn through the real endpoint-aware path at
    /// the given artifact-capture flags, returning the canonical request payload
    /// the sink handed back.
    async fn dispatch_payload_at(base: &str, capture_raw: bool) -> Bytes {
        let clock = crate::clock::RealClock::new();
        let sink = TransportSink::new_multi_configured(
            clock.clone(),
            clock.now_ns(),
            std::slice::from_ref(&base.to_string()),
            "m",
            crate::transport::http::TransportSinkConfig {
                capture_raw,
                ..crate::transport::http::TransportSinkConfig::default()
            },
        )
        .unwrap();
        let endpoint = prepared_streaming("chat");
        let request = Request {
            uuid: uuid::Uuid::new_v4(),
            input_length: 2,
            max_output_tokens: 2,
            prompt_text: Some("hello world".to_string()),
            body: None,
            headers: std::collections::BTreeMap::new(),
            parameters: std::collections::BTreeMap::new(),
            endpoint_path: None,
            streaming: true,
            x_correlation_id: None,
            is_final_turn: true,
            cancel_after_ns: None,
            url_index: None,
            image_count: None,
            recorded_api_time_ns: None,
            recorded_ttft_ns: None,
        };
        let observer = SilentObserver;
        let on_first_token = |_: i64| {};
        sink.dispatch_prepared_endpoint_collect_record_with_hooks(
            request,
            endpoint.as_ref(),
            "m",
            EndpointDispatchHooks::new(
                &observer,
                &on_first_token,
                None,
                TurnDataPolicy::ordinary(),
            ),
        )
        .await
        .unwrap()
        .request_payload
    }

    /// Dispatch the test server's two one-byte chat deltas through the bounded
    /// no-record path. The helper intentionally returns only the reader bytes:
    /// this path has no `RequestRecord` to inspect or retain.
    async fn dispatch_bounded_decision_at(
        base: &str,
        max_decision_bytes: usize,
    ) -> anyhow::Result<Vec<u8>> {
        let clock = crate::clock::RealClock::new();
        let sink = TransportSink::new_multi_configured(
            clock.clone(),
            clock.now_ns(),
            std::slice::from_ref(&base.to_string()),
            "m",
            crate::transport::http::TransportSinkConfig::default(),
        )?;
        let endpoint = prepared_streaming("chat");
        let request = Request {
            uuid: uuid::Uuid::new_v4(),
            input_length: 2,
            max_output_tokens: 2,
            prompt_text: Some("bounded decision".to_string()),
            body: None,
            headers: std::collections::BTreeMap::new(),
            parameters: std::collections::BTreeMap::new(),
            endpoint_path: None,
            streaming: true,
            x_correlation_id: None,
            is_final_turn: true,
            cancel_after_ns: None,
            url_index: None,
            image_count: None,
            recorded_api_time_ns: None,
            recorded_ttft_ns: None,
        };
        let observer = SilentObserver;
        let on_first_token = |_: i64| {};
        let mut reader = sink
            .dispatch_prepared_endpoint_bounded_decision(
                request,
                endpoint.as_ref(),
                "m",
                &observer,
                &on_first_token,
                BoundedDecisionMode::new(max_decision_bytes)?,
            )
            .await?;
        let mut bytes = Vec::new();
        reader.read_to_end(&mut bytes)?;
        Ok(bytes)
    }

    #[tokio::test]
    async fn bounded_decision_streaming_admits_exact_limit_without_a_terminal_record() {
        let local = tokio::task::LocalSet::new();
        local
            .run_until(async {
                let base = crate::test_util::spawn_mock().await;
                let bytes = dispatch_bounded_decision_at(&base, 2).await.unwrap();
                assert_eq!(bytes, b"ab");
            })
            .await;
    }

    #[tokio::test]
    async fn bounded_decision_streaming_aborts_on_the_first_byte_over_limit() {
        let local = tokio::task::LocalSet::new();
        local
            .run_until(async {
                let base = crate::test_util::spawn_mock().await;
                let sink = bounded_sink(&base);
                let endpoint = prepared_streaming("chat");
                let observer = TerminalObserver::new();
                let on_first_token = |_: i64| {};
                let error = sink
                    .dispatch_prepared_endpoint_bounded_decision(
                        bounded_request(None, None),
                        endpoint.as_ref(),
                        "m",
                        &observer,
                        &on_first_token,
                        BoundedDecisionMode::new(1).unwrap(),
                    )
                    .await
                    .unwrap_err();
                assert!(
                    error
                        .to_string()
                        .contains("exceeds the selected 1-byte limit")
                );
                assert_eq!(
                    observer.terminals.into_inner(),
                    vec![ReplayTerminalStatus::Failed]
                );
            })
            .await;
    }

    #[tokio::test]
    async fn bounded_decision_emits_one_failed_terminal_when_request_preparation_fails() {
        let local = tokio::task::LocalSet::new();
        local
            .run_until(async {
                let operation = crate::body_plan::PreparedWsOperation::new(
                    [crate::body_plan::PreparedWsMessage::text(
                        Bytes::from_static(br#"{"type":"response.create"}"#),
                        crate::body_plan::PreparedWsMessageRole::MeasuredInput,
                    )],
                    None,
                );
                let request = bounded_request(
                    Some(crate::body_plan::RequestBody::WebSocket(Arc::new(
                        operation,
                    ))),
                    None,
                );
                let sink = bounded_sink("http://127.0.0.1:9");
                let endpoint = prepared_streaming("chat");
                let observer = TerminalObserver::new();
                let on_first_token = |_: i64| {};

                assert!(
                    sink.dispatch_prepared_endpoint_bounded_decision(
                        request,
                        endpoint.as_ref(),
                        "m",
                        &observer,
                        &on_first_token,
                        BoundedDecisionMode::new(1).unwrap(),
                    )
                    .await
                    .is_err()
                );
                assert_eq!(
                    observer.terminals.into_inner(),
                    vec![ReplayTerminalStatus::Failed]
                );
            })
            .await;
    }

    #[tokio::test]
    async fn bounded_decision_emits_one_cancelled_terminal_for_post_admit_cancellation() {
        let local = tokio::task::LocalSet::new();
        local
            .run_until(async {
                let sink = bounded_sink("http://127.0.0.1:9");
                let endpoint = prepared_streaming("chat");
                let observer = TerminalObserver::new();
                let on_first_token = |_: i64| {};

                assert!(
                    sink.dispatch_prepared_endpoint_bounded_decision(
                        bounded_request(
                            Some(crate::body_plan::RequestBody::wire(Bytes::from_static(
                                b"{}"
                            ))),
                            Some(1),
                        ),
                        endpoint.as_ref(),
                        "m",
                        &observer,
                        &on_first_token,
                        BoundedDecisionMode::new(1).unwrap(),
                    )
                    .await
                    .is_err()
                );
                assert_eq!(
                    observer.terminals.into_inner(),
                    vec![ReplayTerminalStatus::Canceled]
                );
            })
            .await;
    }

    #[tokio::test]
    async fn bounded_decision_emits_one_cancelled_terminal_when_pending_network_future_is_aborted()
    {
        let local = tokio::task::LocalSet::new();
        local
            .run_until(async {
                let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
                let address = listener.local_addr().unwrap();
                let (accepted_tx, accepted_rx) = tokio::sync::oneshot::channel();
                let server = tokio::task::spawn_local(async move {
                    let (_stream, _) = listener.accept().await.unwrap();
                    let _ = accepted_tx.send(());
                    futures::future::pending::<()>().await;
                });
                let sink = bounded_sink(&format!("http://{address}"));
                let endpoint = prepared_streaming("chat");
                let observer = Rc::new(TerminalObserver::new());
                let observer_for_dispatch = observer.clone();
                let on_first_token = |_: i64| {};
                let dispatch = tokio::task::spawn_local(async move {
                    sink.dispatch_prepared_endpoint_bounded_decision(
                        bounded_request(
                            Some(crate::body_plan::RequestBody::wire(Bytes::from_static(
                                b"{}",
                            ))),
                            None,
                        ),
                        endpoint.as_ref(),
                        "m",
                        observer_for_dispatch.as_ref(),
                        &on_first_token,
                        BoundedDecisionMode::new(1).unwrap(),
                    )
                    .await
                });

                accepted_rx.await.unwrap();
                assert_eq!(observer.admits.get(), 1);
                dispatch.abort();
                assert!(dispatch.await.unwrap_err().is_cancelled());
                server.abort();
                let _ = server.await;

                assert_eq!(
                    observer.terminals.borrow().as_slice(),
                    [ReplayTerminalStatus::Canceled]
                );
            })
            .await;
    }

    /// Pin BOTH directions of the canonical-payload gate at the seam that decides
    /// it, which no product artifact can observe on its own.
    ///
    /// A wrongly-CLOSED gate loses an artifact and is caught downstream
    /// (`test_exact_fold_ab_parity`, `test_http_raw_capture`). A wrongly-OPEN one
    /// is silent: the artifacts are identical and only the per-dispatch
    /// allocation returns. `TransportSinkConfig::default()` deliberately opens the
    /// gate, so any future construction site that forgets to stamp the run's
    /// artifact selection reverts the saving with no other signal — this assertion
    /// is that signal.
    #[tokio::test]
    async fn request_payload_is_taken_only_when_an_artifact_consumes_it() {
        let local = tokio::task::LocalSet::new();
        local
            .run_until(async {
                let base = crate::test_util::spawn_mock().await;

                // Closed: the raw artifact is not selected, so nothing would read
                // the payload and no handle is taken.
                //
                // `inputs.json` is deliberately NOT a second leg here. It is
                // projected from the resident dataset at finalize
                // (`compose_sidecars::build_up_front_input_sessions`) and a run
                // that cannot be projected that way is rejected before any phase
                // runs, so no dispatched payload reaches it. Re-introducing a
                // payload-sourced `inputs.json` must reopen this gate — and this
                // assertion is what fails if it does not.
                assert!(
                    dispatch_payload_at(&base, false).await.is_empty(),
                    "no artifact consumes the canonical payload, so the gate must \
                     not take a handle on the assembled body"
                );

                // Open: the raw artifact reads the payload back verbatim.
                let raw = dispatch_payload_at(&base, true).await;
                assert!(
                    !raw.is_empty(),
                    "the raw artifact consumes the canonical payload, so the gate \
                     must capture it"
                );

                // And what is recorded is the canonical chat body actually sent.
                assert_eq!(
                    serde_json::from_slice::<Value>(&raw).unwrap(),
                    crate::endpoints::chat_request_body("m", &[("user", "hello world")], 2),
                );
            })
            .await;
    }

    /// Records the endpoint metrics one dispatch observed.
    #[derive(Default)]
    struct ImageObserver {
        num_images: Cell<Option<usize>>,
    }

    impl RequestObserver for ImageObserver {
        fn on_arrival(&self, _: uuid::Uuid, _: f64, _: usize, _: usize) {}
        fn on_admit(&self, _: uuid::Uuid, _: f64, _: usize) {}
        fn on_token(&self, _: uuid::Uuid, _: f64) {}
        fn on_endpoint_metrics(&self, _: uuid::Uuid, metrics: ObservedEndpointMetrics) {
            self.num_images.set(metrics.num_images);
        }
        fn on_terminal(&self, _: uuid::Uuid, _: ReplayTerminalStatus) {}
    }

    /// Dispatch one multimodal chat body, optionally handing dispatch the image
    /// count composition already established, and report what was observed.
    async fn dispatch_images_at(base: &str, known: Option<u32>) -> (Option<usize>, u64) {
        let clock = crate::clock::RealClock::new();
        let sink = TransportSink::new_multi_configured(
            clock.clone(),
            clock.now_ns(),
            std::slice::from_ref(&base.to_string()),
            "m",
            crate::transport::http::TransportSinkConfig::default(),
        )
        .unwrap();
        let endpoint = prepared_streaming("chat");
        let body = serde_json::json!({
            "model": "m",
            "stream": true,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "what is this"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AB=="}}
            ]}]
        });
        let request = Request {
            uuid: uuid::Uuid::new_v4(),
            input_length: 2,
            max_output_tokens: 2,
            prompt_text: None,
            body: Some(crate::body_plan::RequestBody::wire(Bytes::from(
                serde_json::to_vec(&body).unwrap(),
            ))),
            headers: std::collections::BTreeMap::new(),
            parameters: std::collections::BTreeMap::new(),
            endpoint_path: None,
            streaming: true,
            x_correlation_id: None,
            is_final_turn: true,
            cancel_after_ns: None,
            url_index: None,
            image_count: known,
            recorded_api_time_ns: None,
            recorded_ttft_ns: None,
        };
        let observer = ImageObserver::default();
        let on_first_token = |_: i64| {};
        let before = BODY_PARSE_SKIPS.get();
        sink.dispatch_prepared_endpoint_collect_record_with_hooks(
            request,
            endpoint.as_ref(),
            "m",
            EndpointDispatchHooks::new(
                &observer,
                &on_first_token,
                None,
                TurnDataPolicy::ordinary(),
            ),
        )
        .await
        .unwrap();
        (observer.num_images.get(), BODY_PARSE_SKIPS.get() - before)
    }

    /// Pin that a composed image count actually replaces the body re-parse, and
    /// that it answers what the parse would have.
    ///
    /// Only the second half is observable from any artifact: both paths report the
    /// same `num_images`, so a build that stopped trusting the composed count
    /// would export identical output and silently pay a full multimodal-body
    /// deserialize on every timed dispatch. The skip count is the only signal.
    #[tokio::test]
    async fn a_composed_image_count_replaces_the_body_reparse() {
        let local = tokio::task::LocalSet::new();
        local
            .run_until(async {
                let base = crate::test_util::spawn_mock().await;

                let (parsed_images, parsed_skips) = dispatch_images_at(&base, None).await;
                assert_eq!(parsed_skips, 0, "an unknown count must parse the body");

                let (known_images, known_skips) = dispatch_images_at(&base, Some(2)).await;
                assert_eq!(
                    known_skips, 1,
                    "an established count must skip the body parse"
                );

                assert_eq!(
                    known_images, parsed_images,
                    "the composed count must equal what parsing the same body reports"
                );
                assert_eq!(known_images, Some(2));
            })
            .await;
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

        let chat_without_object =
            SseMessage::parse(r#"data: {"choices":[{"delta":{"content":"compat"}}]}"#, 12);
        let chat_endpoint = prepared_streaming("chat");
        let chat_adapter = WorkerPreparedEndpointAdapter(chat_endpoint.as_ref());
        assert!(meaningful_endpoint_response(
            &chat_adapter,
            &decode_sse_response(&chat_without_object).unwrap()
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
