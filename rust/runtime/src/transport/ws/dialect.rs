// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Endpoint-owned application-event classification for WebSocket execution.

use bytes::Bytes;
use serde::Deserialize;
use serde_json::Value;

use crate::body_plan::{
    PreparedWsMessage, PreparedWsMessageRole, PreparedWsOpcode, PreparedWsOperation,
};
use crate::dispatch::collector::ReplayTerminalStatus;
use crate::dispatch::sink::ObservedUsage;
use crate::endpoints::UsageView;
use crate::endpoints::WebSocketDialect;
use crate::transport::ws::RoundTripTimingState;

const OPERATION_METADATA_KEY: &str = "_aiperf_ws_operation";
const MAX_RESPONSE_METADATA_PAIRS: usize = 16;

#[derive(Clone, Debug)]
pub(crate) struct OperationCorrelation {
    operation_id: Option<String>,
    client_event_ids: Box<[String]>,
}

impl OperationCorrelation {
    pub(crate) fn supports_reused_socket(&self) -> bool {
        self.operation_id.is_some()
    }
}

#[derive(Clone, Debug)]
pub(crate) struct CorrelatedOperation {
    operation: PreparedWsOperation,
    correlation: OperationCorrelation,
}

impl CorrelatedOperation {
    pub(crate) fn operation(&self) -> &PreparedWsOperation {
        &self.operation
    }

    pub(crate) fn correlation(&self) -> &OperationCorrelation {
        &self.correlation
    }
}

pub(crate) fn correlate_operation(
    request: &PreparedWsOperation,
    dialect: WebSocketDialect,
    operation_id: &str,
) -> anyhow::Result<CorrelatedOperation> {
    anyhow::ensure!(
        !operation_id.is_empty(),
        "websocket operation correlation identity must not be empty"
    );
    let mut client_event_ids = Vec::with_capacity(request.messages().len());
    let mut has_operation_marker = false;
    let messages = request
        .messages()
        .iter()
        .enumerate()
        .map(|(index, message)| {
            anyhow::ensure!(
                message.opcode() == PreparedWsOpcode::Text,
                "websocket operation correlation requires text JSON messages"
            );
            let mut value: Value = serde_json::from_slice(message.payload())?;
            let object = value
                .as_object_mut()
                .ok_or_else(|| anyhow::anyhow!("websocket client event must be a JSON object"))?;
            let event_type = object
                .get("type")
                .and_then(Value::as_str)
                .ok_or_else(|| anyhow::anyhow!("websocket client event has no string type"))?
                .to_owned();
            match dialect {
                WebSocketDialect::Responses => {
                    anyhow::ensure!(
                        event_type == "response.create",
                        "Responses websocket operation contains unsupported client event {event_type:?}"
                    );
                    has_operation_marker |= insert_operation_metadata(object, operation_id)?;
                }
                WebSocketDialect::Realtime => {
                    let client_event_id = format!("{operation_id}:{index}");
                    anyhow::ensure!(
                        !object.contains_key("event_id"),
                        "Realtime event_id is reserved for operation correlation"
                    );
                    object.insert("event_id".to_owned(), Value::String(client_event_id.clone()));
                    client_event_ids.push(client_event_id);
                    if event_type == "response.create" {
                        let response = object
                            .entry("response")
                            .or_insert_with(|| Value::Object(Default::default()))
                            .as_object_mut()
                            .ok_or_else(|| {
                                anyhow::anyhow!(
                                    "Realtime response.create response must be an object"
                                )
                            })?;
                        has_operation_marker |=
                            insert_operation_metadata(response, operation_id)?;
                    }
                }
            }
            let payload = serde_json::to_vec(&value)?;
            Ok(PreparedWsMessage::new(
                message.opcode(),
                Bytes::from(payload),
                message.role(),
            ))
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    Ok(CorrelatedOperation {
        operation: request.with_messages(messages),
        correlation: OperationCorrelation {
            operation_id: has_operation_marker.then(|| operation_id.to_owned()),
            client_event_ids: client_event_ids.into_boxed_slice(),
        },
    })
}

fn insert_operation_metadata(
    object: &mut serde_json::Map<String, Value>,
    operation_id: &str,
) -> anyhow::Result<bool> {
    let metadata = object
        .entry("metadata")
        .or_insert_with(|| Value::Object(Default::default()))
        .as_object_mut()
        .ok_or_else(|| anyhow::anyhow!("websocket response metadata must be an object"))?;
    anyhow::ensure!(
        metadata.len() <= MAX_RESPONSE_METADATA_PAIRS,
        "websocket response metadata exceeds the public {MAX_RESPONSE_METADATA_PAIRS}-pair limit"
    );
    if metadata.len() == MAX_RESPONSE_METADATA_PAIRS
        || metadata.contains_key(OPERATION_METADATA_KEY)
    {
        return Ok(false);
    }
    metadata.insert(
        OPERATION_METADATA_KEY.to_owned(),
        Value::String(operation_id.to_owned()),
    );
    Ok(true)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum EventDisposition {
    Attributed { is_terminal: bool },
    AttributedError,
    Unattributed,
    UnsafeUnattributedError,
}

/// One complete Responses application event.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum ResponsesEvent {
    /// The server-assigned identity for this logical response.
    Created {
        response_id: String,
        operation_id: Option<String>,
    },
    /// Non-empty user-visible content.
    Content {
        response_id: Option<String>,
        content: Bytes,
    },
    /// Non-visible reasoning delta.
    Reasoning { response_id: Option<String> },
    /// Binary audio carried inside a Realtime JSON event.
    Audio { response_id: String },
    /// Endpoint usage envelope.
    Usage {
        response_id: String,
        usage: ObservedUsage,
    },
    /// A continuation identity was rejected before visible output.
    RetriableContinuationRejection,
    /// Application error, optionally correlated to a client event.
    Error {
        client_event_id: Option<String>,
        code: Option<String>,
        message: String,
    },
    /// Logical operation completion.
    Terminal {
        response_id: String,
        usage: ObservedUsage,
        content: Bytes,
        status: ReplayTerminalStatus,
    },
    /// A control or irrelevant application envelope.
    Ignored,
}

fn response_id(event: &Value, context: &str) -> anyhow::Result<String> {
    event
        .get("response_id")
        .or_else(|| {
            event
                .get("response")
                .and_then(|response| response.get("id"))
        })
        .and_then(Value::as_str)
        .filter(|identity| !identity.is_empty())
        .map(str::to_owned)
        .ok_or_else(|| anyhow::anyhow!("{context} has no response identity"))
}

fn operation_id(event: &Value) -> Option<String> {
    event
        .get("response")
        .and_then(|response| response.get("metadata"))
        .and_then(|metadata| metadata.get(OPERATION_METADATA_KEY))
        .and_then(Value::as_str)
        .filter(|identity| !identity.is_empty())
        .map(str::to_owned)
}

#[derive(Deserialize)]
struct ResponsesErrorEnvelope {
    #[serde(default)]
    code: Option<String>,
    message: String,
}

#[derive(Deserialize)]
struct RealtimeErrorEnvelope {
    error: RealtimeErrorDetails,
}

#[derive(Deserialize)]
struct RealtimeErrorDetails {
    #[serde(default)]
    code: Option<String>,
    #[serde(default)]
    event_id: Option<String>,
    message: String,
}

/// Classify one complete Responses text message.
pub(crate) fn classify_responses_event(
    payload: &[u8],
    is_text: bool,
) -> anyhow::Result<ResponsesEvent> {
    if !is_text {
        anyhow::bail!("Responses WebSocket requires text application messages");
    }
    let event: Value = serde_json::from_slice(payload)?;
    let kind = event
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or_default();
    match kind {
        "response.created" => Ok(ResponsesEvent::Created {
            response_id: response_id(&event, "response.created")?,
            operation_id: operation_id(&event),
        }),
        "response.output_text.delta" => Ok(event
            .get("delta")
            .and_then(Value::as_str)
            .filter(|delta| !delta.is_empty())
            .map_or(ResponsesEvent::Ignored, |delta| ResponsesEvent::Content {
                response_id: None,
                content: Bytes::copy_from_slice(delta.as_bytes()),
            })),
        "response.reasoning.delta" => Ok(ResponsesEvent::Reasoning { response_id: None }),
        "response.completed" => Ok(ResponsesEvent::Terminal {
            response_id: response_id(&event, "response.completed")?,
            usage: observed_usage(event.get("usage").or_else(|| {
                event
                    .get("response")
                    .and_then(|response| response.get("usage"))
            })),
            content: terminal_content(&event),
            status: ReplayTerminalStatus::Completed,
        }),
        "response.failed" => Ok(ResponsesEvent::Terminal {
            response_id: response_id(&event, "response.failed")?,
            usage: observed_usage(event.get("usage").or_else(|| {
                event
                    .get("response")
                    .and_then(|response| response.get("usage"))
            })),
            content: terminal_content(&event),
            status: ReplayTerminalStatus::Failed,
        }),
        "error" => {
            let error: ResponsesErrorEnvelope = serde_json::from_value(event)?;
            Ok(ResponsesEvent::Error {
                client_event_id: None,
                code: error.code.filter(|code| !code.is_empty()),
                message: error.message,
            })
        }
        "response.incomplete" => Ok(ResponsesEvent::Terminal {
            response_id: response_id(&event, "response.incomplete")?,
            usage: observed_usage(event.get("usage").or_else(|| {
                event
                    .get("response")
                    .and_then(|response| response.get("usage"))
            })),
            content: terminal_content(&event),
            status: ReplayTerminalStatus::Failed,
        }),
        "response.usage" => Ok(ResponsesEvent::Usage {
            response_id: response_id(&event, "response.usage")?,
            usage: observed_usage(event.get("usage").or_else(|| {
                event
                    .get("response")
                    .and_then(|response| response.get("usage"))
            })),
        }),
        "response.continuation_rejected" => Ok(ResponsesEvent::RetriableContinuationRejection),
        _ => Ok(ResponsesEvent::Ignored),
    }
}

/// Classify one complete Realtime text JSON event into the shared operation
/// lifecycle vocabulary. Audio remains in the raw event stream for the
/// endpoint parser; it is not a text-token or round-trip sample.
pub(crate) fn classify_realtime_event(
    payload: &[u8],
    is_text: bool,
) -> anyhow::Result<ResponsesEvent> {
    if !is_text {
        anyhow::bail!("Realtime WebSocket requires text JSON application messages");
    }
    let event: Value = serde_json::from_slice(payload)?;
    match event
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or_default()
    {
        "response.created" => Ok(ResponsesEvent::Created {
            response_id: response_id(&event, "response.created")?,
            operation_id: operation_id(&event),
        }),
        "response.output_text.delta" => {
            let response_id = response_id(&event, "response.output_text.delta")?;
            Ok(event
                .get("delta")
                .and_then(Value::as_str)
                .filter(|delta| !delta.is_empty())
                .map_or(ResponsesEvent::Ignored, |delta| ResponsesEvent::Content {
                    response_id: Some(response_id),
                    content: Bytes::copy_from_slice(delta.as_bytes()),
                }))
        }
        "response.output_audio.delta" => Ok(ResponsesEvent::Audio {
            response_id: response_id(&event, "response.output_audio.delta")?,
        }),
        "response.done" => {
            let response = event
                .get("response")
                .ok_or_else(|| anyhow::anyhow!("Realtime response.done has no response object"))?;
            let status = match response.get("status").and_then(Value::as_str) {
                Some("completed") => ReplayTerminalStatus::Completed,
                Some("cancelled" | "canceled") => ReplayTerminalStatus::Canceled,
                Some("failed" | "incomplete") => ReplayTerminalStatus::Failed,
                Some(status) => {
                    anyhow::bail!("Realtime response.done has unknown response status {status:?}")
                }
                None => anyhow::bail!("Realtime response.done has no response status"),
            };
            Ok(ResponsesEvent::Terminal {
                response_id: response_id(&event, "response.done")?,
                usage: observed_usage(event.get("usage").or_else(|| response.get("usage"))),
                content: terminal_content(&event),
                status,
            })
        }
        "error" => {
            let error: RealtimeErrorEnvelope = serde_json::from_value(event)?;
            Ok(ResponsesEvent::Error {
                client_event_id: error.error.event_id.filter(|identity| !identity.is_empty()),
                code: error.error.code.filter(|code| !code.is_empty()),
                message: error.error.message,
            })
        }
        _ => Ok(ResponsesEvent::Ignored),
    }
}

/// Rebuild the materialized full-history request without a stale continuation.
pub(crate) fn full_history_retry(request: &PreparedWsOperation) -> Option<PreparedWsOperation> {
    let messages = request
        .messages()
        .iter()
        .map(|message| {
            if message.role() != PreparedWsMessageRole::MeasuredInput
                || message.opcode() != PreparedWsOpcode::Text
            {
                return Some(message.clone());
            }
            let mut value: Value = serde_json::from_slice(message.payload()).ok()?;
            value.as_object_mut()?.remove("previous_response_id");
            let payload = serde_json::to_vec(&value).ok()?;
            Some(PreparedWsMessage::text(
                Bytes::from(payload),
                PreparedWsMessageRole::MeasuredInput,
            ))
        })
        .collect::<Option<Vec<_>>>()?;
    Some(request.with_messages(messages))
}

/// Bind a prepared Responses operation to the preceding response on its
/// affinity-owned connection.
pub(crate) fn with_previous_response_id(
    request: &PreparedWsOperation,
    response_id: &str,
) -> Option<PreparedWsOperation> {
    let messages = request
        .messages()
        .iter()
        .map(|message| {
            if message.role() != PreparedWsMessageRole::MeasuredInput
                || message.opcode() != PreparedWsOpcode::Text
            {
                return Some(message.clone());
            }
            let mut value: Value = serde_json::from_slice(message.payload()).ok()?;
            let object = value.as_object_mut()?;
            let input = object.get_mut("input")?.as_array_mut()?;
            let last_assistant = input
                .iter()
                .rposition(|item| item.get("role").and_then(Value::as_str) == Some("assistant"))?;
            input.drain(..=last_assistant);
            (!input.is_empty()).then_some(())?;
            object.insert(
                "previous_response_id".to_owned(),
                Value::String(response_id.to_owned()),
            );
            Some(PreparedWsMessage::text(
                Bytes::from(serde_json::to_vec(&value).ok()?),
                PreparedWsMessageRole::MeasuredInput,
            ))
        })
        .collect::<Option<Vec<_>>>()?;
    Some(request.with_messages(messages))
}

/// Scalar lifecycle state for one turn-serialized operation.
#[derive(Debug, Default)]
pub(crate) struct TurnOperationState {
    timing: RoundTripTimingState,
    visible_content_bytes: usize,
    visible_content_digest: blake3::Hasher,
    has_observer_fact: bool,
    has_terminal: bool,
    response_id: Option<String>,
    operation_id: Option<String>,
    client_event_ids: Box<[String]>,
    is_reused_socket: bool,
    has_verified_correlation: bool,
    unattributed_response_id: Option<String>,
}

impl TurnOperationState {
    pub(crate) fn new(correlation: &OperationCorrelation, is_reused_socket: bool) -> Self {
        Self {
            operation_id: correlation.operation_id.clone(),
            client_event_ids: correlation.client_event_ids.clone(),
            is_reused_socket,
            ..Self::default()
        }
    }

    pub(crate) fn on_correlated_event(
        &mut self,
        event: &ResponsesEvent,
        timestamp_ns: i64,
    ) -> anyhow::Result<EventDisposition> {
        match event {
            ResponsesEvent::Created {
                response_id,
                operation_id,
            } if self.operation_id.is_some()
                && self.operation_id.as_deref() == operation_id.as_deref() =>
            {
                self.has_verified_correlation = true;
            }
            ResponsesEvent::Created { .. }
                if self.operation_id.is_none() && !self.is_reused_socket => {}
            ResponsesEvent::Created {
                response_id,
                operation_id: _,
            } => {
                anyhow::ensure!(
                    self.response_id.is_none(),
                    "websocket response identity changed after the correlated response was created"
                );
                match self.unattributed_response_id.as_deref() {
                    None => self.unattributed_response_id = Some(response_id.clone()),
                    Some(stale) if stale == response_id => {
                        anyhow::bail!(
                            "uncorrelated websocket response identity {response_id:?} was created twice"
                        )
                    }
                    Some(stale) => anyhow::bail!(
                        "uncorrelated websocket response identity changed from {stale:?} to {response_id:?}"
                    ),
                }
                return Ok(EventDisposition::Unattributed);
            }
            ResponsesEvent::Content {
                response_id: Some(response_id),
                ..
            }
            | ResponsesEvent::Reasoning {
                response_id: Some(response_id),
            }
            | ResponsesEvent::Audio { response_id }
            | ResponsesEvent::Usage { response_id, .. }
            | ResponsesEvent::Terminal { response_id, .. }
                if self.response_id.is_none()
                    && self.unattributed_response_id.as_deref() == Some(response_id) =>
            {
                if matches!(event, ResponsesEvent::Terminal { .. }) {
                    self.unattributed_response_id = None;
                }
                return Ok(EventDisposition::Unattributed);
            }
            ResponsesEvent::Content {
                response_id: None, ..
            }
            | ResponsesEvent::Reasoning { response_id: None }
                if self.response_id.is_none() && self.unattributed_response_id.is_some() =>
            {
                return Ok(EventDisposition::Unattributed);
            }
            ResponsesEvent::Error {
                client_event_id, ..
            } => {
                if let Some(client_event_id) = client_event_id {
                    return Ok(
                        if self
                            .client_event_ids
                            .iter()
                            .any(|expected| expected == client_event_id)
                        {
                            EventDisposition::AttributedError
                        } else {
                            EventDisposition::Unattributed
                        },
                    );
                }
                return Ok(if !self.is_reused_socket {
                    EventDisposition::AttributedError
                } else {
                    EventDisposition::UnsafeUnattributedError
                });
            }
            ResponsesEvent::Ignored => return Ok(EventDisposition::Unattributed),
            ResponsesEvent::RetriableContinuationRejection
                if self.is_reused_socket && self.response_id.is_none() =>
            {
                return Ok(EventDisposition::UnsafeUnattributedError);
            }
            _ => {}
        }
        self.on_event(event, timestamp_ns)
            .map(|is_terminal| EventDisposition::Attributed { is_terminal })
    }

    pub(crate) fn on_send(&mut self, timestamp_ns: i64) {
        self.timing.on_measured_input_flushed(timestamp_ns);
    }

    pub(crate) fn on_event(
        &mut self,
        event: &ResponsesEvent,
        _timestamp_ns: i64,
    ) -> anyhow::Result<bool> {
        match event {
            ResponsesEvent::Created { response_id, .. } => match self.response_id.as_deref() {
                None => self.response_id = Some(response_id.clone()),
                Some(bound) if bound == response_id => {
                    anyhow::bail!("websocket response identity {response_id:?} was created twice")
                }
                Some(bound) => anyhow::bail!(
                    "websocket response identity changed from {bound:?} to {response_id:?}"
                ),
            },
            ResponsesEvent::Content { response_id, .. }
            | ResponsesEvent::Reasoning { response_id } => {
                let bound = self.response_id.as_deref().ok_or_else(|| {
                    anyhow::anyhow!("websocket response event arrived before response.created")
                })?;
                if response_id
                    .as_deref()
                    .is_some_and(|received| received != bound)
                {
                    let received = response_id
                        .as_deref()
                        .ok_or_else(|| anyhow::anyhow!("websocket response identity was lost"))?;
                    anyhow::bail!(
                        "websocket response identity mismatch: expected {bound:?}, received {received:?}"
                    );
                }
            }
            ResponsesEvent::Audio { response_id }
            | ResponsesEvent::Usage { response_id, .. }
            | ResponsesEvent::Terminal { response_id, .. } => {
                let bound = self.response_id.as_deref().ok_or_else(|| {
                    anyhow::anyhow!(
                        "websocket response event {response_id:?} arrived before response.created"
                    )
                })?;
                if bound != response_id {
                    anyhow::bail!(
                        "websocket response identity mismatch: expected {bound:?}, received {response_id:?}"
                    );
                }
            }
            ResponsesEvent::RetriableContinuationRejection
            | ResponsesEvent::Error { .. }
            | ResponsesEvent::Ignored => {}
        }
        if matches!(
            event,
            ResponsesEvent::Created { .. }
                | ResponsesEvent::Content { .. }
                | ResponsesEvent::Reasoning { .. }
                | ResponsesEvent::Audio { .. }
                | ResponsesEvent::Usage { .. }
                | ResponsesEvent::Terminal { .. }
                | ResponsesEvent::Error { .. }
        ) {
            self.has_observer_fact = true;
        }
        if matches!(event, ResponsesEvent::Terminal { .. }) {
            self.has_terminal = true;
        }
        Ok(self.has_terminal)
    }

    /// Return only content bytes not already emitted by earlier deltas.
    pub(crate) fn content_for_observation(
        &mut self,
        event: &ResponsesEvent,
        timestamp_ns: i64,
    ) -> anyhow::Result<Option<Bytes>> {
        let Some(content) = event.content().filter(|content| !content.is_empty()) else {
            return Ok(None);
        };
        let visible = match event {
            ResponsesEvent::Content { .. } => content.clone(),
            ResponsesEvent::Terminal { .. } => {
                if content.len() < self.visible_content_bytes {
                    anyhow::bail!(
                        "websocket terminal snapshot is shorter than its streamed prefix"
                    );
                }
                let terminal_prefix = blake3::hash(&content[..self.visible_content_bytes]);
                if terminal_prefix != self.visible_content_digest.finalize() {
                    anyhow::bail!("websocket terminal snapshot does not match its streamed prefix");
                }
                content.slice(self.visible_content_bytes..)
            }
            _ => return Ok(None),
        };
        if visible.is_empty() {
            return Ok(None);
        }
        self.visible_content_bytes = self
            .visible_content_bytes
            .checked_add(visible.len())
            .ok_or_else(|| anyhow::anyhow!("websocket visible content byte count overflowed"))?;
        self.visible_content_digest.update(&visible);
        self.timing.on_content_received(timestamp_ns);
        Ok(Some(visible))
    }

    pub(crate) const fn can_retry(&self) -> bool {
        !self.has_observer_fact && !self.has_terminal
    }

    pub(crate) const fn has_verified_correlation(&self) -> bool {
        self.has_verified_correlation
    }

    pub(crate) fn finish(
        &self,
        status: ReplayTerminalStatus,
    ) -> crate::dispatch::sink::ObservedRoundTripMetrics {
        if self.has_terminal && status == ReplayTerminalStatus::Completed {
            self.timing.finish()
        } else {
            crate::dispatch::sink::ObservedRoundTripMetrics::default()
        }
    }
}

impl ResponsesEvent {
    pub(crate) fn error_message(&self) -> Option<String> {
        match self {
            Self::Error { code, message, .. } => Some(code.as_ref().map_or_else(
                || message.clone(),
                |code| format!("{message} (code: {code})"),
            )),
            _ => None,
        }
    }

    pub(crate) fn content(&self) -> Option<&Bytes> {
        match self {
            Self::Content { content, .. } | Self::Terminal { content, .. } => Some(content),
            _ => None,
        }
    }
}

fn terminal_content(event: &Value) -> Bytes {
    let mut content = Vec::new();
    let output = event
        .get("response")
        .and_then(|response| response.get("output"))
        .or_else(|| event.get("output"))
        .and_then(Value::as_array);
    for text in output.into_iter().flatten().flat_map(|item| {
        item.get("content")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
    }) {
        if let Some(text) = text.get("text").and_then(Value::as_str) {
            content.extend_from_slice(text.as_bytes());
        }
    }
    Bytes::from(content)
}

fn observed_usage(value: Option<&Value>) -> ObservedUsage {
    let Some(usage) = value.and_then(UsageView::from_value) else {
        return ObservedUsage::default();
    };
    let as_usize = |value: Option<u64>| value.and_then(|count| usize::try_from(count).ok());
    ObservedUsage {
        prompt_tokens: as_usize(usage.prompt_tokens()),
        completion_tokens: as_usize(usage.completion_tokens()),
        total_tokens: as_usize(usage.total_tokens()),
        reasoning_tokens: as_usize(usage.reasoning_tokens()),
        prompt_cache_read_tokens: as_usize(usage.prompt_cache_read_tokens()),
        prompt_cache_write_tokens: as_usize(usage.prompt_cache_write_tokens()),
        prompt_cache_miss_tokens: as_usize(usage.prompt_cache_miss_tokens()),
        prompt_audio_tokens: as_usize(usage.prompt_audio_tokens()),
        completion_audio_tokens: as_usize(usage.completion_audio_tokens()),
        accepted_prediction_tokens: as_usize(usage.accepted_prediction_tokens()),
        rejected_prediction_tokens: as_usize(usage.rejected_prediction_tokens()),
        tool_use_prompt_tokens: as_usize(usage.tool_use_prompt_tokens()),
        prompt_audio_seconds: usage.prompt_audio_seconds(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn responses_content_and_terminal_are_distinct() {
        assert_eq!(
            classify_responses_event(
                br#"{"type":"response.output_text.delta","response_id":"r1","delta":"hello"}"#,
                true,
            )
            .expect("content event is valid"),
            ResponsesEvent::Content {
                response_id: None,
                content: Bytes::from_static(b"hello"),
            }
        );
        assert!(matches!(
            classify_responses_event(
                br#"{"type":"response.completed","response":{"id":"r1"}}"#,
                true,
            ),
            Ok(ResponsesEvent::Terminal { .. })
        ));
    }

    #[test]
    fn response_created_is_not_ignored_before_correlated_output() {
        let created = classify_responses_event(
            br#"{"type":"response.created","response":{"id":"r1","status":"in_progress"}}"#,
            true,
        )
        .expect("created event is valid");

        assert_ne!(created, ResponsesEvent::Ignored);
    }

    #[test]
    fn responses_delta_without_wire_identity_uses_correlated_created_response() {
        let request = PreparedWsOperation::new(
            [PreparedWsMessage::text(
                Bytes::from_static(
                    br#"{"type":"response.create","model":"model","input":"current"}"#,
                ),
                PreparedWsMessageRole::MeasuredInput,
            )],
            None,
        );
        let correlated =
            correlate_operation(&request, WebSocketDialect::Responses, "current-operation")
                .expect("Responses request accepts operation correlation");
        let mut state = TurnOperationState::new(correlated.correlation(), true);
        let created = classify_responses_event(
            br#"{"type":"response.created","response":{"id":"current","metadata":{"_aiperf_ws_operation":"current-operation"}}}"#,
            true,
        )
        .expect("current response.created parses");
        state
            .on_correlated_event(&created, 1)
            .expect("current marker arms response identity");

        let delta = classify_responses_event(
            br#"{"type":"response.output_text.delta","item_id":"message-1","output_index":0,"content_index":0,"delta":"hello","sequence_number":2}"#,
            true,
        )
        .expect("official Responses output delta parses without a response identity");
        assert_eq!(
            state
                .on_correlated_event(&delta, 2)
                .expect("sequential delta uses the correlated response"),
            EventDisposition::Attributed { is_terminal: false },
        );
        assert_eq!(
            state
                .content_for_observation(&delta, 2)
                .expect("delta is observable"),
            Some(Bytes::from_static(b"hello")),
        );
    }

    #[test]
    fn response_identity_mismatch_is_rejected_before_attribution() {
        let mut state = TurnOperationState::default();
        state
            .on_event(
                &ResponsesEvent::Created {
                    response_id: "expected".to_owned(),
                    operation_id: None,
                },
                1,
            )
            .expect("created identity binds");
        let error = state
            .on_event(
                &ResponsesEvent::Content {
                    response_id: Some("stale".to_owned()),
                    content: Bytes::from_static(b"wrong"),
                },
                2,
            )
            .expect_err("stale socket output must not be attributed");

        assert!(error.to_string().contains("identity mismatch"));
        assert!(
            state
                .content_for_observation(
                    &ResponsesEvent::Content {
                        response_id: Some("expected".to_owned()),
                        content: Bytes::from_static(b"right"),
                    },
                    3,
                )
                .unwrap()
                .is_some()
        );
    }

    #[test]
    fn reused_socket_ignores_stale_response_before_correlated_response() {
        let request = PreparedWsOperation::new(
            [PreparedWsMessage::text(
                Bytes::from_static(
                    br#"{"type":"response.create","model":"model","input":"current"}"#,
                ),
                PreparedWsMessageRole::MeasuredInput,
            )],
            None,
        );
        let correlated =
            correlate_operation(&request, WebSocketDialect::Responses, "current-operation")
                .expect("Responses request accepts operation correlation");
        let mut state = TurnOperationState::new(correlated.correlation(), true);
        for payload in [
            br#"{"type":"response.created","response":{"id":"stale","metadata":{"_aiperf_ws_operation":"stale-operation"}}}"#.as_slice(),
            br#"{"type":"response.output_text.delta","response_id":"stale","delta":"wrong"}"#.as_slice(),
            br#"{"type":"response.completed","response":{"id":"stale"}}"#.as_slice(),
        ] {
            let event = classify_responses_event(payload, true).expect("stale event parses");
            assert_eq!(
                state
                    .on_correlated_event(&event, 1)
                    .expect("stale event is quarantined"),
                EventDisposition::Unattributed,
            );
        }

        let created = classify_responses_event(
            br#"{"type":"response.created","response":{"id":"current","metadata":{"_aiperf_ws_operation":"current-operation"}}}"#,
            true,
        )
        .expect("current response.created parses");
        assert_eq!(
            state
                .on_correlated_event(&created, 2)
                .expect("current marker arms response identity"),
            EventDisposition::Attributed { is_terminal: false },
        );
        let content = classify_responses_event(
            br#"{"type":"response.output_text.delta","response_id":"current","delta":"right"}"#,
            true,
        )
        .expect("current output parses");
        assert_eq!(
            state
                .on_correlated_event(&content, 3)
                .expect("current output is attributed"),
            EventDisposition::Attributed { is_terminal: false },
        );
        assert_eq!(
            state
                .content_for_observation(&content, 3)
                .expect("current content is observable"),
            Some(Bytes::from_static(b"right")),
        );
    }

    #[test]
    fn correlation_is_injected_without_discarding_authored_metadata() {
        let responses = PreparedWsOperation::new(
            [PreparedWsMessage::text(
                Bytes::from_static(
                    br#"{"type":"response.create","model":"model","input":"hello","metadata":{"tenant":"alpha"}}"#,
                ),
                PreparedWsMessageRole::MeasuredInput,
            )],
            None,
        );
        let correlated =
            correlate_operation(&responses, WebSocketDialect::Responses, "operation-1")
                .expect("Responses metadata accepts the reserved marker");
        let payload: Value = serde_json::from_slice(correlated.operation().messages()[0].payload())
            .expect("correlated Responses event remains JSON");
        assert_eq!(payload["metadata"]["tenant"], "alpha");
        assert_eq!(payload["metadata"]["_aiperf_ws_operation"], "operation-1");

        let realtime = PreparedWsOperation::new(
            [
                PreparedWsMessage::text(
                    Bytes::from_static(
                        br#"{"type":"conversation.item.create","item":{"type":"message"}}"#,
                    ),
                    PreparedWsMessageRole::MeasuredInput,
                ),
                PreparedWsMessage::text(
                    Bytes::from_static(
                        br#"{"type":"response.create","response":{"metadata":{"tenant":"alpha"}}}"#,
                    ),
                    PreparedWsMessageRole::Control,
                ),
            ],
            None,
        );
        let correlated = correlate_operation(&realtime, WebSocketDialect::Realtime, "operation-2")
            .expect("Realtime events accept operation correlation");
        let input: Value = serde_json::from_slice(correlated.operation().messages()[0].payload())
            .expect("correlated Realtime input remains JSON");
        let create: Value = serde_json::from_slice(correlated.operation().messages()[1].payload())
            .expect("correlated Realtime create remains JSON");
        assert_eq!(input["event_id"], "operation-2:0");
        assert_eq!(create["event_id"], "operation-2:1");
        assert_eq!(create["response"]["metadata"]["tenant"], "alpha");
        assert_eq!(
            create["response"]["metadata"]["_aiperf_ws_operation"],
            "operation-2"
        );
    }

    #[test]
    fn full_public_metadata_capacity_is_preserved_without_an_internal_pair() {
        let authored = (0..16)
            .map(|index| {
                (
                    format!("key-{index}"),
                    Value::String(format!("value-{index}")),
                )
            })
            .collect::<serde_json::Map<_, _>>();
        let request = PreparedWsOperation::new(
            [PreparedWsMessage::text(
                Bytes::from(
                    serde_json::to_vec(&serde_json::json!({
                        "type": "response.create",
                        "model": "model",
                        "input": "hello",
                        "metadata": authored,
                    }))
                    .expect("request serializes"),
                ),
                PreparedWsMessageRole::MeasuredInput,
            )],
            None,
        );

        let correlated = correlate_operation(&request, WebSocketDialect::Responses, "operation-16")
            .expect("valid public metadata remains dispatchable");
        let payload: Value = serde_json::from_slice(correlated.operation().messages()[0].payload())
            .expect("correlated request remains JSON");
        let metadata = payload["metadata"]
            .as_object()
            .expect("metadata remains an object");
        assert_eq!(metadata.len(), 16);
        assert_eq!(metadata["key-0"], "value-0");
        assert_eq!(metadata["key-15"], "value-15");
        assert!(!metadata.contains_key(OPERATION_METADATA_KEY));
        assert!(!correlated.correlation().supports_reused_socket());
    }

    #[test]
    fn authored_metadata_is_never_repurposed_as_an_internal_marker() {
        let request = PreparedWsOperation::new(
            [PreparedWsMessage::text(
                Bytes::from_static(
                    br#"{"type":"response.create","metadata":{"_aiperf_ws_operation":"authored-value"}}"#,
                ),
                PreparedWsMessageRole::MeasuredInput,
            )],
            None,
        );

        let correlated =
            correlate_operation(&request, WebSocketDialect::Responses, "internal-operation")
                .expect("valid authored metadata remains valid");
        let message: Value = serde_json::from_slice(correlated.operation().messages()[0].payload())
            .expect("correlated request remains JSON");

        assert_eq!(
            message["metadata"][OPERATION_METADATA_KEY],
            "authored-value"
        );
        assert!(!correlated.correlation().supports_reused_socket());
    }

    #[test]
    fn authored_marker_like_metadata_completes_a_fresh_response_lifecycle() {
        let request = PreparedWsOperation::new(
            [PreparedWsMessage::text(
                Bytes::from_static(
                    br#"{"type":"response.create","metadata":{"_aiperf_ws_operation":"authored-value"}}"#,
                ),
                PreparedWsMessageRole::MeasuredInput,
            )],
            None,
        );
        let correlated =
            correlate_operation(&request, WebSocketDialect::Responses, "internal-operation")
                .expect("valid authored metadata remains valid");
        let mut state = TurnOperationState::new(correlated.correlation(), false);

        for (timestamp_ns, payload, expected) in [
            (
                1,
                br#"{"type":"response.created","response":{"id":"current","metadata":{"_aiperf_ws_operation":"authored-value"}}}"#.as_slice(),
                EventDisposition::Attributed { is_terminal: false },
            ),
            (
                2,
                br#"{"type":"response.output_text.delta","item_id":"message-1","output_index":0,"content_index":0,"delta":"hello","sequence_number":2}"#.as_slice(),
                EventDisposition::Attributed { is_terminal: false },
            ),
            (
                3,
                br#"{"type":"response.completed","response":{"id":"current","status":"completed","metadata":{"_aiperf_ws_operation":"authored-value"}}}"#.as_slice(),
                EventDisposition::Attributed { is_terminal: true },
            ),
        ] {
            let event = classify_responses_event(payload, true)
                .expect("official Responses lifecycle event parses");
            assert_eq!(
                state
                    .on_correlated_event(&event, timestamp_ns)
                    .expect("fresh response lifecycle remains attributable"),
                expected,
            );
        }

        assert!(!state.has_verified_correlation());
    }

    #[test]
    fn uncorrelated_error_is_never_attributed_to_reused_operation() {
        let request = PreparedWsOperation::new(
            [PreparedWsMessage::text(
                Bytes::from_static(br#"{"type":"response.create","response":{}}"#),
                PreparedWsMessageRole::Control,
            )],
            None,
        );
        let correlated = correlate_operation(&request, WebSocketDialect::Realtime, "operation-3")
            .expect("Realtime request accepts correlation");
        let mut state = TurnOperationState::new(correlated.correlation(), true);
        let stale = classify_realtime_event(
            br#"{"type":"error","event_id":"server-event-old","error":{"message":"stale","event_id":"older-operation:0"}}"#,
            true,
        )
        .expect("Realtime error envelope parses");
        assert_eq!(
            state
                .on_correlated_event(&stale, 1)
                .expect("stale error is quarantined"),
            EventDisposition::Unattributed,
        );
        let created = classify_realtime_event(
            br#"{"type":"response.created","response":{"id":"current","metadata":{"_aiperf_ws_operation":"operation-3"}}}"#,
            true,
        )
        .expect("correlated response.created parses");
        assert_eq!(
            state
                .on_correlated_event(&created, 2)
                .expect("correlated response is armed"),
            EventDisposition::Attributed { is_terminal: false },
        );
        assert_eq!(
            state
                .on_correlated_event(&stale, 3)
                .expect("explicitly stale error remains quarantined after creation"),
            EventDisposition::Unattributed,
        );
        let current = classify_realtime_event(
            br#"{"type":"error","event_id":"server-event-current","error":{"message":"current","event_id":"operation-3:0"}}"#,
            true,
        )
        .expect("correlated Realtime error envelope parses");
        assert_eq!(
            state
                .on_correlated_event(&current, 4)
                .expect("current error is attributed"),
            EventDisposition::AttributedError,
        );
    }

    #[test]
    fn realtime_error_uses_nested_client_event_identity() {
        let request = PreparedWsOperation::new(
            [PreparedWsMessage::text(
                Bytes::from_static(br#"{"type":"response.create","response":{}}"#),
                PreparedWsMessageRole::Control,
            )],
            None,
        );
        let correlated = correlate_operation(&request, WebSocketDialect::Realtime, "operation-4")
            .expect("Realtime request accepts correlation");
        let mut state = TurnOperationState::new(correlated.correlation(), true);
        let error = classify_realtime_event(
            br#"{"type":"error","event_id":"server-event-1","error":{"type":"invalid_request_error","code":"invalid_value","message":"bad request","param":null,"event_id":"operation-4:0"}}"#,
            true,
        )
        .expect("official Realtime error envelope parses");

        assert_eq!(
            state
                .on_correlated_event(&error, 1)
                .expect("nested client identity correlates the error"),
            EventDisposition::AttributedError,
        );
    }

    #[test]
    fn markerless_error_after_created_is_unsafe_on_a_reused_socket() {
        let request = PreparedWsOperation::new(
            [PreparedWsMessage::text(
                Bytes::from_static(
                    br#"{"type":"response.create","model":"model","input":"current"}"#,
                ),
                PreparedWsMessageRole::MeasuredInput,
            )],
            None,
        );
        let correlated =
            correlate_operation(&request, WebSocketDialect::Responses, "current-operation")
                .expect("Responses request accepts operation correlation");
        let mut state = TurnOperationState::new(correlated.correlation(), true);
        let created = classify_responses_event(
            br#"{"type":"response.created","response":{"id":"current","metadata":{"_aiperf_ws_operation":"current-operation"}}}"#,
            true,
        )
        .expect("current response.created parses");
        state
            .on_correlated_event(&created, 1)
            .expect("current marker arms response identity");
        let stale = classify_responses_event(
            br#"{"type":"error","code":"old_failure","message":"delayed stale error","param":null,"sequence_number":3}"#,
            true,
        )
        .expect("official Responses error parses");

        assert_eq!(
            state
                .on_correlated_event(&stale, 2)
                .expect("ambiguous error is handled fail-safe"),
            EventDisposition::UnsafeUnattributedError,
        );
    }

    #[test]
    fn responses_error_uses_top_level_diagnostic() {
        let error = classify_responses_event(
            br#"{"type":"error","code":"rate_limit_exceeded","message":"rate limited","param":null,"sequence_number":7}"#,
            true,
        )
        .expect("official Responses error envelope parses");

        assert_eq!(
            error.error_message(),
            Some("rate limited (code: rate_limit_exceeded)".to_owned())
        );
    }

    #[test]
    fn retry_removes_stale_continuation_identity() {
        let request = PreparedWsOperation::new(
            [PreparedWsMessage::text(
                Bytes::from_static(
                    br#"{"type":"response.create","input":"history","previous_response_id":"stale"}"#,
                ),
                PreparedWsMessageRole::MeasuredInput,
            )],
            None,
        );
        let replay = full_history_retry(&request).expect("full history can be replayed");
        let event: Value = serde_json::from_slice(replay.messages()[0].payload())
            .expect("replayed event remains JSON");
        assert!(event.get("previous_response_id").is_none());
    }

    #[test]
    fn continuation_injection_replaces_stale_identity() {
        let request = PreparedWsOperation::new(
            [PreparedWsMessage::text(
                Bytes::from_static(
                    br#"{"type":"response.create","input":[{"role":"assistant","content":"old"},{"role":"user","content":"new"}],"previous_response_id":"stale"}"#,
                ),
                PreparedWsMessageRole::MeasuredInput,
            )],
            None,
        );
        let continued = with_previous_response_id(&request, "response-2")
            .expect("Responses request can carry continuation identity");
        let event: Value = serde_json::from_slice(continued.messages()[0].payload())
            .expect("continued event remains JSON");
        assert_eq!(event["previous_response_id"], "response-2");
        assert_eq!(
            event["input"],
            serde_json::json!([{"role":"user","content":"new"}])
        );
    }

    #[test]
    fn completed_terminal_carries_content_before_completion() {
        let event = classify_responses_event(
            br#"{"type":"response.completed","response":{"id":"r1","output":[{"content":[{"type":"output_text","text":"hello"}]}]}}"#,
            true,
        )
        .expect("terminal event is valid");
        assert!(matches!(
            event,
            ResponsesEvent::Terminal {
                content,
                status: ReplayTerminalStatus::Completed,
                ..
            } if content == Bytes::from_static(b"hello")
        ));
    }

    #[test]
    fn streamed_text_is_not_repeated_by_terminal_snapshot() {
        let mut state = TurnOperationState::default();
        let delta = ResponsesEvent::Content {
            response_id: Some("r1".to_owned()),
            content: Bytes::from_static(b"hello "),
        };
        let terminal = ResponsesEvent::Terminal {
            response_id: "r1".to_owned(),
            usage: ObservedUsage::default(),
            content: Bytes::from_static(b"hello world"),
            status: ReplayTerminalStatus::Completed,
        };
        assert_eq!(
            state.content_for_observation(&delta, 10).unwrap(),
            Some(Bytes::from_static(b"hello "))
        );
        assert_eq!(
            state.content_for_observation(&terminal, 20).unwrap(),
            Some(Bytes::from_static(b"world"))
        );
    }

    #[test]
    fn terminal_snapshot_must_extend_the_exact_streamed_prefix() {
        let mut state = TurnOperationState::default();
        let delta = ResponsesEvent::Content {
            response_id: Some("r1".to_owned()),
            content: Bytes::from_static(b"hello"),
        };
        let terminal = ResponsesEvent::Terminal {
            response_id: "r1".to_owned(),
            usage: ObservedUsage::default(),
            content: Bytes::from_static(b"jello world"),
            status: ReplayTerminalStatus::Completed,
        };
        assert_eq!(
            state.content_for_observation(&delta, 10).unwrap(),
            Some(Bytes::from_static(b"hello"))
        );
        let error = state
            .content_for_observation(&terminal, 20)
            .expect_err("a divergent terminal snapshot is a protocol failure");
        assert!(error.to_string().contains("streamed prefix"));
    }

    #[test]
    fn continuation_sends_only_items_after_the_last_assistant_turn() {
        let request = PreparedWsOperation::new(
            [PreparedWsMessage::text(
                Bytes::from_static(
                    br#"{"type":"response.create","input":[{"role":"user","content":"old"},{"role":"assistant","content":"answer"},{"role":"user","content":"new"}]}"#,
                ),
                PreparedWsMessageRole::MeasuredInput,
            )],
            None,
        );
        let continued = with_previous_response_id(&request, "response-2")
            .expect("Responses request can carry continuation identity");
        let event: Value = serde_json::from_slice(continued.messages()[0].payload())
            .expect("continued event remains JSON");
        assert_eq!(event["previous_response_id"], "response-2");
        assert_eq!(
            event["input"],
            serde_json::json!([{"role":"user","content":"new"}])
        );
    }

    #[test]
    fn continuation_never_pairs_an_identity_with_full_history() {
        let request = PreparedWsOperation::new(
            [PreparedWsMessage::text(
                Bytes::from_static(
                    br#"{"type":"response.create","input":[{"role":"user","content":"history"}]}"#,
                ),
                PreparedWsMessageRole::MeasuredInput,
            )],
            None,
        );
        assert!(with_previous_response_id(&request, "response-2").is_none());
    }

    #[test]
    fn incomplete_terminal_is_failed_and_never_retryable() {
        let event = classify_responses_event(
            br#"{"type":"response.incomplete","response":{"id":"r1"}}"#,
            true,
        )
        .expect("incomplete event is valid");
        assert!(matches!(
            event,
            ResponsesEvent::Terminal {
                status: ReplayTerminalStatus::Failed,
                ..
            }
        ));
        let mut state = TurnOperationState::default();
        state
            .on_event(
                &ResponsesEvent::Created {
                    response_id: "r1".to_owned(),
                    operation_id: None,
                },
                9,
            )
            .unwrap();
        assert!(state.on_event(&event, 10).unwrap());
        assert!(!state.can_retry());
        assert_eq!(
            state.finish(ReplayTerminalStatus::Failed),
            Default::default()
        );
    }

    #[test]
    fn reasoning_or_usage_disables_automatic_replay() {
        for event in [
            ResponsesEvent::Reasoning {
                response_id: Some("r1".to_owned()),
            },
            ResponsesEvent::Usage {
                response_id: "r1".to_owned(),
                usage: ObservedUsage::default(),
            },
        ] {
            let mut state = TurnOperationState::default();
            state
                .on_event(
                    &ResponsesEvent::Created {
                        response_id: "r1".to_owned(),
                        operation_id: None,
                    },
                    9,
                )
                .unwrap();
            state.on_event(&event, 10).unwrap();
            assert!(!state.can_retry());
        }
    }

    #[test]
    fn realtime_text_and_terminal_events_are_distinct() {
        assert_eq!(
            classify_realtime_event(
                br#"{"type":"response.output_text.delta","response_id":"r1","delta":"hello"}"#,
                true,
            )
            .expect("Realtime text event is valid"),
            ResponsesEvent::Content {
                response_id: Some("r1".to_owned()),
                content: Bytes::from_static(b"hello"),
            }
        );
        assert!(matches!(
            classify_realtime_event(
                br#"{"type":"response.done","response":{"id":"r1","status":"completed"}}"#,
                true,
            ),
            Ok(ResponsesEvent::Terminal {
                status: ReplayTerminalStatus::Completed,
                ..
            })
        ));
    }

    #[test]
    fn realtime_audio_is_an_observer_fact_without_a_text_token() {
        assert_eq!(
            classify_realtime_event(
                br#"{"type":"response.output_audio.delta","response_id":"r1","delta":"AAE="}"#,
                true,
            )
            .expect("Realtime audio event is valid"),
            ResponsesEvent::Audio {
                response_id: "r1".to_owned(),
            }
        );
    }

    #[test]
    fn realtime_done_maps_non_completed_response_statuses_to_failure() {
        for status in ["failed", "incomplete", "cancelled"] {
            let payload = format!(
                r#"{{"type":"response.done","response":{{"id":"r1","status":"{status}"}}}}"#
            );
            let event = classify_realtime_event(payload.as_bytes(), true)
                .expect("Realtime terminal envelope is valid");
            let expected = if status == "cancelled" {
                ReplayTerminalStatus::Canceled
            } else {
                ReplayTerminalStatus::Failed
            };
            assert!(matches!(
                event,
                ResponsesEvent::Terminal { status, .. } if status == expected
            ));
        }
    }

    #[test]
    fn realtime_done_rejects_missing_or_unknown_response_status() {
        for payload in [
            br#"{"type":"response.done","response":{"id":"r1"}}"#.as_slice(),
            br#"{"type":"response.done","response":{"id":"r1","status":"mystery"}}"#.as_slice(),
        ] {
            assert!(classify_realtime_event(payload, true).is_err());
        }
    }
}
