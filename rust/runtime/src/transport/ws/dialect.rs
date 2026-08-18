// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Responses application-event classification for WebSocket execution.

use bytes::Bytes;
use serde_json::Value;

use crate::body_plan::{
    PreparedWsMessage, PreparedWsMessageRole, PreparedWsOpcode, PreparedWsOperation,
};
use crate::dispatch::sink::ObservedUsage;
use crate::endpoints::UsageView;
use crate::transport::ws::RoundTripTimingState;

/// One complete Responses application event.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum ResponsesEvent {
    /// Non-empty user-visible content.
    Content(Bytes),
    /// Non-visible reasoning delta.
    Reasoning,
    /// Endpoint usage envelope.
    Usage(ObservedUsage),
    /// A continuation identity was rejected before visible output.
    RetriableContinuationRejection,
    /// Logical operation completion.
    Terminal {
        response_id: Option<String>,
        usage: ObservedUsage,
    },
    /// A control or irrelevant application envelope.
    Ignored,
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
        "response.output_text.delta" => Ok(event
            .get("delta")
            .and_then(Value::as_str)
            .filter(|delta| !delta.is_empty())
            .map_or(ResponsesEvent::Ignored, |delta| {
                ResponsesEvent::Content(Bytes::copy_from_slice(delta.as_bytes()))
            })),
        "response.reasoning.delta" => Ok(ResponsesEvent::Reasoning),
        "response.completed" => Ok(ResponsesEvent::Terminal {
            response_id: event
                .get("response")
                .and_then(|response| response.get("id"))
                .and_then(Value::as_str)
                .map(str::to_owned),
            usage: observed_usage(event.get("usage").or_else(|| {
                event
                    .get("response")
                    .and_then(|response| response.get("usage"))
            })),
        }),
        "response.failed" | "error" => anyhow::bail!("Responses WebSocket operation failed"),
        "response.incomplete" => Ok(ResponsesEvent::Terminal {
            response_id: None,
            usage: ObservedUsage::default(),
        }),
        "response.usage" => Ok(ResponsesEvent::Usage(observed_usage(
            event.get("usage").or_else(|| {
                event
                    .get("response")
                    .and_then(|response| response.get("usage"))
            }),
        ))),
        "response.continuation_rejected" => Ok(ResponsesEvent::RetriableContinuationRejection),
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
    Some(PreparedWsOperation::new(
        messages,
        request.http_sse_fallback_body().cloned(),
    ))
}

/// Scalar lifecycle state for one turn-serialized operation.
#[derive(Debug, Default)]
pub(crate) struct TurnOperationState {
    timing: RoundTripTimingState,
    has_visible_output: bool,
    has_terminal: bool,
}

impl TurnOperationState {
    pub(crate) fn on_send(&mut self, timestamp_ns: i64) {
        self.timing.on_measured_input_flushed(timestamp_ns);
    }

    pub(crate) fn on_event(&mut self, event: &ResponsesEvent, timestamp_ns: i64) -> bool {
        if let ResponsesEvent::Content(content) = event
            && !content.is_empty()
        {
            self.has_visible_output = true;
            self.timing.on_content_received(timestamp_ns);
        }
        if matches!(event, ResponsesEvent::Terminal { .. }) {
            self.has_terminal = true;
        }
        self.has_terminal
    }

    pub(crate) const fn can_retry(&self) -> bool {
        !self.has_visible_output && !self.has_terminal
    }

    pub(crate) fn finish(&self) -> crate::dispatch::sink::ObservedRoundTripMetrics {
        if self.has_terminal {
            self.timing.finish()
        } else {
            crate::dispatch::sink::ObservedRoundTripMetrics::default()
        }
    }
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
                br#"{"type":"response.output_text.delta","delta":"hello"}"#,
                true,
            )
            .expect("content event is valid"),
            ResponsesEvent::Content(Bytes::from_static(b"hello"))
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
}
