// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Endpoint-owned application-event classification for WebSocket execution.

use bytes::Bytes;
use serde_json::Value;

use crate::body_plan::{
    PreparedWsMessage, PreparedWsMessageRole, PreparedWsOpcode, PreparedWsOperation,
};
use crate::dispatch::collector::ReplayTerminalStatus;
use crate::dispatch::sink::ObservedUsage;
use crate::endpoints::UsageView;
use crate::transport::ws::RoundTripTimingState;

/// One complete Responses application event.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum ResponsesEvent {
    /// The server-assigned identity for this logical response.
    Created { response_id: String },
    /// Non-empty user-visible content.
    Content { response_id: String, content: Bytes },
    /// Non-visible reasoning delta.
    Reasoning { response_id: String },
    /// Binary audio carried inside a Realtime JSON event.
    Audio { response_id: String },
    /// Endpoint usage envelope.
    Usage {
        response_id: String,
        usage: ObservedUsage,
    },
    /// A continuation identity was rejected before visible output.
    RetriableContinuationRejection,
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
        }),
        "response.output_text.delta" => {
            let response_id = response_id(&event, "response.output_text.delta")?;
            Ok(event
                .get("delta")
                .and_then(Value::as_str)
                .filter(|delta| !delta.is_empty())
                .map_or(ResponsesEvent::Ignored, |delta| ResponsesEvent::Content {
                    response_id,
                    content: Bytes::copy_from_slice(delta.as_bytes()),
                }))
        }
        "response.reasoning.delta" => Ok(ResponsesEvent::Reasoning {
            response_id: response_id(&event, "response.reasoning.delta")?,
        }),
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
        "error" => anyhow::bail!("Responses WebSocket operation failed"),
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
        }),
        "response.output_text.delta" => {
            let response_id = response_id(&event, "response.output_text.delta")?;
            Ok(event
                .get("delta")
                .and_then(Value::as_str)
                .filter(|delta| !delta.is_empty())
                .map_or(ResponsesEvent::Ignored, |delta| ResponsesEvent::Content {
                    response_id,
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
        "error" => anyhow::bail!("Realtime WebSocket operation failed"),
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
}

impl TurnOperationState {
    pub(crate) fn on_send(&mut self, timestamp_ns: i64) {
        self.timing.on_measured_input_flushed(timestamp_ns);
    }

    pub(crate) fn on_event(
        &mut self,
        event: &ResponsesEvent,
        _timestamp_ns: i64,
    ) -> anyhow::Result<bool> {
        match event {
            ResponsesEvent::Created { response_id } => match self.response_id.as_deref() {
                None => self.response_id = Some(response_id.clone()),
                Some(bound) if bound == response_id => {
                    anyhow::bail!("websocket response identity {response_id:?} was created twice")
                }
                Some(bound) => anyhow::bail!(
                    "websocket response identity changed from {bound:?} to {response_id:?}"
                ),
            },
            ResponsesEvent::Content { response_id, .. }
            | ResponsesEvent::Reasoning { response_id }
            | ResponsesEvent::Audio { response_id }
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
            ResponsesEvent::RetriableContinuationRejection | ResponsesEvent::Ignored => {}
        }
        if matches!(
            event,
            ResponsesEvent::Created { .. }
                | ResponsesEvent::Content { .. }
                | ResponsesEvent::Reasoning { .. }
                | ResponsesEvent::Audio { .. }
                | ResponsesEvent::Usage { .. }
                | ResponsesEvent::Terminal { .. }
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
                response_id: "r1".to_owned(),
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
    fn output_delta_without_response_identity_is_rejected() {
        let error = classify_responses_event(
            br#"{"type":"response.output_text.delta","delta":"hello"}"#,
            true,
        )
        .expect_err("uncorrelated output must not be attributed by socket order");

        assert!(error.to_string().contains("response identity"));
    }

    #[test]
    fn response_identity_mismatch_is_rejected_before_attribution() {
        let mut state = TurnOperationState::default();
        state
            .on_event(
                &ResponsesEvent::Created {
                    response_id: "expected".to_owned(),
                },
                1,
            )
            .expect("created identity binds");
        let error = state
            .on_event(
                &ResponsesEvent::Content {
                    response_id: "stale".to_owned(),
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
                        response_id: "expected".to_owned(),
                        content: Bytes::from_static(b"right"),
                    },
                    3,
                )
                .unwrap()
                .is_some()
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
            response_id: "r1".to_owned(),
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
            response_id: "r1".to_owned(),
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
                response_id: "r1".to_owned(),
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
                response_id: "r1".to_owned(),
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
