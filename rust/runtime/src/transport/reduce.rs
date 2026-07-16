// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Transport-neutral reduction of decoded endpoint responses into observer
//! facts and aggregated model/usage/metrics state.
//!
//! Every native transport (`http`, `grpc`, and any future wire) decodes its
//! own record into a sequence of [`ServerResponse`] items and then reduces that
//! sequence identically: parse each response, absorb usage/data/endpoint
//! metrics, emit token callbacks, and reconstruct the assistant turn. That
//! reduction is the same regardless of how the bytes arrived, so it lives here
//! once instead of being copied per transport. A transport contributes only the
//! wire-decode that produces the [`ServerResponse`] iterator and the mapping
//! from its own error enum to a [`ReplayTerminalStatus`].

use serde_json::Value;

use crate::endpoints::{ParsedResponse, ResponseData, Turn, UsageView};
use crate::scheduled::ModelResponseMetadata;
use loadgen_core::sink::{ObservedEndpointMetrics, ObservedTokenKind, ObservedUsage};

/// Classify a response chunk as reasoning or output for token emission.
pub(crate) fn token_kind(data: &ResponseData) -> ObservedTokenKind {
    match data {
        ResponseData::Reasoning { reasoning, .. } if !reasoning.is_empty() => {
            ObservedTokenKind::Reasoning
        }
        _ => ObservedTokenKind::Output,
    }
}

/// Fold one decoded response's textual/token payload into the aggregated model
/// response, returning the plain text it contributed.
pub(crate) fn absorb_response_data(
    data: &ResponseData,
    metadata: &mut ModelResponseMetadata,
) -> String {
    match data {
        ResponseData::Text { text } => append_text(&mut metadata.content, text),
        ResponseData::Reasoning { content, reasoning } => {
            metadata.content.get_or_insert_with(String::new);
            append_text(&mut metadata.reasoning, reasoning);
            if let Some(content) = content {
                append_text(&mut metadata.content, content);
            }
        }
        ResponseData::ToolCall {
            tool_call_text,
            content,
        } => {
            if let Some(content) = content {
                append_text(&mut metadata.content, content);
            }
            append_text(&mut metadata.content, tool_call_text);
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

fn append_text(target: &mut Option<String>, text: &str) {
    target.get_or_insert_with(String::new).push_str(text);
}

/// Absorb per-endpoint auxiliary metrics (currently video timing/memory).
pub(crate) fn absorb_endpoint_metrics(data: &ResponseData, metrics: &mut ObservedEndpointMetrics) {
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

/// Reconcile the extended usage fields from one parsed response into the
/// terminal [`ObservedUsage`], preferring the latest reported value.
pub(crate) fn absorb_usage(parsed: &ParsedResponse, observed: &mut ObservedUsage) {
    let Some(usage) = parsed.usage.as_ref().and_then(UsageView::from_value) else {
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

/// Reconstruct the assistant message JSON from a rebuilt turn: prefer the raw
/// wire message, else synthesize `{role, content}` from the turn's texts.
pub(crate) fn assistant_message(turn: &Turn) -> Option<Value> {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn usage_absorption_retains_extended_endpoint_facts() {
        let parsed = ParsedResponse {
            perf_ns: 1,
            data: None,
            usage: Some(serde_json::json!({
                "prompt_tokens_details": {"audio_tokens": 2},
                "completion_tokens_details": {
                    "audio_tokens": 3,
                    "accepted_prediction_tokens": 4,
                    "rejected_prediction_tokens": 5
                },
                "toolUsePromptTokenCount": 6,
                "prompt_audio_seconds": 1.5
            })),
            sources: None,
        };
        let mut observed = ObservedUsage::default();
        absorb_usage(&parsed, &mut observed);

        assert_eq!(observed.prompt_audio_tokens, Some(2));
        assert_eq!(observed.completion_audio_tokens, Some(3));
        assert_eq!(observed.accepted_prediction_tokens, Some(4));
        assert_eq!(observed.rejected_prediction_tokens, Some(5));
        assert_eq!(observed.tool_use_prompt_tokens, Some(6));
        assert_eq!(observed.prompt_audio_seconds, Some(1.5));
    }
}
