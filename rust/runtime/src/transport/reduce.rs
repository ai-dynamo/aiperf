// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Transport-neutral reduction of decoded endpoint responses into observer
//! facts and aggregated model/usage/metrics state.
//!
//! Transports decode wire records into [`ServerResponse`] items. This module
//! parses those items, reconciles usage, accumulates response data and endpoint
//! metrics, emits token observations, and reconstructs assistant turns.

use std::borrow::Cow;
use std::cell::Cell;

use serde_json::Value;
use smallvec::SmallVec;
use uuid::Uuid;

use crate::dispatch::sink::{
    ObservedEndpointMetrics, ObservedTokenKind, ObservedUsage, RequestObserver,
};
use crate::endpoints::{ParsedResponse, ResponseData, Turn, UsageView};
use crate::scheduled::ModelResponseMetadata;
use crate::transport::core::{BoundedDecisionAdmission, DecisionAdmissionError};

/// Mutable state accumulated across parsed responses.
pub(crate) struct EndpointReduceAccumulators<'a> {
    /// Concatenated user-visible text across responses.
    pub response_text: &'a mut String,
    /// The rebuilt model response (content/reasoning/token-ids/errors).
    pub model_response: &'a mut ModelResponseMetadata,
    /// Endpoint-specific auxiliary metrics (video timing/memory).
    pub endpoint_metrics: &'a mut ObservedEndpointMetrics,
    /// Reconciled terminal usage counts.
    pub observed_usage: &'a mut ObservedUsage,
}

/// Token-emission context supplied by a transport.
pub(crate) struct TokenEmitter<'a> {
    /// Request correlation id.
    pub uuid: Uuid,
    /// Whether the endpoint produces streamable tokens.
    pub produces_tokens: bool,
    /// Run origin, for the first-token ns delta.
    pub start_ns: i64,
    /// Measurement observer.
    pub obs: &'a dyn RequestObserver,
    /// Map an absolute perf-ns instant to run-relative ms.
    pub to_ms: &'a dyn Fn(i64) -> f64,
    /// Shared once-only first-token latch.
    pub first_token_released: &'a Cell<bool>,
    /// First-token callback taking a run-relative ns delta.
    pub on_first_token: &'a dyn Fn(i64),
}

/// Fold a parsed response into `acc`, returning whether it carried content.
///
/// Usage is reconciled even when the response has no content payload.
pub(crate) fn reduce_parsed_response(
    parsed: &ParsedResponse,
    emit: &TokenEmitter<'_>,
    acc: EndpointReduceAccumulators<'_>,
) -> bool {
    absorb_usage(parsed, acc.observed_usage);
    let Some(data) = parsed.data.as_ref() else {
        return false;
    };
    absorb_endpoint_metrics(data, acc.endpoint_metrics);
    let text = absorb_response_data(data, acc.model_response);
    acc.response_text.push_str(&text);
    if emit.produces_tokens {
        let at_ns = i64::try_from(parsed.perf_ns).unwrap_or(i64::MAX);
        if let ResponseData::TokenIds { token_ids } = data
            && !token_ids.is_empty()
        {
            if !emit.first_token_released.replace(true) {
                (emit.on_first_token)(at_ns.saturating_sub(emit.start_ns));
            }
            // All token ids in one chunk share the same arrival instant; keep a
            // stack-inline buffer so the common single-token streaming chunk
            // avoids a heap allocation. Observed metrics are identical.
            let at_ms = (emit.to_ms)(at_ns);
            let timestamps: SmallVec<[f64; 8]> = smallvec::smallvec![at_ms; token_ids.len()];
            emit.obs.on_output_tokens(emit.uuid, &timestamps);
        } else if !text.is_empty() {
            if !emit.first_token_released.replace(true) {
                (emit.on_first_token)(at_ns.saturating_sub(emit.start_ns));
            }
            emit.obs
                .on_classified_token(emit.uuid, (emit.to_ms)(at_ns), token_kind(data));
        }
    }
    true
}

/// Admit the textual decision facts from one decoded response without creating
/// terminal response metadata or a concatenated response string.
///
/// The bounded decision transport calls this while it still owns the live
/// response frame. Splitting combined response variants avoids constructing an
/// intermediate combined string before the selected cap is enforced.
pub(crate) fn admit_parsed_decision(
    parsed: &ParsedResponse,
    admission: &mut BoundedDecisionAdmission,
) -> Result<bool, DecisionAdmissionError> {
    let Some(data) = parsed.data.as_ref() else {
        return Ok(false);
    };
    match data {
        ResponseData::Text { text } => admission.push(text.as_bytes())?,
        ResponseData::Reasoning { content, reasoning } => {
            admission.push(reasoning.as_bytes())?;
            if let Some(content) = content {
                admission.push(content.as_bytes())?;
            }
        }
        ResponseData::ToolCall {
            tool_call_text,
            content,
        } => {
            if let Some(content) = content {
                admission.push(content.as_bytes())?;
            }
            admission.push(tool_call_text.as_bytes())?;
        }
        ResponseData::TokenIds { .. }
        | ResponseData::Embeddings { .. }
        | ResponseData::Rankings { .. }
        | ResponseData::ImageRetrieval { .. }
        | ResponseData::Images(_)
        | ResponseData::Audio(_)
        | ResponseData::Video(_) => return Ok(false),
    }
    Ok(true)
}

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
pub(crate) fn absorb_response_data<'a>(
    data: &'a ResponseData,
    metadata: &mut ModelResponseMetadata,
) -> Cow<'a, str> {
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
    data.get_text_cow()
}

fn append_text(target: &mut Option<String>, text: &str) {
    target.get_or_insert_with(String::new).push_str(text);
}

/// Absorb video timing and memory metrics.
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

/// Return the raw assistant message, or synthesize one from the rebuilt turn.
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
