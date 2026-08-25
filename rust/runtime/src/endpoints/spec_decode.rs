// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Canonical normalization of per-request speculative-decoding statistics.

use std::collections::BTreeMap;
use std::fmt::{Display, Formatter, Result as FmtResult};

use serde::Deserialize;
use serde_json::Value;

use crate::dispatch::sink::ObservedSpecDecodeAcceptance;

/// A malformed vLLM speculative-decoding payload.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum SpecDecodePayloadError {
    /// Required fields do not deserialize to the documented wire types.
    InvalidShape(String),
    /// A floating metric is non-finite.
    NonFiniteValue(&'static str),
    /// A histogram key is not a non-negative decimal integer.
    InvalidHistogramBucket(String),
    /// Two wire keys normalize to the same integer bucket.
    DuplicateHistogramBucket(u64),
    /// Aggregate or per-step counts contradict each other.
    InconsistentCounts(&'static str),
    /// Count arithmetic overflowed while validating the payload.
    CountOverflow,
}

impl Display for SpecDecodePayloadError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> FmtResult {
        match self {
            Self::InvalidShape(error) => write!(formatter, "invalid payload shape: {error}"),
            Self::NonFiniteValue(field) => write!(formatter, "{field} must be finite"),
            Self::InvalidHistogramBucket(bucket) => {
                write!(formatter, "invalid acceptance histogram bucket {bucket:?}")
            }
            Self::DuplicateHistogramBucket(bucket) => {
                write!(
                    formatter,
                    "duplicate normalized acceptance histogram bucket {bucket}"
                )
            }
            Self::InconsistentCounts(message) => formatter.write_str(message),
            Self::CountOverflow => formatter.write_str("speculative-decoding count overflow"),
        }
    }
}

impl std::error::Error for SpecDecodePayloadError {}

#[derive(Debug, Deserialize)]
struct VllmSpecDecodeStats {
    mean_acceptance_length: f64,
    draft_acceptance_rate: f64,
    acceptance_histogram: BTreeMap<String, u64>,
    num_accepted_draft_tokens: u64,
    num_draft_tokens: u64,
    num_spec_steps: u64,
    #[serde(default)]
    num_spec_tokens: Option<u64>,
    #[serde(default)]
    per_step_accepted: Option<Vec<u64>>,
    #[serde(default)]
    per_step_drafted: Option<Vec<u64>>,
}

/// Extract vLLM's per-choice stats only when the response has one sequence.
pub(crate) fn extract_vllm_spec_decode_stats(response: &Value) -> Option<&Value> {
    let choices = response.get("choices")?.as_array()?;
    if choices.len() != 1 {
        return None;
    }
    choices
        .first()?
        .as_object()?
        .get("speculative_decoding_stats")
}

/// Normalize one vLLM stats object into the engine-neutral observer record.
pub(crate) fn parse_vllm_spec_decode_stats(
    payload: &Value,
    completion_tokens: Option<u64>,
) -> Result<ObservedSpecDecodeAcceptance, SpecDecodePayloadError> {
    let wire: VllmSpecDecodeStats = serde_json::from_value(payload.clone())
        .map_err(|error| SpecDecodePayloadError::InvalidShape(error.to_string()))?;
    if !wire.mean_acceptance_length.is_finite() {
        return Err(SpecDecodePayloadError::NonFiniteValue(
            "mean_acceptance_length",
        ));
    }
    if !wire.draft_acceptance_rate.is_finite() {
        return Err(SpecDecodePayloadError::NonFiniteValue(
            "draft_acceptance_rate",
        ));
    }
    validate_per_step(&wire)?;

    let mut histogram = BTreeMap::new();
    for (bucket, count) in wire.acceptance_histogram {
        let normalized = bucket
            .parse::<u64>()
            .map_err(|_| SpecDecodePayloadError::InvalidHistogramBucket(bucket))?;
        if histogram.insert(normalized, count).is_some() {
            return Err(SpecDecodePayloadError::DuplicateHistogramBucket(normalized));
        }
    }
    let steps = histogram
        .values()
        .try_fold(0_u64, |sum, count| sum.checked_add(*count))
        .ok_or(SpecDecodePayloadError::CountOverflow)?;
    if steps != wire.num_spec_steps {
        return Err(SpecDecodePayloadError::InconsistentCounts(
            "acceptance histogram counts do not equal num_spec_steps",
        ));
    }
    let accepted = histogram
        .iter()
        .try_fold(0_u64, |sum, (bucket, count)| {
            bucket
                .checked_mul(*count)
                .and_then(|value| sum.checked_add(value))
        })
        .ok_or(SpecDecodePayloadError::CountOverflow)?;
    if accepted != wire.num_accepted_draft_tokens {
        return Err(SpecDecodePayloadError::InconsistentCounts(
            "acceptance histogram weighted sum does not equal num_accepted_draft_tokens",
        ));
    }
    if wire.num_accepted_draft_tokens > wire.num_draft_tokens {
        return Err(SpecDecodePayloadError::InconsistentCounts(
            "accepted draft tokens exceed proposed draft tokens",
        ));
    }
    Ok(ObservedSpecDecodeAcceptance {
        engine: "vllm".to_string(),
        mean_acceptance_length: wire.mean_acceptance_length,
        draft_acceptance_rate: wire.draft_acceptance_rate,
        acceptance_histogram: histogram,
        num_accepted_draft_tokens: wire.num_accepted_draft_tokens,
        num_draft_tokens: wire.num_draft_tokens,
        num_spec_steps: wire.num_spec_steps,
        num_spec_tokens: wire.num_spec_tokens,
        completion_tokens,
        per_step_accepted: wire.per_step_accepted,
        per_step_drafted: wire.per_step_drafted,
    })
}

fn validate_per_step(wire: &VllmSpecDecodeStats) -> Result<(), SpecDecodePayloadError> {
    let expected_len =
        usize::try_from(wire.num_spec_steps).map_err(|_| SpecDecodePayloadError::CountOverflow)?;
    if let Some(accepted) = &wire.per_step_accepted {
        if accepted.len() != expected_len {
            return Err(SpecDecodePayloadError::InconsistentCounts(
                "per_step_accepted length does not equal num_spec_steps",
            ));
        }
        let sum = accepted
            .iter()
            .try_fold(0_u64, |total, value| total.checked_add(*value))
            .ok_or(SpecDecodePayloadError::CountOverflow)?;
        if sum != wire.num_accepted_draft_tokens {
            return Err(SpecDecodePayloadError::InconsistentCounts(
                "per_step_accepted sum does not equal num_accepted_draft_tokens",
            ));
        }
    }
    if let Some(drafted) = &wire.per_step_drafted {
        if drafted.len() != expected_len {
            return Err(SpecDecodePayloadError::InconsistentCounts(
                "per_step_drafted length does not equal num_spec_steps",
            ));
        }
        let sum = drafted
            .iter()
            .try_fold(0_u64, |total, value| total.checked_add(*value))
            .ok_or(SpecDecodePayloadError::CountOverflow)?;
        if sum != wire.num_draft_tokens {
            return Err(SpecDecodePayloadError::InconsistentCounts(
                "per_step_drafted sum does not equal num_draft_tokens",
            ));
        }
    }
    if let (Some(accepted), Some(drafted)) = (&wire.per_step_accepted, &wire.per_step_drafted)
        && accepted
            .iter()
            .zip(drafted)
            .any(|(accepted, drafted)| accepted > drafted)
    {
        return Err(SpecDecodePayloadError::InconsistentCounts(
            "a verification step accepted more drafts than it proposed",
        ));
    }
    Ok(())
}
