// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Open metric identity.

use serde::{Deserialize, Deserializer, Serialize, Serializer, de};
use std::collections::{HashMap, HashSet};
use std::fmt::{Display, Formatter, Result as FmtResult};
use std::sync::{LazyLock, RwLock};

/// Dense index of one registered metric tag.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct MetricTagId(u32);

macro_rules! define_builtin_metric_tags {
    ($($variant:ident => $name:literal),+ $(,)?) => {
        #[repr(u32)]
        enum BuiltinMetricTag {
            $($variant,)+
            Count,
        }

        const BUILTIN_NAMES: [&str; BuiltinMetricTag::Count as usize] = [$($name,)+];
        const BUILTIN_SERDE_NAMES: [&str; BuiltinMetricTag::Count as usize] = [$(stringify!($variant),)+];

        #[allow(non_upper_case_globals)]
        impl MetricTagId {
            $(
                #[doc = concat!("Built-in metric tag `", $name, "`.")]
                pub const $variant: Self = Self(BuiltinMetricTag::$variant as u32);
            )+

            /// Number of built-in metric tags retained by the static catalog.
            pub const COUNT: usize = BuiltinMetricTag::Count as usize;

            const BUILTINS: [Self; Self::COUNT] = [$(Self::$variant,)+];
        }
    };
}

define_builtin_metric_tags! {
    RequestCount => "request_count",
    ErrorRequestCount => "error_request_count",
    CompletedRequestCount => "completed_request_count",
    RequestErrorRate => "request_error_rate",
    GoodRequestCount => "good_request_count",
    Goodput => "goodput",
    GoodRequestFraction => "good_request_fraction",
    MinRequestTimestamp => "min_request_timestamp",
    MaxResponseTimestamp => "max_response_timestamp",
    BenchmarkDuration => "benchmark_duration",
    RequestLatency => "request_latency",
    TimeToFirstToken => "time_to_first_token",
    TimeToSecondToken => "time_to_second_token",
    TimeToFirstOutputToken => "time_to_first_output_token",
    TimeToLastRoundTrip => "time_to_last_round_trip",
    AverageRoundTripTime => "avg_round_trip_time",
    InterTokenLatency => "inter_token_latency",
    DecodeDuration => "decode_duration",
    InterChunkLatency => "inter_chunk_latency",
    CreditDropLatency => "credit_drop_latency",
    OutputSequenceLength => "output_sequence_length",
    InputSequenceLength => "input_sequence_length",
    ErrorInputSequenceLength => "error_isl",
    OutputTokenCount => "output_token_count",
    ReasoningTokenCount => "reasoning_token_count",
    TotalOutputSequenceLength => "total_osl",
    TotalInputSequenceLength => "total_isl",
    TotalErrorInputSequenceLength => "total_error_isl",
    TotalOutputTokens => "total_output_tokens",
    TotalReasoningTokens => "total_reasoning_tokens",
    RequestThroughput => "request_throughput",
    InputTokenThroughput => "input_token_throughput",
    OutputTokenThroughput => "output_token_throughput",
    TotalTokenThroughput => "total_token_throughput",
    OutputTokenThroughputPerUser => "output_token_throughput_per_user",
    E2eOutputTokenThroughput => "e2e_output_token_throughput",
    PrefillThroughputPerUser => "prefill_throughput_per_user",
    Rtfx => "rtfx",
    UsagePromptTokens => "usage_prompt_tokens",
    UsageCompletionTokens => "usage_completion_tokens",
    UsageTotalTokens => "usage_total_tokens",
    UsageReasoningTokens => "usage_reasoning_tokens",
    UsagePromptAudioTokens => "usage_prompt_audio_tokens",
    UsageCompletionAudioTokens => "usage_completion_audio_tokens",
    UsageAcceptedPredictionTokens => "usage_accepted_prediction_tokens",
    UsageRejectedPredictionTokens => "usage_rejected_prediction_tokens",
    UsagePromptCacheReadTokens => "usage_prompt_cache_read_tokens",
    UsagePromptCacheWriteTokens => "usage_prompt_cache_write_tokens",
    UsagePromptCacheMissTokens => "usage_prompt_cache_miss_tokens",
    UsageToolUsePromptTokens => "usage_tool_use_prompt_tokens",
    UsagePromptAudioSeconds => "usage_prompt_audio_seconds",
    TotalUsagePromptTokens => "total_usage_prompt_tokens",
    TotalUsageCompletionTokens => "total_usage_completion_tokens",
    TotalUsageTotalTokens => "total_usage_total_tokens",
    TotalUsageReasoningTokens => "total_usage_reasoning_tokens",
    TotalUsagePromptCacheReadTokens => "total_usage_prompt_cache_read_tokens",
    TotalUsagePromptCacheWriteTokens => "total_usage_prompt_cache_write_tokens",
    TotalUsagePromptCacheMissTokens => "total_usage_prompt_cache_miss_tokens",
    TotalUsagePromptAudioTokens => "total_usage_prompt_audio_tokens",
    TotalUsageCompletionAudioTokens => "total_usage_completion_audio_tokens",
    TotalUsageAcceptedPredictionTokens => "total_usage_accepted_prediction_tokens",
    TotalUsageRejectedPredictionTokens => "total_usage_rejected_prediction_tokens",
    TotalUsageToolUsePromptTokens => "total_usage_tool_use_prompt_tokens",
    TotalUsagePromptAudioSeconds => "total_usage_prompt_audio_seconds",
    OverallUsagePromptCacheReadPct => "overall_usage_prompt_cache_read_pct",
    UsagePromptTokensDiffPct => "usage_prompt_tokens_diff_pct",
    UsageCompletionTokensDiffPct => "usage_completion_tokens_diff_pct",
    UsageReasoningTokensDiffPct => "usage_reasoning_tokens_diff_pct",
    UsageDiscrepancyCount => "usage_discrepancy_count",
    RequestedOutputSequenceLength => "requested_osl",
    OslMismatchDiffPct => "osl_mismatch_diff_pct",
    OslMismatchCount => "osl_mismatch_count",
    ThinkingEfficiency => "thinking_efficiency",
    OverallThinkingEfficiency => "overall_thinking_efficiency",
    TotalGpuPower => "total_gpu_power",
    TotalGpuEnergy => "total_gpu_energy",
    OutputTokensPerJoule => "output_tokens_per_joule",
    EnergyPerUser => "energy_per_user",
    NetworkAdjustedRequestLatency => "network_adjusted_request_latency",
    NetworkAdjustedTimeToFirstToken => "network_adjusted_time_to_first_token",
    NetworkAdjustedTimeToFirstOutputToken => "network_adjusted_time_to_first_output_token",
    NetworkRtt => "network_rtt",
    StreamSetupLatency => "stream_setup_latency",
    StreamPrefillLatency => "stream_prefill_latency",
    AccuracyCorrect => "accuracy.correct",
    AccuracyUnparsed => "accuracy.unparsed",
    AudioDuration => "audio_duration",
    NumImages => "num_images",
    ImageThroughput => "image_throughput",
    ImageLatency => "image_latency",
    TotalNumImages => "total_num_images",
    ImageSamplesPerSecond => "image_samples_per_second",
    VideoInferenceTime => "video_inference_time",
    VideoPeakMemory => "video_peak_memory",
    HttpReqBlocked => "http_req_blocked",
    HttpReqDnsLookup => "http_req_dns_lookup",
    HttpReqConnecting => "http_req_connecting",
    HttpReqSending => "http_req_sending",
    HttpReqWaiting => "http_req_waiting",
    HttpReqReceiving => "http_req_receiving",
    HttpReqDuration => "http_req_duration",
    HttpReqConnectionReused => "http_req_connection_reused",
    HttpReqDataSent => "http_req_data_sent",
    HttpReqDataReceived => "http_req_data_received",
    HttpReqChunksSent => "http_req_chunks_sent",
    HttpReqChunksReceived => "http_req_chunks_received",
    HttpReqConnectionOverhead => "http_req_connection_overhead",
    HttpReqTotal => "http_req_total",
    EffectiveLatency => "effective_latency",
    CreditToStartLatency => "credit_to_start_latency",
    EffectiveConcurrency => "effective_concurrency",
    EffectiveDecodeThroughput => "effective_decode_throughput",
    EffectivePrefillThroughput => "effective_prefill_throughput",
    EffectiveDecodeConcurrency => "effective_decode_concurrency",
    EffectivePrefillConcurrency => "effective_prefill_concurrency",
    EffectiveTotalThroughput => "effective_total_throughput",
    EffectiveDecodeThroughputPerUser => "effective_decode_throughput_per_user",
    EffectivePrefillThroughputPerUser => "effective_prefill_throughput_per_user",
    EffectiveImageSamplesPerSecond => "effective_image_samples_per_second",
    TokensInFlight => "tokens_in_flight",
    ActiveDecodeThroughput => "active_decode_throughput",
    ActivePrefillThroughput => "active_prefill_throughput",
    ActiveDecodeThroughputPerUser => "active_decode_throughput_per_user",
    ActivePrefillThroughputPerUser => "active_prefill_throughput_per_user",
    ActiveImageSamplesPerSecond => "active_image_samples_per_second",
    EffectiveImageSamplesPerSecondPerUser => "effective_image_samples_per_second_per_user",
    ActiveTotalThroughput => "active_total_throughput",
    SpecDecodeAcceptanceLength => "spec_decode_acceptance_length",
    SpecDecodeTokenWeightedAcceptanceLength => "spec_decode_token_weighted_acceptance_length",
    SpecDecodeDraftAcceptanceRate => "spec_decode_draft_acceptance_rate",
    SpecDecodeOverallDraftAcceptanceRate => "spec_decode_overall_draft_acceptance_rate",
    SpecDecodeAcceptedPerVerified => "spec_decode_accepted_per_verified",
    SpecDecodeSteps => "spec_decode_steps",
    SpecDecodeAcceptedDraftTokens => "spec_decode_accepted_draft_tokens",
    SpecDecodeDraftTokens => "spec_decode_draft_tokens",
    TotalSpecDecodeSteps => "total_spec_decode_steps",
    TotalAcceptedDraftTokens => "total_accepted_draft_tokens",
    TotalDraftTokens => "total_draft_tokens",
}

struct MetricTagInterner {
    names: Vec<&'static str>,
    ids: HashMap<&'static str, MetricTagId>,
}

static INTERNER: LazyLock<RwLock<MetricTagInterner>> = LazyLock::new(|| {
    let names = BUILTIN_NAMES.to_vec();
    let ids = names
        .iter()
        .enumerate()
        .map(|(index, name)| (*name, MetricTagId(index as u32)))
        .collect();
    RwLock::new(MetricTagInterner { names, ids })
});

impl MetricTagId {
    /// Return this tag's zero-based dense table index.
    #[inline(always)]
    pub const fn index(self) -> usize {
        self.0 as usize
    }

    /// Return the stable report spelling.
    pub fn as_str(&self) -> &'static str {
        if let Some(name) = BUILTIN_NAMES.get(self.0 as usize) {
            return name;
        }
        let interner = match INTERNER.read() {
            Ok(interner) => interner,
            Err(poisoned) => poisoned.into_inner(),
        };
        interner
            .names
            .get(self.0 as usize)
            .copied()
            .unwrap_or("invalid_metric_tag")
    }

    /// Resolve a process-interned metric tag by stable report spelling.
    pub fn resolve(name: &str) -> Option<Self> {
        let interner = match INTERNER.read() {
            Ok(interner) => interner,
            Err(poisoned) => poisoned.into_inner(),
        };
        interner.ids.get(name).copied()
    }

    /// Resolve a name only when it belongs to `registry`.
    pub fn resolve_in(registry: &MetricTagRegistry, name: &str) -> Option<Self> {
        Self::resolve(name).filter(|id| registry.ids.contains(id))
    }
}

/// Return the static catalog spelling for one built-in tag.
pub(crate) const fn builtin_name(id: MetricTagId) -> &'static str {
    BUILTIN_NAMES[id.index()]
}

impl Display for MetricTagId {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> FmtResult {
        formatter.write_str(self.as_str())
    }
}

impl Serialize for MetricTagId {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let name = BUILTIN_SERDE_NAMES
            .get(self.0 as usize)
            .copied()
            .unwrap_or_else(|| self.as_str());
        serializer.serialize_str(name)
    }
}

impl<'de> Deserialize<'de> for MetricTagId {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let name = String::deserialize(deserializer)?;
        if let Some(id) = Self::resolve(&name) {
            return Ok(id);
        }
        BUILTIN_SERDE_NAMES
            .iter()
            .position(|candidate| *candidate == name)
            .map(|index| Self(index as u32))
            .ok_or_else(|| de::Error::custom(format!("unknown metric tag {name:?}")))
    }
}

/// Set of metric tags available to one host registry.
#[derive(Debug, Clone)]
pub struct MetricTagRegistry {
    ids: HashSet<MetricTagId>,
}

impl MetricTagRegistry {
    /// Construct a registry containing every built-in metric tag.
    pub fn builtin() -> Self {
        Self {
            ids: MetricTagId::BUILTINS.into_iter().collect(),
        }
    }

    /// Register one stable metric-tag spelling.
    pub fn register(&mut self, name: &str) -> Result<MetricTagId, String> {
        if name.is_empty() {
            return Err("metric tag name must not be empty".to_string());
        }
        if MetricTagId::resolve_in(self, name).is_some() {
            return Err(format!("metric tag {name:?} is already registered"));
        }

        let mut interner = match INTERNER.write() {
            Ok(interner) => interner,
            Err(poisoned) => poisoned.into_inner(),
        };
        let id = if let Some(id) = interner.ids.get(name).copied() {
            id
        } else {
            let index = u32::try_from(interner.names.len())
                .map_err(|_| "more than u32::MAX metric tags were registered".to_string())?;
            let name: &'static str = Box::leak(name.to_owned().into_boxed_str());
            let id = MetricTagId(index);
            interner.names.push(name);
            interner.ids.insert(name, id);
            id
        };
        self.ids.insert(id);
        Ok(id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_new_metric_tag_registers_without_touching_the_enum() {
        let mut registry = MetricTagRegistry::builtin();
        let id = registry
            .register("plugin.metric")
            .expect("register a new metric tag");

        assert_eq!(id.as_str(), "plugin.metric");
        assert_eq!(MetricTagId::resolve_in(&registry, "plugin.metric"), Some(id));
    }

    #[test]
    fn duplicate_metric_tag_registration_is_rejected() {
        let mut registry = MetricTagRegistry::builtin();

        assert!(registry.register("request_count").is_err());
    }
}
