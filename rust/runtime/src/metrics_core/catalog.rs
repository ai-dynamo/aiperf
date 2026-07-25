// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Static metric catalog and dependency validation.
//!
//! Each catalog row declares identity, units, flags, kind, aggregation, and metric
//! dependencies. This module validates the registry and dependency order;
//! accumulation is implemented in [`crate::metrics_core::accumulator`].

use crate::metrics_core::definition::{Definition, DefinitionGroup};
use crate::metrics_core::{MetricValueType, Unit};
use bitflags::bitflags;
use petgraph::algo::toposort;
use petgraph::graphmap::DiGraphMap;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeSet, HashMap};
use std::fmt::{Display, Formatter, Result as FmtResult};
use std::sync::LazyLock;

/// Metric identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum MetricTag {
    RequestCount,
    ErrorRequestCount,
    CompletedRequestCount,
    RequestErrorRate,
    GoodRequestCount,
    Goodput,
    GoodRequestFraction,
    MinRequestTimestamp,
    MaxResponseTimestamp,
    BenchmarkDuration,
    RequestLatency,
    TimeToFirstToken,
    TimeToSecondToken,
    TimeToFirstOutputToken,
    InterTokenLatency,
    InterChunkLatency,
    DecodeDuration,
    CreditDropLatency,
    OutputSequenceLength,
    InputSequenceLength,
    ErrorInputSequenceLength,
    OutputTokenCount,
    ReasoningTokenCount,
    TotalOutputSequenceLength,
    TotalInputSequenceLength,
    TotalErrorInputSequenceLength,
    TotalOutputTokens,
    TotalReasoningTokens,
    RequestThroughput,
    InputTokenThroughput,
    OutputTokenThroughput,
    TotalTokenThroughput,
    OutputTokenThroughputPerUser,
    E2eOutputTokenThroughput,
    PrefillThroughputPerUser,
    Rtfx,
    UsagePromptTokens,
    UsageCompletionTokens,
    UsageTotalTokens,
    UsageReasoningTokens,
    UsagePromptAudioTokens,
    UsageCompletionAudioTokens,
    UsageAcceptedPredictionTokens,
    UsageRejectedPredictionTokens,
    UsagePromptCacheReadTokens,
    UsagePromptCacheWriteTokens,
    UsagePromptCacheMissTokens,
    UsageToolUsePromptTokens,
    UsagePromptAudioSeconds,
    TotalUsagePromptTokens,
    TotalUsageCompletionTokens,
    TotalUsageTotalTokens,
    TotalUsageReasoningTokens,
    TotalUsagePromptCacheReadTokens,
    TotalUsagePromptCacheWriteTokens,
    TotalUsagePromptCacheMissTokens,
    TotalUsagePromptAudioTokens,
    TotalUsageCompletionAudioTokens,
    TotalUsageAcceptedPredictionTokens,
    TotalUsageRejectedPredictionTokens,
    TotalUsageToolUsePromptTokens,
    TotalUsagePromptAudioSeconds,
    OverallUsagePromptCacheReadPct,
    UsagePromptTokensDiffPct,
    UsageCompletionTokensDiffPct,
    UsageReasoningTokensDiffPct,
    UsageDiscrepancyCount,
    RequestedOutputSequenceLength,
    OslMismatchDiffPct,
    OslMismatchCount,
    ThinkingEfficiency,
    OverallThinkingEfficiency,
    TotalGpuPower,
    TotalGpuEnergy,
    OutputTokensPerJoule,
    EnergyPerUser,
    NetworkAdjustedRequestLatency,
    NetworkAdjustedTimeToFirstToken,
    NetworkAdjustedTimeToFirstOutputToken,
    NetworkRtt,
    StreamSetupLatency,
    StreamPrefillLatency,
    AccuracyCorrect,
    AccuracyUnparsed,
    AudioDuration,
    NumImages,
    ImageThroughput,
    ImageLatency,
    TotalNumImages,
    ImageSamplesPerSecond,
    VideoInferenceTime,
    VideoPeakMemory,
    HttpReqBlocked,
    HttpReqDnsLookup,
    HttpReqConnecting,
    HttpReqSending,
    HttpReqWaiting,
    HttpReqReceiving,
    HttpReqDuration,
    HttpReqConnectionReused,
    HttpReqDataSent,
    HttpReqDataReceived,
    HttpReqChunksSent,
    HttpReqChunksReceived,
    HttpReqConnectionOverhead,
    HttpReqTotal,
    EffectiveLatency,
    CreditToStartLatency,
    EffectiveConcurrency,
    EffectiveDecodeThroughput,
    EffectivePrefillThroughput,
    EffectiveDecodeConcurrency,
    EffectivePrefillConcurrency,
    EffectiveTotalThroughput,
    EffectiveDecodeThroughputPerUser,
    EffectivePrefillThroughputPerUser,
    EffectiveImageSamplesPerSecond,
    TokensInFlight,
    ActiveDecodeThroughput,
    ActivePrefillThroughput,
    ActiveDecodeThroughputPerUser,
    ActivePrefillThroughputPerUser,
    ActiveImageSamplesPerSecond,
    EffectiveImageSamplesPerSecondPerUser,
    ActiveTotalThroughput,
}

impl MetricTag {
    /// Number of distinct tags. The variants are declared contiguously from
    /// discriminant 0, so `ActiveTotalThroughput` (the last) plus one is the
    /// dense column count. A reordering that makes it no longer last would only
    /// oversize the backing array, never index out of bounds.
    pub const COUNT: usize = MetricTag::ActiveTotalThroughput as usize + 1;

    /// Dense array index for this tag — its zero-based declaration discriminant.
    #[inline(always)]
    pub const fn index(self) -> usize {
        self as usize
    }

    /// Stable snake-case report spelling.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::RequestCount => "request_count",
            Self::ErrorRequestCount => "error_request_count",
            Self::CompletedRequestCount => "completed_request_count",
            Self::RequestErrorRate => "request_error_rate",
            Self::GoodRequestCount => "good_request_count",
            Self::Goodput => "goodput",
            Self::GoodRequestFraction => "good_request_fraction",
            Self::MinRequestTimestamp => "min_request_timestamp",
            Self::MaxResponseTimestamp => "max_response_timestamp",
            Self::BenchmarkDuration => "benchmark_duration",
            Self::RequestLatency => "request_latency",
            Self::TimeToFirstToken => "time_to_first_token",
            Self::TimeToSecondToken => "time_to_second_token",
            Self::TimeToFirstOutputToken => "time_to_first_output_token",
            Self::InterTokenLatency => "inter_token_latency",
            Self::DecodeDuration => "decode_duration",
            Self::InterChunkLatency => "inter_chunk_latency",
            Self::CreditDropLatency => "credit_drop_latency",
            Self::OutputSequenceLength => "output_sequence_length",
            Self::InputSequenceLength => "input_sequence_length",
            Self::ErrorInputSequenceLength => "error_isl",
            Self::OutputTokenCount => "output_token_count",
            Self::ReasoningTokenCount => "reasoning_token_count",
            Self::TotalOutputSequenceLength => "total_osl",
            Self::TotalInputSequenceLength => "total_isl",
            Self::TotalErrorInputSequenceLength => "total_error_isl",
            Self::TotalOutputTokens => "total_output_tokens",
            Self::TotalReasoningTokens => "total_reasoning_tokens",
            Self::RequestThroughput => "request_throughput",
            Self::InputTokenThroughput => "input_token_throughput",
            Self::OutputTokenThroughput => "output_token_throughput",
            Self::TotalTokenThroughput => "total_token_throughput",
            Self::OutputTokenThroughputPerUser => "output_token_throughput_per_user",
            Self::E2eOutputTokenThroughput => "e2e_output_token_throughput",
            Self::PrefillThroughputPerUser => "prefill_throughput_per_user",
            Self::Rtfx => "rtfx",
            Self::UsagePromptTokens => "usage_prompt_tokens",
            Self::UsageCompletionTokens => "usage_completion_tokens",
            Self::UsageTotalTokens => "usage_total_tokens",
            Self::UsageReasoningTokens => "usage_reasoning_tokens",
            Self::UsagePromptAudioTokens => "usage_prompt_audio_tokens",
            Self::UsageCompletionAudioTokens => "usage_completion_audio_tokens",
            Self::UsageAcceptedPredictionTokens => "usage_accepted_prediction_tokens",
            Self::UsageRejectedPredictionTokens => "usage_rejected_prediction_tokens",
            Self::UsagePromptCacheReadTokens => "usage_prompt_cache_read_tokens",
            Self::UsagePromptCacheWriteTokens => "usage_prompt_cache_write_tokens",
            Self::UsagePromptCacheMissTokens => "usage_prompt_cache_miss_tokens",
            Self::UsageToolUsePromptTokens => "usage_tool_use_prompt_tokens",
            Self::UsagePromptAudioSeconds => "usage_prompt_audio_seconds",
            Self::TotalUsagePromptTokens => "total_usage_prompt_tokens",
            Self::TotalUsageCompletionTokens => "total_usage_completion_tokens",
            Self::TotalUsageTotalTokens => "total_usage_total_tokens",
            Self::TotalUsageReasoningTokens => "total_usage_reasoning_tokens",
            Self::TotalUsagePromptCacheReadTokens => "total_usage_prompt_cache_read_tokens",
            Self::TotalUsagePromptCacheWriteTokens => "total_usage_prompt_cache_write_tokens",
            Self::TotalUsagePromptCacheMissTokens => "total_usage_prompt_cache_miss_tokens",
            Self::TotalUsagePromptAudioTokens => "total_usage_prompt_audio_tokens",
            Self::TotalUsageCompletionAudioTokens => "total_usage_completion_audio_tokens",
            Self::TotalUsageAcceptedPredictionTokens => "total_usage_accepted_prediction_tokens",
            Self::TotalUsageRejectedPredictionTokens => "total_usage_rejected_prediction_tokens",
            Self::TotalUsageToolUsePromptTokens => "total_usage_tool_use_prompt_tokens",
            Self::TotalUsagePromptAudioSeconds => "total_usage_prompt_audio_seconds",
            Self::OverallUsagePromptCacheReadPct => "overall_usage_prompt_cache_read_pct",
            Self::UsagePromptTokensDiffPct => "usage_prompt_tokens_diff_pct",
            Self::UsageCompletionTokensDiffPct => "usage_completion_tokens_diff_pct",
            Self::UsageReasoningTokensDiffPct => "usage_reasoning_tokens_diff_pct",
            Self::UsageDiscrepancyCount => "usage_discrepancy_count",
            Self::RequestedOutputSequenceLength => "requested_osl",
            Self::OslMismatchDiffPct => "osl_mismatch_diff_pct",
            Self::OslMismatchCount => "osl_mismatch_count",
            Self::ThinkingEfficiency => "thinking_efficiency",
            Self::OverallThinkingEfficiency => "overall_thinking_efficiency",
            Self::TotalGpuPower => "total_gpu_power",
            Self::TotalGpuEnergy => "total_gpu_energy",
            Self::OutputTokensPerJoule => "output_tokens_per_joule",
            Self::EnergyPerUser => "energy_per_user",
            Self::NetworkAdjustedRequestLatency => "network_adjusted_request_latency",
            Self::NetworkAdjustedTimeToFirstToken => "network_adjusted_time_to_first_token",
            Self::NetworkAdjustedTimeToFirstOutputToken => {
                "network_adjusted_time_to_first_output_token"
            }
            Self::NetworkRtt => "network_rtt",
            Self::StreamSetupLatency => "stream_setup_latency",
            Self::StreamPrefillLatency => "stream_prefill_latency",
            Self::AccuracyCorrect => "accuracy.correct",
            Self::AccuracyUnparsed => "accuracy.unparsed",
            Self::AudioDuration => "audio_duration",
            Self::NumImages => "num_images",
            Self::ImageThroughput => "image_throughput",
            Self::ImageLatency => "image_latency",
            Self::TotalNumImages => "total_num_images",
            Self::ImageSamplesPerSecond => "image_samples_per_second",
            Self::VideoInferenceTime => "video_inference_time",
            Self::VideoPeakMemory => "video_peak_memory",
            Self::HttpReqBlocked => "http_req_blocked",
            Self::HttpReqDnsLookup => "http_req_dns_lookup",
            Self::HttpReqConnecting => "http_req_connecting",
            Self::HttpReqSending => "http_req_sending",
            Self::HttpReqWaiting => "http_req_waiting",
            Self::HttpReqReceiving => "http_req_receiving",
            Self::HttpReqDuration => "http_req_duration",
            Self::HttpReqConnectionReused => "http_req_connection_reused",
            Self::HttpReqDataSent => "http_req_data_sent",
            Self::HttpReqDataReceived => "http_req_data_received",
            Self::HttpReqChunksSent => "http_req_chunks_sent",
            Self::HttpReqChunksReceived => "http_req_chunks_received",
            Self::HttpReqConnectionOverhead => "http_req_connection_overhead",
            Self::HttpReqTotal => "http_req_total",
            Self::EffectiveLatency => "effective_latency",
            Self::CreditToStartLatency => "credit_to_start_latency",
            Self::EffectiveConcurrency => "effective_concurrency",
            Self::EffectiveDecodeThroughput => "effective_decode_throughput",
            Self::EffectivePrefillThroughput => "effective_prefill_throughput",
            Self::EffectiveDecodeConcurrency => "effective_decode_concurrency",
            Self::EffectivePrefillConcurrency => "effective_prefill_concurrency",
            Self::EffectiveTotalThroughput => "effective_total_throughput",
            Self::EffectiveDecodeThroughputPerUser => "effective_decode_throughput_per_user",
            Self::EffectivePrefillThroughputPerUser => "effective_prefill_throughput_per_user",
            Self::EffectiveImageSamplesPerSecond => "effective_image_samples_per_second",
            Self::TokensInFlight => "tokens_in_flight",
            Self::ActiveDecodeThroughput => "active_decode_throughput",
            Self::ActivePrefillThroughput => "active_prefill_throughput",
            Self::ActiveDecodeThroughputPerUser => "active_decode_throughput_per_user",
            Self::ActivePrefillThroughputPerUser => "active_prefill_throughput_per_user",
            Self::ActiveImageSamplesPerSecond => "active_image_samples_per_second",
            Self::EffectiveImageSamplesPerSecondPerUser => {
                "effective_image_samples_per_second_per_user"
            }
            Self::ActiveTotalThroughput => "active_total_throughput",
        }
    }
}

impl Display for MetricTag {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        f.write_str(self.as_str())
    }
}

/// Metric compute tier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MetricType {
    /// One value per valid record.
    Record,
    /// Scalar fold over records.
    Aggregate,
    /// Scalar derived from other summarized metrics.
    Derived,
}

/// Fold used by aggregate metrics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AggregationKind {
    /// Sum present values.
    Sum,
    /// Maximum present value.
    Max,
    /// Minimum present value.
    Min,
}

/// Console grouping.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MetricConsoleGroup {
    None,
    Default,
    Usage,
    Cache,
    Prediction,
    Audio,
    Reasoning,
    Effective,
    Active,
}

/// Direction used by plot/threshold consumers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PlotMetricDirection {
    LargerIsBetter,
    SmallerIsBetter,
    Neutral,
}

bitflags! {
    /// Applicability and presentation flags. Bit 3 is intentionally reserved.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct MetricFlags: u64 {
        /// No flags.
        const NONE = 0;
        /// Requires a streaming response.
        const STREAMING_ONLY = 1 << 0;
        /// Metric applies only to failed records.
        const ERROR_ONLY = 1 << 1;
        /// Requires produced output tokens.
        const PRODUCES_TOKENS_ONLY = 1 << 2;
        /// Larger values are preferable; bit 3 remains reserved.
        const LARGER_IS_BETTER = 1 << 4;
        /// Internal helper metric.
        const INTERNAL = 1 << 5;
        /// Supports audio endpoints.
        const SUPPORTS_AUDIO_ONLY = 1 << 6;
        /// Supports image endpoints.
        const SUPPORTS_IMAGE_ONLY = 1 << 7;
        /// Supports reasoning-token endpoints.
        const SUPPORTS_REASONING = 1 << 8;
        /// Experimental output.
        const EXPERIMENTAL = 1 << 9;
        /// Metric participates in goodput.
        const GOODPUT = 1 << 10;
        /// Hide per-record output.
        const NO_INDIVIDUAL_RECORDS = 1 << 11;
        /// Requires tokenized input.
        const TOKENIZES_INPUT_ONLY = 1 << 12;
        /// Supports video-input endpoints.
        const SUPPORTS_VIDEO_ONLY = 1 << 13;
        /// Metric is usage-difference only.
        const USAGE_DIFF_ONLY = 1 << 14;
        /// Metric requires HTTP trace data.
        const HTTP_TRACE_ONLY = 1 << 15;
        /// Requires video-producing endpoints.
        const PRODUCES_VIDEO_ONLY = 1 << 16;
        /// Percentiles include failed requests through adj_*.
        const PERCENTILE_INCLUDES_FAILED_REQUESTS = 1 << 17;
        /// Requires streaming token arrivals and produced output tokens.
        const STREAMING_TOKENS_ONLY = Self::STREAMING_ONLY.bits() | Self::PRODUCES_TOKENS_ONLY.bits();
    }
}

impl MetricFlags {
    /// Returns true when all requested flags are set.
    pub fn has_all(self, flags: Self) -> bool {
        self.contains(flags)
    }

    /// Returns true when any requested flag is set.
    pub fn has_any(self, flags: Self) -> bool {
        self.intersects(flags)
    }

    /// Returns true when none of the requested flags are missing.
    pub fn missing(self, flags: Self) -> bool {
        flags.is_empty() || !self.intersects(flags)
    }
}

/// Declarative metric catalog row.
#[derive(Debug, Clone, Copy)]
pub struct MetricSpec {
    pub tag: MetricTag,
    /// Embedded portable definition (dual-write with the legacy presentation
    /// fields below; later tasks remove the legacy fields).
    pub def: Definition,
    pub header: &'static str,
    pub short_header: Option<&'static str>,
    pub short_header_hide_unit: bool,
    pub unit: Unit,
    pub display_unit: Option<Unit>,
    pub display_order: Option<u32>,
    pub flags: MetricFlags,
    pub console_group: MetricConsoleGroup,
    pub plot_direction: PlotMetricDirection,
    pub required: &'static [MetricTag],
    pub value_type: MetricValueType,
    pub kind: MetricType,
    pub aggregation: Option<AggregationKind>,
}

impl MetricSpec {
    /// Display header, delegating to the embedded [`Definition`].
    pub fn header(&self) -> &'static str {
        self.def.header
    }

    /// Canonical unit, delegating to the embedded [`Definition`].
    pub fn unit(&self) -> Unit {
        self.def.unit
    }

    /// Optional display unit override, delegating to the embedded [`Definition`].
    pub fn display_unit(&self) -> Option<Unit> {
        self.def.display_unit
    }

    /// Optional short header, delegating to the embedded [`Definition`].
    pub fn short_header(&self) -> Option<&'static str> {
        self.def.short_header
    }

    /// Optional display order, delegating to the embedded [`Definition`].
    pub fn display_order(&self) -> Option<u32> {
        self.def.display_order
    }
}

macro_rules! spec {
    ($tag:ident, $header:expr, $unit:ident, $kind:ident, $agg:expr, $flags:expr, [$($req:ident),* $(,)?]) => {
        MetricSpec {
            tag: MetricTag::$tag,
            def: Definition {
                id: MetricTag::$tag.as_str(),
                header: $header,
                short_header: None,
                short_header_hide_unit: false,
                unit: Unit::$unit,
                display_unit: None,
                display_order: None,
                group: DefinitionGroup::Default,
                larger_is_better: $flags.contains(MetricFlags::LARGER_IS_BETTER),
                value_type: MetricValueType::Float,
                aliases: &[],
                deprecated_since: None,
            },
            header: $header,
            short_header: None,
            short_header_hide_unit: false,
            unit: Unit::$unit,
            display_unit: None,
            display_order: None,
            flags: $flags,
            console_group: MetricConsoleGroup::Default,
            plot_direction: if $flags.contains(MetricFlags::LARGER_IS_BETTER) { PlotMetricDirection::LargerIsBetter } else { PlotMetricDirection::SmallerIsBetter },
            required: &[$(MetricTag::$req),*],
            value_type: MetricValueType::Float,
            kind: MetricType::$kind,
            aggregation: $agg,
        }
    };
}

/// Static metric catalog. Injected rows receive values from telemetry, accuracy,
/// or explicit record overrides.
pub static CATALOG: LazyLock<Vec<MetricSpec>> = LazyLock::new(|| {
    let mut catalog = vec![
        spec!(
            RequestCount,
            "Request Count",
            Request,
            Aggregate,
            Some(AggregationKind::Sum),
            MetricFlags::LARGER_IS_BETTER | MetricFlags::NO_INDIVIDUAL_RECORDS,
            []
        ),
        spec!(
            ErrorRequestCount,
            "Error Request Count",
            Request,
            Aggregate,
            Some(AggregationKind::Sum),
            MetricFlags::ERROR_ONLY | MetricFlags::NO_INDIVIDUAL_RECORDS,
            []
        ),
        spec!(
            CompletedRequestCount,
            "Completed Requests (Success + Error)",
            Request,
            Derived,
            None,
            MetricFlags::NO_INDIVIDUAL_RECORDS,
            [RequestCount, ErrorRequestCount]
        ),
        spec!(
            RequestErrorRate,
            "Request Error Rate",
            Percent,
            Derived,
            None,
            MetricFlags::NO_INDIVIDUAL_RECORDS,
            [RequestCount, ErrorRequestCount]
        ),
        spec!(
            GoodRequestCount,
            "Good Request Count",
            Request,
            Aggregate,
            Some(AggregationKind::Sum),
            MetricFlags::GOODPUT,
            []
        ),
        spec!(
            Goodput,
            "Goodput",
            RequestsPerSecond,
            Derived,
            None,
            MetricFlags::GOODPUT,
            [GoodRequestCount, BenchmarkDuration]
        ),
        spec!(
            GoodRequestFraction,
            "GoodRequestFraction",
            Ratio,
            Derived,
            None,
            MetricFlags::GOODPUT | MetricFlags::LARGER_IS_BETTER,
            [GoodRequestCount, RequestCount]
        ),
        spec!(
            MinRequestTimestamp,
            "Minimum Request Timestamp",
            Nanosecond,
            Aggregate,
            Some(AggregationKind::Min),
            MetricFlags::NO_INDIVIDUAL_RECORDS | MetricFlags::INTERNAL,
            []
        ),
        spec!(
            MaxResponseTimestamp,
            "Maximum Response Timestamp",
            Nanosecond,
            Aggregate,
            Some(AggregationKind::Max),
            MetricFlags::NO_INDIVIDUAL_RECORDS | MetricFlags::INTERNAL,
            [RequestLatency]
        ),
        spec!(
            BenchmarkDuration,
            "Benchmark Duration",
            Nanosecond,
            Derived,
            None,
            MetricFlags::NONE,
            [MinRequestTimestamp, MaxResponseTimestamp]
        ),
        spec!(
            RequestLatency,
            "Request Latency",
            Nanosecond,
            Record,
            None,
            MetricFlags::PERCENTILE_INCLUDES_FAILED_REQUESTS,
            []
        ),
        spec!(
            TimeToFirstToken,
            "Time to First Token",
            Nanosecond,
            Record,
            None,
            MetricFlags::STREAMING_TOKENS_ONLY | MetricFlags::PERCENTILE_INCLUDES_FAILED_REQUESTS,
            []
        ),
        spec!(
            TimeToSecondToken,
            "Time to Second Token",
            Nanosecond,
            Record,
            None,
            MetricFlags::STREAMING_TOKENS_ONLY,
            []
        ),
        spec!(
            TimeToFirstOutputToken,
            "Time to First Output Token",
            Nanosecond,
            Record,
            None,
            MetricFlags::STREAMING_TOKENS_ONLY | MetricFlags::SUPPORTS_REASONING,
            []
        ),
        spec!(
            InterTokenLatency,
            "Inter Token Latency",
            Nanosecond,
            Record,
            None,
            MetricFlags::STREAMING_TOKENS_ONLY | MetricFlags::PERCENTILE_INCLUDES_FAILED_REQUESTS,
            [RequestLatency, TimeToFirstToken, OutputSequenceLength]
        ),
        spec!(
            InterChunkLatency,
            "Inter Chunk Latency",
            Nanosecond,
            Record,
            None,
            MetricFlags::STREAMING_TOKENS_ONLY,
            []
        ),
        spec!(
            DecodeDuration,
            "Decode Duration",
            Nanosecond,
            Record,
            None,
            MetricFlags::STREAMING_TOKENS_ONLY | MetricFlags::PERCENTILE_INCLUDES_FAILED_REQUESTS,
            [RequestLatency, TimeToFirstToken]
        ),
        spec!(
            CreditDropLatency,
            "Credit Drop Latency",
            Nanosecond,
            Record,
            None,
            MetricFlags::INTERNAL,
            []
        ),
        spec!(
            OutputSequenceLength,
            "Output Sequence Length",
            Token,
            Record,
            None,
            MetricFlags::PRODUCES_TOKENS_ONLY | MetricFlags::LARGER_IS_BETTER,
            []
        ),
        spec!(
            InputSequenceLength,
            "Input Sequence Length",
            Token,
            Record,
            None,
            MetricFlags::TOKENIZES_INPUT_ONLY | MetricFlags::LARGER_IS_BETTER,
            []
        ),
        spec!(
            ErrorInputSequenceLength,
            "Error Input Sequence Length",
            Token,
            Record,
            None,
            MetricFlags::TOKENIZES_INPUT_ONLY
                | MetricFlags::ERROR_ONLY
                | MetricFlags::LARGER_IS_BETTER,
            []
        ),
        spec!(
            OutputTokenCount,
            "Output Token Count",
            Token,
            Record,
            None,
            MetricFlags::PRODUCES_TOKENS_ONLY | MetricFlags::LARGER_IS_BETTER,
            []
        ),
        spec!(
            ReasoningTokenCount,
            "Reasoning Token Count",
            Token,
            Record,
            None,
            MetricFlags::PRODUCES_TOKENS_ONLY
                | MetricFlags::LARGER_IS_BETTER
                | MetricFlags::SUPPORTS_REASONING,
            []
        ),
        spec!(
            TotalOutputSequenceLength,
            "Total Output Sequence Length",
            Token,
            Derived,
            None,
            MetricFlags::PRODUCES_TOKENS_ONLY | MetricFlags::LARGER_IS_BETTER,
            [OutputSequenceLength]
        ),
        spec!(
            TotalInputSequenceLength,
            "Total Input Sequence Length",
            Token,
            Derived,
            None,
            MetricFlags::TOKENIZES_INPUT_ONLY | MetricFlags::LARGER_IS_BETTER,
            [InputSequenceLength]
        ),
        spec!(
            TotalErrorInputSequenceLength,
            "Total Error Input Sequence Length",
            Token,
            Derived,
            None,
            MetricFlags::TOKENIZES_INPUT_ONLY
                | MetricFlags::ERROR_ONLY
                | MetricFlags::LARGER_IS_BETTER,
            [ErrorInputSequenceLength]
        ),
        spec!(
            TotalOutputTokens,
            "Total Output Tokens",
            Token,
            Derived,
            None,
            MetricFlags::PRODUCES_TOKENS_ONLY | MetricFlags::LARGER_IS_BETTER,
            [OutputTokenCount]
        ),
        spec!(
            TotalReasoningTokens,
            "Total Reasoning Tokens",
            Token,
            Derived,
            None,
            MetricFlags::PRODUCES_TOKENS_ONLY | MetricFlags::LARGER_IS_BETTER,
            [ReasoningTokenCount]
        ),
        spec!(
            RequestThroughput,
            "Request Throughput",
            RequestsPerSecond,
            Derived,
            None,
            MetricFlags::LARGER_IS_BETTER,
            [RequestCount, BenchmarkDuration]
        ),
        spec!(
            InputTokenThroughput,
            "Input Token Throughput",
            TokensPerSecond,
            Derived,
            None,
            MetricFlags::LARGER_IS_BETTER,
            [TotalInputSequenceLength, BenchmarkDuration]
        ),
        spec!(
            OutputTokenThroughput,
            "Output Token Throughput",
            TokensPerSecond,
            Derived,
            None,
            MetricFlags::PRODUCES_TOKENS_ONLY | MetricFlags::LARGER_IS_BETTER,
            [TotalOutputSequenceLength, BenchmarkDuration]
        ),
        spec!(
            TotalTokenThroughput,
            "Total Token Throughput",
            TokensPerSecond,
            Derived,
            None,
            MetricFlags::PRODUCES_TOKENS_ONLY | MetricFlags::LARGER_IS_BETTER,
            [
                TotalInputSequenceLength,
                TotalOutputSequenceLength,
                BenchmarkDuration
            ]
        ),
        spec!(
            OutputTokenThroughputPerUser,
            "Output Token Throughput Per User",
            TokensPerSecondPerUser,
            Record,
            None,
            MetricFlags::STREAMING_TOKENS_ONLY | MetricFlags::LARGER_IS_BETTER,
            [InterTokenLatency]
        ),
        spec!(
            E2eOutputTokenThroughput,
            "E2E Output Token Throughput",
            TokensPerSecondPerUser,
            Record,
            None,
            MetricFlags::PRODUCES_TOKENS_ONLY | MetricFlags::LARGER_IS_BETTER,
            [OutputSequenceLength, RequestLatency]
        ),
        spec!(
            PrefillThroughputPerUser,
            "Prefill Throughput Per User",
            TokensPerSecondPerUser,
            Record,
            None,
            MetricFlags::STREAMING_TOKENS_ONLY
                | MetricFlags::TOKENIZES_INPUT_ONLY
                | MetricFlags::LARGER_IS_BETTER,
            [InputSequenceLength, TimeToFirstToken]
        ),
        spec!(
            Rtfx,
            "Inverse Real-Time Factor (RTFx)",
            Ratio,
            Record,
            None,
            MetricFlags::SUPPORTS_AUDIO_ONLY | MetricFlags::LARGER_IS_BETTER,
            [AudioDuration, RequestLatency]
        ),
        spec!(
            UsagePromptTokens,
            "Usage Prompt Tokens",
            Token,
            Record,
            None,
            MetricFlags::TOKENIZES_INPUT_ONLY | MetricFlags::LARGER_IS_BETTER,
            []
        ),
        spec!(
            UsageCompletionTokens,
            "Usage Completion Tokens",
            Token,
            Record,
            None,
            MetricFlags::PRODUCES_TOKENS_ONLY | MetricFlags::LARGER_IS_BETTER,
            []
        ),
        spec!(
            UsageTotalTokens,
            "Usage Total Tokens",
            Token,
            Record,
            None,
            MetricFlags::PRODUCES_TOKENS_ONLY | MetricFlags::LARGER_IS_BETTER,
            []
        ),
        spec!(
            UsageReasoningTokens,
            "Usage Reasoning Tokens",
            Token,
            Record,
            None,
            MetricFlags::PRODUCES_TOKENS_ONLY | MetricFlags::LARGER_IS_BETTER,
            []
        ),
        spec!(
            UsagePromptAudioTokens,
            "Usage Prompt Audio Tokens",
            Token,
            Record,
            None,
            MetricFlags::LARGER_IS_BETTER | MetricFlags::SUPPORTS_AUDIO_ONLY,
            []
        ),
        spec!(
            UsageCompletionAudioTokens,
            "Usage Completion Audio Tokens",
            Token,
            Record,
            None,
            MetricFlags::PRODUCES_TOKENS_ONLY
                | MetricFlags::LARGER_IS_BETTER
                | MetricFlags::SUPPORTS_AUDIO_ONLY,
            []
        ),
        spec!(
            UsageAcceptedPredictionTokens,
            "Usage Accepted Prediction Tokens",
            Token,
            Record,
            None,
            MetricFlags::PRODUCES_TOKENS_ONLY | MetricFlags::LARGER_IS_BETTER,
            []
        ),
        spec!(
            UsageRejectedPredictionTokens,
            "Usage Rejected Prediction Tokens",
            Token,
            Record,
            None,
            MetricFlags::PRODUCES_TOKENS_ONLY,
            []
        ),
        spec!(
            UsagePromptCacheReadTokens,
            "Usage Prompt Cache Read Tokens",
            Token,
            Record,
            None,
            MetricFlags::LARGER_IS_BETTER,
            []
        ),
        spec!(
            UsagePromptCacheWriteTokens,
            "Usage Prompt Cache Write Tokens",
            Token,
            Record,
            None,
            MetricFlags::NONE,
            []
        ),
        spec!(
            UsagePromptCacheMissTokens,
            "Usage Prompt Cache Miss Tokens",
            Token,
            Record,
            None,
            MetricFlags::NONE,
            []
        ),
        spec!(
            UsageToolUsePromptTokens,
            "Usage Tool Use Prompt Tokens",
            Token,
            Record,
            None,
            MetricFlags::NONE,
            []
        ),
        spec!(
            UsagePromptAudioSeconds,
            "Usage Prompt Audio Seconds",
            Second,
            Record,
            None,
            MetricFlags::LARGER_IS_BETTER | MetricFlags::SUPPORTS_AUDIO_ONLY,
            []
        ),
        spec!(
            TotalUsagePromptTokens,
            "Total Usage Prompt Tokens",
            Token,
            Derived,
            None,
            MetricFlags::TOKENIZES_INPUT_ONLY | MetricFlags::LARGER_IS_BETTER,
            [UsagePromptTokens]
        ),
        spec!(
            TotalUsageCompletionTokens,
            "Total Usage Completion Tokens",
            Token,
            Derived,
            None,
            MetricFlags::PRODUCES_TOKENS_ONLY | MetricFlags::LARGER_IS_BETTER,
            [UsageCompletionTokens]
        ),
        spec!(
            TotalUsageTotalTokens,
            "Total Usage Total Tokens",
            Token,
            Derived,
            None,
            MetricFlags::PRODUCES_TOKENS_ONLY | MetricFlags::LARGER_IS_BETTER,
            [UsageTotalTokens]
        ),
        spec!(
            TotalUsageReasoningTokens,
            "Total Usage Reasoning Tokens",
            Token,
            Derived,
            None,
            MetricFlags::PRODUCES_TOKENS_ONLY | MetricFlags::LARGER_IS_BETTER,
            [UsageReasoningTokens]
        ),
        spec!(
            TotalUsagePromptCacheReadTokens,
            "Total Usage Prompt Cache Read Tokens",
            Token,
            Derived,
            None,
            MetricFlags::LARGER_IS_BETTER,
            [UsagePromptCacheReadTokens]
        ),
        spec!(
            TotalUsagePromptCacheWriteTokens,
            "Total Usage Prompt Cache Write Tokens",
            Token,
            Derived,
            None,
            MetricFlags::NONE,
            [UsagePromptCacheWriteTokens]
        ),
        spec!(
            TotalUsagePromptCacheMissTokens,
            "Total Usage Prompt Cache Miss Tokens",
            Token,
            Derived,
            None,
            MetricFlags::NONE,
            [UsagePromptCacheMissTokens]
        ),
        spec!(
            TotalUsagePromptAudioTokens,
            "Total Usage Prompt Audio Tokens",
            Token,
            Derived,
            None,
            MetricFlags::LARGER_IS_BETTER | MetricFlags::SUPPORTS_AUDIO_ONLY,
            [UsagePromptAudioTokens]
        ),
        spec!(
            TotalUsageCompletionAudioTokens,
            "Total Usage Completion Audio Tokens",
            Token,
            Derived,
            None,
            MetricFlags::PRODUCES_TOKENS_ONLY
                | MetricFlags::LARGER_IS_BETTER
                | MetricFlags::SUPPORTS_AUDIO_ONLY,
            [UsageCompletionAudioTokens]
        ),
        spec!(
            TotalUsageAcceptedPredictionTokens,
            "Total Usage Accepted Prediction Tokens",
            Token,
            Derived,
            None,
            MetricFlags::PRODUCES_TOKENS_ONLY | MetricFlags::LARGER_IS_BETTER,
            [UsageAcceptedPredictionTokens]
        ),
        spec!(
            TotalUsageRejectedPredictionTokens,
            "Total Usage Rejected Prediction Tokens",
            Token,
            Derived,
            None,
            MetricFlags::PRODUCES_TOKENS_ONLY,
            [UsageRejectedPredictionTokens]
        ),
        spec!(
            TotalUsageToolUsePromptTokens,
            "Total Usage Tool Use Prompt Tokens",
            Token,
            Derived,
            None,
            MetricFlags::NONE,
            [UsageToolUsePromptTokens]
        ),
        spec!(
            TotalUsagePromptAudioSeconds,
            "Total Usage Prompt Audio Seconds",
            Second,
            Derived,
            None,
            MetricFlags::LARGER_IS_BETTER | MetricFlags::SUPPORTS_AUDIO_ONLY,
            [UsagePromptAudioSeconds]
        ),
        spec!(
            OverallUsagePromptCacheReadPct,
            "Overall Usage Prompt Cache Read %",
            Percent,
            Derived,
            None,
            MetricFlags::LARGER_IS_BETTER,
            [TotalUsagePromptCacheReadTokens, TotalUsagePromptTokens]
        ),
        spec!(
            UsagePromptTokensDiffPct,
            "Usage Prompt Diff",
            Percent,
            Record,
            None,
            MetricFlags::USAGE_DIFF_ONLY | MetricFlags::TOKENIZES_INPUT_ONLY,
            [InputSequenceLength, UsagePromptTokens]
        ),
        spec!(
            UsageCompletionTokensDiffPct,
            "Usage Completion Diff",
            Percent,
            Record,
            None,
            MetricFlags::USAGE_DIFF_ONLY | MetricFlags::PRODUCES_TOKENS_ONLY,
            [OutputSequenceLength, UsageCompletionTokens]
        ),
        spec!(
            UsageReasoningTokensDiffPct,
            "Usage Reasoning Diff",
            Percent,
            Record,
            None,
            MetricFlags::USAGE_DIFF_ONLY
                | MetricFlags::PRODUCES_TOKENS_ONLY
                | MetricFlags::SUPPORTS_REASONING,
            [ReasoningTokenCount, UsageReasoningTokens]
        ),
        spec!(
            UsageDiscrepancyCount,
            "Usage Discrepancy Count",
            Request,
            Aggregate,
            Some(AggregationKind::Sum),
            MetricFlags::USAGE_DIFF_ONLY | MetricFlags::NO_INDIVIDUAL_RECORDS,
            [UsagePromptTokensDiffPct, UsageCompletionTokensDiffPct]
        ),
        spec!(
            RequestedOutputSequenceLength,
            "Requested OSL",
            Token,
            Record,
            None,
            MetricFlags::INTERNAL | MetricFlags::PRODUCES_TOKENS_ONLY,
            []
        ),
        spec!(
            OslMismatchDiffPct,
            "OSL Mismatch Diff",
            Percent,
            Record,
            None,
            MetricFlags::PRODUCES_TOKENS_ONLY,
            [RequestedOutputSequenceLength, OutputSequenceLength]
        ),
        spec!(
            OslMismatchCount,
            "OSL Mismatch Count",
            Request,
            Aggregate,
            Some(AggregationKind::Sum),
            MetricFlags::PRODUCES_TOKENS_ONLY | MetricFlags::NO_INDIVIDUAL_RECORDS,
            [
                RequestedOutputSequenceLength,
                OutputSequenceLength,
                OslMismatchDiffPct
            ]
        ),
        spec!(
            ThinkingEfficiency,
            "Thinking Efficiency",
            Ratio,
            Record,
            None,
            MetricFlags::EXPERIMENTAL
                | MetricFlags::PRODUCES_TOKENS_ONLY
                | MetricFlags::SUPPORTS_REASONING,
            [ReasoningTokenCount, OutputTokenCount]
        ),
        spec!(
            OverallThinkingEfficiency,
            "Overall Thinking Efficiency",
            Ratio,
            Derived,
            None,
            MetricFlags::EXPERIMENTAL
                | MetricFlags::PRODUCES_TOKENS_ONLY
                | MetricFlags::SUPPORTS_REASONING,
            [TotalReasoningTokens, TotalOutputTokens]
        ),
        spec!(
            TotalGpuPower,
            "Total GPU Power",
            Watt,
            Derived,
            None,
            MetricFlags::NONE,
            []
        ),
        spec!(
            TotalGpuEnergy,
            "Total GPU Energy",
            Joule,
            Derived,
            None,
            MetricFlags::NONE,
            []
        ),
        spec!(
            OutputTokensPerJoule,
            "Output Tokens per Joule",
            TokensPerJoule,
            Derived,
            None,
            MetricFlags::PRODUCES_TOKENS_ONLY | MetricFlags::LARGER_IS_BETTER,
            []
        ),
        spec!(
            EnergyPerUser,
            "Energy per User",
            JoulesPerUser,
            Derived,
            None,
            MetricFlags::NONE,
            []
        ),
        spec!(
            NetworkAdjustedRequestLatency,
            "Network-Adjusted Request Latency",
            Nanosecond,
            Derived,
            None,
            MetricFlags::NONE,
            []
        ),
        spec!(
            NetworkAdjustedTimeToFirstToken,
            "Network-Adjusted Time to First Token",
            Nanosecond,
            Derived,
            None,
            MetricFlags::STREAMING_TOKENS_ONLY,
            []
        ),
        spec!(
            NetworkAdjustedTimeToFirstOutputToken,
            "Network-Adjusted Time to First Output Token",
            Nanosecond,
            Derived,
            None,
            MetricFlags::STREAMING_TOKENS_ONLY | MetricFlags::SUPPORTS_REASONING,
            []
        ),
        spec!(
            NetworkRtt,
            "Network RTT",
            Nanosecond,
            Derived,
            None,
            MetricFlags::NO_INDIVIDUAL_RECORDS,
            []
        ),
        spec!(
            StreamSetupLatency,
            "Stream Setup Latency",
            Nanosecond,
            Record,
            None,
            MetricFlags::STREAMING_ONLY | MetricFlags::EXPERIMENTAL,
            []
        ),
        spec!(
            StreamPrefillLatency,
            "Stream Prefill Latency",
            Nanosecond,
            Record,
            None,
            MetricFlags::STREAMING_TOKENS_ONLY | MetricFlags::EXPERIMENTAL,
            [TimeToFirstToken, StreamSetupLatency]
        ),
        spec!(
            AccuracyCorrect,
            "Accuracy Correct",
            Ratio,
            Aggregate,
            Some(AggregationKind::Sum),
            MetricFlags::INTERNAL,
            []
        ),
        spec!(
            AccuracyUnparsed,
            "Accuracy Unparsed",
            Ratio,
            Aggregate,
            Some(AggregationKind::Sum),
            MetricFlags::INTERNAL,
            []
        ),
        spec!(
            AudioDuration,
            "Audio Duration",
            Second,
            Record,
            None,
            MetricFlags::SUPPORTS_AUDIO_ONLY,
            []
        ),
        spec!(
            NumImages,
            "Number of Images",
            Image,
            Record,
            None,
            MetricFlags::SUPPORTS_IMAGE_ONLY,
            []
        ),
        spec!(
            ImageThroughput,
            "Image Throughput",
            ImagesPerSecond,
            Record,
            None,
            MetricFlags::SUPPORTS_IMAGE_ONLY,
            [NumImages, RequestLatency]
        ),
        spec!(
            ImageLatency,
            "Image Latency",
            MillisecondsPerImage,
            Record,
            None,
            MetricFlags::SUPPORTS_IMAGE_ONLY,
            [NumImages, RequestLatency]
        ),
        spec!(
            TotalNumImages,
            "Total Number of Images",
            Image,
            Derived,
            None,
            MetricFlags::SUPPORTS_IMAGE_ONLY | MetricFlags::LARGER_IS_BETTER,
            [NumImages]
        ),
        spec!(
            ImageSamplesPerSecond,
            "Image Samples Per Second",
            ImagesPerSecond,
            Derived,
            None,
            MetricFlags::SUPPORTS_IMAGE_ONLY | MetricFlags::LARGER_IS_BETTER,
            [TotalNumImages, BenchmarkDuration]
        ),
        spec!(
            VideoInferenceTime,
            "Video Inference Time",
            Second,
            Record,
            None,
            MetricFlags::PRODUCES_VIDEO_ONLY,
            []
        ),
        spec!(
            VideoPeakMemory,
            "Video Peak Memory",
            Megabyte,
            Record,
            None,
            MetricFlags::PRODUCES_VIDEO_ONLY,
            []
        ),
        spec!(
            HttpReqBlocked,
            "HTTP Blocked",
            Nanosecond,
            Record,
            None,
            MetricFlags::HTTP_TRACE_ONLY,
            []
        ),
        spec!(
            HttpReqDnsLookup,
            "HTTP DNS Lookup",
            Nanosecond,
            Record,
            None,
            MetricFlags::HTTP_TRACE_ONLY,
            []
        ),
        spec!(
            HttpReqConnecting,
            "HTTP Connecting",
            Nanosecond,
            Record,
            None,
            MetricFlags::HTTP_TRACE_ONLY,
            []
        ),
        spec!(
            HttpReqSending,
            "HTTP Sending",
            Nanosecond,
            Record,
            None,
            MetricFlags::HTTP_TRACE_ONLY,
            []
        ),
        spec!(
            HttpReqWaiting,
            "HTTP Waiting (TTFB)",
            Nanosecond,
            Record,
            None,
            MetricFlags::HTTP_TRACE_ONLY,
            []
        ),
        spec!(
            HttpReqReceiving,
            "HTTP Receiving",
            Nanosecond,
            Record,
            None,
            MetricFlags::HTTP_TRACE_ONLY,
            []
        ),
        spec!(
            HttpReqDuration,
            "HTTP Duration (excl. conn)",
            Nanosecond,
            Record,
            None,
            MetricFlags::HTTP_TRACE_ONLY,
            []
        ),
        spec!(
            HttpReqConnectionReused,
            "HTTP Connection Reused",
            Ratio,
            Record,
            None,
            MetricFlags::HTTP_TRACE_ONLY,
            []
        ),
        spec!(
            HttpReqDataSent,
            "HTTP Data Sent",
            Byte,
            Record,
            None,
            MetricFlags::HTTP_TRACE_ONLY,
            []
        ),
        spec!(
            HttpReqDataReceived,
            "HTTP Data Received",
            Byte,
            Record,
            None,
            MetricFlags::HTTP_TRACE_ONLY,
            []
        ),
        spec!(
            HttpReqChunksSent,
            "HTTP Chunks Sent",
            Count,
            Record,
            None,
            MetricFlags::HTTP_TRACE_ONLY,
            []
        ),
        spec!(
            HttpReqChunksReceived,
            "HTTP Chunks Received",
            Count,
            Record,
            None,
            MetricFlags::HTTP_TRACE_ONLY,
            []
        ),
        spec!(
            HttpReqConnectionOverhead,
            "HTTP Connection Overhead",
            Nanosecond,
            Record,
            None,
            MetricFlags::HTTP_TRACE_ONLY,
            [HttpReqBlocked, HttpReqDnsLookup, HttpReqConnecting]
        ),
        spec!(
            HttpReqTotal,
            "HTTP Total Time",
            Nanosecond,
            Record,
            None,
            MetricFlags::HTTP_TRACE_ONLY,
            [
                HttpReqBlocked,
                HttpReqDnsLookup,
                HttpReqConnecting,
                HttpReqSending,
                HttpReqWaiting,
                HttpReqReceiving
            ]
        ),
        spec!(
            EffectiveLatency,
            "Effective Latency",
            Millisecond,
            Record,
            None,
            MetricFlags::INTERNAL,
            [RequestLatency]
        ),
        spec!(
            CreditToStartLatency,
            "Credit To Start",
            Millisecond,
            Record,
            None,
            MetricFlags::INTERNAL,
            []
        ),
        spec!(
            EffectiveConcurrency,
            "Effective Concurrency",
            Request,
            Derived,
            None,
            MetricFlags::NO_INDIVIDUAL_RECORDS,
            []
        ),
        spec!(
            EffectiveDecodeThroughput,
            "Effective Decode Throughput",
            TokensPerSecond,
            Derived,
            None,
            MetricFlags::NO_INDIVIDUAL_RECORDS | MetricFlags::LARGER_IS_BETTER,
            []
        ),
        spec!(
            EffectivePrefillThroughput,
            "Effective Prefill Throughput",
            TokensPerSecond,
            Derived,
            None,
            MetricFlags::NO_INDIVIDUAL_RECORDS | MetricFlags::LARGER_IS_BETTER,
            []
        ),
        spec!(
            EffectiveDecodeConcurrency,
            "Effective Decode Concurrency",
            Request,
            Derived,
            None,
            MetricFlags::NO_INDIVIDUAL_RECORDS,
            []
        ),
        spec!(
            EffectivePrefillConcurrency,
            "Effective Prefill Concurrency",
            Request,
            Derived,
            None,
            MetricFlags::NO_INDIVIDUAL_RECORDS,
            []
        ),
        spec!(
            EffectiveTotalThroughput,
            "Effective Total Throughput",
            TokensPerSecond,
            Derived,
            None,
            MetricFlags::NO_INDIVIDUAL_RECORDS | MetricFlags::LARGER_IS_BETTER,
            []
        ),
        spec!(
            EffectiveDecodeThroughputPerUser,
            "Effective Decode Throughput Per User",
            TokensPerSecondPerUser,
            Derived,
            None,
            MetricFlags::NO_INDIVIDUAL_RECORDS | MetricFlags::LARGER_IS_BETTER,
            []
        ),
        spec!(
            EffectivePrefillThroughputPerUser,
            "Effective Prefill Throughput Per User",
            TokensPerSecondPerUser,
            Derived,
            None,
            MetricFlags::NO_INDIVIDUAL_RECORDS | MetricFlags::LARGER_IS_BETTER,
            []
        ),
        spec!(
            EffectiveImageSamplesPerSecond,
            "Effective Image Samples Per Second",
            ImagesPerSecond,
            Derived,
            None,
            MetricFlags::NO_INDIVIDUAL_RECORDS
                | MetricFlags::SUPPORTS_IMAGE_ONLY
                | MetricFlags::LARGER_IS_BETTER,
            []
        ),
        spec!(
            TokensInFlight,
            "Tokens In Flight",
            Token,
            Derived,
            None,
            MetricFlags::NO_INDIVIDUAL_RECORDS,
            []
        ),
        spec!(
            ActiveDecodeThroughput,
            "Active Decode Throughput",
            TokensPerSecond,
            Derived,
            None,
            MetricFlags::NO_INDIVIDUAL_RECORDS | MetricFlags::LARGER_IS_BETTER,
            []
        ),
        spec!(
            ActivePrefillThroughput,
            "Active Prefill Throughput",
            TokensPerSecond,
            Derived,
            None,
            MetricFlags::NO_INDIVIDUAL_RECORDS | MetricFlags::LARGER_IS_BETTER,
            []
        ),
        spec!(
            ActiveDecodeThroughputPerUser,
            "Active Decode Throughput Per User",
            TokensPerSecondPerUser,
            Derived,
            None,
            MetricFlags::NO_INDIVIDUAL_RECORDS | MetricFlags::LARGER_IS_BETTER,
            []
        ),
        spec!(
            ActivePrefillThroughputPerUser,
            "Active Prefill Throughput Per User",
            TokensPerSecondPerUser,
            Derived,
            None,
            MetricFlags::NO_INDIVIDUAL_RECORDS | MetricFlags::LARGER_IS_BETTER,
            []
        ),
        spec!(
            ActiveImageSamplesPerSecond,
            "Active Image Samples Per Second",
            ImagesPerSecond,
            Derived,
            None,
            MetricFlags::NO_INDIVIDUAL_RECORDS
                | MetricFlags::SUPPORTS_IMAGE_ONLY
                | MetricFlags::LARGER_IS_BETTER,
            []
        ),
        spec!(
            EffectiveImageSamplesPerSecondPerUser,
            "Effective Image Samples Per Second Per User",
            ImagesPerSecondPerUser,
            Derived,
            None,
            MetricFlags::NO_INDIVIDUAL_RECORDS
                | MetricFlags::SUPPORTS_IMAGE_ONLY
                | MetricFlags::LARGER_IS_BETTER,
            []
        ),
        spec!(
            ActiveTotalThroughput,
            "Active Total Throughput",
            TokensPerSecond,
            Derived,
            None,
            MetricFlags::NO_INDIVIDUAL_RECORDS | MetricFlags::LARGER_IS_BETTER,
            []
        ),
    ];
    configure_catalog_metadata(&mut catalog);
    catalog
});

fn configure_catalog_metadata(catalog: &mut [MetricSpec]) {
    use MetricConsoleGroup::{Active, Effective, None as Hidden, Usage};
    use MetricTag::*;

    for spec in catalog {
        spec.short_header = match spec.tag {
            AudioDuration => Some("Audio Dur"),
            BenchmarkDuration => Some("Duration"),
            CompletedRequestCount => Some("Completed"),
            CreditDropLatency => Some("Credit Latency"),
            DecodeDuration => Some("Decode Duration"),
            E2eOutputTokenThroughput => Some("E2E Output TPS/User"),
            ErrorInputSequenceLength => Some("Error ISL"),
            ErrorRequestCount => Some("Error Count"),
            GoodRequestFraction => Some("GoodReqFrac"),
            Goodput => Some("Goodput"),
            HttpReqBlocked => Some("Blocked"),
            HttpReqChunksReceived => Some("Chunks Recv"),
            HttpReqChunksSent => Some("Chunks Sent"),
            HttpReqConnecting => Some("Connecting"),
            HttpReqConnectionOverhead => Some("Conn Overhead"),
            HttpReqConnectionReused => Some("Conn Reused"),
            HttpReqDataReceived => Some("Received"),
            HttpReqDataSent => Some("Sent"),
            HttpReqDnsLookup => Some("DNS"),
            HttpReqDuration => Some("Dur (excl)"),
            HttpReqReceiving => Some("Receiving"),
            HttpReqSending => Some("Sending"),
            HttpReqTotal => Some("Total"),
            HttpReqWaiting => Some("TTFB"),
            ImageLatency => Some("Image Latency"),
            ImageThroughput => Some("Image Throughput"),
            InputSequenceLength => Some("ISL"),
            InputTokenThroughput => Some("Input TPS"),
            InterChunkLatency => Some("ICL"),
            InterTokenLatency => Some("ITL"),
            MaxResponseTimestamp => Some("Max Resp"),
            MinRequestTimestamp => Some("Min Req"),
            NetworkAdjustedRequestLatency => Some("Net-Adj Req Latency"),
            NetworkAdjustedTimeToFirstOutputToken => Some("Net-Adj TTFO"),
            NetworkAdjustedTimeToFirstToken => Some("Net-Adj TTFT"),
            NetworkRtt => Some("Net RTT"),
            NumImages => Some("Num Images"),
            OslMismatchCount => Some("OSL Mismatches"),
            OslMismatchDiffPct => Some("OSL Diff"),
            OutputSequenceLength => Some("OSL"),
            OutputTokenCount => Some("Output Tokens"),
            OutputTokenThroughput => Some("Output TPS"),
            OutputTokenThroughputPerUser => Some("Output TPS/User"),
            OverallThinkingEfficiency => Some("Overall Eff."),
            OverallUsagePromptCacheReadPct => Some("Overall Cache Read %"),
            PrefillThroughputPerUser => Some("Prefill TPS/User"),
            ReasoningTokenCount => Some("Reasoning Tokens"),
            RequestCount => Some("Requests"),
            RequestErrorRate => Some("Err %"),
            RequestLatency => Some("Req Latency"),
            RequestThroughput => Some("Req/sec"),
            RequestedOutputSequenceLength => Some("Req OSL"),
            Rtfx => Some("RTFx"),
            TimeToFirstOutputToken => Some("TTFO"),
            TimeToFirstToken => Some("TTFT"),
            TimeToSecondToken => Some("TTST"),
            TotalErrorInputSequenceLength => Some("Total Error ISL"),
            TotalInputSequenceLength => Some("Total ISL"),
            TotalOutputSequenceLength => Some("Total OSL"),
            TotalOutputTokens => Some("Total Output"),
            TotalReasoningTokens => Some("Total Reasoning"),
            TotalTokenThroughput => Some("Total TPS"),
            TotalUsageAcceptedPredictionTokens => Some("Total Usage Accepted Pred"),
            TotalUsageCompletionAudioTokens => Some("Total Usage Comp Audio"),
            TotalUsageCompletionTokens => Some("Total Usage Completion"),
            TotalUsagePromptAudioSeconds => Some("Total Usage Prompt Audio Sec"),
            TotalUsagePromptAudioTokens => Some("Total Usage Prompt Audio"),
            TotalUsagePromptCacheMissTokens => Some("Total Usage Prompt Cache Miss"),
            TotalUsagePromptCacheReadTokens => Some("Total Usage Prompt Cache Read"),
            TotalUsagePromptCacheWriteTokens => Some("Total Usage Prompt Cache Write"),
            TotalUsagePromptTokens => Some("Total Usage Prompt"),
            TotalUsageReasoningTokens => Some("Total Usage Reasoning"),
            TotalUsageRejectedPredictionTokens => Some("Total Usage Rejected Pred"),
            TotalUsageToolUsePromptTokens => Some("Total Usage Tool Prompt"),
            TotalUsageTotalTokens => Some("Total Usage Total"),
            UsageAcceptedPredictionTokens => Some("Usage Accepted Pred"),
            UsageCompletionAudioTokens => Some("Usage Completion Audio"),
            UsageCompletionTokens => Some("Usage Completion"),
            UsageCompletionTokensDiffPct => Some("Completion Diff"),
            UsageDiscrepancyCount => Some("Discrepancies"),
            UsagePromptAudioSeconds => Some("Usage Prompt Audio Sec"),
            UsagePromptAudioTokens => Some("Usage Prompt Audio"),
            UsagePromptCacheMissTokens => Some("Usage Prompt Cache Miss"),
            UsagePromptCacheReadTokens => Some("Usage Prompt Cache Read"),
            UsagePromptCacheWriteTokens => Some("Usage Prompt Cache Write"),
            UsagePromptTokens => Some("Usage Prompt"),
            UsagePromptTokensDiffPct => Some("Prompt Diff"),
            UsageReasoningTokens => Some("Usage Reasoning"),
            UsageReasoningTokensDiffPct => Some("Reasoning Diff"),
            UsageRejectedPredictionTokens => Some("Usage Rejected Pred"),
            UsageToolUsePromptTokens => Some("Usage Tool Prompt"),
            UsageTotalTokens => Some("Usage Total"),
            VideoInferenceTime => Some("Inference Time"),
            VideoPeakMemory => Some("Peak Memory"),
            _ => None,
        };
        spec.short_header_hide_unit = matches!(
            spec.tag,
            BenchmarkDuration
                | CompletedRequestCount
                | E2eOutputTokenThroughput
                | ErrorRequestCount
                | GoodRequestCount
                | GoodRequestFraction
                | Goodput
                | InputTokenThroughput
                | MaxResponseTimestamp
                | MinRequestTimestamp
                | OslMismatchCount
                | OslMismatchDiffPct
                | OutputTokenCount
                | OutputTokenThroughput
                | OutputTokenThroughputPerUser
                | OverallThinkingEfficiency
                | OverallUsagePromptCacheReadPct
                | PrefillThroughputPerUser
                | ReasoningTokenCount
                | RequestCount
                | RequestErrorRate
                | RequestThroughput
                | Rtfx
                | ThinkingEfficiency
                | TotalErrorInputSequenceLength
                | TotalInputSequenceLength
                | TotalOutputSequenceLength
                | TotalOutputTokens
                | TotalReasoningTokens
                | TotalTokenThroughput
                | TotalUsageAcceptedPredictionTokens
                | TotalUsageCompletionAudioTokens
                | TotalUsageCompletionTokens
                | TotalUsagePromptAudioTokens
                | TotalUsagePromptCacheMissTokens
                | TotalUsagePromptCacheReadTokens
                | TotalUsagePromptCacheWriteTokens
                | TotalUsagePromptTokens
                | TotalUsageReasoningTokens
                | TotalUsageRejectedPredictionTokens
                | TotalUsageToolUsePromptTokens
                | TotalUsageTotalTokens
                | UsageAcceptedPredictionTokens
                | UsageCompletionAudioTokens
                | UsageCompletionTokens
                | UsageCompletionTokensDiffPct
                | UsageDiscrepancyCount
                | UsagePromptAudioTokens
                | UsagePromptCacheMissTokens
                | UsagePromptCacheReadTokens
                | UsagePromptCacheWriteTokens
                | UsagePromptTokens
                | UsagePromptTokensDiffPct
                | UsageReasoningTokens
                | UsageReasoningTokensDiffPct
                | UsageRejectedPredictionTokens
                | UsageToolUsePromptTokens
                | UsageTotalTokens
        );
        spec.display_unit = match spec.tag {
            BenchmarkDuration => Some(Unit::Second),
            RequestLatency
            | TimeToFirstToken
            | TimeToSecondToken
            | TimeToFirstOutputToken
            | InterTokenLatency
            | InterChunkLatency
            | DecodeDuration
            | CreditDropLatency
            | StreamSetupLatency
            | StreamPrefillLatency
            | HttpReqBlocked
            | HttpReqDnsLookup
            | HttpReqConnecting
            | HttpReqSending
            | HttpReqWaiting
            | HttpReqReceiving
            | HttpReqDuration
            | HttpReqConnectionOverhead
            | HttpReqTotal
            | NetworkAdjustedRequestLatency
            | NetworkAdjustedTimeToFirstToken
            | NetworkAdjustedTimeToFirstOutputToken
            | NetworkRtt
            | VideoInferenceTime => Some(Unit::Millisecond),
            HttpReqDataSent | HttpReqDataReceived => Some(Unit::Kilobyte),
            _ => None,
        };
        spec.display_order = match spec.tag {
            AudioDuration => Some(870),
            CompletedRequestCount => Some(1075),
            DecodeDuration => Some(350),
            E2eOutputTokenThroughput => Some(510),
            EnergyPerUser => Some(903),
            ErrorInputSequenceLength => Some(700),
            Goodput => Some(1000),
            HttpReqBlocked => Some(2000),
            HttpReqChunksReceived => Some(2100),
            HttpReqChunksSent => Some(2080),
            HttpReqConnecting => Some(2020),
            HttpReqConnectionOverhead => Some(2110),
            HttpReqConnectionReused => Some(2060),
            HttpReqDataReceived => Some(2090),
            HttpReqDataSent => Some(2070),
            HttpReqDnsLookup => Some(2010),
            HttpReqDuration => Some(2120),
            HttpReqReceiving => Some(2050),
            HttpReqSending => Some(2030),
            HttpReqTotal => Some(2130),
            HttpReqWaiting => Some(2040),
            ImageLatency => Some(861),
            ImageThroughput => Some(860),
            InputSequenceLength => Some(700),
            InputTokenThroughput => Some(805),
            InterTokenLatency => Some(400),
            NetworkAdjustedRequestLatency => Some(301),
            NetworkAdjustedTimeToFirstOutputToken => Some(211),
            NetworkAdjustedTimeToFirstToken => Some(101),
            NetworkRtt => Some(305),
            OutputSequenceLength => Some(600),
            OutputTokenThroughput => Some(800),
            OutputTokenThroughputPerUser => Some(500),
            OutputTokensPerJoule => Some(902),
            OverallUsagePromptCacheReadPct => Some(2012),
            RequestCount => Some(1100),
            RequestErrorRate => Some(1080),
            RequestLatency => Some(300),
            RequestThroughput => Some(900),
            Rtfx => Some(850),
            TimeToFirstOutputToken => Some(210),
            TimeToFirstToken => Some(100),
            TimeToSecondToken => Some(200),
            TotalGpuEnergy => Some(901),
            TotalGpuPower => Some(900),
            TotalUsageAcceptedPredictionTokens => Some(2130),
            TotalUsageCompletionAudioTokens => Some(2120),
            TotalUsageCompletionTokens => Some(2100),
            TotalUsagePromptAudioSeconds => Some(2040),
            TotalUsagePromptAudioTokens => Some(2020),
            TotalUsagePromptCacheMissTokens => Some(2017),
            TotalUsagePromptCacheReadTokens => Some(2010),
            TotalUsagePromptCacheWriteTokens => Some(2015),
            TotalUsagePromptTokens => Some(2000),
            TotalUsageReasoningTokens => Some(2110),
            TotalUsageRejectedPredictionTokens => Some(2140),
            TotalUsageToolUsePromptTokens => Some(2030),
            TotalUsageTotalTokens => Some(2200),
            UsageAcceptedPredictionTokens => Some(1130),
            UsageCompletionAudioTokens => Some(1120),
            UsageCompletionTokens => Some(1100),
            UsagePromptAudioSeconds => Some(1040),
            UsagePromptAudioTokens => Some(1020),
            UsagePromptCacheMissTokens => Some(1017),
            UsagePromptCacheReadTokens => Some(1010),
            UsagePromptCacheWriteTokens => Some(1015),
            UsagePromptTokens => Some(1000),
            UsageReasoningTokens => Some(1110),
            UsageRejectedPredictionTokens => Some(1140),
            UsageToolUsePromptTokens => Some(1030),
            UsageTotalTokens => Some(1200),
            VideoInferenceTime => Some(310),
            VideoPeakMemory => Some(311),
            _ => None,
        };
        spec.console_group = match spec.tag {
            UsagePromptTokens
            | UsageCompletionTokens
            | UsageTotalTokens
            | UsageReasoningTokens
            | UsagePromptAudioTokens
            | UsageCompletionAudioTokens
            | UsageAcceptedPredictionTokens
            | UsageRejectedPredictionTokens
            | UsagePromptCacheReadTokens
            | UsagePromptCacheWriteTokens
            | UsagePromptCacheMissTokens
            | UsageToolUsePromptTokens
            | UsagePromptAudioSeconds
            | TotalUsagePromptTokens
            | TotalUsageCompletionTokens
            | TotalUsageTotalTokens
            | TotalUsageReasoningTokens
            | TotalUsagePromptAudioTokens
            | TotalUsageCompletionAudioTokens
            | TotalUsageAcceptedPredictionTokens
            | TotalUsageRejectedPredictionTokens
            | TotalUsagePromptCacheReadTokens
            | TotalUsagePromptCacheWriteTokens
            | TotalUsagePromptCacheMissTokens
            | TotalUsageToolUsePromptTokens
            | TotalUsagePromptAudioSeconds
            | OverallUsagePromptCacheReadPct => Usage,
            EffectiveLatency
            | EffectiveConcurrency
            | EffectiveDecodeThroughput
            | EffectivePrefillThroughput
            | EffectiveDecodeConcurrency
            | EffectivePrefillConcurrency
            | EffectiveTotalThroughput
            | EffectiveDecodeThroughputPerUser
            | EffectivePrefillThroughputPerUser
            | TokensInFlight => Effective,
            ActiveDecodeThroughput
            | ActivePrefillThroughput
            | ActiveDecodeThroughputPerUser
            | ActivePrefillThroughputPerUser
            | ActiveTotalThroughput => Active,
            GoodRequestCount
            | GoodRequestFraction
            | MinRequestTimestamp
            | MaxResponseTimestamp
            | BenchmarkDuration
            | InterChunkLatency
            | CreditDropLatency
            | OutputTokenCount
            | ReasoningTokenCount
            | ErrorInputSequenceLength
            | TotalOutputSequenceLength
            | TotalInputSequenceLength
            | TotalErrorInputSequenceLength
            | TotalOutputTokens
            | TotalReasoningTokens
            | TotalTokenThroughput
            | PrefillThroughputPerUser
            | UsagePromptTokensDiffPct
            | UsageCompletionTokensDiffPct
            | UsageReasoningTokensDiffPct
            | UsageDiscrepancyCount
            | RequestedOutputSequenceLength
            | OslMismatchDiffPct
            | OslMismatchCount
            | CreditToStartLatency
            | AccuracyCorrect
            | AccuracyUnparsed
            | AudioDuration
            | NumImages
            | HttpReqBlocked
            | HttpReqDnsLookup
            | HttpReqConnecting
            | HttpReqSending
            | HttpReqWaiting
            | HttpReqReceiving
            | HttpReqDuration
            | HttpReqConnectionReused
            | HttpReqDataSent
            | HttpReqDataReceived
            | HttpReqChunksSent
            | HttpReqChunksReceived
            | HttpReqConnectionOverhead
            | HttpReqTotal => Hidden,
            _ => MetricConsoleGroup::Default,
        };
        spec.value_type = match spec.tag {
            InterChunkLatency => MetricValueType::IntList,
            BenchmarkDuration
            | CompletedRequestCount
            | CreditDropLatency
            | ErrorInputSequenceLength
            | ErrorRequestCount
            | MinRequestTimestamp
            | MaxResponseTimestamp
            | RequestCount
            | RequestLatency
            | TimeToFirstToken
            | TimeToSecondToken
            | TimeToFirstOutputToken
            | OutputSequenceLength
            | InputSequenceLength
            | OutputTokenCount
            | ReasoningTokenCount
            | TotalOutputSequenceLength
            | TotalInputSequenceLength
            | TotalErrorInputSequenceLength
            | TotalOutputTokens
            | TotalReasoningTokens
            | RequestedOutputSequenceLength
            | OslMismatchCount
            | UsageDiscrepancyCount
            | UsagePromptTokens
            | UsageCompletionTokens
            | UsageTotalTokens
            | UsageReasoningTokens
            | UsagePromptAudioTokens
            | UsageCompletionAudioTokens
            | UsageAcceptedPredictionTokens
            | UsageRejectedPredictionTokens
            | UsagePromptCacheReadTokens
            | UsagePromptCacheWriteTokens
            | UsagePromptCacheMissTokens
            | UsageToolUsePromptTokens
            | TotalUsagePromptTokens
            | TotalUsageCompletionTokens
            | TotalUsageTotalTokens
            | TotalUsageReasoningTokens
            | TotalUsagePromptAudioTokens
            | TotalUsageCompletionAudioTokens
            | TotalUsageAcceptedPredictionTokens
            | TotalUsageRejectedPredictionTokens
            | TotalUsagePromptCacheReadTokens
            | TotalUsagePromptCacheWriteTokens
            | TotalUsagePromptCacheMissTokens
            | TotalUsageToolUsePromptTokens
            | NetworkAdjustedRequestLatency
            | NetworkAdjustedTimeToFirstToken
            | NetworkAdjustedTimeToFirstOutputToken
            | StreamSetupLatency
            | StreamPrefillLatency
            | NumImages
            | HttpReqBlocked
            | HttpReqDnsLookup
            | HttpReqConnecting
            | HttpReqSending
            | HttpReqWaiting
            | HttpReqReceiving
            | HttpReqDuration
            | HttpReqConnectionReused
            | HttpReqDataSent
            | HttpReqDataReceived
            | HttpReqChunksSent
            | HttpReqChunksReceived
            | HttpReqConnectionOverhead
            | HttpReqTotal => MetricValueType::Int,
            _ => MetricValueType::Float,
        };
        spec.plot_direction = if spec.flags.contains(MetricFlags::LARGER_IS_BETTER) {
            PlotMetricDirection::LargerIsBetter
        } else {
            PlotMetricDirection::SmallerIsBetter
        };
    }
}

/// Looks up a catalog spec.
pub fn spec_for(tag: MetricTag) -> Option<&'static MetricSpec> {
    CATALOG.iter().find(|spec| spec.tag == tag)
}

/// One per-request metric exposed as a column in a per-record artifact (the
/// Parquet sidecar and the records CSV): the metric tag, its human display
/// header, and its constant display unit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecordMetricColumn {
    /// Metric tag — the machine key (Parquet column name, JSONL metric key).
    pub tag: String,
    /// Human display header (e.g. `Request Latency`), used by the records CSV.
    pub header: String,
    /// Constant display unit (e.g. `ms`, `tokens`, `tokens/sec`).
    pub unit: String,
}

impl RecordMetricColumn {
    /// The records-CSV column name: `{header} ({unit})`. The unit is omitted when it
    /// is empty or `count`/`requests` (case-insensitive), and the parenthesized
    /// unit stands alone when the header is empty.
    pub fn csv_display_name(&self) -> String {
        let lower = self.unit.to_ascii_lowercase();
        if self.unit.is_empty() || lower == "count" || lower == "requests" {
            self.header.clone()
        } else if self.header.is_empty() {
            format!("({})", self.unit)
        } else {
            format!("{} ({})", self.header, self.unit)
        }
    }
}

/// The ordered per-request metric columns: every [`MetricType::Record`] metric
/// that is not hidden from individual records, in catalog order.
///
/// The filter matches the per-record JSONL projection: `MetricType::Record` minus
/// `NO_INDIVIDUAL_RECORDS | INTERNAL | EXPERIMENTAL` — so the Parquet/CSV metric
/// columns line up exactly with the JSONL metric keys. Deriving from the static
/// catalog makes the column set deterministic across runs of the same binary.
/// The unit is the metric's display unit (`display_unit` falling back to `unit`).
pub fn record_metric_columns() -> Vec<RecordMetricColumn> {
    let hidden =
        MetricFlags::NO_INDIVIDUAL_RECORDS | MetricFlags::INTERNAL | MetricFlags::EXPERIMENTAL;
    CATALOG
        .iter()
        .filter(|spec| spec.kind == MetricType::Record && !spec.flags.intersects(hidden))
        .map(|spec| RecordMetricColumn {
            tag: spec.tag.as_str().to_string(),
            header: spec.header.to_string(),
            unit: spec.display_unit.unwrap_or(spec.unit).as_str().to_string(),
        })
        .collect()
}

/// Validates catalog uniqueness, dependency resolution, tier rules, and acyclicity.
pub fn validate_catalog() -> Result<Vec<MetricTag>, String> {
    let mut seen = BTreeSet::new();
    let by_tag: HashMap<MetricTag, &MetricSpec> = CATALOG
        .iter()
        .map(|spec| {
            if !seen.insert(spec.tag) {
                return Err(format!("duplicate metric tag {}", spec.tag));
            }
            Ok((spec.tag, spec))
        })
        .collect::<Result<_, _>>()?;

    let mut graph = DiGraphMap::<MetricTag, ()>::new();
    for spec in CATALOG.iter() {
        graph.add_node(spec.tag);
        for required in spec.required {
            let required_spec = by_tag
                .get(required)
                .ok_or_else(|| format!("{} requires missing {}", spec.tag, required))?;
            match spec.kind {
                MetricType::Record if required_spec.kind != MetricType::Record => {
                    return Err(format!(
                        "record metric {} cannot require {}",
                        spec.tag, required
                    ));
                }
                MetricType::Aggregate if required_spec.kind == MetricType::Derived => {
                    return Err(format!(
                        "aggregate metric {} cannot require derived {}",
                        spec.tag, required
                    ));
                }
                MetricType::Derived | MetricType::Aggregate | MetricType::Record => {}
            }
            graph.add_edge(*required, spec.tag, ());
        }
    }
    toposort(&graph, None).map_err(|cycle| format!("catalog cycle at {}", cycle.node_id()))
}

#[cfg(test)]
mod tests {
    use super::{CATALOG, MetricFlags, MetricTag, spec_for, validate_catalog};

    fn catalog_fingerprint() -> u64 {
        fn feed(hash: &mut u64, bytes: &[u8]) {
            for byte in bytes.iter().copied().chain([0xff]) {
                *hash ^= u64::from(byte);
                *hash = hash.wrapping_mul(0x100000001b3);
            }
        }

        let mut specs = CATALOG.iter().collect::<Vec<_>>();
        specs.sort_by_key(|spec| spec.tag.as_str());
        let mut hash = 0xcbf29ce484222325;
        for spec in specs {
            feed(&mut hash, spec.tag.as_str().as_bytes());
            feed(&mut hash, spec.header.as_bytes());
            feed(&mut hash, spec.short_header.unwrap_or("").as_bytes());
            feed(
                &mut hash,
                spec.short_header_hide_unit.to_string().as_bytes(),
            );
            feed(&mut hash, format!("{:?}", spec.unit).as_bytes());
            feed(&mut hash, format!("{:?}", spec.display_unit).as_bytes());
            feed(&mut hash, format!("{:?}", spec.display_order).as_bytes());
            feed(&mut hash, spec.flags.bits().to_string().as_bytes());
            feed(&mut hash, format!("{:?}", spec.console_group).as_bytes());
            feed(&mut hash, format!("{:?}", spec.plot_direction).as_bytes());
            for dependency in spec.required {
                feed(&mut hash, dependency.as_str().as_bytes());
            }
            feed(&mut hash, format!("{:?}", spec.value_type).as_bytes());
            feed(&mut hash, format!("{:?}", spec.kind).as_bytes());
            feed(&mut hash, format!("{:?}", spec.aggregation).as_bytes());
        }
        hash
    }

    #[test]
    fn metric_flags_preserve_reserved_gap_and_missing_semantics() {
        assert_eq!(MetricFlags::ERROR_ONLY.bits(), 1 << 1);
        assert_eq!(MetricFlags::PRODUCES_TOKENS_ONLY.bits(), 1 << 2);
        assert_eq!(MetricFlags::LARGER_IS_BETTER.bits(), 1 << 4);
        assert_eq!(MetricFlags::TOKENIZES_INPUT_ONLY.bits(), 1 << 12);
        assert_eq!(
            MetricFlags::STREAMING_TOKENS_ONLY.bits(),
            (1 << 0) | (1 << 2)
        );
        assert!(MetricFlags::NONE.missing(MetricFlags::NONE));
        assert!(MetricFlags::NONE.missing(MetricFlags::ERROR_ONLY));
        assert!(!MetricFlags::ERROR_ONLY.missing(MetricFlags::ERROR_ONLY));
        assert!(
            MetricFlags::STREAMING_TOKENS_ONLY
                .has_any(MetricFlags::STREAMING_TOKENS_ONLY | MetricFlags::GOODPUT)
        );
    }

    #[test]
    fn catalog_has_unique_acyclic_resolved_dependencies() {
        let order = validate_catalog().unwrap();
        assert_eq!(CATALOG.len(), 125);
        assert!(order.contains(&MetricTag::RequestLatency));
        assert!(
            CATALOG
                .iter()
                .any(|spec| spec.tag == MetricTag::GoodRequestFraction)
        );
    }

    #[test]
    fn catalog_identity_matches_the_source_grounded_snapshot() {
        // Covers the 104 Python identities plus the native sweep identities.
        // Any intentional metadata change must be re-audited before updating this value.
        // Updated 2026-07-20: added DecodeDuration (client-observed interval from
        // the first to final content response; ports Python's
        // decode_duration_metric.py), hand-porting origin/main's fe999132f.
        // Updated 2026-07-24: added TotalNumImages, ImageSamplesPerSecond (aggregate
        // image-sample rate), EffectiveImageSamplesPerSecond, ActiveImageSamplesPerSecond,
        // and EffectiveImageSamplesPerSecondPerUser (sweep-line effective, active, and
        // per-user variants; per-user divides by overall concurrency per design 0006).
        assert_eq!(catalog_fingerprint(), 7_024_082_593_996_193_480);
    }

    #[test]
    fn metricspec_accessors_delegate_to_def() {
        let s = spec_for(MetricTag::RequestLatency).unwrap();
        assert_eq!(s.header(), s.def.header);
        assert_eq!(s.unit(), s.def.unit);
    }
}
