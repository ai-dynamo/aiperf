// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Static metric catalog and dependency validation.
//!
//! The catalog is data over the columnar engine: each row declares identity,
//! units, flags, kind, aggregation, and true metric dependencies. Computation is
//! implemented in [`crate::accumulator`].

use crate::{MetricValueType, Unit};
use bitflags::bitflags;
use petgraph::algo::toposort;
use petgraph::graphmap::DiGraphMap;
use std::collections::{BTreeSet, HashMap};
use std::fmt::{Display, Formatter, Result as FmtResult};
use std::sync::LazyLock;

/// Metric identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
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
}

impl MetricTag {
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
            Self::InterChunkLatency => "inter_chunk_latency",
            Self::CreditDropLatency => "credit_drop_latency",
            Self::OutputSequenceLength => "output_sequence_length",
            Self::InputSequenceLength => "input_sequence_length",
            Self::ErrorInputSequenceLength => "error_isl",
            Self::OutputTokenCount => "output_token_count",
            Self::ReasoningTokenCount => "reasoning_token_count",
            Self::TotalOutputSequenceLength => "total_output_sequence_length",
            Self::TotalInputSequenceLength => "total_input_sequence_length",
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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
        /// Requires streaming token arrivals.
        const STREAMING_TOKENS_ONLY = 1 << 0;
        /// Requires produced output tokens.
        const PRODUCES_TOKENS_ONLY = 1 << 1;
        /// Requires tokenized input.
        const TOKENIZES_INPUT_ONLY = 1 << 2;
        /// Bit 3 is reserved to match the Python enum layout.
        const LARGER_IS_BETTER = 1 << 4;
        /// Metric applies only to failed records.
        const ERROR_ONLY = 1 << 5;
        /// Internal helper metric.
        const INTERNAL = 1 << 6;
        /// Hide per-record output.
        const NO_INDIVIDUAL_RECORDS = 1 << 7;
        /// Metric participates in goodput.
        const GOODPUT = 1 << 8;
        /// Metric is usage-reporting only.
        const USAGE_ONLY = 1 << 9;
        /// Metric is usage-difference only.
        const USAGE_DIFF_ONLY = 1 << 10;
        /// Metric requires HTTP trace data.
        const HTTP_TRACE_ONLY = 1 << 11;
        /// Percentiles include failed requests through adj_*.
        const PERCENTILE_INCLUDES_FAILED_REQUESTS = 1 << 12;
        /// Supports reasoning-token endpoints.
        const SUPPORTS_REASONING = 1 << 13;
        /// Supports audio endpoints.
        const SUPPORTS_AUDIO_ONLY = 1 << 14;
        /// Experimental output.
        const EXPERIMENTAL = 1 << 15;
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
        flags.is_empty() || self.contains(flags)
    }
}

/// Declarative metric catalog row.
#[derive(Debug, Clone, Copy)]
pub struct MetricSpec {
    pub tag: MetricTag,
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

macro_rules! spec {
    ($tag:ident, $header:expr, $unit:ident, $kind:ident, $agg:expr, $flags:expr, [$($req:ident),* $(,)?]) => {
        MetricSpec {
            tag: MetricTag::$tag,
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

/// Static metric catalog. Rows marked as injected have specs here and receive values
/// from future telemetry/accuracy accumulators or explicit record overrides.
pub static CATALOG: LazyLock<Vec<MetricSpec>> = LazyLock::new(|| {
    vec![
        spec!(
            RequestCount,
            "Requests",
            Request,
            Aggregate,
            Some(AggregationKind::Sum),
            MetricFlags::LARGER_IS_BETTER | MetricFlags::NO_INDIVIDUAL_RECORDS,
            []
        ),
        spec!(
            ErrorRequestCount,
            "Error Count",
            Request,
            Aggregate,
            Some(AggregationKind::Sum),
            MetricFlags::ERROR_ONLY | MetricFlags::NO_INDIVIDUAL_RECORDS,
            []
        ),
        spec!(
            CompletedRequestCount,
            "Completed",
            Request,
            Derived,
            None,
            MetricFlags::NO_INDIVIDUAL_RECORDS,
            [RequestCount, ErrorRequestCount]
        ),
        spec!(
            RequestErrorRate,
            "Err %",
            Percent,
            Derived,
            None,
            MetricFlags::NO_INDIVIDUAL_RECORDS,
            [RequestCount, ErrorRequestCount]
        ),
        spec!(
            GoodRequestCount,
            "Good Requests",
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
            "GoodReqFrac",
            Ratio,
            Derived,
            None,
            MetricFlags::GOODPUT | MetricFlags::LARGER_IS_BETTER,
            [GoodRequestCount, RequestCount]
        ),
        spec!(
            MinRequestTimestamp,
            "Min Req",
            Nanosecond,
            Aggregate,
            Some(AggregationKind::Min),
            MetricFlags::NO_INDIVIDUAL_RECORDS | MetricFlags::INTERNAL,
            []
        ),
        spec!(
            MaxResponseTimestamp,
            "Max Resp",
            Nanosecond,
            Aggregate,
            Some(AggregationKind::Max),
            MetricFlags::NO_INDIVIDUAL_RECORDS | MetricFlags::INTERNAL,
            [RequestLatency]
        ),
        spec!(
            BenchmarkDuration,
            "Duration",
            Nanosecond,
            Derived,
            None,
            MetricFlags::NO_INDIVIDUAL_RECORDS,
            [MinRequestTimestamp, MaxResponseTimestamp]
        ),
        spec!(
            RequestLatency,
            "Req Latency",
            Nanosecond,
            Record,
            None,
            MetricFlags::PERCENTILE_INCLUDES_FAILED_REQUESTS,
            []
        ),
        spec!(
            TimeToFirstToken,
            "TTFT",
            Nanosecond,
            Record,
            None,
            MetricFlags::STREAMING_TOKENS_ONLY | MetricFlags::PERCENTILE_INCLUDES_FAILED_REQUESTS,
            []
        ),
        spec!(
            TimeToSecondToken,
            "TTST",
            Nanosecond,
            Record,
            None,
            MetricFlags::STREAMING_TOKENS_ONLY,
            []
        ),
        spec!(
            TimeToFirstOutputToken,
            "TTFO",
            Nanosecond,
            Record,
            None,
            MetricFlags::STREAMING_TOKENS_ONLY | MetricFlags::SUPPORTS_REASONING,
            []
        ),
        spec!(
            InterTokenLatency,
            "ITL",
            Nanosecond,
            Record,
            None,
            MetricFlags::STREAMING_TOKENS_ONLY | MetricFlags::PERCENTILE_INCLUDES_FAILED_REQUESTS,
            [RequestLatency, TimeToFirstToken, OutputSequenceLength]
        ),
        spec!(
            InterChunkLatency,
            "ICL",
            Nanosecond,
            Record,
            None,
            MetricFlags::STREAMING_TOKENS_ONLY,
            []
        ),
        spec!(
            CreditDropLatency,
            "Credit Latency",
            Nanosecond,
            Record,
            None,
            MetricFlags::INTERNAL,
            []
        ),
        spec!(
            OutputSequenceLength,
            "OSL",
            Token,
            Record,
            None,
            MetricFlags::PRODUCES_TOKENS_ONLY | MetricFlags::LARGER_IS_BETTER,
            []
        ),
        spec!(
            InputSequenceLength,
            "ISL",
            Token,
            Record,
            None,
            MetricFlags::TOKENIZES_INPUT_ONLY | MetricFlags::LARGER_IS_BETTER,
            []
        ),
        spec!(
            ErrorInputSequenceLength,
            "Error ISL",
            Token,
            Record,
            None,
            MetricFlags::TOKENIZES_INPUT_ONLY | MetricFlags::ERROR_ONLY,
            []
        ),
        spec!(
            OutputTokenCount,
            "Output Tokens",
            Token,
            Record,
            None,
            MetricFlags::PRODUCES_TOKENS_ONLY | MetricFlags::LARGER_IS_BETTER,
            []
        ),
        spec!(
            ReasoningTokenCount,
            "Reasoning Tokens",
            Token,
            Record,
            None,
            MetricFlags::PRODUCES_TOKENS_ONLY | MetricFlags::SUPPORTS_REASONING,
            []
        ),
        spec!(
            TotalOutputSequenceLength,
            "Total OSL",
            Token,
            Derived,
            None,
            MetricFlags::PRODUCES_TOKENS_ONLY | MetricFlags::LARGER_IS_BETTER,
            [OutputSequenceLength]
        ),
        spec!(
            TotalInputSequenceLength,
            "Total ISL",
            Token,
            Derived,
            None,
            MetricFlags::TOKENIZES_INPUT_ONLY | MetricFlags::LARGER_IS_BETTER,
            [InputSequenceLength]
        ),
        spec!(
            TotalErrorInputSequenceLength,
            "Total Error ISL",
            Token,
            Derived,
            None,
            MetricFlags::TOKENIZES_INPUT_ONLY | MetricFlags::ERROR_ONLY,
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
            MetricFlags::SUPPORTS_REASONING,
            [ReasoningTokenCount]
        ),
        spec!(
            RequestThroughput,
            "Req/sec",
            RequestsPerSecond,
            Derived,
            None,
            MetricFlags::LARGER_IS_BETTER,
            [RequestCount, BenchmarkDuration]
        ),
        spec!(
            InputTokenThroughput,
            "Input TPS",
            TokensPerSecond,
            Derived,
            None,
            MetricFlags::LARGER_IS_BETTER,
            [TotalInputSequenceLength, BenchmarkDuration]
        ),
        spec!(
            OutputTokenThroughput,
            "Output TPS",
            TokensPerSecond,
            Derived,
            None,
            MetricFlags::PRODUCES_TOKENS_ONLY | MetricFlags::LARGER_IS_BETTER,
            [TotalOutputSequenceLength, BenchmarkDuration]
        ),
        spec!(
            TotalTokenThroughput,
            "Total TPS",
            TokensPerSecond,
            Derived,
            None,
            MetricFlags::LARGER_IS_BETTER,
            [
                TotalInputSequenceLength,
                TotalOutputSequenceLength,
                BenchmarkDuration
            ]
        ),
        spec!(
            OutputTokenThroughputPerUser,
            "Output TPS/User",
            TokensPerSecondPerUser,
            Record,
            None,
            MetricFlags::STREAMING_TOKENS_ONLY | MetricFlags::LARGER_IS_BETTER,
            [InterTokenLatency]
        ),
        spec!(
            E2eOutputTokenThroughput,
            "E2E Output TPS/User",
            TokensPerSecondPerUser,
            Record,
            None,
            MetricFlags::PRODUCES_TOKENS_ONLY | MetricFlags::LARGER_IS_BETTER,
            [OutputSequenceLength, RequestLatency]
        ),
        spec!(
            PrefillThroughputPerUser,
            "Prefill TPS/User",
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
            "RTFx",
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
            MetricFlags::USAGE_ONLY | MetricFlags::LARGER_IS_BETTER,
            []
        ),
        spec!(
            UsageCompletionTokens,
            "Usage Completion Tokens",
            Token,
            Record,
            None,
            MetricFlags::USAGE_ONLY | MetricFlags::LARGER_IS_BETTER,
            []
        ),
        spec!(
            UsageTotalTokens,
            "Usage Total Tokens",
            Token,
            Record,
            None,
            MetricFlags::USAGE_ONLY | MetricFlags::LARGER_IS_BETTER,
            []
        ),
        spec!(
            UsageReasoningTokens,
            "Usage Reasoning Tokens",
            Token,
            Record,
            None,
            MetricFlags::USAGE_ONLY | MetricFlags::SUPPORTS_REASONING,
            []
        ),
        spec!(
            UsagePromptAudioTokens,
            "Usage Prompt Audio Tokens",
            Token,
            Record,
            None,
            MetricFlags::USAGE_ONLY,
            []
        ),
        spec!(
            UsageCompletionAudioTokens,
            "Usage Completion Audio Tokens",
            Token,
            Record,
            None,
            MetricFlags::USAGE_ONLY,
            []
        ),
        spec!(
            UsageAcceptedPredictionTokens,
            "Usage Accepted Prediction Tokens",
            Token,
            Record,
            None,
            MetricFlags::USAGE_ONLY | MetricFlags::LARGER_IS_BETTER,
            []
        ),
        spec!(
            UsageRejectedPredictionTokens,
            "Usage Rejected Prediction Tokens",
            Token,
            Record,
            None,
            MetricFlags::USAGE_ONLY,
            []
        ),
        spec!(
            UsagePromptCacheReadTokens,
            "Usage Prompt Cache Read Tokens",
            Token,
            Record,
            None,
            MetricFlags::USAGE_ONLY | MetricFlags::LARGER_IS_BETTER,
            []
        ),
        spec!(
            UsagePromptCacheWriteTokens,
            "Usage Prompt Cache Write Tokens",
            Token,
            Record,
            None,
            MetricFlags::USAGE_ONLY,
            []
        ),
        spec!(
            UsagePromptCacheMissTokens,
            "Usage Prompt Cache Miss Tokens",
            Token,
            Record,
            None,
            MetricFlags::USAGE_ONLY,
            []
        ),
        spec!(
            UsageToolUsePromptTokens,
            "Usage Tool Use Prompt Tokens",
            Token,
            Record,
            None,
            MetricFlags::USAGE_ONLY,
            []
        ),
        spec!(
            UsagePromptAudioSeconds,
            "Usage Prompt Audio Seconds",
            Second,
            Record,
            None,
            MetricFlags::USAGE_ONLY,
            []
        ),
        spec!(
            TotalUsagePromptTokens,
            "Total Usage Prompt Tokens",
            Token,
            Derived,
            None,
            MetricFlags::USAGE_ONLY | MetricFlags::LARGER_IS_BETTER,
            [UsagePromptTokens]
        ),
        spec!(
            TotalUsageCompletionTokens,
            "Total Usage Completion Tokens",
            Token,
            Derived,
            None,
            MetricFlags::USAGE_ONLY | MetricFlags::LARGER_IS_BETTER,
            [UsageCompletionTokens]
        ),
        spec!(
            TotalUsageTotalTokens,
            "Total Usage Total Tokens",
            Token,
            Derived,
            None,
            MetricFlags::USAGE_ONLY | MetricFlags::LARGER_IS_BETTER,
            [UsageTotalTokens]
        ),
        spec!(
            TotalUsageReasoningTokens,
            "Total Usage Reasoning Tokens",
            Token,
            Derived,
            None,
            MetricFlags::USAGE_ONLY | MetricFlags::SUPPORTS_REASONING,
            [UsageReasoningTokens]
        ),
        spec!(
            TotalUsagePromptCacheReadTokens,
            "Total Usage Prompt Cache Read Tokens",
            Token,
            Derived,
            None,
            MetricFlags::USAGE_ONLY | MetricFlags::LARGER_IS_BETTER,
            [UsagePromptCacheReadTokens]
        ),
        spec!(
            TotalUsagePromptCacheWriteTokens,
            "Total Usage Prompt Cache Write Tokens",
            Token,
            Derived,
            None,
            MetricFlags::USAGE_ONLY,
            [UsagePromptCacheWriteTokens]
        ),
        spec!(
            TotalUsagePromptCacheMissTokens,
            "Total Usage Prompt Cache Miss Tokens",
            Token,
            Derived,
            None,
            MetricFlags::USAGE_ONLY,
            [UsagePromptCacheMissTokens]
        ),
        spec!(
            OverallUsagePromptCacheReadPct,
            "Overall Usage Prompt Cache Read %",
            Percent,
            Derived,
            None,
            MetricFlags::USAGE_ONLY | MetricFlags::LARGER_IS_BETTER,
            [TotalUsagePromptCacheReadTokens, TotalUsagePromptTokens]
        ),
        spec!(
            UsagePromptTokensDiffPct,
            "Usage Prompt Tokens Diff %",
            Percent,
            Record,
            None,
            MetricFlags::USAGE_DIFF_ONLY,
            [InputSequenceLength, UsagePromptTokens]
        ),
        spec!(
            UsageCompletionTokensDiffPct,
            "Usage Completion Tokens Diff %",
            Percent,
            Record,
            None,
            MetricFlags::USAGE_DIFF_ONLY,
            [OutputSequenceLength, UsageCompletionTokens]
        ),
        spec!(
            UsageReasoningTokensDiffPct,
            "Usage Reasoning Tokens Diff %",
            Percent,
            Record,
            None,
            MetricFlags::USAGE_DIFF_ONLY,
            [ReasoningTokenCount, UsageReasoningTokens]
        ),
        spec!(
            UsageDiscrepancyCount,
            "Usage Discrepancy Count",
            Count,
            Aggregate,
            Some(AggregationKind::Sum),
            MetricFlags::USAGE_DIFF_ONLY,
            []
        ),
        spec!(
            RequestedOutputSequenceLength,
            "Requested OSL",
            Token,
            Record,
            None,
            MetricFlags::INTERNAL,
            []
        ),
        spec!(
            OslMismatchDiffPct,
            "OSL Mismatch Diff %",
            Percent,
            Record,
            None,
            MetricFlags::INTERNAL,
            [RequestedOutputSequenceLength, OutputSequenceLength]
        ),
        spec!(
            OslMismatchCount,
            "OSL Mismatch Count",
            Count,
            Aggregate,
            Some(AggregationKind::Sum),
            MetricFlags::INTERNAL,
            []
        ),
        spec!(
            ThinkingEfficiency,
            "Thinking Efficiency",
            Ratio,
            Record,
            None,
            MetricFlags::EXPERIMENTAL | MetricFlags::SUPPORTS_REASONING,
            [ReasoningTokenCount, OutputTokenCount]
        ),
        spec!(
            OverallThinkingEfficiency,
            "Overall Thinking Efficiency",
            Ratio,
            Derived,
            None,
            MetricFlags::EXPERIMENTAL | MetricFlags::SUPPORTS_REASONING,
            [TotalReasoningTokens, TotalOutputTokens]
        ),
        spec!(
            TotalGpuPower,
            "Total GPU Power",
            Watt,
            Derived,
            None,
            MetricFlags::INTERNAL,
            []
        ),
        spec!(
            TotalGpuEnergy,
            "Total GPU Energy",
            Joule,
            Derived,
            None,
            MetricFlags::INTERNAL,
            []
        ),
        spec!(
            OutputTokensPerJoule,
            "Output Tokens/Joule",
            Ratio,
            Derived,
            None,
            MetricFlags::LARGER_IS_BETTER,
            [TotalOutputTokens, TotalGpuEnergy]
        ),
        spec!(
            EnergyPerUser,
            "Energy/User",
            Joule,
            Derived,
            None,
            MetricFlags::INTERNAL,
            [TotalGpuEnergy]
        ),
        spec!(
            NetworkAdjustedRequestLatency,
            "Network Adjusted Req Latency",
            Nanosecond,
            Derived,
            None,
            MetricFlags::INTERNAL,
            [RequestLatency]
        ),
        spec!(
            NetworkAdjustedTimeToFirstToken,
            "Network Adjusted TTFT",
            Nanosecond,
            Derived,
            None,
            MetricFlags::INTERNAL,
            [TimeToFirstToken]
        ),
        spec!(
            NetworkAdjustedTimeToFirstOutputToken,
            "Network Adjusted TTFO",
            Nanosecond,
            Derived,
            None,
            MetricFlags::INTERNAL,
            [TimeToFirstOutputToken]
        ),
        spec!(
            NetworkRtt,
            "Network RTT",
            Nanosecond,
            Derived,
            None,
            MetricFlags::INTERNAL,
            []
        ),
        spec!(
            StreamSetupLatency,
            "Stream Setup Latency",
            Nanosecond,
            Record,
            None,
            MetricFlags::EXPERIMENTAL,
            []
        ),
        spec!(
            StreamPrefillLatency,
            "Stream Prefill Latency",
            Nanosecond,
            Record,
            None,
            MetricFlags::EXPERIMENTAL,
            [TimeToFirstToken, StreamSetupLatency]
        ),
        spec!(
            AccuracyCorrect,
            "Accuracy Correct",
            Count,
            Aggregate,
            Some(AggregationKind::Sum),
            MetricFlags::INTERNAL,
            []
        ),
        spec!(
            AccuracyUnparsed,
            "Accuracy Unparsed",
            Count,
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
            "Images",
            Image,
            Record,
            None,
            MetricFlags::LARGER_IS_BETTER,
            []
        ),
        spec!(
            ImageThroughput,
            "Image Throughput",
            RequestsPerSecond,
            Record,
            None,
            MetricFlags::LARGER_IS_BETTER,
            [NumImages, RequestLatency]
        ),
        spec!(
            ImageLatency,
            "Image Latency",
            Nanosecond,
            Record,
            None,
            MetricFlags::PERCENTILE_INCLUDES_FAILED_REQUESTS,
            [RequestLatency]
        ),
        spec!(
            VideoInferenceTime,
            "Video Inference Time",
            Millisecond,
            Record,
            None,
            MetricFlags::EXPERIMENTAL,
            []
        ),
        spec!(
            VideoPeakMemory,
            "Video Peak Memory",
            Megabyte,
            Record,
            None,
            MetricFlags::EXPERIMENTAL,
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
            "HTTP DNS",
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
            "HTTP Waiting",
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
            "HTTP Duration",
            Nanosecond,
            Record,
            None,
            MetricFlags::HTTP_TRACE_ONLY,
            []
        ),
        spec!(
            HttpReqConnectionReused,
            "HTTP Reused",
            Count,
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
            "HTTP Conn Overhead",
            Nanosecond,
            Record,
            None,
            MetricFlags::HTTP_TRACE_ONLY,
            [HttpReqBlocked, HttpReqDnsLookup, HttpReqConnecting]
        ),
        spec!(
            HttpReqTotal,
            "HTTP Total",
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
    ]
});

/// Looks up a catalog spec.
pub fn spec_for(tag: MetricTag) -> Option<&'static MetricSpec> {
    CATALOG.iter().find(|spec| spec.tag == tag)
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
    use super::{CATALOG, MetricFlags, MetricTag, validate_catalog};

    #[test]
    fn metric_flags_preserve_reserved_gap_and_missing_semantics() {
        assert_eq!(MetricFlags::TOKENIZES_INPUT_ONLY.bits(), 1 << 2);
        assert_eq!(MetricFlags::LARGER_IS_BETTER.bits(), 1 << 4);
        assert!(MetricFlags::NONE.missing(MetricFlags::NONE));
        assert!(
            MetricFlags::STREAMING_TOKENS_ONLY
                .has_any(MetricFlags::STREAMING_TOKENS_ONLY | MetricFlags::GOODPUT)
        );
    }

    #[test]
    fn catalog_has_unique_acyclic_resolved_dependencies() {
        let order = validate_catalog().unwrap();
        assert!(order.contains(&MetricTag::RequestLatency));
        assert!(
            CATALOG
                .iter()
                .any(|spec| spec.tag == MetricTag::GoodRequestFraction)
        );
    }
}
