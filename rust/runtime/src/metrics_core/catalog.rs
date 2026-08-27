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

/// Source-compatible name for the open metric-tag ID.
pub type MetricTag = crate::metrics_core::tag_id::MetricTagId;

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
    SpecDecode,
    Effective,
    Active,
}

impl MetricConsoleGroup {
    /// Stable snake-case name used as the definition group key.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Default => "default",
            Self::Usage => "usage",
            Self::Cache => "cache",
            Self::Prediction => "prediction",
            Self::Audio => "audio",
            Self::Reasoning => "reasoning",
            Self::SpecDecode => "spec_decode",
            Self::Effective => "effective",
            Self::Active => "active",
        }
    }
}

/// Direction used by plot/threshold consumers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PlotMetricDirection {
    LargerIsBetter,
    SmallerIsBetter,
    Neutral,
}

/// Derives the plot direction from a [`Definition`]'s `larger_is_better` flag.
pub const fn plot_direction_for(def: &Definition) -> PlotMetricDirection {
    if def.larger_is_better {
        PlotMetricDirection::LargerIsBetter
    } else {
        PlotMetricDirection::SmallerIsBetter
    }
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

    /// Returns true when none of the requested flags are set, so an empty
    /// request is trivially satisfied.
    pub fn missing(self, flags: Self) -> bool {
        flags.is_empty() || !self.intersects(flags)
    }
}

/// Declarative metric catalog row.
#[derive(Debug, Clone, Copy)]
pub struct MetricSpec {
    pub tag: MetricTag,
    /// Embedded portable definition; the source of truth for all presentation
    /// (header, units, short header, display order, value type).
    pub def: Definition,
    pub flags: MetricFlags,
    pub console_group: MetricConsoleGroup,
    pub required: &'static [MetricTag],
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

    /// Value type, delegating to the embedded [`Definition`].
    pub fn value_type(&self) -> MetricValueType {
        self.def.value_type
    }

    /// Whether the short header omits the unit suffix, delegating to the
    /// embedded [`Definition`].
    pub fn short_header_hide_unit(&self) -> bool {
        self.def.short_header_hide_unit
    }
}

/// Per-tag short header, folded into catalog construction (was a runtime
/// post-pass). Kept as a `const fn` so the catalog is a `const`-built static.
const fn cfg_short_header(tag: MetricTag) -> Option<&'static str> {
    use MetricTag::*;
    match tag {
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
        SpecDecodeAcceptanceLength => Some("Acceptance Length"),
        SpecDecodeTokenWeightedAcceptanceLength => Some("Token-Wtd Accept Len"),
        SpecDecodeDraftAcceptanceRate => Some("Draft Accept Rate"),
        SpecDecodeOverallDraftAcceptanceRate => Some("Overall Draft Accept Rate"),
        SpecDecodeAcceptedPerVerified => Some("Accepted / Verified"),
        SpecDecodeSteps => Some("Spec Decode Steps"),
        SpecDecodeAcceptedDraftTokens => Some("Accepted Draft"),
        SpecDecodeDraftTokens => Some("Draft Tokens"),
        TotalSpecDecodeSteps => Some("Total Spec Decode Steps"),
        TotalAcceptedDraftTokens => Some("Total Accepted Draft"),
        TotalDraftTokens => Some("Total Draft"),
        RequestCount => Some("Requests"),
        RequestErrorRate => Some("Err %"),
        RequestLatency => Some("Req Latency"),
        RequestThroughput => Some("Req/sec"),
        RequestedOutputSequenceLength => Some("Req OSL"),
        Rtfx => Some("RTFx"),
        TimeToFirstOutputToken => Some("TTFO"),
        TimeToLastRoundTrip => Some("Last Round Trip"),
        AverageRoundTripTime => Some("Avg Round Trip"),
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
    }
}

/// Per-tag flag: hide the unit suffix on the short header.
const fn cfg_short_header_hide_unit(tag: MetricTag) -> bool {
    use MetricTag::*;
    matches!(
        tag,
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
            | SpecDecodeAcceptanceLength
            | SpecDecodeTokenWeightedAcceptanceLength
            | SpecDecodeDraftAcceptanceRate
            | SpecDecodeOverallDraftAcceptanceRate
            | SpecDecodeAcceptedPerVerified
            | SpecDecodeSteps
            | SpecDecodeAcceptedDraftTokens
            | SpecDecodeDraftTokens
            | TotalSpecDecodeSteps
            | TotalAcceptedDraftTokens
            | TotalDraftTokens
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
    )
}

/// Per-tag display-unit override.
const fn cfg_display_unit(tag: MetricTag) -> Option<Unit> {
    use MetricTag::*;
    match tag {
        BenchmarkDuration => Some(Unit::Second),
        RequestLatency
        | TimeToFirstToken
        | TimeToSecondToken
        | TimeToFirstOutputToken
        | TimeToLastRoundTrip
        | AverageRoundTripTime
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
    }
}

/// Per-tag display-order override.
const fn cfg_display_order(tag: MetricTag) -> Option<u32> {
    use MetricTag::*;
    match tag {
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
        SpecDecodeAcceptanceLength => Some(5000),
        SpecDecodeTokenWeightedAcceptanceLength => Some(5010),
        SpecDecodeDraftAcceptanceRate => Some(5020),
        SpecDecodeOverallDraftAcceptanceRate => Some(5025),
        SpecDecodeAcceptedPerVerified => Some(5030),
        SpecDecodeSteps => Some(5040),
        SpecDecodeAcceptedDraftTokens => Some(5050),
        SpecDecodeDraftTokens => Some(5060),
        TotalSpecDecodeSteps => Some(5140),
        TotalAcceptedDraftTokens => Some(5150),
        TotalDraftTokens => Some(5160),
        TimeToFirstOutputToken => Some(210),
        TimeToLastRoundTrip => Some(220),
        AverageRoundTripTime => Some(230),
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
    }
}

/// Per-tag console grouping.
const fn cfg_console_group(tag: MetricTag) -> MetricConsoleGroup {
    use MetricConsoleGroup::{Active, Effective, None as Hidden, SpecDecode, Usage};
    use MetricTag::*;
    match tag {
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
        SpecDecodeAcceptanceLength
        | SpecDecodeTokenWeightedAcceptanceLength
        | SpecDecodeDraftAcceptanceRate
        | SpecDecodeOverallDraftAcceptanceRate
        | SpecDecodeAcceptedPerVerified
        | SpecDecodeSteps => SpecDecode,
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
        | SpecDecodeAcceptedDraftTokens
        | SpecDecodeDraftTokens
        | TotalSpecDecodeSteps
        | TotalAcceptedDraftTokens
        | TotalDraftTokens
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
    }
}

/// Per-tag value type.
const fn cfg_value_type(tag: MetricTag) -> MetricValueType {
    use MetricTag::*;
    match tag {
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
        | TimeToLastRoundTrip
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
        | SpecDecodeSteps
        | SpecDecodeAcceptedDraftTokens
        | SpecDecodeDraftTokens
        | TotalSpecDecodeSteps
        | TotalAcceptedDraftTokens
        | TotalDraftTokens
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
    }
}

macro_rules! spec {
    ($tag:ident, $header:expr, $unit:ident, $kind:ident, $agg:expr, $flags:expr, [$($req:ident),* $(,)?]) => {
        MetricSpec {
            tag: MetricTag::$tag,
            def: Definition {
                id: MetricTag::$tag.as_str(),
                header: $header,
                short_header: cfg_short_header(MetricTag::$tag),
                short_header_hide_unit: cfg_short_header_hide_unit(MetricTag::$tag),
                unit: Unit::$unit,
                display_unit: cfg_display_unit(MetricTag::$tag),
                display_order: cfg_display_order(MetricTag::$tag),
                group: DefinitionGroup::Named(cfg_console_group(MetricTag::$tag).as_str()),
                larger_is_better: $flags.contains(MetricFlags::LARGER_IS_BETTER),
                value_type: cfg_value_type(MetricTag::$tag),
                aliases: &[],
                deprecated_since: None,
            },
            flags: $flags,
            console_group: cfg_console_group(MetricTag::$tag),
            required: &[$(MetricTag::$req),*],
            kind: MetricType::$kind,
            aggregation: $agg,
        }
    };
}

/// Static metric catalog. Injected rows receive values from telemetry, accuracy,
/// or explicit record overrides.
pub static CATALOG: [MetricSpec; MetricTag::COUNT] = [
    spec!(
        RequestCount,
        "Request Count",
        Request,
        Aggregate,
        Some(AggregationKind::Sum),
        MetricFlags::LARGER_IS_BETTER.union(MetricFlags::NO_INDIVIDUAL_RECORDS),
        []
    ),
    spec!(
        ErrorRequestCount,
        "Error Request Count",
        Request,
        Aggregate,
        Some(AggregationKind::Sum),
        MetricFlags::ERROR_ONLY.union(MetricFlags::NO_INDIVIDUAL_RECORDS),
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
        MetricFlags::GOODPUT.union(MetricFlags::LARGER_IS_BETTER),
        [GoodRequestCount, RequestCount]
    ),
    spec!(
        MinRequestTimestamp,
        "Minimum Request Timestamp",
        Nanosecond,
        Aggregate,
        Some(AggregationKind::Min),
        MetricFlags::NO_INDIVIDUAL_RECORDS.union(MetricFlags::INTERNAL),
        []
    ),
    spec!(
        MaxResponseTimestamp,
        "Maximum Response Timestamp",
        Nanosecond,
        Aggregate,
        Some(AggregationKind::Max),
        MetricFlags::NO_INDIVIDUAL_RECORDS.union(MetricFlags::INTERNAL),
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
        MetricFlags::STREAMING_TOKENS_ONLY.union(MetricFlags::PERCENTILE_INCLUDES_FAILED_REQUESTS),
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
        MetricFlags::STREAMING_TOKENS_ONLY.union(MetricFlags::SUPPORTS_REASONING),
        []
    ),
    spec!(
        InterTokenLatency,
        "Inter Token Latency",
        Nanosecond,
        Record,
        None,
        MetricFlags::STREAMING_TOKENS_ONLY.union(MetricFlags::PERCENTILE_INCLUDES_FAILED_REQUESTS),
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
        MetricFlags::STREAMING_TOKENS_ONLY.union(MetricFlags::PERCENTILE_INCLUDES_FAILED_REQUESTS),
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
        MetricFlags::PRODUCES_TOKENS_ONLY.union(MetricFlags::LARGER_IS_BETTER),
        []
    ),
    spec!(
        InputSequenceLength,
        "Input Sequence Length",
        Token,
        Record,
        None,
        MetricFlags::TOKENIZES_INPUT_ONLY.union(MetricFlags::LARGER_IS_BETTER),
        []
    ),
    spec!(
        ErrorInputSequenceLength,
        "Error Input Sequence Length",
        Token,
        Record,
        None,
        MetricFlags::TOKENIZES_INPUT_ONLY
            .union(MetricFlags::ERROR_ONLY)
            .union(MetricFlags::LARGER_IS_BETTER),
        []
    ),
    spec!(
        OutputTokenCount,
        "Output Token Count",
        Token,
        Record,
        None,
        MetricFlags::PRODUCES_TOKENS_ONLY.union(MetricFlags::LARGER_IS_BETTER),
        []
    ),
    spec!(
        ReasoningTokenCount,
        "Reasoning Token Count",
        Token,
        Record,
        None,
        MetricFlags::PRODUCES_TOKENS_ONLY
            .union(MetricFlags::LARGER_IS_BETTER)
            .union(MetricFlags::SUPPORTS_REASONING),
        []
    ),
    spec!(
        TotalOutputSequenceLength,
        "Total Output Sequence Length",
        Token,
        Derived,
        None,
        MetricFlags::PRODUCES_TOKENS_ONLY.union(MetricFlags::LARGER_IS_BETTER),
        [OutputSequenceLength]
    ),
    spec!(
        TotalInputSequenceLength,
        "Total Input Sequence Length",
        Token,
        Derived,
        None,
        MetricFlags::TOKENIZES_INPUT_ONLY.union(MetricFlags::LARGER_IS_BETTER),
        [InputSequenceLength]
    ),
    spec!(
        TotalErrorInputSequenceLength,
        "Total Error Input Sequence Length",
        Token,
        Derived,
        None,
        MetricFlags::TOKENIZES_INPUT_ONLY
            .union(MetricFlags::ERROR_ONLY)
            .union(MetricFlags::LARGER_IS_BETTER),
        [ErrorInputSequenceLength]
    ),
    spec!(
        TotalOutputTokens,
        "Total Output Tokens",
        Token,
        Derived,
        None,
        MetricFlags::PRODUCES_TOKENS_ONLY.union(MetricFlags::LARGER_IS_BETTER),
        [OutputTokenCount]
    ),
    spec!(
        TotalReasoningTokens,
        "Total Reasoning Tokens",
        Token,
        Derived,
        None,
        MetricFlags::PRODUCES_TOKENS_ONLY.union(MetricFlags::LARGER_IS_BETTER),
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
        MetricFlags::PRODUCES_TOKENS_ONLY.union(MetricFlags::LARGER_IS_BETTER),
        [TotalOutputSequenceLength, BenchmarkDuration]
    ),
    spec!(
        TotalTokenThroughput,
        "Total Token Throughput",
        TokensPerSecond,
        Derived,
        None,
        MetricFlags::PRODUCES_TOKENS_ONLY.union(MetricFlags::LARGER_IS_BETTER),
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
        MetricFlags::STREAMING_TOKENS_ONLY.union(MetricFlags::LARGER_IS_BETTER),
        [InterTokenLatency]
    ),
    spec!(
        E2eOutputTokenThroughput,
        "E2E Output Token Throughput",
        TokensPerSecondPerUser,
        Record,
        None,
        MetricFlags::PRODUCES_TOKENS_ONLY.union(MetricFlags::LARGER_IS_BETTER),
        [OutputSequenceLength, RequestLatency]
    ),
    spec!(
        PrefillThroughputPerUser,
        "Prefill Throughput Per User",
        TokensPerSecondPerUser,
        Record,
        None,
        MetricFlags::STREAMING_TOKENS_ONLY
            .union(MetricFlags::TOKENIZES_INPUT_ONLY)
            .union(MetricFlags::LARGER_IS_BETTER),
        [InputSequenceLength, TimeToFirstToken]
    ),
    spec!(
        Rtfx,
        "Inverse Real-Time Factor (RTFx)",
        Ratio,
        Record,
        None,
        MetricFlags::SUPPORTS_AUDIO_ONLY.union(MetricFlags::LARGER_IS_BETTER),
        [AudioDuration, RequestLatency]
    ),
    spec!(
        UsagePromptTokens,
        "Usage Prompt Tokens",
        Token,
        Record,
        None,
        MetricFlags::TOKENIZES_INPUT_ONLY.union(MetricFlags::LARGER_IS_BETTER),
        []
    ),
    spec!(
        UsageCompletionTokens,
        "Usage Completion Tokens",
        Token,
        Record,
        None,
        MetricFlags::PRODUCES_TOKENS_ONLY.union(MetricFlags::LARGER_IS_BETTER),
        []
    ),
    spec!(
        UsageTotalTokens,
        "Usage Total Tokens",
        Token,
        Record,
        None,
        MetricFlags::PRODUCES_TOKENS_ONLY.union(MetricFlags::LARGER_IS_BETTER),
        []
    ),
    spec!(
        UsageReasoningTokens,
        "Usage Reasoning Tokens",
        Token,
        Record,
        None,
        MetricFlags::PRODUCES_TOKENS_ONLY.union(MetricFlags::LARGER_IS_BETTER),
        []
    ),
    spec!(
        UsagePromptAudioTokens,
        "Usage Prompt Audio Tokens",
        Token,
        Record,
        None,
        MetricFlags::LARGER_IS_BETTER.union(MetricFlags::SUPPORTS_AUDIO_ONLY),
        []
    ),
    spec!(
        UsageCompletionAudioTokens,
        "Usage Completion Audio Tokens",
        Token,
        Record,
        None,
        MetricFlags::PRODUCES_TOKENS_ONLY
            .union(MetricFlags::LARGER_IS_BETTER)
            .union(MetricFlags::SUPPORTS_AUDIO_ONLY),
        []
    ),
    spec!(
        UsageAcceptedPredictionTokens,
        "Usage Accepted Prediction Tokens",
        Token,
        Record,
        None,
        MetricFlags::PRODUCES_TOKENS_ONLY.union(MetricFlags::LARGER_IS_BETTER),
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
        MetricFlags::LARGER_IS_BETTER.union(MetricFlags::SUPPORTS_AUDIO_ONLY),
        []
    ),
    spec!(
        TotalUsagePromptTokens,
        "Total Usage Prompt Tokens",
        Token,
        Derived,
        None,
        MetricFlags::TOKENIZES_INPUT_ONLY.union(MetricFlags::LARGER_IS_BETTER),
        [UsagePromptTokens]
    ),
    spec!(
        TotalUsageCompletionTokens,
        "Total Usage Completion Tokens",
        Token,
        Derived,
        None,
        MetricFlags::PRODUCES_TOKENS_ONLY.union(MetricFlags::LARGER_IS_BETTER),
        [UsageCompletionTokens]
    ),
    spec!(
        TotalUsageTotalTokens,
        "Total Usage Total Tokens",
        Token,
        Derived,
        None,
        MetricFlags::PRODUCES_TOKENS_ONLY.union(MetricFlags::LARGER_IS_BETTER),
        [UsageTotalTokens]
    ),
    spec!(
        TotalUsageReasoningTokens,
        "Total Usage Reasoning Tokens",
        Token,
        Derived,
        None,
        MetricFlags::PRODUCES_TOKENS_ONLY.union(MetricFlags::LARGER_IS_BETTER),
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
        MetricFlags::LARGER_IS_BETTER.union(MetricFlags::SUPPORTS_AUDIO_ONLY),
        [UsagePromptAudioTokens]
    ),
    spec!(
        TotalUsageCompletionAudioTokens,
        "Total Usage Completion Audio Tokens",
        Token,
        Derived,
        None,
        MetricFlags::PRODUCES_TOKENS_ONLY
            .union(MetricFlags::LARGER_IS_BETTER)
            .union(MetricFlags::SUPPORTS_AUDIO_ONLY),
        [UsageCompletionAudioTokens]
    ),
    spec!(
        TotalUsageAcceptedPredictionTokens,
        "Total Usage Accepted Prediction Tokens",
        Token,
        Derived,
        None,
        MetricFlags::PRODUCES_TOKENS_ONLY.union(MetricFlags::LARGER_IS_BETTER),
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
        MetricFlags::LARGER_IS_BETTER.union(MetricFlags::SUPPORTS_AUDIO_ONLY),
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
        MetricFlags::USAGE_DIFF_ONLY.union(MetricFlags::TOKENIZES_INPUT_ONLY),
        [InputSequenceLength, UsagePromptTokens]
    ),
    spec!(
        UsageCompletionTokensDiffPct,
        "Usage Completion Diff",
        Percent,
        Record,
        None,
        MetricFlags::USAGE_DIFF_ONLY.union(MetricFlags::PRODUCES_TOKENS_ONLY),
        [OutputSequenceLength, UsageCompletionTokens]
    ),
    spec!(
        UsageReasoningTokensDiffPct,
        "Usage Reasoning Diff",
        Percent,
        Record,
        None,
        MetricFlags::USAGE_DIFF_ONLY
            .union(MetricFlags::PRODUCES_TOKENS_ONLY)
            .union(MetricFlags::SUPPORTS_REASONING),
        [ReasoningTokenCount, UsageReasoningTokens]
    ),
    spec!(
        UsageDiscrepancyCount,
        "Usage Discrepancy Count",
        Request,
        Aggregate,
        Some(AggregationKind::Sum),
        MetricFlags::USAGE_DIFF_ONLY.union(MetricFlags::NO_INDIVIDUAL_RECORDS),
        [UsagePromptTokensDiffPct, UsageCompletionTokensDiffPct]
    ),
    spec!(
        RequestedOutputSequenceLength,
        "Requested OSL",
        Token,
        Record,
        None,
        MetricFlags::INTERNAL.union(MetricFlags::PRODUCES_TOKENS_ONLY),
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
        MetricFlags::PRODUCES_TOKENS_ONLY.union(MetricFlags::NO_INDIVIDUAL_RECORDS),
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
            .union(MetricFlags::PRODUCES_TOKENS_ONLY)
            .union(MetricFlags::SUPPORTS_REASONING),
        [ReasoningTokenCount, OutputTokenCount]
    ),
    spec!(
        OverallThinkingEfficiency,
        "Overall Thinking Efficiency",
        Ratio,
        Derived,
        None,
        MetricFlags::EXPERIMENTAL
            .union(MetricFlags::PRODUCES_TOKENS_ONLY)
            .union(MetricFlags::SUPPORTS_REASONING),
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
        MetricFlags::PRODUCES_TOKENS_ONLY.union(MetricFlags::LARGER_IS_BETTER),
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
        MetricFlags::STREAMING_TOKENS_ONLY.union(MetricFlags::SUPPORTS_REASONING),
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
        MetricFlags::STREAMING_ONLY.union(MetricFlags::EXPERIMENTAL),
        []
    ),
    spec!(
        StreamPrefillLatency,
        "Stream Prefill Latency",
        Nanosecond,
        Record,
        None,
        MetricFlags::STREAMING_TOKENS_ONLY.union(MetricFlags::EXPERIMENTAL),
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
        MetricFlags::SUPPORTS_IMAGE_ONLY.union(MetricFlags::LARGER_IS_BETTER),
        [NumImages]
    ),
    spec!(
        ImageSamplesPerSecond,
        "Image Samples Per Second",
        ImagesPerSecond,
        Derived,
        None,
        MetricFlags::SUPPORTS_IMAGE_ONLY.union(MetricFlags::LARGER_IS_BETTER),
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
        MetricFlags::NO_INDIVIDUAL_RECORDS.union(MetricFlags::LARGER_IS_BETTER),
        []
    ),
    spec!(
        EffectivePrefillThroughput,
        "Effective Prefill Throughput",
        TokensPerSecond,
        Derived,
        None,
        MetricFlags::NO_INDIVIDUAL_RECORDS.union(MetricFlags::LARGER_IS_BETTER),
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
        MetricFlags::NO_INDIVIDUAL_RECORDS.union(MetricFlags::LARGER_IS_BETTER),
        []
    ),
    spec!(
        EffectiveDecodeThroughputPerUser,
        "Effective Decode Throughput Per User",
        TokensPerSecondPerUser,
        Derived,
        None,
        MetricFlags::NO_INDIVIDUAL_RECORDS.union(MetricFlags::LARGER_IS_BETTER),
        []
    ),
    spec!(
        EffectivePrefillThroughputPerUser,
        "Effective Prefill Throughput Per User",
        TokensPerSecondPerUser,
        Derived,
        None,
        MetricFlags::NO_INDIVIDUAL_RECORDS.union(MetricFlags::LARGER_IS_BETTER),
        []
    ),
    spec!(
        EffectiveImageSamplesPerSecond,
        "Effective Image Samples Per Second",
        ImagesPerSecond,
        Derived,
        None,
        MetricFlags::NO_INDIVIDUAL_RECORDS
            .union(MetricFlags::SUPPORTS_IMAGE_ONLY)
            .union(MetricFlags::LARGER_IS_BETTER),
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
        MetricFlags::NO_INDIVIDUAL_RECORDS.union(MetricFlags::LARGER_IS_BETTER),
        []
    ),
    spec!(
        ActivePrefillThroughput,
        "Active Prefill Throughput",
        TokensPerSecond,
        Derived,
        None,
        MetricFlags::NO_INDIVIDUAL_RECORDS.union(MetricFlags::LARGER_IS_BETTER),
        []
    ),
    spec!(
        ActiveDecodeThroughputPerUser,
        "Active Decode Throughput Per User",
        TokensPerSecondPerUser,
        Derived,
        None,
        MetricFlags::NO_INDIVIDUAL_RECORDS.union(MetricFlags::LARGER_IS_BETTER),
        []
    ),
    spec!(
        ActivePrefillThroughputPerUser,
        "Active Prefill Throughput Per User",
        TokensPerSecondPerUser,
        Derived,
        None,
        MetricFlags::NO_INDIVIDUAL_RECORDS.union(MetricFlags::LARGER_IS_BETTER),
        []
    ),
    spec!(
        ActiveImageSamplesPerSecond,
        "Active Image Samples Per Second",
        ImagesPerSecond,
        Derived,
        None,
        MetricFlags::NO_INDIVIDUAL_RECORDS
            .union(MetricFlags::SUPPORTS_IMAGE_ONLY)
            .union(MetricFlags::LARGER_IS_BETTER),
        []
    ),
    spec!(
        EffectiveImageSamplesPerSecondPerUser,
        "Effective Image Samples Per Second Per User",
        ImagesPerSecondPerUser,
        Derived,
        None,
        MetricFlags::NO_INDIVIDUAL_RECORDS
            .union(MetricFlags::SUPPORTS_IMAGE_ONLY)
            .union(MetricFlags::LARGER_IS_BETTER),
        []
    ),
    spec!(
        ActiveTotalThroughput,
        "Active Total Throughput",
        TokensPerSecond,
        Derived,
        None,
        MetricFlags::NO_INDIVIDUAL_RECORDS.union(MetricFlags::LARGER_IS_BETTER),
        []
    ),
    spec!(
        TimeToLastRoundTrip,
        "Time to Last Round Trip",
        Nanosecond,
        Record,
        None,
        MetricFlags::NONE,
        []
    ),
    spec!(
        AverageRoundTripTime,
        "Average Round Trip Time",
        Nanosecond,
        Record,
        None,
        MetricFlags::NONE,
        []
    ),
    spec!(
        SpecDecodeAcceptanceLength,
        "Acceptance Length",
        Ratio,
        Record,
        None,
        MetricFlags::LARGER_IS_BETTER,
        []
    ),
    spec!(
        SpecDecodeTokenWeightedAcceptanceLength,
        "Token-Weighted Acceptance Length",
        Ratio,
        Derived,
        None,
        MetricFlags::LARGER_IS_BETTER,
        [TotalAcceptedDraftTokens, TotalSpecDecodeSteps]
    ),
    spec!(
        SpecDecodeDraftAcceptanceRate,
        "Draft Acceptance Rate",
        Percent,
        Record,
        None,
        MetricFlags::LARGER_IS_BETTER,
        []
    ),
    spec!(
        SpecDecodeOverallDraftAcceptanceRate,
        "Overall Draft Acceptance Rate",
        Percent,
        Derived,
        None,
        MetricFlags::LARGER_IS_BETTER,
        [TotalAcceptedDraftTokens, TotalDraftTokens]
    ),
    spec!(
        SpecDecodeAcceptedPerVerified,
        "Accepted per Verified",
        Ratio,
        Record,
        None,
        MetricFlags::LARGER_IS_BETTER,
        []
    ),
    spec!(
        SpecDecodeSteps,
        "Spec Decode Steps",
        Count,
        Record,
        None,
        MetricFlags::NONE,
        []
    ),
    spec!(
        SpecDecodeAcceptedDraftTokens,
        "Accepted Draft Tokens",
        Token,
        Record,
        None,
        MetricFlags::NONE,
        []
    ),
    spec!(
        SpecDecodeDraftTokens,
        "Draft Tokens",
        Token,
        Record,
        None,
        MetricFlags::NONE,
        []
    ),
    spec!(
        TotalSpecDecodeSteps,
        "Total Spec Decode Steps",
        Count,
        Derived,
        None,
        MetricFlags::NONE,
        [SpecDecodeSteps]
    ),
    spec!(
        TotalAcceptedDraftTokens,
        "Total Accepted Draft Tokens",
        Token,
        Derived,
        None,
        MetricFlags::NONE,
        [SpecDecodeAcceptedDraftTokens]
    ),
    spec!(
        TotalDraftTokens,
        "Total Draft Tokens",
        Token,
        Derived,
        None,
        MetricFlags::NONE,
        [SpecDecodeDraftTokens]
    ),
];

/// Returns the [`Definition`] for `tag` in O(1).
///
/// `CATALOG` is ordered by declaration discriminant (guarded by the
/// `catalog_is_discriminant_ordered` test), so `tag.index()` is the tag's own
/// row. The `[MetricSpec; MetricTag::COUNT]` array length makes a variant added
/// without a row a compile error, so this lookup is total.
pub const fn metric_definition(tag: MetricTag) -> &'static Definition {
    &CATALOG[tag.index()].def
}

/// Looks up a catalog spec in O(1).
pub const fn spec_for(tag: MetricTag) -> Option<&'static MetricSpec> {
    Some(&CATALOG[tag.index()])
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
            header: spec.header().to_string(),
            unit: spec
                .display_unit()
                .unwrap_or(spec.unit())
                .as_str()
                .to_string(),
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
    use super::{CATALOG, MetricFlags, MetricTag, metric_definition, spec_for, validate_catalog};

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
            feed(&mut hash, spec.def.header.as_bytes());
            feed(&mut hash, spec.def.short_header.unwrap_or("").as_bytes());
            feed(
                &mut hash,
                spec.def.short_header_hide_unit.to_string().as_bytes(),
            );
            feed(&mut hash, format!("{:?}", spec.def.unit).as_bytes());
            feed(&mut hash, format!("{:?}", spec.def.display_unit).as_bytes());
            feed(
                &mut hash,
                format!("{:?}", spec.def.display_order).as_bytes(),
            );
            feed(&mut hash, spec.flags.bits().to_string().as_bytes());
            feed(&mut hash, format!("{:?}", spec.console_group).as_bytes());
            feed(
                &mut hash,
                format!("{:?}", spec.def.larger_is_better).as_bytes(),
            );
            for dependency in spec.required {
                feed(&mut hash, dependency.as_str().as_bytes());
            }
            feed(&mut hash, format!("{:?}", spec.def.value_type).as_bytes());
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
        assert_eq!(CATALOG.len(), 138);
        assert!(order.contains(&MetricTag::RequestLatency));
        assert!(
            CATALOG
                .iter()
                .any(|spec| spec.tag == MetricTag::GoodRequestFraction)
        );
    }

    #[test]
    fn spec_decode_catalog_contract_is_canonical() {
        let grouped = [
            MetricTag::SpecDecodeAcceptanceLength,
            MetricTag::SpecDecodeTokenWeightedAcceptanceLength,
            MetricTag::SpecDecodeDraftAcceptanceRate,
            MetricTag::SpecDecodeOverallDraftAcceptanceRate,
            MetricTag::SpecDecodeAcceptedPerVerified,
            MetricTag::SpecDecodeSteps,
        ];
        for tag in grouped {
            assert_eq!(
                spec_for(tag).unwrap().console_group,
                super::MetricConsoleGroup::SpecDecode
            );
        }
        for tag in [
            MetricTag::SpecDecodeAcceptedDraftTokens,
            MetricTag::SpecDecodeDraftTokens,
            MetricTag::TotalSpecDecodeSteps,
            MetricTag::TotalAcceptedDraftTokens,
            MetricTag::TotalDraftTokens,
        ] {
            assert_eq!(
                spec_for(tag).unwrap().console_group,
                super::MetricConsoleGroup::None
            );
        }
        assert_eq!(MetricTag::COUNT, 138);
    }

    #[test]
    fn catalog_is_discriminant_ordered() {
        for (i, s) in CATALOG.iter().enumerate() {
            assert_eq!(s.tag as usize, i, "row {i} is {:?}", s.tag);
        }
    }

    #[test]
    fn websocket_lag_tags_append_after_existing_dense_identities() {
        assert_eq!(
            MetricTag::TimeToLastRoundTrip as usize,
            MetricTag::ActiveTotalThroughput as usize + 1
        );
        assert_eq!(
            MetricTag::AverageRoundTripTime as usize,
            MetricTag::ActiveTotalThroughput as usize + 2
        );
    }

    #[test]
    fn metric_definition_matches_catalog() {
        for s in CATALOG.iter() {
            assert!(std::ptr::eq(metric_definition(s.tag), &s.def));
        }
        assert_eq!(
            metric_definition(MetricTag::RequestLatency).id,
            MetricTag::RequestLatency.as_str()
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
        // Updated 2026-07-25: removed the legacy `plot_direction` MetricSpec field;
        // direction now derives from `def.larger_is_better`, which the fingerprint
        // hashes in its place (Task 7 of the definition-registry feature).
        assert_eq!(catalog_fingerprint(), 12_961_133_547_094_123_540);
    }

    #[test]
    fn plot_direction_derives_from_larger_is_better() {
        // Direction is no longer a stored field; it is computed from the
        // embedded definition's `larger_is_better` flag.
        let bigger = spec_for(MetricTag::OutputSequenceLength).unwrap();
        assert!(bigger.def.larger_is_better);
        assert_eq!(
            super::plot_direction_for(&bigger.def),
            super::PlotMetricDirection::LargerIsBetter
        );

        let smaller = spec_for(MetricTag::RequestLatency).unwrap();
        assert!(!smaller.def.larger_is_better);
        assert_eq!(
            super::plot_direction_for(&smaller.def),
            super::PlotMetricDirection::SmallerIsBetter
        );
    }

    #[test]
    fn metricspec_accessors_delegate_to_def() {
        let s = spec_for(MetricTag::RequestLatency).unwrap();
        assert_eq!(s.header(), s.def.header);
        assert_eq!(s.unit(), s.def.unit);
    }

    #[test]
    fn catalog_metadata_is_static_not_postmutated() {
        // OutputSequenceLength had value_type set by the old post-pass; the folded
        // construction must now carry it directly on `def` (previously left Float).
        let s = spec_for(MetricTag::OutputSequenceLength).unwrap();
        assert_eq!(s.def.value_type, super::MetricValueType::Int);
        assert!(s.def.larger_is_better);
    }
}
