// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The one histogram vocabulary every exporter shares.
//!
//! Before this module the OpenTelemetry GenAI bucket boundaries, the report-key
//! aliases that feed them, the display-unit-to-seconds conversion, and the
//! `_OTHER`/`chat` attribute defaults lived inside a single exporter. Any second
//! exporter that wanted the same histograms had to copy them, and a copy drifts.
//! The vocabulary is boundary-owned here instead: an exporter plugin selects a
//! [`GenAiHistogramMetric`] and reads its bounds, unit, aliases, and inclusion
//! rule rather than declaring its own.
//!
//! Bucketing follows OTLP `le` semantics exactly: an observation `value` lands
//! in the first bucket whose upper bound satisfies `value <= bound`, and in the
//! trailing overflow bucket when no bound accepts it. [`bucket_index`] is the
//! single implementation of that rule.

/// The GenAI client histograms AIPerf emits, in stable emission order.
///
/// The three duration metrics are separate metric streams; the two token
/// directions are data points of one merged `gen_ai.client.token.usage` stream,
/// so token usage appears once here and is discriminated by attribute.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum GenAiHistogramMetric {
    /// End-to-end request duration.
    OperationDuration,
    /// Time from dispatch to the first streamed chunk.
    TimeToFirstChunk,
    /// Inter-chunk latency across a streamed response.
    TimePerOutputChunk,
    /// Input and output token counts.
    TokenUsage,
}

/// Every histogram metric in emission order.
pub const GEN_AI_HISTOGRAM_METRICS: &[GenAiHistogramMetric] = &[
    GenAiHistogramMetric::OperationDuration,
    GenAiHistogramMetric::TimeToFirstChunk,
    GenAiHistogramMetric::TimePerOutputChunk,
    GenAiHistogramMetric::TokenUsage,
];

/// Explicit boundaries for `gen_ai.client.operation.duration`, in seconds.
pub const OPERATION_DURATION_BOUNDS_SECONDS: &[f64] = &[
    0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56, 5.12, 10.24, 20.48, 40.96, 81.92,
];

/// Explicit boundaries for `gen_ai.client.operation.time_to_first_chunk`, in seconds.
pub const TIME_TO_FIRST_CHUNK_BOUNDS_SECONDS: &[f64] = &[
    0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2, 0.25, 0.3, 0.35,
    0.4, 0.45, 0.5, 0.75, 1.0, 2.0, 5.0,
];

/// Explicit boundaries for `gen_ai.client.operation.time_per_output_chunk`, in seconds.
pub const TIME_PER_OUTPUT_CHUNK_BOUNDS_SECONDS: &[f64] = &[
    0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2, 0.25, 0.3, 0.35,
    0.4, 0.45, 0.5, 0.75, 1.0, 2.0, 5.0,
];

/// Explicit boundaries for `gen_ai.client.token.usage`, in tokens.
pub const TOKEN_USAGE_BOUNDS_TOKENS: &[f64] = &[
    1.0, 4.0, 16.0, 64.0, 256.0, 1024.0, 4096.0, 16384.0, 65536.0, 262144.0, 1048576.0, 4194304.0,
    16777216.0,
];

/// Report metric keys that source `gen_ai.client.operation.duration`.
pub const OPERATION_DURATION_METRIC_SOURCES: &[&str] = &["request_latency"];

/// Report metric keys that source `gen_ai.client.operation.time_to_first_chunk`.
pub const TIME_TO_FIRST_CHUNK_METRIC_SOURCES: &[&str] = &["time_to_first_token"];

/// Report metric keys that source `gen_ai.client.operation.time_per_output_chunk`.
pub const TIME_PER_OUTPUT_CHUNK_METRIC_SOURCES: &[&str] = &["inter_token_latency"];

/// Report metric keys that source the `gen_ai.token.type=input` data point,
/// in first-present resolution order.
pub const INPUT_TOKEN_METRIC_SOURCES: &[&str] = &["input_token_count", "input_sequence_length"];

/// Report metric keys that source the `gen_ai.token.type=output` data point,
/// in first-present resolution order.
pub const OUTPUT_TOKEN_METRIC_SOURCES: &[&str] = &["output_token_count", "output_sequence_length"];

/// Semantic-convention unit for the three duration histograms.
pub const DURATION_UNIT: &str = "s";

/// Semantic-convention unit for the token-usage histogram.
pub const TOKEN_USAGE_UNIT: &str = "{token}";

impl GenAiHistogramMetric {
    /// The OpenTelemetry semantic-convention metric name.
    pub const fn spec_name(self) -> &'static str {
        match self {
            Self::OperationDuration => "gen_ai.client.operation.duration",
            Self::TimeToFirstChunk => "gen_ai.client.operation.time_to_first_chunk",
            Self::TimePerOutputChunk => "gen_ai.client.operation.time_per_output_chunk",
            Self::TokenUsage => "gen_ai.client.token.usage",
        }
    }

    /// The semantic-convention unit emitted with the metric.
    pub const fn unit(self) -> &'static str {
        match self {
            Self::OperationDuration | Self::TimeToFirstChunk | Self::TimePerOutputChunk => {
                DURATION_UNIT
            }
            Self::TokenUsage => TOKEN_USAGE_UNIT,
        }
    }

    /// The explicit bucket boundaries for the metric.
    pub const fn bounds(self) -> &'static [f64] {
        match self {
            Self::OperationDuration => OPERATION_DURATION_BOUNDS_SECONDS,
            Self::TimeToFirstChunk => TIME_TO_FIRST_CHUNK_BOUNDS_SECONDS,
            Self::TimePerOutputChunk => TIME_PER_OUTPUT_CHUNK_BOUNDS_SECONDS,
            Self::TokenUsage => TOKEN_USAGE_BOUNDS_TOKENS,
        }
    }

    /// The report metric keys that source the metric, in resolution order.
    ///
    /// Token usage returns the input aliases; the output direction is a separate
    /// data point sourced from [`OUTPUT_TOKEN_METRIC_SOURCES`].
    pub const fn metric_sources(self) -> &'static [&'static str] {
        match self {
            Self::OperationDuration => OPERATION_DURATION_METRIC_SOURCES,
            Self::TimeToFirstChunk => TIME_TO_FIRST_CHUNK_METRIC_SOURCES,
            Self::TimePerOutputChunk => TIME_PER_OUTPUT_CHUNK_METRIC_SOURCES,
            Self::TokenUsage => INPUT_TOKEN_METRIC_SOURCES,
        }
    }

    /// Whether observations are scaled from the report's display unit into
    /// seconds before bucketing.
    ///
    /// Token counts are dimensionless and are never scaled.
    pub const fn is_duration(self) -> bool {
        !matches!(self, Self::TokenUsage)
    }

    /// Whether the metric's data points carry the record's `error.type`.
    ///
    /// Only the duration histograms discriminate on failure: token usage carries
    /// `gen_ai.token.type` alone.
    pub const fn includes_error_type(self) -> bool {
        self.is_duration()
    }
}

/// The two token directions merged into `gen_ai.client.token.usage`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TokenDirection {
    /// `gen_ai.token.type=input`.
    Input,
    /// `gen_ai.token.type=output`.
    Output,
}

impl TokenDirection {
    /// The `gen_ai.token.type` attribute value.
    pub const fn attribute_value(self) -> &'static str {
        match self {
            Self::Input => "input",
            Self::Output => "output",
        }
    }

    /// The report metric keys that source this direction, in resolution order.
    pub const fn metric_sources(self) -> &'static [&'static str] {
        match self {
            Self::Input => INPUT_TOKEN_METRIC_SOURCES,
            Self::Output => OUTPUT_TOKEN_METRIC_SOURCES,
        }
    }
}

/// Attribute key carrying the GenAI operation name.
pub const OPERATION_NAME_ATTRIBUTE: &str = "gen_ai.operation.name";

/// Attribute key carrying the GenAI provider name.
pub const PROVIDER_NAME_ATTRIBUTE: &str = "gen_ai.provider.name";

/// Attribute key carrying the requested model.
pub const REQUEST_MODEL_ATTRIBUTE: &str = "gen_ai.request.model";

/// Attribute key carrying the token direction on `gen_ai.client.token.usage`.
pub const TOKEN_TYPE_ATTRIBUTE: &str = "gen_ai.token.type";

/// Attribute key carrying the record's failure classification.
pub const ERROR_TYPE_ATTRIBUTE: &str = "error.type";

/// Resource attribute key naming the run's service.
pub const SERVICE_NAME_RESOURCE_ATTRIBUTE: &str = "service.name";

/// Default `service.name` when the run authored none.
pub const DEFAULT_SERVICE_NAME: &str = "aiperf";

/// Resource attribute key the operation-name normalization reads.
pub const ENDPOINT_TYPE_RESOURCE_ATTRIBUTE: &str = "aiperf.endpoint.type";

/// The semantic-convention value for an unmapped provider.
pub const UNKNOWN_PROVIDER_NAME: &str = "_OTHER";

/// The operation name used when the endpoint type maps to no other name.
pub const DEFAULT_OPERATION_NAME: &str = "chat";

/// Normalize an authored endpoint type into a GenAI operation name.
///
/// Matching is ASCII-case-insensitive, and every unmapped endpoint type folds to
/// [`DEFAULT_OPERATION_NAME`] rather than leaking an AIPerf-internal name into a
/// semantic-convention attribute.
pub fn normalize_operation_name(endpoint_type: &str) -> &'static str {
    if endpoint_type.eq_ignore_ascii_case("completions") {
        "text_completion"
    } else if endpoint_type.eq_ignore_ascii_case("embeddings") {
        "embeddings"
    } else {
        DEFAULT_OPERATION_NAME
    }
}

/// Normalize an optional authored provider override.
///
/// An absent or empty override folds to [`UNKNOWN_PROVIDER_NAME`].
pub fn normalize_provider_name(provider: Option<&str>) -> &str {
    match provider {
        Some(value) if !value.is_empty() => value,
        _ => UNKNOWN_PROVIDER_NAME,
    }
}

/// Scale one value from a report display unit into seconds.
///
/// An unrecognized unit is treated as already in seconds: a wrong scale would
/// corrupt an otherwise correct observation, while an identity scale preserves
/// it. Only duration metrics are scaled; see
/// [`GenAiHistogramMetric::is_duration`].
pub fn seconds_scale(unit: &str) -> f64 {
    match unit {
        "ns" => 1e-9,
        "us" => 1e-6,
        "ms" => 1e-3,
        "sec" | "s" => 1.0,
        _ => 1.0,
    }
}

/// Whether an observation is admitted into a histogram at all.
///
/// Non-finite values are dropped: they have no bucket, and admitting one would
/// poison the stream's `sum`, `min`, and `max` for the whole run.
pub fn is_observable(value: f64) -> bool {
    value.is_finite()
}

/// Select the bucket for one observation under OTLP `le` semantics.
///
/// Returns the index of the first bound satisfying `value <= bound`, or
/// `bounds.len()` for the trailing overflow bucket. A bucket-count vector is
/// always `bounds.len() + 1` long, so the returned index is always in range.
pub fn bucket_index(bounds: &[f64], value: f64) -> usize {
    bounds
        .iter()
        .position(|bound| value <= *bound)
        .unwrap_or(bounds.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_upper_bound_wins_at_the_boundary() {
        let bounds = [1.0, 2.0, 4.0];
        assert_eq!(bucket_index(&bounds, 1.0), 0);
        assert_eq!(bucket_index(&bounds, 1.000_001), 1);
        assert_eq!(bucket_index(&bounds, 4.0), 2);
        assert_eq!(bucket_index(&bounds, 4.000_001), 3);
    }

    #[test]
    fn only_duration_metrics_scale_and_discriminate_on_error() {
        assert!(GenAiHistogramMetric::OperationDuration.includes_error_type());
        assert!(!GenAiHistogramMetric::TokenUsage.includes_error_type());
        assert_eq!(seconds_scale("ms"), 1e-3);
        assert_eq!(seconds_scale("furlongs"), 1.0);
    }

    #[test]
    fn attribute_normalization_folds_unmapped_inputs() {
        assert_eq!(normalize_operation_name("Completions"), "text_completion");
        assert_eq!(normalize_operation_name("rankings"), DEFAULT_OPERATION_NAME);
        assert_eq!(normalize_provider_name(None), UNKNOWN_PROVIDER_NAME);
        assert_eq!(normalize_provider_name(Some("")), UNKNOWN_PROVIDER_NAME);
        assert_eq!(normalize_provider_name(Some("nvidia")), "nvidia");
    }
}
