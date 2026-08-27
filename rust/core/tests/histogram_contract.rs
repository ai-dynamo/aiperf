// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Golden test of the core-owned histogram vocabulary.
//!
//! Every constant here is a wire fact: a boundary array, a metric-source alias,
//! a unit conversion, or an attribute default that a collector on the other end
//! of an OTLP export parses. Changing one silently re-buckets a run's history,
//! so each is pinned literally rather than derived from the constant under test.
//!
//! The bucket-selection rule is exercised at the boundary value itself, because
//! `value <= bound` and `value < bound` differ only there and nowhere else.

use aiperf_core::histogram::{
    DEFAULT_OPERATION_NAME, DEFAULT_SERVICE_NAME, ENDPOINT_TYPE_RESOURCE_ATTRIBUTE,
    ERROR_TYPE_ATTRIBUTE, GEN_AI_HISTOGRAM_METRICS, GenAiHistogramMetric,
    INPUT_TOKEN_METRIC_SOURCES, OPERATION_DURATION_BOUNDS_SECONDS,
    OPERATION_DURATION_METRIC_SOURCES, OPERATION_NAME_ATTRIBUTE, OUTPUT_TOKEN_METRIC_SOURCES,
    PROVIDER_NAME_ATTRIBUTE, REQUEST_MODEL_ATTRIBUTE, SERVICE_NAME_RESOURCE_ATTRIBUTE,
    TIME_PER_OUTPUT_CHUNK_BOUNDS_SECONDS, TIME_PER_OUTPUT_CHUNK_METRIC_SOURCES,
    TIME_TO_FIRST_CHUNK_BOUNDS_SECONDS, TIME_TO_FIRST_CHUNK_METRIC_SOURCES, TOKEN_TYPE_ATTRIBUTE,
    TOKEN_USAGE_BOUNDS_TOKENS, TokenDirection, UNKNOWN_PROVIDER_NAME, bucket_index, is_observable,
    normalize_operation_name, normalize_provider_name, seconds_scale,
};

#[test]
fn explicit_bounds_arrays_are_pinned() {
    assert_eq!(
        OPERATION_DURATION_BOUNDS_SECONDS,
        &[
            0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56, 5.12, 10.24, 20.48, 40.96, 81.92
        ]
    );

    let latency_bounds = [
        0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2, 0.25, 0.3,
        0.35, 0.4, 0.45, 0.5, 0.75, 1.0, 2.0, 5.0,
    ];
    assert_eq!(TIME_TO_FIRST_CHUNK_BOUNDS_SECONDS, &latency_bounds);
    assert_eq!(TIME_PER_OUTPUT_CHUNK_BOUNDS_SECONDS, &latency_bounds);

    assert_eq!(
        TOKEN_USAGE_BOUNDS_TOKENS,
        &[
            1.0, 4.0, 16.0, 64.0, 256.0, 1024.0, 4096.0, 16384.0, 65536.0, 262144.0, 1048576.0,
            4194304.0, 16777216.0
        ]
    );

    for metric in GEN_AI_HISTOGRAM_METRICS {
        let bounds = metric.bounds();
        assert!(
            bounds.windows(2).all(|pair| pair[0] < pair[1]),
            "{} bounds must be strictly ascending",
            metric.spec_name()
        );
    }
}

#[test]
fn metric_names_units_and_sources_are_pinned() {
    let expected: [(GenAiHistogramMetric, &str, &str, &[&str]); 4] = [
        (
            GenAiHistogramMetric::OperationDuration,
            "gen_ai.client.operation.duration",
            "s",
            &["request_latency"],
        ),
        (
            GenAiHistogramMetric::TimeToFirstChunk,
            "gen_ai.client.operation.time_to_first_chunk",
            "s",
            &["time_to_first_token"],
        ),
        (
            GenAiHistogramMetric::TimePerOutputChunk,
            "gen_ai.client.operation.time_per_output_chunk",
            "s",
            &["inter_token_latency"],
        ),
        (
            GenAiHistogramMetric::TokenUsage,
            "gen_ai.client.token.usage",
            "{token}",
            &["input_token_count", "input_sequence_length"],
        ),
    ];

    assert_eq!(GEN_AI_HISTOGRAM_METRICS.len(), expected.len());
    for (index, (metric, spec_name, unit, sources)) in expected.into_iter().enumerate() {
        assert_eq!(GEN_AI_HISTOGRAM_METRICS[index], metric);
        assert_eq!(metric.spec_name(), spec_name);
        assert_eq!(metric.unit(), unit);
        assert_eq!(metric.metric_sources(), sources);
    }

    assert_eq!(OPERATION_DURATION_METRIC_SOURCES, &["request_latency"]);
    assert_eq!(TIME_TO_FIRST_CHUNK_METRIC_SOURCES, &["time_to_first_token"]);
    assert_eq!(
        TIME_PER_OUTPUT_CHUNK_METRIC_SOURCES,
        &["inter_token_latency"]
    );
    assert_eq!(
        INPUT_TOKEN_METRIC_SOURCES,
        &["input_token_count", "input_sequence_length"]
    );
    assert_eq!(
        OUTPUT_TOKEN_METRIC_SOURCES,
        &["output_token_count", "output_sequence_length"]
    );
    assert_eq!(
        TokenDirection::Input.metric_sources(),
        INPUT_TOKEN_METRIC_SOURCES
    );
    assert_eq!(
        TokenDirection::Output.metric_sources(),
        OUTPUT_TOKEN_METRIC_SOURCES
    );
    assert_eq!(TokenDirection::Input.attribute_value(), "input");
    assert_eq!(TokenDirection::Output.attribute_value(), "output");
}

#[test]
fn unit_conversions_are_pinned_and_unknown_units_are_identity() {
    assert_eq!(seconds_scale("ns"), 1e-9);
    assert_eq!(seconds_scale("us"), 1e-6);
    assert_eq!(seconds_scale("ms"), 1e-3);
    assert_eq!(seconds_scale("sec"), 1.0);
    assert_eq!(seconds_scale("s"), 1.0);
    // An unrecognized unit is treated as seconds: a wrong scale corrupts a
    // correct observation, an identity scale preserves it.
    assert_eq!(seconds_scale(""), 1.0);
    assert_eq!(seconds_scale("tokens"), 1.0);
    // Scaling is case-sensitive: "MS" is not a report display unit.
    assert_eq!(seconds_scale("MS"), 1.0);
}

#[test]
fn inclusion_rules_separate_durations_from_token_usage() {
    for metric in [
        GenAiHistogramMetric::OperationDuration,
        GenAiHistogramMetric::TimeToFirstChunk,
        GenAiHistogramMetric::TimePerOutputChunk,
    ] {
        assert!(metric.is_duration());
        assert!(metric.includes_error_type());
    }
    assert!(!GenAiHistogramMetric::TokenUsage.is_duration());
    assert!(!GenAiHistogramMetric::TokenUsage.includes_error_type());

    assert!(is_observable(0.0));
    assert!(is_observable(-1.0));
    assert!(!is_observable(f64::NAN));
    assert!(!is_observable(f64::INFINITY));
    assert!(!is_observable(f64::NEG_INFINITY));
}

#[test]
fn the_first_upper_bound_rule_admits_the_boundary_value() {
    let bounds = OPERATION_DURATION_BOUNDS_SECONDS;

    // `value <= bound`: the boundary value itself lands in its own bucket.
    assert_eq!(bucket_index(bounds, 0.01), 0);
    assert_eq!(bucket_index(bounds, 0.02), 1);
    assert_eq!(bucket_index(bounds, 81.92), bounds.len() - 1);

    // Just past a bound moves to the next bucket.
    assert_eq!(bucket_index(bounds, 0.010_000_1), 1);

    // Below every bound is the first bucket; above every bound is the overflow.
    assert_eq!(bucket_index(bounds, f64::MIN), 0);
    assert_eq!(bucket_index(bounds, 81.920_1), bounds.len());

    // Bucket-count vectors are `bounds.len() + 1` long, so the overflow index
    // is always addressable.
    for metric in GEN_AI_HISTOGRAM_METRICS {
        let metric_bounds = metric.bounds();
        assert!(bucket_index(metric_bounds, f64::MAX) < metric_bounds.len() + 1);
    }
}

#[test]
fn attribute_normalization_constants_are_pinned() {
    assert_eq!(OPERATION_NAME_ATTRIBUTE, "gen_ai.operation.name");
    assert_eq!(PROVIDER_NAME_ATTRIBUTE, "gen_ai.provider.name");
    assert_eq!(REQUEST_MODEL_ATTRIBUTE, "gen_ai.request.model");
    assert_eq!(TOKEN_TYPE_ATTRIBUTE, "gen_ai.token.type");
    assert_eq!(ERROR_TYPE_ATTRIBUTE, "error.type");
    assert_eq!(SERVICE_NAME_RESOURCE_ATTRIBUTE, "service.name");
    assert_eq!(DEFAULT_SERVICE_NAME, "aiperf");
    assert_eq!(ENDPOINT_TYPE_RESOURCE_ATTRIBUTE, "aiperf.endpoint.type");
    assert_eq!(UNKNOWN_PROVIDER_NAME, "_OTHER");
    assert_eq!(DEFAULT_OPERATION_NAME, "chat");

    assert_eq!(normalize_operation_name("completions"), "text_completion");
    assert_eq!(normalize_operation_name("COMPLETIONS"), "text_completion");
    assert_eq!(normalize_operation_name("embeddings"), "embeddings");
    // Every unmapped endpoint type folds to `chat` rather than leaking an
    // AIPerf-internal name into a semantic-convention attribute.
    assert_eq!(normalize_operation_name("rankings"), "chat");
    assert_eq!(normalize_operation_name(""), "chat");

    assert_eq!(normalize_provider_name(None), "_OTHER");
    assert_eq!(normalize_provider_name(Some("")), "_OTHER");
    assert_eq!(normalize_provider_name(Some("nvidia")), "nvidia");
}
