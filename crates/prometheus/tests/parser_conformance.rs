// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Public-API conformance and permanent parser regression fixtures.

use std::cmp::Ordering;

use aiperf_prometheus::NativeCompatibilityProjection;
use aiperf_prometheus::{
    ContentTypeError, CountOrigin, Exposition, ExpositionFormat, ExpositionParser, F64Status,
    InfoLabelPartitionStatus, LimitKind, MetricValue, NoopNativeCompatibilityProjection,
    NumberError, NumberKind, ParseError, ParseErrorKind, ParseLimits, PointTimeStatus,
    SemanticType, SourceTimestamp, StrictExpositionParser, TimestampStatus, WireSampleRole,
    parse_number_lexeme,
};

fn parse(format: ExpositionFormat, body: &str) -> Result<Exposition, ParseError> {
    StrictExpositionParser.parse(format, body.as_bytes(), &ParseLimits::default())
}

fn fixture(name: &str) -> &'static str {
    match name {
        "histogram" => include_str!("fixtures/tachometer_histogram_cross_labels.prom"),
        "quoted" => include_str!("fixtures/tachometer_quoted_label.prom"),
        "precision" => include_str!("fixtures/tachometer_float32_precision.prom"),
        _ => panic!("unknown fixture"),
    }
}

#[test]
fn content_type_selects_one_exact_grammar_without_fallback() {
    assert_eq!(
        ExpositionFormat::from_content_type("text/plain").unwrap(),
        ExpositionFormat::PrometheusText004
    );
    assert_eq!(
        ExpositionFormat::from_content_type("text/plain; version=\"0.0.4\"; charset=\"UTF-8\"")
            .unwrap(),
        ExpositionFormat::PrometheusText004
    );
    assert_eq!(
        ExpositionFormat::from_content_type(
            "application/openmetrics-text; version=1.0.0; charset=utf-8"
        )
        .unwrap(),
        ExpositionFormat::OpenMetricsText100
    );
    assert_eq!(
        ExpositionFormat::from_content_type("application/openmetrics-text; version=1.0.0"),
        Err(ContentTypeError::MissingParameter("charset"))
    );
    assert!(matches!(
        ExpositionFormat::from_content_type("text/plain; version=1.0.0"),
        Err(ContentTypeError::UnsupportedVersion { .. })
    ));
    assert!(matches!(
        ExpositionFormat::from_content_type("text/plain; charset=\"utf-8"),
        Err(ContentTypeError::MalformedParameter(_))
    ));
    assert!(matches!(
        ExpositionFormat::from_content_type("text/plain; q=1"),
        Err(ContentTypeError::UnsupportedParameter(_))
    ));
}

#[test]
fn openmetrics_all_types_metadata_and_wire_roles_are_preserved() {
    let body = concat!(
        "# TYPE mystery unknown\n",
        "# HELP mystery Unknown value.\n",
        "mystery NaN\n",
        "# TYPE temperature gauge\n",
        "temperature -Inf\n",
        "# TYPE process_cpu_seconds counter\n",
        "# UNIT process_cpu_seconds seconds\n",
        "# HELP process_cpu_seconds CPU time.\n",
        "process_cpu_seconds_total 9007199254740993\n",
        "process_cpu_seconds_created 1700000000.0000000001\n",
        "# TYPE service stateset\n",
        "service{node=\"a\",service=\"starting\"} 0\n",
        "service{node=\"a\",service=\"ready\"} 1\n",
        "# TYPE build info\n",
        "build_info{revision=\"abc\",version=\"1.2.3\"} 1\n",
        "# TYPE latency_seconds histogram\n",
        "# UNIT latency_seconds seconds\n",
        "latency_seconds_bucket{route=\"/v1\",le=\"0.1\"} 2 # {trace_id=\"a\"} 0.05\n",
        "latency_seconds_bucket{route=\"/v1\",le=\"+Inf\"} 3\n",
        "latency_seconds_count{route=\"/v1\"} 3\n",
        "latency_seconds_sum{route=\"/v1\"} 0.2\n",
        "latency_seconds_created{route=\"/v1\"} 1700000000\n",
        "# TYPE queue_seconds gaugehistogram\n",
        "# UNIT queue_seconds seconds\n",
        "queue_seconds_bucket{le=\"-1\"} 1\n",
        "queue_seconds_bucket{le=\"+Inf\"} 2\n",
        "queue_seconds_gsum -0.5\n",
        "# TYPE request_seconds summary\n",
        "# UNIT request_seconds seconds\n",
        "request_seconds{quantile=\"0.99\"} 2.5\n",
        "request_seconds{quantile=\"0.5\"} 1.0\n",
        "request_seconds_count 4.0\n",
        "request_seconds_sum 6.0\n",
        "request_seconds_created 1700000000\n",
        "# TYPE documented_only gauge\n",
        "# HELP documented_only Metadata-only family.\n",
        "# EOF\n",
    );
    let exposition = parse(ExpositionFormat::OpenMetricsText100, body).unwrap();
    assert_eq!(exposition.families.len(), 9);
    assert_eq!(exposition.metric_point_count(), 8);
    assert_eq!(exposition.wire_sample_count, 20);
    assert_eq!(
        exposition
            .families
            .iter()
            .take(8)
            .map(|family| family.semantic_type)
            .collect::<Vec<_>>(),
        vec![
            SemanticType::Unknown,
            SemanticType::Gauge,
            SemanticType::Counter,
            SemanticType::StateSet,
            SemanticType::Info,
            SemanticType::Histogram,
            SemanticType::GaugeHistogram,
            SemanticType::Summary,
        ]
    );
    let counter = &exposition.family("process_cpu_seconds").unwrap().metrics[0].points[0];
    let MetricValue::Counter(counter_value) = &counter.value else {
        panic!("expected counter")
    };
    assert_eq!(counter_value.total.f64_status, F64Status::Rounded);
    assert_eq!(
        counter_value.created.value.status,
        TimestampStatus::SubNanosecondPrecision
    );
    assert_eq!(
        counter
            .wire_samples
            .iter()
            .map(|wire| wire.role)
            .collect::<Vec<_>>(),
        vec![WireSampleRole::CounterTotal, WireSampleRole::CounterCreated]
    );

    let histogram = &exposition.family("latency_seconds").unwrap().metrics[0].points[0];
    let MetricValue::Histogram(histogram_value) = &histogram.value else {
        panic!("expected histogram")
    };
    assert_eq!(
        histogram_value.count_origin,
        CountOrigin::EmittedAndValidated
    );
    assert!(histogram_value.buckets[0].exemplar.is_some());

    let gauge_histogram = &exposition.family("queue_seconds").unwrap().metrics[0].points[0];
    let MetricValue::Histogram(gauge_histogram_value) = &gauge_histogram.value else {
        panic!("expected gauge histogram")
    };
    assert_eq!(
        gauge_histogram_value.count_origin,
        CountOrigin::DerivedFromPositiveInfinity
    );

    let summary = &exposition.family("request_seconds").unwrap().metrics[0].points[0];
    let MetricValue::Summary(summary_value) = &summary.value else {
        panic!("expected summary")
    };
    assert_eq!(
        summary_value
            .quantiles
            .iter()
            .map(|quantile| quantile.quantile_lexeme.as_str())
            .collect::<Vec<_>>(),
        vec!["0.5", "0.99"]
    );
    assert!(
        exposition
            .family("documented_only")
            .unwrap()
            .metrics
            .is_empty()
    );
}

#[test]
fn empty_and_metadata_only_metric_sets_are_successful() {
    let empty = parse(ExpositionFormat::OpenMetricsText100, "# EOF\n").unwrap();
    assert!(empty.families.is_empty());
    assert_eq!(empty.wire_sample_count, 0);

    let classic = parse(
        ExpositionFormat::PrometheusText004,
        "# HELP no_points Still useful.\n# TYPE no_points gauge\n",
    )
    .unwrap();
    let family = classic.family("no_points").unwrap();
    assert_eq!(family.help.as_ref().unwrap().value, "Still useful.");
    assert!(family.metrics.is_empty());
}

#[test]
fn repeated_points_and_exact_timestamp_equivalence_remain_distinct() {
    let gauge = parse(
        ExpositionFormat::OpenMetricsText100,
        "# TYPE temperature gauge\ntemperature{room=\"a\"} 20 1.0\ntemperature{room=\"a\"} 21 2.0\n# EOF\n",
    )
    .unwrap();
    let points = &gauge.family("temperature").unwrap().metrics[0].points;
    assert_eq!(points.len(), 2);
    assert_eq!(points[0].metric_point_seq, 2);
    assert_eq!(points[1].source_timestamp.lexeme.as_deref(), Some("2.0"));

    let histogram = parse(
        ExpositionFormat::OpenMetricsText100,
        concat!(
            "# TYPE latency histogram\n",
            "latency_bucket{le=\"+Inf\"} 1 10000000000000000000000000000000000000000.0000000001\n",
            "latency_count 1 10000000000000000000000000000000000000000.00000000010\n",
            "latency_created 10000000000000000000000000000000000000000.0000000001 10000000000000000000000000000000000000000.000000000100\n",
            "# EOF\n",
        ),
    )
    .unwrap();
    let point = &histogram.family("latency").unwrap().metrics[0].points[0];
    assert_eq!(point.point_time_status, PointTimeStatus::UniformExplicit);
    assert_eq!(
        point.source_timestamp.lexeme.as_deref(),
        Some("10000000000000000000000000000000000000000.0000000001")
    );
    assert_eq!(
        point.source_timestamp.status,
        TimestampStatus::SubNanosecondOutOfRange
    );
    let MetricValue::Histogram(value) = &point.value else {
        panic!("expected histogram")
    };
    assert_eq!(
        value.created.value.status,
        TimestampStatus::SubNanosecondOutOfRange
    );
}

#[test]
fn classic_all_timestamp_relationships_preserve_components() {
    let cases = [
        (
            "latency_bucket{le=\"+Inf\"} 1\nlatency_count 1\n",
            PointTimeStatus::AllAbsent,
        ),
        (
            "latency_bucket{le=\"+Inf\"} 1 001\nlatency_count 1 1\n",
            PointTimeStatus::UniformExplicit,
        ),
        (
            "latency_bucket{le=\"+Inf\"} 1 1\nlatency_count 1 2\n",
            PointTimeStatus::MixedComponents,
        ),
        (
            "latency_bucket{le=\"+Inf\"} 1 1\nlatency_count 1\n",
            PointTimeStatus::PartialComponents,
        ),
    ];
    for (samples, expected) in cases {
        let body = format!("# TYPE latency histogram\n{samples}");
        let exposition = parse(ExpositionFormat::PrometheusText004, &body).unwrap();
        let point = &exposition.family("latency").unwrap().metrics[0].points[0];
        assert_eq!(point.point_time_status, expected);
        if expected == PointTimeStatus::UniformExplicit {
            assert_eq!(point.source_timestamp.lexeme.as_deref(), Some("001"));
        } else {
            assert_eq!(point.source_timestamp, SourceTimestamp::absent());
        }
        assert_eq!(point.wire_samples.len(), 2);
    }
}

#[test]
fn exact_number_matrix_uses_binary64_once_and_keeps_integer_facts() {
    let tie = parse_number_lexeme(
        ExpositionFormat::OpenMetricsText100,
        "1.00000000000000011102230246251565404236316680908203125",
    )
    .unwrap();
    assert_eq!(tie.finite_value.unwrap().to_bits(), 1.0_f64.to_bits());
    assert_eq!(tie.f64_status, F64Status::Rounded);

    let signed_underflow =
        parse_number_lexeme(ExpositionFormat::OpenMetricsText100, "-1e-999999").unwrap();
    assert_eq!(
        signed_underflow.finite_value.unwrap().to_bits(),
        (-0.0_f64).to_bits()
    );
    let u64_max =
        parse_number_lexeme(ExpositionFormat::OpenMetricsText100, "18446744073709551615").unwrap();
    assert_eq!(u64_max.exact_u64, Some(u64::MAX));
    assert_eq!(u64_max.f64_status, F64Status::Rounded);

    let wider = parse_number_lexeme(
        ExpositionFormat::OpenMetricsText100,
        "10000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
    )
    .unwrap();
    assert_eq!(wider.kind, NumberKind::Finite);
    assert_eq!(wider.finite_value, None);
    assert_eq!(wider.f64_status, F64Status::Unavailable);
    assert_eq!(
        parse_number_lexeme(ExpositionFormat::OpenMetricsText100, "1e309"),
        Err(NumberError::Binary64Overflow)
    );

    let exponent_with_leading_zeroes = parse_number_lexeme(
        ExpositionFormat::OpenMetricsText100,
        &format!("1e{}1", "0".repeat(100)),
    )
    .unwrap();
    assert_eq!(exponent_with_leading_zeroes.finite_value, Some(10.0));
}

#[test]
fn semantic_role_matrix_rejects_invalid_points_atomically() {
    let invalid = [
        (
            "# TYPE build info\nbuild_info{version=\"1\"} 0\n# EOF\n",
            "Info",
        ),
        (
            "# TYPE service stateset\nservice{service=\"ready\"} 2\n# EOF\n",
            "StateSet",
        ),
        (
            "# TYPE requests counter\nrequests_total NaN\n# EOF\n",
            "counter",
        ),
        (
            "# TYPE latency histogram\nlatency_bucket{le=\"1\"} 1\n# EOF\n",
            "+Inf",
        ),
        (
            "# TYPE latency histogram\nlatency_bucket{le=\"1\"} 2\nlatency_bucket{le=\"+Inf\"} 1\n# EOF\n",
            "cumulative",
        ),
        (
            "# TYPE latency histogram\nlatency_bucket{le=\"+Inf\"} 1\nlatency_count 2\n# EOF\n",
            "must equal",
        ),
        (
            "# TYPE latency histogram\nlatency_bucket{le=\"-1\"} 1\nlatency_bucket{le=\"+Inf\"} 2\nlatency_sum 3\n# EOF\n",
            "negative bound",
        ),
        (
            "# TYPE latency summary\nlatency{quantile=\"1.1\"} 2\n# EOF\n",
            "[0, 1]",
        ),
        (
            "# TYPE latency summary\nlatency_count 1.5\n# EOF\n",
            "integer",
        ),
        (
            "# TYPE queue gaugehistogram\nqueue_bucket{le=\"+Inf\"} 1\nqueue_gsum NaN\n# EOF\n",
            "must not be NaN",
        ),
        (
            "# TYPE gauge gauge\ngauge 1 # {trace=\"x\"} 1\n# EOF\n",
            "cannot own",
        ),
    ];
    for (body, detail) in invalid {
        let error = parse(ExpositionFormat::OpenMetricsText100, body).unwrap_err();
        assert!(
            matches!(
                error.kind,
                ParseErrorKind::Semantic | ParseErrorKind::Exemplar
            ),
            "unexpected error for {detail}: {error:?}"
        );
        assert!(error.message.contains(detail), "{error:?}");
    }
}

#[test]
fn counter_nonfinite_exemplars_remain_tagged_while_bucket_nan_is_rejected() {
    let counter = parse(
        ExpositionFormat::OpenMetricsText100,
        "# TYPE requests counter\nrequests_total 1 # {trace=\"terminal\"} +Inf\n# EOF\n",
    )
    .unwrap();
    let point = &counter.family("requests").unwrap().metrics[0].points[0];
    assert_eq!(
        point.wire_samples[0].exemplar.as_ref().unwrap().value.kind,
        NumberKind::PositiveInfinity
    );

    let bucket_error = parse(
        ExpositionFormat::OpenMetricsText100,
        "# TYPE latency histogram\nlatency_bucket{le=\"+Inf\"} 1 # {} NaN\n# EOF\n",
    )
    .unwrap_err();
    assert_eq!(bucket_error.kind, ParseErrorKind::Exemplar);
}

#[test]
fn metadata_interleaving_collisions_and_timestamp_violations_are_typed() {
    let duplicate = parse(
        ExpositionFormat::OpenMetricsText100,
        "# TYPE value gauge\n# TYPE value gauge\nvalue 1\n# EOF\n",
    )
    .unwrap_err();
    assert_eq!(duplicate.kind, ParseErrorKind::Metadata);

    let bad_unit = parse(
        ExpositionFormat::OpenMetricsText100,
        "# TYPE value gauge\n# UNIT value seconds\nvalue 1\n# EOF\n",
    )
    .unwrap_err();
    assert_eq!(bad_unit.kind, ParseErrorKind::Metadata);

    let collision = parse(
        ExpositionFormat::OpenMetricsText100,
        "# TYPE foo counter\nfoo_total 1\n# TYPE foo_created gauge\n# EOF\n",
    )
    .unwrap_err();
    assert_eq!(collision.kind, ParseErrorKind::Semantic);
    assert!(collision.message.contains("collide"));

    let family_interleaving = parse(
        ExpositionFormat::OpenMetricsText100,
        "# TYPE a gauge\na 1\n# TYPE b gauge\nb 2\na 3\n# EOF\n",
    )
    .unwrap_err();
    assert_eq!(family_interleaving.kind, ParseErrorKind::Semantic);

    let mixed = parse(
        ExpositionFormat::OpenMetricsText100,
        "# TYPE latency histogram\nlatency_bucket{le=\"+Inf\"} 1 1\nlatency_count 1 2\n# EOF\n",
    )
    .unwrap_err();
    assert_eq!(mixed.kind, ParseErrorKind::Semantic);

    let non_monotonic = parse(
        ExpositionFormat::OpenMetricsText100,
        "# TYPE value gauge\nvalue 1 2\nvalue 2 1\n# EOF\n",
    )
    .unwrap_err();
    assert_eq!(non_monotonic.kind, ParseErrorKind::Semantic);
}

#[test]
fn configured_bounds_fail_before_any_partial_document_escapes() {
    let parser = StrictExpositionParser;
    let cases = [
        (
            ParseLimits {
                max_decoded_bytes: 3,
                ..ParseLimits::default()
            },
            "value 1\n",
            LimitKind::DecodedBytes,
        ),
        (
            ParseLimits {
                max_lines: 1,
                ..ParseLimits::default()
            },
            "a 1\nb 2\n",
            LimitKind::Lines,
        ),
        (
            ParseLimits {
                max_line_bytes: 3,
                ..ParseLimits::default()
            },
            "value 1\n",
            LimitKind::LineBytes,
        ),
        (
            ParseLimits {
                max_labels_per_sample: 1,
                ..ParseLimits::default()
            },
            "value{a=\"1\",b=\"2\"} 1\n",
            LimitKind::LabelsPerSample,
        ),
        (
            ParseLimits {
                max_wire_samples: 1,
                ..ParseLimits::default()
            },
            "a 1\nb 2\n",
            LimitKind::WireSamples,
        ),
        (
            ParseLimits {
                max_families: 1,
                ..ParseLimits::default()
            },
            "a 1\nb 2\n",
            LimitKind::Families,
        ),
    ];
    for (limits, body, kind) in cases {
        let error = parser
            .parse(
                ExpositionFormat::PrometheusText004,
                body.as_bytes(),
                &limits,
            )
            .unwrap_err();
        assert_eq!(error.kind, ParseErrorKind::LimitExceeded(kind));
    }

    let bucket_limits = ParseLimits {
        max_buckets_per_point: 1,
        ..ParseLimits::default()
    };
    let bucket_error = parser
        .parse(
            ExpositionFormat::OpenMetricsText100,
            b"# TYPE latency histogram\nlatency_bucket{le=\"1\"} 0\nlatency_bucket{le=\"+Inf\"} 1\n# EOF\n",
            &bucket_limits,
        )
        .unwrap_err();
    assert_eq!(
        bucket_error.kind,
        ParseErrorKind::LimitExceeded(LimitKind::BucketsPerPoint)
    );

    let bounded_cases = [
        (
            ParseLimits {
                max_metric_name_bytes: 1,
                ..ParseLimits::default()
            },
            "long 1\n",
            LimitKind::MetricNameBytes,
        ),
        (
            ParseLimits {
                max_label_name_bytes: 1,
                ..ParseLimits::default()
            },
            "a{long=\"1\"} 1\n",
            LimitKind::LabelNameBytes,
        ),
        (
            ParseLimits {
                max_label_value_bytes: 1,
                ..ParseLimits::default()
            },
            "a{x=\"long\"} 1\n",
            LimitKind::LabelValueBytes,
        ),
        (
            ParseLimits {
                max_metadata_value_bytes: 1,
                ..ParseLimits::default()
            },
            "# HELP a long\na 1\n",
            LimitKind::MetadataValueBytes,
        ),
        (
            ParseLimits {
                max_numeric_lexeme_bytes: 1,
                ..ParseLimits::default()
            },
            "a 10\n",
            LimitKind::NumericLexemeBytes,
        ),
        (
            ParseLimits {
                max_metrics: 1,
                ..ParseLimits::default()
            },
            "# TYPE a gauge\na{x=\"1\"} 1\na{x=\"2\"} 2\n",
            LimitKind::Metrics,
        ),
        (
            ParseLimits {
                max_metric_points: 1,
                ..ParseLimits::default()
            },
            "# TYPE a gauge\na{x=\"1\"} 1\na{x=\"1\"} 2\n",
            LimitKind::MetricPoints,
        ),
    ];
    for (limits, body, kind) in bounded_cases {
        let error = parser
            .parse(
                ExpositionFormat::PrometheusText004,
                body.as_bytes(),
                &limits,
            )
            .unwrap_err();
        assert_eq!(error.kind, ParseErrorKind::LimitExceeded(kind));
    }

    let openmetrics_bounds = [
        (
            ParseLimits {
                max_quantiles_per_point: 1,
                ..ParseLimits::default()
            },
            "# TYPE a summary\na{quantile=\"0.5\"} 1\na{quantile=\"0.9\"} 2\n# EOF\n",
            LimitKind::QuantilesPerPoint,
        ),
        (
            ParseLimits {
                max_states_per_point: 1,
                ..ParseLimits::default()
            },
            "# TYPE a stateset\na{a=\"off\"} 0\na{a=\"on\"} 1\n# EOF\n",
            LimitKind::StatesPerPoint,
        ),
        (
            ParseLimits {
                max_exemplars: 1,
                ..ParseLimits::default()
            },
            "# TYPE a counter\na_total{x=\"1\"} 1 # {trace=\"a\"} 1\na_total{x=\"2\"} 2 # {trace=\"b\"} 2\n# EOF\n",
            LimitKind::Exemplars,
        ),
        (
            ParseLimits {
                max_exemplar_labels: 1,
                ..ParseLimits::default()
            },
            "# TYPE a counter\na_total 1 # {a=\"1\",b=\"2\"} 1\n# EOF\n",
            LimitKind::ExemplarLabels,
        ),
        (
            ParseLimits {
                max_exemplar_label_codepoints: 1,
                ..ParseLimits::default()
            },
            "# TYPE a counter\na_total 1 # {trace=\"x\"} 1\n# EOF\n",
            LimitKind::ExemplarLabelCodepoints,
        ),
    ];
    for (limits, body, kind) in openmetrics_bounds {
        let error = parser
            .parse(
                ExpositionFormat::OpenMetricsText100,
                body.as_bytes(),
                &limits,
            )
            .unwrap_err();
        assert_eq!(error.kind, ParseErrorKind::LimitExceeded(kind));
    }
}

#[test]
fn malformed_utf8_labels_and_eof_fail_the_complete_parse() {
    let parser = StrictExpositionParser;
    let invalid_utf8 = parser
        .parse(
            ExpositionFormat::PrometheusText004,
            b"ok 1\nbad{label=\"\xff\"} 2\n",
            &ParseLimits::default(),
        )
        .unwrap_err();
    assert_eq!(invalid_utf8.kind, ParseErrorKind::InvalidUtf8);

    let duplicate_label = parse(
        ExpositionFormat::PrometheusText004,
        "ok 1\nbad{label=\"a\",label=\"b\"} 2\n",
    )
    .unwrap_err();
    assert_eq!(duplicate_label.kind, ParseErrorKind::Label);
    assert_eq!(duplicate_label.line, 2);

    let missing_eof = parse(ExpositionFormat::OpenMetricsText100, "value 1\n").unwrap_err();
    assert_eq!(missing_eof.kind, ParseErrorKind::EndOfFile);
}

#[test]
fn tachometer_histogram_cross_label_contamination_is_regressed() {
    let exposition = parse(ExpositionFormat::PrometheusText004, fixture("histogram")).unwrap();
    let family = exposition
        .family("vllm_request_queue_time_seconds")
        .unwrap();
    assert_eq!(family.metrics.len(), 2);
    let model_a = family
        .metrics
        .iter()
        .find(|metric| metric.labels["model_name"] == "model-a")
        .unwrap();
    let model_b = family
        .metrics
        .iter()
        .find(|metric| metric.labels["model_name"] == "model-b")
        .unwrap();
    let MetricValue::Histogram(model_a_value) = &model_a.points[0].value else {
        panic!("expected histogram")
    };
    let MetricValue::Histogram(model_b_value) = &model_b.points[0].value else {
        panic!("expected histogram")
    };
    assert_eq!(model_a_value.count.source_lexeme.as_deref(), Some("3"));
    assert_eq!(model_b_value.count.source_lexeme.as_deref(), Some("11"));
    assert_eq!(
        model_a_value.buckets[0]
            .cumulative_count
            .finite_cmp(&model_b_value.buckets[0].cumulative_count),
        Some(Ordering::Less)
    );
}

#[test]
fn tachometer_quoted_label_truncation_is_regressed() {
    let exposition = parse(ExpositionFormat::PrometheusText004, fixture("quoted")).unwrap();
    let metric = &exposition
        .family("vllm:num_requests_running")
        .unwrap()
        .metrics[0];
    assert_eq!(
        metric.labels["model_name"],
        "meta-llama/Llama-3.1-8B, revision=\"prod\", path=C:\\models"
    );
}

#[test]
fn classic_whitespace_and_trailing_label_comma_follow_text_004() {
    let exposition = parse(
        ExpositionFormat::PrometheusText004,
        "  #\tHELP\tvalue\tcomma label  \n  # HELPful is only a comment\n  value{ first = \"a,b\" , second=\"c\", }  3  \t\n",
    )
    .unwrap();
    let metric = &exposition.family("value").unwrap().metrics[0];
    assert_eq!(metric.labels["first"], "a,b");
    assert_eq!(metric.labels["second"], "c");
    assert_eq!(
        exposition
            .family("value")
            .unwrap()
            .help
            .as_ref()
            .unwrap()
            .value,
        "comma label"
    );
}

#[test]
fn tachometer_float32_precision_loss_is_regressed() {
    let exposition = parse(ExpositionFormat::PrometheusText004, fixture("precision")).unwrap();
    let point = &exposition
        .family("vllm:num_requests_running")
        .unwrap()
        .metrics[0]
        .points[0];
    let MetricValue::Scalar { value, .. } = &point.value else {
        panic!("expected scalar")
    };
    assert_eq!(value.source_lexeme.as_deref(), Some("100000001"));
    assert_eq!(value.finite_value, Some(100_000_001.0_f64));
    assert_eq!(value.f64_status, F64Status::Exact);
}

#[test]
fn info_identity_and_native_compatibility_are_explicitly_separate() {
    let exposition = parse(
        ExpositionFormat::OpenMetricsText100,
        "# TYPE build info\nbuild_info{entity=\"runner\",revision=\"abc\"} 1\n# EOF\n",
    )
    .unwrap();
    let point = &exposition.family("build").unwrap().metrics[0].points[0];
    let MetricValue::Info(info) = &point.value else {
        panic!("expected info")
    };
    assert_eq!(info.wire_merged_labels, point.labels);
    assert_eq!(
        info.partition_status,
        InfoLabelPartitionStatus::UnavailableFromText
    );
    assert_eq!(
        NoopNativeCompatibilityProjection
            .project(&exposition)
            .unwrap(),
        None
    );
}
