// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Golden parity tests for the AIPerf v1 summary sink.
//!
//! Every golden string was produced by running the Python oracle serializers
//! (`orjson.dumps(..., OPT_INDENT_2)` and stdlib `csv.writer`) over the same
//! synthetic metric values these tests feed the Rust sink, so the fixtures pin
//! the exact bytes AIPerf's Python exporters emit. Grounding of each shape is
//! cited against `src/aiperf/` in the module-level docs of `genai_perf.rs`.

use super::*;
use crate::metrics_core::{
    AccumulatorSummary, MetricEntry, MetricSeries, NativeReport, ReportCounterStats,
    ReportDistributionStats, ReportScalarStats, ReportStats, ReportValue,
};
use std::collections::BTreeMap;

/// A finite report value.
fn fin(value: f64) -> ReportValue {
    ReportValue::Finite(value)
}

/// Build a metric entry from one unlabeled series.
fn entry(metric_type: &'static str, unit: &str, stats: ReportStats) -> MetricEntry {
    MetricEntry {
        metric_type,
        unit: unit.to_owned(),
        group: "default",
        higher_is_better: false,
        series: vec![MetricSeries {
            labels: None,
            endpoint_url: None,
            stats,
            timeslices: Vec::new(),
        }],
    }
}

/// Build a distribution metric from its full stat set.
fn dist(
    unit: &str,
    count: usize,
    avg: Option<ReportValue>,
    min: Option<ReportValue>,
    max: Option<ReportValue>,
    std: Option<ReportValue>,
    percentiles: BTreeMap<String, ReportValue>,
) -> MetricEntry {
    entry(
        "distribution",
        unit,
        ReportStats::Distribution(ReportDistributionStats {
            count: Some(count),
            avg,
            min,
            max,
            std,
            percentiles,
        }),
    )
}

/// Nine ascending percentiles keyed `p1..p99` (all finite).
fn pcts(values: [f64; 9]) -> BTreeMap<String, ReportValue> {
    PERCENTILE_LABELS
        .iter()
        .zip(values)
        .map(|(label, value)| ((*label).to_owned(), fin(value)))
        .collect()
}

/// A scalar (derived) metric.
fn scalar(unit: &str, value: f64) -> MetricEntry {
    entry(
        "scalar",
        unit,
        ReportStats::Scalar(ReportScalarStats { value: fin(value) }),
    )
}

/// A counter (sum-aggregate) metric.
fn counter(unit: &str, total: f64) -> MetricEntry {
    entry(
        "counter",
        unit,
        ReportStats::Counter(ReportCounterStats {
            total: fin(total),
            rate: None,
        }),
    )
}

/// A deterministic base report with the given metrics.
fn report_with(metrics: BTreeMap<String, MetricEntry>) -> NativeReport {
    let mut report = NativeReport::new(&AccumulatorSummary::new(), None);
    // Pin the version so the golden strings are stable across releases.
    report.aiperf_version = "0.0.0-test".to_owned();
    report.metrics = metrics;
    report.summary.was_cancelled = false;
    report
}

/// Export config with the sink enabled and the given policy fields.
fn cfg(stem: &str, endpoint_type: &str, streaming: bool, goodput: bool) -> ExportConfig {
    ExportConfig {
        genai_perf: GenaiPerfExportConfig {
            enabled: true,
            stem: stem.to_owned(),
            goodput,
            streaming,
            endpoint_type: endpoint_type.to_owned(),
        },
        ..ExportConfig::default()
    }
}

/// The representative streaming + goodput fixture: distribution (with count),
/// streaming distribution, derived scalars, a counter, plus an INTERNAL metric
/// that must be filtered out (`min_request_timestamp`).
fn streaming_report() -> NativeReport {
    let metrics = BTreeMap::from([
        (
            "request_latency".to_owned(),
            dist(
                "ms",
                3,
                Some(fin(200.0)),
                Some(fin(100.0)),
                Some(fin(300.0)),
                Some(fin(50.25)),
                pcts([
                    101.0, 105.0, 110.0, 125.0, 150.0, 175.0, 190.0, 195.0, 199.0,
                ]),
            ),
        ),
        (
            "time_to_first_token".to_owned(),
            dist(
                "ms",
                3,
                Some(fin(10.0)),
                Some(fin(5.0)),
                Some(fin(15.0)),
                Some(fin(2.5)),
                pcts([5.5, 6.0, 6.5, 7.5, 10.0, 12.5, 14.0, 14.5, 14.9]),
            ),
        ),
        ("request_throughput".to_owned(), scalar("requests/sec", 4.0)),
        ("request_count".to_owned(), counter("requests", 3.0)),
        ("goodput".to_owned(), scalar("requests/sec", 2.0)),
        // INTERNAL flag -> dropped from both artifacts.
        ("min_request_timestamp".to_owned(), scalar("ns", 123.0)),
    ]);
    report_with(metrics)
}

#[test]
fn json_matches_python_oracle_streaming_goodput() {
    let report = streaming_report();
    let json = render_json(&report, &cfg("profile_export", "chat", true, true));
    assert_eq!(json, include_str!("../golden/v1_streaming.json"));
    // INTERNAL metric was filtered from the JSON entirely.
    assert!(!json.contains("min_request_timestamp"));
}

#[test]
fn csv_matches_python_oracle_streaming_goodput() {
    let report = streaming_report();
    let csv = render_csv(&report).unwrap();
    let expected = "Metric,avg,min,max,sum,p1,p5,p10,p25,p50,p75,p90,p95,p99,std\r\n\
Request Latency (ms),200.00,100.00,300.00,,101.00,105.00,110.00,125.00,150.00,175.00,190.00,195.00,199.00,50.25\r\n\
Time to First Token (ms),10.00,5.00,15.00,,5.50,6.00,6.50,7.50,10.00,12.50,14.00,14.50,14.90,2.50\r\n\
\r\n\
Metric,Value\r\n\
Goodput (requests/sec),2.00\r\n\
Request Count,3.00\r\n\
Request Throughput (requests/sec),4.00\r\n";
    assert_eq!(csv, expected);
    assert!(!csv.contains("Minimum Request Timestamp"));
}

#[test]
fn non_finite_tail_is_omitted_from_json_and_blank_in_csv() {
    // avg / max / std / p99 are non-finite (present-but-null in the native
    // report) -> Python collapses them to None before exclude_none, so they are
    // ABSENT in JSON and empty in CSV; p50 / min / count survive.
    let mut percentiles = BTreeMap::new();
    percentiles.insert("p50".to_owned(), fin(150.0));
    percentiles.insert("p99".to_owned(), ReportValue::NonFinite);
    let metrics = BTreeMap::from([(
        "request_latency".to_owned(),
        dist(
            "ms",
            2,
            Some(ReportValue::NonFinite),
            Some(fin(100.0)),
            Some(ReportValue::NonFinite),
            None,
            percentiles,
        ),
    )]);
    let report = report_with(metrics);

    let json = render_json(&report, &cfg("profile_export", "", false, false));
    assert_eq!(json, include_str!("../golden/v1_nonfinite.json"));

    let csv = render_csv(&report).unwrap();
    let expected = "Metric,avg,min,max,sum,p1,p5,p10,p25,p50,p75,p90,p95,p99,std\r\n\
Request Latency (ms),,100.00,,,,,,,150.00,,,,,\r\n";
    assert_eq!(csv, expected);
}

#[test]
fn non_streaming_report_omits_absent_streaming_metrics() {
    // No streaming metrics in the report -> none appear (the AIPerf exporters
    // are presence-driven; there is no genai-perf-style streaming skip list).
    let metrics = BTreeMap::from([
        (
            "request_latency".to_owned(),
            dist(
                "ms",
                1,
                Some(fin(120.0)),
                Some(fin(120.0)),
                Some(fin(120.0)),
                Some(fin(0.0)),
                pcts([120.0; 9]),
            ),
        ),
        ("request_throughput".to_owned(), scalar("requests/sec", 8.0)),
    ]);
    let report = report_with(metrics);

    let json = render_json(&report, &cfg("profile_export", "completions", false, false));
    assert!(!json.contains("time_to_first_token"));
    assert!(!json.contains("inter_token_latency"));
    // goodput=false / streaming=false surface into the projected input_config.
    assert!(json.contains("\"streaming\": false"));
    assert!(json.contains("\"goodput\": false"));

    let csv = render_csv(&report).unwrap();
    assert!(!csv.contains("Time to First Token"));
}

#[test]
fn goodput_metric_present_regardless_of_flag() {
    // The v1 summary emits the goodput metric whenever the report carries it;
    // the cfg goodput flag only drives input_config, not metric gating.
    let metrics = BTreeMap::from([("goodput".to_owned(), scalar("requests/sec", 1.5))]);
    let report = report_with(metrics);

    let json = render_json(&report, &cfg("profile_export", "chat", true, false));
    assert!(json.contains("\"goodput\": {"));
    assert!(json.contains("\"avg\": 1.5"));
    assert!(json.contains("\"goodput\": false")); // input_config reflects the flag
}

#[test]
fn export_writes_both_aiperf_files_with_stemmed_names() {
    let report = streaming_report();
    let dir = std::env::temp_dir().join(format!("aiperf-genai-perf-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let config = cfg("profile_export", "chat", true, true);

    GenaiPerfV1Exporter.export(&report, &dir, &config).unwrap();

    let json_path = dir.join("profile_export_aiperf.json");
    let csv_path = dir.join("profile_export_aiperf.csv");
    assert!(json_path.exists());
    assert!(csv_path.exists());
    assert_eq!(
        std::fs::read_to_string(&json_path).unwrap(),
        include_str!("../golden/v1_streaming.json")
    );
    assert!(std::fs::read_to_string(&csv_path).unwrap().contains("\r\n"));

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn export_rejects_path_traversal_stem() {
    let report = report_with(BTreeMap::new());
    let dir = std::env::temp_dir();
    let config = cfg("../evil", "chat", false, false);
    assert!(GenaiPerfV1Exporter.export(&report, &dir, &config).is_err());
}

#[test]
fn empty_report_emits_only_scalar_top_level_fields() {
    let report = report_with(BTreeMap::new());
    let json = render_json(&report, &cfg("profile_export", "chat", false, false));
    assert!(json.contains("\"schema_version\": \"1.4\""));
    assert!(json.contains("\"aiperf_version\": \"0.0.0-test\""));
    assert!(json.contains("\"was_cancelled\": false"));
    // No metric objects.
    assert!(!json.contains("request_latency"));

    // No metrics -> both CSV sections empty -> empty file.
    let csv = render_csv(&report).unwrap();
    assert_eq!(csv, "");
}
