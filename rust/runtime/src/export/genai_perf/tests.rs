// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Golden byte-contract tests for the AIPerf v1 summary sink.
//!
//! Fixtures pin indentation, field order, CRLF records, metric shapes, and
//! omission rules.

use super::*;
use crate::metrics_core::{
    AccumulatorSummary, MetricEntry, MetricSeries, NativeReport, ReportCounterStats,
    ReportDistributionStats, ReportScalarStats, ReportStats, ReportValue,
};
use std::collections::{BTreeMap, HashMap};

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

fn pcts(values: [f64; 9]) -> BTreeMap<String, ReportValue> {
    PERCENTILE_LABELS
        .iter()
        .zip(values)
        .map(|(label, value)| ((*label).to_owned(), fin(value)))
        .collect()
}

fn scalar(unit: &str, value: f64) -> MetricEntry {
    entry(
        "scalar",
        unit,
        ReportStats::Scalar(ReportScalarStats { value: fin(value) }),
    )
}

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

fn report_with(metrics: BTreeMap<String, MetricEntry>) -> NativeReport {
    let mut report = NativeReport::new(&AccumulatorSummary::new(), None);
    // Pin the version so the golden strings are stable across releases.
    report.aiperf_version = "0.0.0-test".to_owned();
    report.metrics = metrics;
    report.summary.was_cancelled = false;
    report
}

fn header_map() -> HashMap<String, String> {
    [
        ("request_latency", "Request Latency"),
        ("time_to_first_token", "Time to First Token"),
        ("time_to_last_round_trip", "Time to Last Round Trip"),
        ("avg_round_trip_time", "Average Round Trip Time"),
        ("inter_token_latency", "Inter Token Latency"),
        ("request_throughput", "Request Throughput"),
        ("request_count", "Request Count"),
        ("goodput", "Goodput"),
        ("min_request_timestamp", "Minimum Request Timestamp"),
    ]
    .iter()
    .map(|(tag, header)| ((*tag).to_owned(), (*header).to_owned()))
    .collect()
}

fn cfg(stem: &str) -> ExportConfig {
    ExportConfig {
        genai_perf: GenaiPerfExportConfig {
            enabled: true,
            // Predates the per-format split; `enabled: true` alone used to emit both.
            json: true,
            csv: true,
            stem: stem.to_owned(),
            header_map: header_map(),
            filtered_tags: ["min_request_timestamp".to_owned()].into_iter().collect(),
            scalar_tags: ["request_throughput", "request_count", "goodput"]
                .iter()
                .map(|tag| (*tag).to_owned())
                .collect(),
            envelope: GenaiPerfEnvelope::default(),
        },
        ..ExportConfig::default()
    }
}

// Representative fixture covering distributions, scalars, counters, and
// filtering of internal metrics.
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
    let json = render_json(&report, &cfg("profile_export").genai_perf);
    assert_eq!(json, include_str!("../golden/v1_streaming.json"));
    assert!(!json.contains("min_request_timestamp"));
}

#[test]
fn csv_matches_python_oracle_streaming_goodput() {
    let report = streaming_report();
    let csv = render_csv(&report, &cfg("profile_export").genai_perf).unwrap();
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
    // report) and therefore absent in JSON and empty in CSV; p50/min/count survive.
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

    let json = render_json(&report, &cfg("profile_export").genai_perf);
    assert_eq!(json, include_str!("../golden/v1_nonfinite.json"));

    let csv = render_csv(&report, &cfg("profile_export").genai_perf).unwrap();
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

    let json = render_json(&report, &cfg("profile_export").genai_perf);
    assert!(!json.contains("time_to_first_token"));
    assert!(!json.contains("inter_token_latency"));
    // error_summary is always emitted (empty here); the sink is presence-driven.
    assert!(json.contains("\"error_summary\": []"));

    let csv = render_csv(&report, &cfg("profile_export").genai_perf).unwrap();
    assert!(!csv.contains("Time to First Token"));
}

#[test]
fn websocket_lag_distributions_render_only_when_present() {
    let report = report_with(BTreeMap::from([
        (
            "time_to_last_round_trip".to_owned(),
            dist(
                "ms",
                1,
                Some(fin(300.0)),
                Some(fin(300.0)),
                Some(fin(300.0)),
                Some(fin(0.0)),
                pcts([300.0; 9]),
            ),
        ),
        (
            "avg_round_trip_time".to_owned(),
            dist(
                "ms",
                1,
                Some(fin(250.0)),
                Some(fin(250.0)),
                Some(fin(250.0)),
                Some(fin(0.0)),
                pcts([250.0; 9]),
            ),
        ),
    ]));

    let json = render_json(&report, &cfg("profile_export").genai_perf);
    assert!(json.contains("time_to_last_round_trip"));
    assert!(json.contains("avg_round_trip_time"));
    let csv = render_csv(&report, &cfg("profile_export").genai_perf).unwrap();
    assert!(csv.contains("Time to Last Round Trip (ms),300.00"));
    assert!(csv.contains("Average Round Trip Time (ms),250.00"));

    let absent = report_with(BTreeMap::new());
    let json = render_json(&absent, &cfg("profile_export").genai_perf);
    assert!(!json.contains("round_trip"));
    let csv = render_csv(&absent, &cfg("profile_export").genai_perf).unwrap();
    assert!(!csv.contains("Round Trip"));
}

#[test]
fn goodput_metric_present_regardless_of_flag() {
    // The v1 summary emits the goodput metric whenever the report carries it.
    let metrics = BTreeMap::from([("goodput".to_owned(), scalar("requests/sec", 1.5))]);
    let report = report_with(metrics);

    let json = render_json(&report, &cfg("profile_export").genai_perf);
    assert!(json.contains("\"goodput\": {"));
    assert!(json.contains("\"avg\": 1.5"));
}

#[test]
fn export_writes_both_aiperf_files_with_stemmed_names() {
    let report = streaming_report();
    let dir = std::env::temp_dir().join(format!("aiperf-genai-perf-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let config = cfg("profile_export");

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
    let config = cfg("../evil");
    assert!(GenaiPerfV1Exporter.export(&report, &dir, &config).is_err());
}

#[test]
fn empty_report_emits_only_scalar_top_level_fields() {
    let report = report_with(BTreeMap::new());
    let json = render_json(&report, &cfg("profile_export").genai_perf);
    assert!(json.contains("\"schema_version\": \"1.4\""));
    assert!(json.contains("\"aiperf_version\": \"0.0.0-test\""));
    assert!(json.contains("\"was_cancelled\": false"));
    // No metric objects.
    assert!(!json.contains("request_latency"));

    // No metrics -> both CSV sections empty -> empty file.
    let csv = render_csv(&report, &cfg("profile_export").genai_perf).unwrap();
    assert_eq!(csv, "");
}

fn gpu_series(gpu: &str, uuid: &str, endpoint: &str, stats: ReportStats) -> MetricSeries {
    MetricSeries {
        labels: Some(BTreeMap::from([
            ("gpu".to_owned(), gpu.to_owned()),
            ("gpu_uuid".to_owned(), uuid.to_owned()),
            ("model_name".to_owned(), "NVIDIA H200".to_owned()),
            ("platform".to_owned(), "nvidia".to_owned()),
            ("hostname".to_owned(), "localhost".to_owned()),
        ])),
        endpoint_url: Some(endpoint.to_owned()),
        stats,
        timeslices: Vec::new(),
    }
}

fn gauge_stats(avg: f64, min: f64, max: f64, std: f64) -> ReportStats {
    ReportStats::Distribution(ReportDistributionStats {
        count: Some(2),
        avg: Some(fin(avg)),
        min: Some(fin(min)),
        max: Some(fin(max)),
        std: Some(fin(std)),
        percentiles: pcts([min, min, min, avg, avg, avg, max, max, max]),
    })
}

#[test]
fn telemetry_data_projects_gpu_series_grouped_by_endpoint_and_gpu() {
    const ENDPOINT: &str = "http://127.0.0.1:9400/dcgm1/metrics";
    let metrics = BTreeMap::from([
        (
            "nvidia_energy_consumption".to_owned(),
            MetricEntry {
                metric_type: "counter",
                unit: "MJ".to_owned(),
                group: "default",
                higher_is_better: false,
                series: vec![
                    gpu_series(
                        "0",
                        "GPU-A",
                        ENDPOINT,
                        ReportStats::Counter(ReportCounterStats {
                            total: fin(0.5),
                            rate: Some(fin(0.1)),
                        }),
                    ),
                    gpu_series(
                        "1",
                        "GPU-B",
                        ENDPOINT,
                        ReportStats::Counter(ReportCounterStats {
                            total: fin(0.7),
                            rate: Some(fin(0.2)),
                        }),
                    ),
                ],
            },
        ),
        (
            "nvidia_power_usage".to_owned(),
            MetricEntry {
                metric_type: "distribution",
                unit: "W".to_owned(),
                group: "default",
                higher_is_better: false,
                series: vec![
                    gpu_series(
                        "0",
                        "GPU-A",
                        ENDPOINT,
                        gauge_stats(300.0, 100.0, 500.0, 40.0),
                    ),
                    gpu_series(
                        "1",
                        "GPU-B",
                        ENDPOINT,
                        gauge_stats(320.0, 120.0, 520.0, 44.0),
                    ),
                ],
            },
        ),
        // A non-GPU request metric (unlabeled series) that must be excluded.
        ("request_latency".to_owned(), scalar("ms", 1.0)),
    ]);
    let report = report_with(metrics);

    let telemetry = render_telemetry_data(&report).expect("telemetry present");

    // One endpoint, keyed by the scheme-stripped, /metrics-trimmed display URL.
    let endpoints = telemetry["endpoints"].as_object().unwrap();
    assert_eq!(endpoints.len(), 1);
    let endpoint = &endpoints["127.0.0.1:9400/dcgm1"];

    let gpus = endpoint["gpus"].as_object().unwrap();
    assert_eq!(
        gpus.keys().collect::<Vec<_>>(),
        vec!["gpu_0", "gpu_1"],
        "GPUs keyed by index, first-seen order"
    );

    let gpu0 = &gpus["gpu_0"];
    assert_eq!(gpu0["gpu_index"], 0);
    assert_eq!(gpu0["gpu_name"], "NVIDIA H200");
    assert_eq!(gpu0["gpu_uuid"], "GPU-A");
    assert_eq!(gpu0["platform"], "nvidia");
    assert_eq!(gpu0["hostname"], "localhost");

    let gpu0_metrics = gpu0["metrics"].as_object().unwrap();
    // Report BTreeMap order is alphabetical: nvidia_energy_consumption before nvidia_power_usage.
    assert_eq!(
        gpu0_metrics.keys().collect::<Vec<_>>(),
        vec!["nvidia_energy_consumption", "nvidia_power_usage"]
    );

    // Gauge: {unit,avg,p1..p99,min,max,std,count}; no sum.
    let gauge = gpu0_metrics["nvidia_power_usage"].as_object().unwrap();
    assert_eq!(gauge["unit"], "W");
    assert_eq!(gauge["avg"], 300.0);
    assert_eq!(gauge["min"], 100.0);
    assert_eq!(gauge["max"], 500.0);
    assert_eq!(gauge["std"], 40.0);
    assert_eq!(gauge["count"], 2);
    assert_eq!(gauge["p50"], 300.0);
    assert!(!gauge.contains_key("sum"), "gauges never carry sum");

    // Counter: {unit,avg,min,max,sum} all equal to the total; no distribution keys.
    let counter = gpu0_metrics["nvidia_energy_consumption"]
        .as_object()
        .unwrap();
    assert_eq!(counter["unit"], "MJ");
    assert_eq!(counter["avg"], 0.5);
    assert_eq!(counter["min"], 0.5);
    assert_eq!(counter["max"], 0.5);
    assert_eq!(counter["sum"], 0.5);
    assert!(!counter.contains_key("std"));
    assert!(!counter.contains_key("count"));
    assert!(!counter.contains_key("p50"));

    // The unlabeled request metric never leaks into telemetry.
    let telemetry_str = telemetry.to_string();
    assert!(!telemetry_str.contains("request_latency"));

    // Summary carries the raw (un-normalized) endpoint URL for configured/successful.
    let summary = &telemetry["summary"];
    assert_eq!(
        summary["endpoints_configured"],
        serde_json::json!([ENDPOINT])
    );
    assert_eq!(
        summary["endpoints_successful"],
        serde_json::json!([ENDPOINT])
    );

    // And the whole block is spliced into the top-level JSON before input_config.
    let json = render_json(&report, &cfg("profile_export").genai_perf);
    assert!(json.contains("\"telemetry_data\""));
}

#[test]
fn telemetry_data_absent_when_no_gpu_series() {
    let metrics = BTreeMap::from([("request_latency".to_owned(), scalar("ms", 1.0))]);
    let report = report_with(metrics);
    assert!(render_telemetry_data(&report).is_none());
    let json = render_json(&report, &cfg("profile_export").genai_perf);
    assert!(!json.contains("telemetry_data"));
}
