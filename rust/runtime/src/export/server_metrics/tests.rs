// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Golden byte-contract tests for the server-metrics summary sink.
//!
//! The JSON and CSV fixtures pin field order, number formatting, omission, and
//! record delimiters. The JSON test substitutes process-local datetime values so
//! the remaining bytes stay timezone-portable.

use std::collections::BTreeMap;

use serde_json::json;

use super::*;
use crate::metrics_core::report::ReportHistogramStats;
use crate::metrics_core::{
    MetricEntry, MetricSeries, NativeReport, ReportCounterStats, ReportDistributionStats,
    ReportServerMetricsEndpointInfo, ReportServerMetricsMetadata, ReportServerMetricsPhaseRange,
    ReportStats, ReportSummary, ReportTimeslice, ReportValue,
};

const EP: &str = "http://localhost:8081/metrics";

fn fin(value: f64) -> ReportValue {
    ReportValue::Finite(value)
}

fn labels(pairs: &[(&str, &str)]) -> BTreeMap<String, String> {
    pairs
        .iter()
        .map(|(key, value)| ((*key).to_string(), (*value).to_string()))
        .collect()
}

fn percentiles(pairs: &[(u32, f64)]) -> BTreeMap<String, ReportValue> {
    pairs
        .iter()
        .map(|(percentile, value)| (format!("p{percentile}"), fin(*value)))
        .collect()
}

fn entry(series: Vec<MetricSeries>) -> MetricEntry {
    MetricEntry {
        metric_type: "distribution",
        unit: String::new(),
        group: "default",
        higher_is_better: false,
        series,
    }
}

fn gauge_series(
    label_pairs: &[(&str, &str)],
    avg: f64,
    min: f64,
    max: f64,
    std: f64,
    pcts: &[(u32, f64)],
    timeslices: Vec<ReportTimeslice>,
) -> MetricSeries {
    MetricSeries {
        labels: Some(labels(label_pairs)),
        endpoint_url: Some(EP.to_string()),
        stats: ReportStats::Distribution(ReportDistributionStats {
            count: Some(10),
            avg: Some(fin(avg)),
            min: Some(fin(min)),
            max: Some(fin(max)),
            std: Some(fin(std)),
            percentiles: percentiles(pcts),
        }),
        timeslices,
    }
}

fn synthetic_report() -> NativeReport {
    let gauge = entry(vec![gauge_series(
        &[("model", "llama")],
        2.5,
        1.0,
        4.0,
        1.2,
        &[
            (1, 1.0),
            (5, 1.0),
            (10, 1.0),
            (25, 2.0),
            (50, 2.5),
            (75, 3.0),
            (90, 4.0),
            (95, 4.0),
            (99, 4.0),
        ],
        vec![ReportTimeslice {
            start_ns: 1_000_000_000,
            end_ns: 2_000_000_000,
            complete: true,
            stats: ReportStats::Distribution(ReportDistributionStats {
                count: Some(5),
                avg: Some(fin(2.0)),
                min: Some(fin(1.0)),
                max: Some(fin(3.0)),
                std: None,
                percentiles: BTreeMap::new(),
            }),
        }],
    )]);

    let counter = entry(vec![MetricSeries {
        labels: None,
        endpoint_url: Some(EP.to_string()),
        stats: ReportStats::Counter(ReportCounterStats {
            total: fin(1000.0),
            rate: Some(fin(16.6667)),
        }),
        timeslices: Vec::new(),
    }]);

    let histogram = entry(vec![MetricSeries {
        labels: None,
        endpoint_url: Some(EP.to_string()),
        stats: ReportStats::Histogram(ReportHistogramStats {
            count: 5,
            sum: fin(3.5),
            avg: Some(fin(0.7)),
            count_rate: Some(fin(0.0833)),
            sum_rate: Some(fin(0.0583)),
            percentiles: percentiles(&[(50, 0.6), (99, 1.4)]),
            buckets: BTreeMap::from([
                ("+Inf".to_string(), 5),
                ("0.1".to_string(), 1),
                ("1.0".to_string(), 3),
            ]),
        }),
        timeslices: Vec::new(),
    }]);

    let info = entry(vec![gauge_series(
        &[("block_size", "16"), ("version", "1.2")],
        1.0,
        1.0,
        1.0,
        0.0,
        &[
            (1, 1.0),
            (5, 1.0),
            (10, 1.0),
            (25, 1.0),
            (50, 1.0),
            (75, 1.0),
            (90, 1.0),
            (95, 1.0),
            (99, 1.0),
        ],
        Vec::new(),
    )]);

    let metadata = ReportServerMetricsMetadata {
        endpoints_configured: vec!["localhost:8081".to_string()],
        endpoints_successful: vec!["localhost:8081".to_string()],
        descriptions: BTreeMap::from([
            (
                "cache_config_info".to_string(),
                "Cache configuration.".to_string(),
            ),
            (
                "num_requests_running".to_string(),
                "Number of running requests.".to_string(),
            ),
            (
                "prompt_tokens_total".to_string(),
                "Total prompt tokens.".to_string(),
            ),
            (
                "request_latency_seconds".to_string(),
                "Request latency in seconds.".to_string(),
            ),
        ]),
        metric_types: BTreeMap::from([
            ("cache_config_info".to_string(), "gauge".to_string()),
            ("num_requests_running".to_string(), "gauge".to_string()),
            ("prompt_tokens_total".to_string(), "counter".to_string()),
            (
                "request_latency_seconds".to_string(),
                "histogram".to_string(),
            ),
        ]),
        endpoint_info: BTreeMap::from([(
            EP.to_string(),
            ReportServerMetricsEndpointInfo {
                total_fetches: 60,
                first_fetch_ns: 1_000_000_000,
                last_fetch_ns: 61_000_000_000,
                avg_fetch_latency_ms: 2.5,
                unique_updates: 50,
                first_update_ns: 1_000_000_000,
                last_update_ns: 60_000_000_000,
                duration_seconds: 59.0,
                avg_update_interval_ms: 1180.0,
                median_update_interval_ms: None,
            },
        )]),
        profiling: Some(ReportServerMetricsPhaseRange {
            start_ns: 1_000_000_000,
            end_ns: 61_000_000_000,
        }),
        warmup: None,
    };

    NativeReport {
        schema_version: crate::metrics_core::NATIVE_REPORT_SCHEMA_VERSION,
        aiperf_version: "0.0.0".to_string(),
        run: Default::default(),
        summary: ReportSummary {
            server_metrics: Some(metadata),
            ..ReportSummary::default()
        },
        metrics: BTreeMap::new(),
        warmup_metrics: None,
        server_metrics: BTreeMap::from([
            ("cache_config_info".to_string(), info),
            ("num_requests_running".to_string(), gauge),
            ("prompt_tokens_total".to_string(), counter),
            ("request_latency_seconds".to_string(), histogram),
        ]),
        warmup_server_metrics: BTreeMap::new(),
        media_metrics: BTreeMap::new(),
        accuracy: None,
        accuracy_records: Vec::new(),
        evaluator: None,
        errors: Vec::new(),
        otel_per_record: None,
    }
}

fn full_policy() -> ServerMetricsExportConfig {
    ServerMetricsExportConfig {
        json: true,
        csv: true,
        // `None` exercises the report-version fallback pinned by the fixture.
        aiperf_version: None,
        benchmark_id: Some("bench-123".to_string()),
        input_config: json!({"model": "llama", "concurrency": 4}),
    }
}

#[test]
fn json_matches_python_exporter_bytes() {
    let report = synthetic_report();
    let meta = report.summary.server_metrics.as_ref().unwrap();
    let produced = build_json(&report, meta, &full_policy());

    // The two summary datetimes render in the process-local zone; the golden was
    // generated with TZ=UTC. Substitute the sink's own render so the byte pin is
    // timezone-portable while pinning field order, numbers, and presence.
    let start = python_isoformat_from_ns(1_000_000_000);
    let end = python_isoformat_from_ns(61_000_000_000);
    let expected = include_str!("../../tests/golden/server_metrics_export.json")
        .trim_end()
        .replace("1970-01-01T00:00:01", &start)
        .replace("1970-01-01T00:01:01", &end);

    assert_eq!(produced, expected);
}

#[test]
fn csv_matches_python_exporter_bytes() {
    let report = synthetic_report();
    let produced = build_csv(&report, &full_policy());
    // The CSV carries no timezone-dependent field, so it is pinned exactly.
    let expected = include_str!("../../tests/golden/server_metrics_export.csv");
    assert_eq!(produced, expected);
}

#[test]
fn export_writes_both_files_and_gates_on_policy() {
    let report = synthetic_report();
    let base = std::env::temp_dir().join(format!(
        "aiperf_sm_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&base).unwrap();

    let mut cfg = ExportConfig::default();
    cfg.server_metrics = full_policy();
    ServerMetricsExporter.export(&report, &base, &cfg).unwrap();
    assert!(base.join("server_metrics_export.json").exists());
    assert!(base.join("server_metrics_export.csv").exists());
    let json_bytes = std::fs::read_to_string(base.join("server_metrics_export.json")).unwrap();
    assert_eq!(
        json_bytes,
        build_json(
            &report,
            report.summary.server_metrics.as_ref().unwrap(),
            &cfg.server_metrics
        )
    );

    // Gating: JSON only.
    let json_only = base.join("json_only");
    std::fs::create_dir_all(&json_only).unwrap();
    cfg.server_metrics = ServerMetricsExportConfig {
        json: true,
        csv: false,
        ..full_policy()
    };
    ServerMetricsExporter
        .export(&report, &json_only, &cfg)
        .unwrap();
    assert!(json_only.join("server_metrics_export.json").exists());
    assert!(!json_only.join("server_metrics_export.csv").exists());

    std::fs::remove_dir_all(&base).ok();
    std::fs::remove_dir_all(&json_only).ok();
}

#[test]
fn empty_server_metrics_writes_nothing() {
    let mut report = synthetic_report();
    // Missing metadata produces no artifacts.
    report.summary.server_metrics = None;
    let base = std::env::temp_dir().join(format!("aiperf_sm_empty_{}", std::process::id()));
    std::fs::create_dir_all(&base).unwrap();
    let mut cfg = ExportConfig::default();
    cfg.server_metrics = full_policy();
    ServerMetricsExporter.export(&report, &base, &cfg).unwrap();
    assert!(!base.join("server_metrics_export.json").exists());
    assert!(!base.join("server_metrics_export.csv").exists());
    std::fs::remove_dir_all(&base).ok();
}

#[test]
fn isoformat_omits_fraction_only_when_microseconds_zero() {
    use chrono::NaiveDate;
    let whole = NaiveDate::from_ymd_opt(1970, 1, 1)
        .unwrap()
        .and_hms_opt(0, 0, 1)
        .unwrap();
    assert_eq!(isoformat_naive(whole, 0), "1970-01-01T00:00:01");
    let fractional = NaiveDate::from_ymd_opt(2026, 7, 14)
        .unwrap()
        .and_hms_micro_opt(12, 34, 56, 789_012)
        .unwrap();
    assert_eq!(
        isoformat_naive(fractional, 789_012),
        "2026-07-14T12:34:56.789012"
    );
}

#[test]
fn normalize_endpoint_display_trims_scheme_and_metrics_suffix() {
    assert_eq!(normalize_endpoint_display(EP), "localhost:8081");
    assert_eq!(
        normalize_endpoint_display("https://host:9400/api/metrics"),
        "host:9400/api"
    );
    assert_eq!(
        normalize_endpoint_display("http://host:8000/custom"),
        "host:8000/custom"
    );
}
