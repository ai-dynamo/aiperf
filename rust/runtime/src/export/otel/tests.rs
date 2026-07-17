// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Parity tests for the native OTLP metrics emitter.
//!
//! Each captured OTLP `ExportMetricsServiceRequest` is decoded with the
//! authoritative `opentelemetry-proto` crate (not the sink's own hand-written
//! prost subset), so a wrong field tag would fail decode. Assertions are pinned
//! to the GenAI semantic-convention contract.

use std::collections::BTreeMap;
use std::io::{Read, Write};
use std::net::TcpListener;
use std::sync::mpsc;

use opentelemetry_proto::tonic::collector::metrics::v1::ExportMetricsServiceRequest;
use opentelemetry_proto::tonic::common::v1::{KeyValue, any_value};
use opentelemetry_proto::tonic::metrics::v1::{Metric, metric};
use prost::Message as _;

use super::{OtelExporter, OtelRecordAccumulator};
use crate::export::{ExportConfig, Exporter, OtelExportConfig};
use crate::metrics_core::catalog::{MetricConsoleGroup, MetricTag};
use crate::metrics_core::report::NativeReport;
use crate::metrics_core::{
    AccumulatorSummary, DistributionStats, MetricResult, MetricResultData, MetricValue,
};

/// Build a finite-valued distribution result under `tag`/`source_tag`.
fn distribution_result(
    tag: &str,
    source_tag: MetricTag,
    unit: &str,
    avg: f64,
    min: f64,
    max: f64,
    count: usize,
) -> MetricResult {
    let mut percentiles = BTreeMap::new();
    percentiles.insert(50, MetricValue::Finite(avg));
    percentiles.insert(99, MetricValue::Finite(max));
    MetricResult {
        tag: tag.to_string(),
        source_tag: Some(source_tag),
        header: tag.to_string(),
        unit: unit.to_string(),
        console_group: MetricConsoleGroup::Default,
        data: MetricResultData::Distribution(DistributionStats {
            tag: tag.to_string(),
            avg: MetricValue::Finite(avg),
            min: MetricValue::Finite(min),
            max: MetricValue::Finite(max),
            std: None,
            sum: MetricValue::Finite(avg * count as f64),
            count,
            percentiles,
        }),
    }
}

/// A report carrying the three GenAI duration metrics and both token-count
/// metrics, with millisecond latency display units (as the accumulator emits).
fn sample_report() -> NativeReport {
    let mut summary = AccumulatorSummary::new();
    summary.insert_result(distribution_result(
        "request_latency",
        MetricTag::RequestLatency,
        "ms",
        320.0,
        100.0,
        640.0,
        4,
    ));
    summary.insert_result(distribution_result(
        "time_to_first_token",
        MetricTag::TimeToFirstToken,
        "ms",
        40.0,
        10.0,
        80.0,
        4,
    ));
    summary.insert_result(distribution_result(
        "inter_token_latency",
        MetricTag::InterTokenLatency,
        "ms",
        20.0,
        5.0,
        40.0,
        4,
    ));
    summary.insert_result(distribution_result(
        "input_sequence_length",
        MetricTag::InputSequenceLength,
        "tokens",
        128.0,
        64.0,
        256.0,
        4,
    ));
    summary.insert_result(distribution_result(
        "output_token_count",
        MetricTag::OutputTokenCount,
        "tokens",
        64.0,
        16.0,
        256.0,
        4,
    ));
    // Run-timeline timestamps used for the OTLP window (no wall-clock read).
    summary.insert_finite(MetricTag::MinRequestTimestamp, 1_000.0);
    summary.insert_finite(MetricTag::MaxResponseTimestamp, 2_000.0);
    NativeReport::new(&summary, None)
}

/// Build export configuration with the configured resource attributes.
fn sample_config(endpoint: String) -> ExportConfig {
    let mut resource_attributes = BTreeMap::new();
    resource_attributes.insert(
        "service.instance.id".to_string(),
        "records-manager".to_string(),
    );
    resource_attributes.insert("aiperf.benchmark.id".to_string(), "bench-123".to_string());
    resource_attributes.insert("aiperf.endpoint.type".to_string(), "chat".to_string());
    resource_attributes.insert("aiperf.model.name".to_string(), "llama-3".to_string());
    resource_attributes.insert("team".to_string(), "perf".to_string());
    ExportConfig {
        otel: OtelExportConfig {
            enabled: true,
            endpoint: Some(endpoint),
            provider: Some("openai".to_string()),
            resource_attributes,
        },
        ..ExportConfig::default()
    }
}

/// Spawn a one-shot HTTP/1.1 sink on a loopback port; returns (endpoint URL,
/// receiver that yields the captured request body bytes).
fn spawn_capture_server() -> (String, mpsc::Receiver<Vec<u8>>) {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind loopback");
    let port = listener.local_addr().unwrap().port();
    let endpoint = format!("http://127.0.0.1:{port}/v1/metrics");
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().expect("accept");
        let mut buf = Vec::new();
        let mut chunk = [0u8; 4096];
        // Read until headers complete, then read the declared body length.
        let header_end = loop {
            if let Some(pos) = find_subslice(&buf, b"\r\n\r\n") {
                break pos + 4;
            }
            let n = stream.read(&mut chunk).expect("read headers");
            if n == 0 {
                break buf.len();
            }
            buf.extend_from_slice(&chunk[..n]);
        };
        let content_length = parse_content_length(&buf[..header_end]);
        while buf.len() < header_end + content_length {
            let n = stream.read(&mut chunk).expect("read body");
            if n == 0 {
                break;
            }
            buf.extend_from_slice(&chunk[..n]);
        }
        let body = buf[header_end..header_end + content_length].to_vec();
        stream
            .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n")
            .expect("write response");
        let _ = stream.flush();
        let _ = tx.send(body);
    });
    (endpoint, rx)
}

fn find_subslice(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

fn parse_content_length(headers: &[u8]) -> usize {
    let text = String::from_utf8_lossy(headers);
    for line in text.split("\r\n") {
        if let Some((name, value)) = line.split_once(':')
            && name.eq_ignore_ascii_case("content-length")
        {
            return value.trim().parse().unwrap_or(0);
        }
    }
    0
}

/// Extract a string attribute value by key.
fn string_attr<'a>(attrs: &'a [KeyValue], key: &str) -> Option<&'a str> {
    attrs.iter().find(|kv| kv.key == key).and_then(|kv| {
        match kv.value.as_ref()?.value.as_ref()? {
            any_value::Value::StringValue(value) => Some(value.as_str()),
            _ => None,
        }
    })
}

/// The histogram body of a metric.
fn histogram(metric: &Metric) -> &opentelemetry_proto::tonic::metrics::v1::Histogram {
    match metric.data.as_ref().expect("metric data") {
        metric::Data::Histogram(histogram) => histogram,
        other => panic!("expected histogram, got {other:?}"),
    }
}

#[test]
fn export_emits_genai_semconv_metrics_with_exact_names_attrs_and_bounds() {
    let (endpoint, rx) = spawn_capture_server();
    let report = sample_report();
    let cfg = sample_config(endpoint);
    let dir = std::env::temp_dir();

    OtelExporter
        .export(&report, &dir, &cfg)
        .expect("otel export succeeds");

    let body = rx
        .recv_timeout(std::time::Duration::from_secs(5))
        .expect("captured OTLP body");
    let request = ExportMetricsServiceRequest::decode(body.as_slice())
        .expect("decode OTLP ExportMetricsServiceRequest");

    let resource_metrics = &request.resource_metrics;
    assert_eq!(resource_metrics.len(), 1);
    let rm = &resource_metrics[0];

    // Resource attributes: service.name is sink-constant; the rest are projected
    // Configured resource attributes are preserved.
    let resource_attrs = &rm.resource.as_ref().expect("resource").attributes;
    assert_eq!(string_attr(resource_attrs, "service.name"), Some("aiperf"));
    assert_eq!(
        string_attr(resource_attrs, "service.instance.id"),
        Some("records-manager")
    );
    assert_eq!(
        string_attr(resource_attrs, "aiperf.benchmark.id"),
        Some("bench-123")
    );
    assert_eq!(
        string_attr(resource_attrs, "aiperf.endpoint.type"),
        Some("chat")
    );
    assert_eq!(
        string_attr(resource_attrs, "aiperf.model.name"),
        Some("llama-3")
    );
    assert_eq!(string_attr(resource_attrs, "team"), Some("perf"));

    // Instrumentation scope is stable.
    let sm = &rm.scope_metrics[0];
    assert_eq!(sm.scope.as_ref().unwrap().name, "aiperf.records");

    let by_name: BTreeMap<&str, &Metric> =
        sm.metrics.iter().map(|m| (m.name.as_str(), m)).collect();

    // Semantic-convention metric names are present.
    let duration = by_name["gen_ai.client.operation.duration"];
    let ttft = by_name["gen_ai.client.operation.time_to_first_chunk"];
    let itl = by_name["gen_ai.client.operation.time_per_output_chunk"];
    let usage = by_name["gen_ai.client.token.usage"];

    // Metric units are stable.
    assert_eq!(duration.unit, "s");
    assert_eq!(ttft.unit, "s");
    assert_eq!(itl.unit, "s");
    assert_eq!(usage.unit, "{token}");

    // Explicit bucket boundaries are stable.
    assert_eq!(
        histogram(duration).data_points[0].explicit_bounds,
        super::DURATION_BOUNDS
    );
    assert_eq!(
        histogram(ttft).data_points[0].explicit_bounds,
        super::TTFT_BOUNDS
    );
    assert_eq!(
        histogram(itl).data_points[0].explicit_bounds,
        super::TIME_PER_OUTPUT_CHUNK_BOUNDS
    );
    assert_eq!(
        histogram(usage).data_points[0].explicit_bounds,
        super::TOKEN_USAGE_BOUNDS
    );

    // bucket_counts length must be bounds+1 (OTLP invariant), all zero in the
    // aggregate, which cannot reconstruct bucket counts.
    let duration_dp = &histogram(duration).data_points[0];
    assert_eq!(
        duration_dp.bucket_counts.len(),
        super::DURATION_BOUNDS.len() + 1
    );
    assert!(duration_dp.bucket_counts.iter().all(|&c| c == 0));

    // Duration attributes: gen_ai.operation.name/provider.name/request.model
    // Chat operation mapping and provider override.
    let duration_attrs = &duration_dp.attributes;
    assert_eq!(
        string_attr(duration_attrs, "gen_ai.operation.name"),
        Some("chat")
    );
    assert_eq!(
        string_attr(duration_attrs, "gen_ai.provider.name"),
        Some("openai")
    );
    assert_eq!(
        string_attr(duration_attrs, "gen_ai.request.model"),
        Some("llama-3")
    );

    // Aggregate count/sum/min/max carried; ms->s converted (320ms avg -> 0.32s).
    assert_eq!(duration_dp.count, 4);
    assert_eq!(duration_dp.min, Some(0.1));
    assert_eq!(duration_dp.max, Some(0.64));
    assert_eq!(duration_dp.sum, Some(0.32 * 4.0));

    // Token usage carries two data points discriminated by gen_ai.token.type
    // Input and output token values are unscaled.
    let usage_hist = histogram(usage);
    assert_eq!(usage_hist.data_points.len(), 2);
    let token_types: Vec<Option<&str>> = usage_hist
        .data_points
        .iter()
        .map(|dp| string_attr(&dp.attributes, "gen_ai.token.type"))
        .collect();
    assert!(token_types.contains(&Some("input")));
    assert!(token_types.contains(&Some("output")));
    let input_dp = usage_hist
        .data_points
        .iter()
        .find(|dp| string_attr(&dp.attributes, "gen_ai.token.type") == Some("input"))
        .unwrap();
    // Input token count avg 128, identity-scaled (no ns->s), so sum == 128*4.
    assert_eq!(input_dp.sum, Some(128.0 * 4.0));
    assert_eq!(input_dp.max, Some(256.0));
}

#[test]
fn disabled_or_endpointless_config_does_not_run() {
    let mut cfg = ExportConfig::default();
    assert!(!OtelExporter.enabled(&cfg));
    cfg.otel.enabled = true;
    // Enabled but no endpoint: still inert.
    assert!(!OtelExporter.enabled(&cfg));
}

#[test]
fn provider_defaults_to_other_when_absent() {
    // Missing provider uses `_OTHER`.
    let (endpoint, rx) = spawn_capture_server();
    let report = sample_report();
    let mut cfg = sample_config(endpoint);
    cfg.otel.provider = None;
    let dir = std::env::temp_dir();

    OtelExporter.export(&report, &dir, &cfg).expect("export");
    let body = rx
        .recv_timeout(std::time::Duration::from_secs(5))
        .expect("body");
    let request = ExportMetricsServiceRequest::decode(body.as_slice()).unwrap();
    let dp = &request.resource_metrics[0].scope_metrics[0]
        .metrics
        .iter()
        .find(|m| m.name == "gen_ai.client.operation.duration")
        .unwrap();
    let attrs = &histogram(dp).data_points[0].attributes;
    assert_eq!(string_attr(attrs, "gen_ai.provider.name"), Some("_OTHER"));
}

#[test]
fn per_record_accumulator_populates_bucket_counts() {
    let (endpoint, rx) = spawn_capture_server();
    let mut report = sample_report();

    // Two profiling requests fed through the same projection the runner computes.
    let mut records = OtelRecordAccumulator::new();
    for (latency, ttft, itl, input, output) in [
        (320.0_f64, 40.0_f64, 20.0_f64, 128.0_f64, 64.0_f64),
        (100.0, 10.0, 5.0, 64.0, 16.0),
    ] {
        let lookup: BTreeMap<&str, (f64, &str)> = BTreeMap::from([
            ("request_latency", (latency, "ms")),
            ("time_to_first_token", (ttft, "ms")),
            ("inter_token_latency", (itl, "ms")),
            ("input_sequence_length", (input, "tokens")),
            ("output_token_count", (output, "tokens")),
        ]);
        records.observe_record(&lookup, None);
    }
    report.otel_per_record = Some(records);

    let cfg = sample_config(endpoint);
    let dir = std::env::temp_dir();
    OtelExporter.export(&report, &dir, &cfg).expect("export");

    let body = rx
        .recv_timeout(std::time::Duration::from_secs(5))
        .expect("body");
    let request = ExportMetricsServiceRequest::decode(body.as_slice()).unwrap();
    let sm = &request.resource_metrics[0].scope_metrics[0];
    let by_name: BTreeMap<&str, &Metric> =
        sm.metrics.iter().map(|m| (m.name.as_str(), m)).collect();

    // Duration histogram: populated buckets that sum to count (OTLP invariant).
    let duration = histogram(by_name["gen_ai.client.operation.duration"]);
    let dp = &duration.data_points[0];
    assert_eq!(dp.count, 2);
    let total: u64 = dp.bucket_counts.iter().sum();
    assert_eq!(total, dp.count);
    assert!(dp.bucket_counts.iter().any(|&c| c != 0));
    // 320 ms -> 0.32 s lands in bucket 5 (v <= bounds[5]=0.32); 100 ms -> 0.1 s
    // in bucket 4 (v <= bounds[4]=0.16, > bounds[3]=0.08).
    assert_eq!(dp.bucket_counts[5], 1);
    assert_eq!(dp.bucket_counts[4], 1);
    // Sum is the real total in seconds; min/max are the real extremes.
    assert_eq!(dp.sum, Some(0.32 + 0.1));
    assert_eq!(dp.min, Some(0.1));
    assert_eq!(dp.max, Some(0.32));
    // No error.type attribute on the success path.
    assert_eq!(string_attr(&dp.attributes, "error.type"), None);

    // Token usage input datapoint: populated buckets summing to count.
    let usage = histogram(by_name["gen_ai.client.token.usage"]);
    let input_dp = usage
        .data_points
        .iter()
        .find(|dp| string_attr(&dp.attributes, "gen_ai.token.type") == Some("input"))
        .unwrap();
    assert_eq!(input_dp.count, 2);
    let input_total: u64 = input_dp.bucket_counts.iter().sum();
    assert_eq!(input_total, 2);
    assert_eq!(input_dp.sum, Some(128.0 + 64.0));
}
