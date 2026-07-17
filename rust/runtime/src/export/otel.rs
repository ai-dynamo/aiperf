// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! OpenTelemetry OTLP/HTTP metrics emitter (native Rust).
//!
//! Emits the GenAI-semconv metric surface defined by
//! `post_processors/otel_metrics_results_processor.py`,
//! `otel_streaming_fanout.py`, `strategies/genai_semconv.py`,
//! and `strategies/metric_results.py`: the GenAI client
//! histograms (`gen_ai.client.operation.duration`,
//! `gen_ai.client.operation.time_to_first_chunk`,
//! `gen_ai.client.operation.time_per_output_chunk`,
//! `gen_ai.client.token.usage`) over OTLP/HTTP to the configured collector, with
//! the same resource attributes (`service.name`, `service.instance.id`,
//! `aiperf.benchmark.id`, `aiperf.endpoint.type`, `aiperf.model.name`, …), the
//! same per-datapoint GenAI attributes (`gen_ai.operation.name`,
//! `gen_ai.provider.name`, `gen_ai.request.model`, `gen_ai.token.type`), and the
//! same explicit histogram bucket boundaries.
//!
//! # Fidelity: populated per-record histograms (approach a)
//! The Python plane records **one histogram observation per request during the
//! run** (`MetricResultsStrategy.process` → `instrument.record`), so its exported
//! histograms carry exact per-bucket counts. This sink reproduces that: the
//! runner feeds each captured record's projected per-request metrics into an
//! [`OtelRecordAccumulator`] (the same projection the live-streaming sink
//! forwards to Python), which buckets every observation into the semconv
//! explicit histograms and is merged at run end. The finalized accumulator rides
//! on the report as a transient, non-serialized side channel
//! ([`NativeReport::otel_per_record`]); this sink then emits OTLP `Histogram`
//! data points with **populated `bucket_counts`** (+ `count`/`sum`/`min`/`max`)
//! that a collector aggregating Python's per-record stream would compute, under
//! the exact **metric names, attributes, and explicit bucket boundaries** Python
//! emits.
//!
//! When the accumulator is absent, including for synthetic reports, the sink
//! falls back to the aggregate
//! [`NativeReport`]: it carries the aggregate `count`/`sum`/`min`/`max` but
//! leaves `bucket_counts` at zero (the aggregate cannot reconstruct the
//! distribution across buckets).
//!
//! # Wire format
//! Encodes the OTLP `ExportMetricsServiceRequest` protobuf ([`proto`], a minimal
//! hand-written subset over the crate's existing `prost` — no OTel SDK is linked
//! into the shipped runner) and POSTs it as `application/x-protobuf`. The commit
//! site is synchronous with no ambient runtime, so this sink drives its own
//! short-lived `current_thread` tokio runtime and enforces a hard wall-clock
//! [`tokio::time::timeout`] so an unreachable collector cannot hang shutdown
//! (design §6). Time fields use the report's own run-timeline timestamps; the
//! sink never calls `SystemTime::now`.
//!
//! # Config projection
//! The runner decodes [`OtelExportConfig`] from the `cfg.export.otel` wire block.
//! The frontend projects these fields from `cfg.otel`:
//! - `enabled`  = `cfg.otel.metrics_url is not None`
//! - `endpoint` = the normalized OTLP/HTTP metrics URL (`…/v1/metrics`)
//! - `provider` = `cfg.otel.gen_ai_provider` (else omit → inferred `_OTHER`)
//! - `resource_attributes` = the exact map `_build_resource_attributes()` builds
//!   (`service.instance.id`, `aiperf.benchmark.id`, `aiperf.endpoint.type`,
//!   `aiperf.model.name`, plus `cfg.otel.custom_resource_attributes`). `service.
//!   name` is always `aiperf` and is set by this sink, so Python need not send it.

use std::collections::BTreeMap;
use std::path::Path;
use std::time::Duration;

use crate::export::{ExportConfig, Exporter};
use crate::metrics_core::NativeReport;
use crate::metrics_core::report::{
    MetricSeries, ReportDistributionStats, ReportStats, ReportValue,
};

mod accumulator;

use accumulator::{BucketHistogram, DurationKind, TokenKind};
pub use accumulator::{OtelRecordAccumulator, classify_spec_error_type};

/// OTLP/HTTP metrics export policy. Disabled unless the frontend provides an
/// OTLP endpoint. Fields mirror the `cfg.otel` surface the emitter needs; every
/// other GenAI/resource fact is derived from [`OtelExportConfig::resource_attributes`]
/// and the [`NativeReport`] so the projected wire surface stays exactly these
/// four fields (see the module docs' projection table).
#[derive(Debug, Clone, Default, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct OtelExportConfig {
    /// Whether OTLP metric export is enabled (frontend sets true when an OTLP
    /// endpoint is configured).
    pub enabled: bool,
    /// Normalized OTLP/HTTP metrics endpoint (`http(s)://host[:port]/v1/metrics`).
    pub endpoint: Option<String>,
    /// Optional GenAI provider-name override (`gen_ai.provider.name`); `_OTHER`
    /// when absent, matching `genai_semconv.infer_provider_name`'s fallback.
    pub provider: Option<String>,
    /// Resource attributes the frontend built (`service.instance.id`,
    /// `aiperf.benchmark.id`, `aiperf.endpoint.type`, `aiperf.model.name`, and
    /// `--otel-resource-attributes`). `service.name=aiperf` is added by the sink.
    #[serde(default)]
    pub resource_attributes: BTreeMap<String, String>,
}

/// Instrumentation scope name Python's fanout uses (`get_meter("aiperf.records")`).
const SCOPE_NAME: &str = "aiperf.records";

/// OTLP request timeout mirroring the Python default
/// (`Environment.OTEL.REQUEST_TIMEOUT_SECONDS`, floored at 1s). A single hard
/// bound so an unreachable collector cannot hang shutdown.
const EXPORT_TIMEOUT: Duration = Duration::from_secs(10);

/// The OTLP/HTTP metrics [`Exporter`].
pub struct OtelExporter;

impl Exporter for OtelExporter {
    fn name(&self) -> &'static str {
        "otel"
    }

    fn enabled(&self, cfg: &ExportConfig) -> bool {
        cfg.otel.enabled && cfg.otel.endpoint.is_some()
    }

    fn export(
        &self,
        report: &NativeReport,
        _artifact_dir: &Path,
        cfg: &ExportConfig,
    ) -> anyhow::Result<()> {
        let endpoint = cfg
            .otel
            .endpoint
            .as_deref()
            .ok_or_else(|| anyhow::anyhow!("OTLP export enabled without an endpoint"))?;

        let request = build_request(report, &cfg.otel);
        if request
            .resource_metrics
            .iter()
            .all(|rm| rm.scope_metrics.iter().all(|sm| sm.metrics.is_empty()))
        {
            // Nothing mappable in this report (e.g. no GenAI-semconv metrics were
            // recorded); emit nothing rather than an empty request.
            tracing::debug!("otel: no GenAI-semconv metrics in report; skipping OTLP POST");
            return Ok(());
        }

        let body = prost::Message::encode_to_vec(&request);
        post_otlp(endpoint, body)
    }
}

/// Build the OTLP `ExportMetricsServiceRequest` from the finalized report.
fn build_request(
    report: &NativeReport,
    cfg: &OtelExportConfig,
) -> proto::ExportMetricsServiceRequest {
    let resource = proto::Resource {
        attributes: resource_attributes(cfg),
        dropped_attributes_count: 0,
    };

    // Run-timeline timestamps stand in for the OTLP time window. They are the
    // report's own nanosecond marks, never a wall clock read here.
    let start_ns = report.summary.start_time.unwrap_or(0).max(0) as u64;
    let end_ns = report.summary.end_time.unwrap_or(0).max(0) as u64;

    let ctx = EmitContext {
        provider: cfg.provider.clone().unwrap_or_else(|| "_OTHER".to_string()),
        operation_name: operation_name(cfg),
        model: cfg.resource_attributes.get("aiperf.model.name").cloned(),
        start_ns,
        end_ns,
    };

    let mut metrics = Vec::new();
    match report
        .otel_per_record
        .as_ref()
        .filter(|records| !records.is_empty())
    {
        // Per-record path: populated `bucket_counts` from the merged accumulator.
        Some(records) => {
            for spec in DURATION_METRICS {
                if let Some(metric) = duration_metric_from_records(records, spec, &ctx) {
                    metrics.push(metric);
                }
            }
            if let Some(metric) = token_usage_metric_from_records(records, &ctx) {
                metrics.push(metric);
            }
        }
        // Aggregate fallback: exact bounds/count/sum but zero `bucket_counts`.
        None => {
            for spec in DURATION_METRICS {
                if let Some(metric) = duration_metric(report, spec, &ctx) {
                    metrics.push(metric);
                }
            }
            if let Some(metric) = token_usage_metric(report, &ctx) {
                metrics.push(metric);
            }
        }
    }

    proto::ExportMetricsServiceRequest {
        resource_metrics: vec![proto::ResourceMetrics {
            resource: Some(resource),
            scope_metrics: vec![proto::ScopeMetrics {
                scope: Some(proto::InstrumentationScope {
                    name: SCOPE_NAME.to_string(),
                    version: String::new(),
                    attributes: Vec::new(),
                    dropped_attributes_count: 0,
                }),
                metrics,
                schema_url: String::new(),
            }],
            schema_url: String::new(),
        }],
    }
}

/// Facts shared across every emitted metric for one report.
struct EmitContext {
    provider: String,
    operation_name: String,
    model: Option<String>,
    start_ns: u64,
    end_ns: u64,
}

/// One GenAI duration histogram: (Rust report metric key, spec metric name,
/// explicit bucket boundaries). Mirrors `genai_semconv.METRIC_NAME_MAP`
/// (`strategies/genai_semconv.py:182-208`) keyed by the Rust report's aggregate
/// metric names, which are byte-identical to the Python latency tags.
struct DurationSpec {
    report_key: &'static str,
    spec_name: &'static str,
    bounds: &'static [f64],
    kind: DurationKind,
}

/// `genai_semconv.py:92-107` — `_DURATION_BUCKET_BOUNDARIES`.
const DURATION_BOUNDS: &[f64] = &[
    0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56, 5.12, 10.24, 20.48, 40.96, 81.92,
];

/// `genai_semconv.py:109-133` — `_TTFT_BUCKET_BOUNDARIES`.
const TTFT_BOUNDS: &[f64] = &[
    0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2, 0.25, 0.3, 0.35,
    0.4, 0.45, 0.5, 0.75, 1.0, 2.0, 5.0,
];

/// `genai_semconv.py:135-159` — `_TIME_PER_OUTPUT_CHUNK_BUCKET_BOUNDARIES`.
const TIME_PER_OUTPUT_CHUNK_BOUNDS: &[f64] = &[
    0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2, 0.25, 0.3, 0.35,
    0.4, 0.45, 0.5, 0.75, 1.0, 2.0, 5.0,
];

/// `genai_semconv.py:161-175` — `_TOKEN_USAGE_BUCKET_BOUNDARIES`.
const TOKEN_USAGE_BOUNDS: &[f64] = &[
    1.0, 4.0, 16.0, 64.0, 256.0, 1024.0, 4096.0, 16384.0, 65536.0, 262144.0, 1048576.0, 4194304.0,
    16777216.0,
];

/// The three GenAI duration histograms in a stable emission order.
const DURATION_METRICS: &[DurationSpec] = &[
    DurationSpec {
        report_key: "request_latency",
        spec_name: "gen_ai.client.operation.duration",
        bounds: DURATION_BOUNDS,
        kind: DurationKind::RequestLatency,
    },
    DurationSpec {
        report_key: "time_to_first_token",
        spec_name: "gen_ai.client.operation.time_to_first_chunk",
        bounds: TTFT_BOUNDS,
        kind: DurationKind::TimeToFirstToken,
    },
    DurationSpec {
        report_key: "inter_token_latency",
        spec_name: "gen_ai.client.operation.time_per_output_chunk",
        bounds: TIME_PER_OUTPUT_CHUNK_BOUNDS,
        kind: DurationKind::InterTokenLatency,
    },
];

/// Report metric key carrying the **input** token count. Python's semconv map
/// names this source `input_token_count`; the aiperf metric that actually
/// carries input-token counts is Input Sequence Length. Both keys are accepted
/// so the emitter fires whichever the report exposes.
const INPUT_TOKEN_KEYS: &[&str] = &["input_token_count", "input_sequence_length"];

/// Report metric key carrying the **output** token count.
const OUTPUT_TOKEN_KEYS: &[&str] = &["output_token_count", "output_sequence_length"];

/// Build one duration histogram metric (unit `s`), one data point per report
/// series. Absent, non-distribution, or empty metrics emit nothing.
fn duration_metric(
    report: &NativeReport,
    spec: &DurationSpec,
    ctx: &EmitContext,
) -> Option<proto::Metric> {
    let entry = report.metrics.get(spec.report_key)?;
    let scale = seconds_scale(&entry.unit);
    let points: Vec<proto::HistogramDataPoint> = entry
        .series
        .iter()
        .filter_map(|series| {
            let dist = distribution(series)?;
            let attrs = duration_attributes(ctx, series);
            histogram_point(dist, spec.bounds, scale, attrs, ctx)
        })
        .collect();
    if points.is_empty() {
        return None;
    }
    Some(histogram_metric(spec.spec_name, "s", points))
}

/// Build the merged `gen_ai.client.token.usage` histogram (unit `{token}`) with
/// `gen_ai.token.type=input` and `type=output` data points, matching
/// `genai_semconv.TOKEN_USAGE_SPECIAL_CASE` (`genai_semconv.py:216-219`) where
/// two aiperf metrics collapse into one spec histogram.
fn token_usage_metric(report: &NativeReport, ctx: &EmitContext) -> Option<proto::Metric> {
    let mut points = Vec::new();
    for point in token_points(report, INPUT_TOKEN_KEYS, "input", ctx) {
        points.push(point);
    }
    for point in token_points(report, OUTPUT_TOKEN_KEYS, "output", ctx) {
        points.push(point);
    }
    if points.is_empty() {
        return None;
    }
    Some(histogram_metric(
        "gen_ai.client.token.usage",
        "{token}",
        points,
    ))
}

/// Data points for one token direction. Token counts are identity-scaled (no
/// unit conversion, matching `genai_semconv._identity`).
fn token_points(
    report: &NativeReport,
    keys: &[&str],
    token_type: &str,
    ctx: &EmitContext,
) -> Vec<proto::HistogramDataPoint> {
    let Some(entry) = keys.iter().find_map(|key| report.metrics.get(*key)) else {
        return Vec::new();
    };
    entry
        .series
        .iter()
        .filter_map(|series| {
            let dist = distribution(series)?;
            let mut attrs = duration_attributes(ctx, series);
            attrs.push(key_value("gen_ai.token.type", token_type));
            histogram_point(dist, TOKEN_USAGE_BOUNDS, 1.0, attrs, ctx)
        })
        .collect()
}

/// Build one duration histogram metric from the per-record accumulator, one data
/// point per observed spec `error.type` (absent on the success path). Populated
/// `bucket_counts` are carried through from the merged per-record observations.
fn duration_metric_from_records(
    records: &OtelRecordAccumulator,
    spec: &DurationSpec,
    ctx: &EmitContext,
) -> Option<proto::Metric> {
    let points: Vec<proto::HistogramDataPoint> = records
        .duration_series(spec.kind)
        .map(|(error_type, histogram)| {
            let mut attributes = base_attributes(ctx);
            if let Some(error_type) = error_type {
                attributes.push(key_value("error.type", error_type));
            }
            populated_histogram_point(histogram, attributes, ctx)
        })
        .collect();
    if points.is_empty() {
        return None;
    }
    Some(histogram_metric(spec.spec_name, "s", points))
}

/// Build the merged `gen_ai.client.token.usage` histogram from the per-record
/// accumulator, one data point per `gen_ai.token.type`. Token usage carries no
/// `error.type` attribute (mirroring `_build_token_usage_attributes`).
fn token_usage_metric_from_records(
    records: &OtelRecordAccumulator,
    ctx: &EmitContext,
) -> Option<proto::Metric> {
    let mut points = Vec::new();
    for (kind, token_type) in [(TokenKind::Input, "input"), (TokenKind::Output, "output")] {
        if let Some(histogram) = records.token_series(kind) {
            let mut attributes = base_attributes(ctx);
            attributes.push(key_value("gen_ai.token.type", token_type));
            points.push(populated_histogram_point(histogram, attributes, ctx));
        }
    }
    if points.is_empty() {
        return None;
    }
    Some(histogram_metric(
        "gen_ai.client.token.usage",
        "{token}",
        points,
    ))
}

/// The run-level GenAI attribute set (`gen_ai.operation.name`,
/// `gen_ai.provider.name`, `gen_ai.request.model`) shared by every per-record
/// data point. Mirrors `_build_duration_attributes` (`genai_semconv.py:303-316`)
/// minus the per-record `error.type`, which the caller appends. The model comes
/// from run config (`cfg.get_model_names()[0]`), constant across records.
fn base_attributes(ctx: &EmitContext) -> Vec<proto::KeyValue> {
    let mut attrs = vec![
        key_value("gen_ai.operation.name", &ctx.operation_name),
        key_value("gen_ai.provider.name", &ctx.provider),
    ];
    if let Some(model) = ctx.model.as_ref() {
        attrs.push(key_value("gen_ai.request.model", model));
    }
    attrs
}

/// Build one populated histogram data point from a merged per-record histogram.
/// Carries the real `count`/`sum`/`min`/`max` and the per-bucket counts (whose
/// sum equals `count`, the OTLP invariant).
fn populated_histogram_point(
    histogram: &BucketHistogram,
    attributes: Vec<proto::KeyValue>,
    ctx: &EmitContext,
) -> proto::HistogramDataPoint {
    proto::HistogramDataPoint {
        attributes,
        start_time_unix_nano: ctx.start_ns,
        time_unix_nano: ctx.end_ns,
        count: histogram.count(),
        sum: Some(histogram.sum()),
        bucket_counts: histogram.bucket_counts().to_vec(),
        explicit_bounds: histogram.bounds().to_vec(),
        flags: 0,
        min: histogram.min(),
        max: histogram.max(),
    }
}

/// Build the GenAI per-datapoint attribute set common to every metric:
/// `gen_ai.operation.name`, `gen_ai.provider.name`, and `gen_ai.request.model`
/// (from the series' `model` label when present, else the resource model).
/// Mirrors `_build_duration_attributes` (`genai_semconv.py:303-316`), minus the
/// per-record `error.type` — the aggregate report has no per-error breakdown.
fn duration_attributes(ctx: &EmitContext, series: &MetricSeries) -> Vec<proto::KeyValue> {
    let mut attrs = vec![
        key_value("gen_ai.operation.name", &ctx.operation_name),
        key_value("gen_ai.provider.name", &ctx.provider),
    ];
    let model = series
        .labels
        .as_ref()
        .and_then(|labels| labels.get("model"))
        .cloned()
        .or_else(|| ctx.model.clone());
    if let Some(model) = model {
        attrs.push(key_value("gen_ai.request.model", &model));
    }
    attrs
}

/// Assemble a `Histogram`-shaped [`proto::Metric`] with cumulative temporality.
fn histogram_metric(
    name: &str,
    unit: &str,
    data_points: Vec<proto::HistogramDataPoint>,
) -> proto::Metric {
    proto::Metric {
        name: name.to_string(),
        description: format!("GenAI semconv metric: {name}"),
        unit: unit.to_string(),
        data: Some(proto::metric::Data::Histogram(proto::Histogram {
            data_points,
            aggregation_temporality: proto::AGGREGATION_TEMPORALITY_CUMULATIVE,
        })),
    }
}

/// Build one aggregate histogram data point. `bucket_counts` are all zero: the
/// aggregate cannot reconstruct the per-bucket distribution (the documented
/// fidelity gap vs the Python per-record path). `count`/`sum`/`min`/`max` carry
/// the aggregate; non-finite tails are omitted. Returns `None` for empty
/// distributions so no zero-count point is emitted.
fn histogram_point(
    dist: &ReportDistributionStats,
    bounds: &'static [f64],
    scale: f64,
    attributes: Vec<proto::KeyValue>,
    ctx: &EmitContext,
) -> Option<proto::HistogramDataPoint> {
    let count = dist.count? as u64;
    if count == 0 {
        return None;
    }
    let avg = finite(dist.avg).map(|value| value * scale);
    let sum = avg.map(|avg| avg * count as f64);
    Some(proto::HistogramDataPoint {
        attributes,
        start_time_unix_nano: ctx.start_ns,
        time_unix_nano: ctx.end_ns,
        count,
        sum,
        // OTLP requires bucket_counts.len() == explicit_bounds.len() + 1.
        bucket_counts: vec![0u64; bounds.len() + 1],
        explicit_bounds: bounds.to_vec(),
        flags: 0,
        min: finite(dist.min).map(|value| value * scale),
        max: finite(dist.max).map(|value| value * scale),
    })
}

/// The distribution stats of a series, or `None` when the series is not a
/// distribution (scalars/counters have no histogram analogue).
fn distribution(series: &MetricSeries) -> Option<&ReportDistributionStats> {
    match &series.stats {
        ReportStats::Distribution(dist) => Some(dist),
        _ => None,
    }
}

/// A finite report value, or `None` for absent/non-finite tails.
fn finite(value: Option<ReportValue>) -> Option<f64> {
    value.and_then(crate::export::finite_passthrough)
}

/// Multiplier converting a report display unit to seconds. The report has
/// already applied the metric's ns→display scaling; latency metrics display in
/// `ms`, so the net conversion to the spec's `s` is `ms→s`. Equivalent to
/// Python's `_ns_to_s` applied to the raw per-record value.
fn seconds_scale(unit: &str) -> f64 {
    match unit {
        "ns" => 1e-9,
        "us" => 1e-6,
        "ms" => 1e-3,
        "sec" | "s" => 1.0,
        // Unknown unit: treat as already in seconds rather than corrupt the value.
        _ => 1.0,
    }
}

/// Map `aiperf.endpoint.type` to `gen_ai.operation.name`, matching
/// `genai_semconv._map_operation_name` (`genai_semconv.py:227-239`): chat →
/// `chat`, completions → `text_completion`, embeddings → `embeddings`, else
/// `chat`.
fn operation_name(cfg: &OtelExportConfig) -> String {
    let endpoint_type = cfg
        .resource_attributes
        .get("aiperf.endpoint.type")
        .map(|value| value.to_ascii_lowercase())
        .unwrap_or_default();
    match endpoint_type.as_str() {
        "completions" => "text_completion",
        "embeddings" => "embeddings",
        _ => "chat",
    }
    .to_string()
}

/// Build the OTLP resource attributes: `service.name=aiperf` (constant, matching
/// `_build_resource_attributes` `otel_metrics_results_processor.py:436`) plus
/// everything the frontend projected. A projected `service.name` overrides.
fn resource_attributes(cfg: &OtelExportConfig) -> Vec<proto::KeyValue> {
    let mut merged: BTreeMap<String, String> = BTreeMap::new();
    merged.insert("service.name".to_string(), "aiperf".to_string());
    for (key, value) in &cfg.resource_attributes {
        merged.insert(key.clone(), value.clone());
    }
    merged
        .into_iter()
        .map(|(key, value)| key_value(&key, &value))
        .collect()
}

/// A string-valued OTLP `KeyValue`.
fn key_value(key: &str, value: &str) -> proto::KeyValue {
    proto::KeyValue {
        key: key.to_string(),
        value: Some(proto::AnyValue {
            value: Some(proto::any_value::Value::StringValue(value.to_string())),
        }),
    }
}

/// POST the encoded OTLP request under a hard timeout on a throwaway
/// `current_thread` runtime (the commit site is sync with no ambient runtime).
fn post_otlp(endpoint: &str, body: Vec<u8>) -> anyhow::Result<()> {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|error| anyhow::anyhow!("otel: build runtime: {error}"))?;
    runtime.block_on(async move {
        tokio::time::timeout(EXPORT_TIMEOUT, send_request(endpoint, body))
            .await
            .map_err(|_| anyhow::anyhow!("otel: OTLP POST timed out after {EXPORT_TIMEOUT:?}"))?
    })
}

/// One-shot HTTP/1.1 POST of `application/x-protobuf` to the OTLP endpoint over
/// hyper (http and https). Non-2xx responses are surfaced as errors so the
/// operator sees the warning; the committed report is unaffected.
async fn send_request(endpoint: &str, body: Vec<u8>) -> anyhow::Result<()> {
    use http_body_util::{BodyExt, Full};
    use hyper::body::Bytes;

    let url = url::Url::parse(endpoint)
        .map_err(|error| anyhow::anyhow!("otel: invalid endpoint {endpoint:?}: {error}"))?;
    let host = url
        .host_str()
        .ok_or_else(|| anyhow::anyhow!("otel: endpoint has no host: {endpoint:?}"))?
        .to_string();
    let https = matches!(url.scheme(), "https");
    let port = url.port().unwrap_or(if https { 443 } else { 80 });
    let authority = format!("{host}:{port}");
    let path = match url.query() {
        Some(query) => format!("{}?{}", url.path(), query),
        None => url.path().to_string(),
    };

    let request = hyper::Request::builder()
        .method(hyper::Method::POST)
        .uri(path)
        .header(hyper::header::HOST, &authority)
        .header(hyper::header::CONTENT_TYPE, "application/x-protobuf")
        .body(Full::new(Bytes::from(body)))
        .map_err(|error| anyhow::anyhow!("otel: build request: {error}"))?;

    let tcp = tokio::net::TcpStream::connect(&authority)
        .await
        .map_err(|error| anyhow::anyhow!("otel: connect {authority}: {error}"))?;

    let response = if https {
        let stream = tls_connect(tcp, &host).await?;
        send_over(stream, request).await?
    } else {
        send_over(tcp, request).await?
    };

    let status = response.status();
    // Drain the body so the connection closes cleanly; content is not consumed.
    let _ = response.into_body().collect().await;
    if !status.is_success() {
        anyhow::bail!("otel: collector returned HTTP {status}");
    }
    Ok(())
}

/// Perform the http1 handshake over `io` and send the request, returning the
/// response.
async fn send_over<I>(
    io: I,
    request: hyper::Request<http_body_util::Full<hyper::body::Bytes>>,
) -> anyhow::Result<hyper::Response<hyper::body::Incoming>>
where
    I: tokio::io::AsyncRead + tokio::io::AsyncWrite + Unpin + Send + 'static,
{
    let (mut sender, connection) =
        hyper::client::conn::http1::handshake(hyper_util::rt::TokioIo::new(io))
            .await
            .map_err(|error| anyhow::anyhow!("otel: handshake: {error}"))?;
    // Drive the connection to completion alongside the request.
    tokio::task::spawn(async move {
        let _ = connection.await;
    });
    sender
        .send_request(request)
        .await
        .map_err(|error| anyhow::anyhow!("otel: send: {error}"))
}

/// Wrap a TCP stream in TLS using webpki roots (mirrors the transport-http
/// rustls config).
async fn tls_connect(
    tcp: tokio::net::TcpStream,
    host: &str,
) -> anyhow::Result<tokio_rustls::client::TlsStream<tokio::net::TcpStream>> {
    use std::sync::Arc;

    let mut roots = rustls::RootCertStore::empty();
    roots.extend(webpki_roots::TLS_SERVER_ROOTS.iter().cloned());
    let config = rustls::ClientConfig::builder()
        .with_root_certificates(roots)
        .with_no_client_auth();
    let connector = tokio_rustls::TlsConnector::from(Arc::new(config));
    let server_name = rustls::pki_types::ServerName::try_from(host.to_string())
        .map_err(|error| anyhow::anyhow!("otel: invalid TLS server name {host:?}: {error}"))?;
    connector
        .connect(server_name, tcp)
        .await
        .map_err(|error| anyhow::anyhow!("otel: TLS handshake: {error}"))
}

/// Minimal OTLP metrics protobuf subset (OTLP v1) needed to encode an
/// `ExportMetricsServiceRequest` of `Histogram` metrics.
///
/// Hand-written over the crate's existing `prost` so the shipped runner links no
/// OpenTelemetry SDK. Field numbers are fixed by the OTLP proto contract
/// (`opentelemetry/proto/{collector/metrics,metrics,common,resource}/v1`); the
/// module test decodes emitted bytes with the authoritative `opentelemetry-proto`
/// crate, which validates every tag here.
mod proto {
    /// OTLP `AggregationTemporality::CUMULATIVE`.
    pub const AGGREGATION_TEMPORALITY_CUMULATIVE: i32 = 2;

    /// `ExportMetricsServiceRequest`.
    #[derive(Clone, PartialEq, prost::Message)]
    pub struct ExportMetricsServiceRequest {
        /// One entry per resource; the sink emits exactly one.
        #[prost(message, repeated, tag = "1")]
        pub resource_metrics: Vec<ResourceMetrics>,
    }

    /// `metrics.v1.ResourceMetrics`.
    #[derive(Clone, PartialEq, prost::Message)]
    pub struct ResourceMetrics {
        /// Resource attributes for every contained metric.
        #[prost(message, optional, tag = "1")]
        pub resource: Option<Resource>,
        /// Scoped metric groups.
        #[prost(message, repeated, tag = "2")]
        pub scope_metrics: Vec<ScopeMetrics>,
        /// Resource schema URL (unused).
        #[prost(string, tag = "3")]
        pub schema_url: String,
    }

    /// `resource.v1.Resource`.
    #[derive(Clone, PartialEq, prost::Message)]
    pub struct Resource {
        /// Resource-level attributes.
        #[prost(message, repeated, tag = "1")]
        pub attributes: Vec<KeyValue>,
        /// Count of dropped attributes (always 0 here).
        #[prost(uint32, tag = "2")]
        pub dropped_attributes_count: u32,
    }

    /// `metrics.v1.ScopeMetrics`.
    #[derive(Clone, PartialEq, prost::Message)]
    pub struct ScopeMetrics {
        /// Instrumentation scope identity.
        #[prost(message, optional, tag = "1")]
        pub scope: Option<InstrumentationScope>,
        /// Metrics in this scope.
        #[prost(message, repeated, tag = "2")]
        pub metrics: Vec<Metric>,
        /// Scope schema URL (unused).
        #[prost(string, tag = "3")]
        pub schema_url: String,
    }

    /// `common.v1.InstrumentationScope`.
    #[derive(Clone, PartialEq, prost::Message)]
    pub struct InstrumentationScope {
        /// Meter name.
        #[prost(string, tag = "1")]
        pub name: String,
        /// Meter version.
        #[prost(string, tag = "2")]
        pub version: String,
        /// Scope attributes (unused).
        #[prost(message, repeated, tag = "3")]
        pub attributes: Vec<KeyValue>,
        /// Count of dropped attributes (always 0 here).
        #[prost(uint32, tag = "4")]
        pub dropped_attributes_count: u32,
    }

    /// `metrics.v1.Metric` restricted to the histogram `data` variant.
    #[derive(Clone, PartialEq, prost::Message)]
    pub struct Metric {
        /// Spec metric name.
        #[prost(string, tag = "1")]
        pub name: String,
        /// Human-readable description.
        #[prost(string, tag = "2")]
        pub description: String,
        /// Metric unit.
        #[prost(string, tag = "3")]
        pub unit: String,
        /// The metric data payload; only `Histogram` (tag 9) is emitted.
        #[prost(oneof = "metric::Data", tags = "9")]
        pub data: Option<metric::Data>,
    }

    /// `Metric.data` oneof.
    pub mod metric {
        /// The metric data variants this sink emits.
        #[derive(Clone, PartialEq, prost::Oneof)]
        pub enum Data {
            /// Explicit-bucket histogram.
            #[prost(message, tag = "9")]
            Histogram(super::Histogram),
        }
    }

    /// `metrics.v1.Histogram`.
    #[derive(Clone, PartialEq, prost::Message)]
    pub struct Histogram {
        /// Histogram data points.
        #[prost(message, repeated, tag = "1")]
        pub data_points: Vec<HistogramDataPoint>,
        /// Aggregation temporality enum value.
        #[prost(int32, tag = "2")]
        pub aggregation_temporality: i32,
    }

    /// `metrics.v1.HistogramDataPoint`.
    #[derive(Clone, PartialEq, prost::Message)]
    pub struct HistogramDataPoint {
        /// Per-datapoint attributes.
        #[prost(message, repeated, tag = "9")]
        pub attributes: Vec<KeyValue>,
        /// Window start in unix nanoseconds.
        #[prost(fixed64, tag = "2")]
        pub start_time_unix_nano: u64,
        /// Window end in unix nanoseconds.
        #[prost(fixed64, tag = "3")]
        pub time_unix_nano: u64,
        /// Total observation count.
        #[prost(fixed64, tag = "4")]
        pub count: u64,
        /// Optional observation sum.
        #[prost(double, optional, tag = "5")]
        pub sum: Option<f64>,
        /// Per-bucket counts (`explicit_bounds.len() + 1`).
        #[prost(fixed64, repeated, tag = "6")]
        pub bucket_counts: Vec<u64>,
        /// Explicit bucket boundaries.
        #[prost(double, repeated, tag = "7")]
        pub explicit_bounds: Vec<f64>,
        /// Data-point flags (0).
        #[prost(uint32, tag = "10")]
        pub flags: u32,
        /// Optional minimum observation.
        #[prost(double, optional, tag = "11")]
        pub min: Option<f64>,
        /// Optional maximum observation.
        #[prost(double, optional, tag = "12")]
        pub max: Option<f64>,
    }

    /// `common.v1.KeyValue`.
    #[derive(Clone, PartialEq, prost::Message)]
    pub struct KeyValue {
        /// Attribute key.
        #[prost(string, tag = "1")]
        pub key: String,
        /// Attribute value.
        #[prost(message, optional, tag = "2")]
        pub value: Option<AnyValue>,
    }

    /// `common.v1.AnyValue` restricted to string values (all attrs are strings).
    #[derive(Clone, PartialEq, prost::Message)]
    pub struct AnyValue {
        /// The wrapped value.
        #[prost(oneof = "any_value::Value", tags = "1")]
        pub value: Option<any_value::Value>,
    }

    /// `AnyValue.value` oneof.
    pub mod any_value {
        /// The value variants this sink emits.
        #[derive(Clone, PartialEq, prost::Oneof)]
        pub enum Value {
            /// UTF-8 string value.
            #[prost(string, tag = "1")]
            StringValue(String),
        }
    }
}

#[cfg(test)]
mod tests;
