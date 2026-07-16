// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Phase-bounded inference-server Prometheus collection.
//!
//! The native telemetry design addendum replaces
//! the inherited asynchronous scrape races with one sequential Clock-driven
//! loop and forces exact snapshots at every warmup/profiling barrier.

use std::cell::{Cell, RefCell};
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::rc::Rc;

use crate::clock::Clock;
use crate::metrics_core::{
    Phase, ReportServerMetricsEndpointInfo, ReportServerMetricsMetadata,
    ReportServerMetricsPhaseRange,
};
use crate::phase_runtime::ScheduledPhaseSidecar;
use crate::server_metrics::{
    MetricSample, PrometheusHttpSource, PrometheusMetricType, ServerMetricsAccumulator,
    ServerMetricsPhaseBoundary, ServerMetricsRecord, ServerMetricsScrapeMode,
    ServerMetricsScrapeOutcome, ServerMetricsSource, ServerMetricsSummary,
};
use crate::transport_http::config::ClientConfig;
use crate::transport_http::transport::http_transport::HttpTransport;
use anyhow::{Context, Result, ensure};
use serde_json::{Map, Value, json};
use tokio::sync::Notify;
use tokio::task::JoinHandle;

use crate::engine::protocol::ServerMetricsSpec;

/// Run-owned server telemetry shared by phase-specific lifecycle adapters.
pub(crate) struct ServerMetricsRun {
    state: Rc<ServerMetricsState>,
}

impl ServerMetricsRun {
    /// Build every configured source over one Clock-injected native transport.
    pub(crate) fn new(spec: &ServerMetricsSpec, clock: Rc<dyn Clock>) -> Result<Self> {
        ensure!(
            spec.collection_interval_ns > 0,
            "server metrics collection_interval_ns must be positive"
        );
        ensure!(
            spec.reachability_timeout_ns > 0,
            "server metrics reachability_timeout_ns must be positive"
        );
        ensure!(
            !spec.urls.is_empty(),
            "server metrics requires at least one endpoint URL"
        );
        let transport = Rc::new(HttpTransport::new(
            clock.clone(),
            ClientConfig {
                connect_timeout_ns: Some(spec.reachability_timeout_ns),
                request_timeout_ns: None,
                total_timeout_ns: None,
                ..ClientConfig::default()
            },
        ));
        let candidates: Vec<Rc<dyn ServerMetricsSource>> = spec
            .urls
            .iter()
            .map(|url| {
                Rc::new(PrometheusHttpSource::new(
                    clock.clone(),
                    transport.clone(),
                    url.clone(),
                )) as Rc<dyn ServerMetricsSource>
            })
            .collect();
        let configured_urls = candidates
            .iter()
            .map(|source| source.endpoint_url())
            .collect();
        Ok(Self {
            state: Rc::new(ServerMetricsState {
                clock,
                collection_interval_ns: spec.collection_interval_ns,
                configured_urls,
                candidates,
                active: RefCell::new(Vec::new()),
                accumulator: RefCell::new(ServerMetricsAccumulator::new()),
                activated: Cell::new(false),
            }),
        })
    }

    /// Bind the shared collector to one authoritative benchmark phase.
    pub(crate) fn sidecar(&self, phase: Phase) -> Rc<dyn ScheduledPhaseSidecar> {
        Rc::new(ServerMetricsPhaseSidecar::new(self.state.clone(), phase))
    }

    /// Compute native server metric series for one phase.
    pub(crate) fn summarize(
        &self,
        phase: Phase,
        slice_duration_ns: Option<i64>,
    ) -> ServerMetricsSummary {
        self.state
            .accumulator
            .borrow()
            .summarize_phase(phase, slice_duration_ns)
    }

    /// Configured endpoint URLs in stable Config-v2 order.
    pub(crate) fn configured_urls(&self) -> &[String] {
        &self.state.configured_urls
    }

    /// Build additive native-v2 metadata used by compatibility renderers.
    pub(crate) fn report_metadata(
        &self,
        profiling: &ServerMetricsSummary,
        warmup: Option<&ServerMetricsSummary>,
    ) -> ReportServerMetricsMetadata {
        let mut descriptions = profiling.descriptions().clone();
        let mut metric_types = profiling
            .metric_types()
            .iter()
            .map(|(name, metric_type)| (name.clone(), metric_type_name(*metric_type).to_string()))
            .collect::<BTreeMap<_, _>>();
        if let Some(warmup) = warmup {
            for (name, description) in warmup.descriptions() {
                descriptions
                    .entry(name.clone())
                    .or_insert_with(|| description.clone());
            }
            for (name, metric_type) in warmup.metric_types() {
                metric_types
                    .entry(name.clone())
                    .or_insert_with(|| metric_type_name(*metric_type).to_string());
            }
        }
        let endpoint_info = self
            .state
            .accumulator
            .borrow()
            .endpoint_info()
            .into_iter()
            .map(|(endpoint, info)| {
                (
                    endpoint,
                    ReportServerMetricsEndpointInfo {
                        total_fetches: info.total_fetches,
                        first_fetch_ns: info.first_fetch_ns,
                        last_fetch_ns: info.last_fetch_ns,
                        avg_fetch_latency_ms: info.avg_fetch_latency_ms,
                        unique_updates: info.unique_updates,
                        first_update_ns: info.first_update_ns,
                        last_update_ns: info.last_update_ns,
                        duration_seconds: info.duration_seconds,
                        avg_update_interval_ms: info.avg_update_interval_ms,
                        median_update_interval_ms: info.median_update_interval_ms,
                    },
                )
            })
            .collect();
        ReportServerMetricsMetadata {
            endpoints_configured: self.configured_urls().to_vec(),
            endpoints_successful: profiling.endpoints_successful().to_vec(),
            descriptions,
            metric_types,
            endpoint_info,
            profiling: profiling
                .boundary()
                .map(|boundary| ReportServerMetricsPhaseRange {
                    start_ns: boundary.start_ns,
                    end_ns: boundary.end_ns,
                }),
            warmup: warmup
                .and_then(ServerMetricsSummary::boundary)
                .map(|boundary| ReportServerMetricsPhaseRange {
                    start_ns: boundary.start_ns,
                    end_ns: boundary.end_ns,
                }),
        }
    }

    /// Write Python-compatible slim scrape records.
    pub(crate) fn write_slim_jsonl(&self, path: &Path) -> Result<()> {
        JsonlServerMetricsArtifactSink.write(path, self.state.accumulator.borrow().records(), true)
    }

    /// Write full records used only by Python's canonical Parquet renderer.
    pub(crate) fn write_parquet_wire_jsonl(&self, path: &Path) -> Result<()> {
        JsonlServerMetricsArtifactSink.write(path, self.state.accumulator.borrow().records(), false)
    }
}

struct ServerMetricsState {
    clock: Rc<dyn Clock>,
    collection_interval_ns: i64,
    configured_urls: Vec<String>,
    candidates: Vec<Rc<dyn ServerMetricsSource>>,
    active: RefCell<Vec<Rc<dyn ServerMetricsSource>>>,
    accumulator: RefCell<ServerMetricsAccumulator>,
    activated: Cell<bool>,
}

#[derive(Clone)]
struct ServerMetricsPhaseSidecar {
    state: Rc<ServerMetricsState>,
    phase: Phase,
    start_records: Rc<RefCell<BTreeMap<String, ServerMetricsRecord>>>,
    stop: Rc<Notify>,
    task: Rc<RefCell<Option<JoinHandle<()>>>>,
    started: Rc<Cell<bool>>,
    finished: Rc<Cell<bool>>,
    phase_start_ns: Rc<Cell<Option<i64>>>,
    phase_end_ns: Rc<Cell<Option<i64>>>,
}

impl ServerMetricsPhaseSidecar {
    fn new(state: Rc<ServerMetricsState>, phase: Phase) -> Self {
        Self {
            state,
            phase,
            start_records: Rc::new(RefCell::new(BTreeMap::new())),
            stop: Rc::new(Notify::new()),
            task: Rc::new(RefCell::new(None)),
            started: Rc::new(Cell::new(false)),
            finished: Rc::new(Cell::new(false)),
            phase_start_ns: Rc::new(Cell::new(None)),
            phase_end_ns: Rc::new(Cell::new(None)),
        }
    }

    async fn start_inner(&self) -> Result<()> {
        if self.started.replace(true) {
            return Ok(());
        }
        let activating = !self.state.activated.replace(true);
        let sources = if activating {
            self.state.candidates.clone()
        } else {
            self.state.active.borrow().clone()
        };
        let mut successful = Vec::new();
        let mut records = BTreeMap::new();
        for source in sources {
            match source.scrape(ServerMetricsScrapeMode::Boundary).await {
                Ok(ServerMetricsScrapeOutcome::Record(mut record)) => {
                    record.benchmark_phase = Some(self.phase);
                    records.insert(record.endpoint_url.clone(), record.clone());
                    self.state.accumulator.borrow_mut().ingest_record(record);
                    successful.push(source);
                }
                Ok(ServerMetricsScrapeOutcome::Empty) => successful.push(source),
                Ok(ServerMetricsScrapeOutcome::Disabled) => {}
                Err(error) => tracing::warn!(
                    source = %source.endpoint_url(),
                    error = %error,
                    "server metrics skipped unreachable source"
                ),
            }
        }
        if activating {
            *self.state.active.borrow_mut() = successful;
        } else {
            self.retain_sources(&successful);
        }
        *self.start_records.borrow_mut() = records;
        if self.state.active.borrow().is_empty() {
            return Ok(());
        }

        let sidecar = self.clone();
        *self.task.borrow_mut() = Some(tokio::task::spawn_local(async move {
            sidecar.collect_continuously().await;
        }));
        Ok(())
    }

    async fn collect_continuously(self) {
        loop {
            let sleep = self
                .state
                .clock
                .clone()
                .sleep(self.state.collection_interval_ns);
            let stopped = self.stop.notified();
            tokio::pin!(sleep);
            tokio::pin!(stopped);
            tokio::select! {
                biased;
                () = &mut stopped => return,
                () = &mut sleep => {}
            }
            let sources = self.state.active.borrow().clone();
            for source in sources {
                match source.scrape(ServerMetricsScrapeMode::Continuous).await {
                    Ok(ServerMetricsScrapeOutcome::Record(mut record)) => {
                        record.benchmark_phase = Some(self.phase);
                        self.state.accumulator.borrow_mut().ingest_record(record);
                    }
                    Ok(ServerMetricsScrapeOutcome::Empty) => {}
                    Ok(ServerMetricsScrapeOutcome::Disabled) => self.remove_source(&source),
                    Err(error) => {
                        tracing::warn!(
                            source = %source.endpoint_url(),
                            error = %error,
                            "server metrics cadence scrape failed"
                        );
                        if error.is_incompatible() {
                            self.remove_source(&source);
                        }
                    }
                }
            }
        }
    }

    async fn finish_inner(&self) -> Result<()> {
        if self.finished.replace(true) {
            return Ok(());
        }
        self.stop.notify_one();
        let task = self.task.borrow_mut().take();
        if let Some(task) = task
            && let Err(error) = task.await
        {
            tracing::warn!(error = %error, "server metrics cadence task failed");
        }

        let sources = self.state.active.borrow().clone();
        let mut end_records = BTreeMap::new();
        for source in sources {
            match source.scrape(ServerMetricsScrapeMode::Boundary).await {
                Ok(ServerMetricsScrapeOutcome::Record(mut record)) => {
                    record.benchmark_phase = Some(self.phase);
                    end_records.insert(record.endpoint_url.clone(), record.clone());
                    self.state.accumulator.borrow_mut().ingest_record(record);
                }
                Ok(ServerMetricsScrapeOutcome::Empty) => {}
                Ok(ServerMetricsScrapeOutcome::Disabled) => self.remove_source(&source),
                Err(error) => {
                    tracing::warn!(
                        source = %source.endpoint_url(),
                        error = %error,
                        "server metrics final scrape failed"
                    );
                    if error.is_incompatible() {
                        self.remove_source(&source);
                    }
                }
            }
        }
        let start_records = self.start_records.borrow().clone();
        let start_ns = self.phase_start_ns.get();
        let end_ns = self.phase_end_ns.get();
        if let (Some(start_ns), Some(end_ns)) = (start_ns, end_ns) {
            self.state
                .accumulator
                .borrow_mut()
                .set_phase_boundary(ServerMetricsPhaseBoundary {
                    phase: self.phase,
                    start_ns,
                    end_ns,
                    start_records,
                    end_records,
                });
        }
        Ok(())
    }

    fn retain_sources(&self, successful: &[Rc<dyn ServerMetricsSource>]) {
        self.state.active.borrow_mut().retain(|candidate| {
            successful
                .iter()
                .any(|source| Rc::ptr_eq(source, candidate))
        });
    }

    fn remove_source(&self, removed: &Rc<dyn ServerMetricsSource>) {
        self.state
            .active
            .borrow_mut()
            .retain(|source| !Rc::ptr_eq(source, removed));
    }
}

impl ScheduledPhaseSidecar for ServerMetricsPhaseSidecar {
    fn start(&self) -> crate::timing::LocalPhaseFuture<Result<()>> {
        let sidecar = self.clone();
        Box::pin(async move { sidecar.start_inner().await })
    }

    fn on_phase_start(&self, timestamp_ns: i64) {
        self.phase_start_ns.set(Some(timestamp_ns));
    }

    fn on_phase_end(&self, timestamp_ns: i64) {
        self.phase_end_ns.set(Some(timestamp_ns));
    }

    fn finish(&self) -> crate::timing::LocalPhaseFuture<Result<()>> {
        let sidecar = self.clone();
        Box::pin(async move { sidecar.finish_inner().await })
    }
}

trait ServerMetricsArtifactSink {
    fn write(&self, path: &Path, records: &[ServerMetricsRecord], slim: bool) -> Result<()>;
}

struct JsonlServerMetricsArtifactSink;

impl ServerMetricsArtifactSink for JsonlServerMetricsArtifactSink {
    fn write(&self, path: &Path, records: &[ServerMetricsRecord], slim: bool) -> Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).with_context(|| {
                format!(
                    "creating server metrics export directory {}",
                    parent.display()
                )
            })?;
        }
        let file = File::create(path)
            .with_context(|| format!("creating server metrics export {}", path.display()))?;
        let mut writer = BufWriter::new(file);
        for record in records {
            if slim && record.is_duplicate {
                continue;
            }
            let row = if slim {
                slim_record(record)
            } else {
                full_record(record)
            };
            serde_json::to_writer(&mut writer, &row)
                .with_context(|| format!("serializing server metrics export {}", path.display()))?;
            writer
                .write_all(b"\n")
                .with_context(|| format!("writing server metrics export {}", path.display()))?;
        }
        writer
            .flush()
            .with_context(|| format!("flushing server metrics export {}", path.display()))
    }
}

fn slim_record(record: &ServerMetricsRecord) -> Value {
    let metrics = record
        .metrics
        .iter()
        .filter(|(name, _)| !name.ends_with("_info"))
        .map(|(name, family)| {
            (
                name.clone(),
                Value::Array(family.samples.iter().map(sample_value).collect()),
            )
        })
        .collect::<Map<_, _>>();
    json!({
        "endpoint_url": record.endpoint_url,
        "timestamp_ns": record.timestamp_ns,
        "endpoint_latency_ns": record.endpoint_latency_ns.unwrap_or(0),
        "metrics": metrics,
        "request_sent_ns": record.request_sent_ns,
        "first_byte_ns": record.first_byte_ns,
        "benchmark_phase": record.benchmark_phase,
    })
}

fn full_record(record: &ServerMetricsRecord) -> Value {
    let metrics = record
        .metrics
        .iter()
        .map(|(name, family)| {
            (
                name.clone(),
                json!({
                    "type": metric_type_name(family.metric_type),
                    "description": family.description,
                    "samples": family.samples.iter().map(sample_value).collect::<Vec<_>>(),
                }),
            )
        })
        .collect::<Map<_, _>>();
    json!({
        "endpoint_url": record.endpoint_url,
        "timestamp_ns": record.timestamp_ns,
        "endpoint_latency_ns": record.endpoint_latency_ns,
        "metrics": metrics,
        "request_sent_ns": record.request_sent_ns,
        "first_byte_ns": record.first_byte_ns,
        "is_duplicate": record.is_duplicate,
        "benchmark_phase": record.benchmark_phase,
    })
}

fn sample_value(sample: &MetricSample) -> Value {
    match sample {
        MetricSample::Scalar { labels, value } => json!({
            "labels": (!labels.is_empty()).then_some(labels),
            "value": value,
        }),
        MetricSample::Histogram { labels, value } => json!({
            "labels": (!labels.is_empty()).then_some(labels),
            "buckets": value.buckets,
            "sum": value.sum,
            "count": value.count,
        }),
    }
}

fn metric_type_name(metric_type: PrometheusMetricType) -> &'static str {
    match metric_type {
        PrometheusMetricType::Counter => "counter",
        PrometheusMetricType::Gauge => "gauge",
        PrometheusMetricType::Histogram => "histogram",
        PrometheusMetricType::Summary => "summary",
        PrometheusMetricType::Unknown => "unknown",
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    use super::*;

    static NEXT_PATH: AtomicU64 = AtomicU64::new(0);

    fn artifact_path() -> PathBuf {
        std::env::temp_dir().join(format!(
            "aiperf-server-metrics-artifact-{}-{}.jsonl",
            std::process::id(),
            NEXT_PATH.fetch_add(1, Ordering::Relaxed)
        ))
    }

    fn record(timestamp_ns: i64, is_duplicate: bool) -> ServerMetricsRecord {
        ServerMetricsRecord {
            endpoint_url: "http://server/metrics".to_string(),
            timestamp_ns,
            endpoint_latency_ns: Some(10),
            request_sent_ns: Some(timestamp_ns - 10),
            first_byte_ns: Some(timestamp_ns),
            is_duplicate,
            benchmark_phase: Some(Phase::Profiling),
            metrics: BTreeMap::from([(
                "queue".to_string(),
                crate::server_metrics::MetricFamily {
                    metric_type: PrometheusMetricType::Gauge,
                    description: "Queued requests".to_string(),
                    samples: vec![MetricSample::Scalar {
                        labels: BTreeMap::new(),
                        value: 3.0,
                    }],
                },
            )]),
        }
    }

    #[test]
    fn unchanged_scrapes_never_reenter_the_slim_artifact_as_duplicate_rows() {
        let path = artifact_path();
        let records = vec![record(10, false), record(20, true), record(30, true)];

        JsonlServerMetricsArtifactSink
            .write(&path, &records, true)
            .unwrap();

        let contents = std::fs::read_to_string(&path).unwrap();
        let rows = contents.lines().collect::<Vec<_>>();
        assert_eq!(rows.len(), 1);
        let row: Value = serde_json::from_str(rows[0]).unwrap();
        assert_eq!(row["timestamp_ns"], 10);
        std::fs::remove_file(path).unwrap();
    }
}
