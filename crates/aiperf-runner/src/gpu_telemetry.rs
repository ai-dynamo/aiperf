// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Profiling-phase GPU telemetry composition for the single-run process.
//!
//! The phase driver supplies the hard barriers. This module forces a DCGM
//! counter scrape before issuance, samples gauges on the injected Clock, forces
//! the closing counter scrape after all returns, and joins the resulting
//! efficiency values into the same native-v2 report as request metrics.

use std::cell::{Cell, RefCell};
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::rc::Rc;

use aiperf::phase_runtime::ScheduledPhaseSidecar;
use aiperf_clock::Clock;
use aiperf_gpu_telemetry::{
    DcgmTelemetrySource, GpuBoundarySnapshot, GpuMetricKind, GpuPhaseBoundary,
    GpuTelemetryAccumulator, GpuTelemetryCollector, GpuTelemetryRecord, GpuTelemetrySummary,
    PythonGpuTelemetryConfig, PythonGpuTelemetrySource, RuntimeGpuMetricSpec,
};
use aiperf_metrics::Unit;
use aiperf_transport::config::ClientConfig;
use aiperf_transport::transport::http_transport::HttpTransport;
use anyhow::{Context, Result, ensure};
use serde::Serialize;
use tokio::sync::Notify;
use tokio::task::JoinHandle;

use crate::protocol::{GpuTelemetrySourceSpec, GpuTelemetrySpec, GpuTelemetryUnitSpec};

/// Run-owned GPU telemetry state and its profiling-phase sidecar.
pub(crate) struct GpuTelemetryRun {
    sidecar: Rc<GpuTelemetrySidecar>,
}

impl GpuTelemetryRun {
    /// Builds all configured sources over one Clock-injected control transport.
    pub(crate) async fn new(spec: &GpuTelemetrySpec, clock: Rc<dyn Clock>) -> Result<Self> {
        ensure!(
            spec.collection_interval_ns > 0,
            "GPU telemetry collection_interval_ns must be positive"
        );
        ensure!(
            spec.request_timeout_ns > 0,
            "GPU telemetry request_timeout_ns must be positive"
        );
        ensure!(
            !spec.sources.is_empty(),
            "GPU telemetry requires at least one source"
        );

        let transport = Rc::new(HttpTransport::new(
            clock.clone(),
            ClientConfig {
                connect_timeout_ns: Some(spec.request_timeout_ns),
                request_timeout_ns: Some(spec.request_timeout_ns),
                ..ClientConfig::default()
            },
        ));
        let mut collectors = Vec::with_capacity(spec.sources.len());
        for source in &spec.sources {
            match source {
                GpuTelemetrySourceSpec::Dcgm { url } => {
                    ensure!(!url.trim().is_empty(), "DCGM telemetry URL cannot be empty");
                    let source = Rc::new(DcgmTelemetrySource::new(
                        clock.clone(),
                        transport.clone(),
                        url.clone(),
                    ));
                    collectors.push(Rc::new(GpuTelemetryCollector::new(source)));
                }
                GpuTelemetrySourceSpec::Python {
                    collector,
                    url,
                    metrics_file,
                    python_executable,
                    worker_module,
                } => {
                    let config = PythonGpuTelemetryConfig {
                        python_executable: python_executable.clone(),
                        worker_module: worker_module.clone(),
                        collector: collector.clone(),
                        url: url.clone(),
                        metrics_file: metrics_file.clone(),
                        request_timeout_seconds: spec.request_timeout_ns as f64 / 1_000_000_000.0,
                    };
                    match PythonGpuTelemetrySource::spawn(clock.clone(), config).await {
                        Ok(source) => {
                            collectors.push(Rc::new(GpuTelemetryCollector::new(Rc::new(source))))
                        }
                        Err(error) => {
                            eprintln!("GPU telemetry skipped unavailable Python source: {error}")
                        }
                    }
                }
            }
        }
        let accumulator = GpuTelemetryAccumulator::new().with_additional_metric_specs(
            spec.custom_metrics
                .iter()
                .map(|metric| RuntimeGpuMetricSpec {
                    name: metric.name.clone(),
                    header: metric.header.clone(),
                    unit: native_unit(metric.unit),
                    kind: GpuMetricKind::Gauge,
                }),
        )?;
        Ok(Self {
            sidecar: Rc::new(GpuTelemetrySidecar::new(
                clock,
                spec.collection_interval_ns,
                collectors,
                accumulator,
            )),
        })
    }

    /// Returns the object-safe phase lifecycle adapter.
    pub(crate) fn sidecar(&self) -> Rc<dyn ScheduledPhaseSidecar> {
        self.sidecar.clone()
    }

    /// Computes native scalar joins and labeled per-GPU series.
    pub(crate) fn summarize(
        &self,
        total_output_tokens: Option<f64>,
        concurrency: Option<u64>,
    ) -> GpuTelemetrySummary {
        self.sidecar.summarize(total_output_tokens, concurrency)
    }

    /// Writes every retained scrape record in the established Python JSONL shape.
    pub(crate) fn write_records_jsonl(&self, path: &Path) -> Result<()> {
        self.sidecar.write_records_jsonl(path)
    }
}

struct GpuTelemetrySidecar {
    state: Rc<GpuTelemetryState>,
}

struct GpuTelemetryState {
    clock: Rc<dyn Clock>,
    collection_interval_ns: i64,
    candidates: Vec<Rc<GpuTelemetryCollector>>,
    active: RefCell<Vec<Rc<GpuTelemetryCollector>>>,
    accumulator: Rc<RefCell<GpuTelemetryAccumulator>>,
    start_snapshots: RefCell<Vec<GpuBoundarySnapshot>>,
    boundary: RefCell<Option<GpuPhaseBoundary>>,
    stop: Rc<Notify>,
    task: RefCell<Option<JoinHandle<()>>>,
    started: Cell<bool>,
    finished: Cell<bool>,
}

impl GpuTelemetrySidecar {
    fn new(
        clock: Rc<dyn Clock>,
        collection_interval_ns: i64,
        candidates: Vec<Rc<GpuTelemetryCollector>>,
        accumulator: GpuTelemetryAccumulator,
    ) -> Self {
        Self {
            state: Rc::new(GpuTelemetryState {
                clock,
                collection_interval_ns,
                candidates,
                active: RefCell::new(Vec::new()),
                accumulator: Rc::new(RefCell::new(accumulator)),
                start_snapshots: RefCell::new(Vec::new()),
                boundary: RefCell::new(None),
                stop: Rc::new(Notify::new()),
                task: RefCell::new(None),
                started: Cell::new(false),
                finished: Cell::new(false),
            }),
        }
    }

    fn summarize(
        &self,
        total_output_tokens: Option<f64>,
        concurrency: Option<u64>,
    ) -> GpuTelemetrySummary {
        let boundary = self.state.boundary.borrow().clone();
        boundary.map_or_else(GpuTelemetrySummary::default, |boundary| {
            self.state.accumulator.borrow().summarize_phase(
                &boundary,
                total_output_tokens,
                concurrency,
            )
        })
    }

    fn write_records_jsonl(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).with_context(|| {
                format!(
                    "creating GPU telemetry export directory {}",
                    parent.display()
                )
            })?;
        }
        let file = File::create(path)
            .with_context(|| format!("creating GPU telemetry export {}", path.display()))?;
        let mut writer = BufWriter::new(file);
        for record in self.state.accumulator.borrow().records() {
            serde_json::to_writer(&mut writer, &TelemetryRow::from(&record))
                .with_context(|| format!("serializing GPU telemetry export {}", path.display()))?;
            writer
                .write_all(b"\n")
                .with_context(|| format!("writing GPU telemetry export {}", path.display()))?;
        }
        writer
            .flush()
            .with_context(|| format!("flushing GPU telemetry export {}", path.display()))
    }
}

impl ScheduledPhaseSidecar for GpuTelemetrySidecar {
    fn start(&self) -> aiperf_timing::LocalPhaseFuture<Result<()>> {
        let state = self.state.clone();
        Box::pin(async move { state.start().await })
    }

    fn finish(&self) -> aiperf_timing::LocalPhaseFuture<Result<()>> {
        let state = self.state.clone();
        Box::pin(async move { state.finish().await })
    }
}

impl GpuTelemetryState {
    async fn start(self: &Rc<Self>) -> Result<()> {
        if self.started.replace(true) {
            return Ok(());
        }

        let mut active = Vec::new();
        let mut snapshots = Vec::new();
        for collector in &self.candidates {
            match collector.collect_boundary().await {
                Ok((scrape, snapshot)) => {
                    GpuTelemetryCollector::ingest_scrape(
                        &scrape,
                        &mut self.accumulator.borrow_mut(),
                    );
                    snapshots.push(snapshot);
                    active.push(collector.clone());
                }
                Err(error) => eprintln!(
                    "GPU telemetry skipped unreachable source {}: {error}",
                    collector.endpoint_url()
                ),
            }
        }
        *self.active.borrow_mut() = active;
        *self.start_snapshots.borrow_mut() = snapshots;
        if self.active.borrow().is_empty() {
            return Ok(());
        }

        let state = self.clone();
        *self.task.borrow_mut() = Some(tokio::task::spawn_local(async move {
            state.collect_continuously().await;
        }));
        Ok(())
    }

    async fn collect_continuously(self: Rc<Self>) {
        loop {
            let sleep = self.clock.clone().sleep(self.collection_interval_ns);
            let stopped = self.stop.notified();
            tokio::pin!(sleep);
            tokio::pin!(stopped);
            tokio::select! {
                biased;
                _ = &mut stopped => return,
                _ = &mut sleep => {}
            }
            let collectors = self.active.borrow().clone();
            for collector in collectors {
                match collector.collect_continuous().await {
                    Ok(Some(scrape)) => GpuTelemetryCollector::ingest_scrape(
                        &scrape,
                        &mut self.accumulator.borrow_mut(),
                    ),
                    Ok(None) => {}
                    Err(error) => {
                        eprintln!(
                            "GPU telemetry cadence scrape failed for {}: {error}",
                            collector.endpoint_url()
                        );
                    }
                }
            }
        }
    }

    async fn finish(self: &Rc<Self>) -> Result<()> {
        if self.finished.replace(true) {
            return Ok(());
        }
        self.stop.notify_one();
        let task = self.task.borrow_mut().take();
        if let Some(task) = task
            && let Err(error) = task.await
        {
            eprintln!("GPU telemetry cadence task failed: {error}");
        }

        let collectors = self.active.borrow().clone();
        let mut end_snapshots = Vec::new();
        for collector in collectors {
            match collector.collect_boundary().await {
                Ok((scrape, snapshot)) => {
                    GpuTelemetryCollector::ingest_scrape(
                        &scrape,
                        &mut self.accumulator.borrow_mut(),
                    );
                    end_snapshots.push(snapshot);
                }
                Err(error) => eprintln!(
                    "GPU telemetry final scrape failed for {}: {error}",
                    collector.endpoint_url()
                ),
            }
        }

        for collector in &self.candidates {
            if let Err(error) = collector.shutdown().await {
                eprintln!(
                    "GPU telemetry source shutdown failed for {}: {error}",
                    collector.endpoint_url()
                );
            }
        }

        let start_snapshots = self.start_snapshots.borrow();
        if let (Some(start), Some(end)) = (
            combine_snapshots(&start_snapshots, BoundarySide::Start),
            combine_snapshots(&end_snapshots, BoundarySide::End),
        ) {
            let boundary = GpuPhaseBoundary::new(start, end)?;
            self.accumulator
                .borrow_mut()
                .set_phase_boundary(boundary.clone());
            *self.boundary.borrow_mut() = Some(boundary);
        }
        Ok(())
    }
}

fn native_unit(unit: GpuTelemetryUnitSpec) -> Unit {
    match unit {
        GpuTelemetryUnitSpec::Count => Unit::Count,
        GpuTelemetryUnitSpec::Kilobyte => Unit::Kilobyte,
        GpuTelemetryUnitSpec::Megabyte => Unit::Megabyte,
        GpuTelemetryUnitSpec::Gigabyte => Unit::Gigabyte,
        GpuTelemetryUnitSpec::Microsecond => Unit::Microsecond,
        GpuTelemetryUnitSpec::Millisecond => Unit::Millisecond,
        GpuTelemetryUnitSpec::Second => Unit::Second,
        GpuTelemetryUnitSpec::Percent => Unit::Percent,
        GpuTelemetryUnitSpec::Watt => Unit::Watt,
        GpuTelemetryUnitSpec::Joule => Unit::Joule,
        GpuTelemetryUnitSpec::Megajoule => Unit::Megajoule,
        GpuTelemetryUnitSpec::Megahertz => Unit::Megahertz,
        GpuTelemetryUnitSpec::Gigahertz => Unit::Gigahertz,
        GpuTelemetryUnitSpec::Celsius => Unit::Celsius,
    }
}

#[derive(Clone, Copy)]
enum BoundarySide {
    Start,
    End,
}

fn combine_snapshots(
    snapshots: &[GpuBoundarySnapshot],
    side: BoundarySide,
) -> Option<GpuBoundarySnapshot> {
    let timestamp_ns = match side {
        BoundarySide::Start => snapshots.iter().map(|item| item.timestamp_ns).min()?,
        BoundarySide::End => snapshots.iter().map(|item| item.timestamp_ns).max()?,
    };
    let mut counters = BTreeMap::new();
    for snapshot in snapshots {
        counters.extend(snapshot.counters.clone());
    }
    Some(GpuBoundarySnapshot {
        timestamp_ns,
        counters,
    })
}

#[derive(Serialize)]
struct TelemetryRow<'a> {
    gpu_index: i32,
    gpu_uuid: &'a str,
    gpu_model_name: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pci_bus_id: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    device: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    hostname: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    namespace: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pod_name: Option<&'a str>,
    timestamp_ns: i64,
    dcgm_url: &'a str,
    telemetry_data: &'a BTreeMap<String, f64>,
}

impl<'a> From<&'a GpuTelemetryRecord> for TelemetryRow<'a> {
    fn from(record: &'a GpuTelemetryRecord) -> Self {
        Self {
            gpu_index: record.metadata.gpu_index,
            gpu_uuid: &record.metadata.gpu_uuid,
            gpu_model_name: &record.metadata.gpu_model_name,
            pci_bus_id: record.metadata.pci_bus_id.as_deref(),
            device: record.metadata.device.as_deref(),
            hostname: record.metadata.hostname.as_deref(),
            namespace: record.metadata.namespace.as_deref(),
            pod_name: record.metadata.pod_name.as_deref(),
            timestamp_ns: record.timestamp_ns,
            dcgm_url: &record.endpoint_url,
            telemetry_data: &record.metrics,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snapshot_merge_keeps_all_endpoints_and_barrier_extrema() {
        let first = GpuBoundarySnapshot {
            timestamp_ns: 10,
            counters: BTreeMap::new(),
        };
        let second = GpuBoundarySnapshot {
            timestamp_ns: 20,
            counters: BTreeMap::new(),
        };
        assert_eq!(
            combine_snapshots(&[first.clone(), second.clone()], BoundarySide::Start)
                .unwrap()
                .timestamp_ns,
            10
        );
        assert_eq!(
            combine_snapshots(&[first, second], BoundarySide::End)
                .unwrap()
                .timestamp_ns,
            20
        );
    }
}
