// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! GPU gauge accumulation, boundary-counter attribution, and efficiency joins.
//!
//! Per-GPU distributions and cross-GPU rollups use a NaN-aware `ddof=1` kernel.
//! Counter deltas use exact synchronous phase snapshots, a source-grounded reset
//! clamp, and MJ-to-J rollup.

use std::collections::BTreeMap;
use std::fmt::{Display, Formatter, Result as FmtResult};

use crate::metrics_core::{
    Accumulator, AccumulatorSummary, ExportContext, MetricTag, MetricValue, MetricsAccumulator,
    SidecarMetric, SidecarSeries, SidecarStats, boundary_counter_delta, linear_distribution,
};

use crate::gpu_telemetry::fields::{
    AMD_METRICS, DCGM_METRICS, GpuMetricKind, RuntimeGpuMetricSpec,
};
use crate::gpu_telemetry::model::{
    GpuBoundarySnapshot, GpuMetadata, GpuSeriesKey, GpuTelemetryRecord,
};
use crate::gpu_telemetry::source::GpuTelemetryError;

const NANOS_PER_SECOND: f64 = 1_000_000_000.0;
const MEGAJOULE_TO_JOULE: f64 = 1_000_000.0;

#[derive(Debug, Clone, PartialEq)]
struct GpuSample {
    timestamp_ns: i64,
    metrics: BTreeMap<String, f64>,
}

#[derive(Debug, Clone, PartialEq)]
struct GpuSeriesState {
    metadata: GpuMetadata,
    samples: Vec<GpuSample>,
}

/// Exact start/final snapshots for one phase.
#[derive(Debug, Clone, PartialEq)]
pub struct GpuPhaseBoundary {
    /// Counter values forced at phase start.
    pub start: GpuBoundarySnapshot,
    /// Counter values forced at phase end.
    pub end: GpuBoundarySnapshot,
}

impl GpuPhaseBoundary {
    /// Validates and builds a phase boundary pair.
    pub fn new(
        start: GpuBoundarySnapshot,
        end: GpuBoundarySnapshot,
    ) -> Result<Self, GpuTelemetryError> {
        if end.timestamp_ns < start.timestamp_ns {
            return Err(GpuTelemetryError::InvalidBoundary {
                start_ns: start.timestamp_ns,
                end_ns: end.timestamp_ns,
            });
        }
        Ok(Self { start, end })
    }

    /// Exact authoritative phase duration in seconds.
    pub fn duration_seconds(&self) -> Option<f64> {
        let duration_ns = self.end.timestamp_ns - self.start.timestamp_ns;
        (duration_ns > 0).then_some(duration_ns as f64 / NANOS_PER_SECOND)
    }
}

/// Incompatibility detected while merging worker-local GPU telemetry state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GpuMergeError {
    /// The same endpoint/UUID key carried different metadata.
    MetadataConflict(GpuSeriesKey),
    /// Workers carried different phase-boundary snapshots.
    BoundaryConflict,
}

/// Invalid runtime metric registration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GpuMetricRegistrationError {
    /// A custom field did not provide a stable name.
    EmptyName,
    /// A custom field collided with a built-in or earlier custom field.
    DuplicateName(String),
}

impl Display for GpuMetricRegistrationError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> FmtResult {
        match self {
            Self::EmptyName => formatter.write_str("GPU telemetry metric name cannot be empty"),
            Self::DuplicateName(name) => {
                write!(formatter, "duplicate GPU telemetry metric {name:?}")
            }
        }
    }
}

impl std::error::Error for GpuMetricRegistrationError {}

impl Display for GpuMergeError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> FmtResult {
        match self {
            Self::MetadataConflict(key) => write!(
                formatter,
                "conflicting GPU metadata for {} at {}",
                key.gpu_uuid, key.endpoint_url
            ),
            Self::BoundaryConflict => {
                formatter.write_str("conflicting GPU phase-boundary snapshots")
            }
        }
    }
}

impl std::error::Error for GpuMergeError {}

/// Fully computed GPU telemetry output for one authoritative phase.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct GpuTelemetrySummary {
    sidecar_metrics: BTreeMap<String, SidecarMetric>,
    injections: BTreeMap<MetricTag, MetricValue>,
    power_gpu_count: usize,
    energy_gpu_count: usize,
}

impl GpuTelemetrySummary {
    /// Returns per-GPU metrics keyed by normalized signal name.
    pub fn sidecar_metrics(&self) -> &BTreeMap<String, SidecarMetric> {
        &self.sidecar_metrics
    }

    /// Returns catalog scalar joins keyed by native metric identity.
    pub fn injections(&self) -> &BTreeMap<MetricTag, MetricValue> {
        &self.injections
    }

    /// Number of GPUs contributing to total power.
    pub fn power_gpu_count(&self) -> usize {
        self.power_gpu_count
    }

    /// Number of GPUs contributing to total energy.
    pub fn energy_gpu_count(&self) -> usize {
        self.energy_gpu_count
    }

    /// Delivers GPU efficiency scalars before the main metrics accumulator summarizes.
    pub fn inject_into(&self, accumulator: &mut MetricsAccumulator) {
        for (tag, value) in &self.injections {
            accumulator.inject_scalar(*tag, *value);
        }
    }

    /// Attaches scalar joins and per-GPU series to an already-produced summary.
    ///
    /// This path lets finalization read `total_output_tokens` from the primary
    /// summary, compute tokens/J, and then complete that same summary without a
    /// second request-metrics pass.
    pub fn attach_to(&self, summary: &mut AccumulatorSummary) {
        for (tag, value) in &self.injections {
            summary.insert(*tag, *value);
        }
        summary.extend_sidecar_metrics(
            self.sidecar_metrics
                .iter()
                .map(|(name, metric)| (name.clone(), metric.clone())),
        );
    }
}

/// Append-only, lock-free GPU telemetry accumulator.
#[derive(Debug, Clone, PartialEq)]
pub struct GpuTelemetryAccumulator {
    timestamps_ns: Vec<i64>,
    series: BTreeMap<GpuSeriesKey, GpuSeriesState>,
    phase_boundary: Option<GpuPhaseBoundary>,
    metric_specs: BTreeMap<String, RuntimeGpuMetricSpec>,
}

impl Default for GpuTelemetryAccumulator {
    fn default() -> Self {
        let metric_specs = DCGM_METRICS
            .iter()
            .chain(AMD_METRICS)
            .map(|spec| {
                let runtime = RuntimeGpuMetricSpec::from(spec);
                (runtime.name.clone(), runtime)
            })
            .collect();
        Self {
            timestamps_ns: Vec::new(),
            series: BTreeMap::new(),
            phase_boundary: None,
            metric_specs,
        }
    }
}

impl GpuTelemetryAccumulator {
    /// Builds an empty accumulator.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds Config-v2 or extension-owned signals without replacing built-ins.
    pub fn with_additional_metric_specs(
        mut self,
        specs: impl IntoIterator<Item = RuntimeGpuMetricSpec>,
    ) -> Result<Self, GpuMetricRegistrationError> {
        for spec in specs {
            if spec.name.trim().is_empty() {
                return Err(GpuMetricRegistrationError::EmptyName);
            }
            if self.metric_specs.contains_key(&spec.name) {
                return Err(GpuMetricRegistrationError::DuplicateName(spec.name));
            }
            self.metric_specs.insert(spec.name.clone(), spec);
        }
        Ok(self)
    }

    /// Number of ingested per-GPU records.
    pub fn len(&self) -> usize {
        self.timestamps_ns.len()
    }

    /// Whether no per-GPU record has been ingested.
    pub fn is_empty(&self) -> bool {
        self.timestamps_ns.is_empty()
    }

    /// Materializes the raw per-GPU scrape records in deterministic time order.
    ///
    /// Native-v2 reporting stays columnar through [`summarize_phase`](Self::summarize_phase).
    /// This colder clone path serves telemetry JSONL artifact consumers.
    pub fn records(&self) -> Vec<GpuTelemetryRecord> {
        let mut records = self
            .series
            .iter()
            .flat_map(|(key, state)| {
                state.samples.iter().map(move |sample| GpuTelemetryRecord {
                    timestamp_ns: sample.timestamp_ns,
                    endpoint_url: key.endpoint_url.clone(),
                    metadata: state.metadata.clone(),
                    metrics: sample.metrics.clone(),
                })
            })
            .collect::<Vec<_>>();
        records.sort_by(|left, right| {
            left.timestamp_ns
                .cmp(&right.timestamp_ns)
                .then_with(|| left.endpoint_url.cmp(&right.endpoint_url))
                .then_with(|| left.metadata.gpu_index.cmp(&right.metadata.gpu_index))
                .then_with(|| left.metadata.gpu_uuid.cmp(&right.metadata.gpu_uuid))
        });
        records
    }

    /// Appends one decoded per-GPU scrape record.
    pub fn ingest_record(&mut self, record: &GpuTelemetryRecord) {
        self.timestamps_ns.push(record.timestamp_ns);
        let key = record.series_key();
        let state = self.series.entry(key).or_insert_with(|| GpuSeriesState {
            metadata: record.metadata.clone(),
            samples: Vec::new(),
        });
        let sample = GpuSample {
            timestamp_ns: record.timestamp_ns,
            metrics: record
                .metrics
                .iter()
                .filter(|(_, value)| value.is_finite())
                .map(|(name, value)| (name.clone(), *value))
                .collect(),
        };
        let position = state
            .samples
            .partition_point(|existing| existing.timestamp_ns <= sample.timestamp_ns);
        state.samples.insert(position, sample);
    }

    /// Stores the exact snapshots used by [`Accumulator::export_results`].
    pub fn set_phase_boundary(&mut self, boundary: GpuPhaseBoundary) {
        self.phase_boundary = Some(boundary);
    }

    /// Computes per-GPU series and run-level efficiency joins for one phase.
    pub fn summarize_phase(
        &self,
        boundary: &GpuPhaseBoundary,
        total_output_tokens: Option<f64>,
        concurrency: Option<u64>,
    ) -> GpuTelemetrySummary {
        let mut summary = GpuTelemetrySummary::default();
        for spec in self.metric_specs.values() {
            let mut output_series = Vec::new();
            for (key, state) in &self.series {
                let stats = match spec.kind {
                    GpuMetricKind::Gauge => self.gauge_stats(key, state, spec, boundary),
                    GpuMetricKind::Counter => self.counter_stats(key, spec, boundary),
                };
                if let Some(stats) = stats {
                    output_series.push(SidecarSeries {
                        labels: Some(labels_for(&state.metadata)),
                        endpoint_url: Some(key.endpoint_url.clone()),
                        stats,
                        timeslices: Vec::new(),
                    });
                }
            }
            if !output_series.is_empty() {
                summary.sidecar_metrics.insert(
                    spec.name.to_string(),
                    SidecarMetric::new(Some(spec.unit), output_series),
                );
            }
        }

        let (total_power, power_gpu_count) = self.total_power(boundary);
        let (total_energy, energy_gpu_count) = self.total_energy(boundary);
        summary.power_gpu_count = power_gpu_count;
        summary.energy_gpu_count = energy_gpu_count;
        if power_gpu_count > 0 {
            summary.injections.insert(
                MetricTag::TotalGpuPower,
                MetricValue::from_f64(total_power, false),
            );
        }
        if energy_gpu_count > 0 {
            summary.injections.insert(
                MetricTag::TotalGpuEnergy,
                MetricValue::from_f64(total_energy, false),
            );
            if total_energy > 0.0 {
                if let Some(tokens) = total_output_tokens.filter(|value| value.is_finite()) {
                    summary.injections.insert(
                        MetricTag::OutputTokensPerJoule,
                        MetricValue::from_f64(tokens / total_energy, false),
                    );
                }
                if let Some(users) = concurrency.filter(|value| *value > 0) {
                    summary.injections.insert(
                        MetricTag::EnergyPerUser,
                        MetricValue::from_f64(total_energy / users as f64, false),
                    );
                }
            }
        }
        summary
    }

    fn gauge_stats(
        &self,
        _key: &GpuSeriesKey,
        state: &GpuSeriesState,
        spec: &RuntimeGpuMetricSpec,
        boundary: &GpuPhaseBoundary,
    ) -> Option<SidecarStats> {
        let values = state
            .samples
            .iter()
            .filter(|sample| {
                sample.timestamp_ns >= boundary.start.timestamp_ns
                    && sample.timestamp_ns <= boundary.end.timestamp_ns
            })
            .filter_map(|sample| sample.metrics.get(&spec.name).copied())
            .filter(|value| value.is_finite())
            .collect::<Vec<_>>();
        let sum = values.iter().sum();
        linear_distribution(&spec.name, values, sum, 1).map(SidecarStats::Gauge)
    }

    fn counter_stats(
        &self,
        key: &GpuSeriesKey,
        spec: &RuntimeGpuMetricSpec,
        boundary: &GpuPhaseBoundary,
    ) -> Option<SidecarStats> {
        let delta = boundary_counter_delta(
            boundary.start.counter(key, &spec.name),
            boundary.end.counter(key, &spec.name),
        )?;
        let rate = boundary
            .duration_seconds()
            .map(|seconds| MetricValue::from_f64(delta.delta / seconds, false));
        Some(SidecarStats::Counter {
            total: MetricValue::from_f64(delta.delta, false),
            rate,
        })
    }

    fn total_power(&self, boundary: &GpuPhaseBoundary) -> (f64, usize) {
        self.series
            .iter()
            .filter_map(|(key, state)| {
                ["nvidia_power_usage", "amd_power"]
                    .into_iter()
                    .find_map(|name| {
                        let spec = self.metric_specs.get(name)?;
                        match self.gauge_stats(key, state, spec, boundary)? {
                            SidecarStats::Gauge(stats) => stats.avg.as_f64(),
                            SidecarStats::Counter { .. } | SidecarStats::Histogram { .. } => None,
                        }
                    })
            })
            .fold((0.0, 0), |(sum, count), value| (sum + value, count + 1))
    }

    fn total_energy(&self, boundary: &GpuPhaseBoundary) -> (f64, usize) {
        self.series
            .keys()
            .filter_map(|key| {
                ["nvidia_energy_consumption", "amd_energy_consumption"]
                    .into_iter()
                    .find_map(|name| {
                        boundary_counter_delta(
                            boundary.start.counter(key, name),
                            boundary.end.counter(key, name),
                        )
                        .map(|delta| delta.delta * MEGAJOULE_TO_JOULE)
                    })
            })
            .fold((0.0, 0), |(sum, count), value| (sum + value, count + 1))
    }

    fn merge_state(&mut self, other: &Self) -> Result<(), GpuMergeError> {
        if self.phase_boundary.is_some()
            && other.phase_boundary.is_some()
            && self.phase_boundary != other.phase_boundary
        {
            return Err(GpuMergeError::BoundaryConflict);
        }
        self.phase_boundary = self
            .phase_boundary
            .clone()
            .or_else(|| other.phase_boundary.clone());
        self.timestamps_ns.extend_from_slice(&other.timestamps_ns);
        for (key, incoming) in &other.series {
            if let Some(existing) = self.series.get_mut(key) {
                if existing.metadata != incoming.metadata {
                    return Err(GpuMergeError::MetadataConflict(key.clone()));
                }
                existing.samples.extend_from_slice(&incoming.samples);
                existing.samples.sort_by_key(|sample| sample.timestamp_ns);
            } else {
                self.series.insert(key.clone(), incoming.clone());
            }
        }
        Ok(())
    }
}

impl Accumulator<GpuTelemetryRecord> for GpuTelemetryAccumulator {
    type Summary = GpuTelemetrySummary;
    type MergeError = GpuMergeError;

    fn process_record(&mut self, record: &GpuTelemetryRecord) {
        self.ingest_record(record);
    }

    fn query_time_range(&self, start_ns: i64, end_ns: i64) -> Vec<bool> {
        self.timestamps_ns
            .iter()
            .map(|timestamp| *timestamp >= start_ns && *timestamp < end_ns)
            .collect()
    }

    fn export_results(&self, context: &ExportContext) -> Self::Summary {
        let Some(boundary) = self.phase_boundary.as_ref() else {
            return GpuTelemetrySummary::default();
        };
        if context
            .start_ns
            .is_some_and(|start| start != boundary.start.timestamp_ns)
            || context
                .end_ns
                .is_some_and(|end| end != boundary.end.timestamp_ns)
        {
            return GpuTelemetrySummary::default();
        }
        self.summarize_phase(boundary, None, None)
    }

    fn merge(&mut self, other: &Self) -> Result<(), Self::MergeError> {
        self.merge_state(other)
    }
}

fn labels_for(metadata: &GpuMetadata) -> BTreeMap<String, String> {
    let mut labels = BTreeMap::from([
        ("gpu".to_string(), metadata.gpu_index.to_string()),
        ("gpu_uuid".to_string(), metadata.gpu_uuid.clone()),
        ("model_name".to_string(), metadata.gpu_model_name.clone()),
        ("platform".to_string(), metadata.platform.clone()),
    ]);
    for (name, value) in [
        ("pci_bus_id", metadata.pci_bus_id.as_ref()),
        ("device", metadata.device.as_ref()),
        ("hostname", metadata.hostname.as_ref()),
        ("namespace", metadata.namespace.as_ref()),
        ("pod", metadata.pod_name.as_ref()),
    ] {
        if let Some(value) = value {
            labels.insert(name.to_string(), value.clone());
        }
    }
    labels
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics_core::Unit;

    fn record(timestamp_ns: i64, gpu: i32, power: f64, energy_mj: f64) -> GpuTelemetryRecord {
        GpuTelemetryRecord {
            timestamp_ns,
            endpoint_url: "http://dcgm/metrics".to_string(),
            metadata: GpuMetadata {
                gpu_index: gpu,
                gpu_uuid: format!("GPU-{gpu}"),
                gpu_model_name: "H100".to_string(),
                pci_bus_id: None,
                device: None,
                hostname: Some("node".to_string()),
                namespace: None,
                pod_name: None,
                platform: crate::gpu_telemetry::model::NVIDIA_GPU_TELEMETRY_PLATFORM.to_string(),
            },
            metrics: BTreeMap::from([
                ("nvidia_power_usage".to_string(), power),
                ("nvidia_energy_consumption".to_string(), energy_mj),
            ]),
        }
    }

    fn snapshot(timestamp_ns: i64, records: &[GpuTelemetryRecord]) -> GpuBoundarySnapshot {
        let scrape = crate::gpu_telemetry::GpuScrape {
            timestamp_ns,
            endpoint_url: "http://dcgm/metrics".to_string(),
            records: records.to_vec(),
        };
        GpuBoundarySnapshot::from_scrape(&scrape)
    }

    #[test]
    fn exact_boundaries_drive_reset_clamped_energy_and_ddof_one_power() {
        let start_records = vec![record(10, 0, 100.0, 0.001), record(10, 1, 200.0, 0.010)];
        let middle_records = vec![record(15, 0, 120.0, 0.002), record(15, 1, 220.0, 0.005)];
        let end_records = vec![record(20, 0, 140.0, 0.003), record(20, 1, 240.0, 0.008)];
        let mut accumulator = GpuTelemetryAccumulator::new();
        for record in start_records
            .iter()
            .chain(&middle_records)
            .chain(&end_records)
        {
            accumulator.ingest_record(record);
        }
        let boundary =
            GpuPhaseBoundary::new(snapshot(10, &start_records), snapshot(20, &end_records))
                .unwrap();
        let summary = accumulator.summarize_phase(&boundary, Some(2_000.0), Some(4));

        // GPU 0 contributes 0.002 MJ = 2,000 J; GPU 1 reset-clamps to zero.
        assert_eq!(
            summary
                .injections()
                .get(&MetricTag::TotalGpuEnergy)
                .and_then(|value| value.as_f64()),
            Some(2_000.0)
        );
        assert_eq!(summary.energy_gpu_count(), 2);
        assert_eq!(
            summary
                .injections()
                .get(&MetricTag::TotalGpuPower)
                .and_then(|value| value.as_f64()),
            Some(340.0)
        );
        assert_eq!(
            summary
                .injections()
                .get(&MetricTag::OutputTokensPerJoule)
                .and_then(|value| value.as_f64()),
            Some(1.0)
        );
        assert_eq!(
            summary
                .injections()
                .get(&MetricTag::EnergyPerUser)
                .and_then(|value| value.as_f64()),
            Some(500.0)
        );

        let power = &summary.sidecar_metrics()["nvidia_power_usage"].series[0].stats;
        let SidecarStats::Gauge(power) = power else {
            panic!("expected gauge")
        };
        assert_eq!(power.avg.as_f64(), Some(120.0));
        assert!((power.std.unwrap() - 20.0).abs() < 1e-12);

        let mut primary = AccumulatorSummary::new();
        summary.attach_to(&mut primary);
        assert_eq!(
            primary.finite_value(MetricTag::TotalGpuEnergy),
            Some(2_000.0)
        );
        assert!(primary.sidecar_metrics().contains_key("nvidia_power_usage"));
    }

    #[test]
    fn query_contract_is_half_open_even_though_phase_gauges_include_end_boundary() {
        let mut accumulator = GpuTelemetryAccumulator::new();
        for timestamp in [10, 20, 30] {
            accumulator.ingest_record(&record(timestamp, 0, 1.0, 1.0));
        }
        assert_eq!(
            accumulator.query_time_range(10, 30),
            vec![true, true, false]
        );
    }

    #[test]
    fn jpg_utilization_is_registered_for_summary_output() {
        assert!(
            GpuTelemetryAccumulator::new()
                .metric_specs
                .contains_key("nvidia_jpg_utilization")
        );
    }

    fn custom_spec(name: &str, unit: Unit) -> RuntimeGpuMetricSpec {
        RuntimeGpuMetricSpec {
            name: name.to_string(),
            header: name.to_string(),
            unit,
            kind: GpuMetricKind::Gauge,
        }
    }

    /// Records carrying custom-CSV signal names only surface when their specs
    /// are registered: the summarizer iterates registered specs, so a scraped
    /// value whose name has no spec is silently dropped, consistent with the
    /// runner projecting `custom_metrics` (from the `--gpu-telemetry` CSV) into
    /// the accumulator so all default + custom fields report.
    #[test]
    fn registered_custom_specs_report_alongside_defaults_with_their_units() {
        let build = |timestamp_ns: i64, sm_clock: f64, mem_clock: f64, memory_temp: f64| {
            GpuTelemetryRecord {
                timestamp_ns,
                endpoint_url: "http://dcgm/metrics".to_string(),
                metadata: GpuMetadata {
                    gpu_index: 0,
                    gpu_uuid: "GPU-0".to_string(),
                    gpu_model_name: "H200".to_string(),
                    pci_bus_id: None,
                    device: None,
                    hostname: Some("node".to_string()),
                    namespace: None,
                    pod_name: None,
                    platform: crate::gpu_telemetry::model::NVIDIA_GPU_TELEMETRY_PLATFORM
                        .to_string(),
                },
                metrics: BTreeMap::from([
                    ("nvidia_power_usage".to_string(), 500.0),
                    ("sm_clock".to_string(), sm_clock),
                    ("mem_clock".to_string(), mem_clock),
                    ("memory_temp".to_string(), memory_temp),
                ]),
            }
        };

        let mut accumulator = GpuTelemetryAccumulator::new()
            .with_additional_metric_specs([
                custom_spec("sm_clock", Unit::Megahertz),
                custom_spec("mem_clock", Unit::Megahertz),
                custom_spec("memory_temp", Unit::Celsius),
            ])
            .unwrap();
        let start = build(10, 1_400.0, 2_600.0, 60.0);
        let end = build(20, 1_500.0, 2_700.0, 62.0);
        accumulator.ingest_record(&start);
        accumulator.ingest_record(&end);
        let boundary = GpuPhaseBoundary::new(snapshot(10, &[start]), snapshot(20, &[end])).unwrap();
        let summary = accumulator.summarize_phase(&boundary, None, None);

        for (name, unit) in [
            ("sm_clock", Unit::Megahertz),
            ("mem_clock", Unit::Megahertz),
            ("memory_temp", Unit::Celsius),
            ("nvidia_power_usage", Unit::Watt),
        ] {
            let metric = summary
                .sidecar_metrics()
                .get(name)
                .unwrap_or_else(|| panic!("missing custom metric {name}"));
            assert_eq!(metric.unit, Some(unit), "wrong unit for {name}");
        }
    }

    /// Registration fails closed on empty or built-in names rather than
    /// silently shadowing a built-in specification.
    #[test]
    fn duplicate_and_empty_custom_specs_are_rejected() {
        assert_eq!(
            GpuTelemetryAccumulator::new()
                .with_additional_metric_specs([custom_spec("nvidia_power_usage", Unit::Watt)])
                .unwrap_err(),
            GpuMetricRegistrationError::DuplicateName("nvidia_power_usage".to_string())
        );
        assert_eq!(
            GpuTelemetryAccumulator::new()
                .with_additional_metric_specs([custom_spec("  ", Unit::Megahertz)])
                .unwrap_err(),
            GpuMetricRegistrationError::EmptyName
        );
    }

    #[test]
    fn gauge_window_includes_boundary_scrapes_without_changing_counter_duration() {
        let mut accumulator = GpuTelemetryAccumulator::new();
        let opening = record(5, 0, 10.0, 0.0);
        let start = record(10, 0, 20.0, 1.0);
        let end = record(20, 0, 30.0, 3.0);
        let closing = record(25, 0, 40.0, 4.0);
        for record in [&opening, &start, &end, &closing] {
            accumulator.ingest_record(record);
        }
        let boundary = GpuPhaseBoundary::new(snapshot(10, &[start]), snapshot(20, &[end]))
            .unwrap()
            .with_gauge_window(5, 25);
        let summary = accumulator.summarize_phase(&boundary, None, None);

        let SidecarStats::Gauge(power) = &summary.sidecar_metrics()["nvidia_power_usage"].series[0].stats else {
            panic!("expected power gauge")
        };
        assert_eq!(power.avg.as_f64(), Some(25.0));
        let SidecarStats::Counter { rate, .. } = &summary.sidecar_metrics()
            ["nvidia_energy_consumption"].series[0].stats
        else {
            panic!("expected energy counter")
        };
        assert_eq!(rate.and_then(|value| value.as_f64()), Some(200_000_000.0));
    }

    #[test]
    fn invalid_boundary_is_rejected_and_missing_concurrency_omits_energy_per_user() {
        let records = vec![record(10, 0, 100.0, 1.0)];
        assert!(matches!(
            GpuPhaseBoundary::new(snapshot(20, &records), snapshot(10, &records)),
            Err(GpuTelemetryError::InvalidBoundary { .. })
        ));

        let mut accumulator = GpuTelemetryAccumulator::new();
        accumulator.ingest_record(&records[0]);
        let end = vec![record(20, 0, 100.0, 2.0)];
        accumulator.ingest_record(&end[0]);
        let boundary = GpuPhaseBoundary::new(snapshot(10, &records), snapshot(20, &end)).unwrap();
        let summary = accumulator.summarize_phase(&boundary, None, None);
        assert!(!summary.injections().contains_key(&MetricTag::EnergyPerUser));
    }
}
