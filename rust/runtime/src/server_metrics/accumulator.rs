// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Phase-boundary server counter attribution and continuous gauge aggregation.
//!
//! Counters and histogram totals use exact phase start/end snapshots, while
//! continuous records supply gauge distributions,
//! timeslices, and histogram bucket-mean learning.

use std::collections::{BTreeMap, HashMap};
use std::fmt::{Display, Formatter, Result as FmtResult};

use crate::metrics_core::{
    Accumulator, AccumulatorSummary, ExportContext, MetricValue, Phase, SidecarMetric,
    SidecarSeries, SidecarStats, SidecarTimeslice, Unit, linear_distribution,
};

use crate::server_metrics::atlas::{ServerMetricAtlas, ServerMetricView, VllmSglangMetricAtlas};
use crate::server_metrics::histogram::{
    HistogramSnapshot, accumulate_bucket_statistics, compute_estimated_percentiles,
};
use crate::server_metrics::model::{
    HistogramValue, MetricSample, PrometheusMetricType, ServerMetricsRecord,
};
use crate::server_metrics::units::infer_unit;

const NANOS_PER_SECOND: f64 = 1_000_000_000.0;
type ScalarLabelKey = Vec<(String, String)>;
type LatestGaugeKey = (String, ScalarLabelKey);
type LatestGaugeMap = BTreeMap<LatestGaugeKey, (i64, f64)>;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct SeriesKey {
    name: String,
    endpoint_url: String,
    labels: Vec<(String, String)>,
}

#[derive(Debug, Clone, PartialEq)]
enum SeriesValue {
    Scalar(f64),
    Histogram(HistogramValue),
}

#[derive(Debug, Clone, PartialEq)]
struct SeriesObservation {
    timestamp_ns: i64,
    value: SeriesValue,
}

#[derive(Debug, Clone, PartialEq)]
struct SeriesState {
    metric_type: PrometheusMetricType,
    description: String,
    observations: Vec<SeriesObservation>,
}

/// Forced start/end snapshots for one authoritative phase.
#[derive(Debug, Clone, PartialEq)]
pub struct ServerMetricsPhaseBoundary {
    /// Phase identity.
    pub phase: Phase,
    /// Earliest successful start-snapshot timestamp.
    pub start_ns: i64,
    /// Latest successful final-snapshot timestamp.
    pub end_ns: i64,
    /// Start snapshots keyed by credential-free endpoint.
    pub start_records: BTreeMap<String, ServerMetricsRecord>,
    /// End snapshots keyed by credential-free endpoint.
    pub end_records: BTreeMap<String, ServerMetricsRecord>,
}

impl ServerMetricsPhaseBoundary {
    /// Exact authoritative phase duration in seconds.
    pub fn duration_seconds(&self) -> Option<f64> {
        let duration_ns = self.end_ns - self.start_ns;
        (duration_ns > 0).then_some(duration_ns as f64 / NANOS_PER_SECOND)
    }
}

/// Fully computed server telemetry for one phase.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ServerMetricsSummary {
    sidecar_metrics: BTreeMap<String, SidecarMetric>,
    endpoints_successful: Vec<String>,
    descriptions: BTreeMap<String, String>,
    metric_types: BTreeMap<String, PrometheusMetricType>,
    boundary: Option<ServerMetricsPhaseBoundary>,
}

impl ServerMetricsSummary {
    /// Native side-channel metrics keyed by Prometheus family name.
    pub fn sidecar_metrics(&self) -> &BTreeMap<String, SidecarMetric> {
        &self.sidecar_metrics
    }

    /// Endpoints that contributed a successful boundary pair.
    pub fn endpoints_successful(&self) -> &[String] {
        &self.endpoints_successful
    }

    /// Prometheus HELP text keyed by metric family name.
    pub fn descriptions(&self) -> &BTreeMap<String, String> {
        &self.descriptions
    }

    /// Prometheus semantic type keyed by normalized metric family name.
    pub fn metric_types(&self) -> &BTreeMap<String, PrometheusMetricType> {
        &self.metric_types
    }

    /// Exact boundary used for this summary.
    pub fn boundary(&self) -> Option<&ServerMetricsPhaseBoundary> {
        self.boundary.as_ref()
    }

    /// Attach every server metric to an already computed request summary.
    pub fn attach_to(&self, summary: &mut AccumulatorSummary) {
        summary.extend_sidecar_metrics(
            self.sidecar_metrics
                .iter()
                .map(|(name, metric)| (name.clone(), metric.clone())),
        );
    }
}

/// Append-only native server-metrics accumulator.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ServerMetricsAccumulator {
    records: Vec<ServerMetricsRecord>,
    boundaries: HashMap<Phase, ServerMetricsPhaseBoundary>,
}

/// Incompatibility detected while merging independently collected server telemetry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ServerMetricsMergeError {
    /// Workers captured different snapshots for the same phase boundary.
    BoundaryConflict(Phase),
}

impl Display for ServerMetricsMergeError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> FmtResult {
        match self {
            Self::BoundaryConflict(phase) => {
                write!(formatter, "conflicting {phase:?} server-metrics boundaries")
            }
        }
    }
}

impl std::error::Error for ServerMetricsMergeError {}

impl ServerMetricsAccumulator {
    /// Build an empty accumulator.
    pub fn new() -> Self {
        Self::default()
    }

    /// Retain one complete parsed scrape.
    pub fn ingest_record(&mut self, record: ServerMetricsRecord) {
        // Insert in sorted position rather than re-sorting the whole Vec on
        // every scrape: keeps `records` ordered by (timestamp, endpoint) at
        // O(n) per insert instead of O(n log n), avoiding O(n^2 log n) growth
        // over a run. `partition_point` with `<=` places new records after any
        // existing equal-keyed record, preserving stable insertion order.
        let index = self.records.partition_point(|existing| {
            existing
                .timestamp_ns
                .cmp(&record.timestamp_ns)
                .then_with(|| existing.endpoint_url.cmp(&record.endpoint_url))
                .is_le()
        });
        self.records.insert(index, record);
    }

    /// Every retained scrape in deterministic timestamp/endpoint order.
    pub fn records(&self) -> &[ServerMetricsRecord] {
        &self.records
    }

    /// Install the exact start/end snapshots for a phase.
    pub fn set_phase_boundary(&mut self, boundary: ServerMetricsPhaseBoundary) {
        self.boundaries.insert(boundary.phase, boundary);
    }

    /// Summarize gauges, counters, and histograms for one phase.
    pub fn summarize_phase(
        &self,
        phase: Phase,
        slice_duration_ns: Option<i64>,
    ) -> ServerMetricsSummary {
        self.summarize_phase_with_atlas(phase, slice_duration_ns, &VllmSglangMetricAtlas)
    }

    /// Summarize one phase with an injected backend metric atlas.
    pub fn summarize_phase_with_atlas(
        &self,
        phase: Phase,
        slice_duration_ns: Option<i64>,
        atlas: &dyn ServerMetricAtlas,
    ) -> ServerMetricsSummary {
        let Some(boundary) = self.boundaries.get(&phase) else {
            return ServerMetricsSummary::default();
        };
        let mut states = self.phase_series(phase, boundary);
        let start_values = flatten_records(&boundary.start_records);
        let end_values = flatten_records(&boundary.end_records);
        let duration_seconds = boundary.duration_seconds();
        let mut metrics = BTreeMap::<String, (Option<Unit>, Vec<SidecarSeries>)>::new();
        let mut descriptions = BTreeMap::new();
        let mut metric_types = BTreeMap::new();

        for (key, state) in &mut states {
            state
                .observations
                .sort_by_key(|observation| observation.timestamp_ns);
            state.observations.dedup_by(|left, right| {
                left.timestamp_ns == right.timestamp_ns && left.value == right.value
            });
            let stats = match state.metric_type {
                PrometheusMetricType::Gauge | PrometheusMetricType::Unknown => {
                    gauge_series(key, state, boundary, slice_duration_ns)
                }
                PrometheusMetricType::Counter => counter_series(
                    key,
                    state,
                    boundary,
                    start_values.get(key),
                    end_values.get(key),
                    duration_seconds,
                    slice_duration_ns,
                ),
                PrometheusMetricType::Histogram => histogram_series(
                    key,
                    state,
                    boundary,
                    start_values.get(key),
                    end_values.get(key),
                    duration_seconds,
                    slice_duration_ns,
                ),
                PrometheusMetricType::Summary => None,
            };
            let Some(series) = stats else {
                continue;
            };
            descriptions
                .entry(key.name.clone())
                .or_insert_with(|| state.description.clone());
            metric_types
                .entry(key.name.clone())
                .or_insert(state.metric_type);
            let unit = infer_unit(&key.name, Some(&state.description));
            metrics
                .entry(key.name.clone())
                .and_modify(|(_, series_list)| series_list.push(series.clone()))
                .or_insert_with(|| (unit, vec![series]));
        }

        let mut sidecar_metrics = metrics
            .into_iter()
            .map(|(name, (unit, mut series))| {
                series.sort_by(|left, right| {
                    left.endpoint_url
                        .cmp(&right.endpoint_url)
                        .then_with(|| left.labels.cmp(&right.labels))
                });
                (name, SidecarMetric::new(unit, series))
            })
            .collect::<BTreeMap<_, _>>();
        // An endpoint is "successful" when it produced at least one valid boundary
        // scrape record (start OR end). A fresh inference server exposes no useful
        // series before it receives traffic — e.g. the mock's `/metrics` returns
        // only the (parser-skipped) `_uptime` gauge at phase-start — so requiring a
        // record at BOTH boundaries would wrongly drop every endpoint whose metric
        // families are created lazily on first request. Counter/histogram deltas
        // still require both boundaries and degrade gracefully (the atlas derivation
        // and delta helpers return nothing when a start record is absent).
        let endpoints_successful: Vec<String> = boundary
            .start_records
            .keys()
            .chain(boundary.end_records.keys())
            .cloned()
            .collect::<std::collections::BTreeSet<String>>()
            .into_iter()
            .collect();
        // Derive atlas metrics per endpoint because every series must reference
        // an endpoint in the collection metadata; a cross-endpoint aggregate
        // has no representable owner.
        for endpoint in &endpoints_successful {
            let atlas_view = PhaseMetricView {
                phase,
                records: &self.records,
                boundary,
                endpoint,
            };
            for (name, metric) in atlas.derive(&atlas_view) {
                let Some(stats) = linear_distribution(&name, vec![metric.value], metric.value, 1)
                else {
                    continue;
                };
                let series = SidecarSeries {
                    labels: None,
                    endpoint_url: Some(endpoint.clone()),
                    stats: SidecarStats::Gauge(stats),
                    timeslices: Vec::new(),
                };
                sidecar_metrics
                    .entry(name.clone())
                    .and_modify(|existing| existing.series.push(series.clone()))
                    .or_insert_with(|| SidecarMetric::new(Some(metric.unit), vec![series]));
                descriptions.insert(name.clone(), metric.description.to_string());
                metric_types.insert(name, PrometheusMetricType::Gauge);
            }
        }
        ServerMetricsSummary {
            sidecar_metrics,
            endpoints_successful,
            descriptions,
            metric_types,
            boundary: Some(boundary.clone()),
        }
    }

    /// Collection/update metadata for every endpoint with at least one record.
    pub fn endpoint_info(&self) -> BTreeMap<String, ServerMetricsEndpointInfo> {
        let mut records = BTreeMap::<String, Vec<&ServerMetricsRecord>>::new();
        for record in &self.records {
            records
                .entry(record.endpoint_url.clone())
                .or_default()
                .push(record);
        }
        records
            .into_iter()
            .map(|(endpoint, mut records)| {
                records.sort_by_key(|record| record.timestamp_ns);
                let first_fetch_ns = records.first().map_or(0, |record| record.timestamp_ns);
                let last_fetch_ns = records.last().map_or(0, |record| record.timestamp_ns);
                let latencies = records
                    .iter()
                    .filter_map(|record| record.endpoint_latency_ns)
                    .collect::<Vec<_>>();
                let avg_fetch_latency_ms = if latencies.is_empty() {
                    0.0
                } else {
                    latencies.iter().map(|value| *value as f64).sum::<f64>()
                        / latencies.len() as f64
                        / 1_000_000.0
                };
                let updates = records
                    .iter()
                    .filter(|record| !record.is_duplicate)
                    .map(|record| record.timestamp_ns)
                    .collect::<Vec<_>>();
                let first_update_ns = updates.first().copied().unwrap_or(0);
                let last_update_ns = updates.last().copied().unwrap_or(0);
                let duration_seconds = if updates.is_empty() {
                    0.0
                } else {
                    last_update_ns.saturating_sub(first_update_ns) as f64 / NANOS_PER_SECOND
                };
                let avg_update_interval_ms = if updates.len() > 1 {
                    duration_seconds * 1_000.0 / (updates.len() - 1) as f64
                } else {
                    0.0
                };
                let median_update_interval_ms = median_interval_ms(&updates);
                let info = ServerMetricsEndpointInfo {
                    total_fetches: records.len(),
                    first_fetch_ns,
                    last_fetch_ns,
                    avg_fetch_latency_ms,
                    unique_updates: updates.len(),
                    first_update_ns,
                    last_update_ns,
                    duration_seconds,
                    avg_update_interval_ms,
                    median_update_interval_ms,
                };
                (endpoint, info)
            })
            .collect()
    }

    fn phase_series(
        &self,
        phase: Phase,
        boundary: &ServerMetricsPhaseBoundary,
    ) -> BTreeMap<SeriesKey, SeriesState> {
        let mut states = BTreeMap::new();
        for record in self
            .records
            .iter()
            .filter(|record| record.benchmark_phase == Some(phase))
        {
            ingest_series(record, &mut states);
        }
        for record in boundary.start_records.values() {
            ingest_series(record, &mut states);
        }
        for record in boundary.end_records.values() {
            ingest_series(record, &mut states);
        }
        states
    }

    fn phase_for_context(&self, context: &ExportContext) -> Option<Phase> {
        let phase = context.phase.or_else(|| {
            [Phase::Profiling, Phase::Warmup]
                .into_iter()
                .find(|phase| self.boundaries.contains_key(phase))
        })?;
        let boundary = self.boundaries.get(&phase)?;
        if context
            .start_ns
            .is_some_and(|start_ns| start_ns != boundary.start_ns)
            || context
                .end_ns
                .is_some_and(|end_ns| end_ns != boundary.end_ns)
        {
            return None;
        }
        Some(phase)
    }
}

impl Accumulator<ServerMetricsRecord> for ServerMetricsAccumulator {
    type Summary = ServerMetricsSummary;
    type MergeError = ServerMetricsMergeError;

    fn process_record(&mut self, record: &ServerMetricsRecord) {
        self.ingest_record(record.clone());
    }

    fn query_time_range(&self, start_ns: i64, end_ns: i64) -> Vec<bool> {
        self.records
            .iter()
            .map(|record| record.timestamp_ns >= start_ns && record.timestamp_ns < end_ns)
            .collect()
    }

    fn export_results(&self, context: &ExportContext) -> Self::Summary {
        self.phase_for_context(context)
            .map_or_else(ServerMetricsSummary::default, |phase| {
                self.summarize_phase(phase, None)
            })
    }

    fn merge(&mut self, other: &Self) -> Result<(), Self::MergeError> {
        for (phase, incoming) in &other.boundaries {
            if self
                .boundaries
                .get(phase)
                .is_some_and(|existing| existing != incoming)
            {
                return Err(ServerMetricsMergeError::BoundaryConflict(*phase));
            }
        }
        for (phase, boundary) in &other.boundaries {
            self.boundaries
                .entry(*phase)
                .or_insert_with(|| boundary.clone());
        }
        self.records.extend(other.records.iter().cloned());
        self.records.sort_by(|left, right| {
            left.timestamp_ns
                .cmp(&right.timestamp_ns)
                .then_with(|| left.endpoint_url.cmp(&right.endpoint_url))
        });
        Ok(())
    }
}

struct PhaseMetricView<'a> {
    phase: Phase,
    records: &'a [ServerMetricsRecord],
    boundary: &'a ServerMetricsPhaseBoundary,
    /// Credential-free endpoint this view is scoped to; derived metrics are
    /// attributed to exactly one scraped endpoint so their series carry a
    /// concrete `endpoint_url`.
    endpoint: &'a str,
}

impl PhaseMetricView<'_> {
    fn latest_gauges_by_endpoint(&self, metric_name: &str) -> BTreeMap<String, f64> {
        let mut latest = LatestGaugeMap::new();
        if let Some(record) = self.boundary.start_records.get(self.endpoint) {
            retain_latest_gauges(record, metric_name, self.boundary.start_ns, &mut latest);
        }
        for record in self.records.iter().filter(|record| {
            record.endpoint_url == self.endpoint
                && record.benchmark_phase == Some(self.phase)
                && record.timestamp_ns >= self.boundary.start_ns
                && record.timestamp_ns <= self.boundary.end_ns
        }) {
            retain_latest_gauges(record, metric_name, record.timestamp_ns, &mut latest);
        }
        if let Some(record) = self.boundary.end_records.get(self.endpoint) {
            retain_latest_gauges(record, metric_name, self.boundary.end_ns, &mut latest);
        }

        let mut endpoints = BTreeMap::<String, f64>::new();
        for ((endpoint, _), (_, value)) in latest {
            endpoints
                .entry(endpoint)
                .and_modify(|current| *current = current.max(value))
                .or_insert(value);
        }
        endpoints
    }
}

impl ServerMetricView for PhaseMetricView<'_> {
    fn counter_delta(&self, metric_name: &str) -> Option<f64> {
        let start_record = self.boundary.start_records.get(self.endpoint)?;
        let end_record = self.boundary.end_records.get(self.endpoint)?;
        let mut found = false;
        let mut total = 0.0;
        let start = typed_scalar_values(start_record, metric_name, PrometheusMetricType::Counter);
        let end = typed_scalar_values(end_record, metric_name, PrometheusMetricType::Counter);
        for (labels, start_value) in start {
            let Some(end_value) = end.get(&labels) else {
                continue;
            };
            total += (*end_value - start_value).max(0.0);
            found = true;
        }
        found.then_some(total)
    }

    fn counter_rate(&self, metric_name: &str) -> Option<f64> {
        let duration_seconds = self.boundary.duration_seconds()?;
        self.counter_delta(metric_name)
            .map(|delta| delta / duration_seconds)
    }

    fn gauge_latest_max(&self, metric_name: &str) -> Option<f64> {
        self.latest_gauges_by_endpoint(metric_name)
            .into_values()
            .max_by(f64::total_cmp)
    }

    fn max_endpoint_gauge_ratio(
        &self,
        numerator_name: &str,
        denominator_name: &str,
    ) -> Option<f64> {
        let numerators = self.latest_gauges_by_endpoint(numerator_name);
        let denominators = self.latest_gauges_by_endpoint(denominator_name);
        numerators
            .into_iter()
            .filter_map(|(endpoint, numerator)| {
                let denominator = *denominators.get(&endpoint)?;
                (denominator > 0.0).then_some(numerator / denominator)
            })
            .max_by(f64::total_cmp)
    }
}

fn typed_scalar_values(
    record: &ServerMetricsRecord,
    metric_name: &str,
    metric_type: PrometheusMetricType,
) -> BTreeMap<ScalarLabelKey, f64> {
    let Some(family) = record
        .metrics
        .get(metric_name)
        .filter(|family| family.metric_type == metric_type)
    else {
        return BTreeMap::new();
    };
    family
        .samples
        .iter()
        .filter_map(|sample| match sample {
            MetricSample::Scalar { labels, value } => Some((
                labels
                    .iter()
                    .map(|(name, value)| (name.clone(), value.clone()))
                    .collect(),
                *value,
            )),
            MetricSample::Histogram { .. } => None,
        })
        .collect()
}

fn retain_latest_gauges(
    record: &ServerMetricsRecord,
    metric_name: &str,
    timestamp_ns: i64,
    latest: &mut LatestGaugeMap,
) {
    let Some(family) = record
        .metrics
        .get(metric_name)
        .filter(|family| family.metric_type == PrometheusMetricType::Gauge)
    else {
        return;
    };
    for sample in &family.samples {
        let MetricSample::Scalar { labels, value } = sample else {
            continue;
        };
        let key = (
            record.endpoint_url.clone(),
            labels
                .iter()
                .map(|(name, value)| (name.clone(), value.clone()))
                .collect(),
        );
        let should_replace = latest
            .get(&key)
            .is_none_or(|(previous_ns, _)| timestamp_ns >= *previous_ns);
        if should_replace {
            latest.insert(key, (timestamp_ns, *value));
        }
    }
}

/// Compatibility collection metadata for one server-metrics endpoint.
#[derive(Debug, Clone, PartialEq)]
pub struct ServerMetricsEndpointInfo {
    /// All successful fetches, including duplicate bodies.
    pub total_fetches: usize,
    /// First successful fetch timestamp.
    pub first_fetch_ns: i64,
    /// Last successful fetch timestamp.
    pub last_fetch_ns: i64,
    /// Mean successful HTTP latency in milliseconds.
    pub avg_fetch_latency_ms: f64,
    /// Successful fetches whose body changed.
    pub unique_updates: usize,
    /// First changed-body timestamp, or zero when absent.
    pub first_update_ns: i64,
    /// Last changed-body timestamp, or zero when absent.
    pub last_update_ns: i64,
    /// Span between first and last changed bodies in seconds.
    pub duration_seconds: f64,
    /// Mean interval between changed bodies in milliseconds.
    pub avg_update_interval_ms: f64,
    /// Median actual changed-body interval in milliseconds.
    pub median_update_interval_ms: Option<f64>,
}

fn median_interval_ms(timestamps: &[i64]) -> Option<f64> {
    if timestamps.len() < 2 {
        return None;
    }
    let mut intervals = timestamps
        .windows(2)
        .map(|pair| pair[1].saturating_sub(pair[0]))
        .collect::<Vec<_>>();
    intervals.sort_unstable();
    let middle = intervals.len() / 2;
    let median_ns = if intervals.len() % 2 == 0 {
        (intervals[middle - 1] as f64 + intervals[middle] as f64) / 2.0
    } else {
        intervals[middle] as f64
    };
    Some(median_ns / 1_000_000.0)
}

fn ingest_series(record: &ServerMetricsRecord, states: &mut BTreeMap<SeriesKey, SeriesState>) {
    for (name, family) in &record.metrics {
        for sample in &family.samples {
            let key = SeriesKey {
                name: name.clone(),
                endpoint_url: record.endpoint_url.clone(),
                labels: sample
                    .labels()
                    .iter()
                    .map(|(key, value)| (key.clone(), value.clone()))
                    .collect(),
            };
            let value = match sample {
                MetricSample::Scalar { value, .. } => SeriesValue::Scalar(*value),
                MetricSample::Histogram { value, .. } => SeriesValue::Histogram(value.clone()),
            };
            let state = states.entry(key).or_insert_with(|| SeriesState {
                metric_type: family.metric_type,
                description: family.description.clone(),
                observations: Vec::new(),
            });
            state.observations.push(SeriesObservation {
                timestamp_ns: record.timestamp_ns,
                value,
            });
        }
    }
}

fn flatten_records(
    records: &BTreeMap<String, ServerMetricsRecord>,
) -> BTreeMap<SeriesKey, SeriesValue> {
    let mut output = BTreeMap::new();
    for record in records.values() {
        for (name, family) in &record.metrics {
            for sample in &family.samples {
                let key = SeriesKey {
                    name: name.clone(),
                    endpoint_url: record.endpoint_url.clone(),
                    labels: sample
                        .labels()
                        .iter()
                        .map(|(key, value)| (key.clone(), value.clone()))
                        .collect(),
                };
                let value = match sample {
                    MetricSample::Scalar { value, .. } => SeriesValue::Scalar(*value),
                    MetricSample::Histogram { value, .. } => SeriesValue::Histogram(value.clone()),
                };
                output.insert(key, value);
            }
        }
    }
    output
}

fn labels(key: &SeriesKey) -> Option<BTreeMap<String, String>> {
    (!key.labels.is_empty()).then(|| key.labels.iter().cloned().collect())
}

fn gauge_series(
    key: &SeriesKey,
    state: &SeriesState,
    boundary: &ServerMetricsPhaseBoundary,
    slice_duration_ns: Option<i64>,
) -> Option<SidecarSeries> {
    let values = state
        .observations
        .iter()
        .filter(|observation| {
            observation.timestamp_ns >= boundary.start_ns
                && observation.timestamp_ns <= boundary.end_ns
        })
        .filter_map(|observation| match observation.value {
            SeriesValue::Scalar(value) => Some(value),
            SeriesValue::Histogram(_) => None,
        })
        .collect::<Vec<_>>();
    let sum: f64 = values.iter().sum();
    let stats = linear_distribution(&key.name, values, sum, 1)?;
    let timeslices = slice_duration_ns
        .filter(|duration| *duration > 0)
        .map(|duration| gauge_timeslices(key, state, boundary, duration))
        .unwrap_or_default();
    Some(SidecarSeries {
        labels: labels(key),
        endpoint_url: Some(key.endpoint_url.clone()),
        stats: SidecarStats::Gauge(stats),
        timeslices,
    })
}

fn counter_series(
    key: &SeriesKey,
    state: &SeriesState,
    boundary: &ServerMetricsPhaseBoundary,
    start: Option<&SeriesValue>,
    end: Option<&SeriesValue>,
    duration_seconds: Option<f64>,
    slice_duration_ns: Option<i64>,
) -> Option<SidecarSeries> {
    let end = scalar_value(end)?;
    // A counter absent from the start boundary scrape appeared during the phase
    // (server exposition families are created lazily on first request), so its
    // value at phase start was zero. Only a missing *end* value omits the series.
    let start = scalar_value(start).unwrap_or(0.0);
    let total = (end - start).max(0.0);
    let rate = duration_seconds.map(|duration| total / duration);
    let timeslices = slice_duration_ns
        .filter(|duration| *duration > 0)
        .map(|duration| counter_timeslices(state, boundary, duration))
        .unwrap_or_default();
    Some(SidecarSeries {
        labels: labels(key),
        endpoint_url: Some(key.endpoint_url.clone()),
        stats: SidecarStats::Counter {
            total: MetricValue::from_f64(total, false),
            rate: rate.map(|value| MetricValue::from_f64(value, false)),
        },
        timeslices,
    })
}

fn histogram_series(
    key: &SeriesKey,
    state: &SeriesState,
    boundary: &ServerMetricsPhaseBoundary,
    start: Option<&SeriesValue>,
    end: Option<&SeriesValue>,
    duration_seconds: Option<f64>,
    slice_duration_ns: Option<i64>,
) -> Option<SidecarSeries> {
    let end = histogram_value(end)?;
    // A histogram absent from the start boundary scrape appeared during the phase
    // (lazily-created exposition family), so its phase-start baseline is empty
    // (sum 0, count 0, no bucket counts). Only a missing *end* sum/count omits it.
    let start = histogram_value(start);
    let start_sum = start.and_then(|value| value.sum).unwrap_or(0.0);
    let start_count = start.and_then(|value| value.count).unwrap_or(0.0);
    let sum = (end.sum? - start_sum).max(0.0);
    let count = (end.count? - start_count).max(0.0) as u64;
    let bucket_names = match start {
        Some(start) => start
            .buckets
            .keys()
            .filter(|name| end.buckets.contains_key(*name))
            .cloned()
            .collect::<Vec<_>>(),
        None => end.buckets.keys().cloned().collect::<Vec<_>>(),
    };
    let buckets = bucket_names
        .into_iter()
        .map(|name| {
            let start_bucket = start
                .and_then(|start| start.buckets.get(&name).copied())
                .unwrap_or(0.0);
            let delta = (end.buckets[&name] - start_bucket).max(0.0) as u64;
            (name, delta)
        })
        .collect::<BTreeMap<_, _>>();
    let cumulative = buckets
        .iter()
        .map(|(name, value)| (name.clone(), *value as f64))
        .collect::<BTreeMap<_, _>>();
    let history = state
        .observations
        .iter()
        .filter_map(|observation| match &observation.value {
            SeriesValue::Histogram(value) => Some(HistogramSnapshot {
                timestamp_ns: observation.timestamp_ns,
                sum: value.sum?,
                count: value.count?,
                buckets: value.buckets.clone(),
            }),
            SeriesValue::Scalar(_) => None,
        })
        .collect::<Vec<_>>();
    let learned = accumulate_bucket_statistics(&history);
    let percentiles = compute_estimated_percentiles(&cumulative, &learned, sum, count)
        .unwrap_or_default()
        .into_iter()
        .map(|(percentile, value)| (percentile, MetricValue::from_f64(value, false)))
        .collect();
    let avg = (count > 0).then(|| MetricValue::from_f64(sum / count as f64, false));
    let count_rate =
        duration_seconds.map(|duration| MetricValue::from_f64(count as f64 / duration, false));
    let sum_rate = duration_seconds.map(|duration| MetricValue::from_f64(sum / duration, false));
    let timeslices = slice_duration_ns
        .filter(|duration| *duration > 0)
        .map(|duration| histogram_timeslices(state, boundary, duration))
        .unwrap_or_default();
    Some(SidecarSeries {
        labels: labels(key),
        endpoint_url: Some(key.endpoint_url.clone()),
        stats: SidecarStats::Histogram {
            count,
            sum: MetricValue::from_f64(sum, false),
            avg,
            count_rate,
            sum_rate,
            percentiles,
            buckets,
        },
        timeslices,
    })
}

fn scalar_value(value: Option<&SeriesValue>) -> Option<f64> {
    match value {
        Some(SeriesValue::Scalar(value)) => Some(*value),
        _ => None,
    }
}

fn histogram_value(value: Option<&SeriesValue>) -> Option<&HistogramValue> {
    match value {
        Some(SeriesValue::Histogram(value)) => Some(value),
        _ => None,
    }
}

fn timeslice_bounds(start_ns: i64, end_ns: i64, duration_ns: i64) -> Vec<(i64, i64, bool)> {
    let mut output = Vec::new();
    let mut start = start_ns;
    while start < end_ns {
        let end = start.saturating_add(duration_ns).min(end_ns);
        output.push((start, end, end - start == duration_ns));
        start = end;
    }
    output
}

fn gauge_timeslices(
    key: &SeriesKey,
    state: &SeriesState,
    boundary: &ServerMetricsPhaseBoundary,
    duration_ns: i64,
) -> Vec<SidecarTimeslice> {
    timeslice_bounds(boundary.start_ns, boundary.end_ns, duration_ns)
        .into_iter()
        .filter_map(|(start_ns, end_ns, complete)| {
            let values = state
                .observations
                .iter()
                .filter(|observation| {
                    observation.timestamp_ns >= start_ns
                        && (observation.timestamp_ns < end_ns
                            || (end_ns == boundary.end_ns && observation.timestamp_ns == end_ns))
                })
                .filter_map(|observation| match observation.value {
                    SeriesValue::Scalar(value) => Some(value),
                    SeriesValue::Histogram(_) => None,
                })
                .collect::<Vec<_>>();
            let sum: f64 = values.iter().sum();
            let stats = linear_distribution(&key.name, values, sum, 1)?;
            Some(SidecarTimeslice {
                start_ns,
                end_ns,
                complete,
                stats: SidecarStats::Gauge(stats),
            })
        })
        .collect()
}

fn counter_timeslices(
    state: &SeriesState,
    boundary: &ServerMetricsPhaseBoundary,
    duration_ns: i64,
) -> Vec<SidecarTimeslice> {
    timeslice_bounds(boundary.start_ns, boundary.end_ns, duration_ns)
        .into_iter()
        .filter_map(|(start_ns, end_ns, complete)| {
            let start = scalar_at_or_before(state, start_ns)?;
            let end = scalar_at_or_before(state, end_ns)?;
            let total = (end - start).max(0.0);
            let duration = (end_ns - start_ns) as f64 / NANOS_PER_SECOND;
            Some(SidecarTimeslice {
                start_ns,
                end_ns,
                complete,
                stats: SidecarStats::Counter {
                    total: MetricValue::from_f64(total, false),
                    rate: (duration > 0.0).then(|| MetricValue::from_f64(total / duration, false)),
                },
            })
        })
        .collect()
}

fn histogram_timeslices(
    state: &SeriesState,
    boundary: &ServerMetricsPhaseBoundary,
    duration_ns: i64,
) -> Vec<SidecarTimeslice> {
    timeslice_bounds(boundary.start_ns, boundary.end_ns, duration_ns)
        .into_iter()
        .filter_map(|(start_ns, end_ns, complete)| {
            let start = histogram_at_or_before(state, start_ns)?;
            let end = histogram_at_or_before(state, end_ns)?;
            let sum = (end.sum? - start.sum?).max(0.0);
            let count = (end.count? - start.count?).max(0.0) as u64;
            let names = start
                .buckets
                .keys()
                .filter(|name| end.buckets.contains_key(*name))
                .cloned()
                .collect::<Vec<_>>();
            let buckets = names
                .into_iter()
                .map(|name| {
                    let delta = (end.buckets[&name] - start.buckets[&name]).max(0.0) as u64;
                    (name, delta)
                })
                .collect();
            let duration = (end_ns - start_ns) as f64 / NANOS_PER_SECOND;
            Some(SidecarTimeslice {
                start_ns,
                end_ns,
                complete,
                stats: SidecarStats::Histogram {
                    count,
                    sum: MetricValue::from_f64(sum, false),
                    avg: (count > 0).then(|| MetricValue::from_f64(sum / count as f64, false)),
                    count_rate: (duration > 0.0)
                        .then(|| MetricValue::from_f64(count as f64 / duration, false)),
                    sum_rate: (duration > 0.0)
                        .then(|| MetricValue::from_f64(sum / duration, false)),
                    percentiles: BTreeMap::new(),
                    buckets,
                },
            })
        })
        .collect()
}

fn scalar_at_or_before(state: &SeriesState, timestamp_ns: i64) -> Option<f64> {
    state
        .observations
        .iter()
        .rev()
        .find(|observation| observation.timestamp_ns <= timestamp_ns)
        .and_then(|observation| match observation.value {
            SeriesValue::Scalar(value) => Some(value),
            SeriesValue::Histogram(_) => None,
        })
}

fn histogram_at_or_before(state: &SeriesState, timestamp_ns: i64) -> Option<&HistogramValue> {
    state
        .observations
        .iter()
        .rev()
        .find(|observation| observation.timestamp_ns <= timestamp_ns)
        .and_then(|observation| match &observation.value {
            SeriesValue::Histogram(value) => Some(value),
            SeriesValue::Scalar(_) => None,
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::server_metrics::model::MetricFamily;

    fn record(
        timestamp_ns: i64,
        counter: f64,
        gauge: f64,
        histogram_count: f64,
    ) -> ServerMetricsRecord {
        let scalar = |value| MetricSample::Scalar {
            labels: BTreeMap::new(),
            value,
        };
        let histogram = MetricSample::Histogram {
            labels: BTreeMap::new(),
            value: HistogramValue {
                buckets: BTreeMap::from([
                    ("1.0".to_string(), histogram_count),
                    ("+Inf".to_string(), histogram_count),
                ]),
                sum: Some(histogram_count * 0.5),
                count: Some(histogram_count),
            },
        };
        ServerMetricsRecord {
            endpoint_url: "http://server/metrics".to_string(),
            timestamp_ns,
            endpoint_latency_ns: Some(10),
            request_sent_ns: Some(timestamp_ns - 10),
            first_byte_ns: Some(timestamp_ns),
            is_duplicate: false,
            benchmark_phase: Some(Phase::Profiling),
            metrics: BTreeMap::from([
                (
                    "requests".to_string(),
                    MetricFamily {
                        metric_type: PrometheusMetricType::Counter,
                        description: "Requests".to_string(),
                        samples: vec![scalar(counter)],
                    },
                ),
                (
                    "queue".to_string(),
                    MetricFamily {
                        metric_type: PrometheusMetricType::Gauge,
                        description: "Queue".to_string(),
                        samples: vec![scalar(gauge)],
                    },
                ),
                (
                    "latency_seconds".to_string(),
                    MetricFamily {
                        metric_type: PrometheusMetricType::Histogram,
                        description: "Latency in seconds".to_string(),
                        samples: vec![histogram],
                    },
                ),
            ]),
        }
    }

    fn insert_scalar(
        record: &mut ServerMetricsRecord,
        name: &str,
        metric_type: PrometheusMetricType,
        value: f64,
    ) {
        record.metrics.insert(
            name.to_string(),
            MetricFamily {
                metric_type,
                description: String::new(),
                samples: vec![MetricSample::Scalar {
                    labels: BTreeMap::new(),
                    value,
                }],
            },
        );
    }

    fn derived_value(summary: &ServerMetricsSummary, name: &str) -> f64 {
        let SidecarStats::Gauge(stats) = &summary.sidecar_metrics()[name].series[0].stats else {
            panic!("derived gauge")
        };
        let MetricValue::Finite(value) = stats.avg else {
            panic!("finite derived gauge")
        };
        value
    }

    #[test]
    fn boundary_deltas_gauges_histograms_and_timeslices_share_one_phase() {
        let start = record(0, 100.0, 2.0, 10.0);
        let middle = record(1_000_000_000, 106.0, 4.0, 16.0);
        let end = record(2_000_000_000, 110.0, 6.0, 20.0);
        let mut accumulator = ServerMetricsAccumulator::new();
        accumulator.ingest_record(start.clone());
        accumulator.ingest_record(middle);
        accumulator.ingest_record(end.clone());
        accumulator.set_phase_boundary(ServerMetricsPhaseBoundary {
            phase: Phase::Profiling,
            start_ns: 0,
            end_ns: 2_000_000_000,
            start_records: BTreeMap::from([(start.endpoint_url.clone(), start)]),
            end_records: BTreeMap::from([(end.endpoint_url.clone(), end)]),
        });
        let summary = accumulator.summarize_phase(Phase::Profiling, Some(1_000_000_000));
        let counter = &summary.sidecar_metrics()["requests"].series[0];
        assert!(matches!(
            counter.stats,
            SidecarStats::Counter {
                total: MetricValue::Finite(10.0),
                rate: Some(MetricValue::Finite(5.0))
            }
        ));
        assert_eq!(counter.timeslices.len(), 2);
        let gauge = &summary.sidecar_metrics()["queue"].series[0];
        let SidecarStats::Gauge(stats) = &gauge.stats else {
            panic!("gauge")
        };
        assert_eq!(stats.avg, MetricValue::Finite(4.0));
        let histogram = &summary.sidecar_metrics()["latency_seconds"].series[0];
        let SidecarStats::Histogram { count, sum, .. } = histogram.stats else {
            panic!("histogram")
        };
        assert_eq!(count, 10);
        assert_eq!(sum, MetricValue::Finite(5.0));
    }

    #[test]
    fn counter_resets_are_clamped_at_the_boundary() {
        let start = record(0, 100.0, 1.0, 10.0);
        let end = record(1_000_000_000, 2.0, 1.0, 2.0);
        let mut accumulator = ServerMetricsAccumulator::new();
        accumulator.ingest_record(start.clone());
        accumulator.ingest_record(end.clone());
        accumulator.set_phase_boundary(ServerMetricsPhaseBoundary {
            phase: Phase::Profiling,
            start_ns: 0,
            end_ns: 1_000_000_000,
            start_records: BTreeMap::from([(start.endpoint_url.clone(), start)]),
            end_records: BTreeMap::from([(end.endpoint_url.clone(), end)]),
        });
        let summary = accumulator.summarize_phase(Phase::Profiling, None);
        assert!(matches!(
            summary.sidecar_metrics()["requests"].series[0].stats,
            SidecarStats::Counter {
                total: MetricValue::Finite(0.0),
                ..
            }
        ));
    }

    #[test]
    fn counters_without_both_boundary_values_are_omitted() {
        let start = record(0, 100.0, 1.0, 10.0);
        let mut end = record(1_000_000_000, 110.0, 1.0, 20.0);
        end.metrics.remove("requests");
        let mut accumulator = ServerMetricsAccumulator::new();
        accumulator.ingest_record(start.clone());
        accumulator.ingest_record(end.clone());
        accumulator.set_phase_boundary(ServerMetricsPhaseBoundary {
            phase: Phase::Profiling,
            start_ns: 0,
            end_ns: 1_000_000_000,
            start_records: BTreeMap::from([(start.endpoint_url.clone(), start)]),
            end_records: BTreeMap::from([(end.endpoint_url.clone(), end)]),
        });

        let summary = accumulator.summarize_phase(Phase::Profiling, None);

        assert!(!summary.sidecar_metrics().contains_key("requests"));
    }

    #[test]
    fn histogram_boundary_delta_uses_only_complete_fields_and_shared_buckets() {
        let start = record(0, 100.0, 1.0, 10.0);
        let mut end = record(1_000_000_000, 110.0, 1.0, 20.0);
        let MetricSample::Histogram { value, .. } =
            &mut end.metrics.get_mut("latency_seconds").unwrap().samples[0]
        else {
            panic!("histogram")
        };
        value.buckets.remove("1.0");
        let mut accumulator = ServerMetricsAccumulator::new();
        accumulator.ingest_record(start.clone());
        accumulator.ingest_record(end.clone());
        accumulator.set_phase_boundary(ServerMetricsPhaseBoundary {
            phase: Phase::Profiling,
            start_ns: 0,
            end_ns: 1_000_000_000,
            start_records: BTreeMap::from([(start.endpoint_url.clone(), start)]),
            end_records: BTreeMap::from([(end.endpoint_url.clone(), end)]),
        });

        let summary = accumulator.summarize_phase(Phase::Profiling, None);
        let SidecarStats::Histogram { buckets, .. } =
            &summary.sidecar_metrics()["latency_seconds"].series[0].stats
        else {
            panic!("histogram")
        };
        assert_eq!(buckets, &BTreeMap::from([("+Inf".to_string(), 10)]));
    }

    #[test]
    fn histogram_without_boundary_sum_is_omitted() {
        let start = record(0, 100.0, 1.0, 10.0);
        let mut end = record(1_000_000_000, 110.0, 1.0, 20.0);
        let MetricSample::Histogram { value, .. } =
            &mut end.metrics.get_mut("latency_seconds").unwrap().samples[0]
        else {
            panic!("histogram")
        };
        value.sum = None;
        let mut accumulator = ServerMetricsAccumulator::new();
        accumulator.ingest_record(start.clone());
        accumulator.ingest_record(end.clone());
        accumulator.set_phase_boundary(ServerMetricsPhaseBoundary {
            phase: Phase::Profiling,
            start_ns: 0,
            end_ns: 1_000_000_000,
            start_records: BTreeMap::from([(start.endpoint_url.clone(), start)]),
            end_records: BTreeMap::from([(end.endpoint_url.clone(), end)]),
        });

        let summary = accumulator.summarize_phase(Phase::Profiling, None);

        assert!(!summary.sidecar_metrics().contains_key("latency_seconds"));
    }

    #[test]
    fn vllm_atlas_uses_exact_counter_boundaries_and_latest_gauges() {
        let mut start = record(0, 100.0, 1.0, 10.0);
        let mut end = record(2_000_000_000, 110.0, 1.0, 20.0);
        for (name, value) in [
            ("vllm:prefix_cache_hits", 100.0),
            ("vllm:prefix_cache_queries", 200.0),
            ("vllm:prompt_tokens", 1_000.0),
            ("vllm:generation_tokens", 500.0),
        ] {
            insert_scalar(&mut start, name, PrometheusMetricType::Counter, value);
        }
        for (name, value) in [
            ("vllm:prefix_cache_hits", 180.0),
            ("vllm:prefix_cache_queries", 300.0),
            ("vllm:prompt_tokens", 1_500.0),
            ("vllm:generation_tokens", 750.0),
        ] {
            insert_scalar(&mut end, name, PrometheusMetricType::Counter, value);
        }
        insert_scalar(
            &mut end,
            "vllm:kv_cache_usage_perc",
            PrometheusMetricType::Gauge,
            0.8,
        );
        let mut accumulator = ServerMetricsAccumulator::new();
        accumulator.ingest_record(start.clone());
        accumulator.ingest_record(end.clone());
        accumulator.set_phase_boundary(ServerMetricsPhaseBoundary {
            phase: Phase::Profiling,
            start_ns: 0,
            end_ns: 2_000_000_000,
            start_records: BTreeMap::from([(start.endpoint_url.clone(), start)]),
            end_records: BTreeMap::from([(end.endpoint_url.clone(), end)]),
        });

        let summary = accumulator.summarize_phase(Phase::Profiling, None);

        assert_eq!(derived_value(&summary, "prefix_cache_hit_rate"), 80.0);
        assert_eq!(derived_value(&summary, "unique_input_tokens_srv"), 20.0);
        assert_eq!(derived_value(&summary, "kv_cache_usage_pct"), 80.0);
        assert_eq!(derived_value(&summary, "input_token_throughput_srv"), 250.0);
        assert_eq!(
            derived_value(&summary, "output_token_throughput_srv"),
            125.0
        );

        // Every derived series must reference its scraped endpoint in the
        // per-endpoint collection metadata.
        for name in [
            "prefix_cache_hit_rate",
            "unique_input_tokens_srv",
            "kv_cache_usage_pct",
            "input_token_throughput_srv",
            "output_token_throughput_srv",
        ] {
            let series = &summary.sidecar_metrics()[name].series;
            assert_eq!(
                series.len(),
                1,
                "{name} should have one per-endpoint series"
            );
            assert_eq!(
                series[0].endpoint_url.as_deref(),
                Some("http://server/metrics"),
                "{name} series must carry its scraped endpoint_url"
            );
        }
    }

    #[test]
    fn derived_metrics_are_attributed_per_endpoint() {
        // Two distinct scraped endpoints, each with its own vLLM counters:
        // every derived series must be tagged with the endpoint it came from.
        let mut start_a = record(0, 100.0, 1.0, 10.0);
        let mut end_a = record(2_000_000_000, 110.0, 1.0, 20.0);
        start_a.endpoint_url = "http://a/metrics".to_string();
        end_a.endpoint_url = "http://a/metrics".to_string();
        let mut start_b = record(0, 100.0, 1.0, 10.0);
        let mut end_b = record(2_000_000_000, 110.0, 1.0, 20.0);
        start_b.endpoint_url = "http://b/metrics".to_string();
        end_b.endpoint_url = "http://b/metrics".to_string();
        insert_scalar(
            &mut start_a,
            "vllm:prompt_tokens",
            PrometheusMetricType::Counter,
            1_000.0,
        );
        insert_scalar(
            &mut end_a,
            "vllm:prompt_tokens",
            PrometheusMetricType::Counter,
            1_500.0,
        );
        insert_scalar(
            &mut start_b,
            "vllm:prompt_tokens",
            PrometheusMetricType::Counter,
            2_000.0,
        );
        insert_scalar(
            &mut end_b,
            "vllm:prompt_tokens",
            PrometheusMetricType::Counter,
            3_000.0,
        );
        let mut accumulator = ServerMetricsAccumulator::new();
        for record in [&start_a, &end_a, &start_b, &end_b] {
            accumulator.ingest_record(record.clone());
        }
        accumulator.set_phase_boundary(ServerMetricsPhaseBoundary {
            phase: Phase::Profiling,
            start_ns: 0,
            end_ns: 2_000_000_000,
            start_records: BTreeMap::from([
                (start_a.endpoint_url.clone(), start_a),
                (start_b.endpoint_url.clone(), start_b),
            ]),
            end_records: BTreeMap::from([
                (end_a.endpoint_url.clone(), end_a),
                (end_b.endpoint_url.clone(), end_b),
            ]),
        });

        let summary = accumulator.summarize_phase(Phase::Profiling, None);
        let series = &summary.sidecar_metrics()["input_token_throughput_srv"].series;
        assert_eq!(series.len(), 2);
        let by_endpoint = series
            .iter()
            .map(|series| {
                let SidecarStats::Gauge(stats) = &series.stats else {
                    panic!("derived gauge")
                };
                let MetricValue::Finite(value) = stats.avg else {
                    panic!("finite")
                };
                (series.endpoint_url.clone().unwrap(), value)
            })
            .collect::<BTreeMap<_, _>>();
        // 500 tokens / 2s and 1000 tokens / 2s, each tagged to its own endpoint.
        assert_eq!(by_endpoint["http://a/metrics"], 250.0);
        assert_eq!(by_endpoint["http://b/metrics"], 500.0);
    }

    #[test]
    fn shared_accumulator_seam_queries_half_open_and_exports_by_phase() {
        let start = record(0, 100.0, 1.0, 10.0);
        let middle = record(1_000_000_000, 105.0, 2.0, 15.0);
        let end = record(2_000_000_000, 110.0, 3.0, 20.0);
        let mut accumulator = ServerMetricsAccumulator::new();
        accumulator.process_record(&start);
        accumulator.process_record(&middle);
        accumulator.process_record(&end);
        accumulator.set_phase_boundary(ServerMetricsPhaseBoundary {
            phase: Phase::Profiling,
            start_ns: 0,
            end_ns: 2_000_000_000,
            start_records: BTreeMap::from([(start.endpoint_url.clone(), start)]),
            end_records: BTreeMap::from([(end.endpoint_url.clone(), end)]),
        });

        assert_eq!(
            accumulator.query_time_range(0, 2_000_000_000),
            vec![true, true, false]
        );
        let summary =
            Accumulator::export_results(&accumulator, &ExportContext::phase(Phase::Profiling));
        assert!(summary.sidecar_metrics().contains_key("requests"));
        let missing =
            Accumulator::export_results(&accumulator, &ExportContext::phase(Phase::Warmup));
        assert!(missing.sidecar_metrics().is_empty());
    }

    #[test]
    fn shared_accumulator_merge_rejects_conflicting_boundaries() {
        let start = record(0, 100.0, 1.0, 10.0);
        let end = record(1_000_000_000, 110.0, 1.0, 20.0);
        let boundary = ServerMetricsPhaseBoundary {
            phase: Phase::Profiling,
            start_ns: 0,
            end_ns: 1_000_000_000,
            start_records: BTreeMap::from([(start.endpoint_url.clone(), start)]),
            end_records: BTreeMap::from([(end.endpoint_url.clone(), end)]),
        };
        let mut left = ServerMetricsAccumulator::new();
        left.set_phase_boundary(boundary.clone());
        let mut right = ServerMetricsAccumulator::new();
        right.set_phase_boundary(ServerMetricsPhaseBoundary {
            end_ns: 2_000_000_000,
            ..boundary
        });

        assert_eq!(
            left.merge(&right),
            Err(ServerMetricsMergeError::BoundaryConflict(Phase::Profiling))
        );
    }
}
