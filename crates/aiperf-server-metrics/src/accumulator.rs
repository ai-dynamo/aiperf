// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Phase-boundary server counter attribution and continuous gauge aggregation.
//!
//! Domain math is grounded in `src/aiperf/server_metrics/accumulator.py:55-837`
//! and `export_stats.py:35-852`. Boundary snapshots intentionally implement
//! the native telemetry-spec addendum: counters and histogram totals use exact
//! phase start/end values, while continuous records supply gauge distributions,
//! timeslices, and histogram bucket-mean learning.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use aiperf_metrics::{
    AccumulatorSummary, MetricValue, Phase, SidecarMetric, SidecarSeries, SidecarStats,
    SidecarTimeslice, Unit, linear_distribution,
};

use crate::histogram::{
    HistogramSnapshot, accumulate_bucket_statistics, compute_estimated_percentiles,
};
use crate::model::{HistogramValue, MetricSample, PrometheusMetricType, ServerMetricsRecord};
use crate::units::infer_unit;

const NANOS_PER_SECOND: f64 = 1_000_000_000.0;

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

impl ServerMetricsAccumulator {
    /// Build an empty accumulator.
    pub fn new() -> Self {
        Self::default()
    }

    /// Retain one complete parsed scrape.
    pub fn ingest_record(&mut self, record: ServerMetricsRecord) {
        self.records.push(record);
        self.records.sort_by(|left, right| {
            left.timestamp_ns
                .cmp(&right.timestamp_ns)
                .then_with(|| left.endpoint_url.cmp(&right.endpoint_url))
        });
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

        let sidecar_metrics = metrics
            .into_iter()
            .map(|(name, (unit, mut series))| {
                series.sort_by(|left, right| {
                    left.endpoint_url
                        .cmp(&right.endpoint_url)
                        .then_with(|| left.labels.cmp(&right.labels))
                });
                (name, SidecarMetric::new(unit, series))
            })
            .collect();
        let endpoints_successful = boundary
            .start_records
            .keys()
            .filter(|endpoint| boundary.end_records.contains_key(*endpoint))
            .cloned()
            .collect();
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
}

/// Python-compatible collection metadata for one server-metrics endpoint.
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
    let stats = linear_distribution(&key.name, values.clone(), values.iter().sum(), 1)?;
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
    let start = scalar_value(start).or_else(|| first_scalar(state))?;
    let end = scalar_value(end).or_else(|| last_scalar(state))?;
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
    let start = histogram_value(start).or_else(|| first_histogram(state))?;
    let end = histogram_value(end).or_else(|| last_histogram(state))?;
    let sum = (end.sum.unwrap_or(0.0) - start.sum.unwrap_or(0.0)).max(0.0);
    let count = (end.count.unwrap_or(0.0) - start.count.unwrap_or(0.0)).max(0.0) as u64;
    let bucket_names = start
        .buckets
        .keys()
        .chain(end.buckets.keys())
        .cloned()
        .collect::<BTreeSet<_>>();
    let buckets = bucket_names
        .into_iter()
        .map(|name| {
            let delta = (end.buckets.get(&name).copied().unwrap_or(0.0)
                - start.buckets.get(&name).copied().unwrap_or(0.0))
            .max(0.0) as u64;
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
                sum: value.sum.unwrap_or(0.0),
                count: value.count.unwrap_or(0.0),
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

fn first_scalar(state: &SeriesState) -> Option<f64> {
    state
        .observations
        .iter()
        .find_map(|observation| match observation.value {
            SeriesValue::Scalar(value) => Some(value),
            SeriesValue::Histogram(_) => None,
        })
}

fn last_scalar(state: &SeriesState) -> Option<f64> {
    state
        .observations
        .iter()
        .rev()
        .find_map(|observation| match observation.value {
            SeriesValue::Scalar(value) => Some(value),
            SeriesValue::Histogram(_) => None,
        })
}

fn first_histogram(state: &SeriesState) -> Option<&HistogramValue> {
    state
        .observations
        .iter()
        .find_map(|observation| match &observation.value {
            SeriesValue::Histogram(value) => Some(value),
            SeriesValue::Scalar(_) => None,
        })
}

fn last_histogram(state: &SeriesState) -> Option<&HistogramValue> {
    state
        .observations
        .iter()
        .rev()
        .find_map(|observation| match &observation.value {
            SeriesValue::Histogram(value) => Some(value),
            SeriesValue::Scalar(_) => None,
        })
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
            let stats = linear_distribution(&key.name, values.clone(), values.iter().sum(), 1)?;
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
            let sum = (end.sum.unwrap_or(0.0) - start.sum.unwrap_or(0.0)).max(0.0);
            let count = (end.count.unwrap_or(0.0) - start.count.unwrap_or(0.0)).max(0.0) as u64;
            let names = start
                .buckets
                .keys()
                .chain(end.buckets.keys())
                .cloned()
                .collect::<BTreeSet<_>>();
            let buckets = names
                .into_iter()
                .map(|name| {
                    let delta = (end.buckets.get(&name).copied().unwrap_or(0.0)
                        - start.buckets.get(&name).copied().unwrap_or(0.0))
                    .max(0.0) as u64;
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
    use crate::model::MetricFamily;

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
}
