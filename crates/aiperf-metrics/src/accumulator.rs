// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Columnar inference-metric accumulation, derivation, windowing, and timeslicing.
//!
//! The dispatch order, authoritative phase masks, sweep injection, and half-open
//! timeslicing port `src/aiperf/metrics/accumulator.py:54-618`; the Rust runtime
//! supplies the transport-neutral [`RecordIngest`] facts consumed here.

use crate::catalog::{
    AggregationKind, CATALOG, MetricConsoleGroup, MetricFlags, MetricSpec, MetricTag, MetricType,
    spec_for, validate_catalog,
};
use crate::ingest::RecordIngest;
use crate::kernel::{DistributionStats, linear_distribution, nearest_distribution};
use crate::sidecar::SidecarMetric;
use crate::store::{ColumnStore, ListMetricBackend};
use crate::sweepline::{IclSeries, SweepLineCurves, SweepMetricResult};
use crate::units::{Unit, UnitConversionError};
use crate::value::MetricValue;
use crate::window::ExportContext;
use rustc_hash::FxHashMap;
use serde::Serialize;
use std::collections::BTreeMap;
use std::fmt::{Display, Formatter, Result as FmtResult};

const NANOS_PER_SECOND: f64 = 1_000_000_000.0;
const DEFAULT_USAGE_DIFF_THRESHOLD_PCT: f64 = 10.0;
const DEFAULT_OSL_MISMATCH_THRESHOLD_PCT: f64 = 5.0;
const DEFAULT_OSL_MISMATCH_MAX_TOKENS: f64 = 50.0;

/// Extension seam for append-only record accumulation and windowed export.
pub trait Accumulator<Record> {
    /// Typed summary returned by this accumulator.
    type Summary;
    /// Error returned when independently accumulated worker state cannot merge.
    type MergeError;

    /// Ingests one record.
    fn process_record(&mut self, record: &Record);

    /// Selects records whose start timestamps fall in `[start_ns, end_ns)`.
    fn query_time_range(&self, start_ns: i64, end_ns: i64) -> Vec<bool>;

    /// Exports a context-scoped summary.
    fn export_results(&self, context: &ExportContext) -> Self::Summary;

    /// Merges another worker's append-only state at a synchronization boundary.
    fn merge(&mut self, other: &Self) -> Result<(), Self::MergeError>;
}

/// Incompatibility detected while merging per-worker metric accumulators.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetricsMergeError {
    /// Workers were constructed with different metric configuration.
    ConfigMismatch,
    /// Workers carried different run-level network RTT calibration.
    NetworkRttMismatch,
    /// Workers injected conflicting values for the same run-level metric.
    InjectedScalarConflict(MetricTag),
}

impl Display for MetricsMergeError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> FmtResult {
        match self {
            Self::ConfigMismatch => formatter.write_str("metrics configurations do not match"),
            Self::NetworkRttMismatch => {
                formatter.write_str("network RTT calibrations do not match")
            }
            Self::InjectedScalarConflict(tag) => {
                write!(formatter, "conflicting injected scalar for {tag}")
            }
        }
    }
}

impl std::error::Error for MetricsMergeError {}

/// One configured goodput service-level threshold in native metric units.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SloThreshold {
    /// Metric evaluated per request.
    pub tag: MetricTag,
    /// Threshold converted to the metric's native unit.
    pub native_value: f64,
}

impl SloThreshold {
    /// Builds a threshold already expressed in native units.
    pub const fn native(tag: MetricTag, native_value: f64) -> Self {
        Self { tag, native_value }
    }

    /// Converts a display-unit threshold into native units using the catalog.
    pub fn from_display(tag: MetricTag, display_value: f64) -> Result<Self, UnitConversionError> {
        let spec = spec_for(tag).expect("SLO tags must resolve in the static catalog");
        let display_unit = spec.display_unit.unwrap_or(spec.unit);
        Ok(Self {
            tag,
            native_value: display_unit.convert_value(display_value, spec.unit)?,
        })
    }
}

/// Runtime-independent configuration for the metrics engine.
#[derive(Debug, Clone, PartialEq)]
pub struct MetricsConfig {
    /// Optional timeslice duration in nanoseconds.
    pub slice_duration_ns: Option<i64>,
    /// Per-request SLOs used by good-request count and goodput.
    pub slos: Vec<SloThreshold>,
    /// Usage client/server discrepancy threshold in percent.
    pub usage_diff_threshold_pct: f64,
    /// Requested-vs-actual OSL percentage threshold.
    pub osl_mismatch_threshold_pct: f64,
    /// Absolute OSL mismatch cap in tokens.
    pub osl_mismatch_max_tokens: f64,
}

impl Default for MetricsConfig {
    fn default() -> Self {
        Self {
            slice_duration_ns: None,
            slos: Vec::new(),
            usage_diff_threshold_pct: DEFAULT_USAGE_DIFF_THRESHOLD_PCT,
            osl_mismatch_threshold_pct: DEFAULT_OSL_MISMATCH_THRESHOLD_PCT,
            osl_mismatch_max_tokens: DEFAULT_OSL_MISMATCH_MAX_TOKENS,
        }
    }
}

/// Type-appropriate statistics carried by a metric result.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "type", content = "stats", rename_all = "snake_case")]
pub enum MetricResultData {
    /// One run-level scalar.
    Scalar {
        /// Boundary-safe scalar value.
        value: MetricValue,
    },
    /// A per-record or time-weighted distribution.
    Distribution(DistributionStats),
}

/// One fully summarized metric with catalog metadata.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct MetricResult {
    /// Stable report tag, including dynamic `adj_*` tags.
    pub tag: String,
    /// Parent catalog tag when one exists.
    #[serde(skip)]
    pub source_tag: Option<MetricTag>,
    /// Human-readable metric header.
    pub header: String,
    /// Unit after display conversion.
    pub unit: String,
    /// Console section; presentation consumers may use it without re-reading the catalog.
    #[serde(skip)]
    pub console_group: MetricConsoleGroup,
    /// Scalar or distribution statistics.
    #[serde(flatten)]
    pub data: MetricResultData,
}

impl MetricResult {
    /// Builds a scalar result using catalog metadata.
    pub fn scalar(tag: MetricTag, value: MetricValue) -> Self {
        let spec = spec_for(tag).expect("metric result tags must resolve in the catalog");
        Self::scalar_from_spec(spec, value)
    }

    /// Builds a finite scalar result using catalog metadata.
    pub fn finite(tag: MetricTag, value: f64) -> Self {
        Self::scalar(tag, MetricValue::from_f64(value, false))
    }

    /// Returns the scalar value or the average of a distribution.
    pub fn representative_value(&self) -> MetricValue {
        match &self.data {
            MetricResultData::Scalar { value } => *value,
            MetricResultData::Distribution(stats) => stats.avg,
        }
    }

    /// Returns the finite representative value.
    pub fn finite_value(&self) -> Option<f64> {
        self.representative_value().as_f64()
    }

    /// Returns distribution statistics when this is a distribution metric.
    pub fn distribution(&self) -> Option<&DistributionStats> {
        match &self.data {
            MetricResultData::Distribution(stats) => Some(stats),
            MetricResultData::Scalar { .. } => None,
        }
    }

    fn scalar_from_spec(spec: &MetricSpec, value: MetricValue) -> Self {
        let display_unit = spec.display_unit.unwrap_or(spec.unit);
        Self {
            tag: spec.tag.as_str().to_string(),
            source_tag: Some(spec.tag),
            header: spec.header.to_string(),
            unit: display_unit.as_str().to_string(),
            console_group: spec.console_group,
            data: MetricResultData::Scalar {
                value: convert_metric_value(value, spec.unit, display_unit),
            },
        }
    }

    fn distribution_from_spec(spec: &MetricSpec, stats: DistributionStats) -> Self {
        let display_unit = spec.display_unit.unwrap_or(spec.unit);
        Self {
            tag: spec.tag.as_str().to_string(),
            source_tag: Some(spec.tag),
            header: spec.header.to_string(),
            unit: display_unit.as_str().to_string(),
            console_group: spec.console_group,
            data: MetricResultData::Distribution(convert_distribution(
                stats,
                spec.unit,
                display_unit,
            )),
        }
    }

    fn adjusted_from_parent(spec: &MetricSpec, stats: DistributionStats) -> Self {
        let display_unit = spec.display_unit.unwrap_or(spec.unit);
        Self {
            tag: format!("adj_{}", spec.tag.as_str()),
            source_tag: Some(spec.tag),
            header: format!("{} (error-adjusted)", spec.header),
            unit: display_unit.as_str().to_string(),
            console_group: spec.console_group,
            data: MetricResultData::Distribution(convert_distribution(
                stats,
                spec.unit,
                display_unit,
            )),
        }
    }
}

/// One chronological timeslice and its metric results.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct MetricTimeslice {
    /// Inclusive timeslice start in nanoseconds.
    pub start_ns: i64,
    /// Exclusive timeslice end in nanoseconds.
    pub end_ns: i64,
    /// Absent for complete slices and false for a run-end-clipped slice.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub complete: Option<bool>,
    /// Results keyed by stable metric name.
    pub metrics: BTreeMap<String, MetricResult>,
}

/// Full summary produced by [`MetricsAccumulator`].
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct AccumulatorSummary {
    results: BTreeMap<String, MetricResult>,
    timeslices: Vec<MetricTimeslice>,
    sidecar_metrics: BTreeMap<String, SidecarMetric>,
}

impl AccumulatorSummary {
    /// Builds an empty summary.
    pub fn new() -> Self {
        Self::default()
    }

    /// Inserts or replaces a scalar metric, retained for analyzer joins.
    pub fn insert(&mut self, tag: MetricTag, value: MetricValue) {
        self.insert_result(MetricResult::scalar(tag, value));
    }

    /// Inserts a finite scalar metric.
    pub fn insert_finite(&mut self, tag: MetricTag, value: f64) {
        self.insert(tag, MetricValue::from_f64(value, false));
    }

    /// Inserts or replaces a full metric result.
    pub fn insert_result(&mut self, result: MetricResult) {
        self.results.insert(result.tag.clone(), result);
    }

    /// Returns a result by catalog tag.
    pub fn result(&self, tag: MetricTag) -> Option<&MetricResult> {
        self.results.get(tag.as_str())
    }

    /// Returns a result by stable string key, including dynamic adjusted metrics.
    pub fn result_by_name(&self, tag: &str) -> Option<&MetricResult> {
        self.results.get(tag)
    }

    /// Returns a representative boundary value by catalog tag.
    pub fn value(&self, tag: MetricTag) -> Option<MetricValue> {
        self.result(tag).map(MetricResult::representative_value)
    }

    /// Returns a finite representative value by catalog tag.
    pub fn finite_value(&self, tag: MetricTag) -> Option<f64> {
        self.result(tag).and_then(MetricResult::finite_value)
    }

    /// Iterates over results in stable tag order.
    pub fn results(&self) -> impl Iterator<Item = (&str, &MetricResult)> {
        self.results
            .iter()
            .map(|(tag, result)| (tag.as_str(), result))
    }

    /// Returns all results as a stable ordered map.
    pub fn result_map(&self) -> &BTreeMap<String, MetricResult> {
        &self.results
    }

    /// Returns chronological non-empty timeslices.
    pub fn timeslices(&self) -> &[MetricTimeslice] {
        &self.timeslices
    }

    /// Inserts or replaces one domain-neutral telemetry/server metric.
    pub fn insert_sidecar_metric(&mut self, name: impl Into<String>, metric: SidecarMetric) {
        self.sidecar_metrics.insert(name.into(), metric);
    }

    /// Extends the summary with externally accumulated metric series.
    pub fn extend_sidecar_metrics(
        &mut self,
        metrics: impl IntoIterator<Item = (String, SidecarMetric)>,
    ) {
        self.sidecar_metrics.extend(metrics);
    }

    /// Returns side-channel metrics in stable name order.
    pub fn sidecar_metrics(&self) -> &BTreeMap<String, SidecarMetric> {
        &self.sidecar_metrics
    }
}

/// Native columnar metrics accumulator.
#[derive(Debug)]
pub struct MetricsAccumulator {
    store: ColumnStore,
    config: MetricsConfig,
    network_rtt_ns: Option<f64>,
    injected_scalars: FxHashMap<MetricTag, MetricValue>,
}

impl Default for MetricsAccumulator {
    fn default() -> Self {
        Self::with_config(MetricsConfig::default())
    }
}

impl MetricsAccumulator {
    /// Builds an accumulator with default thresholds and no timeslicing.
    pub fn new() -> Self {
        Self::default()
    }

    /// Builds an accumulator with explicit runtime-independent configuration.
    pub fn with_config(config: MetricsConfig) -> Self {
        validate_catalog().expect("the static metric catalog must be valid");
        Self {
            store: ColumnStore::new(),
            config,
            network_rtt_ns: None,
            injected_scalars: FxHashMap::default(),
        }
    }

    /// Returns the underlying read-only column store for analyzers.
    pub fn column_store(&self) -> &ColumnStore {
        &self.store
    }

    /// Returns the number of append-only records.
    pub fn record_count(&self) -> usize {
        self.store.row_count()
    }

    /// Ingests one record and computes all record/aggregate inputs available from it.
    pub fn process_record(&mut self, record: &RecordIngest) {
        let row = self.store.push_record(record);
        if record.errored || record.canceled {
            return;
        }
        self.compute_record_metrics(row);
        self.compute_good_request(row);
    }

    /// Replaces the current SLO set. Thresholds must already be in native units.
    pub fn set_slos(&mut self, slos: Vec<SloThreshold>) {
        self.config.slos = slos;
    }

    /// Supplies the mean network RTT used for network-adjusted latency distributions.
    pub fn set_network_rtt_ns(&mut self, network_rtt_ns: Option<f64>) {
        self.network_rtt_ns = network_rtt_ns.filter(|value| value.is_finite() && *value > 0.0);
    }

    /// Injects a run-level scalar owned by another accumulator, such as GPU energy.
    pub fn inject_scalar(&mut self, tag: MetricTag, value: MetricValue) {
        self.injected_scalars.insert(tag, value);
    }

    /// Merges append-only state from another worker without replaying records.
    pub fn merge(&mut self, other: &Self) -> Result<(), MetricsMergeError> {
        if self.config != other.config {
            return Err(MetricsMergeError::ConfigMismatch);
        }
        if self.network_rtt_ns.is_some()
            && other.network_rtt_ns.is_some()
            && self.network_rtt_ns != other.network_rtt_ns
        {
            return Err(MetricsMergeError::NetworkRttMismatch);
        }
        for (tag, value) in &other.injected_scalars {
            if self
                .injected_scalars
                .get(tag)
                .is_some_and(|existing| existing != value)
            {
                return Err(MetricsMergeError::InjectedScalarConflict(*tag));
            }
        }

        self.network_rtt_ns = self.network_rtt_ns.or(other.network_rtt_ns);
        self.injected_scalars.extend(
            other
                .injected_scalars
                .iter()
                .map(|(tag, value)| (*tag, *value)),
        );
        self.store.append_store(&other.store);
        Ok(())
    }

    /// Selects rows by half-open start timestamp.
    pub fn query_time_range(&self, start_ns: i64, end_ns: i64) -> Vec<bool> {
        self.store.mask_started_in(Some(start_ns), Some(end_ns))
    }

    /// Summarizes all records.
    pub fn summarize(&self) -> AccumulatorSummary {
        self.export_results(&ExportContext::all())
    }

    /// Summarizes a phase or time context, including sweeps and configured timeslices.
    pub fn export_results(&self, context: &ExportContext) -> AccumulatorSummary {
        let mask = self.store.mask_for(context);
        if !mask.iter().any(|selected| *selected) {
            return AccumulatorSummary::new();
        }
        let mut results = self.compute_result_map(&mask, context.start_ns, context.end_ns);
        let curves = self.compute_sweep_curves(&mask);
        self.inject_sweep_results(
            &mut results,
            &curves,
            context.start_ns.map(|value| value as f64),
            context.end_ns.map(|value| value as f64),
        );
        let timeslices = self.compute_timeslices(&mask, &curves);
        AccumulatorSummary {
            results,
            timeslices,
            sidecar_metrics: BTreeMap::new(),
        }
    }

    fn compute_record_metrics(&mut self, row: usize) {
        let latency = self.store.metric_f64(row, MetricTag::RequestLatency);
        let ttft = self.store.metric_f64(row, MetricTag::TimeToFirstToken);
        let osl = self.store.metric_f64(row, MetricTag::OutputSequenceLength);
        let isl = self.store.metric_f64(row, MetricTag::InputSequenceLength);

        if let (Some(latency), Some(ttft), Some(osl)) = (latency, ttft, osl)
            && osl >= 2.0
        {
            self.set_finite_record(
                row,
                MetricTag::InterTokenLatency,
                (latency - ttft) / (osl - 1.0),
            );
        }
        if let Some(itl) = self.store.metric_f64(row, MetricTag::InterTokenLatency)
            && itl != 0.0
        {
            self.set_finite_record(
                row,
                MetricTag::OutputTokenThroughputPerUser,
                NANOS_PER_SECOND / itl,
            );
        }
        if let (Some(osl), Some(latency)) = (osl, latency)
            && latency != 0.0
        {
            self.set_finite_record(
                row,
                MetricTag::E2eOutputTokenThroughput,
                osl * NANOS_PER_SECOND / latency,
            );
        }
        if let (Some(isl), Some(ttft)) = (isl, ttft)
            && ttft != 0.0
        {
            self.set_finite_record(
                row,
                MetricTag::PrefillThroughputPerUser,
                isl * NANOS_PER_SECOND / ttft,
            );
        }
        if let (Some(audio_seconds), Some(latency)) = (
            self.store.metric_f64(row, MetricTag::AudioDuration),
            latency,
        ) && latency > 0.0
        {
            self.set_finite_record(
                row,
                MetricTag::Rtfx,
                audio_seconds * NANOS_PER_SECOND / latency,
            );
        }

        self.compute_usage_differences(row);
        self.compute_osl_mismatch(row);

        if let (Some(reasoning), Some(output)) = (
            self.store.metric_f64(row, MetricTag::ReasoningTokenCount),
            self.store.metric_f64(row, MetricTag::OutputTokenCount),
        ) && output != 0.0
        {
            self.set_finite_record(row, MetricTag::ThinkingEfficiency, reasoning / output);
        }

        if let (Some(images), Some(latency)) =
            (self.store.metric_f64(row, MetricTag::NumImages), latency)
        {
            if latency != 0.0 {
                self.set_finite_record(
                    row,
                    MetricTag::ImageThroughput,
                    images * NANOS_PER_SECOND / latency,
                );
            }
            if images != 0.0 {
                self.set_finite_record(
                    row,
                    MetricTag::ImageLatency,
                    latency / 1_000_000.0 / images,
                );
            }
        }

        self.sum_record_metrics(
            row,
            MetricTag::HttpReqConnectionOverhead,
            &[
                MetricTag::HttpReqBlocked,
                MetricTag::HttpReqDnsLookup,
                MetricTag::HttpReqConnecting,
            ],
        );
        self.sum_record_metrics(
            row,
            MetricTag::HttpReqTotal,
            &[
                MetricTag::HttpReqBlocked,
                MetricTag::HttpReqDnsLookup,
                MetricTag::HttpReqConnecting,
                MetricTag::HttpReqSending,
                MetricTag::HttpReqWaiting,
                MetricTag::HttpReqReceiving,
            ],
        );
        if let (Some(ttft), Some(setup)) = (
            ttft,
            self.store.metric_f64(row, MetricTag::StreamSetupLatency),
        ) {
            self.set_finite_record(row, MetricTag::StreamPrefillLatency, ttft - setup);
        }
    }

    fn compute_usage_differences(&mut self, row: usize) {
        let prompt = percent_difference(
            self.store.metric_f64(row, MetricTag::UsagePromptTokens),
            self.store.metric_f64(row, MetricTag::InputSequenceLength),
        );
        let completion = percent_difference(
            self.store.metric_f64(row, MetricTag::UsageCompletionTokens),
            self.store.metric_f64(row, MetricTag::OutputSequenceLength),
        );
        let reasoning = percent_difference(
            self.store.metric_f64(row, MetricTag::UsageReasoningTokens),
            self.store.metric_f64(row, MetricTag::ReasoningTokenCount),
        );
        if let Some(value) = prompt {
            self.set_finite_record(row, MetricTag::UsagePromptTokensDiffPct, value);
        }
        if let Some(value) = completion {
            self.set_finite_record(row, MetricTag::UsageCompletionTokensDiffPct, value);
        }
        if let Some(value) = reasoning {
            self.set_finite_record(row, MetricTag::UsageReasoningTokensDiffPct, value);
        }
        if let (Some(prompt), Some(completion)) = (prompt, completion) {
            let discrepant = prompt > self.config.usage_diff_threshold_pct
                || completion > self.config.usage_diff_threshold_pct
                || reasoning.is_some_and(|value| value > self.config.usage_diff_threshold_pct);
            self.store
                .set_metric_f64(row, MetricTag::UsageDiscrepancyCount, f64::from(discrepant));
        }
    }

    fn compute_osl_mismatch(&mut self, row: usize) {
        let requested = self
            .store
            .metric_f64(row, MetricTag::RequestedOutputSequenceLength);
        let actual = self.store.metric_f64(row, MetricTag::OutputSequenceLength);
        let (Some(requested), Some(actual)) = (requested, actual) else {
            return;
        };
        if requested == 0.0 {
            return;
        }
        let diff_pct = (actual - requested) / requested * 100.0;
        self.set_finite_record(row, MetricTag::OslMismatchDiffPct, diff_pct);
        let threshold = (requested * self.config.osl_mismatch_threshold_pct / 100.0)
            .min(self.config.osl_mismatch_max_tokens);
        self.store.set_metric_f64(
            row,
            MetricTag::OslMismatchCount,
            f64::from((actual - requested).abs() > threshold),
        );
    }

    fn compute_good_request(&mut self, row: usize) {
        if self.config.slos.is_empty() {
            return;
        }
        let passes = self.config.slos.iter().all(|slo| {
            let Some(value) = self.store.metric_f64(row, slo.tag) else {
                return false;
            };
            let larger_is_better = spec_for(slo.tag)
                .is_some_and(|spec| spec.flags.contains(MetricFlags::LARGER_IS_BETTER));
            if larger_is_better {
                value >= slo.native_value
            } else {
                value <= slo.native_value
            }
        });
        self.store
            .set_metric_f64(row, MetricTag::GoodRequestCount, f64::from(passes));
    }

    fn sum_record_metrics(&mut self, row: usize, output: MetricTag, inputs: &[MetricTag]) {
        let values = inputs
            .iter()
            .filter_map(|tag| self.store.metric_f64(row, *tag))
            .collect::<Vec<_>>();
        if !values.is_empty() {
            self.set_finite_record(row, output, values.into_iter().sum());
        }
    }

    fn set_finite_record(&mut self, row: usize, tag: MetricTag, value: f64) {
        if value.is_finite() {
            self.store.set_metric_f64(row, tag, value);
        }
    }

    fn compute_result_map(
        &self,
        mask: &[bool],
        window_start_ns: Option<i64>,
        window_end_ns: Option<i64>,
    ) -> BTreeMap<String, MetricResult> {
        let full_dataset = mask.iter().all(|selected| *selected);
        let mut scalars = FxHashMap::<MetricTag, f64>::default();
        let mut record_arrays = FxHashMap::<MetricTag, (Vec<f64>, f64)>::default();
        let mut results = BTreeMap::new();

        for spec in CATALOG
            .iter()
            .filter(|spec| spec.kind != MetricType::Derived)
        {
            if spec.tag == MetricTag::InterChunkLatency {
                let Some(backend) = self.store.ragged_column(spec.tag) else {
                    continue;
                };
                let values = backend.values_for_mask(mask);
                if values.is_empty() {
                    continue;
                }
                let sum = values.iter().sum::<f64>();
                scalars.insert(spec.tag, sum);
                record_arrays.insert(spec.tag, (values.clone(), sum));
                if let Some(stats) = linear_distribution(spec.tag.as_str(), values, sum, 0) {
                    let result = MetricResult::distribution_from_spec(spec, stats);
                    results.insert(result.tag.clone(), result);
                }
                continue;
            }

            let Some(column) = self.store.numeric_column(spec.tag) else {
                continue;
            };
            let values = column.masked_values(mask);
            if values.is_empty() {
                continue;
            }
            match spec.kind {
                MetricType::Record => {
                    let sum = if full_dataset
                        && column.running_sum().is_finite()
                        && column.present_count() == values.len()
                    {
                        column.running_sum()
                    } else {
                        values.iter().sum()
                    };
                    scalars.insert(spec.tag, sum);
                    record_arrays.insert(spec.tag, (values.clone(), sum));
                    if let Some(stats) = linear_distribution(spec.tag.as_str(), values, sum, 0) {
                        let result = MetricResult::distribution_from_spec(spec, stats);
                        results.insert(result.tag.clone(), result);
                    }
                }
                MetricType::Aggregate => {
                    let scalar =
                        aggregate_values(&values, spec.aggregation.unwrap_or(AggregationKind::Sum));
                    scalars.insert(spec.tag, scalar);
                    let result =
                        MetricResult::scalar_from_spec(spec, MetricValue::from_f64(scalar, false));
                    results.insert(result.tag.clone(), result);
                }
                MetricType::Derived => unreachable!(),
            }
        }

        for (tag, value) in &self.injected_scalars {
            if let Some(value) = value.as_f64() {
                scalars.insert(*tag, value);
            }
            let result = MetricResult::scalar(*tag, *value);
            results.insert(result.tag.clone(), result);
        }

        let observation_duration_ns =
            observation_duration(&scalars, window_start_ns, window_end_ns);
        let order = validate_catalog().expect("catalog validated during construction");
        for tag in order {
            let Some(spec) = spec_for(tag) else {
                continue;
            };
            if spec.kind != MetricType::Derived || scalars.contains_key(&tag) || is_injected(tag) {
                continue;
            }
            if let Some(value) = derive_scalar(tag, &scalars, observation_duration_ns) {
                scalars.insert(tag, value);
                let result =
                    MetricResult::scalar_from_spec(spec, MetricValue::from_f64(value, false));
                results.insert(result.tag.clone(), result);
            }
        }

        let error_count = mask
            .iter()
            .zip(self.store.errored())
            .filter(|(selected, errored)| **selected && **errored)
            .count();
        if error_count > 0 {
            for (tag, (values, sum)) in &record_arrays {
                let Some(spec) = spec_for(*tag) else {
                    continue;
                };
                if !spec
                    .flags
                    .contains(MetricFlags::PERCENTILE_INCLUDES_FAILED_REQUESTS)
                {
                    continue;
                }
                let mut inflated = values.clone();
                inflated.extend(std::iter::repeat_n(f64::INFINITY, error_count));
                if let Some(stats) =
                    nearest_distribution(format!("adj_{}", tag.as_str()), inflated, *sum, true)
                {
                    let result = MetricResult::adjusted_from_parent(spec, stats);
                    results.insert(result.tag.clone(), result);
                }
            }
        }

        self.inject_network_adjusted(mask, &mut results);
        results
    }

    fn inject_network_adjusted(&self, mask: &[bool], results: &mut BTreeMap<String, MetricResult>) {
        let Some(rtt_ns) = self.network_rtt_ns else {
            return;
        };
        for (target, source) in [
            (
                MetricTag::NetworkAdjustedRequestLatency,
                MetricTag::RequestLatency,
            ),
            (
                MetricTag::NetworkAdjustedTimeToFirstToken,
                MetricTag::TimeToFirstToken,
            ),
            (
                MetricTag::NetworkAdjustedTimeToFirstOutputToken,
                MetricTag::TimeToFirstOutputToken,
            ),
        ] {
            let Some(column) = self.store.numeric_column(source) else {
                continue;
            };
            let adjusted = column
                .masked_values(mask)
                .into_iter()
                .map(|value| (value - rtt_ns).max(0.0))
                .collect::<Vec<_>>();
            let sum = adjusted.iter().sum::<f64>();
            let Some(stats) = linear_distribution(target.as_str(), adjusted, sum, 0) else {
                continue;
            };
            let result = MetricResult::distribution_from_spec(
                spec_for(target).expect("network metric is cataloged"),
                stats,
            );
            results.insert(result.tag.clone(), result);
        }
        let result =
            MetricResult::scalar(MetricTag::NetworkRtt, MetricValue::from_f64(rtt_ns, false));
        results.insert(result.tag.clone(), result);
    }

    fn compute_sweep_curves(&self, mask: &[bool]) -> SweepLineCurves {
        let mask_values = |values: &[f64]| {
            values
                .iter()
                .zip(mask)
                .map(|(value, selected)| if *selected { *value } else { f64::NAN })
                .collect::<Vec<_>>()
        };
        let start = mask_values(self.store.start_ns());
        let generation_start = mask_values(self.store.generation_start_ns());
        let end = mask_values(self.store.end_ns());
        let numeric = |tag| {
            self.store.numeric_column(tag).map_or_else(
                || vec![f64::NAN; mask.len()],
                |column| mask_values(column.values()),
            )
        };
        let input = numeric(MetricTag::InputSequenceLength);
        let output = numeric(MetricTag::OutputSequenceLength);

        if let Some(replay) = self.store.inter_chunk_latency_replay()
            && !replay.values.is_empty()
        {
            let mut offsets = replay.offsets.to_vec();
            offsets.resize(mask.len(), 0);
            let icl = IclSeries::new(replay.values, replay.record_indices, &offsets);
            return SweepLineCurves::compute(
                &start,
                &generation_start,
                &end,
                &input,
                &output,
                Some(icl),
            );
        }
        SweepLineCurves::compute(&start, &generation_start, &end, &input, &output, None)
    }

    fn inject_sweep_results(
        &self,
        results: &mut BTreeMap<String, MetricResult>,
        curves: &SweepLineCurves,
        window_start_ns: Option<f64>,
        window_end_ns: Option<f64>,
    ) {
        if curves.concurrency.is_empty() {
            return;
        }
        let start = window_start_ns.unwrap_or(curves.concurrency.timestamps_ns()[0]);
        let end = window_end_ns
            .unwrap_or_else(|| *curves.concurrency.timestamps_ns().last().unwrap_or(&start));
        for sweep in curves.compute_metrics(start, end) {
            if let Some(result) = metric_result_from_sweep(sweep) {
                results.insert(result.tag.clone(), result);
            }
        }
    }

    fn compute_timeslices(
        &self,
        base_mask: &[bool],
        curves: &SweepLineCurves,
    ) -> Vec<MetricTimeslice> {
        let Some(slice_duration_ns) = self.config.slice_duration_ns.filter(|value| *value > 0)
        else {
            return Vec::new();
        };
        let selected = self
            .store
            .start_ns()
            .iter()
            .zip(self.store.end_ns())
            .zip(base_mask)
            .enumerate()
            .filter_map(|(row, ((start, end), selected))| {
                (*selected && !start.is_nan()).then_some((row, *start, *end))
            })
            .collect::<Vec<_>>();
        if selected.is_empty() {
            return Vec::new();
        }
        let min_start = selected
            .iter()
            .map(|(_, start, _)| *start)
            .min_by(f64::total_cmp)
            .unwrap();
        let max_start = selected
            .iter()
            .map(|(_, start, _)| *start)
            .max_by(f64::total_cmp)
            .unwrap();
        let max_end = selected
            .iter()
            .filter_map(|(_, _, end)| (!end.is_nan()).then_some(*end))
            .max_by(f64::total_cmp)
            .unwrap_or(max_start);
        let run_end = max_start.max(max_end);
        let slice_duration = slice_duration_ns as f64;
        let slice_count = ((run_end - min_start) / slice_duration) as usize + 1;
        let mut timeslices = Vec::new();
        for index in 0..slice_count {
            let start = min_start + index as f64 * slice_duration;
            let raw_end = start + slice_duration;
            let mut mask = vec![false; base_mask.len()];
            for (row, record_start, _) in &selected {
                if *record_start >= start && *record_start < raw_end {
                    mask[*row] = true;
                }
            }
            if !mask.iter().any(|selected| *selected) {
                continue;
            }
            let complete = raw_end <= run_end;
            let end = if complete { raw_end } else { run_end };
            let mut metrics = self.compute_result_map(&mask, Some(start as i64), Some(end as i64));
            self.inject_sweep_results(&mut metrics, curves, Some(start), Some(end));
            timeslices.push(MetricTimeslice {
                start_ns: start as i64,
                end_ns: end as i64,
                complete: (!complete).then_some(false),
                metrics,
            });
        }
        timeslices
    }
}

impl Accumulator<RecordIngest> for MetricsAccumulator {
    type Summary = AccumulatorSummary;
    type MergeError = MetricsMergeError;

    fn process_record(&mut self, record: &RecordIngest) {
        MetricsAccumulator::process_record(self, record);
    }

    fn query_time_range(&self, start_ns: i64, end_ns: i64) -> Vec<bool> {
        MetricsAccumulator::query_time_range(self, start_ns, end_ns)
    }

    fn export_results(&self, context: &ExportContext) -> Self::Summary {
        MetricsAccumulator::export_results(self, context)
    }

    fn merge(&mut self, other: &Self) -> Result<(), Self::MergeError> {
        MetricsAccumulator::merge(self, other)
    }
}

fn aggregate_values(values: &[f64], kind: AggregationKind) -> f64 {
    match kind {
        AggregationKind::Sum => values.iter().sum(),
        AggregationKind::Max => values.iter().copied().max_by(f64::total_cmp).unwrap_or(0.0),
        AggregationKind::Min => values.iter().copied().min_by(f64::total_cmp).unwrap_or(0.0),
    }
}

fn observation_duration(
    scalars: &FxHashMap<MetricTag, f64>,
    window_start_ns: Option<i64>,
    window_end_ns: Option<i64>,
) -> Option<f64> {
    let duration = match (window_start_ns, window_end_ns) {
        (Some(start), Some(end)) => (end - start) as f64,
        _ => scalars
            .get(&MetricTag::BenchmarkDuration)
            .copied()
            .or_else(|| {
                let start = scalars.get(&MetricTag::MinRequestTimestamp)?;
                let end = scalars.get(&MetricTag::MaxResponseTimestamp)?;
                Some(end - start)
            })?,
    };
    (duration > 0.0).then_some(duration)
}

fn derive_scalar(
    tag: MetricTag,
    values: &FxHashMap<MetricTag, f64>,
    observation_duration_ns: Option<f64>,
) -> Option<f64> {
    let get = |tag| values.get(&tag).copied();
    let rate = |numerator: f64| {
        let duration = observation_duration_ns?;
        (duration > 0.0).then_some(numerator * NANOS_PER_SECOND / duration)
    };
    match tag {
        MetricTag::CompletedRequestCount => {
            Some(get(MetricTag::RequestCount)? + get(MetricTag::ErrorRequestCount).unwrap_or(0.0))
        }
        MetricTag::RequestErrorRate => {
            let successes = get(MetricTag::RequestCount)?;
            let errors = get(MetricTag::ErrorRequestCount).unwrap_or(0.0);
            let total = successes + errors;
            (total > 0.0).then_some(100.0 * errors / total)
        }
        MetricTag::Goodput => rate(get(MetricTag::GoodRequestCount)?),
        MetricTag::GoodRequestFraction => {
            let good = get(MetricTag::GoodRequestCount)?;
            let attempted =
                get(MetricTag::RequestCount)? + get(MetricTag::ErrorRequestCount).unwrap_or(0.0);
            Some(if attempted == 0.0 {
                0.0
            } else {
                good / attempted
            })
        }
        MetricTag::BenchmarkDuration => {
            let min = get(MetricTag::MinRequestTimestamp)?;
            let max = get(MetricTag::MaxResponseTimestamp)?;
            (min < max).then_some(max - min)
        }
        MetricTag::TotalOutputSequenceLength => get(MetricTag::OutputSequenceLength),
        MetricTag::TotalInputSequenceLength => get(MetricTag::InputSequenceLength),
        MetricTag::TotalErrorInputSequenceLength => get(MetricTag::ErrorInputSequenceLength),
        MetricTag::TotalOutputTokens => get(MetricTag::OutputTokenCount),
        MetricTag::TotalReasoningTokens => get(MetricTag::ReasoningTokenCount),
        MetricTag::RequestThroughput => rate(get(MetricTag::RequestCount)?),
        MetricTag::InputTokenThroughput => rate(get(MetricTag::TotalInputSequenceLength)?),
        MetricTag::OutputTokenThroughput => rate(get(MetricTag::TotalOutputSequenceLength)?),
        MetricTag::TotalTokenThroughput => rate(
            get(MetricTag::TotalInputSequenceLength)? + get(MetricTag::TotalOutputSequenceLength)?,
        ),
        MetricTag::TotalUsagePromptTokens => get(MetricTag::UsagePromptTokens),
        MetricTag::TotalUsageCompletionTokens => get(MetricTag::UsageCompletionTokens),
        MetricTag::TotalUsageTotalTokens => get(MetricTag::UsageTotalTokens),
        MetricTag::TotalUsageReasoningTokens => get(MetricTag::UsageReasoningTokens),
        MetricTag::TotalUsagePromptCacheReadTokens => get(MetricTag::UsagePromptCacheReadTokens),
        MetricTag::TotalUsagePromptCacheWriteTokens => get(MetricTag::UsagePromptCacheWriteTokens),
        MetricTag::TotalUsagePromptCacheMissTokens => get(MetricTag::UsagePromptCacheMissTokens),
        MetricTag::TotalUsagePromptAudioTokens => get(MetricTag::UsagePromptAudioTokens),
        MetricTag::TotalUsageCompletionAudioTokens => get(MetricTag::UsageCompletionAudioTokens),
        MetricTag::TotalUsageAcceptedPredictionTokens => {
            get(MetricTag::UsageAcceptedPredictionTokens)
        }
        MetricTag::TotalUsageRejectedPredictionTokens => {
            get(MetricTag::UsageRejectedPredictionTokens)
        }
        MetricTag::TotalUsageToolUsePromptTokens => get(MetricTag::UsageToolUsePromptTokens),
        MetricTag::TotalUsagePromptAudioSeconds => get(MetricTag::UsagePromptAudioSeconds),
        MetricTag::OverallUsagePromptCacheReadPct => {
            let total = get(MetricTag::TotalUsagePromptTokens)?;
            (total != 0.0)
                .then_some(get(MetricTag::TotalUsagePromptCacheReadTokens)? / total * 100.0)
        }
        MetricTag::OverallThinkingEfficiency => {
            let total = get(MetricTag::TotalOutputTokens)?;
            (total != 0.0).then_some(get(MetricTag::TotalReasoningTokens)? / total)
        }
        _ => None,
    }
}

fn is_injected(tag: MetricTag) -> bool {
    matches!(
        tag,
        MetricTag::TotalGpuPower
            | MetricTag::TotalGpuEnergy
            | MetricTag::OutputTokensPerJoule
            | MetricTag::EnergyPerUser
            | MetricTag::NetworkAdjustedRequestLatency
            | MetricTag::NetworkAdjustedTimeToFirstToken
            | MetricTag::NetworkAdjustedTimeToFirstOutputToken
            | MetricTag::NetworkRtt
            | MetricTag::EffectiveConcurrency
            | MetricTag::EffectiveDecodeThroughput
            | MetricTag::EffectivePrefillThroughput
            | MetricTag::EffectiveDecodeConcurrency
            | MetricTag::EffectivePrefillConcurrency
            | MetricTag::EffectiveTotalThroughput
            | MetricTag::EffectiveDecodeThroughputPerUser
            | MetricTag::EffectivePrefillThroughputPerUser
            | MetricTag::TokensInFlight
            | MetricTag::ActiveDecodeThroughput
            | MetricTag::ActivePrefillThroughput
            | MetricTag::ActiveDecodeThroughputPerUser
            | MetricTag::ActivePrefillThroughputPerUser
            | MetricTag::ActiveTotalThroughput
    )
}

fn percent_difference(server: Option<f64>, client: Option<f64>) -> Option<f64> {
    let (server, client) = (server?, client?);
    (client > 0.0).then_some((server - client).abs() / client * 100.0)
}

fn convert_metric_value(value: MetricValue, source: Unit, target: Unit) -> MetricValue {
    match value {
        MetricValue::Finite(value) => source
            .convert_value(value, target)
            .map_or(MetricValue::Absent, |value| {
                MetricValue::from_f64(value, false)
            }),
        MetricValue::PosInf => MetricValue::PosInf,
        MetricValue::Absent => MetricValue::Absent,
    }
}

fn convert_distribution(
    mut stats: DistributionStats,
    source: Unit,
    target: Unit,
) -> DistributionStats {
    stats.avg = convert_metric_value(stats.avg, source, target);
    stats.min = convert_metric_value(stats.min, source, target);
    stats.max = convert_metric_value(stats.max, source, target);
    stats.sum = convert_metric_value(stats.sum, source, target);
    for value in stats.percentiles.values_mut() {
        *value = convert_metric_value(*value, source, target);
    }
    if let Some(std) = stats.std {
        let zero = source.convert_value(0.0, target).unwrap_or(0.0);
        let one = source.convert_value(1.0, target).unwrap_or(1.0);
        stats.std = MetricValue::from_f64(std * (one - zero).abs(), false).as_f64();
    }
    stats
}

fn metric_result_from_sweep(sweep: SweepMetricResult) -> Option<MetricResult> {
    let tag = sweep_tag(sweep.tag)?;
    let spec = spec_for(tag)?;
    let mut percentiles = BTreeMap::new();
    percentiles.insert(50, sweep.p50);
    percentiles.insert(90, sweep.p90);
    percentiles.insert(95, sweep.p95);
    percentiles.insert(99, sweep.p99);
    let stats = DistributionStats {
        tag: tag.as_str().to_string(),
        avg: sweep.avg,
        min: sweep.min,
        max: sweep.max,
        std: sweep.std,
        sum: MetricValue::Absent,
        count: 0,
        percentiles,
    };
    Some(MetricResult {
        tag: tag.as_str().to_string(),
        source_tag: Some(tag),
        header: sweep.header.to_string(),
        unit: sweep.unit.to_string(),
        console_group: spec.console_group,
        data: MetricResultData::Distribution(stats),
    })
}

fn sweep_tag(tag: &str) -> Option<MetricTag> {
    Some(match tag {
        "effective_concurrency" => MetricTag::EffectiveConcurrency,
        "effective_decode_throughput" => MetricTag::EffectiveDecodeThroughput,
        "effective_prefill_throughput" => MetricTag::EffectivePrefillThroughput,
        "effective_decode_concurrency" => MetricTag::EffectiveDecodeConcurrency,
        "effective_prefill_concurrency" => MetricTag::EffectivePrefillConcurrency,
        "effective_total_throughput" => MetricTag::EffectiveTotalThroughput,
        "effective_decode_throughput_per_user" => MetricTag::EffectiveDecodeThroughputPerUser,
        "effective_prefill_throughput_per_user" => MetricTag::EffectivePrefillThroughputPerUser,
        "tokens_in_flight" => MetricTag::TokensInFlight,
        "active_decode_throughput" => MetricTag::ActiveDecodeThroughput,
        "active_prefill_throughput" => MetricTag::ActivePrefillThroughput,
        "active_decode_throughput_per_user" => MetricTag::ActiveDecodeThroughputPerUser,
        "active_prefill_throughput_per_user" => MetricTag::ActivePrefillThroughputPerUser,
        "active_total_throughput" => MetricTag::ActiveTotalThroughput,
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ingest::{HttpTrace, TokenCounts, UsageMetrics};
    use crate::window::Phase;
    use std::collections::BTreeSet;

    fn successful_record(start_ns: i64, end_ns: i64) -> RecordIngest {
        let mut record = RecordIngest::minimal(start_ns, end_ns, Phase::Profiling);
        record.admit_ns = Some(start_ns - 5_000_000);
        record.first_token_ns = Some(start_ns + 20_000_000);
        record.second_token_ns = Some(start_ns + 30_000_000);
        record.first_output_token_ns = record.first_token_ns;
        record.token_arrival_ns = vec![
            start_ns + 20_000_000,
            start_ns + 30_000_000,
            start_ns + 40_000_000,
        ];
        record.tokens = TokenCounts {
            input: Some(100),
            output: Some(9),
            reasoning: Some(1),
            requested_output: Some(10),
        };
        record.usage = UsageMetrics {
            prompt_tokens: Some(100),
            completion_tokens: Some(10),
            total_tokens: Some(110),
            reasoning_tokens: Some(1),
            prompt_cache_read_tokens: Some(50),
            ..UsageMetrics::default()
        };
        record
    }

    #[test]
    fn summary_keeps_scalar_join_access_and_full_distributions() {
        let mut accumulator = MetricsAccumulator::new();
        accumulator.process_record(&successful_record(1_000_000_000, 1_100_000_000));
        let summary = accumulator.summarize();
        assert_eq!(summary.finite_value(MetricTag::RequestCount), Some(1.0));
        assert_eq!(
            summary.finite_value(MetricTag::RequestThroughput),
            Some(10.0),
            "whole-run rates must use the derived benchmark duration"
        );
        let latency = summary.result(MetricTag::RequestLatency).unwrap();
        assert_eq!(latency.unit, "ms");
        assert_eq!(
            latency.distribution().unwrap().avg,
            MetricValue::Finite(100.0)
        );
    }

    #[test]
    fn record_formulas_preserve_itl_minus_one_and_volume_totals() {
        let mut accumulator = MetricsAccumulator::new();
        accumulator.process_record(&successful_record(1_000_000_000, 1_100_000_000));
        let summary = accumulator.summarize();
        let itl_ms = summary
            .result(MetricTag::InterTokenLatency)
            .unwrap()
            .distribution()
            .unwrap()
            .avg
            .as_f64()
            .unwrap();
        assert!((itl_ms - 80.0 / 9.0).abs() < 1e-12);
        assert_eq!(
            summary.finite_value(MetricTag::TotalOutputSequenceLength),
            Some(10.0)
        );
        assert_eq!(
            summary.finite_value(MetricTag::OverallUsagePromptCacheReadPct),
            Some(50.0)
        );
    }

    #[test]
    fn explicit_window_overrides_rate_denominator() {
        let mut accumulator = MetricsAccumulator::new();
        accumulator.process_record(&successful_record(1_000_000_000, 1_100_000_000));
        let summary =
            accumulator.export_results(&ExportContext::time_range(1_000_000_000, 2_000_000_000));
        assert_eq!(
            summary.finite_value(MetricTag::RequestThroughput),
            Some(1.0)
        );
    }

    #[test]
    fn zero_error_run_does_not_emit_error_counter_or_adjusted_band() {
        let mut accumulator = MetricsAccumulator::new();
        accumulator.process_record(&successful_record(1_000_000_000, 1_100_000_000));
        let summary = accumulator.summarize();
        assert!(summary.result(MetricTag::ErrorRequestCount).is_none());
        assert!(summary.result_by_name("adj_request_latency").is_none());
    }

    #[test]
    fn one_error_in_ten_flips_adjusted_p95_to_infinity() {
        let mut accumulator = MetricsAccumulator::new();
        for index in 0..9 {
            accumulator.process_record(&successful_record(
                1_000_000_000 + index * 200_000_000,
                1_100_000_000 + index * 200_000_000,
            ));
        }
        let mut failed = RecordIngest::minimal(3_000_000_000, 3_000_000_001, Phase::Profiling);
        failed.errored = true;
        accumulator.process_record(&failed);
        let summary = accumulator.summarize();
        let adjusted = summary
            .result_by_name("adj_request_latency")
            .unwrap()
            .distribution()
            .unwrap();
        assert_eq!(adjusted.count, 10);
        assert_eq!(adjusted.std, None);
        assert_eq!(
            adjusted.percentiles.get(&90),
            Some(&MetricValue::Finite(100.0))
        );
        assert_eq!(adjusted.percentiles.get(&95), Some(&MetricValue::PosInf));
        assert_eq!(adjusted.percentiles.get(&99), Some(&MetricValue::PosInf));
    }

    #[test]
    fn phase_context_ignores_conflicting_time_bounds() {
        let mut accumulator = MetricsAccumulator::new();
        accumulator.process_record(&successful_record(100, 200));
        let context = ExportContext {
            start_ns: Some(500),
            end_ns: Some(600),
            phase: Some(Phase::Profiling),
        };
        assert_eq!(
            accumulator
                .export_results(&context)
                .finite_value(MetricTag::RequestCount),
            Some(1.0)
        );
    }

    #[test]
    fn timeslices_drop_empty_bins_and_clip_last_end() {
        let mut accumulator = MetricsAccumulator::with_config(MetricsConfig {
            slice_duration_ns: Some(1_000_000_000),
            ..MetricsConfig::default()
        });
        accumulator.process_record(&successful_record(500_000_000, 1_400_000_000));
        accumulator.process_record(&successful_record(1_500_000_000, 1_700_000_000));
        let summary = accumulator.summarize();
        assert_eq!(summary.timeslices().len(), 2);
        assert_eq!(summary.timeslices()[1].end_ns, 1_700_000_000);
        assert_eq!(summary.timeslices()[1].complete, Some(false));
    }

    #[test]
    fn configured_slo_controls_goodput_direction() {
        let mut accumulator = MetricsAccumulator::with_config(MetricsConfig {
            slos: vec![SloThreshold::native(
                MetricTag::RequestLatency,
                150_000_000.0,
            )],
            ..MetricsConfig::default()
        });
        accumulator.process_record(&successful_record(1_000_000_000, 1_100_000_000));
        accumulator.process_record(&successful_record(1_200_000_000, 1_400_000_000));
        let summary = accumulator.summarize();
        assert_eq!(summary.finite_value(MetricTag::GoodRequestCount), Some(1.0));
        assert_eq!(
            summary.finite_value(MetricTag::GoodRequestFraction),
            Some(0.5)
        );
    }

    #[test]
    fn network_adjustment_clamps_at_zero_and_preserves_raw_metric() {
        let mut accumulator = MetricsAccumulator::new();
        accumulator.process_record(&successful_record(1_000_000_000, 1_100_000_000));
        accumulator.set_network_rtt_ns(Some(120_000_000.0));
        let summary = accumulator.summarize();
        assert_eq!(
            summary
                .result(MetricTag::NetworkAdjustedRequestLatency)
                .unwrap()
                .distribution()
                .unwrap()
                .avg,
            MetricValue::Finite(0.0)
        );
        assert_eq!(
            summary
                .result(MetricTag::RequestLatency)
                .unwrap()
                .distribution()
                .unwrap()
                .avg,
            MetricValue::Finite(100.0)
        );
    }

    #[test]
    fn http_composites_treat_missing_phases_as_zero_when_trace_data_exists() {
        let mut record = successful_record(1_000_000_000, 1_100_000_000);
        record.http = HttpTrace {
            blocked_ns: Some(10),
            connecting_ns: Some(20),
            waiting_ns: Some(30),
            ..HttpTrace::default()
        };
        let mut accumulator = MetricsAccumulator::new();
        accumulator.process_record(&record);
        let summary = accumulator.summarize();
        assert_eq!(
            summary
                .result(MetricTag::HttpReqConnectionOverhead)
                .unwrap()
                .distribution()
                .unwrap()
                .avg,
            MetricValue::Finite(0.000_03)
        );
        assert_eq!(
            summary
                .result(MetricTag::HttpReqTotal)
                .unwrap()
                .distribution()
                .unwrap()
                .avg,
            MetricValue::Finite(0.000_06)
        );
    }

    #[test]
    fn reversed_export_window_omits_rate_metrics() {
        let mut accumulator = MetricsAccumulator::new();
        accumulator.process_record(&successful_record(1_000_000_000, 1_100_000_000));
        let context = ExportContext {
            start_ns: Some(2_000_000_000),
            end_ns: Some(1_000_000_000),
            phase: Some(Phase::Profiling),
        };
        let summary = accumulator.export_results(&context);
        assert!(summary.result(MetricTag::RequestThroughput).is_none());
    }

    #[test]
    fn per_worker_merge_matches_single_accumulator_ingest_order() {
        let first = successful_record(1_000_000_000, 1_100_000_000);
        let mut second = successful_record(1_200_000_000, 1_350_000_000);
        second.worker_id = Some("worker-1".to_string());
        second.turn_index = 1;

        let mut direct = MetricsAccumulator::new();
        direct.process_record(&first);
        direct.process_record(&second);

        let mut left = MetricsAccumulator::new();
        left.process_record(&first);
        let mut right = MetricsAccumulator::new();
        right.process_record(&second);
        left.merge(&right).unwrap();

        assert_eq!(left.summarize(), direct.summarize());
        assert_eq!(
            left.column_store().mask_for_worker("worker-1"),
            vec![false, true]
        );
    }

    #[test]
    fn every_catalog_metric_is_either_computed_injected_or_data_dependent() {
        let derived = CATALOG
            .iter()
            .filter(|spec| spec.kind == MetricType::Derived)
            .map(|spec| spec.tag)
            .collect::<BTreeSet<_>>();
        let handled = derived
            .iter()
            .copied()
            .filter(|tag| {
                is_injected(*tag)
                    || derive_scalar(*tag, &FxHashMap::default(), Some(1.0)).is_some()
                    || matches!(
                        tag,
                        MetricTag::CompletedRequestCount
                            | MetricTag::RequestErrorRate
                            | MetricTag::Goodput
                            | MetricTag::GoodRequestFraction
                            | MetricTag::BenchmarkDuration
                            | MetricTag::TotalOutputSequenceLength
                            | MetricTag::TotalInputSequenceLength
                            | MetricTag::TotalErrorInputSequenceLength
                            | MetricTag::TotalOutputTokens
                            | MetricTag::TotalReasoningTokens
                            | MetricTag::RequestThroughput
                            | MetricTag::InputTokenThroughput
                            | MetricTag::OutputTokenThroughput
                            | MetricTag::TotalTokenThroughput
                            | MetricTag::TotalUsagePromptTokens
                            | MetricTag::TotalUsageCompletionTokens
                            | MetricTag::TotalUsageTotalTokens
                            | MetricTag::TotalUsageReasoningTokens
                            | MetricTag::TotalUsagePromptCacheReadTokens
                            | MetricTag::TotalUsagePromptCacheWriteTokens
                            | MetricTag::TotalUsagePromptCacheMissTokens
                            | MetricTag::TotalUsagePromptAudioTokens
                            | MetricTag::TotalUsageCompletionAudioTokens
                            | MetricTag::TotalUsageAcceptedPredictionTokens
                            | MetricTag::TotalUsageRejectedPredictionTokens
                            | MetricTag::TotalUsageToolUsePromptTokens
                            | MetricTag::TotalUsagePromptAudioSeconds
                            | MetricTag::OverallUsagePromptCacheReadPct
                            | MetricTag::OverallThinkingEfficiency
                    )
            })
            .collect::<BTreeSet<_>>();
        assert_eq!(derived, handled);
    }
}
