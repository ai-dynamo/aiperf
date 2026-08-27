// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Columnar inference-metric accumulation, derivation, windowing, and timeslicing.
//!
//! The dispatch order, authoritative phase masks, sweep injection, and half-open
//! timeslicing are implemented in this module; the Rust runtime supplies the
//! transport-neutral [`RecordIngest`] facts consumed here.

use crate::metrics_core::catalog::{
    AggregationKind, CATALOG, MetricConsoleGroup, MetricFlags, MetricSpec, MetricTag, MetricType,
    spec_for, validate_catalog,
};
use crate::metrics_core::definition::{Definition, Native, metric_definition};
use crate::metrics_core::ingest::{InferenceDimensions, RecordIngest};
use crate::metrics_core::itl::decode_tokens_after_first_chunk;
use crate::metrics_core::kernel::{DistributionStats, linear_distribution, nearest_distribution};
use crate::metrics_core::sidecar::SidecarMetric;
use crate::metrics_core::store::{ColumnStore, ListMetricBackend, MetricsStorageMode};
use crate::metrics_core::sweepline::{IclSeries, StepFn, SweepLineCurves, SweepMetricResult};
use crate::metrics_core::units::{Unit, UnitConversionError};
use crate::metrics_core::value::MetricValue;
use crate::metrics_core::window::{ExportContext, Phase};
use rustc_hash::FxHashMap;
use serde::Serialize;
use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::{Display, Formatter, Result as FmtResult};
use std::sync::LazyLock;

const NANOS_PER_SECOND: f64 = 1_000_000_000.0;
const DEFAULT_USAGE_DIFF_THRESHOLD_PCT: f64 = 10.0;
const DEFAULT_OSL_MISMATCH_THRESHOLD_PCT: f64 = 5.0;
const DEFAULT_OSL_MISMATCH_MAX_TOKENS: f64 = 50.0;
const PARALLEL_SUMMARY_MIN_ROWS: usize = 4_096;

/// Derived-metric topological order over the immutable catalog.
///
/// `validate_catalog` builds a petgraph `DiGraphMap` and toposorts it. The
/// catalog is static, so that graph construction runs once here rather than on
/// every `compute_result_map` call — which fans out per export, per
/// inference-series dimension, and per timeslice. The `expect` also fail-fasts
/// on a malformed catalog at first accumulator construction, preserving the
/// prior `with_config` validation contract.
static DERIVED_TOPO_ORDER: LazyLock<Vec<MetricTag>> =
    LazyLock::new(|| validate_catalog().expect("the static metric catalog must be valid"));

/// Extension seam for request-index-addressed record accumulation and windowed export.
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
    /// Whether the metric passes when the value is `>=` the threshold (larger is
    /// better) versus `<=` it. Resolved from the static catalog at construction so
    /// the per-record good-request path never re-scans it.
    pub larger_is_better: bool,
    /// The metric's static definition, captured once at construction (config-time)
    /// so the per-record path can route through the shared
    /// [`Definition::passes_threshold`] direction logic without a registry lookup.
    pub definition: &'static Definition,
}

impl SloThreshold {
    /// Builds a threshold already expressed in native units.
    pub fn native(tag: MetricTag, native_value: f64) -> Self {
        Self {
            tag,
            native_value,
            larger_is_better: tag_is_larger_is_better(tag),
            definition: metric_definition(tag),
        }
    }

    /// Converts a display-unit threshold into native units using the catalog.
    pub fn from_display(tag: MetricTag, display_value: f64) -> Result<Self, UnitConversionError> {
        let spec = spec_for(tag).expect("SLO tags must resolve in the static catalog");
        let display_unit = spec.display_unit().unwrap_or(spec.unit());
        Ok(Self {
            tag,
            native_value: display_unit.convert_value(display_value, spec.unit())?,
            larger_is_better: spec.flags.contains(MetricFlags::LARGER_IS_BETTER),
            definition: metric_definition(tag),
        })
    }

    /// Whether a per-record native `value` satisfies this threshold. Routes through
    /// the shared [`Definition::passes_threshold`] logic with typed [`Native`] units,
    /// using the definition captured at construction (no per-record registry lookup).
    #[inline]
    pub fn passes(&self, value: f64) -> bool {
        self.definition
            .passes_threshold(Native::new(value), Native::new(self.native_value))
    }
}

/// Whether a metric's catalog spec marks it larger-is-better (absent spec defaults false).
fn tag_is_larger_is_better(tag: MetricTag) -> bool {
    spec_for(tag).is_some_and(|spec| spec.flags.contains(MetricFlags::LARGER_IS_BETTER))
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
    /// Source visible token accounting from the endpoint's `usage` fields
    /// instead of client-side tokenization. When enabled, `TokenCounts.input`
    /// is `usage.prompt_tokens` and `output`/`reasoning` come from
    /// `usage.completion_tokens`/`usage.reasoning_tokens`; otherwise all three
    /// are the client-tokenized counts. `metrics.rs` applies that per-mode
    /// choice, so the accumulator only ever sees the resolved `TokenCounts`.
    pub use_server_token_count: bool,
    /// Per-record retention mode. [`MetricsStorageMode::Sketch`] streams each value
    /// into a bounded-memory t-digest instead of retaining it, trading exact
    /// percentiles for O(1) memory. Off by default.
    pub storage_mode: MetricsStorageMode,
    /// Closed-loop steady-state windowing for concurrency-target runs. Disabled
    /// by default; when enabled with a positive concurrency target the metrics
    /// plane also emits a steady-state summary over the auto-detected window.
    pub steady_state: crate::metrics_core::steady_state::SteadyStateConfig,
}

impl Default for MetricsConfig {
    fn default() -> Self {
        Self {
            slice_duration_ns: None,
            slos: Vec::new(),
            usage_diff_threshold_pct: DEFAULT_USAGE_DIFF_THRESHOLD_PCT,
            osl_mismatch_threshold_pct: DEFAULT_OSL_MISMATCH_THRESHOLD_PCT,
            osl_mismatch_max_tokens: DEFAULT_OSL_MISMATCH_MAX_TOKENS,
            use_server_token_count: false,
            storage_mode: MetricsStorageMode::Exact,
            steady_state: crate::metrics_core::steady_state::SteadyStateConfig::default(),
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
        let display_unit = spec.display_unit().unwrap_or(spec.unit());
        Self {
            tag: spec.tag.as_str().to_string(),
            source_tag: Some(spec.tag),
            header: spec.header().to_string(),
            unit: display_unit.as_str().to_string(),
            console_group: spec.console_group,
            data: MetricResultData::Scalar {
                value: convert_metric_value(value, spec.unit(), display_unit),
            },
        }
    }

    fn distribution_from_spec(spec: &MetricSpec, stats: DistributionStats) -> Self {
        let display_unit = spec.display_unit().unwrap_or(spec.unit());
        Self {
            tag: spec.tag.as_str().to_string(),
            source_tag: Some(spec.tag),
            header: spec.header().to_string(),
            unit: display_unit.as_str().to_string(),
            console_group: spec.console_group,
            data: MetricResultData::Distribution(convert_distribution(
                stats,
                spec.unit(),
                display_unit,
            )),
        }
    }

    fn adjusted_from_parent(spec: &MetricSpec, stats: DistributionStats) -> Self {
        let display_unit = spec.display_unit().unwrap_or(spec.unit());
        Self {
            tag: format!("adj_{}", spec.tag.as_str()),
            source_tag: Some(spec.tag),
            header: format!("{} (error-adjusted)", spec.header()),
            unit: display_unit.as_str().to_string(),
            console_group: spec.console_group,
            data: MetricResultData::Distribution(convert_distribution(
                stats,
                spec.unit(),
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

/// One model/endpoint-specific inference metric series.
///
/// Series retain the same result and timeslice types as the aggregate summary;
/// only the row mask differs. Dimensions are value-sorted before construction,
/// so worker merge order cannot perturb report ordering.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct InferenceMetricSeriesSummary {
    dimensions: InferenceDimensions,
    results: BTreeMap<String, MetricResult>,
    timeslices: Vec<MetricTimeslice>,
}

impl InferenceMetricSeriesSummary {
    /// Returns the exact model/endpoint pair for this series.
    pub fn dimensions(&self) -> &InferenceDimensions {
        &self.dimensions
    }

    /// Returns one result by stable metric name.
    pub fn result_by_name(&self, tag: &str) -> Option<&MetricResult> {
        self.results.get(tag)
    }

    /// Returns all results as a stable ordered map.
    pub fn result_map(&self) -> &BTreeMap<String, MetricResult> {
        &self.results
    }

    /// Returns chronological non-empty timeslices for this dimension pair.
    pub fn timeslices(&self) -> &[MetricTimeslice] {
        &self.timeslices
    }
}

/// Full summary produced by [`MetricsAccumulator`].
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct AccumulatorSummary {
    results: BTreeMap<String, MetricResult>,
    timeslices: Vec<MetricTimeslice>,
    inference_series: Vec<InferenceMetricSeriesSummary>,
    sidecar_metrics: BTreeMap<String, SidecarMetric>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pooled_spec_decode_acceptance_histogram: Option<BTreeMap<u64, u128>>,
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

    /// Returns inference series sorted by endpoint URL and then model.
    pub fn inference_series(&self) -> &[InferenceMetricSeriesSummary] {
        &self.inference_series
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

    /// Returns the exact selected speculative-decoding acceptance histogram.
    pub fn pooled_spec_decode_acceptance_histogram(&self) -> Option<&BTreeMap<u64, u128>> {
        self.pooled_spec_decode_acceptance_histogram.as_ref()
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
        // Forces one-time catalog validation; the topo order is cached for reuse.
        LazyLock::force(&DERIVED_TOPO_ORDER);
        Self {
            store: ColumnStore::with_storage_mode(config.storage_mode),
            config,
            network_rtt_ns: None,
            injected_scalars: FxHashMap::default(),
        }
    }

    /// Builds an accumulator that owns a pre-populated column store.
    ///
    /// The cellular controller uses this at export to wrap the merge of every
    /// cell's [`ColumnStorePartition`](crate::cellular::shard::ColumnStorePartition)
    /// into one accumulator for summarization, without replaying records. Run-level
    /// scalars (network RTT, injected side-channel scalars) are applied afterward
    /// via [`set_network_rtt_ns`](Self::set_network_rtt_ns) /
    /// [`inject_scalar`](Self::inject_scalar).
    pub fn from_column_store(config: MetricsConfig, store: ColumnStore) -> Self {
        LazyLock::force(&DERIVED_TOPO_ORDER);
        Self {
            store,
            config,
            network_rtt_ns: None,
            injected_scalars: FxHashMap::default(),
        }
    }

    /// Returns the underlying read-only column store for analyzers.
    pub fn column_store(&self) -> &ColumnStore {
        &self.store
    }

    /// Builds the in-flight request-concurrency step function over `mask`.
    ///
    /// Reuses the shared sweep-line curve bundle so steady-state detection reads
    /// the same concurrency-over-time signal every other windowed summary is
    /// derived from, rather than re-deriving it from raw start/end timestamps.
    pub(crate) fn concurrency_curve(&self, mask: &[bool]) -> StepFn {
        self.compute_sweep_curves(mask).concurrency
    }

    /// Returns the number of populated request slots.
    pub fn record_count(&self) -> usize {
        self.store.record_count()
    }

    /// Total records ever ingested (monotonic; survives sketch mode's fold-and-clear,
    /// is summed on [`ColumnStore::append_store`], and travels with the shipped store).
    /// Use this — not [`Self::record_count`] — for a cellular cell's ship counters and
    /// the merged sketch outcome, since a sketch store retains no rows and reports
    /// `record_count() == 0`.
    pub fn ingested_count(&self) -> u64 {
        self.store.ingested_count()
    }

    /// Prepares absolute request slots without marking any slot occupied.
    pub fn prepare_request_slots(&mut self, rows: usize) {
        self.store.prepare_request_slots(rows);
    }

    /// Ingests one record and computes all record/aggregate inputs available from it.
    pub fn process_record(&mut self, record: &RecordIngest) {
        self.process_record_with_token_arrivals(record, &record.token_arrival_ns);
    }

    /// Ingests one record while borrowing token arrivals from producer-owned storage.
    pub fn process_record_with_token_arrivals(
        &mut self,
        record: &RecordIngest,
        token_arrivals_ns: &[i64],
    ) {
        // Sketch mode ignores the authored absolute request index: it processes each
        // record into a fresh row-0 scratch, harvests it into the phase-keyed
        // sketch, then clears the row so memory stays O(1) in the record count.
        let sketch_mode = self.store.sketch().is_some();
        let row = if sketch_mode {
            self.store
                .push_record_with_token_arrivals(record, token_arrivals_ns)
        } else {
            match record.request_index {
                Some(row) => {
                    self.store
                        .insert_record_at_with_token_arrivals(row, record, token_arrivals_ns);
                    row
                }
                None => self
                    .store
                    .push_record_with_token_arrivals(record, token_arrivals_ns),
            }
        };
        if !record.errored && !record.canceled {
            self.compute_record_metrics(row, record.tokens.first_content_chunk_tokens);
            self.compute_good_request(row);
        }
        if sketch_mode {
            self.store
                .harvest_row_to_sketch(row, record.phase, record.phase_index);
            self.store.clear_rows();
        }
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
        if self.store.sketch().is_some() {
            return self.export_results_sketch(context);
        }
        let mask = self.store.mask_for(context);
        if !mask.iter().any(|selected| *selected) {
            return AccumulatorSummary::new();
        }
        let (mut results, curves) =
            self.compute_results_and_curves(&mask, context.start_ns, context.end_ns);
        self.inject_sweep_results(
            &mut results,
            &curves,
            context.start_ns.map(|value| value as f64),
            context.end_ns.map(|value| value as f64),
        );
        let timeslices = self.compute_timeslices(&mask, &curves);
        let inference_series = self.compute_inference_series(&mask, context, &results, &timeslices);
        AccumulatorSummary {
            results,
            timeslices,
            inference_series,
            sidecar_metrics: BTreeMap::new(),
            pooled_spec_decode_acceptance_histogram: self
                .store
                .pooled_spec_decode_acceptance_histogram(context),
        }
    }

    /// Summarizes a context from the bounded-memory sketch columns.
    ///
    /// Only phase separation survives a merged sketch, so per-row-only outputs are
    /// dropped: no timeslices (they need per-record timestamps), no inference series
    /// (they need per-record model/endpoint partitioning), no sweep curves, no
    /// error-adjusted (`adj_*`) bands, and no per-record network-adjusted
    /// distributions. Counts, sums, averages, min/max, and rate derivations stay
    /// exact; percentiles are t-digest approximations.
    fn export_results_sketch(&self, context: &ExportContext) -> AccumulatorSummary {
        let results = self.compute_result_map_sketch(context.phase, context.phase_index);
        AccumulatorSummary {
            results,
            timeslices: Vec::new(),
            inference_series: Vec::new(),
            sidecar_metrics: BTreeMap::new(),
            pooled_spec_decode_acceptance_histogram: self
                .store
                .pooled_spec_decode_acceptance_histogram(context),
        }
    }

    fn compute_result_map_sketch(
        &self,
        phase: Option<Phase>,
        phase_index: Option<usize>,
    ) -> BTreeMap<String, MetricResult> {
        let sketch = self
            .store
            .sketch()
            .expect("compute_result_map_sketch requires sketch storage");
        let mut scalars = FxHashMap::<MetricTag, f64>::default();
        let mut results = BTreeMap::new();

        for spec in CATALOG
            .iter()
            .filter(|spec| spec.kind != MetricType::Derived)
        {
            let Some(tag_sketch) = sketch.resolve(phase, phase_index, spec.tag) else {
                continue;
            };
            if tag_sketch.count() == 0 {
                continue;
            }
            match spec.kind {
                MetricType::Record => {
                    scalars.insert(spec.tag, tag_sketch.sum());
                    if let Some(stats) =
                        DistributionStats::from_sketch(spec.tag.as_str(), &tag_sketch, 0)
                    {
                        let result = MetricResult::distribution_from_spec(spec, stats);
                        results.insert(result.tag.clone(), result);
                    }
                }
                MetricType::Aggregate => {
                    let scalar = aggregate_from_sketch(
                        &tag_sketch,
                        spec.aggregation.unwrap_or(AggregationKind::Sum),
                    );
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

        // Phase contexts carry no window bounds, so rate derivations use the exact
        // benchmark duration derived from the min/max timestamp aggregates.
        let observation_duration_ns = observation_duration(&scalars, None, None);
        for &tag in DERIVED_TOPO_ORDER.iter() {
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

        // The per-record network-adjusted distributions cannot be reconstructed
        // from a merged sketch, but the run-level RTT scalar still stands.
        if let Some(rtt_ns) = self.network_rtt_ns {
            let result =
                MetricResult::scalar(MetricTag::NetworkRtt, MetricValue::from_f64(rtt_ns, false));
            results.insert(result.tag.clone(), result);
        }
        results
    }

    fn compute_results_and_curves(
        &self,
        mask: &[bool],
        window_start_ns: Option<i64>,
        window_end_ns: Option<i64>,
    ) -> (BTreeMap<String, MetricResult>, SweepLineCurves) {
        let use_parallel_reduction = self.store.record_count() >= PARALLEL_SUMMARY_MIN_ROWS
            && rayon::current_num_threads() > 1;
        if !use_parallel_reduction {
            return (
                self.compute_result_map(mask, window_start_ns, window_end_ns),
                self.compute_sweep_curves(mask),
            );
        }

        // Both branches read the frozen column store. Joining before any result
        // injection preserves the same deterministic reduction and map order.
        rayon::join(
            || self.compute_result_map(mask, window_start_ns, window_end_ns),
            || self.compute_sweep_curves(mask),
        )
    }

    fn compute_inference_series(
        &self,
        base_mask: &[bool],
        context: &ExportContext,
        aggregate_results: &BTreeMap<String, MetricResult>,
        aggregate_timeslices: &[MetricTimeslice],
    ) -> Vec<InferenceMetricSeriesSummary> {
        // Categorical dimensions stay separate from numeric metrics so grouped
        // analysis can mask the selected model/endpoint pair exactly.
        let dimensions = self
            .store
            .inference_dimensions()
            .iter()
            .filter(|dimensions| {
                self.store
                    .mask_for_inference_dimensions(dimensions)
                    .iter()
                    .zip(base_mask)
                    .any(|(dimension, base)| *dimension && *base)
            })
            .cloned()
            .collect::<BTreeSet<_>>();

        if dimensions.len() == 1 {
            let dimensions = dimensions
                .iter()
                .next()
                .expect("one inference dimension was counted")
                .clone();
            let dimension_mask = self.store.mask_for_inference_dimensions(&dimensions);
            if base_mask
                .iter()
                .zip(&dimension_mask)
                .all(|(base, dimension)| !*base || *dimension)
            {
                let mut results = aggregate_results.clone();
                self.remove_unpartitioned_results(&mut results);
                let mut timeslices = aggregate_timeslices.to_vec();
                for timeslice in &mut timeslices {
                    self.remove_unpartitioned_results(&mut timeslice.metrics);
                }
                return (!results.is_empty())
                    .then_some(InferenceMetricSeriesSummary {
                        dimensions,
                        results,
                        timeslices,
                    })
                    .into_iter()
                    .collect();
            }
        }

        dimensions
            .into_iter()
            .filter_map(|dimensions| {
                let dimension_mask = self.store.mask_for_inference_dimensions(&dimensions);
                let mask = base_mask
                    .iter()
                    .zip(dimension_mask)
                    .map(|(base, dimension)| *base && dimension)
                    .collect::<Vec<_>>();
                let mut results = self.compute_result_map(&mask, context.start_ns, context.end_ns);
                self.remove_unpartitioned_results(&mut results);
                let curves = self.compute_sweep_curves(&mask);
                self.inject_sweep_results(
                    &mut results,
                    &curves,
                    context.start_ns.map(|value| value as f64),
                    context.end_ns.map(|value| value as f64),
                );
                let mut timeslices = self.compute_timeslices(&mask, &curves);
                for timeslice in &mut timeslices {
                    self.remove_unpartitioned_results(&mut timeslice.metrics);
                }
                (!results.is_empty()).then_some(InferenceMetricSeriesSummary {
                    dimensions,
                    results,
                    timeslices,
                })
            })
            .collect()
    }

    fn remove_unpartitioned_results(&self, results: &mut BTreeMap<String, MetricResult>) {
        // Externally injected scalars describe the run/phase as a whole. Copying
        // total GPU energy, for example, into every endpoint/model series would
        // imply a partition the telemetry producer never measured.
        for tag in self.injected_scalars.keys() {
            results.remove(tag.as_str());
        }
        results.remove(MetricTag::NetworkRtt.as_str());
    }

    fn compute_record_metrics(&mut self, row: usize, first_content_chunk_tokens: Option<u64>) {
        let latency = self.store.metric_f64(row, MetricTag::RequestLatency);
        let ttft = self.store.metric_f64(row, MetricTag::TimeToFirstToken);
        let osl = self.store.metric_f64(row, MetricTag::OutputSequenceLength);
        let isl = self.store.metric_f64(row, MetricTag::InputSequenceLength);

        if let (Some(latency), Some(ttft), Some(osl)) = (latency, ttft, osl)
            && let Some(decode_tokens) =
                decode_tokens_after_first_chunk(osl as u64, first_content_chunk_tokens)
        {
            self.set_finite_record(
                row,
                MetricTag::InterTokenLatency,
                (latency - ttft) / decode_tokens as f64,
            );
        }
        // Client-observed interval from the first to final content response.
        // `ttft` is only populated for streaming records, so this naturally
        // limits to streaming (matching the STREAMING_TOKENS_ONLY catalog flag).
        if let (Some(latency), Some(ttft)) = (latency, ttft) {
            self.set_finite_record(row, MetricTag::DecodeDuration, latency - ttft);
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
        // The client-vs-server diff metrics only mean something when the client
        // does its own tokenization. Under `use_server_token_count` the server
        // count is authoritative and overwrites the local input count, so every
        // difference collapses to a meaningless zero, so the tags must be
        // absent rather than zero.
        if self.config.use_server_token_count {
            return;
        }
        let prompt = percent_difference(
            self.store.metric_f64(row, MetricTag::UsagePromptTokens),
            self.store.metric_f64(row, MetricTag::InputSequenceLength),
        );
        let completion = percent_difference(
            self.store.metric_f64(row, MetricTag::UsageCompletionTokens),
            self.store.observed_output_sequence_length(row),
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
            // Direction logic is the shared `Definition::passes_threshold` over typed
            // `Native` units; `slo.definition` was captured at construction, so this
            // per-record path performs no registry lookup.
            slo.passes(value)
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
        let mut scalars = FxHashMap::<MetricTag, f64>::default();
        let mut record_arrays = FxHashMap::<MetricTag, (Vec<f64>, f64)>::default();
        let mut results = BTreeMap::new();
        let error_count = mask
            .iter()
            .zip(self.store.errored())
            .filter(|(selected, errored)| **selected && **errored)
            .count();

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
                if error_count > 0
                    && spec
                        .flags
                        .contains(MetricFlags::PERCENTILE_INCLUDES_FAILED_REQUESTS)
                {
                    record_arrays.insert(spec.tag, (values.clone(), sum));
                }
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
                    // `masked_values` already traverses the absolute request slots
                    // monotonically, so this is both canonical and cheaper than
                    // rescanning the sparse source column for a cached fast path.
                    let sum = values.iter().sum();
                    scalars.insert(spec.tag, sum);
                    if error_count > 0
                        && spec
                            .flags
                            .contains(MetricFlags::PERCENTILE_INCLUDES_FAILED_REQUESTS)
                    {
                        record_arrays.insert(spec.tag, (values.clone(), sum));
                    }
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
        for &tag in DERIVED_TOPO_ORDER.iter() {
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
        let all_selected = mask.iter().all(|selected| *selected);
        let start = masked_values(self.store.start_ns(), mask, all_selected);
        let generation_start = masked_values(self.store.generation_start_ns(), mask, all_selected);
        let end = masked_values(self.store.end_ns(), mask, all_selected);
        let numeric = |tag| {
            self.store.numeric_column(tag).map_or_else(
                || Cow::Owned(vec![f64::NAN; mask.len()]),
                |column| masked_values(column.values(), mask, all_selected),
            )
        };
        let input = numeric(MetricTag::InputSequenceLength);
        let output = numeric(MetricTag::OutputSequenceLength);
        let num_images = numeric(MetricTag::NumImages);

        if let Some(replay) = self.store.inter_chunk_latency_replay()
            && !replay.values.is_empty()
        {
            let mut offsets = replay.offsets.to_vec();
            offsets.resize(mask.len(), 0);
            let mut lengths = replay.lengths.to_vec();
            lengths.resize(mask.len(), 0);
            let icl = IclSeries::new(replay.values, &offsets, &lengths, replay.append_order);
            return SweepLineCurves::compute(
                start.as_ref(),
                generation_start.as_ref(),
                end.as_ref(),
                input.as_ref(),
                output.as_ref(),
                num_images.as_ref(),
                Some(icl),
            );
        }
        SweepLineCurves::compute(
            start.as_ref(),
            generation_start.as_ref(),
            end.as_ref(),
            input.as_ref(),
            output.as_ref(),
            num_images.as_ref(),
            None,
        )
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

fn masked_values<'a>(values: &'a [f64], mask: &[bool], all_selected: bool) -> Cow<'a, [f64]> {
    assert_eq!(values.len(), mask.len());
    if all_selected {
        return Cow::Borrowed(values);
    }

    let mut output = Vec::with_capacity(values.len());
    let mut value_chunks = values.chunks_exact(8);
    let mut mask_chunks = mask.chunks_exact(8);
    for (values, mask) in value_chunks.by_ref().zip(mask_chunks.by_ref()) {
        output.extend_from_slice(&[
            masked_value(values[0], mask[0]),
            masked_value(values[1], mask[1]),
            masked_value(values[2], mask[2]),
            masked_value(values[3], mask[3]),
            masked_value(values[4], mask[4]),
            masked_value(values[5], mask[5]),
            masked_value(values[6], mask[6]),
            masked_value(values[7], mask[7]),
        ]);
    }
    output.extend(
        value_chunks
            .remainder()
            .iter()
            .zip(mask_chunks.remainder())
            .map(|(&value, &selected)| masked_value(value, selected)),
    );
    Cow::Owned(output)
}

#[inline(always)]
fn masked_value(value: f64, selected: bool) -> f64 {
    let selected_bits = 0_u64.wrapping_sub(u64::from(selected));
    f64::from_bits((value.to_bits() & selected_bits) | (f64::NAN.to_bits() & !selected_bits))
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

/// Reduces one tag's sketch to its aggregate scalar, matching [`aggregate_values`]
/// but reading the sketch's exact running sum/min/max instead of a value vector.
fn aggregate_from_sketch(
    sketch: &crate::metrics_core::store::TagSketch,
    kind: AggregationKind,
) -> f64 {
    match kind {
        AggregationKind::Sum => sketch.sum(),
        AggregationKind::Max => sketch.max(),
        AggregationKind::Min => sketch.min(),
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
        MetricTag::TotalNumImages => get(MetricTag::NumImages),
        MetricTag::ImageSamplesPerSecond => rate(get(MetricTag::TotalNumImages)?),
        MetricTag::TotalOutputTokens => get(MetricTag::OutputTokenCount),
        MetricTag::TotalReasoningTokens => get(MetricTag::ReasoningTokenCount),
        MetricTag::TotalSpecDecodeSteps => get(MetricTag::SpecDecodeSteps),
        MetricTag::TotalAcceptedDraftTokens => get(MetricTag::SpecDecodeAcceptedDraftTokens),
        MetricTag::TotalDraftTokens => get(MetricTag::SpecDecodeDraftTokens),
        MetricTag::SpecDecodeTokenWeightedAcceptanceLength => {
            let steps = get(MetricTag::TotalSpecDecodeSteps)?;
            (steps > 0.0).then_some(1.0 + get(MetricTag::TotalAcceptedDraftTokens)? / steps)
        }
        MetricTag::SpecDecodeOverallDraftAcceptanceRate => {
            let draft = get(MetricTag::TotalDraftTokens)?;
            (draft > 0.0).then_some(100.0 * get(MetricTag::TotalAcceptedDraftTokens)? / draft)
        }
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
            | MetricTag::EffectiveImageSamplesPerSecond
            | MetricTag::TokensInFlight
            | MetricTag::ActiveDecodeThroughput
            | MetricTag::ActivePrefillThroughput
            | MetricTag::ActiveDecodeThroughputPerUser
            | MetricTag::ActivePrefillThroughputPerUser
            | MetricTag::ActiveImageSamplesPerSecond
            | MetricTag::EffectiveImageSamplesPerSecondPerUser
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
        "effective_image_samples_per_second" => MetricTag::EffectiveImageSamplesPerSecond,
        "tokens_in_flight" => MetricTag::TokensInFlight,
        "active_decode_throughput" => MetricTag::ActiveDecodeThroughput,
        "active_prefill_throughput" => MetricTag::ActivePrefillThroughput,
        "active_decode_throughput_per_user" => MetricTag::ActiveDecodeThroughputPerUser,
        "active_prefill_throughput_per_user" => MetricTag::ActivePrefillThroughputPerUser,
        "active_image_samples_per_second" => MetricTag::ActiveImageSamplesPerSecond,
        "effective_image_samples_per_second_per_user" => {
            MetricTag::EffectiveImageSamplesPerSecondPerUser
        }
        "active_total_throughput" => MetricTag::ActiveTotalThroughput,
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dispatch::sink::ObservedSpecDecodeAcceptance;
    use crate::metrics_core::ingest::{RequestTrace, TokenCounts, UsageMetrics};
    use crate::metrics_core::window::Phase;
    use std::collections::{BTreeMap, BTreeSet};

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
            first_content_chunk_tokens: None,
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

    fn spec_decode_record(
        start_ns: i64,
        phase: Phase,
        phase_index: Option<usize>,
        accepted_per_step: &[u64],
    ) -> RecordIngest {
        let mut record = RecordIngest::minimal(start_ns, start_ns + 10, phase);
        record.phase_index = phase_index;
        let steps = accepted_per_step.len() as u64;
        let accepted = accepted_per_step.iter().sum::<u64>();
        let drafted = steps * 4;
        let mut histogram = BTreeMap::new();
        for accepted in accepted_per_step {
            *histogram.entry(*accepted).or_insert(0) += 1;
        }
        record.spec_decode_acceptance = Some(ObservedSpecDecodeAcceptance {
            engine: "vllm".to_string(),
            mean_acceptance_length: 1.0 + accepted as f64 / steps as f64,
            draft_acceptance_rate: accepted as f64 / drafted as f64,
            acceptance_histogram: histogram,
            num_accepted_draft_tokens: accepted,
            num_draft_tokens: drafted,
            num_spec_steps: steps,
            num_spec_tokens: Some(4),
            completion_tokens: Some(accepted + steps),
            per_step_accepted: Some(accepted_per_step.to_vec()),
            per_step_drafted: Some(vec![4; accepted_per_step.len()]),
        });
        record
    }

    fn zero_acceptance_record(phase_index: usize, steps: u64) -> RecordIngest {
        let mut record =
            RecordIngest::minimal(phase_index as i64, phase_index as i64 + 1, Phase::Profiling);
        record.phase_index = Some(phase_index);
        record.spec_decode_acceptance = Some(ObservedSpecDecodeAcceptance {
            engine: "vllm".to_string(),
            mean_acceptance_length: 1.0,
            draft_acceptance_rate: 0.0,
            acceptance_histogram: BTreeMap::from([(0, steps)]),
            num_accepted_draft_tokens: 0,
            num_draft_tokens: steps,
            num_spec_steps: steps,
            num_spec_tokens: Some(1),
            completion_tokens: Some(steps),
            per_step_accepted: None,
            per_step_drafted: None,
        });
        record
    }

    fn assert_close(actual: Option<f64>, expected: f64) {
        let actual = actual.expect("metric is present");
        assert!(
            (actual - expected).abs() <= 1e-10,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn spec_decode_worked_example_scopes_all_eleven_metrics_and_full_pool() {
        let mut accumulator = MetricsAccumulator::new();
        accumulator.process_record(&spec_decode_record(
            100,
            Phase::Profiling,
            Some(0),
            &[2, 3, 1, 4, 2, 0, 3, 3],
        ));
        accumulator.process_record(&spec_decode_record(
            200,
            Phase::Profiling,
            Some(1),
            &[1, 1, 0],
        ));
        accumulator.process_record(&spec_decode_record(300, Phase::Warmup, Some(0), &[4]));

        let profiling = accumulator.export_results(&ExportContext::phase(Phase::Profiling));
        assert_close(
            profiling.finite_value(MetricTag::SpecDecodeAcceptanceLength),
            (3.25 + (1.0 + 2.0 / 3.0)) / 2.0,
        );
        assert_close(
            profiling.finite_value(MetricTag::SpecDecodeDraftAcceptanceRate),
            (56.25 + (2.0 / 12.0 * 100.0)) / 2.0,
        );
        assert_close(
            profiling.finite_value(MetricTag::SpecDecodeAcceptedPerVerified),
            (0.65 + (5.0 / 15.0)) / 2.0,
        );
        assert_close(profiling.finite_value(MetricTag::SpecDecodeSteps), 5.5);
        assert_close(
            profiling.finite_value(MetricTag::SpecDecodeAcceptedDraftTokens),
            10.0,
        );
        assert_close(
            profiling.finite_value(MetricTag::SpecDecodeDraftTokens),
            22.0,
        );
        assert_close(
            profiling.finite_value(MetricTag::TotalSpecDecodeSteps),
            11.0,
        );
        assert_close(
            profiling.finite_value(MetricTag::TotalAcceptedDraftTokens),
            20.0,
        );
        assert_close(profiling.finite_value(MetricTag::TotalDraftTokens), 44.0);
        assert_close(
            profiling.finite_value(MetricTag::SpecDecodeTokenWeightedAcceptanceLength),
            1.0 + 20.0 / 11.0,
        );
        assert_close(
            profiling.finite_value(MetricTag::SpecDecodeOverallDraftAcceptanceRate),
            20.0 / 44.0 * 100.0,
        );
        assert_eq!(
            profiling.pooled_spec_decode_acceptance_histogram(),
            Some(&BTreeMap::from([(0, 2), (1, 3), (2, 2), (3, 3), (4, 1)]))
        );

        let first = accumulator.export_results(&ExportContext::phase_index(Phase::Profiling, 0));
        assert_eq!(
            first.pooled_spec_decode_acceptance_histogram(),
            Some(&BTreeMap::from([(0, 1), (1, 1), (2, 2), (3, 3), (4, 1)]))
        );
        assert_close(first.finite_value(MetricTag::TotalSpecDecodeSteps), 8.0);

        let second = accumulator.export_results(&ExportContext::phase_index(Phase::Profiling, 1));
        assert_eq!(
            second.pooled_spec_decode_acceptance_histogram(),
            Some(&BTreeMap::from([(0, 1), (1, 2)]))
        );
        assert_close(second.finite_value(MetricTag::TotalSpecDecodeSteps), 3.0);

        let window = accumulator.export_results(&ExportContext::time_range(100, 150));
        assert!(window.result(MetricTag::SpecDecodeSteps).is_some());
        assert!(window.pooled_spec_decode_acceptance_histogram().is_none());
    }

    #[test]
    fn spec_decode_absence_and_exact_sketch_append_are_consistent() {
        let records = [
            spec_decode_record(100, Phase::Profiling, Some(0), &[2, 3, 1, 4, 2, 0, 3, 3]),
            spec_decode_record(200, Phase::Profiling, Some(1), &[1, 1, 0]),
        ];
        let mut absent = MetricsAccumulator::new();
        absent.process_record(&RecordIngest::minimal(0, 1, Phase::Profiling));
        let absent = absent.summarize();
        assert!(
            absent
                .result(MetricTag::SpecDecodeAcceptanceLength)
                .is_none()
        );
        assert!(absent.pooled_spec_decode_acceptance_histogram().is_none());

        for mode in [
            MetricsStorageMode::Exact,
            MetricsStorageMode::Sketch { compression: 100.0 },
        ] {
            let is_exact = matches!(mode, MetricsStorageMode::Exact);
            let config = MetricsConfig {
                storage_mode: mode,
                ..MetricsConfig::default()
            };
            let mut whole = MetricsAccumulator::with_config(config.clone());
            let mut left = MetricsAccumulator::with_config(config.clone());
            let mut right = MetricsAccumulator::with_config(config.clone());
            for record in &records {
                whole.process_record(record);
            }
            left.process_record(&records[0]);
            right.process_record(&records[1]);
            left.merge(&right).expect("compatible stores append");
            assert_eq!(
                left.column_store().spec_decode_acceptance(0).is_some(),
                is_exact
            );
            assert_eq!(
                left.column_store().spec_decode_acceptance(1).is_some(),
                is_exact
            );

            for context in [
                ExportContext::phase(Phase::Profiling),
                ExportContext::phase_index(Phase::Profiling, 0),
                ExportContext::phase_index(Phase::Profiling, 1),
            ] {
                let direct = whole.export_results(&context);
                let merged = left.export_results(&context);
                assert_eq!(
                    direct.pooled_spec_decode_acceptance_histogram(),
                    merged.pooled_spec_decode_acceptance_histogram()
                );
                for tag in [
                    MetricTag::SpecDecodeAcceptanceLength,
                    MetricTag::SpecDecodeTokenWeightedAcceptanceLength,
                    MetricTag::SpecDecodeDraftAcceptanceRate,
                    MetricTag::SpecDecodeOverallDraftAcceptanceRate,
                    MetricTag::SpecDecodeAcceptedPerVerified,
                    MetricTag::SpecDecodeSteps,
                    MetricTag::SpecDecodeAcceptedDraftTokens,
                    MetricTag::SpecDecodeDraftTokens,
                    MetricTag::TotalSpecDecodeSteps,
                    MetricTag::TotalAcceptedDraftTokens,
                    MetricTag::TotalDraftTokens,
                ] {
                    assert_close(merged.finite_value(tag), direct.finite_value(tag).unwrap());
                }
            }
        }
    }

    #[test]
    fn spec_decode_histogram_ingest_remains_exact_past_u64_max() {
        let mut accumulator = MetricsAccumulator::new();
        accumulator.process_record(&zero_acceptance_record(0, u64::MAX));
        accumulator.process_record(&zero_acceptance_record(0, 1));

        let summary = accumulator.export_results(&ExportContext::phase_index(Phase::Profiling, 0));
        assert_eq!(
            summary
                .pooled_spec_decode_acceptance_histogram()
                .and_then(|histogram| histogram.get(&0))
                .copied()
                .map(u128::from),
            Some(u64::MAX as u128 + 1)
        );
    }

    #[test]
    fn spec_decode_histogram_append_remains_exact_past_u64_max() {
        let mut left = MetricsAccumulator::new();
        left.process_record(&zero_acceptance_record(0, u64::MAX));
        let mut right = MetricsAccumulator::new();
        right.process_record(&zero_acceptance_record(0, 1));
        left.merge(&right).expect("compatible stores append");

        let summary = left.export_results(&ExportContext::phase_index(Phase::Profiling, 0));
        assert_eq!(
            summary
                .pooled_spec_decode_acceptance_histogram()
                .and_then(|histogram| histogram.get(&0))
                .copied()
                .map(u128::from),
            Some(u64::MAX as u128 + 1)
        );
    }

    #[test]
    fn spec_decode_histogram_context_pool_remains_exact_past_u64_max() {
        let mut accumulator = MetricsAccumulator::new();
        accumulator.process_record(&zero_acceptance_record(0, u64::MAX));
        accumulator.process_record(&zero_acceptance_record(1, 1));

        let summary = accumulator.export_results(&ExportContext::phase(Phase::Profiling));
        assert_eq!(
            summary
                .pooled_spec_decode_acceptance_histogram()
                .and_then(|histogram| histogram.get(&0))
                .copied()
                .map(u128::from),
            Some(u64::MAX as u128 + 1)
        );
    }

    #[test]
    fn spec_decode_accepted_per_verified_keeps_large_valid_counts() {
        let mut accumulator = MetricsAccumulator::new();
        accumulator.process_record(&zero_acceptance_record(0, u64::MAX));

        assert_close(
            accumulator
                .export_results(&ExportContext::phase(Phase::Profiling))
                .finite_value(MetricTag::SpecDecodeAcceptedPerVerified),
            0.5,
        );
    }

    #[test]
    fn spec_decode_phase_index_without_phase_aligns_exact_sketch_and_histogram_selection() {
        let records = [
            spec_decode_record(100, Phase::Profiling, Some(0), &[2, 3]),
            spec_decode_record(200, Phase::Warmup, Some(0), &[1]),
            spec_decode_record(300, Phase::Profiling, Some(1), &[4]),
        ];
        let context = ExportContext {
            start_ns: None,
            end_ns: None,
            phase: None,
            phase_index: Some(0),
        };
        let mut summaries = Vec::new();
        for storage_mode in [
            MetricsStorageMode::Exact,
            MetricsStorageMode::Sketch { compression: 100.0 },
        ] {
            let mut accumulator = MetricsAccumulator::with_config(MetricsConfig {
                storage_mode,
                ..MetricsConfig::default()
            });
            for record in &records {
                accumulator.process_record(record);
            }
            summaries.push(accumulator.export_results(&context));
        }

        assert_eq!(
            summaries[0].finite_value(MetricTag::TotalSpecDecodeSteps),
            Some(3.0)
        );
        assert_eq!(
            summaries[0].pooled_spec_decode_acceptance_histogram(),
            Some(&BTreeMap::from([(1, 1), (2, 1), (3, 1)]))
        );
        assert_eq!(
            summaries[1].finite_value(MetricTag::TotalSpecDecodeSteps),
            summaries[0].finite_value(MetricTag::TotalSpecDecodeSteps)
        );
        assert_eq!(
            summaries[1].pooled_spec_decode_acceptance_histogram(),
            summaries[0].pooled_spec_decode_acceptance_histogram()
        );
    }

    #[test]
    fn spec_decode_canonical_value_follows_exact_and_sketch_row_lifecycle() {
        let record = spec_decode_record(100, Phase::Profiling, Some(0), &[2, 3]);
        let expected = record.spec_decode_acceptance.as_ref().unwrap();

        let mut exact = MetricsAccumulator::new();
        exact.process_record(&record);
        assert_eq!(
            exact.column_store().spec_decode_acceptance(0),
            Some(expected)
        );

        let mut sketch = MetricsAccumulator::with_config(MetricsConfig {
            storage_mode: MetricsStorageMode::Sketch { compression: 100.0 },
            ..MetricsConfig::default()
        });
        sketch.process_record(&record);
        assert_eq!(sketch.record_count(), 0);
        assert!(sketch.column_store().spec_decode_acceptance(0).is_none());
        assert_eq!(
            sketch
                .export_results(&ExportContext::phase(Phase::Profiling))
                .pooled_spec_decode_acceptance_histogram(),
            Some(&BTreeMap::from([(2, 1), (3, 1)]))
        );
    }

    /// Deterministic pseudo-random unit values (a small LCG — no wall clock).
    fn lcg_stream(seed: u64) -> impl FnMut() -> f64 {
        let mut state = seed;
        move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    fn latency_record(index: i64, latency_ms: f64) -> RecordIngest {
        let start = 1_000_000_000 + index * 2_000_000;
        let end = start + (latency_ms * 1_000_000.0) as i64;
        successful_record(start, end)
    }

    fn sketch_accumulator() -> MetricsAccumulator {
        MetricsAccumulator::with_config(MetricsConfig {
            storage_mode: MetricsStorageMode::Sketch { compression: 100.0 },
            ..MetricsConfig::default()
        })
    }

    #[test]
    fn sketch_mode_keeps_counts_sums_extrema_exact_and_percentiles_close() {
        let mut next = lcg_stream(0xC0FFEE);
        let mut exact = MetricsAccumulator::new();
        let mut sketch = sketch_accumulator();
        for index in 0..20_000 {
            // Integer-nanosecond latencies keep the f64 running sum order-independent,
            // so the sketch's arrival-order sum is bitwise identical to the exact
            // absolute-row-order sum below 2^53.
            let record = latency_record(index, 50.0 + next() * 500.0);
            exact.process_record(&record);
            sketch.process_record(&record);
        }
        let exact_summary = exact.summarize();
        let sketch_summary = sketch.summarize();

        assert_eq!(
            exact_summary.finite_value(MetricTag::RequestCount),
            sketch_summary.finite_value(MetricTag::RequestCount)
        );
        assert_eq!(
            exact_summary.finite_value(MetricTag::RequestThroughput),
            sketch_summary.finite_value(MetricTag::RequestThroughput),
            "rate derivations stay exact from the exact min/max timestamp aggregates"
        );

        let exact_latency = exact_summary
            .result(MetricTag::RequestLatency)
            .unwrap()
            .distribution()
            .unwrap();
        let sketch_latency = sketch_summary
            .result(MetricTag::RequestLatency)
            .unwrap()
            .distribution()
            .unwrap();
        assert_eq!(exact_latency.count, sketch_latency.count);
        assert_eq!(exact_latency.sum, sketch_latency.sum);
        assert_eq!(exact_latency.avg, sketch_latency.avg);
        assert_eq!(exact_latency.min, sketch_latency.min);
        assert_eq!(exact_latency.max, sketch_latency.max);
        assert!(sketch_latency.std.is_some());

        // Percentiles are t-digest approximations; the latency band spans ~500 ms so
        // 1% of range is ~5 ms. Allow a small absolute + relative tolerance.
        for percentile in crate::metrics_core::PERCENTILES {
            let exact_value = exact_latency
                .percentiles
                .get(&percentile)
                .copied()
                .and_then(MetricValue::as_f64)
                .unwrap();
            let sketch_value = sketch_latency
                .percentiles
                .get(&percentile)
                .copied()
                .and_then(MetricValue::as_f64)
                .unwrap();
            assert!(
                (exact_value - sketch_value).abs() <= 5.0 + exact_value.abs() * 0.02,
                "p{percentile}: exact {exact_value:.4} vs sketch {sketch_value:.4}"
            );
        }
    }

    #[test]
    fn sketch_partitions_merge_associatively() {
        let mut next = lcg_stream(0xABCDEF);
        let mut whole = sketch_accumulator();
        let mut shards = [
            sketch_accumulator(),
            sketch_accumulator(),
            sketch_accumulator(),
        ];
        for index in 0..12_000i64 {
            let record = latency_record(index, 10.0 + next() * 200.0);
            whole.process_record(&record);
            shards[(index % 3) as usize].process_record(&record);
        }
        let mut merged = sketch_accumulator();
        for shard in &shards {
            merged.merge(shard).unwrap();
        }

        let whole_summary = whole.summarize();
        let merged_summary = merged.summarize();
        assert_eq!(
            whole_summary.finite_value(MetricTag::RequestCount),
            merged_summary.finite_value(MetricTag::RequestCount)
        );
        let whole_latency = whole_summary
            .result(MetricTag::RequestLatency)
            .unwrap()
            .distribution()
            .unwrap();
        let merged_latency = merged_summary
            .result(MetricTag::RequestLatency)
            .unwrap()
            .distribution()
            .unwrap();
        assert_eq!(whole_latency.count, merged_latency.count);
        assert_eq!(whole_latency.min, merged_latency.min);
        assert_eq!(whole_latency.max, merged_latency.max);
        assert_eq!(whole_latency.sum, merged_latency.sum);
        for percentile in crate::metrics_core::PERCENTILES {
            let whole_value = whole_latency
                .percentiles
                .get(&percentile)
                .copied()
                .and_then(MetricValue::as_f64)
                .unwrap();
            let merged_value = merged_latency
                .percentiles
                .get(&percentile)
                .copied()
                .and_then(MetricValue::as_f64)
                .unwrap();
            assert!(
                (whole_value - merged_value).abs() <= 5.0 + whole_value.abs() * 0.04,
                "p{percentile}: whole {whole_value:.4} vs merged {merged_value:.4}"
            );
        }
    }

    /// Tolerance for the in-process sharded exact-fold summary.
    ///
    /// Counts, min, max, and percentiles are order-independent set operations over the
    /// merged value multiset, so they must be BIT-EXACT after the append-only shard
    /// merge. Sums and derived means depend on f64 summation order — which the
    /// local-dense concat reorders relative to a single dispatch-order ingest — so they
    /// need match only within a tiny relative epsilon.
    const SHARD_MERGE_REL_EPSILON: f64 = 1e-9;

    fn assert_within_rel_epsilon(reference: f64, candidate: f64, label: &str) {
        let tolerance = SHARD_MERGE_REL_EPSILON * reference.abs().max(1.0);
        assert!(
            (reference - candidate).abs() <= tolerance,
            "{label}: reference {reference} vs merged {candidate} exceeds relative \
             epsilon {SHARD_MERGE_REL_EPSILON}"
        );
    }

    /// Reusable tolerance comparison for one distribution: counts/min/max/percentiles
    /// EXACT; sum/avg within [`SHARD_MERGE_REL_EPSILON`].
    fn assert_distribution_within_tolerance(
        reference: &crate::metrics_core::DistributionStats,
        merged: &crate::metrics_core::DistributionStats,
        tag: MetricTag,
    ) {
        assert_eq!(reference.count, merged.count, "{tag:?} count must be exact");
        assert_eq!(reference.min, merged.min, "{tag:?} min must be exact");
        assert_eq!(reference.max, merged.max, "{tag:?} max must be exact");
        for percentile in crate::metrics_core::PERCENTILES {
            let r = reference.percentiles.get(&percentile).copied();
            let m = merged.percentiles.get(&percentile).copied();
            assert_eq!(r, m, "{tag:?} p{percentile} must be exact");
        }
        if let (Some(r), Some(m)) = (reference.sum.as_f64(), merged.sum.as_f64()) {
            assert_within_rel_epsilon(r, m, &format!("{tag:?} sum"));
        }
        if let (Some(r), Some(m)) = (reference.avg.as_f64(), merged.avg.as_f64()) {
            assert_within_rel_epsilon(r, m, &format!("{tag:?} avg"));
        }
    }

    /// Several dense per-shard exact accumulators, each fed a disjoint set of
    /// records at local-dense `request_index` slots, merged through
    /// `MetricsAccumulator::merge` (`append_store`), yield a summary within tolerance of
    /// a single accumulator fed every record in global order. Counts and percentiles
    /// stay exact; sums/means fall within the relative epsilon.
    #[test]
    fn sharded_exact_fold_merge_is_within_tolerance_of_single_ingest() {
        let shard_count = 4usize;
        let total = 8_000i64;
        let mut next = lcg_stream(0x5EED_A11CE);

        // Reference: one exact accumulator fed every record in global order (each row
        // pushed to a dense global slot).
        let mut reference = MetricsAccumulator::new();
        // Per-shard EXACT accumulators, each stamped with a LOCAL-dense `0..N_shard`
        // request index so its store is dense and `append_store` accepts it.
        let mut shards: Vec<MetricsAccumulator> = (0..shard_count)
            .map(|_| MetricsAccumulator::new())
            .collect();
        let mut shard_next = vec![0usize; shard_count];

        for index in 0..total {
            let record = latency_record(index, 50.0 + next() * 500.0);
            reference.process_record(&record);
            let shard = (index as usize) % shard_count;
            let mut shard_record = record;
            shard_record.request_index = Some(shard_next[shard]);
            shard_next[shard] += 1;
            shards[shard].process_record(&shard_record);
        }

        let mut merged = MetricsAccumulator::new();
        for shard in &shards {
            merged.merge(shard).unwrap();
        }

        let reference_summary = reference.summarize();
        let merged_summary = merged.summarize();

        // Counts are the strict correctness invariant: a dropped or double-counted
        // record moves them.
        assert_eq!(
            reference_summary.finite_value(MetricTag::RequestCount),
            Some(total as f64),
        );
        assert_eq!(
            reference_summary.finite_value(MetricTag::RequestCount),
            merged_summary.finite_value(MetricTag::RequestCount),
            "merged count must equal the single-ingest count exactly",
        );
        // Rates derive from the exact count and exact min/max timestamp aggregates, so
        // they stay exact across the reorder.
        assert_eq!(
            reference_summary.finite_value(MetricTag::RequestThroughput),
            merged_summary.finite_value(MetricTag::RequestThroughput),
            "throughput derives from exact aggregates and stays exact",
        );

        for tag in [
            MetricTag::RequestLatency,
            MetricTag::InterTokenLatency,
            MetricTag::OutputSequenceLength,
        ] {
            let reference_distribution = reference_summary
                .result(tag)
                .unwrap()
                .distribution()
                .unwrap();
            let merged_distribution = merged_summary.result(tag).unwrap().distribution().unwrap();
            assert_distribution_within_tolerance(reference_distribution, merged_distribution, tag);
        }
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
    fn request_index_is_the_absolute_column_slot_not_append_order() {
        let mut accumulator = MetricsAccumulator::new();
        let mut late = RecordIngest::minimal(30, 40, Phase::Profiling);
        late.request_index = Some(2);
        let mut early = RecordIngest::minimal(10, 20, Phase::Profiling);
        early.request_index = Some(0);

        accumulator.process_record(&late);
        accumulator.process_record(&early);

        assert_eq!(accumulator.record_count(), 2);
        assert_eq!(accumulator.column_store().row_count(), 3);
        assert_eq!(accumulator.column_store().start_ns()[0], 10.0);
        assert!(accumulator.column_store().start_ns()[1].is_nan());
        assert_eq!(accumulator.column_store().start_ns()[2], 30.0);
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
        // latency=100ms, ttft=20ms (derived above from itl=(latency-ttft)/9).
        let decode_duration_ms = summary
            .result(MetricTag::DecodeDuration)
            .unwrap()
            .distribution()
            .unwrap()
            .avg
            .as_f64()
            .unwrap();
        assert!((decode_duration_ms - 80.0).abs() < 1e-9);
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
    fn bundled_first_content_chunk_corrects_itl_and_tps_per_user() {
        let mut record = successful_record(1_000_000_000, 1_100_000_000);
        record.tokens.first_content_chunk_tokens = Some(4);
        let mut accumulator = MetricsAccumulator::new();
        accumulator.process_record(&record);
        let summary = accumulator.summarize();

        let itl_ms = summary
            .result(MetricTag::InterTokenLatency)
            .unwrap()
            .distribution()
            .unwrap()
            .avg
            .as_f64()
            .unwrap();
        assert!((itl_ms - 80.0 / 6.0).abs() < 1e-12);
        assert_eq!(
            summary
                .result(MetricTag::OutputTokenThroughputPerUser)
                .unwrap()
                .distribution()
                .unwrap()
                .avg,
            MetricValue::Finite(75.0)
        );
    }

    #[test]
    fn inconsistent_first_content_chunk_counts_fall_back_to_legacy_itl() {
        for first_content_chunk_tokens in [0, 10, 11] {
            let mut record = successful_record(1_000_000_000, 1_100_000_000);
            record.tokens.first_content_chunk_tokens = Some(first_content_chunk_tokens);
            let mut accumulator = MetricsAccumulator::new();
            accumulator.process_record(&record);
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
        }
    }

    #[test]
    fn default_mode_uses_client_token_counts_endpoint_usage_only_feeds_discrepancy() {
        let mut record = successful_record(1_000_000_000, 1_100_000_000);
        // Three observed chunks yield two exact ICL samples and a locally
        // tokenized OSL of ten (client `tokens.output=9` + `reasoning=1`). One
        // endpoint usage object reports twenty completion tokens (four reasoning).
        // In DEFAULT mode (no `use_server_token_count`) the visible token metrics
        // are byte-exact with the Python record metrics: they come from the CLIENT
        // `token_counts`, NOT from endpoint usage. Endpoint usage feeds only the
        // usage_* metrics and the client/server discrepancy diagnostic. Server
        // usage becomes authoritative only under `use_server_token_count`, which
        // the `metrics.rs` observer applies when building the ingest record.
        record.usage.completion_tokens = Some(20);
        record.usage.reasoning_tokens = Some(4);

        let mut accumulator = MetricsAccumulator::new();
        accumulator.process_record(&record);
        let summary = accumulator.summarize();

        // Client OSL (9 output + 1 reasoning), not server completion_tokens (20).
        assert_eq!(
            summary.finite_value(MetricTag::TotalOutputSequenceLength),
            Some(10.0)
        );
        // Client output tokens (9), not server (20 - 4 = 16).
        assert_eq!(
            summary.finite_value(MetricTag::TotalOutputTokens),
            Some(9.0)
        );
        // Client reasoning (1), not server (4).
        assert_eq!(
            summary.finite_value(MetricTag::TotalReasoningTokens),
            Some(1.0)
        );
        // Throughput follows the client OSL of 10 over the 0.1s window.
        assert_eq!(
            summary.finite_value(MetricTag::OutputTokenThroughput),
            Some(100.0)
        );
        let itl_ms = summary
            .result(MetricTag::InterTokenLatency)
            .unwrap()
            .distribution()
            .unwrap()
            .avg
            .as_f64()
            .unwrap();
        // (100ms - 20ms) / (osl - 1) with client osl = 10.
        assert!((itl_ms - 80.0 / 9.0).abs() < 1e-12);
        assert_eq!(
            summary
                .result(MetricTag::E2eOutputTokenThroughput)
                .unwrap()
                .distribution()
                .unwrap()
                .avg,
            MetricValue::Finite(100.0)
        );
        // The discrepancy still compares server usage (20) against the client
        // OSL (10): |20 - 10| / 10 * 100 = 100%.
        assert_eq!(
            summary
                .result(MetricTag::UsageCompletionTokensDiffPct)
                .unwrap()
                .distribution()
                .unwrap()
                .avg,
            MetricValue::Finite(100.0),
            "usage discrepancy compares server usage against the client count"
        );
        let icl = accumulator
            .column_store()
            .inter_chunk_latency_replay()
            .unwrap();
        assert_eq!(icl.values, &[10_000_000.0, 10_000_000.0]);
        assert_eq!(icl.record_indices().collect::<Vec<_>>(), vec![0, 0]);
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
            phase_index: None,
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
    fn slo_threshold_caches_catalog_direction() {
        // OutputSequenceLength is LARGER_IS_BETTER; RequestLatency is smaller-is-better.
        assert!(SloThreshold::native(MetricTag::OutputSequenceLength, 1.0).larger_is_better);
        assert!(!SloThreshold::native(MetricTag::RequestLatency, 1.0).larger_is_better);
        assert!(
            SloThreshold::from_display(MetricTag::OutputSequenceLength, 1.0)
                .unwrap()
                .larger_is_better
        );
    }

    #[test]
    fn slo_threshold_passes_via_native_units_both_directions() {
        // Larger-is-better metric: passes when value >= threshold.
        let lib = SloThreshold::native(MetricTag::OutputSequenceLength, 100.0);
        assert!(lib.passes(100.0));
        assert!(lib.passes(150.0));
        assert!(!lib.passes(99.0));

        // Smaller-is-better metric: passes when value <= threshold.
        let sib = SloThreshold::native(MetricTag::RequestLatency, 150_000_000.0);
        assert!(sib.passes(150_000_000.0));
        assert!(sib.passes(100_000_000.0));
        assert!(!sib.passes(150_000_001.0));

        // Routed through the shared Definition::passes_threshold with typed Native.
        assert_eq!(
            sib.passes(120_000_000.0),
            sib.definition
                .passes_threshold(Native::new(120_000_000.0), Native::new(sib.native_value))
        );
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
        record.http = RequestTrace {
            blocked_ns: Some(10),
            connecting_ns: Some(20),
            waiting_ns: Some(30),
            ..RequestTrace::default()
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
            phase_index: None,
        };
        let summary = accumulator.export_results(&context);
        assert!(summary.result(MetricTag::RequestThroughput).is_none());
    }

    #[test]
    fn per_worker_merge_matches_single_accumulator_ingest_order() {
        let mut first = successful_record(1_000_000_000, 1_100_000_000);
        first.dimensions = InferenceDimensions {
            endpoint_url: Some("https://endpoint-z/v1/chat/completions".to_string()),
            model: Some("model-b".to_string()),
        };
        let mut second = successful_record(1_200_000_000, 1_350_000_000);
        second.worker_id = Some("worker-1".to_string());
        second.turn_index = 1;
        second.dimensions = InferenceDimensions {
            endpoint_url: Some("https://endpoint-a/v1/chat/completions".to_string()),
            model: Some("model-a".to_string()),
        };

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
        let summary = left.summarize();
        assert_eq!(summary.inference_series().len(), 2);
        assert_eq!(
            summary.inference_series()[0]
                .dimensions()
                .endpoint_url
                .as_deref(),
            Some("https://endpoint-a/v1/chat/completions")
        );
        assert_eq!(
            summary.inference_series()[1]
                .dimensions()
                .endpoint_url
                .as_deref(),
            Some("https://endpoint-z/v1/chat/completions")
        );
        assert!(summary.inference_series().iter().all(|series| {
            series
                .result_by_name(MetricTag::RequestCount.as_str())
                .and_then(MetricResult::finite_value)
                == Some(1.0)
        }));
    }

    #[test]
    fn model_endpoint_series_own_timeslices_and_stable_per_series_rates() {
        let mut accumulator = MetricsAccumulator::with_config(MetricsConfig {
            slice_duration_ns: Some(1_000_000_000),
            ..MetricsConfig::default()
        });
        let mut first = successful_record(1_000_000_000, 1_100_000_000);
        first.dimensions = InferenceDimensions {
            endpoint_url: Some("https://endpoint/v1/chat/completions".to_string()),
            model: Some("model-a".to_string()),
        };
        let mut second = successful_record(1_200_000_000, 1_400_000_000);
        second.dimensions = InferenceDimensions {
            endpoint_url: Some("https://endpoint/v1/chat/completions".to_string()),
            model: Some("model-b".to_string()),
        };
        accumulator.process_record(&first);
        accumulator.process_record(&second);

        let summary = accumulator.summarize();
        assert_eq!(summary.inference_series().len(), 2);
        for series in summary.inference_series() {
            assert_eq!(series.timeslices().len(), 1);
            assert_eq!(
                series
                    .result_by_name(MetricTag::RequestCount.as_str())
                    .and_then(MetricResult::finite_value),
                Some(1.0)
            );
            assert_eq!(
                series.timeslices()[0]
                    .metrics
                    .get(MetricTag::RequestCount.as_str())
                    .and_then(MetricResult::finite_value),
                Some(1.0)
            );
        }
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
                        *tag,
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
                            | MetricTag::TotalSpecDecodeSteps
                            | MetricTag::TotalAcceptedDraftTokens
                            | MetricTag::TotalDraftTokens
                            | MetricTag::SpecDecodeTokenWeightedAcceptanceLength
                            | MetricTag::SpecDecodeOverallDraftAcceptanceRate
                            | MetricTag::TotalNumImages
                            | MetricTag::ImageSamplesPerSecond
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
