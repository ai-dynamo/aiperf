// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Typed, deterministic native-v2 report construction.
//!
//! This module is IO-free. It translates an accumulator summary into the
//! metrics-first, type-specific-series representation; application-layer
//! exporters decide where to write it.

use crate::metrics_core::catalog::{
    AggregationKind, MetricConsoleGroup, MetricFlags, MetricTag, MetricType, spec_for,
};
use crate::metrics_core::steady_state::SteadyStateOutcome;
use crate::metrics_core::{
    AccumulatorSummary, AccuracyAnalysis, AccuracyRecord, MetricResult, MetricResultData,
    MetricValue, SidecarMetric, SidecarStats,
};
use serde::Serialize as DeriveSerialize;
use serde::ser::{Serialize, Serializer};
use std::collections::BTreeMap;
use std::error::Error;
use std::fmt::{self, Display};
use std::ops::Deref;

/// Native report schema identifier.
pub const NATIVE_REPORT_SCHEMA_VERSION: &str = "2.0";

/// A present report value: finite numbers serialize normally; non-finite tails
/// serialize as JSON null without colliding with structurally absent fields.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ReportValue {
    /// Finite numeric value.
    Finite(f64),
    /// Present but non-finite value, reserved for error-adjusted tails.
    NonFinite,
}

impl Serialize for ReportValue {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            Self::Finite(value) => serializer.serialize_f64(*value),
            Self::NonFinite => serializer.serialize_none(),
        }
    }
}

/// Distribution statistics used by inference records and gauge series.
#[derive(Debug, Clone, PartialEq, DeriveSerialize)]
pub struct ReportDistributionStats {
    /// Number of observations.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub count: Option<usize>,
    /// Arithmetic or duration-weighted average.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub avg: Option<ReportValue>,
    /// Minimum observation.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min: Option<ReportValue>,
    /// Maximum observation.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max: Option<ReportValue>,
    /// Population standard deviation.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub std: Option<ReportValue>,
    /// Percentiles keyed by `pN`.
    pub percentiles: BTreeMap<String, ReportValue>,
}

/// Scalar statistics used by derived and min/max aggregate metrics.
#[derive(Debug, Clone, PartialEq, DeriveSerialize)]
pub struct ReportScalarStats {
    /// The scalar value.
    pub value: ReportValue,
}

/// Counter statistics used by sum aggregates.
#[derive(Debug, Clone, PartialEq, DeriveSerialize)]
pub struct ReportCounterStats {
    /// Accumulated total.
    pub total: ReportValue,
    /// Optional rate paired with this counter.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rate: Option<ReportValue>,
}

/// Histogram boundary-delta statistics supplied by server telemetry.
#[derive(Debug, Clone, PartialEq, DeriveSerialize)]
pub struct ReportHistogramStats {
    /// Number of phase observations.
    pub count: u64,
    /// Sum of phase observations.
    pub sum: ReportValue,
    /// Mean observation, when count is positive.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub avg: Option<ReportValue>,
    /// Observations per second over the authoritative phase window.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub count_rate: Option<ReportValue>,
    /// Observation-value sum per second.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sum_rate: Option<ReportValue>,
    /// Polynomial percentile estimates keyed by `pN`.
    pub percentiles: BTreeMap<String, ReportValue>,
    /// Reset-clamped cumulative bucket deltas.
    pub buckets: BTreeMap<String, u64>,
}

/// Type-specific statistics serialized without an additional wrapper tag.
#[derive(Debug, Clone, PartialEq, DeriveSerialize)]
#[serde(untagged)]
pub enum ReportStats {
    /// Distribution-shaped statistics.
    Distribution(ReportDistributionStats),
    /// Scalar-shaped statistics.
    Scalar(ReportScalarStats),
    /// Counter-shaped statistics.
    Counter(ReportCounterStats),
    /// Prometheus histogram-shaped statistics.
    Histogram(ReportHistogramStats),
}

/// One metric-series timeslice using the same stats shape as its parent.
#[derive(Debug, Clone, PartialEq, DeriveSerialize)]
pub struct ReportTimeslice {
    /// Inclusive window start in nanoseconds.
    pub start_ns: i64,
    /// Exclusive window end in nanoseconds.
    pub end_ns: i64,
    /// Whether the slice spans its full configured duration.
    pub complete: bool,
    /// Type-appropriate timeslice statistics.
    pub stats: ReportStats,
}

/// One labeled series for a metric.
#[derive(Debug, Clone, PartialEq, DeriveSerialize)]
pub struct MetricSeries {
    /// Stable label set; inference metrics label the selected model when known.
    pub labels: Option<BTreeMap<String, String>>,
    /// Selected inference endpoint or telemetry source endpoint.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub endpoint_url: Option<String>,
    /// Type-appropriate overall statistics.
    pub stats: ReportStats,
    /// Chronological non-empty timeslices.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub timeslices: Vec<ReportTimeslice>,
}

/// One metric keyed by stable name in the native report.
#[derive(Debug, Clone, PartialEq, DeriveSerialize)]
pub struct MetricEntry {
    /// Consumer-facing stats shape.
    #[serde(rename = "type")]
    pub metric_type: &'static str,
    /// Display unit.
    pub unit: String,
    /// Console group.
    pub group: &'static str,
    /// Plot/SLO direction.
    pub higher_is_better: bool,
    /// Deterministically ordered labeled series.
    pub series: Vec<MetricSeries>,
}

/// Typed run identity shared by report consumers.
#[derive(Debug, Clone, Default, PartialEq, Eq, DeriveSerialize)]
pub struct ReportRunInfo {
    /// Execution mode, such as `online` or `graph`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mode: Option<String>,
    /// Requested model name.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
}

/// A finite numeric fact carried by report run metadata.
///
/// Unlike request-metric tails, run metadata cannot use JSON null as a numeric
/// sentinel: a backend either supplies an exact finite fact or omits it. The
/// private representation makes that invariant hold for every serialized
/// value, including values assembled by external runner distributions.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct FiniteReportValue(f64);

impl FiniteReportValue {
    /// Validate and retain one finite value.
    pub fn new(value: f64) -> Result<Self, ReportMetadataError> {
        if value.is_finite() {
            Ok(Self(value))
        } else {
            Err(ReportMetadataError::new(
                "report run metadata numeric values must be finite",
            ))
        }
    }

    /// Return the validated value.
    pub fn get(self) -> f64 {
        self.0
    }
}

impl Serialize for FiniteReportValue {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_f64(self.0)
    }
}

/// Invalid typed common or pair-specific report metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReportMetadataError {
    message: String,
}

impl ReportMetadataError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl Display for ReportMetadataError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl Error for ReportMetadataError {}

/// One statically linked extension identity in the executing distribution.
#[derive(Debug, Clone, PartialEq, Eq, DeriveSerialize)]
pub struct ReportExtensionIdentity {
    /// Stable package-level extension name.
    pub name: String,
    /// Exact package version when the extension exposes one.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,
}

impl ReportExtensionIdentity {
    /// Validate a linked extension identity.
    pub fn new(
        name: impl Into<String>,
        version: Option<String>,
    ) -> Result<Self, ReportMetadataError> {
        let name = name.into();
        validate_nonempty_trimmed(&name, "extension name")?;
        if let Some(version) = &version {
            validate_nonempty_trimmed(version, "extension version")?;
        }
        Ok(Self { name, version })
    }
}

/// Run-local endpoint profile identity resolved by the coordinator registry.
#[derive(Debug, Clone, PartialEq, Eq, DeriveSerialize)]
pub struct ReportEndpointProfileIdentity {
    /// Authored profile name referenced by workloads.
    pub profile_id: String,
    /// Canonical endpoint factory ID selected after alias resolution.
    pub endpoint_id: String,
}

impl ReportEndpointProfileIdentity {
    /// Validate a profile and canonical endpoint identity.
    pub fn new(
        profile_id: impl Into<String>,
        endpoint_id: impl Into<String>,
    ) -> Result<Self, ReportMetadataError> {
        let profile_id = profile_id.into();
        let endpoint_id = endpoint_id.into();
        validate_nonempty_trimmed(&profile_id, "endpoint profile ID")?;
        validate_component_id(&endpoint_id, "endpoint factory ID")?;
        Ok(Self {
            profile_id,
            endpoint_id,
        })
    }
}

/// Coordinator-owned identity stamped exactly once before native-v2 commit.
///
/// Pair adapters deliberately do not construct this value. They return
/// [`ReportPairRunFacts`], while the one process coordinator supplies the
/// executable digest and the identities from its frozen registries.
#[derive(Debug, Clone, PartialEq, Eq, DeriveSerialize)]
pub struct ReportRunMetadata {
    /// BLAKE3 identity of the exact runner executable that performed the run.
    pub distribution_id: String,
    /// Canonical selected transport factory ID.
    pub transport: String,
    /// Canonical selected workload factory ID.
    pub workload: String,
    /// Statically linked extensions in deterministic identity order.
    pub extensions: Vec<ReportExtensionIdentity>,
    /// Endpoint profiles in authored order after canonical alias resolution.
    pub endpoint_profiles: Vec<ReportEndpointProfileIdentity>,
}

impl ReportRunMetadata {
    /// Build the complete coordinator-owned common run metadata block.
    pub fn new(
        distribution_id: impl Into<String>,
        transport: impl Into<String>,
        workload: impl Into<String>,
        mut extensions: Vec<ReportExtensionIdentity>,
        endpoint_profiles: Vec<ReportEndpointProfileIdentity>,
    ) -> Result<Self, ReportMetadataError> {
        let distribution_id = distribution_id.into();
        validate_distribution_id(&distribution_id)?;
        let transport = transport.into();
        validate_component_id(&transport, "transport ID")?;
        let workload = workload.into();
        validate_component_id(&workload, "workload ID")?;

        extensions
            .sort_by(|left, right| (&left.name, &left.version).cmp(&(&right.name, &right.version)));
        for adjacent in extensions.windows(2) {
            if adjacent[0].name == adjacent[1].name {
                return Err(ReportMetadataError::new(format!(
                    "duplicate linked extension {:?}",
                    adjacent[0].name
                )));
            }
        }
        let mut profile_ids = std::collections::BTreeSet::new();
        for profile in &endpoint_profiles {
            if !profile_ids.insert(profile.profile_id.as_str()) {
                return Err(ReportMetadataError::new(format!(
                    "duplicate endpoint profile ID {:?}",
                    profile.profile_id
                )));
            }
        }

        Ok(Self {
            distribution_id,
            transport,
            workload,
            extensions,
            endpoint_profiles,
        })
    }
}

/// Static and terminal Graph-IR facts shared by online and offline pairs.
#[derive(Debug, Clone, PartialEq, Eq, DeriveSerialize)]
pub struct ReportGraphRunInfo {
    /// Direct authored-input adapter selected before generic dataset loading.
    pub input_format: String,
    /// Complete root traces retained after direct lowering.
    pub root_count: usize,
    /// Total static nodes across all retained root-expanded plans.
    pub node_count: usize,
    /// Thread-per-core graph workers used by the pair.
    pub worker_count: usize,
    /// Ordered authored phases executed by the graph workload.
    pub phase_count: usize,
    /// Terminal workload counts when the pair exposes them.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub outcome: Option<ReportGraphOutcomeInfo>,
}

impl ReportGraphRunInfo {
    /// Validate static Graph-IR lowering and placement facts.
    pub fn new(
        input_format: impl Into<String>,
        root_count: usize,
        node_count: usize,
        worker_count: usize,
        phase_count: usize,
    ) -> Result<Self, ReportMetadataError> {
        let input_format = input_format.into();
        validate_component_id(&input_format, "graph input format")?;
        if root_count == 0 {
            return Err(ReportMetadataError::new(
                "graph root_count must be positive",
            ));
        }
        if node_count < root_count {
            return Err(ReportMetadataError::new(
                "graph node_count cannot be smaller than root_count",
            ));
        }
        if worker_count == 0 {
            return Err(ReportMetadataError::new(
                "graph worker_count must be positive",
            ));
        }
        if phase_count == 0 {
            return Err(ReportMetadataError::new(
                "graph phase_count must be positive",
            ));
        }
        Ok(Self {
            input_format,
            root_count,
            node_count,
            worker_count,
            phase_count,
            outcome: None,
        })
    }

    /// Attach terminal trace counts supplied by the graph workload.
    pub fn with_outcome(
        mut self,
        outcome: ReportGraphOutcomeInfo,
    ) -> Result<Self, ReportMetadataError> {
        outcome.validate()?;
        self.outcome = Some(outcome);
        Ok(self)
    }
}

/// Terminal trace accounting from one Graph-IR workload.
#[derive(Debug, Clone, Copy, PartialEq, Eq, DeriveSerialize)]
pub struct ReportGraphOutcomeInfo {
    /// Root traces that acquired whole-trace admission.
    pub admitted: u64,
    /// Root traces that drained successfully.
    pub completed: u64,
    /// Root traces aborted by a node or runtime failure.
    pub failed: u64,
}

impl ReportGraphOutcomeInfo {
    /// Construct terminal graph counts.
    pub fn new(admitted: u64, completed: u64, failed: u64) -> Self {
        Self {
            admitted,
            completed,
            failed,
        }
    }

    fn validate(&self) -> Result<(), ReportMetadataError> {
        let terminal = self
            .completed
            .checked_add(self.failed)
            .ok_or_else(|| ReportMetadataError::new("graph terminal trace count overflowed u64"))?;
        if terminal > self.admitted {
            return Err(ReportMetadataError::new(
                "graph completed + failed traces cannot exceed admitted traces",
            ));
        }
        Ok(())
    }
}

/// Clock family used by a backend-specific typed report block.
#[derive(Debug, Clone, Copy, PartialEq, Eq, DeriveSerialize)]
#[serde(rename_all = "snake_case")]
pub enum ReportClockKind {
    /// Monotonic wall-clock execution.
    Real,
    /// Deterministic discrete-event virtual time.
    Sim,
}

/// Dynamo deployment topology used by the in-process offline backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, DeriveSerialize)]
#[serde(rename_all = "snake_case")]
pub enum ReportDynamoTopology {
    /// One aggregate engine without a router.
    Single,
    /// Multiple aggregate workers behind one router.
    Aggregated,
    /// Separate prefill and decode worker pools.
    Disaggregated,
}

/// Dynamo request-routing policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, DeriveSerialize)]
#[serde(rename_all = "snake_case")]
pub enum ReportDynamoRouter {
    /// Deterministic round-robin routing.
    RoundRobin,
    /// Dynamo's prefix-affinity/load-aware KV routing.
    Kv,
}

/// Whole-summary byte-parity evidence produced by the Dynamo offline backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, DeriveSerialize)]
pub struct ReportDynamoParityInfo {
    /// Fields in the complete common flat summary schema.
    pub shared_fields: usize,
    /// Request/event fields independently accumulated by both collectors.
    pub independently_accumulated_fields: usize,
    /// Capacity/goodput fields imported from the owning backend.
    pub backend_owned_fields: usize,
    /// Bytes in either identical canonical compact JSON representation.
    pub serialized_bytes: usize,
}

impl ReportDynamoParityInfo {
    /// Validate exact byte-parity accounting.
    pub fn new(
        shared_fields: usize,
        independently_accumulated_fields: usize,
        backend_owned_fields: usize,
        serialized_bytes: usize,
    ) -> Result<Self, ReportMetadataError> {
        if independently_accumulated_fields.checked_add(backend_owned_fields) != Some(shared_fields)
        {
            return Err(ReportMetadataError::new(
                "Dynamo parity fields must partition shared_fields exactly",
            ));
        }
        if serialized_bytes == 0 {
            return Err(ReportMetadataError::new(
                "Dynamo parity serialized_bytes must be positive",
            ));
        }
        Ok(Self {
            shared_fields,
            independently_accumulated_fields,
            backend_owned_fields,
            serialized_bytes,
        })
    }
}

/// Backend-owned provisioned-capacity facts from Dynamo's canonical report.
#[derive(Debug, Clone, Copy, PartialEq, DeriveSerialize)]
pub struct ReportDynamoCapacityInfo {
    /// Provisioned prefill-worker time integrated over the run.
    pub prefill_worker_seconds: FiniteReportValue,
    /// Provisioned decode-worker time integrated over the run.
    pub decode_worker_seconds: FiniteReportValue,
    /// GPUs assigned to each prefill worker.
    pub prefill_gpus_per_worker: usize,
    /// GPUs assigned to each decode worker.
    pub decode_gpus_per_worker: usize,
    /// Total provisioned GPU-hours over startup, steady state, and drain.
    pub gpu_hours: FiniteReportValue,
}

impl ReportDynamoCapacityInfo {
    /// Validate and retain Dynamo's five backend-owned capacity facts.
    pub fn new(
        prefill_worker_seconds: f64,
        decode_worker_seconds: f64,
        prefill_gpus_per_worker: usize,
        decode_gpus_per_worker: usize,
        gpu_hours: f64,
    ) -> Result<Self, ReportMetadataError> {
        for (name, value) in [
            ("prefill_worker_seconds", prefill_worker_seconds),
            ("decode_worker_seconds", decode_worker_seconds),
            ("gpu_hours", gpu_hours),
        ] {
            if value < 0.0 {
                return Err(ReportMetadataError::new(format!(
                    "Dynamo {name} must be non-negative"
                )));
            }
        }
        Ok(Self {
            prefill_worker_seconds: FiniteReportValue::new(prefill_worker_seconds)?,
            decode_worker_seconds: FiniteReportValue::new(decode_worker_seconds)?,
            prefill_gpus_per_worker,
            decode_gpus_per_worker,
            gpu_hours: FiniteReportValue::new(gpu_hours)?,
        })
    }
}

/// Typed facts owned by the feature-gated Dynamo offline pair.
#[derive(Debug, Clone, PartialEq, DeriveSerialize)]
pub struct ReportDynamoRunInfo {
    /// Virtual clock used by the passive steppable engine.
    pub clock: ReportClockKind,
    /// Aggregate/disaggregate engine composition.
    pub topology: ReportDynamoTopology,
    /// Router selected for routed topologies.
    pub router: ReportDynamoRouter,
    /// Optional compile-time Dynamo features required by the authored run.
    pub required_features: Vec<String>,
    /// Authored aggregate worker count.
    pub workers: usize,
    /// Authored prefill worker count.
    pub prefill_workers: usize,
    /// Authored decode worker count.
    pub decode_workers: usize,
    /// Exact common-summary parity evidence.
    pub parity: ReportDynamoParityInfo,
    /// Five backend-owned provisioned-capacity facts when exposed by the run.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub capacity: Option<ReportDynamoCapacityInfo>,
}

impl ReportDynamoRunInfo {
    /// Validate static Dynamo engine and parity facts.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        clock: ReportClockKind,
        topology: ReportDynamoTopology,
        router: ReportDynamoRouter,
        mut required_features: Vec<String>,
        workers: usize,
        prefill_workers: usize,
        decode_workers: usize,
        parity: ReportDynamoParityInfo,
    ) -> Result<Self, ReportMetadataError> {
        // Both clocks are valid Dynamo engine report axes: `Sim` for
        // deterministic virtual-clock replay and `Real` for the wall-clock
        // in-process online mode (`--replay-mode online`). No further constraint
        // is imposed here; the enum is exhaustive.
        if workers == 0 || prefill_workers == 0 || decode_workers == 0 {
            return Err(ReportMetadataError::new(
                "Dynamo worker counts must be positive",
            ));
        }
        required_features.sort();
        for feature in &required_features {
            validate_nonempty_trimmed(feature, "Dynamo required feature")?;
        }
        if required_features.windows(2).any(|pair| pair[0] == pair[1]) {
            return Err(ReportMetadataError::new(
                "Dynamo required features must be unique",
            ));
        }
        Ok(Self {
            clock,
            topology,
            router,
            required_features,
            workers,
            prefill_workers,
            decode_workers,
            parity,
            capacity: None,
        })
    }

    /// Attach backend-owned capacity facts after the parity-checked run drains.
    pub fn with_capacity(mut self, capacity: ReportDynamoCapacityInfo) -> Self {
        self.capacity = Some(capacity);
        self
    }
}

/// Optional typed facts returned by one backend/workload pair.
///
/// The pair owns only these mode-specific facts. Common executable, registry,
/// and endpoint identity remains coordinator-owned in [`ReportRunMetadata`].
#[derive(Debug, Clone, Default, PartialEq, DeriveSerialize)]
pub struct ReportPairRunFacts {
    /// Graph-IR lowering, placement, and terminal counts.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub graph: Option<ReportGraphRunInfo>,
    /// Dynamo topology, capacity, and exact metric-parity evidence.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dynamo: Option<ReportDynamoRunInfo>,
}

impl ReportPairRunFacts {
    /// Construct an empty pair-fact set for ordinary scheduled/accuracy runs.
    pub fn new() -> Self {
        Self::default()
    }

    /// Attach Graph-IR facts.
    pub fn with_graph(mut self, graph: ReportGraphRunInfo) -> Self {
        self.graph = Some(graph);
        self
    }

    /// Attach Dynamo offline facts.
    pub fn with_dynamo(mut self, dynamo: ReportDynamoRunInfo) -> Self {
        self.dynamo = Some(dynamo);
        self
    }
}

/// Serialized native-v2 run block. [`Deref`] exposes [`ReportRunInfo`] while
/// serialization flattens protocol-v2 metadata into the same JSON object.
#[derive(Debug, Clone, Default, PartialEq, DeriveSerialize)]
pub struct ReportRun {
    /// Workload-facing run identity retained by existing report producers.
    #[serde(flatten)]
    pub info: ReportRunInfo,
    /// Coordinator-owned exact executable and registry identity.
    #[serde(flatten, skip_serializing_if = "Option::is_none")]
    run_metadata: Option<ReportRunMetadata>,
    /// Pair-owned typed mode facts.
    #[serde(flatten)]
    facts: ReportPairRunFacts,
}

impl ReportRun {
    fn unfinalized(info: ReportRunInfo) -> Self {
        Self {
            info,
            run_metadata: None,
            facts: ReportPairRunFacts::default(),
        }
    }

    /// Whether the coordinator has stamped exact protocol-v2 run metadata.
    pub fn is_finalized(&self) -> bool {
        self.run_metadata.is_some()
    }

    /// Borrow coordinator-owned common run metadata after finalization.
    pub fn run_metadata(&self) -> Option<&ReportRunMetadata> {
        self.run_metadata.as_ref()
    }

    /// Borrow pair-owned typed run facts.
    pub fn facts(&self) -> &ReportPairRunFacts {
        &self.facts
    }

    fn finalize(
        &mut self,
        run_metadata: ReportRunMetadata,
        facts: ReportPairRunFacts,
    ) -> Result<(), ReportMetadataError> {
        if self.run_metadata.is_some() {
            return Err(ReportMetadataError::new(
                "native report run metadata is already finalized",
            ));
        }
        self.run_metadata = Some(run_metadata);
        self.facts = facts;
        Ok(())
    }
}

impl Deref for ReportRun {
    type Target = ReportRunInfo;

    fn deref(&self) -> &Self::Target {
        &self.info
    }
}

fn validate_distribution_id(value: &str) -> Result<(), ReportMetadataError> {
    let digest = value
        .strip_prefix("blake3:")
        .ok_or_else(|| ReportMetadataError::new("distribution_id must use the blake3: prefix"))?;
    if digest.len() != 64
        || !digest
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    {
        return Err(ReportMetadataError::new(
            "distribution_id must contain exactly 64 lowercase hexadecimal digits",
        ));
    }
    Ok(())
}

fn validate_component_id(value: &str, field: &str) -> Result<(), ReportMetadataError> {
    let mut bytes = value.bytes();
    if !bytes.next().is_some_and(|byte| byte.is_ascii_lowercase())
        || !bytes.all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'_')
    {
        return Err(ReportMetadataError::new(format!(
            "{field} must match [a-z][a-z0-9_]*"
        )));
    }
    Ok(())
}

fn validate_nonempty_trimmed(value: &str, field: &str) -> Result<(), ReportMetadataError> {
    if value.is_empty() || value.trim() != value {
        return Err(ReportMetadataError::new(format!(
            "{field} must be non-empty and contain no surrounding whitespace"
        )));
    }
    Ok(())
}

/// Run-level summary metadata outside the metric namespace.
#[derive(Debug, Clone, Default, PartialEq, DeriveSerialize)]
pub struct ReportSummary {
    /// First request timestamp in nanoseconds on the run timeline.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub start_time: Option<i64>,
    /// Last response timestamp in nanoseconds on the run timeline.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub end_time: Option<i64>,
    /// Observation duration in seconds.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub duration_s: Option<f64>,
    /// Whether the run was canceled.
    pub was_cancelled: bool,
    /// Configured endpoints in stable order.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub endpoints_configured: Vec<String>,
    /// Endpoints that returned successful requests.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub endpoints_successful: Vec<String>,
    /// Phase-bounded inference-server Prometheus metadata.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub server_metrics: Option<ReportServerMetricsMetadata>,
}

/// Inclusive native phase window used to aggregate server telemetry.
#[derive(Debug, Clone, PartialEq, Eq, DeriveSerialize)]
pub struct ReportServerMetricsPhaseRange {
    /// Start-boundary snapshot timestamp.
    pub start_ns: i64,
    /// End-boundary snapshot timestamp.
    pub end_ns: i64,
}

/// Fetch/update metadata for one server-metrics endpoint.
#[derive(Debug, Clone, PartialEq, DeriveSerialize)]
pub struct ReportServerMetricsEndpointInfo {
    /// Successful fetch count including duplicate bodies.
    pub total_fetches: usize,
    /// First successful fetch timestamp.
    pub first_fetch_ns: i64,
    /// Last successful fetch timestamp.
    pub last_fetch_ns: i64,
    /// Mean successful HTTP latency in milliseconds.
    pub avg_fetch_latency_ms: f64,
    /// Changed-body count.
    pub unique_updates: usize,
    /// First changed-body timestamp, or zero when absent.
    pub first_update_ns: i64,
    /// Last changed-body timestamp, or zero when absent.
    pub last_update_ns: i64,
    /// Changed-body time span in seconds.
    pub duration_seconds: f64,
    /// Mean changed-body interval in milliseconds.
    pub avg_update_interval_ms: f64,
    /// Median changed-body interval in milliseconds.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub median_update_interval_ms: Option<f64>,
}

/// Metadata needed to render server-metrics artifacts.
#[derive(Debug, Clone, Default, PartialEq, DeriveSerialize)]
pub struct ReportServerMetricsMetadata {
    /// Configured normalized endpoint URLs in stable order.
    pub endpoints_configured: Vec<String>,
    /// Endpoints contributing a complete profiling boundary pair.
    pub endpoints_successful: Vec<String>,
    /// Prometheus HELP text keyed by metric family name.
    pub descriptions: BTreeMap<String, String>,
    /// Original Prometheus semantic type keyed by metric family name.
    pub metric_types: BTreeMap<String, String>,
    /// Collection statistics keyed by credential-free endpoint URL.
    pub endpoint_info: BTreeMap<String, ReportServerMetricsEndpointInfo>,
    /// Profiling aggregation boundary.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub profiling: Option<ReportServerMetricsPhaseRange>,
    /// Warmup aggregation boundary when a warmup phase ran.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub warmup: Option<ReportServerMetricsPhaseRange>,
}

/// One grouped API error in the unified report.
#[derive(Debug, Clone, PartialEq, Eq, DeriveSerialize)]
pub struct ReportError {
    /// HTTP or application error code.
    pub code: Option<u16>,
    /// Stable error type.
    #[serde(rename = "type")]
    pub error_type: String,
    /// Representative message.
    pub message: String,
    /// Number of matching records.
    pub count: usize,
}

/// Immutable dataset identity reported by the canonical accuracy evaluator.
#[derive(Debug, Clone, PartialEq, Eq, DeriveSerialize)]
pub struct EvaluatorDatasetReportInfo {
    /// Dataset preparation implementation.
    pub provider: String,
    /// Canonical benchmark name.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub benchmark: Option<String>,
    /// Dataset repository, when applicable.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub repository: Option<String>,
    /// Dataset subset/configuration, when applicable.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub subset: Option<String>,
    /// Immutable dataset revision.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub revision: Option<String>,
    /// Evaluation splits selected by the canonical task.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub evaluation_splits: Vec<String>,
    /// Canonical task version, when exposed by the evaluator.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub task_version: Option<u64>,
}

/// Exact evaluator runtime and benchmark identity retained in an accuracy report.
#[derive(Debug, Clone, PartialEq, Eq, DeriveSerialize)]
pub struct EvaluatorReportInfo {
    /// Negotiated stdio protocol version.
    pub protocol: u32,
    /// Version of the AIPerf Python worker adapter.
    pub worker_version: String,
    /// Python runtime version.
    pub python_version: String,
    /// Python executable used for this run.
    pub python_executable: String,
    /// Evaluator package versions; absent optional packages remain null.
    pub packages: BTreeMap<String, Option<String>>,
    /// SHA-256 of the worker source.
    pub worker_source_sha256: String,
    /// SHA-256 of the fully pinned evaluator dependency lock, when available.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dependency_lock_sha256: Option<String>,
    /// Immutable worker container digest, when supplied.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub container_digest: Option<String>,
    /// Worker capabilities negotiated during initialization.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub capabilities: Vec<String>,
    /// Canonical benchmark name resolved by the worker.
    pub benchmark: String,
    /// Canonical grader or Lighteval metric implementation.
    pub grader: String,
    /// Dataset/task identity frozen by the load operation.
    pub dataset: EvaluatorDatasetReportInfo,
}

/// Closed-loop steady-state summary emitted for concurrency-target runs.
///
/// Present only when steady-state windowing is enabled and a concurrency target
/// is set. The `metrics` map uses the same [`MetricEntry`] representation as the
/// whole-run report, computed over the auto-detected saturated window.
#[derive(Debug, Clone, PartialEq, DeriveSerialize)]
pub struct ReportSteadyState {
    /// Inclusive window start in nanoseconds (first time in-flight concurrency
    /// reaches the threshold).
    pub window_start_ns: i64,
    /// Exclusive window end in nanoseconds (last time in-flight concurrency
    /// falls back below the threshold).
    pub window_end_ns: i64,
    /// Window duration in seconds.
    pub duration_s: f64,
    /// Concurrency threshold, `ceil(fraction * target_concurrency)`.
    pub threshold_concurrency: usize,
    /// Peak in-flight concurrency observed over the profiling phase.
    pub peak_concurrency: usize,
    /// True when the window is shorter than `max(10s, 10% of run duration)`.
    pub short_window: bool,
    /// Human-readable short-window warning, when one applies.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub warning: Option<String>,
    /// Steady-window metrics keyed by stable name.
    pub metrics: BTreeMap<String, MetricEntry>,
}

impl ReportSteadyState {
    /// Builds a report-shaped steady-state block from a detector outcome.
    pub fn from_outcome(outcome: &SteadyStateOutcome) -> Self {
        Self {
            window_start_ns: outcome.window.start_ns,
            window_end_ns: outcome.window.end_ns,
            duration_s: outcome.window.duration_ns() as f64 / 1e9,
            threshold_concurrency: outcome.window.threshold,
            peak_concurrency: outcome.window.peak_concurrency,
            short_window: outcome.short_window,
            warning: outcome.warning(),
            metrics: build_metric_map(&outcome.summary),
        }
    }
}

/// Runtime facts supplied to a [`Reporter`].
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RunOutcome {
    /// Run identity.
    pub run: ReportRunInfo,
    /// Summary metadata; missing timestamps/duration are filled from metrics.
    pub summary: ReportSummary,
    /// Optional warmup accumulator output.
    pub warmup: Option<AccumulatorSummary>,
    /// Profiling inference-server Prometheus series, kept outside request metrics.
    pub server_metrics: BTreeMap<String, SidecarMetric>,
    /// Warmup inference-server Prometheus series.
    pub warmup_server_metrics: BTreeMap<String, SidecarMetric>,
    /// Content-server media-fetch distributions (time_to_media_fetch, serving
    /// latency, bytes, count), keyed by metric name. Empty unless a content
    /// server served tagged media.
    pub media_metrics: BTreeMap<String, SidecarMetric>,
    /// Optional accuracy/analyzer output.
    pub accuracy: Option<AccuracyAnalysis>,
    /// Full per-request grading records in deterministic workload order.
    pub accuracy_records: Vec<AccuracyRecord>,
    /// Exact external evaluator identity for accuracy runs.
    pub evaluator: Option<EvaluatorReportInfo>,
    /// Grouped run errors.
    pub errors: Vec<ReportError>,
    /// Closed-loop steady-state summary. Present only when steady-state
    /// windowing is enabled and a concurrency target is configured.
    pub steady_state: Option<SteadyStateOutcome>,
}

/// Borrowed inputs for one IO-free native-v2 report build.
#[derive(Debug, Clone, Copy)]
pub struct NativeReportInput<'a> {
    /// Profiling accumulator summary.
    pub metrics: &'a AccumulatorSummary,
    /// Runtime facts and additive mode-specific outcomes.
    pub outcome: &'a RunOutcome,
}

/// Summary-to-report extension seam.
pub trait Reporter {
    /// Typed report produced by this reporter.
    type Output;

    /// Builds a report without performing IO.
    fn report(&self, input: NativeReportInput<'_>) -> Self::Output;
}

/// Native-v2 metrics-first reporter.
#[derive(Debug, Clone, Copy, Default)]
pub struct NativeReporter;

impl Reporter for NativeReporter {
    type Output = NativeReport;

    fn report(&self, input: NativeReportInput<'_>) -> Self::Output {
        let NativeReportInput { metrics, outcome } = input;
        let mut run_summary = outcome.summary.clone();
        if run_summary.start_time.is_none() {
            run_summary.start_time = metrics
                .finite_value(MetricTag::MinRequestTimestamp)
                .map(|value| value as i64);
        }
        if run_summary.end_time.is_none() {
            run_summary.end_time = metrics
                .finite_value(MetricTag::MaxResponseTimestamp)
                .map(|value| value as i64);
        }
        if run_summary.duration_s.is_none() {
            run_summary.duration_s = metrics.finite_value(MetricTag::BenchmarkDuration);
        }
        NativeReport {
            schema_version: NATIVE_REPORT_SCHEMA_VERSION,
            aiperf_version: env!("CARGO_PKG_VERSION").to_string(),
            run: ReportRun::unfinalized(outcome.run.clone()),
            summary: run_summary,
            metrics: build_metric_map(metrics),
            pooled_spec_decode_acceptance_histogram: metrics
                .pooled_spec_decode_acceptance_histogram()
                .cloned(),
            warmup_metrics: outcome.warmup.as_ref().map(build_metric_map),
            server_metrics: build_sidecar_map(&outcome.server_metrics),
            warmup_server_metrics: build_sidecar_map(&outcome.warmup_server_metrics),
            media_metrics: build_sidecar_map(&outcome.media_metrics),
            accuracy: outcome.accuracy.clone(),
            accuracy_records: outcome.accuracy_records.clone(),
            evaluator: outcome.evaluator.clone(),
            errors: outcome.errors.clone(),
            steady_state: outcome
                .steady_state
                .as_ref()
                .map(ReportSteadyState::from_outcome),
            // Filled by the runner after report construction (see the online
            // execution path); the aggregate reporter has no per-record samples.
            otel_per_record: None,
        }
    }
}

/// Native version-2 unified report shape.
#[derive(Debug, Clone, PartialEq, DeriveSerialize)]
pub struct NativeReport {
    /// Native report schema version.
    pub schema_version: &'static str,
    /// AIPerf package version.
    pub aiperf_version: String,
    /// Run identity.
    pub run: ReportRun,
    /// Run-level summary metadata.
    pub summary: ReportSummary,
    /// Profiling metrics keyed by stable name.
    pub metrics: BTreeMap<String, MetricEntry>,
    /// Exact accepted-draft bucket counts pooled across the selected profiling phase.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pooled_spec_decode_acceptance_histogram: Option<BTreeMap<u64, u128>>,
    /// Warmup metrics using the same representation.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub warmup_metrics: Option<BTreeMap<String, MetricEntry>>,
    /// Profiling server telemetry keyed by original Prometheus family name.
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub server_metrics: BTreeMap<String, MetricEntry>,
    /// Warmup server telemetry keyed by original Prometheus family name.
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub warmup_server_metrics: BTreeMap<String, MetricEntry>,
    /// Content-server media-fetch metrics keyed by metric name.
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub media_metrics: BTreeMap<String, MetricEntry>,
    /// Optional accuracy analysis.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub accuracy: Option<AccuracyAnalysis>,
    /// Full per-request grading records. Empty outside accuracy mode.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub accuracy_records: Vec<AccuracyRecord>,
    /// Exact canonical evaluator identity. Absent outside accuracy mode.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub evaluator: Option<EvaluatorReportInfo>,
    /// Grouped run errors.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub errors: Vec<ReportError>,
    /// Closed-loop steady-state summary. Absent unless steady-state windowing is
    /// enabled and a concurrency target is configured.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub steady_state: Option<ReportSteadyState>,
    /// Transient per-record GenAI-semconv histograms, filled by the runner's
    /// per-record path and consumed by the OTLP sink to emit populated
    /// `bucket_counts`. Never serialized into the committed native-v2 report
    /// (`#[serde(skip)]`) — it is an in-memory side channel from execution to the
    /// post-report export plane, so the authoritative report bytes are unchanged.
    #[serde(skip)]
    pub otel_per_record: Option<crate::export::otel::OtelRecordAccumulator>,
}

impl NativeReport {
    /// Builds a native report from metrics and optional accuracy analysis.
    pub fn new(metrics: &AccumulatorSummary, accuracy: Option<AccuracyAnalysis>) -> Self {
        NativeReporter.report(NativeReportInput {
            metrics,
            outcome: &RunOutcome {
                accuracy,
                ..RunOutcome::default()
            },
        })
    }

    /// Builds a native report with explicit run metadata.
    pub fn from_outcome(metrics: &AccumulatorSummary, outcome: &RunOutcome) -> Self {
        Self::from_input(NativeReportInput { metrics, outcome })
    }

    /// Builds a native report from profiling metrics and runtime facts.
    pub fn from_input(input: NativeReportInput<'_>) -> Self {
        NativeReporter.report(input)
    }

    /// Stamp coordinator-owned run metadata and pair-owned typed facts exactly
    /// once before the sole native-v2 serialization.
    ///
    /// This operates on the report model, never on serialized JSON. A second
    /// call is rejected so no downstream exporter can replace the executable
    /// identity or reinterpret pair facts after coordinator finalization.
    pub fn finalize_run(
        mut self,
        run_metadata: ReportRunMetadata,
        facts: ReportPairRunFacts,
    ) -> Result<Self, ReportMetadataError> {
        self.run.finalize(run_metadata, facts)?;
        Ok(self)
    }
}

fn build_metric_map(summary: &AccumulatorSummary) -> BTreeMap<String, MetricEntry> {
    let mut metrics = summary
        .results()
        .filter_map(|(name, result)| {
            let stats = report_stats(result, summary.result_map())?;
            let spec = result.source_tag.and_then(spec_for)?;
            let series = report_inference_series(summary, name, stats.clone());
            Some((
                name.to_string(),
                MetricEntry {
                    metric_type: stats_type(&stats),
                    unit: result.unit.clone(),
                    group: console_group_name(spec.console_group),
                    higher_is_better: spec.flags.contains(MetricFlags::LARGER_IS_BETTER),
                    series,
                },
            ))
        })
        .collect::<BTreeMap<_, _>>();
    for (name, metric) in summary.sidecar_metrics() {
        metrics
            .entry(name.clone())
            .or_insert_with(|| report_sidecar_metric(metric));
    }
    metrics
}

fn report_inference_series(
    summary: &AccumulatorSummary,
    name: &str,
    aggregate_stats: ReportStats,
) -> Vec<MetricSeries> {
    let series = summary
        .inference_series()
        .iter()
        .filter_map(|inference| {
            let result = inference.result_by_name(name)?;
            let stats = report_stats(result, inference.result_map())?;
            let dimensions = inference.dimensions();
            let labels = dimensions
                .model
                .as_ref()
                .map(|model| BTreeMap::from([("model".to_string(), model.clone())]));
            Some(MetricSeries {
                labels,
                endpoint_url: dimensions.endpoint_url.clone(),
                stats,
                timeslices: report_inference_timeslices(inference.timeslices(), name),
            })
        })
        .collect::<Vec<_>>();
    if !series.is_empty() {
        return series;
    }
    vec![MetricSeries {
        labels: None,
        endpoint_url: None,
        stats: aggregate_stats,
        timeslices: report_inference_timeslices(summary.timeslices(), name),
    }]
}

fn report_inference_timeslices(
    timeslices: &[crate::metrics_core::MetricTimeslice],
    name: &str,
) -> Vec<ReportTimeslice> {
    timeslices
        .iter()
        .filter_map(|timeslice| {
            let slice_result = timeslice.metrics.get(name)?;
            Some(ReportTimeslice {
                start_ns: timeslice.start_ns,
                end_ns: timeslice.end_ns,
                complete: timeslice.complete.unwrap_or(true),
                stats: report_stats(slice_result, &timeslice.metrics)?,
            })
        })
        .collect()
}

fn build_sidecar_map(metrics: &BTreeMap<String, SidecarMetric>) -> BTreeMap<String, MetricEntry> {
    metrics
        .iter()
        .map(|(name, metric)| (name.clone(), report_sidecar_metric(metric)))
        .collect()
}

fn report_sidecar_metric(metric: &SidecarMetric) -> MetricEntry {
    let series = metric
        .series
        .iter()
        .map(|series| MetricSeries {
            labels: series.labels.clone(),
            endpoint_url: series.endpoint_url.clone(),
            stats: report_sidecar_stats(&series.stats),
            timeslices: series
                .timeslices
                .iter()
                .map(|slice| ReportTimeslice {
                    start_ns: slice.start_ns,
                    end_ns: slice.end_ns,
                    complete: slice.complete,
                    stats: report_sidecar_stats(&slice.stats),
                })
                .collect(),
        })
        .collect::<Vec<_>>();
    let metric_type = series
        .first()
        .map(|series| stats_type(&series.stats))
        .unwrap_or("distribution");
    MetricEntry {
        metric_type,
        unit: metric
            .unit
            .map_or_else(String::new, |unit| unit.as_str().to_string()),
        group: console_group_name(metric.console_group),
        higher_is_better: metric.higher_is_better,
        series,
    }
}

fn report_sidecar_stats(stats: &SidecarStats) -> ReportStats {
    match stats {
        SidecarStats::Gauge(stats) => ReportStats::Distribution(report_distribution(stats, false)),
        SidecarStats::Counter { total, rate } => ReportStats::Counter(ReportCounterStats {
            total: report_value(*total).unwrap_or(ReportValue::NonFinite),
            rate: rate.and_then(report_value),
        }),
        SidecarStats::Histogram {
            count,
            sum,
            avg,
            count_rate,
            sum_rate,
            percentiles,
            buckets,
        } => ReportStats::Histogram(ReportHistogramStats {
            count: *count,
            sum: report_value(*sum).unwrap_or(ReportValue::NonFinite),
            avg: avg.and_then(report_value),
            count_rate: count_rate.and_then(report_value),
            sum_rate: sum_rate.and_then(report_value),
            percentiles: percentiles
                .iter()
                .filter_map(|(percentile, value)| {
                    report_value(*value).map(|value| (format!("p{percentile}"), value))
                })
                .collect(),
            buckets: buckets.clone(),
        }),
    }
}

fn report_distribution(
    stats: &crate::metrics_core::DistributionStats,
    adjusted: bool,
) -> ReportDistributionStats {
    ReportDistributionStats {
        count: (stats.count > 0).then_some(stats.count),
        avg: report_value(stats.avg),
        min: report_value(stats.min),
        max: report_value(stats.max),
        std: stats
            .std
            .map(ReportValue::Finite)
            .or(adjusted.then_some(ReportValue::NonFinite)),
        percentiles: stats
            .percentiles
            .iter()
            .filter_map(|(percentile, value)| {
                report_value(*value).map(|value| (format!("p{percentile}"), value))
            })
            .collect(),
    }
}

fn report_stats(
    result: &MetricResult,
    all_results: &BTreeMap<String, MetricResult>,
) -> Option<ReportStats> {
    match &result.data {
        MetricResultData::Distribution(stats) => {
            let adjusted = result.tag.starts_with("adj_");
            Some(ReportStats::Distribution(report_distribution(
                stats, adjusted,
            )))
        }
        MetricResultData::Scalar { value } => {
            let value = report_value(*value)?;
            let spec = result.source_tag.and_then(spec_for)?;
            if spec.kind == MetricType::Aggregate && spec.aggregation == Some(AggregationKind::Sum)
            {
                let rate = counter_rate(spec.tag)
                    .and_then(|tag| all_results.get(tag.as_str()))
                    .and_then(|result| report_value(result.representative_value()));
                Some(ReportStats::Counter(ReportCounterStats {
                    total: value,
                    rate,
                }))
            } else {
                Some(ReportStats::Scalar(ReportScalarStats { value }))
            }
        }
    }
}

fn report_value(value: MetricValue) -> Option<ReportValue> {
    match value {
        MetricValue::Finite(value) if value.is_finite() => Some(ReportValue::Finite(value)),
        MetricValue::PosInf => Some(ReportValue::NonFinite),
        MetricValue::Finite(_) | MetricValue::Absent => None,
    }
}

fn counter_rate(tag: MetricTag) -> Option<MetricTag> {
    match tag {
        MetricTag::RequestCount => Some(MetricTag::RequestThroughput),
        MetricTag::GoodRequestCount => Some(MetricTag::Goodput),
        _ => None,
    }
}

fn stats_type(stats: &ReportStats) -> &'static str {
    match stats {
        ReportStats::Distribution(_) => "distribution",
        ReportStats::Scalar(_) => "scalar",
        ReportStats::Counter(_) => "counter",
        ReportStats::Histogram(_) => "histogram",
    }
}

fn console_group_name(group: MetricConsoleGroup) -> &'static str {
    match group {
        MetricConsoleGroup::None => "none",
        MetricConsoleGroup::Default => "default",
        MetricConsoleGroup::Usage => "usage",
        MetricConsoleGroup::Cache => "cache",
        MetricConsoleGroup::Prediction => "prediction",
        MetricConsoleGroup::Audio => "audio",
        MetricConsoleGroup::Reasoning => "reasoning",
        MetricConsoleGroup::SpecDecode => "spec_decode",
        MetricConsoleGroup::Effective => "effective",
        MetricConsoleGroup::Active => "active",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dispatch::sink::ObservedSpecDecodeAcceptance;
    use crate::metrics_core::{
        InferenceDimensions, MetricResult, MetricResultData, MetricsAccumulator, MetricsConfig,
        Phase, RecordIngest, SidecarMetric, SidecarSeries, SidecarStats, Unit,
    };

    #[test]
    fn native_report_retains_the_full_spec_decode_histogram() {
        let mut accumulator = MetricsAccumulator::new();
        let mut record = RecordIngest::minimal(0, 1, Phase::Profiling);
        record.spec_decode_acceptance = Some(ObservedSpecDecodeAcceptance {
            engine: "vllm".to_string(),
            mean_acceptance_length: 3.25,
            draft_acceptance_rate: 0.5625,
            acceptance_histogram: BTreeMap::from([(0, 1), (1, 1), (2, 2), (3, 3), (4, 1)]),
            num_accepted_draft_tokens: 18,
            num_draft_tokens: 32,
            num_spec_steps: 8,
            num_spec_tokens: Some(4),
            completion_tokens: Some(26),
            per_step_accepted: None,
            per_step_drafted: None,
        });
        accumulator.process_record(&record);

        let report = NativeReport::new(&accumulator.summarize(), None);
        assert_eq!(
            report.pooled_spec_decode_acceptance_histogram,
            Some(BTreeMap::from([(0, 1), (1, 1), (2, 2), (3, 3), (4, 1)]))
        );
        let serialized = serde_json::to_value(&report).unwrap();
        assert_eq!(
            serialized["pooled_spec_decode_acceptance_histogram"]["3"],
            3
        );
    }

    #[test]
    fn v2_uses_type_specific_series_and_null_for_non_finite_tail() {
        let mut summary = AccumulatorSummary::new();
        summary.insert_finite(MetricTag::RequestCount, 2.0);
        summary.insert_finite(MetricTag::RequestThroughput, 4.0);
        let mut percentiles = BTreeMap::new();
        percentiles.insert(50, MetricValue::Finite(10.0));
        percentiles.insert(99, MetricValue::PosInf);
        summary.insert_result(MetricResult {
            tag: "adj_request_latency".to_string(),
            source_tag: Some(MetricTag::RequestLatency),
            header: "Request Latency (error-adjusted)".to_string(),
            unit: "ms".to_string(),
            console_group: MetricConsoleGroup::Default,
            data: MetricResultData::Distribution(crate::metrics_core::DistributionStats {
                tag: "adj_request_latency".to_string(),
                avg: MetricValue::PosInf,
                min: MetricValue::Finite(10.0),
                max: MetricValue::PosInf,
                std: None,
                sum: MetricValue::PosInf,
                count: 2,
                percentiles,
            }),
        });

        let report = NativeReport::new(&summary, None);
        let serialized = serde_json::to_string_pretty(&report).unwrap();
        assert_eq!(
            serialized,
            include_str!("../tests/golden/native_v2.json").trim_end()
        );
        let value = serde_json::to_value(report).unwrap();
        assert_eq!(value["schema_version"], "2.0");
        assert_eq!(value["metrics"]["request_count"]["type"], "counter");
        assert_eq!(
            value["metrics"]["request_count"]["series"][0]["stats"]["total"],
            2.0
        );
        assert_eq!(
            value["metrics"]["request_count"]["series"][0]["stats"]["rate"],
            4.0
        );
        assert_eq!(
            value["metrics"]["adj_request_latency"]["type"],
            "distribution"
        );
        assert!(value["metrics"]["adj_request_latency"]["series"][0]["stats"]["avg"].is_null());
        assert!(
            value["metrics"]["adj_request_latency"]["series"][0]["stats"]["percentiles"]["p99"]
                .is_null()
        );
        assert!(value.get("warmup_metrics").is_none());
        assert!(value.get("accuracy").is_none());
        assert!(value.get("accuracy_records").is_none());
        assert!(value.get("evaluation").is_none());
        assert!(value["run"].get("distribution_id").is_none());
        assert!(value["run"].get("graph").is_none());
        assert!(value["run"].get("dynamo").is_none());
    }

    #[test]
    fn websocket_lag_metrics_are_optional_native_distributions() {
        let mut accumulator = MetricsAccumulator::new();
        let mut measured = RecordIngest::minimal(0, 500_000_000, Phase::Profiling);
        measured.metric_overrides = vec![
            (
                MetricTag::TimeToLastRoundTrip,
                MetricValue::Finite(300_000_000.0),
            ),
            (
                MetricTag::AverageRoundTripTime,
                MetricValue::Finite(250_000_000.5),
            ),
        ];
        accumulator.process_record(&measured);
        accumulator.process_record(&RecordIngest::minimal(
            600_000_000,
            700_000_000,
            Phase::Profiling,
        ));

        let value =
            serde_json::to_value(NativeReport::new(&accumulator.summarize(), None)).unwrap();
        let last = &value["metrics"]["time_to_last_round_trip"];
        let average = &value["metrics"]["avg_round_trip_time"];
        assert_eq!(last["unit"], "ms");
        assert_eq!(last["series"][0]["stats"]["count"], 1);
        assert_eq!(last["series"][0]["stats"]["avg"], 300.0);
        assert_eq!(average["series"][0]["stats"]["count"], 1);
        assert_eq!(average["series"][0]["stats"]["avg"], 250.0000005);

        let empty =
            serde_json::to_value(NativeReport::new(&AccumulatorSummary::new(), None)).unwrap();
        assert!(empty["metrics"].get("time_to_last_round_trip").is_none());
        assert!(empty["metrics"].get("avg_round_trip_time").is_none());
    }

    #[test]
    fn coordinator_finalization_flattens_common_and_pair_facts_into_run() {
        let run_metadata = ReportRunMetadata::new(
            format!("blake3:{}", "a".repeat(64)),
            "dynosim",
            "graph",
            vec![
                ReportExtensionIdentity::new("zeta", Some("2.0.0".into())).unwrap(),
                ReportExtensionIdentity::new("alpha", None).unwrap(),
            ],
            vec![
                ReportEndpointProfileIdentity::new("primary", "messages").unwrap(),
                ReportEndpointProfileIdentity::new("judge", "chat").unwrap(),
            ],
        )
        .unwrap();
        let graph = ReportGraphRunInfo::new("dag_jsonl", 3, 9, 1, 2)
            .unwrap()
            .with_outcome(ReportGraphOutcomeInfo::new(6, 5, 1))
            .unwrap();
        let parity = ReportDynamoParityInfo::new(74, 69, 5, 4_096).unwrap();
        let capacity = ReportDynamoCapacityInfo::new(2.5, 7.5, 2, 4, 0.009_027_777).unwrap();
        let dynamo = ReportDynamoRunInfo::new(
            ReportClockKind::Sim,
            ReportDynamoTopology::Disaggregated,
            ReportDynamoRouter::Kv,
            vec!["dynamo-profile".into(), "dynamo-kvbm-offload".into()],
            1,
            2,
            4,
            parity,
        )
        .unwrap()
        .with_capacity(capacity);
        let facts = ReportPairRunFacts::new()
            .with_graph(graph)
            .with_dynamo(dynamo);
        let report = NativeReport::new(&AccumulatorSummary::new(), None)
            .finalize_run(run_metadata, facts)
            .unwrap();

        let value = serde_json::to_value(&report).unwrap();
        let run = &value["run"];
        assert_eq!(run["distribution_id"], format!("blake3:{}", "a".repeat(64)));
        assert_eq!(run["transport"], "dynosim");
        assert_eq!(run["workload"], "graph");
        assert_eq!(run["extensions"][0]["name"], "alpha");
        assert!(run["extensions"][0].get("version").is_none());
        assert_eq!(run["extensions"][1]["version"], "2.0.0");
        assert_eq!(run["endpoint_profiles"][0]["profile_id"], "primary");
        assert_eq!(run["endpoint_profiles"][1]["endpoint_id"], "chat");
        assert_eq!(run["graph"]["input_format"], "dag_jsonl");
        assert_eq!(run["graph"]["outcome"]["completed"], 5);
        assert_eq!(run["dynamo"]["topology"], "disaggregated");
        assert_eq!(run["dynamo"]["router"], "kv");
        assert_eq!(
            run["dynamo"]["required_features"],
            serde_json::json!(["dynamo-kvbm-offload", "dynamo-profile"])
        );
        assert_eq!(run["dynamo"]["parity"]["shared_fields"], 74);
        assert_eq!(run["dynamo"]["capacity"]["prefill_worker_seconds"], 2.5);
        assert!(run.get("evaluation_compatibility").is_none());
        assert_eq!(
            report.run.run_metadata().unwrap().distribution_id,
            format!("blake3:{}", "a".repeat(64))
        );
        assert!(report.run.is_finalized());
    }

    #[test]
    fn coordinator_run_metadata_rejects_ambiguous_or_inexact_identity() {
        let endpoint = ReportEndpointProfileIdentity::new("default", "chat").unwrap();
        let invalid_digest = ReportRunMetadata::new(
            "blake3:abc",
            "online_http",
            "scheduled",
            Vec::new(),
            vec![endpoint.clone()],
        )
        .unwrap_err();
        assert!(invalid_digest.to_string().contains("64 lowercase"));

        let duplicate_profile = ReportRunMetadata::new(
            format!("blake3:{}", "b".repeat(64)),
            "online_http",
            "scheduled",
            Vec::new(),
            vec![endpoint.clone(), endpoint],
        )
        .unwrap_err();
        assert!(
            duplicate_profile
                .to_string()
                .contains("duplicate endpoint profile")
        );

        let duplicate_extension = ReportRunMetadata::new(
            format!("blake3:{}", "b".repeat(64)),
            "online_http",
            "scheduled",
            vec![
                ReportExtensionIdentity::new("same", None).unwrap(),
                ReportExtensionIdentity::new("same", Some("2".into())).unwrap(),
            ],
            vec![],
        )
        .unwrap_err();
        assert!(
            duplicate_extension
                .to_string()
                .contains("duplicate linked extension")
        );
    }

    #[test]
    fn pair_facts_reject_non_finite_capacity_and_inconsistent_counts() {
        assert!(ReportDynamoCapacityInfo::new(f64::NAN, 1.0, 1, 1, 1.0).is_err());
        assert!(ReportDynamoCapacityInfo::new(1.0, 1.0, 1, 1, f64::INFINITY).is_err());
        assert!(ReportDynamoParityInfo::new(74, 68, 5, 100).is_err());
        assert!(
            ReportGraphRunInfo::new("dag_jsonl", 1, 1, 1, 1)
                .unwrap()
                .with_outcome(ReportGraphOutcomeInfo::new(1, 1, 1))
                .is_err()
        );
    }

    #[test]
    fn report_run_metadata_can_only_be_finalized_once() {
        let run_metadata = || {
            ReportRunMetadata::new(
                format!("blake3:{}", "c".repeat(64)),
                "online_http",
                "scheduled",
                Vec::new(),
                Vec::new(),
            )
            .unwrap()
        };
        let report = NativeReport::new(&AccumulatorSummary::new(), None)
            .finalize_run(run_metadata(), ReportPairRunFacts::new())
            .unwrap();
        let error = report
            .finalize_run(run_metadata(), ReportPairRunFacts::new())
            .unwrap_err();
        assert!(error.to_string().contains("already finalized"));
    }

    #[test]
    fn v2_retains_labeled_endpoint_histogram_sidecars() {
        let mut summary = AccumulatorSummary::new();
        summary.insert_sidecar_metric(
            "vllm:request_latency_seconds",
            SidecarMetric::new(
                Some(Unit::Second),
                vec![SidecarSeries {
                    labels: Some(BTreeMap::from([("model".to_string(), "m".to_string())])),
                    endpoint_url: Some("http://server/metrics".to_string()),
                    stats: SidecarStats::Histogram {
                        count: 2,
                        sum: MetricValue::Finite(0.3),
                        avg: Some(MetricValue::Finite(0.15)),
                        count_rate: Some(MetricValue::Finite(2.0)),
                        sum_rate: Some(MetricValue::Finite(0.3)),
                        percentiles: BTreeMap::from([(99, MetricValue::Finite(0.2))]),
                        buckets: BTreeMap::from([("0.1".to_string(), 1), ("+Inf".to_string(), 2)]),
                    },
                    timeslices: Vec::new(),
                }],
            ),
        );

        let value = serde_json::to_value(NativeReport::new(&summary, None)).unwrap();
        let metric = &value["metrics"]["vllm:request_latency_seconds"];
        assert_eq!(metric["type"], "histogram");
        assert_eq!(metric["unit"], "sec");
        assert_eq!(metric["series"][0]["endpoint_url"], "http://server/metrics");
        assert_eq!(metric["series"][0]["labels"]["model"], "m");
        assert_eq!(metric["series"][0]["stats"]["percentiles"]["p99"], 0.2);
        assert_eq!(metric["series"][0]["stats"]["buckets"]["+Inf"], 2);
    }

    #[test]
    fn v2_inference_series_are_endpoint_model_sorted_with_owned_timeslices() {
        let mut accumulator = MetricsAccumulator::with_config(MetricsConfig {
            slice_duration_ns: Some(1_000_000_000),
            ..MetricsConfig::default()
        });
        let mut endpoint_z = RecordIngest::minimal(100_000_000, 200_000_000, Phase::Profiling);
        endpoint_z.dimensions = InferenceDimensions {
            endpoint_url: Some("https://endpoint-z/v1/chat/completions".to_string()),
            model: Some("model-b".to_string()),
        };
        let mut endpoint_a = RecordIngest::minimal(300_000_000, 500_000_000, Phase::Profiling);
        endpoint_a.dimensions = InferenceDimensions {
            endpoint_url: Some("https://endpoint-a/v1/chat/completions".to_string()),
            model: Some("model-a".to_string()),
        };
        // Deliberately ingest reverse lexical order: report ordering is a value
        // contract, not an insertion/worker completion accident.
        accumulator.process_record(&endpoint_z);
        accumulator.process_record(&endpoint_a);

        let report = NativeReport::new(&accumulator.summarize(), None);
        let serialized = serde_json::to_string_pretty(&report.metrics["request_count"]).unwrap();
        assert_eq!(
            serialized,
            include_str!("../tests/golden/native_v2_inference_series.json").trim_end()
        );
        let value = serde_json::to_value(report).unwrap();
        let latency = &value["metrics"]["request_latency"]["series"];
        assert_eq!(latency[0]["stats"]["avg"], 200.0);
        assert_eq!(latency[1]["stats"]["avg"], 100.0);
        assert_eq!(latency[0]["timeslices"][0]["stats"]["avg"], 200.0);
        assert_eq!(latency[1]["timeslices"][0]["stats"]["avg"], 100.0);
    }
}
