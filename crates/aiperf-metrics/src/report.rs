// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Typed, deterministic native-v2 report construction.
//!
//! This module is IO-free. It translates an accumulator summary into the
//! metrics-first, type-specific-series representation; application-layer
//! exporters decide where to write it.

use crate::catalog::{
    AggregationKind, MetricConsoleGroup, MetricFlags, MetricTag, MetricType, spec_for,
};
use crate::{
    AccumulatorSummary, AccuracyAnalysis, AccuracyRecord, MetricResult, MetricResultData,
    MetricValue, SidecarMetric, SidecarStats,
};
use serde::ser::{Serialize, Serializer};
use serde::{Deserialize as DeriveDeserialize, Serialize as DeriveSerialize};
use serde_json::Value;
use std::collections::BTreeMap;
use std::error::Error;
use std::fmt::{self, Display};
use std::ops::Deref;

/// Native report schema identifier.
pub const NATIVE_REPORT_SCHEMA_VERSION: &str = "2.0";

/// Schema identifier for the additive native-v2 telemetry-archive block.
pub const TELEMETRY_ARCHIVE_REPORT_SCHEMA_VERSION: &str = "1.0";

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

/// A finite numeric fact carried by report provenance.
///
/// Unlike request-metric tails, provenance cannot use JSON null as a numeric
/// sentinel: a backend either supplies an exact finite fact or omits it. The
/// private representation makes that invariant hold for every serialized
/// value, including values assembled by external runner distributions.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct FiniteReportValue(f64);

impl FiniteReportValue {
    /// Validate and retain one finite value.
    pub fn new(value: f64) -> Result<Self, ReportProvenanceError> {
        if value.is_finite() {
            Ok(Self(value))
        } else {
            Err(ReportProvenanceError::new(
                "report provenance numeric values must be finite",
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

/// Invalid typed common or pair-specific report provenance.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReportProvenanceError {
    message: String,
}

impl ReportProvenanceError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl Display for ReportProvenanceError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl Error for ReportProvenanceError {}

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
    ) -> Result<Self, ReportProvenanceError> {
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
    ) -> Result<Self, ReportProvenanceError> {
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
pub struct ReportRunProvenance {
    /// BLAKE3 identity of the exact runner executable that performed the run.
    pub distribution_id: String,
    /// Canonical selected backend factory ID.
    pub backend: String,
    /// Canonical selected workload factory ID.
    pub workload: String,
    /// Statically linked extensions in deterministic identity order.
    pub extensions: Vec<ReportExtensionIdentity>,
    /// Endpoint profiles in authored order after canonical alias resolution.
    pub endpoint_profiles: Vec<ReportEndpointProfileIdentity>,
}

impl ReportRunProvenance {
    /// Build the complete coordinator-owned common provenance block.
    pub fn new(
        distribution_id: impl Into<String>,
        backend: impl Into<String>,
        workload: impl Into<String>,
        mut extensions: Vec<ReportExtensionIdentity>,
        endpoint_profiles: Vec<ReportEndpointProfileIdentity>,
    ) -> Result<Self, ReportProvenanceError> {
        let distribution_id = distribution_id.into();
        validate_distribution_id(&distribution_id)?;
        let backend = backend.into();
        validate_component_id(&backend, "backend ID")?;
        let workload = workload.into();
        validate_component_id(&workload, "workload ID")?;

        extensions
            .sort_by(|left, right| (&left.name, &left.version).cmp(&(&right.name, &right.version)));
        for adjacent in extensions.windows(2) {
            if adjacent[0].name == adjacent[1].name {
                return Err(ReportProvenanceError::new(format!(
                    "duplicate linked extension {:?}",
                    adjacent[0].name
                )));
            }
        }
        let mut profile_ids = std::collections::BTreeSet::new();
        for profile in &endpoint_profiles {
            if !profile_ids.insert(profile.profile_id.as_str()) {
                return Err(ReportProvenanceError::new(format!(
                    "duplicate endpoint profile ID {:?}",
                    profile.profile_id
                )));
            }
        }

        Ok(Self {
            distribution_id,
            backend,
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
    ) -> Result<Self, ReportProvenanceError> {
        let input_format = input_format.into();
        validate_component_id(&input_format, "graph input format")?;
        if root_count == 0 {
            return Err(ReportProvenanceError::new(
                "graph root_count must be positive",
            ));
        }
        if node_count < root_count {
            return Err(ReportProvenanceError::new(
                "graph node_count cannot be smaller than root_count",
            ));
        }
        if worker_count == 0 {
            return Err(ReportProvenanceError::new(
                "graph worker_count must be positive",
            ));
        }
        if phase_count == 0 {
            return Err(ReportProvenanceError::new(
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
    ) -> Result<Self, ReportProvenanceError> {
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

    fn validate(&self) -> Result<(), ReportProvenanceError> {
        let terminal = self.completed.checked_add(self.failed).ok_or_else(|| {
            ReportProvenanceError::new("graph terminal trace count overflowed u64")
        })?;
        if terminal > self.admitted {
            return Err(ReportProvenanceError::new(
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

/// Whole-summary byte-parity proof produced by the Dynamo offline backend.
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
    ) -> Result<Self, ReportProvenanceError> {
        if independently_accumulated_fields.checked_add(backend_owned_fields) != Some(shared_fields)
        {
            return Err(ReportProvenanceError::new(
                "Dynamo parity fields must partition shared_fields exactly",
            ));
        }
        if serialized_bytes == 0 {
            return Err(ReportProvenanceError::new(
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
    ) -> Result<Self, ReportProvenanceError> {
        for (name, value) in [
            ("prefill_worker_seconds", prefill_worker_seconds),
            ("decode_worker_seconds", decode_worker_seconds),
            ("gpu_hours", gpu_hours),
        ] {
            if value < 0.0 {
                return Err(ReportProvenanceError::new(format!(
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
    ) -> Result<Self, ReportProvenanceError> {
        // Both clocks are valid Dynamo engine report axes: `Sim` for
        // deterministic virtual-clock replay and `Real` for the wall-clock
        // in-process online mode (`--replay-mode online`). No further constraint
        // is imposed here; the enum is exhaustive.
        if workers == 0 || prefill_workers == 0 || decode_workers == 0 {
            return Err(ReportProvenanceError::new(
                "Dynamo worker counts must be positive",
            ));
        }
        required_features.sort();
        for feature in &required_features {
            validate_nonempty_trimmed(feature, "Dynamo required feature")?;
        }
        if required_features.windows(2).any(|pair| pair[0] == pair[1]) {
            return Err(ReportProvenanceError::new(
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

/// Exact secret-free post-plan authority for a compatibility proxy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, DeriveSerialize)]
pub struct ReportEvaluationCompatibilityGrantLimits {
    pub max_operations: u64,
    pub max_concurrent_operations: u64,
    pub max_request_bytes: u64,
    pub max_response_bytes: u64,
    pub max_stream_events: u64,
    pub expires_after_ms: u64,
}

impl ReportEvaluationCompatibilityGrantLimits {
    /// Reject an empty or internally inconsistent effective grant.
    pub fn validate(self) -> Result<Self, ReportProvenanceError> {
        if self.max_operations == 0
            || self.max_concurrent_operations == 0
            || self.max_concurrent_operations > self.max_operations
            || self.max_request_bytes == 0
            || self.max_response_bytes == 0
            || self.expires_after_ms == 0
        {
            return Err(ReportProvenanceError::new(
                "evaluation compatibility effective grant was empty or inconsistent",
            ));
        }
        Ok(self)
    }
}

/// Exact local adapter-policy identity for one proxy-enabled evaluation run.
#[derive(Debug, Clone, PartialEq, Eq, DeriveSerialize)]
pub struct ReportEvaluationCompatibilityInfo {
    /// Sorted frozen compatibility-dialect factory IDs.
    pub dialect_ids: Vec<String>,
    /// SHA-256 of exact dialect IDs, allowed routes, effective adapters, and grant limits.
    pub descriptor_sha256: String,
    /// Exact post-plan authority installed in both worker binding and Rust runtime.
    pub effective_grant: ReportEvaluationCompatibilityGrantLimits,
}

impl ReportEvaluationCompatibilityInfo {
    /// Validate and freeze one proxy-enabled evaluation compatibility identity.
    pub fn new(
        mut dialect_ids: Vec<String>,
        descriptor_sha256: impl Into<String>,
        effective_grant: ReportEvaluationCompatibilityGrantLimits,
    ) -> Result<Self, ReportProvenanceError> {
        if dialect_ids.is_empty() {
            return Err(ReportProvenanceError::new(
                "evaluation compatibility dialect IDs cannot be empty",
            ));
        }
        dialect_ids.sort();
        for dialect in &dialect_ids {
            validate_component_id(dialect, "evaluation compatibility dialect ID")?;
        }
        if dialect_ids.windows(2).any(|pair| pair[0] == pair[1]) {
            return Err(ReportProvenanceError::new(
                "evaluation compatibility dialect IDs must be unique",
            ));
        }
        let descriptor_sha256 = descriptor_sha256.into();
        validate_lowercase_sha256(
            &descriptor_sha256,
            "evaluation compatibility descriptor SHA-256",
        )?;
        let effective_grant = effective_grant.validate()?;
        Ok(Self {
            dialect_ids,
            descriptor_sha256,
            effective_grant,
        })
    }
}

/// Optional typed facts returned by one backend/workload pair.
///
/// The pair owns only these mode-specific facts. Common executable, registry,
/// and endpoint identity remains coordinator-owned in [`ReportRunProvenance`].
#[derive(Debug, Clone, Default, PartialEq, DeriveSerialize)]
pub struct ReportPairRunFacts {
    /// Graph-IR lowering, placement, and terminal counts.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub graph: Option<ReportGraphRunInfo>,
    /// Dynamo topology, capacity, and exact metric-parity evidence.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dynamo: Option<ReportDynamoRunInfo>,
    /// Exact local compatibility policy for a proxy-enabled evaluator run.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub evaluation_compatibility: Option<ReportEvaluationCompatibilityInfo>,
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

    /// Attach exact evaluator compatibility policy identity.
    pub fn with_evaluation_compatibility(
        mut self,
        compatibility: ReportEvaluationCompatibilityInfo,
    ) -> Self {
        self.evaluation_compatibility = Some(compatibility);
        self
    }
}

/// Serialized native-v2 run block.
///
/// The wrapper preserves the original `mode`/`model` Rust value through
/// [`Deref`] while flattening additive protocol-v2 provenance into the same
/// JSON object required by the native report contract.
#[derive(Debug, Clone, Default, PartialEq, DeriveSerialize)]
pub struct ReportRun {
    /// Workload-facing run identity retained by existing report producers.
    #[serde(flatten)]
    pub info: ReportRunInfo,
    /// Coordinator-owned exact executable and registry identity.
    #[serde(flatten, skip_serializing_if = "Option::is_none")]
    provenance: Option<ReportRunProvenance>,
    /// Pair-owned typed mode facts.
    #[serde(flatten)]
    facts: ReportPairRunFacts,
}

impl ReportRun {
    fn unfinalized(info: ReportRunInfo) -> Self {
        Self {
            info,
            provenance: None,
            facts: ReportPairRunFacts::default(),
        }
    }

    /// Whether the coordinator has stamped exact protocol-v2 provenance.
    pub fn is_finalized(&self) -> bool {
        self.provenance.is_some()
    }

    /// Borrow coordinator-owned common provenance after finalization.
    pub fn provenance(&self) -> Option<&ReportRunProvenance> {
        self.provenance.as_ref()
    }

    /// Borrow pair-owned typed run facts.
    pub fn facts(&self) -> &ReportPairRunFacts {
        &self.facts
    }

    fn finalize(
        &mut self,
        provenance: ReportRunProvenance,
        facts: ReportPairRunFacts,
    ) -> Result<(), ReportProvenanceError> {
        if self.provenance.is_some() {
            return Err(ReportProvenanceError::new(
                "native report run provenance is already finalized",
            ));
        }
        self.provenance = Some(provenance);
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

fn validate_distribution_id(value: &str) -> Result<(), ReportProvenanceError> {
    let digest = value
        .strip_prefix("blake3:")
        .ok_or_else(|| ReportProvenanceError::new("distribution_id must use the blake3: prefix"))?;
    if digest.len() != 64
        || !digest
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    {
        return Err(ReportProvenanceError::new(
            "distribution_id must contain exactly 64 lowercase hexadecimal digits",
        ));
    }
    Ok(())
}

fn validate_component_id(value: &str, field: &str) -> Result<(), ReportProvenanceError> {
    let mut bytes = value.bytes();
    if !bytes.next().is_some_and(|byte| byte.is_ascii_lowercase())
        || !bytes.all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'_')
    {
        return Err(ReportProvenanceError::new(format!(
            "{field} must match [a-z][a-z0-9_]*"
        )));
    }
    Ok(())
}

fn validate_lowercase_sha256(value: &str, field: &str) -> Result<(), ReportProvenanceError> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    {
        return Err(ReportProvenanceError::new(format!(
            "{field} must contain exactly 64 lowercase hexadecimal digits"
        )));
    }
    Ok(())
}

fn validate_nonempty_trimmed(value: &str, field: &str) -> Result<(), ReportProvenanceError> {
    if value.is_empty() || value.trim() != value {
        return Err(ReportProvenanceError::new(format!(
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

/// Metadata needed to render canonical Python server-metrics artifacts.
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

/// Exact stateful harness identity retained beside the generic worker identity.
#[derive(Debug, Clone, PartialEq, Eq, DeriveSerialize)]
pub struct AgenticEvaluatorReportInfo {
    /// Canonical harness name.
    pub harness: String,
    /// Exact harness package version.
    pub harness_version: String,
    /// SHA-256 over the installed harness sources.
    pub harness_source_sha256: String,
    /// Agent scaffold name.
    pub agent: String,
    /// Exact adapter and inherited scaffold version.
    pub agent_version: String,
    /// Provider-owned canonical agent controls used for the evaluation.
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub canonical_agent_config: BTreeMap<String, Value>,
    /// Environment provider used for task sandboxes.
    pub environment: String,
    /// Canonical verifier implementation description.
    pub verifier: String,
}

/// Reproducibility-relevant configuration for one agentic evaluation.
#[derive(Debug, Clone, PartialEq, Eq, DeriveSerialize)]
pub struct AgenticRunConfigReport {
    /// Requested immutable Harbor package or local dataset path.
    pub dataset: String,
    /// Optional exact task names selected from the package.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub task_names: Option<Vec<String>>,
    /// Optional deterministic episode cap.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_episodes: Option<usize>,
    /// Maximum simultaneously active task environments.
    pub task_concurrency: usize,
    /// Maximum simultaneously active model calls.
    pub model_concurrency: usize,
    /// Harness artifact root.
    pub output_dir: String,
    /// Optional model-call limit per episode.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_turns: Option<usize>,
    /// Maximum generated tokens per model call.
    pub max_tokens: usize,
    /// Explicit context-window limit used by the agent scaffold.
    pub context_window: usize,
    /// Canonical agent command parser.
    pub parser: String,
    /// Whether canonical context summarization was enabled.
    pub enable_summarize: bool,
    /// Optional explicitly selected primary verifier reward.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub primary_reward: Option<String>,
    /// Whether cached task packages could be replaced.
    pub overwrite: bool,
    /// Rust-owned callback ingress advertised to evaluator environments.
    ///
    /// The per-run bearer credential is intentionally never reported.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub inference_gateway_base_url: Option<String>,
}

/// Generic aggregate statistics over one canonical verifier reward.
#[derive(Debug, Clone, PartialEq, DeriveSerialize)]
pub struct AgenticRewardSummary {
    /// Number of completed episodes reporting this reward.
    pub n: usize,
    /// Arithmetic mean over canonical verifier values.
    pub avg: f64,
    /// Minimum canonical verifier value.
    pub min: f64,
    /// Maximum canonical verifier value.
    pub max: f64,
}

/// Run-level agentic result summary.
#[derive(Debug, Clone, PartialEq, DeriveSerialize)]
pub struct AgenticEvaluationSummary {
    /// Every selected episode, regardless of terminal class.
    pub episode_count: usize,
    /// Episodes that reached canonical verification.
    pub completed_count: usize,
    /// Episodes that failed in inference, environment, harness, or verification infrastructure.
    pub infrastructure_error_count: usize,
    /// Episodes explicitly cancelled by Rust policy.
    pub cancelled_count: usize,
    /// All primary, environment, and verifier calls dispatched by Rust.
    pub model_calls: usize,
    /// Canonical agent calls emitted through the evaluator protocol.
    pub primary_model_calls: usize,
    /// Calls requested by task environments and canonical verifiers.
    pub auxiliary_model_calls: usize,
    /// Auxiliary calls requested by task environments.
    pub environment_model_calls: usize,
    /// Auxiliary calls requested by canonical verifiers.
    pub verifier_model_calls: usize,
    /// Prompt tokens across all calls when every call reported usage.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_tokens: Option<u64>,
    /// Completion tokens across all calls when every call reported usage.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completion_tokens: Option<u64>,
    /// Cached tokens across all calls when every call reported usage.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cached_tokens: Option<u64>,
    /// Prompt tokens from canonical agent calls only.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub primary_prompt_tokens: Option<u64>,
    /// Completion tokens from canonical agent calls only.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub primary_completion_tokens: Option<u64>,
    /// Cached tokens from canonical agent calls only.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub primary_cached_tokens: Option<u64>,
    /// Prompt tokens from environment and verifier calls only.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub auxiliary_prompt_tokens: Option<u64>,
    /// Completion tokens from environment and verifier calls only.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub auxiliary_completion_tokens: Option<u64>,
    /// Cached tokens from environment and verifier calls only.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub auxiliary_cached_tokens: Option<u64>,
    /// Uniform primary reward selected for the run, when available.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub primary_reward: Option<String>,
    /// Mean primary reward over completed episodes only.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub primary_score: Option<f64>,
    /// Canonical reward aggregates keyed by verifier-owned name.
    pub rewards: BTreeMap<String, AgenticRewardSummary>,
}

/// Terminal class for one report-safe agentic episode record.
#[derive(Debug, Clone, Copy, PartialEq, Eq, DeriveSerialize)]
#[serde(rename_all = "snake_case")]
pub enum AgenticEpisodeReportOutcome {
    /// The canonical verifier returned rewards.
    Completed,
    /// Inference, environment, harness, or verifier infrastructure failed.
    InfrastructureError,
    /// Rust policy cancelled the episode.
    Cancelled,
}

/// Full canonical result for one opaque agentic episode.
#[derive(Debug, Clone, PartialEq, DeriveSerialize)]
pub struct AgenticEpisodeReport {
    /// Opaque evaluator-owned episode identifier.
    pub episode_id: String,
    /// Canonical task label.
    pub task: String,
    /// Explicit terminal classification.
    pub outcome: AgenticEpisodeReportOutcome,
    /// Finite verifier rewards, empty for non-completed episodes.
    pub rewards: BTreeMap<String, f64>,
    /// Per-episode selected primary reward.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub primary_reward: Option<String>,
    /// End-to-end harness wall time.
    pub duration_seconds: f64,
    /// Number of Rust-owned inference calls.
    pub model_calls: usize,
    /// Canonical agent calls emitted through the evaluator protocol.
    pub primary_model_calls: usize,
    /// Calls requested by task environments and canonical verifiers.
    pub auxiliary_model_calls: usize,
    /// Auxiliary calls requested by the task environment.
    pub environment_model_calls: usize,
    /// Auxiliary calls requested by the canonical verifier.
    pub verifier_model_calls: usize,
    /// Aggregate prompt tokens reported by Rust.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_tokens: Option<u64>,
    /// Aggregate completion tokens reported by Rust.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completion_tokens: Option<u64>,
    /// Aggregate cached prompt tokens reported by Rust.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cached_tokens: Option<u64>,
    /// Prompt tokens from canonical agent calls only.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub primary_prompt_tokens: Option<u64>,
    /// Completion tokens from canonical agent calls only.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub primary_completion_tokens: Option<u64>,
    /// Cached prompt tokens from canonical agent calls only.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub primary_cached_tokens: Option<u64>,
    /// Prompt tokens from environment and verifier calls only.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub auxiliary_prompt_tokens: Option<u64>,
    /// Completion tokens from environment and verifier calls only.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub auxiliary_completion_tokens: Option<u64>,
    /// Cached prompt tokens from environment and verifier calls only.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub auxiliary_cached_tokens: Option<u64>,
    /// Infrastructure or cancellation category.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error_kind: Option<String>,
    /// Infrastructure or cancellation detail.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error_message: Option<String>,
    /// Canonical harness artifact path.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub artifact_path: Option<String>,
}

/// Typed native-v2 agentic evaluation block.
#[derive(Debug, Clone, PartialEq, DeriveSerialize)]
pub struct AgenticEvaluationReport {
    /// Exact harness, agent, environment, and verifier identity.
    pub evaluator: AgenticEvaluatorReportInfo,
    /// Reproducibility-relevant authored configuration.
    pub config: AgenticRunConfigReport,
    /// Generic aggregates over canonical verifier outputs.
    pub summary: AgenticEvaluationSummary,
    /// Complete results in frozen evaluator order.
    pub records: Vec<AgenticEpisodeReport>,
}

/// Terminal semantic class assigned by the selected evaluator provider.
///
/// This is deliberately independent from the Rust transport terminal stored in
/// request metrics. A completed score of zero is therefore never confused with
/// infrastructure failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq, DeriveSerialize)]
#[serde(rename_all = "snake_case")]
pub enum EvaluationCaseOutcomeKind {
    /// Provider scoring completed and public score projections may be present.
    Completed,
    /// A provider, host, or evaluator infrastructure stage failed.
    InfrastructureError,
    /// Rust or the provider explicitly cancelled the case.
    Cancelled,
}

/// Report-safe evaluator error metadata.
#[derive(Debug, Clone, PartialEq, Eq, DeriveSerialize)]
pub struct EvaluationCaseErrorReport {
    /// Provider-defined lifecycle stage, validated as a bounded identifier.
    pub stage: String,
    /// Stable error category with secret-bearing detail removed.
    pub kind: String,
    /// Whether provider semantic policy permits a new semantic attempt.
    pub retryable: bool,
    /// Redacted diagnostic suitable for the public report.
    pub message: String,
}

/// One factory-schema-validated public score projection.
///
/// The provider's complete native score tree is never stored here. It remains
/// in the restricted sealed bundle with no public content digest.
#[derive(Debug, Clone, PartialEq, DeriveSerialize)]
pub struct EvaluationPublicScoreReport {
    /// Canonical public value after Rust validation and reserialization.
    pub value: Value,
    /// Versioned factory-owned projection schema.
    pub projection_schema: String,
}

/// Report-safe terminal record for one opaque evaluator case occurrence.
#[derive(Debug, Clone, PartialEq, DeriveSerialize)]
pub struct EvaluationCaseReport {
    /// Opaque occurrence identifier in frozen canonical order.
    pub case_id: String,
    /// Opaque provider template identifier.
    pub template_id: String,
    /// Safe provider-owned task label.
    pub task: String,
    /// Safe immutable source label.
    pub source: String,
    /// Explicit semantic terminal class.
    pub outcome: EvaluationCaseOutcomeKind,
    /// Reviewed public score projections keyed by factory-owned public label.
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub scores: BTreeMap<String, EvaluationPublicScoreReport>,
    /// Finite scalar projections eligible for score/performance joins.
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub numeric_metrics: BTreeMap<String, f64>,
    /// Factory-owned primary public score label, when reviewed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub primary_score: Option<String>,
    /// Redacted infrastructure or cancellation metadata.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<EvaluationCaseErrorReport>,
    /// Host-assigned opaque artifact references; restricted paths stay private.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub artifact_refs: Vec<String>,
}

/// One factory-validated aggregate projected into the public report.
#[derive(Debug, Clone, PartialEq, DeriveSerialize)]
pub struct EvaluationAggregateMetricReport {
    /// Factory-owned stable public scorer label.
    pub scorer: String,
    /// Factory-owned stable public reducer label.
    pub reducer: String,
    /// Factory-owned stable public metric label.
    pub metric: String,
    /// Finite provider-computed aggregate value.
    pub value: f64,
    /// Number of factory-validated aggregation units in the denominator.
    pub scored_count: usize,
    /// Number of factory-validated units excluded from the denominator.
    pub unscored_count: usize,
    /// Exact executable factory aggregate-rule fingerprint.
    pub projection_schema: String,
}

/// Rust-authoritative traffic totals for one logical evaluator route.
#[derive(Debug, Clone, Default, PartialEq, Eq, DeriveSerialize)]
pub struct EvaluationRouteSummaryReport {
    /// Logical host operations accepted by Rust.
    pub logical_operations: usize,
    /// Concrete upstream transport attempts made by Rust.
    pub transport_attempts: usize,
    /// Attempts after the first attempt of a logical operation.
    pub retries: usize,
    /// Rust-owned cache or explicitly identity-bound replay hits.
    pub cache_hits: usize,
    /// Operations that completed normally at the transport layer.
    pub completed: usize,
    /// Operations that failed or were rejected at the transport layer.
    pub failed: usize,
    /// Operations cancelled before or during transport.
    pub cancelled: usize,
    /// Prompt tokens when every contributing operation reported usage.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_tokens: Option<u64>,
    /// Completion tokens when every contributing operation reported usage.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completion_tokens: Option<u64>,
    /// Reasoning tokens when every contributing operation reported usage.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_tokens: Option<u64>,
    /// Cached prompt tokens when every contributing operation reported usage.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cached_tokens: Option<u64>,
}

/// Rust-verified artifact manifest entry.
#[derive(Debug, Clone, PartialEq, Eq, DeriveSerialize)]
pub struct EvaluationArtifactReport {
    /// Host-assigned opaque reference used by case reports.
    pub artifact_ref: String,
    /// Factory-reviewed public path; absent for restricted artifacts.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,
    /// Factory-reviewed public media type; absent for restricted artifacts.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub media_type: Option<String>,
    /// Factory-authorized visibility (`public` or `restricted`).
    pub visibility: String,
    /// Factory-reviewed public byte length; absent for restricted artifacts.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub size_bytes: Option<u64>,
    /// Factory-reviewed public SHA-256; absent for restricted artifacts.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub artifact_content_sha256: Option<String>,
    /// Factory-owned public artifact projection schema; absent when restricted.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub projection_schema: Option<String>,
}

/// Safe identity graph for a replaceable evaluator-provider run.
#[derive(Debug, Clone, PartialEq, Eq, DeriveSerialize)]
pub struct EvaluationIdentityReport {
    /// Evaluator-worker protocol version.
    pub evaluator_protocol: u32,
    /// Open provider registry ID.
    pub provider: String,
    /// Factory-owned immutable distribution ID.
    pub distribution: String,
    /// Provider package/source identity digest.
    pub provider_source_sha256: String,
    /// Factory-attested worker source digest.
    pub worker_source_sha256: String,
    /// Factory-attested dependency lock digest.
    pub dependency_lock_sha256: String,
    /// Fingerprint of the pure authored-configuration schema.
    pub authored_schema_fingerprint: String,
    /// Canonical secret-redacted resolved evaluator configuration digest.
    pub resolved_config_sha256: String,
    /// Ordered case/unit manifest digest.
    pub ordered_manifest_sha256: String,
    /// Rust host implementation and capability inventory digest.
    pub host_identity_sha256: String,
    /// Enforced evaluator isolation proof digest.
    pub isolation_proof_sha256: String,
    /// Optional immutable OCI identity.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub container_digest: Option<String>,
    /// Provider-specific, factory-approved identity fields.
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub components: BTreeMap<String, String>,
}

/// Secret-free prepared logical route inventory.
#[derive(Debug, Clone, PartialEq, Eq, DeriveSerialize)]
pub struct EvaluationRouteReport {
    /// Logical service identifier requested by the evaluator.
    pub service_id: String,
    /// Safe purpose label.
    pub purpose: String,
    /// Authored model alias, not an endpoint locator.
    pub model: String,
    /// Prepared endpoint profile ID.
    pub endpoint_profile: String,
    /// Digest of the credential-free prepared binding identity.
    pub prepared_identity_sha256: String,
}

/// Generic native-v2 evaluation result shared by static and stateful providers.
#[derive(Debug, Clone, PartialEq, DeriveSerialize)]
pub struct EvaluationReport {
    /// Factory-attested evaluator/provider/host identity.
    pub identity: EvaluationIdentityReport,
    /// Factory-schema-projected safe resolved configuration.
    pub config: Value,
    /// Secret-free logical route inventory in deterministic service order.
    pub routes: Vec<EvaluationRouteReport>,
    /// Number of frozen case occurrences.
    pub case_count: usize,
    /// Cases with a provider semantic result, including valid zero scores.
    pub completed_count: usize,
    /// Cases excluded because infrastructure failed.
    pub infrastructure_error_count: usize,
    /// Cases excluded because they were cancelled.
    pub cancelled_count: usize,
    /// Ordered report-safe case outcomes.
    pub cases: Vec<EvaluationCaseReport>,
    /// Factory-validated public aggregate projections.
    pub aggregates: Vec<EvaluationAggregateMetricReport>,
    /// Rust traffic summaries keyed by logical service ID.
    pub route_summaries: BTreeMap<String, EvaluationRouteSummaryReport>,
    /// Rust-verified report projection of the sealed artifact manifest.
    pub artifacts: Vec<EvaluationArtifactReport>,
}

/// Terminal lifecycle state reported by a telemetry archive execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, DeriveSerialize, DeriveDeserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReportTelemetryArchiveState {
    /// Runtime inputs were validated but no durable archive exists yet.
    Prepared,
    /// The immutable genesis generation is durable locally.
    GenesisDurable,
    /// Sources and the archive writer are active.
    Running,
    /// Shutdown was requested and no new source work should begin.
    StopRequested,
    /// Accepted work is draining toward the final record fence.
    Draining,
    /// A final immutable local generation and head are durable.
    LocallyFinalized,
    /// The final generation and head are durable remotely.
    RemotelyFinalized,
    /// The archive operation failed.
    Failed,
}

/// One credential-free immutable head and generation reference.
#[derive(Debug, Clone, PartialEq, Eq, DeriveSerialize, DeriveDeserialize)]
pub struct ReportTelemetryArchiveHead {
    /// Stable local or remote head URI; signed URLs and credentials are excluded.
    pub head_uri: String,
    /// Immutable generation-manifest URI.
    pub generation_uri: String,
    /// BLAKE3 digest of the exact generation manifest.
    pub generation_hash: String,
    /// BLAKE3 digest of the generation's root index page.
    pub index_root_hash: String,
}

/// Phase-boundary role retained by an exact telemetry loss range.
#[derive(Debug, Clone, Copy, PartialEq, Eq, DeriveSerialize, DeriveDeserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReportTelemetryBoundaryRole {
    /// Boundary starts phase membership for the source.
    PhaseStart,
    /// Boundary ends phase membership for the source.
    PhaseEnd,
}

/// One structured phase-boundary reference associated with telemetry loss.
#[derive(Debug, Clone, PartialEq, Eq, DeriveSerialize, DeriveDeserialize)]
pub struct ReportTelemetryBoundaryReference {
    /// Stable transition identity.
    pub transition_id: String,
    /// Stable boundary identity within the transition.
    pub boundary_id: String,
    /// Authored phase identity.
    pub phase_id: String,
    /// Source expected to observe the boundary.
    pub source_id: String,
    /// Whether the boundary starts or ends phase membership.
    pub role: ReportTelemetryBoundaryRole,
    /// Optional group whose members share one atomically sealed transition.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub coalescing_group_id: Option<String>,
}

/// Frozen telemetry-loss class from archive schema v1.
#[derive(Debug, Clone, Copy, PartialEq, Eq, DeriveSerialize, DeriveDeserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReportTelemetryLossKind {
    /// A cadence tick was skipped before issuing source work.
    MissedCadence,
    /// Native work completed but archive admission rejected its projection.
    ArchiveRejected,
    /// Accepted archive projection failed.
    ProjectionFailed,
    /// The archive writer failed.
    WriterFailed,
    /// Shutdown abandoned accepted work at its deadline.
    ShutdownAbandoned,
}

/// Frozen telemetry-loss reason from archive schema v1.
#[derive(Debug, Clone, Copy, PartialEq, Eq, DeriveSerialize, DeriveDeserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReportTelemetryLossReason {
    /// The previous scrape overran a fixed cadence deadline.
    CadenceOverrun,
    /// Bounded archive admission was unavailable.
    ArchiveAdmissionRejected,
    /// Projection of an accepted attempt failed.
    ProjectionError,
    /// The archive writer stopped making valid progress.
    WriterError,
    /// The shutdown deadline expired before accepted work completed.
    ShutdownDeadline,
}

/// One exact, coalesced telemetry-loss range retained for report consumers.
///
/// Nullable range pairs preserve the distinction between missed cadence and
/// already-issued work. Boundary overflow is explicit and content-addressed;
/// the report never silently truncates boundary evidence.
#[derive(Debug, Clone, PartialEq, Eq, DeriveSerialize, DeriveDeserialize)]
pub struct ReportTelemetryLossRange {
    /// Source identity, or null for a global archive loss.
    pub source_id: Option<String>,
    /// Semantic loss class.
    pub loss_kind: ReportTelemetryLossKind,
    /// Frozen reason within the loss class.
    pub reason: ReportTelemetryLossReason,
    /// Number of omitted entries represented by the inclusive ranges.
    pub count: u64,
    /// First omitted source-local record sequence.
    pub first_source_record_seq: Option<u64>,
    /// Last omitted source-local record sequence.
    pub last_source_record_seq: Option<u64>,
    /// First omitted physical request-attempt sequence.
    pub first_request_attempt_seq: Option<u64>,
    /// Last omitted physical request-attempt sequence.
    pub last_request_attempt_seq: Option<u64>,
    /// First missed cadence tick.
    pub first_tick: Option<u64>,
    /// Last missed cadence tick.
    pub last_tick: Option<u64>,
    /// First missed absolute Clock deadline.
    pub first_deadline_ns: Option<i64>,
    /// Last missed absolute Clock deadline.
    pub last_deadline_ns: Option<i64>,
    /// Receipt Clock value at which the range was sealed.
    pub loss_observed_ns: i64,
    /// Exact retained phase-boundary references.
    pub boundary_refs: Vec<ReportTelemetryBoundaryReference>,
    /// Boundary references folded into the overflow digest.
    pub boundary_overflow_count: u64,
    /// Digest of overflowed canonical boundary-reference bytes.
    pub boundary_overflow_digest: Option<String>,
}

/// Latest cumulative summary for one fixed-memory loss-saturation slot.
///
/// Consumers select the greatest `saturation_snapshot_seq` per stable slot and
/// must not sum cumulative snapshots. Only that latest snapshot enters a
/// native report.
#[derive(Debug, Clone, PartialEq, Eq, DeriveSerialize, DeriveDeserialize)]
pub struct ReportTelemetryLossSaturationSummary {
    /// Source identity, or null for a global archive loss.
    pub source_id: Option<String>,
    /// Semantic class shared by every omitted entry in this slot.
    pub loss_kind: ReportTelemetryLossKind,
    /// Frozen reason shared by every omitted entry in this slot.
    pub reason: ReportTelemetryLossReason,
    /// Stable BLAKE3 identity of the preallocated saturation slot.
    pub saturation_slot_id: String,
    /// Monotonic slot-local sequence of this latest cumulative snapshot.
    pub saturation_snapshot_seq: u64,
    /// Exact number of non-coalescible ranges omitted from enumeration.
    pub cumulative_omitted_range_count: u64,
    /// Exact number of omitted entries represented by those ranges.
    pub cumulative_omitted_entry_count: u64,
    /// Order-sensitive digest over every canonical omitted entry.
    pub omitted_rolling_digest: String,
    /// First omitted source-local record sequence, when applicable.
    pub first_source_record_seq: Option<u64>,
    /// Last omitted source-local record sequence, when applicable.
    pub last_source_record_seq: Option<u64>,
    /// First omitted physical request-attempt sequence, when applicable.
    pub first_request_attempt_seq: Option<u64>,
    /// Last omitted physical request-attempt sequence, when applicable.
    pub last_request_attempt_seq: Option<u64>,
    /// First missed cadence tick, when applicable.
    pub first_tick: Option<u64>,
    /// Last missed cadence tick, when applicable.
    pub last_tick: Option<u64>,
    /// First omitted absolute Clock deadline, when applicable.
    pub first_deadline_ns: Option<i64>,
    /// Last omitted absolute Clock deadline, when applicable.
    pub last_deadline_ns: Option<i64>,
    /// Receipt Clock value at which the latest snapshot was sealed.
    pub loss_observed_ns: i64,
}

/// Bounded archive-writer and loss-ledger health reported at finalization.
#[derive(Debug, Clone, PartialEq, Eq, DeriveSerialize, DeriveDeserialize)]
pub struct ReportTelemetryArchiveSpoolBudget {
    /// Whether ordinary/control admission is permanently closed.
    pub closed: bool,
    /// Whether the protected finalization transaction began.
    pub finalizing: bool,
    /// Reconciled logical bytes plus conservative committed growth.
    pub accounted_bytes: u64,
    /// Reconciled logical files plus conservative committed growth.
    pub accounted_files: u64,
    /// Current ordinary-lane committed byte growth.
    pub ordinary_growth_bytes: u64,
    /// Current ordinary-lane committed file growth.
    pub ordinary_growth_files: u64,
    /// Current control-lane committed byte growth.
    pub control_growth_bytes: u64,
    /// Current control-lane committed file growth.
    pub control_growth_files: u64,
    /// Outstanding ordinary frame reservations.
    pub ordinary_frames: u64,
    /// Outstanding control frame reservations.
    pub control_frames: u64,
    /// Reservations not yet committed or released.
    pub outstanding_leases: u64,
    /// Bytes unavailable to ordinary admission.
    pub protected_reserve_bytes: u64,
    /// Files unavailable to ordinary admission.
    pub protected_reserve_files: u64,
    /// Bytes unavailable even to the control lane.
    pub finalization_reserve_bytes: u64,
    /// Files unavailable even to the control lane.
    pub finalization_reserve_files: u64,
    /// Highest accounted/reserved byte usage observed.
    pub high_water_bytes: u64,
    /// Highest accounted/reserved file usage observed.
    pub high_water_files: u64,
}

/// Bounded archive-writer and loss-ledger health reported at finalization.
#[derive(Debug, Clone, PartialEq, Eq, DeriveSerialize, DeriveDeserialize)]
pub struct ReportTelemetryArchiveHealth {
    /// Exact ranges that remain individually enumerable.
    pub loss_ranges: Vec<ReportTelemetryLossRange>,
    /// Latest cumulative snapshots for saturated fixed-memory slots.
    pub loss_saturation_summaries: Vec<ReportTelemetryLossSaturationSummary>,
    /// Whether every loss remains represented by an exact range.
    pub complete_ranges: bool,
    /// Whether the archive writer remained alive through the reported snapshot.
    pub writer_alive: bool,
    /// Final transaction-reserve accounting when an archive owner was active.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub spool_budget: Option<ReportTelemetryArchiveSpoolBudget>,
}

/// Additive telemetry-archive outcome embedded in a native-v2 report.
///
/// The fixed shape deliberately excludes credentials, raw labels, signed
/// URLs, and arbitrary diagnostics. Local and remote heads are independently
/// optional because a best-effort attachment can succeed as a benchmark even
/// when the archive writer or remote publication fails.
#[derive(Debug, Clone, PartialEq, Eq, DeriveSerialize, DeriveDeserialize)]
pub struct ReportTelemetryArchive {
    /// Telemetry archive report-block schema version.
    pub schema_version: String,
    /// Stable archive UUID.
    pub archive_id: String,
    /// UUID of this runner execution.
    pub execution_id: String,
    /// BLAKE3 identity of the receipt observer Clock epoch.
    pub receipt_observer_epoch_id: String,
    /// Session created by this execution; null for source-free remote finalization.
    pub collection_session_id: Option<String>,
    /// Greatest collection session reachable from the reported generation.
    pub latest_collection_session_id: Option<String>,
    /// Terminal archive lifecycle state.
    pub state: ReportTelemetryArchiveState,
    /// Credential-free local publication-receipt head URI.
    pub publication_receipts_uri: String,
    /// Final local head, when one became durable.
    pub local_head: Option<ReportTelemetryArchiveHead>,
    /// Final remote head, when one became durable.
    pub remote_head: Option<ReportTelemetryArchiveHead>,
    /// Whether a final immutable local generation is authoritative.
    pub finalized_local: bool,
    /// Whether a final immutable remote generation is authoritative.
    pub finalized_remote: bool,
    /// Whether any archive evidence was lost or could not be enumerated.
    pub lossy: bool,
    /// Bounded writer and loss-ledger health.
    pub health: ReportTelemetryArchiveHealth,
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
    /// Optional accuracy/analyzer output.
    pub accuracy: Option<AccuracyAnalysis>,
    /// Full per-request grading records in deterministic workload order.
    pub accuracy_records: Vec<AccuracyRecord>,
    /// Exact external evaluator identity for accuracy runs.
    pub evaluator: Option<EvaluatorReportInfo>,
    /// Optional stateful agentic evaluator result block.
    pub agentic: Option<AgenticEvaluationReport>,
    /// Generic provider-neutral evaluation result block.
    pub evaluation: Option<EvaluationReport>,
    /// Optional typed telemetry-archive outcome.
    pub telemetry_archive: Option<ReportTelemetryArchive>,
    /// Grouped run errors.
    pub errors: Vec<ReportError>,
}

/// Borrowed inputs for one IO-free native-v2 report build.
///
/// Standalone telemetry watch passes `metrics: None`; ordinary benchmark runs
/// pass their real accumulator summary through [`NativeReport::from_outcome`].
/// This distinction prevents a telemetry-only execution from fabricating a
/// request distribution or benchmark duration.
#[derive(Debug, Clone, Copy)]
pub struct NativeReportInput<'a> {
    /// Profiling accumulator summary, absent for telemetry-only execution.
    pub metrics: Option<&'a AccumulatorSummary>,
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
        if let Some(metrics) = metrics {
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
        }
        NativeReport {
            schema_version: NATIVE_REPORT_SCHEMA_VERSION,
            aiperf_version: env!("CARGO_PKG_VERSION").to_string(),
            run: ReportRun::unfinalized(outcome.run.clone()),
            summary: run_summary,
            metrics: metrics.map_or_else(BTreeMap::new, build_metric_map),
            warmup_metrics: outcome.warmup.as_ref().map(build_metric_map),
            server_metrics: build_sidecar_map(&outcome.server_metrics),
            warmup_server_metrics: build_sidecar_map(&outcome.warmup_server_metrics),
            accuracy: outcome.accuracy.clone(),
            accuracy_records: outcome.accuracy_records.clone(),
            evaluator: outcome.evaluator.clone(),
            agentic: outcome.agentic.clone(),
            evaluation: outcome.evaluation.clone(),
            telemetry_archive: outcome.telemetry_archive.clone(),
            errors: outcome.errors.clone(),
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
    /// Warmup metrics using the same representation.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub warmup_metrics: Option<BTreeMap<String, MetricEntry>>,
    /// Profiling server telemetry keyed by original Prometheus family name.
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub server_metrics: BTreeMap<String, MetricEntry>,
    /// Warmup server telemetry keyed by original Prometheus family name.
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub warmup_server_metrics: BTreeMap<String, MetricEntry>,
    /// Optional accuracy analysis.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub accuracy: Option<AccuracyAnalysis>,
    /// Full per-request grading records. Empty outside accuracy mode.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub accuracy_records: Vec<AccuracyRecord>,
    /// Exact canonical evaluator identity. Absent outside accuracy mode.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub evaluator: Option<EvaluatorReportInfo>,
    /// Stateful harness identity, configuration, summary, and episode records.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub agentic: Option<AgenticEvaluationReport>,
    /// Provider-neutral evaluator identity, results, traffic, and artifact digests.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub evaluation: Option<EvaluationReport>,
    /// Typed telemetry archive outcome for standalone watch or attached collection.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub telemetry_archive: Option<ReportTelemetryArchive>,
    /// Grouped run errors.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub errors: Vec<ReportError>,
}

impl NativeReport {
    /// Builds a native report from metrics and optional accuracy analysis.
    pub fn new(metrics: &AccumulatorSummary, accuracy: Option<AccuracyAnalysis>) -> Self {
        NativeReporter.report(NativeReportInput {
            metrics: Some(metrics),
            outcome: &RunOutcome {
                accuracy,
                ..RunOutcome::default()
            },
        })
    }

    /// Builds a native report with explicit run metadata.
    pub fn from_outcome(metrics: &AccumulatorSummary, outcome: &RunOutcome) -> Self {
        Self::from_input(NativeReportInput {
            metrics: Some(metrics),
            outcome,
        })
    }

    /// Builds a native report from optional profiling metrics and runtime facts.
    pub fn from_input(input: NativeReportInput<'_>) -> Self {
        NativeReporter.report(input)
    }

    /// Stamp coordinator-owned provenance and pair-owned typed facts exactly
    /// once before the sole native-v2 serialization.
    ///
    /// This operates on the report model, never on serialized JSON. A second
    /// call is rejected so no downstream exporter can replace the executable
    /// identity or reinterpret pair facts after coordinator finalization.
    pub fn finalize_run(
        mut self,
        provenance: ReportRunProvenance,
        facts: ReportPairRunFacts,
    ) -> Result<Self, ReportProvenanceError> {
        self.run.finalize(provenance, facts)?;
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
    timeslices: &[crate::MetricTimeslice],
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
    stats: &crate::DistributionStats,
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
        MetricConsoleGroup::Effective => "effective",
        MetricConsoleGroup::Active => "active",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        InferenceDimensions, MetricResult, MetricResultData, MetricsAccumulator, MetricsConfig,
        Phase, RecordIngest, SidecarMetric, SidecarSeries, SidecarStats, Unit,
    };

    fn archive_head(prefix: &str) -> ReportTelemetryArchiveHead {
        ReportTelemetryArchiveHead {
            head_uri: format!("{prefix}/LATEST"),
            generation_uri: format!(
                "{prefix}/manifests/generation-7-blake3-{}.json",
                "b".repeat(64)
            ),
            generation_hash: format!("blake3:{}", "b".repeat(64)),
            index_root_hash: format!("blake3:{}", "c".repeat(64)),
        }
    }

    fn healthy_archive_report() -> ReportTelemetryArchive {
        ReportTelemetryArchive {
            schema_version: TELEMETRY_ARCHIVE_REPORT_SCHEMA_VERSION.to_string(),
            archive_id: "11111111-1111-4111-8111-111111111111".to_string(),
            execution_id: "22222222-2222-4222-8222-222222222222".to_string(),
            receipt_observer_epoch_id: format!("blake3:{}", "a".repeat(64)),
            collection_session_id: Some("33333333-3333-4333-8333-333333333333".to_string()),
            latest_collection_session_id: Some("33333333-3333-4333-8333-333333333333".to_string()),
            state: ReportTelemetryArchiveState::RemotelyFinalized,
            publication_receipts_uri: "file:///var/lib/aiperf/archive/LOCAL-RECEIPTS".to_string(),
            local_head: Some(archive_head("file:///var/lib/aiperf/archive")),
            remote_head: Some(archive_head("s3://aiperf-telemetry/archive")),
            finalized_local: true,
            finalized_remote: true,
            lossy: false,
            health: ReportTelemetryArchiveHealth {
                loss_ranges: Vec::new(),
                loss_saturation_summaries: Vec::new(),
                complete_ranges: true,
                writer_alive: true,
                spool_budget: None,
            },
        }
    }

    fn lossy_attached_archive_report() -> ReportTelemetryArchive {
        ReportTelemetryArchive {
            schema_version: TELEMETRY_ARCHIVE_REPORT_SCHEMA_VERSION.to_string(),
            archive_id: "44444444-4444-4444-8444-444444444444".to_string(),
            execution_id: "55555555-5555-4555-8555-555555555555".to_string(),
            receipt_observer_epoch_id: format!("blake3:{}", "d".repeat(64)),
            collection_session_id: Some("66666666-6666-4666-8666-666666666666".to_string()),
            latest_collection_session_id: Some("66666666-6666-4666-8666-666666666666".to_string()),
            state: ReportTelemetryArchiveState::LocallyFinalized,
            publication_receipts_uri: "file:///var/lib/aiperf/attached/LOCAL-RECEIPTS".to_string(),
            local_head: Some(archive_head("file:///var/lib/aiperf/attached")),
            remote_head: None,
            finalized_local: true,
            finalized_remote: false,
            lossy: true,
            health: ReportTelemetryArchiveHealth {
                loss_ranges: vec![ReportTelemetryLossRange {
                    source_id: Some("server_metrics_primary".to_string()),
                    loss_kind: ReportTelemetryLossKind::ArchiveRejected,
                    reason: ReportTelemetryLossReason::ArchiveAdmissionRejected,
                    count: 2,
                    first_source_record_seq: Some(7),
                    last_source_record_seq: Some(8),
                    first_request_attempt_seq: Some(12),
                    last_request_attempt_seq: Some(13),
                    first_tick: None,
                    last_tick: None,
                    first_deadline_ns: None,
                    last_deadline_ns: None,
                    loss_observed_ns: 6_000,
                    boundary_refs: vec![ReportTelemetryBoundaryReference {
                        transition_id: "profiling-finish".to_string(),
                        boundary_id: "profiling-end-server".to_string(),
                        phase_id: "profiling".to_string(),
                        source_id: "server_metrics_primary".to_string(),
                        role: ReportTelemetryBoundaryRole::PhaseEnd,
                        coalescing_group_id: None,
                    }],
                    boundary_overflow_count: 0,
                    boundary_overflow_digest: None,
                }],
                loss_saturation_summaries: vec![ReportTelemetryLossSaturationSummary {
                    source_id: Some("server_metrics_primary".to_string()),
                    loss_kind: ReportTelemetryLossKind::WriterFailed,
                    reason: ReportTelemetryLossReason::WriterError,
                    saturation_slot_id: format!("blake3:{}", "e".repeat(64)),
                    saturation_snapshot_seq: 3,
                    cumulative_omitted_range_count: 4,
                    cumulative_omitted_entry_count: 5,
                    omitted_rolling_digest: format!("blake3:{}", "f".repeat(64)),
                    first_source_record_seq: Some(20),
                    last_source_record_seq: Some(24),
                    first_request_attempt_seq: Some(30),
                    last_request_attempt_seq: Some(34),
                    first_tick: None,
                    last_tick: None,
                    first_deadline_ns: None,
                    last_deadline_ns: None,
                    loss_observed_ns: 8_000,
                }],
                complete_ranges: false,
                writer_alive: false,
                spool_budget: None,
            },
        }
    }

    fn finalized_report(
        report: NativeReport,
        distribution_digit: char,
        backend: &str,
        workload: &str,
        endpoint_profiles: Vec<ReportEndpointProfileIdentity>,
    ) -> NativeReport {
        report
            .finalize_run(
                ReportRunProvenance::new(
                    format!("blake3:{}", distribution_digit.to_string().repeat(64)),
                    backend,
                    workload,
                    Vec::new(),
                    endpoint_profiles,
                )
                .unwrap(),
                ReportPairRunFacts::new(),
            )
            .unwrap()
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
            data: MetricResultData::Distribution(crate::DistributionStats {
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
    fn standalone_watch_report_has_archive_provenance_without_fake_metrics() {
        let outcome = RunOutcome {
            run: ReportRunInfo {
                mode: Some("telemetry_watch".to_string()),
                model: None,
            },
            telemetry_archive: Some(healthy_archive_report()),
            ..RunOutcome::default()
        };
        let report = finalized_report(
            NativeReport::from_input(NativeReportInput {
                metrics: None,
                outcome: &outcome,
            }),
            '1',
            "telemetry_archive",
            "watch",
            Vec::new(),
        );

        assert!(report.metrics.is_empty());
        assert_eq!(report.summary.start_time, None);
        assert_eq!(report.summary.end_time, None);
        assert_eq!(report.summary.duration_s, None);
        let serialized = serde_json::to_string_pretty(&report).unwrap();
        assert_eq!(
            serialized,
            include_str!("../tests/golden/native_v2_telemetry_standalone.json").trim_end()
        );
    }

    #[test]
    fn attached_report_keeps_real_metrics_and_structured_degradation() {
        let mut metrics = AccumulatorSummary::new();
        metrics.insert_finite(MetricTag::RequestCount, 3.0);
        let outcome = RunOutcome {
            run: ReportRunInfo {
                mode: Some("online".to_string()),
                model: Some("candidate-model".to_string()),
            },
            telemetry_archive: Some(lossy_attached_archive_report()),
            ..RunOutcome::default()
        };
        let report = finalized_report(
            NativeReport::from_input(NativeReportInput {
                metrics: Some(&metrics),
                outcome: &outcome,
            }),
            '2',
            "online_http",
            "scheduled",
            vec![ReportEndpointProfileIdentity::new("primary", "chat").unwrap()],
        );

        assert!(report.metrics.contains_key("request_count"));
        let archive = report.telemetry_archive.as_ref().unwrap();
        assert!(archive.lossy);
        assert!(!archive.health.writer_alive);
        assert!(!archive.health.complete_ranges);
        let serialized = serde_json::to_string_pretty(&report).unwrap();
        assert_eq!(
            serialized,
            include_str!("../tests/golden/native_v2_telemetry_attached.json").trim_end()
        );
    }

    #[test]
    fn telemetry_archive_extension_is_old_and_new_reader_compatible() {
        #[derive(DeriveDeserialize)]
        struct LegacyNativeReport {
            schema_version: String,
            metrics: BTreeMap<String, Value>,
        }

        #[derive(DeriveDeserialize)]
        struct ArchiveAwareNativeReport {
            schema_version: String,
            #[serde(default)]
            telemetry_archive: Option<ReportTelemetryArchive>,
        }

        let standalone = include_str!("../tests/golden/native_v2_telemetry_standalone.json");
        let attached = include_str!("../tests/golden/native_v2_telemetry_attached.json");
        let absent = include_str!("../tests/golden/native_v2.json");

        let legacy_standalone: LegacyNativeReport = serde_json::from_str(standalone).unwrap();
        assert_eq!(
            legacy_standalone.schema_version,
            NATIVE_REPORT_SCHEMA_VERSION
        );
        assert!(legacy_standalone.metrics.is_empty());
        let legacy_attached: LegacyNativeReport = serde_json::from_str(attached).unwrap();
        assert_eq!(legacy_attached.schema_version, NATIVE_REPORT_SCHEMA_VERSION);
        assert!(legacy_attached.metrics.contains_key("request_count"));

        let new_absent: ArchiveAwareNativeReport = serde_json::from_str(absent).unwrap();
        assert_eq!(new_absent.schema_version, NATIVE_REPORT_SCHEMA_VERSION);
        assert!(new_absent.telemetry_archive.is_none());
        let new_standalone: ArchiveAwareNativeReport = serde_json::from_str(standalone).unwrap();
        let archive = new_standalone.telemetry_archive.unwrap();
        assert_eq!(
            archive.schema_version,
            TELEMETRY_ARCHIVE_REPORT_SCHEMA_VERSION
        );
        assert_eq!(
            archive.state,
            ReportTelemetryArchiveState::RemotelyFinalized
        );
        let new_attached: ArchiveAwareNativeReport = serde_json::from_str(attached).unwrap();
        let health = new_attached.telemetry_archive.unwrap().health;
        assert_eq!(health.loss_ranges.len(), 1);
        assert_eq!(health.loss_saturation_summaries.len(), 1);
    }

    #[test]
    fn provider_neutral_evaluation_keeps_zero_score_distinct_from_failure() {
        let zero_case = EvaluationCaseReport {
            case_id: "case-0".into(),
            template_id: "template-0".into(),
            task: "fixture".into(),
            source: "sha256:dataset".into(),
            outcome: EvaluationCaseOutcomeKind::Completed,
            scores: BTreeMap::from([(
                "accuracy".into(),
                EvaluationPublicScoreReport {
                    value: serde_json::json!(0),
                    projection_schema: "fixture-score-v1".into(),
                },
            )]),
            numeric_metrics: BTreeMap::from([("accuracy".into(), 0.0)]),
            primary_score: Some("accuracy".into()),
            error: None,
            artifact_refs: Vec::new(),
        };
        let failed_case = EvaluationCaseReport {
            case_id: "case-1".into(),
            template_id: "template-1".into(),
            task: "fixture".into(),
            source: "sha256:dataset".into(),
            outcome: EvaluationCaseOutcomeKind::InfrastructureError,
            scores: BTreeMap::new(),
            numeric_metrics: BTreeMap::new(),
            primary_score: None,
            error: Some(EvaluationCaseErrorReport {
                stage: "inference".into(),
                kind: "transport_failure".into(),
                retryable: false,
                message: "upstream attempt failed".into(),
            }),
            artifact_refs: Vec::new(),
        };
        let evaluation = EvaluationReport {
            identity: EvaluationIdentityReport {
                evaluator_protocol: 2,
                provider: "openbench".into(),
                distribution: "openbench_fixture_locked".into(),
                provider_source_sha256: "a".repeat(64),
                worker_source_sha256: "b".repeat(64),
                dependency_lock_sha256: "c".repeat(64),
                authored_schema_fingerprint: "d".repeat(64),
                resolved_config_sha256: "e".repeat(64),
                ordered_manifest_sha256: "f".repeat(64),
                host_identity_sha256: "1".repeat(64),
                isolation_proof_sha256: "2".repeat(64),
                container_digest: None,
                components: BTreeMap::new(),
            },
            config: serde_json::json!({"benchmark": "fixture"}),
            routes: vec![EvaluationRouteReport {
                service_id: "primary".into(),
                purpose: "primary".into(),
                model: "candidate".into(),
                endpoint_profile: "candidate_openai".into(),
                prepared_identity_sha256: "3".repeat(64),
            }],
            case_count: 2,
            completed_count: 1,
            infrastructure_error_count: 1,
            cancelled_count: 0,
            cases: vec![zero_case, failed_case],
            aggregates: vec![EvaluationAggregateMetricReport {
                scorer: "fixture".into(),
                reducer: "mean".into(),
                metric: "accuracy".into(),
                value: 0.0,
                scored_count: 1,
                unscored_count: 1,
                projection_schema: "6".repeat(64),
            }],
            route_summaries: BTreeMap::from([(
                "primary".into(),
                EvaluationRouteSummaryReport {
                    logical_operations: 2,
                    transport_attempts: 2,
                    completed: 1,
                    failed: 1,
                    ..EvaluationRouteSummaryReport::default()
                },
            )]),
            artifacts: vec![EvaluationArtifactReport {
                artifact_ref: "artifact-00000000".into(),
                path: None,
                media_type: None,
                visibility: "restricted".into(),
                size_bytes: None,
                artifact_content_sha256: None,
                projection_schema: None,
            }],
        };

        let report = NativeReport::from_outcome(
            &AccumulatorSummary::new(),
            &RunOutcome {
                evaluation: Some(evaluation),
                ..RunOutcome::default()
            },
        );
        let value = serde_json::to_value(report).unwrap();
        assert_eq!(value["evaluation"]["cases"][0]["outcome"], "completed");
        assert_eq!(
            value["evaluation"]["cases"][0]["scores"]["accuracy"]["value"],
            0
        );
        assert!(value["evaluation"]["cases"][0].get("error").is_none());
        assert_eq!(
            value["evaluation"]["cases"][1]["outcome"],
            "infrastructure_error"
        );
        assert!(value["evaluation"]["cases"][1].get("scores").is_none());
        assert_eq!(
            value["evaluation"]["artifacts"][0]["visibility"],
            "restricted"
        );
        for field in [
            "path",
            "media_type",
            "size_bytes",
            "artifact_content_sha256",
            "projection_schema",
        ] {
            assert!(value["evaluation"]["artifacts"][0].get(field).is_none());
        }
        assert!(
            value["evaluation"]
                .get("canonical_bundle_artifact_content_sha256")
                .is_none()
        );
        assert!(
            value["evaluation"]
                .get("normalized_result_sha256")
                .is_none()
        );
        assert!(
            value["evaluation"]["aggregates"][0]
                .get("definition")
                .is_none()
        );
        assert_eq!(
            value["evaluation"]["aggregates"][0]["projection_schema"],
            "6".repeat(64)
        );
    }

    #[test]
    fn coordinator_finalization_flattens_common_and_pair_facts_into_run() {
        let provenance = ReportRunProvenance::new(
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
            .with_dynamo(dynamo)
            .with_evaluation_compatibility(
                ReportEvaluationCompatibilityInfo::new(
                    vec!["openai_chat_completions".into()],
                    "d".repeat(64),
                    ReportEvaluationCompatibilityGrantLimits {
                        max_operations: 1,
                        max_concurrent_operations: 1,
                        max_request_bytes: 1024,
                        max_response_bytes: 2048,
                        max_stream_events: 1,
                        expires_after_ms: 1000,
                    },
                )
                .unwrap(),
            );
        let report = NativeReport::new(&AccumulatorSummary::new(), None)
            .finalize_run(provenance, facts)
            .unwrap();

        let value = serde_json::to_value(&report).unwrap();
        let run = &value["run"];
        assert_eq!(run["distribution_id"], format!("blake3:{}", "a".repeat(64)));
        assert_eq!(run["backend"], "dynosim");
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
        assert_eq!(
            run["evaluation_compatibility"]["dialect_ids"],
            serde_json::json!(["openai_chat_completions"])
        );
        assert_eq!(
            run["evaluation_compatibility"]["descriptor_sha256"],
            "d".repeat(64)
        );
        assert_eq!(
            run["evaluation_compatibility"]["effective_grant"]["max_operations"],
            1
        );
        assert_eq!(
            report.run.provenance().unwrap().distribution_id,
            format!("blake3:{}", "a".repeat(64))
        );
        assert!(report.run.is_finalized());
    }

    #[test]
    fn coordinator_provenance_rejects_ambiguous_or_inexact_identity() {
        let endpoint = ReportEndpointProfileIdentity::new("default", "chat").unwrap();
        let invalid_digest = ReportRunProvenance::new(
            "blake3:abc",
            "online_http",
            "scheduled",
            Vec::new(),
            vec![endpoint.clone()],
        )
        .unwrap_err();
        assert!(invalid_digest.to_string().contains("64 lowercase"));

        let duplicate_profile = ReportRunProvenance::new(
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

        let duplicate_extension = ReportRunProvenance::new(
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
        let grant = || ReportEvaluationCompatibilityGrantLimits {
            max_operations: 1,
            max_concurrent_operations: 1,
            max_request_bytes: 1,
            max_response_bytes: 1,
            max_stream_events: 1,
            expires_after_ms: 1,
        };
        assert!(ReportDynamoCapacityInfo::new(f64::NAN, 1.0, 1, 1, 1.0).is_err());
        assert!(ReportDynamoCapacityInfo::new(1.0, 1.0, 1, 1, f64::INFINITY).is_err());
        assert!(ReportDynamoParityInfo::new(74, 68, 5, 100).is_err());
        assert!(
            ReportGraphRunInfo::new("dag_jsonl", 1, 1, 1, 1)
                .unwrap()
                .with_outcome(ReportGraphOutcomeInfo::new(1, 1, 1))
                .is_err()
        );
        assert!(
            ReportEvaluationCompatibilityInfo::new(Vec::new(), "a".repeat(64), grant()).is_err()
        );
        assert!(
            ReportEvaluationCompatibilityInfo::new(
                vec![
                    "openai_chat_completions".into(),
                    "openai_chat_completions".into()
                ],
                "a".repeat(64),
                grant(),
            )
            .is_err()
        );
        assert!(
            ReportEvaluationCompatibilityInfo::new(
                vec!["openai_chat_completions".into()],
                "not-a-digest",
                grant(),
            )
            .is_err()
        );
    }

    #[test]
    fn report_run_provenance_can_only_be_finalized_once() {
        let provenance = || {
            ReportRunProvenance::new(
                format!("blake3:{}", "c".repeat(64)),
                "online_http",
                "scheduled",
                Vec::new(),
                Vec::new(),
            )
            .unwrap()
        };
        let report = NativeReport::new(&AccumulatorSummary::new(), None)
            .finalize_run(provenance(), ReportPairRunFacts::new())
            .unwrap();
        let error = report
            .finalize_run(provenance(), ReportPairRunFacts::new())
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
