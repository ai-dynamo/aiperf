// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Frozen transport/workload composition for protocol-v2 runs.
//!
//! There is no central transport/workload compatibility matrix. Each workload
//! validates and resolves its transport-specific execution path during
//! preparation.

use std::any::Any;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Debug;
use std::sync::Arc;

use crate::endpoints::{
    EndpointId, EndpointRegistry, RawEndpointConfig, RequestContentType, ResetKvCacheConfig,
    ServerProfilerConfig,
};
use crate::extensions::{
    AIPerfExtension, AIPerfRegistry, DuplicateName, ExtensionError, RegistryId,
};
use crate::failure::OnFailure;
use crate::metrics_core::{
    NativeReport, ReportEndpointProfileIdentity, ReportExtensionIdentity, ReportPairRunFacts,
    ReportRunMetadata,
};
use crate::transport::core::ConnectionReuseStrategy;
use crate::transport::http::config::ClientConfig;
use crate::transport::http::models::HttpVersion;
use anyhow::{Result, anyhow, bail, ensure};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, value::RawValue};

use crate::engine::dataset_input::DatasetInputAdapterResolver;
use crate::engine::execution_factories::ExecutionFactories;
use crate::engine::graph_input::GraphInputAdapterResolver;
use crate::engine::protocol::PhaseSpec;
use crate::engine::protocol_v2::{
    AuthoredRunSpecV2, ComponentId, FailureStageV2, NamedRunnerComponentSpecV2,
    RunDiagnosticArtifactV2, RunResourceV2,
};
use crate::engine::sidecar_input::PreparedSidecarInputs;

/// Clock family supplied by one prepared transport.
#[derive(Clone, Copy, Debug, Serialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum ClockKind {
    /// Monotonic wall-clock execution.
    Real,
    /// Deterministic discrete-event virtual time.
    Sim,
}

/// Deterministic capability facts for one linked transport factory.
///
/// These feed the `--capabilities` `Catalog` only. Each workload validates and
/// resolves its transport-specific execution path during preparation.
#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct TransportDescriptor {
    /// Stable registry ID.
    pub id: &'static str,
    /// Human-readable implementation description.
    pub description: &'static str,
    /// Clock family exposed by this transport (catalog fact).
    pub clock: ClockKind,
    /// Statically compiled transport feature IDs (catalog fact).
    pub features: &'static [&'static str],
    /// URL schemes this transport accepts (catalog fact). Projected verbatim
    /// into the capabilities catalog, so a new transport declares its schemes
    /// here at its single registration site rather than in a separate match.
    pub url_schemes: &'static [&'static str],
}

/// Deterministic capability facts for one linked workload factory.
#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct WorkloadDescriptor {
    /// Stable registry ID.
    pub id: &'static str,
    /// Human-readable implementation description.
    pub description: &'static str,
}

/// Config-specific requirements computed by a workload factory.
///
/// This gates config-level transport policy (e.g. the isolated control-plane
/// HTTP client) and authored resource presence. It is **not** a compatibility
/// predicate: no field admits or rejects a transport×workload combination.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct WorkloadRequirements {
    /// Required transport feature IDs for config-level transport gating (e.g.
    /// `control_plane_http`). Empty for built-in workloads.
    pub transport_features: BTreeSet<String>,
    /// Required/optional/forbidden authored resource blocks.
    pub resources: ResourceRequirementsV2,
}

/// Presence rule for one workload-scoped resource block.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ResourceRequirementV2 {
    /// The selected workload cannot prepare without this resource.
    Required,
    /// The resource may be present or omitted.
    Optional,
    /// Presence would create inert or unsafe configuration and is rejected.
    Forbidden,
}

/// Closed presence matrix returned by every workload factory.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ResourceRequirementsV2 {
    /// Model-selection policy.
    pub models: ResourceRequirementV2,
    /// Endpoint profiles.
    pub endpoints: ResourceRequirementV2,
    /// Native metric policy.
    pub metrics: ResourceRequirementV2,
    /// Generic artifact policy.
    pub artifacts: ResourceRequirementV2,
    /// Sidecar resources.
    pub sidecars: ResourceRequirementV2,
}

impl ResourceRequirementsV2 {
    /// Resource matrix shared by inference workloads in protocol v2.
    #[must_use]
    pub const fn inference() -> Self {
        Self {
            models: ResourceRequirementV2::Required,
            endpoints: ResourceRequirementV2::Required,
            metrics: ResourceRequirementV2::Optional,
            artifacts: ResourceRequirementV2::Optional,
            sidecars: ResourceRequirementV2::Optional,
        }
    }

    fn entries(self) -> [(RunResourceV2, ResourceRequirementV2); 5] {
        [
            (RunResourceV2::Models, self.models),
            (RunResourceV2::Endpoints, self.endpoints),
            (RunResourceV2::Metrics, self.metrics),
            (RunResourceV2::Artifacts, self.artifacts),
            (RunResourceV2::Sidecars, self.sidecars),
        ]
    }
}

impl Default for ResourceRequirementsV2 {
    fn default() -> Self {
        Self::inference()
    }
}

/// Type-erased, strictly validated transport configuration.
pub trait ValidatedTransportConfig: Debug + Send + Sync {
    /// Return the concrete startup-only value for a registered pair factory.
    fn as_any(&self) -> &dyn Any;

    /// Consume the erased startup value for a pair that owns its preparation.
    fn into_any(self: Box<Self>) -> Box<dyn Any + Send + Sync>;
}

impl<T> ValidatedTransportConfig for T
where
    T: Any + Debug + Send + Sync,
{
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn into_any(self: Box<Self>) -> Box<dyn Any + Send + Sync> {
        self
    }
}

/// Type-erased, strictly validated workload configuration.
pub trait ValidatedWorkloadConfig: Debug + Send + Sync {
    /// Return the concrete startup-only value for a registered pair factory.
    fn as_any(&self) -> &dyn Any;

    /// Consume the erased startup value for a pair that owns its preparation.
    fn into_any(self: Box<Self>) -> Box<dyn Any + Send + Sync>;
}

impl<T> ValidatedWorkloadConfig for T
where
    T: Any + Debug + Send + Sync,
{
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn into_any(self: Box<Self>) -> Box<dyn Any + Send + Sync> {
        self
    }
}

/// Startup-only transport validation and preparation half of the registry.
pub trait TransportFactory: Debug + Send + Sync {
    /// Describe the exact compiled implementation.
    fn descriptor(&self) -> &'static TransportDescriptor;

    /// Strictly decode and validate factory-owned authored config.
    fn validate(
        &self,
        authored: &RawValue,
        requirements: &WorkloadRequirements,
    ) -> Result<Box<dyn ValidatedTransportConfig>>;

    /// Resolve this transport's native execution binding from a validated
    /// config, or `None` when the transport uses a non-`RequestExecutor`
    /// execution mode (dynosim's virtual-clock co-simulation) prepared through
    /// its own module.
    ///
    /// This is the seam that makes a transport *swappable*: the workload asks the
    /// registered transport for its execution binding and drives the shared
    /// scheduled/graph runtime through it, never matching on a transport kind.
    /// The default returns `None`; every RequestExecutor transport
    /// (`http`/`grpc`/`dry_run`) overrides it.
    fn native_execution(
        &self,
        _config: &dyn ValidatedTransportConfig,
        _context: &RunContext,
    ) -> Result<Option<Arc<dyn NativeTransportExecution>>> {
        Ok(None)
    }
}

/// Execution binding produced by a validated *native* transport — one that
/// drives the `RequestExecutor` seam over a real clock (`http`, `grpc`,
/// `dry_run`).
///
/// The scheduled and graph workloads resolve one of these from the
/// transport-registry entry and drive the entire native runtime (pacing,
/// admission, phase orchestration, metrics, export) through it. There is no
/// `match` on a closed transport enum: adding a native transport means
/// registering a factory that returns its own binding, and nothing in the
/// workloads changes. A transport whose execution mode is not the
/// `RequestExecutor` seam (dynosim) returns `None` from
/// [`TransportFactory::native_execution`] instead.
pub trait NativeTransportExecution: Send + Sync {
    /// Turn-placement factory this transport drives below the shared dispatcher.
    fn executor_factory(&self) -> Arc<dyn crate::engine::turn_execution::RequestExecutorFactory>;

    /// Whether a model-readiness probe runs before profiling. Only the HTTP
    /// transport polls a live server; `grpc`/`dry_run` skip it.
    fn readiness_enabled(&self) -> bool;

    /// Whether this transport drives the run on a virtual `SimClock` (idle-pumped
    /// deterministic virtual time) rather than the real wall clock. The single
    /// native driver layer reads this to construct the clock — and thereby select
    /// the reactor discipline ([`crate::clock::Clock::drive`]),
    /// graph placement, and worker count. Default real; only the `dry_run`
    /// transport with `clock: sim` overrides it.
    fn uses_virtual_clock(&self) -> bool {
        false
    }

    /// Build one graph endpoint profile's dispatcher for this transport, or
    /// return an error when it does not (yet) drive the whole-trace graph
    /// workload. The graph runtime dispatches over the returned
    /// `Rc<dyn Dispatcher>` and never matches on a transport kind — adding a
    /// transport is implementing this method, nothing else.
    #[allow(clippy::too_many_arguments)]
    fn build_graph_dispatcher(
        &self,
        clock: std::rc::Rc<dyn crate::clock::Clock>,
        run_origin_ns: i64,
        urls: &[String],
        model: &str,
        transport_config: crate::transport::http::TransportSinkConfig,
        endpoints: std::rc::Rc<crate::endpoints::PreparedEndpointTable>,
        capture_raw: bool,
    ) -> Result<std::rc::Rc<dyn crate::transport::core::Dispatcher>>;

    /// Human-readable transport label used in graph tracing/diagnostics.
    fn graph_transport_label(&self) -> &'static str {
        "unknown"
    }

    /// Transport-specific run-level validation (endpoint URL schemes, server
    /// reachability policy, …), performed after component configs decode.
    fn validate_run(&self, run: &AuthoredRunSpecV2, context: &RunContext) -> Result<()>;

    /// Additive transport run metadata stamped onto the terminal response.
    fn run_metadata(&self) -> BTreeMap<String, String>;
}

/// Startup-only workload validation half of the registry.
pub trait WorkloadFactory: Debug + Send + Sync {
    /// Describe the exact compiled implementation.
    fn descriptor(&self) -> &'static WorkloadDescriptor;

    /// Strictly decode and validate factory-owned authored config.
    fn validate(&self, authored: &RawValue) -> Result<Box<dyn ValidatedWorkloadConfig>>;

    /// Derive transport requirements from the validated authored config.
    fn requirements(&self, config: &dyn ValidatedWorkloadConfig) -> Result<WorkloadRequirements>;

    /// Check run-level invariants for one authored run over the resolved
    /// transport, without external IO or artifact creation.
    ///
    /// The workload owns workload-level checks; any transport-specific endpoint
    /// check is resolved by `transport_id` (a map lookup, never a runtime string
    /// switch in the coordinator). The default performs no extra validation.
    fn validate_run(
        &self,
        _run: &AuthoredRunSpecV2,
        _context: &RunContext,
        _transport: &dyn ValidatedTransportConfig,
        _workload: &dyn ValidatedWorkloadConfig,
        _transport_id: &str,
    ) -> Result<()> {
        Ok(())
    }

    /// Complete workload preparation over the already-validated transport.
    ///
    /// The workload lowers its own authored config once and resolves the
    /// transport's dispatcher/placement factory from the coordinator-frozen
    /// [`ExecutionFactories`] by `transport_id`; it is otherwise
    /// transport-blind. The default rejects execution so a descriptor-only
    /// workload cannot silently run.
    fn prepare_with_context(
        &self,
        _run: &AuthoredRunSpecV2,
        _context: &RunContext,
        _transport: Box<dyn ValidatedTransportConfig>,
        _workload: Box<dyn ValidatedWorkloadConfig>,
        _transport_id: &str,
    ) -> Result<Box<dyn PreparedRunnerOperation>> {
        bail!(
            "workload {:?} is not executable in this runner distribution",
            self.descriptor().id
        )
    }
}

/// One-shot lifecycle acknowledgement after durable report persistence.
pub trait PreparedReportCommit: Debug {
    /// Acknowledge evaluator lifecycle completion after the authoritative
    /// native report has been atomically persisted.
    fn commit(self: Box<Self>) -> Result<()>;
}

/// Typed prepared-operation failure with a coordinator-visible lifecycle stage.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PreparedRunFailure {
    /// Stable failed stage.
    pub stage: FailureStageV2,
    /// Stable lowercase-snake-case diagnostic code.
    pub code: String,
    /// Secret-safe failure detail.
    pub message: String,
    /// Non-authoritative evidence that the coordinator may expose on failure.
    pub diagnostic_artifacts: Vec<RunDiagnosticArtifactV2>,
}

impl PreparedRunFailure {
    /// Construct a validated failure that must not produce a native report path.
    pub fn new(
        stage: FailureStageV2,
        code: impl Into<String>,
        message: impl Into<String>,
    ) -> Result<Self> {
        let code = code.into();
        let message = message.into();
        ensure!(
            !code.is_empty()
                && code.trim() == code
                && code
                    .bytes()
                    .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'_'),
            "prepared run failure code must be lowercase snake case"
        );
        ensure!(
            !message.trim().is_empty(),
            "prepared run failure message cannot be empty"
        );
        Ok(Self {
            stage,
            code,
            message,
            diagnostic_artifacts: Vec::new(),
        })
    }

    /// Reporting-policy failure after execution produced only a diagnostic artifact.
    pub fn reporting(code: impl Into<String>, message: impl Into<String>) -> Result<Self> {
        Self::new(FailureStageV2::Reporting, code, message)
    }

    /// Attach already-persisted, content-addressed diagnostic evidence.
    pub fn with_diagnostic_artifacts(
        mut self,
        mut artifacts: Vec<RunDiagnosticArtifactV2>,
    ) -> Result<Self> {
        for artifact in &artifacts {
            let kind = artifact.kind.as_str();
            ensure!(
                !kind.is_empty()
                    && kind.trim() == kind
                    && kind.bytes().all(|byte| {
                        byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'_'
                    }),
                "diagnostic artifact kind must be lowercase snake case"
            );
            ensure!(
                artifact.relative_path.is_relative()
                    && artifact
                        .relative_path
                        .components()
                        .all(|component| matches!(component, std::path::Component::Normal(_))),
                "diagnostic artifact path must be a normalized relative path"
            );
            ensure!(
                artifact.relative_path != std::path::Path::new("native-v2.json"),
                "diagnostic artifact cannot name the authoritative native report"
            );
            let digest = artifact.content_hash.as_str();
            ensure!(
                digest.len() == "blake3:".len() + 64
                    && digest.starts_with("blake3:")
                    && digest["blake3:".len()..]
                        .bytes()
                        .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase()),
                "diagnostic artifact hash must be a tagged lowercase BLAKE3 digest"
            );
        }
        artifacts.sort_by(|left, right| {
            (&left.kind, &left.relative_path).cmp(&(&right.kind, &right.relative_path))
        });
        ensure!(
            artifacts.windows(2).all(|pair| {
                pair[0].kind != pair[1].kind || pair[0].relative_path != pair[1].relative_path
            }),
            "diagnostic artifacts must have unique kind/path identities"
        );
        self.diagnostic_artifacts = artifacts;
        Ok(self)
    }
}

impl std::fmt::Display for PreparedRunFailure {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{}: {}", self.code, self.message)
    }
}

impl std::error::Error for PreparedRunFailure {}

/// Result returned after one prepared pair executes successfully.
#[derive(Debug)]
pub struct PreparedRunOutcome {
    /// Uncommitted native-v2 report returned to the process coordinator.
    pub native_report: NativeReport,
    /// Typed transport/workload facts joined during coordinator finalization.
    pub report_facts: ReportPairRunFacts,
    /// Additive transport/workload run metadata for the terminal response.
    pub run_metadata: BTreeMap<String, String>,
    /// Optional acknowledgement invoked only after the authoritative native
    /// report write and atomic rename succeed.
    pub report_commit: Option<Box<dyn PreparedReportCommit>>,
}

/// One completely prepared operation.
///
/// This trait intentionally has no `Send` bound: current high-performance
/// implementations own `Rc`/`RefCell` state on a current-thread Tokio runtime.
pub trait PreparedRunnerOperation: Debug {
    /// Execute the prepared run without serializing the authoritative report.
    fn execute(self: Box<Self>) -> Result<PreparedRunOutcome>;
}

/// Fully validated component selection retained between validation and prepare.
///
/// There is no transport×workload cell object: the selection retains the two
/// independently resolved factories and the descriptor predicate has already
/// admitted the pair. Run-level validation and preparation are dispatched to the
/// workload factory, which resolves the transport's dispatcher/placement factory
/// by `transport_id`.
pub struct ValidatedRunnerSelection {
    transport_id: String,
    workload_id: String,
    transport: Box<dyn ValidatedTransportConfig>,
    workload: Box<dyn ValidatedWorkloadConfig>,
    workload_factory: Arc<dyn WorkloadFactory>,
}

impl Debug for ValidatedRunnerSelection {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ValidatedRunnerSelection")
            .field("transport_id", &self.transport_id)
            .field("workload_id", &self.workload_id)
            .finish_non_exhaustive()
    }
}

impl ValidatedRunnerSelection {
    /// Selected transport ID.
    pub fn transport_id(&self) -> &str {
        &self.transport_id
    }

    /// Selected workload ID.
    pub fn workload_id(&self) -> &str {
        &self.workload_id
    }
}

/// Protocol-v2 transport/workload registration and selection over the one
/// unified [`AIPerfRegistry`].
///
/// Transports and workloads are independent name-keyed sub-registries. There is
/// no central compatibility matrix; each workload resolves and validates its
/// transport-specific execution path during preparation.
impl AIPerfRegistry {
    /// Register one transport factory, rejecting duplicate IDs.
    pub fn register_transport(&mut self, factory: Arc<dyn TransportFactory>) -> Result<()> {
        let id = checked_descriptor_id(factory.descriptor().id, "transport")?;
        self.transports
            .insert(id.into_string(), factory)
            .map_err(|DuplicateName(id)| anyhow!("duplicate runner transport ID {id:?}"))
    }

    /// Register one workload factory, rejecting duplicate IDs.
    pub fn register_workload(&mut self, factory: Arc<dyn WorkloadFactory>) -> Result<()> {
        let id = checked_descriptor_id(factory.descriptor().id, "workload")?;
        self.workloads
            .insert(id.into_string(), factory)
            .map_err(|DuplicateName(id)| anyhow!("duplicate runner workload ID {id:?}"))
    }

    /// Resolve one registered transport factory by ID for native-execution
    /// binding, returning `None` when no transport with that ID is compiled in.
    pub fn transport_factory(&self, id: &str) -> Option<Arc<dyn TransportFactory>> {
        self.transports.get(id).cloned()
    }

    /// Return transport descriptors in deterministic ID order.
    pub fn transport_descriptors(&self) -> Vec<&'static TransportDescriptor> {
        self.transports
            .values()
            .map(|factory| factory.descriptor())
            .collect()
    }

    /// Return workload descriptors in deterministic ID order.
    pub fn workload_descriptors(&self) -> Vec<&'static WorkloadDescriptor> {
        self.workloads
            .values()
            .map(|factory| factory.descriptor())
            .collect()
    }

    /// Resolve and statically validate one authored transport/workload selection.
    pub fn validate_selection(
        &self,
        transport: &NamedRunnerComponentSpecV2,
        workload: &NamedRunnerComponentSpecV2,
    ) -> Result<ValidatedRunnerSelection> {
        self.validate_selection_inner(transport, workload, None)
    }

    /// Resolve components and enforce workload-specific resource presence
    /// before transport preparation or any endpoint/sidecar work.
    pub fn validate_selection_for_run(
        &self,
        run: &AuthoredRunSpecV2,
    ) -> Result<ValidatedRunnerSelection> {
        self.validate_selection_inner(&run.transport, &run.workload, Some(run))
    }

    fn validate_selection_inner(
        &self,
        transport: &NamedRunnerComponentSpecV2,
        workload: &NamedRunnerComponentSpecV2,
        run: Option<&AuthoredRunSpecV2>,
    ) -> Result<ValidatedRunnerSelection> {
        let workload_factory = self
            .workloads
            .get(workload.id.as_str())
            .ok_or_else(|| unknown_component("workload", &workload.id, self.workloads.keys()))?;
        let validated_workload = workload_factory
            .validate(&workload.config)
            .map_err(|error| anyhow!("workload {:?}: {error:#}", workload.id.as_str()))?;
        let requirements = workload_factory.requirements(validated_workload.as_ref())?;
        if let Some(run) = run {
            validate_resource_requirements(run, requirements.resources)?;
        }

        let transport_factory = self
            .transports
            .get(transport.id.as_str())
            .ok_or_else(|| unknown_component("transport", &transport.id, self.transports.keys()))?;
        let validated_transport = transport_factory
            .validate(&transport.config, &requirements)
            .map_err(|error| anyhow!("transport {:?}: {error:#}", transport.id.as_str()))?;

        // Each workload resolves and validates its transport-specific execution
        // path during preparation.
        Ok(ValidatedRunnerSelection {
            transport_id: transport.id.as_str().to_owned(),
            workload_id: workload.id.as_str().to_owned(),
            transport: validated_transport,
            workload: validated_workload,
            workload_factory: workload_factory.clone(),
        })
    }

    /// Apply workload-owned run-level validation after component configs and
    /// endpoint profiles have been strictly decoded.
    pub fn validate_run(
        &self,
        run: &AuthoredRunSpecV2,
        context: &RunContext,
        selection: &ValidatedRunnerSelection,
    ) -> Result<()> {
        selection.workload_factory.validate_run(
            run,
            context,
            selection.transport.as_ref(),
            selection.workload.as_ref(),
            &selection.transport_id,
        )
    }

    /// Prepare through the coordinator-owned product registry and normalized
    /// endpoint-profile context.
    ///
    /// Preparation is dispatched to the workload factory, which resolves the
    /// transport's dispatcher/placement factory by `transport_id`.
    pub fn prepare_with_context(
        &self,
        run: &AuthoredRunSpecV2,
        context: &RunContext,
        selection: ValidatedRunnerSelection,
    ) -> Result<Box<dyn PreparedRunnerOperation>> {
        selection.workload_factory.prepare_with_context(
            run,
            context,
            selection.transport,
            selection.workload,
            &selection.transport_id,
        )
    }
}

fn checked_descriptor_id(value: &str, kind: &str) -> Result<ComponentId> {
    value
        .parse()
        .map_err(|error: String| anyhow!("invalid runner {kind} descriptor ID: {error}"))
}

fn validate_resource_requirements(
    run: &AuthoredRunSpecV2,
    requirements: ResourceRequirementsV2,
) -> Result<()> {
    for (resource, requirement) in requirements.entries() {
        let present = run.resource_is_present(resource);
        match (requirement, present) {
            (ResourceRequirementV2::Required, false) => {
                bail!(
                    "workload {:?} requires run.resources.{}",
                    run.workload.id.as_str(),
                    resource.field_name()
                );
            }
            (ResourceRequirementV2::Forbidden, true) => {
                bail!(
                    "workload {:?} forbids run.resources.{}",
                    run.workload.id.as_str(),
                    resource.field_name()
                );
            }
            (ResourceRequirementV2::Required, true)
            | (ResourceRequirementV2::Optional, _)
            | (ResourceRequirementV2::Forbidden, false) => {}
        }
    }
    Ok(())
}

fn unknown_component<'a>(
    kind: &str,
    requested: &ComponentId,
    available: impl Iterator<Item = &'a RegistryId>,
) -> anyhow::Error {
    let choices = available
        .map(RegistryId::as_str)
        .collect::<Vec<_>>()
        .join(", ");
    anyhow!(
        "runner {kind} {:?} is not compiled into this distribution; available: {}",
        requested.as_str(),
        if choices.is_empty() {
            "<none>"
        } else {
            &choices
        }
    )
}

fn extension_rejected(error: anyhow::Error) -> ExtensionError {
    ExtensionError::rejected(format!("{error:#}"))
}

/// Built-in native HTTP transport plus the transport-neutral executable
/// workloads (`scheduled`, `graph`).
///
/// The workloads run over any registered transport (http, grpc, dynosim) and
/// resolve transport-specific execution inside their own prepare, so the stock
/// distribution registers them once alongside the always-present HTTP path.
#[derive(Debug, Clone, Copy, Default)]
pub struct HttpExtension;

impl AIPerfExtension for HttpExtension {
    fn name(&self) -> &str {
        "aiperf.transport.http"
    }

    fn register(&self, registry: &mut AIPerfRegistry) -> Result<(), ExtensionError> {
        registry
            .register_transport(Arc::new(OnlineHttpTransportFactoryV2))
            .map_err(extension_rejected)?;
        // The always-built lightweight `dry_run` transport rides alongside the
        // HTTP path (no feature gate): it reuses the native scheduled/graph
        // runtime and swaps only the execution leaf, so it is available in every
        // distribution that has the HTTP path.
        crate::engine::dry_run::register_dry_run_transport(registry).map_err(extension_rejected)?;
        crate::engine::online_execution::register_online_workloads(registry)
            .map_err(extension_rejected)
    }
}

/// Built-in native gRPC transport (Clock-injected Tonic HTTP/2 channels).
///
/// gRPC is one transport over the same `TransportFactory` seam as
/// `http`/`dynosim`, gated behind the `grpc` Cargo feature: a lite/HTTP-only
/// runner drops this extension (and the `tonic` framework) entirely. The
/// transport-neutral Riva ASR/TTS/NLP and KServe-gRPC endpoint *dialects* are
/// pure request/response formatting with no `tonic`/`prost` imports, so they
/// stay registered by [`BuiltinEndpointsExtension`] regardless and simply fail
/// closed at transport selection when `grpc` is compiled out.
///
/// [`BuiltinEndpointsExtension`]: crate::extensions::BuiltinEndpointsExtension
#[cfg(feature = "grpc")]
#[derive(Debug, Clone, Copy, Default)]
pub struct GrpcExtension;

#[cfg(feature = "grpc")]
impl AIPerfExtension for GrpcExtension {
    fn name(&self) -> &str {
        "aiperf.transport.grpc"
    }

    fn register(&self, registry: &mut AIPerfRegistry) -> Result<(), ExtensionError> {
        registry
            .register_transport(Arc::new(OnlineGrpcTransportFactoryV2))
            .map_err(extension_rejected)
    }
}

/// Feature-gated Dynamo co-simulation transports (`dynosim_offline`,
/// `dynosim_online`).
#[cfg(feature = "dynosim")]
#[derive(Debug, Clone, Copy, Default)]
pub struct DynosimExtension;

#[cfg(feature = "dynosim")]
impl AIPerfExtension for DynosimExtension {
    fn name(&self) -> &str {
        "aiperf.transport.dynosim"
    }

    fn register(&self, registry: &mut AIPerfRegistry) -> Result<(), ExtensionError> {
        crate::engine::offline_execution::register_dynosim_transport(registry)
            .map_err(extension_rejected)
    }
}

/// Strict validated config owned by the built-in `http` transport.
///
/// Endpoint profiles own inference transport policy. The optional client block
/// is accepted only when a workload requests the isolated control-plane HTTP
/// capability, so no transport field can become inert on ordinary inference or
/// source-free archive synchronization paths.
#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct OnlineHttpTransportConfigV2 {
    /// Transport-wide ceilings inherited by every control-plane source handle.
    #[serde(default)]
    pub client: OnlineHttpControlClientConfigV2,
}

/// Strict transport ceiling for isolated control-plane HTTP.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct OnlineHttpControlClientConfigV2 {
    /// Maximum DNS/TCP/TLS/HTTP-handshake lifetime for any source.
    #[serde(default)]
    pub connect_timeout_ns: Option<i64>,
}

/// Strict validated config owned by the built-in `grpc` transport.
///
/// Endpoint profiles own targets, deadlines, metadata, and channel-reuse
/// policy, so the transport object remains intentionally empty and strict.
#[cfg(feature = "grpc")]
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OnlineGrpcTransportConfigV2 {}

/// Strict built-in scheduled-workload authored config.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScheduledWorkloadConfigV2 {
    /// Number of local execution workers.
    pub worker_count: usize,
    /// Dataset-factory-owned authored object.
    pub dataset: Box<RawValue>,
    /// Tokenizer-factory-owned authored object.
    pub tokenizer: Box<RawValue>,
    /// Phase-factory-owned authored objects.
    pub phases: Vec<PhaseSpec>,
    /// Optional run-failure policy; absent selects the scheduled default
    /// (resilient — record failed requests and continue).
    #[serde(default)]
    pub failure_policy: Option<OnFailure>,
}

impl Debug for ScheduledWorkloadConfigV2 {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ScheduledWorkloadConfigV2")
            .field("worker_count", &self.worker_count)
            .field("phase_count", &self.phases.len())
            .finish_non_exhaustive()
    }
}

/// Strict built-in direct Graph-IR workload authored config.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GraphWorkloadConfigV2 {
    /// Number of whole-trace placement workers.
    pub worker_count: usize,
    /// Direct graph-input-adapter-owned authored object.
    pub dataset: Box<RawValue>,
    /// Tokenizer-factory-owned authored object.
    pub tokenizer: Box<RawValue>,
    /// Ordered graph phase policy objects.
    pub phases: Vec<PhaseSpec>,
    /// Optional run-failure policy; absent selects the graph default
    /// (fail-fast — abort the run on the first non-cancellation failure).
    #[serde(default)]
    pub failure_policy: Option<OnFailure>,
    /// WEKA reconstruction semantics selector (`legacy`|`graph-ir`); absent/`graph-ir`
    /// uses the graph-ir path, `legacy` routes weka runs to the byte-exact AgentX
    /// agentic pipeline (requires the `agentx` feature).
    #[serde(default)]
    pub weka_semantics: Option<String>,
    /// Legacy AgentX replay global idle cap, seconds.
    #[serde(default)]
    pub system_idle_gap_cap_seconds: Option<f64>,
    /// Ignore recorded trace inter-message/inter-request delays: fire every node
    /// as soon as its inputs are ready (sets `ExecutorFlags::ignore_edge_delays`).
    #[serde(default)]
    pub ignore_trace_delays: bool,
}

impl Debug for GraphWorkloadConfigV2 {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("GraphWorkloadConfigV2")
            .field("worker_count", &self.worker_count)
            .field("phase_count", &self.phases.len())
            .finish_non_exhaustive()
    }
}

/// Strict built-in static-accuracy workload authored config.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StaticAccuracyWorkloadConfigV2 {
    /// Number of local inference workers.
    pub worker_count: usize,
    /// Evaluator dataset intent retained for canonical worker preparation.
    pub dataset: Box<RawValue>,
    /// Tokenizer-factory-owned authored object.
    pub tokenizer: Box<RawValue>,
    /// Ordered scheduled inference phase objects.
    pub phases: Vec<PhaseSpec>,
    /// Canonical evaluator-owned authored accuracy object.
    pub accuracy: Box<RawValue>,
}

impl Debug for StaticAccuracyWorkloadConfigV2 {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("StaticAccuracyWorkloadConfigV2")
            .field("worker_count", &self.worker_count)
            .field("phase_count", &self.phases.len())
            .finish_non_exhaustive()
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ResetKvCacheConfigV2 {
    #[serde(default, alias = "timeoutSeconds")]
    timeout_seconds: Option<f64>,
    #[serde(default)]
    path: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ServerProfilerConfigV2 {
    #[serde(default, alias = "timeoutSeconds")]
    timeout_seconds: Option<f64>,
    #[serde(default, alias = "startPath")]
    start_path: Option<String>,
    #[serde(default, alias = "stopPath")]
    stop_path: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct EndpointProfileConfigV2 {
    id: String,
    #[serde(rename = "type")]
    endpoint_id: EndpointId,
    urls: Vec<String>,
    #[serde(default)]
    path: Option<String>,
    #[serde(default, alias = "resetKvCache")]
    reset_kv_cache: Option<ResetKvCacheConfigV2>,
    #[serde(default, alias = "serverProfiler")]
    server_profiler: Option<ServerProfilerConfigV2>,
    #[serde(default)]
    streaming: bool,
    #[serde(default)]
    request_content_type: Option<RequestContentType>,
    #[serde(default)]
    template: Option<String>,
    #[serde(default)]
    response_field: Option<String>,
    #[serde(default = "default_timeout_seconds")]
    timeout_seconds: f64,
    #[serde(default = "default_polling_interval_seconds")]
    polling_interval_seconds: f64,
    #[serde(default)]
    download_video_content: bool,
    #[serde(default)]
    wait_for_model_timeout: f64,
    #[serde(default = "default_wait_for_model_interval")]
    wait_for_model_interval: f64,
    #[serde(default = "default_wait_for_model_mode")]
    wait_for_model_mode: String,
    #[serde(
        default,
        rename = "use_legacy_max_tokens",
        alias = "useLegacyMaxTokens"
    )]
    use_legacy_max_tokens: bool,
    #[serde(default)]
    use_server_token_count: bool,
    #[serde(default)]
    headers: BTreeMap<String, String>,
    #[serde(default)]
    api_key: Option<String>,
    #[serde(default)]
    extra: Map<String, Value>,
    #[serde(default)]
    connection_reuse: ConnectionReuseStrategy,
    #[serde(default)]
    session_header: Option<String>,
    #[serde(default)]
    http2: bool,
    #[serde(default = "default_ssl_verify", alias = "sslVerify")]
    ssl_verify: bool,
    /// Optional Unix-domain socket path. When set, the HTTP transport connects
    /// over this socket (HTTP/1.1) instead of TCP; the endpoint `urls` still
    /// supply the request path and `Host` header. Carried out-of-band alongside
    /// `urls` (like `ssl_verify`/`http2`) because a `unix://` URL is rejected by
    /// the frontend's http(s)-only scheme validators.
    #[serde(default)]
    uds_path: Option<String>,
    #[serde(default = "default_connection_limit")]
    connection_limit: usize,
    #[serde(default = "default_keepalive_timeout")]
    keepalive_timeout: f64,
    /// Extra attempts to (re)establish a connection after a connect-phase
    /// failure (DNS/TCP/TLS/handshake). Retries apply only before any request
    /// bytes are sent, so a partially processed request is never re-issued.
    /// `0` (the default) preserves fail-fast behavior.
    #[serde(default)]
    max_connect_retries: u32,
    /// Base linear backoff, in seconds, slept between connect retries. Attempt
    /// `n` (1-based) waits `connect_retry_backoff_seconds * n`. `0` (the
    /// default) retries with no wait.
    #[serde(default)]
    connect_retry_backoff_seconds: f64,
    /// Explicit forward-proxy URL for benchmark traffic. When set, HTTP CONNECT
    /// tunnels every connection to this endpoint through the proxy (applied as
    /// authored, including for loopback). Adds the proxy hop to connection
    /// establishment; steady-state per-request latency over a warm tunnel is
    /// unaffected.
    #[serde(default)]
    proxy: Option<String>,
    /// Honor the ambient proxy environment (`HTTPS_PROXY`/`NO_PROXY`, loopback
    /// excluded) for benchmark traffic. Ignored when `proxy` is set.
    #[serde(default)]
    proxy_from_env: bool,
}

/// One statically validated endpoint profile and its normalized policy.
#[derive(Clone, Debug, PartialEq)]
pub struct ValidatedEndpointProfileV2 {
    /// Run-local profile name referenced by workload inputs.
    pub profile_id: String,
    /// Canonical endpoint dialect selected from the frozen registry.
    pub endpoint_id: EndpointId,
    /// Identity-free endpoint policy after descriptor/factory normalization.
    pub config: RawEndpointConfig,
    /// Authored HTTP connection reuse policy.
    pub connection_reuse: ConnectionReuseStrategy,
    /// Fully validated HTTP client policy retained directly for worker-local
    /// transport construction.
    pub client: ClientConfig,
    /// Optional authored session-affinity header name.
    pub session_header: Option<String>,
}

/// Coordinator-owned immutable context shared by static run-level validation and
/// complete execution preparation.
///
/// One fresh runner process builds exactly one [`AIPerfRegistry`]. Workload
/// factories clone this cheap handle into their prepared operation; they never
/// invoke a built-in registry factory or independently decode endpoint policy.
#[derive(Clone)]
pub struct RunContext {
    distribution_id: String,
    product_registry: Arc<AIPerfRegistry>,
    execution_factories: ExecutionFactories,
    graph_inputs: Arc<dyn GraphInputAdapterResolver>,
    dataset_inputs: Arc<dyn DatasetInputAdapterResolver>,
    sidecar_inputs: Arc<PreparedSidecarInputs>,
    endpoint_profiles: Arc<Vec<ValidatedEndpointProfileV2>>,
    endpoint_profile_indexes: Arc<BTreeMap<String, usize>>,
}

impl Debug for RunContext {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("RunContext")
            .field("distribution_id", &self.distribution_id)
            .field("execution_factories", &self.execution_factories)
            .field(
                "sidecar_inputs",
                &self.sidecar_inputs.ids().collect::<Vec<_>>(),
            )
            .field(
                "endpoint_profiles",
                &self
                    .endpoint_profiles
                    .iter()
                    .map(|profile| profile.profile_id.as_str())
                    .collect::<Vec<_>>(),
            )
            .finish_non_exhaustive()
    }
}

impl RunContext {
    /// Freeze one validated profile collection beside the process registry.
    pub fn new(
        distribution_id: impl Into<String>,
        product_registry: Arc<AIPerfRegistry>,
        execution_factories: ExecutionFactories,
        graph_inputs: Arc<dyn GraphInputAdapterResolver>,
        dataset_inputs: Arc<dyn DatasetInputAdapterResolver>,
        sidecar_inputs: Arc<PreparedSidecarInputs>,
        profiles: Vec<ValidatedEndpointProfileV2>,
    ) -> Result<Self> {
        let mut endpoint_profile_indexes = BTreeMap::new();
        for (index, profile) in profiles.iter().enumerate() {
            let profile_id = profile.profile_id.clone();
            ensure!(
                endpoint_profile_indexes
                    .insert(profile_id.clone(), index)
                    .is_none(),
                "duplicate validated endpoint profile ID {profile_id:?}"
            );
        }
        let distribution_id = distribution_id.into();
        ensure!(
            distribution_id
                .strip_prefix("blake3:")
                .is_some_and(|value| value.len() == 64
                    && value
                        .bytes()
                        .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())),
            "runner context distribution ID must be blake3 plus 64 lowercase hexadecimal digits"
        );
        Ok(Self {
            distribution_id,
            product_registry,
            execution_factories,
            graph_inputs,
            dataset_inputs,
            sidecar_inputs,
            endpoint_profiles: Arc::new(profiles),
            endpoint_profile_indexes: Arc::new(endpoint_profile_indexes),
        })
    }

    /// Exact executable distribution identity validated by the coordinator.
    pub fn distribution_id(&self) -> &str {
        &self.distribution_id
    }

    /// Borrow the single frozen product registry composed by the coordinator.
    pub fn product_registry(&self) -> &AIPerfRegistry {
        self.product_registry.as_ref()
    }

    /// Clone the cheap shared registry handle into a prepared operation.
    pub fn product_registry_handle(&self) -> Arc<AIPerfRegistry> {
        self.product_registry.clone()
    }

    /// Borrow the execution-placement universe frozen by the coordinator.
    pub fn execution_factories(&self) -> &ExecutionFactories {
        &self.execution_factories
    }

    /// Retain the exact execution-placement universe in a prepared operation.
    pub fn execution_factories_handle(&self) -> ExecutionFactories {
        self.execution_factories.clone()
    }

    /// Borrow the frozen graph-input resolver composed by the coordinator.
    pub fn graph_inputs(&self) -> &dyn GraphInputAdapterResolver {
        self.graph_inputs.as_ref()
    }

    /// Borrow the frozen dataset-input resolver composed by the coordinator.
    pub fn dataset_inputs(&self) -> &dyn DatasetInputAdapterResolver {
        self.dataset_inputs.as_ref()
    }

    /// Clone the dataset-input resolver into a prepared scheduled operation.
    pub fn dataset_inputs_handle(&self) -> Arc<dyn DatasetInputAdapterResolver> {
        self.dataset_inputs.clone()
    }

    /// Borrow the exact sidecar adapter outputs prepared by the coordinator.
    pub fn sidecar_inputs(&self) -> &PreparedSidecarInputs {
        self.sidecar_inputs.as_ref()
    }

    /// Retain the same prepared sidecar bundle through pair execution.
    pub fn sidecar_inputs_handle(&self) -> Arc<PreparedSidecarInputs> {
        self.sidecar_inputs.clone()
    }

    /// Build the coordinator-owned native-report identity from frozen values.
    ///
    /// This is the sole bridge from process composition into report
    /// metadata. It never reopens authored JSON or asks a pair adapter to
    /// repeat endpoint or extension discovery.
    pub fn report_run_metadata(
        &self,
        distribution_id: impl Into<String>,
        transport_id: impl Into<String>,
        workload_id: impl Into<String>,
    ) -> Result<ReportRunMetadata> {
        let extensions = self
            .product_registry
            .extension_names()
            .map(|name| ReportExtensionIdentity::new(name, None))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let endpoint_profiles = self
            .endpoint_profiles
            .iter()
            .map(|profile| {
                ReportEndpointProfileIdentity::new(
                    profile.profile_id.clone(),
                    profile.endpoint_id.to_string(),
                )
            })
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(ReportRunMetadata::new(
            distribution_id,
            transport_id,
            workload_id,
            extensions,
            endpoint_profiles,
        )?)
    }

    /// Resolve a run-local endpoint profile without reparsing authored JSON.
    pub fn endpoint_profile(&self, profile_id: &str) -> Result<&ValidatedEndpointProfileV2> {
        self.endpoint_profile_indexes
            .get(profile_id)
            .map(|index| &self.endpoint_profiles[*index])
            .ok_or_else(|| {
                anyhow!(
                    "endpoint profile {profile_id:?} is not defined; available: {}",
                    self.endpoint_profiles
                        .iter()
                        .map(|profile| profile.profile_id.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            })
    }

    /// Resolve the endpoint profile identified by `default`.
    pub fn default_endpoint_profile(&self) -> Result<&ValidatedEndpointProfileV2> {
        self.endpoint_profile("default")
    }

    /// Iterate normalized profiles in authored order.
    ///
    /// Keeping this sequence intact lets prepared endpoint tables and report
    /// run metadata consume the same coordinator-owned identity ordering.
    pub fn endpoint_profiles(
        &self,
    ) -> impl ExactSizeIterator<Item = (&str, &ValidatedEndpointProfileV2)> {
        self.endpoint_profiles
            .iter()
            .map(|profile| (profile.profile_id.as_str(), profile))
    }

    /// Retain the exact validated profile collection in a prepared adapter.
    ///
    /// This is intentionally a clone of the shared handle, not a projection
    /// into an execution-specific DTO. Transport/workload adapters therefore
    /// pass the same normalized objects from validation into worker-local
    /// endpoint and transport preparation.
    pub fn endpoint_profiles_handle(&self) -> Arc<Vec<ValidatedEndpointProfileV2>> {
        self.endpoint_profiles.clone()
    }
}

/// Strictly decode and statically validate every authored endpoint profile
/// through the exact frozen endpoint registry.
///
/// Preparation performed here is local and side-effect free: it compiles any
/// endpoint-owned template/selector and constructs readiness request policy,
/// but never creates artifacts, opens sockets, loads datasets, or starts a
/// worker. Execute repeats this validation and retains fresh worker-local
/// bindings during full preparation.
pub fn validate_endpoint_profiles_v2(
    run: &AuthoredRunSpecV2,
    endpoints: &EndpointRegistry,
) -> Result<Vec<ValidatedEndpointProfileV2>> {
    let mut validated = Vec::with_capacity(run.endpoints.profiles.len());
    for (index, authored) in run.endpoints.profiles.iter().enumerate() {
        let config = strict_decode::<EndpointProfileConfigV2>(
            authored,
            &format!("endpoint profile {index}"),
        )?;
        ensure!(
            !config.id.is_empty() && config.id.trim() == config.id,
            "endpoint profile {index}.id must be non-empty and contain no surrounding whitespace"
        );
        ensure!(
            !config.urls.is_empty(),
            "endpoint profile {index}.urls cannot be empty"
        );
        ensure!(
            config.connection_limit > 0,
            "endpoint profile {index}.connection_limit must be positive"
        );
        ensure!(
            matches!(
                config.wait_for_model_mode.as_str(),
                "models" | "inference" | "both"
            ),
            "endpoint profile {index}.wait_for_model_mode must be models, inference, or both"
        );
        if let Some(header) = &config.session_header {
            ensure!(
                !header.trim().is_empty() && header.trim() == header,
                "endpoint profile {index}.session_header must be non-empty and contain no surrounding whitespace"
            );
        }
        let canonical_id = endpoints.canonical_id(&config.endpoint_id)?.clone();
        let raw = RawEndpointConfig {
            urls: config.urls,
            path: config.path,
            reset_kv_cache: config.reset_kv_cache.map(|config| ResetKvCacheConfig {
                timeout_seconds: config.timeout_seconds,
                path: config.path,
            }),
            server_profiler: config.server_profiler.map(|config| ServerProfilerConfig {
                timeout_seconds: config.timeout_seconds,
                start_path: config.start_path,
                stop_path: config.stop_path,
            }),
            streaming: config.streaming,
            request_content_type: config.request_content_type,
            template: config.template,
            response_field: config.response_field,
            timeout_seconds: config.timeout_seconds,
            polling_interval_seconds: config.polling_interval_seconds,
            download_video_content: config.download_video_content,
            wait_for_model_timeout: config.wait_for_model_timeout,
            wait_for_model_interval: config.wait_for_model_interval,
            wait_for_model_mode: config.wait_for_model_mode,
            wait_for_model_interval_set: true,
            wait_for_model_mode_set: true,
            use_legacy_max_tokens: config.use_legacy_max_tokens,
            use_server_token_count: config.use_server_token_count,
            headers: config.headers,
            api_key: config.api_key,
            extra: (!config.extra.is_empty()).then_some(config.extra),
        };
        let readiness_enabled = raw.wait_for_model_timeout > 0.0;
        let prepared = endpoints.prepare(&canonical_id, raw)?;
        if readiness_enabled {
            for model in &run.models.items {
                prepared.readiness_policy(&model.name)?;
            }
        }
        let effective = prepared.config().to_raw();
        let proxy_target = effective.urls.first().and_then(|u| url::Url::parse(u).ok());
        let proxy = crate::transport::http::client::proxy::resolve(
            config.proxy.as_deref(),
            config.proxy_from_env,
            proxy_target.as_ref(),
        )
        .map_err(|e| anyhow!("endpoint profile {index}: {e}"))?;
        if proxy.is_some() {
            tracing::warn!(
                profile = %config.id,
                "benchmark traffic is routed through a proxy; connection-establishment \
                 latency includes the proxy hop (per-request latency over a warm tunnel is not)"
            );
        }
        let client = ClientConfig {
            total_timeout_ns: seconds_to_optional_ns(
                effective.timeout_seconds,
                &format!("endpoint profile {index}.timeout_seconds"),
            )?,
            ssl_verify: config.ssl_verify,
            // A Unix-domain socket connection is HTTP/1.1-only (the transport's
            // UDS branch hard-codes h1), so force Http1Only when a socket path is
            // set rather than silently downgrading an authored `http2: true`.
            http_version: if config.uds_path.is_some() {
                HttpVersion::Http1Only
            } else if config.http2 {
                HttpVersion::Http2PriorKnowledge
            } else {
                HttpVersion::Auto
            },
            uds_path: config.uds_path.clone(),
            keepalive_ns: Some(seconds_to_ns(
                config.keepalive_timeout,
                &format!("endpoint profile {index}.keepalive_timeout"),
            )?),
            max_connections_per_origin: config.connection_limit,
            max_connect_retries: config.max_connect_retries,
            connect_retry_backoff_ns: seconds_to_ns(
                config.connect_retry_backoff_seconds,
                &format!("endpoint profile {index}.connect_retry_backoff_seconds"),
            )?,
            proxy,
            ..ClientConfig::default()
        };
        validated.push(ValidatedEndpointProfileV2 {
            profile_id: config.id,
            endpoint_id: canonical_id,
            config: effective,
            connection_reuse: config.connection_reuse,
            client,
            session_header: config.session_header,
        });
    }
    Ok(validated)
}

const fn default_timeout_seconds() -> f64 {
    6.0 * 60.0 * 60.0
}

const fn default_polling_interval_seconds() -> f64 {
    0.1
}

const fn default_wait_for_model_interval() -> f64 {
    5.0
}

fn default_wait_for_model_mode() -> String {
    "inference".to_owned()
}

const fn default_ssl_verify() -> bool {
    true
}

const fn default_connection_limit() -> usize {
    2_500
}

const fn default_keepalive_timeout() -> f64 {
    300.0
}

fn seconds_to_ns(value: f64, field: &str) -> Result<i64> {
    ensure!(
        value.is_finite() && value >= 0.0,
        "{field} must be finite and non-negative"
    );
    let nanoseconds = value * 1_000_000_000.0;
    ensure!(
        nanoseconds < i64::MAX as f64,
        "{field} exceeds the native Clock range"
    );
    let nanoseconds = nanoseconds.round() as i64;
    ensure!(
        value == 0.0 || nanoseconds > 0,
        "{field} must be zero or at least one native Clock nanosecond"
    );
    Ok(nanoseconds)
}

fn seconds_to_optional_ns(value: f64, field: &str) -> Result<Option<i64>> {
    let nanoseconds = seconds_to_ns(value, field)?;
    Ok((nanoseconds > 0).then_some(nanoseconds))
}

/// Built-in online HTTP transport descriptor.
pub static ONLINE_HTTP_TRANSPORT_DESCRIPTOR: TransportDescriptor = TransportDescriptor {
    id: "http",
    description: "Clock-injected native HTTP transport over a real clock",
    clock: ClockKind::Real,
    features: &["control_plane_http", "h1", "h2c", "http", "tls", "uds"],
    url_schemes: &["http", "https"],
};

/// Built-in online gRPC transport descriptor.
#[cfg(feature = "grpc")]
pub static ONLINE_GRPC_TRANSPORT_DESCRIPTOR: TransportDescriptor = TransportDescriptor {
    id: "grpc",
    description: "Clock-injected native gRPC transport over Tonic HTTP/2 channels",
    clock: ClockKind::Real,
    features: &["grpc", "h2", "tls"],
    url_schemes: &["grpc", "grpcs"],
};

/// Built-in scheduled workload descriptor.
pub static SCHEDULED_WORKLOAD_DESCRIPTOR: WorkloadDescriptor = WorkloadDescriptor {
    id: "scheduled",
    description: "Clock-paced scheduled request and session workloads",
};

/// Built-in direct Graph-IR workload descriptor.
pub static GRAPH_WORKLOAD_DESCRIPTOR: WorkloadDescriptor = WorkloadDescriptor {
    id: "graph",
    description: "Direct authored DAG lowering and whole-trace Graph-IR execution",
};

/// Built-in static-accuracy workload descriptor (HTTP only).
pub static STATIC_ACCURACY_WORKLOAD_DESCRIPTOR: WorkloadDescriptor = WorkloadDescriptor {
    id: "static_accuracy",
    description: "Canonical-evaluator static accuracy over the scheduled HTTP path",
};

#[derive(Debug)]
struct OnlineHttpTransportFactoryV2;

impl TransportFactory for OnlineHttpTransportFactoryV2 {
    fn descriptor(&self) -> &'static TransportDescriptor {
        &ONLINE_HTTP_TRANSPORT_DESCRIPTOR
    }

    fn validate(
        &self,
        authored: &RawValue,
        requirements: &WorkloadRequirements,
    ) -> Result<Box<dyn ValidatedTransportConfig>> {
        let config =
            strict_decode::<OnlineHttpTransportConfigV2>(authored, "http transport config")?;
        ensure!(
            config
                .client
                .connect_timeout_ns
                .is_none_or(|timeout| timeout > 0),
            "http control client connect_timeout_ns must be positive"
        );
        ensure!(
            requirements
                .transport_features
                .contains("control_plane_http")
                || config.client == OnlineHttpControlClientConfigV2::default(),
            "http client config is forbidden when the workload does not request control_plane_http"
        );
        Ok(Box::new(config))
    }

    fn native_execution(
        &self,
        _config: &dyn ValidatedTransportConfig,
        context: &RunContext,
    ) -> Result<Option<Arc<dyn NativeTransportExecution>>> {
        Ok(Some(Arc::new(
            crate::engine::online_execution::HttpNativeExecution::new(
                context.execution_factories().http_handle(),
            ),
        )))
    }
}

#[cfg(feature = "grpc")]
#[derive(Debug)]
struct OnlineGrpcTransportFactoryV2;

#[cfg(feature = "grpc")]
impl TransportFactory for OnlineGrpcTransportFactoryV2 {
    fn descriptor(&self) -> &'static TransportDescriptor {
        &ONLINE_GRPC_TRANSPORT_DESCRIPTOR
    }

    fn validate(
        &self,
        authored: &RawValue,
        _requirements: &WorkloadRequirements,
    ) -> Result<Box<dyn ValidatedTransportConfig>> {
        Ok(Box::new(strict_decode::<OnlineGrpcTransportConfigV2>(
            authored,
            "grpc transport config",
        )?))
    }

    fn native_execution(
        &self,
        _config: &dyn ValidatedTransportConfig,
        _context: &RunContext,
    ) -> Result<Option<Arc<dyn NativeTransportExecution>>> {
        Ok(Some(Arc::new(
            crate::engine::grpc_execution::GrpcNativeExecution::new(),
        )))
    }
}

pub(crate) fn strict_decode<T>(authored: &RawValue, label: &str) -> Result<T>
where
    T: DeserializeOwned,
{
    // The workspace enables serde_json's `arbitrary_precision` feature for >u64
    // recorded-trace hashes. Streaming `from_str` then represents every number as an
    // internal map, so any `#[serde(flatten)]` (e.g. `PhaseSpec`'s
    // `PhaseCommonSpec`) or `#[serde(untagged)]` field whose type is `f64` fails
    // with "invalid type: map, expected f64". Buffering through `serde_json::Value`
    // first reconstructs numbers in a form the derived impls accept, at the cost
    // of one extra parse.
    let value: serde_json::Value =
        serde_json::from_str(authored.get()).map_err(|error| anyhow!("{label}: {error}"))?;
    serde_json::from_value(value).map_err(|error| anyhow!("{label}: {error}"))
}

pub(crate) fn validate_common_workload(
    worker_count: usize,
    dataset: &RawValue,
    tokenizer: &RawValue,
    phases: &[PhaseSpec],
) -> Result<()> {
    ensure!(worker_count > 0, "workload worker_count must be positive");
    let dataset = raw_object(dataset, "workload dataset")?;
    ensure!(
        dataset
            .get("type")
            .and_then(Value::as_str)
            .is_some_and(|value| !value.trim().is_empty()),
        "workload dataset.type must be a non-empty string"
    );
    raw_object(tokenizer, "workload tokenizer")?;
    ensure!(!phases.is_empty(), "workload phases cannot be empty");
    let mut names = BTreeSet::new();
    for (index, phase) in phases.iter().enumerate() {
        let name = phase.common().name.as_str();
        ensure!(
            !name.trim().is_empty(),
            "workload phase {index}.name must be a non-empty string"
        );
        ensure!(
            names.insert(name.to_owned()),
            "duplicate workload phase name {name:?}"
        );
    }
    Ok(())
}

fn raw_object(raw: &RawValue, label: &str) -> Result<Map<String, Value>> {
    let value: Value =
        serde_json::from_str(raw.get()).map_err(|error| anyhow!("{label}: {error}"))?;
    value
        .as_object()
        .cloned()
        .ok_or_else(|| anyhow!("{label} must be a JSON object"))
}

/// Config-level requirements for a built-in inference workload.
///
/// Built-in inference workloads use the inference resource matrix without
/// transport feature requirements.
pub(crate) fn inference_workload_requirements() -> WorkloadRequirements {
    WorkloadRequirements {
        transport_features: BTreeSet::new(),
        resources: ResourceRequirementsV2::inference(),
    }
}

#[cfg(test)]
mod tests {
    use serde::Deserialize;
    use serde_json::Value;

    use super::*;
    use crate::extensions::AIPerfRegistryFactory;

    static TRANSPORT: TransportDescriptor = TransportDescriptor {
        id: "acme_remote",
        description: "fixture remote placement",
        clock: ClockKind::Real,
        features: &["http"],
        url_schemes: &[],
    };
    static WORKLOAD: WorkloadDescriptor = WorkloadDescriptor {
        id: "acme_workload",
        description: "fixture workload",
    };

    #[derive(Debug, Deserialize)]
    #[serde(deny_unknown_fields)]
    struct TransportConfig {
        node: usize,
    }

    #[derive(Debug, Deserialize)]
    #[serde(deny_unknown_fields)]
    struct WorkloadConfig {
        message: String,
    }

    #[derive(Debug)]
    struct Transport;

    impl TransportFactory for Transport {
        fn descriptor(&self) -> &'static TransportDescriptor {
            &TRANSPORT
        }

        fn validate(
            &self,
            authored: &RawValue,
            _requirements: &WorkloadRequirements,
        ) -> Result<Box<dyn ValidatedTransportConfig>> {
            Ok(Box::new(serde_json::from_str::<TransportConfig>(
                authored.get(),
            )?))
        }
    }

    #[derive(Debug)]
    struct Workload;

    impl WorkloadFactory for Workload {
        fn descriptor(&self) -> &'static WorkloadDescriptor {
            &WORKLOAD
        }

        fn validate(&self, authored: &RawValue) -> Result<Box<dyn ValidatedWorkloadConfig>> {
            Ok(Box::new(serde_json::from_str::<WorkloadConfig>(
                authored.get(),
            )?))
        }

        fn requirements(
            &self,
            config: &dyn ValidatedWorkloadConfig,
        ) -> Result<WorkloadRequirements> {
            let config = config
                .as_any()
                .downcast_ref::<WorkloadConfig>()
                .ok_or_else(|| anyhow!("wrong workload config type"))?;
            ensure!(!config.message.is_empty(), "message cannot be empty");
            Ok(WorkloadRequirements {
                transport_features: BTreeSet::from(["http".to_owned()]),
                resources: ResourceRequirementsV2::inference(),
            })
        }

        fn validate_run(
            &self,
            _run: &AuthoredRunSpecV2,
            _context: &RunContext,
            transport: &dyn ValidatedTransportConfig,
            workload: &dyn ValidatedWorkloadConfig,
            _transport_id: &str,
        ) -> Result<()> {
            let transport = transport
                .as_any()
                .downcast_ref::<TransportConfig>()
                .ok_or_else(|| anyhow!("wrong transport config type"))?;
            let workload = workload
                .as_any()
                .downcast_ref::<WorkloadConfig>()
                .ok_or_else(|| anyhow!("wrong workload config type"))?;
            ensure!(transport.node > 0, "node must be positive");
            ensure!(!workload.message.is_empty(), "message cannot be empty");
            Ok(())
        }

        fn prepare_with_context(
            &self,
            _run: &AuthoredRunSpecV2,
            _context: &RunContext,
            transport: Box<dyn ValidatedTransportConfig>,
            workload: Box<dyn ValidatedWorkloadConfig>,
            _transport_id: &str,
        ) -> Result<Box<dyn PreparedRunnerOperation>> {
            let node = ValidatedTransportConfig::as_any(transport.as_ref())
                .downcast_ref::<TransportConfig>()
                .ok_or_else(|| anyhow!("wrong transport config type"))?
                .node;
            let message = ValidatedWorkloadConfig::as_any(workload.as_ref())
                .downcast_ref::<WorkloadConfig>()
                .ok_or_else(|| anyhow!("wrong workload config type"))?
                .message
                .clone();
            Ok(Box::new(Operation { node, message }))
        }
    }

    #[derive(Debug)]
    struct Operation {
        node: usize,
        message: String,
    }

    impl PreparedRunnerOperation for Operation {
        fn execute(self: Box<Self>) -> Result<PreparedRunOutcome> {
            Ok(PreparedRunOutcome {
                native_report: NativeReport::new(
                    &crate::metrics_core::AccumulatorSummary::new(),
                    None,
                ),
                report_facts: ReportPairRunFacts::new(),
                run_metadata: BTreeMap::from([(
                    "fixture".into(),
                    format!("{}-{}", self.node, self.message),
                )]),
                report_commit: None,
            })
        }
    }

    fn registry() -> AIPerfRegistry {
        let mut registry = AIPerfRegistry::builtin().unwrap();
        registry.register_transport(Arc::new(Transport)).unwrap();
        registry.register_workload(Arc::new(Workload)).unwrap();
        registry
    }

    fn component(id: &str, config: Value) -> NamedRunnerComponentSpecV2 {
        NamedRunnerComponentSpecV2 {
            id: id.parse().unwrap(),
            config: RawValue::from_string(config.to_string()).unwrap(),
        }
    }

    fn authored_run(resources: Value) -> AuthoredRunSpecV2 {
        serde_json::from_str(
            &serde_json::json!({
                "identity": {"benchmark_id": "resource-test"},
                "artifact_target": "/tmp/resource-test",
                "transport": {"type": "acme_remote", "config": {"node": 4}},
                "workload": {"type": "acme_workload", "config": {"message": "hello"}},
                "resources": resources,
            })
            .to_string(),
        )
        .unwrap()
    }

    #[test]
    fn custom_selection_validates_without_a_core_enum_or_match() {
        let registry = registry();
        let selection = registry
            .validate_selection(
                &component("acme_remote", serde_json::json!({"node": 4})),
                &component("acme_workload", serde_json::json!({"message": "hello"})),
            )
            .unwrap();
        assert_eq!(selection.transport_id(), "acme_remote");
        assert_eq!(selection.workload_id(), "acme_workload");
    }

    #[test]
    fn factory_owned_configs_are_strict() {
        let error = registry()
            .validate_selection(
                &component(
                    "acme_remote",
                    serde_json::json!({"node": 4, "unknown": true}),
                ),
                &component("acme_workload", serde_json::json!({"message": "hello"})),
            )
            .unwrap_err()
            .to_string();
        assert!(error.contains("unknown field `unknown`"), "{error}");
    }

    #[test]
    fn workload_resources_fail_required_and_forbidden_presence_before_transport_prepare() {
        let missing = authored_run(serde_json::json!({}));
        let error = registry()
            .validate_selection_for_run(&missing)
            .unwrap_err()
            .to_string();
        assert!(error.contains("requires run.resources.models"), "{error}");

        let source_only = authored_run(serde_json::json!({
            "models": {"items": [{"name": "model"}]}
        }));
        let models_forbidden = ResourceRequirementsV2 {
            models: ResourceRequirementV2::Forbidden,
            endpoints: ResourceRequirementV2::Forbidden,
            metrics: ResourceRequirementV2::Forbidden,
            artifacts: ResourceRequirementV2::Optional,
            sidecars: ResourceRequirementV2::Forbidden,
        };
        let error = validate_resource_requirements(&source_only, models_forbidden)
            .unwrap_err()
            .to_string();
        assert!(error.contains("forbids run.resources.models"), "{error}");

        let inference = authored_run(serde_json::json!({
            "models": {"items": [{"name": "model"}]},
            "endpoints": {"profiles": [{}]}
        }));
        registry().validate_selection_for_run(&inference).unwrap();
    }

    #[test]
    fn unknown_ids_and_duplicate_registration_fail_closed() {
        let registry = registry();
        let error = registry
            .validate_selection(
                &component("missing", serde_json::json!({})),
                &component("acme_workload", serde_json::json!({"message": "hello"})),
            )
            .unwrap_err()
            .to_string();
        assert!(error.contains("missing") && error.contains("acme_remote"));

        let mut registry = AIPerfRegistry::builtin().unwrap();
        registry.register_transport(Arc::new(Transport)).unwrap();
        assert!(registry.register_transport(Arc::new(Transport)).is_err());
    }

    #[test]
    fn deterministic_capability_inventory_comes_from_frozen_factories() {
        let registry = registry();
        assert_eq!(registry.transport_descriptors(), vec![&TRANSPORT]);
        assert_eq!(registry.workload_descriptors(), vec![&WORKLOAD]);
    }

    #[test]
    fn run_context_retains_authored_endpoint_profile_order() {
        let profile = |profile_id: &str, endpoint_id: &str| ValidatedEndpointProfileV2 {
            profile_id: profile_id.to_owned(),
            endpoint_id: EndpointId::new(endpoint_id).unwrap(),
            config: RawEndpointConfig {
                urls: vec![format!("http://{profile_id}.example")],
                ..RawEndpointConfig::default()
            },
            connection_reuse: ConnectionReuseStrategy::default(),
            client: ClientConfig::default(),
            session_header: None,
        };
        let context = RunContext::new(
            format!("blake3:{}", "a".repeat(64)),
            Arc::new(AIPerfRegistry::builtin().unwrap()),
            crate::engine::execution_factories::native_execution_factories(),
            Arc::new(crate::engine::graph_input::BuiltinRunnerGraphInputAdapterResolver::new()),
            Arc::new(crate::engine::dataset_input::BuiltinRunnerDatasetInputAdapterResolver::new()),
            Arc::new(crate::engine::sidecar_input::PreparedSidecarInputs::default()),
            vec![
                profile("default", "chat"),
                profile("z-last", "messages"),
                profile("a-first", "chat"),
            ],
        )
        .unwrap();

        assert_eq!(
            context
                .endpoint_profiles()
                .map(|(profile_id, _)| profile_id)
                .collect::<Vec<_>>(),
            vec!["default", "z-last", "a-first"]
        );
        assert_eq!(
            context.endpoint_profile("a-first").unwrap().profile_id,
            "a-first"
        );
        let retained = context.endpoint_profiles_handle();
        assert!(std::ptr::eq(
            context.default_endpoint_profile().unwrap(),
            &retained[0]
        ));
        let run_metadata = context
            .report_run_metadata(format!("blake3:{}", "a".repeat(64)), "http", "graph")
            .unwrap();
        assert_eq!(
            run_metadata
                .endpoint_profiles
                .iter()
                .map(|profile| profile.profile_id.as_str())
                .collect::<Vec<_>>(),
            vec!["default", "z-last", "a-first"]
        );
    }

    #[test]
    fn endpoint_profile_retains_one_fully_validated_http_client_policy() {
        let run: AuthoredRunSpecV2 = serde_json::from_value(serde_json::json!({
            "identity": {"benchmark_id": "http-policy"},
            "artifact_target": "/tmp/http-policy",
            "transport": {"type": "http", "config": {}},
            "workload": {"type": "scheduled", "config": {}},
            "resources": {
                "models": {"items": [{"name": "model"}]},
                "endpoints": {"profiles": [{
                    "id": "default",
                    "type": "chat",
                    "urls": ["https://example.test"],
                    "timeout_seconds": 0.5,
                    "http2": true,
                    "ssl_verify": false,
                    "connection_limit": 7,
                    "keepalive_timeout": 0.25
                }]}
            }
        }))
        .unwrap();

        let profiles =
            validate_endpoint_profiles_v2(&run, &EndpointRegistry::builtin().unwrap()).unwrap();
        let client = &profiles[0].client;
        assert_eq!(client.http_version, HttpVersion::Http2PriorKnowledge);
        assert!(!client.ssl_verify);
        assert_eq!(client.max_connections_per_origin, 7);
        assert_eq!(client.keepalive_ns, Some(250_000_000));
        assert_eq!(client.total_timeout_ns, Some(500_000_000));
    }

    #[test]
    fn endpoint_profile_rejects_zero_connection_capacity() {
        let run: AuthoredRunSpecV2 = serde_json::from_value(serde_json::json!({
            "identity": {"benchmark_id": "http-policy-invalid"},
            "artifact_target": "/tmp/http-policy-invalid",
            "transport": {"type": "http", "config": {}},
            "workload": {"type": "scheduled", "config": {}},
            "resources": {
                "models": {"items": [{"name": "model"}]},
                "endpoints": {"profiles": [{
                    "id": "default",
                    "type": "chat",
                    "urls": ["http://example.test"],
                    "connection_limit": 0
                }]}
            }
        }))
        .unwrap();

        let error = validate_endpoint_profiles_v2(&run, &EndpointRegistry::builtin().unwrap())
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("connection_limit must be positive"),
            "{error}"
        );
    }

    #[test]
    fn endpoint_control_hook_paths_must_be_relative() {
        let run: AuthoredRunSpecV2 = serde_json::from_value(serde_json::json!({
            "identity": {"benchmark_id": "hook-path-invalid"},
            "artifact_target": "/tmp/hook-path-invalid",
            "transport": {"type": "http", "config": {}},
            "workload": {"type": "scheduled", "config": {}},
            "resources": {
                "models": {"items": [{"name": "model"}]},
                "endpoints": {"profiles": [{
                    "id": "default",
                    "type": "chat",
                    "urls": ["http://example.test"],
                    "reset_kv_cache": {"path": "http://bad.example/reset_prefix_cache"}
                }]}
            }
        }))
        .unwrap();

        let error = validate_endpoint_profiles_v2(&run, &EndpointRegistry::builtin().unwrap())
            .unwrap_err()
            .to_string();
        assert!(error.contains("relative"), "{error}");
    }

    #[test]
    fn control_plane_transport_timeout_is_positive_and_never_inert() {
        let factory = OnlineHttpTransportFactoryV2;
        let control_requirements = WorkloadRequirements {
            transport_features: BTreeSet::from(["control_plane_http".to_owned()]),
            resources: ResourceRequirementsV2::inference(),
        };
        let valid = RawValue::from_string(
            serde_json::json!({"client": {"connect_timeout_ns": 50_000_000_i64}}).to_string(),
        )
        .unwrap();
        assert!(factory.validate(&valid, &control_requirements).is_ok());

        let invalid = RawValue::from_string(
            serde_json::json!({"client": {"connect_timeout_ns": 0}}).to_string(),
        )
        .unwrap();
        assert!(factory.validate(&invalid, &control_requirements).is_err());

        let ordinary_requirements = WorkloadRequirements {
            resources: ResourceRequirementsV2::inference(),
            ..Default::default()
        };
        let inert = RawValue::from_string(
            serde_json::json!({"client": {"connect_timeout_ns": 50_000_000_i64}}).to_string(),
        )
        .unwrap();
        assert!(factory.validate(&inert, &ordinary_requirements).is_err());
    }

    #[test]
    fn builtin_inventory_does_not_claim_protocol_v1_or_library_only_execution() {
        let registry = crate::extensions::BuiltinAIPerfRegistryFactory
            .build()
            .unwrap();
        // Transports are exactly the components the composition-root extension
        // list registered; each feature gate flips its transport into or out of
        // the expected set (the registry returns them in sorted ID order).
        // `http` and the always-built `dry_run` fake leaf are unconditional; the
        // rest flip with their Cargo feature.
        let mut expected_transports = vec!["http", "dry_run"];
        #[cfg(feature = "grpc")]
        expected_transports.push("grpc");
        #[cfg(feature = "dynosim")]
        {
            expected_transports.push("dynosim_offline");
            expected_transports.push("dynosim_online");
        }
        expected_transports.sort_unstable();
        assert_eq!(
            registry
                .transport_descriptors()
                .into_iter()
                .map(|descriptor| descriptor.id)
                .collect::<Vec<_>>(),
            expected_transports
        );
        // Workloads and transports are two independent registries; there is no
        // pair/cross-product inventory. Any registered workload runs over any
        // registered transport (grpc + graph included), with transport-specific
        // execution resolved inside the workload's prepare.
        let expected_workloads = vec!["graph", "scheduled"];
        assert_eq!(
            registry
                .workload_descriptors()
                .into_iter()
                .map(|descriptor| descriptor.id)
                .collect::<Vec<_>>(),
            expected_workloads
        );
    }

    #[test]
    fn capabilities_catalog_auto_derives_from_the_registered_component_set() {
        use crate::engine::protocol::Catalog;

        let registry = crate::extensions::BuiltinAIPerfRegistryFactory
            .build()
            .unwrap();
        let catalog = Catalog::from_registry(&registry);

        // Transports are exactly the components the composition-root extension
        // list registered; gating an extension out drops its transport from the
        // catalog with no separate bookkeeping (grpc/dynosim flip the set).
        // `http` and the always-built `dry_run` fake leaf are unconditional; the
        // rest flip with their Cargo feature.
        let mut expected_transports = vec!["http", "dry_run"];
        #[cfg(feature = "grpc")]
        expected_transports.push("grpc");
        #[cfg(feature = "dynosim")]
        {
            expected_transports.push("dynosim_offline");
            expected_transports.push("dynosim_online");
        }
        expected_transports.sort_unstable();
        assert_eq!(
            catalog
                .transport
                .keys()
                .map(String::as_str)
                .collect::<Vec<_>>(),
            expected_transports
        );
        assert_eq!(
            catalog.transport["http"].metadata["url_schemes"],
            serde_json::json!(["http", "https"])
        );
        #[cfg(feature = "grpc")]
        assert_eq!(
            catalog.transport["grpc"].metadata["url_schemes"],
            serde_json::json!(["grpc", "grpcs"])
        );

        // The endpoint catalog is a pure projection of the frozen endpoint
        // registry: the catalog keys equal the registry's descriptor IDs one
        // for one, so a gated-out endpoint dialect vanishes automatically.
        let mut registry_endpoint_ids = registry
            .endpoints()
            .descriptors()
            .map(|descriptor| descriptor.id.to_owned())
            .collect::<Vec<_>>();
        registry_endpoint_ids.sort();
        assert_eq!(
            catalog.endpoint.keys().cloned().collect::<Vec<_>>(),
            registry_endpoint_ids
        );
        // Spot-check that the built-in dialect families are all present,
        // including the gRPC-served Riva endpoints and the KServe factories.
        for expected in [
            "chat",
            "messages",
            "vllm_generate",
            "riva_asr",
            "kserve_v2_infer",
        ] {
            assert!(
                catalog.endpoint.contains_key(expected),
                "missing endpoint {expected:?} in {:?}",
                catalog.endpoint.keys().collect::<Vec<_>>()
            );
        }

        // Workloads are not part of the plugins.yaml-shaped catalog document,
        // but they auto-derive from the same registry for internal inventory.
        assert_eq!(
            registry
                .workload_descriptors()
                .into_iter()
                .map(|descriptor| descriptor.id)
                .collect::<Vec<_>>(),
            vec!["graph", "scheduled"]
        );

        // The stock build layers only built-in extensions, so the native
        // The report's third-party extension metadata stays empty.
        assert_eq!(registry.extension_names().count(), 0);
    }

    #[test]
    fn builtin_factory_configs_fail_closed_before_execution() {
        let registry = crate::extensions::BuiltinAIPerfRegistryFactory
            .build()
            .unwrap();
        let transport = component("http", serde_json::json!({}));
        let workload = component(
            "scheduled",
            serde_json::json!({
                "worker_count": 1,
                "dataset": {"type": "synthetic"},
                "tokenizer": {"name": "builtin"},
                "phases": [{
                    "name": "profiling",
                    "type": "concurrency",
                    "exclude_from_results": false,
                    "concurrency": 1
                }],
                "unknown": true
            }),
        );

        let error = registry
            .validate_selection(&transport, &workload)
            .unwrap_err()
            .to_string();
        assert!(error.contains("unknown field `unknown`"), "{error}");
    }

    #[test]
    fn prepared_failure_accepts_only_safe_non_report_diagnostics() {
        let artifact = RunDiagnosticArtifactV2 {
            kind: "archive_failure_diagnostic".to_owned(),
            relative_path: "archive-failure-diagnostic.json".into(),
            content_hash: format!("blake3:{}", "a".repeat(64)),
        };
        let failure = PreparedRunFailure::reporting("archive_failed", "archive failed")
            .unwrap()
            .with_diagnostic_artifacts(vec![artifact.clone()])
            .unwrap();
        assert_eq!(failure.diagnostic_artifacts, vec![artifact]);

        for invalid in [
            RunDiagnosticArtifactV2 {
                kind: "ArchiveFailure".to_owned(),
                relative_path: "diagnostic.json".into(),
                content_hash: format!("blake3:{}", "a".repeat(64)),
            },
            RunDiagnosticArtifactV2 {
                kind: "archive_failure".to_owned(),
                relative_path: "../diagnostic.json".into(),
                content_hash: format!("blake3:{}", "a".repeat(64)),
            },
            RunDiagnosticArtifactV2 {
                kind: "archive_failure".to_owned(),
                relative_path: "native-v2.json".into(),
                content_hash: format!("blake3:{}", "a".repeat(64)),
            },
            RunDiagnosticArtifactV2 {
                kind: "archive_failure".to_owned(),
                relative_path: "diagnostic.json".into(),
                content_hash: "sha256:wrong".to_owned(),
            },
        ] {
            assert!(
                PreparedRunFailure::reporting("archive_failed", "archive failed")
                    .unwrap()
                    .with_diagnostic_artifacts(vec![invalid])
                    .is_err()
            );
        }
    }
}
