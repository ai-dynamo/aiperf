// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Frozen backend/workload/pair composition for protocol-v2 runs.
//!
//! Backend and workload factories validate their own authored objects. A third
//! pair factory performs open double dispatch: the coordinator never matches on
//! component strings, and a future remote placement can register a new backend
//! plus compatible pairs without changing scheduling, workload, or reporting
//! core code. Factory lookup and downcasting happen only during preparation;
//! hot request/token loops retain their typed implementations.

use std::any::Any;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Debug;
use std::path::PathBuf;
use std::sync::Arc;

use aiperf_endpoints::{EndpointId, EndpointRegistry, RawEndpointConfig, RequestContentType};
use aiperf_extensions::AiperfRegistry;
use aiperf_transport_http::models::ConnectionReuseStrategy;
use anyhow::{Result, anyhow, bail, ensure};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, value::RawValue};

use crate::protocol::PhaseSpec;
use crate::protocol_v2::{AuthoredRunSpecV2, NamedRunnerComponentSpecV2, RunnerComponentId};

/// Clock family supplied by one prepared backend.
#[derive(Clone, Copy, Debug, Serialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum RunnerClockKind {
    /// Monotonic wall-clock execution.
    Real,
    /// Deterministic discrete-event virtual time.
    Sim,
}

/// Deterministic capability facts for one linked backend factory.
#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct RunnerBackendDescriptor {
    /// Stable registry ID.
    pub id: &'static str,
    /// Human-readable implementation description.
    pub description: &'static str,
    /// Clock family exposed to compatible workloads.
    pub clock: RunnerClockKind,
    /// Whether responses contain model-semantic answer text.
    pub semantic_responses: bool,
    /// Statically compiled backend feature IDs.
    pub features: &'static [&'static str],
}

/// Deterministic capability facts for one linked workload factory.
#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct RunnerWorkloadDescriptor {
    /// Stable registry ID.
    pub id: &'static str,
    /// Human-readable implementation description.
    pub description: &'static str,
    /// Whether the workload requires semantic model responses.
    pub requires_semantic_responses: bool,
    /// Clock families this workload can consume.
    pub clock_kinds: &'static [RunnerClockKind],
    /// Backend features required for every run of this workload.
    pub required_backend_features: &'static [&'static str],
}

/// Config-specific requirements computed by a workload factory.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct WorkloadRequirements {
    /// Whether this authored config needs semantic response text.
    pub semantic_responses: bool,
    /// Accepted backend clock families.
    pub clock_kinds: BTreeSet<RunnerClockKind>,
    /// Required backend feature IDs.
    pub backend_features: BTreeSet<String>,
}

/// Type-erased, strictly validated backend configuration.
pub trait ValidatedBackendConfig: Debug + Send + Sync {
    /// Return the concrete startup-only value for a registered pair factory.
    fn as_any(&self) -> &dyn Any;
}

impl<T> ValidatedBackendConfig for T
where
    T: Any + Debug + Send + Sync,
{
    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Type-erased, strictly validated workload configuration.
pub trait ValidatedWorkloadConfig: Debug + Send + Sync {
    /// Return the concrete startup-only value for a registered pair factory.
    fn as_any(&self) -> &dyn Any;
}

impl<T> ValidatedWorkloadConfig for T
where
    T: Any + Debug + Send + Sync,
{
    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Startup-only backend validation and preparation half of the registry.
pub trait RunnerBackendFactory: Debug + Send + Sync {
    /// Describe the exact compiled implementation.
    fn descriptor(&self) -> &'static RunnerBackendDescriptor;

    /// Strictly decode and validate factory-owned authored config.
    fn validate(
        &self,
        authored: &RawValue,
        requirements: &WorkloadRequirements,
    ) -> Result<Box<dyn ValidatedBackendConfig>>;
}

/// Startup-only workload validation half of the registry.
pub trait RunnerWorkloadFactory: Debug + Send + Sync {
    /// Describe the exact compiled implementation.
    fn descriptor(&self) -> &'static RunnerWorkloadDescriptor;

    /// Strictly decode and validate factory-owned authored config.
    fn validate(&self, authored: &RawValue) -> Result<Box<dyn ValidatedWorkloadConfig>>;

    /// Derive backend requirements from the validated authored config.
    fn requirements(&self, config: &dyn ValidatedWorkloadConfig) -> Result<WorkloadRequirements>;
}

/// Result returned after one prepared pair executes successfully.
#[derive(Debug)]
pub struct PreparedRunOutcome {
    /// Authoritative native-v2 report path.
    pub report_path: PathBuf,
    /// Additive backend/workload provenance for the terminal response.
    pub provenance: BTreeMap<String, String>,
}

/// One completely prepared operation.
///
/// This trait intentionally has no `Send` bound: current high-performance
/// implementations own `Rc`/`RefCell` state on a current-thread Tokio runtime.
pub trait PreparedRunnerOperation: Debug {
    /// Execute the prepared run and commit its authoritative report.
    fn execute(self: Box<Self>) -> Result<PreparedRunOutcome>;
}

/// Open composition adapter for one backend/workload pair.
///
/// Pair factories are the deliberate double-dispatch seam Rust otherwise
/// lacks for two open trait-object families. They may downcast only the two
/// validated startup values registered for their exact key. The coordinator
/// remains unchanged when a new backend, workload, or remote placement is
/// statically linked.
pub trait RunnerPairFactory: Debug + Send + Sync {
    /// Backend registry ID consumed by this adapter.
    fn backend_id(&self) -> &'static str;

    /// Workload registry ID consumed by this adapter.
    fn workload_id(&self) -> &'static str;

    /// Check cross-component invariants without IO or artifact creation.
    fn validate_pair(
        &self,
        backend: &dyn ValidatedBackendConfig,
        workload: &dyn ValidatedWorkloadConfig,
    ) -> Result<()>;

    /// Check run-level invariants against the already validated product
    /// registry and endpoint profiles without performing external IO.
    fn validate_run(
        &self,
        _run: &AuthoredRunSpecV2,
        _context: &RunnerRunContext,
        backend: &dyn ValidatedBackendConfig,
        workload: &dyn ValidatedWorkloadConfig,
    ) -> Result<()> {
        self.validate_pair(backend, workload)
    }

    /// Complete pair-specific preparation after all static validation passes.
    fn prepare(
        &self,
        run: &AuthoredRunSpecV2,
        backend: Box<dyn ValidatedBackendConfig>,
        workload: Box<dyn ValidatedWorkloadConfig>,
    ) -> Result<Box<dyn PreparedRunnerOperation>>;

    /// Prepare with the coordinator-owned frozen product registry and
    /// normalized endpoint profiles.
    ///
    /// Existing context-free adapters retain the default. Online adapters
    /// override this hook so every pair consumes the one registry composed by
    /// the process coordinator instead of building a private registry.
    fn prepare_with_context(
        &self,
        run: &AuthoredRunSpecV2,
        _context: &RunnerRunContext,
        backend: Box<dyn ValidatedBackendConfig>,
        workload: Box<dyn ValidatedWorkloadConfig>,
    ) -> Result<Box<dyn PreparedRunnerOperation>> {
        self.prepare(run, backend, workload)
    }
}

/// Fully validated component selection retained between validation and prepare.
pub struct ValidatedRunnerSelection {
    backend_id: String,
    workload_id: String,
    backend: Box<dyn ValidatedBackendConfig>,
    workload: Box<dyn ValidatedWorkloadConfig>,
    pair: Arc<dyn RunnerPairFactory>,
}

impl Debug for ValidatedRunnerSelection {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ValidatedRunnerSelection")
            .field("backend_id", &self.backend_id)
            .field("workload_id", &self.workload_id)
            .finish_non_exhaustive()
    }
}

impl ValidatedRunnerSelection {
    /// Selected backend ID.
    pub fn backend_id(&self) -> &str {
        &self.backend_id
    }

    /// Selected workload ID.
    pub fn workload_id(&self) -> &str {
        &self.workload_id
    }
}

/// Mutable, transactionally checked registry construction surface.
#[derive(Default)]
pub struct RunnerRegistryBuilder {
    backends: BTreeMap<String, Arc<dyn RunnerBackendFactory>>,
    workloads: BTreeMap<String, Arc<dyn RunnerWorkloadFactory>>,
    pairs: BTreeMap<(String, String), Arc<dyn RunnerPairFactory>>,
}

impl RunnerRegistryBuilder {
    /// Construct an empty registry builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register one backend factory, rejecting duplicate IDs.
    pub fn register_backend(&mut self, factory: Arc<dyn RunnerBackendFactory>) -> Result<()> {
        let id = checked_descriptor_id(factory.descriptor().id, "backend")?;
        if self.backends.contains_key(id.as_str()) {
            bail!("duplicate runner backend ID {id:?}");
        }
        self.backends.insert(id.into_string(), factory);
        Ok(())
    }

    /// Register one workload factory, rejecting duplicate IDs.
    pub fn register_workload(&mut self, factory: Arc<dyn RunnerWorkloadFactory>) -> Result<()> {
        let id = checked_descriptor_id(factory.descriptor().id, "workload")?;
        if self.workloads.contains_key(id.as_str()) {
            bail!("duplicate runner workload ID {id:?}");
        }
        self.workloads.insert(id.into_string(), factory);
        Ok(())
    }

    /// Register one explicit pair adapter, rejecting duplicate keys.
    pub fn register_pair(&mut self, factory: Arc<dyn RunnerPairFactory>) -> Result<()> {
        let backend = checked_descriptor_id(factory.backend_id(), "pair backend")?;
        let workload = checked_descriptor_id(factory.workload_id(), "pair workload")?;
        let key = (backend.into_string(), workload.into_string());
        if self.pairs.contains_key(&key) {
            bail!(
                "duplicate runner backend/workload pair ({:?}, {:?})",
                key.0,
                key.1
            );
        }
        self.pairs.insert(key, factory);
        Ok(())
    }

    /// Freeze the complete registry after reference and descriptor validation.
    pub fn freeze(self) -> Result<RunnerRegistry> {
        for ((backend_id, workload_id), pair) in &self.pairs {
            let backend = self.backends.get(backend_id).ok_or_else(|| {
                anyhow!(
                    "runner pair ({backend_id:?}, {workload_id:?}) references an unregistered backend"
                )
            })?;
            let workload = self.workloads.get(workload_id).ok_or_else(|| {
                anyhow!(
                    "runner pair ({backend_id:?}, {workload_id:?}) references an unregistered workload"
                )
            })?;
            ensure!(
                pair.backend_id() == backend.descriptor().id
                    && pair.workload_id() == workload.descriptor().id,
                "runner pair key changed while the registry was being frozen"
            );
            validate_descriptor_compatibility(backend.descriptor(), workload.descriptor())?;
        }
        let mut statically_compatible_pairs = BTreeSet::new();
        for (backend_id, backend) in &self.backends {
            for (workload_id, workload) in &self.workloads {
                if descriptors_are_compatible(backend.descriptor(), workload.descriptor()) {
                    statically_compatible_pairs.insert((backend_id.clone(), workload_id.clone()));
                }
            }
        }
        Ok(RunnerRegistry {
            backends: self.backends,
            workloads: self.workloads,
            pairs: self.pairs,
            statically_compatible_pairs,
        })
    }
}

/// Immutable runner composition used by capabilities, validation, and execute.
pub struct RunnerRegistry {
    backends: BTreeMap<String, Arc<dyn RunnerBackendFactory>>,
    workloads: BTreeMap<String, Arc<dyn RunnerWorkloadFactory>>,
    pairs: BTreeMap<(String, String), Arc<dyn RunnerPairFactory>>,
    statically_compatible_pairs: BTreeSet<(String, String)>,
}

impl RunnerRegistry {
    /// Return backend descriptors in deterministic ID order.
    pub fn backend_descriptors(&self) -> Vec<&'static RunnerBackendDescriptor> {
        self.backends
            .values()
            .map(|factory| factory.descriptor())
            .collect()
    }

    /// Return workload descriptors in deterministic ID order.
    pub fn workload_descriptors(&self) -> Vec<&'static RunnerWorkloadDescriptor> {
        self.workloads
            .values()
            .map(|factory| factory.descriptor())
            .collect()
    }

    /// Return explicitly linked compatible pairs in deterministic tuple order.
    pub fn supported_pairs(&self) -> Vec<(&str, &str)> {
        self.pairs
            .keys()
            .map(|(backend, workload)| (backend.as_str(), workload.as_str()))
            .collect()
    }

    /// Return descriptor-compatible pairs whether or not protocol-v2 execution
    /// has a registered pair adapter.
    ///
    /// Capability consumers must use [`Self::supported_pairs`] to decide what
    /// this exact binary can execute. This separate inventory makes recognized
    /// design compatibility visible without claiming product reachability for
    /// library-only or protocol-v1-only implementations.
    pub fn statically_compatible_pairs(&self) -> Vec<(&str, &str)> {
        self.statically_compatible_pairs
            .iter()
            .map(|(backend, workload)| (backend.as_str(), workload.as_str()))
            .collect()
    }

    /// Resolve and statically validate one authored backend/workload selection.
    pub fn validate_selection(
        &self,
        backend: &NamedRunnerComponentSpecV2,
        workload: &NamedRunnerComponentSpecV2,
    ) -> Result<ValidatedRunnerSelection> {
        let workload_factory = self
            .workloads
            .get(workload.id.as_str())
            .ok_or_else(|| unknown_component("workload", &workload.id, self.workloads.keys()))?;
        let validated_workload = workload_factory
            .validate(&workload.config)
            .map_err(|error| anyhow!("workload {:?}: {error:#}", workload.id.as_str()))?;
        let requirements = workload_factory.requirements(validated_workload.as_ref())?;

        let backend_factory = self
            .backends
            .get(backend.id.as_str())
            .ok_or_else(|| unknown_component("backend", &backend.id, self.backends.keys()))?;
        validate_requirements(backend_factory.descriptor(), &requirements)?;
        let validated_backend = backend_factory
            .validate(&backend.config, &requirements)
            .map_err(|error| anyhow!("backend {:?}: {error:#}", backend.id.as_str()))?;

        let pair_key = (
            backend.id.as_str().to_owned(),
            workload.id.as_str().to_owned(),
        );
        let pair = self.pairs.get(&pair_key).cloned().ok_or_else(|| {
            anyhow!(
                "runner distribution does not contain backend/workload pair ({:?}, {:?})",
                pair_key.0,
                pair_key.1
            )
        })?;
        pair.validate_pair(validated_backend.as_ref(), validated_workload.as_ref())?;
        Ok(ValidatedRunnerSelection {
            backend_id: pair_key.0,
            workload_id: pair_key.1,
            backend: validated_backend,
            workload: validated_workload,
            pair,
        })
    }

    /// Apply pair-owned run-level validation after component configs and
    /// endpoint profiles have been strictly decoded.
    pub fn validate_run(
        &self,
        run: &AuthoredRunSpecV2,
        context: &RunnerRunContext,
        selection: &ValidatedRunnerSelection,
    ) -> Result<()> {
        selection.pair.validate_run(
            run,
            context,
            selection.backend.as_ref(),
            selection.workload.as_ref(),
        )
    }

    /// Prepare one previously validated selection without repeating lookup or
    /// central backend/workload branching.
    pub fn prepare(
        &self,
        run: &AuthoredRunSpecV2,
        selection: ValidatedRunnerSelection,
    ) -> Result<Box<dyn PreparedRunnerOperation>> {
        selection
            .pair
            .prepare(run, selection.backend, selection.workload)
    }

    /// Prepare through the coordinator-owned product registry and normalized
    /// endpoint-profile context.
    pub fn prepare_with_context(
        &self,
        run: &AuthoredRunSpecV2,
        context: &RunnerRunContext,
        selection: ValidatedRunnerSelection,
    ) -> Result<Box<dyn PreparedRunnerOperation>> {
        selection
            .pair
            .prepare_with_context(run, context, selection.backend, selection.workload)
    }
}

fn checked_descriptor_id(value: &str, kind: &str) -> Result<RunnerComponentId> {
    value
        .parse()
        .map_err(|error: String| anyhow!("invalid runner {kind} descriptor ID: {error}"))
}

fn validate_descriptor_compatibility(
    backend: &RunnerBackendDescriptor,
    workload: &RunnerWorkloadDescriptor,
) -> Result<()> {
    ensure!(
        workload.clock_kinds.contains(&backend.clock),
        "runner pair ({:?}, {:?}) has incompatible clock descriptors",
        backend.id,
        workload.id
    );
    ensure!(
        !workload.requires_semantic_responses || backend.semantic_responses,
        "runner pair ({:?}, {:?}) requires semantic responses from a non-semantic backend",
        backend.id,
        workload.id
    );
    for feature in workload.required_backend_features {
        ensure!(
            backend.features.contains(feature),
            "runner pair ({:?}, {:?}) requires missing backend feature {:?}",
            backend.id,
            workload.id,
            feature
        );
    }
    Ok(())
}

fn descriptors_are_compatible(
    backend: &RunnerBackendDescriptor,
    workload: &RunnerWorkloadDescriptor,
) -> bool {
    workload.clock_kinds.contains(&backend.clock)
        && (!workload.requires_semantic_responses || backend.semantic_responses)
        && workload
            .required_backend_features
            .iter()
            .all(|feature| backend.features.contains(feature))
}

fn validate_requirements(
    backend: &RunnerBackendDescriptor,
    requirements: &WorkloadRequirements,
) -> Result<()> {
    ensure!(
        requirements.clock_kinds.contains(&backend.clock),
        "backend {:?} uses {:?} clock, which the authored workload does not support",
        backend.id,
        backend.clock
    );
    ensure!(
        !requirements.semantic_responses || backend.semantic_responses,
        "backend {:?} cannot provide the semantic responses required by the workload",
        backend.id
    );
    for feature in &requirements.backend_features {
        ensure!(
            backend.features.contains(&feature.as_str()),
            "backend {:?} is missing required compiled feature {:?}",
            backend.id,
            feature
        );
    }
    Ok(())
}

fn unknown_component<'a>(
    kind: &str,
    requested: &RunnerComponentId,
    available: impl Iterator<Item = &'a String>,
) -> anyhow::Error {
    let choices = available.map(String::as_str).collect::<Vec<_>>().join(", ");
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

/// Composition seam for the exact backend/workload universe linked into one
/// runner executable.
pub trait RunnerRegistryFactory {
    /// Build and freeze a fresh deterministic registry.
    fn build(&self) -> Result<RunnerRegistry>;
}

/// Stock protocol-v2 registry composition.
///
/// Optional feature modules extend the same builder before it freezes. Merely
/// registering backend/workload descriptors does not advertise execution:
/// [`RunnerRegistry::supported_pairs`] contains only explicit pair factories.
#[derive(Clone, Copy, Debug, Default)]
pub struct BuiltinRunnerRegistryFactory;

impl RunnerRegistryFactory for BuiltinRunnerRegistryFactory {
    fn build(&self) -> Result<RunnerRegistry> {
        let mut builder = RunnerRegistryBuilder::new();
        builder.register_backend(Arc::new(OnlineHttpBackendFactoryV2))?;
        builder.register_workload(Arc::new(ScheduledWorkloadFactoryV2))?;
        builder.register_workload(Arc::new(GraphWorkloadFactoryV2))?;
        builder.register_workload(Arc::new(StaticAccuracyWorkloadFactoryV2))?;
        register_optional_builtin_components(&mut builder)?;
        builder.freeze()
    }
}

fn register_optional_builtin_components(builder: &mut RunnerRegistryBuilder) -> Result<()> {
    // Feature-bearing modules add registrations here without changing lookup,
    // validation, or dispatch. Keeping the composition function even in the
    // base build prevents capability code from growing mode string branches.
    crate::agentic_execution::register_agentic_workload(builder)?;
    crate::agentic_execution::register_agentic_online_pair(builder)?;
    crate::online_execution::register_online_http_pairs(builder)?;
    #[cfg(feature = "dynamo-offline")]
    crate::offline_execution::register_dynamo_offline_backend(builder)?;
    Ok(())
}

/// Strict validated config owned by the built-in `online_http` backend.
///
/// Endpoint profiles and transport policy are run-level inputs, so the backend
/// currently owns an intentionally empty object. New backend-owned fields must
/// be added here and therefore fail closed on older runners.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OnlineHttpBackendConfigV2 {}

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
struct EndpointProfileConfigV2 {
    id: String,
    #[serde(rename = "type")]
    endpoint_id: EndpointId,
    urls: Vec<String>,
    #[serde(default)]
    path: Option<String>,
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
    #[serde(default)]
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
    /// Whether cleartext HTTP/2 prior knowledge was requested.
    pub http2: bool,
    /// Optional authored session-affinity header name.
    pub session_header: Option<String>,
}

/// Coordinator-owned immutable context shared by static pair validation and
/// complete execution preparation.
///
/// One fresh runner process builds exactly one [`AiperfRegistry`]. Pair
/// adapters clone this cheap handle into their prepared operation; they never
/// invoke a built-in registry factory or independently decode endpoint policy.
#[derive(Clone)]
pub struct RunnerRunContext {
    product_registry: Arc<AiperfRegistry>,
    endpoint_profiles: Arc<Vec<ValidatedEndpointProfileV2>>,
    endpoint_profile_indexes: Arc<BTreeMap<String, usize>>,
}

impl Debug for RunnerRunContext {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("RunnerRunContext")
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

impl RunnerRunContext {
    /// Freeze one validated profile collection beside the process registry.
    pub fn new(
        product_registry: Arc<AiperfRegistry>,
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
        ensure!(
            !profiles.is_empty(),
            "runner context requires at least one endpoint profile"
        );
        Ok(Self {
            product_registry,
            endpoint_profiles: Arc::new(profiles),
            endpoint_profile_indexes: Arc::new(endpoint_profile_indexes),
        })
    }

    /// Borrow the single frozen product registry composed by the coordinator.
    pub fn product_registry(&self) -> &AiperfRegistry {
        self.product_registry.as_ref()
    }

    /// Clone the cheap shared registry handle into a prepared operation.
    pub fn product_registry_handle(&self) -> Arc<AiperfRegistry> {
        self.product_registry.clone()
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

    /// Resolve the conventional profile injected by the Python projection.
    pub fn default_endpoint_profile(&self) -> Result<&ValidatedEndpointProfileV2> {
        self.endpoint_profile("default")
    }

    /// Iterate normalized profiles in authored order.
    ///
    /// Keeping this sequence intact lets prepared endpoint tables and report
    /// provenance consume the same coordinator-owned identity ordering.
    pub fn endpoint_profiles(
        &self,
    ) -> impl ExactSizeIterator<Item = (&str, &ValidatedEndpointProfileV2)> {
        self.endpoint_profiles
            .iter()
            .map(|profile| (profile.profile_id.as_str(), profile))
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
        validated.push(ValidatedEndpointProfileV2 {
            profile_id: config.id,
            endpoint_id: canonical_id,
            config: prepared.config().to_raw(),
            connection_reuse: config.connection_reuse,
            http2: config.http2,
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

/// Built-in online HTTP backend descriptor.
pub static ONLINE_HTTP_BACKEND_DESCRIPTOR: RunnerBackendDescriptor = RunnerBackendDescriptor {
    id: "online_http",
    description: "Clock-injected native HTTP transport over a real clock",
    clock: RunnerClockKind::Real,
    semantic_responses: true,
    features: &["h1", "h2c", "http", "tls", "uds"],
};

/// Built-in scheduled workload descriptor.
pub static SCHEDULED_WORKLOAD_DESCRIPTOR: RunnerWorkloadDescriptor = RunnerWorkloadDescriptor {
    id: "scheduled",
    description: "Clock-paced scheduled request and session workloads",
    requires_semantic_responses: false,
    clock_kinds: &[RunnerClockKind::Real, RunnerClockKind::Sim],
    required_backend_features: &[],
};

/// Built-in direct Graph-IR workload descriptor.
pub static GRAPH_WORKLOAD_DESCRIPTOR: RunnerWorkloadDescriptor = RunnerWorkloadDescriptor {
    id: "graph",
    description: "Direct authored DAG lowering and whole-trace Graph-IR execution",
    requires_semantic_responses: false,
    clock_kinds: &[RunnerClockKind::Real, RunnerClockKind::Sim],
    required_backend_features: &[],
};

/// Built-in static Python-evaluated accuracy workload descriptor.
pub static STATIC_ACCURACY_WORKLOAD_DESCRIPTOR: RunnerWorkloadDescriptor =
    RunnerWorkloadDescriptor {
        id: "static_accuracy",
        description: "Canonical supervised Python static evaluator with Rust inference",
        requires_semantic_responses: true,
        clock_kinds: &[RunnerClockKind::Real],
        required_backend_features: &["http"],
    };

#[derive(Debug)]
struct OnlineHttpBackendFactoryV2;

impl RunnerBackendFactory for OnlineHttpBackendFactoryV2 {
    fn descriptor(&self) -> &'static RunnerBackendDescriptor {
        &ONLINE_HTTP_BACKEND_DESCRIPTOR
    }

    fn validate(
        &self,
        authored: &RawValue,
        _requirements: &WorkloadRequirements,
    ) -> Result<Box<dyn ValidatedBackendConfig>> {
        Ok(Box::new(strict_decode::<OnlineHttpBackendConfigV2>(
            authored,
            "online_http backend config",
        )?))
    }
}

#[derive(Debug)]
struct ScheduledWorkloadFactoryV2;

impl RunnerWorkloadFactory for ScheduledWorkloadFactoryV2 {
    fn descriptor(&self) -> &'static RunnerWorkloadDescriptor {
        &SCHEDULED_WORKLOAD_DESCRIPTOR
    }

    fn validate(&self, authored: &RawValue) -> Result<Box<dyn ValidatedWorkloadConfig>> {
        let config =
            strict_decode::<ScheduledWorkloadConfigV2>(authored, "scheduled workload config")?;
        validate_common_workload(
            config.worker_count,
            &config.dataset,
            &config.tokenizer,
            &config.phases,
        )?;
        Ok(Box::new(config))
    }

    fn requirements(&self, _config: &dyn ValidatedWorkloadConfig) -> Result<WorkloadRequirements> {
        Ok(requirements_from_descriptor(self.descriptor()))
    }
}

#[derive(Debug)]
struct GraphWorkloadFactoryV2;

impl RunnerWorkloadFactory for GraphWorkloadFactoryV2 {
    fn descriptor(&self) -> &'static RunnerWorkloadDescriptor {
        &GRAPH_WORKLOAD_DESCRIPTOR
    }

    fn validate(&self, authored: &RawValue) -> Result<Box<dyn ValidatedWorkloadConfig>> {
        let config = strict_decode::<GraphWorkloadConfigV2>(authored, "graph workload config")?;
        validate_common_workload(
            config.worker_count,
            &config.dataset,
            &config.tokenizer,
            &config.phases,
        )?;
        let dataset = raw_object(&config.dataset, "graph workload dataset")?;
        ensure!(
            dataset.get("format").and_then(Value::as_str) == Some("dag_jsonl"),
            "graph workload dataset.format must be \"dag_jsonl\""
        );
        Ok(Box::new(config))
    }

    fn requirements(&self, _config: &dyn ValidatedWorkloadConfig) -> Result<WorkloadRequirements> {
        Ok(requirements_from_descriptor(self.descriptor()))
    }
}

#[derive(Debug)]
struct StaticAccuracyWorkloadFactoryV2;

impl RunnerWorkloadFactory for StaticAccuracyWorkloadFactoryV2 {
    fn descriptor(&self) -> &'static RunnerWorkloadDescriptor {
        &STATIC_ACCURACY_WORKLOAD_DESCRIPTOR
    }

    fn validate(&self, authored: &RawValue) -> Result<Box<dyn ValidatedWorkloadConfig>> {
        let config = strict_decode::<StaticAccuracyWorkloadConfigV2>(
            authored,
            "static_accuracy workload config",
        )?;
        validate_common_workload(
            config.worker_count,
            &config.dataset,
            &config.tokenizer,
            &config.phases,
        )?;
        let accuracy = raw_object(&config.accuracy, "static_accuracy accuracy config")?;
        ensure!(
            accuracy
                .get("benchmark")
                .and_then(Value::as_str)
                .is_some_and(|value| !value.trim().is_empty()),
            "static_accuracy accuracy.benchmark must be a non-empty string"
        );
        Ok(Box::new(config))
    }

    fn requirements(&self, _config: &dyn ValidatedWorkloadConfig) -> Result<WorkloadRequirements> {
        Ok(requirements_from_descriptor(self.descriptor()))
    }
}

fn strict_decode<T>(authored: &RawValue, label: &str) -> Result<T>
where
    T: DeserializeOwned,
{
    serde_json::from_str(authored.get()).map_err(|error| anyhow!("{label}: {error}"))
}

fn validate_common_workload(
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

fn requirements_from_descriptor(descriptor: &RunnerWorkloadDescriptor) -> WorkloadRequirements {
    WorkloadRequirements {
        semantic_responses: descriptor.requires_semantic_responses,
        clock_kinds: descriptor.clock_kinds.iter().copied().collect(),
        backend_features: descriptor
            .required_backend_features
            .iter()
            .map(|feature| (*feature).to_owned())
            .collect(),
    }
}

#[cfg(test)]
mod tests {
    use serde::Deserialize;
    use serde_json::Value;

    use super::*;

    static BACKEND: RunnerBackendDescriptor = RunnerBackendDescriptor {
        id: "acme_remote",
        description: "fixture remote placement",
        clock: RunnerClockKind::Real,
        semantic_responses: true,
        features: &["http"],
    };
    static WORKLOAD: RunnerWorkloadDescriptor = RunnerWorkloadDescriptor {
        id: "acme_workload",
        description: "fixture workload",
        requires_semantic_responses: true,
        clock_kinds: &[RunnerClockKind::Real],
        required_backend_features: &["http"],
    };

    #[derive(Debug, Deserialize)]
    #[serde(deny_unknown_fields)]
    struct BackendConfig {
        node: usize,
    }

    #[derive(Debug, Deserialize)]
    #[serde(deny_unknown_fields)]
    struct WorkloadConfig {
        message: String,
    }

    #[derive(Debug)]
    struct Backend;

    impl RunnerBackendFactory for Backend {
        fn descriptor(&self) -> &'static RunnerBackendDescriptor {
            &BACKEND
        }

        fn validate(
            &self,
            authored: &RawValue,
            _requirements: &WorkloadRequirements,
        ) -> Result<Box<dyn ValidatedBackendConfig>> {
            Ok(Box::new(serde_json::from_str::<BackendConfig>(
                authored.get(),
            )?))
        }
    }

    #[derive(Debug)]
    struct Workload;

    impl RunnerWorkloadFactory for Workload {
        fn descriptor(&self) -> &'static RunnerWorkloadDescriptor {
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
                semantic_responses: true,
                clock_kinds: BTreeSet::from([RunnerClockKind::Real]),
                backend_features: BTreeSet::from(["http".to_owned()]),
            })
        }
    }

    #[derive(Debug)]
    struct Pair;

    impl RunnerPairFactory for Pair {
        fn backend_id(&self) -> &'static str {
            BACKEND.id
        }

        fn workload_id(&self) -> &'static str {
            WORKLOAD.id
        }

        fn validate_pair(
            &self,
            backend: &dyn ValidatedBackendConfig,
            workload: &dyn ValidatedWorkloadConfig,
        ) -> Result<()> {
            let backend = backend
                .as_any()
                .downcast_ref::<BackendConfig>()
                .ok_or_else(|| anyhow!("wrong backend config type"))?;
            let workload = workload
                .as_any()
                .downcast_ref::<WorkloadConfig>()
                .ok_or_else(|| anyhow!("wrong workload config type"))?;
            ensure!(backend.node > 0, "node must be positive");
            ensure!(!workload.message.is_empty(), "message cannot be empty");
            Ok(())
        }

        fn prepare(
            &self,
            _run: &AuthoredRunSpecV2,
            backend: Box<dyn ValidatedBackendConfig>,
            workload: Box<dyn ValidatedWorkloadConfig>,
        ) -> Result<Box<dyn PreparedRunnerOperation>> {
            let node = ValidatedBackendConfig::as_any(backend.as_ref())
                .downcast_ref::<BackendConfig>()
                .ok_or_else(|| anyhow!("wrong backend config type"))?
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
                report_path: PathBuf::from(format!("{}-{}.json", self.node, self.message)),
                provenance: BTreeMap::new(),
            })
        }
    }

    fn registry() -> RunnerRegistry {
        let mut builder = RunnerRegistryBuilder::new();
        builder.register_backend(Arc::new(Backend)).unwrap();
        builder.register_workload(Arc::new(Workload)).unwrap();
        builder.register_pair(Arc::new(Pair)).unwrap();
        builder.freeze().unwrap()
    }

    fn component(id: &str, config: Value) -> NamedRunnerComponentSpecV2 {
        NamedRunnerComponentSpecV2 {
            id: id.parse().unwrap(),
            config: RawValue::from_string(config.to_string()).unwrap(),
        }
    }

    #[test]
    fn custom_pair_validates_without_a_core_enum_or_match() {
        let registry = registry();
        let selection = registry
            .validate_selection(
                &component("acme_remote", serde_json::json!({"node": 4})),
                &component("acme_workload", serde_json::json!({"message": "hello"})),
            )
            .unwrap();
        assert_eq!(selection.backend_id(), "acme_remote");
        assert_eq!(selection.workload_id(), "acme_workload");
        assert_eq!(
            registry.supported_pairs(),
            vec![("acme_remote", "acme_workload")]
        );
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

        let mut builder = RunnerRegistryBuilder::new();
        builder.register_backend(Arc::new(Backend)).unwrap();
        assert!(builder.register_backend(Arc::new(Backend)).is_err());
    }

    #[test]
    fn deterministic_capability_inventory_comes_from_frozen_factories() {
        let registry = registry();
        assert_eq!(registry.backend_descriptors(), vec![&BACKEND]);
        assert_eq!(registry.workload_descriptors(), vec![&WORKLOAD]);
        assert_eq!(registry.supported_pairs(), vec![(BACKEND.id, WORKLOAD.id)]);
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
            http2: false,
            session_header: None,
        };
        let context = RunnerRunContext::new(
            Arc::new(AiperfRegistry::builtin().unwrap()),
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
    }

    #[test]
    fn builtin_inventory_does_not_claim_protocol_v1_or_library_only_execution() {
        let registry = BuiltinRunnerRegistryFactory.build().unwrap();
        #[cfg(feature = "dynamo-offline")]
        let expected_backends = vec!["dynamo_offline", "online_http"];
        #[cfg(not(feature = "dynamo-offline"))]
        let expected_backends = vec!["online_http"];
        assert_eq!(
            registry
                .backend_descriptors()
                .into_iter()
                .map(|descriptor| descriptor.id)
                .collect::<Vec<_>>(),
            expected_backends
        );
        assert_eq!(
            registry
                .workload_descriptors()
                .into_iter()
                .map(|descriptor| descriptor.id)
                .collect::<Vec<_>>(),
            vec!["agentic", "graph", "scheduled", "static_accuracy"]
        );
        #[cfg(feature = "dynamo-offline")]
        let expected_supported = vec![
            ("dynamo_offline", "graph"),
            ("dynamo_offline", "scheduled"),
            ("online_http", "agentic"),
            ("online_http", "graph"),
        ];
        #[cfg(not(feature = "dynamo-offline"))]
        let expected_supported = vec![("online_http", "agentic"), ("online_http", "graph")];
        #[cfg(feature = "dynamo-offline")]
        let expected_static = vec![
            ("dynamo_offline", "graph"),
            ("dynamo_offline", "scheduled"),
            ("online_http", "agentic"),
            ("online_http", "graph"),
            ("online_http", "scheduled"),
            ("online_http", "static_accuracy"),
        ];
        #[cfg(not(feature = "dynamo-offline"))]
        let expected_static = vec![
            ("online_http", "agentic"),
            ("online_http", "graph"),
            ("online_http", "scheduled"),
            ("online_http", "static_accuracy"),
        ];
        assert_eq!(registry.supported_pairs(), expected_supported);
        assert_eq!(registry.statically_compatible_pairs(), expected_static);
    }

    #[test]
    fn builtin_factory_configs_fail_closed_before_pair_lookup() {
        let registry = BuiltinRunnerRegistryFactory.build().unwrap();
        let backend = component("online_http", serde_json::json!({}));
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
            .validate_selection(&backend, &workload)
            .unwrap_err()
            .to_string();
        assert!(error.contains("unknown field `unknown`"), "{error}");
    }
}
