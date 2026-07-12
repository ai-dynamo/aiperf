// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Replaceable evaluator-provider seam and deterministic open registry.
//!
//! Factories register process-protocol implementations, never benchmarks or
//! graders. A factory owns the immutable worker launch identity and the pure
//! authored-config schema; run configuration cannot select an executable,
//! module, argument vector, environment, or upstream endpoint.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::{self, Display};
use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::canonical::{
    CANONICAL_JSON_CODEC, CanonicalJson, is_sha256, sha256_hex, validate_no_secret_control_value,
};
use crate::provider_protocol::{
    EVALUATOR_WORKER_PROTOCOL_V2, EvaluationDistributionId, EvaluationEventBatch,
    EvaluationExecutionGranularity, EvaluationFinishCandidate, EvaluationIdentity, EvaluationPlan,
    EvaluationPlanRequest, EvaluationProtocolError, EvaluationProviderId, EvaluationSchedulingMode,
    EvaluationSessionId, EvaluationUnitId, EvaluationUnitOccurrence,
    EvaluationUnitOccurrenceRequest, EvaluationUnitPage, EvaluationWorkerIdentity,
    HostOperationEvent, LogicalServiceId, ResolvedEvaluationAsset, ScopedProxyBinding,
    SemanticOperationId,
};

/// Process-protocol operation schema and endpoint compatibility descriptor.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvaluationOperationDescriptor {
    /// Open semantic operation identity.
    pub operation_id: SemanticOperationId,
    /// Canonical input schema digest.
    pub input_schema_sha256: String,
    /// Canonical terminal output schema digest.
    pub output_schema_sha256: String,
    /// Canonical stream-event schema digest when true streaming is supported.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stream_schema_sha256: Option<String>,
    /// Whether authoritative usage is expected.
    pub reports_usage: bool,
    /// Supported content/modality labels.
    pub modalities: Vec<String>,
    /// Endpoint capability labels required by an executable adapter.
    pub endpoint_capabilities: Vec<String>,
}

impl EvaluationOperationDescriptor {
    fn validate(&self) -> Result<(), ProviderRegistryError> {
        if !is_sha256(&self.input_schema_sha256)
            || !is_sha256(&self.output_schema_sha256)
            || self
                .stream_schema_sha256
                .as_deref()
                .is_some_and(|digest| !is_sha256(digest))
            || self.modalities.is_empty()
            || self.endpoint_capabilities.is_empty()
        {
            return Err(ProviderRegistryError::InvalidDescriptor(format!(
                "operation {} had incomplete or mutable schemas/capabilities",
                self.operation_id
            )));
        }
        Ok(())
    }
}

/// Observable worker and descendant isolation requirements.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvaluatorIsolationRequirements {
    /// Child environment is cleared then populated from a factory allowlist.
    pub clean_environment: bool,
    /// Worker sees a private/contained filesystem view.
    pub contained_filesystem: bool,
    /// Worker runs under an unprivileged identity with `no_new_privs`.
    pub unprivileged_no_new_privs: bool,
    /// `/proc` is absent or restricted.
    pub restricted_proc: bool,
    /// CPU, memory, process, file, and descriptor resources are bounded.
    pub bounded_resources: bool,
    /// Unexpected inherited descriptors and host-service sockets are denied.
    pub descriptor_confinement: bool,
    /// Direct DNS/Internet/host/peer networking is denied.
    pub deny_external_network: bool,
    /// A scoped Rust proxy may be granted to a declared process subtree.
    pub allows_scoped_proxy: bool,
}

impl EvaluatorIsolationRequirements {
    /// Full mandatory AIPerf evaluator isolation profile.
    pub fn strict_process_tree() -> Self {
        Self {
            clean_environment: true,
            contained_filesystem: true,
            unprivileged_no_new_privs: true,
            restricted_proc: true,
            bounded_resources: true,
            descriptor_confinement: true,
            deny_external_network: true,
            allows_scoped_proxy: true,
        }
    }

    pub(crate) fn validate(&self) -> Result<(), ProviderRegistryError> {
        if !self.clean_environment
            || !self.contained_filesystem
            || !self.unprivileged_no_new_privs
            || !self.restricted_proc
            || !self.bounded_resources
            || !self.descriptor_confinement
            || !self.deny_external_network
        {
            return Err(ProviderRegistryError::InvalidDescriptor(
                "provider isolation requirements weakened the mandatory process-tree profile"
                    .to_string(),
            ));
        }
        Ok(())
    }
}

/// Factory-owned expected worker identity for one immutable distribution.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvaluationDistributionDescriptor {
    /// Immutable selectable distribution identity.
    pub distribution_id: EvaluationDistributionId,
    /// Exact package name.
    pub package: String,
    /// Exact package version.
    pub package_version: String,
    /// Provider source/commit digest.
    pub provider_source_sha256: String,
    /// Worker bootstrap source digest.
    pub worker_source_sha256: String,
    /// Fully pinned dependency lock digest.
    pub dependency_lock_sha256: String,
    /// Optional immutable OCI image digest.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub oci_digest: Option<String>,
    /// Factory-attested executable/closure digest.
    pub launch_closure_sha256: String,
}

impl EvaluationDistributionDescriptor {
    fn validate(&self) -> Result<(), ProviderRegistryError> {
        if self.package.trim().is_empty() || self.package_version.trim().is_empty() {
            return Err(ProviderRegistryError::InvalidDescriptor(format!(
                "distribution {} had empty package identity",
                self.distribution_id
            )));
        }
        for digest in [
            self.provider_source_sha256.as_str(),
            self.worker_source_sha256.as_str(),
            self.dependency_lock_sha256.as_str(),
            self.launch_closure_sha256.as_str(),
        ] {
            if !is_sha256(digest) {
                return Err(ProviderRegistryError::InvalidDescriptor(format!(
                    "distribution {} contained an invalid digest",
                    self.distribution_id
                )));
            }
        }
        if self
            .oci_digest
            .as_deref()
            .is_some_and(|digest| !digest.strip_prefix("sha256:").is_some_and(is_sha256))
        {
            return Err(ProviderRegistryError::InvalidDescriptor(format!(
                "distribution {} contained a mutable OCI identity",
                self.distribution_id
            )));
        }
        Ok(())
    }

    /// Build the exact worker identity expected after `hello`.
    pub fn expected_worker_identity(
        &self,
        provider_id: EvaluationProviderId,
        python_version: String,
        launch_nonce: String,
        operations: Vec<String>,
    ) -> EvaluationWorkerIdentity {
        EvaluationWorkerIdentity {
            evaluator_protocol: EVALUATOR_WORKER_PROTOCOL_V2,
            provider_id,
            distribution_id: self.distribution_id.clone(),
            package: self.package.clone(),
            package_version: self.package_version.clone(),
            provider_source_sha256: self.provider_source_sha256.clone(),
            worker_source_sha256: self.worker_source_sha256.clone(),
            dependency_lock_sha256: self.dependency_lock_sha256.clone(),
            python_version,
            launch_nonce,
            oci_digest: self.oci_digest.clone(),
            operations,
        }
    }
}

/// Complete registered evaluator-provider descriptor.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvaluationProviderDescriptor {
    /// Open provider identity.
    pub provider_id: EvaluationProviderId,
    /// Safe human-facing label.
    pub display_name: String,
    /// Supported evaluator-worker protocol versions.
    pub worker_protocol_versions: Vec<u32>,
    /// Supported execution granularities.
    pub execution_granularities: Vec<EvaluationExecutionGranularity>,
    /// Supported occurrence scheduling modes.
    pub scheduling_modes: Vec<EvaluationSchedulingMode>,
    /// Registered semantic operation schemas.
    pub operations: Vec<EvaluationOperationDescriptor>,
    /// Mandatory isolation outcome contract.
    pub isolation: EvaluatorIsolationRequirements,
    /// Factory-owned authored-config schema version.
    pub config_schema_version: u32,
    /// Canonical schema fingerprint.
    pub config_schema_sha256: String,
    /// Registered reviewed public-projection schemas by projection name.
    #[serde(default)]
    pub public_projection_schemas: BTreeMap<String, String>,
    /// Immutable selectable worker distributions.
    pub distributions: Vec<EvaluationDistributionDescriptor>,
}

impl EvaluationProviderDescriptor {
    /// Validate deterministic identity, schema, and distribution invariants.
    pub fn validate(&self) -> Result<(), ProviderRegistryError> {
        if self.display_name.trim().is_empty()
            || self.worker_protocol_versions != [EVALUATOR_WORKER_PROTOCOL_V2]
            || self.execution_granularities.is_empty()
            || self.scheduling_modes.is_empty()
            || self.operations.is_empty()
            || self.config_schema_version == 0
            || !is_sha256(&self.config_schema_sha256)
            || self.distributions.is_empty()
        {
            return Err(ProviderRegistryError::InvalidDescriptor(format!(
                "provider {} descriptor was incomplete",
                self.provider_id
            )));
        }
        self.isolation.validate()?;
        let mut operation_ids = BTreeSet::new();
        for operation in &self.operations {
            operation.validate()?;
            if !operation_ids.insert(&operation.operation_id) {
                return Err(ProviderRegistryError::InvalidDescriptor(format!(
                    "provider {} duplicated operation {}",
                    self.provider_id, operation.operation_id
                )));
            }
        }
        let mut distributions = BTreeSet::new();
        for distribution in &self.distributions {
            distribution.validate()?;
            if !distributions.insert(&distribution.distribution_id) {
                return Err(ProviderRegistryError::InvalidDescriptor(format!(
                    "provider {} duplicated distribution {}",
                    self.provider_id, distribution.distribution_id
                )));
            }
        }
        for (projection, digest) in &self.public_projection_schemas {
            if projection.trim().is_empty() || !is_sha256(digest) {
                return Err(ProviderRegistryError::InvalidDescriptor(format!(
                    "provider {} public projection schema was invalid",
                    self.provider_id
                )));
            }
        }
        Ok(())
    }

    /// Find one exact registered immutable distribution.
    pub fn distribution(
        &self,
        id: &EvaluationDistributionId,
    ) -> Option<&EvaluationDistributionDescriptor> {
        self.distributions
            .iter()
            .find(|distribution| &distribution.distribution_id == id)
    }
}

/// Factory-owned validated authored configuration.
///
/// Fields are private so only the factory schema validator can mint a value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidatedProviderConfig {
    provider_id: EvaluationProviderId,
    schema_version: u32,
    schema_sha256: String,
    config: CanonicalJson,
    config_sha256: String,
}

impl ValidatedProviderConfig {
    /// Provider that owns this validation result.
    pub fn provider_id(&self) -> &EvaluationProviderId {
        &self.provider_id
    }

    /// Factory-owned schema version.
    pub fn schema_version(&self) -> u32 {
        self.schema_version
    }

    /// Factory-owned schema fingerprint.
    pub fn schema_sha256(&self) -> &str {
        &self.schema_sha256
    }

    /// Strict provider-authored canonical value.
    pub fn config(&self) -> &CanonicalJson {
        &self.config
    }

    /// Secret-free canonical authored-config digest.
    pub fn config_sha256(&self) -> &str {
        &self.config_sha256
    }

    /// Create the exact worker planning request for one Rust-minted session.
    pub fn plan_request(
        &self,
        session_id: EvaluationSessionId,
        distribution_id: EvaluationDistributionId,
        reproducible: bool,
    ) -> EvaluationPlanRequest {
        EvaluationPlanRequest {
            session_id,
            provider_id: self.provider_id.clone(),
            distribution_id,
            config_schema_version: self.schema_version,
            config_schema_sha256: self.schema_sha256.clone(),
            provider_config: self.config.clone(),
            reproducible,
        }
    }
}

/// Pure side-effect-free provider-authored config validator.
pub trait ProviderConfigValidator: Send + Sync {
    /// Validate and normalize the exact authored value without effects.
    fn validate(&self, config: &CanonicalJson) -> Result<CanonicalJson, ProviderRegistryError>;
}

/// Bounds for the dedicated evaluator-worker protocol.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvaluatorProtocolLimits {
    /// Maximum encoded JSONL message bytes, excluding newline.
    pub max_message_bytes: usize,
    /// Maximum items in any protocol operation batch.
    pub max_collection_items: usize,
    /// Maximum long-poll wait requested from a worker.
    pub max_poll_wait_ms: u64,
    /// Maximum remembered idempotency keys for one session.
    pub max_idempotency_keys: usize,
}

impl Default for EvaluatorProtocolLimits {
    fn default() -> Self {
        Self {
            max_message_bytes: 8 * 1024 * 1024,
            max_collection_items: 4_096,
            max_poll_wait_ms: 60_000,
            max_idempotency_keys: 1_000_000,
        }
    }
}

impl EvaluatorProtocolLimits {
    /// Reject zero or impractically unbounded protocol limits.
    pub fn validate(&self) -> Result<(), ProviderRegistryError> {
        if !(1_024..=64 * 1024 * 1024).contains(&self.max_message_bytes)
            || !(1..=65_536).contains(&self.max_collection_items)
            || self.max_poll_wait_ms > 300_000
            || !(1..=10_000_000).contains(&self.max_idempotency_keys)
        {
            return Err(ProviderRegistryError::InvalidLaunch(
                "evaluator protocol limits were zero or exceeded hard safety ceilings".to_string(),
            ));
        }
        Ok(())
    }
}

/// Rust-owned context supplied to a factory launch.
#[derive(Debug, Clone)]
pub struct ProviderLaunchContext {
    /// Rust-minted evaluation session.
    pub session_id: EvaluationSessionId,
    /// Contained provider artifact staging root.
    pub staging_dir: PathBuf,
    /// Optional scoped local compatibility proxy.
    pub proxy: Option<ScopedProxyBinding>,
    /// Hard evaluator protocol bounds.
    pub protocol_limits: EvaluatorProtocolLimits,
    /// Rust-minted nonce binding `hello` to this exact launch.
    pub launch_nonce: String,
}

impl ProviderLaunchContext {
    /// Validate the contained launch context before process creation.
    pub fn validate(&self) -> Result<(), ProviderRegistryError> {
        self.protocol_limits.validate()?;
        if !self.staging_dir.is_absolute()
            || self.launch_nonce.len() < 32
            || self.launch_nonce.len() > 512
            || self.launch_nonce.chars().any(char::is_whitespace)
        {
            return Err(ProviderRegistryError::InvalidLaunch(
                "provider launch context lacked an absolute staging root or strong nonce"
                    .to_string(),
            ));
        }
        if let Some(proxy) = &self.proxy {
            proxy
                .validate()
                .map_err(|error| ProviderRegistryError::InvalidLaunch(error.to_string()))?;
        }
        Ok(())
    }
}

/// Unified provider-neutral evaluator session seam.
#[async_trait(?Send)]
pub trait EvaluationProvider {
    /// Exact negotiated and attested worker identity.
    fn identity(&self) -> &EvaluationWorkerIdentity;

    /// Current strict worker/session lifecycle state.
    fn lifecycle_state(&self) -> crate::lifecycle::EvaluationLifecycleState;

    /// Process-tree quiescence evidence available only after successful shutdown.
    fn quiescence_proof(&self) -> Option<&crate::isolation::IsolationQuiescenceProof>;

    /// Perform side-effect-free provider-owned dynamic planning.
    async fn plan(
        &mut self,
        request: &EvaluationPlanRequest,
    ) -> Result<EvaluationPlan, EvaluationProviderError>;

    /// Bind Rust-resolved immutable assets and freeze evaluation identity.
    async fn bind_assets(
        &mut self,
        assets: &[ResolvedEvaluationAsset],
    ) -> Result<EvaluationIdentity, EvaluationProviderError>;

    /// Retrieve one ordered page of finite units.
    async fn next_units(
        &mut self,
        offset: usize,
        limit: usize,
    ) -> Result<EvaluationUnitPage, EvaluationProviderError>;

    /// Deterministically instantiate Rust-scheduled occurrences.
    async fn instantiate_units(
        &mut self,
        requests: &[EvaluationUnitOccurrenceRequest],
    ) -> Result<Vec<EvaluationUnitOccurrence>, EvaluationProviderError>;

    /// Start provider-owned background unit tasks without synchronously waiting.
    async fn start_units(
        &mut self,
        ids: &[EvaluationUnitId],
    ) -> Result<(), EvaluationProviderError>;

    /// Bounded long-poll for semantic events and typed host operations.
    async fn poll_events(
        &mut self,
        limit: usize,
        wait_ms: u64,
    ) -> Result<EvaluationEventBatch, EvaluationProviderError>;

    /// Submit typed stream/usage/terminal events produced by Rust host executors.
    async fn submit_host_events(
        &mut self,
        events: &[HostOperationEvent],
    ) -> Result<(), EvaluationProviderError>;

    /// Idempotently cancel active evaluator units.
    async fn cancel_units(
        &mut self,
        ids: &[EvaluationUnitId],
    ) -> Result<(), EvaluationProviderError>;

    /// Finalize only after every unit and host operation drains.
    async fn finalize_candidate(
        &mut self,
    ) -> Result<EvaluationFinishCandidate, EvaluationProviderError>;

    /// Gracefully stop the worker and wait for complete process-tree quiescence.
    async fn shutdown(&mut self) -> Result<(), EvaluationProviderError>;

    /// Record that Rust sealed every candidate artifact after quiescence.
    fn mark_artifacts_sealed(&mut self) -> Result<(), EvaluationProviderError>;

    /// Record native report commit after artifact sealing.
    fn mark_report_committed(&mut self) -> Result<(), EvaluationProviderError>;
}

/// Factory launch implementation separated from provider identity/config policy.
#[async_trait(?Send)]
pub trait EvaluationProviderLauncher: Send + Sync {
    /// Attest, isolate, launch, and negotiate one exact registered distribution.
    async fn launch(
        &self,
        descriptor: &EvaluationProviderDescriptor,
        distribution: &EvaluationDistributionDescriptor,
        config: &ValidatedProviderConfig,
        context: &ProviderLaunchContext,
    ) -> Result<Box<dyn EvaluationProvider>, EvaluationProviderError>;
}

/// Object-safe provider factory registered with the runner.
#[async_trait(?Send)]
pub trait EvaluationProviderFactory: Send + Sync {
    /// Immutable capability, schema, isolation, and distribution descriptor.
    fn descriptor(&self) -> &EvaluationProviderDescriptor;

    /// Strictly validate authored configuration without any external effect.
    fn validate_authored_config(
        &self,
        config: &CanonicalJson,
    ) -> Result<ValidatedProviderConfig, ProviderRegistryError>;

    /// Launch one registered immutable distribution.
    async fn launch(
        &self,
        distribution: &EvaluationDistributionId,
        config: &ValidatedProviderConfig,
        context: &ProviderLaunchContext,
    ) -> Result<Box<dyn EvaluationProvider>, EvaluationProviderError>;
}

struct RegisteredProviderFactory {
    descriptor: EvaluationProviderDescriptor,
    validator: Arc<dyn ProviderConfigValidator>,
    launcher: Arc<dyn EvaluationProviderLauncher>,
}

impl RegisteredProviderFactory {
    fn new(
        descriptor: EvaluationProviderDescriptor,
        validator: Arc<dyn ProviderConfigValidator>,
        launcher: Arc<dyn EvaluationProviderLauncher>,
    ) -> Result<Self, ProviderRegistryError> {
        descriptor.validate()?;
        Ok(Self {
            descriptor,
            validator,
            launcher,
        })
    }
}

#[async_trait(?Send)]
impl EvaluationProviderFactory for RegisteredProviderFactory {
    fn descriptor(&self) -> &EvaluationProviderDescriptor {
        &self.descriptor
    }

    fn validate_authored_config(
        &self,
        config: &CanonicalJson,
    ) -> Result<ValidatedProviderConfig, ProviderRegistryError> {
        validate_no_secret_control_value(config)
            .map_err(|error| ProviderRegistryError::InvalidConfig(error.to_string()))?;
        let normalized = self.validator.validate(config)?;
        validate_no_secret_control_value(&normalized)
            .map_err(|error| ProviderRegistryError::InvalidConfig(error.to_string()))?;
        let config_sha256 = sha256_hex(&normalized.to_bytes());
        Ok(ValidatedProviderConfig {
            provider_id: self.descriptor.provider_id.clone(),
            schema_version: self.descriptor.config_schema_version,
            schema_sha256: self.descriptor.config_schema_sha256.clone(),
            config: normalized,
            config_sha256,
        })
    }

    async fn launch(
        &self,
        distribution: &EvaluationDistributionId,
        config: &ValidatedProviderConfig,
        context: &ProviderLaunchContext,
    ) -> Result<Box<dyn EvaluationProvider>, EvaluationProviderError> {
        context
            .validate()
            .map_err(EvaluationProviderError::registry)?;
        if config.provider_id != self.descriptor.provider_id
            || config.schema_version != self.descriptor.config_schema_version
            || config.schema_sha256 != self.descriptor.config_schema_sha256
        {
            return Err(EvaluationProviderError::FactoryMismatch(
                "validated provider configuration did not belong to the selected factory"
                    .to_string(),
            ));
        }
        let distribution = self.descriptor.distribution(distribution).ok_or_else(|| {
            EvaluationProviderError::FactoryMismatch(format!(
                "distribution {distribution} is not registered for provider {}",
                self.descriptor.provider_id
            ))
        })?;
        self.launcher
            .launch(&self.descriptor, distribution, config, context)
            .await
    }
}

/// Deterministic builder for the open evaluator-provider registry.
#[derive(Default)]
pub struct EvaluationProviderRegistryBuilder {
    factories: BTreeMap<EvaluationProviderId, Arc<dyn EvaluationProviderFactory>>,
}

impl EvaluationProviderRegistryBuilder {
    /// Create an empty registry builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register one provider factory transactionally.
    pub fn register(
        &mut self,
        factory: Arc<dyn EvaluationProviderFactory>,
    ) -> Result<&mut Self, ProviderRegistryError> {
        factory.descriptor().validate()?;
        let id = factory.descriptor().provider_id.clone();
        if self.factories.contains_key(&id) {
            return Err(ProviderRegistryError::DuplicateProvider(id.to_string()));
        }
        self.factories.insert(id, factory);
        Ok(self)
    }

    /// Freeze deterministic provider order and reject an empty inventory.
    pub fn freeze(self) -> Result<EvaluationProviderRegistry, ProviderRegistryError> {
        if self.factories.is_empty() {
            return Err(ProviderRegistryError::EmptyRegistry);
        }
        let mut operation_schemas =
            BTreeMap::<SemanticOperationId, EvaluationOperationDescriptor>::new();
        for factory in self.factories.values() {
            for operation in &factory.descriptor().operations {
                if let Some(existing) = operation_schemas.get(&operation.operation_id) {
                    if existing != operation {
                        return Err(ProviderRegistryError::ConflictingOperationSchema(
                            operation.operation_id.to_string(),
                        ));
                    }
                } else {
                    operation_schemas.insert(operation.operation_id.clone(), operation.clone());
                }
            }
        }
        Ok(EvaluationProviderRegistry {
            factories: self.factories,
        })
    }
}

/// Frozen deterministic open evaluator-provider registry.
#[derive(Clone)]
pub struct EvaluationProviderRegistry {
    factories: BTreeMap<EvaluationProviderId, Arc<dyn EvaluationProviderFactory>>,
}

impl EvaluationProviderRegistry {
    /// Resolve one exact provider without string-mode branching.
    pub fn get(
        &self,
        provider_id: &EvaluationProviderId,
    ) -> Option<&Arc<dyn EvaluationProviderFactory>> {
        self.factories.get(provider_id)
    }

    /// Strictly decode and validate authored JSON bytes for one provider/distribution.
    ///
    /// The byte-oriented entry point preserves duplicate-key rejection before
    /// `serde_json::Value` could overwrite an earlier key.
    pub fn validate_authored_bytes(
        &self,
        provider_id: &EvaluationProviderId,
        distribution_id: &EvaluationDistributionId,
        bytes: &[u8],
    ) -> Result<ValidatedProviderConfig, ProviderRegistryError> {
        let canonical = CanonicalJson::from_slice(bytes, Default::default())
            .map_err(|error| ProviderRegistryError::InvalidConfig(error.to_string()))?;
        self.validate_authored_config(provider_id, distribution_id, &canonical)
    }

    /// Validate an already decoded authored value for one provider/distribution.
    pub fn validate_authored_value(
        &self,
        provider_id: &EvaluationProviderId,
        distribution_id: &EvaluationDistributionId,
        value: &Value,
    ) -> Result<ValidatedProviderConfig, ProviderRegistryError> {
        let canonical = CanonicalJson::new(value.clone())
            .map_err(|error| ProviderRegistryError::InvalidConfig(error.to_string()))?;
        self.validate_authored_config(provider_id, distribution_id, &canonical)
    }

    /// Validate canonical authored config and exact immutable distribution selection.
    pub fn validate_authored_config(
        &self,
        provider_id: &EvaluationProviderId,
        distribution_id: &EvaluationDistributionId,
        config: &CanonicalJson,
    ) -> Result<ValidatedProviderConfig, ProviderRegistryError> {
        let factory = self
            .factories
            .get(provider_id)
            .ok_or_else(|| ProviderRegistryError::UnknownProvider(provider_id.to_string()))?;
        if factory.descriptor().distribution(distribution_id).is_none() {
            return Err(ProviderRegistryError::UnknownDistribution {
                provider: provider_id.to_string(),
                distribution: distribution_id.to_string(),
            });
        }
        factory.validate_authored_config(config)
    }

    /// Iterate descriptors in stable provider-ID order.
    pub fn descriptors(&self) -> impl ExactSizeIterator<Item = &EvaluationProviderDescriptor> {
        self.factories.values().map(|factory| factory.descriptor())
    }

    /// Number of registered provider process protocols.
    pub fn len(&self) -> usize {
        self.factories.len()
    }

    /// Whether no providers are registered.
    pub fn is_empty(&self) -> bool {
        self.factories.is_empty()
    }
}

/// NeMo Evaluator provider factory with an immutable deployment-owned launch.
pub struct NemoEvaluatorProviderFactory(RegisteredProviderFactory);

impl NemoEvaluatorProviderFactory {
    /// Build the stock `nemo_evaluator` process-protocol factory.
    pub fn new(
        distributions: Vec<EvaluationDistributionDescriptor>,
        launcher: Arc<dyn EvaluationProviderLauncher>,
    ) -> Result<Self, ProviderRegistryError> {
        let schema = nemo_schema();
        let descriptor = stock_descriptor(
            "nemo_evaluator",
            "NeMo Evaluator",
            schema.normalized_result_sha256(),
            distributions,
            vec![
                EvaluationExecutionGranularity::Case,
                EvaluationExecutionGranularity::HostBatch,
            ],
            vec![
                EvaluationSchedulingMode::Finite,
                EvaluationSchedulingMode::RustOccurrences,
            ],
        )?;
        Ok(Self(RegisteredProviderFactory::new(
            descriptor,
            Arc::new(NemoConfigValidator),
            launcher,
        )?))
    }
}

/// OpenBench/pinned-Inspect provider factory with an immutable launch.
pub struct OpenBenchProviderFactory(RegisteredProviderFactory);

impl OpenBenchProviderFactory {
    /// Build the stock `openbench` process-protocol factory.
    pub fn new(
        distributions: Vec<EvaluationDistributionDescriptor>,
        launcher: Arc<dyn EvaluationProviderLauncher>,
    ) -> Result<Self, ProviderRegistryError> {
        let schema = openbench_schema();
        let descriptor = stock_descriptor(
            "openbench",
            "OpenBench / Inspect AI",
            schema.normalized_result_sha256(),
            distributions,
            vec![EvaluationExecutionGranularity::HostBatch],
            vec![EvaluationSchedulingMode::Finite],
        )?;
        Ok(Self(RegisteredProviderFactory::new(
            descriptor,
            Arc::new(OpenBenchConfigValidator),
            launcher,
        )?))
    }
}

macro_rules! delegate_factory {
    ($type:ty) => {
        #[async_trait(?Send)]
        impl EvaluationProviderFactory for $type {
            fn descriptor(&self) -> &EvaluationProviderDescriptor {
                self.0.descriptor()
            }

            fn validate_authored_config(
                &self,
                config: &CanonicalJson,
            ) -> Result<ValidatedProviderConfig, ProviderRegistryError> {
                self.0.validate_authored_config(config)
            }

            async fn launch(
                &self,
                distribution: &EvaluationDistributionId,
                config: &ValidatedProviderConfig,
                context: &ProviderLaunchContext,
            ) -> Result<Box<dyn EvaluationProvider>, EvaluationProviderError> {
                self.0.launch(distribution, config, context).await
            }
        }
    };
}

delegate_factory!(NemoEvaluatorProviderFactory);
delegate_factory!(OpenBenchProviderFactory);

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct NemoAuthoredConfig {
    benchmark: String,
    #[serde(default)]
    task_names: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    max_cases: Option<usize>,
    #[serde(default)]
    seed: u64,
    #[serde(default)]
    shuffle: bool,
    #[serde(default)]
    shard_index: usize,
    #[serde(default = "one_usize")]
    shard_count: usize,
    #[serde(default = "one_usize")]
    repeat: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    solver: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    environment: Option<String>,
    #[serde(default)]
    provider_options: BTreeMap<String, CanonicalJson>,
}

struct NemoConfigValidator;

impl ProviderConfigValidator for NemoConfigValidator {
    fn validate(&self, config: &CanonicalJson) -> Result<CanonicalJson, ProviderRegistryError> {
        let decoded: NemoAuthoredConfig = serde_json::from_value(config.value().clone())
            .map_err(|error| ProviderRegistryError::InvalidConfig(error.to_string()))?;
        if decoded.benchmark.trim().is_empty()
            || decoded.max_cases == Some(0)
            || decoded.shard_count == 0
            || decoded.shard_index >= decoded.shard_count
            || decoded.repeat == 0
            || decoded.repeat > 10_000
            || decoded.task_names.iter().any(|task| task.trim().is_empty())
        {
            return Err(ProviderRegistryError::InvalidConfig(
                "NeMo Evaluator config had an empty benchmark/task or invalid selection policy"
                    .to_string(),
            ));
        }
        normalize_config(decoded)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct OpenBenchAuthoredConfig {
    task: String,
    #[serde(default)]
    task_args: BTreeMap<String, CanonicalJson>,
    #[serde(default = "one_usize")]
    epochs: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    max_samples: Option<usize>,
    #[serde(default)]
    seed: u64,
    #[serde(default = "default_candidate_service")]
    candidate_service: LogicalServiceId,
    #[serde(default)]
    auxiliary_services: BTreeMap<String, LogicalServiceId>,
}

struct OpenBenchConfigValidator;

impl ProviderConfigValidator for OpenBenchConfigValidator {
    fn validate(&self, config: &CanonicalJson) -> Result<CanonicalJson, ProviderRegistryError> {
        let decoded: OpenBenchAuthoredConfig = serde_json::from_value(config.value().clone())
            .map_err(|error| ProviderRegistryError::InvalidConfig(error.to_string()))?;
        if decoded.task.trim().is_empty()
            || decoded.task.contains(',')
            || decoded.epochs == 0
            || decoded.epochs > 10_000
            || decoded.max_samples == Some(0)
            || decoded
                .auxiliary_services
                .keys()
                .any(|role| role.trim().is_empty())
        {
            return Err(ProviderRegistryError::InvalidConfig(
                "OpenBench config must select one exact task and bounded nonzero epochs/samples"
                    .to_string(),
            ));
        }
        normalize_config(decoded)
    }
}

fn normalize_config<T: Serialize>(value: T) -> Result<CanonicalJson, ProviderRegistryError> {
    let value = serde_json::to_value(value)
        .map_err(|error| ProviderRegistryError::InvalidConfig(error.to_string()))?;
    CanonicalJson::new(value)
        .map_err(|error| ProviderRegistryError::InvalidConfig(error.to_string()))
}

fn one_usize() -> usize {
    1
}

fn default_candidate_service() -> LogicalServiceId {
    LogicalServiceId::new("candidate").expect("stock logical service is valid")
}

fn nemo_schema() -> CanonicalJson {
    CanonicalJson::new(serde_json::json!({
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "additionalProperties": false,
        "properties": {
            "benchmark": {"type": "string", "minLength": 1},
            "environment": {"type": ["string", "null"]},
            "max_cases": {"type": ["integer", "null"], "minimum": 1},
            "provider_options": {"type": "object"},
            "repeat": {"type": "integer", "minimum": 1, "maximum": 10000},
            "seed": {"type": "integer", "minimum": 0},
            "shard_count": {"type": "integer", "minimum": 1},
            "shard_index": {"type": "integer", "minimum": 0},
            "shuffle": {"type": "boolean"},
            "solver": {"type": ["string", "null"]},
            "task_names": {"type": "array", "items": {"type": "string", "minLength": 1}}
        },
        "required": ["benchmark"],
        "title": "aiperf-nemo-evaluator-config-v1",
        "type": "object"
    }))
    .expect("stock schema is canonical JSON")
}

fn openbench_schema() -> CanonicalJson {
    CanonicalJson::new(serde_json::json!({
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "additionalProperties": false,
        "properties": {
            "auxiliary_services": {"type": "object"},
            "candidate_service": {"type": "string", "pattern": "^[a-z0-9][a-z0-9_.-]*$"},
            "epochs": {"type": "integer", "minimum": 1, "maximum": 10000},
            "max_samples": {"type": ["integer", "null"], "minimum": 1},
            "seed": {"type": "integer", "minimum": 0},
            "task": {"type": "string", "minLength": 1},
            "task_args": {"type": "object"}
        },
        "required": ["task"],
        "title": "aiperf-openbench-config-v1",
        "type": "object"
    }))
    .expect("stock schema is canonical JSON")
}

fn stock_descriptor(
    provider: &str,
    display_name: &str,
    config_schema_sha256: String,
    distributions: Vec<EvaluationDistributionDescriptor>,
    execution_granularities: Vec<EvaluationExecutionGranularity>,
    scheduling_modes: Vec<EvaluationSchedulingMode>,
) -> Result<EvaluationProviderDescriptor, ProviderRegistryError> {
    let operations = [
        ("model.generate", true),
        ("model.complete", true),
        ("model.responses", true),
        ("model.embed", false),
        ("sandbox.create", false),
        ("sandbox.execute", true),
        ("sandbox.destroy", false),
        ("process.start", false),
        ("process.interact", true),
        ("process.stop", false),
        ("resource.invoke", true),
    ]
    .into_iter()
    .map(|(operation, streaming)| {
        let operation_id = SemanticOperationId::new(operation)
            .map_err(|error| ProviderRegistryError::InvalidDescriptor(error.to_string()))?;
        let schema_seed = format!("{CANONICAL_JSON_CODEC}:{operation}:v1");
        Ok(EvaluationOperationDescriptor {
            operation_id,
            input_schema_sha256: sha256_hex(format!("{schema_seed}:input").as_bytes()),
            output_schema_sha256: sha256_hex(format!("{schema_seed}:output").as_bytes()),
            stream_schema_sha256: streaming
                .then(|| sha256_hex(format!("{schema_seed}:stream").as_bytes())),
            reports_usage: operation.starts_with("model."),
            modalities: if operation == "model.embed" {
                vec!["text".to_string(), "embedding".to_string()]
            } else {
                vec!["text".to_string(), "structured".to_string()]
            },
            endpoint_capabilities: vec![operation.to_string()],
        })
    })
    .collect::<Result<Vec<_>, ProviderRegistryError>>()?;
    let descriptor = EvaluationProviderDescriptor {
        provider_id: EvaluationProviderId::new(provider)
            .map_err(|error| ProviderRegistryError::InvalidDescriptor(error.to_string()))?,
        display_name: display_name.to_string(),
        worker_protocol_versions: vec![EVALUATOR_WORKER_PROTOCOL_V2],
        execution_granularities,
        scheduling_modes,
        operations,
        isolation: EvaluatorIsolationRequirements::strict_process_tree(),
        config_schema_version: 1,
        config_schema_sha256,
        public_projection_schemas: BTreeMap::new(),
        distributions,
    };
    descriptor.validate()?;
    Ok(descriptor)
}

/// Deterministic registry/config validation failure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProviderRegistryError {
    /// Provider ID was registered more than once.
    DuplicateProvider(String),
    /// Provider identity was not registered in this exact runner image.
    UnknownProvider(String),
    /// Distribution identity was not registered for the selected provider.
    UnknownDistribution {
        /// Selected provider.
        provider: String,
        /// Selected distribution.
        distribution: String,
    },
    /// Providers assigned incompatible schemas/capabilities to one semantic operation ID.
    ConflictingOperationSchema(String),
    /// A frozen product registry cannot be empty.
    EmptyRegistry,
    /// Provider descriptor was incomplete or mutable.
    InvalidDescriptor(String),
    /// Pure authored configuration failed strict schema validation.
    InvalidConfig(String),
    /// Factory-owned launch context was invalid.
    InvalidLaunch(String),
}

impl Display for ProviderRegistryError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DuplicateProvider(id) => write!(formatter, "duplicate evaluator provider {id}"),
            Self::UnknownProvider(id) => write!(formatter, "unknown evaluator provider {id}"),
            Self::UnknownDistribution {
                provider,
                distribution,
            } => write!(
                formatter,
                "unknown evaluator distribution {distribution} for provider {provider}"
            ),
            Self::ConflictingOperationSchema(operation) => write!(
                formatter,
                "conflicting evaluator-provider schemas for semantic operation {operation}"
            ),
            Self::EmptyRegistry => formatter.write_str("evaluator provider registry was empty"),
            Self::InvalidDescriptor(message) => {
                write!(
                    formatter,
                    "invalid evaluator provider descriptor: {message}"
                )
            }
            Self::InvalidConfig(message) => {
                write!(formatter, "invalid evaluator provider config: {message}")
            }
            Self::InvalidLaunch(message) => {
                write!(formatter, "invalid evaluator provider launch: {message}")
            }
        }
    }
}

impl std::error::Error for ProviderRegistryError {}

/// Evaluator process, protocol, lifecycle, or remote infrastructure failure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EvaluationProviderError {
    /// Factory/config/distribution identities did not match.
    FactoryMismatch(String),
    /// Launch attestation or isolation failed closed.
    Launch(String),
    /// Dedicated control descriptor I/O failed.
    Io(String),
    /// Strict evaluator-worker protocol invariant failed.
    Protocol(String),
    /// Evaluator lifecycle state transition was illegal.
    Lifecycle(String),
    /// Provider reported an infrastructure error.
    Remote(crate::provider_protocol::EvaluationError),
    /// Worker or a required descendant exited unexpectedly.
    Crashed(String),
    /// Worker tree failed to become quiescent.
    Quiescence(String),
}

impl EvaluationProviderError {
    pub(crate) fn registry(error: ProviderRegistryError) -> Self {
        Self::Launch(error.to_string())
    }

    /// Wrap a provider-protocol validation failure.
    pub fn protocol(error: EvaluationProtocolError) -> Self {
        Self::Protocol(error.to_string())
    }
}

impl Display for EvaluationProviderError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::FactoryMismatch(message) => {
                write!(formatter, "provider factory mismatch: {message}")
            }
            Self::Launch(message) => {
                write!(formatter, "evaluator provider launch failed: {message}")
            }
            Self::Io(message) => write!(
                formatter,
                "evaluator provider control I/O failed: {message}"
            ),
            Self::Protocol(message) => {
                write!(formatter, "evaluator-worker protocol v2 failed: {message}")
            }
            Self::Lifecycle(message) => write!(formatter, "evaluator lifecycle failed: {message}"),
            Self::Remote(error) => write!(
                formatter,
                "evaluator provider operation failed at {} ({}): {}",
                error.stage, error.error_kind, error.message
            ),
            Self::Crashed(message) => write!(formatter, "evaluator worker crashed: {message}"),
            Self::Quiescence(message) => write!(
                formatter,
                "evaluator worker tree did not quiesce: {message}"
            ),
        }
    }
}

impl std::error::Error for EvaluationProviderError {}

impl From<EvaluationProtocolError> for EvaluationProviderError {
    fn from(error: EvaluationProtocolError) -> Self {
        Self::protocol(error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct NeverLaunch;

    #[async_trait(?Send)]
    impl EvaluationProviderLauncher for NeverLaunch {
        async fn launch(
            &self,
            _descriptor: &EvaluationProviderDescriptor,
            _distribution: &EvaluationDistributionDescriptor,
            _config: &ValidatedProviderConfig,
            _context: &ProviderLaunchContext,
        ) -> Result<Box<dyn EvaluationProvider>, EvaluationProviderError> {
            panic!("pure validation test must not launch a provider")
        }
    }

    fn distribution(name: &str) -> EvaluationDistributionDescriptor {
        EvaluationDistributionDescriptor {
            distribution_id: EvaluationDistributionId::new(name).unwrap(),
            package: name.to_string(),
            package_version: "1.0.0".to_string(),
            provider_source_sha256: "a".repeat(64),
            worker_source_sha256: "b".repeat(64),
            dependency_lock_sha256: "c".repeat(64),
            oci_digest: Some(format!("sha256:{}", "d".repeat(64))),
            launch_closure_sha256: "e".repeat(64),
        }
    }

    #[test]
    fn stock_validators_are_strict_pure_and_fingerprinted() {
        let factory = NemoEvaluatorProviderFactory::new(
            vec![distribution("nemo-locked")],
            Arc::new(NeverLaunch),
        )
        .unwrap();
        let config = CanonicalJson::new(serde_json::json!({
            "benchmark": "mmlu",
            "repeat": 2,
            "seed": 7
        }))
        .unwrap();
        let validated = factory.validate_authored_config(&config).unwrap();
        assert_eq!(validated.provider_id().as_str(), "nemo_evaluator");
        assert!(is_sha256(validated.schema_sha256()));
        assert!(is_sha256(validated.config_sha256()));

        let unknown = CanonicalJson::new(serde_json::json!({
            "benchmark": "mmlu",
            "executable": "/tmp/evil"
        }))
        .unwrap();
        assert!(factory.validate_authored_config(&unknown).is_err());
    }

    #[test]
    fn authored_connection_authority_fails_before_provider_validation() {
        let factory = OpenBenchProviderFactory::new(
            vec![distribution("openbench-locked")],
            Arc::new(NeverLaunch),
        )
        .unwrap();
        let config = CanonicalJson::new(serde_json::json!({
            "task": "simpleqa",
            "task_args": {"base_url": "https://real.invalid"}
        }))
        .unwrap();
        assert!(
            factory
                .validate_authored_config(&config)
                .unwrap_err()
                .to_string()
                .contains("connection authority")
        );
    }

    #[test]
    fn registry_is_deterministic_and_duplicate_safe() {
        let nemo: Arc<dyn EvaluationProviderFactory> = Arc::new(
            NemoEvaluatorProviderFactory::new(
                vec![distribution("nemo-locked")],
                Arc::new(NeverLaunch),
            )
            .unwrap(),
        );
        let openbench: Arc<dyn EvaluationProviderFactory> = Arc::new(
            OpenBenchProviderFactory::new(
                vec![distribution("openbench-locked")],
                Arc::new(NeverLaunch),
            )
            .unwrap(),
        );
        let mut builder = EvaluationProviderRegistryBuilder::new();
        builder.register(nemo.clone()).unwrap();
        builder.register(openbench).unwrap();
        assert!(builder.register(nemo).is_err());
        let registry = builder.freeze().unwrap();
        assert_eq!(
            registry
                .descriptors()
                .map(|descriptor| descriptor.provider_id.as_str())
                .collect::<Vec<_>>(),
            ["nemo_evaluator", "openbench"]
        );
    }
}
