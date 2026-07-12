// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Replaceable evaluator-provider seam and deterministic open registry.
//!
//! Factories register process-protocol implementations, never benchmarks or
//! graders. A factory owns the immutable worker launch identity and the pure
//! authored-config schema; run configuration cannot select an executable,
//! module, argument vector, environment, or upstream endpoint.

use std::any::Any;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::{self, Display};
use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::canonical::{CanonicalJson, is_sha256, sha256_hex, validate_no_secret_control_value};
use crate::provider_protocol::{
    EVALUATOR_WORKER_PROTOCOL_V2, EvaluationDistributionId, EvaluationEventBatch,
    EvaluationExecutionGranularity, EvaluationFinishCandidate, EvaluationIdentity, EvaluationPlan,
    EvaluationPlanRequest, EvaluationProtocolError, EvaluationProviderId, EvaluationSchedulingMode,
    EvaluationSessionId, EvaluationUnitId, EvaluationUnitOccurrence,
    EvaluationUnitOccurrenceRequest, EvaluationUnitPage, EvaluationWorkerIdentity,
    HostOperationEvent, ResolvedEvaluationAsset, ScopedProxyBinding, SemanticOperationId,
};
use crate::score_projection::PublicScoreProjectionPolicy;

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

/// Cross-language canonical schema digests for one stock inference operation.
///
/// These values are computed from the exact canonical schema components in
/// `src/aiperf/accuracy/evaluation/operation_schemas.py:15-409`. A terminal-only
/// operation still pins its canonical null-stream schema here while its public
/// descriptor advertises no true-streaming adapter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StockEvaluationOperationSchema {
    /// Open semantic operation ID.
    pub operation_id: &'static str,
    /// Canonical SHA-256 for the full `{request,response,stream}` capability schema.
    pub combined_schema_sha256: &'static str,
    /// Canonical request JSON Schema SHA-256.
    pub request_schema_sha256: &'static str,
    /// Canonical response JSON Schema SHA-256.
    pub response_schema_sha256: &'static str,
    /// Canonical stream JSON Schema SHA-256, including a null schema for terminal-only operations.
    pub canonical_stream_schema_sha256: &'static str,
    /// Whether the stock host contract permits true incremental events.
    pub true_streaming: bool,
    /// Endpoint capability required by the operation adapter.
    pub endpoint_capability: &'static str,
    /// Canonical modality inventory.
    pub modalities: &'static [&'static str],
}

/// Exact stock operation schemas shared with the evaluator-provider workers.
pub const STOCK_EVALUATION_OPERATION_SCHEMAS: &[StockEvaluationOperationSchema] = &[
    StockEvaluationOperationSchema {
        operation_id: "model.generate",
        combined_schema_sha256: "d468bbc4f1fdbbc54360cede8194732b2ebaabbdfb55490bc572c4bb44f89cdf",
        request_schema_sha256: "c2f30f5396f4af6e44025d80294b2685916492c23dd730cd1e2a6ebdb6ae5d21",
        response_schema_sha256: "6c8d726e5a0c05a22de946ce2495d6a4bcf3b3b7bb7a48e5c39bad07ff954ca0",
        canonical_stream_schema_sha256: "84a861ea0a983368cd48e6db2fa4ac71b8219d7685065718859f0bfc4ea49206",
        true_streaming: true,
        endpoint_capability: "chat",
        modalities: &[
            "audio",
            "document",
            "image",
            "structured",
            "text",
            "tool",
            "video",
        ],
    },
    StockEvaluationOperationSchema {
        operation_id: "model.complete",
        combined_schema_sha256: "bf476cc2ec39d6168c4dab8872bab0f70932aacf30afaa20a7a001800d135083",
        request_schema_sha256: "a2b90611bd9bef0849c7e9472820124432d0b68578a9c6d403fda7e7ea108901",
        response_schema_sha256: "274926adc0a1d6944d77b28882965e80e528bc7724da6aee075381b22d37af68",
        canonical_stream_schema_sha256: "c1b02dc133d9bed9f61fee3b434a9bb59dc0f0d834ad09715c9af549be1ab9f4",
        true_streaming: true,
        endpoint_capability: "completion",
        modalities: &["text"],
    },
    StockEvaluationOperationSchema {
        operation_id: "model.responses",
        combined_schema_sha256: "c58154fbc26e5a787a170b438eaccb391b00f2bc06a16ea6be82f9926c6e3286",
        request_schema_sha256: "ecfbf3cb2741066d555df42e01c4c3c68d37c3674d818116feeaf8fdead2d5b3",
        response_schema_sha256: "b1702513240975373a50c2720a9f1d39df8e2466560fe90174e63a0d631cd3e7",
        canonical_stream_schema_sha256: "800695bec0f214e79c9eb0b469ce020fc2931167126cb0d4e0ca7cce2d2f262e",
        true_streaming: true,
        endpoint_capability: "responses",
        modalities: &["document", "image", "structured", "text", "tool"],
    },
    StockEvaluationOperationSchema {
        operation_id: "model.embed",
        combined_schema_sha256: "9f68f85de35c85a33536a3bde8a2e727c81cd217479755f53eeadcc01f775749",
        request_schema_sha256: "d46a1567b54ffeb888fbecbb451e8f2790b846e5928e339bf7afeedd4be34f86",
        response_schema_sha256: "ddea46c3ffba13ff7b860ad0d6c6277c19a2f51b71adbc0cef76c0d8fd2c6bc1",
        canonical_stream_schema_sha256: "bcde375ebd4cbacf651311181173836b169d5a360c6ac158c6a2cdaf49be3f61",
        true_streaming: false,
        endpoint_capability: "embedding",
        modalities: &["embedding", "text"],
    },
];

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
    /// Factory-attested closure digest over sorted normalized worker-root-relative
    /// UTF-8 path, NUL, content SHA-256, and LF records.
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

/// Rust-only spawned-process authority supplied to the supervised launcher.
pub trait EvaluatorProcessRootBinder: fmt::Debug + Send + Sync {
    /// Bind the exact root PID returned by the supervised spawn operation.
    fn bind_attested_root(&self, root_pid: u32) -> Result<(), ProviderRegistryError>;
}

/// Rust-owned context bound into a one-shot prepared provider launch.
#[derive(Debug, Clone)]
pub struct ProviderLaunchContext {
    /// Rust-minted evaluation session.
    pub session_id: EvaluationSessionId,
    /// Contained provider artifact staging root.
    pub staging_dir: PathBuf,
    /// Optional scoped local compatibility proxy.
    pub proxy: Option<ScopedProxyBinding>,
    /// Optional Rust-only callback binding the spawned evaluator process tree.
    pub process_root_binder: Option<Arc<dyn EvaluatorProcessRootBinder>>,
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
            if proxy.grant.session_id != self.session_id {
                return Err(ProviderRegistryError::InvalidLaunch(
                    "scoped proxy grant session did not match provider launch session".to_string(),
                ));
            }
        }
        if self.proxy.is_some() != self.process_root_binder.is_some() {
            return Err(ProviderRegistryError::InvalidLaunch(
                "scoped proxy and evaluator process-root binder must be configured together"
                    .to_string(),
            ));
        }
        Ok(())
    }

    /// Digest the exact session/staging/proxy/limit/nonce context without
    /// exposing proxy capability material.
    pub fn binding_sha256(&self) -> Result<String, ProviderRegistryError> {
        self.validate()?;
        let staging_dir = self.staging_dir.to_str().ok_or_else(|| {
            ProviderRegistryError::InvalidLaunch(
                "provider staging root was not valid UTF-8".to_string(),
            )
        })?;
        let proxy = self.proxy.as_ref().map(|proxy| {
            serde_json::json!({
                "grant_id": proxy.grant.grant_id,
                "session_id": proxy.grant.session_id,
                "local_locator": proxy.local_locator,
                "host_socket_path": proxy.host_socket_path,
                "max_concurrent_operations": proxy.grant.max_concurrent_operations,
                "max_operations": proxy.grant.max_operations,
                "max_request_bytes": proxy.grant.max_request_bytes,
                "max_response_bytes": proxy.grant.max_response_bytes,
                "max_stream_events": proxy.grant.max_stream_events,
                "expires_after_ms": proxy.grant.expires_after_ms,
                "process_scope_sha256": proxy.grant.process_scope_sha256,
                "secret_sha256": sha256_hex(proxy.grant.secret.expose_secret().as_bytes()),
                "semantic_operation_ids": proxy.grant.semantic_operation_ids,
                "purposes": proxy.grant.purposes,
                "service_ids": proxy.grant.service_ids,
            })
        });
        CanonicalJson::new(serde_json::json!({
            "launch_nonce_sha256": sha256_hex(self.launch_nonce.as_bytes()),
            "protocol_limits": self.protocol_limits,
            "proxy": proxy,
            "process_root_binder": self.process_root_binder.is_some(),
            "session_id": self.session_id,
            "staging_dir": staging_dir,
        }))
        .map_err(|error| ProviderRegistryError::InvalidLaunch(error.to_string()))
        .map(|value| value.normalized_result_sha256())
    }
}

/// Opaque one-shot launch token minted by the selected factory launcher.
///
/// Implementations keep prepared command/isolation state private. Consuming
/// the boxed token prevents reuse and lets the launcher reject foreign tokens.
pub trait PreparedEvaluationProviderLaunch: fmt::Debug + Send {
    /// Exact immutable distribution bound into this preparation.
    fn distribution_id(&self) -> &EvaluationDistributionId;

    /// Exact prelaunch context digest bound into isolation preparation.
    fn prelaunch_context_sha256(&self) -> &str;

    /// Independently inspectable isolation evidence minted before host binding.
    fn isolation_evidence(&self) -> &crate::isolation::EvaluatorIsolationEvidence;

    /// Consume this launcher-specific token for checked downcasting.
    fn into_any(self: Box<Self>) -> Box<dyn Any + Send>;
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
    /// Prove the launch recipe, immutable closure, and isolation mechanism are
    /// available in this exact runner image without starting the worker.
    fn check_distribution_available(
        &self,
        distribution: &EvaluationDistributionDescriptor,
    ) -> Result<(), EvaluationProviderError>;

    /// Attest and isolate one exact launch context without starting a worker.
    fn prepare_launch(
        &self,
        descriptor: &EvaluationProviderDescriptor,
        distribution: &EvaluationDistributionDescriptor,
        context: ProviderLaunchContext,
    ) -> Result<Box<dyn PreparedEvaluationProviderLaunch>, EvaluationProviderError>;

    /// Consume one prepared launch, bind Rust host identity, spawn, and negotiate.
    async fn launch(
        &self,
        descriptor: &EvaluationProviderDescriptor,
        config: &ValidatedProviderConfig,
        host_binding: crate::provider_protocol::EvaluationHostBinding,
        prepared: Box<dyn PreparedEvaluationProviderLaunch>,
    ) -> Result<Box<dyn EvaluationProvider>, EvaluationProviderError>;
}

/// Object-safe provider factory registered with the runner.
#[async_trait(?Send)]
pub trait EvaluationProviderFactory: Send + Sync {
    /// Immutable capability, schema, isolation, and distribution descriptor.
    fn descriptor(&self) -> &EvaluationProviderDescriptor;

    /// Factory-owned executable validators for every public score schema.
    fn public_score_projection_policy(&self) -> &PublicScoreProjectionPolicy;

    /// Strictly validate authored configuration without any external effect.
    fn validate_authored_config(
        &self,
        config: &CanonicalJson,
    ) -> Result<ValidatedProviderConfig, ProviderRegistryError>;

    /// Prove one descriptor-backed distribution is executable by this factory.
    fn check_distribution_available(
        &self,
        distribution: &EvaluationDistributionId,
    ) -> Result<(), EvaluationProviderError>;

    /// Mint an opaque one-shot prepared token for one exact prelaunch context.
    fn prepare_launch(
        &self,
        distribution: &EvaluationDistributionId,
        context: ProviderLaunchContext,
    ) -> Result<Box<dyn PreparedEvaluationProviderLaunch>, EvaluationProviderError>;

    /// Consume a prepared token and bind the Rust-owned host identity.
    async fn launch(
        &self,
        prepared: Box<dyn PreparedEvaluationProviderLaunch>,
        config: &ValidatedProviderConfig,
        host_binding: crate::provider_protocol::EvaluationHostBinding,
    ) -> Result<Box<dyn EvaluationProvider>, EvaluationProviderError>;
}

struct RegisteredProviderFactory {
    descriptor: EvaluationProviderDescriptor,
    validator: Arc<dyn ProviderConfigValidator>,
    public_score_projection_policy: PublicScoreProjectionPolicy,
    launcher: Arc<dyn EvaluationProviderLauncher>,
}

impl RegisteredProviderFactory {
    fn new(
        descriptor: EvaluationProviderDescriptor,
        validator: Arc<dyn ProviderConfigValidator>,
        public_score_projection_policy: PublicScoreProjectionPolicy,
        launcher: Arc<dyn EvaluationProviderLauncher>,
    ) -> Result<Self, ProviderRegistryError> {
        descriptor.validate()?;
        public_score_projection_policy
            .validate_descriptor_fingerprints(&descriptor.public_projection_schemas)
            .map_err(|error| ProviderRegistryError::InvalidDescriptor(error.to_string()))?;
        Ok(Self {
            descriptor,
            validator,
            public_score_projection_policy,
            launcher,
        })
    }
}

#[async_trait(?Send)]
impl EvaluationProviderFactory for RegisteredProviderFactory {
    fn descriptor(&self) -> &EvaluationProviderDescriptor {
        &self.descriptor
    }

    fn public_score_projection_policy(&self) -> &PublicScoreProjectionPolicy {
        &self.public_score_projection_policy
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

    fn check_distribution_available(
        &self,
        distribution: &EvaluationDistributionId,
    ) -> Result<(), EvaluationProviderError> {
        let distribution = self.descriptor.distribution(distribution).ok_or_else(|| {
            EvaluationProviderError::FactoryMismatch(format!(
                "distribution {distribution} is not registered for provider {}",
                self.descriptor.provider_id
            ))
        })?;
        self.launcher.check_distribution_available(distribution)
    }

    fn prepare_launch(
        &self,
        distribution: &EvaluationDistributionId,
        context: ProviderLaunchContext,
    ) -> Result<Box<dyn PreparedEvaluationProviderLaunch>, EvaluationProviderError> {
        context
            .validate()
            .map_err(EvaluationProviderError::registry)?;
        let distribution = self.descriptor.distribution(distribution).ok_or_else(|| {
            EvaluationProviderError::FactoryMismatch(format!(
                "distribution {distribution} is not registered for provider {}",
                self.descriptor.provider_id
            ))
        })?;
        self.launcher.check_distribution_available(distribution)?;
        let prepared = self
            .launcher
            .prepare_launch(&self.descriptor, distribution, context)?;
        if prepared.distribution_id() != &distribution.distribution_id
            || !is_sha256(prepared.prelaunch_context_sha256())
            || prepared.isolation_evidence().enforced != self.descriptor.isolation
        {
            return Err(EvaluationProviderError::FactoryMismatch(
                "prepared launch token drifted from distribution/context/isolation policy"
                    .to_string(),
            ));
        }
        prepared.isolation_evidence().validate_strict()?;
        Ok(prepared)
    }

    async fn launch(
        &self,
        prepared: Box<dyn PreparedEvaluationProviderLaunch>,
        config: &ValidatedProviderConfig,
        host_binding: crate::provider_protocol::EvaluationHostBinding,
    ) -> Result<Box<dyn EvaluationProvider>, EvaluationProviderError> {
        host_binding.validate()?;
        if !is_sha256(prepared.prelaunch_context_sha256())
            || prepared.isolation_evidence().enforced != self.descriptor.isolation
        {
            return Err(EvaluationProviderError::FactoryMismatch(
                "prepared launch token did not satisfy factory isolation policy".to_string(),
            ));
        }
        prepared.isolation_evidence().validate_strict()?;
        if config.provider_id != self.descriptor.provider_id
            || config.schema_version != self.descriptor.config_schema_version
            || config.schema_sha256 != self.descriptor.config_schema_sha256
        {
            return Err(EvaluationProviderError::FactoryMismatch(
                "validated provider configuration did not belong to the selected factory"
                    .to_string(),
            ));
        }
        let distribution = self
            .descriptor
            .distribution(prepared.distribution_id())
            .ok_or_else(|| {
                EvaluationProviderError::FactoryMismatch(format!(
                    "prepared distribution {} is not registered for provider {}",
                    prepared.distribution_id(),
                    self.descriptor.provider_id
                ))
            })?;
        if prepared.isolation_evidence().proof_sha256 != host_binding.host.isolation_proof_sha256 {
            return Err(EvaluationProviderError::FactoryMismatch(
                "host binding isolation proof did not match the prepared launch token".to_string(),
            ));
        }
        self.launcher.check_distribution_available(distribution)?;
        self.launcher
            .launch(&self.descriptor, config, host_binding, prepared)
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
        factory
            .public_score_projection_policy()
            .validate_descriptor_fingerprints(&factory.descriptor().public_projection_schemas)
            .map_err(|error| ProviderRegistryError::InvalidDescriptor(error.to_string()))?;
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
        let mut available_distributions = BTreeMap::new();
        let mut unavailable_distributions = BTreeMap::new();
        for factory in self.factories.values() {
            let provider_id = factory.descriptor().provider_id.clone();
            let mut provider_available = BTreeSet::new();
            for distribution in &factory.descriptor().distributions {
                match factory.check_distribution_available(&distribution.distribution_id) {
                    Ok(()) => {
                        provider_available.insert(distribution.distribution_id.clone());
                    }
                    Err(error) => {
                        unavailable_distributions.insert(
                            (provider_id.clone(), distribution.distribution_id.clone()),
                            crate::canonical::redact_diagnostic(&error.to_string()),
                        );
                    }
                }
            }
            available_distributions.insert(provider_id, provider_available);
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
            available_distributions,
            unavailable_distributions,
        })
    }
}

/// Frozen deterministic open evaluator-provider registry.
#[derive(Clone)]
pub struct EvaluationProviderRegistry {
    factories: BTreeMap<EvaluationProviderId, Arc<dyn EvaluationProviderFactory>>,
    available_distributions: BTreeMap<EvaluationProviderId, BTreeSet<EvaluationDistributionId>>,
    unavailable_distributions: BTreeMap<(EvaluationProviderId, EvaluationDistributionId), String>,
}

impl EvaluationProviderRegistry {
    /// Resolve one exact provider without string-mode branching.
    pub fn get(
        &self,
        provider_id: &EvaluationProviderId,
    ) -> Option<&Arc<dyn EvaluationProviderFactory>> {
        self.factories.get(provider_id)
    }

    /// Resolve the exact factory-owned score projection validators for a provider.
    pub fn public_score_projection_policy(
        &self,
        provider_id: &EvaluationProviderId,
    ) -> Option<&PublicScoreProjectionPolicy> {
        self.factories
            .get(provider_id)
            .map(|factory| factory.public_score_projection_policy())
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
        if !self.distribution_is_available(provider_id, distribution_id) {
            return Err(ProviderRegistryError::UnavailableDistribution {
                provider: provider_id.to_string(),
                distribution: distribution_id.to_string(),
                reason: self
                    .unavailable_distributions
                    .get(&(provider_id.clone(), distribution_id.clone()))
                    .cloned()
                    .unwrap_or_else(|| {
                        "factory did not attest this distribution as executable".to_string()
                    }),
            });
        }
        factory.validate_authored_config(config)
    }

    /// Return whether one registered distribution passed launch/isolation
    /// availability attestation when this registry was frozen.
    pub fn distribution_is_available(
        &self,
        provider_id: &EvaluationProviderId,
        distribution_id: &EvaluationDistributionId,
    ) -> bool {
        self.available_distributions
            .get(provider_id)
            .is_some_and(|available| available.contains(distribution_id))
    }

    /// Borrow executable distributions in deterministic descriptor order.
    pub fn available_distributions(
        &self,
        provider_id: &EvaluationProviderId,
    ) -> Vec<&EvaluationDistributionDescriptor> {
        let Some(factory) = self.factories.get(provider_id) else {
            return Vec::new();
        };
        factory
            .descriptor()
            .distributions
            .iter()
            .filter(|distribution| {
                self.distribution_is_available(provider_id, &distribution.distribution_id)
            })
            .collect()
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
            PublicScoreProjectionPolicy::restricted_only(),
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
            PublicScoreProjectionPolicy::restricted_only(),
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

            fn public_score_projection_policy(&self) -> &PublicScoreProjectionPolicy {
                self.0.public_score_projection_policy()
            }

            fn validate_authored_config(
                &self,
                config: &CanonicalJson,
            ) -> Result<ValidatedProviderConfig, ProviderRegistryError> {
                self.0.validate_authored_config(config)
            }

            fn check_distribution_available(
                &self,
                distribution: &EvaluationDistributionId,
            ) -> Result<(), EvaluationProviderError> {
                self.0.check_distribution_available(distribution)
            }

            fn prepare_launch(
                &self,
                distribution: &EvaluationDistributionId,
                context: ProviderLaunchContext,
            ) -> Result<Box<dyn PreparedEvaluationProviderLaunch>, EvaluationProviderError> {
                self.0.prepare_launch(distribution, context)
            }

            async fn launch(
                &self,
                prepared: Box<dyn PreparedEvaluationProviderLaunch>,
                config: &ValidatedProviderConfig,
                host_binding: crate::provider_protocol::EvaluationHostBinding,
            ) -> Result<Box<dyn EvaluationProvider>, EvaluationProviderError> {
                self.0.launch(prepared, config, host_binding).await
            }
        }
    };
}

delegate_factory!(NemoEvaluatorProviderFactory);
delegate_factory!(OpenBenchProviderFactory);

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct NemoAuthoredConfig {
    environment: String,
    solver: String,
    #[serde(default)]
    environment_config: BTreeMap<String, CanonicalJson>,
    solver_config: BTreeMap<String, CanonicalJson>,
    selection: NemoSelection,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct NemoSelection {
    limit: usize,
    seed: i64,
}

struct NemoConfigValidator;

impl ProviderConfigValidator for NemoConfigValidator {
    fn validate(&self, config: &CanonicalJson) -> Result<CanonicalJson, ProviderRegistryError> {
        let decoded: NemoAuthoredConfig = serde_json::from_value(config.value().clone())
            .map_err(|error| ProviderRegistryError::InvalidConfig(error.to_string()))?;
        const SOLVER_FIELDS: &[&str] = &[
            "max_tokens",
            "temperature",
            "top_p",
            "seed",
            "stop",
            "frequency_penalty",
            "presence_penalty",
        ];
        let max_tokens = decoded
            .solver_config
            .get("max_tokens")
            .and_then(|value| value.value().as_u64());
        let numeric_valid = [
            "temperature",
            "top_p",
            "frequency_penalty",
            "presence_penalty",
        ]
        .iter()
        .all(|field| {
            decoded
                .solver_config
                .get(*field)
                .is_none_or(|value| value.value().is_number())
        });
        let seed_valid = decoded.solver_config.get("seed").is_none_or(|value| {
            value.value().as_i64().is_some() || value.value().as_u64().is_some()
        });
        let stop_valid = decoded.solver_config.get("stop").is_none_or(|value| {
            value
                .value()
                .as_array()
                .is_some_and(|items| items.iter().all(serde_json::Value::is_string))
        });
        if decoded.environment != "gsm8k"
            || decoded.solver != "chat"
            || !decoded.environment_config.is_empty()
            || decoded
                .solver_config
                .keys()
                .any(|field| !SOLVER_FIELDS.contains(&field.as_str()))
            || !max_tokens.is_some_and(|value| value > 0)
            || !numeric_valid
            || !seed_valid
            || !stop_valid
            || !(1..=5).contains(&decoded.selection.limit)
            || decoded.selection.seed != 0
        {
            return Err(ProviderRegistryError::InvalidConfig(
                "NeMo Evaluator config did not match the frozen GSM8K/chat canary".to_string(),
            ));
        }
        normalize_config(decoded)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct OpenBenchAuthoredConfig {
    task: String,
    task_args: BTreeMap<String, CanonicalJson>,
    epochs: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    limit: Option<usize>,
}

struct OpenBenchConfigValidator;

impl ProviderConfigValidator for OpenBenchConfigValidator {
    fn validate(&self, config: &CanonicalJson) -> Result<CanonicalJson, ProviderRegistryError> {
        let decoded: OpenBenchAuthoredConfig = serde_json::from_value(config.value().clone())
            .map_err(|error| ProviderRegistryError::InvalidConfig(error.to_string()))?;
        if decoded.task != "gsm8k"
            || !decoded.task_args.is_empty()
            || !(1..=8).contains(&decoded.epochs)
            || decoded.limit.is_some_and(|limit| !(1..=5).contains(&limit))
        {
            return Err(ProviderRegistryError::InvalidConfig(
                "OpenBench config did not match the frozen GSM8K/Inspect canary".to_string(),
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

fn nemo_schema() -> CanonicalJson {
    CanonicalJson::new(serde_json::json!({
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "additionalProperties": false,
        "properties": {
            "environment": {"const": "gsm8k"},
            "solver": {"const": "chat"},
            "environment_config": {"type": "object", "additionalProperties": false},
            "solver_config": {
                "type": "object",
                "additionalProperties": false,
                "required": ["max_tokens"],
                "properties": {
                    "max_tokens": {"type": "integer", "minimum": 1},
                    "temperature": {"type": "number"},
                    "top_p": {"type": "number"},
                    "seed": {"type": "integer"},
                    "stop": {"type": "array", "items": {"type": "string"}},
                    "frequency_penalty": {"type": "number"},
                    "presence_penalty": {"type": "number"}
                }
            },
            "selection": {
                "type": "object",
                "additionalProperties": false,
                "required": ["limit", "seed"],
                "properties": {
                    "limit": {"type": "integer", "minimum": 1, "maximum": 5},
                    "seed": {"const": 0}
                }
            }
        },
        "required": ["environment", "solver", "solver_config", "selection"],
        "type": "object"
    }))
    .expect("stock schema is canonical JSON")
}

fn openbench_schema() -> CanonicalJson {
    CanonicalJson::new(serde_json::json!({
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "additionalProperties": false,
        "properties": {
            "task": {"const": "gsm8k"},
            "task_args": {"type": "object", "additionalProperties": false},
            "epochs": {"type": "integer", "minimum": 1, "maximum": 8},
            "limit": {"type": "integer", "minimum": 1, "maximum": 5}
        },
        "required": ["task", "task_args", "epochs"],
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
    let operations = STOCK_EVALUATION_OPERATION_SCHEMAS
        .iter()
        .map(|schema| {
            let operation_id = SemanticOperationId::new(schema.operation_id)
                .map_err(|error| ProviderRegistryError::InvalidDescriptor(error.to_string()))?;
            Ok(EvaluationOperationDescriptor {
                operation_id,
                input_schema_sha256: schema.request_schema_sha256.to_string(),
                output_schema_sha256: schema.response_schema_sha256.to_string(),
                stream_schema_sha256: schema
                    .true_streaming
                    .then(|| schema.canonical_stream_schema_sha256.to_string()),
                reports_usage: true,
                modalities: schema
                    .modalities
                    .iter()
                    .map(|value| (*value).to_string())
                    .collect(),
                endpoint_capabilities: vec![schema.endpoint_capability.to_string()],
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
    /// A registered distribution lacks a verified launch closure or isolation mechanism.
    UnavailableDistribution {
        /// Selected provider.
        provider: String,
        /// Selected distribution.
        distribution: String,
        /// Redacted fail-closed availability reason.
        reason: String,
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
            Self::UnavailableDistribution {
                provider,
                distribution,
                reason,
            } => write!(
                formatter,
                "evaluator distribution {distribution} for provider {provider} is unavailable: {reason}"
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
        fn check_distribution_available(
            &self,
            _distribution: &EvaluationDistributionDescriptor,
        ) -> Result<(), EvaluationProviderError> {
            Ok(())
        }

        fn prepare_launch(
            &self,
            _descriptor: &EvaluationProviderDescriptor,
            _distribution: &EvaluationDistributionDescriptor,
            _context: ProviderLaunchContext,
        ) -> Result<Box<dyn PreparedEvaluationProviderLaunch>, EvaluationProviderError> {
            panic!("pure validation test must not prepare a provider")
        }

        async fn launch(
            &self,
            _descriptor: &EvaluationProviderDescriptor,
            _config: &ValidatedProviderConfig,
            _host_binding: crate::provider_protocol::EvaluationHostBinding,
            _prepared: Box<dyn PreparedEvaluationProviderLaunch>,
        ) -> Result<Box<dyn EvaluationProvider>, EvaluationProviderError> {
            panic!("pure validation test must not launch a provider")
        }
    }

    struct UnavailableLaunch;

    #[async_trait(?Send)]
    impl EvaluationProviderLauncher for UnavailableLaunch {
        fn check_distribution_available(
            &self,
            _distribution: &EvaluationDistributionDescriptor,
        ) -> Result<(), EvaluationProviderError> {
            Err(EvaluationProviderError::Launch(
                "fixture isolation mechanism unavailable".to_string(),
            ))
        }

        fn prepare_launch(
            &self,
            _descriptor: &EvaluationProviderDescriptor,
            _distribution: &EvaluationDistributionDescriptor,
            _context: ProviderLaunchContext,
        ) -> Result<Box<dyn PreparedEvaluationProviderLaunch>, EvaluationProviderError> {
            panic!("an unavailable distribution must never prepare")
        }

        async fn launch(
            &self,
            _descriptor: &EvaluationProviderDescriptor,
            _config: &ValidatedProviderConfig,
            _host_binding: crate::provider_protocol::EvaluationHostBinding,
            _prepared: Box<dyn PreparedEvaluationProviderLaunch>,
        ) -> Result<Box<dyn EvaluationProvider>, EvaluationProviderError> {
            panic!("an unavailable distribution must never launch")
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
            "environment": "gsm8k",
            "solver": "chat",
            "solver_config": {"max_tokens": 64},
            "selection": {"limit": 1, "seed": 0}
        }))
        .unwrap();
        let validated = factory.validate_authored_config(&config).unwrap();
        assert_eq!(validated.provider_id().as_str(), "nemo_evaluator");
        assert!(is_sha256(validated.schema_sha256()));
        assert!(is_sha256(validated.config_sha256()));

        let unknown = CanonicalJson::new(serde_json::json!({
            "environment": "gsm8k",
            "solver": "chat",
            "solver_config": {"max_tokens": 64},
            "selection": {"limit": 1, "seed": 0},
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
        let nemo_id = EvaluationProviderId::new("nemo_evaluator").unwrap();
        assert_eq!(registry.available_distributions(&nemo_id).len(), 1);
    }

    #[test]
    fn registry_excludes_and_rejects_unavailable_distributions() {
        let distribution = distribution("nemo-unavailable");
        let distribution_id = distribution.distribution_id.clone();
        let provider_id = EvaluationProviderId::new("nemo_evaluator").unwrap();
        let factory =
            NemoEvaluatorProviderFactory::new(vec![distribution], Arc::new(UnavailableLaunch))
                .unwrap();
        let mut builder = EvaluationProviderRegistryBuilder::new();
        builder.register(Arc::new(factory)).unwrap();
        let registry = builder.freeze().unwrap();
        assert!(!registry.distribution_is_available(&provider_id, &distribution_id));
        assert!(registry.available_distributions(&provider_id).is_empty());

        let config = CanonicalJson::new(serde_json::json!({"benchmark": "mmlu"})).unwrap();
        assert!(matches!(
            registry.validate_authored_config(&provider_id, &distribution_id, &config),
            Err(ProviderRegistryError::UnavailableDistribution {
                provider,
                distribution,
                reason,
            }) if provider == "nemo_evaluator"
                && distribution == "nemo-unavailable"
                && reason.contains("isolation mechanism unavailable")
        ));
    }

    #[test]
    fn stock_providers_share_exact_worker_schema_digests() {
        let nemo = NemoEvaluatorProviderFactory::new(
            vec![distribution("nemo-locked")],
            Arc::new(NeverLaunch),
        )
        .unwrap();
        let openbench = OpenBenchProviderFactory::new(
            vec![distribution("openbench-locked")],
            Arc::new(NeverLaunch),
        )
        .unwrap();
        assert_eq!(
            nemo.descriptor().config_schema_sha256,
            "b501baba9601933a8239e15b34fba57aa06ebaa6deb4d3132544ae5c5c9b47c4"
        );
        assert_eq!(
            openbench.descriptor().config_schema_sha256,
            "2c1a1a970a9695dc8d741096f2fc1b92cd57c7823b6b6c53b20f37e78f4da57b"
        );
        assert_eq!(
            nemo.descriptor().operations,
            openbench.descriptor().operations
        );
        assert_eq!(nemo.descriptor().operations.len(), 4);
        assert!(
            nemo.descriptor()
                .operations
                .iter()
                .all(|operation| operation.operation_id.as_str().starts_with("model."))
        );
        let embed = nemo
            .descriptor()
            .operations
            .iter()
            .find(|operation| operation.operation_id.as_str() == "model.embed")
            .unwrap();
        assert_eq!(
            embed.input_schema_sha256,
            "d46a1567b54ffeb888fbecbb451e8f2790b846e5928e339bf7afeedd4be34f86"
        );
        assert_eq!(
            embed.output_schema_sha256,
            "ddea46c3ffba13ff7b860ad0d6c6277c19a2f51b71adbc0cef76c0d8fd2c6bc1"
        );
        assert_eq!(embed.stream_schema_sha256, None);
        assert_eq!(
            STOCK_EVALUATION_OPERATION_SCHEMAS[3].canonical_stream_schema_sha256,
            "bcde375ebd4cbacf651311181173836b169d5a360c6ac158c6a2cdaf49be3f61"
        );
        assert_eq!(
            STOCK_EVALUATION_OPERATION_SCHEMAS[0].combined_schema_sha256,
            "d468bbc4f1fdbbc54360cede8194732b2ebaabbdfb55490bc572c4bb44f89cdf"
        );
    }

    #[test]
    fn factory_rejects_public_schema_without_executable_validator() {
        let stock = NemoEvaluatorProviderFactory::new(
            vec![distribution("nemo-locked")],
            Arc::new(NeverLaunch),
        )
        .unwrap();
        let mut descriptor = stock.descriptor().clone();
        descriptor
            .public_projection_schemas
            .insert("accuracy".to_string(), "a".repeat(64));
        assert!(
            RegisteredProviderFactory::new(
                descriptor,
                Arc::new(NemoConfigValidator),
                PublicScoreProjectionPolicy::restricted_only(),
                Arc::new(NeverLaunch),
            )
            .is_err()
        );
    }

    #[test]
    fn registry_rejects_cross_provider_schema_drift_for_same_operation_id() {
        let nemo: Arc<dyn EvaluationProviderFactory> = Arc::new(
            NemoEvaluatorProviderFactory::new(
                vec![distribution("nemo-locked")],
                Arc::new(NeverLaunch),
            )
            .unwrap(),
        );
        let openbench = OpenBenchProviderFactory::new(
            vec![distribution("openbench-locked")],
            Arc::new(NeverLaunch),
        )
        .unwrap();
        let mut descriptor = openbench.descriptor().clone();
        descriptor.provider_id = EvaluationProviderId::new("schema-drift-fixture").unwrap();
        descriptor.operations[0].input_schema_sha256 = "f".repeat(64);
        let conflicting: Arc<dyn EvaluationProviderFactory> = Arc::new(
            RegisteredProviderFactory::new(
                descriptor,
                Arc::new(OpenBenchConfigValidator),
                PublicScoreProjectionPolicy::restricted_only(),
                Arc::new(NeverLaunch),
            )
            .unwrap(),
        );
        let mut builder = EvaluationProviderRegistryBuilder::new();
        builder.register(nemo).unwrap();
        builder.register(conflicting).unwrap();
        let error = match builder.freeze() {
            Ok(_) => panic!("registry accepted cross-provider schema drift"),
            Err(error) => error,
        };
        assert!(matches!(
            error,
            ProviderRegistryError::ConflictingOperationSchema(operation)
                if operation == "model.generate"
        ));
    }
}
