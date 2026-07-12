// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Provider-neutral evaluator-worker protocol v2 data transfer objects.
//!
//! These types describe evaluation semantics without implementing them. They
//! deliberately contain no upstream URL, credential, HTTP method/path, model
//! client, benchmark prompt builder, hidden answer, or grader.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::{self, Display};
use std::path::PathBuf;

use serde::{Deserialize, Deserializer, Serialize, Serializer, de};
use serde_json::Value;

use crate::canonical::{
    CANONICAL_JSON_CODEC, CanonicalJson, is_sha256, redact_diagnostic,
    validate_no_secret_control_value, validate_no_secret_host_payload,
};

/// Evaluator-worker protocol version implemented by this module.
pub const EVALUATOR_WORKER_PROTOCOL_V2: u32 = 2;

const MAX_OPAQUE_ID_BYTES: usize = 256;
const MAX_OPEN_ID_BYTES: usize = 128;

macro_rules! opaque_id {
    ($name:ident, $wire:literal) => {
        #[doc = concat!("Validated opaque `", $wire, "` identity.")]
        #[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize)]
        #[serde(transparent)]
        pub struct $name(String);

        impl $name {
            #[doc = concat!("Construct a validated `", $wire, "`.")]
            pub fn new(value: impl Into<String>) -> Result<Self, EvaluationProtocolError> {
                let value = value.into();
                validate_opaque_id($wire, &value)?;
                Ok(Self(value))
            }

            /// Borrow the exact wire value.
            pub fn as_str(&self) -> &str {
                &self.0
            }
        }

        impl AsRef<str> for $name {
            fn as_ref(&self) -> &str {
                self.as_str()
            }
        }

        impl Display for $name {
            fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str(&self.0)
            }
        }

        impl<'de> Deserialize<'de> for $name {
            fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
            where
                D: Deserializer<'de>,
            {
                Self::new(String::deserialize(deserializer)?).map_err(de::Error::custom)
            }
        }
    };
}

macro_rules! open_id {
    ($name:ident, $wire:literal) => {
        #[doc = concat!("Validated open `", $wire, "` identity.")]
        #[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize)]
        #[serde(transparent)]
        pub struct $name(String);

        impl $name {
            #[doc = concat!("Construct a validated open `", $wire, "`.")]
            pub fn new(value: impl Into<String>) -> Result<Self, EvaluationProtocolError> {
                let value = value.into();
                validate_open_id($wire, &value)?;
                Ok(Self(value))
            }

            /// Borrow the exact wire value.
            pub fn as_str(&self) -> &str {
                &self.0
            }
        }

        impl AsRef<str> for $name {
            fn as_ref(&self) -> &str {
                self.as_str()
            }
        }

        impl Display for $name {
            fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str(&self.0)
            }
        }

        impl<'de> Deserialize<'de> for $name {
            fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
            where
                D: Deserializer<'de>,
            {
                Self::new(String::deserialize(deserializer)?).map_err(de::Error::custom)
            }
        }
    };
}

open_id!(EvaluationProviderId, "provider_id");
open_id!(EvaluationDistributionId, "distribution_id");
opaque_id!(EvaluationSessionId, "session_id");
opaque_id!(EvaluationUnitTemplateId, "unit_template_id");
opaque_id!(EvaluationUnitId, "unit_id");
opaque_id!(EvaluationCaseTemplateId, "case_template_id");
opaque_id!(EvaluationCaseId, "case_id");
opaque_id!(SemanticAttemptId, "semantic_attempt_id");
opaque_id!(LogicalCallId, "logical_call_id");
opaque_id!(HostOperationId, "host_operation_id");
open_id!(LogicalServiceId, "service_id");
open_id!(SemanticOperationId, "semantic_operation_id");
open_id!(HostCapabilityId, "host_capability_id");
open_id!(EvaluationArtifactId, "artifact_id");
open_id!(EvaluationPhaseId, "phase_id");
open_id!(OperationPurpose, "purpose");

/// Protocol-level validation error containing no provider secret.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvaluationProtocolError(String);

impl EvaluationProtocolError {
    /// Construct a redacted public protocol diagnostic.
    pub fn new(message: impl AsRef<str>) -> Self {
        Self(redact_diagnostic(message.as_ref()))
    }
}

impl Display for EvaluationProtocolError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for EvaluationProtocolError {}

/// Evaluator execution unit granularity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvaluationExecutionGranularity {
    /// Rust starts an independently cancellable case unit.
    Case,
    /// The provider owns one explicitly declared host batch.
    HostBatch,
}

/// How case occurrences are materialized.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvaluationSchedulingMode {
    /// Every occurrence is frozen during preparation.
    Finite,
    /// Rust deterministically requests scheduled occurrences.
    RustOccurrences,
}

/// Requested provider response delivery.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HostResponseMode {
    /// One normalized terminal result.
    Terminal,
    /// Typed deltas followed by exactly one terminal.
    Streaming,
}

/// Provider artifact visibility request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ArtifactVisibility {
    /// Restricted evaluator material; the default and authority floor.
    Restricted,
    /// Requested public projection, subject to a factory-owned allowlist/schema.
    PublicProjection,
}

/// Immutable identity reported by the worker during `hello`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvaluationWorkerIdentity {
    /// Negotiated evaluator-worker protocol version.
    pub evaluator_protocol: u32,
    /// Provider implementation selected by the factory.
    pub provider_id: EvaluationProviderId,
    /// Immutable registered distribution identity.
    pub distribution_id: EvaluationDistributionId,
    /// Provider package name.
    pub package: String,
    /// Exact provider package version.
    pub package_version: String,
    /// Provider source/commit digest.
    pub provider_source_sha256: String,
    /// Minimal worker bootstrap source digest.
    pub worker_source_sha256: String,
    /// Fully pinned dependency lock digest.
    pub dependency_lock_sha256: String,
    /// Python runtime version.
    pub python_version: String,
    /// Exact Rust-minted launch nonce echoed by the minimal bootstrap.
    pub launch_nonce: String,
    /// Optional immutable OCI image digest.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub oci_digest: Option<String>,
    /// Supported protocol operations.
    pub operations: Vec<String>,
}

impl EvaluationWorkerIdentity {
    /// Validate complete, immutable self-description before attestation comparison.
    pub fn validate(&self) -> Result<(), EvaluationProtocolError> {
        if self.evaluator_protocol != EVALUATOR_WORKER_PROTOCOL_V2 {
            return Err(EvaluationProtocolError::new(format!(
                "evaluator-worker protocol {} did not match {}",
                self.evaluator_protocol, EVALUATOR_WORKER_PROTOCOL_V2
            )));
        }
        for (name, value) in [
            ("package", self.package.as_str()),
            ("package_version", self.package_version.as_str()),
            ("python_version", self.python_version.as_str()),
            ("launch_nonce", self.launch_nonce.as_str()),
        ] {
            validate_nonempty_bounded(name, value, 512)?;
        }
        for (name, digest) in [
            (
                "provider_source_sha256",
                self.provider_source_sha256.as_str(),
            ),
            ("worker_source_sha256", self.worker_source_sha256.as_str()),
            (
                "dependency_lock_sha256",
                self.dependency_lock_sha256.as_str(),
            ),
        ] {
            if !is_sha256(digest) {
                return Err(EvaluationProtocolError::new(format!(
                    "worker identity {name} was not a lowercase SHA-256 digest"
                )));
            }
        }
        if let Some(digest) = &self.oci_digest
            && !digest.strip_prefix("sha256:").is_some_and(is_sha256)
        {
            return Err(EvaluationProtocolError::new(
                "worker identity oci_digest was not an immutable sha256 digest",
            ));
        }
        const REQUIRED: &[&str] = &[
            "plan_session",
            "bind_assets",
            "next_units",
            "instantiate_units",
            "start_units",
            "poll_events",
            "submit_host_events",
            "cancel_units",
            "finalize_session",
            "shutdown",
        ];
        let operations = self
            .operations
            .iter()
            .map(String::as_str)
            .collect::<BTreeSet<_>>();
        if operations.len() != self.operations.len() {
            return Err(EvaluationProtocolError::new(
                "worker identity contained duplicate operations",
            ));
        }
        let missing = REQUIRED
            .iter()
            .copied()
            .filter(|operation| !operations.contains(operation))
            .collect::<Vec<_>>();
        if !missing.is_empty() {
            return Err(EvaluationProtocolError::new(format!(
                "worker identity omitted operations: {}",
                missing.join(", ")
            )));
        }
        Ok(())
    }
}

/// Factory-validated plan request sent to the provider worker.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvaluationPlanRequest {
    /// Per-run evaluation session identity minted by Rust.
    pub session_id: EvaluationSessionId,
    /// Selected registered provider.
    pub provider_id: EvaluationProviderId,
    /// Selected immutable distribution.
    pub distribution_id: EvaluationDistributionId,
    /// Authored-config schema version.
    pub config_schema_version: u32,
    /// Factory-owned schema fingerprint.
    pub config_schema_sha256: String,
    /// Strict canonical provider-authored value.
    pub provider_config: CanonicalJson,
    /// Require every mutable/unknown identity to fail planning.
    pub reproducible: bool,
}

impl EvaluationPlanRequest {
    /// Enforce fingerprint and no-secret authored-control invariants.
    pub fn validate(&self) -> Result<(), EvaluationProtocolError> {
        if self.config_schema_version == 0 || !is_sha256(&self.config_schema_sha256) {
            return Err(EvaluationProtocolError::new(
                "provider config schema version/fingerprint was invalid",
            ));
        }
        validate_no_secret_control_value(&self.provider_config)
            .map_err(|error| EvaluationProtocolError::new(error.to_string()))
    }
}

/// Immutable asset requested during side-effect-free provider planning.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvaluationAssetRequirement {
    /// Provider-local logical asset name.
    pub asset_id: String,
    /// Source family such as `hugging_face`, `oci`, or `task_package`.
    pub source_kind: String,
    /// Immutable revision, commit, or OCI digest.
    pub immutable_revision: String,
    /// Required exact content digest.
    pub content_sha256: String,
    /// Expected media type.
    pub media_type: String,
    /// Visibility floor.
    pub visibility: ArtifactVisibility,
}

/// Rust-resolved read-only binding for one immutable asset.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResolvedEvaluationAsset {
    /// Logical asset identity from planning.
    pub asset_id: String,
    /// Contained worker-visible read-only path.
    pub contained_path: String,
    /// Rust-verified exact content digest.
    pub content_sha256: String,
    /// Exact immutable source revision.
    pub immutable_revision: String,
    /// Media type verified by the resolver.
    pub media_type: String,
}

/// One operation required on a logical model service.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LogicalServiceRequirement {
    /// Provider-selected logical service, never an endpoint URL.
    pub service_id: LogicalServiceId,
    /// Reporting purpose for this service.
    pub purpose: OperationPurpose,
    /// Semantic operations required by the provider.
    pub operations: Vec<SemanticOperationId>,
    /// Whether restricted inference bodies may transit this route.
    pub allows_restricted_payload: bool,
}

/// One typed host capability required by the planned session.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HostCapabilityRequirement {
    /// Open host-capability identity.
    pub capability_id: HostCapabilityId,
    /// Exact input/output schema fingerprint.
    pub schema_sha256: String,
    /// Whether planning must fail if unavailable.
    pub required: bool,
}

/// Bounded credit request negotiated at planning.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvaluationQueueCredits {
    /// Maximum active units.
    pub units: usize,
    /// Maximum globally outstanding host operations.
    pub host_operations: usize,
    /// Maximum outstanding operations for one unit.
    pub host_operations_per_unit: usize,
    /// Maximum buffered stream events.
    pub stream_events: usize,
    /// Maximum live sandbox handles.
    pub sandboxes: usize,
    /// Maximum live process handles.
    pub processes: usize,
    /// Maximum declared artifact count.
    pub artifacts: usize,
    /// Maximum total declared artifact bytes.
    pub artifact_bytes: u64,
}

impl EvaluationQueueCredits {
    /// Reject zero or internally inconsistent negotiated bounds.
    pub fn validate(&self) -> Result<(), EvaluationProtocolError> {
        if self.units == 0
            || self.host_operations == 0
            || self.host_operations_per_unit == 0
            || self.stream_events == 0
            || self.artifacts == 0
            || self.artifact_bytes == 0
            || self.host_operations_per_unit > self.host_operations
        {
            return Err(EvaluationProtocolError::new(
                "provider queue credits were zero or inconsistent",
            ));
        }
        Ok(())
    }
}

/// Frozen aggregation and failure semantics owned by the provider.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AggregationPolicy {
    /// Provider-owned policy name.
    pub policy_id: String,
    /// Whether infrastructure cases are excluded from semantic denominators.
    pub exclude_infrastructure: bool,
    /// Whether cancelled cases are excluded from semantic denominators.
    pub exclude_cancelled: bool,
    /// Provider-native repeat/epoch/reducer policy.
    pub definition: CanonicalJson,
}

/// Side-effect-free result of provider planning.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvaluationPlan {
    /// Immutable assets Rust must resolve before binding.
    pub assets: Vec<EvaluationAssetRequirement>,
    /// Typed host capabilities required by this exact plan.
    pub host_requirements: Vec<HostCapabilityRequirement>,
    /// Logical service and operation requirements.
    pub logical_services: Vec<LogicalServiceRequirement>,
    /// Provider-owned aggregation/failure policy.
    pub aggregation_policy: AggregationPolicy,
    /// Unit execution granularity.
    pub execution_granularity: EvaluationExecutionGranularity,
    /// Occurrence scheduling mode.
    pub scheduling_mode: EvaluationSchedulingMode,
    /// Exact finite unit count; absent for Rust-scheduled occurrences.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finite_unit_count: Option<usize>,
    /// Exact finite case count; absent for Rust-scheduled occurrences.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finite_case_count: Option<usize>,
    /// Provider-requested credits that become negotiated when Rust accepts the plan.
    ///
    /// Rust rejects a plan outside its configured ceilings rather than silently
    /// advertising a different bound to the worker.
    pub queue_credits: EvaluationQueueCredits,
}

impl EvaluationPlan {
    /// Validate unique requirements and bounded, immutable identities.
    pub fn validate(&self) -> Result<(), EvaluationProtocolError> {
        self.queue_credits.validate()?;
        match self.scheduling_mode {
            EvaluationSchedulingMode::Finite => {
                if self.finite_unit_count == Some(0)
                    || self.finite_case_count == Some(0)
                    || self.finite_unit_count.is_none()
                    || self.finite_case_count.is_none()
                {
                    return Err(EvaluationProtocolError::new(
                        "finite evaluation plan omitted nonzero unit/case counts",
                    ));
                }
            }
            EvaluationSchedulingMode::RustOccurrences => {
                if self.finite_unit_count.is_some() || self.finite_case_count.is_some() {
                    return Err(EvaluationProtocolError::new(
                        "Rust-occurrence plan reported finite unit/case counts",
                    ));
                }
            }
        }
        validate_nonempty_bounded(
            "aggregation policy",
            &self.aggregation_policy.policy_id,
            256,
        )?;
        let mut assets = BTreeSet::new();
        for asset in &self.assets {
            validate_nonempty_bounded("asset_id", &asset.asset_id, 256)?;
            validate_nonempty_bounded("immutable_revision", &asset.immutable_revision, 512)?;
            if !is_sha256(&asset.content_sha256) || !assets.insert(asset.asset_id.as_str()) {
                return Err(EvaluationProtocolError::new(
                    "asset requirements contained a bad digest or duplicate identity",
                ));
            }
        }
        let mut capabilities = BTreeSet::new();
        for capability in &self.host_requirements {
            if !is_sha256(&capability.schema_sha256)
                || !capabilities.insert(&capability.capability_id)
            {
                return Err(EvaluationProtocolError::new(
                    "host requirements contained a bad schema digest or duplicate identity",
                ));
            }
        }
        let mut services = BTreeSet::new();
        for service in &self.logical_services {
            if service.operations.is_empty() || !services.insert(&service.service_id) {
                return Err(EvaluationProtocolError::new(
                    "logical service requirements were empty or duplicated",
                ));
            }
            if service.operations.iter().collect::<BTreeSet<_>>().len() != service.operations.len()
            {
                return Err(EvaluationProtocolError::new(
                    "logical service contained duplicate operations",
                ));
            }
        }
        Ok(())
    }
}

/// Generic immutable identity component.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvaluationIdentityComponent {
    /// Component role/name.
    pub name: String,
    /// Exact version or immutable revision.
    pub version: String,
    /// Exact source/package digest.
    pub source_sha256: String,
}

/// Rust host and isolation identity bound into an evaluation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvaluationHostIdentity {
    /// Runner binary digest.
    pub runner_sha256: String,
    /// Host capability inventory digest.
    pub capability_inventory_sha256: String,
    /// Host schema inventory digest.
    pub schema_inventory_sha256: String,
    /// Enforced evaluator isolation proof digest.
    pub isolation_proof_sha256: String,
}

/// Rust-authored host/route identity supplied during immutable asset binding.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvaluationHostBinding {
    /// Rust runner/capability/schema/isolation identity.
    pub host: EvaluationHostIdentity,
    /// Secret-free logical route map digest.
    pub route_map_sha256: String,
    /// Prepared endpoint identity inventory digest.
    pub prepared_endpoints_sha256: String,
    /// Optional sandbox image/config digest.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sandbox_sha256: Option<String>,
}

impl EvaluationHostBinding {
    /// Validate every Rust-owned identity component is immutable.
    pub fn validate(&self) -> Result<(), EvaluationProtocolError> {
        for digest in [
            self.host.runner_sha256.as_str(),
            self.host.capability_inventory_sha256.as_str(),
            self.host.schema_inventory_sha256.as_str(),
            self.host.isolation_proof_sha256.as_str(),
            self.route_map_sha256.as_str(),
            self.prepared_endpoints_sha256.as_str(),
        ] {
            if !is_sha256(digest) {
                return Err(EvaluationProtocolError::new(
                    "Rust evaluation host binding contained a mutable or invalid digest",
                ));
            }
        }
        if self
            .sandbox_sha256
            .as_deref()
            .is_some_and(|digest| !is_sha256(digest))
        {
            return Err(EvaluationProtocolError::new(
                "Rust evaluation host binding sandbox digest was invalid",
            ));
        }
        Ok(())
    }

    /// Prove the provider-returned identity preserved every Rust-owned fact.
    pub fn matches(&self, identity: &EvaluationIdentity) -> bool {
        self.host == identity.host
            && self.route_map_sha256 == identity.route_map_sha256
            && self.prepared_endpoints_sha256 == identity.prepared_endpoints_sha256
            && self.sandbox_sha256 == identity.sandbox_sha256
    }
}

/// Complete frozen evaluation identity graph.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvaluationIdentity {
    /// Semantic JSON codec identity.
    pub canonical_json_codec: String,
    /// Attested worker identity.
    pub worker: EvaluationWorkerIdentity,
    /// Authored-config schema fingerprint.
    pub config_schema_sha256: String,
    /// Secret-redacted resolved provider configuration digest.
    pub resolved_config_sha256: String,
    /// Dataset/task/source component.
    pub dataset: EvaluationIdentityComponent,
    /// Provider/plugin/environment/solver/scorer components.
    pub components: Vec<EvaluationIdentityComponent>,
    /// Ordered case and unit manifest digest.
    pub ordered_manifest_sha256: String,
    /// Frozen ordered model-safe case templates.
    pub case_templates: Vec<EvaluationCaseTemplateDescriptor>,
    /// Frozen ordered execution-unit templates.
    pub unit_templates: Vec<EvaluationUnitTemplateDescriptor>,
    /// Provider policy identity graph.
    pub policies: CanonicalJson,
    /// Rust host identity.
    pub host: EvaluationHostIdentity,
    /// Secret-free logical route-map digest.
    pub route_map_sha256: String,
    /// Prepared endpoint identity digest.
    pub prepared_endpoints_sha256: String,
    /// Optional sandbox image/config digest when a sandbox is required.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sandbox_sha256: Option<String>,
}

impl EvaluationIdentity {
    /// Validate every required immutable digest in the identity graph.
    pub fn validate(&self) -> Result<(), EvaluationProtocolError> {
        if self.canonical_json_codec != CANONICAL_JSON_CODEC {
            return Err(EvaluationProtocolError::new(
                "evaluation identity used an unsupported canonical JSON codec",
            ));
        }
        self.worker.validate()?;
        for component in std::iter::once(&self.dataset).chain(self.components.iter()) {
            validate_nonempty_bounded("evaluation component name", &component.name, 512)?;
            validate_nonempty_bounded("evaluation component version", &component.version, 512)?;
        }
        validate_no_secret_control_value(&self.policies)
            .map_err(|error| EvaluationProtocolError::new(error.to_string()))?;
        for digest in [
            self.config_schema_sha256.as_str(),
            self.resolved_config_sha256.as_str(),
            self.dataset.source_sha256.as_str(),
            self.ordered_manifest_sha256.as_str(),
            self.host.runner_sha256.as_str(),
            self.host.capability_inventory_sha256.as_str(),
            self.host.schema_inventory_sha256.as_str(),
            self.host.isolation_proof_sha256.as_str(),
            self.route_map_sha256.as_str(),
            self.prepared_endpoints_sha256.as_str(),
        ] {
            if !is_sha256(digest) {
                return Err(EvaluationProtocolError::new(
                    "evaluation identity contained a mutable or invalid digest",
                ));
            }
        }
        if self
            .components
            .iter()
            .any(|component| !is_sha256(&component.source_sha256))
            || self
                .sandbox_sha256
                .as_deref()
                .is_some_and(|digest| !is_sha256(digest))
        {
            return Err(EvaluationProtocolError::new(
                "evaluation component identity contained an invalid digest",
            ));
        }
        let case_templates = self
            .case_templates
            .iter()
            .map(|template| &template.template_id)
            .collect::<BTreeSet<_>>();
        if case_templates.len() != self.case_templates.len()
            || self.case_templates.iter().any(|template| {
                !is_safe_reporting_label(&template.task)
                    || !is_safe_reporting_label(&template.source)
            })
        {
            return Err(EvaluationProtocolError::new(
                "evaluation identity contained duplicate or empty case templates",
            ));
        }
        let unit_templates = self
            .unit_templates
            .iter()
            .map(|template| &template.unit_template_id)
            .collect::<BTreeSet<_>>();
        if unit_templates.len() != self.unit_templates.len()
            || self.unit_templates.iter().any(|template| {
                template.case_template_ids.is_empty()
                    || template.scheduling_class.trim().is_empty()
                    || template
                        .case_template_ids
                        .iter()
                        .any(|case_id| !case_templates.contains(case_id))
                    || template
                        .case_template_ids
                        .iter()
                        .collect::<BTreeSet<_>>()
                        .len()
                        != template.case_template_ids.len()
            })
        {
            return Err(EvaluationProtocolError::new(
                "evaluation identity contained duplicate, empty, or dangling unit templates",
            ));
        }
        Ok(())
    }
}

/// Model-safe template descriptor; no prompt or truth material is permitted.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvaluationCaseTemplateDescriptor {
    /// Stable provider-owned template identity.
    pub template_id: EvaluationCaseTemplateId,
    /// Safe reporting task label.
    pub task: String,
    /// Safe reporting source label.
    pub source: String,
}

/// Frozen template for one independently scheduled unit or host batch.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvaluationUnitTemplateDescriptor {
    /// Stable provider-owned unit template identity.
    pub unit_template_id: EvaluationUnitTemplateId,
    /// Ordered case templates contained by this unit.
    pub case_template_ids: Vec<EvaluationCaseTemplateId>,
    /// Explicit execution granularity.
    pub granularity: EvaluationExecutionGranularity,
    /// Provider-owned scheduling class label.
    pub scheduling_class: String,
}

/// One concrete occurrence of a provider case template.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvaluationCaseOccurrenceDescriptor {
    /// Globally unique case occurrence identity.
    pub case_id: EvaluationCaseId,
    /// Parent frozen template.
    pub template_id: EvaluationCaseTemplateId,
    /// Rust global issue ordinal.
    pub issue_ordinal: u64,
    /// Rust phase identity.
    pub phase_id: EvaluationPhaseId,
    /// Rust scheduler cycle index.
    pub cycle_index: u64,
}

/// Rust-authored deterministic request to instantiate a scheduled unit.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvaluationUnitOccurrenceRequest {
    /// Frozen provider unit template.
    pub unit_template_id: EvaluationUnitTemplateId,
    /// Rust phase identity.
    pub phase_id: EvaluationPhaseId,
    /// Rust global issue ordinal.
    pub issue_ordinal: u64,
    /// Rust scheduler cycle index.
    pub cycle_index: u64,
}

/// One concrete provider execution unit occurrence.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvaluationUnitOccurrence {
    /// Globally unique unit occurrence identity.
    pub unit_id: EvaluationUnitId,
    /// Parent frozen template.
    pub unit_template_id: EvaluationUnitTemplateId,
    /// Ordered concrete cases.
    pub cases: Vec<EvaluationCaseOccurrenceDescriptor>,
}

/// Bounded ordered page of finite units.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvaluationUnitPage {
    /// Units in frozen canonical order.
    pub items: Vec<EvaluationUnitOccurrence>,
    /// Offset for the next request.
    pub next_offset: usize,
    /// Whether the finite plan is exhausted.
    pub done: bool,
}

/// Immutable correlation context for one provider-authored host operation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HostCallContext {
    /// Evaluation session.
    pub session_id: EvaluationSessionId,
    /// Parent execution unit.
    pub unit_id: EvaluationUnitId,
    /// Parent case occurrence.
    pub case_id: EvaluationCaseId,
    /// Provider semantic-attempt identity.
    pub semantic_attempt_id: SemanticAttemptId,
    /// Stable logical call identity across transport attempts.
    pub logical_call_id: LogicalCallId,
}

/// Disclosure scope attached to a restricted judge/verifier body.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RestrictedDisclosure {
    /// Sole logical service allowed to receive the body.
    pub service_id: LogicalServiceId,
    /// Sole purpose allowed to receive the body.
    pub purpose: OperationPurpose,
    /// Must remain false: content logging is forbidden.
    pub allow_content_logging: bool,
    /// Must remain false: caching is forbidden.
    pub allow_cache: bool,
    /// Must remain false: public hashing is forbidden.
    pub allow_public_hash: bool,
}

/// Sensitive inference body allowed only on its declared auxiliary route.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RestrictedInferencePayload {
    /// Restricted semantic body.
    pub body: CanonicalJson,
    /// Fail-closed route/purpose disclosure scope.
    pub disclosure: RestrictedDisclosure,
}

impl fmt::Debug for RestrictedInferencePayload {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RestrictedInferencePayload")
            .field("body", &"[REDACTED]")
            .field("disclosure", &self.disclosure)
            .finish()
    }
}

impl RestrictedInferencePayload {
    /// Enforce non-logging/non-cache disclosure and route confinement.
    pub fn validate_for(
        &self,
        service_id: &LogicalServiceId,
        purpose: &OperationPurpose,
    ) -> Result<(), EvaluationProtocolError> {
        if &self.disclosure.service_id != service_id
            || &self.disclosure.purpose != purpose
            || self.disclosure.allow_content_logging
            || self.disclosure.allow_cache
            || self.disclosure.allow_public_hash
        {
            return Err(EvaluationProtocolError::new(
                "restricted inference payload exceeded its declared route or handling scope",
            ));
        }
        Ok(())
    }
}

/// One typed host operation requested by a provider.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HostOperationRequest {
    /// Unique operation identity.
    pub operation_id: HostOperationId,
    /// Immutable evaluation/call correlation.
    pub context: HostCallContext,
    /// Logical Rust-owned route.
    pub service_id: LogicalServiceId,
    /// Reporting purpose.
    pub purpose: OperationPurpose,
    /// Registered semantic operation, never an HTTP method/path.
    pub semantic_operation_id: SemanticOperationId,
    /// Schema-validated ordinary request payload.
    pub payload: CanonicalJson,
    /// Optional restricted auxiliary inference body.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub restricted_payload: Option<RestrictedInferencePayload>,
    /// Terminal or true-streaming delivery.
    pub response_mode: HostResponseMode,
    /// Optional logical deadline in monotonic duration milliseconds.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub deadline_ms: Option<u64>,
    /// Provider idempotency hint bound to this logical operation.
    pub idempotency_key: String,
}

impl HostOperationRequest {
    /// Validate no-secret ordinary payload and restricted-route constraints.
    pub fn validate(&self) -> Result<(), EvaluationProtocolError> {
        validate_nonempty_bounded("host operation idempotency_key", &self.idempotency_key, 512)?;
        validate_no_secret_host_payload(&self.payload)
            .map_err(|error| EvaluationProtocolError::new(error.to_string()))?;
        if let Some(restricted) = &self.restricted_payload {
            validate_no_secret_host_payload(&restricted.body)
                .map_err(|error| EvaluationProtocolError::new(error.to_string()))?;
            restricted.validate_for(&self.service_id, &self.purpose)?;
        }
        Ok(())
    }
}

/// Provider request to cancel one queued or active host operation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HostOperationCancelRequest {
    /// Original operation identity.
    pub operation_id: HostOperationId,
    /// Original semantic attempt identity.
    pub semantic_attempt_id: SemanticAttemptId,
    /// Safe provider cancellation reason.
    pub reason: String,
}

/// Provider progress counters safe for diagnostics.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvaluationProgress {
    /// Units started.
    pub started_units: u64,
    /// Cases terminal.
    pub terminal_cases: u64,
    /// Host operations still outstanding.
    pub outstanding_host_operations: u64,
}

/// One evaluator semantic event.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum EvaluationEvent {
    /// Typed host work is ready for Rust admission.
    HostOperationRequested {
        /// Complete request.
        request: Box<HostOperationRequest>,
    },
    /// Provider asks Rust to cancel one operation.
    HostOperationCancelRequested {
        /// Correlated cancellation.
        request: HostOperationCancelRequest,
    },
    /// One case reached provider-semantic terminal.
    CaseTerminal {
        /// Typed outcome.
        outcome: Box<CaseOutcome>,
    },
    /// Non-secret progress update.
    Progress {
        /// Bounded counters.
        progress: EvaluationProgress,
    },
    /// Non-secret diagnostic message.
    Diagnostic {
        /// Severity label.
        level: String,
        /// Redacted diagnostic text.
        message: String,
    },
}

/// Monotonically ordered, idempotent evaluator event.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SequencedEvaluationEvent {
    /// Strictly increasing session event sequence.
    pub sequence: u64,
    /// Session-unique idempotency key.
    pub idempotency_key: String,
    /// Typed semantic event.
    pub event: EvaluationEvent,
}

/// Bounded long-poll result.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvaluationEventBatch {
    /// Ordered events, possibly empty after timeout.
    pub events: Vec<SequencedEvaluationEvent>,
    /// Sequence expected after this batch.
    pub next_sequence: u64,
    /// True only when every unit and operation is terminal.
    pub drained: bool,
    /// Current worker-side remaining credits.
    pub remaining_credits: EvaluationQueueCredits,
}

/// Token usage returned by a Rust-owned operation executor.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HostOperationUsage {
    /// Prompt/input tokens.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_tokens: Option<u64>,
    /// Completion/output tokens.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub completion_tokens: Option<u64>,
    /// Reasoning tokens.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_tokens: Option<u64>,
    /// Cached prompt tokens.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cached_tokens: Option<u64>,
}

/// Stage of an infrastructure error or cancellation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct EvaluationStage(String);

impl EvaluationStage {
    /// Construct a bounded stage label.
    pub fn new(value: impl Into<String>) -> Result<Self, EvaluationProtocolError> {
        let value = value.into();
        validate_open_id("evaluation stage", &value)?;
        Ok(Self(value))
    }

    /// Borrow the stage label.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl AsRef<str> for EvaluationStage {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

impl Display for EvaluationStage {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

/// Safe structured evaluation infrastructure error.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvaluationError {
    /// Semantic lifecycle stage.
    pub stage: EvaluationStage,
    /// Stable error category.
    pub error_kind: String,
    /// Whether provider policy permits a new semantic attempt.
    pub retryable: bool,
    /// Redacted bounded diagnostic.
    pub message: String,
}

impl EvaluationError {
    /// Construct an error whose message cannot echo known secret shapes.
    pub fn new(
        stage: EvaluationStage,
        error_kind: impl Into<String>,
        retryable: bool,
        message: impl AsRef<str>,
    ) -> Result<Self, EvaluationProtocolError> {
        let error_kind = error_kind.into();
        validate_open_id("evaluation error kind", &error_kind)?;
        Ok(Self {
            stage,
            error_kind,
            retryable,
            message: redact_diagnostic(message.as_ref()),
        })
    }

    /// Normalize a deserialized diagnostic before public use.
    pub fn validate_and_redact(&mut self) -> Result<(), EvaluationProtocolError> {
        validate_open_id("evaluation error kind", &self.error_kind)?;
        self.message = "Evaluator provider reported an infrastructure error".to_string();
        Ok(())
    }
}

/// Rust terminal disposition for one host operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HostOperationDisposition {
    /// Normal typed result.
    Completed,
    /// Transport/executor infrastructure failed.
    InfrastructureError,
    /// Rust cancelled queued or active work.
    Cancelled,
    /// A cancellation raced with an already recorded terminal.
    AlreadyTerminal,
}

/// Exactly one terminal result for a host operation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HostOperationTerminal {
    /// Correlated operation.
    pub operation_id: HostOperationId,
    /// Correlated semantic attempt.
    pub semantic_attempt_id: SemanticAttemptId,
    /// Terminal classification.
    pub disposition: HostOperationDisposition,
    /// Typed normalized result for completed operations.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub result: Option<CanonicalJson>,
    /// Structured infrastructure/cancellation error.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<EvaluationError>,
    /// Rust-authoritative usage.
    #[serde(default)]
    pub usage: HostOperationUsage,
    /// Whether any output was externally observed before terminal.
    pub observed_output: bool,
}

impl HostOperationTerminal {
    /// Enforce disjoint result/error axes.
    pub fn validate(&mut self) -> Result<(), EvaluationProtocolError> {
        match self.disposition {
            HostOperationDisposition::Completed => {
                if self.result.is_none() || self.error.is_some() {
                    return Err(EvaluationProtocolError::new(
                        "completed host operation must contain a result and no error",
                    ));
                }
            }
            HostOperationDisposition::InfrastructureError | HostOperationDisposition::Cancelled => {
                if self.result.is_some() || self.error.is_none() {
                    return Err(EvaluationProtocolError::new(
                        "non-completed host operation must contain one error and no result",
                    ));
                }
            }
            HostOperationDisposition::AlreadyTerminal => {
                if self.result.is_some() || self.error.is_some() {
                    return Err(EvaluationProtocolError::new(
                        "already-terminal acknowledgement cannot contain result or error",
                    ));
                }
            }
        }
        if let Some(error) = &mut self.error {
            error.validate_and_redact()?;
        }
        Ok(())
    }
}

/// Typed Rust-to-provider host event.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum HostOperationEvent {
    /// One normalized streaming delta.
    StreamDelta {
        /// Correlated operation.
        operation_id: HostOperationId,
        /// Strictly increasing per-operation stream sequence.
        stream_sequence: u64,
        /// Registered typed delta.
        delta: CanonicalJson,
    },
    /// One terminal usage update before the terminal event.
    Usage {
        /// Correlated operation.
        operation_id: HostOperationId,
        /// Rust-authoritative usage.
        usage: HostOperationUsage,
    },
    /// Exactly one terminal event.
    Terminal {
        /// Terminal result.
        terminal: HostOperationTerminal,
    },
    /// Idempotent acknowledgement when cancellation raced with an existing terminal.
    CancellationAcknowledged {
        /// Correlated operation.
        operation_id: HostOperationId,
        /// Correlated semantic attempt.
        semantic_attempt_id: SemanticAttemptId,
        /// True when Rust had already recorded the sole operation terminal.
        already_terminal: bool,
    },
}

impl HostOperationEvent {
    /// Borrow the correlated host-operation identity.
    pub fn operation_id(&self) -> &HostOperationId {
        match self {
            Self::StreamDelta { operation_id, .. }
            | Self::Usage { operation_id, .. }
            | Self::CancellationAcknowledged { operation_id, .. } => operation_id,
            Self::Terminal { terminal } => &terminal.operation_id,
        }
    }

    /// Return whether this event is terminal.
    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::Terminal { .. })
    }
}

/// Provider-native score value plus optional reviewed public projection.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderScore {
    /// Complete native score algebra, restricted by default.
    pub value: CanonicalJson,
    /// Factory-schema-owned safe public projection when registered.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub public_projection: Option<CanonicalJson>,
}

/// Reference to one provider-declared artifact.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArtifactRef {
    /// Manifest artifact identity.
    pub artifact_id: EvaluationArtifactId,
    /// Contained relative path.
    pub path: String,
    /// Visibility floor/request.
    pub visibility: ArtifactVisibility,
}

/// Completed case semantics.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CompletedCaseOutcome {
    /// Named provider score trees.
    pub scores: BTreeMap<String, ProviderScore>,
    /// Separate finite scalar projection for score/performance joins.
    pub numeric_metrics: BTreeMap<String, FiniteF64>,
    /// Provider-selected headline score name.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub primary_score: Option<String>,
    /// Restricted provider annotations.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub annotations: Option<CanonicalJson>,
}

/// Provider-semantic terminal classification for one case.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum CaseOutcomeKind {
    /// Provider scoring completed; a valid zero remains completed.
    Completed {
        /// Complete score payload.
        completed: CompletedCaseOutcome,
    },
    /// Evaluator/host infrastructure prevented semantic completion.
    InfrastructureError {
        /// Structured error without scores.
        error: EvaluationError,
    },
    /// Rust/provider cancellation prevented semantic completion.
    Cancelled {
        /// Cancellation stage.
        stage: EvaluationStage,
        /// Redacted bounded reason.
        reason: String,
    },
}

/// Exactly one terminal outcome for one case occurrence.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CaseOutcome {
    /// Correlated concrete case occurrence.
    pub case_id: EvaluationCaseId,
    /// Disjoint terminal outcome.
    pub outcome: CaseOutcomeKind,
    /// Restricted-by-default provider artifacts.
    #[serde(default)]
    pub artifact_refs: Vec<ArtifactRef>,
}

impl CaseOutcome {
    /// Validate score names, headline identity, finite metrics, and redaction.
    pub fn validate(&mut self) -> Result<(), EvaluationProtocolError> {
        match &mut self.outcome {
            CaseOutcomeKind::Completed { completed } => {
                if completed.scores.is_empty() {
                    return Err(EvaluationProtocolError::new(
                        "completed case outcome contained no scores",
                    ));
                }
                for name in completed
                    .scores
                    .keys()
                    .chain(completed.numeric_metrics.keys())
                {
                    validate_open_id("score name", name)?;
                }
                if let Some(primary) = &completed.primary_score
                    && !completed.scores.contains_key(primary)
                {
                    return Err(EvaluationProtocolError::new(
                        "completed case primary_score was absent from scores",
                    ));
                }
            }
            CaseOutcomeKind::InfrastructureError { error } => error.validate_and_redact()?,
            CaseOutcomeKind::Cancelled { reason, .. } => {
                *reason = redact_diagnostic(reason);
            }
        }
        if self
            .artifact_refs
            .iter()
            .map(|item| &item.artifact_id)
            .collect::<BTreeSet<_>>()
            .len()
            != self.artifact_refs.len()
        {
            return Err(EvaluationProtocolError::new(
                "case outcome contained duplicate artifact references",
            ));
        }
        Ok(())
    }
}

/// Finite IEEE-754 value safe to serialize in a public numeric projection.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct FiniteF64(f64);

impl FiniteF64 {
    /// Construct a finite value, normalizing negative zero to zero.
    pub fn new(value: f64) -> Result<Self, EvaluationProtocolError> {
        if !value.is_finite() {
            return Err(EvaluationProtocolError::new(
                "public evaluation numeric value must be finite",
            ));
        }
        Ok(Self(if value == 0.0 { 0.0 } else { value }))
    }

    /// Return the finite numeric value.
    pub fn get(self) -> f64 {
        self.0
    }
}

impl Eq for FiniteF64 {}

impl Serialize for FiniteF64 {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_f64(self.0)
    }
}

impl<'de> Deserialize<'de> for FiniteF64 {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        Self::new(f64::deserialize(deserializer)?).map_err(de::Error::custom)
    }
}

/// Provider-authored aggregate with a finite public value.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AggregateMetric {
    /// Provider scorer identity.
    pub scorer: String,
    /// Provider reducer identity.
    pub reducer: String,
    /// Metric name.
    pub metric: String,
    /// Finite aggregate value.
    pub value: FiniteF64,
    /// Cases included in this aggregate.
    pub scored_count: u64,
    /// Cases excluded/unscored by provider policy.
    pub unscored_count: u64,
    /// Canonical provider definition.
    pub definition: CanonicalJson,
}

/// One declared provider artifact candidate.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvaluationArtifactManifestEntry {
    /// Stable artifact identity.
    pub artifact_id: EvaluationArtifactId,
    /// Contained relative path below the staging root.
    pub path: String,
    /// Media type.
    pub media_type: String,
    /// Provider visibility request; Rust may only narrow it.
    pub visibility: ArtifactVisibility,
    /// Claimed exact byte length.
    pub size_bytes: u64,
    /// Claimed exact raw artifact digest.
    pub artifact_content_sha256: String,
}

/// Manifest candidate returned before worker quiescence and Rust sealing.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvaluationFinishCandidate {
    /// Complete frozen evaluation identity.
    pub identity: EvaluationIdentity,
    /// Ordered case outcomes in canonical order.
    pub outcomes: Vec<CaseOutcome>,
    /// Provider-native canonical aggregate projections.
    pub aggregates: Vec<AggregateMetric>,
    /// Every declared staged artifact.
    pub artifacts: Vec<EvaluationArtifactManifestEntry>,
    /// Canonical provider bundle artifact.
    pub provider_bundle: ArtifactRef,
    /// Digest of the provider-normalized semantic result domain.
    pub normalized_result_sha256: String,
}

impl EvaluationFinishCandidate {
    /// Validate terminal uniqueness, immutable identity, and manifest claims.
    pub fn validate(&mut self) -> Result<(), EvaluationProtocolError> {
        self.identity.validate()?;
        if !is_sha256(&self.normalized_result_sha256) {
            return Err(EvaluationProtocolError::new(
                "finish candidate normalized_result_sha256 was invalid",
            ));
        }
        let mut case_ids = BTreeSet::new();
        for outcome in &mut self.outcomes {
            if !case_ids.insert(outcome.case_id.clone()) {
                return Err(EvaluationProtocolError::new(
                    "finish candidate contained duplicate case terminals",
                ));
            }
            outcome.validate()?;
        }
        let mut artifact_ids = BTreeSet::new();
        let mut paths = BTreeSet::new();
        for artifact in &self.artifacts {
            if !is_sha256(&artifact.artifact_content_sha256)
                || !artifact_ids.insert(&artifact.artifact_id)
                || !paths.insert(artifact.path.as_str())
            {
                return Err(EvaluationProtocolError::new(
                    "finish artifact manifest contained an empty, duplicate, or invalid entry",
                ));
            }
            validate_relative_artifact_path(&artifact.path)?;
        }
        if !artifact_ids.contains(&self.provider_bundle.artifact_id) {
            return Err(EvaluationProtocolError::new(
                "provider bundle was absent from the artifact manifest",
            ));
        }
        Ok(())
    }
}

/// Secret capability material serialized only to the isolated worker.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ScopedProxySecret(String);

impl ScopedProxySecret {
    /// Construct sufficiently strong Rust-minted proxy capability material.
    pub fn new(value: impl Into<String>) -> Result<Self, EvaluationProtocolError> {
        let value = value.into();
        if value.len() < 32 || value.len() > 512 || value.chars().any(char::is_whitespace) {
            return Err(EvaluationProtocolError::new(
                "scoped proxy secret did not meet length/shape requirements",
            ));
        }
        Ok(Self(value))
    }

    /// Expose capability material only to the proxy-binding serializer/caller.
    pub fn expose_secret(&self) -> &str {
        &self.0
    }
}

impl fmt::Debug for ScopedProxySecret {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("ScopedProxySecret([REDACTED])")
    }
}

/// Rust-minted scoped compatibility-proxy capability.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScopedProxyGrant {
    /// Opaque grant identity.
    pub grant_id: String,
    /// Exact evaluation session bound before worker launch.
    pub session_id: EvaluationSessionId,
    /// Secret bearer capability checked only by Rust proxy ingress.
    pub secret: ScopedProxySecret,
    /// Exact allowed logical services.
    pub service_ids: Vec<LogicalServiceId>,
    /// Exact allowed semantic operations.
    pub semantic_operation_ids: Vec<SemanticOperationId>,
    /// Exact allowed semantic purposes.
    pub purposes: Vec<OperationPurpose>,
    /// Exact evaluator process-subtree scope digest.
    pub process_scope_sha256: String,
    /// Maximum logical operations accepted by the grant.
    pub max_operations: u64,
    /// Maximum simultaneously active logical operations.
    pub max_concurrent_operations: u64,
    /// Maximum aggregate local request bytes.
    pub max_request_bytes: u64,
    /// Maximum aggregate normalized response bytes.
    pub max_response_bytes: u64,
    /// Maximum aggregate normalized stream events.
    pub max_stream_events: u64,
    /// Monotonic grant lifetime after worker launch.
    pub expires_after_ms: u64,
}

impl ScopedProxyGrant {
    /// Reject empty, duplicated, inconsistent, or unscoped capability fields.
    pub fn validate(&self) -> Result<(), EvaluationProtocolError> {
        validate_opaque_id("scoped proxy grant_id", &self.grant_id)?;
        if self.service_ids.is_empty()
            || self.semantic_operation_ids.is_empty()
            || self.purposes.is_empty()
            || self.service_ids.iter().collect::<BTreeSet<_>>().len() != self.service_ids.len()
            || self
                .semantic_operation_ids
                .iter()
                .collect::<BTreeSet<_>>()
                .len()
                != self.semantic_operation_ids.len()
            || self.purposes.iter().collect::<BTreeSet<_>>().len() != self.purposes.len()
            || self.max_operations == 0
            || self.max_concurrent_operations == 0
            || self.max_concurrent_operations > self.max_operations
            || self.max_request_bytes == 0
            || self.max_response_bytes == 0
            || self.max_stream_events == 0
            || self.expires_after_ms == 0
            || !is_sha256(&self.process_scope_sha256)
        {
            return Err(EvaluationProtocolError::new(
                "scoped proxy grant was not unique, bounded, and process-scoped",
            ));
        }
        Ok(())
    }

    /// Prove this grant only reduces route and numeric authority from one exact ceiling.
    pub fn validate_narrowing_of(&self, ceiling: &Self) -> Result<(), EvaluationProtocolError> {
        self.validate()?;
        ceiling.validate()?;
        if self.grant_id != ceiling.grant_id
            || self.session_id != ceiling.session_id
            || self.secret != ceiling.secret
            || !self
                .service_ids
                .iter()
                .all(|value| ceiling.service_ids.contains(value))
            || !self
                .semantic_operation_ids
                .iter()
                .all(|value| ceiling.semantic_operation_ids.contains(value))
            || !self
                .purposes
                .iter()
                .all(|value| ceiling.purposes.contains(value))
            || self.process_scope_sha256 != ceiling.process_scope_sha256
            || self.max_operations > ceiling.max_operations
            || self.max_concurrent_operations > ceiling.max_concurrent_operations
            || self.max_request_bytes > ceiling.max_request_bytes
            || self.max_response_bytes > ceiling.max_response_bytes
            || self.max_stream_events > ceiling.max_stream_events
            || self.expires_after_ms > ceiling.expires_after_ms
        {
            return Err(EvaluationProtocolError::new(
                "scoped proxy grant changed identity or expanded its registered ceiling",
            ));
        }
        Ok(())
    }
}

/// Runner-to-worker local compatibility proxy binding.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScopedProxyBinding {
    /// Fixed worker-visible Unix-domain locator inside the isolated rootfs.
    pub local_locator: String,
    /// Host-side Unix socket bind source, never serialized to the worker.
    #[serde(skip, default)]
    pub host_socket_path: PathBuf,
    /// Narrow per-run grant.
    pub grant: ScopedProxyGrant,
}

impl ScopedProxyBinding {
    /// Reject non-loopback locators and empty/unbounded grants.
    pub fn validate(&self) -> Result<(), EvaluationProtocolError> {
        self.grant.validate()?;
        let worker_socket = self.local_locator == "unix:///run/aiperf/evaluator-proxy.sock";
        if !worker_socket
            || !self.host_socket_path.is_absolute()
            || self.host_socket_path.components().any(|component| {
                matches!(
                    component,
                    std::path::Component::ParentDir | std::path::Component::CurDir
                )
            })
        {
            return Err(EvaluationProtocolError::new(
                "scoped proxy binding was not loopback-only, bounded, and process-scoped",
            ));
        }
        Ok(())
    }
}

/// Strict request envelope emitted by Rust on the dedicated control descriptor.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub(crate) enum EvaluatorWorkerRequest {
    Hello {
        id: u64,
        protocol: u32,
        max_message_bytes: usize,
        max_collection_items: usize,
        launch_nonce: String,
    },
    PlanSession {
        id: u64,
        request: EvaluationPlanRequest,
    },
    BindAssets {
        id: u64,
        assets: Vec<ResolvedEvaluationAsset>,
        #[serde(skip_serializing_if = "Option::is_none")]
        proxy: Option<ScopedProxyBinding>,
        host_binding: Box<EvaluationHostBinding>,
    },
    NextUnits {
        id: u64,
        offset: usize,
        limit: usize,
    },
    InstantiateUnits {
        id: u64,
        requests: Vec<EvaluationUnitOccurrenceRequest>,
    },
    StartUnits {
        id: u64,
        unit_ids: Vec<EvaluationUnitId>,
    },
    PollEvents {
        id: u64,
        limit: usize,
        wait_ms: u64,
    },
    SubmitHostEvents {
        id: u64,
        events: Vec<HostOperationEvent>,
    },
    CancelUnits {
        id: u64,
        unit_ids: Vec<EvaluationUnitId>,
    },
    FinalizeSession {
        id: u64,
    },
    Shutdown {
        id: u64,
    },
}

impl EvaluatorWorkerRequest {
    pub(crate) fn id(&self) -> u64 {
        match self {
            Self::Hello { id, .. }
            | Self::PlanSession { id, .. }
            | Self::BindAssets { id, .. }
            | Self::NextUnits { id, .. }
            | Self::InstantiateUnits { id, .. }
            | Self::StartUnits { id, .. }
            | Self::PollEvents { id, .. }
            | Self::SubmitHostEvents { id, .. }
            | Self::CancelUnits { id, .. }
            | Self::FinalizeSession { id }
            | Self::Shutdown { id } => *id,
        }
    }
}

/// Strict worker reply envelope.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct EvaluatorWorkerResponse {
    pub(crate) id: u64,
    pub(crate) ok: bool,
    #[serde(default)]
    pub(crate) result: Option<Value>,
    #[serde(default)]
    pub(crate) error: Option<EvaluationError>,
}

/// `bind_assets` acknowledgement.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct BindAssetsResult {
    pub(crate) identity: EvaluationIdentity,
}

/// `start_units` acknowledgement.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct StartedUnitsResult {
    pub(crate) started: Vec<EvaluationUnitId>,
}

/// `submit_host_events` acknowledgement.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct SubmittedHostEventsResult {
    pub(crate) accepted: Vec<HostOperationId>,
}

/// `cancel_units` acknowledgement.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct CancelledUnitsResult {
    pub(crate) cancelled: Vec<EvaluationUnitId>,
}

/// `shutdown` acknowledgement.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ProviderShutdownResult {
    pub(crate) shutdown: bool,
}

fn validate_opaque_id(field: &str, value: &str) -> Result<(), EvaluationProtocolError> {
    validate_nonempty_bounded(field, value, MAX_OPAQUE_ID_BYTES)?;
    if value != value.trim() || value.chars().any(char::is_control) {
        return Err(EvaluationProtocolError::new(format!(
            "{field} contained whitespace padding or a control character"
        )));
    }
    Ok(())
}

fn validate_open_id(field: &str, value: &str) -> Result<(), EvaluationProtocolError> {
    validate_nonempty_bounded(field, value, MAX_OPEN_ID_BYTES)?;
    let mut bytes = value.bytes();
    let Some(first) = bytes.next() else {
        return Err(EvaluationProtocolError::new(format!("{field} was empty")));
    };
    if !first.is_ascii_lowercase() && !first.is_ascii_digit() {
        return Err(EvaluationProtocolError::new(format!(
            "{field} must begin with a lowercase ASCII letter or digit"
        )));
    }
    if bytes.any(|byte| {
        !byte.is_ascii_lowercase()
            && !byte.is_ascii_digit()
            && byte != b'_'
            && byte != b'-'
            && byte != b'.'
    }) {
        return Err(EvaluationProtocolError::new(format!(
            "{field} contained a character outside [a-z0-9_.-]"
        )));
    }
    Ok(())
}

fn validate_nonempty_bounded(
    field: &str,
    value: &str,
    max_bytes: usize,
) -> Result<(), EvaluationProtocolError> {
    if value.trim().is_empty() || value.len() > max_bytes {
        return Err(EvaluationProtocolError::new(format!(
            "{field} was empty or exceeded {max_bytes} UTF-8 bytes"
        )));
    }
    Ok(())
}

fn validate_relative_artifact_path(path: &str) -> Result<(), EvaluationProtocolError> {
    use std::path::{Component, Path};

    if path.is_empty() || path.len() > 4_096 || path.contains('\0') {
        return Err(EvaluationProtocolError::new(
            "artifact path was empty or exceeded bounds",
        ));
    }
    let parsed = Path::new(path);
    if parsed.is_absolute()
        || parsed
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err(EvaluationProtocolError::new(
            "artifact path was absolute or contained traversal",
        ));
    }
    Ok(())
}

fn is_safe_reporting_label(value: &str) -> bool {
    !value.trim().is_empty()
        && value.len() <= 512
        && value == value.trim()
        && !value.chars().any(char::is_control)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids_are_open_but_bounded_and_map_friendly() {
        let id = SemanticOperationId::new("model.responses-v2").unwrap();
        assert_eq!(id.as_str(), "model.responses-v2");
        assert_eq!(id.to_string(), "model.responses-v2");
        assert_eq!(id.as_ref(), "model.responses-v2");
        assert!(SemanticOperationId::new("POST /v1/chat").is_err());
        assert!(EvaluationCaseId::new(" ").is_err());
    }

    #[test]
    fn restricted_payload_is_route_confined_and_debug_redacted() {
        let service = LogicalServiceId::new("judge").unwrap();
        let purpose = OperationPurpose::new("verifier").unwrap();
        let restricted = RestrictedInferencePayload {
            body: CanonicalJson::new(serde_json::json!({"reference": "hidden-sentinel"})).unwrap(),
            disclosure: RestrictedDisclosure {
                service_id: service.clone(),
                purpose: purpose.clone(),
                allow_content_logging: false,
                allow_cache: false,
                allow_public_hash: false,
            },
        };
        restricted.validate_for(&service, &purpose).unwrap();
        assert!(!format!("{restricted:?}").contains("hidden-sentinel"));
        assert!(
            restricted
                .validate_for(&LogicalServiceId::new("primary").unwrap(), &purpose)
                .is_err()
        );

        let mut request = HostOperationRequest {
            operation_id: HostOperationId::new("operation-1").unwrap(),
            context: HostCallContext {
                session_id: EvaluationSessionId::new("session-1").unwrap(),
                unit_id: EvaluationUnitId::new("unit-1").unwrap(),
                case_id: EvaluationCaseId::new("case-1").unwrap(),
                semantic_attempt_id: SemanticAttemptId::new("attempt-1").unwrap(),
                logical_call_id: LogicalCallId::new("call-1").unwrap(),
            },
            service_id: service,
            purpose,
            semantic_operation_id: SemanticOperationId::new("model.generate").unwrap(),
            payload: CanonicalJson::new(serde_json::json!({})).unwrap(),
            restricted_payload: Some(restricted),
            response_mode: HostResponseMode::Terminal,
            deadline_ms: None,
            idempotency_key: "idempotency-1".to_string(),
        };
        request.validate().unwrap();
        request.restricted_payload.as_mut().unwrap().body =
            CanonicalJson::new(serde_json::json!({"upstream_url": "https://hidden.invalid"}))
                .unwrap();
        assert!(request.validate().is_err());
    }

    #[test]
    fn completed_zero_and_infrastructure_are_disjoint() {
        let mut completed = CaseOutcome {
            case_id: EvaluationCaseId::new("case-1").unwrap(),
            outcome: CaseOutcomeKind::Completed {
                completed: CompletedCaseOutcome {
                    scores: BTreeMap::from([(
                        "accuracy".to_string(),
                        ProviderScore {
                            value: CanonicalJson::new(serde_json::json!(0)).unwrap(),
                            public_projection: Some(
                                CanonicalJson::new(serde_json::json!(0)).unwrap(),
                            ),
                        },
                    )]),
                    numeric_metrics: BTreeMap::from([(
                        "accuracy".to_string(),
                        FiniteF64::new(0.0).unwrap(),
                    )]),
                    primary_score: Some("accuracy".to_string()),
                    annotations: None,
                },
            },
            artifact_refs: Vec::new(),
        };
        completed.validate().unwrap();

        let encoded = serde_json::to_value(&completed).unwrap();
        assert_eq!(encoded["outcome"]["kind"], "completed");
        assert_eq!(
            encoded["outcome"]["completed"]["numeric_metrics"]["accuracy"],
            0.0
        );
    }

    #[test]
    fn proxy_binding_rejects_real_endpoint_and_redacts_secret_debug() {
        let binding = ScopedProxyBinding {
            local_locator: "https://model.example/v1".to_string(),
            host_socket_path: PathBuf::from("/run/aiperf-host/proxy.sock"),
            grant: ScopedProxyGrant {
                grant_id: "grant-1".to_string(),
                session_id: EvaluationSessionId::new("session-1").unwrap(),
                secret: ScopedProxySecret::new("x".repeat(32)).unwrap(),
                service_ids: vec![LogicalServiceId::new("primary").unwrap()],
                semantic_operation_ids: vec![SemanticOperationId::new("model.generate").unwrap()],
                purposes: vec![OperationPurpose::new("primary").unwrap()],
                process_scope_sha256: "a".repeat(64),
                max_operations: 1,
                max_concurrent_operations: 1,
                max_request_bytes: 1024,
                max_response_bytes: 2048,
                max_stream_events: 8,
                expires_after_ms: 1000,
            },
        };
        assert!(binding.validate().is_err());
        assert!(!format!("{:?}", binding.grant.secret).contains(&"x".repeat(32)));
    }

    #[test]
    fn scoped_proxy_grant_only_accepts_identity_preserving_reductions() {
        let ceiling = ScopedProxyGrant {
            grant_id: "grant-1".to_string(),
            session_id: EvaluationSessionId::new("session-1").unwrap(),
            secret: ScopedProxySecret::new("x".repeat(32)).unwrap(),
            service_ids: vec![
                LogicalServiceId::new("primary").unwrap(),
                LogicalServiceId::new("judge").unwrap(),
            ],
            semantic_operation_ids: vec![
                SemanticOperationId::new("model.generate").unwrap(),
                SemanticOperationId::new("model.responses").unwrap(),
            ],
            purposes: vec![
                OperationPurpose::new("primary").unwrap(),
                OperationPurpose::new("judge").unwrap(),
            ],
            process_scope_sha256: "a".repeat(64),
            max_operations: 40,
            max_concurrent_operations: 40,
            max_request_bytes: 40 * 1024,
            max_response_bytes: 40 * 2048,
            max_stream_events: 40,
            expires_after_ms: 1000,
        };
        let mut narrowed = ceiling.clone();
        narrowed.max_operations = 1;
        narrowed.max_concurrent_operations = 1;
        narrowed.max_request_bytes = 1024;
        narrowed.max_response_bytes = 2048;
        narrowed.max_stream_events = 1;
        narrowed.service_ids.truncate(1);
        narrowed.semantic_operation_ids.truncate(1);
        narrowed.purposes.truncate(1);
        narrowed.validate_narrowing_of(&ceiling).unwrap();

        let mut expanded = narrowed.clone();
        expanded.max_operations = 41;
        assert!(expanded.validate_narrowing_of(&ceiling).is_err());
        let mut changed_identity = narrowed;
        changed_identity.grant_id = "grant-2".to_string();
        assert!(changed_identity.validate_narrowing_of(&ceiling).is_err());
        let mut broadened_route = ceiling.clone();
        broadened_route.service_ids = vec![LogicalServiceId::new("unregistered").unwrap()];
        assert!(broadened_route.validate_narrowing_of(&ceiling).is_err());
    }

    #[test]
    fn finish_candidate_rejects_duplicate_cases_and_traversal() {
        assert!(validate_relative_artifact_path("../private").is_err());
        assert!(validate_relative_artifact_path("bundle/result.json").is_ok());
    }

    #[test]
    fn provider_authored_error_message_is_always_replaced_not_marker_redacted() {
        let mut error = EvaluationError::new(
            EvaluationStage::new("provider").unwrap(),
            "provider_error",
            false,
            "EXPECTED_ANSWER_SENTINEL arbitrary private verifier state",
        )
        .unwrap();
        error.validate_and_redact().unwrap();
        assert_eq!(
            error.message,
            "Evaluator provider reported an infrastructure error"
        );
        assert!(!error.message.contains("EXPECTED_ANSWER_SENTINEL"));
    }
}
