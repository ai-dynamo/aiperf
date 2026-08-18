// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

//! Schema-1.1 NativeGraph package parsing and immutable identity material.

use std::{
    collections::BTreeSet,
    num::NonZeroU64,
    path::{Component, Path},
    sync::Arc,
};

use serde::{Deserialize, Deserializer, Serialize};
use url::Url;

use crate::{
    endpoints::EndpointId,
    eval::{AcquiredSource, ArtifactDigest, HarborImportError, append_identity_field},
};

macro_rules! native_graph_id {
    ($name:ident, $description:literal) => {
        #[doc = $description]
        #[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
        pub struct $name(String);

        impl $name {
            /// Borrows the validated identifier.
            pub fn as_str(&self) -> &str {
                &self.0
            }

            fn parse(value: String) -> Result<Self, String> {
                if value.trim() != value
                    || value.is_empty()
                    || !value.bytes().enumerate().all(|(index, byte)| {
                        if index == 0 {
                            byte.is_ascii_lowercase()
                        } else {
                            byte.is_ascii_lowercase()
                                || byte.is_ascii_digit()
                                || matches!(byte, b'-' | b'_')
                        }
                    })
                {
                    return Err(format!("invalid {} {:?}", stringify!($name), value));
                }
                Ok(Self(value))
            }
        }

        impl<'de> Deserialize<'de> for $name {
            fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
            where
                D: Deserializer<'de>,
            {
                Self::parse(String::deserialize(deserializer)?).map_err(serde::de::Error::custom)
            }
        }
    };
}

native_graph_id!(
    AdapterId,
    "Identifier for one supervised NativeGraph adapter."
);
native_graph_id!(
    ModelBindingId,
    "Identifier for one Rust-owned NativeGraph model binding."
);
native_graph_id!(
    ModelSecretId,
    "Logical identifier for a NativeGraph model secret, never its value."
);
native_graph_id!(
    AdapterProtocolFactoryId,
    "Canonical identifier for a NativeGraph adapter-protocol factory."
);
native_graph_id!(
    AdapterRuntimeProviderId,
    "Canonical identifier for a NativeGraph adapter-runtime provider."
);
native_graph_id!(
    EnvironmentStepperFactoryId,
    "Canonical identifier for a NativeGraph environment-stepper factory."
);

/// Exact execution profile selected by a schema-1.1 task.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NativeGraphProfile {
    /// Rust owns graph traversal and every model request.
    NativeGraph,
    /// One supervised compatibility driver owns episode decisions.
    ExternallyDriven,
}

impl NativeGraphProfile {
    fn as_str(self) -> &'static str {
        match self {
            Self::NativeGraph => "native_graph",
            Self::ExternallyDriven => "externally_driven",
        }
    }
}

/// Declared responsibility of one supervised adapter.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AdapterRole {
    /// Executes a Rust-authorized tool operation.
    Tool,
    /// Supplies a bounded policy decision to the Rust graph controller.
    Policy,
    /// Resets or advances a declared external environment.
    Environment,
    /// Supplies a bounded heuristic result to the Rust graph controller.
    Heuristic,
    /// Owns the compatibility-only externally driven episode protocol.
    Driver,
}

impl AdapterRole {
    fn as_str(self) -> &'static str {
        match self {
            Self::Tool => "tool",
            Self::Policy => "policy",
            Self::Environment => "environment",
            Self::Heuristic => "heuristic",
            Self::Driver => "driver",
        }
    }
}

/// Pinned tokenizer policy for a model binding.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TokenizerBindingSpec {
    /// A registered local tokenizer identity and revision.
    Local {
        /// Local tokenizer name.
        name: String,
        /// Pinned tokenizer revision.
        revision: String,
        /// Whether tokenization applies the endpoint chat template.
        apply_chat_template: bool,
    },
    /// A pinned server tokenizer endpoint.
    Server {
        /// Server tokenizer origin.
        url: String,
        /// Whether tokenization applies the endpoint chat template.
        apply_chat_template: bool,
    },
}

/// Finite generation settings pinned by a model binding.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct GenerationDefaults {
    /// Sampling temperature when the endpoint supports it.
    pub temperature: Option<f64>,
    /// Nucleus-sampling probability when the endpoint supports it.
    pub top_p: Option<f64>,
    /// Top-k sampling bound when the endpoint supports it.
    pub top_k: Option<u32>,
    /// Maximum generated tokens when the endpoint supports it.
    pub max_tokens: Option<u32>,
    /// Minimum generated tokens when the endpoint supports it.
    pub min_tokens: Option<u32>,
    /// Provider sampling seed when the endpoint supports it.
    pub seed: Option<u64>,
    /// Presence penalty when the endpoint supports it.
    pub presence_penalty: Option<f64>,
    /// Frequency penalty when the endpoint supports it.
    pub frequency_penalty: Option<f64>,
    /// Repetition penalty when the endpoint supports it.
    pub repetition_penalty: Option<f64>,
}

/// Bounded raw-exchange capture policy for a model binding.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelCapturePolicy {
    /// Retain no raw model exchange.
    None,
    /// Retain only metadata derived by Rust.
    Metadata,
    /// Retain raw exchange only after configured redaction.
    RedactedRaw,
}

impl ModelCapturePolicy {
    fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Metadata => "metadata",
            Self::RedactedRaw => "redacted_raw",
        }
    }
}

/// One header whose runtime value is resolved from a logical secret identifier.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HeaderSecretRef {
    /// Canonical lowercase HTTP header name.
    pub header: String,
    /// Logical secret identifier; package bytes never include its value.
    pub secret: ModelSecretId,
}

/// Complete immutable selection for one Rust-owned model binding.
#[derive(Clone, Debug, PartialEq)]
pub struct ModelBindingSpec {
    /// Binding identifier referenced by graph model nodes.
    pub id: ModelBindingId,
    /// Authored endpoint profile identifier.
    pub endpoint_profile_id: String,
    /// Registered endpoint factory identifier.
    pub endpoint_factory_id: EndpointId,
    /// Registered transport factory identifier.
    pub transport_factory_id: String,
    /// Provider-local model identifier.
    pub model: String,
    /// Ordered provider endpoint origins.
    pub urls: Vec<String>,
    /// Whether this binding requests streaming responses.
    pub streaming: bool,
    /// Pinned tokenizer policy.
    pub tokenizer: TokenizerBindingSpec,
    /// Header names paired with logical secret identifiers.
    pub authentication: Vec<HeaderSecretRef>,
    /// Finite generation defaults.
    pub generation: GenerationDefaults,
    /// Number of pre-send connection retries.
    pub max_connect_retries: u32,
    /// Positive request deadline in milliseconds.
    pub request_timeout_ms: NonZeroU64,
    /// Bounded raw-exchange capture policy.
    pub capture: ModelCapturePolicy,
}

impl ModelBindingSpec {
    /// Computes the immutable identity of every runtime-relevant binding field.
    pub fn identity_digest(&self) -> ArtifactDigest {
        let mut material = Vec::new();
        append_identity_field(
            &mut material,
            "native-graph-model-identity.domain",
            b"aiperf-native-graph-model-v1",
        );
        append_model_binding_identity(&mut material, self);
        ArtifactDigest::from_bytes(&material)
    }

    /// Borrows the registered endpoint factory identifier.
    pub fn endpoint_factory_id(&self) -> &EndpointId {
        &self.endpoint_factory_id
    }

    /// Borrows the registered transport factory identifier.
    pub fn transport_factory_id(&self) -> &str {
        &self.transport_factory_id
    }

    /// Borrows the logical authentication references.
    pub fn authentication(&self) -> &[HeaderSecretRef] {
        &self.authentication
    }
}

/// Exact supervised adapter manifest entry retained by a NativeGraph package.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AdapterSpec {
    /// Adapter identifier.
    pub id: AdapterId,
    /// Adapter responsibility.
    pub role: AdapterRole,
    /// Exact argv used to launch the adapter.
    pub argv: Vec<String>,
    /// Relative path to the adapter executable source.
    pub executable: String,
    /// Explicitly selected adapter configuration files.
    pub config: Vec<String>,
    /// Explicitly selected adapter policy files.
    pub policy: Vec<String>,
}

/// Immutable graph program bytes selected from the acquired package snapshot.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NativeGraphProgramSource {
    path: String,
    bytes: Arc<[u8]>,
    digest: ArtifactDigest,
}

/// Immutable reset-source bytes selected from the acquired rollout snapshot.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NativeGraphRolloutResetSource {
    path: String,
    bytes: Arc<[u8]>,
    digest: ArtifactDigest,
}

impl NativeGraphRolloutResetSource {
    /// Returns the canonical package-relative reset-source path.
    pub fn path(&self) -> &str {
        &self.path
    }

    /// Returns the exact reset-source bytes retained at import time.
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Returns the digest of the retained reset-source bytes.
    pub fn digest(&self) -> &ArtifactDigest {
        &self.digest
    }
}

/// Bounded protocol values selected for one environment adapter session.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NativeGraphEnvironmentProtocolLimits {
    max_frame_bytes: u64,
    max_identifier_bytes: u64,
    max_json_bytes: u64,
    max_json_depth: u64,
    max_json_array_entries: u64,
    max_json_object_entries: u64,
    max_operation_ledger_entries: u64,
    max_model_call_lineage_entries: u64,
    max_session_model_call_lineage_entries: u64,
    max_session_model_call_lineage_bytes: u64,
    max_artifact_handles: u64,
    max_artifact_bytes: u64,
}

impl NativeGraphEnvironmentProtocolLimits {
    /// Returns the maximum JSONL frame bytes admitted for the environment session.
    pub const fn max_frame_bytes(&self) -> u64 {
        self.max_frame_bytes
    }

    /// Returns the maximum bytes in a typed identifier.
    pub const fn max_identifier_bytes(&self) -> u64 {
        self.max_identifier_bytes
    }

    /// Returns the maximum JSON payload bytes.
    pub const fn max_json_bytes(&self) -> u64 {
        self.max_json_bytes
    }

    /// Returns the maximum JSON nesting depth.
    pub const fn max_json_depth(&self) -> u64 {
        self.max_json_depth
    }

    /// Returns the maximum entries in one JSON array.
    pub const fn max_json_array_entries(&self) -> u64 {
        self.max_json_array_entries
    }

    /// Returns the maximum entries in one JSON object.
    pub const fn max_json_object_entries(&self) -> u64 {
        self.max_json_object_entries
    }

    /// Returns the maximum retained protocol operation correlations.
    pub const fn max_operation_ledger_entries(&self) -> u64 {
        self.max_operation_ledger_entries
    }

    /// Returns the maximum model-call lineage entries per pending policy decision.
    pub const fn max_model_call_lineage_entries(&self) -> u64 {
        self.max_model_call_lineage_entries
    }

    /// Returns the maximum model-call lineage entries retained by the environment session.
    pub const fn max_session_model_call_lineage_entries(&self) -> u64 {
        self.max_session_model_call_lineage_entries
    }

    /// Returns the maximum model-call lineage bytes retained by the environment session.
    pub const fn max_session_model_call_lineage_bytes(&self) -> u64 {
        self.max_session_model_call_lineage_bytes
    }

    /// Returns the maximum artifact capabilities carried by the environment protocol.
    pub const fn max_artifact_handles(&self) -> u64 {
        self.max_artifact_handles
    }

    /// Returns the maximum bytes in one protocol artifact operation.
    pub const fn max_artifact_bytes(&self) -> u64 {
        self.max_artifact_bytes
    }
}

/// Bounded Rust-owned artifact-store values selected for one environment episode.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NativeGraphEnvironmentArtifactLimits {
    max_artifacts: u64,
    max_total_bytes: u64,
    max_artifact_bytes: u64,
    max_download_handles: u64,
}

impl NativeGraphEnvironmentArtifactLimits {
    /// Returns the maximum frozen artifacts plus upload reservations.
    pub const fn max_artifacts(&self) -> u64 {
        self.max_artifacts
    }

    /// Returns the maximum bytes retained or reserved by the episode store.
    pub const fn max_total_bytes(&self) -> u64 {
        self.max_total_bytes
    }

    /// Returns the maximum bytes in one artifact upload.
    pub const fn max_artifact_bytes(&self) -> u64 {
        self.max_artifact_bytes
    }

    /// Returns the maximum live artifact-download capabilities.
    pub const fn max_download_handles(&self) -> u64 {
        self.max_download_handles
    }
}

/// Immutable RL policy facts selected by one NativeGraph rollout package.
#[derive(Clone, Debug, PartialEq)]
pub struct NativeGraphRolloutPolicy {
    environment: String,
    horizon: u32,
    gamma: f64,
}

impl NativeGraphRolloutPolicy {
    /// Returns the declared environment identity.
    pub fn environment(&self) -> &str {
        &self.environment
    }

    /// Returns the maximum rollout horizon.
    pub const fn horizon(&self) -> u32 {
        self.horizon
    }

    /// Returns the finite discount factor.
    pub const fn gamma(&self) -> f64 {
        self.gamma
    }
}

/// Immutable bounds selected for a NativeGraph rollout policy.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NativeGraphRolloutLimits {
    max_environment_bytes: u64,
    max_horizon: u32,
}

impl NativeGraphRolloutLimits {
    /// Returns the maximum environment-identity bytes.
    pub const fn max_environment_bytes(&self) -> u64 {
        self.max_environment_bytes
    }

    /// Returns the maximum rollout horizon allowed by the package.
    pub const fn max_horizon(&self) -> u32 {
        self.max_horizon
    }
}

/// Immutable environment-adapter selection retained by an imported rollout.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NativeGraphRolloutEnvironment {
    adapter_id: AdapterId,
    protocol_factory_id: AdapterProtocolFactoryId,
    runtime_provider_id: AdapterRuntimeProviderId,
    stepper_factory_id: EnvironmentStepperFactoryId,
    operation_deadline_ms: NonZeroU64,
    protocol_limits: NativeGraphEnvironmentProtocolLimits,
    artifact_limits: NativeGraphEnvironmentArtifactLimits,
    reset_source: NativeGraphRolloutResetSource,
}

impl NativeGraphRolloutEnvironment {
    /// Returns the declared adapter that must have the `environment` role.
    pub fn adapter_id(&self) -> &AdapterId {
        &self.adapter_id
    }

    /// Returns the canonical protocol-factory selection.
    pub fn protocol_factory_id(&self) -> &AdapterProtocolFactoryId {
        &self.protocol_factory_id
    }

    /// Returns the canonical runtime-provider selection.
    pub fn runtime_provider_id(&self) -> &AdapterRuntimeProviderId {
        &self.runtime_provider_id
    }

    /// Returns the canonical environment-stepper factory selection.
    pub fn stepper_factory_id(&self) -> &EnvironmentStepperFactoryId {
        &self.stepper_factory_id
    }

    /// Returns the positive per-operation deadline selected by the package.
    pub const fn operation_deadline_ms(&self) -> NonZeroU64 {
        self.operation_deadline_ms
    }

    /// Returns the exact environment protocol caps selected by the package.
    pub fn protocol_limits(&self) -> &NativeGraphEnvironmentProtocolLimits {
        &self.protocol_limits
    }

    /// Returns the exact Rust-owned artifact-store caps selected by the package.
    pub fn artifact_limits(&self) -> &NativeGraphEnvironmentArtifactLimits {
        &self.artifact_limits
    }

    /// Returns the acquired reset-source snapshot.
    pub fn reset_source(&self) -> &NativeGraphRolloutResetSource {
        &self.reset_source
    }
}

/// Immutable optional rollout selection retained by a NativeGraph package.
#[derive(Clone, Debug, PartialEq)]
pub struct NativeGraphRolloutPlan {
    environment: NativeGraphRolloutEnvironment,
    policy: NativeGraphRolloutPolicy,
    limits: NativeGraphRolloutLimits,
}

impl Eq for NativeGraphRolloutPlan {}

impl NativeGraphRolloutPlan {
    /// Returns the environment adapter and capability selection.
    pub fn environment(&self) -> &NativeGraphRolloutEnvironment {
        &self.environment
    }

    /// Returns the rollout policy selected at import time.
    pub fn policy(&self) -> &NativeGraphRolloutPolicy {
        &self.policy
    }

    /// Returns the bounds selected for the rollout policy.
    pub const fn limits(&self) -> NativeGraphRolloutLimits {
        self.limits
    }
}

impl NativeGraphProgramSource {
    /// Returns the canonical package-relative program path.
    pub fn path(&self) -> &str {
        &self.path
    }

    /// Returns the exact validated program bytes retained from the package snapshot.
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Returns the digest of the retained program bytes.
    pub fn digest(&self) -> &ArtifactDigest {
        &self.digest
    }
}

/// Immutable schema-1.1 NativeGraph package plan.
#[derive(Clone, Debug, PartialEq)]
pub struct NativeGraphPackagePlan {
    profile: NativeGraphProfile,
    program: Option<NativeGraphProgramSource>,
    model_bindings: Vec<ModelBindingSpec>,
    adapters: Vec<AdapterSpec>,
    driver: Option<AdapterId>,
    rollout: Option<NativeGraphRolloutPlan>,
    executable_source_digest: ArtifactDigest,
}

impl Eq for NativeGraphPackagePlan {}

impl NativeGraphPackagePlan {
    /// Returns the selected execution profile.
    pub const fn profile(&self) -> NativeGraphProfile {
        self.profile
    }

    /// Returns the graph program digest when Rust owns graph traversal.
    pub fn program(&self) -> Option<&ArtifactDigest> {
        self.program.as_ref().map(NativeGraphProgramSource::digest)
    }

    /// Returns the validated graph program snapshot when Rust owns graph traversal.
    pub fn program_source(&self) -> Option<&NativeGraphProgramSource> {
        self.program.as_ref()
    }

    /// Returns the complete ordered model-binding table.
    pub fn model_bindings(&self) -> &[ModelBindingSpec] {
        &self.model_bindings
    }

    /// Returns the complete ordered adapter table.
    pub fn adapters(&self) -> &[AdapterSpec] {
        &self.adapters
    }

    /// Returns the explicit compatibility driver when selected.
    pub fn driver(&self) -> Option<&AdapterId> {
        self.driver.as_ref()
    }

    /// Returns the immutable adapter selected to drive an externally driven episode.
    pub fn driver_adapter(&self) -> Option<&AdapterSpec> {
        let driver = self.driver.as_ref()?;
        self.adapters.iter().find(|adapter| adapter.id == *driver)
    }

    /// Returns the optional strict rollout selection retained from `rollout.toml`.
    pub fn rollout(&self) -> Option<&NativeGraphRolloutPlan> {
        self.rollout.as_ref()
    }

    /// Returns the executable-source projection retained by the package.
    pub fn executable_source_digest(&self) -> &ArtifactDigest {
        &self.executable_source_digest
    }

    pub(crate) fn append_identity_material(&self, material: &mut Vec<u8>) {
        append_identity_field(
            material,
            "native-graph-package.profile",
            self.profile.as_str().as_bytes(),
        );
        append_optional_digest(
            material,
            "native-graph-package.program",
            self.program.as_ref().map(NativeGraphProgramSource::digest),
        );
        append_identity_field(
            material,
            "native-graph-package.model-binding-count",
            &(self.model_bindings.len() as u64).to_le_bytes(),
        );
        for binding in &self.model_bindings {
            append_model_binding_identity(material, binding);
        }
        append_identity_field(
            material,
            "native-graph-package.adapter-count",
            &(self.adapters.len() as u64).to_le_bytes(),
        );
        for adapter in &self.adapters {
            append_adapter_identity(material, adapter);
        }
        append_optional_id(
            material,
            "native-graph-package.driver",
            self.driver.as_ref().map(AdapterId::as_str),
        );
        if let Some(rollout) = &self.rollout {
            append_rollout_identity(material, rollout);
        }
        append_identity_field(
            material,
            "native-graph-package.executable-source-digest",
            self.executable_source_digest.as_str().as_bytes(),
        );
    }
}

/// Strict authored schema-1.1 NativeGraph section.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct NativeGraphSectionDto {
    profile: NativeGraphProfile,
    program: Option<String>,
    model_bindings: Option<String>,
    adapter_manifest: Option<String>,
    driver: Option<AdapterId>,
}

/// Parsed NativeGraph package data awaiting its complete source projection.
pub(crate) struct NativeGraphPackageDraft {
    profile: NativeGraphProfile,
    program: Option<NativeGraphProgramSource>,
    model_bindings: Vec<ModelBindingSpec>,
    adapters: Vec<AdapterSpec>,
    driver: Option<AdapterId>,
    rollout: Option<NativeGraphRolloutPlan>,
    executable_source_paths: BTreeSet<String>,
}

impl NativeGraphPackageDraft {
    /// Borrows every exact source path needed for NativeGraph execution.
    pub(crate) fn executable_source_paths(&self) -> impl Iterator<Item = &str> {
        self.executable_source_paths.iter().map(String::as_str)
    }

    /// Attaches the complete package executable projection after task planning.
    pub(crate) fn into_plan(
        self,
        executable_source_digest: ArtifactDigest,
    ) -> NativeGraphPackagePlan {
        NativeGraphPackagePlan {
            profile: self.profile,
            program: self.program,
            model_bindings: self.model_bindings,
            adapters: self.adapters,
            driver: self.driver,
            rollout: self.rollout,
            executable_source_digest,
        }
    }
}

/// Resolves strict manifest bytes from the acquired source snapshot only.
pub(crate) fn resolve_native_graph_package(
    source: &AcquiredSource,
    section: NativeGraphSectionDto,
) -> Result<NativeGraphPackageDraft, HarborImportError> {
    let adapter_manifest_path =
        required_relative_path("native_graph.adapter_manifest", section.adapter_manifest)?;
    let adapter_bytes = read_selected_file(source, &adapter_manifest_path)?;
    let adapter_manifest: AdapterManifestDto = decode_toml(&adapter_manifest_path, adapter_bytes)?;
    let adapters = resolve_adapters(source, adapter_manifest.adapters)?;
    let mut executable_source_paths = BTreeSet::from([adapter_manifest_path]);
    for adapter in &adapters {
        executable_source_paths.insert(adapter.executable.clone());
        executable_source_paths.extend(adapter.config.iter().cloned());
        executable_source_paths.extend(adapter.policy.iter().cloned());
    }
    let rollout = resolve_rollout(source, section.profile, &adapters)?;
    if let Some(rollout) = &rollout {
        executable_source_paths.insert(ROLLOUT_MANIFEST_PATH.to_owned());
        executable_source_paths.insert(rollout.environment.reset_source.path.clone());
    }

    match section.profile {
        NativeGraphProfile::NativeGraph => {
            if section.driver.is_some() {
                return invalid("native_graph profile must not declare a driver");
            }
            if adapters
                .iter()
                .any(|adapter| adapter.role == AdapterRole::Driver)
            {
                return invalid("native_graph profile must not declare a driver adapter");
            }
            let program_path = required_relative_path("native_graph.program", section.program)?;
            let program_bytes = source.read_owned(&program_path)?;
            let program = NativeGraphProgramSource {
                path: program_path.clone(),
                digest: ArtifactDigest::from_bytes(&program_bytes),
                bytes: program_bytes,
            };
            let model_manifest_path =
                required_relative_path("native_graph.model_bindings", section.model_bindings)?;
            let model_bytes = read_selected_file(source, &model_manifest_path)?;
            let model_manifest: ModelManifestDto = decode_toml(&model_manifest_path, model_bytes)?;
            let model_bindings = resolve_model_bindings(model_manifest.model_bindings)?;
            if model_bindings.is_empty() {
                return invalid("native_graph profile requires at least one model binding");
            }
            executable_source_paths.insert(program_path);
            executable_source_paths.insert(model_manifest_path);
            Ok(NativeGraphPackageDraft {
                profile: NativeGraphProfile::NativeGraph,
                program: Some(program),
                model_bindings,
                adapters,
                driver: None,
                rollout,
                executable_source_paths,
            })
        }
        NativeGraphProfile::ExternallyDriven => {
            if section.program.is_some() || section.model_bindings.is_some() {
                return invalid(
                    "externally_driven profile must not declare a graph program or model bindings",
                );
            }
            if rollout.is_some() {
                return invalid("rollout.toml requires the native_graph profile");
            }
            let driver = section.driver.ok_or_else(|| {
                HarborImportError::InvalidPackage(
                    "externally_driven profile requires native_graph.driver".to_owned(),
                )
            })?;
            if adapters.len() != 1
                || adapters[0].id != driver
                || adapters[0].role != AdapterRole::Driver
            {
                return invalid(
                    "externally_driven profile requires exactly one declared driver adapter",
                );
            }
            Ok(NativeGraphPackageDraft {
                profile: NativeGraphProfile::ExternallyDriven,
                program: None,
                model_bindings: Vec::new(),
                adapters,
                driver: Some(driver),
                rollout: None,
                executable_source_paths,
            })
        }
    }
}

const ROLLOUT_MANIFEST_PATH: &str = "rollout.toml";

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RolloutManifestDto {
    environment: RolloutEnvironmentDto,
    artifacts: RolloutArtifactLimitsDto,
    policy: RolloutPolicyDto,
    limits: RolloutLimitsDto,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RolloutEnvironmentDto {
    adapter_id: AdapterId,
    protocol_factory_id: AdapterProtocolFactoryId,
    runtime_provider_id: AdapterRuntimeProviderId,
    stepper_factory_id: EnvironmentStepperFactoryId,
    operation_deadline_ms: NonZeroU64,
    reset_source: String,
    max_frame_bytes: u64,
    max_identifier_bytes: u64,
    max_json_bytes: u64,
    max_json_depth: u64,
    max_json_array_entries: u64,
    max_json_object_entries: u64,
    max_operation_ledger_entries: u64,
    max_model_call_lineage_entries: u64,
    max_session_model_call_lineage_entries: u64,
    max_session_model_call_lineage_bytes: u64,
    max_artifact_handles: u64,
    max_artifact_bytes: u64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RolloutArtifactLimitsDto {
    max_artifacts: u64,
    max_total_bytes: u64,
    max_artifact_bytes: u64,
    max_download_handles: u64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RolloutPolicyDto {
    environment: String,
    horizon: u32,
    gamma: f64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RolloutLimitsDto {
    max_environment_bytes: u64,
    max_horizon: u32,
}

fn resolve_rollout(
    source: &AcquiredSource,
    profile: NativeGraphProfile,
    adapters: &[AdapterSpec],
) -> Result<Option<NativeGraphRolloutPlan>, HarborImportError> {
    let rollout_bytes = match source.read(ROLLOUT_MANIFEST_PATH) {
        Ok(bytes) => bytes,
        Err(HarborImportError::InvalidPackage(message))
            if message == "source file is missing: \"rollout.toml\"" =>
        {
            return Ok(None);
        }
        Err(error) => return Err(error),
    };
    if profile != NativeGraphProfile::NativeGraph {
        return invalid("rollout.toml requires the native_graph profile");
    }
    let manifest: RolloutManifestDto = decode_toml(ROLLOUT_MANIFEST_PATH, rollout_bytes)?;
    let limits = NativeGraphRolloutLimits {
        max_environment_bytes: manifest.limits.max_environment_bytes,
        max_horizon: manifest.limits.max_horizon,
    };
    if limits.max_environment_bytes == 0 || limits.max_horizon == 0 {
        return invalid("rollout.limits must be positive");
    }
    let policy_environment =
        required_text("rollout.policy.environment", manifest.policy.environment)?;
    if u64::try_from(policy_environment.len()).map_err(|_| {
        HarborImportError::InvalidPackage("rollout.policy.environment is too long".to_owned())
    })? > limits.max_environment_bytes
    {
        return invalid("rollout.policy.environment exceeds rollout.limits.max_environment_bytes");
    }
    if manifest.policy.horizon == 0 || manifest.policy.horizon > limits.max_horizon {
        return invalid(
            "rollout.policy.horizon must be positive and within rollout.limits.max_horizon",
        );
    }
    if !manifest.policy.gamma.is_finite() || !(0.0..=1.0).contains(&manifest.policy.gamma) {
        return invalid("rollout.policy.gamma must be finite and within [0, 1]");
    }
    let environment =
        resolve_rollout_environment(source, manifest.environment, manifest.artifacts, adapters)?;
    Ok(Some(NativeGraphRolloutPlan {
        environment,
        policy: NativeGraphRolloutPolicy {
            environment: policy_environment,
            horizon: manifest.policy.horizon,
            gamma: manifest.policy.gamma,
        },
        limits,
    }))
}

fn resolve_rollout_environment(
    source: &AcquiredSource,
    environment: RolloutEnvironmentDto,
    artifacts: RolloutArtifactLimitsDto,
    adapters: &[AdapterSpec],
) -> Result<NativeGraphRolloutEnvironment, HarborImportError> {
    match adapters
        .iter()
        .find(|adapter| adapter.id == environment.adapter_id)
    {
        Some(adapter) if adapter.role == AdapterRole::Environment => {}
        Some(_) => {
            return invalid("rollout.environment.adapter_id must select an environment adapter");
        }
        None => return invalid("rollout.environment.adapter_id is not declared"),
    }
    let protocol_limits = NativeGraphEnvironmentProtocolLimits {
        max_frame_bytes: environment.max_frame_bytes,
        max_identifier_bytes: environment.max_identifier_bytes,
        max_json_bytes: environment.max_json_bytes,
        max_json_depth: environment.max_json_depth,
        max_json_array_entries: environment.max_json_array_entries,
        max_json_object_entries: environment.max_json_object_entries,
        max_operation_ledger_entries: environment.max_operation_ledger_entries,
        max_model_call_lineage_entries: environment.max_model_call_lineage_entries,
        max_session_model_call_lineage_entries: environment.max_session_model_call_lineage_entries,
        max_session_model_call_lineage_bytes: environment.max_session_model_call_lineage_bytes,
        max_artifact_handles: environment.max_artifact_handles,
        max_artifact_bytes: environment.max_artifact_bytes,
    };
    if [
        protocol_limits.max_frame_bytes,
        protocol_limits.max_identifier_bytes,
        protocol_limits.max_json_bytes,
        protocol_limits.max_json_depth,
        protocol_limits.max_json_array_entries,
        protocol_limits.max_json_object_entries,
        protocol_limits.max_operation_ledger_entries,
        protocol_limits.max_model_call_lineage_entries,
        protocol_limits.max_session_model_call_lineage_entries,
        protocol_limits.max_session_model_call_lineage_bytes,
        protocol_limits.max_artifact_handles,
        protocol_limits.max_artifact_bytes,
    ]
    .contains(&0)
    {
        return invalid("rollout.environment protocol limits must be positive");
    }
    let artifact_limits = NativeGraphEnvironmentArtifactLimits {
        max_artifacts: artifacts.max_artifacts,
        max_total_bytes: artifacts.max_total_bytes,
        max_artifact_bytes: artifacts.max_artifact_bytes,
        max_download_handles: artifacts.max_download_handles,
    };
    if [
        artifact_limits.max_artifacts,
        artifact_limits.max_total_bytes,
        artifact_limits.max_artifact_bytes,
        artifact_limits.max_download_handles,
    ]
    .contains(&0)
        || artifact_limits.max_artifact_bytes > artifact_limits.max_total_bytes
    {
        return invalid("rollout.artifacts limits are invalid");
    }
    let reset_path =
        canonical_relative_path("rollout.environment.reset_source", environment.reset_source)?;
    let reset_bytes = source.read_owned(&reset_path)?;
    let reset_source = NativeGraphRolloutResetSource {
        path: reset_path,
        digest: ArtifactDigest::from_bytes(&reset_bytes),
        bytes: reset_bytes,
    };
    Ok(NativeGraphRolloutEnvironment {
        adapter_id: environment.adapter_id,
        protocol_factory_id: environment.protocol_factory_id,
        runtime_provider_id: environment.runtime_provider_id,
        stepper_factory_id: environment.stepper_factory_id,
        operation_deadline_ms: environment.operation_deadline_ms,
        protocol_limits,
        artifact_limits,
        reset_source,
    })
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ModelManifestDto {
    #[serde(default)]
    model_bindings: Vec<ModelBindingDto>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ModelBindingDto {
    id: ModelBindingId,
    endpoint_profile_id: String,
    endpoint_factory_id: EndpointId,
    transport_factory_id: String,
    model: String,
    urls: Vec<String>,
    streaming: bool,
    tokenizer: TokenizerBindingDto,
    #[serde(default)]
    authentication: Vec<HeaderSecretRefDto>,
    generation: GenerationDefaultsDto,
    #[serde(default)]
    max_connect_retries: u32,
    request_timeout_ms: NonZeroU64,
    capture: ModelCapturePolicy,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct TokenizerBindingDto {
    #[serde(rename = "type")]
    kind: TokenizerKind,
    name: Option<String>,
    revision: Option<String>,
    url: Option<String>,
    #[serde(default)]
    apply_chat_template: bool,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
enum TokenizerKind {
    Local,
    Server,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct HeaderSecretRefDto {
    header: String,
    secret: ModelSecretId,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct GenerationDefaultsDto {
    temperature: Option<f64>,
    top_p: Option<f64>,
    top_k: Option<u32>,
    max_tokens: Option<u32>,
    min_tokens: Option<u32>,
    seed: Option<u64>,
    presence_penalty: Option<f64>,
    frequency_penalty: Option<f64>,
    repetition_penalty: Option<f64>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct AdapterManifestDto {
    #[serde(default)]
    adapters: Vec<AdapterSpecDto>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct AdapterSpecDto {
    id: AdapterId,
    role: AdapterRole,
    argv: Vec<String>,
    executable: String,
    #[serde(default)]
    config: Vec<String>,
    #[serde(default)]
    policy: Vec<String>,
}

fn resolve_model_bindings(
    bindings: Vec<ModelBindingDto>,
) -> Result<Vec<ModelBindingSpec>, HarborImportError> {
    let mut ids = BTreeSet::new();
    let mut resolved = Vec::with_capacity(bindings.len());
    for binding in bindings {
        if !ids.insert(binding.id.clone()) {
            return invalid("model binding id is duplicated");
        }
        let endpoint_profile_id = required_text(
            "model_binding.endpoint_profile_id",
            binding.endpoint_profile_id,
        )?;
        let transport_factory_id = required_text(
            "model_binding.transport_factory_id",
            binding.transport_factory_id,
        )?;
        let model = required_text("model_binding.model", binding.model)?;
        if binding.urls.is_empty() {
            return invalid("model_binding.urls must not be empty");
        }
        let urls = binding
            .urls
            .into_iter()
            .map(|url| validate_http_url("model_binding.urls", url))
            .collect::<Result<Vec<_>, _>>()?;
        let tokenizer = resolve_tokenizer(binding.tokenizer)?;
        let authentication = resolve_authentication(binding.authentication)?;
        let generation = resolve_generation(binding.generation)?;
        resolved.push(ModelBindingSpec {
            id: binding.id,
            endpoint_profile_id,
            endpoint_factory_id: binding.endpoint_factory_id,
            transport_factory_id,
            model,
            urls,
            streaming: binding.streaming,
            tokenizer,
            authentication,
            generation,
            max_connect_retries: binding.max_connect_retries,
            request_timeout_ms: binding.request_timeout_ms,
            capture: binding.capture,
        });
    }
    Ok(resolved)
}

fn resolve_tokenizer(dto: TokenizerBindingDto) -> Result<TokenizerBindingSpec, HarborImportError> {
    match dto.kind {
        TokenizerKind::Local => {
            if dto.url.is_some() {
                return invalid("local tokenizer must not declare url");
            }
            Ok(TokenizerBindingSpec::Local {
                name: required_text(
                    "tokenizer.name",
                    dto.name.ok_or_else(|| missing("tokenizer.name"))?,
                )?,
                revision: required_text(
                    "tokenizer.revision",
                    dto.revision.ok_or_else(|| missing("tokenizer.revision"))?,
                )?,
                apply_chat_template: dto.apply_chat_template,
            })
        }
        TokenizerKind::Server => {
            if dto.name.is_some() || dto.revision.is_some() {
                return invalid("server tokenizer must not declare name or revision");
            }
            Ok(TokenizerBindingSpec::Server {
                url: validate_http_url(
                    "tokenizer.url",
                    dto.url.ok_or_else(|| missing("tokenizer.url"))?,
                )?,
                apply_chat_template: dto.apply_chat_template,
            })
        }
    }
}

fn resolve_authentication(
    authentication: Vec<HeaderSecretRefDto>,
) -> Result<Vec<HeaderSecretRef>, HarborImportError> {
    let mut headers = BTreeSet::new();
    let mut resolved = Vec::with_capacity(authentication.len());
    for reference in authentication {
        let header = normalize_header_name(&reference.header)?;
        if !headers.insert(header.clone()) {
            return invalid("model binding authentication header is duplicated");
        }
        resolved.push(HeaderSecretRef {
            header,
            secret: reference.secret,
        });
    }
    Ok(resolved)
}

fn resolve_generation(dto: GenerationDefaultsDto) -> Result<GenerationDefaults, HarborImportError> {
    for (field, value) in [
        ("generation.temperature", dto.temperature),
        ("generation.top_p", dto.top_p),
        ("generation.presence_penalty", dto.presence_penalty),
        ("generation.frequency_penalty", dto.frequency_penalty),
        ("generation.repetition_penalty", dto.repetition_penalty),
    ] {
        if value.is_some_and(|value| !value.is_finite()) {
            return invalid(&format!("{field} must be finite"));
        }
    }
    Ok(GenerationDefaults {
        temperature: dto.temperature,
        top_p: dto.top_p,
        top_k: dto.top_k,
        max_tokens: dto.max_tokens,
        min_tokens: dto.min_tokens,
        seed: dto.seed,
        presence_penalty: dto.presence_penalty,
        frequency_penalty: dto.frequency_penalty,
        repetition_penalty: dto.repetition_penalty,
    })
}

fn resolve_adapters(
    source: &AcquiredSource,
    adapters: Vec<AdapterSpecDto>,
) -> Result<Vec<AdapterSpec>, HarborImportError> {
    let mut ids = BTreeSet::new();
    let mut resolved = Vec::with_capacity(adapters.len());
    for adapter in adapters {
        if !ids.insert(adapter.id.clone()) {
            return invalid("adapter id is duplicated");
        }
        if adapter.argv.is_empty()
            || adapter
                .argv
                .iter()
                .any(|argument| argument.trim().is_empty())
        {
            return invalid("adapter.argv must be a nonempty argv");
        }
        let executable = canonical_relative_path("adapter.executable", adapter.executable)?;
        if adapter.argv.first() != Some(&executable) {
            return invalid("adapter.argv must start with adapter.executable");
        }
        read_selected_file(source, &executable)?;
        let config = resolve_adapter_paths(source, "adapter.config", adapter.config)?;
        let policy = resolve_adapter_paths(source, "adapter.policy", adapter.policy)?;
        resolved.push(AdapterSpec {
            id: adapter.id,
            role: adapter.role,
            argv: adapter.argv,
            executable,
            config,
            policy,
        });
    }
    Ok(resolved)
}

fn resolve_adapter_paths(
    source: &AcquiredSource,
    field: &str,
    paths: Vec<String>,
) -> Result<Vec<String>, HarborImportError> {
    let mut selected = BTreeSet::new();
    for path in paths {
        let path = canonical_relative_path(field, path)?;
        if !selected.insert(path.clone()) {
            return invalid(&format!("{field} contains a duplicate path"));
        }
        read_selected_file(source, &path)?;
    }
    Ok(selected.into_iter().collect())
}

fn read_selected_file<'a>(
    source: &'a AcquiredSource,
    path: &str,
) -> Result<&'a [u8], HarborImportError> {
    source.read(path)
}

fn decode_toml<T: serde::de::DeserializeOwned>(
    path: &str,
    bytes: &[u8],
) -> Result<T, HarborImportError> {
    let text = std::str::from_utf8(bytes)
        .map_err(|error| HarborImportError::InvalidPackage(format!("{path}: {error}")))?;
    toml::from_str(text)
        .map_err(|error| HarborImportError::InvalidPackage(format!("{path}: {error}")))
}

fn required_relative_path(field: &str, value: Option<String>) -> Result<String, HarborImportError> {
    canonical_relative_path(field, value.ok_or_else(|| missing(field))?)
}

fn canonical_relative_path(field: &str, value: String) -> Result<String, HarborImportError> {
    let path = Path::new(&value);
    if value.is_empty()
        || path.is_absolute()
        || path
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        return invalid(&format!("{field} must be a canonical relative path"));
    }
    let normalized = path
        .components()
        .filter_map(|component| match component {
            Component::Normal(segment) => Some(segment.to_string_lossy()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("/");
    if value != normalized {
        return invalid(&format!("{field} must be a canonical relative path"));
    }
    Ok(value)
}

fn required_text(field: &str, value: String) -> Result<String, HarborImportError> {
    if value.is_empty() || value.trim() != value {
        return invalid(&format!("{field} must be nonempty and trimmed"));
    }
    Ok(value)
}

fn validate_http_url(field: &str, value: String) -> Result<String, HarborImportError> {
    if value.is_empty() || value.trim() != value {
        return invalid(&format!("{field} must contain nonempty trimmed URLs"));
    }
    let parsed = Url::parse(&value)
        .map_err(|error| HarborImportError::InvalidPackage(format!("{field}: {error}")))?;
    if !matches!(parsed.scheme(), "http" | "https") || parsed.host_str().is_none() {
        return invalid(&format!("{field} must contain absolute HTTP URLs"));
    }
    if !parsed.username().is_empty() || parsed.password().is_some() {
        return invalid(&format!("{field} must not contain URL userinfo"));
    }
    if parsed.query().is_some() || parsed.fragment().is_some() {
        return invalid(&format!("{field} must not contain URL query or fragment"));
    }
    Ok(value)
}

fn normalize_header_name(value: &str) -> Result<String, HarborImportError> {
    if value.is_empty()
        || value.trim() != value
        || !value.bytes().all(|byte| {
            byte.is_ascii_alphanumeric()
                || matches!(
                    byte,
                    b'!' | b'#'
                        | b'$'
                        | b'%'
                        | b'&'
                        | b'\''
                        | b'*'
                        | b'+'
                        | b'-'
                        | b'.'
                        | b'^'
                        | b'_'
                        | b'`'
                        | b'|'
                        | b'~'
                )
        })
    {
        return invalid("authentication.header must be an HTTP token");
    }
    Ok(value.to_ascii_lowercase())
}

fn append_model_binding_identity(material: &mut Vec<u8>, binding: &ModelBindingSpec) {
    append_identity_field(
        material,
        "native-graph-model.id",
        binding.id.as_str().as_bytes(),
    );
    append_identity_field(
        material,
        "native-graph-model.endpoint-profile-id",
        binding.endpoint_profile_id.as_bytes(),
    );
    append_identity_field(
        material,
        "native-graph-model.endpoint-factory-id",
        binding.endpoint_factory_id.as_str().as_bytes(),
    );
    append_identity_field(
        material,
        "native-graph-model.transport-factory-id",
        binding.transport_factory_id.as_bytes(),
    );
    append_identity_field(
        material,
        "native-graph-model.model",
        binding.model.as_bytes(),
    );
    append_identity_field(
        material,
        "native-graph-model.url-count",
        &(binding.urls.len() as u64).to_le_bytes(),
    );
    for url in &binding.urls {
        append_identity_field(material, "native-graph-model.url", url.as_bytes());
    }
    append_identity_field(
        material,
        "native-graph-model.streaming",
        &[u8::from(binding.streaming)],
    );
    append_tokenizer_identity(material, &binding.tokenizer);
    append_identity_field(
        material,
        "native-graph-model.authentication-count",
        &(binding.authentication.len() as u64).to_le_bytes(),
    );
    for authentication in &binding.authentication {
        append_identity_field(
            material,
            "native-graph-model.authentication-header",
            authentication.header.as_bytes(),
        );
        append_identity_field(
            material,
            "native-graph-model.authentication-secret-id",
            authentication.secret.as_str().as_bytes(),
        );
    }
    append_generation_identity(material, &binding.generation);
    append_identity_field(
        material,
        "native-graph-model.max-connect-retries",
        &binding.max_connect_retries.to_le_bytes(),
    );
    append_identity_field(
        material,
        "native-graph-model.request-timeout-ms",
        &binding.request_timeout_ms.get().to_le_bytes(),
    );
    append_identity_field(
        material,
        "native-graph-model.capture",
        binding.capture.as_str().as_bytes(),
    );
}

fn append_tokenizer_identity(material: &mut Vec<u8>, tokenizer: &TokenizerBindingSpec) {
    match tokenizer {
        TokenizerBindingSpec::Local {
            name,
            revision,
            apply_chat_template,
        } => {
            append_identity_field(material, "native-graph-tokenizer.type", b"local");
            append_identity_field(material, "native-graph-tokenizer.name", name.as_bytes());
            append_identity_field(
                material,
                "native-graph-tokenizer.revision",
                revision.as_bytes(),
            );
            append_identity_field(
                material,
                "native-graph-tokenizer.apply-chat-template",
                &[u8::from(*apply_chat_template)],
            );
        }
        TokenizerBindingSpec::Server {
            url,
            apply_chat_template,
        } => {
            append_identity_field(material, "native-graph-tokenizer.type", b"server");
            append_identity_field(material, "native-graph-tokenizer.url", url.as_bytes());
            append_identity_field(
                material,
                "native-graph-tokenizer.apply-chat-template",
                &[u8::from(*apply_chat_template)],
            );
        }
    }
}

fn append_generation_identity(material: &mut Vec<u8>, generation: &GenerationDefaults) {
    append_optional_f64(
        material,
        "native-graph-generation.temperature",
        generation.temperature,
    );
    append_optional_f64(material, "native-graph-generation.top-p", generation.top_p);
    append_optional_u32(material, "native-graph-generation.top-k", generation.top_k);
    append_optional_u32(
        material,
        "native-graph-generation.max-tokens",
        generation.max_tokens,
    );
    append_optional_u32(
        material,
        "native-graph-generation.min-tokens",
        generation.min_tokens,
    );
    append_optional_u64(material, "native-graph-generation.seed", generation.seed);
    append_optional_f64(
        material,
        "native-graph-generation.presence-penalty",
        generation.presence_penalty,
    );
    append_optional_f64(
        material,
        "native-graph-generation.frequency-penalty",
        generation.frequency_penalty,
    );
    append_optional_f64(
        material,
        "native-graph-generation.repetition-penalty",
        generation.repetition_penalty,
    );
}

fn append_adapter_identity(material: &mut Vec<u8>, adapter: &AdapterSpec) {
    append_identity_field(
        material,
        "native-graph-adapter.id",
        adapter.id.as_str().as_bytes(),
    );
    append_identity_field(
        material,
        "native-graph-adapter.role",
        adapter.role.as_str().as_bytes(),
    );
    append_identity_field(
        material,
        "native-graph-adapter.argv-count",
        &(adapter.argv.len() as u64).to_le_bytes(),
    );
    for argument in &adapter.argv {
        append_identity_field(material, "native-graph-adapter.argv", argument.as_bytes());
    }
    append_identity_field(
        material,
        "native-graph-adapter.executable",
        adapter.executable.as_bytes(),
    );
    append_path_list_identity(material, "native-graph-adapter.config", &adapter.config);
    append_path_list_identity(material, "native-graph-adapter.policy", &adapter.policy);
}

fn append_rollout_identity(material: &mut Vec<u8>, rollout: &NativeGraphRolloutPlan) {
    append_identity_field(material, "native-graph-rollout.format", b"1");
    append_identity_field(
        material,
        "native-graph-rollout.environment.adapter-id",
        rollout.environment.adapter_id.as_str().as_bytes(),
    );
    append_identity_field(
        material,
        "native-graph-rollout.environment.protocol-factory-id",
        rollout.environment.protocol_factory_id.as_str().as_bytes(),
    );
    append_identity_field(
        material,
        "native-graph-rollout.environment.runtime-provider-id",
        rollout.environment.runtime_provider_id.as_str().as_bytes(),
    );
    append_identity_field(
        material,
        "native-graph-rollout.environment.stepper-factory-id",
        rollout.environment.stepper_factory_id.as_str().as_bytes(),
    );
    append_identity_field(
        material,
        "native-graph-rollout.environment.operation-deadline-ms",
        &rollout
            .environment
            .operation_deadline_ms
            .get()
            .to_le_bytes(),
    );
    append_environment_protocol_limits_identity(material, &rollout.environment.protocol_limits);
    append_environment_artifact_limits_identity(material, &rollout.environment.artifact_limits);
    append_identity_field(
        material,
        "native-graph-rollout.environment.reset-source-path",
        rollout.environment.reset_source.path.as_bytes(),
    );
    append_identity_field(
        material,
        "native-graph-rollout.environment.reset-source-digest",
        rollout.environment.reset_source.digest.as_str().as_bytes(),
    );
    append_identity_field(
        material,
        "native-graph-rollout.policy.environment",
        rollout.policy.environment.as_bytes(),
    );
    append_identity_field(
        material,
        "native-graph-rollout.policy.horizon",
        &rollout.policy.horizon.to_le_bytes(),
    );
    append_identity_field(
        material,
        "native-graph-rollout.policy.gamma",
        &rollout.policy.gamma.to_bits().to_le_bytes(),
    );
    append_identity_field(
        material,
        "native-graph-rollout.limits.max-environment-bytes",
        &rollout.limits.max_environment_bytes.to_le_bytes(),
    );
    append_identity_field(
        material,
        "native-graph-rollout.limits.max-horizon",
        &rollout.limits.max_horizon.to_le_bytes(),
    );
}

fn append_environment_protocol_limits_identity(
    material: &mut Vec<u8>,
    limits: &NativeGraphEnvironmentProtocolLimits,
) {
    for (field, value) in [
        ("max-frame-bytes", limits.max_frame_bytes),
        ("max-identifier-bytes", limits.max_identifier_bytes),
        ("max-json-bytes", limits.max_json_bytes),
        ("max-json-depth", limits.max_json_depth),
        ("max-json-array-entries", limits.max_json_array_entries),
        ("max-json-object-entries", limits.max_json_object_entries),
        (
            "max-operation-ledger-entries",
            limits.max_operation_ledger_entries,
        ),
        (
            "max-model-call-lineage-entries",
            limits.max_model_call_lineage_entries,
        ),
        (
            "max-session-model-call-lineage-entries",
            limits.max_session_model_call_lineage_entries,
        ),
        (
            "max-session-model-call-lineage-bytes",
            limits.max_session_model_call_lineage_bytes,
        ),
        ("max-artifact-handles", limits.max_artifact_handles),
        ("max-artifact-bytes", limits.max_artifact_bytes),
    ] {
        append_identity_field(
            material,
            &format!("native-graph-rollout.environment.protocol-limits.{field}"),
            &value.to_le_bytes(),
        );
    }
}

fn append_environment_artifact_limits_identity(
    material: &mut Vec<u8>,
    limits: &NativeGraphEnvironmentArtifactLimits,
) {
    for (field, value) in [
        ("max-artifacts", limits.max_artifacts),
        ("max-total-bytes", limits.max_total_bytes),
        ("max-artifact-bytes", limits.max_artifact_bytes),
        ("max-download-handles", limits.max_download_handles),
    ] {
        append_identity_field(
            material,
            &format!("native-graph-rollout.environment.artifact-limits.{field}"),
            &value.to_le_bytes(),
        );
    }
}

fn append_path_list_identity(material: &mut Vec<u8>, field: &str, paths: &[String]) {
    append_identity_field(
        material,
        &format!("{field}-count"),
        &(paths.len() as u64).to_le_bytes(),
    );
    for path in paths {
        append_identity_field(material, field, path.as_bytes());
    }
}

fn append_optional_digest(material: &mut Vec<u8>, field: &str, digest: Option<&ArtifactDigest>) {
    append_identity_field(
        material,
        &format!("{field}-present"),
        &[u8::from(digest.is_some())],
    );
    if let Some(digest) = digest {
        append_identity_field(material, field, digest.as_str().as_bytes());
    }
}

fn append_optional_id(material: &mut Vec<u8>, field: &str, value: Option<&str>) {
    append_identity_field(
        material,
        &format!("{field}-present"),
        &[u8::from(value.is_some())],
    );
    if let Some(value) = value {
        append_identity_field(material, field, value.as_bytes());
    }
}

fn append_optional_f64(material: &mut Vec<u8>, field: &str, value: Option<f64>) {
    append_identity_field(
        material,
        &format!("{field}-present"),
        &[u8::from(value.is_some())],
    );
    if let Some(value) = value {
        append_identity_field(material, field, &value.to_bits().to_le_bytes());
    }
}

fn append_optional_u32(material: &mut Vec<u8>, field: &str, value: Option<u32>) {
    append_identity_field(
        material,
        &format!("{field}-present"),
        &[u8::from(value.is_some())],
    );
    if let Some(value) = value {
        append_identity_field(material, field, &value.to_le_bytes());
    }
}

fn append_optional_u64(material: &mut Vec<u8>, field: &str, value: Option<u64>) {
    append_identity_field(
        material,
        &format!("{field}-present"),
        &[u8::from(value.is_some())],
    );
    if let Some(value) = value {
        append_identity_field(material, field, &value.to_le_bytes());
    }
}

fn missing(field: &str) -> HarborImportError {
    HarborImportError::InvalidPackage(format!("{field} is required"))
}

fn invalid<T>(message: &str) -> Result<T, HarborImportError> {
    Err(HarborImportError::InvalidPackage(message.to_owned()))
}
