// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

//! Immutable, host-independent execution plans for standard Harbor tasks.

use std::{
    collections::{BTreeMap, BTreeSet},
    net::{IpAddr, Ipv4Addr, Ipv6Addr},
    path::{Component, Path, PathBuf},
    time::Duration,
};

use crate::eval::{ArtifactDigest, EvalExecutionError, VerifierMode};

/// A normalized environment variable binding.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EnvBinding(EnvBindingValue);

#[derive(Clone, Debug, PartialEq, Eq)]
enum EnvBindingValue {
    Literal(String),
    SecretReference(String),
}

impl EnvBinding {
    /// Parses a literal value or one complete `${NAME}` host-secret reference.
    pub fn parse(value: &str) -> Result<Self, String> {
        if !value.contains("${") {
            return Ok(Self(EnvBindingValue::Literal(value.to_owned())));
        }
        let Some(name) = value
            .strip_prefix("${")
            .and_then(|remainder| remainder.strip_suffix('}'))
        else {
            return Err("environment secret reference must be exactly ${NAME}".to_owned());
        };
        if !is_env_name(name) {
            return Err("environment secret reference name is invalid".to_owned());
        }
        Ok(Self(EnvBindingValue::SecretReference(name.to_owned())))
    }

    /// Returns the secret name when this binding is a host-secret reference.
    pub fn secret_reference(&self) -> Option<&str> {
        match &self.0 {
            EnvBindingValue::Literal(_) => None,
            EnvBindingValue::SecretReference(name) => Some(name),
        }
    }

    /// Returns the literal value when this binding is not secret-backed.
    pub fn literal(&self) -> Option<&str> {
        match &self.0 {
            EnvBindingValue::Literal(value) => Some(value),
            EnvBindingValue::SecretReference(_) => None,
        }
    }
}

/// A network policy whose allowlist entries are normalized at import time.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NetworkPolicy(NetworkPolicyKind);

#[derive(Clone, Debug, PartialEq, Eq)]
enum NetworkPolicyKind {
    Public,
    NoNetwork,
    Allowlist(Vec<String>),
}

impl NetworkPolicy {
    /// Returns the provider-defined public benchmark network policy.
    pub const fn public() -> Self {
        Self(NetworkPolicyKind::Public)
    }

    /// Returns the no-outbound-network policy.
    pub const fn no_network() -> Self {
        Self(NetworkPolicyKind::NoNetwork)
    }

    /// Normalizes an allowlist without consulting DNS or the host environment.
    pub fn allowlist<I, S>(entries: I) -> Result<Self, String>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let mut normalized = BTreeSet::new();
        for entry in entries {
            let entry = normalize_allowlist_entry(entry.as_ref())?;
            if !normalized.insert(entry.clone()) {
                return Err(format!(
                    "allowed_hosts contains a duplicate entry {entry:?}"
                ));
            }
        }
        if normalized.is_empty() {
            return Err("allowed_hosts must not be empty for an allowlist network".to_owned());
        }
        Ok(Self(NetworkPolicyKind::Allowlist(
            normalized.into_iter().collect(),
        )))
    }

    /// Returns normalized allowlist entries, if this is an allowlist policy.
    pub fn allowed_hosts(&self) -> Option<&[String]> {
        match &self.0 {
            NetworkPolicyKind::Allowlist(hosts) => Some(hosts),
            NetworkPolicyKind::Public | NetworkPolicyKind::NoNetwork => None,
        }
    }

    /// Reports whether this policy isolates the environment from all networks.
    pub const fn is_no_network(&self) -> bool {
        matches!(self.0, NetworkPolicyKind::NoNetwork)
    }

    /// Reports whether this policy uses the provider's public network lease.
    pub const fn is_public(&self) -> bool {
        matches!(self.0, NetworkPolicyKind::Public)
    }

    fn required_capability(&self) -> &'static str {
        match self.0 {
            NetworkPolicyKind::Public => "public_network",
            NetworkPolicyKind::NoNetwork => "no_network",
            NetworkPolicyKind::Allowlist(_) => "allowlist_egress",
        }
    }
}

/// The immutable image input for a standard task environment.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImageSource {
    kind: ImageSourceKind,
    digest: ArtifactDigest,
}

/// The provenance of an immutable environment image input.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImageSourceKind {
    /// A standard task directory's complete `environment/` build context.
    TaskDockerfile,
    /// A legacy JSON package's immutable environment artifact.
    LegacyArtifact,
}

/// One normalized service name in a generated-main Compose project.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct ComposeServiceName(String);

impl ComposeServiceName {
    pub(crate) fn parse(value: &str) -> Result<Self, String> {
        let trimmed = value.trim();
        if trimmed != value {
            return Err(format!(
                "Compose service name must not contain surrounding whitespace: {value:?}"
            ));
        }
        let value = trimmed;
        let mut bytes = value.bytes();
        if value.len() > 63
            || !matches!(bytes.next(), Some(b'a'..=b'z' | b'0'..=b'9'))
            || !bytes.all(|byte| matches!(byte, b'a'..=b'z' | b'0'..=b'9' | b'_' | b'.' | b'-'))
        {
            return Err(format!(
                "Compose service name must match [a-z0-9][a-z0-9_.-]{{0,62}}: {value:?}"
            ));
        }
        Ok(Self(value.to_owned()))
    }

    pub(crate) fn main() -> Self {
        Self("main".to_owned())
    }

    /// Returns the normalized service name.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Immutable authored-sidecar metadata for one generated-main Compose project.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ComposeProjectPlan {
    pub(crate) definition_path: String,
    pub(crate) services: BTreeSet<ComposeServiceName>,
    pub(crate) build_timeout: Duration,
    pub(crate) startup_timeout: Duration,
}

impl ComposeProjectPlan {
    /// Returns the exact owned Compose overlay path.
    pub fn definition_path(&self) -> &str {
        &self.definition_path
    }

    /// Returns the sorted normalized service set, including generated `main`.
    pub fn services(&self) -> &BTreeSet<ComposeServiceName> {
        &self.services
    }

    /// Returns the complete project build deadline.
    pub const fn build_timeout(&self) -> Duration {
        self.build_timeout
    }

    /// Returns the complete project startup deadline.
    pub const fn startup_timeout(&self) -> Duration {
        self.startup_timeout
    }
}

impl ImageSource {
    pub(crate) fn task_dockerfile(build_context_digest: ArtifactDigest) -> Self {
        Self {
            kind: ImageSourceKind::TaskDockerfile,
            digest: build_context_digest,
        }
    }

    pub(crate) fn legacy_artifact(digest: ArtifactDigest) -> Self {
        Self {
            kind: ImageSourceKind::LegacyArtifact,
            digest,
        }
    }

    /// Returns the source provenance without exposing construction internals.
    pub const fn kind(&self) -> ImageSourceKind {
        self.kind
    }

    /// Returns the immutable source digest.
    pub fn digest(&self) -> &ArtifactDigest {
        &self.digest
    }

    /// Returns the standard task's complete `environment/` build-context identity.
    ///
    /// The method name is retained for API compatibility; the digest binds every
    /// selected build-context entry, including empty directories and file modes.
    pub fn dockerfile_digest(&self) -> Option<&ArtifactDigest> {
        (self.kind == ImageSourceKind::TaskDockerfile).then_some(&self.digest)
    }
}

/// CPU and memory limits that must be authored together.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ContainerResources {
    pub(crate) cpus: u64,
    pub(crate) memory_mb: u64,
}

impl ContainerResources {
    /// Returns the whole-CPU limit.
    pub const fn cpus(&self) -> u64 {
        self.cpus
    }

    /// Returns the memory limit in MiB.
    pub const fn memory_mb(&self) -> u64 {
        self.memory_mb
    }
}

/// Readiness behavior for an environment before its agent phase begins.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HealthcheckPlan {
    pub(crate) command: Vec<String>,
    pub(crate) start_period: Option<Duration>,
    pub(crate) start_interval: Option<Duration>,
    pub(crate) interval: Option<Duration>,
    pub(crate) timeout: Option<Duration>,
    pub(crate) retries: Option<u32>,
}

impl HealthcheckPlan {
    /// Returns the readiness command argv.
    pub fn command(&self) -> &[String] {
        &self.command
    }

    /// Returns the initial grace period.
    pub const fn start_period(&self) -> Option<Duration> {
        self.start_period
    }

    /// Returns the initial poll interval.
    pub const fn start_interval(&self) -> Option<Duration> {
        self.start_interval
    }

    /// Returns the steady-state poll interval.
    pub const fn interval(&self) -> Option<Duration> {
        self.interval
    }

    /// Returns the per-attempt readiness deadline.
    pub const fn timeout(&self) -> Option<Duration> {
        self.timeout
    }

    /// Returns allowed consecutive failures after the grace period.
    pub const fn retries(&self) -> Option<u32> {
        self.retries
    }
}

/// One immutable environment baseline.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EnvironmentPlan {
    pub(crate) image_source: ImageSource,
    pub(crate) resources: Option<ContainerResources>,
    pub(crate) workdir: Option<String>,
    pub(crate) user: Option<String>,
    pub(crate) env: BTreeMap<String, EnvBinding>,
    pub(crate) network: NetworkPolicy,
    pub(crate) healthcheck: Option<HealthcheckPlan>,
}

impl EnvironmentPlan {
    /// Returns the image input used to create this environment.
    pub fn image_source(&self) -> &ImageSource {
        &self.image_source
    }

    /// Returns authored resource limits.
    pub const fn resources(&self) -> Option<ContainerResources> {
        self.resources
    }

    /// Returns the authored absolute workdir, if any.
    pub fn workdir(&self) -> Option<&str> {
        self.workdir.as_deref()
    }

    /// Returns the authored effective user, if any.
    pub fn user(&self) -> Option<&str> {
        self.user.as_deref()
    }

    /// Returns baseline environment bindings ordered by variable name.
    pub fn env(&self) -> &BTreeMap<String, EnvBinding> {
        &self.env
    }

    /// Returns the baseline network policy.
    pub fn network(&self) -> &NetworkPolicy {
        &self.network
    }

    /// Returns optional readiness behavior.
    pub fn healthcheck(&self) -> Option<&HealthcheckPlan> {
        self.healthcheck.as_ref()
    }
}

/// Phase-specific execution settings.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PhasePlan {
    pub(crate) user: Option<String>,
    pub(crate) env: BTreeMap<String, EnvBinding>,
    pub(crate) network: NetworkPolicy,
    pub(crate) timeout: Option<Duration>,
}

impl PhasePlan {
    /// Returns the phase-specific user override, if any.
    pub fn user(&self) -> Option<&str> {
        self.user.as_deref()
    }

    /// Returns phase bindings that override baseline bindings by name.
    pub fn env(&self) -> &BTreeMap<String, EnvBinding> {
        &self.env
    }

    /// Returns the fully resolved phase network policy.
    pub fn network(&self) -> &NetworkPolicy {
        &self.network
    }

    /// Returns the optional phase deadline.
    pub const fn timeout(&self) -> Option<Duration> {
        self.timeout
    }
}

/// The verifier topology and its immutable environment.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifierPlan {
    pub(crate) phase: PhasePlan,
    pub(crate) mode: VerifierMode,
    pub(crate) environment: EnvironmentPlan,
}

/// The reward aggregation rule for a benchmark with explicit steps.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MultiStepRewardStrategy {
    /// Average every completed step reward.
    Mean,
    /// Use the final completed step reward.
    Final,
}

/// One fully resolved immutable benchmark step.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BenchmarkStepPlan {
    name: String,
    instruction: String,
    verifier_test_root: String,
    agent: PhasePlan,
    verifier: VerifierPlan,
    artifacts: Vec<ArtifactSpec>,
    collect_hooks: Vec<VerifierCollectHook>,
    collection_timeout: Duration,
}

impl BenchmarkStepPlan {
    pub(crate) fn new(
        name: String,
        instruction: String,
        verifier_test_root: String,
        agent: PhasePlan,
        verifier: VerifierPlan,
        artifacts: Vec<ArtifactSpec>,
    ) -> Self {
        Self {
            name,
            instruction,
            verifier_test_root,
            agent,
            verifier,
            artifacts,
            collect_hooks: Vec::new(),
            collection_timeout: Duration::from_secs(120),
        }
    }

    pub(crate) fn with_collection(
        mut self,
        collect_hooks: Vec<VerifierCollectHook>,
        collection_timeout: Duration,
    ) -> Self {
        self.collect_hooks = collect_hooks;
        self.collection_timeout = collection_timeout;
        self
    }

    /// Returns the authored step name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the step-specific agent instruction.
    pub fn instruction(&self) -> &str {
        &self.instruction
    }

    /// Returns the selected verifier test directory relative to the task root.
    pub fn verifier_test_root(&self) -> &str {
        &self.verifier_test_root
    }

    /// Returns fully resolved agent phase settings for this step.
    pub fn agent(&self) -> &PhasePlan {
        &self.agent
    }

    /// Returns fully resolved verifier settings for this step.
    pub fn verifier(&self) -> &VerifierPlan {
        &self.verifier
    }

    /// Returns effective artifact declarations in collection order.
    pub fn artifacts(&self) -> &[ArtifactSpec] {
        &self.artifacts
    }

    /// Returns effective collection hooks in root-then-step authored order.
    pub fn collect_hooks(&self) -> &[VerifierCollectHook] {
        &self.collect_hooks
    }

    /// Returns the deadline for all collection work in this step.
    pub const fn collection_timeout(&self) -> Duration {
        self.collection_timeout
    }
}

impl VerifierPlan {
    /// Returns phase-only verifier settings.
    pub fn phase(&self) -> &PhasePlan {
        &self.phase
    }

    /// Returns shared or separate verifier topology.
    pub const fn mode(&self) -> VerifierMode {
        self.mode
    }

    /// Returns the verifier's immutable baseline environment.
    pub fn environment(&self) -> &EnvironmentPlan {
        &self.environment
    }
}

/// One declared benchmark artifact.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ArtifactSpec(ArtifactSpecKind);

/// One exact argv hook executed against a task service before artifact collection.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifierCollectHook {
    pub(crate) service: ComposeServiceName,
    pub(crate) command: Vec<String>,
    pub(crate) timeout: Duration,
    pub(crate) user: Option<String>,
}

impl VerifierCollectHook {
    /// Returns the target task service.
    pub fn service(&self) -> &ComposeServiceName {
        &self.service
    }

    /// Returns the exact command argv without an implicit shell.
    pub fn command(&self) -> &[String] {
        &self.command
    }

    /// Returns the hook deadline.
    pub const fn timeout(&self) -> Duration {
        self.timeout
    }

    /// Returns the optional container user.
    pub fn user(&self) -> Option<&str> {
        self.user.as_deref()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum ArtifactSpecKind {
    ExactFile {
        source: String,
        service: ComposeServiceName,
    },
    Collected {
        source: String,
        destination: Option<String>,
        exclude: Vec<String>,
        service: ComposeServiceName,
    },
}

impl ArtifactSpec {
    pub(crate) fn exact_file(source: String) -> Self {
        Self::exact_file_for_service(source, ComposeServiceName::main())
    }

    pub(crate) fn exact_file_for_service(source: String, service: ComposeServiceName) -> Self {
        Self(ArtifactSpecKind::ExactFile { source, service })
    }

    pub(crate) fn collected(
        source: String,
        destination: Option<String>,
        exclude: Vec<String>,
    ) -> Self {
        Self::collected_for_service(source, destination, exclude, ComposeServiceName::main())
    }

    pub(crate) fn collected_for_service(
        source: String,
        destination: Option<String>,
        exclude: Vec<String>,
        service: ComposeServiceName,
    ) -> Self {
        Self(ArtifactSpecKind::Collected {
            source,
            destination,
            exclude,
            service,
        })
    }

    /// Returns the validated absolute container source path.
    pub fn source(&self) -> &str {
        match &self.0 {
            ArtifactSpecKind::ExactFile { source, .. }
            | ArtifactSpecKind::Collected { source, .. } => source,
        }
    }

    /// Returns the normalized task service that owns the source path.
    pub fn service(&self) -> &str {
        match &self.0 {
            ArtifactSpecKind::ExactFile { service, .. }
            | ArtifactSpecKind::Collected { service, .. } => service.as_str(),
        }
    }

    /// Reports whether this declaration names exactly one regular file.
    pub const fn is_exact_file(&self) -> bool {
        matches!(self.0, ArtifactSpecKind::ExactFile { .. })
    }

    /// Returns the optional relative collection destination.
    pub fn destination(&self) -> Option<&str> {
        match &self.0 {
            ArtifactSpecKind::ExactFile { .. } => None,
            ArtifactSpecKind::Collected { destination, .. } => destination.as_deref(),
        }
    }

    /// Returns structured-declaration exclusion patterns in sorted unique order.
    pub fn exclude(&self) -> &[String] {
        match &self.0 {
            ArtifactSpecKind::ExactFile { .. } => &[],
            ArtifactSpecKind::Collected { exclude, .. } => exclude,
        }
    }

    pub(crate) fn output_target(&self) -> String {
        self.destination().map(str::to_owned).unwrap_or_else(|| {
            Path::new(self.source())
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or_default()
                .to_owned()
        })
    }
}

/// Declarative guarantees supplied by an execution provider.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ProviderCapabilities {
    docker: bool,
    image_source: bool,
    resource_limits: bool,
    users: bool,
    phase_env: bool,
    workdir: bool,
    phase_timeouts: bool,
    separate_verifier: bool,
    healthchecks: bool,
    no_network: bool,
    public_network: bool,
    allowlist_egress: bool,
    compose_project: bool,
    compose_config: bool,
    service_exec: bool,
    service_archive: bool,
    service_stop: bool,
}

impl ProviderCapabilities {
    /// Returns a provider which guarantees no benchmark-execution properties.
    pub const fn none() -> Self {
        Self {
            docker: false,
            image_source: false,
            resource_limits: false,
            users: false,
            phase_env: false,
            workdir: false,
            phase_timeouts: false,
            separate_verifier: false,
            healthchecks: false,
            no_network: false,
            public_network: false,
            allowlist_egress: false,
            compose_project: false,
            compose_config: false,
            service_exec: false,
            service_archive: false,
            service_stop: false,
        }
    }

    /// Declares Docker-backed execution support.
    pub const fn with_docker(mut self) -> Self {
        self.docker = true;
        self
    }

    /// Declares support for the normalized immutable image source.
    pub const fn with_image_source(mut self) -> Self {
        self.image_source = true;
        self
    }

    /// Declares resource-limit enforcement.
    pub const fn with_resource_limits(mut self) -> Self {
        self.resource_limits = true;
        self
    }

    /// Declares effective-user enforcement.
    pub const fn with_users(mut self) -> Self {
        self.users = true;
        self
    }

    /// Declares phase environment binding enforcement.
    pub const fn with_phase_env(mut self) -> Self {
        self.phase_env = true;
        self
    }

    /// Declares effective working-directory enforcement.
    pub const fn with_workdir(mut self) -> Self {
        self.workdir = true;
        self
    }

    /// Declares command-phase deadline enforcement.
    pub const fn with_phase_timeouts(mut self) -> Self {
        self.phase_timeouts = true;
        self
    }

    /// Declares isolated separate-verifier environment enforcement.
    pub const fn with_separate_verifier(mut self) -> Self {
        self.separate_verifier = true;
        self
    }

    /// Declares readiness healthcheck enforcement.
    pub const fn with_healthchecks(mut self) -> Self {
        self.healthchecks = true;
        self
    }

    /// Declares no-network enforcement.
    pub const fn with_no_network(mut self) -> Self {
        self.no_network = true;
        self
    }

    /// Declares provider-defined public networking.
    pub const fn with_public_network(mut self) -> Self {
        self.public_network = true;
        self
    }

    /// Declares mediated allowlist egress enforcement.
    pub const fn with_allowlist_egress(mut self) -> Self {
        self.allowlist_egress = true;
        self
    }

    /// Declares task-owned Docker Compose project support.
    pub const fn with_compose_project(mut self) -> Self {
        self.compose_project = true;
        self
    }

    /// Declares read-only Docker Compose configuration support.
    pub const fn with_compose_config(mut self) -> Self {
        self.compose_config = true;
        self
    }

    /// Declares service-targeted command execution support.
    pub const fn with_service_exec(mut self) -> Self {
        self.service_exec = true;
        self
    }

    /// Declares service-targeted archive collection support.
    pub const fn with_service_archive(mut self) -> Self {
        self.service_archive = true;
        self
    }

    /// Declares service-targeted stop support.
    pub const fn with_service_stop(mut self) -> Self {
        self.service_stop = true;
        self
    }

    fn supports(self, capability: &str) -> bool {
        match capability {
            "docker" => self.docker,
            "image_source" => self.image_source,
            "resource_limits" => self.resource_limits,
            "users" => self.users,
            "phase_env" => self.phase_env,
            "workdir" => self.workdir,
            "phase_timeouts" => self.phase_timeouts,
            "separate_verifier" => self.separate_verifier,
            "healthchecks" => self.healthchecks,
            "no_network" => self.no_network,
            "public_network" => self.public_network,
            "allowlist_egress" => self.allowlist_egress,
            "compose_project" => self.compose_project,
            "compose_config" => self.compose_config,
            "service_exec" => self.service_exec,
            "service_archive" => self.service_archive,
            "service_stop" => self.service_stop,
            _ => false,
        }
    }
}

/// Immutable execution intent compiled from one standard task manifest.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BenchmarkExecutionPlan {
    pub(crate) environment: EnvironmentPlan,
    pub(crate) agent: PhasePlan,
    pub(crate) verifier: VerifierPlan,
    pub(crate) artifacts: Vec<ArtifactSpec>,
    pub(crate) compose: Option<ComposeProjectPlan>,
    pub(crate) steps: Vec<BenchmarkStepPlan>,
    pub(crate) has_explicit_steps: bool,
    pub(crate) multi_step_reward_strategy: Option<MultiStepRewardStrategy>,
}

impl BenchmarkExecutionPlan {
    /// Returns the environment baseline used to build and start the task.
    pub fn environment(&self) -> &EnvironmentPlan {
        &self.environment
    }

    /// Returns agent phase settings.
    pub fn agent(&self) -> &PhasePlan {
        &self.agent
    }

    /// Returns verifier topology, phase settings, and environment.
    pub fn verifier(&self) -> &VerifierPlan {
        &self.verifier
    }

    /// Returns artifact declarations in authored order.
    pub fn artifacts(&self) -> &[ArtifactSpec] {
        &self.artifacts
    }

    /// Returns generated-main Compose metadata for a Compose-backed standard task.
    pub fn compose(&self) -> Option<&ComposeProjectPlan> {
        self.compose.as_ref()
    }

    /// Returns resolved benchmark steps in authored order.
    pub fn steps(&self) -> &[BenchmarkStepPlan] {
        &self.steps
    }

    /// Reports whether the package authored an explicit step layout.
    pub fn is_multi_step(&self) -> bool {
        self.has_explicit_steps
    }

    pub(crate) fn uses_shared_verifier(&self) -> bool {
        self.steps
            .iter()
            .any(|step| step.verifier().mode() == VerifierMode::Shared)
    }

    /// Returns the authored reward aggregation rule for explicit multi-step tasks.
    pub const fn multi_step_reward_strategy(&self) -> Option<MultiStepRewardStrategy> {
        self.multi_step_reward_strategy
    }

    pub(crate) fn append_identity_material(&self, material: &mut Vec<u8>) {
        let mut encoder = IdentityEncoder { material };
        let has_compose = self.compose.is_some();
        encoder.field(
            "execution-plan.format",
            if has_compose { b"2" } else { b"1" },
        );
        append_environment_identity(&mut encoder, &self.environment);
        append_phase_identity(&mut encoder, &self.agent);
        append_verifier_identity(&mut encoder, &self.verifier);
        append_artifacts_identity(&mut encoder, &self.artifacts, has_compose);
        if has_compose {
            append_compose_identity(&mut encoder, self.compose.as_ref());
        }
        encoder.bool("execution-plan.has-explicit-steps", self.has_explicit_steps);
        encoder.field(
            "execution-plan.reward-strategy",
            match self.multi_step_reward_strategy {
                None => b"none",
                Some(MultiStepRewardStrategy::Mean) => b"mean",
                Some(MultiStepRewardStrategy::Final) => b"final",
            },
        );
        encoder.usize("execution-plan.step-count", self.steps.len());
        for step in &self.steps {
            encoder.field("step.name", step.name.as_bytes());
            encoder.field("step.instruction", step.instruction.as_bytes());
            encoder.field(
                "step.verifier-test-root",
                step.verifier_test_root.as_bytes(),
            );
            append_phase_identity(&mut encoder, &step.agent);
            append_verifier_identity(&mut encoder, &step.verifier);
            append_artifacts_identity(&mut encoder, &step.artifacts, has_compose);
            if has_compose {
                append_collect_hooks_identity(&mut encoder, &step.collect_hooks);
                encoder.duration("step.collection-timeout", step.collection_timeout);
            }
        }
    }

    /// Refuses execution when a provider cannot enforce every authored guarantee.
    pub fn validate_for(
        &self,
        capabilities: ProviderCapabilities,
    ) -> Result<(), EvalExecutionError> {
        require(capabilities, "docker")?;
        require(capabilities, "image_source")?;
        if self.compose.is_some() {
            require(capabilities, "compose_project")?;
            require(capabilities, "compose_config")?;
            require(capabilities, "service_exec")?;
            require(capabilities, "service_archive")?;
            require(capabilities, "service_stop")?;
            require(capabilities, "public_network")?;
        }
        for environment in std::iter::once(&self.environment)
            .chain(self.steps.iter().map(|step| step.verifier.environment()))
        {
            if environment.resources.is_some() {
                require(capabilities, "resource_limits")?;
            }
            if environment.user.is_some() {
                require(capabilities, "users")?;
            }
            if !environment.env.is_empty() {
                require(capabilities, "phase_env")?;
            }
            if environment.workdir.is_some() {
                require(capabilities, "workdir")?;
            }
            if environment.healthcheck.is_some() {
                require(capabilities, "healthchecks")?;
            }
        }
        for phase in self
            .steps
            .iter()
            .flat_map(|step| [step.agent(), step.verifier().phase()].into_iter())
        {
            if phase.user.is_some() {
                require(capabilities, "users")?;
            }
            if !phase.env.is_empty() {
                require(capabilities, "phase_env")?;
            }
            if phase.timeout.is_some() {
                require(capabilities, "phase_timeouts")?;
            }
        }
        if self
            .steps
            .iter()
            .any(|step| step.verifier().mode() == VerifierMode::Separate)
        {
            require(capabilities, "separate_verifier")?;
        }
        require_network(capabilities, &self.environment.network)?;
        for step in &self.steps {
            require_network(capabilities, step.agent().network())?;
            require_network(capabilities, step.verifier().environment().network())?;
            require_network(capabilities, step.verifier().phase().network())?;
        }
        Ok(())
    }
}

pub(crate) struct CanonicalPackagePlan<'a> {
    task_id: &'a str,
    agent_command: &'a [String],
    verifier_command: &'a [String],
    execution_plan: &'a BenchmarkExecutionPlan,
}

impl<'a> CanonicalPackagePlan<'a> {
    pub(crate) fn new(
        task_id: &'a str,
        agent_command: &'a [String],
        verifier_command: &'a [String],
        execution_plan: &'a BenchmarkExecutionPlan,
    ) -> Self {
        Self {
            task_id,
            agent_command,
            verifier_command,
            execution_plan,
        }
    }

    pub(crate) fn digest(&self) -> ArtifactDigest {
        let mut material = Vec::new();
        {
            let mut encoder = IdentityEncoder {
                material: &mut material,
            };
            encoder.field("canonical-package-plan.format", b"2");
            encoder.field("canonical-package-plan.task-id", self.task_id.as_bytes());
            append_command_identity(
                &mut encoder,
                "canonical-package-plan.agent-command-count",
                "canonical-package-plan.agent-command-argument",
                self.agent_command,
            );
            append_command_identity(
                &mut encoder,
                "canonical-package-plan.verifier-command-count",
                "canonical-package-plan.verifier-command-argument",
                self.verifier_command,
            );
        }
        self.execution_plan.append_identity_material(&mut material);
        ArtifactDigest::from_bytes(&material)
    }
}

struct IdentityEncoder<'a> {
    material: &'a mut Vec<u8>,
}

impl IdentityEncoder<'_> {
    fn field(&mut self, tag: &str, value: &[u8]) {
        append_identity_field(self.material, tag, value);
    }

    fn bool(&mut self, tag: &str, value: bool) {
        self.field(tag, &[u8::from(value)]);
    }

    fn u32(&mut self, tag: &str, value: u32) {
        self.field(tag, &value.to_le_bytes());
    }

    fn u64(&mut self, tag: &str, value: u64) {
        self.field(tag, &value.to_le_bytes());
    }

    fn u128(&mut self, tag: &str, value: u128) {
        self.field(tag, &value.to_le_bytes());
    }

    fn usize(&mut self, tag: &str, value: usize) {
        self.u64(tag, value as u64);
    }

    fn optional_str(&mut self, tag: &str, value: Option<&str>) {
        self.bool(tag, value.is_some());
        if let Some(value) = value {
            self.field(tag, value.as_bytes());
        }
    }

    fn optional_duration(&mut self, tag: &str, value: Option<Duration>) {
        self.bool(tag, value.is_some());
        if let Some(value) = value {
            self.u64(tag, value.as_secs());
            self.u32(tag, value.subsec_nanos());
        }
    }

    fn duration(&mut self, tag: &str, value: Duration) {
        self.u128(tag, value.as_nanos());
    }

    fn optional_u32(&mut self, tag: &str, value: Option<u32>) {
        self.bool(tag, value.is_some());
        if let Some(value) = value {
            self.u32(tag, value);
        }
    }
}

pub(crate) fn append_identity_field(material: &mut Vec<u8>, tag: &str, value: &[u8]) {
    material.extend_from_slice(&(tag.len() as u64).to_le_bytes());
    material.extend_from_slice(tag.as_bytes());
    material.extend_from_slice(&(value.len() as u64).to_le_bytes());
    material.extend_from_slice(value);
}

fn append_environment_identity(encoder: &mut IdentityEncoder<'_>, environment: &EnvironmentPlan) {
    encoder.field(
        "environment.image-kind",
        match environment.image_source.kind {
            ImageSourceKind::TaskDockerfile => b"task-dockerfile",
            ImageSourceKind::LegacyArtifact => b"legacy-artifact",
        },
    );
    encoder.field(
        "environment.image-digest",
        environment.image_source.digest.as_str().as_bytes(),
    );
    encoder.bool("environment.resources", environment.resources.is_some());
    if let Some(resources) = environment.resources {
        encoder.u64("environment.resources.cpus", resources.cpus);
        encoder.u64("environment.resources.memory-mb", resources.memory_mb);
    }
    encoder.optional_str("environment.workdir", environment.workdir.as_deref());
    encoder.optional_str("environment.user", environment.user.as_deref());
    append_env_identity(encoder, &environment.env);
    append_network_identity(encoder, &environment.network);
    encoder.bool("environment.healthcheck", environment.healthcheck.is_some());
    if let Some(healthcheck) = &environment.healthcheck {
        encoder.usize("healthcheck.command-count", healthcheck.command.len());
        for part in &healthcheck.command {
            encoder.field("healthcheck.command", part.as_bytes());
        }
        encoder.optional_duration("healthcheck.start-period", healthcheck.start_period);
        encoder.optional_duration("healthcheck.start-interval", healthcheck.start_interval);
        encoder.optional_duration("healthcheck.interval", healthcheck.interval);
        encoder.optional_duration("healthcheck.timeout", healthcheck.timeout);
        encoder.optional_u32("healthcheck.retries", healthcheck.retries);
    }
}

fn append_phase_identity(encoder: &mut IdentityEncoder<'_>, phase: &PhasePlan) {
    encoder.optional_str("phase.user", phase.user.as_deref());
    append_env_identity(encoder, &phase.env);
    append_network_identity(encoder, &phase.network);
    encoder.optional_duration("phase.timeout", phase.timeout);
}

fn append_verifier_identity(encoder: &mut IdentityEncoder<'_>, verifier: &VerifierPlan) {
    encoder.field(
        "verifier.mode",
        match verifier.mode {
            VerifierMode::Shared => b"shared",
            VerifierMode::Separate => b"separate",
        },
    );
    append_environment_identity(encoder, &verifier.environment);
    append_phase_identity(encoder, &verifier.phase);
}

fn append_env_identity(encoder: &mut IdentityEncoder<'_>, bindings: &BTreeMap<String, EnvBinding>) {
    encoder.usize("environment-binding.count", bindings.len());
    for (name, binding) in bindings {
        encoder.field("environment-binding.name", name.as_bytes());
        match &binding.0 {
            EnvBindingValue::Literal(value) => {
                encoder.field("environment-binding.kind", b"literal");
                encoder.field("environment-binding.value", value.as_bytes());
            }
            EnvBindingValue::SecretReference(name) => {
                encoder.field("environment-binding.kind", b"secret-reference");
                encoder.field("environment-binding.value", name.as_bytes());
            }
        }
    }
}

fn append_network_identity(encoder: &mut IdentityEncoder<'_>, network: &NetworkPolicy) {
    match &network.0 {
        NetworkPolicyKind::Public => {
            encoder.field("network.kind", b"public");
            encoder.usize("network.allowed-host-count", 0);
        }
        NetworkPolicyKind::NoNetwork => {
            encoder.field("network.kind", b"no-network");
            encoder.usize("network.allowed-host-count", 0);
        }
        NetworkPolicyKind::Allowlist(hosts) => {
            encoder.field("network.kind", b"allowlist");
            encoder.usize("network.allowed-host-count", hosts.len());
            for host in hosts {
                encoder.field("network.allowed-host", host.as_bytes());
            }
        }
    }
}

fn append_artifacts_identity(
    encoder: &mut IdentityEncoder<'_>,
    artifacts: &[ArtifactSpec],
    has_service_identity: bool,
) {
    encoder.usize("artifact.count", artifacts.len());
    for artifact in artifacts {
        match &artifact.0 {
            ArtifactSpecKind::ExactFile { source, service } => {
                encoder.field("artifact.kind", b"exact-file");
                if has_service_identity {
                    encoder.field("artifact.service", service.as_str().as_bytes());
                }
                encoder.field("artifact.source", source.as_bytes());
                encoder.bool("artifact.destination", false);
                encoder.usize("artifact.exclude-count", 0);
            }
            ArtifactSpecKind::Collected {
                source,
                destination,
                exclude,
                service,
            } => {
                encoder.field("artifact.kind", b"collected");
                if has_service_identity {
                    encoder.field("artifact.service", service.as_str().as_bytes());
                }
                encoder.field("artifact.source", source.as_bytes());
                encoder.optional_str("artifact.destination", destination.as_deref());
                encoder.usize("artifact.exclude-count", exclude.len());
                for pattern in exclude {
                    encoder.field("artifact.exclude", pattern.as_bytes());
                }
            }
        }
    }
}

fn append_compose_identity(
    encoder: &mut IdentityEncoder<'_>,
    compose: Option<&ComposeProjectPlan>,
) {
    encoder.bool("compose.present", compose.is_some());
    if let Some(compose) = compose {
        encoder.field(
            "compose.definition-path",
            compose.definition_path.as_bytes(),
        );
        encoder.usize("compose.service-count", compose.services.len());
        for service in &compose.services {
            encoder.field("compose.service", service.as_str().as_bytes());
        }
        encoder.duration("compose.build-timeout", compose.build_timeout);
        encoder.duration("compose.startup-timeout", compose.startup_timeout);
    }
}

fn append_collect_hooks_identity(encoder: &mut IdentityEncoder<'_>, hooks: &[VerifierCollectHook]) {
    encoder.usize("collect-hook.count", hooks.len());
    for hook in hooks {
        encoder.field("collect-hook.service", hook.service.as_str().as_bytes());
        encoder.usize("collect-hook.command-count", hook.command.len());
        for argument in &hook.command {
            encoder.field("collect-hook.command", argument.as_bytes());
        }
        encoder.duration("collect-hook.timeout", hook.timeout);
        encoder.optional_str("collect-hook.user", hook.user.as_deref());
    }
}

fn append_command_identity(
    encoder: &mut IdentityEncoder<'_>,
    count_tag: &str,
    argument_tag: &str,
    command: &[String],
) {
    encoder.usize(count_tag, command.len());
    for argument in command {
        encoder.field(argument_tag, argument.as_bytes());
    }
}

const RESERVED_VERIFIER_PATHS: [&str; 2] = ["/tests", "/logs/verifier"];

pub(crate) fn artifact_source_overlaps_reserved_verifier_path(artifact: &ArtifactSpec) -> bool {
    path_overlaps_reserved_verifier_path(Path::new(artifact.source()))
}

pub(crate) fn verifier_artifact_target_collision(
    workdir: &str,
    artifacts: &[ArtifactSpec],
) -> Result<Option<String>, String> {
    let workdir = normalize_absolute_container_path(workdir)?;
    for artifact in artifacts {
        let target = workdir.join(artifact.output_target());
        if path_overlaps_reserved_verifier_path(&target) {
            return Ok(Some(target.to_string_lossy().into_owned()));
        }
    }
    Ok(None)
}

pub(crate) fn shared_workdir_conflicts_reserved_verifier_path(
    workdir: &str,
) -> Result<bool, String> {
    let workdir = normalize_absolute_container_path(workdir)?;
    Ok(RESERVED_VERIFIER_PATHS
        .iter()
        .any(|reserved| workdir.starts_with(Path::new(reserved))))
}

fn normalize_absolute_container_path(path: &str) -> Result<PathBuf, String> {
    let path = Path::new(path);
    if !path.is_absolute() {
        return Err(format!(
            "container workdir must be absolute: {}",
            path.display()
        ));
    }
    let mut normalized = PathBuf::from("/");
    for component in path.components() {
        match component {
            Component::RootDir => {}
            Component::Normal(component) => normalized.push(component),
            Component::CurDir | Component::ParentDir | Component::Prefix(_) => {
                return Err(format!(
                    "container workdir must be an isolated path: {}",
                    path.display()
                ));
            }
        }
    }
    Ok(normalized)
}

fn path_overlaps_reserved_verifier_path(path: &Path) -> bool {
    RESERVED_VERIFIER_PATHS.iter().any(|reserved| {
        let reserved = Path::new(reserved);
        path.starts_with(reserved) || reserved.starts_with(path)
    })
}

fn require(
    capabilities: ProviderCapabilities,
    capability: &'static str,
) -> Result<(), EvalExecutionError> {
    if capabilities.supports(capability) {
        Ok(())
    } else {
        Err(EvalExecutionError::UnsupportedEnforcement(capability))
    }
}

fn require_network(
    capabilities: ProviderCapabilities,
    network: &NetworkPolicy,
) -> Result<(), EvalExecutionError> {
    require(capabilities, network.required_capability())
}

pub(crate) fn validate_env_name(name: &str) -> Result<(), String> {
    if is_env_name(name) {
        Ok(())
    } else {
        Err(format!("environment variable name is invalid: {name:?}"))
    }
}

pub(crate) fn validate_user(user: &str) -> Result<(), String> {
    if user.is_empty() || user.contains(char::is_whitespace) {
        return Err("container user must not be empty or contain whitespace".to_owned());
    }
    let mut parts = user.split(':');
    let first = parts.next().unwrap_or_default();
    let second = parts.next();
    if parts.next().is_some()
        || !is_user_component(first)
        || second.is_some_and(|part| !is_user_component(part))
    {
        return Err(format!("container user is invalid: {user:?}"));
    }
    Ok(())
}

fn is_env_name(name: &str) -> bool {
    let mut bytes = name.bytes();
    matches!(bytes.next(), Some(b'A'..=b'Z' | b'a'..=b'z' | b'_'))
        && bytes.all(|byte| matches!(byte, b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'_'))
}

fn is_user_component(component: &str) -> bool {
    !component.is_empty()
        && component
            .bytes()
            .all(|byte| matches!(byte, b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'_' | b'-'))
}

fn normalize_allowlist_entry(entry: &str) -> Result<String, String> {
    if entry.is_empty() || entry.trim() != entry {
        return Err(format!("allowed_hosts entry is invalid: {entry:?}"));
    }
    if entry.contains('/') {
        return normalize_cidr(entry);
    }
    if let Ok(address) = entry.parse::<IpAddr>() {
        return Ok(address.to_string());
    }
    let (wildcard, hostname) = entry
        .strip_prefix("*.")
        .map_or((false, entry), |hostname| (true, hostname));
    if !is_dns_name(hostname) {
        return Err(format!("allowed_hosts entry is invalid: {entry:?}"));
    }
    Ok(format!("{}{hostname}", if wildcard { "*." } else { "" }).to_ascii_lowercase())
}

fn is_dns_name(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= 253
        && name.split('.').all(|label| {
            !label.is_empty()
                && label.len() <= 63
                && !label.starts_with('-')
                && !label.ends_with('-')
                && label
                    .bytes()
                    .all(|byte| matches!(byte, b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-'))
        })
}

fn normalize_cidr(entry: &str) -> Result<String, String> {
    let Some((address, prefix)) = entry.split_once('/') else {
        return Err(format!("allowed_hosts CIDR is invalid: {entry:?}"));
    };
    if prefix.contains('/') {
        return Err(format!("allowed_hosts CIDR is invalid: {entry:?}"));
    }
    let prefix = prefix
        .parse::<u8>()
        .map_err(|_| format!("allowed_hosts CIDR is invalid: {entry:?}"))?;
    match address.parse::<IpAddr>() {
        Ok(IpAddr::V4(address)) if prefix <= 32 => {
            let bits = u32::from(address);
            let mask = if prefix == 0 {
                0
            } else {
                u32::MAX << (32 - prefix)
            };
            Ok(format!("{}/{}", Ipv4Addr::from(bits & mask), prefix))
        }
        Ok(IpAddr::V6(address)) if prefix <= 128 => {
            let bits = u128::from(address);
            let mask = if prefix == 0 {
                0
            } else {
                u128::MAX << (128 - prefix)
            };
            Ok(format!("{}/{}", Ipv6Addr::from(bits & mask), prefix))
        }
        _ => Err(format!("allowed_hosts CIDR is invalid: {entry:?}")),
    }
}
