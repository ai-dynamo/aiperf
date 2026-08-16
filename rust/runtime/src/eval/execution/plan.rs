// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

//! Immutable, host-independent execution plans for standard Harbor tasks.

use std::{
    collections::{BTreeMap, BTreeSet},
    net::{IpAddr, Ipv4Addr, Ipv6Addr},
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
    /// A standard task directory's `environment/Dockerfile`.
    TaskDockerfile,
    /// A legacy JSON package's immutable environment artifact.
    LegacyArtifact,
}

impl ImageSource {
    pub(crate) fn task_dockerfile(dockerfile_digest: ArtifactDigest) -> Self {
        Self {
            kind: ImageSourceKind::TaskDockerfile,
            digest: dockerfile_digest,
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

    /// Returns the digest of the standard task's `environment/Dockerfile`.
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

#[derive(Clone, Debug, PartialEq, Eq)]
enum ArtifactSpecKind {
    ExactFile {
        source: String,
    },
    Collected {
        source: String,
        destination: Option<String>,
        exclude: Vec<String>,
    },
}

impl ArtifactSpec {
    pub(crate) fn exact_file(source: String) -> Self {
        Self(ArtifactSpecKind::ExactFile { source })
    }

    pub(crate) fn collected(
        source: String,
        destination: Option<String>,
        exclude: Vec<String>,
    ) -> Self {
        Self(ArtifactSpecKind::Collected {
            source,
            destination,
            exclude,
        })
    }

    /// Returns the validated absolute container source path.
    pub fn source(&self) -> &str {
        match &self.0 {
            ArtifactSpecKind::ExactFile { source } | ArtifactSpecKind::Collected { source, .. } => {
                source
            }
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

    /// Returns structured-declaration exclusion patterns in authored order.
    pub fn exclude(&self) -> &[String] {
        match &self.0 {
            ArtifactSpecKind::ExactFile { .. } => &[],
            ArtifactSpecKind::Collected { exclude, .. } => exclude,
        }
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

    /// Refuses execution when a provider cannot enforce every authored guarantee.
    pub fn validate_for(
        &self,
        capabilities: ProviderCapabilities,
    ) -> Result<(), EvalExecutionError> {
        require(capabilities, "docker")?;
        require(capabilities, "image_source")?;
        for environment in [&self.environment, &self.verifier.environment] {
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
        for phase in [&self.agent, &self.verifier.phase] {
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
        if self.verifier.mode == VerifierMode::Separate {
            require(capabilities, "separate_verifier")?;
        }
        require_network(capabilities, &self.environment.network)?;
        require_network(capabilities, &self.agent.network)?;
        require_network(capabilities, &self.verifier.environment.network)?;
        require_network(capabilities, &self.verifier.phase.network)?;
        Ok(())
    }
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
