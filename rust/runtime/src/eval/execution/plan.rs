// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

//! Immutable, host-independent execution plans for standard Harbor tasks.

use std::{
    collections::BTreeSet,
    net::{IpAddr, Ipv4Addr, Ipv6Addr},
    time::Duration,
};

use crate::eval::VerifierMode;

/// A normalized environment variable binding.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EnvBinding {
    /// A literal value that is safe to retain in the imported package.
    Literal(String),
    /// A host variable name resolved only when the owning phase executes.
    SecretReference(String),
}

impl EnvBinding {
    /// Parses a literal value or one complete `${NAME}` host-secret reference.
    pub fn parse(value: &str) -> Result<Self, String> {
        if !value.contains("${") {
            return Ok(Self::Literal(value.to_owned()));
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
        Ok(Self::SecretReference(name.to_owned()))
    }
}

/// A network policy whose allowlist entries are normalized at import time.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NetworkPolicy {
    /// The provider-defined public benchmark network.
    Public,
    /// No outbound network access.
    NoNetwork,
    /// Egress is limited to the normalized host or CIDR entries.
    Allowlist {
        /// Deterministically sorted normalized allowlist entries.
        allowed_hosts: Vec<String>,
    },
}

impl NetworkPolicy {
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
        Ok(Self::Allowlist {
            allowed_hosts: normalized.into_iter().collect(),
        })
    }
}

/// CPU and memory limits that must be authored together.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ContainerResources {
    /// Whole CPU limit.
    pub cpus: u64,
    /// Memory limit in MiB.
    pub memory_mb: u64,
}

/// Readiness behavior for an environment before its agent phase begins.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HealthcheckPlan {
    /// Command argv executed for each readiness attempt.
    pub command: Vec<String>,
    /// Initial grace period before failures count.
    pub start_period: Option<Duration>,
    /// Poll interval during the initial grace period.
    pub start_interval: Option<Duration>,
    /// Poll interval after the initial grace period.
    pub interval: Option<Duration>,
    /// Per-attempt deadline.
    pub timeout: Option<Duration>,
    /// Consecutive failed attempts allowed after the grace period.
    pub retries: Option<u32>,
}

/// One immutable environment baseline.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EnvironmentPlan {
    /// Optional CPU and memory limits.
    pub resources: Option<ContainerResources>,
    /// Optional absolute workdir; absence preserves the image `WORKDIR`.
    pub workdir: Option<String>,
    /// Optional effective user for the environment.
    pub user: Option<String>,
    /// Baseline environment bindings ordered by variable name.
    pub env: std::collections::BTreeMap<String, EnvBinding>,
    /// Baseline network policy.
    pub network: NetworkPolicy,
    /// Optional readiness command.
    pub healthcheck: Option<HealthcheckPlan>,
}

impl Default for EnvironmentPlan {
    fn default() -> Self {
        Self {
            resources: None,
            workdir: None,
            user: None,
            env: std::collections::BTreeMap::new(),
            network: NetworkPolicy::Public,
            healthcheck: None,
        }
    }
}

/// Phase-specific execution settings.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PhasePlan {
    /// Optional effective-user override for this phase.
    pub user: Option<String>,
    /// Phase bindings that override identical environment binding names.
    pub env: std::collections::BTreeMap<String, EnvBinding>,
    /// Fully resolved phase network policy.
    pub network: NetworkPolicy,
    /// Optional execution deadline.
    pub timeout: Option<Duration>,
}

/// The verifier topology and its immutable environment.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifierPlan {
    /// Phase-only verifier settings.
    pub phase: PhasePlan,
    /// Shared or separate verifier topology.
    pub mode: VerifierMode,
    /// The environment baseline for the verifier.
    pub environment: EnvironmentPlan,
}

/// One declared benchmark artifact.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ArtifactSpec {
    /// A legacy string declaration for one exact regular file.
    ExactFile {
        /// Absolute source path in the task container.
        source: String,
    },
    /// A structured declaration that may collect a directory tree.
    Collected {
        /// Absolute source path in the task container.
        source: String,
        /// Optional relative destination below the collection root.
        destination: Option<String>,
        /// Glob exclusions retained in authored order.
        exclude: Vec<String>,
    },
}

impl ArtifactSpec {
    /// Returns the validated absolute container source path.
    pub fn source(&self) -> &str {
        match self {
            Self::ExactFile { source } | Self::Collected { source, .. } => source,
        }
    }

    /// Reports whether this declaration names exactly one regular file.
    pub const fn is_exact_file(&self) -> bool {
        matches!(self, Self::ExactFile { .. })
    }

    /// Returns the optional relative collection destination.
    pub fn destination(&self) -> Option<&str> {
        match self {
            Self::ExactFile { .. } => None,
            Self::Collected { destination, .. } => destination.as_deref(),
        }
    }

    /// Returns structured-declaration exclusion patterns in authored order.
    pub fn exclude(&self) -> &[String] {
        match self {
            Self::ExactFile { .. } => &[],
            Self::Collected { exclude, .. } => exclude,
        }
    }
}

/// Immutable execution intent compiled from one standard task manifest.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BenchmarkExecutionPlan {
    /// Environment baseline used for image construction and the agent container.
    pub environment: EnvironmentPlan,
    /// Agent phase settings.
    pub agent: PhasePlan,
    /// Verifier phase topology and environment.
    pub verifier: VerifierPlan,
    /// Artifact declarations in authored order.
    pub artifacts: Vec<ArtifactSpec>,
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
