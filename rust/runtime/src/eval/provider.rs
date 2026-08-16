// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

//! Provider capability contracts for native evaluation trials.

use std::collections::BTreeSet;
use std::fmt::{self, Display, Formatter};

/// A capability that a selected evaluation provider must guarantee.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum ProviderCapability {
    /// Branches receive a copy-on-write workspace.
    OverlayWorkspace,
    /// Egress policy can be restricted by the provider.
    NetworkIsolation,
    /// Sequential task steps can retain their explicitly declared workspace state.
    PersistentWorkspace,
    /// The task base workspace is immutable while an agent runs.
    ReadOnlyBaseWorkspace,
    /// Declared artifacts can be copied into a verifier-only staging area.
    ArtifactStaging,
    /// Agent credentials are excluded from verifier execution.
    SecretIsolation,
    /// The provider terminates task descendants at the trial boundary.
    DescendantTermination,
    /// The provider enforces the trial's declared resource limits.
    ResourceLimits,
}

/// Named provider capabilities resolved before trial start.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProviderProfile {
    name: String,
    capabilities: Vec<ProviderCapability>,
}

impl ProviderProfile {
    /// Creates a provider profile with a nonempty stable name.
    pub fn new(
        name: impl Into<String>,
        capabilities: Vec<ProviderCapability>,
    ) -> Result<Self, ProviderError> {
        let name = name.into();
        if name.trim().is_empty() {
            return Err(ProviderError::EmptyName);
        }
        let mut declared = BTreeSet::new();
        for capability in &capabilities {
            if !declared.insert(*capability) {
                return Err(ProviderError::DuplicateCapability(*capability));
            }
        }
        Ok(Self { name, capabilities })
    }

    /// Borrows the stable provider identity used for preflight diagnostics.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Borrows the capabilities the provider explicitly guarantees.
    pub fn capabilities(&self) -> &[ProviderCapability] {
        &self.capabilities
    }

    /// Refuses missing capabilities before a trial may be started.
    pub fn require(&self, required: &[ProviderCapability]) -> Result<(), ProviderError> {
        required
            .iter()
            .find(|capability| !self.capabilities.contains(capability))
            .map_or(Ok(()), |capability| {
                Err(ProviderError::MissingCapability(*capability))
            })
    }

    /// Negotiates every required capability and reports the complete shortfall.
    pub fn require_all(&self, required: &[ProviderCapability]) -> Result<(), ProviderError> {
        let mut considered = BTreeSet::new();
        let missing: Vec<_> = required
            .iter()
            .copied()
            .filter(|capability| {
                considered.insert(*capability) && !self.capabilities.contains(capability)
            })
            .collect();

        if missing.is_empty() {
            Ok(())
        } else {
            Err(ProviderError::MissingCapabilities(missing))
        }
    }
}

/// Provider capability preflight failure.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ProviderError {
    /// Provider identity was empty.
    EmptyName,
    /// Provider declared the same capability more than once.
    DuplicateCapability(ProviderCapability),
    /// Provider cannot satisfy an authored trial requirement.
    MissingCapability(ProviderCapability),
    /// Provider cannot satisfy one or more distinct trial requirements.
    MissingCapabilities(Vec<ProviderCapability>),
}

impl Display for ProviderError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyName => formatter.write_str("provider name must not be empty"),
            Self::DuplicateCapability(capability) => {
                write!(
                    formatter,
                    "provider declared duplicate capability {capability:?}"
                )
            }
            Self::MissingCapability(capability) => {
                write!(formatter, "provider missing capability {capability:?}")
            }
            Self::MissingCapabilities(capabilities) => {
                write!(formatter, "provider missing capabilities {capabilities:?}")
            }
        }
    }
}

impl std::error::Error for ProviderError {}
