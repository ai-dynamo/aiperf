// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

//! Provider capability contracts for native evaluation trials.

use std::fmt::{self, Display, Formatter};

/// A capability that a selected evaluation provider must guarantee.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProviderCapability {
    /// Branches receive a copy-on-write workspace.
    OverlayWorkspace,
    /// Egress policy can be restricted by the provider.
    NetworkIsolation,
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
        Ok(Self { name, capabilities })
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
}

/// Provider capability preflight failure.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProviderError {
    /// Provider identity was empty.
    EmptyName,
    /// Provider cannot satisfy an authored trial requirement.
    MissingCapability(ProviderCapability),
}

impl Display for ProviderError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyName => formatter.write_str("provider name must not be empty"),
            Self::MissingCapability(capability) => {
                write!(formatter, "provider missing capability {capability:?}")
            }
        }
    }
}

impl std::error::Error for ProviderError {}
