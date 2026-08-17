// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

//! Provider capability contracts for native evaluation trials.

use std::collections::BTreeSet;
use std::fmt::{self, Display, Formatter};

use url::Url;

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
    /// The provider prevents an adapter from reaching Rust-owned model endpoints.
    ModelEndpointIsolation,
}

/// One host-and-port authority held exclusively by the host model runtime.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct ModelEndpointAuthority(String);

impl ModelEndpointAuthority {
    /// Parses an absolute benchmark model endpoint into its network authority.
    pub fn parse(endpoint: impl AsRef<str>) -> Result<Self, ProviderError> {
        let endpoint = endpoint.as_ref();
        let url = Url::parse(endpoint)
            .map_err(|_| ProviderError::InvalidModelEndpointAuthority(endpoint.to_owned()))?;
        let host = url
            .host_str()
            .ok_or_else(|| ProviderError::InvalidModelEndpointAuthority(endpoint.to_owned()))?;
        let port = url
            .port_or_known_default()
            .ok_or_else(|| ProviderError::InvalidModelEndpointAuthority(endpoint.to_owned()))?;
        Ok(Self(format!("{}:{port}", host.to_ascii_lowercase())))
    }

    /// Parses a canonical provider proof authority.
    pub fn parse_proof(value: impl Into<String>) -> Result<Self, ProviderError> {
        let value = value.into();
        if value.trim().is_empty()
            || value.contains(['/', '@'])
            || !value.contains(':')
            || value.bytes().any(|byte| byte.is_ascii_whitespace())
        {
            return Err(ProviderError::InvalidModelEndpointAuthority(value));
        }
        Ok(Self(value.to_ascii_lowercase()))
    }

    /// Borrows the canonical host-and-port authority.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Evidence that an adapter cannot bypass host-owned model dispatch.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ModelEndpointIsolationProof {
    /// The adapter has no egress path at all.
    NoAdapterEgress,
    /// The provider mediates adapter traffic and blocks each listed authority.
    Mediated {
        /// Every endpoint authority the mediation layer denies to the adapter.
        denied_authorities: BTreeSet<ModelEndpointAuthority>,
    },
}

/// Typed provider cleanup disposition retained separately from a healthy episode result.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ProviderRecovery {
    /// The provider confirmed all task-owned resources were reclaimed.
    Recovered,
    /// A failed adapter was reaped and its replacement must be freshly provisioned.
    RequiresFreshAdapter,
    /// The provider could not establish a safe terminal cleanup state.
    Failed { reason: String },
}

impl ModelEndpointIsolationProof {
    /// Creates a mediation proof from canonical host-and-port authorities.
    pub fn mediated(
        authorities: impl IntoIterator<Item = impl AsRef<str>>,
    ) -> Result<Self, ProviderError> {
        let mut denied_authorities = BTreeSet::new();
        for authority in authorities {
            denied_authorities.insert(ModelEndpointAuthority::parse_proof(authority.as_ref())?);
        }
        Ok(Self::Mediated { denied_authorities })
    }

    /// Reports whether this proof covers every resolved model endpoint authority.
    pub fn covers(&self, authorities: &BTreeSet<ModelEndpointAuthority>) -> bool {
        match self {
            Self::NoAdapterEgress => true,
            Self::Mediated { denied_authorities } => authorities.is_subset(denied_authorities),
        }
    }
}

/// Named provider capabilities resolved before trial start.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProviderProfile {
    name: String,
    capabilities: Vec<ProviderCapability>,
    model_endpoint_isolation: Option<ModelEndpointIsolationProof>,
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
        Ok(Self {
            name,
            capabilities,
            model_endpoint_isolation: None,
        })
    }

    /// Borrows the stable provider identity used for preflight diagnostics.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Borrows the capabilities the provider explicitly guarantees.
    pub fn capabilities(&self) -> &[ProviderCapability] {
        &self.capabilities
    }

    /// Attaches a typed proof for the declared model-endpoint-isolation capability.
    pub fn with_model_endpoint_isolation(
        mut self,
        proof: ModelEndpointIsolationProof,
    ) -> Result<Self, ProviderError> {
        self.require(&[ProviderCapability::ModelEndpointIsolation])?;
        self.model_endpoint_isolation = Some(proof);
        Ok(self)
    }

    /// Borrows the model-endpoint isolation proof, if the provider supplied one.
    pub fn model_endpoint_isolation(&self) -> Option<&ModelEndpointIsolationProof> {
        self.model_endpoint_isolation.as_ref()
    }

    /// Refuses a NativeGraph exact profile unless its typed proof covers each endpoint.
    pub fn require_model_endpoint_isolation(
        &self,
        authorities: &BTreeSet<ModelEndpointAuthority>,
    ) -> Result<(), ProviderError> {
        if !self
            .capabilities
            .contains(&ProviderCapability::ModelEndpointIsolation)
        {
            return Err(ProviderError::MissingModelEndpointIsolationProof);
        }
        let Some(proof) = self.model_endpoint_isolation() else {
            return Err(ProviderError::MissingModelEndpointIsolationProof);
        };
        if proof.covers(authorities) {
            return Ok(());
        }
        let missing = match proof {
            ModelEndpointIsolationProof::NoAdapterEgress => BTreeSet::new(),
            ModelEndpointIsolationProof::Mediated { denied_authorities } => authorities
                .difference(denied_authorities)
                .cloned()
                .collect(),
        };
        Err(ProviderError::ModelEndpointAuthoritiesNotDenied(missing))
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
    /// A model endpoint URL did not name one valid host-and-port authority.
    InvalidModelEndpointAuthority(String),
    /// The provider declared model isolation without its mandatory typed proof.
    MissingModelEndpointIsolationProof,
    /// The mediation proof leaves one or more resolved model endpoint authorities reachable.
    ModelEndpointAuthoritiesNotDenied(BTreeSet<ModelEndpointAuthority>),
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
            Self::InvalidModelEndpointAuthority(authority) => {
                write!(formatter, "invalid model endpoint authority {authority:?}")
            }
            Self::MissingModelEndpointIsolationProof => {
                formatter.write_str("provider missing model endpoint isolation proof")
            }
            Self::ModelEndpointAuthoritiesNotDenied(authorities) => {
                let authorities = authorities
                    .iter()
                    .map(ModelEndpointAuthority::as_str)
                    .collect::<Vec<_>>()
                    .join(", ");
                write!(
                    formatter,
                    "provider model endpoint isolation does not deny {authorities}"
                )
            }
        }
    }
}

impl std::error::Error for ProviderError {}
