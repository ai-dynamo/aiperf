// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

//! Validated immutable identifiers used by native evaluation records.

use std::fmt::{self, Display, Formatter};

use serde::{Deserialize, Serialize};

/// A validated `blake3:` artifact digest.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[serde(transparent)]
pub struct ArtifactDigest(String);

impl ArtifactDigest {
    /// Parses a lowercase hexadecimal BLAKE3 digest with its algorithm prefix.
    pub fn parse(value: impl Into<String>) -> Result<Self, EvalIdentityError> {
        let value = value.into();
        let Some(hex) = value.strip_prefix("blake3:") else {
            return Err(EvalIdentityError::InvalidDigest(value));
        };
        if hex.len() != 64
            || !hex
                .bytes()
                .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit())
        {
            return Err(EvalIdentityError::InvalidDigest(value));
        }
        Ok(Self(value))
    }

    /// Constructs a canonical digest from arbitrary immutable bytes.
    pub fn from_bytes(bytes: &[u8]) -> Self {
        Self(format!("blake3:{}", blake3::hash(bytes).to_hex()))
    }

    /// Borrows the canonical digest representation.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// A task identifier within the `eval` namespace.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[serde(transparent)]
pub struct EvalTaskId(String);

impl EvalTaskId {
    /// Constructs a nonempty task identifier.
    pub fn new(value: impl Into<String>) -> Result<Self, EvalIdentityError> {
        let value = value.into();
        if value.trim().is_empty() {
            return Err(EvalIdentityError::Empty("task id"));
        }
        Ok(Self(value))
    }

    /// Borrows the task identifier.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Immutable reference to one content-addressed evaluation task.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[serde(deny_unknown_fields)]
pub struct EvalTaskRef {
    /// Logical task identifier.
    pub id: EvalTaskId,
    /// Canonical task digest.
    pub digest: ArtifactDigest,
}

impl EvalTaskRef {
    /// Constructs an immutable task reference.
    pub fn new(id: impl Into<String>, digest: ArtifactDigest) -> Result<Self, EvalIdentityError> {
        Ok(Self {
            id: EvalTaskId::new(id)?,
            digest,
        })
    }
}

/// Immutable selected agent variant.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[serde(transparent)]
pub struct AgentVariantRef(String);

impl AgentVariantRef {
    /// Constructs a nonempty agent variant reference.
    pub fn new(value: impl Into<String>) -> Result<Self, EvalIdentityError> {
        nonempty(value, "agent variant").map(Self)
    }

    /// Borrows the agent variant reference.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Immutable provider/model selection.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[serde(deny_unknown_fields)]
pub struct ModelIdentity {
    /// Provider identity.
    pub provider: String,
    /// Provider-local model identity.
    pub model: String,
}

impl ModelIdentity {
    /// Constructs a provider/model pair.
    pub fn new(
        provider: impl Into<String>,
        model: impl Into<String>,
    ) -> Result<Self, EvalIdentityError> {
        Ok(Self {
            provider: nonempty(provider, "model provider")?,
            model: nonempty(model, "model")?,
        })
    }
}

/// Immutable policy snapshot identity.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[serde(transparent)]
pub struct PolicyIdentity(pub ArtifactDigest);

impl PolicyIdentity {
    /// Constructs a policy identity from its canonical digest.
    pub fn new(digest: ArtifactDigest) -> Self {
        Self(digest)
    }

    /// Borrows the policy digest.
    pub fn digest(&self) -> &ArtifactDigest {
        &self.0
    }
}

/// Immutable runtime identity.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[serde(transparent)]
pub struct RuntimeIdentity(String);

impl RuntimeIdentity {
    /// Constructs a nonempty runtime identity.
    pub fn new(value: impl Into<String>) -> Result<Self, EvalIdentityError> {
        nonempty(value, "runtime").map(Self)
    }

    /// Borrows the runtime identity.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

fn nonempty(value: impl Into<String>, field: &'static str) -> Result<String, EvalIdentityError> {
    let value = value.into();
    if value.trim().is_empty() {
        Err(EvalIdentityError::Empty(field))
    } else {
        Ok(value)
    }
}

/// Invalid immutable evaluation identity.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EvalIdentityError {
    /// A required identifier was empty.
    Empty(&'static str),
    /// A digest was not canonical BLAKE3 text.
    InvalidDigest(String),
}

impl Display for EvalIdentityError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty(field) => write!(formatter, "{field} must not be empty"),
            Self::InvalidDigest(digest) => write!(formatter, "invalid BLAKE3 digest {digest:?}"),
        }
    }
}

impl std::error::Error for EvalIdentityError {}
