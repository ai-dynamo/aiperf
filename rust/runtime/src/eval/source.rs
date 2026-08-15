// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

//! Immutable source-suite and dataset-manifest identities.

use serde::{Deserialize, Serialize};

use super::{ArtifactDigest, EvalIdentityError, EvalTaskRef};

/// A stable identifier for an immutable evaluation dataset manifest.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[serde(transparent)]
pub struct EvalDatasetId(String);

impl EvalDatasetId {
    /// Creates a nonempty dataset identifier.
    pub fn new(value: impl Into<String>) -> Result<Self, EvalIdentityError> {
        let value = value.into();
        if value.trim().is_empty() {
            return Err(EvalIdentityError::Empty("dataset id"));
        }
        Ok(Self(value))
    }

    /// Borrows the dataset identifier.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Immutable, reproducible selection of evaluation tasks.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct EvalDatasetManifest {
    /// Logical suite identifier.
    pub id: EvalDatasetId,
    /// Publisher-authored immutable suite version.
    pub version: String,
    /// Ordered immutable task selection.
    pub tasks: Vec<EvalTaskRef>,
    /// Digest over this complete resolved manifest.
    pub digest: ArtifactDigest,
}

impl EvalDatasetManifest {
    /// Creates an immutable manifest from an explicit task selection.
    pub fn new(
        id: impl Into<String>,
        version: impl Into<String>,
        tasks: Vec<EvalTaskRef>,
    ) -> Result<Self, EvalIdentityError> {
        let id = EvalDatasetId::new(id)?;
        let version = version.into();
        if version.trim().is_empty() {
            return Err(EvalIdentityError::Empty("dataset version"));
        }
        if tasks.is_empty() {
            return Err(EvalIdentityError::Empty("dataset tasks"));
        }
        let digest = ArtifactDigest::from_bytes(&canonical_manifest_bytes(&id, &version, &tasks));
        Ok(Self {
            id,
            version,
            tasks,
            digest,
        })
    }
}

fn canonical_manifest_bytes(id: &EvalDatasetId, version: &str, tasks: &[EvalTaskRef]) -> Vec<u8> {
    let mut bytes = format!("id={}\u{1f}version={version}", id.as_str()).into_bytes();
    for task in tasks {
        bytes.extend_from_slice(b"\x1etask=");
        bytes.extend_from_slice(task.id.as_str().as_bytes());
        bytes.extend_from_slice(b"\x1fdigest=");
        bytes.extend_from_slice(task.digest.as_str().as_bytes());
    }
    bytes
}
