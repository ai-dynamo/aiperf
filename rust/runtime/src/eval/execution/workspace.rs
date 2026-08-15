// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

//! Immutable branch overlays and candidate patches.

use crate::eval::ArtifactDigest;

use super::EvalExecutionError;

/// A copy-on-write branch rooted in an immutable workspace snapshot.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WorkspaceOverlay {
    base: ArtifactDigest,
    branch: Option<String>,
}

impl WorkspaceOverlay {
    /// Creates the canonical immutable workspace root.
    pub fn canonical(base: impl Into<String>) -> Result<Self, EvalExecutionError> {
        let base = ArtifactDigest::parse(base.into())
            .map_err(|error| EvalExecutionError::InvalidWorkspace(error.to_string()))?;
        Ok(Self { base, branch: None })
    }

    /// Creates an isolated named branch that cannot mutate the canonical root.
    pub fn branch(&self, name: impl Into<String>) -> Result<Self, EvalExecutionError> {
        let name = name.into();
        if name.trim().is_empty() {
            return Err(EvalExecutionError::InvalidWorkspace(
                "branch name must not be empty".to_owned(),
            ));
        }
        Ok(Self {
            base: self.base.clone(),
            branch: Some(name),
        })
    }

    /// Returns the immutable workspace snapshot that this overlay descends from.
    pub fn base_digest(&self) -> &ArtifactDigest {
        &self.base
    }

    /// Returns an immutable candidate patch without modifying the canonical root.
    pub fn complete(&self, patch: impl Into<String>) -> Result<ImmutablePatch, EvalExecutionError> {
        if self.branch.is_none() {
            return Err(EvalExecutionError::InvalidWorkspace(
                "only a branch may produce a patch".to_owned(),
            ));
        }
        let digest = ArtifactDigest::parse(patch.into())
            .map_err(|error| EvalExecutionError::InvalidWorkspace(error.to_string()))?;
        Ok(ImmutablePatch {
            parent: self.base.clone(),
            digest,
        })
    }
}

/// An immutable candidate patch emitted from an isolated branch.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImmutablePatch {
    parent: ArtifactDigest,
    /// Digest of the candidate patch artifact.
    pub digest: ArtifactDigest,
}

impl ImmutablePatch {
    /// Returns the immutable canonical snapshot that the patch was derived from.
    pub fn parent_digest(&self) -> &ArtifactDigest {
        &self.parent
    }
}
