// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

//! Exact artifact declarations for an isolated verifier handoff.

use std::fmt::{self, Display, Formatter};

use crate::eval::ArtifactDigest;

/// Exact artifact paths and immutable contents available to an isolated verifier.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeclaredArtifactTransfer {
    artifacts: Vec<(String, ArtifactDigest)>,
}

impl DeclaredArtifactTransfer {
    /// Creates a transfer containing only absolute, explicitly declared artifact paths.
    pub fn new(artifacts: Vec<(&str, ArtifactDigest)>) -> Result<Self, ArtifactTransferError> {
        let mut declared = Vec::with_capacity(artifacts.len());
        for (path, digest) in artifacts {
            if !path.starts_with('/') {
                return Err(ArtifactTransferError::RelativePath(path.to_owned()));
            }
            declared.push((path.to_owned(), digest));
        }
        Ok(Self { artifacts: declared })
    }

    /// Returns the complete declared artifact transfer, excluding workspace state.
    pub fn artifacts(&self) -> &[(String, ArtifactDigest)] {
        &self.artifacts
    }
}

/// Invalid artifact declaration for a verifier handoff.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ArtifactTransferError {
    /// Artifact paths must be absolute within the verifier sandbox.
    RelativePath(String),
}

impl Display for ArtifactTransferError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::RelativePath(path) => write!(formatter, "artifact path must be absolute: {path:?}"),
        }
    }
}

impl std::error::Error for ArtifactTransferError {}
