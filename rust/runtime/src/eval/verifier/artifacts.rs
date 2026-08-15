// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

//! Exact artifact declarations for an isolated verifier handoff.

use std::{
    collections::BTreeSet,
    fmt::{self, Display, Formatter},
    path::{Component, Path},
};

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
        let mut paths = BTreeSet::new();
        for (path, digest) in artifacts {
            let path = normalize_artifact_path(path)?;
            if !paths.insert(path.clone()) {
                return Err(ArtifactTransferError::DuplicatePath(path));
            }
            declared.push((path, digest));
        }
        Ok(Self {
            artifacts: declared,
        })
    }

    /// Returns the complete declared artifact transfer, excluding workspace state.
    pub fn artifacts(&self) -> &[(String, ArtifactDigest)] {
        &self.artifacts
    }
}

fn normalize_artifact_path(path: &str) -> Result<String, ArtifactTransferError> {
    let parsed = Path::new(path);
    if !parsed.is_absolute() || parsed == Path::new("/") {
        return Err(ArtifactTransferError::InvalidPath(path.to_owned()));
    }
    if parsed.components().any(|component| {
        matches!(
            component,
            Component::ParentDir | Component::CurDir | Component::Prefix(_)
        )
    }) {
        return Err(ArtifactTransferError::InvalidPath(path.to_owned()));
    }
    Ok(format!(
        "/{}",
        parsed
            .components()
            .filter_map(|component| match component {
                Component::Normal(segment) => Some(segment.to_string_lossy().into_owned()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("/")
    ))
}

/// Invalid artifact declaration for a verifier handoff.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ArtifactTransferError {
    /// Artifact paths must be absolute, non-root, and free from traversal.
    InvalidPath(String),
    /// Artifact paths must identify exactly one immutable artifact.
    DuplicatePath(String),
}

impl Display for ArtifactTransferError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidPath(path) => {
                write!(
                    formatter,
                    "artifact path must be absolute and isolated: {path:?}"
                )
            }
            Self::DuplicatePath(path) => write!(formatter, "artifact path is duplicated: {path:?}"),
        }
    }
}

impl std::error::Error for ArtifactTransferError {}
