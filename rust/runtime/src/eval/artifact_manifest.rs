// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

//! Canonical artifact manifests retained by Harbor attempt bundles.

use std::{
    fmt::{self, Display, Formatter},
    path::{Component, Path},
};

use serde::{Deserialize, Serialize};

use super::ArtifactDigest;

/// Canonical declarations of artifacts eligible for transfer to a verifier.
#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct DeclaredArtifactManifest {
    /// Normalized absolute artifact paths in canonical bytewise order.
    pub paths: Vec<String>,
    /// Content-addressed identity of the canonical declarations.
    pub digest: ArtifactDigest,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawDeclaredArtifactManifest {
    paths: Vec<String>,
    digest: ArtifactDigest,
}

impl<'de> Deserialize<'de> for DeclaredArtifactManifest {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let raw = RawDeclaredArtifactManifest::deserialize(deserializer)?;
        let manifest = Self::new(raw.paths).map_err(serde::de::Error::custom)?;
        if manifest.digest != raw.digest {
            return Err(serde::de::Error::custom(
                ArtifactManifestError::DeclaredDigestMismatch,
            ));
        }
        Ok(manifest)
    }
}

impl DeclaredArtifactManifest {
    /// Creates a checked manifest from unordered absolute artifact paths.
    pub fn new(paths: impl IntoIterator<Item = String>) -> Result<Self, ArtifactManifestError> {
        let paths = canonical_paths(paths)?;
        let digest = declared_digest(&paths);
        Ok(Self { paths, digest })
    }

    /// Returns the canonical identity for this declaration.
    pub fn identity_digest(&self) -> &ArtifactDigest {
        &self.digest
    }
}

/// Canonical artifact paths together with their immutable content digests.
#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct MaterializedArtifactManifest {
    /// Normalized artifact paths and their content digests in canonical bytewise order.
    pub artifacts: Vec<(String, ArtifactDigest)>,
    /// Content-addressed identity of the canonical materialized artifacts.
    pub digest: ArtifactDigest,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawMaterializedArtifactManifest {
    artifacts: Vec<(String, ArtifactDigest)>,
    digest: ArtifactDigest,
}

impl<'de> Deserialize<'de> for MaterializedArtifactManifest {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let raw = RawMaterializedArtifactManifest::deserialize(deserializer)?;
        let manifest = Self::new(raw.artifacts).map_err(serde::de::Error::custom)?;
        if manifest.digest != raw.digest {
            return Err(serde::de::Error::custom(
                ArtifactManifestError::MaterializedDigestMismatch,
            ));
        }
        Ok(manifest)
    }
}

impl MaterializedArtifactManifest {
    /// Creates a checked manifest from unordered artifact paths and content digests.
    pub fn new(
        artifacts: impl IntoIterator<Item = (String, ArtifactDigest)>,
    ) -> Result<Self, ArtifactManifestError> {
        let mut artifacts = artifacts
            .into_iter()
            .map(|(path, digest)| Ok((normalize_artifact_path(&path)?, digest)))
            .collect::<Result<Vec<_>, ArtifactManifestError>>()?;
        artifacts.sort_unstable_by(|left, right| left.0.cmp(&right.0));
        reject_duplicate_paths(artifacts.iter().map(|(path, _)| path))?;

        let digest = materialized_digest(&artifacts);
        Ok(Self { artifacts, digest })
    }

    /// Returns the canonical identity for these materialized artifacts.
    pub fn identity_digest(&self) -> &ArtifactDigest {
        &self.digest
    }
}

fn canonical_paths(
    paths: impl IntoIterator<Item = String>,
) -> Result<Vec<String>, ArtifactManifestError> {
    let mut paths = paths
        .into_iter()
        .map(|path| normalize_artifact_path(&path))
        .collect::<Result<Vec<_>, _>>()?;
    paths.sort_unstable();
    reject_duplicate_paths(paths.iter())?;
    Ok(paths)
}

fn reject_duplicate_paths<'a>(
    paths: impl IntoIterator<Item = &'a String>,
) -> Result<(), ArtifactManifestError> {
    let mut previous = None;
    for path in paths {
        if previous == Some(path) {
            return Err(ArtifactManifestError::DuplicatePath(path.clone()));
        }
        previous = Some(path);
    }
    Ok(())
}

fn normalize_artifact_path(path: &str) -> Result<String, ArtifactManifestError> {
    let parsed = Path::new(path);
    if !parsed.is_absolute() || parsed == Path::new("/") {
        return Err(ArtifactManifestError::InvalidPath(path.to_owned()));
    }
    if parsed.components().any(|component| {
        matches!(
            component,
            Component::ParentDir | Component::CurDir | Component::Prefix(_)
        )
    }) {
        return Err(ArtifactManifestError::InvalidPath(path.to_owned()));
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

fn declared_digest(paths: &[String]) -> ArtifactDigest {
    let mut bytes = b"harbor-declared-artifacts-v1\x1f".to_vec();
    append_length(&mut bytes, paths.len());
    for path in paths {
        bytes.push(0x1e);
        append_length_prefixed(&mut bytes, path.as_bytes());
    }
    ArtifactDigest::from_bytes(&bytes)
}

fn materialized_digest(artifacts: &[(String, ArtifactDigest)]) -> ArtifactDigest {
    let mut bytes = b"harbor-materialized-artifacts-v1\x1f".to_vec();
    append_length(&mut bytes, artifacts.len());
    for (path, digest) in artifacts {
        bytes.push(0x1e);
        append_length_prefixed(&mut bytes, path.as_bytes());
        bytes.push(0x1f);
        append_length_prefixed(&mut bytes, digest.as_str().as_bytes());
    }
    ArtifactDigest::from_bytes(&bytes)
}

fn append_length_prefixed(bytes: &mut Vec<u8>, value: &[u8]) {
    append_length(bytes, value.len());
    bytes.extend_from_slice(value);
}

fn append_length(bytes: &mut Vec<u8>, length: usize) {
    bytes.extend_from_slice(&(length as u64).to_be_bytes());
}

/// Failed artifact-manifest validation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ArtifactManifestError {
    /// An artifact path was not absolute, isolated, and non-root.
    InvalidPath(String),
    /// Multiple declarations normalized to one artifact path.
    DuplicatePath(String),
    /// A serialized declared-manifest digest did not match its paths.
    DeclaredDigestMismatch,
    /// A serialized materialized-manifest digest did not match its artifacts.
    MaterializedDigestMismatch,
}

impl Display for ArtifactManifestError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidPath(path) => {
                write!(
                    formatter,
                    "artifact path must be absolute and isolated: {path:?}"
                )
            }
            Self::DuplicatePath(path) => write!(formatter, "artifact path is duplicated: {path:?}"),
            Self::DeclaredDigestMismatch => {
                formatter.write_str("declared artifact manifest digest does not match paths")
            }
            Self::MaterializedDigestMismatch => formatter
                .write_str("materialized artifact manifest digest does not match artifacts"),
        }
    }
}

impl std::error::Error for ArtifactManifestError {}
