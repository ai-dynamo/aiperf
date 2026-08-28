// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Typed error codes for plugin manifest and acquisition operations.

use std::path::PathBuf;

/// Errors produced during manifest parsing and normalization.
#[derive(Debug, thiserror::Error)]
pub enum ManifestError {
    /// YAML/JSON parse failure.
    #[error("manifest parse error: {0}")]
    ParseError(String),

    /// The manifest declares `schema_version: "1.0"`, which is the Python plugin
    /// format. Stable error code: `python-plugin-manifest-not-native`.
    #[error("python-plugin-manifest-not-native: schema version {0} is not the native format")]
    PythonManifest(String),

    /// The `schema_version` field is present but not a recognized native version.
    #[error("unsupported schema version: {0}")]
    UnsupportedSchemaVersion(String),

    /// A field or value unknown to this schema version was encountered.
    #[error("unknown field: {0}")]
    UnknownField(String),

    /// A required field was absent.
    #[error("missing required field: {0}")]
    MissingField(String),

    /// A version string is not valid canonical SemVer (X.Y.Z).
    #[error("invalid semver: {0}")]
    InvalidSemVer(String),

    /// An artifact path is absolute, contains `..`, or uses Windows ADS (`:stream`).
    #[error("invalid path: {0}")]
    InvalidPath(String),

    /// Two artifact entries declare the same target triple.
    #[error("duplicate artifact for target: {0}")]
    DuplicateBaselineArtifact(String),

    /// A category tag is not one of the supported values.
    #[error("unsupported category: {0}")]
    UnsupportedCategory(String),

    /// The package declares an out-of-range priority value.
    #[error("invalid priority: {0}")]
    InvalidPriority(i64),

    /// A package declares no category entries.
    #[error("no categories defined")]
    NoCategories,
}

/// Errors produced during immutable artifact acquisition and staging.
#[derive(Debug, thiserror::Error)]
pub enum AcquireError {
    /// The path at the final component is a symbolic link.
    #[error("path is a symlink: {0}")]
    Symlink(PathBuf),

    /// The acquired bytes do not match the expected BLAKE3 digest.
    #[error("digest mismatch: expected {expected}, got {actual}")]
    DigestMismatch { expected: String, actual: String },

    /// The staged bytes were tampered with after staging.
    #[error("tampered staged bytes: {0}")]
    StagedTamper(PathBuf),

    /// Two plugins claim the same `(loader_id, digest)` with conflicting identity.
    #[error("conflicting loader identity for digest {digest}: {a} vs {b}")]
    ConflictingLoaderIdentity {
        digest: String,
        a: String,
        b: String,
    },

    /// Underlying I/O error.
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    /// Manifest parse or normalization error.
    #[error("manifest error: {0}")]
    Manifest(#[from] ManifestError),
}
