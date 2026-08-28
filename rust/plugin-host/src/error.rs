// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Typed error codes for plugin host operations.

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

/// Errors produced during artifact acquisition (Task 11 / Task 12).
#[derive(Debug, thiserror::Error)]
pub enum AcquireError {
    /// The source path is a symlink; no-follow policy refuses to acquire it.
    #[error("path is a symlink: {0}")]
    Symlink(PathBuf),

    /// The acquired bytes do not match the manifest-declared digest.
    #[error("digest mismatch: expected {expected}, got {actual}")]
    DigestMismatch { expected: String, actual: String },

    /// Staged bytes were altered after staging (re-verify mismatch).
    #[error("tampered staged bytes: {0}")]
    StagedTamper(PathBuf),

    /// Two loader artifacts with different staged paths claim the same content digest
    /// under the same loader identity.
    #[error("conflicting loader identity: digest={digest} a={a} b={b}")]
    ConflictingLoaderIdentity { digest: String, a: String, b: String },

    /// An I/O error occurred while acquiring or staging.
    #[error("io: {0}")]
    Io(#[from] std::io::Error),

    /// Manifest decode failed during acquisition.
    #[error("manifest: {0}")]
    Manifest(#[from] ManifestError),
}

/// Errors produced during static binary inspection (Task 12).
#[derive(Debug, thiserror::Error)]
pub enum InspectError {
    /// The file could not be read.
    #[error("io: {0}")]
    Io(#[from] std::io::Error),

    /// The binary format is known but malformed.
    #[error("malformed object: {0}")]
    MalformedObject(String),
}

/// Errors produced during plugin discovery (Task 13).
#[derive(Debug, thiserror::Error)]
pub enum DiscoveryError {
    /// A discovery source directory could not be read.
    #[error("io scanning {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    /// A manifest found during discovery failed to parse.
    #[error("manifest at {path}: {source}")]
    ManifestAtPath {
        path: PathBuf,
        #[source]
        source: ManifestError,
    },
}

/// Errors produced during native library loading (Task 14).
#[derive(Debug, thiserror::Error)]
pub enum LoadError {
    /// dlopen returned an error string.
    #[error("dlopen {path}: {detail}")]
    DlopenFailed { path: PathBuf, detail: String },

    /// The library was already loaded under a different staged path.
    #[error("residency conflict: digest={digest} existing={existing} new={new}")]
    ResidencyConflict {
        digest: String,
        existing: PathBuf,
        new: PathBuf,
    },

    /// Acquire error propagated into the load phase.
    #[error("acquire: {0}")]
    Acquire(#[from] AcquireError),
}
