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

/// Errors produced during static binary inspection.
#[derive(Debug, thiserror::Error)]
pub enum InspectError {
    /// The file could not be read.
    #[error("io: {0}")]
    Io(#[from] std::io::Error),

    /// The binary format is known but malformed.
    #[error("malformed object: {0}")]
    MalformedObject(String),
}

/// Errors produced during plugin discovery.
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

/// Errors produced during native library loading.
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

/// Errors produced while proving that a discovery path is under trusted
/// authority.
///
/// Every variant is a refusal: the host never downgrades an authority failure
/// into a warning, because a path it cannot prove immutable is a path an
/// unprivileged user may swap between inspection and `dlopen`.
#[derive(Debug, thiserror::Error)]
pub enum AuthorityError {
    /// The final path component is a symbolic link, so the bytes that would be
    /// loaded are chosen by whoever controls the link.
    #[error("path is a symlink: {0}")]
    Symlink(PathBuf),

    /// The path grants write permission to every user on the host.
    #[error("world-writable path {path} (mode {mode:04o})")]
    WorldWritable {
        /// Offending path.
        path: PathBuf,
        /// Permission bits observed.
        mode: u32,
    },

    /// The path grants write permission to its owning group.
    #[error("group-writable path {path} (mode {mode:04o})")]
    GroupWritable {
        /// Offending path.
        path: PathBuf,
        /// Permission bits observed.
        mode: u32,
    },

    /// The path is owned by a uid outside the trusted set.
    #[error("path {path} is owned by untrusted uid {uid}")]
    UntrustedOwner {
        /// Offending path.
        path: PathBuf,
        /// Owning uid observed.
        uid: u32,
    },

    /// The host cannot read or interpret the platform's access-control state
    /// for this path, so it fails closed rather than trusting the mode bits.
    #[error("unknown ACL semantics for {path}: {detail}")]
    UnknownAclSemantics {
        /// Offending path.
        path: PathBuf,
        /// Why the probe was inconclusive.
        detail: String,
    },

    /// The path could not be inspected at all.
    #[error("io inspecting {path}: {source}")]
    Io {
        /// Path being inspected.
        path: PathBuf,
        /// Underlying error.
        #[source]
        source: std::io::Error,
    },
}

/// Errors produced while publishing or verifying an authenticated inventory.
#[derive(Debug, thiserror::Error)]
pub enum InventoryError {
    /// The inventory document could not be read or written.
    #[error("io: {0}")]
    Io(#[from] std::io::Error),

    /// The inventory path is a symbolic link.
    #[error("inventory path is a symlink: {0}")]
    Symlink(PathBuf),

    /// The inventory bytes are not a valid inventory document.
    #[error("inventory parse error: {0}")]
    Parse(String),

    /// The document declares a schema version this host does not implement.
    #[error("unsupported inventory schema version: {0}")]
    UnsupportedSchemaVersion(u32),

    /// A digest is not a canonical 64-character BLAKE3 hex string.  The host
    /// refuses rather than normalizing, because a digest it had to repair is a
    /// digest it cannot attribute to the publisher.
    #[error("malformed digest for {context}: {value}")]
    MalformedDigest {
        /// What the digest was supposed to authenticate.
        context: String,
        /// Digest text observed.
        value: String,
    },

    /// The stored digest does not authenticate the stored payload.
    #[error("inventory digest mismatch")]
    DigestMismatch,

    /// A required inventory field is empty.
    #[error("empty inventory field: {0}")]
    EmptyField(String),

    /// An inventory field carries an absolute or platform-rooted path. The
    /// inventory names content, never locations, so a path here would let a
    /// publisher redirect the host outside its own install root.
    #[error("inventory field carries an absolute path: {0}")]
    AbsolutePath(String),

    /// Two entries declare the same package id.
    #[error("duplicate inventory package: {0}")]
    DuplicatePackage(String),

    /// A package listed in `required_packages` has no entry.
    #[error("missing required package: {0}")]
    MissingRequiredPackage(String),

    /// An entry is signed by a key outside the inventory's declared key set.
    #[error("package {package_id} is signed by untrusted key {key_id}")]
    UntrustedSigningKey {
        /// Offending package.
        package_id: String,
        /// Key the package claims.
        key_id: String,
    },

    /// An entry digest is not a canonical 64-character BLAKE3 hex string, so
    /// the bytes it is supposed to name cannot be identified.
    #[error("package {package_id} carries a malformed digest: {value}")]
    InvalidEntryDigest {
        /// Offending package.
        package_id: String,
        /// Digest text observed.
        value: String,
    },

    /// An entry depends on a package the inventory does not carry, so the
    /// closure it publishes is not self-contained.
    #[error("package {package_id} depends on absent package {missing}")]
    IncompleteClosure {
        /// Package with the dangling edge.
        package_id: String,
        /// Dependency that has no entry.
        missing: String,
    },
}

/// Errors produced while installing, resolving, or collecting generations.
#[derive(Debug, thiserror::Error)]
pub enum InstallError {
    /// A filesystem operation on the install root failed.
    #[error("io at {path}: {source}")]
    Io {
        /// Path being operated on.
        path: PathBuf,
        /// Underlying error.
        #[source]
        source: std::io::Error,
    },

    /// An install file names a path that is absolute or escapes the generation
    /// directory.
    #[error("invalid relative install path: {0}")]
    InvalidRelativePath(String),

    /// A rollback was requested with no previous generation recorded.
    #[error("no previous generation to roll back to")]
    NoPreviousGeneration,

    /// The named generation is absent or was never completed.
    #[error("generation {0} is not a complete installed generation")]
    GenerationNotFound(u64),

    /// The generation exists but records no inventory, so nothing authenticates
    /// the bytes it carries.
    #[error("generation {0} records no inventory digest")]
    IncompleteGeneration(u64),

    /// The generation's recorded inventory does not match the one supplied.
    #[error("inventory digest mismatch: expected {expected}, found {actual}")]
    InventoryDigestMismatch {
        /// Digest the caller expected.
        expected: String,
        /// Digest recorded in the generation.
        actual: String,
    },

    /// Inventory error propagated into the install path.
    #[error("inventory: {0}")]
    Inventory(#[from] InventoryError),
}
