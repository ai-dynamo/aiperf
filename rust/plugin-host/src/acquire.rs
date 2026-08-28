// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Immutable manifest and artifact acquisition (Task 11).
//!
//! Reads source files through no-follow descriptors into owned byte vectors
//! and verifies their BLAKE3 digests against the manifest declaration.
//! Nothing from this module mutates disk after the initial copy.

use std::path::{Path, PathBuf};

use crate::{error::AcquireError, manifest::PluginManifestV2};

/// An immutable, digest-verified copy of a plugin manifest file.
#[derive(Debug, Clone)]
pub struct AcquiredManifest {
    /// Verbatim bytes read from the source file.
    pub raw_bytes: Vec<u8>,
    /// Parsed and normalized manifest.
    pub canonical: PluginManifestV2,
    /// Absolute path of the source file at acquisition time.
    pub source_path: PathBuf,
    /// Hex-encoded BLAKE3 digest of `raw_bytes`.
    pub digest: String,
}

/// An immutable, digest-verified copy of one artifact file.
#[derive(Debug, Clone)]
pub struct AcquiredArtifact {
    /// Verbatim bytes read from the source file.
    pub raw_bytes: Vec<u8>,
    /// Absolute path of the source file at acquisition time.
    pub source_path: PathBuf,
    /// Hex-encoded BLAKE3 digest of `raw_bytes`.
    pub digest: String,
    /// Rustc target triple this artifact was built for.
    pub target: String,
}

/// A digest-verified closure: manifest plus all declared artifacts for
/// the requested target triple(s).
#[derive(Debug, Clone)]
pub struct AcquiredClosure {
    pub manifest: AcquiredManifest,
    pub artifacts: Vec<AcquiredArtifact>,
}

impl AcquiredClosure {
    /// Acquire and verify the manifest at `manifest_path` and all declared
    /// artifacts matching any of `targets`.
    ///
    /// Returns `AcquireError::Symlink` if any source file is a symbolic link.
    /// Returns `AcquireError::DigestMismatch` if any file's BLAKE3 digest
    /// does not match its manifest declaration.
    pub fn acquire_from_manifest(
        _manifest_path: &Path,
        _targets: &[&str],
    ) -> Result<Self, AcquireError> {
        unimplemented!("Task 11: immutable acquisition not yet implemented")
    }
}
