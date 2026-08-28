// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Artifact closure acquisition: reads a manifest and all listed artifacts
//! for the requested target triples in a single transaction.

use std::path::Path;

use crate::acquire::{AcquiredArtifact, AcquiredManifest};
use crate::error::AcquireError;

/// A manifest together with all acquired artifacts for the requested targets.
pub struct AcquiredClosure {
    pub manifest: AcquiredManifest,
    pub artifacts: Vec<AcquiredArtifact>,
}

impl AcquiredClosure {
    /// Acquire the manifest at `manifest_path`, then acquire every artifact
    /// listed in the manifest whose `target` is in `targets`.
    ///
    /// Artifact paths in the manifest are resolved relative to the directory
    /// containing the manifest.
    pub fn acquire_from_manifest(
        manifest_path: &Path,
        targets: &[&str],
    ) -> Result<Self, AcquireError> {
        let manifest = AcquiredManifest::acquire(manifest_path)?;
        let manifest_dir = manifest_path.parent().unwrap_or(Path::new("."));

        let mut artifacts = Vec::new();
        for pkg in &manifest.canonical.packages {
            for record in &pkg.artifacts {
                if targets.is_empty() || targets.contains(&record.target.as_str()) {
                    let artifact_path = manifest_dir.join(&record.path);
                    let artifact =
                        AcquiredArtifact::acquire(&artifact_path, &record.digest, &record.target)?;
                    artifacts.push(artifact);
                }
            }
        }

        Ok(Self {
            manifest,
            artifacts,
        })
    }
}
