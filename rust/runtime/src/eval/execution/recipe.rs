// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

//! Immutable sandbox recipes that pin image and workspace selection.

use crate::eval::ArtifactDigest;

use super::EvalExecutionError;

/// A resolved native evaluation sandbox recipe.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HarborSandboxRecipe {
    /// Immutable container image digest.
    pub image: String,
    /// Sandbox working directory.
    pub workdir: String,
    workdir_policy: WorkdirPolicy,
}

/// Selects how a recipe resolves its runtime working directory.
#[derive(Clone, Debug, PartialEq, Eq)]
enum WorkdirPolicy {
    /// Retain the legacy recipe fallback when the manifest does not supply a workdir.
    FallbackRecipe,
    /// Preserve the normalized standard-task workdir, including an image `WORKDIR`.
    PreserveStandardTask,
    /// Use an explicitly supplied runtime override without changing the normalized plan.
    Override(String),
}

impl HarborSandboxRecipe {
    /// Creates a recipe that pins an image by digest and an absolute work directory.
    pub fn new(
        image: impl Into<String>,
        workdir: impl Into<String>,
    ) -> Result<Self, EvalExecutionError> {
        let image = image.into();
        let workdir = workdir.into();
        if !image.starts_with("sha256:") || image.len() != 71 {
            return Err(EvalExecutionError::InvalidRecipe("image digest"));
        }
        validate_workdir(&workdir)?;
        Ok(Self {
            image,
            workdir,
            workdir_policy: WorkdirPolicy::FallbackRecipe,
        })
    }

    /// Creates a standard-task recipe with an optional runtime-only workdir override.
    pub fn for_standard_task(
        image: impl Into<String>,
        workdir_override: Option<String>,
    ) -> Result<Self, EvalExecutionError> {
        let image = image.into();
        if !image.starts_with("sha256:") || image.len() != 71 {
            return Err(EvalExecutionError::InvalidRecipe("image digest"));
        }
        let workdir_policy = match workdir_override {
            Some(workdir) => {
                validate_workdir(&workdir)?;
                WorkdirPolicy::Override(workdir)
            }
            None => WorkdirPolicy::PreserveStandardTask,
        };
        Ok(Self {
            image,
            // Kept for legacy callers and recipe identity compatibility. Standard-task
            // execution resolves through `workdir_policy` instead of this fallback.
            workdir: "/work".to_owned(),
            workdir_policy,
        })
    }

    /// Resolves this recipe's runtime directory against an immutable task workdir.
    pub(crate) fn resolve_workdir<'a>(
        &'a self,
        manifest_workdir: Option<&'a str>,
    ) -> Option<&'a str> {
        match &self.workdir_policy {
            WorkdirPolicy::FallbackRecipe => manifest_workdir.or(Some(self.workdir.as_str())),
            WorkdirPolicy::PreserveStandardTask => manifest_workdir,
            WorkdirPolicy::Override(workdir) => Some(workdir),
        }
    }

    /// Returns an immutable digest for the complete recipe identity.
    pub fn identity_digest(&self) -> ArtifactDigest {
        ArtifactDigest::from_bytes(
            format!(
                "image={}\u{1f}workdir={}\u{1f}workdir_policy={:?}",
                self.image, self.workdir, self.workdir_policy
            )
            .as_bytes(),
        )
    }
}

fn validate_workdir(workdir: &str) -> Result<(), EvalExecutionError> {
    if !workdir.starts_with('/')
        || workdir
            .split('/')
            .any(|component| component == "." || component == "..")
    {
        return Err(EvalExecutionError::InvalidRecipe("workdir"));
    }
    Ok(())
}
