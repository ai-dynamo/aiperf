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
}

impl HarborSandboxRecipe {
    /// Creates a recipe that pins an image by digest and an absolute work directory.
    pub fn new(image: impl Into<String>, workdir: impl Into<String>) -> Result<Self, EvalExecutionError> {
        let image = image.into();
        let workdir = workdir.into();
        if !image.starts_with("sha256:") || image.len() != 71 {
            return Err(EvalExecutionError::InvalidRecipe("image digest"));
        }
        if !workdir.starts_with('/') {
            return Err(EvalExecutionError::InvalidRecipe("workdir"));
        }
        Ok(Self { image, workdir })
    }

    /// Returns an immutable digest for the complete recipe identity.
    pub fn identity_digest(&self) -> ArtifactDigest {
        ArtifactDigest::from_bytes(format!("image={}\u{1f}workdir={}", self.image, self.workdir).as_bytes())
    }
}
