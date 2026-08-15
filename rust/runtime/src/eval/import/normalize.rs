// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

//! Strict native normalization for the minimum Harbor task package contract.

use serde::Deserialize;

use crate::eval::{ArtifactDigest, EvalTaskRef};

use super::HarborImportError;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct PackageTask {
    pub(super) id: String,
    pub(super) instruction: String,
    pub(super) environment: String,
    pub(super) verifier: String,
}

pub(super) fn normalize(bytes: &[u8]) -> Result<(PackageTask, EvalTaskRef), HarborImportError> {
    let task = serde_json::from_slice::<PackageTask>(bytes)
        .map_err(|error| HarborImportError::InvalidPackage(error.to_string()))?;
    if task.instruction.trim().is_empty() {
        return Err(HarborImportError::InvalidPackage(
            "instruction must not be empty".to_owned(),
        ));
    }
    let environment = ArtifactDigest::parse(task.environment.clone())
        .map_err(|error| HarborImportError::InvalidPackage(error.to_string()))?;
    let verifier = ArtifactDigest::parse(task.verifier.clone())
        .map_err(|error| HarborImportError::InvalidPackage(error.to_string()))?;
    let digest = ArtifactDigest::from_bytes(
        format!(
            "id={}\u{1f}instruction={}\u{1f}environment={}\u{1f}verifier={}",
            task.id,
            task.instruction,
            environment.as_str(),
            verifier.as_str(),
        )
        .as_bytes(),
    );
    let reference = EvalTaskRef::new(task.id.clone(), digest)
        .map_err(|error| HarborImportError::InvalidPackage(error.to_string()))?;
    Ok((task, reference))
}
