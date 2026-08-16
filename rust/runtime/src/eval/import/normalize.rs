// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

//! Strict native normalization for the executable Harbor task package contract.

use std::{
    collections::BTreeSet,
    path::{Component, Path},
};

use serde::Deserialize;

use crate::eval::{ArtifactDigest, EvalTaskRef};

use super::HarborImportError;

/// Executable material retained from one strict Harbor task package.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HarborTaskPackage {
    id: String,
    instruction: String,
    environment: String,
    verifier: String,
    agent_command: Vec<String>,
    verifier_command: Vec<String>,
    declared_artifacts: Vec<String>,
    source_digest: ArtifactDigest,
}

impl HarborTaskPackage {
    /// Returns the authored task identifier.
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Returns the authored instruction presented to the agent.
    pub fn instruction(&self) -> &str {
        &self.instruction
    }

    /// Returns the immutable environment artifact identity.
    pub fn environment(&self) -> &str {
        &self.environment
    }

    /// Returns the immutable verifier artifact identity.
    pub fn verifier(&self) -> &str {
        &self.verifier
    }

    /// Returns the exact argv used to invoke the agent.
    pub fn agent_command(&self) -> &[String] {
        &self.agent_command
    }

    /// Returns the exact argv used to invoke the verifier.
    pub fn verifier_command(&self) -> &[String] {
        &self.verifier_command
    }

    /// Returns normalized absolute artifact paths in authored order.
    pub fn declared_artifacts(&self) -> &[String] {
        &self.declared_artifacts
    }

    /// Returns the digest of the complete authored package bytes.
    pub fn source_digest(&self) -> ArtifactDigest {
        self.source_digest.clone()
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct PackageTaskDto {
    id: String,
    instruction: String,
    environment: String,
    verifier: String,
    agent_command: Vec<String>,
    verifier_command: Vec<String>,
    declared_artifacts: Vec<String>,
}

pub(super) fn normalize(
    bytes: &[u8],
) -> Result<(HarborTaskPackage, EvalTaskRef), HarborImportError> {
    let task = serde_json::from_slice::<PackageTaskDto>(bytes)
        .map_err(|error| HarborImportError::InvalidPackage(error.to_string()))?;
    if task.instruction.trim().is_empty() {
        return Err(HarborImportError::InvalidPackage(
            "instruction must not be empty".to_owned(),
        ));
    }
    validate_command("agent_command", &task.agent_command)?;
    validate_command("verifier_command", &task.verifier_command)?;
    let declared_artifacts = normalize_declared_artifacts(task.declared_artifacts)?;
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
    let package = HarborTaskPackage {
        id: task.id,
        instruction: task.instruction,
        environment: task.environment,
        verifier: task.verifier,
        agent_command: task.agent_command,
        verifier_command: task.verifier_command,
        declared_artifacts,
        source_digest: ArtifactDigest::from_bytes(bytes),
    };
    Ok((package, reference))
}

fn validate_command(field: &'static str, command: &[String]) -> Result<(), HarborImportError> {
    if command.is_empty() || command.iter().any(|part| part.trim().is_empty()) {
        return Err(HarborImportError::InvalidPackage(format!(
            "{field} must be a nonempty argv"
        )));
    }
    Ok(())
}

fn normalize_declared_artifacts(artifacts: Vec<String>) -> Result<Vec<String>, HarborImportError> {
    let mut normalized = Vec::with_capacity(artifacts.len());
    let mut paths = BTreeSet::new();
    for path in artifacts {
        let path = normalize_artifact_path(&path)?;
        if !paths.insert(path.clone()) {
            return Err(HarborImportError::InvalidPackage(format!(
                "declared artifact path is duplicated: {path:?}"
            )));
        }
        normalized.push(path);
    }
    Ok(normalized)
}

fn normalize_artifact_path(path: &str) -> Result<String, HarborImportError> {
    let parsed = Path::new(path);
    if !parsed.is_absolute() || parsed == Path::new("/") {
        return Err(invalid_artifact_path(path));
    }
    if parsed.components().any(|component| {
        matches!(
            component,
            Component::ParentDir | Component::CurDir | Component::Prefix(_)
        )
    }) {
        return Err(invalid_artifact_path(path));
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

fn invalid_artifact_path(path: &str) -> HarborImportError {
    HarborImportError::InvalidPackage(format!(
        "declared artifact path must be absolute and isolated: {path:?}"
    ))
}
