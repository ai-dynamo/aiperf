// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

//! Strict native normalization for the executable Harbor task package contract.

use std::{
    collections::BTreeSet,
    fs,
    path::{Component, Path},
};

use serde::Deserialize;

use crate::eval::{ArtifactDigest, EvalTaskRef, VerifierMode};

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
    verifier_mode: VerifierMode,
    declared_artifacts: Vec<String>,
    source_digest: ArtifactDigest,
    source_bytes: Vec<u8>,
    source_root: Option<std::path::PathBuf>,
    is_standard_directory: bool,
    container_resources: Option<(u64, u64)>,
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

    /// Returns the task-authored verifier topology.
    pub const fn verifier_mode(&self) -> VerifierMode {
        self.verifier_mode
    }

    /// Returns normalized absolute artifact paths in authored order.
    pub fn declared_artifacts(&self) -> &[String] {
        &self.declared_artifacts
    }

    /// Returns the digest of the complete authored package bytes.
    pub fn source_digest(&self) -> ArtifactDigest {
        self.source_digest.clone()
    }

    /// Returns the immutable, exactly acquired package bytes.
    pub fn source_bytes(&self) -> &[u8] {
        &self.source_bytes
    }

    /// Returns the local source tree retained for fixture materialization, when available.
    pub(crate) fn source_root(&self) -> Option<&std::path::Path> {
        self.source_root.as_deref()
    }

    /// Reports whether this package originated from a standard task directory.
    pub const fn is_standard_directory(&self) -> bool {
        self.is_standard_directory
    }

    /// Returns authored CPU and memory limits for a standard task container.
    pub const fn container_resources(&self) -> Option<(u64, u64)> {
        self.container_resources
    }

    /// Associates an acquired local source tree with this immutable package material.
    pub(crate) fn set_source_root(&mut self, source_root: std::path::PathBuf) {
        self.source_root = Some(source_root);
    }

    /// Replaces the source identity after acquiring a directory-backed package.
    pub(crate) fn set_source_digest(&mut self, source_digest: ArtifactDigest) {
        self.source_digest = source_digest;
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
        verifier_mode: VerifierMode::Separate,
        declared_artifacts,
        source_digest: ArtifactDigest::from_bytes(bytes),
        source_bytes: bytes.to_vec(),
        source_root: None,
        is_standard_directory: false,
        container_resources: None,
    };
    Ok((package, reference))
}

#[derive(Debug, Deserialize)]
struct StandardTaskManifest {
    schema_version: String,
    task: StandardTaskSection,
    verifier: Option<StandardVerifierSection>,
    #[serde(default)]
    artifacts: Vec<String>,
    environment: Option<StandardEnvironmentSection>,
}

#[derive(Debug, Deserialize)]
struct StandardTaskSection {
    name: String,
}

#[derive(Debug, Deserialize)]
struct StandardVerifierSection {
    environment_mode: Option<String>,
}

#[derive(Debug, Deserialize)]
struct StandardEnvironmentSection {
    cpus: Option<u64>,
    memory_mb: Option<u64>,
}

/// Normalizes a standard task directory without executing its contents.
pub(super) fn normalize_standard_directory(
    source_root: &Path,
    manifest_bytes: &[u8],
) -> Result<(HarborTaskPackage, EvalTaskRef), HarborImportError> {
    let manifest = std::str::from_utf8(manifest_bytes)
        .map_err(|error| HarborImportError::InvalidPackage(error.to_string()))?
        .parse::<toml::Value>()
        .map_err(|error| HarborImportError::InvalidPackage(error.to_string()))?
        .try_into::<StandardTaskManifest>()
        .map_err(|error| HarborImportError::InvalidPackage(error.to_string()))?;
    if manifest.schema_version != "1.0" {
        return Err(HarborImportError::InvalidPackage(format!(
            "unsupported task schema version {:?}",
            manifest.schema_version
        )));
    }
    if manifest.task.name.trim().is_empty() {
        return Err(HarborImportError::InvalidPackage(
            "task.name must not be empty".to_owned(),
        ));
    }
    let instruction = read_required_source_file(source_root, "instruction.md")?;
    if instruction.trim().is_empty() {
        return Err(HarborImportError::InvalidPackage(
            "instruction.md must not be empty".to_owned(),
        ));
    }
    let environment = ArtifactDigest::from_bytes(
        read_required_source_file(source_root, "environment/Dockerfile")?.as_bytes(),
    );
    let verifier = ArtifactDigest::from_bytes(
        read_required_source_file(source_root, "tests/test.sh")?.as_bytes(),
    );
    let verifier_mode = match manifest
        .verifier
        .and_then(|verifier| verifier.environment_mode)
        .as_deref()
    {
        None | Some("shared") => VerifierMode::Shared,
        Some("separate") => VerifierMode::Separate,
        Some(value) => {
            return Err(HarborImportError::InvalidPackage(format!(
                "unsupported verifier environment_mode {value:?}"
            )));
        }
    };
    let declared_artifacts = normalize_declared_artifacts(manifest.artifacts)?;
    let container_resources = manifest
        .environment
        .and_then(|environment| Some((environment.cpus?, environment.memory_mb?)));
    let reference_digest = ArtifactDigest::from_bytes(
        format!(
            "id={}\u{1f}instruction={}\u{1f}environment={}\u{1f}verifier={}",
            manifest.task.name,
            instruction,
            environment.as_str(),
            verifier.as_str(),
        )
        .as_bytes(),
    );
    let task = EvalTaskRef::new(manifest.task.name.clone(), reference_digest)
        .map_err(|error| HarborImportError::InvalidPackage(error.to_string()))?;
    let package = HarborTaskPackage {
        id: manifest.task.name,
        instruction,
        environment: environment.as_str().to_owned(),
        verifier: verifier.as_str().to_owned(),
        agent_command: vec!["aiperf-task-agent".to_owned()],
        verifier_command: vec!["/bin/sh".to_owned(), "tests/test.sh".to_owned()],
        verifier_mode,
        declared_artifacts,
        source_digest: ArtifactDigest::from_bytes(manifest_bytes),
        source_bytes: manifest_bytes.to_vec(),
        source_root: None,
        is_standard_directory: true,
        container_resources,
    };
    Ok((package, task))
}

fn read_required_source_file(
    source_root: &Path,
    relative_path: &str,
) -> Result<String, HarborImportError> {
    let path = source_root.join(relative_path);
    fs::read_to_string(&path)
        .map_err(|error| HarborImportError::InvalidPackage(format!("{}: {error}", path.display())))
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
