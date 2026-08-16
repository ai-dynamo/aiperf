// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

//! Temporary-root local process sandboxing for native P0 evaluation.

use std::{
    fs,
    path::{Path, PathBuf},
    process::{Command, ExitStatus},
};

use tempfile::TempDir;

use crate::eval::{
    ArtifactDigest, AttemptId, HarborTaskPackage, RegradeError, RewardDocument, ScoreVersion,
    VerifierMode,
};

use super::{EvalExecutionError, HarborSandboxRecipe, ProviderCapabilities};

/// Selects the isolated root materialized for one evaluation participant.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SandboxRole {
    /// Root in which the selected agent executes.
    Agent,
    /// Root for a verifier explicitly authorized to share the task sandbox.
    SharedVerifier,
    /// Fresh root for a separately provisioned verifier.
    SeparateVerifier,
}

/// Concrete local process provider for deterministic P0 package execution.
#[derive(Debug, Default)]
pub struct LocalProcessSandbox;

impl LocalProcessSandbox {
    /// Creates an empty local-process provider.
    pub const fn new() -> Self {
        Self
    }

    /// Writes exactly the acquired package bytes into an isolated temporary root.
    pub fn materialize(
        &self,
        _: &HarborSandboxRecipe,
        package: &HarborTaskPackage,
        _: SandboxRole,
    ) -> Result<MaterializedSandbox, EvalExecutionError> {
        let lease = tempfile::tempdir()
            .map_err(|error| EvalExecutionError::Materialization(error.to_string()))?;
        package.materialize_source_into(lease.path())?;
        fs::create_dir_all(lease.path().join("results"))
            .map_err(|error| EvalExecutionError::Materialization(error.to_string()))?;
        Ok(MaterializedSandbox { lease })
    }

    /// Runs the package agent then its verifier, transferring only declared artifacts.
    pub fn execute(
        &self,
        recipe: &HarborSandboxRecipe,
        package: &HarborTaskPackage,
        verifier_mode: VerifierMode,
    ) -> Result<LocalExecutionResult, EvalExecutionError> {
        self.execute_with_agent_command(recipe, package, package.agent_command(), verifier_mode)
    }

    /// Runs a package with a caller-supplied external agent command.
    pub fn execute_with_agent_command(
        &self,
        recipe: &HarborSandboxRecipe,
        package: &HarborTaskPackage,
        agent_command: &[String],
        verifier_mode: VerifierMode,
    ) -> Result<LocalExecutionResult, EvalExecutionError> {
        if package.execution_plan().is_multi_step() {
            return Err(EvalExecutionError::UnsupportedMultiStep);
        }
        if package.is_standard_directory() {
            package
                .execution_plan()
                .validate_for(ProviderCapabilities::none())?;
        }
        let agent = self.materialize(recipe, package, SandboxRole::Agent)?;
        let environment = vec![(
            "AIPERF_EVAL_INSTRUCTION".to_owned(),
            package.instruction().to_owned(),
        )];
        agent.run(agent_command, &environment)?;
        let artifacts = collect_declared_artifacts(&agent, package)?;
        let reward = match verifier_mode {
            VerifierMode::Shared => {
                agent.run(package.verifier_command(), &environment)?;
                parse_reward(&agent)?
            }
            VerifierMode::Separate => {
                let verifier = self.materialize(recipe, package, SandboxRole::SeparateVerifier)?;
                copy_declared_artifacts(&agent, &verifier, package)?;
                verifier.run(package.verifier_command(), &environment)?;
                parse_reward(&verifier)?
            }
        };
        let verifier = ArtifactDigest::parse(package.verifier())
            .map_err(|error| EvalExecutionError::Materialization(error.to_string()))?;
        Ok(LocalExecutionResult {
            artifacts,
            reward,
            verifier,
        })
    }
}

/// A live local sandbox root whose lease removes it when evaluation finishes.
#[derive(Debug)]
pub struct MaterializedSandbox {
    lease: TempDir,
}

impl MaterializedSandbox {
    /// Returns the private process root.
    pub fn root(&self) -> &Path {
        self.lease.path()
    }

    /// Maps a declared absolute artifact path into this private root.
    pub fn artifact_path(&self, declared_path: &str) -> Result<PathBuf, EvalExecutionError> {
        let relative = declared_path
            .strip_prefix('/')
            .filter(|path| {
                !path.is_empty() && !path.split('/').any(|part| part == "." || part == "..")
            })
            .ok_or_else(|| {
                EvalExecutionError::Materialization("invalid declared artifact path".to_owned())
            })?;
        Ok(self.root().join(relative))
    }

    /// Runs an argv with no inherited environment in this sandbox root.
    pub fn run(
        &self,
        argv: &[String],
        environment: &[(String, String)],
    ) -> Result<ProcessOutput, EvalExecutionError> {
        let (program, arguments) = argv
            .split_first()
            .ok_or(EvalExecutionError::InvalidCommand)?;
        if program.trim().is_empty() || arguments.iter().any(|argument| argument.trim().is_empty())
        {
            return Err(EvalExecutionError::InvalidCommand);
        }
        let output = Command::new(program)
            .args(arguments)
            .current_dir(self.root())
            .env_clear()
            .env("PATH", "/usr/bin:/bin")
            .env("AIPERF_EVAL_ROOT", self.root())
            .envs(environment.iter().map(|(key, value)| (key, value)))
            .output()
            .map_err(|_| EvalExecutionError::ProcessSpawn(program.clone()))?;
        if !output.status.success() {
            return Err(EvalExecutionError::ProcessFailure(program.clone()));
        }
        Ok(ProcessOutput {
            status: output.status,
            stdout: output.stdout,
            stderr: output.stderr,
        })
    }
}

/// Captured output from a successful local sandbox process.
#[derive(Debug)]
pub struct ProcessOutput {
    /// Terminal process status.
    pub status: ExitStatus,
    /// Captured standard output bytes.
    pub stdout: Vec<u8>,
    /// Captured standard error bytes.
    pub stderr: Vec<u8>,
}

/// Immutable artifacts and reward emitted by one completed native local evaluation.
#[derive(Clone, Debug, PartialEq)]
pub struct LocalExecutionResult {
    /// Declared artifact paths paired with content digests.
    pub artifacts: Vec<(String, ArtifactDigest)>,
    /// Finite verifier reward metrics.
    pub reward: RewardDocument,
    /// Immutable verifier implementation identity that produced the reward.
    pub verifier: ArtifactDigest,
}

impl LocalExecutionResult {
    /// Produces the initial immutable score revision from this verifier result.
    pub fn initial_score(
        &self,
        attempt: AttemptId,
        metric: impl Into<String>,
        rationale: ArtifactDigest,
    ) -> Result<ScoreVersion, RegradeError> {
        let metric = metric.into();
        let value = self
            .reward
            .metrics
            .get(&metric)
            .copied()
            .ok_or_else(|| RegradeError::MetricNotFound(metric.clone()))?;
        ScoreVersion::initial(
            attempt,
            self.verifier.clone(),
            self.artifacts
                .iter()
                .map(|(_, digest)| digest.clone())
                .collect(),
            metric,
            value,
            rationale,
        )
        .map_err(RegradeError::InvalidScore)
    }
}

fn collect_declared_artifacts(
    sandbox: &MaterializedSandbox,
    package: &HarborTaskPackage,
) -> Result<Vec<(String, ArtifactDigest)>, EvalExecutionError> {
    package
        .declared_artifacts()
        .iter()
        .map(|path| {
            let artifact_path = sandbox.artifact_path(path)?;
            let metadata = fs::symlink_metadata(&artifact_path)
                .map_err(|error| EvalExecutionError::ArtifactCollection(error.to_string()))?;
            if metadata.file_type().is_symlink() || !metadata.file_type().is_file() {
                return Err(EvalExecutionError::ArtifactCollection(format!(
                    "declared artifact is not a regular file: {}",
                    path
                )));
            }
            let bytes = fs::read(artifact_path)
                .map_err(|error| EvalExecutionError::ArtifactCollection(error.to_string()))?;
            Ok((path.clone(), ArtifactDigest::from_bytes(&bytes)))
        })
        .collect()
}

fn copy_declared_artifacts(
    source: &MaterializedSandbox,
    destination: &MaterializedSandbox,
    package: &HarborTaskPackage,
) -> Result<(), EvalExecutionError> {
    for path in package.declared_artifacts() {
        let source_path = source.artifact_path(path)?;
        let destination_path = destination.artifact_path(path)?;
        let metadata = fs::symlink_metadata(&source_path)
            .map_err(|error| EvalExecutionError::ArtifactCollection(error.to_string()))?;
        if metadata.file_type().is_symlink() || !metadata.file_type().is_file() {
            return Err(EvalExecutionError::ArtifactCollection(format!(
                "declared artifact is not a regular file: {}",
                path
            )));
        }
        let bytes = fs::read(source_path)
            .map_err(|error| EvalExecutionError::ArtifactCollection(error.to_string()))?;
        if let Some(parent) = destination_path.parent() {
            fs::create_dir_all(parent)
                .map_err(|error| EvalExecutionError::Materialization(error.to_string()))?;
        }
        fs::write(destination_path, bytes)
            .map_err(|error| EvalExecutionError::Materialization(error.to_string()))?;
    }
    Ok(())
}

fn parse_reward(sandbox: &MaterializedSandbox) -> Result<RewardDocument, EvalExecutionError> {
    let reward_json = fs::read(sandbox.root().join("reward.json")).ok();
    let reward_txt = fs::read(sandbox.root().join("reward.txt")).ok();
    RewardDocument::parse(reward_json.as_deref(), reward_txt.as_deref())
        .map_err(|error| EvalExecutionError::ProcessFailure(format!("verifier reward: {error}")))
}
