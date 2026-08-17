// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

//! Temporary-root local process execution for shared-verifier compatibility.

use std::{
    fs,
    io::Read,
    path::{Path, PathBuf},
    process::{Command, ExitStatus, Stdio},
};

use tempfile::TempDir;

use crate::eval::{
    ArtifactDigest, AttemptId, HarborTaskPackage, RegradeError, RewardDocument, ScoreVersion,
    VerifierMode,
};

use super::{EvalExecutionError, HarborSandboxRecipe, ProviderCapabilities};

const MAX_LOCAL_FILE_BYTES: u64 = 1024 * 1024;

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

    /// Runs the package agent and an explicitly shared verifier in one temporary root.
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
        if verifier_mode == VerifierMode::Separate {
            return Err(EvalExecutionError::UnsupportedEnforcement(
                "separate verifier isolation",
            ));
        }
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
        agent.run(package.verifier_command(), &environment)?;
        let reward = parse_reward(&agent)?;
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
        let status = Command::new(program)
            .args(arguments)
            .current_dir(self.root())
            .env_clear()
            .env("PATH", "/usr/bin:/bin")
            .env("AIPERF_EVAL_ROOT", self.root())
            .envs(environment.iter().map(|(key, value)| (key, value)))
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .map_err(|_| EvalExecutionError::ProcessSpawn(program.clone()))?;
        if !status.success() {
            return Err(EvalExecutionError::ProcessFailure(program.clone()));
        }
        Ok(ProcessOutput {
            status,
            stdout: Vec::new(),
            stderr: Vec::new(),
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
            let bytes = read_file_bounded(&artifact_path)?;
            Ok((path.clone(), ArtifactDigest::from_bytes(&bytes)))
        })
        .collect()
}

fn parse_reward(sandbox: &MaterializedSandbox) -> Result<RewardDocument, EvalExecutionError> {
    let reward_json = read_optional_file_bounded(&sandbox.root().join("reward.json"))?;
    let reward_txt = read_optional_file_bounded(&sandbox.root().join("reward.txt"))?;
    RewardDocument::parse(reward_json.as_deref(), reward_txt.as_deref())
        .map_err(|error| EvalExecutionError::ProcessFailure(format!("verifier reward: {error}")))
}

fn read_optional_file_bounded(path: &Path) -> Result<Option<Vec<u8>>, EvalExecutionError> {
    match fs::File::open(path) {
        Ok(file) => read_open_file_bounded(file).map(Some),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(EvalExecutionError::ArtifactCollection(error.to_string())),
    }
}

fn read_file_bounded(path: &Path) -> Result<Vec<u8>, EvalExecutionError> {
    let file = fs::File::open(path)
        .map_err(|error| EvalExecutionError::ArtifactCollection(error.to_string()))?;
    read_open_file_bounded(file)
}

fn read_open_file_bounded(file: fs::File) -> Result<Vec<u8>, EvalExecutionError> {
    let metadata = file
        .metadata()
        .map_err(|error| EvalExecutionError::ArtifactCollection(error.to_string()))?;
    if metadata.len() > MAX_LOCAL_FILE_BYTES {
        return Err(EvalExecutionError::ArtifactCollection(
            "local artifact exceeds the maximum size".to_owned(),
        ));
    }
    let mut bytes = Vec::with_capacity(metadata.len() as usize);
    file.take(MAX_LOCAL_FILE_BYTES + 1)
        .read_to_end(&mut bytes)
        .map_err(|error| EvalExecutionError::ArtifactCollection(error.to_string()))?;
    if bytes.len() as u64 > MAX_LOCAL_FILE_BYTES {
        return Err(EvalExecutionError::ArtifactCollection(
            "local artifact exceeds the maximum size".to_owned(),
        ));
    }
    Ok(bytes)
}
