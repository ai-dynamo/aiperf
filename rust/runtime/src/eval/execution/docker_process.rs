// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

//! Docker-backed execution for conventional native task directories.

use std::{fs, process::Command};

use tempfile::TempDir;

use crate::eval::{HarborTaskPackage, RewardDocument, VerifierMode};

use super::{EvalExecutionError, HarborSandboxRecipe, LocalExecutionResult};

/// Executes a conventional task in a task-built Docker environment.
#[derive(Debug, Default)]
pub struct DockerProcessSandbox;

impl DockerProcessSandbox {
    /// Creates a Docker-backed task executor.
    pub const fn new() -> Self {
        Self
    }

    /// Builds the task environment, executes an external agent, and runs a shared verifier.
    pub fn execute(
        &self,
        recipe: &HarborSandboxRecipe,
        package: &HarborTaskPackage,
        agent_command: &[String],
        verifier_mode: VerifierMode,
    ) -> Result<LocalExecutionResult, EvalExecutionError> {
        if !package.is_standard_directory() {
            return Err(EvalExecutionError::Materialization(
                "Docker execution requires a standard task directory".to_owned(),
            ));
        }
        if verifier_mode != VerifierMode::Shared {
            return Err(EvalExecutionError::Materialization(
                "separate Docker verifier execution is not available yet".to_owned(),
            ));
        }
        let source_root = package.source_root().ok_or_else(|| {
            EvalExecutionError::Materialization(
                "standard task directory was not retained after import".to_owned(),
            )
        })?;
        let environment = source_root.join("environment");
        if !environment.join("Dockerfile").is_file() {
            return Err(EvalExecutionError::Materialization(
                "standard task is missing environment/Dockerfile".to_owned(),
            ));
        }
        let workspace = tempfile::tempdir()
            .map_err(|error| EvalExecutionError::Materialization(error.to_string()))?;
        let name_suffix = format!(
            "{}-{}",
            std::process::id(),
            package.source_digest().as_str()
        );
        let safe_suffix = name_suffix
            .chars()
            .filter(|character| character.is_ascii_alphanumeric())
            .take(32)
            .collect::<String>();
        let image = format!("aiperf-eval:{safe_suffix}");
        let container = format!("aiperf-eval-{safe_suffix}");
        let lease = DockerLease { container, image };
        docker(
            [
                "build",
                "--tag",
                &lease.image,
                environment.to_string_lossy().as_ref(),
            ],
            "build task environment",
        )?;
        docker(
            [
                "create",
                "--name",
                &lease.container,
                "--network",
                "none",
                "--workdir",
                &recipe.workdir,
                "--env",
                &format!("AIPERF_EVAL_INSTRUCTION={}", package.instruction()),
                "--volume",
                &format!("{}:{}", workspace.path().display(), recipe.workdir),
                &lease.image,
                "sleep",
                "infinity",
            ],
            "create task container",
        )?;
        docker(["start", &lease.container], "start task container")?;
        docker_exec(&lease.container, agent_command, "run agent")?;
        docker(
            [
                "cp",
                &format!("{}/.", source_root.join("tests").display()),
                &format!("{}:/tests", lease.container),
            ],
            "install verifier files",
        )?;
        docker(
            [
                "exec",
                "--user",
                "root",
                &lease.container,
                "/bin/sh",
                "-c",
                "mkdir -p /logs/verifier && chmod 0777 /logs /logs/verifier",
            ],
            "prepare verifier logs",
        )?;
        docker_exec(
            &lease.container,
            &["/bin/sh".to_owned(), "/tests/test.sh".to_owned()],
            "run verifier",
        )?;
        let reward = read_reward(&lease.container, &workspace)?;
        Ok(LocalExecutionResult {
            artifacts: Vec::new(),
            reward,
            verifier: package.source_digest(),
        })
    }
}

#[derive(Debug)]
struct DockerLease {
    container: String,
    image: String,
}

impl Drop for DockerLease {
    fn drop(&mut self) {
        let _ = Command::new("docker")
            .args(["rm", "--force", &self.container])
            .output();
        let _ = Command::new("docker")
            .args(["image", "rm", &self.image])
            .output();
    }
}

fn docker<'a>(
    arguments: impl IntoIterator<Item = &'a str>,
    action: &str,
) -> Result<Vec<u8>, EvalExecutionError> {
    let arguments = arguments.into_iter().collect::<Vec<_>>();
    let output = Command::new("docker")
        .args(&arguments)
        .output()
        .map_err(|_| EvalExecutionError::ProcessSpawn(format!("docker {action}")))?;
    if output.status.success() {
        Ok(output.stdout)
    } else {
        Err(EvalExecutionError::ProcessFailure(format!(
            "docker {action}: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        )))
    }
}

fn docker_exec(
    container: &str,
    command: &[String],
    action: &str,
) -> Result<(), EvalExecutionError> {
    if command.is_empty() || command.iter().any(|part| part.trim().is_empty()) {
        return Err(EvalExecutionError::InvalidCommand);
    }
    let mut arguments = vec!["exec", container];
    arguments.extend(command.iter().map(String::as_str));
    docker(arguments, action).map(|_| ())
}

fn read_reward(container: &str, workspace: &TempDir) -> Result<RewardDocument, EvalExecutionError> {
    let json = copy_optional(
        container,
        "/logs/verifier/reward.json",
        workspace,
        "reward.json",
    )?;
    let text = copy_optional(
        container,
        "/logs/verifier/reward.txt",
        workspace,
        "reward.txt",
    )?;
    RewardDocument::parse(json.as_deref(), text.as_deref())
        .map_err(|error| EvalExecutionError::ProcessFailure(format!("verifier reward: {error}")))
}

fn copy_optional(
    container: &str,
    source: &str,
    workspace: &TempDir,
    destination: &str,
) -> Result<Option<Vec<u8>>, EvalExecutionError> {
    let destination_path = workspace.path().join(destination);
    let output = Command::new("docker")
        .args([
            "cp",
            &format!("{container}:{source}"),
            destination_path.to_string_lossy().as_ref(),
        ])
        .output()
        .map_err(|_| EvalExecutionError::ProcessSpawn("docker copy verifier reward".to_owned()))?;
    if output.status.success() {
        fs::read(destination_path)
            .map(Some)
            .map_err(|error| EvalExecutionError::Materialization(error.to_string()))
    } else {
        Ok(None)
    }
}
