// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

//! Docker-backed execution for conventional native task directories.

use std::{fs, process::Command};

use tempfile::TempDir;

use crate::eval::{ArtifactDigest, HarborTaskPackage, RewardDocument, VerifierMode};

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
        create_container(
            &lease.container,
            &lease.image,
            workspace.path(),
            recipe,
            package,
            true,
        )?;
        docker(["start", &lease.container], "start task container")?;
        docker_exec(&lease.container, agent_command, "run agent")?;
        let artifacts = collect_workspace_artifacts(&workspace, recipe, package)?;
        let verifier_workspace = tempfile::tempdir()
            .map_err(|error| EvalExecutionError::Materialization(error.to_string()))?;
        let verifier_container = if verifier_mode == VerifierMode::Separate {
            copy_workspace_artifacts(&workspace, &verifier_workspace, recipe, package)?;
            let container = format!("{}-verifier", lease.container);
            let verifier_lease = ContainerLease { container };
            create_container(
                &verifier_lease.container,
                &lease.image,
                verifier_workspace.path(),
                recipe,
                package,
                false,
            )?;
            docker(
                ["start", &verifier_lease.container],
                "start separate verifier container",
            )?;
            Some(verifier_lease)
        } else {
            None
        };
        let verifier = verifier_container
            .as_ref()
            .map_or(lease.container.as_str(), |lease| lease.container.as_str());
        docker(
            [
                "cp",
                &format!("{}/.", source_root.join("tests").display()),
                &format!("{verifier}:/tests"),
            ],
            "install verifier files",
        )?;
        docker(
            [
                "exec",
                "--user",
                "root",
                verifier,
                "/bin/sh",
                "-c",
                "mkdir -p /logs/verifier && chmod 0777 /logs /logs/verifier",
            ],
            "prepare verifier logs",
        )?;
        docker_exec(
            verifier,
            &["/bin/sh".to_owned(), "/tests/test.sh".to_owned()],
            "run verifier",
        )?;
        let reward = read_reward(
            verifier,
            if verifier_mode == VerifierMode::Separate {
                &verifier_workspace
            } else {
                &workspace
            },
        )?;
        Ok(LocalExecutionResult {
            artifacts,
            reward,
            verifier: package.source_digest(),
        })
    }
}

fn create_container(
    container: &str,
    image: &str,
    workspace: &std::path::Path,
    recipe: &HarborSandboxRecipe,
    package: &HarborTaskPackage,
    has_agent_instruction: bool,
) -> Result<(), EvalExecutionError> {
    let mut arguments = vec![
        "create".to_owned(),
        "--name".to_owned(),
        container.to_owned(),
        "--network".to_owned(),
        "none".to_owned(),
        "--workdir".to_owned(),
        recipe.workdir.clone(),
        "--volume".to_owned(),
        format!("{}:{}", workspace.display(), recipe.workdir),
    ];
    if let Some((cpus, memory_mb)) = package.container_resources() {
        arguments.extend([
            "--cpus".to_owned(),
            cpus.to_string(),
            "--memory".to_owned(),
            format!("{memory_mb}m"),
        ]);
    }
    if has_agent_instruction {
        arguments.extend([
            "--env".to_owned(),
            format!("AIPERF_EVAL_INSTRUCTION={}", package.instruction()),
        ]);
    }
    arguments.extend([image.to_owned(), "sleep".to_owned(), "infinity".to_owned()]);
    docker(
        arguments.iter().map(String::as_str),
        "create task container",
    )
    .map(|_| ())
}

#[derive(Debug)]
struct ContainerLease {
    container: String,
}

impl Drop for ContainerLease {
    fn drop(&mut self) {
        let _ = Command::new("docker")
            .args(["rm", "--force", &self.container])
            .output();
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

fn collect_workspace_artifacts(
    workspace: &TempDir,
    recipe: &HarborSandboxRecipe,
    package: &HarborTaskPackage,
) -> Result<Vec<(String, ArtifactDigest)>, EvalExecutionError> {
    package
        .declared_artifacts()
        .iter()
        .map(|path| {
            let relative = path
                .strip_prefix(&recipe.workdir)
                .and_then(|path| path.strip_prefix('/'))
                .ok_or_else(|| {
                    EvalExecutionError::Materialization(format!(
                        "Docker artifact must be under {}: {path}",
                        recipe.workdir
                    ))
                })?;
            let bytes = fs::read(workspace.path().join(relative))
                .map_err(|error| EvalExecutionError::Materialization(error.to_string()))?;
            Ok((path.clone(), ArtifactDigest::from_bytes(&bytes)))
        })
        .collect()
}

fn copy_workspace_artifacts(
    source: &TempDir,
    destination: &TempDir,
    recipe: &HarborSandboxRecipe,
    package: &HarborTaskPackage,
) -> Result<(), EvalExecutionError> {
    for path in package.declared_artifacts() {
        let relative = path
            .strip_prefix(&recipe.workdir)
            .and_then(|path| path.strip_prefix('/'))
            .ok_or_else(|| {
                EvalExecutionError::Materialization(format!(
                    "Docker artifact must be under {}: {path}",
                    recipe.workdir
                ))
            })?;
        let destination_path = destination.path().join(relative);
        if let Some(parent) = destination_path.parent() {
            fs::create_dir_all(parent)
                .map_err(|error| EvalExecutionError::Materialization(error.to_string()))?;
        }
        fs::copy(source.path().join(relative), destination_path)
            .map_err(|error| EvalExecutionError::Materialization(error.to_string()))?;
    }
    Ok(())
}
