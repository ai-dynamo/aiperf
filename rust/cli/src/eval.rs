// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
//! Native execution of one Harbor-compatible evaluation package.

use std::{collections::BTreeMap, path::PathBuf, process::Command};

use aiperf_runtime::eval::{
    ArtifactDigest, DockerProcessSandbox, EvalExecutionError, HarborImporter, HarborSandboxRecipe,
    HarborSource, LocalExecutionResult, LocalProcessSandbox, MultiStepExecutionResult,
    NativeSourceAcquirer, VerifierMode,
};
use clap::{Parser, ValueEnum};
use serde::Serialize;

/// Run a single native Harbor-compatible package without a Harbor runtime.
#[derive(Debug, Parser)]
#[command(name = "eval", disable_help_subcommand = true)]
struct EvalFlags {
    /// Local task package JSON file.
    #[arg(long, conflicts_with_all = ["git_repository", "git_revision", "git_path"])]
    task: Option<PathBuf>,
    /// Local or remote Git repository containing a package pinned by `--git-revision`.
    #[arg(long, requires_all = ["git_revision", "git_path"])]
    git_repository: Option<String>,
    /// Exact 40-hex commit for a Git package source.
    #[arg(long, requires_all = ["git_repository", "git_path"])]
    git_revision: Option<String>,
    /// Repository-relative path to a Git package JSON file.
    #[arg(long, requires_all = ["git_repository", "git_revision"])]
    git_path: Option<String>,
    /// Immutable image identity used by the sandbox recipe.
    #[arg(long)]
    image: Option<String>,
    /// Absolute runtime working-directory override for a standard task.
    #[arg(long)]
    workdir: Option<String>,
    /// Whether the verifier shares the agent sandbox or receives a fresh root.
    #[arg(long, value_enum)]
    verifier_mode: Option<VerifierModeFlag>,
    /// Shell command for an external agent; required by standard task directories.
    #[arg(long)]
    agent_command: Option<String>,
    /// Sandbox backend; `auto` uses Docker for standard task directories.
    #[arg(long, value_enum, default_value_t = SandboxFlag::Auto)]
    sandbox: SandboxFlag,
}

/// User-facing verifier sandbox topology.
#[derive(Clone, Copy, Debug, ValueEnum)]
enum VerifierModeFlag {
    /// Run the verifier in the task sandbox.
    Shared,
    /// Copy only declared artifacts into a fresh verifier sandbox.
    Separate,
}

/// User-facing execution backend selection.
#[derive(Clone, Copy, Debug, ValueEnum)]
enum SandboxFlag {
    /// Use Docker for conventional task directories and local execution otherwise.
    Auto,
    /// Use the temporary-root local process backend.
    Local,
    /// Build and execute a conventional task directory in Docker.
    Docker,
}

impl From<VerifierModeFlag> for VerifierMode {
    fn from(value: VerifierModeFlag) -> Self {
        match value {
            VerifierModeFlag::Shared => Self::Shared,
            VerifierModeFlag::Separate => Self::Separate,
        }
    }
}

#[derive(Serialize)]
struct EvalOutput<'a> {
    task: &'a str,
    artifacts: &'a [(String, ArtifactDigest)],
    reward: &'a BTreeMap<String, f64>,
}

/// Completed evaluation output selected by the resolved task layout.
pub enum EvalExecutionResult {
    /// The unchanged output from an implicit single-step task.
    Single(LocalExecutionResult),
    /// The additive output from an explicit multi-step task.
    MultiStep(MultiStepExecutionResult),
}

#[derive(Serialize)]
struct MultiStepEvalOutput {
    task: String,
    artifacts: Vec<(String, ArtifactDigest)>,
    reward: BTreeMap<String, f64>,
    steps: Vec<MultiStepEvalStepOutput>,
}

#[derive(Serialize)]
struct MultiStepEvalStepOutput {
    name: String,
    artifacts: Vec<(String, ArtifactDigest)>,
    reward: BTreeMap<String, f64>,
}

/// Serializes a completed native evaluation result into its public JSON shape.
pub fn serialize_eval_result(
    task: &str,
    result: EvalExecutionResult,
) -> anyhow::Result<serde_json::Value> {
    match result {
        EvalExecutionResult::Single(result) => Ok(serde_json::to_value(EvalOutput {
            task,
            artifacts: &result.artifacts,
            reward: &result.reward.metrics,
        })?),
        EvalExecutionResult::MultiStep(result) => {
            let MultiStepExecutionResult { steps, reward, .. } = result;
            let artifacts = steps
                .last()
                .map(|step| step.artifacts.clone())
                .ok_or_else(|| {
                    anyhow::anyhow!("multi-step execution returned no verified steps")
                })?;
            Ok(serde_json::to_value(MultiStepEvalOutput {
                task: task.to_owned(),
                artifacts,
                reward: reward.metrics,
                steps: steps
                    .into_iter()
                    .map(|step| MultiStepEvalStepOutput {
                        name: step.name,
                        artifacts: step.artifacts,
                        reward: step.reward.metrics,
                    })
                    .collect(),
            })?)
        }
    }
}

/// Runs the native Harbor package lifecycle and emits one JSON summary.
pub fn run(args: &[String]) -> anyhow::Result<i32> {
    let flags =
        EvalFlags::try_parse_from(std::iter::once("eval".to_owned()).chain(args.iter().cloned()))?;
    let source = source_from_flags(
        flags.task,
        flags.git_repository,
        flags.git_revision,
        flags.git_path,
    )?;
    let (_pinned_tree, source) = materialize_pinned_directory(&source)?;
    let imported = HarborImporter::new(&NativeSourceAcquirer).import(&source)?;
    let use_docker = match flags.sandbox {
        SandboxFlag::Auto => imported.package.is_standard_directory(),
        SandboxFlag::Local => false,
        SandboxFlag::Docker => true,
    };
    let image = match flags.image {
        Some(image) => image,
        None if use_docker => {
            "sha256:0000000000000000000000000000000000000000000000000000000000000000".to_owned()
        }
        None => anyhow::bail!("--image is required for the local sandbox backend"),
    };
    let recipe = if imported.package.is_standard_directory() {
        HarborSandboxRecipe::for_standard_task(image, flags.workdir)?
    } else {
        HarborSandboxRecipe::new(image, flags.workdir.unwrap_or_else(|| "/work".to_owned()))?
    };
    let agent_command = flags
        .agent_command
        .as_deref()
        .map(agent_command_argv)
        .unwrap_or_else(|| imported.package.agent_command().to_vec());
    let requested_verifier_mode = flags.verifier_mode.map(VerifierMode::from);
    let verifier_mode = requested_verifier_mode.unwrap_or_else(|| imported.package.verifier_mode());
    if use_docker && imported.package.is_standard_directory() {
        let plan = imported.package.execution_plan();
        let has_verifier_mode_conflict = requested_verifier_mode.is_some_and(|requested| {
            if plan.is_multi_step() {
                plan.steps()
                    .iter()
                    .any(|step| requested != step.verifier().mode())
            } else {
                requested != plan.verifier().mode()
            }
        });
        if has_verifier_mode_conflict {
            anyhow::bail!(
                "--verifier-mode conflicts with the standard task's normalized verifier environment"
            );
        }
    }
    if imported.package.execution_plan().is_multi_step() {
        if !use_docker {
            return Err(EvalExecutionError::UnsupportedMultiStep.into());
        }
        let result = DockerProcessSandbox::new().execute_multi_step(
            &recipe,
            &imported.package,
            &agent_command,
        )?;
        println!(
            "{}",
            serde_json::to_string(&serialize_eval_result(
                imported.task.id.as_str(),
                EvalExecutionResult::MultiStep(result),
            )?)?
        );
    } else {
        let result = if use_docker {
            DockerProcessSandbox::new().execute(
                &recipe,
                &imported.package,
                &agent_command,
                verifier_mode,
            )?
        } else {
            LocalProcessSandbox::new().execute_with_agent_command(
                &recipe,
                &imported.package,
                &agent_command,
                verifier_mode,
            )?
        };
        println!(
            "{}",
            serde_json::to_string(&EvalOutput {
                task: imported.task.id.as_str(),
                artifacts: &result.artifacts,
                reward: &result.reward.metrics,
            })?
        );
    }
    Ok(0)
}

fn materialize_pinned_directory(
    source: &HarborSource,
) -> anyhow::Result<(Option<tempfile::TempDir>, HarborSource)> {
    let HarborSource::PinnedGit {
        repository,
        revision,
        package_path,
    } = source
    else {
        return Ok((None, source.clone()));
    };
    let root = tempfile::tempdir()?;
    let checkout = root.path().join("repository");
    let clone = Command::new("git")
        .args(["clone", "--no-checkout", repository])
        .arg(&checkout)
        .output()?;
    if !clone.status.success() {
        anyhow::bail!(
            "unable to clone pinned Git repository: {}",
            String::from_utf8_lossy(&clone.stderr).trim()
        );
    }
    let checkout_result = Command::new("git")
        .args([
            "-C",
            checkout.to_string_lossy().as_ref(),
            "checkout",
            "--detach",
            revision,
        ])
        .output()?;
    if !checkout_result.status.success() {
        anyhow::bail!(
            "unable to check out pinned Git revision: {}",
            String::from_utf8_lossy(&checkout_result.stderr).trim()
        );
    }
    let package = checkout.join(package_path);
    let location = if package_path.ends_with("task.toml") {
        package
            .parent()
            .map_or_else(|| checkout.clone(), std::path::Path::to_path_buf)
    } else {
        package
    };
    Ok((Some(root), HarborSource::local(location.to_string_lossy())?))
}

fn agent_command_argv(command: &str) -> Vec<String> {
    vec!["/bin/sh".to_owned(), "-c".to_owned(), command.to_owned()]
}

fn source_from_flags(
    task: Option<PathBuf>,
    git_repository: Option<String>,
    git_revision: Option<String>,
    git_path: Option<String>,
) -> anyhow::Result<HarborSource> {
    match (task, git_repository, git_revision, git_path) {
        (Some(path), None, None, None) => Ok(HarborSource::local(path.to_string_lossy())?),
        (None, Some(repository), Some(revision), Some(path)) => {
            Ok(HarborSource::pinned_git(repository, revision, path)?)
        }
        _ => anyhow::bail!(
            "provide either --task or --git-repository with --git-revision and --git-path"
        ),
    }
}

#[cfg(test)]
mod tests {
    use std::{fs, process::Command};

    use super::{agent_command_argv, materialize_pinned_directory, source_from_flags};
    use aiperf_runtime::eval::HarborSource;

    #[test]
    fn source_flags_require_one_complete_source_form() {
        assert!(source_from_flags(Some("task.json".into()), None, None, None).is_ok());
        assert!(
            source_from_flags(
                None,
                Some("tasks".to_owned()),
                Some("a".repeat(40)),
                Some("task.json".to_owned()),
            )
            .is_ok()
        );
        assert!(source_from_flags(None, Some("tasks".to_owned()), None, None).is_err());
    }

    #[test]
    fn external_agent_command_uses_a_shell_argv() {
        assert_eq!(
            agent_command_argv("printf result > result.txt"),
            ["/bin/sh", "-c", "printf result > result.txt"]
        );
    }

    #[test]
    fn materializes_a_pinned_standard_task_tree() {
        let temporary = tempfile::tempdir().unwrap();
        let repository = temporary.path().join("tasks");
        fs::create_dir(&repository).unwrap();
        for arguments in [
            ["init"].as_slice(),
            ["config", "user.email", "eval@example.invalid"].as_slice(),
            ["config", "user.name", "Eval"].as_slice(),
        ] {
            assert!(
                Command::new("git")
                    .arg("-C")
                    .arg(&repository)
                    .args(arguments)
                    .status()
                    .unwrap()
                    .success()
            );
        }
        fs::create_dir_all(repository.join("task/environment")).unwrap();
        fs::create_dir_all(repository.join("task/tests")).unwrap();
        fs::write(
            repository.join("task/task.toml"),
            "schema_version = \"1.0\"\n[task]\nname = \"example/pinned\"\n",
        )
        .unwrap();
        fs::write(repository.join("task/instruction.md"), "Do work.\n").unwrap();
        fs::write(
            repository.join("task/environment/Dockerfile"),
            "FROM scratch\n",
        )
        .unwrap();
        fs::write(repository.join("task/tests/test.sh"), "exit 0\n").unwrap();
        assert!(
            Command::new("git")
                .arg("-C")
                .arg(&repository)
                .args(["add", "."])
                .status()
                .unwrap()
                .success()
        );
        assert!(
            Command::new("git")
                .arg("-c")
                .arg("commit.gpgsign=false")
                .arg("-C")
                .arg(&repository)
                .args(["commit", "-m", "task"])
                .status()
                .unwrap()
                .success()
        );
        let revision = String::from_utf8(
            Command::new("git")
                .arg("-C")
                .arg(&repository)
                .args(["rev-parse", "HEAD"])
                .output()
                .unwrap()
                .stdout,
        )
        .unwrap()
        .trim()
        .to_owned();
        let (tree, source) = materialize_pinned_directory(
            &HarborSource::pinned_git(repository.to_string_lossy(), revision, "task/task.toml")
                .unwrap(),
        )
        .unwrap();
        assert!(tree.unwrap().path().join("task/task.toml").is_file());
        assert!(matches!(source, HarborSource::Local(_)));
    }
}
