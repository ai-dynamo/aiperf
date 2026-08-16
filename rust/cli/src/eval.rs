// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
//! Native execution of one Harbor-compatible evaluation package.

use std::path::PathBuf;

use aiperf_runtime::eval::{
    DockerProcessSandbox, HarborImporter, HarborSandboxRecipe, HarborSource, LocalProcessSandbox,
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
    image: String,
    /// Absolute working directory inside the selected sandbox recipe.
    #[arg(long, default_value = "/work")]
    workdir: String,
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
    artifacts: &'a [(String, aiperf_runtime::eval::ArtifactDigest)],
    reward: &'a std::collections::BTreeMap<String, f64>,
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
    let imported = HarborImporter::new(&NativeSourceAcquirer).import(&source)?;
    let recipe = HarborSandboxRecipe::new(flags.image, flags.workdir)?;
    let agent_command = flags
        .agent_command
        .as_deref()
        .map(agent_command_argv)
        .unwrap_or_else(|| imported.package.agent_command().to_vec());
    let verifier_mode = flags
        .verifier_mode
        .map(VerifierMode::from)
        .unwrap_or_else(|| imported.package.verifier_mode());
    let use_docker = match flags.sandbox {
        SandboxFlag::Auto => imported.package.is_standard_directory(),
        SandboxFlag::Local => false,
        SandboxFlag::Docker => true,
    };
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
    Ok(0)
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
    use super::{agent_command_argv, source_from_flags};

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
}
