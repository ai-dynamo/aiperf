// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
//! Native execution of one Harbor-compatible evaluation package.

use std::path::PathBuf;

use aiperf_runtime::eval::{
    HarborImporter, HarborSandboxRecipe, HarborSource, LocalProcessSandbox, NativeSourceAcquirer,
    VerifierMode,
};
use clap::{Parser, ValueEnum};
use serde::Serialize;

/// Run a single native Harbor-compatible package without a Harbor runtime.
#[derive(Debug, Parser)]
#[command(name = "eval", disable_help_subcommand = true)]
struct EvalFlags {
    /// Local task package JSON file.
    #[arg(long)]
    task: PathBuf,
    /// Immutable image identity used by the sandbox recipe.
    #[arg(long)]
    image: String,
    /// Absolute working directory inside the selected sandbox recipe.
    #[arg(long, default_value = "/work")]
    workdir: String,
    /// Whether the verifier shares the agent sandbox or receives a fresh root.
    #[arg(long, value_enum, default_value_t = VerifierModeFlag::Separate)]
    verifier_mode: VerifierModeFlag,
}

/// User-facing verifier sandbox topology.
#[derive(Clone, Copy, Debug, ValueEnum)]
enum VerifierModeFlag {
    /// Run the verifier in the task sandbox.
    Shared,
    /// Copy only declared artifacts into a fresh verifier sandbox.
    Separate,
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
    let source = HarborSource::local(flags.task.to_string_lossy())?;
    let imported = HarborImporter::new(&NativeSourceAcquirer).import(&source)?;
    let recipe = HarborSandboxRecipe::new(flags.image, flags.workdir)?;
    let result = LocalProcessSandbox::new().execute(
        &recipe,
        &imported.package,
        flags.verifier_mode.into(),
    )?;
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
