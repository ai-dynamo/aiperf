// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! The native `aiperf profile` command (single run + sweeps).
//!
//! Flow: parse flags → expand any comma-list sweep → for each cell load the
//! native [`BenchmarkRun`], serialize the protocol-v2 execute envelope, spawn the
//! unchanged `aiperf-runner`, and map its terminal outcome. A single run is a
//! degenerate one-cell sweep. YAML `--config` currently takes the single-run path.

use std::path::Path;

use crate::model::{Operation, RunnerRequest};
use crate::sweep::artifact_dir::IterationOrder;
use crate::sweep::{self, run as sweep_run};
use crate::{execute, flags::ProfileFlags, load, runner_install, yaml};

/// Eagerly create the artifact dir and remove any prior `native-v2.json` so a
/// re-run into the same directory doesn't trip the runner's write-once guard
/// (ports `rust_executor._clear_prior_report` + the eager mkdir Python does in
/// `setup_rich_logging`).
fn clear_prior_report(artifact_dir: &Path) {
    let _ = std::fs::create_dir_all(artifact_dir);
    let _ = std::fs::remove_file(artifact_dir.join("native-v2.json"));
}

/// Run `aiperf profile <args>` natively. Returns the process exit code.
pub fn run(args: &[String]) -> anyhow::Result<i32> {
    let flags = match ProfileFlags::parse_from_args(args) {
        Ok(flags) => flags,
        Err(err) => {
            err.print().ok();
            return Ok(err.exit_code());
        }
    };

    // YAML `--config` uses the native YAML surface (single run for now).
    if let Some(path) = &flags.config_file {
        let run = yaml::resolve(path, flags.artifact_dir.clone())?;
        return run_single(run);
    }

    let sweep_type = match flags.sweep_type.as_str() {
        "grid" => sweep::SweepType::Grid,
        "zip" => sweep::SweepType::Zip,
        other => anyhow::bail!("unknown --sweep-type {other:?} (grid/zip)"),
    };
    let expansion = sweep::expand(&flags, sweep_type)?;
    if !expansion.is_sweep {
        return run_single(load::resolve(&flags)?);
    }
    run_sweep(&flags, &expansion)
}

/// Execute one built run through the runner and map its terminal outcome, echoing
/// the runner's console summary to stdout on success.
fn run_single(run: crate::model::BenchmarkRun) -> anyhow::Result<i32> {
    let artifact_dir = run.artifact_dir.clone();
    clear_prior_report(&artifact_dir);
    let request = RunnerRequest::new(Operation::Execute, run);
    let payload = serde_json::to_vec(&request)
        .map_err(|e| anyhow::anyhow!("failed to serialize the runner request: {e}"))?;
    let runner = runner_install::resolve()?;
    let child_pid = crate::signals::install();
    let terminal = execute::run_once(&runner, &payload, &child_pid)?;
    if terminal.success {
        if let Some(path) = &terminal.report_path {
            crate::render::print_console_summary(path);
            tracing::info!(report = %path, "run complete");
        }
        Ok(0)
    } else {
        let detail = terminal
            .error
            .as_deref()
            .unwrap_or("native benchmark failed");
        eprintln!("aiperf: {detail}");
        Ok(if terminal.returncode == 0 {
            1
        } else {
            terminal.returncode
        })
    }
}

/// Execute a sweep: run every `(variation, trial)` cell in turn, then render the
/// sweep table and write the aggregate artifacts.
fn run_sweep(flags: &ProfileFlags, expansion: &sweep::Expansion) -> anyhow::Result<i32> {
    let sweep_id = uuid::Uuid::new_v4().simple().to_string();
    let base_seed = flags.random_seed.unwrap_or(sweep_run::DEFAULT_SWEEP_SEED);
    let cells = sweep_run::plan_cells(
        flags,
        expansion,
        1, // trials; multi-run trial repetition is a follow-up
        IterationOrder::Repeated,
        &sweep_id,
        base_seed,
        load::resolve,
    )?;

    let runner = runner_install::resolve()?;
    let child_pid = crate::signals::install();
    eprintln!("aiperf: sweep of {} runs", cells.len());
    let mut outcomes = Vec::new();
    for (n, cell) in cells.iter().enumerate() {
        eprintln!(
            "aiperf: [{}/{}] {} -> {}",
            n + 1,
            cells.len(),
            cell.label,
            cell.run.artifact_dir.display()
        );
        clear_prior_report(&cell.run.artifact_dir);
        let request = RunnerRequest::new(Operation::Execute, cell.run.clone());
        let payload = serde_json::to_vec(&request)?;
        let terminal = execute::run_once(&runner, &payload, &child_pid)?;
        outcomes.push(sweep::aggregate::CellOutcome {
            label: cell.label.clone(),
            values: cell.run.variation.clone(),
            artifact_dir: cell.run.artifact_dir.clone(),
            report_path: terminal.report_path.clone(),
            success: terminal.success,
        });
        if !terminal.success {
            eprintln!(
                "aiperf: cell failed: {}",
                terminal.error.as_deref().unwrap_or("(no detail)")
            );
        }
    }

    sweep::aggregate::finish(flags, &outcomes)?;
    let failed = outcomes.iter().filter(|o| !o.success).count();
    if failed > 0 {
        eprintln!("aiperf: {failed}/{} sweep cells failed", outcomes.len());
        Ok(1)
    } else {
        Ok(0)
    }
}
