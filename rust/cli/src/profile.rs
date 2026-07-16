// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! The native `aiperf profile` command (single run + sweeps).
//!
//! Flow: parse flags → expand any comma-list sweep (or a YAML `sweep:` block) →
//! for each cell load the native [`BenchmarkRun`], serialize the protocol-v2
//! execute envelope, spawn the unchanged `aiperf-runner`, and map its terminal
//! outcome. A single run is a degenerate one-cell sweep. A YAML `--config` with a
//! `sweep:` block expands to a native sweep; otherwise it is one run.

use std::path::Path;

use crate::model::{Operation, RunnerRequest};
use crate::sweep::artifact_dir::IterationOrder;
use crate::sweep::{self, run as sweep_run};
use crate::{delegate, execute, flags::ProfileFlags, load, runner_install, yaml};

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
            use clap::error::ErrorKind;
            // A flag the native front door doesn't model is very likely a
            // Python-only surface (accuracy benchmarks, adaptive/BO search
            // recipes, synthesis knobs, …). Rather than fail, transparently
            // delegate the whole run to the pure-Python `aiperf profile`, so the
            // native binary is at least as capable as Python for every flag
            // (100% parity by construction). The native fast path still owns
            // every flag it DOES model. `--help`/`--version` and genuine value
            // errors are surfaced normally.
            if matches!(err.kind(), ErrorKind::UnknownArgument) {
                return delegate::exec_python_profile(args);
            }
            err.print().ok();
            return Ok(err.exit_code());
        }
    };

    // YAML `--config`: a `sweep:` block expands to a native sweep; otherwise it
    // is one run through the native YAML surface.
    if let Some(path) = &flags.config_file {
        let mut base = yaml::read_env_substituted(path)?;
        if let Some(sweep) = crate::sweep::yaml_sweep::parse(&base)? {
            // Normalize dataset/model/warmup shorthands to their list forms so
            // dotted sweep paths (e.g. `datasets.default.prompts.isl`) resolve.
            crate::sweep::yaml_sweep::normalize_benchmark(&mut base);
            return run_yaml_sweep(&flags, base, sweep);
        }
        let expanded = crate::expand::render_with_context(base)?;
        let run = yaml::resolve_expanded_value(expanded, flags.artifact_dir.clone())?;
        return run_single(run);
    }

    let sweep_type = match flags.sweep_type.as_str() {
        "grid" => sweep::SweepType::Grid,
        "zip" => sweep::SweepType::Zip,
        other => anyhow::bail!("unknown --sweep-type {other:?} (grid/zip)"),
    };
    let expansion = sweep::expand(&flags, sweep_type)?;
    let trials = flags.num_profile_runs.unwrap_or(1).max(1);
    if !expansion.is_sweep && trials <= 1 {
        return run_single(load::resolve(&flags)?);
    }
    let order = match flags.parameter_sweep_mode.as_str() {
        "independent" => IterationOrder::Independent,
        "repeated" => IterationOrder::Repeated,
        other => anyhow::bail!("unknown --parameter-sweep-mode {other:?} (repeated/independent)"),
    };
    run_sweep(&flags, &expansion, trials, order)
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

/// Execute a YAML `sweep:` block: expand its variations, resolve+stamp each into
/// a run, then run every cell and write the aggregate (single trial per cell).
fn run_yaml_sweep(
    flags: &ProfileFlags,
    base: serde_json::Value,
    sweep: crate::sweep::yaml_sweep::YamlSweep,
) -> anyhow::Result<i32> {
    let sweep_id = uuid::Uuid::new_v4().simple().to_string();
    let cells = plan_yaml_cells(flags.artifact_dir.clone(), &base, &sweep, &sweep_id)?;
    run_cells(flags, &cells)
}

/// Expand a YAML `sweep:` block into stamped per-cell runs (single trial each).
/// Each variation is Jinja-rendered, resolved to a run, and stamped with the
/// sweep envelope (`sweep_id`, `variation`, `random_seed = base_seed + index`)
/// and its per-cell artifact dir. Testable independently of execution.
pub fn plan_yaml_cells(
    artifact_dir: Option<std::path::PathBuf>,
    base: &serde_json::Value,
    sweep: &crate::sweep::yaml_sweep::YamlSweep,
    sweep_id: &str,
) -> anyhow::Result<Vec<sweep_run::Cell>> {
    let variations = sweep.expand(base)?;
    // Base seed: the config's `randomSeed` (or the shared default), then `+index`.
    let base_seed = base
        .get("randomSeed")
        .or_else(|| base.get("random_seed"))
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(sweep_run::DEFAULT_SWEEP_SEED);

    let mut cells = Vec::with_capacity(variations.len());
    for v in &variations {
        let expanded = crate::expand::render_with_context(v.config.clone())?;
        let mut run = yaml::resolve_expanded_value(expanded, artifact_dir.clone())?;
        let dir = crate::sweep::artifact_dir::resolve(
            &run.artifact_dir,
            true,
            1,
            &v.dir_name,
            0,
            IterationOrder::Repeated,
        );
        run.sweep_id = Some(sweep_id.to_string());
        let values: serde_json::Map<String, serde_json::Value> =
            v.values.iter().map(|(k, val)| (k.clone(), val.clone())).collect();
        run.variation = Some(serde_json::json!({
            "index": v.index,
            "label": v.label,
            "values": values,
        }));
        run.random_seed = Some(base_seed + v.index as u64);
        run.trial = 0;
        run.artifact_dir = dir;
        cells.push(sweep_run::Cell {
            index: v.index,
            trial: 0,
            label: v.label.clone(),
            run,
        });
    }
    Ok(cells)
}

/// Execute a sweep: run every `(variation, trial)` cell in turn, then render the
/// sweep table and write the aggregate artifacts.
fn run_sweep(
    flags: &ProfileFlags,
    expansion: &sweep::Expansion,
    trials: u32,
    order: IterationOrder,
) -> anyhow::Result<i32> {
    let sweep_id = uuid::Uuid::new_v4().simple().to_string();
    let base_seed = flags.random_seed.unwrap_or(sweep_run::DEFAULT_SWEEP_SEED);
    let disable_warmup = !flags.no_profile_run_disable_warmup_after_first
        && flags.profile_run_disable_warmup_after_first;
    let cells = sweep_run::plan_cells(
        flags,
        expansion,
        trials,
        order,
        &sweep_id,
        base_seed,
        disable_warmup,
        load::resolve,
    )?;
    run_cells(flags, &cells)
}

/// Run every planned cell in turn (with an optional inter-cell cooldown), render
/// the sweep table, and write the aggregate. Shared by the flag-driven sweep,
/// the multi-run path, and the YAML `sweep:` path.
fn run_cells(flags: &ProfileFlags, cells: &[sweep_run::Cell]) -> anyhow::Result<i32> {
    let runner = runner_install::resolve()?;
    let child_pid = crate::signals::install();
    let cooldown = flags
        .profile_run_cooldown_seconds
        .filter(|s| *s > 0.0)
        .map(std::time::Duration::from_secs_f64);
    eprintln!("aiperf: {} runs", cells.len());
    let mut outcomes = Vec::new();
    for (n, cell) in cells.iter().enumerate() {
        if let Some(d) = cooldown
            && n > 0
        {
            std::thread::sleep(d);
        }
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
            trial: cell.trial,
            error: terminal.error.clone(),
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
