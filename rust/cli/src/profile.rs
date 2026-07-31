// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! The native `aiperf profile` command (single run + sweeps).
//!
//! Resolves profile inputs into protocol-v2 runs and executes single runs or
//! sweeps through the `aiperf` binary.

use std::path::Path;

use crate::search_history::{HistoryConfig, IterationRecord};
use crate::sweep::artifact_dir::IterationOrder;
use crate::sweep::{self, run as sweep_run};
use crate::{exec_bin, execute, flags::ProfileFlags, load, yaml};

/// Remove a prior report before the runner's write-once output.
fn clear_prior_report(artifact_dir: &Path) {
    let _ = std::fs::create_dir_all(artifact_dir);
    let _ = std::fs::remove_file(artifact_dir.join("native-v2.json"));
}

/// Authoring-tagged `--execute` wire body.
///
/// The single-run and flag/YAML sweep paths send normalized authoring
/// [`load::Inputs`] under an `authoring` tag so the runtime resolves them at
/// `--execute` (matching the runtime
/// [`decode_execute_wire`](aiperf_runtime::engine::protocol_v2::decode_execute_wire)
/// union). The sweep paths additionally carry the per-cell sweep envelope
/// (`sweep_id`/`variation`/`trial`) so the runtime overlays it onto the resolved
/// run (per-cell `artifact_dir`/`random_seed` ride inside the `Inputs`). The CLI
/// therefore no longer resolves before the child launch — except that a configured
/// `reset_kv_cache` / `server_profiler` endpoint control hook is a live pre-flight
/// action that still needs the resolved endpoint, so that (rare) case resolves
/// locally purely to run the hook.
#[derive(serde::Serialize)]
struct AuthoringWire<'a> {
    authoring: &'a load::Inputs,
    #[serde(skip_serializing_if = "Option::is_none")]
    sweep_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    variation: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "trial_is_zero")]
    trial: u32,
}

/// A zero trial is the single-run/base default; omit it from the wire so the
/// single-run envelope stays `{"authoring": …}` exactly as before.
fn trial_is_zero(trial: &u32) -> bool {
    *trial == 0
}

/// Drive one child from authoring [`load::Inputs`], sending the authoring wire body
/// so the runtime performs the authoritative resolution at `--execute`. `envelope`
/// supplies the per-cell sweep coordinates (`sweep_id`/`variation`/`trial`) for the
/// sweep paths; the single-run path passes `None` for a bare `{"authoring": …}` body.
fn run_benchmark_child_authoring(
    inputs: &load::Inputs,
    envelope: Option<&crate::model::BenchmarkRun>,
    runner: &Path,
    child_pid: &crate::signals::ChildPid,
) -> anyhow::Result<crate::execute::Terminal> {
    // A live endpoint control hook (reset_kv_cache / server_profiler) must run before
    // the child launches and needs the resolved endpoint; resolve locally only then.
    if inputs.reset_kv_cache.is_some() || inputs.server_profiler.is_some() {
        let run = load::build(inputs.clone())?;
        crate::control_hooks::run_reset_kv_cache_before_run(&run)?;
    }
    clear_prior_report(&inputs.artifact_dir);
    let wire = AuthoringWire {
        authoring: inputs,
        sweep_id: envelope.and_then(|r| r.sweep_id.clone()),
        variation: envelope.and_then(|r| r.variation.clone()),
        trial: envelope.map(|r| r.trial).unwrap_or(0),
    };
    let payload = serde_json::to_vec(&wire)
        .map_err(|e| anyhow::anyhow!("failed to serialize the runner request: {e}"))?;
    execute::run_once(runner, &payload, child_pid)
}

/// Ship one adaptive-search probe on the authoring wire.
///
/// Clones the base authoring [`load::Inputs`], overrides the swept profiling
/// concurrency on the authoring scalar (the runtime resolves it at `--execute`,
/// expanding it exactly as the old resolved-cfg override did), stamps the per-cell
/// artifact dir + run seed onto the inputs, and carries the sweep envelope
/// (`sweep_id`/`variation`/`trial`) on a clone of the base run. The CLI no longer
/// resolves the probe run itself — the runtime does.
#[allow(clippy::too_many_arguments)]
fn run_search_probe(
    base_inputs: &load::Inputs,
    base: &crate::model::BenchmarkRun,
    sweep_id: &str,
    iter: i64,
    label: &str,
    dir: std::path::PathBuf,
    value: i64,
    seed: Option<u64>,
    runner: &Path,
    child_pid: &crate::signals::ChildPid,
) -> anyhow::Result<crate::execute::Terminal> {
    let mut inputs = base_inputs.clone();
    crate::search::apply_override_inputs(
        &mut inputs,
        crate::search::AxisKind::PhaseConcurrency,
        value,
    );
    inputs.artifact_dir = dir;
    inputs.random_seed = seed;
    let mut envelope = base.clone();
    envelope.sweep_id = Some(sweep_id.to_string());
    envelope.variation = Some(serde_json::json!({
        "index": iter,
        "label": label,
        "values": { "phases.profiling.concurrency": value },
    }));
    envelope.trial = 0;
    run_benchmark_child_authoring(&inputs, Some(&envelope), runner, child_pid)
}

/// Run `aiperf profile <args>` natively. Returns the process exit code.
pub fn run(args: &[String]) -> anyhow::Result<i32> {
    let mut flags = match ProfileFlags::parse_from_args(args) {
        Ok(flags) => flags,
        Err(err) => {
            err.print().ok();
            return Ok(err.exit_code());
        }
    };

    // A config's `multiRun:` block selects the same trial policy as the
    // `--num-profile-runs` family, so it is folded onto the flags before
    // validation and before either config path branches.
    let config_base = match &flags.config_file {
        Some(path) => {
            let base = yaml::read_env_substituted(path)?;
            yaml::apply_multi_run(&base, &mut flags)?;
            Some(base)
        }
        None => None,
    };
    let flags = flags;

    validate_multi_run(&flags)?;

    if let Some(mut base) = config_base {
        if let Some(sweep) = crate::sweep::yaml_sweep::parse(&base)? {
            // Normalize dataset/model/warmup shorthands to their list forms so
            // dotted sweep paths (e.g. `datasets.default.prompts.isl`) resolve.
            crate::sweep::yaml_sweep::normalize_benchmark(&mut base);
            return run_yaml_sweep(&flags, base, sweep);
        }
        let expanded = crate::expand::render_with_context(base)?;
        let inputs =
            yaml::resolve_expanded_inputs(expanded, flags.artifact_dir.clone(), Some(&flags))?;
        // `multiRun.numRuns > 1` without a `sweep:` block is still a repeated
        // run: dropping to `run_single` would silently execute one trial.
        let trials = flags.num_profile_runs.unwrap_or(1);
        if trials > 1 {
            return run_config_trials(&flags, inputs, trials);
        }
        return run_single(inputs);
    }

    // `max-concurrency-under-sla --search-style monotonic` runs a dynamic
    // ask-tell loop (exponential probe + bisection) rather than a static sweep:
    // each probe is one runner invocation whose feasibility verdict steers the
    // next probe. Intercept it before the static-grid recipe expander.
    if flags.search_recipe.as_deref() == Some("max-concurrency-under-sla")
        && flags.search_style.as_deref() == Some("monotonic")
    {
        return run_search_loop(&flags);
    }

    // Smooth isotonic search requires the `search-pyo3` scipy integration.
    #[cfg(feature = "search-pyo3")]
    if flags.search_recipe.as_deref() == Some("max-concurrency-under-sla")
        && matches!(
            flags.search_style.as_deref(),
            None | Some("smooth_isotonic")
        )
    {
        return run_isotonic_loop(&flags);
    }

    // Bayesian search requires the `search-pyo3` Optuna integration.
    #[cfg(feature = "search-pyo3")]
    if flags.search_recipe.as_deref() == Some("max-concurrency-under-sla")
        && matches!(flags.search_style.as_deref(), Some("bo") | Some("optuna"))
    {
        return run_bayes_loop(&flags);
    }

    // `max-goodput-under-slo` maximizes goodput under a
    // `good_request_fraction >= --slo-attainment-fraction` outcome constraint via
    // the same optuna BO seam; per-request TTFT/TPOT/E2E SLOs are installed as
    // config `slos` so the runtime computes goodput / good_request_fraction.
    #[cfg(feature = "search-pyo3")]
    if flags.search_recipe.as_deref() == Some("max-goodput-under-slo") {
        return run_goodput_loop(&flags);
    }

    // A grid `--search-recipe` expands its search space into a static grid sweep
    // over config paths (mutating the built cfg per variation). (bayes/isotonic
    // recipes run a dynamic ask-tell loop, handled elsewhere.)
    if let Some(recipe) = crate::search::expand_recipe(&flags)? {
        return run_recipe_sweep(&flags, recipe);
    }

    let sweep_type = match flags.sweep_type.as_str() {
        "grid" => sweep::SweepType::Grid,
        "zip" => sweep::SweepType::Zip,
        other => anyhow::bail!("unknown --sweep-type {other:?} (grid/zip)"),
    };
    let expansion = sweep::expand(&flags, sweep_type)?;
    let trials = flags.num_profile_runs.unwrap_or(1);
    if !expansion.is_sweep && trials <= 1 {
        return run_single(load::resolve_inputs(&flags)?);
    }
    let order = match flags.parameter_sweep_mode.as_str() {
        "independent" => IterationOrder::Independent,
        "repeated" => IterationOrder::Repeated,
        other => anyhow::bail!("unknown --parameter-sweep-mode {other:?} (repeated/independent)"),
    };
    run_sweep(&flags, &expansion, trials, order)
}

/// Flags the native binary parses for Python-CLI compatibility but does not act
/// on. Each is accepted so an existing command line keeps working, and each is
/// warned about so a run never silently ignores an authored intent.
///
/// Entries leave this table by gaining a consumer, not by being deleted: dropping
/// one returns the flag to silently-ignored, which is the failure this guards.
type FlagIsAuthored = fn(&ProfileFlags) -> bool;

const UNIMPLEMENTED_FLAGS: &[(&str, FlagIsAuthored)] = &[
    ("--adaptive-scale-control", |f| {
        f.adaptive_scale_control.is_some()
    }),
    ("--api-host", |f| f.api_host.is_some()),
    ("--api-port", |f| f.api_port.is_some()),
    ("--bo-constraint-mode", |f| f.bo_constraint_mode.is_some()),
    ("--convergence-mode", |f| f.convergence_mode.is_some()),
    ("--convergence-stat", |f| f.convergence_stat.is_some()),
    ("--convergence-threshold", |f| {
        f.convergence_threshold.is_some()
    }),
    ("--extra-verbose", |f| f.extra_verbose.is_some()),
    ("--optuna-acquisition", |f| f.optuna_acquisition.is_some()),
    ("--optuna-terminator", |f| f.optuna_terminator.is_some()),
    ("--plot-required", |f| f.plot_required.is_some()),
    ("--record-processor-service-count", |f| {
        f.record_processor_service_count.is_some()
    }),
    ("--search-direction", |f| f.search_direction.is_some()),
    ("--search-metric", |f| f.search_metric.is_some()),
    ("--search-percentile-pooling", |f| {
        f.search_percentile_pooling.is_some()
    }),
    ("--search-sla", |f| f.search_sla.is_some()),
    ("--search-sla-tier", |f| f.search_sla_tier.is_some()),
    ("--search-stat", |f| f.search_stat.is_some()),
    ("--stats-interval", |f| f.stats_interval.is_some()),
    ("--ui-type", |f| f.ui_type.is_some()),
    ("--workers-max", |f| f.workers_max.is_some()),
    ("--zmq-dual-bind", |f| f.zmq_dual_bind.is_some()),
    ("--zmq-host", |f| f.zmq_host.is_some()),
    ("--zmq-ipc-path", |f| f.zmq_ipc_path.is_some()),
];

/// Warn once per authored flag the native binary accepts but does not act on.
fn warn_unimplemented_flags(flags: &ProfileFlags) {
    let authored: Vec<&str> = UNIMPLEMENTED_FLAGS
        .iter()
        .filter(|(_, is_set)| is_set(flags))
        .map(|(name, _)| *name)
        .collect();
    if !authored.is_empty() {
        tracing::warn!(
            flags = authored.join(", "),
            "ignored by the native runtime; these flags are accepted for \
             compatibility with the Python CLI and have no effect on this run"
        );
    }
}

/// Validate multi-run and convergence bounds before execution.
fn validate_multi_run(flags: &ProfileFlags) -> anyhow::Result<()> {
    warn_unimplemented_flags(flags);
    if let Some(n) = flags.num_profile_runs
        && !(1..=10).contains(&n)
    {
        anyhow::bail!(
            "--num-profile-runs must be between 1 and 10 (got {n}); \
             the trials-per-variation ceiling is 10."
        );
    }
    if let Some(c) = flags.confidence_level
        && !(c > 0.0 && c < 1.0)
    {
        anyhow::bail!(
            "--confidence-level must be between 0 and 1 (exclusive) (got {c}); \
             common values are 0.90, 0.95, 0.99."
        );
    }
    for (name, cooldown) in [
        (
            "--profile-run-cooldown-seconds",
            flags.profile_run_cooldown_seconds,
        ),
        (
            "--parameter-sweep-cooldown-seconds",
            flags.parameter_sweep_cooldown_seconds,
        ),
    ] {
        if let Some(s) = cooldown {
            if s < 0.0 {
                anyhow::bail!("{name} must not be negative (got {s}).");
            }
            if s > 86400.0 {
                anyhow::bail!("{name} cooldown exceeds the 24h (86400s) maximum (got {s}).");
            }
        }
    }
    if flags.convergence_metric.is_some() && flags.num_profile_runs.unwrap_or(1) <= 1 {
        anyhow::bail!(
            "--convergence-metric requires --num-profile-runs > 1. \
             Set --num-profile-runs to at least 2 to enable adaptive convergence."
        );
    }
    Ok(())
}

/// Execute one built run through the runner and map its terminal outcome, echoing
/// the runner's console summary to stdout on success.
///
/// The single-run path sends authoring [`load::Inputs`] on the wire; the runtime
/// resolves them at `--execute`.
fn run_single(inputs: load::Inputs) -> anyhow::Result<i32> {
    let artifact_dir = inputs.artifact_dir.clone();
    // Bind logging before execution so startup events reach the run artifact.
    crate::logging::set_log_file(&artifact_dir);
    tracing::info!("Starting native AIPerf run");
    let runner = exec_bin::resolve()?;
    let child_pid = crate::signals::install();
    let terminal = run_benchmark_child_authoring(&inputs, None, &runner, &child_pid)?;
    if terminal.success {
        tracing::info!("Native AIPerf run completed");
        if let Some(path) = &terminal.report_path {
            crate::render::print_console_summary(path);
            tracing::debug!(report = %path, "report written");
        }
        // Echo the `--dry-run` dataset-analysis report when it was emitted.
        crate::render::print_dataset_analysis(&artifact_dir);
        Ok(0)
    } else {
        let detail = terminal
            .error
            .as_deref()
            .unwrap_or("native benchmark failed");
        tracing::error!("Native AIPerf run failed: {detail}");
        Ok(if terminal.returncode == 0 {
            1
        } else {
            terminal.returncode
        })
    }
}

/// Execute a YAML `sweep:` block: expand its variations, resolve+stamp each into
/// a run, then run every cell and write the aggregate (single trial per cell).
/// Execute a non-sweep config `multiRun.numRuns > 1` as repeated trials of one
/// variation, reusing the shared cell runner so cooldown, seeding, warmup
/// suppression, and the aggregate match the `--num-profile-runs` flag path.
fn run_config_trials(
    flags: &ProfileFlags,
    inputs: crate::config_inputs::Inputs,
    trials: u32,
) -> anyhow::Result<i32> {
    let sweep_id = uuid::Uuid::new_v4().simple().to_string();
    let seed = seed_policy(flags);
    let disable_warmup = !flags
        .no_profile_run_disable_warmup_after_first
        .unwrap_or(false)
        && flags.profile_run_disable_warmup_after_first.unwrap_or(true);
    let base_run = load::build(inputs.clone())?;
    let base_dir = base_run.artifact_dir.clone();

    let mut cells = Vec::with_capacity(trials as usize);
    for trial in 0..trials {
        let mut inputs = inputs.clone();
        let mut run = base_run.clone();
        let dir = crate::sweep::artifact_dir::resolve(
            &base_dir,
            false,
            trials,
            "",
            trial,
            IterationOrder::Repeated,
        );
        run.sweep_id = Some(sweep_id.clone());
        run.trial = trial;
        run.random_seed = seed.seed(0, trial);
        run.artifact_dir = dir.clone();
        inputs.artifact_dir = dir;
        inputs.random_seed = seed.seed(0, trial);
        if trial > 0 && disable_warmup {
            if let Some(phases) = run.cfg.phases.as_mut() {
                phases.retain(|p| p.common.name != "warmup");
            }
            inputs.warmup = None;
        }
        cells.push(sweep_run::Cell {
            index: 0,
            trial,
            label: String::new(),
            run,
            inputs: Some(inputs),
        });
    }
    run_cells(flags, &cells, false, IterationOrder::Repeated)
}

fn run_yaml_sweep(
    flags: &ProfileFlags,
    base: serde_json::Value,
    sweep: crate::sweep::yaml_sweep::YamlSweep,
) -> anyhow::Result<i32> {
    let sweep_id = uuid::Uuid::new_v4().simple().to_string();
    let cells = plan_yaml_cells(
        flags.artifact_dir.clone(),
        &base,
        &sweep,
        &sweep_id,
        Some(flags),
    )?;
    run_cells(flags, &cells, true, IterationOrder::Repeated)
}

/// Execute a grid search recipe as stamped sweep cells.
fn run_recipe_sweep(
    flags: &ProfileFlags,
    recipe: crate::search::RecipeSweep,
) -> anyhow::Result<i32> {
    let sweep_id = uuid::Uuid::new_v4().simple().to_string();
    let cells = plan_recipe_cells(flags, &recipe, &sweep_id)?;
    // The sweep base dir is the parent common to every per-cell dir.
    let base = flags.artifact_dir.clone().or_else(|| {
        cells
            .first()
            .and_then(|c| c.run.artifact_dir.parent().map(Path::to_path_buf))
    });
    let code = run_cells(flags, &cells, true, IterationOrder::Repeated)?;

    // Optional post-process (SLA-breach knee / degradation knee / TTFT curve /
    // ITL surface): runs after the sweep aggregate lands so it can read the
    // per-combination metrics back off disk.
    if let (Some(pp), Some(base)) = (recipe.post_process.as_ref(), base.as_ref())
        && let Err(e) = crate::search::run_post_process(base, pp)
    {
        tracing::warn!("failed to write {}: {e}", pp.output_filename());
    }
    Ok(code)
}

/// Drive monotonic SLA saturation with probe and bisection iterations.
fn run_search_loop(flags: &ProfileFlags) -> anyhow::Result<i32> {
    let spec = crate::search::MonotonicSpec::from_flags(flags)?;
    let filters = spec.sla_filters.clone();
    let (lo, hi, max_iterations) = (spec.lo, spec.hi, spec.max_iterations);
    let mut planner = crate::search::MonotonicPlanner::new(spec);
    let mut records: Vec<IterationRecord> = Vec::new();

    // The planner owns concurrency, so the base run uses a neutral scalar.
    let mut base_flags = flags.clone();
    base_flags.concurrency = Some("1".to_string());
    let base = load::resolve(&base_flags)?;
    let base_inputs = load::resolve_inputs(&base_flags)?;
    let base_artifact_dir = base.artifact_dir.clone();
    crate::logging::set_log_file(&base_artifact_dir);

    let sweep_id = uuid::Uuid::new_v4().simple().to_string();
    let seed = seed_policy(flags);
    let runner = exec_bin::resolve()?;
    let child_pid = crate::signals::install();

    eprintln!("aiperf: monotonic SLA search (probe + bisection)");
    let mut any_failure = false;
    while let Some(value) = planner.ask() {
        let iter = planner.iteration();
        let label = format!("search_iter_{iter:04}");

        // Build this probe's run: mutate the built cfg's profiling concurrency.
        let dir = crate::sweep::artifact_dir::resolve(
            &base_artifact_dir,
            true,
            1,
            &label,
            0,
            IterationOrder::Repeated,
        );
        eprintln!(
            "aiperf: [iter {iter}] concurrency={value} -> {}",
            dir.display()
        );
        let terminal = run_search_probe(
            &base_inputs,
            &base,
            &sweep_id,
            iter,
            &label,
            dir,
            value,
            seed.seed(iter as usize, 0),
            &runner,
            &child_pid,
        )?;

        // Read the report once: per-iteration feasibility (every SLA filter
        // satisfied) and the objective value recorded for search_history.json.
        let report = terminal
            .success
            .then(|| terminal.report_path.as_deref())
            .flatten()
            .and_then(|p| std::fs::read(p).ok())
            .and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).ok());
        let feasible = report
            .as_ref()
            .map(|r| {
                filters
                    .iter()
                    .all(|f| f.satisfied_by(report_metric(r, &f.metric_tag, &f.stat)))
            })
            .unwrap_or(false);
        let objective = report
            .as_ref()
            .and_then(|r| report_metric(r, "output_token_throughput", "avg"))
            .filter(|v| v.is_finite());
        if !terminal.success {
            any_failure = true;
            eprintln!(
                "aiperf: [iter {iter}] run failed (treated as infeasible): {}",
                terminal.error.as_deref().unwrap_or("(no detail)")
            );
        } else {
            eprintln!(
                "aiperf: [iter {iter}] concurrency={value} -> {}",
                if feasible { "FEASIBLE" } else { "infeasible" }
            );
        }
        let warned_before = planner.non_monotonic_warning;
        planner.tell(feasible);
        records.push(IterationRecord {
            iteration_idx: iter,
            concurrency: value,
            objective,
            feasible,
            non_monotonic_warning: planner.non_monotonic_warning && !warned_before,
        });
    }

    let reason = planner.convergence_reason().unwrap_or("converged");
    println!(
        "\naiperf: monotonic SLA boundary\n  max feasible concurrency: {}\n  min infeasible concurrency: {}\n  convergence: {reason}{}",
        planner
            .feasible_max
            .map(|v| v.to_string())
            .unwrap_or_else(|| "-".into()),
        planner
            .infeasible_min
            .map(|v| v.to_string())
            .unwrap_or_else(|| "-".into()),
        if planner.non_monotonic_warning {
            "\n  warning: non-monotonic SLA boundary observed"
        } else {
            ""
        },
    );
    write_search_boundary(&base_artifact_dir, &planner, reason)?;
    let config = HistoryConfig {
        planner: "monotonic_sla".into(),
        objective_metric: "output_token_throughput".into(),
        objective_stat: "avg".into(),
        direction: "MAXIMIZE".into(),
        max_iterations,
        n_initial_points: 1,
        sla_filters: filters.clone(),
        lo,
        hi,
        kind: "int".into(),
        swept_dim_path: "phases.profiling.concurrency".into(),
        random_seed: flags.search_random_seed,
    };
    crate::search_history::write_search_history(
        &base_artifact_dir,
        &records,
        &config,
        Some(reason),
        Some("max-concurrency-under-sla"),
    )?;

    Ok(if any_failure { 1 } else { 0 })
}

/// Read one metric stat from a `native-v2.json` report (scalar `value`, or a
/// distribution `avg`/`min`/`max`/`std`/`count` / `percentiles.pNN`). Shares the
/// shape of `sweep::aggregate::headline_value`.
fn report_metric(report: &serde_json::Value, tag: &str, stat: &str) -> Option<f64> {
    let stats = report
        .get("metrics")?
        .get(tag)?
        .get("series")?
        .as_array()?
        .first()?
        .get("stats")?;
    if let Some(v) = stats.get("value").and_then(serde_json::Value::as_f64) {
        return Some(v);
    }
    match stat {
        "avg" | "min" | "max" | "std" | "count" => stats.get(stat)?.as_f64(),
        p => stats.get("percentiles")?.get(p)?.as_f64(),
    }
}

/// Persist the native monotonic-search boundary summary beside the run artifacts.
fn write_search_boundary(
    base_artifact_dir: &Path,
    planner: &crate::search::MonotonicPlanner,
    reason: &str,
) -> anyhow::Result<()> {
    let _ = std::fs::create_dir_all(base_artifact_dir);
    let summary = serde_json::json!({
        "swept_dim_path": "phases.profiling.concurrency",
        "feasible_max": planner.feasible_max,
        "infeasible_min": planner.infeasible_min,
        "non_monotonic_warning": planner.non_monotonic_warning,
        "convergence_reason": reason,
    });
    let path = base_artifact_dir.join("search_boundary.json");
    std::fs::write(&path, serde_json::to_vec_pretty(&summary)?)
        .map_err(|e| anyhow::anyhow!("failed to write {}: {e}", path.display()))?;
    Ok(())
}

/// Drive smooth-isotonic SLA search using a scipy PAVA/PCHIP fit.
#[cfg(feature = "search-pyo3")]
fn run_isotonic_loop(flags: &ProfileFlags) -> anyhow::Result<i32> {
    use std::collections::HashMap;

    let spec = crate::isotonic::IsotonicSpec::from_flags(flags)?;
    let (lo, hi, max_iterations) = (spec.lo, spec.hi, spec.max_iterations);
    let filters_for_history = spec.sla_filters.clone();
    let mut planner = crate::isotonic::SmoothIsotonicPlanner::new(spec);
    let mut records: Vec<IterationRecord> = Vec::new();
    // (filter_key, filter) pairs so the margin map keys match the planner's.
    let filters: Vec<(String, crate::search::SlaFilter)> = planner
        .filters()
        .iter()
        .enumerate()
        .map(|(i, f)| (planner.filter_key(i).to_string(), f.clone()))
        .collect();

    let mut base_flags = flags.clone();
    base_flags.concurrency = Some("1".to_string());
    let base = load::resolve(&base_flags)?;
    let base_inputs = load::resolve_inputs(&base_flags)?;
    let base_artifact_dir = base.artifact_dir.clone();
    crate::logging::set_log_file(&base_artifact_dir);

    let sweep_id = uuid::Uuid::new_v4().simple().to_string();
    let seed = seed_policy(flags);
    let runner = exec_bin::resolve()?;
    let child_pid = crate::signals::install();

    eprintln!("aiperf: smooth-isotonic SLA search (PAVA+PCHIP, scipy)");
    let mut any_failure = false;
    while let Some(value) = planner.ask() {
        let iter = planner.iteration();
        let label = format!("search_iter_{iter:04}");

        let dir = crate::sweep::artifact_dir::resolve(
            &base_artifact_dir,
            true,
            1,
            &label,
            0,
            IterationOrder::Repeated,
        );
        eprintln!(
            "aiperf: [iter {iter}] concurrency={value} -> {}",
            dir.display()
        );
        let terminal = run_search_probe(
            &base_inputs,
            &base,
            &sweep_id,
            iter,
            &label,
            dir,
            value,
            seed.seed(iter as usize, 0),
            &runner,
            &child_pid,
        )?;

        // Per-iteration feasibility + per-filter signed margins from the report.
        let mut feasible = false;
        let mut margins: HashMap<String, f64> = HashMap::new();
        let mut objective: Option<f64> = None;
        if terminal.success
            && let Some(path) = terminal.report_path.as_deref()
            && let Ok(bytes) = std::fs::read(path)
            && let Ok(report) = serde_json::from_slice::<serde_json::Value>(&bytes)
        {
            feasible = filters
                .iter()
                .all(|(_, f)| f.satisfied_by(report_metric(&report, &f.metric_tag, &f.stat)));
            objective =
                report_metric(&report, "output_token_throughput", "avg").filter(|v| v.is_finite());
            for (key, f) in &filters {
                if let Some(obs) = report_metric(&report, &f.metric_tag, &f.stat) {
                    // Signed margin: negative = feasible, increasing in x
                    // (`_signed_margins`): lt/le → obs-thr; gt/ge → thr-obs.
                    let margin = match f.op {
                        crate::search::SlaOp::Lt | crate::search::SlaOp::Le => obs - f.threshold,
                        crate::search::SlaOp::Gt | crate::search::SlaOp::Ge => f.threshold - obs,
                    };
                    margins.insert(key.clone(), margin);
                }
            }
        }
        if !terminal.success {
            any_failure = true;
            eprintln!(
                "aiperf: [iter {iter}] run failed (treated as infeasible): {}",
                terminal.error.as_deref().unwrap_or("(no detail)")
            );
        } else {
            eprintln!(
                "aiperf: [iter {iter}] concurrency={value} -> {}",
                if feasible { "FEASIBLE" } else { "infeasible" }
            );
        }
        let warned_before = planner.non_monotonic_warning;
        planner.tell(feasible, margins)?;
        records.push(IterationRecord {
            iteration_idx: iter,
            concurrency: value,
            objective,
            feasible,
            non_monotonic_warning: planner.non_monotonic_warning && !warned_before,
        });
    }

    let reason = planner
        .convergence_reason()
        .unwrap_or("converged")
        .to_string();
    println!(
        "\naiperf: smooth-isotonic SLA boundary\n  max feasible concurrency: {}\n  min infeasible concurrency: {}\n  boundary type: {}\n  convergence: {reason}",
        planner
            .feasible_max
            .map(|v| v.to_string())
            .unwrap_or_else(|| "-".into()),
        planner
            .infeasible_min
            .map(|v| v.to_string())
            .unwrap_or_else(|| "-".into()),
        planner.boundary_type.unwrap_or("smooth"),
    );
    let _ = std::fs::create_dir_all(&base_artifact_dir);
    let summary = serde_json::json!({
        "swept_dim_path": "phases.profiling.concurrency",
        "feasible_max": planner.feasible_max,
        "infeasible_min": planner.infeasible_min,
        "boundary_type": planner.boundary_type,
        "non_monotonic_warning": planner.non_monotonic_warning,
        "convergence_reason": reason,
    });
    std::fs::write(
        base_artifact_dir.join("search_boundary.json"),
        serde_json::to_vec_pretty(&summary)?,
    )?;
    let config = HistoryConfig {
        planner: "smooth_isotonic".into(),
        objective_metric: "output_token_throughput".into(),
        objective_stat: "avg".into(),
        direction: "MAXIMIZE".into(),
        max_iterations,
        n_initial_points: 1,
        sla_filters: filters_for_history.clone(),
        lo,
        hi,
        kind: "int".into(),
        swept_dim_path: "phases.profiling.concurrency".into(),
        random_seed: flags.search_random_seed,
    };
    crate::search_history::write_search_history(
        &base_artifact_dir,
        &records,
        &config,
        Some(&reason),
        Some("max-concurrency-under-sla"),
    )?;

    Ok(if any_failure { 1 } else { 0 })
}

/// Drive constrained Optuna search for `--search-style bo|optuna`.
#[cfg(feature = "search-pyo3")]
fn run_bayes_loop(flags: &ProfileFlags) -> anyhow::Result<i32> {
    // The recipe's objective is output_token_throughput / avg (maximize).
    const OBJ_METRIC: &str = "output_token_throughput";
    const OBJ_STAT: &str = "avg";

    let spec = crate::bayes::BayesSpec::from_flags(flags)?;
    let filters = spec.sla_filters.clone();
    let (lo, hi, max_iterations, n_initial_points) =
        (spec.lo, spec.hi, spec.max_iterations, spec.n_initial_points);
    let planner_id = match flags.search_style.as_deref() {
        Some("optuna") => "optuna",
        _ => "bayesian",
    };
    let mut planner = crate::bayes::OptunaPlanner::new(spec)?;
    let mut records: Vec<IterationRecord> = Vec::new();

    let mut base_flags = flags.clone();
    base_flags.concurrency = Some("1".to_string());
    let base = load::resolve(&base_flags)?;
    let base_inputs = load::resolve_inputs(&base_flags)?;
    let base_artifact_dir = base.artifact_dir.clone();
    crate::logging::set_log_file(&base_artifact_dir);

    let sweep_id = uuid::Uuid::new_v4().simple().to_string();
    let seed = seed_policy(flags);
    let runner = exec_bin::resolve()?;
    let child_pid = crate::signals::install();

    eprintln!("aiperf: optuna BO SLA search");
    let mut any_failure = false;
    let mut best_feasible: Option<i64> = None;
    while let Some(value) = planner.ask()? {
        let iter = planner.iteration();
        let label = format!("search_iter_{iter:04}");

        let dir = crate::sweep::artifact_dir::resolve(
            &base_artifact_dir,
            true,
            1,
            &label,
            0,
            IterationOrder::Repeated,
        );
        eprintln!(
            "aiperf: [iter {iter}] concurrency={value} -> {}",
            dir.display()
        );
        let terminal = run_search_probe(
            &base_inputs,
            &base,
            &sweep_id,
            iter,
            &label,
            dir,
            value,
            seed.seed(iter as usize, 0),
            &runner,
            &child_pid,
        )?;

        let mut objective: Option<f64> = None;
        let mut sla_observed: Vec<Option<f64>> = vec![None; filters.len()];
        let mut feasible = false;
        if terminal.success
            && let Some(path) = terminal.report_path.as_deref()
            && let Ok(bytes) = std::fs::read(path)
            && let Ok(report) = serde_json::from_slice::<serde_json::Value>(&bytes)
        {
            objective = report_metric(&report, OBJ_METRIC, OBJ_STAT).filter(|v| v.is_finite());
            for (i, f) in filters.iter().enumerate() {
                sla_observed[i] = report_metric(&report, &f.metric_tag, &f.stat);
            }
            feasible = filters
                .iter()
                .all(|f| f.satisfied_by(report_metric(&report, &f.metric_tag, &f.stat)));
        }
        if !terminal.success {
            any_failure = true;
            eprintln!(
                "aiperf: [iter {iter}] run failed (treated as infeasible): {}",
                terminal.error.as_deref().unwrap_or("(no detail)")
            );
        } else {
            if feasible && best_feasible.is_none_or(|b| value > b) {
                best_feasible = Some(value);
            }
            eprintln!(
                "aiperf: [iter {iter}] concurrency={value} throughput={} -> {}",
                objective
                    .map(|v| format!("{v:.1}"))
                    .unwrap_or_else(|| "-".into()),
                if feasible { "FEASIBLE" } else { "infeasible" }
            );
        }
        planner.tell(objective, &sla_observed, feasible)?;
        records.push(IterationRecord {
            iteration_idx: iter,
            concurrency: value,
            objective,
            feasible,
            non_monotonic_warning: false,
        });
    }

    let reason = planner
        .convergence_reason()
        .unwrap_or("converged")
        .to_string();
    println!(
        "\naiperf: optuna BO SLA boundary\n  best feasible concurrency: {}\n  convergence: {reason}",
        best_feasible
            .map(|v| v.to_string())
            .unwrap_or_else(|| "-".into()),
    );
    let _ = std::fs::create_dir_all(&base_artifact_dir);
    let summary = serde_json::json!({
        "swept_dim_path": "phases.profiling.concurrency",
        "best_feasible": best_feasible,
        "convergence_reason": reason,
    });
    std::fs::write(
        base_artifact_dir.join("search_boundary.json"),
        serde_json::to_vec_pretty(&summary)?,
    )?;
    let config = HistoryConfig {
        planner: planner_id.into(),
        objective_metric: OBJ_METRIC.into(),
        objective_stat: OBJ_STAT.into(),
        direction: "MAXIMIZE".into(),
        max_iterations,
        n_initial_points,
        sla_filters: filters.clone(),
        lo,
        hi,
        kind: "int".into(),
        swept_dim_path: "phases.profiling.concurrency".into(),
        random_seed: flags.search_random_seed,
    };
    crate::search_history::write_search_history(
        &base_artifact_dir,
        &records,
        &config,
        Some(&reason),
        Some("max-concurrency-under-sla"),
    )?;

    Ok(if any_failure { 1 } else { 0 })
}

/// Drive `max-goodput-under-slo`: optimize `goodput` over log-uniform concurrency with a
/// `good_request_fraction >= --slo-attainment-fraction` outcome constraint. The
/// three TTFT/TPOT/E2E SLO thresholds are installed as config `slos` (via the
/// `--goodput` projection) so the runtime marks each request good/bad and
/// computes the `goodput` / `good_request_fraction` metrics. Requires
/// `search-pyo3`. Emits `search_history.json`.
#[cfg(feature = "search-pyo3")]
fn run_goodput_loop(flags: &ProfileFlags) -> anyhow::Result<i32> {
    const OBJ_METRIC: &str = "goodput";
    const OBJ_STAT: &str = "avg";

    // The goodput formula needs all three per-request SLOs.
    let ttft = flags.ttft_sla_ms;
    let tpot = flags.tpot_sla_ms.or(flags.itl_sla_ms);
    let e2e = flags.e2e_sla_ms;
    let missing: Vec<&str> = [
        ("--ttft-sla-ms", ttft),
        ("--tpot-sla-ms / --itl-sla-ms", tpot),
        ("--e2e-sla-ms", e2e),
    ]
    .iter()
    .filter(|(_, v)| v.is_none())
    .map(|(flag, _)| *flag)
    .collect();
    anyhow::ensure!(
        missing.is_empty(),
        "recipe 'max-goodput-under-slo' requires {}; all three define what \
         'good' means per request for the goodput formula",
        missing.join(", ")
    );
    let (ttft, tpot, e2e) = (ttft.unwrap(), tpot.unwrap(), e2e.unwrap());

    let spec = crate::bayes::BayesSpec::for_goodput(flags)?;
    let filters = spec.sla_filters.clone();
    let (lo, hi, max_iterations, n_initial_points) =
        (spec.lo, spec.hi, spec.max_iterations, spec.n_initial_points);
    let mut planner = crate::bayes::OptunaPlanner::new(spec)?;
    let mut records: Vec<IterationRecord> = Vec::new();

    // Install the per-request SLOs as config `slos` (keyed by native metric tag)
    // so the runtime computes goodput / good_request_fraction.
    let mut base_flags = flags.clone();
    base_flags.concurrency = Some("1".to_string());
    base_flags.goodput = Some(format!(
        "time_to_first_token:{ttft} inter_token_latency:{tpot} request_latency:{e2e}"
    ));
    let base = load::resolve(&base_flags)?;
    let base_inputs = load::resolve_inputs(&base_flags)?;
    let base_artifact_dir = base.artifact_dir.clone();
    crate::logging::set_log_file(&base_artifact_dir);

    let sweep_id = uuid::Uuid::new_v4().simple().to_string();
    let seed = seed_policy(flags);
    let runner = exec_bin::resolve()?;
    let child_pid = crate::signals::install();

    eprintln!("aiperf: optuna BO goodput search (max-goodput-under-slo)");
    let mut any_failure = false;
    let mut best_feasible: Option<i64> = None;
    while let Some(value) = planner.ask()? {
        let iter = planner.iteration();
        let label = format!("search_iter_{iter:04}");

        let dir = crate::sweep::artifact_dir::resolve(
            &base_artifact_dir,
            true,
            1,
            &label,
            0,
            IterationOrder::Repeated,
        );
        eprintln!(
            "aiperf: [iter {iter}] concurrency={value} -> {}",
            dir.display()
        );
        let terminal = run_search_probe(
            &base_inputs,
            &base,
            &sweep_id,
            iter,
            &label,
            dir,
            value,
            seed.seed(iter as usize, 0),
            &runner,
            &child_pid,
        )?;

        let mut objective: Option<f64> = None;
        let mut sla_observed: Vec<Option<f64>> = vec![None; filters.len()];
        let mut feasible = false;
        if terminal.success
            && let Some(path) = terminal.report_path.as_deref()
            && let Ok(bytes) = std::fs::read(path)
            && let Ok(report) = serde_json::from_slice::<serde_json::Value>(&bytes)
        {
            objective = report_metric(&report, OBJ_METRIC, OBJ_STAT).filter(|v| v.is_finite());
            for (i, f) in filters.iter().enumerate() {
                sla_observed[i] = report_metric(&report, &f.metric_tag, &f.stat);
            }
            feasible = filters
                .iter()
                .all(|f| f.satisfied_by(report_metric(&report, &f.metric_tag, &f.stat)));
        }
        if !terminal.success {
            any_failure = true;
            eprintln!(
                "aiperf: [iter {iter}] run failed (treated as infeasible): {}",
                terminal.error.as_deref().unwrap_or("(no detail)")
            );
        } else {
            if feasible && best_feasible.is_none_or(|b| value > b) {
                best_feasible = Some(value);
            }
            eprintln!(
                "aiperf: [iter {iter}] concurrency={value} goodput={} -> {}",
                objective
                    .map(|v| format!("{v:.1}"))
                    .unwrap_or_else(|| "-".into()),
                if feasible { "FEASIBLE" } else { "infeasible" }
            );
        }
        planner.tell(objective, &sla_observed, feasible)?;
        records.push(IterationRecord {
            iteration_idx: iter,
            concurrency: value,
            objective,
            feasible,
            non_monotonic_warning: false,
        });
    }

    let reason = planner
        .convergence_reason()
        .unwrap_or("converged")
        .to_string();
    println!(
        "\naiperf: optuna BO goodput boundary\n  best feasible concurrency: {}\n  convergence: {reason}",
        best_feasible
            .map(|v| v.to_string())
            .unwrap_or_else(|| "-".into()),
    );
    let _ = std::fs::create_dir_all(&base_artifact_dir);
    let config = HistoryConfig {
        planner: "bayesian".into(),
        objective_metric: OBJ_METRIC.into(),
        objective_stat: OBJ_STAT.into(),
        direction: "MAXIMIZE".into(),
        max_iterations,
        n_initial_points,
        sla_filters: filters.clone(),
        lo,
        hi,
        kind: "int".into(),
        swept_dim_path: "phases.profiling.concurrency".into(),
        random_seed: flags.search_random_seed,
    };
    crate::search_history::write_search_history(
        &base_artifact_dir,
        &records,
        &config,
        Some(&reason),
        Some("max-goodput-under-slo"),
    )?;

    Ok(if any_failure { 1 } else { 0 })
}

/// Build the stamped per-cell runs for a grid `--search-recipe` (testable
/// independently of execution): resolve the base run, mutate the built cfg at
/// each recipe axis per variation, and stamp the sweep envelope + artifact dir.
pub fn plan_recipe_cells(
    flags: &ProfileFlags,
    recipe: &crate::search::RecipeSweep,
    sweep_id: &str,
) -> anyhow::Result<Vec<sweep_run::Cell>> {
    let seed = seed_policy(flags);
    // The recipe owns the swept axes and overrides them per variation, so resolve
    // the base authoring inputs with a single (non-sweep) value for any axis flag it
    // consumes (e.g. pareto-sweep's `--concurrency 1,4` list).
    let mut base_flags = flags.clone();
    base_flags.concurrency = Some("1".to_string());
    let base_inputs = load::resolve_inputs(&base_flags)?;

    let mut cells = Vec::with_capacity(recipe.variations.len());
    for v in &recipe.variations {
        // Override each recipe axis on the authoring inputs (concurrency / isl / osl
        // scalar); the runtime resolves at `--execute`, and a CLI-side run is built
        // purely for per-cell planning metadata + the parity oracle.
        let mut inputs = base_inputs.clone();
        for (kind, value) in &v.overrides {
            crate::search::apply_override_inputs(&mut inputs, *kind, *value);
        }
        let mut run = load::build(inputs.clone())?;
        let dir = crate::sweep::artifact_dir::resolve(
            &run.artifact_dir,
            true,
            1,
            &v.dir_name,
            0,
            IterationOrder::Repeated,
        );
        run.sweep_id = Some(sweep_id.to_string());
        let values: serde_json::Map<String, serde_json::Value> = v
            .values
            .iter()
            .map(|(k, val)| (k.clone(), serde_json::Value::from(*val)))
            .collect();
        run.variation = Some(serde_json::json!({
            "index": v.index,
            "label": v.label,
            "values": values,
        }));
        run.random_seed = seed.seed(v.index, 0);
        run.trial = 0;
        run.artifact_dir = dir.clone();
        // Mirror the per-cell artifact dir + run seed onto the authoring inputs so the
        // runtime's resolution reproduces the CLI-side run byte-for-byte.
        inputs.artifact_dir = dir;
        inputs.random_seed = seed.seed(v.index, 0);
        cells.push(sweep_run::Cell {
            index: v.index,
            trial: 0,
            label: v.label.clone(),
            run,
            inputs: Some(inputs),
        });
    }
    Ok(cells)
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
    overrides: Option<&crate::flags::ProfileFlags>,
) -> anyhow::Result<Vec<sweep_run::Cell>> {
    let variations = sweep.expand(base)?;
    // Base seed: the config's `randomSeed` (or the shared default), then `+index`.
    let seed = sweep_run::SeedPolicy {
        base: Some(
            base.get("randomSeed")
                .or_else(|| base.get("random_seed"))
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(sweep_run::DEFAULT_SWEEP_SEED),
        ),
        same_seed: false,
        vary_per_trial: false,
    };

    let mut cells = Vec::with_capacity(variations.len());
    for v in &variations {
        let expanded = crate::expand::render_with_context(v.config.clone())?;
        // Build the authoring `Inputs` this cell ships on the wire (the runtime
        // resolves at `--execute`), and resolve a CLI-side run purely for per-cell
        // planning metadata (artifact dir base, sweep envelope, and the parity oracle).
        let mut inputs = yaml::resolve_expanded_inputs(expanded, artifact_dir.clone(), overrides)?;
        let mut run = load::build(inputs.clone())?;
        let dir = crate::sweep::artifact_dir::resolve(
            &run.artifact_dir,
            true,
            1,
            &v.dir_name,
            0,
            IterationOrder::Repeated,
        );
        run.sweep_id = Some(sweep_id.to_string());
        let values: serde_json::Map<String, serde_json::Value> = v
            .values
            .iter()
            .map(|(k, val)| (k.clone(), val.clone()))
            .collect();
        run.variation = Some(serde_json::json!({
            "index": v.index,
            "label": v.label,
            "values": values,
        }));
        run.random_seed = seed.seed(v.index, 0);
        run.trial = 0;
        run.artifact_dir = dir.clone();
        // Mirror the per-cell artifact dir + run seed onto the authoring inputs so the
        // runtime's resolution reproduces the CLI-side run byte-for-byte.
        inputs.artifact_dir = dir;
        inputs.random_seed = seed.seed(v.index, 0);
        cells.push(sweep_run::Cell {
            index: v.index,
            trial: 0,
            label: v.label.clone(),
            run,
            inputs: Some(inputs),
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
    let disable_warmup = !flags
        .no_profile_run_disable_warmup_after_first
        .unwrap_or(false)
        && flags.profile_run_disable_warmup_after_first.unwrap_or(true);
    let cells = sweep_run::plan_cells(
        flags,
        expansion,
        trials,
        order,
        &sweep_id,
        seed_policy(flags),
        disable_warmup,
        load::resolve,
    )?;
    run_cells(flags, &cells, expansion.is_sweep, order)
}

/// Resolve per-variation seed policy from sweep seed flags.
pub fn seed_policy(flags: &ProfileFlags) -> sweep_run::SeedPolicy {
    let consistent =
        flags.set_consistent_seed.unwrap_or(true) && !flags.no_set_consistent_seed.unwrap_or(false);
    let base = flags
        .random_seed
        .or_else(|| consistent.then_some(sweep_run::DEFAULT_SWEEP_SEED));
    let same_seed = flags.parameter_sweep_same_seed.unwrap_or(false)
        && !flags.no_parameter_sweep_same_seed.unwrap_or(false);
    let vary_per_trial = flags.vary_seed_per_trial.unwrap_or(false)
        && !flags.no_vary_seed_per_trial.unwrap_or(false);
    sweep_run::SeedPolicy {
        base,
        same_seed,
        vary_per_trial,
    }
}

/// Run every planned cell in turn (with an optional inter-cell cooldown), render
/// the sweep table, and write the aggregate. Shared by the flag-driven sweep,
/// the multi-run path, and the YAML `sweep:` path.
fn run_cells(
    flags: &ProfileFlags,
    cells: &[sweep_run::Cell],
    is_sweep: bool,
    order: IterationOrder,
) -> anyhow::Result<i32> {
    let runner = exec_bin::resolve()?;
    let child_pid = crate::signals::install();
    let cooldown = flags
        .profile_run_cooldown_seconds
        .or(flags.parameter_sweep_cooldown_seconds)
        .filter(|s| *s > 0.0)
        .map(std::time::Duration::from_secs_f64);

    // Use the common artifact directory so all child log output shares one
    // top-level `logs/aiperf.log`.
    if let Some(base) = flags.artifact_dir.clone().or_else(|| {
        cells
            .first()
            .and_then(|c| c.run.artifact_dir.parent().map(Path::to_path_buf))
    }) {
        crate::logging::set_log_file(&base);
    }

    let total = cells.len();
    tracing::info!("{}", "=".repeat(80));
    tracing::info!("Starting Multi-Run Benchmark");
    tracing::info!("  Total runs: {total}");
    if let Some(d) = cooldown {
        tracing::info!("  Cooldown between runs: {}s", d.as_secs_f64());
    }
    tracing::info!("{}", "=".repeat(80));

    let mut outcomes = Vec::new();
    for (n, cell) in cells.iter().enumerate() {
        if let Some(d) = cooldown
            && n > 0
        {
            tracing::info!("Cooldown: {}s", d.as_secs_f64());
            std::thread::sleep(d);
        }
        tracing::info!(
            artifact_dir = %cell.run.artifact_dir.display(),
            "[{}/{}] Executing {}...",
            n + 1,
            total,
            cell.label,
        );
        // Every sweep/multi-run/recipe cell now carries authoring `Inputs`; the
        // runtime resolves them at `--execute`. The CLI never lowers.
        let inputs = cell.inputs.as_ref().ok_or_else(|| {
            anyhow::anyhow!(
                "internal: sweep cell {} missing authoring inputs",
                cell.label
            )
        })?;
        let terminal = run_benchmark_child_authoring(inputs, Some(&cell.run), &runner, &child_pid)?;
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
            tracing::error!(
                "Run failed ({}): {}",
                cell.label,
                terminal.error.as_deref().unwrap_or("(no detail)")
            );
        }
    }

    let successful = outcomes.iter().filter(|o| o.success).count();
    tracing::info!("{}", "=".repeat(80));
    tracing::info!("All runs complete: {successful}/{total} successful");
    tracing::info!("{}", "=".repeat(80));

    tracing::info!("Computing aggregate statistics...");
    let exit_code = sweep::aggregate::finish(flags, &outcomes, is_sweep, order)?;
    if exit_code != 0 {
        let failed = total - successful;
        tracing::warn!("{failed}/{total} sweep cells failed");
    }
    Ok(exit_code)
}

#[cfg(test)]
mod authoring_wire_tests {
    use aiperf_runtime::engine::protocol_v2::{BenchmarkRunWireV2, decode_execute_wire};

    /// The authoring wire (`{"authoring": <Inputs>}`) that every profile path now
    /// ships must resolve — through the runtime's shared
    /// `aiperf_runtime::config::resolve::resolve` — to the same run the CLI would
    /// build from the identical inputs. This pins the no-lower collapse: the CLI
    /// never resolves before the child launch, the runtime does, and both project to
    /// an identical `AuthoredRunSpecV2`.
    #[test]
    fn authoring_wire_matches_cli_resolved() {
        // `resolve` builds the large `BenchmarkConfig` on the stack, which overflows
        // the default test-thread stack; run the body on a generous one.
        std::thread::Builder::new()
            .stack_size(32 * 1024 * 1024)
            .spawn(authoring_wire_matches_cli_resolved_body)
            .expect("spawn worker")
            .join()
            .expect("worker panicked");
    }

    fn authoring_wire_matches_cli_resolved_body() {
        use crate::flags::ProfileFlags;

        let args = [
            "-m",
            "mock-model",
            "--url",
            "http://localhost:8000",
            "--endpoint-type",
            "chat",
            "--concurrency",
            "4",
            "--request-count",
            "8",
            "--isl",
            "128",
            "--osl",
            "16",
            "--streaming",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>();
        let flags = ProfileFlags::parse_from_args(&args).expect("parse flags");

        let inputs = crate::load::resolve_inputs(&flags).expect("normalize inputs");
        // The CLI-side resolved run is the oracle; the authoring wire the runtime
        // decodes originates from the same normalized inputs.
        let resolved = crate::load::build(inputs.clone()).expect("resolve run");
        let resolved_bytes = serde_json::to_vec(&resolved).expect("serialize resolved run");
        let authoring_bytes = serde_json::to_vec(&super::AuthoringWire {
            authoring: &inputs,
            sweep_id: None,
            variation: None,
            trial: 0,
        })
        .expect("serialize authoring wire");

        // Oracle: the CLI-side resolved run projected directly onto the wire shape
        // (the same projection the runner performs after decode).
        let via_resolved = serde_json::from_slice::<BenchmarkRunWireV2>(&resolved_bytes)
            .expect("decode resolved run")
            .into_authored()
            .expect("project resolved run");
        let via_authoring = decode_execute_wire(&authoring_bytes)
            .expect("decode authoring wire")
            .into_authored()
            .expect("project authoring run");

        assert_eq!(
            via_authoring.transport.id, via_resolved.transport.id,
            "transport selection must match the CLI-side resolution"
        );
        assert_eq!(
            via_authoring.workload.id, via_resolved.workload.id,
            "workload selection must match the CLI-side resolution"
        );
        assert_eq!(
            via_authoring.dispatch, via_resolved.dispatch,
            "dispatch default must match the CLI-side resolution"
        );
        assert_eq!(
            via_authoring.artifact_target, via_resolved.artifact_target,
            "artifact target must match the CLI-side resolution"
        );
        // `benchmark_id` is a fresh UUID per `resolve` (random by design), so the two
        // resolutions differ there; every other identity field must match.
        assert_eq!(via_authoring.identity.label, via_resolved.identity.label);
        assert_eq!(via_authoring.identity.trial, via_resolved.identity.trial);
        assert_eq!(
            via_authoring.identity.random_seed,
            via_resolved.identity.random_seed
        );
    }

    /// A search/recipe override applied to the authoring `Inputs` (the new no-lower
    /// path) and then resolved must produce a byte-identical resolved `cfg` to the
    /// old resolved-cfg override (`apply_override` on the built cfg). This pins the
    /// conversion of the adaptive-search loops + recipe sweeps from resolved-cfg
    /// mutation to authoring-scalar override.
    #[test]
    fn search_override_authoring_matches_resolved_cfg() {
        std::thread::Builder::new()
            .stack_size(32 * 1024 * 1024)
            .spawn(search_override_authoring_matches_resolved_cfg_body)
            .expect("spawn worker")
            .join()
            .expect("worker panicked");
    }

    fn search_override_authoring_matches_resolved_cfg_body() {
        use crate::flags::ProfileFlags;
        use crate::search::{AxisKind, apply_override, apply_override_inputs};

        let args = [
            "-m",
            "mock-model",
            "--url",
            "http://localhost:8000",
            "--endpoint-type",
            "chat",
            "--concurrency",
            "1",
            "--request-count",
            "8",
            "--isl",
            "128",
            "--osl",
            "16",
            "--streaming",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>();
        let flags = ProfileFlags::parse_from_args(&args).expect("parse flags");
        let base_inputs = crate::load::resolve_inputs(&flags).expect("normalize inputs");

        // Every axis a search loop or recipe drives: profiling concurrency + isl/osl
        // scalar. Each is applied both ways and the resolved cfg JSON must match.
        for (kind, value) in [
            (AxisKind::PhaseConcurrency, 137i64),
            (AxisKind::IslScalar, 512),
            (AxisKind::OslScalar, 300),
        ] {
            // Old path: resolve the base run, then mutate the resolved cfg JSON.
            let base_run = crate::load::build(base_inputs.clone()).expect("resolve base run");
            let mut old_cfg = serde_json::to_value(&base_run.cfg).expect("serialize base cfg");
            apply_override(&mut old_cfg, kind, value);

            // New path: override the authoring inputs, then resolve.
            let mut inputs = base_inputs.clone();
            apply_override_inputs(&mut inputs, kind, value);
            let new_run = crate::load::build(inputs).expect("resolve overridden run");
            let mut new_cfg = serde_json::to_value(&new_run.cfg).expect("serialize overridden cfg");

            // `benchmark_id` is a fresh UUID per `resolve` (random by design), and the
            // `input_config` echo is a reporting-only mirror of the authored config —
            // the old resolved-cfg patch left it stale (it never patched the echo),
            // while the new authoring path resolves it consistently. Strip both so the
            // comparison pins the *executable* resolved config (phases/datasets/…),
            // which must be byte-identical.
            strip_keys(&mut old_cfg);
            strip_keys(&mut new_cfg);

            assert_eq!(
                new_cfg, old_cfg,
                "authoring override {kind:?}={value} must resolve to the same executable \
                 cfg as the old resolved-cfg override"
            );
        }
    }

    /// Recursively remove `benchmark_id` (a random UUID per resolution) and the
    /// `input_config` reporting echo so two independently-resolved cfgs compare on
    /// their executable structure alone.
    fn strip_keys(value: &mut serde_json::Value) {
        match value {
            serde_json::Value::Object(map) => {
                map.remove("benchmark_id");
                map.remove("input_config");
                for v in map.values_mut() {
                    strip_keys(v);
                }
            }
            serde_json::Value::Array(items) => {
                for v in items.iter_mut() {
                    strip_keys(v);
                }
            }
            _ => {}
        }
    }
}

#[cfg(test)]
mod unimplemented_flag_tests {
    use super::{UNIMPLEMENTED_FLAGS, warn_unimplemented_flags};
    use crate::flags::ProfileFlags;

    /// `ProfileFlags` is large enough that clap's derived parser overflows the
    /// default test-thread stack; see `flags.rs`'s `on_big_stack` for the same
    /// workaround.
    fn on_big_stack(body: impl FnOnce() + Send + 'static) {
        std::thread::Builder::new()
            .stack_size(32 * 1024 * 1024)
            .spawn(body)
            .expect("spawn worker")
            .join()
            .expect("worker panicked");
    }

    fn parse(extra: &[&str]) -> ProfileFlags {
        let mut args: Vec<String> = ["-m", "mock-model", "--url", "http://127.0.0.1:8000"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        args.extend(extra.iter().map(|s| s.to_string()));
        ProfileFlags::parse_from_args(&args).expect("flags parse")
    }

    /// Every table entry must name a flag clap actually accepts, and setting it
    /// must flip that entry's predicate. A renamed flag or a predicate pointing at
    /// the wrong field would otherwise leave the flag silently ignored again.
    #[test]
    fn every_table_entry_detects_its_own_flag() {
        on_big_stack(|| {
        for (name, is_set) in UNIMPLEMENTED_FLAGS {
            let baseline = parse(&[]);
            assert!(!is_set(&baseline), "{name} reads as set with no args");

            // Bool-valued flags accept `=true`; the rest need a value. Try the
            // valued form first and fall back to the bare switch.
            let flags = ProfileFlags::parse_from_args(
                &["-m", "mock-model", "--url", "http://127.0.0.1:8000", name, "1"]
                    .iter()
                    .map(|s| s.to_string())
                    .collect::<Vec<_>>(),
            )
            .or_else(|_| {
                ProfileFlags::parse_from_args(
                    &["-m", "mock-model", "--url", "http://127.0.0.1:8000", name]
                        .iter()
                        .map(|s| s.to_string())
                        .collect::<Vec<_>>(),
                )
            })
            .unwrap_or_else(|e| panic!("{name} is not an accepted flag: {e}"));

            assert!(is_set(&flags), "{name} did not flip its own predicate");
        }
        });
    }

    /// The warn path must stay silent on a clean command line, so a default run
    /// never emits a spurious compatibility warning.
    #[test]
    fn a_clean_command_line_warns_about_nothing() {
        on_big_stack(|| {
        let flags = parse(&[]);
        assert!(
            !UNIMPLEMENTED_FLAGS.iter().any(|(_, is_set)| is_set(&flags)),
            "a minimal command line must trip no entry"
        );
        warn_unimplemented_flags(&flags);
        });
    }
}
