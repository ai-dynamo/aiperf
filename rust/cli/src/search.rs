// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Native adaptive-search recipes (`--search-recipe`).
//!
//! Ports Python's `aiperf.search_recipes` GRID recipes (`concurrency-ramp`,
//! `prefill-ttft-curve`, `decode-itl-curve`). A grid recipe expands its search
//! space into a STATIC grid sweep at config time — log-spaced value lists over
//! config paths — which is then run like any sweep. Because a recipe sweeps
//! CONFIG paths (a scalar `datasets.main.prompts.isl = N` becomes `{value:N}`,
//! not the `--isl` mean), the recipe path resolves the base run once and mutates
//! the built `cfg` per variation, mirroring Python's raw-config override.
//!
//! The `monotonic` search style runs a dynamic ask-tell loop rather than a
//! static sweep: [`MonotonicPlanner`] is a byte-exact pure-logic port of
//! `aiperf.orchestrator.search_planner.monotonic::MonotonicSLASearchPlanner`
//! (exponential probe + bisection over a 1D SLA-saturation boundary), driven by
//! [`crate::profile::run_search_loop`] which runs one `aiperf` child per
//! probe and feeds back a per-iteration feasibility verdict. The `bayes` /
//! `optuna` / `smooth_isotonic` styles additionally need GP/isotonic fits (the
//! former via in-process pyo3 optuna) and remain future work.

use std::collections::{HashMap, HashSet};

use serde_json::Value;

use crate::flags::ProfileFlags;

pub mod sla_breach;

/// How a recipe axis value maps onto the built `cfg`.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum AxisKind {
    /// Set the `profiling` phase's `concurrency` (integer).
    PhaseConcurrency,
    /// Replace `datasets[0].prompts.isl` with a fixed scalar `{value:N}`.
    IslScalar,
    /// Replace `datasets[0].prompts.osl` with a fixed scalar `{value:N}`.
    OslScalar,
}

/// One recipe sweep axis: its config dotted path (for the label), directory
/// segment, the log-spaced values, and how the value maps onto `cfg`.
pub struct RecipeAxis {
    pub path: &'static str,
    pub seg: &'static str,
    pub values: Vec<i64>,
    pub kind: AxisKind,
}

/// An expanded grid recipe: its ordered per-variation cells.
pub struct RecipeSweep {
    pub variations: Vec<RecipeVariation>,
    /// Optional post-process step to run after the sweep aggregate is written
    /// (e.g. the SLA-breach knee for `max-concurrency-under-sla --search-style grid`).
    pub post_process: Option<SlaBreachSpec>,
}

/// Post-process spec for the `sla_breach_knee` handler: the swept dotted path and
/// the SLA filters to evaluate over each swept-value's per-combination metrics.
pub struct SlaBreachSpec {
    /// Dotted path swept on the axis (e.g. `phases.profiling.concurrency`).
    pub swept_param: String,
    /// SLA filters echoed from the recipe.
    pub filters: Vec<SlaFilter>,
}

/// One expanded recipe variation.
pub struct RecipeVariation {
    pub index: usize,
    pub label: String,
    pub dir_name: String,
    /// `(kind, value)` overrides to apply to the built `cfg`.
    pub overrides: Vec<(AxisKind, i64)>,
    /// `(dotted_path, value)` for the stamped `variation.values`.
    pub values: Vec<(String, i64)>,
}

/// `steps` log-spaced integer values in `[lo, hi]` inclusive, endpoints forced,
/// rounding duplicates collapsed, ascending — byte-exact port of Python
/// `aiperf.search_recipes.builtins::_logspace_int_steps`.
pub fn logspace_int_steps(lo: f64, hi: f64, steps: i64) -> anyhow::Result<Vec<i64>> {
    anyhow::ensure!(steps >= 2, "search steps must be >= 2 (got {steps})");
    anyhow::ensure!(lo > 0.0, "search lower bound must be > 0 (got {lo})");
    anyhow::ensure!(hi > lo, "search upper bound ({hi}) must be > lower ({lo})");
    let (log_lo, log_hi) = (lo.ln(), hi.ln());
    let mut vals: Vec<i64> = (0..steps)
        .map(|i| {
            let v = (log_lo + (log_hi - log_lo) * i as f64 / (steps - 1) as f64).exp();
            (python_round(v) as i64).max(1)
        })
        .collect();
    vals.sort_unstable();
    vals.dedup();
    Ok(vals)
}

/// Python 3 `round()` — round-half-to-even (banker's rounding).
fn python_round(v: f64) -> f64 {
    let floor = v.floor();
    let diff = v - floor;
    if diff < 0.5 {
        floor
    } else if diff > 0.5 {
        floor + 1.0
    } else if (floor as i64) % 2 == 0 {
        floor
    } else {
        floor + 1.0
    }
}

/// Expand a grid `--search-recipe` into its axes. `Ok(None)` when no recipe is
/// set; errors on an unknown / not-yet-supported (bayes/isotonic) recipe.
pub fn expand_recipe(flags: &ProfileFlags) -> anyhow::Result<Option<RecipeSweep>> {
    let Some(recipe) = flags.search_recipe.as_deref() else {
        return Ok(None);
    };
    if recipe == "pareto-sweep" {
        return Ok(Some(RecipeSweep {
            variations: expand_pareto(flags)?,
            post_process: None,
        }));
    }
    let mut post_process: Option<SlaBreachSpec> = None;
    let axes = match recipe {
        "concurrency-ramp" => vec![RecipeAxis {
            path: "phases.profiling.concurrency",
            seg: "concurrency",
            values: logspace_int_steps(
                flags.concurrency_min.unwrap_or(1) as f64,
                flags.concurrency_max.unwrap_or(1000) as f64,
                flags.concurrency_steps.unwrap_or(8),
            )?,
            kind: AxisKind::PhaseConcurrency,
        }],
        "prefill-ttft-curve" => vec![
            RecipeAxis {
                path: "datasets.main.prompts.isl",
                seg: "isl",
                values: logspace_int_steps(
                    flags.isl_min.unwrap_or(256) as f64,
                    flags.isl_max.unwrap_or(32768) as f64,
                    flags.isl_steps.unwrap_or(8),
                )?,
                kind: AxisKind::IslScalar,
            },
            RecipeAxis {
                path: "phases.profiling.concurrency",
                seg: "concurrency",
                values: vec![1],
                kind: AxisKind::PhaseConcurrency,
            },
        ],
        "decode-itl-curve" => vec![
            RecipeAxis {
                path: "phases.profiling.concurrency",
                seg: "concurrency",
                values: logspace_int_steps(
                    flags.concurrency_min.unwrap_or(1) as f64,
                    flags.concurrency_max.unwrap_or(200) as f64,
                    flags.concurrency_steps.unwrap_or(6),
                )?,
                kind: AxisKind::PhaseConcurrency,
            },
            RecipeAxis {
                path: "datasets.main.prompts.osl",
                seg: "osl",
                values: logspace_int_steps(
                    flags.osl_min.unwrap_or(64) as f64,
                    flags.osl_max.unwrap_or(1024) as f64,
                    flags.osl_steps.unwrap_or(4),
                )?,
                kind: AxisKind::OslScalar,
            },
        ],
        "max-concurrency-under-sla" => {
            // Only the static `--search-style grid` variant expands to a sweep
            // here. The dynamic styles run their own ask-tell loop, intercepted
            // in `profile::run` before this expander: `monotonic` is pure Rust;
            // `smooth_isotonic` (default) / `bo` / `optuna` need the scipy/optuna
            // numerical core, available only in the `search-pyo3` build. Reaching
            // this branch means the feature is OFF and the style is non-grid.
            let style = flags.search_style.as_deref().unwrap_or("smooth_isotonic");
            anyhow::ensure!(
                style == "grid",
                "max-concurrency-under-sla --search-style {style:?} needs the \
                 scipy/optuna numerical core, which requires a `search-pyo3` build \
                 of aiperf (embeds Python). This binary was built without it — use \
                 --search-style grid (static sweep) or --search-style monotonic \
                 (pure-Rust probe+bisection), or rebuild with --features search-pyo3"
            );
            // After the sweep aggregate is written, locate the SLA-feasibility
            // boundary along the swept concurrency (`sla_breach.json`).
            post_process = Some(SlaBreachSpec {
                swept_param: "phases.profiling.concurrency".to_string(),
                filters: build_sla_filters(flags),
            });
            vec![RecipeAxis {
                path: "phases.profiling.concurrency",
                seg: "concurrency",
                // Grid style is a fixed 8-step log-spaced concurrency sweep.
                values: logspace_int_steps(
                    flags.concurrency_min.unwrap_or(1) as f64,
                    flags.concurrency_max.unwrap_or(1000) as f64,
                    8,
                )?,
                kind: AxisKind::PhaseConcurrency,
            }]
        }
        other => anyhow::bail!(
            "search recipe {other:?} is not yet supported natively (grid recipes: \
             concurrency-ramp, prefill-ttft-curve, decode-itl-curve, pareto-sweep, \
             max-concurrency-under-sla --search-style grid)"
        ),
    };
    Ok(Some(RecipeSweep {
        variations: expand_axes(&axes),
        post_process,
    }))
}

/// Cartesian-product expansion of recipe axes (sorted by dotted path, last axis
/// fastest — Python `itertools.product` over sorted `sweep_parameters`). Labels
/// are `"path=value, ..."`, dir names `"seg_value__..."`, values keyed by path.
fn expand_axes(axes: &[RecipeAxis]) -> Vec<RecipeVariation> {
    let mut order: Vec<usize> = (0..axes.len()).collect();
    order.sort_by_key(|&i| axes[i].path);

    let mut combos: Vec<Vec<usize>> = vec![vec![]];
    for &ai in &order {
        let mut next = Vec::new();
        for prefix in &combos {
            for vi in 0..axes[ai].values.len() {
                let mut p = prefix.clone();
                p.push(vi);
                next.push(p);
            }
        }
        combos = next;
    }

    combos
        .into_iter()
        .enumerate()
        .map(|(index, combo)| {
            let mut label = Vec::new();
            let mut dir = Vec::new();
            let mut overrides = Vec::new();
            let mut values = Vec::new();
            for (&ai, &vi) in order.iter().zip(combo.iter()) {
                let axis = &axes[ai];
                let v = axis.values[vi];
                label.push(format!("{}={v}", axis.path));
                dir.push(format!("{}_{v}", axis.seg));
                overrides.push((axis.kind, v));
                values.push((axis.path.to_string(), v));
            }
            RecipeVariation {
                index,
                label: label.join(", "),
                dir_name: dir.join("__"),
                overrides,
                values,
            }
        })
        .collect()
}

/// Expand `pareto-sweep`: each `--isl-osl-pairs isl/osl` shape (outer) crossed
/// with each `--concurrency` value (inner). Custom `shape_{isl}_{osl}_c{conc}`
/// labels, `isl_{isl}__osl_{osl}__concurrency_{conc}` dirs, and `{concurrency,
/// isl, osl}` values (Python `_pareto_sweep`).
fn expand_pareto(flags: &ProfileFlags) -> anyhow::Result<Vec<RecipeVariation>> {
    let pairs_raw = flags
        .isl_osl_pairs
        .as_ref()
        .filter(|v| !v.is_empty())
        .ok_or_else(|| anyhow::anyhow!("pareto-sweep requires --isl-osl-pairs"))?;
    // clap may split on whitespace; join then split on comma for `isl/osl` pairs.
    let mut pairs: Vec<(i64, i64)> = Vec::new();
    for token in pairs_raw.join(",").split(',').filter(|s| !s.is_empty()) {
        let (isl, osl) = token
            .split_once('/')
            .ok_or_else(|| anyhow::anyhow!("--isl-osl-pairs {token:?} expected 'isl/osl'"))?;
        pairs.push((
            isl.trim()
                .parse()
                .map_err(|_| anyhow::anyhow!("bad isl in {token:?}"))?,
            osl.trim()
                .parse()
                .map_err(|_| anyhow::anyhow!("bad osl in {token:?}"))?,
        ));
    }
    // Concurrency list (default [1]); comma-list from --concurrency.
    let conc: Vec<i64> = match flags.concurrency.as_deref() {
        Some(c) => c
            .split(',')
            .map(|s| {
                s.trim()
                    .parse()
                    .map_err(|_| anyhow::anyhow!("bad concurrency {s:?}"))
            })
            .collect::<anyhow::Result<_>>()?,
        None => vec![1],
    };

    let mut out = Vec::new();
    let mut index = 0;
    for &(isl, osl) in &pairs {
        for &c in &conc {
            out.push(RecipeVariation {
                index,
                label: format!("shape_{isl}_{osl}_c{c}"),
                dir_name: format!("isl_{isl}__osl_{osl}__concurrency_{c}"),
                overrides: vec![
                    (AxisKind::IslScalar, isl),
                    (AxisKind::OslScalar, osl),
                    (AxisKind::PhaseConcurrency, c),
                ],
                values: vec![
                    ("concurrency".to_string(), c),
                    ("isl".to_string(), isl),
                    ("osl".to_string(), osl),
                ],
            });
            index += 1;
        }
    }
    Ok(out)
}

/// Apply one recipe override onto a built `cfg` value (mirrors Python's raw
/// config override + resolution): concurrency sets the profiling phase's
/// `concurrency`; isl/osl replace the prompts distribution with a fixed scalar.
pub fn apply_override(cfg: &mut Value, kind: AxisKind, value: i64) {
    match kind {
        AxisKind::PhaseConcurrency => {
            if let Some(phases) = cfg.get_mut("phases").and_then(Value::as_array_mut) {
                for p in phases.iter_mut() {
                    if p.get("name").and_then(Value::as_str) == Some("profiling")
                        && let Some(o) = p.as_object_mut()
                    {
                        o.insert("concurrency".into(), Value::from(value));
                    }
                }
            }
        }
        AxisKind::IslScalar | AxisKind::OslScalar => {
            let field = if kind == AxisKind::IslScalar {
                "isl"
            } else {
                "osl"
            };
            if let Some(prompts) = cfg
                .get_mut("datasets")
                .and_then(Value::as_array_mut)
                .and_then(|d| d.first_mut())
                .and_then(|d| d.get_mut("prompts"))
                .and_then(Value::as_object_mut)
            {
                prompts.insert(field.into(), serde_json::json!({ "value": value as f64 }));
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Dynamic monotonic SLA-saturation planner (exponential probe + bisection).
// ---------------------------------------------------------------------------

/// Comparison operator for an [`SlaFilter`]. Mirrors the Python `SLAFilter.op`
/// literal (`"lt" | "le" | "gt" | "ge"`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SlaOp {
    Lt,
    Le,
    Gt,
    Ge,
}

/// One SLA feasibility constraint on a run-summary metric stat. Port of
/// `aiperf.config.sweep.adaptive::SLAFilter`.
#[derive(Clone, Debug)]
pub struct SlaFilter {
    /// Metric tag key into the report's `metrics` map (e.g. `time_to_first_token`).
    pub metric_tag: String,
    /// Stat name (`avg`/`min`/`max`/`std`/`count` or a percentile `pNN`).
    pub stat: String,
    /// Comparison operator.
    pub op: SlaOp,
    /// Right-hand-side threshold the observed value is compared against.
    pub threshold: f64,
}

impl SlaFilter {
    /// True iff `observed` satisfies this filter. A missing / non-finite
    /// observation is **infeasible** (mirrors `_sla_helpers::trial_satisfies`:
    /// the planner has no signal to rank against, so silently passing would
    /// invert the bracket). Strict ops treat `value == threshold` as infeasible.
    pub fn satisfied_by(&self, observed: Option<f64>) -> bool {
        let Some(v) = observed.filter(|v| v.is_finite()) else {
            return false;
        };
        match self.op {
            SlaOp::Lt => v < self.threshold,
            SlaOp::Le => v <= self.threshold,
            SlaOp::Gt => v > self.threshold,
            SlaOp::Ge => v >= self.threshold,
        }
    }

    /// Serialize this filter to its `{metric_tag, stat, op, threshold}` dict
    /// (Python `sweep_sla_filter.sla_filter_to_dict` over an `SLAFilter`).
    pub fn to_dict(&self) -> Value {
        serde_json::json!({
            "metric_tag": self.metric_tag,
            "stat": self.stat,
            "op": op_str(self.op),
            "threshold": self.threshold,
        })
    }
}

/// The literal string for an [`SlaOp`] (`"lt"`/`"le"`/`"gt"`/`"ge"`).
pub fn op_str(op: SlaOp) -> &'static str {
    match op {
        SlaOp::Lt => "lt",
        SlaOp::Le => "le",
        SlaOp::Gt => "gt",
        SlaOp::Ge => "ge",
    }
}

/// Build the SLA filter list for the `max-concurrency-under-sla` recipe from the
/// CLI SLA flags. Byte-exact port of
/// `MaxConcurrencyUnderSLA._build_sla_filters`: TTFT p95, inter-token p95
/// (`--tpot-sla-ms`/`--itl-sla-ms` alias), e2e p99, error-rate p99 — in that
/// order. An empty result means no SLA target was supplied.
pub fn build_sla_filters(flags: &ProfileFlags) -> Vec<SlaFilter> {
    let mut filters = Vec::new();
    if let Some(ttft) = flags.ttft_sla_ms {
        filters.push(SlaFilter {
            metric_tag: "time_to_first_token".into(),
            stat: "p95".into(),
            op: SlaOp::Lt,
            threshold: ttft,
        });
    }
    // `--tpot-sla-ms` and `--itl-sla-ms` are aliases for the same inter-token SLA;
    // Python's `get_inter_token_sla_ms` prefers tpot then itl.
    if let Some(itl) = flags.tpot_sla_ms.or(flags.itl_sla_ms) {
        filters.push(SlaFilter {
            metric_tag: "inter_token_latency".into(),
            stat: "p95".into(),
            op: SlaOp::Lt,
            threshold: itl,
        });
    }
    if let Some(e2e) = flags.e2e_sla_ms {
        filters.push(SlaFilter {
            metric_tag: "request_latency".into(),
            stat: "p99".into(),
            op: SlaOp::Lt,
            threshold: e2e,
        });
    }
    if let Some(err) = flags.error_rate_sla {
        filters.push(SlaFilter {
            metric_tag: "request_error_rate".into(),
            stat: "p99".into(),
            op: SlaOp::Lt,
            threshold: err,
        });
    }
    filters
}

/// Per-swept-value pass/fail tally for the stability window. Port of the Python
/// `_PointLog`: a verdict is provisional until `required` trials agree.
struct PointLog {
    required: i64,
    passes: i64,
    fails: i64,
}

impl PointLog {
    fn new(required: i64) -> Self {
        Self {
            required,
            passes: 0,
            fails: 0,
        }
    }

    fn record(&mut self, feasible: bool) {
        if feasible {
            self.passes += 1;
        } else {
            self.fails += 1;
        }
    }

    /// Latched verdict, or `None` while still provisional.
    fn verdict(&self) -> Option<bool> {
        if self.passes >= self.required {
            Some(true)
        } else if self.fails >= self.required {
            Some(false)
        } else {
            None
        }
    }
}

/// Algorithm phase for [`MonotonicPlanner`].
#[derive(Clone, Copy, PartialEq, Eq)]
enum Phase {
    Probe,
    Bisect,
}

/// Static resolution of the monotonic planner's config (byte-exact with
/// `MaxConcurrencyUnderSLA._build_monotonic_output` + `resolve_concurrency_bounds`).
pub struct MonotonicSpec {
    pub lo: i64,
    pub hi: i64,
    pub max_iterations: i64,
    pub stability_trials: i64,
    pub precision: f64,
    pub sla_filters: Vec<SlaFilter>,
}

impl MonotonicSpec {
    /// Resolve the monotonic planner spec for `max-concurrency-under-sla
    /// --search-style monotonic` from the CLI flags. Defaults: `[1, 1000]`
    /// concurrency, 20 iterations, stability window 2, 5% precision.
    pub fn from_flags(flags: &ProfileFlags) -> anyhow::Result<Self> {
        let lo = flags.concurrency_min.unwrap_or(1);
        let hi = flags.concurrency_max.unwrap_or(1000);
        anyhow::ensure!(lo >= 1, "concurrency lower bound must be >= 1 (got {lo})");
        anyhow::ensure!(
            hi > lo,
            "concurrency upper bound ({hi}) must be > lower ({lo})"
        );
        let sla_filters = build_sla_filters(flags);
        anyhow::ensure!(
            !sla_filters.is_empty(),
            "recipe 'max-concurrency-under-sla' requires at least one of \
             --ttft-sla-ms / --tpot-sla-ms / --itl-sla-ms / --e2e-sla-ms / \
             --error-rate-sla; pass at least one on the CLI alongside --search-recipe"
        );
        Ok(Self {
            lo,
            hi,
            // `--search-max-iterations` is a recipe-tunable override of the
            // recipe's `_MONOTONIC_MAX_ITERATIONS` default (20); see
            // `recipes._RECIPE_TUNABLE_FIELD_TO_SWEEP_FIELD`.
            max_iterations: flags.search_max_iterations.unwrap_or(20),
            // `monotonic_stability_trials` is a config-only field (no CLI flag);
            // its default is 2.
            stability_trials: 2,
            precision: 0.05,
            sla_filters,
        })
    }
}

/// Exponential-probe + bisection 1D SLA-saturation planner. Byte-exact port of
/// `MonotonicSLASearchPlanner`: [`Self::ask`] yields the next concurrency to
/// probe, the caller runs it and computes a feasibility verdict, then
/// [`Self::tell`] absorbs it and plans the next probe. [`Self::is_converged`]
/// latches a terminal reason.
pub struct MonotonicPlanner {
    hi: i64,
    max_iterations: i64,
    stability_trials: i64,
    precision: f64,

    /// Highest concurrency with a latched feasible verdict (Python `feasible_max`).
    pub feasible_max: Option<i64>,
    /// Lowest concurrency with a latched infeasible verdict (Python `infeasible_min`).
    pub infeasible_min: Option<i64>,

    point_logs: HashMap<i64, PointLog>,
    phase: Phase,
    next_value: i64,
    pending_value: Option<i64>,
    iter: i64,
    convergence_reason: Option<String>,

    /// Set when a non-monotonic transition surfaced during the search.
    pub non_monotonic_warning: bool,
    warned_iterations: HashSet<i64>,
}

impl MonotonicPlanner {
    /// Construct a planner from a resolved [`MonotonicSpec`].
    pub fn new(spec: MonotonicSpec) -> Self {
        Self {
            hi: spec.hi,
            max_iterations: spec.max_iterations,
            stability_trials: spec.stability_trials,
            precision: spec.precision,
            feasible_max: None,
            infeasible_min: None,
            point_logs: HashMap::new(),
            phase: Phase::Probe,
            next_value: spec.lo,
            pending_value: None,
            iter: 0,
            convergence_reason: None,
            non_monotonic_warning: false,
            warned_iterations: HashSet::new(),
        }
    }

    /// Return the next concurrency to probe (latched as pending), or `None` once
    /// a convergence reason is set. The zero-based iteration index of this probe
    /// is [`Self::iteration`].
    pub fn ask(&mut self) -> Option<i64> {
        if self.is_converged() {
            return None;
        }
        let value = self.next_value;
        self.pending_value = Some(value);
        Some(value)
    }

    /// Zero-based index of the probe most recently returned by [`Self::ask`].
    pub fn iteration(&self) -> i64 {
        self.iter
    }

    /// Absorb the feasibility verdict for the pending probe and plan the next
    /// step. Panics if called without a matching [`Self::ask`].
    pub fn tell(&mut self, feasible: bool) {
        let value = self
            .pending_value
            .take()
            .expect("tell() without matching ask()");
        let stability = self.stability_trials;
        let log = self
            .point_logs
            .entry(value)
            .or_insert_with(|| PointLog::new(stability));
        log.record(feasible);
        let verdict = log.verdict();

        let mut non_monotonic = false;
        if let Some(v) = verdict {
            non_monotonic = self.absorb_verdict(value, v);
        }
        self.plan_next_step(value, verdict);

        if non_monotonic {
            self.warned_iterations.insert(self.iter);
        }
        self.iter += 1;
    }

    /// True once a boundary reason or `max_iterations` has stopped the search.
    /// May latch `max_iterations` when the budget is exhausted.
    pub fn is_converged(&mut self) -> bool {
        if self.convergence_reason.is_some() {
            return true;
        }
        if self.iter >= self.max_iterations {
            self.convergence_reason = Some("max_iterations".into());
            return true;
        }
        false
    }

    /// The latched convergence reason, if any.
    pub fn convergence_reason(&self) -> Option<&str> {
        self.convergence_reason.as_deref()
    }

    fn absorb_verdict(&mut self, value: i64, verdict: bool) -> bool {
        let mut non_monotonic = false;
        if verdict {
            if self.infeasible_min.is_some_and(|m| value >= m) {
                non_monotonic = true;
                self.non_monotonic_warning = true;
            }
            if self.feasible_max.is_none_or(|f| value > f)
                && self.infeasible_min.is_none_or(|m| value < m)
            {
                self.feasible_max = Some(value);
            }
        } else {
            if self.feasible_max.is_some_and(|f| value <= f) {
                non_monotonic = true;
                self.non_monotonic_warning = true;
            }
            if self.infeasible_min.is_none_or(|m| value < m)
                && self.feasible_max.is_none_or(|f| value > f)
            {
                self.infeasible_min = Some(value);
            }
        }
        non_monotonic
    }

    fn plan_next_step(&mut self, value: i64, verdict: Option<bool>) {
        match verdict {
            None => {
                // Stability window: re-ask the same value until a verdict latches.
                self.next_value = value;
            }
            Some(v) => {
                if self.phase == Phase::Probe {
                    self.plan_probe_step(value, v);
                } else {
                    self.plan_bisect_step();
                }
            }
        }
    }

    fn plan_probe_step(&mut self, value: i64, verdict: bool) {
        if !verdict {
            // First failure during probing — bracket found.
            if self.feasible_max.is_none() {
                self.convergence_reason = Some("monotonic_no_pass_in_range".into());
                return;
            }
            self.phase = Phase::Bisect;
            self.plan_bisect_step();
            return;
        }
        // Passed: try double, capped at hi.
        let next_value = value * 2;
        if next_value >= self.hi {
            if value >= self.hi {
                self.feasible_max = Some(self.hi);
                self.convergence_reason = Some("monotonic_no_failure_in_range".into());
                return;
            }
            self.next_value = self.hi;
            return;
        }
        self.next_value = next_value;
    }

    fn plan_bisect_step(&mut self) {
        let (Some(feasible_max), Some(infeasible_min)) = (self.feasible_max, self.infeasible_min)
        else {
            self.convergence_reason = Some(
                if self.feasible_max.is_none() {
                    "monotonic_no_pass_in_range"
                } else {
                    "monotonic_no_failure_in_range"
                }
                .into(),
            );
            return;
        };
        let gap = infeasible_min - feasible_max;
        if gap <= 1 {
            self.convergence_reason = Some("monotonic_precision_reached".into());
            return;
        }
        let relative = gap as f64 / infeasible_min.max(1) as f64;
        if relative < self.precision {
            self.convergence_reason = Some("monotonic_precision_reached".into());
            return;
        }
        // Integer midpoint biased downward — keeps the bracket tightening.
        let mut midpoint = feasible_max + gap / 2;
        if midpoint <= feasible_max {
            midpoint = feasible_max + 1;
        }
        if midpoint >= infeasible_min {
            midpoint = infeasible_min - 1;
        }
        self.next_value = midpoint;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn logspace_matches_python() {
        assert_eq!(
            logspace_int_steps(1.0, 100.0, 5).unwrap(),
            vec![1, 3, 10, 32, 100]
        );
        assert_eq!(
            logspace_int_steps(1.0, 1000.0, 8).unwrap(),
            vec![1, 3, 7, 19, 52, 139, 373, 1000]
        );
        assert_eq!(
            logspace_int_steps(256.0, 32768.0, 8).unwrap(),
            vec![256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
        );
    }
}
