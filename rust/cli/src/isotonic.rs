// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Smooth-isotonic SLA-saturation planner (default `--search-style`).
//!
//! The planner brackets the boundary, fits an isotonic curve, replicates
//! uncertain candidates, and bisects cliffs. [`crate::pyfit`] provides PAVA,
//! PCHIP root solving, cliff detection, and bootstrap confidence intervals.
//! Bootstrap confidence intervals are unseeded and therefore nondeterministic.

use std::collections::HashMap;

use crate::pyfit;
use crate::search::{SlaFilter, SlaOp};

/// Relative boundary-precision target.
const SLA_PRECISION_DEFAULT: f64 = 0.05;
/// Internal probes added before the first fit.
const FIT_INTERNAL_PROBES: i64 = 3;
/// Refit when PAVA produces at most this many distinct values.
const FIT_MIN_DISTINCT: usize = 3;
/// Maximum fit cycles.
const MAX_REFIT_CYCLES: i64 = 3;

/// Return the first key maximizing `margin / max(sigma, 0.01 * |threshold|)`.
fn normalize_margins(
    keys: &[String],
    margins: &HashMap<String, f64>,
    sigmas: Option<&HashMap<String, f64>>,
    thresholds: &HashMap<String, f64>,
) -> String {
    let sigma_floor_frac = 0.01;
    // Fall back to raw argmax when no sigmas are available.
    let use_sigmas = sigmas.is_some_and(|s| s.values().any(|&v| v > 0.0));
    let score = |key: &str| -> f64 {
        let raw = margins[key];
        if !use_sigmas {
            return raw;
        }
        let sigma = sigmas.and_then(|s| s.get(key)).copied().unwrap_or(0.0);
        let threshold = thresholds.get(key).copied().unwrap_or(0.0);
        let sigma_floored = sigma.max(sigma_floor_frac * threshold.abs());
        if sigma_floored == 0.0 {
            raw
        } else {
            raw / sigma_floored
        }
    };
    // Strict comparison preserves the first key on ties.
    let mut best_key = keys
        .iter()
        .find(|k| margins.contains_key(*k))
        .cloned()
        .unwrap();
    let mut best = score(&best_key);
    for key in keys.iter().filter(|k| margins.contains_key(*k)) {
        let s = score(key);
        if s > best {
            best = s;
            best_key = key.clone();
        }
    }
    best_key
}

/// Return the replicate budget, clamped to `[3, 20]`.
fn replicate_count(sigma_margin: f64, threshold: f64, override_n: i64) -> i64 {
    const FLOOR: i64 = 3;
    const CEIL: i64 = 20;
    const GAIN: f64 = 4.0;
    const THRESHOLD_EPS: f64 = 1e-9;
    if override_n > 0 {
        return override_n;
    }
    let denom = threshold.abs().max(THRESHOLD_EPS);
    let ratio_sq = (sigma_margin / denom).powi(2);
    let raw = (GAIN * ratio_sq).ceil() as i64;
    CEIL.min(FLOOR.max(raw))
}

/// Algorithm phase for [`SmoothIsotonicPlanner`].
#[derive(Clone, Copy, PartialEq, Eq)]
enum Phase {
    Bracket,
    Fit,
    Replicate,
    CliffBisect,
}

/// Resolved smooth-isotonic planner configuration.
pub struct IsotonicSpec {
    pub lo: i64,
    pub hi: i64,
    pub max_iterations: i64,
    pub sla_replicates: i64,
    pub sla_filters: Vec<SlaFilter>,
}

impl IsotonicSpec {
    /// Resolve from the CLI flags: `--concurrency-min/max` bounds (default
    /// `[1,1000]`), `--search-max-iterations` (default 30), SLA filters, and
    /// `--sla-replicates`.
    pub fn from_flags(flags: &crate::flags::ProfileFlags) -> anyhow::Result<Self> {
        let lo = flags.concurrency_min.unwrap_or(1);
        let hi = flags.concurrency_max.unwrap_or(1000);
        anyhow::ensure!(lo >= 1, "concurrency lower bound must be >= 1 (got {lo})");
        anyhow::ensure!(
            hi > lo,
            "concurrency upper bound ({hi}) must be > lower ({lo})"
        );
        let sla_filters = crate::search::build_sla_filters(flags);
        anyhow::ensure!(
            !sla_filters.is_empty(),
            "recipe 'max-concurrency-under-sla' requires at least one of \
             --ttft-sla-ms / --tpot-sla-ms / --itl-sla-ms / --e2e-sla-ms / --error-rate-sla"
        );
        Ok(Self {
            lo,
            hi,
            max_iterations: flags.search_max_iterations.unwrap_or(30),
            // Zero selects the adaptive replicate budget.
            sla_replicates: 0,
            sla_filters,
        })
    }
}

/// Smooth-isotonic one-dimensional SLA-saturation planner.
pub struct SmoothIsotonicPlanner {
    hi: i64,
    max_iterations: i64,
    sla_replicates: i64,
    sla_filters: Vec<SlaFilter>,
    filter_keys: Vec<String>,
    thresholds: HashMap<String, f64>,

    /// Highest concurrency with a latched feasible verdict.
    pub feasible_max: Option<i64>,
    /// Lowest concurrency with a latched infeasible verdict.
    pub infeasible_min: Option<i64>,

    /// Per-x list of per-filter signed margins (negative = feasible).
    raw_probes: HashMap<i64, Vec<HashMap<String, f64>>>,
    phase: Phase,
    next_value: i64,
    pending_value: Option<i64>,
    probe_queue: Vec<i64>,
    candidate_x: Option<i64>,
    fit_count: i64,
    iter: i64,
    convergence_reason: Option<String>,
    /// `"smooth"` or `"cliff"` once a candidate boundary is classified.
    pub boundary_type: Option<&'static str>,
    binding_constraint: Option<String>,
    pub non_monotonic_warning: bool,
}

impl SmoothIsotonicPlanner {
    /// Construct from a resolved [`IsotonicSpec`].
    pub fn new(spec: IsotonicSpec) -> Self {
        let filter_keys: Vec<String> = spec
            .sla_filters
            .iter()
            .enumerate()
            .map(|(i, f)| {
                format!(
                    "{i}:{}.{}.{}.{}",
                    f.metric_tag,
                    f.stat,
                    op_str(f.op),
                    f.threshold
                )
            })
            .collect();
        let thresholds: HashMap<String, f64> = filter_keys
            .iter()
            .zip(&spec.sla_filters)
            .map(|(k, f)| (k.clone(), f.threshold))
            .collect();
        Self {
            hi: spec.hi,
            max_iterations: spec.max_iterations,
            sla_replicates: spec.sla_replicates,
            sla_filters: spec.sla_filters,
            filter_keys,
            thresholds,
            feasible_max: None,
            infeasible_min: None,
            raw_probes: HashMap::new(),
            phase: Phase::Bracket,
            next_value: spec.lo,
            pending_value: None,
            probe_queue: Vec::new(),
            candidate_x: None,
            fit_count: 0,
            iter: 0,
            convergence_reason: None,
            boundary_type: None,
            binding_constraint: None,
            non_monotonic_warning: false,
        }
    }

    /// The SLA filters, in registration order (for the run loop to compute margins).
    pub fn filters(&self) -> &[SlaFilter] {
        &self.sla_filters
    }

    /// The filter key for filter index `i` (stable margin-map key).
    pub fn filter_key(&self, i: usize) -> &str {
        &self.filter_keys[i]
    }

    /// Return the next concurrency to probe (drains the internal queue first), or
    /// `None` once converged.
    pub fn ask(&mut self) -> Option<i64> {
        if self.is_converged() {
            return None;
        }
        if !self.probe_queue.is_empty() {
            self.next_value = self.probe_queue.remove(0);
        }
        let value = self.next_value;
        self.pending_value = Some(value);
        Some(value)
    }

    /// Zero-based index of the current probe.
    pub fn iteration(&self) -> i64 {
        self.iter
    }

    /// Absorb the pending probe's feasibility + signed margins and advance the
    /// phase machine. `margins` maps filter-key → signed margin (absent keys are
    /// unmeasured). Panics without a matching [`Self::ask`].
    pub fn tell(&mut self, feasible: bool, margins: HashMap<String, f64>) -> anyhow::Result<()> {
        let value = self
            .pending_value
            .take()
            .expect("tell() without matching ask()");
        self.raw_probes.entry(value).or_default().push(margins);
        self.absorb_verdict(value, feasible);
        match self.phase {
            Phase::Bracket => self.plan_bracket_step(value, feasible),
            Phase::Fit => self.plan_fit_step()?,
            Phase::CliffBisect => self.plan_cliff_bisect_step(),
            Phase::Replicate => self.plan_replicate_step()?,
        }
        self.iter += 1;
        Ok(())
    }

    /// True once a stop reason is latched; may latch `max_iterations`.
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

    fn absorb_verdict(&mut self, value: i64, feasible: bool) {
        if feasible {
            if self.infeasible_min.is_some_and(|m| value >= m) {
                self.non_monotonic_warning = true;
                return;
            }
            if self.feasible_max.is_none_or(|f| value > f) {
                self.feasible_max = Some(value);
            }
        } else {
            if self.feasible_max.is_some_and(|f| value <= f) {
                self.non_monotonic_warning = true;
                return;
            }
            if self.infeasible_min.is_none_or(|m| value < m) {
                self.infeasible_min = Some(value);
            }
        }
    }

    fn plan_bracket_step(&mut self, value: i64, feasible: bool) {
        if !feasible {
            if self.feasible_max.is_none() {
                self.convergence_reason = Some("smooth_isotonic_no_pass_in_range".into());
                return;
            }
            self.enter_fit_phase();
            return;
        }
        let next_value = value * 2;
        if next_value >= self.hi {
            if value >= self.hi {
                self.feasible_max = Some(self.hi);
                self.convergence_reason = Some("smooth_isotonic_no_failure_in_range".into());
                return;
            }
            self.next_value = self.hi;
            return;
        }
        self.next_value = next_value;
    }

    fn enter_fit_phase(&mut self) {
        self.phase = Phase::Fit;
        self.queue_internal_probes(FIT_INTERNAL_PROBES);
    }

    fn queue_internal_probes(&mut self, count: i64) {
        let (Some(fmax), Some(imin)) = (self.feasible_max, self.infeasible_min) else {
            return;
        };
        let gap = imin - fmax;
        if gap <= 1 {
            return;
        }
        for k in 1..=count {
            let frac = k as f64 / (count + 1) as f64;
            let mut x = fmax + (python_round(gap as f64 * frac) as i64).max(1);
            x = x.min(imin - 1);
            x = x.max(fmax + 1);
            if !self.raw_probes.contains_key(&x) {
                self.probe_queue.push(x);
            }
        }
        // Probe order affects the fitted trajectory.
        let mut seen = std::collections::HashSet::new();
        self.probe_queue.retain(|x| seen.insert(*x));
    }

    fn plan_fit_step(&mut self) -> anyhow::Result<()> {
        if !self.probe_queue.is_empty() {
            return Ok(());
        }
        self.fit_count += 1;
        let candidate = match self.fit_and_solve()? {
            Some(c) => c,
            None => match self.bisection_fallback() {
                Some(c) => c,
                None => {
                    self.finalize("smooth_isotonic_pchip_fallback_bisection");
                    return Ok(());
                }
            },
        };
        if self.needs_more_fit_data()? {
            if self.fit_count >= MAX_REFIT_CYCLES {
                self.candidate_x = Some(candidate);
                self.enter_replicate_or_terminate(candidate)?;
                return Ok(());
            }
            self.queue_more_probes_for_refit();
            return Ok(());
        }
        self.candidate_x = Some(candidate);
        self.enter_replicate_or_terminate(candidate)
    }

    /// Per-x averaged margin series for `key` over sorted xs. Returns `None` when
    /// the key is missing at any probed x (curve unavailable).
    fn full_series(&self, xs: &[i64], key: &str) -> Option<Vec<f64>> {
        let mut ys = Vec::new();
        for &x in xs {
            let samples: Vec<f64> = self.raw_probes[&x]
                .iter()
                .filter_map(|m| m.get(key).copied())
                .collect();
            if samples.is_empty() {
                continue;
            }
            ys.push(samples.iter().sum::<f64>() / samples.len() as f64);
        }
        (ys.len() == xs.len()).then_some(ys)
    }

    fn fit_and_solve(&mut self) -> anyhow::Result<Option<i64>> {
        let mut xs: Vec<i64> = self.raw_probes.keys().copied().collect();
        xs.sort_unstable();
        if xs.len() < 2 {
            return Ok(None);
        }
        let last_x = *xs.last().unwrap();

        let mut margins: HashMap<String, f64> = HashMap::new();
        let mut sigmas: HashMap<String, f64> = HashMap::new();
        let mut series: HashMap<String, Vec<f64>> = HashMap::new();
        for key in &self.filter_keys {
            match self.full_series(&xs, key) {
                Some(ys) => {
                    margins.insert(key.clone(), *ys.last().unwrap());
                    let last_samples: Vec<f64> = self.raw_probes[&last_x]
                        .iter()
                        .filter_map(|m| m.get(key).copied())
                        .collect();
                    sigmas.insert(key.clone(), sample_std(&last_samples));
                    series.insert(key.clone(), ys);
                }
                None => {
                    margins.insert(key.clone(), 0.0);
                    sigmas.insert(key.clone(), 0.0);
                }
            }
        }

        let binding_key =
            normalize_margins(&self.filter_keys, &margins, Some(&sigmas), &self.thresholds);
        self.binding_constraint = Some(binding_key.clone());
        let Some(ys) = series.get(&binding_key) else {
            return Ok(None);
        };
        let (Some(fmax), Some(imin)) = (self.feasible_max, self.infeasible_min) else {
            return Ok(None);
        };
        let root = pyfit::fit_root(&xs, ys, fmax as f64, imin as f64)?;
        let Some(root) = root else {
            return Ok(None);
        };
        let mut candidate = python_round(root) as i64;
        if candidate <= fmax {
            candidate = fmax + 1;
        }
        if candidate >= imin {
            candidate = imin - 1;
        }
        Ok(Some(candidate))
    }

    fn needs_more_fit_data(&self) -> anyhow::Result<bool> {
        let Some(binding) = &self.binding_constraint else {
            return Ok(false);
        };
        let mut xs: Vec<i64> = self.raw_probes.keys().copied().collect();
        xs.sort_unstable();
        let mut ys = Vec::new();
        for &x in &xs {
            let samples: Vec<f64> = self.raw_probes[&x]
                .iter()
                .filter_map(|m| m.get(binding).copied())
                .collect();
            if !samples.is_empty() {
                ys.push(samples.iter().sum::<f64>() / samples.len() as f64);
            }
        }
        if ys.len() < 4 {
            return Ok(true);
        }
        Ok(pyfit::isotonic_distinct(&ys)? < FIT_MIN_DISTINCT)
    }

    fn queue_more_probes_for_refit(&mut self) {
        let (Some(fmax), Some(imin)) = (self.feasible_max, self.infeasible_min) else {
            return;
        };
        let gap = imin - fmax;
        if gap <= 1 {
            return;
        }
        for frac in [0.125_f64, 0.625] {
            let mut x = fmax + (python_round(gap as f64 * frac) as i64).max(1);
            x = x.min(imin - 1);
            x = x.max(fmax + 1);
            if !self.raw_probes.contains_key(&x) && !self.probe_queue.contains(&x) {
                self.probe_queue.push(x);
            }
        }
    }

    fn bisection_fallback(&self) -> Option<i64> {
        let (fmax, imin) = (self.feasible_max?, self.infeasible_min?);
        let gap = imin - fmax;
        if gap <= 1 {
            return None;
        }
        Some(fmax + gap / 2)
    }

    fn enter_replicate_or_terminate(&mut self, candidate: i64) -> anyhow::Result<()> {
        let cliff = self.check_cliff()?;
        self.boundary_type = Some(if cliff { "cliff" } else { "smooth" });
        if cliff {
            if self.bracket_precision_reached() {
                self.finalize("smooth_isotonic_cliff_precision_reached");
                return Ok(());
            }
            match self.cliff_bisect_midpoint() {
                Some(mid) => {
                    self.probe_queue.push(mid);
                    self.phase = Phase::CliffBisect;
                }
                None => self.finalize("smooth_isotonic_cliff_precision_reached"),
            }
            return Ok(());
        }
        if self.bracket_precision_reached() {
            self.finalize("smooth_isotonic_precision_reached");
            return Ok(());
        }
        let budget = self.replicate_budget();
        if budget <= 0 {
            if !self.raw_probes.contains_key(&candidate) {
                self.probe_queue.push(candidate);
                self.phase = Phase::Fit;
            } else {
                self.finalize("smooth_isotonic_precision_reached");
            }
            return Ok(());
        }
        for _ in 0..budget {
            self.probe_queue.push(candidate);
        }
        self.phase = Phase::Replicate;
        Ok(())
    }

    fn plan_cliff_bisect_step(&mut self) {
        if !self.probe_queue.is_empty() {
            return;
        }
        if self.bracket_precision_reached() {
            self.finalize("smooth_isotonic_cliff_precision_reached");
            return;
        }
        match self.cliff_bisect_midpoint() {
            Some(mid) => self.probe_queue.push(mid),
            None => self.finalize("smooth_isotonic_cliff_precision_reached"),
        }
    }

    fn plan_replicate_step(&mut self) -> anyhow::Result<()> {
        if !self.probe_queue.is_empty() {
            return Ok(());
        }
        let (Some(candidate), Some(binding)) = (self.candidate_x, self.binding_constraint.clone())
        else {
            self.finalize("smooth_isotonic_precision_reached");
            return Ok(());
        };
        let margins_at_candidate: Vec<f64> = self
            .raw_probes
            .get(&candidate)
            .map(|v| v.iter().filter_map(|m| m.get(&binding).copied()).collect())
            .unwrap_or_default();
        if margins_at_candidate.len() < 2 {
            self.finalize("smooth_isotonic_precision_reached");
            return Ok(());
        }
        let (ci_low, ci_high) = pyfit::boundary_ci(&margins_at_candidate)?;
        if ci_low <= 0.0 && 0.0 <= ci_high {
            self.queue_more_probes_for_refit();
            self.phase = Phase::Fit;
        } else {
            self.finalize("smooth_isotonic_precision_reached");
        }
        Ok(())
    }

    fn cliff_bisect_midpoint(&self) -> Option<i64> {
        let (fmax, imin) = (self.feasible_max?, self.infeasible_min?);
        let gap = imin - fmax;
        if gap <= 1 {
            return None;
        }
        let mut mid = fmax + gap / 2;
        if mid <= fmax {
            mid = fmax + 1;
        }
        if mid >= imin {
            mid = imin - 1;
        }
        if mid <= fmax || mid >= imin {
            return None;
        }
        Some(mid)
    }

    fn check_cliff(&self) -> anyhow::Result<bool> {
        let Some(binding) = &self.binding_constraint else {
            return Ok(false);
        };
        let mut xs: Vec<i64> = self.raw_probes.keys().copied().collect();
        xs.sort_unstable();
        let mut cxs = Vec::new();
        let mut ys = Vec::new();
        for &x in &xs {
            let samples: Vec<f64> = self.raw_probes[&x]
                .iter()
                .filter_map(|m| m.get(binding).copied())
                .collect();
            if !samples.is_empty() {
                cxs.push(x);
                ys.push(samples.iter().sum::<f64>() / samples.len() as f64);
            }
        }
        if ys.len() < 2 {
            return Ok(false);
        }
        pyfit::detect_cliff(
            &cxs,
            &ys,
            self.feasible_max,
            self.infeasible_min,
            self.hi,
            SLA_PRECISION_DEFAULT,
        )
    }

    fn replicate_budget(&self) -> i64 {
        if self.sla_replicates > 0 {
            return 20.min(self.sla_replicates);
        }
        let Some(binding) = &self.binding_constraint else {
            return 0;
        };
        let Some(&last_x) = self.raw_probes.keys().max() else {
            return 0;
        };
        let samples: Vec<f64> = self.raw_probes[&last_x]
            .iter()
            .filter_map(|m| m.get(binding).copied())
            .collect();
        if samples.len() < 2 {
            return 0;
        }
        let sigma = sample_std(&samples);
        let threshold = self.thresholds[binding];
        replicate_count(sigma, threshold, 0)
    }

    fn bracket_precision_reached(&self) -> bool {
        let (Some(fmax), Some(imin)) = (self.feasible_max, self.infeasible_min) else {
            return false;
        };
        let gap = imin - fmax;
        if gap <= 1 {
            return true;
        }
        (gap as f64 / imin.max(1) as f64) < SLA_PRECISION_DEFAULT
    }

    fn finalize(&mut self, reason: &str) {
        if self.convergence_reason.is_none() {
            self.convergence_reason = Some(reason.to_string());
        }
        if self.boundary_type.is_none() {
            self.boundary_type = Some("smooth");
        }
    }
}

fn op_str(op: SlaOp) -> &'static str {
    match op {
        SlaOp::Lt => "lt",
        SlaOp::Le => "le",
        SlaOp::Gt => "gt",
        SlaOp::Ge => "ge",
    }
}

/// Sample standard deviation (Bessel's correction); 0.0 for < 2 samples.
fn sample_std(samples: &[f64]) -> f64 {
    if samples.len() < 2 {
        return 0.0;
    }
    let mean = samples.iter().sum::<f64>() / samples.len() as f64;
    let var = samples.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / (samples.len() - 1) as f64;
    var.sqrt()
}

/// Round half to even.
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
