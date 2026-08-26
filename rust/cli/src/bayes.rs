// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Optuna-backed BO SLA search planner (`--search-style bo|optuna`).
//!
//! [`crate::pyopt`] owns the study and sampler. This module owns ask/tell
//! orchestration, failure sentinels, improvement tracking, and convergence.
//! Seeded TPE suggestions are deterministic; torch-based BoTorch acquisition is
//! not guaranteed to be reproducible across processes.

use crate::pyopt::OptunaStudy;
use crate::search::{SlaFilter, SlaOp, op_str, resolve_bounds_and_sla_filters};

/// Minimum mean magnitude for the plateau-CV test.
const PLATEAU_MEAN_EPSILON: f64 = 1e-9;
/// Failure loss used before any finite objective is observed.
const NO_DATA_SENTINEL_LOSS: f64 = 1.0e6;

/// Optimization direction for the single objective.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    Maximize,
    Minimize,
}

/// Resolved BO planner configuration.
pub struct BayesSpec {
    pub lo: i64,
    pub hi: i64,
    pub log: bool,
    pub max_iterations: i64,
    pub n_initial_points: i64,
    pub plateau_window: usize,
    pub plateau_threshold: f64,
    pub improvement_patience: i64,
    pub sampler: String,
    pub seed: Option<u64>,
    pub direction: Direction,
    pub sla_filters: Vec<SlaFilter>,
}

impl BayesSpec {
    /// Resolve `max-concurrency-under-sla --search-style bo|optuna`.
    ///
    /// The default sampler is BoTorch; `--optuna-sampler` overrides it.
    /// An absent `--search-random-seed` leaves sampling nondeterministic.
    pub fn from_flags(flags: &crate::flags::ProfileFlags) -> anyhow::Result<Self> {
        let (lo, hi, sla_filters) = resolve_bounds_and_sla_filters(flags)?;
        let sampler = flags
            .optuna_sampler
            .clone()
            .unwrap_or_else(|| "botorch".to_string());
        Ok(Self {
            lo,
            hi,
            log: true,
            max_iterations: flags.search_max_iterations.unwrap_or(30),
            n_initial_points: flags.search_initial_points.unwrap_or(5),
            plateau_window: 8,
            plateau_threshold: 0.01,
            improvement_patience: 10,
            sampler,
            seed: flags.search_random_seed,
            direction: Direction::Maximize,
            sla_filters,
        })
    }

    /// Resolve `max-goodput-under-slo`.
    ///
    /// Optimizes `goodput` over log-uniform concurrency `[1, 1000]` subject to
    /// `good_request_fraction >= --slo-attainment-fraction` (default `0.95`).
    pub fn for_goodput(flags: &crate::flags::ProfileFlags) -> anyhow::Result<Self> {
        let lo = flags.concurrency_min.unwrap_or(1);
        let hi = flags.concurrency_max.unwrap_or(1000);
        anyhow::ensure!(lo >= 1, "concurrency lower bound must be >= 1 (got {lo})");
        anyhow::ensure!(
            hi > lo,
            "concurrency upper bound ({hi}) must be > lower ({lo})"
        );
        let attainment = flags.slo_attainment_fraction.unwrap_or(0.95);
        anyhow::ensure!(
            attainment > 0.0 && attainment <= 1.0,
            "--slo-attainment-fraction must be in (0, 1] (got {attainment})"
        );
        let sla_filters = vec![SlaFilter {
            metric_tag: "good_request_fraction".into(),
            stat: "avg".into(),
            op: SlaOp::Ge,
            threshold: attainment,
        }];
        let sampler = flags
            .optuna_sampler
            .clone()
            .unwrap_or_else(|| "botorch".to_string());
        Ok(Self {
            lo,
            hi,
            log: true,
            max_iterations: flags.search_max_iterations.unwrap_or(30),
            n_initial_points: flags.search_initial_points.unwrap_or(5),
            plateau_window: 8,
            plateau_threshold: 0.01,
            improvement_patience: 10,
            sampler,
            seed: flags.search_random_seed,
            direction: Direction::Maximize,
            sla_filters,
        })
    }
}

struct HistoryEntry {
    objective_value: Option<f64>,
    #[allow(dead_code)]
    feasible: bool,
}

/// Optuna-backed BO planner. Drives the real optuna study via [`OptunaStudy`];
/// owns the loop, sentinel, improvement tracking, and convergence in Rust.
pub struct OptunaPlanner {
    study: OptunaStudy,
    path: String,
    lo: i64,
    hi: i64,
    log: bool,
    max_iterations: i64,
    plateau_window: usize,
    plateau_threshold: f64,
    improvement_patience: i64,
    direction: Direction,
    sla_filters: Vec<SlaFilter>,
    attr_keys: Vec<String>,

    history: Vec<HistoryEntry>,
    best_loss: Option<f64>,
    iters_since_improvement: i64,
    iter: i64,
    convergence_reason: Option<String>,
}

impl OptunaPlanner {
    /// Construct from a resolved [`BayesSpec`]; creates the optuna study, seeded
    /// only when the spec carries a seed.
    pub fn new(spec: BayesSpec) -> anyhow::Result<Self> {
        // BoTorch requires optional Python packages; TPE remains available.
        let sampler = match spec.sampler.as_str() {
            "botorch" | "bo" if !optuna_has_botorch() => {
                eprintln!("aiperf: optuna botorch sampler unavailable; falling back to tpe");
                "tpe"
            }
            other => other,
        };
        let attr_keys: Vec<String> = spec
            .sla_filters
            .iter()
            .map(|f| {
                format!(
                    "sla:{}:{}:{}:{}",
                    f.metric_tag,
                    f.stat,
                    op_str(f.op),
                    f.threshold
                )
            })
            .collect();
        let specs: Vec<(String, String, f64)> = spec
            .sla_filters
            .iter()
            .zip(&attr_keys)
            .map(|(f, k)| (k.clone(), op_str(f.op).to_string(), f.threshold))
            .collect();
        let direction_str = match spec.direction {
            Direction::Maximize => "maximize",
            Direction::Minimize => "minimize",
        };
        let study = OptunaStudy::new(
            direction_str,
            sampler,
            spec.n_initial_points,
            spec.seed,
            &specs,
        )?;
        Ok(Self {
            study,
            path: "phases.profiling.concurrency".to_string(),
            lo: spec.lo,
            hi: spec.hi,
            log: spec.log,
            max_iterations: spec.max_iterations,
            plateau_window: spec.plateau_window,
            plateau_threshold: spec.plateau_threshold,
            improvement_patience: spec.improvement_patience,
            direction: spec.direction,
            sla_filters: spec.sla_filters,
            attr_keys,
            history: Vec::new(),
            best_loss: None,
            iters_since_improvement: 0,
            iter: 0,
            convergence_reason: None,
        })
    }

    /// The SLA filters (for the run loop to compute observations).
    pub fn filters(&self) -> &[SlaFilter] {
        &self.sla_filters
    }

    /// Ask optuna for the next concurrency to probe, or `None` once converged.
    pub fn ask(&mut self) -> anyhow::Result<Option<i64>> {
        if self.is_converged() {
            return Ok(None);
        }
        let value = self.study.ask_int(&self.path, self.lo, self.hi, self.log)?;
        Ok(Some(value))
    }

    /// Zero-based index of the current probe.
    pub fn iteration(&self) -> i64 {
        self.iter
    }

    /// Absorb this probe's outcome. `objective` is the averaged objective-metric
    /// value (None = unmeasurable), `sla_observed` the per-filter averaged
    /// observation (None = unmeasurable), `feasible` the per-trial SLA verdict.
    pub fn tell(
        &mut self,
        objective: Option<f64>,
        sla_observed: &[Option<f64>],
        feasible: bool,
    ) -> anyhow::Result<()> {
        let has_unmeasurable = sla_observed.iter().any(|o| o.is_none());
        let attrs: Vec<(String, Option<f64>)> = self
            .attr_keys
            .iter()
            .cloned()
            .zip(sla_observed.iter().copied())
            .collect();

        let objective_for_history = match objective {
            Some(v) if v.is_finite() => {
                self.study.tell(v, &attrs)?;
                Some(v)
            }
            _ => {
                let sentinel = self.failure_sentinel();
                self.study.tell(sentinel, &attrs)?;
                None
            }
        };
        self.track_scalar_improvement(objective_for_history);
        let iteration_feasible = feasible && !has_unmeasurable;
        self.history.push(HistoryEntry {
            objective_value: objective_for_history,
            feasible: iteration_feasible,
        });
        self.iter += 1;
        Ok(())
    }

    /// Latch convergence on maximum iterations, improvement patience, or
    /// plateau CV.
    pub fn is_converged(&mut self) -> bool {
        if self.convergence_reason.is_some() {
            return true;
        }
        if self.iter >= self.max_iterations {
            self.convergence_reason = Some("max_iterations".into());
            return true;
        }
        if self.iters_since_improvement >= self.improvement_patience {
            self.convergence_reason = Some("improvement_patience".into());
            return true;
        }
        if self.history.len() >= self.plateau_window {
            let recent: Vec<f64> = self.history[self.history.len() - self.plateau_window..]
                .iter()
                .filter_map(|h| h.objective_value)
                .collect();
            if recent.len() == self.plateau_window {
                let n = recent.len() as f64;
                let mean = recent.iter().sum::<f64>() / n;
                if mean.abs() >= PLATEAU_MEAN_EPSILON {
                    let var = recent.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0);
                    let cv = var.sqrt() / mean.abs();
                    if cv < self.plateau_threshold {
                        self.convergence_reason = Some("plateau_cv".into());
                        return true;
                    }
                }
            }
        }
        false
    }

    /// The latched convergence reason, if any.
    pub fn convergence_reason(&self) -> Option<&str> {
        self.convergence_reason.as_deref()
    }

    fn objective_to_loss(&self, objective: f64) -> f64 {
        match self.direction {
            Direction::Maximize => -objective,
            Direction::Minimize => objective,
        }
    }

    fn track_scalar_improvement(&mut self, objective_for_history: Option<f64>) {
        let Some(obj) = objective_for_history else {
            self.iters_since_improvement += 1;
            return;
        };
        let iter_loss = self.objective_to_loss(obj);
        if self.best_loss.is_none_or(|b| iter_loss < b) {
            self.best_loss = Some(iter_loss);
            self.iters_since_improvement = 0;
        } else {
            self.iters_since_improvement += 1;
        }
    }

    fn failure_sentinel(&self) -> f64 {
        let prior: Vec<f64> = self
            .history
            .iter()
            .filter_map(|h| h.objective_value)
            .filter(|v| v.is_finite())
            .collect();
        if prior.is_empty() {
            return match self.direction {
                Direction::Maximize => -NO_DATA_SENTINEL_LOSS,
                Direction::Minimize => NO_DATA_SENTINEL_LOSS,
            };
        }
        match self.direction {
            Direction::Maximize => {
                let worst = prior.iter().cloned().fold(f64::INFINITY, f64::min);
                let margin = (worst.abs() * 0.1).max(1.0);
                worst - margin
            }
            Direction::Minimize => {
                let worst = prior.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let margin = (worst.abs() * 0.1).max(1.0);
                worst + margin
            }
        }
    }
}

/// True if optuna's botorch integration is importable in this interpreter.
fn optuna_has_botorch() -> bool {
    use pyo3::prelude::*;
    Python::with_gil(|py| py.import("optuna_integration").is_ok() && py.import("botorch").is_ok())
}
