// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Real Python outer loops driving one fresh Rust process per benchmark run.
//!
//! The Python originals in `tests/integration/test_rust_orchestrator_outer_loops.py`
//! do NOT go through the `aiperf profile` CLI. They import and drive the Python
//! orchestration internals directly:
//!
//!   - `aiperf.config.AIPerfConfig` / `build_benchmark_plan`
//!   - `aiperf.orchestrator.orchestrator.MultiRunOrchestrator`
//!   - `aiperf.orchestrator.rust_executor.RustSubprocessExecutor`
//!   - `aiperf.cli_runner._strategy._build_search_planner`
//!
//! and assert on the in-memory Python result objects they return
//! (`plan.variations`, `result.variation_values`, `result.trial_index`,
//! `result.variation_label`, `result.summary_metrics`, the sampling-design and
//! search-history JSON side artifacts). They also spin up a bespoke Python
//! `ThreadingHTTPServer` chat handler to count and time concurrent requests.
//!
//! None of that surface is reachable through the CLI-only `AIPerfHarness`
//! (which runs `python -m aiperf profile ...` and reads back artifact files as
//! untyped JSON). Faithfully reproducing these tests requires the Python
//! orchestrator API and the custom request-timing HTTP server, so each is
//! ported as an `#[ignore]`d test documenting the required Python-only
//! infrastructure, per the porting rules.

mod common;
#[allow(unused_imports)]
use common::*;

/// Grid sweep: cartesian product of `concurrency` x `isl` runs all 4 coordinates.
#[tokio::test]
#[ignore] // requires: Python orchestrator API (MultiRunOrchestrator / RustSubprocessExecutor / build_benchmark_plan) + request-counting HTTP server
async fn test_grid_cartesian_product_runs_all_coordinates_in_rust() {}

/// Zip and scenarios sweeps preserve the authored pairing of parameters.
#[tokio::test]
#[ignore] // requires: Python orchestrator API (MultiRunOrchestrator / RustSubprocessExecutor / build_benchmark_plan) + request-counting HTTP server
async fn test_zip_and_scenarios_preserve_authored_pairing() {}

/// Sobol QMC design coordinates execute in Rust and match `sampling_design.json`.
#[tokio::test]
#[ignore] // requires: Python orchestrator API (MultiRunOrchestrator / RustSubprocessExecutor) + sampling_design.json side artifact
async fn test_qmc_design_coordinates_execute_in_rust_sobol() {}

/// Latin-hypercube QMC design coordinates execute in Rust and match `sampling_design.json`.
#[tokio::test]
#[ignore] // requires: Python orchestrator API (MultiRunOrchestrator / RustSubprocessExecutor) + sampling_design.json side artifact
async fn test_qmc_design_coordinates_execute_in_rust_latin_hypercube() {}

/// Trials with `repeated` iteration order build the canonical `profile_runs/trial_*` tree.
#[tokio::test]
#[ignore] // requires: Python orchestrator API (MultiRunOrchestrator / RustSubprocessExecutor) + trial artifact-tree layout
async fn test_trials_use_both_iteration_orders_and_canonical_artifact_trees_repeated() {}

/// Trials with `independent` iteration order build the canonical `*/profile_runs/trial_*` tree.
#[tokio::test]
#[ignore] // requires: Python orchestrator API (MultiRunOrchestrator / RustSubprocessExecutor) + trial artifact-tree layout
async fn test_trials_use_both_iteration_orders_and_canonical_artifact_trees_independent() {}

/// Native samples drive the `cv` convergence mode (min_runs=2, expected_runs=2).
#[tokio::test]
#[ignore] // requires: Python orchestrator API (MultiRunOrchestrator / RustSubprocessExecutor) + convergence-mode plan internals
async fn test_native_samples_drive_all_convergence_modes_cv() {}

/// Native samples drive the `ci_width` convergence mode (min_runs=2, expected_runs=2).
#[tokio::test]
#[ignore] // requires: Python orchestrator API (MultiRunOrchestrator / RustSubprocessExecutor) + convergence-mode plan internals
async fn test_native_samples_drive_all_convergence_modes_ci_width() {}

/// Native samples drive the `distribution` convergence mode (min_runs=3, expected_runs=3).
#[tokio::test]
#[ignore] // requires: Python orchestrator API (MultiRunOrchestrator / RustSubprocessExecutor) + convergence-mode plan internals
async fn test_native_samples_drive_all_convergence_modes_distribution() {}

/// Two-parameter adaptive (optuna/TPE) search changes real Rust load between iterations.
#[tokio::test]
#[ignore] // requires: Python orchestrator API (MultiRunOrchestrator / RustSubprocessExecutor / _build_search_planner) + optuna + request-timing HTTP server + search_history.json side artifact
async fn test_two_parameter_adaptive_search_changes_real_rust_load() {}
