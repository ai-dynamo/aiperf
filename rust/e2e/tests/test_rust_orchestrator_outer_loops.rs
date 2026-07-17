// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Outer-loop checks requiring the Python orchestrator API and request-timing server.

mod common;
#[allow(unused_imports)]
use common::*;

/// Grid sweep: cartesian product of `concurrency` x `isl` runs all 4 coordinates.
#[tokio::test]
#[ignore = "requires grid plan internals"]
async fn test_grid_cartesian_product_runs_all_coordinates_in_rust() {}

/// Zip and scenarios sweeps preserve the authored pairing of parameters.
#[tokio::test]
#[ignore = "requires paired-sweep plan internals"]
async fn test_zip_and_scenarios_preserve_authored_pairing() {}

/// Sobol QMC design coordinates execute in Rust and match `sampling_design.json`.
#[tokio::test]
#[ignore = "requires Sobol sampling-design artifact"]
async fn test_qmc_design_coordinates_execute_in_rust_sobol() {}

/// Latin-hypercube QMC design coordinates execute in Rust and match `sampling_design.json`.
#[tokio::test]
#[ignore = "requires Latin-hypercube sampling-design artifact"]
async fn test_qmc_design_coordinates_execute_in_rust_latin_hypercube() {}

/// Trials with `repeated` iteration order build the canonical `profile_runs/trial_*` tree.
#[tokio::test]
#[ignore = "requires repeated trial artifact tree"]
async fn test_trials_use_both_iteration_orders_and_canonical_artifact_trees_repeated() {}

/// Trials with `independent` iteration order build the canonical `*/profile_runs/trial_*` tree.
#[tokio::test]
#[ignore = "requires independent trial artifact tree"]
async fn test_trials_use_both_iteration_orders_and_canonical_artifact_trees_independent() {}

/// Native samples drive the `cv` convergence mode (min_runs=2, expected_runs=2).
#[tokio::test]
#[ignore = "requires cv convergence internals"]
async fn test_native_samples_drive_all_convergence_modes_cv() {}

/// Native samples drive the `ci_width` convergence mode (min_runs=2, expected_runs=2).
#[tokio::test]
#[ignore = "requires ci-width convergence internals"]
async fn test_native_samples_drive_all_convergence_modes_ci_width() {}

/// Native samples drive the `distribution` convergence mode (min_runs=3, expected_runs=3).
#[tokio::test]
#[ignore = "requires distribution convergence internals"]
async fn test_native_samples_drive_all_convergence_modes_distribution() {}

/// Two-parameter adaptive (optuna/TPE) search changes real Rust load between iterations.
#[tokio::test]
#[ignore = "requires adaptive planner and search-history artifact"]
async fn test_two_parameter_adaptive_search_changes_real_rust_load() {}
