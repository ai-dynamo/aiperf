// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Outer-loop checks requiring the Python orchestrator API and request-timing server.

mod common;
#[allow(unused_imports)]
use common::*;

/// Grid sweep: cartesian product of `concurrency` x `isl` runs all 4 coordinates.
#[tokio::test]
#[ignore = "requires grid plan internals"]
async fn test_grid_cartesian_product_runs_all_coordinates() {}

#[tokio::test]
#[ignore = "requires paired-sweep plan internals"]
async fn test_zip_and_scenarios_preserve_authored_pairing() {}

#[tokio::test]
#[ignore = "requires Sobol sampling-design artifact"]
async fn test_qmc_sobol_coordinates_match_sampling_design() {}

#[tokio::test]
#[ignore = "requires Latin-hypercube sampling-design artifact"]
async fn test_qmc_latin_hypercube_coordinates_match_sampling_design() {}

/// `cv` requires `min_runs=2` and completes after two matching runs.
#[tokio::test]
#[ignore = "requires cv convergence internals"]
async fn test_samples_drive_convergence_mode_cv() {}

/// `ci_width` requires `min_runs=2` and completes after two matching runs.
#[tokio::test]
#[ignore = "requires ci-width convergence internals"]
async fn test_samples_drive_convergence_mode_ci_width() {}

/// `distribution` requires `min_runs=3` and completes after three matching runs.
#[tokio::test]
#[ignore = "requires distribution convergence internals"]
async fn test_samples_drive_convergence_mode_distribution() {}

/// Uses a two-parameter Optuna/TPE search.
#[tokio::test]
#[ignore = "requires adaptive planner and search-history artifact"]
async fn test_two_parameter_adaptive_search_changes_load_between_iterations() {}
