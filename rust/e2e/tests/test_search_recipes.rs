// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! End-to-end smoke tests for each built-in search recipe against the dynamic
//! mock server (scheduler-enabled, batch-size knees, optional goodput collapse).
//!
//! One test per recipe. Each recipe drives `aiperf profile --search-recipe ...`
//! through a real subprocess against a mock server tuned to make the recipe's
//! target signal observable. Assertions stay qualitative -- artifacts exist, the
//! post-process produced a populated payload, planners ran for the expected
//! minimum number of iterations -- because exact knee values are noisy at smoke
//! budgets and the goal here is to prove the recipes wire end-to-end against the
//! dynamic mock features, not to pin specific saturation points.

mod common;
use common::*;

use std::path::Path;

use aiperf_mock_server::config::MockServerConfig;

/// Read a JSON artifact relative to the harness artifact dir.
fn read_artifact_json(dir: &Path, rel: &str) -> serde_json::Value {
    let path = dir.join(rel);
    let bytes = std::fs::read(&path)
        .unwrap_or_else(|e| panic!("failed to read artifact {}: {e}", path.display()));
    serde_json::from_slice(&bytes)
        .unwrap_or_else(|e| panic!("failed to parse artifact {}: {e}", path.display()))
}

/// concurrency-ramp grid + degradation-knee handler land a positive knee.
///
/// Mock has a tight saturation shelf at concurrency=8 (max_batch_size) plus a
/// quadratic TTFT penalty so request latency p99 inflates well past the 20%
/// baseline cutoff somewhere on the recipe's [1, 1000] log grid.
#[tokio::test]
async fn test_concurrency_ramp_detects_degradation_knee() {
    let mut cfg = MockServerConfig::default();
    cfg.scheduler_enabled = true;
    cfg.scheduler_step_ms = 5.0;
    cfg.scheduler_max_batch_size = 8;
    cfg.scheduler_max_prefill_chunks_per_step = 64;
    cfg.ttft_concurrency_quad_ms = 5.0;
    cfg.ttft = 0.0;
    cfg.itl = 0.0;
    cfg.workers = 1;

    let h = AIPerfHarness::new_with(cfg).await;
    let r = h.run_timeout(
        &format!(
            "--model mock-model \
             --url {} \
             --endpoint-type chat \
             --streaming \
             --search-recipe concurrency-ramp \
             --degradation-threshold 0.50 \
             --request-count 30 \
             --warmup-request-count 4 \
             --synthetic-input-tokens-mean 16 \
             --synthetic-input-tokens-stddev 0 \
             --output-tokens-mean 32 \
             --output-tokens-stddev 0 \
             --extra-inputs ignore_eos:true \
             --ui none",
            h.mock.url
        ),
        300,
    );
    assert!(r.success(), "aiperf profile failed: {}", r.stderr);

    let dir = h.artifact_path();
    let knee_path = dir.join("sweep_aggregate").join("degradation_knee.json");
    assert!(
        knee_path.exists(),
        "recipe did not emit {}; post-process handler may not have run",
        knee_path.display()
    );
    let knee = read_artifact_json(dir, "sweep_aggregate/degradation_knee.json");

    // Saturation knee must land somewhere on the swept grid (>= the batch-size
    // shelf at 8). The recipe's default grid covers [1, 1000] log-spaced over 8
    // steps, so any positive int >= 8 is plausible.
    assert!(!knee["knee_concurrency"].is_null(), "{knee}");
    assert!(knee["knee_concurrency"].as_f64().unwrap() >= 8.0, "{knee}");
    assert_eq!(
        knee["baseline_concurrency"].as_f64().unwrap(),
        1.0,
        "{knee}"
    );
    assert_eq!(knee["stat"], "p99");
    assert_eq!(knee["swept_metric"], "request_latency");
    assert!(knee["all_points"].as_array().unwrap().len() >= 2, "{knee}");
}

/// prefill-ttft-curve + ttft_curve_fit emit a populated curve fit.
///
/// Mock TTFT scales linearly with ISL (0.05 ms/token) so a linear fit should
/// explain most of the variance.
#[tokio::test]
async fn test_prefill_ttft_curve_fits_linear_with_isl_penalty() {
    let mut cfg = MockServerConfig::default();
    cfg.scheduler_enabled = true;
    cfg.scheduler_step_ms = 5.0;
    cfg.scheduler_max_batch_size = 8;
    cfg.scheduler_prefill_chunk_tokens = 256;
    cfg.scheduler_max_prefill_chunks_per_step = 64;
    cfg.ttft_per_isl_token_ms = 0.05;
    cfg.ttft = 0.0;
    cfg.itl = 0.0;
    cfg.workers = 1;

    let h = AIPerfHarness::new_with(cfg).await;
    let r = h.run_timeout(
        &format!(
            "--model mock-model \
             --url {} \
             --endpoint-type chat \
             --streaming \
             --search-recipe prefill-ttft-curve \
             --request-count 20 \
             --warmup-request-count 2 \
             --output-tokens-mean 16 \
             --output-tokens-stddev 0 \
             --extra-inputs ignore_eos:true \
             --ui none",
            h.mock.url
        ),
        300,
    );
    assert!(r.success(), "aiperf profile failed: {}", r.stderr);

    let dir = h.artifact_path();
    // Recipe writes prefill_curve.json (output_filename in builtins.py).
    let curve_path = dir.join("sweep_aggregate").join("prefill_curve.json");
    assert!(
        curve_path.exists(),
        "recipe did not emit {}; post-process handler may not have run",
        curve_path.display()
    );
    let curve = read_artifact_json(dir, "sweep_aggregate/prefill_curve.json");

    assert_eq!(curve["swept_metric"], "time_to_first_token");
    assert_eq!(curve["stat"], "avg");
    // The handler always populates raw_points and a fit form; the linear fit's
    // r^2 should be >= 0.5 with a clean linear ISL penalty even when only a
    // subset of the 8 default ISL points completed.
    let fit_form = curve["fit_form"].as_str().unwrap();
    assert!(fit_form == "linear" || fit_form == "quadratic", "{curve}");
    assert!(
        curve["raw_points"].as_array().unwrap().len() >= 2,
        "{curve}"
    );
    if fit_form == "linear" {
        assert!(curve["r_squared"].as_f64().unwrap() >= 0.5, "{curve}");
    } else {
        // Quadratic fallback fired (linear r^2 < floor); accept it as long as
        // the points it fitted are ours.
        assert_eq!(
            curve["coefficients"].as_array().unwrap().len(),
            3,
            "{curve}"
        );
    }
}

/// decode-itl-curve + itl_surface_fit emit a populated 2D surface.
///
/// Mock has both per-OSL-token and concurrency-linear ITL penalties so each
/// surface cell has a distinct ITL. ITL must be at least the 5ms scheduler
/// step floor.
#[tokio::test]
async fn test_decode_itl_curve_emits_2d_surface() {
    let mut cfg = MockServerConfig::default();
    cfg.scheduler_enabled = true;
    cfg.scheduler_step_ms = 5.0;
    cfg.scheduler_max_batch_size = 8;
    cfg.scheduler_max_prefill_chunks_per_step = 64;
    cfg.itl_per_osl_token_ms = 0.01;
    cfg.itl_concurrency_lin_ms = 0.05;
    cfg.ttft = 0.0;
    cfg.itl = 0.0;
    cfg.workers = 1;

    let h = AIPerfHarness::new_with(cfg).await;
    let r = h.run_timeout(
        &format!(
            "--model mock-model \
             --url {} \
             --endpoint-type chat \
             --streaming \
             --search-recipe decode-itl-curve \
             --concurrency-steps 3 \
             --osl-steps 2 \
             --request-count 4 \
             --warmup-request-count 1 \
             --synthetic-input-tokens-mean 16 \
             --synthetic-input-tokens-stddev 0 \
             --extra-inputs ignore_eos:true \
             --ui none",
            h.mock.url
        ),
        300,
    );
    assert!(r.success(), "aiperf profile failed: {}", r.stderr);

    let dir = h.artifact_path();
    let surface_path = dir.join("sweep_aggregate").join("decode_itl_surface.json");
    assert!(
        surface_path.exists(),
        "recipe did not emit {}; post-process handler may not have run",
        surface_path.display()
    );
    let surface = read_artifact_json(dir, "sweep_aggregate/decode_itl_surface.json");

    assert_eq!(surface["swept_metric"], "inter_token_latency");
    assert_eq!(surface["stat"], "avg");
    assert!(surface.get("surface").is_some(), "{surface}");
    let grid = surface["surface"]["itl_grid"].as_array().unwrap();
    // Flatten + drop nulls for unmeasured cells; the recipe's full grid is
    // 6x4 = 24 combinations -- a smoke budget will measure a subset.
    let measured: Vec<f64> = grid
        .iter()
        .flat_map(|row| row.as_array().unwrap().iter())
        .filter(|v| !v.is_null())
        .map(|v| v.as_f64().unwrap())
        .collect();
    assert!(!measured.is_empty(), "{surface}");
    // ITL floor: scheduler_step_ms = 5ms; allow some jitter slack.
    let min_measured = measured.iter().cloned().fold(f64::INFINITY, f64::min);
    assert!(min_measured >= 4.0, "({min_measured}, {surface})");
    assert!(
        surface["raw_points"].as_array().unwrap().len() >= 1,
        "{surface}"
    );
}

/// max-concurrency-under-sla --search-style monotonic finds a finite boundary.
///
/// Mock has a saturation knee at concurrency=8 plus a strong concurrency-linear
/// ITL penalty (0.5 ms per concurrent request) so ITL stretches past the 50ms
/// TPOT SLA somewhere above the knee.
#[tokio::test]
async fn test_max_concurrency_under_sla_finds_boundary() {
    let mut cfg = MockServerConfig::default();
    cfg.scheduler_enabled = true;
    cfg.scheduler_step_ms = 5.0;
    cfg.scheduler_max_batch_size = 8;
    cfg.scheduler_max_prefill_chunks_per_step = 64;
    cfg.itl_concurrency_lin_ms = 0.5;
    cfg.ttft = 0.0;
    cfg.itl = 0.0;
    cfg.workers = 1;

    let h = AIPerfHarness::new_with(cfg).await;
    let r = h.run_timeout(
        &format!(
            "--model mock-model \
             --url {} \
             --endpoint-type chat \
             --streaming \
             --search-recipe max-concurrency-under-sla \
             --search-style monotonic \
             --tpot-sla-ms 50 \
             --request-count 30 \
             --warmup-request-count 4 \
             --synthetic-input-tokens-mean 16 \
             --synthetic-input-tokens-stddev 0 \
             --output-tokens-mean 32 \
             --output-tokens-stddev 0 \
             --extra-inputs ignore_eos:true \
             --ui none",
            h.mock.url
        ),
        600,
    );
    assert!(r.success(), "aiperf profile failed: {}", r.stderr);

    let dir = h.artifact_path();
    let history_path = dir.join("search_history.json");
    assert!(
        history_path.exists(),
        "recipe did not emit search_history.json; planner may not have run"
    );
    let history = read_artifact_json(dir, "search_history.json");

    let iterations = history["iterations"]
        .as_array()
        .cloned()
        .unwrap_or_default();
    assert!(
        iterations.len() >= 3,
        "monotonic planner ran {} iters; expected >= 3",
        iterations.len()
    );
    assert!(!history["convergence_reason"].is_null(), "{history}");

    let boundary = &history["boundary_summary"];
    assert!(!boundary.is_null(), "{history}");
    let feasible_max = &boundary["feasible_max"];
    // Accept either feasible_max set (planner found a feasible point) or the
    // planner ran multiple iterations and produced an infeasible_min, i.e. the
    // boundary was at least bracketed even if no point passed.
    assert!(
        !feasible_max.is_null() || !boundary["infeasible_min"].is_null(),
        "{history}"
    );
    if !feasible_max.is_null() {
        assert!(feasible_max["value"].as_f64().unwrap() >= 1.0, "{history}");
    }
}

/// max-goodput-under-slo BO finds a finite, positive concurrency.
///
/// Mock combines the saturation knee at 8 with goodput collapse and TTFT + ITL
/// penalties past it -- past the knee, request_count completion rate drops below
/// 95% so the BO objective collapses too. We don't pin an exact concurrency (BO
/// at smoke budget is noisy) -- just that the planner ran multiple iterations
/// and reported a finite winning value.
#[tokio::test]
async fn test_max_goodput_under_slo_lands_near_collapse_point() {
    let mut cfg = MockServerConfig::default();
    cfg.scheduler_enabled = true;
    cfg.scheduler_step_ms = 5.0;
    cfg.scheduler_max_batch_size = 8;
    cfg.scheduler_max_prefill_chunks_per_step = 64;
    cfg.scheduler_goodput_collapse_enabled = true;
    cfg.scheduler_goodput_collapse_threshold = 1.0;
    cfg.scheduler_goodput_collapse_slope = 1.0;
    cfg.scheduler_goodput_collapse_floor = 0.3;
    cfg.ttft_concurrency_quad_ms = 2.0;
    cfg.itl_concurrency_lin_ms = 0.3;
    cfg.ttft = 0.0;
    cfg.itl = 0.0;
    cfg.workers = 1;

    let h = AIPerfHarness::new_with(cfg).await;
    let r = h.run_timeout(
        &format!(
            "--model mock-model \
             --url {} \
             --endpoint-type chat \
             --streaming \
             --search-recipe max-goodput-under-slo \
             --ttft-sla-ms 200 \
             --tpot-sla-ms 50 \
             --e2e-sla-ms 5000 \
             --request-count 30 \
             --warmup-request-count 4 \
             --synthetic-input-tokens-mean 16 \
             --synthetic-input-tokens-stddev 0 \
             --output-tokens-mean 32 \
             --output-tokens-stddev 0 \
             --extra-inputs ignore_eos:true \
             --ui none",
            h.mock.url
        ),
        420,
    );
    assert!(r.success(), "aiperf profile failed: {}", r.stderr);

    let dir = h.artifact_path();
    let history_path = dir.join("search_history.json");
    assert!(
        history_path.exists(),
        "recipe did not emit search_history.json; planner may not have run"
    );
    let history = read_artifact_json(dir, "search_history.json");

    let iterations = history["iterations"]
        .as_array()
        .cloned()
        .unwrap_or_default();
    assert!(
        iterations.len() >= 3,
        "BO planner ran {} iters; expected >= 3",
        iterations.len()
    );

    let best_trials = history["best_trials"].as_array();
    assert!(
        best_trials.map(|a| !a.is_empty()).unwrap_or(false),
        "{history}"
    );
    let best = &history["best_trials"][0];
    // Best variation block must include a concurrency value; the recipe's 1D
    // search space is on phases.profiling.concurrency.
    let variation = &best["variation_values"];
    assert!(
        variation.is_object() && !variation.as_object().unwrap().is_empty(),
        "{history}"
    );
    let concurrency = variation.as_object().unwrap().values().next().unwrap();
    assert!(concurrency.is_number(), "{variation}");
    assert!(concurrency.as_f64().unwrap() >= 1.0, "{variation}");
}
