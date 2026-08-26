// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
mod common;
use common::*;

use aiperf_mock_server::config::MockServerConfig;
use serde_json::Value;

fn collapse_mock_config() -> MockServerConfig {
    let mut cfg = MockServerConfig::default();
    cfg.no_tokenizer = true;
    cfg.scheduler_enabled = true;
    cfg.scheduler_step_ms = 5.0;
    cfg.scheduler_max_batch_size = 8;
    cfg.scheduler_max_prefill_chunks_per_step = 64;
    cfg.scheduler_goodput_collapse_enabled = true;
    cfg.scheduler_goodput_collapse_threshold = 1.0;
    cfg.scheduler_goodput_collapse_slope = 1.0;
    cfg.scheduler_goodput_collapse_floor = 0.3;
    cfg.ttft = 0.0;
    cfg.itl = 0.0;
    cfg.workers = 1;
    cfg
}

fn read_search_history(r: &RunResult) -> Value {
    let path = r
        .artifacts
        .find_file("**/search_history.json")
        .expect("adaptive recipe did not emit search_history.json; planner may have crashed before convergence");
    let bytes = std::fs::read(&path).expect("read search_history.json");
    serde_json::from_slice(&bytes).expect("parse search_history.json")
}

fn extract_best_concurrency(history: &Value) -> i64 {
    let best_trials = history
        .get("best_trials")
        .and_then(|v| v.as_array())
        .filter(|a| !a.is_empty())
        .unwrap_or_else(|| panic!("search_history.json has no 'best_trials' key: {history:?}"));
    let best = &best_trials[0];
    let variation = best.get("variation_values").cloned().unwrap_or(Value::Null);
    let concurrency = variation
        .get("phases.profiling.concurrency")
        .unwrap_or_else(|| panic!("best.variation_values missing concurrency key: {variation:?}"));
    concurrency
        .as_f64()
        .unwrap_or_else(|| panic!("concurrency not numeric: {concurrency:?}")) as i64
}

#[tokio::test]
async fn test_grid_recipe_locates_sla_breach_knee_single_trial() {
    grid_recipe_locates_sla_breach_knee(1).await;
}

#[tokio::test]
async fn test_grid_recipe_locates_sla_breach_knee_multi_trial() {
    grid_recipe_locates_sla_breach_knee(2).await;
}

async fn grid_recipe_locates_sla_breach_knee(num_profile_runs: u32) {
    let h = AIPerfHarness::new_with(collapse_mock_config()).await;
    let r = h.run_timeout(
        &format!(
            "--model mock-model --url {} --endpoint-type chat --streaming \
             --search-recipe max-concurrency-under-sla --search-style grid \
             --tpot-sla-ms 12 --request-count 16 --warmup-request-count 4 \
             --synthetic-input-tokens-mean 16 --synthetic-input-tokens-stddev 0 \
             --output-tokens-mean 32 --output-tokens-stddev 0 \
             --extra-inputs ignore_eos:true --num-profile-runs {num_profile_runs} --ui none",
            h.mock.url
        ),
        600,
    );
    assert!(r.success(), "aiperf profile failed: {}", r.stderr);

    let breach_path = r
        .artifacts
        .find_file("**/sweep_aggregate/sla_breach.json")
        .expect("recipe did not emit sla_breach.json — post-process handler may not have run");
    let breach: Value =
        serde_json::from_slice(&std::fs::read(&breach_path).expect("read sla_breach.json"))
            .expect("parse sla_breach.json");

    let all_points = breach["all_points"].as_array().expect("all_points array");
    let concurrencies: Vec<i64> = all_points
        .iter()
        .map(|p| p["concurrency"].as_i64().unwrap())
        .collect();
    assert_eq!(
        concurrencies,
        vec![1, 3, 7, 19, 52, 139, 373, 1000],
        "{breach:?}"
    );

    assert_eq!(
        breach["max_passing_concurrency"].as_i64(),
        Some(7),
        "{breach:?}"
    );
    assert_eq!(
        breach["first_failing_concurrency"].as_i64(),
        Some(19),
        "{breach:?}"
    );

    assert_eq!(
        breach["monotonicity_check"].as_bool(),
        Some(true),
        "{breach:?}"
    );

    let feasibility: std::collections::HashMap<i64, bool> = all_points
        .iter()
        .map(|p| {
            (
                p["concurrency"].as_i64().unwrap(),
                p["feasible"].as_bool().unwrap(),
            )
        })
        .collect();
    assert_eq!(feasibility.get(&1), Some(&true));
    assert_eq!(feasibility.get(&3), Some(&true));
    assert_eq!(feasibility.get(&7), Some(&true));
    assert_eq!(feasibility.get(&19), Some(&false));
    assert_eq!(feasibility.get(&52), Some(&false));

    let breach_record = &breach["first_failing_breach"];
    assert_eq!(
        breach_record["metric_tag"].as_str(),
        Some("inter_token_latency")
    );
    assert_eq!(breach_record["op"].as_str(), Some("lt"));
    assert_eq!(breach_record["threshold"].as_f64(), Some(12.0));
    let observed = breach_record["observed"]
        .as_f64()
        .expect("observed must not be null");
    assert!(observed > 12.0, "observed={observed}");
}

#[tokio::test]
async fn test_max_concurrency_under_sla_finds_knee_monotonic() {
    max_concurrency_under_sla_finds_knee("monotonic", (4, 16)).await;
}

/// `--search-style bo` needs the scipy/optuna numerical core, which only exists
/// in a `search-pyo3` build (it embeds Python). `search-pyo3` is deliberately
/// left out of the shipped feature set (`Makefile:215`), so the CLI rejects the
/// run before any search happens. Run with `--features search-pyo3` to exercise it.
#[tokio::test]
#[ignore = "requires a --features search-pyo3 build (embedded Python scipy/optuna); \
            the default e2e binary rejects --search-style bo"]
async fn test_max_concurrency_under_sla_finds_knee_bo() {
    max_concurrency_under_sla_finds_knee("bo", (4, 16)).await;
}

async fn max_concurrency_under_sla_finds_knee(search_style: &str, knee_band: (i64, i64)) {
    let h = AIPerfHarness::new_with(collapse_mock_config()).await;
    let r = h.run_timeout(
        &format!(
            "--model mock-model --url {} --endpoint-type chat --streaming \
             --search-recipe max-concurrency-under-sla --search-style {search_style} \
             --tpot-sla-ms 12 --request-count 32 --warmup-request-count 8 \
             --synthetic-input-tokens-mean 16 --synthetic-input-tokens-stddev 0 \
             --output-tokens-mean 16 --output-tokens-stddev 0 \
             --extra-inputs ignore_eos:true --ui none",
            h.mock.url
        ),
        600,
    );
    assert!(r.success(), "aiperf profile failed: {}", r.stderr);

    let history = read_search_history(&r);
    let best_concurrency = extract_best_concurrency(&history);
    let (lo, hi) = knee_band;
    assert!(
        lo <= best_concurrency && best_concurrency <= hi,
        "recipe selected concurrency={best_concurrency}, expected in [{lo}, {hi}]; convergence_reason={:?}",
        history.get("convergence_reason")
    );

    let any_infeasible = history["iterations"]
        .as_array()
        .expect("iterations array")
        .iter()
        .any(|it| it.get("feasible").and_then(|v| v.as_bool()) == Some(false));
    assert!(
        any_infeasible,
        "no iteration was marked infeasible; SLA filter may not have been applied"
    );
}

/// The `max-goodput-under-slo` recipe is implemented natively (`run_goodput_loop`)
/// but gated behind `search-pyo3`, which the shipped feature set excludes. Without
/// it the recipe falls through to grid expansion and is rejected, naming the grid
/// recipes: `concurrency-ramp`, `prefill-ttft-curve`, `decode-itl-curve`,
/// `pareto-sweep`, and `max-concurrency-under-sla --search-style grid`. The
/// assertions below are the intended contract -- do not weaken them.
#[tokio::test]
#[ignore = "requires a --features search-pyo3 build (embedded Python optuna); \
            the default e2e binary rejects --search-recipe max-goodput-under-slo"]
async fn test_max_goodput_under_slo_finds_knee() {
    let h = AIPerfHarness::new_with(collapse_mock_config()).await;
    let r = h.run_timeout(
        &format!(
            "--model mock-model --url {} --endpoint-type chat --streaming \
             --search-recipe max-goodput-under-slo \
             --ttft-sla-ms 200 --tpot-sla-ms 12 --e2e-sla-ms 2000 \
             --slo-attainment-fraction 0.9 --request-count 32 --warmup-request-count 8 \
             --synthetic-input-tokens-mean 16 --synthetic-input-tokens-stddev 0 \
             --output-tokens-mean 16 --output-tokens-stddev 0 \
             --extra-inputs ignore_eos:true --ui none",
            h.mock.url
        ),
        600,
    );
    assert!(r.success(), "aiperf profile failed: {}", r.stderr);

    let history = read_search_history(&r);
    let best_concurrency = extract_best_concurrency(&history);
    assert!(
        (1..=64).contains(&best_concurrency),
        "goodput recipe selected concurrency={best_concurrency}; expected near knee (~8). \
         convergence_reason={:?}",
        history.get("convergence_reason")
    );
}
