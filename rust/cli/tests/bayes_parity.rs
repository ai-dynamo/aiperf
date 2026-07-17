// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Seeded TPE planner golden-sequence coverage.
//!
//! Requires the `search-pyo3` feature (embeds optuna). Run with:
//!   LD_LIBRARY_PATH=<py libdir> cargo test -p aiperf-cli --features search-pyo3 \
//!     --test bayes_parity
#![cfg(feature = "search-pyo3")]

use aiperf_cli::bayes::{BayesSpec, Direction, OptunaPlanner};
use aiperf_cli::search::{SlaFilter, SlaOp};

fn golden() -> serde_json::Value {
    let path = "../../tools/parity/bayes_golden/bayes.json";
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    serde_json::from_slice(&bytes).expect("golden json")
}

fn run_native(
    seed: u64,
    lo: i64,
    hi: i64,
    max_iterations: i64,
    n_initial: i64,
    ttft_slope: f64,
    threshold: f64,
) -> (Vec<i64>, Option<String>) {
    let spec = BayesSpec {
        lo,
        hi,
        log: true,
        max_iterations,
        n_initial_points: n_initial,
        plateau_window: 8,
        plateau_threshold: 0.01,
        improvement_patience: 10,
        sampler: "tpe".into(),
        seed: Some(seed),
        direction: Direction::Maximize,
        sla_filters: vec![SlaFilter {
            metric_tag: "time_to_first_token".into(),
            stat: "p95".into(),
            op: SlaOp::Lt,
            threshold,
        }],
    };
    let mut planner = OptunaPlanner::new(spec).expect("planner");
    let mut asks = Vec::new();
    while let Some(value) = planner.ask().expect("ask") {
        asks.push(value);
        let throughput = value as f64 * 10.0;
        let ttft = threshold - 50.0 + ttft_slope * value as f64;
        let feasible = ttft < threshold;
        planner
            .tell(Some(throughput), &[Some(ttft)], feasible)
            .expect("tell");
    }
    (asks, planner.convergence_reason().map(str::to_owned))
}

#[test]
fn bayes_probe_sequence_matches_oracle() {
    let golden = golden();
    let cases = golden["cases"].as_array().expect("cases array");
    assert!(!cases.is_empty(), "golden has no cases");

    for (i, case) in cases.iter().enumerate() {
        let seed = case["seed"].as_u64().unwrap();
        let lo = case["lo"].as_i64().unwrap();
        let hi = case["hi"].as_i64().unwrap();
        let max_iterations = case["max_iterations"].as_i64().unwrap();
        let n_initial = case["n_initial_points"].as_i64().unwrap();
        let ttft_slope = case["ttft_slope"].as_f64().unwrap();
        let threshold = case["threshold"].as_f64().unwrap();

        let (asks, reason) = run_native(
            seed,
            lo,
            hi,
            max_iterations,
            n_initial,
            ttft_slope,
            threshold,
        );

        let want_asks: Vec<i64> = case["asks"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_i64().unwrap())
            .collect();
        assert_eq!(
            asks, want_asks,
            "[case {i} seed={seed}] ask sequence diverges\n got {asks:?}\nwant {want_asks:?}"
        );
        assert_eq!(
            reason.as_deref(),
            case["convergence_reason"].as_str(),
            "[case {i} seed={seed}] convergence_reason"
        );
    }
}
