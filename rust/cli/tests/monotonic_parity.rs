// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Monotonic SLA planner golden-sequence coverage.

use aiperf_cli::search::{MonotonicPlanner, MonotonicSpec, SlaFilter, SlaOp};

fn golden() -> serde_json::Value {
    let path = "../../tools/parity/monotonic_golden/monotonic.json";
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    serde_json::from_slice(&bytes).expect("golden json")
}

fn run_native(
    lo: i64,
    hi: i64,
    max_iterations: i64,
    stability_trials: i64,
    boundary: i64,
    threshold: f64,
) -> (Vec<i64>, Option<i64>, Option<i64>, Option<String>) {
    let spec = MonotonicSpec {
        lo,
        hi,
        max_iterations,
        stability_trials,
        precision: 0.05,
        sla_filters: vec![SlaFilter {
            metric_tag: "time_to_first_token".into(),
            stat: "p95".into(),
            op: SlaOp::Lt,
            threshold,
        }],
    };
    let mut planner = MonotonicPlanner::new(spec);
    let mut asks = Vec::new();
    while let Some(value) = planner.ask() {
        asks.push(value);
        let observed = if value <= boundary {
            threshold - 1.0
        } else {
            threshold + 1.0
        };
        let feasible = planner_filter(threshold).satisfied_by(Some(observed));
        planner.tell(feasible);
    }
    (
        asks,
        planner.feasible_max,
        planner.infeasible_min,
        planner.convergence_reason().map(str::to_owned),
    )
}

fn planner_filter(threshold: f64) -> SlaFilter {
    SlaFilter {
        metric_tag: "time_to_first_token".into(),
        stat: "p95".into(),
        op: SlaOp::Lt,
        threshold,
    }
}

#[test]
fn monotonic_probe_sequence_matches_oracle() {
    let golden = golden();
    let cases = golden["cases"].as_array().expect("cases array");
    assert!(!cases.is_empty(), "golden has no cases");

    for (i, case) in cases.iter().enumerate() {
        let lo = case["lo"].as_i64().unwrap();
        let hi = case["hi"].as_i64().unwrap();
        let max_iterations = case["max_iterations"].as_i64().unwrap();
        let stability_trials = case["stability_trials"].as_i64().unwrap();
        let boundary = case["boundary"].as_i64().unwrap();
        let threshold = case["threshold"].as_f64().unwrap();

        let (asks, feasible_max, infeasible_min, reason) = run_native(
            lo,
            hi,
            max_iterations,
            stability_trials,
            boundary,
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
            "[case {i} boundary={boundary}] ask sequence diverges\n got {asks:?}\nwant {want_asks:?}"
        );
        assert_eq!(
            feasible_max,
            case["feasible_max"].as_i64(),
            "[case {i} boundary={boundary}] feasible_max"
        );
        assert_eq!(
            infeasible_min,
            case["infeasible_min"].as_i64(),
            "[case {i} boundary={boundary}] infeasible_min"
        );
        assert_eq!(
            reason.as_deref(),
            case["convergence_reason"].as_str(),
            "[case {i} boundary={boundary}] convergence_reason"
        );
    }
}
