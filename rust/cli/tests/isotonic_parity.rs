// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Smooth-isotonic planner golden-sequence coverage.
//!
//! Requires the `search-pyo3` feature (embeds scipy). Run with:
//!   LD_LIBRARY_PATH=<py libdir> cargo test -p aiperf-cli --features search-pyo3 \
//!     --test isotonic_parity
#![cfg(feature = "search-pyo3")]

use std::collections::HashMap;

use aiperf_cli::isotonic::{IsotonicSpec, SmoothIsotonicPlanner};
use aiperf_cli::search::{SlaFilter, SlaOp};

fn golden() -> serde_json::Value {
    let path = "../../tools/parity/isotonic_golden/isotonic.json";
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    serde_json::from_slice(&bytes).expect("golden json")
}

fn run_native(
    lo: i64,
    hi: i64,
    max_iterations: i64,
    boundary: i64,
    slope: f64,
    threshold: f64,
) -> (Vec<i64>, Option<i64>, Option<i64>, Option<String>) {
    let spec = IsotonicSpec {
        lo,
        hi,
        max_iterations,
        sla_replicates: 0,
        sla_filters: vec![SlaFilter {
            metric_tag: "time_to_first_token".into(),
            stat: "p95".into(),
            op: SlaOp::Lt,
            threshold,
        }],
    };
    let mut planner = SmoothIsotonicPlanner::new(spec);
    let key = planner.filter_key(0).to_string();
    let mut asks = Vec::new();
    while let Some(value) = planner.ask() {
        asks.push(value);
        let observed = threshold + slope * (value - boundary) as f64;
        let feasible = observed < threshold;
        let mut margins = HashMap::new();
        margins.insert(key.clone(), observed - threshold);
        planner.tell(feasible, margins).expect("tell");
    }
    (
        asks,
        planner.feasible_max,
        planner.infeasible_min,
        planner.convergence_reason().map(str::to_owned),
    )
}

#[test]
fn isotonic_probe_sequence_matches_oracle() {
    let golden = golden();
    let cases = golden["cases"].as_array().expect("cases array");
    assert!(!cases.is_empty(), "golden has no cases");

    for (i, case) in cases.iter().enumerate() {
        let lo = case["lo"].as_i64().unwrap();
        let hi = case["hi"].as_i64().unwrap();
        let max_iterations = case["max_iterations"].as_i64().unwrap();
        let boundary = case["boundary"].as_i64().unwrap();
        let slope = case["slope"].as_f64().unwrap();
        let threshold = case["threshold"].as_f64().unwrap();

        let (asks, feasible_max, infeasible_min, reason) =
            run_native(lo, hi, max_iterations, boundary, slope, threshold);

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
