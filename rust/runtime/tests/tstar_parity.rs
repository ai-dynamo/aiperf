// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Cross-language golden-vector parity for `t*` sampling.
//!
//! Python numpy is authoritative: `tools/gen_tstar_parity_vectors.py` computes
//! `t_star_us` with the exact agentx logic from
//! `src/aiperf/timing/graph_ir_source.py:113-150` (`_sample_t_star` +
//! `_seed_for_trace_lane`) and commits the grid as
//! `tests/data/tstar_parity_vectors.json`, storing each `t_star_us` as its f64
//! bit pattern. This test replays every row through
//! `aiperf_runtime::graph::tstar::WindowTStarSampler` and asserts bit-exact equality; a
//! mismatch is a parity bug in the Rust `NumpyPcg64` (A1) or sampler (A2), never
//! in the Python-authored JSON.

use aiperf_runtime::graph::tstar::{TStarSampler, WindowTStarSampler};

#[test]
fn tstar_matches_python_numpy_golden_vectors() {
    let raw = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/tstar_parity_vectors.json"
    ))
    .unwrap();
    let rows: Vec<serde_json::Value> = serde_json::from_str(&raw).unwrap();
    assert!(!rows.is_empty());
    for r in rows {
        let s = WindowTStarSampler {
            start_min_ratio: r["min"].as_f64().unwrap(),
            start_max_ratio: r["max"].as_f64().unwrap(),
            random_seed: r["base_seed"].as_u64().unwrap(),
        };
        let got = s.sample_t_star(
            r["trace_id"].as_str().unwrap(),
            r["lane"].as_u64().unwrap(),
            r["duration_us"].as_f64().unwrap(),
        );
        let want_bits = r["t_star_us_bits"].as_u64().unwrap();
        assert_eq!(got.to_bits(), want_bits, "row {r:?}");
    }
}
