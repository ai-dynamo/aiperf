// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Byte-exact parity of Rust `detect_agent_chains` against Python
//! `weka_agent_chains.detect_agent_chains`.
//!
//! Golden produced by `tools/agentx_chains_golden.py`; this replays each
//! scenario and diffs the full partition (main index, worker indices, seams,
//! per-chain request lists, fork metadata, spliced_into).

#![cfg(feature = "agentx")]

use aiperf_runtime::agentx::chains::{detect_agent_chains, ChainReq};
use serde_json::Value;
use std::path::PathBuf;

fn golden_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("tests/fixtures/agentx/chains_golden.json")
}

fn ints(v: &Value) -> Vec<i64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_i64().unwrap())
        .collect()
}

fn opt_usize(v: &Value) -> Option<usize> {
    if v.is_null() {
        None
    } else {
        Some(v.as_i64().unwrap() as usize)
    }
}

fn opt_i64(v: &Value) -> Option<i64> {
    if v.is_null() {
        None
    } else {
        Some(v.as_i64().unwrap())
    }
}

#[test]
fn detect_agent_chains_matches_python_golden() {
    let raw = std::fs::read(golden_path()).expect("read chains_golden.json");
    let scenarios: Value = serde_json::from_slice(&raw).unwrap();

    for sc in scenarios.as_array().unwrap() {
        let name = sc["name"].as_str().unwrap();
        let normals: Vec<(i64, ChainReq)> = sc["normals"]
            .as_array()
            .unwrap()
            .iter()
            .map(|n| {
                let api_time = if n["api_time"].is_null() {
                    None
                } else {
                    Some(n["api_time"].as_f64().unwrap())
                };
                (
                    n["outer"].as_i64().unwrap(),
                    ChainReq {
                        t: n["t"].as_f64().unwrap(),
                        api_time,
                        model: n["model"].as_str().unwrap().to_string(),
                        hash_ids: ints(&n["hash_ids"]),
                        input_length: n["in"].as_i64().unwrap(),
                        output_length: n["out"].as_i64().unwrap(),
                    },
                )
            })
            .collect();

        let r = detect_agent_chains(normals, 3600.0, 0.5);
        let want = &sc["result"];

        assert_eq!(
            r.main_index,
            want["main_index"].as_i64().unwrap() as usize,
            "{name}: main_index"
        );
        let want_workers: Vec<usize> = ints(&want["worker_indices"])
            .into_iter()
            .map(|x| x as usize)
            .collect();
        assert_eq!(r.worker_indices, want_workers, "{name}: worker_indices");
        assert_eq!(
            r.seams_merged,
            want["seams_merged"].as_i64().unwrap(),
            "{name}: seams_merged"
        );
        assert_eq!(
            r.unclassified_empty_hash,
            want["unclassified_empty_hash"].as_i64().unwrap(),
            "{name}: unclassified"
        );

        let want_chains = want["chains"].as_array().unwrap();
        assert_eq!(r.chains.len(), want_chains.len(), "{name}: chain count");
        for (i, (c, wc)) in r.chains.iter().zip(want_chains).enumerate() {
            let got_reqs: Vec<i64> = c.requests.iter().map(|(oi, _)| *oi).collect();
            assert_eq!(got_reqs, ints(&wc["requests"]), "{name} chain {i}: requests");
            assert_eq!(
                c.spliced_into,
                opt_usize(&wc["spliced_into"]),
                "{name} chain {i}: spliced_into"
            );
            match (&c.fork, wc["fork"].is_null()) {
                (None, true) => {}
                (Some(fk), false) => {
                    let wf = &wc["fork"];
                    assert_eq!(
                        fk.parent_chain,
                        opt_usize(&wf["parent_chain"]),
                        "{name} chain {i}: fork.parent_chain"
                    );
                    assert_eq!(
                        fk.fork_outer_idx,
                        opt_i64(&wf["fork_outer_idx"]),
                        "{name} chain {i}: fork.fork_outer_idx"
                    );
                    assert_eq!(fk.depth, wf["depth"].as_i64().unwrap(), "{name} chain {i}: fork.depth");
                }
                _ => panic!("{name} chain {i}: fork presence mismatch"),
            }
        }
    }
}
