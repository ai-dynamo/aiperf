// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Byte-exact parity of Rust `expand_subagent_to_child_plans` against the real
//! Python `weka_trace._expand_subagent_to_child_plans`.

use aiperf_runtime::agentx::config::WekaConfig;
use aiperf_runtime::agentx::subagent::expand_subagent_to_child_plans;
use aiperf_runtime::agentx::trace::{WekaInnerRequest, WekaNormalRequest, WekaSubagentEntry};
use serde_json::Value;
use std::path::PathBuf;

fn golden_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("tests/fixtures/agentx/subagent_golden.json")
}

fn ints(v: &Value) -> Vec<i64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_i64().unwrap())
        .collect()
}

#[test]
fn subagent_expansion_matches_python_golden() {
    let raw = std::fs::read(golden_path()).expect("read subagent_golden.json");
    let scenarios: Value = serde_json::from_slice(&raw).unwrap();

    for sc in scenarios.as_array().unwrap() {
        let name = sc["name"].as_str().unwrap();
        let requests: Vec<WekaInnerRequest> = sc["requests"]
            .as_array()
            .unwrap()
            .iter()
            .map(|r| {
                WekaInnerRequest::Normal(WekaNormalRequest {
                    t: r["t"].as_f64().unwrap(),
                    model: r["model"].as_str().unwrap().to_string(),
                    input_length: r["in"].as_i64().unwrap(),
                    output_length: r["out"].as_i64().unwrap(),
                    hash_ids: ints(&r["hash_ids"]),
                    input_types: vec![],
                    output_types: vec![],
                    stop: String::new(),
                    api_time: if r["api_time"].is_null() {
                        None
                    } else {
                        Some(r["api_time"].as_f64().unwrap())
                    },
                    think_time: None,
                })
            })
            .collect();

        let entry = WekaSubagentEntry {
            t: sc["entry_t"].as_f64().unwrap(),
            agent_id: sc["agent_id"].as_str().unwrap().to_string(),
            subagent_type: "Explore".to_string(),
            duration_ms: Some(1000),
            total_tokens: None,
            tool_use_count: None,
            status: "completed".to_string(),
            requests,
            models: vec!["m".to_string()],
            tool_tokens: sc["tool_tokens"].as_i64().unwrap(),
            system_tokens: sc["system_tokens"].as_i64().unwrap(),
        };

        let plans = expand_subagent_to_child_plans(
            sc["trace_id"].as_str().unwrap(),
            sc["sa_index"].as_i64().unwrap() as usize,
            sc["source_outer_idx"].as_i64().unwrap(),
            &entry,
            sc["block_size"].as_i64().unwrap(),
            &WekaConfig::default(),
        );

        let want = sc["plans"].as_array().unwrap();
        assert_eq!(plans.len(), want.len(), "{name}: plan count");
        for (i, (p, w)) in plans.iter().zip(want).enumerate() {
            assert_eq!(
                p.session_id,
                w["session_id"].as_str().unwrap(),
                "{name} p{i} sid"
            );
            assert_eq!(
                p.chain_index,
                w["chain_index"].as_i64().unwrap() as usize,
                "{name} p{i} cidx"
            );
            let want_idx: Vec<usize> = ints(&w["request_inner_indices"])
                .into_iter()
                .map(|x| x as usize)
                .collect();
            assert_eq!(
                p.request_inner_indices, want_idx,
                "{name} p{i} inner_indices"
            );
            let want_ts: Vec<f64> = w["request_ts"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_f64().unwrap())
                .collect();
            let got_ts: Vec<f64> = p.requests.iter().map(|r| r.t).collect();
            assert_eq!(got_ts, want_ts, "{name} p{i} request_ts");
            assert_eq!(
                p.init_tool_tokens,
                w["init_tool_tokens"].as_i64().unwrap(),
                "{name} p{i} init_tool"
            );
            assert_eq!(
                p.init_system_tokens,
                w["init_system_tokens"].as_i64().unwrap(),
                "{name} p{i} init_sys"
            );
            assert_eq!(
                p.is_aux,
                w["is_aux"].as_bool().unwrap(),
                "{name} p{i} is_aux"
            );
        }
    }
}
