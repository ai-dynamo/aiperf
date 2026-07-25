// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Byte-exact parity of the Rust main-conversation loader loop against the real
//! Python `WekaTraceLoader._reconstruct_serial` main path.
//!
//! Golden produced by `tools/agentx_loader_golden.py` (real `ConversationReconstructor`
//! + real loop helpers + stub token generator). This replays with an identical
//! stub and diffs the full reconstructed conversation.

#![cfg(feature = "agentx")]

use aiperf_runtime::agentx::loader::{
    reconstruct_main_conversation, MainReconstructOptions, NormalReq, TurnInputKind,
};
use aiperf_runtime::agentx::synth::TokenSynth;
use serde_json::Value;
use std::collections::HashMap;
use std::path::PathBuf;

struct StubSynth {
    bs: i64,
}
impl TokenSynth for StubSynth {
    fn decode_block_tokens(&mut self, hash_ids: &[i64]) -> Vec<u32> {
        hash_ids
            .iter()
            .flat_map(|&h| (0..self.bs).map(move |i| (h as u32) * 1000 + i as u32))
            .collect()
    }
    fn sample_partial_tail_tokens(&mut self, n: usize, _seed: &str) -> Vec<u32> {
        (0..n as u32).map(|i| 900_000 + i).collect()
    }
    fn decode_tokens_to_text(&self, tokens: &[u32]) -> String {
        tokens.iter().map(|t| t.to_string()).collect::<Vec<_>>().join(" ")
    }
}

fn golden_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("tests/fixtures/agentx/loader_golden.json")
}

fn ints(v: &Value) -> Vec<i64> {
    v.as_array().unwrap().iter().map(|x| x.as_i64().unwrap()).collect()
}
fn strs(v: &Value) -> Vec<String> {
    v.as_array().unwrap().iter().map(|x| x.as_str().unwrap().to_string()).collect()
}
fn opt_f64(v: &Value) -> Option<f64> {
    if v.is_null() { None } else { Some(v.as_f64().unwrap()) }
}

#[test]
fn main_conversation_matches_python_golden() {
    let raw = std::fs::read(golden_path()).expect("read loader_golden.json");
    let scenarios: Value = serde_json::from_slice(&raw).unwrap();

    for sc in scenarios.as_array().unwrap() {
        let name = sc["name"].as_str().unwrap();
        let tid = sc["trace_id"].as_str().unwrap();
        let bs = sc["block_size"].as_i64().unwrap();
        let normals: Vec<(i64, NormalReq)> = sc["normals"]
            .as_array()
            .unwrap()
            .iter()
            .map(|n| {
                (
                    n["outer"].as_i64().unwrap(),
                    NormalReq {
                        t: n["t"].as_f64().unwrap(),
                        api_time: opt_f64(&n["api_time"]),
                        think_time: opt_f64(&n["think_time"]),
                        model: n["model"].as_str().unwrap().to_string(),
                        hash_ids: ints(&n["hash_ids"]),
                        input_length: n["in"].as_i64().unwrap(),
                        output_length: n["out"].as_i64().unwrap(),
                        input_types: strs(&n["input_types"]),
                        stop: n["stop"].as_str().unwrap().to_string(),
                    },
                )
            })
            .collect();

        let mut synth = StubSynth { bs };
        let opts = MainReconstructOptions {
            think_time_only: sc["think_time_only"].as_bool().unwrap(),
            ..Default::default()
        };
        let conv = reconstruct_main_conversation(
            tid,
            bs,
            sc["tool_tokens"].as_i64().unwrap(),
            sc["system_tokens"].as_i64().unwrap(),
            &normals,
            &mut synth,
            &HashMap::new(),
            &opts,
        )
        .unwrap_or_else(|e| panic!("{name}: {e}"));

        let want_turns = sc["turns"].as_array().unwrap();
        assert_eq!(conv.turns.len(), want_turns.len(), "{name}: turn count");
        for (i, (t, w)) in conv.turns.iter().zip(want_turns).enumerate() {
            assert_eq!(t.timestamp_ms, opt_f64(&w["timestamp_ms"]), "{name} t{i} timestamp");
            assert_eq!(t.delay_ms, opt_f64(&w["delay_ms"]), "{name} t{i} delay");
            assert_eq!(t.api_time_ms, opt_f64(&w["api_time_ms"]), "{name} t{i} api_time");
            assert_eq!(t.source_outer_idx, w["source_outer_idx"].as_i64().unwrap(), "{name} t{i} outer");
            assert_eq!(t.source_kind, w["source_kind"].as_str().unwrap(), "{name} t{i} kind");
            assert_eq!(t.model, w["model"].as_str().unwrap(), "{name} t{i} model");
            assert_eq!(t.max_tokens, w["max_tokens"].as_i64().unwrap(), "{name} t{i} max_tokens");
            assert_eq!(t.reset_context, w["reset_context"].as_bool().unwrap(), "{name} t{i} reset");
            assert_eq!(
                t.theoretical_prefix_cache_hit_blocks,
                w["theoretical_prefix_cache_hit_blocks"].as_i64().unwrap(),
                "{name} t{i} hit"
            );
            assert_eq!(
                t.theoretical_prefix_cache_total_blocks,
                w["theoretical_prefix_cache_total_blocks"].as_i64().unwrap(),
                "{name} t{i} total"
            );
            let want_ik = match w["input_kind"].as_str() {
                None => None,
                Some("user_input") => Some(TurnInputKind::UserInput),
                Some("tool_result") => Some(TurnInputKind::ToolResult),
                other => panic!("bad input_kind {other:?}"),
            };
            assert_eq!(t.input_kind, want_ik, "{name} t{i} input_kind");
            // raw_messages role/content.
            let want_msgs = w["raw_messages"].as_array().unwrap();
            assert_eq!(t.raw_messages.len(), want_msgs.len(), "{name} t{i} msg count");
            for (m, wm) in t.raw_messages.iter().zip(want_msgs) {
                assert_eq!(m.role, wm["role"].as_str().unwrap(), "{name} t{i} role");
                assert_eq!(m.content, wm["content"].as_str().unwrap(), "{name} t{i} content");
            }
        }
    }
}
