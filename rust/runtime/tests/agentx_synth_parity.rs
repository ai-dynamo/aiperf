// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Byte-exact parity of the Rust `ConversationReconstructor` against the Python
//! `weka_synth_buf.ConversationReconstructor`.
//!
//! The golden file `tests/fixtures/agentx/synth_golden.json` is produced by
//! `tools/agentx_synth_golden.py` running the real Python reconstructor with a
//! deterministic stub token generator. This test replays the same scenarios with
//! an identical stub and asserts the emitted `TurnDelta`s and full segment state
//! match the Python output field-for-field.


use aiperf_runtime::agentx::synth::{ConversationReconstructor, TokenSynth};
use serde_json::Value;
use std::path::PathBuf;

/// Deterministic stub identical to `make_stub` in `tools/agentx_synth_golden.py`.
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
        tokens
            .iter()
            .map(|t| t.to_string())
            .collect::<Vec<_>>()
            .join(" ")
    }
}

fn golden_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("tests/fixtures/agentx/synth_golden.json")
}

fn ints(v: &Value) -> Vec<i64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_i64().unwrap())
        .collect()
}

#[test]
fn reconstructor_matches_python_golden() {
    let raw = std::fs::read(golden_path()).expect("read synth_golden.json");
    let scenarios: Value = serde_json::from_slice(&raw).unwrap();

    for sc in scenarios.as_array().unwrap() {
        let name = sc["name"].as_str().unwrap();
        let bs = sc["block_size"].as_i64().unwrap();
        let mut synth = StubSynth { bs };
        let mut r = ConversationReconstructor::new(bs, false);

        let steps_in = sc["steps_input"].as_array().unwrap();
        let steps_out = sc["steps_output"].as_array().unwrap();
        assert_eq!(steps_in.len(), steps_out.len(), "{name}: step count");

        for (i, (sin, sout)) in steps_in.iter().zip(steps_out).enumerate() {
            match sin["op"].as_str().unwrap() {
                "init" => {
                    r.init_turn_0(
                        &mut synth,
                        &ints(&sin["hash_ids"]),
                        sin["in"].as_i64().unwrap(),
                        sin["tool"].as_i64().unwrap(),
                        sin["system"].as_i64().unwrap(),
                        sin["seed"].as_str().unwrap(),
                    )
                    .unwrap_or_else(|e| panic!("{name} step {i} init: {e}"));
                }
                "advance" => {
                    let max_asst = sin["max_asst_blocks"].as_i64();
                    r.advance_turn(
                        &mut synth,
                        &ints(&sin["prev_hash_ids"]),
                        sin["prev_out"].as_i64().unwrap(),
                        &ints(&sin["hash_ids"]),
                        sin["in"].as_i64().unwrap(),
                        sin["seed"].as_str().unwrap(),
                        sin["is_tool_result"].as_bool().unwrap(),
                        max_asst,
                    );
                }
                other => panic!("unknown op {other}"),
            }

            let delta = r.turn_delta();

            // Compare TurnDelta.
            let want_reset = sout["reset_context"].as_bool().unwrap();
            assert_eq!(delta.reset_context, want_reset, "{name} step {i}: reset");
            let want_msgs = sout["delta_messages"].as_array().unwrap();
            assert_eq!(
                delta.delta_messages.len(),
                want_msgs.len(),
                "{name} step {i}: msg count"
            );
            for (m, wm) in delta.delta_messages.iter().zip(want_msgs) {
                assert_eq!(m.role, wm["role"].as_str().unwrap(), "{name} step {i}: role");
                assert_eq!(
                    m.content,
                    wm["content"].as_str().unwrap(),
                    "{name} step {i}: content"
                );
            }

            // Compare full segment state.
            let want_segs = sout["segments"].as_array().unwrap();
            let got = r.segments();
            assert_eq!(got.len(), want_segs.len(), "{name} step {i}: seg count");
            for (seg, ws) in got.iter().zip(want_segs) {
                assert_eq!(seg.role.as_str(), ws["role"].as_str().unwrap(), "{name} s{i} role");
                assert_eq!(
                    seg.block_start,
                    ws["block_start"].as_i64().unwrap(),
                    "{name} s{i} block_start"
                );
                assert_eq!(
                    seg.block_count,
                    ws["block_count"].as_i64().unwrap(),
                    "{name} s{i} block_count"
                );
                let want_tokens: Vec<u32> =
                    ints(&ws["tokens"]).into_iter().map(|x| x as u32).collect();
                assert_eq!(seg.tokens, want_tokens, "{name} s{i} tokens");
                assert_eq!(
                    seg.content,
                    ws["content"].as_str().unwrap(),
                    "{name} s{i} content"
                );
                let want_trt = ws["tool_result_turn"].as_i64();
                assert_eq!(seg.tool_result_turn, want_trt, "{name} s{i} tool_result_turn");
            }
        }

        // Compare trailing-non-user turns.
        let want_trailing = ints(&sc["trailing_non_user_turns"]);
        assert_eq!(
            r.trailing_non_user_turns(),
            want_trailing.as_slice(),
            "{name}: trailing_non_user_turns"
        );
    }
}
