// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Byte-exact parity of the Rust warmup-handoff residual/rebuild logic against
//! the real Python [`AgenticReplayStrategy`] oracle.
//!
//! The golden (`tools/agentx_handoff_residual_golden.py`) drives the **real**
//! Python `_handoff_base_delay_ms` / `_handoff_residual_delay_ms` over fixed
//! `(base inputs, returned_ns, finalized_ns, cap)` rows (exercising the
//! `delay_ms` path, the timestamp fallback, the non-finite guard, the
//! elapsed-subtraction floor, and the idle-gap-cap clamp), plus the real
//! `_build_handoff_replay_boundaries` + `_build_handoff_trajectories` state sort
//! `(agent_depth, x_correlation_id)`, boundary merge, and empty-lane recycle
//! draw. Correlation ids for the recycle draw are injected from an identical
//! seeded sequence on both sides.
//!
//! All wall values are Clock-derived nanoseconds (never `Instant::now`);
//! byte-exactness holds only under `SimClock`.

#![cfg(feature = "agentx")]

use std::collections::BTreeMap;
use std::path::PathBuf;

use aiperf_runtime::agentx::handoff::{
    BranchMode, FinalizeInputs, HandoffBaseDelayInputs, HandoffCredit, base_delay_ms, finalize,
    residual_delay_ms,
};
use aiperf_runtime::agentx::replay_dependencies::ReplayResumeBoundary;
use serde_json::Value;

fn golden_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("tests/fixtures/agentx/handoff_residual_golden.json")
}

fn load() -> Value {
    let bytes = std::fs::read(golden_path()).expect("read handoff golden fixture");
    serde_json::from_slice(&bytes).expect("parse handoff golden fixture")
}

/// Decode the JSON-safe float encoding: `null` -> None, `"inf"/"-inf"/"nan"` ->
/// the non-finite float, a number -> its value.
fn decode_f64(v: &Value) -> Option<f64> {
    match v {
        Value::Null => None,
        Value::String(s) => Some(match s.as_str() {
            "inf" => f64::INFINITY,
            "-inf" => f64::NEG_INFINITY,
            "nan" => f64::NAN,
            other => panic!("unexpected float string {other}"),
        }),
        Value::Number(n) => Some(n.as_f64().expect("f64")),
        other => panic!("unexpected float value {other:?}"),
    }
}

#[test]
fn residual_and_base_delay_match_python_oracle() {
    let golden = load();
    let rows = golden["residual_rows"].as_array().expect("residual_rows");
    assert!(!rows.is_empty());

    for row in rows {
        let name = row["name"].as_str().unwrap_or("<unnamed>");
        let base_inputs = HandoffBaseDelayInputs {
            next_delay_ms: decode_f64(&row["next_delay_ms"]),
            prev_timestamp_ms: decode_f64(&row["prev_timestamp_ms"]),
            next_timestamp_ms: decode_f64(&row["next_timestamp_ms"]),
            prev_api_time_ms: decode_f64(&row["prev_api_time_ms"]),
        };
        let base = base_delay_ms(&base_inputs);
        let expected_base = row["expected_base_ms"].as_f64().unwrap();
        assert_eq!(base, expected_base, "base delay mismatch for {name}");

        let returned_ns = row["returned_ns"].as_i64();
        let finalized_ns = row["finalized_ns"].as_i64().unwrap();
        let cap_ms = decode_f64(&row["cap_ms"]);
        let residual = residual_delay_ms(base, returned_ns, finalized_ns, cap_ms);
        let expected_residual = row["expected_residual_ms"].as_f64().unwrap();
        assert_eq!(residual, expected_residual, "residual mismatch for {name}");
    }
}

#[test]
fn finalize_rebuild_matches_python_trajectory_oracle() {
    let golden = load();
    let traj = &golden["trajectory"];
    let num_lanes = traj["num_lanes"].as_u64().unwrap() as usize;

    // Reconstruct the input states as returned mid-flight credits: a
    // ConversationState with next_turn_index N corresponds to a returned credit
    // on turn N-1 that is not its final turn.
    let mut handoff_credits: BTreeMap<String, HandoffCredit> = BTreeMap::new();
    let mut root_to_lane: BTreeMap<String, usize> = BTreeMap::new();
    for state in traj["input_states"].as_array().unwrap() {
        let lane = state["lane"].as_u64().unwrap() as usize;
        let next_turn_index = state["next_turn_index"].as_i64().unwrap();
        let x_correlation_id = state["x_correlation_id"].as_str().unwrap().to_string();
        let root_correlation_id = state["root_correlation_id"].as_str().map(|s| s.to_string());
        let credit = HandoffCredit {
            conversation_id: state["conversation_id"].as_str().unwrap().to_string(),
            x_correlation_id: x_correlation_id.clone(),
            // next_turn_index = turn_index + 1; keep well short of num_turns so
            // the credit is non-final and produces a live handoff state.
            turn_index: (next_turn_index - 1) as usize,
            num_turns: (next_turn_index + 5) as usize,
            agent_depth: state["agent_depth"].as_i64().unwrap(),
            parent_correlation_id: None,
            root_correlation_id: root_correlation_id.clone(),
            branch_mode: BranchMode::default(),
        };
        let effective_root = root_correlation_id.unwrap_or(x_correlation_id.clone());
        root_to_lane.insert(effective_root, lane);
        handoff_credits.insert(x_correlation_id, credit);
    }

    // Completed-prefix history per tree root, from the fixture.
    let mut completed: BTreeMap<String, Vec<ReplayResumeBoundary>> = BTreeMap::new();
    for (root, boundaries) in traj["completed_prefixes"].as_object().unwrap() {
        let bs = boundaries
            .as_array()
            .unwrap()
            .iter()
            .map(|b| {
                let pair = b.as_array().unwrap();
                ReplayResumeBoundary {
                    conversation_id: pair[0].as_str().unwrap().to_string(),
                    next_turn_index: pair[1].as_i64().unwrap(),
                }
            })
            .collect();
        completed.insert(root.clone(), bs);
    }

    // Deterministic recycle draws, injected identically to the Python oracle.
    let recycle_convs: Vec<String> = traj["recycle_conversation_ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap().to_string())
        .collect();
    let recycle_corrs: Vec<String> = traj["recycle_correlation_ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap().to_string())
        .collect();
    let mut draw_idx = 0usize;

    let return_wall_ns: BTreeMap<String, i64> = BTreeMap::new();
    let correlation_to_lane: BTreeMap<String, usize> = BTreeMap::new();
    let pending_by_root = BTreeMap::new();

    let handoff = finalize(FinalizeInputs {
        handoff_credits: &handoff_credits,
        return_wall_ns: &return_wall_ns,
        pending_by_root: &pending_by_root,
        root_to_lane: &root_to_lane,
        correlation_to_lane: &correlation_to_lane,
        num_lanes,
        finalized_ns: 0,
        cap_ms: None,
        base_delay_inputs: |_c: &HandoffCredit| HandoffBaseDelayInputs::default(),
        completed_prefixes: |root: &str| completed.get(root).cloned().unwrap_or_default(),
        recycle_draw: || {
            if draw_idx < recycle_convs.len() {
                let draw = (
                    recycle_convs[draw_idx].clone(),
                    recycle_corrs[draw_idx].clone(),
                );
                draw_idx += 1;
                Some(draw)
            } else {
                None
            }
        },
        prev_lanes: &[],
    });

    for expected_lane in traj["expected_lanes"].as_array().unwrap() {
        let lane = expected_lane["lane"].as_u64().unwrap() as usize;
        let got = handoff.lanes.get(&lane).expect("lane present");

        let got_order: Vec<(i64, String)> = got
            .states
            .iter()
            .map(|s| (s.agent_depth, s.x_correlation_id.clone()))
            .collect();
        let expected_order: Vec<(i64, String)> = expected_lane["state_order"]
            .as_array()
            .unwrap()
            .iter()
            .map(|p| {
                let pair = p.as_array().unwrap();
                (
                    pair[0].as_i64().unwrap(),
                    pair[1].as_str().unwrap().to_string(),
                )
            })
            .collect();
        assert_eq!(
            got_order, expected_order,
            "state order mismatch lane {lane}"
        );

        let got_bounds: Vec<(String, i64)> = got
            .boundaries
            .iter()
            .map(|b| (b.conversation_id.clone(), b.next_turn_index))
            .collect();
        let expected_bounds: Vec<(String, i64)> = expected_lane["boundaries"]
            .as_array()
            .unwrap()
            .iter()
            .map(|p| {
                let pair = p.as_array().unwrap();
                (
                    pair[0].as_str().unwrap().to_string(),
                    pair[1].as_i64().unwrap(),
                )
            })
            .collect();
        assert_eq!(got_bounds, expected_bounds, "boundary mismatch lane {lane}");
    }
}
