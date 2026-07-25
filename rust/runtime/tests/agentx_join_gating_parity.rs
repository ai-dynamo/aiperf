// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Byte-exact parity of the Rust `agentic_replay` join-gating decision against the
//! real Python `AgenticReplayStrategy` snapshot kernel.
//!
//! The golden (`tools/agentx_join_gating_golden.py`) drives the **real** Python
//! [`aiperf.timing.trajectory_source.TrajectorySource._snapshot_for`] over a fixed
//! root+subagent trace at a deterministic `t*` and records which conversation
//! states are `waiting_on_children` (the gated parent joins), the child ids that
//! gate each, and the child-terminal release order.
//!
//! This test reconstructs the identical logical trace (carried verbatim in the
//! fixture's `trace` block) into [`ReconstructedConversation`]s, builds the join
//! gate description via [`build_tree_specs`], drives a [`TreeGate`], and asserts
//! its independent decision sequence equals the Python-produced `waiting_before`
//! / `gating_children` / `release_order` byte-for-byte:
//!
//! * every `(conversation_id, join_turn_index)` in `waiting_before` is
//!   [`TreeGate::is_waiting`] BEFORE any child terminal;
//! * the gate's join children equal the Python `gating_children`;
//! * terminating children in `release_order` clears each waiting join, and a
//!   join with >1 required child stays waiting until its LAST child terminates.
//!
//! Index note: the Python snapshot's `next_turn_index` is the absolute
//! metadata turn index; this test builds the tree specs from the **unsliced**
//! reconstruction so the join turn index is likewise absolute — the two index
//! spaces coincide, giving a true byte-exact index match on the gating decision.
//! (The separate per-lane `slice_trajectories_at_tstar` rebasing is a dispatch
//! concern, not part of the join-gating rule under test.)

#![cfg(feature = "agentx")]

use std::collections::HashMap;
use std::path::PathBuf;

use aiperf_runtime::agentic_replay::{build_tree_specs, TreeGate};
use aiperf_runtime::agentx::loader::{
    JoinPrerequisite, ReconstructedConversation, ReconstructedTurn,
};
use serde_json::Value;

fn golden_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("tests/fixtures/agentx/join_gating_golden.json")
}

/// A bare reconstructed turn at `ts` ms, optionally carrying a join prerequisite.
fn turn(ts: f64, join: Option<(String, Vec<String>)>) -> ReconstructedTurn {
    ReconstructedTurn {
        timestamp_ms: Some(ts),
        delay_ms: None,
        api_time_ms: None,
        source_trace_id: "trace".into(),
        source_outer_idx: 0,
        source_kind: "weka_main".into(),
        model: "m".into(),
        max_tokens: 1,
        raw_messages: vec![],
        reset_context: false,
        theoretical_prefix_cache_hit_blocks: 0,
        theoretical_prefix_cache_total_blocks: 0,
        input_kind: None,
        spawn_branch: None,
        join_prerequisite: join.map(|(branch_id, child_session_ids)| JoinPrerequisite {
            branch_id,
            child_session_ids,
        }),
    }
}

/// Reconstruct the identical trace the Python golden consumed, from the fixture's
/// `trace` block: the root (with a join prerequisite on its join turn) plus each
/// subagent child conversation.
fn reconstruct_from_fixture(trace: &Value) -> Vec<ReconstructedConversation> {
    let root = &trace["root"];
    let root_id = root["conversation_id"].as_str().unwrap().to_string();
    let root_ts: Vec<f64> = root["turns_ms"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| t.as_f64().unwrap())
        .collect();
    let join = &root["join"];
    let join_turn = join["turn_index"].as_u64().unwrap() as usize;
    let join_branch = join["branch_id"].as_str().unwrap().to_string();
    let join_children: Vec<String> = join["child_conversation_ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| c.as_str().unwrap().to_string())
        .collect();

    let root_turns: Vec<ReconstructedTurn> = root_ts
        .iter()
        .enumerate()
        .map(|(i, &ts)| {
            if i == join_turn {
                turn(ts, Some((join_branch.clone(), join_children.clone())))
            } else {
                turn(ts, None)
            }
        })
        .collect();

    let mut convs = vec![ReconstructedConversation {
        session_id: root_id.clone(),
        replay_scope_id: root_id.clone(),
        parent_conversation_id: None,
        turns: root_turns,
    }];

    for child in trace["children"].as_array().unwrap() {
        let cid = child["conversation_id"].as_str().unwrap().to_string();
        let parent = child["parent_conversation_id"].as_str().unwrap().to_string();
        let turns: Vec<ReconstructedTurn> = child["turns_ms"]
            .as_array()
            .unwrap()
            .iter()
            .map(|t| turn(t.as_f64().unwrap(), None))
            .collect();
        convs.push(ReconstructedConversation {
            session_id: cid,
            replay_scope_id: root_id.clone(),
            parent_conversation_id: Some(parent),
            turns,
        });
    }
    convs
}

#[test]
fn join_gating_decision_matches_python_golden() {
    let raw = std::fs::read(golden_path()).expect("read join_gating_golden.json");
    let golden: Value = serde_json::from_slice(&raw).unwrap();

    let convs = reconstruct_from_fixture(&golden["trace"]);
    let specs = build_tree_specs(&convs);

    // The reconstruction yields exactly the one gated tree the Python snapshot saw.
    assert_eq!(specs.len(), 1, "expected one tree spec");
    let spec = &specs[0];

    // `waiting_before`: every (conversation_id, join_turn_index) Python reported as
    // `waiting_on_children` must be `is_waiting` on the gate BEFORE any terminal.
    let gate = TreeGate::new(&specs);
    let waiting_before = golden["waiting_before"].as_array().unwrap();
    assert!(!waiting_before.is_empty(), "golden has no waiting states");
    for entry in waiting_before {
        let conv = entry[0].as_str().unwrap();
        let idx = entry[1].as_u64().unwrap() as usize;
        assert!(
            gate.is_waiting(conv, idx),
            "gate must be waiting for ({conv}, {idx}) before any child terminal"
        );
    }

    // `gating_children`: the gate's join children for each waiting root equal the
    // Python-reported gating child set.
    let gating: &serde_json::Map<String, Value> =
        golden["gating_children"].as_object().unwrap();
    // Build root -> join_turn -> children from the spec for cross-checking.
    let mut spec_children_by_root: HashMap<&str, Vec<String>> = HashMap::new();
    for (_turn_idx, children) in &spec.join_turns {
        spec_children_by_root
            .entry(spec.root.as_str())
            .or_default()
            .extend(children.iter().cloned());
    }
    for (root, kids) in gating {
        let want: Vec<String> = kids
            .as_array()
            .unwrap()
            .iter()
            .map(|c| c.as_str().unwrap().to_string())
            .collect();
        let mut got = spec_children_by_root.get(root.as_str()).cloned().unwrap_or_default();
        got.sort();
        let mut want_sorted = want.clone();
        want_sorted.sort();
        assert_eq!(got, want_sorted, "gating children mismatch for root {root}");
    }

    // `release_order`: terminating children in the recorded order clears each
    // waiting join. A join with >1 required child stays waiting until its LAST
    // child terminates; the single-child golden releases on that one terminal.
    let release_order: Vec<String> = golden["release_order"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| c.as_str().unwrap().to_string())
        .collect();

    // Track, per waiting (conv, idx), the required children still outstanding.
    let mut outstanding: HashMap<(String, usize), std::collections::HashSet<String>> =
        HashMap::new();
    for entry in waiting_before {
        let conv = entry[0].as_str().unwrap().to_string();
        let idx = entry[1].as_u64().unwrap() as usize;
        let kids = gating
            .get(&conv)
            .unwrap()
            .as_array()
            .unwrap()
            .iter()
            .map(|c| c.as_str().unwrap().to_string())
            .collect();
        outstanding.insert((conv, idx), kids);
    }

    for child in &release_order {
        gate.on_child_terminal(child);
        for ((conv, idx), remaining) in outstanding.iter_mut() {
            remaining.remove(child);
            if remaining.is_empty() {
                assert!(
                    !gate.is_waiting(conv, *idx),
                    "join ({conv}, {idx}) must release once all children terminate"
                );
            } else {
                assert!(
                    gate.is_waiting(conv, *idx),
                    "join ({conv}, {idx}) must stay waiting while children remain"
                );
            }
        }
    }

    // After the full release order, no golden waiting join remains gated.
    for entry in waiting_before {
        let conv = entry[0].as_str().unwrap();
        let idx = entry[1].as_u64().unwrap() as usize;
        assert!(
            !gate.is_waiting(conv, idx),
            "({conv}, {idx}) must be released after all children terminate"
        );
    }
}
