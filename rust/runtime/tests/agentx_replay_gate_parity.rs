// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Byte-exact parity of the Rust [`ReplayGate`] against the real Python
//! [`aiperf.timing.replay_dependencies.ReplayBarrierCoordinator`].
//!
//! The golden (`tools/agentx_replay_gate_golden.py`) drives the **real** Python
//! coordinator through a fixed scripted scenario over a recorded predecessor
//! graph spanning three runtime roots (R1: cross-stream, join-width-2 barrier
//! release chain; R2: seeded resume prefix; R3: pause-retains-newly-ready) and
//! records the release order, `completed_prefixes` at checkpoints, and a
//! `pending_turns_by_root` snapshot.
//!
//! This test replays the IDENTICAL scripted scenario through the Rust gate and
//! asserts release order + completed prefixes + pending-by-root match the golden
//! byte-for-byte. The scenario is authored in both places (there is no wire
//! serialization of the ops), so the parity guarantee is: given the same fixed
//! predecessor graph and the same op sequence, both implementations produce the
//! same observable release/prefix/pending outputs.

use std::path::PathBuf;

use aiperf_runtime::agentx::replay_dependencies::{ReplayResumeBoundary, ReplayTurnKey};
use aiperf_runtime::agentx::replay_gate::{ReplayGate, ReplayTurn};
use serde_json::Value;

const R1: &str = "root-1";
const R2: &str = "root-2";
const R3: &str = "root-3";

fn golden_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("tests/fixtures/agentx/replay_gate_golden.json")
}

fn key(conv: &str, ti: i64) -> ReplayTurnKey {
    ReplayTurnKey {
        conversation_id: conv.into(),
        turn_index: ti,
    }
}

/// The identical recorded predecessor graph the golden's `build_dataset` encodes:
/// B0<-{A0}, C0<-{A1,B0}, G0<-{F0}; A0/A1/F0 have none.
fn predecessors() -> std::collections::BTreeMap<ReplayTurnKey, Vec<ReplayTurnKey>> {
    let mut m = std::collections::BTreeMap::new();
    m.insert(key("A", 0), vec![]);
    m.insert(key("A", 1), vec![]);
    m.insert(key("B", 0), vec![key("A", 0)]);
    m.insert(key("C", 0), vec![key("A", 1), key("B", 0)]);
    m.insert(key("F", 0), vec![]);
    m.insert(key("G", 0), vec![key("F", 0)]);
    m
}

/// `[["A",0],...]` (release order / pending) -> `Vec<(conv, idx)>`.
fn pairs(value: &Value) -> Vec<(String, i64)> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|e| (e[0].as_str().unwrap().to_string(), e[1].as_i64().unwrap()))
        .collect()
}

#[test]
fn replay_gate_matches_python_golden() {
    let raw = std::fs::read(golden_path()).expect("read replay_gate_golden.json");
    let golden: Value = serde_json::from_slice(&raw).unwrap();

    let mut gate = ReplayGate::new(predecessors());
    gate.activate();

    // --- R1: cross-stream barrier release chain (unpaused) -------------------
    gate.submit(ReplayTurn::new(R1, "A", 0)).unwrap();
    gate.submit(ReplayTurn::new(R1, "B", 0)).unwrap();
    gate.submit(ReplayTurn::new(R1, "C", 0)).unwrap();
    gate.submit(ReplayTurn::new(R1, "A", 1)).unwrap();
    gate.complete(R1, key("A", 0));
    gate.complete(R1, key("A", 1));
    gate.complete(R1, key("B", 0));

    let r1_prefixes = gate.completed_prefixes(R1).unwrap();

    // --- R2: seed a resume prefix, then read it back -------------------------
    gate.seed_completed_prefixes(
        R2,
        &[ReplayResumeBoundary {
            conversation_id: "D".into(),
            next_turn_index: 2,
        }],
    )
    .unwrap();
    let r2_prefixes = gate.completed_prefixes(R2).unwrap();

    // --- R3: pause then submit; newly-ready work is retained -----------------
    gate.pause_releases();
    gate.submit(ReplayTurn::new(R3, "F", 0)).unwrap();
    gate.submit(ReplayTurn::new(R3, "G", 0)).unwrap();
    gate.complete(R3, key("F", 0));

    // --- Assert release order byte-for-byte ----------------------------------
    let want_release = pairs(&golden["release_order"]);
    let got_release: Vec<(String, i64)> = gate
        .released()
        .iter()
        .map(|k| (k.conversation_id.clone(), k.turn_index))
        .collect();
    assert_eq!(got_release, want_release, "release order mismatch");

    // --- Assert completed_prefixes checkpoints -------------------------------
    let cp = &golden["completed_prefixes"];
    let want_r1 = pairs(&cp["r1_final"]);
    let got_r1: Vec<(String, i64)> = r1_prefixes
        .iter()
        .map(|b| (b.conversation_id.clone(), b.next_turn_index))
        .collect();
    assert_eq!(got_r1, want_r1, "r1 completed_prefixes mismatch");

    let want_r2 = pairs(&cp["r2_after_seed"]);
    let got_r2: Vec<(String, i64)> = r2_prefixes
        .iter()
        .map(|b| (b.conversation_id.clone(), b.next_turn_index))
        .collect();
    assert_eq!(got_r2, want_r2, "r2 completed_prefixes mismatch");

    // --- Assert pending_turns_by_root snapshot -------------------------------
    let want_pending = golden["pending_turns_by_root"].as_object().unwrap();
    let got_pending = gate.pending_turns_by_root();
    assert_eq!(
        got_pending.len(),
        want_pending.len(),
        "pending root-count mismatch"
    );
    for (root, turns) in want_pending {
        let want_turns = pairs(turns);
        let got_turns: Vec<(String, i64)> = got_pending
            .get(root)
            .unwrap_or_else(|| panic!("missing pending root {root}"))
            .iter()
            .map(|t| (t.key.conversation_id.clone(), t.key.turn_index))
            .collect();
        assert_eq!(got_turns, want_turns, "pending turns mismatch for {root}");
    }
}
