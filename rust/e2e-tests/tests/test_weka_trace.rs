// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! WEKA-trace replay: prefix-hash scoping, flattened-subagent prefix sharing, and
//! rerun invariance.
//!
//! `hash_id_scope` decides whether a trace's `hash_ids` name a per-trace namespace
//! or one shared across every trace in the set
//! (`runtime/src/graph/recorded/weka/mod.rs:94` — `hash_scope` is `Some(trace.id)`
//! under `"local"` and `None` under `"global"`). That is invisible in aggregates:
//! it only shows up in the rendered prompt bytes, which is what these tests read.
//!
//! Note the structural difference from the Python model this ports: the Rust loader
//! flattens `WekaEntry::Subagent` into its parent chain, so a subagent shares the
//! parent's `x_correlation_id` and emits no `parent_correlation_id`. Grouping is
//! therefore by `(x_correlation_id, turn_index)`, not by parent/child linkage.

mod common;
use common::*;

use serde_json::{Value, json};
use std::collections::{HashMap, HashSet};
use std::path::Path;

/// One 64-token block, so `in` is an exact multiple of `hash_ids.len()`.
const BLOCK: u64 = 64;
/// The hash ids shared between the two traces in the scoping pair.
const SHARED_HASHES: [u64; 3] = [10, 11, 12];

/// A plain single-request WEKA turn over `hashes`, sized to one block per hash.
fn turn(hashes: &[u64], stop: &str) -> Value {
    json!({
        "t": 0.0,
        "type": "n",
        "model": "test-model",
        "in": BLOCK * hashes.len() as u64,
        "out": 8,
        "hash_ids": hashes,
        "input_types": ["text"],
        "output_types": ["text"],
        "stop": stop,
        "api_time": 0.05,
        "think_time": 0.0,
    })
}

/// A whole WEKA trace document.
fn trace(id: &str, scope: &str, requests: Vec<Value>) -> Value {
    json!({
        "id": id,
        "models": ["test-model"],
        "block_size": BLOCK,
        "hash_id_scope": scope,
        "tool_tokens": 0,
        "system_tokens": 0,
        "requests": requests,
    })
}

/// Write one `.json` document per trace into a fresh temp dir; the WEKA loader
/// reads every `.json` in a directory non-recursively.
fn write_trace_dir(traces: &[Value]) -> (tempfile::TempDir, String) {
    let dir = tempfile::tempdir().expect("weka fixture tempdir");
    for (i, t) in traces.iter().enumerate() {
        std::fs::write(
            dir.path().join(format!("trace-{i:03}.json")),
            serde_json::to_vec(t).expect("serialize weka trace"),
        )
        .expect("write weka trace");
    }
    let path = dir.path().display().to_string();
    (dir, path)
}

/// Replay `input` for `conversations` conversations and return the raw records.
fn replay(h: &AIPerfHarness, input: &str, conversations: u32) -> Vec<Value> {
    let r = h.run_timeout(
        &format!(
            "--model test-model --url {} --endpoint-type chat --input-file {input} \
             --custom-dataset-type weka_trace --num-conversations {conversations} \
             --concurrency 1 --export-level raw --ui simple",
            h.mock.url
        ),
        300,
    );
    assert!(r.success(), "weka replay failed: {}", r.stderr);
    let raw = r.artifacts.raw_records();
    assert!(!raw.is_empty(), "weka replay produced no records");
    raw
}

/// The concatenated user text of a record's wire payload — the only place prefix
/// scoping is observable.
fn user_text(rec: &Value) -> String {
    rec["payload"]["messages"]
        .as_array()
        .expect("payload.messages array")
        .iter()
        .filter(|m| m["role"] == "user")
        .map(|m| match &m["content"] {
            Value::String(s) => s.clone(),
            Value::Array(parts) => parts
                .iter()
                .filter_map(|p| p["text"].as_str())
                .collect::<String>(),
            other => panic!("unexpected message content: {other}"),
        })
        .collect()
}

fn meta_str(rec: &Value, key: &str) -> String {
    rec["metadata"][key]
        .as_str()
        .unwrap_or_else(|| panic!("metadata.{key} missing/non-string: {}", rec["metadata"]))
        .to_string()
}

fn meta_u64(rec: &Value, key: &str) -> u64 {
    rec["metadata"][key]
        .as_u64()
        .unwrap_or_else(|| panic!("metadata.{key} missing/non-int: {}", rec["metadata"]))
}

/// Replay two traces whose `hash_ids` are identical and whose only difference is
/// `hash_id_scope`, and return the set of distinct rendered prompts for each.
///
/// This is the discriminating pair: under `"local"` the shared ids sit in two
/// separate per-trace namespaces and must render two different prompts; under
/// `"global"` they name the same blocks and must render one identical prompt. A
/// regression that collapses or ignores the scope fails exactly one half, so
/// neither half can pass for the wrong reason on its own.
fn distinct_prompts_for_scope(h: &AIPerfHarness, scope: &str) -> HashSet<String> {
    let (_guard, dir) = write_trace_dir(&[
        trace("trace_a", scope, vec![turn(&SHARED_HASHES, "end_turn")]),
        trace("trace_b", scope, vec![turn(&SHARED_HASHES, "end_turn")]),
    ]);
    let raw = replay(h, &dir, 2);
    assert_eq!(raw.len(), 2, "expected one record per trace: {}", raw.len());
    raw.iter().map(user_text).collect()
}

/// `hash_id_scope: "local"` keeps each trace's hash ids in its own namespace.
#[tokio::test]
async fn test_local_hash_scope_renders_distinct_prefixes_per_trace() {
    let h = AIPerfHarness::new().await;
    let prompts = distinct_prompts_for_scope(&h, "local");
    assert_eq!(
        prompts.len(),
        2,
        "two traces sharing hash_ids {SHARED_HASHES:?} under a local scope must render \
         different prefixes; got {} distinct prompt(s)",
        prompts.len()
    );
}

/// `hash_id_scope: "global"` puts every trace's hash ids in one namespace, so the
/// same ids must resolve to the same blocks.
#[tokio::test]
async fn test_global_hash_scope_shares_prefixes_across_traces() {
    let h = AIPerfHarness::new().await;
    let prompts = distinct_prompts_for_scope(&h, "global");
    assert_eq!(
        prompts.len(),
        1,
        "two traces sharing hash_ids {SHARED_HASHES:?} under a global scope must render \
         the same prefix; got {} distinct prompt(s)",
        prompts.len()
    );
}

/// Subagents flattened into the parent chain share the parent's hash namespace.
///
/// The parent turn and both sibling subagents reference the same `hash_ids`, so all
/// three must render byte-identical user text; the following turn extends that same
/// prefix rather than starting a new one. This is the property the Python
/// `test_weka_hash_id_scope` asserted through parent/child grouping, restated for
/// the flattened Rust shape.
#[tokio::test]
async fn test_flattened_subagents_share_the_parent_hash_scope() {
    fn subagent(agent_id: &str, t: f64) -> Value {
        json!({
            "t": t,
            "type": "subagent",
            "agent_id": agent_id,
            "subagent_type": "Explore",
            "duration_ms": 100,
            "total_tokens": 100,
            "tool_use_count": 1,
            "status": "completed",
            "models": ["test-model"],
            "tool_tokens": 0,
            "system_tokens": 0,
            "requests": [turn(&SHARED_HASHES, "end_turn")],
        })
    }

    let mut extended = SHARED_HASHES.to_vec();
    extended.push(13);
    let doc = trace(
        "scope_stress",
        "local",
        vec![
            turn(&SHARED_HASHES, "tool_use"),
            subagent("agent_001", 0.2),
            subagent("agent_002", 0.3),
            turn(&extended, "end_turn"),
        ],
    );

    let h = AIPerfHarness::new().await;
    let (_guard, dir) = write_trace_dir(&[doc]);
    let raw = replay(&h, &dir, 1);

    let mut by_turn: HashMap<u64, Vec<&Value>> = HashMap::new();
    for rec in &raw {
        by_turn
            .entry(meta_u64(rec, "turn_index"))
            .or_default()
            .push(rec);
    }

    let turn0 = by_turn.get(&0).expect("no turn_index 0 records");
    assert_eq!(
        turn0.len(),
        3,
        "the parent turn and both flattened subagents must land on turn_index 0; got {}",
        turn0.len()
    );
    let shared: HashSet<String> = turn0.iter().map(|r| user_text(r)).collect();
    assert_eq!(
        shared.len(),
        1,
        "the parent and both subagents reference hash_ids {SHARED_HASHES:?} in one scope, \
         so their prompts must be byte-identical; got {} distinct",
        shared.len()
    );
    let shared = shared.into_iter().next().expect("one shared prompt");

    let turn1 = by_turn.get(&1).expect("no turn_index 1 records");
    let extended_text = user_text(turn1[0]);
    assert!(
        extended_text.starts_with(&shared),
        "the extended turn must reuse the shared prefix verbatim, not re-render it"
    );
    assert!(
        extended_text.len() > shared.len(),
        "the extended turn adds a fourth block, so it must be longer than the prefix"
    );

    // Flattening, not parent/child linkage: one correlation id, no parent pointer.
    let corr: HashSet<String> = raw
        .iter()
        .map(|r| meta_str(r, "x_correlation_id"))
        .collect();
    assert_eq!(
        corr.len(),
        1,
        "flattened subagents stay in the parent conversation; got ids {corr:?}"
    );
}

/// Two runs of the same directory fixture must issue the same per-record input
/// sequence lengths.
///
/// Asserting on prompt *text* here would be flaky: the synthetic filler is not
/// seeded run to run, and two runs at identical settings were observed to differ in
/// every conversation's rendered text. The block structure the trace prescribes is
/// what must be stable, and `metrics.input_sequence_length` keyed by
/// `(x_correlation_id, turn_index)` is exactly that.
#[tokio::test]
async fn test_replay_input_lengths_are_identical_across_reruns() {
    const FIXTURE: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/weka_traces_small"
    );
    assert!(Path::new(FIXTURE).exists(), "fixture missing: {FIXTURE}");

    /// `(x_correlation_id, turn_index) -> input_sequence_length` for one run.
    fn isl_by_turn(h: &AIPerfHarness) -> HashMap<(String, u64), u64> {
        let r = h.run_timeout(
            &format!(
                "--model test-model --url {} --endpoint-type chat --input-file {FIXTURE} \
                 --custom-dataset-type weka_trace --num-conversations 10 --concurrency 4 \
                 --ui simple",
                h.mock.url
            ),
            300,
        );
        assert!(r.success(), "weka replay failed: {}", r.stderr);
        let recs = r.artifacts.jsonl();
        assert!(!recs.is_empty(), "weka replay produced no records");
        recs.iter()
            .map(|rec| {
                let isl = rec["metrics"]["input_sequence_length"]["value"]
                    .as_f64()
                    .unwrap_or_else(|| {
                        panic!("metrics.input_sequence_length missing: {}", rec["metrics"])
                    });
                (
                    (
                        meta_str(rec, "x_correlation_id"),
                        meta_u64(rec, "turn_index"),
                    ),
                    isl as u64,
                )
            })
            .collect()
    }

    let h = AIPerfHarness::new().await;
    let first = isl_by_turn(&h);
    let second = isl_by_turn(&h);

    assert_eq!(
        first.keys().collect::<HashSet<_>>(),
        second.keys().collect::<HashSet<_>>(),
        "reruns must replay the same (conversation, turn) set"
    );
    let drift: Vec<_> = first
        .iter()
        .filter(|(k, v)| second.get(*k) != Some(*v))
        .collect();
    assert!(
        drift.is_empty(),
        "input sequence lengths drifted between identical reruns: {drift:?}"
    );
}
