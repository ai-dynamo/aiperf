// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
mod common;
use common::*;

// Full two-branch DAG topology coverage.

use serde_json::Value;

const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../tests/fixtures/dag/full.dag.jsonl"
);

const ROOT_SYS: &str = "root system prompt";
const ROOT_USER: &str = "root user prompt";

const A0_USER_A: &str = "branch-a turn-0 user message A";
const A0_USER_B: &str = "branch-a turn-0 user message B";

const A1_USER_A: &str = "branch-a turn-1 user message A";
const A1_USER_B: &str = "branch-a turn-1 user message B";

const B0_USER: &str = "branch-b turn-0 user message";
const B1_USER: &str = "branch-b turn-1 user message";

/// Extract a string representation of a message content.
fn text_of(msg: &Value) -> Option<String> {
    match msg.get("content") {
        Some(Value::String(s)) => Some(s.clone()),
        Some(Value::Array(parts)) => {
            let mut out = String::new();
            for p in parts {
                if let Some(t) = p.get("text").and_then(Value::as_str) {
                    out.push_str(t);
                } else if let Value::String(s) = p {
                    out.push_str(s);
                }
            }
            if out.is_empty() { None } else { Some(out) }
        }
        _ => None,
    }
}

/// Map a message list to (role, text) pairs.
fn roles_contents(messages: &Value) -> Vec<(Option<String>, Option<String>)> {
    messages
        .as_array()
        .map(|arr| {
            arr.iter()
                .map(|m| {
                    (
                        m.get("role").and_then(Value::as_str).map(String::from),
                        text_of(m),
                    )
                })
                .collect()
        })
        .unwrap_or_default()
}

/// Identify a request by matching a unique literal from its payload.
fn classify(record: &Value) -> String {
    let msgs = &record["payload"]["messages"];
    let joined = msgs
        .as_array()
        .map(|arr| {
            arr.iter()
                .map(|m| text_of(m).unwrap_or_default())
                .collect::<Vec<_>>()
                .join(" || ")
        })
        .unwrap_or_default();

    if joined.contains(A1_USER_A) {
        return "branch-a-turn-1".into();
    }
    if joined.contains(A0_USER_A) {
        return "branch-a-turn-0".into();
    }
    if joined.contains(B1_USER) {
        return "branch-b-turn-1".into();
    }
    if joined.contains(B0_USER) {
        return "branch-b-turn-0".into();
    }
    if joined.contains(ROOT_USER) && !joined.contains(A0_USER_A) && !joined.contains(B0_USER) {
        return "root".into();
    }
    panic!("Unclassifiable record payload: {joined:?}");
}

/// Assert a record's wire-payload messages match the expected (role, content) sequence.
/// `None` expected content means "assistant content must be non-empty".
fn assert_messages(rec: &Value, expected: &[(&str, Option<&str>)], label: &str) {
    let got = roles_contents(&rec["payload"]["messages"]);
    assert_eq!(
        got.len(),
        expected.len(),
        "{label}: expected {} messages, got {}: {got:?}",
        expected.len(),
        got.len()
    );
    for (i, ((exp_role, exp_content), (g_role, g_content))) in
        expected.iter().zip(got.iter()).enumerate()
    {
        assert_eq!(
            g_role.as_deref(),
            Some(*exp_role),
            "{label}[{i}] role: expected {exp_role:?}, got {g_role:?}"
        );
        match exp_content {
            None => {
                assert!(
                    g_content.as_ref().map(|s| !s.is_empty()).unwrap_or(false),
                    "{label}[{i}]: assistant content must be non-empty"
                );
            }
            Some(exp) => {
                assert_eq!(
                    g_content.as_deref(),
                    Some(*exp),
                    "{label}[{i}] content: expected {exp:?}, got {g_content:?}"
                );
            }
        }
    }
}

fn ns(meta: &Value, key: &str) -> i64 {
    meta[key]
        .as_i64()
        .unwrap_or_else(|| panic!("metadata.{key} missing/non-int: {meta:?}"))
}

#[tokio::test]
#[ignore = "requires distinct DAG correlation IDs for each branch"]
async fn test_full_dag_payload_merge_and_stats() {
    assert!(
        std::path::Path::new(FIXTURE).exists(),
        "fixture missing: {FIXTURE}"
    );

    let h = AIPerfHarness::new().await;
    let r = h.run_timeout(
        &format!(
            "--model Qwen3-0.6B --url {} --endpoint-type chat --input-file {FIXTURE} \
             --custom-dataset-type dag_jsonl --num-conversations 1 --concurrency 1 \
             --workers-max 2 --export-level raw --ui simple",
            h.mock.url
        ),
        300,
    );

    assert!(r.success(), "run failed: {}", r.stderr);

    let raw = r.artifacts.raw_records();
    assert_eq!(raw.len(), 5, "Expected 5 raw records, got {}", raw.len());

    let mut by_kind: std::collections::HashMap<String, Vec<&Value>> =
        std::collections::HashMap::new();
    for rec in &raw {
        by_kind.entry(classify(rec)).or_default().push(rec);
    }

    let kinds: std::collections::HashSet<String> = by_kind.keys().cloned().collect();
    let expected_kinds: std::collections::HashSet<String> = [
        "root",
        "branch-a-turn-0",
        "branch-a-turn-1",
        "branch-b-turn-0",
        "branch-b-turn-1",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    assert_eq!(kinds, expected_kinds, "Unexpected record kinds: {kinds:?}");

    let root_rec = by_kind["root"][0];
    let a0 = by_kind["branch-a-turn-0"][0];
    let a1 = by_kind["branch-a-turn-1"][0];
    let b0 = by_kind["branch-b-turn-0"][0];
    let b1 = by_kind["branch-b-turn-1"][0];

    let root_corr = &root_rec["metadata"]["x_correlation_id"];
    let branch_a_corr = &a0["metadata"]["x_correlation_id"];
    let branch_b_corr = &b0["metadata"]["x_correlation_id"];

    assert!(!root_corr.is_null());
    assert!(!branch_a_corr.is_null());
    assert!(!branch_b_corr.is_null());

    let corr_set: std::collections::HashSet<String> = [root_corr, branch_a_corr, branch_b_corr]
        .iter()
        .map(|v| v.to_string())
        .collect();
    assert_eq!(corr_set.len(), 3);

    assert_eq!(&a1["metadata"]["x_correlation_id"], branch_a_corr);
    assert_eq!(&b1["metadata"]["x_correlation_id"], branch_b_corr);

    assert!(root_rec["metadata"]["parent_correlation_id"].is_null());
    for rec in [a0, a1, b0, b1] {
        assert_eq!(&rec["metadata"]["parent_correlation_id"], root_corr);
    }

    assert_eq!(root_rec["metadata"]["agent_depth"], 0);
    for rec in [a0, a1, b0, b1] {
        assert_eq!(rec["metadata"]["agent_depth"], 1);
    }

    assert!(ns(&root_rec["metadata"], "request_end_ns") <= ns(&a0["metadata"], "request_start_ns"));
    assert!(ns(&root_rec["metadata"], "request_end_ns") <= ns(&b0["metadata"], "request_start_ns"));
    assert!(ns(&a0["metadata"], "request_end_ns") <= ns(&a1["metadata"], "request_start_ns"));
    assert!(ns(&b0["metadata"], "request_end_ns") <= ns(&b1["metadata"], "request_start_ns"));

    let sibling_skew_ns =
        (ns(&a0["metadata"], "request_start_ns") - ns(&b0["metadata"], "request_start_ns")).abs();
    assert!(sibling_skew_ns < 2_000_000_000);

    assert_messages(
        root_rec,
        &[("system", Some(ROOT_SYS)), ("user", Some(ROOT_USER))],
        "root",
    );

    assert_messages(
        a0,
        &[
            ("system", Some(ROOT_SYS)),
            ("user", Some(ROOT_USER)),
            ("assistant", None),
            ("user", Some(A0_USER_A)),
            ("user", Some(A0_USER_B)),
        ],
        "branch-a turn 0",
    );

    assert_messages(
        a1,
        &[
            ("system", Some(ROOT_SYS)),
            ("user", Some(ROOT_USER)),
            ("assistant", None),
            ("user", Some(A0_USER_A)),
            ("user", Some(A0_USER_B)),
            ("assistant", None),
            ("user", Some(A1_USER_A)),
            ("user", Some(A1_USER_B)),
        ],
        "branch-a turn 1",
    );

    assert_messages(
        b0,
        &[
            ("system", Some(ROOT_SYS)),
            ("user", Some(ROOT_USER)),
            ("assistant", None),
            ("user", Some(B0_USER)),
        ],
        "branch-b turn 0",
    );

    assert_messages(
        b1,
        &[
            ("system", Some(ROOT_SYS)),
            ("user", Some(ROOT_USER)),
            ("assistant", None),
            ("user", Some(B0_USER)),
            ("assistant", None),
            ("user", Some(B1_USER)),
        ],
        "branch-b turn 1",
    );

    let json = r.artifacts.json();
    let branch_stats = &json["branch_stats"];
    assert!(!branch_stats.is_null(), "branch_stats must exist");
    assert_eq!(branch_stats["children_spawned"], 2);
    assert_eq!(branch_stats["children_completed"], 2);
    assert_eq!(branch_stats["children_errored"], 0);

    let worker_ids: std::collections::HashSet<String> = raw
        .iter()
        .map(|rec| rec["metadata"]["worker_id"].to_string())
        .collect();
    assert_eq!(
        worker_ids.len(),
        1,
        "All 5 DAG requests must route to the same worker via sticky routing; saw workers {worker_ids:?}"
    );
}
