// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
mod common;
use common::*;

// SPAWN-mode children start with fresh context and are not sticky-routed.

use serde_json::Value;

const FIXTURE: &str =
    "/home/anthony/nvidia/projects/aiperf/ajc/rust/tests/fixtures/dag/spawn_minimal.dag.jsonl";

const ROOT_SYS: &str = "root-sys";
const ROOT_USER: &str = "root-u";
const SPAWN_SYS: &str = "spawn-sys";
const SPAWN_USER: &str = "spawn-u";

/// Extract the text of a single chat message, flattening string or list content.
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

#[tokio::test]
#[ignore = "DAG spawn label/context fields not yet emitted by Rust runner"]
async fn test_spawn_child_has_fresh_context_and_is_not_sticky_pinned() {
    assert!(
        std::path::Path::new(FIXTURE).exists(),
        "fixture missing: {FIXTURE}"
    );

    let h = AIPerfHarness::new().await;
    let r = h.run_timeout(
        &format!(
            "--model test-model --url {} --endpoint-type chat --input-file {FIXTURE} \
             --custom-dataset-type dag_jsonl --num-conversations 1 --concurrency 1 \
             --workers-max 2 --export-level raw --ui simple",
            h.mock.url
        ),
        300,
    );

    assert!(r.success(), "run failed: {}", r.stderr);

    let raw = r.artifacts.raw_records();
    assert_eq!(raw.len(), 2, "Expected 2 raw records, got {}", raw.len());

    let mut root_rec: Option<&Value> = None;
    let mut child_rec: Option<&Value> = None;
    for rec in &raw {
        let messages = &rec["payload"]["messages"];
        let first_sys = messages.get(0).and_then(text_of);
        match first_sys.as_deref() {
            Some(ROOT_SYS) => root_rec = Some(rec),
            Some(SPAWN_SYS) => child_rec = Some(rec),
            _ => {}
        }
    }
    let root_rec = root_rec.expect("root record not found");
    let child_rec = child_rec.expect("spawn-mode child record not found");

    assert_eq!(
        roles_contents(&root_rec["payload"]["messages"]),
        vec![
            (Some("system".into()), Some(ROOT_SYS.into())),
            (Some("user".into()), Some(ROOT_USER.into())),
        ]
    );

    assert_eq!(
        roles_contents(&child_rec["payload"]["messages"]),
        vec![
            (Some("system".into()), Some(SPAWN_SYS.into())),
            (Some("user".into()), Some(SPAWN_USER.into())),
        ],
        "SPAWN-mode child must start with a fresh context (no parent turn_list inherited)"
    );

    // SPAWN changes context ownership and routing, not parent linkage.
    assert!(root_rec["metadata"]["parent_correlation_id"].is_null());
    assert_eq!(
        child_rec["metadata"]["parent_correlation_id"],
        root_rec["metadata"]["x_correlation_id"]
    );

    let json = r.artifacts.json();
    let branch_stats = &json["branch_stats"];
    assert!(!branch_stats.is_null(), "branch_stats must exist");
    assert_eq!(branch_stats["children_spawned"], 1);
    assert_eq!(branch_stats["children_completed"], 1);
    assert_eq!(branch_stats["children_errored"], 0);
}
