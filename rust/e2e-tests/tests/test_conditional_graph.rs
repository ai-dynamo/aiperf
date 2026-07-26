// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
mod common;
use common::*;

// End-to-end coverage of the authored `conditional_graph` format through the
// real `aiperf` binary: model-independent branch resolution, taken-subgraph
// pruning, and recorded replay nodes folded into channel state. Three pinned
// traces exercise the three branch combinations of a shopping-assistant diamond.

use std::collections::HashMap;

use serde_json::Value;

const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/conditional/conditional_shopping.yaml"
);

/// The distinct uppercase keyword each LLM node stamps into its system prompt.
const NODE_KEYWORDS: &[(&str, &str)] = &[
    ("route", "ROUTE"),
    ("plan", "PLAN"),
    ("brandmap", "BRANDMAP"),
    ("summarize", "SUMMARIZE"),
    ("safety", "SAFETY"),
    ("redirect", "REDIRECT"),
];

/// Join every message's text content in a record's dispatched payload.
fn payload_text(record: &Value) -> String {
    record["payload"]["messages"]
        .as_array()
        .map(|messages| {
            messages
                .iter()
                .map(|message| match message.get("content") {
                    Some(Value::String(text)) => text.clone(),
                    Some(Value::Array(parts)) => parts
                        .iter()
                        .filter_map(|part| {
                            part.get("text")
                                .and_then(Value::as_str)
                                .map(str::to_string)
                                .or_else(|| part.as_str().map(str::to_string))
                        })
                        .collect::<Vec<_>>()
                        .join(" "),
                    _ => String::new(),
                })
                .collect::<Vec<_>>()
                .join(" || ")
        })
        .unwrap_or_default()
}

/// Which authored node dispatched this record, by its system-prompt keyword.
fn classify_node(record: &Value) -> String {
    let text = payload_text(record);
    let matched: Vec<&str> = NODE_KEYWORDS
        .iter()
        .filter(|(_, keyword)| text.contains(keyword))
        .map(|(node, _)| *node)
        .collect();
    assert_eq!(
        matched.len(),
        1,
        "record must map to exactly one node; matched {matched:?} in payload {text:?}"
    );
    matched[0].to_string()
}

#[tokio::test]
async fn conditional_graph_resolves_branches_and_folds_replay_end_to_end() {
    assert!(
        std::path::Path::new(FIXTURE).exists(),
        "fixture missing: {FIXTURE}"
    );

    let harness = AIPerfHarness::new().await;
    let result = harness.run_timeout(
        &format!(
            "--model Qwen3-0.6B --url {} --endpoint-type chat --input-file {FIXTURE} \
             --custom-dataset-type conditional_graph --num-conversations 3 --concurrency 1 \
             --workers-max 2 --export-level raw --ui simple",
            harness.mock.url
        ),
        300,
    );
    assert!(result.success(), "run failed: {}", result.stderr);

    let raw = result.artifacts.raw_records();

    // Count dispatches per node across all three traces.
    let mut counts: HashMap<String, usize> = HashMap::new();
    for record in &raw {
        *counts.entry(classify_node(record)).or_default() += 1;
    }

    // shopping fires {route, plan, brandmap, summarize, safety}; non_shopping
    // fires {route, safety}; unsafe fires {route, plan, brandmap, summarize,
    // redirect, safety}. Summed: route x3, safety x3, plan x2, brandmap x2,
    // summarize x2, redirect x1 = 13 dispatches.
    let expected: HashMap<String, usize> = [
        ("route", 3),
        ("safety", 3),
        ("plan", 2),
        ("brandmap", 2),
        ("summarize", 2),
        ("redirect", 1),
    ]
    .into_iter()
    .map(|(node, count)| (node.to_string(), count))
    .collect();
    assert_eq!(counts, expected, "per-node dispatch counts");

    assert_eq!(raw.len(), 13, "total dispatched requests");

    // The replay nodes never dispatch: no record carries a tool_exec/preprocess
    // marker, and every record is a well-formed chat request with a model.
    for record in &raw {
        assert_eq!(
            record["metadata"]["error"],
            Value::Null,
            "no record should error: {record:?}"
        );
        assert!(
            record["payload"]["messages"].is_array(),
            "every dispatched record must carry a chat message array"
        );
    }
}
