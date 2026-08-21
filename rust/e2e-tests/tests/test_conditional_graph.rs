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

fn timestamp_ns(record: &Value, field: &str) -> i64 {
    record["metadata"][field]
        .as_i64()
        .unwrap_or_else(|| panic!("record missing integer metadata.{field}: {record:?}"))
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

    // Concurrency one keeps the two shopping traces sequential. Within each,
    // replay folding must retain tool_exec's 80 ms and preprocess's 5 ms as
    // completion-anchored gaps between the surrounding dispatched requests.
    let mut by_node: HashMap<String, Vec<&Value>> = HashMap::new();
    for record in &raw {
        by_node
            .entry(classify_node(record))
            .or_default()
            .push(record);
    }
    for records in by_node.values_mut() {
        records.sort_by_key(|record| timestamp_ns(record, "request_start_ns"));
    }
    assert_eq!(by_node["plan"].len(), 2, "timed tool predecessors");
    assert_eq!(by_node["brandmap"].len(), 2, "timed tool successors");
    assert_eq!(by_node["summarize"].len(), 2, "timed preprocess successors");
    let tool_gaps = by_node["plan"]
        .iter()
        .zip(&by_node["brandmap"])
        .map(|(plan, brandmap)| {
            timestamp_ns(brandmap, "request_start_ns") - timestamp_ns(plan, "request_end_ns")
        })
        .collect::<Vec<_>>();
    let preprocess_gaps = by_node["brandmap"]
        .iter()
        .zip(&by_node["summarize"])
        .map(|(brandmap, summarize)| {
            timestamp_ns(summarize, "request_start_ns") - timestamp_ns(brandmap, "request_end_ns")
        })
        .collect::<Vec<_>>();
    for &folded_gap_ns in &tool_gaps {
        assert!(
            (80_000_000..1_000_000_000).contains(&folded_gap_ns),
            "tool_exec's folded delay was {folded_gap_ns} ns, expected [80 ms, 1 s)"
        );
    }
    for &folded_gap_ns in &preprocess_gaps {
        assert!(
            (5_000_000..1_000_000_000).contains(&folded_gap_ns),
            "preprocess's folded delay was {folded_gap_ns} ns, expected [5 ms, 1 s)"
        );
    }
    for (&tool_gap_ns, &preprocess_gap_ns) in tool_gaps.iter().zip(&preprocess_gaps) {
        let authored_delta_ns = tool_gap_ns - preprocess_gap_ns;
        assert!(
            (50_000_000..200_000_000).contains(&authored_delta_ns),
            "folded 80 ms and 5 ms delays differed by {authored_delta_ns} ns; an absent fold would remove the authored 75 ms separation"
        );
    }

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
