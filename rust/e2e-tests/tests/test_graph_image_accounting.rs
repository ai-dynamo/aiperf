// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `num_images` accounting across both graph-node dispatch paths.
//!
//! Graph dispatch establishes the wire image count from the node's authored
//! prompt program when it can, and otherwise leaves it unknown so the transport
//! derives it by parsing the serialized body. Both paths are exercised here
//! against real per-record output, because a wrongly-claimed zero is filtered
//! out rather than reported and would otherwise vanish silently from a run that
//! completes successfully:
//!
//! - A WEKA-trace graph reconstructs its prompts from text-only segments, so
//!   every node reports a known zero and no image metric is recorded.
//! - A `dag_jsonl` graph retains an authored message array verbatim, and a
//!   `conditional_graph` node splices a channel a trace's `initial_state` seeds
//!   verbatim. Both can carry `image_url` parts the graph layer never inspects,
//!   and both must still be counted through the parse fallback.

mod common;
use common::*;

use serde_json::{Value, json};

/// One 64-token block per authored hash id, so `in` divides evenly.
const BLOCK: u64 = 64;

/// A `dag_jsonl` session whose single turn authors `images` `image_url` content
/// parts alongside its text. The graph layer keeps this array as opaque raw
/// wire, so the count is recoverable only from the serialized body.
fn image_session(images: usize) -> Value {
    let mut content = vec![json!({"type": "text", "text": "describe these"})];
    for index in 0..images {
        content.push(json!({
            "type": "image_url",
            "image_url": {"url": format!("https://example.invalid/{index}.png")},
        }));
    }
    json!({
        "session_id": "images",
        "turns": [{
            "model": DEFAULT_MODEL,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": 4,
        }],
    })
}

/// A single-node `conditional_graph` whose prompt is exactly one channel
/// reference, over a trace whose `initial_state` seeds that channel with
/// `images` `image_url` parts.
///
/// `@messages` compiles to exactly `[PromptItem::Splice]` — no raw-message item
/// — and `channel_value` seeds a messages-typed channel from the authored array
/// verbatim, so this is the shape that carries images with nothing on the graph
/// path ever inspecting them.
fn spliced_image_graph(images: usize) -> String {
    let mut parts = vec![r#"{"type": "text", "text": "describe these"}"#.to_string()];
    for index in 0..images {
        parts.push(format!(
            r#"{{"type": "image_url", "image_url": {{"url": "https://example.invalid/{index}.png"}}}}"#
        ));
    }
    format!(
        "graph:\n  \
           state:\n    messages: {{type: messages, reducer: add_messages}}\n    \
           reply: {{type: text}}\n  \
           nodes:\n    ask:\n      node_type: llm\n      prompt: [\"@messages\"]\n      \
             output: reply\n      streaming: false\n      max_tokens: 4\n  \
           edges:\n    - {{source: START, target: ask}}\n\
         traces:\n  - id: t-img\n    initial_state:\n      \
             messages: [{{\"role\": \"user\", \"content\": [{}]}}]\n",
        parts.join(", ")
    )
}

/// A minimal text-only WEKA trace: the loader rebuilds its prompt from text
/// segments, so the graph node's prompt program has no raw-message item.
fn text_only_weka_trace() -> (tempfile::TempDir, String) {
    let trace = json!({
        "id": "text-only",
        "models": [DEFAULT_MODEL],
        "block_size": BLOCK,
        "hash_id_scope": "global",
        "tool_tokens": 0,
        "system_tokens": 0,
        "requests": [{
            "t": 0.0,
            "type": "n",
            "model": DEFAULT_MODEL,
            "in": BLOCK * 2,
            "out": 4,
            "hash_ids": [1, 2],
            "input_types": ["text"],
            "output_types": ["text"],
            "stop": "end",
            "api_time": 0.01,
            "think_time": 0.0,
        }],
    });
    let dir = tempfile::tempdir().expect("weka fixture tempdir");
    std::fs::write(
        dir.path().join("trace-000.json"),
        serde_json::to_vec(&trace).expect("serialize weka trace"),
    )
    .expect("write weka trace");
    let path = dir.path().display().to_string();
    (dir, path)
}

/// Per-record `num_images` values, treating an absent metric as "no image count
/// recorded" (the runtime drops a zero count rather than emitting it).
fn record_image_counts(records: &[Value]) -> Vec<Option<f64>> {
    records
        .iter()
        .map(|record| record["metrics"]["num_images"]["value"].as_f64())
        .collect()
}

#[tokio::test]
async fn a_text_only_graph_run_records_no_image_metric() {
    let h = AIPerfHarness::new().await;
    let (_fixture, input) = text_only_weka_trace();
    let r = h.run_timeout(
        &format!(
            "--model {DEFAULT_MODEL} --url {} --endpoint-type chat --input-file {input} \
             --custom-dataset-type weka_trace --num-conversations 2 --concurrency 1 \
             --export-level raw --ui simple",
            h.mock.url
        ),
        300,
    );
    assert!(r.success(), "weka graph run failed: {}", r.stderr);

    let records = r.artifacts.jsonl();
    assert!(!records.is_empty(), "run produced no records");
    assert_eq!(
        record_image_counts(&records),
        vec![None; records.len()],
        "a text-only graph node must not report any images"
    );
    assert!(
        r.artifacts.json()["num_images"].is_null(),
        "a text-only graph run must not emit a num_images summary"
    );
}

#[tokio::test]
async fn authored_raw_messages_carrying_images_are_still_counted() {
    const IMAGES: usize = 2;

    let h = AIPerfHarness::new().await;
    let input = write_jsonl(
        h.artifact_path(),
        "images.dag.jsonl",
        &[image_session(IMAGES)],
    )
    .display()
    .to_string();
    let r = h.run_timeout(
        &format!(
            "--model {DEFAULT_MODEL} --url {} --endpoint-type chat --input-file {input} \
             --custom-dataset-type dag_jsonl --num-conversations 1 --concurrency 1 \
             --export-level raw --ui simple",
            h.mock.url
        ),
        300,
    );
    assert!(r.success(), "dag graph run failed: {}", r.stderr);

    let records = r.artifacts.jsonl();
    assert_eq!(records.len(), 1, "expected one record, got {records:?}");
    assert_eq!(
        record_image_counts(&records),
        vec![Some(IMAGES as f64)],
        "authored image parts ride in opaque raw messages, so the parse fallback \
         is the only thing that can count them"
    );
    assert_eq!(
        r.artifacts.json()["num_images"]["avg"].as_f64(),
        Some(IMAGES as f64),
        "the summary must agree with the per-record counts"
    );
}

/// The silent-zero case: a node whose whole prompt is one channel reference, over
/// a trace that seeds that channel with images. Nothing on the graph path
/// inspects the seeded wires, so claiming a known zero here would drop
/// `num_images` entirely from a run that reports success.
#[tokio::test]
async fn a_spliced_channel_seeded_with_images_is_still_counted() {
    const IMAGES: usize = 3;

    let h = AIPerfHarness::new().await;
    let input = write_text(
        h.artifact_path(),
        "spliced_images.yaml",
        &spliced_image_graph(IMAGES),
    )
    .display()
    .to_string();
    let r = h.run_timeout(
        &format!(
            "--model {DEFAULT_MODEL} --url {} --endpoint-type chat --input-file {input} \
             --custom-dataset-type conditional_graph --num-conversations 1 --concurrency 1 \
             --export-level raw --ui simple",
            h.mock.url
        ),
        300,
    );
    assert!(r.success(), "conditional graph run failed: {}", r.stderr);

    let records = r.artifacts.jsonl();
    assert_eq!(records.len(), 1, "expected one record, got {records:?}");
    assert_eq!(
        record_image_counts(&records),
        vec![Some(IMAGES as f64)],
        "a splice resolves authored `initial_state` wires verbatim, so the node \
         cannot claim a known zero"
    );
    assert_eq!(
        r.artifacts.json()["num_images"]["avg"].as_f64(),
        Some(IMAGES as f64),
        "the summary must agree with the per-record counts"
    );
}
