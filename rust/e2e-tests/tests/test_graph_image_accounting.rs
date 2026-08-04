// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `num_images` accounting across both graph-node dispatch paths.
//!
//! Graph dispatch establishes the wire image count from the node's authored
//! prompt program when it can, and otherwise leaves it unknown so the transport
//! derives it by parsing the serialized body. The two paths are exercised here
//! against real per-record output because they are byte-identical in aggregate
//! and a regression in either is otherwise silent:
//!
//! - A WEKA-trace graph reconstructs its prompts from text-only segments, so
//!   every node reports a known zero and no image metric is recorded.
//! - A `dag_jsonl` graph retains an authored message array verbatim, which can
//!   carry `image_url` parts the graph layer never inspects. Those must still be
//!   counted, which only the parse fallback can do.

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

/// Per-record `num_images` values, treating an absent field as "no image metric
/// recorded" (the runtime drops a zero count rather than emitting it).
fn record_image_counts(records: &[Value]) -> Vec<Option<u64>> {
    records
        .iter()
        .map(|record| record["num_images"].as_u64())
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

    let records = r.artifacts.raw_records();
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
    let input = write_jsonl(h.artifact_path(), "images.dag.jsonl", &[image_session(IMAGES)])
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

    let records = r.artifacts.raw_records();
    assert_eq!(records.len(), 1, "expected one record, got {records:?}");
    assert_eq!(
        record_image_counts(&records),
        vec![Some(IMAGES as u64)],
        "authored image parts ride in opaque raw messages, so the parse fallback \
         is the only thing that can count them"
    );
}
