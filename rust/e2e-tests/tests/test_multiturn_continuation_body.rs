// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The wire `messages` array of an ordinary multi-turn chat continuation turn.
//!
//! Under the default context mode (`DeltasWithoutResponses`) turn *k* sends
//! `t0 r0 t1 r1 … tk` — every earlier authored turn and every reply captured
//! from the server so far, the replies in **interior** positions. That body is
//! assembled from a plan the dataset precomputed over the authored turns alone,
//! with the captured replies spliced in at dispatch.
//!
//! Nothing else in the suite reads back a dispatched multi-turn message array.
//! The metrics do not cover it: `num_images` is established up front from the
//! dataset and the reply count, and the token counts come from the authored
//! turns and the recorded completions — so a body that dropped every reply, or
//! placed them in the wrong order, produces a run that completes successfully
//! with unchanged metrics and silently benchmarks the wrong prompt. This is
//! therefore pinned from the raw per-record request payload, which is the
//! bytes the transport actually sent.

mod common;
use common::*;

use serde_json::{Value, json};

/// Authored turns per conversation.
const TURNS: usize = 4;

/// Distinct per-turn text, so a message's position in the array identifies which
/// authored turn produced it and a reordering cannot pass.
fn user_text(turn: usize) -> String {
    format!("authored question number {turn}")
}

fn multiturn_conversation() -> Value {
    let turns: Vec<Value> = (0..TURNS)
        .map(|turn| json!({"text": user_text(turn)}))
        .collect();
    json!({"session_id": "continuation", "turns": turns})
}

/// The dispatched `messages` array of one raw record.
fn payload_messages(record: &Value) -> &Vec<Value> {
    record["payload"]["messages"]
        .as_array()
        .unwrap_or_else(|| panic!("raw record carries no dispatched messages array: {record}"))
}

#[tokio::test]
async fn continuation_turn_body_interleaves_every_captured_reply() {
    let h = AIPerfHarness::new().await;
    let input = write_jsonl(
        h.artifact_path(),
        "multiturn_continuation.jsonl",
        &[multiturn_conversation()],
    )
    .display()
    .to_string();
    let r = h.run_timeout(
        &format!(
            "--model {DEFAULT_MODEL} --url {} --endpoint-type chat --streaming \
             --input-file {input} --custom-dataset-type multi_turn \
             --num-conversations 1 --concurrency 1 --workers-max 1 --random-seed 42 \
             --output-tokens-mean 4 --export-level raw --ui simple --tokenizer builtin",
            h.mock.url
        ),
        300,
    );
    assert!(r.success(), "multi-turn continuation run failed: {}", r.stderr);

    let mut records = r.artifacts.raw_records();
    assert_eq!(
        records.len(),
        TURNS,
        "expected one raw record per authored turn, got {records:?}"
    );
    records.sort_by_key(|record| {
        record["metadata"]["turn_index"]
            .as_u64()
            .unwrap_or_else(|| panic!("raw record carries no turn index: {record}"))
    });

    for (turn, record) in records.iter().enumerate() {
        let messages = payload_messages(record);
        // `t0 r0 … t(k-1) r(k-1) tk` — one authored turn per index plus one
        // captured reply between each pair. A body that dropped the replies
        // would carry `turn + 1`.
        assert_eq!(
            messages.len(),
            2 * turn + 1,
            "turn {turn} did not resend every authored turn and captured reply: {messages:?}"
        );
        for (index, message) in messages.iter().enumerate() {
            if index % 2 == 0 {
                assert_eq!(
                    message["role"], "user",
                    "turn {turn} message {index} should be the authored turn {}: {messages:?}",
                    index / 2
                );
                assert_eq!(
                    message["content"],
                    Value::String(user_text(index / 2)),
                    "turn {turn} message {index} carries the wrong authored turn: {messages:?}"
                );
            } else {
                // The reply's text is server-generated, so only its shape and
                // its position are contractual here.
                assert_eq!(
                    message["role"], "assistant",
                    "turn {turn} message {index} should be the reply captured after \
                     authored turn {}: {messages:?}",
                    index / 2
                );
                assert!(
                    message["content"].is_string() || message["content"].is_array(),
                    "turn {turn} message {index} carries no reply content: {messages:?}"
                );
            }
        }
    }
}
