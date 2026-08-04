// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `num_images` accounting across the turns of a live multi-turn conversation.
//!
//! A continuation turn's wire body is not just the turn the user authored: the
//! delta context modes resend every earlier turn, so an image authored once in
//! turn 0 rides on the wire again in turns 1..N. Whatever establishes the
//! dispatch-time image count must therefore account for accumulated history, not
//! only the current turn's own content.
//!
//! This pins the observable contract from raw per-record output because an
//! under-count is silent: the runtime drops a zero rather than reporting it, so a
//! turn that wrongly claims zero simply loses its `num_images` from a run that
//! completes successfully.

mod common;
use common::*;

use serde_json::{Value, json};

/// Authored turns per conversation. Turn 0 carries the image; 1..3 are text.
const TURNS: usize = 4;

/// One conversation: an image plus text in turn 0, text-only continuations after.
fn image_first_conversation() -> Value {
    let mut turns = vec![json!({
        "text": "describe this picture",
        "images": ["https://example.invalid/first.png"],
    })];
    for turn in 1..TURNS {
        turns.push(json!({"text": format!("follow-up question {turn}")}));
    }
    json!({"session_id": "img-first", "turns": turns})
}

/// Per-record `(turn_index, num_images)`, treating an absent metric as "no image
/// count recorded" (the runtime drops a zero count rather than emitting it).
fn turn_image_counts(records: &[Value]) -> Vec<(u64, Option<f64>)> {
    let mut out: Vec<(u64, Option<f64>)> = records
        .iter()
        .map(|record| {
            (
                record["metadata"]["turn_index"]
                    .as_u64()
                    .expect("record carries a turn index"),
                record["metrics"]["num_images"]["value"].as_f64(),
            )
        })
        .collect();
    out.sort_by_key(|(turn_index, _)| *turn_index);
    out
}

/// The default context mode (`DeltasWithoutResponses`) resends turns `0..=k` plus
/// the captured assistant replies, so turn 0's single image is on the wire for
/// every one of the four dispatches and every record must report exactly one.
#[tokio::test]
async fn accumulated_history_images_are_counted_on_every_continuation_turn() {
    let h = AIPerfHarness::new().await;
    let input = write_jsonl(
        h.artifact_path(),
        "multiturn_images.jsonl",
        &[image_first_conversation()],
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
    assert!(r.success(), "multi-turn image run failed: {}", r.stderr);

    let records = r.artifacts.jsonl();
    assert_eq!(
        records.len(),
        TURNS,
        "expected one record per authored turn, got {records:?}"
    );
    assert_eq!(
        turn_image_counts(&records),
        (0..TURNS as u64).map(|turn| (turn, Some(1.0))).collect::<Vec<_>>(),
        "turn 0's image is resent as accumulated history, so every continuation \
         turn must report it too"
    );
    assert_eq!(
        r.artifacts.json()["num_images"]["avg"].as_f64(),
        Some(1.0),
        "the summary must agree with the per-record counts"
    );
}

/// A text-only conversation must report no image metric at all, on every turn.
/// This is the counterpart gate: it fails if a change starts claiming images that
/// are not on the wire.
#[tokio::test]
async fn a_text_only_multi_turn_run_records_no_image_metric() {
    let h = AIPerfHarness::new().await;
    let turns: Vec<Value> = (0..TURNS)
        .map(|turn| json!({"text": format!("plain question {turn}")}))
        .collect();
    let input = write_jsonl(
        h.artifact_path(),
        "multiturn_text.jsonl",
        &[json!({"session_id": "text-only", "turns": turns})],
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
    assert!(r.success(), "text-only multi-turn run failed: {}", r.stderr);

    let records = r.artifacts.jsonl();
    assert_eq!(records.len(), TURNS, "expected one record per authored turn");
    assert_eq!(
        turn_image_counts(&records),
        (0..TURNS as u64).map(|turn| (turn, None)).collect::<Vec<_>>(),
        "a text-only conversation must not report any images"
    );
    assert!(
        r.artifacts.json()["num_images"].is_null(),
        "a text-only run must not emit a num_images summary"
    );
}
