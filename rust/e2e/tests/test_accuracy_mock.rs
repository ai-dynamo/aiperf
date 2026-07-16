// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! End-to-end accuracy-dataset tests: drive `aiperf profile` against a mock
//! server loaded with a ground-truth dataset and verify the raw per-record
//! output carries the expected answers.
//!
//! The dataset is a single combined JSONL file both sides read: `aiperf`
//! consumes it as a `single_turn` input file (the `text` field becomes the user
//! prompt), and the mock consumes it as its `--accuracy-dataset` (the same
//! `text` plus `ground_truth`). Because ground truth never crosses the wire in
//! AIPerf, the mock loads it independently and keys on the prompt — this test
//! proves that whole path deterministically.

mod common;
use common::*;

use aiperf_mock_server::accuracy::AccuracyFormat;
use aiperf_mock_server::config::MockServerConfig;
use serde_json::{Value, json};

/// Reconstruct the streamed assistant `content` from one raw record by
/// concatenating every `choices[0].delta.content` token across its SSE frames.
fn record_content(record: &Value) -> String {
    let mut out = String::new();
    if let Some(responses) = record.get("responses").and_then(Value::as_array) {
        for resp in responses {
            let Some(packets) = resp.get("packets").and_then(Value::as_array) else {
                continue;
            };
            for packet in packets {
                if packet.get("name").and_then(Value::as_str) != Some("data") {
                    continue;
                }
                let Some(raw) = packet.get("value").and_then(Value::as_str) else {
                    continue;
                };
                let trimmed = raw.trim();
                if trimmed == "[DONE]" {
                    continue;
                }
                if let Ok(obj) = serde_json::from_str::<Value>(trimmed) {
                    if let Some(c) = obj
                        .pointer("/choices/0/delta/content")
                        .and_then(Value::as_str)
                    {
                        out.push_str(c);
                    }
                }
            }
        }
    }
    out
}

/// Reconstruct the streamed `reasoning_content` from one raw record.
fn record_reasoning(record: &Value) -> String {
    let mut out = String::new();
    if let Some(responses) = record.get("responses").and_then(Value::as_array) {
        for resp in responses {
            let Some(packets) = resp.get("packets").and_then(Value::as_array) else {
                continue;
            };
            for packet in packets {
                if packet.get("name").and_then(Value::as_str) != Some("data") {
                    continue;
                }
                let Some(raw) = packet.get("value").and_then(Value::as_str) else {
                    continue;
                };
                if let Ok(obj) = serde_json::from_str::<Value>(raw.trim()) {
                    if let Some(c) = obj
                        .pointer("/choices/0/delta/reasoning_content")
                        .and_then(Value::as_str)
                    {
                        out.push_str(c);
                    }
                }
            }
        }
    }
    out
}

/// Write a combined `{text, ground_truth, task}` JSONL dataset into `dir`,
/// returning its path. Every gold answer is `B` so a record's correctness is
/// readable from its content alone, independent of which prompt it used.
fn write_accuracy_dataset(dir: &std::path::Path, n: usize) -> std::path::PathBuf {
    let records: Vec<Value> = (0..n)
        .map(|i| {
            json!({
                "text": format!("Question {i}: which option is correct, A, B, C, or D?"),
                "ground_truth": "B",
                "task": "demo",
            })
        })
        .collect();
    write_jsonl(dir, "accuracy.jsonl", &records)
}

fn accuracy_cfg(dataset: &std::path::Path) -> MockServerConfig {
    MockServerConfig {
        fast: true,
        no_tokenizer: true,
        random_seed: Some(7),
        accuracy_dataset: Some(dataset.to_string_lossy().into_owned()),
        accuracy_format: AccuracyFormat::Mmlu,
        ..MockServerConfig::default()
    }
}

/// Ground truth is wired through: at correct-rate 1.0 every raw record's
/// streamed content is exactly the grader-formatted correct answer.
#[tokio::test]
async fn accuracy_correct_ground_truth_in_raw_records() {
    let ds_dir = tempfile::TempDir::new().unwrap();
    let dataset = write_accuracy_dataset(ds_dir.path(), 6);
    let mut cfg = accuracy_cfg(&dataset);
    cfg.accuracy_correct_rate = 1.0;

    let h = AIPerfHarness::new_with(cfg).await;
    let r = h.run(&format!(
        "--model gpt-4 --url {} --endpoint-type chat --streaming \
         --input-file {} --custom-dataset-type single_turn \
         --request-count 6 --concurrency 2 --workers-max 1 \
         --random-seed 7 --export-level raw --ui simple",
        h.mock.url,
        dataset.display(),
    ));
    assert!(r.success(), "stderr: {}", r.stderr);

    let records = r.artifacts.raw_records();
    assert_eq!(records.len(), 6, "expected 6 raw records");
    for rec in &records {
        assert_eq!(
            record_content(rec),
            "The answer is (B)",
            "record content should be the correct grader-formatted answer"
        );
    }
}

/// The seeded correct-rate is honored end-to-end: at 0.5, roughly half the
/// records carry the correct `(B)` answer and the rest a different letter — and
/// every record is a well-formed multiple-choice answer.
#[tokio::test]
async fn accuracy_seeded_correct_rate_split() {
    let ds_dir = tempfile::TempDir::new().unwrap();
    let n = 24;
    let dataset = write_accuracy_dataset(ds_dir.path(), n);
    let mut cfg = accuracy_cfg(&dataset);
    cfg.accuracy_correct_rate = 0.5;

    let h = AIPerfHarness::new_with(cfg).await;
    let r = h.run(&format!(
        "--model gpt-4 --url {} --endpoint-type chat --streaming \
         --input-file {} --custom-dataset-type single_turn \
         --request-count {n} --concurrency 4 --workers-max 1 \
         --random-seed 7 --export-level raw --ui simple",
        h.mock.url,
        dataset.display(),
    ));
    assert!(r.success(), "stderr: {}", r.stderr);

    let records = r.artifacts.raw_records();
    assert_eq!(records.len(), n);
    let mut correct = 0usize;
    for rec in &records {
        let content = record_content(rec);
        assert!(
            content.starts_with("The answer is ("),
            "malformed answer: {content:?}"
        );
        if content == "The answer is (B)" {
            correct += 1;
        }
    }
    // Binomial(24, 0.5): mean 12, sd ~2.4 — allow a wide band.
    assert!(
        (5..=19).contains(&correct),
        "correct={correct}/{n} outside expected band"
    );
}

/// CoT mode streams reasoning in the separate `reasoning_content` field while
/// the answer stays clean in `content` (reasoning-model shape).
#[tokio::test]
async fn accuracy_cot_streams_reasoning_separately() {
    let ds_dir = tempfile::TempDir::new().unwrap();
    let dataset = write_accuracy_dataset(ds_dir.path(), 6);
    let mut cfg = accuracy_cfg(&dataset);
    cfg.accuracy_correct_rate = 1.0;
    cfg.accuracy_cot_rate = 1.0;
    cfg.accuracy_reasoning_field = true;

    let h = AIPerfHarness::new_with(cfg).await;
    let r = h.run(&format!(
        "--model gpt-4 --url {} --endpoint-type chat --streaming \
         --input-file {} --custom-dataset-type single_turn \
         --request-count 6 --concurrency 2 --workers-max 1 \
         --random-seed 7 --export-level raw --ui simple",
        h.mock.url,
        dataset.display(),
    ));
    assert!(r.success(), "stderr: {}", r.stderr);

    let records = r.artifacts.raw_records();
    assert_eq!(records.len(), 6);
    for rec in &records {
        assert_eq!(record_content(rec), "The answer is (B)");
        let reasoning = record_reasoning(rec);
        assert!(
            reasoning.contains("The answer is (B)"),
            "reasoning should carry the answer, got {reasoning:?}"
        );
    }
}

/// The mock's live `/accuracy` tally matches the actual served run: `matched`
/// equals the raw-record count and `correct` equals the number of records that
/// carry the correct answer. This is the oracle a user compares against what
/// AIPerf's own grader reports.
#[tokio::test]
async fn accuracy_live_endpoint_matches_raw_records() {
    let ds_dir = tempfile::TempDir::new().unwrap();
    let n = 24;
    let dataset = write_accuracy_dataset(ds_dir.path(), n);
    let mut cfg = accuracy_cfg(&dataset);
    cfg.accuracy_correct_rate = 0.5;

    let h = AIPerfHarness::new_with(cfg).await;
    let url = h.mock.url.clone();
    let r = h.run(&format!(
        "--model gpt-4 --url {url} --endpoint-type chat --streaming \
         --input-file {} --custom-dataset-type single_turn \
         --request-count {n} --concurrency 4 --workers-max 1 \
         --random-seed 7 --export-level raw --ui simple",
        dataset.display(),
    ));
    assert!(r.success(), "stderr: {}", r.stderr);

    let records = r.artifacts.raw_records();
    let raw_correct = records
        .iter()
        .filter(|rec| record_content(rec) == "The answer is (B)")
        .count();

    let acc: serde_json::Value = reqwest::Client::builder()
        .no_proxy()
        .build()
        .unwrap()
        .get(format!("{url}/accuracy"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();

    assert_eq!(acc["enabled"], true);
    assert_eq!(
        acc["matched"].as_u64().unwrap() as usize,
        records.len(),
        "live matched should equal raw-record count"
    );
    assert_eq!(
        acc["correct"].as_u64().unwrap() as usize,
        raw_correct,
        "live correct should equal raw records carrying the correct answer"
    );
}

/// Adversarial mode emits the parser-choking shapes (github #1010 `object:null`
/// frame, #1136 reasoning-only content, boxed/whitespace/case mangling). The
/// whole run must still COMPLETE — a brittle parser would crash a worker.
#[tokio::test]
async fn accuracy_adversarial_responses_do_not_crash_the_run() {
    let ds_dir = tempfile::TempDir::new().unwrap();
    let n = 24;
    let dataset = write_accuracy_dataset(ds_dir.path(), n);
    let mut cfg = accuracy_cfg(&dataset);
    cfg.fast = false; // exercise the real streaming loop (null-object frame path)
    cfg.ttft = 0.0;
    cfg.itl = 0.0;
    cfg.accuracy_correct_rate = 1.0;
    cfg.accuracy_adversarial_rate = 1.0;

    let h = AIPerfHarness::new_with(cfg).await;
    let r = h.run(&format!(
        "--model gpt-4 --url {} --endpoint-type chat --streaming \
         --input-file {} --custom-dataset-type single_turn \
         --request-count {n} --concurrency 4 --workers-max 1 \
         --random-seed 7 --export-level raw --ui simple",
        h.mock.url,
        dataset.display(),
    ));
    // The core assertion: the run survives every adversarial shape.
    assert!(
        r.success(),
        "adversarial run crashed (parser not robust): {}",
        r.stderr
    );
    let records = r.artifacts.raw_records();
    assert_eq!(records.len(), n, "every request should still produce a record");
}
