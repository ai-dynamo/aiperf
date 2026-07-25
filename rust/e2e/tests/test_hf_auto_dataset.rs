// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Offline, per-record end-to-end coverage for the source-agnostic `hf` dataset
//! format (`HfAutoDatasetLoader` + `HfAutoComposer`).
//!
//! The `hf` format auto-detects prompt/completion columns from arbitrary rows.
//! Because `HfAutoDatasetLoader::load` routes a local-file source through the
//! shared `jsonl_or_json_rows` reader, selecting `--custom-dataset-type hf` on a
//! local JSONL file exercises the full compose -> request -> record path with NO
//! network.
//!
//! Row layout is inferred from the FIRST row and applied to the whole
//! dataset, so the two detection paths are exercised as two separate homogeneous
//! runs rather than one mixed file (a mixed file would infer a flat prompt from
//! row 0 and silently drop the chat row):
//!   * a `prompt`/`completion` text row  -> `RowLayout::Prompt`
//!   * a `conversation` chat row (first user message -> prompt, first assistant
//!     message -> completion) -> `RowLayout::Messages`
//!
//! Per record we assert:
//!   * `input_sequence_length` == the tokenized prompt (distinct per fixture:
//!     15 for the text prompt, 14 for the chat prompt under `cl100k_base`,
//!     proving the prompt was composed from the right column);
//!   * `output_sequence_length` == the completion-derived output length (9, a
//!     small value proving the completion column drives it, not the 128-token
//!     default);
//!   * the wire response is a streamed HTTP 200 carrying the pinned model with
//!     no error.
//!
//! ISL/OSL are pinned constants: tokenizer, prompts, and completions are fixed
//! and the deterministic mock generates exactly the composer's derived output
//! cap, so the tokenized counts are byte-stable.

mod common;
use common::*;

use serde_json::{Value, json};

/// Pinned tokenizer so the tokenized ISL/OSL below are byte-stable.
const TOKENIZER: &str = "cl100k_base";
const MODEL: &str = "gpt-4";

const TEXT_PROMPT: &str =
    "Explain how a four-stroke internal combustion engine turns fuel into motion.";
const TEXT_COMPLETION: &str = "Intake, compression, power, exhaust.";
const TEXT_ISL: i64 = 15;
const TEXT_OSL: i64 = 9;

const CHAT_PROMPT: &str = "Summarize the plot of Romeo and Juliet in two sentences please.";
const CHAT_COMPLETION: &str = "Two young lovers from feuding families die.";
const CHAT_ISL: i64 = 14;
const CHAT_OSL: i64 = 9;

/// The single profiling processed record's ISL/OSL, verifying exactly one record
/// was produced.
fn sole_isl_osl(r: &RunResult) -> (i64, i64) {
    let records: Vec<Value> = r
        .artifacts
        .jsonl()
        .into_iter()
        .filter(|rec| rec["metadata"]["benchmark_phase"] == "profiling")
        .collect();
    assert_eq!(
        records.len(),
        1,
        "expected exactly one processed profiling record, got {}",
        records.len()
    );
    let rec = &records[0];
    assert_eq!(rec["error"], Value::Null, "record carried an error: {rec}");
    let isl = rec["metrics"]["input_sequence_length"]["value"]
        .as_f64()
        .expect("input_sequence_length") as i64;
    let osl = rec["metrics"]["output_sequence_length"]["value"]
        .as_f64()
        .expect("output_sequence_length") as i64;
    (isl, osl)
}

/// Assert the sole raw record is a streamed HTTP 200 carrying `MODEL`. Uses the
/// shared `extract_timing` DATA surface (status / content-chunk count / model)
/// without asserting wall-clock timing, so the check is robust in a sandboxed /
/// fast-forwarded-timer environment.
fn assert_streamed_ok(r: &RunResult) {
    let raw = r.artifacts.raw_records();
    assert_eq!(
        raw.len(),
        1,
        "expected exactly one raw record (did --export-level raw run?), got {}",
        raw.len()
    );
    let timing = extract_timing(&raw[0]);
    assert_eq!(timing.status, Some(200), "raw record status: {}", raw[0]);
    assert!(
        timing.osl >= 1,
        "raw record has no streamed content (generated-token) chunks: {}",
        raw[0]
    );
    assert_eq!(
        timing.model.as_deref(),
        Some(MODEL),
        "raw record model: {}",
        raw[0]
    );
}

/// Run one homogeneous fixture through the `hf` file route and return the run.
fn run_hf(h: &AIPerfHarness, fixture: &std::path::Path) -> RunResult {
    let r = h.run(&format!(
        "--model {MODEL} --url {} --endpoint-type chat --streaming \
         --input-file {} --custom-dataset-type hf --tokenizer {TOKENIZER} \
         --num-conversations 1 --request-count 1 --concurrency 1 --workers-max 1 \
         --export-level raw --ui simple",
        h.mock.url,
        fixture.display(),
    ));
    assert!(r.success(), "hf profile run failed:\nstderr: {}", r.stderr);
    r
}

#[tokio::test]
async fn hf_auto_detect_text_column_over_local_jsonl() {
    let dir = tempfile::TempDir::new().unwrap();
    let fixture = write_jsonl(
        dir.path(),
        "text.jsonl",
        &[json!({ "prompt": TEXT_PROMPT, "completion": TEXT_COMPLETION })],
    );

    // Deterministic mock (fixed TTFT/ITL, zero jitter, analytic scheduling) so the
    // generated content-token count equals the composer's derived output cap.
    let h = AIPerfHarness::new_with(tuned_mock_config(20.0, 5.0)).await;
    let r = run_hf(&h, &fixture);

    let (isl, osl) = sole_isl_osl(&r);
    assert_eq!(isl, TEXT_ISL, "text-column ISL should match the tokenized prompt");
    assert_eq!(
        osl, TEXT_OSL,
        "text-column OSL should be the completion-derived length"
    );
    assert_streamed_ok(&r);
}

#[tokio::test]
async fn hf_auto_detect_chat_column_over_local_jsonl() {
    let dir = tempfile::TempDir::new().unwrap();
    let fixture = write_jsonl(
        dir.path(),
        "chat.jsonl",
        &[json!({
            "conversation": [
                {"role": "user", "content": CHAT_PROMPT},
                {"role": "assistant", "content": CHAT_COMPLETION},
            ]
        })],
    );

    let h = AIPerfHarness::new_with(tuned_mock_config(20.0, 5.0)).await;
    let r = run_hf(&h, &fixture);

    let (isl, osl) = sole_isl_osl(&r);
    assert_eq!(
        isl, CHAT_ISL,
        "chat-column ISL should match the tokenized first-user-message prompt"
    );
    assert_eq!(
        osl, CHAT_OSL,
        "chat-column OSL should be the first-assistant-message-derived length"
    );
    assert_streamed_ok(&r);
}
