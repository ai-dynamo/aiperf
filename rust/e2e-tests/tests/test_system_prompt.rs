// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Full-stack request-wire coverage for verbatim system prompts.

mod common;
use common::*;

use serde_json::{Value, json};

fn only_payload(result: &RunResult) -> Value {
    assert!(
        result.success(),
        "profile failed (exit {}):\nstdout:\n{}\nstderr:\n{}",
        result.exit_code,
        result.stdout,
        result.stderr
    );
    let records = result.artifacts.raw_records();
    assert_eq!(records.len(), 1, "expected one raw record: {records:?}");
    assert_eq!(records[0]["status"], 200, "request failed: {}", records[0]);
    records[0]["payload"].clone()
}

#[tokio::test]
async fn file_prompt_is_verbatim_and_leads_the_authored_chat_system() {
    let harness = AIPerfHarness::new().await;
    let dataset = write_jsonl(
        harness.artifact_path(),
        "authored-system.jsonl",
        &[json!({
            "session_id":"system-file-e2e",
            "turns":[
                {"role":"system","text":"authored system"},
                {"role":"user","text":"hello"}
            ]
        })],
    );
    let verbatim = "  verbatim line one\nline two  ";
    let prompt = write_text(harness.artifact_path(), "system-prompt.txt", verbatim);

    let result = harness.run(&format!(
        "--model mock-model --url {} --endpoint-type chat \
         --input-file {} --custom-dataset-type multi_turn \
         --num-conversations 1 --concurrency 1 --workers-max 1 \
         --output-tokens-mean 1 --export-level raw --ui none --tokenizer builtin \
         --system-prompt-file {}",
        harness.mock.url,
        dataset.display(),
        prompt.display()
    ));
    let payload = only_payload(&result);

    assert_eq!(
        payload["messages"],
        json!([
            {"role":"system","content":format!("{verbatim}\n\nauthored system")},
            {"role":"user","content":"hello"}
        ])
    );
}

#[tokio::test]
async fn inline_prompt_reaches_responses_and_anthropic_production_wires() {
    for endpoint in ["responses", "messages"] {
        let harness = AIPerfHarness::new().await;
        let result = harness.run(&format!(
            "--model mock-model --url {} --endpoint-type {endpoint} \
             --request-count 1 --concurrency 1 --workers-max 1 \
             --synthetic-input-tokens-mean 8 --synthetic-input-tokens-stddev 0 \
             --output-tokens-mean 1 --export-level raw --ui none --tokenizer builtin \
             --system-prompt 'verbatim system'",
            harness.mock.url
        ));
        let payload = only_payload(&result);

        match endpoint {
            "responses" => assert_eq!(payload["instructions"], "verbatim system"),
            "messages" => assert_eq!(payload["system"], "verbatim system"),
            _ => unreachable!(),
        }
    }
}
