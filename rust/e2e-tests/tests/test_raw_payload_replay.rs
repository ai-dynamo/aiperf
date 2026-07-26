// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
mod common;
use common::*;

use serde_json::json;

#[tokio::test]
async fn test_raw_payload_replays_authored_body_verbatim() {
    let h = AIPerfHarness::new().await;

    let payload = json!({
        "messages": [{"role": "user", "content": "raw-payload body"}],
        "model": DEFAULT_MODEL,
        "stream": false,
        "max_tokens": 7,
        "temperature": 0.01,
        "vendor_flag": {"preserve": true},
    });

    let input_dir = tempfile::TempDir::new().unwrap();
    let input_file = write_text(
        input_dir.path(),
        "payloads.jsonl",
        &format!("{}\n", serde_json::to_string(&payload).unwrap()),
    );

    let r = h.run_timeout(
        &format!(
            "--model {DEFAULT_MODEL} --url {} --endpoint-type chat \
                 --custom-dataset-type raw_payload --input-file {} \
                 --concurrency 1 --num-conversations 1 --workers-max 1 \
                 --export-level raw --ui simple",
            h.mock.url,
            input_file.display()
        ),
        120,
    );

    assert!(r.success(), "run failed: {}", r.stderr);
    let raw = r.artifacts.raw_records();
    assert_eq!(raw.len(), 1);
    assert_eq!(raw[0]["payload"], payload);
}

#[tokio::test]
async fn test_inputs_json_replays_stored_payload_verbatim() {
    let h = AIPerfHarness::new().await;

    let payload = json!({
        "messages": [{"role": "user", "content": "inputs-json body"}],
        "model": DEFAULT_MODEL,
        "stream": false,
        "max_tokens": 9,
        "temperature": 0.02,
        "vendor_flag": {"preserve": "inputs"},
    });

    let input_dir = tempfile::TempDir::new().unwrap();
    let inputs = json!({
        "data": [
            {
                "session_id": "session-raw-replay",
                "payloads": [payload],
            }
        ]
    });
    let input_file = write_text(
        input_dir.path(),
        "inputs.json",
        &serde_json::to_string(&inputs).unwrap(),
    );

    let r = h.run_timeout(
        &format!(
            "--model {DEFAULT_MODEL} --url {} --endpoint-type chat \
                 --custom-dataset-type inputs_json --input-file {} \
                 --concurrency 1 --num-conversations 1 --workers-max 1 \
                 --export-level raw --ui simple",
            h.mock.url,
            input_file.display()
        ),
        120,
    );

    assert!(r.success(), "run failed: {}", r.stderr);
    let raw = r.artifacts.raw_records();
    assert_eq!(raw.len(), 1);
    assert_eq!(raw[0]["payload"], payload);
}
