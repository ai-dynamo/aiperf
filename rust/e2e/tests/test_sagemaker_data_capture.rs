// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
mod common;
use common::*;

use base64::Engine as _;
use serde_json::{Value, json};
use std::path::{Path, PathBuf};

fn has_all_outputs(r: &RunResult) -> bool {
    !r.artifacts.json().is_null()
        && !r.artifacts.csv().is_empty()
        && !r.artifacts.inputs().is_null()
        && !r.artifacts.jsonl().is_empty()
}

fn create_sagemaker_capture_record(
    messages: Value,
    max_tokens: Option<i64>,
    inference_time: &str,
    encoding: &str,
) -> Value {
    let prompt_tokens = 28;
    let completion_tokens = 15;
    let event_id = "e4378ff2-0000-0000-0000-000000000000";

    let mut input_payload = json!({ "messages": messages });
    if let Some(mt) = max_tokens {
        input_payload["max_tokens"] = json!(mt);
    }

    let output_payload = json!({
        "id": "chatcmpl-test",
        "choices": [{"message": {"role": "assistant", "content": "Hi"}}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    });

    let encode = |payload: &Value| -> (String, String) {
        let raw = serde_json::to_vec(payload).unwrap();
        match encoding {
            "BASE64" => (
                base64::engine::general_purpose::STANDARD.encode(&raw),
                "BASE64".to_string(),
            ),
            "JSON" => (String::from_utf8(raw).unwrap(), "JSON".to_string()),
            other => panic!("Unsupported encoding for test helper: {other}"),
        }
    };

    let (input_data, input_enc) = encode(&input_payload);
    let (output_data, output_enc) = encode(&output_payload);

    json!({
        "captureData": {
            "endpointInput": {
                "observedContentType": "application/json",
                "mode": "INPUT",
                "data": input_data,
                "encoding": input_enc,
            },
            "endpointOutput": {
                "observedContentType": "application/json",
                "mode": "OUTPUT",
                "data": output_data,
                "encoding": output_enc,
            },
        },
        "eventMetadata": {
            "eventId": event_id,
            "inferenceTime": inference_time,
        },
        "eventVersion": "0",
    })
}

fn create_sagemaker_capture_file(dir: &Path, records: &[Value], filename: &str) -> PathBuf {
    write_jsonl(dir, filename, records)
}

#[tokio::test]
async fn test_basic_capture_replay() {
    let h = AIPerfHarness::new().await;
    let records = vec![
        create_sagemaker_capture_record(
            json!([{"role": "user", "content": "What is the capital of France?"}]),
            Some(50),
            "2026-04-29T00:00:00Z",
            "JSON",
        ),
        create_sagemaker_capture_record(
            json!([{"role": "user", "content": "Tell me about Python."}]),
            Some(80),
            "2026-04-29T00:00:02Z",
            "JSON",
        ),
        create_sagemaker_capture_record(
            json!([{"role": "user", "content": "What is machine learning?"}]),
            Some(60),
            "2026-04-29T00:00:04Z",
            "JSON",
        ),
    ];
    let capture_file =
        create_sagemaker_capture_file(h.artifact_dir.path(), &records, "capture.jsonl");
    let request_count = records.len();

    let r = h.run(&format!(
        "--model {DEFAULT_MODEL} --url {} --endpoint-type chat \
         --input-file {} --custom-dataset-type sagemaker_data_capture \
         --request-count {} --fixed-schedule --fixed-schedule-auto-offset \
         --workers-max 1 --ui simple",
        h.mock.url,
        capture_file.display(),
        request_count,
    ));

    assert_eq!(r.exit_code, 0);
    assert_eq!(r.artifacts.request_count() as usize, request_count);
    assert!(has_all_outputs(&r));
}

#[tokio::test]
async fn test_capture_with_system_message() {
    let h = AIPerfHarness::new().await;
    let records = vec![
        create_sagemaker_capture_record(
            json!([
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello"},
            ]),
            Some(30),
            "2026-04-29T00:00:00Z",
            "JSON",
        ),
        create_sagemaker_capture_record(
            json!([
                {"role": "system", "content": "You are a coding expert."},
                {"role": "user", "content": "Write a hello world in Python."},
            ]),
            Some(100),
            "2026-04-29T00:00:02Z",
            "JSON",
        ),
    ];
    let capture_file =
        create_sagemaker_capture_file(h.artifact_dir.path(), &records, "capture.jsonl");
    let request_count = records.len();

    let r = h.run(&format!(
        "--model {DEFAULT_MODEL} --url {} --endpoint-type chat \
         --input-file {} --custom-dataset-type sagemaker_data_capture \
         --request-count {} --fixed-schedule --fixed-schedule-auto-offset \
         --workers-max 1 --ui simple",
        h.mock.url,
        capture_file.display(),
        request_count,
    ));

    assert_eq!(r.exit_code, 0);
    assert_eq!(r.artifacts.request_count() as usize, request_count);
}

#[tokio::test]
async fn test_capture_directory_input() {
    let h = AIPerfHarness::new().await;
    let base = h.artifact_dir.path().join("captures");
    let hour_00 = base.join("2026").join("04").join("29").join("00");
    let hour_01 = base.join("2026").join("04").join("29").join("01");
    std::fs::create_dir_all(&hour_00).unwrap();
    std::fs::create_dir_all(&hour_01).unwrap();

    create_sagemaker_capture_file(
        &hour_00,
        &[create_sagemaker_capture_record(
            json!([{"role": "user", "content": "Request from hour 0"}]),
            Some(50),
            "2026-04-29T00:00:00Z",
            "JSON",
        )],
        "capture-00.jsonl",
    );
    create_sagemaker_capture_file(
        &hour_01,
        &[create_sagemaker_capture_record(
            json!([{"role": "user", "content": "Request from hour 1"}]),
            Some(50),
            "2026-04-29T00:00:02Z",
            "JSON",
        )],
        "capture-01.jsonl",
    );

    let r = h.run(&format!(
        "--model {DEFAULT_MODEL} --url {} --endpoint-type chat \
         --input-file {} --custom-dataset-type sagemaker_data_capture \
         --fixed-schedule --fixed-schedule-auto-offset \
         --workers-max 1 --ui simple",
        h.mock.url,
        base.display(),
    ));

    assert_eq!(r.exit_code, 0);
    assert_eq!(r.artifacts.request_count() as usize, 2);
    assert!(has_all_outputs(&r));
}

#[tokio::test]
async fn test_capture_auto_detection() {
    let h = AIPerfHarness::new().await;
    let records = vec![create_sagemaker_capture_record(
        json!([{"role": "user", "content": "Auto-detect test"}]),
        Some(50),
        "2026-04-29T00:00:00Z",
        "JSON",
    )];
    let capture_file =
        create_sagemaker_capture_file(h.artifact_dir.path(), &records, "capture.jsonl");

    let r = h.run(&format!(
        "--model {DEFAULT_MODEL} --url {} --endpoint-type chat \
         --input-file {} --request-count 1 \
         --fixed-schedule --fixed-schedule-auto-offset \
         --workers-max 1 --ui simple",
        h.mock.url,
        capture_file.display(),
    ));

    assert_eq!(r.exit_code, 0);
    assert_eq!(r.artifacts.request_count() as usize, 1);
}

#[tokio::test]
async fn test_capture_with_base64_encoding() {
    let h = AIPerfHarness::new().await;
    let records = vec![
        create_sagemaker_capture_record(
            json!([{"role": "user", "content": "Base64 test request 1"}]),
            Some(40),
            "2026-04-29T00:00:00Z",
            "BASE64",
        ),
        create_sagemaker_capture_record(
            json!([{"role": "user", "content": "Base64 test request 2"}]),
            Some(60),
            "2026-04-29T00:00:02Z",
            "BASE64",
        ),
    ];
    let capture_file =
        create_sagemaker_capture_file(h.artifact_dir.path(), &records, "capture.jsonl");
    let request_count = records.len();

    let r = h.run(&format!(
        "--model {DEFAULT_MODEL} --url {} --endpoint-type chat \
         --input-file {} --custom-dataset-type sagemaker_data_capture \
         --request-count {} --fixed-schedule --fixed-schedule-auto-offset \
         --workers-max 1 --ui simple",
        h.mock.url,
        capture_file.display(),
        request_count,
    ));

    assert_eq!(r.exit_code, 0);
    assert_eq!(r.artifacts.request_count() as usize, request_count);
    assert!(has_all_outputs(&r));
}
