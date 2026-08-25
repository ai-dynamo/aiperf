// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

mod common;
use common::*;

use serde_json::Value;

fn non_streaming_body(record: &Value) -> Option<Value> {
    let responses = record.get("responses").and_then(Value::as_array)?;
    for resp in responses {
        if let Some(text) = resp.get("text").and_then(Value::as_str) {
            if let Ok(body) = serde_json::from_str::<Value>(text) {
                return Some(body);
            }
        }
    }
    None
}

#[tokio::test]
async fn test_audio_transcription_passes_generated_audio_and_extra_inputs() {
    let h = AIPerfHarness::new().await;
    let r = h.run(&format!(
        "--model openai/whisper-1 \
         --tokenizer gpt2 \
         --url {} \
         --endpoint-type audio_transcription \
         --audio-length-mean 0.1 \
         --audio-format wav \
         --extra-inputs language:en temperature:0.0 \
         --request-count 4 \
         --concurrency 2 \
         --workers-max 1 \
         --export-level raw \
         --ui simple",
        h.mock.url
    ));
    assert!(
        r.success(),
        "audio transcription run failed (exit {}):\nstdout:\n{}\nstderr:\n{}",
        r.exit_code,
        r.stdout,
        r.stderr
    );
    assert_eq!(r.artifacts.request_count() as u32, 4);

    let records = r.artifacts.raw_records();
    assert_eq!(records.len(), 4, "expected 4 raw records");
    for (index, record) in records.iter().enumerate() {
        assert_eq!(
            record.get("status").and_then(Value::as_u64),
            Some(200),
            "record {index}: status not 200\n{record}"
        );
        assert!(
            record["request_headers"]["Content-Type"]
                .as_str()
                .is_some_and(|value| value.starts_with("multipart/form-data; boundary=")),
            "record {index}: request did not go over native multipart\n{record}"
        );
        let body = non_streaming_body(record)
            .unwrap_or_else(|| panic!("record {index}: no parseable response body\n{record}"));
        assert_eq!(body["text"], "mock transcription", "record {index}");
        assert_eq!(body["language"], "en", "record {index}");
        assert_eq!(body["temperature"], 0.0, "record {index}");
        assert_eq!(body["usage"]["input_tokens"], 1, "record {index}");
    }
}
