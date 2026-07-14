// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#[path = "common/mod.rs"]
mod common;
use common::*;

// Tests for different output export formats.
// Ported from `tests/integration/test_exporters.py`.

const REQUEST_COUNT: u32 = 10;
const CONCURRENCY: u32 = 2;
const WORKERS_MAX: u32 = 1;
const UI: &str = "simple";

/// CSV export format validation.
#[tokio::test]
async fn test_csv_export() {
    let h = AIPerfHarness::new().await;
    let r = h.run(&format!(
        "--model Qwen/Qwen2.5-Coder-32B-Instruct \
         --url {} \
         --endpoint-type chat \
         --streaming \
         --request-count {REQUEST_COUNT} \
         --concurrency {CONCURRENCY} \
         --workers-max {WORKERS_MAX} \
         --ui {UI}",
        h.mock.url
    ));
    let csv = r.artifacts.csv();
    assert!(csv.contains("Metric"), "csv should contain 'Metric'");
    assert!(
        csv.contains("Request Latency"),
        "csv should contain 'Request Latency'"
    );
}

/// JSON export format validation.
#[tokio::test]
async fn test_json_export() {
    let h = AIPerfHarness::new().await;
    let r = h.run(&format!(
        "--model microsoft/Phi-4-reasoning \
         --url {} \
         --endpoint-type chat \
         --request-count {REQUEST_COUNT} \
         --concurrency {CONCURRENCY} \
         --workers-max {WORKERS_MAX} \
         --ui {UI}",
        h.mock.url
    ));
    let json = r.artifacts.json();
    assert!(!json.is_null(), "json should exist");
    assert!(
        !json.get("request_count").unwrap_or(&serde_json::Value::Null).is_null(),
        "json.request_count should exist"
    );
    assert!(
        !json.get("request_latency").unwrap_or(&serde_json::Value::Null).is_null(),
        "json.request_latency should exist"
    );
}

/// Test that raw records are properly created using --export-level raw.
#[tokio::test]
async fn test_raw_export_level() {
    // Skipif Darwin: flaky on macOS in GitHub Actions.
    if cfg!(target_os = "macos") {
        return;
    }

    let h = AIPerfHarness::new().await;
    let r = h.run(&format!(
        "--model Qwen/Qwen2.5-Coder-32B-Instruct \
         --url {} \
         --endpoint-type chat \
         --streaming \
         --request-count {REQUEST_COUNT} \
         --concurrency {CONCURRENCY} \
         --workers-max {WORKERS_MAX} \
         --export-level raw \
         --ui {UI}",
        h.mock.url
    ));

    // Verify raw records file exists.
    let raw_records = r.artifacts.raw_records();
    assert!(!raw_records.is_empty(), "raw records should exist");
    assert_eq!(
        raw_records.len(),
        REQUEST_COUNT as usize,
        "raw record count should match request count"
    );

    // Validate raw record structure and content.
    for record in &raw_records {
        // Verify metadata exists and has required fields.
        let metadata = record.get("metadata").expect("metadata should exist");
        assert!(!metadata.is_null(), "metadata should not be null");
        assert!(
            metadata.get("turn_index").and_then(|v| v.as_i64()).is_some(),
            "turn_index should be an int"
        );
        assert!(
            metadata
                .get("request_start_ns")
                .and_then(|v| v.as_i64())
                .is_some(),
            "request_start_ns should be an int"
        );
        assert!(
            !metadata
                .get("worker_id")
                .unwrap_or(&serde_json::Value::Null)
                .is_null(),
            "worker_id should exist"
        );
        assert!(
            !metadata
                .get("record_processor_id")
                .unwrap_or(&serde_json::Value::Null)
                .is_null(),
            "record_processor_id should exist"
        );
        assert!(
            !metadata
                .get("benchmark_phase")
                .unwrap_or(&serde_json::Value::Null)
                .is_null(),
            "benchmark_phase should exist"
        );

        // Verify raw record fields exist.
        assert!(
            record.get("start_perf_ns").and_then(|v| v.as_i64()).is_some(),
            "start_perf_ns should be an int"
        );
        let payload = record.get("payload").expect("payload should exist");
        assert!(payload.is_object(), "payload should be a dict");

        // Verify payload has expected structure for chat endpoint.
        let messages = payload.get("messages").expect("payload.messages should exist");
        assert!(messages.is_array(), "messages should be a list");
        assert!(
            !messages.as_array().unwrap().is_empty(),
            "messages should be non-empty"
        );

        // Verify status code exists and is valid.
        let status = record
            .get("status")
            .and_then(|v| v.as_i64())
            .expect("status should be an int");
        assert!((200..300).contains(&status), "status should be 2xx");

        // Verify responses exist (should have at least one for streaming).
        let responses = record.get("responses").expect("responses should exist");
        assert!(responses.is_array(), "responses should be a list");

        // Verify error is None for successful requests.
        assert!(
            record
                .get("error")
                .unwrap_or(&serde_json::Value::Null)
                .is_null(),
            "error should be null"
        );

        // Verify request headers exist.
        let request_headers = record
            .get("request_headers")
            .expect("request_headers should exist");
        assert!(request_headers.is_object(), "request_headers should be a dict");
    }

    // Verify standard exports still exist.
    assert!(!r.artifacts.json().is_null(), "json export should exist");
    assert!(!r.artifacts.csv().is_empty(), "csv export should exist");
    assert!(!r.artifacts.jsonl().is_empty(), "jsonl export should exist");
}
