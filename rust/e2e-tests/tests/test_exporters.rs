// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
mod common;
use common::*;

const REQUEST_COUNT: u32 = 10;
const CONCURRENCY: u32 = 2;
const WORKERS_MAX: u32 = 1;
const UI: &str = "simple";

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
        !json
            .get("request_count")
            .unwrap_or(&serde_json::Value::Null)
            .is_null(),
        "json.request_count should exist"
    );
    assert!(
        !json
            .get("request_latency")
            .unwrap_or(&serde_json::Value::Null)
            .is_null(),
        "json.request_latency should exist"
    );
}

#[tokio::test]
async fn test_raw_export_level() {
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

    let raw_records = r.artifacts.raw_records();
    assert!(!raw_records.is_empty(), "raw records should exist");
    assert_eq!(
        raw_records.len(),
        REQUEST_COUNT as usize,
        "raw record count should match request count"
    );

    for record in &raw_records {
        let metadata = record.get("metadata").expect("metadata should exist");
        assert!(!metadata.is_null(), "metadata should not be null");
        assert!(
            metadata
                .get("turn_index")
                .and_then(|v| v.as_i64())
                .is_some(),
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

        assert!(
            record
                .get("start_perf_ns")
                .and_then(|v| v.as_i64())
                .is_some(),
            "start_perf_ns should be an int"
        );
        let payload = record.get("payload").expect("payload should exist");
        assert!(payload.is_object(), "payload should be a dict");

        let messages = payload
            .get("messages")
            .expect("payload.messages should exist");
        assert!(messages.is_array(), "messages should be a list");
        assert!(
            !messages.as_array().unwrap().is_empty(),
            "messages should be non-empty"
        );

        let status = record
            .get("status")
            .and_then(|v| v.as_i64())
            .expect("status should be an int");
        assert!((200..300).contains(&status), "status should be 2xx");

        let responses = record.get("responses").expect("responses should exist");
        assert!(responses.is_array(), "responses should be a list");

        assert!(
            record
                .get("error")
                .unwrap_or(&serde_json::Value::Null)
                .is_null(),
            "error should be null"
        );

        let request_headers = record
            .get("request_headers")
            .expect("request_headers should exist");
        assert!(
            request_headers.is_object(),
            "request_headers should be a dict"
        );
    }

    assert!(!r.artifacts.json().is_null(), "json export should exist");
    assert!(!r.artifacts.csv().is_empty(), "csv export should exist");
    assert!(!r.artifacts.jsonl().is_empty(), "jsonl export should exist");
}
