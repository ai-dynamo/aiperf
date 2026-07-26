// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
mod common;
use common::*;

const WORKERS_MAX: u32 = 1;
const CONCURRENCY: u32 = 2;
const REQUEST_COUNT: u32 = 10;
const UI: &str = "simple";

const USAGE_METRIC_TAGS: [&str; 3] = [
    "usage_prompt_tokens",
    "usage_completion_tokens",
    "usage_total_tokens",
];

const USAGE_METRIC_CSV_NAMES: [&str; 3] = [
    "Usage Prompt Tokens",
    "Usage Completion Tokens",
    "Usage Total Tokens",
];

fn record_metric_keys(record: &serde_json::Value) -> Vec<String> {
    record
        .get("metrics")
        .and_then(|m| m.as_object())
        .map(|m| m.keys().cloned().collect())
        .unwrap_or_default()
}

async fn usage_metrics_in_exports(endpoint_type: &str, model: &str) {
    let h = AIPerfHarness::new().await;
    let r = h.run(&format!(
        "--model {model} --url {} --endpoint-type {endpoint_type} \
         --request-count {REQUEST_COUNT} --concurrency {CONCURRENCY} \
         --workers-max {WORKERS_MAX} --ui {UI}",
        h.mock.url
    ));

    assert_eq!(r.exit_code, 0);
    assert_eq!(r.artifacts.request_count() as u32, REQUEST_COUNT);

    let json_data = r.artifacts.json();
    if let Some(obj) = json_data.as_object() {
        let found_metrics: Vec<&str> = USAGE_METRIC_TAGS
            .iter()
            .filter(|m| {
                obj.get(**m)
                    .map(|v| !v.is_null() && !is_falsy(v))
                    .unwrap_or(false)
            })
            .copied()
            .collect();
        assert!(!found_metrics.is_empty());
    }

    let csv_content = r.artifacts.csv();
    if !csv_content.is_empty() {
        let usage_cols: Vec<&str> = USAGE_METRIC_CSV_NAMES
            .iter()
            .filter(|col| csv_content.contains(**col))
            .copied()
            .collect();
        assert!(!usage_cols.is_empty());
    }

    let jsonl = r.artifacts.jsonl();
    if !jsonl.is_empty() {
        let records_with_usage: Vec<&serde_json::Value> = jsonl
            .iter()
            .filter(|record| {
                record_metric_keys(record)
                    .iter()
                    .any(|k| k.starts_with("usage_"))
            })
            .collect();
        assert!(!records_with_usage.is_empty());
    }
}

/// Matches the export contract that excludes null, zero, and empty metrics.
fn is_falsy(v: &serde_json::Value) -> bool {
    match v {
        serde_json::Value::Null => true,
        serde_json::Value::Bool(b) => !b,
        serde_json::Value::Number(n) => n.as_f64().map(|f| f == 0.0).unwrap_or(false),
        serde_json::Value::String(s) => s.is_empty(),
        serde_json::Value::Array(a) => a.is_empty(),
        serde_json::Value::Object(o) => o.is_empty(),
    }
}

#[tokio::test]
async fn test_usage_metrics_in_exports_chat() {
    usage_metrics_in_exports("chat", "openai/gpt-oss-120b").await;
}

#[tokio::test]
async fn test_usage_metrics_in_exports_completions() {
    usage_metrics_in_exports("completions", "openai/gpt-oss-120b").await;
}

fn has_streaming_metrics(r: &RunResult) -> bool {
    let json = r.artifacts.json();
    [
        "time_to_first_token",
        "inter_token_latency",
        "inter_chunk_latency",
        "time_to_second_token",
    ]
    .iter()
    .all(|k| json.get(k).map(|v| !v.is_null()).unwrap_or(false))
}

#[tokio::test]
async fn test_streaming_usage_passthrough() {
    let h = AIPerfHarness::new().await;
    let r = h.run(&format!(
        "--model Qwen/Qwen2.5-32B-Instruct --url {} --endpoint-type chat --streaming \
         --extra-inputs '{{\"stream_options\": {{\"include_usage\": true}}}}' \
         --request-count {REQUEST_COUNT} --concurrency {CONCURRENCY} \
         --workers-max {WORKERS_MAX} --ui {UI}",
        h.mock.url
    ));

    assert_eq!(r.exit_code, 0);
    assert!(has_streaming_metrics(&r));

    let jsonl = r.artifacts.jsonl();
    let records_with_both: Vec<&serde_json::Value> = jsonl
        .iter()
        .filter(|record| {
            let keys = record_metric_keys(record);
            keys.iter().any(|k| k == "output_token_count")
                && keys.iter().any(|k| k.starts_with("usage_"))
        })
        .collect();
    assert!(!records_with_both.is_empty());
}
