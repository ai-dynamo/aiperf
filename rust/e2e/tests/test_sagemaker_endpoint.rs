// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
mod common;
use common::*;

use serde_json::Value;

const REQUEST_COUNT: u32 = 6;
const CONCURRENCY: u32 = 2;
const SAGEMAKER_MODEL: &str = "my-endpoint";

/// Non-streaming bodies are stored in `responses[0].text`; the SageMaker mock
/// route always responds OpenAI chat-completion shaped.
fn first_response_json(record: &Value) -> Value {
    let text = record
        .get("responses")
        .and_then(Value::as_array)
        .and_then(|responses| responses.first())
        .and_then(|response| response.get("text"))
        .and_then(Value::as_str)
        .unwrap_or("");
    serde_json::from_str(text).unwrap_or(Value::Null)
}

fn record_status(record: &Value) -> Option<u64> {
    record.get("status").and_then(Value::as_u64)
}

async fn run(endpoint_type: &str, streaming: bool) -> (AIPerfHarness, RunResult) {
    let h = AIPerfHarness::new().await;
    let streaming_flag = if streaming { "--streaming" } else { "" };
    let r = h.run(&format!(
        "--model {SAGEMAKER_MODEL} --url {} --endpoint-type {endpoint_type} \
         {streaming_flag} --isl 32 \
         --osl 16 --osl-stddev 0 \
         --request-count {REQUEST_COUNT} --concurrency {CONCURRENCY} --workers-max 1 \
         --ui none --export-level raw",
        h.mock.url,
    ));
    assert!(
        r.success(),
        "{endpoint_type} run failed (exit {}):\nstdout:\n{}\nstderr:\n{}",
        r.exit_code,
        r.stdout,
        r.stderr
    );
    (h, r)
}

/// AWS SageMaker Runtime `InvokeEndpoint`: `/endpoints/{model}/invocations`.
/// Non-streaming; every record is a single OpenAI chat-completion-shaped body.
#[tokio::test]
async fn test_sagemaker_invoke_non_streaming() {
    let (_h, r) = run("sagemaker", false).await;
    let records = r.artifacts.raw_records();
    assert_eq!(
        records.len(),
        REQUEST_COUNT as usize,
        "one record per request"
    );
    for (i, record) in records.iter().enumerate() {
        assert_eq!(record_status(record), Some(200), "record {i} status");
        let body = first_response_json(record);
        let content = body["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or_else(|| panic!("record {i}: no choices[0].message.content in {body}"));
        assert!(!content.is_empty(), "record {i}: empty content");
        assert_eq!(
            body["model"].as_str(),
            Some(SAGEMAKER_MODEL),
            "record {i} response model"
        );
        assert_eq!(
            record["payload"]["model"].as_str(),
            Some(SAGEMAKER_MODEL),
            "record {i} request payload model"
        );
        assert!(
            body["usage"]["prompt_tokens"].as_u64().unwrap_or(0) > 0,
            "record {i}: missing ISL"
        );
        assert!(
            body["usage"]["completion_tokens"].as_u64().unwrap_or(0) > 0,
            "record {i}: missing OSL"
        );
    }
}

/// AWS SageMaker Runtime `InvokeEndpointWithResponseStream`:
/// `/endpoints/{model}/invocations-response-stream`, framed as real AWS
/// eventstream binary frames (not SSE) on the wire; the client transport
/// decodes them via `eventstream_to_sse` before the shared SSE-record
/// aggregation path assembles them into the same per-record shape as the
/// non-streaming variant.
#[tokio::test]
async fn test_sagemaker_invoke_streaming() {
    let (_h, r) = run("sagemaker_stream", true).await;
    let records = r.artifacts.raw_records();
    assert_eq!(
        records.len(),
        REQUEST_COUNT as usize,
        "one record per request"
    );
    for (i, record) in records.iter().enumerate() {
        assert_eq!(record_status(record), Some(200), "record {i} status");
        let responses = record
            .get("responses")
            .and_then(Value::as_array)
            .unwrap_or_else(|| panic!("record {i}: no responses array"));
        assert!(
            responses.len() > 1,
            "record {i}: expected multiple streamed chunks, got {}",
            responses.len()
        );
        let full_text: String = responses
            .iter()
            .flat_map(|r| {
                r.get("packets")
                    .and_then(Value::as_array)
                    .map(|packets| packets.as_slice())
                    .unwrap_or(&[])
            })
            .filter(|packet| packet.get("name").and_then(Value::as_str) == Some("data"))
            .filter_map(|packet| packet.get("value").and_then(Value::as_str))
            .filter_map(|text| serde_json::from_str::<Value>(text).ok())
            .filter_map(|chunk| {
                chunk["choices"][0]["delta"]["content"]
                    .as_str()
                    .map(str::to_owned)
            })
            .collect();
        assert!(!full_text.is_empty(), "record {i}: empty streamed text");
        assert_eq!(
            record["payload"]["model"].as_str(),
            Some(SAGEMAKER_MODEL),
            "record {i} request payload model"
        );

        let request_ack_ns = record["metadata"]["request_ack_ns"].as_i64().unwrap_or(0);
        let request_start_ns = record["metadata"]["request_start_ns"].as_i64().unwrap_or(0);
        assert!(
            request_ack_ns > request_start_ns,
            "record {i}: missing TTFT (request_ack_ns should follow request_start_ns)"
        );
    }
}
