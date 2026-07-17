// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
mod common;
use common::*;

use serde_json::Value;

const REQUEST_COUNT: u32 = 6;
const CONCURRENCY: u32 = 2;
/// KServe model names occupy one URL path segment; this model also emits text.
const KSERVE_MODEL: &str = "gpt-4";

/// Non-streaming HTTP and unary gRPC bodies are stored in `responses[0].text`.
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

/// A successful `grpc-status: 0` maps to status 200.
fn record_status(record: &Value) -> Option<u64> {
    record.get("status").and_then(Value::as_u64)
}

fn first_output(body: &Value) -> Option<&Value> {
    body.get("outputs")
        .and_then(Value::as_array)
        .and_then(|outputs| outputs.first())
}

fn http_infer_config(url: &str, endpoint_type: &str) -> String {
    format!(
        "schemaVersion: \"2.0\"\n\
         benchmark:\n\
        \x20 models: [{KSERVE_MODEL}]\n\
        \x20 endpoint:\n\
        \x20   urls: [\"{url}\"]\n\
        \x20   type: {endpoint_type}\n\
        \x20   streaming: false\n\
        \x20   waitForModelTimeout: 0.0\n\
        \x20 dataset:\n\
        \x20   type: synthetic\n\
        \x20   entries: {REQUEST_COUNT}\n\
        \x20   prompts:\n\
        \x20     isl: 32\n\
        \x20     osl: 16\n\
        \x20 phases:\n\
        \x20   - name: profiling\n\
        \x20     type: concurrency\n\
        \x20     requests: {REQUEST_COUNT}\n\
        \x20     concurrency: {CONCURRENCY}\n\
        \x20 gpuTelemetry: {{enabled: false}}\n\
        \x20 serverMetrics: {{enabled: false}}\n\
        \x20 transport: {{type: http}}\n\
        \x20 runtime: {{ui: none}}\n"
    )
}

/// Builds the config from the harness URL because each harness binds independently.
async fn run_http(endpoint_type: &str) -> (AIPerfHarness, RunResult) {
    let h = AIPerfHarness::new().await;
    let config = http_infer_config(&h.mock.url, endpoint_type);
    let cfg = h.artifact_dir.path().join("kserve.yaml");
    std::fs::write(&cfg, config).unwrap();
    let r = h.run(&format!("--config {} --export-level raw", cfg.display()));
    assert!(
        r.success(),
        "run failed (exit {}):\nstdout:\n{}\nstderr:\n{}",
        r.exit_code,
        r.stdout,
        r.stderr
    );
    (h, r)
}

/// HTTP KServe v2 `ModelInfer`: each raw record carries a `text_output` tensor.
#[tokio::test]
async fn test_kserve_http_v2_infer_text() {
    let (_h, r) = run_http("kserve_v2_infer").await;
    let records = r.artifacts.raw_records();
    assert_eq!(
        records.len(),
        REQUEST_COUNT as usize,
        "one record per request"
    );
    for (i, record) in records.iter().enumerate() {
        assert_eq!(record_status(record), Some(200), "record {i} status");
        let body = first_response_json(record);
        let output =
            first_output(&body).unwrap_or_else(|| panic!("record {i}: no output tensor in {body}"));
        assert_eq!(output["name"], "text_output", "record {i} output name");
        assert_eq!(output["datatype"], "BYTES", "record {i} output datatype");
        let text = output["data"][0].as_str().unwrap_or("");
        assert!(!text.is_empty(), "record {i}: empty text_output");
    }
}

/// HTTP KServe v1 `:predict`: each raw record carries `predictions[].output`.
#[tokio::test]
async fn test_kserve_http_v1_predict() {
    let (_h, r) = run_http("kserve_v1_predict").await;
    let records = r.artifacts.raw_records();
    assert_eq!(
        records.len(),
        REQUEST_COUNT as usize,
        "one record per request"
    );
    for (i, record) in records.iter().enumerate() {
        assert_eq!(record_status(record), Some(200), "record {i} status");
        let body = first_response_json(record);
        let output = body
            .get("predictions")
            .and_then(Value::as_array)
            .and_then(|predictions| predictions.first())
            .and_then(|prediction| prediction.get("output"))
            .and_then(Value::as_str)
            .unwrap_or_else(|| panic!("record {i}: no predictions[].output in {body}"));
        assert!(!output.is_empty(), "record {i}: empty prediction output");
    }
}

/// `image_retrieval` defaults to `/v1/infer`, an alias of `/v1/image/infer`.
#[tokio::test]
async fn test_kserve_http_v1_infer_alias() {
    let h = AIPerfHarness::new().await;
    let r = h.run(&format!(
        "--model nvidia/page-elements-v2 --url {} --endpoint-type image_retrieval \
         --image-width-mean 64 --image-height-mean 64 \
         --request-count {REQUEST_COUNT} --concurrency {CONCURRENCY} --workers-max 1 \
         --ui none --export-level raw",
        h.mock.url,
    ));
    assert!(r.success(), "stderr:\n{}", r.stderr);
    let records = r.artifacts.raw_records();
    assert_eq!(
        records.len(),
        REQUEST_COUNT as usize,
        "one record per request"
    );
    for (i, record) in records.iter().enumerate() {
        assert_eq!(record_status(record), Some(200), "record {i} status");
        let body = first_response_json(record);
        assert!(
            body.get("data").and_then(Value::as_array).is_some(),
            "record {i}: no bounding-box `data` array in {body}"
        );
    }
}

fn grpc_config(grpc_url: &str, endpoint_type: &str, streaming: bool, records: &str) -> String {
    format!(
        "schemaVersion: \"2.0\"\n\
         benchmark:\n\
        \x20 models: [{KSERVE_MODEL}]\n\
        \x20 endpoint:\n\
        \x20   urls: [\"{grpc_url}\"]\n\
        \x20   type: {endpoint_type}\n\
        \x20   streaming: {streaming}\n\
        \x20   waitForModelTimeout: 0.0\n\
        {records}\
        \x20 phases:\n\
        \x20   - name: profiling\n\
        \x20     type: concurrency\n\
        \x20     requests: {REQUEST_COUNT}\n\
        \x20     concurrency: {CONCURRENCY}\n\
        \x20 gpuTelemetry: {{enabled: false}}\n\
        \x20 serverMetrics: {{enabled: false}}\n\
        \x20 transport: {{type: grpc}}\n\
        \x20 runtime: {{ui: none}}\n"
    )
}

async fn run_grpc(
    endpoint_type: &str,
    streaming: bool,
    records: &str,
) -> (AIPerfHarness, RunResult) {
    let h = AIPerfHarness::new_with_grpc().await;
    let grpc_url = h
        .mock
        .grpc_url
        .clone()
        .expect("mock started with grpc listener");
    let cfg = h.artifact_dir.path().join("kserve_grpc.yaml");
    std::fs::write(
        &cfg,
        grpc_config(&grpc_url, endpoint_type, streaming, records),
    )
    .unwrap();
    let r = h.run(&format!("--config {} --export-level raw", cfg.display()));
    assert!(
        r.success(),
        "grpc {endpoint_type} run failed (exit {}):\nstdout:\n{}\nstderr:\n{}",
        r.exit_code,
        r.stdout,
        r.stderr
    );
    (h, r)
}

const RANKINGS_RECORDS: &str = "\x20 dataset:\n\
    \x20   type: file\n\
    \x20   format: single_turn\n\
    \x20   sampling: sequential\n\
    \x20   records:\n\
    \x20     - texts:\n\
    \x20         - {name: query, contents: [\"what is machine learning\"]}\n\
    \x20         - {name: passages, contents: [\"ml is a field of ai\", \"cooking recipes\", \"deep neural networks\"]}\n\
    \x20     - texts:\n\
    \x20         - {name: query, contents: [\"capital of france\"]}\n\
    \x20         - {name: passages, contents: [\"paris is the capital\", \"london is in the uk\"]}\n";

const IMAGE_RECORDS: &str = "\x20 dataset:\n\
    \x20   type: file\n\
    \x20   format: single_turn\n\
    \x20   sampling: sequential\n\
    \x20   records:\n\
    \x20     - texts: [{name: text, contents: [\"a red bicycle on a hill\"]}]\n\
    \x20     - texts: [{name: text, contents: [\"a blue mountain sunset\"]}]\n";

const VLM_DATASET: &str = "\x20 dataset:\n\
    \x20   type: synthetic\n\
    \x20   entries: 6\n\
    \x20   prompts:\n\
    \x20     isl: 24\n\
    \x20     osl: 12\n\
    \x20   images:\n\
    \x20     batch_size: 1\n\
    \x20     width: 64\n\
    \x20     height: 64\n";

/// gRPC KServe v2 rankings (unary `ModelInfer`): each raw record carries a
/// numeric `scores` tensor with one score per passage, positionally.
#[tokio::test]
async fn test_kserve_grpc_rankings() {
    let (_h, r) = run_grpc("kserve_v2_rankings", false, RANKINGS_RECORDS).await;
    let records = r.artifacts.raw_records();
    assert_eq!(
        records.len(),
        REQUEST_COUNT as usize,
        "one record per request"
    );
    for (i, record) in records.iter().enumerate() {
        assert_eq!(record_status(record), Some(200), "record {i} status");
        let passages = record
            .get("payload")
            .and_then(|p| p.get("inputs"))
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .find(|tensor| tensor.get("name").and_then(Value::as_str) == Some("passages"))
            .and_then(|tensor| tensor.get("data").and_then(Value::as_array))
            .map(|data| data.len())
            .unwrap_or_else(|| panic!("record {i}: no passages input tensor"));

        let body = first_response_json(record);
        let output =
            first_output(&body).unwrap_or_else(|| panic!("record {i}: no output tensor in {body}"));
        assert_eq!(output["name"], "scores", "record {i} output name");
        let scores = output["data"]
            .as_array()
            .unwrap_or_else(|| panic!("record {i}: scores data not an array"));
        assert_eq!(scores.len(), passages, "record {i}: one score per passage");
        assert!(
            scores.iter().all(|score| score.is_number()),
            "record {i}: non-numeric score in {scores:?}"
        );
    }
}

/// gRPC KServe v2 images (unary `ModelInfer`): each raw record carries a
/// `generated_image` BYTES tensor with a non-empty base64 image string.
#[tokio::test]
async fn test_kserve_grpc_images() {
    let (_h, r) = run_grpc("kserve_v2_images", false, IMAGE_RECORDS).await;
    let records = r.artifacts.raw_records();
    assert_eq!(
        records.len(),
        REQUEST_COUNT as usize,
        "one record per request"
    );
    for (i, record) in records.iter().enumerate() {
        assert_eq!(record_status(record), Some(200), "record {i} status");
        let body = first_response_json(record);
        let output =
            first_output(&body).unwrap_or_else(|| panic!("record {i}: no output tensor in {body}"));
        assert_eq!(output["name"], "generated_image", "record {i} output name");
        assert_eq!(output["datatype"], "BYTES", "record {i} output datatype");
        let image = output["data"][0].as_str().unwrap_or("");
        assert!(!image.is_empty(), "record {i}: empty generated_image");
        // `/9j/` is the base64-encoded JPEG SOI marker.
        assert!(
            image.starts_with("/9j/"),
            "record {i}: generated_image is not a base64 JPEG: {}",
            &image[..image.len().min(16)]
        );
    }
}

/// gRPC KServe v2 vlm (server-streaming `ModelStreamInfer`): the request carries
/// an `image` input tensor (consumed by the mock) plus `text_input`, and the
/// mock streams generated text back. Each raw record must show the `image`
/// tensor was sent and assemble non-empty streamed text.
#[tokio::test]
async fn test_kserve_grpc_vlm_streaming() {
    let (_h, r) = run_grpc("kserve_v2_vlm", true, VLM_DATASET).await;
    let records = r.artifacts.raw_records();
    assert_eq!(
        records.len(),
        REQUEST_COUNT as usize,
        "one record per request"
    );
    for (i, record) in records.iter().enumerate() {
        assert_eq!(record_status(record), Some(200), "record {i} status");

        let input_names: Vec<&str> = record
            .get("payload")
            .and_then(|p| p.get("inputs"))
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(|tensor| tensor.get("name").and_then(Value::as_str))
            .collect();
        assert!(
            input_names.contains(&"image"),
            "record {i}: no `image` input tensor (VLM must send one), got {input_names:?}"
        );
        assert!(
            input_names.contains(&"text_input"),
            "record {i}: no `text_input` tensor, got {input_names:?}"
        );

        let mut assembled = String::new();
        for response in record
            .get("responses")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            let Some(text) = response.get("text").and_then(Value::as_str) else {
                continue;
            };
            let Ok(body) = serde_json::from_str::<Value>(text) else {
                continue;
            };
            if let Some(output) = first_output(&body)
                && output["name"] == "text_output"
                && let Some(chunk) = output["data"][0].as_str()
            {
                assembled.push_str(chunk);
            }
        }
        assert!(
            !assembled.is_empty(),
            "record {i}: streamed VLM text was empty"
        );
    }
}
