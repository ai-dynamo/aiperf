// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
mod common;
use common::*;

use serde_json::{Value, json};
use std::path::{Path, PathBuf};

fn create_mooncake_trace_file(dir: &Path, traces: &[Value]) -> PathBuf {
    write_jsonl(dir, "traces.jsonl", traces)
}

fn has_all_outputs(r: &RunResult) -> bool {
    !r.artifacts.json().is_null()
        && !r.artifacts.csv().is_empty()
        && !r.artifacts.inputs().is_null()
        && !r.artifacts.jsonl().is_empty()
}

#[tokio::test]
async fn test_basic_mooncake_trace_with_input_length() {
    let h = AIPerfHarness::new().await;
    let traces = vec![
        json!({"timestamp": 0, "input_length": 6755, "output_length": 500, "hash_ids": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]}),
        json!({"timestamp": 0, "input_length": 7319, "output_length": 490, "hash_ids": [0, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]}),
        json!({"timestamp": 0, "input_length": 7234, "output_length": 794, "hash_ids": [0, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41]}),
        json!({"timestamp": 0, "input_length": 2287, "output_length": 316, "hash_ids": [0, 42, 43, 44, 45]}),
        json!({"timestamp": 0, "input_length": 9013, "output_length": 3, "hash_ids": [46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]}),
    ];
    let trace_file = create_mooncake_trace_file(h.artifact_dir.path(), &traces);
    let request_count = traces.len();

    let r = h.run(&format!(
        "--model {DEFAULT_MODEL} --url {} --endpoint-type chat \
         --input-file {} --custom-dataset-type mooncake_trace \
         --request-count {} --fixed-schedule --workers-max 1 --ui simple",
        h.mock.url,
        trace_file.display(),
        request_count,
    ));

    assert_eq!(r.artifacts.request_count() as usize, request_count);
    assert!(has_all_outputs(&r));
}

#[tokio::test]
async fn test_mooncake_trace_with_text_input() {
    let h = AIPerfHarness::new().await;
    let traces = vec![
        json!({"timestamp": 0, "text_input": "What is the capital of France?", "output_length": 20}),
        json!({"timestamp": 100, "text_input": "Explain quantum computing briefly.", "output_length": 30}),
        json!({"timestamp": 200, "text_input": "Write a haiku about programming.", "output_length": 25}),
        json!({"timestamp": 300, "text_input": "What is machine learning?", "output_length": 40}),
        json!({"timestamp": 400, "text_input": "Describe the solar system.", "output_length": 35}),
    ];
    let trace_file = create_mooncake_trace_file(h.artifact_dir.path(), &traces);
    let request_count = traces.len();

    let r = h.run(&format!(
        "--model {DEFAULT_MODEL} --url {} --endpoint-type chat \
         --input-file {} --custom-dataset-type mooncake_trace \
         --request-count {} --fixed-schedule --workers-max 1 --ui simple",
        h.mock.url,
        trace_file.display(),
        request_count,
    ));

    assert_eq!(r.artifacts.request_count() as usize, request_count);
    assert!(has_all_outputs(&r));
}

#[tokio::test]
async fn test_mooncake_trace_with_messages_field() {
    let h = AIPerfHarness::new().await;
    let traces = vec![
        json!({"timestamp": 0, "messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "What is the capital of France?"}], "output_length": 20}),
        json!({"timestamp": 100, "messages": [{"role": "user", "content": "Explain quantum computing."}], "output_length": 30}),
        json!({"timestamp": 200, "messages": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi there!"}, {"role": "user", "content": "Thanks for the help."}], "output_length": 25}),
    ];
    let trace_file = create_mooncake_trace_file(h.artifact_dir.path(), &traces);
    let request_count = traces.len();

    let r = h.run(&format!(
        "--model {DEFAULT_MODEL} --url {} --endpoint-type chat \
         --input-file {} --custom-dataset-type mooncake_trace \
         --request-count {} --fixed-schedule --workers-max 1 --ui simple",
        h.mock.url,
        trace_file.display(),
        request_count,
    ));

    assert_eq!(r.artifacts.request_count() as usize, request_count);
    assert!(has_all_outputs(&r));
}

#[tokio::test]
async fn test_mooncake_trace_with_messages_and_tools() {
    let h = AIPerfHarness::new().await;
    let traces = vec![
        json!({"timestamp": 0, "messages": [{"role": "user", "content": "What's the weather?"}], "tools": [{"type": "function", "function": {"name": "get_weather", "description": "Get weather", "parameters": {"type": "object", "properties": {"location": {"type": "string"}}}}}], "output_length": 50}),
        json!({"timestamp": 100, "messages": [{"role": "user", "content": "Hello"}], "output_length": 20}),
    ];
    let trace_file = create_mooncake_trace_file(h.artifact_dir.path(), &traces);
    let request_count = traces.len();

    let r = h.run(&format!(
        "--model {DEFAULT_MODEL} --url {} --endpoint-type chat \
         --input-file {} --custom-dataset-type mooncake_trace \
         --request-count {} --fixed-schedule --workers-max 1 --ui simple",
        h.mock.url,
        trace_file.display(),
        request_count,
    ));

    assert_eq!(r.artifacts.request_count() as usize, request_count);
    assert!(has_all_outputs(&r));
}

#[tokio::test]
async fn test_mooncake_trace_multi_turn_with_session_id() {
    let h = AIPerfHarness::new().await;
    let traces = vec![
        json!({"session_id": "session-1", "timestamp": 0, "input_length": 100, "output_length": 40}),
        json!({"session_id": "session-1", "delay": 500, "input_length": 150, "output_length": 50}),
        json!({"session_id": "session-2", "timestamp": 100, "input_length": 200, "output_length": 60}),
        json!({"session_id": "session-3", "timestamp": 200, "input_length": 80, "output_length": 30}),
        json!({"session_id": "session-3", "delay": 300, "input_length": 120, "output_length": 45}),
        json!({"session_id": "session-3", "delay": 400, "input_length": 90, "output_length": 35}),
    ];
    let trace_file = create_mooncake_trace_file(h.artifact_dir.path(), &traces);
    let request_count = traces.len();

    let r = h.run(&format!(
        "--model {DEFAULT_MODEL} --url {} --endpoint-type chat \
         --input-file {} --custom-dataset-type mooncake_trace \
         --request-count {} --fixed-schedule --workers-max 1 --ui simple",
        h.mock.url,
        trace_file.display(),
        request_count,
    ));

    assert_eq!(r.artifacts.request_count() as usize, request_count);
    assert!(has_all_outputs(&r));
}

#[tokio::test]
async fn test_mooncake_trace_block_size_override_replays_trace() {
    let h = AIPerfHarness::new().await;
    // input_length 48 with 3 hash blocks is consistent with block_size 16
    // (final block = 48 - 2*16 = 16) but not with the default 512.
    let traces = vec![
        json!({"input_length": 48, "output_length": 8, "hash_ids": [1, 2, 3], "timestamp": 100}),
        json!({"input_length": 48, "output_length": 8, "hash_ids": [4, 5, 6], "timestamp": 200}),
    ];
    let trace_file = create_mooncake_trace_file(h.artifact_dir.path(), &traces);
    let request_count = traces.len();

    let r = h.run(&format!(
        "--model {DEFAULT_MODEL} --url {} --endpoint-type chat \
         --input-file {} --custom-dataset-type mooncake_trace \
         --isl-block-size 16 --request-count {} --concurrency 1 \
         --workers-max 1 --export-level records --ui simple",
        h.mock.url,
        trace_file.display(),
        request_count,
    ));

    assert!(r.success(), "run failed: {}", r.stderr);
    assert_eq!(r.artifacts.request_count() as usize, request_count);
}

#[tokio::test]
async fn test_mooncake_trace_text_input_with_synthesis_speedup() {
    let h = AIPerfHarness::new().await;
    let traces = vec![
        json!({"timestamp": 1000, "text_input": "What is AI?", "output_length": 50}),
        json!({"timestamp": 2000, "text_input": "Explain quantum computing", "output_length": 100}),
        json!({"timestamp": 3000, "text_input": "How does machine learning work?", "output_length": 75}),
        json!({"timestamp": 4000, "text_input": "What are neural networks?", "output_length": 80}),
        json!({"timestamp": 5000, "text_input": "Describe the benefits of cloud computing", "output_length": 120}),
    ];
    let trace_file = create_mooncake_trace_file(h.artifact_dir.path(), &traces);
    let request_count = traces.len();

    let r = h.run(&format!(
        "--model {DEFAULT_MODEL} --url {} --endpoint-type chat \
         --input-file {} --custom-dataset-type mooncake_trace \
         --request-count {} --fixed-schedule --synthesis-speedup-ratio 2.0 \
         --workers-max 1 --ui simple",
        h.mock.url,
        trace_file.display(),
        request_count,
    ));

    assert_eq!(r.artifacts.request_count() as usize, request_count);
    assert!(has_all_outputs(&r));
}
