// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
mod common;
use common::*;

fn has_streaming_metrics(json: &serde_json::Value) -> bool {
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
async fn test_huggingface_generate_non_streaming() {
    let h = AIPerfHarness::new().await;
    let r = h.run(&format!(
        "--model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --url {} \
         --endpoint-type huggingface_generate --request-count 10 --ui simple",
        h.mock.url
    ));
    assert!(r.success());
    assert_eq!(r.artifacts.request_count() as u32, 10);
    assert!(!has_streaming_metrics(&r.artifacts.json()));
}

#[tokio::test]
async fn test_huggingface_generate_streaming() {
    let h = AIPerfHarness::new().await;
    let r = h.run(&format!(
        "--model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --url {} \
         --endpoint-type huggingface_generate --streaming --request-count 10 --ui simple",
        h.mock.url
    ));
    assert!(r.success());
    assert_eq!(r.artifacts.request_count() as u32, 10);
    assert!(has_streaming_metrics(&r.artifacts.json()));
}
