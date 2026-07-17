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
async fn test_basic_completions() {
    let h = AIPerfHarness::new().await;
    let r = h.run(&format!(
        "--model {DEFAULT_MODEL} --url {} --endpoint-type completions \
         --request-count 10 --concurrency 2 --workers-max 1 --ui simple",
        h.mock.url
    ));
    assert!(r.success());
    assert_eq!(r.artifacts.request_count() as u32, 10);
}

#[tokio::test]
async fn test_streaming_completions() {
    let h = AIPerfHarness::new().await;
    let r = h.run(&format!(
        "--model {DEFAULT_MODEL} --url {} --endpoint-type completions --streaming \
         --request-count 10 --concurrency 2 --workers-max 1 --ui simple",
        h.mock.url
    ));
    assert!(r.success());
    assert_eq!(r.artifacts.request_count() as u32, 10);
    assert!(has_streaming_metrics(&r.artifacts.json()));
}
