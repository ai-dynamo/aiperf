// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
mod common;
use common::*;

use std::path::{Path, PathBuf};

fn create_rankings_dataset(dir: &Path, num_entries: usize) -> PathBuf {
    let mut records = Vec::new();
    for i in 0..num_entries {
        records.push(serde_json::json!({
            "texts": [
                {"name": "query", "contents": [format!("What is AI topic {i}?")]},
                {"name": "passages", "contents": [format!("AI passage {i}")]},
            ]
        }));
    }
    write_jsonl(dir, "rankings.jsonl", &records)
}

#[tokio::test]
async fn test_rankings_with_custom_dataset_nim_rankings() {
    rankings_with_custom_dataset("nim_rankings").await;
}

#[tokio::test]
async fn test_rankings_with_custom_dataset_hf_tei_rankings() {
    rankings_with_custom_dataset("hf_tei_rankings").await;
}

#[tokio::test]
async fn test_rankings_with_custom_dataset_cohere_rankings() {
    rankings_with_custom_dataset("cohere_rankings").await;
}

async fn rankings_with_custom_dataset(endpoint_type: &str) {
    let h = AIPerfHarness::new().await;
    let dataset_path = create_rankings_dataset(h.artifact_dir.path(), 5);

    let r = h.run(&format!(
        "--model test-reranker --url {} --endpoint-type {endpoint_type} \
         --input-file {} --custom-dataset-type single_turn \
         --request-count 10 --concurrency 2 --workers-max 1 --ui simple",
        h.mock.url,
        dataset_path.display(),
    ));

    assert!(r.success(), "stderr: {}", r.stderr);
    assert_eq!(r.artifacts.request_count() as u32, 10);
}

#[tokio::test]
async fn test_rankings_with_synthetic_data() {
    let h = AIPerfHarness::new().await;

    let r = h.run(&format!(
        "--model test-reranker --url {} --endpoint-type nim_rankings \
         --request-count 10 --concurrency 2 --workers-max 1 \
         --rankings-passages-mean 6 --rankings-passages-stddev 2 \
         --rankings-passages-prompt-token-mean 32 --rankings-passages-prompt-token-stddev 8 \
         --rankings-query-prompt-token-mean 16 --rankings-query-prompt-token-stddev 4 \
         --ui simple",
        h.mock.url,
    ));

    assert!(r.success(), "stderr: {}", r.stderr);
    assert_eq!(r.artifacts.request_count() as u32, 10);
}
