// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
mod common;
use common::*;

#[tokio::test]
async fn test_basic_embeddings() {
    let h = AIPerfHarness::new().await;
    let r = h.run(&format!(
        "--model nomic-ai/nomic-embed-text-v1.5 \
         --url {} \
         --endpoint-type embeddings \
         --request-count {DEFAULT_REQUEST_COUNT} \
         --concurrency {DEFAULT_CONCURRENCY} \
         --workers-max {DEFAULT_CONCURRENCY} \
         --ui simple",
        h.mock.url
    ));
    assert_eq!(r.artifacts.request_count() as u32, DEFAULT_REQUEST_COUNT);
    assert!(
        r.artifacts
            .json()
            .get("time_to_first_token")
            .map_or(true, |v| v.is_null())
    );
}
