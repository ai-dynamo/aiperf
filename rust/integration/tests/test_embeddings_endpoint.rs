// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#[path = "common/mod.rs"]
mod common;
use common::*;

// Tests for /v1/embeddings endpoint.

/// Basic embeddings request completes with expected request count.
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
    // Embeddings are non-streaming, so streaming metrics should not be present.
    assert!(
        r.artifacts
            .json()
            .get("time_to_first_token")
            .map_or(true, |v| v.is_null())
    );
}
