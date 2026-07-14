// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#[path = "common/mod.rs"]
mod common;
use common::*;

// Tests for /v1/images/generations endpoint.
// Based on: docs/tutorials/sglang-image-generation.md

/// Image generation completes requests without token-based streaming metrics.
///
/// Unlike text generation endpoints, image generation does not produce
/// time-to-first-token or inter-token-latency metrics since there is no
/// token streaming - the entire image is returned as a single response.
#[tokio::test]
async fn test_image_generation_produces_no_streaming_metrics() {
    let h = AIPerfHarness::new().await;
    let r = h.run(&format!(
        "--model black-forest-labs/FLUX.1-dev \
         --url {} \
         --endpoint-type image_generation \
         --synthetic-input-tokens-mean 150 \
         --synthetic-input-tokens-stddev 30 \
         --request-count 10 \
         --concurrency 2 \
         --workers-max 1 \
         --ui simple",
        h.mock.url
    ));
    assert!(r.success());
    assert_eq!(r.artifacts.request_count() as u32, 10);

    let json = r.artifacts.json();

    // Image generation should not have token-based streaming metrics
    assert!(json["time_to_first_token"].is_null());
    assert!(json["inter_token_latency"].is_null());
    assert!(json["time_to_second_token"].is_null());

    // But should have basic request metrics
    assert!(!json["request_latency"].is_null());
    assert!(!json["request_throughput"].is_null());
}
