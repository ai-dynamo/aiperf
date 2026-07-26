// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
mod common;
use common::*;

/// Unlike text generation endpoints, image generation does not produce
/// token-streaming metrics because the image is returned as one response.
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

    assert!(json["time_to_first_token"].is_null());
    assert!(json["inter_token_latency"].is_null());
    assert!(json["time_to_second_token"].is_null());

    assert!(!json["request_latency"].is_null());
    assert!(!json["request_throughput"].is_null());
}
