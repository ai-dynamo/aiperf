// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
mod common;
use common::*;

#[tokio::test]
async fn test_basic_image_retrieval() {
    let h = AIPerfHarness::new().await;
    let r = h.run(&format!(
        "--model nvidia/page-elements-v2 \
         --url {} \
         --endpoint-type image_retrieval \
         --endpoint /v1/image/infer \
         --image-width-mean 64 \
         --image-height-mean 64 \
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
    assert!(!json["request_latency"].is_null());
    assert!(!json["request_throughput"].is_null());
    assert!(!json["image_throughput"].is_null());
    assert!(!json["image_latency"].is_null());
}
