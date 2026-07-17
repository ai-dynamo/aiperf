// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
mod common;
use common::*;

/// The endpoint POSTs a prompt plus a reference image as multipart/form-data.
/// `request_content_type` defaults to multipart for `image_edit`.
#[tokio::test]
async fn test_image_edit_produces_no_streaming_metrics() {
    let h = AIPerfHarness::new().await;
    let r = h.run(&format!(
        "--model black-forest-labs/FLUX.2-klein-4B \
         --tokenizer gpt2 \
         --url {} \
         --endpoint-type image_edit \
         --image-batch-size 1 \
         --image-width-mean 64 \
         --image-height-mean 64 \
         --synthetic-input-tokens-mean 50 \
         --synthetic-input-tokens-stddev 10 \
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

#[tokio::test]
async fn test_image_edit_extra_inputs_pass_through() {
    let h = AIPerfHarness::new().await;
    let r = h.run(&format!(
        "--model black-forest-labs/FLUX.2-klein-4B \
         --tokenizer gpt2 \
         --url {} \
         --endpoint-type image_edit \
         --image-batch-size 1 \
         --image-width-mean 64 \
         --image-height-mean 64 \
         --extra-inputs size:512x512 num_inference_steps:4 guidance_scale:1.0 \
         --request-count 10 \
         --concurrency 2 \
         --workers-max 1 \
         --ui simple",
        h.mock.url
    ));
    assert!(r.success());
    assert_eq!(r.artifacts.request_count() as u32, 10);

    let json = r.artifacts.json();
    assert!(!json["request_latency"].is_null());
    assert!(!json["request_throughput"].is_null());
}
