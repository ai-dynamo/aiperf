// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
mod common;
use common::*;

#[tokio::test]
async fn test_same_seed_identical_inputs() {
    let h1 = AIPerfHarness::new().await;
    let r1 = h1.run(&format!(
        "--model {DEFAULT_MODEL} --url {} --endpoint-type chat \
         --request-count {DEFAULT_REQUEST_COUNT} --concurrency 2 --random-seed 42 \
         --image-width-mean 64 --image-height-mean 64 --audio-length-mean 0.1 \
         --workers-max 5 --ui simple",
        h1.mock.url
    ));
    assert!(r1.success(), "run1 failed: {}", r1.stderr);

    let h2 = AIPerfHarness::new().await;
    let r2 = h2.run(&format!(
        "--model {DEFAULT_MODEL} --url {} --endpoint-type chat \
         --request-count {DEFAULT_REQUEST_COUNT} --concurrency 2 --random-seed 42 \
         --image-width-mean 64 --image-height-mean 64 --audio-length-mean 0.1 \
         --workers-max 5 --ui simple",
        h2.mock.url
    ));
    assert!(r2.success(), "run2 failed: {}", r2.stderr);

    assert_eq!(r1.artifacts.request_count() as u32, DEFAULT_REQUEST_COUNT);
    assert_eq!(r2.artifacts.request_count() as u32, DEFAULT_REQUEST_COUNT);

    let inputs_1 = r1.artifacts.inputs();
    let inputs_2 = r2.artifacts.inputs();

    let data_1 = inputs_1["data"].as_array().expect("inputs_1 data array");
    let data_2 = inputs_2["data"].as_array().expect("inputs_2 data array");

    assert_eq!(data_1.len(), data_2.len(), "Session counts differ");

    for (s1, s2) in data_1.iter().zip(data_2.iter()) {
        assert_eq!(s1["payloads"], s2["payloads"]);
    }
}

#[tokio::test]
async fn test_different_seeds_different_inputs() {
    let h1 = AIPerfHarness::new().await;
    let r1 = h1.run(&format!(
        "--model {DEFAULT_MODEL} --url {} --endpoint-type chat \
         --request-count {DEFAULT_REQUEST_COUNT} --concurrency 2 --random-seed 42 \
         --image-width-mean 128 --image-height-mean 128 --workers-max 5 --ui simple",
        h1.mock.url
    ));
    assert!(r1.success(), "run1 failed: {}", r1.stderr);

    let h2 = AIPerfHarness::new().await;
    let r2 = h2.run(&format!(
        "--model {DEFAULT_MODEL} --url {} --endpoint-type chat \
         --request-count {DEFAULT_REQUEST_COUNT} --concurrency 2 --random-seed 123 \
         --image-width-mean 128 --image-height-mean 128 --workers-max 5 --ui simple",
        h2.mock.url
    ));
    assert!(r2.success(), "run2 failed: {}", r2.stderr);

    assert_eq!(r1.artifacts.request_count() as u32, DEFAULT_REQUEST_COUNT);
    assert_eq!(r2.artifacts.request_count() as u32, DEFAULT_REQUEST_COUNT);

    let inputs_1 = r1.artifacts.inputs();
    let inputs_2 = r2.artifacts.inputs();

    let data_1 = inputs_1["data"].as_array().expect("inputs_1 data array");
    let data_2 = inputs_2["data"].as_array().expect("inputs_2 data array");

    let mut payloads_different = false;
    for (s1, s2) in data_1.iter().zip(data_2.iter()) {
        if s1["payloads"] != s2["payloads"] {
            payloads_different = true;
            break;
        }
    }

    assert!(
        payloads_different,
        "Different seeds should produce different payloads"
    );
}
