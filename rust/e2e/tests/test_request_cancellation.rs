// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
mod common;
use common::*;

/// Request cancellation doesn't break pipeline.
#[tokio::test]
async fn test_request_cancellation() {
    let h = AIPerfHarness::new().await;
    let timeout = if cfg!(target_os = "windows") {
        300
    } else {
        120
    };
    let r = h.run_timeout(
        &format!(
            "--model {DEFAULT_MODEL} --url {} --endpoint-type chat --streaming \
             --request-count 50 --concurrency 5 --random-seed 42 \
             --image-width-mean 64 --image-height-mean 64 --osl 100000 \
             --request-cancellation-rate 30 --request-cancellation-delay 0 \
             --ui simple",
            h.mock.url
        ),
        timeout,
    );

    for request in r.artifacts.jsonl() {
        let was_cancelled = request["metadata"]["was_cancelled"]
            .as_bool()
            .unwrap_or(false);
        if was_cancelled {
            assert!(!request["error"].is_null());
            assert_eq!(request["error"]["code"].as_i64().unwrap(), 499);
            assert_eq!(
                request["error"]["type"].as_str().unwrap(),
                "RequestCancellationError"
            );
            let error_isl = &request["metrics"]["error_isl"];
            assert!(!error_isl.is_null());
            assert!(error_isl["value"].as_f64().unwrap() > 0.0);
        }
    }

    let json = r.artifacts.json();
    assert_eq!(json["was_cancelled"].as_bool(), Some(false));
    let error_summary = &json["error_summary"];
    assert!(!error_summary.is_null());
    let summary = error_summary.as_array().unwrap();
    assert!(!summary.is_empty());
    assert!(summary[0]["count"].as_i64().unwrap() > 0);
    assert_eq!(summary[0]["error_details"]["code"].as_i64().unwrap(), 499);
    assert_eq!(
        summary[0]["error_details"]["type"].as_str().unwrap(),
        "RequestCancellationError"
    );
}
