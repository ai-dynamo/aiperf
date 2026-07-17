// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Live-streaming checks require a sidecar worker and execute-capture harness.

mod common;
use common::*;

#[tokio::test]
#[ignore = "requires live-streaming sidecar worker and execute-capture harness"]
async fn test_config_v2_reaches_live_streaming_worker() {
    let h = AIPerfHarness::new().await;
    let r = h.run(&format!(
        "--model {DEFAULT_MODEL} --url {} --endpoint-type chat --streaming \
         --concurrency 1 --request-count 2 --ui simple",
        h.mock.url
    ));
    assert!(r.success());
    assert_eq!(r.artifacts.request_count() as u32, 2);
}
