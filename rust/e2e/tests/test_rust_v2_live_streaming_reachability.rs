// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
mod common;
use common::*;

// Live-worker activation requires the Python sidecar worker and request-capture harness.

#[tokio::test]
#[ignore] // requires: Python live-streaming sidecar worker + Installation execute-capture harness
async fn test_python_config_v2_reaches_live_worker_without_v1_or_early_artifacts() {
    let h = AIPerfHarness::new().await;
    let r = h.run(&format!(
        "--model {DEFAULT_MODEL} --url {} --endpoint-type chat --streaming \
         --concurrency 1 --request-count 2 --ui simple",
        h.mock.url
    ));
    assert!(r.success());
    assert_eq!(r.artifacts.request_count() as u32, 2);
}
