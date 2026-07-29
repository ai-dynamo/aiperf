// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Live-streaming reachability.

mod common;
use common::*;

// This body runs green today, but only because it asserts a plain streaming run
// completes — nothing here observes the live-streaming sidecar it is named for.
// Un-ignoring it would add a passing test that cannot fail for the right reason.
// Making it real needs a way to observe records as they are emitted mid-run
// (a results-sidecar subscriber, or execute-mode stdio capture); neither is
// exposed to this harness.
#[tokio::test]
#[ignore = "would pass vacuously: asserts only that a streaming run completes, and \
            nothing here observes live record emission mid-run"]
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
