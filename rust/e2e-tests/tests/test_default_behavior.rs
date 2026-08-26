// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
mod common;
use common::*;

/// A minimal invocation -- model, url, endpoint type, nothing else -- runs and
/// lands on the default request count.
///
/// `--endpoint-type` is part of the minimum: it has no default anywhere in the
/// product, and `cli/src/load.rs` bails with "--endpoint-type is required" without it.
#[tokio::test]
async fn test_default_behavior() {
    let h = AIPerfHarness::new().await;
    let r = h.run(&format!(
        "--model {DEFAULT_MODEL} --url {} --endpoint-type chat",
        h.mock.url
    ));
    // Exit code first: a nonzero exit makes the count trivially `0`, so
    // asserting the count first reports `left: 0, right: 10` and hides the
    // actual CLI error.
    assert_eq!(
        r.exit_code, 0,
        "default-behavior run failed:\nstdout:\n{}\nstderr:\n{}",
        r.stdout, r.stderr
    );
    assert_eq!(r.artifacts.request_count() as u32, DEFAULT_REQUEST_COUNT);
}
