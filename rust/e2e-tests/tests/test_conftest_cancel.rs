// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
mod common;
use common::*;

// These two asserted that the *Python test harness* dispatched the right
// platform signal when its own run timeout expired — a property of the harness,
// not of aiperf. The Rust harness has no equivalent timeout-cancel path to
// assert against, and the product-level behavior they stood in for (SIGINT
// producing a graceful cancel with written results) is covered live by
// test_ctrl_c_cancellation.rs.

#[tokio::test]
#[ignore = "harness-internal: no Rust equivalent of the Python conftest timeout-cancel path; \
            product SIGINT behavior is covered by test_ctrl_c_cancellation.rs"]
async fn test_cancel_calls_send_signal_sigint_on_unix() {
    let _ = AIPerfHarness::new().await;
}

#[tokio::test]
#[ignore = "harness-internal, and windows-only: the suite drives POSIX signals and \
            `nix` is gated to cfg(unix), so there is no terminate path to assert"]
async fn test_cancel_calls_terminate_on_windows() {
    let _ = AIPerfHarness::new().await;
}
