// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#[path = "common/mod.rs"]
mod common;
use common::*;

// Regression test for Bug 7: subprocess.send_signal(SIGINT) is unsupported on
// Windows.
//
// The original Python tests exercise `tests.integration.conftest._cancel_aiperf_for_timeout`,
// a Python-only test-harness helper that sends SIGINT (graceful) on Unix or
// terminate() (hard kill) on Windows. That helper and its `MagicMock`/`patch`
// machinery do not exist in the Rust harness (the Rust `AIPerfHarness` owns its
// own timeout/cancellation path), so there is nothing behaviorally equivalent
// to port. The tests are retained as ignored placeholders for parity tracking.

// requires: Python test-harness conftest (_cancel_aiperf_for_timeout, MagicMock/patch)
#[tokio::test]
#[ignore]
async fn test_cancel_calls_send_signal_sigint_on_unix() {
    let _ = AIPerfHarness::new().await;
}

// requires: Python test-harness conftest (_cancel_aiperf_for_timeout, MagicMock/patch)
#[tokio::test]
#[ignore]
async fn test_cancel_calls_terminate_on_windows() {
    let _ = AIPerfHarness::new().await;
}
