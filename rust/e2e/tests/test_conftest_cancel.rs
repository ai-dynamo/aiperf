// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
mod common;
use common::*;

// Platform-specific timeout cancellation requires the Python conftest harness.

#[tokio::test]
#[ignore = "requires Unix SIGINT conftest helper"]
async fn test_cancel_calls_send_signal_sigint_on_unix() {
    let _ = AIPerfHarness::new().await;
}

#[tokio::test]
#[ignore = "requires Windows terminate conftest helper"]
async fn test_cancel_calls_terminate_on_windows() {
    let _ = AIPerfHarness::new().await;
}
