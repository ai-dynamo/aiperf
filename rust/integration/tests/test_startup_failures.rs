// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for startup failure scenarios.
//!
//! These tests verify that aiperf exits with a non-zero exit code (rather than
//! hanging) when services fail during startup or configuration.

#[path = "common/mod.rs"]
mod common;
use common::*;

/// Test that an invalid model name causes aiperf to exit with an error.
///
/// This test verifies the fail-fast behavior: when a service (like
/// DatasetManager) fails during configuration, the system should exit promptly
/// with a non-zero exit code rather than hanging indefinitely.
#[tokio::test]
async fn test_invalid_model_name_exits_with_error() {
    let h = AIPerfHarness::new().await;
    let r = h.run_timeout(
        &format!(
            "--model this-model-does-not-exist-and-will-fail \
             --url {} \
             --request-count 10 \
             --concurrency 2",
            h.mock.url
        ),
        60,
    );
    assert_ne!(
        r.exit_code, 0,
        "Expected non-zero exit code when model/tokenizer fails to load"
    );
}
