// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
mod common;
use common::*;

/// Test that only providing the model and nothing else still works.
///
/// NOTE: We still have to provide the server's url due to the nature of it
/// being on a non-default port.
#[tokio::test]
async fn test_default_behavior() {
    let h = AIPerfHarness::new().await;
    let r = h.run(&format!("--model {DEFAULT_MODEL} --url {}", h.mock.url));
    assert_eq!(r.artifacts.request_count() as u32, DEFAULT_REQUEST_COUNT);
    assert_eq!(r.exit_code, 0);
}
