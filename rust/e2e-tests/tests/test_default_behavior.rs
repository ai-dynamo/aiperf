// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
mod common;
use common::*;

#[tokio::test]
async fn test_default_behavior() {
    let h = AIPerfHarness::new().await;
    let r = h.run(&format!("--model {DEFAULT_MODEL} --url {}", h.mock.url));
    assert_eq!(r.artifacts.request_count() as u32, DEFAULT_REQUEST_COUNT);
    assert_eq!(r.exit_code, 0);
}
