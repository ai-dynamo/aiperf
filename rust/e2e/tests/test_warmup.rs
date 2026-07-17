// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Warmup requests are excluded from profiling metrics.

mod common;
use common::*;

#[tokio::test]
async fn test_warmup_phase() {
    let h = AIPerfHarness::new().await;
    let r = h.run(&format!(
        "--model {DEFAULT_MODEL} \
         --url {} \
         --endpoint-type chat \
         --warmup-request-count 5 \
         --request-count 15 \
         --concurrency {DEFAULT_CONCURRENCY} \
         --workers-max 8 \
         --ui simple",
        h.mock.url
    ));
    assert!(r.success());
    assert_eq!(r.artifacts.request_count() as u32, 15);
}
