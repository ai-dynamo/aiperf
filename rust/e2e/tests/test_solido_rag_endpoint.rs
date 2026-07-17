// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

mod common;
use common::*;

#[tokio::test]
async fn test_basic_solido_rag() {
    let h = AIPerfHarness::new().await;
    let r = h.run(&format!(
        "--model rag-model --url {} --endpoint-type solido_rag \
         --request-count {DEFAULT_REQUEST_COUNT} --concurrency {DEFAULT_CONCURRENCY} \
         --workers-max {DEFAULT_CONCURRENCY} --ui simple",
        h.mock.url
    ));
    assert!(r.success());
    assert_eq!(r.artifacts.request_count() as u32, DEFAULT_REQUEST_COUNT);
}
