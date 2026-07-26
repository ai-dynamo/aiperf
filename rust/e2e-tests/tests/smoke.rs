// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

mod common;

use common::{DEFAULT_MODEL, MockServer};

#[tokio::test]
async fn mock_server_boots_and_serves_health() {
    let server = MockServer::start();
    assert!(server.url.starts_with("http://127.0.0.1:"));
    assert_eq!(server.dcgm_urls().len(), 2);
    assert!(server.server_metrics_urls().contains_key("vllm"));

    let body = reqwest::get(format!("{}/health", server.url))
        .await
        .expect("GET /health")
        .text()
        .await
        .expect("read /health body");
    assert!(body.contains("healthy"), "unexpected health body: {body}");

    assert_eq!(DEFAULT_MODEL, "openai/gpt-oss-120b");
}
