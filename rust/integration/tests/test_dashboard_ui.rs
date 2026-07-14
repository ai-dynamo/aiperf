// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Tests for dashboard UI mode with duration-based termination.
//!
//! Dashboard mode with request-count termination is tested elsewhere:
//! - test_stress.rs::test_high_worker_count_streaming
//! - test_gpu_telemetry.rs
//! - test_server_metrics.rs

#[path = "common/mod.rs"]
mod common;
use common::*;

use aiperf_mock_rs::config::MockServerConfig;

/// Dashboard UI with duration-based benchmark termination produces correct output.
#[tokio::test]
async fn test_duration_based_termination() {
    // Use faster mock server settings for reliability
    let mut cfg = MockServerConfig::default();
    cfg.ttft = 10.0;
    cfg.itl = 5.0;

    let h = AIPerfHarness::new_with(cfg).await;
    let r = h.run(&format!(
        "--model {DEFAULT_MODEL} \
         --url {} \
         --tokenizer gpt2 \
         --endpoint-type chat \
         --ui dashboard \
         --benchmark-duration 5 \
         --benchmark-grace-period 10 \
         --concurrency 3 \
         --image-width-mean 64 \
         --image-height-mean 64 \
         --audio-length-mean 0.1",
        h.mock.url
    ));

    // Verify benchmark completed and CSV contains duration config
    assert!(r.artifacts.request_count() >= 1.0);
    assert!(r.artifacts.csv().contains("Benchmark Duration"));
}
