// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

mod common;
use common::*;

use aiperf_mock_server::config::MockServerConfig;

// `--ui-type`/`--ui` is accepted by clap (`ProfileFlags::ui_type`) and then only
// warned about — it is listed in `profile.rs`'s `UNIMPLEMENTED_FLAGS`, so there is
// no dashboard renderer to exercise and `--ui dashboard` is inert. This body passes for that reason
// alone, which is why it stays ignored rather than being promoted — its
// remaining assertions are duration-termination, already covered elsewhere.
#[tokio::test]
#[ignore = "no dashboard UI exists to exercise: --ui-type is parsed but never consumed, \
            so this asserts only duration-based termination"]
async fn test_duration_based_termination() {
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

    assert!(r.artifacts.request_count() >= 1.0);
    assert!(r.artifacts.csv().contains("Benchmark Duration"));
}
