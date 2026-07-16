// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
mod common;
use common::*;

// Config v2 -> v2 runner -> post-artifact live-worker activation proof.
//
// The Python original drives the orchestrator internals directly: it injects a
// synthetic `aiperf.post_processors.native_streaming_worker` module onto
// `PYTHONPATH`, monkeypatches `Installation.execute` to capture the exact
// protocol-v2 request the Python orchestrator emits, spins up a hand-rolled
// streaming `/v1/chat/completions` HTTP server, and asserts that the runner
// reaches the live-streaming sidecar worker only after artifacts are committed.
// None of that machinery (the fixture worker module, `Installation`
// request capture/monkeypatch, the `live_streaming` sidecar activation proof)
// is reachable through the black-box `AIPerfHarness` CLI surface, so the port
// is marked `#[ignore]` and kept as a behavioral placeholder.

#[tokio::test]
#[ignore] // requires: Python live-streaming sidecar worker + Installation execute-capture harness
async fn test_python_config_v2_reaches_live_worker_without_v1_or_early_artifacts() {
    let h = AIPerfHarness::new().await;
    let r = h.run(&format!(
        "--model {DEFAULT_MODEL} --url {} --endpoint-type chat --streaming \
         --concurrency 1 --request-count 2 --ui simple",
        h.mock.url
    ));
    assert!(r.success());
    assert_eq!(r.artifacts.request_count() as u32, 2);
}
