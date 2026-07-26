// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
mod common;
use common::*;

const CONCURRENCY: u32 = 2;
const REQUEST_COUNT: u32 = 8;

/// The harness appends `--artifact-dir` and `--tokenizer`, which override the
/// corresponding config fields.
fn grpc_config(grpc_url: &str, streaming: bool) -> String {
    format!(
        "schemaVersion: \"2.0\"\n\
         benchmark:\n\
        \x20 models: [{DEFAULT_MODEL}]\n\
        \x20 endpoint:\n\
        \x20   urls: [\"{grpc_url}\"]\n\
        \x20   type: kserve_v2_infer\n\
        \x20   streaming: {streaming}\n\
        \x20   waitForModelTimeout: 0.0\n\
        \x20 dataset:\n\
        \x20   type: synthetic\n\
        \x20   entries: {REQUEST_COUNT}\n\
        \x20   prompts:\n\
        \x20     isl: 32\n\
        \x20     osl: 16\n\
        \x20 phases:\n\
        \x20   - name: profiling\n\
        \x20     type: concurrency\n\
        \x20     requests: {REQUEST_COUNT}\n\
        \x20     concurrency: {CONCURRENCY}\n\
        \x20 gpuTelemetry: {{enabled: false}}\n\
        \x20 serverMetrics: {{enabled: false}}\n\
        \x20 transport:\n\
        \x20   type: grpc\n\
        \x20 runtime:\n\
        \x20   ui: none\n"
    )
}

/// Returns the harness to keep its artifact `TempDir` alive for assertions.
async fn run_grpc(streaming: bool) -> (AIPerfHarness, RunResult) {
    let h = AIPerfHarness::new_with_grpc().await;
    let grpc_url = h
        .mock
        .grpc_url
        .clone()
        .expect("mock started with grpc listener");
    let tmp = tempfile::TempDir::new().unwrap();
    let cfg_file = tmp.path().join("kserve_grpc.yaml");
    std::fs::write(&cfg_file, grpc_config(&grpc_url, streaming)).unwrap();

    let r = h.run(&format!("--config {}", cfg_file.display()));
    assert!(
        r.success(),
        "grpc run failed (exit {}):\nstdout:\n{}\nstderr:\n{}",
        r.exit_code,
        r.stdout,
        r.stderr
    );
    (h, r)
}

#[tokio::test]
async fn test_kserve_grpc_unary() {
    let (_h, r) = run_grpc(false).await;
    assert_eq!(r.artifacts.request_count() as u32, REQUEST_COUNT);
}

#[tokio::test]
async fn test_kserve_grpc_streaming() {
    let (_h, r) = run_grpc(true).await;
    assert_eq!(r.artifacts.request_count() as u32, REQUEST_COUNT);
    let json = r.artifacts.json();
    assert!(
        json.get("time_to_first_token")
            .map(|v| !v.is_null())
            .unwrap_or(false),
        "streaming gRPC run should report time_to_first_token"
    );
}
