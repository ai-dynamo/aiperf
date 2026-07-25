// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Product-level proof for the gRPC HTTP-compatibility-record gate.
//!
//! `GrpcTransportSink::dispatch_collect` builds an HTTP-compatibility
//! `RequestRecord` (re-serializing every decoded response) only when a raw
//! artifact will consume it — the `raw_enabled` flag threaded into the sink. This
//! test pins both sides of that gate against a deterministic `aiperf-mock-server`
//! KServe gRPC endpoint:
//!
//!   * **raw ON** (`--export-level raw`): `profile_export_raw.jsonl` still carries
//!     one record per request, each with its full `responses[]` array — proving
//!     the gate keeps the record byte-populated when raw capture is on.
//!   * **raw OFF** (default): the run's per-record metrics are unchanged and no
//!     raw artifact is emitted — proving skipping the (discarded) record does not
//!     perturb observable output.

mod common;
use common::*;

const CONCURRENCY: u32 = 2;
const REQUEST_COUNT: u32 = 8;
const ISL: u32 = 32;
const OSL: u32 = 16;

/// A deterministic KServe gRPC streaming benchmark. The harness appends
/// `--artifact-dir` and `--tokenizer`, overriding the corresponding fields.
fn grpc_config(grpc_url: &str) -> String {
    format!(
        "schemaVersion: \"2.0\"\n\
         benchmark:\n\
        \x20 models: [{DEFAULT_MODEL}]\n\
        \x20 endpoint:\n\
        \x20   urls: [\"{grpc_url}\"]\n\
        \x20   type: kserve_v2_infer\n\
        \x20   streaming: true\n\
        \x20   waitForModelTimeout: 0.0\n\
        \x20 dataset:\n\
        \x20   type: synthetic\n\
        \x20   entries: {REQUEST_COUNT}\n\
        \x20   prompts:\n\
        \x20     isl: {ISL}\n\
        \x20     osl: {OSL}\n\
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

/// Run the shared gRPC config, optionally requesting raw artifacts. Returns the
/// harness (to keep the artifact `TempDir` alive) and the run result.
async fn run_grpc(raw: bool) -> (AIPerfHarness, RunResult) {
    let h = AIPerfHarness::new_with_grpc().await;
    let grpc_url = h
        .mock
        .grpc_url
        .clone()
        .expect("mock started with grpc listener");
    let tmp = tempfile::TempDir::new().unwrap();
    let cfg_file = tmp.path().join("grpc_raw.yaml");
    std::fs::write(&cfg_file, grpc_config(&grpc_url)).unwrap();

    let extra = if raw { " --export-level raw" } else { "" };
    let r = h.run(&format!("--config {}{extra}", cfg_file.display()));
    assert!(
        r.success(),
        "grpc run (raw={raw}) failed (exit {}):\nstdout:\n{}\nstderr:\n{}",
        r.exit_code,
        r.stdout,
        r.stderr
    );
    (h, r)
}

/// With raw capture ON, the gate builds the compatibility record: every request
/// appears in `profile_export_raw.jsonl` carrying a non-empty `responses[]` array
/// (a streamed KServe run yields multiple decoded responses per record — exactly
/// the per-response re-serialization path the gate governs).
#[tokio::test]
async fn test_grpc_raw_capture_records_responses() {
    let (_h, r) = run_grpc(true).await;
    assert_eq!(r.artifacts.request_count() as u32, REQUEST_COUNT);

    let raw = r.artifacts.raw_records();
    assert_eq!(
        raw.len() as u32,
        REQUEST_COUNT,
        "raw capture must emit one raw record per request"
    );
    for (i, record) in raw.iter().enumerate() {
        let responses = record
            .get("responses")
            .and_then(serde_json::Value::as_array)
            .unwrap_or_else(|| panic!("raw record {i} missing responses array: {record}"));
        assert!(
            !responses.is_empty(),
            "raw record {i} has an empty responses array — the gRPC compatibility \
             record dropped its responses: {record}"
        );
    }
}

/// With raw capture OFF (the default), the gate skips building the discarded
/// record. The run still completes and reports the same per-record metrics, and
/// no raw artifact is written.
#[tokio::test]
async fn test_grpc_default_run_skips_raw_but_keeps_metrics() {
    let (_h, r) = run_grpc(false).await;
    assert_eq!(
        r.artifacts.request_count() as u32,
        REQUEST_COUNT,
        "skipping the compatibility record must not perturb the request metrics"
    );
    // Streaming metrics still derive from the native response loop, not the
    // (now-skipped) compatibility record.
    let json = r.artifacts.json();
    assert!(
        json.get("time_to_first_token")
            .map(|v| !v.is_null())
            .unwrap_or(false),
        "streaming gRPC run should still report time_to_first_token without raw capture"
    );
    assert!(
        r.artifacts.raw_records().is_empty(),
        "no raw artifact should be produced on the default (no-raw) run"
    );
}
