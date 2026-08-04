// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Product-level proof for the gRPC dispatch-capture gates.
//!
//! `GrpcTransportSink::dispatch_collect` skips two pieces of per-request work
//! whose output no configured artifact would read: the HTTP-compatibility
//! `RequestRecord` (which re-serializes every decoded response) and the canonical
//! `request_payload`. The record is gated on `capture_raw` alone; the payload is
//! gated on `capture_raw || inputs_enabled`, because it feeds both the raw
//! artifact and `inputs.json`. This test pins every side of both gates against a
//! deterministic `aiperf-mock-server` KServe gRPC endpoint:
//!
//!   * **raw ON** (`--export-level raw`): `profile_export_raw.jsonl` still carries
//!     one record per request, each with its full `responses[]` array and its
//!     `payload` object — proving both gates keep their output byte-populated
//!     when raw capture is on.
//!   * **records, raw OFF** (`--export-level records`): per-record artifacts force
//!     the retain path, so `inputs.json` is captured during dispatch rather than
//!     generated up front. Its payloads must still be populated — this is the case
//!     a payload gate on `capture_raw` alone would break.
//!   * **raw OFF** (default): the run's per-record metrics are unchanged and no
//!     raw artifact is emitted — proving skipping the (discarded) record and
//!     payload does not perturb observable output.

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

/// Run the shared gRPC config at `export_level` (unset for the default summary
/// run). Returns the harness (to keep the artifact `TempDir` alive) and the run
/// result.
async fn run_grpc_at(export_level: Option<&str>) -> (AIPerfHarness, RunResult) {
    let h = AIPerfHarness::new_with_grpc().await;
    let grpc_url = h
        .mock
        .grpc_url
        .clone()
        .expect("mock started with grpc listener");
    let tmp = tempfile::TempDir::new().unwrap();
    let cfg_file = tmp.path().join("grpc_raw.yaml");
    std::fs::write(&cfg_file, grpc_config(&grpc_url)).unwrap();

    let extra = export_level.map_or_else(String::new, |level| format!(" --export-level {level}"));
    let r = h.run(&format!("--config {}{extra}", cfg_file.display()));
    assert!(
        r.success(),
        "grpc run (export_level={export_level:?}) failed (exit {}):\nstdout:\n{}\nstderr:\n{}",
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
    let (_h, r) = run_grpc_at(Some("raw")).await;
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
        // The `payload` column is the sink's canonical `request_payload`. An empty
        // one would have aborted the export (the writer parses it as JSON), so this
        // pins the populated shape rather than merely a present key.
        let payload = record
            .get("payload")
            .and_then(serde_json::Value::as_object)
            .unwrap_or_else(|| panic!("raw record {i} missing request payload object: {record}"));
        assert!(
            !payload.is_empty(),
            "raw record {i} carries an empty request payload — the payload gate \
             dropped it while raw capture was on: {record}"
        );
        assert!(
            payload
                .get("inputs")
                .and_then(serde_json::Value::as_array)
                .is_some_and(|inputs| !inputs.is_empty()),
            "raw record {i} request payload lost its KServe v2 input tensors: {record}"
        );
    }
}

/// Per-record artifacts force the retain path, so `inputs.json` is captured from
/// each dispatch rather than generated up front from the resident dataset. The
/// canonical payload must therefore still be built even though raw capture is
/// off — a payload gated on `capture_raw` alone would empty every entry here
/// (in practice: abort the export, since an empty payload does not parse).
#[tokio::test]
async fn test_grpc_records_run_retains_inputs_payloads_without_raw() {
    let (_h, r) = run_grpc_at(Some("records")).await;
    assert_eq!(r.artifacts.request_count() as u32, REQUEST_COUNT);
    assert!(
        r.artifacts.raw_records().is_empty(),
        "a records-level run must not emit a raw artifact"
    );

    let inputs = r.artifacts.inputs();
    let sessions = inputs
        .get("data")
        .and_then(serde_json::Value::as_array)
        .unwrap_or_else(|| panic!("inputs.json missing data array: {inputs}"));
    assert!(!sessions.is_empty(), "inputs.json captured no sessions");
    let mut payload_count = 0;
    for session in sessions {
        let payloads = session
            .get("payloads")
            .and_then(serde_json::Value::as_array)
            .unwrap_or_else(|| panic!("inputs.json session missing payloads: {session}"));
        for payload in payloads {
            let payload = payload
                .as_object()
                .unwrap_or_else(|| panic!("inputs.json payload is not an object: {payload}"));
            assert!(
                payload
                    .get("inputs")
                    .and_then(serde_json::Value::as_array)
                    .is_some_and(|inputs| !inputs.is_empty()),
                "inputs.json payload lost its KServe v2 input tensors: {payload:?}"
            );
            payload_count += 1;
        }
    }
    assert_eq!(
        payload_count, REQUEST_COUNT,
        "inputs.json must retain one canonical payload per dispatched turn"
    );
}

/// With raw capture OFF (the default), the gate skips building the discarded
/// record. The run still completes and reports the same per-record metrics, and
/// no raw artifact is written.
#[tokio::test]
async fn test_grpc_default_run_skips_raw_but_keeps_metrics() {
    let (_h, r) = run_grpc_at(None).await;
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
