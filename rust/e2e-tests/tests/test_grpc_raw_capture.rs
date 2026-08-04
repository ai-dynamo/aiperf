// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Product-level proof for the gRPC dispatch-capture gates.
//!
//! `GrpcTransportSink::dispatch_collect` skips two pieces of per-request work
//! whose output no configured artifact would read: the HTTP-compatibility
//! `RequestRecord` (which re-serializes every decoded response) and the canonical
//! `request_payload`. Both are gated on `capture_raw`, the raw artifact being the
//! only consumer of either — `inputs.json` is projected from the resident dataset
//! at finalize and never reads a dispatched body. This test pins every side of
//! both gates against a deterministic `aiperf-mock-server` KServe gRPC endpoint:
//!
//!   * **raw ON** (`--export-level raw`): `profile_export_raw.jsonl` still carries
//!     one record per request, each with its full `responses[]` array and its
//!     `payload` object — proving both gates keep their output byte-populated
//!     when raw capture is on.
//!   * **raw OFF, exact-fold A/B** (`AIPERF_RUNTIME_EXACT_FOLD=0` against the
//!     default): `inputs.json` must be byte-identical on the retain path and the
//!     exact-fold path, neither of which reads a dispatched payload. Both runs
//!     assert the retention marker, so neither can silently drift off the branch
//!     it pins.
//!   * **raw OFF** (default): the run's per-record metrics are unchanged and no
//!     raw artifact is emitted — proving skipping the (discarded) record and
//!     payload does not perturb observable output.

mod common;
use common::*;

const CONCURRENCY: u32 = 2;
const REQUEST_COUNT: u32 = 8;
const ISL: u32 = 32;
const OSL: u32 = 16;
/// Pinned run seed so repeated runs of the same config are byte-comparable.
const SEED: u32 = 4242;

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

/// Run the shared gRPC config at `export_level` (unset for the CLI default).
/// Returns the harness (to keep the artifact `TempDir` alive) and the run result.
async fn run_grpc_at(export_level: Option<&str>) -> (AIPerfHarness, RunResult) {
    run_grpc_on(export_level, true).await
}

/// Run the shared gRPC config, selecting the record-retention path.
///
/// `exact_fold` picks the default fold-and-drop path (`true`), where `inputs.json`
/// is generated up front from the resident dataset and the sink's payload gate is
/// therefore CLOSED, or retained-record execution (`false`, via
/// `AIPERF_RUNTIME_EXACT_FOLD=0` as `test_exact_fold_ab_parity` does), where
/// `inputs.json` is captured from each dispatch and the gate is OPEN.
///
/// `--ui simple` plus `AIPERF_LOG=aiperf=info` are required for the
/// `record retention path selected` marker the callers assert on.
async fn run_grpc_on(export_level: Option<&str>, exact_fold: bool) -> (AIPerfHarness, RunResult) {
    let h = AIPerfHarness::new_with_grpc().await;
    let grpc_url = h
        .mock
        .grpc_url
        .clone()
        .expect("mock started with grpc listener");
    let tmp = tempfile::TempDir::new().unwrap();
    let cfg_file = tmp.path().join("grpc_raw.yaml");
    std::fs::write(&cfg_file, grpc_config(&grpc_url)).unwrap();

    let mut env: Vec<(&str, &str)> = vec![("AIPERF_LOG", "aiperf=info")];
    if !exact_fold {
        env.push(("AIPERF_RUNTIME_EXACT_FOLD", "0"));
    }
    let extra = export_level.map_or_else(String::new, |level| format!(" --export-level {level}"));
    // Pinned so two runs of the same config generate identical synthetic prompts;
    // the A/B below compares their artifacts byte for byte.
    let r = h.run_env(
        &format!(
            "--config {}{extra} --random-seed {SEED} --ui simple",
            cfg_file.display()
        ),
        &env,
    );
    assert!(
        r.success(),
        "grpc run (export_level={export_level:?}, exact_fold={exact_fold}) failed (exit {}):\n\
         stdout:\n{}\nstderr:\n{}",
        r.exit_code,
        r.stdout,
        r.stderr
    );
    (h, r)
}

/// The `record retention path selected` log line, which reports the `exact_fold`
/// the run actually chose. Asserted so these tests fail loudly if a future
/// eligibility change silently moves them off the branch they mean to pin.
fn retention_marker(r: &RunResult) -> String {
    let path = r
        .artifacts
        .find_file("**/aiperf.log")
        .expect("logs/aiperf.log should exist");
    std::fs::read_to_string(&path)
        .unwrap_or_default()
        .lines()
        .find(|line| line.contains("record retention path selected"))
        .unwrap_or("<no retention marker>")
        .to_string()
}

/// Every `inputs.json` payload, flattened across sessions, each asserted to carry
/// its KServe v2 input tensors. An empty canonical payload could not reach here:
/// `write_inputs_json` parses each retained body and fails the export instead.
fn inputs_payloads(r: &RunResult) -> Vec<String> {
    let inputs = r.artifacts.inputs();
    let sessions = inputs
        .get("data")
        .and_then(serde_json::Value::as_array)
        .unwrap_or_else(|| panic!("inputs.json missing data array: {inputs}"))
        .clone();
    assert!(!sessions.is_empty(), "inputs.json captured no sessions");
    let mut payloads = Vec::new();
    for session in &sessions {
        let entries = session
            .get("payloads")
            .and_then(serde_json::Value::as_array)
            .unwrap_or_else(|| panic!("inputs.json session missing payloads: {session}"));
        for payload in entries {
            assert!(
                payload
                    .get("inputs")
                    .and_then(serde_json::Value::as_array)
                    .is_some_and(|inputs| !inputs.is_empty()),
                "inputs.json payload lost its KServe v2 input tensors: {payload}"
            );
            payloads.push(payload.to_string());
        }
    }
    payloads.sort();
    payloads
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

/// `inputs.json` on a run with the payload gate CLOSED, A/B across the two record
/// retention paths — the only test here that runs with raw capture off on both
/// sides.
///
/// A records-level request does NOT by itself disqualify exact-fold:
/// `wants_per_record_artifacts` (`engine/execute/plan.rs`) ignores `records_path`
/// entirely (records stream through `RecordArtifactLane`).
/// `AIPERF_RUNTIME_EXACT_FOLD=0` is what forces the retain path. Both runs assert
/// the retention marker so this cannot silently stop testing what it claims.
///
/// `inputs.json` is projected from the resident dataset on BOTH paths, so the
/// closed gate must be invisible to it. Byte-identity between the two runs is the
/// acceptance oracle — skipping the canonical payload must not change the
/// artifact, on either retention path.
#[tokio::test]
async fn test_grpc_inputs_match_across_retention_paths_without_raw() {
    let (_h_open, open) = run_grpc_on(Some("records"), false).await;
    let (_h_closed, closed) = run_grpc_on(Some("records"), true).await;

    let open_marker = retention_marker(&open);
    assert!(
        open_marker.contains("exact_fold=false"),
        "this run must select the retain path; marker was: {open_marker}"
    );
    let closed_marker = retention_marker(&closed);
    assert!(
        closed_marker.contains("exact_fold=true"),
        "this run must select exact-fold; marker was: {closed_marker}"
    );

    for r in [&open, &closed] {
        assert_eq!(r.artifacts.request_count() as u32, REQUEST_COUNT);
        assert!(
            r.artifacts.raw_records().is_empty(),
            "a records-level run must not emit a raw artifact"
        );
    }

    let open_payloads = inputs_payloads(&open);
    assert_eq!(
        open_payloads.len() as u32,
        REQUEST_COUNT,
        "the retain path must project one canonical payload per dataset turn"
    );
    assert_eq!(
        open_payloads,
        inputs_payloads(&closed),
        "the record retention path changed inputs.json: it is projected from the \
         resident dataset either way, so the two must be byte-identical"
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
