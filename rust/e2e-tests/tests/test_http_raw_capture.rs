// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Product-level proof for the HTTP dispatch-capture gate.
//!
//! The HTTP counterpart of `test_grpc_raw_capture`. `TransportSink` takes the
//! canonical `request_payload` handle only when an artifact will read it back —
//! `capture_raw` — because the assembled body is a promotable `Bytes`
//! (`BytesMut::freeze()`-derived, `len == capacity`), so an unconditional second
//! handle heap-allocates a shared control block on every dispatch of every run,
//! including the runs that export no raw artifact.
//!
//! The raw artifact is the payload's only consumer: it writes the payload verbatim
//! (`engine/records.rs::write_raw_record_jsonl_row`), while `inputs.json` is
//! projected from the resident dataset at finalize
//! (`engine/execute/compose_sidecars.rs::build_up_front_input_sessions`) and never
//! reads a dispatched body. This test pins every side against a deterministic
//! `aiperf-mock-server` chat endpoint:
//!
//!   * **raw ON** (`--export-level raw`): every `profile_export_raw.jsonl` record
//!     carries a populated `payload` object — the gate keeps its output
//!     byte-populated when raw capture is on.
//!   * **raw OFF, exact-fold A/B** (`AIPERF_RUNTIME_EXACT_FOLD=0` against the
//!     default): `inputs.json` must be byte-identical on the retain path and the
//!     exact-fold path, with the payload gate CLOSED on both. Both runs assert the
//!     retention marker so neither can silently drift off the branch it pins.
//!   * **default**: no raw artifact, unchanged per-record metrics, and an
//!     `inputs.json` byte-identical to the raw run's — closing the gate perturbs
//!     no observable output.
//!
//! Scope note: these three cases catch a wrongly-CLOSED gate (artifact loss). A
//! wrongly-OPEN gate produces byte-identical artifacts and is therefore invisible
//! to any product test; it is pinned at the seam that decides it, by
//! `request_payload_is_taken_only_when_an_artifact_consumes_it` in
//! `runtime/src/transport/http/sink/endpoint_dispatch.rs`.

mod common;
use common::*;

const CONCURRENCY: u32 = 2;
const REQUEST_COUNT: u32 = 8;
const ISL: u32 = 32;
const OSL: u32 = 16;
/// Pinned run seed so repeated runs of the same config are byte-comparable.
const SEED: u32 = 4243;

/// A deterministic single-turn streaming chat benchmark. The harness appends
/// `--artifact-dir` and `--tokenizer`, overriding the corresponding fields.
///
/// `workers: 1` forces the single-thread scheduled path, the only path exact-fold
/// is eligible on — the `AIPERF_RUNTIME_EXACT_FOLD=0` A/B below depends on it.
fn http_config(url: &str) -> String {
    format!(
        "schemaVersion: \"2.0\"\n\
         benchmark:\n\
        \x20 models: [{DEFAULT_MODEL}]\n\
        \x20 endpoint:\n\
        \x20   urls: [\"{url}\"]\n\
        \x20   type: chat\n\
        \x20   streaming: true\n\
        \x20 dataset:\n\
        \x20   type: synthetic\n\
        \x20   entries: {REQUEST_COUNT}\n\
        \x20   random_seed: {SEED}\n\
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
         runtime:\n\
        \x20 ui: none\n\
        \x20 workers: 1\n"
    )
}

/// Run the shared chat config at `export_level` (unset for the CLI default).
async fn run_http_at(export_level: Option<&str>) -> (AIPerfHarness, RunResult) {
    run_http_on(export_level, true).await
}

/// Run the shared chat config, selecting the record-retention path.
///
/// `exact_fold` picks the default fold-and-drop path (`true`), where `inputs.json`
/// is generated up front from the resident dataset and the sink's payload gate is
/// therefore CLOSED, or retained-record execution (`false`, via
/// `AIPERF_RUNTIME_EXACT_FOLD=0` as `test_exact_fold_ab_parity` does), where
/// `inputs.json` is captured from each dispatch and the gate is OPEN.
///
/// `--ui simple` plus `AIPERF_LOG=aiperf=info` are required for the
/// `record retention path selected` marker the callers assert on.
async fn run_http_on(export_level: Option<&str>, exact_fold: bool) -> (AIPerfHarness, RunResult) {
    let h = AIPerfHarness::new().await;
    let url = h.mock.url.clone();
    let tmp = tempfile::TempDir::new().unwrap();
    let cfg_file = tmp.path().join("http_raw.yaml");
    std::fs::write(&cfg_file, http_config(&url)).unwrap();

    let mut env: Vec<(&str, &str)> = vec![("AIPERF_LOG", "aiperf=info")];
    if !exact_fold {
        env.push(("AIPERF_RUNTIME_EXACT_FOLD", "0"));
    }
    let extra = export_level.map_or_else(String::new, |level| format!(" --export-level {level}"));
    let r = h.run_env(
        &format!(
            "--config {}{extra} --random-seed {SEED} --ui simple",
            cfg_file.display()
        ),
        &env,
    );
    assert!(
        r.success(),
        "http run (export_level={export_level:?}, exact_fold={exact_fold}) failed (exit {}):\n\
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
/// its chat message array. An empty canonical payload could not reach here:
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
                    .get("messages")
                    .and_then(serde_json::Value::as_array)
                    .is_some_and(|messages| !messages.is_empty()),
                "inputs.json payload lost its chat messages: {payload}"
            );
            payloads.push(payload.to_string());
        }
    }
    payloads.sort();
    payloads
}

/// With raw capture ON the gate takes the payload: every request appears in
/// `profile_export_raw.jsonl` carrying a populated `payload` object.
#[tokio::test]
async fn test_http_raw_capture_records_request_payloads() {
    let (_h, r) = run_http_at(Some("raw")).await;
    assert_eq!(r.artifacts.request_count() as u32, REQUEST_COUNT);

    let raw = r.artifacts.raw_records();
    assert_eq!(
        raw.len() as u32,
        REQUEST_COUNT,
        "raw capture must emit one raw record per request"
    );
    for (i, record) in raw.iter().enumerate() {
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
                .get("messages")
                .and_then(serde_json::Value::as_array)
                .is_some_and(|messages| !messages.is_empty()),
            "raw record {i} request payload lost its chat messages: {record}"
        );
    }
}

/// `inputs.json` on a run with the payload gate CLOSED, A/B across the two record
/// retention paths — the only case here that runs with raw capture OFF on both
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
async fn test_http_inputs_match_across_retention_paths_without_raw() {
    let (_h_open, open) = run_http_on(Some("records"), false).await;
    let (_h_closed, closed) = run_http_on(Some("records"), true).await;

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

/// With the gate fully CLOSED (the default: no raw artifact, and exact-fold
/// generates `inputs.json` up front), the run still reports the same per-record
/// metrics and the same `inputs.json` the raw run produced with the gate open.
#[tokio::test]
async fn test_http_default_run_closes_the_gate_without_changing_artifacts() {
    let (_h_default, default) = run_http_at(None).await;
    let (_h_raw, raw) = run_http_at(Some("raw")).await;

    let marker = retention_marker(&default);
    assert!(
        marker.contains("exact_fold=true"),
        "the default run must select exact-fold, which is what closes the payload \
         gate (inputs.json generated up front, no raw artifact); marker was: {marker}"
    );
    assert!(
        default.artifacts.raw_records().is_empty(),
        "no raw artifact should be produced on the default (no-raw) run"
    );
    assert_eq!(
        default.artifacts.request_count() as u32,
        REQUEST_COUNT,
        "skipping the canonical payload must not perturb the request metrics"
    );
    let json = default.artifacts.json();
    assert!(
        json.get("time_to_first_token")
            .map(|value| !value.is_null())
            .unwrap_or(false),
        "streaming run should still report time_to_first_token with the gate closed"
    );
    assert_eq!(
        inputs_payloads(&default),
        inputs_payloads(&raw),
        "closing the payload gate changed inputs.json"
    );
}
