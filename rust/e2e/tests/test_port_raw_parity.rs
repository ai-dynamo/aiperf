// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Python-vs-Rust `--export-level raw` parity proofs for the origin/main ports
//! re-implemented on the Rust engine.
//!
//! Each test runs the SAME `aiperf profile --export-level raw` config through
//! the Rust engine (`h.run`) and the Python engine
//! (`h.run_env(.., AIPERF_RUNTIME_ENGINE=python)`) against the same in-process
//! mock, then asserts a deterministic, timing-free projection of
//! `profile_export_raw.jsonl` is byte-identical across the two engines. Literal
//! whole-file identity is impossible across engines (per-run UUID
//! `x_request_id`, `worker_id`, and wall-clock ns differ), so — exactly like
//! `test_main_delta_raw_parity` and `test_seeded_poisson_parity` — parity is
//! asserted on the feature-relevant deterministic fields.

mod common;
use common::*;

use serde_json::{Value, json};

fn sorted(mut v: Vec<String>) -> Vec<String> {
    v.sort();
    v
}

fn header<'a>(record: &'a Value, name: &str) -> Option<&'a str> {
    record
        .get("request_headers")
        .and_then(Value::as_object)
        .and_then(|h| {
            h.iter()
                .find(|(k, _)| k.eq_ignore_ascii_case(name))
                .and_then(|(_, v)| v.as_str())
        })
}

// ---------------------------------------------------------------------------
// #26 — opt-in session-affinity headers (X-Session-ID / X-SMG-Routing-Key)
// ---------------------------------------------------------------------------

/// Deterministic per-record projection proving the additive session-affinity
/// derivation without depending on the per-run correlation-id literal: for each
/// record, whether X-Session-ID / X-SMG-Routing-Key are present and whether each
/// equals that record's own `metadata.x_correlation_id`.
fn affinity_projection(records: &[Value]) -> Vec<String> {
    sorted(
        records
            .iter()
            .map(|r| {
                let corr = r
                    .get("metadata")
                    .and_then(|m| m.get("x_correlation_id"))
                    .and_then(Value::as_str);
                let session = header(r, "X-Session-ID");
                let smg = header(r, "X-SMG-Routing-Key");
                json!({
                    "session_num": r["metadata"]["session_num"],
                    "turn_index": r["metadata"]["turn_index"],
                    "has_session": session.is_some(),
                    "has_smg": smg.is_some(),
                    "session_eq_corr": session.is_some() && session == corr,
                    "smg_eq_corr": smg.is_some() && smg == corr,
                })
                .to_string()
            })
            .collect(),
    )
}

#[tokio::test]
async fn port26_session_affinity_headers_raw_parity() {
    let h = AIPerfHarness::new().await;
    let args = format!(
        "--model {DEFAULT_MODEL} --url {} --endpoint-type chat --streaming \
         --concurrency 1 --request-count 6 --workers-max 1 --random-seed 42 \
         --synthetic-input-tokens-mean 16 --output-tokens-mean 4 \
         --export-level raw --ui simple --tokenizer builtin",
        h.mock.url
    );
    let env = &[
        ("AIPERF_HTTP_X_SESSION_ID_FROM_CORRELATION_ID", "1"),
        ("AIPERF_HTTP_X_SMG_ROUTING_KEY_FROM_CORRELATION_ID", "1"),
    ];

    let rust = h.run_env(&args, env);
    assert!(rust.success(), "rust run failed:\n{}", rust.stderr);
    let mut rust_env = env.to_vec();
    rust_env.push(("AIPERF_RUNTIME_ENGINE", "python"));
    let py = h.run_env(&args, &rust_env);
    assert!(py.success(), "python run failed:\n{}", py.stderr);

    let rust_raw = rust.artifacts.raw_records();
    let py_raw = py.artifacts.raw_records();
    assert!(!rust_raw.is_empty(), "rust produced no raw records");
    assert!(!py_raw.is_empty(), "python produced no raw records");

    // Invariant the feature guarantees, per engine: both affinity headers are
    // present on every record and each equals that record's correlation id.
    for (label, recs) in [("rust", &rust_raw), ("python", &py_raw)] {
        for (i, r) in recs.iter().enumerate() {
            let corr = r["metadata"]["x_correlation_id"].as_str();
            assert!(
                corr.is_some(),
                "{label} record {i}: missing metadata.x_correlation_id\n{r}"
            );
            assert_eq!(
                header(r, "X-Session-ID"),
                corr,
                "{label} record {i}: X-Session-ID != correlation id"
            );
            assert_eq!(
                header(r, "X-SMG-Routing-Key"),
                corr,
                "{label} record {i}: X-SMG-Routing-Key != correlation id"
            );
        }
    }

    // Cross-engine byte-identical deterministic projection.
    assert_eq!(
        affinity_projection(&rust_raw),
        affinity_projection(&py_raw),
        "session-affinity raw projection diverged between rust and python engines"
    );
}

/// Control: without the opt-in env flags, neither engine emits the headers.
#[tokio::test]
async fn port26_session_affinity_headers_absent_by_default_raw_parity() {
    let h = AIPerfHarness::new().await;
    let args = format!(
        "--model {DEFAULT_MODEL} --url {} --endpoint-type chat --streaming \
         --concurrency 1 --request-count 4 --workers-max 1 --random-seed 42 \
         --synthetic-input-tokens-mean 16 --output-tokens-mean 4 \
         --export-level raw --ui simple --tokenizer builtin",
        h.mock.url
    );
    let rust = h.run(&args);
    assert!(rust.success(), "rust run failed:\n{}", rust.stderr);
    let py = h.run_env(&args, &[("AIPERF_RUNTIME_ENGINE", "python")]);
    assert!(py.success(), "python run failed:\n{}", py.stderr);

    for (label, recs) in [("rust", rust.artifacts.raw_records()), ("python", py.artifacts.raw_records())] {
        assert!(!recs.is_empty(), "{label} produced no raw records");
        for (i, r) in recs.iter().enumerate() {
            assert!(header(r, "X-Session-ID").is_none(), "{label} record {i}: unexpected X-Session-ID");
            assert!(header(r, "X-SMG-Routing-Key").is_none(), "{label} record {i}: unexpected X-SMG-Routing-Key");
        }
    }
}
