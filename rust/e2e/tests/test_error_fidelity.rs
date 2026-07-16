// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! End-to-end error-injection fidelity tests: drive `aiperf profile` against a
//! mock server configured to inject specific HTTP status codes and mid-stream
//! SSE errors, then verify the raw per-record output carries the injected error
//! exactly as the runner classifies it.
//!
//! These exercise the runner's error-absorption path
//! (`aiperf_runtime::transport::http::absorb_transport_error`, `rust/aiperf/src/http.rs:1062`),
//! which keys off the transport's status code + `ErrorKind`. The raw-record
//! `status` / `error` fields come from `aiperf_runtime::engine::records`
//! (`raw_record_row`, `rust/aiperf/src/engine/records.rs:928`):
//!   * a non-2xx HTTP response -> `status: <code>`, `error: {code, type:
//!     "HttpError", message}` (`ErrorDetails::http`),
//!   * a mid-stream `event: error` SSE frame -> `error: {code: 502, type:
//!     "SSEResponseError", message}` (`ErrorDetails::sse`,
//!     `rust/aiperf/src/transport::http/models/error.rs:44`).
//!
//! The mock's configurable status-code menu, Retry-After header, and mid-stream
//! SSE error live in `rust/mock-server/src/{config,state,handlers}.rs`.

mod common;
use common::*;

use aiperf_mock_server::config::MockServerConfig;
use serde_json::Value;

/// The `error` object on a raw record, if the record terminated in error.
fn record_error(record: &Value) -> Option<&Value> {
    record.get("error").filter(|e| !e.is_null())
}

/// The classified error code on a raw record (e.g. 429 for HTTP, 502 for SSE).
fn error_code(record: &Value) -> Option<u64> {
    record.pointer("/error/code").and_then(Value::as_u64)
}

/// The stable classified error type on a raw record (`HttpError`,
/// `SSEResponseError`).
fn error_type(record: &Value) -> Option<&str> {
    record.pointer("/error/type").and_then(Value::as_str)
}

/// The HTTP status captured on a raw record, if any.
fn record_status(record: &Value) -> Option<u64> {
    record.get("status").and_then(Value::as_u64)
}

/// Count of generated-content SSE chunks (`choices[0].delta.content` present)
/// in a raw record's `responses[]`, excluding `[DONE]`, usage, and reasoning.
fn content_chunk_count(record: &Value) -> usize {
    let Some(responses) = record.get("responses").and_then(Value::as_array) else {
        return 0;
    };
    let mut n = 0;
    for resp in responses {
        let Some(packets) = resp.get("packets").and_then(Value::as_array) else {
            continue;
        };
        for packet in packets {
            if packet.get("name").and_then(Value::as_str) != Some("data") {
                continue;
            }
            let Some(raw) = packet.get("value").and_then(Value::as_str) else {
                continue;
            };
            let trimmed = raw.trim();
            if trimmed == "[DONE]" {
                continue;
            }
            if let Ok(obj) = serde_json::from_str::<Value>(trimmed) {
                let is_content = obj
                    .pointer("/choices/0/delta/content")
                    .map(|c| !c.is_null())
                    .unwrap_or(false);
                if is_content {
                    n += 1;
                }
            }
        }
    }
    n
}

/// A fast, tokenizer-free mock with a seeded error stream.
fn error_cfg() -> MockServerConfig {
    MockServerConfig {
        fast: true,
        no_tokenizer: true,
        random_seed: Some(7),
        ..MockServerConfig::default()
    }
}

/// The injected status code appears verbatim on the raw records: with a moderate
/// error-rate and a single-entry `429` menu, the errored records carry HTTP 429
/// (classified `http_error` / code 429) — NOT the historical hardcoded 500 — and
/// the run still produces successful records alongside them (a real mix).
#[tokio::test]
async fn injected_429_status_code_shows_in_raw_records() {
    let mut cfg = error_cfg();
    // A partial rate yields a mix of 429s and successes in one run, so the run
    // completes (exit 0) and both classes are present in the raw records.
    cfg.error_rate = 45.0;
    cfg.error_status_codes = vec![429];
    cfg.error_retry_after = 3;

    let h = AIPerfHarness::new_with(cfg).await;
    let r = h.run(&format!(
        "--model gpt-4 --url {} --endpoint-type chat \
         --request-count 40 --concurrency 4 --workers-max 1 \
         --random-seed 7 --export-level raw --ui simple",
        h.mock.url,
    ));
    // A partial error-rate run tolerates errors and still exits successfully.
    assert!(r.success(), "stderr: {}", r.stderr);

    let records = r.artifacts.raw_records();
    assert_eq!(
        records.len(),
        40,
        "expected 40 raw records (mix of ok + error)"
    );

    let errored: Vec<&Value> = records
        .iter()
        .filter(|r| record_error(r).is_some())
        .collect();
    let ok: Vec<&Value> = records
        .iter()
        .filter(|r| record_error(r).is_none())
        .collect();

    assert!(
        !errored.is_empty(),
        "expected at least one injected error at error-rate 45%"
    );
    assert!(
        !ok.is_empty(),
        "expected at least one success at error-rate 45% (proves a real mix, not all-error)"
    );

    // Every injected error is a 429 (the menu), classified http_error — proving
    // the status is configurable and the hardcoded 500 is gone.
    for rec in &errored {
        assert_eq!(
            error_code(rec),
            Some(429),
            "errored record must carry the injected 429 code, got {:?}\n{rec}",
            error_code(rec)
        );
        assert_eq!(
            error_type(rec),
            Some("HttpError"),
            "429 must classify as an HTTP transport error\n{rec}"
        );
        assert_eq!(
            record_status(rec),
            Some(429),
            "errored record status must be 429\n{rec}"
        );
    }

    // No record anywhere shows the old hardcoded 500.
    assert!(
        records
            .iter()
            .all(|r| error_code(r) != Some(500) && record_status(r) != Some(500)),
        "no record should carry a 500 — the menu replaced the hardcoded status"
    );
}

/// A menu of several codes is exercised: with an all-error rate over a menu of
/// `{429, 503, 400}`, every errored record carries one of exactly those codes
/// (never 500), proving the seeded per-error code selection walks the menu.
#[tokio::test]
async fn injected_status_code_menu_is_walked() {
    let mut cfg = error_cfg();
    cfg.error_rate = 100.0;
    cfg.error_status_codes = vec![429, 503, 400];

    let h = AIPerfHarness::new_with(cfg).await;
    let r = h.run(&format!(
        "--model gpt-4 --url {} --endpoint-type chat \
         --request-count 30 --concurrency 4 --workers-max 1 \
         --random-seed 7 --export-level raw --ui simple",
        h.mock.url,
    ));
    // With a 100% error-rate the run may exit non-zero (all requests failed);
    // the raw records are still flushed and are what we verify.
    let _ = r.success();

    let records = r.artifacts.raw_records();
    assert!(
        !records.is_empty(),
        "raw records must be written even when every request errors; stderr: {}",
        r.stderr
    );

    let mut seen = std::collections::BTreeSet::new();
    for rec in &records {
        let code = error_code(rec)
            .unwrap_or_else(|| panic!("every record should be errored at 100% error-rate\n{rec}"));
        assert!(
            [429u64, 503, 400].contains(&code),
            "injected code {code} not in the configured menu {{429,503,400}}\n{rec}"
        );
        seen.insert(code);
    }
    // Over 30 seeded draws the selection should hit more than one menu entry —
    // proving the code is chosen per-error, not fixed.
    assert!(
        seen.len() >= 2,
        "expected the seeded selection to walk >1 menu code, saw {seen:?}"
    );
}

/// A mid-stream SSE error truncates a streaming record and the run still
/// completes: at a partial mid-stream rate, some streaming records emit a few
/// content frames then a terminal `event: error` (classified `sse_error` /
/// code 502) with fewer content chunks than the requested cap, while other
/// records complete in full — and the overall report is written.
#[tokio::test]
async fn midstream_sse_error_truncates_record_and_run_completes() {
    let mut cfg = error_cfg();
    // Partial rate -> a mix of truncated and full streams in one run.
    cfg.error_midstream_rate = 0.5;

    let h = AIPerfHarness::new_with(cfg).await;
    let r = h.run(&format!(
        "--model gpt-4 --url {} --endpoint-type chat --streaming \
         --output-tokens-mean 20 --output-tokens-stddev 0 \
         --request-count 24 --concurrency 4 --workers-max 1 \
         --random-seed 7 --export-level raw --ui simple",
        h.mock.url,
    ));
    // The runner must handle mid-stream errors gracefully: a partial-rate run
    // completes and writes its report.
    assert!(
        r.success(),
        "run with mid-stream errors should still complete; stderr: {}",
        r.stderr
    );
    assert!(
        !r.artifacts.json().is_null(),
        "a summary report (profile_export_aiperf.json) must be written"
    );

    let records = r.artifacts.raw_records();
    assert_eq!(records.len(), 24, "expected 24 raw records");

    let errored: Vec<&Value> = records
        .iter()
        .filter(|r| record_error(r).is_some())
        .collect();
    let ok: Vec<&Value> = records
        .iter()
        .filter(|r| record_error(r).is_none())
        .collect();

    assert!(
        !errored.is_empty(),
        "expected at least one mid-stream SSE error at rate 0.5"
    );
    assert!(
        !ok.is_empty(),
        "expected at least one full streaming success at rate 0.5"
    );

    // Every mid-stream failure is classified as an SSE transport error and is
    // truncated: at most the mock's pre-error token budget of content chunks,
    // strictly fewer than the 20-token successful streams.
    for rec in &errored {
        assert_eq!(
            error_type(rec),
            Some("SSEResponseError"),
            "mid-stream failure must classify as an SSE response error\n{rec}"
        );
        assert_eq!(
            error_code(rec),
            Some(502),
            "SSE error carries pseudo-status 502\n{rec}"
        );
        let chunks = content_chunk_count(rec);
        assert!(
            (1..=3).contains(&chunks),
            "truncated record should carry 1..=3 partial content chunks, got {chunks}\n{rec}"
        );
    }

    // Successful records streamed the full requested output (20 content chunks),
    // proving the errored records are genuinely truncated, not merely short.
    for rec in &ok {
        assert_eq!(
            content_chunk_count(rec),
            20,
            "successful streaming record should carry the full 20-token output\n{rec}"
        );
    }
}
