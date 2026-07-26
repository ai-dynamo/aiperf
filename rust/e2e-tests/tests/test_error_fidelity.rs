// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

mod common;
use common::*;

use aiperf_mock_server::config::MockServerConfig;
use serde_json::Value;

fn record_error(record: &Value) -> Option<&Value> {
    record.get("error").filter(|e| !e.is_null())
}

fn error_code(record: &Value) -> Option<u64> {
    record.pointer("/error/code").and_then(Value::as_u64)
}

fn error_type(record: &Value) -> Option<&str> {
    record.pointer("/error/type").and_then(Value::as_str)
}

fn record_status(record: &Value) -> Option<u64> {
    record.get("status").and_then(Value::as_u64)
}

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

fn error_cfg() -> MockServerConfig {
    MockServerConfig {
        fast: true,
        no_tokenizer: true,
        random_seed: Some(7),
        ..MockServerConfig::default()
    }
}

#[tokio::test]
async fn injected_429_status_code_shows_in_raw_records() {
    let mut cfg = error_cfg();
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

    assert!(
        records
            .iter()
            .all(|r| error_code(r) != Some(500) && record_status(r) != Some(500)),
        "configured status selection must not emit an unconfigured 500"
    );
}

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
    assert!(
        seen.len() >= 2,
        "expected the seeded selection to walk >1 menu code, saw {seen:?}"
    );
}

#[tokio::test]
async fn midstream_sse_error_truncates_record_and_run_completes() {
    let mut cfg = error_cfg();
    cfg.error_midstream_rate = 0.5;

    let h = AIPerfHarness::new_with(cfg).await;
    let r = h.run(&format!(
        "--model gpt-4 --url {} --endpoint-type chat --streaming \
         --output-tokens-mean 20 --output-tokens-stddev 0 \
         --request-count 24 --concurrency 4 --workers-max 1 \
         --random-seed 7 --export-level raw --ui simple",
        h.mock.url,
    ));
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

    for rec in &ok {
        assert_eq!(
            content_chunk_count(rec),
            20,
            "successful streaming record should carry the full 20-token output\n{rec}"
        );
    }
}
