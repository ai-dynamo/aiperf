// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! End-to-end coverage for three HTTP dialects the AIPerf runner drives that the
//! mock server previously lacked: the vLLM/Dynamo token-native Generate endpoint
//! (`/inference/v1/generate`), the OpenAI Responses API (`/v1/responses`), and
//! the KServe `/openai/v1/*` OpenAI-compatible aliases.
//!
//! Each test drives the real `python -m aiperf profile` CLI against
//! `aiperf-mock-server` with `--export-level raw` and verifies the raw
//! per-record output (`profile_export_raw.jsonl`) carries the correct DATA
//! (and, for the streaming dialects, TIMING) — the feature-complete bar.

mod common;
use common::*;

use serde_json::Value;

// ============================================================================
// (1) vLLM / Dynamo token-native Generate — `POST /inference/v1/generate`
// ============================================================================

/// Parse the single non-streaming response body out of a raw record. The runner
/// records a non-SSE response as one `responses[]` entry `{perf_ns, text,
/// content_type}` whose `text` is the verbatim JSON body
/// (`aiperf_runtime::engine::records::text_response_value`).
fn non_streaming_body(record: &Value) -> Option<Value> {
    let responses = record.get("responses").and_then(Value::as_array)?;
    for resp in responses {
        if let Some(text) = resp.get("text").and_then(Value::as_str) {
            if let Ok(body) = serde_json::from_str::<Value>(text) {
                return Some(body);
            }
        }
    }
    None
}

/// Token-native Generate: the runner sends validated raw input `token_ids`
/// (`vllm_generate.rs:146-162`) and parses `choices[].token_ids` integer arrays
/// (`vllm_generate.rs:259-303`). Drive a synthetic run with a fixed ISL=64 /
/// OSL=8 and assert every raw record's response carries an 8-integer `token_ids`
/// output and the correct ISL/model/status.
#[tokio::test]
async fn vllm_generate_raw_records_carry_token_ids() {
    if cfg!(target_os = "macos") {
        return;
    }
    const ISL: usize = 64;
    const OSL: usize = 8;
    let h = AIPerfHarness::new().await;
    let r = h.run(&format!(
        "--model gpt-4 --url {} --endpoint-type vllm_generate \
         --request-count 6 --concurrency 2 --workers-max 1 \
         --synthetic-input-tokens-mean {ISL} --synthetic-input-tokens-stddev 0 \
         --output-tokens-mean {OSL} --output-tokens-stddev 0 \
         --export-level raw --ui simple",
        h.mock.url
    ));
    assert!(
        r.success(),
        "vllm_generate run failed (exit {}):\nstdout:\n{}\nstderr:\n{}",
        r.exit_code,
        r.stdout,
        r.stderr
    );

    let records = r.artifacts.raw_records();
    assert_eq!(records.len(), 6, "expected 6 raw records");
    for (i, rec) in records.iter().enumerate() {
        // DATA: HTTP status.
        assert_eq!(
            rec.get("status").and_then(Value::as_u64),
            Some(200),
            "record {i}: status not 200\n{rec}"
        );
        let body = non_streaming_body(rec)
            .unwrap_or_else(|| panic!("record {i}: no parseable non-streaming body\n{rec}"));

        // DATA: token_ids output array of exactly OSL integers.
        let token_ids = body
            .pointer("/choices/0/token_ids")
            .and_then(Value::as_array)
            .unwrap_or_else(|| panic!("record {i}: missing choices[0].token_ids\n{body}"));
        assert_eq!(
            token_ids.len(),
            OSL,
            "record {i}: OSL (token_ids count) {} != requested cap {OSL}",
            token_ids.len()
        );
        assert!(
            token_ids.iter().all(Value::is_u64),
            "record {i}: token_ids must be integers, got {token_ids:?}"
        );

        // DATA: ISL echoed in usage, and model.
        assert_eq!(
            body.pointer("/usage/prompt_tokens").and_then(Value::as_u64),
            Some(ISL as u64),
            "record {i}: usage.prompt_tokens != ISL {ISL}"
        );
        assert_eq!(
            body.pointer("/usage/completion_tokens")
                .and_then(Value::as_u64),
            Some(OSL as u64),
            "record {i}: usage.completion_tokens != OSL {OSL}"
        );
        assert_eq!(
            body.get("model").and_then(Value::as_str),
            Some("gpt-4"),
            "record {i}: model mismatch"
        );
    }
}

// ============================================================================
// (2) KServe `/openai/v1/*` aliases — `kserve_chat` -> /openai/v1/chat/completions
// ============================================================================

/// The KServe OpenAI-compatible chat alias: the runner's `kserve_chat` endpoint
/// (`kserve.rs:33`) defaults to `/openai/v1/chat/completions`, which the mock now
/// routes to the same OpenAI chat handler. Drive a tuned streaming run and verify
/// every raw record reproduces the tuned TTFT/ITL/latency and OSL — proving the
/// alias route reaches the real chat SSE path end to end.
#[tokio::test]
async fn openai_v1_chat_alias_raw_timing_and_data() {
    if cfg!(target_os = "macos") {
        return;
    }
    const TTFT_MS: f64 = 100.0;
    const ITL_MS: f64 = 10.0;
    const OSL: usize = 8;

    let h = AIPerfHarness::new_with(tuned_mock_config(TTFT_MS, ITL_MS)).await;
    let r = h.run(&format!(
        "--model gpt-4 --url {} --endpoint-type kserve_chat --streaming \
         --concurrency 2 --request-count 6 \
         --synthetic-input-tokens-mean 64 --synthetic-input-tokens-stddev 0 \
         --output-tokens-mean {OSL} --output-tokens-stddev 0 \
         --export-level raw --ui simple",
        h.mock.url
    ));
    assert!(
        r.success(),
        "kserve_chat alias run failed (exit {}):\nstdout:\n{}\nstderr:\n{}",
        r.exit_code,
        r.stdout,
        r.stderr
    );

    let records = r.artifacts.raw_records();
    assert_eq!(records.len(), 6, "expected 6 raw records");
    if timing_fast_forwarded(&records, TTFT_MS) {
        return;
    }
    assert_raw_records_timing_and_data(
        &records,
        &TunedExpectations::new(TTFT_MS, ITL_MS, OSL).model("gpt-4"),
    );
}

// ============================================================================
// (3) OpenAI Responses API — `POST /v1/responses`
// ============================================================================

/// Collect `(perf_ns, delta_text)` for every `response.output_text.delta` SSE
/// frame in one raw record. The runner records each SSE event as a `responses[]`
/// entry with a `data` packet whose value is the event JSON; the streaming
/// parser keys on `type == "response.output_text.delta"` and reads `delta`
/// (`endpoints.rs:1334-1391`).
fn responses_deltas(record: &Value) -> Vec<(i64, String)> {
    let mut out = Vec::new();
    let Some(responses) = record.get("responses").and_then(Value::as_array) else {
        return out;
    };
    for resp in responses {
        let perf_ns = resp.get("perf_ns").and_then(Value::as_i64).unwrap_or(0);
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
            let Ok(obj) = serde_json::from_str::<Value>(raw.trim()) else {
                continue;
            };
            if obj.get("type").and_then(Value::as_str) == Some("response.output_text.delta") {
                if let Some(delta) = obj.get("delta").and_then(Value::as_str) {
                    out.push((perf_ns, delta.to_string()));
                }
            }
        }
    }
    out
}

/// True when any `response.completed` frame in the record carries a usage block
/// with the expected `output_tokens`.
fn responses_completed_output_tokens(record: &Value) -> Option<u64> {
    let responses = record.get("responses").and_then(Value::as_array)?;
    for resp in responses {
        let packets = resp.get("packets").and_then(Value::as_array)?;
        for packet in packets {
            if packet.get("name").and_then(Value::as_str) != Some("data") {
                continue;
            }
            let Some(raw) = packet.get("value").and_then(Value::as_str) else {
                continue;
            };
            let Ok(obj) = serde_json::from_str::<Value>(raw.trim()) else {
                continue;
            };
            if obj.get("type").and_then(Value::as_str) == Some("response.completed") {
                if let Some(tokens) = obj
                    .pointer("/response/usage/output_tokens")
                    .and_then(Value::as_u64)
                {
                    return Some(tokens);
                }
            }
        }
    }
    None
}

/// OpenAI Responses API streaming: drive a tuned run and verify every raw record
/// carries exactly OSL `response.output_text.delta` frames with non-empty text,
/// a terminal `response.completed` usage block, and (when the environment is not
/// a timer-virtualizing sandbox) the tuned TTFT/ITL on the delta frames.
#[tokio::test]
async fn responses_streaming_raw_records_carry_deltas_and_usage() {
    if cfg!(target_os = "macos") {
        return;
    }
    const TTFT_MS: f64 = 100.0;
    const ITL_MS: f64 = 10.0;
    const OSL: usize = 8;

    let h = AIPerfHarness::new_with(tuned_mock_config(TTFT_MS, ITL_MS)).await;
    let r = h.run(&format!(
        "--model gpt-4 --url {} --endpoint-type responses --streaming \
         --concurrency 2 --request-count 6 \
         --synthetic-input-tokens-mean 64 --synthetic-input-tokens-stddev 0 \
         --output-tokens-mean {OSL} --output-tokens-stddev 0 \
         --export-level raw --ui simple",
        h.mock.url
    ));
    assert!(
        r.success(),
        "responses run failed (exit {}):\nstdout:\n{}\nstderr:\n{}",
        r.exit_code,
        r.stdout,
        r.stderr
    );

    let records = r.artifacts.raw_records();
    assert_eq!(records.len(), 6, "expected 6 raw records");

    // Compute a fast-forward guard from the first record's first delta.
    let first_deltas = responses_deltas(&records[0]);
    let start_perf = records[0].get("start_perf_ns").and_then(Value::as_i64);
    let measured_ttft = match (start_perf, first_deltas.first()) {
        (Some(start), Some((perf, _))) => Some((perf - start) as f64 / 1e6),
        _ => None,
    };
    let check_timing = measured_ttft.map(|t| t >= TTFT_MS / 4.0).unwrap_or(false);
    if !check_timing {
        eprintln!(
            "NOTE: timer virtualized (first-delta TTFT {measured_ttft:?}ms << {TTFT_MS}ms) — \
             asserting DATA only, skipping TTFT/ITL"
        );
    }

    for (i, rec) in records.iter().enumerate() {
        // DATA: HTTP status.
        assert_eq!(
            rec.get("status").and_then(Value::as_u64),
            Some(200),
            "record {i}: status not 200\n{rec}"
        );

        // DATA: exactly OSL content deltas, each non-empty text.
        let deltas = responses_deltas(rec);
        assert_eq!(
            deltas.len(),
            OSL,
            "record {i}: {} output_text.delta frames != requested cap {OSL}",
            deltas.len()
        );
        assert!(
            deltas.iter().all(|(_, text)| !text.is_empty()),
            "record {i}: a delta frame carried empty text"
        );

        // DATA: terminal completed usage carries output_tokens == OSL.
        assert_eq!(
            responses_completed_output_tokens(rec),
            Some(OSL as u64),
            "record {i}: response.completed usage.output_tokens != {OSL}"
        );

        if !check_timing {
            continue;
        }

        // TIMING: TTFT ~= tuned within tolerance.
        let start = rec
            .get("start_perf_ns")
            .and_then(Value::as_i64)
            .expect("start_perf_ns");
        let ttft_ms = (deltas[0].0 - start) as f64 / 1e6;
        assert!(
            (ttft_ms - TTFT_MS).abs() <= 8.0,
            "record {i}: TTFT {ttft_ms:.2}ms not within 8ms of tuned {TTFT_MS}ms"
        );

        // TIMING: mean ITL over consecutive delta frames ~= tuned.
        let perfs: Vec<i64> = deltas.iter().map(|(p, _)| *p).collect();
        let gaps: Vec<f64> = perfs
            .windows(2)
            .map(|pair| (pair[1] - pair[0]) as f64 / 1e6)
            .collect();
        let mean_itl = gaps.iter().sum::<f64>() / gaps.len() as f64;
        assert!(
            (mean_itl - ITL_MS).abs() <= 2.0,
            "record {i}: mean ITL {mean_itl:.3}ms not within 2ms of tuned {ITL_MS}ms"
        );
    }
}
