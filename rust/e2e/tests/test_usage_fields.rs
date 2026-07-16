// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! End-to-end extended-usage-accounting tests: drive `aiperf profile` against a
//! mock server configured with the `--usage-*` knobs and verify the streamed
//! usage object in the raw per-record export carries every field AIPerf's
//! `UsageView` parses, and that the derived `usage_*` catalog metrics land in
//! the summary export with the exact configured values.
//!
//! Each field is pinned to a distinct nonzero value so a record's usage frame is
//! fully assertable. The runner is put in `--use-server-token-count` mode so the
//! request carries `stream_options.include_usage` and the terminal usage chunk
//! is emitted. The specific JSON keys / nesting mirror what
//! `aiperf::endpoints::usage::UsageView` reads:
//!   - top-level `cache_creation_input_tokens` -> `usage_prompt_cache_write_tokens`
//!   - top-level `prompt_cache_miss_tokens`    -> `usage_prompt_cache_miss_tokens`
//!   - top-level `toolUsePromptTokenCount`     -> `usage_tool_use_prompt_tokens`
//!   - top-level `prompt_audio_seconds`        -> `usage_prompt_audio_seconds`
//!   - `prompt_tokens_details.audio_tokens`    -> `usage_prompt_audio_tokens`
//!   - `completion_tokens_details.audio_tokens`               -> `usage_completion_audio_tokens`
//!   - `completion_tokens_details.accepted_prediction_tokens` -> `usage_accepted_prediction_tokens`
//!   - `completion_tokens_details.rejected_prediction_tokens` -> `usage_rejected_prediction_tokens`

mod common;
use common::*;

use aiperf_mock_server::config::MockServerConfig;
use serde_json::{Value, json};

// Distinct nonzero values pinned via the `--usage-*` knobs, so every emitted
// usage sub-field is individually assertable at the raw-record level.
const CACHE_WRITE: u64 = 11;
const CACHE_MISS: u64 = 22;
const CACHE_READ: u64 = 33; // Anthropic-only; not asserted on the OpenAI path.
const PROMPT_AUDIO_TOKENS: u64 = 44;
const COMPLETION_AUDIO_TOKENS: u64 = 55;
const PROMPT_AUDIO_SECONDS: f64 = 6.5;
const ACCEPTED_PREDICTION: u64 = 77;
const REJECTED_PREDICTION: u64 = 88;
const TOOL_USE_PROMPT: u64 = 99;

const REQUESTS: usize = 4;

/// Write a tiny `single_turn` input file (the `text` field becomes the user
/// prompt). No ground truth is involved; only usage accounting is under test.
fn write_prompts(dir: &std::path::Path) -> std::path::PathBuf {
    let records: Vec<Value> = (0..REQUESTS)
        .map(|i| json!({ "text": format!("Usage probe {i}: reply briefly.") }))
        .collect();
    write_jsonl(dir, "prompts.jsonl", &records)
}

/// A `--fast` mock config with every extended usage knob pinned nonzero.
fn usage_cfg() -> MockServerConfig {
    MockServerConfig {
        fast: true,
        no_tokenizer: true,
        usage_cache_write_tokens: CACHE_WRITE as usize,
        usage_cache_miss_tokens: CACHE_MISS as usize,
        usage_cache_read_tokens: CACHE_READ as usize,
        usage_prompt_audio_tokens: PROMPT_AUDIO_TOKENS as usize,
        usage_completion_audio_tokens: COMPLETION_AUDIO_TOKENS as usize,
        usage_prompt_audio_seconds: PROMPT_AUDIO_SECONDS,
        usage_accepted_prediction_tokens: ACCEPTED_PREDICTION as usize,
        usage_rejected_prediction_tokens: REJECTED_PREDICTION as usize,
        usage_tool_use_prompt_tokens: TOOL_USE_PROMPT as usize,
        ..MockServerConfig::default()
    }
}

/// Extract the streamed terminal `usage` object from one raw record by scanning
/// its SSE `data:` packets for the frame that carries a non-null `usage` key.
fn record_usage(record: &Value) -> Value {
    let responses = record
        .get("responses")
        .and_then(Value::as_array)
        .expect("record has responses");
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
                if obj.get("usage").map(|u| !u.is_null()).unwrap_or(false) {
                    return obj["usage"].clone();
                }
            }
        }
    }
    panic!("no streamed usage frame found in record: {record}");
}

/// The raw per-record usage object carries every extended field with its exact
/// configured value, at the exact key/nesting AIPerf's `UsageView` reads.
#[tokio::test]
async fn extended_usage_fields_present_in_raw_records() {
    let dir = tempfile::TempDir::new().unwrap();
    let prompts = write_prompts(dir.path());

    let h = AIPerfHarness::new_with(usage_cfg()).await;
    let r = h.run(&format!(
        "--model gpt-4 --url {} --endpoint-type chat --streaming \
         --use-server-token-count \
         --input-file {} --custom-dataset-type single_turn \
         --request-count {REQUESTS} --concurrency 2 --workers-max 1 \
         --export-level raw --ui simple",
        h.mock.url,
        prompts.display(),
    ));
    assert!(r.success(), "stderr: {}", r.stderr);

    let records = r.artifacts.raw_records();
    assert_eq!(records.len(), REQUESTS, "expected one record per request");

    for rec in &records {
        let u = record_usage(rec);
        // Top-level OpenAI-dialect keys.
        assert_eq!(
            u["cache_creation_input_tokens"].as_u64(),
            Some(CACHE_WRITE),
            "cache_creation_input_tokens (usage.rs:110 -> usage_prompt_cache_write_tokens)"
        );
        assert_eq!(
            u["prompt_cache_miss_tokens"].as_u64(),
            Some(CACHE_MISS),
            "prompt_cache_miss_tokens (usage.rs:114 -> usage_prompt_cache_miss_tokens)"
        );
        assert_eq!(
            u["toolUsePromptTokenCount"].as_u64(),
            Some(TOOL_USE_PROMPT),
            "toolUsePromptTokenCount (usage.rs:161 -> usage_tool_use_prompt_tokens)"
        );
        assert_eq!(
            u["prompt_audio_seconds"].as_f64(),
            Some(PROMPT_AUDIO_SECONDS),
            "prompt_audio_seconds (usage.rs:165 -> usage_prompt_audio_seconds)"
        );
        // Nested detail keys.
        assert_eq!(
            u["prompt_tokens_details"]["audio_tokens"].as_u64(),
            Some(PROMPT_AUDIO_TOKENS),
            "prompt_tokens_details.audio_tokens (usage.rs:128 -> usage_prompt_audio_tokens)"
        );
        assert_eq!(
            u["completion_tokens_details"]["audio_tokens"].as_u64(),
            Some(COMPLETION_AUDIO_TOKENS),
            "completion_tokens_details.audio_tokens (usage.rs:136 -> usage_completion_audio_tokens)"
        );
        assert_eq!(
            u["completion_tokens_details"]["accepted_prediction_tokens"].as_u64(),
            Some(ACCEPTED_PREDICTION),
            "accepted_prediction_tokens (usage.rs:144 -> usage_accepted_prediction_tokens)"
        );
        assert_eq!(
            u["completion_tokens_details"]["rejected_prediction_tokens"].as_u64(),
            Some(REJECTED_PREDICTION),
            "rejected_prediction_tokens (usage.rs:152 -> usage_rejected_prediction_tokens)"
        );
    }
}

/// The derived `usage_*` record metrics land in the summary export with exactly
/// the per-record configured values (avg) and their run totals (= value * N).
#[tokio::test]
async fn extended_usage_metrics_present_in_summary() {
    let dir = tempfile::TempDir::new().unwrap();
    let prompts = write_prompts(dir.path());

    let h = AIPerfHarness::new_with(usage_cfg()).await;
    let r = h.run(&format!(
        "--model gpt-4 --url {} --endpoint-type chat --streaming \
         --use-server-token-count \
         --input-file {} --custom-dataset-type single_turn \
         --request-count {REQUESTS} --concurrency 2 --workers-max 1 \
         --export-level raw --ui simple",
        h.mock.url,
        prompts.display(),
    ));
    assert!(r.success(), "stderr: {}", r.stderr);

    let summary = r.artifacts.json();
    assert!(!summary.is_null(), "summary export should exist");

    // (metric tag, per-record value) — each is a Record metric, so `avg` is the
    // fixed per-request value and `total_usage_*` is that times the request count.
    let expected: &[(&str, f64)] = &[
        ("usage_prompt_cache_write_tokens", CACHE_WRITE as f64),
        ("usage_prompt_cache_miss_tokens", CACHE_MISS as f64),
        ("usage_prompt_audio_tokens", PROMPT_AUDIO_TOKENS as f64),
        (
            "usage_completion_audio_tokens",
            COMPLETION_AUDIO_TOKENS as f64,
        ),
        ("usage_prompt_audio_seconds", PROMPT_AUDIO_SECONDS),
        (
            "usage_accepted_prediction_tokens",
            ACCEPTED_PREDICTION as f64,
        ),
        (
            "usage_rejected_prediction_tokens",
            REJECTED_PREDICTION as f64,
        ),
        ("usage_tool_use_prompt_tokens", TOOL_USE_PROMPT as f64),
    ];

    for (tag, value) in expected {
        let avg = summary
            .get(tag)
            .and_then(|m| m.get("avg"))
            .and_then(Value::as_f64)
            .unwrap_or_else(|| panic!("summary missing metric `{tag}`: {summary}"));
        assert!(
            (avg - value).abs() < 1e-9,
            "metric `{tag}` avg={avg}, expected {value}"
        );

        let total_tag = format!("total_{tag}");
        let total = summary
            .get(&total_tag)
            .and_then(Value::as_f64)
            .or_else(|| {
                summary
                    .get(&total_tag)
                    .and_then(|m| m.get("avg"))
                    .and_then(Value::as_f64)
            })
            .unwrap_or_else(|| panic!("summary missing metric `{total_tag}`: {summary}"));
        assert!(
            (total - value * REQUESTS as f64).abs() < 1e-6,
            "metric `{total_tag}`={total}, expected {}",
            value * REQUESTS as f64
        );
    }
}
