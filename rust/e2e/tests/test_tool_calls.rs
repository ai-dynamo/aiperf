// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! End-to-end tool-call / function-call tests: drive `aiperf profile` against a
//! mock server configured with `--tool-call-rate 1.0` and verify the raw
//! per-record export carries the function tool call the runner parses.
//!
//! Both wire shapes are covered at the raw-record level:
//!   - streaming: the argument string is split across two `delta.tool_calls`
//!     frames; merging them by `index` reconstructs the full function name and
//!     arguments (this is exactly the runner's streamed tool-call merge in
//!     `aiperf::endpoints::endpoints::merge_tool_call_delta`). A frame carries
//!     `finish_reason: "tool_calls"`.
//!   - non-streaming: the single response body carries `message.tool_calls` with
//!     the same function name/arguments and `finish_reason: "tool_calls"`.
//!
//! The tool-definition prompt tokens are reported as `toolUsePromptTokenCount`
//! (the exact key `aiperf::endpoints::usage::UsageView` reads into
//! `usage_tool_use_prompt_tokens`), which the streaming path exposes in the
//! terminal usage frame and the summary export surfaces as a derived metric.

mod common;
use common::*;

use aiperf_mock_server::config::MockServerConfig;
use serde_json::{Value, json};

const REQUESTS: usize = 6;
const TOOL_NAME: &str = "get_weather";
const TOOL_ARGS: &str = r#"{"location":"NYC"}"#;

/// Write a tiny `single_turn` input file (the `text` field becomes the user
/// prompt). Nothing about the prompt drives tool emission — that is the seeded
/// `--tool-call-rate` draw — so any prompt works.
fn write_prompts(dir: &std::path::Path) -> std::path::PathBuf {
    let records: Vec<Value> = (0..REQUESTS)
        .map(|i| json!({ "text": format!("Tool probe {i}: what is the weather?") }))
        .collect();
    write_jsonl(dir, "prompts.jsonl", &records)
}

/// A `--fast` mock config that answers every chat request with a function tool
/// call (`--tool-call-rate 1.0`).
fn tool_call_cfg() -> MockServerConfig {
    MockServerConfig {
        fast: true,
        no_tokenizer: true,
        tool_call_rate: 1.0,
        tool_call_name: TOOL_NAME.to_string(),
        tool_call_arguments: TOOL_ARGS.to_string(),
        random_seed: Some(7),
        ..MockServerConfig::default()
    }
}

/// Every `data:` SSE packet in a raw record, parsed as JSON (skipping `[DONE]`).
fn record_data_frames(record: &Value) -> Vec<Value> {
    let mut out = Vec::new();
    let Some(responses) = record.get("responses").and_then(Value::as_array) else {
        return out;
    };
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
                out.push(obj);
            }
        }
    }
    out
}

/// Reconstruct the streamed tool call (index 0) from one raw record by merging
/// `choices[0].delta.tool_calls` across every frame: `function.name` is set on
/// the opening frame, and `function.arguments` fragments concatenate — exactly
/// what the runner does. Also reports whether any frame carried
/// `finish_reason: "tool_calls"`.
fn reconstruct_streamed_tool_call(record: &Value) -> (String, String, bool) {
    let mut name = String::new();
    let mut arguments = String::new();
    let mut saw_finish = false;
    for obj in record_data_frames(record) {
        let choice = &obj["choices"][0];
        if choice["finish_reason"] == "tool_calls" {
            saw_finish = true;
        }
        if let Some(tcs) = choice["delta"]["tool_calls"].as_array() {
            for tc in tcs {
                if tc["index"].as_u64() != Some(0) {
                    continue;
                }
                if let Some(n) = tc["function"]["name"].as_str() {
                    name.push_str(n);
                }
                if let Some(a) = tc["function"]["arguments"].as_str() {
                    arguments.push_str(a);
                }
            }
        }
    }
    (name, arguments, saw_finish)
}

/// Reconstruct the streamed assistant `content` (generated tokens) from one raw
/// record — proves the tool-call stream still carries observable output tokens.
fn reconstruct_streamed_content(record: &Value) -> String {
    let mut out = String::new();
    for obj in record_data_frames(record) {
        if let Some(c) = obj
            .pointer("/choices/0/delta/content")
            .and_then(Value::as_str)
        {
            out.push_str(c);
        }
    }
    out
}

/// Pull the non-streaming response body (the frame that carries
/// `choices[0].message`) from a raw record. A non-streaming response is captured
/// as `responses[].text` (the whole JSON body), not as SSE `data:` packets, so
/// parse that; fall back to any `data:` frame that carries a message.
fn record_message_body(record: &Value) -> Value {
    if let Some(responses) = record.get("responses").and_then(Value::as_array) {
        for resp in responses {
            if let Some(text) = resp.get("text").and_then(Value::as_str) {
                if let Ok(obj) = serde_json::from_str::<Value>(text.trim()) {
                    if obj.pointer("/choices/0/message").is_some() {
                        return obj;
                    }
                }
            }
        }
    }
    for obj in record_data_frames(record) {
        if obj.pointer("/choices/0/message").is_some() {
            return obj;
        }
    }
    panic!("no non-streaming message body found in record: {record}");
}

/// Streaming: at rate 1.0 every raw record's SSE frames carry `delta.tool_calls`
/// deltas that reconstruct the configured function name + arguments, a frame
/// carries `finish_reason: "tool_calls"`, generated content tokens are still
/// present (observable output), and the terminal usage frame reports
/// `toolUsePromptTokenCount`.
#[tokio::test]
async fn tool_calls_streamed_in_raw_records() {
    let dir = tempfile::TempDir::new().unwrap();
    let prompts = write_prompts(dir.path());

    let h = AIPerfHarness::new_with(tool_call_cfg()).await;
    let r = h.run(&format!(
        "--model gpt-4 --url {} --endpoint-type chat --streaming \
         --use-server-token-count \
         --input-file {} --custom-dataset-type single_turn \
         --request-count {REQUESTS} --concurrency 2 --workers-max 1 \
         --random-seed 7 --export-level raw --ui simple",
        h.mock.url,
        prompts.display(),
    ));
    assert!(r.success(), "stderr: {}", r.stderr);

    let records = r.artifacts.raw_records();
    assert_eq!(records.len(), REQUESTS, "expected one record per request");

    for rec in &records {
        let (name, arguments, saw_finish) = reconstruct_streamed_tool_call(rec);
        assert_eq!(
            name, TOOL_NAME,
            "reconstructed function name from streamed tool_calls deltas"
        );
        assert_eq!(
            arguments, TOOL_ARGS,
            "reconstructed function arguments from streamed tool_calls deltas"
        );
        assert!(
            saw_finish,
            "a streamed frame must carry finish_reason=tool_calls"
        );

        // The generated content tokens still stream (the tool call is emitted in
        // addition to output content, so the token/latency model is intact).
        let content = reconstruct_streamed_content(rec);
        assert!(
            !content.is_empty(),
            "tool-call stream should still carry generated output tokens"
        );

        // toolUsePromptTokenCount is observable in the terminal usage frame.
        let usage_frame = record_data_frames(rec)
            .into_iter()
            .find(|o| o.get("usage").map(|u| !u.is_null()).unwrap_or(false))
            .expect("a terminal usage frame is present with --use-server-token-count");
        assert!(
            usage_frame["usage"]["toolUsePromptTokenCount"]
                .as_u64()
                .unwrap_or(0)
                > 0,
            "usage frame should report toolUsePromptTokenCount > 0"
        );
    }
}

/// Non-streaming: the single response body carries `message.tool_calls` with the
/// configured function name/arguments, `finish_reason: "tool_calls"`, and the
/// usage object reports `toolUsePromptTokenCount`.
#[tokio::test]
async fn tool_calls_non_streaming_in_raw_records() {
    let dir = tempfile::TempDir::new().unwrap();
    let prompts = write_prompts(dir.path());

    let h = AIPerfHarness::new_with(tool_call_cfg()).await;
    let r = h.run(&format!(
        "--model gpt-4 --url {} --endpoint-type chat \
         --input-file {} --custom-dataset-type single_turn \
         --request-count {REQUESTS} --concurrency 2 --workers-max 1 \
         --random-seed 7 --export-level raw --ui simple",
        h.mock.url,
        prompts.display(),
    ));
    assert!(r.success(), "stderr: {}", r.stderr);

    let records = r.artifacts.raw_records();
    assert_eq!(records.len(), REQUESTS, "expected one record per request");

    for rec in &records {
        let body = record_message_body(rec);
        let choice = &body["choices"][0];
        assert_eq!(
            choice["finish_reason"], "tool_calls",
            "non-streaming finish reason should be tool_calls"
        );
        let tc = &choice["message"]["tool_calls"][0];
        assert_eq!(tc["type"], "function");
        assert!(
            tc["id"].as_str().unwrap_or("").starts_with("call_"),
            "tool call id should be present"
        );
        assert_eq!(tc["function"]["name"], TOOL_NAME);
        assert_eq!(tc["function"]["arguments"], TOOL_ARGS);
        assert!(
            body["usage"]["toolUsePromptTokenCount"]
                .as_u64()
                .unwrap_or(0)
                > 0,
            "non-streaming usage should report toolUsePromptTokenCount > 0"
        );
    }
}

/// The derived `usage_tool_use_prompt_tokens` metric lands in the summary export
/// when tool calls are emitted (proves the count flows all the way through the
/// runner's usage accounting, not just the raw frame).
#[tokio::test]
async fn tool_use_prompt_tokens_metric_in_summary() {
    let dir = tempfile::TempDir::new().unwrap();
    let prompts = write_prompts(dir.path());

    let h = AIPerfHarness::new_with(tool_call_cfg()).await;
    let r = h.run(&format!(
        "--model gpt-4 --url {} --endpoint-type chat --streaming \
         --use-server-token-count \
         --input-file {} --custom-dataset-type single_turn \
         --request-count {REQUESTS} --concurrency 2 --workers-max 1 \
         --random-seed 7 --export-level raw --ui simple",
        h.mock.url,
        prompts.display(),
    ));
    assert!(r.success(), "stderr: {}", r.stderr);

    let summary = r.artifacts.json();
    assert!(!summary.is_null(), "summary export should exist");
    let avg = summary
        .get("usage_tool_use_prompt_tokens")
        .and_then(|m| m.get("avg"))
        .and_then(Value::as_f64)
        .unwrap_or_else(|| panic!("summary missing usage_tool_use_prompt_tokens: {summary}"));
    assert!(
        avg > 0.0,
        "usage_tool_use_prompt_tokens avg should be > 0, got {avg}"
    );
}
