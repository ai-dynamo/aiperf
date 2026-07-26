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

    for (label, recs) in [
        ("rust", rust.artifacts.raw_records()),
        ("python", py.artifacts.raw_records()),
    ] {
        assert!(!recs.is_empty(), "{label} produced no raw records");
        for (i, r) in recs.iter().enumerate() {
            assert!(
                header(r, "X-Session-ID").is_none(),
                "{label} record {i}: unexpected X-Session-ID"
            );
            assert!(
                header(r, "X-SMG-Routing-Key").is_none(),
                "{label} record {i}: unexpected X-SMG-Routing-Key"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// #18 — hoist leading system turn into conversation system prompt
// ---------------------------------------------------------------------------

/// The `messages` array of a record's request payload, as `role: text` strings.
fn payload_messages(record: &Value) -> Vec<String> {
    record
        .get("payload")
        .and_then(|p| p.get("messages"))
        .and_then(Value::as_array)
        .map(|msgs| {
            msgs.iter()
                .map(|m| {
                    let role = m.get("role").and_then(Value::as_str).unwrap_or("");
                    let content = match m.get("content") {
                        Some(Value::String(s)) => s.clone(),
                        // chat content can be an array of parts; join text parts.
                        Some(Value::Array(parts)) => parts
                            .iter()
                            .filter_map(|p| p.get("text").and_then(Value::as_str))
                            .collect::<Vec<_>>()
                            .join(""),
                        _ => String::new(),
                    };
                    format!("{role}: {content}")
                })
                .collect()
        })
        .unwrap_or_default()
}

/// Deterministic per-record projection: session/turn identity + the full
/// rendered `messages` role/text sequence. The hoist's observable effect is
/// entirely here — the leading system prompt is prepended to every turn's
/// messages and NO standalone system-only record exists.
fn hoist_projection(records: &[Value]) -> Vec<String> {
    sorted(
        records
            .iter()
            .map(|r| {
                json!({
                    "conversation_id": r["metadata"]["conversation_id"],
                    "turn_index": r["metadata"]["turn_index"],
                    "messages": payload_messages(r),
                })
                .to_string()
            })
            .collect(),
    )
}

#[tokio::test]
async fn port18_leading_system_turn_hoist_raw_parity() {
    let h = AIPerfHarness::new().await;
    // Two sessions, each authored with a leading text-only system turn followed
    // by two user turns. The system prompt must persist across both user turns
    // and NOT appear as its own dispatched record.
    let convo = |sid: &str| {
        json!({
            "session_id": sid,
            "turns": [
                {"role": "system", "text": "You are a helpful assistant."},
                {"text": "What is deep learning?"},
                {"text": "Explain it for a five year old."},
            ]
        })
    };
    let traces = vec![convo("s1"), convo("s2")];
    let trace_file = write_jsonl(h.artifact_dir.path(), "multiturn_system.jsonl", &traces);
    let args = format!(
        "--model {DEFAULT_MODEL} --url {} --endpoint-type chat --streaming \
         --input-file {} --custom-dataset-type multi_turn \
         --num-conversations 2 --concurrency 1 --workers-max 1 --random-seed 42 \
         --output-tokens-mean 4 --export-level raw --ui simple --tokenizer builtin",
        h.mock.url,
        trace_file.display()
    );

    let rust = h.run(&args);
    assert!(rust.success(), "rust run failed:\n{}", rust.stderr);
    let py = h.run_env(&args, &[("AIPERF_RUNTIME_ENGINE", "python")]);
    assert!(py.success(), "python run failed:\n{}", py.stderr);

    let rust_raw = rust.artifacts.raw_records();
    let py_raw = py.artifacts.raw_records();
    assert!(!rust_raw.is_empty(), "rust produced no raw records");

    // Per engine: exactly 2 user turns per session (4 records), no standalone
    // system-only record, and every record's messages LEAD with the hoisted
    // system prompt.
    for (label, recs) in [("rust", &rust_raw), ("python", &py_raw)] {
        assert_eq!(
            recs.len(),
            4,
            "{label}: expected 4 user-turn records (2 sessions x 2 turns), got {} \
             (did a standalone system turn leak a record?)",
            recs.len()
        );
        for (i, r) in recs.iter().enumerate() {
            let msgs = payload_messages(r);
            assert_eq!(
                msgs.first().map(String::as_str),
                Some("system: You are a helpful assistant."),
                "{label} record {i}: messages do not lead with the hoisted system prompt: {msgs:?}"
            );
            assert!(
                msgs.iter().filter(|m| m.starts_with("system:")).count() == 1,
                "{label} record {i}: expected exactly one system message: {msgs:?}"
            );
        }
    }

    // Cross-engine byte-identical rendered-messages projection.
    assert_eq!(
        hoist_projection(&rust_raw),
        hoist_projection(&py_raw),
        "leading-system-turn hoist raw projection diverged between rust and python engines"
    );
}

// ---------------------------------------------------------------------------
// #31 — tool_calls extraction: assistant tool_calls counted once in ISL
// ---------------------------------------------------------------------------

/// Deterministic ISL projection from the records file (`profile_export.jsonl`):
/// session/turn identity + input_sequence_length. The tool_calls double-count
/// fix is observable only in ISL — the wire payload is unchanged (tool_calls are
/// always sent), so this compares the tokenized `input_sequence_length` the
/// chat-shape extraction produces, not the raw payload.
fn isl_projection(records: &[Value]) -> Vec<String> {
    sorted(
        records
            .iter()
            .map(|r| {
                json!({
                    "conversation_id": r["metadata"]["conversation_id"],
                    "turn_index": r["metadata"]["turn_index"],
                    "input_sequence_length": r["metrics"]["input_sequence_length"],
                })
                .to_string()
            })
            .collect(),
    )
}

#[tokio::test]
async fn port31_assistant_tool_calls_isl_parity() {
    let h = AIPerfHarness::new().await;
    // A mooncake `messages`-mode row whose assistant turn replays tool_calls.
    // The chat-shape extraction must count the tool-call name/arguments toward
    // input_sequence_length exactly once (in `texts`, not double-counted via
    // `tool_texts`), identically to the Python engine.
    let traces = vec![json!({
        "timestamp": 0,
        "messages": [
            {"role": "user", "content": "What is the weather in New York City today?"},
            {"role": "assistant", "content": null, "tool_calls": [
                {"id": "call_1", "type": "function", "function": {
                    "name": "get_current_weather",
                    "arguments": "{\"location\":\"New York City\",\"unit\":\"fahrenheit\"}"
                }}
            ]},
            {"role": "tool", "tool_call_id": "call_1", "content": "72F and sunny"},
            {"role": "user", "content": "Thanks, and what about tomorrow?"}
        ],
        "output_length": 8
    })];
    let trace_file = write_jsonl(h.artifact_dir.path(), "tool_calls_msgs.jsonl", &traces);
    // Default export level so profile_export.jsonl (records + metrics) is written.
    let args = format!(
        "--model {DEFAULT_MODEL} --url {} --endpoint-type chat --streaming \
         --input-file {} --custom-dataset-type mooncake_trace \
         --request-count 1 --concurrency 1 --workers-max 1 --random-seed 42 \
         --ui simple --tokenizer builtin",
        h.mock.url,
        trace_file.display()
    );

    let rust = h.run(&args);
    assert!(rust.success(), "rust run failed:\n{}", rust.stderr);
    let py = h.run_env(&args, &[("AIPERF_RUNTIME_ENGINE", "python")]);
    assert!(py.success(), "python run failed:\n{}", py.stderr);

    let rust_recs = rust.artifacts.jsonl();
    let py_recs = py.artifacts.jsonl();
    assert!(!rust_recs.is_empty(), "rust produced no records");
    assert!(!py_recs.is_empty(), "python produced no records");

    // Byte-identical ISL projection across engines. If the Rust fix double-counted
    // (tool_texts) or dropped (never passed through) the tool_calls, its ISL would
    // diverge from Python's.
    assert_eq!(
        isl_projection(&rust_recs),
        isl_projection(&py_recs),
        "assistant tool_calls ISL diverged between rust and python engines"
    );
}

// ---------------------------------------------------------------------------
// #25 — outputs.json per-record metric allowlist (ISL/TTFT/ITL added)
// ---------------------------------------------------------------------------

/// Sorted metric-key set present on each `outputs.json` row (keyed by
/// session/turn). Metric VALUES are timing-dependent, but which metrics the
/// allowlist emits is deterministic and must match Python.
fn outputs_metric_keys(outputs: &Value) -> Vec<String> {
    sorted(
        outputs
            .get("data")
            .and_then(Value::as_array)
            .map(|rows| {
                rows.iter()
                    .map(|r| {
                        let mut keys: Vec<&str> = r
                            .get("metrics")
                            .and_then(Value::as_object)
                            .map(|m| m.keys().map(String::as_str).collect())
                            .unwrap_or_default();
                        keys.sort();
                        json!({
                            "session_num": r["session_num"],
                            "turn_index": r["turn_index"],
                            "metric_keys": keys,
                        })
                        .to_string()
                    })
                    .collect()
            })
            .unwrap_or_default(),
    )
}

#[tokio::test]
async fn port25_outputs_json_metric_allowlist_parity() {
    let h = AIPerfHarness::new().await;
    // The native CLI enables outputs.json via config (artifacts.export_outputs_json),
    // not a --export-outputs-json flag, so both engines run from a Config-v2 YAML.
    let config_dir = tempfile::TempDir::new().expect("config tempdir");
    let config_path = config_dir.path().join("outputs_json.yaml");
    let config_body = format!(
        r#"schemaVersion: "2.0"
benchmark:
  model: {model}
  endpoint:
    url: {url}
    type: chat
    streaming: true
  dataset:
    type: synthetic
    entries: 100
    prompts:
      isl: 16
      osl: 4
  artifacts:
    export_outputs_json: true
    records:
      - jsonl
  phases:
    - name: profiling
      type: concurrency
      concurrency: 1
      requests: 4
"#,
        model = DEFAULT_MODEL,
        url = h.mock.url,
    );
    std::fs::write(&config_path, config_body).expect("write config");
    let args = format!(
        "--config {} --workers-max 1 --random-seed 42 --ui simple --tokenizer builtin",
        config_path.display()
    );
    let rust = h.run(&args);
    assert!(rust.success(), "rust run failed:\n{}", rust.stderr);
    let py = h.run_env(&args, &[("AIPERF_RUNTIME_ENGINE", "python")]);
    assert!(py.success(), "python run failed:\n{}", py.stderr);

    let rust_out = rust.artifacts.read_json_file("**/outputs.json");
    let py_out = py.artifacts.read_json_file("**/outputs.json");

    // The three metrics this port adds must be present on every rust row.
    let rust_rows = rust_out
        .get("data")
        .expect("rust outputs.json has data array");
    for r in rust_rows
        .as_array()
        .expect("rust outputs.json data is an array")
    {
        let keys = r
            .get("metrics")
            .and_then(Value::as_object)
            .expect("metrics obj");
        for added in [
            "input_sequence_length",
            "time_to_first_token",
            "inter_token_latency",
        ] {
            assert!(
                keys.contains_key(added),
                "rust outputs.json row missing {added}: {r}"
            );
        }
    }

    // Byte-identical metric-key set across engines.
    assert_eq!(
        outputs_metric_keys(&rust_out),
        outputs_metric_keys(&py_out),
        "outputs.json metric allowlist diverged between rust and python engines"
    );
}

// ---------------------------------------------------------------------------
// #13 — console export width honors AIPERF_UI_CONSOLE_EXPORT_WIDTH
// ---------------------------------------------------------------------------

/// The widest rendered line in `profile_export_console.txt` (in display columns,
/// counting chars — the ASCII box art the table uses is 1 col/char).
fn max_console_line_width(path: &std::path::Path) -> usize {
    std::fs::read_to_string(path)
        .expect("read console txt")
        .lines()
        .map(|line| line.chars().count())
        .max()
        .unwrap_or(0)
}

#[tokio::test]
async fn port13_console_export_width_env_parity() {
    let h = AIPerfHarness::new().await;
    let args = format!(
        "--model {DEFAULT_MODEL} --url {} --endpoint-type chat --streaming \
         --concurrency 1 --request-count 4 --workers-max 1 --random-seed 42 \
         --synthetic-input-tokens-mean 16 --output-tokens-mean 4 \
         --ui simple --tokenizer builtin",
        h.mock.url
    );
    // A non-default width both engines must pin the console artifact to.
    let env = &[("AIPERF_UI_CONSOLE_EXPORT_WIDTH", "80")];

    let rust = h.run_env(&args, env);
    assert!(rust.success(), "rust run failed:\n{}", rust.stderr);
    let mut py_env = env.to_vec();
    py_env.push(("AIPERF_RUNTIME_ENGINE", "python"));
    let py = h.run_env(&args, &py_env);
    assert!(py.success(), "python run failed:\n{}", py.stderr);

    let rust_txt = rust
        .artifacts
        .find_file("**/profile_export_console.txt")
        .expect("rust profile_export_console.txt exists");
    let py_txt = py
        .artifacts
        .find_file("**/profile_export_console.txt")
        .expect("python profile_export_console.txt exists");

    let rust_w = max_console_line_width(&rust_txt);
    let py_w = max_console_line_width(&py_txt);
    // Both engines pin the table to the configured 80 columns.
    assert_eq!(rust_w, 80, "rust console width {rust_w} != configured 80");
    assert_eq!(py_w, 80, "python console width {py_w} != configured 80");
}
