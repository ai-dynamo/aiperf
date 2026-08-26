// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Native-binary and Python A/B coverage for cumulative per-chunk usage.

mod common;
use common::*;

use std::path::{Path, PathBuf};
use std::process::Command;

use aiperf_mock_server::RequestCapture;
use serde_json::{Value, json};

const PYTHON_ORACLE_COMMIT: &str = "324bb05773b3f99743c6516018f3c30cfe33de0b";
const TTFT_MS: f64 = 100.0;
const ITL_MS: f64 = 25.0;
const OUTPUT_TOKENS: usize = 6;
const FIRST_CHUNK_TOKENS: usize = 3;
const REQUEST_COUNT: usize = 3;

fn mock_config() -> MockServerConfig {
    let mut cfg = MockServerConfig::default();
    cfg.no_tokenizer = true;
    cfg.ttft = TTFT_MS;
    cfg.itl = ITL_MS;
    cfg.ttft_jitter_cv = 0.0;
    cfg.itl_jitter_cv = 0.0;
    cfg.scheduler_enabled = false;
    cfg.fast = false;
    cfg.fixed_output_tokens = Some(OUTPUT_TOKENS);
    cfg.workers = 1;
    // Capture includes health/control traffic as well as workload POSTs; leave
    // enough room that those frames cannot evict a benchmark request.
    cfg.request_capture_capacity = 64;
    cfg
}

struct ExactPythonOracle {
    repository: PathBuf,
    checkout: PathBuf,
    _temporary: tempfile::TempDir,
    is_removed: bool,
}

impl ExactPythonOracle {
    fn materialize() -> Self {
        let repository = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(Path::parent)
            .unwrap_or_else(|| panic!("cannot resolve repository root from CARGO_MANIFEST_DIR"))
            .to_path_buf();
        let temporary = tempfile::TempDir::new().expect("create Python oracle parent directory");
        let checkout = temporary.path().join("origin-main-324bb05773");
        let output = Command::new("git")
            .arg("-C")
            .arg(&repository)
            .args(["worktree", "add", "--detach"])
            .arg(&checkout)
            .arg(PYTHON_ORACLE_COMMIT)
            .output()
            .expect("launch git worktree add for Python oracle");
        assert!(
            output.status.success(),
            "materializing exact Python oracle failed:\nstdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );

        let oracle = Self {
            repository,
            checkout,
            _temporary: temporary,
            is_removed: false,
        };
        oracle.assert_identity();
        oracle.assert_python_import();
        oracle
    }

    fn source_path(&self) -> PathBuf {
        self.checkout.join("src")
    }

    fn assert_identity(&self) {
        let output = Command::new("git")
            .arg("-C")
            .arg(&self.checkout)
            .args(["rev-parse", "HEAD"])
            .output()
            .expect("launch git rev-parse for Python oracle");
        assert!(output.status.success(), "cannot resolve Python oracle HEAD");
        assert_eq!(
            String::from_utf8_lossy(&output.stdout).trim(),
            PYTHON_ORACLE_COMMIT,
            "Python oracle checkout has the wrong commit"
        );
    }

    fn assert_python_import(&self) {
        let virtual_env = std::env::var_os("VIRTUAL_ENV")
            .unwrap_or_else(|| panic!("VIRTUAL_ENV must name the source .venv"));
        let python = PathBuf::from(virtual_env).join("bin/python");
        let output = Command::new(&python)
            .args([
                "-c",
                "from pathlib import Path; import aiperf; print(Path(aiperf.__file__).resolve())",
            ])
            .env("PYTHONPATH", self.source_path())
            .output()
            .unwrap_or_else(|error| panic!("launch exact Python oracle import: {error}"));
        assert!(
            output.status.success(),
            "exact Python oracle import failed:\n{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let imported = PathBuf::from(String::from_utf8_lossy(&output.stdout).trim());
        assert!(
            imported.starts_with(self.source_path()),
            "Python imported {imported:?}, not the exact oracle at {:?}",
            self.source_path()
        );
    }

    fn remove(mut self) {
        self.remove_inner(true);
    }

    fn remove_inner(&mut self, must_succeed: bool) {
        if self.is_removed {
            return;
        }
        let output = Command::new("git")
            .arg("-C")
            .arg(&self.repository)
            .args(["worktree", "remove", "--force"])
            .arg(&self.checkout)
            .output();
        let succeeded = output.as_ref().is_ok_and(|result| result.status.success());
        self.is_removed = succeeded;
        if must_succeed {
            let output = output.expect("launch git worktree remove for Python oracle");
            assert!(
                output.status.success(),
                "removing exact Python oracle failed:\nstdout:\n{}\nstderr:\n{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
        }
    }
}

impl Drop for ExactPythonOracle {
    fn drop(&mut self) {
        self.remove_inner(false);
    }
}

fn captured_chat_requests(harness: &AIPerfHarness) -> Vec<RequestCapture> {
    harness
        .mock
        .state
        .request_captures()
        .into_iter()
        .filter(|capture| capture.route == "/v1/chat/completions")
        .collect()
}

fn fixed_dataset(harness: &AIPerfHarness) -> PathBuf {
    let rows: Vec<Value> = (0..REQUEST_COUNT)
        .map(|index| {
            json!({
                "text": format!("Fixed per-chunk parity prompt {index}"),
                "output_length": OUTPUT_TOKENS,
            })
        })
        .collect();
    write_jsonl(harness.artifact_path(), "fixed-prompts.jsonl", &rows)
}

fn profile_args(url: &str, input_file: &Path, usage_mode: UsageMode) -> String {
    let (per_chunk_flag, extra_inputs) = match usage_mode {
        UsageMode::Requested => (
            "--per-chunk-usage",
            json!({"mock_first_chunk_tokens": FIRST_CHUNK_TOKENS}),
        ),
        UsageMode::Absent => ("", json!({"mock_first_chunk_tokens": FIRST_CHUNK_TOKENS})),
        UsageMode::ExplicitFalse => (
            "--per-chunk-usage",
            json!({
                "mock_first_chunk_tokens": FIRST_CHUNK_TOKENS,
                "stream_options": {"continuous_usage_stats": false},
            }),
        ),
    };
    format!(
        "--model gpt-4 --url {url} --endpoint-type chat --streaming \
         --use-server-token-count {per_chunk_flag} \
         --extra-inputs '{}' --concurrency 1 --request-count {REQUEST_COUNT} \
         --input-file '{}' --custom-dataset-type single_turn \
         --dataset-sampling-strategy sequential \
         --workers-max 1 --random-seed 42 \
         --output-tokens-mean {OUTPUT_TOKENS} --export-level raw --ui none \
         --tokenizer builtin",
        extra_inputs,
        input_file.display()
    )
}

#[derive(Clone, Copy)]
enum UsageMode {
    Requested,
    Absent,
    ExplicitFalse,
}

#[derive(Debug, PartialEq, Eq)]
struct WireProjection {
    continuous_option: Option<bool>,
    include_usage: Option<bool>,
    first_chunk_tokens: Option<u64>,
    content_completion_tokens: Vec<Option<u64>>,
    content_usage_shapes: Vec<Option<(u64, u64, u64)>>,
    terminal_completion_tokens: Vec<u64>,
}

fn data_chunks(record: &Value) -> Vec<Value> {
    record
        .get("responses")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .flat_map(|response| {
            response
                .get("packets")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
        })
        .filter(|packet| packet.get("name").and_then(Value::as_str) == Some("data"))
        .filter_map(|packet| packet.get("value").and_then(Value::as_str))
        .filter(|value| value.trim() != "[DONE]")
        .filter_map(|value| serde_json::from_str(value.trim()).ok())
        .collect()
}

fn content_text(chunk: &Value) -> Option<&str> {
    chunk
        .pointer("/choices/0/delta/content")
        .and_then(Value::as_str)
        .filter(|text| !text.is_empty())
}

fn usage_shape(chunk: &Value) -> Option<(u64, u64, u64)> {
    let usage = chunk.get("usage")?;
    Some((
        usage.get("prompt_tokens")?.as_u64()?,
        usage.get("completion_tokens")?.as_u64()?,
        usage.get("total_tokens")?.as_u64()?,
    ))
}

fn project(record: &Value) -> WireProjection {
    let payload = record
        .get("payload")
        .unwrap_or_else(|| panic!("raw record has no request payload: {record}"));
    let chunks = data_chunks(record);
    let content_chunks: Vec<&Value> = chunks
        .iter()
        .filter(|chunk| content_text(chunk).is_some())
        .collect();
    let terminal_completion_tokens = chunks
        .iter()
        .filter(|chunk| {
            chunk
                .get("choices")
                .and_then(Value::as_array)
                .is_some_and(Vec::is_empty)
        })
        .filter_map(|chunk| chunk.pointer("/usage/completion_tokens"))
        .filter_map(Value::as_u64)
        .collect();

    WireProjection {
        continuous_option: payload
            .pointer("/stream_options/continuous_usage_stats")
            .and_then(Value::as_bool),
        include_usage: payload
            .pointer("/stream_options/include_usage")
            .and_then(Value::as_bool),
        first_chunk_tokens: payload
            .get("mock_first_chunk_tokens")
            .and_then(Value::as_u64),
        content_completion_tokens: content_chunks
            .iter()
            .map(|chunk| {
                chunk
                    .pointer("/usage/completion_tokens")
                    .and_then(Value::as_u64)
            })
            .collect(),
        content_usage_shapes: content_chunks
            .iter()
            .map(|chunk| usage_shape(chunk))
            .collect(),
        terminal_completion_tokens,
    }
}

fn metric_avg(result: &RunResult, name: &str) -> f64 {
    result
        .artifacts
        .json()
        .get(name)
        .and_then(|metric| metric.get("avg"))
        .and_then(Value::as_f64)
        .unwrap_or_else(|| panic!("missing {name}.avg in summary: {}", result.artifacts.json()))
}

fn assert_requested(records: &[Value]) -> Vec<WireProjection> {
    assert_eq!(records.len(), REQUEST_COUNT, "unexpected raw record count");
    records
        .iter()
        .map(project)
        .inspect(|projection| {
            assert_eq!(projection.continuous_option, Some(true));
            assert_eq!(projection.include_usage, Some(true));
            assert_eq!(
                projection.first_chunk_tokens,
                Some(FIRST_CHUNK_TOKENS as u64)
            );
            assert_eq!(
                projection.content_completion_tokens,
                vec![Some(3), Some(4), Some(5), Some(6)],
                "every generated content frame must carry cumulative usage"
            );
            assert_eq!(projection.content_usage_shapes.len(), 4);
            for (index, shape) in projection.content_usage_shapes.iter().enumerate() {
                let (prompt, completion, total) =
                    shape.unwrap_or_else(|| panic!("content frame {index} has no usage"));
                assert_eq!(completion, [3, 4, 5, 6][index]);
                assert_eq!(total, prompt + completion);
            }
            assert_eq!(
                projection.terminal_completion_tokens,
                vec![OUTPUT_TOKENS as u64],
                "include_usage must retain one independent terminal usage frame"
            );
        })
        .collect()
}

fn assert_suppressed(records: &[Value], expected_option: Option<bool>) {
    assert_eq!(records.len(), REQUEST_COUNT, "unexpected raw record count");
    for projection in records.iter().map(project) {
        assert_eq!(projection.continuous_option, expected_option);
        assert_eq!(projection.include_usage, Some(true));
        assert_eq!(
            projection.first_chunk_tokens,
            Some(FIRST_CHUNK_TOKENS as u64)
        );
        assert_eq!(
            projection.content_completion_tokens,
            vec![None, None, None, None],
            "absent/false continuous usage must suppress usage on content frames"
        );
        assert!(projection.content_usage_shapes.iter().all(Option::is_none));
        assert_eq!(
            projection.terminal_completion_tokens,
            vec![OUTPUT_TOKENS as u64],
            "terminal include_usage behavior is independent of continuous usage"
        );
    }
}

fn assert_near(actual: f64, expected: f64, tolerance: f64, label: &str) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "{label}: {actual:.3}ms is not within {tolerance:.3}ms of {expected:.3}ms"
    );
}

#[tokio::test]
async fn native_binary_and_python_match_per_chunk_usage_wire_and_itl() {
    let oracle = ExactPythonOracle::materialize();
    let harness = AIPerfHarness::new_with(mock_config()).await;
    let input_file = fixed_dataset(&harness);
    let args = profile_args(&harness.mock.url, &input_file, UsageMode::Requested);

    let rust = harness.run_in(&args, "native");
    assert!(rust.success(), "native Rust run failed:\n{}", rust.stderr);
    let rust_projection = assert_requested(&rust.artifacts.raw_records());
    let rust_itl = metric_avg(&rust, "inter_token_latency");
    assert_near(rust_itl, ITL_MS, 3.0, "native corrected ITL");
    let rust_requests = captured_chat_requests(&harness);
    assert_eq!(rust_requests.len(), REQUEST_COUNT);
    harness.mock.state.clear_request_captures();

    let python_path = oracle.source_path().display().to_string();
    let python = harness.run_env_in(
        &args,
        "python-324bb05773",
        &[
            ("AIPERF_RUNTIME_ENGINE", "python"),
            ("AIPERF_E2E_PYTHON_MODULE", "aiperf"),
            ("PYTHONPATH", python_path.as_str()),
        ],
    );
    assert!(python.success(), "Python run failed:\n{}", python.stderr);
    let python_projection = assert_requested(&python.artifacts.raw_records());
    let python_itl = metric_avg(&python, "inter_token_latency");
    assert_near(python_itl, ITL_MS, 3.0, "Python corrected ITL");
    let python_requests = captured_chat_requests(&harness);
    assert_eq!(python_requests.len(), REQUEST_COUNT);

    assert_eq!(
        rust_projection, python_projection,
        "feature-relevant raw wire projection must be byte-exact across engines"
    );
    for (index, (rust_request, python_request)) in
        rust_requests.iter().zip(&python_requests).enumerate()
    {
        assert_eq!(
            rust_request.method, python_request.method,
            "request {index}"
        );
        assert_eq!(rust_request.route, python_request.route, "request {index}");
        assert_eq!(
            rust_request.body, python_request.body,
            "request {index} outbound body bytes differ"
        );
    }
    assert!(
        (rust_itl - python_itl).abs() <= 2.0,
        "corrected ITL parity diverged: rust={rust_itl:.3}ms python={python_itl:.3}ms"
    );
    oracle.remove();
}

#[tokio::test]
async fn native_binary_suppresses_absent_or_false_continuous_usage_and_falls_back() {
    for (mode, expected_option) in [
        (UsageMode::Absent, None),
        (UsageMode::ExplicitFalse, Some(false)),
    ] {
        let harness = AIPerfHarness::new_with(mock_config()).await;
        let input_file = fixed_dataset(&harness);
        let result = harness.run(&profile_args(&harness.mock.url, &input_file, mode));
        assert!(
            result.success(),
            "native Rust run failed:\n{}",
            result.stderr
        );
        assert_suppressed(&result.artifacts.raw_records(), expected_option);

        let legacy_itl =
            ITL_MS * (OUTPUT_TOKENS - FIRST_CHUNK_TOKENS) as f64 / (OUTPUT_TOKENS - 1) as f64;
        assert_near(
            metric_avg(&result, "inter_token_latency"),
            legacy_itl,
            3.0,
            "legacy ITL fallback",
        );
    }
}
