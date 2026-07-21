// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Integration coverage for the `dataset_load_bench` timed adapter example.
//!
//! Exercises the public load → compose → tokenize path for the four generated
//! fixture formats, the one-line `Sample` JSON schema, and structured errors for
//! unknown formats and missing CLI arguments.

#[path = "../examples/dataset_load_bench.rs"]
mod dataset_load_bench;

use std::fs;
use std::path::PathBuf;

use dataset_load_bench::{
    Args, BENCHMARK_FORMATS, Sample, inject_prepared_prompt_generator, measure,
    needs_prepared_prompt_generator, parse_args, registry_format_name,
};
use serde_json::{Value, json};
use tempfile::TempDir;

fn write_jsonl(dir: &TempDir, name: &str, rows: &[Value]) -> PathBuf {
    let path = dir.path().join(name);
    let mut body = String::new();
    for row in rows {
        body.push_str(&serde_json::to_string(row).expect("serialize fixture row"));
        body.push('\n');
    }
    fs::write(&path, body).expect("write jsonl fixture");
    path
}

fn write_json(dir: &TempDir, name: &str, value: &Value) -> PathBuf {
    let path = dir.path().join(name);
    fs::write(
        &path,
        serde_json::to_vec_pretty(value).expect("serialize json fixture"),
    )
    .expect("write json fixture");
    path
}

fn args_for(format: &str, path: PathBuf, fixture_id: &str) -> Args {
    Args {
        format: format.to_string(),
        path,
        options_json: "{}".to_string(),
        fixture_id: fixture_id.to_string(),
        seed: 42,
        model: "test-model".to_string(),
    }
}

fn assert_successful_sample(sample: &Sample, format: &str, fixture_id: &str) {
    assert_eq!(sample.implementation, "rust");
    assert_eq!(sample.format, format);
    assert_eq!(sample.fixture_id, fixture_id);
    assert!(
        sample.error.is_none(),
        "unexpected error: {:?}",
        sample.error
    );
    assert!(sample.row_count > 0, "row_count must be positive");
    assert!(
        sample.conversation_count > 0,
        "conversation_count must be positive"
    );
    assert!(sample.turn_count > 0, "turn_count must be positive");
    assert!(sample.elapsed_ns > 0, "elapsed_ns must be positive");
}

#[tokio::test(flavor = "current_thread")]
async fn single_turn_fixture_reports_positive_counts() {
    let dir = TempDir::new().expect("tempdir");
    let path = write_jsonl(
        &dir,
        "single_turn.jsonl",
        &[
            json!({"text": "alpha beta gamma"}),
            json!({"session_id": "s-a", "text": "turn one"}),
            json!({"session_id": "s-a", "text": "turn two"}),
        ],
    );
    let sample = measure(&args_for("single_turn", path, "single_turn-seed42")).await;
    assert_successful_sample(&sample, "single_turn", "single_turn-seed42");
    assert_eq!(sample.row_count, 3);
    assert_eq!(sample.conversation_count, 2);
    assert_eq!(sample.turn_count, 3);
    assert!(sample.total_input_tokens.unwrap_or(0) > 0);
}

#[tokio::test(flavor = "current_thread")]
async fn multi_turn_fixture_reports_positive_counts() {
    let dir = TempDir::new().expect("tempdir");
    let path = write_jsonl(
        &dir,
        "multi_turn.jsonl",
        &[
            json!({"session_id": "m1", "turns": [{"text": "q1"}, {"text": "q2"}]}),
            json!({"session_id": "m2", "turns": [{"text": "only"}]}),
        ],
    );
    let sample = measure(&args_for("multi_turn", path, "multi_turn-seed42")).await;
    assert_successful_sample(&sample, "multi_turn", "multi_turn-seed42");
    assert_eq!(sample.row_count, 2);
    assert_eq!(sample.conversation_count, 2);
    assert_eq!(sample.turn_count, 3);
    assert!(sample.total_input_tokens.unwrap_or(0) > 0);
}

#[tokio::test(flavor = "current_thread")]
async fn raw_payload_fixture_reports_positive_counts() {
    let dir = TempDir::new().expect("tempdir");
    let path = write_jsonl(
        &dir,
        "raw_payload.jsonl",
        &[
            json!({
                "messages": [{"role": "user", "content": "hi"}],
                "model": "test-model",
                "max_tokens": 16
            }),
            json!({
                "messages": [{"role": "user", "content": "bye"}],
                "model": "test-model",
                "max_tokens": 16
            }),
        ],
    );
    let sample = measure(&args_for("raw_payload", path, "raw_payload-seed42")).await;
    assert_successful_sample(&sample, "raw_payload", "raw_payload-seed42");
    assert_eq!(sample.row_count, 2);
    assert_eq!(sample.conversation_count, 2);
    assert_eq!(sample.turn_count, 2);
    assert_eq!(sample.total_input_tokens, None);
}

#[tokio::test(flavor = "current_thread")]
async fn inputs_json_fixture_reports_positive_counts() {
    let dir = TempDir::new().expect("tempdir");
    let raw_payload = [
        json!({
            "messages": [{"role": "user", "content": "hi"}],
            "model": "test-model",
            "max_tokens": 16
        }),
        json!({
            "messages": [{"role": "user", "content": "bye"}],
            "model": "test-model",
            "max_tokens": 16
        }),
    ];
    let path = write_json(
        &dir,
        "inputs.json",
        &json!({
            "data": [
                {"session_id": "session-001", "payloads": raw_payload},
                {"session_id": "session-002", "payloads": [raw_payload[0]]},
            ]
        }),
    );
    let sample = measure(&args_for("inputs_json", path, "inputs_json-seed42")).await;
    assert_successful_sample(&sample, "inputs_json", "inputs_json-seed42");
    assert_eq!(sample.row_count, 3);
    assert_eq!(sample.conversation_count, 2);
    assert_eq!(sample.turn_count, 3);
    assert_eq!(sample.total_input_tokens, None);
}

#[tokio::test(flavor = "current_thread")]
async fn five_additional_local_formats_report_expected_counts() {
    let dir = TempDir::new().expect("tempdir");
    let random_pool = write_jsonl(
        &dir,
        "random_pool.jsonl",
        &[json!({"text": "alpha beta gamma"})],
    );
    let mooncake = write_jsonl(
        &dir,
        "mooncake.jsonl",
        &[
            json!({"timestamp": 0, "text_input": "alpha beta gamma", "output_length": 16}),
            json!({"timestamp": 1, "text_input": "turn one", "output_length": 16}),
        ],
    );
    let bailian = write_jsonl(
        &dir,
        "bailian.jsonl",
        &[
            json!({"chat_id": 1, "parent_chat_id": -1, "timestamp": 0, "input_length": 3, "output_length": 16, "type": "text", "turn": 1}),
            json!({"chat_id": 2, "parent_chat_id": 1, "timestamp": 1, "input_length": 2, "output_length": 16, "type": "text", "turn": 2}),
        ],
    );
    let burst = dir.path().join("burst.csv");
    fs::write(
        &burst,
        "Timestamp,Request tokens,Response tokens\n0,3,16\n1,2,16\n",
    )
    .expect("write BurstGPT fixture");
    let captured_input =
        r#"{"messages":[{"role":"user","content":"alpha beta gamma"}],"max_tokens":16}"#;
    let captured_output = r#"{"usage":{"completion_tokens":2}}"#;
    let sagemaker = write_jsonl(
        &dir,
        "sagemaker.jsonl",
        &[json!({
            "captureData": {
                "endpointInput": {"data": captured_input, "encoding": "JSON"},
                "endpointOutput": {"data": captured_output, "encoding": "JSON"}
            },
            "eventMetadata": {
                "eventId": "event-0",
                "inferenceTime": "2026-07-20T00:00:00Z"
            }
        })],
    );

    for (format, path, expected) in [
        ("random_pool", random_pool, (1, 1, 1)),
        ("mooncake_trace", mooncake, (2, 2, 2)),
        ("bailian_trace", bailian, (2, 1, 2)),
        ("burst_gpt_trace", burst, (2, 2, 2)),
        ("sagemaker_data_capture", sagemaker, (1, 1, 1)),
    ] {
        let sample = measure(&args_for(format, path, format)).await;
        assert_successful_sample(&sample, format, format);
        assert!(
            sample.total_input_tokens.unwrap_or(0) > 0,
            "format={format}"
        );
        assert_eq!(
            (
                sample.row_count,
                sample.conversation_count,
                sample.turn_count
            ),
            expected,
            "format={format}"
        );
    }
}

#[test]
fn sample_serializes_to_exact_one_line_schema() {
    let sample = Sample {
        implementation: "rust",
        format: "single_turn".to_string(),
        fixture_id: "fixture-a".to_string(),
        row_count: 3,
        conversation_count: 2,
        turn_count: 3,
        total_input_tokens: Some(11),
        elapsed_ns: 12345,
        error: None,
    };
    let line = sample.to_json_line();
    assert!(!line.contains('\n'), "sample must be one line");
    let value: Value = serde_json::from_str(&line).expect("parse sample json");
    assert_eq!(
        value
            .as_object()
            .expect("object")
            .keys()
            .collect::<Vec<_>>(),
        vec![
            "implementation",
            "format",
            "fixture_id",
            "row_count",
            "conversation_count",
            "turn_count",
            "total_input_tokens",
            "elapsed_ns",
            "error",
        ]
    );
    assert_eq!(value["implementation"], "rust");
    assert_eq!(value["format"], "single_turn");
    assert_eq!(value["fixture_id"], "fixture-a");
    assert_eq!(value["row_count"], 3);
    assert_eq!(value["conversation_count"], 2);
    assert_eq!(value["turn_count"], 3);
    assert_eq!(value["total_input_tokens"], 11);
    assert_eq!(value["elapsed_ns"], 12345);
    assert!(value["error"].is_null());
}

#[tokio::test(flavor = "current_thread")]
async fn unknown_format_emits_structured_error_sample() {
    let dir = TempDir::new().expect("tempdir");
    let path = write_jsonl(&dir, "empty.jsonl", &[json!({"text": "x"})]);
    let sample = measure(&args_for("not_a_format", path, "bad-format")).await;
    assert_eq!(sample.implementation, "rust");
    assert_eq!(sample.format, "not_a_format");
    assert_eq!(sample.fixture_id, "bad-format");
    assert!(sample.error.is_some(), "error must be populated");
    assert_eq!(sample.row_count, 0);
    assert_eq!(sample.conversation_count, 0);
    assert_eq!(sample.turn_count, 0);
    assert_eq!(sample.total_input_tokens, None);
    assert_eq!(sample.elapsed_ns, 0);
}

#[tokio::test(flavor = "current_thread")]
async fn nonempty_options_emit_structured_error_sample() {
    let dir = TempDir::new().expect("tempdir");
    let path = write_jsonl(&dir, "single_turn.jsonl", &[json!({"text": "hello"})]);
    let mut args = args_for("single_turn", path, "options");
    args.options_json = r#"{"text_field":"text"}"#.to_string();
    let sample = measure(&args).await;
    assert_eq!(sample.implementation, "rust");
    assert_eq!(sample.format, "single_turn");
    assert_eq!(sample.fixture_id, "options");
    assert_eq!(
        sample.error.as_deref(),
        Some(dataset_load_bench::NON_EMPTY_OPTIONS_REASON)
    );
    assert_eq!(sample.elapsed_ns, 0);
    assert_eq!(sample.row_count, 0);
}

#[test]
fn missing_required_argument_emits_structured_error_sample() {
    let sample = parse_args([
        "dataset_load_bench".into(),
        "--format".into(),
        "single_turn".into(),
    ])
    .expect_err("missing --path must fail");
    assert_eq!(sample.implementation, "rust");
    assert!(sample.error.is_some());
    assert_eq!(sample.elapsed_ns, 0);
    assert_eq!(sample.row_count, 0);
}

#[test]
fn unknown_cli_flag_emits_structured_error_sample() {
    let sample = parse_args([
        "dataset_load_bench".into(),
        "--format".into(),
        "single_turn".into(),
        "--path".into(),
        "/tmp/x".into(),
        "--fixture-id".into(),
        "x".into(),
        "--seed".into(),
        "42".into(),
        "--model".into(),
        "test-model".into(),
        "--options-json".into(),
        "{}".into(),
        "--not-a-flag".into(),
    ])
    .expect_err("unknown flag must fail");
    assert_eq!(sample.implementation, "rust");
    assert!(
        sample
            .error
            .as_deref()
            .is_some_and(|error| error.contains("unknown")),
        "error={:?}",
        sample.error
    );
}

#[tokio::test(flavor = "current_thread")]
async fn counts_are_deterministic_across_two_runs() {
    let dir = TempDir::new().expect("tempdir");
    let path = write_jsonl(
        &dir,
        "single_turn.jsonl",
        &[
            json!({"text": "alpha beta gamma"}),
            json!({"session_id": "s-a", "text": "turn one"}),
            json!({"session_id": "s-a", "text": "turn two"}),
        ],
    );
    let args = args_for("single_turn", path, "det");
    let first = measure(&args).await;
    let second = measure(&args).await;
    assert_eq!(first.row_count, second.row_count);
    assert_eq!(first.conversation_count, second.conversation_count);
    assert_eq!(first.turn_count, second.turn_count);
    assert_eq!(first.total_input_tokens, second.total_input_tokens);
    assert!(first.total_input_tokens.unwrap_or(0) > 0);
}

#[test]
fn prepared_prompt_generator_flags_cover_entire_supported_catalog() {
    // One authoritative table drives prepare dispatch; this test locks the
    // prepare set to the three composers that always call `create`, and asserts
    // every other verified harness format stays cold so fairness cannot drift.
    let preparing: Vec<&str> = BENCHMARK_FORMATS
        .iter()
        .filter(|format| format.prepare_prompt_generator)
        .map(|format| format.name)
        .collect();
    assert_eq!(
        preparing,
        ["mooncake_trace", "bailian_trace", "burst_gpt_trace"]
    );

    for format in BENCHMARK_FORMATS {
        assert_eq!(
            needs_prepared_prompt_generator(format.name),
            format.prepare_prompt_generator,
            "dispatch must follow BENCHMARK_FORMATS for {}",
            format.name
        );
        assert_eq!(
            format.prepare_prompt_generator,
            matches!(
                format.name,
                "mooncake_trace" | "bailian_trace" | "burst_gpt_trace"
            ),
            "unexpected prepare flag for {}",
            format.name
        );
        assert!(!format.registry_name.is_empty());
        assert_eq!(registry_format_name(format.name), format.registry_name);
    }

    assert_eq!(BENCHMARK_FORMATS.len(), 9);
    assert_eq!(registry_format_name("burst_gpt_trace"), "burst_gpt");
    assert!(!needs_prepared_prompt_generator("not_a_format"));
}

#[tokio::test(flavor = "current_thread")]
async fn prepared_trace_compose_does_not_reencode_corpus() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use aiperf_runtime::dataset::{
        ComposeConfig, DatasetSource, LoadConfig, LoaderRegistry, PromptGeneratorFactory,
        SegmentPool, TextTokenizer, TiktokenTokenizer,
    };
    use aiperf_runtime::rng::RngRoot;

    struct CountingTokenizer {
        inner: TiktokenTokenizer,
        encodes: AtomicUsize,
    }

    impl CountingTokenizer {
        fn new() -> Self {
            Self {
                inner: TiktokenTokenizer::builtin(),
                encodes: AtomicUsize::new(0),
            }
        }

        fn encode_count(&self) -> usize {
            self.encodes.load(Ordering::SeqCst)
        }
    }

    impl TextTokenizer for CountingTokenizer {
        fn encode(&self, text: &str) -> aiperf_runtime::dataset::Result<Vec<u32>> {
            self.encodes.fetch_add(1, Ordering::SeqCst);
            self.inner.encode(text)
        }

        fn decode(&self, token_ids: &[u32]) -> aiperf_runtime::dataset::Result<String> {
            self.inner.decode(token_ids)
        }

        fn bos_token_id(&self) -> Option<u32> {
            self.inner.bos_token_id()
        }

        fn eos_token_id(&self) -> Option<u32> {
            self.inner.eos_token_id()
        }

        fn vocab_size(&self) -> Option<u32> {
            self.inner.vocab_size()
        }

        fn name(&self) -> &str {
            "counting"
        }
    }

    let dir = TempDir::new().expect("tempdir");
    let path = write_jsonl(
        &dir,
        "bailian.jsonl",
        &[
            json!({"chat_id": 1, "parent_chat_id": -1, "timestamp": 0, "input_length": 3, "output_length": 16, "type": "text", "turn": 1}),
            json!({"chat_id": 2, "parent_chat_id": 1, "timestamp": 1, "input_length": 2, "output_length": 16, "type": "text", "turn": 2}),
        ],
    );

    let tokenizer = CountingTokenizer::new();
    let mut compose_config = ComposeConfig::new("test-model", RngRoot::new(Some(42)));
    inject_prepared_prompt_generator(&mut compose_config, &tokenizer).expect("prepare");
    let encodes_after_prepare = tokenizer.encode_count();
    assert!(
        encodes_after_prepare > 0,
        "prepare must tokenize the sonnet corpus"
    );

    // Cold create would re-tokenize the full corpus; prepared create must not.
    PromptGeneratorFactory::create(
        compose_config.prompt_generator.as_ref(),
        &tokenizer,
        RngRoot::new(Some(42)),
    )
    .expect("create from prepared");
    assert_eq!(
        tokenizer.encode_count(),
        encodes_after_prepare,
        "create after prepare must not re-encode the corpus"
    );

    let registry = LoaderRegistry::with_builtin_formats().expect("registry");
    let registration = registry.get("bailian_trace").expect("bailian");
    let load_config = LoadConfig::new(DatasetSource::Path(path));
    let rows = registration
        .loader
        .load(&load_config)
        .await
        .expect("load bailian rows");
    let mut pool = SegmentPool::new();
    let conversations = registration
        .composer
        .compose(rows, &compose_config, &tokenizer, &mut pool)
        .expect("compose bailian");
    assert_eq!(conversations.len(), 1);
    assert_eq!(conversations[0].turns.len(), 2);
    assert_eq!(
        tokenizer.encode_count(),
        encodes_after_prepare,
        "timed compose path must sample prepared tokens without corpus encode"
    );
}
