// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Integration coverage for the `dataset_load_bench` timed adapter example.
//!
//! Exercises the public load → compose → tokenize path for the four generated
//! fixture formats, the one-line `Sample` JSON schema, and structured errors for
//! unknown formats and missing CLI arguments.
#![cfg(feature = "engine")]

#[path = "../examples/dataset_load_bench.rs"]
mod dataset_load_bench;

use std::fs;
use std::path::PathBuf;
use std::sync::Arc;

use aiperf_runtime::dataset::{
    ComposeConfig, DatasetSource, GeneratedPrompt, LoadConfig, LoaderRegistry, Payload,
    PromptGenerator, PromptGeneratorFactory, SegmentPool, SyntheticDatasetConfig,
    SyntheticPromptConfig, TextTokenizer, TiktokenTokenizer,
};
use aiperf_runtime::rng::{RngRoot, SamplingDistribution};
use dataset_load_bench::{
    Args, BENCHMARK_FORMATS, Sample, inject_prepared_prompt_generator, measure,
    needs_prepared_prompt_generator, parse_args, registry_format_name, verified_options,
};
use serde_json::{Value, json};
use tempfile::TempDir;

#[cfg(feature = "parquet")]
use arrow::array::{Int64Array, RecordBatch, StringArray};
#[cfg(feature = "parquet")]
use arrow::datatypes::{DataType, Field, Schema};

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

#[cfg(feature = "parquet")]
fn write_baseten_columnar_fixtures(dir: &TempDir) -> (PathBuf, PathBuf) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("timestamp_start_unix_ms", DataType::Int64, false),
        Field::new("prompt", DataType::Utf8, false),
        Field::new("input_tokens", DataType::Int64, false),
        Field::new("output_tokens", DataType::Int64, false),
        Field::new("provided_session_id", DataType::Utf8, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int64Array::from(vec![0, 10])),
            Arc::new(StringArray::from(vec!["first", "second"])),
            Arc::new(Int64Array::from(vec![4, 5])),
            Arc::new(Int64Array::from(vec![2, 3])),
            Arc::new(StringArray::from(vec!["shared", "shared"])),
        ],
    )
    .expect("Baseten record batch");

    let parquet_path = dir.path().join("baseten.parquet");
    let parquet_file = fs::File::create(&parquet_path).expect("create Baseten Parquet");
    let mut parquet_writer =
        parquet::arrow::ArrowWriter::try_new(parquet_file, Arc::clone(&schema), None)
            .expect("Parquet writer");
    parquet_writer.write(&batch).expect("write Parquet batch");
    parquet_writer.close().expect("close Parquet writer");

    let arrow_path = dir.path().join("baseten.arrow");
    let arrow_file = fs::File::create(&arrow_path).expect("create Baseten Arrow IPC");
    let mut arrow_writer =
        arrow::ipc::writer::FileWriter::try_new(arrow_file, &schema).expect("Arrow IPC writer");
    arrow_writer.write(&batch).expect("write Arrow IPC batch");
    arrow_writer.finish().expect("finish Arrow IPC writer");
    (parquet_path, arrow_path)
}

fn local_source_json(path: &PathBuf) -> String {
    serde_json::to_string(&json!({
        "kind": "local_file",
        "path": path.to_string_lossy(),
    }))
    .expect("serialize local source")
}

fn synthetic_source_json(format: &str, entries: usize) -> String {
    let inline = match format {
        "synthetic" => json!({
            "marker": "__aiperf_synthetic",
            "synthetic_config": {
                "entries": entries,
                "turns": 1.0,
                "prompts": {
                    "input_tokens": 12.0,
                    "output_tokens": 8.0,
                    "batch_size": 1
                }
            }
        }),
        "synthetic_rankings" => json!({
            "marker": "__aiperf_synthetic_rankings",
            "synthetic_config": {
                "entries": entries,
                "turns": 1.0,
                "rankings": {
                    "passages": 2.0,
                    "passage_tokens": 8.0,
                    "query_tokens": 4.0
                }
            }
        }),
        other => panic!("unsupported synthetic test format {other}"),
    };
    serde_json::to_string(&json!({
        "kind": "inline_synthetic",
        "inline": inline,
    }))
    .expect("serialize synthetic source")
}

fn args_for(format: &str, path: PathBuf, fixture_id: &str) -> Args {
    let options = verified_options(format).expect("verified options");
    Args {
        format: format.to_string(),
        path: Some(path.clone()),
        options_json: serde_json::to_string(&options).expect("serialize options"),
        source_json: local_source_json(&path),
        fixture_id: fixture_id.to_string(),
        seed: 42,
        model: "test-model".to_string(),
        tokenizer: "builtin".to_string(),
        apply_chat_template: false,
        exact_isl: false,
    }
}

fn synthetic_args_for(format: &str, fixture_id: &str, entries: usize) -> Args {
    let options = verified_options(format).expect("verified options");
    Args {
        format: format.to_string(),
        path: None,
        options_json: serde_json::to_string(&options).expect("serialize options"),
        source_json: synthetic_source_json(format, entries),
        fixture_id: fixture_id.to_string(),
        seed: 42,
        model: "test-model".to_string(),
        tokenizer: "builtin".to_string(),
        apply_chat_template: false,
        exact_isl: false,
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

#[cfg(feature = "parquet")]
#[tokio::test(flavor = "current_thread")]
async fn baseten_parquet_and_arrow_adapter_samples_match() {
    let dir = TempDir::new().expect("tempdir");
    let (parquet_path, arrow_path) = write_baseten_columnar_fixtures(&dir);
    let parquet = measure(&args_for("baseten_trace", parquet_path, "baseten-parquet")).await;
    let arrow = measure(&args_for("baseten_trace", arrow_path, "baseten-arrow")).await;
    assert_successful_sample(&parquet, "baseten_trace", "baseten-parquet");
    assert_successful_sample(&arrow, "baseten_trace", "baseten-arrow");
    assert_eq!(arrow.row_count, parquet.row_count);
    assert_eq!(arrow.conversation_count, parquet.conversation_count);
    assert_eq!(arrow.turn_count, parquet.turn_count);
    assert_eq!(arrow.total_input_tokens, parquet.total_input_tokens);
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
async fn synthetic_fixture_reports_logical_row_count_and_tokens() {
    let sample = measure(&synthetic_args_for("synthetic", "synthetic-seed42", 3)).await;
    assert_successful_sample(&sample, "synthetic", "synthetic-seed42");
    assert_eq!(sample.row_count, 3);
    assert_eq!(sample.conversation_count, 3);
    assert_eq!(sample.turn_count, 3);
    assert_eq!(sample.total_input_tokens, Some(36));
}

#[tokio::test(flavor = "current_thread")]
async fn synthetic_rankings_fixture_reports_logical_row_count_and_tokens() {
    let sample = measure(&synthetic_args_for(
        "synthetic_rankings",
        "synthetic-rankings-seed42",
        3,
    ))
    .await;
    assert_successful_sample(&sample, "synthetic_rankings", "synthetic-rankings-seed42");
    assert_eq!(sample.row_count, 3);
    assert_eq!(sample.conversation_count, 3);
    assert_eq!(sample.turn_count, 3);
    assert_eq!(sample.total_input_tokens, Some(60));
}

#[tokio::test(flavor = "current_thread")]
async fn synthetic_template_counts_use_chat_template_when_requested() {
    struct TemplateOnlyTokenizer {
        inner: TiktokenTokenizer,
    }

    impl TemplateOnlyTokenizer {
        fn new() -> Self {
            Self {
                inner: TiktokenTokenizer::builtin(),
            }
        }
    }

    impl TextTokenizer for TemplateOnlyTokenizer {
        fn encode(&self, _text: &str) -> aiperf_runtime::dataset::Result<Vec<u32>> {
            panic!("template-aware synthetic counting should not fall back to bare encode");
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
            "template-only"
        }

        fn apply_chat_template(
            &self,
            messages: &[Value],
            add_generation_prompt: bool,
        ) -> aiperf_runtime::dataset::Result<Option<Vec<u32>>> {
            assert!(add_generation_prompt);
            assert_eq!(messages.len(), 1);
            assert_eq!(messages[0]["role"], "user");
            assert!(
                messages[0]["content"]
                    .as_str()
                    .is_some_and(|content| !content.is_empty())
            );
            Ok(Some(vec![1, 2, 3, 4, 5]))
        }
    }

    let registry = LoaderRegistry::with_builtin_formats().expect("registry");
    let load = LoadConfig::new(DatasetSource::Inline(json!({"__aiperf_synthetic": true})));
    let builtin = TiktokenTokenizer::builtin();
    let mut compose = ComposeConfig::new("test-model", RngRoot::new(Some(42)));
    compose.synthetic_config = Some(SyntheticDatasetConfig {
        entries: 2,
        turns: SamplingDistribution::fixed(1.0).expect("fixed turns"),
        prompts: Some(SyntheticPromptConfig {
            input_tokens: SamplingDistribution::fixed(12.0).expect("fixed isl"),
            batch_size: 1,
            ..SyntheticPromptConfig::default()
        }),
        ..SyntheticDatasetConfig::default()
    });

    let registration = registry.get("synthetic").expect("synthetic format");
    let rows = registration.loader.load(&load).await.expect("load rows");
    let mut pool = SegmentPool::new();
    let conversations = registration
        .composer
        .compose(rows, &compose, &builtin, &mut pool)
        .expect("compose");
    let dataset = aiperf_runtime::dataset::Dataset::new(
        conversations,
        Arc::new(pool.freeze()),
        load.sampling_strategy.as_deref().unwrap_or("sequential"),
        registration
            .loader
            .default_context_mode()
            .unwrap_or(compose.default_context_mode),
    )
    .expect("dataset");

    let tokenizer = TemplateOnlyTokenizer::new();
    let total = dataset_load_bench::benchmark_total_input_tokens(
        "synthetic",
        &dataset,
        &tokenizer,
        "HuggingFaceTB/SmolLM2-135M-Instruct",
        true,
        false,
    )
    .expect("count templated synthetic inputs");

    assert_eq!(total, Some(10));
}

#[tokio::test(flavor = "current_thread")]
async fn synthetic_text_payload_retokenizes_to_the_stored_exact_count() {
    let registry = LoaderRegistry::with_builtin_formats().expect("registry");
    let load = LoadConfig::new(DatasetSource::Inline(json!({"__aiperf_synthetic": true})));
    let tokenizer = TiktokenTokenizer::builtin();
    let mut compose = ComposeConfig::new("test-model", RngRoot::new(Some(42)));
    compose.synthetic_config = Some(SyntheticDatasetConfig {
        entries: 2,
        turns: SamplingDistribution::fixed(1.0).expect("fixed turns"),
        prompts: Some(SyntheticPromptConfig {
            input_tokens: SamplingDistribution::fixed(12.0).expect("fixed isl"),
            batch_size: 1,
            ..SyntheticPromptConfig::default()
        }),
        ..SyntheticDatasetConfig::default()
    });
    inject_prepared_prompt_generator(&mut compose, &tokenizer).expect("prepare");

    let registration = registry.get("synthetic").expect("synthetic format");
    let rows = registration.loader.load(&load).await.expect("load rows");
    let mut pool = SegmentPool::new();
    let conversations = registration
        .composer
        .compose(rows, &compose, &tokenizer, &mut pool)
        .expect("compose");
    for conversation in &conversations {
        for turn in &conversation.turns {
            assert_eq!(turn.input_tokens, Some(12));
        }
    }
    let dataset = aiperf_runtime::dataset::Dataset::new(
        conversations,
        std::sync::Arc::new(pool.freeze()),
        load.sampling_strategy.as_deref().unwrap_or("sequential"),
        registration
            .loader
            .default_context_mode()
            .unwrap_or(compose.default_context_mode),
    )
    .expect("dataset");

    for conversation in dataset.conversations() {
        let handle = conversation.turns[0].content[0].handles[0];
        let Payload::Text {
            bytes, token_count, ..
        } = dataset.segments().get(handle).expect("text payload")
        else {
            panic!("synthetic turn must store text payload");
        };
        let encoded = tokenizer
            .encode(std::str::from_utf8(bytes).expect("utf8"))
            .unwrap();
        assert_eq!(
            encoded.len(),
            *token_count as usize,
            "stored text payload should re-tokenize to its authoritative count"
        );
    }
}

#[tokio::test(flavor = "current_thread")]
async fn synthetic_composition_reuses_generated_tokens_without_reencoding_text() {
    struct StaticPromptGenerator {
        prompt: GeneratedPrompt,
    }

    impl PromptGenerator for StaticPromptGenerator {
        fn generate_token_ids(
            &mut self,
            num_tokens: usize,
            _hash_ids: &[i64],
            _block_size: usize,
        ) -> aiperf_runtime::dataset::Result<Vec<u32>> {
            assert_eq!(num_tokens, self.prompt.tokens.len());
            Ok(self.prompt.tokens.clone())
        }

        fn generate(
            &mut self,
            num_tokens: usize,
            _hash_ids: &[i64],
            _block_size: usize,
        ) -> aiperf_runtime::dataset::Result<GeneratedPrompt> {
            assert_eq!(num_tokens, self.prompt.tokens.len());
            Ok(self.prompt.clone())
        }
    }

    struct StaticPromptGeneratorFactory {
        prompt: GeneratedPrompt,
    }

    impl PromptGeneratorFactory for StaticPromptGeneratorFactory {
        fn create<'a>(
            &self,
            _tokenizer: &'a dyn TextTokenizer,
            _root: RngRoot,
        ) -> aiperf_runtime::dataset::Result<Box<dyn PromptGenerator + 'a>> {
            Ok(Box::new(StaticPromptGenerator {
                prompt: self.prompt.clone(),
            }))
        }
    }

    struct RejectGeneratedTextTokenizer {
        inner: TiktokenTokenizer,
    }

    impl RejectGeneratedTextTokenizer {
        fn new() -> Self {
            Self {
                inner: TiktokenTokenizer::builtin(),
            }
        }
    }

    impl TextTokenizer for RejectGeneratedTextTokenizer {
        fn encode(&self, text: &str) -> aiperf_runtime::dataset::Result<Vec<u32>> {
            if text == "generated prompt" {
                return Err(aiperf_runtime::dataset::DatasetError::Tokenizer(
                    "unexpected synthetic prompt re-encode".into(),
                ));
            }
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
            self.inner.name()
        }
    }

    let registry = LoaderRegistry::with_builtin_formats().expect("registry");
    let load = LoadConfig::new(DatasetSource::Inline(json!({"__aiperf_synthetic": true})));
    let tokenizer = RejectGeneratedTextTokenizer::new();
    let mut compose = ComposeConfig::new("test-model", RngRoot::new(Some(42)));
    compose.prompt_generator = Arc::new(StaticPromptGeneratorFactory {
        prompt: GeneratedPrompt {
            text: "generated prompt".into(),
            tokens: vec![11, 12, 13],
        },
    });
    compose.synthetic_config = Some(SyntheticDatasetConfig {
        entries: 1,
        turns: SamplingDistribution::fixed(1.0).expect("fixed turns"),
        prompts: Some(SyntheticPromptConfig {
            input_tokens: SamplingDistribution::fixed(3.0).expect("fixed isl"),
            batch_size: 1,
            ..SyntheticPromptConfig::default()
        }),
        ..SyntheticDatasetConfig::default()
    });

    let dataset = registry
        .build_dataset(Some("synthetic"), &load, &compose, &tokenizer)
        .await
        .expect("synthetic loader should reuse generated tokens");
    let turn = &dataset.conversations()[0].turns[0];
    assert_eq!(turn.input_tokens, Some(3));
    let handle = turn.content[0].handles[0];
    let Payload::Text {
        bytes, token_count, ..
    } = dataset.segments().get(handle).expect("text payload")
    else {
        panic!("synthetic turn must store text payload");
    };
    assert_eq!(
        std::str::from_utf8(bytes).expect("utf8"),
        "generated prompt"
    );
    assert_eq!(*token_count, 3);
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
        "--path".into(),
        "/tmp/x".into(),
        "--options-json".into(),
        "{}".into(),
        "--fixture-id".into(),
        "x".into(),
        "--seed".into(),
        "42".into(),
        "--model".into(),
        "test-model".into(),
    ])
    .expect_err("missing --source-json must fail");
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
        "--source-json".into(),
        r#"{"kind":"local_file","path":"/tmp/x"}"#.into(),
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

#[test]
fn parse_args_accepts_tokenizer_and_chat_template_flags() {
    let args = parse_args([
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
        "--source-json".into(),
        r#"{"kind":"local_file","path":"/tmp/x"}"#.into(),
        "--tokenizer".into(),
        "openai-community/gpt2".into(),
        "--apply-chat-template".into(),
        "--exact-isl".into(),
    ])
    .expect("flags should parse");

    assert_eq!(args.tokenizer, "openai-community/gpt2");
    assert!(args.apply_chat_template);
    assert!(args.exact_isl);
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

    assert_eq!(BENCHMARK_FORMATS.len(), 22);
    assert_eq!(registry_format_name("burst_gpt_trace"), "burst_gpt_trace");
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
