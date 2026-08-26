// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Timed Rust adapter for the Python/Rust dataset-load comparison harness.
//!
//! Resolves and warms the `--tokenizer` spec (built-in tiktoken encoding, local
//! tokenizer directory/file, or a downloaded Hugging Face repo) before
//! measurement, then times load → compose → benchmark accounting for one
//! fixture. Emits a single one-line JSON [`Sample`] record on stdout. Argument
//! parsing is deliberately minimal and dependency-free. Benchmark-mode usage
//! notes live in `dev/benchmarks/README.md`.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use aiperf_runtime::dataset::{
    ComposeConfig, Conversation, CorpusPromptGeneratorFactory, Dataset, DatasetSource,
    HuggingFaceTokenizer, LoadConfig, LoaderRegistry, MediaKind, NativeTiktokenTokenizer, Payload,
    SegmentPool, SyntheticDatasetConfig, SyntheticPromptConfig, SyntheticRankingsConfig,
    TextTokenizer, TiktokenEncoding, TiktokenTokenizer, Turn, download_hugging_face_tokenizer,
    find_tiktoken_model_file,
};
use aiperf_runtime::endpoints::extract_payload;
use aiperf_runtime::rng::{RngRoot, SamplingDistribution};
use serde::Serialize;
use serde_json::{Map, Value, json};

/// Shared reject reason for options outside the verified mapping.
pub const NON_EMPTY_OPTIONS_REASON: &str =
    "options do not match the verified cross-stack option mapping";

/// One verified harness format and how the Rust adapter should treat it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BenchmarkFormat {
    pub name: &'static str,
    pub registry_name: &'static str,
    pub prepare_prompt_generator: bool,
    pub opaque_token_counts: bool,
}

/// Authoritative catalog of formats this adapter may measure.
pub const BENCHMARK_FORMATS: &[BenchmarkFormat] = &[
    BenchmarkFormat {
        name: "single_turn",
        registry_name: "single_turn",
        prepare_prompt_generator: false,
        opaque_token_counts: false,
    },
    BenchmarkFormat {
        name: "multi_turn",
        registry_name: "multi_turn",
        prepare_prompt_generator: false,
        opaque_token_counts: false,
    },
    BenchmarkFormat {
        name: "raw_payload",
        registry_name: "raw_payload",
        prepare_prompt_generator: false,
        opaque_token_counts: true,
    },
    BenchmarkFormat {
        name: "inputs_json",
        registry_name: "inputs_json",
        prepare_prompt_generator: false,
        opaque_token_counts: true,
    },
    BenchmarkFormat {
        name: "random_pool",
        registry_name: "random_pool",
        prepare_prompt_generator: false,
        opaque_token_counts: false,
    },
    BenchmarkFormat {
        name: "mooncake_trace",
        registry_name: "mooncake_trace",
        prepare_prompt_generator: true,
        opaque_token_counts: false,
    },
    BenchmarkFormat {
        name: "bailian_trace",
        registry_name: "bailian_trace",
        prepare_prompt_generator: true,
        opaque_token_counts: false,
    },
    BenchmarkFormat {
        name: "burst_gpt_trace",
        registry_name: "burst_gpt_trace",
        prepare_prompt_generator: true,
        opaque_token_counts: false,
    },
    BenchmarkFormat {
        name: "sagemaker_data_capture",
        registry_name: "sagemaker_data_capture",
        prepare_prompt_generator: false,
        opaque_token_counts: false,
    },
    BenchmarkFormat {
        name: "baseten_trace",
        registry_name: "baseten_trace",
        prepare_prompt_generator: false,
        opaque_token_counts: false,
    },
    BenchmarkFormat {
        name: "speed_bench",
        registry_name: "speed_bench",
        prepare_prompt_generator: false,
        opaque_token_counts: false,
    },
    BenchmarkFormat {
        name: "synthetic",
        registry_name: "synthetic",
        prepare_prompt_generator: false,
        opaque_token_counts: false,
    },
    BenchmarkFormat {
        name: "synthetic_rankings",
        registry_name: "synthetic_rankings",
        prepare_prompt_generator: false,
        opaque_token_counts: false,
    },
    BenchmarkFormat {
        name: "sharegpt",
        registry_name: "sharegpt",
        prepare_prompt_generator: false,
        opaque_token_counts: true,
    },
    BenchmarkFormat {
        name: "hf_instruction_response",
        registry_name: "hf_instruction_response",
        prepare_prompt_generator: false,
        opaque_token_counts: true,
    },
    BenchmarkFormat {
        name: "hf_conversation",
        registry_name: "hf_conversation",
        prepare_prompt_generator: false,
        opaque_token_counts: true,
    },
    BenchmarkFormat {
        name: "hf_asr",
        registry_name: "hf_asr",
        prepare_prompt_generator: false,
        opaque_token_counts: true,
    },
    BenchmarkFormat {
        name: "mt_bench",
        registry_name: "mt_bench",
        prepare_prompt_generator: false,
        opaque_token_counts: true,
    },
    BenchmarkFormat {
        name: "mmvu",
        registry_name: "mmvu",
        prepare_prompt_generator: false,
        opaque_token_counts: true,
    },
    BenchmarkFormat {
        name: "spec_bench",
        registry_name: "spec_bench",
        prepare_prompt_generator: false,
        opaque_token_counts: true,
    },
    BenchmarkFormat {
        name: "exgentic",
        registry_name: "exgentic",
        prepare_prompt_generator: false,
        opaque_token_counts: true,
    },
    BenchmarkFormat {
        name: "exgentic_v2",
        registry_name: "exgentic_v2",
        prepare_prompt_generator: false,
        opaque_token_counts: true,
    },
];

/// Verified option objects keyed by harness format name.
pub fn verified_options(format: &str) -> Result<Map<String, Value>, String> {
    let mut map = Map::new();
    match format {
        "speed_bench" => {
            map.insert("category".into(), Value::String("coding".into()));
        }
        "hf_instruction_response" => {
            map.insert("prompt_column".into(), Value::String("question".into()));
        }
        "hf_conversation" => {
            map.insert(
                "conversation_column".into(),
                Value::String("conversations".into()),
            );
            map.insert("message_content_key".into(), Value::String("value".into()));
        }
        "hf_asr" => {
            map.insert("audio_column".into(), Value::String("audio".into()));
        }
        "mmvu" => {
            map.insert("video_column".into(), Value::String("video".into()));
        }
        "exgentic" | "exgentic_v2" => {
            map.insert("max_conversations".into(), Value::from(3_u64));
        }
        _ => {}
    }
    Ok(map)
}

pub fn benchmark_format(name: &str) -> Option<&'static BenchmarkFormat> {
    BENCHMARK_FORMATS.iter().find(|format| format.name == name)
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct Sample {
    pub implementation: &'static str,
    pub format: String,
    pub fixture_id: String,
    pub row_count: usize,
    pub conversation_count: usize,
    pub turn_count: usize,
    pub total_input_tokens: Option<u64>,
    pub elapsed_ns: u128,
    pub error: Option<String>,
}

impl Sample {
    pub fn error_sample(
        format: impl Into<String>,
        fixture_id: impl Into<String>,
        error: impl Into<String>,
    ) -> Self {
        Self {
            implementation: "rust",
            format: format.into(),
            fixture_id: fixture_id.into(),
            row_count: 0,
            conversation_count: 0,
            turn_count: 0,
            total_input_tokens: None,
            elapsed_ns: 0,
            error: Some(error.into()),
        }
    }

    pub fn to_json_line(&self) -> String {
        serde_json::to_string(self).expect("Sample serialization is infallible")
    }
}

#[derive(Debug, Clone)]
pub struct Args {
    pub format: String,
    pub path: Option<PathBuf>,
    pub options_json: String,
    pub source_json: String,
    pub fixture_id: String,
    pub seed: u64,
    pub model: String,
    pub tokenizer: String,
    pub apply_chat_template: bool,
    pub exact_isl: bool,
}

pub fn parse_args(argv: impl IntoIterator<Item = String>) -> Result<Args, Sample> {
    let mut argv = argv.into_iter();
    let _exe = argv.next();

    let mut format = None;
    let mut path = None;
    let mut options_json = None;
    let mut source_json = None;
    let mut fixture_id = None;
    let mut seed = None;
    let mut model = None;
    let mut tokenizer = Some("builtin".to_string());
    let mut apply_chat_template = false;
    let mut exact_isl = false;

    while let Some(flag) = argv.next() {
        match flag.as_str() {
            "--format" | "--path" | "--options-json" | "--source-json" | "--fixture-id"
            | "--seed" | "--model" | "--tokenizer" => {}
            "--apply-chat-template" => {
                apply_chat_template = true;
                continue;
            }
            "--exact-isl" => {
                exact_isl = true;
                continue;
            }
            other => {
                return Err(Sample::error_sample(
                    format.clone().unwrap_or_default(),
                    fixture_id.clone().unwrap_or_default(),
                    format!("unknown argument {other}"),
                ));
            }
        }
        let value = argv.next().ok_or_else(|| {
            Sample::error_sample(
                format.clone().unwrap_or_default(),
                fixture_id.clone().unwrap_or_default(),
                format!("missing value for argument {flag}"),
            )
        })?;
        match flag.as_str() {
            "--format" => format = Some(value),
            "--path" => {
                path = if value.is_empty() {
                    None
                } else {
                    Some(PathBuf::from(value))
                }
            }
            "--options-json" => options_json = Some(value),
            "--source-json" => source_json = Some(value),
            "--fixture-id" => fixture_id = Some(value),
            "--seed" => {
                let parsed = value.parse::<u64>().map_err(|error| {
                    Sample::error_sample(
                        format.clone().unwrap_or_default(),
                        fixture_id.clone().unwrap_or_default(),
                        format!("invalid --seed {value:?}: {error}"),
                    )
                })?;
                seed = Some(parsed);
            }
            "--model" => model = Some(value),
            "--tokenizer" => tokenizer = Some(value),
            _ => unreachable!("flag validated above"),
        }
    }

    let format = format.ok_or_else(|| {
        Sample::error_sample(
            "",
            fixture_id.clone().unwrap_or_default(),
            "missing required argument --format",
        )
    })?;
    let options_json = options_json.ok_or_else(|| {
        Sample::error_sample(
            format.clone(),
            fixture_id.clone().unwrap_or_default(),
            "missing required argument --options-json",
        )
    })?;
    let source_json = source_json.ok_or_else(|| {
        Sample::error_sample(
            format.clone(),
            fixture_id.clone().unwrap_or_default(),
            "missing required argument --source-json",
        )
    })?;
    let fixture_id = fixture_id.ok_or_else(|| {
        Sample::error_sample(format.clone(), "", "missing required argument --fixture-id")
    })?;
    let seed = seed.ok_or_else(|| {
        Sample::error_sample(
            format.clone(),
            fixture_id.clone(),
            "missing required argument --seed",
        )
    })?;
    let model = model.ok_or_else(|| {
        Sample::error_sample(
            format.clone(),
            fixture_id.clone(),
            "missing required argument --model",
        )
    })?;

    Ok(Args {
        format,
        path,
        options_json,
        source_json,
        fixture_id,
        seed,
        model,
        tokenizer: tokenizer.expect("default tokenizer is always set"),
        apply_chat_template,
        exact_isl,
    })
}

pub async fn measure(args: &Args) -> Sample {
    let tokenizer = match load_tokenizer(&args.tokenizer).await {
        Ok(tokenizer) => tokenizer,
        Err(error) => return Sample::error_sample(&args.format, &args.fixture_id, error),
    };
    if let Err(error) = tokenizer.encode("warm") {
        return Sample::error_sample(&args.format, &args.fixture_id, error.to_string());
    }

    let options = match parse_options_map(&args.options_json) {
        Ok(options) => options,
        Err(error) => return Sample::error_sample(&args.format, &args.fixture_id, error),
    };
    let expected = match verified_options(&args.format) {
        Ok(expected) => expected,
        Err(error) => return Sample::error_sample(&args.format, &args.fixture_id, error),
    };
    if options != expected {
        return Sample::error_sample(&args.format, &args.fixture_id, NON_EMPTY_OPTIONS_REASON);
    }

    let source = match parse_source_envelope(&args.source_json) {
        Ok(source) => source,
        Err(error) => return Sample::error_sample(&args.format, &args.fixture_id, error),
    };

    let registry = match LoaderRegistry::with_builtin_formats() {
        Ok(registry) => registry,
        Err(error) => {
            return Sample::error_sample(&args.format, &args.fixture_id, error.to_string());
        }
    };

    let mut load_config =
        match build_load_config(&args.format, &source, args.path.clone(), &options) {
            Ok(config) => config,
            Err(error) => return Sample::error_sample(&args.format, &args.fixture_id, error),
        };
    load_config.options = options.clone();

    let mut compose_config = ComposeConfig::new(args.model.as_str(), RngRoot::new(Some(args.seed)));
    compose_config.format_options = options;
    if let Some((synthetic, output_length)) = synthetic_config_from_source(&source) {
        compose_config.synthetic_config = Some(synthetic);
        compose_config.output_length_distribution = output_length;
    }
    if needs_prepared_prompt_generator(&args.format) {
        if let Err(error) =
            inject_prepared_prompt_generator(&mut compose_config, tokenizer.as_ref())
        {
            return Sample::error_sample(&args.format, &args.fixture_id, error);
        }
    }

    match timed_load_compose(
        &registry,
        &args.format,
        &load_config,
        &compose_config,
        tokenizer.as_ref(),
        &args.tokenizer,
        args.apply_chat_template,
        args.exact_isl,
    )
    .await
    {
        Ok(sample_parts) => Sample {
            implementation: "rust",
            format: args.format.clone(),
            fixture_id: args.fixture_id.clone(),
            row_count: sample_parts.row_count,
            conversation_count: sample_parts.conversation_count,
            turn_count: sample_parts.turn_count,
            total_input_tokens: sample_parts.total_input_tokens,
            elapsed_ns: sample_parts.elapsed_ns,
            error: None,
        },
        Err(error) => Sample::error_sample(args.format.clone(), args.fixture_id.clone(), error),
    }
}

pub fn needs_prepared_prompt_generator(format: &str) -> bool {
    benchmark_format(format).is_some_and(|entry| entry.prepare_prompt_generator)
}

pub fn registry_format_name(format: &str) -> &str {
    benchmark_format(format)
        .map(|entry| entry.registry_name)
        .unwrap_or(format)
}

pub fn inject_prepared_prompt_generator(
    compose_config: &mut ComposeConfig,
    tokenizer: &dyn TextTokenizer,
) -> Result<(), String> {
    let prepared = CorpusPromptGeneratorFactory::default()
        .prepare(tokenizer)
        .map_err(|error| error.to_string())?;
    compose_config.prompt_generator = Arc::new(prepared);
    Ok(())
}

async fn load_tokenizer(spec: &str) -> Result<Arc<dyn TextTokenizer>, String> {
    if let Ok(encoding) = spec.parse::<TiktokenEncoding>() {
        return Ok(Arc::new(TiktokenTokenizer::new(encoding)));
    }

    let path = Path::new(spec);
    if path.is_dir() {
        return load_tokenizer_from_directory(path);
    }
    if path.is_file() {
        return load_tokenizer_from_file(path);
    }

    let downloaded = download_hugging_face_tokenizer(spec)
        .await
        .map_err(|error| error.to_string())?;
    load_tokenizer_from_directory(&downloaded)
}

fn load_tokenizer_from_directory(path: &Path) -> Result<Arc<dyn TextTokenizer>, String> {
    if path.join("tokenizer.json").is_file() {
        return HuggingFaceTokenizer::from_directory(path)
            .map(|tokenizer| Arc::new(tokenizer) as Arc<dyn TextTokenizer>)
            .map_err(|error| error.to_string());
    }
    if find_tiktoken_model_file(path).is_some() {
        return NativeTiktokenTokenizer::from_directory(path)
            .map(|tokenizer| Arc::new(tokenizer) as Arc<dyn TextTokenizer>)
            .map_err(|error| error.to_string());
    }
    Err(format!(
        "tokenizer source {} does not contain tokenizer.json or a tiktoken model",
        path.display()
    ))
}

fn load_tokenizer_from_file(path: &Path) -> Result<Arc<dyn TextTokenizer>, String> {
    if path
        .file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name == "tokenizer.json")
    {
        return HuggingFaceTokenizer::from_file(path)
            .map(|tokenizer| Arc::new(tokenizer) as Arc<dyn TextTokenizer>)
            .map_err(|error| error.to_string());
    }
    if path
        .extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| {
            extension.eq_ignore_ascii_case("model") || extension.eq_ignore_ascii_case("tiktoken")
        })
    {
        let directory = path
            .parent()
            .ok_or_else(|| format!("tokenizer path {} has no parent directory", path.display()))?;
        return NativeTiktokenTokenizer::from_model_file(path, directory)
            .map(|tokenizer| Arc::new(tokenizer) as Arc<dyn TextTokenizer>)
            .map_err(|error| error.to_string());
    }
    Err(format!(
        "unsupported tokenizer file {}; use tokenizer.json or a tiktoken model file",
        path.display()
    ))
}

struct MeasuredParts {
    row_count: usize,
    conversation_count: usize,
    turn_count: usize,
    total_input_tokens: Option<u64>,
    elapsed_ns: u128,
}

async fn timed_load_compose(
    registry: &LoaderRegistry,
    format: &str,
    load_config: &LoadConfig,
    compose_config: &ComposeConfig,
    tokenizer: &dyn TextTokenizer,
    tokenizer_name: &str,
    apply_chat_template: bool,
    exact_isl: bool,
) -> Result<MeasuredParts, String> {
    let registration = registry
        .get(registry_format_name(format))
        .map_err(|error| error.to_string())?;

    let started = Instant::now();
    let mut rows = registration
        .loader
        .load(load_config)
        .await
        .map_err(|error| error.to_string())?;
    if let Some(max_rows) = load_config.max_rows {
        rows.truncate(max_rows);
    }
    let loaded_row_count = rows.len();

    let mut pool = SegmentPool::new();
    let conversations = registration
        .composer
        .compose(rows, compose_config, tokenizer, &mut pool)
        .map_err(|error| error.to_string())?;

    let context_mode = registration
        .loader
        .default_context_mode()
        .unwrap_or(compose_config.default_context_mode);
    let dataset = Dataset::new(
        conversations,
        Arc::new(pool.freeze()),
        load_config
            .sampling_strategy
            .as_deref()
            .unwrap_or_else(|| registration.loader.preferred_sampling_strategy()),
        context_mode,
    )
    .map_err(|error| error.to_string())?;

    let conversation_count = dataset.conversations().len();
    let turn_count = dataset
        .conversations()
        .iter()
        .map(|conversation| conversation.turns.len())
        .sum();
    let row_count = benchmark_row_count(format, loaded_row_count, conversation_count);
    let total_input_tokens = benchmark_total_input_tokens(
        format,
        &dataset,
        tokenizer,
        tokenizer_name,
        apply_chat_template,
        exact_isl,
    )?;
    let elapsed_ns = started.elapsed().as_nanos();

    Ok(MeasuredParts {
        row_count,
        conversation_count,
        turn_count,
        total_input_tokens,
        elapsed_ns,
    })
}

fn uses_exact_isl_text_recount(format: &str, exact_isl: bool) -> bool {
    exact_isl && matches!(format, "synthetic" | "synthetic_rankings")
}

fn benchmark_row_count(format: &str, loaded_row_count: usize, conversation_count: usize) -> usize {
    if matches!(format, "synthetic" | "synthetic_rankings") {
        conversation_count
    } else {
        loaded_row_count
    }
}

fn rich_token_counting_requested(tokenizer_name: &str, apply_chat_template: bool) -> bool {
    apply_chat_template || !tokenizer_name.eq_ignore_ascii_case("builtin")
}

fn add_text_count(
    tokenizer: &dyn TextTokenizer,
    count: u64,
    texts: &[String],
) -> Result<u64, String> {
    if texts.is_empty() {
        return Ok(count);
    }
    let joined = texts.join(" ");
    let tokens = u64::try_from(
        tokenizer
            .count(&joined)
            .map_err(|error| error.to_string())?,
    )
    .map_err(|_| "input token count exceeds u64".to_string())?;
    count
        .checked_add(tokens)
        .ok_or_else(|| "input token count overflowed u64".to_string())
}

fn count_payload_input_tokens(
    tokenizer: &dyn TextTokenizer,
    payload: &Value,
    authored_input_tokens: Option<u64>,
    apply_chat_template: bool,
) -> Result<Option<u64>, String> {
    let extracted = extract_payload(payload);
    if apply_chat_template
        && let Some(messages) = extracted
            .messages
            .as_deref()
            .filter(|items| !items.is_empty())
        && let Some(tokens) = tokenizer.apply_chat_template(messages, true).ok().flatten()
    {
        let templated = u64::try_from(tokens.len())
            .map_err(|_| "templated input token count exceeds u64".to_string())?;
        let count = extracted
            .pretokenised_token_count
            .checked_add(templated)
            .ok_or_else(|| "input token count overflowed u64".to_string())?;
        return add_text_count(tokenizer, count, &extracted.tool_texts).map(Some);
    }
    if !extracted.texts.is_empty() {
        return add_text_count(
            tokenizer,
            extracted.pretokenised_token_count,
            &extracted.texts,
        )
        .map(Some);
    }
    if extracted.pretokenised_token_count > 0 {
        return Ok(Some(extracted.pretokenised_token_count));
    }
    Ok(authored_input_tokens)
}

fn raw_json_value(dataset: &Dataset, handle: u32, field_name: &str) -> Result<Value, String> {
    let payload = dataset
        .segments()
        .get(aiperf_runtime::dataset::Handle::new(handle))
        .map_err(|error| error.to_string())?;
    let Payload::Raw { wire } = payload else {
        return Err(format!(
            "{field_name} handle {handle} used unexpected {} payload",
            payload.kind_name()
        ));
    };
    serde_json::from_slice(&wire)
        .map_err(|error| format!("invalid {field_name} JSON at handle {handle}: {error}"))
}

fn raw_json_handle_value(
    dataset: &Dataset,
    handle: aiperf_runtime::dataset::Handle,
    field_name: &str,
) -> Result<Value, String> {
    raw_json_value(dataset, handle.index(), field_name)
}

fn resolve_text_handle(
    dataset: &Dataset,
    handle: Option<aiperf_runtime::dataset::Handle>,
    field_name: &str,
) -> Result<Option<String>, String> {
    let Some(handle) = handle else {
        return Ok(None);
    };
    let payload = dataset
        .segments()
        .get(handle)
        .map_err(|error| error.to_string())?;
    let Payload::Text { bytes, .. } = payload else {
        return Err(format!(
            "{field_name} handle {} used unexpected {} payload",
            handle.index(),
            payload.kind_name()
        ));
    };
    std::str::from_utf8(&bytes)
        .map(str::to_owned)
        .map(Some)
        .map_err(|error| format!("invalid UTF-8 in {field_name}: {error}"))
}

fn turn_text_content(dataset: &Dataset, turn: &Turn) -> Result<String, String> {
    let mut text = String::new();
    for group in &turn.content {
        if group.kind != MediaKind::Text {
            continue;
        }
        for handle in &group.handles {
            let payload = dataset
                .segments()
                .get(*handle)
                .map_err(|error| error.to_string())?;
            let Payload::Text { bytes, .. } = payload else {
                return Err(format!(
                    "text content handle {} used unexpected {} payload",
                    handle.index(),
                    payload.kind_name()
                ));
            };
            text.push_str(std::str::from_utf8(&bytes).map_err(|error| error.to_string())?);
        }
    }
    Ok(text)
}

fn inject_chat_context(
    dataset: &Dataset,
    conversation: &Conversation,
    messages: Vec<Value>,
) -> Result<Vec<Value>, String> {
    let first_is_system = messages
        .first()
        .and_then(Value::as_object)
        .and_then(|message| message.get("role").and_then(Value::as_str))
        == Some("system");
    let mut prefixed = Vec::new();
    if let Some(system_message) = resolve_text_handle(dataset, conversation.system, "system")?
        && !first_is_system
    {
        prefixed.push(json!({"role": "system", "content": system_message}));
    }
    if let Some(user_context) =
        resolve_text_handle(dataset, conversation.user_context, "user_context")?
    {
        prefixed.push(json!({"role": "user", "content": user_context}));
    }
    prefixed.extend(messages);
    Ok(prefixed)
}

fn turn_payload_for_counting(
    dataset: &Dataset,
    conversation: &Conversation,
    turn: &Turn,
) -> Result<Option<Value>, String> {
    if let Some(raw_handle) = turn.body.first().copied() {
        let payload = dataset
            .segments()
            .get(raw_handle)
            .map_err(|error| error.to_string())?;
        if matches!(payload, Payload::Raw { .. }) {
            return raw_json_handle_value(dataset, raw_handle, "raw_payload").map(Some);
        }
    }

    let rendered_messages = if let Some(handle) = turn.raw_messages {
        let value = raw_json_handle_value(dataset, handle, "raw_messages")?;
        let Value::Array(messages) = value else {
            return Err(format!(
                "raw_messages handle {} must contain a JSON array",
                handle.index()
            ));
        };
        messages
    } else {
        vec![json!({
            "role": turn.role.as_ref().map(|role| role.as_str()).unwrap_or("user"),
            "content": turn_text_content(dataset, turn)?,
        })]
    };
    let messages = inject_chat_context(dataset, conversation, rendered_messages)?;
    let mut payload = Map::new();
    payload.insert("messages".to_string(), Value::Array(messages));
    if let Some(handle) = turn.tools {
        payload.insert(
            "tools".to_string(),
            raw_json_handle_value(dataset, handle, "tools")?,
        );
    }
    Ok(Some(Value::Object(payload)))
}

pub fn benchmark_total_input_tokens(
    format: &str,
    dataset: &Dataset,
    tokenizer: &dyn TextTokenizer,
    tokenizer_name: &str,
    apply_chat_template: bool,
    exact_isl: bool,
) -> Result<Option<u64>, String> {
    let rich_token_counts = rich_token_counting_requested(tokenizer_name, apply_chat_template);
    let opaque = benchmark_format(format).is_some_and(|entry| entry.opaque_token_counts);
    if opaque && !rich_token_counts {
        return Ok(None);
    }
    if !rich_token_counts && uses_exact_isl_text_recount(format, exact_isl) {
        let mut total = 0_u64;
        for turn in dataset
            .conversations()
            .iter()
            .flat_map(|conversation| conversation.turns.iter())
        {
            for group in &turn.content {
                if group.kind != MediaKind::Text {
                    continue;
                }
                for handle in &group.handles {
                    let payload = dataset
                        .segments()
                        .get(*handle)
                        .map_err(|error| error.to_string())?;
                    let Payload::Text { bytes, .. } = payload else {
                        return Err(format!(
                            "synthetic text group used unexpected {} payload",
                            payload.kind_name()
                        ));
                    };
                    let text = std::str::from_utf8(&bytes).map_err(|error| error.to_string())?;
                    let count = tokenizer.count(text).map_err(|error| error.to_string())?;
                    total = total.saturating_add(count as u64);
                }
            }
        }
        return Ok(Some(total));
    }

    let mut known = false;
    let mut total = 0_u64;
    for conversation in dataset.conversations() {
        for turn in &conversation.turns {
            if rich_token_counts {
                if let Some(payload) = turn_payload_for_counting(dataset, conversation, turn)? {
                    if let Some(tokens) = count_payload_input_tokens(
                        tokenizer,
                        &payload,
                        turn.input_tokens,
                        apply_chat_template,
                    )? {
                        known = true;
                        total = total
                            .checked_add(tokens)
                            .ok_or_else(|| "input token count overflowed u64".to_string())?;
                    }
                    continue;
                }
            }
            if let Some(tokens) = turn.input_tokens {
                known = true;
                total = total
                    .checked_add(tokens)
                    .ok_or_else(|| "input token count overflowed u64".to_string())?;
            }
        }
    }
    Ok(known.then_some(total))
}

fn parse_options_map(options_json: &str) -> Result<Map<String, Value>, String> {
    let value: Value = serde_json::from_str(options_json)
        .map_err(|error| format!("invalid --options-json: {error}"))?;
    match value {
        Value::Object(map) => Ok(map),
        other => Err(format!(
            "invalid --options-json: expected object, got {}",
            json_type_label(&other)
        )),
    }
}

fn parse_source_envelope(source_json: &str) -> Result<Map<String, Value>, String> {
    let value: Value = serde_json::from_str(source_json)
        .map_err(|error| format!("invalid --source-json: {error}"))?;
    match value {
        Value::Object(map) => Ok(map),
        other => Err(format!(
            "invalid --source-json: expected object, got {}",
            json_type_label(&other)
        )),
    }
}

fn json_type_label(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "bool",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

fn source_kind(source: &Map<String, Value>) -> Result<&str, String> {
    source
        .get("kind")
        .and_then(Value::as_str)
        .ok_or_else(|| "source.kind is required".into())
}

fn build_load_config(
    format: &str,
    source: &Map<String, Value>,
    path: Option<PathBuf>,
    options: &Map<String, Value>,
) -> Result<LoadConfig, String> {
    match source_kind(source)? {
        "local_file" => {
            let configured = source
                .get("path")
                .and_then(Value::as_str)
                .map(PathBuf::from)
                .or(path)
                .ok_or_else(|| format!("{format} requires a local file path"))?;
            Ok(LoadConfig::new(DatasetSource::Path(configured)))
        }
        "inline_synthetic" => {
            let inline = source
                .get("inline")
                .and_then(Value::as_object)
                .ok_or_else(|| "inline_synthetic source requires inline object")?;
            let marker = inline
                .get("marker")
                .and_then(Value::as_str)
                .unwrap_or("__aiperf_synthetic");
            Ok(LoadConfig::new(DatasetSource::Inline(
                json!({ marker: true }),
            )))
        }
        "public_cached" => build_public_load_config(source, options),
        other => Err(format!("unsupported source kind {other:?}")),
    }
}

fn build_public_load_config(
    source: &Map<String, Value>,
    options: &Map<String, Value>,
) -> Result<LoadConfig, String> {
    let rust_public = source
        .get("rust_public")
        .and_then(Value::as_object)
        .ok_or_else(|| "public_cached source requires rust_public object")?;
    let public_source = rust_public
        .get("source")
        .and_then(Value::as_object)
        .ok_or_else(|| "rust_public.source is required")?;
    let row_limit = rust_public
        .get("identity")
        .and_then(Value::as_object)
        .and_then(|identity| identity.get("row_limit"))
        .and_then(Value::as_u64)
        .map(|value| value as usize)
        .or_else(|| {
            options
                .get("max_conversations")
                .and_then(Value::as_u64)
                .map(|value| value as usize)
        })
        .unwrap_or(3);
    let mut merged = Map::new();
    if let Some(base) = rust_public.get("options").and_then(Value::as_object) {
        merged.extend(base.clone());
    }
    merged.extend(options.clone());

    match public_source.get("type").and_then(Value::as_str) {
        Some("url") => {
            let url = public_source
                .get("url")
                .and_then(Value::as_str)
                .ok_or_else(|| "url source requires url".to_string())?;
            let mut config = LoadConfig::new(DatasetSource::Url(url.to_string()));
            config.max_rows = Some(row_limit);
            config.options = merged;
            Ok(config)
        }
        Some("hugging_face") => {
            let dataset = public_source
                .get("dataset")
                .and_then(Value::as_str)
                .ok_or_else(|| "hugging_face source requires dataset".to_string())?;
            let subset = public_source
                .get("subset")
                .and_then(Value::as_str)
                .unwrap_or("default")
                .to_string();
            let split = public_source
                .get("split")
                .and_then(Value::as_str)
                .unwrap_or("train")
                .to_string();
            let revision = public_source
                .get("revision")
                .and_then(Value::as_str)
                .map(str::to_string);
            let mut config = LoadConfig::new(DatasetSource::HuggingFace {
                dataset: dataset.to_string(),
                config: subset,
                split,
                max_rows: Some(row_limit),
                revision,
            });
            config.options = merged;
            Ok(config)
        }
        other => Err(format!("unsupported public source type {other:?}")),
    }
}

fn synthetic_config_from_source(
    source: &Map<String, Value>,
) -> Option<(SyntheticDatasetConfig, Option<SamplingDistribution>)> {
    if source_kind(source).ok()? != "inline_synthetic" {
        return None;
    }
    let inline = source.get("inline")?.as_object()?;
    let shape = inline.get("synthetic_config")?.as_object()?;
    let entries = shape.get("entries").and_then(Value::as_u64).unwrap_or(3) as usize;
    let turns = fixed_distribution(shape.get("turns").and_then(Value::as_f64).unwrap_or(1.0))?;
    let mut config = SyntheticDatasetConfig {
        entries,
        turns,
        ..SyntheticDatasetConfig::default()
    };
    let mut output_length = None;
    if let Some(prompts) = shape.get("prompts").and_then(Value::as_object) {
        config.prompts = Some(SyntheticPromptConfig {
            input_tokens: fixed_distribution(
                prompts
                    .get("input_tokens")
                    .and_then(Value::as_f64)
                    .unwrap_or(12.0),
            )?,
            batch_size: prompts
                .get("batch_size")
                .and_then(Value::as_u64)
                .unwrap_or(1) as usize,
            prefix_reuse_fraction: 0.0,
            prefix_reuse_ratio: 0.5,
        });
        output_length = prompts
            .get("output_tokens")
            .and_then(Value::as_f64)
            .and_then(fixed_distribution);
    }
    if let Some(rankings) = shape.get("rankings").and_then(Value::as_object) {
        config.rankings = Some(SyntheticRankingsConfig {
            passages: fixed_distribution(
                rankings
                    .get("passages")
                    .and_then(Value::as_f64)
                    .unwrap_or(2.0),
            )?,
            passage_tokens: fixed_distribution(
                rankings
                    .get("passage_tokens")
                    .and_then(Value::as_f64)
                    .unwrap_or(8.0),
            )?,
            query_tokens: fixed_distribution(
                rankings
                    .get("query_tokens")
                    .and_then(Value::as_f64)
                    .unwrap_or(4.0),
            )?,
        });
    }
    Some((config, output_length))
}

fn fixed_distribution(value: f64) -> Option<SamplingDistribution> {
    SamplingDistribution::fixed(value).ok()
}

#[cfg(not(test))]
fn emit(sample: &Sample) {
    println!("{}", sample.to_json_line());
}

#[cfg(not(test))]
#[tokio::main(flavor = "current_thread")]
async fn main() {
    let sample = match parse_args(std::env::args()) {
        Ok(args) => measure(&args).await,
        Err(sample) => sample,
    };
    emit(&sample);
    if sample.error.is_some() {
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;

    struct CountingTokenizer {
        inner: TiktokenTokenizer,
        counted_tokens: usize,
        count_calls: AtomicUsize,
    }

    impl CountingTokenizer {
        fn new(counted_tokens: usize) -> Self {
            Self {
                inner: TiktokenTokenizer::builtin(),
                counted_tokens,
                count_calls: AtomicUsize::new(0),
            }
        }

        fn count_calls(&self) -> usize {
            self.count_calls.load(Ordering::Relaxed)
        }
    }

    impl TextTokenizer for CountingTokenizer {
        fn encode(&self, text: &str) -> aiperf_runtime::dataset::error::Result<Vec<u32>> {
            self.inner.encode(text)
        }

        fn decode(&self, token_ids: &[u32]) -> aiperf_runtime::dataset::error::Result<String> {
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

        fn count(&self, _text: &str) -> aiperf_runtime::dataset::error::Result<usize> {
            self.count_calls.fetch_add(1, Ordering::Relaxed);
            Ok(self.counted_tokens)
        }
    }

    #[test]
    fn verified_option_catalog_matches_speed_bench() {
        let options = verified_options("speed_bench").expect("speed_bench options");
        assert_eq!(
            options.get("category").and_then(Value::as_str),
            Some("coding")
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn synthetic_benchmark_total_input_tokens_uses_authoritative_segments_without_recount() {
        let registry = LoaderRegistry::with_builtin_formats().expect("registry");
        let source = json!({
            "kind": "inline_synthetic",
            "inline": {
                "marker": "__aiperf_synthetic",
                "synthetic_config": {
                    "entries": 2,
                    "turns": 1.0,
                    "prompts": {
                        "input_tokens": 12.0,
                        "output_tokens": 8.0,
                        "batch_size": 1
                    }
                }
            }
        });
        let load_config = build_load_config(
            "synthetic",
            source.as_object().expect("object"),
            None,
            &Map::new(),
        )
        .expect("load config");
        let mut compose_config = ComposeConfig::new("test-model", RngRoot::new(Some(42)));
        let (synthetic, output_length) =
            synthetic_config_from_source(source.as_object().expect("object")).expect("synthetic");
        compose_config.synthetic_config = Some(synthetic);
        compose_config.output_length_distribution = output_length;

        let registration = registry.get("synthetic").expect("synthetic format");
        let rows = registration
            .loader
            .load(&load_config)
            .await
            .expect("load rows");
        let mut pool = SegmentPool::new();
        let conversations = registration
            .composer
            .compose(
                rows,
                &compose_config,
                &TiktokenTokenizer::builtin(),
                &mut pool,
            )
            .expect("compose");
        let dataset = Dataset::new(
            conversations,
            Arc::new(pool.freeze()),
            load_config
                .sampling_strategy
                .as_deref()
                .unwrap_or_else(|| registration.loader.preferred_sampling_strategy()),
            registration
                .loader
                .default_context_mode()
                .unwrap_or(compose_config.default_context_mode),
        )
        .expect("dataset");

        let expected = dataset
            .conversations()
            .iter()
            .flat_map(|conversation| conversation.turns.iter())
            .map(|turn| turn.input_tokens.expect("synthetic turn input tokens"))
            .sum::<u64>();
        let tokenizer = CountingTokenizer::new(7);
        let total = benchmark_total_input_tokens(
            "synthetic",
            &dataset,
            &tokenizer,
            "builtin",
            false,
            false,
        )
        .expect("count synthetic tokens");
        assert_eq!(total, Some(expected));
        assert!(
            tokenizer.count_calls() == 0,
            "synthetic benchmark should use stored token counts"
        );
    }
}
