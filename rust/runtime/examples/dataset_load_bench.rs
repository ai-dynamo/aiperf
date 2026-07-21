// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Timed Rust adapter for the Python/Rust dataset-load comparison harness.
//!
//! Initializes the built-in Tiktoken tokenizer before measurement, then times
//! only load → compose → freeze for one local fixture. Emits a single one-line
//! JSON [`Sample`] record on stdout. Argument parsing is deliberately minimal
//! and dependency-free.
//!
//! Run:
//! ```bash
//! cargo run -p aiperf-runtime --release --example dataset_load_bench -- \
//!   --format single_turn --path /tmp/fixture.jsonl --options-json '{}' \
//!   --fixture-id single_turn-seed42 --seed 42 --model test-model
//! ```

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use aiperf_runtime::dataset::{
    ComposeConfig, CorpusPromptGeneratorFactory, Dataset, DatasetSource, LoadConfig,
    LoaderRegistry, SegmentPool, TextTokenizer, TiktokenTokenizer,
};
use aiperf_runtime::rng::RngRoot;
use serde::Serialize;
use serde_json::{Map, Value};

/// Shared reject reason for non-empty loader options until mapping is verified.
pub const NON_EMPTY_OPTIONS_REASON: &str =
    "non-empty options are unsupported until cross-stack option mapping is verified";

/// One verified harness format and how the Rust adapter should treat it.
///
/// This table is the single source of truth for prepare dispatch and for the
/// Python-canonical → Rust-registry name alias (`burst_gpt_trace` → `burst_gpt`).
/// Keep it aligned with `SUPPORTED_FORMATS` in `dataset_load_compare.py`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BenchmarkFormat {
    /// Python-canonical format name used on the CLI and in Sample records.
    pub name: &'static str,
    /// Name passed to [`LoaderRegistry::get`] (may differ from [`Self::name`]).
    pub registry_name: &'static str,
    /// Whether composers always call `prompt_generator.create` on compose.
    pub prepare_prompt_generator: bool,
}

/// Authoritative catalog of formats this adapter may measure.
pub const BENCHMARK_FORMATS: &[BenchmarkFormat] = &[
    BenchmarkFormat {
        name: "single_turn",
        registry_name: "single_turn",
        prepare_prompt_generator: false,
    },
    BenchmarkFormat {
        name: "multi_turn",
        registry_name: "multi_turn",
        prepare_prompt_generator: false,
    },
    BenchmarkFormat {
        name: "raw_payload",
        registry_name: "raw_payload",
        prepare_prompt_generator: false,
    },
    BenchmarkFormat {
        name: "inputs_json",
        registry_name: "inputs_json",
        prepare_prompt_generator: false,
    },
    BenchmarkFormat {
        name: "random_pool",
        registry_name: "random_pool",
        prepare_prompt_generator: false,
    },
    BenchmarkFormat {
        name: "mooncake_trace",
        registry_name: "mooncake_trace",
        prepare_prompt_generator: true,
    },
    BenchmarkFormat {
        name: "bailian_trace",
        registry_name: "bailian_trace",
        prepare_prompt_generator: true,
    },
    BenchmarkFormat {
        name: "burst_gpt_trace",
        registry_name: "burst_gpt",
        prepare_prompt_generator: true,
    },
    BenchmarkFormat {
        name: "sagemaker_data_capture",
        registry_name: "sagemaker_data_capture",
        prepare_prompt_generator: false,
    },
];

/// Look up a verified harness format by its Python-canonical name.
pub fn benchmark_format(name: &str) -> Option<&'static BenchmarkFormat> {
    BENCHMARK_FORMATS.iter().find(|format| format.name == name)
}

/// One-line JSON sample record consumed by the comparison orchestrator.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct Sample {
    /// Adapter identity (`"rust"`).
    pub implementation: &'static str,
    /// Dataset format name passed on the CLI.
    pub format: String,
    /// Fixture identity echoed from `--fixture-id`.
    pub fixture_id: String,
    /// Parsed row count before composition.
    pub row_count: usize,
    /// Frozen conversation count.
    pub conversation_count: usize,
    /// Total turns across all conversations.
    pub turn_count: usize,
    /// Sum of known per-turn `input_tokens` after composition.
    ///
    /// `None` when every composed turn left the count unset (opaque raw
    /// payloads / inputs.json).
    pub total_input_tokens: Option<u64>,
    /// Measured load → compose → freeze duration in nanoseconds.
    pub elapsed_ns: u128,
    /// Structured failure message when the format cannot be measured.
    pub error: Option<String>,
}

impl Sample {
    /// Build a zeroed error sample for CLI or measurement failures.
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

    /// Serialize as a single JSON line.
    pub fn to_json_line(&self) -> String {
        serde_json::to_string(self).expect("Sample serialization is infallible")
    }
}

/// Parsed CLI arguments for one timed measurement.
#[derive(Debug, Clone)]
pub struct Args {
    /// Explicit dataset format name.
    pub format: String,
    /// Local fixture path.
    pub path: PathBuf,
    /// Format-specific options JSON object.
    pub options_json: String,
    /// Fixture identity echoed into the sample.
    pub fixture_id: String,
    /// RNG seed for composition.
    pub seed: u64,
    /// Model name for composition.
    pub model: String,
}

/// Parse adapter CLI arguments without an external argparse dependency.
///
/// On failure returns a structured [`Sample`] with `error` populated so the
/// orchestrator can record the failure without guessing exit semantics.
pub fn parse_args(argv: impl IntoIterator<Item = String>) -> Result<Args, Sample> {
    let mut argv = argv.into_iter();
    let _exe = argv.next();

    let mut format = None;
    let mut path = None;
    let mut options_json = None;
    let mut fixture_id = None;
    let mut seed = None;
    let mut model = None;

    while let Some(flag) = argv.next() {
        match flag.as_str() {
            "--format" | "--path" | "--options-json" | "--fixture-id" | "--seed" | "--model" => {}
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
            "--path" => path = Some(PathBuf::from(value)),
            "--options-json" => options_json = Some(value),
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
    let path = path.ok_or_else(|| {
        Sample::error_sample(
            format.clone(),
            fixture_id.clone().unwrap_or_default(),
            "missing required argument --path",
        )
    })?;
    let options_json = options_json.ok_or_else(|| {
        Sample::error_sample(
            format.clone(),
            fixture_id.clone().unwrap_or_default(),
            "missing required argument --options-json",
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
        fixture_id,
        seed,
        model,
    })
}

/// Time load → compose → freeze for one fixture and return a [`Sample`].
///
/// Tokenizer construction, a warm encode, and (for authored-length trace
/// formats) one-shot corpus prompt-generator preparation happen before the
/// timer starts. Failures return a zeroed sample with `error` populated rather
/// than panicking.
pub async fn measure(args: &Args) -> Sample {
    let tokenizer = TiktokenTokenizer::builtin();
    // Warm the encoding tables outside the timed region.
    if let Err(error) = tokenizer.encode("warm") {
        return Sample::error_sample(&args.format, &args.fixture_id, error.to_string());
    }

    let options = match parse_options_map(&args.options_json) {
        Ok(options) => options,
        Err(error) => return Sample::error_sample(&args.format, &args.fixture_id, error),
    };
    if !options.is_empty() {
        return Sample::error_sample(&args.format, &args.fixture_id, NON_EMPTY_OPTIONS_REASON);
    }

    let registry = match LoaderRegistry::with_builtin_formats() {
        Ok(registry) => registry,
        Err(error) => {
            return Sample::error_sample(&args.format, &args.fixture_id, error.to_string());
        }
    };

    let mut load_config = LoadConfig::new(DatasetSource::Path(args.path.clone()));
    load_config.options = options.clone();

    let mut compose_config = ComposeConfig::new(args.model.as_str(), RngRoot::new(Some(args.seed)));
    compose_config.format_options = options;
    // Trace composers call `prompt_generator.create` inside compose. Preparing
    // the Shakespeare corpus once here mirrors Python constructing
    // `PromptGenerator` before its timer, so timed samples exclude ~350ms of
    // corpus tokenization while keeping generated-content parity identical.
    if needs_prepared_prompt_generator(&args.format) {
        if let Err(error) = inject_prepared_prompt_generator(&mut compose_config, &tokenizer) {
            return Sample::error_sample(&args.format, &args.fixture_id, error);
        }
    }

    match timed_load_compose(
        &registry,
        &args.format,
        &load_config,
        &compose_config,
        &tokenizer,
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

/// Whether this format's composer always creates a corpus prompt generator.
///
/// Driven solely by [`BENCHMARK_FORMATS`]; unknown names do not prepare.
pub fn needs_prepared_prompt_generator(format: &str) -> bool {
    benchmark_format(format).is_some_and(|entry| entry.prepare_prompt_generator)
}

/// Resolve the Rust [`LoaderRegistry`] name for a harness format spelling.
///
/// Unknown names pass through unchanged so structured errors still report the
/// caller-supplied identifier.
pub fn registry_format_name(format: &str) -> &str {
    benchmark_format(format)
        .map(|entry| entry.registry_name)
        .unwrap_or(format)
}

/// Tokenize the default corpus once and install the prepared factory.
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
    tokenizer: &TiktokenTokenizer,
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
    let row_count = rows.len();

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
    let elapsed_ns = started.elapsed().as_nanos();

    let conversation_count = dataset.conversations().len();
    let turn_count = dataset
        .conversations()
        .iter()
        .map(|conversation| conversation.turns.len())
        .sum();
    let total_input_tokens = {
        let mut known = false;
        let mut total = 0_u64;
        for turn in dataset
            .conversations()
            .iter()
            .flat_map(|conversation| conversation.turns.iter())
        {
            if let Some(tokens) = turn.input_tokens {
                known = true;
                total = total.saturating_add(tokens);
            }
        }
        known.then_some(total)
    };

    Ok(MeasuredParts {
        row_count,
        conversation_count,
        turn_count,
        total_input_tokens,
        elapsed_ns,
    })
}

fn parse_options_map(options_json: &str) -> Result<Map<String, Value>, String> {
    let value: Value = serde_json::from_str(options_json)
        .map_err(|error| format!("invalid --options-json: {error}"))?;
    match value {
        Value::Object(map) => Ok(map),
        other => Err(format!(
            "invalid --options-json: expected object, got {}",
            match other {
                Value::Null => "null",
                Value::Bool(_) => "bool",
                Value::Number(_) => "number",
                Value::String(_) => "string",
                Value::Array(_) => "array",
                Value::Object(_) => unreachable!(),
            }
        )),
    }
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
