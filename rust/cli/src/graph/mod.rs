// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Native local-only graph inspection command boundary.

mod explain;
pub mod report;
mod validate;
mod visualize;

use std::path::{Path, PathBuf};

use aiperf_runtime::config::model::workload_kind::GRAPH_FORMATS;
use aiperf_runtime::engine::graph_input::{
    BuiltinRunnerGraphInputAdapterResolver, PreparedRunnerGraphInput,
    prepare_local_graph_inspection_input,
};
use aiperf_runtime::engine::preparation::{LocalTokenizerError, load_local_tokenizer};
use clap::builder::PossibleValuesParser;
use clap::{Args, Parser, Subcommand, ValueEnum};
use tokio::runtime::Builder;
use tokio::task::LocalSet;

use self::report::{GraphCommandError, GraphCommandErrorCode, GraphOperation};

const BUILTIN_FORMAT_HELP: &str = "Supported built-in graph formats: dag_jsonl, conditional_graph, weka_trace, dynamo_trace, aiperf_trace, agent_recording, otlp_genai";

#[derive(Debug, Parser)]
#[command(name = "graph", disable_help_subcommand = true, subcommand_required = true, arg_required_else_help = true, after_help = BUILTIN_FORMAT_HELP)]
struct GraphCli {
    #[command(subcommand)]
    command: GraphCommand,
}

#[derive(Debug, Subcommand)]
enum GraphCommand {
    /// Validate a graph input after lowering it once.
    Validate(ValidateArgs),
    /// Explain a graph input after lowering it once.
    Explain(ExplainArgs),
    /// Visualize a graph input after lowering it once.
    Visualize(VisualizeArgs),
}

#[derive(Debug, Args)]
struct CommonArgs {
    /// Existing local file or directory containing the graph input.
    path: PathBuf,
    /// Supported built-in graph format.
    #[arg(long, required = true, value_parser = graph_format_parser(), help = "Supported built-in graph formats")]
    graph_format: String,
    /// Built-in tokenizer name or existing local tokenizer path.
    #[arg(long, default_value = "builtin")]
    tokenizer: String,
    /// Deterministic graph lowering seed.
    #[arg(long, default_value_t = 0)]
    seed: u64,
}

#[derive(Debug, Args)]
struct ValidateArgs {
    #[command(flatten)]
    common: CommonArgs,
    /// Require authored profiling-plan arrival offsets.
    #[arg(long, value_enum)]
    pace: Option<Pace>,
    /// Select human text or a JSON report.
    #[arg(long, value_enum, default_value_t = TextJsonFormat::Text)]
    output_format: TextJsonFormat,
}

#[derive(Debug, Args)]
struct ExplainArgs {
    #[command(flatten)]
    common: CommonArgs,
    /// Select human text or a JSON report.
    #[arg(long, value_enum, default_value_t = TextJsonFormat::Text)]
    output_format: TextJsonFormat,
}

#[derive(Debug, Args)]
struct VisualizeArgs {
    #[command(flatten)]
    common: CommonArgs,
    /// Select a retained trace by identifier.
    #[arg(long)]
    trace: Option<String>,
    /// Write rendered output to this path.
    #[arg(long)]
    output: Option<PathBuf>,
    /// Select Markdown or raw Mermaid output.
    #[arg(long, value_enum, default_value_t = VisualizeFormat::Markdown)]
    output_format: VisualizeFormat,
    /// Render best-effort topology even when validation reports errors.
    #[arg(long)]
    no_validate: bool,
}

#[derive(Clone, Copy, Debug, Default, ValueEnum)]
enum TextJsonFormat {
    /// Human-readable text.
    #[default]
    Text,
    /// Versioned JSON report.
    Json,
}

#[derive(Clone, Copy, Debug, Default, ValueEnum)]
enum VisualizeFormat {
    /// Complete Markdown document.
    #[default]
    Markdown,
    /// Standalone Mermaid source.
    Mermaid,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Pace {
    /// Require authored arrival offsets.
    Arrival,
}

/// One local source retained after exactly one adapter load.
pub(crate) struct LoadedGraphInput {
    /// Canonical source passed to the adapter.
    pub(crate) source: PathBuf,
    /// Retained graph input for later inspection renderers.
    pub(crate) prepared: PreparedRunnerGraphInput,
}

/// Run `aiperf graph <args>` without Python delegation.
pub fn run(argv: &[String]) -> anyhow::Result<i32> {
    let full: Vec<String> = std::iter::once("graph".to_owned())
        .chain(argv.iter().cloned())
        .collect();
    let cli = match GraphCli::try_parse_from(&full) {
        Ok(cli) => cli,
        Err(error) => {
            if requested_json(argv) {
                let failure = GraphCommandError::new(
                    GraphCommandErrorCode::InvalidArguments,
                    format!("invalid arguments: {}", error.kind()),
                    None,
                );
                write_error(operation_from_argv(argv), &failure, true);
            } else {
                error.print().ok();
            }
            return Ok(error.exit_code());
        }
    };
    match cli.command {
        GraphCommand::Validate(args) => run_validate(args),
        GraphCommand::Explain(args) => run_explain(args),
        GraphCommand::Visualize(args) => run_visualize(args),
    }
}

fn run_validate(args: ValidateArgs) -> anyhow::Result<i32> {
    match load(args.common) {
        Ok(input) => {
            let _ = args.pace;
            validate::run(input);
            Ok(0)
        }
        Err(error) => {
            write_error(
                GraphOperation::Validate,
                &error,
                matches!(args.output_format, TextJsonFormat::Json),
            );
            Ok(2)
        }
    }
}

fn run_explain(args: ExplainArgs) -> anyhow::Result<i32> {
    match load(args.common) {
        Ok(input) => {
            explain::run(input);
            Ok(0)
        }
        Err(error) => {
            write_error(
                GraphOperation::Explain,
                &error,
                matches!(args.output_format, TextJsonFormat::Json),
            );
            Ok(2)
        }
    }
}

fn run_visualize(args: VisualizeArgs) -> anyhow::Result<i32> {
    match load(args.common) {
        Ok(input) => {
            let _ = (
                args.trace,
                args.output,
                args.output_format,
                args.no_validate,
            );
            visualize::run(input);
            Ok(0)
        }
        Err(error) => {
            write_error(GraphOperation::Visualize, &error, false);
            Ok(2)
        }
    }
}

fn load(args: CommonArgs) -> Result<LoadedGraphInput, GraphCommandError> {
    let source = validate_local_source(&args.path)?;
    let source_text = source.display().to_string();
    let tokenizer = load_local_tokenizer(Some(&args.tokenizer)).map_err(|error| match error {
        LocalTokenizerError::Unsupported { .. } => GraphCommandError::new(
            GraphCommandErrorCode::TokenizerUnsupported,
            "tokenizer must be a built-in encoding or an existing local path",
            Some(source_text.clone()),
        ),
        LocalTokenizerError::Load { .. } => GraphCommandError::new(
            GraphCommandErrorCode::TokenizerLoadFailed,
            "local tokenizer could not be loaded",
            Some(source_text.clone()),
        ),
    })?;
    let resolver = BuiltinRunnerGraphInputAdapterResolver::new();
    let runtime = Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|_| {
            GraphCommandError::new(
                GraphCommandErrorCode::InputLoweringFailed,
                "could not initialize local graph inspection runtime",
                Some(source_text.clone()),
            )
        })?;
    let local = LocalSet::new();
    let prepared = local.block_on(
        &runtime,
        prepare_local_graph_inspection_input(
            &resolver,
            &source,
            &args.graph_format,
            tokenizer.as_ref(),
            args.seed,
        ),
    );
    let prepared = prepared.map_err(|error| {
        let chain = error
            .chain()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(" ");
        let code = if chain.contains("decoding direct") || chain.contains("serde") {
            GraphCommandErrorCode::InputDecodeFailed
        } else {
            GraphCommandErrorCode::InputLoweringFailed
        };
        let message = if matches!(code, GraphCommandErrorCode::InputDecodeFailed) {
            "graph input could not be decoded"
        } else {
            "graph input could not be lowered"
        };
        GraphCommandError::new(code, message, Some(source_text))
    })?;
    Ok(LoadedGraphInput { source, prepared })
}

fn validate_local_source(path: &Path) -> Result<PathBuf, GraphCommandError> {
    let raw = path.to_string_lossy();
    let normalized = raw.to_ascii_lowercase();
    if raw == "-"
        || ["http://", "https://", "hf://", "huggingface://"]
            .iter()
            .any(|scheme| normalized.starts_with(scheme))
    {
        return Err(GraphCommandError::new(
            GraphCommandErrorCode::SourceNotLocal,
            "graph source must be an existing local file or directory",
            None,
        ));
    }
    match std::fs::metadata(path) {
        Ok(metadata) if metadata.is_file() || metadata.is_dir() => {}
        Ok(_) => {
            return Err(GraphCommandError::new(
                GraphCommandErrorCode::SourceNotLocal,
                "graph source must be a local file or directory",
                Some(path.display().to_string()),
            ));
        }
        Err(_) => {
            return Err(GraphCommandError::new(
                GraphCommandErrorCode::SourceNotFound,
                "graph source does not exist locally",
                Some(path.display().to_string()),
            ));
        }
    }
    path.canonicalize().map_err(|_| {
        GraphCommandError::new(
            GraphCommandErrorCode::SourceNotFound,
            "graph source could not be canonicalized",
            Some(path.display().to_string()),
        )
    })
}

fn graph_format_parser() -> PossibleValuesParser {
    PossibleValuesParser::new(GRAPH_FORMATS)
}

fn requested_json(argv: &[String]) -> bool {
    argv.windows(2)
        .any(|window| window[0] == "--output-format" && window[1].eq_ignore_ascii_case("json"))
        || argv
            .iter()
            .any(|argument| argument.eq_ignore_ascii_case("--output-format=json"))
}

fn operation_from_argv(argv: &[String]) -> GraphOperation {
    match argv.first().map(String::as_str) {
        Some("explain") => GraphOperation::Explain,
        Some("visualize") => GraphOperation::Visualize,
        _ => GraphOperation::Validate,
    }
}

fn write_error(operation: GraphOperation, error: &GraphCommandError, is_json: bool) {
    if is_json {
        match serde_json::to_string(&error.report(operation)) {
            Ok(report) => println!("{report}"),
            Err(serialization_error) => eprintln!(
                "aiperf graph {}: failed to serialize error report: {serialization_error}",
                operation.as_str()
            ),
        }
    } else {
        eprintln!("aiperf graph {}: {}", operation.as_str(), error.message);
    }
}
