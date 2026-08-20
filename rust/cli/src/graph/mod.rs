// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Native local-only graph inspection command boundary.

mod explain;
pub mod report;
mod validate;
mod visualize;

use std::io::{self, Write};
use std::path::{Path, PathBuf};

use aiperf_runtime::config::model::dataset::RecordedAgentSourceFormat;
use aiperf_runtime::engine::graph_input::{
    BuiltinRunnerGraphInputAdapterResolver, GraphInputAdapterResolver, PreparedRunnerGraphInput,
    prepare_local_graph_inspection_input,
};
use aiperf_runtime::engine::preparation::{LocalTokenizerError, load_local_tokenizer};
use clap::builder::PossibleValuesParser;
use clap::{Args, CommandFactory, FromArgMatches, Parser, Subcommand, ValueEnum};
use tokio::runtime::Builder;
use tokio::task::LocalSet;

use self::report::{GraphCommandError, GraphCommandErrorCode, GraphOperation};

#[derive(Debug, Parser)]
#[command(
    name = "graph",
    disable_help_subcommand = true,
    subcommand_required = true,
    arg_required_else_help = true
)]
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
    /// Endpoint type used to validate graph-input compatibility.
    #[arg(long, default_value = "chat")]
    endpoint_type: String,
    /// Imported recorded-agent source format.
    #[arg(long)]
    source_format: Option<RecordedAgentSourceFormat>,
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
    let cli = match parse_graph_cli(&full) {
        Ok(cli) => cli,
        Err(error) => {
            if requested_json(argv) {
                let failure = GraphCommandError::new(
                    GraphCommandErrorCode::InvalidArguments,
                    format!("invalid arguments: {}", error.kind()),
                    None,
                );
                write_error(operation_from_argv(argv), &failure, true)?;
            } else {
                error.print()?;
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

fn parse_graph_cli(argv: &[String]) -> Result<GraphCli, clap::Error> {
    let command = GraphCli::command().after_help(graph_format_help());
    let matches = command.try_get_matches_from(argv)?;
    GraphCli::from_arg_matches(&matches)
}

fn graph_format_help() -> String {
    format!(
        "Supported built-in graph formats: {}",
        BuiltinRunnerGraphInputAdapterResolver::new()
            .supported_formats()
            .join(", ")
    )
}

fn run_validate(args: ValidateArgs) -> anyhow::Result<i32> {
    let requires_arrival_offsets = matches!(args.pace, Some(Pace::Arrival));
    match load(args.common) {
        Ok(input) => match validate::run(input, args.output_format, requires_arrival_offsets) {
            Ok(status) => Ok(status),
            Err(error) => {
                write_error(
                    GraphOperation::Validate,
                    &error,
                    matches!(args.output_format, TextJsonFormat::Json),
                )?;
                Ok(2)
            }
        },
        Err(error) => {
            write_error(
                GraphOperation::Validate,
                &error,
                matches!(args.output_format, TextJsonFormat::Json),
            )?;
            Ok(2)
        }
    }
}

fn run_explain(args: ExplainArgs) -> anyhow::Result<i32> {
    let is_json = matches!(args.output_format, TextJsonFormat::Json);
    match load(args.common) {
        Ok(input) => match explain::run(input, args.output_format) {
            Ok(()) => Ok(0),
            Err(error) => Ok(write_explain_error(&error, is_json)),
        },
        Err(error) => Ok(write_explain_error(&error, is_json)),
    }
}

fn write_explain_error(error: &GraphCommandError, is_json: bool) -> i32 {
    if is_json {
        let stdout = io::stdout();
        let mut output = stdout.lock();
        explain_error_status(error, true, &mut output)
    } else {
        let stderr = io::stderr();
        let mut output = stderr.lock();
        explain_error_status(error, false, &mut output)
    }
}

fn explain_error_status<W: Write>(error: &GraphCommandError, is_json: bool, output: &mut W) -> i32 {
    let _ = write_error_to(GraphOperation::Explain, error, is_json, output);
    2
}

fn run_visualize(args: VisualizeArgs) -> anyhow::Result<i32> {
    match load(args.common) {
        Ok(input) => match visualize::run(
            input,
            args.trace.as_deref(),
            args.output.as_deref(),
            args.output_format,
            args.no_validate,
        ) {
            Ok(status) => Ok(status),
            Err(error) => {
                write_error(GraphOperation::Visualize, &error, false)?;
                Ok(2)
            }
        },
        Err(error) => {
            write_error(GraphOperation::Visualize, &error, false)?;
            Ok(2)
        }
    }
}

fn load(args: CommonArgs) -> Result<LoadedGraphInput, GraphCommandError> {
    let source = validate_local_source(&args.path)?;
    let source_text = source.display().to_string();
    let tokenizer = load_local_tokenizer(Some(&args.tokenizer)).map_err(|error| {
        let code = match error {
            LocalTokenizerError::Unsupported { .. } => GraphCommandErrorCode::TokenizerUnsupported,
            LocalTokenizerError::Load { .. } => GraphCommandErrorCode::TokenizerLoadFailed,
        };
        let message = match code {
            GraphCommandErrorCode::TokenizerUnsupported => {
                "tokenizer must be a built-in encoding or an existing local path"
            }
            _ => "local tokenizer could not be loaded",
        };
        GraphCommandError::with_cause(
            code,
            message,
            Some(source_text.clone()),
            anyhow::Error::new(error),
        )
    })?;
    let resolver = BuiltinRunnerGraphInputAdapterResolver::new();
    let runtime = Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|error| {
            GraphCommandError::with_cause(
                GraphCommandErrorCode::InputLoweringFailed,
                "could not initialize local graph inspection runtime",
                Some(source_text.clone()),
                anyhow::Error::new(error),
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
            &args.endpoint_type,
            (args.graph_format == "agent_recording")
                .then_some(args.source_format)
                .flatten(),
            args.seed,
        ),
    );
    let prepared = prepared.map_err(|error| {
        let code = if is_input_decode_error(&error) {
            GraphCommandErrorCode::InputDecodeFailed
        } else {
            GraphCommandErrorCode::InputLoweringFailed
        };
        let message = if matches!(code, GraphCommandErrorCode::InputDecodeFailed) {
            "graph input could not be decoded"
        } else {
            "graph input could not be lowered"
        };
        GraphCommandError::with_cause(code, message, Some(source_text), error)
    })?;
    Ok(LoadedGraphInput { source, prepared })
}

fn is_input_decode_error(error: &anyhow::Error) -> bool {
    error.chain().any(|cause| {
        if cause.downcast_ref::<serde_json::Error>().is_some() {
            return true;
        }
        if cause.downcast_ref::<serde_yaml::Error>().is_some() {
            return true;
        }
        let text = cause.to_string().to_ascii_lowercase();
        text.contains("decoding direct")
            || text.contains("decoding otlp json")
            || text.contains("invalid json:")
    })
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
        Err(error) => {
            return Err(GraphCommandError::with_cause(
                GraphCommandErrorCode::SourceNotFound,
                "graph source does not exist locally",
                Some(path.display().to_string()),
                anyhow::Error::new(error),
            ));
        }
    }
    path.canonicalize().map_err(|error| {
        GraphCommandError::with_cause(
            GraphCommandErrorCode::SourceNotFound,
            "graph source could not be canonicalized",
            Some(path.display().to_string()),
            anyhow::Error::new(error),
        )
    })
}

fn graph_format_parser() -> PossibleValuesParser {
    PossibleValuesParser::new(
        BuiltinRunnerGraphInputAdapterResolver::new()
            .supported_formats()
            .into_iter()
            .map(str::to_owned)
            .collect::<Vec<_>>(),
    )
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

fn write_error(
    operation: GraphOperation,
    error: &GraphCommandError,
    is_json: bool,
) -> anyhow::Result<()> {
    if is_json {
        let stdout = io::stdout();
        let mut output = stdout.lock();
        write_error_to(operation, error, true, &mut output)
    } else {
        let stderr = io::stderr();
        let mut output = stderr.lock();
        write_error_to(operation, error, false, &mut output)
    }
}

fn write_error_to<W: Write>(
    operation: GraphOperation,
    error: &GraphCommandError,
    is_json: bool,
    output: &mut W,
) -> anyhow::Result<()> {
    if is_json {
        let mut report = serde_json::to_vec(&error.report(operation))?;
        report.push(b'\n');
        output.write_all(&report)?;
    } else {
        writeln!(
            output,
            "aiperf graph {}: {}",
            operation.as_str(),
            error.message
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::io::{self, Write};

    use super::{
        GraphOperation, TextJsonFormat, explain_error_status, is_input_decode_error, validate,
        write_error_to,
    };
    use crate::graph::report::{
        GraphCommandError, GraphCommandErrorCode, GraphErrorReport, GraphIssueReport,
        GraphIssueSeverityReport, GraphIssueSummary, GraphValidateReport,
    };

    struct FailFirstWriter {
        writes: usize,
        bytes: Vec<u8>,
    }

    impl Write for FailFirstWriter {
        fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
            self.writes += 1;
            if self.writes == 1 {
                return Err(io::Error::new(
                    io::ErrorKind::BrokenPipe,
                    "test broken pipe",
                ));
            }
            self.bytes.extend_from_slice(buffer);
            Ok(buffer.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    struct PersistentFailWriter;

    impl Write for PersistentFailWriter {
        fn write(&mut self, _buffer: &[u8]) -> io::Result<usize> {
            Err(io::Error::new(
                io::ErrorKind::BrokenPipe,
                "persistent test failure",
            ))
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    fn validation_report() -> GraphValidateReport {
        GraphValidateReport {
            schema_version: "aiperf.graph.validate.v1".to_owned(),
            source: "/tmp/input.graph".to_owned(),
            format: "dag_jsonl".to_owned(),
            root_count: 1,
            node_count: 1,
            issues: vec![GraphIssueReport {
                code: "test".to_owned(),
                severity: GraphIssueSeverityReport::Error,
                trace_id: None,
                phase: None,
                location: None,
                message: "test issue".to_owned(),
                context: BTreeMap::new(),
            }],
            summary: GraphIssueSummary {
                errors: 1,
                warnings: 0,
            },
        }
    }

    #[test]
    fn decode_named_lowering_error_is_not_an_input_decode_error() {
        let error = anyhow::anyhow!("tokenizer decode_lossy failed")
            .context("lowering recorded graph materialization");

        assert!(!is_input_decode_error(&error));
    }

    #[test]
    fn json_validation_write_failure_emits_only_one_fatal_document() {
        let report = validation_report();
        let mut writer = FailFirstWriter {
            writes: 0,
            bytes: Vec::new(),
        };
        let error = validate::emit_report(&report, TextJsonFormat::Json, &mut writer)
            .expect_err("first write rejects validation JSON");

        assert_eq!(error.code.as_str(), "output-write-failed");
        write_error_to(GraphOperation::Validate, &error, true, &mut writer)
            .expect("second write emits fatal document");
        let fatal: GraphErrorReport =
            serde_json::from_slice(&writer.bytes).expect("one parseable fatal JSON document");
        assert_eq!(fatal.schema_version, "aiperf.graph.error.v1");
        assert_eq!(fatal.code.as_str(), "output-write-failed");
        assert_eq!(fatal.source.as_deref(), Some("/tmp/input.graph"));
    }

    #[test]
    fn persistent_explain_output_failure_returns_exit_two_without_fallback_error() {
        let error = GraphCommandError::new(
            GraphCommandErrorCode::OutputWriteFailed,
            "could not write graph explanation report",
            Some("/tmp/canonical.graph".to_owned()),
        );
        let mut writer = PersistentFailWriter;

        assert_eq!(explain_error_status(&error, true, &mut writer), 2);
    }
}
