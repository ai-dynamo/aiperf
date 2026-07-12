// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Stdio entry point for one orchestrator-authored benchmark run.

use std::collections::BTreeMap;
use std::io::{self, Read, Write};
use std::sync::Arc;

use aiperf_graph::input::GraphInputAdapterRegistry;
use aiperf_runner::coordinator::RunnerV2Coordinator;
use aiperf_runner::dataset_input::BuiltinRunnerDatasetInputAdapterResolver;
use aiperf_runner::protocol_v2::{
    RUNNER_PROTOCOL_V2, RunTerminalV2, RunValidationV2, RunnerDiagnosticV2, RunnerEnvelopeV2,
    RunnerFailureStageV2, RunnerOperationV2, ValidationCompletenessV2,
};
use aiperf_runner::redaction::redact_diagnostic;
use aiperf_runner::registry::BuiltinRunnerRegistryFactory;
use aiperf_runner::{
    RUNNER_PROTOCOL_VERSION, RunRequest, RunTerminal, RunnerCapabilities, execute_run,
};
use serde::Deserialize;
use serde_json::{Value, value::RawValue};

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

fn main() {
    let arguments = std::env::args_os().skip(1).collect::<Vec<_>>();
    if arguments.len() == 1 && arguments[0] == "--capabilities" {
        match RunnerCapabilities::current() {
            Ok(capabilities) => write_json_line(&capabilities, 0),
            Err(error) => {
                eprintln!("failed to identify executing aiperf-runner image: {error}");
                std::process::exit(2);
            }
        }
    }
    if !arguments.is_empty() {
        eprintln!("usage: aiperf-runner [--capabilities]");
        std::process::exit(2);
    }

    let mut input = Vec::new();
    if let Err(error) = io::stdin().read_to_end(&mut input) {
        eprintln!("failed to read runner request from stdin: {error}");
        std::process::exit(2);
    }
    let probe = match serde_json::from_slice::<ProtocolVersionProbe>(&input) {
        Ok(probe) => probe,
        Err(error) => write_json_line(
            &RunTerminal::failed(None, "protocol", format!("invalid run request: {error}")),
            2,
        ),
    };
    match probe.protocol_version {
        RUNNER_PROTOCOL_VERSION => run_v1(&input),
        RUNNER_PROTOCOL_V2 => run_v2(&input),
        version => write_json_line(
            &RunTerminal::failed(
                None,
                "protocol",
                format!(
                    "runner protocol {version} is unsupported; expected one of [{RUNNER_PROTOCOL_VERSION}, {RUNNER_PROTOCOL_V2}]"
                ),
            ),
            2,
        ),
    }
}

#[derive(Deserialize)]
struct ProtocolVersionProbe {
    protocol_version: u32,
}

#[derive(Deserialize)]
struct EnvelopeBootstrapV2 {
    protocol_version: u32,
    operation: RunnerOperationV2,
    expected_distribution_id: String,
    run: Box<RawValue>,
}

fn run_v1(input: &[u8]) -> ! {
    let terminal = match serde_json::from_slice::<RunRequest>(input) {
        Ok(request) if request.protocol_version == RUNNER_PROTOCOL_VERSION => {
            let run_id = request.run.benchmark_id.clone();
            match execute_run(request) {
                Ok(result) => result,
                Err(error) => RunTerminal::failed(Some(run_id), "execution", format!("{error:#}")),
            }
        }
        Ok(request) => RunTerminal::failed(
            Some(request.run.benchmark_id),
            "protocol",
            format!(
                "runner protocol {} is unsupported; expected {}",
                request.protocol_version, RUNNER_PROTOCOL_VERSION
            ),
        ),
        Err(error) => {
            RunTerminal::failed(None, "protocol", format!("invalid run request: {error}"))
        }
    };
    write_json_line(&terminal, if terminal.success { 0 } else { 1 });
}

fn run_v2(input: &[u8]) -> ! {
    let distribution_id = match aiperf_runner::current_distribution_id() {
        Ok(distribution_id) => distribution_id,
        Err(error) => {
            eprintln!("failed to identify executing aiperf-runner image: {error}");
            std::process::exit(2);
        }
    };
    let bootstrap = match serde_json::from_slice::<EnvelopeBootstrapV2>(input) {
        Ok(bootstrap) => bootstrap,
        Err(error) => {
            write_v2_protocol_failure(
                operation_hint(input),
                distribution_id,
                benchmark_id_hint(input),
                "invalid_request",
                format!("invalid protocol-v2 request: {error}"),
            );
        }
    };
    if bootstrap.protocol_version != RUNNER_PROTOCOL_V2 {
        write_v2_protocol_failure(
            Some(bootstrap.operation),
            distribution_id,
            benchmark_id_from_raw(&bootstrap.run),
            "unsupported_protocol",
            format!(
                "runner protocol {} is unsupported; expected {RUNNER_PROTOCOL_V2}",
                bootstrap.protocol_version
            ),
        );
    }
    if bootstrap.expected_distribution_id != distribution_id {
        write_v2_protocol_failure(
            Some(bootstrap.operation),
            distribution_id,
            benchmark_id_from_raw(&bootstrap.run),
            "distribution_mismatch",
            "expected_distribution_id does not match the image executing this process".to_owned(),
        );
    }

    let envelope = match serde_json::from_slice::<RunnerEnvelopeV2>(input) {
        Ok(envelope) => envelope,
        Err(error) => write_v2_protocol_failure(
            Some(bootstrap.operation),
            distribution_id,
            benchmark_id_from_raw(&bootstrap.run),
            "invalid_request",
            format!("invalid protocol-v2 request: {error}"),
        ),
    };
    let graph_inputs = Arc::new(GraphInputAdapterRegistry::with_builtin_adapters());
    let dataset_inputs = Arc::new(BuiltinRunnerDatasetInputAdapterResolver::new());
    let coordinator = match RunnerV2Coordinator::new(
        distribution_id.clone(),
        &BuiltinRunnerRegistryFactory,
        &aiperf_extensions::BuiltinAiperfRegistryFactory,
        graph_inputs,
        dataset_inputs,
    ) {
        Ok(coordinator) => coordinator,
        Err(error) => write_v2_validation_failure(
            envelope.operation,
            distribution_id,
            Some(envelope.run.identity.benchmark_id.clone()),
            "coordinator_initialization_failed",
            format!("{error:#}"),
        ),
    };
    let result = coordinator.handle(envelope);
    write_json_line(&result.response, result.exit_code);
}

fn operation_hint(input: &[u8]) -> Option<RunnerOperationV2> {
    let value: Value = serde_json::from_slice(input).ok()?;
    match value.get("operation")?.as_str()? {
        "validate" => Some(RunnerOperationV2::Validate),
        "execute" => Some(RunnerOperationV2::Execute),
        _ => None,
    }
}

fn benchmark_id_hint(input: &[u8]) -> Option<String> {
    let value: Value = serde_json::from_slice(input).ok()?;
    value
        .pointer("/run/identity/benchmark_id")?
        .as_str()
        .map(str::to_owned)
}

fn benchmark_id_from_raw(run: &RawValue) -> Option<String> {
    let value: Value = serde_json::from_str(run.get()).ok()?;
    value
        .pointer("/identity/benchmark_id")?
        .as_str()
        .map(str::to_owned)
}

fn write_v2_protocol_failure(
    operation: Option<RunnerOperationV2>,
    distribution_id: String,
    benchmark_id: Option<String>,
    code: &str,
    message: String,
) -> ! {
    match operation {
        Some(RunnerOperationV2::Validate) => write_json_line(
            &RunValidationV2 {
                protocol_version: RUNNER_PROTOCOL_V2,
                event: "run_validation",
                distribution_id,
                benchmark_id,
                success: false,
                completeness: ValidationCompletenessV2::Static,
                deferred_checks: Vec::new(),
                errors: vec![diagnostic(code, message)],
            },
            2,
        ),
        Some(RunnerOperationV2::Execute) | None => write_v2_terminal_failure(
            distribution_id,
            benchmark_id,
            RunnerFailureStageV2::Protocol,
            code,
            message,
            2,
        ),
    }
}

fn write_v2_validation_failure(
    operation: RunnerOperationV2,
    distribution_id: String,
    benchmark_id: Option<String>,
    code: &str,
    message: String,
) -> ! {
    match operation {
        RunnerOperationV2::Validate => write_json_line(
            &RunValidationV2 {
                protocol_version: RUNNER_PROTOCOL_V2,
                event: "run_validation",
                distribution_id,
                benchmark_id,
                success: false,
                completeness: ValidationCompletenessV2::Static,
                deferred_checks: Vec::new(),
                errors: vec![diagnostic(code, message)],
            },
            1,
        ),
        RunnerOperationV2::Execute => write_v2_terminal_failure(
            distribution_id,
            benchmark_id,
            RunnerFailureStageV2::Validation,
            code,
            message,
            1,
        ),
    }
}

fn write_v2_terminal_failure(
    distribution_id: String,
    benchmark_id: Option<String>,
    stage: RunnerFailureStageV2,
    code: &str,
    message: String,
    exit_code: i32,
) -> ! {
    write_json_line(
        &RunTerminalV2 {
            protocol_version: RUNNER_PROTOCOL_V2,
            event: "run_terminal",
            distribution_id,
            benchmark_id,
            success: false,
            report_path: None,
            stage: Some(stage),
            errors: vec![diagnostic(code, message)],
            provenance: BTreeMap::new(),
        },
        exit_code,
    )
}

fn diagnostic(code: &str, message: String) -> RunnerDiagnosticV2 {
    RunnerDiagnosticV2 {
        code: code.to_owned(),
        message: redact_diagnostic(message),
        path: None,
    }
}

fn write_json_line(value: &impl serde::Serialize, exit_code: i32) -> ! {
    let mut stdout = io::stdout().lock();
    if serde_json::to_writer(&mut stdout, value).is_err()
        || stdout.write_all(b"\n").is_err()
        || stdout.flush().is_err()
    {
        std::process::exit(2);
    }
    std::process::exit(exit_code);
}
