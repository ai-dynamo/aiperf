// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Injected protocol-v2 composition root for one runner process.
//!
//! The stock binary and statically linked custom distributions use the same
//! coordinator. Concrete registries meet here exactly once; backend/workload
//! pair adapters receive only the frozen [`RunnerRunContext`] and their own
//! validated configuration.

use std::collections::BTreeMap;
use std::sync::Arc;

use aiperf_extensions::{AiperfRegistry, AiperfRegistryFactory};
use aiperf_graph::input::GraphInputAdapterResolver;
use anyhow::{Context, Result, ensure};
use serde::Serialize;

use crate::protocol_v2::{
    DeferredCheckV2, RUNNER_PROTOCOL_V2, RunTerminalV2, RunValidationV2, RunnerDiagnosticV2,
    RunnerEnvelopeV2, RunnerFailureStageV2, RunnerOperationV2, ValidationCompletenessV2,
};
use crate::redaction::redact_diagnostic;
use crate::registry::{
    RunnerRegistry, RunnerRegistryFactory, RunnerRunContext, validate_endpoint_profiles_v2,
};

/// Exactly one typed response emitted for a protocol-v2 request.
#[derive(Debug, Serialize)]
#[serde(untagged)]
pub enum RunnerResponseV2 {
    /// Response to a side-effect-free validation operation.
    Validation(RunValidationV2),
    /// Terminal response to an execution operation.
    Terminal(RunTerminalV2),
}

/// One coordinator result plus the process exit status that carries it.
#[derive(Debug)]
pub struct RunnerProcessResultV2 {
    /// Exactly one response object for stdout JSONL.
    pub response: RunnerResponseV2,
    /// Zero for success, one for a validated run failure, or two for protocol failure.
    pub exit_code: i32,
}

/// Frozen implementation universe for one fresh runner child.
///
/// A custom statically linked executable injects alternate registry factories
/// or graph-input adapters here. The execution coordinator and all hot-path
/// scheduling code remain unchanged.
pub struct RunnerV2Coordinator {
    distribution_id: String,
    runner_registry: RunnerRegistry,
    product_registry: Arc<AiperfRegistry>,
    graph_inputs: Arc<dyn GraphInputAdapterResolver>,
}

impl std::fmt::Debug for RunnerV2Coordinator {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("RunnerV2Coordinator")
            .field("distribution_id", &self.distribution_id)
            .field("supported_pairs", &self.runner_registry.supported_pairs())
            .finish_non_exhaustive()
    }
}

impl RunnerV2Coordinator {
    /// Compose every startup registry exactly once for this child process.
    pub fn new(
        distribution_id: impl Into<String>,
        runner_registry_factory: &dyn RunnerRegistryFactory,
        product_registry_factory: &dyn AiperfRegistryFactory,
        graph_inputs: Arc<dyn GraphInputAdapterResolver>,
    ) -> Result<Self> {
        let distribution_id = distribution_id.into();
        validate_distribution_id(&distribution_id)?;
        let runner_registry = runner_registry_factory
            .build()
            .context("composing runner backend/workload registry")?;
        let product_registry = Arc::new(
            product_registry_factory
                .build()
                .context("composing AIPerf product registry")?,
        );
        Ok(Self {
            distribution_id,
            runner_registry,
            product_registry,
            graph_inputs,
        })
    }

    /// Validate or execute one strict authored envelope through the frozen registries.
    pub fn handle(&self, envelope: RunnerEnvelopeV2) -> RunnerProcessResultV2 {
        let operation = envelope.operation;
        let benchmark_id = Some(envelope.run.identity.benchmark_id.clone());
        if envelope.expected_distribution_id != self.distribution_id {
            return failure(
                operation,
                self.distribution_id.clone(),
                benchmark_id,
                RunnerFailureStageV2::Protocol,
                "distribution_mismatch",
                "expected_distribution_id does not match the image executing this process",
                2,
            );
        }
        if let Err(error) = envelope.validate_outer() {
            return failure(
                operation,
                self.distribution_id.clone(),
                benchmark_id,
                RunnerFailureStageV2::Validation,
                "invalid_run",
                format!("{error:#}"),
                1,
            );
        }

        let endpoint_profiles =
            match validate_endpoint_profiles_v2(&envelope.run, self.product_registry.endpoints()) {
                Ok(profiles) => profiles,
                Err(error) => {
                    return failure(
                        operation,
                        self.distribution_id.clone(),
                        benchmark_id,
                        RunnerFailureStageV2::Validation,
                        "invalid_endpoint_profiles",
                        format!("{error:#}"),
                        1,
                    );
                }
            };
        let context = match RunnerRunContext::new(
            self.product_registry.clone(),
            self.graph_inputs.clone(),
            endpoint_profiles,
        ) {
            Ok(context) => context,
            Err(error) => {
                return failure(
                    operation,
                    self.distribution_id.clone(),
                    benchmark_id,
                    RunnerFailureStageV2::Validation,
                    "invalid_run_context",
                    format!("{error:#}"),
                    1,
                );
            }
        };
        if has_unavailable_sidecar(&envelope) {
            return failure(
                operation,
                self.distribution_id.clone(),
                benchmark_id,
                RunnerFailureStageV2::Validation,
                "unsupported_sidecar",
                "protocol-v2 sidecar preparation adapters are not registered in this distribution",
                1,
            );
        }
        let selection = match self
            .runner_registry
            .validate_selection(&envelope.run.backend, &envelope.run.workload)
        {
            Ok(selection) => selection,
            Err(error) => {
                return failure(
                    operation,
                    self.distribution_id.clone(),
                    benchmark_id,
                    RunnerFailureStageV2::Validation,
                    "invalid_backend_workload_selection",
                    format!("{error:#}"),
                    1,
                );
            }
        };
        if let Err(error) = self
            .runner_registry
            .validate_run(&envelope.run, &context, &selection)
        {
            return failure(
                operation,
                self.distribution_id.clone(),
                benchmark_id,
                RunnerFailureStageV2::Validation,
                "invalid_backend_workload_run",
                format!("{error:#}"),
                1,
            );
        }

        if operation == RunnerOperationV2::Validate {
            return RunnerProcessResultV2 {
                response: RunnerResponseV2::Validation(RunValidationV2 {
                    protocol_version: RUNNER_PROTOCOL_V2,
                    event: "run_validation",
                    distribution_id: self.distribution_id.clone(),
                    benchmark_id,
                    success: true,
                    completeness: ValidationCompletenessV2::Static,
                    deferred_checks: vec![DeferredCheckV2 {
                        code: "workload_preparation".to_owned(),
                        path: "run.workload".to_owned(),
                        reason: "dataset, tokenizer, endpoint-profile references, and backend resources require execution preparation"
                            .to_owned(),
                    }],
                    errors: Vec::new(),
                }),
                exit_code: 0,
            };
        }

        let operation =
            match self
                .runner_registry
                .prepare_with_context(&envelope.run, &context, selection)
            {
                Ok(operation) => operation,
                Err(error) => {
                    return terminal_failure(
                        self.distribution_id.clone(),
                        benchmark_id,
                        RunnerFailureStageV2::Preparation,
                        "preparation_failed",
                        format!("{error:#}"),
                        1,
                    );
                }
            };
        match operation.execute() {
            Ok(outcome) => RunnerProcessResultV2 {
                response: RunnerResponseV2::Terminal(RunTerminalV2 {
                    protocol_version: RUNNER_PROTOCOL_V2,
                    event: "run_terminal",
                    distribution_id: self.distribution_id.clone(),
                    benchmark_id,
                    success: true,
                    report_path: Some(outcome.report_path),
                    stage: None,
                    errors: Vec::new(),
                    provenance: outcome.provenance,
                }),
                exit_code: 0,
            },
            Err(error) => terminal_failure(
                self.distribution_id.clone(),
                benchmark_id,
                RunnerFailureStageV2::Execution,
                "execution_failed",
                format!("{error:#}"),
                1,
            ),
        }
    }

    /// Borrow the exact frozen product registry used by this process.
    pub fn product_registry(&self) -> &AiperfRegistry {
        self.product_registry.as_ref()
    }
}

fn validate_distribution_id(value: &str) -> Result<()> {
    let digest = value.strip_prefix("blake3:").unwrap_or_default();
    ensure!(
        digest.len() == 64
            && digest
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase()),
        "runner distribution_id must use blake3: followed by 64 lowercase hexadecimal digits"
    );
    Ok(())
}

fn has_unavailable_sidecar(envelope: &RunnerEnvelopeV2) -> bool {
    let sidecars = &envelope.run.sidecars;
    sidecars.gpu_telemetry.is_some()
        || sidecars.network_latency.is_some()
        || sidecars.server_metrics.is_some()
        || sidecars.live_streaming.is_some()
}

#[allow(clippy::too_many_arguments)]
fn failure(
    operation: RunnerOperationV2,
    distribution_id: String,
    benchmark_id: Option<String>,
    stage: RunnerFailureStageV2,
    code: &str,
    message: impl Into<String>,
    exit_code: i32,
) -> RunnerProcessResultV2 {
    if operation == RunnerOperationV2::Validate {
        RunnerProcessResultV2 {
            response: RunnerResponseV2::Validation(RunValidationV2 {
                protocol_version: RUNNER_PROTOCOL_V2,
                event: "run_validation",
                distribution_id,
                benchmark_id,
                success: false,
                completeness: ValidationCompletenessV2::Static,
                deferred_checks: Vec::new(),
                errors: vec![diagnostic(code, message)],
            }),
            exit_code,
        }
    } else {
        terminal_failure(
            distribution_id,
            benchmark_id,
            stage,
            code,
            message,
            exit_code,
        )
    }
}

fn terminal_failure(
    distribution_id: String,
    benchmark_id: Option<String>,
    stage: RunnerFailureStageV2,
    code: &str,
    message: impl Into<String>,
    exit_code: i32,
) -> RunnerProcessResultV2 {
    RunnerProcessResultV2 {
        response: RunnerResponseV2::Terminal(RunTerminalV2 {
            protocol_version: RUNNER_PROTOCOL_V2,
            event: "run_terminal",
            distribution_id,
            benchmark_id,
            success: false,
            report_path: None,
            stage: Some(stage),
            errors: vec![diagnostic(code, message)],
            provenance: BTreeMap::new(),
        }),
        exit_code,
    }
}

fn diagnostic(code: &str, message: impl Into<String>) -> RunnerDiagnosticV2 {
    RunnerDiagnosticV2 {
        code: code.to_owned(),
        message: redact_diagnostic(message.into()),
        path: None,
    }
}
