// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Injected protocol-v2 composition root for one runner process.
//!
//! The stock binary and statically linked custom distributions use the same
//! coordinator. Concrete registries meet here exactly once; backend/workload
//! pair adapters receive only the frozen [`RunnerRunContext`] and their own
//! validated configuration.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

use aiperf::report::finalize_and_write_native_report_json;
use aiperf_extensions::{AiperfRegistry, AiperfRegistryFactory};
use aiperf_metrics::ReportRunProvenance;
use anyhow::{Context, Result, ensure};
use serde::Serialize;

use crate::dataset_input::RunnerDatasetInputAdapterResolver;
use crate::execution_factories::RunnerExecutionFactories;
use crate::graph_input::RunnerGraphInputAdapterResolver;
use crate::protocol::RunnerCapabilities;
use crate::protocol_v2::{
    DeferredCheckV2, RUNNER_PROTOCOL_V2, RunTerminalV2, RunValidationV2, RunnerDiagnosticV2,
    RunnerEnvelopeV2, RunnerFailureStageV2, RunnerOperationV2, ValidationCompletenessV2,
};
use crate::redaction::redact_diagnostic;
use crate::registry::{
    PreparedRunOutcome, RunnerRegistry, RunnerRegistryFactory, RunnerRunContext,
    validate_endpoint_profiles_v2,
};
use crate::sidecar_input::RunnerSidecarInputAdapterResolver;

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
    execution_factories: RunnerExecutionFactories,
    graph_inputs: Arc<dyn RunnerGraphInputAdapterResolver>,
    dataset_inputs: Arc<dyn RunnerDatasetInputAdapterResolver>,
    sidecar_inputs: Arc<dyn RunnerSidecarInputAdapterResolver>,
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
        execution_factories: RunnerExecutionFactories,
        graph_inputs: Arc<dyn RunnerGraphInputAdapterResolver>,
        dataset_inputs: Arc<dyn RunnerDatasetInputAdapterResolver>,
        sidecar_inputs: Arc<dyn RunnerSidecarInputAdapterResolver>,
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
            execution_factories,
            graph_inputs,
            dataset_inputs,
            sidecar_inputs,
        })
    }

    /// Advertise capabilities from this coordinator's exact frozen registries.
    ///
    /// Custom distributions use this method for `--capabilities` so discovery,
    /// validation, and execution observe one implementation universe. No
    /// registry factory is invoked a second time.
    pub fn capabilities(&self) -> RunnerCapabilities {
        RunnerCapabilities::from_registries(
            self.distribution_id.clone(),
            &self.runner_registry,
            self.product_registry.as_ref(),
        )
    }

    /// Borrow the single authored graph-input resolver shared by protocol-v2
    /// preparation and protocol-v1 compatibility execution.
    pub fn graph_inputs(&self) -> &dyn RunnerGraphInputAdapterResolver {
        self.graph_inputs.as_ref()
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

        let selection = match self
            .runner_registry
            .validate_selection_for_run(&envelope.run)
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
        let sidecar_inputs = match self
            .sidecar_inputs
            .prepare(&envelope.run.sidecars.authored_inputs())
        {
            Ok(sidecars) => Arc::new(sidecars),
            Err(error) => {
                return failure(
                    operation,
                    self.distribution_id.clone(),
                    benchmark_id,
                    RunnerFailureStageV2::Validation,
                    "invalid_sidecars",
                    format!("{error:#}"),
                    1,
                );
            }
        };
        let context = match RunnerRunContext::new(
            self.product_registry.clone(),
            self.execution_factories.clone(),
            self.graph_inputs.clone(),
            self.dataset_inputs.clone(),
            sidecar_inputs,
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
        let backend_id = selection.backend_id().to_owned();
        let workload_id = selection.workload_id().to_owned();
        let report_provenance = match context.report_provenance(
            self.distribution_id.clone(),
            backend_id.clone(),
            workload_id.clone(),
        ) {
            Ok(provenance) => provenance,
            Err(error) => {
                return failure(
                    operation,
                    self.distribution_id.clone(),
                    benchmark_id,
                    RunnerFailureStageV2::Validation,
                    "invalid_report_provenance",
                    format!("{error:#}"),
                    1,
                );
            }
        };

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

        let report_path = envelope.run.artifact_target.join("native-v2.json");
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
            Ok(outcome) => {
                let mut provenance =
                    match persist_prepared_report(outcome, report_provenance, &report_path) {
                        Ok(provenance) => provenance,
                        Err(error) => {
                            return terminal_failure(
                                self.distribution_id.clone(),
                                benchmark_id,
                                RunnerFailureStageV2::Reporting,
                                error.code,
                                error.message,
                                1,
                            );
                        }
                    };
                provenance.insert("backend".into(), backend_id);
                provenance.insert("workload".into(), workload_id);
                RunnerProcessResultV2 {
                    response: RunnerResponseV2::Terminal(RunTerminalV2 {
                        protocol_version: RUNNER_PROTOCOL_V2,
                        event: "run_terminal",
                        distribution_id: self.distribution_id.clone(),
                        benchmark_id,
                        success: true,
                        report_path: Some(report_path),
                        stage: None,
                        errors: Vec::new(),
                        provenance,
                    }),
                    exit_code: 0,
                }
            }
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

#[derive(Debug)]
struct ReportPersistenceFailure {
    code: &'static str,
    message: String,
}

fn persist_prepared_report(
    outcome: PreparedRunOutcome,
    report_provenance: ReportRunProvenance,
    report_path: &Path,
) -> std::result::Result<BTreeMap<String, String>, ReportPersistenceFailure> {
    if report_path.exists() {
        return Err(ReportPersistenceFailure {
            code: "report_target_exists",
            message: format!(
                "native-v2 report target already exists: {}",
                report_path.display()
            ),
        });
    }
    let PreparedRunOutcome {
        native_report,
        report_facts,
        provenance,
        report_commit,
    } = outcome;
    finalize_and_write_native_report_json(
        native_report,
        report_provenance,
        report_facts,
        report_path,
    )
    .map_err(|error| ReportPersistenceFailure {
        code: "reporting_failed",
        message: format!("{error:#}"),
    })?;
    if let Some(report_commit) = report_commit {
        report_commit
            .commit()
            .map_err(|error| ReportPersistenceFailure {
                code: "report_commit_failed",
                message: format!("post-persistence report lifecycle commit failed: {error:#}"),
            })?;
    }
    Ok(provenance)
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

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;
    use crate::registry::PreparedReportCommit;

    #[derive(Debug)]
    struct TrackingCommit {
        calls: Arc<AtomicUsize>,
        fail: bool,
    }

    impl PreparedReportCommit for TrackingCommit {
        fn commit(self: Box<Self>) -> Result<()> {
            let previous = self.calls.fetch_add(1, Ordering::SeqCst);
            ensure!(
                previous == 0,
                "report commit hook was invoked more than once"
            );
            ensure!(!self.fail, "fixture report commit failure");
            Ok(())
        }
    }

    fn provenance() -> ReportRunProvenance {
        ReportRunProvenance::new(
            format!("blake3:{}", "a".repeat(64)),
            "online_http",
            "evaluation",
            Vec::new(),
            vec![aiperf_metrics::ReportEndpointProfileIdentity::new("default", "chat").unwrap()],
        )
        .unwrap()
    }

    fn outcome(calls: Arc<AtomicUsize>, fail: bool) -> PreparedRunOutcome {
        PreparedRunOutcome {
            native_report: aiperf_metrics::NativeReport::new(
                &aiperf_metrics::AccumulatorSummary::new(),
                None,
            ),
            report_facts: aiperf_metrics::ReportPairRunFacts::new(),
            provenance: BTreeMap::from([("fixture".to_owned(), "durable".to_owned())]),
            report_commit: Some(Box::new(TrackingCommit { calls, fail })),
        }
    }

    #[test]
    fn existing_report_target_never_invokes_lifecycle_commit() {
        let root = tempfile::tempdir().unwrap();
        let report_path = root.path().join("native-v2.json");
        std::fs::write(&report_path, b"existing-authority").unwrap();
        let calls = Arc::new(AtomicUsize::new(0));

        let error =
            persist_prepared_report(outcome(calls.clone(), false), provenance(), &report_path)
                .unwrap_err();

        assert_eq!(error.code, "report_target_exists");
        assert_eq!(calls.load(Ordering::SeqCst), 0);
        assert_eq!(std::fs::read(report_path).unwrap(), b"existing-authority");
    }

    #[test]
    fn failed_report_write_never_invokes_lifecycle_commit() {
        let root = tempfile::tempdir().unwrap();
        let report_path = root.path().join("missing-parent/native-v2.json");
        let calls = Arc::new(AtomicUsize::new(0));

        let error =
            persist_prepared_report(outcome(calls.clone(), false), provenance(), &report_path)
                .unwrap_err();

        assert_eq!(error.code, "reporting_failed");
        assert_eq!(calls.load(Ordering::SeqCst), 0);
        assert!(!report_path.exists());
    }

    #[test]
    fn atomic_report_write_invokes_lifecycle_commit_exactly_once() {
        let root = tempfile::tempdir().unwrap();
        let report_path = root.path().join("native-v2.json");
        let calls = Arc::new(AtomicUsize::new(0));

        let persisted =
            persist_prepared_report(outcome(calls.clone(), false), provenance(), &report_path)
                .unwrap();

        assert_eq!(persisted["fixture"], "durable");
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert!(report_path.is_file());
        let report: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&report_path).unwrap()).unwrap();
        assert_eq!(report["run"]["workload"], "evaluation");
        assert!(std::fs::read_dir(root.path()).unwrap().all(|entry| {
            !entry
                .unwrap()
                .file_name()
                .to_string_lossy()
                .ends_with(".tmp")
        }));
    }

    #[test]
    fn lifecycle_commit_failure_is_a_reporting_failure_after_durable_write() {
        let root = tempfile::tempdir().unwrap();
        let report_path = root.path().join("native-v2.json");
        let calls = Arc::new(AtomicUsize::new(0));

        let error =
            persist_prepared_report(outcome(calls.clone(), true), provenance(), &report_path)
                .unwrap_err();

        assert_eq!(error.code, "report_commit_failed");
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert!(report_path.is_file());
    }
}
