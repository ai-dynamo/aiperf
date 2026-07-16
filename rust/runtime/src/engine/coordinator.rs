// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Injected protocol-v2 composition root for one runner process.
//!
//! The stock binary and statically linked custom distributions use the same
//! coordinator. Concrete registries meet here exactly once; transport/workload
//! pair adapters receive only the frozen [`RunnerRunContext`] and their own
//! validated configuration.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

use crate::extensions::{AIPerfRegistry, AIPerfRegistryFactory};
use crate::metrics_core::ReportRunProvenance;
use crate::report::finalize_and_write_native_report_json;
use anyhow::{Context, Result, ensure};
use serde::Serialize;

use crate::engine::dataset_input::RunnerDatasetInputAdapterResolver;
use crate::engine::execution_factories::RunnerExecutionFactories;
use crate::engine::graph_input::RunnerGraphInputAdapterResolver;
use crate::engine::protocol::RunnerCatalog;
use crate::engine::protocol_v2::{
    DeferredCheckV2, RUNNER_PROTOCOL_V2, RunTerminalV2, RunValidationV2, RunnerDiagnosticV2,
    RunnerEnvelopeV2, RunnerFailureStageV2, RunnerOperationV2, ValidationCompletenessV2,
};
use crate::engine::redaction::redact_diagnostic;
use crate::engine::registry::{
    PreparedRunFailure, PreparedRunOutcome, RunnerRunContext, validate_endpoint_profiles_v2,
};
use crate::engine::sidecar_input::RunnerSidecarInputAdapterResolver;

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
    product_registry: Arc<AIPerfRegistry>,
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
            .field("transports", &self.product_registry.transport_descriptors())
            .field("workloads", &self.product_registry.workload_descriptors())
            .finish_non_exhaustive()
    }
}

impl RunnerV2Coordinator {
    /// Compose every startup registry exactly once for this child process.
    pub fn new(
        distribution_id: impl Into<String>,
        product_registry_factory: &dyn AIPerfRegistryFactory,
        execution_factories: RunnerExecutionFactories,
        graph_inputs: Arc<dyn RunnerGraphInputAdapterResolver>,
        dataset_inputs: Arc<dyn RunnerDatasetInputAdapterResolver>,
        sidecar_inputs: Arc<dyn RunnerSidecarInputAdapterResolver>,
    ) -> Result<Self> {
        let distribution_id = distribution_id.into();
        validate_distribution_id(&distribution_id)?;
        // One registry of record: transports, workloads, endpoints, samplers, and
        // loaders all live in the single product registry composed here.
        let product_registry = Arc::new(
            product_registry_factory
                .build()
                .context("composing AIPerf product registry")?,
        );
        Ok(Self {
            distribution_id,
            product_registry,
            execution_factories,
            graph_inputs,
            dataset_inputs,
            sidecar_inputs,
        })
    }

    /// Return the plugins.yaml-shaped catalog from this process's frozen registry.
    pub fn catalog(&self) -> RunnerCatalog {
        RunnerCatalog::from_registry(self.product_registry.as_ref())
    }

    /// Validate or execute one strict authored envelope through the frozen registries.
    pub fn handle(&self, envelope: RunnerEnvelopeV2) -> RunnerProcessResultV2 {
        let operation = envelope.operation;
        let benchmark_id = Some(envelope.run.benchmark_id.clone());
        if let Err(error) = envelope.validate_outer() {
            let message = format!("{error:#}");
            let path = message
                .strip_prefix("run.cfg.workload")
                .map(|_| "run.cfg.workload")
                .or_else(|| {
                    message
                        .strip_prefix("run.cfg.accuracy")
                        .map(|_| "run.cfg.accuracy")
                });
            return failure_with_path(
                operation,
                self.distribution_id.clone(),
                benchmark_id,
                RunnerFailureStageV2::Validation,
                "invalid_run",
                message,
                path,
                1,
            );
        }
        let run = match envelope.run.into_authored() {
            Ok(run) => run,
            Err(error) => {
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
        };

        let selection = match self.product_registry.validate_selection_for_run(&run) {
            Ok(selection) => selection,
            Err(error) => {
                return failure(
                    operation,
                    self.distribution_id.clone(),
                    benchmark_id,
                    RunnerFailureStageV2::Validation,
                    "invalid_transport_workload_selection",
                    format!("{error:#}"),
                    1,
                );
            }
        };

        let endpoint_profiles =
            match validate_endpoint_profiles_v2(&run, self.product_registry.endpoints()) {
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
        // In cellular mode every cell runs this coordinator, but sidecar telemetry
        // (GPU/DCGM, network-latency calibration, server-metrics scraping) is
        // host-level and the controller drops all but one cell's anyway
        // (`warn_dropped_sidecar_telemetry`). Running the collectors on every cell
        // is pure waste — N DCGM/Prometheus scrapes, N localhost telemetry probes.
        // So only the primary cell (`cell_id == 0`, or any non-cellular run) starts
        // the collectors; secondary cells prepare an empty sidecar set.
        let run_sidecars = crate::cellular::ModuloCellPartition::from_env()
            .map(|partition| {
                !(crate::cellular::CellPartition::cell_count(&partition) > 1
                    && crate::cellular::CellPartition::cell_id(&partition) != 0)
            })
            .unwrap_or(true);
        let authored_sidecars = if run_sidecars {
            run.sidecars.authored_inputs()
        } else {
            Vec::new()
        };
        let sidecar_inputs = match self.sidecar_inputs.prepare(&authored_sidecars) {
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
            self.distribution_id.clone(),
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
            .product_registry
            .validate_run(&run, &context, &selection)
        {
            return failure(
                operation,
                self.distribution_id.clone(),
                benchmark_id,
                RunnerFailureStageV2::Validation,
                "invalid_transport_workload_run",
                format!("{error:#}"),
                1,
            );
        }
        let transport_id = selection.transport_id().to_owned();
        let workload_id = selection.workload_id().to_owned();
        let report_provenance = match context.report_provenance(
            self.distribution_id.clone(),
            transport_id.clone(),
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
                    benchmark_id,
                    success: true,
                    completeness: ValidationCompletenessV2::Static,
                    deferred_checks: vec![DeferredCheckV2 {
                        code: "workload_preparation".to_owned(),
                        path: "run.cfg".to_owned(),
                        reason: "dataset, tokenizer, endpoint-profile references, and transport resources require execution preparation"
                            .to_owned(),
                    }],
                    errors: Vec::new(),
                }),
                exit_code: 0,
            };
        }

        let report_path = run.artifact_target.join("native-v2.json");
        let operation = match self
            .product_registry
            .prepare_with_context(&run, &context, selection)
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
                let mut provenance = match persist_prepared_report(
                    outcome,
                    report_provenance,
                    &report_path,
                    &run.artifact_target,
                    &run.export,
                    self.product_registry.exporters(),
                ) {
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
                provenance.insert("transport".into(), transport_id);
                provenance.insert("workload".into(), workload_id);
                RunnerProcessResultV2 {
                    response: RunnerResponseV2::Terminal(RunTerminalV2 {
                        protocol_version: RUNNER_PROTOCOL_V2,
                        event: "run_terminal",
                        benchmark_id,
                        success: true,
                        report_path: Some(report_path),
                        stage: None,
                        errors: Vec::new(),
                        diagnostic_artifacts: Vec::new(),
                        provenance,
                    }),
                    exit_code: 0,
                }
            }
            Err(error) => {
                if let Some(failure) = error.downcast_ref::<PreparedRunFailure>() {
                    terminal_failure_with_artifacts(
                        self.distribution_id.clone(),
                        benchmark_id,
                        failure.stage,
                        &failure.code,
                        failure.message.clone(),
                        failure.diagnostic_artifacts.clone(),
                        1,
                    )
                } else {
                    terminal_failure(
                        self.distribution_id.clone(),
                        benchmark_id,
                        RunnerFailureStageV2::Execution,
                        "execution_failed",
                        format!("{error:#}"),
                        1,
                    )
                }
            }
        }
    }

    /// Borrow the exact frozen product registry used by this process.
    pub fn product_registry(&self) -> &AIPerfRegistry {
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
    artifact_dir: &Path,
    export: &crate::export::ExportConfig,
    exporters: &crate::export::ExporterRegistry,
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
    tracing::info!("Processing records results...");
    let finalized = finalize_and_write_native_report_json(
        native_report,
        report_provenance,
        report_facts,
        report_path,
    )
    .map_err(|error| ReportPersistenceFailure {
        code: "reporting_failed",
        message: format!("{error:#}"),
    })?;
    tracing::info!(report = %report_path.display(), "Report written to: {}", report_path.display());
    // Native post-report export plane. Best-effort: the native-v2 report above is
    // the committed authority; genai-perf compat / OTLP / MLflow side outputs log
    // and never fail the run (see `crate::export`).
    //
    // `AIPERF_EXPORT_SUBDIR` (parity-proof only) redirects the native sink outputs
    // into `<artifact_dir>/<subdir>/` so they coexist with the legacy Python
    // exporter files under `<artifact_dir>/`, enabling a same-`native-v2.json`
    // byte-diff. Unset in normal runs (sinks write to the artifact root).
    let export_dir = match std::env::var("AIPERF_EXPORT_SUBDIR") {
        Ok(subdir) if !subdir.is_empty() => {
            let dir = artifact_dir.join(subdir);
            let _ = std::fs::create_dir_all(&dir);
            dir
        }
        _ => artifact_dir.to_path_buf(),
    };
    tracing::info!("Exporting all records");
    exporters.run(&finalized, &export_dir, export);
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

#[allow(clippy::too_many_arguments)]
fn failure_with_path(
    operation: RunnerOperationV2,
    distribution_id: String,
    benchmark_id: Option<String>,
    stage: RunnerFailureStageV2,
    code: &str,
    message: impl Into<String>,
    path: Option<&str>,
    exit_code: i32,
) -> RunnerProcessResultV2 {
    let message = message.into();
    if operation != RunnerOperationV2::Validate {
        return terminal_failure(
            distribution_id,
            benchmark_id,
            stage,
            code,
            message,
            exit_code,
        );
    }
    RunnerProcessResultV2 {
        response: RunnerResponseV2::Validation(RunValidationV2 {
            protocol_version: RUNNER_PROTOCOL_V2,
            event: "run_validation",
            benchmark_id,
            success: false,
            completeness: ValidationCompletenessV2::Static,
            deferred_checks: Vec::new(),
            errors: vec![RunnerDiagnosticV2 {
                code: code.to_owned(),
                message: redact_diagnostic(message),
                path: path.map(str::to_owned),
            }],
        }),
        exit_code,
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
    terminal_failure_with_artifacts(
        distribution_id,
        benchmark_id,
        stage,
        code,
        message,
        Vec::new(),
        exit_code,
    )
}

fn terminal_failure_with_artifacts(
    _distribution_id: String,
    benchmark_id: Option<String>,
    stage: RunnerFailureStageV2,
    code: &str,
    message: impl Into<String>,
    diagnostic_artifacts: Vec<crate::engine::protocol_v2::RunDiagnosticArtifactV2>,
    exit_code: i32,
) -> RunnerProcessResultV2 {
    RunnerProcessResultV2 {
        response: RunnerResponseV2::Terminal(RunTerminalV2 {
            protocol_version: RUNNER_PROTOCOL_V2,
            event: "run_terminal",
            benchmark_id,
            success: false,
            report_path: None,
            stage: Some(stage),
            errors: vec![diagnostic(code, message)],
            diagnostic_artifacts,
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
    use crate::engine::protocol_v2::RunDiagnosticArtifactV2;
    use crate::engine::registry::PreparedReportCommit;

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
            "http",
            "evaluation",
            Vec::new(),
            vec![
                crate::metrics_core::ReportEndpointProfileIdentity::new("default", "chat").unwrap(),
            ],
        )
        .unwrap()
    }

    fn outcome(calls: Arc<AtomicUsize>, fail: bool) -> PreparedRunOutcome {
        PreparedRunOutcome {
            native_report: crate::metrics_core::NativeReport::new(
                &crate::metrics_core::AccumulatorSummary::new(),
                None,
            ),
            report_facts: crate::metrics_core::ReportPairRunFacts::new(),
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

        let error = persist_prepared_report(
            outcome(calls.clone(), false),
            provenance(),
            &report_path,
            root.path(),
            &crate::export::ExportConfig::default(),
            &crate::export::ExporterRegistry::with_builtin_exporters(),
        )
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

        let error = persist_prepared_report(
            outcome(calls.clone(), false),
            provenance(),
            &report_path,
            root.path(),
            &crate::export::ExportConfig::default(),
            &crate::export::ExporterRegistry::with_builtin_exporters(),
        )
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

        let persisted = persist_prepared_report(
            outcome(calls.clone(), false),
            provenance(),
            &report_path,
            root.path(),
            &crate::export::ExportConfig::default(),
            &crate::export::ExporterRegistry::with_builtin_exporters(),
        )
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

        let error = persist_prepared_report(
            outcome(calls.clone(), true),
            provenance(),
            &report_path,
            root.path(),
            &crate::export::ExportConfig::default(),
            &crate::export::ExporterRegistry::with_builtin_exporters(),
        )
        .unwrap_err();

        assert_eq!(error.code, "report_commit_failed");
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert!(report_path.is_file());
    }

    #[test]
    fn diagnostic_failure_never_exposes_an_authoritative_report() {
        let artifact = RunDiagnosticArtifactV2 {
            kind: "archive_failure_diagnostic".to_owned(),
            relative_path: "archive-failure-diagnostic.json".into(),
            content_hash: format!("blake3:{}", "a".repeat(64)),
        };

        let result = terminal_failure_with_artifacts(
            format!("blake3:{}", "b".repeat(64)),
            Some("watch-1".to_owned()),
            RunnerFailureStageV2::Reporting,
            "archive_remote_finalization_failed",
            "remote archive unavailable",
            vec![artifact.clone()],
            1,
        );

        let RunnerResponseV2::Terminal(terminal) = result.response else {
            panic!("expected a terminal response");
        };
        assert!(!terminal.success);
        assert_eq!(terminal.report_path, None);
        assert_eq!(terminal.stage, Some(RunnerFailureStageV2::Reporting));
        assert_eq!(terminal.diagnostic_artifacts, vec![artifact]);
    }
}
