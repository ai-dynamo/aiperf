// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Injected protocol-v2 composition root for one runner process.
//!
//! The stock binary and statically linked custom distributions use the same
//! coordinator. Concrete registries meet here exactly once; transport/workload
//! pair adapters receive only the frozen [`RunContext`] and their own
//! validated configuration.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

use crate::extensions::{AIPerfRegistry, AIPerfRegistryFactory};
use crate::metrics_core::{ReportRunMetadata, ReportStats, ReportValue};
use crate::report::finalize_and_write_native_report_json;
use anyhow::{Context, Result, ensure};
use serde::Serialize;

use crate::engine::dataset_input::DatasetInputAdapterResolver;
use crate::engine::execution_factories::ExecutionFactories;
use crate::engine::graph_input::GraphInputAdapterResolver;
use crate::engine::protocol::Catalog;
use crate::engine::protocol_v2::{
    DeferredCheckV2, DiagnosticV2, EnvelopeV2, FailureStageV2, OperationV2, PROTOCOL_V2,
    RunTerminalV2, RunValidationV2, ValidationCompletenessV2,
};
use crate::engine::redaction::redact_diagnostic;
use crate::engine::registry::{
    PreparedRunFailure, PreparedRunOutcome, RunContext, validate_endpoint_profiles_v2,
};
use crate::engine::sidecar_input::SidecarInputAdapterResolver;

/// Exactly one typed response emitted for a protocol-v2 request.
#[derive(Debug, Serialize)]
#[serde(untagged)]
pub enum ResponseV2 {
    /// Response to a side-effect-free validation operation.
    Validation(RunValidationV2),
    /// Terminal response to an execution operation.
    Terminal(RunTerminalV2),
}

/// One coordinator result plus the process exit status that carries it.
#[derive(Debug)]
pub struct ProcessResultV2 {
    /// Exactly one response object for stdout JSONL.
    pub response: ResponseV2,
    /// Zero for success, one for a validated run failure, or two for protocol failure.
    pub exit_code: i32,
}

/// Frozen implementation universe for one fresh runner child.
///
/// A custom statically linked executable injects alternate registry factories
/// or graph-input adapters here. The execution coordinator and all hot-path
/// scheduling code remain unchanged.
pub struct Coordinator {
    distribution_id: String,
    product_registry: Arc<AIPerfRegistry>,
    execution_factories: ExecutionFactories,
    graph_inputs: Arc<dyn GraphInputAdapterResolver>,
    dataset_inputs: Arc<dyn DatasetInputAdapterResolver>,
    sidecar_inputs: Arc<dyn SidecarInputAdapterResolver>,
}

impl std::fmt::Debug for Coordinator {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Coordinator")
            .field("distribution_id", &self.distribution_id)
            .field("transports", &self.product_registry.transport_descriptors())
            .field("workloads", &self.product_registry.workload_descriptors())
            .finish_non_exhaustive()
    }
}

impl Coordinator {
    /// Compose every startup registry exactly once for this child process.
    pub fn new(
        distribution_id: impl Into<String>,
        product_registry_factory: &dyn AIPerfRegistryFactory,
        execution_factories: ExecutionFactories,
        graph_inputs: Arc<dyn GraphInputAdapterResolver>,
        dataset_inputs: Arc<dyn DatasetInputAdapterResolver>,
        sidecar_inputs: Arc<dyn SidecarInputAdapterResolver>,
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
    pub fn catalog(&self) -> Catalog {
        Catalog::from_registry(self.product_registry.as_ref())
    }

    /// Builds a NativeGraph model-stage context from this application's frozen seams.
    ///
    /// NativeGraph model execution has no protocol-v2 sidecars. It still receives
    /// the exact registry, execution factories, and input resolvers that this
    /// coordinator composed at process startup.
    pub fn native_graph_context(
        &self,
        endpoint_profiles: Vec<crate::engine::registry::ValidatedEndpointProfileV2>,
    ) -> Result<RunContext> {
        let sidecars = Arc::new(self.sidecar_inputs.prepare(&[])?);
        RunContext::new(
            self.distribution_id.clone(),
            self.product_registry.clone(),
            self.execution_factories.clone(),
            self.graph_inputs.clone(),
            self.dataset_inputs.clone(),
            sidecars,
            endpoint_profiles,
        )
    }

    /// Validate or execute one strict authored envelope through the frozen registries.
    pub fn handle(&self, envelope: EnvelopeV2) -> ProcessResultV2 {
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
                FailureStageV2::Validation,
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
                    FailureStageV2::Validation,
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
                    FailureStageV2::Validation,
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
                        FailureStageV2::Validation,
                        "invalid_endpoint_profiles",
                        format!("{error:#}"),
                        1,
                    );
                }
            };
        // Sidecar telemetry is host-level, so only the primary cell starts
        // collectors; the controller discards duplicate cell telemetry.
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
                    FailureStageV2::Validation,
                    "invalid_sidecars",
                    format!("{error:#}"),
                    1,
                );
            }
        };
        let context = match RunContext::new(
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
                    FailureStageV2::Validation,
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
                FailureStageV2::Validation,
                "invalid_transport_workload_run",
                format!("{error:#}"),
                1,
            );
        }
        let transport_id = selection.transport_id().to_owned();
        let workload_id = selection.workload_id().to_owned();
        let report_run_metadata = match context.report_run_metadata(
            self.distribution_id.clone(),
            transport_id.clone(),
            workload_id.clone(),
        ) {
            Ok(run_metadata) => run_metadata,
            Err(error) => {
                return failure(
                    operation,
                    self.distribution_id.clone(),
                    benchmark_id,
                    FailureStageV2::Validation,
                    "invalid_report_metadata",
                    format!("{error:#}"),
                    1,
                );
            }
        };

        if operation == OperationV2::Validate {
            return ProcessResultV2 {
                response: ResponseV2::Validation(RunValidationV2 {
                    protocol_version: PROTOCOL_V2,
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
                    FailureStageV2::Preparation,
                    "preparation_failed",
                    format!("{error:#}"),
                    1,
                );
            }
        };
        match operation.execute() {
            Ok(outcome) => {
                // A run whose every request failed is not a successful run. The
                // report is still persisted first so the per-request errors are
                // available for diagnosis, but the terminal envelope reports the
                // failure so the process exits non-zero.
                let all_requests_failed = all_requests_failed(&outcome);
                let mut run_metadata = match persist_prepared_report(
                    outcome,
                    report_run_metadata,
                    &report_path,
                    &run.artifact_target,
                    &run.export,
                    self.product_registry.exporters(),
                ) {
                    Ok(run_metadata) => run_metadata,
                    Err(error) => {
                        return terminal_failure(
                            self.distribution_id.clone(),
                            benchmark_id,
                            FailureStageV2::Reporting,
                            error.code,
                            error.message,
                            1,
                        );
                    }
                };
                run_metadata.insert("transport".into(), transport_id);
                run_metadata.insert("workload".into(), workload_id);
                if let Some(errors) = all_requests_failed {
                    return ProcessResultV2 {
                        response: ResponseV2::Terminal(RunTerminalV2 {
                            protocol_version: PROTOCOL_V2,
                            event: "run_terminal",
                            benchmark_id,
                            success: false,
                            report_path: Some(report_path),
                            stage: Some(FailureStageV2::Execution),
                            errors: vec![diagnostic(
                                "all_requests_failed",
                                format!(
                                    "All {errors} inference request(s) failed. No successful \
                                     responses were collected — check the server URL, endpoint \
                                     path, and response format. See the persisted report for \
                                     per-request error details."
                                ),
                            )],
                            diagnostic_artifacts: Vec::new(),
                            run_metadata,
                        }),
                        exit_code: 1,
                    };
                }
                ProcessResultV2 {
                    response: ResponseV2::Terminal(RunTerminalV2 {
                        protocol_version: PROTOCOL_V2,
                        event: "run_terminal",
                        benchmark_id,
                        success: true,
                        report_path: Some(report_path),
                        stage: None,
                        errors: Vec::new(),
                        diagnostic_artifacts: Vec::new(),
                        run_metadata,
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
                        FailureStageV2::Execution,
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

    /// Borrow the frozen execution factories selected for this application image.
    pub fn execution_factories(&self) -> &ExecutionFactories {
        &self.execution_factories
    }
}

/// The failed-request count when a run produced errors and zero successes.
///
/// The native metrics plane omits `request_count` entirely when nothing
/// succeeded, so an absent success counter plus a positive error counter is the
/// exact "everything failed" signature. A run that recorded no requests at all
/// (`--dry-run`, an empty schedule) has neither counter and is not a failure
/// here, and neither is a run whose every request was cancelled by policy.
/// Mirrors the python engine's `system_controller` guard.
fn all_requests_failed(outcome: &PreparedRunOutcome) -> Option<u64> {
    let counter_total = |name: &str| -> Option<f64> {
        let entry = outcome.native_report.metrics.get(name)?;
        let mut total = 0.0;
        for series in &entry.series {
            match &series.stats {
                ReportStats::Counter(counter) => match counter.total {
                    ReportValue::Finite(value) => total += value,
                    ReportValue::NonFinite => return None,
                },
                ReportStats::Scalar(scalar) => match scalar.value {
                    ReportValue::Finite(value) => total += value,
                    ReportValue::NonFinite => return None,
                },
                _ => return None,
            }
        }
        Some(total)
    };
    let successes = counter_total("request_count").unwrap_or(0.0);
    if successes >= 1.0 {
        return None;
    }
    // `error_request_count` also counts policy cancellations, which are an
    // intended outcome of a `cancellation:`-configured run rather than a
    // failure, so subtract them before deciding the run failed.
    let cancelled: usize = outcome
        .native_report
        .errors
        .iter()
        .filter(|error| error.error_type == "RequestCancellationError")
        .map(|error| error.count)
        .sum();
    let errors = counter_total("error_request_count").unwrap_or(0.0) - cancelled as f64;
    (errors >= 1.0).then_some(errors as u64)
}

#[derive(Debug)]
struct ReportPersistenceFailure {
    code: &'static str,
    message: String,
}

fn persist_prepared_report(
    outcome: PreparedRunOutcome,
    report_run_metadata: ReportRunMetadata,
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
        run_metadata,
        report_commit,
    } = outcome;
    tracing::info!("Processing records results...");
    let finalized = finalize_and_write_native_report_json(
        native_report,
        report_run_metadata,
        report_facts,
        report_path,
    )
    .map_err(|error| ReportPersistenceFailure {
        code: "reporting_failed",
        message: format!("{error:#}"),
    })?;
    tracing::info!(report = %report_path.display(), "Report written to: {}", report_path.display());
    // The native-v2 report is authoritative; optional side outputs are
    // best-effort and never fail the run.
    // `AIPERF_EXPORT_SUBDIR` redirects sink outputs for parity checks.
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
    Ok(run_metadata)
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
    operation: OperationV2,
    distribution_id: String,
    benchmark_id: Option<String>,
    stage: FailureStageV2,
    code: &str,
    message: impl Into<String>,
    exit_code: i32,
) -> ProcessResultV2 {
    if operation == OperationV2::Validate {
        ProcessResultV2 {
            response: ResponseV2::Validation(RunValidationV2 {
                protocol_version: PROTOCOL_V2,
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
    operation: OperationV2,
    distribution_id: String,
    benchmark_id: Option<String>,
    stage: FailureStageV2,
    code: &str,
    message: impl Into<String>,
    path: Option<&str>,
    exit_code: i32,
) -> ProcessResultV2 {
    let message = message.into();
    if operation != OperationV2::Validate {
        return terminal_failure(
            distribution_id,
            benchmark_id,
            stage,
            code,
            message,
            exit_code,
        );
    }
    ProcessResultV2 {
        response: ResponseV2::Validation(RunValidationV2 {
            protocol_version: PROTOCOL_V2,
            event: "run_validation",
            benchmark_id,
            success: false,
            completeness: ValidationCompletenessV2::Static,
            deferred_checks: Vec::new(),
            errors: vec![DiagnosticV2 {
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
    stage: FailureStageV2,
    code: &str,
    message: impl Into<String>,
    exit_code: i32,
) -> ProcessResultV2 {
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
    stage: FailureStageV2,
    code: &str,
    message: impl Into<String>,
    diagnostic_artifacts: Vec<crate::engine::protocol_v2::RunDiagnosticArtifactV2>,
    exit_code: i32,
) -> ProcessResultV2 {
    ProcessResultV2 {
        response: ResponseV2::Terminal(RunTerminalV2 {
            protocol_version: PROTOCOL_V2,
            event: "run_terminal",
            benchmark_id,
            success: false,
            report_path: None,
            stage: Some(stage),
            errors: vec![diagnostic(code, message)],
            diagnostic_artifacts,
            run_metadata: BTreeMap::new(),
        }),
        exit_code,
    }
}

fn diagnostic(code: &str, message: impl Into<String>) -> DiagnosticV2 {
    DiagnosticV2 {
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

    fn run_metadata() -> ReportRunMetadata {
        ReportRunMetadata::new(
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
            run_metadata: BTreeMap::from([("fixture".to_owned(), "durable".to_owned())]),
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
            run_metadata(),
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
            run_metadata(),
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
            run_metadata(),
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
            run_metadata(),
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
            FailureStageV2::Reporting,
            "archive_remote_finalization_failed",
            "remote archive unavailable",
            vec![artifact.clone()],
            1,
        );

        let ResponseV2::Terminal(terminal) = result.response else {
            panic!("expected a terminal response");
        };
        assert!(!terminal.success);
        assert_eq!(terminal.report_path, None);
        assert_eq!(terminal.stage, Some(FailureStageV2::Reporting));
        assert_eq!(terminal.diagnostic_artifacts, vec![artifact]);
    }
}
