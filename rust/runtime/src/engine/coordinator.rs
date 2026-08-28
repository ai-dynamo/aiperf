// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Injected protocol-v2 composition root for one runner process.
//!
//! The stock binary and statically linked custom distributions use the same
//! coordinator. Concrete registries meet here exactly once; transport/workload
//! pair adapters receive only the frozen [`RunContext`] and their own
//! validated configuration.

use std::collections::BTreeMap;
use std::future::Future;
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
    PreparedReportCommit, PreparedRunFailure, PreparedRunOutcome, RunContext,
    validate_endpoint_profiles_v2,
};
use crate::engine::sidecar_input::SidecarInputAdapterResolver;
#[cfg(feature = "streaming")]
use crate::streaming::results::{ResultSinkState, SinkFailureReason};

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
    /// Zero for success or one for a validated run failure; the coordinator
    /// emits no other value. Protocol failures (unreadable envelope,
    /// unserializable response) never reach here and exit 2 from the
    /// `execute_mode` wrapper instead.
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
        // The stream resource is admitted before every preparation seam: no
        // source is opened, no factory prepares, no clock is read, and no lease
        // is taken until this returns. A refusal here is therefore free of
        // observable effects at the source or the endpoint. The prepared plan is
        // dropped: the constructor that consumes it is a later slice, and
        // retaining a value nothing reads would be dead state.
        #[cfg(feature = "streaming")]
        if let Err(error) = self
            .product_registry
            .validate_dataset_streams_for_run(&run, &context, &selection)
        {
            return failure_with_path(
                operation,
                self.distribution_id.clone(),
                benchmark_id,
                FailureStageV2::Validation,
                "invalid_dataset_streams",
                format!("{error:#}"),
                Some("run.resources.dataset_streams"),
                1,
            );
        }
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
                let mut run_metadata = match block_on_report_persistence(persist_prepared_report(
                    outcome,
                    report_run_metadata,
                    &report_path,
                    &run.artifact_target,
                    &run.export,
                    self.product_registry.exporters(),
                )) {
                    Ok(persisted) => {
                        // A retryable persistence failure leaves no
                        // authoritative report to report, but it also rolls
                        // nothing back: the committed generation and its
                        // durable pending-retry status are retained for the
                        // bounded supervisor.
                        #[cfg(feature = "streaming")]
                        if !matches!(persisted.state(), ResultSinkState::Complete { .. }) {
                            return terminal_failure(
                                self.distribution_id.clone(),
                                benchmark_id,
                                FailureStageV2::Reporting,
                                "report_persistence_incomplete",
                                format!(
                                    "the authoritative report is not durable; the derived sink \
                                     retained {:?}",
                                    persisted.state()
                                ),
                                1,
                            );
                        }
                        persisted.run_metadata
                    }
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

/// One completed report-persistence pass.
#[derive(Debug)]
struct PersistedReport {
    run_metadata: BTreeMap<String, String>,
    /// The uncommitted report lease, handed back when the sink owes a retry.
    ///
    /// Dropping the commit releases the report lease, so a retryable failure
    /// returns it to the caller rather than releasing it beside a report that
    /// was never written.
    retained_commit: Option<Box<dyn PreparedReportCommit>>,
    /// Reported state of the authoritative-report sink.
    #[cfg(feature = "streaming")]
    state: ResultSinkState,
}

#[cfg(feature = "streaming")]
impl PersistedReport {
    /// Return the reported state of the authoritative-report sink.
    const fn state(&self) -> ResultSinkState {
        self.state
    }
}

#[cfg(test)]
thread_local! {
    /// Ordered persistence milestones observed by the in-module ordering tests.
    static REPORT_MILESTONES: std::cell::RefCell<Option<std::rc::Rc<std::cell::RefCell<Vec<&'static str>>>>> =
        const { std::cell::RefCell::new(None) };
}

/// Record one ordered report-persistence milestone.
///
/// The contract this module owes its caller — final generation, leased
/// compaction, durable rename, commit, lease release, in that order — is only
/// observable as a sequence, so the ordering tests install a recorder here.
/// Non-test builds compile the call away.
#[cfg(test)]
fn record_milestone(name: &'static str) {
    REPORT_MILESTONES.with(|slot| {
        if let Some(events) = slot.borrow().as_ref() {
            events.borrow_mut().push(name);
        }
    });
}

#[cfg(not(test))]
#[inline]
const fn record_milestone(_name: &'static str) {}

/// Digest and length of the durable authoritative report.
///
/// The identity is read back from the renamed file rather than derived from the
/// in-memory report, so it names what a restarted process would find.
#[cfg(feature = "streaming")]
fn durable_output_identity(
    report_path: &Path,
) -> std::result::Result<(crate::streaming::identity::ContentDigest, u64), ReportPersistenceFailure>
{
    let bytes = std::fs::read(report_path).map_err(|error| ReportPersistenceFailure {
        code: "reporting_failed",
        message: format!(
            "the persisted report at {} is unreadable: {error}",
            report_path.display()
        ),
    })?;
    let length = u64::try_from(bytes.len()).map_err(|_| ReportPersistenceFailure {
        code: "reporting_failed",
        message: "the persisted report length does not fit u64".to_owned(),
    })?;
    Ok((
        crate::streaming::identity::ContentDigest::from_bytes(*blake3::hash(&bytes).as_bytes()),
        length,
    ))
}

/// Drive the report-persistence future on one fresh current-thread runtime.
///
/// Execution has already finished and torn its own reactor down by the time the
/// coordinator persists, so this is the only runtime alive; the future is
/// `!Send` and never leaves this thread.
fn block_on_report_persistence(
    future: impl Future<Output = std::result::Result<PersistedReport, ReportPersistenceFailure>>,
) -> std::result::Result<PersistedReport, ReportPersistenceFailure> {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|error| ReportPersistenceFailure {
            code: "reporting_failed",
            message: format!("report persistence runtime unavailable: {error}"),
        })?;
    tokio::task::LocalSet::new().block_on(&runtime, future)
}

/// Persist the authoritative report and close the report lease in order.
///
/// The committed final generation and its leased compaction happen before the
/// prepared outcome reaches here; this function continues the same order with
/// the durable rename, the synchronous commit hook, and the lease release the
/// hook performs. An ordinary persistence failure under a streaming retry
/// authority is recorded as a durable pending retry and returns the derived
/// sink state: nothing rolls back, the commit hook is not called, and the
/// retained generation stays exactly as the barrier committed it.
async fn persist_prepared_report(
    outcome: PreparedRunOutcome,
    report_run_metadata: ReportRunMetadata,
    report_path: &Path,
    artifact_dir: &Path,
    export: &crate::export::ExportConfig,
    exporters: &crate::export::ExporterRegistry,
) -> std::result::Result<PersistedReport, ReportPersistenceFailure> {
    if report_path.exists() {
        return Err(ReportPersistenceFailure {
            code: "report_target_exists",
            message: format!(
                "native-v2 report target already exists: {}",
                report_path.display()
            ),
        });
    }
    let mut outcome = outcome;
    #[cfg(feature = "streaming")]
    let mut report_retry = outcome.report_retry.take();
    let PreparedRunOutcome {
        native_report,
        report_facts,
        run_metadata,
        report_commit,
        ..
    } = outcome;
    tracing::info!("Processing records results...");
    let written = finalize_and_write_native_report_json(
        native_report,
        report_run_metadata,
        report_facts,
        report_path,
    )
    .map_err(|error| ReportPersistenceFailure {
        code: "reporting_failed",
        message: format!("{error:#}"),
    });

    #[cfg(feature = "streaming")]
    let (finalized, state) = match written
        .and_then(|finalized| durable_output_identity(report_path).map(|id| (finalized, id)))
    {
        Ok((finalized, (output_digest, output_length))) => (
            finalized,
            ResultSinkState::Complete {
                output_digest,
                output_length,
            },
        ),
        Err(failure) => {
            let Some(authority) = report_retry.as_mut() else {
                return Err(failure);
            };
            let state = authority
                .record_failure(SinkFailureReason::ReportPersistence)
                .await
                .map_err(|error| ReportPersistenceFailure {
                    code: "report_retry_failed",
                    message: format!(
                        "{}; recording the pending retry failed: {error}",
                        failure.message
                    ),
                })?;
            // The commit hook is neither called nor dropped here: the report
            // lease travels back to the caller with the pending-retry state.
            return Ok(PersistedReport {
                run_metadata,
                retained_commit: report_commit,
                state,
            });
        }
    };
    #[cfg(not(feature = "streaming"))]
    let finalized = written?;

    record_milestone("report_rename");
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
    // The report JSON is already written and renamed above; the outcomes are
    // observational only, so a failed sink never reaches back into the report.
    let outcomes = exporters.run_collect(&finalized, &export_dir, export);
    for outcome in outcomes.iter().filter(|outcome| !outcome.success) {
        tracing::debug!(
            exporter = %outcome.descriptor_id,
            error = outcome.error_message.as_deref().unwrap_or("unknown"),
            "exporter outcome recorded as failed"
        );
    }
    if let Some(report_commit) = report_commit {
        report_commit
            .commit()
            .map_err(|error| ReportPersistenceFailure {
                code: "report_commit_failed",
                message: format!("post-persistence report lifecycle commit failed: {error:#}"),
            })?;
    }
    Ok(PersistedReport {
        run_metadata,
        retained_commit: None,
        #[cfg(feature = "streaming")]
        state,
    })
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
    use std::cell::{Cell, RefCell};
    use std::rc::Rc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;
    use crate::engine::protocol_v2::RunDiagnosticArtifactV2;
    use crate::engine::registry::PreparedReportCommit;
    #[cfg(feature = "streaming")]
    use crate::streaming::{
        budget::{BudgetLimits, StreamingResourceBudget},
        checkpoint::{CommittedCheckpointGeneration, StreamRunIdentity},
        identity::{ContentDigest, LogicalReplayRunId},
        reliability::{
            BudgetOwnedStreamingIssueReporter, PreparedStreamingIssuePolicy, StreamingIssueClass,
            StreamingIssueComponentId, StreamingIssueDisposition, StreamingIssueScopeKind,
            StreamingIssueThresholdRule,
        },
        results::{
            PreparedStreamingReport,
            sink_status::{
                DerivedSinkStatusStore, DerivedStatusSubstrate, DurableReportRetryAuthority,
                committed_final_generation_for_test,
            },
        },
    };

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
            #[cfg(feature = "streaming")]
            report_retry: None,
        }
    }

    /// One-shot commit that records the ordered commit and lease-release steps.
    ///
    /// It mirrors the product's leased commit: the acknowledgement consumes the
    /// box and dropping it is what releases the lease.
    #[cfg(feature = "streaming")]
    #[derive(Debug)]
    struct RecordingCommit {
        calls: Rc<Cell<usize>>,
    }

    #[cfg(feature = "streaming")]
    impl PreparedReportCommit for RecordingCommit {
        fn commit(self: Box<Self>) -> Result<()> {
            self.calls.set(self.calls.get() + 1);
            record_milestone("report_commit");
            drop(self);
            Ok(())
        }
    }

    #[cfg(feature = "streaming")]
    impl Drop for RecordingCommit {
        fn drop(&mut self) {
            record_milestone("lease_release");
        }
    }

    #[cfg(feature = "streaming")]
    struct ReportPersistenceFixture {
        root: tempfile::TempDir,
        report_path: std::path::PathBuf,
        events: Rc<RefCell<Vec<&'static str>>>,
        commit_calls: Rc<Cell<usize>>,
        substrate: DerivedStatusSubstrate,
        run: StreamRunIdentity,
        final_generation: CommittedCheckpointGeneration,
        sink_id: StreamingIssueComponentId,
        outcome: RefCell<Option<PreparedRunOutcome>>,
        export: crate::export::ExportConfig,
        exporters: crate::export::ExporterRegistry,
    }

    #[cfg(feature = "streaming")]
    impl ReportPersistenceFixture {
        fn events(&self) -> Rc<RefCell<Vec<&'static str>>> {
            self.events.clone()
        }

        fn outcome(&self) -> PreparedRunOutcome {
            self.outcome
                .borrow_mut()
                .take()
                .expect("the fixture prepares exactly one outcome")
        }

        fn report_run_metadata(&self) -> ReportRunMetadata {
            run_metadata()
        }

        fn report_path(&self) -> &Path {
            &self.report_path
        }

        fn artifact_dir(&self) -> &Path {
            self.root.path()
        }

        fn export_config(&self) -> &crate::export::ExportConfig {
            &self.export
        }

        fn exporters(&self) -> &crate::export::ExporterRegistry {
            &self.exporters
        }

        fn report_commit_calls(&self) -> usize {
            self.commit_calls.get()
        }

        /// Whether a replacement process reopens the same committed generation.
        ///
        /// The store is reopened over the surviving medium, so this is the
        /// ledger-free reconstruction a restart performs, not a live handle.
        fn final_generation_is_reconstructable(&self) -> bool {
            let store = DerivedSinkStatusStore::open(
                self.run,
                self.substrate.clone(),
                test_budget(8, 4096),
            );
            futures::executor::block_on(
                store.reopen_verified_status(&self.final_generation, &self.sink_id),
            )
            .is_ok()
        }
    }

    #[cfg(feature = "streaming")]
    fn test_budget(items: usize, bytes: usize) -> StreamingResourceBudget {
        StreamingResourceBudget::new(BudgetLimits {
            max_items: items,
            max_bytes: bytes,
        })
        .expect("valid fixture budget")
    }

    #[cfg(feature = "streaming")]
    fn export_retry_policy() -> PreparedStreamingIssuePolicy {
        PreparedStreamingIssuePolicy::new([StreamingIssueThresholdRule::new(
            StreamingIssueComponentId::new("export_retryable").expect("valid rule identity"),
            StreamingIssueScopeKind::Export,
            StreamingIssueClass::Retryable,
            None,
            3,
            StreamingIssueDisposition::ExportIncomplete,
            None,
        )
        .expect("valid retryable export rule")])
        .expect("valid export policy")
    }

    /// Build the fixture in the product's order: commit the final generation,
    /// compact it under lease, then hand the prepared outcome to persistence.
    #[cfg(feature = "streaming")]
    fn build_report_persistence_fixture(relative_report: &str) -> ReportPersistenceFixture {
        let events = Rc::new(RefCell::new(Vec::new()));
        REPORT_MILESTONES.with(|slot| *slot.borrow_mut() = Some(events.clone()));
        let root = tempfile::tempdir().expect("fixture artifact root");
        let run = StreamRunIdentity::new(LogicalReplayRunId::from_bytes([0x6d; 32]));

        let final_generation = committed_final_generation_for_test(run, 7);
        record_milestone("final_generation");

        let sink_id = StreamingIssueComponentId::new("native_report").expect("valid sink identity");
        let substrate = DerivedStatusSubstrate::new();
        let commit_calls = Rc::new(Cell::new(0));
        let prepared = PreparedStreamingReport {
            native_report: crate::metrics_core::NativeReport::new(
                &crate::metrics_core::AccumulatorSummary::new(),
                None,
            ),
            report_digest: ContentDigest::from_bytes([0x6e; 32]),
            report_commit: Box::new(RecordingCommit {
                calls: commit_calls.clone(),
            }),
        };
        record_milestone("compact");

        let authority = DurableReportRetryAuthority::new(
            DerivedSinkStatusStore::open(run, substrate.clone(), test_budget(8, 4096)),
            final_generation.clone(),
            sink_id.clone(),
            BudgetOwnedStreamingIssueReporter::new(
                run,
                export_retry_policy(),
                test_budget(64, 128 * 1024),
            )
            .expect("budget-owned reporter"),
            test_budget(16, 64 * 1024),
        );
        let outcome = prepared.into_run_outcome(
            BTreeMap::from([("fixture".to_owned(), "durable".to_owned())]),
            Some(Box::new(authority)),
        );

        ReportPersistenceFixture {
            report_path: root.path().join(relative_report),
            root,
            events,
            commit_calls,
            substrate,
            run,
            final_generation,
            sink_id,
            outcome: RefCell::new(Some(outcome)),
            export: crate::export::ExportConfig::default(),
            exporters: crate::export::ExporterRegistry::with_builtin_exporters(),
        }
    }

    #[cfg(feature = "streaming")]
    fn report_persistence_fixture() -> ReportPersistenceFixture {
        build_report_persistence_fixture("native-v2.json")
    }

    /// The report target sits under a directory that does not exist, so the
    /// durable rename fails without touching the committed generation.
    #[cfg(feature = "streaming")]
    fn failing_report_persistence_fixture() -> ReportPersistenceFixture {
        build_report_persistence_fixture("missing-parent/native-v2.json")
    }

    #[tokio::test(flavor = "current_thread")]
    async fn existing_report_target_never_invokes_lifecycle_commit() {
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
        .await
        .unwrap_err();

        assert_eq!(error.code, "report_target_exists");
        assert_eq!(calls.load(Ordering::SeqCst), 0);
        assert_eq!(std::fs::read(report_path).unwrap(), b"existing-authority");
    }

    #[tokio::test(flavor = "current_thread")]
    async fn failed_report_write_never_invokes_lifecycle_commit() {
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
        .await
        .unwrap_err();

        assert_eq!(error.code, "reporting_failed");
        assert_eq!(calls.load(Ordering::SeqCst), 0);
        assert!(!report_path.exists());
    }

    #[tokio::test(flavor = "current_thread")]
    async fn atomic_report_write_invokes_lifecycle_commit_exactly_once() {
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
        .await
        .unwrap();

        assert_eq!(persisted.run_metadata["fixture"], "durable");
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

    #[tokio::test(flavor = "current_thread")]
    async fn lifecycle_commit_failure_is_a_reporting_failure_after_durable_write() {
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
        .await
        .unwrap_err();

        assert_eq!(error.code, "report_commit_failed");
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert!(report_path.is_file());
    }

    #[cfg(feature = "streaming")]
    #[tokio::test(flavor = "current_thread")]
    async fn streaming_report_persists_before_commit_lease_release() {
        let fixture = report_persistence_fixture();
        let events = fixture.events();

        persist_prepared_report(
            fixture.outcome(),
            fixture.report_run_metadata(),
            fixture.report_path(),
            fixture.artifact_dir(),
            fixture.export_config(),
            fixture.exporters(),
        )
        .await
        .unwrap();

        assert_eq!(
            events.borrow().as_slice(),
            [
                "final_generation",
                "compact",
                "report_rename",
                "report_commit",
                "lease_release"
            ],
        );
    }

    #[cfg(feature = "streaming")]
    #[tokio::test(flavor = "current_thread")]
    async fn streaming_report_failure_records_retry_and_skips_commit_hook() {
        let fixture = failing_report_persistence_fixture();

        let status = persist_prepared_report(
            fixture.outcome(),
            fixture.report_run_metadata(),
            fixture.report_path(),
            fixture.artifact_dir(),
            fixture.export_config(),
            fixture.exporters(),
        )
        .await
        .unwrap();

        assert!(matches!(
            status.state(),
            ResultSinkState::PendingRetry { .. }
        ));
        assert!(fixture.final_generation_is_reconstructable());
        assert_eq!(fixture.report_commit_calls(), 0);
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
