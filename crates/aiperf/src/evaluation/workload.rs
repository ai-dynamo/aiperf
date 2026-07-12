// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Provider-neutral evaluation orchestration over Rust-owned host effects.
//!
//! The selected provider owns case semantics and aggregation. This module owns
//! finite and Rust-scheduled unit admission, bounded fair operation queues,
//! exact operation/attempt accounting, logical-route preparation, cancellation,
//! process-tree quiescence, and artifact sealing. New asset sources, occurrence
//! policies, artifact stores, and host operations enter through traits rather
//! than provider-specific branches.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::path::{Path, PathBuf};
use std::rc::Rc;

use aiperf_accuracy::{
    CanonicalJson, EvaluationArtifactSealer, EvaluationAssetRequirement, EvaluationCaseId,
    EvaluationCaseTemplateId, EvaluationEvent, EvaluationFinishCandidate, EvaluationIdentity,
    EvaluationPlan, EvaluationPlanRequest, EvaluationProvider, EvaluationQueueCredits,
    EvaluationSchedulingMode, EvaluationStage, EvaluationUnitId, EvaluationUnitOccurrence,
    EvaluationUnitOccurrenceRequest, HostCapabilityId, HostOperationDisposition,
    HostOperationEvent, HostOperationId, HostOperationTerminal, HostOperationUsage,
    HostResponseMode, ResolvedEvaluationAsset, SealedEvaluationArtifacts, SemanticAttemptId,
};
use aiperf_clock::Clock;
use aiperf_metrics::EvaluationRouteSummaryReport;
use anyhow::{Context, Result, anyhow, ensure};
use async_trait::async_trait;
use tokio::sync::mpsc;

use super::arbiter::{FairOperationArbiter, FairQueueLimits};
use super::host::{
    EvaluationRouteTable, HostExecutionDelta, HostExecutionEventSink, HostExecutionTerminal,
    HostExecutorRegistry, HostExecutorRuntime, HostOperationEnvelope, RegisteredOperationId,
};
use super::ledger::{
    HostTerminalClass, OperationLedger, OperationRecord, OperationRegistration, OperationState,
};
use super::report::{EvaluationCaseReportFacts, EvaluationReportFacts};
use super::retry::OperationCancellation;

/// Rust-side immutable asset acquisition seam.
#[async_trait(?Send)]
pub trait EvaluationAssetResolver {
    /// Resolve every requirement to a verified read-only contained binding.
    async fn resolve(
        &self,
        requirements: &[EvaluationAssetRequirement],
    ) -> Result<Vec<ResolvedEvaluationAsset>>;
}

/// Side-effect-free host capability inventory used during plan validation.
#[derive(Debug, Clone, Default)]
pub struct EvaluationHostCapabilityInventory {
    schemas: BTreeMap<HostCapabilityId, String>,
}

impl EvaluationHostCapabilityInventory {
    /// Freeze unique capability/schema pairs.
    pub fn new(capabilities: impl IntoIterator<Item = (HostCapabilityId, String)>) -> Result<Self> {
        let mut schemas = BTreeMap::new();
        for (capability, schema) in capabilities {
            ensure!(
                is_sha256(&schema),
                "host capability {capability} has an invalid schema digest"
            );
            ensure!(
                schemas.insert(capability.clone(), schema).is_none(),
                "duplicate host capability {capability}"
            );
        }
        Ok(Self { schemas })
    }

    /// Validate every required capability and exact schema fingerprint.
    pub fn validate_plan(&self, plan: &EvaluationPlan) -> Result<()> {
        for requirement in &plan.host_requirements {
            match self.schemas.get(&requirement.capability_id) {
                Some(schema) if schema == &requirement.schema_sha256 => {}
                Some(_) => {
                    return Err(anyhow!(
                        "host capability {} schema does not match the provider plan",
                        requirement.capability_id
                    ));
                }
                None if requirement.required => {
                    return Err(anyhow!(
                        "required host capability {} is not executable in this runner",
                        requirement.capability_id
                    ));
                }
                None => {}
            }
        }
        Ok(())
    }
}

/// Replaceable source of Rust-authored unit occurrences.
#[async_trait(?Send)]
pub trait EvaluationOccurrenceSource {
    /// Validate/freeze provider templates before measured issuance.
    fn bind(&mut self, identity: &EvaluationIdentity) -> Result<()>;

    /// Produce the next deterministic occurrence request, or end the schedule.
    async fn next(
        &mut self,
        clock: Rc<dyn Clock>,
    ) -> Result<Option<EvaluationUnitOccurrenceRequest>>;
}

/// Quiescence-gated artifact finalization seam.
#[async_trait(?Send)]
pub trait EvaluationArtifactFinalizer {
    /// Shut down the complete worker tree, prove quiescence, and seal artifacts.
    async fn finalize(
        &self,
        provider: &mut dyn EvaluationProvider,
        candidate: &mut EvaluationFinishCandidate,
    ) -> Result<SealedEvaluationArtifacts>;
}

/// Production no-follow artifact finalizer.
pub struct SealingEvaluationArtifactFinalizer {
    sealer: EvaluationArtifactSealer,
    staging_root: PathBuf,
    promoted_root: PathBuf,
}

impl SealingEvaluationArtifactFinalizer {
    /// Bind one sealer to its contained staging and immutable destination roots.
    pub fn new(
        sealer: EvaluationArtifactSealer,
        staging_root: impl Into<PathBuf>,
        promoted_root: impl Into<PathBuf>,
    ) -> Result<Self> {
        let staging_root = staging_root.into();
        let promoted_root = promoted_root.into();
        ensure!(
            staging_root.is_absolute() && promoted_root.is_absolute(),
            "evaluation artifact roots must be absolute"
        );
        ensure!(
            staging_root != promoted_root,
            "evaluation staging and promoted roots must differ"
        );
        Ok(Self {
            sealer,
            staging_root,
            promoted_root,
        })
    }
}

#[async_trait(?Send)]
impl EvaluationArtifactFinalizer for SealingEvaluationArtifactFinalizer {
    async fn finalize(
        &self,
        provider: &mut dyn EvaluationProvider,
        candidate: &mut EvaluationFinishCandidate,
    ) -> Result<SealedEvaluationArtifacts> {
        provider
            .shutdown()
            .await
            .map_err(provider_error)
            .context("shutting down evaluator provider tree")?;
        let proof = provider.quiescence_proof().cloned().ok_or_else(|| {
            anyhow!("evaluator shutdown returned no process-tree quiescence proof")
        })?;
        let sealed = self
            .sealer
            .seal(&self.staging_root, &self.promoted_root, candidate, &proof)
            .map_err(|error| anyhow!(error.to_string()))?;
        provider
            .mark_artifacts_sealed()
            .map_err(provider_error)
            .context("recording evaluator artifact seal")?;
        Ok(sealed)
    }
}

/// Hard host ceilings and poll batching policy for one evaluation run.
#[derive(Debug, Clone, Copy)]
pub struct EvaluationWorkloadLimits {
    /// User/runner maximum simultaneously started units.
    pub unit_concurrency: usize,
    /// Maximum provider-requested queue/resource credits.
    pub credit_ceiling: EvaluationQueueCredits,
    /// Maximum finite unit page requested from the provider.
    pub unit_page_size: usize,
    /// Maximum provider events requested per poll.
    pub event_batch_size: usize,
    /// Clock duration between empty non-blocking provider polls.
    pub idle_poll_ns: i64,
}

impl EvaluationWorkloadLimits {
    /// Validate positive limits and internally consistent credit ceilings.
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.unit_concurrency > 0,
            "evaluation unit concurrency must be positive"
        );
        self.credit_ceiling
            .validate()
            .map_err(|error| anyhow!(error.to_string()))?;
        ensure!(
            self.unit_page_size > 0 && self.event_batch_size > 0,
            "evaluation page and event batch sizes must be positive"
        );
        ensure!(
            self.idle_poll_ns >= 0,
            "evaluation idle poll duration must be non-negative"
        );
        Ok(())
    }

    fn accept_plan(&self, plan: &EvaluationPlan) -> Result<()> {
        self.validate()?;
        let requested = plan.queue_credits;
        let ceiling = self.credit_ceiling;
        ensure!(
            requested.units <= self.unit_concurrency && requested.units <= ceiling.units,
            "provider requested {} unit credits above the configured ceiling",
            requested.units
        );
        for (name, value, maximum) in [
            (
                "host_operations",
                requested.host_operations as u64,
                ceiling.host_operations as u64,
            ),
            (
                "host_operations_per_unit",
                requested.host_operations_per_unit as u64,
                ceiling.host_operations_per_unit as u64,
            ),
            (
                "stream_events",
                requested.stream_events as u64,
                ceiling.stream_events as u64,
            ),
            (
                "sandboxes",
                requested.sandboxes as u64,
                ceiling.sandboxes as u64,
            ),
            (
                "processes",
                requested.processes as u64,
                ceiling.processes as u64,
            ),
            (
                "artifacts",
                requested.artifacts as u64,
                ceiling.artifacts as u64,
            ),
            (
                "artifact_bytes",
                requested.artifact_bytes,
                ceiling.artifact_bytes,
            ),
        ] {
            ensure!(
                value <= maximum,
                "provider requested {value} {name} credits above ceiling {maximum}"
            );
        }
        Ok(())
    }
}

/// Fully drained provider result before native-report projection.
pub struct EvaluationExecutionResult {
    /// Frozen provider plan.
    pub plan: EvaluationPlan,
    /// Frozen identity returned after immutable asset binding.
    pub identity: EvaluationIdentity,
    /// Canonical provider candidate validated after complete drain.
    pub candidate: EvaluationFinishCandidate,
    /// Rust-verified immutable artifacts.
    pub sealed_artifacts: SealedEvaluationArtifacts,
    /// Exact host operation/attempt ledger in deterministic operation order.
    pub operations: Vec<OperationRecord>,
    /// Provider-safe configuration/case identities and Rust route accounting
    /// required by the generic report converter.
    pub report_facts: EvaluationReportFacts,
    /// One-shot lifecycle acknowledgement retained until the native report is
    /// atomically committed by the runner coordinator.
    pub report_commit: EvaluationReportCommit,
}

impl std::fmt::Debug for EvaluationExecutionResult {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("EvaluationExecutionResult")
            .field("plan", &self.plan)
            .field("identity", &self.identity)
            .field("candidate", &self.candidate)
            .field("sealed_artifacts", &self.sealed_artifacts)
            .field("operations", &self.operations)
            .field("report_facts", &self.report_facts)
            .field("report_commit", &self.report_commit)
            .finish()
    }
}

/// One-shot post-persistence evaluator lifecycle acknowledgement.
pub struct EvaluationReportCommit {
    provider: Option<Box<dyn EvaluationProvider>>,
}

impl std::fmt::Debug for EvaluationReportCommit {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("EvaluationReportCommit")
            .field("pending", &self.provider.is_some())
            .finish()
    }
}

impl EvaluationReportCommit {
    /// Mark the evaluator lifecycle committed after the native report rename succeeds.
    pub fn commit(mut self) -> Result<()> {
        let mut provider = self
            .provider
            .take()
            .ok_or_else(|| anyhow!("evaluation report commit was already consumed"))?;
        provider
            .mark_report_committed()
            .map_err(provider_error)
            .context("committing evaluator report lifecycle")
    }
}

/// Provider-neutral evaluation workload.
pub struct EvaluationWorkload {
    provider: Box<dyn EvaluationProvider>,
    plan_request: EvaluationPlanRequest,
    routes: EvaluationRouteTable,
    host_executors: HostExecutorRegistry,
    host_runtime: HostExecutorRuntime,
    asset_resolver: Rc<dyn EvaluationAssetResolver>,
    host_capabilities: EvaluationHostCapabilityInventory,
    public_score_projection_schemas: BTreeMap<String, String>,
    occurrence_source: Option<Box<dyn EvaluationOccurrenceSource>>,
    finalizer: Rc<dyn EvaluationArtifactFinalizer>,
    limits: EvaluationWorkloadLimits,
    cancellation: OperationCancellation,
}

impl EvaluationWorkload {
    /// Compose one already-attested provider with every Rust-owned dependency.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        provider: Box<dyn EvaluationProvider>,
        plan_request: EvaluationPlanRequest,
        routes: EvaluationRouteTable,
        host_executors: HostExecutorRegistry,
        host_runtime: HostExecutorRuntime,
        asset_resolver: Rc<dyn EvaluationAssetResolver>,
        host_capabilities: EvaluationHostCapabilityInventory,
        public_score_projection_schemas: BTreeMap<String, String>,
        occurrence_source: Option<Box<dyn EvaluationOccurrenceSource>>,
        finalizer: Rc<dyn EvaluationArtifactFinalizer>,
        limits: EvaluationWorkloadLimits,
        cancellation: OperationCancellation,
    ) -> Result<Self> {
        limits.validate()?;
        for (name, schema) in &public_score_projection_schemas {
            ensure!(
                !name.trim().is_empty() && is_sha256(schema),
                "public score projection {name:?} has an invalid schema fingerprint"
            );
        }
        Ok(Self {
            provider,
            plan_request,
            routes,
            host_executors,
            host_runtime,
            asset_resolver,
            host_capabilities,
            public_score_projection_schemas,
            occurrence_source,
            finalizer,
            limits,
            cancellation,
        })
    }

    /// Plan, bind, execute, drain, finalize, quiesce, and seal one evaluation.
    pub async fn execute(mut self) -> Result<EvaluationExecutionResult> {
        let plan = self
            .provider
            .plan(&self.plan_request)
            .await
            .map_err(provider_error)
            .context("planning evaluator session")?;
        plan.validate()
            .map_err(|error| anyhow!(error.to_string()))?;
        self.limits.accept_plan(&plan)?;
        self.host_capabilities.validate_plan(&plan)?;
        validate_logical_services(&plan, &self.routes, &self.host_executors)?;

        let assets = self
            .asset_resolver
            .resolve(&plan.assets)
            .await
            .context("resolving immutable evaluator assets")?;
        validate_resolved_assets(&plan.assets, &assets)?;
        let identity = self
            .provider
            .bind_assets(&assets)
            .await
            .map_err(provider_error)
            .context("binding evaluator assets")?;
        identity
            .validate()
            .map_err(|error| anyhow!(error.to_string()))?;

        let (mut pending_units, mut occurrence_rx) = match plan.scheduling_mode {
            EvaluationSchedulingMode::Finite => {
                ensure!(
                    self.occurrence_source.is_none(),
                    "finite evaluator plan cannot consume a Rust occurrence source"
                );
                (
                    collect_finite_units(
                        self.provider.as_mut(),
                        plan.finite_unit_count.expect("validated finite count"),
                        self.limits.unit_page_size,
                    )
                    .await?,
                    None,
                )
            }
            EvaluationSchedulingMode::RustOccurrences => {
                let mut source = self.occurrence_source.take().ok_or_else(|| {
                    anyhow!("Rust-occurrence evaluator plan has no registered occurrence source")
                })?;
                source.bind(&identity)?;
                (
                    VecDeque::new(),
                    Some(spawn_occurrence_source(
                        source,
                        self.host_runtime
                            .require_scheduled()
                            .context("Rust occurrences require a scheduled runtime")?
                            .clock(),
                        plan.queue_credits.units,
                    )),
                )
            }
        };

        let mut state = ExecutionState::new(&plan)?;
        for unit in &pending_units {
            state.register_unit(unit)?;
        }
        let stream_capacity = plan.queue_credits.stream_events;
        let (host_tx, mut host_rx) = mpsc::channel(stream_capacity);
        let mut occurrence_done = plan.scheduling_mode == EvaluationSchedulingMode::Finite;
        let clock = self
            .host_runtime
            .require_scheduled()
            .context("evaluation workload requires a scheduled runtime")?
            .clock();

        loop {
            let mut progressed = false;
            if self.cancellation.is_cancelled() && !state.run_cancellation_started {
                state.run_cancellation_started = true;
                let ids = state.active_units.iter().cloned().collect::<Vec<_>>();
                if !ids.is_empty() {
                    self.provider
                        .cancel_units(&ids)
                        .await
                        .map_err(provider_error)
                        .context("cancelling evaluator units")?;
                }
                for active in state.active_operations.values() {
                    active.cancellation.cancel();
                }
                progressed = true;
            }

            if let Some(receiver) = occurrence_rx.as_mut() {
                loop {
                    match receiver.try_recv() {
                        Ok(OccurrenceMessage::Request(request)) => {
                            let units = self
                                .provider
                                .instantiate_units(&[request])
                                .await
                                .map_err(provider_error)
                                .context("instantiating evaluator occurrence")?;
                            ensure!(
                                units.len() == 1,
                                "provider returned {} units for one occurrence request",
                                units.len()
                            );
                            let unit = units.into_iter().next().expect("length checked");
                            state.register_unit(&unit)?;
                            pending_units.push_back(unit);
                            progressed = true;
                        }
                        Ok(OccurrenceMessage::Done) => {
                            occurrence_done = true;
                            occurrence_rx = None;
                            progressed = true;
                            break;
                        }
                        Ok(OccurrenceMessage::Failed(message)) => {
                            return Err(anyhow!("evaluation occurrence source failed: {message}"));
                        }
                        Err(mpsc::error::TryRecvError::Empty) => break,
                        Err(mpsc::error::TryRecvError::Disconnected) => {
                            return Err(anyhow!(
                                "evaluation occurrence source disconnected before completion"
                            ));
                        }
                    }
                }
            }

            if !state.run_cancellation_started {
                let available = plan
                    .queue_credits
                    .units
                    .saturating_sub(state.active_units.len());
                let ids = pending_units
                    .iter()
                    .take(available)
                    .map(|unit| unit.unit_id.clone())
                    .collect::<Vec<_>>();
                if !ids.is_empty() {
                    self.provider
                        .start_units(&ids)
                        .await
                        .map_err(provider_error)
                        .context("starting evaluator units")?;
                    for id in ids {
                        let unit = pending_units.pop_front().expect("selected pending unit");
                        ensure!(unit.unit_id == id, "pending evaluator unit order changed");
                        state.active_units.insert(id);
                    }
                    progressed = true;
                }
            }

            while let Ok(message) = host_rx.try_recv() {
                self.process_host_message(&mut state, message).await?;
                progressed = true;
            }

            let mut batch = self
                .provider
                .poll_events(self.limits.event_batch_size, 0)
                .await
                .map_err(provider_error)
                .context("polling evaluator events")?;
            let drained = batch.drained;
            if !batch.events.is_empty() {
                progressed = true;
            }
            for sequenced in batch.events.drain(..) {
                self.process_provider_event(&plan, &mut state, sequenced.event)
                    .await?;
            }

            while state.active_operations.len() < plan.queue_credits.host_operations {
                let Some((_unit_id, queued)) = state.arbiter.pop() else {
                    break;
                };
                self.start_host_operation(&mut state, queued, host_tx.clone())?;
                progressed = true;
            }

            if drained {
                ensure!(
                    occurrence_done,
                    "provider drained before Rust occurrence source ended"
                );
                ensure!(
                    pending_units.is_empty(),
                    "provider drained with pending units"
                );
                ensure!(
                    state.active_units.is_empty(),
                    "provider drained with active units"
                );
                ensure!(
                    state.arbiter.is_empty(),
                    "provider drained with queued host operations"
                );
                ensure!(
                    state.active_operations.is_empty(),
                    "provider drained with active host operations"
                );
                break;
            }
            if !progressed {
                clock.clone().sleep(self.limits.idle_poll_ns).await;
            }
        }

        state.arbiter.validate()?;
        state.ledger.validate_drained()?;
        let mut candidate = self
            .provider
            .finalize_candidate()
            .await
            .map_err(provider_error)
            .context("finalizing evaluator result candidate")?;
        candidate
            .validate()
            .map_err(|error| anyhow!(error.to_string()))?;
        ensure!(
            candidate.identity == identity,
            "evaluator identity drifted at finalization"
        );
        let sealed_artifacts = self
            .finalizer
            .finalize(self.provider.as_mut(), &mut candidate)
            .await?;
        let report_facts = state.build_report_facts(
            &identity,
            &self.routes,
            self.plan_request.provider_config.value().clone(),
            self.public_score_projection_schemas,
        )?;
        let operations = state.ledger.operations().cloned().collect();
        let report_commit = EvaluationReportCommit {
            provider: Some(self.provider),
        };
        Ok(EvaluationExecutionResult {
            plan,
            identity,
            candidate,
            sealed_artifacts,
            operations,
            report_facts,
            report_commit,
        })
    }

    async fn process_provider_event(
        &mut self,
        plan: &EvaluationPlan,
        state: &mut ExecutionState,
        event: EvaluationEvent,
    ) -> Result<()> {
        match event {
            EvaluationEvent::HostOperationRequested { request } => {
                ensure!(
                    request.context.session_id == self.plan_request.session_id,
                    "host operation referenced a different evaluation session"
                );
                ensure!(
                    state.active_units.contains(&request.context.unit_id),
                    "host operation referenced an inactive unit"
                );
                ensure!(
                    state
                        .unit_cases
                        .get(&request.context.unit_id)
                        .is_some_and(|cases| cases.contains(&request.context.case_id)),
                    "host operation referenced a case outside its unit"
                );
                ensure!(
                    state.arbiter.len() + state.active_operations.len()
                        < plan.queue_credits.host_operations,
                    "provider exceeded accepted global host-operation credits"
                );
                let envelope = operation_envelope(plan, request.as_ref(), &self.routes)?;
                let prepared =
                    self.host_executors
                        .prepare(&envelope, &self.routes, &self.host_runtime)?;
                state.ledger.register(OperationRegistration {
                    operation_id: envelope.operation_id.clone(),
                    unit_id: envelope.unit_id.clone(),
                    case_id: envelope.case_id.clone(),
                    semantic_attempt_id: envelope.semantic_attempt_id.clone(),
                    logical_call_id: envelope.logical_call_id.clone(),
                    service_id: envelope.service_id.clone(),
                    semantic_operation_id: envelope.semantic_operation_id.to_string(),
                    replay_safe_after_output: false,
                })?;
                state
                    .arbiter
                    .push(
                        request.context.unit_id.clone(),
                        QueuedOperation {
                            envelope,
                            executor: prepared,
                            operation_id: request.operation_id.clone(),
                            semantic_attempt_id: request.context.semantic_attempt_id.clone(),
                        },
                    )
                    .map_err(|rejection| {
                        anyhow!("provider exceeded accepted fair queue credits: {rejection:?}")
                    })?;
            }
            EvaluationEvent::HostOperationCancelRequested { request } => {
                let operation_id = request.operation_id.as_str();
                let (record_state, record_attempt, record_unit) = {
                    let record = state.ledger.operation(operation_id)?;
                    (
                        record.state,
                        record.registration.semantic_attempt_id.clone(),
                        record.registration.unit_id.clone(),
                    )
                };
                ensure!(
                    record_attempt == request.semantic_attempt_id.as_str(),
                    "operation cancellation changed semantic attempt"
                );
                match record_state {
                    OperationState::Terminal => {
                        self.provider
                            .submit_host_events(&[HostOperationEvent::CancellationAcknowledged {
                                operation_id: request.operation_id,
                                semantic_attempt_id: request.semantic_attempt_id,
                                already_terminal: true,
                            }])
                            .await
                            .map_err(provider_error)?;
                    }
                    OperationState::Queued => {
                        state.ledger.request_cancel(operation_id)?;
                        let unit_id = EvaluationUnitId::new(record_unit)
                            .map_err(|error| anyhow!(error.to_string()))?;
                        let queued = state
                            .arbiter
                            .remove_where(&unit_id, |queued| {
                                queued.operation_id == request.operation_id
                            })
                            .ok_or_else(|| anyhow!("queued cancellation lost its operation"))?;
                        state
                            .ledger
                            .finish_operation(operation_id, HostTerminalClass::Cancelled)?;
                        state
                            .operation_usage
                            .insert(operation_id.to_string(), HostOperationUsage::default());
                        self.submit_terminal(
                            queued.operation_id,
                            queued.semantic_attempt_id,
                            cancelled_terminal_payload("queued", &request.reason)?,
                            false,
                        )
                        .await?;
                    }
                    OperationState::Admitted
                    | OperationState::Dispatching
                    | OperationState::Streaming
                    | OperationState::Cancelling => {
                        state.ledger.request_cancel(operation_id)?;
                        if let Some(active) = state.active_operations.get(operation_id) {
                            active.cancellation.cancel();
                        }
                    }
                }
            }
            EvaluationEvent::CaseTerminal { outcome } => {
                let case_id = outcome.case_id.clone();
                ensure!(
                    state.terminal_cases.insert(case_id.clone()),
                    "provider emitted a duplicate case terminal"
                );
                let unit_id = state
                    .case_units
                    .get(&case_id)
                    .cloned()
                    .ok_or_else(|| anyhow!("provider terminal referenced an unknown case"))?;
                if state.unit_cases[&unit_id]
                    .iter()
                    .all(|case| state.terminal_cases.contains(case))
                {
                    state.active_units.remove(&unit_id);
                }
            }
            EvaluationEvent::Progress { .. } | EvaluationEvent::Diagnostic { .. } => {}
        }
        Ok(())
    }

    fn start_host_operation(
        &self,
        state: &mut ExecutionState,
        queued: QueuedOperation,
        host_tx: mpsc::Sender<RuntimeHostMessage>,
    ) -> Result<()> {
        let operation_id = queued.operation_id.as_str().to_string();
        state.ledger.admit(&operation_id)?;
        state
            .ledger
            .start_attempt(&operation_id, format!("{operation_id}:transport:0"))?;
        let cancellation = OperationCancellation::default();
        let active = ActiveOperation {
            cancellation: cancellation.clone(),
            semantic_attempt_id: queued.semantic_attempt_id.clone(),
            semantic_operation_id: queued.envelope.semantic_operation_id.clone(),
        };
        ensure!(
            state
                .active_operations
                .insert(operation_id.clone(), active)
                .is_none(),
            "duplicate active evaluator operation"
        );
        let sink = ChannelHostEventSink {
            operation_id: queued.operation_id.clone(),
            sender: host_tx.clone(),
        };
        tokio::task::spawn_local(async move {
            let result = queued
                .executor
                .execute(&queued.envelope, &sink, cancellation)
                .await;
            let message = RuntimeHostMessage::Terminal {
                operation_id: queued.operation_id,
                semantic_attempt_id: queued.semantic_attempt_id,
                result: result.map_err(|error| error.to_string()),
            };
            let _ = host_tx.send(message).await;
        });
        Ok(())
    }

    async fn process_host_message(
        &mut self,
        state: &mut ExecutionState,
        message: RuntimeHostMessage,
    ) -> Result<()> {
        match message {
            RuntimeHostMessage::Delta {
                operation_id,
                delta,
            } => {
                let active = state
                    .active_operations
                    .get(operation_id.as_str())
                    .ok_or_else(|| anyhow!("stream delta referenced an inactive operation"))?;
                self.host_executors
                    .validate_stream(&active.semantic_operation_id, &delta.payload)?;
                state.ledger.observe_output(
                    operation_id.as_str(),
                    &format!("{}:transport:0", operation_id.as_str()),
                )?;
                let stream_sequence = u64::try_from(delta.ordinal)
                    .map_err(|_| anyhow!("stream sequence exceeds u64"))?;
                self.provider
                    .submit_host_events(&[HostOperationEvent::StreamDelta {
                        operation_id,
                        stream_sequence,
                        delta: CanonicalJson::new(delta.payload)
                            .map_err(|error| anyhow!(error.to_string()))?,
                    }])
                    .await
                    .map_err(provider_error)?;
            }
            RuntimeHostMessage::Terminal {
                operation_id,
                semantic_attempt_id,
                result,
            } => {
                let active = state
                    .active_operations
                    .remove(operation_id.as_str())
                    .ok_or_else(|| anyhow!("host terminal referenced an inactive operation"))?;
                ensure!(
                    active.semantic_attempt_id == semantic_attempt_id,
                    "host terminal changed semantic attempt"
                );
                let terminal = match result {
                    Ok(terminal) => terminal,
                    Err(_message) => HostExecutionTerminal {
                        class: HostTerminalClass::Failed,
                        payload: serde_json::json!({"error_kind":"executor_error"}),
                        usage: HostOperationUsage::default(),
                        retryable: false,
                    },
                };
                if terminal.class == HostTerminalClass::Completed {
                    self.host_executors
                        .validate_response(&active.semantic_operation_id, &terminal.payload)?;
                }
                let attempt_id = format!("{}:transport:0", operation_id.as_str());
                let observed_output = state
                    .ledger
                    .operation(operation_id.as_str())?
                    .attempts
                    .last()
                    .is_some_and(|attempt| attempt.output_observed);
                state
                    .ledger
                    .finish_attempt(operation_id.as_str(), &attempt_id, terminal.class)?;
                state
                    .ledger
                    .finish_operation(operation_id.as_str(), terminal.class)?;
                state
                    .operation_usage
                    .insert(operation_id.to_string(), terminal.usage);
                self.submit_terminal(operation_id, semantic_attempt_id, terminal, observed_output)
                    .await?;
            }
        }
        Ok(())
    }

    async fn submit_terminal(
        &mut self,
        operation_id: HostOperationId,
        semantic_attempt_id: SemanticAttemptId,
        terminal: HostExecutionTerminal,
        observed_output: bool,
    ) -> Result<()> {
        let (disposition, result, error) = match terminal.class {
            HostTerminalClass::Completed => (
                HostOperationDisposition::Completed,
                Some(
                    CanonicalJson::new(terminal.payload)
                        .map_err(|error| anyhow!(error.to_string()))?,
                ),
                None,
            ),
            HostTerminalClass::Cancelled => (
                HostOperationDisposition::Cancelled,
                None,
                Some(
                    aiperf_accuracy::EvaluationError::new(
                        EvaluationStage::new("host_operation")
                            .map_err(|error| anyhow!(error.to_string()))?,
                        "cancelled",
                        false,
                        "Rust cancelled the evaluator host operation",
                    )
                    .map_err(|error| anyhow!(error.to_string()))?,
                ),
            ),
            HostTerminalClass::Failed | HostTerminalClass::Rejected => (
                HostOperationDisposition::InfrastructureError,
                None,
                Some(
                    aiperf_accuracy::EvaluationError::new(
                        EvaluationStage::new("host_operation")
                            .map_err(|error| anyhow!(error.to_string()))?,
                        if terminal.class == HostTerminalClass::Rejected {
                            "rejected"
                        } else {
                            "executor_error"
                        },
                        terminal.retryable,
                        "Rust host executor did not complete the operation",
                    )
                    .map_err(|error| anyhow!(error.to_string()))?,
                ),
            ),
        };
        self.provider
            .submit_host_events(&[HostOperationEvent::Terminal {
                terminal: HostOperationTerminal {
                    operation_id,
                    semantic_attempt_id,
                    disposition,
                    result,
                    error,
                    usage: terminal.usage,
                    observed_output,
                },
            }])
            .await
            .map_err(provider_error)?;
        Ok(())
    }
}

struct ExecutionState {
    arbiter: FairOperationArbiter<EvaluationUnitId, QueuedOperation>,
    ledger: OperationLedger,
    unit_cases: BTreeMap<EvaluationUnitId, BTreeSet<EvaluationCaseId>>,
    case_units: BTreeMap<EvaluationCaseId, EvaluationUnitId>,
    case_templates: BTreeMap<EvaluationCaseId, EvaluationCaseTemplateId>,
    active_units: BTreeSet<EvaluationUnitId>,
    terminal_cases: BTreeSet<EvaluationCaseId>,
    active_operations: BTreeMap<String, ActiveOperation>,
    operation_usage: BTreeMap<String, HostOperationUsage>,
    run_cancellation_started: bool,
}

impl ExecutionState {
    fn new(plan: &EvaluationPlan) -> Result<Self> {
        Ok(Self {
            arbiter: FairOperationArbiter::new(FairQueueLimits::new(
                plan.queue_credits.host_operations,
                plan.queue_credits.host_operations_per_unit,
            )?),
            ledger: OperationLedger::default(),
            unit_cases: BTreeMap::new(),
            case_units: BTreeMap::new(),
            case_templates: BTreeMap::new(),
            active_units: BTreeSet::new(),
            terminal_cases: BTreeSet::new(),
            active_operations: BTreeMap::new(),
            operation_usage: BTreeMap::new(),
            run_cancellation_started: false,
        })
    }

    fn register_unit(&mut self, unit: &EvaluationUnitOccurrence) -> Result<()> {
        ensure!(!unit.cases.is_empty(), "evaluation unit contains no cases");
        let mut cases = BTreeSet::new();
        for case in &unit.cases {
            ensure!(cases.insert(case.case_id.clone()), "unit duplicated a case");
            ensure!(
                self.case_units
                    .insert(case.case_id.clone(), unit.unit_id.clone())
                    .is_none(),
                "evaluation case occurrence appeared in multiple units"
            );
            ensure!(
                self.case_templates
                    .insert(case.case_id.clone(), case.template_id.clone())
                    .is_none(),
                "evaluation case template identity appeared more than once"
            );
        }
        ensure!(
            self.unit_cases
                .insert(unit.unit_id.clone(), cases)
                .is_none(),
            "duplicate evaluation unit occurrence"
        );
        Ok(())
    }

    fn build_report_facts(
        &self,
        identity: &EvaluationIdentity,
        routes: &EvaluationRouteTable,
        safe_config: serde_json::Value,
        public_score_projection_schemas: BTreeMap<String, String>,
    ) -> Result<EvaluationReportFacts> {
        let templates = identity
            .case_templates
            .iter()
            .map(|template| (&template.template_id, template))
            .collect::<BTreeMap<_, _>>();
        let cases = self
            .case_templates
            .iter()
            .map(|(case_id, template_id)| {
                let template = templates.get(template_id).ok_or_else(|| {
                    anyhow!("case {case_id} referenced unknown frozen template {template_id}")
                })?;
                Ok((
                    case_id.clone(),
                    EvaluationCaseReportFacts {
                        template_id: template_id.clone(),
                        task: template.task.clone(),
                        source: template.source.clone(),
                    },
                ))
            })
            .collect::<Result<BTreeMap<_, _>>>()?;

        let mut route_summaries = routes
            .routes()
            .map(|route| {
                (
                    route.service_id.clone(),
                    EvaluationRouteSummaryReport::default(),
                )
            })
            .collect::<BTreeMap<_, _>>();
        for operation in self.ledger.operations() {
            let summary = route_summaries
                .get_mut(&operation.registration.service_id)
                .ok_or_else(|| anyhow!("operation ledger referenced an unknown route"))?;
            let first = summary.logical_operations == 0;
            summary.logical_operations += 1;
            summary.transport_attempts += operation.attempts.len();
            summary.retries += operation.attempts.len().saturating_sub(1);
            match operation.terminal {
                Some(HostTerminalClass::Completed) => summary.completed += 1,
                Some(HostTerminalClass::Cancelled) => summary.cancelled += 1,
                Some(HostTerminalClass::Failed | HostTerminalClass::Rejected) => {
                    summary.failed += 1;
                }
                None => return Err(anyhow!("reporting encountered a non-terminal operation")),
            }
            let usage = self
                .operation_usage
                .get(&operation.registration.operation_id)
                .copied()
                .ok_or_else(|| anyhow!("terminal operation omitted Rust usage accounting"))?;
            accumulate_optional(&mut summary.prompt_tokens, usage.prompt_tokens, first);
            accumulate_optional(
                &mut summary.completion_tokens,
                usage.completion_tokens,
                first,
            );
            accumulate_optional(&mut summary.reasoning_tokens, usage.reasoning_tokens, first);
            accumulate_optional(&mut summary.cached_tokens, usage.cached_tokens, first);
        }
        Ok(EvaluationReportFacts {
            safe_config,
            cases,
            public_score_projection_schemas,
            route_summaries,
        })
    }
}

fn accumulate_optional(total: &mut Option<u64>, value: Option<u64>, first: bool) {
    if first {
        *total = value;
    } else {
        *total = total
            .zip(value)
            .and_then(|(left, right)| left.checked_add(right));
    }
}

struct QueuedOperation {
    envelope: HostOperationEnvelope,
    executor: Rc<dyn super::host::HostOperationExecutor>,
    operation_id: HostOperationId,
    semantic_attempt_id: SemanticAttemptId,
}

struct ActiveOperation {
    cancellation: OperationCancellation,
    semantic_attempt_id: SemanticAttemptId,
    semantic_operation_id: RegisteredOperationId,
}

enum RuntimeHostMessage {
    Delta {
        operation_id: HostOperationId,
        delta: HostExecutionDelta,
    },
    Terminal {
        operation_id: HostOperationId,
        semantic_attempt_id: SemanticAttemptId,
        result: std::result::Result<HostExecutionTerminal, String>,
    },
}

struct ChannelHostEventSink {
    operation_id: HostOperationId,
    sender: mpsc::Sender<RuntimeHostMessage>,
}

#[async_trait(?Send)]
impl HostExecutionEventSink for ChannelHostEventSink {
    async fn publish(&self, delta: HostExecutionDelta) -> Result<()> {
        self.sender
            .send(RuntimeHostMessage::Delta {
                operation_id: self.operation_id.clone(),
                delta,
            })
            .await
            .map_err(|_| anyhow!("evaluation host event channel closed"))
    }
}

enum OccurrenceMessage {
    Request(EvaluationUnitOccurrenceRequest),
    Done,
    Failed(String),
}

fn spawn_occurrence_source(
    mut source: Box<dyn EvaluationOccurrenceSource>,
    clock: Rc<dyn Clock>,
    capacity: usize,
) -> mpsc::Receiver<OccurrenceMessage> {
    let (sender, receiver) = mpsc::channel(capacity);
    tokio::task::spawn_local(async move {
        loop {
            match source.next(clock.clone()).await {
                Ok(Some(request)) => {
                    if sender
                        .send(OccurrenceMessage::Request(request))
                        .await
                        .is_err()
                    {
                        return;
                    }
                }
                Ok(None) => {
                    let _ = sender.send(OccurrenceMessage::Done).await;
                    return;
                }
                Err(error) => {
                    let _ = sender
                        .send(OccurrenceMessage::Failed(error.to_string()))
                        .await;
                    return;
                }
            }
        }
    });
    receiver
}

async fn collect_finite_units(
    provider: &mut dyn EvaluationProvider,
    expected: usize,
    page_size: usize,
) -> Result<VecDeque<EvaluationUnitOccurrence>> {
    let mut offset = 0usize;
    let mut units = VecDeque::with_capacity(expected);
    loop {
        let page = provider
            .next_units(offset, page_size)
            .await
            .map_err(provider_error)
            .context("paging finite evaluator units")?;
        ensure!(
            page.next_offset == offset.saturating_add(page.items.len()),
            "finite evaluator unit page changed its next offset"
        );
        ensure!(
            !page.items.is_empty() || page.done,
            "finite evaluator returned an empty non-terminal page"
        );
        offset = page.next_offset;
        units.extend(page.items);
        ensure!(
            units.len() <= expected,
            "finite evaluator returned more units than planned"
        );
        if page.done {
            break;
        }
    }
    ensure!(
        units.len() == expected,
        "finite evaluator planned {expected} units but returned {}",
        units.len()
    );
    Ok(units)
}

fn validate_logical_services(
    plan: &EvaluationPlan,
    routes: &EvaluationRouteTable,
    executors: &HostExecutorRegistry,
) -> Result<()> {
    for requirement in &plan.logical_services {
        let route = routes.resolve(requirement.service_id.as_str())?;
        ensure!(
            route.purpose == requirement.purpose.as_str(),
            "logical service {} purpose differs from the provider plan",
            requirement.service_id
        );
        for operation in &requirement.operations {
            let operation_id = RegisteredOperationId::new(operation.as_str())?;
            let descriptor = executors.factory(&operation_id)?.descriptor();
            let missing = descriptor
                .endpoint_capabilities
                .difference(&route.endpoint_capabilities)
                .collect::<Vec<_>>();
            ensure!(
                missing.is_empty(),
                "logical service {} cannot execute operation {}",
                requirement.service_id,
                operation
            );
        }
    }
    Ok(())
}

fn operation_envelope(
    plan: &EvaluationPlan,
    request: &aiperf_accuracy::HostOperationRequest,
    routes: &EvaluationRouteTable,
) -> Result<HostOperationEnvelope> {
    let service = plan
        .logical_services
        .iter()
        .find(|service| service.service_id == request.service_id)
        .ok_or_else(|| anyhow!("host operation requested an undeclared logical service"))?;
    ensure!(
        service.operations.contains(&request.semantic_operation_id),
        "logical service did not declare the requested semantic operation"
    );
    ensure!(
        service.purpose == request.purpose,
        "host operation changed its planned purpose"
    );
    let route = routes.resolve(request.service_id.as_str())?;
    ensure!(
        route.purpose == request.purpose.as_str(),
        "host operation purpose does not match its Rust route"
    );
    let (payload, restricted) = if let Some(restricted) = &request.restricted_payload {
        ensure!(
            service.allows_restricted_payload,
            "logical service forbids restricted evaluator payloads"
        );
        (restricted.body.value().clone(), true)
    } else {
        (request.payload.value().clone(), false)
    };
    Ok(HostOperationEnvelope {
        operation_id: request.operation_id.to_string(),
        unit_id: request.context.unit_id.to_string(),
        case_id: request.context.case_id.to_string(),
        semantic_attempt_id: request.context.semantic_attempt_id.to_string(),
        logical_call_id: request.context.logical_call_id.to_string(),
        service_id: request.service_id.to_string(),
        semantic_operation_id: RegisteredOperationId::new(request.semantic_operation_id.as_str())?,
        purpose: request.purpose.to_string(),
        payload,
        restricted,
        stream: request.response_mode == HostResponseMode::Streaming,
    })
}

fn validate_resolved_assets(
    requirements: &[EvaluationAssetRequirement],
    resolved: &[ResolvedEvaluationAsset],
) -> Result<()> {
    ensure!(
        requirements.len() == resolved.len(),
        "asset resolver returned a different number of bindings"
    );
    for (requirement, binding) in requirements.iter().zip(resolved) {
        ensure!(
            binding.asset_id == requirement.asset_id
                && binding.content_sha256 == requirement.content_sha256
                && binding.immutable_revision == requirement.immutable_revision
                && binding.media_type == requirement.media_type,
            "asset resolver changed immutable identity for {:?}",
            requirement.asset_id
        );
        ensure!(
            Path::new(&binding.contained_path).is_absolute(),
            "resolved evaluator asset path must be absolute inside the contained view"
        );
    }
    Ok(())
}

fn cancelled_terminal_payload(stage: &str, reason: &str) -> Result<HostExecutionTerminal> {
    ensure!(!stage.is_empty(), "cancellation stage must not be empty");
    let _ = reason;
    Ok(HostExecutionTerminal {
        class: HostTerminalClass::Cancelled,
        payload: serde_json::json!({"status":"cancelled"}),
        usage: HostOperationUsage::default(),
        retryable: false,
    })
}

fn provider_error(error: aiperf_accuracy::EvaluationProviderError) -> anyhow::Error {
    anyhow!(error.to_string())
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}
