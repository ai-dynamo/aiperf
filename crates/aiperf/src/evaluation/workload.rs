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
    EvaluationPhaseId, EvaluationPlan, EvaluationPlanRequest, EvaluationProvider,
    EvaluationQueueCredits, EvaluationSchedulingMode, EvaluationStage, EvaluationUnitId,
    EvaluationUnitOccurrence, EvaluationUnitOccurrenceRequest, EvaluationUnitTemplateId,
    HostCapabilityId, HostOperationDisposition, HostOperationEvent, HostOperationId,
    HostOperationTerminal, HostOperationUsage, HostResponseMode, PublicScoreProjectionPolicy,
    ResolvedEvaluationAsset, SealedEvaluationArtifacts, SemanticAttemptId,
};
use aiperf_clock::Clock;
use aiperf_metrics::EvaluationRouteSummaryReport;
use aiperf_timing::IntervalGenerator;
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

/// One Clock-paced phase for Rust-authored evaluator occurrences.
pub struct EvaluationOccurrencePhase {
    phase_id: EvaluationPhaseId,
    intervals: Box<dyn IntervalGenerator>,
    max_occurrences: Option<u64>,
    duration_ns: Option<i64>,
}

impl EvaluationOccurrencePhase {
    /// Build one phase. Either bound may be absent; external cancellation then
    /// remains the stop authority for an otherwise unbounded phase.
    pub fn new(
        phase_id: EvaluationPhaseId,
        intervals: Box<dyn IntervalGenerator>,
        max_occurrences: Option<u64>,
        duration_ns: Option<i64>,
    ) -> Result<Self> {
        ensure!(
            max_occurrences != Some(0),
            "evaluation phase occurrence bound must be positive"
        );
        ensure!(
            duration_ns.is_none_or(|duration| duration > 0),
            "evaluation phase duration must be positive"
        );
        Ok(Self {
            phase_id,
            intervals,
            max_occurrences,
            duration_ns,
        })
    }
}

struct ActiveOccurrencePhase {
    definition: EvaluationOccurrencePhase,
    start_ns: Option<i64>,
    last_target_ns: Option<i64>,
    issued: u64,
}

/// Normal Clock-paced occurrence source cycling frozen provider templates.
pub struct ClockedEvaluationOccurrenceSource {
    phases: VecDeque<ActiveOccurrencePhase>,
    templates: Vec<EvaluationUnitTemplateId>,
    issue_ordinal: u64,
    cancellation: OperationCancellation,
    bound: bool,
}

impl ClockedEvaluationOccurrenceSource {
    /// Compose ordered phases and one shared run cancellation latch.
    pub fn new(
        phases: impl IntoIterator<Item = EvaluationOccurrencePhase>,
        cancellation: OperationCancellation,
    ) -> Result<Self> {
        let phases = phases
            .into_iter()
            .map(|definition| ActiveOccurrencePhase {
                definition,
                start_ns: None,
                last_target_ns: None,
                issued: 0,
            })
            .collect::<VecDeque<_>>();
        ensure!(
            !phases.is_empty(),
            "evaluation occurrence source requires at least one phase"
        );
        Ok(Self {
            phases,
            templates: Vec::new(),
            issue_ordinal: 0,
            cancellation,
            bound: false,
        })
    }
}

#[async_trait(?Send)]
impl EvaluationOccurrenceSource for ClockedEvaluationOccurrenceSource {
    fn bind(&mut self, identity: &EvaluationIdentity) -> Result<()> {
        ensure!(!self.bound, "evaluation occurrence source was bound twice");
        ensure!(
            !identity.unit_templates.is_empty(),
            "Rust-occurrence evaluator identity contains no unit templates"
        );
        self.templates = identity
            .unit_templates
            .iter()
            .map(|template| template.unit_template_id.clone())
            .collect();
        self.bound = true;
        Ok(())
    }

    async fn next(
        &mut self,
        clock: Rc<dyn Clock>,
    ) -> Result<Option<EvaluationUnitOccurrenceRequest>> {
        ensure!(self.bound, "evaluation occurrence source is not bound");
        loop {
            if self.cancellation.is_cancelled() {
                return Ok(None);
            }
            let Some(phase) = self.phases.front_mut() else {
                return Ok(None);
            };
            let start_ns = *phase.start_ns.get_or_insert_with(|| clock.now_ns());
            if phase
                .definition
                .max_occurrences
                .is_some_and(|maximum| phase.issued >= maximum)
            {
                self.phases.pop_front();
                continue;
            }
            let target_ns = if phase.issued == 0 {
                start_ns
            } else {
                phase
                    .last_target_ns
                    .unwrap_or(start_ns)
                    .saturating_add(phase.definition.intervals.next_interval_ns())
            };
            if phase
                .definition
                .duration_ns
                .is_some_and(|duration| target_ns.saturating_sub(start_ns) >= duration)
            {
                self.phases.pop_front();
                continue;
            }
            let wait_ns = target_ns.saturating_sub(clock.now_ns()).max(0);
            if wait_ns > 0 {
                let sleep = clock.clone().sleep(wait_ns);
                let cancelled = self.cancellation.cancelled();
                tokio::pin!(sleep);
                tokio::pin!(cancelled);
                tokio::select! {
                    _ = &mut sleep => {}
                    _ = &mut cancelled => return Ok(None),
                }
            }
            if self.cancellation.is_cancelled() {
                return Ok(None);
            }
            let template_count = u64::try_from(self.templates.len())
                .map_err(|_| anyhow!("evaluation template count exceeds u64"))?;
            let template_index = usize::try_from(self.issue_ordinal % template_count)
                .expect("modulo template count fits usize");
            let request = EvaluationUnitOccurrenceRequest {
                unit_template_id: self.templates[template_index].clone(),
                phase_id: phase.definition.phase_id.clone(),
                issue_ordinal: self.issue_ordinal,
                cycle_index: self.issue_ordinal / template_count,
            };
            phase.issued = phase
                .issued
                .checked_add(1)
                .ok_or_else(|| anyhow!("evaluation phase issue count overflow"))?;
            phase.last_target_ns = Some(target_ns);
            self.issue_ordinal = self
                .issue_ordinal
                .checked_add(1)
                .ok_or_else(|| anyhow!("evaluation issue ordinal overflow"))?;
            return Ok(Some(request));
        }
    }
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
    public_score_projection_policy: PublicScoreProjectionPolicy,
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
        public_score_projection_policy: PublicScoreProjectionPolicy,
        occurrence_source: Option<Box<dyn EvaluationOccurrenceSource>>,
        finalizer: Rc<dyn EvaluationArtifactFinalizer>,
        limits: EvaluationWorkloadLimits,
        cancellation: OperationCancellation,
    ) -> Result<Self> {
        limits.validate()?;
        Ok(Self {
            provider,
            plan_request,
            routes,
            host_executors,
            host_runtime,
            asset_resolver,
            host_capabilities,
            public_score_projection_policy,
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
                occurrence_rx = None;
                occurrence_done = true;
                progressed = true;
            }

            if !state.run_cancellation_started
                && let Some(receiver) = occurrence_rx.as_mut()
            {
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

            let available = plan
                .queue_credits
                .units
                .saturating_sub(state.active_units.len());
            if available > 0 {
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

            if state.run_cancellation_started {
                let ids = state
                    .active_units
                    .difference(&state.cancel_requested_units)
                    .cloned()
                    .collect::<Vec<_>>();
                if !ids.is_empty() {
                    self.provider
                        .cancel_units(&ids)
                        .await
                        .map_err(provider_error)
                        .context("cancelling evaluator units")?;
                    state.cancel_requested_units.extend(ids);
                    progressed = true;
                }
                for active in state.active_operations.values() {
                    progressed |= active.cancellation.cancel();
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
                self.process_provider_event(&plan, &mut state, sequenced.event, &clock)
                    .await?;
            }

            let expired = state
                .operation_deadlines
                .iter()
                .filter_map(|(operation_id, deadline_ns)| {
                    (*deadline_ns <= clock.now_ns()).then_some(operation_id.clone())
                })
                .collect::<Vec<_>>();
            for operation_id in expired {
                state.operation_deadlines.remove(&operation_id);
                self.cancel_host_operation(&mut state, &operation_id, "deadline")
                    .await?;
                progressed = true;
            }

            if state.run_cancellation_started {
                let queued = state.arbiter.drain();
                for (_unit_id, queued) in queued {
                    let operation_id = queued.operation_id.as_str().to_string();
                    state.operation_deadlines.remove(&operation_id);
                    state.ledger.request_cancel(&operation_id)?;
                    state
                        .ledger
                        .finish_operation(&operation_id, HostTerminalClass::Cancelled)?;
                    state
                        .operation_usage
                        .insert(operation_id, HostOperationUsage::default());
                    self.submit_terminal(
                        queued.operation_id,
                        queued.semantic_attempt_id,
                        cancelled_terminal_payload("queued", "run_cancelled")?,
                        false,
                    )
                    .await?;
                    progressed = true;
                }
                for active in state.active_operations.values() {
                    progressed |= active.cancellation.cancel();
                }
            }

            while !state.run_cancellation_started
                && state.active_operations.len() < plan.queue_credits.host_operations
            {
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
                let wait_ns = state
                    .operation_deadlines
                    .values()
                    .map(|deadline_ns| deadline_ns.saturating_sub(clock.now_ns()).max(0))
                    .min()
                    .map_or(self.limits.idle_poll_ns, |deadline_wait_ns| {
                        deadline_wait_ns.min(self.limits.idle_poll_ns)
                    });
                clock.clone().sleep(wait_ns).await;
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
            self.public_score_projection_policy,
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
        clock: &Rc<dyn Clock>,
    ) -> Result<()> {
        match event {
            EvaluationEvent::HostOperationRequested { request } => {
                request
                    .validate()
                    .map_err(|error| anyhow!(error.to_string()))?;
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
                    idempotency_key: request.idempotency_key.clone(),
                    service_id: envelope.service_id.clone(),
                    semantic_operation_id: envelope.semantic_operation_id.to_string(),
                    replay_safe_after_output: false,
                })?;
                if let Some(deadline_ms) = request.deadline_ms {
                    let duration_ns = i64::try_from(deadline_ms)
                        .ok()
                        .and_then(|milliseconds| milliseconds.checked_mul(1_000_000))
                        .ok_or_else(|| {
                            anyhow!("host operation deadline exceeds i64 nanoseconds")
                        })?;
                    let deadline_ns = clock
                        .now_ns()
                        .checked_add(duration_ns)
                        .ok_or_else(|| anyhow!("host operation deadline overflow"))?;
                    ensure!(
                        state
                            .operation_deadlines
                            .insert(request.operation_id.to_string(), deadline_ns)
                            .is_none(),
                        "duplicate host operation deadline identity"
                    );
                }
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
                        state.operation_deadlines.remove(operation_id);
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
                        state.operation_deadlines.remove(operation_id);
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

    async fn cancel_host_operation(
        &mut self,
        state: &mut ExecutionState,
        operation_id: &str,
        reason: &str,
    ) -> Result<()> {
        let (record_state, record_unit) = {
            let record = state.ledger.operation(operation_id)?;
            (record.state, record.registration.unit_id.clone())
        };
        match record_state {
            OperationState::Terminal | OperationState::Cancelling => {}
            OperationState::Queued => {
                state.ledger.request_cancel(operation_id)?;
                let unit_id = EvaluationUnitId::new(record_unit)
                    .map_err(|error| anyhow!(error.to_string()))?;
                let queued = state
                    .arbiter
                    .remove_where(&unit_id, |queued| {
                        queued.operation_id.as_str() == operation_id
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
                    cancelled_terminal_payload("queued", reason)?,
                    false,
                )
                .await?;
            }
            OperationState::Admitted | OperationState::Dispatching | OperationState::Streaming => {
                let active = state
                    .active_operations
                    .get(operation_id)
                    .ok_or_else(|| anyhow!("active cancellation lost its operation"))?;
                active.cancellation.cancel();
            }
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
                        transport_attempts: Vec::new(),
                    },
                };
                state.operation_deadlines.remove(operation_id.as_str());
                if terminal.class == HostTerminalClass::Completed {
                    self.host_executors
                        .validate_response(&active.semantic_operation_id, &terminal.payload)?;
                }
                let observed_output = terminal
                    .transport_attempts
                    .iter()
                    .any(|attempt| attempt.output_observed);
                let attempt_count = terminal.transport_attempts.len();
                for (index, attempt) in terminal.transport_attempts.iter().enumerate() {
                    ensure!(
                        attempt.ordinal == index,
                        "host executor returned non-contiguous transport attempt ordinals"
                    );
                    let expected_id = format!("{}:transport:{index}", operation_id.as_str());
                    ensure!(
                        attempt.attempt_id == expected_id,
                        "host executor changed Rust transport attempt identity"
                    );
                    state
                        .ledger
                        .start_attempt(operation_id.as_str(), attempt.attempt_id.clone())?;
                    if attempt.output_observed {
                        state
                            .ledger
                            .observe_output(operation_id.as_str(), &attempt.attempt_id)?;
                    }
                    let replay_safe = state.ledger.finish_attempt(
                        operation_id.as_str(),
                        &attempt.attempt_id,
                        attempt.terminal,
                    )?;
                    ensure!(
                        index + 1 == attempt_count || replay_safe,
                        "host executor retried after externally observed output"
                    );
                }
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
    operation_deadlines: BTreeMap<String, i64>,
    run_cancellation_started: bool,
    cancel_requested_units: BTreeSet<EvaluationUnitId>,
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
            operation_deadlines: BTreeMap::new(),
            run_cancellation_started: false,
            cancel_requested_units: BTreeSet::new(),
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
        public_score_projection_policy: PublicScoreProjectionPolicy,
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
            public_score_projection_policy,
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
        transport_attempts: Vec::new(),
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

#[cfg(test)]
mod tests {
    use std::cell::RefCell;

    use aiperf_accuracy::{
        AggregateMetric, AggregationPolicy, ArtifactRef, ArtifactVisibility, CaseOutcome,
        CaseOutcomeKind, CompletedCaseOutcome, EvaluationArtifactId,
        EvaluationArtifactManifestEntry, EvaluationCaseOccurrenceDescriptor,
        EvaluationCaseTemplateDescriptor, EvaluationDistributionId, EvaluationEventBatch,
        EvaluationExecutionGranularity, EvaluationHostIdentity, EvaluationIdentityComponent,
        EvaluationLifecycleState, EvaluationProviderError, EvaluationProviderId,
        EvaluationUnitPage, EvaluationUnitTemplateDescriptor, EvaluationWorkerIdentity, FiniteF64,
        HostCallContext, HostOperationRequest, IsolationQuiescenceProof, LogicalCallId,
        LogicalServiceId, LogicalServiceRequirement, OperationPurpose, ProviderScore,
        SemanticAttemptId, SemanticOperationId, SequencedEvaluationEvent,
    };
    use aiperf_clock::SimClock;
    use aiperf_graph::runtime::drive_sim;
    use aiperf_metrics::InferenceDimensions;
    use aiperf_timing::StopConfig;
    use aiperf_timing::intervals::Constant;
    use async_trait::async_trait;
    use loadgen_core::sink::RequestObserver;
    use serde_json::json;

    use super::*;
    use crate::evaluation::host::{
        EvaluationRoute, HostExecutionEventSink, HostExecutorRegistryBuilder,
        HostOperationDescriptor, HostOperationExecutor, HostOperationExecutorFactory,
        HostOperationFamily, HostOperationSchemaValidator,
    };
    use crate::multiturn::TurnToSend;
    use crate::scheduled::{ScheduledRuntime, TurnDispatchOutcome, TurnDispatcher};

    fn occurrence_identity() -> EvaluationIdentity {
        let case_template = EvaluationCaseTemplateId::new("case-template").unwrap();
        EvaluationIdentity {
            canonical_json_codec: aiperf_accuracy::CANONICAL_JSON_CODEC.into(),
            worker: EvaluationWorkerIdentity {
                evaluator_protocol: 2,
                provider_id: EvaluationProviderId::new("fixture").unwrap(),
                distribution_id: EvaluationDistributionId::new("fixture_dist").unwrap(),
                package: "fixture".into(),
                package_version: "1".into(),
                provider_source_sha256: "a".repeat(64),
                worker_source_sha256: "b".repeat(64),
                dependency_lock_sha256: "c".repeat(64),
                python_version: "3.12".into(),
                launch_nonce: "n".repeat(32),
                oci_digest: None,
                operations: [
                    "plan_session",
                    "bind_assets",
                    "next_units",
                    "instantiate_units",
                    "start_units",
                    "poll_events",
                    "submit_host_events",
                    "cancel_units",
                    "finalize_session",
                    "shutdown",
                ]
                .into_iter()
                .map(str::to_string)
                .collect(),
            },
            config_schema_sha256: "d".repeat(64),
            resolved_config_sha256: "e".repeat(64),
            dataset: EvaluationIdentityComponent {
                name: "dataset".into(),
                version: "1".into(),
                source_sha256: "f".repeat(64),
            },
            components: Vec::new(),
            ordered_manifest_sha256: "1".repeat(64),
            case_templates: vec![EvaluationCaseTemplateDescriptor {
                template_id: case_template.clone(),
                task: "fixture".into(),
                source: "fixture_source".into(),
            }],
            unit_templates: ["unit-a", "unit-b"]
                .into_iter()
                .map(|id| EvaluationUnitTemplateDescriptor {
                    unit_template_id: EvaluationUnitTemplateId::new(id).unwrap(),
                    case_template_ids: vec![case_template.clone()],
                    granularity: EvaluationExecutionGranularity::Case,
                    scheduling_class: "paced".into(),
                })
                .collect(),
            policies: CanonicalJson::new(json!({"schedule":"fixture"})).unwrap(),
            host: EvaluationHostIdentity {
                runner_sha256: "2".repeat(64),
                capability_inventory_sha256: "3".repeat(64),
                schema_inventory_sha256: "4".repeat(64),
                isolation_proof_sha256: "5".repeat(64),
            },
            route_map_sha256: "6".repeat(64),
            prepared_endpoints_sha256: "7".repeat(64),
            sandbox_sha256: None,
        }
    }

    #[test]
    fn clocked_occurrences_cycle_templates_with_exact_phase_issue_and_time() {
        let clock = Rc::new(SimClock::new());
        let cancellation = OperationCancellation::default();
        let phase = EvaluationOccurrencePhase::new(
            EvaluationPhaseId::new("profiling").unwrap(),
            Box::new(Constant::new(100_000_000.0)),
            Some(3),
            None,
        )
        .unwrap();
        let mut source = ClockedEvaluationOccurrenceSource::new([phase], cancellation).unwrap();
        source.bind(&occurrence_identity()).unwrap();
        let requests = Rc::new(RefCell::new(Vec::new()));
        let requests_for_run = requests.clone();
        let clock_for_run = clock.clone();
        let outcome = drive_sim(clock, move |_handle| async move {
            while let Some(request) = source.next(clock_for_run.clone()).await.unwrap() {
                requests_for_run
                    .borrow_mut()
                    .push((request, clock_for_run.now_ns()));
            }
        });
        assert!(!outcome.deadlocked);
        let requests = requests.borrow();
        assert_eq!(requests.len(), 3);
        assert_eq!(requests[0].0.unit_template_id.as_str(), "unit-a");
        assert_eq!(requests[1].0.unit_template_id.as_str(), "unit-b");
        assert_eq!(requests[2].0.unit_template_id.as_str(), "unit-a");
        assert_eq!(requests[2].0.cycle_index, 1);
        assert_eq!(requests[0].1, 0);
        assert_eq!(requests[1].1, 10);
        assert_eq!(requests[2].1, 20);
    }

    #[test]
    fn occurrence_cancellation_interrupts_clock_sleep() {
        let clock = Rc::new(SimClock::new());
        let cancellation = OperationCancellation::default();
        let phase = EvaluationOccurrencePhase::new(
            EvaluationPhaseId::new("profiling").unwrap(),
            Box::new(Constant::new(1.0)),
            None,
            None,
        )
        .unwrap();
        let mut source =
            ClockedEvaluationOccurrenceSource::new([phase], cancellation.clone()).unwrap();
        source.bind(&occurrence_identity()).unwrap();
        let result = Rc::new(RefCell::new(None));
        let result_for_run = result.clone();
        let clock_for_run = clock.clone();
        let outcome = drive_sim(clock, move |_handle| async move {
            assert!(source.next(clock_for_run.clone()).await.unwrap().is_some());
            cancellation.cancel();
            *result_for_run.borrow_mut() = Some(source.next(clock_for_run).await.unwrap());
        });
        assert!(!outcome.deadlocked);
        assert!(result.borrow().as_ref().is_some_and(Option::is_none));
    }

    #[derive(Default)]
    struct ProviderProofState {
        submitted: Vec<HostOperationEvent>,
        cancelled_units: Vec<EvaluationUnitId>,
        shutdown: bool,
        sealed: bool,
        committed: bool,
    }

    struct ProofProvider {
        identity: EvaluationIdentity,
        plan: EvaluationPlan,
        unit: EvaluationUnitOccurrence,
        candidate: EvaluationFinishCandidate,
        session_id: aiperf_accuracy::EvaluationSessionId,
        events: VecDeque<SequencedEvaluationEvent>,
        next_sequence: u64,
        page_returned: bool,
        terminal_submitted: bool,
        operation_deadline_ms: Option<u64>,
        state: Rc<RefCell<ProviderProofState>>,
    }

    impl ProofProvider {
        fn enqueue(&mut self, event: EvaluationEvent) {
            let sequence = self.next_sequence;
            self.next_sequence += 1;
            self.events.push_back(SequencedEvaluationEvent {
                sequence,
                idempotency_key: format!("event-{sequence}"),
                event,
            });
        }
    }

    #[async_trait(?Send)]
    impl EvaluationProvider for ProofProvider {
        fn identity(&self) -> &EvaluationWorkerIdentity {
            &self.identity.worker
        }

        fn lifecycle_state(&self) -> EvaluationLifecycleState {
            if self.state.borrow().committed {
                EvaluationLifecycleState::ReportCommitted
            } else if self.state.borrow().sealed {
                EvaluationLifecycleState::ArtifactsSealed
            } else if self.state.borrow().shutdown {
                EvaluationLifecycleState::WorkerExited
            } else {
                EvaluationLifecycleState::Running
            }
        }

        fn quiescence_proof(&self) -> Option<&IsolationQuiescenceProof> {
            None
        }

        async fn plan(
            &mut self,
            _request: &EvaluationPlanRequest,
        ) -> std::result::Result<EvaluationPlan, EvaluationProviderError> {
            Ok(self.plan.clone())
        }

        async fn bind_assets(
            &mut self,
            assets: &[ResolvedEvaluationAsset],
        ) -> std::result::Result<EvaluationIdentity, EvaluationProviderError> {
            if !assets.is_empty() {
                return Err(EvaluationProviderError::Protocol(
                    "fixture received unexpected assets".into(),
                ));
            }
            Ok(self.identity.clone())
        }

        async fn next_units(
            &mut self,
            offset: usize,
            _limit: usize,
        ) -> std::result::Result<EvaluationUnitPage, EvaluationProviderError> {
            if offset == 0 && !self.page_returned {
                self.page_returned = true;
                Ok(EvaluationUnitPage {
                    items: vec![self.unit.clone()],
                    next_offset: 1,
                    done: true,
                })
            } else {
                Ok(EvaluationUnitPage {
                    items: Vec::new(),
                    next_offset: offset,
                    done: true,
                })
            }
        }

        async fn instantiate_units(
            &mut self,
            _requests: &[EvaluationUnitOccurrenceRequest],
        ) -> std::result::Result<Vec<EvaluationUnitOccurrence>, EvaluationProviderError> {
            Err(EvaluationProviderError::Protocol(
                "finite fixture cannot instantiate occurrences".into(),
            ))
        }

        async fn start_units(
            &mut self,
            ids: &[EvaluationUnitId],
        ) -> std::result::Result<(), EvaluationProviderError> {
            if ids != [self.unit.unit_id.clone()] {
                return Err(EvaluationProviderError::Protocol(
                    "fixture start IDs changed".into(),
                ));
            }
            self.enqueue(EvaluationEvent::HostOperationRequested {
                request: Box::new(HostOperationRequest {
                    operation_id: HostOperationId::new("operation-1").unwrap(),
                    context: HostCallContext {
                        session_id: self.session_id.clone(),
                        unit_id: self.unit.unit_id.clone(),
                        case_id: self.unit.cases[0].case_id.clone(),
                        semantic_attempt_id: SemanticAttemptId::new("attempt-1").unwrap(),
                        logical_call_id: LogicalCallId::new("call-1").unwrap(),
                    },
                    service_id: LogicalServiceId::new("primary").unwrap(),
                    purpose: OperationPurpose::new("primary").unwrap(),
                    semantic_operation_id: SemanticOperationId::new("model.generate").unwrap(),
                    payload: CanonicalJson::new(json!({"prompt":"fixture"})).unwrap(),
                    restricted_payload: None,
                    response_mode: HostResponseMode::Streaming,
                    deadline_ms: self.operation_deadline_ms,
                    idempotency_key: "operation-1-key".into(),
                }),
            });
            Ok(())
        }

        async fn poll_events(
            &mut self,
            limit: usize,
            _wait_ms: u64,
        ) -> std::result::Result<EvaluationEventBatch, EvaluationProviderError> {
            let mut events = Vec::new();
            while events.len() < limit
                && let Some(event) = self.events.pop_front()
            {
                events.push(event);
            }
            Ok(EvaluationEventBatch {
                events,
                next_sequence: self.next_sequence,
                drained: self.terminal_submitted && self.events.is_empty(),
                remaining_credits: self.plan.queue_credits,
            })
        }

        async fn submit_host_events(
            &mut self,
            events: &[HostOperationEvent],
        ) -> std::result::Result<(), EvaluationProviderError> {
            self.state.borrow_mut().submitted.extend_from_slice(events);
            if events
                .iter()
                .any(|event| matches!(event, HostOperationEvent::Terminal { .. }))
            {
                self.terminal_submitted = true;
                self.enqueue(EvaluationEvent::CaseTerminal {
                    outcome: Box::new(self.candidate.outcomes[0].clone()),
                });
            }
            Ok(())
        }

        async fn cancel_units(
            &mut self,
            ids: &[EvaluationUnitId],
        ) -> std::result::Result<(), EvaluationProviderError> {
            self.state
                .borrow_mut()
                .cancelled_units
                .extend_from_slice(ids);
            Ok(())
        }

        async fn finalize_candidate(
            &mut self,
        ) -> std::result::Result<EvaluationFinishCandidate, EvaluationProviderError> {
            Ok(self.candidate.clone())
        }

        async fn shutdown(&mut self) -> std::result::Result<(), EvaluationProviderError> {
            self.state.borrow_mut().shutdown = true;
            Ok(())
        }

        fn mark_artifacts_sealed(&mut self) -> std::result::Result<(), EvaluationProviderError> {
            if !self.state.borrow().shutdown {
                return Err(EvaluationProviderError::Lifecycle(
                    "fixture seal preceded shutdown".into(),
                ));
            }
            self.state.borrow_mut().sealed = true;
            Ok(())
        }

        fn mark_report_committed(&mut self) -> std::result::Result<(), EvaluationProviderError> {
            if !self.state.borrow().sealed {
                return Err(EvaluationProviderError::Lifecycle(
                    "fixture commit preceded seal".into(),
                ));
            }
            self.state.borrow_mut().committed = true;
            Ok(())
        }
    }

    struct EmptyAssetResolver;

    #[async_trait(?Send)]
    impl EvaluationAssetResolver for EmptyAssetResolver {
        async fn resolve(
            &self,
            requirements: &[EvaluationAssetRequirement],
        ) -> Result<Vec<ResolvedEvaluationAsset>> {
            ensure!(requirements.is_empty(), "fixture expected no assets");
            Ok(Vec::new())
        }
    }

    struct ProofArtifactFinalizer;

    #[async_trait(?Send)]
    impl EvaluationArtifactFinalizer for ProofArtifactFinalizer {
        async fn finalize(
            &self,
            provider: &mut dyn EvaluationProvider,
            candidate: &mut EvaluationFinishCandidate,
        ) -> Result<SealedEvaluationArtifacts> {
            provider.shutdown().await.map_err(provider_error)?;
            provider.mark_artifacts_sealed().map_err(provider_error)?;
            let artifact = &candidate.artifacts[0];
            Ok(SealedEvaluationArtifacts {
                root: "/tmp/evaluation-proof-sealed".into(),
                entries: vec![aiperf_accuracy::SealedEvaluationArtifact {
                    artifact_id: artifact.artifact_id.clone(),
                    path: artifact.path.clone(),
                    media_type: artifact.media_type.clone(),
                    visibility: artifact.visibility,
                    size_bytes: artifact.size_bytes,
                    artifact_content_sha256: artifact.artifact_content_sha256.clone(),
                    public_projection_schema_sha256: None,
                }],
                provider_bundle_sha256: artifact.artifact_content_sha256.clone(),
                quiescence_proof_sha256: "f".repeat(64),
            })
        }
    }

    struct ObjectSchema;

    impl HostOperationSchemaValidator for ObjectSchema {
        fn validate_request(&self, payload: &serde_json::Value) -> Result<()> {
            ensure!(payload.is_object(), "fixture request must be an object");
            Ok(())
        }

        fn validate_stream(&self, payload: &serde_json::Value) -> Result<()> {
            ensure!(payload.is_object(), "fixture delta must be an object");
            Ok(())
        }

        fn validate_response(&self, payload: &serde_json::Value) -> Result<()> {
            ensure!(payload.is_object(), "fixture terminal must be an object");
            Ok(())
        }
    }

    struct ProofHostExecutor;

    #[async_trait(?Send)]
    impl HostOperationExecutor for ProofHostExecutor {
        async fn execute(
            &self,
            _operation: &HostOperationEnvelope,
            events: &dyn HostExecutionEventSink,
            _cancellation: OperationCancellation,
        ) -> Result<HostExecutionTerminal> {
            events
                .publish(HostExecutionDelta {
                    ordinal: 0,
                    payload: json!({
                        "choice_index":0,
                        "delta":{"role":"assistant","content":"ok"}
                    }),
                })
                .await?;
            Ok(HostExecutionTerminal {
                class: HostTerminalClass::Completed,
                payload: json!({
                    "choices":[{
                        "message":{"role":"assistant","content":"ok"},
                        "stop_reason":"stop"
                    }],
                    "usage":{"prompt_tokens":3,"completion_tokens":1}
                }),
                usage: HostOperationUsage {
                    prompt_tokens: Some(3),
                    completion_tokens: Some(1),
                    reasoning_tokens: None,
                    cached_tokens: None,
                },
                retryable: false,
            })
        }
    }

    struct ProofHostFactory {
        descriptor: HostOperationDescriptor,
    }

    impl HostOperationExecutorFactory for ProofHostFactory {
        fn descriptor(&self) -> &HostOperationDescriptor {
            &self.descriptor
        }

        fn validator(&self) -> &dyn HostOperationSchemaValidator {
            &ObjectSchema
        }

        fn prepare(
            &self,
            _runtime: &HostExecutorRuntime,
            _route: &EvaluationRoute,
        ) -> Result<Rc<dyn HostOperationExecutor>> {
            Ok(Rc::new(ProofHostExecutor))
        }
    }

    struct UnusedDispatcher;

    #[async_trait(?Send)]
    impl TurnDispatcher for UnusedDispatcher {
        fn inference_dimensions(&self, _turn: &TurnToSend) -> InferenceDimensions {
            InferenceDimensions::default()
        }

        async fn dispatch_turn(
            &self,
            _turn: TurnToSend,
            _observer: &dyn RequestObserver,
            _on_first_token: &dyn Fn(i64),
        ) -> Result<TurnDispatchOutcome> {
            Err(anyhow!("fixture dispatcher must not be called"))
        }
    }

    fn provider_proof_fixture(
        state: Rc<RefCell<ProviderProofState>>,
    ) -> (ProofProvider, EvaluationPlanRequest) {
        let mut identity = occurrence_identity();
        identity.unit_templates.truncate(1);
        identity.case_templates[0].task = "gsm8k".into();
        let session_id = aiperf_accuracy::EvaluationSessionId::new("session-1").unwrap();
        let case_id = EvaluationCaseId::new("case-1").unwrap();
        let unit = EvaluationUnitOccurrence {
            unit_id: EvaluationUnitId::new("unit-1").unwrap(),
            unit_template_id: identity.unit_templates[0].unit_template_id.clone(),
            cases: vec![EvaluationCaseOccurrenceDescriptor {
                case_id: case_id.clone(),
                template_id: identity.case_templates[0].template_id.clone(),
                issue_ordinal: 0,
                phase_id: EvaluationPhaseId::new("finite").unwrap(),
                cycle_index: 0,
            }],
        };
        let credits = EvaluationQueueCredits {
            units: 1,
            host_operations: 2,
            host_operations_per_unit: 2,
            stream_events: 4,
            sandboxes: 0,
            processes: 0,
            artifacts: 1,
            artifact_bytes: 1024,
        };
        let plan = EvaluationPlan {
            assets: Vec::new(),
            host_requirements: Vec::new(),
            logical_services: vec![LogicalServiceRequirement {
                service_id: LogicalServiceId::new("primary").unwrap(),
                purpose: OperationPurpose::new("primary").unwrap(),
                operations: vec![SemanticOperationId::new("model.generate").unwrap()],
                allows_restricted_payload: false,
            }],
            aggregation_policy: AggregationPolicy {
                policy_id: "mean".into(),
                exclude_infrastructure: true,
                exclude_cancelled: true,
                definition: CanonicalJson::new(json!({"reducer":"mean"})).unwrap(),
            },
            execution_granularity: EvaluationExecutionGranularity::Case,
            scheduling_mode: EvaluationSchedulingMode::Finite,
            finite_unit_count: Some(1),
            finite_case_count: Some(1),
            queue_credits: credits,
        };
        let artifact_id = EvaluationArtifactId::new("bundle").unwrap();
        let candidate = EvaluationFinishCandidate {
            identity: identity.clone(),
            outcomes: vec![CaseOutcome {
                case_id,
                outcome: CaseOutcomeKind::Completed {
                    completed: CompletedCaseOutcome {
                        scores: BTreeMap::from([(
                            "accuracy".into(),
                            ProviderScore {
                                value: CanonicalJson::new(json!(1)).unwrap(),
                                public_projection: None,
                            },
                        )]),
                        numeric_metrics: BTreeMap::from([(
                            "accuracy".into(),
                            FiniteF64::new(1.0).unwrap(),
                        )]),
                        primary_score: Some("accuracy".into()),
                        annotations: None,
                    },
                },
                artifact_refs: Vec::new(),
            }],
            aggregates: vec![AggregateMetric {
                scorer: "gsm8k".into(),
                reducer: "mean".into(),
                metric: "accuracy".into(),
                value: FiniteF64::new(1.0).unwrap(),
                scored_count: 1,
                unscored_count: 0,
                definition: CanonicalJson::new(json!({"reducer":"mean"})).unwrap(),
            }],
            artifacts: vec![EvaluationArtifactManifestEntry {
                artifact_id: artifact_id.clone(),
                path: "bundle.json".into(),
                media_type: "application/json".into(),
                visibility: ArtifactVisibility::Restricted,
                size_bytes: 2,
                artifact_content_sha256: "9".repeat(64),
            }],
            provider_bundle: ArtifactRef {
                artifact_id,
                path: "bundle.json".into(),
                visibility: ArtifactVisibility::Restricted,
            },
            normalized_result_sha256: "8".repeat(64),
        };
        let request = EvaluationPlanRequest {
            session_id: session_id.clone(),
            provider_id: identity.worker.provider_id.clone(),
            distribution_id: identity.worker.distribution_id.clone(),
            config_schema_version: 1,
            config_schema_sha256: identity.config_schema_sha256.clone(),
            provider_config: CanonicalJson::new(json!({"benchmark":"gsm8k"})).unwrap(),
            reproducible: true,
        };
        (
            ProofProvider {
                identity,
                plan,
                unit,
                candidate,
                session_id,
                events: VecDeque::new(),
                next_sequence: 0,
                page_returned: false,
                terminal_submitted: false,
                operation_deadline_ms: Some(1_000),
                state,
            },
            request,
        )
    }

    fn proof_workload(
        provider: ProofProvider,
        plan_request: EvaluationPlanRequest,
        clock: Rc<SimClock>,
        cancellation: OperationCancellation,
    ) -> EvaluationWorkload {
        let credits = provider.plan.queue_credits;
        let scheduled = ScheduledRuntime::new(
            clock,
            0,
            Rc::new(UnusedDispatcher),
            StopConfig::default(),
            false,
        );
        let mut host_builder = HostExecutorRegistryBuilder::default();
        host_builder
            .register(Rc::new(ProofHostFactory {
                descriptor: HostOperationDescriptor {
                    operation_id: RegisteredOperationId::new("model.generate").unwrap(),
                    family: HostOperationFamily::new("inference").unwrap(),
                    request_schema_fingerprint: "1".repeat(64),
                    response_schema_fingerprint: "2".repeat(64),
                    stream_schema_fingerprint: Some("3".repeat(64)),
                    true_streaming: true,
                    max_request_bytes: 1024,
                    max_response_bytes: 4096,
                    endpoint_capabilities: BTreeSet::from(["chat".into()]),
                },
            }))
            .unwrap();
        let routes = EvaluationRouteTable::new([EvaluationRoute {
            service_id: "primary".into(),
            purpose: "primary".into(),
            model: "candidate".into(),
            endpoint_profile: "candidate_profile".into(),
            prepared_identity_sha256: "4".repeat(64),
            endpoint_capabilities: BTreeSet::from(["chat".into()]),
        }])
        .unwrap();
        EvaluationWorkload::new(
            Box::new(provider),
            plan_request,
            routes,
            host_builder.freeze().unwrap(),
            HostExecutorRuntime::scheduled(scheduled),
            Rc::new(EmptyAssetResolver),
            EvaluationHostCapabilityInventory::default(),
            PublicScoreProjectionPolicy::restricted_only(),
            None,
            Rc::new(ProofArtifactFinalizer),
            EvaluationWorkloadLimits {
                unit_concurrency: 1,
                credit_ceiling: credits,
                unit_page_size: 1,
                event_batch_size: 8,
                idle_poll_ns: 1,
            },
            cancellation,
        )
        .unwrap()
    }

    #[test]
    fn workload_drives_provider_host_stream_drain_seal_and_deferred_commit() {
        let state = Rc::new(RefCell::new(ProviderProofState::default()));
        let (provider, plan_request) = provider_proof_fixture(state.clone());
        let clock = Rc::new(SimClock::new());
        let workload = proof_workload(
            provider,
            plan_request,
            clock.clone(),
            OperationCancellation::default(),
        );
        let result = Rc::new(RefCell::new(None));
        let result_for_run = result.clone();
        let outcome = drive_sim(clock, move |_handle| async move {
            *result_for_run.borrow_mut() = Some(workload.execute().await.unwrap());
        });
        assert!(!outcome.deadlocked);
        let result = result.borrow_mut().take().unwrap();
        assert_eq!(result.operations.len(), 1);
        assert_eq!(
            result.report_facts.route_summaries["primary"].logical_operations,
            1
        );
        assert_eq!(
            result.report_facts.route_summaries["primary"].prompt_tokens,
            Some(3)
        );
        assert!(state.borrow().shutdown);
        assert!(state.borrow().sealed);
        assert!(!state.borrow().committed);
        assert!(matches!(
            state.borrow().submitted.as_slice(),
            [
                HostOperationEvent::StreamDelta { .. },
                HostOperationEvent::Terminal { .. }
            ]
        ));
        result.report_commit.commit().unwrap();
        assert!(state.borrow().committed);
    }

    #[test]
    fn zero_deadline_cancels_queued_operation_before_executor_admission() {
        let state = Rc::new(RefCell::new(ProviderProofState::default()));
        let (mut provider, plan_request) = provider_proof_fixture(state.clone());
        provider.operation_deadline_ms = Some(0);
        let clock = Rc::new(SimClock::new());
        let workload = proof_workload(
            provider,
            plan_request,
            clock.clone(),
            OperationCancellation::default(),
        );
        let result = Rc::new(RefCell::new(None));
        let result_for_run = result.clone();
        let outcome = drive_sim(clock, move |_handle| async move {
            *result_for_run.borrow_mut() = Some(workload.execute().await.unwrap());
        });
        assert!(!outcome.deadlocked);
        let result = result.borrow_mut().take().unwrap();
        assert_eq!(
            result.operations[0].terminal,
            Some(HostTerminalClass::Cancelled)
        );
        assert!(result.operations[0].attempts.is_empty());
        assert!(matches!(
            state.borrow().submitted.as_slice(),
            [HostOperationEvent::Terminal { .. }]
        ));
    }

    #[test]
    fn run_cancellation_starts_and_cancels_pending_units_and_drains_queued_work() {
        let state = Rc::new(RefCell::new(ProviderProofState::default()));
        let (provider, plan_request) = provider_proof_fixture(state.clone());
        let cancellation = OperationCancellation::default();
        cancellation.cancel();
        let clock = Rc::new(SimClock::new());
        let workload = proof_workload(provider, plan_request, clock.clone(), cancellation);
        let result = Rc::new(RefCell::new(None));
        let result_for_run = result.clone();
        let outcome = drive_sim(clock, move |_handle| async move {
            *result_for_run.borrow_mut() = Some(workload.execute().await.unwrap());
        });
        assert!(!outcome.deadlocked);
        let result = result.borrow_mut().take().unwrap();
        assert_eq!(
            result.operations[0].terminal,
            Some(HostTerminalClass::Cancelled)
        );
        assert_eq!(state.borrow().cancelled_units.len(), 1);
        assert!(matches!(
            state.borrow().submitted.as_slice(),
            [HostOperationEvent::Terminal { .. }]
        ));
    }
}
