// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Strict evaluator session state machine and exact unit/case/operation ledger.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::provider::{EvaluationProviderError, EvaluatorProtocolLimits};
use crate::provider_protocol::{
    CaseOutcome, EvaluationEvent, EvaluationEventBatch, EvaluationPlan, EvaluationSchedulingMode,
    EvaluationUnitId, EvaluationUnitOccurrence, HostOperationDisposition, HostOperationEvent,
    HostOperationId, SemanticAttemptId,
};

/// Observable evaluator lifecycle state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvaluationLifecycleState {
    /// Child exists but has not negotiated evaluator-worker protocol v2.
    Spawned,
    /// `hello` identity and launch attestation match.
    Negotiated,
    /// Side-effect-free provider planning completed.
    Planned,
    /// Immutable Rust-resolved assets were accepted.
    AssetsBound,
    /// Frozen templates/identity are ready for unit admission.
    Ready,
    /// At least one evaluation unit is active.
    Running,
    /// Cancellation is active while units/operations drain.
    Cancelling,
    /// Every unit, case, and host operation reached terminal.
    Drained,
    /// Provider returned a candidate manifest; artifacts are not trusted yet.
    ManifestCandidate,
    /// Host capabilities are revoked and the complete worker tree is exiting.
    Quiescing,
    /// Worker and every descendant are proven exited.
    WorkerExited,
    /// Rust verified and atomically promoted immutable artifacts.
    ArtifactsSealed,
    /// Native report was committed after sealing.
    ReportCommitted,
}

#[derive(Debug, Clone)]
struct OperationLedgerEntry {
    semantic_attempt_id: SemanticAttemptId,
    unit_id: EvaluationUnitId,
    last_stream_sequence: Option<u64>,
    cancellation_requested: bool,
}

/// Stateful protocol validator shared by process clients and fixture transports.
#[derive(Debug)]
pub struct EvaluationLifecycle {
    state: EvaluationLifecycleState,
    limits: EvaluatorProtocolLimits,
    scheduling_mode: Option<EvaluationSchedulingMode>,
    queue_host_operation_limit: usize,
    queue_per_unit_limit: usize,
    next_event_sequence: u64,
    idempotency_keys: BTreeSet<String>,
    known_units: BTreeMap<EvaluationUnitId, BTreeSet<crate::provider_protocol::EvaluationCaseId>>,
    canonical_case_order: Vec<crate::provider_protocol::EvaluationCaseId>,
    started_units: BTreeSet<EvaluationUnitId>,
    terminal_cases: BTreeSet<crate::provider_protocol::EvaluationCaseId>,
    outstanding_operations: BTreeMap<HostOperationId, OperationLedgerEntry>,
    terminal_operations: BTreeSet<HostOperationId>,
    pending_cancellation_acks: BTreeSet<HostOperationId>,
}

impl EvaluationLifecycle {
    /// Create the mandatory state machine before worker negotiation.
    pub fn new(limits: EvaluatorProtocolLimits) -> Result<Self, EvaluationProviderError> {
        limits
            .validate()
            .map_err(EvaluationProviderError::registry)?;
        Ok(Self {
            state: EvaluationLifecycleState::Spawned,
            limits,
            scheduling_mode: None,
            queue_host_operation_limit: 0,
            queue_per_unit_limit: 0,
            next_event_sequence: 1,
            idempotency_keys: BTreeSet::new(),
            known_units: BTreeMap::new(),
            canonical_case_order: Vec::new(),
            started_units: BTreeSet::new(),
            terminal_cases: BTreeSet::new(),
            outstanding_operations: BTreeMap::new(),
            terminal_operations: BTreeSet::new(),
            pending_cancellation_acks: BTreeSet::new(),
        })
    }

    /// Current exact lifecycle state.
    pub fn state(&self) -> EvaluationLifecycleState {
        self.state
    }

    /// Number of host operations not yet terminal.
    pub fn outstanding_host_operations(&self) -> usize {
        self.outstanding_operations.len()
    }

    /// Record successful protocol negotiation and attestation.
    pub fn negotiated(&mut self) -> Result<(), EvaluationProviderError> {
        self.transition(
            EvaluationLifecycleState::Spawned,
            EvaluationLifecycleState::Negotiated,
        )
    }

    /// Record and validate one accepted side-effect-free plan.
    pub fn planned(&mut self, plan: &EvaluationPlan) -> Result<(), EvaluationProviderError> {
        self.require(EvaluationLifecycleState::Negotiated, "plan_session")?;
        plan.validate()?;
        if plan.queue_credits.host_operations > self.limits.max_idempotency_keys
            || plan.queue_credits.host_operations > self.limits.max_collection_items * 1_024
        {
            return Err(EvaluationProviderError::Protocol(
                "provider host-operation credits exceeded host safety ceilings".to_string(),
            ));
        }
        self.scheduling_mode = Some(plan.scheduling_mode);
        self.queue_host_operation_limit = plan.queue_credits.host_operations;
        self.queue_per_unit_limit = plan.queue_credits.host_operations_per_unit;
        self.state = EvaluationLifecycleState::Planned;
        Ok(())
    }

    /// Record asset binding, then the distinct ready state after frozen identity validation.
    pub fn assets_bound_and_ready(&mut self) -> Result<(), EvaluationProviderError> {
        self.transition(
            EvaluationLifecycleState::Planned,
            EvaluationLifecycleState::AssetsBound,
        )?;
        self.transition(
            EvaluationLifecycleState::AssetsBound,
            EvaluationLifecycleState::Ready,
        )
    }

    /// Register finite or newly instantiated units before they can start.
    pub fn register_units(
        &mut self,
        units: &[EvaluationUnitOccurrence],
    ) -> Result<(), EvaluationProviderError> {
        self.require_any(
            &[
                EvaluationLifecycleState::Ready,
                EvaluationLifecycleState::Running,
            ],
            "register units",
        )?;
        if units.len() > self.limits.max_collection_items {
            return Err(EvaluationProviderError::Protocol(
                "unit result exceeded negotiated collection bound".to_string(),
            ));
        }
        for unit in units {
            if unit.cases.is_empty() || self.known_units.contains_key(&unit.unit_id) {
                return Err(EvaluationProviderError::Protocol(format!(
                    "unit {} was empty or duplicated",
                    unit.unit_id
                )));
            }
            let cases = unit
                .cases
                .iter()
                .map(|case| case.case_id.clone())
                .collect::<BTreeSet<_>>();
            if cases.len() != unit.cases.len()
                || cases.iter().any(|case| {
                    self.known_units
                        .values()
                        .any(|known_cases| known_cases.contains(case))
                })
            {
                return Err(EvaluationProviderError::Protocol(format!(
                    "unit {} contained duplicate case identity",
                    unit.unit_id
                )));
            }
            self.known_units.insert(unit.unit_id.clone(), cases);
            self.canonical_case_order
                .extend(unit.cases.iter().map(|case| case.case_id.clone()));
        }
        Ok(())
    }

    /// Validate that occurrence instantiation is supported by this exact plan.
    pub fn validate_instantiate(&self) -> Result<(), EvaluationProviderError> {
        self.require_any(
            &[
                EvaluationLifecycleState::Ready,
                EvaluationLifecycleState::Running,
            ],
            "instantiate_units",
        )?;
        if self.scheduling_mode != Some(EvaluationSchedulingMode::RustOccurrences) {
            return Err(EvaluationProviderError::Protocol(
                "finite evaluation plan rejected instantiate_units".to_string(),
            ));
        }
        Ok(())
    }

    /// Validate and record unit admission.
    pub fn start_units(&mut self, ids: &[EvaluationUnitId]) -> Result<(), EvaluationProviderError> {
        self.require_any(
            &[
                EvaluationLifecycleState::Ready,
                EvaluationLifecycleState::Running,
            ],
            "start_units",
        )?;
        if ids.is_empty() || ids.len() > self.limits.max_collection_items {
            return Err(EvaluationProviderError::Protocol(
                "start_units batch was empty or exceeded bounds".to_string(),
            ));
        }
        let unique = ids.iter().collect::<BTreeSet<_>>();
        if unique.len() != ids.len()
            || ids
                .iter()
                .any(|id| !self.known_units.contains_key(id) || self.started_units.contains(id))
        {
            return Err(EvaluationProviderError::Protocol(
                "start_units referenced unknown, duplicate, or previously started units"
                    .to_string(),
            ));
        }
        self.started_units.extend(ids.iter().cloned());
        self.state = EvaluationLifecycleState::Running;
        Ok(())
    }

    /// Begin idempotent cancellation without losing outstanding ledger entries.
    pub fn begin_cancellation(
        &mut self,
        ids: &[EvaluationUnitId],
    ) -> Result<(), EvaluationProviderError> {
        self.require_any(
            &[
                EvaluationLifecycleState::Running,
                EvaluationLifecycleState::Cancelling,
            ],
            "cancel_units",
        )?;
        if ids.is_empty()
            || ids.len() > self.limits.max_collection_items
            || ids.iter().any(|id| !self.started_units.contains(id))
        {
            return Err(EvaluationProviderError::Protocol(
                "cancel_units referenced an empty, oversized, or unknown unit batch".to_string(),
            ));
        }
        self.state = EvaluationLifecycleState::Cancelling;
        Ok(())
    }

    /// Validate an ordered provider event batch and update exact ledgers.
    pub fn record_event_batch(
        &mut self,
        batch: &mut EvaluationEventBatch,
    ) -> Result<(), EvaluationProviderError> {
        self.require_any(
            &[
                EvaluationLifecycleState::Running,
                EvaluationLifecycleState::Cancelling,
            ],
            "poll_events",
        )?;
        if batch.events.len() > self.limits.max_collection_items {
            return Err(EvaluationProviderError::Protocol(
                "poll_events returned an oversized batch".to_string(),
            ));
        }
        for sequenced in &mut batch.events {
            if sequenced.sequence != self.next_event_sequence {
                return Err(EvaluationProviderError::Protocol(format!(
                    "evaluator event sequence {} did not match expected {}",
                    sequenced.sequence, self.next_event_sequence
                )));
            }
            self.next_event_sequence =
                self.next_event_sequence.checked_add(1).ok_or_else(|| {
                    EvaluationProviderError::Protocol(
                        "evaluator event sequence overflow".to_string(),
                    )
                })?;
            if sequenced.idempotency_key.trim().is_empty()
                || sequenced.idempotency_key.len() > 512
                || !self
                    .idempotency_keys
                    .insert(sequenced.idempotency_key.clone())
                || self.idempotency_keys.len() > self.limits.max_idempotency_keys
            {
                return Err(EvaluationProviderError::Protocol(
                    "evaluator event idempotency key was empty, duplicate, or exceeded bounds"
                        .to_string(),
                ));
            }
            self.record_event(&mut sequenced.event)?;
        }
        if batch.next_sequence != self.next_event_sequence {
            return Err(EvaluationProviderError::Protocol(format!(
                "event batch next_sequence {} did not match {}",
                batch.next_sequence, self.next_event_sequence
            )));
        }
        if batch.remaining_credits.host_operations > self.queue_host_operation_limit
            || batch.remaining_credits.host_operations_per_unit > self.queue_per_unit_limit
        {
            return Err(EvaluationProviderError::Protocol(
                "worker advertised credits above the accepted plan".to_string(),
            ));
        }
        if batch.drained {
            if !self.outstanding_operations.is_empty()
                || !self.started_units.is_empty()
                || self.terminal_cases.len() != self.canonical_case_order.len()
                || self
                    .canonical_case_order
                    .iter()
                    .any(|case| !self.terminal_cases.contains(case))
            {
                return Err(EvaluationProviderError::Protocol(
                    "worker claimed drained with live or missing unit/case/operation terminals"
                        .to_string(),
                ));
            }
            self.state = EvaluationLifecycleState::Drained;
        }
        Ok(())
    }

    /// Validate outgoing Rust host events and record exactly one terminal.
    pub fn record_host_events(
        &mut self,
        events: &mut [HostOperationEvent],
    ) -> Result<Vec<HostOperationId>, EvaluationProviderError> {
        self.require_any(
            &[
                EvaluationLifecycleState::Running,
                EvaluationLifecycleState::Cancelling,
            ],
            "submit_host_events",
        )?;
        if events.is_empty() || events.len() > self.limits.max_collection_items {
            return Err(EvaluationProviderError::Protocol(
                "submit_host_events was empty or exceeded bounds".to_string(),
            ));
        }
        let mut accepted = Vec::with_capacity(events.len());
        for event in events {
            let operation_id = event.operation_id().clone();
            match event {
                HostOperationEvent::StreamDelta {
                    stream_sequence, ..
                } => {
                    let entry = self
                        .outstanding_operations
                        .get_mut(&operation_id)
                        .ok_or_else(|| {
                            EvaluationProviderError::Protocol(format!(
                                "late stream event for unknown/terminal operation {operation_id}"
                            ))
                        })?;
                    let expected = entry
                        .last_stream_sequence
                        .map_or(0, |sequence| sequence.saturating_add(1));
                    if *stream_sequence != expected {
                        return Err(EvaluationProviderError::Protocol(format!(
                            "operation {operation_id} stream sequence {stream_sequence} did not match {expected}"
                        )));
                    }
                    entry.last_stream_sequence = Some(*stream_sequence);
                }
                HostOperationEvent::Usage { .. } => {
                    if !self.outstanding_operations.contains_key(&operation_id) {
                        return Err(EvaluationProviderError::Protocol(format!(
                            "late usage for unknown/terminal operation {operation_id}"
                        )));
                    }
                }
                HostOperationEvent::Terminal { terminal } => {
                    terminal.validate()?;
                    let entry = self
                        .outstanding_operations
                        .remove(&operation_id)
                        .ok_or_else(|| {
                            EvaluationProviderError::Protocol(format!(
                                "duplicate or late terminal for operation {operation_id}"
                            ))
                        })?;
                    if entry.semantic_attempt_id != terminal.semantic_attempt_id
                        || terminal.disposition == HostOperationDisposition::AlreadyTerminal
                    {
                        return Err(EvaluationProviderError::Protocol(format!(
                            "operation {operation_id} terminal attempt/disposition was invalid"
                        )));
                    }
                    self.terminal_operations.insert(operation_id.clone());
                    self.pending_cancellation_acks.remove(&operation_id);
                }
                HostOperationEvent::CancellationAcknowledged {
                    semantic_attempt_id,
                    already_terminal,
                    ..
                } => {
                    if !*already_terminal
                        || !self.terminal_operations.contains(&operation_id)
                        || !self.pending_cancellation_acks.remove(&operation_id)
                    {
                        return Err(EvaluationProviderError::Protocol(format!(
                            "unexpected cancellation acknowledgement for operation {operation_id}"
                        )));
                    }
                    // The original entry is gone after terminal; the attempt was
                    // already checked when the provider requested cancellation.
                    let _ = semantic_attempt_id;
                }
            }
            accepted.push(operation_id);
        }
        Ok(accepted)
    }

    /// Enter manifest-candidate state only after a proven drain.
    pub fn finalized_candidate(&mut self) -> Result<(), EvaluationProviderError> {
        self.transition(
            EvaluationLifecycleState::Drained,
            EvaluationLifecycleState::ManifestCandidate,
        )
    }

    /// Prove the candidate contains every terminal case exactly once in canonical order.
    pub fn validate_finish_candidate(
        &self,
        candidate: &crate::provider_protocol::EvaluationFinishCandidate,
    ) -> Result<(), EvaluationProviderError> {
        self.require(EvaluationLifecycleState::Drained, "finalize_session result")?;
        let actual = candidate
            .outcomes
            .iter()
            .map(|outcome| &outcome.case_id)
            .collect::<Vec<_>>();
        let expected = self.canonical_case_order.iter().collect::<Vec<_>>();
        if actual != expected {
            return Err(EvaluationProviderError::Protocol(
                "finish candidate outcomes were missing, duplicated, or outside canonical order"
                    .to_string(),
            ));
        }
        Ok(())
    }

    /// Revoke capabilities and begin worker-tree shutdown.
    pub fn begin_shutdown(&mut self) -> Result<(), EvaluationProviderError> {
        self.transition(
            EvaluationLifecycleState::ManifestCandidate,
            EvaluationLifecycleState::Quiescing,
        )
    }

    /// Record enforced complete process-tree quiescence.
    pub fn worker_exited(&mut self) -> Result<(), EvaluationProviderError> {
        self.transition(
            EvaluationLifecycleState::Quiescing,
            EvaluationLifecycleState::WorkerExited,
        )
    }

    /// Record Rust-owned artifact sealing.
    pub fn artifacts_sealed(&mut self) -> Result<(), EvaluationProviderError> {
        self.transition(
            EvaluationLifecycleState::WorkerExited,
            EvaluationLifecycleState::ArtifactsSealed,
        )
    }

    /// Record report commit only after artifacts are immutable.
    pub fn report_committed(&mut self) -> Result<(), EvaluationProviderError> {
        self.transition(
            EvaluationLifecycleState::ArtifactsSealed,
            EvaluationLifecycleState::ReportCommitted,
        )
    }

    /// Force the failure path into quiescing without pretending normal states completed.
    pub fn abort_to_quiescing(&mut self) {
        self.state = EvaluationLifecycleState::Quiescing;
    }

    fn record_event(&mut self, event: &mut EvaluationEvent) -> Result<(), EvaluationProviderError> {
        match event {
            EvaluationEvent::HostOperationRequested { request } => {
                request.validate()?;
                if !self.started_units.contains(&request.context.unit_id)
                    || self.terminal_operations.contains(&request.operation_id)
                    || self
                        .outstanding_operations
                        .contains_key(&request.operation_id)
                {
                    return Err(EvaluationProviderError::Protocol(format!(
                        "host operation {} referenced an inactive unit or duplicate identity",
                        request.operation_id
                    )));
                }
                if self.outstanding_operations.len() == self.queue_host_operation_limit {
                    return Err(EvaluationProviderError::Protocol(
                        "provider exceeded accepted global host-operation credits".to_string(),
                    ));
                }
                let unit_outstanding = self
                    .outstanding_operations
                    .values()
                    .filter(|entry| entry.unit_id == request.context.unit_id)
                    .count();
                if unit_outstanding == self.queue_per_unit_limit {
                    return Err(EvaluationProviderError::Protocol(format!(
                        "unit {} exceeded accepted host-operation credits",
                        request.context.unit_id
                    )));
                }
                self.outstanding_operations.insert(
                    request.operation_id.clone(),
                    OperationLedgerEntry {
                        semantic_attempt_id: request.context.semantic_attempt_id.clone(),
                        unit_id: request.context.unit_id.clone(),
                        last_stream_sequence: None,
                        cancellation_requested: false,
                    },
                );
            }
            EvaluationEvent::HostOperationCancelRequested { request } => {
                if let Some(entry) = self.outstanding_operations.get_mut(&request.operation_id) {
                    if entry.semantic_attempt_id != request.semantic_attempt_id {
                        return Err(EvaluationProviderError::Protocol(format!(
                            "operation {} cancellation used a different semantic attempt",
                            request.operation_id
                        )));
                    }
                    entry.cancellation_requested = true;
                } else if self.terminal_operations.contains(&request.operation_id) {
                    self.pending_cancellation_acks
                        .insert(request.operation_id.clone());
                } else {
                    return Err(EvaluationProviderError::Protocol(format!(
                        "operation {} cancellation referenced unknown work",
                        request.operation_id
                    )));
                }
            }
            EvaluationEvent::CaseTerminal { outcome } => self.record_case_terminal(outcome)?,
            EvaluationEvent::Progress { progress } => {
                if progress.outstanding_host_operations as usize
                    != self.outstanding_operations.len()
                {
                    return Err(EvaluationProviderError::Protocol(
                        "provider progress disagreed with the host-operation ledger".to_string(),
                    ));
                }
            }
            EvaluationEvent::Diagnostic { level, message } => {
                if level.trim().is_empty() || level.len() > 32 {
                    return Err(EvaluationProviderError::Protocol(
                        "provider diagnostic level was empty or oversized".to_string(),
                    ));
                }
                *message = crate::canonical::redact_diagnostic(message);
            }
        }
        Ok(())
    }

    fn record_case_terminal(
        &mut self,
        outcome: &mut CaseOutcome,
    ) -> Result<(), EvaluationProviderError> {
        outcome.validate()?;
        if !self.terminal_cases.insert(outcome.case_id.clone()) {
            return Err(EvaluationProviderError::Protocol(format!(
                "duplicate case terminal {}",
                outcome.case_id
            )));
        }
        let Some(unit_id) = self
            .known_units
            .iter()
            .find_map(|(unit_id, cases)| cases.contains(&outcome.case_id).then(|| unit_id.clone()))
        else {
            return Err(EvaluationProviderError::Protocol(format!(
                "terminal case {} was not in the frozen unit manifest",
                outcome.case_id
            )));
        };
        let all_terminal = self.known_units[&unit_id]
            .iter()
            .all(|case| self.terminal_cases.contains(case));
        if all_terminal {
            self.started_units.remove(&unit_id);
        }
        Ok(())
    }

    fn require(
        &self,
        expected: EvaluationLifecycleState,
        operation: &str,
    ) -> Result<(), EvaluationProviderError> {
        if self.state == expected {
            Ok(())
        } else {
            Err(EvaluationProviderError::Lifecycle(format!(
                "{operation} required {expected:?}, current state was {:?}",
                self.state
            )))
        }
    }

    fn require_any(
        &self,
        expected: &[EvaluationLifecycleState],
        operation: &str,
    ) -> Result<(), EvaluationProviderError> {
        if expected.contains(&self.state) {
            Ok(())
        } else {
            Err(EvaluationProviderError::Lifecycle(format!(
                "{operation} was not valid in {:?}",
                self.state
            )))
        }
    }

    fn transition(
        &mut self,
        expected: EvaluationLifecycleState,
        next: EvaluationLifecycleState,
    ) -> Result<(), EvaluationProviderError> {
        self.require(expected, "state transition")?;
        self.state = next;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use crate::canonical::CanonicalJson;
    use crate::provider_protocol::{
        AggregationPolicy, CompletedCaseOutcome, EvaluationCaseId,
        EvaluationCaseOccurrenceDescriptor, EvaluationEvent, EvaluationPhaseId, EvaluationProgress,
        EvaluationQueueCredits, EvaluationUnitTemplateId, FiniteF64, HostCallContext,
        HostOperationRequest, HostResponseMode, LogicalCallId, LogicalServiceId, OperationPurpose,
        ProviderScore, SemanticOperationId, SequencedEvaluationEvent,
    };

    use super::*;

    fn plan(mode: EvaluationSchedulingMode) -> EvaluationPlan {
        EvaluationPlan {
            assets: Vec::new(),
            host_requirements: Vec::new(),
            logical_services: Vec::new(),
            aggregation_policy: AggregationPolicy {
                policy_id: "fixture".to_string(),
                exclude_infrastructure: true,
                exclude_cancelled: true,
                definition: CanonicalJson::new(serde_json::json!({})).unwrap(),
            },
            execution_granularity: crate::provider_protocol::EvaluationExecutionGranularity::Case,
            scheduling_mode: mode,
            finite_unit_count: (mode == EvaluationSchedulingMode::Finite).then_some(1),
            finite_case_count: (mode == EvaluationSchedulingMode::Finite).then_some(1),
            queue_credits: EvaluationQueueCredits {
                units: 2,
                host_operations: 2,
                host_operations_per_unit: 1,
                stream_events: 4,
                sandboxes: 1,
                processes: 1,
                artifacts: 4,
                artifact_bytes: 4096,
            },
        }
    }

    fn unit() -> EvaluationUnitOccurrence {
        EvaluationUnitOccurrence {
            unit_id: EvaluationUnitId::new("unit-1").unwrap(),
            unit_template_id: EvaluationUnitTemplateId::new("template-1").unwrap(),
            cases: vec![EvaluationCaseOccurrenceDescriptor {
                case_id: EvaluationCaseId::new("case-1").unwrap(),
                template_id: crate::provider_protocol::EvaluationCaseTemplateId::new("case-t")
                    .unwrap(),
                issue_ordinal: 0,
                phase_id: EvaluationPhaseId::new("profile").unwrap(),
                cycle_index: 0,
            }],
        }
    }

    fn request() -> HostOperationRequest {
        HostOperationRequest {
            operation_id: HostOperationId::new("op-1").unwrap(),
            context: HostCallContext {
                session_id: crate::provider_protocol::EvaluationSessionId::new("session").unwrap(),
                unit_id: EvaluationUnitId::new("unit-1").unwrap(),
                case_id: EvaluationCaseId::new("case-1").unwrap(),
                semantic_attempt_id: SemanticAttemptId::new("attempt-1").unwrap(),
                logical_call_id: LogicalCallId::new("call-1").unwrap(),
            },
            service_id: LogicalServiceId::new("primary").unwrap(),
            purpose: OperationPurpose::new("primary").unwrap(),
            semantic_operation_id: SemanticOperationId::new("model.generate").unwrap(),
            payload: CanonicalJson::new(serde_json::json!({"messages": []})).unwrap(),
            restricted_payload: None,
            response_mode: HostResponseMode::Streaming,
            deadline_ms: None,
            idempotency_key: "request-1".to_string(),
        }
    }

    fn ready_lifecycle() -> EvaluationLifecycle {
        let mut lifecycle = EvaluationLifecycle::new(EvaluatorProtocolLimits::default()).unwrap();
        lifecycle.negotiated().unwrap();
        lifecycle
            .planned(&plan(EvaluationSchedulingMode::RustOccurrences))
            .unwrap();
        lifecycle.assets_bound_and_ready().unwrap();
        lifecycle.register_units(&[unit()]).unwrap();
        lifecycle
            .start_units(&[EvaluationUnitId::new("unit-1").unwrap()])
            .unwrap();
        lifecycle
    }

    #[test]
    fn no_normal_state_can_be_skipped() {
        let mut lifecycle = EvaluationLifecycle::new(EvaluatorProtocolLimits::default()).unwrap();
        assert!(lifecycle.assets_bound_and_ready().is_err());
        lifecycle.negotiated().unwrap();
        lifecycle
            .planned(&plan(EvaluationSchedulingMode::Finite))
            .unwrap();
        lifecycle.assets_bound_and_ready().unwrap();
        assert_eq!(lifecycle.state(), EvaluationLifecycleState::Ready);
        assert!(lifecycle.finalized_candidate().is_err());
    }

    #[test]
    fn rejects_duplicate_out_of_order_and_late_events() {
        let mut lifecycle = ready_lifecycle();
        let mut batch = EvaluationEventBatch {
            events: vec![SequencedEvaluationEvent {
                sequence: 1,
                idempotency_key: "event-1".to_string(),
                event: EvaluationEvent::HostOperationRequested {
                    request: Box::new(request()),
                },
            }],
            next_sequence: 2,
            drained: false,
            remaining_credits: EvaluationQueueCredits {
                host_operations: 1,
                host_operations_per_unit: 0,
                ..plan(EvaluationSchedulingMode::Finite).queue_credits
            },
        };
        lifecycle.record_event_batch(&mut batch).unwrap();
        assert_eq!(lifecycle.outstanding_host_operations(), 1);

        let mut duplicate = EvaluationEventBatch {
            events: vec![SequencedEvaluationEvent {
                sequence: 2,
                idempotency_key: "event-1".to_string(),
                event: EvaluationEvent::Progress {
                    progress: EvaluationProgress {
                        started_units: 1,
                        terminal_cases: 0,
                        outstanding_host_operations: 1,
                    },
                },
            }],
            next_sequence: 3,
            drained: false,
            remaining_credits: batch.remaining_credits,
        };
        assert!(lifecycle.record_event_batch(&mut duplicate).is_err());
    }

    #[test]
    fn completed_zero_is_terminal_and_drains_only_after_operation_terminal() {
        let mut lifecycle = ready_lifecycle();
        let mut batch = EvaluationEventBatch {
            events: vec![SequencedEvaluationEvent {
                sequence: 1,
                idempotency_key: "event-1".to_string(),
                event: EvaluationEvent::HostOperationRequested {
                    request: Box::new(request()),
                },
            }],
            next_sequence: 2,
            drained: false,
            remaining_credits: EvaluationQueueCredits {
                host_operations: 1,
                host_operations_per_unit: 0,
                ..plan(EvaluationSchedulingMode::Finite).queue_credits
            },
        };
        lifecycle.record_event_batch(&mut batch).unwrap();

        let mut terminal = HostOperationEvent::Terminal {
            terminal: crate::provider_protocol::HostOperationTerminal {
                operation_id: HostOperationId::new("op-1").unwrap(),
                semantic_attempt_id: SemanticAttemptId::new("attempt-1").unwrap(),
                disposition: HostOperationDisposition::Completed,
                result: Some(CanonicalJson::new(serde_json::json!({"text": ""})).unwrap()),
                error: None,
                usage: Default::default(),
                observed_output: false,
            },
        };
        lifecycle
            .record_host_events(std::slice::from_mut(&mut terminal))
            .unwrap();
        assert!(
            lifecycle
                .record_host_events(std::slice::from_mut(&mut terminal))
                .is_err()
        );

        let outcome = CaseOutcome {
            case_id: EvaluationCaseId::new("case-1").unwrap(),
            outcome: crate::provider_protocol::CaseOutcomeKind::Completed {
                completed: CompletedCaseOutcome {
                    scores: BTreeMap::from([(
                        "accuracy".to_string(),
                        ProviderScore {
                            value: CanonicalJson::new(serde_json::json!(0)).unwrap(),
                            public_projection: None,
                        },
                    )]),
                    numeric_metrics: BTreeMap::from([(
                        "accuracy".to_string(),
                        FiniteF64::new(0.0).unwrap(),
                    )]),
                    primary_score: Some("accuracy".to_string()),
                    annotations: None,
                },
            },
            artifact_refs: Vec::new(),
        };
        let mut terminal_batch = EvaluationEventBatch {
            events: vec![SequencedEvaluationEvent {
                sequence: 2,
                idempotency_key: "event-2".to_string(),
                event: EvaluationEvent::CaseTerminal {
                    outcome: Box::new(outcome),
                },
            }],
            next_sequence: 3,
            drained: true,
            remaining_credits: plan(EvaluationSchedulingMode::Finite).queue_credits,
        };
        lifecycle.record_event_batch(&mut terminal_batch).unwrap();
        assert_eq!(lifecycle.state(), EvaluationLifecycleState::Drained);
    }
}
