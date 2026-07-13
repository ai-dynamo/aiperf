// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Exact evaluator host-operation and transport-attempt ledger.
//!
//! Provider semantic outcomes and Rust transport terminals are independent
//! axes. This ledger owns only host effects: one logical operation can contain
//! multiple Rust transport attempts, but it receives exactly one terminal and
//! can never retry after observable output unless its registered replay policy
//! explicitly permits that behavior.

use std::collections::{BTreeMap, BTreeSet};

use anyhow::{Result, anyhow, ensure};

/// Lifecycle state of one logical host operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OperationState {
    /// Accepted into the bounded fair queue.
    Queued,
    /// Admitted to an executor but no transport attempt is active.
    Admitted,
    /// One transport attempt is active.
    Dispatching,
    /// At least one output delta has crossed the Rust boundary.
    Streaming,
    /// Cancellation has been requested and terminal acknowledgement is pending.
    Cancelling,
    /// Exactly one terminal has been recorded.
    Terminal,
}

/// Rust-owned terminal class for a logical host operation or transport attempt.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum HostTerminalClass {
    /// Normal host result was returned.
    Completed,
    /// Host execution failed after admission.
    Failed,
    /// Capability, validation, or scheduling policy rejected the operation.
    Rejected,
    /// Operation was cancelled before or during execution.
    Cancelled,
}

/// Immutable correlation and route metadata for one logical operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OperationRegistration {
    /// Globally unique logical operation ID.
    pub operation_id: String,
    /// Parent evaluation unit.
    pub unit_id: String,
    /// Parent opaque case occurrence.
    pub case_id: String,
    /// Provider semantic attempt under that case.
    pub semantic_attempt_id: String,
    /// Provider logical call identity.
    pub logical_call_id: String,
    /// Provider idempotency identity bound one-to-one to this logical operation.
    pub idempotency_key: String,
    /// Logical service resolved by the Rust route table.
    pub service_id: String,
    /// Registered open semantic operation ID.
    pub semantic_operation_id: String,
    /// Whether retry after output is proven safe for this operation and route.
    pub replay_safe_after_output: bool,
}

/// One Rust-owned transport attempt below a logical operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TransportAttemptRecord {
    /// Fresh attempt ID allocated by Rust.
    pub attempt_id: String,
    /// Zero-based lineage ordinal.
    pub ordinal: usize,
    /// Whether any output became externally observable.
    pub output_observed: bool,
    /// Terminal class once the attempt ends.
    pub terminal: Option<HostTerminalClass>,
}

/// Complete logical operation entry retained for report reconciliation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OperationRecord {
    /// Immutable provider and route correlations.
    pub registration: OperationRegistration,
    /// Current logical lifecycle state.
    pub state: OperationState,
    /// Ordered Rust transport attempts.
    pub attempts: Vec<TransportAttemptRecord>,
    /// Final logical terminal, exactly once.
    pub terminal: Option<HostTerminalClass>,
}

/// Exact host-effect ledger for one evaluator session.
#[derive(Debug, Default)]
pub struct OperationLedger {
    operations: BTreeMap<String, OperationRecord>,
    logical_call_ids: BTreeMap<String, String>,
    idempotency_keys: BTreeMap<String, String>,
    attempt_ids: BTreeSet<String>,
}

impl OperationLedger {
    /// Register one newly accepted logical operation in queued state.
    pub fn register(&mut self, registration: OperationRegistration) -> Result<()> {
        self.check_registration(&registration)?;
        let operation_id = registration.operation_id.clone();
        self.logical_call_ids
            .insert(registration.logical_call_id.clone(), operation_id.clone());
        self.idempotency_keys
            .insert(registration.idempotency_key.clone(), operation_id.clone());
        self.operations.insert(
            operation_id,
            OperationRecord {
                registration,
                state: OperationState::Queued,
                attempts: Vec::new(),
                terminal: None,
            },
        );
        Ok(())
    }

    /// Validate all identity indexes before a caller commits queue state.
    pub fn check_registration(&self, registration: &OperationRegistration) -> Result<()> {
        validate_registration(registration)?;
        ensure!(
            !self.operations.contains_key(&registration.operation_id),
            "duplicate evaluator host operation {:?}",
            registration.operation_id
        );
        ensure!(
            !self
                .logical_call_ids
                .contains_key(&registration.logical_call_id),
            "duplicate evaluator logical call {:?}",
            registration.logical_call_id
        );
        ensure!(
            !self
                .idempotency_keys
                .contains_key(&registration.idempotency_key),
            "duplicate evaluator idempotency key {:?}",
            registration.idempotency_key
        );
        Ok(())
    }

    /// Move a queued operation into its registered executor.
    pub fn admit(&mut self, operation_id: &str) -> Result<()> {
        let record = self.operation_mut(operation_id)?;
        ensure!(
            record.state == OperationState::Queued,
            "operation {operation_id:?} cannot be admitted from {:?}",
            record.state
        );
        record.state = OperationState::Admitted;
        Ok(())
    }

    /// Begin one fresh Rust transport attempt.
    pub fn start_attempt(&mut self, operation_id: &str, attempt_id: String) -> Result<usize> {
        ensure!(
            !attempt_id.trim().is_empty(),
            "transport attempt ID must not be empty"
        );
        ensure!(
            self.attempt_ids.insert(attempt_id.clone()),
            "duplicate transport attempt ID {attempt_id:?}"
        );
        let record = self.operation_mut(operation_id)?;
        ensure!(
            matches!(record.state, OperationState::Admitted),
            "operation {operation_id:?} cannot start an attempt from {:?}",
            record.state
        );
        ensure!(
            record.terminal.is_none(),
            "terminal operation started an attempt"
        );
        let ordinal = record.attempts.len();
        record.attempts.push(TransportAttemptRecord {
            attempt_id,
            ordinal,
            output_observed: false,
            terminal: None,
        });
        record.state = OperationState::Dispatching;
        Ok(ordinal)
    }

    /// Record the first or a subsequent externally visible stream delta.
    pub fn observe_output(&mut self, operation_id: &str, attempt_id: &str) -> Result<()> {
        let record = self.operation_mut(operation_id)?;
        ensure!(
            matches!(
                record.state,
                OperationState::Dispatching | OperationState::Streaming
            ),
            "operation {operation_id:?} produced output from {:?}",
            record.state
        );
        let attempt = active_attempt_mut(record, attempt_id)?;
        attempt.output_observed = true;
        record.state = OperationState::Streaming;
        Ok(())
    }

    /// Finish the active transport attempt and return whether another attempt
    /// may be considered without violating replay safety.
    pub fn finish_attempt(
        &mut self,
        operation_id: &str,
        attempt_id: &str,
        terminal: HostTerminalClass,
    ) -> Result<bool> {
        let record = self.operation_mut(operation_id)?;
        ensure!(
            matches!(
                record.state,
                OperationState::Dispatching
                    | OperationState::Streaming
                    | OperationState::Cancelling
            ),
            "operation {operation_id:?} cannot finish an attempt from {:?}",
            record.state
        );
        let cancelling = record.state == OperationState::Cancelling;
        let replay_safe_after_output = record.registration.replay_safe_after_output;
        let attempt = active_attempt_mut(record, attempt_id)?;
        ensure!(
            attempt.terminal.is_none(),
            "transport attempt {attempt_id:?} completed more than once"
        );
        attempt.terminal = Some(terminal);
        let output_observed = attempt.output_observed;
        record.state = if cancelling {
            OperationState::Cancelling
        } else {
            OperationState::Admitted
        };
        Ok(!cancelling && (!output_observed || replay_safe_after_output))
    }

    /// Request idempotent cancellation at any non-terminal stage.
    ///
    /// Returns `true` only for the first request, allowing callers to avoid
    /// duplicate transport teardown while still acknowledging repeated provider
    /// cancellation messages.
    pub fn request_cancel(&mut self, operation_id: &str) -> Result<bool> {
        let record = self.operation_mut(operation_id)?;
        match record.state {
            OperationState::Terminal | OperationState::Cancelling => Ok(false),
            OperationState::Queued
            | OperationState::Admitted
            | OperationState::Dispatching
            | OperationState::Streaming => {
                record.state = OperationState::Cancelling;
                Ok(true)
            }
        }
    }

    /// Record the one logical terminal after all attempt work is quiescent.
    pub fn finish_operation(
        &mut self,
        operation_id: &str,
        terminal: HostTerminalClass,
    ) -> Result<()> {
        let record = self.operation_mut(operation_id)?;
        ensure!(
            record.terminal.is_none(),
            "evaluator host operation {operation_id:?} completed more than once"
        );
        ensure!(
            record
                .attempts
                .last()
                .is_none_or(|attempt| attempt.terminal.is_some()),
            "evaluator host operation {operation_id:?} terminated with a live attempt"
        );
        record.terminal = Some(terminal);
        record.state = OperationState::Terminal;
        Ok(())
    }

    /// Immutable operation record for dispatch/report joins.
    pub fn operation(&self, operation_id: &str) -> Result<&OperationRecord> {
        self.operations
            .get(operation_id)
            .ok_or_else(|| anyhow!("unknown evaluator host operation {operation_id:?}"))
    }

    /// Operations in deterministic logical ID order.
    pub fn operations(&self) -> impl ExactSizeIterator<Item = &OperationRecord> {
        self.operations.values()
    }

    /// Reject finalization unless every operation has exactly one terminal and
    /// every transport attempt is terminal.
    pub fn validate_drained(&self) -> Result<()> {
        for (operation_id, record) in &self.operations {
            ensure!(
                record.state == OperationState::Terminal && record.terminal.is_some(),
                "evaluator host operation {operation_id:?} remained in {:?}",
                record.state
            );
            for attempt in &record.attempts {
                ensure!(
                    attempt.terminal.is_some(),
                    "transport attempt {:?} remained live",
                    attempt.attempt_id
                );
            }
        }
        Ok(())
    }

    fn operation_mut(&mut self, operation_id: &str) -> Result<&mut OperationRecord> {
        self.operations
            .get_mut(operation_id)
            .ok_or_else(|| anyhow!("unknown evaluator host operation {operation_id:?}"))
    }
}

fn validate_registration(registration: &OperationRegistration) -> Result<()> {
    for (name, value) in [
        ("operation_id", registration.operation_id.as_str()),
        ("unit_id", registration.unit_id.as_str()),
        ("case_id", registration.case_id.as_str()),
        (
            "semantic_attempt_id",
            registration.semantic_attempt_id.as_str(),
        ),
        ("logical_call_id", registration.logical_call_id.as_str()),
        ("idempotency_key", registration.idempotency_key.as_str()),
        ("service_id", registration.service_id.as_str()),
        (
            "semantic_operation_id",
            registration.semantic_operation_id.as_str(),
        ),
    ] {
        ensure!(!value.trim().is_empty(), "{name} must not be empty");
    }
    Ok(())
}

fn active_attempt_mut<'a>(
    record: &'a mut OperationRecord,
    attempt_id: &str,
) -> Result<&'a mut TransportAttemptRecord> {
    let attempt = record
        .attempts
        .last_mut()
        .ok_or_else(|| anyhow!("operation has no active transport attempt"))?;
    ensure!(
        attempt.attempt_id == attempt_id,
        "transport attempt identity changed from {:?} to {attempt_id:?}",
        attempt.attempt_id
    );
    Ok(attempt)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn registration(id: &str, replay_safe_after_output: bool) -> OperationRegistration {
        OperationRegistration {
            operation_id: id.into(),
            unit_id: "unit".into(),
            case_id: "case".into(),
            semantic_attempt_id: "semantic-attempt".into(),
            logical_call_id: format!("call-{id}"),
            idempotency_key: format!("idempotency-{id}"),
            service_id: "primary".into(),
            semantic_operation_id: "model.generate".into(),
            replay_safe_after_output,
        }
    }

    #[test]
    fn duplicate_and_late_terminals_fail_closed() {
        let mut ledger = OperationLedger::default();
        ledger.register(registration("operation", false)).unwrap();
        assert!(ledger.register(registration("operation", false)).is_err());
        ledger.admit("operation").unwrap();
        ledger
            .start_attempt("operation", "attempt-0".into())
            .unwrap();
        ledger
            .finish_attempt("operation", "attempt-0", HostTerminalClass::Completed)
            .unwrap();
        ledger
            .finish_operation("operation", HostTerminalClass::Completed)
            .unwrap();
        assert!(
            ledger
                .finish_operation("operation", HostTerminalClass::Completed)
                .is_err()
        );
        assert!(
            ledger
                .start_attempt("operation", "attempt-1".into())
                .is_err()
        );
        ledger.validate_drained().unwrap();
    }

    #[test]
    fn logical_call_and_idempotency_identities_are_unique_across_ingresses() {
        let mut ledger = OperationLedger::default();
        ledger
            .register(registration("pipe-operation", false))
            .unwrap();

        let mut duplicate_call = registration("proxy-operation", false);
        duplicate_call.logical_call_id = "call-pipe-operation".into();
        assert!(ledger.register(duplicate_call).is_err());

        let mut duplicate_idempotency = registration("proxy-operation", false);
        duplicate_idempotency.idempotency_key = "idempotency-pipe-operation".into();
        assert!(ledger.register(duplicate_idempotency).is_err());

        ledger
            .register(registration("proxy-operation", false))
            .unwrap();
        assert_eq!(ledger.operations().len(), 2);
    }

    #[test]
    fn retry_after_observed_output_fails_closed_without_explicit_replay_safety() {
        let mut ledger = OperationLedger::default();
        ledger.register(registration("operation", false)).unwrap();
        ledger.admit("operation").unwrap();
        ledger
            .start_attempt("operation", "attempt-0".into())
            .unwrap();
        ledger.observe_output("operation", "attempt-0").unwrap();
        let may_retry = ledger
            .finish_attempt("operation", "attempt-0", HostTerminalClass::Failed)
            .unwrap();
        assert!(!may_retry);
        ledger
            .finish_operation("operation", HostTerminalClass::Failed)
            .unwrap();
    }

    #[test]
    fn replay_safe_operation_may_retry_with_unique_attempt_lineage() {
        let mut ledger = OperationLedger::default();
        ledger.register(registration("operation", true)).unwrap();
        ledger.admit("operation").unwrap();
        assert_eq!(
            ledger
                .start_attempt("operation", "attempt-0".into())
                .unwrap(),
            0
        );
        ledger.observe_output("operation", "attempt-0").unwrap();
        assert!(
            ledger
                .finish_attempt("operation", "attempt-0", HostTerminalClass::Failed)
                .unwrap()
        );
        assert_eq!(
            ledger
                .start_attempt("operation", "attempt-1".into())
                .unwrap(),
            1
        );
        assert!(
            ledger
                .start_attempt("operation", "attempt-1".into())
                .is_err()
        );
        ledger
            .finish_attempt("operation", "attempt-1", HostTerminalClass::Completed)
            .unwrap();
        ledger
            .finish_operation("operation", HostTerminalClass::Completed)
            .unwrap();
        ledger.validate_drained().unwrap();
    }

    #[test]
    fn provider_cancellation_is_idempotent_and_leaves_no_attempt_live() {
        let mut ledger = OperationLedger::default();
        ledger.register(registration("operation", false)).unwrap();
        ledger.admit("operation").unwrap();
        ledger
            .start_attempt("operation", "attempt-0".into())
            .unwrap();
        assert!(ledger.request_cancel("operation").unwrap());
        assert!(!ledger.request_cancel("operation").unwrap());
        assert!(
            !ledger
                .finish_attempt("operation", "attempt-0", HostTerminalClass::Cancelled)
                .unwrap()
        );
        ledger
            .finish_operation("operation", HostTerminalClass::Cancelled)
            .unwrap();
        assert!(!ledger.request_cancel("operation").unwrap());
        ledger.validate_drained().unwrap();
    }

    #[test]
    fn finalization_rejects_missing_operation_or_attempt_terminal() {
        let mut ledger = OperationLedger::default();
        ledger.register(registration("queued", false)).unwrap();
        assert!(ledger.validate_drained().is_err());
        ledger.admit("queued").unwrap();
        ledger.start_attempt("queued", "attempt-0".into()).unwrap();
        assert!(ledger.validate_drained().is_err());
    }
}
