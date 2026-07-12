// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Clock-driven inference retry above the one-attempt transport seam.
//!
//! The evaluator and compatibility-proxy client never retry upstream work.
//! This module allocates unique transport-attempt lineage under one logical
//! operation, delegates one attempt at a time, and sleeps only through the
//! injected [`Clock`]. Policy and attempt execution are independently
//! replaceable traits so endpoint-specific replay rules remain outside the
//! provider protocol.

use std::cell::Cell;
use std::collections::BTreeSet;
use std::rc::Rc;

use aiperf_accuracy::HostOperationUsage;
use aiperf_clock::Clock;
use anyhow::{Result, ensure};
use async_trait::async_trait;
use serde_json::Value;
use tokio::sync::Notify;

use super::ledger::HostTerminalClass;
use crate::scheduled::DispatchCancellation;

/// Cancellation latch shared by queue, dispatch, streaming, and backoff.
#[derive(Debug, Clone, Default)]
pub struct OperationCancellation {
    cancelled: Rc<Cell<bool>>,
    notify: Rc<Notify>,
}

impl OperationCancellation {
    /// Request cancellation. Returns `true` only for the first request.
    pub fn cancel(&self) -> bool {
        let first = !self.cancelled.replace(true);
        if first {
            self.notify.notify_waiters();
        }
        first
    }

    /// Whether cancellation has already been requested.
    pub fn is_cancelled(&self) -> bool {
        self.cancelled.get()
    }

    /// Wait until cancellation is requested without losing a preceding wakeup.
    pub async fn cancelled(&self) {
        loop {
            let notified = self.notify.notified();
            if self.is_cancelled() {
                return;
            }
            notified.await;
        }
    }
}

impl DispatchCancellation for OperationCancellation {
    fn is_cancelled(&self) -> bool {
        OperationCancellation::is_cancelled(self)
    }

    fn cancelled(&self) -> std::pin::Pin<Box<dyn std::future::Future<Output = ()> + '_>> {
        Box::pin(OperationCancellation::cancelled(self))
    }
}

/// Facts presented to a replaceable transport retry policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RetryContext {
    /// Completed attempt ordinal, starting at zero.
    pub attempt_ordinal: usize,
    /// Rust transport terminal.
    pub terminal: HostTerminalClass,
    /// Whether the attempt became externally observable.
    pub output_observed: bool,
    /// Whether the one-attempt executor classified the terminal retryable.
    pub retryable_hint: bool,
    /// Whether registered endpoint/operation policy proves replay after output safe.
    pub replay_safe_after_output: bool,
}

/// Retry decision returned after one failed transport attempt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RetryDecision {
    /// Return the current terminal to the provider.
    Stop,
    /// Sleep for the specified Clock duration before a fresh attempt.
    RetryAfterNs(i64),
}

/// Replaceable Rust transport retry policy.
pub trait TransportRetryPolicy {
    /// Decide whether and when to allocate another attempt.
    fn decide(&self, context: RetryContext) -> RetryDecision;
}

/// Bounded exponential retry policy without hidden jitter or wall-clock use.
#[derive(Debug, Clone)]
pub struct ExponentialTransportRetryPolicy {
    max_attempts: usize,
    initial_backoff_ns: i64,
    max_backoff_ns: i64,
    retryable_terminals: BTreeSet<HostTerminalClass>,
}

impl ExponentialTransportRetryPolicy {
    /// Build a validated policy. `max_attempts` includes the initial attempt.
    pub fn new(
        max_attempts: usize,
        initial_backoff_ns: i64,
        max_backoff_ns: i64,
        retryable_terminals: impl IntoIterator<Item = HostTerminalClass>,
    ) -> Result<Self> {
        ensure!(max_attempts > 0, "transport max_attempts must be positive");
        ensure!(
            initial_backoff_ns >= 0,
            "transport initial backoff must be non-negative"
        );
        ensure!(
            max_backoff_ns >= initial_backoff_ns,
            "transport maximum backoff must cover initial backoff"
        );
        Ok(Self {
            max_attempts,
            initial_backoff_ns,
            max_backoff_ns,
            retryable_terminals: retryable_terminals.into_iter().collect(),
        })
    }

    fn backoff_ns(&self, attempt_ordinal: usize) -> i64 {
        let exponent = u32::try_from(attempt_ordinal).unwrap_or(u32::MAX).min(62);
        self.initial_backoff_ns
            .saturating_mul(1_i64.checked_shl(exponent).unwrap_or(i64::MAX))
            .min(self.max_backoff_ns)
    }
}

impl TransportRetryPolicy for ExponentialTransportRetryPolicy {
    fn decide(&self, context: RetryContext) -> RetryDecision {
        let attempts_so_far = context.attempt_ordinal.saturating_add(1);
        if attempts_so_far >= self.max_attempts
            || !context.retryable_hint
            || !self.retryable_terminals.contains(&context.terminal)
            || (context.output_observed && !context.replay_safe_after_output)
        {
            return RetryDecision::Stop;
        }
        RetryDecision::RetryAfterNs(self.backoff_ns(context.attempt_ordinal))
    }
}

/// One completed transport attempt plus its provider-safe terminal payload.
#[derive(Debug, Clone, PartialEq)]
pub struct AttemptExecution {
    /// Rust transport terminal.
    pub terminal: HostTerminalClass,
    /// Whether any output delta crossed the evaluator/proxy boundary.
    pub output_observed: bool,
    /// Whether endpoint policy classifies this terminal retryable.
    pub retryable: bool,
    /// Typed terminal payload; upstream raw SSE never appears here.
    pub payload: Value,
    /// Rust-authoritative usage for this concrete attempt.
    pub usage: HostOperationUsage,
}

/// Immutable lineage and accounting for one completed transport attempt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InferenceTransportAttempt {
    /// Fresh Rust-owned attempt identity under the logical operation.
    pub attempt_id: String,
    /// Zero-based attempt ordinal.
    pub ordinal: usize,
    /// Attempt terminal before any logical retry decision.
    pub terminal: HostTerminalClass,
    /// Whether typed output crossed the evaluator or local-proxy boundary.
    pub output_observed: bool,
    /// Rust-authoritative attempt usage.
    pub usage: HostOperationUsage,
}

/// One-attempt inference executor below retry policy.
#[async_trait(?Send)]
pub trait OneAttemptInference {
    /// Execute one fresh attempt and honor the shared cancellation latch.
    async fn execute_attempt(
        &self,
        operation_id: &str,
        attempt_id: &str,
        attempt_ordinal: usize,
        cancellation: OperationCancellation,
    ) -> Result<AttemptExecution>;
}

/// Final logical result after zero or more Rust transport retries.
#[derive(Debug, Clone, PartialEq)]
pub struct InferenceExecutionResult {
    /// Logical terminal returned to the evaluator.
    pub terminal: HostTerminalClass,
    /// Terminal typed payload.
    pub payload: Value,
    /// Number of concrete transport attempts.
    pub attempt_count: usize,
    /// Exact completed transport lineage in dispatch order.
    pub attempts: Vec<InferenceTransportAttempt>,
    /// Rust-authoritative usage aggregated across every transport attempt.
    pub usage: HostOperationUsage,
}

/// Object-safe operation executor above a one-attempt inference seam.
#[async_trait(?Send)]
pub trait InferenceAttemptExecutor {
    /// Execute one already-admitted logical operation to exactly one terminal.
    async fn execute(
        &self,
        operation_id: &str,
        replay_safe_after_output: bool,
        attempt: &dyn OneAttemptInference,
        cancellation: OperationCancellation,
    ) -> Result<InferenceExecutionResult>;
}

/// Clock-injected implementation of [`InferenceAttemptExecutor`].
pub struct ClockedInferenceAttemptExecutor {
    clock: Rc<dyn Clock>,
    retry_policy: Rc<dyn TransportRetryPolicy>,
}

impl ClockedInferenceAttemptExecutor {
    /// Compose one clock and one replaceable route retry policy.
    pub fn new(clock: Rc<dyn Clock>, retry_policy: Rc<dyn TransportRetryPolicy>) -> Self {
        Self {
            clock,
            retry_policy,
        }
    }
}

#[async_trait(?Send)]
impl InferenceAttemptExecutor for ClockedInferenceAttemptExecutor {
    async fn execute(
        &self,
        operation_id: &str,
        replay_safe_after_output: bool,
        attempt: &dyn OneAttemptInference,
        cancellation: OperationCancellation,
    ) -> Result<InferenceExecutionResult> {
        let mut ordinal = 0usize;
        let mut attempts = Vec::new();
        loop {
            if cancellation.is_cancelled() {
                let usage = aggregate_usage(&attempts)?;
                return Ok(InferenceExecutionResult {
                    terminal: HostTerminalClass::Cancelled,
                    payload: serde_json::json!({"status": "cancelled"}),
                    attempt_count: ordinal,
                    attempts,
                    usage,
                });
            }
            let attempt_id = format!("{operation_id}:transport:{ordinal}");
            let outcome = attempt
                .execute_attempt(operation_id, &attempt_id, ordinal, cancellation.clone())
                .await?;
            attempts.push(InferenceTransportAttempt {
                attempt_id,
                ordinal,
                terminal: outcome.terminal,
                output_observed: outcome.output_observed,
                usage: outcome.usage,
            });
            let decision = self.retry_policy.decide(RetryContext {
                attempt_ordinal: ordinal,
                terminal: outcome.terminal,
                output_observed: outcome.output_observed,
                retryable_hint: outcome.retryable,
                replay_safe_after_output,
            });
            if let RetryDecision::RetryAfterNs(backoff_ns) = decision {
                let sleep = self.clock.clone().sleep(backoff_ns);
                let cancelled = cancellation.cancelled();
                tokio::pin!(sleep);
                tokio::pin!(cancelled);
                tokio::select! {
                    _ = &mut sleep => {
                        ordinal = ordinal
                            .checked_add(1)
                            .ok_or_else(|| anyhow::anyhow!("transport attempt ordinal overflow"))?;
                        continue;
                    }
                    _ = &mut cancelled => {
                        let usage = aggregate_usage(&attempts)?;
                        return Ok(InferenceExecutionResult {
                            terminal: HostTerminalClass::Cancelled,
                            payload: serde_json::json!({"status": "cancelled"}),
                            attempt_count: ordinal.saturating_add(1),
                            attempts,
                            usage,
                        });
                    }
                }
            }
            let usage = aggregate_usage(&attempts)?;
            return Ok(InferenceExecutionResult {
                terminal: outcome.terminal,
                payload: outcome.payload,
                attempt_count: ordinal.saturating_add(1),
                attempts,
                usage,
            });
        }
    }
}

fn aggregate_usage(attempts: &[InferenceTransportAttempt]) -> Result<HostOperationUsage> {
    Ok(HostOperationUsage {
        prompt_tokens: sum_usage_field(attempts, |usage| usage.prompt_tokens)?,
        completion_tokens: sum_usage_field(attempts, |usage| usage.completion_tokens)?,
        reasoning_tokens: sum_usage_field(attempts, |usage| usage.reasoning_tokens)?,
        cached_tokens: sum_usage_field(attempts, |usage| usage.cached_tokens)?,
    })
}

fn sum_usage_field(
    attempts: &[InferenceTransportAttempt],
    field: impl Fn(HostOperationUsage) -> Option<u64>,
) -> Result<Option<u64>> {
    if attempts.is_empty() {
        return Ok(None);
    }
    attempts
        .iter()
        .map(|attempt| field(attempt.usage))
        .try_fold(Some(0_u64), |total, value| {
            Ok(match (total, value) {
                (Some(total), Some(value)) => Some(
                    total
                        .checked_add(value)
                        .ok_or_else(|| anyhow::anyhow!("transport attempt usage overflow"))?,
                ),
                _ => None,
            })
        })
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::collections::VecDeque;

    use aiperf_clock::SimClock;
    use aiperf_graph::runtime::drive_sim;

    use super::*;

    struct FixtureAttempts {
        outcomes: RefCell<VecDeque<AttemptExecution>>,
        starts: RefCell<Vec<(String, i64)>>,
        clock: Rc<SimClock>,
    }

    #[async_trait(?Send)]
    impl OneAttemptInference for FixtureAttempts {
        async fn execute_attempt(
            &self,
            _operation_id: &str,
            attempt_id: &str,
            _attempt_ordinal: usize,
            _cancellation: OperationCancellation,
        ) -> Result<AttemptExecution> {
            self.starts
                .borrow_mut()
                .push((attempt_id.to_string(), self.clock.now_ns()));
            Ok(self.outcomes.borrow_mut().pop_front().unwrap())
        }
    }

    #[test]
    fn retry_backoff_uses_exact_sim_clock_and_unique_lineage() {
        let clock = Rc::new(SimClock::new());
        let policy = Rc::new(
            ExponentialTransportRetryPolicy::new(3, 100, 1_000, [HostTerminalClass::Failed])
                .unwrap(),
        );
        let executor = Rc::new(ClockedInferenceAttemptExecutor::new(clock.clone(), policy));
        let attempts = Rc::new(FixtureAttempts {
            outcomes: RefCell::new(VecDeque::from([
                AttemptExecution {
                    terminal: HostTerminalClass::Failed,
                    output_observed: false,
                    retryable: true,
                    payload: serde_json::json!({"attempt": 0}),
                    usage: HostOperationUsage {
                        prompt_tokens: Some(2),
                        completion_tokens: None,
                        reasoning_tokens: None,
                        cached_tokens: Some(0),
                    },
                },
                AttemptExecution {
                    terminal: HostTerminalClass::Failed,
                    output_observed: false,
                    retryable: true,
                    payload: serde_json::json!({"attempt": 1}),
                    usage: HostOperationUsage {
                        prompt_tokens: Some(2),
                        completion_tokens: None,
                        reasoning_tokens: None,
                        cached_tokens: Some(0),
                    },
                },
                AttemptExecution {
                    terminal: HostTerminalClass::Completed,
                    output_observed: true,
                    retryable: false,
                    payload: serde_json::json!({"attempt": 2}),
                    usage: HostOperationUsage {
                        prompt_tokens: Some(2),
                        completion_tokens: Some(1),
                        reasoning_tokens: None,
                        cached_tokens: Some(0),
                    },
                },
            ])),
            starts: RefCell::new(Vec::new()),
            clock: clock.clone(),
        });
        let starts = attempts.clone();
        let result = Rc::new(RefCell::new(None));
        let result_for_run = result.clone();
        let outcome = drive_sim(clock, move |_handle| async move {
            let completed = executor
                .execute(
                    "operation",
                    false,
                    attempts.as_ref(),
                    OperationCancellation::default(),
                )
                .await
                .unwrap();
            *result_for_run.borrow_mut() = Some(completed);
        });
        assert!(!outcome.deadlocked);
        let result = result.borrow_mut().take().unwrap();
        assert_eq!(result.terminal, HostTerminalClass::Completed);
        assert_eq!(result.attempt_count, 3);
        assert_eq!(result.attempts.len(), 3);
        assert_eq!(result.usage.prompt_tokens, Some(6));
        assert_eq!(result.usage.completion_tokens, None);
        assert_eq!(
            starts.starts.borrow().as_slice(),
            &[
                ("operation:transport:0".into(), 0),
                ("operation:transport:1".into(), 100),
                ("operation:transport:2".into(), 300),
            ]
        );
    }

    #[test]
    fn observed_output_forbids_retry_when_replay_is_not_safe() {
        let clock = Rc::new(SimClock::new());
        let policy = Rc::new(
            ExponentialTransportRetryPolicy::new(3, 100, 1_000, [HostTerminalClass::Failed])
                .unwrap(),
        );
        let executor = Rc::new(ClockedInferenceAttemptExecutor::new(clock.clone(), policy));
        let attempts = Rc::new(FixtureAttempts {
            outcomes: RefCell::new(VecDeque::from([AttemptExecution {
                terminal: HostTerminalClass::Failed,
                output_observed: true,
                retryable: true,
                payload: serde_json::json!({"partial": true}),
                usage: HostOperationUsage::default(),
            }])),
            starts: RefCell::new(Vec::new()),
            clock: clock.clone(),
        });
        let starts = attempts.clone();
        let result = Rc::new(RefCell::new(None));
        let result_for_run = result.clone();
        let outcome = drive_sim(clock, move |_handle| async move {
            let completed = executor
                .execute(
                    "operation",
                    false,
                    attempts.as_ref(),
                    OperationCancellation::default(),
                )
                .await
                .unwrap();
            *result_for_run.borrow_mut() = Some(completed);
        });
        assert!(!outcome.deadlocked);
        assert_eq!(result.borrow_mut().take().unwrap().attempt_count, 1);
        assert_eq!(starts.starts.borrow().len(), 1);
    }

    #[test]
    fn cancellation_before_dispatch_creates_no_transport_attempt() {
        let clock = Rc::new(SimClock::new());
        let policy = Rc::new(
            ExponentialTransportRetryPolicy::new(3, 100, 1_000, [HostTerminalClass::Failed])
                .unwrap(),
        );
        let executor = Rc::new(ClockedInferenceAttemptExecutor::new(clock.clone(), policy));
        let attempts = Rc::new(FixtureAttempts {
            outcomes: RefCell::new(VecDeque::new()),
            starts: RefCell::new(Vec::new()),
            clock: clock.clone(),
        });
        let cancellation = OperationCancellation::default();
        cancellation.cancel();
        let result = Rc::new(RefCell::new(None));
        let result_for_run = result.clone();
        let outcome = drive_sim(clock, move |_handle| async move {
            let completed = executor
                .execute("operation", false, attempts.as_ref(), cancellation)
                .await
                .unwrap();
            *result_for_run.borrow_mut() = Some(completed);
        });
        assert!(!outcome.deadlocked);
        assert_eq!(result.borrow_mut().take().unwrap().attempt_count, 0);
    }
}
