// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Continuation-priority request-rate scheduling.
//!
//! This module realizes the continuation-priority request-rate policy on the
//! native [`Workload`] and [`ScheduledRuntime`] seams. One interval generator
//! tick admits at most one turn. A FIFO continuation queued by a returned request
//! always wins over the cached first turn of a new sampled session. New-session
//! admission is nonblocking, while a continuation may wait for prefill capacity
//! because it already owns a session slot.
//!
//! Session guards live from turn zero through the final return. Prefill guards
//! live from issuance through the first meaningful token, with terminal return
//! as the no-token fallback. Think time delays continuation queue insertion on
//! the injected [`Clock`](crate::clock::Clock), so the same workload remains
//! deterministic under `SimClock`. This linear policy owns only root
//! conversation chains.

use std::cell::RefCell;
use std::collections::{HashMap, VecDeque};
use std::rc::Rc;

use crate::timing::{
    ArrivalPattern, IntervalGenerator, SlotGuard, SlotPool, make_interval_generator,
};
use anyhow::{Result, anyhow, bail};
use async_trait::async_trait;
use tokio::sync::Notify;

use crate::dispatch::collector::ReplayTerminalStatus;
use crate::failure::OnFailure;
use crate::fixed_schedule::milliseconds_to_ns;
use crate::multiturn::{ConversationSource, TurnToSend};
use crate::scheduled::{ScheduledRuntime, Workload};
use crate::scheduler::LocalTaskScheduler;

/// Arrival and admission settings for a request-rate workload.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RequestRateConfig {
    /// Distribution used to draw one interval per turn.
    pub arrival_pattern: ArrivalPattern,
    /// Mean turns per second. Required except for `ConcurrencyBurst`.
    pub request_rate: Option<f64>,
    /// Gamma shape parameter; omitted means `1.0`.
    pub arrival_smoothness: Option<f64>,
    /// Maximum simultaneously active root conversations.
    pub session_concurrency: Option<usize>,
    /// Maximum requests waiting for their first meaningful token.
    pub prefill_concurrency: Option<usize>,
    /// Deterministic arrival-generator seed.
    pub seed: u64,
}

impl RequestRateConfig {
    fn validate(self) -> Result<Self> {
        if self.arrival_pattern != ArrivalPattern::ConcurrencyBurst
            && self
                .request_rate
                .is_none_or(|rate| !rate.is_finite() || rate <= 0.0)
        {
            bail!("request-rate workload requires a positive finite request rate");
        }
        if self.arrival_pattern == ArrivalPattern::Gamma
            && self
                .arrival_smoothness
                .is_some_and(|value| !value.is_finite() || value <= 0.0)
        {
            bail!("gamma arrival smoothness must be positive and finite");
        }
        if self.session_concurrency == Some(0) {
            bail!("request-rate session concurrency must be positive");
        }
        if self.prefill_concurrency == Some(0) {
            bail!("request-rate prefill concurrency must be positive");
        }
        Ok(self)
    }
}

#[derive(Default)]
struct RequestRateState {
    continuations: RefCell<VecDeque<TurnToSend>>,
    session_guards: RefCell<HashMap<String, SlotGuard>>,
    failure: RefCell<Option<String>>,
    progress: Notify,
}

impl RequestRateState {
    fn enqueue(&self, turn: TurnToSend) {
        self.continuations.borrow_mut().push_back(turn);
        self.progress.notify_one();
    }

    fn pop_continuation(&self) -> Option<TurnToSend> {
        self.continuations.borrow_mut().pop_front()
    }

    fn hold_session(&self, correlation_id: String, guard: SlotGuard) -> Result<()> {
        let mut guards = self.session_guards.borrow_mut();
        if guards.contains_key(&correlation_id) {
            bail!("duplicate active request-rate session {correlation_id:?}");
        }
        guards.insert(correlation_id, guard);
        Ok(())
    }

    fn release_session(&self, correlation_id: &str) {
        self.session_guards.borrow_mut().remove(correlation_id);
        self.progress.notify_one();
    }

    fn release_all_sessions(&self) {
        self.session_guards.borrow_mut().clear();
    }

    fn fail(&self, error: impl Into<String>) {
        let mut failure = self.failure.borrow_mut();
        if failure.is_none() {
            *failure = Some(error.into());
        }
    }

    fn has_failed(&self) -> bool {
        self.failure.borrow().is_some()
    }

    fn take_failure(&self) -> Option<String> {
        self.failure.borrow_mut().take()
    }
}

/// A single-loop, continuation-priority request-rate [`Workload`].
///
/// The conversation source remains behind its trait, and the interval and slot
/// handles are exposed for phase ramps and adaptive actuators. Constructing the
/// workload samples and caches exactly one first turn before run start; another
/// sample is drawn only after that cached turn is successfully issued.
pub struct RequestRateWorkload {
    conversations: Rc<RefCell<Box<dyn ConversationSource>>>,
    next_new_turn: RefCell<Option<TurnToSend>>,
    intervals: Rc<RefCell<Box<dyn IntervalGenerator>>>,
    session_slots: Option<Rc<SlotPool>>,
    prefill_slots: Option<Rc<SlotPool>>,
    state: Rc<RequestRateState>,
    on_failure: OnFailure,
}

impl RequestRateWorkload {
    /// Validate the source/configuration and cache the first sampled session.
    pub fn new(
        config: RequestRateConfig,
        conversations: Box<dyn ConversationSource>,
    ) -> Result<Self> {
        let config = config.validate()?;
        Self::with_components(
            conversations,
            Rc::new(RefCell::new(make_interval_generator(
                config.arrival_pattern,
                config.request_rate,
                config.arrival_smoothness,
                config.seed,
            ))),
            config
                .session_concurrency
                .map(|limit| Rc::new(SlotPool::new(limit))),
            config
                .prefill_concurrency
                .map(|limit| Rc::new(SlotPool::new(limit))),
        )
    }

    /// Build with injected arrival and admission components.
    ///
    /// Custom arrival distributions implement [`IntervalGenerator`] and enter
    /// here without extending an enum or changing the issuer loop. Supplying
    /// existing slot pools also lets a phase orchestrator own their lifecycle
    /// while this workload owns only acquire/release policy.
    pub fn with_components(
        mut conversations: Box<dyn ConversationSource>,
        intervals: Rc<RefCell<Box<dyn IntervalGenerator>>>,
        session_slots: Option<Rc<SlotPool>>,
        prefill_slots: Option<Rc<SlotPool>>,
    ) -> Result<Self> {
        if conversations.conversations().is_empty() {
            bail!("request-rate conversation dataset cannot be empty");
        }
        if let Some(empty) = conversations
            .conversations()
            .iter()
            .find(|conversation| conversation.turns.is_empty())
        {
            bail!(
                "request-rate conversation {:?} has no turns",
                empty.conversation_id
            );
        }
        let first = conversations.next(None)?.build_first_turn(None)?;
        Ok(Self {
            conversations: Rc::new(RefCell::new(conversations)),
            next_new_turn: RefCell::new(Some(first)),
            intervals,
            session_slots,
            prefill_slots,
            state: Rc::new(RequestRateState::default()),
            // Transport failures are recorded and issuance continues by default.
            on_failure: OnFailure::for_scheduled_default(),
        })
    }

    /// Select the run-failure discipline. `Abort` latches the whole run on the
    /// first `Failed` terminal (a real transport failure — cancellations and
    /// admission rejections never latch); `Continue` (the default) records the
    /// failed request and keeps issuing.
    pub fn with_failure_policy(mut self, on_failure: OnFailure) -> Self {
        self.on_failure = on_failure;
        self
    }

    /// Live interval generator used by the issuer and rate actuators.
    pub fn intervals(&self) -> Rc<RefCell<Box<dyn IntervalGenerator>>> {
        self.intervals.clone()
    }

    /// Session admission pool, when configured.
    pub fn session_slots(&self) -> Option<Rc<SlotPool>> {
        self.session_slots.clone()
    }

    /// Prefill admission pool, when configured.
    pub fn prefill_slots(&self) -> Option<Rc<SlotPool>> {
        self.prefill_slots.clone()
    }

    async fn issue_continuation(
        &self,
        runtime: Rc<ScheduledRuntime>,
        turn: TurnToSend,
        scheduled_ns: i64,
    ) -> bool {
        // Continuations may wait for prefill admission because turn zero already
        // holds the session slot.
        let prefill_guard = match &self.prefill_slots {
            Some(pool) => Some(pool.acquire().await),
            None => None,
        };
        if self.state.has_failed() || !runtime.can_issue(false) {
            drop(prefill_guard);
            self.state.release_session(&turn.x_correlation_id);
            return false;
        }
        issue_rate_turn(
            runtime,
            self.conversations.clone(),
            self.state.clone(),
            turn,
            scheduled_ns,
            prefill_guard,
            self.on_failure,
        )
    }

    fn try_issue_new_session(
        &self,
        runtime: Rc<ScheduledRuntime>,
        scheduled_ns: i64,
    ) -> Result<NewSessionOutcome> {
        if !runtime.can_issue(true) {
            return Ok(NewSessionOutcome::Stopped);
        }

        // New sessions use nonblocking admission. A failed prefill acquisition
        // drops the just-acquired session guard so this same cached sample can
        // retry on a later rate tick.
        let session_guard = match &self.session_slots {
            Some(pool) => match pool.try_acquire() {
                Some(guard) => Some(guard),
                None => return Ok(NewSessionOutcome::NoSlot),
            },
            None => None,
        };
        let prefill_guard = match &self.prefill_slots {
            Some(pool) => match pool.try_acquire() {
                Some(guard) => Some(guard),
                None => {
                    drop(session_guard);
                    return Ok(NewSessionOutcome::NoSlot);
                }
            },
            None => None,
        };
        if !runtime.can_issue(true) {
            drop(prefill_guard);
            drop(session_guard);
            return Ok(NewSessionOutcome::Stopped);
        }

        let turn = self
            .next_new_turn
            .borrow_mut()
            .take()
            .ok_or_else(|| anyhow!("request-rate issuer lost its cached new session"))?;
        if let Some(guard) = session_guard {
            self.state
                .hold_session(turn.x_correlation_id.clone(), guard)?;
        }
        let correlation_id = turn.x_correlation_id.clone();
        if !issue_rate_turn(
            runtime,
            self.conversations.clone(),
            self.state.clone(),
            turn,
            scheduled_ns,
            prefill_guard,
            self.on_failure,
        ) {
            self.state.release_session(&correlation_id);
            return Ok(NewSessionOutcome::Stopped);
        }

        // Cache the next sample only after successful issuance. Sequential and
        // shuffle samplers therefore do not advance on a skipped interval.
        let next = self
            .conversations
            .borrow_mut()
            .next(None)?
            .build_first_turn(None)?;
        *self.next_new_turn.borrow_mut() = Some(next);
        Ok(NewSessionOutcome::Issued)
    }

    async fn wait_for_closed_loop_progress(&self) {
        self.state.progress.notified().await;
    }
}

enum NewSessionOutcome {
    Issued,
    NoSlot,
    Stopped,
}

#[async_trait(?Send)]
impl Workload for RequestRateWorkload {
    fn name(&self) -> &'static str {
        "request_rate"
    }

    async fn execute(&self, runtime: Rc<ScheduledRuntime>) -> Result<()> {
        // Arrival pacing is the (AfterInterval, Reanchor) policy named in
        // `crate::timing::arrival`: the first arrival is one interval in and a target
        // that has fallen behind re-anchors to `now` with no catch-up burst. This
        // loop keeps its own arithmetic rather than calling `next_arrival_target`
        // because it *eagerly* advances and peeks the next target for closed-loop
        // backpressure (the `NoSlot` block-vs-yield decision below), which the shared
        // pure helper does not express.
        let mut next_target_ns = runtime
            .start_ns()
            .saturating_add(self.intervals.borrow_mut().next_interval_ns());

        loop {
            if self.state.has_failed() || !runtime.can_issue(false) {
                break;
            }

            // Absolute pacing with no catch-up burst. The interval for the
            // following tick is drawn before this tick attempts admission.
            let now_ns = runtime.now_ns();
            if next_target_ns < now_ns {
                next_target_ns = now_ns;
            }
            let scheduled_ns = next_target_ns;
            if scheduled_ns > now_ns {
                if !runtime.wait_until_or_stop(scheduled_ns).await {
                    break;
                }
            } else {
                // Zero/rounded-zero intervals must yield so dispatch returns can
                // enqueue continuations and release slots.
                tokio::task::yield_now().await;
            }
            next_target_ns =
                scheduled_ns.saturating_add(self.intervals.borrow_mut().next_interval_ns());

            if self.state.has_failed() || !runtime.can_issue(false) {
                break;
            }

            if let Some(turn) = self.state.pop_continuation() {
                if !self
                    .issue_continuation(runtime.clone(), turn, scheduled_ns)
                    .await
                    && !runtime.can_issue(false)
                {
                    break;
                }
                continue;
            }

            if runtime.can_issue(true) {
                match self.try_issue_new_session(runtime.clone(), scheduled_ns) {
                    Ok(NewSessionOutcome::Issued) => {}
                    Ok(NewSessionOutcome::NoSlot) => {
                        // A `Global`-backed session pool (`global`/`global-hop`
                        // dispatch) may next free a slot on a DIFFERENT worker
                        // thread's release, which never fires this thread's own
                        // `state.progress` `Notify` (only THIS thread's own
                        // `enqueue`/`release_session` calls do). Blocking on it
                        // here would deadlock a thread holding zero local
                        // guards forever, so fall through to the yield-and-retry
                        // path instead — see `SlotPool::is_global`'s doc comment.
                        let session_pool_is_global = self
                            .session_slots
                            .as_ref()
                            .is_some_and(|pool| pool.is_global());
                        if next_target_ns <= runtime.now_ns() && !session_pool_is_global {
                            // A session slot stays held until its continuation
                            // completes, so waiting on that slot here can hide
                            // the queued continuation that must release it.
                            self.wait_for_closed_loop_progress().await;
                        } else {
                            // Paced modes preserve the nonblocking skipped-tick
                            // behavior and retry at the next authored arrival.
                            tokio::task::yield_now().await;
                        }
                    }
                    Ok(NewSessionOutcome::Stopped) => {
                        if !runtime.can_issue(false) {
                            break;
                        }
                    }
                    Err(error) => self.state.fail(format!(
                        "request-rate new-session issuance failed: {error:#}"
                    )),
                }
            } else if !runtime.can_issue(false) {
                break;
            } else {
                // The session quota/cap is full, but returned requests may still
                // produce continuations. Consume this tick without busy-spinning.
                tokio::task::yield_now().await;
            }
        }

        runtime.scheduler().cancel_pending();
        runtime.scheduler().wait_idle().await;
        self.state.release_all_sessions();
        if let Some(error) = self.state.take_failure() {
            bail!(error);
        }
        Ok(())
    }
}

#[allow(clippy::too_many_arguments)]
fn issue_rate_turn(
    runtime: Rc<ScheduledRuntime>,
    conversations: Rc<RefCell<Box<dyn ConversationSource>>>,
    state: Rc<RequestRateState>,
    turn: TurnToSend,
    scheduled_ns: i64,
    prefill_guard: Option<SlotGuard>,
    on_failure: OnFailure,
) -> bool {
    let correlation_id = turn.x_correlation_id.clone();
    let prefill_guard = Rc::new(RefCell::new(prefill_guard));
    let prefill_for_first_token = prefill_guard.clone();
    let prefill_for_terminal = prefill_guard.clone();
    let runtime_for_completion = runtime.clone();
    let state_for_completion = state.clone();
    let issued = runtime.issue_turn_with_hooks(
        turn,
        scheduled_ns,
        None,
        Box::new(move |_ttft_ns| {
            prefill_for_first_token.borrow_mut().take();
        }),
        Box::new(move |credit, outcome| {
            Box::pin(async move {
                // Terminal is the fallback for errors, cancellation, or an empty
                // response that never produced the first-token callback.
                prefill_for_terminal.borrow_mut().take();
                let correlation_id = credit.turn.x_correlation_id.clone();
                // Abort policy latches transport failures only; cancellations and
                // admission rejections remain ordinary terminal outcomes.
                if on_failure.is_abort()
                    && matches!(outcome.terminal, ReplayTerminalStatus::Failed)
                {
                    state_for_completion.fail(format!(
                        "request-rate turn {correlation_id:?} failed under abort-on-failure policy"
                    ));
                    state_for_completion.release_session(&correlation_id);
                    return;
                }
                if credit.is_final_turn() {
                    state_for_completion.release_session(&correlation_id);
                    return;
                }
                if state_for_completion.has_failed() || !runtime_for_completion.can_issue(false) {
                    state_for_completion.release_session(&correlation_id);
                    return;
                }

                let next_turn = match conversations
                    .borrow()
                    .next_turn(&credit, outcome.to_turn_response())
                {
                    Ok(Some(turn)) => turn,
                    Ok(None) => {
                        state_for_completion.release_session(&correlation_id);
                        return;
                    }
                    Err(error) => {
                        state_for_completion.fail(format!(
                            "request-rate continuation materialization failed for {correlation_id:?}: {error:#}"
                        ));
                        state_for_completion.release_session(&correlation_id);
                        return;
                    }
                };

                let delay_ns = match next_turn.delay_ms {
                    Some(delay_ms) => match milliseconds_to_ns(delay_ms) {
                        Ok(delay_ns) if delay_ns >= 0 => delay_ns,
                        Ok(_) => {
                            state_for_completion.fail(format!(
                                "request-rate continuation {correlation_id:?} has negative think time"
                            ));
                            state_for_completion.release_session(&correlation_id);
                            return;
                        }
                        Err(error) => {
                            state_for_completion.fail(format!(
                                "request-rate continuation {correlation_id:?} has invalid think time: {error:#}"
                            ));
                            state_for_completion.release_session(&correlation_id);
                            return;
                        }
                    },
                    None => 0,
                };
                if delay_ns == 0 {
                    state_for_completion.enqueue(next_turn);
                } else {
                    let state_for_delay = state_for_completion.clone();
                    runtime_for_completion.scheduler().schedule_later(
                        delay_ns,
                        Box::pin(async move {
                            if !state_for_delay.has_failed() {
                                state_for_delay.enqueue(next_turn);
                            }
                        }),
                    );
                }
            })
        }),
    );
    if !issued {
        prefill_guard.borrow_mut().take();
        state.release_session(&correlation_id);
    }
    issued
}

#[cfg(test)]
mod tests {
    use std::rc::Rc;

    use crate::clock::{Clock, SimClock};
    use crate::dispatch::collector::ReplayTerminalStatus;
    use crate::dispatch::sink::RequestObserver;
    use crate::graph::runtime::drive_sim;
    use crate::timing::{ArrivalPattern, StopConfig};
    use anyhow::Result;
    use async_trait::async_trait;

    use super::*;
    use crate::scheduled::{ModelResponseMetadata, TurnDispatchOutcome, TurnDispatcher};
    use crate::test_util::synthetic_prepared_source;

    struct DelayedDispatcher {
        clock: Rc<dyn Clock>,
    }

    #[async_trait(?Send)]
    impl TurnDispatcher for DelayedDispatcher {
        async fn dispatch_turn(
            &self,
            turn: TurnToSend,
            observer: &dyn RequestObserver,
            _on_first_token: &dyn Fn(i64),
        ) -> Result<TurnDispatchOutcome> {
            self.clock.clone().sleep(1).await;
            observer.on_terminal(turn.uuid, ReplayTerminalStatus::Completed);
            Ok(TurnDispatchOutcome {
                start_ns: self.clock.now_ns().saturating_sub(1),
                end_ns: self.clock.now_ns(),
                terminal: ReplayTerminalStatus::Completed,
                response_text: "answer".into(),
                model_response: ModelResponseMetadata::default(),
                prompt_tokens: None,
                completion_tokens: None,
                http: Default::default(),
            })
        }
    }

    #[test]
    fn closed_loop_issues_continuations_when_session_slot_is_full() {
        // Build the native dataset source on a throwaway runtime before the
        // deterministic sim driver takes over the executor.
        let source = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(synthetic_prepared_source(2, 8, 4, Some(0), "m"));
        let clock = Rc::new(SimClock::new());
        let runtime_clock: Rc<dyn Clock> = clock.clone();
        let outcome = drive_sim(clock, move |_handle| async move {
            let workload = RequestRateWorkload::new(
                RequestRateConfig {
                    arrival_pattern: ArrivalPattern::ConcurrencyBurst,
                    request_rate: None,
                    arrival_smoothness: None,
                    session_concurrency: Some(1),
                    prefill_concurrency: None,
                    seed: 0,
                },
                source,
            )
            .unwrap();
            let runtime = ScheduledRuntime::new(
                runtime_clock.clone(),
                0,
                Rc::new(DelayedDispatcher {
                    clock: runtime_clock,
                }),
                StopConfig {
                    total_expected_requests: Some(4),
                    ..StopConfig::default()
                },
                true,
            );

            workload.execute(runtime).await.unwrap();
        });

        assert!(!outcome.deadlocked);
    }

    /// Dispatcher whose every turn ends in the scripted way, so failure-policy
    /// behavior can be exercised deterministically under `SimClock`.
    enum ScriptedResult {
        /// Real transport failure: `dispatch_turn` returns `Err`. The runtime
        /// synthesizes a `Failed` terminal (the resilient path) before invoking
        /// the completion hook.
        TransportError,
        /// A terminal outcome the dispatcher reports as `Ok` (e.g. a
        /// cancellation), so the hook sees the exact `ReplayTerminalStatus`.
        Terminal(ReplayTerminalStatus),
    }

    struct ScriptedDispatcher {
        clock: Rc<dyn Clock>,
        result: ScriptedResult,
    }

    #[async_trait(?Send)]
    impl TurnDispatcher for ScriptedDispatcher {
        async fn dispatch_turn(
            &self,
            turn: TurnToSend,
            observer: &dyn RequestObserver,
            _on_first_token: &dyn Fn(i64),
        ) -> Result<TurnDispatchOutcome> {
            self.clock.clone().sleep(1).await;
            match &self.result {
                ScriptedResult::TransportError => {
                    Err(anyhow!("scripted transport failure for {:?}", turn.uuid))
                }
                ScriptedResult::Terminal(terminal) => {
                    observer.on_terminal(turn.uuid, *terminal);
                    Ok(TurnDispatchOutcome {
                        start_ns: self.clock.now_ns().saturating_sub(1),
                        end_ns: self.clock.now_ns(),
                        terminal: *terminal,
                        response_text: String::new(),
                        model_response: ModelResponseMetadata::default(),
                        prompt_tokens: None,
                        completion_tokens: None,
                        http: Default::default(),
                    })
                }
            }
        }
    }

    fn run_single_turn_with_policy(result: ScriptedResult, on_failure: OnFailure) -> Result<()> {
        let source = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(synthetic_prepared_source(1, 8, 4, Some(0), "m"));
        let clock = Rc::new(SimClock::new());
        let runtime_clock: Rc<dyn Clock> = clock.clone();
        let mut captured: Option<Result<()>> = None;
        let captured_slot = &mut captured;
        drive_sim(clock, move |_handle| {
            let source = source;
            async move {
                let workload = RequestRateWorkload::new(
                    RequestRateConfig {
                        arrival_pattern: ArrivalPattern::ConcurrencyBurst,
                        request_rate: None,
                        arrival_smoothness: None,
                        session_concurrency: Some(1),
                        prefill_concurrency: None,
                        seed: 0,
                    },
                    source,
                )
                .unwrap()
                .with_failure_policy(on_failure);
                let runtime = ScheduledRuntime::new(
                    runtime_clock.clone(),
                    0,
                    Rc::new(ScriptedDispatcher {
                        clock: runtime_clock,
                        result,
                    }),
                    StopConfig {
                        total_expected_requests: Some(3),
                        ..StopConfig::default()
                    },
                    true,
                );
                *captured_slot = Some(workload.execute(runtime).await);
            }
        });
        captured.expect("workload future ran to completion")
    }

    #[test]
    fn abort_policy_latches_run_on_transport_failure() {
        let result = run_single_turn_with_policy(ScriptedResult::TransportError, OnFailure::Abort);
        assert!(
            result.is_err(),
            "abort-on-failure must bail the run on the first transport failure"
        );
    }

    #[test]
    fn continue_policy_records_failure_and_run_succeeds() {
        let result =
            run_single_turn_with_policy(ScriptedResult::TransportError, OnFailure::Continue);
        assert!(
            result.is_ok(),
            "resilient policy records failed requests and completes the run"
        );
    }

    #[test]
    fn abort_policy_ignores_cancellation() {
        let result = run_single_turn_with_policy(
            ScriptedResult::Terminal(ReplayTerminalStatus::Canceled),
            OnFailure::Abort,
        );
        assert!(
            result.is_ok(),
            "cancellation is an authored outcome and must not latch abort-on-failure"
        );
    }
}
