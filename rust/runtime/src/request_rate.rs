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

use std::cell::{Cell, RefCell};
use std::collections::{HashMap, VecDeque};
use std::rc::Rc;
use std::sync::Arc;

use crate::timing::{
    ArrivalPattern, GlobalRateGate, IntervalGenerator, SlotGuard, SlotPool, make_interval_generator,
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
    /// When set (`global`/`global-hop` dispatch on a rate phase), arrival pacing
    /// draws its base fire time from this cell-shared gate instead of this
    /// thread's local `intervals`, so aggregate issuance across all `W` worker
    /// threads matches one global rate exactly.
    /// `intervals` is still consulted for the mean-zero Poisson/Gamma jitter
    /// offset added to each claimed base slot.
    rate_gate: Option<Arc<GlobalRateGate>>,
    /// Whether each new session is drawn at the corpus position of the rate slot
    /// it will be admitted on, instead of from the source's own draw cursor.
    ///
    /// Selected by [`Self::with_rate_gate`] when the phase can support it; see
    /// [`Self::can_address_draws_by_slot`].
    slot_addressed_draws: bool,
    /// Base-grid offset of the slot the cached `next_new_turn` was drawn for.
    /// `Some` exactly when `slot_addressed_draws` holds and a sample is cached,
    /// so a tick that cannot admit retries the SAME slot and position rather
    /// than abandoning the position it already claimed.
    pending_slot_offset_ns: Cell<Option<i64>>,
    state: Rc<RequestRateState>,
    on_failure: OnFailure,
    /// Whether this phase routes credits whose request body the WORKER builds
    /// (`--dispatch global-push`), so the issuer does not pay materialization on
    /// the one thread that bounds that mode.
    ///
    /// Applied to SINGLE-TURN sessions only. A continuation's body can splice
    /// the live model reply, which the worker's replay of
    /// `build_turn_at` cannot reproduce, so a multi-turn session keeps
    /// issuer-side materialization and stays byte-identical.
    defer_single_turn_bodies: bool,
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
        // Both checks read `conversations()` — this shard's residue class — even
        // though a position-addressed shard DRAWS the full corpus through
        // `next_at_position`. That narrower basis is exact, not an oversight, so
        // do not widen it to `sampled_conversations()`:
        //
        // - A turn-less conversation cannot reach any source. `Dataset::new`
        //   rejects one for EVERY conversation at construction, before a
        //   partition exists (`dataset/dataset.rs`, `validate_conversation_handles`),
        //   so no residue can hide a row the full corpus would have caught.
        // - An empty residue cannot reach here either: a partitioned source bails
        //   with "partition owns no sampleable sessions" and an unpartitioned one
        //   bails with `DatasetError::EmptySampler`, both inside the source
        //   constructor. This bail guards a non-native `ConversationSource`, whose
        //   default `sampled_conversations()` is `conversations()` anyway.
        //
        // So a widened scan could not reject anything the narrow one accepts.
        // It is declined because it is redundant, not because of what it would
        // cost: `can_address_draws_by_slot` below runs exactly that full-corpus
        // scan per shard, where the answer is not decidable any other way.
        //
        // The invariant the narrow basis rests on is pinned by
        // `multiturn::tests::request_rate_zero_turn_validation_cannot_miss_a_row_outside_the_residue`.
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
            rate_gate: None,
            slot_addressed_draws: false,
            pending_slot_offset_ns: Cell::new(None),
            state: Rc::new(RequestRateState::default()),
            // Transport failures are recorded and issuance continues by default.
            on_failure: OnFailure::for_scheduled_default(),
            defer_single_turn_bodies: false,
        })
    }

    /// Route single-turn credits without materializing their bodies, leaving
    /// that to the worker (`--dispatch global-push`).
    pub fn with_deferred_single_turn_bodies(mut self, defer: bool) -> Self {
        self.defer_single_turn_bodies = defer;
        // The cached first sample was built eagerly by the constructor; rebuild
        // it under the new policy so the very first issued turn is deferred too.
        if defer
            && let Some(cached) = self.next_new_turn.borrow_mut().as_mut()
            && cached.num_turns == 1
            && let Ok(deferred) = cached.session_handle().build_deferred_turn(0, None)
        {
            *cached = deferred;
        }
        self
    }

    /// Build the next new session's turn under this phase's materialization
    /// policy: deferred when the session is single-turn and the worker will
    /// build it, eagerly materialized otherwise.
    fn build_new_session_turn(
        &self,
        sampled: &crate::multiturn::SampledSession,
    ) -> Result<TurnToSend> {
        if self.defer_single_turn_bodies && sampled.available_turns() == 1 {
            sampled.build_deferred_turn(0, None)
        } else {
            sampled.build_first_turn(None)
        }
    }

    /// Select the run-failure discipline. `Abort` latches the whole run on the
    /// first `Failed` terminal (a real transport failure — cancellations and
    /// admission rejections never latch); `Continue` (the default) records the
    /// failed request and keeps issuing.
    pub fn with_failure_policy(mut self, on_failure: OnFailure) -> Self {
        self.on_failure = on_failure;
        self
    }

    /// Attach a cell-shared [`GlobalRateGate`] so this worker paces against the
    /// single global request rate rather than its own local `intervals` grid.
    ///
    /// Passing `None` (the default) leaves per-thread local pacing unchanged —
    /// the `Sharded`-dispatch path, and any run that authored no `rate`. When a
    /// gate is attached, every worker thread in the cell shares the same
    /// `Arc<GlobalRateGate>`; each `execute` tick claims one distinct base slot
    /// from it, so the `W` threads together emit exactly the configured global
    /// rate instead of `W` independent full-rate streams.
    /// A gate also carries the only cell-wide ordering this phase has, so
    /// attaching one selects slot-addressed draws when the phase supports them
    /// ([`Self::can_address_draws_by_slot`]): the conversation admitted on slot
    /// `n` becomes corpus position `n`, so sorting records by their admission
    /// time recovers the sequential corpus walk.
    pub fn with_rate_gate(mut self, rate_gate: Option<Arc<GlobalRateGate>>) -> Self {
        self.slot_addressed_draws = rate_gate.is_some() && self.can_address_draws_by_slot();
        if self.slot_addressed_draws {
            // The constructor's cached sample came from the source's own cursor
            // with no claimed slot behind it. Drop it so the first tick draws at
            // the slot it claims; `slot_addressed_offset_ns` refills the cache
            // before any tick can consume it.
            *self.next_new_turn.borrow_mut() = None;
            self.pending_slot_offset_ns.set(None);
        }
        self.rate_gate = rate_gate;
        self
    }

    /// Whether every new-session draw can be tied to the rate slot it is
    /// admitted on.
    ///
    /// Two properties are required, and both are decidable here:
    ///
    /// - The source addresses absolute corpus positions (`workers > 1` under
    ///   `global` dispatch with a closed-form sampler). Otherwise a position is
    ///   not a conversation.
    /// - Every sampleable conversation is single-turn. A multi-turn corpus
    ///   admits continuations, and a continuation consumes a rate slot without
    ///   drawing a new conversation — so slot indices would stop enumerating the
    ///   corpus densely and conversations at the skipped positions would never
    ///   be drawn at all.
    fn can_address_draws_by_slot(&self) -> bool {
        let conversations = self.conversations.borrow();
        conversations.is_position_addressed()
            && conversations
                .sampled_conversations()
                .iter()
                .all(|conversation| conversation.turns.len() == 1)
    }

    /// Base-grid offset of the slot the next new session is drawn for, claiming
    /// a fresh slot — and drawing at its corpus position — when none is pending.
    ///
    /// The claim and the draw happen together so `admit_ns` order and corpus
    /// order are two pure functions of one `fetch_add`. A pending slot is
    /// retained across a tick that could not admit, which keeps the claimed
    /// position sequence dense: nothing is drawn and then thrown away.
    fn slot_addressed_offset_ns(&self, gate: &Arc<GlobalRateGate>) -> Result<i64> {
        if let Some(offset_ns) = self.pending_slot_offset_ns.get() {
            return Ok(offset_ns);
        }
        let slot = gate.claim_slot();
        let sampled = self
            .conversations
            .borrow_mut()
            .next_at_position(slot.index, None)?;
        let turn = self.build_new_session_turn(&sampled)?;
        *self.next_new_turn.borrow_mut() = Some(turn);
        self.pending_slot_offset_ns.set(Some(slot.offset_ns));
        Ok(slot.offset_ns)
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
        // The claimed slot is consumed with the sample it was drawn for; the
        // next tick claims a fresh one. Every early return above happens before
        // the take, so a tick that could not admit leaves both in place.
        self.pending_slot_offset_ns.set(None);
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
        // A slot-addressed phase instead draws at the next tick's claimed slot,
        // so pre-drawing here would materialize a body that tick discards.
        if !self.slot_addressed_draws {
            let sampled = self.conversations.borrow_mut().next(None)?;
            let next = self.build_new_session_turn(&sampled)?;
            *self.next_new_turn.borrow_mut() = Some(next);
        }
        Ok(NewSessionOutcome::Issued)
    }

    async fn wait_for_closed_loop_progress(&self) {
        self.state.progress.notified().await;
    }

    /// Consume one saturated tick without starving a virtual clock.
    ///
    /// Under a real clock a bare yield is correct: wall time advances on its own,
    /// so retrying costs a task reschedule and an in-flight response eventually
    /// frees a slot. Under [`SimClock`](crate::clock::SimClock) nothing advances
    /// virtual time unless a task is parked on a clock timer, and a self-waking
    /// yield makes the virtual driver re-poll the same instant forever. Park on
    /// the clock instead, racing this thread's own progress notification, so the
    /// driver advances to the earliest pending event and the slot is released.
    async fn wait_for_capacity(&self, runtime: &ScheduledRuntime) {
        let clock = runtime.clock();
        if !clock.is_virtual() {
            tokio::task::yield_now().await;
            return;
        }
        let progress = self.state.progress.notified();
        let idle = clock.sleep(VIRTUAL_IDLE_HORIZON_NS);
        tokio::select! {
            biased;
            () = progress => {}
            () = idle => {}
        }
    }
}

/// Virtual-time horizon for a saturated retry when no event is pending.
///
/// Any in-flight request's own clock event is earlier than this, so the horizon
/// only bounds a fully idle wait; it never delays a release that is already
/// scheduled.
const VIRTUAL_IDLE_HORIZON_NS: i64 = 1_000_000_000;

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
            let scheduled_ns = if let Some(gate) = &self.rate_gate {
                // Global/global-hop dispatch: claim one distinct base slot from
                // the cell-shared gate (evenly-spaced across all worker threads,
                // so their union is exactly the global rate grid). Anchor it to
                // phase start plus one interval — matching the local path's
                // "first arrival is one interval in" — and add a mean-zero
                // jitter offset from this thread's own generator. This keeps
                // the aggregate rate exact regardless of jitter, but is a
                // bounded-scatter approximation, not a reproduction of
                // Poisson/Gamma arrival-process statistics (see
                // `GlobalRateGate`'s module doc) — full arrival-pattern parity
                // for jittered phases is `global-hop`'s job. No per-thread
                // re-anchor: a claimed slot already in the past pages through
                // via the `scheduled_ns <= now` yield path below.
                let jitter_ns = self
                    .intervals
                    .borrow_mut()
                    .next_interval_ns()
                    .saturating_sub(gate.interval_ns());
                // A slot-addressed phase claims the slot AND draws this tick's
                // conversation at that slot's corpus position together, so
                // `admit_ns` order and corpus order coincide by construction.
                // Nonzero jitter still reorders adjacent slots, which is why
                // exact admission order holds for `constant` arrival only.
                let base_offset_ns = if self.slot_addressed_draws {
                    match self.slot_addressed_offset_ns(gate) {
                        Ok(offset_ns) => offset_ns,
                        Err(error) => {
                            self.state.fail(format!(
                                "request-rate slot-addressed draw failed: {error:#}"
                            ));
                            break;
                        }
                    }
                } else {
                    gate.claim_offset_ns()
                };
                runtime
                    .start_ns()
                    .saturating_add(gate.interval_ns())
                    .saturating_add(base_offset_ns)
                    .saturating_add(jitter_ns)
            } else {
                if next_target_ns < now_ns {
                    next_target_ns = now_ns;
                }
                next_target_ns
            };
            if scheduled_ns > now_ns {
                if !runtime.wait_until_or_stop(scheduled_ns).await {
                    break;
                }
            } else {
                // Zero/rounded-zero intervals must yield so dispatch returns can
                // enqueue continuations and release slots. This tick has not yet
                // attempted admission, so it stays a bare yield: parking on the
                // clock here would delay an arrival that capacity can accept.
                tokio::task::yield_now().await;
            }
            if self.rate_gate.is_none() {
                next_target_ns =
                    scheduled_ns.saturating_add(self.intervals.borrow_mut().next_interval_ns());
            }

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
                        let is_saturated_now = next_target_ns <= runtime.now_ns();
                        if is_saturated_now && !session_pool_is_global {
                            // A session slot stays held until its continuation
                            // completes, so waiting on that slot here can hide
                            // the queued continuation that must release it.
                            self.wait_for_closed_loop_progress().await;
                        } else if is_saturated_now {
                            // Global pool, no authored arrival left to pace
                            // against: a bare yield would spin, and under a
                            // virtual clock it would freeze time outright.
                            self.wait_for_capacity(&runtime).await;
                        } else {
                            // Paced modes preserve the nonblocking skipped-tick
                            // behavior and retry at the next authored arrival,
                            // which is itself the clock event that advances time.
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
                if next_target_ns <= runtime.now_ns() {
                    self.wait_for_capacity(&runtime).await;
                } else {
                    tokio::task::yield_now().await;
                }
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

    /// Records `(admission time, conversation id)` for every dispatched turn
    /// into one shared log, so a multi-worker cell's admission ORDER is
    /// observable without a real transport.
    struct OrderRecordingDispatcher {
        clock: Rc<dyn Clock>,
        admissions: Rc<RefCell<Vec<(i64, String)>>>,
    }

    #[async_trait(?Send)]
    impl TurnDispatcher for OrderRecordingDispatcher {
        async fn dispatch_turn(
            &self,
            turn: TurnToSend,
            observer: &dyn RequestObserver,
            _on_first_token: &dyn Fn(i64),
        ) -> Result<TurnDispatchOutcome> {
            // Under `SimClock` no virtual time passes between the issuer's
            // `now_ns()` admission stamp and this entry, so this reading IS the
            // record's `admit_ns`.
            self.admissions
                .borrow_mut()
                .push((self.clock.now_ns(), turn.conversation_id.clone()));
            self.clock.clone().sleep(1).await;
            observer.on_terminal(turn.uuid, ReplayTerminalStatus::Completed);
            Ok(TurnDispatchOutcome {
                start_ns: self.clock.now_ns().saturating_sub(1),
                end_ns: self.clock.now_ns(),
                terminal: ReplayTerminalStatus::Completed,
                response_text: String::new(),
                model_response: ModelResponseMetadata::default(),
                prompt_tokens: None,
                completion_tokens: None,
                http: Default::default(),
            })
        }
    }

    /// `Global` at `workers > 1` on a rate-paced phase must admit conversations
    /// in sequential corpus order: sorting the run's records by admission time
    /// reproduces the single issuer's walk `ids[k % entries]`.
    ///
    /// Every worker shares one [`GlobalRateGate`], so a worker's fire time and
    /// the conversation it draws come from the same claim. The fixture is
    /// deliberately awkward for a static per-worker partition: `5 % 4 == 1`, so
    /// the residue classes are uneven, and 12 requests over 5 conversations is
    /// 2.4 passes, so the final pass is partial.
    ///
    /// The workers are started in DESCENDING cell-id order, which is what makes
    /// this test discriminating. Which worker wins a given claim is racy — a
    /// live 16-worker rig run shows one worker taking four slots before the
    /// others start — and nothing depends on the assignment. But a static
    /// per-worker residue walk only agrees with the claim order while workers
    /// happen to claim in cell-id order; start them in any other order and the
    /// two sequences separate immediately, which is precisely the defect.
    ///
    /// `SimClock` makes the assertion exact — a worker wakes at precisely its
    /// claimed slot. On a real clock the same tie holds on the claim, but
    /// per-thread wake-up skew scatters stamps within roughly one wake interval
    /// of their slot.
    #[test]
    fn global_rate_paced_admission_order_is_the_sequential_corpus_walk() {
        const WORKERS: u32 = 4;
        const ENTRIES: usize = 5;
        const PER_WORKER_REQUESTS: u64 = 3;
        const TOTAL_REQUESTS: usize = WORKERS as usize * PER_WORKER_REQUESTS as usize;

        let boot = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        // Descending cell id: the start order, and hence the claim order, is
        // deliberately not the partition order.
        let sources: Vec<Box<dyn ConversationSource>> = (0..WORKERS)
            .rev()
            .map(|cell_id| {
                boot.block_on(crate::test_util::partitioned_single_turn_source(
                    ENTRIES, cell_id, WORKERS, "m",
                ))
            })
            .collect();

        let clock = Rc::new(SimClock::new());
        let runtime_clock: Rc<dyn Clock> = clock.clone();
        let admissions = Rc::new(RefCell::new(Vec::new()));
        let recorded = admissions.clone();
        // One shared gate across the cell, exactly as `GlobalAdmission` hands it
        // to every worker thread of a `global`-dispatch phase.
        let gate = GlobalRateGate::new(1_000.0);
        drive_sim(clock, move |_handle| async move {
            let mut running = Vec::new();
            for source in sources {
                let workload = RequestRateWorkload::with_components(
                    source,
                    Rc::new(RefCell::new(make_interval_generator(
                        ArrivalPattern::Constant,
                        Some(1_000.0),
                        None,
                        0,
                    ))),
                    None,
                    None,
                )
                .unwrap()
                .with_rate_gate(Some(gate.clone()));
                let runtime = ScheduledRuntime::new(
                    runtime_clock.clone(),
                    0,
                    Rc::new(OrderRecordingDispatcher {
                        clock: runtime_clock.clone(),
                        admissions: recorded.clone(),
                    }),
                    StopConfig {
                        total_expected_requests: Some(PER_WORKER_REQUESTS),
                        ..StopConfig::default()
                    },
                    true,
                );
                running.push(tokio::task::spawn_local(async move {
                    workload.execute(runtime).await
                }));
            }
            for task in running {
                task.await.unwrap().unwrap();
            }
        });

        let mut observed = admissions.borrow().clone();
        assert_eq!(
            observed.len(),
            TOTAL_REQUESTS,
            "every worker must issue its whole request budget"
        );
        observed.sort_by_key(|(admit_ns, _)| *admit_ns);
        let by_admission: Vec<String> = observed.into_iter().map(|(_, id)| id).collect();
        let expected: Vec<String> = (0..TOTAL_REQUESTS)
            .map(|position| format!("session_{:04}", position % ENTRIES))
            .collect();
        assert_eq!(
            by_admission, expected,
            "sorting a global rate-paced run by admission time must give the \
             sequential corpus walk"
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
