// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Per-user open-loop pacing with virtual-history steady-state seeding.
//!
//! The pure setup math lives in `crate::timing::plan_user_centric`; this module
//! binds the plan to sampled sessions, schedules initial users, maintains a
//! deterministic absolute spawn heap, paces each continuation at
//! `max(now, previous_send + turn_gap)`, holds an optional session slot from
//! turn 0 through the final response, and exposes passive adaptive user control.

use std::cell::{Cell, RefCell};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};
use std::rc::Rc;

use crate::timing::user_centric::next_replacement_spawn_ns;
use crate::timing::{SlotGuard, SlotPool, plan_user_centric};
use anyhow::{Result, anyhow, bail};
use async_trait::async_trait;
use tokio::sync::Notify;

use crate::multiturn::{ConversationSource, SampledSession, TurnToSend};
use crate::scheduled::{ScheduledRuntime, UserControlSnapshot, Workload};
use crate::scheduler::LocalTaskScheduler;

/// User-centric workload configuration.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct UserCentricConfig {
    /// Initial number of simulated users.
    pub num_users: usize,
    /// Aggregate request rate across those users, in requests per second.
    pub request_rate: f64,
    /// Optional concurrent-session ceiling. `None` preserves strict open-loop
    /// churn; `Some` is the documented closed-loop exception.
    pub concurrency: Option<usize>,
}

impl UserCentricConfig {
    fn validate(self) -> Result<Self> {
        if self.num_users == 0 {
            bail!("num_users must be set and non-zero for user-centric rate mode");
        }
        if !self.request_rate.is_finite() || self.request_rate <= 0.0 {
            bail!("request_rate must be set and positive for user-centric rate mode");
        }
        if self.concurrency == Some(0) {
            bail!("user-centric concurrency must be positive");
        }
        Ok(self)
    }
}

struct User {
    user_id: u64,
    next_send_ns: i64,
    max_turns: usize,
    order: usize,
    session_guard: Option<SlotGuard>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct SpawnEntry {
    at_ns: i64,
    seq_no: u64,
}

impl Ord for SpawnEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse chronological order so BinaryHeap pops the earliest entry.
        other
            .at_ns
            .cmp(&self.at_ns)
            .then_with(|| other.seq_no.cmp(&self.seq_no))
    }
}

impl PartialOrd for SpawnEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Single-owner pool for per-user timing, churn, and adaptive state.
pub struct UserPool {
    users: RefCell<HashMap<String, User>>,
    target_users: Cell<usize>,
    adaptive_enabled: Cell<bool>,
    request_rate: f64,
    stagger_ns: i64,
    turn_gap_ns: Cell<i64>,
    next_user_id: Cell<u64>,
    spawns: RefCell<BinaryHeap<SpawnEntry>>,
    next_spawn_seq: Cell<u64>,
    queue_initialized: Cell<bool>,
    queue_changed: Notify,
    retired_user_cancellations: Cell<usize>,
    session_slots: Option<Rc<SlotPool>>,
}

impl UserPool {
    fn new(
        config: UserCentricConfig,
        stagger_ns: i64,
        turn_gap_ns: i64,
        next_user_id: u64,
    ) -> Rc<Self> {
        Rc::new(Self {
            users: RefCell::new(HashMap::new()),
            target_users: Cell::new(config.num_users),
            adaptive_enabled: Cell::new(false),
            request_rate: config.request_rate,
            stagger_ns,
            turn_gap_ns: Cell::new(turn_gap_ns),
            next_user_id: Cell::new(next_user_id),
            spawns: RefCell::new(BinaryHeap::new()),
            next_spawn_seq: Cell::new(0),
            queue_initialized: Cell::new(false),
            queue_changed: Notify::new(),
            retired_user_cancellations: Cell::new(0),
            session_slots: config
                .concurrency
                .map(|limit| Rc::new(SlotPool::new(limit))),
        })
    }

    fn insert_user(
        &self,
        user_id: u64,
        session: &SampledSession,
        next_send_ns: i64,
        max_turns: usize,
        order: usize,
    ) -> Result<()> {
        if self.users.borrow().contains_key(&session.x_correlation_id) {
            bail!("duplicate user correlation id {}", session.x_correlation_id);
        }
        self.users.borrow_mut().insert(
            session.x_correlation_id.clone(),
            User {
                user_id,
                next_send_ns,
                max_turns,
                order,
                session_guard: None,
            },
        );
        Ok(())
    }

    fn set_initial_send(&self, correlation_id: &str, at_ns: i64) -> Result<()> {
        let mut users = self.users.borrow_mut();
        let user = users
            .get_mut(correlation_id)
            .ok_or_else(|| anyhow!("user not found for x_correlation_id: {correlation_id}"))?;
        user.next_send_ns = at_ns;
        Ok(())
    }

    fn user_id(&self, correlation_id: &str) -> Option<u64> {
        self.users
            .borrow()
            .get(correlation_id)
            .map(|user| user.user_id)
    }

    fn max_turns(&self, correlation_id: &str) -> Option<usize> {
        self.users
            .borrow()
            .get(correlation_id)
            .map(|user| user.max_turns)
    }

    fn order(&self, correlation_id: &str) -> Option<usize> {
        self.users
            .borrow()
            .get(correlation_id)
            .map(|user| user.order)
    }

    async fn acquire_session_slot(&self, correlation_id: &str) -> bool {
        let Some(slots) = self.session_slots.clone() else {
            return self.users.borrow().contains_key(correlation_id);
        };
        if self
            .users
            .borrow()
            .get(correlation_id)
            .is_some_and(|user| user.session_guard.is_some())
        {
            return true;
        }
        let guard = slots.acquire().await;
        let mut users = self.users.borrow_mut();
        let Some(user) = users.get_mut(correlation_id) else {
            drop(guard);
            return false;
        };
        user.session_guard = Some(guard);
        true
    }

    fn pace_next(&self, correlation_id: &str, now_ns: i64) -> Result<i64> {
        let mut users = self.users.borrow_mut();
        let user = users
            .get_mut(correlation_id)
            .ok_or_else(|| anyhow!("user not found for x_correlation_id: {correlation_id}"))?;
        user.next_send_ns = now_ns.max(user.next_send_ns.saturating_add(self.turn_gap_ns.get()));
        Ok(user.next_send_ns)
    }

    fn retire(&self, correlation_id: &str) {
        self.users.borrow_mut().remove(correlation_id);
    }

    fn active_user_count(&self) -> usize {
        self.users.borrow().len()
    }

    fn should_spawn(&self) -> bool {
        !self.adaptive_enabled.get() || self.active_user_count() < self.target_users.get()
    }

    fn should_spawn_replacement(&self) -> bool {
        !self.adaptive_enabled.get() || self.active_user_count() <= self.target_users.get()
    }

    fn push_spawn(&self, at_ns: i64) {
        let seq_no = self.next_spawn_seq.get();
        self.next_spawn_seq.set(seq_no.wrapping_add(1));
        self.spawns.borrow_mut().push(SpawnEntry { at_ns, seq_no });
        self.queue_changed.notify_waiters();
    }

    fn peek_spawn(&self) -> Option<i64> {
        self.spawns.borrow().peek().map(|entry| entry.at_ns)
    }

    fn pop_due_spawn(&self, now_ns: i64) -> Option<i64> {
        let mut spawns = self.spawns.borrow_mut();
        if spawns.peek().is_some_and(|entry| entry.at_ns <= now_ns) {
            spawns.pop().map(|entry| entry.at_ns)
        } else {
            None
        }
    }

    fn defer_spawn(&self, now_ns: i64) {
        self.push_spawn(now_ns.saturating_add(self.stagger_ns));
    }

    fn schedule_replacement(&self, spawned_at_ns: i64, max_turns: usize) {
        if self.should_spawn_replacement() {
            self.push_spawn(next_replacement_spawn_ns(
                spawned_at_ns,
                max_turns,
                self.turn_gap_ns.get(),
            ));
        }
    }

    fn schedule_replacement_for_user(
        &self,
        correlation_id: &str,
        spawned_at_ns: i64,
    ) -> Result<()> {
        let max_turns = self
            .max_turns(correlation_id)
            .ok_or_else(|| anyhow!("user not found for x_correlation_id: {correlation_id}"))?;
        self.schedule_replacement(spawned_at_ns, max_turns);
        Ok(())
    }

    fn allocate_user_id(&self) -> u64 {
        let id = self.next_user_id.get();
        self.next_user_id.set(id.wrapping_add(1));
        id
    }

    fn snapshot(&self) -> UserControlSnapshot {
        let active = self.active_user_count();
        let target = self.target_users.get();
        UserControlSnapshot {
            target_value: target,
            actual_value: active,
            active_users: active,
            retiring_users: active.saturating_sub(target),
            cancelled: self.retired_user_cancellations.get(),
        }
    }
}

/// Adaptive-control seam for user-centric workloads.
pub trait UserTargetController {
    /// Current target user count.
    fn target_users(&self) -> usize;

    /// Change the target at `now_ns`. Scale-up schedules new users at staggered
    /// offsets; scale-down drains excess users by suppressing replacements.
    fn set_target_users(&self, value: usize, now_ns: i64) -> Result<()>;

    /// Current target/actual/retiring snapshot.
    fn snapshot(&self) -> UserControlSnapshot;
}

/// Cloneable handle implementing [`UserTargetController`].
#[derive(Clone)]
pub struct UserCentricControl {
    pool: Rc<UserPool>,
}

impl UserTargetController for UserCentricControl {
    fn target_users(&self) -> usize {
        self.pool.target_users.get()
    }

    fn set_target_users(&self, value: usize, now_ns: i64) -> Result<()> {
        if value == 0 {
            bail!("target users must be positive");
        }
        let old_target = self.pool.target_users.replace(value);
        self.pool.adaptive_enabled.set(true);
        self.pool
            .turn_gap_ns
            .set(seconds_to_ns(value as f64 / self.pool.request_rate)?);
        if self.pool.queue_initialized.get() && value > old_target {
            for slot in 0..(value - old_target) {
                self.pool
                    .push_spawn(now_ns.saturating_add(slot as i64 * self.pool.stagger_ns));
            }
        }
        Ok(())
    }

    fn snapshot(&self) -> UserControlSnapshot {
        self.pool.snapshot()
    }
}

impl crate::adaptive_core::UserTarget for UserCentricControl {
    fn set_target_users(
        &self,
        value: usize,
        now_ns: i64,
    ) -> Result<(), crate::adaptive_core::AdaptiveError> {
        UserTargetController::set_target_users(self, value, now_ns)
            .map_err(|error| crate::adaptive_core::AdaptiveError::Actuator(error.to_string()))
    }

    fn user_control_snapshot(&self) -> crate::adaptive_core::ControlSnapshot {
        let snapshot = UserTargetController::snapshot(self);
        crate::adaptive_core::ControlSnapshot {
            target_value: snapshot.target_value as f64,
            actual_value: snapshot.actual_value as f64,
            active_users: Some(snapshot.active_users),
            retiring_users: Some(snapshot.retiring_users),
            cancelled: Some(snapshot.cancelled),
        }
    }
}

#[derive(Clone)]
struct InitialUser {
    turn: TurnToSend,
}

/// Fully prepared user-centric [`Workload`].
pub struct UserCentricWorkload {
    source: Rc<RefCell<Box<dyn ConversationSource>>>,
    pool: Rc<UserPool>,
    initial_users: Vec<InitialUser>,
}

impl UserCentricWorkload {
    /// Bind the deterministic virtual-history plan to concrete sampled sessions.
    /// This setup runs before the benchmark start timestamp is captured.
    pub fn new(config: UserCentricConfig, mut source: Box<dyn ConversationSource>) -> Result<Self> {
        let config = config.validate()?;
        // The DRAW corpus, not the enumeration corpus. Under `Global` dispatch a
        // shard strides absolute corpus positions, so averaging the residue class
        // it merely enumerates would shape the virtual-history plan for a
        // population it does not sample. This mean sets `session_lifetime` (hence
        // every seeded user's turn cap and the coprime `spacing_step`) and gates
        // the `average turns >= 2` admission below, so a residue class whose mean
        // turn count differs from the corpus mean would also reject runs the
        // corpus admits — a hard failure whose presence depended on `workers`.
        // `Sharded` shards keep the residue basis: there the shard's draw and its
        // enumeration are the same self-contained sub-corpus.
        let average_turns = {
            let corpus = source.sampled_conversations();
            if corpus.is_empty() {
                bail!("user-centric conversation dataset cannot be empty");
            }
            corpus
                .iter()
                .map(|conversation| conversation.turns.len())
                .sum::<usize>() as f64
                / corpus.len() as f64
        };
        let rounded_turns = average_turns.round_ties_even() as usize;
        if rounded_turns < 2 {
            bail!("user-centric mode requires multi-turn conversations (average turns >= 2)");
        }
        let plan = plan_user_centric(config.num_users, rounded_turns, config.request_rate);
        let next_user_id = plan
            .initial_users
            .iter()
            .map(|user| user.user_id)
            .max()
            .unwrap_or(0)
            .saturating_add(1);
        let pool = UserPool::new(config, plan.stagger_ns, plan.turn_gap_ns, next_user_id);

        let mut initial_users = Vec::with_capacity(plan.initial_users.len());
        for (index, planned) in plan.initial_users.iter().enumerate() {
            let sampled = source.next(Some(planned.user_id.to_string()))?;
            // The last plan entry is the fresh t=0 replacement: bind it to the
            // concrete sample's full length rather than the sampler-free average.
            let cap = if index + 1 == plan.initial_users.len() {
                None
            } else {
                Some(planned.max_turns)
            };
            let turn = sampled.build_first_turn(cap)?;
            let max_turns = turn.num_turns;
            pool.insert_user(planned.user_id, &sampled, 0, max_turns, planned.order)?;
            initial_users.push(InitialUser { turn });
        }

        Ok(Self {
            source: Rc::new(RefCell::new(source)),
            pool,
            initial_users,
        })
    }

    /// Adaptive target handle for ramping controllers.
    pub fn control(&self) -> UserCentricControl {
        UserCentricControl {
            pool: self.pool.clone(),
        }
    }

    /// Session admission pool consumed by phase ramps and adaptive actuators.
    pub fn session_slots(&self) -> Option<Rc<SlotPool>> {
        self.pool.session_slots.clone()
    }
}

#[async_trait(?Send)]
impl Workload for UserCentricWorkload {
    fn name(&self) -> &'static str {
        "user_centric_rate"
    }

    async fn execute(&self, runtime: Rc<ScheduledRuntime>) -> Result<()> {
        self.pool.queue_initialized.set(true);

        // Initial steady state: all first turns are registered up front at their
        // absolute order*stagger targets, then each target seeds one replacement.
        for initial in &self.initial_users {
            let order = self
                .pool
                .order(&initial.turn.x_correlation_id)
                .ok_or_else(|| anyhow!("initial user disappeared before execution"))?;
            let target_ns = runtime
                .start_ns()
                .saturating_add(order as i64 * self.pool.stagger_ns);
            self.pool
                .set_initial_send(&initial.turn.x_correlation_id, target_ns)?;
            self.pool
                .schedule_replacement_for_user(&initial.turn.x_correlation_id, target_ns)?;
            schedule_user_turn(
                runtime.clone(),
                self.source.clone(),
                self.pool.clone(),
                initial.turn.clone(),
                target_ns,
            );
        }

        // Open-loop churn pump. It awaits only clock pacing and the optional
        // session slot; HTTP completion remains in independent dispatch tasks.
        while runtime.can_issue(false) {
            let Some(target_ns) = self.pool.peek_spawn() else {
                // The queue can be empty transiently during adaptive drain.
                let changed = self.pool.queue_changed.notified();
                let stop = runtime.wait_until_or_stop(runtime.now_ns().saturating_add(100_000_000));
                tokio::pin!(changed);
                tokio::pin!(stop);
                tokio::select! {
                    _ = &mut changed => continue,
                    keep_running = &mut stop => {
                        if !keep_running { break; }
                        continue;
                    }
                }
            };

            let changed = self.pool.queue_changed.notified();
            let wait = runtime.wait_until_or_stop(target_ns);
            tokio::pin!(changed);
            tokio::pin!(wait);
            let reached = tokio::select! {
                _ = &mut changed => false,
                reached = &mut wait => reached,
            };
            if !reached {
                if !runtime.can_issue(false) {
                    break;
                }
                continue;
            }
            let Some(spawn_ns) = self.pool.pop_due_spawn(runtime.now_ns()) else {
                continue;
            };

            if !self.pool.should_spawn() {
                self.pool.defer_spawn(runtime.now_ns());
                continue;
            }

            let user_id = self.pool.allocate_user_id();
            let sampled = self.source.borrow_mut().next(Some(user_id.to_string()))?;
            let turn = sampled.build_first_turn(None)?;
            let max_turns = turn.num_turns;
            self.pool
                .insert_user(user_id, &sampled, spawn_ns, max_turns, 0)?;
            if !issue_user_turn(
                runtime.clone(),
                self.source.clone(),
                self.pool.clone(),
                turn,
                spawn_ns,
            )
            .await
            {
                // A session-count bound can reject this new turn-0 while
                // already-started users still have continuations to drain.
                // Only terminate the pump when the less-restrictive any-turn
                // gate is also closed.
                if !runtime.can_issue(false) {
                    break;
                }
                continue;
            }
            self.pool
                .schedule_replacement_for_user(&sampled.x_correlation_id, spawn_ns)?;
        }

        runtime.scheduler().cancel_pending();
        runtime.scheduler().wait_idle().await;
        Ok(())
    }

    fn user_control_snapshot(&self) -> Option<UserControlSnapshot> {
        Some(self.pool.snapshot())
    }
}

fn schedule_user_turn(
    runtime: Rc<ScheduledRuntime>,
    source: Rc<RefCell<Box<dyn ConversationSource>>>,
    pool: Rc<UserPool>,
    turn: TurnToSend,
    target_ns: i64,
) {
    let scheduler = runtime.scheduler();
    scheduler.schedule_at_ns(
        target_ns,
        Box::pin(async move {
            issue_user_turn(runtime, source, pool, turn, target_ns).await;
        }),
    );
}

async fn issue_user_turn(
    runtime: Rc<ScheduledRuntime>,
    source: Rc<RefCell<Box<dyn ConversationSource>>>,
    pool: Rc<UserPool>,
    turn: TurnToSend,
    target_ns: i64,
) -> bool {
    let correlation_id = turn.x_correlation_id.clone();
    if turn.turn_index == 0 {
        if !runtime.can_issue(true) {
            pool.retire(&correlation_id);
            return false;
        }
        if !pool.acquire_session_slot(&correlation_id).await {
            return false;
        }
        if !runtime.can_issue(true) {
            pool.retire(&correlation_id);
            return false;
        }
    } else if !runtime.can_issue(false) {
        pool.retire(&correlation_id);
        return false;
    }

    let user_id = pool.user_id(&correlation_id);
    let runtime_for_completion = runtime.clone();
    let source_for_completion = source.clone();
    let pool_for_completion = pool.clone();
    let issued = runtime.issue_turn(
        turn,
        target_ns,
        user_id,
        Box::new(move |credit, outcome| {
            Box::pin(async move {
                let correlation_id = credit.turn.x_correlation_id.clone();
                if credit.is_final_turn() {
                    pool_for_completion.retire(&correlation_id);
                    return;
                }
                if !runtime_for_completion.can_issue(false) {
                    pool_for_completion.retire(&correlation_id);
                    return;
                }

                let next_turn = {
                    let source = source_for_completion.borrow();
                    match source.next_turn(&credit, outcome.to_turn_response()) {
                        Ok(Some(turn)) => turn,
                        Ok(None) => {
                            pool_for_completion.retire(&correlation_id);
                            return;
                        }
                        Err(error) => {
                            tracing::warn!(
                                error = %error,
                                %correlation_id,
                                "user-centric continuation materialization failed"
                            );
                            pool_for_completion.retire(&correlation_id);
                            return;
                        }
                    }
                };
                let next_target = match pool_for_completion
                    .pace_next(&correlation_id, runtime_for_completion.now_ns())
                {
                    Ok(target) => target,
                    Err(error) => {
                        tracing::warn!(error = %error, "user-centric pacing failed");
                        return;
                    }
                };
                schedule_user_turn(
                    runtime_for_completion,
                    source_for_completion,
                    pool_for_completion,
                    next_turn,
                    next_target,
                );
            })
        }),
    );
    if !issued {
        pool.retire(&correlation_id);
    }
    issued
}

fn seconds_to_ns(seconds: f64) -> Result<i64> {
    if !seconds.is_finite() || seconds < 0.0 {
        bail!("time interval must be finite and non-negative");
    }
    let nanoseconds = seconds * 1_000_000_000.0;
    if nanoseconds >= i64::MAX as f64 {
        bail!("time interval is outside the i64 nanosecond range");
    }
    Ok(nanoseconds.round_ties_even() as i64)
}

#[cfg(test)]
mod tests {
    use crate::test_util::synthetic_prepared_source;

    use super::*;

    async fn workload(users: usize, rate: f64, turns: usize) -> UserCentricWorkload {
        let source = synthetic_prepared_source(turns, 4, 1, None, "m").await;
        UserCentricWorkload::new(
            UserCentricConfig {
                num_users: users,
                request_rate: rate,
                concurrency: None,
            },
            source,
        )
        .unwrap()
    }

    #[tokio::test]
    async fn setup_binds_fresh_user_to_actual_sample_length() {
        let workload = workload(4, 20.0, 3).await;
        let fresh = workload
            .initial_users
            .iter()
            .find(|user| workload.pool.order(&user.turn.x_correlation_id) == Some(0))
            .unwrap();
        assert_eq!(fresh.turn.num_turns, 3);
        assert_eq!(workload.initial_users.len(), 4);
    }

    #[tokio::test]
    async fn adaptive_scale_up_staggers_and_scale_down_reports_retiring() {
        let workload = workload(4, 10.0, 3).await;
        workload.pool.queue_initialized.set(true);
        let control = workload.control();
        control.set_target_users(6, 1_000).unwrap();
        assert_eq!(control.target_users(), 6);
        assert_eq!(workload.pool.turn_gap_ns.get(), 600_000_000);
        let mut spawn_times = workload
            .pool
            .spawns
            .borrow()
            .iter()
            .map(|entry| entry.at_ns)
            .collect::<Vec<_>>();
        spawn_times.sort_unstable();
        assert_eq!(spawn_times, vec![1_000, 100_001_000]);

        control.set_target_users(2, 2_000).unwrap();
        let snapshot = control.snapshot();
        assert_eq!(snapshot.target_value, 2);
        assert_eq!(snapshot.actual_value, 4);
        assert_eq!(snapshot.retiring_users, 2);
        assert!(!workload.pool.should_spawn());
        assert!(!workload.pool.should_spawn_replacement());
    }

    #[tokio::test]
    async fn adaptive_target_rejects_zero() {
        let workload = workload(2, 10.0, 2).await;
        assert!(
            workload
                .control()
                .set_target_users(0, 0)
                .unwrap_err()
                .to_string()
                .contains("positive")
        );
    }

    #[tokio::test]
    async fn single_turn_dataset_is_rejected() {
        let source = synthetic_prepared_source(1, 4, 1, None, "m").await;
        let result = UserCentricWorkload::new(
            UserCentricConfig {
                num_users: 2,
                request_rate: 10.0,
                concurrency: None,
            },
            source,
        );
        assert!(matches!(result, Err(error) if error.to_string().contains("multi-turn")));
    }
}
