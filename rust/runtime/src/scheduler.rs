// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Clock-backed local-task scheduling for timing workloads.
//!
//! Absolute, relative, and immediate tasks stay `!Send` on the caller's
//! `LocalSet`; the injected [`Clock`] supports real and deterministic execution.

use std::cell::Cell;
use std::future::Future;
use std::pin::Pin;
use std::rc::Rc;

use crate::clock::Clock;
use tokio::sync::Notify;

/// Boxed `!Send` future accepted by [`LocalTaskScheduler`].
pub type LocalTask = Pin<Box<dyn Future<Output = ()> + 'static>>;

/// Scheduler seam shared by clock-driven workload strategies.
///
pub trait LocalTaskScheduler {
    /// Execute `task` on the local executor as soon as it can be polled.
    fn execute_async(&self, task: LocalTask);

    /// Execute `task` at absolute `target_ns` on the injected clock timeline.
    /// A target at or before `now` executes immediately after one task yield.
    fn schedule_at_ns(&self, target_ns: i64, task: LocalTask);

    /// Execute `task` after `delay_ns` on the injected clock timeline.
    /// A non-positive delay executes immediately after one task yield.
    fn schedule_later(&self, delay_ns: i64, task: LocalTask);

    /// Cancel delayed tasks that have not started yet. Already-running tasks
    /// are allowed to drain so HTTP dispatch is never abandoned mid-response.
    fn cancel_pending(&self);

    /// Cancel every delayed or running task tracked by this scheduler.
    ///
    /// Phase grace escalation uses this after asking the backend to cancel its
    /// in-flight requests. Dropping a running dispatch future is the local-loop
    /// backstop when a backend does not produce a terminal callback.
    fn cancel_all(&self);

    /// Number of delayed or running tasks currently tracked by the scheduler.
    fn task_count(&self) -> usize;

    /// Resolve once every tracked task has drained.
    fn wait_idle(&self) -> LocalTask;
}

struct SchedulerState {
    clock: Rc<dyn Clock>,
    tasks: Cell<usize>,
    idle: Notify,
    cancel_epoch: Cell<u64>,
    pending_cancelled: Notify,
    abort_epoch: Cell<u64>,
    all_cancelled: Notify,
}

struct TaskCountGuard {
    state: Rc<SchedulerState>,
}

impl Drop for TaskCountGuard {
    fn drop(&mut self) {
        self.state.finish_task();
    }
}

impl SchedulerState {
    fn start_task(&self) {
        self.tasks.set(self.tasks.get() + 1);
    }

    fn finish_task(&self) {
        let remaining = self
            .tasks
            .get()
            .checked_sub(1)
            .expect("scheduler task count cannot underflow");
        self.tasks.set(remaining);
        if remaining == 0 {
            self.idle.notify_waiters();
        }
    }
}

/// [`Clock`]-backed implementation of [`LocalTaskScheduler`].
///
/// Delayed tasks capture a cancellation generation. Cancelling advances that
/// generation and wakes parked tasks without interrupting in-flight transport.
#[derive(Clone)]
pub struct ClockTaskScheduler {
    state: Rc<SchedulerState>,
}

impl ClockTaskScheduler {
    /// Create a scheduler over `clock`. It must be used from a Tokio `LocalSet`.
    pub fn new(clock: Rc<dyn Clock>) -> Self {
        Self {
            state: Rc::new(SchedulerState {
                clock,
                tasks: Cell::new(0),
                idle: Notify::new(),
                cancel_epoch: Cell::new(0),
                pending_cancelled: Notify::new(),
                abort_epoch: Cell::new(0),
                all_cancelled: Notify::new(),
            }),
        }
    }

    fn spawn_tracked(&self, task: LocalTask) {
        self.state.start_task();
        let state = self.state.clone();
        let abort_epoch = state.abort_epoch.get();
        tokio::task::spawn_local(async move {
            let _task_count = TaskCountGuard {
                state: state.clone(),
            };
            let cancelled = state.all_cancelled.notified();
            tokio::pin!(cancelled);
            cancelled.as_mut().enable();
            if state.abort_epoch.get() == abort_epoch {
                tokio::pin!(task);
                tokio::select! {
                    _ = &mut task => {}
                    _ = &mut cancelled => {}
                }
            }
        });
    }

    fn spawn_delayed(&self, target_ns: i64, task: LocalTask) {
        let state = self.state.clone();
        let epoch = state.cancel_epoch.get();
        self.spawn_tracked(Box::pin(async move {
            let wait_ns = target_ns.saturating_sub(state.clock.now_ns());
            if wait_ns > 0 {
                let cancelled = state.pending_cancelled.notified();
                tokio::pin!(cancelled);
                // Register before checking the generation so a cancellation
                // between task creation and this first poll cannot be missed by
                // `Notify::notify_waiters`.
                cancelled.as_mut().enable();
                if state.cancel_epoch.get() != epoch {
                    return;
                }
                let sleep = state.clock.clone().sleep(wait_ns);
                tokio::pin!(sleep);
                tokio::select! {
                    _ = &mut sleep => {}
                    _ = &mut cancelled => {
                        if state.cancel_epoch.get() != epoch {
                            return;
                        }
                    }
                }
            } else {
                tokio::task::yield_now().await;
            }

            if state.cancel_epoch.get() == epoch {
                task.await;
            }
        }));
    }
}

impl LocalTaskScheduler for ClockTaskScheduler {
    fn execute_async(&self, task: LocalTask) {
        self.spawn_tracked(task);
    }

    fn schedule_at_ns(&self, target_ns: i64, task: LocalTask) {
        self.spawn_delayed(target_ns, task);
    }

    fn schedule_later(&self, delay_ns: i64, task: LocalTask) {
        let target = self.state.clock.now_ns().saturating_add(delay_ns);
        self.spawn_delayed(target, task);
    }

    fn cancel_pending(&self) {
        self.state
            .cancel_epoch
            .set(self.state.cancel_epoch.get().wrapping_add(1));
        self.state.pending_cancelled.notify_waiters();
    }

    fn cancel_all(&self) {
        self.cancel_pending();
        self.state
            .abort_epoch
            .set(self.state.abort_epoch.get().wrapping_add(1));
        self.state.all_cancelled.notify_waiters();
    }

    fn task_count(&self) -> usize {
        self.state.tasks.get()
    }

    fn wait_idle(&self) -> LocalTask {
        let state = self.state.clone();
        Box::pin(async move {
            loop {
                let idle = state.idle.notified();
                if state.tasks.get() == 0 {
                    return;
                }
                idle.await;
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;

    use crate::clock::sim_clock::SimClock;
    use crate::graph::runtime::drive_sim;

    use super::*;

    #[test]
    fn absolute_relative_and_immediate_tasks_share_the_clock() {
        let clock = Rc::new(SimClock::new());
        let log = Rc::new(RefCell::new(Vec::new()));
        let observed = log.clone();
        let clock_for_body = clock.clone();

        let outcome = drive_sim(clock.clone(), move |_handle| async move {
            let clock_dyn: Rc<dyn Clock> = clock_for_body.clone();
            let scheduler = ClockTaskScheduler::new(clock_dyn);

            let log_now = observed.clone();
            scheduler.execute_async(Box::pin(async move {
                log_now.borrow_mut().push(("now", 0));
            }));

            let log_abs = observed.clone();
            let c_abs = clock_for_body.clone();
            scheduler.schedule_at_ns(
                200,
                Box::pin(async move {
                    log_abs.borrow_mut().push(("absolute", c_abs.now_ns()));
                }),
            );

            let log_rel = observed.clone();
            let c_rel = clock_for_body.clone();
            scheduler.schedule_later(
                100,
                Box::pin(async move {
                    log_rel.borrow_mut().push(("relative", c_rel.now_ns()));
                }),
            );
            scheduler.wait_idle().await;
        });

        assert!(!outcome.deadlocked);
        assert_eq!(
            *log.borrow(),
            vec![("now", 0), ("relative", 100), ("absolute", 200)]
        );
    }

    #[test]
    fn cancelling_pending_does_not_cancel_running_work() {
        let clock = Rc::new(SimClock::new());
        let log = Rc::new(RefCell::new(Vec::new()));
        let observed = log.clone();
        let clock_for_body = clock.clone();

        drive_sim(clock.clone(), move |_handle| async move {
            let scheduler = ClockTaskScheduler::new(clock_for_body.clone());
            let running_log = observed.clone();
            scheduler.execute_async(Box::pin(async move {
                running_log.borrow_mut().push("running");
            }));
            let pending_log = observed.clone();
            scheduler.schedule_at_ns(
                100,
                Box::pin(async move {
                    pending_log.borrow_mut().push("pending");
                }),
            );
            scheduler.cancel_pending();
            scheduler.wait_idle().await;
        });

        assert_eq!(*log.borrow(), vec!["running"]);
        assert_eq!(
            clock.now_ns(),
            0,
            "cancelled timer must not advance SimClock"
        );
    }

    #[test]
    fn cancelling_all_drops_running_work_without_advancing_virtual_time() {
        let clock = Rc::new(SimClock::new());
        let log = Rc::new(RefCell::new(Vec::new()));
        let observed = log.clone();
        let clock_for_body = clock.clone();

        drive_sim(clock.clone(), move |_handle| async move {
            let scheduler = ClockTaskScheduler::new(clock_for_body.clone());
            let running_log = observed.clone();
            let task_clock = clock_for_body.clone();
            scheduler.execute_async(Box::pin(async move {
                running_log.borrow_mut().push("started");
                task_clock.sleep(100).await;
                running_log.borrow_mut().push("finished");
            }));
            tokio::task::yield_now().await;
            scheduler.cancel_all();
            scheduler.wait_idle().await;
        });

        assert_eq!(*log.borrow(), vec!["started"]);
        assert_eq!(clock.now_ns(), 0);
    }

    #[test]
    fn panicking_task_releases_idle_accounting() {
        let clock = Rc::new(SimClock::new());
        let clock_for_body = clock.clone();

        let outcome = drive_sim(clock, move |_handle| async move {
            let scheduler = ClockTaskScheduler::new(clock_for_body);
            scheduler.execute_async(Box::pin(async move {
                panic!("fixture scheduler task panic");
            }));
            scheduler.wait_idle().await;
            assert_eq!(scheduler.task_count(), 0);
        });

        assert!(!outcome.deadlocked);
    }
}
