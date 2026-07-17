// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Dynamic-capacity concurrency slots.
//!
//! A [`SlotPool`] is a semaphore whose limit can change at runtime:
//!
//! - **Increase** cancels outstanding *debt* first, then adds the remaining slots
//!   — immediate extra capacity.
//! - **Decrease** drains currently-available slots immediately, and records the
//!   shortfall it could not drain as **debt**. While `debt > 0`, each
//!   release is absorbed by debt instead of freeing a slot. In-flight holders
//!   therefore drain without making effective capacity negative.
//!
//! Debt exists because the underlying semaphore's permit count must never go
//! negative (a negative count would let `acquire` bypass blocking). Debt keeps
//! that count `>= 0` while still enforcing the reduced effective capacity.
//!
//! [`SlotPool`] is policy-neutral. Session and prefill admission use independent
//! pools, and callers decide when guards are released.
//!
//! Single-threaded design: state lives behind `Rc`/`Cell` (the crate runs on a
//! `LocalSet`, `?Send`), while [`tokio::sync::Semaphore`] itself carries the
//! wakeup queue for waiting acquirers.

use std::cell::Cell;
use std::rc::Rc;

use tokio::sync::Semaphore;

/// Instrumentation counters for a [`SlotPool`], for observability and testing.
///
/// Safe under single-threaded async: mutations between `.await` points are
/// atomic with respect to other tasks.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ConcurrencyStats {
    /// Number of times a slot was successfully acquired.
    pub acquire_count: u64,
    /// Number of times a slot was released.
    pub release_count: u64,
    /// Number of times an [`acquire`](SlotPool::acquire) had to wait for a slot.
    pub wait_count: u64,
}

/// Shared inner state of a [`SlotPool`], held via `Rc` so that a [`SlotGuard`]'s
/// `Drop` can call back into [`release`](SlotInner::release).
struct SlotInner {
    /// Underlying permit source. Permits are `forget()`-ten on acquire and
    /// re-added explicitly on release so that debt can intercept releases.
    semaphore: Semaphore,
    /// Outstanding debt: releases to absorb before slots are freed again.
    debt: Cell<usize>,
    /// The current configured concurrency limit.
    current_limit: Cell<usize>,
    /// Instrumentation counters.
    stats: Cell<ConcurrencyStats>,
}

impl SlotInner {
    /// Release a slot. If debt is outstanding (from a limit decrease), the
    /// release is absorbed by the debt instead of freeing a slot for acquirers.
    fn release(&self) {
        let mut stats = self.stats.get();
        stats.release_count += 1;
        self.stats.set(stats);

        let debt = self.debt.get();
        if debt > 0 {
            self.debt.set(debt - 1);
        } else {
            self.semaphore.add_permits(1);
        }
    }

    /// Record a successful acquisition in the stats.
    fn note_acquire(&self) {
        let mut stats = self.stats.get();
        stats.acquire_count += 1;
        self.stats.set(stats);
    }
}

/// A dynamic-capacity concurrency semaphore with debt-tracked graceful drain.
///
/// See the [module docs](self) for the increase/decrease semantics.
pub struct SlotPool {
    inner: Rc<SlotInner>,
}

impl SlotPool {
    /// Create a pool with `initial_limit` available slots.
    pub fn new(initial_limit: usize) -> Self {
        Self {
            inner: Rc::new(SlotInner {
                semaphore: Semaphore::new(initial_limit),
                debt: Cell::new(0),
                current_limit: Cell::new(initial_limit),
                stats: Cell::new(ConcurrencyStats::default()),
            }),
        }
    }

    /// The current configured concurrency limit.
    pub fn current_limit(&self) -> usize {
        self.inner.current_limit.get()
    }

    /// Outstanding debt: releases still to be absorbed before slots free up.
    pub fn debt(&self) -> usize {
        self.inner.debt.get()
    }

    /// Slots currently available to acquirers, accounting for debt
    /// (`available_permits - debt`, floored at 0).
    pub fn effective_slots(&self) -> usize {
        self.inner
            .semaphore
            .available_permits()
            .saturating_sub(self.inner.debt.get())
    }

    /// Whether the underlying semaphore has no free permits.
    ///
    /// This reflects the raw permit count and does **not** subtract debt; use
    /// [`effective_slots`](Self::effective_slots) for debt-adjusted capacity.
    pub fn locked(&self) -> bool {
        self.inner.semaphore.available_permits() == 0
    }

    /// A snapshot of the instrumentation counters.
    pub fn stats(&self) -> ConcurrencyStats {
        self.inner.stats.get()
    }

    /// Adjust the concurrency limit.
    ///
    /// - **Increase**: cancel outstanding debt first, then add the remaining slots.
    /// - **Decrease**: drain currently-available slots immediately; whatever could
    ///   not be drained becomes debt, to be absorbed by future releases.
    ///
    /// Synchronous: any waiters woken by added permits run only once the caller
    /// yields to the runtime.
    pub fn set_limit(&self, new_limit: usize) {
        let current = self.inner.current_limit.get();
        let diff = new_limit as i64 - current as i64;

        if diff > 0 {
            // Increase: cancel debt first, then add the leftover as real slots.
            let diff = diff as usize;
            let debt = self.inner.debt.get();
            let cancel = diff.min(debt);
            self.inner.debt.set(debt - cancel);
            let to_add = diff - cancel;
            if to_add > 0 {
                self.inner.semaphore.add_permits(to_add);
            }
        } else if diff < 0 {
            // Decrease: drain available permits now, track the remainder as debt.
            let mut shortfall = (-diff) as usize;
            let available = self.inner.semaphore.available_permits();
            let to_drain = shortfall.min(available);
            if to_drain > 0 {
                // Permanently remove `to_drain` currently-free permits. Capped at
                // `available`, so this acquisition cannot fail.
                self.inner
                    .semaphore
                    .try_acquire_many(to_drain as u32)
                    .expect("draining <= available permits cannot fail")
                    .forget();
            }
            shortfall -= to_drain;
            self.inner.debt.set(self.inner.debt.get() + shortfall);
        }

        self.inner.current_limit.set(new_limit);
    }

    /// Acquire a slot, awaiting a free one if necessary.
    ///
    /// The returned [`SlotGuard`] frees the slot (honoring debt) when dropped.
    pub async fn acquire(&self) -> SlotGuard {
        if self.locked() {
            let mut stats = self.inner.stats.get();
            stats.wait_count += 1;
            self.inner.stats.set(stats);
        }
        // Forget the permit so releasing goes through our debt-aware `release`
        // rather than the permit's own `Drop`.
        self.inner
            .semaphore
            .acquire()
            .await
            .expect("SlotPool semaphore is never closed")
            .forget();
        self.inner.note_acquire();
        SlotGuard {
            inner: Rc::clone(&self.inner),
        }
    }

    /// Try to acquire a slot without blocking.
    ///
    /// Returns `Some(guard)` if a slot was free, or `None` otherwise. Never waits.
    pub fn try_acquire(&self) -> Option<SlotGuard> {
        match self.inner.semaphore.try_acquire() {
            Ok(permit) => {
                permit.forget();
                self.inner.note_acquire();
                Some(SlotGuard {
                    inner: Rc::clone(&self.inner),
                })
            }
            Err(_) => None,
        }
    }
}

/// RAII handle for one acquired slot. Dropping it releases the slot back to its
/// [`SlotPool`] via the pool's debt-aware release path.
#[must_use = "dropping the SlotGuard immediately releases the slot"]
pub struct SlotGuard {
    inner: Rc<SlotInner>,
}

impl Drop for SlotGuard {
    fn drop(&mut self) {
        self.inner.release();
    }
}

/// Convenience bundle of two independent [`SlotPool`]s — one gating concurrent
/// sessions, one gating in-flight prefill requests. Purely a container: it adds
/// no policy. The caller decides when to acquire each and when to drop the
/// returned guards (e.g. dropping the `prefill` guard on the first-token event).
pub struct ConcurrencyManager {
    /// Slots gating concurrent sessions.
    pub session: SlotPool,
    /// Slots gating in-flight prefill (requests awaiting first token).
    pub prefill: SlotPool,
}

impl ConcurrencyManager {
    /// Create both pools with the given initial limits.
    pub fn new(session_limit: usize, prefill_limit: usize) -> Self {
        Self {
            session: SlotPool::new(session_limit),
            prefill: SlotPool::new(prefill_limit),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn try_acquire_exhausts_capacity_then_returns_none() {
        let pool = SlotPool::new(2);
        let _g1 = pool.try_acquire().expect("first slot free");
        let _g2 = pool.try_acquire().expect("second slot free");
        assert!(
            pool.try_acquire().is_none(),
            "capacity of 2 must be exhausted"
        );
        assert!(pool.locked());
        assert_eq!(pool.effective_slots(), 0);
    }

    #[test]
    fn guard_drop_frees_a_slot() {
        let pool = SlotPool::new(1);
        let g = pool.try_acquire().expect("slot free");
        assert!(pool.locked());
        drop(g);
        assert!(!pool.locked(), "dropping the guard must free the slot");
        assert!(pool.try_acquire().is_some(), "freed slot is re-acquirable");
    }

    #[test]
    fn set_limit_increase_adds_capacity_and_cancels_debt_first() {
        let pool = SlotPool::new(1);
        let _g = pool.try_acquire().expect("slot free");
        assert!(pool.try_acquire().is_none());

        pool.set_limit(3);
        assert_eq!(pool.current_limit(), 3);
        let _g2 = pool.try_acquire().expect("added capacity 1");
        let _g3 = pool.try_acquire().expect("added capacity 2");
        assert!(pool.try_acquire().is_none(), "only two slots were added");
    }

    #[test]
    fn increase_cancels_debt_before_adding_slots() {
        let pool = SlotPool::new(4);
        // Hold everything so a decrease produces debt.
        let mut guards: Vec<SlotGuard> = (0..4).map(|_| pool.try_acquire().unwrap()).collect();
        assert_eq!(pool.effective_slots(), 0);

        pool.set_limit(1); // available is 0, so all 3 removed become debt.
        assert_eq!(pool.debt(), 3);

        // Increase by 2: cancels 2 debt, adds 0 real slots.
        pool.set_limit(3);
        assert_eq!(pool.debt(), 1, "increase cancels debt before adding slots");
        assert_eq!(pool.effective_slots(), 0);

        // Draining held guards: first release absorbed by remaining debt, then slots free.
        guards.pop(); // release -> debt 1 -> 0
        assert_eq!(pool.debt(), 0);
        assert_eq!(pool.effective_slots(), 0);
        guards.pop(); // release -> frees a real slot
        assert_eq!(pool.effective_slots(), 1);
        drop(guards);
    }

    #[test]
    fn set_limit_decrease_creates_debt_and_release_is_absorbed() {
        let pool = SlotPool::new(4);
        let g1 = pool.try_acquire().unwrap();
        let g2 = pool.try_acquire().unwrap();
        // available = 2, debt = 0
        assert_eq!(pool.effective_slots(), 2);

        // Decrease 4 -> 1: drain the 2 available, remaining shortfall (1) -> debt.
        pool.set_limit(1);
        assert_eq!(pool.current_limit(), 1);
        assert_eq!(pool.debt(), 1);
        assert_eq!(pool.effective_slots(), 0);
        assert!(pool.locked());

        // First release is absorbed by debt: no slot frees, effective stays reduced.
        drop(g1);
        assert_eq!(pool.debt(), 0);
        assert_eq!(pool.effective_slots(), 0);
        assert!(pool.locked());

        // Second release now frees a real slot, landing at the reduced capacity of 1.
        drop(g2);
        assert_eq!(pool.effective_slots(), 1);
        assert!(!pool.locked());
    }

    #[test]
    fn stats_increment_on_acquire_and_release() {
        let pool = SlotPool::new(2);
        let g = pool.try_acquire().unwrap();
        let s = pool.stats();
        assert_eq!(s.acquire_count, 1);
        assert_eq!(s.release_count, 0);
        assert_eq!(s.wait_count, 0);

        drop(g);
        let s = pool.stats();
        assert_eq!(s.acquire_count, 1);
        assert_eq!(s.release_count, 1);
    }

    #[tokio::test]
    async fn acquire_returns_immediately_when_a_slot_is_free() {
        let pool = SlotPool::new(1);
        let g = pool.acquire().await;
        assert!(pool.locked());
        assert_eq!(pool.stats().acquire_count, 1);
        assert_eq!(pool.stats().wait_count, 0);
        drop(g);
        assert!(!pool.locked());
    }

    #[tokio::test]
    async fn acquire_waits_until_a_slot_is_released() {
        let pool = SlotPool::new(1);
        let g1 = pool.acquire().await;

        // A second acquire cannot complete while the only slot is held.
        let pending = pool.acquire();
        tokio::pin!(pending);
        tokio::select! {
            biased;
            _ = &mut pending => panic!("acquire completed while locked"),
            _ = async {} => {}
        }
        assert_eq!(pool.stats().wait_count, 1, "waiting acquire is counted");

        // Freeing the slot lets the pending acquire resolve.
        drop(g1);
        let g2 = pending.await;
        assert_eq!(pool.stats().acquire_count, 2);
        drop(g2);
    }

    #[test]
    fn concurrency_manager_bundles_two_independent_pools() {
        let mgr = ConcurrencyManager::new(1, 2);
        let _s = mgr.session.try_acquire().unwrap();
        assert!(mgr.session.try_acquire().is_none());
        // Prefill pool is independent and still has capacity.
        let _p1 = mgr.prefill.try_acquire().unwrap();
        let _p2 = mgr.prefill.try_acquire().unwrap();
        assert!(mgr.prefill.try_acquire().is_none());
    }
}
