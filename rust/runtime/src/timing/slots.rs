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
use std::sync::Arc;

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

/// Which concurrency admission backend a [`SlotPool`] delegates to.
///
/// `Local` is the original single-threaded debt-aware semaphore, used by
/// `sharded` dispatch and by every phase type that has no cross-thread
/// concern (`workers == 1`, or admission that is not thread-shared). `Global`
/// wraps a [`GlobalSlotPool`] shared (via `Arc`) across every worker thread in
/// a `global`/`global-hop` dispatch cell, so `acquire`/`try_acquire` admit
/// from one true cross-thread counter instead of a thread-local share. A
/// `SlotPool` in `Global` mode carries the same debt-tracked graceful-drain
/// semantics as `Local` (see [`GlobalSlotPool::set_limit`]), so concurrency
/// ramps are exact under `global`/`global-hop` dispatch — a limit decrease
/// never transiently over-admits.
enum SlotPoolBackend {
    Local(Rc<SlotInner>),
    Global(Arc<GlobalSlotPool>),
}

/// A dynamic-capacity concurrency semaphore with debt-tracked graceful drain.
///
/// See the [module docs](self) for the increase/decrease semantics. See
/// [`SlotPoolBackend`] for the `Local`/`Global` distinction.
pub struct SlotPool {
    backend: SlotPoolBackend,
}

impl SlotPool {
    /// Create a pool with `initial_limit` available slots, backed by a
    /// thread-local semaphore (the original behavior).
    pub fn new(initial_limit: usize) -> Self {
        Self {
            backend: SlotPoolBackend::Local(Rc::new(SlotInner {
                semaphore: Semaphore::new(initial_limit),
                debt: Cell::new(0),
                current_limit: Cell::new(initial_limit),
                stats: Cell::new(ConcurrencyStats::default()),
            })),
        }
    }

    /// Create a pool that delegates every operation to a shared
    /// [`GlobalSlotPool`], so this thread's admission draws from the same
    /// cross-thread counter as every other worker thread holding a `SlotPool`
    /// over the same `Arc<GlobalSlotPool>`. Used by `global`/`global-hop`
    /// dispatch to enforce one true global concurrency limit.
    pub fn new_global(pool: Arc<GlobalSlotPool>) -> Self {
        Self {
            backend: SlotPoolBackend::Global(pool),
        }
    }

    /// Whether this pool draws from a shared cross-thread [`GlobalSlotPool`]
    /// rather than a thread-local semaphore.
    ///
    /// Callers that pair a `SlotPool` with a thread-local completion
    /// notification (e.g. a `tokio::sync::Notify` only ever fired by this
    /// SAME thread's own releases) must check this before blocking on that
    /// notification: a `Global`-backed pool's next release may come from a
    /// DIFFERENT worker thread entirely, so a thread holding zero local
    /// guards would wait on a notification that never fires — see
    /// `request_rate::RequestRateWorkload::execute`'s `NoSlot` handling.
    pub fn is_global(&self) -> bool {
        matches!(self.backend, SlotPoolBackend::Global(_))
    }

    /// The current configured concurrency limit.
    pub fn current_limit(&self) -> usize {
        match &self.backend {
            SlotPoolBackend::Local(inner) => inner.current_limit.get(),
            SlotPoolBackend::Global(pool) => pool.current_limit(),
        }
    }

    /// Outstanding debt: releases still to be absorbed before slots free up.
    ///
    /// A `Global`-backed pool tracks debt with the same semantics as `Local`
    /// (see [`GlobalSlotPool::set_limit`]).
    pub fn debt(&self) -> usize {
        match &self.backend {
            SlotPoolBackend::Local(inner) => inner.debt.get(),
            SlotPoolBackend::Global(pool) => pool.debt(),
        }
    }

    /// Slots currently available to acquirers, accounting for debt
    /// (`available_permits - debt`, floored at 0).
    pub fn effective_slots(&self) -> usize {
        match &self.backend {
            SlotPoolBackend::Local(inner) => inner
                .semaphore
                .available_permits()
                .saturating_sub(inner.debt.get()),
            SlotPoolBackend::Global(pool) => pool.effective_slots(),
        }
    }

    /// Whether the underlying semaphore has no free permits.
    ///
    /// This reflects the raw permit count and does **not** subtract debt; use
    /// [`effective_slots`](Self::effective_slots) for debt-adjusted capacity.
    pub fn locked(&self) -> bool {
        match &self.backend {
            SlotPoolBackend::Local(inner) => inner.semaphore.available_permits() == 0,
            SlotPoolBackend::Global(pool) => pool.effective_slots() == 0,
        }
    }

    /// A snapshot of the instrumentation counters.
    pub fn stats(&self) -> ConcurrencyStats {
        match &self.backend {
            SlotPoolBackend::Local(inner) => inner.stats.get(),
            SlotPoolBackend::Global(pool) => pool.stats(),
        }
    }

    /// Adjust the concurrency limit.
    ///
    /// - **Increase**: cancel outstanding debt first, then add the remaining slots.
    /// - **Decrease**: drain currently-available slots immediately; whatever could
    ///   not be drained becomes debt, to be absorbed by future releases.
    ///
    /// Synchronous: any waiters woken by added permits run only once the caller
    /// yields to the runtime. A `Global`-backed pool applies the same
    /// debt-tracked decrease semantics (see [`GlobalSlotPool::set_limit`]);
    /// calling this with the pool's own current limit (the common case: a
    /// phase's `configure` resetting the limit to the value the pool was
    /// already built with) is always a no-op.
    pub fn set_limit(&self, new_limit: usize) {
        match &self.backend {
            SlotPoolBackend::Local(inner) => {
                let current = inner.current_limit.get();
                let diff = new_limit as i64 - current as i64;

                if diff > 0 {
                    // Increase: cancel debt first, then add the leftover as real slots.
                    let diff = diff as usize;
                    let debt = inner.debt.get();
                    let cancel = diff.min(debt);
                    inner.debt.set(debt - cancel);
                    let to_add = diff - cancel;
                    if to_add > 0 {
                        inner.semaphore.add_permits(to_add);
                    }
                } else if diff < 0 {
                    // Decrease: drain available permits now, track the remainder as debt.
                    let mut shortfall = (-diff) as usize;
                    let available = inner.semaphore.available_permits();
                    let to_drain = shortfall.min(available);
                    if to_drain > 0 {
                        // Permanently remove `to_drain` currently-free permits. Capped at
                        // `available`, so this acquisition cannot fail.
                        inner
                            .semaphore
                            .try_acquire_many(to_drain as u32)
                            .expect("draining <= available permits cannot fail")
                            .forget();
                    }
                    shortfall -= to_drain;
                    inner.debt.set(inner.debt.get() + shortfall);
                }

                inner.current_limit.set(new_limit);
            }
            SlotPoolBackend::Global(pool) => pool.set_limit(new_limit),
        }
    }

    /// Acquire a slot, awaiting a free one if necessary.
    ///
    /// The returned [`SlotGuard`] frees the slot (honoring debt) when dropped.
    pub async fn acquire(&self) -> SlotGuard {
        match &self.backend {
            SlotPoolBackend::Local(inner) => {
                if inner.semaphore.available_permits() == 0 {
                    let mut stats = inner.stats.get();
                    stats.wait_count += 1;
                    inner.stats.set(stats);
                }
                // Forget the permit so releasing goes through our debt-aware `release`
                // rather than the permit's own `Drop`.
                inner
                    .semaphore
                    .acquire()
                    .await
                    .expect("SlotPool semaphore is never closed")
                    .forget();
                inner.note_acquire();
                SlotGuard {
                    backend: SlotGuardBackend::Local(Rc::clone(inner)),
                }
            }
            SlotPoolBackend::Global(pool) => SlotGuard {
                backend: SlotGuardBackend::Global(pool.acquire().await),
            },
        }
    }

    /// Try to acquire a slot without blocking.
    ///
    /// Returns `Some(guard)` if a slot was free, or `None` otherwise. Never waits.
    pub fn try_acquire(&self) -> Option<SlotGuard> {
        match &self.backend {
            SlotPoolBackend::Local(inner) => match inner.semaphore.try_acquire() {
                Ok(permit) => {
                    permit.forget();
                    inner.note_acquire();
                    Some(SlotGuard {
                        backend: SlotGuardBackend::Local(Rc::clone(inner)),
                    })
                }
                Err(_) => None,
            },
            SlotPoolBackend::Global(pool) => pool.try_acquire().map(|guard| SlotGuard {
                backend: SlotGuardBackend::Global(guard),
            }),
        }
    }
}

/// Backend a [`SlotGuard`] releases into on drop; mirrors [`SlotPoolBackend`].
enum SlotGuardBackend {
    Local(Rc<SlotInner>),
    /// Held only for its `Drop` impl, which releases the shared slot.
    Global(#[allow(dead_code)] GlobalSlotGuard),
}

/// RAII handle for one acquired slot. Dropping it releases the slot back to its
/// [`SlotPool`] via the pool's debt-aware release path (`Local`) or the shared
/// [`GlobalSlotPool`] (`Global`, via [`GlobalSlotGuard`]'s own `Drop`).
#[must_use = "dropping the SlotGuard immediately releases the slot"]
pub struct SlotGuard {
    backend: SlotGuardBackend,
}

impl Drop for SlotGuard {
    fn drop(&mut self) {
        if let SlotGuardBackend::Local(inner) = &self.backend {
            inner.release();
        }
        // `Global` releases automatically: dropping the held `GlobalSlotGuard`
        // runs its own `Drop` impl.
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

/// Cross-thread concurrency admission gate.
///
/// Unlike [`SlotPool`] (single-threaded, `Rc`-based, used by `sharded` dispatch),
/// this type is `Send + Sync` and shared via `Arc` across worker threads so
/// `global` dispatch enforces one true global concurrency limit instead of `W`
/// independent local limits.
pub struct GlobalSlotPool {
    semaphore: Semaphore,
    acquire_count: std::sync::atomic::AtomicU64,
    release_count: std::sync::atomic::AtomicU64,
    wait_count: std::sync::atomic::AtomicU64,
    current_limit: std::sync::atomic::AtomicUsize,
    /// Outstanding debt: releases to absorb before slots are freed again after
    /// a limit decrease. Mirrors [`SlotInner::debt`] but as an atomic, since
    /// releases run concurrently across worker threads. Every mutation uses a
    /// compare-and-swap loop so two concurrent releases can never both absorb
    /// the same unit of debt and drive the count below zero.
    debt: std::sync::atomic::AtomicUsize,
}

impl GlobalSlotPool {
    /// Create a pool with `initial_limit` globally shared slots.
    pub fn new(initial_limit: usize) -> Arc<Self> {
        Arc::new(Self {
            semaphore: Semaphore::new(initial_limit),
            acquire_count: std::sync::atomic::AtomicU64::new(0),
            release_count: std::sync::atomic::AtomicU64::new(0),
            wait_count: std::sync::atomic::AtomicU64::new(0),
            current_limit: std::sync::atomic::AtomicUsize::new(initial_limit),
            debt: std::sync::atomic::AtomicUsize::new(0),
        })
    }

    /// Outstanding debt: releases still to be absorbed before slots free up.
    pub fn debt(&self) -> usize {
        self.debt.load(std::sync::atomic::Ordering::Acquire)
    }

    /// Acquire one globally-shared slot, waiting if none are free.
    pub async fn acquire(self: &Arc<Self>) -> GlobalSlotGuard {
        let had_permit_immediately = self.semaphore.available_permits() > 0;
        if !had_permit_immediately {
            self.wait_count
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        let permit = self
            .semaphore
            .acquire()
            .await
            .expect("GlobalSlotPool semaphore is never closed");
        permit.forget();
        self.acquire_count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        GlobalSlotGuard { pool: self.clone() }
    }

    /// Try to acquire one globally-shared slot without blocking.
    ///
    /// Returns `None` immediately if no slot is currently free across any
    /// worker thread. Mirrors [`SlotPool::try_acquire`]'s nonblocking contract
    /// for new-session admission under `global`/`global-hop` dispatch.
    pub fn try_acquire(self: &Arc<Self>) -> Option<GlobalSlotGuard> {
        match self.semaphore.try_acquire() {
            Ok(permit) => {
                permit.forget();
                self.acquire_count
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                Some(GlobalSlotGuard { pool: self.clone() })
            }
            Err(_) => None,
        }
    }

    /// The configured global concurrency limit.
    pub fn current_limit(&self) -> usize {
        self.current_limit
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Adjust the global concurrency limit, matching [`SlotPool::set_limit`]'s
    /// debt-tracked semantics exactly.
    ///
    /// - **Increase**: cancel outstanding debt first (via a CAS loop, since
    ///   concurrent releases also mutate debt), then add the leftover as real
    ///   permits — immediate extra capacity.
    /// - **Decrease**: drain currently-available permits immediately (one at a
    ///   time, robust against concurrent acquires stealing a permit mid-drain),
    ///   and record whatever shortfall could not be drained as **debt**. While
    ///   `debt > 0`, each [`release`](Self::release) is absorbed by debt instead
    ///   of freeing a slot, so effective capacity never transiently exceeds
    ///   `new_limit` even while in-flight holders are still draining down.
    ///
    /// The common caller (a phase's `configure` resetting the limit to the
    /// value this pool was already built with) takes the `diff == 0` no-op path.
    /// `set_limit` is expected to run from the single control path that applies
    /// ramp steps, not concurrently with itself.
    pub fn set_limit(&self, new_limit: usize) {
        use std::sync::atomic::Ordering;

        let current = self.current_limit.swap(new_limit, Ordering::Relaxed);
        let diff = new_limit as i64 - current as i64;
        if diff > 0 {
            // Increase: cancel debt first, then add the leftover as real permits.
            let diff = diff as usize;
            let mut cancel;
            loop {
                let debt = self.debt.load(Ordering::Acquire);
                cancel = diff.min(debt);
                if cancel == 0 {
                    break;
                }
                if self
                    .debt
                    .compare_exchange_weak(debt, debt - cancel, Ordering::AcqRel, Ordering::Acquire)
                    .is_ok()
                {
                    break;
                }
            }
            let to_add = diff - cancel;
            if to_add > 0 {
                self.semaphore.add_permits(to_add);
            }
        } else if diff < 0 {
            // Decrease: drain available permits now, track the remainder as debt.
            let mut remaining = (-diff) as usize;
            while remaining > 0 {
                match self.semaphore.try_acquire() {
                    Ok(permit) => {
                        permit.forget();
                        remaining -= 1;
                    }
                    // No free permit right now: the rest becomes debt, to be
                    // absorbed by future releases instead of freeing slots.
                    Err(_) => break,
                }
            }
            if remaining > 0 {
                self.debt.fetch_add(remaining, Ordering::AcqRel);
            }
        }
    }

    /// Slots currently available to acquirers across all worker threads,
    /// accounting for outstanding debt (`available_permits - debt`, floored
    /// at 0). Mirrors [`SlotPool::effective_slots`].
    pub fn effective_slots(&self) -> usize {
        self.semaphore
            .available_permits()
            .saturating_sub(self.debt.load(std::sync::atomic::Ordering::Acquire))
    }

    /// A snapshot of instrumentation counters, in the shared [`ConcurrencyStats`] shape.
    pub fn stats(&self) -> ConcurrencyStats {
        ConcurrencyStats {
            acquire_count: self
                .acquire_count
                .load(std::sync::atomic::Ordering::Relaxed),
            release_count: self
                .release_count
                .load(std::sync::atomic::Ordering::Relaxed),
            wait_count: self.wait_count.load(std::sync::atomic::Ordering::Relaxed),
        }
    }

    /// Release one slot. If debt is outstanding (from a limit decrease), the
    /// release is absorbed by the debt instead of freeing a permit for
    /// acquirers, mirroring [`SlotInner::release`].
    ///
    /// The absorb path is a CAS loop rather than a bare `fetch_sub`: a plain
    /// `fetch_sub` would let two concurrent releases both observe `debt == 1`
    /// and each subtract, wrapping the unsigned counter below zero and
    /// permanently starving acquirers. The CAS re-checks `debt > 0` on every
    /// attempt, so exactly one release absorbs each unit of debt.
    fn release(&self) {
        use std::sync::atomic::Ordering;

        self.release_count.fetch_add(1, Ordering::Relaxed);

        loop {
            let debt = self.debt.load(Ordering::Acquire);
            if debt == 0 {
                // No debt: free a real permit for acquirers.
                self.semaphore.add_permits(1);
                return;
            }
            if self
                .debt
                .compare_exchange_weak(debt, debt - 1, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                // Debt absorbed this release; no permit freed.
                return;
            }
        }
    }
}

/// RAII guard for one [`GlobalSlotPool`] slot; releases on drop.
pub struct GlobalSlotGuard {
    pool: Arc<GlobalSlotPool>,
}

impl Drop for GlobalSlotGuard {
    fn drop(&mut self) {
        self.pool.release();
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

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn global_slot_pool_enforces_true_cross_thread_limit() {
        let pool = GlobalSlotPool::new(2);
        let concurrent = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let max_seen = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let mut handles = Vec::new();
        for _ in 0..8 {
            let pool = pool.clone();
            let concurrent = concurrent.clone();
            let max_seen = max_seen.clone();
            handles.push(tokio::spawn(async move {
                let _guard = pool.acquire().await;
                let now = concurrent.fetch_add(1, std::sync::atomic::Ordering::SeqCst) + 1;
                max_seen.fetch_max(now, std::sync::atomic::Ordering::SeqCst);
                tokio::time::sleep(std::time::Duration::from_millis(5)).await;
                concurrent.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
            }));
        }
        for h in handles {
            h.await.unwrap();
        }
        assert_eq!(max_seen.load(std::sync::atomic::Ordering::SeqCst), 2);
    }

    /// Proves `SlotPool::new_global` is a transparent `Global`-backed facade:
    /// several independent `SlotPool` handles (as if built on different worker
    /// threads, one per thread here for a realistic `!Send` `Rc` story) that
    /// all wrap the SAME `Arc<GlobalSlotPool>` enforce one true aggregate cap
    /// across every handle combined, never per-handle.
    #[test]
    fn slot_pool_new_global_enforces_one_cross_thread_cap_across_handles() {
        use std::sync::Mutex;
        use std::thread;

        let global = GlobalSlotPool::new(2);
        let concurrent = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let max_seen = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let errors: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));

        let handles: Vec<_> = (0..4)
            .map(|thread_id| {
                let global = global.clone();
                let concurrent = concurrent.clone();
                let max_seen = max_seen.clone();
                let errors = errors.clone();
                thread::spawn(move || {
                    // Each OS thread builds its OWN `SlotPool` (a `!Send`,
                    // `Rc`-based handle) over the shared `Arc<GlobalSlotPool>` —
                    // exactly the shape `execute_scheduled_shard` would build
                    // per worker thread under `global`/`global-hop` dispatch.
                    let pool = SlotPool::new_global(global.clone());
                    let runtime = tokio::runtime::Builder::new_current_thread()
                        .enable_all()
                        .build()
                        .unwrap();
                    let local = tokio::task::LocalSet::new();
                    local.block_on(&runtime, async {
                        for _ in 0..6 {
                            let guard = pool.acquire().await;
                            let now =
                                concurrent.fetch_add(1, std::sync::atomic::Ordering::SeqCst) + 1;
                            max_seen.fetch_max(now, std::sync::atomic::Ordering::SeqCst);
                            if now > 2 {
                                errors.lock().unwrap().push(format!(
                                    "thread {thread_id} observed {now} concurrent holders \
                                     (cap is 2)"
                                ));
                            }
                            tokio::time::sleep(std::time::Duration::from_millis(2)).await;
                            concurrent.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
                            drop(guard);
                        }
                    });
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }

        let errors = errors.lock().unwrap();
        assert!(
            errors.is_empty(),
            "aggregate concurrency across all SlotPool handles must never exceed the \
             shared global cap: {errors:?}"
        );
        assert_eq!(
            max_seen.load(std::sync::atomic::Ordering::SeqCst),
            2,
            "the cap must actually be reached (proves the test exercises real \
             contention, not just an under-subscribed run)"
        );
    }

    #[test]
    fn global_set_limit_decrease_creates_debt_and_release_is_absorbed() {
        // Deterministic single-thread mirror of
        // `set_limit_decrease_creates_debt_and_release_is_absorbed` for the
        // `GlobalSlotPool` backend.
        let pool = GlobalSlotPool::new(4);
        let g1 = pool.try_acquire().unwrap();
        let g2 = pool.try_acquire().unwrap();
        // available = 2, debt = 0
        assert_eq!(pool.effective_slots(), 2);
        assert_eq!(pool.debt(), 0);

        // Decrease 4 -> 1: drain the 2 available, remaining shortfall (1) -> debt.
        pool.set_limit(1);
        assert_eq!(pool.current_limit(), 1);
        assert_eq!(pool.debt(), 1);
        assert_eq!(pool.effective_slots(), 0);

        // First release is absorbed by debt: no slot frees.
        drop(g1);
        assert_eq!(pool.debt(), 0);
        assert_eq!(pool.effective_slots(), 0);

        // Second release now frees a real slot, landing at the reduced cap of 1.
        drop(g2);
        assert_eq!(pool.effective_slots(), 1);
    }

    #[test]
    fn global_increase_cancels_debt_before_adding_slots() {
        let pool = GlobalSlotPool::new(4);
        let mut guards: Vec<GlobalSlotGuard> =
            (0..4).map(|_| pool.try_acquire().unwrap()).collect();
        assert_eq!(pool.effective_slots(), 0);

        pool.set_limit(1); // available 0, so all 3 removed become debt.
        assert_eq!(pool.debt(), 3);

        pool.set_limit(3); // increase by 2: cancels 2 debt, adds 0 real slots.
        assert_eq!(pool.debt(), 1);
        assert_eq!(pool.effective_slots(), 0);

        guards.pop(); // release -> debt 1 -> 0
        assert_eq!(pool.debt(), 0);
        assert_eq!(pool.effective_slots(), 0);
        guards.pop(); // release -> frees a real slot
        assert_eq!(pool.effective_slots(), 1);
        drop(guards);
    }

    /// The core property, under real cross-thread concurrency: after a limit
    /// decrease applied while holders are in flight, the pool must not free
    /// permits fast enough to let NEW holders push concurrency back above the
    /// reduced limit — in-flight holders must drain down first.
    ///
    /// Design that isolates debt from the benign "already-admitted holders
    /// drain above the new limit" noise: main holds all 8 initial slots, then
    /// decreases the limit to 2. Eight worker OS threads churn `acquire ->
    /// work -> release` continuously. Main then releases the 8 held guards one
    /// by one. We only record the max concurrency observed STRICTLY AFTER the
    /// 8 original guards are fully released, i.e. once the pool has settled at
    /// the reduced limit. With debt, the first 6 releases of the held guards
    /// are absorbed (concurrency drains 8 -> 2 admitting no new worker), so
    /// post-drain concurrency is capped at 2. WITHOUT debt, each held release
    /// frees a permit immediately, letting workers rush in and hold up to 8
    /// concurrent — the assertion would then fail.
    #[test]
    fn global_limit_decrease_never_transiently_over_admits() {
        use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
        use std::thread;

        let pool = GlobalSlotPool::new(8);
        // Hold all 8 slots on the main thread so the decrease produces debt.
        let held: Vec<GlobalSlotGuard> = (0..8).map(|_| pool.try_acquire().unwrap()).collect();

        let concurrent = Arc::new(AtomicUsize::new(0));
        let max_after_drain = Arc::new(AtomicUsize::new(0));
        // Flipped true once all 8 original held guards have been released;
        // only then do workers contribute to `max_after_drain`.
        let drained = Arc::new(AtomicBool::new(false));
        let stop = Arc::new(AtomicBool::new(false));

        let mut handles = Vec::new();
        for _ in 0..8 {
            let pool = pool.clone();
            let concurrent = concurrent.clone();
            let max_after_drain = max_after_drain.clone();
            let drained = drained.clone();
            let stop = stop.clone();
            handles.push(thread::spawn(move || {
                let runtime = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .unwrap();
                let local = tokio::task::LocalSet::new();
                local.block_on(&runtime, async {
                    while !stop.load(Ordering::Relaxed) {
                        let guard = pool.acquire().await;
                        let now = concurrent.fetch_add(1, Ordering::SeqCst) + 1;
                        if drained.load(Ordering::SeqCst) {
                            max_after_drain.fetch_max(now, Ordering::SeqCst);
                        }
                        tokio::time::sleep(std::time::Duration::from_millis(1)).await;
                        concurrent.fetch_sub(1, Ordering::SeqCst);
                        drop(guard);
                    }
                });
            }));
        }

        // Let workers pile up as waiters (all 8 slots are held by main).
        thread::sleep(std::time::Duration::from_millis(20));

        // Apply the debt-tracked decrease: available == 0, so all 6 removed
        // slots become debt.
        pool.set_limit(2);
        assert_eq!(pool.debt(), 6);

        // Release the 8 held guards one at a time; the first 6 must be absorbed
        // by debt (no worker admitted) before any real slot frees.
        for g in held {
            drop(g);
            thread::sleep(std::time::Duration::from_millis(2));
        }
        // Original holders fully drained; from here the pool has settled at 2.
        drained.store(true, Ordering::SeqCst);

        // Let workers churn under the settled reduced limit.
        thread::sleep(std::time::Duration::from_millis(60));
        stop.store(true, Ordering::Relaxed);
        for h in handles {
            h.join().unwrap();
        }

        let observed = max_after_drain.load(Ordering::SeqCst);
        assert!(
            observed <= 2,
            "post-decrease concurrency {observed} exceeded the reduced limit 2 \
             (debt failed to prevent transient over-admission)"
        );
        assert_eq!(
            observed, 2,
            "the reduced limit must actually be reached (proves real contention)"
        );
        assert_eq!(pool.current_limit(), 2);
        assert_eq!(pool.debt(), 0, "all debt repaid once holders drained");
    }
}
