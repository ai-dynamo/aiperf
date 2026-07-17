// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Producer-owned SPMC broadcast with replay on attach.
//!
//! One producer appends items and many consumers receive the complete prefix
//! followed by the live tail. `attach`, `add`, and `finalize` share one lock, so
//! each consumer observes producer order without a gap or duplicate at its
//! replay/live boundary.

use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;

/// Locks the inner state, recovering the guard even if a prior holder panicked. The
/// broadcast is control-plane (off the per-request hot path) and holds the lock only
/// for O(consumers) fan-out, so a `std` mutex is sufficient; poison only means an
/// earlier panic, and the invariants here are re-established on each call.
fn lock<T>(inner: &Mutex<Inner<T>>) -> std::sync::MutexGuard<'_, Inner<T>> {
    inner
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// One event a consumer observes on its subscription stream: an item (in producer
/// `add` order across the replay ⊎ live boundary), or the terminal `Finalized`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum BroadcastEvent<T> {
    /// An item the producer added.
    Item(T),
    /// The producer sealed the stream; no more items follow. Delivered exactly once
    /// to every consumer (in its replay if it attached after `finalize`).
    Finalized,
}

/// A consumer's subscription: the replay snapshot (everything added before this
/// consumer attached, in producer order, plus a trailing [`BroadcastEvent::Finalized`]
/// if the stream was already sealed) followed by the live receiver for subsequent
/// items. Draining `replay` then `live` yields the producer's full `add` order with no
/// gap or duplicate at the attach seam.
pub struct Subscription<T> {
    /// Items (and a trailing `Finalized`) that were added before this attach, in order.
    pub replay: Vec<BroadcastEvent<T>>,
    /// Live events added after this attach.
    pub live: mpsc::UnboundedReceiver<BroadcastEvent<T>>,
}

impl<T: Clone> Subscription<T> {
    /// Drains the whole subscription — replay then live — into one ordered `Vec`,
    /// stopping at (and including) the terminal `Finalized`. Convenience for consumers
    /// that want the complete stream rather than to react incrementally; requires the
    /// producer to eventually `finalize` (else the live half blocks forever).
    pub async fn collect_until_finalized(mut self) -> Vec<T> {
        let mut out = Vec::new();
        for event in self.replay.drain(..) {
            match event {
                BroadcastEvent::Item(item) => out.push(item),
                BroadcastEvent::Finalized => return out,
            }
        }
        while let Some(event) = self.live.recv().await {
            match event {
                BroadcastEvent::Item(item) => out.push(item),
                BroadcastEvent::Finalized => break,
            }
        }
        out
    }
}

struct Inner<T> {
    /// Append-only producer history in `add` order.
    history: Vec<T>,
    /// Set once `finalize` is called; a late attach then replays `Finalized` at the end.
    finalized: bool,
    /// One live sender per currently-attached consumer. A closed sender (consumer
    /// dropped its receiver) is pruned on the next `add`/`finalize`.
    senders: Vec<mpsc::UnboundedSender<BroadcastEvent<T>>>,
}

/// A producer-owned SPMC broadcast: one producer `add`s items; many consumers `attach`
/// and each gets a replay-on-attach snapshot then the live tail. Cheap to clone
/// (`Arc`); clones share one broadcast.
#[derive(Clone)]
pub struct Broadcast<T> {
    inner: Arc<Mutex<Inner<T>>>,
}

impl<T: Clone> Default for Broadcast<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Clone> Broadcast<T> {
    /// A fresh, empty broadcast.
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(Inner {
                history: Vec::new(),
                finalized: false,
                senders: Vec::new(),
            })),
        }
    }

    /// Snapshots replay history and registers the live sender under one lock, so
    /// no item can slip between them.
    pub fn attach(&self) -> Subscription<T> {
        let mut inner = lock(&self.inner);
        let mut replay: Vec<BroadcastEvent<T>> = inner
            .history
            .iter()
            .cloned()
            .map(BroadcastEvent::Item)
            .collect();
        if inner.finalized {
            // A consumer that attaches after finalize gets the full history + terminal;
            // it never needs a live sender (nothing more will be added).
            replay.push(BroadcastEvent::Finalized);
            return Subscription {
                replay,
                live: mpsc::unbounded_channel().1,
            };
        }
        let (tx, live) = mpsc::unbounded_channel();
        inner.senders.push(tx);
        Subscription { replay, live }
    }

    /// Appends and fans out under the same lock as [`Self::attach`]. Returns
    /// `false` after finalization.
    pub fn add(&self, item: T) -> bool {
        let mut inner = lock(&self.inner);
        if inner.finalized {
            return false;
        }
        inner.history.push(item.clone());
        inner
            .senders
            .retain(|tx| tx.send(BroadcastEvent::Item(item.clone())).is_ok());
        true
    }

    /// Idempotently seals the stream. Late subscribers receive the full history
    /// followed by `Finalized`.
    pub fn finalize(&self) {
        let mut inner = lock(&self.inner);
        if inner.finalized {
            return;
        }
        inner.finalized = true;
        for tx in inner.senders.drain(..) {
            let _ = tx.send(BroadcastEvent::Finalized);
        }
    }

    /// The number of items added so far (diagnostics).
    pub fn len(&self) -> usize {
        lock(&self.inner).history.len()
    }

    /// Whether no item has been added yet.
    pub fn is_empty(&self) -> bool {
        lock(&self.inner).history.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every consumer's (replay ⊎ live) equals the producer's full `add` order,
    /// regardless of when it attached — the core contract.
    #[tokio::test]
    async fn replay_plus_live_reconstructs_full_order_for_every_attach_time() {
        let b = Broadcast::<u32>::new();
        // Consumer A attaches before anything is added.
        let a = b.attach();
        b.add(0);
        b.add(1);
        // Consumer B attaches mid-stream (sees {0,1} as replay, then the live tail).
        let bee = b.attach();
        b.add(2);
        b.add(3);
        // Consumer C attaches even later.
        let c = b.attach();
        b.add(4);
        b.finalize();
        let d = b.attach();

        let full = vec![0, 1, 2, 3, 4];
        assert_eq!(a.collect_until_finalized().await, full);
        assert_eq!(bee.collect_until_finalized().await, full);
        assert_eq!(c.collect_until_finalized().await, full);
        assert_eq!(d.collect_until_finalized().await, full);
    }

    /// The attach seam: an item added "concurrently" with an attach lands in exactly
    /// one of {replay, live} — never both, never neither. We can't truly race a
    /// `parking_lot::Mutex` deterministically, but we assert the boundary invariant by
    /// interleaving attach and add and checking each consumer sees a gap-free prefix.
    #[tokio::test]
    async fn no_gap_or_duplicate_at_the_attach_seam() {
        let b = Broadcast::<u32>::new();
        let mut subs = Vec::new();
        // Interleave: attach a consumer, then add, repeatedly. Each new consumer must
        // see every item from its attach point onward with none missing or doubled.
        for i in 0..50u32 {
            subs.push((i, b.attach()));
            b.add(i);
        }
        b.finalize();
        for (attached_at, sub) in subs {
            let got = sub.collect_until_finalized().await;
            // This consumer must see a contiguous suffix starting no later than its
            // attach index (items added strictly before attach are replay; the item
            // added right after is live) — and the union is gap-free.
            let expected: Vec<u32> = (0..50).collect();
            assert_eq!(
                got, expected,
                "consumer attached at {attached_at} lost or duplicated an item"
            );
        }
    }

    #[tokio::test]
    async fn add_after_finalize_is_rejected_and_finalize_is_idempotent() {
        let b = Broadcast::<u32>::new();
        b.add(1);
        b.finalize();
        assert!(!b.add(2), "add after finalize must be rejected");
        b.finalize(); // idempotent, no panic
        let sub = b.attach();
        assert_eq!(sub.collect_until_finalized().await, vec![1]);
    }

    #[tokio::test]
    async fn a_dropped_consumer_does_not_block_the_producer_or_others() {
        let b = Broadcast::<u32>::new();
        let slow = b.attach();
        let live = b.attach();
        b.add(1);
        // Drop the slow consumer without draining — its sender closes; the producer
        // prunes it on the next add and keeps serving the live consumer.
        drop(slow);
        b.add(2);
        b.finalize();
        assert_eq!(live.collect_until_finalized().await, vec![1, 2]);
    }
}
