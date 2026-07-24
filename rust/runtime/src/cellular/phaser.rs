// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Monotonic phaser control plane.
//!
//! A phaser is a monotonic generation counter the controller increments as the
//! benchmark progresses and broadcasts to all cells with replay on attach. It
//! carries synchronized start, dataset availability, phase transitions, and
//! completion.
//!
//! Monotonic + replay-on-attach ⇒ a late-joining cell reads "current generation = G"
//! atomically, then follows live events without missing a transition. Consumers
//! gate on `generation >= threshold`, never equality, so cyclic progressions
//! cannot suffer ABA.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::{collections::BTreeMap, fmt};

use serde::{Deserialize, Serialize};

use super::broadcast::{Broadcast, BroadcastEvent, Subscription};

/// Meaning carried by one monotonic generation step.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PhaseTransition {
    /// Generation 1: begin the benchmark (the synchronized/`barrier-free` START).
    Started,
    /// Dataset shards `[0, k)` are available to pull.
    ShardsAvailable(u64),
    /// A benchmark phase boundary (e.g. `"warmup"` → `"profiling"` → `"drain"`).
    PhaseAdvance(String),
    /// Terminal: the run is complete; no further generations follow.
    Done,
}

impl PhaseTransition {
    /// The named phase when this transition is a [`Self::PhaseAdvance`].
    pub fn phase_name(&self) -> Option<&str> {
        match self {
            Self::PhaseAdvance(name) => Some(name.as_str()),
            _ => None,
        }
    }
}

/// One phaser step: a monotonic `generation` and what it signals.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PhaseEvent {
    /// Monotonic generation (starts at 0 = "not started"; the first `advance` is 1).
    pub generation: u64,
    /// What this generation means.
    pub transition: PhaseTransition,
}

/// Producer-side phaser with shared broadcast and generation state.
#[derive(Clone)]
pub struct Phaser {
    broadcast: Broadcast<PhaseEvent>,
    generation: Arc<AtomicU64>,
}

impl Default for Phaser {
    fn default() -> Self {
        Self::new()
    }
}

impl Phaser {
    /// A fresh phaser at generation 0 (not started).
    pub fn new() -> Self {
        Self {
            broadcast: Broadcast::new(),
            generation: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Increment the generation and broadcast the transition. Returns the new
    /// generation. Monotonic and cyclic-safe: the counter never decreases or resets, so
    /// consumers gate on `>=`. A no-op-safe `Done` should be the last `advance`; call
    /// [`Self::finalize`] after it to seal the stream.
    pub fn advance(&self, transition: PhaseTransition) -> u64 {
        let generation = self.generation.fetch_add(1, Ordering::SeqCst) + 1;
        self.broadcast.add(PhaseEvent {
            generation,
            transition,
        });
        generation
    }

    /// Convenience: `advance(Done)` then seal the stream so every subscriber (incl. a
    /// late attach) sees the terminal and their `next`/`await` completes.
    pub fn finalize(&self) {
        self.advance(PhaseTransition::Done);
        self.broadcast.finalize();
    }

    /// The current generation (diagnostics; a subscriber reads it via replay instead).
    pub fn current_generation(&self) -> u64 {
        self.generation.load(Ordering::SeqCst)
    }

    /// Subscribe a consumer. It receives replay-on-attach: every generation so far
    /// (so it learns the current generation atomically) then the live tail.
    pub fn subscribe(&self) -> PhaserSubscription {
        PhaserSubscription::from_subscription(self.broadcast.attach())
    }

    /// The raw broadcast subscription (replay snapshot + live receiver, split at the
    /// atomic attach seam) — for the velo distribution layer, which ships the replay in
    /// the subscribe response and pushes the live tail to the cell. In-process callers
    /// use [`Self::subscribe`].
    pub fn attach_raw(&self) -> Subscription<PhaseEvent> {
        self.broadcast.attach()
    }
}

/// Consumer-side view of the phaser: replay then live, tracking the highest generation
/// observed. `await_generation` blocks until the producer reaches a target generation.
pub struct PhaserSubscription {
    sub: Subscription<PhaseEvent>,
    /// Cursor into the replay snapshot.
    cursor: usize,
    /// Highest generation observed so far.
    seen_generation: u64,
    /// Named phase-advance generations already observed through replay or live events.
    seen_phase_advances: BTreeMap<String, u64>,
    /// Set once the terminal `Finalized` was observed.
    finalized: bool,
}

impl PhaserSubscription {
    /// Wrap a raw broadcast [`Subscription`] (replay + live) as a phaser subscription.
    /// The velo cell client uses this to reconstruct a subscription from the replay it
    /// received in the subscribe response plus the live channel the push handler feeds.
    pub fn from_subscription(sub: Subscription<PhaseEvent>) -> Self {
        Self {
            sub,
            cursor: 0,
            seen_generation: 0,
            seen_phase_advances: BTreeMap::new(),
            finalized: false,
        }
    }

    /// The highest generation this subscriber has observed (from replay + drained live).
    pub fn seen_generation(&self) -> u64 {
        self.seen_generation
    }

    /// The generation at which this subscriber observed the named phase advance, if any.
    pub fn seen_phase_advance(&self, phase: &str) -> Option<u64> {
        self.seen_phase_advances.get(phase).copied()
    }

    /// Pull the next phaser event (replay first, then live), updating the observed
    /// generation. `None` once the stream is finalized and drained.
    pub async fn next(&mut self) -> Option<PhaseEvent> {
        if self.cursor < self.sub.replay.len() {
            let event = self.sub.replay[self.cursor].clone();
            self.cursor += 1;
            return self.record(event);
        }
        if self.finalized {
            return None;
        }
        match self.sub.live.recv().await {
            Some(event) => self.record(event),
            None => {
                self.finalized = true;
                None
            }
        }
    }

    fn record(&mut self, event: BroadcastEvent<PhaseEvent>) -> Option<PhaseEvent> {
        match event {
            BroadcastEvent::Item(phase) => {
                self.seen_generation = self.seen_generation.max(phase.generation);
                if let Some(name) = phase.transition.phase_name() {
                    self.seen_phase_advances
                        .insert(name.to_owned(), phase.generation);
                }
                Some(phase)
            }
            BroadcastEvent::Finalized => {
                self.finalized = true;
                None
            }
        }
    }

    /// Block until the producer has reached `target` generation (`generation >= target`).
    /// Returns `Ok(())` on reaching it, or `Err` if the stream finalized first (the
    /// producer sealed the phaser before that generation — e.g. an aborted run). Because
    /// generations are monotonic, "reached target" = observing any event with
    /// `generation >= target`; already-passed targets return immediately from replay.
    pub async fn await_generation(&mut self, target: u64) -> Result<(), PhaserClosed> {
        if self.seen_generation >= target {
            return Ok(());
        }
        while let Some(event) = self.next().await {
            if event.generation >= target {
                return Ok(());
            }
        }
        Err(PhaserClosed)
    }

    /// Block until the named phase advance has been observed through replay or the live
    /// tail, returning the generation that carried it.
    pub async fn await_phase_advance(&mut self, phase: &str) -> Result<u64, PhaserClosed> {
        if let Some(generation) = self.seen_phase_advance(phase) {
            return Ok(generation);
        }
        while let Some(event) = self.next().await {
            if matches!(
                &event.transition,
                PhaseTransition::PhaseAdvance(name) if name == phase
            ) {
                return Ok(event.generation);
            }
        }
        Err(PhaserClosed)
    }

    /// Await the START (generation 1). Sugar over `await_generation(1)`.
    pub async fn await_started(&mut self) -> Result<(), PhaserClosed> {
        self.await_generation(1).await
    }
}

/// The phaser was finalized before the awaited generation was reached (an aborted or
/// completed run). A plain marker error per the crate's error convention.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PhaserClosed;

impl std::fmt::Display for PhaserClosed {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "phaser finalized before the awaited generation was reached"
        )
    }
}

impl std::error::Error for PhaserClosed {}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn generations_are_monotonic_and_carry_transitions() {
        let p = Phaser::new();
        assert_eq!(p.current_generation(), 0);
        assert_eq!(p.advance(PhaseTransition::Started), 1);
        assert_eq!(p.advance(PhaseTransition::ShardsAvailable(10)), 2);
        assert_eq!(
            p.advance(PhaseTransition::PhaseAdvance("profiling".into())),
            3
        );
        assert_eq!(p.current_generation(), 3);

        let mut sub = p.subscribe();
        assert_eq!(
            sub.next().await.unwrap().transition,
            PhaseTransition::Started
        );
        assert_eq!(
            sub.next().await.unwrap().transition,
            PhaseTransition::ShardsAvailable(10)
        );
        assert_eq!(
            sub.next().await.unwrap().transition,
            PhaseTransition::PhaseAdvance("profiling".into())
        );
    }

    #[tokio::test]
    async fn await_generation_returns_from_replay_for_already_passed_targets() {
        let p = Phaser::new();
        p.advance(PhaseTransition::Started);
        p.advance(PhaseTransition::ShardsAvailable(5));
        // A cell that attaches late still reaches generation 2 immediately (replay).
        let mut sub = p.subscribe();
        sub.await_generation(2)
            .await
            .expect("gen 2 already reached");
        assert!(sub.seen_generation() >= 2);
    }

    #[tokio::test]
    async fn await_generation_blocks_then_wakes_on_live_advance() {
        let p = Phaser::new();
        let mut sub = p.subscribe();
        let waiter = tokio::spawn(async move {
            sub.await_started().await.expect("started");
            sub.await_generation(3).await.expect("gen 3");
            sub.seen_generation()
        });
        // Not started yet; drive the phaser forward.
        p.advance(PhaseTransition::Started);
        p.advance(PhaseTransition::PhaseAdvance("warmup".into()));
        p.advance(PhaseTransition::PhaseAdvance("profiling".into()));
        assert!(waiter.await.unwrap() >= 3);
    }

    #[tokio::test]
    async fn cyclic_gate_on_ge_never_reuses_a_generation() {
        // Multi-round: never reset; each round is a higher generation. A consumer
        // gating on `>= round*2` advances round by round with no ABA.
        let p = Phaser::new();
        for round in 1..=3u64 {
            p.advance(PhaseTransition::PhaseAdvance(format!(
                "round-{round}-start"
            )));
            p.advance(PhaseTransition::PhaseAdvance(format!("round-{round}-end")));
        }
        let mut sub = p.subscribe();
        for round in 1..=3u64 {
            sub.await_generation(round * 2 - 1)
                .await
                .expect("round start");
            sub.await_generation(round * 2).await.expect("round end");
        }
        assert_eq!(sub.seen_generation(), 6);
    }

    #[tokio::test]
    async fn await_after_finalize_errors_if_target_never_reached() {
        let p = Phaser::new();
        p.advance(PhaseTransition::Started); // generation 1
        p.finalize(); // advance(Done)=2, then seal
        let mut sub = p.subscribe();
        // Generation 2 (Done) was reached, so awaiting 2 is fine...
        sub.await_generation(2)
            .await
            .expect("reached Done generation");
        // ...but a target beyond the sealed stream errors.
        let mut sub2 = p.subscribe();
        assert!(sub2.await_generation(99).await.is_err());
    }
}
