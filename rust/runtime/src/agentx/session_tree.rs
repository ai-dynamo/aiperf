// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Per-session-tree slot-release accounting, ported from
//! `src/aiperf/timing/session_tree.py`.
//!
//! One session slot is held per live **tree** (a depth-0 root plus all recursive
//! descendants: subagents, flat-async, spawn/fork children), keyed by
//! `root_correlation_id`, and released exactly once when the tree DRAINS (root
//! done AND no descendant work outstanding). This bounds concurrency to the
//! configured slot count regardless of fan-out depth.
//!
//! The concurrency manager's `release_session_slot(phase)` is abstracted behind
//! [`SlotReleaser`]; the recycle `on_drain` hook is an optional callback. Phase
//! runtime keys are modeled as opaque string identities.

use std::collections::HashMap;

/// Opaque phase runtime key (the slot is released under the key it was acquired).
pub type PhaseKey = String;

/// The concurrency manager's slot-release seam.
pub trait SlotReleaser {
    /// Release one session slot acquired under `phase`.
    fn release_session_slot(&mut self, phase: &PhaseKey);
}

/// Liveness of one session tree (Python `_TreeState`).
#[derive(Debug, Clone)]
struct TreeState {
    phase: PhaseKey,
    root_pending: bool,
    outstanding: i64,
    released: bool,
}

impl TreeState {
    fn drained(&self) -> bool {
        !self.root_pending && self.outstanding <= 0
    }
}

/// Owns session-slot release per session tree (Python `SessionTreeRegistry`).
pub struct SessionTreeRegistry<R: SlotReleaser> {
    releaser: R,
    trees: HashMap<String, TreeState>,
    on_drain: Option<Box<dyn FnMut(&str, &PhaseKey)>>,
    pending_descendants: HashMap<String, i64>,
    peak_open: usize,
    late_events: i64,
}

impl<R: SlotReleaser> SessionTreeRegistry<R> {
    /// Construct with the concurrency-manager releaser.
    pub fn new(releaser: R) -> Self {
        Self {
            releaser,
            trees: HashMap::new(),
            on_drain: None,
            pending_descendants: HashMap::new(),
            peak_open: 0,
            late_events: 0,
        }
    }

    /// Register the drain callback (fired with `root_corr, phase` on normal
    /// release, NOT on `release_all` teardown).
    pub fn set_drain_callback(&mut self, callback: Option<Box<dyn FnMut(&str, &PhaseKey)>>) {
        self.on_drain = callback;
    }

    /// Maximum simultaneously-open trees seen (== peak session-slot occupancy).
    pub fn peak_open(&self) -> usize {
        self.peak_open
    }

    /// Count of returns for already-released trees (premature-drain evidence).
    pub fn late_events(&self) -> i64 {
        self.late_events
    }

    /// True when this registry is tracking `root_corr`.
    pub fn has_tree(&self, root_corr: &str) -> bool {
        self.trees.contains_key(root_corr)
    }

    /// Record a newly admitted tree after its session slot was acquired.
    pub fn open_tree(&mut self, root_corr: &str, phase: PhaseKey, root_pending: bool) {
        if let Some(existing) = self.trees.get(root_corr)
            && !existing.released
        {
            // Duplicate open for a still-live tree: keep the original.
            return;
        }
        let mut state = TreeState {
            phase,
            root_pending,
            outstanding: 0,
            released: false,
        };
        state.outstanding += self.pending_descendants.remove(root_corr).unwrap_or(0);
        self.trees.insert(root_corr.to_string(), state);
        if self.trees.len() > self.peak_open {
            self.peak_open = self.trees.len();
        }
    }

    /// Add `n` descendants (spawned or snapshot-seeded) to a tree. Buffered when
    /// the tree is not yet open.
    pub fn register_descendants(&mut self, root_corr: &str, n: i64) {
        if n <= 0 {
            return;
        }
        match self.trees.get_mut(root_corr) {
            Some(state) => state.outstanding += n,
            None => {
                *self
                    .pending_descendants
                    .entry(root_corr.to_string())
                    .or_insert(0) += n;
            }
        }
    }

    /// Account one descendant terminally completing. Releases the slot iff the
    /// tree is now drained. Returns true if the slot was released.
    pub fn on_descendant_done(&mut self, root_corr: &str) -> bool {
        if !self.trees.contains_key(root_corr) {
            // Tree not open yet: decrement the pending buffer, or count a late event.
            let pending = self
                .pending_descendants
                .get(root_corr)
                .copied()
                .unwrap_or(0);
            if pending > 1 {
                self.pending_descendants
                    .insert(root_corr.to_string(), pending - 1);
            } else if pending == 1 {
                self.pending_descendants.remove(root_corr);
            } else {
                self.late_events += 1;
            }
            return false;
        }
        if let Some(state) = self.trees.get_mut(root_corr)
            && state.outstanding > 0
        {
            state.outstanding -= 1;
        }
        self.maybe_release(root_corr)
    }

    /// Account the root's terminal turn returning. Clears `root_pending`;
    /// releases the slot iff the tree is now drained.
    pub fn on_root_terminal(&mut self, root_corr: &str) -> bool {
        if let Some(state) = self.trees.get_mut(root_corr) {
            state.root_pending = false;
        } else {
            return false;
        }
        self.maybe_release(root_corr)
    }

    fn maybe_release(&mut self, root_corr: &str) -> bool {
        let should = match self.trees.get(root_corr) {
            Some(s) => !s.released && s.drained(),
            None => false,
        };
        if !should {
            return false;
        }
        let state = self.trees.remove(root_corr).unwrap();
        self.releaser.release_session_slot(&state.phase);
        if let Some(cb) = self.on_drain.as_mut() {
            cb(root_corr, &state.phase);
        }
        true
    }

    /// Release every still-open tree's slot for `phase` at teardown. Does NOT
    /// fire the drain callback. Returns the number of slots released.
    pub fn release_all(&mut self, phase: &PhaseKey) -> usize {
        let to_release: Vec<String> = self
            .trees
            .iter()
            .filter(|(_, s)| &s.phase == phase && !s.released)
            .map(|(k, _)| k.clone())
            .collect();
        for root_corr in &to_release {
            if let Some(state) = self.trees.remove(root_corr)
                && !state.released
            {
                self.releaser.release_session_slot(phase);
            }
        }
        to_release.len()
    }

    /// Number of trees currently holding a slot (optionally for one phase).
    pub fn open_count(&self, phase: Option<&PhaseKey>) -> usize {
        match phase {
            None => self.trees.len(),
            Some(p) => self.trees.values().filter(|s| &s.phase == p).count(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Default)]
    struct RecRel {
        released: Vec<PhaseKey>,
    }
    impl SlotReleaser for RecRel {
        fn release_session_slot(&mut self, phase: &PhaseKey) {
            self.released.push(phase.clone());
        }
    }

    #[test]
    fn tree_drains_on_root_and_descendants() {
        let mut reg = SessionTreeRegistry::new(RecRel::default());
        reg.open_tree("root", "p0".into(), true);
        reg.register_descendants("root", 2);
        // Root terminal but descendants remain -> not released.
        assert!(!reg.on_root_terminal("root"));
        assert!(!reg.on_descendant_done("root"));
        // Last descendant done -> drained -> released.
        assert!(reg.on_descendant_done("root"));
        assert_eq!(reg.releaser.released, vec!["p0".to_string()]);
        assert!(!reg.has_tree("root"));
    }

    #[test]
    fn pending_descendants_folded_at_open() {
        let mut reg = SessionTreeRegistry::new(RecRel::default());
        // Snapshot seeds a descendant before the tree opens.
        reg.register_descendants("root", 1);
        reg.open_tree("root", "p0".into(), false); // rootless lane
        // The pre-seeded descendant keeps the tree alive until it completes.
        assert!(reg.on_descendant_done("root"));
    }

    #[test]
    fn descendant_done_before_open_decrements_pending() {
        let mut reg = SessionTreeRegistry::new(RecRel::default());
        reg.register_descendants("root", 2);
        // One completes before the tree opens -> pending folds to 1.
        assert!(!reg.on_descendant_done("root"));
        reg.open_tree("root", "p0".into(), false);
        // Only one descendant remains outstanding.
        assert!(reg.on_descendant_done("root"));
        assert_eq!(reg.late_events(), 0);
    }
}
