// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Byte-exact port of the Python replay interval-barrier coordinator
//! (`ReplayBarrierCoordinator` in `src/aiperf/timing/replay_dependencies.py`,
//! lines 156-348).
//!
//! A turn is released (issued) only after all of its recorded
//! [`ReplayTurnKey`] predecessors have reached a terminal outcome. Per runtime
//! root, the coordinator holds a completion frontier plus the dispatches waiting
//! on it; when a completion makes retained turns ready it releases every newly
//! ready turn in `sorted(key)` order.
//!
//! The Python coordinator is async and dispatches releases through detached
//! `asyncio` tasks. This is the single-central-driver ("global-hop") port: it
//! carries no async or I/O and models a "release" as synchronously pushing the
//! turn's key onto an ordered [`ReplayGate::released`] output list. Ordered
//! iteration uses [`BTreeMap`]/[`BTreeSet`] so releases and snapshots are
//! deterministic and match the Python `sorted(...)` semantics exactly.
//!
//! The `ReplayIssueGate` async CreditIssuer adapter
//! (`src/aiperf/timing/replay_dependencies.py`, lines 351-434) is NOT ported:
//! it is a thin transport-facing wrapper with no barrier logic of its own.

use std::collections::{BTreeMap, BTreeSet};

use super::replay_dependencies::{ReplayResumeBoundary, ReplayTurnKey};

/// A turn queued through the barrier gate. The gate reads only the runtime root
/// id and the turn's dataset identity (`conversation_id`, `turn_index`), which is
/// exactly what the Python coordinator reads off `TurnToSend`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReplayTurn {
    /// Effective root correlation id of this turn's session tree
    /// (`TurnToSend.effective_root_correlation_id`).
    pub root_id: String,
    /// Dataset identity (conversation + turn) of this request.
    pub key: ReplayTurnKey,
}

impl ReplayTurn {
    /// Construct a turn for `root_id` addressing `(conversation_id, turn_index)`.
    pub fn new(
        root_id: impl Into<String>,
        conversation_id: impl Into<String>,
        turn_index: i64,
    ) -> Self {
        Self {
            root_id: root_id.into(),
            key: ReplayTurnKey {
                conversation_id: conversation_id.into(),
                turn_index,
            },
        }
    }
}

/// Errors raised by the coordinator, mirroring the Python `RuntimeError`/
/// `ValueError` conditions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReplayGateError {
    /// A second deferred dispatch was submitted for a key already retained
    /// (Python `RuntimeError: Duplicate deferred replay dispatch`).
    DuplicatePending {
        /// Runtime root the duplicate was submitted under.
        root_id: String,
        /// The already-retained key.
        key: ReplayTurnKey,
    },
    /// `seed_completed_prefixes` was called after a turn had already been
    /// retained for the root (Python `RuntimeError: Cannot seed ... after dispatch`).
    SeedAfterDispatch {
        /// Runtime root that already had pending dispatches.
        root_id: String,
    },
    /// A resume boundary carried a negative `next_turn_index`
    /// (Python `ValueError`).
    NegativeBoundaryIndex {
        /// The offending conversation id.
        conversation_id: String,
    },
    /// The recorded completion history for a conversation is not a contiguous
    /// prefix (Python `RuntimeError: ... not a contiguous stream prefix`).
    NonContiguousPrefix {
        /// Runtime root whose history had a hole.
        root_id: String,
        /// The conversation with the non-contiguous prefix.
        conversation_id: String,
    },
}

/// Per-tree completion frontier and the dispatches waiting on it
/// (Python `_RootBarrierState`).
#[derive(Debug, Default)]
struct RootBarrierState {
    /// Keys of requests on this tree that have recorded completion.
    completed: BTreeSet<ReplayTurnKey>,
    /// Dispatches keyed by request, waiting on their predecessors to complete.
    pending: BTreeMap<ReplayTurnKey, ReplayTurn>,
}

/// Release requests only after their recorded frontier has completed
/// (Python `ReplayBarrierCoordinator`, single-central-driver port).
#[derive(Debug)]
pub struct ReplayGate {
    /// Fixed per-key predecessor map derived from the recorded dataset.
    predecessors: BTreeMap<ReplayTurnKey, Vec<ReplayTurnKey>>,
    /// Per-runtime-root barrier state.
    roots: BTreeMap<String, RootBarrierState>,
    /// Barriers apply only once active; before activation submits issue eagerly.
    active: bool,
    /// When set, newly ready dispatches are retained rather than released.
    releases_paused: bool,
    /// Ordered log of released (issued) keys, one push per issue.
    released: Vec<ReplayTurnKey>,
}

impl ReplayGate {
    /// Build a coordinator over a fixed predecessor map. Keys absent from the map
    /// are treated as having no predecessors (Python `_predecessors.get(key, ())`).
    pub fn new(predecessors: BTreeMap<ReplayTurnKey, Vec<ReplayTurnKey>>) -> Self {
        Self {
            predecessors,
            roots: BTreeMap::new(),
            active: false,
            releases_paused: false,
            released: Vec::new(),
        }
    }

    /// Enable barriers after baseline cache priming completes. Idempotent.
    pub fn activate(&mut self) {
        self.active = true;
    }

    /// Retain newly ready dispatches for an explicit phase handoff.
    pub fn pause_releases(&mut self) {
        self.releases_paused = true;
    }

    /// Issue now when ready, otherwise retain one deferred dispatch.
    ///
    /// Before activation, or when the turn's predecessors are already complete
    /// and releases are not paused, the turn is issued immediately (pushed to the
    /// release log). Otherwise it is retained; retaining a key twice is an error.
    pub fn submit(&mut self, turn: ReplayTurn) -> Result<(), ReplayGateError> {
        if !self.active {
            self.released.push(turn.key.clone());
            return Ok(());
        }
        let root_id = turn.root_id.clone();
        let ready = self.ready(&root_id, &turn.key);
        let state = self.roots.entry(root_id.clone()).or_default();
        if ready && !self.releases_paused {
            self.released.push(turn.key.clone());
            return Ok(());
        }
        if state.pending.contains_key(&turn.key) {
            return Err(ReplayGateError::DuplicatePending {
                root_id,
                key: turn.key,
            });
        }
        state.pending.insert(turn.key.clone(), turn);
        Ok(())
    }

    /// Record any terminal request outcome and release newly ready work.
    ///
    /// No-op before activation. Adds the key to the root's completed set, then
    /// (unless paused) releases every retained turn that is now ready in
    /// `sorted(key)` order.
    pub fn complete(&mut self, root_id: &str, key: ReplayTurnKey) {
        if !self.active {
            return;
        }
        self.roots
            .entry(root_id.to_string())
            .or_default()
            .completed
            .insert(key);
        if self.releases_paused {
            return;
        }
        // Collect newly ready keys (BTreeMap iterates in sorted key order).
        let ready_keys: Vec<ReplayTurnKey> = {
            let state = self.roots.get(root_id).expect("root inserted above");
            let completed = &state.completed;
            state
                .pending
                .keys()
                .filter(|k| {
                    self.predecessors
                        .get(*k)
                        .map(|preds| preds.iter().all(|p| completed.contains(p)))
                        .unwrap_or(true)
                })
                .cloned()
                .collect()
        };
        let state = self.roots.get_mut(root_id).expect("root inserted above");
        for k in ready_keys {
            state.pending.remove(&k);
            self.released.push(k);
        }
    }

    /// Discard completed runtime state when a recycled tree drains.
    pub fn close_root(&mut self, root_id: &str) {
        self.roots.remove(root_id);
    }

    /// Seed exact pre-resume history before any turn can be submitted.
    ///
    /// Errors if the root already has retained dispatches, or if a boundary
    /// carries a negative `next_turn_index`.
    pub fn seed_completed_prefixes(
        &mut self,
        root_id: &str,
        boundaries: &[ReplayResumeBoundary],
    ) -> Result<(), ReplayGateError> {
        let state = self.roots.entry(root_id.to_string()).or_default();
        if !state.pending.is_empty() {
            return Err(ReplayGateError::SeedAfterDispatch {
                root_id: root_id.to_string(),
            });
        }
        for boundary in boundaries {
            if boundary.next_turn_index < 0 {
                return Err(ReplayGateError::NegativeBoundaryIndex {
                    conversation_id: boundary.conversation_id.clone(),
                });
            }
            for turn_index in 0..boundary.next_turn_index {
                state.completed.insert(ReplayTurnKey {
                    conversation_id: boundary.conversation_id.clone(),
                    turn_index,
                });
            }
        }
        Ok(())
    }

    /// Return the contiguous completed prefix of every replay stream, sorted by
    /// conversation id. Errors if any conversation's history has a hole.
    pub fn completed_prefixes(
        &self,
        root_id: &str,
    ) -> Result<Vec<ReplayResumeBoundary>, ReplayGateError> {
        let state = match self.roots.get(root_id) {
            Some(state) => state,
            None => return Ok(Vec::new()),
        };
        // next_turn per conversation = max(turn_index + 1). BTreeMap keeps
        // conversations sorted for the deterministic output order.
        let mut next_turn: BTreeMap<String, i64> = BTreeMap::new();
        for key in &state.completed {
            let entry = next_turn.entry(key.conversation_id.clone()).or_insert(0);
            *entry = (*entry).max(key.turn_index + 1);
        }
        for (conversation_id, &next_turn_index) in &next_turn {
            let hole = (0..next_turn_index).any(|turn_index| {
                !state.completed.contains(&ReplayTurnKey {
                    conversation_id: conversation_id.clone(),
                    turn_index,
                })
            });
            if hole {
                return Err(ReplayGateError::NonContiguousPrefix {
                    root_id: root_id.to_string(),
                    conversation_id: conversation_id.clone(),
                });
            }
        }
        Ok(next_turn
            .into_iter()
            .map(|(conversation_id, next_turn_index)| ReplayResumeBoundary {
                conversation_id,
                next_turn_index,
            })
            .collect())
    }

    /// Return barrier-retained turns for one root, sorted by [`ReplayTurnKey`].
    pub fn pending_turns(&self, root_id: &str) -> Vec<ReplayTurn> {
        match self.roots.get(root_id) {
            Some(state) => state.pending.values().cloned().collect(),
            None => Vec::new(),
        }
    }

    /// Return all barrier-retained turns grouped by runtime root id. Roots with
    /// no retained turns are omitted (mirrors the Python dict comprehension guard).
    pub fn pending_turns_by_root(&self) -> BTreeMap<String, Vec<ReplayTurn>> {
        self.roots
            .iter()
            .filter(|(_, state)| !state.pending.is_empty())
            .map(|(root_id, state)| (root_id.clone(), state.pending.values().cloned().collect()))
            .collect()
    }

    /// The ordered log of released (issued) keys.
    pub fn released(&self) -> &[ReplayTurnKey] {
        &self.released
    }

    /// A turn is ready iff all recorded predecessors are in the root's completed
    /// set (Python `_ready`). Absent predecessor entry => no predecessors.
    fn ready(&self, root_id: &str, key: &ReplayTurnKey) -> bool {
        let preds = match self.predecessors.get(key) {
            Some(preds) => preds,
            None => return true,
        };
        match self.roots.get(root_id) {
            Some(state) => preds.iter().all(|p| state.completed.contains(p)),
            None => preds.is_empty(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(conv: &str, ti: i64) -> ReplayTurnKey {
        ReplayTurnKey {
            conversation_id: conv.into(),
            turn_index: ti,
        }
    }

    /// Predecessor map used across the unit tests: B0<-A0, C0<-{A1,B0}.
    fn preds() -> BTreeMap<ReplayTurnKey, Vec<ReplayTurnKey>> {
        let mut m = BTreeMap::new();
        m.insert(key("A", 0), vec![]);
        m.insert(key("A", 1), vec![]);
        m.insert(key("B", 0), vec![key("A", 0)]);
        m.insert(key("C", 0), vec![key("A", 1), key("B", 0)]);
        m
    }

    #[test]
    fn barrier_releases_in_sorted_order_when_frontier_completes() {
        let mut gate = ReplayGate::new(preds());
        gate.activate();
        gate.submit(ReplayTurn::new("R", "A", 0)).unwrap(); // ready -> issue
        gate.submit(ReplayTurn::new("R", "B", 0)).unwrap(); // retained
        gate.submit(ReplayTurn::new("R", "C", 0)).unwrap(); // retained
        gate.submit(ReplayTurn::new("R", "A", 1)).unwrap(); // ready -> issue
        assert_eq!(gate.released(), &[key("A", 0), key("A", 1)]);

        gate.complete("R", key("A", 0)); // releases B0
        gate.complete("R", key("A", 1)); // C0 still needs B0 completion
        gate.complete("R", key("B", 0)); // releases C0
        assert_eq!(
            gate.released(),
            &[key("A", 0), key("A", 1), key("B", 0), key("C", 0)]
        );
        assert!(gate.pending_turns("R").is_empty());
    }

    #[test]
    fn pause_retains_newly_ready_and_seed_marks_prefix() {
        let mut gate = ReplayGate::new(preds());
        gate.activate();

        // Seeding a resume prefix marks the 0..next range completed.
        gate.seed_completed_prefixes(
            "R2",
            &[ReplayResumeBoundary {
                conversation_id: "D".into(),
                next_turn_index: 2,
            }],
        )
        .unwrap();
        assert_eq!(
            gate.completed_prefixes("R2").unwrap(),
            vec![ReplayResumeBoundary {
                conversation_id: "D".into(),
                next_turn_index: 2,
            }]
        );

        // Under pause, a ready turn is retained and completion releases nothing.
        gate.pause_releases();
        gate.submit(ReplayTurn::new("R", "A", 0)).unwrap(); // ready but paused
        gate.submit(ReplayTurn::new("R", "B", 0)).unwrap();
        gate.complete("R", key("A", 0));
        assert!(gate.released().is_empty());
        assert_eq!(
            gate.pending_turns_by_root().get("R").unwrap(),
            &vec![ReplayTurn::new("R", "A", 0), ReplayTurn::new("R", "B", 0)]
        );
    }

    #[test]
    fn duplicate_pending_and_negative_boundary_error() {
        let mut gate = ReplayGate::new(preds());
        gate.activate();
        gate.submit(ReplayTurn::new("R", "B", 0)).unwrap();
        assert!(matches!(
            gate.submit(ReplayTurn::new("R", "B", 0)),
            Err(ReplayGateError::DuplicatePending { .. })
        ));

        assert!(matches!(
            gate.seed_completed_prefixes(
                "R3",
                &[ReplayResumeBoundary {
                    conversation_id: "X".into(),
                    next_turn_index: -1,
                }],
            ),
            Err(ReplayGateError::NegativeBoundaryIndex { .. })
        ));
    }

    #[test]
    fn seed_after_dispatch_errors_and_before_active_issues_eagerly() {
        let mut gate = ReplayGate::new(preds());
        // Before activation, submit issues immediately regardless of predecessors.
        gate.submit(ReplayTurn::new("R", "C", 0)).unwrap();
        assert_eq!(gate.released(), &[key("C", 0)]);

        gate.activate();
        gate.submit(ReplayTurn::new("R", "B", 0)).unwrap(); // retained
        assert!(matches!(
            gate.seed_completed_prefixes("R", &[]),
            Err(ReplayGateError::SeedAfterDispatch { .. })
        ));
    }
}
