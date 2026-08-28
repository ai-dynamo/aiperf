// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Session closure proofs and bounded causality policies.
//!
//! Partition exhaustion is a decoder event, never closure evidence. A session
//! closes only against one of five checked proofs: a producer-authored close, a
//! hard event-time watermark, a verified finite seal, a verified complete
//! externally sorted run, or an exhausted authored missing-predecessor policy.
//! Every other observation either waits or fails; nothing invents a
//! disposition.
//!
//! Retained causal state must be provably bounded. An authored configuration
//! that can grow a session's pending set without a finite bound and without a
//! spill, drop, or fail disposition is refused before the first fragment.

use serde::{Deserialize, Serialize};

use crate::streaming::{
    failure::{SessionCoordinatorError, SessionFailureCode},
    identity::{ContentDigest, StableSessionKey},
    unit::{EventTimeUtc, StateBudgetFailureCode},
};

/// Checked proof that permitted a session to be closed or quarantined.
///
/// Constructed only by [`SessionClosurePolicy::decide`]; partition exhaustion
/// has no variant here by construction.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SessionQuarantineClosureProof {
    /// The producer authored an explicit terminal record for this session.
    AuthoredClose,
    /// A hard session event-time watermark passed the session's last event.
    HardWatermark,
    /// A finite source seal covered the session with no causal gap.
    VerifiedFiniteSeal,
    /// A verified complete externally sorted run covered the session.
    VerifiedCompleteSortedRun,
    /// The authored missing-predecessor policy ran out of admissible options.
    ExhaustedMissingPredecessorPolicy,
}

impl SessionQuarantineClosureProof {
    /// Return the stable machine-readable proof tag.
    #[must_use]
    pub const fn tag(self) -> &'static str {
        match self {
            Self::AuthoredClose => "authored_close",
            Self::HardWatermark => "hard_watermark",
            Self::VerifiedFiniteSeal => "verified_finite_seal",
            Self::VerifiedCompleteSortedRun => "verified_complete_sorted_run",
            Self::ExhaustedMissingPredecessorPolicy => "exhausted_missing_predecessor_policy",
        }
    }

    /// Return the canonical single-byte encoding used by tombstone digests.
    #[must_use]
    pub const fn canonical_tag(self) -> u8 {
        match self {
            Self::AuthoredClose => 0,
            Self::HardWatermark => 1,
            Self::VerifiedFiniteSeal => 2,
            Self::VerifiedCompleteSortedRun => 3,
            Self::ExhaustedMissingPredecessorPolicy => 4,
        }
    }
}

/// Observation offered to the closure policy for one session.
///
/// [`SessionClosureEvidence::PartitionEof`] exists so the policy can state
/// explicitly that partition exhaustion resolves to a wait, not so it can close.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SessionClosureEvidence {
    /// A producer-authored terminal record was incorporated.
    AuthoredClose,
    /// A soft event-time watermark passed the session's inactivity deadline.
    SoftWatermarkBelowDeadline {
        /// Soft watermark proven complete by the format.
        watermark: EventTimeUtc,
        /// Greatest event time observed for this session.
        session_event_time: EventTimeUtc,
    },
    /// A hard session watermark passed the session's greatest event time.
    HardWatermarkPastSession {
        /// Hard watermark proven complete by the format.
        watermark: EventTimeUtc,
        /// Greatest event time observed for this session.
        session_event_time: EventTimeUtc,
    },
    /// A finite source seal covered this session.
    FiniteSeal {
        /// Whether the session still holds an unresolved causal gap.
        has_causal_gap: bool,
    },
    /// An externally sorted run covered this session.
    CompleteSortedRun {
        /// Whether the sorted run was verified complete for this session.
        is_verified_complete: bool,
    },
    /// A partition ended while the session still declares a predecessor.
    PartitionEof {
        /// Whether a declared predecessor is still unresolved.
        has_missing_predecessor: bool,
    },
}

/// Disposition selected by the closure policy for one session.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SessionClosureDecision {
    /// Close the session under the returned checked proof.
    Close(SessionQuarantineClosureProof),
    /// Retain the session and keep waiting for more evidence.
    Wait,
    /// The evidence proves the session can never be completed.
    Fail(SessionFailureCode),
}

/// Disposition for an action whose declared predecessors never arrive.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MissingPredecessorPolicy {
    /// Hold the successor indefinitely; only valid with a finite state bound.
    #[default]
    Wait,
    /// Move held successors to a private bounded spill.
    Spill,
    /// Drop the held successor once the authored deadline is exhausted.
    Drop,
    /// Fail the run once the authored deadline is exhausted.
    Fail,
}

impl MissingPredecessorPolicy {
    /// Whether this policy can retire a held successor without new evidence.
    #[must_use]
    pub const fn can_retire_without_evidence(self) -> bool {
        matches!(self, Self::Spill | Self::Drop | Self::Fail)
    }
}

/// Authored inferred-closure policy for one session program.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SessionClosurePolicy {
    /// Event-time inactivity span that closes a session below the soft
    /// watermark, absent when inactivity closure is not authored.
    #[serde(default)]
    pub inactivity_deadline_ns: Option<i64>,
    /// Whether a hard session watermark may close a session.
    #[serde(default = "default_true")]
    pub honors_hard_watermark: bool,
    /// Whether a finite seal must fail a session that still holds a causal gap.
    #[serde(default = "default_true")]
    pub finite_seal_requires_complete: bool,
    /// Whether a verified complete sorted run may close a session.
    #[serde(default = "default_true")]
    pub accepts_complete_sorted_run: bool,
    /// Disposition for successors whose declared predecessors never arrive.
    #[serde(default)]
    pub missing_predecessor: MissingPredecessorPolicy,
}

const fn default_true() -> bool {
    true
}

impl Default for SessionClosurePolicy {
    fn default() -> Self {
        Self {
            inactivity_deadline_ns: None,
            honors_hard_watermark: true,
            finite_seal_requires_complete: true,
            accepts_complete_sorted_run: true,
            missing_predecessor: MissingPredecessorPolicy::Wait,
        }
    }
}

impl SessionClosurePolicy {
    /// Decide one session's disposition from exactly one observation.
    #[must_use]
    pub fn decide(&self, evidence: SessionClosureEvidence) -> SessionClosureDecision {
        match evidence {
            SessionClosureEvidence::AuthoredClose => {
                SessionClosureDecision::Close(SessionQuarantineClosureProof::AuthoredClose)
            }
            SessionClosureEvidence::SoftWatermarkBelowDeadline {
                watermark,
                session_event_time,
            } => self.decide_inactivity(watermark, session_event_time),
            SessionClosureEvidence::HardWatermarkPastSession {
                watermark,
                session_event_time,
            } => {
                if self.honors_hard_watermark && watermark.get() > session_event_time.get() {
                    SessionClosureDecision::Close(SessionQuarantineClosureProof::HardWatermark)
                } else {
                    SessionClosureDecision::Wait
                }
            }
            SessionClosureEvidence::FiniteSeal { has_causal_gap } => {
                if has_causal_gap && self.finite_seal_requires_complete {
                    // A sealed source can never resolve the gap, so waiting is
                    // an unbounded lie: this is a terminal failure.
                    SessionClosureDecision::Fail(SessionFailureCode::MissingPredecessor)
                } else {
                    SessionClosureDecision::Close(SessionQuarantineClosureProof::VerifiedFiniteSeal)
                }
            }
            SessionClosureEvidence::CompleteSortedRun {
                is_verified_complete,
            } => {
                if is_verified_complete && self.accepts_complete_sorted_run {
                    SessionClosureDecision::Close(
                        SessionQuarantineClosureProof::VerifiedCompleteSortedRun,
                    )
                } else {
                    SessionClosureDecision::Wait
                }
            }
            // Partition exhaustion is a decoder event. Indefinite follow waits.
            SessionClosureEvidence::PartitionEof { .. } => SessionClosureDecision::Wait,
        }
    }

    fn decide_inactivity(
        &self,
        watermark: EventTimeUtc,
        session_event_time: EventTimeUtc,
    ) -> SessionClosureDecision {
        let Some(deadline) = self.inactivity_deadline_ns else {
            return SessionClosureDecision::Wait;
        };
        let Some(closes_at) = session_event_time.get().checked_add(deadline) else {
            return SessionClosureDecision::Wait;
        };
        if watermark.get() >= closes_at {
            // Inactivity closure is lossy and is only reachable because the
            // authored policy names the deadline that just ran out.
            SessionClosureDecision::Close(
                SessionQuarantineClosureProof::ExhaustedMissingPredecessorPolicy,
            )
        } else {
            SessionClosureDecision::Wait
        }
    }
}

/// Authored bound on one session program's retained causal state.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SessionCausalityLimits {
    /// Maximum simultaneously live sessions, absent when unbounded.
    pub max_active_sessions: Option<usize>,
    /// Maximum held out-of-order successors per session, absent when unbounded.
    pub max_pending_per_session: Option<usize>,
    /// Maximum retained bytes per session, absent when unbounded.
    pub max_retained_bytes_per_session: Option<usize>,
    /// Disposition for successors whose predecessors never arrive.
    pub missing_predecessor: MissingPredecessorPolicy,
}

/// Refuse an authored configuration whose causal state cannot be bounded.
///
/// A finite bound on every retained dimension is sufficient. Without one, the
/// authored missing-predecessor policy must be able to retire held state on its
/// own; an indefinite `Wait` over an unbounded set is exactly the unbounded
/// causality state this refuses.
pub fn validate_session_limits(
    limits: SessionCausalityLimits,
) -> Result<(), SessionCoordinatorError> {
    let is_finite = |bound: Option<usize>| matches!(bound, Some(value) if value > 0);
    let is_fully_bounded = is_finite(limits.max_active_sessions)
        && is_finite(limits.max_pending_per_session)
        && is_finite(limits.max_retained_bytes_per_session);
    if is_fully_bounded || limits.missing_predecessor.can_retire_without_evidence() {
        return Ok(());
    }
    Err(SessionCoordinatorError::session(
        SessionFailureCode::UnboundedCausalityState,
    ))
}

/// Receipt proving one entire rooted producer tree closed.
///
/// Minted only after the exact declared descendant inventory is closed under a
/// hard completeness proof. Root closure, descendant discovery, and partition
/// exhaustion never mint it.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WholeProducerTreeClosureReceipt {
    root: StableSessionKey,
    descendants: Vec<StableSessionKey>,
    completeness_proof: SessionQuarantineClosureProof,
    inventory_digest: ContentDigest,
}

impl WholeProducerTreeClosureReceipt {
    /// Return the root session of the closed producer tree.
    #[must_use]
    pub const fn root(&self) -> StableSessionKey {
        self.root
    }

    /// Borrow the exact closed descendant inventory, in canonical order.
    #[must_use]
    pub fn descendants(&self) -> &[StableSessionKey] {
        &self.descendants
    }

    /// Return the hard completeness proof that permitted the mint.
    #[must_use]
    pub const fn completeness_proof(&self) -> SessionQuarantineClosureProof {
        self.completeness_proof
    }

    /// Return the digest binding the root and its exact descendant inventory.
    #[must_use]
    pub const fn inventory_digest(&self) -> ContentDigest {
        self.inventory_digest
    }
}

/// Bounded tracker for one rooted producer tree's closure state.
#[derive(Debug)]
pub struct ProducerTreeClosureTracker {
    root: StableSessionKey,
    is_root_closed: bool,
    declared: Vec<StableSessionKey>,
    closed: Vec<StableSessionKey>,
    completeness_proof: Option<SessionQuarantineClosureProof>,
    max_descendants: usize,
}

impl ProducerTreeClosureTracker {
    /// Begin tracking one rooted producer tree.
    #[must_use]
    pub const fn new(root: StableSessionKey, max_descendants: usize) -> Self {
        Self {
            root,
            is_root_closed: false,
            declared: Vec::new(),
            closed: Vec::new(),
            completeness_proof: None,
            max_descendants,
        }
    }

    /// Declare one descendant that must close before the tree can be complete.
    pub fn declare_descendant(
        &mut self,
        session: StableSessionKey,
    ) -> Result<(), SessionCoordinatorError> {
        if self.declared.contains(&session) {
            return Ok(());
        }
        if self.declared.len() >= self.max_descendants {
            return Err(SessionCoordinatorError::state_budget(
                StateBudgetFailureCode::ItemCapacity,
            ));
        }
        self.declared.push(session);
        self.declared.sort_unstable();
        Ok(())
    }

    /// Record that the root session itself closed.
    pub fn close_root(&mut self) {
        self.is_root_closed = true;
    }

    /// Record that one declared descendant closed.
    pub fn close_descendant(&mut self, session: StableSessionKey) {
        if self.declared.contains(&session) && !self.closed.contains(&session) {
            self.closed.push(session);
            self.closed.sort_unstable();
        }
    }

    /// Record a hard completeness proof covering the whole tree inventory.
    pub fn observe_completeness_proof(&mut self, proof: SessionQuarantineClosureProof) {
        self.completeness_proof = Some(proof);
    }

    /// Record partition exhaustion, which changes no closure state.
    pub const fn observe_partition_eof(&self) {}

    /// Mint the whole-tree receipt when, and only when, every proof is present.
    #[must_use]
    pub fn whole_tree_receipt(&self) -> Option<WholeProducerTreeClosureReceipt> {
        if !self.is_root_closed || self.closed != self.declared {
            return None;
        }
        let completeness_proof = self.completeness_proof?;
        if !matches!(
            completeness_proof,
            SessionQuarantineClosureProof::VerifiedFiniteSeal
                | SessionQuarantineClosureProof::VerifiedCompleteSortedRun
        ) {
            return None;
        }
        Some(WholeProducerTreeClosureReceipt {
            root: self.root,
            descendants: self.closed.clone(),
            completeness_proof,
            inventory_digest: self.inventory_digest(completeness_proof),
        })
    }

    fn inventory_digest(&self, proof: SessionQuarantineClosureProof) -> ContentDigest {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"aiperf.stream.session.producer-tree.v1");
        hasher.update(self.root.as_bytes());
        hasher.update(&[proof.canonical_tag()]);
        hasher.update(&(self.closed.len() as u64).to_le_bytes());
        for descendant in &self.closed {
            hasher.update(descendant.as_bytes());
        }
        ContentDigest::from_bytes(*hasher.finalize().as_bytes())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn event_time(nanoseconds: i64) -> EventTimeUtc {
        EventTimeUtc::new(nanoseconds).expect("test event times are non-negative")
    }

    #[test]
    fn closure_requires_authored_proof_not_partition_eof() {
        let policy = SessionClosurePolicy {
            inactivity_deadline_ns: Some(10),
            ..SessionClosurePolicy::default()
        };
        assert_eq!(
            policy.decide(SessionClosureEvidence::SoftWatermarkBelowDeadline {
                watermark: event_time(100),
                session_event_time: event_time(50),
            }),
            SessionClosureDecision::Close(
                SessionQuarantineClosureProof::ExhaustedMissingPredecessorPolicy
            )
        );
        assert_eq!(
            policy.decide(SessionClosureEvidence::HardWatermarkPastSession {
                watermark: event_time(100),
                session_event_time: event_time(50),
            }),
            SessionClosureDecision::Close(SessionQuarantineClosureProof::HardWatermark)
        );
        assert_eq!(
            policy.decide(SessionClosureEvidence::FiniteSeal {
                has_causal_gap: true
            }),
            SessionClosureDecision::Fail(SessionFailureCode::MissingPredecessor)
        );
        assert_eq!(
            policy.decide(SessionClosureEvidence::CompleteSortedRun {
                is_verified_complete: true
            }),
            SessionClosureDecision::Close(SessionQuarantineClosureProof::VerifiedCompleteSortedRun)
        );
        assert_eq!(
            policy.decide(SessionClosureEvidence::PartitionEof {
                has_missing_predecessor: true
            }),
            SessionClosureDecision::Wait
        );
    }

    #[test]
    fn whole_tree_receipt_requires_every_descendant_not_individual_session_close() {
        let root = StableSessionKey::from_bytes([1; 32]);
        let child = StableSessionKey::from_bytes([2; 32]);
        let mut tree = ProducerTreeClosureTracker::new(root, 8);
        assert!(tree.declare_descendant(child).is_ok());
        tree.close_root();
        assert!(tree.whole_tree_receipt().is_none());
        tree.observe_partition_eof();
        assert!(tree.whole_tree_receipt().is_none());
        tree.close_descendant(child);
        // A closed inventory alone is not a hard completeness proof.
        assert!(tree.whole_tree_receipt().is_none());
        tree.observe_completeness_proof(SessionQuarantineClosureProof::VerifiedFiniteSeal);
        let receipt = tree.whole_tree_receipt();
        assert_eq!(
            receipt.map(|receipt| receipt.descendants().to_vec()),
            Some(vec![child])
        );
    }

    #[test]
    fn unbounded_session_without_spill_drop_or_fail_is_refused() {
        let unbounded = SessionCausalityLimits {
            max_active_sessions: None,
            max_pending_per_session: None,
            max_retained_bytes_per_session: None,
            missing_predecessor: MissingPredecessorPolicy::Wait,
        };
        assert_eq!(
            validate_session_limits(unbounded),
            Err(SessionCoordinatorError::session(
                SessionFailureCode::UnboundedCausalityState
            ))
        );
        assert!(
            validate_session_limits(SessionCausalityLimits {
                missing_predecessor: MissingPredecessorPolicy::Drop,
                ..unbounded
            })
            .is_ok()
        );
    }
}
