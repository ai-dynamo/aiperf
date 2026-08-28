// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Delivery restart and target-idempotency policy for the result plane.
//!
//! A restart cannot invent facts. The only delivery-relevant thing a new
//! incarnation knows is the committed checkpoint cut it resumes from and the
//! idempotency capability the endpoint target proved at configuration time.
//! Those two inputs decide three things: which logical actions the restart
//! re-emits, what end-to-end delivery claim the run may honestly publish, and
//! which of duplication or loss the restart leaves possible.
//!
//! The policy is deliberately asymmetric between [`CheckpointDeliveryMode::Admitted`]
//! and the weaker modes. Durably knowing that an action was *admitted*, without a
//! durable terminal fact, is exactly the knowledge that lets a restart suppress
//! it; a weaker cut that does not record admission has no suppression handle and
//! must re-emit the whole derived suffix. Suppression trades duplication for
//! loss, so an admitted-cut run without target idempotency is at-most-once while
//! a decoded-cut run without it is at-least-once.

use std::{collections::BTreeSet, fmt};

use serde::{Deserialize, Serialize};

use crate::streaming::{
    checkpoint::CheckpointCut,
    identity::{ContentDigest, GlobalSequence, StableActionId},
};

use super::ResultProjectionId;

/// Strongest delivery-relevant fact a committed checkpoint cut records.
///
/// The variants are ordered from strongest to weakest. Each names the last
/// stage of the action pipeline whose progress survives process replacement.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CheckpointDeliveryMode {
    /// Contiguous terminal action facts are durable.
    Terminal,
    /// Action admission is durable; terminal outcome is not.
    Admitted,
    /// Decoder progress is durable; action admission is not.
    Decoded,
    /// Only immutable source acquisition is durable.
    Acquired,
    /// Nothing delivery-relevant is durable.
    None,
}

impl CheckpointDeliveryMode {
    /// Every delivery mode, strongest first.
    pub const ALL: [Self; 5] = [
        Self::Terminal,
        Self::Admitted,
        Self::Decoded,
        Self::Acquired,
        Self::None,
    ];

    /// Stable lowercase tag used in diagnostics and canonical encodings.
    #[must_use]
    pub const fn tag(self) -> &'static str {
        match self {
            Self::Terminal => "terminal",
            Self::Admitted => "admitted",
            Self::Decoded => "decoded",
            Self::Acquired => "acquired",
            Self::None => "none",
        }
    }

    /// Whether a restart from this cut re-emits actions whose target effect is
    /// uncertain.
    ///
    /// `Admitted` is the only mode whose answer depends on the target: it is the
    /// only cut that can both identify an uncertain action and prove nothing
    /// about its outcome, so suppression is available and is the safe default
    /// unless the target deduplicates.
    #[must_use]
    pub const fn reissues_uncertain_actions(self, capability: TargetIdempotencyCapability) -> bool {
        match self {
            Self::Terminal | Self::Decoded | Self::Acquired => true,
            Self::Admitted => capability.deduplicates_by_action_key(),
            Self::None => false,
        }
    }

    /// Whether a cut in this mode carries an authoritative committed result root.
    #[must_use]
    pub const fn has_authoritative_results(self) -> bool {
        !matches!(self, Self::None)
    }
}

/// Proof the endpoint target deduplicates a re-submitted logical action.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TargetIdempotencyCapability {
    /// No proof; a re-submission duplicates the target effect.
    #[default]
    Unsupported,
    /// The target collapses re-submissions carrying the exact logical action key.
    VerifiedLogicalActionKey,
}

impl TargetIdempotencyCapability {
    /// Every target idempotency capability.
    pub const ALL: [Self; 2] = [Self::Unsupported, Self::VerifiedLogicalActionKey];

    /// Stable lowercase tag used in diagnostics and canonical encodings.
    #[must_use]
    pub const fn tag(self) -> &'static str {
        match self {
            Self::Unsupported => "unsupported",
            Self::VerifiedLogicalActionKey => "verified_logical_action_key",
        }
    }

    /// Whether re-submitting the exact logical action key is effect-free.
    #[must_use]
    pub const fn deduplicates_by_action_key(self) -> bool {
        matches!(self, Self::VerifiedLogicalActionKey)
    }
}

/// Point in the commit/dispatch cycle at which an incarnation died.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DeliveryCrashPoint {
    /// After the cut committed, before any post-cut action was dispatched.
    BeforeDispatch,
    /// After dispatch, before any terminal fact was observed.
    AfterDispatchBeforeTerminal,
    /// After a terminal fact was observed, before it entered a committed cut.
    AfterTerminalBeforeCommit,
    /// After the following cut committed, leaving nothing outstanding.
    AfterCommit,
}

impl DeliveryCrashPoint {
    /// Every modelled crash point, in cycle order.
    pub const ALL: [Self; 4] = [
        Self::BeforeDispatch,
        Self::AfterDispatchBeforeTerminal,
        Self::AfterTerminalBeforeCommit,
        Self::AfterCommit,
    ];

    /// Stable lowercase tag used in diagnostics.
    #[must_use]
    pub const fn tag(self) -> &'static str {
        match self {
            Self::BeforeDispatch => "before_dispatch",
            Self::AfterDispatchBeforeTerminal => "after_dispatch_before_terminal",
            Self::AfterTerminalBeforeCommit => "after_terminal_before_commit",
            Self::AfterCommit => "after_commit",
        }
    }
}

/// End-to-end delivery claim a run may honestly publish.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DeliveryClaim {
    /// Every derived action reaches the target; a restart may duplicate one.
    AtLeastOnce,
    /// No action reaches the target twice; a restart may drop an uncertain one.
    AtMostOnce,
    /// Re-submission is at-least-once on the wire and collapses at the target.
    IdempotentAtLeastOnceSubmission,
    /// Only ingestion of the acquired source is claimed; action delivery is not.
    IngestionOnly,
    /// Nothing durable supports any claim.
    None,
}

impl DeliveryClaim {
    /// Stable lowercase tag used in diagnostics and canonical encodings.
    #[must_use]
    pub const fn tag(self) -> &'static str {
        match self {
            Self::AtLeastOnce => "at_least_once",
            Self::AtMostOnce => "at_most_once",
            Self::IdempotentAtLeastOnceSubmission => "idempotent_at_least_once_submission",
            Self::IngestionOnly => "ingestion_only",
            Self::None => "none",
        }
    }

    /// Derive the claim supported by one delivery mode and target capability.
    #[must_use]
    pub const fn derive(
        mode: CheckpointDeliveryMode,
        capability: TargetIdempotencyCapability,
    ) -> Self {
        match mode {
            // Nothing durable identifies what was already delivered.
            CheckpointDeliveryMode::None => Self::None,
            // An acquisition-only cut records that source bytes were ingested and
            // nothing about action derivation, so no action-level claim exists
            // even though the replayed suffix is re-emitted.
            CheckpointDeliveryMode::Acquired => Self::IngestionOnly,
            CheckpointDeliveryMode::Terminal | CheckpointDeliveryMode::Decoded => {
                if capability.deduplicates_by_action_key() {
                    Self::IdempotentAtLeastOnceSubmission
                } else {
                    Self::AtLeastOnce
                }
            }
            CheckpointDeliveryMode::Admitted => {
                if capability.deduplicates_by_action_key() {
                    Self::IdempotentAtLeastOnceSubmission
                } else {
                    Self::AtMostOnce
                }
            }
        }
    }
}

/// Which of duplication or loss one restart leaves possible at the target.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DuplicateWindow {
    /// The restart may apply an already-applied effect a second time.
    pub may_duplicate_target_effect: bool,
    /// The restart may drop an effect that never reached the target.
    pub may_lose_target_effect: bool,
}

impl DuplicateWindow {
    /// A restart with nothing outstanding.
    pub const CLOSED: Self = Self {
        may_duplicate_target_effect: false,
        may_lose_target_effect: false,
    };

    /// Whether this restart leaves the target's effect set exactly reproduced.
    #[must_use]
    pub const fn is_closed(self) -> bool {
        !self.may_duplicate_target_effect && !self.may_lose_target_effect
    }
}

/// State of one action left outstanding by a dead incarnation.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OutstandingActionState {
    /// Derived from the durable cut but never dispatched.
    NotDispatched,
    /// Dispatched with no committed terminal fact; the target effect is unknown.
    AdmittedNotTerminal,
    /// A terminal fact was observed but never committed; the effect happened.
    TerminalUncommitted,
}

impl OutstandingActionState {
    /// Whether the target effect of this action is unknown to the restart.
    ///
    /// `TerminalUncommitted` is uncertain from the restart's point of view: the
    /// observation died with the incarnation that made it.
    #[must_use]
    pub const fn is_uncertain(self) -> bool {
        matches!(self, Self::AdmittedNotTerminal | Self::TerminalUncommitted)
    }
}

/// One logical action left outstanding across a restart.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct OutstandingAction {
    /// Incarnation-free identity of the logical action.
    pub action: StableActionId,
    /// Dense position in global replay order.
    pub sequence: GlobalSequence,
    /// What the dead incarnation had done with it.
    pub state: OutstandingActionState,
}

/// Identity a restart must reproduce exactly before it may re-emit anything.
///
/// A restart that changes topology, result projection, or membership scheme is
/// a different run: its action identities and membership keys are not comparable
/// with the committed ones, so resuming would silently mix two logical runs.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DeliveryTopologyBinding {
    /// Digest of the frozen execution topology.
    pub topology_digest: ContentDigest,
    /// Result projection the committed cut was produced under.
    pub projection: ResultProjectionId,
    /// Digest of the membership scheme deriving canonical membership roots.
    pub membership_scheme_digest: ContentDigest,
}

/// Everything a restart knows about the run it is resuming.
#[derive(Clone, Debug)]
pub struct DeliveryRestartRequest<'a> {
    /// Delivery mode of the committed cut being resumed from.
    pub mode: CheckpointDeliveryMode,
    /// Idempotency capability proved by the endpoint target.
    pub capability: TargetIdempotencyCapability,
    /// The committed cut, when the mode carries one.
    pub cut: Option<&'a CheckpointCut>,
    /// Committed result index root, when the mode carries one.
    pub result_index_root: Option<ContentDigest>,
    /// Binding recorded alongside the committed cut.
    pub committed_binding: &'a DeliveryTopologyBinding,
    /// Binding declared by the restarting incarnation.
    pub restarting_binding: &'a DeliveryTopologyBinding,
    /// Actions the dead incarnation left outstanding.
    pub outstanding: &'a [OutstandingAction],
}

/// Restart decision derived from one committed cut and target capability.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DeliveryRestartDecision {
    /// Logical actions the restart re-emits, in global replay order.
    pub reissue: Vec<StableActionId>,
    /// Committed result root that remains authoritative across the restart.
    pub authoritative_results: Option<ContentDigest>,
    /// End-to-end claim the resumed run may publish.
    pub claim: DeliveryClaim,
    /// What this restart leaves possible at the target.
    pub duplicate_window: DuplicateWindow,
}

impl DeliveryRestartDecision {
    /// Whether every re-emitted action appears exactly once.
    ///
    /// Logical membership is keyed by incarnation-free action identity, so a
    /// repeated entry would submit one logical action twice from a single
    /// restart rather than once per incarnation.
    #[must_use]
    pub fn logical_membership_is_unique(&self) -> bool {
        let mut seen = BTreeSet::new();
        self.reissue.iter().all(|action| seen.insert(*action))
    }
}

/// Refusal raised before a restart may re-emit anything.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DeliveryRestartError {
    /// The restarting incarnation declared a different execution topology.
    TopologyChanged,
    /// The restarting incarnation declared a different result projection.
    ProjectionChanged,
    /// The restarting incarnation declared a different membership scheme.
    MembershipSchemeChanged,
    /// A mode carrying a committed cut was resumed without one.
    MissingCommittedCut,
    /// A mode carrying no committed cut was resumed with one.
    UnexpectedCommittedCut,
    /// An outstanding action sat at or below the committed terminal horizon.
    OutstandingBelowTerminalHorizon {
        /// Sequence of the offending outstanding action.
        sequence: GlobalSequence,
    },
    /// The same logical action was reported outstanding more than once.
    DuplicateOutstandingAction,
}

impl DeliveryRestartError {
    /// Return the stable machine-readable error code.
    #[must_use]
    pub const fn code(&self) -> &'static str {
        match self {
            Self::TopologyChanged => "topology_changed",
            Self::ProjectionChanged => "projection_changed",
            Self::MembershipSchemeChanged => "membership_scheme_changed",
            Self::MissingCommittedCut => "missing_committed_cut",
            Self::UnexpectedCommittedCut => "unexpected_committed_cut",
            Self::OutstandingBelowTerminalHorizon { .. } => "outstanding_below_terminal_horizon",
            Self::DuplicateOutstandingAction => "duplicate_outstanding_action",
        }
    }
}

impl fmt::Display for DeliveryRestartError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OutstandingBelowTerminalHorizon { sequence } => write!(
                formatter,
                "{}: sequence {} is already terminal in the committed cut",
                self.code(),
                sequence.get()
            ),
            _ => write!(formatter, "{}", self.code()),
        }
    }
}

impl std::error::Error for DeliveryRestartError {}

/// Derive the delivery restart decision for one resumed run.
///
/// The restart is refused before any action is re-emitted when the resumed
/// identity does not reproduce the committed one, when the mode and the supplied
/// cut disagree, or when the outstanding set contradicts the committed cut.
pub fn deliver_restart_decision(
    request: &DeliveryRestartRequest<'_>,
) -> Result<DeliveryRestartDecision, DeliveryRestartError> {
    verify_binding(request.committed_binding, request.restarting_binding)?;

    let cut = match (request.mode.has_authoritative_results(), request.cut) {
        (true, Some(cut)) => Some(cut),
        (true, None) => return Err(DeliveryRestartError::MissingCommittedCut),
        (false, Some(_)) => return Err(DeliveryRestartError::UnexpectedCommittedCut),
        (false, None) => None,
    };

    let mut seen = BTreeSet::new();
    for outstanding in request.outstanding {
        if !seen.insert(outstanding.action) {
            return Err(DeliveryRestartError::DuplicateOutstandingAction);
        }
        // A committed terminal horizon is contiguous, so anything at or below it
        // already has an authoritative terminal fact and cannot be outstanding.
        if let Some(cut) = cut
            && request.mode == CheckpointDeliveryMode::Terminal
            && outstanding.sequence <= *cut.terminal.get()
        {
            return Err(DeliveryRestartError::OutstandingBelowTerminalHorizon {
                sequence: outstanding.sequence,
            });
        }
    }

    let reissues_uncertain = request.mode.reissues_uncertain_actions(request.capability);
    let mut selected: Vec<&OutstandingAction> = request
        .outstanding
        .iter()
        .filter(|outstanding| {
            // A mode with no durable cut has no authoritative record naming what
            // to re-emit, so it re-emits nothing at all.
            request.mode.has_authoritative_results()
                && (!outstanding.state.is_uncertain() || reissues_uncertain)
        })
        .collect();
    selected.sort_by_key(|outstanding| outstanding.sequence);

    let has_uncertain = request
        .outstanding
        .iter()
        .any(|outstanding| outstanding.state.is_uncertain());
    let reissues_any_uncertain =
        has_uncertain && reissues_uncertain && request.mode.has_authoritative_results();

    let duplicate_window = DuplicateWindow {
        may_duplicate_target_effect: reissues_any_uncertain
            && !request.capability.deduplicates_by_action_key(),
        may_lose_target_effect: has_uncertain && !reissues_any_uncertain,
    };

    Ok(DeliveryRestartDecision {
        reissue: selected
            .into_iter()
            .map(|outstanding| outstanding.action)
            .collect(),
        authoritative_results: request
            .mode
            .has_authoritative_results()
            .then_some(request.result_index_root)
            .flatten(),
        claim: DeliveryClaim::derive(request.mode, request.capability),
        duplicate_window,
    })
}

fn verify_binding(
    committed: &DeliveryTopologyBinding,
    restarting: &DeliveryTopologyBinding,
) -> Result<(), DeliveryRestartError> {
    if committed.topology_digest != restarting.topology_digest {
        return Err(DeliveryRestartError::TopologyChanged);
    }
    if committed.projection != restarting.projection {
        return Err(DeliveryRestartError::ProjectionChanged);
    }
    if committed.membership_scheme_digest != restarting.membership_scheme_digest {
        return Err(DeliveryRestartError::MembershipSchemeChanged);
    }
    Ok(())
}
