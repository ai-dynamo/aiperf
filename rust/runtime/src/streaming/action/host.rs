// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Private action-host ownership of the checked action proofs.
//!
//! This module is the sole mint for the checked failed-attempt evidence,
//! checked terminal membership, and frozen action inventory that the
//! reliability owner consumes as borrowed `&dyn` views. `super` declares it as
//! `mod host;`, so the module path is private to the action subtree: no sibling
//! of `super`, and no other module in this crate, can name these types or reach
//! their production mints. Later P2/P4 action-host implementations are added as
//! descendants of this module — `streaming/action/host/<name>.rs` — and inherit
//! that mint authority; a host authored as a sibling of `host` would not, which
//! is the intended constraint.

use std::collections::BTreeMap;

mod inventory;
mod multiplexed;

pub use inventory::ActionInventoryLedger;
pub use multiplexed::{
    ActionEventBatch, ActiveExecution, ActiveExecutionSet, BudgetOwnedActionTerminalReceipt,
    StreamingActionBindingSet, StreamingActionHost, action_kind, canonical_action_schema,
};

use super::{
    ActionTerminalMembershipOutcomeView, CheckedActionFailureTerminalEvidenceView,
    CheckedActionTerminalMembershipView, FrozenActionInventoryView, reliability_view_seal,
};
use crate::streaming::{
    checkpoint::StreamRunIdentity,
    identity::{ContentDigest, GlobalSequence, StableActionId},
};

/// Action-host-owned sealed evidence that one failed attempt reached terminal.
///
/// The reliability owner receives this only as
/// `&dyn CheckedActionFailureTerminalEvidenceView`.
// The production mint lands with the P2/P4 action hosts; until then the type is
// exercised only by the crate-private test fixture.
#[allow(dead_code)]
pub struct CheckedActionFailureTerminalEvidence {
    run: StreamRunIdentity,
    action_id: StableActionId,
    sequence: GlobalSequence,
    terminal_evidence_digest: ContentDigest,
}

#[allow(dead_code)]
impl CheckedActionFailureTerminalEvidence {
    /// Mint checked terminal evidence from action-host-owned state.
    ///
    /// Deliberately declared without a visibility modifier: production mint
    /// authority belongs to this host subtree and to nothing else in the crate.
    const fn new(
        run: StreamRunIdentity,
        action_id: StableActionId,
        sequence: GlobalSequence,
        terminal_evidence_digest: ContentDigest,
    ) -> Self {
        Self {
            run,
            action_id,
            sequence,
            terminal_evidence_digest,
        }
    }
}

#[cfg(test)]
impl CheckedActionFailureTerminalEvidence {
    /// Mint checked terminal evidence for in-crate reliability unit tests.
    ///
    /// This fixture does not exist in any production build.
    pub(crate) const fn for_test(
        run: StreamRunIdentity,
        action_id: StableActionId,
        sequence: GlobalSequence,
        terminal_evidence_digest: ContentDigest,
    ) -> Self {
        Self::new(run, action_id, sequence, terminal_evidence_digest)
    }
}

impl reliability_view_seal::CheckedActionFailureTerminalEvidenceView
    for CheckedActionFailureTerminalEvidence
{
}

impl CheckedActionFailureTerminalEvidenceView for CheckedActionFailureTerminalEvidence {
    fn run(&self) -> &StreamRunIdentity {
        &self.run
    }

    fn action_id(&self) -> StableActionId {
        self.action_id
    }

    fn sequence(&self) -> GlobalSequence {
        self.sequence
    }

    fn terminal_evidence_digest(&self) -> ContentDigest {
        self.terminal_evidence_digest
    }
}

/// Action-host-owned sealed terminal membership for one finalized action.
#[allow(dead_code)]
pub struct CheckedActionTerminalMembership {
    run: StreamRunIdentity,
    action_id: StableActionId,
    sequence: GlobalSequence,
    outcome: ActionTerminalMembershipOutcomeView,
    membership_digest: ContentDigest,
}

#[allow(dead_code)]
impl CheckedActionTerminalMembership {
    /// Mint checked terminal membership from action-host-owned state.
    ///
    /// Host-subtree-private for the same reason as
    /// [`CheckedActionFailureTerminalEvidence::new`].
    const fn new(
        run: StreamRunIdentity,
        action_id: StableActionId,
        sequence: GlobalSequence,
        outcome: ActionTerminalMembershipOutcomeView,
        membership_digest: ContentDigest,
    ) -> Self {
        Self {
            run,
            action_id,
            sequence,
            outcome,
            membership_digest,
        }
    }
}

#[cfg(test)]
impl CheckedActionTerminalMembership {
    /// Mint checked terminal membership for in-crate reliability unit tests.
    pub(crate) const fn for_test(
        run: StreamRunIdentity,
        action_id: StableActionId,
        sequence: GlobalSequence,
        outcome: ActionTerminalMembershipOutcomeView,
        membership_digest: ContentDigest,
    ) -> Self {
        Self::new(run, action_id, sequence, outcome, membership_digest)
    }
}

impl reliability_view_seal::CheckedActionTerminalMembershipView
    for CheckedActionTerminalMembership
{
}

impl CheckedActionTerminalMembershipView for CheckedActionTerminalMembership {
    fn run(&self) -> &StreamRunIdentity {
        &self.run
    }

    fn action_id(&self) -> StableActionId {
        self.action_id
    }

    fn sequence(&self) -> GlobalSequence {
        self.sequence
    }

    fn outcome(&self) -> ActionTerminalMembershipOutcomeView {
        self.outcome
    }

    fn membership_digest(&self) -> ContentDigest {
        self.membership_digest
    }
}

/// Action-host-owned immutable inventory proving dense action gap closure.
#[allow(dead_code)]
pub struct FrozenActionInventory {
    run: StreamRunIdentity,
    through: GlobalSequence,
    membership_root: ContentDigest,
    terminals: BTreeMap<GlobalSequence, ContentDigest>,
}

#[allow(dead_code)]
impl FrozenActionInventory {
    /// Freeze the host's terminal membership map for gap-closure proof.
    ///
    /// Host-subtree-private for the same reason as
    /// [`CheckedActionFailureTerminalEvidence::new`].
    fn new(
        run: StreamRunIdentity,
        through: GlobalSequence,
        membership_root: ContentDigest,
        terminals: BTreeMap<GlobalSequence, ContentDigest>,
    ) -> Self {
        Self {
            run,
            through,
            membership_root,
            terminals,
        }
    }
}

#[cfg(test)]
impl FrozenActionInventory {
    /// Freeze a terminal membership map for in-crate reliability unit tests.
    pub(crate) fn for_test(
        run: StreamRunIdentity,
        through: GlobalSequence,
        membership_root: ContentDigest,
        terminals: BTreeMap<GlobalSequence, ContentDigest>,
    ) -> Self {
        Self::new(run, through, membership_root, terminals)
    }
}

impl reliability_view_seal::FrozenActionInventoryView for FrozenActionInventory {}

impl FrozenActionInventoryView for FrozenActionInventory {
    fn run(&self) -> &StreamRunIdentity {
        &self.run
    }

    fn through(&self) -> GlobalSequence {
        self.through
    }

    fn membership_root(&self) -> ContentDigest {
        self.membership_root
    }

    fn contains_terminal(
        &self,
        sequence: GlobalSequence,
        membership_digest: ContentDigest,
    ) -> bool {
        self.terminals.get(&sequence) == Some(&membership_digest)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::streaming::identity::LogicalReplayRunId;

    const PARENT_SOURCE: &str = include_str!("../action.rs");
    const HOST_SOURCE: &str = include_str!("host.rs");

    /// Return `host.rs` with its own test module removed.
    ///
    /// Without this the assertion needles below would match themselves.
    fn host_production_source() -> &'static str {
        HOST_SOURCE
            .split_once("\n#[cfg(test)]\nmod tests {")
            .map_or(HOST_SOURCE, |(head, _)| head)
    }

    #[test]
    fn action_host_module_is_not_reachable_from_siblings() {
        assert!(
            PARENT_SOURCE.contains("\nmod host;\n"),
            "action.rs must declare the action host as a private child module"
        );
        for widened in ["pub mod host", "pub(crate) mod host", "pub(super) mod host"] {
            assert!(
                !PARENT_SOURCE.contains(widened),
                "the action host module must stay private: {widened}"
            );
        }
    }

    #[test]
    fn action_proof_reexport_is_test_only() {
        let before = PARENT_SOURCE
            .split_once("pub(crate) use host::{")
            .map(|(head, _)| head)
            .unwrap_or_else(|| panic!("action.rs must re-export host proofs for in-crate tests"));
        assert!(
            before.trim_end().ends_with("#[cfg(test)]"),
            "the host proof re-export must be gated on cfg(test)"
        );
    }

    #[test]
    fn action_proof_mints_are_host_subtree_private() {
        let source = host_production_source();
        assert_eq!(
            source.matches("    const fn new(").count(),
            2,
            "both const proof mints must carry no visibility modifier"
        );
        assert_eq!(
            source.matches("    fn new(").count(),
            1,
            "the inventory mint must carry no visibility modifier"
        );
        for widened in [
            "pub const fn new(",
            "pub(crate) const fn new(",
            "pub(super) const fn new(",
            "pub fn new(",
            "pub(crate) fn new(",
            "pub(super) fn new(",
        ] {
            assert!(
                !source.contains(widened),
                "action proof mint visibility was widened: {widened}"
            );
        }
    }

    #[test]
    fn host_minted_proofs_are_the_only_construction_path() {
        let run = StreamRunIdentity::new(LogicalReplayRunId::from_bytes([0x31; 32]));
        let action_id = StableActionId::from_bytes([0x32; 32]);
        let digest = ContentDigest::from_bytes([0x33; 32]);

        let evidence = CheckedActionFailureTerminalEvidence::for_test(
            run,
            action_id,
            GlobalSequence::new(7),
            digest,
        );
        assert_eq!(evidence.run(), &run);
        assert_eq!(evidence.action_id(), action_id);
        assert_eq!(evidence.sequence(), GlobalSequence::new(7));
        assert_eq!(evidence.terminal_evidence_digest(), digest);

        let membership = CheckedActionTerminalMembership::for_test(
            run,
            action_id,
            GlobalSequence::new(7),
            ActionTerminalMembershipOutcomeView::Failed { issue_id: digest },
            digest,
        );
        assert_eq!(
            membership.outcome(),
            ActionTerminalMembershipOutcomeView::Failed { issue_id: digest }
        );
        assert_eq!(membership.membership_digest(), digest);

        let mut terminals = BTreeMap::new();
        terminals.insert(GlobalSequence::new(7), digest);
        let inventory =
            FrozenActionInventory::for_test(run, GlobalSequence::new(7), digest, terminals);
        assert!(inventory.contains_terminal(GlobalSequence::new(7), digest));
        assert!(!inventory.contains_terminal(GlobalSequence::new(8), digest));
        assert_eq!(inventory.through(), GlobalSequence::new(7));
        assert_eq!(inventory.membership_root(), digest);
    }
}
