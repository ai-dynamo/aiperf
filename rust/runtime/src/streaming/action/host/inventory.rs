// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Accumulating action inventory that mints the sealed frozen gap-closure proof.
//!
//! The reliability owner will accept a `no-more-actions-before` frontier only
//! when it is accompanied by a [`FrozenActionInventory`], and that type's mint
//! is private to the action-host subtree. This module is the production
//! accumulator that fills it: every finalized action records its terminal
//! membership digest at its dense global sequence, and freezing refuses while
//! any sequence at or below the requested frontier is still unterminated.
//!
//! That refusal is the whole point. A gap proof states "no action before this
//! sequence remains outstanding"; if the accumulator could freeze past an
//! unterminated sequence, the reporter would publish a frontier covering work
//! that had not finished. The dense `Vec` makes the check a length comparison
//! plus a scan for `None` rather than a trusted counter.

use std::collections::BTreeMap;

use super::FrozenActionInventory;
use crate::streaming::{
    action::{ActionExecutionError, ActionFailureCode},
    checkpoint::StreamRunIdentity,
    identity::{ContentDigest, GlobalSequence},
};

/// Domain separator binding one frozen inventory's membership root.
const MEMBERSHIP_ROOT_DOMAIN: &[u8] = b"aiperf.stream.action.inventory.membership_root.v1";

/// Dense record of every accepted action sequence and its terminal membership.
///
/// Deliberately not `Clone` and not `Deserialize`: the only way to obtain the
/// sealed view the reporter accepts is to have actually recorded the terminals,
/// so a workload or adapter cannot restore a fabricated inventory from bytes.
#[derive(Debug)]
pub struct ActionInventoryLedger {
    run: StreamRunIdentity,
    /// Dense by construction: index `n` is `GlobalSequence::new(n)`.
    terminals: Vec<Option<ContentDigest>>,
    /// Highest sequence ever accepted, used to distinguish "never seen" from
    /// "seen and unterminated" when a caller freezes past the accepted range.
    accepted_through: Option<GlobalSequence>,
}

impl ActionInventoryLedger {
    /// Open an empty inventory for one logical run.
    #[must_use]
    pub const fn new(run: StreamRunIdentity) -> Self {
        Self {
            run,
            terminals: Vec::new(),
            accepted_through: None,
        }
    }

    /// Borrow the logical run this inventory is bound to.
    #[must_use]
    pub const fn run(&self) -> &StreamRunIdentity {
        &self.run
    }

    /// Note that `sequence` was accepted without yet reaching terminal.
    ///
    /// Recording acceptance separately from terminal is what lets
    /// [`Self::freeze_through`] tell an outstanding action apart from a
    /// sequence the host never issued: both are `None` in `terminals`, and only
    /// the accepted range is required to be complete.
    pub fn record_accepted(&mut self, sequence: GlobalSequence) -> Result<(), ActionExecutionError> {
        self.reserve_through(sequence)?;
        if self
            .accepted_through
            .is_none_or(|accepted| sequence > accepted)
        {
            self.accepted_through = Some(sequence);
        }
        Ok(())
    }

    /// Record one finalized action's terminal membership at its dense sequence.
    ///
    /// A repeated record with the identical digest is accepted so a replayed
    /// terminal is idempotent; a repeated record with a different digest is
    /// refused as a duplicate terminal rather than silently overwritten.
    pub fn record_terminal(
        &mut self,
        sequence: GlobalSequence,
        membership_digest: ContentDigest,
    ) -> Result<(), ActionExecutionError> {
        self.record_accepted(sequence)?;
        let index = Self::index(sequence)?;
        match self.terminals.get(index) {
            Some(Some(existing)) if *existing != membership_digest => Err(
                ActionExecutionError::action(ActionFailureCode::DuplicateTerminal),
            ),
            Some(Some(_)) => Ok(()),
            Some(None) => {
                self.terminals[index] = Some(membership_digest);
                Ok(())
            }
            // `reserve_through` above grew the vector past `index`.
            None => Err(ActionExecutionError::action(ActionFailureCode::EventOrder)),
        }
    }

    /// Return whether every accepted sequence through `through` has terminated.
    #[must_use]
    pub fn is_dense_through(&self, through: GlobalSequence) -> bool {
        let Ok(index) = Self::index(through) else {
            return false;
        };
        match self.terminals.get(..=index) {
            Some(covered) => covered.iter().all(Option::is_some),
            None => false,
        }
    }

    /// Freeze through `sequence` and mint the sealed gap-closure inventory.
    ///
    /// Fails while any sequence at or below `sequence` is unterminated, and
    /// fails when `sequence` is beyond anything the host ever accepted: a proof
    /// may not outrun either the terminals it covers or the work it describes.
    pub fn freeze_through(
        &self,
        sequence: GlobalSequence,
    ) -> Result<FrozenActionInventory, ActionExecutionError> {
        let accepted = self
            .accepted_through
            .ok_or_else(|| ActionExecutionError::action(ActionFailureCode::UnknownAction))?;
        if sequence > accepted {
            return Err(ActionExecutionError::action(
                ActionFailureCode::UnknownAction,
            ));
        }
        if !self.is_dense_through(sequence) {
            return Err(ActionExecutionError::action(ActionFailureCode::EventOrder));
        }
        let index = Self::index(sequence)?;
        let mut terminals = BTreeMap::new();
        for (offset, digest) in self.terminals[..=index].iter().enumerate() {
            let digest = digest
                .ok_or_else(|| ActionExecutionError::action(ActionFailureCode::EventOrder))?;
            let position = u64::try_from(offset)
                .map_err(|_| ActionExecutionError::action(ActionFailureCode::EventOrder))?;
            terminals.insert(GlobalSequence::new(position), digest);
        }
        let membership_root = membership_root(&self.run, sequence, &terminals);
        Ok(FrozenActionInventory::new(
            self.run,
            sequence,
            membership_root,
            terminals,
        ))
    }

    /// Grow the dense vector so `sequence` is addressable.
    fn reserve_through(&mut self, sequence: GlobalSequence) -> Result<(), ActionExecutionError> {
        let index = Self::index(sequence)?;
        let required = index
            .checked_add(1)
            .ok_or_else(|| ActionExecutionError::action(ActionFailureCode::EventOrder))?;
        if self.terminals.len() < required {
            self.terminals.resize(required, None);
        }
        Ok(())
    }

    fn index(sequence: GlobalSequence) -> Result<usize, ActionExecutionError> {
        usize::try_from(sequence.get())
            .map_err(|_| ActionExecutionError::action(ActionFailureCode::EventOrder))
    }
}

/// Bind the run, frontier, and exact ordered membership into one root digest.
fn membership_root(
    run: &StreamRunIdentity,
    through: GlobalSequence,
    terminals: &BTreeMap<GlobalSequence, ContentDigest>,
) -> ContentDigest {
    let mut hasher = blake3::Hasher::new();
    hasher.update(MEMBERSHIP_ROOT_DOMAIN);
    hasher.update(run.logical_replay_run().as_bytes());
    hasher.update(&through.get().to_le_bytes());
    for (sequence, digest) in terminals {
        hasher.update(&sequence.get().to_le_bytes());
        hasher.update(digest.as_bytes());
    }
    ContentDigest::from_bytes(*hasher.finalize().as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::streaming::{action::FrozenActionInventoryView, identity::LogicalReplayRunId};

    fn ledger() -> ActionInventoryLedger {
        ActionInventoryLedger::new(StreamRunIdentity::new(LogicalReplayRunId::from_bytes(
            [0x41; 32],
        )))
    }

    fn digest(byte: u8) -> ContentDigest {
        ContentDigest::from_bytes([byte; 32])
    }

    #[test]
    fn freeze_refuses_until_every_covered_sequence_terminates() {
        let mut inventory = ledger();
        inventory
            .record_accepted(GlobalSequence::new(0))
            .expect("accept 0");
        inventory
            .record_accepted(GlobalSequence::new(1))
            .expect("accept 1");
        inventory
            .record_terminal(GlobalSequence::new(0), digest(0xA0))
            .expect("terminal 0");

        assert!(inventory.freeze_through(GlobalSequence::new(1)).is_err());

        inventory
            .record_terminal(GlobalSequence::new(1), digest(0xA1))
            .expect("terminal 1");
        let frozen = inventory
            .freeze_through(GlobalSequence::new(1))
            .expect("dense freeze");
        assert_eq!(frozen.through(), GlobalSequence::new(1));
        assert!(frozen.contains_terminal(GlobalSequence::new(0), digest(0xA0)));
        assert!(frozen.contains_terminal(GlobalSequence::new(1), digest(0xA1)));
        assert!(!frozen.contains_terminal(GlobalSequence::new(1), digest(0xA0)));
    }

    #[test]
    fn membership_root_binds_the_exact_covered_membership() {
        let mut wide = ledger();
        wide.record_terminal(GlobalSequence::new(0), digest(0xB0))
            .expect("terminal 0");
        wide.record_terminal(GlobalSequence::new(1), digest(0xB1))
            .expect("terminal 1");

        let narrow = wide
            .freeze_through(GlobalSequence::new(0))
            .expect("freeze 0");
        let full = wide
            .freeze_through(GlobalSequence::new(1))
            .expect("freeze 1");
        assert_ne!(narrow.membership_root(), full.membership_root());
        assert!(!narrow.contains_terminal(GlobalSequence::new(1), digest(0xB1)));
    }

    #[test]
    fn conflicting_terminal_digest_is_refused_and_repeat_is_idempotent() {
        let mut inventory = ledger();
        inventory
            .record_terminal(GlobalSequence::new(0), digest(0xC0))
            .expect("terminal 0");
        inventory
            .record_terminal(GlobalSequence::new(0), digest(0xC0))
            .expect("idempotent repeat");
        assert!(
            inventory
                .record_terminal(GlobalSequence::new(0), digest(0xC1))
                .is_err()
        );
    }
}
