// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Durable budgeted quarantine tombstones owned by the session host.
//!
//! A tombstone is the retired identity of one quarantined session. It is keyed
//! by the exact `(input_domain, session)` pair, holds private fields, and is not
//! `Clone`: only [`SessionQuarantineTombstoneMap::install`] — the checked
//! installer in this host subtree — constructs one, and checkpoint or result
//! transfer borrows it rather than moving the wrapper.
//!
//! The map exposes exactly one outward proof, [`SessionQuarantineTombstoneMap::checked_view`],
//! which mints the host-private `CheckedSessionQuarantineTombstoneView`. Because
//! the view borrows the map's canonical entry bytes, it can neither outlive nor
//! move the map, and no adapter can forge one.

use std::collections::BTreeMap;

use bytes::Bytes;

use super::super::closure::SessionQuarantineClosureProof;
use super::{CheckedSessionQuarantineTombstoneView, SessionQuarantineTombstoneView};
use crate::streaming::{
    budget::{BudgetError, BudgetLease, StreamingResourceBudget},
    checkpoint::{BudgetedCheckpointBytes, StreamRunIdentity},
    failure::{SessionCoordinatorError, SessionFailureCode},
    identity::{ContentDigest, SessionCausalFrontier, StableSessionKey},
    reliability::StreamingInputDomainIdentity,
    unit::StateBudgetFailureCode,
};

/// Domain separator for one canonical tombstone entry.
const TOMBSTONE_ENTRY_DOMAIN: &[u8] = b"aiperf.stream.session.quarantine-tombstone.v1";

/// Domain separator for the canonical tombstone map root.
const TOMBSTONE_MAP_DOMAIN: &[u8] = b"aiperf.stream.session.quarantine-tombstone-map.v1";

/// Exact retained identity of one quarantined session.
///
/// Deliberately not `Clone`: the budget charge is bound to this single owner.
pub struct SessionQuarantineTombstone {
    run: StreamRunIdentity,
    input_domain: StreamingInputDomainIdentity,
    session_key: StableSessionKey,
    issue_id: ContentDigest,
    causal_frontier: SessionCausalFrontier,
    closure_proof: SessionQuarantineClosureProof,
    encoded: BudgetedCheckpointBytes,
    // Charges the parsed in-memory projection the encoded bytes do not cover.
    parsed_lease: BudgetLease,
}

impl std::fmt::Debug for SessionQuarantineTombstone {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("SessionQuarantineTombstone")
            .field("session_key", &self.session_key)
            .field("issue_id", &self.issue_id)
            .field("closure_proof", &self.closure_proof)
            .finish_non_exhaustive()
    }
}

impl SessionQuarantineTombstone {
    /// Borrow the logical run that retired this session.
    #[must_use]
    pub const fn run(&self) -> &StreamRunIdentity {
        &self.run
    }

    /// Borrow the exact stream and source domain owning the retired session.
    #[must_use]
    pub const fn input_domain(&self) -> &StreamingInputDomainIdentity {
        &self.input_domain
    }

    /// Return the stable key of the retired session.
    #[must_use]
    pub const fn session_key(&self) -> StableSessionKey {
        self.session_key
    }

    /// Return the reliability issue identity that authorized quarantine.
    #[must_use]
    pub const fn issue_id(&self) -> ContentDigest {
        self.issue_id
    }

    /// Borrow the causal frontier the retired session had proven complete.
    #[must_use]
    pub const fn causal_frontier(&self) -> &SessionCausalFrontier {
        &self.causal_frontier
    }

    /// Return the checked proof that permitted closure.
    #[must_use]
    pub const fn closure_proof(&self) -> SessionQuarantineClosureProof {
        self.closure_proof
    }

    /// Borrow the canonical encoded entry bytes.
    #[must_use]
    pub fn canonical_bytes(&self) -> &[u8] {
        self.encoded.as_bytes()
    }

    /// Return the byte capacity charged for this retained tombstone.
    #[must_use]
    pub fn charged_bytes(&self) -> usize {
        self.encoded
            .charged_bytes()
            .saturating_add(self.parsed_lease.charged_bytes())
    }
}

/// Durable budgeted map of retired sessions, keyed by `(input_domain, session)`.
#[derive(Debug)]
pub struct SessionQuarantineTombstoneMap {
    run: StreamRunIdentity,
    budget: StreamingResourceBudget,
    max_entries: usize,
    entries: BTreeMap<(StreamingInputDomainIdentity, StableSessionKey), SessionQuarantineTombstone>,
    canonical_entries: Vec<u8>,
    root: ContentDigest,
    revision: u64,
}

impl SessionQuarantineTombstoneMap {
    /// Begin an empty retained tombstone map for one run.
    #[must_use]
    pub fn new(
        run: StreamRunIdentity,
        budget: StreamingResourceBudget,
        max_entries: usize,
    ) -> Self {
        let mut map = Self {
            run,
            budget,
            max_entries,
            entries: BTreeMap::new(),
            canonical_entries: Vec::new(),
            root: ContentDigest::from_bytes([0; 32]),
            revision: 0,
        };
        map.recanonicalize();
        map
    }

    /// Return the number of retained tombstones.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether no session has been retired yet.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Return the content-addressed root of the retained map.
    #[must_use]
    pub const fn root(&self) -> ContentDigest {
        self.root
    }

    /// Return the monotonic revision of the retained map.
    #[must_use]
    pub const fn revision(&self) -> u64 {
        self.revision
    }

    /// Borrow one retired session's tombstone.
    #[must_use]
    pub fn get(
        &self,
        input_domain: &StreamingInputDomainIdentity,
        session_key: StableSessionKey,
    ) -> Option<&SessionQuarantineTombstone> {
        self.entries.get(&(input_domain.clone(), session_key))
    }

    /// Whether the exact `(input_domain, session)` pair has been retired.
    #[must_use]
    pub fn contains(
        &self,
        input_domain: &StreamingInputDomainIdentity,
        session_key: StableSessionKey,
    ) -> bool {
        self.entries
            .contains_key(&(input_domain.clone(), session_key))
    }

    /// Iterate retained tombstones in canonical key order.
    pub fn iter(&self) -> impl Iterator<Item = &SessionQuarantineTombstone> {
        self.entries.values()
    }

    /// Install one checked tombstone; the only construction path that exists.
    ///
    /// Re-installing an identical retirement is idempotent. A conflicting
    /// retirement for the same key is refused rather than silently replaced.
    pub fn install(
        &mut self,
        input_domain: StreamingInputDomainIdentity,
        session_key: StableSessionKey,
        issue_id: ContentDigest,
        causal_frontier: SessionCausalFrontier,
        closure_proof: SessionQuarantineClosureProof,
    ) -> Result<(), SessionCoordinatorError> {
        let key = (input_domain.clone(), session_key);
        if let Some(existing) = self.entries.get(&key) {
            if existing.issue_id != issue_id || existing.closure_proof != closure_proof {
                return Err(SessionCoordinatorError::session(
                    SessionFailureCode::ConflictingMutation,
                ));
            }
            return Ok(());
        }
        if self.entries.len() >= self.max_entries {
            return Err(SessionCoordinatorError::state_budget(
                StateBudgetFailureCode::ItemCapacity,
            ));
        }
        let bytes = canonical_entry_bytes(
            &input_domain,
            session_key,
            issue_id,
            &causal_frontier,
            closure_proof,
        );
        let encoded_lease = self
            .budget
            .try_acquire(1, bytes.len())
            .map_err(map_budget_error)?;
        let encoded =
            BudgetedCheckpointBytes::new(Bytes::from(bytes), encoded_lease).map_err(|_| {
                SessionCoordinatorError::state_budget(StateBudgetFailureCode::ByteCapacity)
            })?;
        let parsed_lease = self
            .budget
            .try_acquire(1, size_of::<SessionQuarantineTombstone>())
            .map_err(map_budget_error)?;
        self.entries.insert(
            key,
            SessionQuarantineTombstone {
                run: self.run,
                input_domain,
                session_key,
                issue_id,
                causal_frontier,
                closure_proof,
                encoded,
                parsed_lease,
            },
        );
        self.recanonicalize();
        Ok(())
    }

    /// Extend a retired session's frontier with a later excluded fragment.
    ///
    /// A later fragment never resurrects the session: it is excluded, the
    /// retained frontier is checked-extended, and the map's root and revision
    /// both move so any prepared install acknowledgement is invalidated.
    pub fn extend_frontier(
        &mut self,
        input_domain: &StreamingInputDomainIdentity,
        session_key: StableSessionKey,
        frontier: SessionCausalFrontier,
    ) -> Result<(), SessionCoordinatorError> {
        let key = (input_domain.clone(), session_key);
        let Some(existing) = self.entries.get(&key) else {
            return Err(SessionCoordinatorError::session(
                SessionFailureCode::MissingPredecessor,
            ));
        };
        if frontier.through_sequence < existing.causal_frontier.through_sequence {
            // A frontier only ever extends; a regression is a producer conflict.
            return Err(SessionCoordinatorError::session(
                SessionFailureCode::ConflictingMutation,
            ));
        }
        if frontier == existing.causal_frontier {
            return Ok(());
        }
        let bytes = canonical_entry_bytes(
            input_domain,
            session_key,
            existing.issue_id,
            &frontier,
            existing.closure_proof,
        );
        let encoded_lease = self
            .budget
            .try_acquire(1, bytes.len())
            .map_err(map_budget_error)?;
        let encoded =
            BudgetedCheckpointBytes::new(Bytes::from(bytes), encoded_lease).map_err(|_| {
                SessionCoordinatorError::state_budget(StateBudgetFailureCode::ByteCapacity)
            })?;
        // The new charge is held before the old one is released, so the retained
        // map is never briefly undercharged.
        if let Some(entry) = self.entries.get_mut(&key) {
            entry.causal_frontier = frontier;
            entry.encoded = encoded;
        }
        self.recanonicalize();
        Ok(())
    }

    /// Borrow the retained map as a checked, sealed, non-moving proof.
    ///
    /// The returned proof is minted in this host subtree and borrows the map's
    /// canonical bytes, so 1D-R can prepare a separately charged move-only
    /// install acknowledgement without ever taking ownership of a tombstone.
    #[must_use]
    pub fn checked_view(&self) -> impl SessionQuarantineTombstoneView + '_ {
        CheckedSessionQuarantineTombstoneView::new(
            self.run,
            self.root,
            self.revision,
            &self.canonical_entries,
        )
    }

    /// Recompute the canonical entry bytes, root, and revision after a mutation.
    fn recanonicalize(&mut self) {
        // The root is exactly `blake3(canonical_entries)`: the reliability
        // installer recomputes it from the borrowed slice alone, so every
        // domain and framing byte must live inside the slice itself.
        let mut canonical = Vec::with_capacity(TOMBSTONE_MAP_DOMAIN.len() + 8);
        canonical.extend_from_slice(TOMBSTONE_MAP_DOMAIN);
        canonical.extend_from_slice(&(self.entries.len() as u64).to_le_bytes());
        for tombstone in self.entries.values() {
            let bytes = tombstone.canonical_bytes();
            canonical.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
            canonical.extend_from_slice(bytes);
        }
        self.root = ContentDigest::from_bytes(*blake3::hash(&canonical).as_bytes());
        self.canonical_entries = canonical;
        self.revision = self.revision.saturating_add(1);
    }
}

/// Canonical, order-independent bytes for one retained tombstone.
fn canonical_entry_bytes(
    input_domain: &StreamingInputDomainIdentity,
    session_key: StableSessionKey,
    issue_id: ContentDigest,
    causal_frontier: &SessionCausalFrontier,
    closure_proof: SessionQuarantineClosureProof,
) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(TOMBSTONE_ENTRY_DOMAIN.len() + 160);
    bytes.extend_from_slice(TOMBSTONE_ENTRY_DOMAIN);
    bytes.extend_from_slice(input_domain.stream_identity().as_bytes());
    bytes.extend_from_slice(input_domain.source_identity().as_bytes());
    bytes.extend_from_slice(session_key.as_bytes());
    bytes.extend_from_slice(issue_id.as_bytes());
    bytes.extend_from_slice(&causal_frontier.through_sequence.get().to_le_bytes());
    bytes.extend_from_slice(
        &causal_frontier
            .event_time
            .map_or(i64::MIN, |time| time.get())
            .to_le_bytes(),
    );
    bytes.extend_from_slice(causal_frontier.digest.as_bytes());
    bytes.push(closure_proof.canonical_tag());
    bytes
}

const fn map_budget_error(error: BudgetError) -> SessionCoordinatorError {
    match error {
        BudgetError::CapacityUnavailable | BudgetError::RequestExceedsCapacity => {
            SessionCoordinatorError::state_budget(StateBudgetFailureCode::ByteCapacity)
        }
        _ => SessionCoordinatorError::state_budget(StateBudgetFailureCode::ItemCapacity),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::streaming::{
        budget::BudgetLimits,
        identity::{GlobalSequence, ImmutableObjectIdentity, LogicalReplayRunId},
    };

    fn budget() -> StreamingResourceBudget {
        StreamingResourceBudget::new(BudgetLimits {
            max_items: 1_024,
            max_bytes: 1 << 20,
        })
        .expect("a positive budget is constructible")
    }

    fn domain() -> StreamingInputDomainIdentity {
        StreamingInputDomainIdentity::new(
            ContentDigest::from_bytes([3; 32]),
            ImmutableObjectIdentity::from_bytes([4; 32]),
        )
    }

    fn frontier(through: u64) -> SessionCausalFrontier {
        SessionCausalFrontier {
            through_sequence: GlobalSequence::new(through),
            event_time: None,
            digest: ContentDigest::from_bytes([5; 32]),
        }
    }

    fn map() -> SessionQuarantineTombstoneMap {
        SessionQuarantineTombstoneMap::new(
            StreamRunIdentity::new(LogicalReplayRunId::from_bytes([9; 32])),
            budget(),
            8,
        )
    }

    #[test]
    fn install_is_idempotent_and_refuses_a_conflicting_retirement() {
        let mut tombstones = map();
        let session = StableSessionKey::from_bytes([1; 32]);
        let issue = ContentDigest::from_bytes([2; 32]);
        tombstones
            .install(
                domain(),
                session,
                issue,
                frontier(4),
                SessionQuarantineClosureProof::HardWatermark,
            )
            .expect("first install is accepted");
        tombstones
            .install(
                domain(),
                session,
                issue,
                frontier(4),
                SessionQuarantineClosureProof::HardWatermark,
            )
            .expect("an identical retirement is idempotent");
        assert_eq!(tombstones.len(), 1);
        assert!(
            tombstones
                .install(
                    domain(),
                    session,
                    issue,
                    frontier(4),
                    SessionQuarantineClosureProof::AuthoredClose,
                )
                .is_err()
        );
    }

    #[test]
    fn a_later_fragment_extends_the_frontier_and_moves_the_view_revision() {
        let mut tombstones = map();
        let session = StableSessionKey::from_bytes([1; 32]);
        tombstones
            .install(
                domain(),
                session,
                ContentDigest::from_bytes([2; 32]),
                frontier(4),
                SessionQuarantineClosureProof::HardWatermark,
            )
            .expect("install is accepted");
        let stale_root = tombstones.root();
        let stale_revision = tombstones.revision();
        tombstones
            .extend_frontier(&domain(), session, frontier(5))
            .expect("a later fragment extends the retained frontier");
        assert_eq!(
            tombstones
                .get(&domain(), session)
                .map(|tombstone| tombstone.causal_frontier().through_sequence.get()),
            Some(5)
        );
        assert_ne!(tombstones.root(), stale_root);
        assert_ne!(tombstones.revision(), stale_revision);
        assert!(
            tombstones
                .extend_frontier(&domain(), session, frontier(4))
                .is_err()
        );
    }

    #[test]
    fn the_checked_view_binds_the_exact_canonical_entries() {
        let mut tombstones = map();
        tombstones
            .install(
                domain(),
                StableSessionKey::from_bytes([1; 32]),
                ContentDigest::from_bytes([2; 32]),
                frontier(4),
                SessionQuarantineClosureProof::VerifiedFiniteSeal,
            )
            .expect("install is accepted");
        let view = tombstones.checked_view();
        assert_eq!(view.tombstone_root(), tombstones.root());
        assert_eq!(view.revision(), tombstones.revision());
        assert_eq!(
            view.canonical_encoded_entries(),
            tombstones.canonical_entries.as_slice()
        );
    }
}
