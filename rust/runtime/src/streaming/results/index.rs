// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Bounded, persistent, content-addressed result-segment index.
//!
//! The index maps one canonical logical membership to one committed result
//! descriptor. It is versioned by checkpoint generation: the root produced by
//! [`ResultIndexBuilder::stage`] is the value the backend folds into a committed
//! generation's result index root, and entries become reachable only through
//! [`ResultIndexBuilder::confirm_committed`].
//!
//! This index is unrelated to [`crate::dataset::segment::SegmentStore`], which
//! interns request *input* payloads. It is also unrelated to
//! `streaming::action::FrozenActionInventoryView::membership_root`, which binds
//! terminal-action gap closure rather than result membership.

use std::{
    collections::BTreeMap,
    num::{NonZeroU64, NonZeroUsize},
};

use super::{
    ResultPlaneError, ResultProjectionId, ResultSegmentDescriptor, canonical_result_index_object,
    descriptor_retained_bytes,
};
use crate::streaming::{
    checkpoint::{CheckpointEpoch, StreamRunIdentity},
    identity::{ContentDigest, GlobalSequence, StableActionId},
    reliability::HandledIssueCut,
};

/// Canonical hash domain for an interval-shaped result membership.
const MEMBERSHIP_INTERVAL_DOMAIN: &[u8] = b"aiperf.streaming.result-membership-interval.v1";
/// Canonical hash domain for an enumerated result membership block.
const MEMBERSHIP_BLOCK_DOMAIN: &[u8] = b"aiperf.streaming.result-membership-block.v1";
/// Canonical hash domain for an interior result-index block.
const INDEX_BLOCK_DOMAIN: &[u8] = b"aiperf.streaming.result-index-block.v1";

/// Append one length-prefixed field so no boundary can be forged.
fn update_field(hasher: &mut blake3::Hasher, field: &[u8]) {
    hasher.update(&(field.len() as u64).to_le_bytes());
    hasher.update(field);
}

/// Caller-owned upper bounds for one index instance.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ResultIndexLimits {
    /// Maximum descriptors encoded into one leaf block.
    pub max_entries_per_block: NonZeroUsize,
    /// Maximum reachable plus staged entries retained by one builder.
    pub max_retained_entries: NonZeroUsize,
    /// Maximum reachable plus staged descriptor bytes retained by one builder.
    pub max_retained_bytes: NonZeroU64,
}

/// Canonical logical membership represented by one result partition.
///
/// Placement uses [`ResultMembershipKey::Interval`] only when it proves the
/// partition covers a contiguous exclusive range and nothing else; every other
/// shape enumerates its sorted logical action identities.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ResultMembershipKey {
    /// A contiguous, exclusively owned global-sequence range.
    Interval {
        /// First covered global sequence.
        first: GlobalSequence,
        /// Last covered global sequence.
        last: GlobalSequence,
        /// Logical items covered by the interval.
        item_count: u64,
    },
    /// An explicit sorted, deduplicated logical action enumeration.
    Enumerated {
        /// Ascending unique logical action identities.
        actions: Box<[StableActionId]>,
    },
}

impl ResultMembershipKey {
    /// Build an interval membership, refusing an empty or inverted range.
    pub fn interval(
        first: GlobalSequence,
        last: GlobalSequence,
        item_count: u64,
    ) -> Result<Self, ResultPlaneError> {
        if first > last || item_count == 0 {
            return Err(ResultPlaneError::InvalidCoverage);
        }
        Ok(Self::Interval {
            first,
            last,
            item_count,
        })
    }

    /// Build an enumerated membership by sorting and deduplicating in place.
    pub fn enumerated(mut actions: Vec<StableActionId>) -> Result<Self, ResultPlaneError> {
        if actions.is_empty() {
            return Err(ResultPlaneError::InvalidCoverage);
        }
        actions.sort_unstable();
        actions.dedup();
        Ok(Self::Enumerated {
            actions: actions.into_boxed_slice(),
        })
    }

    /// Derive the canonical domain-separated membership root.
    ///
    /// The two domains are disjoint, so an interval membership can never
    /// collide with an enumerated one covering the same actions.
    #[must_use]
    pub fn root(&self) -> ContentDigest {
        let mut hasher = blake3::Hasher::new();
        match self {
            Self::Interval {
                first,
                last,
                item_count,
            } => {
                update_field(&mut hasher, MEMBERSHIP_INTERVAL_DOMAIN);
                update_field(&mut hasher, &first.get().to_le_bytes());
                update_field(&mut hasher, &last.get().to_le_bytes());
                update_field(&mut hasher, &item_count.to_le_bytes());
            }
            Self::Enumerated { actions } => {
                update_field(&mut hasher, MEMBERSHIP_BLOCK_DOMAIN);
                update_field(&mut hasher, &(actions.len() as u64).to_le_bytes());
                for action in actions.iter() {
                    update_field(&mut hasher, action.as_bytes());
                }
            }
        }
        ContentDigest::from_bytes(*hasher.finalize().as_bytes())
    }

    /// Logical items covered by this membership.
    #[must_use]
    pub fn item_count(&self) -> u64 {
        match self {
            Self::Interval { item_count, .. } => *item_count,
            Self::Enumerated { actions } => actions.len() as u64,
        }
    }
}

/// Projection-scoped identity of one reachable membership.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct ResultIndexKey {
    /// Projection owning the membership.
    pub projection: ResultProjectionId,
    /// Canonical membership root.
    pub membership_root: ContentDigest,
}

/// One descriptor bound to the membership its root must re-derive from.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResultIndexEntry {
    descriptor: ResultSegmentDescriptor,
    membership: ResultMembershipKey,
}

impl ResultIndexEntry {
    /// Bind a descriptor to its membership, refusing a root that disagrees.
    pub fn new(
        descriptor: ResultSegmentDescriptor,
        membership: ResultMembershipKey,
    ) -> Result<Self, ResultPlaneError> {
        if membership.root() != descriptor.membership_root {
            return Err(ResultPlaneError::SegmentVerification);
        }
        if descriptor.first_sequence > descriptor.last_sequence
            || descriptor.item_count == 0
            || descriptor.item_count != membership.item_count()
        {
            return Err(ResultPlaneError::InvalidCoverage);
        }
        Ok(Self {
            descriptor,
            membership,
        })
    }

    /// Borrow the checked descriptor.
    #[must_use]
    pub const fn descriptor(&self) -> &ResultSegmentDescriptor {
        &self.descriptor
    }

    /// Borrow the canonical membership.
    #[must_use]
    pub const fn membership(&self) -> &ResultMembershipKey {
        &self.membership
    }

    /// Return the projection-scoped index key.
    #[must_use]
    pub fn key(&self) -> ResultIndexKey {
        ResultIndexKey {
            projection: self.descriptor.projection.clone(),
            membership_root: self.descriptor.membership_root,
        }
    }

    /// Total canonical order: dense position first, identity as tie-break.
    fn order_key(&self) -> (GlobalSequence, GlobalSequence, &str, u32, u32, &[u8; 32]) {
        (
            self.descriptor.first_sequence,
            self.descriptor.last_sequence,
            self.descriptor.projection.as_str(),
            self.descriptor.cell_id.get(),
            self.descriptor.worker_id.get(),
            self.descriptor.membership_root.as_bytes(),
        )
    }
}

/// Restart admission authority derived from a restored handled-issue cut.
///
/// The index never derives which actions sit under a handled root; that is
/// ledger work. It requires the caller to attest the exact cut it was built
/// from, so a stale filter cannot silently readmit a skipped input.
pub trait HandledMembershipAdmission {
    /// Borrow the exact handled cut this admission was derived from.
    fn attested_cut(&self) -> &HandledIssueCut;

    /// Whether the key names input that was permanently skipped.
    fn is_permanently_skipped(&self, key: &ResultIndexKey) -> bool;

    /// Whether the key names a session withdrawn by a quarantine tombstone.
    fn is_quarantined(&self, key: &ResultIndexKey) -> bool;
}

/// Admission for a restored generation whose handled cut is canonically empty.
#[derive(Clone, Debug)]
pub struct AdmitEveryRestoredMembership {
    cut: HandledIssueCut,
}

impl Default for AdmitEveryRestoredMembership {
    fn default() -> Self {
        Self {
            cut: HandledIssueCut::empty(),
        }
    }
}

impl HandledMembershipAdmission for AdmitEveryRestoredMembership {
    fn attested_cut(&self) -> &HandledIssueCut {
        &self.cut
    }

    fn is_permanently_skipped(&self, _key: &ResultIndexKey) -> bool {
        false
    }

    fn is_quarantined(&self, _key: &ResultIndexKey) -> bool {
        false
    }
}

/// Frozen canonical index for one epoch, not yet made reachable.
///
/// ```compile_fail
/// # use aiperf_runtime::streaming::results::index::StagedResultIndex;
/// # fn cannot_forge_root(value: StagedResultIndex) {
/// let _root = value.root;
/// # }
/// ```
#[derive(Debug)]
pub struct StagedResultIndex {
    epoch: CheckpointEpoch,
    root: ContentDigest,
    entries: Vec<ResultIndexEntry>,
    retained_bytes: u64,
}

impl StagedResultIndex {
    /// Borrow the canonical root this epoch would publish.
    #[must_use]
    pub const fn root(&self) -> &ContentDigest {
        &self.root
    }

    /// Return the epoch this staging was frozen for.
    #[must_use]
    pub const fn epoch(&self) -> CheckpointEpoch {
        self.epoch
    }

    /// Borrow the canonically ordered reachable-if-committed descriptors.
    #[must_use]
    pub fn descriptors(&self) -> impl ExactSizeIterator<Item = &ResultSegmentDescriptor> {
        self.entries.iter().map(ResultIndexEntry::descriptor)
    }

    /// Return the exact retained descriptor bytes for this staging.
    #[must_use]
    pub const fn retained_bytes(&self) -> u64 {
        self.retained_bytes
    }
}

/// Copy-on-write logical result index for one logical run.
///
/// Committed entries are reachable; staged entries are not, and an abandoned
/// staging leaves orphans the logical index never consults.
#[derive(Debug)]
pub struct ResultIndexBuilder {
    run: StreamRunIdentity,
    limits: ResultIndexLimits,
    committed: BTreeMap<ResultIndexKey, ResultIndexEntry>,
    pending: BTreeMap<ResultIndexKey, ResultIndexEntry>,
    committed_root: Option<ContentDigest>,
    retained_bytes: u64,
}

impl ResultIndexBuilder {
    /// Construct an empty index for one logical run.
    #[must_use]
    pub fn new(run: StreamRunIdentity, limits: ResultIndexLimits) -> Self {
        Self {
            run,
            limits,
            committed: BTreeMap::new(),
            pending: BTreeMap::new(),
            committed_root: None,
            retained_bytes: 0,
        }
    }

    /// Rebuild the reachable index from one committed generation's pages.
    ///
    /// Descriptors are consumed page by page under the reader's budget; no
    /// cumulative descriptor vector is materialized beyond the retained index
    /// itself. The derived root must equal the generation's published root, and
    /// the admission must attest the generation's own handled cut.
    pub fn restore_from_committed(
        run: StreamRunIdentity,
        limits: ResultIndexLimits,
        epoch: CheckpointEpoch,
        committed_root: ContentDigest,
        restored_cut: &HandledIssueCut,
        admission: &dyn HandledMembershipAdmission,
        pages: impl IntoIterator<Item = Vec<ResultIndexEntry>>,
    ) -> Result<Self, ResultPlaneError> {
        if admission.attested_cut() != restored_cut {
            return Err(ResultPlaneError::SegmentVerification);
        }
        let mut builder = Self::new(run, limits);
        for page in pages {
            for entry in page {
                if entry.descriptor.run != builder.run || entry.descriptor.epoch > epoch {
                    return Err(ResultPlaneError::SegmentVerification);
                }
                builder.insert(entry)?;
            }
        }
        let staged = builder.stage(epoch)?;
        if staged.root != committed_root {
            return Err(ResultPlaneError::MembershipConflict {
                membership_root: committed_root,
            });
        }
        builder.confirm_committed(staged, &committed_root)?;
        Ok(builder)
    }

    /// Insert one entry, idempotently for identical content.
    ///
    /// Returns [`ResultPlaneError::MembershipConflict`] when a reachable or
    /// pending membership already holds different content. Entries left behind
    /// by an abandoned staging are unreachable and are never consulted.
    pub fn insert(&mut self, entry: ResultIndexEntry) -> Result<(), ResultPlaneError> {
        if entry.descriptor.run != self.run {
            return Err(ResultPlaneError::SegmentVerification);
        }
        let key = entry.key();
        if let Some(existing) = self.committed.get(&key).or_else(|| self.pending.get(&key)) {
            if existing == &entry {
                return Ok(());
            }
            return Err(ResultPlaneError::MembershipConflict {
                membership_root: key.membership_root,
            });
        }
        let charge = entry_charge(&entry)?;
        let retained_entries = self.committed.len() + self.pending.len() + 1;
        let retained_bytes = self.retained_bytes.saturating_add(charge);
        if retained_entries > self.limits.max_retained_entries.get()
            || retained_bytes > self.limits.max_retained_bytes.get()
        {
            return Err(ResultPlaneError::ProvisionalCapacityExceeded {
                items: retained_entries as u64,
                bytes: retained_bytes,
            });
        }
        self.retained_bytes = retained_bytes;
        self.pending.insert(key, entry);
        Ok(())
    }

    /// Insert one entry after restart, honoring handled-issue admission.
    pub fn insert_after_restart(
        &mut self,
        entry: ResultIndexEntry,
        admission: &dyn HandledMembershipAdmission,
    ) -> Result<(), ResultPlaneError> {
        let key = entry.key();
        // A membership already published by a committed generation stays
        // published; the handled cut only gates memberships new since restart.
        let is_new_membership = !self.committed.contains_key(&key);
        if is_new_membership
            && (admission.is_permanently_skipped(&key) || admission.is_quarantined(&key))
        {
            return Err(ResultPlaneError::MembershipConflict {
                membership_root: key.membership_root,
            });
        }
        self.insert(entry)
    }

    /// Freeze committed plus pending entries into one canonical epoch root.
    ///
    /// Purely borrowing: no caller-owned value is taken, drained, or mutated, so
    /// a refusal here cannot strand a partially consumed staging handoff.
    pub fn stage(&self, epoch: CheckpointEpoch) -> Result<StagedResultIndex, ResultPlaneError> {
        let mut entries: Vec<ResultIndexEntry> = self
            .committed
            .values()
            .chain(self.pending.values())
            .cloned()
            .collect();
        entries.sort_by(|left, right| left.order_key().cmp(&right.order_key()));
        validate_interval_coverage(&entries)?;
        if entries.len() > self.limits.max_entries_per_block.get() {
            // Multi-block persistence needs a backend object kind this task does
            // not own; refuse truthfully rather than mint an unstorable root.
            return Err(ResultPlaneError::ProvisionalCapacityExceeded {
                items: entries.len() as u64,
                bytes: self.retained_bytes,
            });
        }
        // A single-block root is byte-identical to the flat canonical index the
        // landed backend re-derives, so staging changes no stored encoding.
        let (root, _encoded) =
            canonical_result_index_object(entries.iter().map(ResultIndexEntry::descriptor))
                .map_err(|error| ResultPlaneError::Compaction {
                    message: format!("could not encode result index: {error}"),
                })?;
        Ok(StagedResultIndex {
            epoch,
            root,
            entries,
            retained_bytes: self.retained_bytes,
        })
    }

    /// Promote a staging whose root the generation actually published.
    ///
    /// This is the index-side mirror of the reliability ledger's pre-CAS bind
    /// and the coordinator's post-CAS equality: the same root, checked a third
    /// time, on the reachability side of the fence.
    pub fn confirm_committed(
        &mut self,
        staged: StagedResultIndex,
        committed_root: &ContentDigest,
    ) -> Result<(), ResultPlaneError> {
        if &staged.root != committed_root {
            return Err(ResultPlaneError::MembershipConflict {
                membership_root: *committed_root,
            });
        }
        for entry in staged.entries {
            self.committed.insert(entry.key(), entry);
        }
        self.pending.clear();
        self.committed_root = Some(*committed_root);
        Ok(())
    }

    /// Discard a staging that never became authoritative.
    ///
    /// Infallible, so a failed attempt cannot fail again while cleaning up. The
    /// discarded entries become orphans the logical index never consults.
    pub fn abandon_staged(&mut self, staged: StagedResultIndex) {
        drop(staged);
        for entry in self.pending.values() {
            let charge = entry_charge(entry).unwrap_or(u64::MAX);
            self.retained_bytes = self.retained_bytes.saturating_sub(charge);
        }
        self.pending.clear();
    }

    /// Borrow the last root this index made reachable.
    #[must_use]
    pub const fn committed_root(&self) -> Option<&ContentDigest> {
        self.committed_root.as_ref()
    }

    /// Look up one reachable descriptor by its projection-scoped membership.
    #[must_use]
    pub fn reachable(&self, key: &ResultIndexKey) -> Option<&ResultSegmentDescriptor> {
        self.committed.get(key).map(ResultIndexEntry::descriptor)
    }

    /// Number of reachable memberships.
    #[must_use]
    pub fn reachable_len(&self) -> usize {
        self.committed.len()
    }
}

/// Return the exact retained-allocation charge of one entry's descriptor.
fn entry_charge(entry: &ResultIndexEntry) -> Result<u64, ResultPlaneError> {
    let bytes =
        descriptor_retained_bytes(&entry.descriptor).map_err(|_| ResultPlaneError::Compaction {
            message: "descriptor retained size overflowed".to_string(),
        })?;
    u64::try_from(bytes).map_err(|_| ResultPlaneError::Compaction {
        message: "descriptor retained size exceeded u64".to_string(),
    })
}

/// Refuse overlapping interval memberships, which cannot both be exclusive.
fn validate_interval_coverage(entries: &[ResultIndexEntry]) -> Result<(), ResultPlaneError> {
    let mut previous: Option<(&str, GlobalSequence)> = None;
    for entry in entries {
        let ResultMembershipKey::Interval { first, last, .. } = &entry.membership else {
            continue;
        };
        let projection = entry.descriptor.projection.as_str();
        if let Some((previous_projection, previous_last)) = previous
            && previous_projection == projection
            && *first <= previous_last
        {
            return Err(ResultPlaneError::InvalidCoverage);
        }
        previous = Some((projection, *last));
    }
    Ok(())
}

/// Derive the digest of one interior index block over its children.
///
/// Retained for the multi-block persistence follow-up: the encoder and its
/// domain are fixed here so the eventual backend change cannot re-invent them.
#[must_use]
pub fn interior_block_digest(level: u64, children: &[(ContentDigest, u64, u64)]) -> ContentDigest {
    let mut hasher = blake3::Hasher::new();
    update_field(&mut hasher, INDEX_BLOCK_DOMAIN);
    update_field(&mut hasher, &level.to_le_bytes());
    update_field(&mut hasher, &(children.len() as u64).to_le_bytes());
    for (digest, item_count, byte_length) in children {
        update_field(&mut hasher, digest.as_bytes());
        update_field(&mut hasher, &item_count.to_le_bytes());
        update_field(&mut hasher, &byte_length.to_le_bytes());
    }
    ContentDigest::from_bytes(*hasher.finalize().as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::streaming::{
        identity::LogicalReplayRunId,
        results::{CellId, ResultSchemaVersion, WorkerId, canonical_result_index_root},
    };

    fn run() -> StreamRunIdentity {
        StreamRunIdentity::new(LogicalReplayRunId::from_bytes([7; 32]))
    }

    fn limits() -> ResultIndexLimits {
        ResultIndexLimits {
            max_entries_per_block: NonZeroUsize::new(8).expect("nonzero literal"),
            max_retained_entries: NonZeroUsize::new(64).expect("nonzero literal"),
            max_retained_bytes: NonZeroU64::new(1 << 20).expect("nonzero literal"),
        }
    }

    fn entry(projection: &str, first: u64, last: u64, payload: &[u8]) -> ResultIndexEntry {
        let item_count = last - first + 1;
        let membership = ResultMembershipKey::interval(
            GlobalSequence::new(first),
            GlobalSequence::new(last),
            item_count,
        )
        .expect("valid interval");
        let descriptor = ResultSegmentDescriptor {
            run: run(),
            epoch: CheckpointEpoch::new(1),
            cell_id: CellId::new(0),
            worker_id: WorkerId::new(0),
            projection: ResultProjectionId::new(projection).expect("nonempty projection"),
            schema: ResultSchemaVersion::new(1),
            first_sequence: GlobalSequence::new(first),
            last_sequence: GlobalSequence::new(last),
            item_count,
            byte_length: payload.len() as u64,
            membership_root: membership.root(),
            payload_digest: ContentDigest::from_bytes(*blake3::hash(payload).as_bytes()),
        };
        ResultIndexEntry::new(descriptor, membership).expect("checked entry")
    }

    #[test]
    fn conflicting_payload_for_committed_membership_is_rejected() {
        let mut builder = ResultIndexBuilder::new(run(), limits());
        builder
            .insert(entry("records", 1, 2, b"first"))
            .expect("insert");
        let error = builder
            .insert(entry("records", 1, 2, b"second"))
            .expect_err("conflicting payload must be refused");
        assert_eq!(error.code(), "membership_conflict");
    }

    #[test]
    fn identical_insert_is_idempotent_and_order_does_not_change_the_root() {
        let mut left = ResultIndexBuilder::new(run(), limits());
        left.insert(entry("records", 1, 2, b"a")).expect("insert");
        left.insert(entry("records", 1, 2, b"a"))
            .expect("idempotent");
        left.insert(entry("records", 3, 4, b"b")).expect("insert");

        let mut right = ResultIndexBuilder::new(run(), limits());
        right.insert(entry("records", 3, 4, b"b")).expect("insert");
        right.insert(entry("records", 1, 2, b"a")).expect("insert");

        let epoch = CheckpointEpoch::new(1);
        let staged = left.stage(epoch).expect("stage");
        assert_eq!(staged.root(), right.stage(epoch).expect("stage").root());

        // The single-block root must stay byte-identical to the flat canonical
        // index the landed backend re-derives.
        let descriptors: Vec<_> = staged.descriptors().cloned().collect();
        assert_eq!(
            *staged.root(),
            canonical_result_index_root(&descriptors).expect("canonical root")
        );
    }

    #[test]
    fn interval_and_enumerated_memberships_do_not_collide() {
        let interval =
            ResultMembershipKey::interval(GlobalSequence::new(1), GlobalSequence::new(3), 3)
                .expect("valid interval");
        let enumerated = ResultMembershipKey::enumerated(vec![
            StableActionId::from_bytes([1; 32]),
            StableActionId::from_bytes([2; 32]),
            StableActionId::from_bytes([3; 32]),
        ])
        .expect("valid enumeration");
        assert_ne!(interval.root(), enumerated.root());
    }

    #[test]
    fn committed_root_mismatch_leaves_the_index_unchanged() {
        let mut builder = ResultIndexBuilder::new(run(), limits());
        builder
            .insert(entry("records", 1, 2, b"a"))
            .expect("insert");
        let staged = builder.stage(CheckpointEpoch::new(1)).expect("stage");
        let error = builder
            .confirm_committed(staged, &ContentDigest::from_bytes([9; 32]))
            .expect_err("foreign root must refuse promotion");
        assert_eq!(error.code(), "membership_conflict");
        assert_eq!(builder.reachable_len(), 0);
        assert!(builder.committed_root().is_none());
    }

    #[test]
    fn overlapping_intervals_in_one_projection_are_invalid_coverage() {
        let mut builder = ResultIndexBuilder::new(run(), limits());
        builder
            .insert(entry("records", 1, 3, b"a"))
            .expect("insert");
        builder
            .insert(entry("records", 2, 4, b"b"))
            .expect("insert");
        let error = builder
            .stage(CheckpointEpoch::new(1))
            .expect_err("overlapping exclusive intervals must refuse");
        assert_eq!(error.code(), "invalid_coverage");
    }

    #[test]
    fn restore_rebuilds_exactly_the_reachable_set() {
        let epoch = CheckpointEpoch::new(1);
        let mut builder = ResultIndexBuilder::new(run(), limits());
        builder
            .insert(entry("records", 1, 2, b"a"))
            .expect("insert");
        builder
            .insert(entry("records", 3, 4, b"b"))
            .expect("insert");
        let staged = builder.stage(epoch).expect("stage");
        let root = *staged.root();
        builder.confirm_committed(staged, &root).expect("promote");

        let admission = AdmitEveryRestoredMembership::default();
        let restored = ResultIndexBuilder::restore_from_committed(
            run(),
            limits(),
            epoch,
            root,
            &HandledIssueCut::empty(),
            &admission,
            vec![
                vec![entry("records", 3, 4, b"b")],
                vec![entry("records", 1, 2, b"a")],
            ],
        )
        .expect("restore");
        assert_eq!(restored.reachable_len(), builder.reachable_len());
        assert_eq!(restored.committed_root(), Some(&root));
    }
}
