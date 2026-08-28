// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Reachability leases and bounded mark/grace/sweep collection.
//!
//! A lease is a reachability hold, not a snapshot: it keeps the objects
//! transitively reachable from one committed generation readable, and says
//! nothing about the authoritative head. Authority is the kernel-held advisory
//! lock on an open descriptor; the recorded expiry is the second, independent
//! witness that a filesystem with a non-functional `flock` degrades to rather
//! than collecting unsafely.
//!
//! Everything declared here is backend-neutral so an object-store backend
//! implements the same vocabulary instead of inventing a parallel one. The
//! filesystem-bound half — lease guards, the two-witness probe, and the sweep —
//! lives with the local store that owns those effects.

use std::collections::{BTreeMap, BTreeSet};

use async_trait::async_trait;

use crate::streaming::{
    budget::BudgetLease,
    checkpoint::{CheckpointEpoch, CheckpointError, CheckpointGeneration, StreamRunIdentity},
    identity::ContentDigest,
};

/// Liveness of one lease file under a run's `leases/` directory.
///
/// Both witnesses must agree before a lease is reclaimable: the recorded expiry
/// must have passed on the injected clock and the advisory lock must be
/// acquirable. Modification time is never consulted.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LeaseLiveness {
    /// A holder still owns this lease, or its recorded expiry is in the future.
    Live,
    /// The lease is absent, or expired with its advisory lock available.
    Reclaimable,
}

/// Whether a collection cycle may unlink under `objects/` and `generations/`.
///
/// Sweeping requires the same exclusive per-run writer lease publication
/// requires, because a concurrent publisher legitimately skips writing an object
/// it observes already present. Holding the lock is the whole answer to "did the
/// writer crash or is it still writing" on the sweep path: a sweep can only run
/// when no live writer exists.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SweepAuthority {
    /// The exclusive per-run writer lease is held for the duration of the sweep.
    Held,
    /// Another live writer holds the lease; this cycle marks but does not sweep.
    Unavailable,
}

impl SweepAuthority {
    /// Combine two per-run authorities; any refusal makes the whole cycle one.
    #[must_use]
    pub const fn fold(self, other: Self) -> Self {
        match (self, other) {
            (Self::Held, Self::Held) => Self::Held,
            _ => Self::Unavailable,
        }
    }
}

/// Authored retention policy for one checkpoint store.
///
/// Every field is validated before it is retained, so an invalid policy performs
/// no filesystem effect at all.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CheckpointRetentionPolicy {
    /// Committed generations retained as resume roots, newest first.
    pub resume_roots: std::num::NonZeroUsize,
    /// Additional non-final generations retained beyond the resume roots.
    pub partial_history: usize,
    /// Retain final generations independently of the resume-root window.
    ///
    /// Export completion is durable state the result-sink layer owns; it reaches
    /// this backend only as an explicit report-lease release, never as a status
    /// document this backend parses.
    pub retain_final_until_exported: bool,
    /// Retain source-cache objects through the oldest resume root.
    ///
    /// Validated and retained; it has no local-backend effect because the local
    /// layout has no source-cache object kind. An object-store prefix space does,
    /// and consumes it there.
    pub retain_source_cache_through_resume_root: bool,
    /// Time an unreachable object stays condemned before it may be swept.
    pub orphan_grace_ns: u64,
    /// Lifetime granted to one prepare lease.
    pub prepare_lease_ns: u64,
    /// Lifetime granted to one reader or report lease.
    pub reader_lease_ns: u64,
}

impl CheckpointRetentionPolicy {
    /// Validate every field in declaration order.
    ///
    /// Nanosecond lifetimes must fit `i64` because every clock comparison runs
    /// against the injected clock's signed nanoseconds; a `u64` that does not fit
    /// is unrepresentable rather than saturating.
    pub fn validate(&self) -> Result<ValidatedRetentionPolicy, CheckpointError> {
        let orphan_grace_ns = checked_lease_ns(self.orphan_grace_ns)?;
        let prepare_lease_ns = checked_positive_lease_ns(self.prepare_lease_ns)?;
        let reader_lease_ns = checked_positive_lease_ns(self.reader_lease_ns)?;
        Ok(ValidatedRetentionPolicy {
            resume_roots: self.resume_roots.get(),
            partial_history: self.partial_history,
            retain_final_until_exported: self.retain_final_until_exported,
            retain_source_cache_through_resume_root: self.retain_source_cache_through_resume_root,
            orphan_grace_ns,
            prepare_lease_ns,
            reader_lease_ns,
        })
    }
}

fn checked_lease_ns(value: u64) -> Result<i64, CheckpointError> {
    i64::try_from(value).map_err(|_| CheckpointError::ObjectVerification)
}

fn checked_positive_lease_ns(value: u64) -> Result<i64, CheckpointError> {
    let value = checked_lease_ns(value)?;
    if value == 0 {
        // A zero-lifetime lease is born expired and pins nothing, which would
        // silently disable every reader pin.
        return Err(CheckpointError::ObjectVerification);
    }
    Ok(value)
}

/// Retention policy whose fields are proven representable.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ValidatedRetentionPolicy {
    resume_roots: usize,
    partial_history: usize,
    retain_final_until_exported: bool,
    retain_source_cache_through_resume_root: bool,
    orphan_grace_ns: i64,
    prepare_lease_ns: i64,
    reader_lease_ns: i64,
}

impl ValidatedRetentionPolicy {
    /// Policy derived for a store whose operator has authored no retention.
    ///
    /// It keeps the head plus one resume root, retains final generations, and
    /// serves no grace, which is the least surprising behavior before a policy
    /// is installed. Lifetimes are clamped positive because a zero-lifetime lease
    /// is born expired and would pin nothing.
    #[must_use]
    pub const fn derived(prepare_lease_ns: i64, reader_lease_ns: i64) -> Self {
        Self {
            resume_roots: 1,
            partial_history: 0,
            retain_final_until_exported: true,
            retain_source_cache_through_resume_root: false,
            orphan_grace_ns: 0,
            prepare_lease_ns: if prepare_lease_ns > 0 {
                prepare_lease_ns
            } else {
                1
            },
            reader_lease_ns: if reader_lease_ns > 0 {
                reader_lease_ns
            } else {
                1
            },
        }
    }

    /// Committed generations retained as resume roots.
    #[must_use]
    pub const fn resume_roots(&self) -> usize {
        self.resume_roots
    }

    /// Additional non-final generations retained beyond the resume roots.
    #[must_use]
    pub const fn partial_history(&self) -> usize {
        self.partial_history
    }

    /// Whether final generations are retained past the resume-root window.
    #[must_use]
    pub const fn retains_final_until_exported(&self) -> bool {
        self.retain_final_until_exported
    }

    /// Whether source-cache objects are retained through the oldest resume root.
    #[must_use]
    pub const fn retains_source_cache_through_resume_root(&self) -> bool {
        self.retain_source_cache_through_resume_root
    }

    /// Grace an unreachable object serves before it may be swept.
    #[must_use]
    pub const fn orphan_grace_ns(&self) -> i64 {
        self.orphan_grace_ns
    }

    /// Lifetime granted to one prepare lease.
    #[must_use]
    pub const fn prepare_lease_ns(&self) -> i64 {
        self.prepare_lease_ns
    }

    /// Lifetime granted to one reader or report lease.
    #[must_use]
    pub const fn reader_lease_ns(&self) -> i64 {
        self.reader_lease_ns
    }
}

/// Digests proven reachable from at least one pinned generation.
///
/// Charged against the existing read budget, because the set is derived entirely
/// from reads. A store whose mark set does not fit refuses to collect rather than
/// sweeping from a partial mark set, because a partial mark set deletes live data.
#[derive(Debug)]
pub struct ObjectMarkSet {
    marked: BTreeSet<ContentDigest>,
    _lease: BudgetLease,
}

impl ObjectMarkSet {
    /// Bind one proven-reachable digest set to the read charge that admitted it.
    #[must_use]
    pub const fn new(marked: BTreeSet<ContentDigest>, lease: BudgetLease) -> Self {
        Self {
            marked,
            _lease: lease,
        }
    }

    /// Return whether one object digest is reachable.
    #[must_use]
    pub fn contains(&self, digest: &ContentDigest) -> bool {
        self.marked.contains(digest)
    }

    /// Return the number of reachable digests.
    #[must_use]
    pub fn len(&self) -> usize {
        self.marked.len()
    }

    /// Return whether nothing is reachable.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.marked.is_empty()
    }
}

/// In-process record of when each object was first observed unreachable.
///
/// Grace is measured from a condemnation observation, never from modification
/// time and never from durable state. A restart empties the ledger and restarts
/// the grace, which retains longer rather than sweeping sooner.
#[derive(Debug, Default)]
pub struct CondemnationLedger {
    condemned: BTreeMap<ContentDigest, i64>,
}

impl CondemnationLedger {
    /// Record or retain a condemnation and report whether the grace has elapsed.
    pub fn condemn(&mut self, digest: ContentDigest, now_ns: i64, grace_ns: i64) -> bool {
        let condemned_at_ns = *self.condemned.entry(digest).or_insert(now_ns);
        now_ns.saturating_sub(condemned_at_ns) >= grace_ns
    }

    /// Clear a condemnation because the object became reachable again.
    ///
    /// Structural index-block sharing makes this a real case: an object re-enters
    /// the mark set when a lease pins an older root that still names it.
    pub fn absolve(&mut self, digest: &ContentDigest) {
        self.condemned.remove(digest);
    }

    /// Forget condemnations for digests that no longer exist in the store.
    pub fn retain_present(&mut self, present: &BTreeSet<ContentDigest>) {
        self.condemned.retain(|digest, _| present.contains(digest));
    }

    /// Number of objects currently condemned.
    #[must_use]
    pub fn len(&self) -> usize {
        self.condemned.len()
    }

    /// Whether nothing is currently condemned.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.condemned.is_empty()
    }
}

/// Outcome of one complete collection cycle.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct GcReport {
    /// Whether this cycle was permitted to unlink.
    pub authority: SweepAuthority,
    /// Generations retained as pinned roots.
    pub pinned_generations: usize,
    /// Objects proven reachable from a pinned generation.
    pub marked_objects: usize,
    /// Objects observed unreachable and serving their grace after this cycle.
    pub condemned_objects: usize,
    /// Objects unlinked by this cycle.
    pub swept_objects: usize,
    /// Generation records unlinked by this cycle.
    pub swept_generations: usize,
    /// Expired lease files unlinked by this cycle.
    pub swept_leases: usize,
    /// Transaction scratch subtrees reclaimed by this cycle.
    pub reclaimed_transactions: usize,
}

impl Default for SweepAuthority {
    fn default() -> Self {
        Self::Held
    }
}

impl GcReport {
    /// Combine one per-run outcome into a whole-cycle outcome.
    #[must_use]
    pub const fn fold(self, other: Self) -> Self {
        Self {
            authority: self.authority.fold(other.authority),
            pinned_generations: self.pinned_generations + other.pinned_generations,
            marked_objects: self.marked_objects + other.marked_objects,
            condemned_objects: self.condemned_objects + other.condemned_objects,
            swept_objects: self.swept_objects + other.swept_objects,
            swept_generations: self.swept_generations + other.swept_generations,
            swept_leases: self.swept_leases + other.swept_leases,
            reclaimed_transactions: self.reclaimed_transactions + other.reclaimed_transactions,
        }
    }
}

/// Bounded mark/grace/sweep collection over one checkpoint store.
///
/// Declared apart from the storage backend trait so backends that collect
/// nothing are not forced to implement it, and so a second persistent backend
/// reuses this vocabulary instead of inventing a parallel one.
///
/// Collection is an explicit, awaitable, page-bounded, caller-driven call. It
/// spawns nothing and sleeps on no timer: a hidden background sweep could not be
/// asserted against an injected clock, and a sweep inside the commit barrier
/// would run unbounded reclamation in the commit critical path.
#[async_trait(?Send)]
pub trait CheckpointGarbageCollector {
    /// Install the authored retention policy for one run.
    async fn set_retention_policy(
        &self,
        run: &StreamRunIdentity,
        policy: CheckpointRetentionPolicy,
    ) -> Result<(), CheckpointError>;

    /// Lower retention to `generations` committed roots besides the head.
    ///
    /// The authoritative head is retained unconditionally and is never a
    /// candidate, so zero means "retain only the head" and can never destroy a
    /// run.
    async fn retain_last_generations(&self, generations: usize) -> Result<(), CheckpointError>;

    /// Run one complete bounded mark/grace/sweep cycle over every known run.
    async fn collect_garbage(&self) -> Result<GcReport, CheckpointError>;
}

/// Lowercase hex of one 32-byte digest.
#[must_use]
pub(crate) fn hex32(bytes: &[u8; 32]) -> String {
    let mut text = String::with_capacity(64);
    for byte in bytes {
        // Two lowercase nibbles per byte; the radix is always valid.
        text.push(char::from_digit(u32::from(byte >> 4), 16).unwrap_or('0'));
        text.push(char::from_digit(u32::from(byte & 0x0f), 16).unwrap_or('0'));
    }
    text
}

/// Parse exactly 64 lowercase hex characters into digest bytes.
pub(crate) fn parse_hex32(text: &str) -> Option<[u8; 32]> {
    if text.len() != 64 {
        return None;
    }
    let mut bytes = [0u8; 32];
    for (index, slot) in bytes.iter_mut().enumerate() {
        let pair = text.get(index * 2..index * 2 + 2)?;
        *slot = u8::from_str_radix(pair, 16).ok()?;
    }
    Some(bytes)
}

/// Filename prefix of one generation-pinned lease.
///
/// The pinned generation is encoded in the name rather than added to the lease
/// record, which is frozen at four fields with `deny_unknown_fields`. Encoding it
/// here is what lets the mark phase learn every pinned generation from one
/// bounded directory page without opening a single lease file.
pub(crate) const READER_LEASE_PREFIX: &str = "reader";

/// Filename prefix of one report-retention lease.
pub(crate) const REPORT_LEASE_PREFIX: &str = "report";

/// Canonical `leases/` file name for one generation-pinned reachability lease.
///
/// The 20-digit zero-padded epoch mirrors the `generations/` naming, so
/// lexicographic order equals epoch order for leases too.
pub(crate) fn generation_lease_file_name(
    prefix: &str,
    pinned: &CheckpointGeneration,
    holder_hex: &str,
) -> String {
    format!(
        "{prefix}-{:020}-{}-{holder_hex}",
        pinned.epoch().get(),
        hex32(pinned.digest().as_bytes()),
    )
}

/// Decode the generation a `reader-`/`report-` lease file name pins.
///
/// Returns `None` for any name outside the two generation-pinned prefixes, so
/// `writer` and `prepare-<id>` entries can never be mistaken for pins.
pub(crate) fn pinned_generation_from_name(name: &str) -> Option<(bool, CheckpointGeneration)> {
    let (is_report, rest) = match name.split_once('-') {
        Some((READER_LEASE_PREFIX, rest)) => (false, rest),
        Some((REPORT_LEASE_PREFIX, rest)) => (true, rest),
        _ => return None,
    };
    let mut parts = rest.splitn(3, '-');
    let epoch = parts.next()?.parse::<u64>().ok()?;
    let digest = ContentDigest::from_bytes(parse_hex32(parts.next()?)?);
    // The holder segment is present but is not needed to identify the pin.
    parts.next()?;
    Some((
        is_report,
        CheckpointGeneration::new(CheckpointEpoch::new(epoch), digest),
    ))
}

/// Decode the generation one `generations/` entry name records.
///
/// The name is `<20-digit-epoch>-<64-hex-digest>.json`, so this is the inverse of
/// the publication naming and needs no file read.
pub(crate) fn generation_from_record_name(name: &str) -> Option<CheckpointGeneration> {
    let stem = name.strip_suffix(".json")?;
    let (epoch, digest) = stem.split_once('-')?;
    Some(CheckpointGeneration::new(
        CheckpointEpoch::new(epoch.parse::<u64>().ok()?),
        ContentDigest::from_bytes(parse_hex32(digest)?),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn generation(epoch: u64, byte: u8) -> CheckpointGeneration {
        CheckpointGeneration::new(
            CheckpointEpoch::new(epoch),
            ContentDigest::from_bytes([byte; 32]),
        )
    }

    #[test]
    fn lease_names_round_trip_and_reject_ungeneration_pinned_entries() {
        let pinned = generation(7, 0xab);
        let name = generation_lease_file_name(READER_LEASE_PREFIX, &pinned, &"c".repeat(32));

        assert_eq!(pinned_generation_from_name(&name), Some((false, pinned)));
        assert_eq!(pinned_generation_from_name("writer"), None);
        assert_eq!(pinned_generation_from_name("prepare-0123abcd"), None);
    }

    #[test]
    fn zero_and_unrepresentable_lease_lifetimes_are_refused() {
        let mut policy = CheckpointRetentionPolicy {
            resume_roots: std::num::NonZeroUsize::new(1).expect("nonzero"),
            partial_history: 0,
            retain_final_until_exported: false,
            retain_source_cache_through_resume_root: false,
            orphan_grace_ns: 0,
            prepare_lease_ns: 1,
            reader_lease_ns: 0,
        };
        assert_eq!(
            policy.validate(),
            Err(CheckpointError::ObjectVerification),
            "a zero-lifetime reader lease pins nothing"
        );

        policy.reader_lease_ns = u64::MAX;
        assert_eq!(policy.validate(), Err(CheckpointError::ObjectVerification));
    }

    #[test]
    fn grace_elapses_only_after_a_condemnation_is_retained_across_cycles() {
        let mut ledger = CondemnationLedger::default();
        let digest = ContentDigest::from_bytes([1; 32]);

        assert!(!ledger.condemn(digest, 0, 100), "first pass only condemns");
        assert!(!ledger.condemn(digest, 50, 100));
        assert!(ledger.condemn(digest, 100, 100));

        ledger.absolve(&digest);
        assert!(
            !ledger.condemn(digest, 200, 100),
            "absolution restarts grace"
        );
    }
}
