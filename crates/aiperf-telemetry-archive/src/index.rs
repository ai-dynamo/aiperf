// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Deterministic content-addressed persistent B-tree index.
//!
//! The tree follows the v1 128–256 non-root occupancy, 128/129 insertion
//! split, left-first borrow, and left-preferred merge rules. Mutations are
//! validated as one set and applied as ascending removals followed by ascending
//! additions, so caller permutation cannot change page bytes or the root hash.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::{self, Debug, Display, Formatter};

use crate::descriptor::INDEX_V1;
use crate::{CanonicalJsonError, CanonicalJsonValue, Digest, SessionId, TableId, domain_digest};

const PAGE_MAGIC: &str = "aiperf.archive.index-page.v1";
const ROOT_MAGIC: &str = "aiperf.archive.index-root.v1";
const MIN_NON_ROOT: usize = 128;
const MAX_PAGE_ARITY: usize = 256;
const SPLIT_LEFT_ARITY: usize = 128;
const MAX_PAGE_BYTES: usize = 1 << 20;

/// Closed object-kind discriminants for the primary manifest index.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[repr(u8)]
pub enum IndexObjectKind {
    /// One immutable homogeneous table partition.
    TablePartition = 1,
    /// One frame/table projection-coverage descriptor.
    ProjectionCoverage = 2,
    /// One shared encrypted raw object.
    SharedRawObject = 3,
    /// One append-only raw nonce reservation.
    RawNonceReservation = 4,
}

impl IndexObjectKind {
    fn from_u8(value: u8) -> Result<Self, IndexError> {
        match value {
            1 => Ok(Self::TablePartition),
            2 => Ok(Self::ProjectionCoverage),
            3 => Ok(Self::SharedRawObject),
            4 => Ok(Self::RawNonceReservation),
            _ => Err(IndexError::UnknownObjectKind(value)),
        }
    }

    fn permits_replacement(self) -> bool {
        matches!(Self::from_u8(self as u8), Ok(Self::ProjectionCoverage))
    }

    const fn mask(self) -> u64 {
        1_u64 << (self as u8 - 1)
    }
}

/// Opaque lexicographic index key bytes.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct IndexKey(Vec<u8>);

impl IndexKey {
    /// Constructs a non-empty tagged key.
    pub fn new(bytes: Vec<u8>) -> Result<Self, IndexError> {
        if bytes.is_empty() {
            return Err(IndexError::EmptyKey);
        }
        Ok(Self(bytes))
    }

    /// Returns exact key bytes.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        &self.0
    }

    fn manifest_object_kind(&self) -> Result<IndexObjectKind, IndexError> {
        self.0
            .first()
            .copied()
            .ok_or(IndexError::EmptyKey)
            .and_then(IndexObjectKind::from_u8)
    }

    fn primary_view(&self) -> Option<PrimaryIndexKeyView<'_>> {
        parse_primary_key(&self.0)
    }
}

#[derive(Clone, Copy, Debug)]
struct PrimaryIndexKeyView<'a> {
    kind: IndexObjectKind,
    table: Option<TableId>,
    session_id: Option<&'a [u8; 16]>,
    source_id: Option<&'a str>,
    global_source: bool,
    clock_ns: Option<i64>,
}

/// Constructor authority for §8.3 primary-manifest composite keys.
#[derive(Clone, Copy, Debug, Default)]
pub struct CompositeIndexKeyV1;

impl CompositeIndexKeyV1 {
    /// Builds a homogeneous table-partition key.
    pub fn table_partition(
        table: TableId,
        session_id: SessionId,
        source_id: Option<&str>,
        minimum_clock_ns: i64,
        logical_object_id: Digest,
    ) -> Result<IndexKey, IndexError> {
        primary_key(
            IndexObjectKind::TablePartition,
            Some(table),
            Some(session_id),
            source_id,
            Some(minimum_clock_ns),
            logical_object_id,
        )
    }

    /// Builds a frame/table projection-coverage key.
    pub fn projection_coverage(
        table: TableId,
        session_id: SessionId,
        source_id: Option<&str>,
        authoritative_clock_ns: i64,
        logical_object_id: Digest,
    ) -> Result<IndexKey, IndexError> {
        primary_key(
            IndexObjectKind::ProjectionCoverage,
            Some(table),
            Some(session_id),
            source_id,
            Some(authoritative_clock_ns),
            logical_object_id,
        )
    }

    /// Builds a shared raw-object key using all none/global sentinels.
    pub fn shared_raw_object(raw_object_id: Digest) -> IndexKey {
        primary_key(
            IndexObjectKind::SharedRawObject,
            None,
            None,
            None,
            None,
            raw_object_id,
        )
        .expect("closed raw-object key is always valid")
    }

    /// Builds an append-only raw nonce-reservation key.
    pub fn raw_nonce_reservation(key_id: &str, nonce: &[u8]) -> Result<IndexKey, IndexError> {
        if key_id.is_empty() {
            return Err(IndexError::EmptyKeyId);
        }
        if nonce.is_empty() {
            return Err(IndexError::EmptyNonce);
        }
        let logical_object_id = domain_digest(
            "aiperf.archive.raw-nonce-reservation.v1",
            &[key_id.as_bytes(), nonce],
        );
        primary_key(
            IndexObjectKind::RawNonceReservation,
            None,
            None,
            None,
            None,
            logical_object_id,
        )
    }
}

/// One canonical descriptor indexed under one exact key.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct IndexEntry {
    key: IndexKey,
    descriptor_hash: Digest,
    descriptor_bytes: Vec<u8>,
    pruning_summary: IndexPruningSummaryV1,
}

impl IndexEntry {
    /// Constructs an entry from exact canonical descriptor JSON.
    pub fn new(key: IndexKey, descriptor_bytes: Vec<u8>) -> Result<Self, IndexError> {
        let descriptor = CanonicalJsonValue::parse_canonical(&descriptor_bytes)
            .map_err(IndexError::Canonical)?;
        let pruning_summary = IndexPruningSummaryV1::from_entry(&key, &descriptor)?;
        let descriptor_hash = domain_digest(
            "aiperf.archive.index-descriptor.v1",
            &[key.as_bytes(), &descriptor_bytes],
        );
        Ok(Self {
            key,
            descriptor_hash,
            descriptor_bytes,
            pruning_summary,
        })
    }

    /// Returns the exact search key.
    #[must_use]
    pub const fn key(&self) -> &IndexKey {
        &self.key
    }

    /// Returns the key-bound descriptor hash.
    #[must_use]
    pub const fn descriptor_hash(&self) -> Digest {
        self.descriptor_hash
    }

    /// Returns the exact canonical descriptor bytes.
    #[must_use]
    pub fn descriptor_bytes(&self) -> &[u8] {
        &self.descriptor_bytes
    }

    /// Returns exact key/descriptor-derived pruning facts for this entry.
    #[must_use]
    pub const fn pruning_summary(&self) -> &IndexPruningSummaryV1 {
        &self.pruning_summary
    }
}

/// One exact removal precondition.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct IndexRemoval {
    /// Exact key to remove.
    pub key: IndexKey,
    /// Exact descriptor hash required in the parent root.
    pub expected_descriptor_hash: Digest,
}

/// Canonical multi-key mutation set, independent of authored operation order.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct IndexMutationSetV1 {
    removals: Vec<IndexRemoval>,
    additions: Vec<IndexEntry>,
}

impl IndexMutationSetV1 {
    /// Validates duplicate keys and freezes ascending removal/addition order.
    pub fn new(
        mut removals: Vec<IndexRemoval>,
        mut additions: Vec<IndexEntry>,
    ) -> Result<Self, IndexError> {
        removals.sort_unstable_by(|left, right| left.key.cmp(&right.key));
        additions.sort_unstable_by(|left, right| left.key.cmp(&right.key));
        for pair in removals.windows(2) {
            if pair[0].key == pair[1].key {
                return Err(IndexError::DuplicateRemoval(pair[0].key.clone()));
            }
        }
        for pair in additions.windows(2) {
            if pair[0].key == pair[1].key {
                return Err(IndexError::DuplicateAddition(pair[0].key.clone()));
            }
        }
        Ok(Self {
            removals,
            additions,
        })
    }

    /// Returns removals in canonical ascending order.
    #[must_use]
    pub fn removals(&self) -> &[IndexRemoval] {
        &self.removals
    }

    /// Returns additions in canonical ascending order.
    #[must_use]
    pub fn additions(&self) -> &[IndexEntry] {
        &self.additions
    }
}

/// Whether exact already-present additions are forbidden or recovery-idempotent.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MutationMode {
    /// Normal authoring rejects an existing key even when bytes match.
    Normal,
    /// Recovery of the same transaction treats an exact addition as a no-op.
    Recovery,
}

/// Inclusive Clock range summarized for one table in one index subtree.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct IndexClockRangeV1 {
    /// Smallest authoritative Clock represented by the subtree.
    pub minimum_clock_ns: i64,
    /// Largest authoritative Clock represented by the subtree.
    pub maximum_clock_ns: i64,
}

impl IndexClockRangeV1 {
    fn include(&mut self, other: Self) {
        self.minimum_clock_ns = self.minimum_clock_ns.min(other.minimum_clock_ns);
        self.maximum_clock_ns = self.maximum_clock_ns.max(other.maximum_clock_ns);
    }

    fn overlaps(self, minimum: Option<i64>, maximum: Option<i64>) -> bool {
        minimum.is_none_or(|value| self.maximum_clock_ns >= value)
            && maximum.is_none_or(|value| self.minimum_clock_ns <= value)
    }
}

/// Exact aggregate facts persisted on every page and internal child reference.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct IndexPruningSummaryV1 {
    object_kind_mask: u64,
    table_mask: u64,
    session_ids: Vec<[u8; 16]>,
    source_ids: Vec<String>,
    contains_global_source: bool,
    table_clock_ranges: BTreeMap<TableId, IndexClockRangeV1>,
    opaque_entry_count: u64,
}

impl IndexPruningSummaryV1 {
    fn empty() -> Self {
        Self {
            object_kind_mask: 0,
            table_mask: 0,
            session_ids: Vec::new(),
            source_ids: Vec::new(),
            contains_global_source: false,
            table_clock_ranges: BTreeMap::new(),
            opaque_entry_count: 0,
        }
    }

    fn opaque() -> Self {
        Self {
            opaque_entry_count: 1,
            ..Self::empty()
        }
    }

    fn from_entry(key: &IndexKey, descriptor: &CanonicalJsonValue) -> Result<Self, IndexError> {
        let Some(view) = key.primary_view() else {
            return Ok(Self::opaque());
        };
        let mut summary = Self::empty();
        summary.object_kind_mask = view.kind.mask();
        if let Some(table) = view.table {
            summary.table_mask = table_mask(table);
        }
        if let Some(session_id) = view.session_id {
            summary.session_ids.push(*session_id);
        }
        match view.source_id {
            Some(source_id) => summary.source_ids.push(source_id.to_owned()),
            None => summary.contains_global_source = view.global_source,
        }
        if let (Some(table), Some(clock_ns)) = (view.table, view.clock_ns) {
            let range = entry_clock_range(view.kind, clock_ns, descriptor)?;
            summary.table_clock_ranges.insert(table, range);
        }
        Ok(summary)
    }

    fn merge<'a>(summaries: impl IntoIterator<Item = &'a Self>) -> Result<Self, IndexError> {
        let mut merged = Self::empty();
        let mut sessions = BTreeSet::new();
        let mut sources = BTreeSet::new();
        for summary in summaries {
            merged.object_kind_mask |= summary.object_kind_mask;
            merged.table_mask |= summary.table_mask;
            merged.contains_global_source |= summary.contains_global_source;
            merged.opaque_entry_count = merged
                .opaque_entry_count
                .checked_add(summary.opaque_entry_count)
                .ok_or(IndexError::LengthOverflow)?;
            sessions.extend(summary.session_ids.iter().copied());
            sources.extend(summary.source_ids.iter().cloned());
            for (table, range) in &summary.table_clock_ranges {
                merged
                    .table_clock_ranges
                    .entry(*table)
                    .and_modify(|existing| existing.include(*range))
                    .or_insert(*range);
            }
        }
        merged.session_ids = sessions.into_iter().collect();
        merged.source_ids = sources.into_iter().collect();
        Ok(merged)
    }

    /// Returns the frozen object-kind bit mask.
    #[must_use]
    pub const fn object_kind_mask(&self) -> u64 {
        self.object_kind_mask
    }

    /// Returns the frozen table bit mask.
    #[must_use]
    pub const fn table_mask(&self) -> u64 {
        self.table_mask
    }

    /// Returns exact sorted nonzero session identifiers.
    #[must_use]
    pub fn session_ids(&self) -> &[[u8; 16]] {
        &self.session_ids
    }

    /// Returns exact UTF-8-byte-sorted physical source identifiers.
    #[must_use]
    pub fn source_ids(&self) -> &[String] {
        &self.source_ids
    }

    /// Whether at least one primary key carries the global/no-source sentinel.
    #[must_use]
    pub const fn contains_global_source(&self) -> bool {
        self.contains_global_source
    }

    /// Returns the inclusive aggregate Clock range for one table.
    #[must_use]
    pub fn table_clock_range(&self, table: TableId) -> Option<IndexClockRangeV1> {
        self.table_clock_ranges.get(&table).copied()
    }

    /// Returns the number of non-primary keys that deliberately disable pruning.
    #[must_use]
    pub const fn opaque_entry_count(&self) -> u64 {
        self.opaque_entry_count
    }

    fn contains_kind(&self, kind: IndexObjectKind) -> bool {
        self.object_kind_mask & kind.mask() != 0
    }

    fn contains_table(&self, table: TableId) -> bool {
        self.table_mask & table_mask(table) != 0
    }

    fn to_value(&self) -> CanonicalJsonValue {
        let clock_ranges = self
            .table_clock_ranges
            .iter()
            .map(|(table, range)| {
                object(vec![
                    (
                        "maximum_clock_ns",
                        integer(i128::from(range.maximum_clock_ns)),
                    ),
                    (
                        "minimum_clock_ns",
                        integer(i128::from(range.minimum_clock_ns)),
                    ),
                    ("table_id", integer(i128::from(*table as u8))),
                ])
            })
            .collect();
        object(vec![
            (
                "contains_global_source",
                CanonicalJsonValue::Bool(self.contains_global_source),
            ),
            (
                "object_kind_mask",
                integer(i128::from(self.object_kind_mask)),
            ),
            (
                "opaque_entry_count",
                integer(i128::from(self.opaque_entry_count)),
            ),
            (
                "session_ids",
                CanonicalJsonValue::Array(
                    self.session_ids
                        .iter()
                        .map(|session| string(hex(session)))
                        .collect(),
                ),
            ),
            (
                "source_ids",
                CanonicalJsonValue::Array(self.source_ids.iter().cloned().map(string).collect()),
            ),
            (
                "table_clock_ranges",
                CanonicalJsonValue::Array(clock_ranges),
            ),
            ("table_mask", integer(i128::from(self.table_mask))),
        ])
    }

    fn from_value(value: &CanonicalJsonValue) -> Result<Self, IndexError> {
        let fields = value
            .as_object()
            .ok_or(IndexError::InvalidPage("pruning_summary"))?;
        let object_kind_mask = require_u64(fields, "object_kind_mask")?;
        let table_mask_value = require_u64(fields, "table_mask")?;
        let opaque_entry_count = require_u64(fields, "opaque_entry_count")?;
        let contains_global_source = match fields.get("contains_global_source") {
            Some(CanonicalJsonValue::Bool(value)) => *value,
            _ => return Err(IndexError::InvalidPage("contains_global_source")),
        };
        let mut session_ids = Vec::new();
        for value in as_array(fields.get("session_ids"), "session_ids")? {
            let text = value
                .as_str()
                .ok_or(IndexError::InvalidPage("session_ids"))?;
            let bytes: [u8; 16] = decode_hex(text)?
                .try_into()
                .map_err(|_| IndexError::InvalidPage("session_ids"))?;
            if bytes == [0; 16]
                || session_ids
                    .last()
                    .is_some_and(|previous| previous >= &bytes)
            {
                return Err(IndexError::InvalidPage("session_ids"));
            }
            session_ids.push(bytes);
        }
        let mut source_ids = Vec::new();
        for value in as_array(fields.get("source_ids"), "source_ids")? {
            let source = value
                .as_str()
                .ok_or(IndexError::InvalidPage("source_ids"))?;
            if source.is_empty()
                || source_ids
                    .last()
                    .is_some_and(|previous: &String| previous.as_bytes() >= source.as_bytes())
            {
                return Err(IndexError::InvalidPage("source_ids"));
            }
            source_ids.push(source.to_owned());
        }
        let mut table_clock_ranges = BTreeMap::new();
        for value in as_array(fields.get("table_clock_ranges"), "table_clock_ranges")? {
            let range = value
                .as_object()
                .ok_or(IndexError::InvalidPage("table_clock_ranges"))?;
            let table_value = require_u64(range, "table_id")?;
            let table = table_from_u8(
                u8::try_from(table_value).map_err(|_| IndexError::InvalidPage("table_id"))?,
            )
            .ok_or(IndexError::InvalidPage("table_id"))?;
            let minimum_clock_ns = require_i64(range, "minimum_clock_ns")?;
            let maximum_clock_ns = require_i64(range, "maximum_clock_ns")?;
            if minimum_clock_ns > maximum_clock_ns
                || table_clock_ranges
                    .insert(
                        table,
                        IndexClockRangeV1 {
                            minimum_clock_ns,
                            maximum_clock_ns,
                        },
                    )
                    .is_some()
            {
                return Err(IndexError::InvalidPage("table_clock_ranges"));
            }
        }
        let summary = Self {
            object_kind_mask,
            table_mask: table_mask_value,
            session_ids,
            source_ids,
            contains_global_source,
            table_clock_ranges,
            opaque_entry_count,
        };
        if summary
            .table_clock_ranges
            .keys()
            .any(|table| !summary.contains_table(*table))
        {
            return Err(IndexError::InvalidPage("table_mask"));
        }
        Ok(summary)
    }
}

/// Source/global selection understood directly by the persistent index.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum IndexSourceSelectionV1 {
    /// Any source or global entry.
    Any,
    /// One exact non-empty source ID.
    Exact(String),
    /// Only the explicit global/no-source sentinel.
    Global,
}

/// Primary-manifest key and subtree-summary pruning predicate.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct IndexScanPredicateV1 {
    object_kind: IndexObjectKind,
    tables: BTreeSet<TableId>,
    session_id: Option<[u8; 16]>,
    source: IndexSourceSelectionV1,
    minimum_clock_ns: Option<i64>,
    maximum_clock_ns: Option<i64>,
}

impl IndexScanPredicateV1 {
    /// Constructs a validated table-partition predicate.
    pub fn table_partitions(
        tables: BTreeSet<TableId>,
        session_id: Option<SessionId>,
        source: IndexSourceSelectionV1,
        minimum_clock_ns: Option<i64>,
        maximum_clock_ns: Option<i64>,
    ) -> Result<Self, IndexError> {
        if matches!(&source, IndexSourceSelectionV1::Exact(value) if value.is_empty()) {
            return Err(IndexError::EmptySourceId);
        }
        if minimum_clock_ns
            .zip(maximum_clock_ns)
            .is_some_and(|(minimum, maximum)| minimum > maximum)
        {
            return Err(IndexError::ReversedClockRange);
        }
        Ok(Self {
            object_kind: IndexObjectKind::TablePartition,
            tables,
            session_id: session_id.map(|value| *value.as_bytes()),
            source,
            minimum_clock_ns,
            maximum_clock_ns,
        })
    }

    fn may_match_summary(&self, summary: &IndexPruningSummaryV1) -> bool {
        if summary.opaque_entry_count != 0 {
            return true;
        }
        if !summary.contains_kind(self.object_kind) {
            return false;
        }
        if !self.tables.is_empty()
            && !self
                .tables
                .iter()
                .any(|table| summary.contains_table(*table))
        {
            return false;
        }
        if self
            .session_id
            .is_some_and(|session| summary.session_ids.binary_search(&session).is_err())
        {
            return false;
        }
        match &self.source {
            IndexSourceSelectionV1::Any => {}
            IndexSourceSelectionV1::Exact(source)
                if summary
                    .source_ids
                    .binary_search_by(|candidate| candidate.as_bytes().cmp(source.as_bytes()))
                    .is_err() =>
            {
                return false;
            }
            IndexSourceSelectionV1::Global if !summary.contains_global_source => return false,
            IndexSourceSelectionV1::Exact(_) | IndexSourceSelectionV1::Global => {}
        }
        let tables = if self.tables.is_empty() {
            summary
                .table_clock_ranges
                .keys()
                .copied()
                .collect::<Vec<_>>()
        } else {
            self.tables.iter().copied().collect()
        };
        tables.into_iter().any(|table| {
            summary
                .table_clock_range(table)
                .is_some_and(|range| range.overlaps(self.minimum_clock_ns, self.maximum_clock_ns))
        })
    }

    fn matches_entry(&self, entry: &IndexEntry) -> bool {
        let Some(view) = entry.key.primary_view() else {
            return false;
        };
        if view.kind != self.object_kind
            || !view
                .table
                .is_some_and(|table| self.tables.is_empty() || self.tables.contains(&table))
            || self
                .session_id
                .is_some_and(|session| view.session_id != Some(&session))
        {
            return false;
        }
        match &self.source {
            IndexSourceSelectionV1::Any => {}
            IndexSourceSelectionV1::Exact(source) if view.source_id != Some(source.as_str()) => {
                return false;
            }
            IndexSourceSelectionV1::Global if !view.global_source => return false,
            IndexSourceSelectionV1::Exact(_) | IndexSourceSelectionV1::Global => {}
        }
        self.may_match_summary(&entry.pruning_summary)
    }
}

/// Observable bounded work performed by one index scan.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct IndexScanStatsV1 {
    /// Root plus child pages whose bytes must be examined.
    pub pages_read: u64,
    /// Direct child page references rejected from persisted summaries.
    pub child_pages_pruned: u64,
    /// Leaf entries examined after page pruning.
    pub entries_examined: u64,
}

/// Borrowed matching entries plus their exact scan-work accounting.
#[derive(Debug)]
pub struct IndexScanV1 {
    entries: Vec<IndexEntry>,
    stats: IndexScanStatsV1,
}

impl IndexScanV1 {
    /// Returns matching primary-index entries in composite-key order.
    #[must_use]
    pub fn entries(&self) -> &[IndexEntry] {
        &self.entries
    }

    /// Returns exact page/entry work performed by this scan.
    #[must_use]
    pub const fn stats(&self) -> IndexScanStatsV1 {
        self.stats
    }
}

/// Immutable root descriptor for one complete logical index.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct IndexRootV1 {
    /// Content hash of the root page.
    pub root_hash: Digest,
    /// Exact root page byte length.
    pub root_byte_length: u64,
    /// Tree height, where a root leaf is one.
    pub height: u16,
    /// Complete logical entry count.
    pub logical_entry_count: u64,
    /// Inclusive minimum key, absent only for the canonical empty root.
    pub minimum_key: Option<IndexKey>,
    /// Inclusive maximum key, absent only for the canonical empty root.
    pub maximum_key: Option<IndexKey>,
    /// Exact root aggregate used before reading any child page.
    pub pruning_summary: IndexPruningSummaryV1,
}

impl IndexRootV1 {
    /// Encodes the exact canonical root descriptor.
    #[must_use]
    pub fn canonical_bytes(&self) -> Vec<u8> {
        object(vec![
            (
                "descriptor_fingerprint",
                string(INDEX_V1.fingerprint().to_hex()),
            ),
            ("height", integer(i128::from(self.height))),
            (
                "logical_entry_count",
                integer(i128::from(self.logical_entry_count)),
            ),
            ("magic", string(ROOT_MAGIC)),
            ("maximum_key", optional_key(self.maximum_key.as_ref())),
            ("minimum_key", optional_key(self.minimum_key.as_ref())),
            ("pruning_summary", self.pruning_summary.to_value()),
            (
                "root_byte_length",
                integer(i128::from(self.root_byte_length)),
            ),
            ("root_hash", string(self.root_hash.to_hex())),
            ("version", integer(1)),
        ])
        .to_bytes()
    }

    pub(crate) fn embedded_value(&self) -> CanonicalJsonValue {
        object(vec![
            ("height", integer(i128::from(self.height))),
            (
                "logical_entry_count",
                integer(i128::from(self.logical_entry_count)),
            ),
            ("maximum_key", optional_key(self.maximum_key.as_ref())),
            ("minimum_key", optional_key(self.minimum_key.as_ref())),
            ("pruning_summary", self.pruning_summary.to_value()),
            (
                "root_byte_length",
                integer(i128::from(self.root_byte_length)),
            ),
            ("root_hash", string(self.root_hash.to_hex())),
        ])
    }

    pub(crate) fn from_embedded_value(value: &CanonicalJsonValue) -> Result<Self, IndexError> {
        let fields = value
            .as_object()
            .ok_or(IndexError::InvalidPage("index_root"))?;
        let root = Self {
            root_hash: Digest::parse(require_text(fields, "root_hash")?)
                .map_err(|_| IndexError::InvalidPage("root_hash"))?,
            root_byte_length: require_u64(fields, "root_byte_length")?,
            height: require_u16(fields, "height")?,
            logical_entry_count: require_u64(fields, "logical_entry_count")?,
            minimum_key: decode_optional_key(fields.get("minimum_key"))?,
            maximum_key: decode_optional_key(fields.get("maximum_key"))?,
            pruning_summary: IndexPruningSummaryV1::from_value(
                fields
                    .get("pruning_summary")
                    .ok_or(IndexError::InvalidPage("pruning_summary"))?,
            )?,
        };
        if root.height == 0
            || (root.logical_entry_count == 0
                && (root.minimum_key.is_some()
                    || root.maximum_key.is_some()
                    || root.pruning_summary != IndexPruningSummaryV1::empty()))
            || (root.logical_entry_count != 0
                && (root.minimum_key.is_none() || root.maximum_key.is_none()))
            || root.minimum_key > root.maximum_key
        {
            return Err(IndexError::InvalidPage("index_root"));
        }
        Ok(root)
    }
}

/// One verified immutable B-tree snapshot and all reachable page bytes.
#[derive(Clone, Debug)]
pub struct IndexSnapshot {
    root: IndexRootV1,
    tree: Node,
    pages: BTreeMap<Digest, Vec<u8>>,
}

impl IndexSnapshot {
    /// Constructs the one canonical empty root leaf.
    pub fn empty() -> Result<Self, IndexError> {
        Self::from_tree(Node::Leaf(Vec::new()))
    }

    /// Returns the immutable root descriptor.
    #[must_use]
    pub const fn root(&self) -> &IndexRootV1 {
        &self.root
    }

    /// Returns an entry by exact key.
    #[must_use]
    pub fn get(&self, key: &IndexKey) -> Option<&IndexEntry> {
        self.tree.get(key)
    }

    /// Iterates every logical entry in key order.
    pub fn entries(&self) -> impl Iterator<Item = &IndexEntry> {
        let mut entries = Vec::new();
        self.tree.collect_entries(&mut entries);
        entries.into_iter()
    }

    /// Scans primary-manifest entries while pruning from authenticated summaries.
    pub fn scan(
        &self,
        predicate: &IndexScanPredicateV1,
        max_pages_read: u64,
        max_entries_examined: u64,
    ) -> Result<IndexScanV1, IndexError> {
        if max_pages_read == 0 || max_entries_examined == 0 {
            return Err(IndexError::ZeroWorkBound);
        }
        if self.root.pruning_summary.opaque_entry_count != 0 {
            return Err(IndexError::OpaqueManifestEntries(
                self.root.pruning_summary.opaque_entry_count,
            ));
        }
        let mut scan = IndexScanV1 {
            entries: Vec::new(),
            stats: IndexScanStatsV1::default(),
        };
        scan_persisted_page(
            self.root.root_hash,
            self.root.root_byte_length,
            &self.pages,
            predicate,
            max_pages_read,
            max_entries_examined,
            &mut scan,
        )?;
        Ok(scan)
    }

    /// Iterates every reachable content-addressed page in hash order.
    pub fn page_objects(&self) -> impl Iterator<Item = (Digest, &[u8])> {
        self.pages
            .iter()
            .map(|(hash, bytes)| (*hash, bytes.as_slice()))
    }

    /// Applies one validated canonical mutation transaction.
    pub fn apply(
        &self,
        mutation_set: &IndexMutationSetV1,
        mode: MutationMode,
    ) -> Result<Self, IndexError> {
        validate_against_parent(self, mutation_set, mode)?;
        let mut tree = self.tree.clone();
        for removal in &mutation_set.removals {
            tree.remove_root(&removal.key)?;
        }
        for addition in &mutation_set.additions {
            if mode == MutationMode::Recovery
                && self
                    .get(&addition.key)
                    .is_some_and(|existing| existing == addition)
                && !mutation_set
                    .removals
                    .iter()
                    .any(|removal| removal.key == addition.key)
            {
                continue;
            }
            tree.insert_root(addition.clone())?;
        }
        tree.validate(true, None)?;
        Self::from_tree(tree)
    }

    /// Persists every reachable page through create-if-absent exact-byte semantics.
    pub fn persist(&self, sink: &mut dyn IndexPageSink) -> Result<(), IndexError> {
        for (hash, bytes) in &self.pages {
            sink.put_if_absent(*hash, bytes)?;
        }
        Ok(())
    }

    /// Reloads and verifies a snapshot from its immutable root and page source.
    pub fn load(root: IndexRootV1, source: &dyn IndexPageSource) -> Result<Self, IndexError> {
        let mut pages = BTreeMap::new();
        let mut visiting = BTreeSet::new();
        let tree = load_node(
            root.root_hash,
            root.root_byte_length,
            source,
            &mut pages,
            &mut visiting,
            true,
        )?;
        tree.validate(true, None)?;
        let rebuilt = Self::from_tree(tree)?;
        if rebuilt.root != root {
            return Err(IndexError::RootMismatch);
        }
        Ok(Self { pages, ..rebuilt })
    }

    fn from_tree(tree: Node) -> Result<Self, IndexError> {
        tree.validate(true, None)?;
        let mut pages = BTreeMap::new();
        let persisted = persist_node(&tree, &mut pages)?;
        let root = IndexRootV1 {
            root_hash: persisted.hash,
            root_byte_length: persisted.byte_length,
            height: persisted.height,
            logical_entry_count: persisted.entry_count,
            minimum_key: persisted.minimum_key,
            maximum_key: persisted.maximum_key,
            pruning_summary: persisted.pruning_summary,
        };
        Ok(Self { root, tree, pages })
    }
}

/// Immutable page reads used by recovery.
pub trait IndexPageSource: Debug {
    /// Returns exact bytes for one expected content hash.
    fn get(&self, hash: Digest) -> Result<Vec<u8>, IndexError>;
}

/// Create-if-absent page writes used by local and remote persistence.
pub trait IndexPageSink: Debug {
    /// Creates a page or verifies an already-present byte-identical page.
    fn put_if_absent(&mut self, hash: Digest, bytes: &[u8]) -> Result<(), IndexError>;
}

/// Deterministic in-memory content-addressed page store.
#[derive(Clone, Debug, Default)]
pub struct MemoryIndexPageStore {
    pages: BTreeMap<Digest, Vec<u8>>,
}

impl MemoryIndexPageStore {
    /// Returns the number of immutable page objects.
    #[must_use]
    pub fn len(&self) -> usize {
        self.pages.len()
    }

    /// Returns whether no page object exists.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.pages.is_empty()
    }

    /// Mutates one page for corruption-path tests and recovery tooling.
    pub fn replace_for_test(&mut self, hash: Digest, bytes: Vec<u8>) {
        self.pages.insert(hash, bytes);
    }
}

impl IndexPageSource for MemoryIndexPageStore {
    fn get(&self, hash: Digest) -> Result<Vec<u8>, IndexError> {
        self.pages
            .get(&hash)
            .cloned()
            .ok_or(IndexError::MissingPage(hash))
    }
}

impl IndexPageSink for MemoryIndexPageStore {
    fn put_if_absent(&mut self, hash: Digest, bytes: &[u8]) -> Result<(), IndexError> {
        let calculated = page_hash(bytes);
        if calculated != hash {
            return Err(IndexError::PageHashMismatch(hash));
        }
        match self.pages.get(&hash) {
            Some(existing) if existing == bytes => Ok(()),
            Some(_) => Err(IndexError::ContentAddressCollision(hash)),
            None => {
                self.pages.insert(hash, bytes.to_vec());
                Ok(())
            }
        }
    }
}

/// Invalid keys, mutations, pages, or tree invariants.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum IndexError {
    /// A generic key is empty.
    EmptyKey,
    /// A source component is present but empty.
    EmptySourceId,
    /// Raw nonce key ID is empty.
    EmptyKeyId,
    /// Raw nonce bytes are empty.
    EmptyNonce,
    /// An object-kind discriminant is unknown.
    UnknownObjectKind(u8),
    /// Canonical descriptor/page JSON failed.
    Canonical(CanonicalJsonError),
    /// A primary table descriptor cannot provide exact summary facts.
    InvalidPrimaryDescriptor(&'static str),
    /// A mutation repeats one removal key.
    DuplicateRemoval(IndexKey),
    /// A mutation repeats one addition key.
    DuplicateAddition(IndexKey),
    /// A removal target is absent.
    MissingRemoval(IndexKey),
    /// A removal names the wrong parent descriptor hash.
    RemovalHashMismatch(IndexKey),
    /// An addition collides with an existing entry outside allowed replacement.
    AdditionCollision(IndexKey),
    /// A same-key removal/addition is prohibited for this object kind.
    ReplacementForbidden(IndexKey),
    /// Tree mutation attempted a duplicate key.
    DuplicateTreeKey(IndexKey),
    /// Tree mutation attempted to remove an absent key.
    MissingTreeKey(IndexKey),
    /// A page is below or above its required occupancy.
    PageOccupancy {
        /// Observed entry/child count.
        actual: usize,
        /// Whether this was the root page.
        root: bool,
    },
    /// An internal page contains children of different heights.
    UnequalChildHeight,
    /// Keys or child ranges overlap or are unsorted.
    UnsortedKeys,
    /// Internal node unexpectedly has no child.
    EmptyInternal,
    /// Canonical page exceeds 1 MiB.
    PageTooLarge(usize),
    /// A count or byte length overflowed its frozen width.
    LengthOverflow,
    /// A referenced content-addressed page is absent.
    MissingPage(Digest),
    /// A page source failed before bytes could be verified.
    PageSource(String),
    /// A page's bytes do not match its expected hash.
    PageHashMismatch(Digest),
    /// Same content key resolved to unequal bytes.
    ContentAddressCollision(Digest),
    /// Page reference byte length disagrees with actual bytes.
    PageLengthMismatch(Digest),
    /// A content-addressed page graph contains a cycle.
    PageCycle(Digest),
    /// Page JSON shape or field validity is wrong.
    InvalidPage(&'static str),
    /// Rebuilt root descriptor disagrees with the supplied root.
    RootMismatch,
    /// A scan Clock minimum exceeds its maximum.
    ReversedClockRange,
    /// A scan work bound is zero.
    ZeroWorkBound,
    /// A manifest scan found non-primary keys that cannot be pruned safely.
    OpaqueManifestEntries(u64),
    /// A scan would read more authenticated pages than its hard bound.
    PageWorkBoundExceeded(u64),
    /// A scan would inspect more leaf entries than its hard bound.
    EntryWorkBoundExceeded(u64),
}

impl Display for IndexError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyKey => formatter.write_str("index key cannot be empty"),
            Self::EmptySourceId => formatter.write_str("index source ID cannot be empty"),
            Self::EmptyKeyId => formatter.write_str("raw nonce key ID cannot be empty"),
            Self::EmptyNonce => formatter.write_str("raw nonce cannot be empty"),
            Self::UnknownObjectKind(kind) => write!(formatter, "unknown index object kind {kind}"),
            Self::Canonical(error) => write!(formatter, "invalid canonical index JSON: {error}"),
            Self::InvalidPrimaryDescriptor(field) => {
                write!(formatter, "invalid primary-index descriptor field {field}")
            }
            Self::DuplicateRemoval(key) => write!(formatter, "duplicate index removal {key:?}"),
            Self::DuplicateAddition(key) => write!(formatter, "duplicate index addition {key:?}"),
            Self::MissingRemoval(key) => write!(formatter, "missing index removal target {key:?}"),
            Self::RemovalHashMismatch(key) => {
                write!(formatter, "index removal hash mismatch {key:?}")
            }
            Self::AdditionCollision(key) => write!(formatter, "index addition collision {key:?}"),
            Self::ReplacementForbidden(key) => {
                write!(formatter, "index replacement forbidden {key:?}")
            }
            Self::DuplicateTreeKey(key) => write!(formatter, "duplicate B-tree key {key:?}"),
            Self::MissingTreeKey(key) => write!(formatter, "missing B-tree key {key:?}"),
            Self::PageOccupancy { actual, root } => write!(
                formatter,
                "invalid {} page occupancy {actual}",
                if *root { "root" } else { "non-root" }
            ),
            Self::UnequalChildHeight => formatter.write_str("index children have unequal heights"),
            Self::UnsortedKeys => formatter.write_str("index keys/ranges are unsorted"),
            Self::EmptyInternal => formatter.write_str("index internal page has no child"),
            Self::PageTooLarge(bytes) => {
                write!(formatter, "index page has {bytes} bytes above 1 MiB")
            }
            Self::LengthOverflow => formatter.write_str("index count or byte length overflow"),
            Self::MissingPage(hash) => write!(formatter, "missing index page {hash}"),
            Self::PageSource(message) => write!(formatter, "index page source failed: {message}"),
            Self::PageHashMismatch(hash) => write!(formatter, "index page hash mismatch {hash}"),
            Self::ContentAddressCollision(hash) => {
                write!(formatter, "index content-address collision {hash}")
            }
            Self::PageLengthMismatch(hash) => {
                write!(formatter, "index page length mismatch {hash}")
            }
            Self::PageCycle(hash) => write!(formatter, "index page cycle at {hash}"),
            Self::InvalidPage(field) => write!(formatter, "invalid index page field {field}"),
            Self::RootMismatch => formatter.write_str("index root descriptor mismatch"),
            Self::ReversedClockRange => formatter.write_str("index Clock range is reversed"),
            Self::ZeroWorkBound => formatter.write_str("index scan work bounds must be positive"),
            Self::OpaqueManifestEntries(count) => {
                write!(formatter, "manifest index contains {count} opaque entries")
            }
            Self::PageWorkBoundExceeded(bound) => {
                write!(formatter, "index scan exceeded {bound} page reads")
            }
            Self::EntryWorkBoundExceeded(bound) => {
                write!(formatter, "index scan exceeded {bound} examined entries")
            }
        }
    }
}

impl std::error::Error for IndexError {}

#[derive(Clone, Debug)]
enum Node {
    Leaf(Vec<IndexEntry>),
    Internal(Vec<Node>),
}

impl Node {
    fn arity(&self) -> usize {
        match self {
            Self::Leaf(entries) => entries.len(),
            Self::Internal(children) => children.len(),
        }
    }

    fn height(&self) -> u16 {
        match self {
            Self::Leaf(_) => 1,
            Self::Internal(children) => children
                .first()
                .map_or(1, |child| child.height().saturating_add(1)),
        }
    }

    fn entry_count(&self) -> u64 {
        match self {
            Self::Leaf(entries) => u64::try_from(entries.len()).expect("page arity fits u64"),
            Self::Internal(children) => children.iter().map(Self::entry_count).sum(),
        }
    }

    fn minimum_key(&self) -> Option<&IndexKey> {
        match self {
            Self::Leaf(entries) => entries.first().map(IndexEntry::key),
            Self::Internal(children) => children.first().and_then(Self::minimum_key),
        }
    }

    fn maximum_key(&self) -> Option<&IndexKey> {
        match self {
            Self::Leaf(entries) => entries.last().map(IndexEntry::key),
            Self::Internal(children) => children.last().and_then(Self::maximum_key),
        }
    }

    fn get(&self, key: &IndexKey) -> Option<&IndexEntry> {
        match self {
            Self::Leaf(entries) => entries
                .binary_search_by(|entry| entry.key.cmp(key))
                .ok()
                .map(|index| &entries[index]),
            Self::Internal(children) => {
                let index = child_index(children, key);
                children.get(index).and_then(|child| child.get(key))
            }
        }
    }

    fn collect_entries<'a>(&'a self, output: &mut Vec<&'a IndexEntry>) {
        match self {
            Self::Leaf(entries) => output.extend(entries),
            Self::Internal(children) => {
                for child in children {
                    child.collect_entries(output);
                }
            }
        }
    }

    fn insert_root(&mut self, entry: IndexEntry) -> Result<(), IndexError> {
        if let Some(right) = self.insert(entry)? {
            let left = std::mem::replace(self, Self::Leaf(Vec::new()));
            *self = Self::Internal(vec![left, right]);
        }
        Ok(())
    }

    fn insert(&mut self, entry: IndexEntry) -> Result<Option<Self>, IndexError> {
        match self {
            Self::Leaf(entries) => {
                match entries.binary_search_by(|existing| existing.key.cmp(&entry.key)) {
                    Ok(_) => return Err(IndexError::DuplicateTreeKey(entry.key)),
                    Err(index) => entries.insert(index, entry),
                }
                if entries.len() <= MAX_PAGE_ARITY {
                    return Ok(None);
                }
                let right = entries.split_off(SPLIT_LEFT_ARITY);
                Ok(Some(Self::Leaf(right)))
            }
            Self::Internal(children) => {
                let index = child_index(children, &entry.key);
                if let Some(right) = children[index].insert(entry)? {
                    children.insert(index + 1, right);
                }
                if children.len() <= MAX_PAGE_ARITY {
                    return Ok(None);
                }
                let right = children.split_off(SPLIT_LEFT_ARITY);
                Ok(Some(Self::Internal(right)))
            }
        }
    }

    fn remove_root(&mut self, key: &IndexKey) -> Result<(), IndexError> {
        self.remove(key)?;
        loop {
            let collapse = matches!(self, Self::Internal(children) if children.len() == 1);
            if !collapse {
                break;
            }
            let Self::Internal(children) = self else {
                unreachable!();
            };
            *self = children.remove(0);
        }
        Ok(())
    }

    fn remove(&mut self, key: &IndexKey) -> Result<IndexEntry, IndexError> {
        match self {
            Self::Leaf(entries) => {
                let index = entries
                    .binary_search_by(|entry| entry.key.cmp(key))
                    .map_err(|_| IndexError::MissingTreeKey(key.clone()))?;
                Ok(entries.remove(index))
            }
            Self::Internal(children) => {
                let index = child_index(children, key);
                let removed = children[index].remove(key)?;
                if children[index].arity() < MIN_NON_ROOT {
                    rebalance(children, index)?;
                }
                Ok(removed)
            }
        }
    }

    fn validate(&self, root: bool, expected_height: Option<u16>) -> Result<(), IndexError> {
        let arity = self.arity();
        let valid = if root {
            match self {
                Self::Leaf(_) => arity <= MAX_PAGE_ARITY,
                Self::Internal(_) => (2..=MAX_PAGE_ARITY).contains(&arity),
            }
        } else {
            (MIN_NON_ROOT..=MAX_PAGE_ARITY).contains(&arity)
        };
        if !valid {
            return Err(IndexError::PageOccupancy {
                actual: arity,
                root,
            });
        }
        if expected_height.is_some_and(|height| height != self.height()) {
            return Err(IndexError::UnequalChildHeight);
        }
        match self {
            Self::Leaf(entries) => {
                if entries.windows(2).any(|pair| pair[0].key >= pair[1].key) {
                    return Err(IndexError::UnsortedKeys);
                }
            }
            Self::Internal(children) => {
                if children.is_empty() {
                    return Err(IndexError::EmptyInternal);
                }
                let child_height = children[0].height();
                for child in children {
                    child.validate(false, Some(child_height))?;
                }
                for pair in children.windows(2) {
                    if pair[0].maximum_key() >= pair[1].minimum_key() {
                        return Err(IndexError::UnsortedKeys);
                    }
                }
            }
        }
        Ok(())
    }
}

fn node_pruning_summary(node: &Node) -> Result<IndexPruningSummaryV1, IndexError> {
    match node {
        Node::Leaf(entries) => {
            IndexPruningSummaryV1::merge(entries.iter().map(IndexEntry::pruning_summary))
        }
        Node::Internal(children) => {
            let summaries = children
                .iter()
                .map(node_pruning_summary)
                .collect::<Result<Vec<_>, _>>()?;
            IndexPruningSummaryV1::merge(summaries.iter())
        }
    }
}

fn child_index(children: &[Node], key: &IndexKey) -> usize {
    children
        .iter()
        .position(|child| child.maximum_key().is_some_and(|maximum| maximum >= key))
        .unwrap_or_else(|| children.len().saturating_sub(1))
}

fn rebalance(children: &mut Vec<Node>, index: usize) -> Result<(), IndexError> {
    if index > 0 && children[index - 1].arity() > MIN_NON_ROOT {
        let (left, right) = children.split_at_mut(index);
        let borrowed = take_last(&mut left[index - 1])?;
        push_first(&mut right[0], borrowed)?;
        return Ok(());
    }
    if index + 1 < children.len() && children[index + 1].arity() > MIN_NON_ROOT {
        let (left, right) = children.split_at_mut(index + 1);
        let borrowed = take_first(&mut right[0])?;
        push_last(&mut left[index], borrowed)?;
        return Ok(());
    }
    if index > 0 {
        let right = children.remove(index);
        merge_into(&mut children[index - 1], right)?;
    } else if children.len() > 1 {
        let right = children.remove(1);
        merge_into(&mut children[0], right)?;
    } else {
        return Err(IndexError::PageOccupancy {
            actual: children[0].arity(),
            root: false,
        });
    }
    Ok(())
}

enum Borrowed {
    Entry(IndexEntry),
    Child(Node),
}

fn take_last(node: &mut Node) -> Result<Borrowed, IndexError> {
    match node {
        Node::Leaf(entries) => {
            entries
                .pop()
                .map(Borrowed::Entry)
                .ok_or(IndexError::PageOccupancy {
                    actual: 0,
                    root: false,
                })
        }
        Node::Internal(children) => {
            children
                .pop()
                .map(Borrowed::Child)
                .ok_or(IndexError::PageOccupancy {
                    actual: 0,
                    root: false,
                })
        }
    }
}

fn take_first(node: &mut Node) -> Result<Borrowed, IndexError> {
    match node {
        Node::Leaf(entries) if !entries.is_empty() => Ok(Borrowed::Entry(entries.remove(0))),
        Node::Internal(children) if !children.is_empty() => Ok(Borrowed::Child(children.remove(0))),
        _ => Err(IndexError::PageOccupancy {
            actual: 0,
            root: false,
        }),
    }
}

fn push_first(node: &mut Node, borrowed: Borrowed) -> Result<(), IndexError> {
    match (node, borrowed) {
        (Node::Leaf(entries), Borrowed::Entry(entry)) => entries.insert(0, entry),
        (Node::Internal(children), Borrowed::Child(child)) => children.insert(0, child),
        _ => return Err(IndexError::UnequalChildHeight),
    }
    Ok(())
}

fn push_last(node: &mut Node, borrowed: Borrowed) -> Result<(), IndexError> {
    match (node, borrowed) {
        (Node::Leaf(entries), Borrowed::Entry(entry)) => entries.push(entry),
        (Node::Internal(children), Borrowed::Child(child)) => children.push(child),
        _ => return Err(IndexError::UnequalChildHeight),
    }
    Ok(())
}

fn merge_into(left: &mut Node, right: Node) -> Result<(), IndexError> {
    match (left, right) {
        (Node::Leaf(left), Node::Leaf(mut right)) => left.append(&mut right),
        (Node::Internal(left), Node::Internal(mut right)) => left.append(&mut right),
        _ => return Err(IndexError::UnequalChildHeight),
    }
    Ok(())
}

#[derive(Clone, Debug)]
struct PersistedNode {
    hash: Digest,
    byte_length: u64,
    height: u16,
    entry_count: u64,
    minimum_key: Option<IndexKey>,
    maximum_key: Option<IndexKey>,
    pruning_summary: IndexPruningSummaryV1,
}

fn persist_node(
    node: &Node,
    pages: &mut BTreeMap<Digest, Vec<u8>>,
) -> Result<PersistedNode, IndexError> {
    let (bytes, pruning_summary) = match node {
        Node::Leaf(entries) => {
            let mut encoded_entries = Vec::with_capacity(entries.len());
            for entry in entries {
                let descriptor = CanonicalJsonValue::parse_canonical(&entry.descriptor_bytes)
                    .map_err(IndexError::Canonical)?;
                encoded_entries.push(object(vec![
                    ("descriptor", descriptor),
                    ("descriptor_hash", string(entry.descriptor_hash.to_hex())),
                    ("key", string(hex(entry.key.as_bytes()))),
                ]));
            }
            let summary =
                IndexPruningSummaryV1::merge(entries.iter().map(IndexEntry::pruning_summary))?;
            (
                object(vec![
                    (
                        "descriptor_fingerprint",
                        string(INDEX_V1.fingerprint().to_hex()),
                    ),
                    ("entries", CanonicalJsonValue::Array(encoded_entries)),
                    ("kind", string("leaf")),
                    ("magic", string(PAGE_MAGIC)),
                    ("pruning_summary", summary.to_value()),
                    ("version", integer(1)),
                ])
                .to_bytes(),
                summary,
            )
        }
        Node::Internal(children) => {
            let mut encoded_children = Vec::with_capacity(children.len());
            let mut persisted_children = Vec::with_capacity(children.len());
            for child in children {
                let persisted = persist_node(child, pages)?;
                encoded_children.push(object(vec![
                    ("byte_length", integer(i128::from(persisted.byte_length))),
                    ("entry_count", integer(i128::from(persisted.entry_count))),
                    ("hash", string(persisted.hash.to_hex())),
                    ("height", integer(i128::from(persisted.height))),
                    ("maximum_key", optional_key(persisted.maximum_key.as_ref())),
                    ("minimum_key", optional_key(persisted.minimum_key.as_ref())),
                    ("pruning_summary", persisted.pruning_summary.to_value()),
                ]));
                persisted_children.push(persisted);
            }
            let summary = IndexPruningSummaryV1::merge(
                persisted_children
                    .iter()
                    .map(|persisted| &persisted.pruning_summary),
            )?;
            (
                object(vec![
                    ("children", CanonicalJsonValue::Array(encoded_children)),
                    (
                        "descriptor_fingerprint",
                        string(INDEX_V1.fingerprint().to_hex()),
                    ),
                    ("kind", string("internal")),
                    ("magic", string(PAGE_MAGIC)),
                    ("pruning_summary", summary.to_value()),
                    ("version", integer(1)),
                ])
                .to_bytes(),
                summary,
            )
        }
    };
    if bytes.len() > MAX_PAGE_BYTES {
        return Err(IndexError::PageTooLarge(bytes.len()));
    }
    let hash = page_hash(&bytes);
    match pages.insert(hash, bytes.clone()) {
        Some(existing) if existing != bytes => {
            return Err(IndexError::ContentAddressCollision(hash));
        }
        _ => {}
    }
    Ok(PersistedNode {
        hash,
        byte_length: u64::try_from(bytes.len()).map_err(|_| IndexError::LengthOverflow)?,
        height: node.height(),
        entry_count: node.entry_count(),
        minimum_key: node.minimum_key().cloned(),
        maximum_key: node.maximum_key().cloned(),
        pruning_summary,
    })
}

fn load_node(
    hash: Digest,
    expected_length: u64,
    source: &dyn IndexPageSource,
    pages: &mut BTreeMap<Digest, Vec<u8>>,
    visiting: &mut BTreeSet<Digest>,
    root: bool,
) -> Result<Node, IndexError> {
    if !visiting.insert(hash) {
        return Err(IndexError::PageCycle(hash));
    }
    let bytes = source.get(hash)?;
    if u64::try_from(bytes.len()).map_err(|_| IndexError::LengthOverflow)? != expected_length {
        return Err(IndexError::PageLengthMismatch(hash));
    }
    if page_hash(&bytes) != hash {
        return Err(IndexError::PageHashMismatch(hash));
    }
    if bytes.len() > MAX_PAGE_BYTES {
        return Err(IndexError::PageTooLarge(bytes.len()));
    }
    let value = CanonicalJsonValue::parse_canonical(&bytes).map_err(IndexError::Canonical)?;
    let object = value.as_object().ok_or(IndexError::InvalidPage("object"))?;
    require_string(object, "magic", PAGE_MAGIC)?;
    require_string(
        object,
        "descriptor_fingerprint",
        &INDEX_V1.fingerprint().to_hex(),
    )?;
    require_integer(object, "version", 1)?;
    let expected_page_summary = IndexPruningSummaryV1::from_value(
        object
            .get("pruning_summary")
            .ok_or(IndexError::InvalidPage("pruning_summary"))?,
    )?;
    let kind = object
        .get("kind")
        .and_then(CanonicalJsonValue::as_str)
        .ok_or(IndexError::InvalidPage("kind"))?;
    let node = match kind {
        "leaf" => {
            let values = as_array(object.get("entries"), "entries")?;
            let mut entries = Vec::with_capacity(values.len());
            for value in values {
                let fields = value.as_object().ok_or(IndexError::InvalidPage("entry"))?;
                let key = IndexKey::new(decode_hex(require_text(fields, "key")?)?)?;
                let descriptor_hash = Digest::parse(require_text(fields, "descriptor_hash")?)
                    .map_err(|_| IndexError::InvalidPage("descriptor_hash"))?;
                let descriptor = fields
                    .get("descriptor")
                    .ok_or(IndexError::InvalidPage("descriptor"))?;
                let entry = IndexEntry::new(key, descriptor.to_bytes())?;
                if entry.descriptor_hash != descriptor_hash {
                    return Err(IndexError::InvalidPage("descriptor_hash"));
                }
                entries.push(entry);
            }
            Node::Leaf(entries)
        }
        "internal" => {
            let values = as_array(object.get("children"), "children")?;
            let mut children = Vec::with_capacity(values.len());
            for value in values {
                let fields = value.as_object().ok_or(IndexError::InvalidPage("child"))?;
                let child_hash = Digest::parse(require_text(fields, "hash")?)
                    .map_err(|_| IndexError::InvalidPage("child hash"))?;
                let byte_length = require_u64(fields, "byte_length")?;
                let expected_entry_count = require_u64(fields, "entry_count")?;
                let expected_height = require_u16(fields, "height")?;
                let expected_minimum = decode_optional_key(fields.get("minimum_key"))?;
                let expected_maximum = decode_optional_key(fields.get("maximum_key"))?;
                let expected_pruning_summary = IndexPruningSummaryV1::from_value(
                    fields
                        .get("pruning_summary")
                        .ok_or(IndexError::InvalidPage("child pruning_summary"))?,
                )?;
                let child = load_node(child_hash, byte_length, source, pages, visiting, false)?;
                if child.entry_count() != expected_entry_count
                    || child.height() != expected_height
                    || child.minimum_key().cloned() != expected_minimum
                    || child.maximum_key().cloned() != expected_maximum
                    || node_pruning_summary(&child)? != expected_pruning_summary
                {
                    return Err(IndexError::InvalidPage("child summary"));
                }
                children.push(child);
            }
            Node::Internal(children)
        }
        _ => return Err(IndexError::InvalidPage("kind")),
    };
    node.validate(root, None)?;
    if node_pruning_summary(&node)? != expected_page_summary {
        return Err(IndexError::InvalidPage("pruning_summary"));
    }
    pages.insert(hash, bytes);
    visiting.remove(&hash);
    Ok(node)
}

fn scan_persisted_page(
    hash: Digest,
    expected_length: u64,
    pages: &BTreeMap<Digest, Vec<u8>>,
    predicate: &IndexScanPredicateV1,
    max_pages_read: u64,
    max_entries_examined: u64,
    scan: &mut IndexScanV1,
) -> Result<(), IndexError> {
    let next_pages = scan
        .stats
        .pages_read
        .checked_add(1)
        .ok_or(IndexError::LengthOverflow)?;
    if next_pages > max_pages_read {
        return Err(IndexError::PageWorkBoundExceeded(max_pages_read));
    }
    scan.stats.pages_read = next_pages;
    let bytes = pages.get(&hash).ok_or(IndexError::MissingPage(hash))?;
    if u64::try_from(bytes.len()).map_err(|_| IndexError::LengthOverflow)? != expected_length {
        return Err(IndexError::PageLengthMismatch(hash));
    }
    if page_hash(bytes) != hash {
        return Err(IndexError::PageHashMismatch(hash));
    }
    let value = CanonicalJsonValue::parse_canonical(bytes).map_err(IndexError::Canonical)?;
    let fields = value.as_object().ok_or(IndexError::InvalidPage("object"))?;
    require_string(fields, "magic", PAGE_MAGIC)?;
    require_string(
        fields,
        "descriptor_fingerprint",
        &INDEX_V1.fingerprint().to_hex(),
    )?;
    require_integer(fields, "version", 1)?;
    let summary = IndexPruningSummaryV1::from_value(
        fields
            .get("pruning_summary")
            .ok_or(IndexError::InvalidPage("pruning_summary"))?,
    )?;
    if !predicate.may_match_summary(&summary) {
        return Ok(());
    }
    match require_text(fields, "kind")? {
        "leaf" => {
            for value in as_array(fields.get("entries"), "entries")? {
                let next_entries = scan
                    .stats
                    .entries_examined
                    .checked_add(1)
                    .ok_or(IndexError::LengthOverflow)?;
                if next_entries > max_entries_examined {
                    return Err(IndexError::EntryWorkBoundExceeded(max_entries_examined));
                }
                scan.stats.entries_examined = next_entries;
                let entry = decode_page_entry(value)?;
                if predicate.matches_entry(&entry) {
                    scan.entries.push(entry);
                }
            }
        }
        "internal" => {
            for value in as_array(fields.get("children"), "children")? {
                let child = value.as_object().ok_or(IndexError::InvalidPage("child"))?;
                let child_summary = IndexPruningSummaryV1::from_value(
                    child
                        .get("pruning_summary")
                        .ok_or(IndexError::InvalidPage("child pruning_summary"))?,
                )?;
                if !predicate.may_match_summary(&child_summary) {
                    scan.stats.child_pages_pruned = scan
                        .stats
                        .child_pages_pruned
                        .checked_add(1)
                        .ok_or(IndexError::LengthOverflow)?;
                    continue;
                }
                let child_hash = Digest::parse(require_text(child, "hash")?)
                    .map_err(|_| IndexError::InvalidPage("child hash"))?;
                scan_persisted_page(
                    child_hash,
                    require_u64(child, "byte_length")?,
                    pages,
                    predicate,
                    max_pages_read,
                    max_entries_examined,
                    scan,
                )?;
            }
        }
        _ => return Err(IndexError::InvalidPage("kind")),
    }
    Ok(())
}

fn decode_page_entry(value: &CanonicalJsonValue) -> Result<IndexEntry, IndexError> {
    let fields = value.as_object().ok_or(IndexError::InvalidPage("entry"))?;
    let key = IndexKey::new(decode_hex(require_text(fields, "key")?)?)?;
    let descriptor_hash = Digest::parse(require_text(fields, "descriptor_hash")?)
        .map_err(|_| IndexError::InvalidPage("descriptor_hash"))?;
    let descriptor = fields
        .get("descriptor")
        .ok_or(IndexError::InvalidPage("descriptor"))?;
    let entry = IndexEntry::new(key, descriptor.to_bytes())?;
    if entry.descriptor_hash != descriptor_hash {
        return Err(IndexError::InvalidPage("descriptor_hash"));
    }
    Ok(entry)
}

fn validate_against_parent(
    parent: &IndexSnapshot,
    mutation_set: &IndexMutationSetV1,
    mode: MutationMode,
) -> Result<(), IndexError> {
    let removals: BTreeMap<&IndexKey, &IndexRemoval> = mutation_set
        .removals
        .iter()
        .map(|removal| (&removal.key, removal))
        .collect();
    for removal in &mutation_set.removals {
        let existing = parent
            .get(&removal.key)
            .ok_or_else(|| IndexError::MissingRemoval(removal.key.clone()))?;
        if existing.descriptor_hash != removal.expected_descriptor_hash {
            return Err(IndexError::RemovalHashMismatch(removal.key.clone()));
        }
    }
    for addition in &mutation_set.additions {
        let existing = parent.get(&addition.key);
        let removal = removals.get(&addition.key);
        if removal.is_some() {
            let kind = addition.key.manifest_object_kind()?;
            if !kind.permits_replacement() {
                return Err(IndexError::ReplacementForbidden(addition.key.clone()));
            }
            continue;
        }
        if let Some(existing) = existing {
            if mode == MutationMode::Recovery && existing == addition {
                continue;
            }
            return Err(IndexError::AdditionCollision(addition.key.clone()));
        }
    }
    Ok(())
}

fn primary_key(
    kind: IndexObjectKind,
    table: Option<TableId>,
    session: Option<SessionId>,
    source: Option<&str>,
    clock_ns: Option<i64>,
    logical_object_id: Digest,
) -> Result<IndexKey, IndexError> {
    if source.is_some_and(str::is_empty) {
        return Err(IndexError::EmptySourceId);
    }
    let mut bytes = Vec::new();
    bytes.push(kind as u8);
    bytes.push(table.map_or(0, |table| table as u8));
    match session {
        Some(session) => bytes.extend_from_slice(session.as_bytes()),
        None => bytes.extend_from_slice(&[0; 16]),
    }
    match source {
        None => bytes.push(0),
        Some(source) => {
            bytes.push(1);
            bytes.extend_from_slice(
                &u32::try_from(source.len())
                    .map_err(|_| IndexError::LengthOverflow)?
                    .to_be_bytes(),
            );
            bytes.extend_from_slice(source.as_bytes());
        }
    }
    let clock_key = clock_ns.map_or(0, |clock| (clock as u64) ^ 0x8000_0000_0000_0000);
    bytes.extend_from_slice(&clock_key.to_be_bytes());
    bytes.extend_from_slice(logical_object_id.as_bytes());
    IndexKey::new(bytes)
}

fn parse_primary_key(bytes: &[u8]) -> Option<PrimaryIndexKeyView<'_>> {
    if bytes.len() < 59 {
        return None;
    }
    let kind = IndexObjectKind::from_u8(*bytes.first()?).ok()?;
    let table_byte = *bytes.get(1)?;
    let session_id: &[u8; 16] = bytes.get(2..18)?.try_into().ok()?;
    let mut offset = 18;
    let source_tag = *bytes.get(offset)?;
    offset += 1;
    let (source_id, global_source) = match source_tag {
        0 => (None, true),
        1 => {
            let length = u32::from_be_bytes(bytes.get(offset..offset + 4)?.try_into().ok()?);
            offset += 4;
            let length = usize::try_from(length).ok()?;
            if length == 0 {
                return None;
            }
            let source =
                std::str::from_utf8(bytes.get(offset..offset.checked_add(length)?)?).ok()?;
            offset += length;
            (Some(source), false)
        }
        _ => return None,
    };
    if bytes.len() != offset.checked_add(40)? {
        return None;
    }
    let clock_key = u64::from_be_bytes(bytes.get(offset..offset + 8)?.try_into().ok()?);
    match kind {
        IndexObjectKind::TablePartition | IndexObjectKind::ProjectionCoverage => {
            let table = table_from_u8(table_byte)?;
            if session_id == &[0; 16] {
                return None;
            }
            Some(PrimaryIndexKeyView {
                kind,
                table: Some(table),
                session_id: Some(session_id),
                source_id,
                global_source,
                clock_ns: Some((clock_key ^ 0x8000_0000_0000_0000) as i64),
            })
        }
        IndexObjectKind::SharedRawObject | IndexObjectKind::RawNonceReservation => {
            if table_byte != 0 || session_id != &[0; 16] || source_tag != 0 || clock_key != 0 {
                return None;
            }
            Some(PrimaryIndexKeyView {
                kind,
                table: None,
                session_id: None,
                source_id: None,
                global_source: true,
                clock_ns: None,
            })
        }
    }
}

fn entry_clock_range(
    kind: IndexObjectKind,
    key_clock_ns: i64,
    descriptor: &CanonicalJsonValue,
) -> Result<IndexClockRangeV1, IndexError> {
    if kind == IndexObjectKind::ProjectionCoverage {
        return Ok(IndexClockRangeV1 {
            minimum_clock_ns: key_clock_ns,
            maximum_clock_ns: key_clock_ns,
        });
    }
    let fields = descriptor
        .as_object()
        .ok_or(IndexError::InvalidPrimaryDescriptor("object"))?;
    let minimum_clock_ns = descriptor_i64(fields, "minimum_clock_ns")?;
    let maximum_clock_ns = descriptor_i64(fields, "maximum_clock_ns")?;
    if minimum_clock_ns != key_clock_ns || maximum_clock_ns < minimum_clock_ns {
        return Err(IndexError::InvalidPrimaryDescriptor("clock_range"));
    }
    Ok(IndexClockRangeV1 {
        minimum_clock_ns,
        maximum_clock_ns,
    })
}

const fn table_mask(table: TableId) -> u64 {
    1_u64 << (table as u8 - 1)
}

const fn table_from_u8(value: u8) -> Option<TableId> {
    match value {
        1 => Some(TableId::Attempts),
        2 => Some(TableId::Families),
        3 => Some(TableId::Samples),
        4 => Some(TableId::Markers),
        5 => Some(TableId::Losses),
        6 => Some(TableId::RawReferences),
        _ => None,
    }
}

fn descriptor_i64(
    object: &BTreeMap<String, CanonicalJsonValue>,
    field: &'static str,
) -> Result<i64, IndexError> {
    object
        .get(field)
        .and_then(CanonicalJsonValue::as_i128)
        .and_then(|value| i64::try_from(value).ok())
        .ok_or(IndexError::InvalidPrimaryDescriptor(field))
}

fn page_hash(bytes: &[u8]) -> Digest {
    domain_digest("aiperf.archive.index-node.v1", &[bytes])
}

fn object(entries: Vec<(&str, CanonicalJsonValue)>) -> CanonicalJsonValue {
    CanonicalJsonValue::object(
        entries
            .into_iter()
            .map(|(key, value)| (key.to_owned(), value)),
    )
    .expect("static archive JSON keys are unique")
}

fn string(value: impl Into<String>) -> CanonicalJsonValue {
    CanonicalJsonValue::String(value.into())
}

const fn integer(value: i128) -> CanonicalJsonValue {
    CanonicalJsonValue::Integer(value)
}

fn optional_key(key: Option<&IndexKey>) -> CanonicalJsonValue {
    key.map_or(CanonicalJsonValue::Null, |key| string(hex(key.as_bytes())))
}

fn hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        output.push(char::from(HEX[usize::from(byte >> 4)]));
        output.push(char::from(HEX[usize::from(byte & 0x0f)]));
    }
    output
}

fn decode_hex(value: &str) -> Result<Vec<u8>, IndexError> {
    if !value.len().is_multiple_of(2) {
        return Err(IndexError::InvalidPage("hex length"));
    }
    let mut output = Vec::with_capacity(value.len() / 2);
    for pair in value.as_bytes().chunks_exact(2) {
        let high = nibble(pair[0]).ok_or(IndexError::InvalidPage("hex"))?;
        let low = nibble(pair[1]).ok_or(IndexError::InvalidPage("hex"))?;
        output.push((high << 4) | low);
    }
    Ok(output)
}

fn nibble(value: u8) -> Option<u8> {
    match value {
        b'0'..=b'9' => Some(value - b'0'),
        b'a'..=b'f' => Some(value - b'a' + 10),
        _ => None,
    }
}

fn require_text<'a>(
    object: &'a BTreeMap<String, CanonicalJsonValue>,
    field: &'static str,
) -> Result<&'a str, IndexError> {
    object
        .get(field)
        .and_then(CanonicalJsonValue::as_str)
        .ok_or(IndexError::InvalidPage(field))
}

fn require_string(
    object: &BTreeMap<String, CanonicalJsonValue>,
    field: &'static str,
    expected: &str,
) -> Result<(), IndexError> {
    if require_text(object, field)? != expected {
        return Err(IndexError::InvalidPage(field));
    }
    Ok(())
}

fn require_integer(
    object: &BTreeMap<String, CanonicalJsonValue>,
    field: &'static str,
    expected: i128,
) -> Result<(), IndexError> {
    if object.get(field).and_then(CanonicalJsonValue::as_i128) != Some(expected) {
        return Err(IndexError::InvalidPage(field));
    }
    Ok(())
}

fn require_u64(
    object: &BTreeMap<String, CanonicalJsonValue>,
    field: &'static str,
) -> Result<u64, IndexError> {
    object
        .get(field)
        .and_then(CanonicalJsonValue::as_i128)
        .and_then(|value| u64::try_from(value).ok())
        .ok_or(IndexError::InvalidPage(field))
}

fn require_u16(
    object: &BTreeMap<String, CanonicalJsonValue>,
    field: &'static str,
) -> Result<u16, IndexError> {
    require_u64(object, field)?
        .try_into()
        .map_err(|_| IndexError::InvalidPage(field))
}

fn require_i64(
    object: &BTreeMap<String, CanonicalJsonValue>,
    field: &'static str,
) -> Result<i64, IndexError> {
    object
        .get(field)
        .and_then(CanonicalJsonValue::as_i128)
        .and_then(|value| i64::try_from(value).ok())
        .ok_or(IndexError::InvalidPage(field))
}

fn as_array<'a>(
    value: Option<&'a CanonicalJsonValue>,
    field: &'static str,
) -> Result<&'a [CanonicalJsonValue], IndexError> {
    match value {
        Some(CanonicalJsonValue::Array(values)) => Ok(values),
        _ => Err(IndexError::InvalidPage(field)),
    }
}

fn decode_optional_key(value: Option<&CanonicalJsonValue>) -> Result<Option<IndexKey>, IndexError> {
    match value {
        Some(CanonicalJsonValue::Null) => Ok(None),
        Some(CanonicalJsonValue::String(value)) => IndexKey::new(decode_hex(value)?).map(Some),
        _ => Err(IndexError::InvalidPage("optional key")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn session() -> SessionId {
        SessionId::new([0x11; 16]).unwrap()
    }

    fn key(number: u64) -> IndexKey {
        IndexKey::new(number.to_be_bytes().to_vec()).unwrap()
    }

    fn entry(number: u64, value: u64) -> IndexEntry {
        IndexEntry::new(key(number), format!("{{\"value\":{value}}}").into_bytes()).unwrap()
    }

    fn primary_entry(
        number: u64,
        table: TableId,
        source: Option<&str>,
        minimum_clock_ns: i64,
        maximum_clock_ns: i64,
    ) -> IndexEntry {
        let mut digest = [0_u8; 32];
        digest[24..].copy_from_slice(&number.to_be_bytes());
        let key = CompositeIndexKeyV1::table_partition(
            table,
            session(),
            source,
            minimum_clock_ns,
            Digest::from_bytes(digest),
        )
        .unwrap();
        IndexEntry::new(
            key,
            format!(
                "{{\"maximum_clock_ns\":{maximum_clock_ns},\"minimum_clock_ns\":{minimum_clock_ns}}}"
            )
            .into_bytes(),
        )
        .unwrap()
    }

    fn add_range(start: u64, end: u64) -> IndexMutationSetV1 {
        IndexMutationSetV1::new(vec![], (start..end).map(|n| entry(n, n)).collect()).unwrap()
    }

    #[test]
    fn composite_keys_pin_sentinels_and_signed_clock_order() {
        let digest = Digest::from_bytes([0x22; 32]);
        let negative = CompositeIndexKeyV1::table_partition(
            TableId::Attempts,
            session(),
            Some("source-a"),
            -1,
            digest,
        )
        .unwrap();
        let zero = CompositeIndexKeyV1::table_partition(
            TableId::Attempts,
            session(),
            Some("source-a"),
            0,
            digest,
        )
        .unwrap();
        assert!(negative < zero);
        let raw = CompositeIndexKeyV1::shared_raw_object(digest);
        assert_eq!(raw.as_bytes()[0], IndexObjectKind::SharedRawObject as u8);
        assert_eq!(raw.as_bytes()[1], 0);
        assert_eq!(&raw.as_bytes()[2..18], &[0; 16]);
        assert_eq!(raw.as_bytes()[18], 0);
        assert_eq!(&raw.as_bytes()[19..27], &[0; 8]);
    }

    #[test]
    fn insertion_split_is_exactly_128_left_129_right() {
        let snapshot = IndexSnapshot::empty()
            .unwrap()
            .apply(&add_range(0, 257), MutationMode::Normal)
            .unwrap();
        assert_eq!(snapshot.root.height, 2);
        let Node::Internal(children) = &snapshot.tree else {
            panic!("expected internal root");
        };
        assert_eq!(children[0].arity(), 128);
        assert_eq!(children[1].arity(), 129);
        assert_eq!(
            snapshot.root.root_hash.to_hex(),
            "b8f6bee8e8d1b7b2703e4ad3627d76ca4926aa9144cdbbaeca0323538bb84f09"
        );
    }

    #[test]
    fn left_borrow_precedes_merge_when_the_left_sibling_can_lend() {
        let even_entries = (0..257).map(|number| entry(number * 2, number)).collect();
        let parent = IndexSnapshot::empty()
            .unwrap()
            .apply(
                &IndexMutationSetV1::new(vec![], even_entries).unwrap(),
                MutationMode::Normal,
            )
            .unwrap();
        let expanded_left = parent
            .apply(
                &IndexMutationSetV1::new(vec![], vec![entry(1, 999)]).unwrap(),
                MutationMode::Normal,
            )
            .unwrap();
        let removals = [256_u64, 258]
            .into_iter()
            .map(|number| IndexRemoval {
                key: key(number),
                expected_descriptor_hash: expanded_left.get(&key(number)).unwrap().descriptor_hash,
            })
            .collect();
        let borrowed = expanded_left
            .apply(
                &IndexMutationSetV1::new(removals, vec![]).unwrap(),
                MutationMode::Normal,
            )
            .unwrap();
        let Node::Internal(children) = &borrowed.tree else {
            panic!("expected internal root");
        };
        assert_eq!((children[0].arity(), children[1].arity()), (128, 128));
        assert!(children[0].maximum_key() < children[1].minimum_key());
    }

    #[test]
    fn internal_overflow_uses_the_same_128_129_split_rule() {
        let snapshot = IndexSnapshot::empty()
            .unwrap()
            .apply(&add_range(0, 32_897), MutationMode::Normal)
            .unwrap();
        assert_eq!(snapshot.root.height, 3);
        let Node::Internal(children) = &snapshot.tree else {
            panic!("expected height-three root");
        };
        assert_eq!((children[0].arity(), children[1].arity()), (128, 129));
    }

    #[test]
    fn right_borrow_then_merge_and_root_collapse_follow_frozen_rules() {
        let snapshot = IndexSnapshot::empty()
            .unwrap()
            .apply(&add_range(0, 257), MutationMode::Normal)
            .unwrap();
        let first_key = key(0);
        let removal = IndexMutationSetV1::new(
            vec![IndexRemoval {
                key: first_key.clone(),
                expected_descriptor_hash: snapshot.get(&first_key).unwrap().descriptor_hash,
            }],
            vec![],
        )
        .unwrap();
        let borrowed = snapshot.apply(&removal, MutationMode::Normal).unwrap();
        let Node::Internal(children) = &borrowed.tree else {
            panic!("expected internal root");
        };
        assert_eq!((children[0].arity(), children[1].arity()), (128, 128));

        let next_key = key(1);
        let collapse = IndexMutationSetV1::new(
            vec![IndexRemoval {
                key: next_key.clone(),
                expected_descriptor_hash: borrowed.get(&next_key).unwrap().descriptor_hash,
            }],
            vec![],
        )
        .unwrap();
        let collapsed = borrowed.apply(&collapse, MutationMode::Normal).unwrap();
        assert_eq!(collapsed.root.height, 1);
        assert_eq!(collapsed.root.logical_entry_count, 255);
    }

    #[test]
    fn mutation_permutations_have_identical_pages_and_root() {
        let parent = IndexSnapshot::empty()
            .unwrap()
            .apply(&add_range(0, 300), MutationMode::Normal)
            .unwrap();
        let remove_keys = [3, 4, 129, 130];
        let mut removals: Vec<_> = remove_keys
            .into_iter()
            .map(|number| IndexRemoval {
                key: key(number),
                expected_descriptor_hash: parent.get(&key(number)).unwrap().descriptor_hash,
            })
            .collect();
        let mut additions = vec![entry(1_000, 1), entry(1_001, 2), entry(1_002, 3)];
        let forward = IndexMutationSetV1::new(removals.clone(), additions.clone()).unwrap();
        removals.reverse();
        additions.reverse();
        let reverse = IndexMutationSetV1::new(removals, additions).unwrap();
        let left = parent.apply(&forward, MutationMode::Normal).unwrap();
        let right = parent.apply(&reverse, MutationMode::Normal).unwrap();
        assert_eq!(left.root, right.root);
        assert_eq!(left.pages, right.pages);
    }

    #[test]
    fn recovery_only_allows_exact_idempotent_addition() {
        let snapshot = IndexSnapshot::empty()
            .unwrap()
            .apply(
                &IndexMutationSetV1::new(vec![], vec![entry(1, 1)]).unwrap(),
                MutationMode::Normal,
            )
            .unwrap();
        let exact = IndexMutationSetV1::new(vec![], vec![entry(1, 1)]).unwrap();
        assert!(matches!(
            snapshot.apply(&exact, MutationMode::Normal),
            Err(IndexError::AdditionCollision(_))
        ));
        assert_eq!(
            snapshot.apply(&exact, MutationMode::Recovery).unwrap().root,
            snapshot.root
        );
        let unequal = IndexMutationSetV1::new(vec![], vec![entry(1, 2)]).unwrap();
        assert!(matches!(
            snapshot.apply(&unequal, MutationMode::Recovery),
            Err(IndexError::AdditionCollision(_))
        ));
    }

    #[test]
    fn persisted_pages_round_trip_and_corruption_fails_closed() {
        let snapshot = IndexSnapshot::empty()
            .unwrap()
            .apply(&add_range(0, 600), MutationMode::Normal)
            .unwrap();
        let mut store = MemoryIndexPageStore::default();
        snapshot.persist(&mut store).unwrap();
        let loaded = IndexSnapshot::load(snapshot.root.clone(), &store).unwrap();
        assert_eq!(loaded.root, snapshot.root);
        assert_eq!(loaded.entries().count(), 600);

        let root_hash = snapshot.root.root_hash;
        let mut corrupt = store.get(root_hash).unwrap();
        corrupt[0] ^= 1;
        store.replace_for_test(root_hash, corrupt);
        assert!(matches!(
            IndexSnapshot::load(snapshot.root.clone(), &store),
            Err(IndexError::PageHashMismatch(hash)) if hash == root_hash
        ));
    }

    #[test]
    fn authenticated_summaries_bound_source_table_time_page_reads() {
        let mut additions = Vec::new();
        for clock in 0..300_i64 {
            additions.push(primary_entry(
                u64::try_from(clock).unwrap(),
                TableId::Families,
                Some("source-a"),
                clock,
                clock,
            ));
        }
        for offset in 0..300_i64 {
            additions.push(primary_entry(
                1_000 + u64::try_from(offset).unwrap(),
                TableId::Samples,
                Some("source-b"),
                10_000 + offset,
                10_000 + offset,
            ));
        }
        let snapshot = IndexSnapshot::empty()
            .unwrap()
            .apply(
                &IndexMutationSetV1::new(Vec::new(), additions).unwrap(),
                MutationMode::Normal,
            )
            .unwrap();
        assert_eq!(
            snapshot.root.pruning_summary.source_ids(),
            ["source-a", "source-b"]
        );
        assert_eq!(
            snapshot
                .root
                .pruning_summary
                .table_clock_range(TableId::Families),
            Some(IndexClockRangeV1 {
                minimum_clock_ns: 0,
                maximum_clock_ns: 299,
            })
        );
        let predicate = IndexScanPredicateV1::table_partitions(
            BTreeSet::from([TableId::Families]),
            Some(session()),
            IndexSourceSelectionV1::Exact("source-a".to_owned()),
            Some(10),
            Some(10),
        )
        .unwrap();
        let scan = snapshot.scan(&predicate, 2, 128).unwrap();
        assert_eq!(scan.entries().len(), 1);
        assert_eq!(
            scan.stats(),
            IndexScanStatsV1 {
                pages_read: 2,
                child_pages_pruned: 3,
                entries_examined: 128,
            }
        );
        assert!(matches!(
            snapshot.scan(&predicate, 1, 128),
            Err(IndexError::PageWorkBoundExceeded(1))
        ));
        assert!(matches!(
            snapshot.scan(&predicate, 2, 127),
            Err(IndexError::EntryWorkBoundExceeded(127))
        ));
    }

    #[test]
    fn forged_internal_pruning_summary_fails_reload_even_under_a_new_root_hash() {
        let additions = (0..300_i64)
            .map(|clock| {
                primary_entry(
                    u64::try_from(clock).unwrap(),
                    TableId::Families,
                    Some("source-a"),
                    clock,
                    clock,
                )
            })
            .collect();
        let snapshot = IndexSnapshot::empty()
            .unwrap()
            .apply(
                &IndexMutationSetV1::new(Vec::new(), additions).unwrap(),
                MutationMode::Normal,
            )
            .unwrap();
        let mut store = MemoryIndexPageStore::default();
        snapshot.persist(&mut store).unwrap();
        let mut forged =
            CanonicalJsonValue::parse_canonical(&store.get(snapshot.root.root_hash).unwrap())
                .unwrap();
        let CanonicalJsonValue::Object(root_fields) = &mut forged else {
            panic!("root page must be an object")
        };
        let Some(CanonicalJsonValue::Array(children)) = root_fields.get_mut("children") else {
            panic!("root page must be internal")
        };
        let CanonicalJsonValue::Object(first_child) = &mut children[0] else {
            panic!("child reference must be an object")
        };
        let Some(CanonicalJsonValue::Object(summary)) = first_child.get_mut("pruning_summary")
        else {
            panic!("child reference must carry a pruning summary")
        };
        summary.insert(
            "source_ids".to_owned(),
            CanonicalJsonValue::Array(vec![string("forged-source")]),
        );
        let forged_bytes = forged.to_bytes();
        let forged_hash = page_hash(&forged_bytes);
        store.put_if_absent(forged_hash, &forged_bytes).unwrap();
        let mut forged_root = snapshot.root.clone();
        forged_root.root_hash = forged_hash;
        forged_root.root_byte_length = u64::try_from(forged_bytes.len()).unwrap();
        assert!(matches!(
            IndexSnapshot::load(forged_root, &store),
            Err(IndexError::InvalidPage("child summary"))
        ));
    }

    #[test]
    fn replacement_is_closed_and_nonce_reservations_are_append_only() {
        let coverage_key = CompositeIndexKeyV1::projection_coverage(
            TableId::Samples,
            session(),
            Some("source"),
            10,
            Digest::from_bytes([1; 32]),
        )
        .unwrap();
        let old = IndexEntry::new(coverage_key.clone(), b"{\"v\":1}".to_vec()).unwrap();
        let parent = IndexSnapshot::empty()
            .unwrap()
            .apply(
                &IndexMutationSetV1::new(vec![], vec![old.clone()]).unwrap(),
                MutationMode::Normal,
            )
            .unwrap();
        let replacement = IndexEntry::new(coverage_key.clone(), b"{\"v\":2}".to_vec()).unwrap();
        let mutation = IndexMutationSetV1::new(
            vec![IndexRemoval {
                key: coverage_key,
                expected_descriptor_hash: old.descriptor_hash,
            }],
            vec![replacement],
        )
        .unwrap();
        assert!(parent.apply(&mutation, MutationMode::Normal).is_ok());

        let nonce_key = CompositeIndexKeyV1::raw_nonce_reservation("key-a", &[7; 12]).unwrap();
        let nonce_old = IndexEntry::new(nonce_key.clone(), b"{\"v\":1}".to_vec()).unwrap();
        let nonce_parent = IndexSnapshot::empty()
            .unwrap()
            .apply(
                &IndexMutationSetV1::new(vec![], vec![nonce_old.clone()]).unwrap(),
                MutationMode::Normal,
            )
            .unwrap();
        let nonce_mutation = IndexMutationSetV1::new(
            vec![IndexRemoval {
                key: nonce_key.clone(),
                expected_descriptor_hash: nonce_old.descriptor_hash,
            }],
            vec![IndexEntry::new(nonce_key, b"{\"v\":2}".to_vec()).unwrap()],
        )
        .unwrap();
        assert!(matches!(
            nonce_parent.apply(&nonce_mutation, MutationMode::Normal),
            Err(IndexError::ReplacementForbidden(_))
        ));
    }
}
