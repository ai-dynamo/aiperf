// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Verified head/root partition discovery, Parquet reads, and compaction proofs.
//!
//! Query authority is an immutable generation plus its exact persistent index
//! root. Filesystem or object-store enumeration is deliberately absent from
//! every trait in this module.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::{self, Debug, Display, Formatter};

use arrow_array::RecordBatch;
use bytes::Bytes;
use parquet::arrow::arrow_reader::{ArrowReaderOptions, ParquetRecordBatchReaderBuilder};

use crate::parquet::{PartitionDescriptorV1, PartitionProjectionEvidenceV1};
use crate::{
    ArchiveId, ArchiveSchemasV1, Digest, GenerationObjectV1, HeadDescriptorV1, IndexObjectKind,
    IndexSnapshot, SchemaError, SessionId, TableId, domain_digest,
};

/// Source selection over the explicit global/source partition key.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SourcePredicateV1 {
    /// Any source or global partition.
    Any,
    /// Exactly one non-empty source ID.
    Exact(String),
    /// Only the explicit global/no-source sentinel.
    Global,
}

/// Bounded manifest-index partition predicate.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PartitionPredicateV1 {
    /// Empty means every primary table.
    pub tables: BTreeSet<TableId>,
    /// Optional exact collection session.
    pub session_id: Option<SessionId>,
    /// Source/global selection.
    pub source: SourcePredicateV1,
    /// Inclusive minimum authoritative Clock.
    pub minimum_clock_ns: Option<i64>,
    /// Inclusive maximum authoritative Clock.
    pub maximum_clock_ns: Option<i64>,
    /// Hard ceiling on index entries examined by this resolver invocation.
    pub max_index_entries: u64,
    /// Hard ceiling on returned partition descriptors.
    pub max_partitions: usize,
}

impl Default for PartitionPredicateV1 {
    fn default() -> Self {
        Self {
            tables: BTreeSet::new(),
            session_id: None,
            source: SourcePredicateV1::Any,
            minimum_clock_ns: None,
            maximum_clock_ns: None,
            max_index_entries: 1_000_000,
            max_partitions: 100_000,
        }
    }
}

impl PartitionPredicateV1 {
    /// Validates explicit query work bounds and range ordering.
    pub fn validate(&self) -> Result<(), QueryError> {
        if self.max_index_entries == 0 || self.max_partitions == 0 {
            return Err(QueryError::ZeroWorkBound);
        }
        if self
            .minimum_clock_ns
            .zip(self.maximum_clock_ns)
            .is_some_and(|(minimum, maximum)| minimum > maximum)
        {
            return Err(QueryError::ReversedClockRange);
        }
        if matches!(&self.source, SourcePredicateV1::Exact(source) if source.is_empty()) {
            return Err(QueryError::EmptySourceId);
        }
        Ok(())
    }
}

/// Bounded result and observable index-work count.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PartitionDiscoveryV1 {
    /// Verified matching descriptors in composite-index order.
    pub partitions: Vec<PartitionDescriptorV1>,
    /// Number of root-reachable entries examined; never a directory count.
    pub index_entries_examined: u64,
}

/// Query resolver bound to one verified immutable head/generation/root triple.
#[derive(Debug)]
pub struct VerifiedQueryRootV1<'a> {
    head: &'a HeadDescriptorV1,
    generation: &'a GenerationObjectV1,
    index: &'a IndexSnapshot,
    schemas: &'a ArchiveSchemasV1,
}

impl<'a> VerifiedQueryRootV1<'a> {
    /// Verifies head, generation envelope, root, archive, and schema authority.
    pub fn new(
        head: &'a HeadDescriptorV1,
        generation: &'a GenerationObjectV1,
        index: &'a IndexSnapshot,
        schemas: &'a ArchiveSchemasV1,
    ) -> Result<Self, QueryError> {
        let decoded =
            GenerationObjectV1::decode(&generation.bytes).map_err(QueryError::Manifest)?;
        if &decoded != generation {
            return Err(QueryError::GenerationEnvelopeMismatch);
        }
        let expected_head =
            HeadDescriptorV1::from_generation(generation).map_err(QueryError::Manifest)?;
        if &expected_head != head {
            return Err(QueryError::HeadGenerationMismatch);
        }
        if index.root() != &generation.generation.index_root
            || head.index_root_hash != index.root().root_hash
        {
            return Err(QueryError::IndexRootMismatch);
        }
        if schemas.iter().count() != 6 {
            return Err(QueryError::IncompleteSchemaRegistry);
        }
        Ok(Self {
            head,
            generation,
            index,
            schemas,
        })
    }

    /// Returns the authoritative immutable generation.
    #[must_use]
    pub const fn generation(&self) -> &GenerationObjectV1 {
        self.generation
    }

    /// Walks only root-reachable index entries and applies source/time pruning.
    pub fn discover(
        &self,
        predicate: &PartitionPredicateV1,
    ) -> Result<PartitionDiscoveryV1, QueryError> {
        predicate.validate()?;
        let mut examined = 0_u64;
        let mut partitions = Vec::new();
        for entry in self.index.entries() {
            examined = examined.checked_add(1).ok_or(QueryError::LengthOverflow)?;
            if examined > predicate.max_index_entries {
                return Err(QueryError::IndexWorkBoundExceeded {
                    bound: predicate.max_index_entries,
                });
            }
            if entry.key().as_bytes().first().copied()
                != Some(IndexObjectKind::TablePartition as u8)
            {
                continue;
            }
            let descriptor = PartitionDescriptorV1::from_canonical_bytes(entry.descriptor_bytes())
                .map_err(QueryError::Partition)?;
            self.verify_partition_entry(entry.key(), &descriptor)?;
            if !matches_predicate(&descriptor, predicate) {
                continue;
            }
            if partitions.len() >= predicate.max_partitions {
                return Err(QueryError::PartitionBoundExceeded {
                    bound: predicate.max_partitions,
                });
            }
            partitions.push(descriptor);
        }
        Ok(PartitionDiscoveryV1 {
            partitions,
            index_entries_examined: examined,
        })
    }

    fn verify_partition_entry(
        &self,
        index_key: &crate::IndexKey,
        descriptor: &PartitionDescriptorV1,
    ) -> Result<(), QueryError> {
        if descriptor.archive_id != self.head.archive_id
            || descriptor.archive_id != self.generation.generation.archive_id
        {
            return Err(QueryError::PartitionArchiveMismatch);
        }
        let schema = self.schemas.table(descriptor.table)?;
        if descriptor.schema_fingerprint != schema.fingerprint() {
            return Err(QueryError::PartitionSchemaMismatch(descriptor.table));
        }
        if &descriptor.index_key().map_err(QueryError::Index)? != index_key {
            return Err(QueryError::PartitionIndexKeyMismatch);
        }
        Ok(())
    }
}

fn matches_predicate(descriptor: &PartitionDescriptorV1, predicate: &PartitionPredicateV1) -> bool {
    if !predicate.tables.is_empty() && !predicate.tables.contains(&descriptor.table) {
        return false;
    }
    if predicate
        .session_id
        .is_some_and(|session| session != descriptor.session_id)
    {
        return false;
    }
    match &predicate.source {
        SourcePredicateV1::Any => {}
        SourcePredicateV1::Exact(source)
            if descriptor.source_id.as_deref() != Some(source.as_str()) =>
        {
            return false;
        }
        SourcePredicateV1::Global if descriptor.source_id.is_some() => return false,
        SourcePredicateV1::Exact(_) | SourcePredicateV1::Global => {}
    }
    if predicate
        .minimum_clock_ns
        .is_some_and(|minimum| descriptor.maximum_clock_ns < minimum)
    {
        return false;
    }
    if predicate
        .maximum_clock_ns
        .is_some_and(|maximum| descriptor.minimum_clock_ns > maximum)
    {
        return false;
    }
    true
}

/// Exact-key immutable partition object reads; listing is intentionally absent.
pub trait PartitionObjectSourceV1: Debug {
    /// Reads one exact object key selected from a verified partition descriptor.
    fn get_exact(&self, key: &str) -> Result<Vec<u8>, PartitionObjectReadError>;
}

/// Deterministic in-memory exact-key object source for tests and replay.
#[derive(Clone, Debug, Default)]
pub struct MemoryPartitionObjectSourceV1 {
    objects: BTreeMap<String, Vec<u8>>,
}

impl MemoryPartitionObjectSourceV1 {
    /// Installs one immutable object or verifies an identical retry.
    pub fn put(
        &mut self,
        descriptor: &PartitionDescriptorV1,
        bytes: &[u8],
    ) -> Result<(), QueryError> {
        verify_partition_bytes(descriptor, bytes)?;
        match self.objects.get(&descriptor.physical_object_key) {
            Some(existing) if existing == bytes => Ok(()),
            Some(_) => Err(QueryError::ObjectCollision(
                descriptor.physical_object_key.clone(),
            )),
            None => {
                self.objects
                    .insert(descriptor.physical_object_key.clone(), bytes.to_vec());
                Ok(())
            }
        }
    }

    /// Returns the number of exact-key objects.
    #[must_use]
    pub fn len(&self) -> usize {
        self.objects.len()
    }

    /// Whether no partition objects are installed.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.objects.is_empty()
    }
}

impl PartitionObjectSourceV1 for MemoryPartitionObjectSourceV1 {
    fn get_exact(&self, key: &str) -> Result<Vec<u8>, PartitionObjectReadError> {
        self.objects
            .get(key)
            .cloned()
            .ok_or_else(|| PartitionObjectReadError::Missing(key.to_string()))
    }
}

/// Reads one verified immutable partition using its exact descriptor schema.
pub fn read_partition_v1(
    source: &dyn PartitionObjectSourceV1,
    descriptor: &PartitionDescriptorV1,
    schemas: &ArchiveSchemasV1,
) -> Result<Vec<RecordBatch>, QueryError> {
    descriptor.validate().map_err(QueryError::Partition)?;
    let bytes = source
        .get_exact(&descriptor.physical_object_key)
        .map_err(QueryError::ObjectRead)?;
    verify_partition_bytes(descriptor, &bytes)?;
    let table_schema = schemas.table(descriptor.table)?;
    if table_schema.fingerprint() != descriptor.schema_fingerprint {
        return Err(QueryError::PartitionSchemaMismatch(descriptor.table));
    }
    let options = ArrowReaderOptions::new().with_schema(table_schema.schema().clone());
    let builder =
        ParquetRecordBatchReaderBuilder::try_new_with_options(Bytes::from(bytes), options)
            .map_err(QueryError::Parquet)?;
    if builder.schema().as_ref() != table_schema.schema().as_ref() {
        return Err(QueryError::ReaderSchemaMismatch(descriptor.table));
    }
    let reader = builder.build().map_err(QueryError::Parquet)?;
    let mut batches = Vec::new();
    let mut rows = 0_u64;
    for batch in reader {
        let batch = batch.map_err(QueryError::Arrow)?;
        if batch.schema().as_ref() != table_schema.schema().as_ref() {
            return Err(QueryError::ReaderSchemaMismatch(descriptor.table));
        }
        rows = rows
            .checked_add(u64::try_from(batch.num_rows()).map_err(|_| QueryError::LengthOverflow)?)
            .ok_or(QueryError::LengthOverflow)?;
        batches.push(batch);
    }
    if rows != descriptor.row_count {
        return Err(QueryError::ReaderRowCount {
            expected: descriptor.row_count,
            actual: rows,
        });
    }
    Ok(batches)
}

fn verify_partition_bytes(
    descriptor: &PartitionDescriptorV1,
    bytes: &[u8],
) -> Result<(), QueryError> {
    let length = u64::try_from(bytes.len()).map_err(|_| QueryError::LengthOverflow)?;
    if length != descriptor.physical_byte_length {
        return Err(QueryError::PartitionByteLength {
            expected: descriptor.physical_byte_length,
            actual: length,
        });
    }
    let digest = domain_digest("aiperf.archive.partition.v1", &[bytes]);
    if digest != descriptor.physical_content_hash {
        return Err(QueryError::PartitionContentHash);
    }
    Ok(())
}

/// Verified per-projection logical equality for one bounded compaction.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CompactionLogicalProofV1 {
    /// Number of homogeneous archive/session/table/source groups.
    pub group_count: usize,
    /// Number of exact frame/table projection facts.
    pub projection_count: usize,
    /// Equal logical rows on each side.
    pub logical_row_count: u64,
}

/// Proves compaction equality independently of Parquet bytes and row order.
pub fn verify_compaction_logical_equality_v1(
    inputs: &[PartitionDescriptorV1],
    outputs: &[PartitionDescriptorV1],
) -> Result<CompactionLogicalProofV1, QueryError> {
    if inputs.is_empty() || outputs.is_empty() {
        return Err(QueryError::EmptyCompactionSide);
    }
    let input = compaction_groups(inputs)?;
    let output = compaction_groups(outputs)?;
    if input != output {
        return Err(QueryError::CompactionLogicalMismatch);
    }
    let projection_count = input.values().map(|group| group.projections.len()).sum();
    let logical_row_count = input.values().try_fold(0_u64, |total, group| {
        total
            .checked_add(group.row_count)
            .ok_or(QueryError::LengthOverflow)
    })?;
    Ok(CompactionLogicalProofV1 {
        group_count: input.len(),
        projection_count,
        logical_row_count,
    })
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct CompactionGroupKey {
    archive_id: ArchiveId,
    session_id: SessionId,
    source_id: Option<String>,
    table: TableId,
    schema_fingerprint: Digest,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct CompactionGroupEvidence {
    row_count: u64,
    minimum_clock_ns: i64,
    maximum_clock_ns: i64,
    projections: BTreeMap<FrameIdKey, (u64, Digest)>,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct FrameIdKey(Digest);

fn compaction_groups(
    partitions: &[PartitionDescriptorV1],
) -> Result<BTreeMap<CompactionGroupKey, CompactionGroupEvidence>, QueryError> {
    let mut physical_ids = BTreeSet::new();
    let mut groups = BTreeMap::<CompactionGroupKey, CompactionGroupEvidence>::new();
    for partition in partitions {
        partition.validate().map_err(QueryError::Partition)?;
        if !physical_ids.insert(partition.physical_content_hash) {
            return Err(QueryError::DuplicateCompactionPartition(
                partition.physical_content_hash,
            ));
        }
        let key = CompactionGroupKey {
            archive_id: partition.archive_id,
            session_id: partition.session_id,
            source_id: partition.source_id.clone(),
            table: partition.table,
            schema_fingerprint: partition.schema_fingerprint,
        };
        let group = groups.entry(key).or_insert(CompactionGroupEvidence {
            row_count: 0,
            minimum_clock_ns: partition.minimum_clock_ns,
            maximum_clock_ns: partition.maximum_clock_ns,
            projections: BTreeMap::new(),
        });
        group.row_count = group
            .row_count
            .checked_add(partition.row_count)
            .ok_or(QueryError::LengthOverflow)?;
        group.minimum_clock_ns = group.minimum_clock_ns.min(partition.minimum_clock_ns);
        group.maximum_clock_ns = group.maximum_clock_ns.max(partition.maximum_clock_ns);
        for projection in &partition.projections {
            insert_projection(group, projection)?;
        }
    }
    Ok(groups)
}

fn insert_projection(
    group: &mut CompactionGroupEvidence,
    projection: &PartitionProjectionEvidenceV1,
) -> Result<(), QueryError> {
    if group
        .projections
        .insert(
            FrameIdKey(projection.frame_id.digest()),
            (projection.row_count, projection.logical_multiset_digest),
        )
        .is_some()
    {
        return Err(QueryError::SplitFrameProjection(
            projection.frame_id.digest(),
        ));
    }
    Ok(())
}

/// Exact-key partition object failure.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PartitionObjectReadError {
    /// Exact immutable object is absent.
    Missing(String),
    /// Backing provider failed without returning bytes.
    Provider(String),
}

impl Display for PartitionObjectReadError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Missing(key) => write!(formatter, "missing partition object {key:?}"),
            Self::Provider(message) => write!(formatter, "partition provider failed: {message}"),
        }
    }
}

impl std::error::Error for PartitionObjectReadError {}

/// Invalid query authority, discovery, object, reader, or compaction proof.
#[derive(Debug)]
pub enum QueryError {
    /// Manifest generation/head validation failed.
    Manifest(crate::ManifestError),
    /// Index key construction failed.
    Index(crate::IndexError),
    /// Schema registry failed.
    Schema(SchemaError),
    /// Partition descriptor failed.
    Partition(crate::parquet::ParquetProjectionError),
    /// Exact-key object read failed.
    ObjectRead(PartitionObjectReadError),
    /// Parquet metadata/reader construction failed.
    Parquet(parquet::errors::ParquetError),
    /// Arrow batch read failed.
    Arrow(arrow_schema::ArrowError),
    /// Re-decoded generation differs from supplied object.
    GenerationEnvelopeMismatch,
    /// Head does not exactly describe the generation.
    HeadGenerationMismatch,
    /// Generation/head root does not equal the supplied verified index.
    IndexRootMismatch,
    /// All six schema authorities were not present.
    IncompleteSchemaRegistry,
    /// Query work bound was zero.
    ZeroWorkBound,
    /// Clock predicate minimum exceeds maximum.
    ReversedClockRange,
    /// Exact source predicate was empty.
    EmptySourceId,
    /// Index entry scan reached its explicit ceiling.
    IndexWorkBoundExceeded {
        /// Configured entry ceiling.
        bound: u64,
    },
    /// Matching partitions reached their explicit ceiling.
    PartitionBoundExceeded {
        /// Configured result ceiling.
        bound: usize,
    },
    /// Descriptor archive does not equal the verified head.
    PartitionArchiveMismatch,
    /// Descriptor fingerprint does not equal the selected table schema.
    PartitionSchemaMismatch(TableId),
    /// Descriptor-derived composite key differs from its index key.
    PartitionIndexKeyMismatch,
    /// Exact-key create retry collided with unequal bytes.
    ObjectCollision(String),
    /// Physical object length differs from its descriptor.
    PartitionByteLength {
        /// Descriptor length.
        expected: u64,
        /// Actual bytes.
        actual: u64,
    },
    /// Physical object's exact-byte hash differs.
    PartitionContentHash,
    /// Reader schema differs from the fingerprint-selected schema.
    ReaderSchemaMismatch(TableId),
    /// Reader row total differs from the descriptor.
    ReaderRowCount {
        /// Descriptor rows.
        expected: u64,
        /// Decoded rows.
        actual: u64,
    },
    /// Compaction input or output set was empty.
    EmptyCompactionSide,
    /// One physical partition occurred twice on one compaction side.
    DuplicateCompactionPartition(Digest),
    /// One frame/table projection was split across physical descriptors.
    SplitFrameProjection(Digest),
    /// Input/output logical projection multisets differ.
    CompactionLogicalMismatch,
    /// Count or work accounting overflowed.
    LengthOverflow,
}

impl From<SchemaError> for QueryError {
    fn from(value: SchemaError) -> Self {
        Self::Schema(value)
    }
}

impl Display for QueryError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Manifest(error) => write!(formatter, "query manifest failed: {error}"),
            Self::Index(error) => write!(formatter, "query index failed: {error}"),
            Self::Schema(error) => write!(formatter, "query schema failed: {error}"),
            Self::Partition(error) => write!(formatter, "query partition failed: {error}"),
            Self::ObjectRead(error) => write!(formatter, "query object read failed: {error}"),
            Self::Parquet(error) => write!(formatter, "query Parquet failed: {error}"),
            Self::Arrow(error) => write!(formatter, "query Arrow failed: {error}"),
            Self::GenerationEnvelopeMismatch => formatter.write_str("generation envelope mismatch"),
            Self::HeadGenerationMismatch => formatter.write_str("head/generation mismatch"),
            Self::IndexRootMismatch => formatter.write_str("head/generation/index root mismatch"),
            Self::IncompleteSchemaRegistry => {
                formatter.write_str("query schema registry is incomplete")
            }
            Self::ZeroWorkBound => formatter.write_str("query work bounds must be positive"),
            Self::ReversedClockRange => formatter.write_str("query Clock range is reversed"),
            Self::EmptySourceId => formatter.write_str("query source ID cannot be empty"),
            Self::IndexWorkBoundExceeded { bound } => {
                write!(formatter, "query exceeded {bound} index entries")
            }
            Self::PartitionBoundExceeded { bound } => {
                write!(formatter, "query exceeded {bound} partitions")
            }
            Self::PartitionArchiveMismatch => {
                formatter.write_str("partition archive does not match head")
            }
            Self::PartitionSchemaMismatch(table) => {
                write!(formatter, "partition schema mismatch for {table:?}")
            }
            Self::PartitionIndexKeyMismatch => formatter.write_str("partition/index key mismatch"),
            Self::ObjectCollision(key) => {
                write!(formatter, "partition object collision at {key:?}")
            }
            Self::PartitionByteLength { expected, actual } => write!(
                formatter,
                "partition length mismatch: expected {expected}, found {actual}"
            ),
            Self::PartitionContentHash => formatter.write_str("partition content hash mismatch"),
            Self::ReaderSchemaMismatch(table) => {
                write!(formatter, "reader schema mismatch for {table:?}")
            }
            Self::ReaderRowCount { expected, actual } => write!(
                formatter,
                "reader row count mismatch: expected {expected}, found {actual}"
            ),
            Self::EmptyCompactionSide => {
                formatter.write_str("compaction input/output cannot be empty")
            }
            Self::DuplicateCompactionPartition(id) => {
                write!(formatter, "duplicate compaction partition {id}")
            }
            Self::SplitFrameProjection(id) => write!(formatter, "split frame projection {id}"),
            Self::CompactionLogicalMismatch => {
                formatter.write_str("compaction logical multiset mismatch")
            }
            Self::LengthOverflow => formatter.write_str("query count or length overflow"),
        }
    }
}

impl std::error::Error for QueryError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ArchiveState, CanonicalJsonValue, EpochAnchor, GenerationTransactionKind, GenerationV1,
        GenesisV1, IndexMutationSetV1, MutationMode, TimeDomain,
    };

    fn archive() -> ArchiveId {
        ArchiveId::new([0x11; 16]).unwrap()
    }

    fn session() -> SessionId {
        SessionId::new([0x22; 16]).unwrap()
    }

    #[test]
    fn verified_root_discovery_filters_source_and_time_without_listing() {
        let schemas = ArchiveSchemasV1::load().unwrap();
        let descriptors = [
            descriptor("source-a", 10, 20, 1, TableId::Families, &schemas),
            descriptor("source-b", 100, 120, 2, TableId::Families, &schemas),
            descriptor("source-a", 15, 15, 3, TableId::Samples, &schemas),
        ];
        let additions = descriptors
            .iter()
            .map(|descriptor| descriptor.index_entry().unwrap())
            .collect();
        let index = IndexSnapshot::empty()
            .unwrap()
            .apply(
                &IndexMutationSetV1::new(Vec::new(), additions).unwrap(),
                MutationMode::Normal,
            )
            .unwrap();
        let generation = GenerationObjectV1::new(genesis(index.root().clone())).unwrap();
        let head = HeadDescriptorV1::from_generation(&generation).unwrap();
        let resolver = VerifiedQueryRootV1::new(&head, &generation, &index, &schemas).unwrap();
        let result = resolver
            .discover(&PartitionPredicateV1 {
                tables: BTreeSet::from([TableId::Families]),
                session_id: Some(session()),
                source: SourcePredicateV1::Exact("source-a".to_string()),
                minimum_clock_ns: Some(15),
                maximum_clock_ns: Some(25),
                max_index_entries: 10,
                max_partitions: 10,
            })
            .unwrap();
        assert_eq!(result.partitions, vec![descriptors[0].clone()]);
        assert_eq!(result.index_entries_examined, 3);
    }

    #[test]
    fn discovery_fails_closed_at_explicit_index_and_result_bounds() {
        let schemas = ArchiveSchemasV1::load().unwrap();
        let partition = descriptor("source-a", 10, 20, 1, TableId::Families, &schemas);
        let entry = partition.index_entry().unwrap();
        let index = IndexSnapshot::empty()
            .unwrap()
            .apply(
                &IndexMutationSetV1::new(Vec::new(), vec![entry]).unwrap(),
                MutationMode::Normal,
            )
            .unwrap();
        let generation = GenerationObjectV1::new(genesis(index.root().clone())).unwrap();
        let head = HeadDescriptorV1::from_generation(&generation).unwrap();
        let resolver = VerifiedQueryRootV1::new(&head, &generation, &index, &schemas).unwrap();
        let error = resolver
            .discover(&PartitionPredicateV1 {
                max_index_entries: 1,
                max_partitions: 0,
                ..PartitionPredicateV1::default()
            })
            .unwrap_err();
        assert!(matches!(error, QueryError::ZeroWorkBound));
    }

    #[test]
    fn compaction_requires_exact_per_frame_logical_multiset_equality() {
        let schemas = ArchiveSchemasV1::load().unwrap();
        let first = descriptor("source-a", 10, 10, 1, TableId::Families, &schemas);
        let second = descriptor("source-a", 20, 20, 2, TableId::Families, &schemas);
        let mut compacted = descriptor("source-a", 10, 20, 9, TableId::Families, &schemas);
        compacted.row_count = first.row_count + second.row_count;
        compacted.projections = first
            .projections
            .iter()
            .chain(&second.projections)
            .cloned()
            .collect();
        refresh_logical_object_id(&mut compacted);
        let proof = verify_compaction_logical_equality_v1(
            &[first.clone(), second.clone()],
            &[compacted.clone()],
        )
        .unwrap();
        assert_eq!(proof.projection_count, 2);
        compacted.projections[0].logical_multiset_digest = Digest::from_bytes([0xee; 32]);
        refresh_logical_object_id(&mut compacted);
        assert!(matches!(
            verify_compaction_logical_equality_v1(&[first, second], &[compacted]),
            Err(QueryError::CompactionLogicalMismatch)
        ));
    }

    fn refresh_logical_object_id(descriptor: &mut PartitionDescriptorV1) {
        descriptor.logical_object_id = crate::parquet::partition_logical_object_id_v1(
            descriptor.table,
            descriptor.schema_fingerprint,
            descriptor.physical_content_hash,
            &descriptor.projections,
        );
    }

    fn descriptor(
        source: &str,
        minimum_clock_ns: i64,
        maximum_clock_ns: i64,
        seed: u8,
        table: TableId,
        schemas: &ArchiveSchemasV1,
    ) -> PartitionDescriptorV1 {
        let content = Digest::from_bytes([seed; 32]);
        let frame = Digest::from_bytes([seed.saturating_add(20); 32]);
        let projections = vec![PartitionProjectionEvidenceV1 {
            frame_id: crate::FrameId::new(frame),
            row_count: 1,
            logical_multiset_digest: Digest::from_bytes([seed.saturating_add(60); 32]),
        }];
        let schema_fingerprint = schemas.table(table).unwrap().fingerprint();
        let logical = crate::parquet::partition_logical_object_id_v1(
            table,
            schema_fingerprint,
            content,
            &projections,
        );
        PartitionDescriptorV1 {
            archive_id: archive(),
            session_id: session(),
            source_id: Some(source.to_string()),
            table,
            time_bucket: minimum_clock_ns.div_euclid(100),
            schema_fingerprint,
            physical_content_hash: content,
            physical_object_key: format!(
                "partitions/{}/part-{}.parquet",
                crate::table_name(table),
                content.to_hex()
            ),
            physical_byte_length: 100,
            row_count: 1,
            minimum_clock_ns,
            maximum_clock_ns,
            logical_object_id: logical,
            projections,
        }
    }

    fn genesis(root: crate::IndexRootV1) -> GenerationV1 {
        GenerationV1 {
            archive_id: archive(),
            local_commit_seq: 0,
            parent_generation_hash: None,
            genesis_hash: None,
            index_root: root,
            archive_state: ArchiveState::Open,
            transaction_kind: GenerationTransactionKind::Genesis,
            session_id: Some(session()),
            mutations: vec![],
            genesis: Some(GenesisV1 {
                archive_id: archive(),
                canonical_spool_id: Digest::from_bytes([1; 32]),
                archive_identity_digest: Digest::from_bytes([2; 32]),
                archive_key_digest: Digest::from_bytes([3; 32]),
                writer_compatibility_id: Digest::from_bytes([4; 32]),
                runner_distribution_id: Digest::from_bytes([5; 32]),
                source_descriptors: CanonicalJsonValue::Array(vec![]),
                persistent_writer_identity: CanonicalJsonValue::object([(
                    "writer".to_string(),
                    CanonicalJsonValue::String("parquet-v1".to_string()),
                )])
                .unwrap(),
                initial_session_id: Some(session()),
                time_domain: TimeDomain::Real,
                epoch_anchor: Some(EpochAnchor {
                    clock_ns: 10,
                    unix_epoch_ns: 1_700_000_000_000_000_000,
                    capture_uncertainty_ns: 2,
                }),
            }),
            termination_reason: None,
        }
    }
}
