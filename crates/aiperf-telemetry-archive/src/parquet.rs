// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Whole-frame deterministic Parquet partition projection and rotation.
//!
//! Builders are homogeneous by archive/session/table/source/time bucket,
//! rotate only between complete frame/table projections, and emit explicit
//! coverage even when a required projection contains zero rows. Physical row
//! ordering is table-defined and never influences canonical logical evidence.

use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::{self, Display, Formatter};
use std::sync::Arc;

use arrow_array::{
    Array, FixedSizeBinaryArray, Int64Array, RecordBatch, StringArray, UInt32Array, UInt64Array,
};
use arrow_schema::{ArrowError, DataType};
use arrow_select::concat::concat_batches;
use arrow_select::take::take;
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::errors::ParquetError;
use parquet::file::properties::{WriterProperties, WriterVersion};
use serde::{Deserialize, Serialize};

use crate::{
    ArchiveId, ArchiveSchemasV1, CanonicalJsonError, CanonicalJsonValue, CanonicalLogicalRow,
    CompositeIndexKeyV1, Digest, FrameId, IndexEntry, IndexError, IndexKey, ProjectionEvidence,
    SchemaError, SessionId, TableId, domain_digest, table_name,
};

const PARTITION_DESCRIPTOR_VERSION: u8 = 1;
const PARTITION_CREATED_BY: &str = "aiperf-telemetry-archive-v1";

/// One complete frame/table projection before physical partitioning.
#[derive(Clone, Debug)]
pub struct FrameTableProjectionV1 {
    /// Archive identity shared by every row.
    pub archive_id: ArchiveId,
    /// Collection session shared by every row.
    pub session_id: SessionId,
    /// Exact physical source, or the global sentinel.
    pub source_id: Option<String>,
    /// Terminal frame identity.
    pub frame_id: FrameId,
    /// One authoritative Clock value for every timed row in the frame.
    pub authoritative_frame_clock_ns: i64,
    /// Closed table identity.
    pub table: TableId,
    /// Arrow rows under the exact table schema.
    pub batch: RecordBatch,
    /// Canonical logical rows in any order; their multiset must equal the Arrow rows.
    pub logical_rows: Vec<CanonicalLogicalRow>,
}

impl FrameTableProjectionV1 {
    /// Validates schema, identity, Clock, and logical evidence as one unit.
    pub fn validate(
        &self,
        schemas: &ArchiveSchemasV1,
    ) -> Result<ProjectionEvidence, ParquetProjectionError> {
        if self.source_id.as_deref() == Some("") {
            return Err(ParquetProjectionError::EmptySourceId);
        }
        let table_schema = schemas.table(self.table)?;
        if self.batch.schema().as_ref() != table_schema.schema().as_ref() {
            return Err(ParquetProjectionError::SchemaMismatch(self.table));
        }
        table_schema.validate_record_batch(&self.batch)?;
        if self.batch.num_rows() != self.logical_rows.len() {
            return Err(ParquetProjectionError::LogicalRowCount {
                arrow: self.batch.num_rows(),
                logical: self.logical_rows.len(),
            });
        }
        for row in &self.logical_rows {
            if row.table() != self.table {
                return Err(ParquetProjectionError::LogicalRowTableMismatch);
            }
            if row.schema_fingerprint() != table_schema.fingerprint() {
                return Err(ParquetProjectionError::LogicalRowSchemaMismatch);
            }
        }
        validate_row_identity_and_clock(self)?;
        let physical_rows = table_schema.canonical_rows(&self.batch)?;
        let physical_evidence = ProjectionEvidence::from_rows(&physical_rows)
            .map_err(ParquetProjectionError::LogicalRow)?;
        let supplied_evidence = ProjectionEvidence::from_rows(&self.logical_rows)
            .map_err(ParquetProjectionError::LogicalRow)?;
        if physical_evidence != supplied_evidence {
            return Err(ParquetProjectionError::PhysicalLogicalEvidenceMismatch);
        }
        Ok(physical_evidence)
    }
}

/// Persistent exact coverage for one required frame/table projection.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProjectionCoverageV1 {
    /// Archive identity.
    pub archive_id: ArchiveId,
    /// Collection session.
    pub session_id: SessionId,
    /// Physical source or global sentinel.
    pub source_id: Option<String>,
    /// Terminal frame identity.
    pub frame_id: FrameId,
    /// Required table.
    pub table: TableId,
    /// Equal minimum/maximum authoritative frame Clock.
    pub authoritative_frame_clock_ns: i64,
    /// Exact logical row count.
    pub row_count: u64,
    /// Exact logical multiset digest.
    pub logical_multiset_digest: Digest,
    /// Zero or one physical fragment ID in v1.
    pub fragment_ids: Vec<Digest>,
}

impl ProjectionCoverageV1 {
    /// Builds a coverage row under the no-split v1 invariant.
    pub fn new(
        projection: &FrameTableProjectionV1,
        evidence: ProjectionEvidence,
        fragment_id: Option<Digest>,
    ) -> Result<Self, ParquetProjectionError> {
        let expected_fragment = evidence.row_count > 0;
        if expected_fragment != fragment_id.is_some() {
            return Err(ParquetProjectionError::CoverageFragmentCardinality);
        }
        Ok(Self {
            archive_id: projection.archive_id,
            session_id: projection.session_id,
            source_id: projection.source_id.clone(),
            frame_id: projection.frame_id,
            table: projection.table,
            authoritative_frame_clock_ns: projection.authoritative_frame_clock_ns,
            row_count: evidence.row_count,
            logical_multiset_digest: evidence.logical_multiset_digest,
            fragment_ids: fragment_id.into_iter().collect(),
        })
    }

    /// Returns the exact primary-index key for this coverage fact.
    pub fn index_key(&self) -> Result<IndexKey, IndexError> {
        let table = [self.table as u8];
        let logical_id = domain_digest(
            "aiperf.archive.projection-coverage.v1",
            &[self.frame_id.digest().as_bytes(), &table],
        );
        CompositeIndexKeyV1::projection_coverage(
            self.table,
            self.session_id,
            self.source_id.as_deref(),
            self.authoritative_frame_clock_ns,
            logical_id,
        )
    }
}

/// One frame's logical evidence stored in an immutable partition descriptor.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PartitionProjectionEvidenceV1 {
    /// Terminal frame identity.
    pub frame_id: FrameId,
    /// Logical rows from that frame in this table.
    pub row_count: u64,
    /// Logical row-multiset digest.
    pub logical_multiset_digest: Digest,
}

/// Immutable content-addressed Parquet partition descriptor.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PartitionDescriptorV1 {
    /// Archive identity.
    pub archive_id: ArchiveId,
    /// Homogeneous collection session.
    pub session_id: SessionId,
    /// Exact physical source, or global sentinel.
    pub source_id: Option<String>,
    /// Closed table identity.
    pub table: TableId,
    /// Floor-divided Clock bucket used only for clustering.
    pub time_bucket: i64,
    /// Exact table schema fingerprint.
    pub schema_fingerprint: Digest,
    /// Content hash of exact Parquet bytes.
    pub physical_content_hash: Digest,
    /// Content-addressed physical object key.
    pub physical_object_key: String,
    /// Exact Parquet byte length.
    pub physical_byte_length: u64,
    /// Exact physical/logical row count.
    pub row_count: u64,
    /// Inclusive authoritative Clock minimum.
    pub minimum_clock_ns: i64,
    /// Inclusive authoritative Clock maximum.
    pub maximum_clock_ns: i64,
    /// Domain-separated logical partition identity used in its index key.
    pub logical_object_id: Digest,
    /// Per-frame no-split logical evidence.
    pub projections: Vec<PartitionProjectionEvidenceV1>,
}

impl PartitionDescriptorV1 {
    /// Validates immutable descriptor invariants.
    pub fn validate(&self) -> Result<(), ParquetProjectionError> {
        if self.source_id.as_deref() == Some("") {
            return Err(ParquetProjectionError::EmptySourceId);
        }
        if self.row_count == 0 || self.projections.is_empty() {
            return Err(ParquetProjectionError::EmptyPhysicalPartition);
        }
        if self.minimum_clock_ns > self.maximum_clock_ns {
            return Err(ParquetProjectionError::ClockRange);
        }
        let mut frame_ids = BTreeSet::new();
        let mut rows = 0_u64;
        for projection in &self.projections {
            if projection.row_count == 0 {
                return Err(ParquetProjectionError::ZeroRowsInPhysicalEvidence);
            }
            if !frame_ids.insert(projection.frame_id) {
                return Err(ParquetProjectionError::DuplicateFrameEvidence(
                    projection.frame_id,
                ));
            }
            rows = rows
                .checked_add(projection.row_count)
                .ok_or(ParquetProjectionError::LengthOverflow)?;
        }
        if rows != self.row_count {
            return Err(ParquetProjectionError::PartitionRowCount {
                descriptor: self.row_count,
                projections: rows,
            });
        }
        if !self.physical_object_key.ends_with(&format!(
            "part-{}.parquet",
            self.physical_content_hash.to_hex()
        )) {
            return Err(ParquetProjectionError::ObjectKeyHashMismatch);
        }
        if !self
            .physical_object_key
            .starts_with(&format!("partitions/{}/", table_name(self.table)))
        {
            return Err(ParquetProjectionError::ObjectKeyTableMismatch);
        }
        if partition_logical_object_id_v1(
            self.table,
            self.schema_fingerprint,
            self.physical_content_hash,
            &self.projections,
        ) != self.logical_object_id
        {
            return Err(ParquetProjectionError::LogicalObjectIdMismatch);
        }
        Ok(())
    }

    /// Returns the exact primary-index key for this partition.
    pub fn index_key(&self) -> Result<IndexKey, IndexError> {
        CompositeIndexKeyV1::table_partition(
            self.table,
            self.session_id,
            self.source_id.as_deref(),
            self.minimum_clock_ns,
            self.logical_object_id,
        )
    }

    /// Produces one index entry containing canonical descriptor bytes.
    pub fn index_entry(&self) -> Result<IndexEntry, ParquetProjectionError> {
        Ok(IndexEntry::new(self.index_key()?, self.canonical_bytes()?)?)
    }

    /// Encodes the immutable descriptor under canonical-json-v1.
    pub fn canonical_bytes(&self) -> Result<Vec<u8>, ParquetProjectionError> {
        self.validate()?;
        let wire = PartitionDescriptorWireV1::from(self);
        let ordinary = serde_json::to_vec(&wire).map_err(ParquetProjectionError::Json)?;
        let canonical = CanonicalJsonValue::parse(&ordinary)
            .map_err(ParquetProjectionError::Canonical)?
            .to_bytes();
        Ok(canonical)
    }

    /// Decodes and validates exact canonical descriptor bytes.
    pub fn from_canonical_bytes(bytes: &[u8]) -> Result<Self, ParquetProjectionError> {
        CanonicalJsonValue::parse_canonical(bytes).map_err(ParquetProjectionError::Canonical)?;
        let wire: PartitionDescriptorWireV1 =
            serde_json::from_slice(bytes).map_err(ParquetProjectionError::Json)?;
        let descriptor = Self::try_from(wire)?;
        descriptor.validate()?;
        Ok(descriptor)
    }
}

/// Complete immutable Parquet object plus its query/index evidence.
#[derive(Clone, Debug)]
pub struct CompletedPartitionV1 {
    /// Immutable partition descriptor.
    pub descriptor: PartitionDescriptorV1,
    /// Exact deterministic Parquet bytes.
    pub parquet_bytes: Vec<u8>,
    /// One coverage entry per frame/table projection in the object.
    pub coverage: Vec<ProjectionCoverageV1>,
}

/// Outputs completed by one append or final drain.
#[derive(Clone, Debug, Default)]
pub struct PartitionBuildOutputV1 {
    /// Newly sealed non-empty physical partitions.
    pub partitions: Vec<CompletedPartitionV1>,
    /// Explicit required zero-row coverage entries.
    pub zero_row_coverage: Vec<ProjectionCoverageV1>,
}

/// Deterministic rotation and hard-bound policy.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ParquetRotationConfigV1 {
    /// Rotate before adding a frame that would exceed this row target.
    pub target_rows: u64,
    /// Rotate before adding a frame that would exceed this Arrow-memory target.
    pub target_uncompressed_bytes: u64,
    /// Reject one frame/table projection above this row bound.
    pub hard_rows: u64,
    /// Reject one frame/table projection or encoded partition above this byte bound.
    pub hard_bytes: u64,
    /// Positive Clock bucket width used in physical paths.
    pub time_bucket_ns: i64,
}

impl Default for ParquetRotationConfigV1 {
    fn default() -> Self {
        Self {
            target_rows: 100_000,
            target_uncompressed_bytes: 64 * 1024 * 1024,
            hard_rows: 1_000_000,
            hard_bytes: 1024 * 1024 * 1024,
            time_bucket_ns: 60_000_000_000,
        }
    }
}

impl ParquetRotationConfigV1 {
    /// Validates targets against hard whole-frame bounds.
    pub fn validate(self) -> Result<Self, ParquetProjectionError> {
        if self.target_rows == 0
            || self.target_uncompressed_bytes == 0
            || self.hard_rows == 0
            || self.hard_bytes == 0
            || self.time_bucket_ns <= 0
        {
            return Err(ParquetProjectionError::InvalidRotationConfig);
        }
        if self.target_rows > self.hard_rows || self.target_uncompressed_bytes > self.hard_bytes {
            return Err(ParquetProjectionError::InvalidRotationConfig);
        }
        Ok(self)
    }
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct OpenPartitionKey {
    archive_id: ArchiveId,
    session_id: SessionId,
    table: TableId,
    source_id: Option<String>,
    time_bucket: i64,
}

#[derive(Debug)]
struct OpenPartition {
    key: OpenPartitionKey,
    row_count: u64,
    uncompressed_bytes: u64,
    projections: Vec<ValidatedProjection>,
}

#[derive(Debug)]
struct ValidatedProjection {
    projection: FrameTableProjectionV1,
    evidence: ProjectionEvidence,
}

#[derive(Debug)]
enum PreparedFrameProjection {
    Zero {
        projection: FrameTableProjectionV1,
        evidence: ProjectionEvidence,
    },
    Nonzero {
        key: OpenPartitionKey,
        projection: ValidatedProjection,
        uncompressed_bytes: u64,
    },
}

#[derive(Clone, Copy, Debug)]
struct PlannedOpenAction {
    rotates_existing: bool,
    seals_combined: bool,
    combined_rows: u64,
    combined_uncompressed_bytes: u64,
}

/// Independent-table, whole-frame deterministic partition builder.
#[derive(Debug)]
pub struct ParquetPartitionBuilderV1 {
    schemas: ArchiveSchemasV1,
    config: ParquetRotationConfigV1,
    open: BTreeMap<OpenPartitionKey, OpenPartition>,
}

impl ParquetPartitionBuilderV1 {
    /// Creates an empty builder from exact schemas and validated bounds.
    pub fn new(
        schemas: ArchiveSchemasV1,
        config: ParquetRotationConfigV1,
    ) -> Result<Self, ParquetProjectionError> {
        Ok(Self {
            schemas,
            config: config.validate()?,
            open: BTreeMap::new(),
        })
    }

    /// Appends every table projection of one terminal frame atomically.
    ///
    /// A non-empty projection is either retained whole in one open builder or
    /// returned whole in one sealed partition. Required zero-row projections
    /// return durable coverage evidence without creating an empty Parquet file.
    pub fn append_frame(
        &mut self,
        projections: Vec<FrameTableProjectionV1>,
    ) -> Result<PartitionBuildOutputV1, ParquetProjectionError> {
        validate_frame_set(&projections)?;
        let prepared = prepare_frame_projections(projections, &self.schemas, self.config)?;
        let mut output = PartitionBuildOutputV1::default();
        let mut actions = Vec::with_capacity(prepared.len());
        for item in &prepared {
            let PreparedFrameProjection::Nonzero {
                key,
                projection,
                uncompressed_bytes,
            } = item
            else {
                let PreparedFrameProjection::Zero {
                    projection,
                    evidence,
                } = item
                else {
                    unreachable!("prepared projection variants are exhaustive")
                };
                output
                    .zero_row_coverage
                    .push(ProjectionCoverageV1::new(projection, *evidence, None)?);
                actions.push(None);
                continue;
            };
            let existing = self.open.get(key);
            let rotates_existing = existing.is_some_and(|open| {
                open.row_count
                    .saturating_add(projection.evidence.row_count)
                    .gt(&self.config.target_rows)
                    || open
                        .uncompressed_bytes
                        .saturating_add(*uncompressed_bytes)
                        .gt(&self.config.target_uncompressed_bytes)
            });
            if rotates_existing {
                output.partitions.push(seal_partition(
                    existing.expect("rotation requires an existing partition"),
                    &self.schemas,
                    self.config,
                )?);
            }
            let base = if rotates_existing { None } else { existing };
            let combined_rows = base
                .map_or(0, |open| open.row_count)
                .checked_add(projection.evidence.row_count)
                .ok_or(ParquetProjectionError::LengthOverflow)?;
            let combined_uncompressed_bytes = base
                .map_or(0, |open| open.uncompressed_bytes)
                .checked_add(*uncompressed_bytes)
                .ok_or(ParquetProjectionError::LengthOverflow)?;
            let seals_combined = combined_rows >= self.config.target_rows
                || combined_uncompressed_bytes >= self.config.target_uncompressed_bytes;
            if seals_combined {
                let mut projection_refs = base
                    .into_iter()
                    .flat_map(|open| open.projections.iter())
                    .collect::<Vec<_>>();
                projection_refs.push(projection);
                output.partitions.push(seal_projection_refs(
                    key,
                    combined_rows,
                    &projection_refs,
                    &self.schemas,
                    self.config,
                )?);
            }
            actions.push(Some(PlannedOpenAction {
                rotates_existing,
                seals_combined,
                combined_rows,
                combined_uncompressed_bytes,
            }));
        }

        for (item, action) in prepared.into_iter().zip(actions) {
            let PreparedFrameProjection::Nonzero {
                key, projection, ..
            } = item
            else {
                continue;
            };
            let action = action.expect("nonzero projection has a planned action");
            if action.rotates_existing {
                self.open
                    .remove(&key)
                    .expect("planned rotation retains its existing partition");
            }
            if action.seals_combined {
                if !action.rotates_existing {
                    self.open.remove(&key);
                }
                continue;
            }
            if let Some(open) = self.open.get_mut(&key) {
                open.row_count = action.combined_rows;
                open.uncompressed_bytes = action.combined_uncompressed_bytes;
                open.projections.push(projection);
            } else {
                self.open.insert(
                    key.clone(),
                    OpenPartition {
                        key,
                        row_count: action.combined_rows,
                        uncompressed_bytes: action.combined_uncompressed_bytes,
                        projections: vec![projection],
                    },
                );
            }
        }
        Ok(output)
    }

    /// Seals every remaining builder in deterministic composite-key order.
    pub fn finish(mut self) -> Result<PartitionBuildOutputV1, ParquetProjectionError> {
        let mut output = PartitionBuildOutputV1::default();
        for (_, open) in std::mem::take(&mut self.open) {
            output
                .partitions
                .push(seal_partition(&open, &self.schemas, self.config)?);
        }
        Ok(output)
    }
}

fn prepare_frame_projections(
    projections: Vec<FrameTableProjectionV1>,
    schemas: &ArchiveSchemasV1,
    config: ParquetRotationConfigV1,
) -> Result<Vec<PreparedFrameProjection>, ParquetProjectionError> {
    projections
        .into_iter()
        .map(|mut projection| {
            let evidence = projection.validate(schemas)?;
            if evidence.row_count == 0 {
                return Ok(PreparedFrameProjection::Zero {
                    projection,
                    evidence,
                });
            }
            let uncompressed_bytes = u64::try_from(projection.batch.get_array_memory_size())
                .map_err(|_| ParquetProjectionError::LengthOverflow)?;
            if evidence.row_count > config.hard_rows || uncompressed_bytes > config.hard_bytes {
                return Err(ParquetProjectionError::ProjectionExceedsHardBound);
            }
            let key = OpenPartitionKey {
                archive_id: projection.archive_id,
                session_id: projection.session_id,
                table: projection.table,
                source_id: projection.source_id.clone(),
                time_bucket: projection
                    .authoritative_frame_clock_ns
                    .div_euclid(config.time_bucket_ns),
            };
            projection.logical_rows.clear();
            Ok(PreparedFrameProjection::Nonzero {
                key,
                projection: ValidatedProjection {
                    projection,
                    evidence,
                },
                uncompressed_bytes,
            })
        })
        .collect()
}

fn validate_frame_set(
    projections: &[FrameTableProjectionV1],
) -> Result<(), ParquetProjectionError> {
    let Some(first) = projections.first() else {
        return Err(ParquetProjectionError::EmptyFrameProjectionSet);
    };
    let mut tables = BTreeSet::new();
    for projection in projections {
        if projection.archive_id != first.archive_id
            || projection.session_id != first.session_id
            || projection.frame_id != first.frame_id
            || projection.authoritative_frame_clock_ns != first.authoritative_frame_clock_ns
            || projection.source_id != first.source_id
        {
            return Err(ParquetProjectionError::MixedFrameIdentity);
        }
        if !tables.insert(projection.table) {
            return Err(ParquetProjectionError::DuplicateFrameTable(
                projection.table,
            ));
        }
    }
    Ok(())
}

fn seal_partition(
    open: &OpenPartition,
    schemas: &ArchiveSchemasV1,
    config: ParquetRotationConfigV1,
) -> Result<CompletedPartitionV1, ParquetProjectionError> {
    let projections = open.projections.iter().collect::<Vec<_>>();
    seal_projection_refs(&open.key, open.row_count, &projections, schemas, config)
}

fn seal_projection_refs(
    key: &OpenPartitionKey,
    row_count: u64,
    projection_refs: &[&ValidatedProjection],
    schemas: &ArchiveSchemasV1,
    config: ParquetRotationConfigV1,
) -> Result<CompletedPartitionV1, ParquetProjectionError> {
    if projection_refs.is_empty() || row_count == 0 {
        return Err(ParquetProjectionError::EmptyPhysicalPartition);
    }
    let table_schema = schemas.table(key.table)?;
    let batch = concat_batches(
        table_schema.schema(),
        projection_refs
            .iter()
            .map(|projection| &projection.projection.batch),
    )?;
    let sorted = sort_record_batch(key.table, &batch)?;
    let parquet_bytes = encode_parquet(table_schema.schema().clone(), &sorted)?;
    let physical_byte_length =
        u64::try_from(parquet_bytes.len()).map_err(|_| ParquetProjectionError::LengthOverflow)?;
    if physical_byte_length > config.hard_bytes {
        return Err(ParquetProjectionError::EncodedPartitionExceedsHardBound);
    }
    let physical_content_hash = domain_digest("aiperf.archive.partition.v1", &[&parquet_bytes]);
    let minimum_clock_ns = projection_refs
        .iter()
        .map(|projection| projection.projection.authoritative_frame_clock_ns)
        .min()
        .ok_or(ParquetProjectionError::EmptyPhysicalPartition)?;
    let maximum_clock_ns = projection_refs
        .iter()
        .map(|projection| projection.projection.authoritative_frame_clock_ns)
        .max()
        .ok_or(ParquetProjectionError::EmptyPhysicalPartition)?;
    let mut projections = projection_refs
        .iter()
        .map(|projection| PartitionProjectionEvidenceV1 {
            frame_id: projection.projection.frame_id,
            row_count: projection.evidence.row_count,
            logical_multiset_digest: projection.evidence.logical_multiset_digest,
        })
        .collect::<Vec<_>>();
    projections.sort_unstable_by_key(|projection| projection.frame_id);
    let logical_object_id = partition_logical_object_id_v1(
        key.table,
        table_schema.fingerprint(),
        physical_content_hash,
        &projections,
    );
    let source_component = key.source_id.as_deref().map_or_else(
        || "global".to_string(),
        |source| {
            format!(
                "source-{}",
                domain_digest("aiperf.archive.partition-source.v1", &[source.as_bytes()]).to_hex()
            )
        },
    );
    let physical_object_key = format!(
        "partitions/{}/session-{}/{source_component}/bucket-{}/part-{}.parquet",
        table_name(key.table),
        hex(key.session_id.as_bytes()),
        key.time_bucket,
        physical_content_hash.to_hex(),
    );
    let descriptor = PartitionDescriptorV1 {
        archive_id: key.archive_id,
        session_id: key.session_id,
        source_id: key.source_id.clone(),
        table: key.table,
        time_bucket: key.time_bucket,
        schema_fingerprint: table_schema.fingerprint(),
        physical_content_hash,
        physical_object_key,
        physical_byte_length,
        row_count,
        minimum_clock_ns,
        maximum_clock_ns,
        logical_object_id,
        projections,
    };
    descriptor.validate()?;
    let coverage = projection_refs
        .iter()
        .map(|projection| {
            ProjectionCoverageV1::new(
                &projection.projection,
                projection.evidence,
                Some(physical_content_hash),
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(CompletedPartitionV1 {
        descriptor,
        parquet_bytes,
        coverage,
    })
}

/// Derives the immutable logical partition identity from exact physical and projection evidence.
#[must_use]
pub fn partition_logical_object_id_v1(
    table: TableId,
    schema_fingerprint: Digest,
    physical_content_hash: Digest,
    projections: &[PartitionProjectionEvidenceV1],
) -> Digest {
    let table = [table as u8];
    let mut projection_bytes = Vec::with_capacity(projections.len() * 72);
    for projection in projections {
        projection_bytes.extend_from_slice(projection.frame_id.digest().as_bytes());
        projection_bytes.extend_from_slice(&projection.row_count.to_be_bytes());
        projection_bytes.extend_from_slice(projection.logical_multiset_digest.as_bytes());
    }
    domain_digest(
        "aiperf.archive.partition-logical-object.v1",
        &[
            &table,
            schema_fingerprint.as_bytes(),
            physical_content_hash.as_bytes(),
            &projection_bytes,
        ],
    )
}

fn encode_parquet(
    schema: Arc<arrow_schema::Schema>,
    batch: &RecordBatch,
) -> Result<Vec<u8>, ParquetProjectionError> {
    let properties = WriterProperties::builder()
        .set_writer_version(WriterVersion::PARQUET_2_0)
        .set_created_by(PARTITION_CREATED_BY.to_string())
        .set_compression(Compression::UNCOMPRESSED)
        .set_dictionary_enabled(false)
        .set_max_row_group_row_count(None)
        .set_max_row_group_bytes(None)
        .build();
    let mut writer = ArrowWriter::try_new(Vec::new(), schema, Some(properties))?;
    writer.write(batch)?;
    writer.into_inner().map_err(Into::into)
}

fn sort_record_batch(
    table: TableId,
    batch: &RecordBatch,
) -> Result<RecordBatch, ParquetProjectionError> {
    if batch.num_rows() < 2 {
        return Ok(batch.clone());
    }
    let sort_fields = sort_fields(table);
    let columns = sort_fields
        .iter()
        .map(|name| {
            batch
                .schema()
                .index_of(name)
                .map(|index| batch.column(index).clone())
                .map_err(ParquetProjectionError::Arrow)
        })
        .collect::<Result<Vec<_>, _>>()?;
    for column in &columns {
        validate_sort_type(column.data_type())?;
    }
    let mut order = (0..batch.num_rows()).collect::<Vec<_>>();
    order.sort_by(|left, right| {
        compare_sort_tuple(&columns, *left, *right).then_with(|| left.cmp(right))
    });
    if order
        .windows(2)
        .any(|pair| compare_sort_tuple(&columns, pair[0], pair[1]) == Ordering::Equal)
    {
        return Err(ParquetProjectionError::DuplicatePhysicalSortKey(table));
    }
    let indices = UInt32Array::from(
        order
            .into_iter()
            .map(|index| u32::try_from(index).expect("partition row count was hard-bounded"))
            .collect::<Vec<_>>(),
    );
    let arrays = batch
        .columns()
        .iter()
        .map(|array| take(array.as_ref(), &indices, None))
        .collect::<Result<Vec<_>, _>>()?;
    RecordBatch::try_new(batch.schema(), arrays).map_err(Into::into)
}

fn compare_sort_tuple(columns: &[Arc<dyn Array>], left: usize, right: usize) -> Ordering {
    columns
        .iter()
        .map(|column| compare_array_values(column.as_ref(), left, right))
        .find(|ordering| *ordering != Ordering::Equal)
        .unwrap_or(Ordering::Equal)
}

fn sort_fields(table: TableId) -> &'static [&'static str] {
    match table {
        TableId::Samples => &[
            "metric_family",
            "series_key",
            "clock_ns",
            "record_seq",
            "metric_point_seq",
        ],
        TableId::Attempts => &["source_id", "source_record_seq", "record_seq"],
        TableId::Families => &["source_id", "record_seq", "family_seq"],
        TableId::Markers => &["clock_ns", "record_seq", "marker_seq"],
        TableId::Losses => &["source_id", "record_seq", "loss_seq"],
        TableId::RawReferences => &["source_id", "record_seq"],
    }
}

fn validate_sort_type(data_type: &DataType) -> Result<(), ParquetProjectionError> {
    if matches!(
        data_type,
        DataType::Utf8 | DataType::FixedSizeBinary(_) | DataType::Int64 | DataType::UInt64
    ) {
        Ok(())
    } else {
        Err(ParquetProjectionError::UnsupportedSortType(
            data_type.clone(),
        ))
    }
}

fn compare_array_values(array: &dyn Array, left: usize, right: usize) -> Ordering {
    match (array.is_null(left), array.is_null(right)) {
        (true, true) => return Ordering::Equal,
        (true, false) => return Ordering::Less,
        (false, true) => return Ordering::Greater,
        (false, false) => {}
    }
    match array.data_type() {
        DataType::Utf8 => {
            let array = array
                .as_any()
                .downcast_ref::<StringArray>()
                .expect("validated Utf8 array");
            array
                .value(left)
                .as_bytes()
                .cmp(array.value(right).as_bytes())
        }
        DataType::FixedSizeBinary(_) => {
            let array = array
                .as_any()
                .downcast_ref::<FixedSizeBinaryArray>()
                .expect("validated fixed-binary array");
            array.value(left).cmp(array.value(right))
        }
        DataType::Int64 => {
            let array = array
                .as_any()
                .downcast_ref::<Int64Array>()
                .expect("validated Int64 array");
            array.value(left).cmp(&array.value(right))
        }
        DataType::UInt64 => {
            let array = array
                .as_any()
                .downcast_ref::<UInt64Array>()
                .expect("validated UInt64 array");
            array.value(left).cmp(&array.value(right))
        }
        _ => unreachable!("sort types were validated"),
    }
}

fn validate_row_identity_and_clock(
    projection: &FrameTableProjectionV1,
) -> Result<(), ParquetProjectionError> {
    let rows = projection.batch.num_rows();
    validate_fixed_column(
        &projection.batch,
        "archive_id",
        projection.archive_id.as_bytes(),
    )?;
    validate_fixed_column(
        &projection.batch,
        "session_id",
        projection.session_id.as_bytes(),
    )?;
    validate_fixed_column(
        &projection.batch,
        "frame_id",
        projection.frame_id.digest().as_bytes(),
    )?;
    if projection
        .batch
        .schema()
        .field_with_name("source_id")
        .is_ok()
    {
        validate_source_column(&projection.batch, projection.source_id.as_deref())?;
    }
    match projection.table {
        TableId::Attempts => {
            if rows != 1 {
                return Err(ParquetProjectionError::AttemptFrameCardinality(rows));
            }
            let outcome = dictionary_string(&projection.batch, "outcome", 0)?;
            let field = if matches!(outcome, "success" | "empty") {
                "capture_ns"
            } else {
                "outcome_observed_ns"
            };
            validate_clock_column(
                &projection.batch,
                field,
                projection.authoritative_frame_clock_ns,
            )?;
        }
        TableId::Samples => validate_clock_column(
            &projection.batch,
            "clock_ns",
            projection.authoritative_frame_clock_ns,
        )?,
        TableId::Markers => {
            if rows > 1 {
                return Err(ParquetProjectionError::ControlFrameCardinality(rows));
            }
            validate_clock_column(
                &projection.batch,
                "clock_ns",
                projection.authoritative_frame_clock_ns,
            )?;
        }
        TableId::Losses => {
            if rows > 1 {
                return Err(ParquetProjectionError::ControlFrameCardinality(rows));
            }
            validate_clock_column(
                &projection.batch,
                "loss_observed_ns",
                projection.authoritative_frame_clock_ns,
            )?;
        }
        TableId::Families | TableId::RawReferences => {}
    }
    Ok(())
}

fn validate_fixed_column(
    batch: &RecordBatch,
    name: &str,
    expected: &[u8],
) -> Result<(), ParquetProjectionError> {
    let column = batch
        .column_by_name(name)
        .ok_or_else(|| ParquetProjectionError::MissingColumn(name.to_string()))?;
    let array = column
        .as_any()
        .downcast_ref::<FixedSizeBinaryArray>()
        .ok_or_else(|| ParquetProjectionError::ColumnType(name.to_string()))?;
    if (0..array.len()).any(|row| array.is_null(row) || array.value(row) != expected) {
        return Err(ParquetProjectionError::RowIdentity(name.to_string()));
    }
    Ok(())
}

fn validate_source_column(
    batch: &RecordBatch,
    expected: Option<&str>,
) -> Result<(), ParquetProjectionError> {
    let column = batch
        .column_by_name("source_id")
        .ok_or_else(|| ParquetProjectionError::MissingColumn("source_id".to_string()))?;
    let array = column
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| ParquetProjectionError::ColumnType("source_id".to_string()))?;
    for row in 0..array.len() {
        match expected {
            Some(expected) if !array.is_null(row) && array.value(row) == expected => {}
            None if array.is_null(row) => {}
            _ => return Err(ParquetProjectionError::RowIdentity("source_id".to_string())),
        }
    }
    Ok(())
}

fn validate_clock_column(
    batch: &RecordBatch,
    name: &str,
    expected: i64,
) -> Result<(), ParquetProjectionError> {
    let column = batch
        .column_by_name(name)
        .ok_or_else(|| ParquetProjectionError::MissingColumn(name.to_string()))?;
    let array = column
        .as_any()
        .downcast_ref::<Int64Array>()
        .ok_or_else(|| ParquetProjectionError::ColumnType(name.to_string()))?;
    if (0..array.len()).any(|row| array.is_null(row) || array.value(row) != expected) {
        return Err(ParquetProjectionError::AuthoritativeClock(name.to_string()));
    }
    Ok(())
}

fn dictionary_string<'a>(
    batch: &'a RecordBatch,
    name: &str,
    row: usize,
) -> Result<&'a str, ParquetProjectionError> {
    use arrow_array::DictionaryArray;
    use arrow_array::types::Int8Type;

    let column = batch
        .column_by_name(name)
        .ok_or_else(|| ParquetProjectionError::MissingColumn(name.to_string()))?;
    let dictionary = column
        .as_any()
        .downcast_ref::<DictionaryArray<Int8Type>>()
        .ok_or_else(|| ParquetProjectionError::ColumnType(name.to_string()))?;
    if dictionary.is_null(row) {
        return Err(ParquetProjectionError::RowIdentity(name.to_string()));
    }
    let values = dictionary
        .values()
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| ParquetProjectionError::ColumnType(name.to_string()))?;
    let key = usize::try_from(dictionary.keys().value(row))
        .map_err(|_| ParquetProjectionError::ColumnType(name.to_string()))?;
    Ok(values.value(key))
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct PartitionDescriptorWireV1 {
    archive_id: String,
    logical_object_id: String,
    maximum_clock_ns: i64,
    minimum_clock_ns: i64,
    physical_byte_length: u64,
    physical_content_hash: String,
    physical_object_key: String,
    projections: Vec<PartitionProjectionEvidenceWireV1>,
    row_count: u64,
    schema_fingerprint: String,
    session_id: String,
    source_id: Option<String>,
    table: String,
    time_bucket: i64,
    version: u8,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct PartitionProjectionEvidenceWireV1 {
    frame_id: String,
    logical_multiset_digest: String,
    row_count: u64,
}

impl From<&PartitionDescriptorV1> for PartitionDescriptorWireV1 {
    fn from(value: &PartitionDescriptorV1) -> Self {
        Self {
            archive_id: hex(value.archive_id.as_bytes()),
            logical_object_id: value.logical_object_id.to_hex(),
            maximum_clock_ns: value.maximum_clock_ns,
            minimum_clock_ns: value.minimum_clock_ns,
            physical_byte_length: value.physical_byte_length,
            physical_content_hash: value.physical_content_hash.to_hex(),
            physical_object_key: value.physical_object_key.clone(),
            projections: value
                .projections
                .iter()
                .map(|projection| PartitionProjectionEvidenceWireV1 {
                    frame_id: projection.frame_id.digest().to_hex(),
                    logical_multiset_digest: projection.logical_multiset_digest.to_hex(),
                    row_count: projection.row_count,
                })
                .collect(),
            row_count: value.row_count,
            schema_fingerprint: value.schema_fingerprint.to_hex(),
            session_id: hex(value.session_id.as_bytes()),
            source_id: value.source_id.clone(),
            table: table_name(value.table).to_string(),
            time_bucket: value.time_bucket,
            version: PARTITION_DESCRIPTOR_VERSION,
        }
    }
}

impl TryFrom<PartitionDescriptorWireV1> for PartitionDescriptorV1 {
    type Error = ParquetProjectionError;

    fn try_from(value: PartitionDescriptorWireV1) -> Result<Self, Self::Error> {
        if value.version != PARTITION_DESCRIPTOR_VERSION {
            return Err(ParquetProjectionError::DescriptorVersion(value.version));
        }
        Ok(Self {
            archive_id: ArchiveId::new(hex_array(&value.archive_id)?)
                .map_err(ParquetProjectionError::FrameIdentity)?,
            session_id: SessionId::new(hex_array(&value.session_id)?)
                .map_err(ParquetProjectionError::FrameIdentity)?,
            source_id: value.source_id,
            table: crate::table_id(&value.table)?,
            time_bucket: value.time_bucket,
            schema_fingerprint: Digest::parse(&value.schema_fingerprint)
                .map_err(ParquetProjectionError::Digest)?,
            physical_content_hash: Digest::parse(&value.physical_content_hash)
                .map_err(ParquetProjectionError::Digest)?,
            physical_object_key: value.physical_object_key,
            physical_byte_length: value.physical_byte_length,
            row_count: value.row_count,
            minimum_clock_ns: value.minimum_clock_ns,
            maximum_clock_ns: value.maximum_clock_ns,
            logical_object_id: Digest::parse(&value.logical_object_id)
                .map_err(ParquetProjectionError::Digest)?,
            projections: value
                .projections
                .into_iter()
                .map(|projection| {
                    Ok::<_, ParquetProjectionError>(PartitionProjectionEvidenceV1 {
                        frame_id: FrameId::new(
                            Digest::parse(&projection.frame_id)
                                .map_err(ParquetProjectionError::Digest)?,
                        ),
                        row_count: projection.row_count,
                        logical_multiset_digest: Digest::parse(&projection.logical_multiset_digest)
                            .map_err(ParquetProjectionError::Digest)?,
                    })
                })
                .collect::<Result<_, _>>()?,
        })
    }
}

fn hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        output.push(DIGITS[usize::from(byte >> 4)] as char);
        output.push(DIGITS[usize::from(byte & 0x0f)] as char);
    }
    output
}

fn hex_array<const N: usize>(value: &str) -> Result<[u8; N], ParquetProjectionError> {
    if value.len() != N * 2 || !value.is_ascii() {
        return Err(ParquetProjectionError::InvalidHex);
    }
    let mut output = [0_u8; N];
    for (index, slot) in output.iter_mut().enumerate() {
        let offset = index * 2;
        *slot = u8::from_str_radix(&value[offset..offset + 2], 16)
            .map_err(|_| ParquetProjectionError::InvalidHex)?;
    }
    Ok(output)
}

/// Invalid physical projection, rotation, descriptor, or Parquet bytes.
#[derive(Debug)]
pub enum ParquetProjectionError {
    /// Frozen Arrow schema failed to load.
    Schema(SchemaError),
    /// Arrow operation failed.
    Arrow(ArrowError),
    /// Parquet encoding failed.
    Parquet(ParquetError),
    /// Canonical JSON failed.
    Canonical(CanonicalJsonError),
    /// Ordinary serde JSON conversion failed.
    Json(serde_json::Error),
    /// Logical row encoding failed.
    LogicalRow(crate::LogicalRowError),
    /// Primary index construction failed.
    Index(IndexError),
    /// Digest text failed.
    Digest(crate::DigestError),
    /// Typed UUID construction failed.
    FrameIdentity(crate::FrameIdentityError),
    /// Rotation configuration is zero, negative, or exceeds a hard bound.
    InvalidRotationConfig,
    /// Source identity was present but empty.
    EmptySourceId,
    /// RecordBatch schema does not equal the exact table schema.
    SchemaMismatch(TableId),
    /// Arrow/logical row counts disagree.
    LogicalRowCount {
        /// Arrow rows.
        arrow: usize,
        /// Logical rows.
        logical: usize,
    },
    /// Logical row belongs to another table.
    LogicalRowTableMismatch,
    /// Logical row carries another schema fingerprint.
    LogicalRowSchemaMismatch,
    /// Caller-supplied logical evidence does not describe the exact Arrow rows.
    PhysicalLogicalEvidenceMismatch,
    /// One append call contained no required table projections.
    EmptyFrameProjectionSet,
    /// One append call mixed terminal frame identities.
    MixedFrameIdentity,
    /// One frame declared one table twice.
    DuplicateFrameTable(TableId),
    /// One projection exceeds the validated hard row/memory bound.
    ProjectionExceedsHardBound,
    /// Encoded Parquet bytes exceeded the validated hard bound.
    EncodedPartitionExceedsHardBound,
    /// A zero-row projection named a fragment or a nonzero projection omitted one.
    CoverageFragmentCardinality,
    /// A physical partition had no rows/projections.
    EmptyPhysicalPartition,
    /// Physical descriptor contains zero-row evidence.
    ZeroRowsInPhysicalEvidence,
    /// Physical descriptor repeats one frame.
    DuplicateFrameEvidence(FrameId),
    /// Descriptor row total disagrees with projection evidence.
    PartitionRowCount {
        /// Descriptor total.
        descriptor: u64,
        /// Sum of projections.
        projections: u64,
    },
    /// Descriptor Clock range is reversed.
    ClockRange,
    /// Object key does not end in its exact content hash.
    ObjectKeyHashMismatch,
    /// Object key is outside its exact table prefix.
    ObjectKeyTableMismatch,
    /// Descriptor logical ID is not derived from its exact contents.
    LogicalObjectIdMismatch,
    /// Numeric conversion or addition overflowed.
    LengthOverflow,
    /// Sort column uses a type outside the six frozen table orders.
    UnsupportedSortType(DataType),
    /// Two rows share a supposedly total physical sort tuple.
    DuplicatePhysicalSortKey(TableId),
    /// Required physical column is absent.
    MissingColumn(String),
    /// Physical column has an unexpected Arrow type.
    ColumnType(String),
    /// Row identity does not equal its projection header.
    RowIdentity(String),
    /// Timed row does not equal the authoritative frame Clock.
    AuthoritativeClock(String),
    /// One attempt frame did not contain exactly one attempt row.
    AttemptFrameCardinality(usize),
    /// Lifecycle/loss frame contained more than one timed row.
    ControlFrameCardinality(usize),
    /// Partition descriptor has an unknown version.
    DescriptorVersion(u8),
    /// Descriptor UUID/digest hex is malformed.
    InvalidHex,
}

impl From<SchemaError> for ParquetProjectionError {
    fn from(value: SchemaError) -> Self {
        Self::Schema(value)
    }
}

impl From<ArrowError> for ParquetProjectionError {
    fn from(value: ArrowError) -> Self {
        Self::Arrow(value)
    }
}

impl From<ParquetError> for ParquetProjectionError {
    fn from(value: ParquetError) -> Self {
        Self::Parquet(value)
    }
}

impl From<IndexError> for ParquetProjectionError {
    fn from(value: IndexError) -> Self {
        Self::Index(value)
    }
}

impl Display for ParquetProjectionError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Schema(error) => write!(formatter, "archive schema failed: {error}"),
            Self::Arrow(error) => write!(formatter, "Arrow projection failed: {error}"),
            Self::Parquet(error) => write!(formatter, "Parquet projection failed: {error}"),
            Self::Canonical(error) => write!(formatter, "canonical descriptor failed: {error}"),
            Self::Json(error) => write!(formatter, "partition descriptor JSON failed: {error}"),
            Self::LogicalRow(error) => write!(formatter, "logical row failed: {error}"),
            Self::Index(error) => write!(formatter, "partition index failed: {error}"),
            Self::Digest(error) => write!(formatter, "partition digest failed: {error}"),
            Self::FrameIdentity(error) => write!(formatter, "partition identity failed: {error}"),
            Self::InvalidRotationConfig => formatter.write_str("invalid Parquet rotation config"),
            Self::EmptySourceId => formatter.write_str("partition source ID cannot be empty"),
            Self::SchemaMismatch(table) => {
                write!(formatter, "{table:?} RecordBatch schema mismatch")
            }
            Self::LogicalRowCount { arrow, logical } => write!(
                formatter,
                "Arrow/logical row count mismatch: {arrow} versus {logical}"
            ),
            Self::LogicalRowTableMismatch => formatter.write_str("logical row table mismatch"),
            Self::LogicalRowSchemaMismatch => formatter.write_str("logical row schema mismatch"),
            Self::PhysicalLogicalEvidenceMismatch => {
                formatter.write_str("Arrow rows do not match supplied logical evidence")
            }
            Self::EmptyFrameProjectionSet => formatter.write_str("frame projection set is empty"),
            Self::MixedFrameIdentity => {
                formatter.write_str("frame projection set mixes identities")
            }
            Self::DuplicateFrameTable(table) => write!(formatter, "frame repeats {table:?}"),
            Self::ProjectionExceedsHardBound => {
                formatter.write_str("projection exceeds hard partition bound")
            }
            Self::EncodedPartitionExceedsHardBound => {
                formatter.write_str("encoded partition exceeds hard byte bound")
            }
            Self::CoverageFragmentCardinality => {
                formatter.write_str("invalid coverage fragment cardinality")
            }
            Self::EmptyPhysicalPartition => {
                formatter.write_str("physical partition cannot be empty")
            }
            Self::ZeroRowsInPhysicalEvidence => {
                formatter.write_str("physical evidence cannot contain zero rows")
            }
            Self::DuplicateFrameEvidence(frame) => {
                write!(formatter, "duplicate partition frame evidence {frame:?}")
            }
            Self::PartitionRowCount {
                descriptor,
                projections,
            } => write!(
                formatter,
                "partition row count {descriptor} differs from projection sum {projections}"
            ),
            Self::ClockRange => formatter.write_str("partition Clock range is reversed"),
            Self::ObjectKeyHashMismatch => {
                formatter.write_str("partition object key/content hash mismatch")
            }
            Self::ObjectKeyTableMismatch => {
                formatter.write_str("partition object key/table mismatch")
            }
            Self::LogicalObjectIdMismatch => {
                formatter.write_str("partition logical object ID mismatch")
            }
            Self::LengthOverflow => formatter.write_str("partition count or length overflow"),
            Self::UnsupportedSortType(data_type) => {
                write!(formatter, "unsupported sort type {data_type:?}")
            }
            Self::DuplicatePhysicalSortKey(table) => {
                write!(formatter, "duplicate physical sort key in {table:?}")
            }
            Self::MissingColumn(name) => write!(formatter, "missing physical column {name:?}"),
            Self::ColumnType(name) => {
                write!(formatter, "invalid physical column type for {name:?}")
            }
            Self::RowIdentity(name) => write!(formatter, "row identity mismatch in {name:?}"),
            Self::AuthoritativeClock(name) => {
                write!(formatter, "authoritative frame Clock mismatch in {name:?}")
            }
            Self::AttemptFrameCardinality(rows) => {
                write!(formatter, "attempt frame contains {rows} rows")
            }
            Self::ControlFrameCardinality(rows) => {
                write!(formatter, "control frame contains {rows} rows")
            }
            Self::DescriptorVersion(version) => write!(
                formatter,
                "unsupported partition descriptor version {version}"
            ),
            Self::InvalidHex => formatter.write_str("invalid partition descriptor hex"),
        }
    }
}

impl std::error::Error for ParquetProjectionError {}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::builder::StringDictionaryBuilder;
    use arrow_array::types::Int8Type;
    use arrow_array::{BooleanArray, FixedSizeBinaryArray, UInt64Array};
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

    use super::*;
    use crate::{LogicalValue, ProjectionEvidence};

    fn ids() -> (ArchiveId, SessionId, FrameId) {
        (
            ArchiveId::new([1; 16]).unwrap(),
            SessionId::new([2; 16]).unwrap(),
            FrameId::new(Digest::from_bytes([3; 32])),
        )
    }

    #[test]
    fn zero_row_required_projection_emits_coverage_without_parquet() {
        let schemas = ArchiveSchemasV1::load().unwrap();
        let (archive_id, session_id, frame_id) = ids();
        let table = schemas.table(TableId::Families).unwrap();
        let projection = FrameTableProjectionV1 {
            archive_id,
            session_id,
            source_id: Some("source-a".to_string()),
            frame_id,
            authoritative_frame_clock_ns: 10,
            table: TableId::Families,
            batch: RecordBatch::new_empty(table.schema().clone()),
            logical_rows: Vec::new(),
        };
        let mut builder =
            ParquetPartitionBuilderV1::new(schemas, ParquetRotationConfigV1::default()).unwrap();
        let output = builder.append_frame(vec![projection]).unwrap();
        assert!(output.partitions.is_empty());
        assert_eq!(output.zero_row_coverage.len(), 1);
        assert_eq!(output.zero_row_coverage[0].row_count, 0);
        assert!(output.zero_row_coverage[0].fragment_ids.is_empty());
        assert_eq!(
            output.zero_row_coverage[0].logical_multiset_digest,
            ProjectionEvidence::empty().logical_multiset_digest
        );
        assert!(builder.finish().unwrap().partitions.is_empty());
    }

    #[test]
    fn whole_frame_rows_sort_and_encode_to_byte_identical_parquet() {
        let first = build_family_partition();
        let second = build_family_partition();
        assert_eq!(first.descriptor, second.descriptor);
        assert_eq!(first.parquet_bytes, second.parquet_bytes);
        assert_eq!(first.coverage.len(), 1);
        assert_eq!(first.coverage[0].fragment_ids.len(), 1);
        let golden: serde_json::Value = serde_json::from_str(include_str!(
            "../tests/fixtures/families-parquet-v1-golden.json"
        ))
        .unwrap();
        assert_eq!(golden["table"], "families");
        assert_eq!(
            golden["parquet_byte_length"].as_u64().unwrap(),
            first.parquet_bytes.len() as u64
        );
        assert_eq!(
            golden["physical_content_hash"],
            first.descriptor.physical_content_hash.to_hex()
        );
        assert_eq!(
            golden["descriptor_hash"],
            domain_digest(
                "aiperf.archive.partition-descriptor-golden.v1",
                &[&first.descriptor.canonical_bytes().unwrap()]
            )
            .to_hex()
        );

        let bytes = bytes::Bytes::from(first.parquet_bytes.clone());
        let builder = ParquetRecordBatchReaderBuilder::try_new(bytes).unwrap();
        let expected = ArchiveSchemasV1::load().unwrap();
        assert_eq!(
            builder.schema().as_ref(),
            expected.table(TableId::Families).unwrap().schema().as_ref()
        );
        let mut reader = builder.build().unwrap();
        let batch = reader.next().unwrap().unwrap();
        let families = batch
            .column_by_name("metric_family")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(families.value(0), "a_family");
        assert_eq!(families.value(1), "z_family");
        assert_eq!(
            batch.schema().fields(),
            expected.table(TableId::Families).unwrap().schema().fields()
        );
    }

    #[test]
    fn canonical_partition_descriptor_round_trips_and_rebuilds_index_key() {
        let partition = build_family_partition();
        let bytes = partition.descriptor.canonical_bytes().unwrap();
        let decoded = PartitionDescriptorV1::from_canonical_bytes(&bytes).unwrap();
        assert_eq!(decoded, partition.descriptor);
        assert_eq!(
            decoded.index_key().unwrap(),
            partition.descriptor.index_key().unwrap()
        );
        assert_eq!(decoded.index_entry().unwrap().descriptor_bytes(), bytes);
    }

    #[test]
    fn projection_rejects_logical_evidence_not_derived_from_arrow_rows() {
        let (schemas, mut projection) = family_projection();
        projection.logical_rows[0] = projection.logical_rows[1].clone();
        assert!(matches!(
            projection.validate(&schemas),
            Err(ParquetProjectionError::PhysicalLogicalEvidenceMismatch)
        ));
    }

    #[test]
    fn failing_later_table_leaves_no_partial_frame_in_open_partitions() {
        let (schemas, family) = family_projection();
        let invalid_attempt = FrameTableProjectionV1 {
            archive_id: family.archive_id,
            session_id: family.session_id,
            source_id: family.source_id.clone(),
            frame_id: family.frame_id,
            authoritative_frame_clock_ns: family.authoritative_frame_clock_ns,
            table: TableId::Attempts,
            batch: RecordBatch::new_empty(
                schemas.table(TableId::Attempts).unwrap().schema().clone(),
            ),
            logical_rows: Vec::new(),
        };
        let mut builder = ParquetPartitionBuilderV1::new(
            schemas,
            ParquetRotationConfigV1 {
                target_rows: 10,
                target_uncompressed_bytes: 1 << 20,
                hard_rows: 10,
                hard_bytes: 1 << 20,
                time_bucket_ns: 100,
            },
        )
        .unwrap();
        assert!(matches!(
            builder.append_frame(vec![family, invalid_attempt]),
            Err(ParquetProjectionError::AttemptFrameCardinality(0))
        ));
        let output = builder.finish().unwrap();
        assert!(output.partitions.is_empty());
        assert!(output.zero_row_coverage.is_empty());
    }

    fn build_family_partition() -> CompletedPartitionV1 {
        let (schemas, projection) = family_projection();
        let mut builder = ParquetPartitionBuilderV1::new(
            schemas,
            ParquetRotationConfigV1 {
                target_rows: 2,
                target_uncompressed_bytes: 1 << 20,
                hard_rows: 10,
                hard_bytes: 1 << 20,
                time_bucket_ns: 100,
            },
        )
        .unwrap();
        let mut output = builder.append_frame(vec![projection]).unwrap();
        assert!(builder.finish().unwrap().partitions.is_empty());
        assert_eq!(output.partitions.len(), 1);
        output.partitions.remove(0)
    }

    fn family_projection() -> (ArchiveSchemasV1, FrameTableProjectionV1) {
        let schemas = ArchiveSchemasV1::load().unwrap();
        let (archive_id, session_id, frame_id) = ids();
        let table = schemas.table(TableId::Families).unwrap();
        let archive = [1_u8; 16];
        let session = [2_u8; 16];
        let frame = [3_u8; 32];
        let batch_id = [4_u8; 32];
        let mut semantic = StringDictionaryBuilder::<Int8Type>::new();
        semantic.append("gauge").unwrap();
        semantic.append("gauge").unwrap();
        let arrays: Vec<Arc<dyn Array>> = vec![
            Arc::new(FixedSizeBinaryArray::from(vec![&archive, &archive])),
            Arc::new(FixedSizeBinaryArray::from(vec![&session, &session])),
            Arc::new(StringArray::from(vec!["source-a", "source-a"])),
            Arc::new(FixedSizeBinaryArray::from(vec![&frame, &frame])),
            Arc::new(FixedSizeBinaryArray::from(vec![&batch_id, &batch_id])),
            Arc::new(UInt64Array::from(vec![10, 10])),
            Arc::new(UInt64Array::from(vec![2, 1])),
            Arc::new(StringArray::from(vec!["z_family", "a_family"])),
            Arc::new(StringArray::from(vec!["gauge", "gauge"])),
            Arc::new(semantic.finish()),
            Arc::new(BooleanArray::from(vec![false, false])),
            Arc::new(BooleanArray::from(vec![true, true])),
            Arc::new(BooleanArray::from(vec![false, false])),
            Arc::new(StringArray::from(vec![None::<&str>, None])),
            Arc::new(StringArray::from(vec![None::<&str>, None])),
            Arc::new(UInt64Array::from(vec![None, None])),
            Arc::new(UInt64Array::from(vec![Some(1), Some(1)])),
            Arc::new(UInt64Array::from(vec![None, None])),
            Arc::new(UInt64Array::from(vec![1, 1])),
            Arc::new(UInt64Array::from(vec![1, 1])),
            Arc::new(UInt64Array::from(vec![1, 1])),
        ];
        let batch = RecordBatch::try_new(table.schema().clone(), arrays).unwrap();
        let row = |family: &str, family_seq: u64| {
            CanonicalLogicalRow::encode(
                table.logical_schema(),
                &[
                    LogicalValue::Binary(archive.to_vec()),
                    LogicalValue::Binary(session.to_vec()),
                    LogicalValue::String("source-a".to_string()),
                    LogicalValue::Binary(frame.to_vec()),
                    LogicalValue::Binary(batch_id.to_vec()),
                    LogicalValue::Unsigned(10),
                    LogicalValue::Unsigned(u128::from(family_seq)),
                    LogicalValue::String(family.to_string()),
                    LogicalValue::String("gauge".to_string()),
                    LogicalValue::String("gauge".to_string()),
                    LogicalValue::Bool(false),
                    LogicalValue::Bool(true),
                    LogicalValue::Bool(false),
                    LogicalValue::Null,
                    LogicalValue::Null,
                    LogicalValue::Null,
                    LogicalValue::Unsigned(1),
                    LogicalValue::Null,
                    LogicalValue::Unsigned(1),
                    LogicalValue::Unsigned(1),
                    LogicalValue::Unsigned(1),
                ],
            )
            .unwrap()
        };
        let projection = FrameTableProjectionV1 {
            archive_id,
            session_id,
            source_id: Some("source-a".to_string()),
            frame_id,
            authoritative_frame_clock_ns: 10,
            table: TableId::Families,
            batch,
            logical_rows: vec![row("z_family", 2), row("a_family", 1)],
        };
        (schemas, projection)
    }
}
