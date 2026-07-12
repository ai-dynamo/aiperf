// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Durable telemetry archive foundations.
//!
//! This crate owns the byte and durability authorities shared by future
//! archive writers. It deliberately has no Arrow, Parquet, metrics, telemetry
//! source, runner, or object-store SDK dependency. Physical table writers and
//! provider adapters consume these types rather than defining competing frame,
//! manifest, receipt, or content-addressing rules.

pub mod attempt;
pub mod boundary;
pub mod canonical_json;
pub mod decode;
pub mod descriptor;
pub mod digest;
pub mod evidence;
pub mod identity;
pub mod index;
pub mod key;
pub mod manifest;
pub mod object_store;
pub mod parquet;
pub mod projection;
pub mod query;
pub mod receipt;
pub mod scheduling;
pub mod schema;
pub mod spool;
pub mod time;
pub mod wal;

pub use attempt::{
    ArchiveScrapeFrameV1, ArchiveScrapeRecordV1, AttemptValidationError, ScrapeReasonV1,
};
pub use boundary::{
    BoundaryCapturePlan, BoundaryPlanError, BoundaryPlanRegistry, BoundaryReference,
    BoundaryReferenceKey, BoundaryRole, SealedBoundaryCapturePlan, SourceBoundarySnapshotCommand,
};
pub use canonical_json::{CanonicalJsonError, CanonicalJsonValue};
pub use decode::{
    AttemptDecoder, AttemptFacts, CompatibilityFallback, DecodeConfigError, DecodeLimits,
    DecodedAttempt, ExactEntityLease, FetchDisposition, FetchedAttempt, NativeDecodeOutcome,
    NativeEntityDecoder, NoopNativeEntityDecoder, ParseOutcome, PrometheusAttemptDecoder,
    StrictParseView,
};
pub use descriptor::{CanonicalDescriptor, DescriptorError};
pub use digest::{Digest, DigestError, domain_digest};
pub use evidence::{
    CanonicalLogicalRow, LogicalField, LogicalRowError, LogicalSchema, LogicalType, LogicalValue,
    ProjectionEvidence, RequiredProjection, TableId,
};
pub use identity::{
    ArchiveId, BatchId, ExactLossBatchInput, FrameId, FrameIdentityError, FrameIdentityV1,
    LifecycleBatchInput, ProjectionReservationId, ReservationKind, SaturationBatchInput, SessionId,
    SourceOutcome, TerminalKind,
};
pub use index::{
    CompositeIndexKeyV1, IndexEntry, IndexError, IndexKey, IndexMutationSetV1, IndexObjectKind,
    IndexPageSink, IndexPageSource, IndexRemoval, IndexRootV1, IndexSnapshot, MemoryIndexPageStore,
    MutationMode,
};
pub use key::{
    ArchiveKeyError, ArchiveKeyProvider, ArchiveSubkey, Blake3ArchiveKeyProvider,
    keyed_domain_digest,
};
pub use manifest::{
    ArchiveState, GenerationMutationV1, GenerationObjectV1, GenerationTransactionKind,
    GenerationV1, GenesisV1, HeadDescriptorV1, LocalLatestV1, ManifestError, TimeDomain,
};
pub use object_store::{
    ArchiveObjectStore, ArchiveStoreCapabilities, ArchiveStoreError, CreateReceipt,
    HeadUpdateError, MemoryArchiveObjectStore, MemoryStoreFault, NamedObjectVisibility,
    VersionedHead, archive_object_digest,
};
pub use parquet::{
    CompletedPartitionV1, FrameTableProjectionV1, ParquetPartitionBuilderV1,
    ParquetProjectionError, ParquetRotationConfigV1, PartitionBuildOutputV1, PartitionDescriptorV1,
    PartitionProjectionEvidenceV1, ProjectionCoverageV1, partition_logical_object_id_v1,
};
pub use projection::{
    ArchiveInfoLabelPartitionStatus, ArchiveSampleView, ArchiveSanitizer, AttributeMap,
    EnrichmentError, ExpositionProjectionContextV1, ExpositionProjectionError, ExpositionRowsV1,
    MetricFamilyRowV1, MetricPointRowV1, NoopEnricher, NoopSanitizer, SanitizationError,
    SanitizedSample, StaticLabelEnricher, TelemetryEnricher, project_exposition_v1,
};
pub use query::{
    CompactionLogicalProofV1, MemoryPartitionObjectSourceV1, PartitionDiscoveryV1,
    PartitionObjectReadError, PartitionObjectSourceV1, PartitionPredicateV1, QueryError,
    SourcePredicateV1, VerifiedQueryRootV1, read_partition_v1,
    verify_compaction_logical_equality_v1,
};
pub use receipt::{
    ExecutionId, ObjectVersionKind, ObservationKind, ReceiptBatchId, ReceiptBatchV1, ReceiptError,
    ReceiptEventId, ReceiptEventV1, ReceiptIndexKeyV1, ReceiptJournal, ReceiptObserverEpochId,
    ReceiptObserverEpochV1, ReceiptTargetId, ReceiptTargetKind, ReceiptTargetV1,
    RemotePublicationTargetV1, StableObjectVersion, WalRangeTargetV1, WriterClaimState,
    receipt_range_coverage,
};
pub use scheduling::{
    AbsoluteCallDeadline, CadenceAdvance, CadenceDeadline, FixedDeadlineCadence,
    IssuedSourceAttempt, MissedCadenceRange, SchedulingError, SourceAttemptGate, SourceAttemptKind,
};
pub use schema::{
    ALL_ARROW_SCHEMA_DESCRIPTORS_V1, ATTEMPTS_ARROW_SCHEMA_V1, ArchiveSchemasV1,
    ArchiveTableSchemaV1, FAMILIES_ARROW_SCHEMA_V1, LOSSES_ARROW_SCHEMA_V1,
    MARKERS_ARROW_SCHEMA_V1, RAW_REFERENCES_ARROW_SCHEMA_V1, SAMPLES_ARROW_SCHEMA_V1, SchemaError,
    arrow_schema_fingerprint, table_id, table_name,
};
pub use spool::{
    DurabilityEdge, DurabilityFaultInjector, FailAtDurabilityEdge, FilesystemKind,
    LocalArchiveRepository, LocalWalWriter, NoDurabilityFaults, QualifiedSpool,
    RecoveryExpectation, SpoolError, SpoolQualification,
};
pub use time::{EpochAnchor, EpochAnchorError, EpochAnchorProvider, SystemEpochAnchorProvider};
pub use wal::{
    Crc32c, RecoveredWal, SEALED_WAL_FOOTER_BYTES, SealedWalSegment, WAL_FOOTER_MAGIC, WalError,
    WalFrame, WalFrameHeaderV1, WalSegmentBuilder, WalSegmentHeaderV1,
};
