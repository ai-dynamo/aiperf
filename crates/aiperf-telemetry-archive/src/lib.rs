// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Durable telemetry archive foundations.
//!
//! This crate owns the byte, schema, physical partition, query, and durability
//! authorities shared by archive writers. It deliberately has no metrics,
//! telemetry-source, runner, or provider SDK dependency. Runtime source and
//! provider adapters consume these types rather than defining competing frame,
//! manifest, receipt, schema, or content-addressing rules.

pub mod attempt;
pub mod boundary;
pub mod canonical_json;
pub mod control_frame_codec;
pub mod decode;
pub mod descriptor;
pub mod digest;
pub mod driver;
pub mod evidence;
pub mod filesystem_store;
pub mod frame_codec;
pub mod identity;
pub mod index;
pub mod key;
pub mod lifecycle;
pub mod loss;
pub mod manifest;
pub mod object_store;
pub mod owner;
pub mod parquet;
pub mod policy;
pub mod projection;
pub mod query;
pub mod raw;
pub mod receipt;
pub mod scheduling;
pub mod schema;
pub mod sink;
pub mod spool;
pub mod sync;
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
pub use control_frame_codec::{ControlFrameCodecError, ControlFrameCodecV1};
pub use decode::{
    AttemptDecoder, AttemptFacts, CompatibilityFallback, DecodeConfigError, DecodeLimits,
    DecodedAttempt, ExactEntityLease, FetchDisposition, FetchedAttempt, NativeDecodeOutcome,
    NativeEntityDecoder, NoopNativeEntityDecoder, ParseOutcome, PrometheusAttemptDecoder,
    StrictParseView,
};
pub use descriptor::{CanonicalDescriptor, DescriptorError};
pub use digest::{Digest, DigestError, domain_digest};
pub use driver::{
    ArchiveSourceError, DriverConsumerError, DriverStartError, DriverStopError, FetchRequest,
    FixedDeadlineTelemetryDriver, LocalCancellationSignal, PreparedTelemetryDriver,
    RunningTelemetryDriver, TelemetryAttemptConsumer, TelemetryDriverConfig,
    TelemetryDriverSummary, TelemetryFetcher,
};
pub use evidence::{
    CanonicalLogicalRow, LogicalField, LogicalRowError, LogicalSchema, LogicalType, LogicalValue,
    ProjectionEvidence, RequiredProjection, TableId,
};
pub use filesystem_store::FileArchiveObjectStore;
pub use frame_codec::{SourceFrameCodecError, SourceFrameCodecV1};
pub use identity::{
    ArchiveId, BatchId, ExactLossBatchInput, FrameId, FrameIdentityError, FrameIdentityV1,
    LifecycleBatchInput, ProjectionReservationId, ReservationKind, SaturationBatchInput, SessionId,
    SourceOutcome, TerminalKind,
};
pub use index::{
    CompositeIndexKeyV1, IndexClockRangeV1, IndexEntry, IndexError, IndexIdSetV1, IndexKey,
    IndexMutationSetV1, IndexObjectKind, IndexPageSink, IndexPageSource, IndexPruningSummaryV1,
    IndexRemoval, IndexRootV1, IndexScanPredicateV1, IndexScanStatsV1, IndexScanV1, IndexSnapshot,
    IndexSourceSelectionV1, MAX_INDEX_SOURCE_ID_BYTES, MAX_PRUNING_SUMMARY_EXACT_ID_BYTES,
    MAX_PRUNING_SUMMARY_EXACT_IDS, MemoryIndexPageStore, MutationMode, VerifiedIndexScannerV1,
};
pub use key::{
    ArchiveKeyError, ArchiveKeyProvider, ArchiveSubkey, Blake3ArchiveKeyProvider,
    keyed_domain_digest,
};
pub use lifecycle::{
    LifecycleCompletionReasonV1, LifecycleMarkerError, LifecycleMarkerKindV1, LifecycleMarkerV1,
    LifecyclePhaseStateV1,
};
pub use loss::{
    ExactLossRangeV1, LossKindV1, LossReasonV1, LossSaturationSnapshotV1, LossValidationError,
    loss_saturation_slot_id_v1,
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
pub use owner::{
    ArchiveFrameSequencerV1, ArchiveFrameTimingV1, FrameSequencingError, SequencedArchiveFrameV1,
    SourceProjectionPolicyV1,
};
pub use parquet::{
    CompletedPartitionV1, FrameTableProjectionV1, ParquetPartitionBuilderV1,
    ParquetProjectionError, ParquetRotationConfigV1, PartitionBuildOutputV1, PartitionDescriptorV1,
    PartitionProjectionEvidenceV1, ProjectionCoverageV1, partition_logical_object_id_v1,
    partition_object_key_v1,
};
pub use policy::{
    AdmissionRejection, AnyRotationPolicy, ArchiveAdmissionMode, ArchiveAdmissionPolicy,
    ArchiveIngressState, ArchiveProjectionFootprint, ArchiveProjectionPermit, ArchiveRecoveryError,
    ArchiveRecoveryPolicy, AttachedBestEffortAdmissionPolicy, BoundedSegmentRotationPolicy,
    CreateNewRecoveryPolicy, ExactResumeRecoveryPolicy, LocalArchiveState, OpenSegmentState,
    PolicyError, PrimaryWatchAdmissionPolicy, RecoveryPlan, RemoteArchiveState,
    SegmentRotationPolicy,
};
pub use projection::{
    ArchiveInfoLabelPartitionStatus, ArchiveSampleView, ArchiveSanitizer, AttributeMap,
    EnrichmentError, ExpositionProjectionContextV1, ExpositionProjectionError, ExpositionRowsV1,
    MetricFamilyRowV1, MetricPointRowV1, NoopEnricher, NoopSanitizer, SanitizationError,
    SanitizedSample, StaticLabelEnricher, TelemetryEnricher, TelemetryEnricherChain,
    project_exposition_v1,
};
pub use query::{
    CompactionLogicalProofV1, MemoryPartitionObjectSourceV1, PartitionDiscoveryV1,
    PartitionObjectReadError, PartitionObjectSourceV1, PartitionPredicateV1, QueryError,
    SourcePredicateV1, VerifiedQueryRootV1, read_partition_v1,
    verify_compaction_logical_equality_v1,
};
pub use raw::{
    AES_256_GCM_SIV_RANDOM96_V1_DESCRIPTOR, Aes256GcmSivRandom96V1, ArchiveRawKeyProvider,
    MemoryRawKeyProvider, OsRawNonceSource, RAW_ENVELOPE_AAD_BYTES, RAW_ENVELOPE_ALGORITHM_V1,
    RAW_ENVELOPE_KEY_BYTES, RAW_ENVELOPE_MAX_NONCE_DRAWS, RAW_ENVELOPE_MAX_OBJECTS_PER_KEY,
    RAW_ENVELOPE_MAX_PLAINTEXT_BYTES, RAW_ENVELOPE_NONCE_BYTES, RAW_ENVELOPE_TAG_BYTES,
    RAW_ENVELOPE_V1, RawCoverageRequirementV1, RawEnvelope, RawEnvelopeDescriptor,
    RawEnvelopeError, RawEnvelopeKey, RawEnvelopeObjectV1, RawEnvelopeProfile,
    RawEnvelopePublicHeaderV1, RawKeyError, RawNonceError, RawNonceReservationV1, RawNonceSource,
    RawObjectCandidate, RawObjectDescriptorV1, RawObjectRegistry, RawObjectStateV1,
    RawPrepareContext, RawPrepareDispositionV1, RawPrepareOutcomeV1, RawRegisteredObjectV1,
    RawRegistryError, RawRegistryLimitsV1, raw_envelope_aad_v1, raw_object_id_v1,
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
    ALL_ARROW_SCHEMA_DESCRIPTORS_V1, ARROW_ALIASES_V1, ATTEMPTS_ARROW_SCHEMA_V1, ArchiveSchemasV1,
    ArchiveTableSchemaV1, FAMILIES_ARROW_SCHEMA_V1, LOSSES_ARROW_SCHEMA_V1,
    MARKERS_ARROW_SCHEMA_V1, RAW_REFERENCES_ARROW_SCHEMA_V1, SAMPLES_ARROW_SCHEMA_V1, SchemaError,
    arrow_schema_fingerprint, table_id, table_name,
};
pub use sink::{
    AppendReceipt, ArchiveSink, ArchiveSinkError, ArchiveWalFrame, ArchiveWalFrameDecoder,
    CheckpointCompletion, DurabilityCompletion, FinalizeCompletion, LocalParquetArchiveSink,
    MemoryArchiveSink, MemoryArchiveSinkFault, OwnedLocalArchiveSink, OwnedLocalArchiveSinkFactory,
    OwnedReceiptJournalMode, ReceiptEventDraft, RecoveredArchive, TerminationReason,
};
pub use spool::{
    DurabilityEdge, DurabilityFaultInjector, FailAtDurabilityEdge, FilesystemKind,
    LocalArchiveRepository, LocalWalWriter, NoDurabilityFaults, OwnedLocalWalWriter,
    QualifiedSpool, RecoveryExpectation, SpoolError, SpoolQualification,
};
pub use sync::{
    ArchiveRemoteSynchronizer, ArchiveSyncError, RemoteLatestV1, RemotePublicationCompletionV1,
    RemotePublicationObservationV1, WriterClaimV1,
};
pub use time::{EpochAnchor, EpochAnchorError, EpochAnchorProvider, SystemEpochAnchorProvider};
pub use wal::{
    Crc32c, RecoveredWal, SEALED_WAL_FOOTER_BYTES, SealedWalSegment, WAL_FOOTER_MAGIC, WalError,
    WalFrame, WalFrameHeaderV1, WalSegmentBuilder, WalSegmentHeaderV1,
};
