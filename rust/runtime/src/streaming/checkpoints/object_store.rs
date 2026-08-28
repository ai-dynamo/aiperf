// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Conditional object-store checkpoint backend and its bounded object I/O contract.
//!
//! The backend publishes one generation by writing content-addressed immutable
//! objects and then conditionally replacing exactly one pointer object using the
//! exact prior provider version. Two writers racing the same predecessor
//! therefore land at most one complete generation: the loser's compare-and-swap
//! observes a different version and refuses, and its already-written immutable
//! objects remain unreferenced rather than partially visible.
//!
//! Every provider interaction is bounded. Declared object, page, and chunk
//! lengths are checked against the caller's budget *before* the provider is
//! asked for bytes, so an oversized or hostile declaration is refused without
//! allocating. Uploads stream through [`BudgetOwnedObjectReader`] and restores
//! stream through ranged reads, so a multi-gibibyte object is never assembled
//! whole.
//!
//! `list_versions` and `delete_version` are checkpoint-prefix retention
//! authority only. There is deliberately no source discovery or reconciliation
//! operation on this trait: checkpoint code must never become a second, weaker
//! path to the streaming source surface.

use std::{
    collections::{BTreeMap, BTreeSet},
    fmt::Debug,
    num::NonZeroUsize,
    rc::Rc,
};

use async_trait::async_trait;
use bytes::Bytes;
use serde::{Deserialize, Serialize};

use crate::streaming::{
    budget::{BudgetLease, BudgetLimits},
    checkpoint::{
        BudgetedCheckpointBytes, CheckpointBackendBudgetFailureCode, CheckpointBackendBudgetKind,
        CheckpointError, CheckpointGeneration, CommittedCheckpointGeneration,
        CommittedParticipantState, CurrentV4ParticipantStateContext, DecodedCheckpointGeneration,
        LegacyParticipantState, LegacyV3CheckpointGeneration, ParticipantStateDescriptor,
        PreparedParticipantState, PrevalidatedCheckpointGenerationCandidate, StreamRunIdentity,
        decode_versioned_checkpoint_generation,
    },
    checkpoint_backend::{
        CheckpointCommitMetadata, CheckpointGenerationExpectations, CurrentV4CheckpointGeneration,
        FrozenGenerationTransactionInputs, LeasedCheckpointGeneration, LeasedGenerationReader,
        LegacyV3LeasedGenerationReader, LegacyV3ReadOnlyFixture, StreamingCheckpointBackend,
        StreamingGenerationTransaction, build_prevalidated_candidate, sealed,
        validate_commit_metadata,
    },
    checkpoints::budget::{BackendBudget, backend_error},
    identity::ContentDigest,
    reliability::PreparedIssueReceiptResultPartition,
    results::{
        BudgetedResultDescriptors, PreparedResultEpoch, ResultIndexCursor, ResultIndexPage,
        ResultIndexReadBudget, ResultPartition, ResultSegmentDescriptor, ResultSegmentReader,
        canonical_result_index_object, canonical_result_index_root, descriptor_retained_bytes,
        result_totals,
    },
};

/// Registry identifier of the conditional object-store checkpoint backend.
pub const OBJECT_STORE_CHECKPOINT_BACKEND_ID: &str = "object_store";

/// Exact storage message a provider refusing conditional pointer update reports.
///
/// Capability disagreement has to be distinguishable from an ordinary provider
/// fault, because the first is a configuration error that must fail before any
/// effect and the second is a retryable runtime condition.
pub const CONDITIONAL_WRITE_UNSUPPORTED_MESSAGE: &str =
    "object store does not support exact conditional pointer update";

/// Exact storage message reported when a writer's pointer version is stale.
pub const STALE_WRITER_MESSAGE: &str = "object store pointer changed under this writer";

/// Provider-neutral object key.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct ObjectKey(String);

impl ObjectKey {
    /// Build one key from an exact provider-neutral path.
    #[must_use]
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    /// Borrow the exact key text.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Opaque provider version identifying one exact stored object revision.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct ObjectVersion(String);

impl ObjectVersion {
    /// Build one version from exact provider-supplied text.
    #[must_use]
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    /// Borrow the exact version text.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Complete pointer payload offered to one conditional replacement.
#[derive(Debug)]
pub struct PointerObject {
    /// Exact encoded pointer bytes.
    pub bytes: Bytes,
    /// Digest of the exact encoded bytes.
    pub digest: ContentDigest,
    /// Permit retaining those bytes until the pointer write completes.
    pub lease: BudgetLease,
}

/// Half-open byte range requested from one exact object version.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ObjectReadRange {
    /// First byte offset read.
    pub offset: u64,
    /// Exact number of bytes requested.
    pub length: u64,
}

/// Caller-owned bound on one ranged read.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ObjectReadBudget {
    /// Largest chunk the caller will retain from one provider response.
    pub max_chunk_bytes: usize,
}

/// Caller-owned bound on one retention listing page.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ObjectListBudget {
    /// Largest number of entries one page may return.
    pub max_items: NonZeroUsize,
    /// Largest retained metadata allocation one page may hold.
    pub max_metadata_bytes: NonZeroUsize,
}

/// Opaque provider continuation token for retention listing.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ObjectListCursor(String);

impl ObjectListCursor {
    /// Build one cursor from exact provider-supplied text.
    #[must_use]
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    /// Borrow the exact cursor text.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Declared identity and length of one stored object version.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ObjectMetadata {
    /// Exact object key.
    pub key: ObjectKey,
    /// Exact provider version.
    pub version: ObjectVersion,
    /// Length the provider declares for this version.
    pub byte_length: u64,
}

/// One listing page whose retained metadata owns its permits until dropped.
#[derive(Debug)]
pub struct BudgetOwnedObjectPage {
    /// Exact retained entries.
    pub objects: Box<[ObjectMetadata]>,
    /// Continuation cursor, present exactly when more entries remain.
    pub next: Option<ObjectListCursor>,
    /// Permit retaining the entries above.
    pub lease: BudgetLease,
}

/// One bounded object chunk whose bytes own their permit until dropped.
#[derive(Debug)]
pub struct BudgetOwnedObjectChunk {
    /// Exact chunk bytes.
    pub bytes: Bytes,
    /// Permit retaining those bytes.
    pub lease: BudgetLease,
}

/// Bounded streaming source for one immutable object upload.
///
/// The reader declares its complete length and digest up front so the store can
/// refuse an oversized object before requesting a single byte, then yields
/// bounded chunks that retain their own permits.
#[async_trait(?Send)]
pub trait BudgetOwnedObjectReader {
    /// Exact complete object length.
    fn content_length(&self) -> u64;

    /// Exact digest of the complete object.
    fn content_digest(&self) -> ContentDigest;

    /// Yield the next chunk of at most `max_bytes`, or `None` at end of object.
    async fn next_chunk(
        &mut self,
        max_bytes: usize,
    ) -> Result<Option<BudgetOwnedObjectChunk>, CheckpointError>;
}

/// Stable classification of one object-store checkpoint failure.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CheckpointFailureCode {
    /// A declared object, page, or chunk length exceeded an owned limit.
    ObjectLimitExceeded,
    /// The provider cannot perform exact conditional pointer update.
    ConditionalWriteUnsupported,
    /// The pointer moved under this writer between begin and publication.
    StaleWriter,
    /// The provider failed without changing the authoritative pointer.
    Provider,
}

/// Stable failure classification for object-store checkpoint errors.
pub trait CheckpointErrorCode {
    /// Classify this error for object-store callers.
    fn code(&self) -> CheckpointFailureCode;
}

impl CheckpointErrorCode for CheckpointError {
    fn code(&self) -> CheckpointFailureCode {
        match self {
            Self::BackendBudget { code, .. } => match code {
                CheckpointBackendBudgetFailureCode::ItemCapacity
                | CheckpointBackendBudgetFailureCode::ByteCapacity => {
                    CheckpointFailureCode::ObjectLimitExceeded
                }
                _ => CheckpointFailureCode::Provider,
            },
            Self::ResultIndexReadBudgetTooSmall { .. } => CheckpointFailureCode::ObjectLimitExceeded,
            Self::GenerationConflict { .. } | Self::LeaseLost { .. } => {
                CheckpointFailureCode::StaleWriter
            }
            Self::Storage { message } if message == CONDITIONAL_WRITE_UNSUPPORTED_MESSAGE => {
                CheckpointFailureCode::ConditionalWriteUnsupported
            }
            Self::Storage { message } if message == STALE_WRITER_MESSAGE => {
                CheckpointFailureCode::StaleWriter
            }
            _ => CheckpointFailureCode::Provider,
        }
    }
}

/// Object store offering immutable writes and exact conditional pointer update.
#[async_trait(?Send)]
pub trait ConditionalObjectStore: Debug {
    /// Write one content-addressed immutable object, streaming its bytes.
    async fn put_immutable(
        &self,
        object: Box<dyn BudgetOwnedObjectReader>,
    ) -> Result<ObjectVersion, CheckpointError>;

    /// Replace one pointer only when its current version equals `expected`.
    async fn compare_and_swap_pointer(
        &self,
        key: &ObjectKey,
        expected: Option<&ObjectVersion>,
        next: PointerObject,
    ) -> Result<ObjectVersion, CheckpointError>;

    /// Read one bounded range of one exact object version.
    async fn get_version_range(
        &self,
        key: &ObjectKey,
        version: &ObjectVersion,
        range: ObjectReadRange,
        budget: ObjectReadBudget,
    ) -> Result<BudgetOwnedObjectChunk, CheckpointError>;

    /// List retained versions under one checkpoint prefix.
    async fn list_versions(
        &self,
        prefix: &ObjectKey,
        cursor: Option<&ObjectListCursor>,
        budget: ObjectListBudget,
    ) -> Result<BudgetOwnedObjectPage, CheckpointError>;

    /// Delete one exact retained version.
    async fn delete_version(
        &self,
        key: &ObjectKey,
        version: &ObjectVersion,
    ) -> Result<(), CheckpointError>;
}

/// Derive the exact content-addressed key one immutable object lands at.
///
/// The backend and its store must agree on this derivation: `put_immutable`
/// takes no key because an immutable object's address is a pure function of the
/// prefix and its content digest. The object's kind is deliberately absent — a
/// store writing an object knows only its bytes, and a digest already
/// distinguishes every object the checkpoint plane retains.
#[must_use]
pub fn immutable_object_key(prefix: &ObjectKey, digest: &ContentDigest) -> ObjectKey {
    ObjectKey::new(format!(
        "{}/objects/{}",
        prefix.as_str(),
        hex_digest(digest)
    ))
}

/// Derive the exact pointer key one logical run's head lands at.
#[must_use]
pub fn run_pointer_key(prefix: &ObjectKey, run: &StreamRunIdentity) -> ObjectKey {
    ObjectKey::new(format!(
        "{}/pointers/{}",
        prefix.as_str(),
        hex_bytes(run.logical_replay_run().as_bytes())
    ))
}

fn hex_digest(digest: &ContentDigest) -> String {
    hex_bytes(digest.as_bytes())
}

fn hex_bytes(bytes: &[u8]) -> String {
    use std::fmt::Write as _;
    bytes.iter().fold(String::new(), |mut text, byte| {
        // Writing to a `String` cannot fail; the result is discarded rather than
        // unwrapped so no panic path exists here.
        let _ = write!(text, "{byte:02x}");
        text
    })
}

/// Storage version recorded by one committed pointer.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum PointerStorageVersion {
    CurrentV4,
    LegacyV3ReadOnly,
}

/// Exact encoded head pointer for one logical run.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct CheckpointPointerDocument {
    run: StreamRunIdentity,
    storage_version: PointerStorageVersion,
    generation: CheckpointGeneration,
    generation_object: ObjectKey,
    generation_version: ObjectVersion,
    generation_byte_length: u64,
}

/// Capacity limits for each independently owned object-backend resource.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ObjectCheckpointLimits {
    /// Simultaneously live generation transactions.
    pub transactions: BudgetLimits,
    /// Descriptors retained by staged transaction indexes.
    pub prepared_indexes: BudgetLimits,
    /// Bytes staged for and streamed into immutable uploads.
    pub storage: BudgetLimits,
    /// Descriptor summaries returned from result staging.
    pub result_summaries: BudgetLimits,
    /// Concurrent generation, participant, result, and page readers.
    pub reads: BudgetLimits,
    /// Largest chunk one upload or restore retains from one provider response.
    pub max_chunk_bytes: NonZeroUsize,
    /// Bound applied to every retention listing page.
    pub list: ObjectListBudget,
}

#[derive(Clone)]
struct ObjectBudgets {
    transactions: BackendBudget,
    prepared_indexes: BackendBudget,
    storage: BackendBudget,
    result_summaries: BackendBudget,
    reads: BackendBudget,
}

/// Conditional object-store checkpoint backend.
#[derive(Clone)]
pub struct ObjectCheckpointBackend {
    store: Rc<dyn ConditionalObjectStore>,
    prefix: ObjectKey,
    budgets: ObjectBudgets,
    limits: ObjectCheckpointLimits,
}

impl Debug for ObjectCheckpointBackend {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ObjectCheckpointBackend")
            .field("prefix", &self.prefix)
            .field("store", &self.store)
            .finish()
    }
}

/// One pointer observation: its version plus the head it names.
struct ObservedPointer {
    version: ObjectVersion,
    document: CheckpointPointerDocument,
}

impl ObjectCheckpointBackend {
    /// Construct a backend after validating all five budgets in field order.
    ///
    /// `prefix` must be the same prefix the store writes under: immutable object
    /// addresses are derived from it by [`immutable_object_key`], and a
    /// disagreement would make committed objects unreadable.
    pub fn new(
        store: Rc<dyn ConditionalObjectStore>,
        prefix: ObjectKey,
        limits: ObjectCheckpointLimits,
    ) -> Result<Self, CheckpointError> {
        let budgets = ObjectBudgets {
            transactions: BackendBudget::new(
                CheckpointBackendBudgetKind::Transaction,
                limits.transactions,
            )?,
            prepared_indexes: BackendBudget::new(
                CheckpointBackendBudgetKind::PreparedIndex,
                limits.prepared_indexes,
            )?,
            storage: BackendBudget::new(CheckpointBackendBudgetKind::Storage, limits.storage)?,
            result_summaries: BackendBudget::new(
                CheckpointBackendBudgetKind::ResultSummary,
                limits.result_summaries,
            )?,
            reads: BackendBudget::new(CheckpointBackendBudgetKind::Read, limits.reads)?,
        };
        Ok(Self {
            store,
            prefix,
            budgets,
            limits,
        })
    }

    /// Borrow the checkpoint prefix every object address derives from.
    #[must_use]
    pub const fn prefix(&self) -> &ObjectKey {
        &self.prefix
    }

    /// Restore the current committed generation under one caller read budget.
    ///
    /// Declared pointer and generation lengths are checked against the caller's
    /// budget before the provider is asked for bytes, so an oversized or hostile
    /// declaration is refused without allocating.
    pub async fn restore_current(
        &self,
        budget: ObjectReadBudget,
    ) -> Result<Option<CheckpointGeneration>, CheckpointError> {
        let run = self.sole_pointer_run(budget).await?;
        let Some(observed) = self.observe_pointer(&run, budget).await? else {
            return Ok(None);
        };
        Ok(Some(observed.document.generation))
    }

    /// Import one fully precharged, strictly verified legacy-v3 read-only head.
    #[doc(hidden)]
    pub async fn import_legacy_v3_read_only_fixture(
        &self,
        fixture: LegacyV3ReadOnlyFixture,
    ) -> Result<(), CheckpointError> {
        let generation = fixture.generation().clone();
        let run = *generation.run();
        let (
            fixture_run,
            head,
            generation_object,
            participant_objects,
            result_index_object,
            result_objects,
        ) = fixture.into_parts();
        if fixture_run != run || head != *generation.generation() {
            return Err(CheckpointError::ObjectVerification);
        }
        let mut objects: Vec<(ContentDigest, Bytes)> = Vec::new();
        let (generation_digest, generation_bytes) = generation_object.into_storage_parts();
        let generation_length = u64::try_from(generation_bytes.len())
            .map_err(|_| provider_error("object length"))?;
        objects.push((generation_digest, generation_bytes));
        for object in participant_objects.into_objects() {
            let (digest, bytes) = object.into_storage_parts();
            objects.push((digest, bytes));
        }
        let (digest, bytes) = result_index_object.into_storage_parts();
        objects.push((digest, bytes));
        for object in result_objects.into_objects() {
            let (digest, bytes) = object.into_storage_parts();
            objects.push((digest, bytes));
        }
        let mut generation_version = None;
        for (digest, bytes) in objects {
            let version = self.upload_object(digest, bytes).await?;
            if digest == generation_digest {
                generation_version = Some(version);
            }
        }
        let generation_version = generation_version.ok_or(CheckpointError::ObjectVerification)?;
        let generation_key = immutable_object_key(&self.prefix, &generation_digest);
        let generation_byte_length = generation_length;
        let document = CheckpointPointerDocument {
            run,
            storage_version: PointerStorageVersion::LegacyV3ReadOnly,
            generation: head,
            generation_object: generation_key,
            generation_version,
            generation_byte_length,
        };
        self.write_pointer(&run, None, document).await.map(|_| ())
    }

    async fn open_latest_inner(
        &self,
        run: &StreamRunIdentity,
        expected: &CheckpointGenerationExpectations,
    ) -> Result<Option<LeasedCheckpointGeneration>, CheckpointError> {
        if run != &expected.run {
            return Err(CheckpointError::ObjectVerification);
        }
        let budget = self.read_budget();
        let Some(observed) = self.observe_pointer(run, budget).await? else {
            return Ok(None);
        };
        let (bytes, lease) = self
            .read_exact_version(
                &observed.document.generation_object,
                &observed.document.generation_version,
                observed.document.generation_byte_length,
                *observed.document.generation.digest(),
            )
            .await?;
        let decoded =
            decode_versioned_checkpoint_generation(&bytes, self.budgets.storage.limits().max_bytes)?;
        let opened = match (observed.document.storage_version, decoded) {
            (
                PointerStorageVersion::CurrentV4,
                DecodedCheckpointGeneration::CurrentV4(candidate),
            ) => {
                if candidate.generation() != observed.document.generation {
                    return Err(CheckpointError::ObjectVerification);
                }
                let committed = candidate
                    .prevalidate_for_publication(
                        run,
                        &expected.participant_plan,
                        &expected.execution_plan_digest,
                        &expected.result_plan_digest,
                    )?
                    .into_committed_after_publication_fence();
                LeasedCheckpointGeneration::current_v4(ObjectGenerationReader {
                    backend: self.clone(),
                    generation: committed,
                    _generation_lease: lease,
                })
            }
            (
                PointerStorageVersion::LegacyV3ReadOnly,
                DecodedCheckpointGeneration::LegacyV3ReadOnly(generation),
            ) => {
                if generation.generation() != &observed.document.generation {
                    return Err(CheckpointError::ObjectVerification);
                }
                generation.verify_against(
                    run,
                    &expected.participant_plan,
                    &expected.execution_plan_digest,
                    &expected.result_plan_digest,
                )?;
                LeasedCheckpointGeneration::legacy_v3(ObjectLegacyV3GenerationReader {
                    backend: self.clone(),
                    generation,
                    _generation_lease: lease,
                })
            }
            _ => return Err(CheckpointError::ObjectVerification),
        };
        Ok(Some(opened))
    }

    async fn begin_generation_inner(
        &self,
        run: StreamRunIdentity,
        expected: Option<CurrentV4CheckpointGeneration>,
        expectations: CheckpointGenerationExpectations,
    ) -> Result<ObjectGenerationTransaction, CheckpointError> {
        if run != expectations.run {
            return Err(CheckpointError::ObjectVerification);
        }
        let budget = self.read_budget();
        let observed = self.observe_pointer(&run, budget).await?;
        // A legacy-v3 head has no successor authority, so the refusal lands
        // before any pointer compare-and-swap is even attempted.
        if let Some(observed) = &observed {
            if observed.document.storage_version == PointerStorageVersion::LegacyV3ReadOnly {
                return Err(CheckpointError::LegacyReadOnlyHead);
            }
        }
        let actual = observed
            .as_ref()
            .map(|observed| observed.document.generation.clone());
        let expected_generation = expected
            .as_ref()
            .map(CurrentV4CheckpointGeneration::generation)
            .cloned();
        if actual != expected_generation {
            return Err(CheckpointError::GenerationConflict {
                expected: expected_generation,
                actual,
            });
        }
        let lease = self.budgets.transactions.acquire(1, 1).await?;
        Ok(ObjectGenerationTransaction {
            backend: self.clone(),
            run,
            expected,
            expectations,
            pointer_version: observed.map(|observed| observed.version),
            _transaction_lease: lease,
            participants: Vec::new(),
            staged_results: None,
        })
    }

    const fn read_budget(&self) -> ObjectReadBudget {
        ObjectReadBudget {
            max_chunk_bytes: self.limits.max_chunk_bytes.get(),
        }
    }

    /// Locate the sole pointer under this prefix, if any run has published.
    async fn sole_pointer_run(
        &self,
        budget: ObjectReadBudget,
    ) -> Result<StreamRunIdentity, CheckpointError> {
        let prefix = ObjectKey::new(format!("{}/pointers/", self.prefix.as_str()));
        let page = self
            .store
            .list_versions(&prefix, None, self.limits.list)
            .await?;
        let Some(metadata) = page.objects.first() else {
            return Err(provider_error("no published checkpoint pointer"));
        };
        self.check_declared_length(metadata.byte_length, budget)?;
        let bytes = self
            .read_exact_bytes(&metadata.key, &metadata.version, metadata.byte_length)
            .await?;
        let document: CheckpointPointerDocument =
            serde_json::from_slice(&bytes).map_err(|_| CheckpointError::ObjectVerification)?;
        Ok(document.run)
    }

    async fn observe_pointer(
        &self,
        run: &StreamRunIdentity,
        budget: ObjectReadBudget,
    ) -> Result<Option<ObservedPointer>, CheckpointError> {
        let key = run_pointer_key(&self.prefix, run);
        let page = self
            .store
            .list_versions(&key, None, self.limits.list)
            .await?;
        let Some(metadata) = page
            .objects
            .iter()
            .find(|metadata| metadata.key == key)
            .cloned()
        else {
            return Ok(None);
        };
        drop(page);
        // The declared length is checked against the caller's budget before a
        // single provider byte is requested.
        self.check_declared_length(metadata.byte_length, budget)?;
        let bytes = self
            .read_exact_bytes(&metadata.key, &metadata.version, metadata.byte_length)
            .await?;
        let document: CheckpointPointerDocument =
            serde_json::from_slice(&bytes).map_err(|_| CheckpointError::ObjectVerification)?;
        if document.run != *run {
            return Err(CheckpointError::ObjectVerification);
        }
        Ok(Some(ObservedPointer {
            version: metadata.version,
            document,
        }))
    }

    fn check_declared_length(
        &self,
        declared: u64,
        budget: ObjectReadBudget,
    ) -> Result<(), CheckpointError> {
        let limit = u64::try_from(self.budgets.reads.limits().max_bytes)
            .map_err(|_| provider_error("read limit"))?;
        let chunk = u64::try_from(budget.max_chunk_bytes).map_err(|_| provider_error("chunk"))?;
        if chunk == 0 || declared > limit {
            return Err(backend_error(
                CheckpointBackendBudgetKind::Read,
                CheckpointBackendBudgetFailureCode::ByteCapacity,
            ));
        }
        Ok(())
    }

    /// Stream one exact version under a permit and verify its complete digest.
    async fn read_exact_version(
        &self,
        key: &ObjectKey,
        version: &ObjectVersion,
        declared: u64,
        digest: ContentDigest,
    ) -> Result<(Bytes, BudgetLease), CheckpointError> {
        self.check_declared_length(declared, self.read_budget())?;
        let length = usize::try_from(declared).map_err(|_| provider_error("object length"))?;
        let lease = self.budgets.reads.acquire(1, length).await?;
        let bytes = self.stream_version(key, version, declared).await?;
        if bytes.len() != length
            || ContentDigest::from_bytes(*blake3::hash(&bytes).as_bytes()) != digest
        {
            return Err(CheckpointError::ObjectVerification);
        }
        Ok((bytes, lease))
    }

    async fn read_exact_bytes(
        &self,
        key: &ObjectKey,
        version: &ObjectVersion,
        declared: u64,
    ) -> Result<Bytes, CheckpointError> {
        let length = usize::try_from(declared).map_err(|_| provider_error("object length"))?;
        let _lease = self.budgets.reads.acquire(1, length).await?;
        self.stream_version(key, version, declared).await
    }

    async fn stream_version(
        &self,
        key: &ObjectKey,
        version: &ObjectVersion,
        declared: u64,
    ) -> Result<Bytes, CheckpointError> {
        let budget = self.read_budget();
        let chunk = u64::try_from(budget.max_chunk_bytes).map_err(|_| provider_error("chunk"))?;
        let capacity = usize::try_from(declared).map_err(|_| provider_error("object length"))?;
        let mut assembled = Vec::with_capacity(capacity);
        let mut offset = 0u64;
        while offset < declared {
            let length = chunk.min(declared - offset);
            let piece = self
                .store
                .get_version_range(key, version, ObjectReadRange { offset, length }, budget)
                .await?;
            let received =
                u64::try_from(piece.bytes.len()).map_err(|_| provider_error("chunk length"))?;
            if received == 0 || received > length {
                return Err(CheckpointError::ObjectVerification);
            }
            assembled.extend_from_slice(&piece.bytes);
            offset += received;
        }
        Ok(Bytes::from(assembled.into_boxed_slice()))
    }

    async fn read_object_by_digest(
        &self,
        digest: ContentDigest,
    ) -> Result<BudgetedCheckpointBytes, CheckpointError> {
        let key = immutable_object_key(&self.prefix, &digest);
        let metadata = self.resolve_version(&key).await?;
        let (bytes, lease) = self
            .read_exact_version(&key, &metadata.version, metadata.byte_length, digest)
            .await?;
        BudgetedCheckpointBytes::new(bytes, lease)
    }

    async fn resolve_version(&self, key: &ObjectKey) -> Result<ObjectMetadata, CheckpointError> {
        let page = self
            .store
            .list_versions(key, None, self.limits.list)
            .await?;
        page.objects
            .iter()
            .find(|metadata| metadata.key == *key)
            .cloned()
            .ok_or(CheckpointError::ObjectVerification)
    }

    /// Stream one immutable object into the store and verify it reads back.
    async fn upload_object(
        &self,
        digest: ContentDigest,
        bytes: Bytes,
    ) -> Result<ObjectVersion, CheckpointError> {
        let length = u64::try_from(bytes.len()).map_err(|_| provider_error("object length"))?;
        if ContentDigest::from_bytes(*blake3::hash(&bytes).as_bytes()) != digest {
            return Err(CheckpointError::ObjectVerification);
        }
        let lease = self.budgets.storage.acquire(1, bytes.len()).await?;
        let reader = LeasedBytesObjectReader {
            bytes,
            offset: 0,
            digest,
            lease,
        };
        let version = self.store.put_immutable(Box::new(reader)).await?;
        let key = immutable_object_key(&self.prefix, &digest);
        // Verify before the pointer can ever reference this version: a pointer
        // must only name objects that were written whole and read back exact.
        let (_, verify_lease) = self
            .read_exact_version(&key, &version, length, digest)
            .await?;
        drop(verify_lease);
        Ok(version)
    }

    async fn write_pointer(
        &self,
        run: &StreamRunIdentity,
        expected: Option<&ObjectVersion>,
        document: CheckpointPointerDocument,
    ) -> Result<ObjectVersion, CheckpointError> {
        let encoded = serde_json::to_vec(&document)
            .map_err(|_| provider_error("encode checkpoint pointer"))?;
        let bytes = Bytes::from(encoded.into_boxed_slice());
        let lease = self.budgets.storage.acquire(1, bytes.len()).await?;
        let digest = ContentDigest::from_bytes(*blake3::hash(&bytes).as_bytes());
        let key = run_pointer_key(&self.prefix, run);
        self.store
            .compare_and_swap_pointer(
                &key,
                expected,
                PointerObject {
                    bytes,
                    digest,
                    lease,
                },
            )
            .await
    }
}

#[async_trait(?Send)]
impl StreamingCheckpointBackend for ObjectCheckpointBackend {
    async fn open_latest(
        &self,
        run: &StreamRunIdentity,
        expected: &CheckpointGenerationExpectations,
    ) -> Result<Option<LeasedCheckpointGeneration>, CheckpointError> {
        self.open_latest_inner(run, expected).await
    }

    async fn begin_generation(
        &self,
        run: StreamRunIdentity,
        expected: Option<CurrentV4CheckpointGeneration>,
        expectations: CheckpointGenerationExpectations,
    ) -> Result<Box<dyn StreamingGenerationTransaction>, CheckpointError> {
        Ok(Box::new(
            self.begin_generation_inner(run, expected, expectations)
                .await?,
        ))
    }
}

/// In-memory upload source whose chunks carry permits split from one lease.
struct LeasedBytesObjectReader {
    bytes: Bytes,
    offset: usize,
    digest: ContentDigest,
    lease: BudgetLease,
}

#[async_trait(?Send)]
impl BudgetOwnedObjectReader for LeasedBytesObjectReader {
    fn content_length(&self) -> u64 {
        // A staged object is bounded by the storage budget, so its length always
        // fits; a hypothetical overflow reports zero rather than panicking.
        u64::try_from(self.bytes.len()).unwrap_or_default()
    }

    fn content_digest(&self) -> ContentDigest {
        self.digest
    }

    async fn next_chunk(
        &mut self,
        max_bytes: usize,
    ) -> Result<Option<BudgetOwnedObjectChunk>, CheckpointError> {
        if max_bytes == 0 {
            return Err(backend_error(
                CheckpointBackendBudgetKind::Storage,
                CheckpointBackendBudgetFailureCode::ByteCapacity,
            ));
        }
        if self.offset >= self.bytes.len() {
            return Ok(None);
        }
        let end = self.bytes.len().min(self.offset + max_bytes);
        let chunk = self.bytes.slice(self.offset..end);
        let lease = self
            .lease
            .split_off(0, chunk.len())
            .map_err(|_| CheckpointError::ObjectVerification)?;
        self.offset = end;
        Ok(Some(BudgetOwnedObjectChunk {
            bytes: chunk,
            lease,
        }))
    }
}

struct StagedParticipant {
    descriptor: ParticipantStateDescriptor,
    payload: BudgetedCheckpointBytes,
}

struct StagedResultEpoch {
    index_root: ContentDigest,
    descriptors: BudgetedResultDescriptors,
    payloads: Vec<BudgetedCheckpointBytes>,
    item_count: u64,
    byte_length: u64,
}

/// One conditional object-store generation transaction.
pub struct ObjectGenerationTransaction {
    backend: ObjectCheckpointBackend,
    run: StreamRunIdentity,
    expected: Option<CurrentV4CheckpointGeneration>,
    expectations: CheckpointGenerationExpectations,
    pointer_version: Option<ObjectVersion>,
    _transaction_lease: BudgetLease,
    participants: Vec<StagedParticipant>,
    staged_results: Option<StagedResultEpoch>,
}

impl ObjectGenerationTransaction {
    fn stage_participant_inner(
        &mut self,
        state: PreparedParticipantState,
    ) -> Result<(), CheckpointError> {
        if state.run() != &self.run
            || self.participants.iter().any(|existing| {
                existing.descriptor.participant_id == state.descriptor().participant_id
            })
        {
            return Err(CheckpointError::ObjectVerification);
        }
        let (_, descriptor, payload) = state.into_parts();
        self.participants.push(StagedParticipant {
            descriptor,
            payload,
        });
        Ok(())
    }

    async fn prepare_result_partitions(
        &mut self,
        partitions: &mut Vec<ResultPartition>,
        issue_receipts: &mut Option<PreparedIssueReceiptResultPartition>,
    ) -> Result<PreparedResultEpoch, CheckpointError> {
        if self.staged_results.is_some() {
            return Err(CheckpointError::ObjectVerification);
        }
        let issue_partition = issue_receipts
            .as_ref()
            .map(PreparedIssueReceiptResultPartition::partition);
        if partitions
            .iter()
            .chain(issue_partition)
            .any(|partition| partition.descriptor().run != self.run)
        {
            return Err(CheckpointError::ObjectVerification);
        }
        let staged: Vec<&ResultPartition> = partitions.iter().chain(issue_partition).collect();
        let plan = CheckedResultStagePlan::from_partitions(&staged)?;
        drop(staged);
        let prepared_lease = self
            .backend
            .budgets
            .prepared_indexes
            .acquire(plan.descriptor_items, plan.descriptor_bytes)
            .await?;
        let summary_lease = self
            .backend
            .budgets
            .result_summaries
            .acquire(plan.descriptor_items, plan.descriptor_bytes)
            .await?;
        self.install_result_partitions(
            partitions,
            issue_receipts,
            plan,
            prepared_lease,
            summary_lease,
        )
    }

    fn install_result_partitions(
        &mut self,
        partitions: &mut Vec<ResultPartition>,
        issue_receipts: &mut Option<PreparedIssueReceiptResultPartition>,
        plan: CheckedResultStagePlan,
        prepared_lease: BudgetLease,
        summary_lease: BudgetLease,
    ) -> Result<PreparedResultEpoch, CheckpointError> {
        let issue_partition = issue_receipts
            .as_ref()
            .map(PreparedIssueReceiptResultPartition::partition);
        let prepared_descriptors = partitions
            .iter()
            .chain(issue_partition)
            .map(|partition| partition.descriptor().clone())
            .collect::<Vec<_>>()
            .into_boxed_slice();
        let summary_descriptors = prepared_descriptors.to_vec().into_boxed_slice();
        let prepared_descriptors =
            BudgetedResultDescriptors::new(prepared_descriptors, prepared_lease)?;
        let summary_descriptors =
            BudgetedResultDescriptors::new(summary_descriptors, summary_lease)?;
        let (issue_payload_partition, binding) = match issue_receipts.take() {
            Some(handoff) => {
                let (partition, binding) = handoff.into_staged_parts(plan.index_root);
                (Some(partition), Some(binding))
            }
            None => (None, None),
        };
        let prepared_summary = PreparedResultEpoch::new(
            plan.index_root,
            summary_descriptors,
            plan.item_count,
            plan.byte_length,
            binding,
        )?;

        // Publication fence: every fallible construction above has succeeded, so
        // the drain below cannot fail and cannot leave a partial epoch.
        let mut payloads = Vec::with_capacity(plan.descriptor_items);
        for partition in std::mem::take(partitions)
            .into_iter()
            .chain(issue_payload_partition)
        {
            let (budgeted_descriptor, payload) = partition.into_parts();
            let (input_descriptor, input_lease) = budgeted_descriptor.into_backend_parts();
            payloads.push(payload);
            drop(input_descriptor);
            drop(input_lease);
        }
        self.staged_results = Some(StagedResultEpoch {
            index_root: plan.index_root,
            descriptors: prepared_descriptors,
            payloads,
            item_count: plan.item_count,
            byte_length: plan.byte_length,
        });
        Ok(prepared_summary)
    }

    async fn commit_inner(
        self,
        metadata: CheckpointCommitMetadata,
    ) -> Result<CommittedCheckpointGeneration, CheckpointError> {
        let Self {
            backend,
            run,
            expected,
            expectations,
            pointer_version,
            _transaction_lease,
            participants,
            staged_results,
        } = self;
        // Shared lineage/candidate prevalidation runs before the first provider
        // call: a refused lineage must never leave an uploaded object behind.
        let validated = validate_commit_metadata(&expected, metadata)?;
        let epoch = validated.epoch();
        let results = staged_results
            .as_ref()
            .ok_or(CheckpointError::ObjectVerification)?;
        let mut participant_descriptors = participants
            .iter()
            .map(|participant| participant.descriptor.clone())
            .collect::<Vec<_>>();
        participant_descriptors
            .sort_unstable_by(|left, right| left.participant_id.cmp(&right.participant_id));
        if participant_descriptors
            .iter()
            .map(|descriptor| &descriptor.participant_id)
            .ne(expectations.participant_plan.ids().iter())
        {
            return Err(CheckpointError::ParticipantSetMismatch);
        }
        if results
            .descriptors
            .descriptors()
            .iter()
            .any(|descriptor| descriptor.run != run || descriptor.epoch != epoch)
        {
            return Err(CheckpointError::ObjectVerification);
        }
        let (result_items, result_bytes) = result_totals(results.descriptors.descriptors())?;
        if result_items != results.item_count || result_bytes != results.byte_length {
            return Err(CheckpointError::ObjectVerification);
        }
        let frozen = FrozenGenerationTransactionInputs::new(
            run,
            expectations,
            participant_descriptors,
            results.index_root,
        );
        let prevalidated: PrevalidatedCheckpointGenerationCandidate =
            build_prevalidated_candidate(frozen, validated)?;
        let generation_bytes = prevalidated.encode_for_storage()?;
        let (index_root, index_bytes) =
            canonical_result_index_object(results.descriptors.descriptors().iter())?;
        if index_root != results.index_root {
            return Err(CheckpointError::ObjectVerification);
        }

        // Every immutable object is written and verified before the single
        // pointer replacement below.
        let mut written = BTreeSet::new();
        for participant in &participants {
            if written.insert(participant.descriptor.content_digest) {
                backend
                    .upload_object(
                        participant.descriptor.content_digest,
                        Bytes::copy_from_slice(participant.payload.as_bytes()),
                    )
                    .await?;
            }
        }
        for (descriptor, payload) in results
            .descriptors
            .descriptors()
            .iter()
            .zip(&results.payloads)
        {
            if written.insert(descriptor.payload_digest) {
                backend
                    .upload_object(
                        descriptor.payload_digest,
                        Bytes::copy_from_slice(payload.as_bytes()),
                    )
                    .await?;
            }
        }
        backend
            .upload_object(index_root, Bytes::from(index_bytes.into_boxed_slice()))
            .await?;
        let generation_digest = *prevalidated.generation().digest();
        let generation_length =
            u64::try_from(generation_bytes.len()).map_err(|_| provider_error("object length"))?;
        let generation_version = backend
            .upload_object(generation_digest, generation_bytes)
            .await?;

        let document = CheckpointPointerDocument {
            run,
            storage_version: PointerStorageVersion::CurrentV4,
            generation: prevalidated.generation().clone(),
            generation_object: immutable_object_key(&backend.prefix, &generation_digest),
            generation_version,
            generation_byte_length: generation_length,
        };
        backend
            .write_pointer(&run, pointer_version.as_ref(), document)
            .await?;
        Ok(prevalidated.into_committed_after_publication_fence())
    }
}

#[async_trait(?Send)]
impl StreamingGenerationTransaction for ObjectGenerationTransaction {
    async fn stage_participant(
        &mut self,
        state: PreparedParticipantState,
    ) -> Result<(), CheckpointError> {
        self.stage_participant_inner(state)
    }

    async fn stage_results(
        &mut self,
        partitions: &mut Vec<ResultPartition>,
        issue_receipts: &mut Option<PreparedIssueReceiptResultPartition>,
    ) -> Result<PreparedResultEpoch, CheckpointError> {
        self.prepare_result_partitions(partitions, issue_receipts)
            .await
    }

    async fn commit(
        self: Box<Self>,
        metadata: CheckpointCommitMetadata,
    ) -> Result<CommittedCheckpointGeneration, CheckpointError> {
        (*self).commit_inner(metadata).await
    }
}

struct CheckedResultStagePlan {
    descriptor_items: usize,
    descriptor_bytes: usize,
    index_root: ContentDigest,
    item_count: u64,
    byte_length: u64,
}

impl CheckedResultStagePlan {
    fn from_partitions(partitions: &[&ResultPartition]) -> Result<Self, CheckpointError> {
        let descriptor_bytes = partitions.iter().try_fold(0usize, |total, partition| {
            total
                .checked_add(descriptor_retained_bytes(partition.descriptor())?)
                .ok_or(CheckpointError::ObjectVerification)
        })?;
        let (item_count, byte_length) =
            partitions
                .iter()
                .try_fold((0u64, 0u64), |(items, bytes), partition| {
                    let descriptor = partition.descriptor();
                    Ok((
                        items
                            .checked_add(descriptor.item_count)
                            .ok_or(CheckpointError::ObjectVerification)?,
                        bytes
                            .checked_add(descriptor.byte_length)
                            .ok_or(CheckpointError::ObjectVerification)?,
                    ))
                })?;
        let (index_root, _) = canonical_result_index_object(
            partitions.iter().copied().map(ResultPartition::descriptor),
        )?;
        Ok(Self {
            descriptor_items: partitions.len(),
            descriptor_bytes,
            index_root,
            item_count,
            byte_length,
        })
    }
}

/// Shared reachable-result authority for both storage versions.
struct ObjectResultReadAuthority<'a> {
    backend: &'a ObjectCheckpointBackend,
    run: &'a StreamRunIdentity,
    result_index_root: &'a ContentDigest,
}

impl ObjectResultReadAuthority<'_> {
    async fn reachable_descriptors(&self) -> Result<Vec<ResultSegmentDescriptor>, CheckpointError> {
        let object = self
            .backend
            .read_object_by_digest(*self.result_index_root)
            .await?;
        let descriptors: Vec<ResultSegmentDescriptor> =
            serde_json::from_slice(object.as_bytes())
                .map_err(|_| CheckpointError::ObjectVerification)?;
        if canonical_result_index_root(&descriptors)? != *self.result_index_root
            || descriptors
                .iter()
                .any(|descriptor| descriptor.run != *self.run)
        {
            return Err(CheckpointError::ObjectVerification);
        }
        Ok(descriptors)
    }

    async fn scan_result_index(
        &self,
        after: Option<ResultIndexCursor>,
        budget: ResultIndexReadBudget,
    ) -> Result<ResultIndexPage, CheckpointError> {
        let root = *self.result_index_root;
        if after
            .as_ref()
            .is_some_and(|cursor| cursor.root != root || cursor.block != root)
        {
            return Err(CheckpointError::ObjectVerification);
        }
        let descriptors = self.reachable_descriptors().await?;
        let start = match after.as_ref() {
            None => 0usize,
            Some(cursor) => {
                let offset = usize::try_from(cursor.item_offset)
                    .map_err(|_| CheckpointError::ObjectVerification)?;
                if offset >= descriptors.len() {
                    return Err(CheckpointError::ObjectVerification);
                }
                offset
            }
        };
        if start == descriptors.len() {
            let lease = self.backend.budgets.reads.acquire(0, 0).await?;
            return ResultIndexPage::new(
                BudgetedResultDescriptors::new(Vec::new().into_boxed_slice(), lease)?,
                None,
            );
        }
        let first_required = descriptor_retained_bytes(&descriptors[start])?;
        let first_required_u64 =
            u64::try_from(first_required).map_err(|_| CheckpointError::ObjectVerification)?;
        if first_required_u64 > budget.max_bytes.get() {
            return Err(CheckpointError::ResultIndexReadBudgetTooSmall {
                required_bytes: first_required_u64,
                max_bytes: budget.max_bytes.get(),
            });
        }
        let mut end = start;
        let mut retained = 0usize;
        while end < descriptors.len() && end - start < budget.max_items.get() {
            let next = descriptor_retained_bytes(&descriptors[end])?;
            let Some(total) = retained.checked_add(next) else {
                return Err(CheckpointError::ObjectVerification);
            };
            if u64::try_from(total).map_err(|_| CheckpointError::ObjectVerification)?
                > budget.max_bytes.get()
            {
                break;
            }
            retained = total;
            end += 1;
        }
        let lease = self
            .backend
            .budgets
            .reads
            .acquire(end - start, retained)
            .await?;
        let page_descriptors = descriptors
            .get(start..end)
            .ok_or(CheckpointError::ObjectVerification)?
            .to_vec()
            .into_boxed_slice();
        let next = if end < descriptors.len() {
            Some(ResultIndexCursor {
                root,
                block: root,
                item_offset: u32::try_from(end).map_err(|_| CheckpointError::ObjectVerification)?,
            })
        } else {
            None
        };
        ResultIndexPage::new(
            BudgetedResultDescriptors::new(page_descriptors, lease)?,
            next,
        )
    }

    async fn read_segment(
        &self,
        descriptor: &ResultSegmentDescriptor,
    ) -> Result<ResultSegmentReader, CheckpointError> {
        if !self.reachable_descriptors().await?.contains(descriptor) {
            return Err(CheckpointError::ObjectVerification);
        }
        let payload = self
            .backend
            .read_object_by_digest(descriptor.payload_digest)
            .await?;
        ResultSegmentReader::new(descriptor, payload)
    }
}

/// Leased reader for one committed current-v4 object-store generation.
pub struct ObjectGenerationReader {
    backend: ObjectCheckpointBackend,
    generation: CommittedCheckpointGeneration,
    _generation_lease: BudgetLease,
}

impl ObjectGenerationReader {
    fn result_authority(&self) -> ObjectResultReadAuthority<'_> {
        ObjectResultReadAuthority {
            backend: &self.backend,
            run: self.generation.run(),
            result_index_root: self.generation.result_index_root(),
        }
    }
}

impl sealed::LeasedGenerationReader for ObjectGenerationReader {}

#[async_trait(?Send)]
impl LeasedGenerationReader for ObjectGenerationReader {
    fn generation(&self) -> &CommittedCheckpointGeneration {
        &self.generation
    }

    async fn scan_result_index(
        &self,
        after: Option<ResultIndexCursor>,
        budget: ResultIndexReadBudget,
    ) -> Result<ResultIndexPage, CheckpointError> {
        self.result_authority()
            .scan_result_index(after, budget)
            .await
    }

    async fn read_segment(
        &self,
        descriptor: &ResultSegmentDescriptor,
    ) -> Result<ResultSegmentReader, CheckpointError> {
        self.result_authority().read_segment(descriptor).await
    }

    async fn read_participant(
        &self,
        descriptor: &ParticipantStateDescriptor,
    ) -> Result<CommittedParticipantState, CheckpointError> {
        if !self
            .generation
            .participant_descriptors()
            .contains(descriptor)
        {
            return Err(CheckpointError::ObjectVerification);
        }
        let payload = self
            .backend
            .read_object_by_digest(descriptor.content_digest)
            .await?;
        let context = CurrentV4ParticipantStateContext::for_reachable_descriptor(
            &self.generation,
            descriptor,
        )?;
        if context.generation() != self.generation.generation_ref() {
            return Err(CheckpointError::ObjectVerification);
        }
        CommittedParticipantState::from_current_v4_reader(&context, descriptor.clone(), payload)
    }
}

/// Leased read/export authority for one verified legacy-v3 object-store head.
pub struct ObjectLegacyV3GenerationReader {
    backend: ObjectCheckpointBackend,
    generation: LegacyV3CheckpointGeneration,
    _generation_lease: BudgetLease,
}

impl ObjectLegacyV3GenerationReader {
    fn result_authority(&self) -> ObjectResultReadAuthority<'_> {
        ObjectResultReadAuthority {
            backend: &self.backend,
            run: self.generation.run(),
            result_index_root: self.generation.result_index_root(),
        }
    }
}

#[async_trait(?Send)]
impl LegacyV3LeasedGenerationReader for ObjectLegacyV3GenerationReader {
    fn generation(&self) -> &CheckpointGeneration {
        self.generation.generation()
    }

    async fn scan_result_index(
        &self,
        after: Option<ResultIndexCursor>,
        budget: ResultIndexReadBudget,
    ) -> Result<ResultIndexPage, CheckpointError> {
        self.result_authority()
            .scan_result_index(after, budget)
            .await
    }

    async fn read_segment(
        &self,
        descriptor: &ResultSegmentDescriptor,
    ) -> Result<ResultSegmentReader, CheckpointError> {
        self.result_authority().read_segment(descriptor).await
    }

    async fn read_legacy_participant(
        &self,
        descriptor: &ParticipantStateDescriptor,
    ) -> Result<LegacyParticipantState, CheckpointError> {
        if !self
            .generation
            .participant_descriptors()
            .contains(descriptor)
        {
            return Err(CheckpointError::ObjectVerification);
        }
        let payload = self
            .backend
            .read_object_by_digest(descriptor.content_digest)
            .await?;
        LegacyParticipantState::from_legacy_v3_reader(descriptor.clone(), payload)
    }
}

/// Build one ordinary provider failure that left the pointer unchanged.
#[must_use]
pub fn provider_error(context: &str) -> CheckpointError {
    CheckpointError::Storage {
        message: format!("object checkpoint store: {context}"),
    }
}

/// Build the exact capability-disagreement refusal.
#[must_use]
pub fn conditional_write_unsupported_error() -> CheckpointError {
    CheckpointError::Storage {
        message: CONDITIONAL_WRITE_UNSUPPORTED_MESSAGE.to_string(),
    }
}

/// Build the exact stale-writer refusal.
#[must_use]
pub fn stale_writer_error() -> CheckpointError {
    CheckpointError::Storage {
        message: STALE_WRITER_MESSAGE.to_string(),
    }
}

/// Object-limit refusal used before any allocation or provider byte transfer.
#[must_use]
pub fn object_limit_exceeded_error() -> CheckpointError {
    backend_error(
        CheckpointBackendBudgetKind::Read,
        CheckpointBackendBudgetFailureCode::ByteCapacity,
    )
}

/// Retention view of every immutable object retained under one prefix.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ObjectRetentionInventory {
    entries: BTreeMap<ObjectKey, ObjectVersion>,
}

impl ObjectRetentionInventory {
    /// Build one inventory from exact listed metadata.
    #[must_use]
    pub fn from_metadata(metadata: impl IntoIterator<Item = ObjectMetadata>) -> Self {
        Self {
            entries: metadata
                .into_iter()
                .map(|entry| (entry.key, entry.version))
                .collect(),
        }
    }

    /// Return the exact retained object count.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Return whether nothing is retained.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}
