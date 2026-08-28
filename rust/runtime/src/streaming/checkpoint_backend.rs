// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Atomic generation backend contract and shared publication prevalidation.

use std::{
    any::Any,
    collections::BTreeMap,
    mem::size_of,
    num::{NonZeroU64, NonZeroUsize},
    rc::Rc,
    sync::atomic::{AtomicU64, Ordering},
};

use async_trait::async_trait;
use bytes::Bytes;
use serde::{Deserialize, Serialize};
use serde_json::value::RawValue;

use crate::clock::Clock;

use super::{
    budget::{BudgetLease, StreamingResourceBudget},
    checkpoint::{
        CheckpointCut, CheckpointEpoch, CheckpointError, CheckpointGeneration,
        CheckpointGenerationCandidate, CheckpointParticipantPlan, CheckpointTerminalReason,
        CommittedCheckpointGeneration, CommittedParticipantState, DecodedCheckpointGeneration,
        LegacyParticipantState, LegacyV3CheckpointGeneration, ParticipantStateDescriptor,
        PreparedParticipantState, PrevalidatedCheckpointGenerationCandidate, StreamRunIdentity,
        decode_versioned_checkpoint_generation,
    },
    identity::ContentDigest,
    reliability::PreparedIssueReceiptResultPartition,
    results::{
        PreparedResultEpoch, ResultIndexCursor, ResultIndexPage, ResultIndexReadBudget,
        ResultPartition, ResultSegmentDescriptor, ResultSegmentReader,
        canonical_result_index_object,
    },
};

const INITIAL_CHECKPOINT_EPOCH: CheckpointEpoch = CheckpointEpoch::new(1);

/// Immutable registry metadata for one checkpoint backend implementation.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct StreamingCheckpointBackendDescriptor {
    /// Stable registry identifier.
    pub id: &'static str,
    /// Human-readable implementation description.
    pub description: &'static str,
    /// Whether the backend durably publishes atomic generations.
    pub is_durable: bool,
    /// Whether opened generations remain reachable through explicit read leases.
    pub has_leased_readers: bool,
    /// Whether participant and result state publish as one atomic generation.
    pub has_atomic_generations: bool,
    /// Whether checkpoint-native result segments are supported.
    pub has_result_segments: bool,
    /// Whether sensitive participant state is protected at rest.
    pub protects_sensitive_state: bool,
    /// Reachability and collection policy for committed objects.
    pub retention: CheckpointRetention,
    /// Backend placement visible to cellular validation.
    pub placement: CheckpointBackendPlacement,
    /// Whether backend progress can run under a virtual clock.
    pub supports_virtual_clock: bool,
}

/// Committed checkpoint object retention behavior.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CheckpointRetention {
    /// State is process-local and retained only while its run is active.
    Ephemeral,
    /// Objects remain reachable through committed generation roots.
    GenerationReachability,
}

/// Checkpoint backend placement behavior.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CheckpointBackendPlacement {
    /// Backend state is local to the controller process.
    ControllerLocal,
    /// One authoritative backend is reachable across cells.
    SharedAcrossCells,
}

/// Capabilities required from a selected checkpoint backend.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CheckpointBackendRequirements {
    /// Whether execution must resume after process replacement.
    pub needs_restartable_execution: bool,
    /// Whether partial results must survive process replacement.
    pub needs_durable_partial_results: bool,
    /// Whether retained participant state carries live target output and so
    /// must be protected at rest by the selected backend.
    pub needs_sensitive_state_protection: bool,
}

/// Type-erased, strictly validated checkpoint backend configuration.
pub trait ValidatedCheckpointBackendConfig: std::fmt::Debug + Send + Sync {
    /// Borrow the concrete startup-only value.
    fn as_any(&self) -> &dyn Any;

    /// Consume the concrete startup-only value.
    fn into_any(self: Box<Self>) -> Box<dyn Any + Send + Sync>;
}

impl<T> ValidatedCheckpointBackendConfig for T
where
    T: Any + std::fmt::Debug + Send + Sync,
{
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn into_any(self: Box<Self>) -> Box<dyn Any + Send + Sync> {
        self
    }
}

/// Host-owned checkpoint backend preparation context.
#[derive(Clone)]
pub struct CheckpointBackendPrepareContext {
    /// Exact stable logical run namespace.
    pub run: StreamRunIdentity,
    /// Run clock every lease deadline and expiry check is read from.
    ///
    /// A backend must never call `Instant::now`: lease expiry has to advance
    /// with virtual time so a simulated run reaches the same generations.
    pub clock: Rc<dyn Clock>,
}

// `Clock` has no `Debug` supertrait, so the derived `Debug` is replaced with one
// that reports the clock's discipline rather than its identity.
impl std::fmt::Debug for CheckpointBackendPrepareContext {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("CheckpointBackendPrepareContext")
            .field("run", &self.run)
            .field("is_virtual_clock", &self.clock.is_virtual())
            .finish()
    }
}

/// Startup checkpoint backend validation and preparation contract.
pub trait StreamingCheckpointBackendFactory: std::fmt::Debug + Send + Sync {
    /// Describe the exact compiled backend implementation.
    fn descriptor(&self) -> &'static StreamingCheckpointBackendDescriptor;

    /// Strictly decode and validate backend-owned configuration.
    fn validate(
        &self,
        authored: &RawValue,
        requirements: &CheckpointBackendRequirements,
    ) -> Result<Box<dyn ValidatedCheckpointBackendConfig>, CheckpointError>;

    /// Prepare one run-bound atomic checkpoint backend.
    fn prepare(
        &self,
        config: Box<dyn ValidatedCheckpointBackendConfig>,
        context: &CheckpointBackendPrepareContext,
    ) -> Result<Box<dyn StreamingCheckpointBackend>, CheckpointError>;
}

/// Caller-supplied semantic metadata for one atomic generation publication.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CheckpointCommitMetadata {
    /// Complete predecessor expected by the writer.
    pub previous: Option<CheckpointGeneration>,
    /// Exact successor epoch being committed.
    pub epoch: CheckpointEpoch,
    /// Complete checkpoint cut represented by every participant.
    pub cut: CheckpointCut,
    /// Semantic digest of the execution plan.
    pub execution_plan_digest: ContentDigest,
    /// Semantic digest of the result plan.
    pub result_plan_digest: ContentDigest,
    /// Whether this generation terminates the logical run.
    pub is_final: bool,
    /// Terminal reason, present exactly for final generations.
    pub terminal_reason: Option<CheckpointTerminalReason>,
}

/// Exact run and semantic plan expected while opening or writing generations.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CheckpointGenerationExpectations {
    /// Logical run being opened or written.
    pub run: StreamRunIdentity,
    /// Frozen exact participant inventory.
    pub participant_plan: CheckpointParticipantPlan,
    /// Frozen execution-plan semantic digest.
    pub execution_plan_digest: ContentDigest,
    /// Frozen result-plan semantic digest.
    pub result_plan_digest: ContentDigest,
}

/// Atomic streaming checkpoint storage backend.
#[async_trait(?Send)]
pub trait StreamingCheckpointBackend {
    /// Open and verify the latest authoritative generation for one exact run.
    async fn open_latest(
        &self,
        run: &StreamRunIdentity,
        expected: &CheckpointGenerationExpectations,
    ) -> Result<Option<LeasedCheckpointGeneration>, CheckpointError>;

    /// Begin a transaction frozen to one exact run, head, and semantic plan.
    async fn begin_generation(
        &self,
        run: StreamRunIdentity,
        expected: Option<CurrentV4CheckpointGeneration>,
        expectations: CheckpointGenerationExpectations,
    ) -> Result<Box<dyn StreamingGenerationTransaction>, CheckpointError>;
}

pub(crate) mod sealed {
    pub trait LeasedGenerationReader {}
    pub trait VersionedLeasedGenerationReader {}
}

/// Move-only authority to follow one exact verified current-v4 head.
///
/// The authority cannot be cloned and must be moved into generation begin:
///
/// ```compile_fail
/// # use aiperf_runtime::streaming::checkpoint_backend::CurrentV4CheckpointGeneration;
/// # fn cannot_clone(authority: CurrentV4CheckpointGeneration) {
/// let _second = authority.clone();
/// # }
/// ```
#[derive(Debug, Eq, PartialEq)]
pub struct CurrentV4CheckpointGeneration(CheckpointGeneration);

impl CurrentV4CheckpointGeneration {
    /// Borrow the exact immutable predecessor identity.
    #[must_use]
    pub const fn generation(&self) -> &CheckpointGeneration {
        &self.0
    }
}

/// Read authority scoped to the reachable objects of one committed generation.
#[async_trait(?Send)]
pub trait LeasedGenerationReader: sealed::LeasedGenerationReader {
    /// Borrow the authoritative committed generation.
    fn generation(&self) -> &CommittedCheckpointGeneration;

    /// Mint move-only successor authority after comparing the retained identity.
    fn current_v4_predecessor(
        &self,
        expected: &CheckpointGeneration,
    ) -> Result<CurrentV4CheckpointGeneration, CheckpointError> {
        let actual = self.generation().generation();
        if &actual != expected {
            return Err(CheckpointError::GenerationConflict {
                expected: Some(expected.clone()),
                actual: Some(actual),
            });
        }
        Ok(CurrentV4CheckpointGeneration(actual))
    }

    /// Scan reachable result descriptors under caller and backend budgets.
    async fn scan_result_index(
        &self,
        after: Option<ResultIndexCursor>,
        budget: ResultIndexReadBudget,
    ) -> Result<ResultIndexPage, CheckpointError>;

    /// Read one exact result segment reachable from this generation.
    async fn read_segment(
        &self,
        descriptor: &ResultSegmentDescriptor,
    ) -> Result<ResultSegmentReader, CheckpointError>;

    /// Read one exact participant object reachable from this generation.
    async fn read_participant(
        &self,
        descriptor: &ParticipantStateDescriptor,
    ) -> Result<CommittedParticipantState, CheckpointError>;
}

/// Read/export authority for one verified legacy-v3 generation.
///
/// Legacy readers deliberately have no current-v4 predecessor accessor:
///
/// ```compile_fail
/// # use aiperf_runtime::streaming::checkpoint::CheckpointGeneration;
/// # use aiperf_runtime::streaming::checkpoint_backend::LegacyV3LeasedGenerationReader;
/// # fn cannot_follow(
/// #     reader: &dyn LegacyV3LeasedGenerationReader,
/// #     expected: &CheckpointGeneration,
/// # ) {
/// let _ = reader.current_v4_predecessor(expected);
/// # }
/// ```
#[async_trait(?Send)]
pub trait LegacyV3LeasedGenerationReader {
    /// Borrow the strictly decoded semantic generation identity.
    fn generation(&self) -> &CheckpointGeneration;

    /// Scan result descriptors reachable from this exact legacy generation.
    async fn scan_result_index(
        &self,
        after: Option<ResultIndexCursor>,
        budget: ResultIndexReadBudget,
    ) -> Result<ResultIndexPage, CheckpointError>;

    /// Read one result payload reachable from this exact legacy generation.
    async fn read_segment(
        &self,
        descriptor: &ResultSegmentDescriptor,
    ) -> Result<ResultSegmentReader, CheckpointError>;

    /// Read one participant for borrow-only export, never initialization.
    async fn read_legacy_participant(
        &self,
        descriptor: &ParticipantStateDescriptor,
    ) -> Result<LegacyParticipantState, CheckpointError>;
}

/// Storage version of one leased authoritative head.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CheckpointGenerationStorageVersion {
    /// Explicit current-v4 generation with successor authority available.
    CurrentV4,
    /// Strictly verified legacy-v3 generation available only for reads/export.
    LegacyV3ReadOnly,
}

enum LeasedCheckpointGenerationInner {
    CurrentV4(Box<dyn LeasedGenerationReader>),
    LegacyV3ReadOnly(Box<dyn LegacyV3LeasedGenerationReader>),
}

/// Opaque versioned leased generation returned by backend open.
///
/// This wrapper is read authority, not successor authority:
///
/// ```compile_fail
/// # use aiperf_runtime::streaming::checkpoint::StreamRunIdentity;
/// # use aiperf_runtime::streaming::checkpoint_backend::{
/// #     CheckpointGenerationExpectations, LeasedCheckpointGeneration,
/// #     StreamingCheckpointBackend,
/// # };
/// # async fn cannot_succeed(
/// #     backend: &dyn StreamingCheckpointBackend,
/// #     run: StreamRunIdentity,
/// #     opened: LeasedCheckpointGeneration,
/// #     expectations: CheckpointGenerationExpectations,
/// # ) {
/// let _ = backend.begin_generation(run, Some(opened), expectations).await;
/// # }
/// ```
pub struct LeasedCheckpointGeneration(LeasedCheckpointGenerationInner);

/// Borrowed exhaustive view of current-v4 versus legacy-v3 authority.
pub enum LeasedCheckpointGenerationView<'a> {
    /// Current-v4 reader that alone can mint successor and participant authority.
    CurrentV4(&'a dyn LeasedGenerationReader),
    /// Legacy-v3 reader restricted to borrow-only state and result reads.
    LegacyV3ReadOnly(&'a dyn LegacyV3LeasedGenerationReader),
}

impl LeasedCheckpointGeneration {
    pub(crate) fn current_v4(reader: impl LeasedGenerationReader + 'static) -> Self {
        Self(LeasedCheckpointGenerationInner::CurrentV4(Box::new(reader)))
    }

    pub(crate) fn legacy_v3(reader: impl LegacyV3LeasedGenerationReader + 'static) -> Self {
        Self(LeasedCheckpointGenerationInner::LegacyV3ReadOnly(Box::new(
            reader,
        )))
    }

    /// Return the explicitly verified storage version.
    #[must_use]
    pub const fn version(&self) -> CheckpointGenerationStorageVersion {
        match &self.0 {
            LeasedCheckpointGenerationInner::CurrentV4(_) => {
                CheckpointGenerationStorageVersion::CurrentV4
            }
            LeasedCheckpointGenerationInner::LegacyV3ReadOnly(_) => {
                CheckpointGenerationStorageVersion::LegacyV3ReadOnly
            }
        }
    }

    /// Borrow the common immutable generation identity.
    #[must_use]
    pub fn generation(&self) -> &CheckpointGeneration {
        match &self.0 {
            LeasedCheckpointGenerationInner::CurrentV4(reader) => {
                reader.generation().generation_ref()
            }
            LeasedCheckpointGenerationInner::LegacyV3ReadOnly(reader) => reader.generation(),
        }
    }

    /// Borrow the branch-specific reader authority.
    #[must_use]
    pub fn view(&self) -> LeasedCheckpointGenerationView<'_> {
        match &self.0 {
            LeasedCheckpointGenerationInner::CurrentV4(reader) => {
                LeasedCheckpointGenerationView::CurrentV4(reader.as_ref())
            }
            LeasedCheckpointGenerationInner::LegacyV3ReadOnly(reader) => {
                LeasedCheckpointGenerationView::LegacyV3ReadOnly(reader.as_ref())
            }
        }
    }
}

/// Sealed common result-reading surface for versioned leased generations.
#[async_trait(?Send)]
pub trait VersionedLeasedGenerationReader: sealed::VersionedLeasedGenerationReader {
    /// Return the explicit verified storage version.
    fn version(&self) -> CheckpointGenerationStorageVersion;
    /// Borrow the common immutable generation identity.
    fn generation(&self) -> &CheckpointGeneration;
    /// Borrow branch-specific participant and predecessor authority.
    fn view(&self) -> LeasedCheckpointGenerationView<'_>;
    /// Scan common result-index authority.
    async fn scan_result_index(
        &self,
        after: Option<ResultIndexCursor>,
        budget: ResultIndexReadBudget,
    ) -> Result<ResultIndexPage, CheckpointError>;
    /// Read one reachable result payload.
    async fn read_segment(
        &self,
        descriptor: &ResultSegmentDescriptor,
    ) -> Result<ResultSegmentReader, CheckpointError>;
}

impl sealed::VersionedLeasedGenerationReader for LeasedCheckpointGeneration {}

#[async_trait(?Send)]
impl VersionedLeasedGenerationReader for LeasedCheckpointGeneration {
    fn version(&self) -> CheckpointGenerationStorageVersion {
        self.version()
    }

    fn generation(&self) -> &CheckpointGeneration {
        self.generation()
    }

    fn view(&self) -> LeasedCheckpointGenerationView<'_> {
        self.view()
    }

    async fn scan_result_index(
        &self,
        after: Option<ResultIndexCursor>,
        budget: ResultIndexReadBudget,
    ) -> Result<ResultIndexPage, CheckpointError> {
        match &self.0 {
            LeasedCheckpointGenerationInner::CurrentV4(reader) => {
                reader.scan_result_index(after, budget).await
            }
            LeasedCheckpointGenerationInner::LegacyV3ReadOnly(reader) => {
                reader.scan_result_index(after, budget).await
            }
        }
    }

    async fn read_segment(
        &self,
        descriptor: &ResultSegmentDescriptor,
    ) -> Result<ResultSegmentReader, CheckpointError> {
        match &self.0 {
            LeasedCheckpointGenerationInner::CurrentV4(reader) => {
                reader.read_segment(descriptor).await
            }
            LeasedCheckpointGenerationInner::LegacyV3ReadOnly(reader) => {
                reader.read_segment(descriptor).await
            }
        }
    }
}

/// Explicit authored bounds for one checked legacy-v3 integration fixture.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LegacyV3FixtureLimits {
    /// Maximum immutable objects in the complete fixture.
    pub max_objects: NonZeroUsize,
    /// Maximum aggregate encoded object bytes in the complete fixture.
    pub max_bytes: NonZeroU64,
}

static NEXT_LEGACY_FIXTURE_ID: AtomicU64 = AtomicU64::new(1);

/// Whole-fixture atomic precharge required before any legacy bytes are copied.
#[doc(hidden)]
#[derive(Debug)]
pub struct LegacyV3FixturePrecharge {
    fixture_id: u64,
    remaining_objects: usize,
    remaining_encoded_bytes: usize,
    remaining_inventory_objects: usize,
    lease: BudgetLease,
}

impl LegacyV3FixturePrecharge {
    /// Atomically precharge all payload and boxed-inventory retention.
    pub async fn acquire(
        budget: &StreamingResourceBudget,
        limits: LegacyV3FixtureLimits,
        exact_objects: usize,
        exact_encoded_bytes: usize,
    ) -> Result<Self, CheckpointError> {
        let max_bytes = usize::try_from(limits.max_bytes.get())
            .map_err(|_| CheckpointError::ObjectVerification)?;
        if exact_objects < 2
            || exact_objects > limits.max_objects.get()
            || exact_encoded_bytes > max_bytes
        {
            return Err(CheckpointError::ObjectVerification);
        }
        let inventory_objects = exact_objects - 2;
        let inventory_bytes = inventory_objects
            .checked_mul(size_of::<LegacyV3FixtureObject>())
            .ok_or(CheckpointError::ObjectVerification)?;
        let total_bytes = exact_encoded_bytes
            .checked_add(inventory_bytes)
            .ok_or(CheckpointError::ObjectVerification)?;
        let lease = budget
            .acquire(exact_objects, total_bytes)
            .await
            .map_err(|_| CheckpointError::ObjectVerification)?;
        Ok(Self {
            fixture_id: NEXT_LEGACY_FIXTURE_ID.fetch_add(1, Ordering::Relaxed),
            remaining_objects: exact_objects,
            remaining_encoded_bytes: exact_encoded_bytes,
            remaining_inventory_objects: inventory_objects,
            lease,
        })
    }

    /// Split one precharged payload reservation, then compact-copy its bytes.
    pub fn compact_object(
        &mut self,
        digest: ContentDigest,
        encoded: &[u8],
    ) -> Result<LegacyV3FixtureObject, CheckpointError> {
        if self.remaining_objects == 0 || encoded.len() > self.remaining_encoded_bytes {
            return Err(CheckpointError::ObjectVerification);
        }
        let payload_lease = self
            .lease
            .split_off(1, encoded.len())
            .map_err(|_| CheckpointError::ObjectVerification)?;
        self.remaining_objects -= 1;
        self.remaining_encoded_bytes -= encoded.len();
        Ok(LegacyV3FixtureObject {
            fixture_id: self.fixture_id,
            digest,
            encoded: Bytes::from(encoded.to_vec().into_boxed_slice()),
            _payload_lease: payload_lease,
        })
    }

    /// Move precharged objects into one exact boxed inventory.
    pub fn collect_inventory<I>(
        &mut self,
        objects: I,
    ) -> Result<BudgetOwnedLegacyV3FixtureInventory, CheckpointError>
    where
        I: ExactSizeIterator<Item = LegacyV3FixtureObject>,
    {
        let object_count = objects.len();
        if object_count > self.remaining_inventory_objects {
            return Err(CheckpointError::ObjectVerification);
        }
        let structural_bytes = object_count
            .checked_mul(size_of::<LegacyV3FixtureObject>())
            .ok_or(CheckpointError::ObjectVerification)?;
        let structural_lease = self
            .lease
            .split_off(0, structural_bytes)
            .map_err(|_| CheckpointError::ObjectVerification)?;
        let mut retained = Vec::with_capacity(object_count);
        for object in objects {
            if retained.len() == object_count || object.fixture_id != self.fixture_id {
                return Err(CheckpointError::ObjectVerification);
            }
            retained.push(object);
        }
        if retained.len() != object_count {
            return Err(CheckpointError::ObjectVerification);
        }
        self.remaining_inventory_objects -= object_count;
        Ok(BudgetOwnedLegacyV3FixtureInventory {
            fixture_id: self.fixture_id,
            objects: retained.into_boxed_slice(),
            _structural_lease: structural_lease,
        })
    }

    /// Finish a strictly verified, read-only legacy-v3 fixture.
    #[allow(clippy::too_many_arguments)]
    pub fn finish(
        self,
        run: StreamRunIdentity,
        head: CheckpointGeneration,
        generation_object: LegacyV3FixtureObject,
        participant_objects: BudgetOwnedLegacyV3FixtureInventory,
        result_index_object: LegacyV3FixtureObject,
        result_objects: BudgetOwnedLegacyV3FixtureInventory,
    ) -> Result<LegacyV3ReadOnlyFixture, CheckpointError> {
        if self.remaining_objects != 0
            || self.remaining_encoded_bytes != 0
            || self.remaining_inventory_objects != 0
            || self.lease.charged_items() != 0
            || self.lease.charged_bytes() != 0
            || generation_object.fixture_id != self.fixture_id
            || participant_objects.fixture_id != self.fixture_id
            || result_index_object.fixture_id != self.fixture_id
            || result_objects.fixture_id != self.fixture_id
        {
            return Err(CheckpointError::ObjectVerification);
        }
        drop(self.lease);
        let generation = validate_legacy_v3_fixture(
            &run,
            &head,
            &generation_object,
            &participant_objects,
            &result_index_object,
            &result_objects,
        )?;
        Ok(LegacyV3ReadOnlyFixture {
            run,
            head,
            generation,
            generation_object,
            participant_objects,
            result_index_object,
            result_objects,
        })
    }
}

/// One compact immutable object owned by a whole-fixture precharge.
#[doc(hidden)]
#[derive(Debug)]
pub struct LegacyV3FixtureObject {
    fixture_id: u64,
    digest: ContentDigest,
    encoded: Bytes,
    _payload_lease: BudgetLease,
}

impl LegacyV3FixtureObject {
    pub(crate) fn into_storage_parts(self) -> (ContentDigest, Bytes) {
        (self.digest, self.encoded)
    }
}

/// Exact boxed inventory whose structural allocation remains precharged.
#[doc(hidden)]
#[derive(Debug)]
pub struct BudgetOwnedLegacyV3FixtureInventory {
    fixture_id: u64,
    objects: Box<[LegacyV3FixtureObject]>,
    _structural_lease: BudgetLease,
}

impl BudgetOwnedLegacyV3FixtureInventory {
    pub(crate) fn into_objects(self) -> Box<[LegacyV3FixtureObject]> {
        self.objects
    }
}

/// Completely precharged, strictly verified legacy-v3 read-only import.
#[doc(hidden)]
#[derive(Debug)]
pub struct LegacyV3ReadOnlyFixture {
    run: StreamRunIdentity,
    head: CheckpointGeneration,
    generation: LegacyV3CheckpointGeneration,
    generation_object: LegacyV3FixtureObject,
    participant_objects: BudgetOwnedLegacyV3FixtureInventory,
    result_index_object: LegacyV3FixtureObject,
    result_objects: BudgetOwnedLegacyV3FixtureInventory,
}

impl LegacyV3ReadOnlyFixture {
    pub(crate) const fn generation(&self) -> &LegacyV3CheckpointGeneration {
        &self.generation
    }

    pub(crate) fn encoded_object_count(&self) -> usize {
        2 + self.participant_objects.objects.len() + self.result_objects.objects.len()
    }

    pub(crate) fn encoded_byte_count(&self) -> Result<usize, CheckpointError> {
        self.participant_objects
            .objects
            .iter()
            .chain(self.result_objects.objects.iter())
            .fold(
                self.generation_object
                    .encoded
                    .len()
                    .checked_add(self.result_index_object.encoded.len())
                    .ok_or(CheckpointError::ObjectVerification),
                |total, object| {
                    total?
                        .checked_add(object.encoded.len())
                        .ok_or(CheckpointError::ObjectVerification)
                },
            )
    }

    pub(crate) fn into_parts(
        self,
    ) -> (
        StreamRunIdentity,
        CheckpointGeneration,
        LegacyV3FixtureObject,
        BudgetOwnedLegacyV3FixtureInventory,
        LegacyV3FixtureObject,
        BudgetOwnedLegacyV3FixtureInventory,
    ) {
        (
            self.run,
            self.head,
            self.generation_object,
            self.participant_objects,
            self.result_index_object,
            self.result_objects,
        )
    }
}

fn validate_legacy_v3_fixture(
    run: &StreamRunIdentity,
    head: &CheckpointGeneration,
    generation_object: &LegacyV3FixtureObject,
    participant_objects: &BudgetOwnedLegacyV3FixtureInventory,
    result_index_object: &LegacyV3FixtureObject,
    result_objects: &BudgetOwnedLegacyV3FixtureInventory,
) -> Result<LegacyV3CheckpointGeneration, CheckpointError> {
    if generation_object.digest != *head.digest() {
        return Err(CheckpointError::ObjectVerification);
    }
    let generation = match decode_versioned_checkpoint_generation(
        &generation_object.encoded,
        generation_object.encoded.len(),
    )? {
        DecodedCheckpointGeneration::LegacyV3ReadOnly(generation) => generation,
        DecodedCheckpointGeneration::CurrentV4(_) => {
            return Err(CheckpointError::ObjectVerification);
        }
    };
    if generation.run() != run || generation.generation() != head {
        return Err(CheckpointError::ObjectVerification);
    }
    verify_fixture_object_inventory(
        generation.participant_descriptors(),
        &participant_objects.objects,
        |descriptor| (descriptor.content_digest, descriptor.byte_length),
    )?;
    let descriptors: Vec<ResultSegmentDescriptor> =
        serde_json::from_slice(&result_index_object.encoded)
            .map_err(|_| CheckpointError::ObjectVerification)?;
    let (root, canonical) = canonical_result_index_object(descriptors.iter())?;
    if root != *generation.result_index_root()
        || result_index_object.digest != root
        || result_index_object.encoded.as_ref() != canonical
        || descriptors.iter().any(|descriptor| {
            descriptor.run != *run || descriptor.epoch != generation.generation().epoch()
        })
    {
        return Err(CheckpointError::ObjectVerification);
    }
    verify_fixture_object_inventory(&descriptors, &result_objects.objects, |descriptor| {
        (descriptor.payload_digest, descriptor.byte_length)
    })?;
    Ok(generation)
}

fn verify_fixture_object_inventory<T>(
    descriptors: &[T],
    objects: &[LegacyV3FixtureObject],
    identity: impl Fn(&T) -> (ContentDigest, u64),
) -> Result<(), CheckpointError> {
    if descriptors.len() != objects.len() {
        return Err(CheckpointError::ObjectVerification);
    }
    let mut inventory = BTreeMap::new();
    for object in objects {
        if inventory.insert(object.digest, &object.encoded).is_some() {
            return Err(CheckpointError::ObjectVerification);
        }
    }
    for descriptor in descriptors {
        let (digest, length) = identity(descriptor);
        let encoded = inventory
            .remove(&digest)
            .ok_or(CheckpointError::ObjectVerification)?;
        if u64::try_from(encoded.len()).map_err(|_| CheckpointError::ObjectVerification)? != length
            || ContentDigest::from_bytes(*blake3::hash(encoded).as_bytes()) != digest
        {
            return Err(CheckpointError::ObjectVerification);
        }
    }
    if !inventory.is_empty() {
        return Err(CheckpointError::ObjectVerification);
    }
    Ok(())
}

/// One atomic streaming generation transaction.
#[async_trait(?Send)]
pub trait StreamingGenerationTransaction {
    /// Stage one participant's prepared immutable state.
    async fn stage_participant(
        &mut self,
        state: PreparedParticipantState,
    ) -> Result<(), CheckpointError>;

    /// Stage exactly one complete result epoch.
    ///
    /// The optional detailed-receipt partition is staged beside the ordinary
    /// partitions and is taken only when the whole epoch is prepared, so a
    /// refused staging leaves the caller's move-only handoff intact.
    async fn stage_results(
        &mut self,
        partitions: &mut Vec<ResultPartition>,
        issue_receipts: &mut Option<PreparedIssueReceiptResultPartition>,
    ) -> Result<PreparedResultEpoch, CheckpointError>;

    /// Atomically publish the fully staged generation.
    async fn commit(
        self: Box<Self>,
        metadata: CheckpointCommitMetadata,
    ) -> Result<CommittedCheckpointGeneration, CheckpointError>;
}

pub(crate) struct ValidatedCommitMetadata {
    previous_digest: Option<ContentDigest>,
    epoch: CheckpointEpoch,
    metadata: CheckpointCommitMetadata,
}

impl ValidatedCommitMetadata {
    pub(crate) const fn epoch(&self) -> CheckpointEpoch {
        self.epoch
    }
}

pub(crate) fn validate_commit_metadata(
    expected: &Option<CurrentV4CheckpointGeneration>,
    metadata: CheckpointCommitMetadata,
) -> Result<ValidatedCommitMetadata, CheckpointError> {
    let expected_generation = expected
        .as_ref()
        .map(CurrentV4CheckpointGeneration::generation);
    if metadata.previous.as_ref() != expected_generation {
        return Err(CheckpointError::ObjectVerification);
    }
    let epoch = match expected_generation {
        None => INITIAL_CHECKPOINT_EPOCH,
        Some(previous) => {
            CheckpointEpoch::new(previous.epoch().get().checked_add(1).ok_or_else(|| {
                CheckpointError::GenerationEpochOverflow {
                    previous: previous.clone(),
                }
            })?)
        }
    };
    if metadata.epoch != epoch {
        return Err(CheckpointError::ObjectVerification);
    }
    Ok(ValidatedCommitMetadata {
        previous_digest: expected_generation.map(|generation| *generation.digest()),
        epoch,
        metadata,
    })
}

pub(crate) struct FrozenGenerationTransactionInputs {
    run: StreamRunIdentity,
    expectations: CheckpointGenerationExpectations,
    participant_descriptors: Vec<ParticipantStateDescriptor>,
    result_index_root: ContentDigest,
}

impl FrozenGenerationTransactionInputs {
    pub(crate) fn new(
        run: StreamRunIdentity,
        expectations: CheckpointGenerationExpectations,
        participant_descriptors: Vec<ParticipantStateDescriptor>,
        result_index_root: ContentDigest,
    ) -> Self {
        Self {
            run,
            expectations,
            participant_descriptors,
            result_index_root,
        }
    }
}

pub(crate) fn build_prevalidated_candidate(
    transaction: FrozenGenerationTransactionInputs,
    validated: ValidatedCommitMetadata,
) -> Result<PrevalidatedCheckpointGenerationCandidate, CheckpointError> {
    if transaction.run != transaction.expectations.run {
        return Err(CheckpointError::ObjectVerification);
    }
    let ValidatedCommitMetadata {
        previous_digest,
        epoch,
        metadata,
    } = validated;
    CheckpointGenerationCandidate::new(
        transaction.run,
        epoch,
        previous_digest,
        metadata.cut,
        &transaction.expectations.participant_plan,
        metadata.execution_plan_digest,
        metadata.result_plan_digest,
        transaction.participant_descriptors,
        transaction.result_index_root,
        metadata.is_final,
        metadata.terminal_reason,
    )?
    .prevalidate_for_publication(
        &transaction.expectations.run,
        &transaction.expectations.participant_plan,
        &transaction.expectations.execution_plan_digest,
        &transaction.expectations.result_plan_digest,
    )
}
