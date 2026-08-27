// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Typed checkpoint cuts, generation identity, and stable participant contracts.

use std::fmt;

use async_trait::async_trait;
use bytes::Bytes;
use serde::{Deserialize, Serialize};

use super::{
    budget::BudgetLease,
    identity::{ContentDigest, GlobalSequence, SessionCausalFrontier},
    unit::{EventTimeUtc, SourcePosition, StateBudgetFailureCode},
};

/// Stable identity of one stateful checkpoint owner in a frozen run plan.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(transparent)]
pub struct CheckpointParticipantId(String);

impl CheckpointParticipantId {
    /// Construct a participant identity from stable plan-owned text.
    #[must_use]
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    /// Borrow the stable text identity.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

macro_rules! typed_horizon {
    ($(#[$meta:meta])* $name:ident, $inner:ty) => {
        $(#[$meta])*
        #[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
        #[serde(transparent)]
        pub struct $name($inner);

        impl $name {
            /// Construct a horizon in this domain.
            #[must_use]
            pub const fn new(value: $inner) -> Self {
                Self(value)
            }

            /// Borrow the domain-specific horizon value.
            #[must_use]
            pub const fn get(&self) -> &$inner {
                &self.0
            }
        }
    };
}

typed_horizon!(
    /// Greatest source position discovered by the source owner.
    DiscoveryHorizon,
    SourcePosition
);
typed_horizon!(
    /// Greatest source position acquired under immutable source identity.
    AcquisitionHorizon,
    SourcePosition
);
typed_horizon!(
    /// Greatest source position represented by decoder and downstream state.
    DecodeHorizon,
    SourcePosition
);
typed_horizon!(
    /// Greatest action assigned a stable global order.
    OrderedActionHorizon,
    GlobalSequence
);
typed_horizon!(
    /// Greatest globally ordered action admitted for execution.
    AdmissionHorizon,
    GlobalSequence
);
typed_horizon!(
    /// Greatest contiguous action with one authoritative terminal fact.
    TerminalActionHorizon,
    GlobalSequence
);

/// Event-time completeness represented by a checkpoint cut.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum EventTimeWatermark {
    /// The source cannot currently prove event-time completeness.
    Unknown,
    /// No later event can precede this time.
    Hard {
        /// Greatest proven-complete event time.
        through: EventTimeUtc,
    },
    /// Completeness is policy-estimated and bound to the named late-data policy.
    Estimated {
        /// Greatest estimated-complete event time.
        through: EventTimeUtc,
        /// Semantic digest of the late-data policy supporting the estimate.
        late_policy_digest: ContentDigest,
    },
}

/// Complete typed progress cut represented by one checkpoint generation.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CheckpointCut {
    /// Source discovery progress.
    pub discovered: DiscoveryHorizon,
    /// Immutable source acquisition progress.
    pub acquired: AcquisitionHorizon,
    /// Decoder progress represented by retained state.
    pub decoded: DecodeHorizon,
    /// Stable action-order progress.
    pub ordered: OrderedActionHorizon,
    /// Action-admission progress.
    pub admitted: AdmissionHorizon,
    /// Contiguous terminal-action progress.
    pub terminal: TerminalActionHorizon,
    /// Event-time completeness at this cut.
    pub event_watermark: EventTimeWatermark,
    /// Session-causal completeness at this cut.
    pub causal_frontier: SessionCausalFrontier,
}

/// Monotonic checkpoint epoch number.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(transparent)]
pub struct CheckpointEpoch(u64);

impl CheckpointEpoch {
    /// Construct an epoch number.
    #[must_use]
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    /// Return the underlying epoch number.
    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// Content-addressed identity of one checkpoint generation.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CheckpointGeneration {
    /// Monotonic generation epoch.
    pub epoch: CheckpointEpoch,
    /// Digest of the committed canonical generation record.
    pub digest: ContentDigest,
}

impl CheckpointGeneration {
    /// Construct a generation identity.
    #[must_use]
    pub fn new(epoch: CheckpointEpoch, digest: ContentDigest) -> Self {
        Self { epoch, digest }
    }

    /// Return this generation's epoch.
    #[must_use]
    pub const fn epoch(&self) -> CheckpointEpoch {
        self.epoch
    }

    /// Borrow this generation's content digest.
    #[must_use]
    pub const fn digest(&self) -> &ContentDigest {
        &self.digest
    }
}

/// Terminal condition recorded by a final checkpoint generation.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CheckpointTerminalReason {
    /// The stream completed under its authored stopping condition.
    Completed,
    /// Execution aborted because of a terminal failure.
    Aborted,
    /// Execution was cancelled by control policy.
    Cancelled,
}

/// Immutable descriptor for one participant's checkpoint object.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ParticipantStateDescriptor {
    /// Stable owner identity.
    pub participant_id: CheckpointParticipantId,
    /// Participant-owned state schema identity.
    pub schema_id: String,
    /// Participant-owned state schema version.
    pub schema_version: u32,
    /// Greatest complete cut represented by this state object.
    pub represented_cut: CheckpointCut,
    /// BLAKE3 digest of the exact immutable payload bytes.
    pub content_digest: ContentDigest,
    /// Logical item count represented by the payload.
    pub item_count: u64,
    /// Exact payload byte length.
    pub byte_length: u64,
}

impl ParticipantStateDescriptor {
    /// Digest the complete descriptor for a generation-bound commit receipt.
    pub fn digest(&self) -> Result<ContentDigest, CheckpointError> {
        let bytes = serde_json::to_vec(self).map_err(|error| CheckpointError::Storage {
            message: format!("could not encode participant descriptor: {error}"),
        })?;
        Ok(ContentDigest::from_bytes(*blake3::hash(&bytes).as_bytes()))
    }
}

/// Move-only immutable bytes retaining their checkpoint-state budget charge.
///
/// The bytes and lease are intentionally inseparable. Callers can borrow the
/// bytes, but cannot extract either field and leave uncharged retained storage.
///
/// ```compile_fail
/// # use aiperf_runtime::streaming::checkpoint::BudgetedCheckpointBytes;
/// # fn cannot_separate(value: BudgetedCheckpointBytes) {
/// let _bytes = value.bytes;
/// let _lease = value.lease;
/// # }
/// ```
#[derive(Debug)]
pub struct BudgetedCheckpointBytes {
    bytes: Bytes,
    lease: BudgetLease,
}

impl BudgetedCheckpointBytes {
    /// Copy bytes into compact owned storage and bind their exact budget charge.
    pub fn new(bytes: Bytes, lease: BudgetLease) -> Result<Self, CheckpointError> {
        if lease.charged_items() != 1 || lease.charged_bytes() != bytes.len() {
            return Err(CheckpointError::ObjectVerification);
        }
        let bytes = Bytes::from(bytes.as_ref().to_vec().into_boxed_slice());
        Ok(Self { bytes, lease })
    }

    /// Borrow the exact immutable participant payload.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Return the exact compact allocation retained by this payload.
    #[must_use]
    pub fn retained_allocation_bytes(&self) -> usize {
        self.bytes.len()
    }

    /// Return the byte capacity charged for the retained payload.
    #[must_use]
    pub fn charged_bytes(&self) -> usize {
        self.lease.charged_bytes()
    }
}

/// Immutable participant state prepared before generation commit.
///
/// ```compile_fail
/// # use aiperf_runtime::streaming::checkpoint::PreparedParticipantState;
/// # fn cannot_forge_or_separate(value: PreparedParticipantState) {
/// let _descriptor = value.descriptor;
/// let _payload = value.payload;
/// # }
/// ```
#[derive(Debug)]
pub struct PreparedParticipantState {
    descriptor: ParticipantStateDescriptor,
    payload: BudgetedCheckpointBytes,
}

impl PreparedParticipantState {
    /// Construct a prepared state and derive its exact length and BLAKE3 digest.
    pub fn new(
        participant_id: CheckpointParticipantId,
        schema_id: impl Into<String>,
        schema_version: u32,
        represented_cut: CheckpointCut,
        item_count: u64,
        payload: BudgetedCheckpointBytes,
    ) -> Result<Self, CheckpointError> {
        validate_payload_charge(&payload)?;
        let byte_length =
            u64::try_from(payload.bytes.len()).map_err(|_| CheckpointError::ObjectVerification)?;
        let content_digest = digest_bytes(&payload.bytes);
        Ok(Self {
            descriptor: ParticipantStateDescriptor {
                participant_id,
                schema_id: schema_id.into(),
                schema_version,
                represented_cut,
                content_digest,
                item_count,
                byte_length,
            },
            payload,
        })
    }

    /// Borrow the checked descriptor derived from the retained payload.
    #[must_use]
    pub fn descriptor(&self) -> &ParticipantStateDescriptor {
        &self.descriptor
    }

    /// Borrow the exact immutable participant payload.
    #[must_use]
    pub fn payload_bytes(&self) -> &[u8] {
        self.payload.as_bytes()
    }

    /// Move the checked descriptor and inseparable budgeted payload together.
    #[must_use]
    pub fn into_parts(self) -> (ParticipantStateDescriptor, BudgetedCheckpointBytes) {
        (self.descriptor, self.payload)
    }
}

/// Verified participant state restored from one committed generation.
///
/// ```compile_fail
/// # use aiperf_runtime::streaming::checkpoint::CommittedParticipantState;
/// # fn cannot_forge_or_separate(value: CommittedParticipantState) {
/// let _descriptor = value.descriptor;
/// let _payload = value.payload;
/// # }
/// ```
#[derive(Debug)]
pub struct CommittedParticipantState {
    descriptor: ParticipantStateDescriptor,
    payload: BudgetedCheckpointBytes,
}

impl CommittedParticipantState {
    /// Verify exact length, budget ownership, and BLAKE3 before restoration.
    pub fn new(
        descriptor: ParticipantStateDescriptor,
        payload: BudgetedCheckpointBytes,
    ) -> Result<Self, CheckpointError> {
        let state = Self {
            descriptor,
            payload,
        };
        state.verify()
    }

    /// Re-verify a state assembled by a storage implementation before using it.
    pub fn verify(self) -> Result<Self, CheckpointError> {
        validate_payload_charge(&self.payload)?;
        let byte_length = u64::try_from(self.payload.bytes.len())
            .map_err(|_| CheckpointError::ObjectVerification)?;
        if self.descriptor.byte_length != byte_length
            || self.descriptor.content_digest != digest_bytes(&self.payload.bytes)
        {
            return Err(CheckpointError::ObjectVerification);
        }
        Ok(self)
    }

    /// Borrow the verified committed descriptor.
    #[must_use]
    pub fn descriptor(&self) -> &ParticipantStateDescriptor {
        &self.descriptor
    }

    /// Borrow the verified restored participant payload.
    #[must_use]
    pub fn payload_bytes(&self) -> &[u8] {
        self.payload.as_bytes()
    }
}

fn validate_payload_charge(payload: &BudgetedCheckpointBytes) -> Result<(), CheckpointError> {
    if payload.lease.charged_items() != 1 || payload.lease.charged_bytes() != payload.bytes.len() {
        return Err(CheckpointError::ObjectVerification);
    }
    Ok(())
}

fn digest_bytes(bytes: &[u8]) -> ContentDigest {
    ContentDigest::from_bytes(*blake3::hash(bytes).as_bytes())
}

/// Complete metadata for one atomically committed checkpoint generation.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CommittedCheckpointGeneration {
    generation: CheckpointGeneration,
    previous: Option<ContentDigest>,
    cut: CheckpointCut,
    participant_descriptors: Vec<ParticipantStateDescriptor>,
    result_index_root: ContentDigest,
    is_final: bool,
    terminal_reason: Option<CheckpointTerminalReason>,
}

impl CommittedCheckpointGeneration {
    /// Canonicalize and validate a complete committed generation.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        epoch: CheckpointEpoch,
        previous: Option<ContentDigest>,
        cut: CheckpointCut,
        participant_plan: &CheckpointParticipantPlan,
        mut participant_descriptors: Vec<ParticipantStateDescriptor>,
        result_index_root: ContentDigest,
        is_final: bool,
        terminal_reason: Option<CheckpointTerminalReason>,
    ) -> Result<Self, CheckpointError> {
        participant_descriptors
            .sort_unstable_by(|left, right| left.participant_id.cmp(&right.participant_id));
        validate_committed_generation_shape(
            &cut,
            &participant_descriptors,
            is_final,
            terminal_reason,
        )?;
        if participant_descriptors
            .iter()
            .map(|descriptor| &descriptor.participant_id)
            .ne(participant_plan.ids().iter())
        {
            return Err(CheckpointError::ParticipantSetMismatch);
        }
        let digest = committed_generation_digest(
            epoch,
            previous.as_ref(),
            &cut,
            &participant_descriptors,
            &result_index_root,
            is_final,
            terminal_reason,
        )?;
        Ok(Self {
            generation: CheckpointGeneration::new(epoch, digest),
            previous,
            cut,
            participant_descriptors,
            result_index_root,
            is_final,
            terminal_reason,
        })
    }

    /// Clone the small content-addressed generation identity.
    #[must_use]
    pub fn generation(&self) -> CheckpointGeneration {
        self.generation.clone()
    }

    /// Borrow the content-addressed generation identity.
    #[must_use]
    pub const fn generation_ref(&self) -> &CheckpointGeneration {
        &self.generation
    }

    /// Borrow the preceding committed generation digest, when present.
    #[must_use]
    pub const fn previous(&self) -> Option<&ContentDigest> {
        self.previous.as_ref()
    }

    /// Borrow the complete represented cut.
    #[must_use]
    pub const fn cut(&self) -> &CheckpointCut {
        &self.cut
    }

    /// Borrow the canonical descriptor inventory.
    #[must_use]
    pub fn participant_descriptors(&self) -> &[ParticipantStateDescriptor] {
        &self.participant_descriptors
    }

    /// Borrow the immutable result-index root.
    #[must_use]
    pub const fn result_index_root(&self) -> &ContentDigest {
        &self.result_index_root
    }

    /// Return whether this generation terminates the run.
    #[must_use]
    pub const fn is_final(&self) -> bool {
        self.is_final
    }

    /// Return the terminal reason, present exactly for final generations.
    #[must_use]
    pub const fn terminal_reason(&self) -> Option<CheckpointTerminalReason> {
        self.terminal_reason
    }

    /// Verify canonical order, participant set, terminal state, and digest.
    pub fn verify(&self) -> Result<(), CheckpointError> {
        validate_committed_generation_shape(
            &self.cut,
            &self.participant_descriptors,
            self.is_final,
            self.terminal_reason,
        )?;
        let expected = committed_generation_digest(
            self.generation.epoch(),
            self.previous.as_ref(),
            &self.cut,
            &self.participant_descriptors,
            &self.result_index_root,
            self.is_final,
            self.terminal_reason,
        )?;
        if self.generation.digest() != &expected {
            return Err(CheckpointError::ObjectVerification);
        }
        Ok(())
    }

    fn descriptor(
        &self,
        participant_id: &CheckpointParticipantId,
    ) -> Option<&ParticipantStateDescriptor> {
        self.participant_descriptors
            .binary_search_by(|descriptor| descriptor.participant_id.cmp(participant_id))
            .ok()
            .map(|index| &self.participant_descriptors[index])
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SerializedCommittedCheckpointGeneration {
    generation: CheckpointGeneration,
    previous: Option<ContentDigest>,
    cut: CheckpointCut,
    participant_descriptors: Vec<ParticipantStateDescriptor>,
    result_index_root: ContentDigest,
    is_final: bool,
    terminal_reason: Option<CheckpointTerminalReason>,
}

impl<'de> Deserialize<'de> for CommittedCheckpointGeneration {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let serialized = SerializedCommittedCheckpointGeneration::deserialize(deserializer)?;
        let generation = Self {
            generation: serialized.generation,
            previous: serialized.previous,
            cut: serialized.cut,
            participant_descriptors: serialized.participant_descriptors,
            result_index_root: serialized.result_index_root,
            is_final: serialized.is_final,
            terminal_reason: serialized.terminal_reason,
        };
        generation.verify().map_err(serde::de::Error::custom)?;
        Ok(generation)
    }
}

fn validate_committed_generation_shape(
    cut: &CheckpointCut,
    descriptors: &[ParticipantStateDescriptor],
    is_final: bool,
    terminal_reason: Option<CheckpointTerminalReason>,
) -> Result<(), CheckpointError> {
    if is_final != terminal_reason.is_some() {
        return Err(CheckpointError::ObjectVerification);
    }
    if descriptors
        .windows(2)
        .any(|pair| pair[0].participant_id >= pair[1].participant_id)
    {
        return Err(CheckpointError::ParticipantSetMismatch);
    }
    if descriptors
        .iter()
        .any(|descriptor| descriptor.represented_cut != *cut)
    {
        return Err(CheckpointError::ParticipantSetMismatch);
    }
    Ok(())
}

fn committed_generation_digest(
    epoch: CheckpointEpoch,
    previous: Option<&ContentDigest>,
    cut: &CheckpointCut,
    descriptors: &[ParticipantStateDescriptor],
    result_index_root: &ContentDigest,
    is_final: bool,
    terminal_reason: Option<CheckpointTerminalReason>,
) -> Result<ContentDigest, CheckpointError> {
    let cut = serde_json::to_vec(cut).map_err(generation_encoding_error)?;
    let descriptors = serde_json::to_vec(descriptors).map_err(generation_encoding_error)?;
    let terminal_state = match terminal_reason {
        None => [0, 0],
        Some(CheckpointTerminalReason::Completed) => [1, 1],
        Some(CheckpointTerminalReason::Aborted) => [1, 2],
        Some(CheckpointTerminalReason::Cancelled) => [1, 3],
    };
    let mut hasher = blake3::Hasher::new();
    update_generation_digest_field(
        &mut hasher,
        b"aiperf.streaming.committed-checkpoint-generation.v1",
    );
    update_generation_digest_field(&mut hasher, &epoch.get().to_le_bytes());
    match previous {
        None => update_generation_digest_field(&mut hasher, &[0]),
        Some(previous) => {
            update_generation_digest_field(&mut hasher, &[1]);
            update_generation_digest_field(&mut hasher, previous.as_bytes());
        }
    }
    update_generation_digest_field(&mut hasher, &cut);
    update_generation_digest_field(&mut hasher, &descriptors);
    update_generation_digest_field(&mut hasher, result_index_root.as_bytes());
    update_generation_digest_field(&mut hasher, &[u8::from(is_final)]);
    update_generation_digest_field(&mut hasher, &terminal_state);
    Ok(ContentDigest::from_bytes(*hasher.finalize().as_bytes()))
}

fn generation_encoding_error(error: serde_json::Error) -> CheckpointError {
    CheckpointError::Storage {
        message: format!("could not encode committed checkpoint generation: {error}"),
    }
}

fn update_generation_digest_field(hasher: &mut blake3::Hasher, field: &[u8]) {
    hasher.update(&(field.len() as u64).to_le_bytes());
    hasher.update(field);
}

/// Coordinator-selected checkpoint barrier presented to every participant.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CheckpointBarrier {
    /// Epoch being prepared.
    pub epoch: CheckpointEpoch,
    /// Complete typed cut participants must represent.
    pub cut: CheckpointCut,
    /// Digest of the frozen participant and execution plan.
    pub plan_digest: ContentDigest,
}

/// Post-CAS notification binding a participant state to its committed generation.
///
/// Receipt fields are private so only an exact descriptor present in a verified
/// committed generation can produce a notification.
///
/// ```compile_fail
/// # use aiperf_runtime::streaming::checkpoint::CommittedParticipantReceipt;
/// # fn cannot_forge(receipt: CommittedParticipantReceipt) {
/// let _generation = receipt.generation;
/// let _participant = receipt.participant_id;
/// # }
/// ```
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CommittedParticipantReceipt {
    generation: CheckpointGeneration,
    participant_id: CheckpointParticipantId,
    descriptor_digest: ContentDigest,
    represented_cut: CheckpointCut,
}

impl CommittedParticipantReceipt {
    /// Construct a receipt for an exact descriptor in a verified generation.
    pub fn new(
        generation: &CommittedCheckpointGeneration,
        descriptor: &ParticipantStateDescriptor,
    ) -> Result<Self, CheckpointError> {
        generation.verify()?;
        if generation.descriptor(&descriptor.participant_id) != Some(descriptor) {
            return Err(CheckpointError::ObjectVerification);
        }
        Ok(Self {
            generation: generation.generation(),
            participant_id: descriptor.participant_id.clone(),
            descriptor_digest: descriptor.digest()?,
            represented_cut: descriptor.represented_cut.clone(),
        })
    }

    /// Borrow the exact authoritative generation identity.
    #[must_use]
    pub const fn generation(&self) -> &CheckpointGeneration {
        &self.generation
    }

    /// Borrow the stable participant receiving this notification.
    #[must_use]
    pub const fn participant_id(&self) -> &CheckpointParticipantId {
        &self.participant_id
    }

    /// Borrow the digest of the committed descriptor.
    #[must_use]
    pub const fn descriptor_digest(&self) -> &ContentDigest {
        &self.descriptor_digest
    }

    /// Borrow the exact cut represented by the committed descriptor.
    #[must_use]
    pub const fn represented_cut(&self) -> &CheckpointCut {
        &self.represented_cut
    }
}

/// Stateful streaming owner participating in atomic checkpoint generations.
#[async_trait(?Send)]
pub trait StreamingCheckpointParticipant {
    /// Return the stable identity frozen before source polling begins.
    fn participant_id(&self) -> CheckpointParticipantId;

    /// Prepare a non-destructive immutable view at the requested barrier.
    async fn checkpoint_view(
        &mut self,
        barrier: &CheckpointBarrier,
    ) -> Result<PreparedParticipantState, CheckpointError>;

    /// Initialize fresh state or restore exactly one committed state object.
    async fn initialize(
        &mut self,
        state: Option<CommittedParticipantState>,
    ) -> Result<(), CheckpointError>;

    /// Apply an idempotent post-commit notification and release pre-cut state.
    async fn checkpoint_committed(
        &mut self,
        receipt: &CommittedParticipantReceipt,
    ) -> Result<(), CheckpointError>;
}

/// One-shot initialization guard shared by participant implementations.
#[derive(Debug, Default)]
pub struct ParticipantInitialization {
    is_initialized: bool,
}

impl ParticipantInitialization {
    /// Mark a participant initialized, rejecting every subsequent attempt.
    pub fn initialize_once(&mut self) -> Result<(), CheckpointError> {
        if self.is_initialized {
            return Err(CheckpointError::AlreadyInitialized);
        }
        self.is_initialized = true;
        Ok(())
    }
}

/// Named stateful owners required in a frozen streaming checkpoint plan.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CheckpointParticipantOwners {
    /// Source discovery and cursor owner.
    pub source: Option<CheckpointParticipantId>,
    /// Streaming format/decode owner.
    pub format: Option<CheckpointParticipantId>,
    /// Event-time and global-order policy owner.
    pub event_time_order_policy: Option<CheckpointParticipantId>,
    /// Cross-record session state owner.
    pub session_coordinator: Option<CheckpointParticipantId>,
    /// Every prepared action-driver binding, aggregated by stable binding.
    pub action_driver_bindings: Vec<CheckpointParticipantId>,
    /// Placement policy owner.
    pub placement_policy: Option<CheckpointParticipantId>,
    /// Placement driver owner.
    pub placement_driver: Option<CheckpointParticipantId>,
    /// Owner aggregating all dynamic active action handles.
    pub active_execution_set: Option<CheckpointParticipantId>,
    /// Owner aggregating accepted blocking jobs and durable derived state.
    pub blocking_owner: Option<CheckpointParticipantId>,
    /// Result and terminal epoch coordinator.
    pub result_epoch: Option<CheckpointParticipantId>,
}

/// Required stateful role omitted from a frozen checkpoint plan.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RequiredCheckpointOwner {
    /// Source owner.
    Source,
    /// Streaming format owner.
    Format,
    /// Event-time and ordering owner.
    EventTimeOrderPolicy,
    /// Session coordinator.
    SessionCoordinator,
    /// At least one prepared action-driver binding.
    ActionDriverBinding,
    /// Placement policy owner.
    PlacementPolicy,
    /// Placement driver owner.
    PlacementDriver,
    /// Active-execution set owner.
    ActiveExecutionSet,
    /// Blocking-work owner.
    BlockingOwner,
    /// Result and terminal epoch owner.
    ResultEpoch,
}

/// Exact sorted stable participant inventory frozen before execution.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CheckpointParticipantPlan {
    ids: Vec<CheckpointParticipantId>,
}

impl CheckpointParticipantPlan {
    /// Sort participant IDs and reject duplicates.
    pub fn new(
        ids: impl IntoIterator<Item = CheckpointParticipantId>,
    ) -> Result<Self, CheckpointParticipantPlanError> {
        let mut ids: Vec<_> = ids.into_iter().collect();
        ids.sort_unstable();
        if let Some(duplicate) = ids
            .windows(2)
            .find_map(|pair| (pair[0] == pair[1]).then(|| pair[0].clone()))
        {
            return Err(CheckpointParticipantPlanError::DuplicateParticipant(
                duplicate,
            ));
        }
        Ok(Self { ids })
    }

    /// Validate and freeze every required stateful owner in stable ID order.
    pub fn from_required_owners(
        mut owners: CheckpointParticipantOwners,
    ) -> Result<Self, CheckpointParticipantPlanError> {
        let mut ids = Vec::with_capacity(owners.action_driver_bindings.len() + 9);
        ids.push(require_owner(
            owners.source.take(),
            RequiredCheckpointOwner::Source,
        )?);
        ids.push(require_owner(
            owners.format.take(),
            RequiredCheckpointOwner::Format,
        )?);
        ids.push(require_owner(
            owners.event_time_order_policy.take(),
            RequiredCheckpointOwner::EventTimeOrderPolicy,
        )?);
        ids.push(require_owner(
            owners.session_coordinator.take(),
            RequiredCheckpointOwner::SessionCoordinator,
        )?);
        if owners.action_driver_bindings.is_empty() {
            return Err(CheckpointParticipantPlanError::MissingRequiredOwner(
                RequiredCheckpointOwner::ActionDriverBinding,
            ));
        }
        ids.append(&mut owners.action_driver_bindings);
        ids.push(require_owner(
            owners.placement_policy.take(),
            RequiredCheckpointOwner::PlacementPolicy,
        )?);
        ids.push(require_owner(
            owners.placement_driver.take(),
            RequiredCheckpointOwner::PlacementDriver,
        )?);
        ids.push(require_owner(
            owners.active_execution_set.take(),
            RequiredCheckpointOwner::ActiveExecutionSet,
        )?);
        ids.push(require_owner(
            owners.blocking_owner.take(),
            RequiredCheckpointOwner::BlockingOwner,
        )?);
        ids.push(require_owner(
            owners.result_epoch.take(),
            RequiredCheckpointOwner::ResultEpoch,
        )?);
        Self::new(ids)
    }

    /// Borrow the exact stable ID inventory in canonical order.
    #[must_use]
    pub fn ids(&self) -> &[CheckpointParticipantId] {
        &self.ids
    }

    /// Compute the canonical digest used to bind barriers to this plan.
    #[must_use]
    pub fn digest(&self) -> ContentDigest {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"aiperf.streaming.checkpoint-participant-plan.v1");
        hasher.update(&(self.ids.len() as u64).to_le_bytes());
        for id in &self.ids {
            hasher.update(&(id.as_str().len() as u64).to_le_bytes());
            hasher.update(id.as_str().as_bytes());
        }
        ContentDigest::from_bytes(*hasher.finalize().as_bytes())
    }
}

fn require_owner(
    id: Option<CheckpointParticipantId>,
    owner: RequiredCheckpointOwner,
) -> Result<CheckpointParticipantId, CheckpointParticipantPlanError> {
    id.ok_or(CheckpointParticipantPlanError::MissingRequiredOwner(owner))
}

/// Invalid frozen checkpoint-participant inventory.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CheckpointParticipantPlanError {
    /// Two stateful owners claimed the same stable participant identity.
    DuplicateParticipant(CheckpointParticipantId),
    /// A mandatory stateful owner was absent.
    MissingRequiredOwner(RequiredCheckpointOwner),
}

impl fmt::Display for CheckpointParticipantPlanError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DuplicateParticipant(participant) => write!(
                formatter,
                "duplicate checkpoint participant {:?}",
                participant.as_str()
            ),
            Self::MissingRequiredOwner(owner) => {
                write!(formatter, "missing required checkpoint owner {owner:?}")
            }
        }
    }
}

impl std::error::Error for CheckpointParticipantPlanError {}

/// Stable checkpoint preparation, storage, restore, or notification failure.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CheckpointError {
    /// Participant initialization was attempted more than once.
    AlreadyInitialized,
    /// Generation compare-and-swap observed a different committed root.
    GenerationConflict {
        /// Generation expected by the writer.
        expected: Option<CheckpointGeneration>,
        /// Generation actually committed by the backend.
        actual: Option<CheckpointGeneration>,
    },
    /// Prepared or restored participants differ from the frozen exact set.
    ParticipantSetMismatch,
    /// Accepted work prevents one participant from representing the cut.
    CutBlockedByInflight {
        /// Stable owner of the in-flight jobs.
        participant: CheckpointParticipantId,
        /// Number of jobs preventing a complete view.
        job_count: usize,
    },
    /// Checkpoint state exceeded an explicitly owned resource budget.
    StateBudget {
        /// Participant requesting unavailable state capacity.
        participant: CheckpointParticipantId,
        /// Stable nested state-budget failure code.
        code: StateBudgetFailureCode,
    },
    /// Immutable object length, digest, or budget ownership did not verify.
    ObjectVerification,
    /// The backend could not retain a generation read lease.
    LeaseLost {
        /// Generation whose lease was lost.
        generation: CheckpointGeneration,
    },
    /// A participant failed after the generation became authoritative.
    PostCommitNotification {
        /// Participant whose idempotent notification failed.
        participant: CheckpointParticipantId,
    },
    /// The committed source generation cannot be reacquired during resume.
    SourceUnavailableOnResume,
    /// Checkpoint storage failed without changing the authoritative generation.
    Storage {
        /// Stable, user-readable storage context.
        message: String,
    },
}

impl fmt::Display for CheckpointError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AlreadyInitialized => {
                write!(formatter, "checkpoint participant already initialized")
            }
            Self::GenerationConflict { expected, actual } => write!(
                formatter,
                "checkpoint generation conflict: expected {expected:?}, actual {actual:?}"
            ),
            Self::ParticipantSetMismatch => {
                write!(formatter, "checkpoint participant set mismatch")
            }
            Self::CutBlockedByInflight {
                participant,
                job_count,
            } => write!(
                formatter,
                "checkpoint cut blocked by {job_count} in-flight jobs for {:?}",
                participant.as_str()
            ),
            Self::StateBudget { participant, code } => write!(
                formatter,
                "checkpoint state budget failed for {:?}: {code:?}",
                participant.as_str()
            ),
            Self::ObjectVerification => write!(formatter, "checkpoint object verification failed"),
            Self::LeaseLost { generation } => {
                write!(
                    formatter,
                    "checkpoint generation lease lost: {generation:?}"
                )
            }
            Self::PostCommitNotification { participant } => write!(
                formatter,
                "checkpoint post-commit notification failed for {:?}",
                participant.as_str()
            ),
            Self::SourceUnavailableOnResume => {
                write!(formatter, "checkpoint source unavailable on resume")
            }
            Self::Storage { message } => write!(formatter, "checkpoint storage failed: {message}"),
        }
    }
}

impl std::error::Error for CheckpointError {}
