// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Cross-partition conversation session program.
//!
//! One coordinator owns every live conversation for a run. Fragments are joined
//! by `(stream_identity, StableSessionKey)` so a conversation spans arbitrary
//! partition boundaries; partition exhaustion is a decoder event and is never
//! session closure. Endpoint replies observed on the execution edge are folded
//! into the same transcript as authored turns, which is what makes an
//! authored/endpoint pair one durable, checkpointable unit.
//!
//! Inferred closure, missing-predecessor disposition, and quarantine tombstones
//! are decided by the checked policies in [`super::closure`]. Partition
//! exhaustion is never closure evidence, an unresolved declared predecessor is
//! held pending rather than given an invented disposition, and a quarantined
//! session is retired into a durable budgeted tombstone that a later fragment
//! extends instead of resurrecting.

use std::collections::{BTreeMap, btree_map::Entry};

use async_trait::async_trait;
use bytes::Bytes;
use serde::{Deserialize, Serialize};
use serde_json::value::RawValue;
use smallvec::SmallVec;

use crate::engine::registry::WorkloadDescriptor;
use crate::streaming::{
    action::{ActionExecutionEvent, EndpointSessionUpdate},
    budget::{BudgetError, BudgetLease, StreamingResourceBudget},
    checkpoint::{
        BudgetedCheckpointBytes, CheckpointBarrier, CheckpointError, CheckpointParticipantId,
        CommittedParticipantReceipt, CommittedParticipantState, DecodeHorizon,
        ParticipantInitialization, PreparedParticipantState, StreamRunIdentity,
        StreamingCheckpointParticipant,
    },
    failure::SessionFailureCode,
    format::{SessionWatermark, StreamingFormatDescriptor},
    identity::{
        ContentDigest, DuplicateDisposition, GlobalSequence, ImmutableObjectIdentity,
        LogicalRecordReceipt, SessionCausalFrontier, StableActionId, StableOrderKey,
        StableRecordId, StableSessionKey, classify_logical_duplicate, stable_action_id,
    },
    reliability::StreamingInputDomainIdentity,
    session::{
        DatasetActionSink, SessionClosureCapability, SessionCoordinatorError, SessionPlacement,
        SessionQuarantineTombstoneMap, SessionSealReceipt, SessionStateRetention,
        StreamingSessionCoordinator, StreamingSessionPrepareContext,
        StreamingSessionProgramDescriptor, StreamingSessionProgramFactory,
        ValidatedStreamingSessionProgramConfig,
        closure::{
            MissingPredecessorPolicy, SessionCausalityLimits, SessionClosureDecision,
            SessionClosureEvidence, SessionClosurePolicy, SessionQuarantineClosureProof,
            validate_session_limits,
        },
    },
    source::SourceSeal,
    unit::{
        ActionContentLeaseSet, DatasetActionKind, DatasetActionV1, EventTimeUtc,
        ExecutableDatasetAction, SessionFragmentLease, SessionMutationV1, SessionRequestAction,
        SessionTerminalAction, SourcePosition, StateBudgetFailureCode, StreamingSessionFragment,
        UnitProvenance,
    },
};

/// Stable registry identity of this session program.
pub const CONVERSATION_SESSION_PROGRAM_ID: &str = "conversation";

/// Canonical fragment schema this program joins.
pub const CONVERSATION_FRAGMENT_SCHEMA: &str = "aiperf.stream.session-fragment.v1";

/// Canonical action schema this program emits.
pub const CONVERSATION_ACTION_SCHEMA: &str = "aiperf.stream.action.v1";

/// Checkpoint schema identity for retained conversation state.
const CONVERSATION_CHECKPOINT_SCHEMA_ID: &str = "aiperf.stream.session.conversation";

/// Checkpoint schema version for retained conversation state.
const CONVERSATION_CHECKPOINT_SCHEMA_VERSION: u32 = 1;

/// Monotonic version of one conversation's canonical retained state.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(transparent)]
pub struct SessionStateVersion(u64);

impl SessionStateVersion {
    /// Version of a session that has incorporated nothing.
    pub const INITIAL: Self = Self(0);

    /// Return the underlying version number.
    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }

    /// Advance one version, refusing instead of wrapping.
    pub const fn next(self) -> Result<Self, SessionCoordinatorError> {
        match self.0.checked_add(1) {
            Some(value) => Ok(Self(value)),
            None => Err(SessionCoordinatorError::state_budget(
                StateBudgetFailureCode::ItemCapacity,
            )),
        }
    }
}

/// Cross-partition identity of one conversation.
///
/// The immutable partition is deliberately excluded: including it would make a
/// conversation unable to span partitions, which is this program's whole
/// purpose.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ConversationSessionScope {
    stream_identity: ContentDigest,
    session_key: StableSessionKey,
}

impl ConversationSessionScope {
    /// Bind a frozen stream identity to one stable session key.
    #[must_use]
    pub const fn new(stream_identity: ContentDigest, session_key: StableSessionKey) -> Self {
        Self {
            stream_identity,
            session_key,
        }
    }

    /// Return the frozen semantic identity of the owning stream.
    #[must_use]
    pub const fn stream_identity(&self) -> ContentDigest {
        self.stream_identity
    }

    /// Return the stable session key joining fragments across partitions.
    #[must_use]
    pub const fn session_key(&self) -> StableSessionKey {
        self.session_key
    }

    /// Derive the stable external session ordinal reported as `session_num`.
    ///
    /// `as_bytes` returns `&[u8; 32]`, so the eight-byte prefix copy is total.
    /// BLAKE3 output is uniform, so any fixed prefix is as good a projection as
    /// any other, and the value is identical before and after a restart.
    #[must_use]
    pub fn stable_ordinal(&self) -> u64 {
        let mut prefix = [0u8; 8];
        prefix.copy_from_slice(&self.session_key.as_bytes()[..8]);
        u64::from_le_bytes(prefix)
    }
}

/// Origin of one transcript entry.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TranscriptOrigin {
    /// The entry came from a decoded authored source mutation.
    Authored,
    /// The entry came from an observed endpoint session update.
    Endpoint,
}

/// One entry of a conversation's durable transcript.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TranscriptEntry {
    /// Authored conversation role.
    pub role: String,
    /// Canonical content bytes.
    pub content: Vec<u8>,
    /// Authored turn ordinal this entry belongs to.
    pub turn_ordinal: u64,
    /// Whether the entry was authored or produced by the endpoint.
    pub origin: TranscriptOrigin,
}

/// Restart-durable continuity facts for one conversation.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ConversationContinuity {
    scope: ConversationSessionScope,
    version: SessionStateVersion,
    folded_through_turn: Option<u64>,
    emitted_through_turn: Option<u64>,
    terminal_through_turn: Option<u64>,
    next_causal_ordinal: u64,
    unrepresented_from: Option<SourcePosition>,
    pending_close_reason: Option<String>,
}

impl ConversationContinuity {
    /// Begin continuity for a session that has incorporated nothing.
    #[must_use]
    pub const fn fresh(scope: ConversationSessionScope) -> Self {
        Self {
            scope,
            version: SessionStateVersion::INITIAL,
            folded_through_turn: None,
            emitted_through_turn: None,
            terminal_through_turn: None,
            next_causal_ordinal: 0,
            unrepresented_from: None,
            pending_close_reason: None,
        }
    }

    /// Borrow the cross-partition scope owning this continuity.
    #[must_use]
    pub const fn scope(&self) -> &ConversationSessionScope {
        &self.scope
    }

    /// Return the monotonic retained-state version.
    #[must_use]
    pub const fn version(&self) -> SessionStateVersion {
        self.version
    }

    /// Return the greatest contiguous authored ordinal in the transcript.
    #[must_use]
    pub const fn folded_through_turn(&self) -> Option<u64> {
        self.folded_through_turn
    }

    /// Return the last authored ordinal whose action left the coordinator.
    #[must_use]
    pub const fn emitted_through_turn(&self) -> Option<u64> {
        self.emitted_through_turn
    }

    /// Return the last authored ordinal with an observed terminal receipt.
    #[must_use]
    pub const fn terminal_through_turn(&self) -> Option<u64> {
        self.terminal_through_turn
    }

    /// Return the earliest source coordinate this state does not represent.
    #[must_use]
    pub const fn unrepresented_from(&self) -> Option<SourcePosition> {
        self.unrepresented_from
    }

    /// Return the authored turn ordinal emitted but not yet terminal.
    ///
    /// This is the exact set that must be re-emitted after a restart.
    #[must_use]
    pub const fn in_flight_turn(&self) -> Option<u64> {
        match (self.emitted_through_turn, self.terminal_through_turn) {
            (Some(emitted), Some(terminal)) if emitted > terminal => Some(emitted),
            (Some(emitted), None) => Some(emitted),
            _ => None,
        }
    }

    /// Whether a producer-authored close is retained but not yet emitted.
    #[must_use]
    pub const fn has_pending_close(&self) -> bool {
        self.pending_close_reason.is_some()
    }
}

/// Validated startup-only configuration for the `conversation` program.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ConversationProgramConfig {
    /// Maximum simultaneously live conversations.
    #[serde(default = "default_max_active_sessions")]
    pub max_active_sessions: usize,
    /// Maximum authored turns retained for one conversation.
    #[serde(default = "default_max_turns_per_session")]
    pub max_turns_per_session: u64,
    /// Maximum retained transcript bytes for one conversation.
    #[serde(default = "default_max_transcript_bytes")]
    pub max_transcript_bytes: usize,
    /// Maximum out-of-order authored turns held for one conversation.
    #[serde(default = "default_max_pending_mutations")]
    pub max_pending_mutations_per_session: usize,
    /// Maximum retained quarantine tombstones for this run.
    #[serde(default = "default_max_quarantine_tombstones")]
    pub max_quarantine_tombstones: usize,
    /// Inferred-closure and missing-predecessor policy.
    #[serde(default)]
    pub closure: SessionClosurePolicy,
}

const fn default_max_active_sessions() -> usize {
    4096
}

const fn default_max_turns_per_session() -> u64 {
    256
}

const fn default_max_transcript_bytes() -> usize {
    1 << 20
}

const fn default_max_pending_mutations() -> usize {
    16
}

const fn default_max_quarantine_tombstones() -> usize {
    4096
}

impl Default for ConversationProgramConfig {
    fn default() -> Self {
        Self {
            max_active_sessions: default_max_active_sessions(),
            max_turns_per_session: default_max_turns_per_session(),
            max_transcript_bytes: default_max_transcript_bytes(),
            max_pending_mutations_per_session: default_max_pending_mutations(),
            max_quarantine_tombstones: default_max_quarantine_tombstones(),
            closure: SessionClosurePolicy::default(),
        }
    }
}

impl ConversationProgramConfig {
    fn validate_limits(self) -> Result<Self, SessionCoordinatorError> {
        if self.max_quarantine_tombstones == 0 {
            return Err(SessionCoordinatorError::session(
                SessionFailureCode::UnboundedCausalityState,
            ));
        }
        if let Some(deadline) = self.closure.inactivity_deadline_ns
            && deadline <= 0
        {
            return Err(SessionCoordinatorError::session(
                SessionFailureCode::UnboundedCausalityState,
            ));
        }
        // A zero bound is an unbounded causality state with extra steps, and the
        // shared refusal is what proves the authored policy can retire state.
        validate_session_limits(SessionCausalityLimits {
            max_active_sessions: Some(self.max_active_sessions),
            max_pending_per_session: Some(self.max_pending_mutations_per_session),
            max_retained_bytes_per_session: Some(self.max_transcript_bytes),
            missing_predecessor: self.closure.missing_predecessor,
        })?;
        if self.max_turns_per_session == 0 {
            return Err(SessionCoordinatorError::session(
                SessionFailureCode::UnboundedCausalityState,
            ));
        }
        Ok(self)
    }
}

/// Immutable registry metadata for the `conversation` session program.
pub static CONVERSATION_SESSION_PROGRAM: StreamingSessionProgramDescriptor =
    StreamingSessionProgramDescriptor {
        id: CONVERSATION_SESSION_PROGRAM_ID,
        description: "Cross-partition conversation session program",
        fragment_input_schemas: &[CONVERSATION_FRAGMENT_SCHEMA],
        action_schemas: &[CONVERSATION_ACTION_SCHEMA],
        closure: &[
            SessionClosureCapability::ExplicitClose,
            SessionClosureCapability::MonotonicSequence,
            SessionClosureCapability::HardWatermark,
            SessionClosureCapability::FiniteSeal,
            SessionClosureCapability::LossyInactivity,
        ],
        retention: SessionStateRetention::BoundedMemory,
        placement: SessionPlacement::ControllerCanonical,
        supports_virtual_clock: true,
    };

/// Startup factory for the `conversation` session program.
#[derive(Clone, Copy, Debug, Default)]
pub struct StreamingConversationProgramFactory;

impl StreamingSessionProgramFactory for StreamingConversationProgramFactory {
    fn descriptor(&self) -> &'static StreamingSessionProgramDescriptor {
        &CONVERSATION_SESSION_PROGRAM
    }

    fn validate(
        &self,
        authored: &RawValue,
        format: &StreamingFormatDescriptor,
        _workload: &WorkloadDescriptor,
    ) -> Result<Box<dyn ValidatedStreamingSessionProgramConfig>, SessionCoordinatorError> {
        if !CONVERSATION_SESSION_PROGRAM
            .fragment_input_schemas
            .contains(&format.output_schema)
        {
            return Err(SessionCoordinatorError::session(
                SessionFailureCode::UnsupportedMutation,
            ));
        }
        let config: ConversationProgramConfig =
            serde_json::from_str(authored.get()).map_err(|_| {
                SessionCoordinatorError::session(SessionFailureCode::UnsupportedMutation)
            })?;
        Ok(Box::new(config.validate_limits()?))
    }

    fn prepare(
        &self,
        config: Box<dyn ValidatedStreamingSessionProgramConfig>,
        context: &StreamingSessionPrepareContext,
    ) -> Result<Box<dyn StreamingSessionCoordinator>, SessionCoordinatorError> {
        // `Box<dyn ValidatedStreamingSessionProgramConfig>` itself satisfies the
        // blanket impl, so the erased value must be reached through an explicit
        // reborrow: `config.as_any()` would erase the box rather than the value.
        let config = *ValidatedStreamingSessionProgramConfig::as_any(config.as_ref())
            .downcast_ref::<ConversationProgramConfig>()
            .ok_or_else(|| {
                SessionCoordinatorError::session(SessionFailureCode::UnsupportedMutation)
            })?;
        Ok(Box::new(StreamingConversationCoordinator::new(
            config, context,
        )))
    }
}

/// Run-scoped owner of every live conversation.
#[derive(Debug)]
pub struct StreamingConversationCoordinator {
    run: StreamRunIdentity,
    participant_id: CheckpointParticipantId,
    program_semantic_digest: ContentDigest,
    stream_identity: ContentDigest,
    input_domain: StreamingInputDomainIdentity,
    limits: ConversationProgramConfig,
    state_budget: StreamingResourceBudget,
    checkpoint_budget: StreamingResourceBudget,
    sessions: BTreeMap<ConversationSessionScope, ConversationSession>,
    in_flight: BTreeMap<StableActionId, ConversationSessionScope>,
    tombstones: SessionQuarantineTombstoneMap,
    initialization: ParticipantInitialization,
    causal_frontier: SessionCausalFrontier,
    next_global_sequence: u64,
    latest_event_time: Option<EventTimeUtc>,
}

#[derive(Debug)]
struct ConversationSession {
    continuity: ConversationContinuity,
    transcript: Vec<TranscriptEntry>,
    transcript_bytes: usize,
    receipts: BTreeMap<StableRecordId, LogicalRecordReceipt>,
    pending: BTreeMap<u64, PendingTurn>,
    last_action: Option<StableActionId>,
    last_provenance: UnitProvenance,
    last_stable_order: StableOrderKey,
    last_source_position: SourcePosition,
    first_source_position: SourcePosition,
    last_event_time: Option<EventTimeUtc>,
    needs_reemission: bool,
    // One lease per retained transcript entry. A `BudgetLease` can only shrink,
    // so a growing transcript charges incrementally rather than reallocating a
    // single lease and briefly holding both charges.
    state_leases: Vec<BudgetLease>,
}

#[derive(Debug)]
struct PendingTurn {
    role: String,
    content: Vec<u8>,
    source_position: SourcePosition,
    event_time: Option<EventTimeUtc>,
    stable_order: StableOrderKey,
    predecessors: SmallVec<[StableRecordId; 2]>,
    provenance: UnitProvenance,
}

impl StreamingConversationCoordinator {
    /// Construct one run-scoped coordinator from a validated configuration.
    #[must_use]
    pub fn new(
        config: ConversationProgramConfig,
        context: &StreamingSessionPrepareContext,
    ) -> Self {
        Self {
            run: context.run,
            participant_id: context.participant_id.clone(),
            program_semantic_digest: context.program_semantic_digest,
            stream_identity: context.stream_semantic_digest,
            input_domain: StreamingInputDomainIdentity::new(
                context.stream_semantic_digest,
                context.source_identity,
            ),
            limits: config,
            state_budget: context.session_state_budget.clone(),
            checkpoint_budget: context.checkpoint_budget.clone(),
            sessions: BTreeMap::new(),
            in_flight: BTreeMap::new(),
            tombstones: SessionQuarantineTombstoneMap::new(
                context.run,
                context.session_state_budget.clone(),
                config.max_quarantine_tombstones,
            ),
            initialization: ParticipantInitialization::default(),
            causal_frontier: SessionCausalFrontier {
                through_sequence: GlobalSequence::new(0),
                event_time: None,
                digest: ContentDigest::from_bytes([0; 32]),
            },
            next_global_sequence: 0,
            latest_event_time: None,
        }
    }

    /// Return the number of live conversations.
    #[must_use]
    pub fn active_session_count(&self) -> usize {
        self.sessions.len()
    }

    /// Borrow one conversation's restart-durable continuity facts.
    #[must_use]
    pub fn continuity(&self, scope: &ConversationSessionScope) -> Option<&ConversationContinuity> {
        self.sessions.get(scope).map(|session| &session.continuity)
    }

    /// Borrow one conversation's durable authored-and-endpoint transcript.
    #[must_use]
    pub fn transcript(&self, scope: &ConversationSessionScope) -> Option<&[TranscriptEntry]> {
        self.sessions
            .get(scope)
            .map(|session| session.transcript.as_slice())
    }

    /// Return the causal frontier this coordinator has proven complete.
    #[must_use]
    pub fn causal_frontier(&self) -> &SessionCausalFrontier {
        &self.causal_frontier
    }

    /// Return the proven upper bound on retained request bytes for one action.
    ///
    /// This is the one session-owned input of the terminal-record size proof;
    /// the endpoint, tokenizer, measurement, and capture inputs belong to the
    /// scheduled-request sink.
    #[must_use]
    pub fn max_request_retained_bytes(&self) -> u64 {
        u64::try_from(self.limits.max_transcript_bytes).unwrap_or(u64::MAX)
    }

    async fn ingest_mutation(
        &mut self,
        fragment: StreamingSessionFragment,
        output: &mut dyn DatasetActionSink,
    ) -> Result<(), SessionCoordinatorError> {
        self.flush_reemissions(output).await?;
        let scope = ConversationSessionScope::new(self.stream_identity, fragment.session_key);
        if self.tombstones.contains(&self.input_domain, fragment.session_key) {
            // A retired session is never recreated. The later fragment is
            // excluded and checked-extends the retained frontier, which moves
            // the tombstone root and invalidates any prepared acknowledgement.
            return self.exclude_into_tombstone(fragment, output).await;
        }
        // Vocabulary acceptance precedes every state change, so an unaccepted
        // mutation never opens a session.
        let accepted = AcceptedMutation::classify(&fragment.mutation)?;
        let receipt = logical_receipt(&fragment, &accepted);

        if let Some(session) = self.sessions.get(&scope)
            && let Some(existing) = session.receipts.get(&fragment.record_id)
        {
            let disposition = classify_logical_duplicate(existing, &receipt).map_err(|_| {
                SessionCoordinatorError::session(SessionFailureCode::ConflictingMutation)
            })?;
            if matches!(disposition, DuplicateDisposition::Identical) {
                // Idempotent replay: dropping the fragment returns its charge.
                return Ok(());
            }
        }

        self.admit_session(scope, &fragment)?;
        self.incorporate(scope, fragment, accepted, receipt)?;
        self.drain_ready(scope, output).await
    }

    fn admit_session(
        &mut self,
        scope: ConversationSessionScope,
        fragment: &StreamingSessionFragment,
    ) -> Result<(), SessionCoordinatorError> {
        if self.sessions.contains_key(&scope) {
            return Ok(());
        }
        if self.sessions.len() >= self.limits.max_active_sessions {
            return Err(SessionCoordinatorError::state_budget(
                StateBudgetFailureCode::ItemCapacity,
            ));
        }
        self.sessions.insert(
            scope,
            ConversationSession {
                continuity: ConversationContinuity::fresh(scope),
                transcript: Vec::new(),
                transcript_bytes: 0,
                receipts: BTreeMap::new(),
                pending: BTreeMap::new(),
                last_action: None,
                last_provenance: fragment.provenance.clone(),
                last_stable_order: fragment.stable_tie_break,
                last_source_position: fragment.source_position,
                first_source_position: fragment.source_position,
                last_event_time: fragment.event_time,
                needs_reemission: false,
                state_leases: Vec::new(),
            },
        );
        Ok(())
    }

    fn incorporate(
        &mut self,
        scope: ConversationSessionScope,
        fragment: StreamingSessionFragment,
        accepted: AcceptedMutation,
        receipt: LogicalRecordReceipt,
    ) -> Result<(), SessionCoordinatorError> {
        let limits = self.limits;
        let session = self.sessions.get_mut(&scope).ok_or_else(|| {
            SessionCoordinatorError::session(SessionFailureCode::MissingPredecessor)
        })?;
        let next_version = session.continuity.version.next()?;

        match accepted {
            AcceptedMutation::Turn {
                role,
                content,
                turn_ordinal,
            } => {
                if turn_ordinal >= limits.max_turns_per_session {
                    return Err(SessionCoordinatorError::state_budget(
                        StateBudgetFailureCode::ItemCapacity,
                    ));
                }
                if session.transcript_bytes.saturating_add(content.len())
                    > limits.max_transcript_bytes
                {
                    return Err(SessionCoordinatorError::state_budget(
                        StateBudgetFailureCode::ByteCapacity,
                    ));
                }
                match session.pending.entry(turn_ordinal) {
                    Entry::Occupied(_) => {
                        // An ordinal already held pending under a different
                        // record identity is a producer conflict, not a replay.
                        return Err(SessionCoordinatorError::session(
                            SessionFailureCode::ConflictingMutation,
                        ));
                    }
                    Entry::Vacant(slot) => {
                        if limits.max_pending_mutations_per_session == 0 {
                            return Err(SessionCoordinatorError::session(
                                SessionFailureCode::UnboundedCausalityState,
                            ));
                        }
                        slot.insert(PendingTurn {
                            role,
                            content,
                            source_position: fragment.source_position,
                            event_time: fragment.event_time,
                            stable_order: fragment.stable_tie_break,
                            predecessors: fragment.predecessors.clone(),
                            provenance: fragment.provenance.clone(),
                        });
                    }
                }
                if session.pending.len() > limits.max_pending_mutations_per_session {
                    session.pending.remove(&turn_ordinal);
                    return Err(SessionCoordinatorError::state_budget(
                        StateBudgetFailureCode::ItemCapacity,
                    ));
                }
            }
            AcceptedMutation::Close { reason } => {
                session.continuity.pending_close_reason = Some(reason);
            }
        }

        session.receipts.insert(fragment.record_id, receipt);
        session.continuity.version = next_version;
        session.last_provenance = fragment.provenance.clone();
        session.last_source_position = fragment.source_position;
        if fragment.event_time.is_some() {
            session.last_event_time = fragment.event_time;
        }
        if fragment.source_position < session.first_source_position {
            session.first_source_position = fragment.source_position;
        }
        self.observe_event_time(fragment.event_time);
        // The fragment's own lease is released here: its bytes now live under
        // this coordinator's own state charge.
        drop(fragment.lease);
        Ok(())
    }

    async fn drain_ready(
        &mut self,
        scope: ConversationSessionScope,
        output: &mut dyn DatasetActionSink,
    ) -> Result<(), SessionCoordinatorError> {
        while let Some(next_ordinal) = self.next_ready_ordinal(&scope) {
            self.fold_authored_turn(scope, next_ordinal)?;
            let action = self.build_request_action(scope, next_ordinal)?;
            let action_id = action.action_id();
            output.send_action(action).await?;
            self.record_emission(scope, next_ordinal, action_id)?;
        }
        self.settle_pending_close(scope, output).await?;
        self.publish_frontier(output).await
    }

    fn next_ready_ordinal(&self, scope: &ConversationSessionScope) -> Option<u64> {
        let session = self.sessions.get(scope)?;
        let next = session.continuity.folded_through_turn.map_or(0, |t| t + 1);
        let turn = session.pending.get(&next)?;
        // An unresolved declared predecessor is held, never given a
        // disposition here.
        turn.predecessors
            .iter()
            .all(|record| session.receipts.contains_key(record))
            .then_some(next)
    }

    fn fold_authored_turn(
        &mut self,
        scope: ConversationSessionScope,
        ordinal: u64,
    ) -> Result<(), SessionCoordinatorError> {
        let lease = self.acquire_state_lease(self.pending_content_len(&scope, ordinal)?)?;
        let session = self.sessions.get_mut(&scope).ok_or_else(|| {
            SessionCoordinatorError::session(SessionFailureCode::MissingPredecessor)
        })?;
        let turn = session.pending.remove(&ordinal).ok_or_else(|| {
            SessionCoordinatorError::session(SessionFailureCode::MissingPredecessor)
        })?;
        session.transcript_bytes = session.transcript_bytes.saturating_add(turn.content.len());
        session.transcript.push(TranscriptEntry {
            role: turn.role,
            content: turn.content,
            turn_ordinal: ordinal,
            origin: TranscriptOrigin::Authored,
        });
        session.state_leases.push(lease);
        session.continuity.folded_through_turn = Some(ordinal);
        session.last_provenance = turn.provenance;
        session.last_stable_order = turn.stable_order;
        session.last_source_position = turn.source_position;
        if turn.event_time.is_some() {
            session.last_event_time = turn.event_time;
        }
        Ok(())
    }

    fn pending_content_len(
        &self,
        scope: &ConversationSessionScope,
        ordinal: u64,
    ) -> Result<usize, SessionCoordinatorError> {
        self.sessions
            .get(scope)
            .and_then(|session| session.pending.get(&ordinal))
            .map(|turn| turn.content.len())
            .ok_or_else(|| SessionCoordinatorError::session(SessionFailureCode::MissingPredecessor))
    }

    fn record_emission(
        &mut self,
        scope: ConversationSessionScope,
        ordinal: u64,
        action_id: StableActionId,
    ) -> Result<(), SessionCoordinatorError> {
        self.in_flight.insert(action_id, scope);
        self.next_global_sequence = self.next_global_sequence.saturating_add(1);
        let session = self.sessions.get_mut(&scope).ok_or_else(|| {
            SessionCoordinatorError::session(SessionFailureCode::MissingPredecessor)
        })?;
        session.continuity.emitted_through_turn = Some(ordinal);
        session.continuity.next_causal_ordinal = ordinal.saturating_add(1);
        session.last_action = Some(action_id);
        session.needs_reemission = false;
        Ok(())
    }

    /// Build one conversation-pair request over the whole accumulated transcript.
    ///
    /// The action identity is a pure function of program digest, session key,
    /// incorporated record causes, and the authored turn ordinal, so re-deriving
    /// it after a restart yields the identical [`StableActionId`].
    fn build_request_action(
        &self,
        scope: ConversationSessionScope,
        causal_ordinal: u64,
    ) -> Result<ExecutableDatasetAction, SessionCoordinatorError> {
        let session = self.sessions.get(&scope).ok_or_else(|| {
            SessionCoordinatorError::session(SessionFailureCode::MissingPredecessor)
        })?;
        let request = encode_transcript(&session.transcript)?;
        let causes: Vec<StableRecordId> = session.receipts.keys().copied().collect();
        let action_id = stable_action_id(
            self.program_semantic_digest.as_bytes(),
            scope.session_key,
            &causes,
            DatasetActionKind::Request,
            causal_ordinal,
        );
        let predecessors: SmallVec<[StableActionId; 2]> = session.last_action.into_iter().collect();
        // The envelope charge covers the serialized request bytes the retained
        // transcript leases do not: `ExecutableDatasetAction::new` refuses an
        // undercharged payload.
        let envelope = acquire_retained(&self.state_budget, request.len())?;
        ExecutableDatasetAction::new(
            action_id,
            scope.session_key,
            predecessors,
            session.last_event_time,
            session.last_stable_order,
            session.last_source_position,
            session.last_provenance.clone(),
            DatasetActionV1::Request(SessionRequestAction { request }),
            ActionContentLeaseSet::from_retained(envelope),
        )
        .map_err(map_budget_error)
    }

    async fn settle_pending_close(
        &mut self,
        scope: ConversationSessionScope,
        output: &mut dyn DatasetActionSink,
    ) -> Result<(), SessionCoordinatorError> {
        let Some(session) = self.sessions.get(&scope) else {
            return Ok(());
        };
        if !session.continuity.has_pending_close() || session.continuity.in_flight_turn().is_some()
        {
            return Ok(());
        }
        if !session.pending.is_empty() {
            return Ok(());
        }
        let action = self.build_terminal_action(scope)?;
        output.send_action(action).await?;
        self.next_global_sequence = self.next_global_sequence.saturating_add(1);
        if let Some(session) = self.sessions.get_mut(&scope) {
            session.continuity.pending_close_reason = None;
        }
        Ok(())
    }

    fn build_terminal_action(
        &self,
        scope: ConversationSessionScope,
    ) -> Result<ExecutableDatasetAction, SessionCoordinatorError> {
        let session = self.sessions.get(&scope).ok_or_else(|| {
            SessionCoordinatorError::session(SessionFailureCode::MissingPredecessor)
        })?;
        let reason = session
            .continuity
            .pending_close_reason
            .clone()
            .ok_or_else(|| {
                SessionCoordinatorError::session(SessionFailureCode::MissingPredecessor)
            })?;
        let causes: Vec<StableRecordId> = session.receipts.keys().copied().collect();
        let action_id = stable_action_id(
            self.program_semantic_digest.as_bytes(),
            scope.session_key,
            &causes,
            DatasetActionKind::SessionTerminal,
            session.continuity.next_causal_ordinal,
        );
        let predecessors: SmallVec<[StableActionId; 2]> = session.last_action.into_iter().collect();
        let envelope = acquire_retained(&self.state_budget, reason.capacity())?;
        ExecutableDatasetAction::new(
            action_id,
            scope.session_key,
            predecessors,
            session.last_event_time,
            session.last_stable_order,
            session.last_source_position,
            session.last_provenance.clone(),
            DatasetActionV1::SessionTerminal(SessionTerminalAction { reason }),
            ActionContentLeaseSet::from_retained(envelope),
        )
        .map_err(map_budget_error)
    }

    /// Emit every action a session already produced before the last restart.
    ///
    /// The re-emitted action is byte-identical and carries the identical stable
    /// identity, so an action host keyed by [`StableActionId`] absorbs it as a
    /// duplicate submission rather than a second logical request.
    async fn flush_reemissions(
        &mut self,
        output: &mut dyn DatasetActionSink,
    ) -> Result<(), SessionCoordinatorError> {
        let pending: Vec<(ConversationSessionScope, u64)> = self
            .sessions
            .iter()
            .filter(|(_, session)| session.needs_reemission)
            .filter_map(|(scope, session)| {
                session
                    .continuity
                    .in_flight_turn()
                    .map(|ordinal| (*scope, ordinal))
            })
            .collect();
        for (scope, ordinal) in pending {
            let action = self.build_request_action(scope, ordinal)?;
            let action_id = action.action_id();
            output.send_action(action).await?;
            self.in_flight.insert(action_id, scope);
            if let Some(session) = self.sessions.get_mut(&scope) {
                session.needs_reemission = false;
                session.last_action = Some(action_id);
            }
        }
        Ok(())
    }

    async fn apply_watermark(
        &mut self,
        watermark: SessionWatermark,
        output: &mut dyn DatasetActionSink,
    ) -> Result<(), SessionCoordinatorError> {
        self.flush_reemissions(output).await?;
        self.observe_event_time(Some(watermark.through));
        // A format watermark is soft completeness evidence. It closes a session
        // only through an authored inactivity deadline or the hard-watermark
        // policy; on its own it proves nothing about session closure.
        for scope in self.closable_under_watermark(watermark.through) {
            self.retire_session(scope);
        }
        self.publish_frontier(output).await
    }

    /// Return every session the authored closure policy closes at `watermark`.
    fn closable_under_watermark(&self, watermark: EventTimeUtc) -> Vec<ConversationSessionScope> {
        self.sessions
            .iter()
            .filter(|(_, session)| {
                // An emitted-but-unterminated turn is still causally in flight.
                session.continuity.in_flight_turn().is_none()
            })
            .filter_map(|(scope, session)| {
                let session_event_time = session.last_event_time?;
                let inactivity = self.limits.closure.decide(
                    SessionClosureEvidence::SoftWatermarkBelowDeadline {
                        watermark,
                        session_event_time,
                    },
                );
                let decision = match inactivity {
                    SessionClosureDecision::Wait => self.limits.closure.decide(
                        SessionClosureEvidence::HardWatermarkPastSession {
                            watermark,
                            session_event_time,
                        },
                    ),
                    closed => closed,
                };
                matches!(decision, SessionClosureDecision::Close(_)).then_some(*scope)
            })
            .collect()
    }

    /// Drop one session's live state and return its retained charge.
    fn retire_session(&mut self, scope: ConversationSessionScope) {
        if let Some(session) = self.sessions.remove(&scope) {
            self.in_flight.retain(|_, owner| *owner != scope);
            // `state_leases` release the retained transcript charge on drop.
            drop(session);
        }
    }

    /// Retire one session into a durable budgeted quarantine tombstone.
    ///
    /// The tombstone binds the run, exact input domain, session key, issue
    /// identity, retained causal frontier, and the checked closure proof; live,
    /// pending, and emitted state is retired in the same call.
    pub fn quarantine_session(
        &mut self,
        session_key: StableSessionKey,
        issue_id: ContentDigest,
        closure_proof: SessionQuarantineClosureProof,
    ) -> Result<(), SessionCoordinatorError> {
        let scope = ConversationSessionScope::new(self.stream_identity, session_key);
        self.tombstones.install(
            self.input_domain.clone(),
            session_key,
            issue_id,
            self.causal_frontier.clone(),
            closure_proof,
        )?;
        self.retire_session(scope);
        Ok(())
    }

    /// Borrow the retained quarantine tombstone map.
    #[must_use]
    pub const fn tombstones(&self) -> &SessionQuarantineTombstoneMap {
        &self.tombstones
    }

    /// Exclude one fragment addressed to a retired session.
    async fn exclude_into_tombstone(
        &mut self,
        fragment: StreamingSessionFragment,
        output: &mut dyn DatasetActionSink,
    ) -> Result<(), SessionCoordinatorError> {
        let session_key = fragment.session_key;
        self.observe_event_time(fragment.event_time);
        self.next_global_sequence = self.next_global_sequence.saturating_add(1);
        drop(fragment.lease);
        let frontier = self.derive_frontier();
        self.tombstones
            .extend_frontier(&self.input_domain, session_key, frontier)?;
        self.publish_frontier(output).await
    }

    async fn apply_execution(
        &mut self,
        event: ActionExecutionEvent,
        output: &mut dyn DatasetActionSink,
    ) -> Result<(), SessionCoordinatorError> {
        // Callers must route this through the action host, never from inside a
        // terminal-lane record processor: that processor runs on the lane's
        // single drain owner and cannot hold `&mut` to this coordinator.
        self.flush_reemissions(output).await?;
        match event {
            ActionExecutionEvent::Admitted(_) | ActionExecutionEvent::FirstToken(_) => Ok(()),
            ActionExecutionEvent::SessionUpdate(update) => self.fold_endpoint_reply(update),
            ActionExecutionEvent::Terminal(receipt) => {
                let Some(scope) = self.in_flight.remove(&receipt.event.action_id) else {
                    return Ok(());
                };
                self.settle_terminal(scope, output).await
            }
        }
    }

    fn fold_endpoint_reply(
        &mut self,
        update: EndpointSessionUpdate,
    ) -> Result<(), SessionCoordinatorError> {
        // A missing update for an admitted action is absent state, not a fatal
        // invariant: the entry is simply not appended.
        let Some(scope) = self.in_flight.get(&update.event.action_id).copied() else {
            return Ok(());
        };
        let content = update.payload.as_bytes().to_vec();
        let Some(session) = self.sessions.get(&scope) else {
            return Ok(());
        };
        if session.transcript_bytes.saturating_add(content.len()) > self.limits.max_transcript_bytes
        {
            return Err(SessionCoordinatorError::state_budget(
                StateBudgetFailureCode::ByteCapacity,
            ));
        }
        let lease = self.acquire_state_lease(content.len())?;
        let Some(session) = self.sessions.get_mut(&scope) else {
            return Ok(());
        };
        let turn_ordinal = session.continuity.emitted_through_turn.unwrap_or(0);
        session.continuity.version = session.continuity.version.next()?;
        session.transcript_bytes = session.transcript_bytes.saturating_add(content.len());
        session.transcript.push(TranscriptEntry {
            role: "assistant".to_string(),
            content,
            turn_ordinal,
            origin: TranscriptOrigin::Endpoint,
        });
        session.state_leases.push(lease);
        Ok(())
    }

    async fn settle_terminal(
        &mut self,
        scope: ConversationSessionScope,
        output: &mut dyn DatasetActionSink,
    ) -> Result<(), SessionCoordinatorError> {
        if let Some(session) = self.sessions.get_mut(&scope) {
            session.continuity.terminal_through_turn = session.continuity.emitted_through_turn;
        }
        self.drain_ready(scope, output).await
    }

    async fn seal_explicit(
        &mut self,
        seal: SourceSeal,
        output: &mut dyn DatasetActionSink,
    ) -> Result<SessionSealReceipt, SessionCoordinatorError> {
        self.flush_reemissions(output).await?;
        // A finite seal is hard completeness evidence: a session that still
        // holds an unresolved causal gap can never complete, so it fails rather
        // than waiting forever, and the rest close under a verified seal.
        let sealed: Vec<(ConversationSessionScope, bool)> = self
            .sessions
            .iter()
            .map(|(scope, session)| (*scope, !session.pending.is_empty()))
            .collect();
        for (scope, has_causal_gap) in sealed {
            match self
                .limits
                .closure
                .decide(SessionClosureEvidence::FiniteSeal { has_causal_gap })
            {
                SessionClosureDecision::Close(_) => self.retire_session(scope),
                SessionClosureDecision::Wait => {}
                SessionClosureDecision::Fail(code) => {
                    return Err(SessionCoordinatorError::session(code));
                }
            }
        }
        self.publish_frontier(output).await?;
        Ok(SessionSealReceipt {
            digest: self.seal_digest(&seal),
            causal_frontier: self.causal_frontier.clone(),
        })
    }

    fn seal_digest(&self, seal: &SourceSeal) -> ContentDigest {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"aiperf.stream.session.conversation.seal.v1");
        hasher.update(seal.digest.as_bytes());
        hasher.update(
            &seal
                .final_position
                .map_or(u64::MAX, SourcePosition::get)
                .to_le_bytes(),
        );
        hasher.update(self.stream_identity.as_bytes());
        hasher.update(&(self.sessions.len() as u64).to_le_bytes());
        for (scope, session) in &self.sessions {
            hasher.update(scope.session_key.as_bytes());
            hasher.update(&session.continuity.version.get().to_le_bytes());
        }
        ContentDigest::from_bytes(*hasher.finalize().as_bytes())
    }

    /// Derive the causal frontier from current retained state.
    fn derive_frontier(&self) -> SessionCausalFrontier {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"aiperf.stream.session.conversation.frontier.v1");
        hasher.update(&self.next_global_sequence.to_le_bytes());
        for (scope, session) in &self.sessions {
            hasher.update(scope.session_key.as_bytes());
            hasher.update(&session.continuity.version.get().to_le_bytes());
        }
        SessionCausalFrontier {
            through_sequence: GlobalSequence::new(self.next_global_sequence),
            event_time: self.latest_event_time,
            digest: ContentDigest::from_bytes(*hasher.finalize().as_bytes()),
        }
    }

    async fn publish_frontier(
        &mut self,
        output: &mut dyn DatasetActionSink,
    ) -> Result<(), SessionCoordinatorError> {
        self.causal_frontier = self.derive_frontier();
        output
            .advance_causal_frontier(self.causal_frontier.clone())
            .await
    }

    fn observe_event_time(&mut self, event_time: Option<EventTimeUtc>) {
        let Some(candidate) = event_time else {
            return;
        };
        self.latest_event_time = Some(match self.latest_event_time {
            Some(existing) if existing >= candidate => existing,
            _ => candidate,
        });
    }

    fn acquire_state_lease(&self, bytes: usize) -> Result<BudgetLease, SessionCoordinatorError> {
        self.state_budget
            .try_acquire(1, bytes)
            .map_err(map_budget_error)
    }

    /// Prepare complete state, or roll the decode horizon back before the first
    /// mutation this payload does not represent.
    fn prepare_complete_or_rolled_back_view(
        &mut self,
        barrier: &CheckpointBarrier,
    ) -> Result<PreparedParticipantState, CheckpointError> {
        let (bytes, lease, first_unrepresented, item_count) = self.encode_within_budget()?;
        let payload = BudgetedCheckpointBytes::new(Bytes::from(bytes), lease)?;
        let mut represented = barrier.cut.clone();
        if let Some(position) = first_unrepresented {
            represented.decoded = DecodeHorizon::new(position);
        }
        PreparedParticipantState::new(
            self.run,
            self.participant_id.clone(),
            CONVERSATION_CHECKPOINT_SCHEMA_ID,
            CONVERSATION_CHECKPOINT_SCHEMA_VERSION,
            represented,
            item_count,
            payload,
        )
    }

    /// Encode complete state, dropping whole sessions from the tail of the
    /// encode order until the payload fits the checkpoint budget.
    fn encode_within_budget(
        &mut self,
    ) -> Result<(Vec<u8>, BudgetLease, Option<SourcePosition>, u64), CheckpointError> {
        let mut records: Vec<ConversationSessionRecordV1> =
            self.sessions.values().map(record_of).collect();
        // Tombstones are never dropped to fit: retiring a session and then
        // forgetting it would let the next incarnation resurrect it.
        let tombstones: Vec<QuarantineTombstoneRecordV1> =
            self.tombstones.iter().map(tombstone_record_of).collect();
        let mut first_unrepresented: Option<SourcePosition> = None;
        loop {
            let state = ConversationCheckpointStateV1 {
                program_semantic_digest: self.program_semantic_digest,
                stream_identity: self.stream_identity,
                sessions: records,
                tombstones: tombstones.clone(),
            };
            let bytes = rmp_serde::to_vec(&state).map_err(|error| CheckpointError::Storage {
                message: format!("could not encode conversation session state: {error}"),
            })?;
            match self.checkpoint_budget.try_acquire(1, bytes.len()) {
                Ok(lease) => {
                    let item_count = u64::try_from(state.sessions.len()).unwrap_or(u64::MAX);
                    return Ok((bytes, lease, first_unrepresented, item_count));
                }
                Err(BudgetError::Closed) => {
                    return Err(CheckpointError::ParticipantUnavailable {
                        participant: self.participant_id.clone(),
                    });
                }
                Err(_) => {}
            }
            records = state.sessions;
            let Some(dropped) = records.pop() else {
                return Err(CheckpointError::StateBudget {
                    participant: self.participant_id.clone(),
                    code: StateBudgetFailureCode::ByteCapacity,
                });
            };
            first_unrepresented = Some(match first_unrepresented {
                Some(existing) if existing <= dropped.first_source_position => existing,
                _ => dropped.first_source_position,
            });
        }
    }

    fn restore_sessions(
        &mut self,
        state: Option<CommittedParticipantState>,
    ) -> Result<(), CheckpointError> {
        self.initialization.initialize_once()?;
        let Some(state) = state else {
            return Ok(());
        };
        let decoded: ConversationCheckpointStateV1 = rmp_serde::from_slice(state.payload_bytes())
            .map_err(|error| CheckpointError::Storage {
            message: format!("could not decode conversation session state: {error}"),
        })?;
        if decoded.program_semantic_digest != self.program_semantic_digest
            || decoded.stream_identity != self.stream_identity
        {
            return Err(CheckpointError::ObjectVerification);
        }
        for tombstone in decoded.tombstones {
            self.tombstones
                .install(
                    StreamingInputDomainIdentity::new(
                        self.stream_identity,
                        tombstone.source_identity,
                    ),
                    tombstone.session_key,
                    tombstone.issue_id,
                    tombstone.causal_frontier,
                    tombstone.closure_proof,
                )
                .map_err(|_| CheckpointError::StateBudget {
                    participant: self.participant_id.clone(),
                    code: StateBudgetFailureCode::ItemCapacity,
                })?;
        }
        for record in decoded.sessions {
            let scope = *record.continuity.scope();
            let mut state_leases = Vec::with_capacity(record.transcript.len());
            let mut transcript_bytes = 0usize;
            for entry in &record.transcript {
                let lease = self
                    .state_budget
                    .try_acquire(1, entry.content.len())
                    .map_err(|_| CheckpointError::StateBudget {
                        participant: self.participant_id.clone(),
                        code: StateBudgetFailureCode::ByteCapacity,
                    })?;
                transcript_bytes = transcript_bytes.saturating_add(entry.content.len());
                state_leases.push(lease);
            }
            let receipts = record
                .receipts
                .into_iter()
                .map(|receipt| (receipt.record_id, receipt))
                .collect();
            let needs_reemission = record.continuity.in_flight_turn().is_some();
            self.sessions.insert(
                scope,
                ConversationSession {
                    continuity: record.continuity,
                    transcript: record.transcript,
                    transcript_bytes,
                    receipts,
                    pending: BTreeMap::new(),
                    last_action: record.last_action,
                    last_provenance: record.last_provenance,
                    last_stable_order: record.last_stable_order,
                    last_source_position: record.last_source_position,
                    first_source_position: record.first_source_position,
                    last_event_time: record.last_event_time,
                    needs_reemission,
                    state_leases,
                },
            );
        }
        Ok(())
    }
}

#[async_trait(?Send)]
impl StreamingSessionCoordinator for StreamingConversationCoordinator {
    async fn ingest(
        &mut self,
        fragment: StreamingSessionFragment,
        output: &mut dyn DatasetActionSink,
    ) -> Result<(), SessionCoordinatorError> {
        self.ingest_mutation(fragment, output).await
    }

    async fn advance_watermark(
        &mut self,
        watermark: SessionWatermark,
        output: &mut dyn DatasetActionSink,
    ) -> Result<(), SessionCoordinatorError> {
        self.apply_watermark(watermark, output).await
    }

    async fn observe_execution(
        &mut self,
        event: ActionExecutionEvent,
        output: &mut dyn DatasetActionSink,
    ) -> Result<(), SessionCoordinatorError> {
        self.apply_execution(event, output).await
    }

    async fn seal(
        &mut self,
        seal: SourceSeal,
        output: &mut dyn DatasetActionSink,
    ) -> Result<SessionSealReceipt, SessionCoordinatorError> {
        self.seal_explicit(seal, output).await
    }
}

#[async_trait(?Send)]
impl StreamingCheckpointParticipant for StreamingConversationCoordinator {
    fn participant_id(&self) -> CheckpointParticipantId {
        self.participant_id.clone()
    }

    async fn checkpoint_view(
        &mut self,
        barrier: &CheckpointBarrier,
    ) -> Result<PreparedParticipantState, CheckpointError> {
        self.prepare_complete_or_rolled_back_view(barrier)
    }

    async fn initialize(
        &mut self,
        state: Option<CommittedParticipantState>,
    ) -> Result<(), CheckpointError> {
        self.restore_sessions(state)
    }

    async fn checkpoint_committed(
        &mut self,
        receipt: &CommittedParticipantReceipt,
    ) -> Result<(), CheckpointError> {
        // Retained state is the live state; there is no separate pre-cut copy to
        // release, so the notification is idempotent by construction.
        let _ = receipt;
        Ok(())
    }
}

/// Mutation vocabulary this program accepts, already projected to its content.
#[derive(Debug)]
enum AcceptedMutation {
    Turn {
        role: String,
        content: Vec<u8>,
        turn_ordinal: u64,
    },
    Close {
        reason: String,
    },
}

impl AcceptedMutation {
    fn classify(mutation: &SessionMutationV1) -> Result<Self, SessionCoordinatorError> {
        match mutation {
            SessionMutationV1::ConversationTurn(turn) => Ok(Self::Turn {
                role: turn.role.clone(),
                content: turn.content.clone(),
                turn_ordinal: turn.turn_ordinal,
            }),
            SessionMutationV1::SessionClose(close) => Ok(Self::Close {
                reason: close.reason.clone(),
            }),
            SessionMutationV1::AgentEvent(_)
            | SessionMutationV1::GraphNode(_)
            | SessionMutationV1::GraphEdge(_)
            | SessionMutationV1::DeferredRecordedRequest(_) => Err(
                SessionCoordinatorError::session(SessionFailureCode::UnsupportedMutation),
            ),
        }
    }

    fn canonical_content_digest(&self) -> ContentDigest {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"aiperf.stream.session.conversation.content.v1");
        match self {
            Self::Turn {
                role,
                content,
                turn_ordinal,
            } => {
                hasher.update(&[0u8]);
                hasher.update(&turn_ordinal.to_le_bytes());
                hasher.update(&(role.len() as u64).to_le_bytes());
                hasher.update(role.as_bytes());
                hasher.update(&(content.len() as u64).to_le_bytes());
                hasher.update(content);
            }
            Self::Close { reason } => {
                hasher.update(&[1u8]);
                hasher.update(&(reason.len() as u64).to_le_bytes());
                hasher.update(reason.as_bytes());
            }
        }
        ContentDigest::from_bytes(*hasher.finalize().as_bytes())
    }
}

fn logical_receipt(
    fragment: &StreamingSessionFragment,
    accepted: &AcceptedMutation,
) -> LogicalRecordReceipt {
    LogicalRecordReceipt {
        record_id: fragment.record_id,
        content_digest: accepted.canonical_content_digest(),
        provenance: fragment.provenance.clone(),
    }
}

/// Encode one transcript into canonical request bytes at exactly its length.
///
/// `ExecutableDatasetAction::new` charges the payload's `Vec` capacity, so the
/// buffer is trimmed to its exact length before it is handed over.
fn encode_transcript(transcript: &[TranscriptEntry]) -> Result<Vec<u8>, SessionCoordinatorError> {
    let encoded = rmp_serde::to_vec(transcript).map_err(|_| {
        SessionCoordinatorError::state_budget(StateBudgetFailureCode::PermanentError)
    })?;
    Ok(encoded.into_boxed_slice().into_vec())
}

fn acquire_retained(
    budget: &StreamingResourceBudget,
    bytes: usize,
) -> Result<crate::streaming::unit::RetainedContentLease, SessionCoordinatorError> {
    let lease = budget.try_acquire(1, bytes).map_err(map_budget_error)?;
    Ok(SessionFragmentLease::try_from(lease)
        .map_err(map_budget_error)?
        .into_retained())
}

const fn map_budget_error(error: BudgetError) -> SessionCoordinatorError {
    match error {
        BudgetError::CapacityUnavailable
        | BudgetError::RequestExceedsCapacity
        | BudgetError::ActionPayloadUndercharged { .. } => {
            SessionCoordinatorError::state_budget(StateBudgetFailureCode::ByteCapacity)
        }
        _ => SessionCoordinatorError::state_budget(StateBudgetFailureCode::ItemCapacity),
    }
}

fn tombstone_record_of(
    tombstone: &crate::streaming::session::SessionQuarantineTombstone,
) -> QuarantineTombstoneRecordV1 {
    QuarantineTombstoneRecordV1 {
        source_identity: *tombstone.input_domain().source_identity(),
        session_key: tombstone.session_key(),
        issue_id: tombstone.issue_id(),
        causal_frontier: tombstone.causal_frontier().clone(),
        closure_proof: tombstone.closure_proof(),
    }
}

fn record_of(session: &ConversationSession) -> ConversationSessionRecordV1 {
    ConversationSessionRecordV1 {
        continuity: session.continuity.clone(),
        transcript: session.transcript.clone(),
        receipts: session.receipts.values().cloned().collect(),
        last_action: session.last_action,
        last_provenance: session.last_provenance.clone(),
        last_stable_order: session.last_stable_order,
        last_source_position: session.last_source_position,
        first_source_position: session.first_source_position,
        last_event_time: session.last_event_time,
    }
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ConversationCheckpointStateV1 {
    program_semantic_digest: ContentDigest,
    stream_identity: ContentDigest,
    sessions: Vec<ConversationSessionRecordV1>,
    tombstones: Vec<QuarantineTombstoneRecordV1>,
}

/// Durable projection of one retained quarantine tombstone.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct QuarantineTombstoneRecordV1 {
    source_identity: ImmutableObjectIdentity,
    session_key: StableSessionKey,
    issue_id: ContentDigest,
    causal_frontier: SessionCausalFrontier,
    closure_proof: SessionQuarantineClosureProof,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ConversationSessionRecordV1 {
    continuity: ConversationContinuity,
    transcript: Vec<TranscriptEntry>,
    receipts: Vec<LogicalRecordReceipt>,
    last_action: Option<StableActionId>,
    last_provenance: UnitProvenance,
    last_stable_order: StableOrderKey,
    last_source_position: SourcePosition,
    first_source_position: SourcePosition,
    last_event_time: Option<EventTimeUtc>,
}
