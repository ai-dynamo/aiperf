// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Attached telemetry archive orchestration for scheduled benchmark runs.
//!
//! This module is deliberately source-implementation agnostic. Sidecar
//! factories prepare each physical telemetry source exactly once, then expose
//! its one run-owned driver through [`RunOwnedTelemetryDriver`]. The bridge
//! owns phase membership, atomically sealed source-cardinal boundary plans,
//! synchronous native-first fanout, nonblocking archive admission, visible
//! loss, lifecycle observation, and archive-independent native parity facts.
//! It never constructs a source, opens a transport, or starts a second scrape
//! loop.

use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::{self, Debug, Display, Formatter};
use std::future::Future;
use std::pin::Pin;
use std::rc::Rc;

use aiperf_telemetry_archive::{
    AdmissionRejection, BoundaryCapturePlan, BoundaryPlanError, BoundaryPlanRegistry,
    BoundaryReference, BoundaryRole, CanonicalJsonValue, Digest, LossKindV1, LossReasonV1,
    SourceBoundarySnapshotCommand, domain_digest,
};
use aiperf_timing::{
    PhaseBranchStats, PhaseCompletionReason, PhaseConfig, PhaseObserver, PhaseStats,
};

use crate::telemetry_archive_owner::ArchiveLifecycleObservation;

/// LocalSet-compatible future returned by a run-owned source driver.
pub type LocalBoundaryFuture<T> = Pin<Box<dyn Future<Output = T> + 'static>>;

/// One terminal outcome for every reference in a source boundary command.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SourceBoundaryTerminal {
    /// One physical attempt satisfied every structured reference.
    Attempt {
        /// Per-source sequence assigned to the physical attempt.
        source_record_seq: u64,
        /// Per-source network sequence when IO began.
        request_attempt_seq: Option<u64>,
        /// Complete references copied from the sealed source command.
        boundary_refs: Vec<BoundaryReference>,
    },
    /// One exact loss row closed every structured reference.
    Loss {
        /// Closed archive loss class.
        loss_kind: LossKindV1,
        /// Closed reason paired with the loss class.
        reason: LossReasonV1,
        /// Complete references copied from the sealed source command.
        boundary_refs: Vec<BoundaryReference>,
    },
}

impl SourceBoundaryTerminal {
    fn boundary_refs(&self) -> &[BoundaryReference] {
        match self {
            Self::Attempt { boundary_refs, .. } | Self::Loss { boundary_refs, .. } => boundary_refs,
        }
    }
}

/// Source-scoped result returned only after a planned capture is terminal.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SourceBoundaryResult {
    /// Stable physical source identity.
    pub source_id: String,
    /// Stable adjacent-phase transition identity.
    pub transition_id: String,
    /// Exactly one attempt-or-loss terminal result.
    pub terminal: SourceBoundaryTerminal,
}

/// One already prepared physical source driver shared by every phase.
///
/// `submit_boundary` must only enqueue into capacity reserved during source
/// preparation. The returned future waits for the exact attempt-or-loss result;
/// submission itself must not block and must never start another physical
/// source. Phase membership is captured by the driver at each snapshot instant.
pub trait RunOwnedTelemetryDriver: Debug {
    /// Stable prepared source ID used by attachment resolution.
    fn source_id(&self) -> &str;

    /// Add or remove one active phase from future continuous snapshot membership.
    fn set_phase_active(&self, phase_id: &str, active: bool) -> Result<(), AttachedTelemetryError>;

    /// Submit one command that was already validated in a complete sealed plan.
    fn submit_boundary(
        &self,
        command: SourceBoundarySnapshotCommand,
    ) -> LocalBoundaryFuture<Result<SourceBoundaryResult, AttachedTelemetryError>>;
}

/// Run-owned source inventory and attached boundary coordinator.
///
/// The inventory may contain sources that feed native metrics without archival.
/// `archived_source_ids` selects an exact non-empty subset and never carries
/// source configuration, so attaching an archive cannot duplicate physical IO.
pub struct AttachedTelemetryRuntime {
    drivers: BTreeMap<String, Rc<dyn RunOwnedTelemetryDriver>>,
    archived_source_ids: BTreeSet<String>,
    boundary_registry: RefCell<BoundaryPlanRegistry>,
    active_phases: RefCell<BTreeSet<String>>,
}

impl AttachedTelemetryRuntime {
    /// Resolve an attachment against already prepared unique physical sources.
    pub fn new(
        drivers: impl IntoIterator<Item = Rc<dyn RunOwnedTelemetryDriver>>,
        archived_source_ids: impl IntoIterator<Item = String>,
    ) -> Result<Self, AttachedTelemetryError> {
        let mut by_id = BTreeMap::new();
        for driver in drivers {
            let source_id = driver.source_id().to_owned();
            validate_identifier("prepared source_id", &source_id)?;
            if by_id.insert(source_id.clone(), driver).is_some() {
                return Err(AttachedTelemetryError::DuplicatePreparedSource(source_id));
            }
        }
        if by_id.is_empty() {
            return Err(AttachedTelemetryError::NoPreparedSources);
        }

        let mut selected = BTreeSet::new();
        for source_id in archived_source_ids {
            validate_identifier("attachment source_id", &source_id)?;
            if !selected.insert(source_id.clone()) {
                return Err(AttachedTelemetryError::DuplicateAttachedSource(source_id));
            }
            if !by_id.contains_key(&source_id) {
                return Err(AttachedTelemetryError::UnknownAttachedSource {
                    source_id,
                    available: by_id.keys().cloned().collect(),
                });
            }
        }
        if selected.is_empty() {
            return Err(AttachedTelemetryError::NoAttachedSources);
        }
        let boundary_registry = BoundaryPlanRegistry::new(selected.iter().cloned())
            .map_err(AttachedTelemetryError::BoundaryPlan)?;
        Ok(Self {
            drivers: by_id,
            archived_source_ids: selected,
            boundary_registry: RefCell::new(boundary_registry),
            active_phases: RefCell::new(BTreeSet::new()),
        })
    }

    /// All physical source IDs in deterministic order.
    pub fn physical_source_ids(&self) -> impl ExactSizeIterator<Item = &str> {
        self.drivers.keys().map(String::as_str)
    }

    /// Archive-selected source IDs in deterministic order.
    pub fn archived_source_ids(&self) -> impl ExactSizeIterator<Item = &str> {
        self.archived_source_ids.iter().map(String::as_str)
    }

    /// Activate one phase on every physical source before its first snapshot.
    pub fn activate_phase(&self, phase_id: &str) -> Result<(), AttachedTelemetryError> {
        validate_identifier("phase_id", phase_id)?;
        if !self.active_phases.borrow_mut().insert(phase_id.to_owned()) {
            return Err(AttachedTelemetryError::PhaseAlreadyActive(
                phase_id.to_owned(),
            ));
        }
        if let Err(error) = self.set_phase_on_all_sources(phase_id, true) {
            self.active_phases.borrow_mut().remove(phase_id);
            let _ = self.set_phase_on_all_sources(phase_id, false);
            return Err(error);
        }
        Ok(())
    }

    /// Deactivate one phase only after its authoritative complete observation.
    pub fn deactivate_phase(&self, phase_id: &str) -> Result<(), AttachedTelemetryError> {
        if !self.active_phases.borrow_mut().remove(phase_id) {
            return Err(AttachedTelemetryError::PhaseNotActive(phase_id.to_owned()));
        }
        if let Err(error) = self.set_phase_on_all_sources(phase_id, false) {
            self.active_phases.borrow_mut().insert(phase_id.to_owned());
            let _ = self.set_phase_on_all_sources(phase_id, true);
            return Err(error);
        }
        Ok(())
    }

    /// Copy the current run-owned membership in deterministic order.
    pub fn active_phases(&self) -> Vec<String> {
        self.active_phases.borrow().iter().cloned().collect()
    }

    /// Build one deterministic source-cardinal adjacent-phase transition plan.
    ///
    /// At least one phase role is required. When both roles are present, each
    /// source gets its own coalescing group and one physical snapshot carries
    /// the end and start references.
    pub fn build_transition_plan(
        &self,
        transition_id: impl Into<String>,
        ending_phase_id: Option<&str>,
        starting_phase_id: Option<&str>,
        absolute_deadline_ns: i64,
    ) -> Result<BoundaryCapturePlan, AttachedTelemetryError> {
        let transition_id = transition_id.into();
        validate_identifier("transition_id", &transition_id)?;
        if absolute_deadline_ns <= 0 {
            return Err(AttachedTelemetryError::InvalidBoundaryDeadline(
                absolute_deadline_ns,
            ));
        }
        if ending_phase_id.is_none() && starting_phase_id.is_none() {
            return Err(AttachedTelemetryError::EmptyTransition);
        }
        if let Some(phase_id) = ending_phase_id {
            validate_identifier("ending phase_id", phase_id)?;
        }
        if let Some(phase_id) = starting_phase_id {
            validate_identifier("starting phase_id", phase_id)?;
        }
        if ending_phase_id == starting_phase_id && ending_phase_id.is_some() {
            return Err(AttachedTelemetryError::SameAdjacentPhase(
                ending_phase_id.unwrap_or_default().to_owned(),
            ));
        }

        let mut commands = Vec::with_capacity(self.archived_source_ids.len());
        for source_id in &self.archived_source_ids {
            let coalesced = ending_phase_id.is_some() && starting_phase_id.is_some();
            let group = coalesced.then(|| {
                stable_boundary_id(
                    "aiperf.telemetry.boundary-group.v1",
                    &transition_id,
                    source_id,
                    "adjacent",
                    "group",
                )
            });
            let mut subscribers = Vec::with_capacity(
                usize::from(ending_phase_id.is_some()) + usize::from(starting_phase_id.is_some()),
            );
            if let Some(phase_id) = ending_phase_id {
                subscribers.push(boundary_reference(
                    &transition_id,
                    source_id,
                    phase_id,
                    BoundaryRole::PhaseEnd,
                    group.as_deref(),
                ));
            }
            if let Some(phase_id) = starting_phase_id {
                subscribers.push(boundary_reference(
                    &transition_id,
                    source_id,
                    phase_id,
                    BoundaryRole::PhaseStart,
                    group.as_deref(),
                ));
            }
            commands.push(SourceBoundarySnapshotCommand {
                source_id: source_id.clone(),
                coalescing_group_id: group,
                subscribers,
                absolute_deadline_ns,
            });
        }
        Ok(BoundaryCapturePlan {
            transition_id,
            commands,
        })
    }

    /// Atomically seal a complete plan, route every command, and validate joins.
    ///
    /// Registration completes before the first driver sees a command. Every
    /// driver submission is synchronous and nonblocking; only then are result
    /// futures awaited. A driver must close every reference as one attempt or
    /// one exact loss rather than returning a partial boundary result.
    pub async fn capture_boundary_plan(
        &self,
        plan: BoundaryCapturePlan,
    ) -> Result<Vec<SourceBoundaryResult>, AttachedTelemetryError> {
        let sealed = self
            .boundary_registry
            .borrow_mut()
            .seal(plan)
            .map_err(AttachedTelemetryError::BoundaryPlan)?;
        let plan = sealed.into_plan();
        let transition_id = plan.transition_id.clone();
        let mut pending = Vec::with_capacity(plan.commands.len());
        for command in plan.commands {
            let driver = self
                .drivers
                .get(&command.source_id)
                .expect("sealed boundary plans contain only prepared sources");
            let expected = command.clone();
            let future = driver.submit_boundary(command);
            pending.push((expected, future));
        }

        let mut results = Vec::with_capacity(pending.len());
        for (expected, future) in pending {
            let result = future.await?;
            validate_boundary_result(&transition_id, &expected, &result)?;
            results.push(result);
        }
        results.sort_by(|left, right| left.source_id.cmp(&right.source_id));
        Ok(results)
    }

    /// Derive immutable observer contexts from a sealed-plan-shaped value.
    ///
    /// Callers install these contexts before the matching authoritative phase
    /// observer transition. No sidecar reconstructs boundary membership.
    pub fn phase_transition_contexts(
        &self,
        plan: &BoundaryCapturePlan,
    ) -> Result<Vec<PhaseTransitionContext>, AttachedTelemetryError> {
        let mut by_phase = BTreeMap::<(String, BoundaryRole), Vec<BoundaryReference>>::new();
        for command in &plan.commands {
            if !self.archived_source_ids.contains(&command.source_id) {
                return Err(AttachedTelemetryError::UnknownBoundarySource(
                    command.source_id.clone(),
                ));
            }
            for reference in &command.subscribers {
                if reference.transition_id != plan.transition_id
                    || reference.source_id != command.source_id
                {
                    return Err(AttachedTelemetryError::BoundaryResultMismatch(
                        "plan reference identity is inconsistent".to_owned(),
                    ));
                }
                by_phase
                    .entry((reference.phase_id.clone(), reference.role))
                    .or_default()
                    .push(reference.clone());
            }
        }
        let mut contexts = Vec::with_capacity(by_phase.len());
        for ((phase_id, role), mut boundaries) in by_phase {
            boundaries.sort();
            contexts.push(PhaseTransitionContext {
                transition_id: plan.transition_id.clone(),
                phase_id,
                role,
                boundaries,
            });
        }
        Ok(contexts)
    }

    fn set_phase_on_all_sources(
        &self,
        phase_id: &str,
        active: bool,
    ) -> Result<(), AttachedTelemetryError> {
        for driver in self.drivers.values() {
            driver.set_phase_active(phase_id, active)?;
        }
        Ok(())
    }
}

impl Debug for AttachedTelemetryRuntime {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AttachedTelemetryRuntime")
            .field("physical_sources", &self.drivers.keys().collect::<Vec<_>>())
            .field("archived_sources", &self.archived_source_ids)
            .field("active_phases", &self.active_phases.borrow())
            .finish_non_exhaustive()
    }
}

/// Immutable boundary references delivered with one authoritative phase event.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PhaseTransitionContext {
    /// Transition allocated once before forced source capture.
    pub transition_id: String,
    /// Phase receiving the lifecycle references.
    pub phase_id: String,
    /// Start/end role shared by this phase-local context.
    pub role: BoundaryRole,
    /// One complete reference per archive-selected physical source.
    pub boundaries: Vec<BoundaryReference>,
}

impl PhaseTransitionContext {
    fn validate(&self, expected_sources: &BTreeSet<String>) -> Result<(), AttachedTelemetryError> {
        validate_identifier("transition context transition_id", &self.transition_id)?;
        validate_identifier("transition context phase_id", &self.phase_id)?;
        let mut sources = BTreeSet::new();
        for reference in &self.boundaries {
            if reference.transition_id != self.transition_id
                || reference.phase_id != self.phase_id
                || reference.role != self.role
            {
                return Err(AttachedTelemetryError::InvalidTransitionContext(
                    self.phase_id.clone(),
                ));
            }
            if !sources.insert(reference.source_id.clone()) {
                return Err(AttachedTelemetryError::DuplicateTransitionSource {
                    phase_id: self.phase_id.clone(),
                    source_id: reference.source_id.clone(),
                });
            }
        }
        if sources != *expected_sources {
            return Err(AttachedTelemetryError::TransitionSourceCardinality {
                phase_id: self.phase_id.clone(),
                expected: expected_sources.iter().cloned().collect(),
                actual: sources.into_iter().collect(),
            });
        }
        Ok(())
    }
}

/// Stable identity and phase membership of one physical attempt.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PhysicalAttemptContext {
    /// Stable source ID.
    pub source_id: String,
    /// Per-source sequence assigned once by the run-owned driver.
    pub source_record_seq: u64,
    /// Network-attempt sequence when IO began.
    pub request_attempt_seq: Option<u64>,
    /// Authoritative snapshot Clock, when one exists.
    pub capture_ns: Option<i64>,
    /// Active phase IDs captured at the same snapshot instant.
    pub active_phase_ids: BTreeSet<String>,
    /// Explicit boundary subscribers, independent of active membership.
    pub boundary_refs: Vec<BoundaryReference>,
}

impl PhysicalAttemptContext {
    fn validate(&self) -> Result<(), AttachedTelemetryError> {
        validate_identifier("attempt source_id", &self.source_id)?;
        for phase_id in &self.active_phase_ids {
            validate_identifier("attempt phase_id", phase_id)?;
        }
        let mut boundary_keys = BTreeSet::new();
        for reference in &self.boundary_refs {
            if reference.source_id != self.source_id {
                return Err(AttachedTelemetryError::AttemptBoundarySourceMismatch {
                    attempt_source: self.source_id.clone(),
                    boundary_source: reference.source_id.clone(),
                });
            }
            if !boundary_keys.insert((
                reference.transition_id.clone(),
                reference.source_id.clone(),
                reference.boundary_id.clone(),
            )) {
                return Err(AttachedTelemetryError::DuplicateAttemptBoundary(
                    reference.boundary_id.clone(),
                ));
            }
        }
        Ok(())
    }
}

/// One decoded physical attempt split into native and archive projections.
#[derive(Debug)]
pub struct AttachedPhysicalAttempt<NativeRecord, ArchiveRecord> {
    /// Shared physical attempt facts.
    pub context: PhysicalAttemptContext,
    /// Optional supported native projection.
    pub native_record: Option<NativeRecord>,
    /// Every-outcome archive projection.
    pub archive_record: ArchiveRecord,
}

/// Receipt produced synchronously by authoritative native delivery.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NativeDeliveryReceipt {
    /// Whether a supported native record was delivered.
    pub delivered_record: bool,
    /// Canonical digest of the native event after delivery semantics are fixed.
    pub native_event_digest: Digest,
}

/// Synchronous native accumulator/projection seam.
pub trait NativeAttemptDelivery<NativeRecord>: Debug {
    /// Deliver exactly once before archive admission and return parity evidence.
    fn deliver(
        &self,
        context: &PhysicalAttemptContext,
        record: Option<&NativeRecord>,
    ) -> Result<NativeDeliveryReceipt, AttachedTelemetryError>;
}

/// Projection offered to the bounded archive ingress after native delivery.
#[derive(Debug)]
pub struct AttachedArchiveProjection<ArchiveRecord> {
    /// Shared physical attempt facts.
    pub context: PhysicalAttemptContext,
    /// Every-outcome archive entity.
    pub record: ArchiveRecord,
}

/// Exact visible loss captured when archive projection cannot be admitted.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VisibleArchiveLoss {
    /// Stable source ID.
    pub source_id: String,
    /// First and last omitted source sequence; one attempt in v1 fanout.
    pub source_record_seq: u64,
    /// Omitted request attempt when IO began.
    pub request_attempt_seq: Option<u64>,
    /// Closed rejection reason from the nonblocking admission policy.
    pub rejection: AdmissionRejection,
    /// Exact boundary references that must terminally join through this loss.
    pub boundary_refs: Vec<BoundaryReference>,
}

/// Nonblocking data and reserved-loss ingress for an attached archive.
///
/// `try_admit` must return immediately and may only take already reserved
/// capacity. `record_visible_loss` uses a separately reserved control lane;
/// returning success means the exact loss identity was accepted for durable
/// terminalization. Neither operation may await local or remote durability.
pub trait AttachedArchiveIngress<ArchiveRecord>: Debug {
    /// Try to enqueue one every-outcome archive projection without waiting.
    fn try_admit(
        &self,
        projection: AttachedArchiveProjection<ArchiveRecord>,
    ) -> Result<(), AdmissionRejection>;

    /// Record exact loss after an issued native event was not archived.
    fn record_visible_loss(&self, loss: VisibleArchiveLoss) -> Result<(), AttachedTelemetryError>;
}

/// Archive admission fact exposed only after native delivery.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AttachedArchiveAdmission {
    /// No archive was attached; native behavior is unchanged.
    Disabled,
    /// Projection entered the bounded archive ingress.
    Accepted,
    /// Projection was rejected and one visible loss was accepted instead.
    Rejected(AdmissionRejection),
}

/// Post-native factual observation containing no decoded entity or body bytes.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PostNativeAttemptObservation {
    /// Shared physical identity and membership.
    pub context: PhysicalAttemptContext,
    /// Authoritative native delivery receipt.
    pub native: NativeDeliveryReceipt,
    /// Immediate archive admission result.
    pub archive_admission: AttachedArchiveAdmission,
}

/// Low-rate post-native attempt observer for health and tests.
pub trait AttachedAttemptObserver: Debug {
    /// Observe one terminal factual result after native and archive fanout.
    fn observe(&self, observation: &PostNativeAttemptObservation);
}

/// Observer that intentionally discards post-native attempt facts.
#[derive(Clone, Copy, Debug, Default)]
pub struct NoopAttachedAttemptObserver;

impl AttachedAttemptObserver for NoopAttachedAttemptObserver {
    fn observe(&self, _observation: &PostNativeAttemptObservation) {}
}

/// Archive-independent native telemetry event retained for parity comparison.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NativeTelemetryParityEventV1 {
    /// Native observation order within this run.
    pub event_seq: u64,
    /// Stable source identity.
    pub source_id: String,
    /// Per-source physical event sequence.
    pub source_record_seq: u64,
    /// Per-source network sequence when IO began.
    pub request_attempt_seq: Option<u64>,
    /// Source snapshot Clock used by native semantics.
    pub capture_ns: Option<i64>,
    /// Active phase membership in deterministic order.
    pub active_phase_ids: Vec<String>,
    /// Whether the native domain accepted a record.
    pub delivered_record: bool,
    /// Canonical digest returned by native delivery.
    pub native_event_digest: Digest,
}

impl NativeTelemetryParityEventV1 {
    fn canonical_value(&self) -> CanonicalJsonValue {
        CanonicalJsonValue::object([
            (
                "active_phase_ids".to_owned(),
                CanonicalJsonValue::Array(
                    self.active_phase_ids
                        .iter()
                        .cloned()
                        .map(CanonicalJsonValue::String)
                        .collect(),
                ),
            ),
            ("capture_ns".to_owned(), optional_i64(self.capture_ns)),
            (
                "delivered_record".to_owned(),
                CanonicalJsonValue::Bool(self.delivered_record),
            ),
            (
                "event_seq".to_owned(),
                CanonicalJsonValue::Integer(i128::from(self.event_seq)),
            ),
            (
                "native_event_digest".to_owned(),
                CanonicalJsonValue::String(self.native_event_digest.to_tagged_hex()),
            ),
            (
                "request_attempt_seq".to_owned(),
                optional_u64(self.request_attempt_seq),
            ),
            (
                "source_id".to_owned(),
                CanonicalJsonValue::String(self.source_id.clone()),
            ),
            (
                "source_record_seq".to_owned(),
                CanonicalJsonValue::Integer(i128::from(self.source_record_seq)),
            ),
        ])
        .expect("native parity fields are statically unique")
    }
}

/// Frozen telemetry slice of `NativeMeasurementParityV1`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NativeTelemetryParityFactsV1 {
    /// Ordered native events, excluding all archive-only facts.
    pub events: Vec<NativeTelemetryParityEventV1>,
    /// Domain-separated digest of the canonical event array.
    pub digest: Digest,
}

impl NativeTelemetryParityFactsV1 {
    /// Serialize the frozen telemetry parity descriptor canonically.
    #[must_use]
    pub fn canonical_bytes(&self) -> Vec<u8> {
        CanonicalJsonValue::object([
            (
                "digest".to_owned(),
                CanonicalJsonValue::String(self.digest.to_tagged_hex()),
            ),
            (
                "events".to_owned(),
                CanonicalJsonValue::Array(
                    self.events
                        .iter()
                        .map(NativeTelemetryParityEventV1::canonical_value)
                        .collect(),
                ),
            ),
            (
                "version".to_owned(),
                CanonicalJsonValue::String("native_telemetry_parity_v1".to_owned()),
            ),
        ])
        .expect("native parity descriptor fields are statically unique")
        .to_bytes()
    }
}

/// Run-owned recorder proving archive on/off consumes one native event stream.
#[derive(Default)]
pub struct NativeTelemetryParityRecorder {
    events: RefCell<Vec<NativeTelemetryParityEventV1>>,
    identities: RefCell<BTreeSet<(String, u64)>>,
}

impl NativeTelemetryParityRecorder {
    /// Record one native delivery before any archive admission fact exists.
    pub fn record(
        &self,
        context: &PhysicalAttemptContext,
        receipt: NativeDeliveryReceipt,
    ) -> Result<(), AttachedTelemetryError> {
        let identity = (context.source_id.clone(), context.source_record_seq);
        if !self.identities.borrow_mut().insert(identity.clone()) {
            return Err(AttachedTelemetryError::DuplicateNativeAttempt {
                source_id: identity.0,
                source_record_seq: identity.1,
            });
        }
        let event_seq = u64::try_from(self.events.borrow().len())
            .map_err(|_| AttachedTelemetryError::SequenceOverflow)?;
        self.events.borrow_mut().push(NativeTelemetryParityEventV1 {
            event_seq,
            source_id: context.source_id.clone(),
            source_record_seq: context.source_record_seq,
            request_attempt_seq: context.request_attempt_seq,
            capture_ns: context.capture_ns,
            active_phase_ids: context.active_phase_ids.iter().cloned().collect(),
            delivered_record: receipt.delivered_record,
            native_event_digest: receipt.native_event_digest,
        });
        Ok(())
    }

    /// Snapshot canonical parity facts without closing further observation.
    pub fn snapshot(&self) -> NativeTelemetryParityFactsV1 {
        let events = self.events.borrow().clone();
        let values = CanonicalJsonValue::Array(
            events
                .iter()
                .map(NativeTelemetryParityEventV1::canonical_value)
                .collect(),
        );
        let bytes = values.to_bytes();
        NativeTelemetryParityFactsV1 {
            events,
            digest: domain_digest("aiperf.native-telemetry-parity-events.v1", &[&bytes]),
        }
    }
}

impl Debug for NativeTelemetryParityRecorder {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("NativeTelemetryParityRecorder")
            .field("event_count", &self.events.borrow().len())
            .finish()
    }
}

/// Fixed-size attached archive health facts for native-v2 reporting.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AttachedArchiveHealthFacts {
    /// Every physical attempt delivered natively through this fanout.
    pub native_attempts: u64,
    /// Attempts accepted by archive ingress.
    pub archive_accepted: u64,
    /// Attempts replaced by explicit archive loss.
    pub archive_rejected: u64,
    /// Whether exact attached coverage remains complete.
    pub complete: bool,
    /// First bounded archive failure or rejection classification.
    pub first_failure: Option<String>,
}

impl Default for AttachedArchiveHealthFacts {
    fn default() -> Self {
        Self {
            complete: true,
            native_attempts: 0,
            archive_accepted: 0,
            archive_rejected: 0,
            first_failure: None,
        }
    }
}

/// Native-first attached attempt fanout shared by archive-off and archive-on.
pub struct AttachedAttemptFanout<NativeRecord, ArchiveRecord> {
    native: Rc<dyn NativeAttemptDelivery<NativeRecord>>,
    archive: Option<Rc<dyn AttachedArchiveIngress<ArchiveRecord>>>,
    observer: Rc<dyn AttachedAttemptObserver>,
    parity: Rc<NativeTelemetryParityRecorder>,
    health: RefCell<AttachedArchiveHealthFacts>,
    archive_required: bool,
}

impl<NativeRecord, ArchiveRecord> AttachedAttemptFanout<NativeRecord, ArchiveRecord> {
    /// Compose one physical-attempt fanout without constructing a source.
    pub fn new(
        native: Rc<dyn NativeAttemptDelivery<NativeRecord>>,
        archive: Option<Rc<dyn AttachedArchiveIngress<ArchiveRecord>>>,
        observer: Rc<dyn AttachedAttemptObserver>,
        parity: Rc<NativeTelemetryParityRecorder>,
        archive_required: bool,
    ) -> Self {
        Self {
            native,
            archive,
            observer,
            parity,
            health: RefCell::new(AttachedArchiveHealthFacts::default()),
            archive_required,
        }
    }

    /// Deliver native semantics, then attempt nonblocking archive admission.
    ///
    /// A rejected archive projection is converted into exact visible loss
    /// before the factual observer runs. Required mode reports failure only
    /// after native delivery and loss capture, so it cannot perturb formulas.
    pub fn process(
        &self,
        attempt: AttachedPhysicalAttempt<NativeRecord, ArchiveRecord>,
    ) -> Result<AttachedArchiveAdmission, AttachedTelemetryError> {
        attempt.context.validate()?;
        let observer_context = attempt.context.clone();
        let native_receipt = self
            .native
            .deliver(&attempt.context, attempt.native_record.as_ref())?;
        self.parity.record(&attempt.context, native_receipt)?;
        checked_increment(&mut self.health.borrow_mut().native_attempts)?;

        let admission = if let Some(archive) = &self.archive {
            let context = attempt.context.clone();
            match archive.try_admit(AttachedArchiveProjection {
                context: attempt.context,
                record: attempt.archive_record,
            }) {
                Ok(()) => {
                    checked_increment(&mut self.health.borrow_mut().archive_accepted)?;
                    AttachedArchiveAdmission::Accepted
                }
                Err(rejection) => {
                    let loss = VisibleArchiveLoss {
                        source_id: context.source_id.clone(),
                        source_record_seq: context.source_record_seq,
                        request_attempt_seq: context.request_attempt_seq,
                        rejection,
                        boundary_refs: context.boundary_refs.clone(),
                    };
                    archive.record_visible_loss(loss)?;
                    let mut health = self.health.borrow_mut();
                    checked_increment(&mut health.archive_rejected)?;
                    health.complete = false;
                    health
                        .first_failure
                        .get_or_insert_with(|| format!("archive admission {rejection}"));
                    AttachedArchiveAdmission::Rejected(rejection)
                }
            }
        } else {
            AttachedArchiveAdmission::Disabled
        };

        self.observer.observe(&PostNativeAttemptObservation {
            context: observer_context,
            native: native_receipt,
            archive_admission: admission,
        });

        if self.archive_required
            && let AttachedArchiveAdmission::Rejected(rejection) = admission
        {
            return Err(AttachedTelemetryError::RequiredArchiveRejected(rejection));
        }
        Ok(admission)
    }

    /// Copy bounded health counters for report assembly.
    pub fn health(&self) -> AttachedArchiveHealthFacts {
        self.health.borrow().clone()
    }

    /// Return the archive-independent native telemetry parity snapshot.
    pub fn parity(&self) -> NativeTelemetryParityFactsV1 {
        self.parity.snapshot()
    }
}

impl<NativeRecord, ArchiveRecord> Debug for AttachedAttemptFanout<NativeRecord, ArchiveRecord> {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AttachedAttemptFanout")
            .field("native", &self.native)
            .field("archive_enabled", &self.archive.is_some())
            .field("observer", &self.observer)
            .field("archive_required", &self.archive_required)
            .field("health", &self.health.borrow())
            .finish()
    }
}

/// Nonblocking lifecycle marker ingress used by the phase observer tee.
pub trait AttachedLifecycleIngress: Debug {
    /// Try to enqueue one exact owner-stamped marker on reserved control capacity.
    fn try_observe_lifecycle(
        &self,
        observation: ArchiveLifecycleObservation,
    ) -> Result<(), AttachedTelemetryError>;
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum PhaseLifecycleEvent {
    Start,
    Complete,
}

/// Authoritative phase observer tee for membership and lifecycle markers.
pub struct AttachedPhaseObserver {
    run_id: String,
    delegate: Rc<dyn PhaseObserver>,
    runtime: Rc<AttachedTelemetryRuntime>,
    lifecycle: Rc<dyn AttachedLifecycleIngress>,
    contexts: RefCell<BTreeMap<(String, PhaseLifecycleEvent), PhaseTransitionContext>>,
    first_error: RefCell<Option<AttachedTelemetryError>>,
}

impl AttachedPhaseObserver {
    /// Compose a tee over the exact observer used by the phase runner.
    pub fn new(
        run_id: impl Into<String>,
        delegate: Rc<dyn PhaseObserver>,
        runtime: Rc<AttachedTelemetryRuntime>,
        lifecycle: Rc<dyn AttachedLifecycleIngress>,
    ) -> Result<Self, AttachedTelemetryError> {
        let run_id = run_id.into();
        validate_identifier("run_id", &run_id)?;
        Ok(Self {
            run_id,
            delegate,
            runtime,
            lifecycle,
            contexts: RefCell::new(BTreeMap::new()),
            first_error: RefCell::new(None),
        })
    }

    /// Install a complete immutable boundary context before its phase callback.
    pub fn install_transition_context(
        &self,
        context: PhaseTransitionContext,
    ) -> Result<(), AttachedTelemetryError> {
        context.validate(&self.runtime.archived_source_ids)?;
        let event = match context.role {
            BoundaryRole::PhaseStart => PhaseLifecycleEvent::Start,
            BoundaryRole::PhaseEnd => PhaseLifecycleEvent::Complete,
        };
        let key = (context.phase_id.clone(), event);
        if self
            .contexts
            .borrow_mut()
            .insert(key.clone(), context)
            .is_some()
        {
            return Err(AttachedTelemetryError::TransitionContextAlreadyInstalled {
                phase_id: key.0,
                event: match key.1 {
                    PhaseLifecycleEvent::Start => "start",
                    PhaseLifecycleEvent::Complete => "complete",
                },
            });
        }
        Ok(())
    }

    /// Return the first lifecycle/membership error retained by infallible callbacks.
    pub fn take_error(&self) -> Option<AttachedTelemetryError> {
        self.first_error.borrow_mut().take()
    }

    fn record_error(&self, error: AttachedTelemetryError) {
        let mut first = self.first_error.borrow_mut();
        if first.is_none() {
            *first = Some(error);
        }
    }

    fn emit_phase_marker(
        &self,
        kind: aiperf_telemetry_archive::LifecycleMarkerKindV1,
        phase_state: aiperf_telemetry_archive::LifecyclePhaseStateV1,
        stats: &PhaseStats,
        event: Option<PhaseLifecycleEvent>,
        branch_stats: Option<&PhaseBranchStats>,
    ) {
        let observed_ns = match phase_state {
            aiperf_telemetry_archive::LifecyclePhaseStateV1::Started => stats.start_ns,
            aiperf_telemetry_archive::LifecyclePhaseStateV1::SendingComplete => stats.sent_end_ns,
            aiperf_telemetry_archive::LifecyclePhaseStateV1::Complete => stats.requests_end_ns,
        };
        let Some(observed_ns) = observed_ns else {
            self.record_error(AttachedTelemetryError::MissingPhaseTimestamp {
                phase_id: stats.phase_id.clone(),
                state: phase_state.as_str(),
            });
            return;
        };
        let context = event.and_then(|event| {
            self.contexts
                .borrow_mut()
                .remove(&(stats.phase_id.clone(), event))
        });
        let boundaries = context
            .map(|context| context.boundaries)
            .unwrap_or_default();
        if boundaries.is_empty() {
            self.emit_one_marker(kind, phase_state, stats, observed_ns, None, branch_stats);
        } else {
            for boundary in boundaries {
                self.emit_one_marker(
                    kind,
                    phase_state,
                    stats,
                    observed_ns,
                    Some(boundary),
                    branch_stats,
                );
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn emit_one_marker(
        &self,
        kind: aiperf_telemetry_archive::LifecycleMarkerKindV1,
        phase_state: aiperf_telemetry_archive::LifecyclePhaseStateV1,
        stats: &PhaseStats,
        observed_ns: i64,
        boundary: Option<BoundaryReference>,
        branch_stats: Option<&PhaseBranchStats>,
    ) {
        let mut attributes = BTreeMap::new();
        if let Some(branch) = branch_stats {
            attributes.insert(
                "aiperf.phase.branch.pending".to_owned(),
                branch.pending_work.to_string(),
            );
            attributes.insert(
                "aiperf.phase.branch.started".to_owned(),
                branch.started.to_string(),
            );
            attributes.insert(
                "aiperf.phase.branch.completed".to_owned(),
                branch.completed.to_string(),
            );
            attributes.insert(
                "aiperf.phase.branch.suppressed".to_owned(),
                branch.suppressed.to_string(),
            );
        }
        let source_id = boundary.as_ref().map(|value| value.source_id.clone());
        let observation = ArchiveLifecycleObservation {
            kind,
            observed_ns,
            run_id: Some(self.run_id.clone()),
            phase_id: Some(stats.phase_id.clone()),
            source_id,
            phase_state: Some(phase_state),
            completion_reason: (phase_state
                == aiperf_telemetry_archive::LifecyclePhaseStateV1::Complete)
                .then(|| lifecycle_completion_reason(stats)),
            boundary,
            phase_start_ns: stats.start_ns,
            sent_end_ns: stats.sent_end_ns,
            requests_end_ns: stats.requests_end_ns,
            attribute_epoch_id: None,
            attributes,
        };
        if let Err(error) = self.lifecycle.try_observe_lifecycle(observation) {
            self.record_error(error);
        }
    }
}

impl Debug for AttachedPhaseObserver {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AttachedPhaseObserver")
            .field("run_id", &self.run_id)
            .field("runtime", &self.runtime)
            .field(
                "pending_contexts",
                &self.contexts.borrow().keys().collect::<Vec<_>>(),
            )
            .field("has_error", &self.first_error.borrow().is_some())
            .finish_non_exhaustive()
    }
}

impl PhaseObserver for AttachedPhaseObserver {
    fn on_phase_start(&self, config: &PhaseConfig, stats: PhaseStats) {
        if let Err(error) = self.runtime.activate_phase(&stats.phase_id) {
            self.record_error(error);
        }
        self.delegate.on_phase_start(config, stats.clone());
        self.emit_phase_marker(
            aiperf_telemetry_archive::LifecycleMarkerKindV1::PhaseStarted,
            aiperf_telemetry_archive::LifecyclePhaseStateV1::Started,
            &stats,
            Some(PhaseLifecycleEvent::Start),
            None,
        );
    }

    fn on_progress(&self, stats: PhaseStats) {
        self.delegate.on_progress(stats);
    }

    fn on_sending_complete(&self, stats: PhaseStats) {
        self.delegate.on_sending_complete(stats.clone());
        self.emit_phase_marker(
            aiperf_telemetry_archive::LifecycleMarkerKindV1::PhaseSendingComplete,
            aiperf_telemetry_archive::LifecyclePhaseStateV1::SendingComplete,
            &stats,
            None,
            None,
        );
    }

    fn on_phase_complete(&self, stats: PhaseStats, branch_stats: Option<PhaseBranchStats>) {
        self.delegate
            .on_phase_complete(stats.clone(), branch_stats.clone());
        self.emit_phase_marker(
            aiperf_telemetry_archive::LifecycleMarkerKindV1::PhaseComplete,
            aiperf_telemetry_archive::LifecyclePhaseStateV1::Complete,
            &stats,
            Some(PhaseLifecycleEvent::Complete),
            branch_stats.as_ref(),
        );
        if let Err(error) = self.runtime.deactivate_phase(&stats.phase_id) {
            self.record_error(error);
        }
    }

    fn on_phases_complete(&self, stats: Vec<PhaseStats>) {
        self.delegate.on_phases_complete(stats);
        if let Some(((phase_id, event), _)) = self.contexts.borrow().first_key_value() {
            self.record_error(AttachedTelemetryError::UnusedTransitionContext {
                phase_id: phase_id.clone(),
                event: match event {
                    PhaseLifecycleEvent::Start => "start",
                    PhaseLifecycleEvent::Complete => "complete",
                },
            });
        }
    }
}

fn boundary_reference(
    transition_id: &str,
    source_id: &str,
    phase_id: &str,
    role: BoundaryRole,
    group: Option<&str>,
) -> BoundaryReference {
    let role_name = match role {
        BoundaryRole::PhaseStart => "start",
        BoundaryRole::PhaseEnd => "end",
    };
    BoundaryReference {
        transition_id: transition_id.to_owned(),
        boundary_id: stable_boundary_id(
            "aiperf.telemetry.boundary-id.v1",
            transition_id,
            source_id,
            phase_id,
            role_name,
        ),
        phase_id: phase_id.to_owned(),
        source_id: source_id.to_owned(),
        role,
        coalescing_group_id: group.map(str::to_owned),
    }
}

fn stable_boundary_id(
    domain: &str,
    transition_id: &str,
    source_id: &str,
    phase_id: &str,
    role: &str,
) -> String {
    domain_digest(
        domain,
        &[
            transition_id.as_bytes(),
            source_id.as_bytes(),
            phase_id.as_bytes(),
            role.as_bytes(),
        ],
    )
    .to_hex()
}

fn validate_boundary_result(
    transition_id: &str,
    command: &SourceBoundarySnapshotCommand,
    result: &SourceBoundaryResult,
) -> Result<(), AttachedTelemetryError> {
    if result.source_id != command.source_id || result.transition_id != transition_id {
        return Err(AttachedTelemetryError::BoundaryResultMismatch(format!(
            "driver returned ({:?}, {:?}) for ({:?}, {:?})",
            result.transition_id, result.source_id, transition_id, command.source_id
        )));
    }
    if result.terminal.boundary_refs() != command.subscribers {
        return Err(AttachedTelemetryError::BoundaryResultMismatch(format!(
            "driver changed structured references for source {:?}",
            command.source_id
        )));
    }
    if let SourceBoundaryTerminal::Loss {
        loss_kind, reason, ..
    } = &result.terminal
        && loss_kind.reason() != *reason
    {
        return Err(AttachedTelemetryError::BoundaryResultMismatch(format!(
            "driver returned incompatible loss kind/reason for source {:?}",
            command.source_id
        )));
    }
    Ok(())
}

fn lifecycle_completion_reason(
    stats: &PhaseStats,
) -> aiperf_telemetry_archive::LifecycleCompletionReasonV1 {
    use aiperf_telemetry_archive::LifecycleCompletionReasonV1 as ArchiveReason;
    match stats.completion_reason {
        Some(PhaseCompletionReason::Cancelled | PhaseCompletionReason::ForceCompleted) => {
            ArchiveReason::Cancelled
        }
        Some(PhaseCompletionReason::Failed) => ArchiveReason::Failed,
        Some(PhaseCompletionReason::GraceTimeout) => ArchiveReason::Duration,
        Some(PhaseCompletionReason::Completed) | None if stats.timeout_triggered => {
            ArchiveReason::Duration
        }
        Some(PhaseCompletionReason::Completed) | None
            if stats.total_expected_requests.is_some()
                && stats.final_requests_sent == stats.total_expected_requests =>
        {
            ArchiveReason::RequestCount
        }
        Some(PhaseCompletionReason::Completed) | None
            if stats.expected_num_sessions.is_some()
                && stats.final_sent_sessions == stats.expected_num_sessions =>
        {
            ArchiveReason::SessionCount
        }
        Some(PhaseCompletionReason::Completed) | None => ArchiveReason::Completed,
    }
}

fn optional_i64(value: Option<i64>) -> CanonicalJsonValue {
    value.map_or(CanonicalJsonValue::Null, |value| {
        CanonicalJsonValue::Integer(i128::from(value))
    })
}

fn optional_u64(value: Option<u64>) -> CanonicalJsonValue {
    value.map_or(CanonicalJsonValue::Null, |value| {
        CanonicalJsonValue::Integer(i128::from(value))
    })
}

fn checked_increment(value: &mut u64) -> Result<(), AttachedTelemetryError> {
    *value = value
        .checked_add(1)
        .ok_or(AttachedTelemetryError::SequenceOverflow)?;
    Ok(())
}

fn validate_identifier(field: &'static str, value: &str) -> Result<(), AttachedTelemetryError> {
    if value.is_empty() || value.trim() != value || value.chars().any(char::is_control) {
        return Err(AttachedTelemetryError::InvalidIdentifier {
            field,
            value: value.to_owned(),
        });
    }
    Ok(())
}

/// Invalid attachment topology, ordering, or terminal observation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum AttachedTelemetryError {
    /// No physical sidecar source was prepared.
    NoPreparedSources,
    /// No source was selected by the archive attachment.
    NoAttachedSources,
    /// A stable ID was empty, padded, or contained a control character.
    InvalidIdentifier {
        /// Field carrying the invalid identifier.
        field: &'static str,
        /// Rejected redaction-safe value.
        value: String,
    },
    /// Two prepared drivers exposed the same physical ID.
    DuplicatePreparedSource(String),
    /// The attachment repeated one source ID.
    DuplicateAttachedSource(String),
    /// The attachment selected a source outside the prepared inventory.
    UnknownAttachedSource {
        /// Rejected source ID.
        source_id: String,
        /// Deterministic prepared choices.
        available: Vec<String>,
    },
    /// A boundary plan referred to an unselected source.
    UnknownBoundarySource(String),
    /// A boundary deadline was not a positive absolute Clock value.
    InvalidBoundaryDeadline(i64),
    /// A transition contained neither an ending nor starting phase.
    EmptyTransition,
    /// One phase cannot be adjacent to itself.
    SameAdjacentPhase(String),
    /// The archive boundary registry rejected an incomplete or reused plan.
    BoundaryPlan(BoundaryPlanError),
    /// A driver returned identity or references different from its command.
    BoundaryResultMismatch(String),
    /// One phase was activated twice.
    PhaseAlreadyActive(String),
    /// One phase was deactivated before activation or more than once.
    PhaseNotActive(String),
    /// One transition context did not copy complete reference identity.
    InvalidTransitionContext(String),
    /// One context repeated the same physical source.
    DuplicateTransitionSource {
        /// Phase receiving the context.
        phase_id: String,
        /// Repeated source.
        source_id: String,
    },
    /// One context was not cardinal over every archive-selected source.
    TransitionSourceCardinality {
        /// Phase receiving the context.
        phase_id: String,
        /// Expected deterministic source set.
        expected: Vec<String>,
        /// Actual deterministic source set.
        actual: Vec<String>,
    },
    /// A second context attempted to mutate an installed transition.
    TransitionContextAlreadyInstalled {
        /// Phase receiving the context.
        phase_id: String,
        /// Lifecycle edge.
        event: &'static str,
    },
    /// An installed context never reached its authoritative observer callback.
    UnusedTransitionContext {
        /// Phase receiving the context.
        phase_id: String,
        /// Lifecycle edge.
        event: &'static str,
    },
    /// Attempt and boundary source identities disagreed.
    AttemptBoundarySourceMismatch {
        /// Physical attempt source.
        attempt_source: String,
        /// Structured boundary source.
        boundary_source: String,
    },
    /// An attempt repeated one exact boundary identity.
    DuplicateAttemptBoundary(String),
    /// Native parity saw the same physical attempt twice.
    DuplicateNativeAttempt {
        /// Stable source ID.
        source_id: String,
        /// Repeated per-source sequence.
        source_record_seq: u64,
    },
    /// A required attached archive rejected an issued native event.
    RequiredArchiveRejected(AdmissionRejection),
    /// An authoritative phase state omitted its Clock timestamp.
    MissingPhaseTimestamp {
        /// Phase identity.
        phase_id: String,
        /// Expected lifecycle state.
        state: &'static str,
    },
    /// A monotone sequence or health counter overflowed.
    SequenceOverflow,
    /// Internal invariant failure that cannot be classified as authored input.
    Invariant(String),
    /// Driver, native delivery, archive loss, or lifecycle implementation failed.
    Component(String),
}

impl Display for AttachedTelemetryError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoPreparedSources => formatter
                .write_str("attached telemetry requires at least one prepared physical source"),
            Self::NoAttachedSources => {
                formatter.write_str("attached telemetry archive source_ids cannot be empty")
            }
            Self::InvalidIdentifier { field, value } => {
                write!(formatter, "{field} has invalid identifier {value:?}")
            }
            Self::DuplicatePreparedSource(source) => write!(
                formatter,
                "duplicate prepared physical telemetry source {source:?}"
            ),
            Self::DuplicateAttachedSource(source) => {
                write!(formatter, "duplicate attached telemetry source {source:?}")
            }
            Self::UnknownAttachedSource {
                source_id,
                available,
            } => write!(
                formatter,
                "attached telemetry source {source_id:?} is unavailable; prepared sources: {}",
                available.join(", ")
            ),
            Self::UnknownBoundarySource(source) => write!(
                formatter,
                "boundary plan references non-attached source {source:?}"
            ),
            Self::InvalidBoundaryDeadline(deadline) => write!(
                formatter,
                "boundary absolute deadline must be positive, got {deadline}"
            ),
            Self::EmptyTransition => {
                formatter.write_str("boundary transition requires an ending or starting phase")
            }
            Self::SameAdjacentPhase(phase) => write!(
                formatter,
                "boundary transition cannot end and start the same phase {phase:?}"
            ),
            Self::BoundaryPlan(error) => {
                write!(formatter, "invalid attached boundary plan: {error}")
            }
            Self::BoundaryResultMismatch(message) => {
                write!(formatter, "invalid attached boundary result: {message}")
            }
            Self::PhaseAlreadyActive(phase) => {
                write!(formatter, "telemetry phase {phase:?} is already active")
            }
            Self::PhaseNotActive(phase) => {
                write!(formatter, "telemetry phase {phase:?} is not active")
            }
            Self::InvalidTransitionContext(phase) => write!(
                formatter,
                "invalid boundary transition context for phase {phase:?}"
            ),
            Self::DuplicateTransitionSource {
                phase_id,
                source_id,
            } => write!(
                formatter,
                "phase {phase_id:?} transition repeats source {source_id:?}"
            ),
            Self::TransitionSourceCardinality {
                phase_id,
                expected,
                actual,
            } => write!(
                formatter,
                "phase {phase_id:?} transition sources {actual:?} do not match attached sources {expected:?}"
            ),
            Self::TransitionContextAlreadyInstalled { phase_id, event } => write!(
                formatter,
                "phase {phase_id:?} {event} transition context is already installed"
            ),
            Self::UnusedTransitionContext { phase_id, event } => write!(
                formatter,
                "phase {phase_id:?} {event} transition context was never observed"
            ),
            Self::AttemptBoundarySourceMismatch {
                attempt_source,
                boundary_source,
            } => write!(
                formatter,
                "attempt source {attempt_source:?} does not match boundary source {boundary_source:?}"
            ),
            Self::DuplicateAttemptBoundary(boundary) => {
                write!(formatter, "attempt repeats boundary identity {boundary:?}")
            }
            Self::DuplicateNativeAttempt {
                source_id,
                source_record_seq,
            } => write!(
                formatter,
                "native telemetry attempt ({source_id:?}, {source_record_seq}) was delivered twice"
            ),
            Self::RequiredArchiveRejected(rejection) => write!(
                formatter,
                "required attached archive rejected native attempt: {rejection}"
            ),
            Self::MissingPhaseTimestamp { phase_id, state } => write!(
                formatter,
                "phase {phase_id:?} {state} observation omitted its authoritative timestamp"
            ),
            Self::SequenceOverflow => formatter.write_str("attached telemetry sequence overflow"),
            Self::Invariant(message) => {
                write!(formatter, "attached telemetry invariant failed: {message}")
            }
            Self::Component(message) => formatter.write_str(message),
        }
    }
}

impl std::error::Error for AttachedTelemetryError {}

#[cfg(test)]
mod tests {
    use aiperf_timing::{GracePeriod, NoopPhaseObserver, PhaseState};

    use super::*;

    #[derive(Debug)]
    struct FakeDriver {
        source_id: String,
        active: RefCell<BTreeSet<String>>,
        submissions: Rc<RefCell<Vec<String>>>,
    }

    impl FakeDriver {
        fn new(source_id: &str, submissions: Rc<RefCell<Vec<String>>>) -> Self {
            Self {
                source_id: source_id.to_owned(),
                active: RefCell::new(BTreeSet::new()),
                submissions,
            }
        }
    }

    impl RunOwnedTelemetryDriver for FakeDriver {
        fn source_id(&self) -> &str {
            &self.source_id
        }

        fn set_phase_active(
            &self,
            phase_id: &str,
            active: bool,
        ) -> Result<(), AttachedTelemetryError> {
            if active {
                self.active.borrow_mut().insert(phase_id.to_owned());
            } else {
                self.active.borrow_mut().remove(phase_id);
            }
            Ok(())
        }

        fn submit_boundary(
            &self,
            command: SourceBoundarySnapshotCommand,
        ) -> LocalBoundaryFuture<Result<SourceBoundaryResult, AttachedTelemetryError>> {
            self.submissions
                .borrow_mut()
                .push(command.source_id.clone());
            Box::pin(async move {
                Ok(SourceBoundaryResult {
                    source_id: command.source_id,
                    transition_id: command.subscribers[0].transition_id.clone(),
                    terminal: SourceBoundaryTerminal::Attempt {
                        source_record_seq: 1,
                        request_attempt_seq: Some(1),
                        boundary_refs: command.subscribers,
                    },
                })
            })
        }
    }

    fn runtime() -> Rc<AttachedTelemetryRuntime> {
        let submissions = Rc::new(RefCell::new(Vec::new()));
        let drivers: Vec<Rc<dyn RunOwnedTelemetryDriver>> = vec![
            Rc::new(FakeDriver::new("node-b", submissions.clone())),
            Rc::new(FakeDriver::new("node-a", submissions)),
        ];
        Rc::new(
            AttachedTelemetryRuntime::new(drivers, vec!["node-b".to_owned(), "node-a".to_owned()])
                .unwrap(),
        )
    }

    #[tokio::test(flavor = "current_thread")]
    async fn source_cardinal_adjacent_transition_routes_one_command_per_driver() {
        let runtime = runtime();
        let plan = runtime
            .build_transition_plan("transition-1", Some("warmup"), Some("profiling"), 100)
            .unwrap();
        assert_eq!(plan.commands.len(), 2);
        assert!(
            plan.commands
                .iter()
                .all(|command| command.subscribers.len() == 2)
        );
        assert_ne!(
            plan.commands[0].coalescing_group_id,
            plan.commands[1].coalescing_group_id
        );

        let contexts = runtime.phase_transition_contexts(&plan).unwrap();
        assert_eq!(contexts.len(), 2);
        assert!(contexts.iter().all(|context| context.boundaries.len() == 2));
        let results = runtime.capture_boundary_plan(plan.clone()).await.unwrap();
        assert_eq!(
            results
                .iter()
                .map(|value| value.source_id.as_str())
                .collect::<Vec<_>>(),
            vec!["node-a", "node-b"]
        );
        assert!(matches!(
            runtime.capture_boundary_plan(plan).await,
            Err(AttachedTelemetryError::BoundaryPlan(
                BoundaryPlanError::TransitionAlreadySealed(_)
            ))
        ));
    }

    #[test]
    fn attachment_resolves_only_existing_unique_physical_sources() {
        let submissions = Rc::new(RefCell::new(Vec::new()));
        let drivers: Vec<Rc<dyn RunOwnedTelemetryDriver>> =
            vec![Rc::new(FakeDriver::new("node-a", submissions))];
        assert!(matches!(
            AttachedTelemetryRuntime::new(drivers.clone(), Vec::new()),
            Err(AttachedTelemetryError::NoAttachedSources)
        ));
        assert!(matches!(
            AttachedTelemetryRuntime::new(drivers.clone(), vec!["node-b".to_owned()]),
            Err(AttachedTelemetryError::UnknownAttachedSource { .. })
        ));
        assert!(matches!(
            AttachedTelemetryRuntime::new(drivers, vec!["node-a".to_owned(), "node-a".to_owned()]),
            Err(AttachedTelemetryError::DuplicateAttachedSource(_))
        ));
    }

    #[derive(Debug)]
    struct RecordingNative {
        events: Rc<RefCell<Vec<&'static str>>>,
    }

    impl NativeAttemptDelivery<String> for RecordingNative {
        fn deliver(
            &self,
            context: &PhysicalAttemptContext,
            record: Option<&String>,
        ) -> Result<NativeDeliveryReceipt, AttachedTelemetryError> {
            self.events.borrow_mut().push("native");
            let record = record.expect("test native record");
            Ok(NativeDeliveryReceipt {
                delivered_record: true,
                native_event_digest: domain_digest(
                    "test.native-event.v1",
                    &[
                        context.source_id.as_bytes(),
                        &context.source_record_seq.to_le_bytes(),
                        record.as_bytes(),
                    ],
                ),
            })
        }
    }

    #[derive(Debug)]
    struct RejectingArchive {
        events: Rc<RefCell<Vec<&'static str>>>,
        losses: RefCell<Vec<VisibleArchiveLoss>>,
    }

    impl AttachedArchiveIngress<Vec<u8>> for RejectingArchive {
        fn try_admit(
            &self,
            _projection: AttachedArchiveProjection<Vec<u8>>,
        ) -> Result<(), AdmissionRejection> {
            self.events.borrow_mut().push("archive");
            Err(AdmissionRejection::Capacity)
        }

        fn record_visible_loss(
            &self,
            loss: VisibleArchiveLoss,
        ) -> Result<(), AttachedTelemetryError> {
            self.events.borrow_mut().push("loss");
            self.losses.borrow_mut().push(loss);
            Ok(())
        }
    }

    #[derive(Debug)]
    struct RecordingObserver {
        events: Rc<RefCell<Vec<&'static str>>>,
        observations: RefCell<Vec<PostNativeAttemptObservation>>,
    }

    impl AttachedAttemptObserver for RecordingObserver {
        fn observe(&self, observation: &PostNativeAttemptObservation) {
            self.events.borrow_mut().push("observe");
            self.observations.borrow_mut().push(observation.clone());
        }
    }

    fn physical_attempt() -> AttachedPhysicalAttempt<String, Vec<u8>> {
        AttachedPhysicalAttempt {
            context: PhysicalAttemptContext {
                source_id: "node-a".to_owned(),
                source_record_seq: 7,
                request_attempt_seq: Some(6),
                capture_ns: Some(123),
                active_phase_ids: BTreeSet::from(["profiling".to_owned()]),
                boundary_refs: Vec::new(),
            },
            native_record: Some("native-record".to_owned()),
            archive_record: vec![1, 2, 3],
        }
    }

    #[test]
    fn native_delivery_precedes_nonblocking_admission_and_visible_loss() {
        let events = Rc::new(RefCell::new(Vec::new()));
        let native = Rc::new(RecordingNative {
            events: events.clone(),
        });
        let archive = Rc::new(RejectingArchive {
            events: events.clone(),
            losses: RefCell::new(Vec::new()),
        });
        let observer = Rc::new(RecordingObserver {
            events: events.clone(),
            observations: RefCell::new(Vec::new()),
        });
        let parity = Rc::new(NativeTelemetryParityRecorder::default());
        let fanout = AttachedAttemptFanout::new(
            native,
            Some(archive.clone()),
            observer.clone(),
            parity,
            false,
        );

        assert_eq!(
            fanout.process(physical_attempt()).unwrap(),
            AttachedArchiveAdmission::Rejected(AdmissionRejection::Capacity)
        );
        assert_eq!(&*events.borrow(), &["native", "archive", "loss", "observe"]);
        assert_eq!(archive.losses.borrow().len(), 1);
        assert_eq!(fanout.health().native_attempts, 1);
        assert_eq!(fanout.health().archive_rejected, 1);
        assert!(!fanout.health().complete);
        assert_eq!(
            observer.observations.borrow()[0].context.source_record_seq,
            7
        );
    }

    #[test]
    fn parity_bytes_exclude_archive_admission() {
        let off_events = Rc::new(RefCell::new(Vec::new()));
        let off = AttachedAttemptFanout::new(
            Rc::new(RecordingNative { events: off_events }),
            None,
            Rc::new(NoopAttachedAttemptObserver),
            Rc::new(NativeTelemetryParityRecorder::default()),
            false,
        );
        off.process(physical_attempt()).unwrap();

        let on_events = Rc::new(RefCell::new(Vec::new()));
        let on = AttachedAttemptFanout::new(
            Rc::new(RecordingNative {
                events: on_events.clone(),
            }),
            Some(Rc::new(RejectingArchive {
                events: on_events,
                losses: RefCell::new(Vec::new()),
            })),
            Rc::new(NoopAttachedAttemptObserver),
            Rc::new(NativeTelemetryParityRecorder::default()),
            false,
        );
        on.process(physical_attempt()).unwrap();

        assert_eq!(
            off.parity().canonical_bytes(),
            on.parity().canonical_bytes()
        );
        assert_ne!(off.health(), on.health());
    }

    #[derive(Debug, Default)]
    struct RecordingLifecycle {
        observations: RefCell<Vec<ArchiveLifecycleObservation>>,
    }

    impl AttachedLifecycleIngress for RecordingLifecycle {
        fn try_observe_lifecycle(
            &self,
            observation: ArchiveLifecycleObservation,
        ) -> Result<(), AttachedTelemetryError> {
            self.observations.borrow_mut().push(observation);
            Ok(())
        }
    }

    fn stats(phase_id: &str, state: PhaseState) -> PhaseStats {
        PhaseStats {
            phase_id: phase_id.to_owned(),
            kind: aiperf_timing::PhaseKind::Profiling,
            state,
            start_ns: Some(10),
            sent_end_ns: (state != PhaseState::Started).then_some(20),
            requests_end_ns: (state == PhaseState::Complete).then_some(30),
            total_expected_requests: Some(1),
            expected_num_sessions: None,
            expected_duration_ns: None,
            grace_period: GracePeriod::Disabled,
            requests_sent: 1,
            requests_completed: u64::from(state == PhaseState::Complete),
            requests_cancelled: 0,
            request_errors: 0,
            sent_sessions: 1,
            completed_sessions: u64::from(state == PhaseState::Complete),
            cancelled_sessions: 0,
            total_session_turns: 1,
            in_flight_requests: u64::from(state != PhaseState::Complete),
            in_flight_sessions: u64::from(state != PhaseState::Complete),
            in_flight_prefills: 0,
            pending_branch_work: 0,
            stuck_session_slots_released: 0,
            stuck_prefill_slots_released: 0,
            final_requests_sent: (state != PhaseState::Started).then_some(1),
            final_requests_completed: (state == PhaseState::Complete).then_some(1),
            final_requests_cancelled: (state == PhaseState::Complete).then_some(0),
            final_request_errors: (state == PhaseState::Complete).then_some(0),
            final_sent_sessions: (state != PhaseState::Started).then_some(1),
            final_completed_sessions: (state == PhaseState::Complete).then_some(1),
            final_cancelled_sessions: (state == PhaseState::Complete).then_some(0),
            timeout_triggered: false,
            grace_period_timeout_triggered: false,
            cancel_drain_timeout_triggered: false,
            forced_completion: false,
            was_cancelled: false,
            completion_reason: (state == PhaseState::Complete)
                .then_some(PhaseCompletionReason::Completed),
        }
    }

    #[test]
    fn phase_observer_emits_source_scoped_markers_and_tracks_membership() {
        let runtime = runtime();
        let plan = runtime
            .build_transition_plan("transition-start", None, Some("profiling"), 100)
            .unwrap();
        let context = runtime
            .phase_transition_contexts(&plan)
            .unwrap()
            .pop()
            .unwrap();
        let lifecycle = Rc::new(RecordingLifecycle::default());
        let observer = AttachedPhaseObserver::new(
            "run-a",
            Rc::new(NoopPhaseObserver),
            runtime.clone(),
            lifecycle.clone(),
        )
        .unwrap();
        observer.install_transition_context(context).unwrap();
        let config = PhaseConfig::new(
            "profiling",
            aiperf_timing::PhaseKind::Profiling,
            aiperf_timing::StopConfig::default(),
        );

        observer.on_phase_start(&config, stats("profiling", PhaseState::Started));
        assert_eq!(runtime.active_phases(), vec!["profiling"]);
        assert_eq!(lifecycle.observations.borrow().len(), 2);
        assert!(
            lifecycle
                .observations
                .borrow()
                .iter()
                .all(|value| value.boundary.is_some())
        );

        observer.on_sending_complete(stats("profiling", PhaseState::SendingComplete));
        observer.on_phase_complete(stats("profiling", PhaseState::Complete), None);
        assert!(runtime.active_phases().is_empty());
        assert_eq!(lifecycle.observations.borrow().len(), 4);
        assert!(observer.take_error().is_none());
    }

    #[test]
    fn required_archive_reports_failure_only_after_native_and_loss_observation() {
        let events = Rc::new(RefCell::new(Vec::new()));
        let observer = Rc::new(RecordingObserver {
            events: events.clone(),
            observations: RefCell::new(Vec::new()),
        });
        let fanout = AttachedAttemptFanout::new(
            Rc::new(RecordingNative {
                events: events.clone(),
            }),
            Some(Rc::new(RejectingArchive {
                events: events.clone(),
                losses: RefCell::new(Vec::new()),
            })),
            observer,
            Rc::new(NativeTelemetryParityRecorder::default()),
            true,
        );

        assert_eq!(
            fanout.process(physical_attempt()),
            Err(AttachedTelemetryError::RequiredArchiveRejected(
                AdmissionRejection::Capacity
            ))
        );
        assert_eq!(&*events.borrow(), &["native", "archive", "loss", "observe"]);
    }

    #[test]
    fn duplicate_native_attempt_identity_fails_closed() {
        let events = Rc::new(RefCell::new(Vec::new()));
        let fanout = AttachedAttemptFanout::new(
            Rc::new(RecordingNative { events }),
            None,
            Rc::new(NoopAttachedAttemptObserver),
            Rc::new(NativeTelemetryParityRecorder::default()),
            false,
        );
        fanout.process(physical_attempt()).unwrap();
        assert!(matches!(
            fanout.process(physical_attempt()),
            Err(AttachedTelemetryError::DuplicateNativeAttempt { .. })
        ));
    }

    #[test]
    fn lifecycle_completion_reason_prefers_exact_stop_authority() {
        let mut completed = stats("profiling", PhaseState::Complete);
        assert_eq!(
            lifecycle_completion_reason(&completed),
            aiperf_telemetry_archive::LifecycleCompletionReasonV1::RequestCount
        );
        completed.timeout_triggered = true;
        assert_eq!(
            lifecycle_completion_reason(&completed),
            aiperf_telemetry_archive::LifecycleCompletionReasonV1::Duration
        );
        completed.completion_reason = Some(PhaseCompletionReason::Failed);
        assert_eq!(
            lifecycle_completion_reason(&completed),
            aiperf_telemetry_archive::LifecycleCompletionReasonV1::Failed
        );
    }

    #[test]
    fn parity_digest_is_stable_and_archive_free() {
        let recorder = NativeTelemetryParityRecorder::default();
        let attempt = physical_attempt();
        recorder
            .record(
                &attempt.context,
                NativeDeliveryReceipt {
                    delivered_record: true,
                    native_event_digest: Digest::from_bytes([7; 32]),
                },
            )
            .unwrap();
        let first = recorder.snapshot();
        let second = recorder.snapshot();
        assert_eq!(first, second);
        assert!(
            std::str::from_utf8(&first.canonical_bytes())
                .unwrap()
                .contains("native_telemetry_parity_v1")
        );
    }
}
