// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Fixed-memory exact-loss and saturation accounting.
//!
//! Attached telemetry cannot make the benchmark request path wait for archive
//! capacity. This ledger therefore allocates every retained exact-range slot,
//! every boundary-reference slot, and every legal v1 saturation tuple during
//! preparation. Recording a loss only mutates those allocations. Report and
//! persistence views are materialized on demand and remain bounded by the
//! validated preparation limits.
//!
//! The tuple universe is closed by loss schema v1. Every physical source owns
//! one slot for each of the five source-valid kind/reason pairs; the global
//! sentinel owns only `writer_failed` and `shutdown_abandoned`. Once the exact
//! range limit is reached, every non-coalescible input advances its tuple's
//! cumulative, order-sensitive saturation accumulator.

use std::fmt::{self, Display, Formatter};

use crate::{
    ArchiveId, BoundaryReference, Digest, ExactLossRangeV1, LossKindV1, LossReasonV1,
    LossSaturationSnapshotV1, LossValidationError, SessionId, loss_saturation_slot_id_v1,
};

const LOSS_OVERFLOW_DOMAIN_V1: &str = "aiperf.archive.loss-overflow.v1";
const SOURCE_VALID_KINDS_V1: [LossKindV1; 5] = [
    LossKindV1::MissedCadence,
    LossKindV1::ArchiveRejected,
    LossKindV1::ProjectionFailed,
    LossKindV1::WriterFailed,
    LossKindV1::ShutdownAbandoned,
];
const GLOBAL_VALID_KINDS_V1: [LossKindV1; 2] =
    [LossKindV1::WriterFailed, LossKindV1::ShutdownAbandoned];

/// Preparation-time bounds for one attached loss ledger.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LossLedgerLimitsV1 {
    /// Maximum individually enumerable coalesced ranges retained for the run.
    pub max_exact_ranges: usize,
    /// Maximum number of physical source identities in the frozen universe.
    pub max_sources: usize,
    /// Maximum UTF-8 byte length of one physical source identity.
    pub max_source_id_bytes: usize,
    /// Maximum retained boundary references in any one exact range.
    pub max_boundary_refs_per_range: usize,
    /// Maximum UTF-8 byte length of each boundary-reference identifier.
    pub max_boundary_identifier_bytes: usize,
}

impl LossLedgerLimitsV1 {
    fn validate(self) -> Result<(), LossLedgerError> {
        if self.max_exact_ranges == 0 {
            return Err(LossLedgerError::ZeroLimit("max_exact_ranges"));
        }
        if self.max_source_id_bytes == 0 {
            return Err(LossLedgerError::ZeroLimit("max_source_id_bytes"));
        }
        if self.max_boundary_identifier_bytes == 0 {
            return Err(LossLedgerError::ZeroLimit("max_boundary_identifier_bytes"));
        }
        Ok(())
    }
}

/// Stable internal-allocation evidence used by fixed-memory tests and health diagnostics.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LossLedgerAllocationShapeV1 {
    /// Number of prepared physical sources.
    pub source_count: usize,
    /// Reserved source-vector capacity.
    pub source_capacity: usize,
    /// Number of preallocated exact-range slots.
    pub exact_slot_count: usize,
    /// Reserved exact-slot vector capacity.
    pub exact_slot_capacity: usize,
    /// Total boundary-reference capacity across every exact slot.
    pub boundary_reference_capacity: usize,
    /// Number of preallocated legal saturation tuples.
    pub saturation_slot_count: usize,
    /// Reserved saturation-slot vector capacity.
    pub saturation_slot_capacity: usize,
}

/// Result of adding one validated loss to the fixed-memory ledger.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LossLedgerRecordOutcomeV1 {
    /// A new individually enumerable exact range occupied a prepared slot.
    RetainedExact {
        /// Zero-based exact slot index.
        exact_index: usize,
    },
    /// The input extended an existing exact range without consuming a slot.
    CoalescedExact {
        /// Zero-based exact slot index.
        exact_index: usize,
    },
    /// Exact-range capacity was full and a cumulative slot advanced.
    Saturated {
        /// Stable tuple identity.
        saturation_slot_id: Digest,
        /// New immutable slot-local snapshot sequence.
        saturation_snapshot_seq: u64,
        /// Cumulative omitted range count after the update.
        cumulative_omitted_range_count: u64,
        /// Cumulative omitted entry count after the update.
        cumulative_omitted_entry_count: u64,
    },
}

/// Bounded report/query materialization of the ledger's current state.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LossLedgerViewV1 {
    /// Individually enumerable exact ranges in first-observation order.
    pub exact_ranges: Vec<ExactLossRangeV1>,
    /// Latest cumulative snapshot for each active saturation slot.
    pub saturation_snapshots: Vec<LossSaturationSnapshotV1>,
    /// Whether every represented loss remains individually enumerable.
    pub complete_ranges: bool,
}

/// Run-owned, preparation-bounded attached telemetry loss ledger.
#[derive(Debug)]
pub struct FixedLossLedgerV1 {
    archive_id: ArchiveId,
    session_id: SessionId,
    limits: LossLedgerLimitsV1,
    sources: Vec<String>,
    exact_slots: Vec<ExactSlotV1>,
    exact_ranges: usize,
    exact_capacity_exhausted: bool,
    saturation_slots: Vec<SaturationSlotV1>,
    active_saturation_slots: usize,
    recording_started: bool,
}

impl FixedLossLedgerV1 {
    /// Preallocates the complete exact-range and legal saturation-slot universe.
    pub fn new<I, S>(
        archive_id: ArchiveId,
        session_id: SessionId,
        sources: I,
        limits: LossLedgerLimitsV1,
    ) -> Result<Self, LossLedgerError>
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        limits.validate()?;

        let mut prepared_sources = Vec::new();
        reserve_exact(
            &mut prepared_sources,
            limits.max_sources,
            "source identities",
        )?;
        for source in sources {
            if prepared_sources.len() == limits.max_sources {
                return Err(LossLedgerError::TooManySources {
                    maximum: limits.max_sources,
                    actual: limits
                        .max_sources
                        .checked_add(1)
                        .ok_or(LossLedgerError::ArithmeticOverflow)?,
                });
            }
            let source = source.into();
            validate_bounded_identifier("source_id", &source, limits.max_source_id_bytes)?;
            prepared_sources.push(source);
        }
        prepared_sources.sort_unstable_by(|left, right| left.as_bytes().cmp(right.as_bytes()));
        for pair in prepared_sources.windows(2) {
            if pair[0] == pair[1] {
                return Err(LossLedgerError::DuplicateSource(pair[0].clone()));
            }
        }

        let mut exact_slots = Vec::new();
        reserve_exact(
            &mut exact_slots,
            limits.max_exact_ranges,
            "exact loss slots",
        )?;
        for _ in 0..limits.max_exact_ranges {
            let mut boundary_refs = Vec::new();
            reserve_exact(
                &mut boundary_refs,
                limits.max_boundary_refs_per_range,
                "exact loss boundary references",
            )?;
            exact_slots.push(ExactSlotV1 {
                state: None,
                boundary_refs,
            });
        }

        let saturation_count = prepared_sources
            .len()
            .checked_mul(SOURCE_VALID_KINDS_V1.len())
            .and_then(|count| count.checked_add(GLOBAL_VALID_KINDS_V1.len()))
            .ok_or(LossLedgerError::ArithmeticOverflow)?;
        let mut saturation_slots = Vec::new();
        reserve_exact(
            &mut saturation_slots,
            saturation_count,
            "loss saturation slots",
        )?;

        for kind in GLOBAL_VALID_KINDS_V1 {
            saturation_slots.push(SaturationSlotV1::new(archive_id, session_id, None, kind));
        }
        for (source_index, source) in prepared_sources.iter().enumerate() {
            for kind in SOURCE_VALID_KINDS_V1 {
                saturation_slots.push(SaturationSlotV1::new(
                    archive_id,
                    session_id,
                    Some((source_index, source.as_str())),
                    kind,
                ));
            }
        }

        Ok(Self {
            archive_id,
            session_id,
            limits,
            sources: prepared_sources,
            exact_slots,
            exact_ranges: 0,
            exact_capacity_exhausted: false,
            saturation_slots,
            active_saturation_slots: 0,
            recording_started: false,
        })
    }

    /// Preallocates a ledger and restores durable snapshots in per-slot log order.
    ///
    /// The input may contain one latest row per slot or the complete snapshot
    /// history. When history is present, sequences, cumulative counts, range
    /// endpoints, and digests must move monotonically. The greatest valid row
    /// becomes the in-memory continuation point.
    pub fn resume<I, S, D>(
        archive_id: ArchiveId,
        session_id: SessionId,
        sources: I,
        limits: LossLedgerLimitsV1,
        durable_snapshots: D,
    ) -> Result<Self, LossLedgerError>
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
        D: IntoIterator<Item = LossSaturationSnapshotV1>,
    {
        let mut ledger = Self::new(archive_id, session_id, sources, limits)?;
        ledger.restore_durable_snapshots(durable_snapshots)?;
        Ok(ledger)
    }

    /// Restores additional durable snapshots before accepting new loss inputs.
    ///
    /// Rows for each slot must occur in increasing snapshot-sequence order.
    /// Rows from different slots may be freely interleaved, matching WAL order.
    pub fn restore_durable_snapshots<I>(&mut self, snapshots: I) -> Result<(), LossLedgerError>
    where
        I: IntoIterator<Item = LossSaturationSnapshotV1>,
    {
        if self.recording_started {
            return Err(LossLedgerError::ResumeAfterRecording);
        }
        let mut previous_states = Vec::new();
        reserve_exact(
            &mut previous_states,
            self.saturation_slots.len(),
            "saturation recovery rollback state",
        )?;
        previous_states.extend(self.saturation_slots.iter().map(|slot| slot.state));
        let previous_active = self.active_saturation_slots;
        let previous_exhausted = self.exact_capacity_exhausted;

        let result = (|| {
            for snapshot in snapshots {
                snapshot.validate().map_err(LossLedgerError::InvalidLoss)?;
                self.validate_identity(snapshot.archive_id, snapshot.session_id)?;
                self.validate_source_identifier(snapshot.source_id.as_deref())?;
                let source_index = self.resolve_source(snapshot.source_id.as_deref())?;
                let slot_index = self
                    .find_saturation_slot(source_index, snapshot.loss_kind, snapshot.reason)
                    .ok_or(LossLedgerError::IllegalSaturationTuple)?;

                let next = SaturationStateV1::from_snapshot(&snapshot);
                if let Some(previous) = self.saturation_slots[slot_index].state {
                    validate_snapshot_transition(previous, next)?;
                } else {
                    self.active_saturation_slots = self
                        .active_saturation_slots
                        .checked_add(1)
                        .ok_or(LossLedgerError::ArithmeticOverflow)?;
                }
                self.saturation_slots[slot_index].state = Some(next);
                self.exact_capacity_exhausted = true;
            }
            Ok(())
        })();
        if result.is_err() {
            for (slot, previous) in self.saturation_slots.iter_mut().zip(previous_states) {
                slot.state = previous;
            }
            self.active_saturation_slots = previous_active;
            self.exact_capacity_exhausted = previous_exhausted;
        }
        result
    }

    /// Records one already-validated semantic loss without growing ledger storage.
    ///
    /// The row identities are retained for later control-frame materialization.
    /// Callers must therefore assign them from the archive owner's sequence
    /// authority before handing the row to this method.
    pub fn record(
        &mut self,
        mut loss: ExactLossRangeV1,
    ) -> Result<LossLedgerRecordOutcomeV1, LossLedgerError> {
        loss.validate().map_err(LossLedgerError::InvalidLoss)?;
        self.validate_identity(loss.archive_id, loss.session_id)?;
        self.validate_source_identifier(loss.source_id.as_deref())?;
        self.validate_boundaries(&loss.boundary_refs)?;
        if loss.boundary_refs.len() > self.limits.max_boundary_refs_per_range {
            return Err(LossLedgerError::TooManyBoundaryReferences {
                maximum: self.limits.max_boundary_refs_per_range,
                actual: loss.boundary_refs.len(),
            });
        }
        let source_index = self.resolve_source(loss.source_id.as_deref())?;

        if let Some(exact_index) = self.find_coalescible(source_index, &loss) {
            self.coalesce(exact_index, &mut loss)?;
            self.recording_started = true;
            return Ok(LossLedgerRecordOutcomeV1::CoalescedExact { exact_index });
        }

        if !self.exact_capacity_exhausted && self.exact_ranges < self.exact_slots.len() {
            let exact_index = self.exact_ranges;
            self.retain_exact(exact_index, source_index, &mut loss);
            self.exact_ranges += 1;
            self.recording_started = true;
            return Ok(LossLedgerRecordOutcomeV1::RetainedExact { exact_index });
        }

        let slot_index = self
            .find_saturation_slot(source_index, loss.loss_kind, loss.reason)
            .ok_or(LossLedgerError::IllegalSaturationTuple)?;
        let was_inactive = self.saturation_slots[slot_index].state.is_none();
        let next = self.saturation_slots[slot_index].advance(&loss)?;
        self.exact_capacity_exhausted = true;
        if was_inactive {
            self.active_saturation_slots = self
                .active_saturation_slots
                .checked_add(1)
                .ok_or(LossLedgerError::ArithmeticOverflow)?;
        }
        self.recording_started = true;
        Ok(LossLedgerRecordOutcomeV1::Saturated {
            saturation_slot_id: self.saturation_slots[slot_index].slot_id,
            saturation_snapshot_seq: next.snapshot_seq,
            cumulative_omitted_range_count: next.omitted_range_count,
            cumulative_omitted_entry_count: next.omitted_entry_count,
        })
    }

    /// Emits another immutable cumulative snapshot without adding omitted loss.
    ///
    /// Checkpoint/finalize may repeat current cumulative state. The slot-local
    /// sequence and owner-assigned global sequences still advance, while counts
    /// and the rolling digest remain unchanged.
    pub fn reseal_saturation_snapshot(
        &mut self,
        saturation_slot_id: Digest,
        record_seq: u64,
        loss_seq: u64,
        loss_observed_ns: i64,
    ) -> Result<LossSaturationSnapshotV1, LossLedgerError> {
        let slot_index = self
            .saturation_slots
            .iter()
            .position(|slot| slot.slot_id == saturation_slot_id)
            .ok_or(LossLedgerError::UnknownSaturationSlot(saturation_slot_id))?;
        let slot = &mut self.saturation_slots[slot_index];
        let previous = slot
            .state
            .ok_or(LossLedgerError::InactiveSaturationSlot(saturation_slot_id))?;
        if record_seq <= previous.record_seq || loss_seq <= previous.loss_seq {
            return Err(LossLedgerError::NonMonotonicOwnerSequence);
        }
        if loss_observed_ns < previous.loss_observed_ns {
            return Err(LossLedgerError::NonMonotonicObservationClock);
        }
        let mut next = previous;
        next.record_seq = record_seq;
        next.loss_seq = loss_seq;
        next.loss_observed_ns = loss_observed_ns;
        next.snapshot_seq = next
            .snapshot_seq
            .checked_add(1)
            .ok_or(LossLedgerError::ArithmeticOverflow)?;
        slot.state = Some(next);
        self.recording_started = true;
        Ok(self.snapshot_at(slot_index, next))
    }

    /// Materializes one latest saturation snapshot by stable slot ID.
    #[must_use]
    pub fn saturation_snapshot(
        &self,
        saturation_slot_id: Digest,
    ) -> Option<LossSaturationSnapshotV1> {
        self.saturation_slots
            .iter()
            .enumerate()
            .find(|(_, slot)| slot.slot_id == saturation_slot_id)
            .and_then(|(index, slot)| slot.state.map(|state| self.snapshot_at(index, state)))
    }

    /// Materializes a deterministic, preparation-bounded report/query view.
    #[must_use]
    pub fn bounded_view(&self) -> LossLedgerViewV1 {
        let mut exact_ranges = Vec::with_capacity(self.exact_ranges);
        for slot in self.exact_slots.iter().take(self.exact_ranges) {
            exact_ranges.push(self.materialize_exact(slot));
        }

        let mut saturation_snapshots = Vec::with_capacity(self.active_saturation_slots);
        for (index, slot) in self.saturation_slots.iter().enumerate() {
            if let Some(state) = slot.state {
                saturation_snapshots.push(self.snapshot_at(index, state));
            }
        }
        LossLedgerViewV1 {
            exact_ranges,
            saturation_snapshots,
            complete_ranges: self.active_saturation_slots == 0,
        }
    }

    /// Returns the number of currently retained exact ranges.
    #[must_use]
    pub const fn exact_range_count(&self) -> usize {
        self.exact_ranges
    }

    /// Returns the number of saturation tuples that have observed omitted loss.
    #[must_use]
    pub const fn active_saturation_slot_count(&self) -> usize {
        self.active_saturation_slots
    }

    /// Returns the complete number of preparation-time saturation slots.
    #[must_use]
    pub fn preallocated_saturation_slot_count(&self) -> usize {
        self.saturation_slots.len()
    }

    /// Returns stable evidence that mutation has not grown internal allocations.
    #[must_use]
    pub fn allocation_shape(&self) -> LossLedgerAllocationShapeV1 {
        LossLedgerAllocationShapeV1 {
            source_count: self.sources.len(),
            source_capacity: self.sources.capacity(),
            exact_slot_count: self.exact_slots.len(),
            exact_slot_capacity: self.exact_slots.capacity(),
            boundary_reference_capacity: self
                .exact_slots
                .iter()
                .map(|slot| slot.boundary_refs.capacity())
                .sum(),
            saturation_slot_count: self.saturation_slots.len(),
            saturation_slot_capacity: self.saturation_slots.capacity(),
        }
    }

    fn validate_identity(
        &self,
        archive_id: ArchiveId,
        session_id: SessionId,
    ) -> Result<(), LossLedgerError> {
        if archive_id != self.archive_id {
            return Err(LossLedgerError::ArchiveIdentityMismatch);
        }
        if session_id != self.session_id {
            return Err(LossLedgerError::SessionIdentityMismatch);
        }
        Ok(())
    }

    fn validate_source_identifier(&self, source_id: Option<&str>) -> Result<(), LossLedgerError> {
        if let Some(source_id) = source_id {
            validate_bounded_identifier("source_id", source_id, self.limits.max_source_id_bytes)?;
        }
        Ok(())
    }

    fn validate_boundaries(&self, boundaries: &[BoundaryReference]) -> Result<(), LossLedgerError> {
        for boundary in boundaries {
            for (field, value) in [
                ("boundary.transition_id", boundary.transition_id.as_str()),
                ("boundary.boundary_id", boundary.boundary_id.as_str()),
                ("boundary.phase_id", boundary.phase_id.as_str()),
                ("boundary.source_id", boundary.source_id.as_str()),
            ] {
                validate_bounded_identifier(
                    field,
                    value,
                    self.limits.max_boundary_identifier_bytes,
                )?;
            }
            if let Some(group) = &boundary.coalescing_group_id {
                validate_bounded_identifier(
                    "boundary.coalescing_group_id",
                    group,
                    self.limits.max_boundary_identifier_bytes,
                )?;
            }
        }
        Ok(())
    }

    fn resolve_source(&self, source_id: Option<&str>) -> Result<Option<usize>, LossLedgerError> {
        let Some(source_id) = source_id else {
            return Ok(None);
        };
        self.sources
            .binary_search_by(|known| known.as_bytes().cmp(source_id.as_bytes()))
            .map(Some)
            .map_err(|_| LossLedgerError::UnknownSource(source_id.to_owned()))
    }

    fn find_coalescible(
        &self,
        source_index: Option<usize>,
        loss: &ExactLossRangeV1,
    ) -> Option<usize> {
        self.exact_slots
            .iter()
            .take(self.exact_ranges)
            .enumerate()
            .rev()
            .find_map(|(index, slot)| {
                let state = slot.state.as_ref()?;
                can_coalesce(
                    state,
                    &slot.boundary_refs,
                    source_index,
                    loss,
                    self.limits.max_boundary_refs_per_range,
                )
                .then_some(index)
            })
    }

    fn coalesce(
        &mut self,
        exact_index: usize,
        loss: &mut ExactLossRangeV1,
    ) -> Result<(), LossLedgerError> {
        let slot = &mut self.exact_slots[exact_index];
        let state = slot
            .state
            .as_mut()
            .expect("coalescible exact slots are populated");
        state.count = state
            .count
            .checked_add(loss.count)
            .ok_or(LossLedgerError::ArithmeticOverflow)?;
        state.last_source_record_seq = loss.last_source_record_seq;
        state.last_request_attempt_seq = loss.last_request_attempt_seq;
        state.last_tick = loss.last_tick;
        state.last_deadline_ns = loss.last_deadline_ns;
        state.loss_observed_ns = loss.loss_observed_ns;

        if state.boundary_overflow_count == 0 && loss.boundary_overflow_count > 0 {
            state.boundary_overflow_count = loss.boundary_overflow_count;
            state.boundary_overflow_digest = loss.boundary_overflow_digest;
        }
        slot.boundary_refs.append(&mut loss.boundary_refs);
        Ok(())
    }

    fn retain_exact(
        &mut self,
        exact_index: usize,
        source_index: Option<usize>,
        loss: &mut ExactLossRangeV1,
    ) {
        let slot = &mut self.exact_slots[exact_index];
        debug_assert!(slot.state.is_none());
        debug_assert!(slot.boundary_refs.is_empty());
        slot.state = Some(ExactRangeStateV1::from_loss(source_index, loss));
        slot.boundary_refs.append(&mut loss.boundary_refs);
    }

    fn find_saturation_slot(
        &self,
        source_index: Option<usize>,
        kind: LossKindV1,
        reason: LossReasonV1,
    ) -> Option<usize> {
        self.saturation_slots.iter().position(|slot| {
            slot.source_index == source_index && slot.kind == kind && slot.reason == reason
        })
    }

    fn materialize_exact(&self, slot: &ExactSlotV1) -> ExactLossRangeV1 {
        let state = slot
            .state
            .as_ref()
            .expect("retained exact slots are populated");
        ExactLossRangeV1 {
            archive_id: self.archive_id,
            session_id: self.session_id,
            source_id: state.source_index.map(|index| self.sources[index].clone()),
            record_seq: state.record_seq,
            loss_seq: state.loss_seq,
            count: state.count,
            loss_kind: state.kind,
            reason: state.reason,
            first_source_record_seq: state.first_source_record_seq,
            last_source_record_seq: state.last_source_record_seq,
            first_request_attempt_seq: state.first_request_attempt_seq,
            last_request_attempt_seq: state.last_request_attempt_seq,
            first_tick: state.first_tick,
            last_tick: state.last_tick,
            first_deadline_ns: state.first_deadline_ns,
            last_deadline_ns: state.last_deadline_ns,
            loss_observed_ns: state.loss_observed_ns,
            boundary_refs: slot.boundary_refs.clone(),
            boundary_overflow_count: state.boundary_overflow_count,
            boundary_overflow_digest: state.boundary_overflow_digest,
        }
    }

    fn snapshot_at(&self, slot_index: usize, state: SaturationStateV1) -> LossSaturationSnapshotV1 {
        let slot = &self.saturation_slots[slot_index];
        LossSaturationSnapshotV1 {
            archive_id: self.archive_id,
            session_id: self.session_id,
            source_id: slot.source_index.map(|index| self.sources[index].clone()),
            record_seq: state.record_seq,
            loss_seq: state.loss_seq,
            loss_kind: slot.kind,
            reason: slot.reason,
            first_source_record_seq: state.first_source_record_seq,
            last_source_record_seq: state.last_source_record_seq,
            first_request_attempt_seq: state.first_request_attempt_seq,
            last_request_attempt_seq: state.last_request_attempt_seq,
            first_tick: state.first_tick,
            last_tick: state.last_tick,
            first_deadline_ns: state.first_deadline_ns,
            last_deadline_ns: state.last_deadline_ns,
            loss_observed_ns: state.loss_observed_ns,
            saturation_slot_id: slot.slot_id,
            saturation_snapshot_seq: state.snapshot_seq,
            cumulative_omitted_range_count: state.omitted_range_count,
            cumulative_omitted_entry_count: state.omitted_entry_count,
            omitted_rolling_digest: state.rolling_digest,
        }
    }
}

#[derive(Debug)]
struct ExactSlotV1 {
    state: Option<ExactRangeStateV1>,
    boundary_refs: Vec<BoundaryReference>,
}

#[derive(Debug)]
struct ExactRangeStateV1 {
    source_index: Option<usize>,
    record_seq: u64,
    loss_seq: u64,
    count: u64,
    kind: LossKindV1,
    reason: LossReasonV1,
    first_source_record_seq: Option<u64>,
    last_source_record_seq: Option<u64>,
    first_request_attempt_seq: Option<u64>,
    last_request_attempt_seq: Option<u64>,
    first_tick: Option<u64>,
    last_tick: Option<u64>,
    first_deadline_ns: Option<i64>,
    last_deadline_ns: Option<i64>,
    loss_observed_ns: i64,
    boundary_overflow_count: u64,
    boundary_overflow_digest: Option<Digest>,
}

impl ExactRangeStateV1 {
    fn from_loss(source_index: Option<usize>, loss: &ExactLossRangeV1) -> Self {
        Self {
            source_index,
            record_seq: loss.record_seq,
            loss_seq: loss.loss_seq,
            count: loss.count,
            kind: loss.loss_kind,
            reason: loss.reason,
            first_source_record_seq: loss.first_source_record_seq,
            last_source_record_seq: loss.last_source_record_seq,
            first_request_attempt_seq: loss.first_request_attempt_seq,
            last_request_attempt_seq: loss.last_request_attempt_seq,
            first_tick: loss.first_tick,
            last_tick: loss.last_tick,
            first_deadline_ns: loss.first_deadline_ns,
            last_deadline_ns: loss.last_deadline_ns,
            loss_observed_ns: loss.loss_observed_ns,
            boundary_overflow_count: loss.boundary_overflow_count,
            boundary_overflow_digest: loss.boundary_overflow_digest,
        }
    }
}

#[derive(Debug)]
struct SaturationSlotV1 {
    source_index: Option<usize>,
    kind: LossKindV1,
    reason: LossReasonV1,
    slot_id: Digest,
    initial_digest: Digest,
    state: Option<SaturationStateV1>,
}

impl SaturationSlotV1 {
    fn new(
        archive_id: ArchiveId,
        session_id: SessionId,
        source: Option<(usize, &str)>,
        kind: LossKindV1,
    ) -> Self {
        let source_id = source.map(|(_, source_id)| source_id);
        let reason = kind.reason();
        Self {
            source_index: source.map(|(index, _)| index),
            kind,
            reason,
            slot_id: loss_saturation_slot_id_v1(archive_id, session_id, source_id, kind, reason),
            initial_digest: initial_overflow_digest(
                archive_id, session_id, source_id, kind, reason,
            ),
            state: None,
        }
    }

    fn advance(&mut self, loss: &ExactLossRangeV1) -> Result<SaturationStateV1, LossLedgerError> {
        let cumulative_ranges = match self.state {
            None => CumulativeRangesV1::from_loss(loss),
            Some(previous) => {
                if loss.record_seq <= previous.record_seq || loss.loss_seq <= previous.loss_seq {
                    return Err(LossLedgerError::NonMonotonicOwnerSequence);
                }
                if loss.loss_observed_ns < previous.loss_observed_ns {
                    return Err(LossLedgerError::NonMonotonicObservationClock);
                }
                CumulativeRangesV1::advance(previous, loss)?
            }
        };
        let previous_digest = self
            .state
            .map_or(self.initial_digest, |state| state.rolling_digest);
        let rolling_digest = advance_overflow_digest(previous_digest, loss)?;
        let next = match self.state {
            None => SaturationStateV1 {
                record_seq: loss.record_seq,
                loss_seq: loss.loss_seq,
                snapshot_seq: 0,
                omitted_range_count: 1,
                omitted_entry_count: loss.count,
                rolling_digest,
                first_source_record_seq: cumulative_ranges.first_source_record_seq,
                last_source_record_seq: cumulative_ranges.last_source_record_seq,
                first_request_attempt_seq: cumulative_ranges.first_request_attempt_seq,
                last_request_attempt_seq: cumulative_ranges.last_request_attempt_seq,
                first_tick: cumulative_ranges.first_tick,
                last_tick: cumulative_ranges.last_tick,
                first_deadline_ns: cumulative_ranges.first_deadline_ns,
                last_deadline_ns: cumulative_ranges.last_deadline_ns,
                loss_observed_ns: loss.loss_observed_ns,
            },
            Some(previous) => SaturationStateV1 {
                record_seq: loss.record_seq,
                loss_seq: loss.loss_seq,
                snapshot_seq: previous
                    .snapshot_seq
                    .checked_add(1)
                    .ok_or(LossLedgerError::ArithmeticOverflow)?,
                omitted_range_count: previous
                    .omitted_range_count
                    .checked_add(1)
                    .ok_or(LossLedgerError::ArithmeticOverflow)?,
                omitted_entry_count: previous
                    .omitted_entry_count
                    .checked_add(loss.count)
                    .ok_or(LossLedgerError::ArithmeticOverflow)?,
                rolling_digest,
                first_source_record_seq: cumulative_ranges.first_source_record_seq,
                last_source_record_seq: cumulative_ranges.last_source_record_seq,
                first_request_attempt_seq: cumulative_ranges.first_request_attempt_seq,
                last_request_attempt_seq: cumulative_ranges.last_request_attempt_seq,
                first_tick: cumulative_ranges.first_tick,
                last_tick: cumulative_ranges.last_tick,
                first_deadline_ns: cumulative_ranges.first_deadline_ns,
                last_deadline_ns: cumulative_ranges.last_deadline_ns,
                loss_observed_ns: loss.loss_observed_ns,
            },
        };
        self.state = Some(next);
        Ok(next)
    }
}

#[derive(Clone, Copy, Debug)]
struct SaturationStateV1 {
    record_seq: u64,
    loss_seq: u64,
    snapshot_seq: u64,
    omitted_range_count: u64,
    omitted_entry_count: u64,
    rolling_digest: Digest,
    first_source_record_seq: Option<u64>,
    last_source_record_seq: Option<u64>,
    first_request_attempt_seq: Option<u64>,
    last_request_attempt_seq: Option<u64>,
    first_tick: Option<u64>,
    last_tick: Option<u64>,
    first_deadline_ns: Option<i64>,
    last_deadline_ns: Option<i64>,
    loss_observed_ns: i64,
}

impl SaturationStateV1 {
    fn from_snapshot(snapshot: &LossSaturationSnapshotV1) -> Self {
        Self {
            record_seq: snapshot.record_seq,
            loss_seq: snapshot.loss_seq,
            snapshot_seq: snapshot.saturation_snapshot_seq,
            omitted_range_count: snapshot.cumulative_omitted_range_count,
            omitted_entry_count: snapshot.cumulative_omitted_entry_count,
            rolling_digest: snapshot.omitted_rolling_digest,
            first_source_record_seq: snapshot.first_source_record_seq,
            last_source_record_seq: snapshot.last_source_record_seq,
            first_request_attempt_seq: snapshot.first_request_attempt_seq,
            last_request_attempt_seq: snapshot.last_request_attempt_seq,
            first_tick: snapshot.first_tick,
            last_tick: snapshot.last_tick,
            first_deadline_ns: snapshot.first_deadline_ns,
            last_deadline_ns: snapshot.last_deadline_ns,
            loss_observed_ns: snapshot.loss_observed_ns,
        }
    }
}

fn can_coalesce(
    current: &ExactRangeStateV1,
    current_boundaries: &[BoundaryReference],
    source_index: Option<usize>,
    next: &ExactLossRangeV1,
    max_boundaries: usize,
) -> bool {
    if current.source_index != source_index
        || current.kind != next.loss_kind
        || current.reason != next.reason
        || current.count.checked_add(next.count).is_none()
        || source_index.is_none()
    {
        return false;
    }
    let source_contiguous =
        optional_u64_contiguous(current.last_source_record_seq, next.first_source_record_seq);
    let request_contiguous = optional_u64_contiguous(
        current.last_request_attempt_seq,
        next.first_request_attempt_seq,
    );
    let tick_contiguous = optional_u64_contiguous(current.last_tick, next.first_tick);
    let deadline_contiguous =
        optional_i64_increasing(current.last_deadline_ns, next.first_deadline_ns);
    if !(source_contiguous && request_contiguous && tick_contiguous && deadline_contiguous) {
        return false;
    }

    let Some(combined_boundaries) = current_boundaries
        .len()
        .checked_add(next.boundary_refs.len())
    else {
        return false;
    };
    if combined_boundaries > max_boundaries {
        return false;
    }
    if current.boundary_overflow_count > 0
        && (next.boundary_overflow_count > 0 || !next.boundary_refs.is_empty())
    {
        return false;
    }
    !current_boundaries.iter().any(|left| {
        next.boundary_refs
            .iter()
            .any(|right| left.key() == right.key())
    })
}

fn optional_u64_contiguous(current_last: Option<u64>, next_first: Option<u64>) -> bool {
    match (current_last, next_first) {
        (None, None) => true,
        (Some(current_last), Some(next_first)) => current_last.checked_add(1) == Some(next_first),
        _ => false,
    }
}

fn optional_i64_increasing(current_last: Option<i64>, next_first: Option<i64>) -> bool {
    match (current_last, next_first) {
        (None, None) => true,
        (Some(current_last), Some(next_first)) => next_first > current_last,
        _ => false,
    }
}

#[derive(Clone, Copy, Debug)]
struct CumulativeRangesV1 {
    first_source_record_seq: Option<u64>,
    last_source_record_seq: Option<u64>,
    first_request_attempt_seq: Option<u64>,
    last_request_attempt_seq: Option<u64>,
    first_tick: Option<u64>,
    last_tick: Option<u64>,
    first_deadline_ns: Option<i64>,
    last_deadline_ns: Option<i64>,
}

impl CumulativeRangesV1 {
    fn from_loss(loss: &ExactLossRangeV1) -> Self {
        Self {
            first_source_record_seq: loss.first_source_record_seq,
            last_source_record_seq: loss.last_source_record_seq,
            first_request_attempt_seq: loss.first_request_attempt_seq,
            last_request_attempt_seq: loss.last_request_attempt_seq,
            first_tick: loss.first_tick,
            last_tick: loss.last_tick,
            first_deadline_ns: loss.first_deadline_ns,
            last_deadline_ns: loss.last_deadline_ns,
        }
    }

    fn advance(
        previous: SaturationStateV1,
        loss: &ExactLossRangeV1,
    ) -> Result<Self, LossLedgerError> {
        let (first_source_record_seq, last_source_record_seq) = advance_optional_range(
            previous.first_source_record_seq,
            previous.last_source_record_seq,
            loss.first_source_record_seq,
            loss.last_source_record_seq,
        )?;
        let (first_request_attempt_seq, last_request_attempt_seq) = advance_optional_range(
            previous.first_request_attempt_seq,
            previous.last_request_attempt_seq,
            loss.first_request_attempt_seq,
            loss.last_request_attempt_seq,
        )?;
        let (first_tick, last_tick) = advance_optional_range(
            previous.first_tick,
            previous.last_tick,
            loss.first_tick,
            loss.last_tick,
        )?;
        let (first_deadline_ns, last_deadline_ns) = advance_optional_range(
            previous.first_deadline_ns,
            previous.last_deadline_ns,
            loss.first_deadline_ns,
            loss.last_deadline_ns,
        )?;
        Ok(Self {
            first_source_record_seq,
            last_source_record_seq,
            first_request_attempt_seq,
            last_request_attempt_seq,
            first_tick,
            last_tick,
            first_deadline_ns,
            last_deadline_ns,
        })
    }
}

fn advance_optional_range<T: Copy + Ord>(
    previous_first: Option<T>,
    previous_last: Option<T>,
    next_first: Option<T>,
    next_last: Option<T>,
) -> Result<(Option<T>, Option<T>), LossLedgerError> {
    match (previous_first, previous_last, next_first, next_last) {
        (None, None, None, None) => Ok((None, None)),
        (None, None, Some(next_first), Some(next_last)) => Ok((Some(next_first), Some(next_last))),
        (Some(previous_first), Some(previous_last), None, None) => {
            Ok((Some(previous_first), Some(previous_last)))
        }
        (Some(previous_first), Some(previous_last), Some(next_first), Some(next_last))
            if next_first > previous_last =>
        {
            Ok((Some(previous_first), Some(next_last)))
        }
        _ => Err(LossLedgerError::NonMonotonicOmittedIdentity),
    }
}

fn validate_snapshot_transition(
    previous: SaturationStateV1,
    next: SaturationStateV1,
) -> Result<(), LossLedgerError> {
    if next.snapshot_seq <= previous.snapshot_seq {
        return Err(LossLedgerError::NonMonotonicSnapshotSequence);
    }
    if next.record_seq <= previous.record_seq || next.loss_seq <= previous.loss_seq {
        return Err(LossLedgerError::NonMonotonicOwnerSequence);
    }
    if next.loss_observed_ns < previous.loss_observed_ns {
        return Err(LossLedgerError::NonMonotonicObservationClock);
    }

    if next.omitted_range_count == previous.omitted_range_count
        && next.omitted_entry_count == previous.omitted_entry_count
    {
        if next.rolling_digest != previous.rolling_digest || !same_ranges(previous, next) {
            return Err(LossLedgerError::InvalidRepeatedSnapshot);
        }
        return Ok(());
    }
    if next.omitted_range_count <= previous.omitted_range_count
        || next.omitted_entry_count <= previous.omitted_entry_count
    {
        return Err(LossLedgerError::NonMonotonicCumulativeCounts);
    }
    let range_delta = next.omitted_range_count - previous.omitted_range_count;
    let entry_delta = next.omitted_entry_count - previous.omitted_entry_count;
    if entry_delta < range_delta
        || next.rolling_digest == previous.rolling_digest
        || !monotonic_ranges(previous, next)
    {
        return Err(LossLedgerError::InvalidCumulativeSnapshot);
    }
    Ok(())
}

fn same_ranges(left: SaturationStateV1, right: SaturationStateV1) -> bool {
    left.first_source_record_seq == right.first_source_record_seq
        && left.last_source_record_seq == right.last_source_record_seq
        && left.first_request_attempt_seq == right.first_request_attempt_seq
        && left.last_request_attempt_seq == right.last_request_attempt_seq
        && left.first_tick == right.first_tick
        && left.last_tick == right.last_tick
        && left.first_deadline_ns == right.first_deadline_ns
        && left.last_deadline_ns == right.last_deadline_ns
}

fn monotonic_ranges(previous: SaturationStateV1, next: SaturationStateV1) -> bool {
    same_optional_first(
        previous.first_source_record_seq,
        next.first_source_record_seq,
    ) && same_optional_first(
        previous.first_request_attempt_seq,
        next.first_request_attempt_seq,
    ) && same_optional_first(previous.first_tick, next.first_tick)
        && same_optional_first(previous.first_deadline_ns, next.first_deadline_ns)
        && optional_last_monotonic(previous.last_source_record_seq, next.last_source_record_seq)
        && optional_last_monotonic(
            previous.last_request_attempt_seq,
            next.last_request_attempt_seq,
        )
        && optional_last_monotonic(previous.last_tick, next.last_tick)
        && optional_last_monotonic(previous.last_deadline_ns, next.last_deadline_ns)
}

fn same_optional_first<T: Eq>(previous: Option<T>, next: Option<T>) -> bool {
    previous == next
}

fn optional_last_monotonic<T: Ord>(previous: Option<T>, next: Option<T>) -> bool {
    match (previous, next) {
        (None, None) => true,
        (Some(previous), Some(next)) => next >= previous,
        _ => false,
    }
}

fn initial_overflow_digest(
    archive_id: ArchiveId,
    session_id: SessionId,
    source_id: Option<&str>,
    kind: LossKindV1,
    reason: LossReasonV1,
) -> Digest {
    let mut hasher = domain_hasher(LOSS_OVERFLOW_DOMAIN_V1);
    hash_field(&mut hasher, archive_id.as_bytes());
    hash_field(&mut hasher, session_id.as_bytes());
    hash_optional_source_field(&mut hasher, source_id);
    hash_field(&mut hasher, &[kind as u8]);
    hash_field(&mut hasher, &[reason as u8]);
    Digest::from_bytes(*hasher.finalize().as_bytes())
}

fn advance_overflow_digest(
    previous: Digest,
    loss: &ExactLossRangeV1,
) -> Result<Digest, LossLedgerError> {
    let canonical_length = canonical_omitted_entry_length(loss)?;
    let mut hasher = domain_hasher(LOSS_OVERFLOW_DOMAIN_V1);
    hash_field(&mut hasher, previous.as_bytes());
    hasher.update(&canonical_length.to_be_bytes());
    hash_canonical_omitted_entry(&mut hasher, loss)?;
    Ok(Digest::from_bytes(*hasher.finalize().as_bytes()))
}

fn domain_hasher(domain: &str) -> blake3::Hasher {
    let mut hasher = blake3::Hasher::new();
    hasher.update(domain.as_bytes());
    hasher.update(&[0]);
    hasher
}

fn hash_field(hasher: &mut blake3::Hasher, bytes: &[u8]) {
    let length = u64::try_from(bytes.len()).expect("supported usize fits u64");
    hasher.update(&length.to_be_bytes());
    hasher.update(bytes);
}

fn hash_optional_source_field(hasher: &mut blake3::Hasher, source_id: Option<&str>) {
    match source_id {
        None => hash_field(hasher, &[0]),
        Some(source_id) => {
            let length = u64::try_from(source_id.len())
                .expect("supported usize fits u64")
                .checked_add(1)
                .expect("validated source length fits u64");
            hasher.update(&length.to_be_bytes());
            hasher.update(&[1]);
            hasher.update(source_id.as_bytes());
        }
    }
}

fn canonical_omitted_entry_length(loss: &ExactLossRangeV1) -> Result<u64, LossLedgerError> {
    let mut length = 0_u64;
    add_encoded_field(&mut length, 16)?;
    add_encoded_field(&mut length, 16)?;
    add_encoded_field(
        &mut length,
        optional_source_payload_length(loss.source_id.as_deref())?,
    )?;
    for payload_length in [8_u64, 8, 8, 1, 1] {
        add_encoded_field(&mut length, payload_length)?;
    }
    for value in [
        loss.first_source_record_seq,
        loss.last_source_record_seq,
        loss.first_request_attempt_seq,
        loss.last_request_attempt_seq,
        loss.first_tick,
        loss.last_tick,
    ] {
        add_encoded_field(
            &mut length,
            optional_fixed_payload_length(value.is_some(), 8),
        )?;
    }
    for value in [loss.first_deadline_ns, loss.last_deadline_ns] {
        add_encoded_field(
            &mut length,
            optional_fixed_payload_length(value.is_some(), 8),
        )?;
    }
    add_encoded_field(&mut length, 8)?;

    let mut boundaries_length = 8_u64;
    for boundary in &loss.boundary_refs {
        let boundary_length = canonical_boundary_length(boundary)?;
        boundaries_length = boundaries_length
            .checked_add(8)
            .and_then(|value| value.checked_add(boundary_length))
            .ok_or(LossLedgerError::ArithmeticOverflow)?;
    }
    add_encoded_field(&mut length, boundaries_length)?;
    add_encoded_field(&mut length, 8)?;
    add_encoded_field(
        &mut length,
        optional_fixed_payload_length(loss.boundary_overflow_digest.is_some(), 32),
    )?;
    Ok(length)
}

fn canonical_boundary_length(boundary: &BoundaryReference) -> Result<u64, LossLedgerError> {
    let mut length = 0_u64;
    for value in [
        boundary.transition_id.as_bytes(),
        boundary.boundary_id.as_bytes(),
        boundary.phase_id.as_bytes(),
        boundary.source_id.as_bytes(),
    ] {
        add_encoded_field(
            &mut length,
            u64::try_from(value.len()).map_err(|_| LossLedgerError::ArithmeticOverflow)?,
        )?;
    }
    add_encoded_field(&mut length, 1)?;
    let group_length = match &boundary.coalescing_group_id {
        None => 1,
        Some(group) => u64::try_from(group.len())
            .map_err(|_| LossLedgerError::ArithmeticOverflow)?
            .checked_add(1)
            .ok_or(LossLedgerError::ArithmeticOverflow)?,
    };
    add_encoded_field(&mut length, group_length)?;
    Ok(length)
}

fn add_encoded_field(total: &mut u64, payload_length: u64) -> Result<(), LossLedgerError> {
    *total = total
        .checked_add(8)
        .and_then(|value| value.checked_add(payload_length))
        .ok_or(LossLedgerError::ArithmeticOverflow)?;
    Ok(())
}

fn optional_source_payload_length(source_id: Option<&str>) -> Result<u64, LossLedgerError> {
    match source_id {
        None => Ok(1),
        Some(source_id) => u64::try_from(source_id.len())
            .map_err(|_| LossLedgerError::ArithmeticOverflow)?
            .checked_add(1)
            .ok_or(LossLedgerError::ArithmeticOverflow),
    }
}

const fn optional_fixed_payload_length(present: bool, fixed_length: u64) -> u64 {
    if present { 1 + fixed_length } else { 1 }
}

fn hash_canonical_omitted_entry(
    hasher: &mut blake3::Hasher,
    loss: &ExactLossRangeV1,
) -> Result<(), LossLedgerError> {
    hash_field(hasher, loss.archive_id.as_bytes());
    hash_field(hasher, loss.session_id.as_bytes());
    hash_optional_source_field(hasher, loss.source_id.as_deref());
    hash_field(hasher, &loss.record_seq.to_be_bytes());
    hash_field(hasher, &loss.loss_seq.to_be_bytes());
    hash_field(hasher, &loss.count.to_be_bytes());
    hash_field(hasher, &[loss.loss_kind as u8]);
    hash_field(hasher, &[loss.reason as u8]);
    for value in [
        loss.first_source_record_seq,
        loss.last_source_record_seq,
        loss.first_request_attempt_seq,
        loss.last_request_attempt_seq,
        loss.first_tick,
        loss.last_tick,
    ] {
        hash_optional_u64_field(hasher, value);
    }
    hash_optional_i64_field(hasher, loss.first_deadline_ns);
    hash_optional_i64_field(hasher, loss.last_deadline_ns);
    hash_field(hasher, &loss.loss_observed_ns.to_be_bytes());

    let mut boundaries_length = 8_u64;
    for boundary in &loss.boundary_refs {
        boundaries_length = boundaries_length
            .checked_add(8)
            .and_then(|value| value.checked_add(canonical_boundary_length(boundary).ok()?))
            .ok_or(LossLedgerError::ArithmeticOverflow)?;
    }
    hasher.update(&boundaries_length.to_be_bytes());
    hasher.update(
        &u64::try_from(loss.boundary_refs.len())
            .map_err(|_| LossLedgerError::ArithmeticOverflow)?
            .to_be_bytes(),
    );
    for boundary in &loss.boundary_refs {
        let boundary_length = canonical_boundary_length(boundary)?;
        hasher.update(&boundary_length.to_be_bytes());
        hash_canonical_boundary(hasher, boundary);
    }
    hash_field(hasher, &loss.boundary_overflow_count.to_be_bytes());
    hash_optional_digest_field(hasher, loss.boundary_overflow_digest);
    Ok(())
}

fn hash_canonical_boundary(hasher: &mut blake3::Hasher, boundary: &BoundaryReference) {
    hash_field(hasher, boundary.transition_id.as_bytes());
    hash_field(hasher, boundary.boundary_id.as_bytes());
    hash_field(hasher, boundary.phase_id.as_bytes());
    hash_field(hasher, boundary.source_id.as_bytes());
    hash_field(hasher, &[boundary.role as u8]);
    match &boundary.coalescing_group_id {
        None => hash_field(hasher, &[0]),
        Some(group) => {
            let length = u64::try_from(group.len())
                .expect("supported usize fits u64")
                .checked_add(1)
                .expect("validated boundary identifier fits u64");
            hasher.update(&length.to_be_bytes());
            hasher.update(&[1]);
            hasher.update(group.as_bytes());
        }
    }
}

fn hash_optional_u64_field(hasher: &mut blake3::Hasher, value: Option<u64>) {
    match value {
        None => hash_field(hasher, &[0]),
        Some(value) => {
            hasher.update(&9_u64.to_be_bytes());
            hasher.update(&[1]);
            hasher.update(&value.to_be_bytes());
        }
    }
}

fn hash_optional_i64_field(hasher: &mut blake3::Hasher, value: Option<i64>) {
    match value {
        None => hash_field(hasher, &[0]),
        Some(value) => {
            hasher.update(&9_u64.to_be_bytes());
            hasher.update(&[1]);
            hasher.update(&value.to_be_bytes());
        }
    }
}

fn hash_optional_digest_field(hasher: &mut blake3::Hasher, value: Option<Digest>) {
    match value {
        None => hash_field(hasher, &[0]),
        Some(value) => {
            hasher.update(&33_u64.to_be_bytes());
            hasher.update(&[1]);
            hasher.update(value.as_bytes());
        }
    }
}

fn validate_bounded_identifier(
    field: &'static str,
    value: &str,
    maximum_bytes: usize,
) -> Result<(), LossLedgerError> {
    if value.is_empty() || value.trim() != value || value.chars().any(char::is_control) {
        return Err(LossLedgerError::InvalidIdentifier {
            field,
            value: value.to_owned(),
        });
    }
    if value.len() > maximum_bytes {
        return Err(LossLedgerError::IdentifierTooLong {
            field,
            maximum_bytes,
            actual_bytes: value.len(),
        });
    }
    Ok(())
}

fn reserve_exact<T>(
    values: &mut Vec<T>,
    additional: usize,
    allocation: &'static str,
) -> Result<(), LossLedgerError> {
    values
        .try_reserve_exact(additional)
        .map_err(|_| LossLedgerError::AllocationFailed(allocation))
}

/// Rejected ledger preparation, mutation, or durable recovery input.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum LossLedgerError {
    /// A preparation limit that must be positive was zero.
    ZeroLimit(&'static str),
    /// The authored source universe exceeded its validated maximum.
    TooManySources {
        /// Validated maximum.
        maximum: usize,
        /// Authored source count.
        actual: usize,
    },
    /// A source identity occurred more than once.
    DuplicateSource(String),
    /// An identifier was empty, padded, or contained control characters.
    InvalidIdentifier {
        /// Invalid field.
        field: &'static str,
        /// Redaction-safe value.
        value: String,
    },
    /// An identifier exceeded its validated UTF-8 byte bound.
    IdentifierTooLong {
        /// Oversized field.
        field: &'static str,
        /// Validated maximum byte length.
        maximum_bytes: usize,
        /// Actual byte length.
        actual_bytes: usize,
    },
    /// A loss carried more retained boundary references than one exact slot permits.
    TooManyBoundaryReferences {
        /// Validated maximum.
        maximum: usize,
        /// Actual reference count.
        actual: usize,
    },
    /// An internal fixed allocation could not be reserved during preparation.
    AllocationFailed(&'static str),
    /// Checked arithmetic overflowed.
    ArithmeticOverflow,
    /// An input row did not belong to this archive.
    ArchiveIdentityMismatch,
    /// An input row did not belong to this collection session.
    SessionIdentityMismatch,
    /// A loss named a source outside the frozen preparation universe.
    UnknownSource(String),
    /// The kind/reason/source role has no preallocated v1 saturation tuple.
    IllegalSaturationTuple,
    /// The existing exact/loss DTO rejected the semantic row.
    InvalidLoss(LossValidationError),
    /// Durable recovery was attempted after new exact loss had been recorded.
    ResumeAfterRecording,
    /// Durable snapshot sequences failed to increase within one slot.
    NonMonotonicSnapshotSequence,
    /// Owner record/loss sequences failed to increase within one slot.
    NonMonotonicOwnerSequence,
    /// Snapshot observation Clock regressed within one slot.
    NonMonotonicObservationClock,
    /// A later omitted range regressed or overlapped cumulative identity facts.
    NonMonotonicOmittedIdentity,
    /// Cumulative omitted counts regressed or only one count advanced.
    NonMonotonicCumulativeCounts,
    /// A repeated cumulative snapshot changed digest or range evidence.
    InvalidRepeatedSnapshot,
    /// An advancing cumulative snapshot had inconsistent counts, digest, or ranges.
    InvalidCumulativeSnapshot,
    /// A stable slot identity was not part of this ledger's prepared universe.
    UnknownSaturationSlot(Digest),
    /// A checkpoint attempted to reseal a tuple that has never saturated.
    InactiveSaturationSlot(Digest),
}

impl Display for LossLedgerError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroLimit(limit) => {
                write!(formatter, "loss ledger limit {limit} must be positive")
            }
            Self::TooManySources { maximum, actual } => write!(
                formatter,
                "loss ledger source count {actual} exceeds validated maximum {maximum}"
            ),
            Self::DuplicateSource(source) => {
                write!(
                    formatter,
                    "loss ledger source {source:?} occurs more than once"
                )
            }
            Self::InvalidIdentifier { field, value } => {
                write!(
                    formatter,
                    "loss ledger {field} has invalid identifier {value:?}"
                )
            }
            Self::IdentifierTooLong {
                field,
                maximum_bytes,
                actual_bytes,
            } => write!(
                formatter,
                "loss ledger {field} is {actual_bytes} bytes; maximum is {maximum_bytes}"
            ),
            Self::TooManyBoundaryReferences { maximum, actual } => write!(
                formatter,
                "loss range carries {actual} boundary references; maximum is {maximum}"
            ),
            Self::AllocationFailed(allocation) => {
                write!(formatter, "could not preallocate {allocation}")
            }
            Self::ArithmeticOverflow => formatter.write_str("loss ledger arithmetic overflowed"),
            Self::ArchiveIdentityMismatch => {
                formatter.write_str("loss ledger archive identity mismatch")
            }
            Self::SessionIdentityMismatch => {
                formatter.write_str("loss ledger session identity mismatch")
            }
            Self::UnknownSource(source) => {
                write!(formatter, "loss ledger source {source:?} was not prepared")
            }
            Self::IllegalSaturationTuple => {
                formatter.write_str("loss tuple has no legal preallocated saturation slot")
            }
            Self::InvalidLoss(error) => write!(formatter, "invalid telemetry loss: {error}"),
            Self::ResumeAfterRecording => {
                formatter.write_str("cannot restore saturation snapshots after recording loss")
            }
            Self::NonMonotonicSnapshotSequence => {
                formatter.write_str("saturation snapshot sequence did not increase")
            }
            Self::NonMonotonicOwnerSequence => {
                formatter.write_str("saturation owner record/loss sequence did not increase")
            }
            Self::NonMonotonicObservationClock => {
                formatter.write_str("saturation observation Clock regressed")
            }
            Self::NonMonotonicOmittedIdentity => {
                formatter.write_str("saturation omitted identity range is not monotonic")
            }
            Self::NonMonotonicCumulativeCounts => {
                formatter.write_str("saturation cumulative counts are not monotonic")
            }
            Self::InvalidRepeatedSnapshot => {
                formatter.write_str("repeated saturation snapshot changed cumulative evidence")
            }
            Self::InvalidCumulativeSnapshot => {
                formatter.write_str("advancing saturation snapshot has invalid cumulative evidence")
            }
            Self::UnknownSaturationSlot(slot) => {
                write!(formatter, "unknown loss saturation slot {slot}")
            }
            Self::InactiveSaturationSlot(slot) => {
                write!(formatter, "loss saturation slot {slot} is inactive")
            }
        }
    }
}

impl std::error::Error for LossLedgerError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::BoundaryRole;

    fn archive_id() -> ArchiveId {
        ArchiveId::new([0x11; 16]).unwrap()
    }

    fn session_id() -> SessionId {
        SessionId::new([0x22; 16]).unwrap()
    }

    fn limits(max_exact_ranges: usize) -> LossLedgerLimitsV1 {
        LossLedgerLimitsV1 {
            max_exact_ranges,
            max_sources: 4,
            max_source_id_bytes: 32,
            max_boundary_refs_per_range: 4,
            max_boundary_identifier_bytes: 64,
        }
    }

    fn issued(
        source: &str,
        source_seq: u64,
        request_seq: Option<u64>,
        kind: LossKindV1,
        owner_seq: u64,
    ) -> ExactLossRangeV1 {
        ExactLossRangeV1 {
            archive_id: archive_id(),
            session_id: session_id(),
            source_id: Some(source.to_owned()),
            record_seq: owner_seq,
            loss_seq: owner_seq,
            count: 1,
            loss_kind: kind,
            reason: kind.reason(),
            first_source_record_seq: Some(source_seq),
            last_source_record_seq: Some(source_seq),
            first_request_attempt_seq: request_seq,
            last_request_attempt_seq: request_seq,
            first_tick: None,
            last_tick: None,
            first_deadline_ns: None,
            last_deadline_ns: None,
            loss_observed_ns: i64::try_from(owner_seq).unwrap() * 100,
            boundary_refs: Vec::new(),
            boundary_overflow_count: 0,
            boundary_overflow_digest: None,
        }
    }

    fn missed(source: &str, tick: u64, owner_seq: u64) -> ExactLossRangeV1 {
        ExactLossRangeV1 {
            archive_id: archive_id(),
            session_id: session_id(),
            source_id: Some(source.to_owned()),
            record_seq: owner_seq,
            loss_seq: owner_seq,
            count: 1,
            loss_kind: LossKindV1::MissedCadence,
            reason: LossReasonV1::CadenceOverrun,
            first_source_record_seq: None,
            last_source_record_seq: None,
            first_request_attempt_seq: None,
            last_request_attempt_seq: None,
            first_tick: Some(tick),
            last_tick: Some(tick),
            first_deadline_ns: Some(i64::try_from(tick).unwrap() * 1_000),
            last_deadline_ns: Some(i64::try_from(tick).unwrap() * 1_000),
            loss_observed_ns: i64::try_from(owner_seq).unwrap() * 100,
            boundary_refs: Vec::new(),
            boundary_overflow_count: 0,
            boundary_overflow_digest: None,
        }
    }

    fn global(kind: LossKindV1, owner_seq: u64) -> ExactLossRangeV1 {
        ExactLossRangeV1 {
            archive_id: archive_id(),
            session_id: session_id(),
            source_id: None,
            record_seq: owner_seq,
            loss_seq: owner_seq,
            count: 1,
            loss_kind: kind,
            reason: kind.reason(),
            first_source_record_seq: None,
            last_source_record_seq: None,
            first_request_attempt_seq: None,
            last_request_attempt_seq: None,
            first_tick: None,
            last_tick: None,
            first_deadline_ns: None,
            last_deadline_ns: None,
            loss_observed_ns: i64::try_from(owner_seq).unwrap() * 100,
            boundary_refs: Vec::new(),
            boundary_overflow_count: 0,
            boundary_overflow_digest: None,
        }
    }

    #[test]
    fn coalesces_only_matching_contiguous_identity_shapes() {
        let mut ledger = FixedLossLedgerV1::new(
            archive_id(),
            session_id(),
            ["source-b", "source-a"],
            limits(8),
        )
        .unwrap();
        assert_eq!(
            ledger
                .record(issued(
                    "source-a",
                    10,
                    Some(50),
                    LossKindV1::ArchiveRejected,
                    1,
                ))
                .unwrap(),
            LossLedgerRecordOutcomeV1::RetainedExact { exact_index: 0 }
        );
        ledger
            .record(issued(
                "source-b",
                1,
                Some(1),
                LossKindV1::ArchiveRejected,
                2,
            ))
            .unwrap();
        assert_eq!(
            ledger
                .record(issued(
                    "source-a",
                    11,
                    Some(51),
                    LossKindV1::ArchiveRejected,
                    3,
                ))
                .unwrap(),
            LossLedgerRecordOutcomeV1::CoalescedExact { exact_index: 0 }
        );

        ledger.record(missed("source-a", 7, 4)).unwrap();
        assert!(matches!(
            ledger.record(missed("source-a", 8, 5)).unwrap(),
            LossLedgerRecordOutcomeV1::CoalescedExact { .. }
        ));
        ledger
            .record(issued("source-a", 12, None, LossKindV1::ArchiveRejected, 6))
            .unwrap();

        let view = ledger.bounded_view();
        assert_eq!(view.exact_ranges.len(), 4);
        assert_eq!(view.exact_ranges[0].count, 2);
        assert_eq!(view.exact_ranges[0].first_source_record_seq, Some(10));
        assert_eq!(view.exact_ranges[0].last_source_record_seq, Some(11));
        assert_eq!(view.exact_ranges[2].count, 2);
        assert_eq!(view.exact_ranges[2].first_tick, Some(7));
        assert_eq!(view.exact_ranges[2].last_tick, Some(8));
        assert!(view.complete_ranges);
    }

    #[test]
    fn exact_capacity_overflow_is_cumulative_and_order_sensitive() {
        fn run(order: [u64; 2]) -> LossSaturationSnapshotV1 {
            let mut ledger =
                FixedLossLedgerV1::new(archive_id(), session_id(), ["source-a"], limits(1))
                    .unwrap();
            ledger
                .record(issued(
                    "source-a",
                    100,
                    Some(100),
                    LossKindV1::ArchiveRejected,
                    1,
                ))
                .unwrap();
            for (index, count) in order.into_iter().enumerate() {
                let mut loss = global(LossKindV1::WriterFailed, u64::try_from(index).unwrap() + 2);
                loss.count = count;
                let outcome = ledger.record(loss).unwrap();
                assert!(matches!(
                    outcome,
                    LossLedgerRecordOutcomeV1::Saturated { .. }
                ));
            }
            let view = ledger.bounded_view();
            assert!(!view.complete_ranges);
            assert_eq!(view.exact_ranges.len(), 1);
            assert_eq!(view.saturation_snapshots.len(), 1);
            let snapshot = view.saturation_snapshots.into_iter().next().unwrap();
            assert_eq!(snapshot.saturation_snapshot_seq, 1);
            assert_eq!(snapshot.cumulative_omitted_range_count, 2);
            assert_eq!(snapshot.cumulative_omitted_entry_count, 3);
            assert_eq!(snapshot.first_source_record_seq, None);
            assert_eq!(snapshot.last_source_record_seq, None);
            snapshot
        }

        let forward = run([1, 2]);
        let reverse = run([2, 1]);
        assert_eq!(forward.saturation_slot_id, reverse.saturation_slot_id);
        assert_ne!(
            forward.omitted_rolling_digest,
            reverse.omitted_rolling_digest
        );
    }

    #[test]
    fn durable_snapshots_resume_monotonically_and_latest_wins() {
        let mut original =
            FixedLossLedgerV1::new(archive_id(), session_id(), ["source-a"], limits(1)).unwrap();
        original
            .record(issued(
                "source-a",
                100,
                Some(100),
                LossKindV1::ProjectionFailed,
                1,
            ))
            .unwrap();
        original
            .record(issued(
                "source-a",
                1,
                Some(1),
                LossKindV1::ProjectionFailed,
                2,
            ))
            .unwrap();
        let first = original.bounded_view().saturation_snapshots.remove(0);
        let repeated = original
            .reseal_saturation_snapshot(first.saturation_slot_id, 3, 3, 300)
            .unwrap();

        let mut resumed = FixedLossLedgerV1::resume(
            archive_id(),
            session_id(),
            ["source-a"],
            limits(1),
            [first.clone(), repeated.clone()],
        )
        .unwrap();
        resumed
            .record(issued(
                "source-a",
                9,
                Some(9),
                LossKindV1::ProjectionFailed,
                4,
            ))
            .unwrap();
        let latest = resumed.bounded_view().saturation_snapshots.remove(0);
        assert_eq!(latest.saturation_snapshot_seq, 2);
        assert_eq!(latest.cumulative_omitted_range_count, 2);
        assert_eq!(latest.cumulative_omitted_entry_count, 2);
        assert_ne!(
            latest.omitted_rolling_digest,
            repeated.omitted_rolling_digest
        );

        let mut regressed = repeated;
        regressed.saturation_snapshot_seq = first.saturation_snapshot_seq;
        assert_eq!(
            FixedLossLedgerV1::resume(
                archive_id(),
                session_id(),
                ["source-a"],
                limits(1),
                [first, regressed],
            )
            .unwrap_err(),
            LossLedgerError::NonMonotonicSnapshotSequence
        );
    }

    #[test]
    fn source_and_global_role_matrix_is_closed_and_preallocated() {
        let mut ledger = FixedLossLedgerV1::new(
            archive_id(),
            session_id(),
            ["source-a", "source-b"],
            limits(8),
        )
        .unwrap();
        assert_eq!(ledger.preallocated_saturation_slot_count(), 2 + 2 * 5);
        assert!(ledger.record(global(LossKindV1::WriterFailed, 1)).is_ok());
        assert!(
            ledger
                .record(global(LossKindV1::ShutdownAbandoned, 2))
                .is_ok()
        );
        assert!(
            ledger
                .record(issued("source-a", 1, None, LossKindV1::WriterFailed, 3,))
                .is_ok()
        );

        let rejected_global = global(LossKindV1::ArchiveRejected, 4);
        assert!(matches!(
            ledger.record(rejected_global),
            Err(LossLedgerError::InvalidLoss(
                LossValidationError::IssuedLossRequiresSource(LossKindV1::ArchiveRejected)
            ))
        ));
        let mut missed_global = missed("source-a", 1, 5);
        missed_global.source_id = None;
        assert!(matches!(
            ledger.record(missed_global),
            Err(LossLedgerError::InvalidLoss(
                LossValidationError::MissedCadenceRequiresSource
            ))
        ));
        assert_eq!(
            ledger
                .record(issued("source-c", 1, None, LossKindV1::WriterFailed, 6,))
                .unwrap_err(),
            LossLedgerError::UnknownSource("source-c".to_owned())
        );
    }

    #[test]
    fn mutation_never_grows_preallocated_internal_storage() {
        let mut custom_limits = limits(3);
        custom_limits.max_boundary_refs_per_range = 2;
        let mut ledger = FixedLossLedgerV1::new(
            archive_id(),
            session_id(),
            ["source-a", "source-b"],
            custom_limits,
        )
        .unwrap();
        let prepared = ledger.allocation_shape();

        let mut with_boundary = issued("source-a", 0, Some(0), LossKindV1::ArchiveRejected, 1);
        with_boundary.boundary_refs.push(BoundaryReference {
            transition_id: "transition-0".to_owned(),
            boundary_id: "boundary-0".to_owned(),
            phase_id: "phase-0".to_owned(),
            source_id: "source-a".to_owned(),
            role: BoundaryRole::PhaseStart,
            coalescing_group_id: None,
        });
        ledger.record(with_boundary).unwrap();

        for index in 1..1_000_u64 {
            let source = if index % 2 == 0 {
                "source-a"
            } else {
                "source-b"
            };
            let kind = match index % 4 {
                0 => LossKindV1::ArchiveRejected,
                1 => LossKindV1::ProjectionFailed,
                2 => LossKindV1::WriterFailed,
                _ => LossKindV1::ShutdownAbandoned,
            };
            ledger
                .record(issued(source, index * 2, None, kind, index + 1))
                .unwrap();
        }

        assert_eq!(ledger.allocation_shape(), prepared);
        assert_eq!(ledger.exact_range_count(), 3);
        assert!(ledger.active_saturation_slot_count() <= 12);
        let view = ledger.bounded_view();
        assert!(view.exact_ranges.len() <= custom_limits.max_exact_ranges);
        assert!(view.saturation_snapshots.len() <= 12);
        assert!(!view.complete_ranges);
    }

    #[test]
    fn alternating_noncoalescible_ranges_preserve_exact_totals() {
        for entries in 1..128_u64 {
            let mut ledger =
                FixedLossLedgerV1::new(archive_id(), session_id(), ["source-a"], limits(1))
                    .unwrap();
            ledger
                .record(issued("source-a", 0, None, LossKindV1::ArchiveRejected, 1))
                .unwrap();
            for index in 0..entries {
                let mut loss = issued(
                    "source-a",
                    index * 2 + 10,
                    None,
                    LossKindV1::ArchiveRejected,
                    index + 2,
                );
                loss.count = 2;
                loss.last_source_record_seq = loss.first_source_record_seq.map(|value| value + 1);
                ledger.record(loss).unwrap();
            }
            let snapshot = ledger.bounded_view().saturation_snapshots.remove(0);
            assert_eq!(snapshot.cumulative_omitted_range_count, entries);
            assert_eq!(snapshot.cumulative_omitted_entry_count, entries * 2);
            assert_eq!(snapshot.count(), entries * 2);
            assert!(snapshot.validate().is_ok());
        }
    }
}
