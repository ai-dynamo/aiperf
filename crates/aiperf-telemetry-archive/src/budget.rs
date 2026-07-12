// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Prepared spool-quota authority and protected transaction reserves.
//!
//! Admission consumes only atomics. Filesystem traversal and `statvfs` remain
//! outside the request path and enter through [`ArchiveSpoolObservation`]. A
//! single owner commits successful leases conservatively and may reconcile at
//! a fence where no lease is outstanding. Ordinary work can never debit the
//! control or finalization lanes, and control work can never debit the final
//! reserve.

use std::fmt::{self, Debug, Display, Formatter};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use crate::{
    AdmissionRejection, ArchiveAdmissionPolicy, ArchiveIngressState, ArchiveProjectionFootprint,
    ArchiveProjectionPermit,
};

const MAX_RESERVATION_RETRIES: usize = 8;

/// Byte and file/inode amount used by one reserve component.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ArchiveSpoolResources {
    /// Conservative logical or physical bytes.
    pub bytes: u64,
    /// Conservative logical files or filesystem inodes.
    pub files: u64,
}

impl ArchiveSpoolResources {
    /// Adds two resource amounts with overflow rejection.
    pub fn checked_add(self, other: Self) -> Result<Self, ArchiveSpoolBudgetError> {
        Ok(Self {
            bytes: self
                .bytes
                .checked_add(other.bytes)
                .ok_or(ArchiveSpoolBudgetError::ArithmeticOverflow)?,
            files: self
                .files
                .checked_add(other.files)
                .ok_or(ArchiveSpoolBudgetError::ArithmeticOverflow)?,
        })
    }

    fn is_positive(self) -> bool {
        self.bytes != 0 && self.files != 0
    }
}

/// Named conservative reserve required to finish or diagnose one archive.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ArchiveSpoolReservePlan {
    /// Largest complete WAL frame which may already be in flight.
    pub largest_wal_frame: ArchiveSpoolResources,
    /// One retained fallback WAL generation/window.
    pub fallback_wal_window: ArchiveSpoolResources,
    /// Every open and temporary Parquet builder at its prepared bound.
    pub open_parquet_builders: ArchiveSpoolResources,
    /// Worst-case copy-on-write index path.
    pub cow_index_path: ArchiveSpoolResources,
    /// Generation object plus current/preceding head transaction.
    pub generation_and_head: ArchiveSpoolResources,
    /// Receipt batch/index/head/pointer transaction.
    pub receipt_transaction: ArchiveSpoolResources,
    /// Optional raw object transaction, zero when raw retention is disabled.
    pub optional_raw_object: ArchiveSpoolResources,
    /// WAL footer, rename, and directory durability allowance.
    pub wal_seal: ArchiveSpoolResources,
    /// Additional final error/health and terminal marker allowance.
    pub emergency_finalization: ArchiveSpoolResources,
    /// Durable attached-loss/lifecycle lane outside ordinary admission.
    pub control_lane: ArchiveSpoolResources,
}

impl ArchiveSpoolReservePlan {
    /// Validates every mandatory named reserve component and checked totals.
    pub fn validate(self) -> Result<(), ArchiveSpoolBudgetError> {
        for (name, resources) in [
            ("largest_wal_frame", self.largest_wal_frame),
            ("fallback_wal_window", self.fallback_wal_window),
            ("open_parquet_builders", self.open_parquet_builders),
            ("cow_index_path", self.cow_index_path),
            ("generation_and_head", self.generation_and_head),
            ("receipt_transaction", self.receipt_transaction),
            ("wal_seal", self.wal_seal),
            ("emergency_finalization", self.emergency_finalization),
            ("control_lane", self.control_lane),
        ] {
            if !resources.is_positive() {
                return Err(ArchiveSpoolBudgetError::InvalidReserveComponent(name));
            }
        }
        if (self.optional_raw_object.bytes == 0) != (self.optional_raw_object.files == 0) {
            return Err(ArchiveSpoolBudgetError::InvalidReserveComponent(
                "optional_raw_object",
            ));
        }
        self.finalization_reserve()?;
        self.protected_reserve()?;
        Ok(())
    }

    /// Returns the resources unavailable even to the reserved control lane.
    pub fn finalization_reserve(self) -> Result<ArchiveSpoolResources, ArchiveSpoolBudgetError> {
        [
            self.largest_wal_frame,
            self.fallback_wal_window,
            self.open_parquet_builders,
            self.cow_index_path,
            self.generation_and_head,
            self.receipt_transaction,
            self.optional_raw_object,
            self.wal_seal,
            self.emergency_finalization,
        ]
        .into_iter()
        .try_fold(ArchiveSpoolResources::default(), |total, component| {
            total.checked_add(component)
        })
    }

    /// Returns finalization plus the separately bounded control lane.
    pub fn protected_reserve(self) -> Result<ArchiveSpoolResources, ArchiveSpoolBudgetError> {
        self.finalization_reserve()?.checked_add(self.control_lane)
    }
}

/// Configured hard quotas and bounded data/control queue capacities.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ArchiveSpoolBudgetLimits {
    /// Authored logical byte and file quotas.
    pub quota: ArchiveSpoolResources,
    /// Maximum simultaneous ordinary data leases.
    pub ordinary_frame_capacity: u64,
    /// Maximum simultaneous control-lane leases.
    pub control_frame_capacity: u64,
    /// Named protected reserve calculation.
    pub reserve: ArchiveSpoolReservePlan,
}

impl ArchiveSpoolBudgetLimits {
    fn validate(self) -> Result<(), ArchiveSpoolBudgetError> {
        if !self.quota.is_positive() {
            return Err(ArchiveSpoolBudgetError::InvalidQuota);
        }
        if self.ordinary_frame_capacity == 0 || self.control_frame_capacity == 0 {
            return Err(ArchiveSpoolBudgetError::InvalidFrameCapacity);
        }
        self.reserve.validate()
    }
}

/// Blocking observation captured from the qualified spool outside admission.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ArchiveSpoolObservation {
    /// Sum of logical lengths for every regular file under the spool.
    pub logical_bytes: u64,
    /// Number of regular files under the spool.
    pub logical_files: u64,
    /// Fresh filesystem bytes available to the unprivileged writer.
    pub filesystem_available_bytes: u64,
    /// Fresh filesystem inodes available to the unprivileged writer.
    pub filesystem_available_files: u64,
}

impl ArchiveSpoolObservation {
    fn validate(self) -> Result<(), ArchiveSpoolBudgetError> {
        if self.filesystem_available_bytes == 0 || self.filesystem_available_files == 0 {
            return Err(ArchiveSpoolBudgetError::FilesystemCapacityUnavailable);
        }
        Ok(())
    }
}

/// Admission lane whose resources are independently fenced.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ArchiveBudgetClass {
    /// Ordinary source-frame projection.
    Ordinary,
    /// Loss/lifecycle work admitted from the reserved control lane.
    Control,
}

/// Immutable accounting and high-water health snapshot.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ArchiveSpoolBudgetSnapshot {
    /// Whether ordinary and control admission is permanently closed.
    pub closed: bool,
    /// Whether the protected finalization transaction has begun.
    pub finalizing: bool,
    /// Latest reconciled logical usage plus committed conservative growth.
    pub accounted_bytes: u64,
    /// Latest reconciled logical file usage plus committed conservative growth.
    pub accounted_files: u64,
    /// Current ordinary-class growth since the last observation.
    pub ordinary_growth_bytes: u64,
    /// Current ordinary-class file growth since the last observation.
    pub ordinary_growth_files: u64,
    /// Current control-class growth since the last observation.
    pub control_growth_bytes: u64,
    /// Current control-class file growth since the last observation.
    pub control_growth_files: u64,
    /// Outstanding ordinary leases occupying frame capacity.
    pub ordinary_frames: u64,
    /// Outstanding control leases occupying frame capacity.
    pub control_frames: u64,
    /// Number of leases not yet committed or released.
    pub outstanding_leases: u64,
    /// Bytes protected from ordinary admission.
    pub protected_reserve_bytes: u64,
    /// Files protected from ordinary admission.
    pub protected_reserve_files: u64,
    /// Bytes protected even from control admission.
    pub finalization_reserve_bytes: u64,
    /// Files protected even from control admission.
    pub finalization_reserve_files: u64,
    /// Highest accounted byte usage or reservation observed.
    pub high_water_bytes: u64,
    /// Highest accounted file usage or reservation observed.
    pub high_water_files: u64,
}

/// Finalization authorization proving ordinary/control admission is closed.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ArchiveFinalizationPermit {
    /// Exact reserve retained for the terminal transaction.
    pub reserve: ArchiveSpoolResources,
}

/// One live projection reservation released on drop unless committed.
pub struct ArchiveProjectionLease {
    permit: ArchiveProjectionPermit,
    class: ArchiveBudgetClass,
    authority: Option<Arc<dyn ArchiveSpoolBudgetAuthority>>,
}

impl Debug for ArchiveProjectionLease {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ArchiveProjectionLease")
            .field("permit", &self.permit)
            .field("class", &self.class)
            .field("settled", &self.authority.is_none())
            .finish()
    }
}

impl ArchiveProjectionLease {
    fn new(
        permit: ArchiveProjectionPermit,
        class: ArchiveBudgetClass,
        authority: Arc<dyn ArchiveSpoolBudgetAuthority>,
    ) -> Self {
        Self {
            permit,
            class,
            authority: Some(authority),
        }
    }

    /// Returns the semantic permit and exact conservative footprint.
    #[must_use]
    pub const fn permit(&self) -> &ArchiveProjectionPermit {
        &self.permit
    }

    /// Converts this live reservation into conservative durable growth.
    pub fn commit(mut self) {
        if let Some(authority) = self.authority.take() {
            authority.settle_projection(self.class, self.permit.footprint, true);
        }
    }
}

impl Drop for ArchiveProjectionLease {
    fn drop(&mut self) {
        if let Some(authority) = self.authority.take() {
            authority.settle_projection(self.class, self.permit.footprint, false);
        }
    }
}

/// Replaceable prepared authority for spool and queue resource admission.
pub trait ArchiveSpoolBudgetAuthority: Debug + Send + Sync {
    /// Nonblockingly reserves one ordinary or control projection transaction.
    fn try_reserve(
        self: Arc<Self>,
        policy: &dyn ArchiveAdmissionPolicy,
        class: ArchiveBudgetClass,
        upper_bound: ArchiveProjectionFootprint,
    ) -> Result<ArchiveProjectionLease, AdmissionRejection>;

    /// Reconciles a fresh blocking observation only at a no-lease fence.
    fn refresh(&self, observation: ArchiveSpoolObservation) -> Result<(), ArchiveSpoolBudgetError>;

    /// Permanently closes ordinary and control frame admission.
    fn close_admission(&self);

    /// Enters the one terminal transaction while retaining its protected reserve.
    fn begin_finalization(&self) -> Result<ArchiveFinalizationPermit, ArchiveSpoolBudgetError>;

    /// Returns current accounting and high-water health.
    fn snapshot(&self) -> ArchiveSpoolBudgetSnapshot;

    /// Settles an exact lease. Implementations must be idempotence-safe by ownership.
    #[doc(hidden)]
    fn settle_projection(
        &self,
        class: ArchiveBudgetClass,
        footprint: ArchiveProjectionFootprint,
        committed: bool,
    );
}

#[derive(Debug)]
struct AtomicBudgetState {
    epoch: AtomicU64,
    closed: AtomicBool,
    finalizing: AtomicBool,
    baseline_bytes: AtomicU64,
    baseline_files: AtomicU64,
    filesystem_available_bytes: AtomicU64,
    filesystem_available_files: AtomicU64,
    ordinary_growth_bytes: AtomicU64,
    ordinary_growth_files: AtomicU64,
    control_growth_bytes: AtomicU64,
    control_growth_files: AtomicU64,
    ordinary_frames: AtomicU64,
    control_frames: AtomicU64,
    outstanding_leases: AtomicU64,
    high_water_bytes: AtomicU64,
    high_water_files: AtomicU64,
}

/// Lock-free budget implementation prepared from one qualified-spool observation.
#[derive(Debug)]
pub struct AtomicArchiveSpoolBudget {
    limits: ArchiveSpoolBudgetLimits,
    protected_reserve: ArchiveSpoolResources,
    finalization_reserve: ArchiveSpoolResources,
    state: AtomicBudgetState,
}

impl AtomicArchiveSpoolBudget {
    /// Validates authored limits against both logical and filesystem capacity.
    pub fn new(
        limits: ArchiveSpoolBudgetLimits,
        observation: ArchiveSpoolObservation,
    ) -> Result<Arc<Self>, ArchiveSpoolBudgetError> {
        limits.validate()?;
        observation.validate()?;
        if observation.logical_bytes > limits.quota.bytes
            || observation.logical_files > limits.quota.files
        {
            return Err(ArchiveSpoolBudgetError::LogicalQuotaExceeded);
        }
        let finalization_reserve = limits.reserve.finalization_reserve()?;
        let protected_reserve = limits.reserve.protected_reserve()?;
        let byte_capacity = limits
            .quota
            .bytes
            .checked_sub(observation.logical_bytes)
            .expect("logical quota checked")
            .min(observation.filesystem_available_bytes);
        let file_capacity = limits
            .quota
            .files
            .checked_sub(observation.logical_files)
            .expect("logical quota checked")
            .min(observation.filesystem_available_files);
        if protected_reserve.bytes > byte_capacity || protected_reserve.files > file_capacity {
            return Err(ArchiveSpoolBudgetError::ProtectedReserveUnavailable);
        }
        Ok(Arc::new(Self {
            limits,
            protected_reserve,
            finalization_reserve,
            state: AtomicBudgetState {
                epoch: AtomicU64::new(0),
                closed: AtomicBool::new(false),
                finalizing: AtomicBool::new(false),
                baseline_bytes: AtomicU64::new(observation.logical_bytes),
                baseline_files: AtomicU64::new(observation.logical_files),
                filesystem_available_bytes: AtomicU64::new(observation.filesystem_available_bytes),
                filesystem_available_files: AtomicU64::new(observation.filesystem_available_files),
                ordinary_growth_bytes: AtomicU64::new(0),
                ordinary_growth_files: AtomicU64::new(0),
                control_growth_bytes: AtomicU64::new(0),
                control_growth_files: AtomicU64::new(0),
                ordinary_frames: AtomicU64::new(0),
                control_frames: AtomicU64::new(0),
                outstanding_leases: AtomicU64::new(0),
                high_water_bytes: AtomicU64::new(observation.logical_bytes),
                high_water_files: AtomicU64::new(observation.logical_files),
            },
        }))
    }

    fn capacity(&self) -> ArchiveSpoolResources {
        let quota_bytes = self
            .limits
            .quota
            .bytes
            .saturating_sub(self.state.baseline_bytes.load(Ordering::Acquire));
        let quota_files = self
            .limits
            .quota
            .files
            .saturating_sub(self.state.baseline_files.load(Ordering::Acquire));
        ArchiveSpoolResources {
            bytes: quota_bytes.min(
                self.state
                    .filesystem_available_bytes
                    .load(Ordering::Acquire),
            ),
            files: quota_files.min(
                self.state
                    .filesystem_available_files
                    .load(Ordering::Acquire),
            ),
        }
    }

    fn growth(&self) -> ArchiveSpoolResources {
        ArchiveSpoolResources {
            bytes: self
                .state
                .ordinary_growth_bytes
                .load(Ordering::Acquire)
                .saturating_add(self.state.control_growth_bytes.load(Ordering::Acquire)),
            files: self
                .state
                .ordinary_growth_files
                .load(Ordering::Acquire)
                .saturating_add(self.state.control_growth_files.load(Ordering::Acquire)),
        }
    }

    fn ingress_state(&self, class: ArchiveBudgetClass) -> ArchiveIngressState {
        let capacity = self.capacity();
        let growth = self.growth();
        let control_growth = ArchiveSpoolResources {
            bytes: self.state.control_growth_bytes.load(Ordering::Acquire),
            files: self.state.control_growth_files.load(Ordering::Acquire),
        };
        let (available_frames, protected) = match class {
            ArchiveBudgetClass::Ordinary => (
                self.limits
                    .ordinary_frame_capacity
                    .saturating_sub(self.state.ordinary_frames.load(Ordering::Acquire)),
                ArchiveSpoolResources {
                    bytes: self.finalization_reserve.bytes.saturating_add(
                        self.limits
                            .reserve
                            .control_lane
                            .bytes
                            .saturating_sub(control_growth.bytes),
                    ),
                    files: self.finalization_reserve.files.saturating_add(
                        self.limits
                            .reserve
                            .control_lane
                            .files
                            .saturating_sub(control_growth.files),
                    ),
                },
            ),
            ArchiveBudgetClass::Control => (
                self.limits
                    .control_frame_capacity
                    .saturating_sub(self.state.control_frames.load(Ordering::Acquire)),
                self.finalization_reserve,
            ),
        };
        ArchiveIngressState {
            closed: self.state.closed.load(Ordering::Acquire),
            available_bytes: capacity.bytes.saturating_sub(growth.bytes),
            available_frames,
            available_files: capacity.files.saturating_sub(growth.files),
            protected_reserve_bytes: protected.bytes,
            protected_reserve_files: protected.files,
        }
    }

    fn class_limits(&self, class: ArchiveBudgetClass) -> (ArchiveSpoolResources, u64) {
        match class {
            ArchiveBudgetClass::Ordinary => {
                let capacity = self.capacity();
                (
                    ArchiveSpoolResources {
                        bytes: capacity.bytes.saturating_sub(self.protected_reserve.bytes),
                        files: capacity.files.saturating_sub(self.protected_reserve.files),
                    },
                    self.limits.ordinary_frame_capacity,
                )
            }
            ArchiveBudgetClass::Control => (
                self.limits.reserve.control_lane,
                self.limits.control_frame_capacity,
            ),
        }
    }

    fn class_counters(&self, class: ArchiveBudgetClass) -> (&AtomicU64, &AtomicU64, &AtomicU64) {
        match class {
            ArchiveBudgetClass::Ordinary => (
                &self.state.ordinary_growth_bytes,
                &self.state.ordinary_growth_files,
                &self.state.ordinary_frames,
            ),
            ArchiveBudgetClass::Control => (
                &self.state.control_growth_bytes,
                &self.state.control_growth_files,
                &self.state.control_frames,
            ),
        }
    }

    fn rollback(
        &self,
        class: ArchiveBudgetClass,
        footprint: ArchiveProjectionFootprint,
        bytes: bool,
        files: bool,
        frames: bool,
    ) {
        let (byte_counter, file_counter, frame_counter) = self.class_counters(class);
        if frames {
            frame_counter.fetch_sub(footprint.frames, Ordering::AcqRel);
        }
        if files {
            file_counter.fetch_sub(footprint.files, Ordering::AcqRel);
        }
        if bytes {
            byte_counter.fetch_sub(footprint.bytes, Ordering::AcqRel);
        }
        self.state.outstanding_leases.fetch_sub(1, Ordering::AcqRel);
    }

    fn update_high_water(&self) {
        let growth = self.growth();
        let bytes = self
            .state
            .baseline_bytes
            .load(Ordering::Acquire)
            .saturating_add(growth.bytes);
        let files = self
            .state
            .baseline_files
            .load(Ordering::Acquire)
            .saturating_add(growth.files);
        self.state
            .high_water_bytes
            .fetch_max(bytes, Ordering::AcqRel);
        self.state
            .high_water_files
            .fetch_max(files, Ordering::AcqRel);
    }
}

impl ArchiveSpoolBudgetAuthority for AtomicArchiveSpoolBudget {
    fn try_reserve(
        self: Arc<Self>,
        policy: &dyn ArchiveAdmissionPolicy,
        class: ArchiveBudgetClass,
        upper_bound: ArchiveProjectionFootprint,
    ) -> Result<ArchiveProjectionLease, AdmissionRejection> {
        if upper_bound.bytes == 0 || upper_bound.files == 0 || upper_bound.frames == 0 {
            return Err(AdmissionRejection::Capacity);
        }
        for _ in 0..MAX_RESERVATION_RETRIES {
            let epoch = self.state.epoch.load(Ordering::Acquire);
            if !epoch.is_multiple_of(2) {
                continue;
            }
            if self.state.closed.load(Ordering::Acquire) {
                return Err(AdmissionRejection::Closed);
            }
            self.state
                .outstanding_leases
                .fetch_update(Ordering::AcqRel, Ordering::Acquire, |value| {
                    value.checked_add(1)
                })
                .map_err(|_| AdmissionRejection::Capacity)?;
            if self.state.epoch.load(Ordering::Acquire) != epoch {
                self.state.outstanding_leases.fetch_sub(1, Ordering::AcqRel);
                continue;
            }
            let permit = match policy.try_reserve(self.ingress_state(class), upper_bound) {
                Ok(permit) => permit,
                Err(error) => {
                    self.state.outstanding_leases.fetch_sub(1, Ordering::AcqRel);
                    return Err(error);
                }
            };
            let (limits, frame_limit) = self.class_limits(class);
            let (byte_counter, file_counter, frame_counter) = self.class_counters(class);
            let bytes = try_add_with_limit(byte_counter, upper_bound.bytes, limits.bytes);
            if !bytes {
                self.state.outstanding_leases.fetch_sub(1, Ordering::AcqRel);
                continue;
            }
            let files = try_add_with_limit(file_counter, upper_bound.files, limits.files);
            if !files {
                self.rollback(class, upper_bound, true, false, false);
                continue;
            }
            let frames = try_add_with_limit(frame_counter, upper_bound.frames, frame_limit);
            if !frames {
                self.rollback(class, upper_bound, true, true, false);
                continue;
            }
            if self.state.epoch.load(Ordering::Acquire) != epoch
                || self.state.closed.load(Ordering::Acquire)
            {
                self.rollback(class, upper_bound, true, true, true);
                continue;
            }
            self.update_high_water();
            let authority: Arc<dyn ArchiveSpoolBudgetAuthority> = self;
            return Ok(ArchiveProjectionLease::new(permit, class, authority));
        }
        Err(AdmissionRejection::Capacity)
    }

    fn refresh(&self, observation: ArchiveSpoolObservation) -> Result<(), ArchiveSpoolBudgetError> {
        observation.validate()?;
        if self.state.finalizing.load(Ordering::Acquire) {
            return Err(ArchiveSpoolBudgetError::AlreadyFinalizing);
        }
        let epoch = self.state.epoch.load(Ordering::Acquire);
        let refreshing_epoch = epoch
            .checked_add(1)
            .ok_or(ArchiveSpoolBudgetError::ArithmeticOverflow)?;
        let next_epoch = epoch
            .checked_add(2)
            .ok_or(ArchiveSpoolBudgetError::ArithmeticOverflow)?;
        if !epoch.is_multiple_of(2)
            || self
                .state
                .epoch
                .compare_exchange(epoch, refreshing_epoch, Ordering::AcqRel, Ordering::Acquire)
                .is_err()
        {
            return Err(ArchiveSpoolBudgetError::RefreshBusy);
        }
        if self.state.outstanding_leases.load(Ordering::Acquire) != 0 {
            self.state.epoch.store(next_epoch, Ordering::Release);
            return Err(ArchiveSpoolBudgetError::RefreshBusy);
        }
        if observation.logical_bytes > self.limits.quota.bytes
            || observation.logical_files > self.limits.quota.files
        {
            self.state.epoch.store(next_epoch, Ordering::Release);
            return Err(ArchiveSpoolBudgetError::LogicalQuotaExceeded);
        }
        let byte_capacity = self
            .limits
            .quota
            .bytes
            .saturating_sub(observation.logical_bytes)
            .min(observation.filesystem_available_bytes);
        let file_capacity = self
            .limits
            .quota
            .files
            .saturating_sub(observation.logical_files)
            .min(observation.filesystem_available_files);
        if self.protected_reserve.bytes > byte_capacity
            || self.protected_reserve.files > file_capacity
        {
            self.state.epoch.store(next_epoch, Ordering::Release);
            return Err(ArchiveSpoolBudgetError::ProtectedReserveUnavailable);
        }
        self.state
            .baseline_bytes
            .store(observation.logical_bytes, Ordering::Release);
        self.state
            .baseline_files
            .store(observation.logical_files, Ordering::Release);
        self.state
            .filesystem_available_bytes
            .store(observation.filesystem_available_bytes, Ordering::Release);
        self.state
            .filesystem_available_files
            .store(observation.filesystem_available_files, Ordering::Release);
        self.state.ordinary_growth_bytes.store(0, Ordering::Release);
        self.state.ordinary_growth_files.store(0, Ordering::Release);
        self.state.control_growth_bytes.store(0, Ordering::Release);
        self.state.control_growth_files.store(0, Ordering::Release);
        self.state.epoch.store(next_epoch, Ordering::Release);
        self.update_high_water();
        Ok(())
    }

    fn close_admission(&self) {
        self.state.closed.store(true, Ordering::Release);
    }

    fn begin_finalization(&self) -> Result<ArchiveFinalizationPermit, ArchiveSpoolBudgetError> {
        self.close_admission();
        if self.state.outstanding_leases.load(Ordering::Acquire) != 0
            || !self.state.epoch.load(Ordering::Acquire).is_multiple_of(2)
        {
            return Err(ArchiveSpoolBudgetError::OutstandingReservations);
        }
        self.state
            .finalizing
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .map_err(|_| ArchiveSpoolBudgetError::AlreadyFinalizing)?;
        let capacity = self.capacity();
        let growth = self.growth();
        if self.finalization_reserve.bytes > capacity.bytes.saturating_sub(growth.bytes)
            || self.finalization_reserve.files > capacity.files.saturating_sub(growth.files)
        {
            self.state.finalizing.store(false, Ordering::Release);
            return Err(ArchiveSpoolBudgetError::FinalizationReserveUnavailable);
        }
        self.state.high_water_bytes.fetch_max(
            self.state
                .baseline_bytes
                .load(Ordering::Acquire)
                .saturating_add(growth.bytes)
                .saturating_add(self.finalization_reserve.bytes),
            Ordering::AcqRel,
        );
        self.state.high_water_files.fetch_max(
            self.state
                .baseline_files
                .load(Ordering::Acquire)
                .saturating_add(growth.files)
                .saturating_add(self.finalization_reserve.files),
            Ordering::AcqRel,
        );
        Ok(ArchiveFinalizationPermit {
            reserve: self.finalization_reserve,
        })
    }

    fn snapshot(&self) -> ArchiveSpoolBudgetSnapshot {
        let baseline_bytes = self.state.baseline_bytes.load(Ordering::Acquire);
        let baseline_files = self.state.baseline_files.load(Ordering::Acquire);
        let ordinary_growth_bytes = self.state.ordinary_growth_bytes.load(Ordering::Acquire);
        let ordinary_growth_files = self.state.ordinary_growth_files.load(Ordering::Acquire);
        let control_growth_bytes = self.state.control_growth_bytes.load(Ordering::Acquire);
        let control_growth_files = self.state.control_growth_files.load(Ordering::Acquire);
        ArchiveSpoolBudgetSnapshot {
            closed: self.state.closed.load(Ordering::Acquire),
            finalizing: self.state.finalizing.load(Ordering::Acquire),
            accounted_bytes: baseline_bytes
                .saturating_add(ordinary_growth_bytes)
                .saturating_add(control_growth_bytes),
            accounted_files: baseline_files
                .saturating_add(ordinary_growth_files)
                .saturating_add(control_growth_files),
            ordinary_growth_bytes,
            ordinary_growth_files,
            control_growth_bytes,
            control_growth_files,
            ordinary_frames: self.state.ordinary_frames.load(Ordering::Acquire),
            control_frames: self.state.control_frames.load(Ordering::Acquire),
            outstanding_leases: self.state.outstanding_leases.load(Ordering::Acquire),
            protected_reserve_bytes: self.protected_reserve.bytes,
            protected_reserve_files: self.protected_reserve.files,
            finalization_reserve_bytes: self.finalization_reserve.bytes,
            finalization_reserve_files: self.finalization_reserve.files,
            high_water_bytes: self.state.high_water_bytes.load(Ordering::Acquire),
            high_water_files: self.state.high_water_files.load(Ordering::Acquire),
        }
    }

    fn settle_projection(
        &self,
        class: ArchiveBudgetClass,
        footprint: ArchiveProjectionFootprint,
        committed: bool,
    ) {
        let (byte_counter, file_counter, frame_counter) = self.class_counters(class);
        frame_counter.fetch_sub(footprint.frames, Ordering::AcqRel);
        if !committed {
            byte_counter.fetch_sub(footprint.bytes, Ordering::AcqRel);
            file_counter.fetch_sub(footprint.files, Ordering::AcqRel);
        }
        self.state.outstanding_leases.fetch_sub(1, Ordering::AcqRel);
    }
}

fn try_add_with_limit(counter: &AtomicU64, amount: u64, limit: u64) -> bool {
    counter
        .fetch_update(Ordering::AcqRel, Ordering::Acquire, |current| {
            current.checked_add(amount).filter(|next| *next <= limit)
        })
        .is_ok()
}

/// Invalid quota/reserve configuration or fenced accounting transition.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ArchiveSpoolBudgetError {
    /// Authored byte/file quota is zero.
    InvalidQuota,
    /// At least one data/control queue capacity is zero.
    InvalidFrameCapacity,
    /// A mandatory named reserve component is empty or half-present.
    InvalidReserveComponent(&'static str),
    /// A resource sum exceeded `u64`.
    ArithmeticOverflow,
    /// The qualified filesystem reported no usable bytes or inodes.
    FilesystemCapacityUnavailable,
    /// Current logical usage is already beyond an authored quota.
    LogicalQuotaExceeded,
    /// Configured or physical headroom cannot retain all protected lanes.
    ProtectedReserveUnavailable,
    /// A refresh raced another refresh or an outstanding lease.
    RefreshBusy,
    /// Finalization cannot begin until every admitted lease settles.
    OutstandingReservations,
    /// The one terminal reserve has already been entered.
    AlreadyFinalizing,
    /// Fresh accounting no longer leaves the complete final reserve.
    FinalizationReserveUnavailable,
}

impl Display for ArchiveSpoolBudgetError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidQuota => formatter.write_str("archive spool quotas must be positive"),
            Self::InvalidFrameCapacity => {
                formatter.write_str("archive data/control frame capacities must be positive")
            }
            Self::InvalidReserveComponent(component) => {
                write!(
                    formatter,
                    "archive reserve component {component} is invalid"
                )
            }
            Self::ArithmeticOverflow => {
                formatter.write_str("archive spool resource arithmetic overflowed")
            }
            Self::FilesystemCapacityUnavailable => {
                formatter.write_str("archive spool filesystem has no available bytes or inodes")
            }
            Self::LogicalQuotaExceeded => {
                formatter.write_str("archive spool logical usage exceeds its configured quota")
            }
            Self::ProtectedReserveUnavailable => formatter
                .write_str("archive spool cannot preserve its control/finalization reserve"),
            Self::RefreshBusy => {
                formatter.write_str("archive spool accounting refresh requires a no-lease fence")
            }
            Self::OutstandingReservations => formatter
                .write_str("archive finalization requires every projection lease to settle"),
            Self::AlreadyFinalizing => {
                formatter.write_str("archive spool finalization already began")
            }
            Self::FinalizationReserveUnavailable => {
                formatter.write_str("archive finalization reserve is no longer fully available")
            }
        }
    }
}

impl std::error::Error for ArchiveSpoolBudgetError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AttachedBestEffortAdmissionPolicy, PrimaryWatchAdmissionPolicy};

    fn resources(bytes: u64, files: u64) -> ArchiveSpoolResources {
        ArchiveSpoolResources { bytes, files }
    }

    fn reserve_plan() -> ArchiveSpoolReservePlan {
        ArchiveSpoolReservePlan {
            largest_wal_frame: resources(10, 1),
            fallback_wal_window: resources(10, 1),
            open_parquet_builders: resources(10, 1),
            cow_index_path: resources(10, 1),
            generation_and_head: resources(10, 1),
            receipt_transaction: resources(10, 1),
            optional_raw_object: ArchiveSpoolResources::default(),
            wal_seal: resources(10, 1),
            emergency_finalization: resources(10, 1),
            control_lane: resources(20, 2),
        }
    }

    fn authority() -> Arc<AtomicArchiveSpoolBudget> {
        AtomicArchiveSpoolBudget::new(
            ArchiveSpoolBudgetLimits {
                quota: resources(220, 22),
                ordinary_frame_capacity: 2,
                control_frame_capacity: 1,
                reserve: reserve_plan(),
            },
            ArchiveSpoolObservation {
                logical_bytes: 20,
                logical_files: 2,
                filesystem_available_bytes: 500,
                filesystem_available_files: 50,
            },
        )
        .unwrap()
    }

    #[test]
    fn named_reserve_totals_and_overflow_are_fail_closed() {
        let plan = reserve_plan();
        assert_eq!(plan.finalization_reserve().unwrap(), resources(80, 8));
        assert_eq!(plan.protected_reserve().unwrap(), resources(100, 10));
        let mut invalid = plan;
        invalid.wal_seal = ArchiveSpoolResources::default();
        assert_eq!(
            invalid.validate(),
            Err(ArchiveSpoolBudgetError::InvalidReserveComponent("wal_seal"))
        );
        let mut overflow = plan;
        overflow.largest_wal_frame.bytes = u64::MAX;
        assert_eq!(
            overflow.finalization_reserve(),
            Err(ArchiveSpoolBudgetError::ArithmeticOverflow)
        );
    }

    #[test]
    fn dropped_leases_release_but_commits_retain_growth_and_release_frames() {
        let authority = authority();
        let footprint = ArchiveProjectionFootprint {
            bytes: 50,
            files: 5,
            frames: 1,
        };
        let lease = Arc::clone(&authority)
            .try_reserve(
                &PrimaryWatchAdmissionPolicy,
                ArchiveBudgetClass::Ordinary,
                footprint,
            )
            .unwrap();
        assert_eq!(lease.permit().footprint, footprint);
        assert_eq!(authority.snapshot().outstanding_leases, 1);
        drop(lease);
        assert_eq!(authority.snapshot().ordinary_growth_bytes, 0);
        let lease = Arc::clone(&authority)
            .try_reserve(
                &PrimaryWatchAdmissionPolicy,
                ArchiveBudgetClass::Ordinary,
                footprint,
            )
            .unwrap();
        lease.commit();
        let snapshot = authority.snapshot();
        assert_eq!(snapshot.ordinary_growth_bytes, 50);
        assert_eq!(snapshot.ordinary_growth_files, 5);
        assert_eq!(snapshot.ordinary_frames, 0);
        assert_eq!(snapshot.outstanding_leases, 0);
        assert_eq!(snapshot.accounted_bytes, 70);
    }

    #[test]
    fn ordinary_and_control_lanes_cannot_consume_each_others_reserve() {
        let authority = authority();
        let ordinary = ArchiveProjectionFootprint {
            bytes: 100,
            files: 10,
            frames: 1,
        };
        Arc::clone(&authority)
            .try_reserve(
                &AttachedBestEffortAdmissionPolicy,
                ArchiveBudgetClass::Ordinary,
                ordinary,
            )
            .unwrap()
            .commit();
        assert!(matches!(
            Arc::clone(&authority).try_reserve(
                &AttachedBestEffortAdmissionPolicy,
                ArchiveBudgetClass::Ordinary,
                ArchiveProjectionFootprint {
                    bytes: 1,
                    files: 1,
                    frames: 1,
                },
            ),
            Err(AdmissionRejection::Capacity)
        ));
        Arc::clone(&authority)
            .try_reserve(
                &AttachedBestEffortAdmissionPolicy,
                ArchiveBudgetClass::Control,
                ArchiveProjectionFootprint {
                    bytes: 20,
                    files: 2,
                    frames: 1,
                },
            )
            .unwrap()
            .commit();
        assert_eq!(authority.snapshot().control_growth_bytes, 20);
        assert!(matches!(
            Arc::clone(&authority).try_reserve(
                &AttachedBestEffortAdmissionPolicy,
                ArchiveBudgetClass::Control,
                ArchiveProjectionFootprint {
                    bytes: 1,
                    files: 1,
                    frames: 1,
                },
            ),
            Err(AdmissionRejection::Capacity)
        ));
    }

    #[test]
    fn refresh_requires_a_fence_and_finalization_closes_admission() {
        let authority = authority();
        let footprint = ArchiveProjectionFootprint {
            bytes: 10,
            files: 1,
            frames: 1,
        };
        let lease = Arc::clone(&authority)
            .try_reserve(
                &PrimaryWatchAdmissionPolicy,
                ArchiveBudgetClass::Ordinary,
                footprint,
            )
            .unwrap();
        let observation = ArchiveSpoolObservation {
            logical_bytes: 30,
            logical_files: 3,
            filesystem_available_bytes: 490,
            filesystem_available_files: 49,
        };
        assert_eq!(
            authority.refresh(observation),
            Err(ArchiveSpoolBudgetError::RefreshBusy)
        );
        assert_eq!(
            authority.begin_finalization(),
            Err(ArchiveSpoolBudgetError::OutstandingReservations)
        );
        drop(lease);
        authority.refresh(observation).unwrap();
        assert_eq!(authority.snapshot().accounted_bytes, 30);
        let permit = authority.begin_finalization().unwrap();
        assert_eq!(permit.reserve, resources(80, 8));
        assert!(authority.snapshot().closed);
        assert!(matches!(
            Arc::clone(&authority).try_reserve(
                &PrimaryWatchAdmissionPolicy,
                ArchiveBudgetClass::Ordinary,
                footprint,
            ),
            Err(AdmissionRejection::Closed)
        ));
        assert_eq!(
            authority.begin_finalization(),
            Err(ArchiveSpoolBudgetError::AlreadyFinalizing)
        );
    }

    #[test]
    fn filesystem_and_configured_limits_are_both_authoritative() {
        let limits = ArchiveSpoolBudgetLimits {
            quota: resources(1_000, 1_000),
            ordinary_frame_capacity: 1,
            control_frame_capacity: 1,
            reserve: reserve_plan(),
        };
        assert_eq!(
            AtomicArchiveSpoolBudget::new(
                limits,
                ArchiveSpoolObservation {
                    logical_bytes: 0,
                    logical_files: 0,
                    filesystem_available_bytes: 99,
                    filesystem_available_files: 100,
                },
            )
            .unwrap_err(),
            ArchiveSpoolBudgetError::ProtectedReserveUnavailable
        );
        assert_eq!(
            AtomicArchiveSpoolBudget::new(
                ArchiveSpoolBudgetLimits {
                    quota: resources(99, 9),
                    ..limits
                },
                ArchiveSpoolObservation {
                    logical_bytes: 0,
                    logical_files: 0,
                    filesystem_available_bytes: 10_000,
                    filesystem_available_files: 10_000,
                },
            )
            .unwrap_err(),
            ArchiveSpoolBudgetError::ProtectedReserveUnavailable
        );
    }

    #[test]
    fn concurrent_cas_admission_never_overcommits_the_ordinary_lane() {
        let authority = AtomicArchiveSpoolBudget::new(
            ArchiveSpoolBudgetLimits {
                quota: resources(220, 22),
                ordinary_frame_capacity: 20,
                control_frame_capacity: 1,
                reserve: reserve_plan(),
            },
            ArchiveSpoolObservation {
                logical_bytes: 20,
                logical_files: 2,
                filesystem_available_bytes: 500,
                filesystem_available_files: 50,
            },
        )
        .unwrap();
        let mut threads = Vec::new();
        for _ in 0..20 {
            let authority = Arc::clone(&authority);
            threads.push(std::thread::spawn(move || {
                let result = authority.try_reserve(
                    &AttachedBestEffortAdmissionPolicy,
                    ArchiveBudgetClass::Ordinary,
                    ArchiveProjectionFootprint {
                        bytes: 10,
                        files: 1,
                        frames: 1,
                    },
                );
                if let Ok(lease) = result {
                    lease.commit();
                    true
                } else {
                    false
                }
            }));
        }
        let accepted = threads
            .into_iter()
            .map(|thread| thread.join().unwrap())
            .filter(|accepted| *accepted)
            .count();
        assert_eq!(accepted, 10);
        let snapshot = authority.snapshot();
        assert_eq!(snapshot.ordinary_growth_bytes, 100);
        assert_eq!(snapshot.ordinary_growth_files, 10);
        assert_eq!(snapshot.outstanding_leases, 0);
    }
}
