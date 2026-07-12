// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Replaceable rotation, ingress-admission, and recovery policy seams.
//!
//! Policies make decisions from immutable snapshots. Resource ownership,
//! waiting, permit release, and durable state transitions remain with the
//! single archive owner, so a policy cannot mutate accounting behind it.

use std::fmt::{self, Debug, Display, Formatter};

use crate::sync::WriterClaimId;
use crate::{ArchiveId, ArchiveState, Digest, SessionId};

/// Current whole-frame state of one open WAL or physical partition segment.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct OpenSegmentState {
    /// Number of complete frames already accepted.
    pub frame_count: u64,
    /// Number of logical rows already accepted.
    pub logical_row_count: u64,
    /// Exact encoded or conservatively reserved bytes already accepted.
    pub byte_count: u64,
    /// Clock value captured when this segment opened.
    pub opened_at_ns: i64,
}

/// Whole-frame segment rotation extension point.
pub trait SegmentRotationPolicy: Debug + Send {
    /// Whether the owner must rotate before accepting another whole frame.
    fn should_rotate(&self, state: &OpenSegmentState, now_ns: i64) -> bool;
}

/// Rotates once any configured row/frame/byte/age target is reached.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BoundedSegmentRotationPolicy {
    maximum_frames: Option<u64>,
    maximum_rows: Option<u64>,
    maximum_bytes: Option<u64>,
    maximum_age_ns: Option<u64>,
}

impl BoundedSegmentRotationPolicy {
    /// Validates at least one positive whole-segment bound.
    pub fn new(
        maximum_frames: Option<u64>,
        maximum_rows: Option<u64>,
        maximum_bytes: Option<u64>,
        maximum_age_ns: Option<u64>,
    ) -> Result<Self, PolicyError> {
        let bounds = [maximum_frames, maximum_rows, maximum_bytes, maximum_age_ns];
        if bounds.iter().flatten().any(|value| *value == 0) || bounds.iter().all(Option::is_none) {
            return Err(PolicyError::InvalidRotationBounds);
        }
        Ok(Self {
            maximum_frames,
            maximum_rows,
            maximum_bytes,
            maximum_age_ns,
        })
    }
}

impl SegmentRotationPolicy for BoundedSegmentRotationPolicy {
    fn should_rotate(&self, state: &OpenSegmentState, now_ns: i64) -> bool {
        self.maximum_frames
            .is_some_and(|maximum| state.frame_count >= maximum)
            || self
                .maximum_rows
                .is_some_and(|maximum| state.logical_row_count >= maximum)
            || self
                .maximum_bytes
                .is_some_and(|maximum| state.byte_count >= maximum)
            || self.maximum_age_ns.is_some_and(|maximum| {
                now_ns
                    .checked_sub(state.opened_at_ns)
                    .and_then(|age| u64::try_from(age).ok())
                    .is_some_and(|age| age >= maximum)
            })
    }
}

/// Rotates when any independently prepared policy requests it.
#[derive(Debug)]
pub struct AnyRotationPolicy {
    policies: Vec<Box<dyn SegmentRotationPolicy>>,
}

impl AnyRotationPolicy {
    /// Requires at least one concrete policy.
    pub fn new(policies: Vec<Box<dyn SegmentRotationPolicy>>) -> Result<Self, PolicyError> {
        if policies.is_empty() {
            return Err(PolicyError::EmptyPolicySet);
        }
        Ok(Self { policies })
    }
}

impl SegmentRotationPolicy for AnyRotationPolicy {
    fn should_rotate(&self, state: &OpenSegmentState, now_ns: i64) -> bool {
        self.policies
            .iter()
            .any(|policy| policy.should_rotate(state, now_ns))
    }
}

/// Conservative owner-authored upper bound for one projection transaction.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ArchiveProjectionFootprint {
    /// WAL, partition, index, manifest, and temporary bytes reserved together.
    pub bytes: u64,
    /// Whole frames occupying the bounded reorder/owner queue.
    pub frames: u64,
    /// Files/inodes conservatively reserved by the transaction.
    pub files: u64,
}

/// Current owner accounting from which a nonblocking permit may be granted.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ArchiveIngressState {
    /// Whether frame admission has been permanently closed.
    pub closed: bool,
    /// Bytes available before the configured hard quota.
    pub available_bytes: u64,
    /// Queue/frame slots currently available.
    pub available_frames: u64,
    /// File/inode budget currently available.
    pub available_files: u64,
    /// Emergency/finalization bytes unavailable to normal admission.
    pub protected_reserve_bytes: u64,
    /// Emergency/finalization files unavailable to normal admission.
    pub protected_reserve_files: u64,
}

/// Admission semantics selected by the product mode.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ArchiveAdmissionMode {
    /// The archive is the primary product and rejection is run-fatal policy.
    PrimaryWatch,
    /// The archive is attached and rejection becomes explicit loss evidence.
    AttachedBestEffort,
}

/// Immutable owner-consumed reservation returned by an admission policy.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ArchiveProjectionPermit {
    /// Product semantics under which this permit was granted.
    pub mode: ArchiveAdmissionMode,
    /// Exact conservative footprint the owner must debit and later release.
    pub footprint: ArchiveProjectionFootprint,
}

/// Nonblocking archive-ingress admission extension point.
pub trait ArchiveAdmissionPolicy: Debug + Send + Sync {
    /// Attempts to reserve one complete worst-case projection footprint.
    fn try_reserve(
        &self,
        state: ArchiveIngressState,
        upper_bound: ArchiveProjectionFootprint,
    ) -> Result<ArchiveProjectionPermit, AdmissionRejection>;
}

/// Primary-watch fail-before-fetch admission semantics.
#[derive(Clone, Copy, Debug, Default)]
pub struct PrimaryWatchAdmissionPolicy;

impl ArchiveAdmissionPolicy for PrimaryWatchAdmissionPolicy {
    fn try_reserve(
        &self,
        state: ArchiveIngressState,
        upper_bound: ArchiveProjectionFootprint,
    ) -> Result<ArchiveProjectionPermit, AdmissionRejection> {
        reserve(state, upper_bound, ArchiveAdmissionMode::PrimaryWatch)
    }
}

/// Attached-mode visible-loss admission semantics.
#[derive(Clone, Copy, Debug, Default)]
pub struct AttachedBestEffortAdmissionPolicy;

impl ArchiveAdmissionPolicy for AttachedBestEffortAdmissionPolicy {
    fn try_reserve(
        &self,
        state: ArchiveIngressState,
        upper_bound: ArchiveProjectionFootprint,
    ) -> Result<ArchiveProjectionPermit, AdmissionRejection> {
        reserve(state, upper_bound, ArchiveAdmissionMode::AttachedBestEffort)
    }
}

fn reserve(
    state: ArchiveIngressState,
    upper_bound: ArchiveProjectionFootprint,
    mode: ArchiveAdmissionMode,
) -> Result<ArchiveProjectionPermit, AdmissionRejection> {
    if state.closed {
        return Err(AdmissionRejection::Closed);
    }
    let usable_bytes = state
        .available_bytes
        .checked_sub(state.protected_reserve_bytes)
        .ok_or(AdmissionRejection::ProtectedReserve)?;
    let usable_files = state
        .available_files
        .checked_sub(state.protected_reserve_files)
        .ok_or(AdmissionRejection::ProtectedReserve)?;
    if upper_bound.bytes > usable_bytes
        || upper_bound.frames > state.available_frames
        || upper_bound.files > usable_files
    {
        return Err(AdmissionRejection::Capacity);
    }
    Ok(ArchiveProjectionPermit {
        mode,
        footprint: upper_bound,
    })
}

/// Typed nonblocking admission rejection.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AdmissionRejection {
    /// The owner has closed frame admission.
    Closed,
    /// Normal capacity cannot cover the complete upper bound.
    Capacity,
    /// Current accounting is already inside the protected reserve.
    ProtectedReserve,
}

impl Display for AdmissionRejection {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Closed => "archive frame admission is closed",
            Self::Capacity => "archive projection capacity is unavailable",
            Self::ProtectedReserve => "archive finalization reserve is protected",
        })
    }
}

impl std::error::Error for AdmissionRejection {}

/// Verified local discovery state supplied to recovery policy.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum LocalArchiveState {
    /// No archive discovery pointer exists.
    Absent,
    /// One exact local head and persistent identity were verified.
    Verified {
        /// Archive identity.
        archive_id: ArchiveId,
        /// Persistent archive/writer identity digest.
        persistent_identity_digest: Digest,
        /// Current local head hash.
        head_hash: Digest,
        /// Current archive state.
        archive_state: ArchiveState,
        /// Session owning an open collection, when any.
        session_id: Option<SessionId>,
        /// Next owner record sequence after verified recovery.
        next_record_seq: u64,
    },
}

/// Verified remote discovery state supplied to recovery policy.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RemoteArchiveState {
    /// Archive identity.
    pub archive_id: ArchiveId,
    /// Persistent archive/writer identity digest.
    pub persistent_identity_digest: Digest,
    /// Current remote head hash.
    pub head_hash: Digest,
    /// Whether remote head is a verified ancestor of the local head.
    pub is_local_ancestor: bool,
    /// Whether local head is a verified ancestor of the remote head.
    pub is_remote_ancestor: bool,
}

/// Exact recovery action selected before any source or session activation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum RecoveryPlan {
    /// Author a new generation-zero archive.
    CreateNew,
    /// Resume the verified local authority.
    ResumeLocal {
        /// Verified local head hash.
        head_hash: Digest,
        /// First record sequence available to the new owner.
        next_record_seq: u64,
    },
    /// Resume locally, then advance an ancestor remote head.
    ResumeAndPublish {
        /// Verified local head hash to publish.
        local_head_hash: Digest,
        /// Exact remote version/head predecessor.
        remote_head_hash: Digest,
        /// First record sequence available to the new owner.
        next_record_seq: u64,
    },
}

/// Verified local/remote recovery decision extension point.
pub trait ArchiveRecoveryPolicy: Debug + Send {
    /// Selects one fail-closed recovery plan without mutating either authority.
    fn recover(
        &self,
        local: &LocalArchiveState,
        remote: Option<&RemoteArchiveState>,
    ) -> Result<RecoveryPlan, ArchiveRecoveryError>;

    /// Returns the explicitly authored crashed claim required for exact takeover.
    fn prior_writer_claim_id(&self) -> Option<WriterClaimId> {
        None
    }
}

/// Allows creation only when both local and remote discovery are absent.
#[derive(Clone, Copy, Debug, Default)]
pub struct CreateNewRecoveryPolicy;

impl ArchiveRecoveryPolicy for CreateNewRecoveryPolicy {
    fn recover(
        &self,
        local: &LocalArchiveState,
        remote: Option<&RemoteArchiveState>,
    ) -> Result<RecoveryPlan, ArchiveRecoveryError> {
        if !matches!(local, LocalArchiveState::Absent) {
            return Err(ArchiveRecoveryError::UnexpectedLocalArchive);
        }
        if remote.is_some() {
            return Err(ArchiveRecoveryError::UnexpectedRemoteArchive);
        }
        Ok(RecoveryPlan::CreateNew)
    }
}

/// Requires exact persistent identity and ancestor-compatible resume state.
#[derive(Clone, Copy, Debug)]
pub struct ExactResumeRecoveryPolicy {
    expected_archive_id: ArchiveId,
    expected_persistent_identity_digest: Digest,
    prior_writer_claim_id: WriterClaimId,
}

impl ExactResumeRecoveryPolicy {
    /// Binds invocation expectations before recovery begins.
    #[must_use]
    pub const fn new(
        expected_archive_id: ArchiveId,
        expected_persistent_identity_digest: Digest,
        prior_writer_claim_id: WriterClaimId,
    ) -> Self {
        Self {
            expected_archive_id,
            expected_persistent_identity_digest,
            prior_writer_claim_id,
        }
    }
}

impl ArchiveRecoveryPolicy for ExactResumeRecoveryPolicy {
    fn recover(
        &self,
        local: &LocalArchiveState,
        remote: Option<&RemoteArchiveState>,
    ) -> Result<RecoveryPlan, ArchiveRecoveryError> {
        let LocalArchiveState::Verified {
            archive_id,
            persistent_identity_digest,
            head_hash,
            next_record_seq,
            ..
        } = local
        else {
            return Err(ArchiveRecoveryError::MissingLocalArchive);
        };
        if *archive_id != self.expected_archive_id
            || *persistent_identity_digest != self.expected_persistent_identity_digest
        {
            return Err(ArchiveRecoveryError::IdentityMismatch);
        }
        let Some(remote) = remote else {
            return Ok(RecoveryPlan::ResumeLocal {
                head_hash: *head_hash,
                next_record_seq: *next_record_seq,
            });
        };
        if remote.archive_id != *archive_id
            || remote.persistent_identity_digest != *persistent_identity_digest
        {
            return Err(ArchiveRecoveryError::IdentityMismatch);
        }
        if remote.head_hash == *head_hash {
            return Ok(RecoveryPlan::ResumeLocal {
                head_hash: *head_hash,
                next_record_seq: *next_record_seq,
            });
        }
        if remote.is_local_ancestor && !remote.is_remote_ancestor {
            return Ok(RecoveryPlan::ResumeAndPublish {
                local_head_hash: *head_hash,
                remote_head_hash: remote.head_hash,
                next_record_seq: *next_record_seq,
            });
        }
        Err(ArchiveRecoveryError::DivergentHeads)
    }

    fn prior_writer_claim_id(&self) -> Option<WriterClaimId> {
        Some(self.prior_writer_claim_id)
    }
}

/// Invalid policy configuration shared by concrete policy implementations.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PolicyError {
    /// Rotation has no bound or contains a zero bound.
    InvalidRotationBounds,
    /// Composite policy has no implementation.
    EmptyPolicySet,
}

impl Display for PolicyError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::InvalidRotationBounds => "segment rotation requires positive bounds",
            Self::EmptyPolicySet => "composite policy requires at least one implementation",
        })
    }
}

impl std::error::Error for PolicyError {}

/// Fail-closed recovery-policy error.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ArchiveRecoveryError {
    /// Create-new found an existing local archive.
    UnexpectedLocalArchive,
    /// Create-new found an existing remote archive.
    UnexpectedRemoteArchive,
    /// Exact resume requires verified local state.
    MissingLocalArchive,
    /// Archive or persistent writer identity differs.
    IdentityMismatch,
    /// Neither verified head is an ancestor of the other.
    DivergentHeads,
}

impl Display for ArchiveRecoveryError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::UnexpectedLocalArchive => "create-new found an existing local archive",
            Self::UnexpectedRemoteArchive => "create-new found an existing remote archive",
            Self::MissingLocalArchive => "exact resume requires a verified local archive",
            Self::IdentityMismatch => "archive persistent identity mismatch",
            Self::DivergentHeads => "local and remote archive heads diverged",
        })
    }
}

impl std::error::Error for ArchiveRecoveryError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn archive() -> ArchiveId {
        ArchiveId::new([1; 16]).unwrap()
    }

    #[test]
    fn rotation_uses_whole_segment_targets_and_checked_age() {
        let policy = BoundedSegmentRotationPolicy::new(Some(3), None, Some(100), Some(20)).unwrap();
        let state = OpenSegmentState {
            frame_count: 2,
            logical_row_count: 9,
            byte_count: 99,
            opened_at_ns: 10,
        };
        assert!(!policy.should_rotate(&state, 29));
        assert!(policy.should_rotate(&state, 30));
        assert!(!policy.should_rotate(&state, i64::MIN));
    }

    #[test]
    fn admission_never_consumes_finalization_reserve() {
        let state = ArchiveIngressState {
            closed: false,
            available_bytes: 100,
            available_frames: 1,
            available_files: 5,
            protected_reserve_bytes: 40,
            protected_reserve_files: 2,
        };
        let exact = ArchiveProjectionFootprint {
            bytes: 60,
            frames: 1,
            files: 3,
        };
        let permit = PrimaryWatchAdmissionPolicy
            .try_reserve(state, exact)
            .unwrap();
        assert_eq!(permit.mode, ArchiveAdmissionMode::PrimaryWatch);
        assert_eq!(
            AttachedBestEffortAdmissionPolicy
                .try_reserve(state, ArchiveProjectionFootprint { bytes: 61, ..exact }),
            Err(AdmissionRejection::Capacity)
        );
        assert_eq!(
            PrimaryWatchAdmissionPolicy.try_reserve(
                ArchiveIngressState {
                    closed: true,
                    ..state
                },
                exact
            ),
            Err(AdmissionRejection::Closed)
        );
    }

    #[test]
    fn recovery_rejects_identity_and_head_divergence_before_activation() {
        let identity = Digest::from_bytes([2; 32]);
        let head = Digest::from_bytes([3; 32]);
        let local = LocalArchiveState::Verified {
            archive_id: archive(),
            persistent_identity_digest: identity,
            head_hash: head,
            archive_state: ArchiveState::Open,
            session_id: None,
            next_record_seq: 9,
        };
        let prior_claim = WriterClaimId::from_digest(Digest::from_bytes([5; 32]));
        let policy = ExactResumeRecoveryPolicy::new(archive(), identity, prior_claim);
        assert_eq!(policy.prior_writer_claim_id(), Some(prior_claim));
        assert_eq!(
            policy.recover(&local, None).unwrap(),
            RecoveryPlan::ResumeLocal {
                head_hash: head,
                next_record_seq: 9,
            }
        );
        let remote = RemoteArchiveState {
            archive_id: archive(),
            persistent_identity_digest: identity,
            head_hash: Digest::from_bytes([4; 32]),
            is_local_ancestor: true,
            is_remote_ancestor: false,
        };
        assert!(matches!(
            policy.recover(&local, Some(&remote)),
            Ok(RecoveryPlan::ResumeAndPublish { .. })
        ));
        let divergent = RemoteArchiveState {
            is_local_ancestor: false,
            ..remote
        };
        assert_eq!(
            policy.recover(&local, Some(&divergent)),
            Err(ArchiveRecoveryError::DivergentHeads)
        );
    }
}
