// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Bounded compatibility-observation facts for externally driven episodes.

use std::fmt::{self, Display, Formatter};

use crate::eval::{ArtifactDigest, AttemptId, EvidenceEvent, EvidenceKind, append_identity_field};

use super::{NativeGraphPackagePlan, NativeGraphProfile};

/// The strongest observation classification supportable by a compatibility episode.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CaptureFidelity {
    /// Rust directly controlled every model and graph decision.
    NativeControlled,
    /// A bounded proxy observed declared HTTP(S) traffic without controlling it.
    ObservedProxy,
    /// Some declared calls were only partially observable.
    Partial,
    /// At least one declared call was unobservable or bypassed capture.
    Missing,
}

/// Fidelity that an externally driven terminal result may report.
///
/// This intentionally has no NativeGraph or exact variant: compatibility capture observes an
/// opaque driver but never owns its model or graph decisions.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CompatibilityFidelity {
    /// A bounded proxy observed declared HTTP(S) traffic without controlling it.
    ObservedProxy,
    /// Some declared calls were only partially observable.
    Partial,
    /// No call was observed or at least one call bypassed observation.
    Missing,
}

/// Immutable authority for one externally driven capture session.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct CompatibilityCaptureSession {
    package_identity: ArtifactDigest,
    source_identity: ArtifactDigest,
    task_identity: ArtifactDigest,
    environment_identity: ArtifactDigest,
    trial_digest: ArtifactDigest,
    attempt_id: AttemptId,
    identity_digest: ArtifactDigest,
}

impl CompatibilityCaptureSession {
    pub(crate) fn new(
        package_identity: ArtifactDigest,
        source_identity: ArtifactDigest,
        task_identity: ArtifactDigest,
        environment_identity: ArtifactDigest,
        trial_digest: ArtifactDigest,
        attempt_id: AttemptId,
    ) -> Self {
        let mut material = Vec::new();
        append_identity_field(
            &mut material,
            "domain",
            b"aiperf-native-graph-compatibility-capture-session-v1",
        );
        append_identity_field(
            &mut material,
            "package",
            package_identity.as_str().as_bytes(),
        );
        append_identity_field(&mut material, "source", source_identity.as_str().as_bytes());
        append_identity_field(&mut material, "task", task_identity.as_str().as_bytes());
        append_identity_field(
            &mut material,
            "environment",
            environment_identity.as_str().as_bytes(),
        );
        append_identity_field(&mut material, "trial", trial_digest.as_str().as_bytes());
        append_identity_field(&mut material, "attempt", attempt_id.as_str().as_bytes());
        Self {
            package_identity,
            source_identity,
            task_identity,
            environment_identity,
            trial_digest,
            attempt_id,
            identity_digest: ArtifactDigest::from_bytes(&material),
        }
    }

    #[allow(dead_code)]
    pub(crate) fn package_identity(&self) -> &ArtifactDigest {
        &self.package_identity
    }

    #[allow(dead_code)]
    pub(crate) fn identity_digest(&self) -> &ArtifactDigest {
        &self.identity_digest
    }
}

/// Opaque, bounded terminal acknowledgement supplied by one external driver session.
///
/// The receipt retains only a domain-separated digest. Its constructor accepts canonical terminal
/// bytes at the private protocol boundary and never preserves them in the evaluation contract.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CompatibilityTerminalReceipt {
    session: CompatibilityCaptureSession,
    identity_digest: ArtifactDigest,
}

impl CompatibilityTerminalReceipt {
    /// Maximum canonical terminal payload bytes accepted before the payload is discarded.
    pub const MAX_CANONICAL_BYTES: usize = 64 * 1024;

    /// Seals a bounded canonical terminal payload without retaining its contents.
    #[allow(dead_code)]
    pub(crate) fn from_canonical_terminal_bytes(
        session: CompatibilityCaptureSession,
        bytes: &[u8],
    ) -> Result<Self, CaptureError> {
        if bytes.len() > Self::MAX_CANONICAL_BYTES {
            return Err(CaptureError::TerminalReceiptLimitExceeded {
                limit: Self::MAX_CANONICAL_BYTES,
            });
        }
        let mut material = Vec::new();
        append_identity_field(
            &mut material,
            "domain",
            b"aiperf-native-graph-compatibility-terminal-receipt-v1",
        );
        append_identity_field(
            &mut material,
            "capture-session",
            session.identity_digest().as_str().as_bytes(),
        );
        append_identity_field(&mut material, "canonical-terminal", bytes);
        Ok(Self {
            session,
            identity_digest: ArtifactDigest::from_bytes(&material),
        })
    }

    /// Returns the opaque identity of the discarded canonical terminal receipt.
    pub fn identity_digest(&self) -> &ArtifactDigest {
        &self.identity_digest
    }

    pub(crate) fn session(&self) -> &CompatibilityCaptureSession {
        &self.session
    }
}

/// Immutable capture authority derived solely from an imported external package plan.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CapturePolicy {
    package_identity: ArtifactDigest,
    capture_session: Option<CompatibilityCaptureSession>,
}

impl CapturePolicy {
    /// Maximum number of digest-only call facts retained by one compatibility summary.
    pub const MAX_OBSERVATIONS: u16 = 1_024;

    /// Derives compatibility authority from an immutable externally driven package plan.
    pub fn from_package(package: &NativeGraphPackagePlan) -> Result<Self, CaptureError> {
        if package.profile() != NativeGraphProfile::ExternallyDriven {
            return Err(CaptureError::RequiresExternallyDrivenProfile);
        }
        let mut material = Vec::new();
        crate::eval::append_identity_field(
            &mut material,
            "domain",
            b"aiperf-native-graph-compatibility-capture-policy-v1",
        );
        package.append_identity_material(&mut material);
        Ok(Self {
            package_identity: ArtifactDigest::from_bytes(&material),
            capture_session: None,
        })
    }

    /// Returns the profile selected by this sealed compatibility authority.
    pub const fn profile(&self) -> NativeGraphProfile {
        NativeGraphProfile::ExternallyDriven
    }

    /// Returns the highest fidelity an externally driven package may claim.
    pub const fn fidelity_ceiling(&self) -> CaptureFidelity {
        CaptureFidelity::ObservedProxy
    }

    /// Returns the immutable package identity bound to this authority.
    pub fn package_identity(&self) -> &ArtifactDigest {
        &self.package_identity
    }

    /// Starts one fixed-capacity, digest-only compatibility observation summary.
    pub fn begin_observation(&self) -> CompatibilityObservation {
        CompatibilityObservation {
            policy: self.clone(),
            observed_https_calls: 0,
            partial_calls: 0,
            unobservable_or_bypassed_calls: 0,
            hasher: observation_hasher(&self.package_identity),
        }
    }

    #[allow(dead_code)]
    pub(crate) fn from_session(session: &CompatibilityCaptureSession) -> Self {
        Self {
            package_identity: session.package_identity().clone(),
            capture_session: Some(session.clone()),
        }
    }
}

/// Rust-owned mutable collector for a fixed-capacity compatibility summary.
pub struct CompatibilityObservation {
    policy: CapturePolicy,
    observed_https_calls: u16,
    partial_calls: u16,
    unobservable_or_bypassed_calls: u16,
    hasher: blake3::Hasher,
}

impl CompatibilityObservation {
    /// Appends one redacted immutable HTTP(S) exchange identity.
    pub fn record_observed_https(&mut self, exchange: ArtifactDigest) -> Result<(), CaptureError> {
        self.reserve_observation()?;
        self.observed_https_calls = self
            .observed_https_calls
            .checked_add(1)
            .ok_or(CaptureError::CounterOverflow)?;
        self.hasher.update(b"\x1eobserved-https=");
        self.hasher.update(exchange.as_str().as_bytes());
        Ok(())
    }

    /// Appends one call whose declared observation was only partial.
    pub fn record_partial_call(&mut self) -> Result<(), CaptureError> {
        self.reserve_observation()?;
        self.partial_calls = self
            .partial_calls
            .checked_add(1)
            .ok_or(CaptureError::CounterOverflow)?;
        self.hasher.update(b"\x1epartial");
        Ok(())
    }

    /// Appends one call that was unobservable or bypassed the compatibility capture path.
    pub fn record_unobservable_or_bypassed_call(&mut self) -> Result<(), CaptureError> {
        self.reserve_observation()?;
        self.unobservable_or_bypassed_calls = self
            .unobservable_or_bypassed_calls
            .checked_add(1)
            .ok_or(CaptureError::CounterOverflow)?;
        self.hasher.update(b"\x1eunobservable-or-bypassed");
        Ok(())
    }

    /// Freezes this bounded summary without retaining any raw call data.
    pub fn freeze(mut self) -> CompatibilityObservationReport {
        self.hasher.update(b"\x1eobserved-count=");
        self.hasher
            .update(self.observed_https_calls.to_le_bytes().as_slice());
        self.hasher.update(b"\x1epartial-count=");
        self.hasher
            .update(self.partial_calls.to_le_bytes().as_slice());
        self.hasher.update(b"\x1emissing-count=");
        self.hasher
            .update(self.unobservable_or_bypassed_calls.to_le_bytes().as_slice());
        let digest = ArtifactDigest::from_bytes(self.hasher.finalize().as_bytes());
        let has_no_observations = self.observed_https_calls == 0
            && self.partial_calls == 0
            && self.unobservable_or_bypassed_calls == 0;
        let fidelity = if has_no_observations || self.unobservable_or_bypassed_calls != 0 {
            CaptureFidelity::Missing
        } else if self.partial_calls != 0 {
            CaptureFidelity::Partial
        } else {
            self.policy.fidelity_ceiling()
        };
        CompatibilityObservationReport {
            package_identity: self.policy.package_identity,
            capture_session: self.policy.capture_session,
            fidelity,
            observed_https_calls: self.observed_https_calls,
            partial_calls: self.partial_calls,
            unobservable_or_bypassed_calls: self.unobservable_or_bypassed_calls,
            digest,
        }
    }

    fn reserve_observation(&self) -> Result<(), CaptureError> {
        let total = self
            .observed_https_calls
            .checked_add(self.partial_calls)
            .and_then(|total| total.checked_add(self.unobservable_or_bypassed_calls))
            .ok_or(CaptureError::CounterOverflow)?;
        if total >= CapturePolicy::MAX_OBSERVATIONS {
            return Err(CaptureError::ObservationLimitExceeded {
                limit: CapturePolicy::MAX_OBSERVATIONS,
            });
        }
        Ok(())
    }
}

/// Immutable bounded compatibility-observation facts for one external episode.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CompatibilityObservationReport {
    package_identity: ArtifactDigest,
    capture_session: Option<CompatibilityCaptureSession>,
    fidelity: CaptureFidelity,
    observed_https_calls: u16,
    partial_calls: u16,
    unobservable_or_bypassed_calls: u16,
    digest: ArtifactDigest,
}

impl CompatibilityObservationReport {
    /// Returns the immutable package identity bound to the report.
    pub fn package_identity(&self) -> &ArtifactDigest {
        &self.package_identity
    }

    /// Returns the compatibility observation classification without upgrading authority.
    pub const fn fidelity(&self) -> CaptureFidelity {
        self.fidelity
    }

    /// Returns the count of redacted HTTP(S) exchange identities observed by the proxy.
    pub const fn observed_https_calls(&self) -> u16 {
        self.observed_https_calls
    }

    /// Returns the count of calls with only partial observation.
    pub const fn partial_calls(&self) -> u16 {
        self.partial_calls
    }

    /// Returns the count of unobservable or capture-bypassing calls.
    pub const fn unobservable_or_bypassed_calls(&self) -> u16 {
        self.unobservable_or_bypassed_calls
    }

    /// Returns the digest of the bounded, redacted observation summary.
    pub fn digest(&self) -> &ArtifactDigest {
        &self.digest
    }

    /// Seals this capture-session-bound report for an externally driven terminal result.
    ///
    /// The conversion is deliberately one way: callers can retain or emit only the bounded
    /// report, never a raw capture or a NativeGraph/exact compatibility classification.
    #[allow(dead_code)]
    pub(crate) fn into_terminal_supplement(
        self,
        receipt: CompatibilityTerminalReceipt,
    ) -> Result<CompatibilityTerminalSupplement, CaptureError> {
        if self.capture_session.as_ref() != Some(receipt.session()) {
            return Err(CaptureError::CaptureSessionIdentityMismatch);
        }
        let fidelity = match self.fidelity {
            CaptureFidelity::ObservedProxy => CompatibilityFidelity::ObservedProxy,
            CaptureFidelity::Partial => CompatibilityFidelity::Partial,
            CaptureFidelity::Missing | CaptureFidelity::NativeControlled => {
                CompatibilityFidelity::Missing
            }
        };
        let mut material = Vec::new();
        append_identity_field(
            &mut material,
            "domain",
            b"aiperf-native-graph-compatibility-terminal-supplement-v1",
        );
        append_identity_field(&mut material, "report", self.digest.as_str().as_bytes());
        append_identity_field(
            &mut material,
            "terminal-receipt",
            receipt.identity_digest().as_str().as_bytes(),
        );
        Ok(CompatibilityTerminalSupplement {
            report: self,
            fidelity,
            receipt,
            digest: ArtifactDigest::from_bytes(&material),
        })
    }
}

/// Sealed bounded compatibility facts attachable only to an externally driven terminal result.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CompatibilityTerminalSupplement {
    report: CompatibilityObservationReport,
    fidelity: CompatibilityFidelity,
    receipt: CompatibilityTerminalReceipt,
    digest: ArtifactDigest,
}

impl CompatibilityTerminalSupplement {
    /// Returns the only fidelity classification an externally driven result may expose.
    pub const fn fidelity(&self) -> CompatibilityFidelity {
        self.fidelity
    }

    /// Borrows the immutable bounded observation report.
    pub fn report(&self) -> &CompatibilityObservationReport {
        &self.report
    }

    /// Returns the sealed bounded compatibility summary identity.
    pub fn digest(&self) -> &ArtifactDigest {
        &self.digest
    }

    /// Emits compatibility facts only as one ordered lifecycle event.
    pub fn lifecycle_evidence(
        &self,
        attempt: AttemptId,
        sequence: u64,
        parent: Option<ArtifactDigest>,
    ) -> EvidenceEvent {
        EvidenceEvent::new(
            attempt,
            sequence,
            EvidenceKind::Compatibility,
            self.digest.clone(),
            parent,
        )
    }

    pub(crate) fn session(&self) -> &CompatibilityCaptureSession {
        self.receipt.session()
    }
}

/// Failed compatibility-capture policy or bounded observation admission.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CaptureError {
    /// Compatibility observation is available only to the external profile.
    RequiresExternallyDrivenProfile,
    /// A bounded observation summary reached its fixed call-fact limit.
    ObservationLimitExceeded {
        /// Maximum allowed call facts.
        limit: u16,
    },
    /// A terminal receipt exceeded its fixed public admission bound.
    TerminalReceiptLimitExceeded {
        /// Maximum canonical terminal receipt bytes.
        limit: usize,
    },
    /// A terminal receipt did not belong to the frozen capture session.
    CaptureSessionIdentityMismatch,
    /// A fixed observation counter could no longer be represented.
    CounterOverflow,
}

impl Display for CaptureError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::RequiresExternallyDrivenProfile => {
                formatter.write_str("compatibility capture requires the externally_driven profile")
            }
            Self::ObservationLimitExceeded { limit } => {
                write!(
                    formatter,
                    "compatibility observation limit {limit} exceeded"
                )
            }
            Self::TerminalReceiptLimitExceeded { limit } => {
                write!(
                    formatter,
                    "compatibility terminal receipt limit {limit} exceeded"
                )
            }
            Self::CaptureSessionIdentityMismatch => {
                formatter.write_str("compatibility receipt does not match the capture session")
            }
            Self::CounterOverflow => {
                formatter.write_str("compatibility observation counter overflowed")
            }
        }
    }
}

impl std::error::Error for CaptureError {}

fn observation_hasher(package_identity: &ArtifactDigest) -> blake3::Hasher {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"aiperf-native-graph-compatibility-observation-v1");
    hasher.update(package_identity.as_str().as_bytes());
    hasher
}
