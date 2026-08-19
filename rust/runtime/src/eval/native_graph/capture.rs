// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Bounded compatibility-observation facts for externally driven episodes.

use std::fmt::{self, Display, Formatter};

use crate::eval::{ArtifactDigest, AttemptId, EvidenceEvent, EvidenceKind};

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

/// Opaque, bounded terminal acknowledgement supplied by one external driver session.
///
/// The receipt retains only a domain-separated digest. Its constructor accepts canonical terminal
/// bytes at the private protocol boundary and never preserves them in the evaluation contract.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CompatibilityTerminalReceipt {
    identity_digest: ArtifactDigest,
}

impl CompatibilityTerminalReceipt {
    /// Maximum canonical terminal payload bytes accepted before the payload is discarded.
    pub const MAX_CANONICAL_BYTES: usize = 64 * 1024;

    /// Seals a bounded canonical terminal payload without retaining its contents.
    pub fn from_canonical_terminal_bytes(bytes: &[u8]) -> Result<Self, CaptureError> {
        if bytes.len() > Self::MAX_CANONICAL_BYTES {
            return Err(CaptureError::TerminalReceiptLimitExceeded {
                limit: Self::MAX_CANONICAL_BYTES,
            });
        }
        let mut material = Vec::new();
        crate::eval::append_identity_field(
            &mut material,
            "domain",
            b"aiperf-native-graph-compatibility-terminal-receipt-v1",
        );
        crate::eval::append_identity_field(&mut material, "canonical-terminal", bytes);
        Ok(Self {
            identity_digest: ArtifactDigest::from_bytes(&material),
        })
    }

    /// Returns the opaque identity of the discarded canonical terminal receipt.
    pub fn identity_digest(&self) -> &ArtifactDigest {
        &self.identity_digest
    }
}

/// Immutable capture authority derived solely from an imported external package plan.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CapturePolicy {
    package_identity: ArtifactDigest,
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

    /// Seals this package-bound report for an externally driven terminal result.
    ///
    /// The conversion is deliberately one way: callers can retain or emit only the bounded
    /// report, never a raw capture or a NativeGraph/exact compatibility classification.
    pub fn into_terminal_supplement(self) -> CompatibilityTerminalSupplement {
        let fidelity = match self.fidelity {
            CaptureFidelity::ObservedProxy => CompatibilityFidelity::ObservedProxy,
            CaptureFidelity::Partial => CompatibilityFidelity::Partial,
            CaptureFidelity::Missing | CaptureFidelity::NativeControlled => {
                CompatibilityFidelity::Missing
            }
        };
        CompatibilityTerminalSupplement {
            report: self,
            fidelity,
        }
    }

    /// Emits the report only as one ordered lifecycle fact, never verifier input evidence.
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
}

/// Sealed bounded compatibility facts attachable only to an externally driven terminal result.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CompatibilityTerminalSupplement {
    report: CompatibilityObservationReport,
    fidelity: CompatibilityFidelity,
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
        self.report.digest()
    }

    /// Emits compatibility facts only as one ordered lifecycle event.
    pub fn lifecycle_evidence(
        &self,
        attempt: AttemptId,
        sequence: u64,
        parent: Option<ArtifactDigest>,
    ) -> EvidenceEvent {
        self.report.lifecycle_evidence(attempt, sequence, parent)
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
