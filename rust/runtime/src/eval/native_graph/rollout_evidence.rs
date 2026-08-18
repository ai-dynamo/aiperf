// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

//! Immutable, path-free verifier evidence for one validated RL rollout.

use std::{
    collections::BTreeSet,
    fmt::{self, Display, Formatter},
    io::{self, Write},
};

use serde::{
    Deserialize, Serialize,
    de::{self, DeserializeSeed, MapAccess, SeqAccess, Visitor},
};

use crate::eval::{ArtifactDigest, AttemptId, EvidenceEvent, EvidenceKind, append_identity_field};

use super::{
    ArtifactError, ArtifactQuota, EnvironmentTransitionRecord, EpisodeArtifactStore,
    FrozenArtifact, FrozenArtifactManifest, FrozenArtifactReference, FrozenRolloutTrajectory,
    RlEvaluationLimits, RlEvaluationPolicy, RlRolloutError,
};

const DEFAULT_MAX_VERIFIER_DOCUMENT_BYTES: usize = 256 * 1024;
const DEFAULT_MAX_VERIFIER_STRING_BYTES: usize = 4 * 1024;
const ARTIFACT_DIGEST_BYTES: usize = "blake3:".len() + 64;

/// Bounded non-raw timing facts for the model calls that produced one live rollout.
///
/// This fact intentionally contains only checked counters and durations. It cannot retain a
/// prompt, observation, decision bytes, capture record, or live model capability.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NativeGraphLivePolicyCallEvidence {
    call_count: u64,
    first_token_count: u64,
    total_first_token_ns: u64,
    max_first_token_ns: u64,
}

impl NativeGraphLivePolicyCallEvidence {
    pub(crate) const fn new(
        call_count: u64,
        first_token_count: u64,
        total_first_token_ns: u64,
        max_first_token_ns: u64,
    ) -> Self {
        Self {
            call_count,
            first_token_count,
            total_first_token_ns,
            max_first_token_ns,
        }
    }

    /// Returns the exact number of policy calls accepted by the selected transport.
    pub const fn call_count(&self) -> u64 {
        self.call_count
    }

    /// Returns the number of calls that reached a first-token boundary.
    pub const fn first_token_count(&self) -> u64 {
        self.first_token_count
    }

    /// Returns the checked total arrival-to-first-token time in nanoseconds.
    pub const fn total_first_token_ns(&self) -> u64 {
        self.total_first_token_ns
    }

    /// Returns the maximum arrival-to-first-token time in nanoseconds.
    pub const fn max_first_token_ns(&self) -> u64 {
        self.max_first_token_ns
    }

    pub(crate) fn identity_digest(&self) -> ArtifactDigest {
        let mut material = Vec::new();
        append_identity_field(
            &mut material,
            "domain",
            b"aiperf-native-graph-live-policy-calls-v1",
        );
        append_identity_field(&mut material, "calls", &self.call_count.to_le_bytes());
        append_identity_field(
            &mut material,
            "first-token-calls",
            &self.first_token_count.to_le_bytes(),
        );
        append_identity_field(
            &mut material,
            "total-first-token-ns",
            &self.total_first_token_ns.to_le_bytes(),
        );
        append_identity_field(
            &mut material,
            "max-first-token-ns",
            &self.max_first_token_ns.to_le_bytes(),
        );
        ArtifactDigest::from_bytes(&material)
    }
}

/// Immutable resource and artifact limits selected for freezing and verifying one rollout.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RolloutEvidenceLimits {
    max_document_bytes: usize,
    max_string_bytes: usize,
    policy: RlEvaluationLimits,
    quota: ArtifactQuota,
}

impl RolloutEvidenceLimits {
    /// Creates bounded document, policy, and artifact admission for one rollout.
    pub fn new(
        max_document_bytes: usize,
        max_string_bytes: usize,
        policy: RlEvaluationLimits,
        quota: ArtifactQuota,
    ) -> Result<Self, RolloutEvidenceLimitsError> {
        if max_document_bytes == 0 || max_string_bytes < ARTIFACT_DIGEST_BYTES {
            return Err(RolloutEvidenceLimitsError::ZeroByteLimit);
        }
        if quota.max_artifacts == 0
            || quota.max_total_bytes == 0
            || quota.max_artifact_bytes == 0
            || quota.max_download_handles == 0
        {
            return Err(RolloutEvidenceLimitsError::InvalidArtifactQuota);
        }
        Ok(Self {
            max_document_bytes,
            max_string_bytes,
            policy,
            quota,
        })
    }

    /// Derives conservative verifier-document bounds from the selected artifact quota.
    pub fn from_artifact_quota(quota: ArtifactQuota) -> Result<Self, RolloutEvidenceLimitsError> {
        Self::new(
            DEFAULT_MAX_VERIFIER_DOCUMENT_BYTES,
            DEFAULT_MAX_VERIFIER_STRING_BYTES,
            RlEvaluationLimits::default(),
            quota,
        )
    }

    /// Returns the maximum source bytes accepted before JSON parsing starts.
    pub const fn max_document_bytes(&self) -> usize {
        self.max_document_bytes
    }

    fn admit_trajectory(
        &self,
        reset_observation: &FrozenArtifactReference,
        actions: &[FrozenArtifactReference],
        trajectory: &FrozenRolloutTrajectory,
        store_quota: ArtifactQuota,
    ) -> Result<(), RolloutAdmissionError> {
        self.admit_trajectory_artifacts(
            reset_observation.artifact(),
            actions.iter().map(FrozenArtifactReference::artifact),
            trajectory,
            store_quota,
        )
    }

    fn admit_descriptor_trajectory(
        &self,
        reset_observation: &FrozenArtifact,
        actions: &[FrozenArtifact],
        trajectory: &FrozenRolloutTrajectory,
        store_quota: ArtifactQuota,
    ) -> Result<(), RolloutAdmissionError> {
        self.admit_trajectory_artifacts(reset_observation, actions.iter(), trajectory, store_quota)
    }

    fn admit_trajectory_artifacts<'a>(
        &self,
        reset_observation: &FrozenArtifact,
        actions: impl ExactSizeIterator<Item = &'a FrozenArtifact>,
        trajectory: &FrozenRolloutTrajectory,
        store_quota: ArtifactQuota,
    ) -> Result<(), RolloutAdmissionError> {
        if trajectory.limits() != trajectory.policy().limits() || trajectory.limits() != self.policy
        {
            return Err(RolloutAdmissionError::PolicyLimitsMismatch);
        }
        let transition_count = trajectory.transitions().len();
        let transition_limit = usize::try_from(self.policy.max_horizon()).map_err(|_| {
            RolloutAdmissionError::TransitionLimitExceeded {
                requested: transition_count,
                limit: usize::MAX,
            }
        })?;
        if transition_count > transition_limit {
            return Err(RolloutAdmissionError::TransitionLimitExceeded {
                requested: transition_count,
                limit: transition_limit,
            });
        }
        let artifact_limit = self.quota.max_artifacts.min(store_quota.max_artifacts);
        let required_artifacts = transition_count
            .checked_mul(3)
            .and_then(|count| count.checked_add(1))
            .ok_or(RolloutAdmissionError::ArtifactCountOverflow)?;
        if required_artifacts > artifact_limit {
            return Err(RolloutAdmissionError::ArtifactLimitExceeded {
                requested: required_artifacts,
                limit: artifact_limit,
            });
        }
        let total_limit = self.quota.max_total_bytes.min(store_quota.max_total_bytes);
        let mut total_bytes = 0;
        self.admit_artifact(reset_observation, store_quota)?;
        admit_descriptor_total(&mut total_bytes, reset_observation, total_limit)?;
        for (action, transition) in actions.zip(trajectory.transitions()) {
            self.admit_artifact(action, store_quota)?;
            self.admit_artifact(transition.observation(), store_quota)?;
            self.admit_artifact(transition.info(), store_quota)?;
            admit_descriptor_total(&mut total_bytes, action, total_limit)?;
            admit_descriptor_total(&mut total_bytes, transition.observation(), total_limit)?;
            admit_descriptor_total(&mut total_bytes, transition.info(), total_limit)?;
        }
        Ok(())
    }

    fn admit_artifact(
        &self,
        artifact: &FrozenArtifact,
        store_quota: ArtifactQuota,
    ) -> Result<(), RolloutAdmissionError> {
        let limit = self
            .quota
            .max_artifact_bytes
            .min(store_quota.max_artifact_bytes);
        if artifact.length() > limit {
            return Err(RolloutAdmissionError::ArtifactBytesExceeded {
                requested: artifact.length(),
                limit,
            });
        }
        Ok(())
    }
}

fn admit_descriptor_total(
    total_bytes: &mut u64,
    artifact: &FrozenArtifact,
    limit: u64,
) -> Result<(), RolloutAdmissionError> {
    *total_bytes = total_bytes
        .checked_add(artifact.length())
        .ok_or(RolloutAdmissionError::ArtifactTotalBytesOverflow)?;
    if *total_bytes > limit {
        return Err(RolloutAdmissionError::ArtifactTotalBytesExceeded {
            requested: *total_bytes,
            limit,
        });
    }
    Ok(())
}

fn admit_canonical_verifier_document(
    verifier_input: &RolloutVerifierInput,
    limit: usize,
) -> Result<(), RolloutEvidenceError> {
    let mut counter = BoundedJsonCounter::new(limit);
    match serde_json::to_writer(&mut counter, verifier_input) {
        Ok(()) => Ok(()),
        Err(_) if counter.is_exceeded => {
            Err(RolloutEvidenceError::VerifierDocumentTooLarge { limit })
        }
        Err(error) => Err(RolloutEvidenceError::VerifierDocumentEncoding(
            error.to_string(),
        )),
    }
}

struct BoundedJsonCounter {
    limit: usize,
    length: usize,
    is_exceeded: bool,
}

impl BoundedJsonCounter {
    const fn new(limit: usize) -> Self {
        Self {
            limit,
            length: 0,
            is_exceeded: false,
        }
    }
}

impl Write for BoundedJsonCounter {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        let Some(length) = self.length.checked_add(bytes.len()) else {
            self.is_exceeded = true;
            return Err(io::Error::other(
                "canonical verifier document length overflow",
            ));
        };
        if length > self.limit {
            self.is_exceeded = true;
            return Err(io::Error::other(
                "canonical verifier document exceeds limit",
            ));
        }
        self.length = length;
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

fn preflight_json_string_tokens(
    document: &[u8],
    limit: usize,
) -> Result<(), RolloutVerifierDecodeError> {
    let mut index = 0;
    while index < document.len() {
        if document[index] != b'"' {
            index += 1;
            continue;
        }
        index += 1;
        let mut decoded_length = 0;
        while let Some(&byte) = document.get(index) {
            if byte == b'"' {
                index += 1;
                break;
            }
            if byte != b'\\' {
                preflight_decoded_string_bytes(&mut decoded_length, 1, limit)?;
                index += 1;
                continue;
            }
            index += 1;
            let Some(&escape) = document.get(index) else {
                return Ok(());
            };
            match escape {
                b'"' | b'\\' | b'/' | b'b' | b'f' | b'n' | b'r' | b't' => {
                    preflight_decoded_string_bytes(&mut decoded_length, 1, limit)?;
                    index += 1;
                }
                b'u' => {
                    index += 1;
                    let Some(unit) = decode_json_unicode_escape(document, &mut index) else {
                        return Ok(());
                    };
                    let decoded_bytes = if (0xd800..=0xdbff).contains(&unit) {
                        let next_escape = index;
                        if document.get(index) == Some(&b'\\')
                            && document.get(index + 1) == Some(&b'u')
                        {
                            let mut low_index = index + 2;
                            if let Some(low) = decode_json_unicode_escape(document, &mut low_index)
                                && (0xdc00..=0xdfff).contains(&low)
                            {
                                index = low_index;
                                4
                            } else {
                                index = next_escape;
                                3
                            }
                        } else {
                            3
                        }
                    } else if (0xdc00..=0xdfff).contains(&unit) {
                        3
                    } else if unit <= 0x7f {
                        1
                    } else if unit <= 0x7ff {
                        2
                    } else {
                        3
                    };
                    preflight_decoded_string_bytes(&mut decoded_length, decoded_bytes, limit)?;
                }
                _ => return Ok(()),
            }
        }
    }
    Ok(())
}

fn preflight_decoded_string_bytes(
    decoded_length: &mut usize,
    additional: usize,
    limit: usize,
) -> Result<(), RolloutVerifierDecodeError> {
    *decoded_length = decoded_length
        .checked_add(additional)
        .ok_or(RolloutVerifierDecodeError::StringTooLong { limit })?;
    if *decoded_length > limit {
        return Err(RolloutVerifierDecodeError::StringTooLong { limit });
    }
    Ok(())
}

fn decode_json_unicode_escape(document: &[u8], index: &mut usize) -> Option<u16> {
    let mut unit = 0u16;
    for _ in 0..4 {
        let digit = match *document.get(*index)? {
            b'0'..=b'9' => document[*index] - b'0',
            b'a'..=b'f' => document[*index] - b'a' + 10,
            b'A'..=b'F' => document[*index] - b'A' + 10,
            _ => return None,
        };
        unit = (unit << 4) | u16::from(digit);
        *index += 1;
    }
    Some(unit)
}

/// Immutable imported provenance and rollout selection for one rollout.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct RolloutEvidenceIdentity {
    source: ArtifactDigest,
    task: ArtifactDigest,
    environment_implementation: ArtifactDigest,
    rollout_selection_digest: ArtifactDigest,
}

impl RolloutEvidenceIdentity {
    /// Creates provenance bound to the imported source, task, and environment implementation.
    ///
    /// Callers that hold an imported rollout selection must bind its digest with
    /// [`Self::with_rollout_selection_digest`] before freezing evidence.
    pub fn new(
        source: ArtifactDigest,
        task: ArtifactDigest,
        environment_implementation: ArtifactDigest,
    ) -> Self {
        Self {
            source,
            task,
            environment_implementation,
            rollout_selection_digest: ArtifactDigest::from_bytes(
                b"aiperf-native-graph-unbound-rollout-selection-v1",
            ),
        }
    }

    /// Binds this provenance to the complete immutable imported rollout selection.
    pub fn with_rollout_selection_digest(mut self, digest: ArtifactDigest) -> Self {
        self.rollout_selection_digest = digest;
        self
    }

    /// Borrows the imported source digest.
    pub fn source(&self) -> &ArtifactDigest {
        &self.source
    }

    /// Borrows the resolved task digest.
    pub fn task(&self) -> &ArtifactDigest {
        &self.task
    }

    /// Borrows the trusted environment implementation digest.
    pub fn environment_implementation(&self) -> &ArtifactDigest {
        &self.environment_implementation
    }

    /// Borrows the digest of the full selected rollout environment and policy facts.
    pub fn rollout_selection_digest(&self) -> &ArtifactDigest {
        &self.rollout_selection_digest
    }
}

/// Immutable policy facts used to derive one rollout return.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct RolloutPolicyEvidence {
    environment: String,
    horizon: u32,
    gamma: f64,
    identity: ArtifactDigest,
}

impl RolloutPolicyEvidence {
    pub(crate) fn from_policy(policy: &RlEvaluationPolicy) -> Self {
        Self::from_imported(policy.environment(), policy.horizon(), policy.gamma())
    }

    /// Retains immutable package policy facts that the importer has already validated.
    pub(crate) fn from_imported(environment: &str, horizon: u32, gamma: f64) -> Self {
        let environment = environment.to_owned();
        Self {
            identity: policy_identity(&environment, horizon, gamma),
            environment,
            horizon,
            gamma,
        }
    }

    /// Borrows the trusted environment identity.
    pub fn environment(&self) -> &str {
        &self.environment
    }

    /// Returns the maximum number of retained transitions.
    pub const fn horizon(&self) -> u32 {
        self.horizon
    }

    /// Returns the finite discount factor used for return derivation.
    pub const fn gamma(&self) -> f64 {
        self.gamma
    }

    /// Borrows the immutable policy identity digest.
    pub fn identity(&self) -> &ArtifactDigest {
        &self.identity
    }
}

/// One immutable transition supplied to an independent verifier.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct RolloutTransitionEvidence {
    step: u32,
    action: FrozenArtifact,
    observation: FrozenArtifact,
    reward: f64,
    terminated: bool,
    truncated: bool,
    info: FrozenArtifact,
}

impl RolloutTransitionEvidence {
    /// Returns the zero-based environment step index.
    pub const fn step(&self) -> u32 {
        self.step
    }

    /// Borrows the immutable action descriptor after its capability was stripped.
    pub fn action(&self) -> &FrozenArtifact {
        &self.action
    }

    /// Borrows the immutable observation descriptor after its capability was stripped.
    pub fn observation(&self) -> &FrozenArtifact {
        &self.observation
    }

    /// Returns the finite environment-authoritative reward.
    pub const fn reward(&self) -> f64 {
        self.reward
    }

    /// Returns whether the step is environment-terminated.
    pub const fn is_terminated(&self) -> bool {
        self.terminated
    }

    /// Returns whether the step is environment-truncated.
    pub const fn is_truncated(&self) -> bool {
        self.truncated
    }

    /// Borrows the immutable diagnostic descriptor after its capability was stripped.
    pub fn info(&self) -> &FrozenArtifact {
        &self.info
    }
}

/// Authoritative return facts persisted after Rust derives them from transitions.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct RolloutReturns {
    undiscounted_return: f64,
    discounted_return: f64,
}

impl RolloutReturns {
    /// Returns the authoritative undiscounted return.
    pub const fn undiscounted_return(&self) -> f64 {
        self.undiscounted_return
    }

    /// Returns the authoritative discounted return.
    pub const fn discounted_return(&self) -> f64 {
        self.discounted_return
    }
}

/// Strict, digest-addressed rollout document an isolated verifier can inspect.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct RolloutVerifierInput {
    identity: RolloutEvidenceIdentity,
    policy: RolloutPolicyEvidence,
    reset_observation: FrozenArtifact,
    transitions: Vec<RolloutTransitionEvidence>,
    returns: RolloutReturns,
    manifest: FrozenArtifactManifest,
    evidence_digest: ArtifactDigest,
}

impl RolloutVerifierInput {
    /// Decodes one strict, resource-bounded verifier document before retaining its fields.
    pub fn decode_bounded(
        document: &[u8],
        limits: &RolloutEvidenceLimits,
    ) -> Result<Self, RolloutVerifierDecodeError> {
        if document.len() > limits.max_document_bytes {
            return Err(RolloutVerifierDecodeError::DocumentTooLarge {
                actual: document.len(),
                limit: limits.max_document_bytes,
            });
        }
        preflight_json_string_tokens(document, limits.max_string_bytes)?;
        let mut deserializer = serde_json::Deserializer::from_slice(document);
        let input = RolloutVerifierInputSeed { limits }
            .deserialize(&mut deserializer)
            .map_err(|error| RolloutVerifierDecodeError::InvalidDocument(error.to_string()))?;
        deserializer
            .end()
            .map_err(|error| RolloutVerifierDecodeError::InvalidDocument(error.to_string()))?;
        input
            .admit_document(limits)
            .map_err(RolloutVerifierDecodeError::Admission)?;
        input
            .verify_return_agreement()
            .map_err(RolloutVerifierDecodeError::Agreement)?;
        Ok(input)
    }
    /// Borrows the immutable source, task, and environment provenance.
    pub fn identity(&self) -> &RolloutEvidenceIdentity {
        &self.identity
    }

    /// Borrows the retained policy facts.
    pub fn policy(&self) -> &RolloutPolicyEvidence {
        &self.policy
    }

    /// Borrows the frozen initial observation descriptor.
    pub fn reset_observation(&self) -> &FrozenArtifact {
        &self.reset_observation
    }

    /// Borrows the ordered frozen transitions.
    pub fn transitions(&self) -> &[RolloutTransitionEvidence] {
        &self.transitions
    }

    /// Borrows Rust-derived return facts.
    pub fn returns(&self) -> &RolloutReturns {
        &self.returns
    }

    /// Borrows the canonical frozen-artifact manifest retained for verification.
    pub fn manifest(&self) -> &FrozenArtifactManifest {
        &self.manifest
    }

    /// Borrows the digest covering every verifier-visible rollout fact.
    pub fn evidence_digest(&self) -> &ArtifactDigest {
        &self.evidence_digest
    }

    /// Independently validates terminal shape, return arithmetic, and frozen provenance.
    pub fn verify_return_agreement(&self) -> Result<(), RolloutReturnAgreementError> {
        if self.policy.horizon == 0 {
            return Err(RolloutReturnAgreementError::InvalidHorizon);
        }
        if !self.policy.gamma.is_finite() || !(0.0..=1.0).contains(&self.policy.gamma) {
            return Err(RolloutReturnAgreementError::InvalidGamma);
        }
        if self.policy.identity
            != policy_identity(
                &self.policy.environment,
                self.policy.horizon,
                self.policy.gamma,
            )
        {
            return Err(RolloutReturnAgreementError::PolicyIdentityMismatch);
        }
        let returns = derive_returns(&self.policy, &self.transitions)?;
        if returns.undiscounted_return.to_bits() != self.returns.undiscounted_return.to_bits() {
            return Err(RolloutReturnAgreementError::UndiscountedReturnMismatch {
                expected: returns.undiscounted_return.to_bits(),
                actual: self.returns.undiscounted_return.to_bits(),
            });
        }
        if returns.discounted_return.to_bits() != self.returns.discounted_return.to_bits() {
            return Err(RolloutReturnAgreementError::DiscountedReturnMismatch {
                expected: returns.discounted_return.to_bits(),
                actual: self.returns.discounted_return.to_bits(),
            });
        }
        if self.referenced_artifacts() != self.manifest.artifacts().iter().cloned().collect() {
            return Err(RolloutReturnAgreementError::ManifestMismatch);
        }
        if self.identity_digest() != self.evidence_digest {
            return Err(RolloutReturnAgreementError::EvidenceDigestMismatch);
        }
        Ok(())
    }

    fn admit_document(&self, limits: &RolloutEvidenceLimits) -> Result<(), RolloutAdmissionError> {
        let policy = RlEvaluationPolicy::new_with_limits(
            &self.policy.environment,
            self.policy.horizon,
            self.policy.gamma,
            limits.policy,
        )
        .map_err(RolloutAdmissionError::Policy)?;
        let trajectory_count = self.transitions.len();
        let transition_limit = usize::try_from(policy.horizon()).map_err(|_| {
            RolloutAdmissionError::TransitionLimitExceeded {
                requested: trajectory_count,
                limit: usize::MAX,
            }
        })?;
        if trajectory_count > transition_limit {
            return Err(RolloutAdmissionError::TransitionLimitExceeded {
                requested: trajectory_count,
                limit: transition_limit,
            });
        }
        if self.manifest.artifacts().len() > limits.quota.max_artifacts {
            return Err(RolloutAdmissionError::ArtifactLimitExceeded {
                requested: self.manifest.artifacts().len(),
                limit: limits.quota.max_artifacts,
            });
        }
        for artifact in self.manifest.artifacts() {
            if artifact.length() > limits.quota.max_artifact_bytes {
                return Err(RolloutAdmissionError::ArtifactBytesExceeded {
                    requested: artifact.length(),
                    limit: limits.quota.max_artifact_bytes,
                });
            }
        }
        let mut total_bytes = 0;
        admit_descriptor_total(
            &mut total_bytes,
            &self.reset_observation,
            limits.quota.max_total_bytes,
        )?;
        for transition in &self.transitions {
            admit_descriptor_total(
                &mut total_bytes,
                &transition.action,
                limits.quota.max_total_bytes,
            )?;
            admit_descriptor_total(
                &mut total_bytes,
                &transition.observation,
                limits.quota.max_total_bytes,
            )?;
            admit_descriptor_total(
                &mut total_bytes,
                &transition.info,
                limits.quota.max_total_bytes,
            )?;
        }
        Ok(())
    }

    fn referenced_artifacts(&self) -> BTreeSet<FrozenArtifact> {
        let mut artifacts = BTreeSet::new();
        artifacts.insert(self.reset_observation.clone());
        for transition in &self.transitions {
            artifacts.insert(transition.action.clone());
            artifacts.insert(transition.observation.clone());
            artifacts.insert(transition.info.clone());
        }
        artifacts
    }

    fn identity_digest(&self) -> ArtifactDigest {
        let mut material = Vec::new();
        append_identity_field(
            &mut material,
            "domain",
            b"aiperf-native-graph-rollout-evidence-v1",
        );
        append_identity_field(
            &mut material,
            "source",
            self.identity.source.as_str().as_bytes(),
        );
        append_identity_field(
            &mut material,
            "task",
            self.identity.task.as_str().as_bytes(),
        );
        append_identity_field(
            &mut material,
            "environment-implementation",
            self.identity.environment_implementation.as_str().as_bytes(),
        );
        append_identity_field(
            &mut material,
            "rollout-selection",
            self.identity.rollout_selection_digest.as_str().as_bytes(),
        );
        append_identity_field(
            &mut material,
            "policy",
            self.policy.identity.as_str().as_bytes(),
        );
        append_artifact(&mut material, "reset-observation", &self.reset_observation);
        for transition in &self.transitions {
            append_identity_field(
                &mut material,
                "transition-step",
                &transition.step.to_le_bytes(),
            );
            append_artifact(&mut material, "transition-action", &transition.action);
            append_artifact(
                &mut material,
                "transition-observation",
                &transition.observation,
            );
            append_identity_field(
                &mut material,
                "transition-reward",
                &transition.reward.to_bits().to_le_bytes(),
            );
            append_identity_field(
                &mut material,
                "transition-terminated",
                &[u8::from(transition.terminated)],
            );
            append_identity_field(
                &mut material,
                "transition-truncated",
                &[u8::from(transition.truncated)],
            );
            append_artifact(&mut material, "transition-info", &transition.info);
        }
        append_identity_field(
            &mut material,
            "undiscounted-return",
            &self.returns.undiscounted_return.to_bits().to_le_bytes(),
        );
        append_identity_field(
            &mut material,
            "discounted-return",
            &self.returns.discounted_return.to_bits().to_le_bytes(),
        );
        for artifact in self.manifest.artifacts() {
            append_artifact(&mut material, "manifest-artifact", artifact);
        }
        ArtifactDigest::from_bytes(&material)
    }
}

/// Descriptor-only receipt retained by a NativeGraph callback during one live rollout.
///
/// The receipt deliberately contains only frozen artifact descriptors. Adapter download
/// capabilities are released before a reset or transition reaches this boundary.
#[derive(Clone, Debug, PartialEq)]
pub struct NativeGraphRolloutReceipt {
    policy: RlEvaluationPolicy,
    reset_observation: Option<FrozenArtifact>,
    transitions: Vec<NativeGraphRolloutTransitionReceipt>,
}

impl NativeGraphRolloutReceipt {
    /// Starts an empty receipt bound to the immutable policy selected by the imported package.
    pub fn new(policy: RlEvaluationPolicy) -> Self {
        Self {
            policy,
            reset_observation: None,
            transitions: Vec::new(),
        }
    }

    /// Retains the one reset observation after the adapter capability has been stripped.
    pub fn record_reset(
        &mut self,
        observation: FrozenArtifact,
    ) -> Result<(), NativeGraphRolloutReceiptError> {
        if self.reset_observation.is_some() {
            return Err(NativeGraphRolloutReceiptError::DuplicateReset);
        }
        self.reset_observation = Some(observation);
        Ok(())
    }

    /// Retains one action descriptor and its authoritative environment transition.
    pub fn record_transition(
        &mut self,
        action: FrozenArtifact,
        transition: EnvironmentTransitionRecord,
    ) -> Result<(), NativeGraphRolloutReceiptError> {
        if self.reset_observation.is_none() {
            return Err(NativeGraphRolloutReceiptError::MissingReset);
        }
        let horizon = usize::try_from(self.policy.horizon()).map_err(|_| {
            NativeGraphRolloutReceiptError::Trajectory(RlRolloutError::HorizonExceeded)
        })?;
        if self.transitions.len() >= horizon {
            return Err(NativeGraphRolloutReceiptError::Trajectory(
                RlRolloutError::HorizonExceeded,
            ));
        }
        if self
            .transitions
            .last()
            .is_some_and(|receipt| receipt.is_terminal())
        {
            return Err(NativeGraphRolloutReceiptError::Trajectory(
                RlRolloutError::PostTerminalStep,
            ));
        }
        let expected_step = u32::try_from(self.transitions.len()).map_err(|_| {
            NativeGraphRolloutReceiptError::Trajectory(RlRolloutError::InvalidStepOrder)
        })?;
        if transition.step() != expected_step {
            return Err(NativeGraphRolloutReceiptError::Trajectory(
                RlRolloutError::InvalidStepOrder,
            ));
        }
        self.transitions
            .push(NativeGraphRolloutTransitionReceipt { action, transition });
        Ok(())
    }

    /// Returns the number of retained transitions without exposing their artifact facts.
    pub const fn transition_count(&self) -> usize {
        self.transitions.len()
    }

    /// Admits only the authoritative current observation before a model decision can begin.
    pub fn admit_observation(
        &self,
        observation: &FrozenArtifact,
    ) -> Result<(), NativeGraphRolloutReceiptError> {
        let horizon = usize::try_from(self.policy.horizon()).map_err(|_| {
            NativeGraphRolloutReceiptError::Trajectory(RlRolloutError::HorizonExceeded)
        })?;
        if self.transitions.len() >= horizon {
            return Err(NativeGraphRolloutReceiptError::Trajectory(
                RlRolloutError::HorizonExceeded,
            ));
        }
        let expected = match self.transitions.last() {
            Some(transition) if transition.is_terminal() => {
                return Err(NativeGraphRolloutReceiptError::Terminal);
            }
            Some(transition) => transition.transition.observation(),
            None => self
                .reset_observation
                .as_ref()
                .ok_or(NativeGraphRolloutReceiptError::MissingReset)?,
        };
        if expected != observation {
            return Err(NativeGraphRolloutReceiptError::UnexpectedObservation);
        }
        Ok(())
    }

    /// Freezes the retained descriptors into verifier-isolated rollout evidence.
    pub fn freeze(
        self,
        identity: RolloutEvidenceIdentity,
        store: &EpisodeArtifactStore,
    ) -> Result<FrozenRolloutEvidence, NativeGraphRolloutReceiptError> {
        let reset_observation = self
            .reset_observation
            .ok_or(NativeGraphRolloutReceiptError::MissingReset)?;
        let mut actions = Vec::with_capacity(self.transitions.len());
        let mut transitions = Vec::with_capacity(self.transitions.len());
        for receipt in self.transitions {
            actions.push(receipt.action);
            transitions.push(receipt.transition);
        }
        let trajectory = self
            .policy
            .trajectory(transitions)
            .map_err(NativeGraphRolloutReceiptError::Trajectory)?;
        FrozenRolloutEvidence::freeze_descriptors(
            identity,
            reset_observation,
            &actions,
            trajectory,
            store,
        )
        .map_err(NativeGraphRolloutReceiptError::Evidence)
    }
}

/// One descriptor-only action and environment transition emitted by a started worker stepper.
#[derive(Clone, Debug, PartialEq)]
pub struct NativeGraphRolloutTransitionReceipt {
    action: FrozenArtifact,
    transition: EnvironmentTransitionRecord,
}

impl NativeGraphRolloutTransitionReceipt {
    pub(crate) fn from_stepper(
        action: FrozenArtifact,
        transition: EnvironmentTransitionRecord,
    ) -> Self {
        Self { action, transition }
    }

    /// Borrows the action descriptor after its single-use capability has been consumed.
    pub fn action(&self) -> &FrozenArtifact {
        &self.action
    }

    /// Borrows the authoritative transition after output capabilities have been released.
    pub fn transition(&self) -> &EnvironmentTransitionRecord {
        &self.transition
    }

    /// Returns whether the environment ended at this transition.
    pub const fn is_terminal(&self) -> bool {
        self.transition.is_terminated() || self.transition.is_truncated()
    }
}

/// Failure while retaining or freezing a descriptor-only callback rollout receipt.
#[derive(Debug)]
pub enum NativeGraphRolloutReceiptError {
    /// The callback tried to retain more than one reset observation.
    DuplicateReset,
    /// A transition or freeze was attempted before the reset observation.
    MissingReset,
    /// A model decision attempted to use a stale or unrelated artifact descriptor.
    UnexpectedObservation,
    /// A model decision attempted after an authoritative terminal transition.
    Terminal,
    /// The retained trajectory violated immutable rollout policy facts.
    Trajectory(RlRolloutError),
    /// Descriptor-only evidence could not be frozen against the trusted artifact store.
    Evidence(RolloutEvidenceError),
}

impl Display for NativeGraphRolloutReceiptError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::DuplicateReset => formatter.write_str("NativeGraph rollout receipt has a duplicate reset"),
            Self::MissingReset => formatter.write_str("NativeGraph rollout receipt is missing its reset observation"),
            Self::UnexpectedObservation => formatter.write_str(
                "NativeGraph rollout receipt observation does not match the authoritative current observation",
            ),
            Self::Terminal => formatter.write_str(
                "NativeGraph rollout receipt cannot admit a model decision after terminal transition",
            ),
            Self::Trajectory(error) => error.fmt(formatter),
            Self::Evidence(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for NativeGraphRolloutReceiptError {}

/// A rollout document frozen after Rust validated the trajectory and its returns.
#[derive(Clone, Debug, PartialEq)]
pub struct FrozenRolloutEvidence {
    verifier_input: RolloutVerifierInput,
    live_policy_calls: Option<NativeGraphLivePolicyCallEvidence>,
}

impl FrozenRolloutEvidence {
    /// Freezes trusted descriptors and return facts without retaining live child capabilities.
    pub fn freeze(
        identity: RolloutEvidenceIdentity,
        reset_observation: FrozenArtifactReference,
        actions: &[FrozenArtifactReference],
        trajectory: FrozenRolloutTrajectory,
        store: &EpisodeArtifactStore,
    ) -> Result<Self, RolloutEvidenceError> {
        let limits = RolloutEvidenceLimits::from_artifact_quota(store.quota())?;
        Self::freeze_with_limits(
            identity,
            reset_observation,
            actions,
            trajectory,
            &limits,
            store,
        )
    }

    /// Freezes one trajectory after selected policy and artifact limits admit it.
    pub fn freeze_with_limits(
        identity: RolloutEvidenceIdentity,
        reset_observation: FrozenArtifactReference,
        actions: &[FrozenArtifactReference],
        trajectory: FrozenRolloutTrajectory,
        limits: &RolloutEvidenceLimits,
        store: &EpisodeArtifactStore,
    ) -> Result<Self, RolloutEvidenceError> {
        let transition_count = trajectory.transitions().len();
        if actions.len() < transition_count {
            let step = u32::try_from(actions.len()).map_err(|_| {
                RolloutEvidenceError::Admission(RolloutAdmissionError::ArtifactCountOverflow)
            })?;
            return Err(RolloutEvidenceError::MissingAction { step });
        }
        if actions.len() > transition_count {
            return Err(RolloutEvidenceError::UnexpectedAction);
        }
        limits.admit_trajectory(&reset_observation, actions, &trajectory, store.quota())?;
        let actions = actions
            .iter()
            .map(|reference| reference.artifact().clone())
            .collect::<Vec<_>>();
        Self::freeze_descriptor_trajectory(
            identity,
            reset_observation.artifact().clone(),
            &actions,
            trajectory,
            limits,
            store,
        )
    }

    fn freeze_descriptors(
        identity: RolloutEvidenceIdentity,
        reset_observation: FrozenArtifact,
        actions: &[FrozenArtifact],
        trajectory: FrozenRolloutTrajectory,
        store: &EpisodeArtifactStore,
    ) -> Result<Self, RolloutEvidenceError> {
        let limits = RolloutEvidenceLimits::from_artifact_quota(store.quota())?;
        limits.admit_descriptor_trajectory(
            &reset_observation,
            actions,
            &trajectory,
            store.quota(),
        )?;
        Self::freeze_descriptor_trajectory(
            identity,
            reset_observation,
            actions,
            trajectory,
            &limits,
            store,
        )
    }

    fn freeze_descriptor_trajectory(
        identity: RolloutEvidenceIdentity,
        reset_observation: FrozenArtifact,
        actions: &[FrozenArtifact],
        trajectory: FrozenRolloutTrajectory,
        limits: &RolloutEvidenceLimits,
        store: &EpisodeArtifactStore,
    ) -> Result<Self, RolloutEvidenceError> {
        let transition_count = trajectory.transitions().len();
        if actions.len() < transition_count {
            let step = u32::try_from(actions.len()).map_err(|_| {
                RolloutEvidenceError::Admission(RolloutAdmissionError::ArtifactCountOverflow)
            })?;
            return Err(RolloutEvidenceError::MissingAction { step });
        }
        if actions.len() > transition_count {
            return Err(RolloutEvidenceError::UnexpectedAction);
        }
        let policy = RolloutPolicyEvidence::from_policy(trajectory.policy());
        let mut transitions = Vec::with_capacity(trajectory.transitions().len());
        for (action, transition) in actions.iter().zip(trajectory.transitions()) {
            transitions.push(RolloutTransitionEvidence::from_transition(
                action.clone(),
                transition,
            ));
        }
        let mut artifacts = BTreeSet::new();
        artifacts.insert(reset_observation.clone());
        for transition in &transitions {
            artifacts.insert(transition.action.clone());
            artifacts.insert(transition.observation.clone());
            artifacts.insert(transition.info.clone());
        }
        let mut verifier_input = RolloutVerifierInput {
            identity,
            policy,
            reset_observation,
            transitions,
            returns: RolloutReturns {
                undiscounted_return: trajectory.undiscounted_return(),
                discounted_return: trajectory.discounted_return(),
            },
            manifest: store.freeze_manifest(artifacts)?,
            evidence_digest: ArtifactDigest::from_bytes(b"uninitialized-rollout-evidence"),
        };
        verifier_input.evidence_digest = verifier_input.identity_digest();
        verifier_input.verify_return_agreement()?;
        admit_canonical_verifier_document(&verifier_input, limits.max_document_bytes)?;
        Ok(Self {
            verifier_input,
            live_policy_calls: None,
        })
    }

    /// Borrows the isolated verifier document.
    pub fn verifier_input(&self) -> &RolloutVerifierInput {
        &self.verifier_input
    }

    /// Returns the immutable verifier document digest.
    pub fn identity_digest(&self) -> ArtifactDigest {
        self.verifier_input.evidence_digest.clone()
    }

    /// Retains non-raw live-policy timing facts outside the verifier document.
    pub(crate) fn with_live_policy_calls(
        mut self,
        live_policy_calls: NativeGraphLivePolicyCallEvidence,
    ) -> Self {
        self.live_policy_calls = Some(live_policy_calls);
        self
    }

    /// Returns bounded non-raw selected-policy facts, when this evidence came from a live rollout.
    pub const fn live_policy_calls(&self) -> Option<NativeGraphLivePolicyCallEvidence> {
        self.live_policy_calls
    }

    /// Projects rollout identity into Task 3 lifecycle evidence without changing verifier inputs.
    pub fn lifecycle_evidence(
        &self,
        attempt: AttemptId,
        sequence: u64,
        parent: Option<ArtifactDigest>,
    ) -> EvidenceEvent {
        EvidenceEvent::new(
            attempt,
            sequence,
            EvidenceKind::Artifact,
            self.identity_digest(),
            parent,
        )
    }
}

impl RolloutTransitionEvidence {
    fn from_transition(action: FrozenArtifact, transition: &EnvironmentTransitionRecord) -> Self {
        Self {
            step: transition.step(),
            action,
            observation: transition.observation().clone(),
            reward: transition.reward(),
            terminated: transition.is_terminated(),
            truncated: transition.is_truncated(),
            info: transition.info().clone(),
        }
    }
}

struct RolloutVerifierInputSeed<'a> {
    limits: &'a RolloutEvidenceLimits,
}

impl<'de> DeserializeSeed<'de> for RolloutVerifierInputSeed<'_> {
    type Value = RolloutVerifierInput;

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: de::Deserializer<'de>,
    {
        deserializer.deserialize_map(RolloutVerifierInputVisitor {
            limits: self.limits,
        })
    }
}

struct RolloutVerifierInputVisitor<'a> {
    limits: &'a RolloutEvidenceLimits,
}

impl<'de> Visitor<'de> for RolloutVerifierInputVisitor<'_> {
    type Value = RolloutVerifierInput;

    fn expecting(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter.write_str("a bounded strict rollout verifier object")
    }

    fn visit_map<M>(self, mut map: M) -> Result<Self::Value, M::Error>
    where
        M: MapAccess<'de>,
    {
        let mut identity = None;
        let mut policy = None;
        let mut reset_observation = None;
        let mut transitions = None;
        let mut returns = None;
        let mut manifest = None;
        let mut evidence_digest = None;
        while let Some(field) = map.next_key()? {
            match field {
                RolloutVerifierField::Identity => set_once(
                    &mut identity,
                    map.next_value_seed(RolloutEvidenceIdentitySeed {
                        limits: self.limits,
                    })?,
                    "identity",
                )?,
                RolloutVerifierField::Policy => set_once(
                    &mut policy,
                    map.next_value_seed(RolloutPolicyEvidenceSeed {
                        limits: self.limits,
                    })?,
                    "policy",
                )?,
                RolloutVerifierField::ResetObservation => set_once(
                    &mut reset_observation,
                    map.next_value_seed(FrozenArtifactSeed {
                        limits: self.limits,
                    })?,
                    "reset_observation",
                )?,
                RolloutVerifierField::Transitions => set_once(
                    &mut transitions,
                    map.next_value_seed(TransitionSequenceSeed {
                        limits: self.limits,
                    })?,
                    "transitions",
                )?,
                RolloutVerifierField::Returns => set_once(
                    &mut returns,
                    map.next_value_seed(RolloutReturnsSeed)?,
                    "returns",
                )?,
                RolloutVerifierField::Manifest => set_once(
                    &mut manifest,
                    map.next_value_seed(FrozenArtifactManifestSeed {
                        limits: self.limits,
                    })?,
                    "manifest",
                )?,
                RolloutVerifierField::EvidenceDigest => set_once(
                    &mut evidence_digest,
                    map.next_value_seed(ArtifactDigestSeed {
                        limits: self.limits,
                    })?,
                    "evidence_digest",
                )?,
            }
        }
        Ok(RolloutVerifierInput {
            identity: identity.ok_or_else(|| de::Error::missing_field("identity"))?,
            policy: policy.ok_or_else(|| de::Error::missing_field("policy"))?,
            reset_observation: reset_observation
                .ok_or_else(|| de::Error::missing_field("reset_observation"))?,
            transitions: transitions.ok_or_else(|| de::Error::missing_field("transitions"))?,
            returns: returns.ok_or_else(|| de::Error::missing_field("returns"))?,
            manifest: manifest.ok_or_else(|| de::Error::missing_field("manifest"))?,
            evidence_digest: evidence_digest
                .ok_or_else(|| de::Error::missing_field("evidence_digest"))?,
        })
    }
}

struct RolloutEvidenceIdentitySeed<'a> {
    limits: &'a RolloutEvidenceLimits,
}

impl<'de> DeserializeSeed<'de> for RolloutEvidenceIdentitySeed<'_> {
    type Value = RolloutEvidenceIdentity;

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: de::Deserializer<'de>,
    {
        deserializer.deserialize_map(RolloutEvidenceIdentityVisitor {
            limits: self.limits,
        })
    }
}

struct RolloutEvidenceIdentityVisitor<'a> {
    limits: &'a RolloutEvidenceLimits,
}

impl<'de> Visitor<'de> for RolloutEvidenceIdentityVisitor<'_> {
    type Value = RolloutEvidenceIdentity;

    fn expecting(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter.write_str("a bounded rollout identity object")
    }

    fn visit_map<M>(self, mut map: M) -> Result<Self::Value, M::Error>
    where
        M: MapAccess<'de>,
    {
        let mut source = None;
        let mut task = None;
        let mut environment_implementation = None;
        let mut rollout_selection_digest = None;
        while let Some(field) = map.next_key()? {
            match field {
                RolloutIdentityField::Source => set_once(
                    &mut source,
                    map.next_value_seed(ArtifactDigestSeed {
                        limits: self.limits,
                    })?,
                    "source",
                )?,
                RolloutIdentityField::Task => set_once(
                    &mut task,
                    map.next_value_seed(ArtifactDigestSeed {
                        limits: self.limits,
                    })?,
                    "task",
                )?,
                RolloutIdentityField::EnvironmentImplementation => set_once(
                    &mut environment_implementation,
                    map.next_value_seed(ArtifactDigestSeed {
                        limits: self.limits,
                    })?,
                    "environment_implementation",
                )?,
                RolloutIdentityField::RolloutSelectionDigest => set_once(
                    &mut rollout_selection_digest,
                    map.next_value_seed(ArtifactDigestSeed {
                        limits: self.limits,
                    })?,
                    "rollout_selection_digest",
                )?,
            }
        }
        Ok(RolloutEvidenceIdentity::new(
            source.ok_or_else(|| de::Error::missing_field("source"))?,
            task.ok_or_else(|| de::Error::missing_field("task"))?,
            environment_implementation
                .ok_or_else(|| de::Error::missing_field("environment_implementation"))?,
        )
        .with_rollout_selection_digest(
            rollout_selection_digest
                .ok_or_else(|| de::Error::missing_field("rollout_selection_digest"))?,
        ))
    }
}

struct RolloutPolicyEvidenceSeed<'a> {
    limits: &'a RolloutEvidenceLimits,
}

impl<'de> DeserializeSeed<'de> for RolloutPolicyEvidenceSeed<'_> {
    type Value = RolloutPolicyEvidence;

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: de::Deserializer<'de>,
    {
        deserializer.deserialize_map(RolloutPolicyEvidenceVisitor {
            limits: self.limits,
        })
    }
}

struct RolloutPolicyEvidenceVisitor<'a> {
    limits: &'a RolloutEvidenceLimits,
}

impl<'de> Visitor<'de> for RolloutPolicyEvidenceVisitor<'_> {
    type Value = RolloutPolicyEvidence;

    fn expecting(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter.write_str("a bounded rollout policy object")
    }

    fn visit_map<M>(self, mut map: M) -> Result<Self::Value, M::Error>
    where
        M: MapAccess<'de>,
    {
        let mut environment = None;
        let mut horizon = None;
        let mut gamma = None;
        let mut identity = None;
        while let Some(field) = map.next_key()? {
            match field {
                RolloutPolicyField::Environment => set_once(
                    &mut environment,
                    map.next_value_seed(BoundedStringSeed {
                        limit: self
                            .limits
                            .max_string_bytes
                            .min(self.limits.policy.max_environment_bytes()),
                        field: "policy.environment",
                    })?,
                    "environment",
                )?,
                RolloutPolicyField::Horizon => {
                    set_once(&mut horizon, map.next_value()?, "horizon")?
                }
                RolloutPolicyField::Gamma => set_once(&mut gamma, map.next_value()?, "gamma")?,
                RolloutPolicyField::Identity => set_once(
                    &mut identity,
                    map.next_value_seed(ArtifactDigestSeed {
                        limits: self.limits,
                    })?,
                    "identity",
                )?,
            }
        }
        Ok(RolloutPolicyEvidence {
            environment: environment.ok_or_else(|| de::Error::missing_field("environment"))?,
            horizon: horizon.ok_or_else(|| de::Error::missing_field("horizon"))?,
            gamma: gamma.ok_or_else(|| de::Error::missing_field("gamma"))?,
            identity: identity.ok_or_else(|| de::Error::missing_field("identity"))?,
        })
    }
}

struct FrozenArtifactSeed<'a> {
    limits: &'a RolloutEvidenceLimits,
}

impl<'de> DeserializeSeed<'de> for FrozenArtifactSeed<'_> {
    type Value = FrozenArtifact;

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: de::Deserializer<'de>,
    {
        deserializer.deserialize_map(FrozenArtifactVisitor {
            limits: self.limits,
        })
    }
}

struct FrozenArtifactVisitor<'a> {
    limits: &'a RolloutEvidenceLimits,
}

impl<'de> Visitor<'de> for FrozenArtifactVisitor<'_> {
    type Value = FrozenArtifact;

    fn expecting(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter.write_str("a bounded frozen artifact descriptor")
    }

    fn visit_map<M>(self, mut map: M) -> Result<Self::Value, M::Error>
    where
        M: MapAccess<'de>,
    {
        let mut digest = None;
        let mut length = None;
        while let Some(field) = map.next_key()? {
            match field {
                FrozenArtifactField::Digest => set_once(
                    &mut digest,
                    map.next_value_seed(ArtifactDigestSeed {
                        limits: self.limits,
                    })?,
                    "digest",
                )?,
                FrozenArtifactField::Length => set_once(&mut length, map.next_value()?, "length")?,
            }
        }
        let length = length.ok_or_else(|| de::Error::missing_field("length"))?;
        if length > self.limits.quota.max_artifact_bytes {
            return Err(de::Error::custom(format!(
                "artifact length {length} exceeds selected limit {}",
                self.limits.quota.max_artifact_bytes
            )));
        }
        Ok(FrozenArtifact::from_descriptor(
            digest.ok_or_else(|| de::Error::missing_field("digest"))?,
            length,
        ))
    }
}

struct RolloutTransitionEvidenceSeed<'a> {
    limits: &'a RolloutEvidenceLimits,
}

impl<'de> DeserializeSeed<'de> for RolloutTransitionEvidenceSeed<'_> {
    type Value = RolloutTransitionEvidence;

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: de::Deserializer<'de>,
    {
        deserializer.deserialize_map(RolloutTransitionEvidenceVisitor {
            limits: self.limits,
        })
    }
}

struct RolloutTransitionEvidenceVisitor<'a> {
    limits: &'a RolloutEvidenceLimits,
}

impl<'de> Visitor<'de> for RolloutTransitionEvidenceVisitor<'_> {
    type Value = RolloutTransitionEvidence;

    fn expecting(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter.write_str("a bounded rollout transition object")
    }

    fn visit_map<M>(self, mut map: M) -> Result<Self::Value, M::Error>
    where
        M: MapAccess<'de>,
    {
        let mut step = None;
        let mut action = None;
        let mut observation = None;
        let mut reward = None;
        let mut terminated = None;
        let mut truncated = None;
        let mut info = None;
        while let Some(field) = map.next_key()? {
            match field {
                RolloutTransitionField::Step => set_once(&mut step, map.next_value()?, "step")?,
                RolloutTransitionField::Action => set_once(
                    &mut action,
                    map.next_value_seed(FrozenArtifactSeed {
                        limits: self.limits,
                    })?,
                    "action",
                )?,
                RolloutTransitionField::Observation => set_once(
                    &mut observation,
                    map.next_value_seed(FrozenArtifactSeed {
                        limits: self.limits,
                    })?,
                    "observation",
                )?,
                RolloutTransitionField::Reward => {
                    set_once(&mut reward, map.next_value()?, "reward")?
                }
                RolloutTransitionField::Terminated => {
                    set_once(&mut terminated, map.next_value()?, "terminated")?
                }
                RolloutTransitionField::Truncated => {
                    set_once(&mut truncated, map.next_value()?, "truncated")?
                }
                RolloutTransitionField::Info => set_once(
                    &mut info,
                    map.next_value_seed(FrozenArtifactSeed {
                        limits: self.limits,
                    })?,
                    "info",
                )?,
            }
        }
        Ok(RolloutTransitionEvidence {
            step: step.ok_or_else(|| de::Error::missing_field("step"))?,
            action: action.ok_or_else(|| de::Error::missing_field("action"))?,
            observation: observation.ok_or_else(|| de::Error::missing_field("observation"))?,
            reward: reward.ok_or_else(|| de::Error::missing_field("reward"))?,
            terminated: terminated.ok_or_else(|| de::Error::missing_field("terminated"))?,
            truncated: truncated.ok_or_else(|| de::Error::missing_field("truncated"))?,
            info: info.ok_or_else(|| de::Error::missing_field("info"))?,
        })
    }
}

struct RolloutReturnsSeed;

impl<'de> DeserializeSeed<'de> for RolloutReturnsSeed {
    type Value = RolloutReturns;

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: de::Deserializer<'de>,
    {
        deserializer.deserialize_map(RolloutReturnsVisitor)
    }
}

struct RolloutReturnsVisitor;

impl<'de> Visitor<'de> for RolloutReturnsVisitor {
    type Value = RolloutReturns;

    fn expecting(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter.write_str("a strict rollout return object")
    }

    fn visit_map<M>(self, mut map: M) -> Result<Self::Value, M::Error>
    where
        M: MapAccess<'de>,
    {
        let mut undiscounted_return = None;
        let mut discounted_return = None;
        while let Some(field) = map.next_key()? {
            match field {
                RolloutReturnsField::UndiscountedReturn => set_once(
                    &mut undiscounted_return,
                    map.next_value()?,
                    "undiscounted_return",
                )?,
                RolloutReturnsField::DiscountedReturn => set_once(
                    &mut discounted_return,
                    map.next_value()?,
                    "discounted_return",
                )?,
            }
        }
        Ok(RolloutReturns {
            undiscounted_return: undiscounted_return
                .ok_or_else(|| de::Error::missing_field("undiscounted_return"))?,
            discounted_return: discounted_return
                .ok_or_else(|| de::Error::missing_field("discounted_return"))?,
        })
    }
}

struct FrozenArtifactManifestSeed<'a> {
    limits: &'a RolloutEvidenceLimits,
}

impl<'de> DeserializeSeed<'de> for FrozenArtifactManifestSeed<'_> {
    type Value = FrozenArtifactManifest;

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: de::Deserializer<'de>,
    {
        deserializer.deserialize_map(FrozenArtifactManifestVisitor {
            limits: self.limits,
        })
    }
}

struct FrozenArtifactManifestVisitor<'a> {
    limits: &'a RolloutEvidenceLimits,
}

impl<'de> Visitor<'de> for FrozenArtifactManifestVisitor<'_> {
    type Value = FrozenArtifactManifest;

    fn expecting(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter.write_str("a bounded canonical frozen artifact manifest")
    }

    fn visit_map<M>(self, mut map: M) -> Result<Self::Value, M::Error>
    where
        M: MapAccess<'de>,
    {
        let mut artifacts = None;
        while let Some(field) = map.next_key()? {
            match field {
                FrozenArtifactManifestField::Artifacts => set_once(
                    &mut artifacts,
                    map.next_value_seed(FrozenArtifactSequenceSeed {
                        limits: self.limits,
                    })?,
                    "artifacts",
                )?,
            }
        }
        FrozenArtifactManifest::from_canonical_artifacts(
            artifacts.ok_or_else(|| de::Error::missing_field("artifacts"))?,
        )
        .ok_or_else(|| de::Error::custom("artifact manifest must be sorted and duplicate-free"))
    }
}

struct TransitionSequenceSeed<'a> {
    limits: &'a RolloutEvidenceLimits,
}

impl<'de> DeserializeSeed<'de> for TransitionSequenceSeed<'_> {
    type Value = Vec<RolloutTransitionEvidence>;

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: de::Deserializer<'de>,
    {
        deserializer.deserialize_seq(TransitionSequenceVisitor {
            limits: self.limits,
        })
    }
}

struct TransitionSequenceVisitor<'a> {
    limits: &'a RolloutEvidenceLimits,
}

impl<'de> Visitor<'de> for TransitionSequenceVisitor<'_> {
    type Value = Vec<RolloutTransitionEvidence>;

    fn expecting(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter.write_str("a bounded rollout transition array")
    }

    fn visit_seq<S>(self, mut sequence: S) -> Result<Self::Value, S::Error>
    where
        S: SeqAccess<'de>,
    {
        let limit = usize::try_from(self.limits.policy.max_horizon())
            .map_err(|_| de::Error::custom("selected transition limit does not fit usize"))?;
        let mut transitions = Vec::new();
        loop {
            if transitions.len() >= limit {
                if sequence.next_element::<de::IgnoredAny>()?.is_some() {
                    return Err(de::Error::custom(format!(
                        "transition count exceeds selected limit {limit}"
                    )));
                }
                return Ok(transitions);
            }
            let Some(transition) = sequence.next_element_seed(RolloutTransitionEvidenceSeed {
                limits: self.limits,
            })?
            else {
                return Ok(transitions);
            };
            transitions.push(transition);
        }
    }
}

struct FrozenArtifactSequenceSeed<'a> {
    limits: &'a RolloutEvidenceLimits,
}

impl<'de> DeserializeSeed<'de> for FrozenArtifactSequenceSeed<'_> {
    type Value = Vec<FrozenArtifact>;

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: de::Deserializer<'de>,
    {
        deserializer.deserialize_seq(FrozenArtifactSequenceVisitor {
            limits: self.limits,
        })
    }
}

struct FrozenArtifactSequenceVisitor<'a> {
    limits: &'a RolloutEvidenceLimits,
}

impl<'de> Visitor<'de> for FrozenArtifactSequenceVisitor<'_> {
    type Value = Vec<FrozenArtifact>;

    fn expecting(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter.write_str("a bounded frozen artifact array")
    }

    fn visit_seq<S>(self, mut sequence: S) -> Result<Self::Value, S::Error>
    where
        S: SeqAccess<'de>,
    {
        let mut artifacts = Vec::new();
        let mut total_bytes = 0_u64;
        loop {
            if artifacts.len() >= self.limits.quota.max_artifacts {
                if sequence.next_element::<de::IgnoredAny>()?.is_some() {
                    return Err(de::Error::custom(format!(
                        "artifact count exceeds selected limit {}",
                        self.limits.quota.max_artifacts
                    )));
                }
                return Ok(artifacts);
            }
            let Some(artifact) = sequence.next_element_seed(FrozenArtifactSeed {
                limits: self.limits,
            })?
            else {
                return Ok(artifacts);
            };
            total_bytes = total_bytes.checked_add(artifact.length()).ok_or_else(|| {
                de::Error::custom("artifact byte count overflows selected total limit")
            })?;
            if total_bytes > self.limits.quota.max_total_bytes {
                return Err(de::Error::custom(format!(
                    "artifact bytes exceed selected limit {}",
                    self.limits.quota.max_total_bytes
                )));
            }
            artifacts.push(artifact);
        }
    }
}

struct ArtifactDigestSeed<'a> {
    limits: &'a RolloutEvidenceLimits,
}

impl<'de> DeserializeSeed<'de> for ArtifactDigestSeed<'_> {
    type Value = ArtifactDigest;

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: de::Deserializer<'de>,
    {
        deserializer.deserialize_str(ArtifactDigestVisitor {
            limit: self.limits.max_string_bytes,
        })
    }
}

struct ArtifactDigestVisitor {
    limit: usize,
}

impl<'de> Visitor<'de> for ArtifactDigestVisitor {
    type Value = ArtifactDigest;

    fn expecting(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter.write_str("a bounded BLAKE3 digest string")
    }

    fn visit_borrowed_str<E>(self, value: &'de str) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        self.parse(value)
    }

    fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        self.parse(value)
    }

    fn visit_string<E>(self, value: String) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        self.parse(&value)
    }
}

impl ArtifactDigestVisitor {
    fn parse<E: de::Error>(&self, value: &str) -> Result<ArtifactDigest, E> {
        if value.len() > self.limit {
            return Err(de::Error::custom(format!(
                "digest is {} bytes above selected string limit {}",
                value.len(),
                self.limit
            )));
        }
        ArtifactDigest::parse(value.to_owned()).map_err(de::Error::custom)
    }
}

struct BoundedStringSeed {
    limit: usize,
    field: &'static str,
}

impl<'de> DeserializeSeed<'de> for BoundedStringSeed {
    type Value = String;

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: de::Deserializer<'de>,
    {
        deserializer.deserialize_str(BoundedStringVisitor {
            limit: self.limit,
            field: self.field,
        })
    }
}

struct BoundedStringVisitor {
    limit: usize,
    field: &'static str,
}

impl<'de> Visitor<'de> for BoundedStringVisitor {
    type Value = String;

    fn expecting(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        write!(formatter, "a bounded {} string", self.field)
    }

    fn visit_borrowed_str<E>(self, value: &'de str) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        self.copy(value)
    }

    fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        self.copy(value)
    }

    fn visit_string<E>(self, value: String) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        if value.len() > self.limit {
            return Err(self.too_long());
        }
        Ok(value)
    }
}

impl BoundedStringVisitor {
    fn copy<E: de::Error>(&self, value: &str) -> Result<String, E> {
        if value.len() > self.limit {
            return Err(self.too_long());
        }
        Ok(value.to_owned())
    }

    fn too_long<E: de::Error>(&self) -> E {
        de::Error::custom(format!(
            "{} exceeds selected string limit {}",
            self.field, self.limit
        ))
    }
}

fn set_once<T, E: de::Error>(slot: &mut Option<T>, value: T, field: &'static str) -> Result<(), E> {
    if slot.replace(value).is_some() {
        return Err(de::Error::duplicate_field(field));
    }
    Ok(())
}

#[derive(Deserialize)]
#[serde(field_identifier, rename_all = "snake_case")]
enum RolloutVerifierField {
    Identity,
    Policy,
    ResetObservation,
    Transitions,
    Returns,
    Manifest,
    EvidenceDigest,
}

#[derive(Deserialize)]
#[serde(field_identifier, rename_all = "snake_case")]
enum RolloutIdentityField {
    Source,
    Task,
    EnvironmentImplementation,
    RolloutSelectionDigest,
}

#[derive(Deserialize)]
#[serde(field_identifier, rename_all = "snake_case")]
enum RolloutPolicyField {
    Environment,
    Horizon,
    Gamma,
    Identity,
}

#[derive(Deserialize)]
#[serde(field_identifier, rename_all = "snake_case")]
enum FrozenArtifactField {
    Digest,
    Length,
}

#[derive(Deserialize)]
#[serde(field_identifier, rename_all = "snake_case")]
enum RolloutTransitionField {
    Step,
    Action,
    Observation,
    Reward,
    Terminated,
    Truncated,
    Info,
}

#[derive(Deserialize)]
#[serde(field_identifier, rename_all = "snake_case")]
enum RolloutReturnsField {
    UndiscountedReturn,
    DiscountedReturn,
}

#[derive(Deserialize)]
#[serde(field_identifier, rename_all = "snake_case")]
enum FrozenArtifactManifestField {
    Artifacts,
}

fn policy_identity(environment: &str, horizon: u32, gamma: f64) -> ArtifactDigest {
    let mut material = Vec::new();
    append_identity_field(
        &mut material,
        "domain",
        b"aiperf-native-graph-rollout-policy-v1",
    );
    append_identity_field(&mut material, "environment", environment.as_bytes());
    append_identity_field(&mut material, "horizon", &horizon.to_le_bytes());
    append_identity_field(&mut material, "gamma", &gamma.to_bits().to_le_bytes());
    ArtifactDigest::from_bytes(&material)
}

fn append_artifact(material: &mut Vec<u8>, tag: &str, artifact: &FrozenArtifact) {
    append_identity_field(material, tag, artifact.digest().as_str().as_bytes());
    append_identity_field(
        material,
        "artifact-length",
        &artifact.length().to_le_bytes(),
    );
}

fn derive_returns(
    policy: &RolloutPolicyEvidence,
    transitions: &[RolloutTransitionEvidence],
) -> Result<RolloutReturns, RolloutReturnAgreementError> {
    if transitions.is_empty() {
        return Err(RolloutReturnAgreementError::MissingTerminal);
    }
    if transitions.len() > policy.horizon as usize {
        return Err(RolloutReturnAgreementError::HorizonExceeded);
    }
    let mut undiscounted_return = 0.0;
    let mut discounted_return = 0.0;
    let mut discount_factor = 1.0;
    for (index, transition) in transitions.iter().enumerate() {
        let expected_step =
            u32::try_from(index).map_err(|_| RolloutReturnAgreementError::StepIndexOverflow)?;
        if transition.step != expected_step {
            return Err(RolloutReturnAgreementError::InvalidStepOrder {
                expected: expected_step,
                actual: transition.step,
            });
        }
        if !transition.reward.is_finite() {
            return Err(RolloutReturnAgreementError::NonFiniteReward {
                step: transition.step,
            });
        }
        if transition.terminated && transition.truncated {
            return Err(RolloutReturnAgreementError::AmbiguousTerminal {
                step: transition.step,
            });
        }
        if (transition.terminated || transition.truncated) && index + 1 != transitions.len() {
            return Err(RolloutReturnAgreementError::PostTerminalStep {
                step: transition.step,
            });
        }
        undiscounted_return += transition.reward;
        let weighted_reward = discount_factor * transition.reward;
        discounted_return += weighted_reward;
        if !undiscounted_return.is_finite()
            || !weighted_reward.is_finite()
            || !discounted_return.is_finite()
        {
            return Err(RolloutReturnAgreementError::NonFiniteReturn);
        }
        discount_factor *= policy.gamma;
        if !discount_factor.is_finite() {
            return Err(RolloutReturnAgreementError::NonFiniteReturn);
        }
    }
    let Some(terminal) = transitions.last() else {
        return Err(RolloutReturnAgreementError::MissingTerminal);
    };
    if !terminal.terminated && !terminal.truncated {
        return Err(RolloutReturnAgreementError::MissingTerminal);
    }
    Ok(RolloutReturns {
        undiscounted_return,
        discounted_return,
    })
}

/// Rejection of an unusable selected rollout-evidence limit set.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RolloutEvidenceLimitsError {
    /// A JSON document bound was zero or a string bound could not hold one digest.
    ZeroByteLimit,
    /// The selected artifact quota could not bound frozen verifier evidence.
    InvalidArtifactQuota,
}

impl Display for RolloutEvidenceLimitsError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroByteLimit => {
                formatter.write_str("rollout evidence byte limits must be positive")
            }
            Self::InvalidArtifactQuota => {
                formatter.write_str("rollout evidence requires a positive artifact quota")
            }
        }
    }
}

impl std::error::Error for RolloutEvidenceLimitsError {}

/// Rejection while admitting trajectory or verifier facts before retention.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum RolloutAdmissionError {
    /// The selected freeze limits differ from the limits that admitted the trajectory.
    PolicyLimitsMismatch,
    /// The retained policy exceeded selected resource bounds.
    Policy(RlRolloutError),
    /// The number of transitions exceeded the selected horizon.
    TransitionLimitExceeded {
        /// The supplied transition count.
        requested: usize,
        /// The selected maximum transition count.
        limit: usize,
    },
    /// The number of descriptor references overflowed while computing a bound.
    ArtifactCountOverflow,
    /// The descriptor count exceeded selected or store artifact capacity.
    ArtifactLimitExceeded {
        /// The required descriptor count.
        requested: usize,
        /// The selected descriptor maximum.
        limit: usize,
    },
    /// One descriptor exceeded selected or store byte capacity.
    ArtifactBytesExceeded {
        /// The descriptor byte length.
        requested: u64,
        /// The selected byte maximum.
        limit: u64,
    },
    /// Descriptor byte accounting overflowed while applying total capacity.
    ArtifactTotalBytesOverflow,
    /// The reset/action/observation/info descriptor bytes exceeded total capacity.
    ArtifactTotalBytesExceeded {
        /// The checked descriptor-byte total.
        requested: u64,
        /// The selected total byte maximum.
        limit: u64,
    },
}

impl Display for RolloutAdmissionError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::PolicyLimitsMismatch => formatter.write_str(
                "rollout trajectory was retained under different selected policy limits",
            ),
            Self::Policy(error) => error.fmt(formatter),
            Self::TransitionLimitExceeded { requested, limit } => write!(
                formatter,
                "rollout contains {requested} transitions above selected limit {limit}"
            ),
            Self::ArtifactCountOverflow => {
                formatter.write_str("rollout descriptor count overflowed selected limits")
            }
            Self::ArtifactLimitExceeded { requested, limit } => write!(
                formatter,
                "rollout requires {requested} artifacts above selected limit {limit}"
            ),
            Self::ArtifactBytesExceeded { requested, limit } => write!(
                formatter,
                "rollout artifact is {requested} bytes above selected limit {limit}"
            ),
            Self::ArtifactTotalBytesOverflow => {
                formatter.write_str("rollout descriptor bytes overflowed selected total capacity")
            }
            Self::ArtifactTotalBytesExceeded { requested, limit } => write!(
                formatter,
                "rollout descriptor bytes {requested} exceed selected total capacity {limit}"
            ),
        }
    }
}

impl std::error::Error for RolloutAdmissionError {}

/// Rejection while decoding one public bounded verifier document.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum RolloutVerifierDecodeError {
    /// Input bytes exceeded the selected document bound before JSON parsing.
    DocumentTooLarge {
        /// The supplied byte count.
        actual: usize,
        /// The selected byte maximum.
        limit: usize,
    },
    /// A lexical JSON string token cannot decode within the selected string bound.
    StringTooLong {
        /// The selected decoded-string byte maximum.
        limit: usize,
    },
    /// The strict document was malformed or exceeded a nested decode limit.
    InvalidDocument(String),
    /// The decoded policy or descriptors exceeded selected resource limits.
    Admission(RolloutAdmissionError),
    /// Independent return or provenance agreement failed after bounded decoding.
    Agreement(RolloutReturnAgreementError),
}

impl Display for RolloutVerifierDecodeError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::DocumentTooLarge { actual, limit } => write!(
                formatter,
                "rollout verifier document is {actual} bytes above selected limit {limit}"
            ),
            Self::StringTooLong { limit } => write!(
                formatter,
                "rollout verifier JSON string cannot decode within selected limit {limit}"
            ),
            Self::InvalidDocument(error) => {
                write!(formatter, "invalid rollout verifier document: {error}")
            }
            Self::Admission(error) => error.fmt(formatter),
            Self::Agreement(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for RolloutVerifierDecodeError {}

/// Failure while freezing path-free rollout evidence.
#[derive(Debug)]
pub enum RolloutEvidenceError {
    /// The selected evidence limits were not internally usable.
    Limits(RolloutEvidenceLimitsError),
    /// The selected policy or artifact limits rejected the trajectory before retention.
    Admission(RolloutAdmissionError),
    /// The trajectory did not have one action descriptor at this step.
    MissingAction {
        /// The step that lacked an action descriptor.
        step: u32,
    },
    /// More action descriptors were supplied than retained transitions.
    UnexpectedAction,
    /// Artifact freezing rejected an unknown or over-quota descriptor.
    Artifact(ArtifactError),
    /// Canonical verifier JSON exceeded the selected document limit before freezing completed.
    VerifierDocumentTooLarge {
        /// The selected canonical document byte maximum.
        limit: usize,
    },
    /// Canonical verifier JSON serialization failed for a reason other than the selected limit.
    VerifierDocumentEncoding(String),
    /// The frozen document did not pass independent agreement checks.
    Agreement(RolloutReturnAgreementError),
}

impl Display for RolloutEvidenceError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Limits(error) => error.fmt(formatter),
            Self::Admission(error) => error.fmt(formatter),
            Self::MissingAction { step } => {
                write!(formatter, "rollout step {step} lacks an action")
            }
            Self::UnexpectedAction => formatter.write_str("rollout has an unexpected action"),
            Self::Artifact(error) => error.fmt(formatter),
            Self::VerifierDocumentTooLarge { limit } => write!(
                formatter,
                "canonical rollout verifier document exceeds selected limit {limit}"
            ),
            Self::VerifierDocumentEncoding(error) => write!(
                formatter,
                "could not serialize canonical rollout verifier document: {error}"
            ),
            Self::Agreement(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for RolloutEvidenceError {}

impl From<RolloutEvidenceLimitsError> for RolloutEvidenceError {
    fn from(error: RolloutEvidenceLimitsError) -> Self {
        Self::Limits(error)
    }
}

impl From<RolloutAdmissionError> for RolloutEvidenceError {
    fn from(error: RolloutAdmissionError) -> Self {
        Self::Admission(error)
    }
}

impl From<ArtifactError> for RolloutEvidenceError {
    fn from(error: ArtifactError) -> Self {
        Self::Artifact(error)
    }
}

impl From<RolloutReturnAgreementError> for RolloutEvidenceError {
    fn from(error: RolloutReturnAgreementError) -> Self {
        Self::Agreement(error)
    }
}

/// Independent-verifier refusal for invalid rollout return or provenance facts.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RolloutReturnAgreementError {
    /// The policy omitted a positive finite horizon.
    InvalidHorizon,
    /// The policy omitted a finite gamma in the closed unit interval.
    InvalidGamma,
    /// The policy identity did not match its retained facts.
    PolicyIdentityMismatch,
    /// No terminal or truncated transition was retained.
    MissingTerminal,
    /// More transitions were retained than the declared horizon allows.
    HorizonExceeded,
    /// A transition index did not fit in the wire type.
    StepIndexOverflow,
    /// A transition index was not contiguous from zero.
    InvalidStepOrder {
        /// The expected index.
        expected: u32,
        /// The supplied index.
        actual: u32,
    },
    /// A transition reward was non-finite.
    NonFiniteReward {
        /// The step with the invalid reward.
        step: u32,
    },
    /// A transition claimed both terminal axes.
    AmbiguousTerminal {
        /// The invalid step.
        step: u32,
    },
    /// A terminal transition was followed by another retained transition.
    PostTerminalStep {
        /// The terminal step that was not final.
        step: u32,
    },
    /// Exact return derivation became non-finite.
    NonFiniteReturn,
    /// The serialized undiscounted return differed from independent derivation.
    UndiscountedReturnMismatch {
        /// Independent return bits.
        expected: u64,
        /// Serialized return bits.
        actual: u64,
    },
    /// The serialized discounted return differed from independent derivation.
    DiscountedReturnMismatch {
        /// Independent return bits.
        expected: u64,
        /// Serialized return bits.
        actual: u64,
    },
    /// The canonical manifest did not exactly cover every referenced descriptor.
    ManifestMismatch,
    /// The document digest did not cover its serialized verifier facts.
    EvidenceDigestMismatch,
}

impl Display for RolloutReturnAgreementError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        write!(formatter, "invalid frozen rollout verifier input: {self:?}")
    }
}

impl std::error::Error for RolloutReturnAgreementError {}
