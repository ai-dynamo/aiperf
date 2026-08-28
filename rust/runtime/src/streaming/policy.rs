// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Session target policies: whether the endpoint's actual reply is discarded
//! after comparison or folded into the state later requests are built from.
//!
//! The distinction is the whole point of this module. Under
//! [`SessionTargetPolicy::RecordedInputs`] a stream is a fixed script: the reply
//! is compared against the recording for divergence reporting and then thrown
//! away, so every later authored request is emitted exactly as recorded and no
//! live target output ever enters retained state. Under
//! [`SessionTargetPolicy::TargetClosedLoop`] retained state becomes a copy of
//! what the model actually said, which lands in checkpoints; that is why
//! [`validate_target_policy`] refuses every composition that cannot protect it.

use serde::{Deserialize, Serialize};

use super::{
    checkpoint::CheckpointGeneration,
    checkpoint_backend::StreamingCheckpointBackendDescriptor,
    failure::{SessionCoordinatorError, SessionFailureCode},
};

/// Opaque configuration-visible key selector.
///
/// This is the *only* key-related value that appears in authored
/// configuration. It names material; it never carries it. Key bytes are
/// resolved out of band from an inherited descriptor or an exact-`0600` file
/// (see `streaming::sensitive_state`).
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Deserialize, Serialize)]
#[serde(transparent)]
pub struct SensitiveStateKeyId(String);

impl SensitiveStateKeyId {
    /// Construct a key id from stable authored text.
    #[must_use]
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    /// Borrow the stable key-id text.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for SensitiveStateKeyId {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

/// How a session program treats the endpoint's actual reply.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SessionTargetPolicy {
    /// Compare the reply against recorded content, then discard the comparison.
    ///
    /// Later authored requests are emitted exactly as recorded. No live target
    /// output enters retained state, so no protection is required.
    #[default]
    RecordedInputs,
    /// Fold the reply into session state that later requests are built from.
    ///
    /// Retained state now contains live target output. Selecting this requires
    /// either a checkpoint backend declaring `protects_sensitive_state` plus a
    /// resolvable key id, or checkpointing disabled with no resume claim.
    TargetClosedLoop,
}

impl SessionTargetPolicy {
    /// Whether retained session state may contain live target output.
    #[must_use]
    pub const fn retains_target_output(self) -> bool {
        matches!(self, Self::TargetClosedLoop)
    }
}

/// Outcome of comparing one endpoint reply against its recorded target content.
///
/// Both policies produce this comparison; they differ only in what happens
/// next, which [`TargetReplyFold`] names.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TargetDivergence {
    /// Whether the observed reply differs from the recorded target content.
    pub is_divergent: bool,
    /// Byte length of the recorded target content.
    pub recorded_len: usize,
    /// Byte length of the observed reply.
    pub observed_len: usize,
}

/// What a session must do with the reply once the comparison exists.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TargetReplyFold {
    /// Report divergence and keep the recorded content as retained state.
    ///
    /// The observed bytes are dropped here, which is what makes
    /// `recorded_inputs` unable to affect any later request.
    ReportOnly(TargetDivergence),
    /// Report divergence and retain the observed bytes as session state.
    Retain {
        /// The comparison against the recording, still reported.
        divergence: TargetDivergence,
        /// Live target output that later requests are built from.
        observed: Vec<u8>,
    },
}

impl TargetReplyFold {
    /// Borrow the divergence comparison regardless of policy.
    #[must_use]
    pub const fn divergence(&self) -> &TargetDivergence {
        match self {
            Self::ReportOnly(divergence) | Self::Retain { divergence, .. } => divergence,
        }
    }

    /// Borrow the content later requests are built from.
    ///
    /// Under `recorded_inputs` this is the recorded content the caller passed
    /// in, so the caller keeps ownership of it and this returns `None`.
    #[must_use]
    pub fn retained(&self) -> Option<&[u8]> {
        match self {
            Self::ReportOnly(_) => None,
            Self::Retain { observed, .. } => Some(observed),
        }
    }
}

/// Apply `policy` to one observed reply against its recorded target content.
///
/// `recorded` is borrowed and never modified: under `recorded_inputs` the
/// observed bytes are consumed and dropped, so a divergent reply cannot reach
/// any later request through this path.
#[must_use]
pub fn fold_target_reply(
    policy: SessionTargetPolicy,
    recorded: &[u8],
    observed: Vec<u8>,
) -> TargetReplyFold {
    let divergence = TargetDivergence {
        is_divergent: recorded != observed.as_slice(),
        recorded_len: recorded.len(),
        observed_len: observed.len(),
    };
    match policy {
        SessionTargetPolicy::RecordedInputs => TargetReplyFold::ReportOnly(divergence),
        SessionTargetPolicy::TargetClosedLoop => TargetReplyFold::Retain {
            divergence,
            observed,
        },
    }
}

/// Refuse a `target_closed_loop` composition that cannot protect its state.
///
/// The two admissible shapes are exhaustive and deliberately narrow:
///  - a selected backend whose descriptor sets `protects_sensitive_state`, plus
///    a key id naming the material that seals the state; or
///  - checkpointing disabled (`backend` is `None`) *and* no resume generation
///    claimed, so the state never leaves the process.
///
/// Anything else — a durable backend without the flag, a protecting backend
/// with no key id, or checkpoint-`none` *with* a resume claim — is refused
/// before any state is produced. `recorded_inputs` is always admissible: it
/// retains nothing sensitive, so it has nothing to protect.
pub fn validate_target_policy(
    policy: SessionTargetPolicy,
    backend: Option<&StreamingCheckpointBackendDescriptor>,
    resume: Option<&CheckpointGeneration>,
    key_id: Option<&SensitiveStateKeyId>,
) -> Result<(), SessionCoordinatorError> {
    if !policy.retains_target_output() {
        return Ok(());
    }
    let refusal = SessionCoordinatorError::session(SessionFailureCode::SensitiveStateUnprotected);
    match backend {
        Some(descriptor) => {
            if descriptor.protects_sensitive_state && key_id.is_some() {
                Ok(())
            } else {
                Err(refusal)
            }
        }
        // No backend means nothing is written, so a resume claim is a
        // contradiction rather than a lesser guarantee: it asserts state that
        // was never protected is being restored.
        None if resume.is_none() => Ok(()),
        None => Err(refusal),
    }
}

/// Capability a selected checkpoint backend must satisfy for `policy`.
///
/// Feeds `CheckpointBackendRequirements.needs_sensitive_state_protection`, so a
/// backend factory can fail closed on its own rather than trusting the caller.
#[must_use]
pub const fn needs_sensitive_state_protection(policy: SessionTargetPolicy) -> bool {
    policy.retains_target_output()
}
