// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

//! Rust-owned admission of policy decisions as frozen environment actions.

use std::{
    fmt::{self, Display, Formatter},
    io::{self, Cursor, Write},
};

use serde_json::Value;

use super::artifacts::{
    ArtifactError, ArtifactQuota, EpisodeArtifactStore, FrozenArtifactReference,
};

const MAX_ACTION_ENCODER_ID_BYTES: usize = 128;

/// Opaque identity for the Rust-selected encoder that admits one declared decision.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(crate) struct ActionEncoderId(String);

impl ActionEncoderId {
    /// Validates a stable, non-secret Rust-owned encoder identifier.
    pub(crate) fn new(value: impl AsRef<str>) -> Result<Self, EpisodeActionEncodingError> {
        let value = value.as_ref();
        if value.is_empty()
            || value.len() > MAX_ACTION_ENCODER_ID_BYTES
            || !value
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
        {
            return Err(EpisodeActionEncodingError::InvalidEncoderId);
        }
        Ok(Self(value.to_owned()))
    }
}

/// Exact limits for one Rust-owned decision-to-action admission.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ActionEncodingLimits {
    max_decision_bytes: usize,
    max_action_bytes: usize,
}

impl ActionEncodingLimits {
    /// Requires positive byte limits before a decision or action reaches the artifact store.
    pub(crate) fn new(
        max_decision_bytes: usize,
        max_action_bytes: usize,
    ) -> Result<Self, EpisodeActionEncodingError> {
        if max_decision_bytes == 0 || max_action_bytes == 0 {
            return Err(EpisodeActionEncodingError::InvalidLimits);
        }
        Ok(Self {
            max_decision_bytes,
            max_action_bytes,
        })
    }
}

/// Bounded, immutable policy output that is not yet an environment-action reference.
#[derive(Clone, Debug)]
pub(crate) struct DeclaredPolicyDecision {
    encoder: ActionEncoderId,
    output: Value,
    canonical_bytes: usize,
}

impl DeclaredPolicyDecision {
    /// Retains only a byte-bounded policy output under the encoder it declares.
    pub(crate) fn new(
        encoder: ActionEncoderId,
        output: Value,
        limits: ActionEncodingLimits,
    ) -> Result<Self, EpisodeActionEncodingError> {
        let canonical_bytes = canonical_json_bytes(&output, limits.max_decision_bytes)
            .map_err(|error| error.into_decision_error(limits.max_decision_bytes))?
            .len();
        Ok(Self {
            encoder,
            output,
            canonical_bytes,
        })
    }

    /// Returns the exact Rust-owned encoder selection declared by this decision.
    pub(crate) fn encoder(&self) -> &ActionEncoderId {
        &self.encoder
    }

    /// Borrows the bounded raw policy output for the selected encoder's validation only.
    pub(crate) fn output(&self) -> &Value {
        &self.output
    }
}

/// Validated action document that cannot become a capability until the store freezes it.
#[derive(Clone, Debug)]
pub(crate) struct FrozenActionDocument(Value);

impl FrozenActionDocument {
    fn new(value: Value) -> Self {
        Self(value)
    }
}

mod sealed {
    pub trait EpisodeActionEncoder {}
}

/// Rust-owned validator and encoder for exactly one declared decision kind.
pub(crate) trait EpisodeActionEncoder: sealed::EpisodeActionEncoder {
    /// Returns the only declared decision kind this encoder may admit.
    fn id(&self) -> &ActionEncoderId;

    /// Validates one bounded decision and returns its untrusted-free action document.
    fn encode(
        &self,
        decision: &DeclaredPolicyDecision,
    ) -> Result<FrozenActionDocument, EpisodeActionEncodingError>;
}

/// Freezes a selected encoder's canonical action through the episode-owned artifact store.
pub(crate) fn freeze_declared_policy_action(
    decision: &DeclaredPolicyDecision,
    encoder: &dyn EpisodeActionEncoder,
    store: &mut EpisodeArtifactStore,
    limits: ActionEncodingLimits,
) -> Result<FrozenArtifactReference, EpisodeActionEncodingError> {
    if decision.canonical_bytes > limits.max_decision_bytes {
        return Err(EpisodeActionEncodingError::DecisionTooLarge {
            limit: limits.max_decision_bytes,
            observed_at_least: decision.canonical_bytes,
        });
    }
    if decision.encoder() != encoder.id() {
        return Err(EpisodeActionEncodingError::EncoderSelectionMismatch);
    }

    store.preflight_reference()?;
    let action = encoder.encode(decision)?;
    let bytes = canonical_json_bytes(&action.0, limits.max_action_bytes)
        .map_err(|error| error.into_action_error(limits.max_action_bytes))?;
    let declared_bytes = u64::try_from(bytes.len())
        .map_err(|_| EpisodeActionEncodingError::ArtifactLengthOverflow)?;
    let upload = store.begin_upload(declared_bytes)?;
    store.write_upload(&upload, &mut Cursor::new(bytes))?;
    let artifact = store.commit_upload(&upload)?;
    Ok(store.issue_reference(&artifact)?)
}

/// Decision-to-action admission failure without echoing policy or action contents.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum EpisodeActionEncodingError {
    /// An encoder identifier was empty, overlong, or not an ASCII identifier.
    InvalidEncoderId,
    /// One decision or action byte limit was zero.
    InvalidLimits,
    /// A declared policy output exceeded its admission limit.
    DecisionTooLarge {
        /// The configured byte limit.
        limit: usize,
        /// The observed encoded byte count, or the first excluded byte position.
        observed_at_least: usize,
    },
    /// An encoded action exceeded its freeze limit.
    ActionTooLarge {
        /// The configured byte limit.
        limit: usize,
        /// The observed encoded byte count, or the first excluded byte position.
        observed_at_least: usize,
    },
    /// The selected Rust encoder was not the decision's declared encoder.
    EncoderSelectionMismatch,
    /// The selected encoder rejected the declared decision's shape or values.
    RejectedDecision(&'static str),
    /// A byte length could not be represented by the artifact store.
    ArtifactLengthOverflow,
    /// The episode-owned artifact store rejected a freeze operation.
    Artifact(ArtifactError),
    /// Canonical JSON serialization failed without exposing the source document.
    Canonicalization,
}

impl Display for EpisodeActionEncodingError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidEncoderId => formatter.write_str("invalid action encoder identifier"),
            Self::InvalidLimits => {
                formatter.write_str("action encoding byte limits must be positive")
            }
            Self::DecisionTooLarge {
                limit,
                observed_at_least,
            } => write!(
                formatter,
                "declared policy decision is at least {observed_at_least} bytes, exceeding limit {limit}"
            ),
            Self::ActionTooLarge {
                limit,
                observed_at_least,
            } => write!(
                formatter,
                "encoded action is at least {observed_at_least} bytes, exceeding limit {limit}"
            ),
            Self::EncoderSelectionMismatch => {
                formatter.write_str("selected action encoder does not match declared encoder")
            }
            Self::RejectedDecision(reason) => write!(
                formatter,
                "selected action encoder rejected decision: {reason}"
            ),
            Self::ArtifactLengthOverflow => {
                formatter.write_str("encoded action length does not fit artifact store")
            }
            Self::Artifact(error) => Display::fmt(error, formatter),
            Self::Canonicalization => {
                formatter.write_str("canonical action JSON serialization failed")
            }
        }
    }
}

impl std::error::Error for EpisodeActionEncodingError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Artifact(error) => Some(error),
            _ => None,
        }
    }
}

impl From<ArtifactError> for EpisodeActionEncodingError {
    fn from(error: ArtifactError) -> Self {
        Self::Artifact(error)
    }
}

enum CanonicalJsonError {
    TooLarge { observed_at_least: usize },
    Serialization,
}

impl CanonicalJsonError {
    fn into_decision_error(self, limit: usize) -> EpisodeActionEncodingError {
        match self {
            Self::TooLarge { observed_at_least } => EpisodeActionEncodingError::DecisionTooLarge {
                limit,
                observed_at_least,
            },
            Self::Serialization => EpisodeActionEncodingError::Canonicalization,
        }
    }

    fn into_action_error(self, limit: usize) -> EpisodeActionEncodingError {
        match self {
            Self::TooLarge { observed_at_least } => EpisodeActionEncodingError::ActionTooLarge {
                limit,
                observed_at_least,
            },
            Self::Serialization => EpisodeActionEncodingError::Canonicalization,
        }
    }
}

fn canonical_json_bytes(value: &Value, limit: usize) -> Result<Vec<u8>, CanonicalJsonError> {
    let mut bytes = BoundedJsonBytes::new(limit);
    write_canonical_json(value, &mut bytes)?;
    Ok(bytes.into_bytes())
}

fn write_canonical_json(
    value: &Value,
    bytes: &mut BoundedJsonBytes,
) -> Result<(), CanonicalJsonError> {
    match value {
        Value::Null => bytes.write_all(b"null").map_err(|_| bytes.error())?,
        Value::Bool(value) => {
            serde_json::to_writer(&mut *bytes, value).map_err(|_| bytes.error())?
        }
        Value::Number(value) => {
            serde_json::to_writer(&mut *bytes, value).map_err(|_| bytes.error())?
        }
        Value::String(value) => {
            serde_json::to_writer(&mut *bytes, value).map_err(|_| bytes.error())?
        }
        Value::Array(values) => {
            bytes.write_all(b"[").map_err(|_| bytes.error())?;
            for (index, value) in values.iter().enumerate() {
                if index > 0 {
                    bytes.write_all(b",").map_err(|_| bytes.error())?;
                }
                write_canonical_json(value, bytes)?;
            }
            bytes.write_all(b"]").map_err(|_| bytes.error())?;
        }
        Value::Object(values) => {
            bytes.write_all(b"{").map_err(|_| bytes.error())?;
            let mut fields = values.iter().collect::<Vec<_>>();
            fields.sort_unstable_by(|(left, _), (right, _)| left.cmp(right));
            for (index, (name, value)) in fields.into_iter().enumerate() {
                if index > 0 {
                    bytes.write_all(b",").map_err(|_| bytes.error())?;
                }
                serde_json::to_writer(&mut *bytes, name).map_err(|_| bytes.error())?;
                bytes.write_all(b":").map_err(|_| bytes.error())?;
                write_canonical_json(value, bytes)?;
            }
            bytes.write_all(b"}").map_err(|_| bytes.error())?;
        }
    }
    Ok(())
}

struct BoundedJsonBytes {
    bytes: Vec<u8>,
    limit: usize,
    is_exceeded: bool,
}

impl BoundedJsonBytes {
    fn new(limit: usize) -> Self {
        Self {
            bytes: Vec::new(),
            limit,
            is_exceeded: false,
        }
    }

    fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }

    fn error(&self) -> CanonicalJsonError {
        if self.is_exceeded {
            CanonicalJsonError::TooLarge {
                observed_at_least: self.limit.saturating_add(1),
            }
        } else {
            CanonicalJsonError::Serialization
        }
    }
}

impl Write for BoundedJsonBytes {
    fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
        if buffer.len() > self.limit.saturating_sub(self.bytes.len()) {
            self.is_exceeded = true;
            return Err(io::Error::new(
                io::ErrorKind::WriteZero,
                "canonical JSON byte limit exceeded",
            ));
        }
        self.bytes.extend_from_slice(buffer);
        Ok(buffer.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use serde_json::json;

    use super::{
        ActionEncoderId, ActionEncodingLimits, ArtifactError, DeclaredPolicyDecision,
        EpisodeActionEncoder, EpisodeActionEncodingError, EpisodeArtifactStore,
        FrozenActionDocument, freeze_declared_policy_action,
    };

    struct MoveEncoder {
        id: ActionEncoderId,
    }

    impl super::sealed::EpisodeActionEncoder for MoveEncoder {}

    impl EpisodeActionEncoder for MoveEncoder {
        fn id(&self) -> &ActionEncoderId {
            &self.id
        }

        fn encode(
            &self,
            decision: &DeclaredPolicyDecision,
        ) -> Result<FrozenActionDocument, EpisodeActionEncodingError> {
            let serde_json::Value::Object(values) = decision.output() else {
                return Err(EpisodeActionEncodingError::RejectedDecision(
                    "move action must be an object",
                ));
            };
            let Some(serde_json::Value::String(direction)) = values.get("direction") else {
                return Err(EpisodeActionEncodingError::RejectedDecision(
                    "move action has no declared direction",
                ));
            };
            if values.len() != 2
                || values.get("kind") != Some(&json!("move"))
                || !matches!(direction.as_str(), "north" | "south" | "west")
            {
                return Err(EpisodeActionEncodingError::RejectedDecision(
                    "move action is not declared",
                ));
            }
            Ok(FrozenActionDocument::new(json!({
                "kind": "move",
                "direction": direction,
            })))
        }
    }

    fn declared_move(
        encoder: ActionEncoderId,
        direction: &str,
        limits: ActionEncodingLimits,
    ) -> DeclaredPolicyDecision {
        DeclaredPolicyDecision::new(
            encoder,
            json!({"kind": "move", "direction": direction}),
            limits,
        )
        .expect("declared move is bounded")
    }

    #[test]
    fn only_a_selected_encoder_can_freeze_a_declared_decision_as_a_one_shot_action() {
        let root = tempfile::tempdir().expect("temporary artifact root");
        let mut store = EpisodeArtifactStore::new(
            root.path(),
            super::ArtifactQuota {
                max_artifacts: 2,
                max_total_bytes: 128,
                max_artifact_bytes: 128,
                max_download_handles: 1,
            },
        )
        .expect("artifact quota is valid");
        let limits = ActionEncodingLimits::new(64, 64).expect("positive action limits");
        let move_id = ActionEncoderId::new("move").expect("declared encoder id is valid");
        let raw = DeclaredPolicyDecision::new(
            move_id.clone(),
            json!({"arbitrary": ["untrusted"]}),
            limits,
        )
        .expect("bounded declared decision is accepted but is not an action reference");
        let encoder = MoveEncoder {
            id: move_id.clone(),
        };

        let error = freeze_declared_policy_action(&raw, &encoder, &mut store, limits)
            .expect_err("an encoder must validate raw decision JSON before an action exists");
        assert!(matches!(
            error,
            EpisodeActionEncodingError::RejectedDecision(_)
        ));

        let selected = DeclaredPolicyDecision::new(
            move_id,
            json!({"kind": "move", "direction": "north"}),
            limits,
        )
        .expect("selected decision is bounded");
        let reference = freeze_declared_policy_action(&selected, &encoder, &mut store, limits)
            .expect("a selected encoder freezes exactly its validated action document");

        let mut bytes = Vec::new();
        store
            .copy_download(reference.download(), &mut bytes)
            .expect("the Rust-issued grant delivers the frozen action once");
        assert_eq!(bytes, br#"{"direction":"north","kind":"move"}"#);
        assert!(
            store
                .copy_download(reference.download(), &mut Cursor::new(Vec::new()))
                .is_err(),
            "the returned action reference carries a one-shot store grant"
        );
    }

    #[test]
    fn grant_exhaustion_rolls_back_the_new_frozen_action_before_retry() {
        let root = tempfile::tempdir().expect("temporary artifact root");
        let mut store = EpisodeArtifactStore::new(
            root.path(),
            super::ArtifactQuota {
                max_artifacts: 2,
                max_total_bytes: 256,
                max_artifact_bytes: 128,
                max_download_handles: 1,
            },
        )
        .expect("artifact quota is valid");
        let limits = ActionEncodingLimits::new(64, 64).expect("positive action limits");
        let move_id = ActionEncoderId::new("move").expect("declared encoder id is valid");
        let encoder = MoveEncoder {
            id: move_id.clone(),
        };

        let first = freeze_declared_policy_action(
            &declared_move(move_id.clone(), "north", limits),
            &encoder,
            &mut store,
            limits,
        )
        .expect("the first action holds the only store grant");
        let error = freeze_declared_policy_action(
            &declared_move(move_id.clone(), "south", limits),
            &encoder,
            &mut store,
            limits,
        )
        .expect_err("a second action cannot receive a second live grant");
        assert!(matches!(
            error,
            EpisodeActionEncodingError::Artifact(ArtifactError::DownloadHandleLimit { .. })
        ));

        store
            .revoke_download(first.download())
            .expect("the first grant can be released");
        freeze_declared_policy_action(
            &declared_move(move_id, "west", limits),
            &encoder,
            &mut store,
            limits,
        )
        .expect("the failed second action retained no frozen artifact quota");
    }
}
