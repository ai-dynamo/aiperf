// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

//! Rust-owned admission of policy decisions as frozen environment actions.

use std::{
    fmt::{self, Display, Formatter},
    io::{self, Cursor, Write},
    rc::Rc,
};

use serde_json::{Value, json};

#[cfg(test)]
use super::artifacts::ArtifactQuota;
use super::artifacts::{ArtifactError, EpisodeArtifactStore, FrozenArtifactReference};
use super::package::ActionEncoderFactoryId;

/// Exact limits for one Rust-owned decision-to-action admission.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ActionEncodingLimits {
    max_decision_bytes: usize,
    max_action_bytes: usize,
}

impl ActionEncodingLimits {
    /// Requires positive byte limits before a decision or action reaches the artifact store.
    pub fn new(
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

    /// Returns the maximum raw policy-decision bytes admitted before JSON deserialization.
    pub const fn max_decision_bytes(&self) -> usize {
        self.max_decision_bytes
    }

    /// Returns the maximum canonical action bytes frozen into the episode store.
    pub const fn max_action_bytes(&self) -> usize {
        self.max_action_bytes
    }
}

/// Bounded, immutable policy output that is not yet an environment-action reference.
#[derive(Clone, Debug)]
pub struct DeclaredPolicyDecision {
    encoder: ActionEncoderFactoryId,
    output: Value,
    canonical_bytes: usize,
}

impl DeclaredPolicyDecision {
    /// Retains only a byte-bounded policy output under the encoder it declares.
    pub(crate) fn new(
        encoder: ActionEncoderFactoryId,
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

    /// Lexically bounds one raw model decision before `serde_json` can allocate a value tree.
    pub fn from_json_bytes(
        encoder: ActionEncoderFactoryId,
        bytes: &[u8],
        limits: ActionEncodingLimits,
    ) -> Result<Self, EpisodeActionEncodingError> {
        preflight_raw_decision_json(bytes, limits.max_decision_bytes)?;
        let output = serde_json::from_slice(bytes)
            .map_err(|_| EpisodeActionEncodingError::InvalidDecisionJson)?;
        Self::new(encoder, output, limits)
    }

    /// Returns the exact Rust-owned encoder selection declared by this decision.
    pub fn encoder(&self) -> &ActionEncoderFactoryId {
        &self.encoder
    }

    /// Borrows the bounded raw policy output for the selected encoder's validation only.
    pub fn output(&self) -> &Value {
        &self.output
    }
}

/// Rust-owned validator and encoder for exactly one declared decision kind.
pub trait NativeGraphActionEncoder: Send {
    /// Returns the only declared decision kind this encoder may admit.
    fn id(&self) -> &str;

    /// Validates one bounded decision and returns one unfrozen canonicalizable action document.
    fn encode(
        &self,
        decision: &DeclaredPolicyDecision,
    ) -> Result<Value, EpisodeActionEncodingError>;
}

/// One package-selected encoder bound from a frozen registry before adapter provisioning.
#[derive(Clone)]
pub struct BoundNativeGraphActionEncoder {
    id: ActionEncoderFactoryId,
    authority: ActionAdmissionAuthority,
    encoder: Rc<dyn NativeGraphActionEncoder>,
}

impl BoundNativeGraphActionEncoder {
    /// Binds one encoder only when it matches the already sealed package selector.
    pub fn new(
        id: ActionEncoderFactoryId,
        encoder: Box<dyn NativeGraphActionEncoder>,
    ) -> Result<Self, EpisodeActionEncodingError> {
        if encoder.id() != id.as_str() {
            return Err(EpisodeActionEncodingError::EncoderSelectionMismatch);
        }
        Ok(Self {
            id,
            authority: ActionAdmissionAuthority(Rc::new(())),
            encoder: Rc::from(encoder),
        })
    }

    /// Returns the exact registry identifier bound to this encoder.
    pub fn id(&self) -> &ActionEncoderFactoryId {
        &self.id
    }

    /// Admits one selected policy decision as the only capability a stepper may dispatch.
    pub fn admit(
        &self,
        decision: DeclaredPolicyDecision,
        store: &mut EpisodeArtifactStore,
        limits: ActionEncodingLimits,
    ) -> Result<AdmittedEnvironmentAction, EpisodeActionEncodingError> {
        self.admit_with_session(decision, store, limits, None)
    }

    /// Admits an issued live-rollout decision for exactly one started environment session.
    pub(crate) fn admit_for_session(
        &self,
        decision: DeclaredPolicyDecision,
        store: &mut EpisodeArtifactStore,
        limits: ActionEncodingLimits,
        session: &ActionSessionAuthority,
    ) -> Result<AdmittedEnvironmentAction, EpisodeActionEncodingError> {
        self.admit_with_session(decision, store, limits, Some(session.clone()))
    }

    fn admit_with_session(
        &self,
        decision: DeclaredPolicyDecision,
        store: &mut EpisodeArtifactStore,
        limits: ActionEncodingLimits,
        session: Option<ActionSessionAuthority>,
    ) -> Result<AdmittedEnvironmentAction, EpisodeActionEncodingError> {
        let reference = freeze_declared_policy_action(&decision, self, store, limits)?;
        Ok(AdmittedEnvironmentAction {
            encoder: self.id.clone(),
            authority: self.authority.clone(),
            session,
            reference,
        })
    }

    pub(crate) fn authority(&self) -> ActionAdmissionAuthority {
        self.authority.clone()
    }

    fn encode(
        &self,
        decision: &DeclaredPolicyDecision,
    ) -> Result<Value, EpisodeActionEncodingError> {
        self.encoder.encode(decision)
    }
}

#[derive(Clone, Debug)]
pub(crate) struct ActionAdmissionAuthority(Rc<()>);

/// Opaque authority minted once for each started package-selected environment session.
#[derive(Clone, Debug)]
pub(crate) struct ActionSessionAuthority(Rc<()>);

impl ActionSessionAuthority {
    pub(crate) fn new() -> Self {
        Self(Rc::new(()))
    }

    fn matches(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.0, &other.0)
    }
}

/// Opaque single-use environment action admitted by one selected bound encoder.
///
/// Only [`BoundNativeGraphActionEncoder`] admission can construct this capability. Its artifact
/// reference remains crate-private so it can only be dispatched by the environment-stepper seam.
pub struct AdmittedEnvironmentAction {
    encoder: ActionEncoderFactoryId,
    authority: ActionAdmissionAuthority,
    session: Option<ActionSessionAuthority>,
    reference: FrozenArtifactReference,
}

impl std::fmt::Debug for AdmittedEnvironmentAction {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AdmittedEnvironmentAction")
            .field("encoder", &self.encoder)
            .finish_non_exhaustive()
    }
}

impl AdmittedEnvironmentAction {
    pub(crate) fn encoder(&self) -> &ActionEncoderFactoryId {
        &self.encoder
    }

    pub(crate) fn matches_authority(&self, authority: &ActionAdmissionAuthority) -> bool {
        Rc::ptr_eq(&self.authority.0, &authority.0)
    }

    pub(crate) fn matches_session(&self, session: &ActionSessionAuthority) -> bool {
        self.session
            .as_ref()
            .is_some_and(|issued| issued.matches(session))
    }

    pub(crate) fn reference(&self) -> &FrozenArtifactReference {
        &self.reference
    }
}

/// Built-in validator for the schema-1 `move_v1` action contract.
pub(crate) struct MoveV1ActionEncoder;

impl NativeGraphActionEncoder for MoveV1ActionEncoder {
    fn id(&self) -> &str {
        "move_v1"
    }

    fn encode(
        &self,
        decision: &DeclaredPolicyDecision,
    ) -> Result<Value, EpisodeActionEncodingError> {
        let Value::Object(values) = decision.output() else {
            return Err(EpisodeActionEncodingError::RejectedDecision(
                "move action must be an object",
            ));
        };
        let Some(Value::String(direction)) = values.get("direction") else {
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
        Ok(json!({
            "kind": "move",
            "direction": direction,
        }))
    }
}

/// Freezes a selected encoder's canonical action through the episode-owned artifact store.
pub(crate) fn freeze_declared_policy_action(
    decision: &DeclaredPolicyDecision,
    encoder: &BoundNativeGraphActionEncoder,
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
    let bytes = canonical_json_bytes(&action, limits.max_action_bytes)
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
pub enum EpisodeActionEncodingError {
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
    /// Raw decision bytes were not a complete lexically valid JSON input.
    InvalidDecisionJson,
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
            Self::InvalidDecisionJson => {
                formatter.write_str("declared policy decision is not valid bounded JSON")
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

fn preflight_raw_decision_json(
    bytes: &[u8],
    limit: usize,
) -> Result<(), EpisodeActionEncodingError> {
    if bytes.len() > limit {
        return Err(EpisodeActionEncodingError::DecisionTooLarge {
            limit,
            observed_at_least: bytes.len(),
        });
    }
    let mut index = 0;
    let mut in_string = false;
    while let Some(&byte) = bytes.get(index) {
        if !in_string {
            if byte == b'"' {
                in_string = true;
            }
            index += 1;
            continue;
        }
        match byte {
            b'"' => {
                in_string = false;
                index += 1;
            }
            b'\\' => {
                index += 1;
                let Some(&escape) = bytes.get(index) else {
                    return Err(EpisodeActionEncodingError::InvalidDecisionJson);
                };
                match escape {
                    b'"' | b'\\' | b'/' | b'b' | b'f' | b'n' | b'r' | b't' => index += 1,
                    b'u' => {
                        let Some(units) = bytes.get(index + 1..index + 5) else {
                            return Err(EpisodeActionEncodingError::InvalidDecisionJson);
                        };
                        if !units.iter().all(u8::is_ascii_hexdigit) {
                            return Err(EpisodeActionEncodingError::InvalidDecisionJson);
                        }
                        index += 5;
                    }
                    _ => return Err(EpisodeActionEncodingError::InvalidDecisionJson),
                }
            }
            0..=0x1f => return Err(EpisodeActionEncodingError::InvalidDecisionJson),
            _ => index += 1,
        }
    }
    if in_string || bytes.is_empty() {
        return Err(EpisodeActionEncodingError::InvalidDecisionJson);
    }
    Ok(())
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
        ActionEncoderFactoryId, ActionEncodingLimits, ArtifactError, DeclaredPolicyDecision,
        EpisodeActionEncodingError, EpisodeArtifactStore, freeze_declared_policy_action,
    };
    use crate::eval::native_graph::factories::{
        MoveV1ActionEncoderFactory, NativeGraphActionEncoderFactory,
    };

    fn declared_move(
        encoder: ActionEncoderFactoryId,
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
    fn raw_policy_decision_bytes_are_bounded_before_json_deserialization() {
        let limits = ActionEncodingLimits::new(64, 64).expect("positive action limits");
        let move_id = move_v1_id();

        let error = DeclaredPolicyDecision::from_json_bytes(
            move_id.clone(),
            &[b'x'; 65],
            limits,
        )
        .expect_err(
            "an oversized malformed model response must fail by the byte cap before JSON parsing",
        );
        assert!(matches!(
            error,
            EpisodeActionEncodingError::DecisionTooLarge {
                limit: 64,
                observed_at_least: 65,
            }
        ));

        let error = DeclaredPolicyDecision::from_json_bytes(
            move_id.clone(),
            br#"{"kind":"move","direction":"n"#,
            limits,
        )
        .expect_err("a lexically incomplete decision cannot reach serde_json allocation");
        assert!(matches!(
            error,
            EpisodeActionEncodingError::InvalidDecisionJson
        ));

        let decision = DeclaredPolicyDecision::from_json_bytes(
            move_id,
            br#"{"direction":"north","kind":"move"}"#,
            limits,
        )
        .expect("a bounded complete decision is admitted");
        assert_eq!(
            decision.output(),
            &json!({"direction": "north", "kind": "move"})
        );
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
        let move_id = move_v1_id();
        let raw = DeclaredPolicyDecision::new(
            move_id.clone(),
            json!({"arbitrary": ["untrusted"]}),
            limits,
        )
        .expect("bounded declared decision is accepted but is not an action reference");
        let encoder = MoveV1ActionEncoderFactory
            .bind(&move_id)
            .expect("the selected factory binds the real move_v1 encoder");

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
        let move_id = move_v1_id();
        let encoder = MoveV1ActionEncoderFactory
            .bind(&move_id)
            .expect("the selected factory binds the real move_v1 encoder");

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

    fn move_v1_id() -> ActionEncoderFactoryId {
        serde_json::from_str("\"move_v1\"").expect("move_v1 is a valid action encoder id")
    }
}
