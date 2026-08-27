// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Strict exporter-observable policy parsing and canonical identity.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::ops::Range;
use std::path::{Component, Path};

use serde::de::{Error as _, MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Serialize};

use crate::exporter_observable::{
    ArtifactTreeEntry, ArtifactTreeKind, ReceiverTranscriptEntry, parse_artifact_tree_observable,
    parse_receiver_transcript_observable,
    reject_duplicate_json_keys as reject_observable_duplicates,
    validate_captured_stream_observable,
};
use crate::plugin_stats::{ExporterEvidenceMode, ExporterMember};

const COMPARISON_MAGIC: &[u8] = b"AIPERF_EXPORTER_COMPARISON_V1\0";

/// A typed refusal at the exporter policy boundary.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExporterPolicyError {
    message: String,
}

impl ExporterPolicyError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for ExporterPolicyError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for ExporterPolicyError {}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
enum ExporterPolicyMode {
    Paired,
    StaticCalibration,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
enum ExporterObservableKind {
    ArtifactTree,
    CapturedStream,
    ReceiverTranscript,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct ExporterTransportFieldsRemoved {
    keys: Vec<String>,
    protocol: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum ExporterOutputSelector {
    ArtifactContent { path: String },
    CapturedStream,
    TranscriptBody { sequence: u64 },
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum ExporterLocator {
    ByteRange { length: u64, offset: u64 },
    JsonPointer { pointer: String },
    WholeOutput,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct ExporterEncodedValue {
    encoding: String,
    value: serde_json::Value,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct ExporterProvenanceSlot {
    #[serde(skip_serializing_if = "Option::is_none")]
    dynamic_expected: Option<ExporterEncodedValue>,
    locator: ExporterLocator,
    output_selector: ExporterOutputSelector,
    replacement: ExporterEncodedValue,
    slot_id: String,
    static_expected: ExporterEncodedValue,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct ExporterObservableScenario {
    allows_empty: bool,
    observable_kind: ExporterObservableKind,
    provenance_slots: Vec<ExporterProvenanceSlot>,
    scenario_id: String,
}

/// A validated schema-1 policy whose source was exact RFC 8785 JCS plus newline.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ExporterObservablePolicyV1 {
    #[serde(skip)]
    authenticated_receiver_protocols: BTreeSet<String>,
    #[serde(skip)]
    authenticated_receiver_protocols_blake3: String,
    mode: ExporterPolicyMode,
    receiver_transport_fields_removed: Vec<ExporterTransportFieldsRemoved>,
    scenarios: Vec<ExporterObservableScenario>,
    schema_version: u8,
}

/// Opaque receiver-protocol identity authenticated by one exact policy authority.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AuthenticatedReceiverProtocolV1 {
    protocol: String,
    policy_blake3: String,
    authenticated_receiver_protocols_blake3: String,
    removed_metadata_keys: BTreeSet<String>,
}

impl AuthenticatedReceiverProtocolV1 {
    /// Canonical receiver-protocol identifier.
    pub fn protocol(&self) -> &str {
        &self.protocol
    }

    /// Digest of the complete authenticated receiver-protocol set.
    pub fn authority_blake3(&self) -> &str {
        &self.authenticated_receiver_protocols_blake3
    }

    pub(crate) fn removed_metadata_keys(&self) -> &BTreeSet<String> {
        &self.removed_metadata_keys
    }
}

impl ExporterObservablePolicyV1 {
    /// Digest of the complete pre-run authenticated receiver-protocol set.
    pub fn receiver_protocol_authority_blake3(&self) -> &str {
        &self.authenticated_receiver_protocols_blake3
    }

    /// Bind one canonical receiver protocol to this policy's authenticated authority.
    pub fn authenticate_receiver_protocol(
        &self,
        protocol: &str,
    ) -> Result<AuthenticatedReceiverProtocolV1, ExporterPolicyError> {
        if !is_policy_identifier(protocol)
            || !self.authenticated_receiver_protocols.contains(protocol)
        {
            return Err(ExporterPolicyError::new(
                "receiver protocol is absent from the authenticated policy authority",
            ));
        }
        let removed_metadata_keys = self
            .receiver_transport_fields_removed
            .iter()
            .find(|removal| removal.protocol == protocol)
            .map(|removal| removal.keys.iter().cloned().collect())
            .unwrap_or_default();
        Ok(AuthenticatedReceiverProtocolV1 {
            protocol: protocol.to_owned(),
            policy_blake3: self.canonical_blake3()?,
            authenticated_receiver_protocols_blake3: self
                .authenticated_receiver_protocols_blake3
                .clone(),
            removed_metadata_keys,
        })
    }

    pub(crate) fn validate_receiver_protocol(
        &self,
        protocol: &AuthenticatedReceiverProtocolV1,
    ) -> Result<(), ExporterPolicyError> {
        if protocol.policy_blake3 != self.canonical_blake3()?
            || protocol.authenticated_receiver_protocols_blake3
                != self.authenticated_receiver_protocols_blake3
            || !self
                .authenticated_receiver_protocols
                .contains(protocol.protocol())
        {
            return Err(ExporterPolicyError::new(
                "receiver protocol identity does not match the runner policy authority",
            ));
        }
        Ok(())
    }

    /// Lifecycle this policy authorizes for controlled exporter evidence.
    pub fn evidence_mode(&self) -> ExporterEvidenceMode {
        match self.mode {
            ExporterPolicyMode::StaticCalibration => ExporterEvidenceMode::StaticCalibration,
            ExporterPolicyMode::Paired => ExporterEvidenceMode::Paired,
        }
    }

    /// Observable class authorized for one policy scenario.
    pub fn observable_kind(
        &self,
        scenario_id: &str,
    ) -> Option<crate::plugin_stats::ExporterObservableKind> {
        self.scenarios
            .iter()
            .find(|scenario| scenario.scenario_id == scenario_id)
            .map(|scenario| match scenario.observable_kind {
                ExporterObservableKind::ArtifactTree => {
                    crate::plugin_stats::ExporterObservableKind::ArtifactTree
                }
                ExporterObservableKind::CapturedStream => {
                    crate::plugin_stats::ExporterObservableKind::CapturedStream
                }
                ExporterObservableKind::ReceiverTranscript => {
                    crate::plugin_stats::ExporterObservableKind::ReceiverTranscript
                }
            })
    }

    /// Serialize the validated policy as RFC 8785 JCS with one trailing newline.
    pub fn canonical_bytes(&self) -> Result<Vec<u8>, ExporterPolicyError> {
        let mut bytes = serde_json_canonicalizer::to_vec(self).map_err(|error| {
            ExporterPolicyError::new(format!("cannot canonicalize exporter policy: {error}"))
        })?;
        bytes.push(b'\n');
        Ok(bytes)
    }

    /// Return the lower-case BLAKE3 digest of the exact canonical policy bytes.
    pub fn canonical_blake3(&self) -> Result<String, ExporterPolicyError> {
        Ok(format!("blake3:{}", blake3::hash(&self.canonical_bytes()?)))
    }

    /// Select the exact policy-required backing payloads from host capture.
    pub(crate) fn select_host_backing_payloads(
        &self,
        scenario_id: &str,
        artifact_contents: &BTreeMap<String, Vec<u8>>,
        captured_stream: Option<&[u8]>,
        transcript_bodies: &[Vec<u8>],
    ) -> Result<Vec<SelectedBackingPayloadV1>, ExporterPolicyError> {
        let scenario = self
            .scenarios
            .iter()
            .find(|scenario| scenario.scenario_id == scenario_id)
            .ok_or_else(|| ExporterPolicyError::new("capture scenario is absent from policy"))?;
        let selectors = scenario
            .provenance_slots
            .iter()
            .map(|slot| backing_key_for_selector(&slot.output_selector))
            .collect::<BTreeSet<_>>();
        selectors
            .into_iter()
            .map(|selector| match selector {
                BackingPayloadKey::ArtifactContent(path) => artifact_contents
                    .get(&path)
                    .cloned()
                    .map(|bytes| SelectedBackingPayloadV1::ArtifactContent { path, bytes })
                    .ok_or_else(|| {
                        ExporterPolicyError::new(
                            "host capture is missing policy-selected artifact content",
                        )
                    }),
                BackingPayloadKey::CapturedStream => captured_stream
                    .map(|bytes| SelectedBackingPayloadV1::CapturedStream {
                        bytes: bytes.to_vec(),
                    })
                    .ok_or_else(|| {
                        ExporterPolicyError::new(
                            "host capture is missing the policy-selected captured stream",
                        )
                    }),
                BackingPayloadKey::TranscriptBody(sequence) => transcript_bodies
                    .get(usize::try_from(sequence).map_err(|_| {
                        ExporterPolicyError::new(
                            "policy-selected transcript sequence does not fit usize",
                        )
                    })?)
                    .cloned()
                    .map(|bytes| SelectedBackingPayloadV1::TranscriptBody { sequence, bytes })
                    .ok_or_else(|| {
                        ExporterPolicyError::new(
                            "host capture is missing a policy-selected transcript body",
                        )
                    }),
            })
            .collect()
    }
}

/// Parse, structurally validate, and authenticate one exact canonical policy.
///
/// `authenticated_receiver_protocols` is the pre-run set of receiver protocols
/// actually used by the policy's scenarios. A metadata-removal rule outside
/// that set is refused before canonicalization.
pub fn parse_exporter_observable_policy(
    bytes: &[u8],
    authenticated_receiver_protocols: &BTreeSet<String>,
) -> Result<ExporterObservablePolicyV1, ExporterPolicyError> {
    reject_duplicate_json_keys(bytes)?;
    let mut policy: ExporterObservablePolicyV1 =
        serde_json::from_slice(bytes).map_err(|error| {
            ExporterPolicyError::new(format!("cannot decode exporter observable policy: {error}"))
        })?;
    validate_exporter_policy(&policy, authenticated_receiver_protocols)?;
    if policy.canonical_bytes()? != bytes {
        return Err(ExporterPolicyError::new(
            "exporter observable policy is not exact RFC 8785 JCS plus newline",
        ));
    }
    policy.authenticated_receiver_protocols = authenticated_receiver_protocols.clone();
    policy.authenticated_receiver_protocols_blake3 =
        canonical_protocol_set_blake3(authenticated_receiver_protocols)?;
    Ok(policy)
}

fn canonical_protocol_set_blake3(
    protocols: &BTreeSet<String>,
) -> Result<String, ExporterPolicyError> {
    let bytes = serde_json_canonicalizer::to_vec(protocols).map_err(|error| {
        ExporterPolicyError::new(format!(
            "cannot canonicalize authenticated receiver protocols: {error}"
        ))
    })?;
    Ok(format!("blake3:{}", blake3::hash(&bytes)))
}

/// Exact backing bytes for one selector named by a policy slot.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SelectedBackingPayloadV1 {
    /// Exact bytes of one retained regular file at `path`.
    ArtifactContent {
        /// Normalized logical artifact path.
        path: String,
        /// Exact retained regular-file bytes.
        bytes: Vec<u8>,
    },
    /// Exact captured stream bytes, identical to the raw observable bytes.
    CapturedStream {
        /// Exact captured bytes.
        bytes: Vec<u8>,
    },
    /// Exact retained body bytes for one receiver acceptance sequence.
    TranscriptBody {
        /// Dense receiver acceptance sequence.
        sequence: u64,
        /// Exact decoder-accepted body bytes.
        bytes: Vec<u8>,
    },
}

/// Complete result of applying one scenario's observable policy.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AppliedExporterObservableV1 {
    /// BLAKE3 of the validated class-specific raw observable bytes.
    pub raw_observable_blake3: String,
    /// Rebuilt class-specific comparison bytes.
    pub comparison_bytes: Vec<u8>,
    /// BLAKE3 of `comparison_bytes`.
    pub comparison_observable_blake3: String,
    /// Canonical `ProvenanceReceiptV1` bytes.
    pub provenance_receipt_bytes: Vec<u8>,
    /// BLAKE3 of `provenance_receipt_bytes`.
    pub provenance_receipt_blake3: String,
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum BackingPayloadKey {
    ArtifactContent(String),
    CapturedStream,
    TranscriptBody(u64),
}

enum ParsedObservable {
    ArtifactTree(Vec<ArtifactTreeEntry>),
    CapturedStream,
    ReceiverTranscript(Vec<ReceiverTranscriptEntry>),
}

/// Apply every slot for the binding's scenario to exact retained evidence.
///
/// The backing list must contain exactly one payload for every distinct output
/// selector used by the scenario and no other payloads. Raw tree/transcript
/// acquisition remains outside this API; the supplied bytes are authenticated
/// against their class-specific raw observable before any provenance value is
/// replaced.
pub fn apply_exporter_observable_policy_v1(
    policy: &ExporterObservablePolicyV1,
    binding: &ProvenanceBindingV1,
    raw_observable: &[u8],
    backing_payloads: &[SelectedBackingPayloadV1],
) -> Result<AppliedExporterObservableV1, ExporterPolicyError> {
    validate_provenance_binding(policy, binding)?;
    let scenario = policy
        .scenarios
        .iter()
        .find(|scenario| scenario.scenario_id == binding.scenario_id)
        .ok_or_else(|| ExporterPolicyError::new("application scenario is absent from policy"))?;
    let parsed = parse_raw_observable(scenario, raw_observable)?;
    let raw_observable_blake3 = format!("blake3:{}", blake3::hash(raw_observable));

    let expected_keys = scenario
        .provenance_slots
        .iter()
        .map(|slot| backing_key_for_selector(&slot.output_selector))
        .collect::<BTreeSet<_>>();
    let mut supplied = BTreeMap::<BackingPayloadKey, &[u8]>::new();
    for payload in backing_payloads {
        let (key, bytes) = backing_payload_parts(payload);
        if supplied.insert(key, bytes).is_some() {
            return Err(ExporterPolicyError::new(
                "application contains an ambiguous duplicate backing payload",
            ));
        }
    }
    if supplied.keys().cloned().collect::<BTreeSet<_>>() != expected_keys {
        return Err(ExporterPolicyError::new(
            "application backing payloads do not exactly match policy selectors",
        ));
    }
    for key in &expected_keys {
        let bytes = supplied.get(key).ok_or_else(|| {
            ExporterPolicyError::new("application is missing a selected backing payload")
        })?;
        validate_backing_identity(&parsed, key, bytes, raw_observable)?;
    }

    if scenario.provenance_slots.is_empty() {
        let provenance = generate_provenance_receipt_v1(policy, binding, &[])?;
        return Ok(AppliedExporterObservableV1 {
            raw_observable_blake3: raw_observable_blake3.clone(),
            comparison_bytes: raw_observable.to_vec(),
            comparison_observable_blake3: raw_observable_blake3,
            provenance_receipt_bytes: provenance.bytes,
            provenance_receipt_blake3: provenance.blake3,
        });
    }

    let mut observations = Vec::with_capacity(scenario.provenance_slots.len());
    let mut selections = BTreeMap::<BackingPayloadKey, Vec<ComparisonSelectionV1>>::new();
    for slot in &scenario.provenance_slots {
        let key = backing_key_for_selector(&slot.output_selector);
        let payload = supplied.get(&key).ok_or_else(|| {
            ExporterPolicyError::new("application is missing a selected backing payload")
        })?;
        let span = resolve_locator_span(payload, &slot.locator)?;
        observations.push(ProvenanceObservationV1 {
            slot_id: slot.slot_id.clone(),
            observed_raw: payload[span.clone()].to_vec(),
        });
        selections
            .entry(key)
            .or_default()
            .push(ComparisonSelectionV1 {
                slot_id: slot.slot_id.clone(),
                offset: u64::try_from(span.start).map_err(|_| {
                    ExporterPolicyError::new("selected span offset does not fit unsigned 64-bit")
                })?,
                length: u64::try_from(span.len()).map_err(|_| {
                    ExporterPolicyError::new("selected span length does not fit unsigned 64-bit")
                })?,
                replacement: replacement_for_comparison(&slot.replacement)?,
            });
    }
    let provenance = generate_provenance_receipt_v1(policy, binding, &observations)?;
    let mut transformed = BTreeMap::new();
    for (key, payload_selections) in &selections {
        let payload = supplied.get(key).ok_or_else(|| {
            ExporterPolicyError::new("application is missing a selected backing payload")
        })?;
        transformed.insert(
            key.clone(),
            build_comparison_payload_v1(payload, payload_selections)?,
        );
    }
    let comparison_bytes = rebuild_comparison_observable(parsed, &transformed)?;
    let comparison_observable_blake3 = format!("blake3:{}", blake3::hash(&comparison_bytes));
    Ok(AppliedExporterObservableV1 {
        raw_observable_blake3,
        comparison_bytes,
        comparison_observable_blake3,
        provenance_receipt_bytes: provenance.bytes,
        provenance_receipt_blake3: provenance.blake3,
    })
}

fn parse_raw_observable(
    scenario: &ExporterObservableScenario,
    raw: &[u8],
) -> Result<ParsedObservable, ExporterPolicyError> {
    match scenario.observable_kind {
        ExporterObservableKind::ArtifactTree => parse_artifact_tree_observable(raw)
            .map(ParsedObservable::ArtifactTree)
            .map_err(|error| {
                ExporterPolicyError::new(format!("invalid artifact-tree observable: {error}"))
            }),
        ExporterObservableKind::CapturedStream => {
            validate_captured_stream_observable(raw, scenario.allows_empty).map_err(|error| {
                ExporterPolicyError::new(format!("invalid captured-stream observable: {error}"))
            })?;
            Ok(ParsedObservable::CapturedStream)
        }
        ExporterObservableKind::ReceiverTranscript => {
            parse_receiver_transcript_observable(raw, scenario.allows_empty)
                .map(ParsedObservable::ReceiverTranscript)
                .map_err(|error| {
                    ExporterPolicyError::new(format!(
                        "invalid receiver-transcript observable: {error}"
                    ))
                })
        }
    }
}

fn backing_key_for_selector(selector: &ExporterOutputSelector) -> BackingPayloadKey {
    match selector {
        ExporterOutputSelector::ArtifactContent { path } => {
            BackingPayloadKey::ArtifactContent(path.clone())
        }
        ExporterOutputSelector::CapturedStream => BackingPayloadKey::CapturedStream,
        ExporterOutputSelector::TranscriptBody { sequence } => {
            BackingPayloadKey::TranscriptBody(*sequence)
        }
    }
}

fn backing_payload_parts(payload: &SelectedBackingPayloadV1) -> (BackingPayloadKey, &[u8]) {
    match payload {
        SelectedBackingPayloadV1::ArtifactContent { path, bytes } => {
            (BackingPayloadKey::ArtifactContent(path.clone()), bytes)
        }
        SelectedBackingPayloadV1::CapturedStream { bytes } => {
            (BackingPayloadKey::CapturedStream, bytes)
        }
        SelectedBackingPayloadV1::TranscriptBody { sequence, bytes } => {
            (BackingPayloadKey::TranscriptBody(*sequence), bytes)
        }
    }
}

fn validate_backing_identity(
    observable: &ParsedObservable,
    key: &BackingPayloadKey,
    bytes: &[u8],
    raw_observable: &[u8],
) -> Result<(), ExporterPolicyError> {
    match (observable, key) {
        (ParsedObservable::ArtifactTree(entries), BackingPayloadKey::ArtifactContent(path)) => {
            let entry = entries
                .iter()
                .find(|entry| entry.path == *path)
                .ok_or_else(|| {
                    ExporterPolicyError::new(
                        "selected artifact path is absent from the raw observable",
                    )
                })?;
            if entry.kind != ArtifactTreeKind::RegularFile {
                return Err(ExporterPolicyError::new(
                    "selected artifact path is not a regular file",
                ));
            }
            validate_length_and_digest(entry.length, &entry.blake3, bytes, "artifact file")
        }
        (ParsedObservable::CapturedStream, BackingPayloadKey::CapturedStream) => {
            if bytes != raw_observable {
                return Err(ExporterPolicyError::new(
                    "selected captured-stream backing differs from the raw observable",
                ));
            }
            Ok(())
        }
        (
            ParsedObservable::ReceiverTranscript(entries),
            BackingPayloadKey::TranscriptBody(sequence),
        ) => {
            let entry = entries
                .iter()
                .find(|entry| entry.sequence == *sequence)
                .ok_or_else(|| {
                    ExporterPolicyError::new(
                        "selected receiver sequence is absent from the raw observable",
                    )
                })?;
            validate_length_and_digest(
                entry.body.length,
                &entry.body.blake3,
                bytes,
                "receiver body",
            )
        }
        _ => Err(ExporterPolicyError::new(
            "selected backing payload class disagrees with the policy scenario",
        )),
    }
}

fn validate_length_and_digest(
    expected_length: u64,
    expected_digest: &str,
    bytes: &[u8],
    description: &str,
) -> Result<(), ExporterPolicyError> {
    let length = u64::try_from(bytes.len()).map_err(|_| {
        ExporterPolicyError::new(format!("selected {description} length does not fit u64"))
    })?;
    let digest = format!("blake3:{}", blake3::hash(bytes));
    if length != expected_length || digest != expected_digest {
        return Err(ExporterPolicyError::new(format!(
            "selected {description} does not match its raw-observable identity"
        )));
    }
    Ok(())
}

fn replacement_for_comparison(
    replacement: &ExporterEncodedValue,
) -> Result<ComparisonReplacementV1, ExporterPolicyError> {
    match replacement.encoding.as_str() {
        "canonical_json" => Ok(ComparisonReplacementV1::CanonicalJson(
            replacement.value.clone(),
        )),
        "hex_bytes" => replacement
            .value
            .as_str()
            .map(|value| ComparisonReplacementV1::HexBytes(value.to_owned()))
            .ok_or_else(|| ExporterPolicyError::new("hex comparison replacement is not a string")),
        _ => Err(ExporterPolicyError::new(
            "comparison replacement uses an unknown encoding",
        )),
    }
}

fn resolve_locator_span(
    payload: &[u8],
    locator: &ExporterLocator,
) -> Result<Range<usize>, ExporterPolicyError> {
    match locator {
        ExporterLocator::ByteRange { length, offset } => {
            let start = usize::try_from(*offset).map_err(|_| {
                ExporterPolicyError::new("byte_range offset does not fit this platform")
            })?;
            let length = usize::try_from(*length).map_err(|_| {
                ExporterPolicyError::new("byte_range length does not fit this platform")
            })?;
            let end = start
                .checked_add(length)
                .filter(|end| *end <= payload.len())
                .ok_or_else(|| {
                    ExporterPolicyError::new("byte_range is outside its selected payload")
                })?;
            Ok(start..end)
        }
        ExporterLocator::JsonPointer { pointer } => json_pointer_raw_span(payload, pointer),
        ExporterLocator::WholeOutput => Ok(0..payload.len()),
    }
}

fn rebuild_comparison_observable(
    observable: ParsedObservable,
    transformed: &BTreeMap<BackingPayloadKey, Vec<u8>>,
) -> Result<Vec<u8>, ExporterPolicyError> {
    match observable {
        ParsedObservable::ArtifactTree(entries) => {
            let mut value = serde_json::to_value(&entries).map_err(|error| {
                ExporterPolicyError::new(format!("cannot rebuild artifact tree: {error}"))
            })?;
            let array = value.as_array_mut().ok_or_else(|| {
                ExporterPolicyError::new("artifact-tree projection is not an array")
            })?;
            for (key, payload) in transformed {
                let BackingPayloadKey::ArtifactContent(path) = key else {
                    return Err(ExporterPolicyError::new(
                        "artifact-tree comparison contains a foreign payload class",
                    ));
                };
                let index = entries
                    .iter()
                    .position(|entry| entry.path == *path)
                    .ok_or_else(|| {
                        ExporterPolicyError::new("selected artifact vanished during rebuild")
                    })?;
                replace_length_and_digest(&mut array[index], payload)?;
            }
            canonical_array_bytes(&value, "artifact-tree comparison")
        }
        ParsedObservable::CapturedStream => transformed
            .get(&BackingPayloadKey::CapturedStream)
            .cloned()
            .ok_or_else(|| {
                ExporterPolicyError::new("captured-stream comparison payload is missing")
            }),
        ParsedObservable::ReceiverTranscript(entries) => {
            let mut value = serde_json::to_value(&entries).map_err(|error| {
                ExporterPolicyError::new(format!("cannot rebuild receiver transcript: {error}"))
            })?;
            let array = value.as_array_mut().ok_or_else(|| {
                ExporterPolicyError::new("receiver-transcript projection is not an array")
            })?;
            for (key, payload) in transformed {
                let BackingPayloadKey::TranscriptBody(sequence) = key else {
                    return Err(ExporterPolicyError::new(
                        "receiver comparison contains a foreign payload class",
                    ));
                };
                let index = entries
                    .iter()
                    .position(|entry| entry.sequence == *sequence)
                    .ok_or_else(|| {
                        ExporterPolicyError::new("selected receiver body vanished during rebuild")
                    })?;
                let body = array[index].get_mut("body").ok_or_else(|| {
                    ExporterPolicyError::new("receiver projection lacks its body")
                })?;
                replace_length_and_digest(body, payload)?;
            }
            canonical_array_bytes(&value, "receiver comparison")
        }
    }
}

fn replace_length_and_digest(
    value: &mut serde_json::Value,
    payload: &[u8],
) -> Result<(), ExporterPolicyError> {
    let object = value.as_object_mut().ok_or_else(|| {
        ExporterPolicyError::new("comparison identity projection is not an object")
    })?;
    object.insert(
        "length".to_owned(),
        serde_json::Value::from(
            u64::try_from(payload.len()).map_err(|_| {
                ExporterPolicyError::new("comparison payload length does not fit u64")
            })?,
        ),
    );
    object.insert(
        "blake3".to_owned(),
        serde_json::Value::String(format!("blake3:{}", blake3::hash(payload))),
    );
    Ok(())
}

fn canonical_array_bytes(
    value: &serde_json::Value,
    description: &str,
) -> Result<Vec<u8>, ExporterPolicyError> {
    let mut bytes = serde_json_canonicalizer::to_vec(value).map_err(|error| {
        ExporterPolicyError::new(format!("cannot canonicalize {description}: {error}"))
    })?;
    bytes.push(b'\n');
    Ok(bytes)
}

fn json_pointer_raw_span(
    payload: &[u8],
    pointer: &str,
) -> Result<Range<usize>, ExporterPolicyError> {
    reject_observable_duplicates(payload).map_err(|error| {
        ExporterPolicyError::new(format!("selected JSON payload is not strict: {error}"))
    })?;
    let target = decoded_json_pointer(pointer)?;
    let mut finder = JsonSpanFinder {
        bytes: payload,
        position: 0,
        target: &target,
        found: None,
    };
    finder.skip_whitespace();
    let mut path = Vec::new();
    finder.parse_value(&mut path)?;
    finder.skip_whitespace();
    if finder.position != payload.len() {
        return Err(ExporterPolicyError::new(
            "selected JSON payload contains trailing bytes",
        ));
    }
    finder
        .found
        .ok_or_else(|| ExporterPolicyError::new("JSON pointer is absent from its selected payload"))
}

struct JsonSpanFinder<'a> {
    bytes: &'a [u8],
    position: usize,
    target: &'a [String],
    found: Option<Range<usize>>,
}

impl JsonSpanFinder<'_> {
    fn skip_whitespace(&mut self) {
        while self
            .bytes
            .get(self.position)
            .is_some_and(|byte| matches!(byte, b' ' | b'\n' | b'\r' | b'\t'))
        {
            self.position += 1;
        }
    }

    fn parse_value(&mut self, path: &mut Vec<String>) -> Result<(), ExporterPolicyError> {
        self.skip_whitespace();
        let start = self.position;
        match self.bytes.get(self.position).copied() {
            Some(b'{') => self.parse_object(path)?,
            Some(b'[') => self.parse_array(path)?,
            Some(b'"') => {
                self.scan_string()?;
            }
            Some(_) => self.scan_primitive(),
            None => {
                return Err(ExporterPolicyError::new(
                    "selected JSON payload ended before its value",
                ));
            }
        }
        let end = self.position;
        if path == self.target && self.found.replace(start..end).is_some() {
            return Err(ExporterPolicyError::new(
                "JSON pointer maps to more than one raw token span",
            ));
        }
        Ok(())
    }

    fn parse_object(&mut self, path: &mut Vec<String>) -> Result<(), ExporterPolicyError> {
        self.position += 1;
        self.skip_whitespace();
        if self.bytes.get(self.position) == Some(&b'}') {
            self.position += 1;
            return Ok(());
        }
        loop {
            self.skip_whitespace();
            let key_start = self.position;
            self.scan_string()?;
            let key: String = serde_json::from_slice(&self.bytes[key_start..self.position])
                .map_err(|error| {
                    ExporterPolicyError::new(format!("cannot decode JSON object key: {error}"))
                })?;
            self.skip_whitespace();
            self.consume(b':')?;
            path.push(key);
            self.parse_value(path)?;
            path.pop();
            self.skip_whitespace();
            match self.bytes.get(self.position) {
                Some(b',') => self.position += 1,
                Some(b'}') => {
                    self.position += 1;
                    return Ok(());
                }
                _ => {
                    return Err(ExporterPolicyError::new(
                        "selected JSON object has invalid framing",
                    ));
                }
            }
        }
    }

    fn parse_array(&mut self, path: &mut Vec<String>) -> Result<(), ExporterPolicyError> {
        self.position += 1;
        self.skip_whitespace();
        if self.bytes.get(self.position) == Some(&b']') {
            self.position += 1;
            return Ok(());
        }
        let mut index = 0_u64;
        loop {
            path.push(index.to_string());
            self.parse_value(path)?;
            path.pop();
            index = index
                .checked_add(1)
                .ok_or_else(|| ExporterPolicyError::new("selected JSON array index overflow"))?;
            self.skip_whitespace();
            match self.bytes.get(self.position) {
                Some(b',') => self.position += 1,
                Some(b']') => {
                    self.position += 1;
                    return Ok(());
                }
                _ => {
                    return Err(ExporterPolicyError::new(
                        "selected JSON array has invalid framing",
                    ));
                }
            }
        }
    }

    fn scan_string(&mut self) -> Result<(), ExporterPolicyError> {
        self.consume(b'"')?;
        loop {
            match self.bytes.get(self.position).copied() {
                Some(b'"') => {
                    self.position += 1;
                    return Ok(());
                }
                Some(b'\\') => {
                    self.position = self.position.checked_add(2).ok_or_else(|| {
                        ExporterPolicyError::new("selected JSON string offset overflow")
                    })?;
                    if self.position > self.bytes.len() {
                        return Err(ExporterPolicyError::new(
                            "selected JSON string has a truncated escape",
                        ));
                    }
                }
                Some(_) => self.position += 1,
                None => {
                    return Err(ExporterPolicyError::new(
                        "selected JSON string is unterminated",
                    ));
                }
            }
        }
    }

    fn scan_primitive(&mut self) {
        while self
            .bytes
            .get(self.position)
            .is_some_and(|byte| !matches!(byte, b',' | b']' | b'}' | b' ' | b'\n' | b'\r' | b'\t'))
        {
            self.position += 1;
        }
    }

    fn consume(&mut self, expected: u8) -> Result<(), ExporterPolicyError> {
        if self.bytes.get(self.position) != Some(&expected) {
            return Err(ExporterPolicyError::new(
                "selected JSON payload has invalid token framing",
            ));
        }
        self.position += 1;
        Ok(())
    }
}

/// Replacement bytes installed into one selected comparison frame.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ComparisonReplacementV1 {
    /// One strict JSON value serialized as RFC 8785 JCS without a newline.
    CanonicalJson(serde_json::Value),
    /// Lower-case even-length hexadecimal bytes.
    HexBytes(String),
}

/// One selected raw span used to construct a comparison payload.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ComparisonSelectionV1 {
    /// Policy slot identifier placed in the selected frame.
    pub slot_id: String,
    /// Zero-based offset in the enclosing raw payload.
    pub offset: u64,
    /// Positive selected raw-span length.
    pub length: u64,
    /// Policy-authorized replacement encoded for the selected frame.
    pub replacement: ComparisonReplacementV1,
}

/// Construct the literal `ComparisonPayloadV1` frame stream for one raw payload.
pub fn build_comparison_payload_v1(
    raw: &[u8],
    selections: &[ComparisonSelectionV1],
) -> Result<Vec<u8>, ExporterPolicyError> {
    if selections.is_empty() {
        return Err(ExporterPolicyError::new(
            "comparison payload requires at least one selected slot",
        ));
    }
    let raw_length = u64::try_from(raw.len()).map_err(|_| {
        ExporterPolicyError::new("comparison payload length does not fit unsigned 64-bit")
    })?;
    let mut ordered = selections.iter().collect::<Vec<_>>();
    ordered.sort_by_key(|selection| selection.offset);
    let permits_empty_whole_output = raw_length == 0 && ordered.len() == 1;
    let mut slot_ids = BTreeSet::new();
    let mut cursor = 0_u64;
    let mut output = COMPARISON_MAGIC.to_vec();
    for selection in ordered {
        if !is_policy_identifier(&selection.slot_id) || !slot_ids.insert(&selection.slot_id) {
            return Err(ExporterPolicyError::new(
                "comparison selections require unique policy slot identifiers",
            ));
        }
        let end = selection
            .offset
            .checked_add(selection.length)
            .filter(|end| {
                (selection.length > 0 || permits_empty_whole_output) && *end <= raw_length
            })
            .ok_or_else(|| {
                ExporterPolicyError::new(
                    "comparison selection is empty, overflowing, or outside its payload",
                )
            })?;
        if selection.offset < cursor {
            return Err(ExporterPolicyError::new(
                "comparison selections overlap in the raw payload",
            ));
        }
        if selection.offset > cursor {
            let start = usize::try_from(cursor).map_err(|_| {
                ExporterPolicyError::new("comparison raw offset does not fit usize")
            })?;
            let stop = usize::try_from(selection.offset).map_err(|_| {
                ExporterPolicyError::new("comparison raw offset does not fit usize")
            })?;
            append_raw_frame(&mut output, &raw[start..stop])?;
        }
        let replacement = comparison_replacement_bytes(&selection.replacement)?;
        output.push(0x01);
        append_length(&mut output, selection.slot_id.len())?;
        output.extend_from_slice(selection.slot_id.as_bytes());
        append_length(&mut output, replacement.len())?;
        output.extend_from_slice(&replacement);
        cursor = end;
    }
    if cursor < raw_length {
        let start = usize::try_from(cursor)
            .map_err(|_| ExporterPolicyError::new("comparison raw offset does not fit usize"))?;
        append_raw_frame(&mut output, &raw[start..])?;
    }
    output.push(0xff);
    Ok(output)
}

/// Immutable enclosing facts copied into every provenance receipt element.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProvenanceBindingV1 {
    /// Digest of the immutable pre-run experiment identity.
    pub experiment_identity_blake3: String,
    /// Zero-based complete-attempt ordinal.
    pub attempt_ordinal: u64,
    /// Policy scenario identifier.
    pub scenario_id: String,
    /// Pair identifier shared by the compared members.
    pub pair_id: String,
    /// Static or dynamic member whose output was observed.
    pub member: ExporterMember,
    /// Zero-based controlled repetition ordinal.
    pub repetition_ordinal: u64,
}

/// Exact selected raw bytes supplied for one policy slot.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProvenanceObservationV1 {
    /// Policy slot identifier, in policy-slot order.
    pub slot_id: String,
    /// Exact selected raw-token or raw-span bytes.
    pub observed_raw: Vec<u8>,
}

/// Canonical provenance receipt bytes and their exact BLAKE3 digest.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CanonicalProvenanceReceiptV1 {
    /// RFC 8785 JCS array with exactly one trailing newline.
    pub bytes: Vec<u8>,
    /// Lower-case BLAKE3 digest of `bytes`.
    pub blake3: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct ProvenanceReceiptEntryV1 {
    attempt_ordinal: u64,
    expected: ExporterEncodedValue,
    experiment_identity_blake3: String,
    locator: ExporterLocator,
    member: ExporterMember,
    observed_raw_hex: String,
    observed_value: ExporterEncodedValue,
    output_selector: ExporterOutputSelector,
    pair_id: String,
    policy_mode: ExporterPolicyMode,
    repetition_ordinal: u64,
    replacement: ExporterEncodedValue,
    scenario_id: String,
    schema_version: u8,
    slot_id: String,
}

/// Generate a receipt from already-selected raw slot bytes.
///
/// Raw artifact/transcript acquisition and JSON-pointer span discovery are
/// deliberately outside this function. The caller supplies each exact selected
/// span; this boundary validates its encoding and policy-authorized value.
pub fn generate_provenance_receipt_v1(
    policy: &ExporterObservablePolicyV1,
    binding: &ProvenanceBindingV1,
    observations: &[ProvenanceObservationV1],
) -> Result<CanonicalProvenanceReceiptV1, ExporterPolicyError> {
    validate_provenance_binding(policy, binding)?;
    let scenario = policy
        .scenarios
        .iter()
        .find(|scenario| scenario.scenario_id == binding.scenario_id)
        .ok_or_else(|| ExporterPolicyError::new("provenance scenario is absent from policy"))?;
    if observations.len() != scenario.provenance_slots.len() {
        return Err(ExporterPolicyError::new(
            "provenance observations do not exactly cover policy slots",
        ));
    }
    let mut entries = Vec::with_capacity(observations.len());
    for (slot, observation) in scenario.provenance_slots.iter().zip(observations) {
        if observation.slot_id != slot.slot_id {
            return Err(ExporterPolicyError::new(
                "provenance observations are not in exact policy-slot order",
            ));
        }
        let expected = expected_for_member(policy.mode, binding.member, slot)?.clone();
        let observed_value = observed_value(&slot.locator, &observation.observed_raw)?;
        if canonical_encoded_value(&observed_value)? != canonical_encoded_value(&expected)? {
            return Err(ExporterPolicyError::new(
                "provenance observed value does not equal policy expectation",
            ));
        }
        entries.push(ProvenanceReceiptEntryV1 {
            attempt_ordinal: binding.attempt_ordinal,
            expected,
            experiment_identity_blake3: binding.experiment_identity_blake3.clone(),
            locator: slot.locator.clone(),
            member: binding.member,
            observed_raw_hex: encode_lower_hex(&observation.observed_raw),
            observed_value,
            output_selector: slot.output_selector.clone(),
            pair_id: binding.pair_id.clone(),
            policy_mode: policy.mode,
            repetition_ordinal: binding.repetition_ordinal,
            replacement: slot.replacement.clone(),
            scenario_id: binding.scenario_id.clone(),
            schema_version: 1,
            slot_id: slot.slot_id.clone(),
        });
    }
    let mut bytes = serde_json_canonicalizer::to_vec(&entries).map_err(|error| {
        ExporterPolicyError::new(format!("cannot canonicalize provenance receipt: {error}"))
    })?;
    bytes.push(b'\n');
    Ok(CanonicalProvenanceReceiptV1 {
        blake3: format!("blake3:{}", blake3::hash(&bytes)),
        bytes,
    })
}

/// Validate exact receipt JCS and all fields against supplied policy evidence.
pub fn validate_provenance_receipt_v1(
    bytes: &[u8],
    policy: &ExporterObservablePolicyV1,
    binding: &ProvenanceBindingV1,
    observations: &[ProvenanceObservationV1],
) -> Result<CanonicalProvenanceReceiptV1, ExporterPolicyError> {
    reject_duplicate_json_keys(bytes)?;
    let parsed: Vec<ProvenanceReceiptEntryV1> = serde_json::from_slice(bytes).map_err(|error| {
        ExporterPolicyError::new(format!("cannot decode provenance receipt: {error}"))
    })?;
    let mut canonical = serde_json_canonicalizer::to_vec(&parsed).map_err(|error| {
        ExporterPolicyError::new(format!("cannot canonicalize provenance receipt: {error}"))
    })?;
    canonical.push(b'\n');
    if canonical != bytes {
        return Err(ExporterPolicyError::new(
            "provenance receipt is not exact RFC 8785 JCS plus newline",
        ));
    }
    let expected = generate_provenance_receipt_v1(policy, binding, observations)?;
    if expected.bytes != bytes {
        return Err(ExporterPolicyError::new(
            "provenance receipt disagrees with policy or supplied evidence",
        ));
    }
    Ok(expected)
}

fn validate_provenance_binding(
    policy: &ExporterObservablePolicyV1,
    binding: &ProvenanceBindingV1,
) -> Result<(), ExporterPolicyError> {
    if !is_blake3_digest(&binding.experiment_identity_blake3) {
        return Err(ExporterPolicyError::new(
            "provenance experiment identity is not canonical BLAKE3",
        ));
    }
    if !is_policy_identifier(&binding.scenario_id)
        || binding.pair_id.is_empty()
        || binding.pair_id.contains('\0')
    {
        return Err(ExporterPolicyError::new(
            "provenance binding contains an invalid identifier or string",
        ));
    }
    match (policy.mode, binding.member) {
        (ExporterPolicyMode::Paired, ExporterMember::Static | ExporterMember::Dynamic)
        | (ExporterPolicyMode::StaticCalibration, ExporterMember::Static) => Ok(()),
        (ExporterPolicyMode::StaticCalibration, ExporterMember::Dynamic) => Err(
            ExporterPolicyError::new("static_calibration provenance member must be static"),
        ),
    }
}

fn expected_for_member(
    mode: ExporterPolicyMode,
    member: ExporterMember,
    slot: &ExporterProvenanceSlot,
) -> Result<&ExporterEncodedValue, ExporterPolicyError> {
    match (mode, member) {
        (ExporterPolicyMode::Paired, ExporterMember::Static)
        | (ExporterPolicyMode::StaticCalibration, ExporterMember::Static) => {
            Ok(&slot.static_expected)
        }
        (ExporterPolicyMode::Paired, ExporterMember::Dynamic) => {
            slot.dynamic_expected.as_ref().ok_or_else(|| {
                ExporterPolicyError::new("paired exporter policy slot lacks dynamic_expected")
            })
        }
        (ExporterPolicyMode::StaticCalibration, ExporterMember::Dynamic) => Err(
            ExporterPolicyError::new("static_calibration provenance member must be static"),
        ),
    }
}

fn observed_value(
    locator: &ExporterLocator,
    raw: &[u8],
) -> Result<ExporterEncodedValue, ExporterPolicyError> {
    match locator {
        ExporterLocator::JsonPointer { .. } => {
            reject_duplicate_json_keys(raw)?;
            let value: serde_json::Value = serde_json::from_slice(raw).map_err(|error| {
                ExporterPolicyError::new(format!(
                    "provenance JSON token is not one strict JSON value: {error}"
                ))
            })?;
            validate_json_value(&value)?;
            Ok(ExporterEncodedValue {
                encoding: "canonical_json".to_owned(),
                value,
            })
        }
        ExporterLocator::ByteRange { length, .. } => {
            if usize::try_from(*length).ok() != Some(raw.len()) {
                return Err(ExporterPolicyError::new(
                    "provenance byte_range span length disagrees with its locator",
                ));
            }
            Ok(ExporterEncodedValue {
                encoding: "hex_bytes".to_owned(),
                value: serde_json::Value::String(encode_lower_hex(raw)),
            })
        }
        ExporterLocator::WholeOutput => Ok(ExporterEncodedValue {
            encoding: "hex_bytes".to_owned(),
            value: serde_json::Value::String(encode_lower_hex(raw)),
        }),
    }
}

fn canonical_encoded_value(value: &ExporterEncodedValue) -> Result<Vec<u8>, ExporterPolicyError> {
    serde_json_canonicalizer::to_vec(value).map_err(|error| {
        ExporterPolicyError::new(format!("cannot canonicalize provenance value: {error}"))
    })
}

fn comparison_replacement_bytes(
    replacement: &ComparisonReplacementV1,
) -> Result<Vec<u8>, ExporterPolicyError> {
    match replacement {
        ComparisonReplacementV1::CanonicalJson(value) => {
            validate_json_value(value)?;
            serde_json_canonicalizer::to_vec(value).map_err(|error| {
                ExporterPolicyError::new(format!(
                    "cannot canonicalize comparison replacement: {error}"
                ))
            })
        }
        ComparisonReplacementV1::HexBytes(encoded) => decode_lower_hex(encoded),
    }
}

fn append_raw_frame(output: &mut Vec<u8>, raw: &[u8]) -> Result<(), ExporterPolicyError> {
    if raw.is_empty() {
        return Err(ExporterPolicyError::new(
            "comparison raw frames must not be empty",
        ));
    }
    output.push(0x00);
    append_length(output, raw.len())?;
    output.extend_from_slice(raw);
    Ok(())
}

fn append_length(output: &mut Vec<u8>, length: usize) -> Result<(), ExporterPolicyError> {
    let length = u64::try_from(length).map_err(|_| {
        ExporterPolicyError::new("comparison frame length does not fit unsigned 64-bit")
    })?;
    output.extend_from_slice(&length.to_be_bytes());
    Ok(())
}

fn decode_lower_hex(encoded: &str) -> Result<Vec<u8>, ExporterPolicyError> {
    if !encoded.len().is_multiple_of(2)
        || !encoded
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(ExporterPolicyError::new(
            "hex bytes must use even-length lower-case hexadecimal",
        ));
    }
    let (pairs, remainder) = encoded.as_bytes().as_chunks::<2>();
    debug_assert!(remainder.is_empty());
    pairs
        .iter()
        .map(|pair| {
            let high = hex_nibble(pair[0]);
            let low = hex_nibble(pair[1]);
            high.zip(low)
                .map(|(high, low)| (high << 4) | low)
                .ok_or_else(|| ExporterPolicyError::new("invalid lower-case hexadecimal byte"))
        })
        .collect()
}

fn hex_nibble(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        _ => None,
    }
}

fn encode_lower_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut encoded = String::with_capacity(bytes.len().saturating_mul(2));
    for byte in bytes {
        encoded.push(char::from(HEX[usize::from(byte >> 4)]));
        encoded.push(char::from(HEX[usize::from(byte & 0x0f)]));
    }
    encoded
}

fn is_blake3_digest(value: &str) -> bool {
    value.strip_prefix("blake3:").is_some_and(|hex| {
        hex.len() == 64
            && hex
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    })
}

fn is_policy_identifier(value: &str) -> bool {
    let bytes = value.as_bytes();
    (1..=128).contains(&bytes.len())
        && (bytes[0].is_ascii_lowercase() || bytes[0].is_ascii_digit())
        && bytes.iter().all(|byte| {
            byte.is_ascii_lowercase() || byte.is_ascii_digit() || matches!(byte, b'_' | b'.' | b'-')
        })
}

fn validate_json_value(value: &serde_json::Value) -> Result<(), ExporterPolicyError> {
    match value {
        serde_json::Value::String(value) if value.contains('\0') => Err(ExporterPolicyError::new(
            "exporter policy strings must not contain NUL",
        )),
        serde_json::Value::Array(values) => {
            for value in values {
                validate_json_value(value)?;
            }
            Ok(())
        }
        serde_json::Value::Object(values) => {
            for (key, value) in values {
                if key.contains('\0') {
                    return Err(ExporterPolicyError::new(
                        "exporter policy object keys must not contain NUL",
                    ));
                }
                validate_json_value(value)?;
            }
            Ok(())
        }
        _ => Ok(()),
    }
}

fn validate_artifact_path(path: &str) -> Result<(), ExporterPolicyError> {
    let parsed = Path::new(path);
    if path.is_empty()
        || path.contains(['\0', '\n', '\r', '\\'])
        || path.starts_with('/')
        || path.ends_with('/')
        || path.contains("//")
        || parsed.is_absolute()
        || parsed
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err(ExporterPolicyError::new(
            "exporter artifact selector path is not normalized",
        ));
    }
    Ok(())
}

fn validate_json_pointer(pointer: &str) -> Result<(), ExporterPolicyError> {
    if pointer.contains('\0') || (!pointer.is_empty() && !pointer.starts_with('/')) {
        return Err(ExporterPolicyError::new(
            "exporter policy contains malformed JSON pointer",
        ));
    }
    let mut bytes = pointer.bytes();
    while let Some(byte) = bytes.next() {
        if byte == b'~'
            && !bytes
                .next()
                .is_some_and(|escaped| matches!(escaped, b'0' | b'1'))
        {
            return Err(ExporterPolicyError::new(
                "exporter policy contains malformed JSON pointer",
            ));
        }
    }
    Ok(())
}

fn decoded_json_pointer(pointer: &str) -> Result<Vec<String>, ExporterPolicyError> {
    validate_json_pointer(pointer)?;
    if pointer.is_empty() {
        return Ok(Vec::new());
    }
    pointer[1..]
        .split('/')
        .map(|token| {
            let mut decoded = String::new();
            let mut characters = token.chars();
            while let Some(character) = characters.next() {
                if character == '~' {
                    decoded.push(match characters.next() {
                        Some('0') => '~',
                        Some('1') => '/',
                        _ => {
                            return Err(ExporterPolicyError::new(
                                "exporter policy contains malformed JSON pointer",
                            ));
                        }
                    });
                } else {
                    decoded.push(character);
                }
            }
            Ok(decoded)
        })
        .collect()
}

fn locators_overlap(
    left: &ExporterLocator,
    right: &ExporterLocator,
) -> Result<bool, ExporterPolicyError> {
    match (left, right) {
        (ExporterLocator::WholeOutput, _) | (_, ExporterLocator::WholeOutput) => Ok(true),
        (
            ExporterLocator::ByteRange {
                length: left_length,
                offset: left_offset,
            },
            ExporterLocator::ByteRange {
                length: right_length,
                offset: right_offset,
            },
        ) => {
            let left_end = left_offset.checked_add(*left_length).ok_or_else(|| {
                ExporterPolicyError::new("exporter byte_range must be nonempty and bounded")
            })?;
            let right_end = right_offset.checked_add(*right_length).ok_or_else(|| {
                ExporterPolicyError::new("exporter byte_range must be nonempty and bounded")
            })?;
            Ok(*left_offset < right_end && *right_offset < left_end)
        }
        (
            ExporterLocator::JsonPointer {
                pointer: left_pointer,
            },
            ExporterLocator::JsonPointer {
                pointer: right_pointer,
            },
        ) => {
            let left = decoded_json_pointer(left_pointer)?;
            let right = decoded_json_pointer(right_pointer)?;
            Ok(left.starts_with(&right) || right.starts_with(&left))
        }
        _ => Ok(false),
    }
}

fn validate_encoded_value(
    value: &ExporterEncodedValue,
    locator: &ExporterLocator,
) -> Result<(), ExporterPolicyError> {
    validate_json_value(&value.value)?;
    match (value.encoding.as_str(), locator) {
        ("canonical_json", ExporterLocator::JsonPointer { .. }) => Ok(()),
        ("hex_bytes", ExporterLocator::ByteRange { .. } | ExporterLocator::WholeOutput) => {
            let encoded = value.value.as_str().ok_or_else(|| {
                ExporterPolicyError::new("hex_bytes exporter policy value must be a string")
            })?;
            if !encoded.len().is_multiple_of(2)
                || !encoded
                    .bytes()
                    .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
            {
                return Err(ExporterPolicyError::new(
                    "hex_bytes exporter policy value must be even lower-case hex",
                ));
            }
            Ok(())
        }
        _ => Err(ExporterPolicyError::new(
            "exporter policy encoding is incompatible with its locator",
        )),
    }
}

fn validate_exporter_policy(
    policy: &ExporterObservablePolicyV1,
    authenticated_receiver_protocols: &BTreeSet<String>,
) -> Result<(), ExporterPolicyError> {
    if authenticated_receiver_protocols
        .iter()
        .any(|protocol| !is_policy_identifier(protocol))
    {
        return Err(ExporterPolicyError::new(
            "authenticated receiver protocol identity is not canonical",
        ));
    }
    if policy.schema_version != 1 {
        return Err(ExporterPolicyError::new(
            "exporter observable policy schema_version must be 1",
        ));
    }
    let mut previous_protocol = None;
    for removal in &policy.receiver_transport_fields_removed {
        if !is_policy_identifier(&removal.protocol)
            || previous_protocol.is_some_and(|previous| previous >= removal.protocol.as_str())
        {
            return Err(ExporterPolicyError::new(
                "exporter policy protocols must be sorted and unique",
            ));
        }
        if !authenticated_receiver_protocols.contains(&removal.protocol) {
            return Err(ExporterPolicyError::new(
                "exporter policy contains a transport removal absent from authenticated receiver protocols",
            ));
        }
        previous_protocol = Some(removal.protocol.as_str());
        let mut previous_key = None;
        for key in &removal.keys {
            if key.is_empty()
                || key.contains('\0')
                || key.bytes().any(|byte| byte.is_ascii_uppercase())
                || previous_key.is_some_and(|previous| previous >= key.as_str())
            {
                return Err(ExporterPolicyError::new(
                    "exporter policy metadata keys must be lower-case, sorted, and unique",
                ));
            }
            previous_key = Some(key.as_str());
        }
    }
    if !policy.receiver_transport_fields_removed.is_empty()
        && !policy
            .scenarios
            .iter()
            .any(|scenario| scenario.observable_kind == ExporterObservableKind::ReceiverTranscript)
    {
        return Err(ExporterPolicyError::new(
            "exporter policy contains a transport removal unused by every receiver scenario",
        ));
    }

    let mut previous_scenario = None;
    for scenario in &policy.scenarios {
        if !is_policy_identifier(&scenario.scenario_id)
            || previous_scenario.is_some_and(|previous| previous >= scenario.scenario_id.as_str())
        {
            return Err(ExporterPolicyError::new(
                "exporter policy scenarios must be sorted and unique",
            ));
        }
        previous_scenario = Some(scenario.scenario_id.as_str());
        let mut previous_slot = None;
        let mut selector_locators = BTreeSet::new();
        let mut locators_by_selector = BTreeMap::<String, Vec<&ExporterLocator>>::new();
        for slot in &scenario.provenance_slots {
            if !is_policy_identifier(&slot.slot_id)
                || previous_slot.is_some_and(|previous| previous >= slot.slot_id.as_str())
            {
                return Err(ExporterPolicyError::new(
                    "exporter policy slots must be sorted and unique",
                ));
            }
            previous_slot = Some(slot.slot_id.as_str());
            match (&scenario.observable_kind, &slot.output_selector) {
                (
                    ExporterObservableKind::ArtifactTree,
                    ExporterOutputSelector::ArtifactContent { path },
                ) => validate_artifact_path(path)?,
                (
                    ExporterObservableKind::CapturedStream,
                    ExporterOutputSelector::CapturedStream,
                )
                | (
                    ExporterObservableKind::ReceiverTranscript,
                    ExporterOutputSelector::TranscriptBody { .. },
                ) => {}
                _ => {
                    return Err(ExporterPolicyError::new(
                        "exporter policy selector is incompatible with observable kind",
                    ));
                }
            }
            match &slot.locator {
                ExporterLocator::ByteRange { length, offset } => {
                    if *length == 0 || offset.checked_add(*length).is_none() {
                        return Err(ExporterPolicyError::new(
                            "exporter byte_range must be nonempty and bounded",
                        ));
                    }
                }
                ExporterLocator::JsonPointer { pointer } => validate_json_pointer(pointer)?,
                ExporterLocator::WholeOutput => {}
            }
            validate_encoded_value(&slot.static_expected, &slot.locator)?;
            validate_encoded_value(&slot.replacement, &slot.locator)?;
            if slot.static_expected.encoding != slot.replacement.encoding {
                return Err(ExporterPolicyError::new(
                    "exporter policy slot encodings disagree",
                ));
            }
            match (policy.mode, slot.dynamic_expected.as_ref()) {
                (ExporterPolicyMode::Paired, Some(dynamic)) => {
                    validate_encoded_value(dynamic, &slot.locator)?;
                    if dynamic.encoding != slot.replacement.encoding {
                        return Err(ExporterPolicyError::new(
                            "exporter policy slot encodings disagree",
                        ));
                    }
                }
                (ExporterPolicyMode::Paired, None) => {
                    return Err(ExporterPolicyError::new(
                        "paired exporter policy slot lacks dynamic_expected",
                    ));
                }
                (ExporterPolicyMode::StaticCalibration, Some(_)) => {
                    return Err(ExporterPolicyError::new(
                        "static_calibration exporter policy slot contains dynamic_expected",
                    ));
                }
                (ExporterPolicyMode::StaticCalibration, None) => {}
            }
            let selector_locator = serde_json::to_string(&(&slot.output_selector, &slot.locator))
                .map_err(|error| {
                ExporterPolicyError::new(format!(
                    "cannot encode exporter selector and locator: {error}"
                ))
            })?;
            if !selector_locators.insert(selector_locator) {
                return Err(ExporterPolicyError::new(
                    "exporter policy contains duplicate selector/locator pair",
                ));
            }
            let selector = serde_json::to_string(&slot.output_selector).map_err(|error| {
                ExporterPolicyError::new(format!("cannot encode exporter selector: {error}"))
            })?;
            let peer_locators = locators_by_selector.entry(selector).or_default();
            for peer in peer_locators.iter() {
                if locators_overlap(peer, &slot.locator)? {
                    return Err(ExporterPolicyError::new(
                        "exporter policy contains overlapping output slots",
                    ));
                }
            }
            peer_locators.push(&slot.locator);
        }
    }
    Ok(())
}

struct DuplicateRejectingJson;

impl<'de> Deserialize<'de> for DuplicateRejectingJson {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct JsonVisitor;

        impl<'de> Visitor<'de> for JsonVisitor {
            type Value = DuplicateRejectingJson;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("a JSON value without duplicate object keys")
            }

            fn visit_bool<E>(self, _value: bool) -> Result<Self::Value, E> {
                Ok(DuplicateRejectingJson)
            }

            fn visit_i64<E>(self, _value: i64) -> Result<Self::Value, E> {
                Ok(DuplicateRejectingJson)
            }

            fn visit_u64<E>(self, _value: u64) -> Result<Self::Value, E> {
                Ok(DuplicateRejectingJson)
            }

            fn visit_f64<E>(self, _value: f64) -> Result<Self::Value, E> {
                Ok(DuplicateRejectingJson)
            }

            fn visit_str<E>(self, _value: &str) -> Result<Self::Value, E> {
                Ok(DuplicateRejectingJson)
            }

            fn visit_none<E>(self) -> Result<Self::Value, E> {
                Ok(DuplicateRejectingJson)
            }

            fn visit_unit<E>(self) -> Result<Self::Value, E> {
                Ok(DuplicateRejectingJson)
            }

            fn visit_seq<A>(self, mut sequence: A) -> Result<Self::Value, A::Error>
            where
                A: SeqAccess<'de>,
            {
                while sequence.next_element::<DuplicateRejectingJson>()?.is_some() {}
                Ok(DuplicateRejectingJson)
            }

            fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
            where
                A: MapAccess<'de>,
            {
                let mut keys = BTreeSet::new();
                while let Some(key) = map.next_key::<String>()? {
                    if !keys.insert(key.clone()) {
                        return Err(A::Error::custom(format!(
                            "duplicate JSON object key `{key}`"
                        )));
                    }
                    map.next_value::<DuplicateRejectingJson>()?;
                }
                Ok(DuplicateRejectingJson)
            }
        }

        deserializer.deserialize_any(JsonVisitor)
    }
}

fn reject_duplicate_json_keys(bytes: &[u8]) -> Result<(), ExporterPolicyError> {
    let mut deserializer = serde_json::Deserializer::from_slice(bytes);
    DuplicateRejectingJson::deserialize(&mut deserializer)
        .map_err(|error| ExporterPolicyError::new(format!("invalid strict JSON: {error}")))?;
    deserializer
        .end()
        .map_err(|error| ExporterPolicyError::new(format!("invalid strict JSON: {error}")))
}
