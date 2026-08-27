// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Strict exporter-observable policy parsing and canonical identity.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::path::{Component, Path};

use serde::de::{Error as _, MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Serialize};

use crate::plugin_stats::ExporterMember;

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
    mode: ExporterPolicyMode,
    receiver_transport_fields_removed: Vec<ExporterTransportFieldsRemoved>,
    scenarios: Vec<ExporterObservableScenario>,
    schema_version: u8,
}

impl ExporterObservablePolicyV1 {
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
    let policy: ExporterObservablePolicyV1 = serde_json::from_slice(bytes).map_err(|error| {
        ExporterPolicyError::new(format!("cannot decode exporter observable policy: {error}"))
    })?;
    validate_exporter_policy(&policy, authenticated_receiver_protocols)?;
    if policy.canonical_bytes()? != bytes {
        return Err(ExporterPolicyError::new(
            "exporter observable policy is not exact RFC 8785 JCS plus newline",
        ));
    }
    Ok(policy)
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
            .filter(|end| selection.length > 0 && *end <= raw_length)
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

fn expected_for_member<'a>(
    mode: ExporterPolicyMode,
    member: ExporterMember,
    slot: &'a ExporterProvenanceSlot,
) -> Result<&'a ExporterEncodedValue, ExporterPolicyError> {
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
    if encoded.len() % 2 != 0
        || !encoded
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(ExporterPolicyError::new(
            "hex bytes must use even-length lower-case hexadecimal",
        ));
    }
    encoded
        .as_bytes()
        .chunks_exact(2)
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
            if encoded.len() % 2 != 0
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
