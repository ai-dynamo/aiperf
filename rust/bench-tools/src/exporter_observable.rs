// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Exact raw-observable validation shared by exporter evidence consumers.

use std::collections::BTreeSet;
use std::path::{Component, Path};

use serde::de::{Error as _, MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Serialize};

/// The retained object type represented by an artifact-tree entry.
#[derive(Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ArtifactTreeKind {
    /// A directory with no retained content.
    EmptyDirectory,
    /// A retained regular file.
    RegularFile,
}

/// One canonical entry in an artifact-tree raw observable.
#[derive(Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ArtifactTreeEntry {
    /// BLAKE3 digest of the retained entry content.
    pub blake3: String,
    /// Retained object type.
    pub kind: ArtifactTreeKind,
    /// Retained content length in bytes.
    pub length: u64,
    /// Normalized path relative to the retained artifact root.
    pub path: String,
}

/// Exact accepted-body identity retained in a receiver transcript.
#[derive(Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ReceiverBody {
    /// Body encoding marker, exactly `bytes`.
    pub encoding: ReceiverBodyEncoding,
    /// Exact accepted body length.
    pub length: u64,
    /// BLAKE3 digest of the exact accepted body bytes.
    pub blake3: String,
}

/// Closed receiver-body encoding vocabulary.
#[derive(Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ReceiverBodyEncoding {
    /// Body identity covers exact decoder-accepted bytes.
    Bytes,
}

/// One decoder-accepted operation in a receiver transcript.
#[derive(Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ReceiverTranscriptEntry {
    /// Unique dense acceptance sequence.
    pub sequence: u64,
    /// Protocol-canonical operation name.
    pub operation: String,
    /// Protocol-canonical logical destination.
    pub target: String,
    /// Strictly key-sorted exact UTF-8 key/value pairs.
    pub metadata: Vec<[String; 2]>,
    /// Exact accepted-body identity.
    pub body: ReceiverBody,
}

/// Validate protocol-canonical receiver metadata before transcript formation.
pub fn validate_receiver_metadata(
    metadata: &[[String; 2]],
) -> Result<(), Box<dyn std::error::Error>> {
    let mut previous_key = None;
    for pair in metadata {
        let [key, value] = pair;
        if key.is_empty()
            || key.contains('\0')
            || value.contains('\0')
            || key.bytes().any(|byte| byte.is_ascii_uppercase())
            || previous_key.is_some_and(|previous| previous >= key.as_str())
        {
            return Err(
                "receiver transcript metadata keys must be lower-case, sorted, and unique".into(),
            );
        }
        previous_key = Some(key.as_str());
    }
    Ok(())
}

/// Rejects a JSON document containing duplicate object keys.
pub fn reject_duplicate_json_keys(bytes: &[u8]) -> Result<(), Box<dyn std::error::Error>> {
    let mut deserializer = serde_json::Deserializer::from_slice(bytes);
    DuplicateRejectingJson::deserialize(&mut deserializer)?;
    deserializer.end()?;
    Ok(())
}

/// Validates a normalized path used by an artifact-tree observable.
pub fn validate_artifact_tree_path(path: &str) -> Result<(), Box<dyn std::error::Error>> {
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
        return Err("exporter artifact selector path is not normalized".into());
    }
    Ok(())
}

/// Parses an exact RFC 8785 JCS artifact-tree manifest with its trailing newline.
pub fn parse_artifact_tree_observable(
    bytes: &[u8],
) -> Result<Vec<ArtifactTreeEntry>, Box<dyn std::error::Error>> {
    reject_duplicate_json_keys(bytes)?;
    let entries: Vec<ArtifactTreeEntry> = serde_json::from_slice(bytes)?;
    let empty_digest = format!("blake3:{}", blake3::hash(b""));
    let mut previous_path = None;
    for entry in &entries {
        validate_artifact_tree_path(&entry.path)?;
        if previous_path.is_some_and(|previous| previous >= entry.path.as_str()) {
            return Err("artifact-tree paths must be sorted and unique".into());
        }
        previous_path = Some(entry.path.as_str());
        if !is_lower_blake3(&entry.blake3) {
            return Err("artifact-tree digest must be lower-case BLAKE3".into());
        }
        if entry.kind == ArtifactTreeKind::EmptyDirectory
            && (entry.length != 0 || entry.blake3 != empty_digest)
        {
            return Err("empty artifact-tree directory has nonempty content identity".into());
        }
    }
    let mut canonical = serde_json_canonicalizer::to_vec(&entries)?;
    canonical.push(b'\n');
    if canonical != bytes {
        return Err("artifact-tree observable is not exact RFC 8785 JCS plus newline".into());
    }
    Ok(entries)
}

/// Validate exact captured-stream bytes and return their canonical digest.
///
/// Captured streams have no JSON wrapper or trailing-newline rule. Empty bytes
/// are accepted only when the frozen scenario explicitly permits them.
pub fn validate_captured_stream_observable(
    bytes: &[u8],
    allows_empty: bool,
) -> Result<String, Box<dyn std::error::Error>> {
    if bytes.is_empty() && !allows_empty {
        return Err("captured-stream observable is empty but the scenario forbids it".into());
    }
    Ok(format!("blake3:{}", blake3::hash(bytes)))
}

/// Parse an exact RFC 8785 JCS receiver transcript with its trailing newline.
pub fn parse_receiver_transcript_observable(
    bytes: &[u8],
    allows_empty: bool,
) -> Result<Vec<ReceiverTranscriptEntry>, Box<dyn std::error::Error>> {
    reject_duplicate_json_keys(bytes)?;
    let entries: Vec<ReceiverTranscriptEntry> = serde_json::from_slice(bytes)?;
    if entries.is_empty() && !allows_empty {
        return Err("receiver transcript is empty but the scenario forbids it".into());
    }
    for (expected_sequence, entry) in entries.iter().enumerate() {
        if entry.sequence != expected_sequence as u64 {
            return Err("receiver transcript sequences must be dense from zero".into());
        }
        if entry.operation.is_empty()
            || entry.target.is_empty()
            || entry.operation.contains('\0')
            || entry.target.contains('\0')
        {
            return Err("receiver transcript operation or target is invalid".into());
        }
        validate_receiver_metadata(&entry.metadata)?;
        if !is_lower_blake3(&entry.body.blake3) {
            return Err("receiver transcript body digest must be lower-case BLAKE3".into());
        }
    }
    let mut canonical = serde_json_canonicalizer::to_vec(&entries)?;
    canonical.push(b'\n');
    if canonical != bytes {
        return Err("receiver transcript is not exact RFC 8785 JCS plus newline".into());
    }
    Ok(entries)
}

/// Validate the exact accepted bodies retained beside a receiver transcript.
pub fn validate_receiver_transcript_bodies(
    entries: &[ReceiverTranscriptEntry],
    bodies: &[&[u8]],
) -> Result<(), Box<dyn std::error::Error>> {
    if entries.len() != bodies.len() {
        return Err("retained receiver body count does not match the transcript".into());
    }
    for (entry, body) in entries.iter().zip(bodies) {
        let body_length = u64::try_from(body.len())?;
        let body_digest = format!("blake3:{}", blake3::hash(body));
        if entry.body.length != body_length || entry.body.blake3 != body_digest {
            return Err("retained receiver body does not match its transcript identity".into());
        }
    }
    Ok(())
}

/// Reports whether a value is a lower-case, `blake3:`-prefixed digest.
pub fn is_lower_blake3(value: &str) -> bool {
    value.strip_prefix("blake3:").is_some_and(|hex| {
        hex.len() == 64
            && hex
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    })
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

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
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
