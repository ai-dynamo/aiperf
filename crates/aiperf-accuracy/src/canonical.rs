// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Strict semantic JSON, digest domains, and control-plane redaction.
//!
//! Evaluator score algebra is intentionally opaque to Rust, but it is not
//! unbounded or weakly parsed. This module is the transport-free
//! `aiperf-canonical-json-v1` implementation used to validate provider values
//! before they enter identities, reports, or digest inputs.

use std::cell::Cell;
use std::collections::BTreeSet;
use std::fmt::{self, Display};

use serde::de::{self, DeserializeSeed, MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::{Map, Number, Value};
use sha2::{Digest, Sha256};

/// Version label bound into every cross-language semantic JSON identity.
pub const CANONICAL_JSON_CODEC: &str = "aiperf-canonical-json-v1";

/// Resource limits applied while decoding one canonical JSON value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CanonicalJsonLimits {
    /// Maximum recursive array/object depth.
    pub max_depth: usize,
    /// Maximum total scalar and container nodes.
    pub max_nodes: usize,
    /// Maximum entries in any single array or object.
    pub max_collection_items: usize,
    /// Maximum UTF-8 bytes in one string or object key.
    pub max_string_bytes: usize,
}

impl Default for CanonicalJsonLimits {
    fn default() -> Self {
        Self {
            max_depth: 64,
            max_nodes: 65_536,
            max_collection_items: 16_384,
            max_string_bytes: 1024 * 1024,
        }
    }
}

/// Strict canonical JSON validation or encoding failure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CanonicalJsonError(String);

impl CanonicalJsonError {
    fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl Display for CanonicalJsonError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for CanonicalJsonError {}

/// A duplicate-key-free, bounded JSON value in the canonical codec domain.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CanonicalJson(Value);

impl CanonicalJson {
    /// Validate a previously constructed JSON value against default bounds.
    pub fn new(value: Value) -> Result<Self, CanonicalJsonError> {
        Self::with_limits(value, CanonicalJsonLimits::default())
    }

    /// Validate a previously constructed JSON value against explicit bounds.
    ///
    /// Duplicate keys can only be detected while parsing; callers receiving
    /// bytes must use [`Self::from_slice`].
    pub fn with_limits(
        value: Value,
        limits: CanonicalJsonLimits,
    ) -> Result<Self, CanonicalJsonError> {
        validate_value(&value, limits, 0, &Cell::new(0))?;
        Ok(Self(value))
    }

    /// Strictly decode bytes, rejecting duplicate keys and trailing input.
    pub fn from_slice(
        bytes: &[u8],
        limits: CanonicalJsonLimits,
    ) -> Result<Self, CanonicalJsonError> {
        let nodes = Cell::new(0);
        let mut deserializer = serde_json::Deserializer::from_slice(bytes);
        let value = StrictValueSeed {
            limits,
            nodes: &nodes,
            depth: 0,
        }
        .deserialize(&mut deserializer)
        .map_err(|error| CanonicalJsonError::new(error.to_string()))?;
        deserializer
            .end()
            .map_err(|error| CanonicalJsonError::new(error.to_string()))?;
        Ok(Self(value))
    }

    /// Borrow the validated semantic value.
    pub fn value(&self) -> &Value {
        &self.0
    }

    /// Consume the wrapper and return the semantic value.
    pub fn into_value(self) -> Value {
        self.0
    }

    /// Encode deterministic canonical bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut output = Vec::new();
        write_canonical(&self.0, &mut output);
        output
    }

    /// Compute the normalized semantic-result digest.
    pub fn normalized_result_sha256(&self) -> String {
        sha256_hex(&self.to_bytes())
    }
}

impl Serialize for CanonicalJson {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.0.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for CanonicalJson {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let nodes = Cell::new(0);
        let value = StrictValueSeed {
            limits: CanonicalJsonLimits::default(),
            nodes: &nodes,
            depth: 0,
        }
        .deserialize(deserializer)?;
        Ok(Self(value))
    }
}

/// Hash exact bytes for an `artifact_content_sha256` domain field.
pub fn artifact_content_sha256(bytes: &[u8]) -> String {
    sha256_hex(bytes)
}

/// Hash already-normalized semantic bytes for a `normalized_result_sha256` field.
pub fn normalized_result_sha256(value: &CanonicalJson) -> String {
    value.normalized_result_sha256()
}

/// Compute a lowercase SHA-256 digest.
pub fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut output = String::with_capacity(64);
    for byte in digest {
        use std::fmt::Write as _;
        let _ = write!(output, "{byte:02x}");
    }
    output
}

/// Return whether a string is an exact lowercase SHA-256 digest.
pub fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

/// Reject connection authority and ordinary credentials in authored control JSON.
///
/// Inert URL text inside benchmark content remains legal because validation is
/// based on authority-bearing field names, not on string shape. Restricted
/// inference bodies and scoped local-proxy grants use dedicated typed DTOs and
/// must never be represented in this generic authored value.
pub fn validate_no_secret_control_value(value: &CanonicalJson) -> Result<(), CanonicalJsonError> {
    validate_no_secret_value(value.value(), "$", false)
}

/// Reject credentials and explicit upstream authority while permitting inert
/// URL-shaped benchmark content inside a typed operation payload.
pub fn validate_no_secret_host_payload(value: &CanonicalJson) -> Result<(), CanonicalJsonError> {
    validate_no_secret_value(value.value(), "$", true)
}

/// Produce a report-safe diagnostic without echoing authority or credential material.
pub fn redact_diagnostic(message: &str) -> String {
    const SENSITIVE_MARKERS: &[&str] = &[
        "http://",
        "https://",
        "authorization",
        "bearer ",
        "api_key",
        "api-key",
        "access_token",
        "token=",
        "password",
        "credential",
        "secret",
        "x-amz-signature",
        "sig=",
    ];
    let lowercase = message.to_ascii_lowercase();
    if SENSITIVE_MARKERS
        .iter()
        .any(|marker| lowercase.contains(marker))
    {
        return "[redacted sensitive evaluator diagnostic]".to_string();
    }
    const MAX_DIAGNOSTIC_BYTES: usize = 2_048;
    if message.len() <= MAX_DIAGNOSTIC_BYTES {
        return message.to_string();
    }
    let mut end = MAX_DIAGNOSTIC_BYTES;
    while !message.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}…[truncated]", &message[..end])
}

fn validate_no_secret_value(
    value: &Value,
    path: &str,
    allow_inert_url_field: bool,
) -> Result<(), CanonicalJsonError> {
    const FORBIDDEN_FIELDS: &[&str] = &[
        "api_key",
        "authorization",
        "base_url",
        "bearer_token",
        "client_secret",
        "credential",
        "credentials",
        "endpoint",
        "endpoint_url",
        "headers",
        "password",
        "proxy_url",
        "signed_url",
        "token",
        "upstream_url",
        "url",
    ];
    match value {
        Value::Object(entries) => {
            for (key, child) in entries {
                let normalized = key.to_ascii_lowercase().replace('-', "_");
                if FORBIDDEN_FIELDS.contains(&normalized.as_str())
                    && !(allow_inert_url_field && normalized == "url")
                {
                    return Err(CanonicalJsonError::new(format!(
                        "control field {path}.{key} may carry connection authority or a credential"
                    )));
                }
                validate_no_secret_value(child, &format!("{path}.{key}"), allow_inert_url_field)?;
            }
        }
        Value::Array(items) => {
            for (index, child) in items.iter().enumerate() {
                validate_no_secret_value(
                    child,
                    &format!("{path}[{index}]"),
                    allow_inert_url_field,
                )?;
            }
        }
        _ => {}
    }
    Ok(())
}

fn validate_value(
    value: &Value,
    limits: CanonicalJsonLimits,
    depth: usize,
    nodes: &Cell<usize>,
) -> Result<(), CanonicalJsonError> {
    count_node(limits, nodes).map_err(CanonicalJsonError::new)?;
    if depth > limits.max_depth {
        return Err(CanonicalJsonError::new(format!(
            "canonical JSON depth {depth} exceeded {}",
            limits.max_depth
        )));
    }
    match value {
        Value::String(text) => validate_string(text, limits).map_err(CanonicalJsonError::new),
        Value::Array(items) => {
            validate_collection_len(items.len(), limits).map_err(CanonicalJsonError::new)?;
            for child in items {
                validate_value(child, limits, depth + 1, nodes)?;
            }
            Ok(())
        }
        Value::Object(entries) => {
            validate_collection_len(entries.len(), limits).map_err(CanonicalJsonError::new)?;
            for (key, child) in entries {
                validate_string(key, limits).map_err(CanonicalJsonError::new)?;
                validate_value(child, limits, depth + 1, nodes)?;
            }
            Ok(())
        }
        Value::Number(number) => validate_number(number).map_err(CanonicalJsonError::new),
        Value::Null | Value::Bool(_) => Ok(()),
    }
}

fn validate_number(number: &Number) -> Result<(), String> {
    if number.as_i64().is_some() || number.as_u64().is_some() {
        return Ok(());
    }
    if number.as_f64().is_some_and(f64::is_finite) {
        Ok(())
    } else {
        Err("canonical JSON number was non-finite or outside the 64-bit domain".to_string())
    }
}

fn validate_collection_len(len: usize, limits: CanonicalJsonLimits) -> Result<(), String> {
    if len > limits.max_collection_items {
        Err(format!(
            "canonical JSON collection length {len} exceeded {}",
            limits.max_collection_items
        ))
    } else {
        Ok(())
    }
}

fn validate_string(text: &str, limits: CanonicalJsonLimits) -> Result<(), String> {
    if text.len() > limits.max_string_bytes {
        Err(format!(
            "canonical JSON string length {} exceeded {} bytes",
            text.len(),
            limits.max_string_bytes
        ))
    } else {
        Ok(())
    }
}

fn count_node(limits: CanonicalJsonLimits, nodes: &Cell<usize>) -> Result<(), String> {
    let next = nodes
        .get()
        .checked_add(1)
        .ok_or_else(|| "canonical JSON node count overflow".to_string())?;
    if next > limits.max_nodes {
        return Err(format!(
            "canonical JSON node count exceeded {}",
            limits.max_nodes
        ));
    }
    nodes.set(next);
    Ok(())
}

fn write_canonical(value: &Value, output: &mut Vec<u8>) {
    match value {
        Value::Null => output.extend_from_slice(b"null"),
        Value::Bool(true) => output.extend_from_slice(b"true"),
        Value::Bool(false) => output.extend_from_slice(b"false"),
        Value::Number(number) => {
            if number.as_f64() == Some(-0.0) && number.to_string().starts_with('-') {
                output.push(b'0');
            } else {
                output.extend_from_slice(number.to_string().as_bytes());
            }
        }
        Value::String(text) => {
            let encoded = serde_json::to_string(text).expect("a Rust string is valid JSON");
            output.extend_from_slice(encoded.as_bytes());
        }
        Value::Array(items) => {
            output.push(b'[');
            for (index, item) in items.iter().enumerate() {
                if index != 0 {
                    output.push(b',');
                }
                write_canonical(item, output);
            }
            output.push(b']');
        }
        Value::Object(entries) => {
            output.push(b'{');
            let mut keys = entries.keys().collect::<Vec<_>>();
            keys.sort_by(|left, right| left.as_bytes().cmp(right.as_bytes()));
            for (index, key) in keys.into_iter().enumerate() {
                if index != 0 {
                    output.push(b',');
                }
                let encoded = serde_json::to_string(key).expect("a Rust string is valid JSON");
                output.extend_from_slice(encoded.as_bytes());
                output.push(b':');
                write_canonical(&entries[key], output);
            }
            output.push(b'}');
        }
    }
}

struct StrictValueSeed<'a> {
    limits: CanonicalJsonLimits,
    nodes: &'a Cell<usize>,
    depth: usize,
}

impl<'de> DeserializeSeed<'de> for StrictValueSeed<'_> {
    type Value = Value;

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: Deserializer<'de>,
    {
        count_node(self.limits, self.nodes).map_err(de::Error::custom)?;
        if self.depth > self.limits.max_depth {
            return Err(de::Error::custom(format!(
                "canonical JSON depth {} exceeded {}",
                self.depth, self.limits.max_depth
            )));
        }
        deserializer.deserialize_any(StrictValueVisitor {
            limits: self.limits,
            nodes: self.nodes,
            depth: self.depth,
        })
    }
}

struct StrictValueVisitor<'a> {
    limits: CanonicalJsonLimits,
    nodes: &'a Cell<usize>,
    depth: usize,
}

impl<'de> Visitor<'de> for StrictValueVisitor<'_> {
    type Value = Value;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a bounded canonical JSON value")
    }

    fn visit_unit<E>(self) -> Result<Self::Value, E> {
        Ok(Value::Null)
    }

    fn visit_none<E>(self) -> Result<Self::Value, E> {
        Ok(Value::Null)
    }

    fn visit_bool<E>(self, value: bool) -> Result<Self::Value, E> {
        Ok(Value::Bool(value))
    }

    fn visit_i64<E>(self, value: i64) -> Result<Self::Value, E> {
        Ok(Value::Number(Number::from(value)))
    }

    fn visit_u64<E>(self, value: u64) -> Result<Self::Value, E> {
        Ok(Value::Number(Number::from(value)))
    }

    fn visit_f64<E>(self, value: f64) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        Number::from_f64(value)
            .map(Value::Number)
            .ok_or_else(|| E::custom("canonical JSON does not permit non-finite floats"))
    }

    fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        self.visit_string(value.to_string())
    }

    fn visit_string<E>(self, value: String) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        validate_string(&value, self.limits).map_err(E::custom)?;
        Ok(Value::String(value))
    }

    fn visit_seq<A>(self, mut sequence: A) -> Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        let mut items = Vec::new();
        while let Some(value) = sequence.next_element_seed(StrictValueSeed {
            limits: self.limits,
            nodes: self.nodes,
            depth: self.depth + 1,
        })? {
            if items.len() == self.limits.max_collection_items {
                return Err(de::Error::custom(format!(
                    "canonical JSON collection exceeded {} items",
                    self.limits.max_collection_items
                )));
            }
            items.push(value);
        }
        Ok(Value::Array(items))
    }

    fn visit_map<A>(self, mut object: A) -> Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        let mut entries = Map::new();
        let mut keys = BTreeSet::new();
        while let Some(key) = object.next_key::<String>()? {
            validate_string(&key, self.limits).map_err(de::Error::custom)?;
            if !keys.insert(key.clone()) {
                return Err(de::Error::custom(format!(
                    "canonical JSON object contained duplicate key {key:?}"
                )));
            }
            if entries.len() == self.limits.max_collection_items {
                return Err(de::Error::custom(format!(
                    "canonical JSON object exceeded {} entries",
                    self.limits.max_collection_items
                )));
            }
            let value = object.next_value_seed(StrictValueSeed {
                limits: self.limits,
                nodes: self.nodes,
                depth: self.depth + 1,
            })?;
            entries.insert(key, value);
        }
        Ok(Value::Object(entries))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_duplicate_keys_before_serde_can_overwrite_them() {
        let error =
            CanonicalJson::from_slice(br#"{"score":1,"score":0}"#, CanonicalJsonLimits::default())
                .unwrap_err();
        assert!(error.to_string().contains("duplicate key \"score\""));
    }

    #[test]
    fn canonical_bytes_sort_utf8_keys_and_normalize_negative_zero() {
        let value = CanonicalJson::from_slice(
            r#"{"z":-0.0,"é":2,"a":1.0}"#.as_bytes(),
            CanonicalJsonLimits::default(),
        )
        .unwrap();
        assert_eq!(
            String::from_utf8(value.to_bytes()).unwrap(),
            r#"{"a":1.0,"z":0,"é":2}"#
        );
    }

    #[test]
    fn digest_domains_have_stable_goldens() {
        let value = CanonicalJson::from_slice(
            br#"{"score":{"C":1},"values":[0,1]}"#,
            CanonicalJsonLimits::default(),
        )
        .unwrap();
        assert_eq!(
            value.normalized_result_sha256(),
            "1df185c8ab12b0b31a84b8ad9c5fc3f7b5c15504241df069cb57e94b153bbbd7"
        );
        assert_eq!(
            artifact_content_sha256(b"provider artifact\n"),
            "9d587291a128331c099b4d183a3ac0b35d1f2e759dbcae3575936f44d3fef591"
        );
    }

    #[test]
    fn no_secret_validation_is_field_aware_and_diagnostics_fail_closed() {
        let inert = CanonicalJson::new(serde_json::json!({
            "question": "read https://example.invalid without dereferencing it",
            "url": "https://example.invalid/semantic-content"
        }))
        .unwrap();
        assert!(validate_no_secret_control_value(&inert).is_err());
        validate_no_secret_host_payload(&inert).unwrap();

        let authority = CanonicalJson::new(serde_json::json!({
            "base_url": "https://model.invalid"
        }))
        .unwrap();
        assert!(validate_no_secret_control_value(&authority).is_err());
        assert_eq!(
            redact_diagnostic("request failed at https://secret.invalid?token=sentinel"),
            "[redacted sensitive evaluator diagnostic]"
        );
    }

    #[test]
    fn applies_depth_node_collection_and_string_bounds() {
        let limits = CanonicalJsonLimits {
            max_depth: 1,
            max_nodes: 4,
            max_collection_items: 2,
            max_string_bytes: 3,
        };
        assert!(CanonicalJson::from_slice(br#"[[0]]"#, limits).is_err());
        assert!(CanonicalJson::from_slice(br#"[0,1,2]"#, limits).is_err());
        assert!(CanonicalJson::from_slice(br#""four""#, limits).is_err());
    }
}
