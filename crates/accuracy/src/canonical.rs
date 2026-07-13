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
        validate_integer_lexemes(bytes)?;
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
    validate_no_secret_value(value.value(), &mut Vec::new(), false)
}

/// Reject credentials and explicit upstream authority in a host-operation payload.
///
/// The sole `url` exception is a strict raster `data:` URI at the exact
/// OpenAI-compatible `messages|input[*].content[*].image_url.url` shape. This
/// keeps inline multimodal compatibility without granting the model server an
/// attacker-selected network or filesystem locator.
pub fn validate_no_secret_host_payload(value: &CanonicalJson) -> Result<(), CanonicalJsonError> {
    validate_no_secret_value(value.value(), &mut Vec::new(), true)
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

#[derive(Clone, Copy)]
enum ControlPathSegment<'a> {
    Field(&'a str),
    Index(usize),
}

fn validate_no_secret_value<'a>(
    value: &'a Value,
    path: &mut Vec<ControlPathSegment<'a>>,
    allow_inline_image_data_url: bool,
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
                let safe_inline_image_url = allow_inline_image_data_url
                    && normalized == "url"
                    && is_inline_image_url_path(path)
                    && child.as_str().is_some_and(is_safe_inline_image_data_url);
                if FORBIDDEN_FIELDS.contains(&normalized.as_str()) && !safe_inline_image_url {
                    return Err(CanonicalJsonError::new(format!(
                        "control field {} may carry connection authority or a credential",
                        display_control_path(path, Some(key))
                    )));
                }
                path.push(ControlPathSegment::Field(key));
                validate_no_secret_value(child, path, allow_inline_image_data_url)?;
                path.pop();
            }
        }
        Value::Array(items) => {
            for (index, child) in items.iter().enumerate() {
                path.push(ControlPathSegment::Index(index));
                validate_no_secret_value(child, path, allow_inline_image_data_url)?;
                path.pop();
            }
        }
        _ => {}
    }
    Ok(())
}

fn is_inline_image_url_path(path: &[ControlPathSegment<'_>]) -> bool {
    matches!(
        path,
        [
            ControlPathSegment::Field("messages" | "input"),
            ControlPathSegment::Index(_),
            ControlPathSegment::Field("content"),
            ControlPathSegment::Index(_),
            ControlPathSegment::Field("image_url"),
        ]
    )
}

/// Validate the sole inline locator admitted by the stock operation schema.
///
/// This is the executable counterpart of `_INLINE_RASTER_DATA_URI_PATTERN`.
/// Request, response,
/// and stream validators share it so their advertised schema fingerprints
/// cannot conceal a broader transport-authority surface.
pub fn is_safe_inline_image_data_url(value: &str) -> bool {
    const PREFIXES: &[&str] = &[
        "data:image/gif;base64,",
        "data:image/jpeg;base64,",
        "data:image/png;base64,",
        "data:image/webp;base64,",
    ];
    if value.len() > CanonicalJsonLimits::default().max_string_bytes {
        return false;
    }
    let Some(encoded) = PREFIXES
        .iter()
        .find_map(|prefix| value.strip_prefix(prefix))
    else {
        return false;
    };
    if encoded.is_empty() || encoded.len() % 4 != 0 {
        return false;
    }
    let padding = encoded
        .as_bytes()
        .iter()
        .rev()
        .take_while(|byte| **byte == b'=')
        .count();
    padding <= 2
        && encoded[..encoded.len() - padding]
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'+' | b'/'))
        && encoded[..encoded.len() - padding]
            .bytes()
            .any(|byte| byte != b'=')
}

fn display_control_path(path: &[ControlPathSegment<'_>], next_field: Option<&str>) -> String {
    use std::fmt::Write as _;

    let mut output = "$".to_string();
    for segment in path {
        match segment {
            ControlPathSegment::Field(field) => {
                let _ = write!(output, ".{field}");
            }
            ControlPathSegment::Index(index) => {
                let _ = write!(output, "[{index}]");
            }
        }
    }
    if let Some(field) = next_field {
        let _ = write!(output, ".{field}");
    }
    output
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

fn validate_integer_lexemes(bytes: &[u8]) -> Result<(), CanonicalJsonError> {
    let mut index = 0;
    while index < bytes.len() {
        match bytes[index] {
            b'"' => {
                index += 1;
                while index < bytes.len() {
                    match bytes[index] {
                        b'\\' => index = index.saturating_add(2),
                        b'"' => {
                            index += 1;
                            break;
                        }
                        _ => index += 1,
                    }
                }
            }
            b'-' | b'0'..=b'9' => {
                let start = index;
                if bytes[index] == b'-' {
                    index += 1;
                }
                while index < bytes.len() && bytes[index].is_ascii_digit() {
                    index += 1;
                }
                let mut is_integer = true;
                if index < bytes.len() && bytes[index] == b'.' {
                    is_integer = false;
                    index += 1;
                    while index < bytes.len() && bytes[index].is_ascii_digit() {
                        index += 1;
                    }
                }
                if index < bytes.len() && matches!(bytes[index], b'e' | b'E') {
                    is_integer = false;
                    index += 1;
                    if index < bytes.len() && matches!(bytes[index], b'+' | b'-') {
                        index += 1;
                    }
                    while index < bytes.len() && bytes[index].is_ascii_digit() {
                        index += 1;
                    }
                }
                if is_integer {
                    let lexeme = std::str::from_utf8(&bytes[start..index]).map_err(|_| {
                        CanonicalJsonError::new("canonical JSON was not valid UTF-8")
                    })?;
                    let in_range = if lexeme.starts_with('-') {
                        lexeme.parse::<i64>().is_ok()
                    } else {
                        lexeme.parse::<u64>().is_ok()
                    };
                    if !in_range {
                        return Err(CanonicalJsonError::new(
                            "canonical JSON integer was outside the signed/unsigned 64-bit domain",
                        ));
                    }
                }
            }
            _ => index += 1,
        }
    }
    Ok(())
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
            if number.as_f64() == Some(0.0) {
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
        // serde_json's `arbitrary_precision` feature (enabled workspace-wide via
        // aiperf-graph) does not route numbers through `visit_f64`/`visit_i64`
        // when a `deserialize_any` visitor is used: it delivers each number as a
        // single-entry map keyed by a private sentinel whose value is the raw
        // lexeme. Without reconstructing the `Number` here, a byte-decoded float
        // (e.g. an authored `temperature: 0.0`) would become a `Value::Object`
        // and fail every downstream `is_number()` check. The sentinel key cannot
        // be produced by ordinary decoded JSON, so this branch is inert when the
        // feature is off.
        const NUMBER_SENTINEL: &str = "$serde_json::private::Number";
        let mut entries = Map::new();
        let mut keys = BTreeSet::new();
        let mut first = true;
        while let Some(key) = object.next_key::<String>()? {
            if first && key == NUMBER_SENTINEL {
                let lexeme: String = object.next_value()?;
                if object.next_key::<String>()?.is_some() {
                    return Err(de::Error::custom(
                        "arbitrary-precision number carried unexpected extra keys",
                    ));
                }
                let number = serde_json::from_str::<Number>(&lexeme)
                    .map_err(|_| de::Error::custom("canonical JSON number lexeme was invalid"))?;
                validate_number(&number).map_err(de::Error::custom)?;
                return Ok(Value::Number(number));
            }
            first = false;
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
    fn canonical_bytes_sort_utf8_keys_and_normalize_both_float_zeros() {
        let value = CanonicalJson::from_slice(
            r#"{"z":-0.0,"é":2,"a":1.0,"p":0.0}"#.as_bytes(),
            CanonicalJsonLimits::default(),
        )
        .unwrap();
        assert_eq!(
            String::from_utf8(value.to_bytes()).unwrap(),
            r#"{"a":1.0,"p":0,"z":0,"é":2}"#
        );
    }

    #[test]
    fn public_score_schema_matches_python_canonical_float_zero_golden() {
        let schema = CanonicalJson::from_slice(
            br#"{"$schema":"https://json-schema.org/draft/2020-12/schema","additionalProperties":false,"properties":{"value":{"maximum":1.0,"minimum":0.0,"type":"number"}},"required":["value"],"type":"object"}"#,
            CanonicalJsonLimits::default(),
        )
        .unwrap();
        assert_eq!(
            schema.normalized_result_sha256(),
            "64440005a209339a632787d5fe39b01c89a120a3a8f64194aa02fdbd4fa42cb9"
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
        }))
        .unwrap();
        validate_no_secret_control_value(&inert).unwrap();
        validate_no_secret_host_payload(&inert).unwrap();

        let inline_image = CanonicalJson::new(serde_json::json!({
            "messages": [{
                "role": "user",
                "content": [{
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,aW5lcnQ="}
                }]
            }]
        }))
        .unwrap();
        assert!(validate_no_secret_control_value(&inline_image).is_err());
        validate_no_secret_host_payload(&inline_image).unwrap();

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
    fn host_payload_rejects_remote_or_misplaced_url_authority() {
        for url in [
            "http://169.254.169.254/latest/meta-data/",
            "https://model.invalid/image.png",
            "file:///etc/passwd",
            "gopher://127.0.0.1:11211/_stats",
            "data:image/svg+xml;base64,PHN2Zz48L3N2Zz4=",
            "data:image/png;base64,not-base64!",
        ] {
            let payload = CanonicalJson::new(serde_json::json!({
                "messages": [{
                    "role": "user",
                    "content": [{
                        "type": "image_url",
                        "image_url": {"url": url}
                    }]
                }]
            }))
            .unwrap();
            assert!(validate_no_secret_host_payload(&payload).is_err(), "{url}");
        }

        let misplaced = CanonicalJson::new(serde_json::json!({
            "metadata": {"image_url": {"url": "data:image/png;base64,aW5lcnQ="}}
        }))
        .unwrap();
        assert!(validate_no_secret_host_payload(&misplaced).is_err());
    }

    #[test]
    fn inline_image_data_url_validator_matches_the_schema_boundary() {
        assert!(is_safe_inline_image_data_url(
            "data:image/png;base64,aW5lcnQ="
        ));
        assert!(is_safe_inline_image_data_url("data:image/jpeg;base64,AAAA"));
        assert!(!is_safe_inline_image_data_url("data:image/png;base64,"));
        assert!(!is_safe_inline_image_data_url(
            "data:image/svg+xml;base64,PHN2Zz48L3N2Zz4="
        ));
        assert!(!is_safe_inline_image_data_url(
            "data:image/png;base64,not-base64!"
        ));
        assert!(!is_safe_inline_image_data_url(&format!(
            "data:image/png;base64,{}",
            "A".repeat(CanonicalJsonLimits::default().max_string_bytes)
        )));
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

    #[test]
    fn rejects_out_of_domain_integer_lexemes_without_rejecting_finite_floats() {
        let limits = CanonicalJsonLimits::default();
        CanonicalJson::from_slice(b"18446744073709551615", limits).unwrap();
        CanonicalJson::from_slice(b"-9223372036854775808", limits).unwrap();
        assert!(CanonicalJson::from_slice(b"18446744073709551616", limits).is_err());
        assert!(CanonicalJson::from_slice(b"-9223372036854775809", limits).is_err());
        CanonicalJson::from_slice(b"18446744073709551616.0", limits).unwrap();
        CanonicalJson::from_slice(b"1.8446744073709552e19", limits).unwrap();
    }
}
