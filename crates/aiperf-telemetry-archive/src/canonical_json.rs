// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Canonical, float-free JSON used by archive authority objects.

use std::collections::BTreeMap;
use std::fmt::{self, Display, Formatter};

use serde::Deserialize;
use serde::de::{self, MapAccess, SeqAccess, Visitor};

/// A value accepted by `aiperf.archive.canonical-json.v1`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CanonicalJsonValue {
    /// JSON `null`.
    Null,
    /// A JSON boolean.
    Bool(bool),
    /// A signed integer serialized in minimal decimal notation.
    Integer(i128),
    /// A Unicode scalar sequence preserved without normalization.
    String(String),
    /// An authored-order JSON array.
    Array(Vec<Self>),
    /// A recursively UTF-8-byte-key-sorted JSON object.
    Object(BTreeMap<String, Self>),
}

impl CanonicalJsonValue {
    /// Parses one complete JSON value while rejecting floats and duplicate keys.
    pub fn parse(bytes: &[u8]) -> Result<Self, CanonicalJsonError> {
        let mut deserializer = serde_json::Deserializer::from_slice(bytes);
        let value = Self::deserialize(&mut deserializer)
            .map_err(|error| CanonicalJsonError::Decode(error.to_string()))?;
        deserializer
            .end()
            .map_err(|error| CanonicalJsonError::Decode(error.to_string()))?;
        Ok(value)
    }

    /// Builds an object and rejects duplicate authored keys before sorting them.
    pub fn object<I>(entries: I) -> Result<Self, CanonicalJsonError>
    where
        I: IntoIterator<Item = (String, Self)>,
    {
        let mut object = BTreeMap::new();
        for (key, value) in entries {
            if object.insert(key.clone(), value).is_some() {
                return Err(CanonicalJsonError::DuplicateKey(key));
            }
        }
        Ok(Self::Object(object))
    }

    /// Encodes the value into the exact canonical UTF-8 representation.
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut output = Vec::new();
        self.write_to(&mut output);
        output
    }

    /// Returns the value as an object when it has object shape.
    #[must_use]
    pub fn as_object(&self) -> Option<&BTreeMap<String, Self>> {
        match self {
            Self::Object(object) => Some(object),
            _ => None,
        }
    }

    /// Returns the value as a string when it has string shape.
    #[must_use]
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(value) => Some(value),
            _ => None,
        }
    }

    /// Returns the value as an integer when it has integer shape.
    #[must_use]
    pub fn as_i128(&self) -> Option<i128> {
        match self {
            Self::Integer(value) => Some(*value),
            _ => None,
        }
    }

    fn write_to(&self, output: &mut Vec<u8>) {
        match self {
            Self::Null => output.extend_from_slice(b"null"),
            Self::Bool(true) => output.extend_from_slice(b"true"),
            Self::Bool(false) => output.extend_from_slice(b"false"),
            Self::Integer(value) => output.extend_from_slice(value.to_string().as_bytes()),
            Self::String(value) => write_string(value, output),
            Self::Array(values) => {
                output.push(b'[');
                for (index, value) in values.iter().enumerate() {
                    if index != 0 {
                        output.push(b',');
                    }
                    value.write_to(output);
                }
                output.push(b']');
            }
            Self::Object(values) => {
                output.push(b'{');
                for (index, (key, value)) in values.iter().enumerate() {
                    if index != 0 {
                        output.push(b',');
                    }
                    write_string(key, output);
                    output.push(b':');
                    value.write_to(output);
                }
                output.push(b'}');
            }
        }
    }
}

impl<'de> Deserialize<'de> for CanonicalJsonValue {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        deserializer.deserialize_any(CanonicalValueVisitor)
    }
}

struct CanonicalValueVisitor;

impl<'de> Visitor<'de> for CanonicalValueVisitor {
    type Value = CanonicalJsonValue;

    fn expecting(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter.write_str("float-free JSON with unique object keys")
    }

    fn visit_unit<E>(self) -> Result<Self::Value, E> {
        Ok(CanonicalJsonValue::Null)
    }

    fn visit_none<E>(self) -> Result<Self::Value, E> {
        Ok(CanonicalJsonValue::Null)
    }

    fn visit_bool<E>(self, value: bool) -> Result<Self::Value, E> {
        Ok(CanonicalJsonValue::Bool(value))
    }

    fn visit_i64<E>(self, value: i64) -> Result<Self::Value, E> {
        Ok(CanonicalJsonValue::Integer(i128::from(value)))
    }

    fn visit_u64<E>(self, value: u64) -> Result<Self::Value, E> {
        Ok(CanonicalJsonValue::Integer(i128::from(value)))
    }

    fn visit_f64<E>(self, _value: f64) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        Err(E::custom("floating-point JSON numbers are forbidden"))
    }

    fn visit_str<E>(self, value: &str) -> Result<Self::Value, E> {
        Ok(CanonicalJsonValue::String(value.to_owned()))
    }

    fn visit_string<E>(self, value: String) -> Result<Self::Value, E> {
        Ok(CanonicalJsonValue::String(value))
    }

    fn visit_seq<A>(self, mut sequence: A) -> Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        let mut values = Vec::new();
        while let Some(value) = sequence.next_element()? {
            values.push(value);
        }
        Ok(CanonicalJsonValue::Array(values))
    }

    fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        let mut values = BTreeMap::new();
        while let Some((key, value)) = map.next_entry::<String, CanonicalJsonValue>()? {
            if values.insert(key.clone(), value).is_some() {
                return Err(de::Error::custom(format!(
                    "duplicate canonical JSON object key {key:?}"
                )));
            }
        }
        Ok(CanonicalJsonValue::Object(values))
    }
}

/// An invalid canonical JSON value or input document.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CanonicalJsonError {
    /// The input cannot be decoded under the canonical profile.
    Decode(String),
    /// An object builder received the same key more than once.
    DuplicateKey(String),
}

impl Display for CanonicalJsonError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Decode(message) => write!(formatter, "invalid canonical JSON: {message}"),
            Self::DuplicateKey(key) => {
                write!(formatter, "duplicate canonical JSON object key {key:?}")
            }
        }
    }
}

impl std::error::Error for CanonicalJsonError {}

fn write_string(value: &str, output: &mut Vec<u8>) {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    output.push(b'\"');
    for character in value.chars() {
        match character {
            '\"' => output.extend_from_slice(b"\\\""),
            '\\' => output.extend_from_slice(b"\\\\"),
            '\u{0008}' => output.extend_from_slice(b"\\b"),
            '\u{000c}' => output.extend_from_slice(b"\\f"),
            '\n' => output.extend_from_slice(b"\\n"),
            '\r' => output.extend_from_slice(b"\\r"),
            '\t' => output.extend_from_slice(b"\\t"),
            '\u{0000}'..='\u{001f}' => {
                let byte = character as u8;
                output.extend_from_slice(b"\\u00");
                output.push(HEX[usize::from(byte >> 4)]);
                output.push(HEX[usize::from(byte & 0x0f)]);
            }
            _ => {
                let mut encoded = [0_u8; 4];
                output.extend_from_slice(character.encode_utf8(&mut encoded).as_bytes());
            }
        }
    }
    output.push(b'\"');
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonicalizes_recursive_order_and_escapes() {
        let input = r#"{"z":"\/é\u0001","a":{"β":2,"a":1},"array":[3,true,null]}"#.as_bytes();
        let value = CanonicalJsonValue::parse(input).unwrap();
        assert_eq!(
            value.to_bytes(),
            "{\"a\":{\"a\":1,\"β\":2},\"array\":[3,true,null],\"z\":\"/é\\u0001\"}".as_bytes()
        );
    }

    #[test]
    fn duplicate_keys_and_floats_fail_closed() {
        assert!(matches!(
            CanonicalJsonValue::parse(br#"{"a":1,"a":2}"#),
            Err(CanonicalJsonError::Decode(message)) if message.contains("duplicate")
        ));
        assert!(matches!(
            CanonicalJsonValue::parse(b"1.0"),
            Err(CanonicalJsonError::Decode(message)) if message.contains("floating-point")
        ));
    }

    #[test]
    fn arrays_preserve_order_and_negative_zero_is_minimal_integer_zero() {
        let value = CanonicalJsonValue::parse(b"[-1,0,1]").unwrap();
        assert_eq!(value.to_bytes(), b"[-1,0,1]");
        assert!(CanonicalJsonValue::parse(b"-0").is_err());
    }
}
