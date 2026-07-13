// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Encoding-independent logical-row and projection evidence.

use std::collections::BTreeSet;
use std::fmt::{self, Display, Formatter};

use crate::{Digest, domain_digest};

const LOGICAL_ROW_MAGIC: &[u8; 10] = b"AIPERFROW\x01";

/// Closed archive table IDs used by WAL declarations and manifest keys.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[repr(u8)]
pub enum TableId {
    /// Issued scrape attempts.
    Attempts = 1,
    /// Parsed family metadata.
    Families = 2,
    /// Structured metric points.
    Samples = 3,
    /// Lifecycle and topology markers.
    Markers = 4,
    /// Exact loss ranges and saturation snapshots.
    Losses = 5,
    /// Per-frame references to shared encrypted raw objects.
    RawReferences = 6,
}

impl TableId {
    /// Decodes a frozen numeric table ID.
    pub fn from_u8(value: u8) -> Result<Self, LogicalRowError> {
        match value {
            1 => Ok(Self::Attempts),
            2 => Ok(Self::Families),
            3 => Ok(Self::Samples),
            4 => Ok(Self::Markers),
            5 => Ok(Self::Losses),
            6 => Ok(Self::RawReferences),
            _ => Err(LogicalRowError::UnknownTable(value)),
        }
    }
}

/// A descriptor-directed logical type whose physical encoding is frozen by v1.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum LogicalType {
    /// Boolean.
    Bool,
    /// Signed 8-bit integer.
    I8,
    /// Signed 16-bit integer.
    I16,
    /// Signed 32-bit integer.
    I32,
    /// Signed 64-bit integer.
    I64,
    /// Unsigned 8-bit integer.
    U8,
    /// Unsigned 16-bit integer.
    U16,
    /// Unsigned 32-bit integer.
    U32,
    /// Unsigned 64-bit integer.
    U64,
    /// Finite IEEE-754 binary64 bits.
    Float64,
    /// Signed two's-complement 128-bit decimal integer.
    Decimal128,
    /// Length-prefixed UTF-8.
    Utf8,
    /// Length-prefixed variable binary.
    Binary,
    /// Exact fixed-size binary.
    FixedBinary(usize),
    /// Logical enum string, independent of Arrow dictionary index.
    Enum,
    /// Descriptor-ordered child fields.
    Struct(Vec<LogicalField>),
    /// Ordered elements under one descriptor field.
    List(Box<LogicalField>),
    /// UTF-8 byte-sorted, duplicate-free string map.
    StringMap,
}

/// One descriptor field and its nullability.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LogicalField {
    /// Whether this field encodes the v1 null/present tag.
    pub nullable: bool,
    /// Descriptor-directed logical type.
    pub logical_type: LogicalType,
}

/// A schema authority sufficient to encode logical rows independently of Arrow.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LogicalSchema {
    /// Closed archive table.
    pub table: TableId,
    /// Checked-in table schema fingerprint.
    pub fingerprint: Digest,
    /// Descriptor-order fields.
    pub fields: Vec<LogicalField>,
}

/// One value supplied to the descriptor-directed row encoder.
#[derive(Clone, Debug, PartialEq)]
pub enum LogicalValue {
    /// A null value, legal only for a nullable descriptor field.
    Null,
    /// Boolean.
    Bool(bool),
    /// Signed integer, range-checked against its descriptor width.
    Signed(i128),
    /// Unsigned integer, range-checked against its descriptor width.
    Unsigned(u128),
    /// Finite binary64, encoded by exact bits including negative zero.
    Float64(f64),
    /// Decimal128 integer.
    Decimal128(i128),
    /// UTF-8 or enum text.
    String(String),
    /// Variable or fixed binary.
    Binary(Vec<u8>),
    /// Descriptor-ordered struct children.
    Struct(Vec<Self>),
    /// Ordered list children.
    List(Vec<Self>),
    /// Authored string-map entries; encoding sorts and rejects duplicates.
    StringMap(Vec<(String, String)>),
}

/// Exact canonical logical-row bytes and their domain-separated digest.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CanonicalLogicalRow {
    table: TableId,
    schema_fingerprint: Digest,
    bytes: Vec<u8>,
    digest: Digest,
}

impl CanonicalLogicalRow {
    /// Encodes one row under a checked-in schema authority.
    pub fn encode(
        schema: &LogicalSchema,
        values: &[LogicalValue],
    ) -> Result<Self, LogicalRowError> {
        if values.len() != schema.fields.len() {
            return Err(LogicalRowError::FieldCount {
                expected: schema.fields.len(),
                actual: values.len(),
            });
        }
        let mut bytes = Vec::new();
        bytes.extend_from_slice(LOGICAL_ROW_MAGIC);
        bytes.push(schema.table as u8);
        bytes.extend_from_slice(schema.fingerprint.as_bytes());
        bytes.extend_from_slice(
            &u32::try_from(schema.fields.len())
                .map_err(|_| LogicalRowError::LengthOverflow)?
                .to_be_bytes(),
        );
        for (field, value) in schema.fields.iter().zip(values) {
            encode_field(field, value, &mut bytes)?;
        }
        let table = [schema.table as u8];
        let digest = domain_digest(
            "aiperf.archive.logical-row.v1",
            &[schema.fingerprint.as_bytes(), &table, &bytes],
        );
        Ok(Self {
            table: schema.table,
            schema_fingerprint: schema.fingerprint,
            bytes,
            digest,
        })
    }

    /// Returns the row's archive table.
    #[must_use]
    pub const fn table(&self) -> TableId {
        self.table
    }

    /// Returns the row's checked-in schema fingerprint.
    #[must_use]
    pub const fn schema_fingerprint(&self) -> Digest {
        self.schema_fingerprint
    }

    /// Returns exact logical-row bytes.
    #[must_use]
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Returns the logical row digest.
    #[must_use]
    pub const fn digest(&self) -> Digest {
        self.digest
    }
}

/// Exact row-count and multiset evidence for one frame/table projection.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProjectionEvidence {
    /// Number of logical rows.
    pub row_count: u64,
    /// Digest of lexicographically sorted logical-row digests.
    pub logical_multiset_digest: Digest,
}

impl ProjectionEvidence {
    /// Computes order-independent projection evidence from logical rows.
    pub fn from_rows<'a, I>(rows: I) -> Result<Self, LogicalRowError>
    where
        I: IntoIterator<Item = &'a CanonicalLogicalRow>,
    {
        let mut digests: Vec<Digest> = rows.into_iter().map(CanonicalLogicalRow::digest).collect();
        digests.sort_unstable();
        let row_count =
            u64::try_from(digests.len()).map_err(|_| LogicalRowError::LengthOverflow)?;
        let fields: Vec<&[u8]> = digests
            .iter()
            .map(Digest::as_bytes)
            .map(AsRef::as_ref)
            .collect();
        Ok(Self {
            row_count,
            logical_multiset_digest: domain_digest(
                "aiperf.archive.projection-multiset.v1",
                &fields,
            ),
        })
    }

    /// Returns explicit evidence for a required zero-row projection.
    #[must_use]
    pub fn empty() -> Self {
        Self {
            row_count: 0,
            logical_multiset_digest: domain_digest("aiperf.archive.projection-multiset.v1", &[]),
        }
    }
}

/// A frame header's required table projection and exact logical evidence.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RequiredProjection {
    /// Required table.
    pub table: TableId,
    /// Exact expected logical evidence, including zero-row evidence.
    pub evidence: ProjectionEvidence,
}

/// A schema/value mismatch or non-canonical logical row.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum LogicalRowError {
    /// A frozen numeric table ID is unknown.
    UnknownTable(u8),
    /// Top-level or struct child count differs from the descriptor.
    FieldCount {
        /// Descriptor field count.
        expected: usize,
        /// Supplied field count.
        actual: usize,
    },
    /// Null was supplied for a non-null field.
    UnexpectedNull,
    /// A value variant disagrees with the descriptor type.
    TypeMismatch(&'static str),
    /// An integer does not fit the descriptor width.
    IntegerOutOfRange(&'static str),
    /// A raw non-finite Float64 attempted to cross the row boundary.
    NonFiniteFloat,
    /// Fixed binary has the wrong byte length.
    FixedBinaryLength {
        /// Descriptor byte length.
        expected: usize,
        /// Supplied byte length.
        actual: usize,
    },
    /// A map repeats a key.
    DuplicateMapKey(String),
    /// A collection length cannot be represented by its fixed width.
    LengthOverflow,
}

impl Display for LogicalRowError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownTable(value) => write!(formatter, "unknown archive table ID {value}"),
            Self::FieldCount { expected, actual } => write!(
                formatter,
                "logical row field count mismatch: expected {expected}, found {actual}"
            ),
            Self::UnexpectedNull => formatter.write_str("non-null logical field received null"),
            Self::TypeMismatch(expected) => {
                write!(formatter, "logical field expected {expected}")
            }
            Self::IntegerOutOfRange(expected) => {
                write!(formatter, "logical integer is outside {expected} range")
            }
            Self::NonFiniteFloat => {
                formatter.write_str("raw non-finite Float64 is forbidden in logical rows")
            }
            Self::FixedBinaryLength { expected, actual } => write!(
                formatter,
                "fixed binary length mismatch: expected {expected}, found {actual}"
            ),
            Self::DuplicateMapKey(key) => write!(formatter, "duplicate map key {key:?}"),
            Self::LengthOverflow => formatter.write_str("logical collection length overflow"),
        }
    }
}

impl std::error::Error for LogicalRowError {}

fn encode_field(
    field: &LogicalField,
    value: &LogicalValue,
    output: &mut Vec<u8>,
) -> Result<(), LogicalRowError> {
    if field.nullable {
        if matches!(value, LogicalValue::Null) {
            output.push(0);
            return Ok(());
        }
        output.push(1);
    } else if matches!(value, LogicalValue::Null) {
        return Err(LogicalRowError::UnexpectedNull);
    }
    encode_present(&field.logical_type, value, output)
}

fn encode_present(
    logical_type: &LogicalType,
    value: &LogicalValue,
    output: &mut Vec<u8>,
) -> Result<(), LogicalRowError> {
    match (logical_type, value) {
        (LogicalType::Bool, LogicalValue::Bool(value)) => output.push(u8::from(*value)),
        (LogicalType::I8, LogicalValue::Signed(value)) => output.extend_from_slice(
            &i8::try_from(*value)
                .map_err(|_| LogicalRowError::IntegerOutOfRange("i8"))?
                .to_be_bytes(),
        ),
        (LogicalType::I16, LogicalValue::Signed(value)) => output.extend_from_slice(
            &i16::try_from(*value)
                .map_err(|_| LogicalRowError::IntegerOutOfRange("i16"))?
                .to_be_bytes(),
        ),
        (LogicalType::I32, LogicalValue::Signed(value)) => output.extend_from_slice(
            &i32::try_from(*value)
                .map_err(|_| LogicalRowError::IntegerOutOfRange("i32"))?
                .to_be_bytes(),
        ),
        (LogicalType::I64, LogicalValue::Signed(value)) => output.extend_from_slice(
            &i64::try_from(*value)
                .map_err(|_| LogicalRowError::IntegerOutOfRange("i64"))?
                .to_be_bytes(),
        ),
        (LogicalType::U8, LogicalValue::Unsigned(value)) => output.extend_from_slice(
            &u8::try_from(*value)
                .map_err(|_| LogicalRowError::IntegerOutOfRange("u8"))?
                .to_be_bytes(),
        ),
        (LogicalType::U16, LogicalValue::Unsigned(value)) => output.extend_from_slice(
            &u16::try_from(*value)
                .map_err(|_| LogicalRowError::IntegerOutOfRange("u16"))?
                .to_be_bytes(),
        ),
        (LogicalType::U32, LogicalValue::Unsigned(value)) => output.extend_from_slice(
            &u32::try_from(*value)
                .map_err(|_| LogicalRowError::IntegerOutOfRange("u32"))?
                .to_be_bytes(),
        ),
        (LogicalType::U64, LogicalValue::Unsigned(value)) => output.extend_from_slice(
            &u64::try_from(*value)
                .map_err(|_| LogicalRowError::IntegerOutOfRange("u64"))?
                .to_be_bytes(),
        ),
        (LogicalType::Float64, LogicalValue::Float64(value)) => {
            if !value.is_finite() {
                return Err(LogicalRowError::NonFiniteFloat);
            }
            output.extend_from_slice(&value.to_bits().to_be_bytes());
        }
        (LogicalType::Decimal128, LogicalValue::Decimal128(value)) => {
            output.extend_from_slice(&value.to_be_bytes());
        }
        (LogicalType::Utf8 | LogicalType::Enum, LogicalValue::String(value)) => {
            write_length_prefixed(value.as_bytes(), output)?;
        }
        (LogicalType::Binary, LogicalValue::Binary(value)) => {
            write_length_prefixed(value, output)?;
        }
        (LogicalType::FixedBinary(expected), LogicalValue::Binary(value)) => {
            if value.len() != *expected {
                return Err(LogicalRowError::FixedBinaryLength {
                    expected: *expected,
                    actual: value.len(),
                });
            }
            output.extend_from_slice(value);
        }
        (LogicalType::Struct(fields), LogicalValue::Struct(values)) => {
            if fields.len() != values.len() {
                return Err(LogicalRowError::FieldCount {
                    expected: fields.len(),
                    actual: values.len(),
                });
            }
            for (field, value) in fields.iter().zip(values) {
                encode_field(field, value, output)?;
            }
        }
        (LogicalType::List(field), LogicalValue::List(values)) => {
            output.extend_from_slice(
                &u64::try_from(values.len())
                    .map_err(|_| LogicalRowError::LengthOverflow)?
                    .to_be_bytes(),
            );
            for value in values {
                encode_field(field, value, output)?;
            }
        }
        (LogicalType::StringMap, LogicalValue::StringMap(entries)) => {
            let mut seen = BTreeSet::new();
            let mut sorted: Vec<_> = entries.iter().collect();
            sorted.sort_unstable_by(|left, right| left.0.as_bytes().cmp(right.0.as_bytes()));
            for (key, _) in &sorted {
                if !seen.insert(key.as_str()) {
                    return Err(LogicalRowError::DuplicateMapKey((*key).clone()));
                }
            }
            output.extend_from_slice(
                &u64::try_from(sorted.len())
                    .map_err(|_| LogicalRowError::LengthOverflow)?
                    .to_be_bytes(),
            );
            for (key, value) in sorted {
                write_length_prefixed(key.as_bytes(), output)?;
                write_length_prefixed(value.as_bytes(), output)?;
            }
        }
        (_, LogicalValue::Null) => return Err(LogicalRowError::UnexpectedNull),
        (LogicalType::Bool, _) => return Err(LogicalRowError::TypeMismatch("boolean")),
        (LogicalType::I8 | LogicalType::I16 | LogicalType::I32 | LogicalType::I64, _) => {
            return Err(LogicalRowError::TypeMismatch("signed integer"));
        }
        (LogicalType::U8 | LogicalType::U16 | LogicalType::U32 | LogicalType::U64, _) => {
            return Err(LogicalRowError::TypeMismatch("unsigned integer"));
        }
        (LogicalType::Float64, _) => return Err(LogicalRowError::TypeMismatch("Float64")),
        (LogicalType::Decimal128, _) => {
            return Err(LogicalRowError::TypeMismatch("Decimal128"));
        }
        (LogicalType::Utf8 | LogicalType::Enum, _) => {
            return Err(LogicalRowError::TypeMismatch("UTF-8 string"));
        }
        (LogicalType::Binary | LogicalType::FixedBinary(_), _) => {
            return Err(LogicalRowError::TypeMismatch("binary"));
        }
        (LogicalType::Struct(_), _) => return Err(LogicalRowError::TypeMismatch("struct")),
        (LogicalType::List(_), _) => return Err(LogicalRowError::TypeMismatch("list")),
        (LogicalType::StringMap, _) => {
            return Err(LogicalRowError::TypeMismatch("string map"));
        }
    }
    Ok(())
}

fn write_length_prefixed(bytes: &[u8], output: &mut Vec<u8>) -> Result<(), LogicalRowError> {
    output.extend_from_slice(
        &u64::try_from(bytes.len())
            .map_err(|_| LogicalRowError::LengthOverflow)?
            .to_be_bytes(),
    );
    output.extend_from_slice(bytes);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn schema() -> LogicalSchema {
        LogicalSchema {
            table: TableId::Samples,
            fingerprint: Digest::from_bytes([0x44; 32]),
            fields: vec![
                LogicalField {
                    nullable: false,
                    logical_type: LogicalType::Float64,
                },
                LogicalField {
                    nullable: true,
                    logical_type: LogicalType::StringMap,
                },
                LogicalField {
                    nullable: false,
                    logical_type: LogicalType::List(Box::new(LogicalField {
                        nullable: true,
                        logical_type: LogicalType::U16,
                    })),
                },
            ],
        }
    }

    #[test]
    fn logical_rows_preserve_negative_zero_and_sort_maps() {
        let left = CanonicalLogicalRow::encode(
            &schema(),
            &[
                LogicalValue::Float64(-0.0),
                LogicalValue::StringMap(vec![
                    ("z".into(), "last".into()),
                    ("a".into(), "first".into()),
                ]),
                LogicalValue::List(vec![LogicalValue::Unsigned(7), LogicalValue::Null]),
            ],
        )
        .unwrap();
        let right = CanonicalLogicalRow::encode(
            &schema(),
            &[
                LogicalValue::Float64(-0.0),
                LogicalValue::StringMap(vec![
                    ("a".into(), "first".into()),
                    ("z".into(), "last".into()),
                ]),
                LogicalValue::List(vec![LogicalValue::Unsigned(7), LogicalValue::Null]),
            ],
        )
        .unwrap();
        assert_eq!(left, right);
        assert!(
            left.bytes()
                .windows(8)
                .any(|window| window == 0x8000_0000_0000_0000_u64.to_be_bytes())
        );
    }

    #[test]
    fn projection_evidence_is_order_independent_and_zero_rows_are_explicit() {
        let first = CanonicalLogicalRow::encode(
            &schema(),
            &[
                LogicalValue::Float64(1.0),
                LogicalValue::Null,
                LogicalValue::List(vec![]),
            ],
        )
        .unwrap();
        let second = CanonicalLogicalRow::encode(
            &schema(),
            &[
                LogicalValue::Float64(2.0),
                LogicalValue::Null,
                LogicalValue::List(vec![]),
            ],
        )
        .unwrap();
        assert_eq!(
            ProjectionEvidence::from_rows([&first, &second]).unwrap(),
            ProjectionEvidence::from_rows([&second, &first]).unwrap()
        );
        assert_eq!(ProjectionEvidence::empty().row_count, 0);
        assert_ne!(
            ProjectionEvidence::empty().logical_multiset_digest,
            ProjectionEvidence::from_rows([&first])
                .unwrap()
                .logical_multiset_digest
        );
    }

    #[test]
    fn malformed_values_fail_before_evidence() {
        assert!(matches!(
            CanonicalLogicalRow::encode(
                &schema(),
                &[
                    LogicalValue::Float64(f64::NAN),
                    LogicalValue::Null,
                    LogicalValue::List(vec![]),
                ],
            ),
            Err(LogicalRowError::NonFiniteFloat)
        ));
        assert!(matches!(
            CanonicalLogicalRow::encode(
                &schema(),
                &[
                    LogicalValue::Float64(1.0),
                    LogicalValue::StringMap(vec![("a".into(), "1".into()), ("a".into(), "2".into())]),
                    LogicalValue::List(vec![]),
                ],
            ),
            Err(LogicalRowError::DuplicateMapKey(key)) if key == "a"
        ));
    }
}
