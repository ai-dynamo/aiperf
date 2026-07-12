// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Exact decimal lexemes with deterministic binary64 and nanosecond projections.

use std::cmp::Ordering;
use std::fmt::{Display, Formatter, Result as FmtResult};

use num_bigint::{BigInt, BigUint, Sign};
use num_rational::BigRational;
use num_traits::{One, ToPrimitive};

use crate::format::ExpositionFormat;

const DECIMAL128_MAX: &str = "99999999999999999999999999999999999999";

/// Tagged source-number class.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NumberKind {
    /// Finite integer or decimal value.
    Finite,
    /// Positive infinity.
    PositiveInfinity,
    /// Negative infinity.
    NegativeInfinity,
    /// Not-a-number.
    NaN,
    /// Synthetic absence of an optional semantic component.
    Absent,
}

/// Lexical production used by one finite source token.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NumberProduction {
    /// An integer token without decimal point or exponent.
    Integer,
    /// A floating token containing a decimal point or exponent.
    Binary64,
    /// No source production because the value is non-finite or absent.
    NotApplicable,
}

/// Relationship between the exact source number and its analytical binary64 projection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum F64Status {
    /// Binary64 represents the exact source value, including zero sign.
    Exact,
    /// Binary64 is the correctly rounded, non-exact source value.
    Rounded,
    /// A valid wider OpenMetrics integer has no finite binary64 projection.
    Unavailable,
    /// The value is non-finite or synthetic absence.
    NotApplicable,
}

/// Canonical exact finite decimal represented as `coefficient * 10^exponent10`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExactDecimal {
    negative: bool,
    coefficient: String,
    exponent10: i64,
}

impl ExactDecimal {
    /// Whether the source carried a negative sign.
    pub const fn is_negative(&self) -> bool {
        self.negative
    }

    /// Canonical unsigned coefficient with no leading or trailing zeroes.
    pub fn coefficient(&self) -> &str {
        &self.coefficient
    }

    /// Base-ten exponent applied to the canonical coefficient.
    pub const fn exponent10(&self) -> i64 {
        self.exponent10
    }

    /// Whether this value is mathematically zero.
    pub fn is_zero(&self) -> bool {
        self.coefficient == "0"
    }

    /// Whether this exact decimal is an integer.
    pub fn is_integer(&self) -> bool {
        self.is_zero() || self.exponent10 >= 0
    }

    /// Compares exact mathematical values, treating positive and negative zero as equal.
    pub fn numeric_cmp(&self, other: &Self) -> Ordering {
        if self.is_zero() && other.is_zero() {
            return Ordering::Equal;
        }
        if self.is_zero() {
            return if other.negative {
                Ordering::Greater
            } else {
                Ordering::Less
            };
        }
        if other.is_zero() {
            return if self.negative {
                Ordering::Less
            } else {
                Ordering::Greater
            };
        }
        if self.negative != other.negative {
            return if self.negative {
                Ordering::Less
            } else {
                Ordering::Greater
            };
        }
        let magnitude = compare_magnitude(self, other);
        if self.negative {
            magnitude.reverse()
        } else {
            magnitude
        }
    }

    /// Tests exact mathematical equality, ignoring only the sign of zero.
    pub fn numeric_eq(&self, other: &Self) -> bool {
        self.numeric_cmp(other) == Ordering::Equal
    }

    fn parse(lexeme: &str) -> Result<(Self, NumberProduction), NumberError> {
        let bytes = lexeme.as_bytes();
        if bytes.is_empty() {
            return Err(NumberError::InvalidSyntax);
        }
        let mut cursor = 0;
        let negative = match bytes[0] {
            b'-' => {
                cursor += 1;
                true
            }
            b'+' => {
                cursor += 1;
                false
            }
            _ => false,
        };
        if cursor == bytes.len() {
            return Err(NumberError::InvalidSyntax);
        }

        let mantissa_start = cursor;
        let mut dot = None;
        let mut exponent_marker = None;
        while cursor < bytes.len() {
            match bytes[cursor] {
                b'0'..=b'9' => cursor += 1,
                b'.' if dot.is_none() && exponent_marker.is_none() => {
                    dot = Some(cursor);
                    cursor += 1;
                }
                b'e' | b'E' if exponent_marker.is_none() => {
                    exponent_marker = Some(cursor);
                    cursor += 1;
                    break;
                }
                _ => return Err(NumberError::InvalidSyntax),
            }
        }
        let mantissa_end = exponent_marker.unwrap_or(bytes.len());
        let mantissa = &lexeme[mantissa_start..mantissa_end];
        let digit_count = mantissa.bytes().filter(u8::is_ascii_digit).count();
        if digit_count == 0 {
            return Err(NumberError::InvalidSyntax);
        }
        if let Some(dot_index) = dot {
            let before = dot_index - mantissa_start;
            let after = mantissa_end - dot_index - 1;
            if before == 0 && after == 0 {
                return Err(NumberError::InvalidSyntax);
            }
        }

        let explicit_exponent = if exponent_marker.is_some() {
            let exponent = &lexeme[cursor..];
            if exponent.is_empty() {
                return Err(NumberError::InvalidSyntax);
            }
            let (negative_exponent, digits) = match exponent.as_bytes()[0] {
                b'-' => (true, &exponent[1..]),
                b'+' => (false, &exponent[1..]),
                _ => (false, exponent),
            };
            if digits.is_empty() || !digits.bytes().all(|byte| byte.is_ascii_digit()) {
                return Err(NumberError::InvalidSyntax);
            }
            let normalized_digits = digits.trim_start_matches('0');
            let magnitude = if normalized_digits.is_empty() {
                0
            } else {
                normalized_digits
                    .parse::<i64>()
                    .map_err(|_| NumberError::ExponentOutOfRange)?
            };
            if negative_exponent {
                magnitude
                    .checked_neg()
                    .ok_or(NumberError::ExponentOutOfRange)?
            } else {
                magnitude
            }
        } else {
            0
        };

        let fractional_digits = dot.map_or(0_i64, |index| {
            i64::try_from(mantissa_end - index - 1).unwrap_or(i64::MAX)
        });
        let mut digits = mantissa
            .bytes()
            .filter(|byte| byte.is_ascii_digit())
            .map(char::from)
            .collect::<String>();
        let first_nonzero = digits.find(|character| character != '0');
        if let Some(index) = first_nonzero {
            digits.drain(..index);
        } else {
            return Ok((
                Self {
                    negative,
                    coefficient: "0".to_string(),
                    exponent10: 0,
                },
                if dot.is_none() && exponent_marker.is_none() {
                    NumberProduction::Integer
                } else {
                    NumberProduction::Binary64
                },
            ));
        }
        let trailing = digits
            .bytes()
            .rev()
            .take_while(|byte| *byte == b'0')
            .count();
        let new_len = digits.len() - trailing;
        digits.truncate(new_len);
        let exponent10 = explicit_exponent
            .checked_sub(fractional_digits)
            .and_then(|value| value.checked_add(i64::try_from(trailing).ok()?))
            .ok_or(NumberError::ExponentOutOfRange)?;
        Ok((
            Self {
                negative,
                coefficient: digits,
                exponent10,
            },
            if dot.is_none() && exponent_marker.is_none() {
                NumberProduction::Integer
            } else {
                NumberProduction::Binary64
            },
        ))
    }

    fn to_big_rational(&self) -> Option<BigRational> {
        if self.exponent10.unsigned_abs() > 10_000 {
            return None;
        }
        let coefficient = BigInt::parse_bytes(self.coefficient.as_bytes(), 10)?;
        let coefficient = if self.negative {
            -coefficient
        } else {
            coefficient
        };
        let power = ten_pow(self.exponent10.unsigned_abs() as u32);
        Some(if self.exponent10 >= 0 {
            BigRational::from_integer(coefficient * power)
        } else {
            BigRational::new(coefficient, power)
        })
    }

    fn scaled(&self, decimal_places: i64) -> Result<Self, NumberError> {
        Ok(Self {
            negative: self.negative,
            coefficient: self.coefficient.clone(),
            exponent10: self
                .exponent10
                .checked_add(decimal_places)
                .ok_or(NumberError::ExponentOutOfRange)?,
        })
    }
}

/// Lossless source number plus its optional analytical projections.
#[derive(Debug, Clone, PartialEq)]
pub struct ExactNumber {
    /// Tagged finite/non-finite/absent class.
    pub kind: NumberKind,
    /// Exact source token, absent only for a synthetic missing component.
    pub source_lexeme: Option<String>,
    /// Finite exact decimal value when `kind` is finite.
    pub exact_decimal: Option<ExactDecimal>,
    /// Correctly rounded finite binary64 projection when available.
    pub finite_value: Option<f64>,
    /// Exact unsigned integer projection for an eligible integer token.
    pub exact_u64: Option<u64>,
    /// Exactness of the analytical binary64 projection.
    pub f64_status: F64Status,
    /// Lexical production used by the source token.
    pub production: NumberProduction,
}

impl ExactNumber {
    /// Constructs a synthetic absent optional component.
    pub fn absent() -> Self {
        Self {
            kind: NumberKind::Absent,
            source_lexeme: None,
            exact_decimal: None,
            finite_value: None,
            exact_u64: None,
            f64_status: F64Status::NotApplicable,
            production: NumberProduction::NotApplicable,
        }
    }

    /// Whether this is a finite mathematical integer.
    pub fn is_integer(&self) -> bool {
        self.exact_decimal
            .as_ref()
            .is_some_and(ExactDecimal::is_integer)
    }

    /// Whether this is exactly mathematical zero.
    pub fn is_zero(&self) -> bool {
        self.exact_decimal
            .as_ref()
            .is_some_and(ExactDecimal::is_zero)
    }

    /// Whether this is exactly mathematical one.
    pub fn is_one(&self) -> bool {
        self.exact_decimal.as_ref().is_some_and(|value| {
            !value.negative && value.coefficient == "1" && value.exponent10 == 0
        })
    }

    /// Whether this is finite and mathematically negative.
    pub fn is_negative(&self) -> bool {
        self.exact_decimal
            .as_ref()
            .is_some_and(|value| value.negative && !value.is_zero())
    }

    /// Compares two finite exact source values.
    pub fn finite_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(
            self.exact_decimal
                .as_ref()?
                .numeric_cmp(other.exact_decimal.as_ref()?),
        )
    }
}

/// Numeric-token rejection independent of source line location.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NumberError {
    /// Token does not match the selected decimal/non-finite grammar.
    InvalidSyntax,
    /// A decimal exponent cannot be represented safely.
    ExponentOutOfRange,
    /// A binary64 production overflows to infinity.
    Binary64Overflow,
    /// The selected format does not permit this non-finite spelling.
    InvalidNonFinite,
    /// A timestamp token is invalid for its format-specific units.
    InvalidTimestamp,
}

impl Display for NumberError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> FmtResult {
        formatter.write_str(match self {
            Self::InvalidSyntax => "invalid numeric syntax",
            Self::ExponentOutOfRange => "decimal exponent is out of range",
            Self::Binary64Overflow => "floating token exceeds binary64 range",
            Self::InvalidNonFinite => "invalid non-finite token for selected format",
            Self::InvalidTimestamp => "invalid timestamp token for selected format",
        })
    }
}

impl std::error::Error for NumberError {}

/// Parses one exact sample-value lexeme under the selected exposition grammar.
pub fn parse_number_lexeme(
    format: ExpositionFormat,
    lexeme: &str,
) -> Result<ExactNumber, NumberError> {
    if let Some(kind) = parse_non_finite(format, lexeme)? {
        return Ok(ExactNumber {
            kind,
            source_lexeme: Some(lexeme.to_string()),
            exact_decimal: None,
            finite_value: None,
            exact_u64: None,
            f64_status: F64Status::NotApplicable,
            production: NumberProduction::NotApplicable,
        });
    }

    let (exact, production) = ExactDecimal::parse(lexeme)?;
    let parsed = lexeme
        .parse::<f64>()
        .map_err(|_| NumberError::InvalidSyntax)?;
    let finite_value = if parsed.is_finite() {
        Some(parsed)
    } else if format == ExpositionFormat::OpenMetricsText100
        && production == NumberProduction::Integer
    {
        None
    } else {
        return Err(NumberError::Binary64Overflow);
    };
    let exact_u64 = if production == NumberProduction::Integer && !lexeme.starts_with('-') {
        let digits = lexeme.trim_start_matches('+');
        BigUint::parse_bytes(digits.as_bytes(), 10).and_then(|value| value.to_u64())
    } else {
        None
    };
    let f64_status = match finite_value {
        None => F64Status::Unavailable,
        Some(value) if exact_binary64(&exact, value) => F64Status::Exact,
        Some(_) => F64Status::Rounded,
    };
    Ok(ExactNumber {
        kind: NumberKind::Finite,
        source_lexeme: Some(lexeme.to_string()),
        exact_decimal: Some(exact),
        finite_value,
        exact_u64,
        f64_status,
        production,
    })
}

/// Timestamp normalization status without silent rounding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimestampStatus {
    /// No timestamp was emitted.
    Absent,
    /// The exact value is an in-range integer number of nanoseconds.
    ExactNanoseconds,
    /// The exact value is in range but has sub-nanosecond precision.
    SubNanosecondPrecision,
    /// The exact value is integral nanoseconds outside Decimal128(38,0).
    OutOfRange,
    /// The exact value is both non-integral nanoseconds and outside the Decimal128 range.
    SubNanosecondOutOfRange,
}

/// Exact optional sample or exemplar observation timestamp.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceTimestamp {
    /// Exact source lexeme, absent only when no timestamp was emitted.
    pub lexeme: Option<String>,
    /// Exact parsed decimal in source units.
    pub exact_decimal: Option<ExactDecimal>,
    /// Exact normalized Unix nanoseconds when representable.
    pub normalized_unix_ns: Option<i128>,
    /// Normalization status.
    pub status: TimestampStatus,
}

impl SourceTimestamp {
    /// Constructs an absent source timestamp.
    pub fn absent() -> Self {
        Self {
            lexeme: None,
            exact_decimal: None,
            normalized_unix_ns: None,
            status: TimestampStatus::Absent,
        }
    }

    /// Parses a format-specific source timestamp.
    pub fn parse(format: ExpositionFormat, lexeme: &str) -> Result<Self, NumberError> {
        let (exact, production) = ExactDecimal::parse(lexeme)?;
        match format {
            ExpositionFormat::PrometheusText004 if production != NumberProduction::Integer => {
                return Err(NumberError::InvalidTimestamp);
            }
            ExpositionFormat::OpenMetricsText100 => {}
            ExpositionFormat::PrometheusText004 => {}
        }
        let scale = match format {
            ExpositionFormat::PrometheusText004 => 6,
            ExpositionFormat::OpenMetricsText100 => 9,
        };
        let scaled = exact.scaled(scale)?;
        let (normalized_unix_ns, status) = normalize_scaled_timestamp(&scaled);
        Ok(Self {
            lexeme: Some(lexeme.to_string()),
            exact_decimal: Some(exact),
            normalized_unix_ns,
            status,
        })
    }

    /// Tests exact temporal equality in the selected format's units.
    pub fn exact_eq(&self, other: &Self) -> bool {
        match (&self.exact_decimal, &other.exact_decimal) {
            (None, None) => true,
            (Some(left), Some(right)) => left.numeric_eq(right),
            _ => false,
        }
    }
}

/// Exact optional semantic creation timestamp, distinct from observation time.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CreatedTimestamp {
    /// Timestamp representation shared with source timestamps but interpreted as creation time.
    pub value: SourceTimestamp,
}

impl CreatedTimestamp {
    /// Constructs an absent semantic creation timestamp.
    pub fn absent() -> Self {
        Self {
            value: SourceTimestamp::absent(),
        }
    }

    /// Parses one OpenMetrics Created value in Unix seconds.
    pub fn parse_openmetrics(lexeme: &str) -> Result<Self, NumberError> {
        Ok(Self {
            value: SourceTimestamp::parse(ExpositionFormat::OpenMetricsText100, lexeme)?,
        })
    }
}

fn parse_non_finite(
    format: ExpositionFormat,
    lexeme: &str,
) -> Result<Option<NumberKind>, NumberError> {
    let kind = match format {
        ExpositionFormat::PrometheusText004 => match lexeme {
            "+Inf" => Some(NumberKind::PositiveInfinity),
            "-Inf" => Some(NumberKind::NegativeInfinity),
            "NaN" => Some(NumberKind::NaN),
            _ => None,
        },
        ExpositionFormat::OpenMetricsText100 => {
            let lowercase = lexeme.to_ascii_lowercase();
            match lowercase.as_str() {
                "inf" | "+inf" | "infinity" | "+infinity" => Some(NumberKind::PositiveInfinity),
                "-inf" | "-infinity" => Some(NumberKind::NegativeInfinity),
                "nan" => Some(NumberKind::NaN),
                value if value.ends_with("nan") => return Err(NumberError::InvalidNonFinite),
                _ => None,
            }
        }
    };
    Ok(kind)
}

fn compare_magnitude(left: &ExactDecimal, right: &ExactDecimal) -> Ordering {
    let left_order =
        i128::try_from(left.coefficient.len()).unwrap_or(i128::MAX) + i128::from(left.exponent10);
    let right_order =
        i128::try_from(right.coefficient.len()).unwrap_or(i128::MAX) + i128::from(right.exponent10);
    match left_order.cmp(&right_order) {
        Ordering::Equal => {
            let width = left.coefficient.len().max(right.coefficient.len());
            for index in 0..width {
                let left_digit = left
                    .coefficient
                    .as_bytes()
                    .get(index)
                    .copied()
                    .unwrap_or(b'0');
                let right_digit = right
                    .coefficient
                    .as_bytes()
                    .get(index)
                    .copied()
                    .unwrap_or(b'0');
                match left_digit.cmp(&right_digit) {
                    Ordering::Equal => {}
                    ordering => return ordering,
                }
            }
            Ordering::Equal
        }
        ordering => ordering,
    }
}

fn exact_binary64(exact: &ExactDecimal, value: f64) -> bool {
    if value == 0.0 {
        return exact.is_zero() && exact.negative == value.is_sign_negative();
    }
    let Some(decimal) = exact.to_big_rational() else {
        return false;
    };
    decimal == binary64_rational(value)
}

fn binary64_rational(value: f64) -> BigRational {
    let bits = value.to_bits();
    let negative = bits >> 63 != 0;
    let exponent_bits = ((bits >> 52) & 0x7ff) as i32;
    let fraction = bits & ((1_u64 << 52) - 1);
    let (significand, exponent2) = if exponent_bits == 0 {
        (fraction, -1022 - 52)
    } else {
        ((1_u64 << 52) | fraction, exponent_bits - 1023 - 52)
    };
    let mut numerator = BigInt::from(significand);
    if negative {
        numerator = -numerator;
    }
    if exponent2 >= 0 {
        BigRational::from_integer(numerator << exponent2 as usize)
    } else {
        BigRational::new(numerator, BigInt::one() << (-exponent2) as usize)
    }
}

fn normalize_scaled_timestamp(value: &ExactDecimal) -> (Option<i128>, TimestampStatus) {
    if value.is_zero() {
        return (Some(0), TimestampStatus::ExactNanoseconds);
    }
    let in_range = magnitude_within_decimal128(value);
    if value.exponent10 < 0 {
        return (
            None,
            if in_range {
                TimestampStatus::SubNanosecondPrecision
            } else {
                TimestampStatus::SubNanosecondOutOfRange
            },
        );
    }
    if !in_range {
        return (None, TimestampStatus::OutOfRange);
    }
    let mut digits = value.coefficient.clone();
    digits.extend(std::iter::repeat_n('0', value.exponent10 as usize));
    let magnitude = digits.parse::<i128>().ok();
    let normalized = magnitude.map(|magnitude| {
        if value.negative {
            -magnitude
        } else {
            magnitude
        }
    });
    (normalized, TimestampStatus::ExactNanoseconds)
}

fn magnitude_within_decimal128(value: &ExactDecimal) -> bool {
    let order =
        i128::try_from(value.coefficient.len()).unwrap_or(i128::MAX) + i128::from(value.exponent10);
    match order.cmp(&38) {
        Ordering::Less => true,
        Ordering::Greater => false,
        Ordering::Equal => {
            let mut digits = value.coefficient.clone();
            if value.exponent10 > 0 {
                digits.extend(std::iter::repeat_n('0', value.exponent10 as usize));
            }
            if digits.len() < DECIMAL128_MAX.len() {
                digits.extend(std::iter::repeat_n(
                    '0',
                    DECIMAL128_MAX.len() - digits.len(),
                ));
            }
            digits.as_str() <= DECIMAL128_MAX
        }
    }
}

fn ten_pow(exponent: u32) -> BigInt {
    BigInt::from_biguint(Sign::Plus, BigUint::from(10_u8).pow(exponent))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finite_values_preserve_lexemes_and_exactness() {
        let exact =
            parse_number_lexeme(ExpositionFormat::OpenMetricsText100, "9007199254740992").unwrap();
        assert_eq!(exact.exact_u64, Some(9_007_199_254_740_992));
        assert_eq!(exact.f64_status, F64Status::Exact);

        let rounded =
            parse_number_lexeme(ExpositionFormat::OpenMetricsText100, "9007199254740993").unwrap();
        assert_eq!(rounded.source_lexeme.as_deref(), Some("9007199254740993"));
        assert_eq!(rounded.exact_u64, Some(9_007_199_254_740_993));
        assert_eq!(rounded.f64_status, F64Status::Rounded);

        let underflow =
            parse_number_lexeme(ExpositionFormat::OpenMetricsText100, "-1e-999999").unwrap();
        assert_eq!(
            underflow.finite_value.unwrap().to_bits(),
            (-0.0_f64).to_bits()
        );
        assert_eq!(underflow.f64_status, F64Status::Rounded);
    }

    #[test]
    fn wider_openmetrics_integer_can_be_unavailable() {
        let value = parse_number_lexeme(
            ExpositionFormat::OpenMetricsText100,
            "10000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
        )
        .unwrap();
        assert_eq!(value.finite_value, None);
        assert_eq!(value.f64_status, F64Status::Unavailable);
    }

    #[test]
    fn timestamp_statuses_are_exact_and_combined() {
        let exact = SourceTimestamp::parse(ExpositionFormat::OpenMetricsText100, "1.25").unwrap();
        assert_eq!(exact.normalized_unix_ns, Some(1_250_000_000));
        assert_eq!(exact.status, TimestampStatus::ExactNanoseconds);

        let sub =
            SourceTimestamp::parse(ExpositionFormat::OpenMetricsText100, "1.0000000001").unwrap();
        assert_eq!(sub.status, TimestampStatus::SubNanosecondPrecision);

        let combined = SourceTimestamp::parse(
            ExpositionFormat::OpenMetricsText100,
            "10000000000000000000000000000000000000000.0000000001",
        )
        .unwrap();
        assert_eq!(combined.status, TimestampStatus::SubNanosecondOutOfRange);
        assert_eq!(combined.normalized_unix_ns, None);
    }

    #[test]
    fn decimal_numeric_comparison_ignores_spelling() {
        let (left, _) = ExactDecimal::parse("1.0").unwrap();
        let (right, _) = ExactDecimal::parse("01e0").unwrap();
        assert!(left.numeric_eq(&right));
        let (small, _) = ExactDecimal::parse("0.999").unwrap();
        assert_eq!(small.numeric_cmp(&left), Ordering::Less);
        let (zero, _) = ExactDecimal::parse("0").unwrap();
        let (negative, _) = ExactDecimal::parse("-0.001").unwrap();
        assert_eq!(small.numeric_cmp(&zero), Ordering::Greater);
        assert_eq!(negative.numeric_cmp(&zero), Ordering::Less);
    }
}
