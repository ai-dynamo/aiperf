// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Pydantic-compatible scalar coercions shared by recorded wire schemas.

use num_bigint::BigInt;
use serde_json::Value;

/// Parse a raw JSON number token (as emitted by a serde_json [`RawValue`]) into
/// an `i128` hash without ever coercing through `f64`.
///
/// Recorded WEKA/Dynamo cache-block hash ids routinely exceed `u64::MAX` (e.g.
/// `184467440737095516170`). Decoding the enclosing document into a plain
/// [`Value`] would round those tokens through `f64` and silently lose the low
/// digits unless `serde_json`'s global `arbitrary_precision` feature is on; that
/// feature has broad side effects (it turns every number into an internal map),
/// so instead callers capture the untouched hash token via [`RawValue`] and hand
/// its text here. Bare integer strings parse directly; Pydantic-compatible
/// integral decimal-float tokens (`"1.2e3"`, `"16.0"`) fall through
/// [`decimal_integer`] to stay byte-exact with the [`integer`] coercion path.
///
/// [`RawValue`]: serde_json::value::RawValue
pub(super) fn hash_i128_from_raw_text(text: &str) -> Option<i128> {
    let text = text.trim();
    if let Ok(value) = text.parse::<i128>() {
        return Some(value);
    }
    i128::try_from(decimal_integer(text)?).ok()
}

pub(super) fn integer(value: &Value) -> Option<BigInt> {
    match value {
        Value::Bool(value) => Some(BigInt::from(u8::from(*value))),
        Value::String(value) => {
            let value = value.trim();
            value.parse().ok().or_else(|| {
                (!value.contains(['e', 'E']))
                    .then(|| decimal_integer(value))
                    .flatten()
            })
        }
        Value::Number(number) => {
            let text = number.to_string();
            text.parse().ok().or_else(|| decimal_integer(&text))
        }
        _ => None,
    }
}

pub(super) fn float(value: &Value) -> Option<f64> {
    match value {
        Value::Bool(value) => Some(if *value { 1.0 } else { 0.0 }),
        Value::String(value) => value.trim().parse().ok(),
        Value::Number(number) => number.as_f64(),
        _ => None,
    }
}

fn decimal_integer(text: &str) -> Option<BigInt> {
    let (mantissa, exponent) = match text.split_once(['e', 'E']) {
        Some((mantissa, exponent)) => (mantissa, exponent.parse::<i64>().ok()?),
        None => (text, 0),
    };
    let (negative, mantissa) = match mantissa.as_bytes().first().copied() {
        Some(b'-') => (true, &mantissa[1..]),
        Some(b'+') => (false, &mantissa[1..]),
        _ => (false, mantissa),
    };
    let (whole, fraction) = mantissa.split_once('.').unwrap_or((mantissa, ""));
    if whole.is_empty() && fraction.is_empty()
        || !whole.bytes().all(|byte| byte.is_ascii_digit())
        || !fraction.bytes().all(|byte| byte.is_ascii_digit())
    {
        return None;
    }
    let mut digits = format!("{whole}{fraction}");
    let scale = exponent.checked_sub(i64::try_from(fraction.len()).ok()?)?;
    if scale >= 0 {
        let zeroes = usize::try_from(scale).ok()?;
        if zeroes > 100_000 {
            return None;
        }
        digits.extend(std::iter::repeat_n('0', zeroes));
    } else {
        let removed = usize::try_from(scale.checked_neg()?).ok()?;
        if removed > digits.len()
            || !digits[digits.len().saturating_sub(removed)..]
                .bytes()
                .all(|byte| byte == b'0')
        {
            return None;
        }
        digits.truncate(digits.len() - removed);
    }
    if digits.is_empty() {
        digits.push('0');
    }
    let magnitude = digits.parse::<BigInt>().ok()?;
    Some(if negative { -magnitude } else { magnitude })
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn integer_accepts_integral_json_floats_strings_and_booleans() {
        assert_eq!(integer(&json!(1.0)), Some(BigInt::from(1)));
        assert_eq!(integer(&json!(1.25)), None);
        assert_eq!(integer(&json!(1.2e3)), Some(BigInt::from(1200)));
        assert_eq!(integer(&json!("42")), Some(BigInt::from(42)));
        assert_eq!(integer(&json!("42.0")), Some(BigInt::from(42)));
        assert_eq!(integer(&json!(" 42 ")), Some(BigInt::from(42)));
        assert_eq!(integer(&json!("1.2e3")), None);
        assert_eq!(integer(&json!(true)), Some(BigInt::from(1)));
    }

    #[test]
    fn hash_from_raw_text_preserves_wide_ids_without_f64_coercion() {
        assert_eq!(
            hash_i128_from_raw_text("184467440737095516170"),
            Some(184467440737095516170_i128)
        );
        assert_eq!(hash_i128_from_raw_text(" 42 "), Some(42));
        assert_eq!(hash_i128_from_raw_text("-1"), Some(-1));
        assert_eq!(hash_i128_from_raw_text("16.0"), Some(16));
        assert_eq!(hash_i128_from_raw_text("1.2e3"), Some(1200));
        assert_eq!(hash_i128_from_raw_text("1.25"), None);
        assert_eq!(hash_i128_from_raw_text("\"42\""), None);
    }
}
