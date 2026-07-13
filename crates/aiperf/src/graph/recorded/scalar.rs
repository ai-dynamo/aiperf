// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Pydantic-compatible scalar coercions shared by recorded wire schemas.

use num_bigint::BigInt;
use serde_json::Value;

/// Fast recorded cache-block hash coercion into a machine `i128`.
///
/// Recorded hashes are u64 in Dynamo/WEKA captures and small negative virtual
/// ids for non-replay turns, so `i128` losslessly covers the entire realistic
/// domain without a per-hash heap allocation. The common `Value::Number`
/// integer path never touches `num_bigint`; the rare string/boolean/oversized
/// forms fall back through [`integer`] so Pydantic-compatible coercion and the
/// arbitrary-precision parity edge cases stay byte-exact. `i128::Display`
/// equals `BigInt::Display` for every in-range value, so content-seed strings
/// are unchanged.
pub(super) fn hash_i128(value: &Value) -> Option<i128> {
    if let Value::Number(number) = value {
        if let Some(unsigned) = number.as_u64() {
            return Some(i128::from(unsigned));
        }
        if let Some(signed) = number.as_i64() {
            return Some(i128::from(signed));
        }
        if let Ok(parsed) = number.to_string().parse::<i128>() {
            return Some(parsed);
        }
    }
    i128::try_from(integer(value)?).ok()
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
}
