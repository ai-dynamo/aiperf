// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Pydantic-compatible scalar coercions shared by recorded wire schemas.

use num_bigint::BigInt;
use serde_json::Value;

pub(super) fn integer(value: &Value) -> Option<BigInt> {
    match value {
        Value::Bool(value) => Some(BigInt::from(u8::from(*value))),
        Value::String(value) => value.parse().ok(),
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
        Value::String(value) => value.parse().ok(),
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
        assert_eq!(integer(&json!(true)), Some(BigInt::from(1)));
    }
}
