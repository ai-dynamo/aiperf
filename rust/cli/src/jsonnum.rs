// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Finite-`f64` to JSON-number conversion shared across output builders.

use serde_json::Value;

/// A finite `f64` as a JSON number, or JSON `null` for non-finite values
/// (`NaN`/`±inf`).
///
/// This is the documented null-vs-absent boundary: a non-finite value serializes
/// to `null`, never to an absent key. Callers that need an absent field must omit
/// the key rather than routing a non-finite value through this helper.
pub fn num(v: f64) -> Value {
    serde_json::Number::from_f64(v)
        .map(Value::Number)
        .unwrap_or(Value::Null)
}
