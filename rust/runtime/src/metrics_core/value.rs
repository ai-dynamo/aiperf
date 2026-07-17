// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Metric value boundary type.
//!
//! Internal accumulators may use NaN as a sparse-column sentinel, but this crate
//! only lets finite numbers, the error-adjusted `+inf` sentinel, or explicit
//! absence cross a public/reporting boundary.

use serde::de::{self, Deserialize, Deserializer, Visitor};
use serde::ser::{Serialize, Serializer};
use std::fmt;

/// A metric value that is safe to pass across a serialization boundary.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MetricValue {
    /// A present finite numeric value.
    Finite(f64),
    /// Positive infinity, reserved for error-adjusted latency distributions.
    PosInf,
    /// No metric value was produced.
    Absent,
}

impl MetricValue {
    /// Builds a boundary value from a raw `f64`.
    pub fn from_f64(value: f64, allow_pos_inf: bool) -> Self {
        if value.is_finite() {
            Self::Finite(value)
        } else if allow_pos_inf && value.is_infinite() && value.is_sign_positive() {
            Self::PosInf
        } else {
            Self::Absent
        }
    }

    /// Returns true when no metric value was produced.
    pub fn is_absent(self) -> bool {
        matches!(self, Self::Absent)
    }

    /// Returns the finite value when present and finite.
    pub fn as_f64(self) -> Option<f64> {
        match self {
            Self::Finite(value) => Some(value),
            Self::PosInf | Self::Absent => None,
        }
    }
}

impl Serialize for MetricValue {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            Self::Finite(value) => serializer.serialize_f64(*value),
            Self::PosInf => serializer.serialize_str("+inf"),
            Self::Absent => serializer.serialize_none(),
        }
    }
}

/// Deserializes exactly what [`Serialize`] emits, decoded from a self-describing
/// format: a finite float is [`Finite`](MetricValue::Finite), the string `"+inf"`
/// is [`PosInf`](MetricValue::PosInf), and nil is [`Absent`](MetricValue::Absent).
/// The `Serialize` impl only ever writes those three forms (`serialize_f64` /
/// `serialize_str` / `serialize_none`), so the visitor handles exactly those and
/// nothing else. Requires `deserialize_any`; the cellular wire uses MessagePack
/// for this reason, keeping a wire-shipped record's injected overrides byte-faithful.
impl<'de> Deserialize<'de> for MetricValue {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct MetricValueVisitor;

        impl<'de> Visitor<'de> for MetricValueVisitor {
            type Value = MetricValue;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("a float, the string \"+inf\", or nil")
            }

            fn visit_f64<E>(self, value: f64) -> Result<MetricValue, E>
            where
                E: de::Error,
            {
                match MetricValue::from_f64(value, false) {
                    finite @ MetricValue::Finite(_) => Ok(finite),
                    MetricValue::PosInf | MetricValue::Absent => {
                        Err(E::custom("metric float must be finite"))
                    }
                }
            }

            fn visit_str<E>(self, value: &str) -> Result<MetricValue, E>
            where
                E: de::Error,
            {
                match value {
                    "+inf" => Ok(MetricValue::PosInf),
                    other => Err(E::custom(format!(
                        "unexpected metric value string {other:?}"
                    ))),
                }
            }

            // MessagePack (and JSON) decode nil via `visit_unit` under `deserialize_any`.
            fn visit_unit<E>(self) -> Result<MetricValue, E> {
                Ok(MetricValue::Absent)
            }
        }

        deserializer.deserialize_any(MetricValueVisitor)
    }
}

#[cfg(test)]
mod tests {
    use super::MetricValue;

    #[test]
    fn metric_value_scrubs_nan_but_keeps_positive_infinity() {
        assert!(MetricValue::from_f64(f64::NAN, false).is_absent());
        assert!(MetricValue::from_f64(f64::INFINITY, false).is_absent());
        assert_eq!(
            MetricValue::from_f64(f64::INFINITY, true),
            MetricValue::PosInf
        );
        assert_eq!(MetricValue::from_f64(7.5, false).as_f64(), Some(7.5));
    }

    #[test]
    fn messagepack_round_trips_boundary_values() {
        for value in [
            MetricValue::Finite(7.5),
            MetricValue::PosInf,
            MetricValue::Absent,
        ] {
            let encoded = rmp_serde::to_vec(&value).expect("encode metric value");
            let decoded: MetricValue =
                rmp_serde::from_slice(&encoded).expect("decode metric value");
            assert_eq!(decoded, value);
        }
    }

    #[test]
    fn messagepack_rejects_non_finite_float_forms() {
        for value in [f64::NAN, f64::NEG_INFINITY, f64::INFINITY] {
            let encoded = rmp_serde::to_vec(&value).expect("encode raw float");
            assert!(
                rmp_serde::from_slice::<MetricValue>(&encoded).is_err(),
                "{value:?} must not decode as a finite metric value"
            );
        }
    }
}
