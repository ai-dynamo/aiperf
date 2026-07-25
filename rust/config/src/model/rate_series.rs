// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Piecewise-linear request-rate schedule configuration.

use std::path::Path;

use serde::{Deserialize, Serialize};

/// One request-rate control point.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct RateSeriesPoint {
    /// Elapsed seconds from series start (after any rate ramp completes).
    #[serde(alias = "timeS")]
    pub time_s: f64,
    /// Request rate in queries per second at this point.
    pub qps: f64,
}

/// Piecewise-linear request-rate schedule.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct RateSeries {
    /// Strictly increasing control points (≥2).
    pub points: Vec<RateSeriesPoint>,
}

impl RateSeries {
    /// Validate point count, finiteness, and monotonicity.
    pub fn validate(&self) -> anyhow::Result<()> {
        validate_points(&self.points)
    }

    /// First configured request rate.
    pub fn initial_qps(&self) -> f64 {
        self.points[0].qps
    }

    /// Load and validate a JSON file (`{"points":[...]}` or a top-level array).
    pub fn from_json_path(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        let text = std::fs::read_to_string(path.as_ref()).map_err(|e| {
            anyhow::anyhow!(
                "Cannot read request-rate series JSON {:?}: {e}",
                path.as_ref()
            )
        })?;
        Self::from_json_str(&text)
    }

    /// Parse inline JSON text.
    pub fn from_json_str(text: &str) -> anyhow::Result<Self> {
        let value: serde_json::Value = serde_json::from_str(text)
            .map_err(|e| anyhow::anyhow!("Invalid request-rate series JSON: {e}"))?;
        let points_value = match value {
            serde_json::Value::Array(arr) => serde_json::Value::Array(arr),
            serde_json::Value::Object(map) => {
                if map.len() != 1 || !map.contains_key("points") {
                    anyhow::bail!(
                        "Request-rate series JSON must contain exactly one top-level key: points"
                    );
                }
                map["points"].clone()
            }
            _ => anyhow::bail!(
                "Request-rate series JSON must be an object with points or a points array"
            ),
        };
        let points: Vec<RateSeriesPoint> = serde_json::from_value(points_value).map_err(|e| {
            anyhow::anyhow!("Request-rate series JSON points must be an array: {e}")
        })?;
        let series = Self { points };
        series.validate()?;
        Ok(series)
    }
}

/// Shared validation for authored or loaded points.
pub fn validate_points(points: &[RateSeriesPoint]) -> anyhow::Result<()> {
    if points.len() < 2 {
        anyhow::bail!("Request-rate series requires at least two points");
    }
    let mut previous_time = points[0].time_s;
    for (index, point) in points.iter().enumerate() {
        if !point.time_s.is_finite() || !point.qps.is_finite() {
            anyhow::bail!("Request-rate series values must be finite");
        }
        if point.qps <= 0.0 {
            anyhow::bail!("Invalid request-rate series point {index}: qps must be greater than 0");
        }
        if index > 0 && point.time_s <= previous_time {
            anyhow::bail!("Request-rate series time_s values must be strictly increasing");
        }
        previous_time = point.time_s;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_inline_points() {
        let series = RateSeries {
            points: vec![
                RateSeriesPoint {
                    time_s: 0.0,
                    qps: 10.0,
                },
                RateSeriesPoint {
                    time_s: 5.0,
                    qps: 20.0,
                },
            ],
        };
        series.validate().unwrap();
        assert_eq!(series.initial_qps(), 10.0);
    }

    #[test]
    fn rejects_single_point() {
        let series = RateSeries {
            points: vec![RateSeriesPoint {
                time_s: 0.0,
                qps: 10.0,
            }],
        };
        assert!(series.validate().is_err());
    }

    #[test]
    fn rejects_non_increasing_times() {
        let series = RateSeries {
            points: vec![
                RateSeriesPoint {
                    time_s: 0.0,
                    qps: 1.0,
                },
                RateSeriesPoint {
                    time_s: 0.0,
                    qps: 7.0,
                },
            ],
        };
        assert!(series.validate().is_err());
    }

    #[test]
    fn parses_top_level_array_json() {
        let series =
            RateSeries::from_json_str(r#"[{"time_s":0,"qps":1},{"time_s":60,"qps":7}]"#).unwrap();
        assert_eq!(series.points.len(), 2);
    }
}
