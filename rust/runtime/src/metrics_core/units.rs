// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unit vocabulary and conversion rules for metric specs.

use std::fmt::{Display, Formatter, Result as FmtResult};

use serde::Serialize;

/// Shape of values emitted by a metric.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MetricValueType {
    /// A scalar floating-point value.
    Float,
    /// A scalar integer value.
    Int,
    /// A list of floating-point values per record.
    FloatList,
    /// A list of integer values per record.
    IntList,
}

/// Units used by the metrics catalog.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Unit {
    /// Unitless scalar.
    Count,
    /// Request count.
    Request,
    /// Token count.
    Token,
    /// Byte count.
    Byte,
    /// Kibibyte count.
    Kilobyte,
    /// Mebibyte count.
    Megabyte,
    /// Gibibyte count.
    Gigabyte,
    /// Tebibyte count.
    Terabyte,
    /// Nanoseconds.
    Nanosecond,
    /// Microseconds.
    Microsecond,
    /// Milliseconds.
    Millisecond,
    /// Seconds.
    Second,
    /// Percentage.
    Percent,
    /// Ratio.
    Ratio,
    /// Requests per second.
    RequestsPerSecond,
    /// Tokens per second.
    TokensPerSecond,
    /// Tokens per second per user.
    TokensPerSecondPerUser,
    /// Images per second.
    ImagesPerSecond,
    /// Images per second per user (concurrent request).
    ImagesPerSecondPerUser,
    /// Milliseconds per image.
    MillisecondsPerImage,
    /// Videos per second.
    VideosPerSecond,
    /// Milliseconds per video.
    MillisecondsPerVideo,
    /// Tokens per joule.
    TokensPerJoule,
    /// Joules per user.
    JoulesPerUser,
    /// Bytes per second.
    BytesPerSecond,
    /// Mebibytes per second.
    MegabytesPerSecond,
    /// Gibibytes per second.
    GigabytesPerSecond,
    /// Watts.
    Watt,
    /// Milliwatts.
    Milliwatt,
    /// Joules.
    Joule,
    /// Millijoules.
    Millijoule,
    /// Megajoules.
    Megajoule,
    /// Hertz.
    Hertz,
    /// Megahertz.
    Megahertz,
    /// Gigahertz.
    Gigahertz,
    /// Celsius.
    Celsius,
    /// Kelvin.
    Kelvin,
    /// Fahrenheit.
    Fahrenheit,
    /// Images.
    Image,
    /// Frames.
    Frame,
    /// Videos.
    Video,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Family {
    Count,
    Ratio,
    Size,
    Time,
    Composite,
    Power,
    Energy,
    Frequency,
    Temperature,
}

impl Unit {
    /// Converts `value` in `self` units into `target` units.
    pub fn convert_value(self, value: f64, target: Unit) -> Result<f64, UnitConversionError> {
        if self == target {
            return Ok(value);
        }
        let source_family = self.family();
        let target_family = target.family();
        if source_family != target_family {
            return Err(UnitConversionError {
                source: self,
                target,
            });
        }
        match source_family {
            Family::Count | Family::Ratio => Err(UnitConversionError {
                source: self,
                target,
            }),
            Family::Composite => self.convert_composite(value, target),
            Family::Size | Family::Time | Family::Power | Family::Energy | Family::Frequency => {
                Ok(value * self.scale_to_base() / target.scale_to_base())
            }
            Family::Temperature => {
                let celsius = self.to_celsius(value);
                Ok(target.value_from_celsius(celsius))
            }
        }
    }

    fn family(self) -> Family {
        match self {
            Self::Byte | Self::Kilobyte | Self::Megabyte | Self::Gigabyte | Self::Terabyte => {
                Family::Size
            }
            Self::Nanosecond | Self::Microsecond | Self::Millisecond | Self::Second => Family::Time,
            Self::RequestsPerSecond
            | Self::TokensPerSecond
            | Self::TokensPerSecondPerUser
            | Self::ImagesPerSecond
            | Self::ImagesPerSecondPerUser
            | Self::MillisecondsPerImage
            | Self::VideosPerSecond
            | Self::MillisecondsPerVideo
            | Self::TokensPerJoule
            | Self::JoulesPerUser
            | Self::BytesPerSecond
            | Self::MegabytesPerSecond
            | Self::GigabytesPerSecond => Family::Composite,
            Self::Watt | Self::Milliwatt => Family::Power,
            Self::Joule | Self::Millijoule | Self::Megajoule => Family::Energy,
            Self::Hertz | Self::Megahertz | Self::Gigahertz => Family::Frequency,
            Self::Celsius | Self::Kelvin | Self::Fahrenheit => Family::Temperature,
            Self::Count | Self::Request | Self::Token | Self::Image | Self::Frame | Self::Video => {
                Family::Count
            }
            Self::Percent | Self::Ratio => Family::Ratio,
        }
    }

    fn scale_to_base(self) -> f64 {
        match self {
            Self::Byte => 1.0,
            Self::Kilobyte => 1024.0,
            Self::Megabyte => 1024.0 * 1024.0,
            Self::Gigabyte => 1024.0 * 1024.0 * 1024.0,
            Self::Terabyte => 1024.0 * 1024.0 * 1024.0 * 1024.0,
            Self::Nanosecond => 1.0,
            Self::Microsecond => 1_000.0,
            Self::Millisecond => 1_000_000.0,
            Self::Second => 1_000_000_000.0,
            Self::Watt => 1.0,
            Self::Milliwatt => 0.001,
            Self::Joule => 1.0,
            Self::Millijoule => 0.001,
            Self::Megajoule => 1_000_000.0,
            Self::Hertz => 1.0,
            Self::Megahertz => 1_000_000.0,
            Self::Gigahertz => 1_000_000_000.0,
            _ => 1.0,
        }
    }

    fn convert_composite(self, value: f64, target: Unit) -> Result<f64, UnitConversionError> {
        let size_per_second_scale = |unit| match unit {
            Self::BytesPerSecond => Some(1.0),
            Self::MegabytesPerSecond => Some(1024.0 * 1024.0),
            Self::GigabytesPerSecond => Some(1024.0 * 1024.0 * 1024.0),
            _ => None,
        };
        match (size_per_second_scale(self), size_per_second_scale(target)) {
            (Some(source), Some(destination)) => Ok(value * source / destination),
            _ => Err(UnitConversionError {
                source: self,
                target,
            }),
        }
    }

    fn to_celsius(self, value: f64) -> f64 {
        match self {
            Self::Celsius => value,
            Self::Kelvin => value - 273.15,
            Self::Fahrenheit => (value - 32.0) * 5.0 / 9.0,
            _ => value,
        }
    }

    fn value_from_celsius(self, value: f64) -> f64 {
        match self {
            Self::Celsius => value,
            Self::Kelvin => value + 273.15,
            Self::Fahrenheit => value * 9.0 / 5.0 + 32.0,
            _ => value,
        }
    }

    /// Stable report spelling.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Count => "count",
            Self::Request => "requests",
            Self::Token => "tokens",
            Self::Byte => "bytes",
            Self::Kilobyte => "KB",
            Self::Megabyte => "MB",
            Self::Gigabyte => "GB",
            Self::Terabyte => "TB",
            Self::Nanosecond => "ns",
            Self::Microsecond => "us",
            Self::Millisecond => "ms",
            Self::Second => "sec",
            Self::Percent => "%",
            Self::Ratio => "ratio",
            Self::RequestsPerSecond => "requests/sec",
            Self::TokensPerSecond => "tokens/sec",
            Self::TokensPerSecondPerUser => "tokens/sec/user",
            Self::ImagesPerSecond => "images/sec",
            Self::ImagesPerSecondPerUser => "images/sec/user",
            Self::MillisecondsPerImage => "ms/image",
            Self::VideosPerSecond => "videos/sec",
            Self::MillisecondsPerVideo => "ms/video",
            Self::TokensPerJoule => "tokens/J",
            Self::JoulesPerUser => "joules/user",
            Self::BytesPerSecond => "bytes/sec",
            Self::MegabytesPerSecond => "MB/sec",
            Self::GigabytesPerSecond => "GB/sec",
            Self::Watt => "W",
            Self::Milliwatt => "mW",
            Self::Joule => "J",
            Self::Millijoule => "mJ",
            Self::Megajoule => "MJ",
            Self::Hertz => "Hz",
            Self::Megahertz => "MHz",
            Self::Gigahertz => "GHz",
            Self::Celsius => "°C",
            Self::Kelvin => "K",
            Self::Fahrenheit => "°F",
            Self::Image => "images",
            Self::Frame => "frames",
            Self::Video => "videos",
        }
    }
}

impl Display for Unit {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        f.write_str(self.as_str())
    }
}

/// Error returned when two units are not in the same conversion family.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UnitConversionError {
    /// Source unit.
    pub source: Unit,
    /// Target unit.
    pub target: Unit,
}

impl Display for UnitConversionError {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(f, "cannot convert {} to {}", self.source, self.target)
    }
}

impl std::error::Error for UnitConversionError {}

#[cfg(test)]
mod tests {
    use super::Unit;

    #[test]
    fn time_and_size_units_convert_with_spec_rules() {
        assert_eq!(
            Unit::Nanosecond
                .convert_value(2_000_000.0, Unit::Millisecond)
                .unwrap(),
            2.0
        );
        assert_eq!(
            Unit::Byte.convert_value(2048.0, Unit::Kilobyte).unwrap(),
            2.0
        );
        assert_eq!(
            Unit::Celsius.convert_value(0.0, Unit::Kelvin).unwrap(),
            273.15
        );
        assert_eq!(
            Unit::MegabytesPerSecond
                .convert_value(1024.0, Unit::GigabytesPerSecond)
                .unwrap(),
            1.0
        );
        assert!(Unit::Token.convert_value(1.0, Unit::Second).is_err());
        assert!(Unit::Token.convert_value(1.0, Unit::Request).is_err());
        assert!(
            Unit::TokensPerSecond
                .convert_value(1.0, Unit::RequestsPerSecond)
                .is_err()
        );
    }
}
