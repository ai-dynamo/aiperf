// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Metric-unit inference for the Parquet `unit` column.
//!
//! Display names are lowercase enum spellings with `_per_second` represented as
//! `/s`. Inference checks description scale, description units, then the longest
//! metric-name suffix and the `num_requests_` shortcut.
//!
//! Priority order:
//!   1. scale from description (`(0-1)` -> ratio, `(0-100)` -> percent),
//!   2. unit from description (`(in <tag>)` parenthetical, then phrase patterns),
//!   3. suffix from the metric name (longest suffix first), then the
//!      `num_requests_` containment shortcut.

use std::sync::LazyLock;

use regex::Regex;

/// An inferred metric unit. Only the variants `infer_unit` can produce are
/// modeled.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Unit {
    Seconds,
    Milliseconds,
    Microseconds,
    Nanoseconds,
    Bytes,
    Kilobytes,
    Megabytes,
    Gigabytes,
    Terabytes,
    Count,
    Tokens,
    Requests,
    Errors,
    Blocks,
    Ratio,
    Percent,
    Celsius,
    Fahrenheit,
    Kelvin,
    Hertz,
    Megahertz,
    Gigahertz,
    Watt,
    Milliwatt,
    Joule,
    Millijoule,
    Megajoule,
    GbPerSecond,
    MbPerSecond,
    TokensPerSecond,
    RequestsPerSecond,
}

impl Unit {
    /// The lowercase `unit` column value, with per-second units using `/s`.
    pub(super) fn display_name(self) -> String {
        match self {
            Unit::Seconds => "seconds",
            Unit::Milliseconds => "milliseconds",
            Unit::Microseconds => "microseconds",
            Unit::Nanoseconds => "nanoseconds",
            Unit::Bytes => "bytes",
            Unit::Kilobytes => "kilobytes",
            Unit::Megabytes => "megabytes",
            Unit::Gigabytes => "gigabytes",
            Unit::Terabytes => "terabytes",
            Unit::Count => "count",
            Unit::Tokens => "tokens",
            Unit::Requests => "requests",
            Unit::Errors => "errors",
            Unit::Blocks => "blocks",
            Unit::Ratio => "ratio",
            Unit::Percent => "percent",
            Unit::Celsius => "celsius",
            Unit::Fahrenheit => "fahrenheit",
            Unit::Kelvin => "kelvin",
            Unit::Hertz => "hertz",
            Unit::Megahertz => "megahertz",
            Unit::Gigahertz => "gigahertz",
            Unit::Watt => "watt",
            Unit::Milliwatt => "milliwatt",
            Unit::Joule => "joule",
            Unit::Millijoule => "millijoule",
            Unit::Megajoule => "megajoule",
            Unit::GbPerSecond => "gb/s",
            Unit::MbPerSecond => "mb/s",
            Unit::TokensPerSecond => "tokens/s",
            Unit::RequestsPerSecond => "requests/s",
        }
        .to_string()
    }
}

/// Infer a unit from description scale, description units, then metric name.
pub(super) fn infer_unit(metric_name: &str, description: &str) -> Option<Unit> {
    let description = if description.is_empty() {
        None
    } else {
        Some(description)
    };
    if let Some(scale) = scale_from_description(description) {
        return Some(scale);
    }
    if let Some(unit) = unit_from_description(description) {
        return Some(unit);
    }
    unit_from_metric_name(metric_name)
}

/// `(name suffix, unit)` pairs in stable precedence order.
const SUFFIX_TABLE: &[(&str, Unit)] = &[
    ("_seconds", Unit::Seconds),
    ("_seconds_total", Unit::Seconds),
    ("_milliseconds", Unit::Milliseconds),
    ("_ms", Unit::Milliseconds),
    ("_ms_total", Unit::Milliseconds),
    ("_nanoseconds", Unit::Nanoseconds),
    ("_ns", Unit::Nanoseconds),
    ("_ns_total", Unit::Nanoseconds),
    ("_bytes", Unit::Bytes),
    ("_kilobytes", Unit::Kilobytes),
    ("_megabytes", Unit::Megabytes),
    ("_gigabytes", Unit::Gigabytes),
    ("_bytes_total", Unit::Bytes),
    ("_total", Unit::Count),
    ("_count", Unit::Count),
    ("_tokens", Unit::Tokens),
    ("_tokens_total", Unit::Tokens),
    ("_requests", Unit::Requests),
    ("_requests_total", Unit::Requests),
    ("request_success", Unit::Requests),
    ("_errors", Unit::Errors),
    ("_errors_total", Unit::Errors),
    ("_error_count", Unit::Errors),
    ("_error_count_total", Unit::Errors),
    ("_reqs", Unit::Requests),
    ("_blocks", Unit::Blocks),
    ("_blocks_total", Unit::Blocks),
    ("_block_count", Unit::Blocks),
    ("_gb_s", Unit::GbPerSecond),
    ("_ratio", Unit::Ratio),
    ("_percent", Unit::Percent),
    ("_perc", Unit::Percent),
    ("_celsius", Unit::Celsius),
    ("_joules", Unit::Joule),
    ("_watts", Unit::Watt),
];

/// Suffixes stably sorted by descending length.
static SORTED_SUFFIXES: LazyLock<Vec<(&'static str, Unit)>> = LazyLock::new(|| {
    let mut table = SUFFIX_TABLE.to_vec();
    table.sort_by(|a, b| b.0.len().cmp(&a.0.len()));
    table
});

/// Infer from name suffix or the `num_requests_` shortcut.
fn unit_from_metric_name(metric_name: &str) -> Option<Unit> {
    let name_lower = metric_name.to_ascii_lowercase();
    for (suffix, unit) in SORTED_SUFFIXES.iter() {
        if name_lower.ends_with(suffix) {
            return Some(*unit);
        }
    }
    if name_lower.contains("num_requests_") {
        return Some(Unit::Requests);
    }
    None
}

/// Case-sensitive `(in <tag>)` unit-tag lookup.
fn tag_to_unit(tag: &str) -> Option<Unit> {
    Some(match tag {
        "MiB" => Unit::Megabytes,
        "GiB" => Unit::Gigabytes,
        "KiB" => Unit::Kilobytes,
        "B" => Unit::Bytes,
        "MB" => Unit::Megabytes,
        "GB" => Unit::Gigabytes,
        "KB" => Unit::Kilobytes,
        "TB" => Unit::Terabytes,
        "C" | "°C" => Unit::Celsius,
        "F" | "°F" => Unit::Fahrenheit,
        "K" => Unit::Kelvin,
        "Hz" => Unit::Hertz,
        "MHz" => Unit::Megahertz,
        "GHz" => Unit::Gigahertz,
        "W" => Unit::Watt,
        "mW" => Unit::Milliwatt,
        "J" => Unit::Joule,
        "mJ" => Unit::Millijoule,
        "MJ" => Unit::Megajoule,
        "s" | "sec" => Unit::Seconds,
        "ms" => Unit::Milliseconds,
        "ns" => Unit::Nanoseconds,
        "us" | "µs" => Unit::Microseconds,
        "GB/s" => Unit::GbPerSecond,
        "MB/s" => Unit::MbPerSecond,
        "%" | "percent" => Unit::Percent,
        _ => return None,
    })
}

struct DescriptionPattern {
    regex: Regex,
    unit: Unit,
}

/// Description phrase patterns in priority order.
static DESCRIPTION_PATTERNS: LazyLock<Vec<DescriptionPattern>> = LazyLock::new(|| {
    let specs: &[(&str, Unit)] = &[
        (r"(?i)(?:\bin\s+|\()seconds?(?:\b|\))", Unit::Seconds),
        (
            r"(?i)(?:\bin\s+|\()(?:milliseconds?|ms)(?:\b|\))",
            Unit::Milliseconds,
        ),
        (
            r"(?i)(?:\bin\s+|\()(?:nanoseconds?|ns)(?:\b|\))",
            Unit::Nanoseconds,
        ),
        (r"(?i)(?:\bin\s+|\()bytes?(?:\b|\))", Unit::Bytes),
        (r"(?i)(?:\bin\s+|\()GB/s(?:\b|\))", Unit::GbPerSecond),
        (r"(?i)(?:\bin\s+|\()MB/s(?:\b|\))", Unit::MbPerSecond),
        (
            r"(?i)(?:\bin\s+|\()tokens?/s(?:ec(?:ond)?)?(?:\b|\))",
            Unit::TokensPerSecond,
        ),
        (
            r"(?i)(?:\bin\s+|\()requests?/s(?:ec(?:ond)?)?(?:\b|\))",
            Unit::RequestsPerSecond,
        ),
        (r"(?i)\bin\s+tokens?\b", Unit::Tokens),
    ];
    specs
        .iter()
        .map(|(pattern, unit)| DescriptionPattern {
            regex: Regex::new(pattern).expect("valid description unit regex"),
            unit: *unit,
        })
        .collect()
});

/// `(in <tag>)` capture regex.
static PARENTHETICAL_IN_UNIT: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\(in\s+([^\s)]+)\)").expect("valid parenthetical regex"));

/// Infer from description text, checking parenthetical tags before phrases.
fn unit_from_description(description: Option<&str>) -> Option<Unit> {
    let description = description?;
    if let Some(caps) = PARENTHETICAL_IN_UNIT.captures(description) {
        let tag = caps.get(1).map(|m| m.as_str().trim()).unwrap_or("");
        if let Some(unit) = tag_to_unit(tag) {
            return Some(unit);
        }
    }
    for pattern in DESCRIPTION_PATTERNS.iter() {
        if pattern.regex.is_match(description) {
            return Some(pattern.unit);
        }
    }
    None
}

/// `(0-1)` ratio-range regex.
static RATIO_RANGE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(concat!(
        r"(?i)\(0(?:\.0)?\s*(?:[-–—]+|to)\s*1(?:\.0)?\)",
        r"|\b0(?:\.0)?\s*(?:[-–—]+|to)\s*1(?:\.0)?\b",
        r"|\brange\s+0(?:\.0)?\s*(?:[-–—]+|to)\s*1(?:\.0)?\b",
        r"|(?:\b|\()1(?:\.0)?\s*(?:means|is|equals?|==?)\s*100\s*(?:%|percent)",
    ))
    .expect("valid ratio range regex")
});

/// `(0-100)` percent-range regex.
static PERCENT_RANGE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(concat!(
        r"(?i)\(0(?:\.0)?\s*(?:[-–—]+|to)\s*100(?:\.0)?\)",
        r"|\brange\s+0\s*(?:[-–—]+|to)\s*100\b",
        r"|\b0\s*(?:[-–—]+|to)\s*100\s*%",
    ))
    .expect("valid percent range regex")
});

/// Detect ratio before percent from range indicators.
fn scale_from_description(description: Option<&str>) -> Option<Unit> {
    let description = description?;
    if RATIO_RANGE.is_match(description) {
        return Some(Unit::Ratio);
    }
    if PERCENT_RANGE.is_match(description) {
        return Some(Unit::Percent);
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn suffix_inference_prefers_longest() {
        assert_eq!(infer_unit("latency_seconds", ""), Some(Unit::Seconds));
        assert_eq!(
            infer_unit("kv_cache_bytes_total", ""),
            Some(Unit::Bytes),
            "longest suffix _bytes_total wins over _total"
        );
        // "_requests_total" needs a leading char, so bare "requests_total" falls
        // through to "_total" -> count under longest-suffix precedence.
        assert_eq!(infer_unit("requests_total", ""), Some(Unit::Count));
        assert_eq!(infer_unit("http_requests_total", ""), Some(Unit::Requests));
        assert_eq!(infer_unit("unknown_metric", ""), None);
    }

    #[test]
    fn description_scale_overrides_suffix() {
        // "_percent" suffix but a 0-1 range in the description => ratio.
        assert_eq!(
            infer_unit("cache_hit_percent", "Hit rate (0.0-1.0)"),
            Some(Unit::Ratio)
        );
        assert_eq!(
            infer_unit("util", "Utilization (0-100)"),
            Some(Unit::Percent)
        );
    }

    #[test]
    fn description_parenthetical_unit() {
        assert_eq!(infer_unit("gpu_power", "Power (in W)"), Some(Unit::Watt));
        assert_eq!(infer_unit("mem", "Memory (in MiB)"), Some(Unit::Megabytes));
    }

    #[test]
    fn display_names_match_python_member_names() {
        assert_eq!(Unit::Seconds.display_name(), "seconds");
        assert_eq!(Unit::GbPerSecond.display_name(), "gb/s");
        assert_eq!(Unit::TokensPerSecond.display_name(), "tokens/s");
        assert_eq!(Unit::Percent.display_name(), "percent");
    }
}
