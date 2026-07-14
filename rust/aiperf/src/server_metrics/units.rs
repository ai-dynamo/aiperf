// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Priority-ordered Prometheus unit inference.
//!
//! The scale override, exact DCGM parenthetical tags, phrase patterns, and
//! longest-suffix rules are implemented here.

use std::cell::RefCell;
use std::collections::{HashMap, VecDeque};
use std::sync::LazyLock;

use crate::metrics_core::Unit;
use regex::Regex;

const CACHE_CAPACITY: usize = 2_048;

/// Extension seam for metric-name/HELP-text unit policies.
pub trait UnitInferer {
    /// Infers a display unit, returning `None` when no rule is authoritative.
    fn infer(&self, metric_name: &str, description: Option<&str>) -> Option<Unit>;
}

/// Cached implementation of AIPerf's native Prometheus inference table.
#[derive(Debug, Default)]
pub struct PrometheusUnitInferer {
    cache: RefCell<HashMap<(String, String), Option<Unit>>>,
    order: RefCell<VecDeque<(String, String)>>,
}

impl UnitInferer for PrometheusUnitInferer {
    fn infer(&self, metric_name: &str, description: Option<&str>) -> Option<Unit> {
        let key = (
            metric_name.to_string(),
            description.unwrap_or_default().to_string(),
        );
        if let Some(unit) = self.cache.borrow().get(&key) {
            return *unit;
        }
        let unit = infer_unit(metric_name, description);
        let mut cache = self.cache.borrow_mut();
        let mut order = self.order.borrow_mut();
        if cache.len() >= CACHE_CAPACITY
            && let Some(oldest) = order.pop_front()
        {
            cache.remove(&oldest);
        }
        order.push_back(key.clone());
        cache.insert(key, unit);
        unit
    }
}

/// Infers a unit using description scale, description unit, then metric name.
pub fn infer_unit(metric_name: &str, description: Option<&str>) -> Option<Unit> {
    parse_scale_from_description(description)
        .or_else(|| parse_unit_from_description(description))
        .or_else(|| parse_unit_from_metric_name(metric_name))
}

fn parse_unit_from_metric_name(metric_name: &str) -> Option<Unit> {
    let name = metric_name.to_ascii_lowercase();
    const SUFFIXES: &[(&str, Unit)] = &[
        ("_milliseconds", Unit::Millisecond),
        ("_requests_total", Unit::Request),
        ("_error_count_total", Unit::Count),
        ("_nanoseconds", Unit::Nanosecond),
        ("_tokens_total", Unit::Token),
        ("_blocks_total", Unit::Count),
        ("_seconds_total", Unit::Second),
        ("_gigabytes", Unit::Gigabyte),
        ("_megabytes", Unit::Megabyte),
        ("_kilobytes", Unit::Kilobyte),
        ("_bytes_total", Unit::Byte),
        ("_errors_total", Unit::Count),
        ("_block_count", Unit::Count),
        ("_requests", Unit::Request),
        ("request_success", Unit::Request),
        ("_seconds", Unit::Second),
        ("_ns_total", Unit::Nanosecond),
        ("_ms_total", Unit::Millisecond),
        ("_tokens", Unit::Token),
        ("_errors", Unit::Count),
        ("_celsius", Unit::Celsius),
        ("_percent", Unit::Percent),
        ("_joules", Unit::Joule),
        ("_watts", Unit::Watt),
        ("_bytes", Unit::Byte),
        ("_blocks", Unit::Count),
        ("_count", Unit::Count),
        ("_ratio", Unit::Ratio),
        ("_total", Unit::Count),
        ("_reqs", Unit::Request),
        ("_gb_s", Unit::GigabytesPerSecond),
        ("_perc", Unit::Percent),
        ("_ms", Unit::Millisecond),
        ("_ns", Unit::Nanosecond),
    ];
    SUFFIXES
        .iter()
        .find_map(|(suffix, unit)| name.ends_with(suffix).then_some(*unit))
        .or_else(|| name.contains("num_requests_").then_some(Unit::Request))
}

static PARENTHETICAL_UNIT: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\(in\s+([^\s\)]+)\)").expect("static unit regex must compile"));

static RATIO_RANGE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?i)(?:\(?0(?:\.0)?\s*(?:[-–—]+|to)\s*1(?:\.0)?(?:\b|\))|(?:^|\b|\()1(?:\.0)?\s*(?:means|is|equals?|==?)\s*100\s*(?:%|percent))",
    )
    .expect("static ratio regex must compile")
});

static PERCENT_RANGE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)(?:\(?0(?:\.0)?\s*(?:[-–—]+|to)\s*100(?:\.0)?\)?\s*%?)")
        .expect("static percent regex must compile")
});

fn parse_scale_from_description(description: Option<&str>) -> Option<Unit> {
    let description = description?;
    if RATIO_RANGE.is_match(description) {
        Some(Unit::Ratio)
    } else if PERCENT_RANGE.is_match(description) {
        Some(Unit::Percent)
    } else {
        None
    }
}

fn parse_unit_from_description(description: Option<&str>) -> Option<Unit> {
    let description = description?;
    if let Some(captures) = PARENTHETICAL_UNIT.captures(description) {
        let tag = captures.get(1)?.as_str();
        if let Some(unit) = exact_unit_tag(tag) {
            return Some(unit);
        }
    }
    let lower = description.to_ascii_lowercase();
    const PHRASES: &[(&str, Unit)] = &[
        ("in milliseconds", Unit::Millisecond),
        ("(milliseconds", Unit::Millisecond),
        ("in nanoseconds", Unit::Nanosecond),
        ("(nanoseconds", Unit::Nanosecond),
        ("in seconds", Unit::Second),
        ("(seconds", Unit::Second),
        ("in requests/sec", Unit::RequestsPerSecond),
        ("in requests/s", Unit::RequestsPerSecond),
        ("(requests/sec", Unit::RequestsPerSecond),
        ("(requests/s", Unit::RequestsPerSecond),
        ("in tokens/sec", Unit::TokensPerSecond),
        ("in tokens/s", Unit::TokensPerSecond),
        ("(tokens/sec", Unit::TokensPerSecond),
        ("(tokens/s", Unit::TokensPerSecond),
        ("in gb/s", Unit::GigabytesPerSecond),
        ("(gb/s", Unit::GigabytesPerSecond),
        ("in mb/s", Unit::MegabytesPerSecond),
        ("(mb/s", Unit::MegabytesPerSecond),
        ("in bytes", Unit::Byte),
        ("(bytes", Unit::Byte),
        ("in tokens", Unit::Token),
        ("in ms", Unit::Millisecond),
        ("(ms)", Unit::Millisecond),
        ("in ns", Unit::Nanosecond),
        ("(ns)", Unit::Nanosecond),
    ];
    PHRASES
        .iter()
        .find_map(|(phrase, unit)| lower.contains(phrase).then_some(*unit))
}

fn exact_unit_tag(tag: &str) -> Option<Unit> {
    match tag {
        "MiB" | "MB" => Some(Unit::Megabyte),
        "GiB" | "GB" => Some(Unit::Gigabyte),
        "KiB" | "KB" => Some(Unit::Kilobyte),
        "B" => Some(Unit::Byte),
        "TB" => Some(Unit::Terabyte),
        "C" | "°C" => Some(Unit::Celsius),
        "F" | "°F" => Some(Unit::Fahrenheit),
        "K" => Some(Unit::Kelvin),
        "Hz" => Some(Unit::Hertz),
        "MHz" => Some(Unit::Megahertz),
        "GHz" => Some(Unit::Gigahertz),
        "W" => Some(Unit::Watt),
        "mW" => Some(Unit::Milliwatt),
        "J" => Some(Unit::Joule),
        "mJ" => Some(Unit::Millijoule),
        "MJ" => Some(Unit::Megajoule),
        "s" | "sec" => Some(Unit::Second),
        "ms" => Some(Unit::Millisecond),
        "ns" => Some(Unit::Nanosecond),
        "us" | "µs" => Some(Unit::Microsecond),
        "GB/s" => Some(Unit::GigabytesPerSecond),
        "MB/s" => Some(Unit::MegabytesPerSecond),
        "%" | "percent" => Some(Unit::Percent),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn longest_suffixes_preserve_semantic_quantities() {
        assert_eq!(
            infer_unit("server_requests_total", None),
            Some(Unit::Request)
        );
        assert_eq!(
            infer_unit("vllm:iteration_tokens_total", None),
            Some(Unit::Token)
        );
        assert_eq!(
            infer_unit("latency_milliseconds", None),
            Some(Unit::Millisecond)
        );
        assert_eq!(
            infer_unit("num_requests_running", None),
            Some(Unit::Request)
        );
        assert_eq!(infer_unit("mystery", None), None);
    }

    #[test]
    fn description_scale_overrides_a_misleading_name() {
        assert_eq!(
            infer_unit("cache_hit_percent", Some("Hit rate (0.0-1.0)")),
            Some(Unit::Ratio)
        );
        assert_eq!(
            infer_unit("cache_hit_ratio", Some("Hit rate range 0 to 100%")),
            Some(Unit::Percent)
        );
    }

    #[test]
    fn parenthetical_tags_are_case_sensitive() {
        assert_eq!(
            infer_unit("energy", Some("Energy (in mJ)")),
            Some(Unit::Millijoule)
        );
        assert_eq!(
            infer_unit("energy", Some("Energy (in MJ)")),
            Some(Unit::Megajoule)
        );
        assert_eq!(
            infer_unit("memory", Some("Used (in MiB)")),
            Some(Unit::Megabyte)
        );
        assert_eq!(
            infer_unit("latency", Some("Latency (in µs)")),
            Some(Unit::Microsecond)
        );
    }

    #[test]
    fn cached_inferer_matches_pure_policy() {
        let inferer = PrometheusUnitInferer::default();
        assert_eq!(
            inferer.infer("latency", Some("Request latency in seconds")),
            Some(Unit::Second)
        );
        assert_eq!(
            inferer.infer("latency", Some("Request latency in seconds")),
            Some(Unit::Second)
        );
        assert_eq!(inferer.cache.borrow().len(), 1);
    }
}
