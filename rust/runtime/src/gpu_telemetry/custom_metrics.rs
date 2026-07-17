// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! DCGM-style custom GPU-metrics CSV loading.
//!
//! Behavior matches the canonical Python loader
//! `src/aiperf/gpu_telemetry/metrics_config.py::MetricsConfigLoader`
//! (`parse_custom_metrics_csv`, `build_custom_metrics_from_csv`,
//! `_infer_unit_from_help`, `_title_case_metric_name`) and its default
//! dedup catalog `src/aiperf/gpu_telemetry/constants.py::DCGM_TO_FIELD_MAPPING`.
//!
//! A `--gpu-telemetry <file>.csv` supplies additional DCGM exporter fields that
//! the native [`DcgmPrometheusDecoder`](crate::gpu_telemetry::parser::DcgmPrometheusDecoder)
//! would otherwise drop (only the built-in [`DCGM_METRICS`](crate::gpu_telemetry::fields::DCGM_METRICS)
//! source fields are decoded). Loading a CSV yields two products:
//!
//! * [`LoadedCustomMetrics::decoder_fields`] — a `source_field -> `
//!   [`CustomDcgmField`] map injected into the decoder so the raw Prometheus
//!   samples are extracted and named, and
//! * [`LoadedCustomMetrics::specs`] — [`RuntimeGpuMetricSpec`] rows registered
//!   with the accumulator so the summarizer surfaces each custom signal with
//!   its inferred unit (the summarizer iterates registered specs; an unregistered
//!   name is silently dropped even when its value is scraped).
//!
//! A missing or unreadable CSV path is a hard error ([`CustomMetricsError`]) so
//! the runner fails closed rather than silently benchmarking with defaults.

use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::fmt::{Display, Formatter, Result as FmtResult};
use std::path::{Path, PathBuf};

use crate::gpu_telemetry::fields::{DCGM_METRICS, GpuMetricKind, RuntimeGpuMetricSpec};
use crate::metrics_core::Unit;

/// One custom DCGM source field resolved from a metrics CSV.
///
/// Custom fields carry no collector scaling (Python's custom collector reports
/// raw exporter values), so [`scale`](Self::scale) is always `1.0`.
#[derive(Debug, Clone, PartialEq)]
pub struct CustomDcgmField {
    /// Normalized telemetry name (`DCGM_FI_DEV_SM_CLOCK` -> `sm_clock`).
    pub name: String,
    /// Multiplier applied to the raw exporter value; custom fields use `1.0`.
    pub scale: f64,
    /// Gauge/counter accumulation policy from the CSV `metric_type` column.
    pub kind: GpuMetricKind,
}

/// The decoder field map and accumulator specs produced from one metrics CSV.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct LoadedCustomMetrics {
    /// `source_field -> CustomDcgmField`, injected into the DCGM decoder.
    pub decoder_fields: BTreeMap<String, CustomDcgmField>,
    /// Registered metric specs, added to the GPU telemetry accumulator.
    pub specs: Vec<RuntimeGpuMetricSpec>,
}

/// Failure loading a custom GPU-metrics CSV.
#[derive(Debug)]
pub enum CustomMetricsError {
    /// The CSV path could not be read (missing file, permission, etc.).
    Read {
        /// Requested CSV path.
        path: PathBuf,
        /// Underlying IO failure.
        source: std::io::Error,
    },
}

impl Display for CustomMetricsError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> FmtResult {
        match self {
            Self::Read { path, source } => write!(
                formatter,
                "GPU metrics file could not be read: {} ({source})",
                path.display()
            ),
        }
    }
}

impl std::error::Error for CustomMetricsError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Read { source, .. } => Some(source),
        }
    }
}

/// Reads and parses a DCGM-style custom metrics CSV at `path`.
///
/// Mirrors `MetricsConfigLoader.build_custom_metrics_from_csv`: a missing file
/// is an error (Python opens the file inside `parse_custom_metrics_csv`; the
/// native front door validates `.csv` existence at parse time, and this final
/// read is the fail-closed backstop). Malformed *rows* are skipped, not fatal.
pub fn load_custom_dcgm_metrics(path: &Path) -> Result<LoadedCustomMetrics, CustomMetricsError> {
    let text = std::fs::read_to_string(path).map_err(|source| CustomMetricsError::Read {
        path: path.to_path_buf(),
        source,
    })?;
    Ok(parse_custom_metrics(&text))
}

/// Parses already-read CSV text into decoder fields and accumulator specs.
///
/// Split out from the IO so the row-level parsing rules stay unit-testable.
pub fn parse_custom_metrics(text: &str) -> LoadedCustomMetrics {
    // Dedup against the built-in DCGM catalog by both source field and
    // normalized name, matching Python's `existing_dcgm_fields` /
    // `existing_field_names` guards (a built-in field must not be re-added).
    let builtin_source_fields: BTreeSet<&str> =
        DCGM_METRICS.iter().map(|spec| spec.source_field).collect();
    let builtin_names: BTreeSet<&str> = DCGM_METRICS.iter().map(|spec| spec.name).collect();

    let mut loaded = LoadedCustomMetrics::default();
    let mut seen_names: BTreeSet<String> = BTreeSet::new();

    for raw_line in text.lines() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        // Split on the first two commas only, preserving commas in the help
        // message (`line.split(",", 2)` in Python).
        let parts: Vec<&str> = line.splitn(3, ',').map(str::trim).collect();
        if parts.len() != 3 {
            continue;
        }
        let (dcgm_field, metric_type, help_msg) = (parts[0], parts[1], parts[2]);

        // Only DCGM feature-identifier fields, and only gauge/counter rows.
        if !dcgm_field.starts_with("DCGM_FI_") {
            continue;
        }
        let kind = match metric_type {
            "gauge" => GpuMetricKind::Gauge,
            "counter" => GpuMetricKind::Counter,
            _ => continue,
        };

        // A field already served by a built-in spec is not re-registered; its
        // value still decodes and reports through the default mapping.
        if builtin_source_fields.contains(dcgm_field) {
            continue;
        }

        let internal_name = dcgm_field.replace("DCGM_FI_DEV_", "").to_lowercase();
        if internal_name.is_empty()
            || builtin_names.contains(internal_name.as_str())
            || !seen_names.insert(internal_name.clone())
        {
            continue;
        }

        let mut display_name = help_msg.split('(').next().unwrap_or("").trim().to_string();
        if display_name.is_empty() {
            display_name = internal_name.replace('_', " ");
        }
        let header = title_case_metric_name(&display_name);
        let unit = infer_unit_from_help(help_msg);

        loaded.decoder_fields.insert(
            dcgm_field.to_string(),
            CustomDcgmField {
                name: internal_name.clone(),
                scale: 1.0,
                kind,
            },
        );
        loaded.specs.push(RuntimeGpuMetricSpec {
            name: internal_name,
            header,
            unit,
            kind,
        });
    }

    loaded
}

/// Infers a metric unit from a `... (in UNIT)` help message.
///
/// Matches `\(in\s+([^\s)]+)\)` case-insensitively,
/// lowercases the captured token, and maps it to a native [`Unit`]. Anything
/// unrecognized falls back to [`Unit::Count`].
fn infer_unit_from_help(help_msg: &str) -> Unit {
    let lower = help_msg.to_lowercase();
    // Scan every `(in` occurrence; the regex requires whitespace then a token
    // then `)`, so a non-matching occurrence (e.g. `(input`) must not abort the
    // search for a later valid `(in UNIT)`.
    let mut search_from = 0;
    while let Some(found) = lower[search_from..].find("(in") {
        let token_start = search_from + found + "(in".len();
        search_from = token_start;
        let after = &lower[token_start..];
        let trimmed = after.trim_start();
        // `\s+` requires at least one whitespace character after `in`.
        if trimmed.len() == after.len() {
            continue;
        }
        let token: String = trimmed
            .chars()
            .take_while(|c| !c.is_whitespace() && *c != ')')
            .collect();
        if token.is_empty() {
            continue;
        }
        if !trimmed[token.len()..].starts_with(')') {
            continue;
        }
        return unit_from_token(&token);
    }
    Unit::Count
}

/// Maps a lowercased unit token to a native [`Unit`] (Python `unit_mapping`).
fn unit_from_token(token: &str) -> Unit {
    match token {
        "w" => Unit::Watt,
        "%" | "percent" => Unit::Percent,
        "gb" => Unit::Gigabyte,
        "mb" => Unit::Megabyte,
        "kb" => Unit::Kilobyte,
        "mhz" => Unit::Megahertz,
        "ghz" => Unit::Gigahertz,
        "c" | "°c" | "celsius" => Unit::Celsius,
        "count" => Unit::Count,
        "us" => Unit::Microsecond,
        "ms" => Unit::Millisecond,
        "s" => Unit::Second,
        "mj" => Unit::Megajoule,
        "j" => Unit::Joule,
        _ => Unit::Count,
    }
}

/// Acronyms kept fully capitalized by [`title_case_metric_name`].
const ACRONYMS: &[&str] = &[
    "gpu", "xid", "sm", "nvlink", "pci", "pcie", "cpu", "ram", "vram", "ecc",
];

/// Title-cases a metric display name (`_title_case_metric_name`), keeping known
/// acronyms fully uppercase and every other word Python-`str.capitalize`-cased.
fn title_case_metric_name(name: &str) -> String {
    name.split_whitespace()
        .map(|word| {
            if ACRONYMS.contains(&word.to_lowercase().as_str()) {
                word.to_uppercase()
            } else {
                capitalize(word)
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// Replicates Python `str.capitalize`: first character upper, all others lower.
fn capitalize(word: &str) -> String {
    let mut chars = word.chars();
    match chars.next() {
        None => String::new(),
        Some(first) => first.to_uppercase().collect::<String>() + &chars.as_str().to_lowercase(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_csv_dedups_defaults_and_registers_custom_fields_with_units() {
        // Mirrors `custom_gpu_metrics_csv` in test_custom_gpu_metrics.rs.
        let csv = "# Custom GPU Metrics Test File\n\
             DCGM_FI_DEV_SM_CLOCK, gauge, SM clock frequency (in MHz)\n\
             DCGM_FI_DEV_MEM_CLOCK, gauge, Memory clock frequency (in MHz)\n\
             DCGM_FI_DEV_MEMORY_TEMP, gauge, Memory temperature (in °C)\n\
             DCGM_FI_DEV_MEM_COPY_UTIL, gauge, Memory copy utilization (in %)\n";
        let loaded = parse_custom_metrics(csv);

        // MEM_COPY_UTIL is a built-in (maps to mem_utilization) -> deduped.
        assert_eq!(loaded.specs.len(), 3);
        assert!(
            !loaded
                .decoder_fields
                .contains_key("DCGM_FI_DEV_MEM_COPY_UTIL")
        );

        let by_name: BTreeMap<&str, &RuntimeGpuMetricSpec> =
            loaded.specs.iter().map(|s| (s.name.as_str(), s)).collect();
        assert_eq!(by_name["sm_clock"].unit, Unit::Megahertz);
        assert_eq!(by_name["sm_clock"].header, "SM Clock Frequency");
        assert_eq!(by_name["mem_clock"].unit, Unit::Megahertz);
        assert_eq!(by_name["memory_temp"].unit, Unit::Celsius);
        assert_eq!(by_name["memory_temp"].header, "Memory Temperature");

        assert_eq!(
            loaded.decoder_fields["DCGM_FI_DEV_SM_CLOCK"],
            CustomDcgmField {
                name: "sm_clock".to_string(),
                scale: 1.0,
                kind: GpuMetricKind::Gauge,
            }
        );
    }

    #[test]
    fn invalid_rows_are_skipped_but_valid_custom_survives() {
        // Mirrors `custom_gpu_metrics_csv_invalid`.
        let csv = "INVALID_FIELD, gauge, Invalid field name\n\
             DCGM_FI_DEV_GPU_UTIL, invalid_type, Invalid metric type\n\
             DCGM_FI_DEV_SM_CLOCK, gauge, SM clock frequency (in MHz)\n";
        let loaded = parse_custom_metrics(csv);
        assert_eq!(loaded.specs.len(), 1);
        assert_eq!(loaded.specs[0].name, "sm_clock");
        assert!(loaded.decoder_fields.contains_key("DCGM_FI_DEV_SM_CLOCK"));
    }

    #[test]
    fn defaults_in_csv_are_deduplicated() {
        // Mirrors `custom_gpu_metrics_csv_with_defaults`.
        let csv = "DCGM_FI_DEV_GPU_UTIL, gauge, GPU utilization (in %)\n\
             DCGM_FI_DEV_POWER_USAGE, gauge, Power draw (in W)\n\
             DCGM_FI_DEV_SM_CLOCK, gauge, SM clock frequency (in MHz)\n\
             DCGM_FI_DEV_MEM_CLOCK, gauge, Memory clock frequency (in MHz)\n";
        let loaded = parse_custom_metrics(csv);
        let names: BTreeSet<&str> = loaded.specs.iter().map(|s| s.name.as_str()).collect();
        assert_eq!(names, BTreeSet::from(["sm_clock", "mem_clock"]));
    }

    #[test]
    fn missing_file_is_an_error() {
        let error =
            load_custom_dcgm_metrics(Path::new("/nonexistent/custom_metrics.csv")).unwrap_err();
        assert!(matches!(error, CustomMetricsError::Read { .. }));
    }

    #[test]
    fn unit_inference_matches_python_tokens() {
        assert_eq!(infer_unit_from_help("Power draw (in W)"), Unit::Watt);
        assert_eq!(infer_unit_from_help("Util (in %)"), Unit::Percent);
        assert_eq!(infer_unit_from_help("Clock (in MHz)"), Unit::Megahertz);
        assert_eq!(infer_unit_from_help("Temp (in °C)"), Unit::Celsius);
        assert_eq!(infer_unit_from_help("Unlabeled help"), Unit::Count);
        // A leading `(input ...)` must not abort a later valid `(in MHz)`.
        assert_eq!(
            infer_unit_from_help("Clock (input pin) freq (in MHz)"),
            Unit::Megahertz
        );
    }

    #[test]
    fn title_case_keeps_acronyms_upper() {
        assert_eq!(title_case_metric_name("gpu power usage"), "GPU Power Usage");
        assert_eq!(title_case_metric_name("xid errors"), "XID Errors");
        assert_eq!(
            title_case_metric_name("sm clock frequency"),
            "SM Clock Frequency"
        );
    }
}
