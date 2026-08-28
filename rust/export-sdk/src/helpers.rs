// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Pure formatting, naming, and numeric-lowering helpers shared by every
//! exporter plugin.
//!
//! Nothing here reads a clock, touches the filesystem, or names a host type.
//! Each function is the single implementation of a behavior several exporters
//! would otherwise re-derive: the CRLF CSV dialect every AIPerf CSV artifact
//! uses, the endpoint key the telemetry summaries render, the default run name
//! the tracking backends fall back to, and the two finiteness policies a report
//! value can be lowered under.
//!
//! # Report values on the boundary
//!
//! The finalized report crosses the plugin boundary as
//! [`aiperf_core::capture::FinalReportV1`], whose payload is the exact JSON the
//! host commits. A report value is therefore a [`serde_json::Value`]: a JSON
//! number for a finite value and `null` for a present-but-non-finite one.
//! [`finite_passthrough`] and [`finite_guarded`] preserve the two distinct host
//! policies over that representation so an exporter leaf keeps its original
//! behavior after the split.

use serde_json::Value;

/// Build the CRLF-terminated writer shared by all CSV artifacts.
///
/// AIPerf CSV artifacts are RFC-4180 CRLF-terminated regardless of host
/// platform, so a run's artifacts are byte-identical everywhere. Every exporter
/// that emits CSV builds its writer here rather than configuring `csv` itself.
pub fn crlf_csv_writer<W: std::io::Write>(writer: W) -> csv::Writer<W> {
    csv::WriterBuilder::new()
        .terminator(csv::Terminator::CRLF)
        .from_writer(writer)
}

/// Drop the URL scheme, query, fragment, and terminal `/metrics` path component.
///
/// Shared by the telemetry summary and the server-metrics exporter so both
/// render the same endpoint keys. Netloc (host, port, any userinfo) is preserved
/// verbatim, so `http://127.0.0.1:9400/dcgm1/metrics` becomes
/// `127.0.0.1:9400/dcgm1`.
pub fn normalize_endpoint_display(url: &str) -> String {
    let after_scheme = match url.find("://") {
        Some(index) => &url[index + 3..],
        None => url,
    };
    let netloc_end = after_scheme
        .find(['/', '?', '#'])
        .unwrap_or(after_scheme.len());
    let netloc = &after_scheme[..netloc_end];
    let rest = &after_scheme[netloc_end..];
    let path_end = rest.find(['?', '#']).unwrap_or(rest.len());
    let path = &rest[..path_end];
    let path = if path.starts_with('/') { path } else { "" };
    let path = path.strip_suffix("/metrics").unwrap_or(path);
    let mut display = netloc.to_string();
    if !path.is_empty() {
        display.push_str(path);
    }
    display
}

/// Derive a default run name `aiperf-<benchmark_id[:8]>` from a benchmark id.
///
/// The id is truncated on a character boundary (never byte-sliced, which panics
/// on a multibyte id). When the id is absent or empty, `fallback` supplies the
/// name; each exporter passes its own no-id fallback because the tracking
/// backends disagree on what an unnamed run should be called.
pub fn default_run_name(benchmark_id: Option<&str>, fallback: impl FnOnce() -> String) -> String {
    match benchmark_id {
        Some(id) if !id.is_empty() => {
            let id8: String = id.chars().take(8).collect();
            format!("aiperf-{id8}")
        }
        _ => fallback(),
    }
}

/// Lower one boundary report value to its `f64` by passthrough.
///
/// A JSON number always yields its payload and every other shape — including the
/// `null` a non-finite value serializes as — yields `None`. The number is
/// trusted as-is: the projection that constructed the report already decided
/// finiteness. Sinks that additionally reject a non-finite payload use
/// [`finite_guarded`].
pub fn finite_passthrough(value: &Value) -> Option<f64> {
    match value {
        Value::Number(number) => number.as_f64(),
        _ => None,
    }
}

/// Lower one boundary report value to its `f64`, dropping a non-finite payload.
///
/// Identical to [`finite_passthrough`] except that a number which is not
/// `is_finite` is refused rather than forwarded. Sinks whose backend rejects
/// NaN/inf outright (the tracking and OTLP exporters) use this; sinks that trust
/// the report's own finiteness use [`finite_passthrough`].
pub fn finite_guarded(value: &Value) -> Option<f64> {
    match value {
        Value::Number(number) => number.as_f64().filter(|number| number.is_finite()),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn endpoint_display_drops_scheme_query_and_metrics_suffix() {
        assert_eq!(
            normalize_endpoint_display("http://127.0.0.1:9400/dcgm1/metrics"),
            "127.0.0.1:9400/dcgm1"
        );
        assert_eq!(
            normalize_endpoint_display("https://host:9400/metrics?job=a#f"),
            "host:9400"
        );
        assert_eq!(normalize_endpoint_display("host:8000"), "host:8000");
    }

    #[test]
    fn default_run_name_truncates_on_a_character_boundary() {
        assert_eq!(
            default_run_name(Some("0123456789"), || "unused".to_owned()),
            "aiperf-01234567"
        );
        assert_eq!(
            default_run_name(Some("ααααααααα"), || "unused".to_owned()),
            "aiperf-αααααααα"
        );
        assert_eq!(default_run_name(None, || "fallback".to_owned()), "fallback");
        assert_eq!(
            default_run_name(Some(""), || "fallback".to_owned()),
            "fallback"
        );
    }

    #[test]
    fn the_two_finiteness_policies_differ_only_on_a_non_finite_number() {
        let finite = serde_json::json!(1.5);
        assert_eq!(finite_passthrough(&finite), Some(1.5));
        assert_eq!(finite_guarded(&finite), Some(1.5));
        assert_eq!(finite_passthrough(&Value::Null), None);
        assert_eq!(finite_guarded(&Value::Null), None);
    }

    #[test]
    fn csv_rows_are_crlf_terminated() {
        let mut writer = crlf_csv_writer(Vec::new());
        writer.write_record(["a", "b"]).expect("write record");
        let written = writer.into_inner().expect("flush");
        assert_eq!(written, b"a,b\r\n");
    }
}
