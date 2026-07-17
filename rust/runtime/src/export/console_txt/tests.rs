// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Golden tests for the console-txt sink.
//!
//! Byte-exact `golden/*.txt` files are authoritative Python-oracle output after
//! `rich.text.Text.from_markup(...).plain`; they are not blessable.
//! `golden/*.regression.txt` files cover approximate Rich box layout and may be
//! regenerated with `BLESS=1`.

use super::*;
use crate::metrics_core::{
    AccumulatorSummary, MetricEntry, MetricSeries, NativeReport, ReportCounterStats,
    ReportDistributionStats, ReportError, ReportStats, ReportValue,
};
use std::collections::BTreeMap;


fn empty_report() -> NativeReport {
    NativeReport::new(&AccumulatorSummary::new(), None)
}

fn counter_entry(total: f64) -> MetricEntry {
    MetricEntry {
        metric_type: "counter",
        unit: "requests".to_string(),
        group: "none",
        higher_is_better: false,
        series: vec![MetricSeries {
            labels: None,
            endpoint_url: None,
            stats: ReportStats::Counter(ReportCounterStats {
                total: ReportValue::Finite(total),
                rate: None,
            }),
            timeslices: Vec::new(),
        }],
    }
}

fn dist_entry(unit: &str, avg: f64) -> MetricEntry {
    let mut percentiles = BTreeMap::new();
    percentiles.insert("p50".to_string(), ReportValue::Finite(avg));
    percentiles.insert("p90".to_string(), ReportValue::Finite(avg));
    percentiles.insert("p99".to_string(), ReportValue::Finite(avg));
    MetricEntry {
        metric_type: "distribution",
        unit: unit.to_string(),
        group: "default",
        higher_is_better: false,
        series: vec![MetricSeries {
            labels: None,
            endpoint_url: None,
            stats: ReportStats::Distribution(ReportDistributionStats {
                count: Some(10),
                avg: Some(ReportValue::Finite(avg)),
                min: Some(ReportValue::Finite(avg - 1.0)),
                max: Some(ReportValue::Finite(avg + 1.0)),
                std: Some(ReportValue::Finite(1.0)),
                percentiles,
            }),
            timeslices: Vec::new(),
        }],
    }
}

fn error(code: Option<u16>, error_type: &str, message: &str, count: usize) -> ReportError {
    ReportError {
        code,
        error_type: error_type.to_string(),
        message: message.to_string(),
        count,
    }
}

/// Compare `actual` against a committed byte-exact contract golden (trailing
/// newlines in the file are ignored so an editor cannot corrupt the fixture).
fn assert_contract(name: &str, actual: &str) {
    let golden = match name {
        "osl_mismatch" => include_str!("golden/osl_mismatch.txt"),
        "usage_discrepancy" => include_str!("golden/usage_discrepancy.txt"),
        "api_max_completion_tokens" => include_str!("golden/api_max_completion_tokens.txt"),
        "api_dynamo_session_control" => include_str!("golden/api_dynamo_session_control.txt"),
        other => panic!("unknown contract golden {other}"),
    };
    assert_eq!(
        actual,
        golden.trim_end_matches('\n'),
        "contract golden {name}"
    );
}

/// Compare `actual` against a regression golden, blessing (writing) it when the
/// file is missing or `BLESS` is set in the environment.
fn assert_regression(name: &str, actual: &str) {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("src/export/console_txt/golden")
        .join(format!("{name}.regression.txt"));
    let bless = std::env::var_os("BLESS").is_some() || !path.exists();
    if bless {
        std::fs::write(&path, actual).expect("write regression golden");
        return;
    }
    let expected = std::fs::read_to_string(&path).expect("read regression golden");
    assert_eq!(actual, expected, "regression golden {name}");
}


#[test]
fn osl_mismatch_body_is_byte_exact() {
    let mut report = empty_report();
    report
        .metrics
        .insert("osl_mismatch_count".to_string(), counter_entry(3.0));
    report
        .metrics
        .insert("request_count".to_string(), counter_entry(10.0));
    report.metrics.insert(
        "osl_mismatch_diff_pct".to_string(),
        dist_entry("percent", 7.5),
    );

    let warning = detect_osl_mismatch(&report).expect("osl warning");
    assert_eq!(warning.title, "Output Sequence Length Mismatch Warning");
    assert_contract("osl_mismatch", &warning.body);
}

#[test]
fn osl_mismatch_absent_without_mismatches_or_records() {
    assert!(detect_osl_mismatch(&empty_report()).is_none());

    let mut zero = empty_report();
    zero.metrics
        .insert("osl_mismatch_count".to_string(), counter_entry(0.0));
    zero.metrics
        .insert("request_count".to_string(), counter_entry(10.0));
    assert!(detect_osl_mismatch(&zero).is_none());

    let mut no_records = empty_report();
    no_records
        .metrics
        .insert("osl_mismatch_count".to_string(), counter_entry(3.0));
    no_records
        .metrics
        .insert("request_count".to_string(), counter_entry(0.0));
    assert!(detect_osl_mismatch(&no_records).is_none());
}

#[test]
fn osl_mismatch_reports_na_when_diff_absent() {
    let mut report = empty_report();
    report
        .metrics
        .insert("osl_mismatch_count".to_string(), counter_entry(3.0));
    report
        .metrics
        .insert("request_count".to_string(), counter_entry(10.0));
    let warning = detect_osl_mismatch(&report).expect("osl warning");
    assert!(warning.body.contains("Average mismatch: N/A"));
}


#[test]
fn usage_discrepancy_body_is_byte_exact() {
    let mut report = empty_report();
    report
        .metrics
        .insert("usage_discrepancy_count".to_string(), counter_entry(2.0));
    report
        .metrics
        .insert("request_count".to_string(), counter_entry(10.0));

    let warning = detect_usage_discrepancy(&report).expect("usage warning");
    assert_eq!(warning.title, "Token Count Discrepancy Warning");
    assert_contract("usage_discrepancy", &warning.body);
}

#[test]
fn usage_discrepancy_absent_when_zero() {
    assert!(detect_usage_discrepancy(&empty_report()).is_none());
    let mut zero = empty_report();
    zero.metrics
        .insert("usage_discrepancy_count".to_string(), counter_entry(0.0));
    zero.metrics
        .insert("request_count".to_string(), counter_entry(10.0));
    assert!(detect_usage_discrepancy(&zero).is_none());
}


#[test]
fn max_completion_tokens_body_is_byte_exact() {
    let mut report = empty_report();
    report.errors = vec![error(
        Some(400),
        "BadRequestError",
        "extra_forbidden: max_completion_tokens - Extra inputs are not permitted",
        5,
    )];
    let warning = detect_max_completion_tokens(&report).expect("mct warning");
    assert_eq!(
        warning.title,
        "Unsupported Parameter: max_completion_tokens"
    );
    assert_contract("api_max_completion_tokens", &warning.body);
}

#[test]
fn max_completion_tokens_matches_json_embedded_message() {
    // The trigger tokens live inside a JSON `message` field, not the raw blob.
    let json = serde_json::json!({
        "message": "extra_forbidden for max_completion_tokens: Extra inputs are not permitted",
        "code": 400
    })
    .to_string();
    let mut report = empty_report();
    report.errors = vec![error(Some(400), "BadRequestError", &json, 1)];
    assert!(detect_max_completion_tokens(&report).is_some());
}

#[test]
fn max_completion_tokens_absent_when_tokens_missing() {
    let mut report = empty_report();
    report.errors = vec![error(Some(500), "ServerError", "internal error", 1)];
    assert!(detect_max_completion_tokens(&report).is_none());
}

#[test]
fn dynamo_session_control_body_is_byte_exact() {
    let mut report = empty_report();
    report.errors = vec![error(
        Some(400),
        "BadRequestError",
        "unknown variant `bind`, expected `open` or `close`",
        3,
    )];
    let warning = detect_dynamo_session_control(&report).expect("dynamo warning");
    assert_eq!(
        warning.title,
        "Unsupported Dynamo session_control action: bind"
    );
    assert_contract("api_dynamo_session_control", &warning.body);
}

#[test]
fn dynamo_session_control_is_case_insensitive() {
    let mut report = empty_report();
    report.errors = vec![error(
        Some(400),
        "BadRequestError",
        "Unknown Variant `BIND`",
        1,
    )];
    assert!(detect_dynamo_session_control(&report).is_some());
}

#[test]
fn detect_api_errors_runs_both_detectors_in_order() {
    let mut report = empty_report();
    report.errors = vec![
        error(
            Some(400),
            "A",
            "unknown variant `bind`, expected `open` or `close`",
            1,
        ),
        error(
            Some(400),
            "B",
            "extra_forbidden max_completion_tokens Extra inputs are not permitted",
            1,
        ),
    ];
    let warnings = detect_api_errors(&report);
    assert_eq!(warnings.len(), 2);
    // MaxCompletionTokens is emitted first (Python `DETECTORS` order).
    assert_eq!(
        warnings[0].title,
        "Unsupported Parameter: max_completion_tokens"
    );
    assert_eq!(
        warnings[1].title,
        "Unsupported Dynamo session_control action: bind"
    );
}

#[test]
fn no_api_error_warnings_on_empty_report() {
    assert!(detect_api_errors(&empty_report()).is_empty());
}


#[test]
fn error_summary_table_absent_without_errors() {
    assert!(error_summary_table(&empty_report(), 140).is_none());
}

#[test]
fn error_summary_table_cell_values_are_exact() {
    let mut report = empty_report();
    report.errors = vec![
        error(Some(429), "RateLimit", "Too Many Requests", 1_234),
        error(None, "", "connection reset", 7),
    ];
    let table = error_summary_table(&report, 140).expect("error table");
    // Byte-exact cell content: N/A for the missing code and type, grouped count.
    assert!(table.contains("429"));
    assert!(table.contains("RateLimit"));
    assert!(table.contains("Too Many Requests"));
    assert!(table.contains("1,234"));
    assert!(table.contains("N/A"), "missing code/type render as N/A");
    assert!(table.contains("connection reset"));
    assert!(table.contains("NVIDIA AIPerf | Error Summary"));
    assert_regression("error_summary_table", &table);
}

#[test]
fn error_code_zero_renders_na() {
    let mut report = empty_report();
    report.errors = vec![error(Some(0), "Type", "msg", 1)];
    let table = error_summary_table(&report, 140).expect("error table");
    // Python treats a falsy code (0) as N/A.
    assert!(table.contains("N/A"));
}

#[test]
fn full_render_regression() {
    let mut report = empty_report();
    // A visible default-group metric (request_latency has a catalog spec).
    report
        .metrics
        .insert("request_latency".to_string(), dist_entry("ms", 123.456));
    // Trigger the OSL warning and the error table + API-error panel.
    report
        .metrics
        .insert("osl_mismatch_count".to_string(), counter_entry(3.0));
    report
        .metrics
        .insert("request_count".to_string(), counter_entry(10.0));
    report.metrics.insert(
        "osl_mismatch_diff_pct".to_string(),
        dist_entry("percent", 7.5),
    );
    report.errors = vec![error(
        Some(400),
        "BadRequestError",
        "extra_forbidden max_completion_tokens Extra inputs are not permitted",
        2,
    )];

    // Project one registered metric (request_latency) exactly as the frontend
    // would; osl_mismatch_* / request_count stay unregistered here (rendered as
    // raw tags in the DEFAULT group, matching Python's unregistered-tag path).
    let mut metrics = BTreeMap::new();
    metrics.insert(
        "request_latency".to_string(),
        ConsoleMetricMeta {
            header: "Request Latency".to_string(),
            group: "default".to_string(),
            display_order: Some(30),
            internal: false,
            experimental: false,
            error_only: false,
        },
    );
    let cfg = ConsoleTxtExportConfig {
        enabled: true,
        width: 140,
        dev: false,
        title: "NVIDIA AIPerf | LLM Metrics".to_string(),
        metrics,
    };
    let text = render_console_txt(&report, &cfg);
    // Structural expectations that must hold regardless of box layout.
    assert!(text.contains("Unsupported Parameter: max_completion_tokens"));
    assert!(text.contains("NVIDIA AIPerf | Error Summary"));
    assert!(text.contains("NVIDIA AIPerf")); // metrics table title
    assert!(text.contains("Output Sequence Length Mismatch Warning"));
    assert_regression("full_render", &text);
}

// ---------------------------------------------------------------------------
// Unicode cell-width parity (Rich `cells.cell_len` / `_cell_widths`)
// ---------------------------------------------------------------------------

#[test]
fn cell_width_matches_rich_cell_len() {
    // ASCII is one cell each.
    assert_eq!(cell_width("hello"), 5);
    // CJK ideographs and kana are two cells (East Asian Wide/Fullwidth).
    assert_eq!(char_cell_size('世'), 2);
    assert_eq!(char_cell_size('本'), 2);
    assert_eq!(cell_width("日本語"), 6);
    // Emoji (astral, Emoji_Presentation) are two cells.
    assert_eq!(char_cell_size('🚀'), 2);
    assert_eq!(char_cell_size('🌟'), 2);
    // Combining marks and zero-width space render in zero cells.
    assert_eq!(char_cell_size('\u{0301}'), 0); // combining acute accent
    assert_eq!(char_cell_size('\u{200B}'), 0); // zero-width space
    assert_eq!(cell_width("e\u{0301}"), 1); // e + combining accent = one cell
}

#[test]
fn set_cell_size_crops_wide_glyph_on_boundary() {
    // A wide glyph straddling the crop boundary is dropped and space-padded so
    // the result is exactly `total` cells (Rich `set_cell_size`).
    assert_eq!(set_cell_size("a世b", 2), "a "); // '世' would overflow 2 → drop + space
    assert_eq!(set_cell_size("a世b", 3), "a世"); // fits exactly at 3 cells
    assert_eq!(set_cell_size("abc", 5), "abc  "); // short → right-padded
}

/// Wide/zero-width Unicode drives column-width solving, word wrap, and ellipsis
/// truncation exactly as Rich does. Pins the fix for the `cell_width`-as-char-count
/// bug: an emoji message that overflows on a trailing space must ellipsize (left
/// column keeps trailing whitespace), while a CJK label in a right-justified
/// column is fully right-stripped. Regenerate with `BLESS=1`.
#[test]
fn unicode_table_render_regression() {
    let left = [Justify::Right, Justify::Right, Justify::Left, Justify::Right];
    let emoji = render_table(
        "NVIDIA AIPerf | Error Summary",
        &["Code", "Type", "Message", "Count"],
        &[vec![
            "503".to_string(),
            "EmojiError".to_string(),
            "service unavailable 🚀 please retry later 🔥 the upstream server returned an error 😀 and could not complete the streaming response for this request 🎉 sorry".to_string(),
            "5".to_string(),
        ]],
        &left,
        140,
    );
    assert_regression("unicode_emoji_wrap", &emoji);

    let right: Vec<Justify> = std::iter::repeat_n(Justify::Right, 8).collect();
    let cjk = render_table(
        "NVIDIA AIPerf | LLM Metrics",
        &["Metric", "avg", "min", "max", "p99", "p90", "p50", "std"],
        &[vec![
            "服误 unavailable 😀 dog the brown fox (tokens/sec)".to_string(),
            "211,823.34".to_string(),
            "532,522.69".to_string(),
            "101,587.36".to_string(),
            "692,065.40".to_string(),
            "636,606.25".to_string(),
            "329,729.82".to_string(),
            "83,073.93".to_string(),
        ]],
        &right,
        140,
    );
    assert_regression("unicode_cjk_metric", &cjk);
}

#[test]
fn number_formatting_matches_python() {
    assert_eq!(comma_int(1_234_567), "1,234,567");
    assert_eq!(comma_int(0), "0");
    assert_eq!(comma_int(-1_000), "-1,000");
    assert_eq!(comma_2dp(1_234.5), "1,234.50");
    assert_eq!(comma_2dp(0.0), "0.00");
    assert_eq!(g_fmt(5.0), "5");
    assert_eq!(g_fmt(10.0), "10");
}
