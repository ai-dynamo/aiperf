// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Fixed-width `profile_export_console.txt` artifact and warning sink.
//!
//! Output is rendered at the configured width, independent of terminal width.
//! Sections appear in this order: API-error panels, error summary, grouped
//! metrics, speculative-decoding metrics, usage-discrepancy warning, and
//! OSL-mismatch warning. Warning bodies, table cells, blank-line prefixes, box
//! glyphs, wrapping, and omission rules are byte-stable.
//!
//! Metric metadata supplies each tag's header, group, display order, and flags.
//! Tags absent from that metadata render under their raw names in the default
//! group. Multi-series metrics use their sole unlabeled aggregate; metrics with
//! no unique aggregate are omitted. Internal, experimental, error-only, cache
//! hint, and HTTP-trace output is excluded.
//!
//! Tables use one-cell padding, heavy-header glyphs, centered titles,
//! width-constrained wrapping, and ellipsis overflow. Panels use rounded glyphs.
//! Metric headers longer than 30 characters place the unit on a second line.

use std::collections::BTreeMap;
use std::path::Path;

mod cell_widths;

use crate::export::{ExportConfig, Exporter, normalize_endpoint_display};
use crate::metrics_core::{
    MetricConsoleGroup, MetricEntry, MetricSeries, NativeReport, ReportStats, ReportValue,
};

/// Artifact filename.
const CONSOLE_TXT_FILENAME: &str = "profile_export_console.txt";

/// OSL mismatch percentage threshold.
const OSL_MISMATCH_PCT_THRESHOLD: f64 = 5.0;
/// OSL mismatch token threshold.
const OSL_MISMATCH_MAX_TOKEN_THRESHOLD: u64 = 50;
/// Usage discrepancy percentage threshold.
const USAGE_PCT_DIFF_THRESHOLD: f64 = 10.0;

/// Console-artifact export policy. Enabled by default (the `.txt` artifact is a
/// stable CI-log surface); the fixed render width is carried here.
///
/// Configuration controls grouping, headers, display order, filtering, and
/// table titles.
#[derive(Debug, Clone, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ConsoleTxtExportConfig {
    /// Emit `profile_export_console.txt`.
    pub enabled: bool,
    /// Fixed render width; defaults to 140.
    pub width: u16,
    /// Include internal or experimental metrics in development mode.
    pub dev: bool,
    /// Base metrics title. The default group uses it verbatim; other groups
    /// append `: <Group>`.
    pub title: String,
    /// Console metadata keyed by metric tag. Absent tags render in the default
    /// group under the raw tag, sort last, and carry no filtering flags.
    #[serde(default)]
    pub metrics: BTreeMap<String, ConsoleMetricMeta>,
    /// Configured model names used to select model-labeled server metrics.
    #[serde(default)]
    pub model_names: Vec<String>,
}

impl Default for ConsoleTxtExportConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            width: 140,
            dev: false,
            title: "NVIDIA AIPerf".to_string(),
            metrics: BTreeMap::new(),
            model_names: Vec::new(),
        }
    }
}

/// Per-tag display and filtering metadata.
#[derive(Debug, Clone, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ConsoleMetricMeta {
    /// Display header.
    pub header: String,
    /// Console group, such as `usage` or `default`.
    pub group: String,
    /// Display order; absent values sort last.
    #[serde(default)]
    pub display_order: Option<u32>,
    /// Whether the metric is internal.
    #[serde(default)]
    pub internal: bool,
    /// Whether the metric is experimental.
    #[serde(default)]
    pub experimental: bool,
    /// Whether the metric is error-only.
    #[serde(default)]
    pub error_only: bool,
}

/// A detected warning/insight panel: a centered title and an already
/// markup-stripped, value-substituted body. `body` is byte-exact; rendering uses
/// a rounded box.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Warning {
    /// Panel title (centered in the top border).
    pub title: String,
    /// Panel body — the byte-exact insight text.
    pub body: String,
}

/// The fixed-width console-artifact [`Exporter`].
pub struct ConsoleTxtExporter;

impl Exporter for ConsoleTxtExporter {
    fn name(&self) -> &'static str {
        "console_txt"
    }

    fn enabled(&self, cfg: &ExportConfig) -> bool {
        cfg.console_txt.enabled
    }

    fn export(
        &self,
        report: &NativeReport,
        artifact_dir: &Path,
        cfg: &ExportConfig,
    ) -> anyhow::Result<()> {
        let text = render_console_txt(report, &cfg.console_txt);
        let path = artifact_dir.join(CONSOLE_TXT_FILENAME);
        std::fs::write(&path, text)
            .map_err(|error| anyhow::anyhow!("write {}: {error}", path.display()))?;
        Ok(())
    }
}

/// Render the full `profile_export_console.txt` body.
///
/// Render the sections in their artifact order.
pub(crate) fn render_console_txt(report: &NativeReport, cfg: &ConsoleTxtExportConfig) -> String {
    let width = cfg.width as usize;
    // Panels have one leading blank line; table blocks have two. Grouped metric
    // tables form one block and therefore share one two-line prefix.
    let mut blocks: Vec<(usize, String)> = Vec::new();

    for warning in detect_api_errors(report) {
        blocks.push((1, warning_panel(&warning, width)));
    }
    if let Some(table) = error_summary_table(report, width) {
        blocks.push((2, table));
    }
    if let Some(tables) = metrics_tables(report, cfg, width) {
        blocks.push((2, tables));
    }
    if let Some(table) = speculative_decoding_table(report, cfg, width) {
        blocks.push((2, table));
    }
    if let Some(warning) = detect_usage_discrepancy(report) {
        blocks.push((1, warning_panel(&warning, width)));
    }
    if let Some(warning) = detect_osl_mismatch(report) {
        blocks.push((1, warning_panel(&warning, width)));
    }

    // Every block ends with one line terminator.
    let mut out = String::new();
    for (lead, block) in &blocks {
        for _ in 0..*lead {
            out.push('\n');
        }
        out.push_str(block);
        out.push('\n');
    }
    out
}

/// Numeric payload of a [`ReportValue`]; non-finite values are absent.
fn value_f64(value: &ReportValue) -> Option<f64> {
    match value {
        ReportValue::Finite(v) => Some(*v),
        ReportValue::NonFinite => None,
    }
}

/// Aggregate counters add across series; other metric kinds use the first
/// series' representative value.
fn metric_avg(report: &NativeReport, tag: &str) -> Option<f64> {
    let entry = report.metrics.get(tag)?;
    let mut counter_sum: Option<f64> = None;
    for series in &entry.series {
        match &series.stats {
            ReportStats::Counter(counter) => {
                if let Some(v) = value_f64(&counter.total) {
                    *counter_sum.get_or_insert(0.0) += v;
                }
            }
            ReportStats::Distribution(dist) => return dist.avg.as_ref().and_then(value_f64),
            ReportStats::Scalar(scalar) => return value_f64(&scalar.value),
            ReportStats::Histogram(hist) => return hist.avg.as_ref().and_then(value_f64),
        }
    }
    counter_sum
}

/// Insert thousands separators into a non-negative integer digit string.
fn group_thousands(digits: &str) -> String {
    let bytes = digits.as_bytes();
    let first = bytes.len() % 3;
    let mut out = String::with_capacity(digits.len() + digits.len() / 3);
    for (index, byte) in bytes.iter().enumerate() {
        if index != 0 && index >= first && (index - first).is_multiple_of(3) {
            out.push(',');
        }
        out.push(*byte as char);
    }
    out
}

/// Format a signed integer with comma-grouped thousands.
fn comma_int(n: i64) -> String {
    let sign = if n < 0 { "-" } else { "" };
    format!("{sign}{}", group_thousands(&n.unsigned_abs().to_string()))
}

/// Format a number with comma-grouped thousands and fixed decimal precision.
fn comma_fixed(v: f64, precision: usize) -> String {
    let sign = if v.is_sign_negative() && v != 0.0 {
        "-"
    } else {
        ""
    };
    let s = format!("{:.precision$}", v.abs());
    let (int, frac) = s.split_once('.').unwrap_or((s.as_str(), ""));
    format!("{sign}{}.{frac}", group_thousands(int))
}

/// Format a number with comma-grouped thousands and two fixed decimals.
fn comma_2dp(v: f64) -> String {
    comma_fixed(v, 2)
}

/// Format integer-valued thresholds without a trailing decimal.
fn g_fmt(v: f64) -> String {
    format!("{v}")
}

/// Emit the OSL-mismatch warning when both mismatch and request counts are positive.
pub(crate) fn detect_osl_mismatch(report: &NativeReport) -> Option<Warning> {
    let mismatch_avg = metric_avg(report, "osl_mismatch_count")?;
    if mismatch_avg <= 0.0 {
        return None;
    }
    let mismatch_count = mismatch_avg.trunc() as i64;
    let total = metric_avg(report, "request_count")
        .map(|v| v.trunc() as i64)
        .filter(|&t| t != 0)?;
    let percentage = (mismatch_count as f64 / total as f64) * 100.0;
    let avg_diff = metric_avg(report, "osl_mismatch_diff_pct");
    let avg_diff_str = avg_diff.map_or_else(|| "N/A".to_string(), |d| format!("{d:.1}%"));

    let pct = g_fmt(OSL_MISMATCH_PCT_THRESHOLD);
    let max_tokens = OSL_MISMATCH_MAX_TOKEN_THRESHOLD;
    let count = comma_int(mismatch_count);
    let total = comma_int(total);
    // Built line-by-line: a `\`-continuation in a Rust string literal eats the
    // next line's leading whitespace, which would drop the `  - ` indentation.
    let body = [
        format!(
            "{count} of {total} requests ({percentage:.1}%) have output length differing from requested by more than the threshold."
        ),
        format!("Threshold (tokens): min(requested x {pct}%, {max_tokens})"),
        format!("Average mismatch: {avg_diff_str}"),
        String::new(),
        "Why: Server hit EOS token before reaching requested output length.".to_string(),
        String::new(),
        "Fix Options:".to_string(),
        "  - --extra-inputs ignore_eos:true - Generate until max_tokens (vLLM, TensorRT-LLM)"
            .to_string(),
        "  - --extra-inputs min_tokens:<N> - Set minimum output length (vLLM, TensorRT-LLM, SGLang)"
            .to_string(),
        "  - --use-server-token-count - Use server-reported token counts if tokenizer mismatch suspected"
            .to_string(),
        String::new(),
        "Diagnostics:".to_string(),
        "  - Review profile_export.jsonl -> osl_mismatch_diff_pct for per-request values"
            .to_string(),
        format!("  - Adjust: AIPERF_METRICS_OSL_MISMATCH_PCT_THRESHOLD={pct}"),
        format!("  - Adjust: AIPERF_METRICS_OSL_MISMATCH_MAX_TOKEN_THRESHOLD={max_tokens}"),
    ]
    .join("\n");
    Some(Warning {
        title: "Output Sequence Length Mismatch Warning".to_string(),
        body,
    })
}

/// Emit the usage-discrepancy warning when both discrepancy and request counts are positive.
pub(crate) fn detect_usage_discrepancy(report: &NativeReport) -> Option<Warning> {
    let discrepancy_avg = metric_avg(report, "usage_discrepancy_count")?;
    if discrepancy_avg <= 0.0 {
        return None;
    }
    let discrepancy_count = discrepancy_avg.trunc() as i64;
    let total = metric_avg(report, "request_count")
        .map(|v| v.trunc() as i64)
        .filter(|&t| t != 0)?;
    let percentage = (discrepancy_count as f64 / total as f64) * 100.0;
    let threshold = g_fmt(USAGE_PCT_DIFF_THRESHOLD);
    let count = comma_int(discrepancy_count);
    let total = comma_int(total);

    let body = [
        format!(
            "{count} of {total} requests ({percentage:.1}%) show a difference exceeding {threshold}% between:"
        ),
        "  - API-reported usage tokens (from 'usage' field)".to_string(),
        "  - Client-computed token counts (from tokenization)".to_string(),
        String::new(),
        "Possible Causes:".to_string(),
        "  - Different tokenization methods (API vs client)".to_string(),
        "  - API special tokens or preprocessing".to_string(),
        String::new(),
        "Investigation Steps:".to_string(),
        "  1. Review profile_export.jsonl for per-request usage_*_diff_pct values".to_string(),
        "  2. Verify client tokenizer matches the model's tokenizer".to_string(),
        "  3. Use server token counts: --use-server-token-count (disables client tokenization and diff metrics)"
            .to_string(),
        format!("  4. Adjust threshold: AIPERF_METRICS_USAGE_PCT_DIFF_THRESHOLD={threshold}"),
    ]
    .join("\n");
    Some(Warning {
        title: "Token Count Discrepancy Warning".to_string(),
        body,
    })
}

/// Detect unsupported completion-token parameters before session-control errors.
pub(crate) fn detect_api_errors(report: &NativeReport) -> Vec<Warning> {
    let mut warnings = Vec::new();
    if let Some(warning) = detect_max_completion_tokens(report) {
        warnings.push(warning);
    }
    if let Some(warning) = detect_dynamo_session_control(report) {
        warnings.push(warning);
    }
    warnings
}

/// Match a JSON object's string `message` field when present, otherwise the raw
/// error text.
fn error_blob(raw: &str) -> String {
    if let Ok(serde_json::Value::Object(map)) = serde_json::from_str::<serde_json::Value>(raw)
        && let Some(serde_json::Value::String(message)) = map.get("message")
    {
        return message.clone();
    }
    raw.to_string()
}

/// Render markup-free insight text. Every investigation line is intentionally
/// prefixed `1.` as part of the artifact contract.
fn format_insight(
    problem: &str,
    causes: &[&str],
    investigation: &[&str],
    fixes: &[&str],
) -> String {
    let mut out = String::new();
    out.push_str(problem);
    out.push_str("\n\nPossible Causes:\n  \u{2022} ");
    out.push_str(&causes.join("\n  \u{2022} "));
    out.push_str("\n\nInvestigation Steps:\n  1. ");
    out.push_str(&investigation.join("\n  1. "));
    out.push_str("\n\nSuggested Fixes:\n  \u{2022} ");
    out.push_str(&fixes.join("\n  \u{2022} "));
    out
}

/// MaxCompletionTokens insight. Trigger: a blob containing all of
/// `extra_forbidden`, `max_completion_tokens`, and `Extra inputs are not
/// permitted` (case-sensitive).
pub(crate) fn detect_max_completion_tokens(report: &NativeReport) -> Option<Warning> {
    for error in &report.errors {
        let blob = error_blob(&error.message);
        if blob.contains("extra_forbidden")
            && blob.contains("max_completion_tokens")
            && blob.contains("Extra inputs are not permitted")
        {
            let body = format_insight(
                "The backend rejected 'max_completion_tokens'. This backend only supports 'max_tokens'.",
                &[
                    "AIPerf generated 'max_completion_tokens' due to --output-tokens-mean.",
                    "The backend rejects 'max_completion_tokens' because it only supports 'max_tokens'.",
                ],
                &[
                    "Inspect request payloads in profile_export.jsonl.",
                    "Check the backend's supported parameters.",
                ],
                &[
                    "Remove --output-tokens-mean.",
                    "Or use --extra-inputs \"max_tokens:<value>\".",
                    "Or run AIPerf with '--use-legacy-max-tokens' to force use of the legacy 'max_tokens' field instead of 'max_completion_tokens'.",
                ],
            );
            return Some(Warning {
                title: "Unsupported Parameter: max_completion_tokens".to_string(),
                body,
            });
        }
    }
    None
}

/// DynamoSessionControl insight. Trigger: a lowercased blob containing both
/// `unknown variant` and `bind`.
pub(crate) fn detect_dynamo_session_control(report: &NativeReport) -> Option<Warning> {
    for error in &report.errors {
        let blob = error_blob(&error.message).to_lowercase();
        if blob.contains("unknown variant") && blob.contains("bind") {
            let body = format_insight(
                "The Dynamo frontend rejected nvext.session_control with action='bind'. This Dynamo build's SessionAction only accepts 'open' and 'close' -- the 'bind' action was added after the v1.2.x release line (first available in v1.3.0-dev / upstream commit d97c889ba).",
                &[
                    "--use-dynamo-conv-aware-routing emits action='bind' on every non-final turn.",
                    "The target Dynamo server predates the 'bind' action (e.g. v1.2.1).",
                ],
                &[
                    "Check the Dynamo frontend version and its supported SessionAction values.",
                    "Inspect request payloads in profile_export.jsonl -> nvext.session_control.",
                ],
                &[
                    "Upgrade Dynamo to a build that supports action='bind' (>= v1.3.0-dev, upstream commit d97c889ba).",
                    "Or run with --use-legacy-dynamo-session-control to emit the v1.2.x-compatible open/close lifecycle (requires the worker to expose a session_control endpoint).",
                    "Or disable --use-dynamo-conv-aware-routing.",
                ],
            );
            return Some(Warning {
                title: "Unsupported Dynamo session_control action: bind".to_string(),
                body,
            });
        }
    }
    None
}

/// Render `N/A` for missing error codes or types and comma-group counts.
pub(crate) fn error_summary_table(report: &NativeReport, width: usize) -> Option<String> {
    if report.errors.is_empty() {
        return None;
    }
    let rows: Vec<Vec<String>> = report
        .errors
        .iter()
        .map(|error| {
            let code = match error.code {
                Some(code) if code != 0 => code.to_string(),
                _ => "N/A".to_string(),
            };
            let error_type = if error.error_type.is_empty() {
                "N/A".to_string()
            } else {
                error.error_type.clone()
            };
            vec![
                code,
                error_type,
                error.message.clone(),
                comma_int(error.count as i64),
            ]
        })
        .collect();

    let justify = [
        Justify::Right,
        Justify::Right,
        Justify::Left,
        Justify::Right,
    ];
    Some(render_table(
        "NVIDIA AIPerf | Error Summary",
        &["Code", "Type", "Message", "Count"],
        &rows,
        &justify,
        width,
    ))
}

/// Console-group render order.
/// `MetricConsoleGroup::None` is intentionally absent — those rows are hidden.
const GROUP_ORDER: &[MetricConsoleGroup] = &[
    MetricConsoleGroup::Effective,
    MetricConsoleGroup::Active,
    MetricConsoleGroup::Usage,
    MetricConsoleGroup::Cache,
    MetricConsoleGroup::Prediction,
    MetricConsoleGroup::Audio,
    MetricConsoleGroup::Reasoning,
    MetricConsoleGroup::Default,
];

/// Stat column order.
const STAT_KEYS: &[&str] = &["avg", "min", "max", "p99", "p90", "p50", "std"];

const SGLANG_SPEC_ACCEPT_RATE: &str = "sglang:spec_accept_rate";
const SGLANG_SPEC_ACCEPT_LENGTH: &str = "sglang:spec_accept_length";
const SGLANG_MODEL_LABEL: &str = "model_name";
const SGLANG_PP_RANK_LABEL: &str = "pp_rank";
const SGLANG_TP_RANK_LABEL: &str = "tp_rank";

/// Display policy for one SGLang speculative-decoding metric family.
struct SpeculativeMetricDisplay {
    source: &'static str,
    row_name: &'static str,
    scale: f64,
    precision: usize,
}

const SPECULATIVE_DECODING_METRICS: &[SpeculativeMetricDisplay] = &[
    SpeculativeMetricDisplay {
        source: SGLANG_SPEC_ACCEPT_RATE,
        row_name: "Accept Rate (%)",
        scale: 100.0,
        precision: 1,
    },
    SpeculativeMetricDisplay {
        source: SGLANG_SPEC_ACCEPT_LENGTH,
        row_name: "Accept Length",
        scale: 1.0,
        precision: 2,
    },
];

/// One model- and rank-selected server metric series.
struct SelectedSpeculativeSeries<'a> {
    series: &'a MetricSeries,
    endpoint: String,
}

/// One rendered speculative-decoding row.
struct SpeculativeDecodingRow {
    cells: Vec<String>,
}

/// A metric row prepared for the table: the projected `display_order` (for the
/// stable sort) plus the rendered cells.
struct MetricRow {
    display_order: u32,
    cells: Vec<String>,
}

/// Render one table per non-empty group in [`GROUP_ORDER`]. Registered tags use
/// configured metadata and filtering; absent tags use their raw names in the
/// default group, sort last, and carry no filtering flags.
pub(crate) fn metrics_tables(
    report: &NativeReport,
    cfg: &ConsoleTxtExportConfig,
    width: usize,
) -> Option<String> {
    let mut grouped: Vec<(MetricConsoleGroup, Vec<MetricRow>)> =
        GROUP_ORDER.iter().map(|g| (*g, Vec::new())).collect();

    for (tag, entry) in &report.metrics {
        // Multi-series metrics require one unlabeled aggregate; labeled sidecar
        // series do not enter the primary table.
        let Some(series) = summary_series(entry) else {
            continue;
        };

        let meta = cfg.metrics.get(tag);
        // Metadata flags apply only to registered metrics.
        if let Some(meta) = meta
            && (meta.internal || meta.experimental || meta.error_only)
        {
            continue;
        }

        // Unregistered metrics use the default group.
        let group = match meta {
            Some(meta) => console_group_from_str(&meta.group),
            None => MetricConsoleGroup::Default,
        };
        let Some(slot) = grouped
            .iter_mut()
            .find(|(candidate, _)| *candidate == group)
        else {
            continue; // group `none` (hidden) or otherwise not rendered.
        };

        // Unregistered metrics use the raw snake-case tag as their header.
        let header: &str = meta.map_or(tag.as_str(), |meta| meta.header.as_str());
        let display_order = meta.and_then(|meta| meta.display_order).unwrap_or(u32::MAX);

        let mut cells = Vec::with_capacity(1 + STAT_KEYS.len());
        // Long headers place the unit on a second physical line.
        let delimiter = if header.chars().count() > 30 {
            "\n"
        } else {
            " "
        };
        cells.push(format!("{header}{delimiter}({})", entry.unit));
        for key in STAT_KEYS {
            cells.push(stat_cell(series, key));
        }
        slot.1.push(MetricRow {
            display_order,
            cells,
        });
    }

    let mut blocks = Vec::new();
    for (group, rows) in &mut grouped {
        if rows.is_empty() {
            continue;
        }
        rows.sort_by_key(|row| row.display_order);
        let title = group_title(*group, &cfg.title);
        let mut header = vec!["Metric".to_string()];
        header.extend(STAT_KEYS.iter().map(|k| (*k).to_string()));
        let header_refs: Vec<&str> = header.iter().map(String::as_str).collect();
        let justify: Vec<Justify> = std::iter::repeat_n(Justify::Right, header.len()).collect();
        let table_rows: Vec<Vec<String>> = rows.iter().map(|row| row.cells.clone()).collect();
        blocks.push(render_table(
            &title,
            &header_refs,
            &table_rows,
            &justify,
            width,
        ));
    }

    if blocks.is_empty() {
        None
    } else {
        Some(blocks.join("\n"))
    }
}

/// Render active, model-matched SGLang speculative-decoding gauge summaries.
fn speculative_decoding_table(
    report: &NativeReport,
    cfg: &ConsoleTxtExportConfig,
    width: usize,
) -> Option<String> {
    if cfg.model_names.is_empty() {
        return None;
    }
    let [rate_metric, length_metric] = SPECULATIVE_DECODING_METRICS else {
        return None;
    };

    let rate_series = select_speculative_series(report, &cfg.model_names, rate_metric.source);
    let length_series = select_speculative_series(report, &cfg.model_names, length_metric.source);
    let activity_series = if length_series.is_empty() {
        &rate_series
    } else {
        &length_series
    };
    if !activity_series.iter().any(|selected| {
        let ReportStats::Distribution(stats) = &selected.series.stats else {
            return false;
        };
        stats
            .max
            .as_ref()
            .and_then(value_f64)
            .is_some_and(|value| value.is_finite() && value > 0.0)
    }) {
        return None;
    }

    let mut rows = Vec::new();
    for (metric, series) in [rate_metric, length_metric]
        .into_iter()
        .zip([&rate_series, &length_series])
    {
        for (index, selected) in series.iter().enumerate() {
            let ReportStats::Distribution(stats) = &selected.series.stats else {
                continue;
            };
            let values = [
                scaled_report_value(stats.avg.as_ref(), metric.scale),
                scaled_report_value(stats.min.as_ref(), metric.scale),
                scaled_report_value(stats.max.as_ref(), metric.scale),
                scaled_report_value(stats.percentiles.get("p50"), metric.scale),
                scaled_report_value(stats.percentiles.get("p90"), metric.scale),
            ];
            let [Some(avg), Some(min), Some(max), Some(p50), Some(p90)] = values else {
                continue;
            };
            let suffix = speculative_series_suffix(series, index);
            let row_name = if suffix.is_empty() {
                metric.row_name.to_string()
            } else {
                format!("{} ({suffix})", metric.row_name)
            };
            rows.push(SpeculativeDecodingRow {
                cells: vec![
                    row_name,
                    comma_fixed(avg, metric.precision),
                    comma_fixed(min, metric.precision),
                    comma_fixed(max, metric.precision),
                    comma_fixed(p50, metric.precision),
                    comma_fixed(p90, metric.precision),
                ],
            });
        }
    }
    if rows.is_empty() {
        return None;
    }

    let table_rows: Vec<Vec<String>> = rows.into_iter().map(|row| row.cells).collect();
    Some(render_table(
        "NVIDIA AIPerf | Server Metrics: Speculative Decoding",
        &["Metric", "mean", "min", "max", "p50", "p90"],
        &table_rows,
        &[
            Justify::Left,
            Justify::Right,
            Justify::Right,
            Justify::Right,
            Justify::Right,
            Justify::Right,
        ],
        width,
    ))
}

/// Select SGLang leader series for one server metric family.
fn select_speculative_series<'a>(
    report: &'a NativeReport,
    model_names: &[String],
    source: &str,
) -> Vec<SelectedSpeculativeSeries<'a>> {
    let Some(entry) = report.server_metrics.get(source) else {
        return Vec::new();
    };

    let mut selected = Vec::new();
    for series in &entry.series {
        let ReportStats::Distribution(_) = &series.stats else {
            continue;
        };
        let Some(labels) = series.labels.as_ref() else {
            continue;
        };
        let Some(model_name) = labels.get(SGLANG_MODEL_LABEL) else {
            continue;
        };
        if !model_names
            .iter()
            .any(|configured| configured.to_lowercase() == model_name.to_lowercase())
        {
            continue;
        }
        if labels
            .get(SGLANG_PP_RANK_LABEL)
            .is_some_and(|rank| rank != "0")
            || labels
                .get(SGLANG_TP_RANK_LABEL)
                .is_some_and(|rank| rank != "0")
        {
            continue;
        }
        selected.push(SelectedSpeculativeSeries {
            series,
            endpoint: normalize_endpoint_display(series.endpoint_url.as_deref().unwrap_or("")),
        });
    }
    selected
}

/// Return a display-only suffix that distinguishes one selected series.
fn speculative_series_suffix(series: &[SelectedSpeculativeSeries<'_>], index: usize) -> String {
    if series.len() <= 1 {
        return String::new();
    }
    let current = &series[index];
    let Some(labels) = current.series.labels.as_ref() else {
        return format!("series={}", index + 1);
    };
    let mut parts = Vec::new();
    if series
        .iter()
        .any(|other| other.endpoint != current.endpoint)
    {
        parts.push(format!("endpoint={}", current.endpoint));
    }
    if let Some(model_name) = labels.get(SGLANG_MODEL_LABEL)
        && label_value_differs(series, index, SGLANG_MODEL_LABEL, model_name)
    {
        parts.push(format!("{SGLANG_MODEL_LABEL}={model_name}"));
    }
    for (label, value) in labels {
        if matches!(
            label.as_str(),
            SGLANG_MODEL_LABEL | SGLANG_PP_RANK_LABEL | SGLANG_TP_RANK_LABEL
        ) || !label_value_differs(series, index, label, value)
        {
            continue;
        }
        parts.push(format!("{label}={value}"));
    }
    if parts.is_empty() {
        parts.push(format!("series={}", index + 1));
    }
    parts.join(", ")
}

/// Whether another selected series has a different value for one label.
fn label_value_differs(
    series: &[SelectedSpeculativeSeries<'_>],
    index: usize,
    label: &str,
    value: &str,
) -> bool {
    series.iter().enumerate().any(|(other_index, other)| {
        other_index != index
            && other
                .series
                .labels
                .as_ref()
                .and_then(|labels| labels.get(label))
                .is_none_or(|other_value| other_value != value)
    })
}

/// Extract a finite report value and apply presentation-only scaling.
fn scaled_report_value(value: Option<&ReportValue>, scale: f64) -> Option<f64> {
    let value = value.and_then(value_f64)?;
    let scaled = value * scale;
    scaled.is_finite().then_some(scaled)
}

/// Map a configured console-group value onto the enum. Unknown values are hidden.
fn console_group_from_str(group: &str) -> MetricConsoleGroup {
    match group {
        "default" => MetricConsoleGroup::Default,
        "usage" => MetricConsoleGroup::Usage,
        "cache" => MetricConsoleGroup::Cache,
        "prediction" => MetricConsoleGroup::Prediction,
        "audio" => MetricConsoleGroup::Audio,
        "reasoning" => MetricConsoleGroup::Reasoning,
        "effective" => MetricConsoleGroup::Effective,
        "active" => MetricConsoleGroup::Active,
        _ => MetricConsoleGroup::None,
    }
}

/// Use the base title for the default group and append the group name otherwise.
fn group_title(group: MetricConsoleGroup, base: &str) -> String {
    match group {
        MetricConsoleGroup::Default | MetricConsoleGroup::None => base.to_string(),
        MetricConsoleGroup::Usage => format!("{base}: Usage"),
        MetricConsoleGroup::Cache => format!("{base}: Cache"),
        MetricConsoleGroup::Prediction => format!("{base}: Prediction"),
        MetricConsoleGroup::Audio => format!("{base}: Audio"),
        MetricConsoleGroup::Reasoning => format!("{base}: Reasoning"),
        MetricConsoleGroup::SpecDecode => format!("{base}: Speculative Decoding"),
        MetricConsoleGroup::Effective => format!("{base}: Effective"),
        MetricConsoleGroup::Active => format!("{base}: Active"),
    }
}

/// Select the sole series or unique unlabeled aggregate. Missing or ambiguous
/// aggregates are omitted.
fn summary_series(entry: &MetricEntry) -> Option<&MetricSeries> {
    match crate::export::summary_series(&entry.series) {
        crate::export::SummarySeries::Selected(series) => Some(series),
        _ => None,
    }
}

/// Format present stats with grouped thousands and two decimals; use `N/A` when absent.
///
/// A scalar maps `value` to `avg`/`min`/`max`; a counter maps `total` to those
/// fields; a histogram exposes `avg` and percentiles; and a distribution carries
/// the full stat set. Missing fields render as `N/A`.
fn stat_cell(series: &MetricSeries, key: &str) -> String {
    let value = match &series.stats {
        ReportStats::Distribution(dist) => match key {
            "avg" => dist.avg.as_ref().and_then(value_f64),
            "min" => dist.min.as_ref().and_then(value_f64),
            "max" => dist.max.as_ref().and_then(value_f64),
            "std" => dist.std.as_ref().and_then(value_f64),
            other => dist.percentiles.get(other).and_then(value_f64),
        },
        ReportStats::Scalar(scalar) => match key {
            "avg" | "min" | "max" => value_f64(&scalar.value),
            _ => None,
        },
        ReportStats::Counter(counter) => match key {
            "avg" | "min" | "max" => value_f64(&counter.total),
            _ => None,
        },
        ReportStats::Histogram(hist) => match key {
            "avg" => hist.avg.as_ref().and_then(value_f64),
            other => hist.percentiles.get(other).and_then(value_f64),
        },
    };
    value.map_or_else(|| "N/A".to_string(), comma_2dp)
}

/// Column text justification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Justify {
    Left,
    Right,
}

/// Terminal-cell width of one character. A `-1` non-printable range has width
/// zero; codepoints outside the table have width one.
fn char_cell_size(character: char) -> usize {
    let codepoint = character as u32;
    let table = &cell_widths::CELL_WIDTHS;
    let mut lo: isize = 0;
    let mut hi: isize = table.len() as isize - 1;
    while lo <= hi {
        let mid = (lo + hi) / 2;
        let (start, end, width) = table[mid as usize];
        if codepoint < start {
            hi = mid - 1;
        } else if codepoint as i32 > end {
            lo = mid + 1;
        } else {
            return if width == -1 { 0 } else { width as usize };
        }
    }
    1
}

/// Display width in terminal columns. Wide glyphs count as two cells and
/// zero-width or combining marks count as zero.
fn cell_width(text: &str) -> usize {
    text.chars().map(char_cell_size).sum()
}

/// Fit `text` to exactly `total` cells. Short text is space-padded; a wide glyph
/// crossing the crop boundary is replaced by one space.
fn set_cell_size(text: &str, total: usize) -> String {
    if cell_width(text) <= total {
        let mut out = text.to_string();
        for _ in 0..total - cell_width(text) {
            out.push(' ');
        }
        return out;
    }
    if total == 0 {
        return String::new();
    }
    let mut out = String::new();
    let mut acc = 0usize;
    for character in text.chars() {
        let size = char_cell_size(character);
        if acc + size > total {
            // Replace a width-two glyph crossing the boundary with one space.
            if size == 2 && acc + size == total + 1 {
                out.push(' ');
            }
            break;
        }
        out.push(character);
        acc += size;
        if acc == total {
            break;
        }
    }
    out
}

/// Pad `text` to `width` columns per `justify`.
fn pad(text: &str, width: usize, justify: Justify) -> String {
    let fill = width.saturating_sub(cell_width(text));
    match justify {
        Justify::Left => format!("{text}{}", " ".repeat(fill)),
        Justify::Right => format!("{}{text}", " ".repeat(fill)),
    }
}

/// Center `text` within `width` columns (used for table titles).
fn center(text: &str, width: usize) -> String {
    let text_width = cell_width(text);
    if text_width >= width {
        return text.to_string();
    }
    let remaining = width - text_width;
    let left = remaining / 2;
    format!("{}{text}{}", " ".repeat(left), " ".repeat(remaining - left))
}

/// Render a heavy-header table with a centered title.
///
/// Flexible columns use their widest line plus one padding cell per side.
/// Oversized tables repeatedly reduce the widest columns, wrap cell text, and
/// ellipsize unbreakable overflow. Explicit newlines increase row height; shorter
/// cells are top-aligned and padded with blank lines.
pub(crate) fn render_table(
    title: &str,
    headers: &[&str],
    rows: &[Vec<String>],
    justify: &[Justify],
    export_width: usize,
) -> String {
    let columns = headers.len();
    let widths = solve_column_widths(headers, rows, export_width);

    let border = |left: char, mid: char, right: char, glyph: char| -> String {
        let parts: Vec<String> = widths
            .iter()
            .map(|w| glyph.to_string().repeat(w + 2))
            .collect();
        format!("{left}{}{right}", parts.join(&mid.to_string()))
    };
    let render_row = |cells: &[String], vbar: char| -> String {
        // Wrap each cell to its resolved width with ellipsis overflow. Row height
        // follows the tallest cell; missing or short cells emit padded blanks.
        let split: Vec<Vec<String>> = (0..columns)
            .map(|index| {
                let just = justify.get(index).copied().unwrap_or(Justify::Left);
                cells.get(index).map_or_else(
                    || vec![String::new()],
                    |cell| wrap_cell(cell, widths[index], just),
                )
            })
            .collect();
        let height = split.iter().map(Vec::len).max().unwrap_or(1);
        let mut out_lines: Vec<String> = Vec::with_capacity(height);
        for line_index in 0..height {
            let mut line = String::new();
            line.push(vbar);
            for (index, &width) in widths.iter().enumerate().take(columns) {
                let text = split[index].get(line_index).map_or("", String::as_str);
                let just = justify.get(index).copied().unwrap_or(Justify::Left);
                line.push(' ');
                line.push_str(&pad(text, width, just));
                line.push(' ');
                line.push(vbar);
            }
            out_lines.push(line);
        }
        out_lines.join("\n")
    };

    let table_width: usize = widths.iter().map(|w| w + 2).sum::<usize>() + columns + 1;
    let mut out = String::new();
    out.push_str(&center(title, table_width));
    out.push('\n');
    out.push_str(&border('\u{250F}', '\u{2533}', '\u{2513}', '\u{2501}')); // ┏━┳━┓
    out.push('\n');
    let header_cells: Vec<String> = headers.iter().map(|h| (*h).to_string()).collect();
    out.push_str(&render_row(&header_cells, '\u{2503}')); // ┃
    out.push('\n');
    out.push_str(&border('\u{2521}', '\u{2547}', '\u{2529}', '\u{2501}')); // ┡╇┩
    out.push('\n');
    for row in rows {
        out.push_str(&render_row(row, '\u{2502}')); // │
        out.push('\n');
    }
    out.push_str(&border('\u{2514}', '\u{2534}', '\u{2518}', '\u{2500}')); // └┴┘
    out
}

/// Resolve each column's content width, excluding one-cell side padding:
///   1. each flexible column measures to its widest cell *line* (`max`) plus the
///      two padding cells;
///   2. if the summed column widths exceed the space left after borders
///      (`export_width - extra_width`), the widest wrapable columns shrink toward
///      the next-widest, then any residual excess is distributed evenly;
///   3. the padding is removed to yield the content width used for rendering.
fn solve_column_widths(headers: &[&str], rows: &[Vec<String>], export_width: usize) -> Vec<usize> {
    let columns = headers.len();
    // Two edge borders plus one divider between each pair of columns.
    let extra_width = columns + 1;
    let available = export_width.saturating_sub(extra_width);

    // Measure the widest line in each column, then add two padding cells.
    let mut widths: Vec<usize> = headers
        .iter()
        .map(|header| cell_width(header) + 2)
        .collect();
    for row in rows {
        for (index, cell) in row.iter().enumerate().take(columns) {
            let cell_max = cell.split('\n').map(cell_width).max().unwrap_or(0) + 2;
            widths[index] = widths[index].max(cell_max);
        }
    }

    if widths.iter().sum::<usize>() > available {
        // Every column is flexible (no fixed width, wrapping enabled).
        let wrapable = vec![true; columns];
        widths = collapse_widths(&widths, &wrapable, available);
        let total: usize = widths.iter().sum();
        if total > available {
            let excess = total - available;
            let ratios = vec![1usize; columns];
            let maximums = widths.clone();
            widths = ratio_reduce(excess, &ratios, &maximums, &widths);
        }
    }

    widths.iter().map(|w| w.saturating_sub(2)).collect()
}

/// Reduce the widest wrapable columns toward the next-widest until the total
/// fits `max_width`.
fn collapse_widths(widths: &[usize], wrapable: &[bool], max_width: usize) -> Vec<usize> {
    let mut widths: Vec<usize> = widths.to_vec();
    let mut total_width: usize = widths.iter().sum();
    if !wrapable.iter().any(|&w| w) {
        return widths;
    }
    while total_width > 0 && total_width > max_width {
        let max_column = (0..widths.len())
            .filter(|&i| wrapable[i])
            .map(|i| widths[i])
            .max()
            .unwrap_or(0);
        let second_max_column = (0..widths.len())
            .map(|i| {
                if wrapable[i] && widths[i] != max_column {
                    widths[i]
                } else {
                    0
                }
            })
            .max()
            .unwrap_or(0);
        let column_difference = max_column - second_max_column;
        let ratios: Vec<usize> = (0..widths.len())
            .map(|i| usize::from(widths[i] == max_column && wrapable[i]))
            .collect();
        let excess_width = total_width - max_width;
        if !ratios.iter().any(|&r| r != 0) || column_difference == 0 {
            break;
        }
        let max_reduce = vec![excess_width.min(column_difference); widths.len()];
        widths = ratio_reduce(excess_width, &ratios, &max_reduce, &widths);
        total_width = widths.iter().sum();
    }
    widths
}

/// Distribute a reduction across values in ratio proportion, capped per slot.
fn ratio_reduce(
    total: usize,
    ratios: &[usize],
    maximums: &[usize],
    values: &[usize],
) -> Vec<usize> {
    let mut ratios: Vec<usize> = ratios
        .iter()
        .zip(maximums)
        .map(|(&ratio, &max)| if max != 0 { ratio } else { 0 })
        .collect();
    let mut total_ratio: usize = ratios.iter().sum();
    if total_ratio == 0 {
        return values.to_vec();
    }
    let mut total_remaining = total as i64;
    let mut result = Vec::with_capacity(values.len());
    for ((ratio, &maximum), &value) in ratios.iter_mut().zip(maximums).zip(values) {
        if *ratio != 0 && total_ratio > 0 {
            // Round the proportional share to nearest, with ties to even.
            let distributed = (maximum as i64).min(round_half_up(
                *ratio as f64 * total_remaining as f64 / total_ratio as f64,
            ));
            result.push((value as i64 - distributed).max(0) as usize);
            total_remaining -= distributed;
            total_ratio -= *ratio;
        } else {
            result.push(value);
        }
    }
    result
}

/// Round a non-negative quantity to nearest, with ties to even.
fn round_half_up(value: f64) -> i64 {
    let floor = value.floor();
    let diff = value - floor;
    if diff > 0.5 {
        floor as i64 + 1
    } else if diff < 0.5 {
        floor as i64
    } else {
        // Exact ties round to even.
        let f = floor as i64;
        if f % 2 == 0 { f } else { f + 1 }
    }
}

/// Remove at most `char_len - size` trailing whitespace characters. A line whose
/// character length fits retains trailing whitespace, which may still trigger
/// ellipsis when its terminal-cell width overflows.
fn rstrip_end(chars: &[char], size: usize) -> Vec<char> {
    let text_length = chars.len();
    if text_length <= size {
        return chars.to_vec();
    }
    let excess = text_length - size;
    let mut whitespace = 0;
    while whitespace < text_length && chars[text_length - 1 - whitespace].is_whitespace() {
        whitespace += 1;
    }
    let crop = whitespace.min(excess);
    chars[..text_length - crop].to_vec()
}

/// Wrap a table cell to `width` content cells. Explicit newlines split logical
/// lines; over-long words are not folded, and overflowing physical lines end
/// with an ellipsis.
///
/// Left-justified lines retain bounded trailing whitespace; right-justified
/// lines remove it before render-time padding.
fn wrap_cell(cell: &str, width: usize, justify: Justify) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for logical in cell.split('\n') {
        let chars: Vec<char> = logical.chars().collect();
        let breaks = divide_line(&chars, width, false);
        let mut prev = 0;
        let push_piece = |piece: &[char], out: &mut Vec<String>| {
            let stripped = rstrip_end(piece, width);
            // Right-justified lines drop trailing whitespace; left-justified
            // lines retain the bounded suffix.
            let text: String = match justify {
                Justify::Left => stripped.iter().collect(),
                Justify::Right => stripped.iter().collect::<String>().trim_end().to_string(),
            };
            out.push(truncate_ellipsis(&text, width));
        };
        for &offset in &breaks {
            push_piece(&chars[prev..offset], &mut out);
            prev = offset;
        }
        push_piece(&chars[prev..], &mut out);
    }
    out
}

/// Crop overflow to `width - 1` cells and append an ellipsis. A wide glyph
/// crossing the boundary is dropped and space-padded.
fn truncate_ellipsis(text: &str, width: usize) -> String {
    if cell_width(text) <= width {
        return text.to_string();
    }
    let head = set_cell_size(text, width.saturating_sub(1));
    format!("{head}\u{2026}") // …
}

/// Cell width of a slice of characters (sum of each glyph's cell size).
fn cells_len(chars: &[char]) -> usize {
    chars.iter().copied().map(char_cell_size).sum()
}

/// Split a long word into cell-width-bounded chunks. A glyph starts a new chunk
/// when it does not fit; wide glyphs advance by two cells.
fn chop_cells(word: &[char], width: usize) -> Vec<Vec<char>> {
    let mut lines: Vec<Vec<char>> = vec![Vec::new()];
    let mut total = 0usize;
    for &character in word {
        let size = char_cell_size(character);
        if total + size > width {
            lines.push(vec![character]);
            total = size;
        } else {
            lines.last_mut().expect("at least one line").push(character);
            total += size;
        }
    }
    lines
}

/// Tokenize into `\s*\S+\s*` spans as `(start, end)` character offsets. Leading
/// whitespace joins the following word; trailing-only whitespace yields no span.
fn word_spans(text: &[char]) -> Vec<(usize, usize)> {
    let mut spans = Vec::new();
    let len = text.len();
    let mut pos = 0;
    while pos < len {
        let start = pos;
        while pos < len && text[pos].is_whitespace() {
            pos += 1;
        }
        if pos >= len {
            break; // only whitespace remained: `\S+` cannot match.
        }
        while pos < len && !text[pos].is_whitespace() {
            pos += 1;
        }
        while pos < len && text[pos].is_whitespace() {
            pos += 1;
        }
        spans.push((start, pos));
    }
    spans
}

/// Compute character offsets that split text into lines of at most `width`
/// cells. With `fold=true`, over-long words are chopped; otherwise they remain
/// intact on their own line for later cropping.
fn divide_line(text: &[char], width: usize, fold: bool) -> Vec<usize> {
    let mut breaks: Vec<usize> = Vec::new();
    let mut cell_offset: isize = 0;
    let width_i = width as isize;
    for (start, end) in word_spans(text) {
        let word = &text[start..end];
        let mut stripped = word.len();
        while stripped > 0 && word[stripped - 1].is_whitespace() {
            stripped -= 1;
        }
        // Widths use terminal cells; break offsets use character indices.
        let word_length = cells_len(&word[..stripped]) as isize;
        let remaining_space = width_i - cell_offset;
        if remaining_space >= word_length {
            cell_offset += cells_len(word) as isize;
        } else if fold && word_length > width_i {
            let folded = chop_cells(word, width.max(1));
            let last_index = folded.len().saturating_sub(1);
            let mut folded_start = start;
            for (index, line) in folded.iter().enumerate() {
                if folded_start != 0 {
                    breaks.push(folded_start);
                }
                if index == last_index {
                    cell_offset = cells_len(line) as isize;
                } else {
                    folded_start += line.len();
                }
            }
        } else if word_length > width_i {
            // Keep an unfolded over-long word on its own line and advance by its
            // full width so the following word breaks.
            if start != 0 {
                breaks.push(start);
            }
            cell_offset = cells_len(word) as isize;
        } else if cell_offset != 0 && start != 0 {
            breaks.push(start);
            cell_offset = cells_len(word) as isize;
        }
    }
    breaks
}

/// Wrap one logical line to `width` cells and remove trailing whitespace from
/// each physical line.
fn wrap_line(line: &str, width: usize) -> Vec<String> {
    let chars: Vec<char> = line.chars().collect();
    let breaks = divide_line(&chars, width, true);
    let mut pieces: Vec<String> = Vec::with_capacity(breaks.len() + 1);
    let mut prev = 0;
    for &offset in &breaks {
        pieces.push(
            chars[prev..offset]
                .iter()
                .collect::<String>()
                .trim_end()
                .to_string(),
        );
        prev = offset;
    }
    pieces.push(
        chars[prev..]
            .iter()
            .collect::<String>()
            .trim_end()
            .to_string(),
    );
    pieces
}

/// Render a non-expanding rounded panel with two-cell horizontal padding and a
/// title centered in the top border.
///
/// The panel fits its content up to the export width. Its space-padded title may
/// widen it, body overflow is folded, and every box glyph occupies one cell.
pub(crate) fn warning_panel(warning: &Warning, width: usize) -> String {
    const PAD: usize = 2; // padding=(0, 2): two cells left and right.
    let max_width = width.max(2 * PAD + 3);
    // Fit the longest body line plus padding, widen for the title, and clamp to
    // the available inner width.
    let available_text = max_width - 2 - 2 * PAD;
    let body_lines: Vec<&str> = warning.body.split('\n').collect();
    let longest = body_lines.iter().map(|l| cell_width(l)).max().unwrap_or(0);
    let content_child = longest.min(available_text) + 2 * PAD;
    // Include one title-space per side and two border corners.
    let title_min = warning.title.chars().count() + 2 + 2;
    let child_width = (max_width - 2).min(content_child.max(title_min));
    let inner_text = child_width - 2 * PAD;

    // Fold the body to the inner text width.
    let mut wrapped: Vec<String> = Vec::new();
    for line in &body_lines {
        wrapped.extend(wrap_line(line, inner_text));
    }

    let mut out = String::new();
    let title_slug = format!(" {} ", warning.title);
    let remaining = child_width - cell_width(&title_slug);
    let left = remaining / 2;
    out.push('\u{256D}'); // ╭
    out.push_str(&"\u{2500}".repeat(left));
    out.push_str(&title_slug);
    out.push_str(&"\u{2500}".repeat(remaining - left));
    out.push('\u{256E}'); // ╮
    out.push('\n');
    for line in &wrapped {
        out.push('\u{2502}'); // │
        out.push_str(&" ".repeat(PAD));
        out.push_str(&pad(line, inner_text, Justify::Left));
        out.push_str(&" ".repeat(PAD));
        out.push('\u{2502}');
        out.push('\n');
    }
    out.push('\u{2570}'); // ╰
    out.push_str(&"\u{2500}".repeat(child_width));
    out.push('\u{256F}'); // ╯
    out
}

#[cfg(test)]
#[path = "console_txt/tests.rs"]
mod tests;
