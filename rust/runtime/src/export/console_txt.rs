// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Native-Rust fixed-width console artifact + warning/insight sink:
//! `profile_export_console.txt`.
//!
//! Ports the width-pinned console record from Python
//! `exporters/exporter_manager.py::_write_console_txt` plus the domain-logic
//! renderers it captures: the grouped metrics tables
//! (`console_metrics_exporter.py`), the error-summary table
//! (`console_error_exporter.py`), and the "earned-in-blood" warning/insight
//! detectors (spec §3): OSL-mismatch, usage-discrepancy, and the API-error
//! insights (MaxCompletionTokens, DynamoSessionControl — exact trigger + fix
//! text/version lore). Render to a fixed `CONSOLE_EXPORT_WIDTH` (140) buffer,
//! decoupled from terminal width; the LIVE terminal Rich rendering stays in the
//! Python parent (the runner subprocess reserves stdout for one JSON line).
//!
//! # Byte-exactness contract (what this module guarantees)
//! The **domain-logic string contracts** are byte-exact against the Python
//! oracles and pinned by golden fixtures in `console_txt/tests.rs`:
//!   * the OSL-mismatch warning body — `console_osl_mismatch_exporter.py`
//!     (`_create_warning_text`);
//!   * the usage-discrepancy warning body —
//!     `console_usage_discrepancy_exporter.py` (`_create_warning_text`);
//!   * the two API-error insight bodies —
//!     `console_api_error_exporter.py` (`_format_text`), including the verbatim
//!     `1.`-prefixed investigation lines (a faithful reproduction of the Python
//!     `"\n  1. ".join(...)` bug) and the v1.3.0-dev / commit d97c889ba version
//!     lore;
//!   * every **cell value** of the error-summary table —
//!     `console_error_exporter.py` (`_format_row`): `N/A` for a missing
//!     code/type and `{count:,}` thousands grouping.
//!
//! Rich console markup (`[bold]`, `[green]`, `[cyan]`, `[dim]…[/dim]`) is
//! stripped exactly as `Console.export_text(styles=False)` does, leaving the
//! literal characters. The threshold values (`5`, `50`, `10`) are the compiled
//! `Environment.METRICS` defaults; an env-overridden threshold is NOT projected
//! onto the runner (the wire `cfg.export.console_txt` carries only
//! `{enabled, width, dev, title, metrics}`), so an operator who overrides
//! `AIPERF_METRICS_*` in Python sees the override in the live terminal render but
//! the compiled default in this artifact.
//!
//! # Rich box-drawing LAYOUT is a faithful port (byte-exact vs Python)
//! The `Table` (`box.HEAVY_HEAD`) and `Panel` (`box.ROUNDED`) renderers here
//! reproduce Rich's geometry byte-for-byte: the `expand=False` column-width
//! solver — including the overflow path where the widest columns are collapsed
//! toward the next-widest (`_collapse_widths` / `ratio_reduce`) and cells are
//! word-wrapped with `overflow="ellipsis"` when the table exceeds the export
//! width — one-cell padding, `HEAVY_HEAD`/`ROUNDED` glyphs, centered space-padded
//! titles, the `header\n(unit)` two-line cell for headers longer than 30 columns
//! (`console_metrics_exporter._format_row`), the `expand=False` panel width
//! solver (content-fit, title-widened, export-width-capped), and the panel-body
//! word-wrap (`Text.wrap` — ports of `rich._wrap.divide_line` /
//! `cells.chop_cells`). Verified equal to `rich==14.1.0` at width 140.
//!
//! The grouped-metrics-table CONTENT (which metric lands in which group, its
//! display header, its display order, the INTERNAL/EXPERIMENTAL filter, and the
//! table titles) is projected from the Python `MetricRegistry`
//! (`ConsoleTxtExportConfig::metrics` / `title`), NOT the Rust `metrics_core`
//! catalog, so the sink reproduces the Python `ConsoleMetricsExporter`
//! byte-for-byte: native-only metrics (`effective_*`, `active_*`,
//! `tokens_in_flight`, `credit_*`, …) are unregistered in Python and therefore
//! render in the DEFAULT group under their raw snake tag rather than in the
//! catalog's Effective/Active tables or being hidden by the catalog's
//! INTERNAL/EXPERIMENTAL flags. A live `AIPERF_EXPORT_SUBDIR=native` same-report
//! diff of `profile_export_console.txt` against the Python exporter is empty.
//!
//! The following residuals remain and are driven by inputs OUTSIDE this sink; the
//! regression goldens below pin this module's own output:
//!   * multi-model runs render the FIRST series' value per metric (Python
//!     pre-aggregates one `MetricResult` per metric);
//!   * the cache-reporting hint line and the dev-only internal / experimental /
//!     http-trace tables are not ported.
//!
//! All trigger data is in the native-v2 report (metric aggregates +
//! `report.errors`). One [`Warning`] value + [`warning_panel`] helper + `detect_*`
//! functions back the panels; no class-per-detector.

use std::collections::BTreeMap;
use std::path::Path;

use crate::export::{ExportConfig, Exporter};
use crate::metrics_core::{
    MetricConsoleGroup, MetricEntry, MetricSeries, NativeReport, ReportStats, ReportValue,
};

/// Artifact filename (Python `artifacts.profile_export_console_txt_file`).
const CONSOLE_TXT_FILENAME: &str = "profile_export_console.txt";

/// Compiled `Environment.METRICS.OSL_MISMATCH_PCT_THRESHOLD` default (percent).
const OSL_MISMATCH_PCT_THRESHOLD: f64 = 5.0;
/// Compiled `Environment.METRICS.OSL_MISMATCH_MAX_TOKEN_THRESHOLD` default.
const OSL_MISMATCH_MAX_TOKEN_THRESHOLD: u64 = 50;
/// Compiled `Environment.METRICS.USAGE_PCT_DIFF_THRESHOLD` default (percent).
const USAGE_PCT_DIFF_THRESHOLD: f64 = 10.0;

/// Console-artifact export policy. Enabled by default (the `.txt` artifact is a
/// stable CI-log surface); the fixed render width is carried here.
///
/// The grouped-metrics-table CONTENT (which metric lands in which
/// [`MetricConsoleGroup`], its display header, its display order, the
/// INTERNAL/EXPERIMENTAL filter, and the group table titles) is projected from
/// the Python `MetricRegistry` via [`ConsoleMetricMeta`] and the `title` field,
/// NOT derived from the Rust `metrics_core` catalog. This is required for byte-exact
/// parity with the Python `ConsoleMetricsExporter`: Python groups differently
/// (native-only `active_*`/`effective_*`/`credit_*` metrics are unregistered and
/// render in the DEFAULT table under their raw snake tag rather than in the
/// catalog's Effective/Active tables), applies a different INTERNAL flag set, and
/// titles the tables `NVIDIA AIPerf | <metrics_title>`. See the frontend
/// projection `rust_wire._console_txt_frontend_projection`.
#[derive(Debug, Clone, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ConsoleTxtExportConfig {
    /// Emit `profile_export_console.txt`.
    pub enabled: bool,
    /// Fixed render width (Python `CONSOLE_EXPORT_WIDTH`, default 140).
    pub width: u16,
    /// Include INTERNAL/EXPERIMENTAL metrics (dev mode). The main metrics table
    /// always hides them (Python `ConsoleMetricsExporter.exclude_flags`); the
    /// dev-only INTERNAL/EXPERIMENTAL/HTTP-trace tables are not ported.
    pub dev: bool,
    /// Base metrics title (Python `ConsoleMetricsExporter._get_title`):
    /// `NVIDIA AIPerf | <endpoint metrics_title>`, or `NVIDIA AIPerf` when the
    /// endpoint dialect is runner-only and has no Python metadata. The DEFAULT
    /// group table uses this verbatim; other groups append `: <Group>`.
    pub title: String,
    /// Registered-metric console metadata keyed by tag, projected from the Python
    /// `MetricRegistry`. A tag ABSENT here is an unregistered (native-only)
    /// metric: it renders in the DEFAULT group under its raw tag, with no display
    /// order (sorts last) and no flag filtering — exactly as Python's
    /// `MetricResult` with an unregistered tag does.
    #[serde(default)]
    pub metrics: BTreeMap<String, ConsoleMetricMeta>,
}

impl Default for ConsoleTxtExportConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            width: 140,
            dev: false,
            title: "NVIDIA AIPerf".to_string(),
            metrics: BTreeMap::new(),
        }
    }
}

/// Per-tag console metadata projected from the Python `MetricRegistry`. Mirrors
/// the `BaseMetric` ClassVars the Python `ConsoleMetricsExporter` reads: the
/// display `header`, the console `group`, the `display_order`, and the
/// INTERNAL/EXPERIMENTAL/ERROR_ONLY flags that gate the standard end-of-run
/// table (`exclude_flags`). Only registered tags appear; absent tags are
/// unregistered native-only metrics (see [`ConsoleTxtExportConfig::metrics`]).
#[derive(Debug, Clone, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ConsoleMetricMeta {
    /// Display header (`BaseMetric.header`).
    pub header: String,
    /// Console group value (`MetricConsoleGroup`, e.g. `usage`, `default`).
    pub group: String,
    /// Display order (`BaseMetric.display_order`); absent sorts last.
    #[serde(default)]
    pub display_order: Option<u32>,
    /// Whether the metric carries `MetricFlags.INTERNAL`.
    #[serde(default)]
    pub internal: bool,
    /// Whether the metric carries `MetricFlags.EXPERIMENTAL`.
    #[serde(default)]
    pub experimental: bool,
    /// Whether the metric carries `MetricFlags.ERROR_ONLY`.
    #[serde(default)]
    pub error_only: bool,
}

/// A detected warning/insight panel: a centered title and an already
/// markup-stripped, value-substituted body. `body` is the byte-exact string
/// contract; [`warning_panel`] wraps it in an approximate Rich `Panel` box.
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
/// Section order mirrors the Python console-exporter registration order
/// (`plugins.yaml` `console_exporter:`), which — because every console exporter
/// prints synchronously before any `await` — is the effective output order:
/// API-error panels, the error-summary table, the grouped metrics tables, the
/// usage-discrepancy panel, then the OSL-mismatch panel.
pub(crate) fn render_console_txt(report: &NativeReport, cfg: &ConsoleTxtExportConfig) -> String {
    let width = cfg.width as usize;
    // Each block carries the count of leading blank lines Rich emits before its
    // renderable, which is set by the originating Python exporter's own
    // `console.print(...)` prefix (recorded verbatim into the export):
    //   * panel exporters call `console.print()` — one blank line;
    //   * the table exporters call `console.print("\n")` — the literal `"\n"`
    //     renderable plus the trailing end newline yield two blank lines.
    // The metrics exporter emits its N grouped tables as a single `Group`, so
    // the whole `metrics_tables` block takes one two-blank prefix (the tables
    // inside the group abut with no separator — see `metrics_tables`).
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
    if let Some(warning) = detect_usage_discrepancy(report) {
        blocks.push((1, warning_panel(&warning, width)));
    }
    if let Some(warning) = detect_osl_mismatch(report) {
        blocks.push((1, warning_panel(&warning, width)));
    }

    // Reproduce the recorded console byte stream: the per-block leading blank
    // lines, the block content, then the block's own trailing line terminator.
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

// ---------------------------------------------------------------------------
// Metric-value access
// ---------------------------------------------------------------------------

/// Numeric payload of a [`ReportValue`] (non-finite tails are treated as absent,
/// matching the detectors' `metric.avg`-is-`None` short circuit).
fn value_f64(value: &ReportValue) -> Option<f64> {
    match value {
        ReportValue::Finite(v) => Some(*v),
        ReportValue::NonFinite => None,
    }
}

/// The aggregate value of a metric equivalent to Python's `MetricResult.avg`:
/// counter totals are summed across series (counters are additive); a
/// distribution/scalar/histogram uses its first series' representative value.
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

// ---------------------------------------------------------------------------
// Number formatting (Python `format()` mini-language subset)
// ---------------------------------------------------------------------------

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

/// Python `f"{n:,}"` for a signed integer.
fn comma_int(n: i64) -> String {
    let sign = if n < 0 { "-" } else { "" };
    format!("{sign}{}", group_thousands(&n.unsigned_abs().to_string()))
}

/// Python `f"{v:,.2f}"` — thousands-grouped, two fixed decimals.
fn comma_2dp(v: f64) -> String {
    let sign = if v.is_sign_negative() && v != 0.0 {
        "-"
    } else {
        ""
    };
    let s = format!("{:.2}", v.abs());
    let (int, frac) = s.split_once('.').unwrap_or((s.as_str(), "00"));
    format!("{sign}{}.{frac}", group_thousands(int))
}

/// Python `f"{v:g}"` for the small integer-valued thresholds used here (`5`,
/// `10`). Rust `Display` already drops a trailing `.0`, matching `%g`.
fn g_fmt(v: f64) -> String {
    format!("{v}")
}

// ---------------------------------------------------------------------------
// Warning / insight detectors
// ---------------------------------------------------------------------------

/// OSL-mismatch warning. Oracle: `console_osl_mismatch_exporter.py`
/// (`export` + `_create_warning_text`). Trigger: `osl_mismatch_count.avg > 0`
/// AND `request_count.avg > 0`.
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

/// Usage-discrepancy warning. Oracle: `console_usage_discrepancy_exporter.py`.
/// Trigger: `usage_discrepancy_count.avg > 0` AND `request_count.avg > 0`.
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

/// The API-error insight detectors, run in the Python `DETECTORS` order
/// (MaxCompletionTokens, then DynamoSessionControl). Oracle:
/// `console_api_error_exporter.py`.
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

/// The error "blob" a detector matches against: the backend `message` field if
/// the raw error message parses as a JSON object with one, else the raw message
/// (mirrors the `orjson.loads(...).get("message")` fallback).
fn error_blob(raw: &str) -> String {
    if let Ok(serde_json::Value::Object(map)) = serde_json::from_str::<serde_json::Value>(raw)
        && let Some(serde_json::Value::String(message)) = map.get("message")
    {
        return message.clone();
    }
    raw.to_string()
}

/// `_format_text` rendering of an insight: markup-stripped, verbatim bullets.
/// Investigation lines are all prefixed `1.`, faithfully reproducing the Python
/// `"\n  1. ".join(...)` off-by-one.
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

// ---------------------------------------------------------------------------
// Error-summary table
// ---------------------------------------------------------------------------

/// The error-summary table. Oracle: `console_error_exporter.py`. Cell values are
/// byte-exact (`N/A` for a missing code/type, `{count:,}` grouping); the box
/// glyphs approximate Rich's `box.HEAVY_HEAD`.
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

// ---------------------------------------------------------------------------
// Grouped metrics tables
// ---------------------------------------------------------------------------

/// Console-group render order (Python `ConsoleMetricsExporter.console_groups`).
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

/// Stat columns (Python `ConsoleMetricsExporter.DEFAULT_STAT_KEYS`).
const STAT_KEYS: &[&str] = &["avg", "min", "max", "p99", "p90", "p50", "std"];

/// A metric row prepared for the table: the projected `display_order` (for the
/// stable sort) plus the rendered cells.
struct MetricRow {
    display_order: u32,
    cells: Vec<String>,
}

/// The grouped metrics tables, one per non-empty console group in
/// [`GROUP_ORDER`]. Oracle: `console_metrics_exporter.py`.
///
/// Grouping / headers / display order / the INTERNAL-EXPERIMENTAL filter / the
/// table titles all come from the frontend-projected [`ConsoleTxtExportConfig`]
/// (the Python `MetricRegistry`), NOT the Rust `metrics_core` catalog, so this
/// reproduces the Python `ConsoleMetricsExporter` byte-for-byte:
///   * a tag present in `cfg.metrics` (registered) uses its projected header /
///     group / display order and is hidden when flagged INTERNAL / EXPERIMENTAL /
///     ERROR_ONLY (Python `exclude_flags`);
///   * a tag absent from `cfg.metrics` (unregistered native-only metric) renders
///     in the DEFAULT group under its raw tag with no display order and no flag
///     filter (Python's `MetricResult` for an unregistered tag).
pub(crate) fn metrics_tables(
    report: &NativeReport,
    cfg: &ConsoleTxtExportConfig,
    width: usize,
) -> Option<String> {
    let mut grouped: Vec<(MetricConsoleGroup, Vec<MetricRow>)> =
        GROUP_ORDER.iter().map(|g| (*g, Vec::new())).collect();

    for (tag, entry) in &report.metrics {
        // Python projects one summary MetricResult per metric via
        // `native_report._summary_series`: a single-series metric uses that
        // series; a multi-series metric uses its one unlabeled aggregate series,
        // and is dropped entirely when it has none (labeled sidecar/server
        // series never reach the primary table).
        let Some(series) = summary_series(entry) else {
            continue;
        };

        let meta = cfg.metrics.get(tag);
        // Registered metrics honor the projected INTERNAL/EXPERIMENTAL/ERROR_ONLY
        // exclusion (Python `ConsoleMetricsExporter.exclude_flags`); unregistered
        // tags carry no flags and are always shown.
        if let Some(meta) = meta
            && (meta.internal || meta.experimental || meta.error_only)
        {
            continue;
        }

        // Registered → projected group; unregistered → DEFAULT (Python
        // `_record_group`: an unregistered tag has no inline override here).
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

        // Registered → projected header; unregistered → the raw snake tag
        // (Python `native_report._metric_result`).
        let header: &str = meta.map_or(tag.as_str(), |meta| meta.header.as_str());
        let display_order = meta.and_then(|meta| meta.display_order).unwrap_or(u32::MAX);

        let mut cells = Vec::with_capacity(1 + STAT_KEYS.len());
        // Python `_format_row`: a header longer than 30 columns pushes the
        // `(unit)` suffix onto a second physical line within the cell; the
        // table renderer honors the embedded newline as a two-line-tall row.
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

/// Map a projected console-group value onto the enum (Python
/// `MetricConsoleGroup`). An unknown value degrades to `None` (hidden), matching
/// a group the render order never includes.
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

/// Title for a console group (Python `_get_group_title`): the projected base
/// title for `DEFAULT`, else `<base>: <Group>` with the enum name title-cased
/// (Python `group.name.title()`).
fn group_title(group: MetricConsoleGroup, base: &str) -> String {
    match group {
        MetricConsoleGroup::Default | MetricConsoleGroup::None => base.to_string(),
        MetricConsoleGroup::Usage => format!("{base}: Usage"),
        MetricConsoleGroup::Cache => format!("{base}: Cache"),
        MetricConsoleGroup::Prediction => format!("{base}: Prediction"),
        MetricConsoleGroup::Audio => format!("{base}: Audio"),
        MetricConsoleGroup::Reasoning => format!("{base}: Reasoning"),
        MetricConsoleGroup::Effective => format!("{base}: Effective"),
        MetricConsoleGroup::Active => format!("{base}: Active"),
    }
}

/// Select the summary series for a metric (Python `native_report._summary_series`):
/// the sole series when there is one, otherwise the unique unlabeled aggregate
/// series among many, or `None` when a multi-series metric has no aggregate (or a
/// malformed second aggregate, which Python raises on — the table drops it here).
fn summary_series(entry: &MetricEntry) -> Option<&MetricSeries> {
    match crate::export::summary_series(&entry.series) {
        crate::export::SummarySeries::Selected(series) => Some(series),
        _ => None,
    }
}

/// Format one stat cell from a metric's first series (`_format_row`): present
/// numbers as `{v:,.2f}`, absent stats as `N/A`.
///
/// The per-type stat projection mirrors the canonical Python
/// `native_report._legacy_stats` exactly (that is the projection the Python
/// console exporter consumes for the same native-v2 report): a **scalar**
/// mirrors its single `value` into `avg`/`min`/`max` (`std` and percentiles are
/// absent → `N/A`); a **counter** mirrors its `total` into `avg`/`min`/`max`; a
/// **histogram** exposes `avg` plus its percentiles (`min`/`max`/`std` absent);
/// a **distribution** carries the full stat set.
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

// ---------------------------------------------------------------------------
// Approximate Rich box renderers
// ---------------------------------------------------------------------------

/// Column text justification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Justify {
    Left,
    Right,
}

/// Display width of a string in terminal columns. This uses the char count as a
/// proxy; the glyphs emitted here (ASCII plus the `•` bullet and box glyphs) are
/// single-width, so this matches Rich for the content rendered.
fn cell_width(text: &str) -> usize {
    text.chars().count()
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

/// Render a Rich `box.HEAVY_HEAD` table with a centered title.
///
/// Column widths follow Rich's `expand=False` solver
/// (`Table._calculate_column_widths`): each flexible column measures to its
/// widest cell *line* plus one cell of padding each side. When the fitted table
/// would exceed the export `width`, the widest columns are collapsed toward the
/// next-widest (`_collapse_widths` / `ratio_reduce`) until the table fits, then
/// each cell is word-wrapped to its column and over-long unbreakable content is
/// ellipsized (`overflow="ellipsis"`) — reproducing Rich's behavior for the wide
/// LLM-metrics table byte-for-byte. A cell may also carry an explicit newline
/// (Python `_format_row`'s `header\n(unit)` split); such a row renders as many
/// physical lines tall as its tallest cell, shorter cells padded with blank
/// lines (Rich's default top vertical alignment).
fn render_table(
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
        // Wrap each cell to its resolved column width (Rich `Text.wrap` with
        // `overflow="ellipsis"`); the row is as tall as the tallest cell. Missing
        // cells / short cells emit blank (padded) lines.
        let split: Vec<Vec<String>> = (0..columns)
            .map(|index| {
                cells.get(index).map_or_else(
                    || vec![String::new()],
                    |cell| wrap_cell(cell, widths[index]),
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

/// Resolve the content width (excluding the one-cell side padding) of each
/// column. Port of Rich `Table._calculate_column_widths` for the `expand=False`,
/// no-`ratio`, no-`min_width` case every table here uses:
///   1. each flexible column measures to its widest cell *line* (`max`) plus the
///      two padding cells;
///   2. if the summed column widths exceed the space left after borders
///      (`export_width - extra_width`), the widest wrapable columns are collapsed
///      toward the next-widest ([`collapse_widths`]), then any residual excess is
///      shaved evenly ([`ratio_reduce`]);
///   3. the padding is removed to yield the content width used for rendering.
///
/// Rich's post-collapse re-measurement only re-clamps each column's maximum to
/// its allocation, so the collapsed widths are final.
fn solve_column_widths(headers: &[&str], rows: &[Vec<String>], export_width: usize) -> Vec<usize> {
    let columns = headers.len();
    // Rich `_extra_width`: two edge borders plus one divider between columns.
    let extra_width = columns + 1;
    let available = export_width.saturating_sub(extra_width);

    // Widest cell *line* per column, over the header cell and every row cell,
    // then plus the two padding cells (Rich measures the padded cell).
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

/// Reduce column widths so their total is under `max_width`, always shaving the
/// widest wrapable column(s) down toward the next-widest. Port of Rich
/// `Table._collapse_widths`.
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

/// Distribute a reduction of `total` across `values` in proportion to `ratios`,
/// each slot capped by its `maximum`. Port of Rich `_ratio.ratio_reduce`.
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
            // Rich rounds ratio * remaining / total_ratio (banker's-free, .5 up).
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

/// Python `round()` for a non-negative quantity: round half to even, matching
/// the CPython banker's rounding Rich's `ratio_reduce` relies on.
fn round_half_up(value: f64) -> i64 {
    let floor = value.floor();
    let diff = value - floor;
    if diff > 0.5 {
        floor as i64 + 1
    } else if diff < 0.5 {
        floor as i64
    } else {
        // Exactly .5 → round to even (Python 3 `round`).
        let f = floor as i64;
        if f % 2 == 0 { f } else { f + 1 }
    }
}

/// Word-wrap one table cell to `width` content cells, reproducing Rich
/// `Text.wrap(overflow="ellipsis")`: split on explicit newlines, word-wrap each
/// logical line without folding over-long words, then crop any physical line
/// wider than the column to `width - 1` cells plus an ellipsis. Right/left
/// justification (padding) happens at render time.
fn wrap_cell(cell: &str, width: usize) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for logical in cell.split('\n') {
        let chars: Vec<char> = logical.chars().collect();
        let breaks = divide_line(&chars, width, false);
        let mut prev = 0;
        let push_piece = |piece: &[char], out: &mut Vec<String>| {
            let text: String = piece.iter().collect::<String>().trim_end().to_string();
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

/// Crop `text` to `width` cells with a trailing ellipsis when it overflows. Port
/// of Rich `Segment`/`Text` truncation for `overflow="ellipsis"`.
fn truncate_ellipsis(text: &str, width: usize) -> String {
    if cell_width(text) <= width {
        return text.to_string();
    }
    if width == 0 {
        return String::new();
    }
    let mut cropped: String = text.chars().take(width - 1).collect();
    cropped.push('\u{2026}'); // …
    cropped
}

/// Split a long word into cell-width-bounded chunks. Port of Rich
/// `cells.chop_cells` (each glyph rendered here is single-width, so cell width
/// equals the character count).
fn chop_cells(word: &[char], width: usize) -> Vec<Vec<char>> {
    let mut lines: Vec<Vec<char>> = vec![Vec::new()];
    let mut total = 0usize;
    for &character in word {
        if total + 1 > width {
            lines.push(vec![character]);
            total = 1;
        } else {
            lines.last_mut().expect("at least one line").push(character);
            total += 1;
        }
    }
    lines
}

/// Tokenize `text` into `\s*\S+\s*` spans (leading whitespace is folded into the
/// following word; trailing-only whitespace yields no span). Port of Rich
/// `_wrap.words`, returning `(start, end)` character offsets.
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

/// Compute the break offsets (character indices) at which `text` must be split
/// to fit `width` cells per line. Port of Rich `_wrap.divide_line`: with
/// `fold=true` an over-long word is chopped across lines; with `fold=false` it
/// is left intact on its own line (the caller then crops/ellipsizes it, as Rich
/// does for `overflow="ellipsis"` table columns).
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
        let word_length = stripped as isize;
        let remaining_space = width_i - cell_offset;
        if remaining_space >= word_length {
            cell_offset += word.len() as isize;
        } else if fold && word_length > width_i {
            let folded = chop_cells(word, width.max(1));
            let last_index = folded.len().saturating_sub(1);
            let mut folded_start = start;
            for (index, line) in folded.iter().enumerate() {
                if folded_start != 0 {
                    breaks.push(folded_start);
                }
                if index == last_index {
                    cell_offset = line.len() as isize;
                } else {
                    folded_start += line.len();
                }
            }
        } else if word_length > width_i {
            // Over-long word with `fold=false` (the fold arm above caught
            // `fold=true`): Rich crops it onto its own line and advances the
            // offset by the FULL word length so the following word breaks.
            if start != 0 {
                breaks.push(start);
            }
            cell_offset = word.len() as isize;
        } else if cell_offset != 0 && start != 0 {
            breaks.push(start);
            cell_offset = word.len() as isize;
        }
    }
    breaks
}

/// Word-wrap a single logical line to `width` cells, returning each physical
/// line with trailing whitespace removed (the panel body pads it back out, so
/// the byte result matches Rich's justify-default padding).
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

/// Render a Rich `Panel` (`box.ROUNDED`, `padding=(0, 2)`, `expand=False`) with
/// the title centered in the top border.
///
/// Reproduces Rich's `Panel.__rich_console__` sizing byte-for-byte: the panel
/// fits its content but is capped at the export `width`; the title (space-padded
/// one cell each side, then centered) can widen a panel narrower than its title;
/// and body lines wider than the inner text region are word-wrapped exactly as
/// Rich's `Text.wrap` (fold overflow) would. All box glyphs are single-width.
pub(crate) fn warning_panel(warning: &Warning, width: usize) -> String {
    const PAD: usize = 2; // padding=(0, 2): two cells left and right.
    let max_width = width.max(2 * PAD + 3);
    // Rich: child_width = measure(Padding(text), max_width-2).maximum, then
    // widened to fit the padded title, clamped to max_width-2. The measured
    // content width is min(longest_line, available_text) + padding.
    let available_text = max_width - 2 - 2 * PAD;
    let body_lines: Vec<&str> = warning.body.split('\n').collect();
    let longest = body_lines.iter().map(|l| cell_width(l)).max().unwrap_or(0);
    let content_child = longest.min(available_text) + 2 * PAD;
    // Rich pads the title with one space on each side, then requires the panel
    // to be at least `padded_title + 2` wide (the two border corners).
    let title_min = warning.title.chars().count() + 2 + 2;
    let child_width = (max_width - 2).min(content_child.max(title_min));
    let inner_text = child_width - 2 * PAD;

    // Word-wrap the body to the inner text region (Rich Text.wrap, fold).
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
