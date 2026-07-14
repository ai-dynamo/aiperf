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
//! `{enabled, width, dev}`), so an operator who overrides `AIPERF_METRICS_*` in
//! Python sees the override in the live terminal render but the compiled default
//! in this artifact.
//!
//! # Rich box-drawing LAYOUT is a faithful port (with documented residuals)
//! The `Table` (`box.HEAVY_HEAD`) and `Panel` (`box.ROUNDED`) renderers here
//! reproduce Rich's geometry byte-for-byte for the content the native report
//! carries: the content-fit column solver (`expand=False`), one-cell cell
//! padding, `HEAVY_HEAD`/`ROUNDED` glyphs, centered space-padded titles, the
//! `header\n(unit)` two-line cell for headers longer than 30 columns
//! (`console_metrics_exporter._format_row`), the `expand=False` panel width
//! solver (content-fit, title-widened, export-width-capped), and the panel-body
//! word-wrap (`Text.wrap`, fold overflow — ports of `rich._wrap.divide_line` /
//! `cells.chop_cells`). Verified equal to `rich==14.1.0` at width 140.
//!
//! The following residuals remain and are driven by inputs OUTSIDE this sink
//! (the native metric CATALOG and the runner's missing Python endpoint
//! metadata), so they cannot be closed from the renderer; the regression
//! goldens below pin this module's own output, not Python parity:
//!   * **Grouping / headers / title.** The native `metrics_core` catalog carries
//!     richer console-group and display-header metadata than Python's
//!     `MetricRegistry`, and native-only metrics (`effective_*`, `active_*`,
//!     `tokens_in_flight`, …) are unregistered in Python. So Python renders one
//!     `NVIDIA AIPerf | LLM Metrics` DEFAULT-group table using raw tag names for
//!     those metrics, while the native sink renders several `NVIDIA AIPerf: <Group>`
//!     tables with proper headers. The catalog also flags some native-only
//!     metrics `INTERNAL`/`EXPERIMENTAL` (e.g. `credit_to_start_latency`,
//!     `effective_latency`), so the native primary tables hide them while Python
//!     — lacking that flag metadata — surfaces them as raw-tag rows. The base
//!     title likewise degrades to `NVIDIA AIPerf` (the runner has no endpoint
//!     `metrics_title`). Closing these needs catalog / registry changes in
//!     `metrics_core`, not this sink.
//!   * a metrics/error table whose fitted width would EXCEED the export width is
//!     not collapsed/re-wrapped (Rich shrinks its flexible columns); the product
//!     metric tables fit within 140, so this is unreached in practice;
//!   * multi-model runs render the FIRST series' value per metric (Python
//!     pre-aggregates one `MetricResult` per metric);
//!   * the cache-reporting hint line and the dev-only internal / experimental /
//!     http-trace tables are not ported.
//!
//! All trigger data is in the native-v2 report (metric aggregates +
//! `report.errors`). One [`Warning`] value + [`warning_panel`] helper + `detect_*`
//! functions back the panels; no class-per-detector.

use std::path::Path;

use crate::export::{ExportConfig, Exporter};
use crate::metrics_core::{
    CATALOG, MetricConsoleGroup, MetricEntry, MetricFlags, NativeReport, ReportStats, ReportValue,
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
#[derive(Debug, Clone, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ConsoleTxtExportConfig {
    /// Emit `profile_export_console.txt`.
    pub enabled: bool,
    /// Fixed render width (Python `CONSOLE_EXPORT_WIDTH`, default 140).
    pub width: u16,
    /// Include INTERNAL/EXPERIMENTAL metrics (dev mode).
    pub dev: bool,
}

impl Default for ConsoleTxtExportConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            width: 140,
            dev: false,
        }
    }
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
    if let Some(table) = error_summary_table(report) {
        blocks.push((2, table));
    }
    if let Some(tables) = metrics_tables(report) {
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
pub(crate) fn error_summary_table(report: &NativeReport) -> Option<String> {
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

/// A metric row prepared for the table: the catalog `display_order` (for the
/// stable sort) plus the rendered cells.
struct MetricRow {
    display_order: u32,
    cells: Vec<String>,
}

/// The grouped metrics tables, one per non-empty console group in
/// [`GROUP_ORDER`]. Oracle: `console_metrics_exporter.py`.
pub(crate) fn metrics_tables(report: &NativeReport) -> Option<String> {
    let mut grouped: Vec<(MetricConsoleGroup, Vec<MetricRow>)> =
        GROUP_ORDER.iter().map(|g| (*g, Vec::new())).collect();

    for (tag, entry) in &report.metrics {
        let Some(spec) = CATALOG.iter().find(|spec| spec.tag.as_str() == tag) else {
            // Sidecar / server-metric entries have no catalog spec — excluded
            // (they own a separate exporter in Python).
            continue;
        };
        // The primary metrics table always hides ERROR_ONLY / INTERNAL /
        // EXPERIMENTAL rows (dev-mode surfaces those via separate tables that
        // are not ported here).
        if spec
            .flags
            .intersects(MetricFlags::ERROR_ONLY | MetricFlags::INTERNAL | MetricFlags::EXPERIMENTAL)
        {
            continue;
        }
        let Some(slot) = grouped
            .iter_mut()
            .find(|(group, _)| *group == spec.console_group)
        else {
            continue; // group None (hidden) or otherwise not rendered.
        };
        let mut cells = Vec::with_capacity(1 + STAT_KEYS.len());
        // Python `_format_row`: a header longer than 30 columns pushes the
        // `(unit)` suffix onto a second physical line within the cell; the
        // table renderer honors the embedded newline as a two-line-tall row.
        let delimiter = if spec.header.chars().count() > 30 {
            "\n"
        } else {
            " "
        };
        cells.push(format!("{}{delimiter}({})", spec.header, entry.unit));
        for key in STAT_KEYS {
            cells.push(stat_cell(entry, key));
        }
        slot.1.push(MetricRow {
            display_order: spec.display_order.unwrap_or(u32::MAX),
            cells,
        });
    }

    let mut blocks = Vec::new();
    for (group, rows) in &mut grouped {
        if rows.is_empty() {
            continue;
        }
        rows.sort_by_key(|row| row.display_order);
        let title = group_title(*group);
        let mut header = vec!["Metric".to_string()];
        header.extend(STAT_KEYS.iter().map(|k| (*k).to_string()));
        let header_refs: Vec<&str> = header.iter().map(String::as_str).collect();
        let justify: Vec<Justify> = std::iter::repeat_n(Justify::Right, header.len()).collect();
        let table_rows: Vec<Vec<String>> = rows.iter().map(|row| row.cells.clone()).collect();
        blocks.push(render_table(&title, &header_refs, &table_rows, &justify));
    }

    if blocks.is_empty() {
        None
    } else {
        Some(blocks.join("\n"))
    }
}

/// Title for a console group (Python `_get_group_title`). The base title
/// degrades to `NVIDIA AIPerf` (no Python endpoint metadata on the runner).
fn group_title(group: MetricConsoleGroup) -> String {
    const BASE: &str = "NVIDIA AIPerf";
    match group {
        MetricConsoleGroup::Default | MetricConsoleGroup::None => BASE.to_string(),
        MetricConsoleGroup::Usage => format!("{BASE}: Usage"),
        MetricConsoleGroup::Cache => format!("{BASE}: Cache"),
        MetricConsoleGroup::Prediction => format!("{BASE}: Prediction"),
        MetricConsoleGroup::Audio => format!("{BASE}: Audio"),
        MetricConsoleGroup::Reasoning => format!("{BASE}: Reasoning"),
        MetricConsoleGroup::Effective => format!("{BASE}: Effective"),
        MetricConsoleGroup::Active => format!("{BASE}: Active"),
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
fn stat_cell(entry: &MetricEntry, key: &str) -> String {
    let Some(series) = entry.series.first() else {
        return "N/A".to_string();
    };
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
/// Column widths fit content (Rich's `expand=False` default): each column is as
/// wide as its widest cell *line*, with one space of padding each side. A cell
/// may contain embedded newlines (Python's `_format_row` `header\n(unit)` split);
/// such a row is rendered as many physical lines tall as its tallest cell, with
/// shorter cells padded out with blank lines (Rich's default top vertical
/// alignment). This matches Rich byte-for-byte while the table's total width
/// stays within the export width; a table whose fitted width would exceed the
/// export width is a documented divergence (Rich collapses/word-wraps its
/// flexible columns; this renderer does not — see the module header).
fn render_table(
    title: &str,
    headers: &[&str],
    rows: &[Vec<String>],
    justify: &[Justify],
) -> String {
    let columns = headers.len();
    let mut widths: Vec<usize> = headers.iter().map(|h| cell_width(h)).collect();
    for row in rows {
        for (index, cell) in row.iter().enumerate().take(columns) {
            let cell_max = cell.split('\n').map(cell_width).max().unwrap_or(0);
            widths[index] = widths[index].max(cell_max);
        }
    }

    let border = |left: char, mid: char, right: char, glyph: char| -> String {
        let parts: Vec<String> = widths
            .iter()
            .map(|w| glyph.to_string().repeat(w + 2))
            .collect();
        format!("{left}{}{right}", parts.join(&mid.to_string()))
    };
    let render_row = |cells: &[String], vbar: char| -> String {
        // Split each cell into physical lines; the row is as tall as the
        // tallest cell. Missing cells / short cells emit blank (padded) lines.
        let split: Vec<Vec<&str>> = (0..columns)
            .map(|index| {
                cells
                    .get(index)
                    .map_or_else(|| vec![""], |cell| cell.split('\n').collect())
            })
            .collect();
        let height = split.iter().map(Vec::len).max().unwrap_or(1);
        let mut out_lines: Vec<String> = Vec::with_capacity(height);
        for line_index in 0..height {
            let mut line = String::new();
            line.push(vbar);
            for (index, &width) in widths.iter().enumerate().take(columns) {
                let text = split[index].get(line_index).copied().unwrap_or("");
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
/// to fit `width` cells per line, folding over-long words. Port of Rich
/// `_wrap.divide_line` with `fold=True`.
fn divide_line(text: &[char], width: usize) -> Vec<usize> {
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
        } else if word_length > width_i {
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
    let breaks = divide_line(&chars, width);
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
