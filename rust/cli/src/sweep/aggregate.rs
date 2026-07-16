// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Sweep aggregation + terminal table.
//!
//! After every cell runs, read each cell's `native-v2.json`, render the live
//! sweep table (gated on a TTY + >1 variation, mirroring Python's
//! `_should_emit_sweep_table`), and write the `sweep_aggregate` artifacts.
//!
//! The `sweep_aggregate/profile_export_aiperf_sweep.{json,csv}` pair is a
//! byte-exact port of Python's `AggregateSweepJsonExporter` /
//! `AggregateSweepCsvExporter` (`src/aiperf/exporters/aggregate/`) driven by
//! `SweepAnalyzer.compute` (`src/aiperf/orchestrator/aggregation/sweep.py`) and
//! the single-trial `_json_metric_to_stats` projection
//! (`src/aiperf/cli_runner/_sweep_aggregate.py`).
//!
//! **Scope: single trial per cell.** With one trial per variation each group's
//! stats come from `_json_metric_to_stats` (a direct read of the cell summary,
//! keyed by metric tag), so `best_configurations`/`pareto_optimal` are always
//! empty (their guards look for flattened `{tag}_{stat}` keys the single-trial
//! projection never produces) and no confidence math runs. The multi-trial
//! (`--num-profile-runs >= 2`) confidence aggregate is intentionally NOT ported:
//! its confidence intervals derive from scipy's Student-t inverse CDF
//! (`scipy.stats.t.ppf`), which is not bit-reproducible in a clean Rust port —
//! that path stays a documented limitation (`docs`), and `finish` warns when it
//! encounters multi-trial groups instead of emitting a divergent summary.

use std::collections::BTreeSet;
use std::io::IsTerminal;
use std::path::{Path, PathBuf};

use serde_json::{Map, Value};

use crate::flags::ProfileFlags;

/// One completed sweep cell.
pub struct CellOutcome {
    /// Variation label (`"path=value, ..."`).
    pub label: String,
    /// The stamped `variation` object (index/label/values).
    pub values: Option<Value>,
    /// The cell's artifact directory.
    pub artifact_dir: PathBuf,
    /// Path to the cell's `native-v2.json` (present on success).
    pub report_path: Option<String>,
    /// Whether the cell committed a report.
    pub success: bool,
    /// Zero-based trial index (groups with >1 trial take the confidence path).
    pub trial: u32,
    /// Failure detail (`None` on success), surfaced in `failed_runs`.
    pub error: Option<String>,
}

/// Headline metrics shown in the live sweep table (tag, stat, column header).
const HEADLINE: &[(&str, &str, &str)] = &[
    ("output_token_throughput", "avg", "out_tok/s"),
    ("time_to_first_token", "p99", "TTFT p99 (ms)"),
    ("inter_token_latency", "p99", "ITL p99 (ms)"),
    ("request_latency", "p95", "req p95 (ms)"),
];

/// Percentile fields carried through the single-trial projection, in the fixed
/// order Python's `_json_metric_to_stats` iterates them.
const PERCENTILE_FIELDS: &[&str] = &["p1", "p5", "p10", "p25", "p50", "p75", "p90", "p95", "p99"];

/// Aggregate the finished cells: render the table and write `sweep_aggregate`.
pub fn finish(flags: &ProfileFlags, outcomes: &[CellOutcome]) -> anyhow::Result<()> {
    let rows: Vec<CellRow> = outcomes.iter().map(row_for).collect();

    // The sweep base dir is the artifact tree the per-cell dirs nest under. For a
    // non-sweep multi-run the cells live at `<base>/profile_runs/run_NNNN`; for a
    // sweep at `<base>/[aggregate/]?(profile_runs/trial_NNNN|)/dir_name`. The
    // sweep aggregate always sits at `<base>/sweep_aggregate` (or, for REPEATED
    // multi-run, `<base>/aggregate/sweep_aggregate`). We derive `<base>` from the
    // configured artifact dir, falling back to the shallowest common ancestor.
    let base = flags
        .artifact_dir
        .clone()
        .or_else(|| common_base(outcomes))
        .unwrap_or_else(|| PathBuf::from("artifacts"));

    if !flags.no_sweep_table && std::io::stdout().is_terminal() && rows.len() > 1 {
        print_table(&rows);
    }

    // Byte-exact aggregate is single-trial only (see module docs). If any group
    // has multiple trials, the confidence path (scipy-bound CIs) applies; we do
    // not emit a divergent summary — warn and skip the aggregate file.
    let multi_trial = outcomes.iter().any(|o| o.trial > 0);
    if multi_trial {
        tracing::warn!(
            "sweep aggregate skipped — multi-trial confidence aggregation \
             (--num-profile-runs >= 2) is not yet native (scipy t-distribution); \
             per-cell reports are on disk."
        );
        return Ok(());
    }

    write_sweep_aggregate(&base, outcomes, flags.confidence_level.unwrap_or(0.95))?;
    Ok(())
}

/// Shallowest common ancestor of every cell's artifact dir (fallback base).
fn common_base(outcomes: &[CellOutcome]) -> Option<PathBuf> {
    outcomes
        .first()
        .and_then(|o| o.artifact_dir.parent())
        .map(Path::to_path_buf)
}

/// A rendered table row: the variation label + extracted headline values.
struct CellRow {
    label: String,
    metrics: Vec<Option<f64>>,
}

fn read_report(o: &CellOutcome) -> Option<Value> {
    o.report_path
        .as_ref()
        .and_then(|p| std::fs::read(p).ok())
        .and_then(|b| serde_json::from_slice::<Value>(&b).ok())
}

fn row_for(o: &CellOutcome) -> CellRow {
    let report = read_report(o);
    let metrics = HEADLINE
        .iter()
        .map(|(tag, stat, _)| report.as_ref().and_then(|r| headline_value(r, tag, stat)))
        .collect();
    CellRow {
        label: o.label.clone(),
        metrics,
    }
}

/// Extract one metric's stat from a `native-v2.json` report (scalar `value`, or a
/// distribution `avg` / `percentiles.pNN`).
fn headline_value(report: &Value, tag: &str, stat: &str) -> Option<f64> {
    let stats = report
        .get("metrics")?
        .get(tag)?
        .get("series")?
        .as_array()?
        .first()?
        .get("stats")?;
    if let Some(v) = stats.get("value").and_then(Value::as_f64) {
        return Some(v);
    }
    match stat {
        "avg" | "min" | "max" | "std" | "count" => stats.get(stat)?.as_f64(),
        p => stats.get("percentiles")?.get(p)?.as_f64(),
    }
}

fn fmt(v: Option<f64>) -> String {
    match v {
        Some(x) => format!("{x:.2}"),
        None => "-".to_string(),
    }
}

/// Render the sweep table to stdout as an aligned text grid.
fn print_table(rows: &[CellRow]) {
    let mut headers = vec!["variation".to_string()];
    headers.extend(HEADLINE.iter().map(|(_, _, h)| h.to_string()));
    let mut grid: Vec<Vec<String>> = vec![headers];
    for r in rows {
        let mut cells = vec![r.label.clone()];
        cells.extend(r.metrics.iter().map(|m| fmt(*m)));
        grid.push(cells);
    }
    let cols = grid[0].len();
    let widths: Vec<usize> = (0..cols)
        .map(|c| grid.iter().map(|row| row[c].len()).max().unwrap_or(0))
        .collect();
    println!();
    for (i, row) in grid.iter().enumerate() {
        let line: Vec<String> = row
            .iter()
            .enumerate()
            .map(|(c, cell)| format!("{cell:<width$}", width = widths[c]))
            .collect();
        println!("{}", line.join("  "));
        if i == 0 {
            println!(
                "{}",
                widths
                    .iter()
                    .map(|w| "-".repeat(*w))
                    .collect::<Vec<_>>()
                    .join("  ")
            );
        }
    }
    println!();
}

// ---------------------------------------------------------------------------
// Byte-exact single-trial sweep aggregate.
// ---------------------------------------------------------------------------

/// A grouped variation cell: its display parameters and per-metric stats.
struct Combo {
    /// Display-name -> config-typed value (leaf name unless it collides).
    parameters: Vec<(String, Value)>,
    /// metric tag -> ordered stats map (`_json_metric_to_stats` shape).
    metrics: Vec<(String, Map<String, Value>)>,
}

/// Build and write the `sweep_aggregate` JSON + CSV pair (single-trial path).
fn write_sweep_aggregate(
    base: &Path,
    outcomes: &[CellOutcome],
    confidence: f64,
) -> anyhow::Result<()> {
    // Group by (label, sorted values), preserving first-seen order. Single-trial
    // means each group is one cell; a duplicate label would pool, but the sweep
    // expander never emits duplicate variation labels.
    let mut combos: Vec<Combo> = Vec::new();
    let mut num_successful = 0usize;
    let mut failed_runs: Vec<Value> = Vec::new();

    for o in outcomes {
        if !o.success {
            failed_runs.push(serde_json::json!({
                "label": o.label,
                "error": o.error,
            }));
            continue;
        }
        let Some(report) = read_report(o) else {
            failed_runs.push(serde_json::json!({
                "label": o.label,
                "error": "missing or unreadable native-v2.json",
            }));
            continue;
        };
        num_successful += 1;
        let parameters = display_parameters(o);
        let metrics = project_summary(&report);
        combos.push(Combo {
            parameters,
            metrics,
        });
    }

    // sweep_parameters: first-seen display-name -> ordered distinct values.
    let sweep_parameters = compute_sweep_parameters(&combos);
    // `_build_metadata`: product of per-parameter distinct-value counts (1 for
    // an empty parameter set — an empty product).
    let num_combinations: i64 = sweep_parameters.iter().map(|p| p.1.len() as i64).product();

    let json = build_sweep_json(
        outcomes.len(),
        num_successful,
        &failed_runs,
        &sweep_parameters,
        num_combinations,
        &combos,
        confidence,
    );
    let csv = build_sweep_csv(
        outcomes.len(),
        num_successful,
        &sweep_parameters,
        num_combinations,
        &combos,
    );

    let dir = base.join("sweep_aggregate");
    std::fs::create_dir_all(&dir)?;
    // orjson OPT_INDENT_2 output; serde_json pretty matches it byte-for-byte for
    // ASCII payloads (2-space indent, `": "`, `\n` newlines, no trailing space).
    let json_path = dir.join("profile_export_aiperf_sweep.json");
    let csv_path = dir.join("profile_export_aiperf_sweep.csv");
    std::fs::write(&json_path, serde_json::to_string_pretty(&json)?)?;
    std::fs::write(&csv_path, csv)?;
    // Mirror Python's `_post_process`/`export_helpers` write lines.
    tracing::info!("Sweep aggregate JSON written to: {}", json_path.display());
    tracing::info!("Sweep aggregate CSV written to: {}", csv_path.display());
    Ok(())
}

/// The `parameters` dict for a cell: display-name -> config-typed value, sorted
/// by dotted path (mirrors Python's `_short_values_dict` over sorted key-values).
fn display_parameters(o: &CellOutcome) -> Vec<(String, Value)> {
    let values = o
        .values
        .as_ref()
        .and_then(|v| v.get("values"))
        .and_then(Value::as_object);
    let Some(values) = values else {
        return Vec::new();
    };
    // Sort by dotted path (Python keys the VariationKey on sorted values).
    let mut pairs: Vec<(String, Value)> =
        values.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
    pairs.sort_by(|a, b| a.0.cmp(&b.0));
    // Resolve leaf display names with collision fallback to the dotted path.
    let display = display_names(&pairs.iter().map(|(k, _)| k.clone()).collect::<Vec<_>>());
    pairs
        .into_iter()
        .map(|(k, v)| (display.get(&k).cloned().unwrap_or(k), v))
        .collect()
}

/// Map dotted paths to leaf display names, falling back to the full dotted path
/// when two paths share a leaf (`_parameter_display_names`).
fn display_names(paths: &[String]) -> std::collections::HashMap<String, String> {
    let leaf = |p: &str| p.rsplit('.').next().unwrap_or(p).to_string();
    let mut leaf_counts: std::collections::HashMap<String, usize> =
        std::collections::HashMap::new();
    for p in paths {
        *leaf_counts.entry(leaf(p)).or_default() += 1;
    }
    paths
        .iter()
        .map(|p| {
            let l = leaf(p);
            let name = if leaf_counts.get(&l) == Some(&1) {
                l
            } else {
                p.clone()
            };
            (p.clone(), name)
        })
        .collect()
}

/// Project a native report's `metrics` map into the single-trial stats shape
/// (`_legacy_stats` -> `_json_metric_to_stats`), preserving report metric order.
fn project_summary(report: &Value) -> Vec<(String, Map<String, Value>)> {
    let Some(metrics) = report.get("metrics").and_then(Value::as_object) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for (tag, entry) in metrics {
        let Some(entry) = entry.as_object() else {
            continue;
        };
        let Some(series) = summary_series(entry) else {
            continue;
        };
        let Some(stats) = series.get("stats").and_then(Value::as_object) else {
            continue;
        };
        let mtype = entry.get("type").and_then(Value::as_str).unwrap_or("");
        let unit = entry.get("unit").and_then(Value::as_str).unwrap_or("");
        if let Some(m) = json_metric_stats(mtype, stats, unit) {
            out.push((tag.clone(), m));
        }
    }
    out
}

/// Pick the single-series or unlabeled-aggregate series (`_summary_series`).
fn summary_series(entry: &Map<String, Value>) -> Option<&Value> {
    let series = entry.get("series")?.as_array()?;
    if series.len() == 1 {
        return series.first();
    }
    let mut aggregate = series
        .iter()
        .filter(|s| s.get("labels").map(Value::is_null).unwrap_or(true));
    let first = aggregate.next();
    if aggregate.next().is_some() {
        return None; // multiple unlabeled aggregates: unrepresentable
    }
    first
}

/// The `_legacy_stats` projection of one native metric series: the raw facts a
/// metric-type carries, before the single-trial `_json_metric_to_stats` collapse.
#[derive(Default)]
struct LegacyStats {
    avg: Option<f64>,
    min: Option<f64>,
    max: Option<f64>,
    count: Option<u64>,
    sum: Option<f64>,
    percentiles: Vec<(String, f64)>,
}

/// Read the present percentile fields (fixed order) off a stats block.
fn read_percentiles(stats: &Map<String, Value>) -> Vec<(String, f64)> {
    stats
        .get("percentiles")
        .and_then(Value::as_object)
        .map(|p| {
            PERCENTILE_FIELDS
                .iter()
                .filter_map(|k| {
                    p.get(*k)
                        .and_then(Value::as_f64)
                        .map(|v| (k.to_string(), v))
                })
                .collect()
        })
        .unwrap_or_default()
}

/// Build the ordered per-metric stats map for one metric, matching
/// `_legacy_stats` (native type projection) fed through `_json_metric_to_stats`
/// (single-trial: std/cv/ci collapse; percentiles/count/sum carried through).
/// The report `std` is intentionally dropped — single-trial std is a hard 0.0.
fn json_metric_stats(
    mtype: &str,
    stats: &Map<String, Value>,
    unit: &str,
) -> Option<Map<String, Value>> {
    let num = |k: &str| stats.get(k).and_then(Value::as_f64);
    let int = |k: &str| stats.get(k).and_then(Value::as_u64);

    let legacy = match mtype {
        "distribution" => LegacyStats {
            avg: num("avg"),
            min: num("min"),
            max: num("max"),
            count: int("count"),
            percentiles: read_percentiles(stats),
            ..Default::default()
        },
        "scalar" => {
            let v = num("value")?;
            LegacyStats {
                avg: Some(v),
                min: Some(v),
                max: Some(v),
                ..Default::default()
            }
        }
        "counter" => {
            let v = num("total")?;
            LegacyStats {
                avg: Some(v),
                min: Some(v),
                max: Some(v),
                sum: Some(v),
                ..Default::default()
            }
        }
        "histogram" => LegacyStats {
            avg: num("avg"),
            count: int("count"),
            sum: num("sum"),
            percentiles: read_percentiles(stats),
            ..Default::default()
        },
        _ => return None,
    };

    // `_json_metric_to_stats`: avg drives every point estimate; std/cv/ci = 0.
    let avg = legacy.avg.unwrap_or(0.0);
    let mut m = Map::new();
    m.insert("mean".into(), f(avg));
    m.insert("avg".into(), f(avg));
    m.insert("std".into(), f(0.0));
    m.insert("min".into(), f(legacy.min.unwrap_or(avg)));
    m.insert("max".into(), f(legacy.max.unwrap_or(avg)));
    m.insert("cv".into(), f(0.0));
    m.insert("ci_low".into(), f(avg));
    m.insert("ci_high".into(), f(avg));
    m.insert("unit".into(), Value::String(unit.to_string()));
    for (k, v) in legacy.percentiles {
        m.insert(k, f(v));
    }
    if let Some(c) = legacy.count {
        m.insert("count".into(), Value::from(c));
    }
    if let Some(s) = legacy.sum {
        m.insert("sum".into(), f(s));
    }
    Some(m)
}

/// Wrap an f64 as a JSON number (finite; NaN/inf never reach here for the
/// single-trial constants and report values are validated numeric).
fn f(v: f64) -> Value {
    serde_json::Number::from_f64(v)
        .map(Value::Number)
        .unwrap_or(Value::Null)
}

/// `_compute_sweep_parameters`: first-seen display name -> distinct values list.
fn compute_sweep_parameters(combos: &[Combo]) -> Vec<(String, Vec<Value>)> {
    let mut out: Vec<(String, Vec<Value>)> = Vec::new();
    for combo in combos {
        for (name, value) in &combo.parameters {
            match out.iter_mut().find(|(n, _)| n == name) {
                Some((_, values)) => {
                    if !values.iter().any(|v| v == value) {
                        values.push(value.clone());
                    }
                }
                None => out.push((name.clone(), vec![value.clone()])),
            }
        }
    }
    out
}

/// Assemble the byte-exact `profile_export_aiperf_sweep.json` document.
fn build_sweep_json(
    num_runs: usize,
    num_successful: usize,
    failed_runs: &[Value],
    sweep_parameters: &[(String, Vec<Value>)],
    num_combinations: i64,
    combos: &[Combo],
    confidence: f64,
) -> Value {
    let sweep_params_json: Vec<Value> = sweep_parameters
        .iter()
        .map(|(name, values)| serde_json::json!({"name": name, "values": values}))
        .collect();

    let mut metadata = Map::new();
    metadata.insert(
        "sweep_parameters".into(),
        Value::Array(sweep_params_json.clone()),
    );
    metadata.insert("num_combinations".into(), Value::from(num_combinations));
    metadata.insert("sweep_mode".into(), Value::String("repeated".into()));
    metadata.insert("confidence_level".into(), f(confidence));
    metadata.insert(
        "num_trials_per_value".into(),
        Value::from(if combos.is_empty() { 0 } else { 1 }),
    );
    metadata.insert("aggregation_type".into(), Value::String("sweep".into()));

    let per_combination: Vec<Value> = combos
        .iter()
        .map(|c| {
            let mut params = Map::new();
            for (name, value) in &c.parameters {
                params.insert(name.clone(), value.clone());
            }
            let mut metrics = Map::new();
            for (tag, stats) in &c.metrics {
                metrics.insert(tag.clone(), Value::Object(stats.clone()));
            }
            serde_json::json!({"parameters": params, "metrics": metrics})
        })
        .collect();

    let mut root = Map::new();
    root.insert("aggregation_type".into(), Value::String("sweep".into()));
    root.insert("num_profile_runs".into(), Value::from(num_runs as u64));
    root.insert(
        "num_successful_runs".into(),
        Value::from(num_successful as u64),
    );
    root.insert("failed_runs".into(), Value::Array(failed_runs.to_vec()));
    root.insert("metadata".into(), Value::Object(metadata));
    root.insert(
        "per_combination_metrics".into(),
        Value::Array(per_combination),
    );
    root.insert("best_configurations".into(), Value::Object(Map::new()));
    root.insert("pareto_optimal".into(), Value::Array(Vec::new()));
    Value::Object(root)
}

/// Assemble the byte-exact `profile_export_aiperf_sweep.csv` (CRLF, csv module
/// quoting rules). Ports `AggregateSweepCsvExporter._generate_content`.
fn build_sweep_csv(
    num_runs: usize,
    num_successful: usize,
    sweep_parameters: &[(String, Vec<Value>)],
    num_combinations: i64,
    combos: &[Combo],
) -> String {
    let param_names: Vec<String> = sweep_parameters.iter().map(|(n, _)| n.clone()).collect();
    let mut w = CsvWriter::new();

    if !combos.is_empty() {
        // Metric column set = UNION of metric tags across combos, sorted.
        let metric_names: BTreeSet<String> = combos
            .iter()
            .flat_map(|c| c.metrics.iter().map(|(t, _)| t.clone()))
            .collect();
        let metric_names: Vec<String> = metric_names.into_iter().collect();

        let mut header = param_names.clone();
        for m in &metric_names {
            header.push(format!("{m}_mean"));
            header.push(format!("{m}_std"));
            header.push(format!("{m}_min"));
            header.push(format!("{m}_max"));
            header.push(format!("{m}_cv"));
        }
        w.row(&header);

        for combo in combos {
            let mut row: Vec<String> = param_names.iter().map(|p| combo_param(combo, p)).collect();
            for m in &metric_names {
                let stats = combo.metrics.iter().find(|(t, _)| t == m).map(|(_, s)| s);
                match stats {
                    Some(s) => {
                        row.push(fmt_num(s.get("mean"), 2));
                        row.push(fmt_num(s.get("std"), 2));
                        row.push(fmt_num(s.get("min"), 2));
                        row.push(fmt_num(s.get("max"), 2));
                        row.push(fmt_num(s.get("cv"), 4));
                    }
                    None => row.extend(["", "", "", "", ""].map(String::from)),
                }
            }
            w.row(&row);
        }
    }

    // Section 2: Best Configurations (always empty on the single-trial path).
    w.row(&[] as &[String]);
    w.row(&["Best Configurations".to_string()]);

    // Section 3: Pareto Optimal Points (always empty on the single-trial path).
    w.row(&[] as &[String]);
    w.row(&["Pareto Optimal Points".to_string()]);
    w.row(&["None".to_string()]);

    // Section 4: Metadata.
    w.row(&[] as &[String]);
    w.row(&["Metadata".to_string()]);
    w.row(&["Field".to_string(), "Value".to_string()]);
    w.row(&["Aggregation Type".to_string(), "sweep".to_string()]);
    w.row(&["Sweep Parameters".to_string(), param_names.join(", ")]);
    w.row(&[
        "Number of Combinations".to_string(),
        num_combinations.to_string(),
    ]);
    w.row(&["Number of Profile Runs".to_string(), num_runs.to_string()]);
    w.row(&[
        "Number of Successful Runs".to_string(),
        num_successful.to_string(),
    ]);

    w.finish()
}

/// Render a parameter value for a CSV cell (matches Python: int stays bare,
/// missing renders empty). Values come straight from the variation dict.
fn combo_param(combo: &Combo, name: &str) -> String {
    match combo.parameters.iter().find(|(n, _)| n == name) {
        Some((_, Value::String(s))) => s.clone(),
        Some((_, v)) => v.to_string(),
        None => String::new(),
    }
}

/// Python `_format_number`: None/non-finite -> ""; float -> fixed decimals;
/// int -> bare. Our stats store every number as f64, so all render as fixed.
fn fmt_num(value: Option<&Value>, decimals: usize) -> String {
    match value.and_then(Value::as_f64) {
        Some(v) if v.is_finite() => format!("{v:.decimals$}"),
        _ => String::new(),
    }
}

/// Minimal CSV writer matching Python's `csv.writer` defaults: `\r\n` line
/// terminator, comma delimiter, `"`-quote only when a field needs it (contains
/// comma, quote, CR, or LF), doubling embedded quotes.
struct CsvWriter {
    buf: String,
}

impl CsvWriter {
    fn new() -> Self {
        CsvWriter { buf: String::new() }
    }

    fn row<S: AsRef<str>>(&mut self, fields: &[S]) {
        let line: Vec<String> = fields.iter().map(|f| Self::quote(f.as_ref())).collect();
        self.buf.push_str(&line.join(","));
        self.buf.push_str("\r\n");
    }

    fn quote(field: &str) -> String {
        if field.contains([',', '"', '\r', '\n']) {
            format!("\"{}\"", field.replace('"', "\"\""))
        } else {
            field.to_string()
        }
    }

    fn finish(self) -> String {
        self.buf
    }
}
