// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Sweep aggregation and terminal table.
//!
//! With one trial per variation, each group's stats come directly from the
//! cell's `native-v2.json` summary keyed by metric tag, so
//! `best_configurations`/`pareto_optimal` are
//! always empty (their guards look for flattened `{tag}_{stat}` keys the
//! single-trial projection never produces) and no confidence math runs.
//! Multi-trial runs (`--num-profile-runs >= 2`) compute confidence aggregates
//! via [`crate::sweep::confidence`]. The non-sweep path writes
//! `<base>/aggregate/profile_export_aiperf_aggregate.{json,csv}` (+ a collated
//! detailed JSON when `--convergence-metric` is set), and the sweep path writes
//! per-variation confidence aggregates plus the cross-variation
//! `sweep_aggregate` with `best_configurations`/`pareto_optimal`. The
//! Student-t inverse CDF uses an approximately `1e-10` bisection tolerance.

use std::collections::BTreeSet;
use std::io::IsTerminal;
use std::path::{Path, PathBuf};

use serde_json::{Map, Value};

use crate::flags::ProfileFlags;
use crate::sweep::artifact_dir::IterationOrder;
use crate::sweep::confidence::{self, ConfidenceMetric};

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
/// serialization order required by the single-trial artifact.
const PERCENTILE_FIELDS: &[&str] = &["p1", "p5", "p10", "p25", "p50", "p75", "p90", "p95", "p99"];

/// Aggregate the finished cells: render the table and write the aggregate
/// artifacts. Returns the process exit code (`0` full success; `1` when fewer
/// than two runs succeeded).
///
/// `is_sweep` selects the sweep vs non-sweep multi-run layout; `order` selects
/// the REPEATED/INDEPENDENT per-variation + sweep-aggregate directory placement.
pub fn finish(
    flags: &ProfileFlags,
    outcomes: &[CellOutcome],
    is_sweep: bool,
    order: IterationOrder,
) -> anyhow::Result<i32> {
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

    let confidence = flags.confidence_level.unwrap_or(0.95);
    let multi_trial = outcomes.iter().any(|o| o.trial > 0);

    if !multi_trial {
        let successful = outcomes.iter().filter(|o| o.success).count();
        write_sweep_aggregate(&base, outcomes, confidence)?;
        let failed = outcomes.len() - successful;
        return Ok(if failed > 0 { 1 } else { 0 });
    }

    let classified: Vec<Classified> = outcomes.iter().map(classify).collect();
    let cooldown = flags
        .profile_run_cooldown_seconds
        .or(flags.parameter_sweep_cooldown_seconds)
        .unwrap_or(0.0);
    let successful = classified.iter().filter(|c| c.success).count();

    if successful < 2 {
        if is_sweep {
            tracing::warn!(
                "Only {successful} variation(s) succeeded - cannot compute sweep \
                 aggregate statistics. At least 2 successful runs are required."
            );
        } else {
            tracing::warn!(
                "Only {successful} successful run(s) - cannot compute confidence \
                 statistics. At least 2 successful runs are required."
            );
        }
        return Ok(1);
    }

    if is_sweep {
        write_sweep_confidence(&base, &classified, order, confidence, cooldown)?;
    } else {
        let total = classified.len();
        let success: Vec<&Classified> = classified.iter().filter(|c| c.success).collect();
        let summaries: Vec<&Value> = success.iter().filter_map(|c| c.summary.as_ref()).collect();
        let metrics = crate::sweep::confidence::collect_confidence_metrics(&summaries, confidence);
        let run_labels: Vec<String> = success.iter().map(|c| c.label.clone()).collect();
        let failed = failed_runs(&classified);
        let dir = base.join("aggregate");
        crate::sweep::confidence::write_confidence_aggregate(
            &dir,
            total,
            success.len(),
            &failed,
            &run_labels,
            confidence,
            cooldown,
            false,
            &metrics,
            &[],
        )?;
        if flags.convergence_metric.is_some() {
            write_detailed(&dir, &classified, cooldown)?;
        }
    }
    Ok(0)
}

/// A cell's stable label and completed-benchmark verdict.
struct Classified {
    /// The run/trial label (`run_NNNN`).
    label: String,
    /// Whether the run committed a usable benchmark.
    success: bool,
    /// Failure detail (`None` on success).
    error: Option<String>,
    /// The `profile_export_aiperf.json` summary (present on success).
    summary: Option<Value>,
    /// The cell's artifact directory (holds the per-request `profile_export.jsonl`).
    artifact_dir: PathBuf,
    /// The cell's variation label (`""` / `"base"` for a non-sweep multi-run).
    variation_label: String,
    /// The cell's stamped variation (index/label/values).
    variation: Option<Value>,
}

/// Mark a process-successful cell failed when it recorded no completed requests.
fn classify(o: &CellOutcome) -> Classified {
    let label = format!("run_{:04}", o.trial + 1);
    let base = Classified {
        label,
        success: false,
        error: o.error.clone(),
        summary: None,
        artifact_dir: o.artifact_dir.clone(),
        variation_label: o.label.clone(),
        variation: o.values.clone(),
    };
    if !o.success {
        return base;
    }
    let summary = crate::sweep::confidence::read_summary(&o.artifact_dir);
    match &summary {
        Some(s) => match crate::sweep::confidence::classify_summary(s) {
            Ok(()) => Classified {
                success: true,
                error: None,
                summary,
                ..base
            },
            Err(e) => Classified {
                error: Some(e),
                ..base
            },
        },
        None => Classified {
            error: Some("No requests completed".to_string()),
            ..base
        },
    }
}

/// The `{label, error}` records for every failed cell.
fn failed_runs(classified: &[Classified]) -> Vec<crate::sweep::confidence::FailedRun> {
    classified
        .iter()
        .filter(|c| !c.success)
        .map(|c| crate::sweep::confidence::FailedRun {
            label: c.label.clone(),
            error: c.error.clone(),
        })
        .collect()
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

/// A grouped variation cell: its display parameters and per-metric stats.
struct Combo {
    /// Display-name -> config-typed value (leaf name unless it collides).
    parameters: Vec<(String, Value)>,
    /// Metric tag to ordered stats map.
    metrics: Vec<(String, Map<String, Value>)>,
}

/// Build and write the `sweep_aggregate` JSON + CSV pair (single-trial path).
fn write_sweep_aggregate(
    base: &Path,
    outcomes: &[CellOutcome],
    confidence: f64,
) -> anyhow::Result<()> {
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

    let sweep_parameters = compute_sweep_parameters(&combos);
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
    // Artifact JSON uses two-space indentation and LF line endings.
    let json_path = dir.join("profile_export_aiperf_sweep.json");
    let csv_path = dir.join("profile_export_aiperf_sweep.csv");
    std::fs::write(&json_path, serde_json::to_string_pretty(&json)?)?;
    std::fs::write(&csv_path, csv)?;
    tracing::info!("Sweep aggregate JSON written to: {}", json_path.display());
    tracing::info!("Sweep aggregate CSV written to: {}", csv_path.display());
    Ok(())
}

/// The `parameters` dict for a cell: display-name -> config-typed value, sorted
/// by dotted path.
fn display_parameters(o: &CellOutcome) -> Vec<(String, Value)> {
    let values = o
        .values
        .as_ref()
        .and_then(|v| v.get("values"))
        .and_then(Value::as_object);
    let Some(values) = values else {
        return Vec::new();
    };
    // Dotted-path order is part of the artifact contract.
    let mut pairs: Vec<(String, Value)> =
        values.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
    pairs.sort_by(|a, b| a.0.cmp(&b.0));
    let display = display_names(&pairs.iter().map(|(k, _)| k.clone()).collect::<Vec<_>>());
    pairs
        .into_iter()
        .map(|(k, v)| (display.get(&k).cloned().unwrap_or(k), v))
        .collect()
}

/// Map dotted paths to leaf display names, retaining full paths on collisions.
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

/// Project a native report's `metrics` map into the single-trial stats shape.
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

/// Pick the sole series or sole unlabeled aggregate series.
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

/// Raw facts carried by one native metric series.
#[derive(Default)]
struct ProjectedStats {
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

/// Build the ordered single-trial stats map for one metric.
/// The report `std` is intentionally dropped — single-trial std is a hard 0.0.
fn json_metric_stats(
    mtype: &str,
    stats: &Map<String, Value>,
    unit: &str,
) -> Option<Map<String, Value>> {
    let num = |k: &str| stats.get(k).and_then(Value::as_f64);
    let int = |k: &str| stats.get(k).and_then(Value::as_u64);

    let projected = match mtype {
        "distribution" => ProjectedStats {
            avg: num("avg"),
            min: num("min"),
            max: num("max"),
            count: int("count"),
            percentiles: read_percentiles(stats),
            ..Default::default()
        },
        "scalar" => {
            let v = num("value")?;
            ProjectedStats {
                avg: Some(v),
                min: Some(v),
                max: Some(v),
                ..Default::default()
            }
        }
        "counter" => {
            let v = num("total")?;
            ProjectedStats {
                avg: Some(v),
                min: Some(v),
                max: Some(v),
                sum: Some(v),
                ..Default::default()
            }
        }
        "histogram" => ProjectedStats {
            avg: num("avg"),
            count: int("count"),
            sum: num("sum"),
            percentiles: read_percentiles(stats),
            ..Default::default()
        },
        _ => return None,
    };

    // Single-trial dispersion and confidence intervals collapse to the mean.
    let avg = projected.avg.unwrap_or(0.0);
    let mut m = Map::new();
    m.insert("mean".into(), f(avg));
    m.insert("avg".into(), f(avg));
    m.insert("std".into(), f(0.0));
    m.insert("min".into(), f(projected.min.unwrap_or(avg)));
    m.insert("max".into(), f(projected.max.unwrap_or(avg)));
    m.insert("cv".into(), f(0.0));
    m.insert("ci_low".into(), f(avg));
    m.insert("ci_high".into(), f(avg));
    m.insert("unit".into(), Value::String(unit.to_string()));
    for (k, v) in projected.percentiles {
        m.insert(k, f(v));
    }
    if let Some(c) = projected.count {
        m.insert("count".into(), Value::from(c));
    }
    if let Some(s) = projected.sum {
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

/// Collect distinct parameter values in first-seen order.
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

/// Assemble `profile_export_aiperf_sweep.json`.
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

/// Assemble `profile_export_aiperf_sweep.csv` with CRLF and standard CSV
/// quoting.
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

    w.row(&[] as &[String]);
    w.row(&["Best Configurations".to_string()]);

    w.row(&[] as &[String]);
    w.row(&["Pareto Optimal Points".to_string()]);
    w.row(&["None".to_string()]);

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

/// Render a parameter value for a CSV cell; integers stay bare and missing
/// values render empty.
fn combo_param(combo: &Combo, name: &str) -> String {
    match combo.parameters.iter().find(|(n, _)| n == name) {
        Some((_, Value::String(s))) => s.clone(),
        Some((_, v)) => v.to_string(),
        None => String::new(),
    }
}

/// Render finite values with fixed decimals and absent values as empty.
fn fmt_num(value: Option<&Value>, decimals: usize) -> String {
    match value.and_then(Value::as_f64) {
        Some(v) if v.is_finite() => format!("{v:.decimals$}"),
        _ => String::new(),
    }
}

/// CSV writer using CRLF, comma delimiters, and doubled embedded quotes.
pub(crate) struct CsvWriter {
    buf: String,
}

impl CsvWriter {
    pub(crate) fn new() -> Self {
        CsvWriter { buf: String::new() }
    }

    pub(crate) fn row<S: AsRef<str>>(&mut self, fields: &[S]) {
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

    pub(crate) fn finish(self) -> String {
        self.buf
    }
}

/// One variation grouped across trials.
struct VariationGroup<'a> {
    /// The variation label (cell identity half of the group key).
    label: String,
    /// Sorted `(dotted_path, config-typed value)` pairs (values half of the key).
    values: Vec<(String, Value)>,
    /// The classified trials of this variation, in run order.
    members: Vec<&'a Classified>,
}

/// Group cells by `(variation_label, sorted values)` in first-seen order.
fn group_by_variation<'a>(classified: &'a [Classified]) -> Vec<VariationGroup<'a>> {
    let mut groups: Vec<VariationGroup> = Vec::new();
    for c in classified {
        let values = variation_values(c);
        if let Some(g) = groups
            .iter_mut()
            .find(|g| g.label == c.variation_label && g.values == values)
        {
            g.members.push(c);
        } else {
            groups.push(VariationGroup {
                label: c.variation_label.clone(),
                values,
                members: vec![c],
            });
        }
    }
    groups
}

/// Extract a cell's sorted `(dotted_path, value)` variation pairs.
fn variation_values(c: &Classified) -> Vec<(String, Value)> {
    let Some(obj) = c
        .variation
        .as_ref()
        .and_then(|v| v.get("values"))
        .and_then(Value::as_object)
    else {
        return Vec::new();
    };
    let mut pairs: Vec<(String, Value)> = obj.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
    pairs.sort_by(|a, b| a.0.cmp(&b.0));
    pairs
}

/// Leaf of a dotted path (`phases.profiling.concurrency` -> `concurrency`).
fn leaf(path: &str) -> String {
    path.rsplit('.').next().unwrap_or(path).to_string()
}

/// Render a config-typed value for a directory-name segment / label (`2`, `2.0`).
fn render_value(v: &Value) -> String {
    match v {
        Value::Number(n) if n.is_i64() || n.is_u64() => n.to_string(),
        Value::Number(n) => {
            let f = n.as_f64().unwrap_or(0.0);
            if f.fract() == 0.0 && f.is_finite() {
                format!("{f:.1}")
            } else {
                format!("{f}")
            }
        }
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

/// The `{seg}_{value}__...` directory name for a variation.
fn group_dir_name(values: &[(String, Value)]) -> String {
    values
        .iter()
        .map(|(path, val)| format!("{}_{}", leaf(path), render_value(val)))
        .collect::<Vec<_>>()
        .join("__")
}

/// Display leaf parameter names unless a leaf collides across groups.
fn group_display_names(groups: &[VariationGroup]) -> std::collections::HashMap<String, String> {
    let mut paths: Vec<String> = Vec::new();
    for g in groups {
        for (path, _) in &g.values {
            if !paths.iter().any(|p| p == path) {
                paths.push(path.clone());
            }
        }
    }
    display_names(&paths)
}

/// The REPEATED/INDEPENDENT string for sweep metadata + directory placement.
fn sweep_mode_str(order: IterationOrder) -> &'static str {
    match order {
        IterationOrder::Repeated => "repeated",
        IterationOrder::Independent => "independent",
    }
}

/// Per-variation confidence-aggregate directory:
/// repeated -> `<base>/aggregate/<dir>`, independent -> `<base>/<dir>/aggregate`.
fn per_variation_aggregate_dir(base: &Path, dir_name: &str, order: IterationOrder) -> PathBuf {
    match order {
        IterationOrder::Repeated => base.join("aggregate").join(dir_name),
        IterationOrder::Independent => base.join(dir_name).join("aggregate"),
    }
}

/// Sweep-level aggregate directory: repeated multi-run uses
/// `<base>/aggregate/sweep_aggregate`; otherwise `<base>/sweep_aggregate`.
fn sweep_aggregate_dir(base: &Path, order: IterationOrder) -> PathBuf {
    match order {
        IterationOrder::Repeated => base.join("aggregate").join("sweep_aggregate"),
        IterationOrder::Independent => base.join("sweep_aggregate"),
    }
}

/// One combination's confidence stats for the cross-variation sweep aggregate.
struct ComboStats {
    /// Display-name -> config-typed value (leaf unless collision).
    params: Vec<(String, Value)>,
    /// Flattened `{tag}_{stat}` -> confidence metric.
    metrics: Vec<(String, ConfidenceMetric)>,
}

/// Write the per-variation confidence aggregates and the cross-variation sweep
/// aggregate for a multi-trial sweep.
fn write_sweep_confidence(
    base: &Path,
    classified: &[Classified],
    order: IterationOrder,
    confidence: f64,
    cooldown: f64,
) -> anyhow::Result<()> {
    let groups = group_by_variation(classified);
    let display = group_display_names(&groups);
    let mode = sweep_mode_str(order);

    for g in &groups {
        let success: Vec<&Classified> = g.members.iter().copied().filter(|c| c.success).collect();
        if success.is_empty() {
            continue;
        }
        let summaries: Vec<&Value> = success.iter().filter_map(|c| c.summary.as_ref()).collect();
        let metrics = confidence::collect_confidence_metrics(&summaries, confidence);
        let run_labels: Vec<String> = success.iter().map(|c| c.label.clone()).collect();
        let failed: Vec<confidence::FailedRun> = g
            .members
            .iter()
            .filter(|c| !c.success)
            .map(|c| confidence::FailedRun {
                label: c.label.clone(),
                error: c.error.clone(),
            })
            .collect();
        let dir_name = group_dir_name(&g.values);
        let dir = per_variation_aggregate_dir(base, &dir_name, order);
        let mut variation_values = Map::new();
        for (path, val) in &g.values {
            variation_values.insert(
                display.get(path).cloned().unwrap_or_else(|| leaf(path)),
                val.clone(),
            );
        }
        let extra = vec![
            (
                "variation_label".to_string(),
                Value::String(g.label.clone()),
            ),
            (
                "variation_values".to_string(),
                Value::Object(variation_values),
            ),
            ("sweep_mode".to_string(), Value::String(mode.to_string())),
        ];
        confidence::write_confidence_aggregate(
            &dir,
            g.members.len(),
            success.len(),
            &failed,
            &run_labels,
            confidence,
            cooldown,
            success.len() == 1,
            &metrics,
            &extra,
        )?;
    }

    let mut combos: Vec<ComboStats> = Vec::new();
    for g in &groups {
        let success: Vec<&Classified> = g.members.iter().copied().filter(|c| c.success).collect();
        if success.is_empty() {
            continue;
        }
        let summaries: Vec<&Value> = success.iter().filter_map(|c| c.summary.as_ref()).collect();
        let metrics = confidence::collect_confidence_metrics(&summaries, confidence);
        if metrics.is_empty() {
            continue;
        }
        let params: Vec<(String, Value)> = g
            .values
            .iter()
            .map(|(path, val)| {
                (
                    display.get(path).cloned().unwrap_or_else(|| leaf(path)),
                    val.clone(),
                )
            })
            .collect();
        combos.push(ComboStats { params, metrics });
    }

    let sweep_parameters = sweep_parameters_from_groups(&groups, &display);
    let num_combinations: i64 = sweep_parameters.iter().map(|p| p.1.len() as i64).product();
    let num_trials_per_value = groups.iter().map(|g| g.members.len()).max().unwrap_or(0);
    let total = classified.len();
    let num_successful = classified.iter().filter(|c| c.success).count();
    let failed = failed_runs(classified);

    let best = best_configurations(&combos);
    let pareto = pareto_optimal(&combos);

    let json = build_sweep_confidence_json(
        total,
        num_successful,
        &failed,
        &sweep_parameters,
        num_combinations,
        num_trials_per_value,
        mode,
        confidence,
        &combos,
        &best,
        &pareto,
    );
    let csv = build_sweep_confidence_csv(
        total,
        num_successful,
        &sweep_parameters,
        num_combinations,
        &combos,
        &best,
        &pareto,
    );

    let dir = sweep_aggregate_dir(base, order);
    std::fs::create_dir_all(&dir)?;
    std::fs::write(
        dir.join("profile_export_aiperf_sweep.json"),
        serde_json::to_string_pretty(&json)?,
    )?;
    std::fs::write(dir.join("profile_export_aiperf_sweep.csv"), csv)?;
    Ok(())
}

/// Build `[{name, values}]` with values in first-seen order.
fn sweep_parameters_from_groups(
    groups: &[VariationGroup],
    display: &std::collections::HashMap<String, String>,
) -> Vec<(String, Vec<Value>)> {
    let mut out: Vec<(String, Vec<Value>)> = Vec::new();
    for g in groups {
        for (path, value) in &g.values {
            let name = display.get(path).cloned().unwrap_or_else(|| leaf(path));
            match out.iter_mut().find(|(n, _)| n == &name) {
                Some((_, values)) => {
                    if !values.iter().any(|v| v == value) {
                        values.push(value.clone());
                    }
                }
                None => out.push((name, vec![value.clone()])),
            }
        }
    }
    out
}

/// A metric's `mean` for a combo, by flattened key.
fn combo_mean(combo: &ComboStats, key: &str) -> Option<f64> {
    combo
        .metrics
        .iter()
        .find(|(k, _)| k == key)
        .map(|(_, m)| m.mean)
}

/// A metric's `unit` for a combo, by flattened key.
fn combo_unit<'a>(combo: &'a ComboStats, key: &str) -> Option<&'a str> {
    combo
        .metrics
        .iter()
        .find(|(k, _)| k == key)
        .map(|(_, m)| m.unit.as_str())
}

const THROUGHPUT_KEY: &str = "request_throughput_avg";
const LATENCY_CANDIDATES: &[&str] = &["time_to_first_token_p99", "request_latency_p99"];

/// Resolve the preferred latency key present in every combination.
fn resolve_latency_key(combos: &[ComboStats]) -> Option<&'static str> {
    LATENCY_CANDIDATES
        .iter()
        .copied()
        .find(|k| combos.iter().all(|c| combo_mean(c, k).is_some()))
}

/// Build the `best_configurations` block.
fn best_configurations(combos: &[ComboStats]) -> Map<String, Value> {
    let mut best = Map::new();
    if combos.is_empty() {
        return best;
    }
    if combos
        .iter()
        .all(|c| combo_mean(c, THROUGHPUT_KEY).is_some())
    {
        let combo = combos
            .iter()
            .max_by(|a, b| {
                combo_mean(a, THROUGHPUT_KEY)
                    .unwrap()
                    .total_cmp(&combo_mean(b, THROUGHPUT_KEY).unwrap())
            })
            .unwrap();
        best.insert(
            "best_throughput".into(),
            serde_json::json!({
                "parameters": params_map(combo),
                "metric": f(combo_mean(combo, THROUGHPUT_KEY).unwrap()),
                "unit": combo_unit(combo, THROUGHPUT_KEY).unwrap_or("requests/sec"),
            }),
        );
    }
    if let Some(latency) = resolve_latency_key(combos) {
        let combo = combos
            .iter()
            .min_by(|a, b| {
                combo_mean(a, latency)
                    .unwrap()
                    .total_cmp(&combo_mean(b, latency).unwrap())
            })
            .unwrap();
        best.insert(
            "best_latency_p99".into(),
            serde_json::json!({
                "parameters": params_map(combo),
                "metric": f(combo_mean(combo, latency).unwrap()),
                "unit": combo_unit(combo, latency).unwrap_or("ms"),
            }),
        );
    }
    best
}

/// Compute the strict Pareto frontier by maximizing throughput and minimizing
/// latency, then sort by `(parameter, value)` pairs.
fn pareto_optimal(combos: &[ComboStats]) -> Vec<Value> {
    if combos.is_empty() {
        return Vec::new();
    }
    let Some(latency) = resolve_latency_key(combos) else {
        return Vec::new();
    };
    if !combos
        .iter()
        .all(|c| combo_mean(c, THROUGHPUT_KEY).is_some())
    {
        return Vec::new();
    }
    let points: Vec<(f64, f64)> = combos
        .iter()
        .map(|c| {
            (
                combo_mean(c, THROUGHPUT_KEY).unwrap(),
                combo_mean(c, latency).unwrap(),
            )
        })
        .collect();
    let dominates = |a: (f64, f64), b: (f64, f64)| -> bool {
        let be = (a.0 >= b.0) && (a.1 <= b.1);
        let strict = (a.0 > b.0) || (a.1 < b.1);
        be && strict
    };
    let mut optimal: Vec<usize> = Vec::new();
    for i in 0..combos.len() {
        let dominated = (0..combos.len())
            .filter(|&j| j != i)
            .any(|j| dominates(points[j], points[i]));
        if !dominated {
            optimal.push(i);
        }
    }
    let mut result: Vec<(Vec<(String, Value)>, Value)> = optimal
        .iter()
        .map(|&i| {
            let mut sorted = combos[i].params.clone();
            sorted.sort_by(|a, b| a.0.cmp(&b.0));
            (sorted, Value::Object(params_map(&combos[i])))
        })
        .collect();
    result.sort_by(|a, b| {
        let ka: Vec<String> = a.0.iter().map(|(k, v)| format!("{k}={v}")).collect();
        let kb: Vec<String> = b.0.iter().map(|(k, v)| format!("{k}={v}")).collect();
        ka.cmp(&kb)
    });
    result.into_iter().map(|(_, v)| v).collect()
}

/// A combo's `{display_name: value}` parameters map.
fn params_map(combo: &ComboStats) -> Map<String, Value> {
    let mut m = Map::new();
    for (name, value) in &combo.params {
        m.insert(name.clone(), value.clone());
    }
    m
}

/// Assemble the multi-trial `profile_export_aiperf_sweep.json` document.
#[allow(clippy::too_many_arguments)]
fn build_sweep_confidence_json(
    total: usize,
    num_successful: usize,
    failed: &[confidence::FailedRun],
    sweep_parameters: &[(String, Vec<Value>)],
    num_combinations: i64,
    num_trials_per_value: usize,
    mode: &str,
    confidence: f64,
    combos: &[ComboStats],
    best: &Map<String, Value>,
    pareto: &[Value],
) -> Value {
    let sweep_params_json: Vec<Value> = sweep_parameters
        .iter()
        .map(|(name, values)| serde_json::json!({"name": name, "values": values}))
        .collect();

    let mut metadata = Map::new();
    metadata.insert("sweep_parameters".into(), Value::Array(sweep_params_json));
    metadata.insert("num_combinations".into(), Value::from(num_combinations));
    metadata.insert("sweep_mode".into(), Value::String(mode.to_string()));
    metadata.insert("confidence_level".into(), f(confidence));
    metadata.insert(
        "num_trials_per_value".into(),
        Value::from(num_trials_per_value as u64),
    );
    metadata.insert("aggregation_type".into(), Value::String("sweep".into()));

    let per_combination: Vec<Value> = combos
        .iter()
        .map(|c| {
            let mut metrics = Map::new();
            for (key, metric) in &c.metrics {
                metrics.insert(key.clone(), metric_sweep_value(metric));
            }
            serde_json::json!({
                "parameters": params_map(c),
                "metrics": metrics,
            })
        })
        .collect();

    let failed_json: Vec<Value> = failed
        .iter()
        .map(|f| serde_json::json!({"label": f.label, "error": f.error}))
        .collect();

    let mut root = Map::new();
    root.insert("aggregation_type".into(), Value::String("sweep".into()));
    root.insert("num_profile_runs".into(), Value::from(total as u64));
    root.insert(
        "num_successful_runs".into(),
        Value::from(num_successful as u64),
    );
    root.insert("failed_runs".into(), Value::Array(failed_json));
    root.insert("metadata".into(), Value::Object(metadata));
    root.insert(
        "per_combination_metrics".into(),
        Value::Array(per_combination),
    );
    root.insert("best_configurations".into(), Value::Object(best.clone()));
    root.insert("pareto_optimal".into(), Value::Array(pareto.to_vec()));
    Value::Object(root)
}

/// The sweep per-combination projection of a confidence metric (drops se/t_critical).
fn metric_sweep_value(metric: &ConfidenceMetric) -> Value {
    let mut m = Map::new();
    m.insert("mean".into(), f(metric.mean));
    m.insert("std".into(), f(metric.std));
    m.insert("min".into(), f(metric.min));
    m.insert("max".into(), f(metric.max));
    m.insert("cv".into(), f(metric.cv));
    m.insert("ci_low".into(), f(metric.ci_low));
    m.insert("ci_high".into(), f(metric.ci_high));
    m.insert("unit".into(), Value::String(metric.unit.clone()));
    Value::Object(m)
}

/// Assemble the multi-trial `profile_export_aiperf_sweep.csv`.
fn build_sweep_confidence_csv(
    total: usize,
    num_successful: usize,
    sweep_parameters: &[(String, Vec<Value>)],
    num_combinations: i64,
    combos: &[ComboStats],
    best: &Map<String, Value>,
    pareto: &[Value],
) -> String {
    let param_names: Vec<String> = sweep_parameters.iter().map(|(n, _)| n.clone()).collect();
    let mut w = CsvWriter::new();

    if !combos.is_empty() {
        let metric_names: BTreeSet<String> = combos
            .iter()
            .flat_map(|c| c.metrics.iter().map(|(k, _)| k.clone()))
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
            let mut row: Vec<String> = param_names
                .iter()
                .map(|p| combo_param_value(combo, p))
                .collect();
            for m in &metric_names {
                match combo.metrics.iter().find(|(k, _)| k == m).map(|(_, v)| v) {
                    Some(metric) => {
                        row.push(fmt_finite(metric.mean, 2));
                        row.push(fmt_finite(metric.std, 2));
                        row.push(fmt_finite(metric.min, 2));
                        row.push(fmt_finite(metric.max, 2));
                        row.push(fmt_finite(metric.cv, 4));
                    }
                    None => row.extend(["", "", "", "", ""].map(String::from)),
                }
            }
            w.row(&row);
        }
    }

    w.row(&[] as &[String]);
    w.row(&["Best Configurations".to_string()]);
    if !best.is_empty() {
        let mut header = vec!["Configuration".to_string()];
        header.extend(param_names.iter().cloned());
        header.push("Metric".to_string());
        header.push("Unit".to_string());
        w.row(&header);
        for (name, data) in best {
            let formatted = name
                .split('_')
                .map(title_word)
                .collect::<Vec<_>>()
                .join(" ");
            let params = data.get("parameters").and_then(Value::as_object);
            let mut row = vec![formatted];
            for p in &param_names {
                row.push(
                    params
                        .and_then(|m| m.get(p))
                        .map(render_value)
                        .unwrap_or_default(),
                );
            }
            row.push(fmt_num(data.get("metric"), 2));
            row.push(
                data.get("unit")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string(),
            );
            w.row(&row);
        }
    }

    w.row(&[] as &[String]);
    w.row(&["Pareto Optimal Points".to_string()]);
    if !pareto.is_empty() {
        w.row(&param_names);
        for combo in pareto {
            let obj = combo.as_object();
            let row: Vec<String> = param_names
                .iter()
                .map(|p| {
                    obj.and_then(|m| m.get(p))
                        .map(render_value)
                        .unwrap_or_default()
                })
                .collect();
            w.row(&row);
        }
    } else {
        w.row(&["None".to_string()]);
    }

    w.row(&[] as &[String]);
    w.row(&["Metadata".to_string()]);
    w.row(&["Field".to_string(), "Value".to_string()]);
    w.row(&["Aggregation Type".to_string(), "sweep".to_string()]);
    w.row(&["Sweep Parameters".to_string(), param_names.join(", ")]);
    w.row(&[
        "Number of Combinations".to_string(),
        num_combinations.to_string(),
    ]);
    w.row(&["Number of Profile Runs".to_string(), total.to_string()]);
    w.row(&[
        "Number of Successful Runs".to_string(),
        num_successful.to_string(),
    ]);

    w.finish()
}

/// Title-case one word (`throughput` -> `Throughput`).
fn title_word(word: &str) -> String {
    let mut chars = word.chars();
    match chars.next() {
        Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
        None => String::new(),
    }
}

/// Render a combo parameter value for a CSV cell (int bare, missing empty).
fn combo_param_value(combo: &ComboStats, name: &str) -> String {
    combo
        .params
        .iter()
        .find(|(n, _)| n == name)
        .map(|(_, v)| render_value(v))
        .unwrap_or_default()
}

/// Render finite CSV values with fixed decimals and non-finite values as empty.
fn fmt_finite(v: f64, decimals: usize) -> String {
    if v.is_finite() {
        format!("{v:.decimals$}")
    } else {
        String::new()
    }
}

/// Write `profile_export_aiperf_collated.json` from each successful run's
/// per-request `profile_export.jsonl`. Only emitted for adaptive convergence.
fn write_detailed(dir: &Path, classified: &[Classified], cooldown: f64) -> anyhow::Result<()> {
    std::fs::create_dir_all(dir)?;
    let success: Vec<&Classified> = classified.iter().filter(|c| c.success).collect();

    let mut per_run_data: Vec<(String, Vec<(String, Vec<f64>)>)> = Vec::new();
    for c in &success {
        let run_metrics = load_jsonl_metrics(&c.artifact_dir);
        for (name, values) in run_metrics {
            match per_run_data.iter_mut().find(|(n, _)| n == &name) {
                Some((_, entries)) => entries.push((c.label.clone(), values)),
                None => per_run_data.push((name, vec![(c.label.clone(), values)])),
            }
        }
    }

    let mut metrics = Map::new();
    for (name, entries) in &per_run_data {
        let mut combined: Vec<f64> = Vec::new();
        for (_, vals) in entries {
            combined.extend(vals.iter().copied());
        }
        if combined.is_empty() {
            continue;
        }
        let per_run: Vec<Value> = entries
            .iter()
            .map(|(label, vals)| {
                let mean = if vals.is_empty() {
                    0.0
                } else {
                    vals.iter().sum::<f64>() / vals.len() as f64
                };
                serde_json::json!({
                    "label": label,
                    "mean": f(mean),
                    "count": vals.len(),
                })
            })
            .collect();
        let mut sorted = combined.clone();
        sorted.sort_by(f64::total_cmp);
        let n = sorted.len();
        let mean = combined.iter().sum::<f64>() / n as f64;
        let std = if n > 1 {
            (combined.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n as f64 - 1.0)).sqrt()
        } else {
            0.0
        };
        let combined_obj = serde_json::json!({
            "mean": f(mean),
            "std": f(std),
            "p50": f(percentile_linear(&sorted, 50.0)),
            "p90": f(percentile_linear(&sorted, 90.0)),
            "p95": f(percentile_linear(&sorted, 95.0)),
            "p99": f(percentile_linear(&sorted, 99.0)),
            "count": n,
        });
        metrics.insert(
            name.clone(),
            serde_json::json!({"combined": combined_obj, "per_run": per_run}),
        );
    }

    let failed = failed_runs(classified);
    let failed_json: Vec<Value> = failed
        .iter()
        .map(|fr| serde_json::json!({"label": fr.label, "error": fr.error}))
        .collect();
    let run_labels: Vec<Value> = success
        .iter()
        .map(|c| Value::String(c.label.clone()))
        .collect();

    let output = serde_json::json!({
        "schema_version": "1.0.0",
        "aiperf_version": crate::model::export::AIPERF_V1_VERSION,
        "description":
            "Collated per-request metrics across all runs. \
             Pools individual request-level values from every run into a single population \
             and computes combined percentiles (p50, p90, p95, p99). \
             Contrast with profile_export_aiperf_aggregate.json, which computes statistics \
             over run-level summary values.",
        "metadata": {
            "aggregation_type": "detailed",
            "num_profile_runs": classified.len(),
            "num_successful_runs": success.len(),
            "failed_runs": failed_json,
            "run_labels": run_labels,
            "cooldown_seconds": f(cooldown),
        },
        "metrics": metrics,
    });

    std::fs::write(
        dir.join("profile_export_aiperf_collated.json"),
        serde_json::to_string_pretty(&output)?,
    )?;
    Ok(())
}

/// Load profiling-phase, non-error metrics from `profile_export.jsonl`.
fn load_jsonl_metrics(dir: &Path) -> Vec<(String, Vec<f64>)> {
    let path = dir.join("profile_export.jsonl");
    let Ok(text) = std::fs::read_to_string(&path) else {
        return Vec::new();
    };
    let mut out: Vec<(String, Vec<f64>)> = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let Ok(record) = serde_json::from_str::<Value>(line) else {
            continue;
        };
        if record
            .get("metadata")
            .and_then(|m| m.get("benchmark_phase"))
            .and_then(Value::as_str)
            != Some("profiling")
        {
            continue;
        }
        if !record.get("error").map(Value::is_null).unwrap_or(true) {
            continue;
        }
        let Some(metrics) = record.get("metrics").and_then(Value::as_object) else {
            continue;
        };
        for (name, entry) in metrics {
            let Some(value) = entry.get("value").and_then(Value::as_f64) else {
                continue;
            };
            match out.iter_mut().find(|(n, _)| n == name) {
                Some((_, vals)) => vals.push(value),
                None => out.push((name.clone(), vec![value])),
            }
        }
    }
    out
}

/// Linear-interpolation percentile over a sorted slice.
fn percentile_linear(sorted: &[f64], p: f64) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return 0.0;
    }
    if n == 1 {
        return sorted[0];
    }
    let idx = p / 100.0 * (n - 1) as f64;
    let lo = idx.floor() as usize;
    let hi = idx.ceil() as usize;
    if lo == hi {
        sorted[lo]
    } else {
        sorted[lo] + (idx - lo as f64) * (sorted[hi] - sorted[lo])
    }
}
