// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Sweep aggregation + terminal table.
//!
//! After every cell runs, read each cell's `native-v2.json`, extract the headline
//! metrics, render the live sweep table (gated on a TTY + >1 variation, mirroring
//! Python's `_should_emit_sweep_table`), and write the `sweep_aggregate` artifacts
//! grouped by `(label, values)` (never values-only). Ports the shape of
//! `cli_runner/_sweep_aggregate.py` + `_sweep_table.py`.

use std::io::IsTerminal;
use std::path::{Path, PathBuf};

use crate::flags::ProfileFlags;

/// One completed sweep cell.
pub struct CellOutcome {
    /// Variation label (`"path=value, ..."`).
    pub label: String,
    /// The stamped `variation` object (index/label/values).
    pub values: Option<serde_json::Value>,
    /// The cell's artifact directory.
    pub artifact_dir: PathBuf,
    /// Path to the cell's `native-v2.json` (present on success).
    pub report_path: Option<String>,
    /// Whether the cell committed a report.
    pub success: bool,
}

/// Headline metrics shown in the sweep table (tag, stat, column header).
const HEADLINE: &[(&str, &str, &str)] = &[
    ("output_token_throughput", "avg", "out_tok/s"),
    ("time_to_first_token", "p99", "TTFT p99 (ms)"),
    ("inter_token_latency", "p99", "ITL p99 (ms)"),
    ("request_latency", "p95", "req p95 (ms)"),
];

/// Aggregate the finished cells: render the table and write `sweep_aggregate`.
pub fn finish(flags: &ProfileFlags, outcomes: &[CellOutcome]) -> anyhow::Result<()> {
    // Per-cell headline metric rows.
    let rows: Vec<CellRow> = outcomes.iter().map(row_for).collect();

    // The base dir is the common parent of the per-cell dirs.
    let base = outcomes
        .first()
        .and_then(|o| o.artifact_dir.parent())
        .map(Path::to_path_buf)
        .or_else(|| flags.artifact_dir.clone())
        .unwrap_or_else(|| PathBuf::from("artifacts"));

    if !flags.no_sweep_table && std::io::stdout().is_terminal() && rows.len() > 1 {
        print_table(&rows);
    }
    write_aggregate(&base, &rows)?;
    Ok(())
}

/// A rendered table row: the variation label + extracted headline values.
struct CellRow {
    label: String,
    values: Option<serde_json::Value>,
    metrics: Vec<Option<f64>>,
    success: bool,
}

fn row_for(o: &CellOutcome) -> CellRow {
    let report = o
        .report_path
        .as_ref()
        .and_then(|p| std::fs::read(p).ok())
        .and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).ok());
    let metrics = HEADLINE
        .iter()
        .map(|(tag, stat, _)| report.as_ref().and_then(|r| headline_value(r, tag, stat)))
        .collect();
    CellRow {
        label: o.label.clone(),
        values: o.values.clone(),
        metrics,
        success: o.success,
    }
}

/// Extract one metric's stat from a `native-v2.json` report (scalar `value`, or a
/// distribution `avg` / `percentiles.pNN`).
fn headline_value(report: &serde_json::Value, tag: &str, stat: &str) -> Option<f64> {
    let stats = report
        .get("metrics")?
        .get(tag)?
        .get("series")?
        .as_array()?
        .first()?
        .get("stats")?;
    if let Some(v) = stats.get("value").and_then(serde_json::Value::as_f64) {
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
            println!("{}", widths.iter().map(|w| "-".repeat(*w)).collect::<Vec<_>>().join("  "));
        }
    }
    println!();
}

/// Write the `sweep_aggregate/profile_export_aiperf_sweep.json` artifact.
fn write_aggregate(base: &Path, rows: &[CellRow]) -> anyhow::Result<()> {
    let dir = base.join("sweep_aggregate");
    std::fs::create_dir_all(&dir)?;
    let cells: Vec<serde_json::Value> = rows
        .iter()
        .map(|r| {
            let metrics: serde_json::Map<String, serde_json::Value> = HEADLINE
                .iter()
                .zip(&r.metrics)
                .map(|((tag, stat, _), v)| {
                    (
                        format!("{tag}:{stat}"),
                        v.map(|x| serde_json::json!(x)).unwrap_or(serde_json::Value::Null),
                    )
                })
                .collect();
            serde_json::json!({
                "label": r.label,
                "variation": r.values,
                "success": r.success,
                "metrics": metrics,
            })
        })
        .collect();
    let doc = serde_json::json!({ "schema": "aiperf-sweep-1", "cells": cells });
    std::fs::write(
        dir.join("profile_export_aiperf_sweep.json"),
        serde_json::to_vec_pretty(&doc)?,
    )?;
    Ok(())
}
