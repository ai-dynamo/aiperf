// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Side-by-side result comparison for `aiperf compare`.
//!
//! Both inputs are AIPerf summary exports (`profile_export_aiperf.json`). Each
//! top-level metric tag holds a flat stat object (`unit`, `avg`, percentiles,
//! `min`, `max`, `std`, `count`, `sum`); this command reads one stat per metric
//! from both files and reports the two values, their absolute delta, and the
//! percent change. A better/worse verdict comes from the metric's preferred
//! direction in the runtime catalog, so no direction is hard-coded here.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde_json::Value;

use aiperf_runtime::metrics_core::catalog::plot_direction_for;
use aiperf_runtime::metrics_core::{CATALOG, PlotMetricDirection};

/// Stat compared by default. `avg` is the scalar summary present for every metric
/// shape (scalar and counter metrics carry their single value under `avg`).
const DEFAULT_STAT: &str = "avg";

/// Run `aiperf compare <file_a> <file_b> [--stat <avg|min|max|sum|std|p50|...>]`.
///
/// The two files are AIPerf summary exports; the named stat is diffed for every
/// metric present in both. A metric absent from one file is skipped.
pub fn run(args: &[String]) -> anyhow::Result<i32> {
    let mut files: Vec<PathBuf> = Vec::new();
    let mut stat = DEFAULT_STAT.to_owned();
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--stat" => {
                stat = it
                    .next()
                    .ok_or_else(|| anyhow::anyhow!("--stat needs a value"))?
                    .clone();
            }
            other if other.starts_with('-') => {
                anyhow::bail!("unknown compare flag {other:?}");
            }
            other => files.push(PathBuf::from(other)),
        }
    }

    if files.len() != 2 {
        anyhow::bail!(
            "compare requires exactly two result files, got {}",
            files.len()
        );
    }

    for file in &files {
        if !file.exists() {
            println!("Error: Result file not found: {}", file.display());
            return Ok(0);
        }
    }

    let summary_a = read_summary(&files[0])?;
    let summary_b = read_summary(&files[1])?;
    let metrics_a = collect_stats(&summary_a, &stat);
    let metrics_b = collect_stats(&summary_b, &stat);
    let direction = direction_by_tag();

    // Shared metrics only, in catalog-agnostic alphabetical order.
    let mut rows: Vec<Comparison> = Vec::new();
    for (tag, left) in &metrics_a {
        let Some(right) = metrics_b.get(tag) else {
            continue;
        };
        rows.push(Comparison::new(
            tag.clone(),
            left.unit.clone(),
            left.value,
            right.value,
            direction.get(tag.as_str()).copied(),
        ));
    }

    if rows.is_empty() {
        anyhow::bail!(
            "no metrics are shared between {} and {} for stat {stat:?}",
            files[0].display(),
            files[1].display()
        );
    }

    print_report(&files[0], &files[1], &stat, &rows);
    Ok(0)
}

/// A metric's unit and the selected stat value from one summary.
struct Stat {
    unit: String,
    value: f64,
}

/// Read a summary export and return its parsed root object. When the file holds
/// several concatenated records (a JSONL stream), the last record wins, matching
/// the convention that the final record is the finalized run.
fn read_summary(path: &Path) -> anyhow::Result<Value> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| anyhow::anyhow!("failed to read {}: {e}", path.display()))?;
    let mut last: Option<Value> = None;
    for record in serde_json::Deserializer::from_str(&text).into_iter::<Value>() {
        last = Some(
            record.map_err(|e| anyhow::anyhow!("bad result JSON in {}: {e}", path.display()))?,
        );
    }
    last.ok_or_else(|| anyhow::anyhow!("{} contained no JSON record", path.display()))
}

/// Pull the requested stat for every metric tag in a summary. A top-level key is
/// a metric when its value is an object with a string `unit` and a finite number
/// under `stat`; envelope keys (`schema_version`, `error_summary`, …) lack that
/// shape and drop out.
fn collect_stats(summary: &Value, stat: &str) -> BTreeMap<String, Stat> {
    let mut out = BTreeMap::new();
    let Some(object) = summary.as_object() else {
        return out;
    };
    for (tag, value) in object {
        let Some(entry) = value.as_object() else {
            continue;
        };
        let Some(unit) = entry.get("unit").and_then(Value::as_str) else {
            continue;
        };
        let Some(number) = entry
            .get(stat)
            .and_then(Value::as_f64)
            .filter(|v| v.is_finite())
        else {
            continue;
        };
        out.insert(
            tag.clone(),
            Stat {
                unit: unit.to_owned(),
                value: number,
            },
        );
    }
    out
}

/// Preferred-direction lookup keyed by the metric's summary spelling.
fn direction_by_tag() -> BTreeMap<&'static str, PlotMetricDirection> {
    CATALOG
        .iter()
        .map(|spec| (spec.tag.as_str(), plot_direction_for(&spec.def)))
        .collect()
}

/// One metric's diff between the two summaries.
struct Comparison {
    tag: String,
    unit: String,
    left: f64,
    right: f64,
    delta: f64,
    percent: Option<f64>,
    verdict: Verdict,
}

impl Comparison {
    /// Build a comparison, deriving the percent change (undefined when the left
    /// value is zero) and the verdict from the metric's preferred direction.
    fn new(
        tag: String,
        unit: String,
        left: f64,
        right: f64,
        direction: Option<PlotMetricDirection>,
    ) -> Self {
        let delta = right - left;
        let percent = (left != 0.0).then(|| delta / left.abs() * 100.0);
        let verdict = Verdict::classify(delta, direction);
        Comparison {
            tag,
            unit,
            left,
            right,
            delta,
            percent,
            verdict,
        }
    }
}

/// Direction-aware outcome of a single metric change.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Verdict {
    Better,
    Worse,
    Same,
    /// No preferred direction, or the metric is not in the catalog.
    Neutral,
}

impl Verdict {
    /// Classify a delta against the metric's preferred direction.
    fn classify(delta: f64, direction: Option<PlotMetricDirection>) -> Self {
        if delta == 0.0 {
            return Verdict::Same;
        }
        match direction {
            Some(PlotMetricDirection::LargerIsBetter) => {
                if delta > 0.0 {
                    Verdict::Better
                } else {
                    Verdict::Worse
                }
            }
            Some(PlotMetricDirection::SmallerIsBetter) => {
                if delta < 0.0 {
                    Verdict::Better
                } else {
                    Verdict::Worse
                }
            }
            Some(PlotMetricDirection::Neutral) | None => Verdict::Neutral,
        }
    }

    /// Short label for the verdict column.
    fn label(self) -> &'static str {
        match self {
            Verdict::Better => "better",
            Verdict::Worse => "worse",
            Verdict::Same => "same",
            Verdict::Neutral => "",
        }
    }
}

/// Print the aligned comparison table to stdout, mirroring the column layout of
/// the other AIPerf reporting subcommands (left-justified labels, right-justified
/// numbers, two-space gutters).
fn print_report(file_a: &Path, file_b: &Path, stat: &str, rows: &[Comparison]) {
    let header = [
        "Metric".to_owned(),
        "Unit".to_owned(),
        format!("A ({stat})"),
        format!("B ({stat})"),
        "Delta".to_owned(),
        "Change".to_owned(),
        "Verdict".to_owned(),
    ];

    let mut body: Vec<[String; 7]> = Vec::with_capacity(rows.len());
    for row in rows {
        body.push([
            row.tag.clone(),
            row.unit.clone(),
            fmt_value(row.left),
            fmt_value(row.right),
            fmt_delta(row.delta),
            match row.percent {
                Some(p) => format!("{p:+.2}%"),
                None => "-".to_owned(),
            },
            row.verdict.label().to_owned(),
        ]);
    }

    let mut widths = [0usize; 7];
    for (i, cell) in header.iter().enumerate() {
        widths[i] = cell.len();
    }
    for row in &body {
        for (i, cell) in row.iter().enumerate() {
            widths[i] = widths[i].max(cell.len());
        }
    }

    // Column 0 and 1 are text (left), the rest are numeric/label (right).
    let render = |cells: &[String; 7]| -> String {
        cells
            .iter()
            .enumerate()
            .map(|(i, cell)| {
                let w = widths[i];
                if i <= 1 {
                    format!("{cell:<w$}")
                } else {
                    format!("{cell:>w$}")
                }
            })
            .collect::<Vec<_>>()
            .join("  ")
    };

    println!("\nComparison ({stat})");
    println!("A: {}", file_a.display());
    println!("B: {}", file_b.display());
    println!();
    println!("{}", render(&header));
    let total: usize = widths.iter().sum::<usize>() + 2 * (widths.len() - 1);
    println!("{}", "-".repeat(total));
    for row in &body {
        println!("{}", render(row));
    }
    println!();
}

/// Format a stat value: integral values without a fraction, others to four
/// decimals.
fn fmt_value(value: f64) -> String {
    if value == value.trunc() && value.abs() < 1e15 {
        format!("{value:.0}")
    } else {
        format!("{value:.4}")
    }
}

/// Format a delta, always signed so the direction reads at a glance.
fn fmt_delta(value: f64) -> String {
    if value == value.trunc() && value.abs() < 1e15 {
        format!("{value:+.0}")
    } else {
        format!("{value:+.4}")
    }
}

#[cfg(test)]
mod tests;
