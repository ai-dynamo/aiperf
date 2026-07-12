// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Aggregate report rendering: a console metrics table and a JSON export.

use std::path::Path;

use aiperf_metrics::{AccuracyAnalysis, AgenticEvaluationSummary, NativeReport};
use anyhow::{Context, Result};
use serde::Serialize;

use loadgen_core::collector::{TraceDistributionStats, TraceSimulationReport};

use crate::accuracy::AccuracyRunReport;
use crate::scheduled::ScheduledRunReport;

/// Width of the horizontal rule separating the latency table's header/body.
/// Historical fixed width (does not track the column format above); kept as a
/// single source of truth so the two rules stay in lock-step.
const TABLE_RULE_WIDTH: usize = 82;

fn metric_row(name: &str, d: &TraceDistributionStats) -> String {
    format!(
        "{name:<26} {:>10.3} {:>10.3} {:>10.3} {:>10.3} {:>10.3} {:>10.3}",
        d.mean_ms, d.min_ms, d.median_ms, d.p90_ms, d.p95_ms, d.p99_ms,
    )
}

/// Print the full metrics table to stdout.
pub fn print_report_table(report: &TraceSimulationReport) {
    let c = &report.request_counts;
    let t = &report.throughput;
    let l = &report.latency;

    // Requests that never completed are surfaced as errors.
    let errors = c.num_requests.saturating_sub(c.completed_requests);

    println!();
    println!(
        "{:<26} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "Latency (ms)", "avg", "min", "p50", "p90", "p95", "p99",
    );
    println!("{}", "-".repeat(TABLE_RULE_WIDTH));
    println!("{}", metric_row("Time to First Token", &l.ttft));
    println!("{}", metric_row("Time to Second Token", &l.ttst));
    println!("{}", metric_row("Inter Token Latency", &l.itl.distribution));
    println!("{}", metric_row("Time per Output Token", &l.tpot));
    println!("{}", metric_row("Request Latency (e2e)", &l.e2e));
    println!("{}", "-".repeat(TABLE_RULE_WIDTH));
    println!(
        "Requests        : {} total, {} completed, {} errors ({:.1}% error rate)",
        c.num_requests,
        c.completed_requests,
        errors,
        if c.num_requests > 0 {
            100.0 * errors as f64 / c.num_requests as f64
        } else {
            0.0
        },
    );
    println!(
        "Input tokens    : {} ({:.1} tok/s)",
        c.total_input_tokens, t.input_throughput_tok_s,
    );
    println!(
        "Output tokens   : {} ({:.1} tok/s)",
        c.total_output_tokens, t.output_throughput_tok_s,
    );
    println!(
        "Throughput      : {:.2} req/s, {:.1} tok/s total",
        t.request_throughput_rps, t.total_throughput_tok_s,
    );
    println!("Wall time       : {:.1} ms", t.wall_time_ms);
}

/// Write the aggregate report as pretty JSON to `path`.
pub fn write_report_json(report: &TraceSimulationReport, path: impl AsRef<Path>) -> Result<()> {
    write_json(report, path)
}

/// Write the unified native-v2 report as pretty JSON to `path`.
pub fn write_native_report_json(report: &NativeReport, path: impl AsRef<Path>) -> Result<()> {
    write_json(report, path)
}

/// Write aggregate metrics plus per-turn expected/observed schedule timing.
pub fn write_scheduled_report_json(
    report: &ScheduledRunReport,
    path: impl AsRef<Path>,
) -> Result<()> {
    write_json(report, path)
}

/// Print overall and per-task accuracy results.
pub fn print_accuracy_table(analysis: &AccuracyAnalysis) {
    println!();
    println!(
        "{:<32} {:>9} {:>9} {:>10} {:>12} {:>23}",
        "Accuracy task", "correct", "total", "unparsed", "accuracy", "95% CI"
    );
    println!("{}", "-".repeat(101));
    for (task, rollup) in &analysis.summary.per_task {
        println!(
            "{:<32} {:>9} {:>9} {:>10} {:>11.2}% {:>10}",
            task.as_str(),
            rollup.correct_count,
            rollup.n,
            rollup.unparsed_count,
            rollup.accuracy.unwrap_or(0.0) * 100.0,
            format_confidence_interval(rollup.ci),
        );
    }
    println!("{}", "-".repeat(101));
    let overall = &analysis.summary.overall;
    println!(
        "{:<32} {:>9} {:>9} {:>10} {:>11.2}% {:>10}",
        "OVERALL",
        overall.correct_count,
        overall.n,
        overall.unparsed_count,
        overall.accuracy.unwrap_or(0.0) * 100.0,
        format_confidence_interval(overall.ci),
    );
    if overall.n > 0 && overall.unparsed_count == overall.n {
        eprintln!(
            "warning: every accuracy response was unparsed; verify the target returns valid completions before trusting this score"
        );
    }
}

/// Print canonical agentic verifier rewards and explicit terminal classes.
pub fn print_agentic_table(summary: &AgenticEvaluationSummary) {
    println!();
    println!(
        "Agentic episodes : {} total, {} verified, {} infrastructure errors, {} cancelled",
        summary.episode_count,
        summary.completed_count,
        summary.infrastructure_error_count,
        summary.cancelled_count,
    );
    println!(
        "Model calls      : {} total, {} agent, {} environment, {} verifier",
        summary.model_calls,
        summary.primary_model_calls,
        summary.environment_model_calls,
        summary.verifier_model_calls,
    );
    println!();
    println!(
        "{:<40} {:>9} {:>14} {:>14} {:>14}",
        "Canonical verifier reward", "n", "avg", "min", "max"
    );
    println!("{}", "-".repeat(96));
    for (name, reward) in &summary.rewards {
        println!(
            "{:<40} {:>9} {:>14.6} {:>14.6} {:>14.6}",
            name, reward.n, reward.avg, reward.min, reward.max,
        );
    }
    if summary.rewards.is_empty() {
        println!("{:<40} {:>9}", "(no completed verifier results)", 0);
    }
    if let (Some(name), Some(score)) = (&summary.primary_reward, summary.primary_score) {
        println!("{}", "-".repeat(96));
        println!("Primary reward  : {name} = {score:.6}");
    }
}

/// Write the stable per-task accuracy summary CSV.
///
/// Column order and four-decimal accuracy formatting preserve the inherited
/// exporter contract from `src/aiperf/accuracy/accuracy_data_exporter.py:53-108`.
pub fn write_accuracy_summary_csv(
    analysis: &AccuracyAnalysis,
    path: impl AsRef<Path>,
) -> Result<()> {
    let path = path.as_ref();
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating accuracy CSV directory {}", parent.display()))?;
    }
    let mut writer = csv::Writer::from_path(path)
        .with_context(|| format!("opening accuracy CSV {}", path.display()))?;
    writer.write_record(["task", "correct", "total", "unparsed", "accuracy"])?;
    for (task, rollup) in &analysis.summary.per_task {
        writer.write_record(accuracy_csv_row(task.as_str(), rollup))?;
    }
    writer.write_record(accuracy_csv_row("OVERALL", &analysis.summary.overall))?;
    writer
        .flush()
        .with_context(|| format!("writing accuracy CSV {}", path.display()))?;
    Ok(())
}

fn accuracy_csv_row(task: &str, rollup: &aiperf_metrics::AccuracyRollup) -> [String; 5] {
    [
        task.to_string(),
        rollup.correct_count.to_string(),
        rollup.n.to_string(),
        rollup.unparsed_count.to_string(),
        rollup
            .accuracy
            .map(|value| format!("{value:.4}"))
            .unwrap_or_default(),
    ]
}

fn format_confidence_interval(interval: Option<aiperf_metrics::ConfidenceInterval>) -> String {
    interval
        .map(|interval| {
            format!(
                "[{:.2}%, {:.2}%]",
                interval.low * 100.0,
                interval.high * 100.0
            )
        })
        .unwrap_or_else(|| "N/A".to_string())
}

/// Write a combined performance/accuracy report as pretty JSON.
pub fn write_accuracy_report_json(
    report: &AccuracyRunReport,
    path: impl AsRef<Path>,
) -> Result<()> {
    write_json(report, path)
}

fn write_json(value: &impl Serialize, path: impl AsRef<Path>) -> Result<()> {
    let path = path.as_ref();
    let json = serde_json::to_string_pretty(value).context("serializing summary report")?;
    std::fs::write(path, json)
        .with_context(|| format!("writing summary report {}", path.display()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn report_json_roundtrips_to_object() {
        // A default report serializes to a JSON object with the expected keys.
        let report = loadgen_core::collector::TraceCollector::default()
            .finish()
            .with_wall_time_ms(1.0);
        let path = std::env::temp_dir().join(format!("aiperf_sum_{}.json", std::process::id()));
        write_report_json(&report, &path).unwrap();
        let value: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        assert!(value.is_object());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn native_report_json_uses_the_metrics_first_v2_shape() {
        let mut summary = aiperf_metrics::AccumulatorSummary::new();
        summary.insert_finite(aiperf_metrics::MetricTag::RequestCount, 1.0);
        let report = NativeReport::new(&summary, None);
        let path =
            std::env::temp_dir().join(format!("aiperf_native_sum_{}.json", std::process::id()));
        write_native_report_json(&report, &path).unwrap();
        let value: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(value["schema_version"], "2.0");
        assert_eq!(value["metrics"]["request_count"]["type"], "counter");
        let _ = std::fs::remove_file(&path);
    }
}
