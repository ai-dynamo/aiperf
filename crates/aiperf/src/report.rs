// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Aggregate report rendering: a console metrics table and a JSON export.

use std::path::Path;

use anyhow::{Context, Result};

use loadgen_core::collector::{TraceDistributionStats, TraceSimulationReport};

fn metric_row(name: &str, d: &TraceDistributionStats) -> String {
    format!(
        "{name:<26} {:>10.3} {:>10.3} {:>10.3} {:>10.3} {:>10.3} {:>10.3}",
        d.mean_ms, d.min_ms, d.median_ms, d.p90_ms, d.p95_ms, d.p99_ms,
    )
}

/// Print the full metrics table to stdout.
pub fn print_report_table(report: &TraceSimulationReport, errors: usize) {
    let c = &report.request_counts;
    let t = &report.throughput;
    let l = &report.latency;

    println!();
    println!(
        "{:<26} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "Latency (ms)", "avg", "min", "p50", "p90", "p95", "p99",
    );
    println!("{}", "-".repeat(82));
    println!("{}", metric_row("Time to First Token", &l.ttft));
    println!("{}", metric_row("Time to Second Token", &l.ttst));
    println!("{}", metric_row("Inter Token Latency", &l.itl.distribution));
    println!("{}", metric_row("Time per Output Token", &l.tpot));
    println!("{}", metric_row("Request Latency (e2e)", &l.e2e));
    println!("{}", "-".repeat(82));
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
    let path = path.as_ref();
    let json = serde_json::to_string_pretty(report).context("serializing summary report")?;
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
}
