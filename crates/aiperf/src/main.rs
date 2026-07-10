// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `aiperf`: real-HTTP benchmarking CLI.
//!
//! Two modes:
//!
//! ```text
//! # online (default): closed-loop concurrency benchmark
//! aiperf [BASE_URL] [MODEL] --concurrency N --requests N --isl N --osl N
//!
//! # graph: Graph-IR E2E streaming throughput (multi-turn DAG conversations)
//! aiperf --mode graph [BASE_URL] [MODEL] \
//!   --turns N --instances N --workers N --concurrency N --osl N \
//!   [--request-concurrency N] [--http2]
//! ```

use aiperf::report::print_report_table;
use aiperf::run::run;
use aiperf::workload::SkeletonWorkload;

fn main() -> anyhow::Result<()> {
    let argv: Vec<String> = std::env::args().skip(1).collect();

    let mut mode = "online".to_string();
    if let Some(i) = argv.iter().position(|a| a == "--mode")
        && let Some(v) = argv.get(i + 1)
    {
        mode = v.clone();
    }

    match mode.as_str() {
        "graph" => run_graph_mode(&argv),
        _ => run_online_mode(&argv),
    }
}

/// Positional args are the non-flag, non-flag-value tokens (base_url, model).
fn positionals(argv: &[String], value_flags: &[&str]) -> Vec<String> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < argv.len() {
        let a = &argv[i];
        if a.starts_with("--") {
            if value_flags.contains(&a.as_str()) {
                i += 2;
            } else {
                i += 1;
            }
            continue;
        }
        out.push(a.clone());
        i += 1;
    }
    out
}

fn flag_val<T: std::str::FromStr>(argv: &[String], name: &str) -> Option<T> {
    argv.iter()
        .position(|a| a == name)
        .and_then(|i| argv.get(i + 1))
        .and_then(|v| v.parse().ok())
}

fn run_online_mode(argv: &[String]) -> anyhow::Result<()> {
    let value_flags = ["--mode", "--concurrency", "--requests", "--isl", "--osl"];
    let pos = positionals(argv, &value_flags);
    let base_url = pos
        .first()
        .cloned()
        .unwrap_or_else(|| "http://localhost:8000".to_string());
    let model = pos.get(1).cloned().unwrap_or_else(|| "model".to_string());

    let concurrency = flag_val(argv, "--concurrency").unwrap_or(16usize);
    let num_requests = flag_val(argv, "--requests").unwrap_or(100usize);
    let isl = flag_val(argv, "--isl").unwrap_or(128usize);
    let osl = flag_val(argv, "--osl").unwrap_or(128usize);

    let workload = SkeletonWorkload {
        num_requests,
        input_tokens: isl,
        output_tokens: osl,
    };
    let rt = tokio::runtime::Runtime::new()?;
    let report = rt.block_on(run(base_url, model, workload, concurrency))?;
    print_report_table(&report, 0);
    Ok(())
}

fn run_graph_mode(argv: &[String]) -> anyhow::Result<()> {
    use aiperf_graph::bench::{BenchConfig, run_bench};

    let value_flags = [
        "--mode",
        "--turns",
        "--instances",
        "--workers",
        "--concurrency",
        "--osl",
        "--request-concurrency",
        "--prefill-concurrency",
    ];
    let pos = positionals(argv, &value_flags);
    let base_url = pos
        .first()
        .cloned()
        .unwrap_or_else(|| "http://127.0.0.1:8000".to_string());
    let model = pos.get(1).cloned().unwrap_or_else(|| "model".to_string());

    let cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(8);
    let turns: usize = flag_val(argv, "--turns").unwrap_or(4);
    let workers: usize = flag_val(argv, "--workers").unwrap_or(cores);
    let concurrency: usize = flag_val(argv, "--concurrency").unwrap_or(64);
    let max_tokens: usize = flag_val(argv, "--osl").unwrap_or(1);
    let instances: usize = flag_val(argv, "--instances").unwrap_or(400_000);
    let request_concurrency: Option<usize> = flag_val(argv, "--request-concurrency");
    let prefill_concurrency: Option<usize> = flag_val(argv, "--prefill-concurrency");
    if argv.iter().any(|a| a == "--http2") {
        // build_client() opts into h2c prior-knowledge when GRAPH_HTTP2 is set.
        // SAFETY: single-threaded startup, before any worker thread spawns.
        unsafe { std::env::set_var("GRAPH_HTTP2", "1") };
    }

    let cfg = BenchConfig {
        base_urls: vec![base_url.clone()],
        model,
        turns,
        instances,
        workers,
        concurrency,
        max_tokens,
        request_concurrency,
        prefill_concurrency,
    };

    eprintln!(
        "aiperf --mode graph: base={base_url} turns={turns} instances={instances} \
         workers={workers} concurrency={concurrency} osl={max_tokens} \
         offered_concurrency={} http2={}",
        workers * concurrency,
        argv.iter().any(|a| a == "--http2"),
    );

    let (report, secs) = run_bench(cfg);
    let total_requests = instances.saturating_mul(turns);
    let rps = total_requests as f64 / secs;

    println!("\n=== aiperf --mode graph (Graph-IR E2E, streaming SSE) ===");
    println!("requests   : {total_requests} (instances={instances} x turns={turns})");
    println!("wall        : {secs:.3} s");
    println!("RPS         : {rps:.0} req/s");
    println!("TTFT p50    : {:.3} ms", report.latency.ttft.median_ms);
    println!("TTFT p90    : {:.3} ms", report.latency.ttft.p90_ms);
    println!("TTFT p99    : {:.3} ms", report.latency.ttft.p99_ms);
    println!("TTFT mean   : {:.3} ms", report.latency.ttft.mean_ms);
    if rps >= 300_000.0 {
        println!("\nPROVEN: aiperf --mode graph >= 300k req/s ({rps:.0})");
    } else {
        println!("\nbelow target: {rps:.0} < 300000");
    }
    Ok(())
}
