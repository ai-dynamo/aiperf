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

// A high-churn benchmark allocator: the graph executor + streaming client
// allocate heavily per request, and glibc malloc/free was the top profiled
// hotspot. mimalloc cuts that churn substantially.
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

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
    use aiperf_graph::transport_bench::run_transport_bench;

    let value_flags = [
        "--mode",
        "--turns",
        "--instances",
        "--workers",
        "--concurrency",
        "--osl",
        "--conns",
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
    // Backend: aiperf-transport (default) or reqwest (--reqwest). For the
    // transport, h2c prior-knowledge is the default; --http1 forces HTTP/1.1.
    let use_reqwest = argv.iter().any(|a| a == "--reqwest");
    // Transport default is HTTP/1.1 keep-alive: for serial per-lane requests it
    // outperforms h2c (no per-stream hpack/flow-control overhead). --http2 opts
    // into h2c prior-knowledge (multiplexed pool).
    let http2 = argv.iter().any(|a| a == "--http2");
    let conns: usize = flag_val(argv, "--conns").unwrap_or(8);
    if use_reqwest && argv.iter().any(|a| a == "--http2") {
        // reqwest build_client() opts into h2c prior-knowledge via GRAPH_HTTP2.
        // SAFETY: single-threaded startup, before any worker thread spawns.
        unsafe { std::env::set_var("GRAPH_HTTP2", "1") };
    }

    let cfg = BenchConfig {
        base_urls: base_url.split(',').map(|s| s.trim().to_string()).collect(),
        model,
        turns,
        instances,
        workers,
        concurrency,
        max_tokens,
        request_concurrency,
        prefill_concurrency,
    };

    let backend = if use_reqwest {
        "reqwest"
    } else {
        "aiperf-transport"
    };
    eprintln!(
        "aiperf --mode graph: backend={backend} base={base_url} turns={turns} \
         instances={instances} workers={workers} concurrency={concurrency} osl={max_tokens} \
         conns/worker={conns} offered_concurrency={} http2={http2}",
        workers * concurrency,
    );

    let total_requests = instances.saturating_mul(turns);
    let (rps, p50, p90, p99, mean, secs, extra) = if use_reqwest {
        let (report, secs) = run_bench(cfg);
        let rps = total_requests as f64 / secs;
        (
            rps,
            report.latency.ttft.median_ms,
            report.latency.ttft.p90_ms,
            report.latency.ttft.p99_ms,
            report.latency.ttft.mean_ms,
            secs,
            format!("requests={total_requests}"),
        )
    } else {
        let r = run_transport_bench(cfg, http2, conns);
        (
            r.rps(),
            r.ttft_p50_ms,
            r.ttft_p90_ms,
            r.ttft_p99_ms,
            r.ttft_mean_ms,
            r.wall_secs,
            format!("completed={} errors={}", r.completed, r.errors),
        )
    };

    println!("\n=== aiperf --mode graph (Graph-IR E2E, streaming SSE, backend={backend}) ===");
    println!("{extra}");
    println!("wall        : {secs:.3} s");
    println!("RPS         : {rps:.0} req/s");
    println!("TTFT p50    : {p50:.3} ms");
    println!("TTFT p90    : {p90:.3} ms");
    println!("TTFT p99    : {p99:.3} ms");
    println!("TTFT mean   : {mean:.3} ms");
    if rps >= 500_000.0 {
        println!("\nPROVEN: aiperf --mode graph >= 500k req/s ({rps:.0}, backend={backend})");
    } else if rps >= 300_000.0 {
        println!("\n>= 300k: {rps:.0} (target 500k)");
    } else {
        println!("\nbelow target: {rps:.0} < 500000");
    }
    Ok(())
}
