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

use std::path::PathBuf;

use aiperf::report::{print_report_table, write_report_json};
use aiperf::run::run;
use aiperf::workload::SkeletonWorkload;
use clap::Parser;

// A high-churn benchmark allocator: the graph executor + streaming client
// allocate heavily per request, and glibc malloc/free was the top profiled
// hotspot. mimalloc cuts that churn substantially.
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

/// Default number of conversation instances for `--mode graph`.
const DEFAULT_INSTANCES: usize = 400_000;
/// RPS thresholds for the graph-mode summary verdict lines.
const RPS_1M: f64 = 1_000_000.0;
const RPS_500K: f64 = 500_000.0;
const RPS_300K: f64 = 300_000.0;

/// Command-line arguments for `aiperf`.
///
/// A single top-level struct models both modes: `--mode` selects online
/// (default) or graph, and the field set is the union of the two modes' flags.
/// Numeric flags whose default differs between modes are `Option`s so the
/// per-mode default can be applied in code (matching the legacy parser exactly).
#[derive(Parser, Debug)]
#[command(disable_help_flag = true)]
struct Cli {
    /// Benchmark mode: `online` (default, closed-loop concurrency) or `graph`.
    #[arg(long, default_value = "online")]
    mode: String,

    /// Positional `[BASE_URL]` (default differs per mode).
    base_url: Option<String>,
    /// Positional `[MODEL]` (default `model`).
    model: Option<String>,

    // --- flags shared between modes (defaults differ, hence Option) ---
    /// Offered concurrency (online default 16, graph default 64).
    #[arg(long)]
    concurrency: Option<usize>,
    /// Output sequence length / max tokens (online default 128, graph default 1).
    #[arg(long)]
    osl: Option<usize>,

    // --- online-only flags ---
    /// Number of requests (online default 100).
    #[arg(long)]
    requests: Option<usize>,
    /// Input sequence length (online default 128).
    #[arg(long)]
    isl: Option<usize>,
    /// Write the aggregate report as JSON to this path (online mode).
    #[arg(long)]
    json: Option<PathBuf>,

    // --- graph-only flags ---
    /// Conversation turns per instance (graph default 4).
    #[arg(long)]
    turns: Option<usize>,
    /// Conversation instances (graph default 400000).
    #[arg(long)]
    instances: Option<usize>,
    /// Worker threads (graph default: available cores).
    #[arg(long)]
    workers: Option<usize>,
    /// Connections per worker (graph default 8).
    #[arg(long)]
    conns: Option<usize>,
    /// Optional per-request concurrency override (graph).
    #[arg(long)]
    request_concurrency: Option<usize>,
    /// Optional prefill concurrency override (graph).
    #[arg(long)]
    prefill_concurrency: Option<usize>,
    /// Force HTTP/1.1 (graph; accepted for compatibility).
    #[arg(long)]
    http1: bool,
    /// Opt into h2c prior-knowledge (graph).
    #[arg(long)]
    http2: bool,
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match cli.mode.as_str() {
        "graph" => run_graph_mode(&cli),
        _ => run_online_mode(&cli),
    }
}

fn run_online_mode(cli: &Cli) -> anyhow::Result<()> {
    let base_url = cli
        .base_url
        .clone()
        .unwrap_or_else(|| "http://localhost:8000".to_string());
    let model = cli.model.clone().unwrap_or_else(|| "model".to_string());

    let concurrency = cli.concurrency.unwrap_or(16usize);
    let num_requests = cli.requests.unwrap_or(100usize);
    let isl = cli.isl.unwrap_or(128usize);
    let osl = cli.osl.unwrap_or(128usize);

    let workload = SkeletonWorkload {
        num_requests,
        input_tokens: isl,
        output_tokens: osl,
    };
    // The online sink is `!Send` (hyper transport over `Rc<dyn Clock>`), so drive
    // the run on a current-thread runtime + LocalSet.
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    let local = tokio::task::LocalSet::new();
    let report = local.block_on(&rt, run(base_url, model, workload, concurrency))?;
    print_report_table(&report);
    if let Some(path) = &cli.json {
        write_report_json(&report, path)?;
    }
    Ok(())
}

/// Parsed graph-mode invocation: the bench config plus transport-selection
/// knobs and the raw `base_url` string (retained for the startup banner).
struct GraphParams {
    cfg: aiperf_graph::bench::BenchConfig,
    base_url: String,
    http2: bool,
    conns: usize,
}

/// Aggregated graph-mode results, formatted by [`print_graph_summary`].
struct GraphSummary {
    rps: f64,
    p50: f64,
    p90: f64,
    p99: f64,
    mean: f64,
    secs: f64,
    extra: String,
}

/// Parse a [`Cli`] into a [`GraphParams`] (positionals + flags → bench config).
fn parse_graph_config(cli: &Cli) -> GraphParams {
    use aiperf_graph::bench::BenchConfig;

    let base_url = cli
        .base_url
        .clone()
        .unwrap_or_else(|| "http://127.0.0.1:8000".to_string());
    let model = cli.model.clone().unwrap_or_else(|| "model".to_string());

    let cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(8);
    let turns: usize = cli.turns.unwrap_or(4);
    let workers: usize = cli.workers.unwrap_or(cores);
    let concurrency: usize = cli.concurrency.unwrap_or(64);
    let max_tokens: usize = cli.osl.unwrap_or(1);
    let instances: usize = cli.instances.unwrap_or(DEFAULT_INSTANCES);
    let request_concurrency: Option<usize> = cli.request_concurrency;
    let prefill_concurrency: Option<usize> = cli.prefill_concurrency;
    // Transport default is HTTP/1.1 keep-alive: for serial per-lane requests it
    // outperforms h2c (no per-stream hpack/flow-control overhead). --http2 opts
    // into h2c prior-knowledge (multiplexed pool).
    let http2 = cli.http2;
    // `--http1` is accepted for compatibility (it was a silent no-op in the
    // legacy parser too): HTTP/1.1 keep-alive is already the transport default,
    // so the flag has no additional effect beyond not passing `--http2`.
    let _http1 = cli.http1;
    let conns: usize = cli.conns.unwrap_or(8);

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

    GraphParams {
        cfg,
        base_url,
        http2,
        conns,
    }
}

/// Render the graph-mode summary to stdout (byte-exact with the legacy output).
fn print_graph_summary(s: &GraphSummary, backend: &str) {
    let rps = s.rps;
    println!("\n=== aiperf --mode graph (Graph-IR E2E, streaming SSE, backend={backend}) ===");
    println!("{}", s.extra);
    println!("wall        : {:.3} s", s.secs);
    println!("RPS         : {rps:.0} req/s");
    println!("TTFT p50    : {:.3} ms", s.p50);
    println!("TTFT p90    : {:.3} ms", s.p90);
    println!("TTFT p99    : {:.3} ms", s.p99);
    println!("TTFT mean   : {:.3} ms", s.mean);
    if rps >= RPS_1M {
        println!("\nPROVEN: aiperf --mode graph >= 1M req/s ({rps:.0}, backend={backend})");
    } else if rps >= RPS_500K {
        println!("\nPROVEN: aiperf --mode graph >= 500k req/s ({rps:.0}, backend={backend})");
    } else if rps >= RPS_300K {
        println!("\n>= 300k: {rps:.0}");
    } else {
        println!("\nbelow 300k: {rps:.0}");
    }
}

fn run_graph_mode(cli: &Cli) -> anyhow::Result<()> {
    use aiperf_graph::transport_bench::run_transport_bench;

    let GraphParams {
        cfg,
        base_url,
        http2,
        conns,
    } = parse_graph_config(cli);

    let backend = "aiperf-transport";

    eprintln!(
        "aiperf --mode graph: backend={backend} base={base_url} turns={} \
         instances={} workers={} concurrency={} osl={} \
         conns/worker={conns} offered_concurrency={} http2={http2}",
        cfg.turns,
        cfg.instances,
        cfg.workers,
        cfg.concurrency,
        cfg.max_tokens,
        cfg.workers * cfg.concurrency,
    );

    let r = run_transport_bench(cfg, http2, conns);
    let summary = GraphSummary {
        rps: r.rps(),
        p50: r.ttft_p50_ms,
        p90: r.ttft_p90_ms,
        p99: r.ttft_p99_ms,
        mean: r.ttft_mean_ms,
        secs: r.wall_secs,
        extra: format!(
            "completed={} errors={} output_tokens={} output_tok/s={:.0}",
            r.completed,
            r.errors,
            r.output_tokens,
            r.output_tps()
        ),
    };

    print_graph_summary(&summary, backend);
    Ok(())
}
