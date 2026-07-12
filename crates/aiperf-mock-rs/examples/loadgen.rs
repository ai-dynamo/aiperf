// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Raw hyper-based load generator for the mock server. Used to measure server
//! throughput independent of the Python aiperf client's overhead.
//!
//! Usage:
//!   cargo run --release --example loadgen -- --url http://127.0.0.1:19901/v1/chat/completions --concurrency 1000 --total 100000 [--streaming]

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use clap::Parser;
use reqwest::Client;

#[derive(Parser)]
struct Args {
    #[arg(long, default_value = "http://127.0.0.1:19901/v1/chat/completions")]
    url: String,
    #[arg(long, default_value_t = 1000)]
    concurrency: usize,
    #[arg(long, default_value_t = 50_000)]
    total: usize,
    #[arg(long, default_value_t = false)]
    streaming: bool,
    #[arg(long, default_value_t = 20)]
    out_tokens: usize,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let client = Client::builder()
        .no_proxy()
        .http1_only()
        .pool_max_idle_per_host(args.concurrency * 2)
        .pool_idle_timeout(Duration::from_secs(30))
        .tcp_nodelay(true)
        .build()?;

    let body = serde_json::json!({
        "model": "gpt2",
        "messages": [{"role": "user", "content": "Hello, how are you today?"}],
        "max_tokens": args.out_tokens,
        "ignore_eos": true,
        "stream": args.streaming,
    })
    .to_string();

    let completed = Arc::new(AtomicU64::new(0));
    let errors = Arc::new(AtomicU64::new(0));
    let total_tokens = Arc::new(AtomicU64::new(0));
    let total_latency_us = Arc::new(AtomicU64::new(0));
    let max_latency_us = Arc::new(AtomicU64::new(0));

    let per_worker = args.total / args.concurrency;
    let start = Instant::now();

    let mut handles = Vec::with_capacity(args.concurrency);
    for _w in 0..args.concurrency {
        let client = client.clone();
        let url = args.url.clone();
        let body = body.clone();
        let completed = completed.clone();
        let errors = errors.clone();
        let total_tokens = total_tokens.clone();
        let total_latency_us = total_latency_us.clone();
        let max_latency_us = max_latency_us.clone();
        let streaming = args.streaming;
        handles.push(tokio::spawn(async move {
            for _ in 0..per_worker {
                let req_start = Instant::now();
                let resp = client
                    .post(&url)
                    .header("content-type", "application/json")
                    .body(body.clone())
                    .send()
                    .await;
                let ok = match resp {
                    Ok(r) => {
                        if r.status().is_success() {
                            if streaming {
                                // drain the body
                                let bytes = r.bytes().await.unwrap_or_default();
                                let chunks = bytes.windows(6).filter(|w| w == b"data: ").count();
                                total_tokens.fetch_add(chunks as u64, Ordering::Relaxed);
                            } else {
                                let _ = r.bytes().await;
                            }
                            true
                        } else {
                            false
                        }
                    }
                    Err(_) => false,
                };
                let us = req_start.elapsed().as_micros() as u64;
                total_latency_us.fetch_add(us, Ordering::Relaxed);
                max_latency_us.fetch_max(us, Ordering::Relaxed);
                if ok {
                    completed.fetch_add(1, Ordering::Relaxed);
                } else {
                    errors.fetch_add(1, Ordering::Relaxed);
                }
            }
        }));
    }
    for h in handles {
        h.await?;
    }
    let elapsed = start.elapsed();
    let completed = completed.load(Ordering::Relaxed);
    let errors = errors.load(Ordering::Relaxed);
    let total_tokens = total_tokens.load(Ordering::Relaxed);
    let sum_us = total_latency_us.load(Ordering::Relaxed);
    let max_us = max_latency_us.load(Ordering::Relaxed);
    let avg_us = sum_us.checked_div(completed).unwrap_or(0);

    println!("elapsed: {:.3}s", elapsed.as_secs_f64());
    println!(
        "completed: {}  errors: {}  rps: {:.0}",
        completed,
        errors,
        (completed as f64) / elapsed.as_secs_f64()
    );
    println!(
        "avg latency: {:.3}ms  max: {:.3}ms",
        (avg_us as f64) / 1000.0,
        (max_us as f64) / 1000.0
    );
    if args.streaming {
        println!(
            "total chunks: {}  avg chunks/req: {:.1}  tok/s: {:.0}",
            total_tokens,
            if completed > 0 {
                (total_tokens as f64) / (completed as f64)
            } else {
                0.0
            },
            (total_tokens as f64) / elapsed.as_secs_f64()
        );
    }
    Ok(())
}
