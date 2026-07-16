// rust/transport-http/examples/rps_bench.rs
//! Throughput load generator for `aiperf-transport-http`: proves sustained
//! requests-per-second against a running OpenAI-compatible server (the
//! `aiperf-mock-server --fast` mock).
//!
//! Design: thread-per-core (`THREADS` OS threads, each a current-thread tokio
//! runtime + `LocalSet`). Each thread opens `CONNS` HTTP/2 (h2c prior-knowledge)
//! connections and fans each into `LANES` multiplexed streams (cloned h2
//! senders). Every lane loops back-to-back non-streaming chat completions on its
//! shared connection, counting completions in a global atomic. The main thread
//! warms up, then measures completions over a fixed window and prints achieved
//! RPS.
//!
//! Env knobs (all optional):
//!   BASE_URL   (default http://127.0.0.1:8000)
//!   THREADS    (default = num_cpus - 4)
//!   CONNS      (connections per thread, default 8)
//!   LANES      (multiplexed streams per connection, default 16)
//!   WARMUP_S   (default 2)
//!   WINDOW_S   (default 5)
//!   HTTP1      (set to 1 to use HTTP/1.1 keep-alive lanes instead of h2c)

use std::collections::BTreeMap;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use bytes::Bytes;

use aiperf_runtime::transport::core::{RequestRecord, TraceData};
use aiperf_runtime::transport::http::RealClock;
use aiperf_runtime::transport::http::client::connection::{Sender, establish};
use aiperf_runtime::transport::http::client::http_client::HttpClient;
use aiperf_runtime::transport::http::config::ClientConfig;
use aiperf_runtime::transport::http::models::HttpVersion;

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn body_bytes() -> Bytes {
    Bytes::from(
        serde_json::to_vec(&serde_json::json!({
            "model": "m",
            "stream": false,
            "max_tokens": 1,
            "messages": [{"role": "user", "content": "hi"}],
        }))
        .unwrap(),
    )
}

fn main() {
    let base_url = std::env::var("BASE_URL").unwrap_or_else(|_| "http://127.0.0.1:8000".into());
    let cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(8);
    let threads = env_usize("THREADS", cores.saturating_sub(4).max(1));
    let conns = env_usize("CONNS", 8);
    let lanes = env_usize("LANES", 16);
    let warmup_s = env_usize("WARMUP_S", 2) as u64;
    let window_s = env_usize("WINDOW_S", 5) as u64;
    let http1 = std::env::var("HTTP1").map(|v| v == "1").unwrap_or(false);

    let completed = Arc::new(AtomicU64::new(0));
    let errors = Arc::new(AtomicU64::new(0));
    let running = Arc::new(AtomicBool::new(true));

    let http_version = if http1 {
        HttpVersion::Http1Only
    } else {
        HttpVersion::Http2PriorKnowledge
    };
    let url = Arc::new(
        base_url
            .split(',')
            .map(|b| {
                url::Url::parse(b.trim())
                    .unwrap()
                    .join("/v1/chat/completions")
                    .unwrap()
            })
            .collect::<Vec<_>>(),
    );

    println!(
        "rps_bench: base={base_url} threads={threads} conns/thread={conns} lanes/conn={lanes} \
         mode={} warmup={warmup_s}s window={window_s}s",
        if http1 {
            "h1-keepalive"
        } else {
            "h2c-multiplex"
        }
    );

    let mut handles = Vec::new();
    for tidx in 0..threads {
        let completed = completed.clone();
        let errors = errors.clone();
        let running = running.clone();
        let urls = url.clone();
        let url = urls[tidx % urls.len()].clone();
        handles.push(std::thread::spawn(move || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();
            let local = tokio::task::LocalSet::new();
            local.block_on(&rt, async move {
                let clock: Rc<dyn aiperf_runtime::transport::http::Clock> = RealClock::new();
                let cfg = ClientConfig {
                    http_version,
                    ..ClientConfig::default()
                };
                let client = Rc::new(HttpClient::new(clock.clone(), cfg.clone()));
                let body = body_bytes();
                let mut headers = BTreeMap::new();
                headers.insert("Content-Type".to_string(), "application/json".to_string());
                headers.insert("Accept".to_string(), "application/json".to_string());
                let headers = Rc::new(headers);

                let mut lane_tasks = Vec::new();
                for _ in 0..conns {
                    // Establish one connection; fan into `lanes` multiplexed lanes.
                    let mut t = TraceData::default();
                    let first = match establish(&url, &cfg, clock.clone(), &mut t).await {
                        Ok((s, _)) => s,
                        Err(e) => {
                            eprintln!("establish failed: {e:?}");
                            continue;
                        }
                    };
                    // For h2, clone the sender per lane (shared connection); for
                    // h1, each lane needs its own connection (no multiplexing).
                    let mut lane_senders: Vec<Sender> = Vec::with_capacity(lanes);
                    if http1 {
                        lane_senders.push(first);
                        for _ in 1..lanes {
                            let mut t2 = TraceData::default();
                            if let Ok((s, _)) = establish(&url, &cfg, clock.clone(), &mut t2).await
                            {
                                lane_senders.push(s);
                            }
                        }
                    } else {
                        // h2c: every lane is a cheap clone of the one connection's
                        // sender, opening an independent multiplexed stream.
                        for _ in 0..lanes {
                            if let Some(s) = first.clone_multiplex() {
                                lane_senders.push(s);
                            }
                        }
                        // The connection driver spawned by `establish` keeps the
                        // connection alive; the clones carry the traffic.
                        drop(first);
                    }

                    for mut sender in lane_senders {
                        let client = client.clone();
                        let url = url.clone();
                        let headers = headers.clone();
                        let body = body.clone();
                        let completed = completed.clone();
                        let errors = errors.clone();
                        let running = running.clone();
                        lane_tasks.push(tokio::task::spawn_local(async move {
                            let mut noop = |_: i64| {};
                            while running.load(Ordering::Relaxed) {
                                let mut trace = TraceData::default();
                                let mut rec = RequestRecord::started(0);
                                let blen = body.len();
                                let r = client
                                    .dispatch(
                                        &mut sender,
                                        &url,
                                        &headers,
                                        body.clone(),
                                        false,
                                        &mut trace,
                                        &mut rec,
                                        &mut noop,
                                        blen,
                                    )
                                    .await;
                                match r {
                                    Ok(()) if rec.status == Some(200) => {
                                        completed.fetch_add(1, Ordering::Relaxed);
                                    }
                                    _ => {
                                        errors.fetch_add(1, Ordering::Relaxed);
                                        // Connection likely broke; stop this lane.
                                        break;
                                    }
                                }
                            }
                        }));
                    }
                }
                for h in lane_tasks {
                    let _ = h.await;
                }
            });
        }));
    }

    // Warm up, then measure completions over the window.
    std::thread::sleep(Duration::from_secs(warmup_s));
    let c0 = completed.load(Ordering::Relaxed);
    let t0 = Instant::now();
    std::thread::sleep(Duration::from_secs(window_s));
    let c1 = completed.load(Ordering::Relaxed);
    let elapsed = t0.elapsed().as_secs_f64();
    running.store(false, Ordering::Relaxed);

    let rps = (c1 - c0) as f64 / elapsed;
    let errs = errors.load(Ordering::Relaxed);
    println!(
        "\n=== RESULT ===\ncompleted_in_window={} elapsed={:.3}s\nRPS = {:.0} req/s\nerrors={}",
        c1 - c0,
        elapsed,
        rps,
        errs
    );
    if rps >= 300_000.0 {
        println!("PROVEN: >= 300k req/s ({rps:.0})");
    } else {
        println!("below target: {rps:.0} < 300000");
    }

    for h in handles {
        let _ = h.join();
    }
}
