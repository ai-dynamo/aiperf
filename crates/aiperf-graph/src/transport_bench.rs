// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Transport-backed throughput benchmark for graph-IR mode.
//!
//! Same Graph-IR E2E path as [`crate::bench`] (thread-per-core workers, each a
//! `current_thread` runtime + `LocalSet` running `concurrency` trace lanes), but
//! HTTP dispatch runs on the Rust-native [`aiperf_transport`] client instead of
//! reqwest. Each serial lane keeps one reused connection:
//!
//! * default: **HTTP/1.1 keep-alive** (fastest for serial lanes — no per-stream
//!   hpack/flow-control overhead);
//! * `--http2`: h2c prior-knowledge, cloning senders off a small per-worker pool
//!   so many lanes multiplex over few connections;
//! * `unix:/path` base URL: **Unix-domain socket** (HTTP/1.1), which bypasses the
//!   TCP/IP loopback softirq tax and is what pushes co-located throughput past
//!   1M req/s.
//!
//! Streaming SSE is parsed incrementally (assistant text + first-token time) via
//! the lean [`HttpClient::dispatch_streaming`], and throughput + TTFT p50/p90/p99
//! are computed from lock-free per-worker accumulators merged once at the end.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use anyhow::Result;
use async_trait::async_trait;
use bytes::Bytes;

use crate::bench::{BenchConfig, build_workload};
use crate::executor::TraceExecutor;
use crate::materialize::SegmentItemsMaterializer;
use crate::model::{GraphRecord, TraceRecord};
use crate::runtime::Handle;
use crate::segment::{SegmentPool, SegmentStore};
use crate::sink::{GraphReply, GraphSink};
use crate::wire::OpenAiChatMessage as Msg;
use aiperf_clock::Clock;
use aiperf_clock::real_clock::RealClock;
use aiperf_core::sse::ChatChunk;
use aiperf_transport::client::connection::{Sender, establish};
use aiperf_transport::client::http_client::HttpClient;
use aiperf_transport::config::ClientConfig;
use aiperf_transport::models::{HttpVersion, SseMessage, TraceData};
use url::Url;

/// Direct-serialize request body (no intermediate `serde_json::Value`).
#[derive(serde::Serialize)]
struct ChatReq<'a> {
    model: &'a str,
    stream: bool,
    stream_options: StreamOpts,
    max_tokens: usize,
    messages: &'a [Msg],
}

#[derive(serde::Serialize)]
struct StreamOpts {
    include_usage: bool,
}

/// Lock-free per-worker measurement accumulator. Each worker thread owns one
/// (shared across its lanes via `Rc`, single-threaded so no locking), and its
/// samples are merged into the global report once at the end — avoiding the
/// per-request collector-mutex contention that caps a shared observer.
#[derive(Default)]
struct WorkerMetrics {
    ttft_ms: Vec<f32>,
    completed: u64,
    errors: u64,
}

/// The transport-bench result: throughput + TTFT distribution.
#[derive(Debug, Clone)]
pub struct GraphRpsReport {
    pub completed: u64,
    pub errors: u64,
    pub wall_secs: f64,
    pub ttft_p50_ms: f64,
    pub ttft_p90_ms: f64,
    pub ttft_p99_ms: f64,
    pub ttft_mean_ms: f64,
}

impl GraphRpsReport {
    pub fn rps(&self) -> f64 {
        if self.wall_secs > 0.0 {
            self.completed as f64 / self.wall_secs
        } else {
            0.0
        }
    }
}

fn percentile(sorted: &[f32], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((p / 100.0) * (sorted.len() as f64 - 1.0)).round() as usize;
    sorted[idx.min(sorted.len() - 1)] as f64
}

/// A per-lane metered sink over [`aiperf_transport`]. Holds one (usually cloned,
/// multiplexed) HTTP/2 sender, reused across the lane's serial requests, and
/// records TTFT into the shared per-worker accumulator.
struct TransportMeteredSink {
    client: Rc<HttpClient>,
    clock: Rc<dyn Clock>,
    cfg: ClientConfig,
    url: Url,
    model: String,
    metrics: Rc<RefCell<WorkerMetrics>>,
    max_tokens: usize,
    headers: BTreeMap<String, String>,
    /// The lane's sender (an h2 clone off the worker pool, or a standalone
    /// re-established connection after a failure).
    sender: RefCell<Option<Sender>>,
}

impl TransportMeteredSink {
    /// Ensure a live sender, re-establishing a standalone connection if the
    /// pooled one closed.
    async fn ensure(&self) -> bool {
        let need = {
            let s = self.sender.borrow();
            s.as_ref().map(|s| s.is_closed()).unwrap_or(true)
        };
        if need {
            let mut t = TraceData::default();
            match establish(&self.url, &self.cfg, self.clock.clone(), &mut t).await {
                Ok((s, _)) => *self.sender.borrow_mut() = Some(s),
                Err(_) => return false,
            }
        }
        true
    }
}

#[async_trait(?Send)]
impl GraphSink<Msg> for TransportMeteredSink {
    async fn dispatch(
        &self,
        _node_id: &str,
        messages: Vec<Msg>,
        max_tokens: Option<usize>,
        on_first_token: &dyn Fn(),
    ) -> Result<GraphReply<Msg>> {
        let mot = max_tokens.unwrap_or(self.max_tokens);

        // Serialize the request directly from a borrowing struct — avoids the
        // per-request `serde_json::Value` (BTreeMap) tree that dominated the
        // allocator profile.
        let req = ChatReq {
            model: &self.model,
            stream: true,
            stream_options: StreamOpts {
                include_usage: true,
            },
            max_tokens: mot,
            messages: &messages,
        };
        let body = Bytes::from(serde_json::to_vec(&req).unwrap_or_default());

        if !self.ensure().await {
            self.metrics.borrow_mut().errors += 1;
            return Ok(GraphReply::failed());
        }

        // Lean streaming dispatch: parse the assistant text in the per-message
        // callback (so the successor turn splices a real reply) with no
        // RequestRecord / Vec<Response> accumulation in the hot loop. TTFT is
        // the first REAL token payload (first non-empty content delta), not the
        // first SSE message (which may be a role-only chunk).
        let req_start = self.clock.now_ns();
        let mut first_token_ns: Option<i64> = None;
        let mut content = String::new();
        let status = {
            let mut c = self.sender.borrow_mut();
            let sender = c.as_mut().unwrap();
            // Transport's first-SSE-message signal (content-agnostic); TTFT and
            // the successor gate below fire on the first real content token.
            let mut on_ft = |_ns: i64| {};
            let mut on_msg = |m: &SseMessage| {
                if let Some(d) = m.data()
                    && d != "[DONE]"
                    && let Ok(chunk) = serde_json::from_str::<ChatChunk>(d)
                {
                    let mut had_token = false;
                    for ch in &chunk.choices {
                        if let Some(t) = &ch.delta.content
                            && !t.is_empty()
                        {
                            content.push_str(t);
                            had_token = true;
                        }
                        if let Some(t) = &ch.delta.reasoning_content
                            && !t.is_empty()
                        {
                            content.push_str(t);
                            had_token = true;
                        }
                    }
                    if had_token && first_token_ns.is_none() {
                        first_token_ns = Some((m.perf_ns - req_start).max(0));
                        on_first_token();
                    }
                }
            };
            self.client
                .dispatch_streaming(
                    sender,
                    &self.url,
                    &self.headers,
                    body,
                    &mut on_ft,
                    &mut on_msg,
                )
                .await
        };

        let ok = matches!(status, Ok(200));
        if ok {
            let mut m = self.metrics.borrow_mut();
            m.completed += 1;
            if let Some(ns) = first_token_ns {
                m.ttft_ms.push((ns as f64 / 1_000_000.0) as f32);
            }
        } else {
            *self.sender.borrow_mut() = None;
            self.metrics.borrow_mut().errors += 1;
        }

        Ok(if ok {
            GraphReply::from_text(content)
        } else {
            GraphReply::failed()
        })
    }
}

fn chat_headers() -> BTreeMap<String, String> {
    let mut h = BTreeMap::new();
    h.insert("Content-Type".to_string(), "application/json".to_string());
    h.insert("Accept".to_string(), "text/event-stream".to_string());
    h
}

#[cfg(target_os = "linux")]
fn raise_fd_limit() {
    unsafe {
        let mut lim = libc::rlimit {
            rlim_cur: 0,
            rlim_max: 0,
        };
        if libc::getrlimit(libc::RLIMIT_NOFILE, &mut lim) == 0 {
            lim.rlim_cur = lim.rlim_max;
            let _ = libc::setrlimit(libc::RLIMIT_NOFILE, &lim);
        }
    }
}
#[cfg(not(target_os = "linux"))]
fn raise_fd_limit() {}

/// Run the transport-backed benchmark and return the throughput + TTFT report.
/// `http2` selects h2c prior-knowledge (required for the multiplexed connection
/// pool); when false, HTTP/1.1 is used (one standalone connection per lane).
/// `conns` is the number of shared connections opened per worker thread.
pub fn run_transport_bench(cfg: BenchConfig, http2: bool, conns: usize) -> GraphRpsReport {
    raise_fd_limit();
    let (pool, graph, _isl) = build_workload(cfg.turns);
    let graph = Arc::new(graph);
    let pool = Arc::new(pool);

    let start = Instant::now();
    let next = Arc::new(AtomicUsize::new(0));
    let merged: Arc<std::sync::Mutex<(Vec<f32>, u64, u64)>> =
        Arc::new(std::sync::Mutex::new((Vec::new(), 0, 0)));

    let servers = if cfg.base_urls.is_empty() {
        vec!["http://127.0.0.1:8000".to_string()]
    } else {
        cfg.base_urls.clone()
    };

    std::thread::scope(|scope| {
        for widx in 0..cfg.workers.max(1) {
            let (graph, pool, next, merged) =
                (graph.clone(), pool.clone(), next.clone(), merged.clone());
            let base_url = servers[widx % servers.len()].clone();
            let model = cfg.model.clone();
            let (instances, concurrency, max_tokens) =
                (cfg.instances, cfg.concurrency, cfg.max_tokens);
            scope.spawn(move || {
                let wm = transport_worker(
                    &graph,
                    &pool,
                    &next,
                    &base_url,
                    &model,
                    instances,
                    concurrency,
                    max_tokens,
                    http2,
                    conns.max(1),
                );
                // Merge this worker's samples into the global report once.
                let mut g = merged.lock().unwrap();
                g.0.extend_from_slice(&wm.ttft_ms);
                g.1 += wm.completed;
                g.2 += wm.errors;
            });
        }
    });

    let wall_secs = start.elapsed().as_secs_f64();
    let (mut ttft, completed, errors) = Arc::try_unwrap(merged)
        .map(|m| m.into_inner().unwrap())
        .unwrap_or_else(|m| m.lock().unwrap().clone());
    ttft.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mean = if ttft.is_empty() {
        0.0
    } else {
        ttft.iter().map(|&x| x as f64).sum::<f64>() / ttft.len() as f64
    };

    GraphRpsReport {
        completed,
        errors,
        wall_secs,
        ttft_p50_ms: percentile(&ttft, 50.0),
        ttft_p90_ms: percentile(&ttft, 90.0),
        ttft_p99_ms: percentile(&ttft, 99.0),
        ttft_mean_ms: mean,
    }
}

#[allow(clippy::too_many_arguments)]
fn transport_worker(
    graph: &Arc<GraphRecord>,
    pool: &Arc<SegmentPool<Msg>>,
    next: &Arc<AtomicUsize>,
    base_url: &str,
    model: &str,
    instances: usize,
    concurrency: usize,
    max_tokens: usize,
    http2: bool,
    conns: usize,
) -> WorkerMetrics {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("worker runtime");
    let local = tokio::task::LocalSet::new();

    let store: Rc<dyn SegmentStore<Msg>> = {
        let pool = (**pool).clone();
        Rc::new(pool)
    };
    let materializer = Rc::new(SegmentItemsMaterializer::new(store));
    // "unix:/path/to.sock" connects over a Unix-domain socket (HTTP/1.1),
    // bypassing the TCP/IP loopback softirq tax for co-located benchmarking.
    let (uds_path, url): (Option<String>, Url) = match base_url.strip_prefix("unix:") {
        Some(p) => (
            Some(p.to_string()),
            Url::parse("http://localhost/v1/chat/completions").expect("valid uds chat url"),
        ),
        None => (
            None,
            Url::parse(base_url)
                .expect("valid base url")
                .join("/v1/chat/completions")
                .expect("valid chat url"),
        ),
    };
    let uds = uds_path.is_some();
    let http_version = if http2 && !uds {
        HttpVersion::Http2PriorKnowledge
    } else {
        HttpVersion::Http1Only
    };
    let http2 = http2 && !uds;
    let metrics = Rc::new(RefCell::new(WorkerMetrics::default()));

    local.block_on(&rt, async {
        let clock: Rc<dyn Clock> = RealClock::new();
        let cfg = ClientConfig {
            http_version,
            uds_path,
            ..ClientConfig::default()
        };
        let client = Rc::new(HttpClient::new(clock.clone(), cfg.clone()));

        // Open the worker's shared connection pool for h2c so many serial lanes
        // multiplex over few connections. For h1 each lane opens its own
        // keep-alive connection lazily, so the pool is skipped.
        let mut base: Vec<Sender> = Vec::with_capacity(conns);
        if http2 {
            for _ in 0..conns {
                let mut t = TraceData::default();
                if let Ok((s, _)) = establish(&url, &cfg, clock.clone(), &mut t).await {
                    base.push(s);
                }
            }
        }

        let mut lanes = Vec::with_capacity(concurrency.max(1));
        for i in 0..concurrency.max(1) {
            // h2: clone a sender off the pool (independent multiplexed stream);
            // h1 or empty pool: each lane establishes its own on first use.
            let lane_sender: Option<Sender> = if http2 && !base.is_empty() {
                base[i % base.len()].clone_multiplex()
            } else {
                None
            };
            let sink: Rc<dyn GraphSink<Msg>> = Rc::new(TransportMeteredSink {
                client: client.clone(),
                clock: clock.clone(),
                cfg: cfg.clone(),
                url: url.clone(),
                model: model.to_string(),
                metrics: metrics.clone(),
                max_tokens,
                headers: chat_headers(),
                sender: RefCell::new(lane_sender),
            });
            lanes.push(transport_run_lane(
                graph.clone(),
                materializer.clone(),
                sink,
                next.clone(),
                instances,
            ));
        }
        futures::future::join_all(lanes).await;
    });

    Rc::try_unwrap(metrics)
        .map(|m| m.into_inner())
        .unwrap_or_default()
}

async fn transport_run_lane(
    graph: Arc<GraphRecord>,
    materializer: Rc<SegmentItemsMaterializer<Msg>>,
    sink: Rc<dyn GraphSink<Msg>>,
    next: Arc<AtomicUsize>,
    instances: usize,
) {
    let graph_rc: Rc<GraphRecord> = Rc::new((*graph).clone());
    loop {
        let i = next.fetch_add(1, Ordering::Relaxed);
        if i >= instances {
            break;
        }
        let handle = Handle::new(RealClock::new());
        let exec = match TraceExecutor::new(
            graph_rc.clone(),
            materializer.clone(),
            sink.clone(),
            handle.clone(),
            false,
            false,
            false,
        ) {
            Ok(e) => e,
            Err(_) => continue,
        };
        let trace = TraceRecord {
            id: format!("t{i}"),
            graph_ref: None,
            initial_state: Default::default(),
        };
        if let Ok(ctx) = exec.build_context(trace) {
            exec.schedule_entries(&ctx);
            handle.wait_idle().await;
        }
    }
}
