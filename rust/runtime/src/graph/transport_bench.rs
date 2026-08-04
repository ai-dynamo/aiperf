// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Transport-backed throughput benchmark for graph-IR mode.
//!
//! Graph-IR E2E path with thread-per-core workers (each a `current_thread`
//! runtime + `LocalSet` running `concurrency` trace lanes); HTTP dispatch runs
//! on the Rust-native [`crate::transport::http`] client. The multi-turn workload
//! scaffolding (segment pool, [`BenchConfig`], server resolution) is shared with
//! [`crate::graph::bench`]. Each serial lane keeps one reused connection:
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
use std::collections::{BTreeMap, HashMap};
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use anyhow::Result;
use async_trait::async_trait;
use bytes::Bytes;

use crate::clock::Clock;
use crate::clock::real_clock::RealClock;
use crate::dataset::{Overrides, build_message_body_from_wires};
use crate::graph::bench::{BenchConfig, build_workload, resolve_servers};
use crate::graph::executor::{ExecutorFlags, TraceExecutor};
use crate::graph::materialize::SegmentItemsMaterializer;
use crate::graph::model::{GraphRecord, TraceRecord};
use crate::graph::runtime::Handle;
use crate::graph::segment::{InMemorySegmentStore, SegmentStore};
use crate::graph::sink::{GraphReply, GraphSink};
use crate::graph::wire::OpenAiChatMessage as Msg;
use crate::metrics_core::{
    AccumulatorSummary, InferenceDimensions, MetricsAccumulator, Phase, RecordIngest, TokenCounts,
    UsageMetrics,
};
use crate::timing::{RunState, StopChecker, StopConfig};
use crate::transport::core::SseMessage;
use crate::transport::core::TraceData;
use crate::transport::http::client::connection::{Sender, establish};
use crate::transport::http::client::http_client::HttpClient;
use crate::transport::http::config::ClientConfig;
use crate::transport::http::models::HttpVersion;
use crate::transport::http::sse::ChatChunk;
use url::Url;

/// Lock-free per-worker measurement accumulator. Each worker thread owns one
/// (shared across its lanes via `Rc`, single-threaded so no locking), and its
/// samples are merged into the global report once at the end — avoiding the
/// per-request collector-mutex contention that caps a shared observer.
#[derive(Default)]
struct WorkerMetrics {
    ttft_ms: Vec<f32>,
    completed: u64,
    errors: u64,
    output_tokens: u64,
    next_record: u64,
    native: MetricsAccumulator,
}

/// Globally merged worker samples and append-only native metric columns.
#[derive(Default)]
struct MergedMetrics {
    ttft_ms: Vec<f32>,
    completed: u64,
    errors: u64,
    output_tokens: u64,
    native: MetricsAccumulator,
}

/// The transport-bench result: throughput + TTFT distribution.
#[derive(Debug, Clone)]
pub struct GraphRpsReport {
    pub completed: u64,
    pub errors: u64,
    pub output_tokens: u64,
    pub wall_secs: f64,
    pub ttft_p50_ms: f64,
    pub ttft_p90_ms: f64,
    pub ttft_p99_ms: f64,
    pub ttft_mean_ms: f64,
    /// Native typed distributions and sweeps merged once across workers.
    pub native_metrics: AccumulatorSummary,
}

impl GraphRpsReport {
    pub fn rps(&self) -> f64 {
        if self.wall_secs > 0.0 {
            self.completed as f64 / self.wall_secs
        } else {
            0.0
        }
    }
    /// Output tokens per second (SSE content chunks / wall).
    pub fn output_tps(&self) -> f64 {
        if self.wall_secs > 0.0 {
            self.output_tokens as f64 / self.wall_secs
        } else {
            0.0
        }
    }
}

fn percentile(sorted: &[f32], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    // One shared nearest-rank definition (dispatch::collector owns the stats math).
    let idx = crate::dispatch::collector::percentile_rank(sorted.len(), p);
    sorted[idx] as f64
}

/// A per-lane metered sink over [`crate::transport::http`]. Holds one (usually cloned,
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
    input_tokens_by_node: Arc<HashMap<String, usize>>,
    worker_id: String,
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
        node_id: &str,
        messages: Vec<Bytes>,
        max_tokens: Option<usize>,
        on_first_token: &dyn Fn(),
    ) -> Result<GraphReply<Msg>> {
        let mot = max_tokens.unwrap_or(self.max_tokens);

        let mut overrides = Overrides::new();
        overrides.set_model(&self.model);
        overrides.set_stream(true);
        overrides.set_include_usage(true);
        overrides.set_max_tokens("max_tokens", u32::try_from(mot).unwrap_or(u32::MAX));
        let body = build_message_body_from_wires(&messages, &overrides)?;

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
        let mut first_output_token_ns: Option<i64> = None;
        let mut token_arrival_ns = Vec::with_capacity(mot);
        let mut tokens: u64 = 0;
        let mut output_tokens: u64 = 0;
        let mut reasoning_tokens: u64 = 0;
        let mut usage_prompt_tokens = None;
        let mut usage_completion_tokens = None;
        let mut content = String::new();
        // Take the sender OUT of the RefCell so its borrow is not held across the
        // `.await` below (clippy `await_holding_refcell_ref`); `ensure()` above
        // guarantees it is `Some`. It is put back for the lane's next request.
        let mut sender = self.sender.borrow_mut().take().unwrap();
        let status = {
            // Transport's first-SSE-message signal (content-agnostic); TTFT and
            // the successor gate below fire on the first real content token.
            let mut on_ft = |_ns: i64| {};
            let mut on_msg = |m: &SseMessage| {
                if let Some(d) = m.data()
                    && d != "[DONE]"
                    && let Ok(chunk) = serde_json::from_str::<ChatChunk>(d)
                {
                    if let Some(usage) = &chunk.usage {
                        usage_prompt_tokens = Some(u64::from(usage.prompt_tokens));
                        usage_completion_tokens = Some(u64::from(usage.completion_tokens));
                    }
                    let delta = chunk.delta_text();
                    if !delta.is_empty() {
                        content.push_str(&delta);
                        tokens += 1;
                        token_arrival_ns.push(m.perf_ns);
                        if chunk.has_output_delta() {
                            output_tokens += 1;
                            first_output_token_ns.get_or_insert(m.perf_ns);
                        } else {
                            reasoning_tokens += 1;
                        }
                        if first_token_ns.is_none() {
                            first_token_ns = Some((m.perf_ns - req_start).max(0));
                            on_first_token();
                        }
                    }
                }
            };
            self.client
                .dispatch_streaming(
                    &mut sender,
                    &self.url,
                    &self.headers,
                    body,
                    &mut on_ft,
                    &mut on_msg,
                )
                .await
        };
        // Return the sender to the lane's slot for reuse (the error path below
        // overrides this with `None` to force a re-establish).
        *self.sender.borrow_mut() = Some(sender);

        let ok = matches!(status, Ok(200));
        let response_end = self.clock.now_ns();
        let mut m = self.metrics.borrow_mut();
        let ordinal = m.next_record;
        m.next_record = m.next_record.saturating_add(1);
        let input_tokens = self.input_tokens_by_node.get(node_id).copied();
        let record = RecordIngest {
            request_index: usize::try_from(ordinal).ok(),
            global_dispatch_index: None,
            correlation_id: format!("{}:{node_id}:{ordinal}", self.worker_id),
            session_num: ordinal,
            turn_index: node_id
                .strip_prefix('n')
                .and_then(|value| value.parse::<u32>().ok())
                .unwrap_or(0),
            worker_id: Some(self.worker_id.clone()),
            worker_assignment_index: None,
            conversation_id: None,
            dimensions: InferenceDimensions {
                endpoint_url: Some(self.url.to_string()),
                model: Some(self.model.clone()),
            },
            phase: Phase::Profiling,
            phase_index: None,
            phase_name: None,
            phase_kind: None,
            profiling_index: None,
            start_ns: req_start,
            end_ns: response_end,
            admit_ns: Some(req_start),
            first_token_ns: token_arrival_ns.first().copied(),
            second_token_ns: token_arrival_ns.get(1).copied(),
            first_output_token_ns,
            token_arrival_ns,
            errored: !ok,
            canceled: false,
            tokens: TokenCounts {
                input: input_tokens.map(|value| value as u64),
                output: Some(output_tokens),
                reasoning: (reasoning_tokens > 0).then_some(reasoning_tokens),
                requested_output: Some(mot as u64),
            },
            usage: UsageMetrics {
                prompt_tokens: usage_prompt_tokens,
                completion_tokens: usage_completion_tokens,
                total_tokens: usage_prompt_tokens
                    .zip(usage_completion_tokens)
                    .map(|(prompt, completion)| prompt.saturating_add(completion)),
                ..UsageMetrics::default()
            },
            http: Default::default(),
            audio_duration_s: None,
            num_images: None,
            video_inference_seconds: None,
            video_peak_memory_mb: None,
            metric_overrides: Vec::new(),
        };
        m.native.process_record(&record);
        if ok {
            m.completed += 1;
            m.output_tokens += tokens;
            if let Some(ns) = first_token_ns {
                m.ttft_ms.push((ns as f64 / 1_000_000.0) as f32);
            }
        } else {
            *self.sender.borrow_mut() = None;
            m.errors += 1;
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

/// Run the transport-backed benchmark and return the throughput + TTFT report.
/// `http2` selects h2c prior-knowledge (required for the multiplexed connection
/// pool); when false, HTTP/1.1 is used (one standalone connection per lane).
/// `conns` is the number of shared connections opened per worker thread.
pub fn run_transport_bench(cfg: BenchConfig, http2: bool, conns: usize) -> GraphRpsReport {
    crate::graph::syslimits::raise_fd_limit();
    let (pool, graph, input_tokens_by_node) = build_workload(cfg.turns);
    let graph = Arc::new(graph);
    let pool = Arc::new(pool);
    let input_tokens_by_node = Arc::new(input_tokens_by_node);

    let wall_clock = RealClock::new();
    let wall_start_ns = wall_clock.now_ns();
    let next = Arc::new(AtomicUsize::new(0));

    let servers = resolve_servers(&cfg.base_urls);

    let workers = std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(cfg.workers.max(1));
        for widx in 0..cfg.workers.max(1) {
            let (graph, pool, input_tokens_by_node, next) = (
                graph.clone(),
                pool.clone(),
                input_tokens_by_node.clone(),
                next.clone(),
            );
            let base_url = servers[widx % servers.len()].clone();
            let model = cfg.model.clone();
            let (instances, concurrency, max_tokens, max_duration_ns) = (
                cfg.instances,
                cfg.concurrency,
                cfg.max_tokens,
                cfg.max_duration_ns,
            );
            handles.push(scope.spawn(move || {
                transport_worker(
                    &graph,
                    &pool,
                    input_tokens_by_node,
                    widx,
                    &next,
                    &base_url,
                    &model,
                    instances,
                    concurrency,
                    max_tokens,
                    max_duration_ns,
                    http2,
                    conns.max(1),
                )
            }));
        }
        handles
            .into_iter()
            .map(|handle| handle.join().expect("graph worker panicked"))
            .collect::<Vec<_>>()
    });

    let wall_secs = wall_clock.now_ns().saturating_sub(wall_start_ns) as f64 / 1_000_000_000.0;
    let mut merged = MergedMetrics::default();
    for worker in workers {
        merged.ttft_ms.extend_from_slice(&worker.ttft_ms);
        merged.completed += worker.completed;
        merged.errors += worker.errors;
        merged.output_tokens += worker.output_tokens;
        merged
            .native
            .merge(&worker.native)
            .expect("workers share one metrics configuration");
    }
    merged
        .ttft_ms
        .sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mean = if merged.ttft_ms.is_empty() {
        0.0
    } else {
        merged.ttft_ms.iter().map(|&x| x as f64).sum::<f64>() / merged.ttft_ms.len() as f64
    };
    let native_metrics = merged.native.summarize();

    GraphRpsReport {
        completed: merged.completed,
        errors: merged.errors,
        output_tokens: merged.output_tokens,
        wall_secs,
        ttft_p50_ms: percentile(&merged.ttft_ms, 50.0),
        ttft_p90_ms: percentile(&merged.ttft_ms, 90.0),
        ttft_p99_ms: percentile(&merged.ttft_ms, 99.0),
        ttft_mean_ms: mean,
        native_metrics,
    }
}

/// Per-worker duration stop gate. Consults the shared [`crate::timing::StopChecker`]
/// so a lane stops pulling new trace instances once this worker's clock passes the
/// deadline. Built only when a `--duration` bound is set; when absent the lane loop
/// adds no per-iteration work and stops purely on instance-count exhaustion.
struct DurationGate {
    checker: StopChecker,
    state: RunState,
    clock: Rc<dyn Clock>,
}

impl DurationGate {
    #[inline]
    fn expired(&self) -> bool {
        !self.checker.can_send_any(&self.state, self.clock.now_ns())
    }
}

#[allow(clippy::too_many_arguments)]
fn transport_worker(
    graph: &Arc<GraphRecord>,
    pool: &Arc<InMemorySegmentStore>,
    input_tokens_by_node: Arc<HashMap<String, usize>>,
    worker_index: usize,
    next: &Arc<AtomicUsize>,
    base_url: &str,
    model: &str,
    instances: usize,
    concurrency: usize,
    max_tokens: usize,
    max_duration_ns: Option<i64>,
    http2: bool,
    conns: usize,
) -> WorkerMetrics {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("worker runtime");
    let local = tokio::task::LocalSet::new();

    let store: Arc<dyn SegmentStore> = pool.clone();
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
        // One duration gate per worker, sharing this worker's clock. `started_at_ns`
        // is stamped now so the [`crate::timing::Duration`] condition measures
        // elapsed time from the worker's start. Shared across the worker's lanes.
        let gate: Option<Rc<DurationGate>> = max_duration_ns.map(|d| {
            Rc::new(DurationGate {
                checker: StopChecker::new(&StopConfig {
                    expected_duration_ns: Some(d),
                    ..Default::default()
                }),
                state: RunState {
                    started_at_ns: clock.now_ns(),
                    ..Default::default()
                },
                clock: clock.clone(),
            })
        });
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
                input_tokens_by_node: input_tokens_by_node.clone(),
                worker_id: format!("worker-{worker_index}"),
                sender: RefCell::new(lane_sender),
            });
            lanes.push(transport_run_lane(
                graph.clone(),
                materializer.clone(),
                sink,
                next.clone(),
                instances,
                gate.clone(),
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
    materializer: Rc<SegmentItemsMaterializer>,
    sink: Rc<dyn GraphSink<Msg>>,
    next: Arc<AtomicUsize>,
    instances: usize,
    gate: Option<Rc<DurationGate>>,
) {
    let graph_rc: Rc<GraphRecord> = Rc::new((*graph).clone());
    loop {
        // Duration bound (when set) is checked before claiming an instance index so
        // an expired lane neither wastes an index nor starts another trace.
        if let Some(g) = &gate
            && g.expired()
        {
            break;
        }
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
            ExecutorFlags::default(),
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

#[cfg(test)]
mod gate_tests {
    use super::*;
    use crate::clock::sim_clock::SimClock;

    // The gate wires the shared `aiperf-timing` duration condition to the worker
    // clock: not expired before the bound, expired once elapsed reaches it.
    #[test]
    fn duration_gate_expires_at_the_bound() {
        let clock = Rc::new(SimClock::new());
        let gate = DurationGate {
            checker: StopChecker::new(&StopConfig {
                expected_duration_ns: Some(1_000),
                ..Default::default()
            }),
            state: RunState {
                started_at_ns: clock.now_ns(),
                ..Default::default()
            },
            clock: clock.clone(),
        };
        assert!(!gate.expired(), "fresh gate must admit work at t=0");
        clock.advance_to(999);
        assert!(!gate.expired(), "still admits just under the bound");
        clock.advance_to(1_000);
        assert!(
            gate.expired(),
            "stops once elapsed reaches the duration bound"
        );
    }
}
