// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Throughput benchmark for graph-IR mode with **real segments**.
//!
//! Builds a multi-turn conversation as a content-addressed segment pool
//! (system + growing user turns, prefix-chained so the static prefix dedups
//! across every trace instance), then fans out many trace instances across all
//! cores (thread-per-core: `workers` OS threads, each a `current_thread` tokio
//! runtime + `LocalSet` running `concurrency` traces at once). Each node's
//! prompt is materialized from the segment store + spliced predecessor replies,
//! dispatched over HTTP to `aiperf-mock-rs --fast`, and measured through the
//! shared `TraceCollector` via the batched `submit` (one lock per request —
//! lock-free token accumulation, the same hot-path shape the driven path uses).

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use anyhow::Result;
use async_trait::async_trait;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use uuid::Uuid;

use loadgen_core::collector::{ReplayTerminalStatus, TraceSimulationReport};

use crate::executor::TraceExecutor;
use crate::materialize::SegmentItemsMaterializer;
use crate::model::{GraphRecord, TraceRecord};
use crate::runtime::Handle;
use crate::segment::{SegmentPool, SegmentStore};
use crate::sink::{GraphReply, GraphSink};
use crate::wire::OpenAiChatMessage as Msg;
use aiperf_clock::real_clock::RealClock;
use aiperf_core::http_sink::{ChatMessage, HttpSink};
use aiperf_core::observer::CollectorObserver;

/// Benchmark configuration.
pub struct BenchConfig {
    /// One or more server base URLs; workers round-robin across them so a
    /// high-concurrency run fans connections out under each server's accept
    /// backlog (and each server's own ephemeral-port space).
    pub base_urls: Vec<String>,
    pub model: String,
    /// Turns per conversation (nodes per trace DAG).
    pub turns: usize,
    /// Total trace instances (conversations) to run.
    pub instances: usize,
    /// OS worker threads (thread-per-core).
    pub workers: usize,
    /// Concurrent traces in flight per worker.
    pub concurrency: usize,
    /// Requested output tokens per node.
    pub max_tokens: usize,
    /// Global cap on concurrent in-flight requests across all workers/lanes.
    /// `None` = unbounded (limited only by trace-lane concurrency + fan-out).
    pub request_concurrency: Option<usize>,
    /// Global cap on concurrent requests in the prefill phase (dispatched but
    /// not yet first-token). `None` = unbounded.
    pub prefill_concurrency: Option<usize>,
}

/// Metered OpenAI-chat sink: streams over HTTP and records one batched `submit`
/// per request (per-node ISL precomputed, so there is zero hot-path tokenization).
struct MeteredSink {
    http: Arc<HttpSink>,
    obs: Arc<CollectorObserver>,
    isl_by_node: HashMap<String, usize>,
    max_tokens: usize,
    /// Global request-concurrency cap (held for the whole request).
    request_sem: Option<Arc<Semaphore>>,
    /// Global prefill-phase cap (held from dispatch until first token).
    prefill_sem: Option<Arc<Semaphore>>,
}

#[async_trait(?Send)]
impl GraphSink<Msg> for MeteredSink {
    async fn dispatch(
        &self,
        node_id: &str,
        messages: Vec<Msg>,
        max_tokens: Option<usize>,
        on_first_token: &dyn Fn(),
    ) -> Result<GraphReply<Msg>> {
        // Request-level cap: hold a permit for the whole request. Acquired
        // before `arrival_ms` so slot-wait is queueing, not request latency.
        let _req_permit: Option<OwnedSemaphorePermit> = match &self.request_sem {
            Some(sem) => Some(sem.clone().acquire_owned().await?),
            None => None,
        };
        // Prefill-phase cap: hold a permit until the first token arrives.
        let prefill_permit: RefCell<Option<OwnedSemaphorePermit>> =
            RefCell::new(match &self.prefill_sem {
                Some(sem) => Some(sem.clone().acquire_owned().await?),
                None => None,
            });

        let arrival_ms = self.obs.now_ms();
        let uuid = Uuid::new_v4();
        let chat: Vec<ChatMessage> = messages
            .into_iter()
            .map(|m| ChatMessage {
                role: m.role,
                content: m.content,
            })
            .collect();
        let mot = max_tokens.unwrap_or(self.max_tokens);
        // Release the prefill permit at the first token (prefill -> decode).
        let on_ft = || {
            prefill_permit.borrow_mut().take();
            on_first_token();
        };
        let outcome = self.http.stream_chat_cb(uuid, &chat, mot, on_ft).await?;
        let isl = self.isl_by_node.get(node_id).copied().unwrap_or(1);
        self.obs.submit(uuid, arrival_ms, isl, mot, &outcome);
        Ok(match outcome.terminal {
            ReplayTerminalStatus::Completed => GraphReply::from_text(outcome.content),
            _ => GraphReply::failed(),
        })
    }
}

/// A rough token estimate for a static segment's content (no tokenizer needed on
/// the hot path; segment content is fixed, so this is computed once at setup).
fn est_tokens(s: &str) -> usize {
    (s.len() / 4).max(1)
}

/// Build the segment pool + chain graph for a `turns`-turn conversation, and the
/// per-node ISL map. The static prefix (system + user turns) is content-addressed
/// and shared across every trace instance.
pub(crate) fn build_workload(
    turns: usize,
) -> (SegmentPool<Msg>, GraphRecord, HashMap<String, usize>) {
    let mut pool: SegmentPool<Msg> = SegmentPool::new();
    let sys_text = "You are a helpful, concise assistant answering benchmark turns.";
    let sys = pool.add(Msg::new("system", sys_text), None);

    // Prefix-chained user-turn segments.
    let mut parent = sys.clone();
    let mut user_segs = Vec::with_capacity(turns);
    let mut user_tokens = Vec::with_capacity(turns);
    for k in 0..turns {
        let text = format!("Turn {k}: please continue the conversation about topic {k}.");
        let tok = est_tokens(&text);
        let seg = pool.add(Msg::new("user", &text), Some(&parent));
        parent = seg.clone();
        user_segs.push(seg);
        user_tokens.push(tok);
    }
    let sys_tok = est_tokens(sys_text);

    // Chain graph: n0 -> n1 -> ... ; node k gates on the prior node's channel and
    // splices every prior reply, so its prompt is the full growing conversation.
    let mut state = serde_json::Map::new();
    let mut nodes = serde_json::Map::new();
    let mut edges = vec![serde_json::json!({"edge_type":"static","source":"START","target":"n0"})];
    let mut isl_by_node = HashMap::new();

    for k in 0..turns {
        state.insert(
            format!("c{k}"),
            serde_json::json!({"type":"messages","reducer":"add_messages"}),
        );
        // items: sys, [u0, splice c0, u1, splice c1, ..., u_{k-1}, splice c_{k-1}], u_k
        let mut items = vec![serde_json::json!({"seg": sys})];
        let mut isl = sys_tok;
        for j in 0..k {
            items.push(serde_json::json!({"seg": user_segs[j]}));
            items.push(serde_json::json!({"splice": format!("c{j}")}));
            isl += user_tokens[j] + 1; // +1 for the spliced 1-token reply
        }
        items.push(serde_json::json!({"seg": user_segs[k]}));
        isl += user_tokens[k];
        isl_by_node.insert(format!("n{k}"), isl);

        let mut node = serde_json::json!({
            "node_type":"llm","prompt":[],"output":format!("c{k}"),"items":items
        });
        if k > 0 {
            node["inputs"] = serde_json::json!([{"channel": format!("c{}", k-1), "count": 1}]);
            edges.push(serde_json::json!({"edge_type":"static","source":format!("n{}",k-1),"target":format!("n{k}")}));
        }
        nodes.insert(format!("n{k}"), node);
    }
    edges.push(
        serde_json::json!({"edge_type":"static","source":format!("n{}",turns-1),"target":"END"}),
    );

    let graph: GraphRecord = serde_json::from_value(serde_json::json!({
        "state": state, "nodes": nodes, "edges": edges
    }))
    .expect("valid bench graph");
    (pool, graph, isl_by_node)
}

/// Raise this process's open-file soft limit to its hard limit, so a
/// high-concurrency run can open tens of thousands of sockets without a shell
/// `ulimit`. No root needed (soft up to hard).
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

/// Run the benchmark and return the aggregated report + elapsed seconds.
pub fn run_bench(cfg: BenchConfig) -> (TraceSimulationReport, f64) {
    raise_fd_limit();
    let (pool, graph, isl_by_node) = build_workload(cfg.turns);
    let graph = Arc::new(graph);
    let pool = Arc::new(pool);
    let isl = Arc::new(isl_by_node);

    let start = Instant::now();
    let obs = Arc::new(CollectorObserver::new(start, false));
    let next = Arc::new(AtomicUsize::new(0));

    // Global caps shared across every worker thread/lane. 0 or None = unbounded.
    let request_sem = cfg
        .request_concurrency
        .filter(|&n| n > 0)
        .map(|n| Arc::new(Semaphore::new(n)));
    let prefill_sem = cfg
        .prefill_concurrency
        .filter(|&n| n > 0)
        .map(|n| Arc::new(Semaphore::new(n)));

    let servers = if cfg.base_urls.is_empty() {
        vec!["http://127.0.0.1:8000".to_string()]
    } else {
        cfg.base_urls.clone()
    };
    std::thread::scope(|scope| {
        for widx in 0..cfg.workers.max(1) {
            let (graph, pool, isl, obs, next) = (
                graph.clone(),
                pool.clone(),
                isl.clone(),
                obs.clone(),
                next.clone(),
            );
            let base_url = servers[widx % servers.len()].clone();
            let model = cfg.model.clone();
            let (turns, instances, concurrency, max_tokens) =
                (cfg.turns, cfg.instances, cfg.concurrency, cfg.max_tokens);
            let (request_sem, prefill_sem) = (request_sem.clone(), prefill_sem.clone());
            scope.spawn(move || {
                worker(
                    &graph,
                    &pool,
                    &isl,
                    &obs,
                    &next,
                    &base_url,
                    &model,
                    start,
                    turns,
                    instances,
                    concurrency,
                    max_tokens,
                    request_sem,
                    prefill_sem,
                );
            });
        }
    });

    let wall_ms = start.elapsed().as_secs_f64() * 1000.0;
    let report = obs.finish(wall_ms);
    (report, start.elapsed().as_secs_f64())
}

#[allow(clippy::too_many_arguments)]
fn worker(
    graph: &Arc<GraphRecord>,
    pool: &Arc<SegmentPool<Msg>>,
    isl: &Arc<HashMap<String, usize>>,
    obs: &Arc<CollectorObserver>,
    next: &Arc<AtomicUsize>,
    base_url: &str,
    model: &str,
    start: Instant,
    _turns: usize,
    instances: usize,
    concurrency: usize,
    max_tokens: usize,
    request_sem: Option<Arc<Semaphore>>,
    prefill_sem: Option<Arc<Semaphore>>,
) {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("worker runtime");
    let local = tokio::task::LocalSet::new();

    // Per-worker shared HTTP client + metered sink + materializer (all cheap to clone).
    let http = Arc::new(HttpSink::new(
        base_url.to_string(),
        model.to_string(),
        start,
    ));
    let sink: Rc<dyn GraphSink<Msg>> = Rc::new(MeteredSink {
        http,
        obs: obs.clone(),
        isl_by_node: (**isl).clone(),
        max_tokens,
        request_sem,
        prefill_sem,
    });
    let store: Rc<dyn SegmentStore<Msg>> = {
        let pool = (**pool).clone();
        Rc::new(pool)
    };
    let materializer = Rc::new(SegmentItemsMaterializer::new(store));

    local.block_on(&rt, async {
        let mut lanes = Vec::with_capacity(concurrency);
        for _ in 0..concurrency.max(1) {
            lanes.push(run_lane(
                graph.clone(),
                materializer.clone(),
                sink.clone(),
                next.clone(),
                instances,
            ));
        }
        futures::future::join_all(lanes).await;
    });
}

async fn run_lane(
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
