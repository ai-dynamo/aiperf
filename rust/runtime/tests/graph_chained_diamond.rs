// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Chained-diamond E2E (customer benchmark shape).
//!
//! Each "root" sends NO request, applies its own think-time, then fans out two
//! parallel branches; the NEXT root AND-waits for BOTH branch tails before its
//! think-time and next fan-out:
//!
//! ```text
//! root1 (no request; think1) -> A1, B1
//!                             -> root2 (AND-wait A1,B1; think2) -> A2, B2
//!                             -> root3 (AND-wait A2,B2; think3) -> A3, B3 ...
//! ```
//!
//! A flat-runtime node either dispatches an inference request (`LlmNode`) or
//! produces a tool observation (`ToolNode`); there is no request-less control
//! node, so the roots are NOT materialized as nodes. Instead each level's two
//! branch nodes carry:
//!   * `inputs` requiring BOTH prior-level tail channels  -> the root's AND-barrier
//!   * `min_start_delay_us` = that root's think-time       -> applied after the barrier
//! Roots therefore emit zero requests by construction.

use std::cell::RefCell;
use std::process::{Child, Command, Stdio};
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use anyhow::Result;
use async_trait::async_trait;
use bytes::Bytes;
use serde_json::{Map, Value, json};
use uuid::Uuid;

use aiperf_runtime::clock::clock::Clock;
use aiperf_runtime::clock::real_clock::RealClock;
use aiperf_runtime::dataset::TiktokenTokenizer;
use aiperf_runtime::dispatch::collector::ReplayTerminalStatus;
use aiperf_runtime::dispatch::sink::RequestObserver;
use aiperf_runtime::graph::materialize::SegmentItemsMaterializer;
use aiperf_runtime::graph::model::{GraphRecord, TraceRecord};
use aiperf_runtime::graph::run::{TimeBase, run_trace};
use aiperf_runtime::graph::segment::{Handle as SegmentHandle, SegmentPool, intern_message};
use aiperf_runtime::graph::sink::{GraphReply, GraphSink};
use aiperf_runtime::graph::transport_sink::TransportChatSink;
use aiperf_runtime::graph::wire::{OpenAiChatMessage as Msg, WireMessage};

use std::path::PathBuf;

/// A child `aiperf-mock-server` on a free port, killed on drop.
struct RealMock {
    child: Child,
    base_url: String,
}

impl RealMock {
    fn spawn() -> Option<RealMock> {
        let port = {
            let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
            l.local_addr().unwrap().port()
        };
        let bin = mock_binary();
        let child = match Command::new(&bin)
            .arg("--host")
            .arg("127.0.0.1")
            .arg("--port")
            .arg(port.to_string())
            .arg("--no-tokenizer")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
        {
            Ok(c) => c,
            Err(e) => {
                eprintln!(
                    "SKIP: cannot launch {}: {e} (set AIPERF_MOCK_RS_BIN)",
                    bin.display()
                );
                return None;
            }
        };
        for _ in 0..250 {
            if std::net::TcpStream::connect(("127.0.0.1", port)).is_ok() {
                return Some(RealMock {
                    child,
                    base_url: format!("http://127.0.0.1:{port}"),
                });
            }
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
        eprintln!("SKIP: mock did not become ready on port {port}");
        let mut m = RealMock {
            child,
            base_url: String::new(),
        };
        let _ = m.child.kill();
        None
    }
}

fn mock_binary() -> PathBuf {
    if let Some(path) = std::env::var_os("AIPERF_MOCK_RS_BIN") {
        return PathBuf::from(path);
    }
    let binary_name = format!("aiperf-mock-server{}", std::env::consts::EXE_SUFFIX);
    if let Ok(current_exe) = std::env::current_exe()
        && let Some(profile_dir) = current_exe.parent().and_then(|deps_dir| deps_dir.parent())
    {
        let candidate = profile_dir.join(&binary_name);
        if candidate.is_file() {
            return candidate;
        }
    }
    let target_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target");
    for profile in ["debug", "release"] {
        let candidate = target_dir.join(profile).join(&binary_name);
        if candidate.is_file() {
            return candidate;
        }
    }
    PathBuf::from(binary_name)
}

impl Drop for RealMock {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

#[derive(Default)]
struct CountObs {
    admits: AtomicUsize,
    tokens: AtomicUsize,
    completed: AtomicUsize,
    canceled: AtomicUsize,
}
impl RequestObserver for CountObs {
    fn on_arrival(&self, _: Uuid, _: f64, _: usize, _: usize) {}
    fn on_admit(&self, _: Uuid, _: f64, _: usize) {
        self.admits.fetch_add(1, Ordering::Relaxed);
    }
    fn on_token(&self, _: Uuid, _: f64) {
        self.tokens.fetch_add(1, Ordering::Relaxed);
    }
    fn on_terminal(&self, _: Uuid, s: ReplayTerminalStatus) {
        if matches!(s, ReplayTerminalStatus::Completed) {
            self.completed.fetch_add(1, Ordering::Relaxed);
        }
        if matches!(s, ReplayTerminalStatus::Canceled) {
            self.canceled.fetch_add(1, Ordering::Relaxed);
        }
    }
}

/// In-process sink recording per-node dispatch order + wall time (micros).
struct RecordingSink {
    start: Instant,
    log: RefCell<Vec<(String, u128)>>,
}
#[async_trait(?Send)]
impl<M: WireMessage> GraphSink<M> for RecordingSink {
    async fn dispatch(
        &self,
        node_id: &str,
        _messages: Vec<Bytes>,
        _max_tokens: Option<usize>,
        on_first_token: &dyn Fn(),
    ) -> Result<GraphReply<M>> {
        on_first_token();
        self.log
            .borrow_mut()
            .push((node_id.to_string(), self.start.elapsed().as_micros()));
        Ok(GraphReply::from_text(format!("[{node_id}] ok")))
    }
}

/// Build an `levels`-deep chained diamond. `think_us[k]` is root-k's think-time,
/// applied as `min_start_delay_us` on level-k's two branch nodes (after the
/// AND-barrier on both level-(k-1) tails). Roots are never nodes -> 0 requests.
fn build_chained_diamond(levels: usize, think_us: &[f64], user: SegmentHandle) -> GraphRecord {
    let mut state = Map::new();
    let mut nodes = Map::new();
    let mut edges: Vec<Value> = Vec::new();

    for k in 0..levels {
        for side in ["a", "b"] {
            let id = format!("{side}{k}");
            state.insert(
                id.clone(),
                json!({"type": "messages", "reducer": "add_messages"}),
            );

            let mut node = Map::new();
            node.insert("node_type".into(), json!("llm"));
            node.insert("output".into(), json!(id.clone()));
            node.insert("items".into(), json!([{"seg": user}]));
            if think_us[k] > 0.0 {
                node.insert("min_start_delay_us".into(), json!(think_us[k]));
            }

            if k == 0 {
                edges.push(json!({"edge_type": "static", "source": "START", "target": id.clone()}));
            } else {
                // AND-barrier: this branch waits for BOTH prior-level tails.
                node.insert(
                    "inputs".into(),
                    json!([
                        {"channel": format!("a{}", k - 1), "count": 1},
                        {"channel": format!("b{}", k - 1), "count": 1},
                    ]),
                );
                edges.push(json!({"edge_type": "static", "source": format!("a{}", k - 1), "target": id.clone()}));
                edges.push(json!({"edge_type": "static", "source": format!("b{}", k - 1), "target": id.clone()}));
            }
            if k == levels - 1 {
                edges.push(json!({"edge_type": "static", "source": id.clone(), "target": "END"}));
            }
            nodes.insert(id, Value::Object(node));
        }
    }

    serde_json::from_value(json!({"state": state, "nodes": nodes, "edges": edges})).unwrap()
}

/// Headline: the whole chained diamond runs end-to-end over the REAL
/// aiperf-mock-server. Exactly two real HTTP requests per level, zero for roots.
#[test]
fn chained_diamond_over_real_mock() {
    let Some(mock) = RealMock::spawn() else {
        return; // mock binary unavailable in this environment
    };
    let levels = 4;
    let think_us = [0.0, 5_000.0, 5_000.0, 5_000.0];

    let tokenizer = TiktokenTokenizer::builtin();
    let mut pool = SegmentPool::new();
    let user = intern_message(
        &mut pool,
        &Msg::new("user", "shopping query"),
        None,
        &tokenizer,
    )
    .unwrap();
    let materializer = Rc::new(SegmentItemsMaterializer::new(Arc::new(pool.freeze())));

    let obs = Rc::new(CountObs::default());
    let clock: Rc<dyn Clock> = RealClock::new();
    let sink = Rc::new(TransportChatSink::new(
        clock,
        &mock.base_url,
        "gpt2",
        obs.clone(),
        16,
        false,
    ));

    let graph = build_chained_diamond(levels, &think_us, user);
    let trace: TraceRecord = serde_json::from_value(json!({"id": "t-1"})).unwrap();

    run_trace(Rc::new(graph), trace, materializer, sink, TimeBase::Wall).unwrap();

    assert_eq!(
        obs.admits.load(Ordering::Relaxed),
        2 * levels,
        "exactly two real HTTP requests per level, zero for the request-less roots"
    );
    assert_eq!(
        obs.completed.load(Ordering::Relaxed),
        2 * levels,
        "every branch request completed against the real mock"
    );
    assert!(
        obs.tokens.load(Ordering::Relaxed) > 0,
        "real output tokens observed"
    );
}

/// Semantics: the AND-barrier ordering and per-root think-time, observed
/// in-process (deterministic wall gaps, no HTTP noise).
#[test]
fn chained_diamond_barrier_ordering_and_think_time() {
    let levels = 4;
    // root-k think-time (us): 0, 20ms, 40ms, 60ms.
    let think_us = [0.0, 20_000.0, 40_000.0, 60_000.0];

    let tokenizer = TiktokenTokenizer::builtin();
    let mut pool = SegmentPool::new();
    let user = intern_message(&mut pool, &Msg::new("user", "req"), None, &tokenizer).unwrap();
    let materializer = Rc::new(SegmentItemsMaterializer::new(Arc::new(pool.freeze())));

    let sink = Rc::new(RecordingSink {
        start: Instant::now(),
        log: RefCell::new(Vec::new()),
    });
    let graph = build_chained_diamond(levels, &think_us, user);
    let trace: TraceRecord = serde_json::from_value(json!({"id": "t-1"})).unwrap();

    run_trace::<Msg>(
        Rc::new(graph),
        trace,
        materializer,
        sink.clone(),
        TimeBase::Wall,
    )
    .unwrap();

    let log = sink.log.borrow();

    // Exactly two branch dispatches per level, and zero root dispatches (roots
    // are not nodes -> they never touch the sink).
    assert_eq!(
        log.len(),
        2 * levels,
        "two branch requests per level, zero for roots"
    );
    let order: Vec<String> = log.iter().map(|(n, _)| n.clone()).collect();
    assert!(
        !order.iter().any(|n| n.starts_with("root")),
        "no request-less root node ever dispatched: {order:?}"
    );

    let idx = |name: &str| order.iter().position(|n| n == name).unwrap();
    let t = |name: &str| log.iter().find(|(n, _)| n == name).unwrap().1;

    for k in 1..levels {
        let prev = [format!("a{}", k - 1), format!("b{}", k - 1)];
        let cur = [format!("a{}", k), format!("b{}", k)];

        // AND-barrier: BOTH level-(k-1) tails dispatch before EITHER level-k branch.
        let prev_max_idx = prev.iter().map(|n| idx(n)).max().unwrap();
        let cur_min_idx = cur.iter().map(|n| idx(n)).min().unwrap();
        assert!(
            prev_max_idx < cur_min_idx,
            "level {} AND-barrier: both tails must precede level {}",
            k - 1,
            k
        );

        // Think-time: level-k dispatch is delayed ~think_us[k] after both
        // level-(k-1) tails complete (replies are instant, so completion ~= dispatch).
        let prev_done = prev.iter().map(|n| t(n)).max().unwrap();
        let cur_start = cur.iter().map(|n| t(n)).min().unwrap();
        let gap = cur_start.saturating_sub(prev_done);
        assert!(
            gap as f64 >= think_us[k] * 0.7,
            "root {k} think-time honored: observed gap {gap}us vs think {}us",
            think_us[k]
        );
    }
}
