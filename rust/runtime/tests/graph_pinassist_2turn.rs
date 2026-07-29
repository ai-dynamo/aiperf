// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Runs our "what Cristian really wants" 2-turn chained PinAssist
//! (`fixtures/pinassist_2turn.yaml`) through the REAL conditional-graph compiler
//! and executor: parallel safety guard, conditional routing, replay tool steps,
//! per-turn think-time, and `terminal_for_user` markers. Proves it compiles
//! (the multi-turn chaining), dispatches over the real mock, and lets us read
//! user-facing latency (to `terminal_for_user`) apart from full-trace latency.

use std::cell::RefCell;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use anyhow::Result;
use async_trait::async_trait;
use bytes::Bytes;
use uuid::Uuid;

use aiperf_runtime::clock::clock::Clock;
use aiperf_runtime::clock::real_clock::RealClock;
use aiperf_runtime::dataset::TiktokenTokenizer;
use aiperf_runtime::dataset::loader::{DatasetSource, LoadConfig};
use aiperf_runtime::dispatch::collector::ReplayTerminalStatus;
use aiperf_runtime::dispatch::sink::RequestObserver;
use aiperf_runtime::graph::conditional::compile_conditional_graph_input;
use aiperf_runtime::graph::input::GraphInputConfig;
use aiperf_runtime::graph::materialize::SegmentItemsMaterializer;
use aiperf_runtime::graph::model::{GraphRecord, TraceRecord};
use aiperf_runtime::graph::run::{TimeBase, run_trace};
use aiperf_runtime::graph::segment::SegmentStore;
use aiperf_runtime::graph::sink::{GraphReply, GraphSink};
use aiperf_runtime::graph::transport_sink::TransportChatSink;
use aiperf_runtime::graph::wire::{OpenAiChatMessage as Msg, WireMessage};

const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/pinassist_2turn.yaml"
);
const FIXTURE_SLOW: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/pinassist_2turn_slow_safety.yaml"
);
const TRACE_ID: &str = "t-2turn-shopping";

/// Run the trace with an in-process recording sink; return
/// (user-facing latency us, full-trace latency us, dispatch order).
fn user_vs_full_latency(fixture: &str) -> (u128, u128, Vec<String>) {
    let (graph, trace, segments, terminals) = compile_plan(fixture);
    let materializer = Rc::new(SegmentItemsMaterializer::new(segments));
    let sink = Rc::new(RecordingSink {
        start: Instant::now(),
        log: RefCell::new(Vec::new()),
    });
    run_trace::<Msg>(
        Rc::new(graph),
        trace,
        materializer,
        sink.clone(),
        TimeBase::Wall,
    )
    .unwrap();
    let log = sink.log.borrow();
    let t = |id: &str| log.iter().find(|(n, _)| n == id).unwrap().1;
    let user_facing = terminals.iter().map(|id| t(id)).max().unwrap();
    let full_trace = log.iter().map(|(_, ts)| *ts).max().unwrap();
    let order = log.iter().map(|(n, _)| n.clone()).collect();
    (user_facing, full_trace, order)
}

/// Compile the fixture and take the single trace's flat graph + terminal set.
fn compile_plan(fixture: &str) -> (GraphRecord, TraceRecord, Arc<dyn SegmentStore>, Vec<String>) {
    let bytes = std::fs::read(fixture).expect("fixture readable");
    let config = GraphInputConfig {
        load: LoadConfig::new(DatasetSource::Bytes(bytes.into())),
        root_limit: None,
    };
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    // Compiling here is the real validation of the multi-turn chaining.
    let bundle = rt
        .block_on(compile_conditional_graph_input(
            config,
            &TiktokenTokenizer::builtin(),
            0,
        ))
        .expect("2-turn chained PinAssist must compile");
    drop(rt);

    let plan = bundle
        .plans
        .into_iter()
        .find(|p| p.trace.id == TRACE_ID)
        .expect("selected trace present");

    let terminals: Vec<String> = plan
        .graph
        .nodes
        .iter()
        .filter(|(_, n)| {
            n.metadata
                .get("terminal_for_user")
                .and_then(|v| v.as_bool())
                .unwrap_or(false)
        })
        .map(|(id, _)| id.clone())
        .collect();

    (plan.graph, plan.trace, bundle.segments, terminals)
}

// ---- mock plumbing (copied from graph_transport_graph.rs) ----
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
        let child = match Command::new(mock_binary())
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
                eprintln!("SKIP: cannot launch mock: {e}");
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
        let mut m = RealMock {
            child,
            base_url: String::new(),
        };
        let _ = m.child.kill();
        None
    }
}
fn mock_binary() -> PathBuf {
    if let Some(p) = std::env::var_os("AIPERF_MOCK_RS_BIN") {
        return PathBuf::from(p);
    }
    let name = format!("aiperf-mock-server{}", std::env::consts::EXE_SUFFIX);
    if let Ok(exe) = std::env::current_exe()
        && let Some(dir) = exe.parent().and_then(|d| d.parent())
    {
        let c = dir.join(&name);
        if c.is_file() {
            return c;
        }
    }
    PathBuf::from(name)
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
    completed: AtomicUsize,
    tokens: AtomicUsize,
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
    }
}

/// In-process sink recording (node_id, wall micros) per dispatch.
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

/// Headline: the 2-turn chained PinAssist compiles and runs end-to-end over the
/// real aiperf-mock-server.
#[test]
fn pinassist_2turn_over_real_mock() {
    let Some(mock) = RealMock::spawn() else {
        return;
    };
    let (graph, trace, segments, terminals) = compile_plan(FIXTURE);
    // Dual-turn user terminals present (proves terminal_for_user survived compile).
    assert!(
        terminals.iter().any(|t| t == "summarize1") && terminals.iter().any(|t| t == "summarize2"),
        "both turns must carry a terminal_for_user summary; got {terminals:?}"
    );

    // LLM nodes dispatched (shopping/safe path, both turns): route,plan,brandmap,
    // summarize + safety, per turn = 5 * 2 = 10. Replay tool steps never dispatch.
    let llm_nodes = graph.nodes.len();
    assert_eq!(
        llm_nodes, 10,
        "10 LLM nodes on the 2-turn shopping/safe path (replay folded out)"
    );

    let materializer = Rc::new(SegmentItemsMaterializer::new(segments));
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

    run_trace(Rc::new(graph), trace, materializer, sink, TimeBase::Wall).unwrap();

    assert_eq!(
        obs.admits.load(Ordering::Relaxed),
        10,
        "10 real HTTP dispatches"
    );
    assert_eq!(obs.completed.load(Ordering::Relaxed), 10, "all completed");
    assert!(
        obs.tokens.load(Ordering::Relaxed) > 0,
        "real tokens observed"
    );
}

/// Semantics: per-node dispatch order + user-facing (`terminal_for_user`) latency
/// vs full-trace latency (which includes the parallel safety guard).
#[test]
fn pinassist_2turn_user_vs_full_latency() {
    let (graph, trace, segments, terminals) = compile_plan(FIXTURE);
    let materializer = Rc::new(SegmentItemsMaterializer::new(segments));
    let sink = Rc::new(RecordingSink {
        start: Instant::now(),
        log: RefCell::new(Vec::new()),
    });

    run_trace::<Msg>(
        Rc::new(graph),
        trace,
        materializer,
        sink.clone(),
        TimeBase::Wall,
    )
    .unwrap();

    let log = sink.log.borrow();
    let dispatched: Vec<&str> = log.iter().map(|(n, _)| n.as_str()).collect();

    // Replay tool steps never hit the sink.
    assert!(
        !dispatched
            .iter()
            .any(|n| n.starts_with("tool_exec") || n.starts_with("preprocess")),
        "replay nodes must not dispatch: {dispatched:?}"
    );
    // Both turns ran; safety ran parallel in each turn.
    for id in [
        "route1",
        "summarize1",
        "safety1",
        "route2",
        "summarize2",
        "safety2",
    ] {
        assert!(
            dispatched.contains(&id),
            "expected {id} to dispatch; got {dispatched:?}"
        );
    }
    // Turn 2 only starts after turn 1's user answer.
    let t = |id: &str| log.iter().find(|(n, _)| n == id).unwrap().1;
    assert!(
        t("route2") > t("summarize1"),
        "turn 2 begins after turn 1's user answer"
    );

    // The two latency numbers.
    let user_facing = terminals.iter().map(|id| t(id)).max().unwrap(); // last user-visible answer
    let full_trace = log.iter().map(|(_, ts)| *ts).max().unwrap(); // last node overall (incl safety)
    eprintln!("dispatch order: {dispatched:?}");
    eprintln!("terminal_for_user nodes: {terminals:?}");
    eprintln!("user-facing latency (to last terminal_for_user) = {user_facing} us");
    eprintln!("full-trace latency  (to last node, incl safety)  = {full_trace} us");
    assert!(
        full_trace >= user_facing,
        "full trace cannot finish before the user answer"
    );
}

/// The two latencies COINCIDE with a fast safety guard but SPLIT when safety is
/// the long pole -- which is exactly when `terminal_for_user` earns its keep.
#[test]
fn pinassist_safety_long_pole_splits_user_vs_full_latency() {
    let (fast_user, fast_full, fast_order) = user_vs_full_latency(FIXTURE);
    let (slow_user, slow_full, slow_order) = user_vs_full_latency(FIXTURE_SLOW);

    eprintln!("FAST safety : user={fast_user}us full={fast_full}us  order={fast_order:?}");
    eprintln!("SLOW safety : user={slow_user}us full={slow_full}us  order={slow_order:?}");
    eprintln!(
        "delta(full-user): fast={}us  slow={}us",
        fast_full.saturating_sub(fast_user),
        slow_full.saturating_sub(slow_user)
    );

    // Fast guard: the user answer IS the tail -> the two latencies coincide.
    assert!(
        fast_full.saturating_sub(fast_user) < 50_000,
        "fast safety: user-facing ~= full-trace (safety is not the waiting factor)"
    );
    // Slow guard (safety2 min_start_delay = 600ms): safety finishes well AFTER
    // the user answer -> full-trace latency is dominated by the guard.
    assert!(
        slow_full > slow_user + 300_000,
        "slow safety must push full-trace >> user-facing; got user={slow_user} full={slow_full}"
    );
    // And the user-facing number is (roughly) unchanged by the slow guard -- the
    // user still gets their answer on the shopping-chain timeline.
    assert!(
        slow_user < fast_user + 100_000,
        "user-facing latency stays on the shopping-chain timeline regardless of the guard"
    );
}
