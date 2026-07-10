// crates/aiperf-graph/tests/transport_graph.rs
//! End-to-end proof that the graph executor dispatches real OpenAI chat
//! completions over HTTP via `TransportChatSink` (backed by `aiperf-transport`)
//! against the REAL `aiperf-mock-rs` binary — not an in-process stub.

use std::process::{Child, Command, Stdio};
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use aiperf_clock::clock::Clock;
use aiperf_clock::real_clock::RealClock;
use aiperf_graph::materialize::SegmentItemsMaterializer;
use aiperf_graph::model::{GraphRecord, TraceRecord};
use aiperf_graph::reducers::ChanVal;
use aiperf_graph::run::{TimeBase, run_trace};
use aiperf_graph::segment::{SegmentPool, SegmentStore};
use aiperf_graph::transport_sink::TransportChatSink;
use aiperf_graph::wire::OpenAiChatMessage as Msg;
use loadgen_core::collector::ReplayTerminalStatus;
use loadgen_core::sink::RequestObserver;
use serde_json::{Value, json};
use uuid::Uuid;

/// A child `aiperf-mock-rs` process on a free port, killed on drop.
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
        let bin =
            std::env::var("AIPERF_MOCK_RS_BIN").unwrap_or_else(|_| "aiperf-mock-rs".to_string());
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
                eprintln!("SKIP: cannot launch {bin}: {e} (set AIPERF_MOCK_RS_BIN)");
                return None;
            }
        };
        // Poll for readiness (up to ~5s).
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

fn two_node_chain(n0_user: &str, n1_user: &str) -> GraphRecord {
    serde_json::from_value(json!({
        "state": {
            "c0": {"type": "messages", "reducer": "add_messages"},
            "c1": {"type": "messages", "reducer": "add_messages"}
        },
        "nodes": {
            "n0": {"node_type": "llm", "prompt": [], "output": "c0",
                   "items": [{"seg": n0_user}]},
            "n1": {"node_type": "llm", "prompt": [], "output": "c1",
                   "inputs": [{"channel": "c0", "count": 1}],
                   "items": [{"splice": "c0"}, {"seg": n1_user}]}
        },
        "edges": [
            {"edge_type": "static", "source": "START", "target": "n0"},
            {"edge_type": "static", "source": "n0", "target": "n1"},
            {"edge_type": "static", "source": "n1", "target": "END"}
        ]
    }))
    .unwrap()
}

#[test]
fn graph_dispatches_over_transport_to_real_mock() {
    let Some(mock) = RealMock::spawn() else {
        return;
    };

    let mut pool: SegmentPool<Msg> = SegmentPool::new();
    let n0_user = pool.add(Msg::new("user", "start"), None);
    let n1_user = pool.add(Msg::new("user", "continue"), None);
    let store: Rc<dyn SegmentStore<Msg>> = Rc::new(pool);
    let materializer = Rc::new(SegmentItemsMaterializer::new(store));

    let obs = Arc::new(CountObs::default());
    let clock: Rc<dyn Clock> = RealClock::new();
    let sink = Rc::new(TransportChatSink::new(
        clock,
        &mock.base_url,
        "gpt2",
        obs.clone(),
        16,
        false,
    ));

    let graph = two_node_chain(&n0_user, &n1_user);
    let trace: TraceRecord = serde_json::from_value(json!({"id": "t-1"})).unwrap();

    let result = run_trace(Rc::new(graph), trace, materializer, sink, TimeBase::Wall).unwrap();

    // Both LLM nodes reached the wire and completed through the shared collector.
    assert_eq!(obs.admits.load(Ordering::Relaxed), 2, "two HTTP dispatches");
    assert_eq!(obs.completed.load(Ordering::Relaxed), 2, "both completed");
    assert!(
        obs.tokens.load(Ordering::Relaxed) > 0,
        "real output tokens observed"
    );

    // n0's real HTTP reply flowed onto its channel as a non-empty assistant msg.
    match result.channels.get("c0") {
        Some(ChanVal::Val(Value::Array(a))) => {
            assert_eq!(a.len(), 1, "c0 holds n0's assistant reply");
            let reply = a[0]["content"].as_str().unwrap_or("");
            assert!(
                !reply.is_empty(),
                "n0 got a non-empty reply from the mock: {reply:?}"
            );
        }
        other => panic!("c0 should hold assistant messages, got {other:?}"),
    }
    // n1 ran after n0 (spliced c0) and also produced a real reply on c1.
    match result.channels.get("c1") {
        Some(ChanVal::Val(Value::Array(a))) => {
            assert_eq!(a.len(), 1, "c1 holds n1's assistant reply");
            assert!(!a[0]["content"].as_str().unwrap_or("").is_empty());
        }
        other => panic!("c1 should hold assistant messages, got {other:?}"),
    }
}

#[test]
fn graph_dispatches_over_transport_h2c_to_real_mock() {
    let Some(mock) = RealMock::spawn() else {
        return;
    };

    let mut pool: SegmentPool<Msg> = SegmentPool::new();
    let n0_user = pool.add(Msg::new("user", "start"), None);
    let n1_user = pool.add(Msg::new("user", "continue"), None);
    let store: Rc<dyn SegmentStore<Msg>> = Rc::new(pool);
    let materializer = Rc::new(SegmentItemsMaterializer::new(store));

    let obs = Arc::new(CountObs::default());
    let clock: Rc<dyn Clock> = RealClock::new();
    // http2 = true -> h2c prior-knowledge over cleartext against the mock.
    let sink = Rc::new(TransportChatSink::new(
        clock,
        &mock.base_url,
        "gpt2",
        obs.clone(),
        16,
        true,
    ));

    let graph = two_node_chain(&n0_user, &n1_user);
    let trace: TraceRecord = serde_json::from_value(json!({"id": "t-2"})).unwrap();

    let result = run_trace(Rc::new(graph), trace, materializer, sink, TimeBase::Wall).unwrap();

    assert_eq!(
        obs.completed.load(Ordering::Relaxed),
        2,
        "both completed over h2c"
    );
    assert!(
        obs.tokens.load(Ordering::Relaxed) > 0,
        "real output tokens over h2c"
    );
    assert!(
        matches!(result.channels.get("c0"), Some(ChanVal::Val(Value::Array(a))) if !a.is_empty())
    );
}
