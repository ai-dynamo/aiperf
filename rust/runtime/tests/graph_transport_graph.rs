// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! End-to-end graph dispatch over HTTP through `TransportChatSink`.

use std::io::Read;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use aiperf_runtime::clock::clock::Clock;
use aiperf_runtime::clock::real_clock::RealClock;
use aiperf_runtime::dataset::TiktokenTokenizer;
use aiperf_runtime::dataset::loader::{DatasetSource, LoadConfig};
use aiperf_runtime::graph::execution::LocalGraphTraceExecutionBackend;
use aiperf_runtime::graph::input::{GraphInputConfig, compile_dag_jsonl_input};
use aiperf_runtime::graph::materialize::SegmentItemsMaterializer;
use aiperf_runtime::graph::model::{GraphRecord, TraceRecord};
use aiperf_runtime::graph::policy::{
    AbortTraceNodeFailurePolicy, CancellationNodePolicy, FailFastRunFailurePolicy,
};
use aiperf_runtime::graph::reducers::ChanVal;
use aiperf_runtime::graph::run::{TimeBase, run_trace};
use aiperf_runtime::graph::segment::{Handle as SegmentHandle, SegmentPool, intern_message};
use aiperf_runtime::graph::transport_sink::TransportChatSink;
use aiperf_runtime::graph::wire::OpenAiChatMessage as Msg;
use aiperf_runtime::graph::workload::{
    GraphTracePlan, GraphTraceSource, GraphWorkload, VecGraphTraceSource,
};
use aiperf_runtime::rng::RngRoot;
use aiperf_runtime::timing::{BernoulliFixedDelay, Phase};
use loadgen_core::collector::ReplayTerminalStatus;
use loadgen_core::sink::RequestObserver;
use serde_json::{Value, json};
use uuid::Uuid;

/// A child `aiperf-mock-server` process on a free port, killed on drop.
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

/// One accepted TCP request whose response is intentionally withheld so the
/// graph's post-send cancellation policy must terminate it.
struct HangingHttpServer {
    base_url: String,
}

/// Minimal real TCP/OpenAI-SSE server used when the external mock binary is not
/// installed. Each response closes its connection, exercising transport
/// re-establishment in a self-contained test.
struct SseHttpServer {
    base_url: String,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl SseHttpServer {
    fn spawn(expected_requests: usize) -> Self {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let thread = std::thread::spawn(move || {
            for _ in 0..expected_requests {
                let (mut stream, _) = listener.accept().unwrap();
                let _ = stream.set_read_timeout(Some(std::time::Duration::from_secs(2)));
                let mut request = [0_u8; 32 * 1024];
                let _ = stream.read(&mut request);
                let body = concat!(
                    "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\n",
                    "data: [DONE]\n\n"
                );
                let response = format!(
                    "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
                    body.len(),
                    body
                );
                use std::io::Write;
                stream.write_all(response.as_bytes()).unwrap();
            }
        });
        Self {
            base_url: format!("http://{address}"),
            thread: Some(thread),
        }
    }
}

impl Drop for SseHttpServer {
    fn drop(&mut self) {
        if let Some(thread) = self.thread.take() {
            thread.join().unwrap();
        }
    }
}

impl HangingHttpServer {
    fn spawn() -> Self {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        std::thread::spawn(move || {
            let Ok((mut stream, _)) = listener.accept() else {
                return;
            };
            let _ = stream.set_read_timeout(Some(std::time::Duration::from_secs(1)));
            let mut bytes = [0_u8; 16 * 1024];
            let _ = stream.read(&mut bytes);
            std::thread::sleep(std::time::Duration::from_millis(250));
        });
        Self {
            base_url: format!("http://{address}"),
        }
    }
}

fn two_node_chain(n0_user: SegmentHandle, n1_user: SegmentHandle) -> GraphRecord {
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

    let tokenizer = aiperf_runtime::dataset::TiktokenTokenizer::builtin();
    let mut pool = SegmentPool::new();
    let n0_user = intern_message(&mut pool, &Msg::new("user", "start"), None, &tokenizer).unwrap();
    let n1_user =
        intern_message(&mut pool, &Msg::new("user", "continue"), None, &tokenizer).unwrap();
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

    let graph = two_node_chain(n0_user, n1_user);
    let trace: TraceRecord = serde_json::from_value(json!({"id": "t-1"})).unwrap();

    let result = run_trace(Rc::new(graph), trace, materializer, sink, TimeBase::Wall).unwrap();

    assert_eq!(obs.admits.load(Ordering::Relaxed), 2, "two HTTP dispatches");
    assert_eq!(obs.completed.load(Ordering::Relaxed), 2, "both completed");
    assert!(
        obs.tokens.load(Ordering::Relaxed) > 0,
        "real output tokens observed"
    );

    match result.channels.get("c0").and_then(ChanVal::as_value) {
        Some(Value::Array(a)) => {
            assert_eq!(a.len(), 1, "c0 holds n0's assistant reply");
            let reply = a[0]["content"].as_str().unwrap_or("");
            assert!(
                !reply.is_empty(),
                "n0 got a non-empty reply from the mock: {reply:?}"
            );
        }
        other => panic!("c0 should hold assistant messages, got {other:?}"),
    }
    match result.channels.get("c1").and_then(ChanVal::as_value) {
        Some(Value::Array(a)) => {
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

    let tokenizer = aiperf_runtime::dataset::TiktokenTokenizer::builtin();
    let mut pool = SegmentPool::new();
    let n0_user = intern_message(&mut pool, &Msg::new("user", "start"), None, &tokenizer).unwrap();
    let n1_user =
        intern_message(&mut pool, &Msg::new("user", "continue"), None, &tokenizer).unwrap();
    let materializer = Rc::new(SegmentItemsMaterializer::new(Arc::new(pool.freeze())));

    let obs = Rc::new(CountObs::default());
    let clock: Rc<dyn Clock> = RealClock::new();
    let sink = Rc::new(TransportChatSink::new(
        clock,
        &mock.base_url,
        "gpt2",
        obs.clone(),
        16,
        true,
    ));

    let graph = two_node_chain(n0_user, n1_user);
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
        matches!(result.channels.get("c0").and_then(ChanVal::as_value), Some(Value::Array(a)) if !a.is_empty())
    );
}

#[test]
fn lowered_dataset_dag_dispatches_fanout_join_over_real_http() {
    let server = SseHttpServer::spawn(3);
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let bundle = runtime.block_on(async {
        compile_dag_jsonl_input(
            GraphInputConfig {
                load: LoadConfig::new(DatasetSource::Inline(json!([
                    {"session_id":"root","turns":[
                        {"messages":[{"role":"user","content":"root-0"}],"spawns":["child"]},
                        {"messages":[{"role":"user","content":"root-1"}]}
                    ]},
                    {"session_id":"child","turns":[
                        {"messages":[{"role":"user","content":"child-0"}]}
                    ]}
                ]))),
                root_limit: None,
            },
            &TiktokenTokenizer::builtin(),
        )
        .await
        .unwrap()
    });
    drop(runtime);
    let plan = bundle.plans.into_iter().next().unwrap();
    let trace = plan.trace;
    let graph = Rc::new(plan.graph);
    let materializer = Rc::new(SegmentItemsMaterializer::new(bundle.segments));
    let observer = Rc::new(CountObs::default());
    let clock: Rc<dyn Clock> = RealClock::new();
    let sink = Rc::new(TransportChatSink::new(
        clock,
        &server.base_url,
        "gpt2",
        observer.clone(),
        8,
        false,
    ));

    let result = run_trace(graph, trace, materializer, sink, TimeBase::Wall).unwrap();
    assert_eq!(observer.admits.load(Ordering::Relaxed), 3);
    assert_eq!(observer.completed.load(Ordering::Relaxed), 3);
    assert!(
        result
            .channels
            .values()
            .filter_map(ChanVal::as_value)
            .any(|value| matches!(value, Value::Array(messages) if !messages.is_empty()))
    );
}

#[test]
fn graph_policy_cancels_real_http_after_send_without_failing_the_run() {
    let server = HangingHttpServer::spawn();
    let tokenizer = aiperf_runtime::dataset::TiktokenTokenizer::builtin();
    let mut pool = SegmentPool::new();
    let user = intern_message(&mut pool, &Msg::new("user", "cancel"), None, &tokenizer).unwrap();
    let materializer = Rc::new(SegmentItemsMaterializer::new(Arc::new(pool.freeze())));
    let graph: GraphRecord = serde_json::from_value(json!({
        "state": {"out": {"type": "messages", "reducer": "add_messages"}},
        "nodes": {"n0": {"output": "out", "items": [{"seg": user}]}},
        "edges": [{"source": "START", "target": "n0"}]
    }))
    .unwrap();
    let source: Rc<dyn GraphTraceSource> = Rc::new(VecGraphTraceSource::new([GraphTracePlan {
        graph,
        trace: TraceRecord {
            id: "cancel-real".into(),
            graph_ref: None,
            initial_state: Default::default(),
        },
        arrival_offset_ns: None,
    }]));
    let observer = Rc::new(CountObs::default());
    let clock: Rc<dyn Clock> = RealClock::new();
    let sink = Rc::new(TransportChatSink::new(
        clock.clone(),
        &server.base_url,
        "gpt2",
        observer.clone(),
        16,
        false,
    ));
    let cancellation = Rc::new(CancellationNodePolicy::new(
        Box::new(BernoulliFixedDelay::new(Some(100.0), 0.001, RngRoot::new(Some(11))).unwrap()),
        Phase::Profiling,
    ));
    let backend = LocalGraphTraceExecutionBackend::new(clock.clone(), materializer, sink)
        .with_node_policy(cancellation)
        .with_node_failure(Rc::new(AbortTraceNodeFailurePolicy));
    let workload = GraphWorkload::new(clock, source, Rc::new(backend))
        .with_run_failure(Rc::new(FailFastRunFailurePolicy::default()));
    let report = Rc::new(std::cell::RefCell::new(None));
    let report_slot = report.clone();
    let outcome = aiperf_runtime::graph::runtime::drive_real(move |_handle| async move {
        *report_slot.borrow_mut() = Some(workload.execute().await.unwrap());
    });
    assert!(!outcome.deadlocked);
    let report = report.borrow_mut().take().unwrap();
    assert_eq!(report.admitted, 1);
    assert_eq!(report.cancelled, 1);
    assert_eq!(report.failed, 0);
    assert_eq!(observer.canceled.load(Ordering::Relaxed), 1);
    assert_eq!(observer.completed.load(Ordering::Relaxed), 0);
}
