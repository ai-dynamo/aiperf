// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Run entrypoint: drive one graph trace's DAG to completion through a
//! [`GraphSink`], returning its final channel snapshot.
//!
//! Live runs use the real ([`RealClock`](aiperf_clock::real_clock::RealClock)) driver so
//! completion-anchored firing gates see real dispatch latency; sim runs use the
//! ns-exact virtual driver.

use std::cell::RefCell;
use std::rc::Rc;

use crate::errors::TraceError;
use crate::executor::{ExecutorFlags, TraceExecutor, TraceResult};
use crate::materialize::PromptMaterializer;
use crate::model::{GraphRecord, TraceRecord};
use crate::runtime::{drive_real, drive_sim};
use crate::sink::GraphSink;
use crate::wire::WireMessage;
use aiperf_clock::sim_clock::SimClock;

/// Which time base the run uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimeBase {
    /// Real wall-clock (ns `timerfd`); dispatches over the wire in real time.
    Wall,
    /// Virtual ns-exact time; timers cost nothing (fast, deterministic).
    Sim,
}

/// Drive one trace's DAG to completion and return its channel snapshot.
pub fn run_trace<M: WireMessage>(
    graph: Rc<GraphRecord>,
    trace: TraceRecord,
    materializer: Rc<dyn PromptMaterializer<M>>,
    sink: Rc<dyn GraphSink<M>>,
    time_base: TimeBase,
) -> Result<TraceResult, TraceError> {
    // Fail fast on structural problems that would otherwise deadlock/hang.
    if let Some(err) = crate::validate::validate(&graph).into_iter().next() {
        return Err(TraceError::Other(err.0));
    }

    let out: Rc<RefCell<Option<Result<TraceResult, TraceError>>>> = Rc::new(RefCell::new(None));

    let body = {
        let out = out.clone();
        move |handle: crate::runtime::Handle| async move {
            let built = TraceExecutor::new(
                graph,
                materializer,
                sink,
                handle.clone(),
                ExecutorFlags::default(),
            )
            .and_then(|exec| {
                let ctx = exec.build_context(trace)?;
                Ok((exec, ctx))
            });
            match built {
                Ok((exec, ctx)) => {
                    exec.schedule_entries(&ctx);
                    handle.wait_idle().await;
                    let result = ctx
                        .abort
                        .borrow()
                        .clone()
                        .map(Err)
                        .unwrap_or_else(|| TraceExecutor::<M>::result(&ctx));
                    *out.borrow_mut() = Some(result);
                }
                Err(e) => *out.borrow_mut() = Some(Err(e)),
            }
        }
    };

    let outcome = match time_base {
        TimeBase::Wall => drive_real(body),
        TimeBase::Sim => drive_sim(Rc::new(SimClock::new()), body),
    };

    out.borrow_mut().take().unwrap_or_else(|| {
        // The body never produced a result: on the Sim clock this is a detected
        // deadlock (a node parked on a channel that will never be written — e.g.
        // a self-dependency, or a producer unreachable from START). The Wall
        // clock cannot detect this and would hang, so untrusted graphs should be
        // dry-run on Sim first.
        if outcome.deadlocked {
            Err(TraceError::Other(
                "trace deadlocked: a node is waiting on a channel that is never written".into(),
            ))
        } else {
            Err(TraceError::Other("run produced no result".into()))
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::materialize::SegmentItemsMaterializer;
    use crate::reducers::ChanVal;
    use crate::segment::{SegmentPool, SegmentStore};
    use crate::sink::EchoSink;
    use crate::wire::OpenAiChatMessage as Msg;
    use serde_json::{Value, json};

    /// Two-node chain: n1 splices n0's reply. Proves dependency ordering + real
    /// content flow through channels via the segment store, no server.
    #[test]
    fn chain_flows_reply_into_successor_prompt() {
        let mut pool: SegmentPool<Msg> = SegmentPool::new();
        let n0_user = pool.add(Msg::new("user", "start"), None);
        let n1_user = pool.add(Msg::new("user", "continue"), None);
        let store: Rc<dyn SegmentStore<Msg>> = Rc::new(pool);
        let materializer = Rc::new(SegmentItemsMaterializer::new(store));
        let sink = Rc::new(EchoSink);

        let graph: GraphRecord = serde_json::from_value(json!({
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
        .unwrap();

        let trace: TraceRecord = serde_json::from_value(json!({"id": "t-1"})).unwrap();
        let result = run_trace(Rc::new(graph), trace, materializer, sink, TimeBase::Sim).unwrap();

        // n0's echoed reply landed on c0.
        let c0 = result.channels.get("c0").unwrap();
        let ChanVal::Val(Value::Array(c0_msgs)) = c0 else {
            panic!("c0 should hold assistant messages, got {c0:?}");
        };
        assert_eq!(c0_msgs.len(), 1);
        let n0_reply = c0_msgs[0]["content"].as_str().unwrap();
        assert!(
            n0_reply.contains("user"),
            "n0 reply echoes its user turn: {n0_reply}"
        );

        // n1's reply must reflect that it SAW n0's reply spliced before its turn.
        let c1 = result.channels.get("c1").unwrap();
        let ChanVal::Val(Value::Array(c1_msgs)) = c1 else {
            panic!("c1 should hold assistant messages");
        };
        let n1_reply = c1_msgs[0]["content"].as_str().unwrap();
        // EchoSink echoes the LAST message (n1's own user turn), proving n1 ran
        // after n0 with n0's reply present in its materialized prompt.
        assert!(n1_reply.contains("continue"), "n1 saw its turn: {n1_reply}");
    }

    /// The same DAG dispatched over REAL HTTP to the mock chat server, proving
    /// the live path (HttpChatSink -> stream_chat -> collector) works end-to-end.
    #[test]
    fn chain_dispatches_over_http() {
        use crate::sink::HttpChatSink;
        use aiperf_core::http_sink::HttpSink;
        use loadgen_core::collector::ReplayTerminalStatus;
        use loadgen_core::sink::RequestObserver;
        use std::sync::Arc;
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::time::Instant;
        use uuid::Uuid;

        #[derive(Default)]
        struct CountObs {
            admits: AtomicUsize,
            completed: AtomicUsize,
        }
        impl RequestObserver for CountObs {
            fn on_arrival(&self, _: Uuid, _: f64, _: usize, _: usize) {}
            fn on_admit(&self, _: Uuid, _: f64, _: usize) {
                self.admits.fetch_add(1, Ordering::Relaxed);
            }
            fn on_token(&self, _: Uuid, _: f64) {}
            fn on_terminal(&self, _: Uuid, s: ReplayTerminalStatus) {
                if matches!(s, ReplayTerminalStatus::Completed) {
                    self.completed.fetch_add(1, Ordering::Relaxed);
                }
            }
        }

        // Host the mock chat server on its own multi-thread runtime.
        let server_rt = tokio::runtime::Runtime::new().unwrap();
        let base = server_rt.block_on(async { crate::test_util::spawn_mock().await });

        let mut pool: SegmentPool<Msg> = SegmentPool::new();
        let n0_user = pool.add(Msg::new("user", "start"), None);
        let n1_user = pool.add(Msg::new("user", "continue"), None);
        let store: Rc<dyn SegmentStore<Msg>> = Rc::new(pool);
        let materializer = Rc::new(SegmentItemsMaterializer::new(store));

        let obs = Arc::new(CountObs::default());
        let http = Arc::new(HttpSink::new(base, "mock-model".into(), Instant::now()));
        let sink = Rc::new(HttpChatSink::new(http, obs.clone(), 16));

        let graph: GraphRecord = serde_json::from_value(json!({
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
        .unwrap();
        let trace: TraceRecord = serde_json::from_value(json!({"id": "t-1"})).unwrap();

        let result = run_trace(Rc::new(graph), trace, materializer, sink, TimeBase::Wall).unwrap();

        // Both nodes reached the wire and completed through the shared collector.
        assert_eq!(obs.admits.load(Ordering::Relaxed), 2, "two HTTP dispatches");
        assert_eq!(obs.completed.load(Ordering::Relaxed), 2, "both completed");
        // n0's HTTP reply flowed onto its channel.
        assert!(
            matches!(result.channels.get("c0"), Some(ChanVal::Val(Value::Array(a))) if !a.is_empty())
        );
    }
}
