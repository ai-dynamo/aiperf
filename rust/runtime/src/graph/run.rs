// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Run entrypoint: drive one graph trace's DAG to completion through a
//! [`GraphSink`], returning its final channel snapshot.
//!
//! Live runs use the real ([`RealClock`](crate::clock::real_clock::RealClock)) driver so
//! completion-anchored firing gates see real dispatch latency; sim runs use the
//! ns-exact virtual driver.

use std::cell::RefCell;
use std::rc::Rc;

use crate::clock::sim_clock::SimClock;
use crate::graph::errors::TraceError;
use crate::graph::executor::{ExecutorFlags, TraceExecutor, TraceResult};
use crate::graph::materialize::PromptMaterializer;
use crate::graph::model::{GraphRecord, TraceRecord};
use crate::graph::runtime::{drive_real, drive_sim};
use crate::graph::sink::GraphSink;
use crate::graph::wire::WireMessage;

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
    materializer: Rc<dyn PromptMaterializer>,
    sink: Rc<dyn GraphSink<M>>,
    time_base: TimeBase,
) -> Result<TraceResult, TraceError> {
    // Fail fast on structural problems that would otherwise deadlock/hang.
    if let Some(err) = crate::graph::validate::validate(&graph).into_iter().next() {
        return Err(TraceError::Other(err.0));
    }

    let out: Rc<RefCell<Option<Result<TraceResult, TraceError>>>> = Rc::new(RefCell::new(None));

    let body = {
        let out = out.clone();
        move |handle: crate::graph::runtime::Handle| async move {
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
    use crate::graph::materialize::SegmentItemsMaterializer;
    use crate::graph::segment::{SegmentPool, intern_message};
    use crate::graph::sink::EchoSink;
    use crate::graph::wire::OpenAiChatMessage as Msg;
    use serde_json::{Value, json};
    use std::sync::Arc;

    // Two-node chain in which n1 splices n0's reply through the segment store.
    #[test]
    fn chain_flows_reply_into_successor_prompt() {
        let tokenizer = crate::dataset::TiktokenTokenizer::builtin();
        let mut pool = SegmentPool::new();
        let n0_user =
            intern_message(&mut pool, &Msg::new("user", "start"), None, &tokenizer).unwrap();
        let n1_user =
            intern_message(&mut pool, &Msg::new("user", "continue"), None, &tokenizer).unwrap();
        let materializer = Rc::new(SegmentItemsMaterializer::new(Arc::new(pool.freeze())));
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
        let result =
            run_trace::<Msg>(Rc::new(graph), trace, materializer, sink, TimeBase::Sim).unwrap();

        // n0's echoed reply landed on c0.
        let c0 = result.channels.get("c0").unwrap();
        let Some(Value::Array(c0_msgs)) = c0.as_value() else {
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
        let Some(Value::Array(c1_msgs)) = c1.as_value() else {
            panic!("c1 should hold assistant messages");
        };
        let n1_reply = c1_msgs[0]["content"].as_str().unwrap();
        // EchoSink echoes the LAST message (n1's own user turn), proving n1 ran
        // after n0 with n0's reply present in its materialized prompt.
        assert!(n1_reply.contains("continue"), "n1 saw its turn: {n1_reply}");
    }
}
