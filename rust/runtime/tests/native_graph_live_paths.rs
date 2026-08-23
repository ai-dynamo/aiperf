// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Model-selected NativeGraph path contracts.
#![cfg(feature = "engine")]

use std::{
    cell::{Cell, RefCell},
    collections::BTreeMap,
    fs,
    path::Path,
    rc::Rc,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
};

use aiperf_runtime::clock::{Clock, SimClock};
use aiperf_runtime::dataset::Handle;
use aiperf_runtime::dataset::InMemorySegmentStore;
use aiperf_runtime::engine::execution_factories::native_execution_factories;
use aiperf_runtime::eval::{
    HarborImporter, HarborSource, NativeGraphControlContract, NativeGraphLiveAgentLoopFactories,
    NativeGraphLiveTraceProgramDriverFactory, NativeSourceAcquirer, lower_native_graph,
};
use aiperf_runtime::graph::agent::{
    AgentInvocationLease, AgentInvocationLeaseFactory, AgentInvocationLeaseFactoryFactory,
    AgentInvocationLeaseOpening, AgentInvocationRequest,
};
use aiperf_runtime::graph::driver::{
    TraceAgentInvocationContext, TraceDriverContext, TraceProgramDriverFactory,
    TraceStageDirective, TraceStageResult, WorkerIdentity,
};
use aiperf_runtime::graph::sink::GraphReplyStatus;
use aiperf_runtime::graph::tools::{InMemoryToolDispatcherFactory, ToolDispatcher};
use async_trait::async_trait;
use bytes::Bytes;
use serde_json::json;

const COUNTERFACTUAL_SOURCE: &str = r#"{
  "schema_version": "1.0",
  "trace_id": "counterfactual-path",
  "stage_bound": 4,
  "channels": {
    "decision": { "type": "messages", "reducer": "add_messages" },
    "observation_a": { "type": "text", "reducer": "overwrite" },
    "observation_b": { "type": "text", "reducer": "overwrite" },
    "selected_observation": { "type": "text", "reducer": "overwrite" },
    "answer": { "type": "messages", "reducer": "add_messages" }
  },
  "nodes": [
    { "id": "route-model", "kind": "model", "binding": "primary", "output": "decision", "streaming": false },
    { "id": "tool-a", "kind": "tool", "adapter": "tool-adapter", "operation": "tool-a", "output": "observation_a" },
    { "id": "tool-b", "kind": "tool", "adapter": "tool-adapter", "operation": "tool-b", "output": "observation_b" },
    { "id": "finish-model", "kind": "model", "binding": "primary", "inputs": ["selected_observation"], "output": "answer", "streaming": false }
  ],
  "edges": [
    { "source": "START", "target": "route-model" },
    { "source": "route-model", "target": "tool-a" },
    { "source": "route-model", "target": "tool-b" },
    { "source": "tool-a", "target": "finish-model" },
    { "source": "tool-b", "target": "finish-model" },
    { "source": "finish-model", "target": "END" }
  ],
  "branches": [{
    "id": "route",
    "selector_node": "route-model",
    "selector_channel": "decision",
    "candidates": [
      {
        "id": "choose-a",
        "match": "choose-a",
        "edge": { "source": "route-model", "target": "tool-a" },
        "nodes": ["tool-a"],
        "channels": ["observation_a"]
      },
      {
        "id": "choose-b",
        "match": "choose-b",
        "edge": { "source": "route-model", "target": "tool-b" },
        "nodes": ["tool-b"],
        "channels": ["observation_b"]
      }
    ]
  }],
  "joins": [{
    "id": "selected-observation",
    "selector": "route",
    "candidates": ["choose-a", "choose-b"],
    "output_channel": "selected_observation",
    "reduction": "selected_candidate"
  }],
  "terminal_outputs": []
}"#;

const LOOP_SOURCE: &str = r#"{
  "schema_version": "1.0",
  "trace_id": "bounded-loop",
  "stage_bound": 5,
  "channels": {
    "decision": { "type": "messages", "reducer": "add_messages" },
    "observation": { "type": "text", "reducer": "overwrite" }
  },
  "nodes": [
    { "id": "decide", "kind": "model", "binding": "primary", "output": "decision", "streaming": false },
    { "id": "attempt", "kind": "tool", "adapter": "tool-adapter", "operation": "attempt", "output": "observation" }
  ],
  "edges": [
    { "source": "START", "target": "decide" },
    { "source": "decide", "target": "attempt" },
    { "source": "attempt", "target": "decide" },
    { "source": "decide", "target": "END" }
  ],
  "loops": [{
    "id": "repair",
    "selector_node": "decide",
    "selector_channel": "decision",
    "continue_match": "again",
    "retry_match": "retry",
    "members": ["attempt", "decide"],
    "entry": { "source": "decide", "target": "attempt" },
    "backedge": { "source": "attempt", "target": "decide" },
    "exit": { "source": "decide", "target": "END" },
    "max_iterations": 2,
    "max_retries": 1
  }],
  "terminal_outputs": []
}"#;

/// A model-selected path must lower to immutable control facts instead of being
/// rejected as an opaque future extension. Removing Task 8's strict control
/// lowering, or treating a model result as an arbitrary path, makes this fail.
#[test]
fn declared_model_selected_paths_lower_to_immutable_control_contract() {
    let task = native_task_fixture(COUNTERFACTUAL_SOURCE.as_bytes());

    let imported = import_native_task(task.path());
    let native = imported
        .package
        .native_graph()
        .expect("fixture contains a NativeGraph package");
    let (program, _) = lower_native_graph(native)
        .expect("declared model-selected paths lower through the live driver");
    let control = program
        .driver
        .data
        .get("control_flow")
        .expect("lowered driver retains immutable control facts");

    assert_eq!(control["branches"][0]["id"], json!("route"));
    assert_eq!(
        control["branches"][0]["candidates"][0]["edge"],
        json!({"source": "route-model", "target": "tool-a"})
    );
    assert_eq!(
        control["joins"][0]["reduction"],
        json!("selected_candidate")
    );
}

/// A branch declaration owns the exact subgraph entered by its conditional
/// edge. Omitting that edge target would otherwise let a malformed source pass
/// lowering and reach driver provisioning before its selected path is known.
#[test]
fn branch_candidate_must_include_its_declared_edge_target_before_driver_execution() {
    let source =
        COUNTERFACTUAL_SOURCE.replace(r#""nodes": ["tool-a"]"#, r#""nodes": ["finish-model"]"#);
    let task = native_task_fixture(source.as_bytes());
    let imported = import_native_task(task.path());
    let native = imported
        .package
        .native_graph()
        .expect("fixture contains a NativeGraph package");

    let error = lower_native_graph(native).expect_err(
        "a branch candidate that omits its edge target must fail before a driver opens",
    );

    assert!(
        error
            .to_string()
            .contains("does not include its declared edge target"),
        "lowering identifies the malformed branch subgraph: {error}"
    );
}

/// A selected branch cannot merge a channel produced only by another candidate.
/// Without this closure check, the declaration can pass lowering but the selected
/// path neither produces the named value nor reaches the declared join.
#[test]
fn join_rejects_a_cross_branch_candidate_channel_before_staging() {
    let source = COUNTERFACTUAL_SOURCE.replace(
        r#""channels": ["observation_a"]"#,
        r#""channels": ["observation_b"]"#,
    );
    let task = native_task_fixture(source.as_bytes());
    let imported = import_native_task(task.path());
    let native = imported
        .package
        .native_graph()
        .expect("fixture contains a NativeGraph package");

    let error = lower_native_graph(native)
        .expect_err("a cross-branch channel must fail before a graph stage can open");

    assert!(
        error
            .to_string()
            .contains("does not produce declared channel"),
        "lowering identifies the cross-branch channel: {error}"
    );
}

/// A branch candidate channel must have a producer in the selected subgraph.
/// Referring to the join's own output would otherwise make the path complete
/// without a candidate-produced observation.
#[test]
fn join_rejects_a_candidate_channel_without_a_subgraph_producer_before_staging() {
    let source = COUNTERFACTUAL_SOURCE.replace(
        r#""channels": ["observation_a"]"#,
        r#""channels": ["selected_observation"]"#,
    );
    let task = native_task_fixture(source.as_bytes());
    let imported = import_native_task(task.path());
    let native = imported
        .package
        .native_graph()
        .expect("fixture contains a NativeGraph package");

    let error = lower_native_graph(native)
        .expect_err("an unproduced candidate channel must fail before a graph stage can open");

    assert!(
        error
            .to_string()
            .contains("does not produce declared channel"),
        "lowering identifies the missing candidate producer: {error}"
    );
}

/// A candidate-produced channel must reach a consumer of the join output by its
/// declared source path. Otherwise the join can leave a selected branch with no
/// executable follow-up stage and silently finish the trace.
#[test]
fn join_rejects_a_candidate_channel_without_a_causal_merge_path_before_staging() {
    let source = COUNTERFACTUAL_SOURCE.replace(
        r#"{ "source": "tool-a", "target": "finish-model" }"#,
        r#"{ "source": "tool-a", "target": "END" }"#,
    );
    let task = native_task_fixture(source.as_bytes());
    let imported = import_native_task(task.path());
    let native = imported
        .package
        .native_graph()
        .expect("fixture contains a NativeGraph package");

    let error = lower_native_graph(native)
        .expect_err("a disconnected candidate merge path must fail before a graph stage can open");

    assert!(
        error
            .to_string()
            .contains("does not reach its declared join"),
        "lowering identifies the disconnected join path: {error}"
    );
}

/// A bounded loop declaration must name every node on its declared feedback
/// path. Omitting an interior member would let a later iteration skip work
/// that remains marked complete in the Rust-owned cursor.
#[test]
fn loop_members_must_cover_the_declared_feedback_path_before_driver_execution() {
    let mut source: serde_json::Value =
        serde_json::from_str(LOOP_SOURCE).expect("loop fixture is valid JSON");
    source["nodes"]
        .as_array_mut()
        .expect("loop fixture has nodes")
        .extend([
            json!({
                "id": "attempt-middle",
                "kind": "tool",
                "adapter": "tool-adapter",
                "operation": "attempt-middle",
                "output": "observation"
            }),
            json!({
                "id": "attempt-last",
                "kind": "tool",
                "adapter": "tool-adapter",
                "operation": "attempt-last",
                "output": "observation"
            }),
        ]);
    let edges = source["edges"]
        .as_array_mut()
        .expect("loop fixture has edges");
    let backedge = edges
        .iter_mut()
        .find(|edge| edge["source"] == "attempt" && edge["target"] == "decide")
        .expect("loop fixture has its original feedback edge");
    backedge["target"] = json!("attempt-middle");
    edges.extend([
        json!({"source": "attempt-middle", "target": "attempt-last"}),
        json!({"source": "attempt-last", "target": "decide"}),
    ]);
    let loop_spec = source["loops"][0]
        .as_object_mut()
        .expect("loop fixture has one loop declaration");
    loop_spec.insert(
        "members".into(),
        json!(["attempt", "attempt-last", "decide"]),
    );
    loop_spec.insert(
        "backedge".into(),
        json!({"source": "attempt-last", "target": "decide"}),
    );
    let task = native_task_fixture(
        serde_json::to_vec(&source)
            .expect("loop fixture serializes")
            .as_slice(),
    );
    let imported = import_native_task(task.path());
    let native = imported
        .package
        .native_graph()
        .expect("fixture contains a NativeGraph package");

    let error = lower_native_graph(native).expect_err(
        "an interior feedback node omitted from loop members must fail before a driver opens",
    );

    assert!(
        error
            .to_string()
            .contains("omits source feedback-path member"),
        "lowering identifies the incomplete loop membership: {error}"
    );
}

/// Dynamic source validation happens before any root dispatcher or invocation
/// lease is acquired. Otherwise a malformed immutable projection can leave an
/// opened session behind after `open` returns an error.
#[tokio::test(flavor = "current_thread")]
async fn invalid_dynamic_source_fails_before_acquiring_a_root_dispatcher_session() {
    let task = native_task_fixture(COUNTERFACTUAL_SOURCE.as_bytes());
    let imported = import_native_task(task.path());
    let native = imported
        .package
        .native_graph()
        .expect("fixture contains a NativeGraph package");
    let (mut program, _) = lower_native_graph(native).expect("fixture lowers");
    let control: NativeGraphControlContract =
        serde_json::from_value(program.driver.data["control_flow"].clone())
            .expect("lowered control contract is strict");
    program
        .profiling
        .graph
        .edges
        .retain(|edge| edge.source != "START");
    let trace = aiperf_runtime::graph::driver::TraceIdentity {
        run_id: "run".into(),
        trajectory_id: "trajectory".into(),
        trace_id: "counterfactual-path".into(),
    };
    let events = Arc::new(Mutex::new(Vec::new()));
    let factory = NativeGraphLiveTraceProgramDriverFactory::default().with_agent_loop_factories(
        NativeGraphLiveAgentLoopFactories::new(
            Arc::new(RecordingLifecycleFactoryFactory {
                requests: Arc::new(Mutex::new(Vec::new())),
                events: events.clone(),
                child_dispatcher: Arc::new(InMemoryToolDispatcherFactory),
            }),
            Arc::new(RecordingSessionToolDispatcherFactory {
                events: events.clone(),
            }),
        ),
    );
    let clock: Rc<dyn Clock> = Rc::new(SimClock::new());
    let segments = InMemorySegmentStore::default();
    let invocation =
        TraceAgentInvocationContext::native_graph(&trace, 0, control.source_snapshot_digest);
    let context = TraceDriverContext::for_execution(&trace, &clock, &segments, &invocation);
    let mut driver = factory
        .create(WorkerIdentity { worker_id: 0 }, &trace, &program.driver)
        .expect("lowered source selects the live driver");

    let error = driver
        .open(&program, &context)
        .await
        .expect_err("invalid dynamic projection must reject before root session acquisition");

    assert!(
        error
            .to_string()
            .contains("nodes are unreachable from START"),
        "the original source-validation error is retained: {error}"
    );
    assert!(
        events.lock().expect("session log is available").is_empty(),
        "a rejected dynamic source cannot open a dispatcher or root lease"
    );
}

/// A model reply may select only its declared path, and the selected tool
/// observation must become the next model stage's input. Replacing the live
/// decision with a pre-authored path, running both tools, or dropping the
/// selected observation makes this fail.
#[tokio::test(flavor = "current_thread")]
async fn model_response_selects_distinct_tool_stage_and_threads_its_observation() {
    selected_path("choose-a", "tool-a", "observation_a", "observation-a").await;
    selected_path("choose-b", "tool-b", "observation_b", "observation-b").await;
}

/// A join may deliberately admit only a subset of its branch's declared
/// candidates. A model-selected but unadmitted candidate must stop before the
/// merge can create a follow-up model stage.
#[tokio::test(flavor = "current_thread")]
async fn restricted_join_rejects_an_unadmitted_selected_candidate_before_merge() {
    let source = COUNTERFACTUAL_SOURCE.replace(
        r#""candidates": ["choose-a", "choose-b"]"#,
        r#""candidates": ["choose-b"]"#,
    );
    let task = native_task_fixture(source.as_bytes());
    let imported = import_native_task(task.path());
    let native = imported
        .package
        .native_graph()
        .expect("fixture contains a NativeGraph package");
    let (program, _) = lower_native_graph(native).expect("restricted join is a valid contract");
    let trace = aiperf_runtime::graph::driver::TraceIdentity {
        run_id: "run".into(),
        trajectory_id: "trajectory".into(),
        trace_id: "counterfactual-path".into(),
    };
    let factories = native_execution_factories();
    let mut driver = factories
        .trace_driver()
        .create(WorkerIdentity { worker_id: 0 }, &trace, &program.driver)
        .expect("lowered source selects the native live driver");
    let context = TraceDriverContext::metadata_only(&trace);
    driver.open(&program, &context).await.expect("driver opens");

    let _ = next_stage(&mut *driver, &context).await;
    let error = driver
        .observe_stage(stage_result(
            "counterfactual-path::stage-0",
            [("decision", json!("choose-a"))],
        ))
        .await
        .expect_err("unadmitted branch candidate must not be scheduled for merge");

    assert!(
        error
            .to_string()
            .contains("does not admit selected candidate"),
        "the error identifies the restricted join admission boundary: {error}"
    );
    assert!(
        driver
            .next_stage(&context)
            .await
            .expect("rejected merge leaves no follow-up stage")
            .is_none(),
        "the rejected candidate cannot create a merged finish stage"
    );
}

/// A feedback edge is re-entered only after its model decision consumes the
/// declared retry/iteration budgets. The shared executor still sees five DAG
/// stages; it never receives a static cycle or unrolled copy of the source.
#[tokio::test(flavor = "current_thread")]
async fn model_selected_retry_and_loop_reenter_only_within_declared_horizons() {
    let task = native_task_fixture(LOOP_SOURCE.as_bytes());
    let imported = import_native_task(task.path());
    let native = imported
        .package
        .native_graph()
        .expect("fixture contains a NativeGraph package");
    let (program, _) = lower_native_graph(native).expect("bounded loop lowers");
    let trace = aiperf_runtime::graph::driver::TraceIdentity {
        run_id: "run".into(),
        trajectory_id: "trajectory".into(),
        trace_id: "bounded-loop".into(),
    };
    let factories = native_execution_factories();
    let mut driver = factories
        .trace_driver()
        .create(WorkerIdentity { worker_id: 0 }, &trace, &program.driver)
        .expect("lowered source selects the native live driver");
    let context = TraceDriverContext::metadata_only(&trace);
    driver.open(&program, &context).await.expect("driver opens");

    for (index, (node, channel, value)) in [
        ("decide", "decision", json!("retry")),
        ("attempt", "observation", json!("first")),
        ("decide", "decision", json!("again")),
        ("attempt", "observation", json!("second")),
        ("decide", "decision", json!("done")),
    ]
    .into_iter()
    .enumerate()
    {
        let stage = next_stage(&mut *driver, &context).await;
        assert_eq!(stage.graph.nodes.keys().collect::<Vec<_>>(), [node]);
        driver
            .observe_stage(stage_result(
                format!("bounded-loop::stage-{index}"),
                [(channel, value)],
            ))
            .await
            .expect("declared loop stage is accepted");
    }

    let Some(TraceStageDirective::Complete(supplement)) = driver
        .next_stage(&context)
        .await
        .expect("bounded loop reaches a terminal directive")
    else {
        panic!("bounded loop did not complete after its declared exit");
    };
    assert_eq!(
        supplement
            .dynamic_control_receipts
            .iter()
            .map(aiperf_runtime::graph::supplement::DynamicControlReceipt::operation)
            .collect::<Vec<_>>(),
        [
            aiperf_runtime::graph::supplement::DynamicControlOperation::Retry,
            aiperf_runtime::graph::supplement::DynamicControlOperation::Loop,
        ]
    );
}

/// Dynamic control evidence is a compact append-only receipt, not a retained
/// model reply, tool payload, secret, or workspace path. Removing receipt
/// publication or collapsing branch and merge selection into an untyped blob
/// makes this fail.
#[tokio::test(flavor = "current_thread")]
async fn selected_path_publishes_typed_append_only_control_receipts() {
    let terminal = completed_selected_path().await;
    let wire = serde_json::to_value(terminal).expect("terminal supplement is serializable");
    let receipts = wire["dynamic_control_receipts"]
        .as_array()
        .expect("dynamic path publishes ordered typed control receipts");

    assert_eq!(receipts.len(), 2);
    assert_eq!(receipts[0]["sequence"], json!(0));
    assert_eq!(receipts[0]["operation"], json!("branch"));
    assert_eq!(receipts[0]["control_id"], json!("route"));
    assert_eq!(receipts[0]["selected_candidate"], json!("choose-a"));
    assert!(
        receipts[0]["control_digest"]
            .as_str()
            .is_some_and(|digest| digest.starts_with("blake3:"))
    );
    assert!(
        receipts[0]["selected_candidate_digest"]
            .as_str()
            .is_some_and(|digest| digest.starts_with("blake3:"))
    );
    assert_eq!(receipts[1]["sequence"], json!(1));
    assert_eq!(receipts[1]["operation"], json!("merge"));
    assert_eq!(receipts[1]["control_id"], json!("selected-observation"));
    assert_eq!(receipts[1]["selected_candidate"], json!("choose-a"));
    assert!(
        !serde_json::to_string(receipts)
            .expect("receipts serialize")
            .contains("observation-a"),
        "receipts never retain raw model or tool payloads"
    );
}

/// Empty dynamic evidence must not alter the legacy terminal-output wire shape.
#[test]
fn empty_dynamic_receipts_preserve_terminal_output_wire() {
    let supplement = aiperf_runtime::graph::supplement::TraceTerminalSupplement::new(
        "run".into(),
        "trajectory".into(),
        "trace".into(),
        0,
        "native_graph_live",
    )
    .with_terminal_outputs(BTreeMap::from([("answer".into(), Handle::new(7))]));

    assert_eq!(
        serde_json::to_string(&supplement).expect("supplement serializes"),
        r#"{"schema_version":1,"run_id":"run","trajectory_id":"trajectory","trace_id":"trace","worker_id":0,"driver_kind":"native_graph_live","completed":true,"trace_wall_ms":0.0,"calls":[],"tools":[],"terminal_outputs":{"answer":7}}"#,
    );
}

/// Dynamic receipt identity is a public artifact boundary, never a place to
/// retain caller-provided paths or credentials. Returning `String` fields from
/// receipt construction would serialize both values unchanged.
#[test]
fn dynamic_receipt_rejects_path_and_secret_like_declared_names() {
    const DIGEST: &str = "blake3:0000000000000000000000000000000000000000000000000000000000000000";
    for (control_id, selected_candidate) in [
        ("../../branch-workspace", "choose-a"),
        ("route", "api_key=top-secret"),
    ] {
        let error = serde_json::from_value::<
            aiperf_runtime::graph::supplement::DynamicControlReceipt,
        >(json!({
            "control_digest": DIGEST,
            "sequence": 0,
            "operation": "branch",
            "control_id": control_id,
            "selected_candidate": selected_candidate,
            "selected_candidate_digest": DIGEST,
            "counters": {
                "completed_stages": 1,
                "loop_iterations": 0,
                "retries": 0
            }
        }))
        .expect_err("untrusted paths and secret-like names must not enter receipt wire data");

        assert!(
            error
                .to_string()
                .contains("invalid declared dynamic control name"),
            "the error is generic and does not echo the rejected value: {error}"
        );
    }
}

/// Live attempts receive native invocation scopes before either workspace is
/// leased. Reusing a recorded-replay scope here would make the root invocation
/// or cleanup authority identical across attempts.
#[tokio::test(flavor = "current_thread")]
async fn live_attempts_mint_distinct_native_invocations_and_root_workspaces() {
    let task = native_task_fixture(COUNTERFACTUAL_SOURCE.as_bytes());
    let imported = import_native_task(task.path());
    let native = imported
        .package
        .native_graph()
        .expect("fixture contains a NativeGraph package");
    let (program, _) = lower_native_graph(native).expect("fixture lowers");
    let control: NativeGraphControlContract =
        serde_json::from_value(program.driver.data["control_flow"].clone())
            .expect("lowered control contract is strict");
    let trace = aiperf_runtime::graph::driver::TraceIdentity {
        run_id: "run".into(),
        trajectory_id: "trajectory".into(),
        trace_id: "counterfactual-path".into(),
    };
    let requests = Arc::new(Mutex::new(Vec::new()));
    let events = Arc::new(Mutex::new(Vec::new()));
    let factory = NativeGraphLiveTraceProgramDriverFactory::default().with_agent_loop_factories(
        NativeGraphLiveAgentLoopFactories::new(
            Arc::new(RecordingLifecycleFactoryFactory {
                requests: requests.clone(),
                events,
                child_dispatcher: Arc::new(InMemoryToolDispatcherFactory),
            }),
            Arc::new(InMemoryToolDispatcherFactory),
        ),
    );
    let clock: Rc<dyn Clock> = Rc::new(SimClock::new());
    let segments = InMemorySegmentStore::default();

    for attempt in 0..2 {
        let invocation = TraceAgentInvocationContext::native_graph(
            &trace,
            attempt,
            control.source_snapshot_digest.clone(),
        );
        assert!(
            invocation
                .root_invocation_id()
                .starts_with("native-graph::")
        );
        assert!(invocation.cleanup_label().starts_with("native-graph-"));
        let context = TraceDriverContext::for_execution(&trace, &clock, &segments, &invocation);
        let mut driver = factory
            .create(WorkerIdentity { worker_id: 0 }, &trace, &program.driver)
            .expect("lowered source selects the live driver");
        driver
            .open(&program, &context)
            .await
            .expect("native invocation context opens one root lease");
        driver.close().await.expect("root lease closes per attempt");
    }

    let requests = requests.lock().expect("request log is available");
    assert_eq!(requests.len(), 2);
    assert_ne!(
        requests[0].identity.invocation_id,
        requests[1].identity.invocation_id
    );
    assert!(matches!(
        requests[0].workspace,
        aiperf_runtime::graph::agent::AgentInvocationWorkspace::Root
    ));
    assert!(matches!(
        requests[1].workspace,
        aiperf_runtime::graph::agent::AgentInvocationWorkspace::Root
    ));
}

/// A dispatcher can finish partial provisioning and then reject a trace open.
/// Failing to close it before releasing the root lease leaves that provisioned
/// resource outside the ordinary reverse cleanup path.
#[tokio::test(flavor = "current_thread")]
async fn dispatcher_open_error_closes_a_partially_provisioned_root_session_once() {
    let task = native_task_fixture(COUNTERFACTUAL_SOURCE.as_bytes());
    let imported = import_native_task(task.path());
    let native = imported
        .package
        .native_graph()
        .expect("fixture contains a NativeGraph package");
    let (program, _) = lower_native_graph(native).expect("fixture lowers");
    let control: NativeGraphControlContract =
        serde_json::from_value(program.driver.data["control_flow"].clone())
            .expect("lowered control contract is strict");
    let trace = aiperf_runtime::graph::driver::TraceIdentity {
        run_id: "run".into(),
        trajectory_id: "trajectory".into(),
        trace_id: "counterfactual-path".into(),
    };
    let events = Arc::new(Mutex::new(Vec::new()));
    let factory = NativeGraphLiveTraceProgramDriverFactory::default().with_agent_loop_factories(
        NativeGraphLiveAgentLoopFactories::new(
            Arc::new(RecordingLifecycleFactoryFactory {
                requests: Arc::new(Mutex::new(Vec::new())),
                events: events.clone(),
                child_dispatcher: Arc::new(InMemoryToolDispatcherFactory),
            }),
            Arc::new(FailingOpenSessionToolDispatcherFactory {
                events: events.clone(),
            }),
        ),
    );
    let clock: Rc<dyn Clock> = Rc::new(SimClock::new());
    let segments = InMemorySegmentStore::default();
    let invocation =
        TraceAgentInvocationContext::native_graph(&trace, 0, control.source_snapshot_digest);
    let context = TraceDriverContext::for_execution(&trace, &clock, &segments, &invocation);
    let mut driver = factory
        .create(WorkerIdentity { worker_id: 0 }, &trace, &program.driver)
        .expect("lowered source selects the live driver");

    let error = driver
        .open(&program, &context)
        .await
        .expect_err("partially provisioned dispatcher rejects the root session");

    assert!(error.to_string().contains("provisioned dispatcher failed"));
    assert_eq!(
        events.lock().expect("session log is available").as_slice(),
        [
            "open:root",
            "dispatcher:open",
            "dispatcher:close",
            "close:root"
        ],
        "the partially provisioned dispatcher closes exactly once before root lease cleanup"
    );
}

/// Cancelling while `open_trace` is still pending must roll back the root
/// dispatcher and lifecycle lease without waiting for a detached opener.
#[tokio::test(flavor = "current_thread")]
async fn cancelling_pending_native_dispatcher_open_closes_root_resources_once() {
    let task = native_task_fixture(COUNTERFACTUAL_SOURCE.as_bytes());
    let imported = import_native_task(task.path());
    let native = imported
        .package
        .native_graph()
        .expect("fixture contains a NativeGraph package");
    let (program, _) = lower_native_graph(native).expect("fixture lowers");
    let control: NativeGraphControlContract =
        serde_json::from_value(program.driver.data["control_flow"].clone())
            .expect("lowered control contract is strict");
    let trace = aiperf_runtime::graph::driver::TraceIdentity {
        run_id: "run".into(),
        trajectory_id: "trajectory".into(),
        trace_id: "counterfactual-path".into(),
    };
    let events = Arc::new(Mutex::new(Vec::new()));
    let dispatcher_started = Arc::new(AtomicBool::new(false));
    let factory = NativeGraphLiveTraceProgramDriverFactory::default().with_agent_loop_factories(
        NativeGraphLiveAgentLoopFactories::new(
            Arc::new(RecordingLifecycleFactoryFactory {
                requests: Arc::new(Mutex::new(Vec::new())),
                events: events.clone(),
                child_dispatcher: Arc::new(InMemoryToolDispatcherFactory),
            }),
            Arc::new(PendingOpenSessionToolDispatcherFactory {
                events: events.clone(),
                started: dispatcher_started.clone(),
            }),
        ),
    );
    let clock: Rc<dyn Clock> = Rc::new(SimClock::new());
    let segments = InMemorySegmentStore::default();
    let invocation =
        TraceAgentInvocationContext::native_graph(&trace, 0, control.source_snapshot_digest);
    let context = TraceDriverContext::for_execution(&trace, &clock, &segments, &invocation);
    let mut driver = factory
        .create(WorkerIdentity { worker_id: 0 }, &trace, &program.driver)
        .expect("lowered source selects the live driver");
    let (abort, registration) = futures::future::AbortHandle::new_pair();

    {
        let open = futures::future::Abortable::new(driver.open(&program, &context), registration);
        tokio::pin!(open);
        futures::future::poll_fn(|task| {
            let poll = open.as_mut().poll(task);
            assert!(
                poll.is_pending(),
                "dispatcher open deliberately remains pending"
            );
            std::task::Poll::Ready(())
        })
        .await;
        assert!(
            dispatcher_started.load(Ordering::SeqCst),
            "the dispatcher has begun partial provisioning before cancellation"
        );
        abort.abort();
        assert!(matches!(
            futures::future::poll_fn(|task| open.as_mut().poll(task)).await,
            Err(futures::future::Aborted)
        ));
    }

    driver
        .abort_open()
        .await
        .expect("native driver rolls back its suspended open");
    assert_eq!(
        events.lock().expect("session log is available").as_slice(),
        [
            "open:root",
            "dispatcher:open",
            "dispatcher:close",
            "close:root"
        ],
        "cancelling an in-flight dispatcher open closes the partial session in reverse order"
    );
}

/// A model selection creates only the chosen branch workspace. Its tool stage
/// must run through the lease-owned dispatcher, and the completed child lease
/// must close before its candidate is merged into the next model stage.
#[tokio::test(flavor = "current_thread")]
async fn selected_branch_uses_its_lease_dispatcher_and_closes_before_merge() {
    let task = native_task_fixture(COUNTERFACTUAL_SOURCE.as_bytes());
    let imported = import_native_task(task.path());
    let native = imported
        .package
        .native_graph()
        .expect("fixture contains a NativeGraph package");
    let (program, _) = lower_native_graph(native).expect("fixture lowers");
    let control: NativeGraphControlContract =
        serde_json::from_value(program.driver.data["control_flow"].clone())
            .expect("lowered control contract is strict");
    let trace = aiperf_runtime::graph::driver::TraceIdentity {
        run_id: "run".into(),
        trajectory_id: "trajectory".into(),
        trace_id: "counterfactual-path".into(),
    };
    let events = Arc::new(Mutex::new(Vec::new()));
    let tool_dispatches = Arc::new(Mutex::new(Vec::new()));
    let tool_factory = Arc::new(RecordingToolDispatcherFactory {
        dispatches: tool_dispatches.clone(),
    });
    let factory = NativeGraphLiveTraceProgramDriverFactory::default().with_agent_loop_factories(
        NativeGraphLiveAgentLoopFactories::new(
            Arc::new(RecordingLifecycleFactoryFactory {
                requests: Arc::new(Mutex::new(Vec::new())),
                events: events.clone(),
                child_dispatcher: tool_factory.clone(),
            }),
            tool_factory,
        ),
    );
    let clock: Rc<dyn Clock> = Rc::new(SimClock::new());
    let segments = InMemorySegmentStore::default();
    let invocation =
        TraceAgentInvocationContext::native_graph(&trace, 0, control.source_snapshot_digest);
    let context = TraceDriverContext::for_execution(&trace, &clock, &segments, &invocation);
    let mut driver = factory
        .create(WorkerIdentity { worker_id: 0 }, &trace, &program.driver)
        .expect("lowered source selects the live driver");
    driver.open(&program, &context).await.expect("root opens");

    let route = next_stage(&mut *driver, &context).await;
    assert_eq!(
        route.graph.nodes.keys().collect::<Vec<_>>(),
        ["route-model"]
    );
    driver
        .observe_stage(stage_result(
            "counterfactual-path::stage-0",
            [("decision", json!("choose-a"))],
        ))
        .await
        .expect("model selection is recorded");

    let selected_tool = next_stage(&mut *driver, &context).await;
    assert_eq!(
        selected_tool.graph.nodes.keys().collect::<Vec<_>>(),
        ["tool-a"]
    );
    driver
        .tool_dispatcher()
        .expect("selected child owns the next stage dispatcher")
        .dispatch(
            aiperf_runtime::graph::tools::ToolDispatchRequest::new("tool-a-call", "tool-a"),
            &aiperf_runtime::graph::tools::ToolDispatchContext::default(),
        )
        .await
        .expect("selected tool runs through ToolDispatcher");
    driver
        .observe_stage(stage_result(
            "counterfactual-path::stage-1",
            [("observation_a", json!("tool-result"))],
        ))
        .await
        .expect("selected child completes before merge");

    let finish = next_stage(&mut *driver, &context).await;
    assert_eq!(
        finish.graph.nodes.keys().collect::<Vec<_>>(),
        ["finish-model"]
    );
    assert_eq!(
        finish.trace.initial_state["selected_observation"],
        json!("tool-result"),
        "the selected tool observation reaches the next model stage"
    );
    driver
        .observe_stage(stage_result(
            "counterfactual-path::stage-2",
            [("answer", json!("done"))],
        ))
        .await
        .expect("finish stage completes");
    let Some(TraceStageDirective::Complete(supplement)) = driver
        .next_stage(&context)
        .await
        .expect("terminal directive succeeds")
    else {
        panic!("selected branch did not complete");
    };
    assert_eq!(
        tool_dispatches.lock().expect("dispatch log").as_slice(),
        ["tool-a-call"]
    );
    let events = events.lock().expect("lifecycle log");
    assert!(events.iter().any(|event| event == "open:choose-a"));
    assert!(!events.iter().any(|event| event == "open:choose-b"));
    let completion = events
        .iter()
        .position(|event| event == "complete:choose-a")
        .expect("selected child completed");
    let close = events
        .iter()
        .position(|event| event == "close:choose-a")
        .expect("selected child closed");
    assert!(completion < close, "child must complete before closing");
    assert!(supplement.dynamic_control_receipts.iter().any(|receipt| {
        receipt.operation() == aiperf_runtime::graph::supplement::DynamicControlOperation::Merge
            && receipt.selected_candidate() == "choose-a"
            && receipt
                .selected_candidate_digest()
                .as_str()
                .starts_with("blake3:")
    }));
}

struct RecordingLifecycleFactoryFactory {
    requests: Arc<Mutex<Vec<AgentInvocationRequest>>>,
    events: Arc<Mutex<Vec<String>>>,
    child_dispatcher: Arc<dyn aiperf_runtime::graph::tools::ToolDispatcherFactory>,
}

impl AgentInvocationLeaseFactoryFactory for RecordingLifecycleFactoryFactory {
    fn create(
        &self,
        trace_id: &str,
        root_dispatcher: Rc<dyn ToolDispatcher>,
    ) -> Result<Box<dyn AgentInvocationLeaseFactory>, aiperf_runtime::graph::agent::AgentLoopError>
    {
        let _ = trace_id;
        Ok(Box::new(RecordingLifecycleFactory {
            root_dispatcher: RefCell::new(Some(root_dispatcher)),
            requests: self.requests.clone(),
            events: self.events.clone(),
            child_dispatcher: self.child_dispatcher.clone(),
        }))
    }
}

struct RecordingLifecycleFactory {
    root_dispatcher: RefCell<Option<Rc<dyn ToolDispatcher>>>,
    requests: Arc<Mutex<Vec<AgentInvocationRequest>>>,
    events: Arc<Mutex<Vec<String>>>,
    child_dispatcher: Arc<dyn aiperf_runtime::graph::tools::ToolDispatcherFactory>,
}

impl AgentInvocationLeaseFactory for RecordingLifecycleFactory {
    fn begin_open(
        &self,
        request: &AgentInvocationRequest,
        parent: Option<&dyn AgentInvocationLease>,
    ) -> Result<Box<dyn AgentInvocationLeaseOpening>, aiperf_runtime::graph::agent::AgentLoopError>
    {
        self.requests
            .lock()
            .expect("request log is available")
            .push(request.clone());
        let (candidate_id, dispatcher) = match &request.workspace {
            aiperf_runtime::graph::agent::AgentInvocationWorkspace::Root => (
                "root".to_string(),
                self.root_dispatcher.borrow_mut().take().ok_or_else(|| {
                    aiperf_runtime::graph::agent::AgentLoopError::new(
                        "recording lifecycle root dispatcher was already claimed",
                    )
                })?,
            ),
            aiperf_runtime::graph::agent::AgentInvocationWorkspace::IsolatedBranch {
                candidate_id,
                ..
            } => (
                candidate_id.clone(),
                self.child_dispatcher
                    .create(&request.identity.invocation_id)
                    .map_err(|error| {
                        aiperf_runtime::graph::agent::AgentLoopError::new(error.to_string())
                    })?,
            ),
        };
        let _ = parent;
        Ok(Box::new(RecordingLifecycleOpening {
            candidate_id: candidate_id.clone(),
            events: self.events.clone(),
            lease: Some(Box::new(RecordingLifecycleLease {
                candidate_id,
                dispatcher,
                events: self.events.clone(),
                is_closed: Cell::new(false),
            })),
        }))
    }
}

struct RecordingLifecycleOpening {
    candidate_id: String,
    events: Arc<Mutex<Vec<String>>>,
    lease: Option<Box<dyn AgentInvocationLease>>,
}

#[async_trait(?Send)]
impl AgentInvocationLeaseOpening for RecordingLifecycleOpening {
    async fn open(
        &mut self,
    ) -> Result<Box<dyn AgentInvocationLease>, aiperf_runtime::graph::agent::AgentLoopError> {
        let lease = self.lease.take().ok_or_else(|| {
            aiperf_runtime::graph::agent::AgentLoopError::new(
                "recording lifecycle opening was already consumed",
            )
        })?;
        self.events
            .lock()
            .expect("lifecycle log is available")
            .push(format!("open:{}", self.candidate_id));
        Ok(lease)
    }

    fn cancel_on_drop(&mut self) {
        if let Some(mut lease) = self.lease.take() {
            lease.close_on_drop();
        }
    }
}

struct RecordingLifecycleLease {
    candidate_id: String,
    dispatcher: Rc<dyn ToolDispatcher>,
    events: Arc<Mutex<Vec<String>>>,
    is_closed: Cell<bool>,
}

#[async_trait(?Send)]
impl AgentInvocationLease for RecordingLifecycleLease {
    fn dispatcher(&self) -> Rc<dyn ToolDispatcher> {
        self.dispatcher.clone()
    }

    fn close_on_drop(&mut self) {
        if !self.is_closed.replace(true) {
            self.events
                .lock()
                .expect("lifecycle log is available")
                .push(format!("close:{}", self.candidate_id));
        }
    }

    async fn complete_workspace(
        &mut self,
    ) -> Result<
        Option<aiperf_runtime::graph::agent::AgentInvocationWorkspaceCandidate>,
        aiperf_runtime::graph::agent::AgentLoopError,
    > {
        if self.is_closed.get() {
            return Err(aiperf_runtime::graph::agent::AgentLoopError::new(
                "recording lifecycle workspace is already closed",
            ));
        }
        if self.candidate_id == "root" {
            return Ok(None);
        }
        self.events
            .lock()
            .expect("lifecycle log is available")
            .push(format!("complete:{}", self.candidate_id));
        Ok(Some(
            aiperf_runtime::graph::agent::AgentInvocationWorkspaceCandidate::new(
                self.candidate_id.clone(),
                aiperf_runtime::eval::ArtifactDigest::from_bytes(self.candidate_id.as_bytes()),
            ),
        ))
    }

    async fn close(&mut self) -> Result<(), aiperf_runtime::graph::agent::AgentLoopError> {
        self.close_on_drop();
        Ok(())
    }
}

struct RecordingToolDispatcherFactory {
    dispatches: Arc<Mutex<Vec<String>>>,
}

impl aiperf_runtime::graph::tools::ToolDispatcherFactory for RecordingToolDispatcherFactory {
    fn create(
        &self,
        _trace_id: &str,
    ) -> Result<Rc<dyn ToolDispatcher>, aiperf_runtime::graph::tools::ToolDispatchError> {
        Ok(Rc::new(RecordingToolDispatcher {
            dispatches: self.dispatches.clone(),
        }))
    }
}

struct RecordingToolDispatcher {
    dispatches: Arc<Mutex<Vec<String>>>,
}

#[async_trait(?Send)]
impl ToolDispatcher for RecordingToolDispatcher {
    async fn open_trace(
        &self,
        _context: aiperf_runtime::graph::tools::TraceOpenContext<'_>,
    ) -> Result<(), aiperf_runtime::graph::tools::ToolDispatchError> {
        Ok(())
    }

    async fn dispatch(
        &self,
        request: aiperf_runtime::graph::tools::ToolDispatchRequest,
        _context: &aiperf_runtime::graph::tools::ToolDispatchContext,
    ) -> Result<
        aiperf_runtime::graph::tools::ToolDispatchResult,
        aiperf_runtime::graph::tools::ToolDispatchError,
    > {
        self.dispatches
            .lock()
            .expect("dispatch log is available")
            .push(request.call_id.clone());
        Ok(aiperf_runtime::graph::tools::ToolDispatchResult::completed(
            request.call_id,
            0,
            Bytes::from_static(b"tool-result"),
        ))
    }

    async fn close_trace(
        &self,
        _trace: &aiperf_runtime::graph::driver::TraceIdentity,
    ) -> Result<(), aiperf_runtime::graph::tools::ToolDispatchError> {
        Ok(())
    }
}

struct RecordingSessionToolDispatcherFactory {
    events: Arc<Mutex<Vec<String>>>,
}

impl aiperf_runtime::graph::tools::ToolDispatcherFactory for RecordingSessionToolDispatcherFactory {
    fn create(
        &self,
        _trace_id: &str,
    ) -> Result<Rc<dyn ToolDispatcher>, aiperf_runtime::graph::tools::ToolDispatchError> {
        Ok(Rc::new(RecordingSessionToolDispatcher {
            events: self.events.clone(),
        }))
    }
}

struct RecordingSessionToolDispatcher {
    events: Arc<Mutex<Vec<String>>>,
}

#[async_trait(?Send)]
impl ToolDispatcher for RecordingSessionToolDispatcher {
    async fn open_trace(
        &self,
        _context: aiperf_runtime::graph::tools::TraceOpenContext<'_>,
    ) -> Result<(), aiperf_runtime::graph::tools::ToolDispatchError> {
        self.events
            .lock()
            .expect("session log is available")
            .push("dispatcher:open".into());
        Ok(())
    }

    async fn dispatch(
        &self,
        _request: aiperf_runtime::graph::tools::ToolDispatchRequest,
        _context: &aiperf_runtime::graph::tools::ToolDispatchContext,
    ) -> Result<
        aiperf_runtime::graph::tools::ToolDispatchResult,
        aiperf_runtime::graph::tools::ToolDispatchError,
    > {
        Err(aiperf_runtime::graph::tools::ToolDispatchError::new(
            "recording session dispatcher cannot dispatch a tool",
        ))
    }

    async fn close_trace(
        &self,
        _trace: &aiperf_runtime::graph::driver::TraceIdentity,
    ) -> Result<(), aiperf_runtime::graph::tools::ToolDispatchError> {
        self.events
            .lock()
            .expect("session log is available")
            .push("dispatcher:close".into());
        Ok(())
    }
}

struct FailingOpenSessionToolDispatcherFactory {
    events: Arc<Mutex<Vec<String>>>,
}

impl aiperf_runtime::graph::tools::ToolDispatcherFactory
    for FailingOpenSessionToolDispatcherFactory
{
    fn create(
        &self,
        _trace_id: &str,
    ) -> Result<Rc<dyn ToolDispatcher>, aiperf_runtime::graph::tools::ToolDispatchError> {
        Ok(Rc::new(FailingOpenSessionToolDispatcher {
            events: self.events.clone(),
        }))
    }
}

struct FailingOpenSessionToolDispatcher {
    events: Arc<Mutex<Vec<String>>>,
}

#[async_trait(?Send)]
impl ToolDispatcher for FailingOpenSessionToolDispatcher {
    async fn open_trace(
        &self,
        _context: aiperf_runtime::graph::tools::TraceOpenContext<'_>,
    ) -> Result<(), aiperf_runtime::graph::tools::ToolDispatchError> {
        self.events
            .lock()
            .expect("session log is available")
            .push("dispatcher:open".into());
        Err(aiperf_runtime::graph::tools::ToolDispatchError::new(
            "provisioned dispatcher failed",
        ))
    }

    async fn dispatch(
        &self,
        _request: aiperf_runtime::graph::tools::ToolDispatchRequest,
        _context: &aiperf_runtime::graph::tools::ToolDispatchContext,
    ) -> Result<
        aiperf_runtime::graph::tools::ToolDispatchResult,
        aiperf_runtime::graph::tools::ToolDispatchError,
    > {
        Err(aiperf_runtime::graph::tools::ToolDispatchError::new(
            "failing session dispatcher cannot dispatch a tool",
        ))
    }

    async fn close_trace(
        &self,
        _trace: &aiperf_runtime::graph::driver::TraceIdentity,
    ) -> Result<(), aiperf_runtime::graph::tools::ToolDispatchError> {
        self.events
            .lock()
            .expect("session log is available")
            .push("dispatcher:close".into());
        Ok(())
    }
}

struct PendingOpenSessionToolDispatcherFactory {
    events: Arc<Mutex<Vec<String>>>,
    started: Arc<AtomicBool>,
}

impl aiperf_runtime::graph::tools::ToolDispatcherFactory
    for PendingOpenSessionToolDispatcherFactory
{
    fn create(
        &self,
        _trace_id: &str,
    ) -> Result<Rc<dyn ToolDispatcher>, aiperf_runtime::graph::tools::ToolDispatchError> {
        Ok(Rc::new(PendingOpenSessionToolDispatcher {
            events: self.events.clone(),
            started: self.started.clone(),
        }))
    }
}

struct PendingOpenSessionToolDispatcher {
    events: Arc<Mutex<Vec<String>>>,
    started: Arc<AtomicBool>,
}

#[async_trait(?Send)]
impl ToolDispatcher for PendingOpenSessionToolDispatcher {
    async fn open_trace(
        &self,
        _context: aiperf_runtime::graph::tools::TraceOpenContext<'_>,
    ) -> Result<(), aiperf_runtime::graph::tools::ToolDispatchError> {
        self.events
            .lock()
            .expect("session log is available")
            .push("dispatcher:open".into());
        self.started.store(true, Ordering::SeqCst);
        std::future::pending().await
    }

    async fn dispatch(
        &self,
        _request: aiperf_runtime::graph::tools::ToolDispatchRequest,
        _context: &aiperf_runtime::graph::tools::ToolDispatchContext,
    ) -> Result<
        aiperf_runtime::graph::tools::ToolDispatchResult,
        aiperf_runtime::graph::tools::ToolDispatchError,
    > {
        Err(aiperf_runtime::graph::tools::ToolDispatchError::new(
            "pending session dispatcher cannot dispatch a tool",
        ))
    }

    async fn close_trace(
        &self,
        _trace: &aiperf_runtime::graph::driver::TraceIdentity,
    ) -> Result<(), aiperf_runtime::graph::tools::ToolDispatchError> {
        self.events
            .lock()
            .expect("session log is available")
            .push("dispatcher:close".into());
        Ok(())
    }
}

async fn selected_path(
    selection: &str,
    expected_tool: &str,
    observation_channel: &str,
    observation: &str,
) {
    let task = native_task_fixture(COUNTERFACTUAL_SOURCE.as_bytes());
    let imported = import_native_task(task.path());
    let native = imported
        .package
        .native_graph()
        .expect("fixture contains a NativeGraph package");
    let (program, _) = lower_native_graph(native).expect("fixture lowers");
    let trace = aiperf_runtime::graph::driver::TraceIdentity {
        run_id: "run".into(),
        trajectory_id: "trajectory".into(),
        trace_id: "counterfactual-path".into(),
    };
    let factories = native_execution_factories();
    let mut driver = factories
        .trace_driver()
        .create(WorkerIdentity { worker_id: 0 }, &trace, &program.driver)
        .expect("lowered source selects the native live driver");
    let context = TraceDriverContext::metadata_only(&trace);
    driver
        .open(&program, &context)
        .await
        .expect("immutable source opens before stage selection");

    let initial = next_stage(&mut *driver, &context).await;
    assert_eq!(
        initial.graph.nodes.keys().collect::<Vec<_>>(),
        ["route-model"],
        "the first model stage cannot preselect either tool"
    );
    driver
        .observe_stage(stage_result(
            "counterfactual-path::stage-0",
            [("decision", json!(selection))],
        ))
        .await
        .expect("the model decision is retained before the next stage");

    let tool = next_stage(&mut *driver, &context).await;
    assert_eq!(
        tool.graph.nodes.keys().collect::<Vec<_>>(),
        [expected_tool],
        "only the model-selected tool may enter the next stage"
    );
    driver
        .observe_stage(stage_result(
            "counterfactual-path::stage-1",
            [(observation_channel, json!(observation))],
        ))
        .await
        .expect("selected tool observation is immutable stage feedback");

    let finish = next_stage(&mut *driver, &context).await;
    assert_eq!(
        finish.graph.nodes.keys().collect::<Vec<_>>(),
        ["finish-model"]
    );
    assert_eq!(
        finish.trace.initial_state["selected_observation"],
        json!(observation),
        "the next model stage receives only the selected tool observation"
    );
}

async fn completed_selected_path() -> aiperf_runtime::graph::supplement::TraceTerminalSupplement {
    let task = native_task_fixture(COUNTERFACTUAL_SOURCE.as_bytes());
    let imported = import_native_task(task.path());
    let native = imported
        .package
        .native_graph()
        .expect("fixture contains a NativeGraph package");
    let (program, _) = lower_native_graph(native).expect("fixture lowers");
    let trace = aiperf_runtime::graph::driver::TraceIdentity {
        run_id: "run".into(),
        trajectory_id: "trajectory".into(),
        trace_id: "counterfactual-path".into(),
    };
    let factories = native_execution_factories();
    let mut driver = factories
        .trace_driver()
        .create(WorkerIdentity { worker_id: 0 }, &trace, &program.driver)
        .expect("lowered source selects the native live driver");
    let context = TraceDriverContext::metadata_only(&trace);
    driver.open(&program, &context).await.expect("driver opens");

    let _ = next_stage(&mut *driver, &context).await;
    driver
        .observe_stage(stage_result(
            "counterfactual-path::stage-0",
            [("decision", json!("choose-a"))],
        ))
        .await
        .expect("branch decision is accepted");
    let _ = next_stage(&mut *driver, &context).await;
    driver
        .observe_stage(stage_result(
            "counterfactual-path::stage-1",
            [("observation_a", json!("observation-a"))],
        ))
        .await
        .expect("selected tool result is accepted");
    let _ = next_stage(&mut *driver, &context).await;
    driver
        .observe_stage(stage_result(
            "counterfactual-path::stage-2",
            [("answer", json!("finished"))],
        ))
        .await
        .expect("final model result is accepted");
    match driver
        .next_stage(&context)
        .await
        .expect("terminal selection succeeds")
    {
        Some(TraceStageDirective::Complete(supplement)) => supplement,
        _ => panic!("dynamic path must finish with a terminal supplement"),
    }
}

async fn next_stage(
    driver: &mut dyn aiperf_runtime::graph::driver::TraceProgramDriver,
    context: &TraceDriverContext<'_>,
) -> aiperf_runtime::graph::model::GraphTracePlan {
    match driver
        .next_stage(context)
        .await
        .expect("driver stage selection succeeds")
    {
        Some(TraceStageDirective::Execute(plan)) => plan,
        Some(TraceStageDirective::Complete(_)) => {
            panic!("driver completed before the expected stage")
        }
        None => panic!("driver ended before the expected stage"),
    }
}

fn stage_result<'a>(
    plan_identity: impl Into<String>,
    channels: impl IntoIterator<Item = (&'a str, serde_json::Value)>,
) -> TraceStageResult {
    TraceStageResult {
        plan_identity: plan_identity.into(),
        terminal_status: GraphReplyStatus::Completed,
        channels: channels
            .into_iter()
            .map(|(channel, value)| (channel.to_owned(), value))
            .collect::<BTreeMap<_, _>>(),
        output_handles: BTreeMap::new(),
    }
}

fn import_native_task(task_root: &Path) -> aiperf_runtime::eval::ImportedTask {
    let source = HarborSource::local(task_root.to_string_lossy())
        .expect("temporary task path is a valid local Harbor source");
    HarborImporter::new(&NativeSourceAcquirer)
        .import(&source)
        .expect("fixture package imports")
}

fn native_task_fixture(program: &[u8]) -> tempfile::TempDir {
    let task = tempfile::tempdir().expect("temporary task root");
    fs::create_dir_all(task.path().join("environment")).expect("task environment directory");
    fs::create_dir_all(task.path().join("tests")).expect("task tests directory");
    fs::create_dir_all(task.path().join("tools")).expect("adapter directory");
    fs::write(
        task.path().join("environment/Dockerfile"),
        b"FROM scratch\n",
    )
    .expect("task Dockerfile");
    fs::write(task.path().join("instruction.md"), b"Choose a path.\n").expect("task instruction");
    fs::write(task.path().join("tests/test.sh"), b"exit 0\n").expect("task verifier");
    fs::write(
        task.path().join("task.toml"),
        r#"schema_version = "1.1"

[task]
name = "example/native-graph-live-path"

[native_graph]
profile = "native_graph"
program = "agent_graph.json"
model_bindings = "models.toml"
adapter_manifest = "adapters.toml"
"#,
    )
    .expect("task manifest");
    fs::write(task.path().join("agent_graph.json"), program).expect("graph source");
    fs::write(
        task.path().join("models.toml"),
        r#"[[model_bindings]]
id = "primary"
endpoint_profile_id = "provider-default"
endpoint_factory_id = "chat"
transport_factory_id = "http"
model = "example-model"
urls = ["https://provider.example/v1"]
streaming = false
request_timeout_ms = 30000
capture = "metadata"

[model_bindings.tokenizer]
type = "local"
name = "builtin"
revision = "main"
apply_chat_template = false

[model_bindings.generation]
"#,
    )
    .expect("model bindings");
    fs::write(
        task.path().join("adapters.toml"),
        r#"[[adapters]]
id = "tool-adapter"
role = "tool"
argv = ["tools/adapter.sh"]
executable = "tools/adapter.sh"
"#,
    )
    .expect("adapter manifest");
    fs::write(task.path().join("tools/adapter.sh"), b"#!/bin/sh\nexit 0\n")
        .expect("adapter executable");
    task
}
