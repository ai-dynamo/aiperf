// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Source-faithful NativeGraph lowering contracts.
#![cfg(feature = "engine")]

use std::fs;
use std::path::Path;

use aiperf_runtime::engine::execution_factories::native_execution_factories;
use aiperf_runtime::eval::{
    GraphLowererFactory, GraphLoweringRequest, HarborImporter, HarborSource,
    NativeGraphControlContract, NativeGraphLowererFactory, NativeGraphLoweringError,
    NativeSourceAcquirer, lower_native_graph,
};
use aiperf_runtime::graph::driver::{
    TraceDriverContext, TraceDriverSpec, TraceIdentity, WorkerIdentity,
};
use aiperf_runtime::graph::model::{ExecutableGraphNode, GraphTraceProgram};
use futures::executor::block_on;
use serde_json::json;

#[test]
fn native_source_lowers_to_the_existing_trace_program_type() {
    let task = native_task_fixture(
        br#"{
  "schema_version": "1.0",
  "trace_id": "model-tool-loop",
  "stage_bound": 2,
  "channels": {
    "model_output": { "type": "messages", "reducer": "add_messages" },
    "tool_output": { "type": "text", "reducer": "overwrite" }
  },
  "nodes": [
    { "id": "model", "kind": "model", "binding": "primary", "output": "model_output", "max_tokens": 23 },
    { "id": "tool", "kind": "tool", "adapter": "tool-adapter", "operation": "inspect", "output": "tool_output" }
  ],
  "edges": [
    { "source": "START", "target": "model" },
    { "source": "model", "target": "tool" },
    { "source": "tool", "target": "END" }
  ],
  "terminal_outputs": ["tool_output"]
}"#,
    );

    let imported = import_native_task(task.path());
    let native = imported
        .package
        .native_graph()
        .expect("fixture contains a NativeGraph package");
    let (program, report): (GraphTraceProgram, _) = lower_native_graph(native)
        .expect("bounded fixture must lower into the existing graph program");

    assert_eq!(program.driver.kind, "native_graph_live");
    assert_eq!(program.profiling.trace.id, "model-tool-loop");
    assert!(
        program
            .driver
            .data
            .get("control_flow")
            .and_then(|value| value.get("static_projection_digest"))
            .and_then(serde_json::Value::as_str)
            .is_some_and(|digest| digest.starts_with("blake3:")),
        "the lowerer binds the canonical static projection into its immutable control contract"
    );
    assert_eq!(
        program
            .driver
            .data
            .get("control_flow")
            .and_then(|value| value.get("stage_bound")),
        Some(&json!(2)),
        "the lowerer retains the source budget in its immutable control contract"
    );
    assert_eq!(
        program
            .driver
            .data
            .get("control_flow")
            .and_then(|value| value.get("stage_node_ids")),
        Some(&json!(["model", "tool"])),
        "the immutable contract binds every executable source node"
    );
    assert_eq!(
        program
            .driver
            .data
            .get("control_flow")
            .and_then(|value| value.get("stage_channel_ids")),
        Some(&json!(["model_output", "tool_output"])),
        "the immutable contract binds every stage channel"
    );
    assert_eq!(
        program
            .driver
            .data
            .get("control_flow")
            .and_then(|value| value.get("loops")),
        Some(&json!([])),
        "Task 6 records only the empty typed Task-8 control reservations"
    );
    assert!(report.nodes().all(|node| node.is_exact()));
    assert!(matches!(
        program.profiling.graph.nodes.get("model"),
        Some(ExecutableGraphNode::Llm(node))
            if node.metadata.get("native_graph.binding") == Some(&json!("primary"))
    ));
    assert!(matches!(
        program.profiling.graph.nodes.get("tool"),
        Some(ExecutableGraphNode::Tool(node)) if node.commands == ["inspect"]
    ));

    let lowerer: Box<dyn GraphLowererFactory> = Box::new(NativeGraphLowererFactory::new(native));
    let source = native
        .program_source()
        .expect("fixture retains the imported program bytes");
    assert!(
        lowerer
            .capabilities()
            .supports_source_schema("native_graph/1.0")
    );
    assert_eq!(
        lowerer
            .lower(GraphLoweringRequest {
                source_schema: "native_graph/1.0",
                execution_profile: "native_graph",
                source: source.bytes(),
            })
            .expect("registered generic lowerer accepts its immutable package snapshot")
            .driver
            .kind,
        "native_graph_live"
    );
}

#[test]
fn untyped_live_control_collections_are_refused_by_the_strict_source_dto() {
    for (collection, declaration) in [
        ("branches", "[{\"future\":\"branch\"}]"),
        ("joins", "[{\"future\":\"join\"}]"),
        ("loops", "[{\"future\":\"loop\"}]"),
    ] {
        let source = format!(
            r#"{{
  "schema_version": "1.0", "trace_id": "live-{collection}", "stage_bound": 1,
  "channels": {{ "output": {{ "type": "messages", "reducer": "add_messages" }} }},
  "nodes": [{{ "id": "model", "kind": "model", "binding": "primary", "output": "output" }}],
  "edges": [{{ "source": "START", "target": "model" }}, {{ "source": "model", "target": "END" }}],
  "{collection}": {declaration},
  "terminal_outputs": []
}}"#,
        );
        let task = native_task_fixture(source.as_bytes());
        let imported = import_native_task(task.path());
        let native = imported
            .package
            .native_graph()
            .expect("fixture contains a NativeGraph package");
        assert!(matches!(
            lower_native_graph(native),
            Err(NativeGraphLoweringError::InvalidSource(_))
        ));
    }
}

#[test]
fn live_driver_rejects_mutated_or_detached_source_provenance() {
    let task = native_task_fixture(
        br#"{
  "schema_version": "1.0", "trace_id": "provenance", "stage_bound": 1,
  "channels": { "output": { "type": "messages", "reducer": "add_messages" } },
  "nodes": [{ "id": "model", "kind": "model", "binding": "primary", "output": "output" }],
  "edges": [{ "source": "START", "target": "model" }, { "source": "model", "target": "END" }],
  "terminal_outputs": []
}"#,
    );
    let imported = import_native_task(task.path());
    let native = imported
        .package
        .native_graph()
        .expect("fixture contains a NativeGraph package");
    let (program, _) = lower_native_graph(native).expect("fixture lowers");
    let trace = TraceIdentity {
        run_id: "run".into(),
        trajectory_id: "trajectory".into(),
        trace_id: "provenance".into(),
    };
    let factories = native_execution_factories();

    let mut mutated = program.driver.clone();
    *mutated
        .data
        .get_mut("control_flow")
        .and_then(|value| value.get_mut("source_snapshot_digest"))
        .expect("lowered contract records a source digest") =
        json!("blake3:0000000000000000000000000000000000000000000000000000000000000000");
    let mutation =
        match factories
            .trace_driver()
            .create(WorkerIdentity { worker_id: 0 }, &trace, &mutated)
        {
            Ok(_) => panic!("a caller cannot substitute its own source identity"),
            Err(error) => error,
        };
    assert!(mutation.to_string().contains("immutable source provenance"));

    let detached = serde_json::from_value::<TraceDriverSpec>(
        serde_json::to_value(&program.driver).expect("driver data is serializable"),
    )
    .expect("serialized caller data still has the public driver shape");
    let detached =
        match factories
            .trace_driver()
            .create(WorkerIdentity { worker_id: 0 }, &trace, &detached)
        {
            Ok(_) => panic!("serialized caller data cannot recreate lowering provenance"),
            Err(error) => error,
        };
    assert!(detached.to_string().contains("immutable source provenance"));
}

#[test]
fn lowered_program_retains_a_typed_bounded_control_flow_contract() {
    let task = native_task_fixture(
        br#"{
  "schema_version": "1.0", "trace_id": "typed-contract", "stage_bound": 1,
  "channels": { "output": { "type": "messages", "reducer": "add_messages" } },
  "nodes": [{ "id": "model", "kind": "model", "binding": "primary", "output": "output" }],
  "edges": [{ "source": "START", "target": "model" }, { "source": "model", "target": "END" }],
  "terminal_outputs": ["output"]
}"#,
    );
    let imported = import_native_task(task.path());
    let native = imported
        .package
        .native_graph()
        .expect("fixture contains a NativeGraph package");
    let (program, _) = lower_native_graph(native).expect("fixture lowers");
    let contract = serde_json::from_value::<NativeGraphControlContract>(
        program
            .driver
            .data
            .get("control_flow")
            .cloned()
            .expect("lowered program retains a control-flow contract"),
    )
    .expect("control-flow contract remains typed");
    assert_eq!(
        contract.source_snapshot_digest,
        native.program_source().unwrap().digest().as_str()
    );
    assert_eq!(contract.stage_bound.get(), 1);
    assert_eq!(contract.terminal_outputs, ["output"]);
    assert!(contract.static_projection_digest.starts_with("blake3:"));
    assert!(contract.branches.is_empty());
    assert!(contract.joins.is_empty());
    assert!(contract.loops.is_empty());
}

#[test]
fn lowered_terminal_requires_frozen_handles_before_stage_execution() {
    let task = native_task_fixture(
        br#"{
  "schema_version": "1.0", "trace_id": "terminal-preflight", "stage_bound": 1,
  "channels": { "output": { "type": "messages", "reducer": "add_messages" } },
  "nodes": [{ "id": "model", "kind": "model", "binding": "primary", "output": "output" }],
  "edges": [{ "source": "START", "target": "model" }, { "source": "model", "target": "END" }],
  "terminal_outputs": ["output"]
}"#,
    );
    let imported = import_native_task(task.path());
    let native = imported
        .package
        .native_graph()
        .expect("fixture contains a NativeGraph package");
    let (program, _) = lower_native_graph(native).expect("fixture lowers");
    let trace = TraceIdentity {
        run_id: "run".into(),
        trajectory_id: "trajectory".into(),
        trace_id: "terminal-preflight".into(),
    };
    let factories = native_execution_factories();
    let mut driver = factories
        .trace_driver()
        .create(WorkerIdentity { worker_id: 0 }, &trace, &program.driver)
        .expect("lowered program selects its live driver");

    let error = block_on(driver.open(&program, &TraceDriverContext::metadata_only(&trace)))
        .expect_err("terminal declarations require frozen handles before stage execution");
    assert!(error.to_string().contains("frozen terminal handles"));
}

#[test]
fn unbounded_static_cycle_is_refused_before_driver_creation() {
    let task = native_task_fixture(
        br#"{
  "schema_version": "1.0",
  "trace_id": "unbounded",
  "stage_bound": 1,
  "channels": { "output": { "type": "text", "reducer": "overwrite" } },
  "nodes": [
    { "id": "model", "kind": "model", "binding": "primary", "output": "output" }
  ],
  "edges": [
    { "source": "START", "target": "model" },
    { "source": "model", "target": "END" },
    { "source": "model", "target": "model" }
  ],
  "terminal_outputs": ["output"]
}"#,
    );

    let imported = import_native_task(task.path());
    let native = imported
        .package
        .native_graph()
        .expect("fixture contains a NativeGraph package");

    assert!(matches!(
        lower_native_graph(native),
        Err(NativeGraphLoweringError::UnboundedCycle { .. })
    ));
}

#[test]
fn untyped_source_feedback_is_refused_by_the_strict_loop_dto() {
    let task = native_task_fixture(
        br#"{
  "schema_version": "1.0", "trace_id": "feedback", "stage_bound": 2,
  "channels": { "output": { "type": "messages", "reducer": "add_messages" } },
  "nodes": [{ "id": "model", "kind": "model", "binding": "primary", "output": "output" }],
  "edges": [{ "source": "START", "target": "model" }, { "source": "model", "target": "model" }],
  "loops": [{ "future": "loop" }],
  "terminal_outputs": ["output"]
}"#,
    );
    let imported = import_native_task(task.path());
    let native = imported
        .package
        .native_graph()
        .expect("NativeGraph fixture");
    assert!(matches!(
        lower_native_graph(native),
        Err(NativeGraphLoweringError::InvalidSource(_))
    ));
}

#[test]
fn dangling_terminal_output_is_refused_before_driver_creation() {
    let task = native_task_fixture(
        br#"{
  "schema_version": "1.0", "trace_id": "dangling", "stage_bound": 1,
  "channels": { "output": { "type": "messages", "reducer": "add_messages" } },
  "nodes": [{ "id": "model", "kind": "model", "binding": "primary", "output": "output" }],
  "edges": [{ "source": "START", "target": "model" }, { "source": "model", "target": "END" }],
  "terminal_outputs": ["missing"]
}"#,
    );
    let imported = import_native_task(task.path());
    let native = imported
        .package
        .native_graph()
        .expect("NativeGraph fixture");
    assert!(matches!(
        lower_native_graph(native),
        Err(NativeGraphLoweringError::InvalidClosure(_))
    ));
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
    fs::write(task.path().join("instruction.md"), b"Do work.\n").expect("task instruction");
    fs::write(task.path().join("tests/test.sh"), b"exit 0\n").expect("task verifier");
    fs::write(
        task.path().join("task.toml"),
        r#"schema_version = "1.1"

[task]
name = "example/native-graph-lowering"

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
streaming = true
request_timeout_ms = 30000
capture = "metadata"

[model_bindings.tokenizer]
type = "local"
name = "builtin"
revision = "main"
apply_chat_template = true

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
