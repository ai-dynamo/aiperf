// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Staged live NativeGraph trace-driver contracts.

use std::{collections::BTreeMap, fs};

use aiperf_runtime::engine::execution_factories::native_execution_factories;
use aiperf_runtime::eval::{
    ArtifactDigest, HarborImporter, HarborSource, NativeSourceAcquirer, lower_native_graph,
};
use aiperf_runtime::graph::driver::{
    TraceDriverContext, TraceIdentity, TraceStageDirective, TraceStageResult, WorkerIdentity,
};
use aiperf_runtime::graph::model::GraphTraceProgram;
use aiperf_runtime::graph::sink::GraphReplyStatus;
use aiperf_runtime::graph::supplement::TraceTerminalSupplement;
use futures::executor::block_on;
use serde_json::json;

#[test]
fn registered_live_driver_progresses_through_a_bounded_graph_stage() {
    let program = live_program();
    let trace = TraceIdentity {
        run_id: "run".into(),
        trajectory_id: "trajectory".into(),
        trace_id: "live-trace".into(),
    };
    let factories = native_execution_factories();
    factories
        .trace_driver()
        .capabilities(&program.driver)
        .expect("native live driver is registered at bootstrap");
    let mut driver = factories
        .trace_driver()
        .create(WorkerIdentity { worker_id: 4 }, &trace, &program.driver)
        .expect("registered live driver creates a trace-local cursor");
    let context = TraceDriverContext::metadata_only(&trace);

    block_on(async {
        driver
            .open(&program, &context)
            .await
            .expect("live driver accepts its lowered program");
        let directive = driver
            .next_stage(&context)
            .await
            .expect("first staged directive")
            .expect("a lowered NativeGraph has its initial stage");
        let TraceStageDirective::Execute(plan) = directive else {
            panic!("the first live directive must execute the lowered graph plan");
        };
        assert_eq!(plan.trace.id, "live-trace");

        driver
            .observe_stage(TraceStageResult {
                plan_identity: "live-trace::stage-0".into(),
                terminal_status: GraphReplyStatus::Completed,
                channels: BTreeMap::new(),
                output_handles: BTreeMap::new(),
            })
            .await
            .expect("completed stage is accepted");
        assert!(matches!(
            driver.next_stage(&context).await.expect("terminal directive"),
            Some(TraceStageDirective::Complete(supplement))
                if supplement.driver_kind == "native_graph_live"
                    && supplement.completed
                    && supplement.terminal_outputs.is_empty()
        ));
        assert!(
            driver
                .next_stage(&context)
                .await
                .expect("completed drivers stay terminal")
                .is_none()
        );
        driver.close().await.expect("driver cleanup");
    });
}

#[test]
fn live_driver_refuses_a_program_that_does_not_match_its_selected_stage_bound() {
    let mut program = live_program();
    let selected_spec = program.driver.clone();
    let trace = TraceIdentity {
        run_id: "run".into(),
        trajectory_id: "trajectory".into(),
        trace_id: "live-trace".into(),
    };
    let factories = native_execution_factories();
    let mut driver = factories
        .trace_driver()
        .create(WorkerIdentity { worker_id: 4 }, &trace, &selected_spec)
        .expect("registered live driver creates from the selected driver spec");
    program
        .driver
        .data
        .get_mut("control_flow")
        .and_then(serde_json::Value::as_object_mut)
        .expect("live program retains a typed control-flow contract")
        .insert("stage_bound".into(), json!(2));

    let error = block_on(driver.open(&program, &TraceDriverContext::metadata_only(&trace)))
        .expect_err("a driver must not execute a different stage budget than its selected spec");
    assert!(
        error
            .to_string()
            .contains("stage bound does not match its selected program")
    );
}

#[test]
fn live_driver_refuses_a_program_with_a_mutated_static_projection() {
    let mut program = live_program();
    let selected_spec = program.driver.clone();
    let trace = TraceIdentity {
        run_id: "run".into(),
        trajectory_id: "trajectory".into(),
        trace_id: "live-trace".into(),
    };
    let factories = native_execution_factories();
    let mut driver = factories
        .trace_driver()
        .create(WorkerIdentity { worker_id: 4 }, &trace, &selected_spec)
        .expect("registered live driver creates from the selected driver spec");
    program
        .profiling
        .graph
        .nodes
        .get_mut("model")
        .and_then(|node| node.as_llm_mut())
        .expect("live fixture contains its model node")
        .max_tokens = Some(2);

    let error = block_on(driver.open(&program, &TraceDriverContext::metadata_only(&trace)))
        .expect_err("a driver must not execute a mutated static projection");
    assert!(
        error
            .to_string()
            .contains("differs from the imported static projection")
    );
}

#[test]
fn live_driver_refuses_a_public_contract_recomputed_for_a_mutated_static_projection() {
    let mut program = live_program();
    program
        .profiling
        .graph
        .nodes
        .get_mut("model")
        .and_then(|node| node.as_llm_mut())
        .expect("live fixture contains its model node")
        .max_tokens = Some(2);
    let static_projection_digest = ArtifactDigest::from_bytes(
        &serde_json::to_vec(&program.profiling)
            .expect("mutated static projection remains serializable"),
    )
    .as_str()
    .to_owned();
    let stage_node_ids = program
        .profiling
        .graph
        .nodes
        .keys()
        .cloned()
        .collect::<Vec<_>>();
    let stage_channel_ids = program
        .profiling
        .graph
        .state
        .keys()
        .cloned()
        .collect::<Vec<_>>();
    let control_flow = program
        .driver
        .data
        .get_mut("control_flow")
        .and_then(serde_json::Value::as_object_mut)
        .expect("live program retains a typed control-flow contract");
    control_flow.insert(
        "static_projection_digest".into(),
        json!(static_projection_digest),
    );
    control_flow.insert("stage_node_ids".into(), json!(stage_node_ids));
    control_flow.insert("stage_channel_ids".into(), json!(stage_channel_ids));

    let trace = TraceIdentity {
        run_id: "run".into(),
        trajectory_id: "trajectory".into(),
        trace_id: "live-trace".into(),
    };
    let error = match native_execution_factories().trace_driver().create(
        WorkerIdentity { worker_id: 4 },
        &trace,
        &program.driver,
    ) {
        Ok(_) => {
            panic!("a caller cannot re-authorize a mutated projection with public contract data")
        }
        Err(error) => error,
    };
    assert!(
        error
            .to_string()
            .contains("immutable static projection provenance"),
        "the factory must reject before graph dispatch"
    );
}

#[test]
fn live_driver_requires_a_typed_control_flow_contract() {
    let mut program = live_program();
    program.driver.data.remove("control_flow");
    let factories = native_execution_factories();
    let trace = TraceIdentity {
        run_id: "run".into(),
        trajectory_id: "trajectory".into(),
        trace_id: "live-trace".into(),
    };
    assert!(
        factories
            .trace_driver()
            .create(WorkerIdentity { worker_id: 0 }, &trace, &program.driver)
            .is_err()
    );
}

#[test]
fn empty_terminal_outputs_preserve_legacy_supplement_wire() {
    let supplement = TraceTerminalSupplement::new(
        "run".into(),
        "trajectory".into(),
        "trace".into(),
        0,
        "static_graph",
    );
    assert!(
        !serde_json::to_string(&supplement)
            .expect("supplement serializes")
            .contains("terminal_outputs")
    );
}

fn live_program() -> GraphTraceProgram {
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
name = "example/native-graph-live-driver"

[native_graph]
profile = "native_graph"
program = "agent_graph.json"
model_bindings = "models.toml"
adapter_manifest = "adapters.toml"
"#,
    )
    .expect("task manifest");
    fs::write(
        task.path().join("agent_graph.json"),
        br#"{
  "schema_version": "1.0", "trace_id": "live-trace", "stage_bound": 1,
  "channels": { "output": { "type": "messages", "reducer": "add_messages" } },
  "nodes": [{ "id": "model", "kind": "model", "binding": "primary", "output": "output" }],
  "edges": [{ "source": "START", "target": "model" }, { "source": "model", "target": "END" }],
  "terminal_outputs": []
}"#,
    )
    .expect("graph source");
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
    let source = HarborSource::local(task.path().to_string_lossy())
        .expect("temporary task path is a valid source");
    let imported = HarborImporter::new(&NativeSourceAcquirer)
        .import(&source)
        .expect("fixture imports");
    let native = imported
        .package
        .native_graph()
        .expect("fixture contains a NativeGraph package");
    lower_native_graph(native).expect("fixture lowers").0
}
