// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Public behavior coverage for recorded-agent Graph-IR lowering.

use std::collections::BTreeMap;
use std::path::PathBuf;

use aiperf_runtime::dataset::{Handle, Payload};
use aiperf_runtime::graph::input::GraphInputWarning;
use aiperf_runtime::graph::model::{ExecutableGraphNode, PromptItem};
use aiperf_runtime::graph::recorded::agent_recording::{
    BuiltinReplayRequestProfileResolver, ExpectedCorpusShape, RecordedAgentEvent,
    RecordedAgentLoweringError, RecordedAgentRecording, RecordedAgentReplayManifest,
    RecordedProviderRequest, ReplayRequestProfile, ReplayRequestProfileResolver,
    ReplayTaskIdentity, ValidatedRecordedAgentCorpus, ValidatedRecordedAgentTrace,
    lower_recorded_agent_corpus,
};
use aiperf_runtime::graph::segment::SegmentPool;
use serde_json::{Map, Value, json, value::RawValue};

struct FixtureProfileResolver {
    execute_tools: bool,
}

impl ReplayRequestProfileResolver for FixtureProfileResolver {
    fn resolve(
        &self,
        task: &ReplayTaskIdentity,
    ) -> Result<ReplayRequestProfile, RecordedAgentLoweringError> {
        let mut additional_body = Map::new();
        if task.adapter == "swebench" {
            additional_body.insert("temperature".into(), json!(0.7));
            additional_body.insert("top_p".into(), json!(0.8));
            additional_body.insert("top_k".into(), json!(20));
            additional_body.insert("min_p".into(), json!(0.0));
            additional_body.insert("parallel_tool_calls".into(), json!(true));
        }
        Ok(ReplayRequestProfile {
            identity: format!("fixture:{}", task.adapter),
            streaming: true,
            fallback_max_tokens: 99,
            execute_tools: self.execute_tools,
            use_recorded_model: true,
            use_recorded_sampling: true,
            is_standard_scenario: task.adapter == "swebench",
            additional_body,
            warning_facts: Vec::new(),
        })
    }
}

#[test]
fn lowerer_keeps_messages_tools_and_trailing_tool_node_in_order() {
    let mut pool = SegmentPool::new();
    let bundle = lower_recorded_agent_corpus(
        &fixture_corpus(),
        &FixtureProfileResolver {
            execute_tools: true,
        },
        &mut pool,
    )
    .expect("valid recording lowers");
    let program = &bundle.programs[0];

    assert_eq!(program.profiling.graph.llm_node_count(), 2);
    assert!(matches!(
        program.profiling.graph.nodes["tool_1"],
        ExecutableGraphNode::Tool(_)
    ));
    assert!(matches!(
        program.profiling.graph.nodes["tool_2"],
        ExecutableGraphNode::Tool(_)
    ));
    assert_eq!(
        program
            .profiling
            .graph
            .edges
            .iter()
            .map(|edge| (edge.source.as_str(), edge.target.as_str()))
            .collect::<Vec<_>>(),
        vec![
            ("START", "llm_0"),
            ("llm_0", "tool_1"),
            ("tool_1", "llm_1"),
            ("llm_1", "tool_2"),
            ("tool_2", "END"),
        ]
    );
    assert!(
        program
            .profiling
            .graph
            .edges
            .iter()
            .all(|edge| edge.delay_after_predecessor_us.is_none())
    );

    let llm0 = program.profiling.graph.nodes["llm_0"]
        .as_llm()
        .expect("first lowered node is an LLM");
    assert_eq!(llm0.max_tokens, Some(5));
    assert_eq!(
        llm0.request
            .as_ref()
            .and_then(|request| request.model.as_deref()),
        Some("recorded-model")
    );
    let message = llm0.items.first().expect("recorded message segment");
    let PromptItem::Seg { seg } = message else {
        panic!("recorded message must be a direct segment")
    };
    let Payload::Message { wire, .. } = bundle.segments.get(*seg).expect("message segment") else {
        panic!("recorded message must retain a pre-serialized message")
    };
    assert_eq!(
        wire.as_ref(),
        br#"{ "z" : 0, "role" : "user", "content" : [{ "type" : "text", "text" : "one" }], "extra" : { "b" : 2, "a" : 1 } }"#
    );

    let tools = llm0
        .request
        .as_ref()
        .and_then(|request| request.tools)
        .expect("recorded tools are retained");
    let Payload::Raw { wire } = bundle.segments.get(tools).expect("tools segment") else {
        panic!("recorded tools must retain raw bytes")
    };
    assert_eq!(
        wire.as_ref(),
        br#"[ { "type" : "function", "function" : { "name" : "bash", "parameters" : { "z" : 1, "a" : 2 } } } ]"#
    );

    let request_fields = llm0
        .request
        .as_ref()
        .and_then(|request| request.additional_body)
        .expect("SWE-Bench profile fields are retained");
    let Payload::Raw { wire } = bundle.segments.get(request_fields).expect("request fields") else {
        panic!("profile fields must retain raw bytes")
    };
    assert_eq!(
        serde_json::from_slice::<Value>(wire).expect("profile JSON"),
        json!({
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "min_p": 0.0,
            "parallel_tool_calls": true,
        })
    );
    assert_eq!(
        program.profiling.graph.nodes["llm_1"]
            .as_llm()
            .expect("second lowered node is an LLM")
            .max_tokens,
        Some(99),
        "zero recorded completion usage uses the resolved fallback cap"
    );
}

#[test]
fn lowerer_uses_relative_model_gap_when_tools_are_disabled() {
    let mut pool = SegmentPool::new();
    let bundle = lower_recorded_agent_corpus(
        &fixture_corpus(),
        &FixtureProfileResolver {
            execute_tools: false,
        },
        &mut pool,
    )
    .expect("valid recording lowers");
    let graph = &bundle.programs[0].profiling.graph;

    assert_eq!(graph.total_node_count(), 2);
    let edge = graph
        .edges
        .iter()
        .find(|edge| edge.source == "llm_0" && edge.target == "llm_1")
        .expect("LLM dependency edge");
    assert_eq!(edge.delay_after_predecessor_us, Some(9_000_000.0));
}

#[test]
fn lowerer_materializes_manifest_extra_body_without_reserializing() {
    let expected = br#"{ "z" : { "second" : 2, "first" : 1 }, "alpha" : [ 3, 2, 1 ] }"#;
    let mut corpus = fixture_corpus();
    corpus.traces[0].identity = Some(ReplayTaskIdentity {
        adapter: "pinchbench".into(),
        family: "fixture".into(),
        task_id: "fixture-task".into(),
        primary_role: None,
    });
    corpus.manifest = Some(manifest_with_extra(
        std::str::from_utf8(expected).expect("fixture is UTF-8"),
    ));
    let resolver = BuiltinReplayRequestProfileResolver::new(true, 99, false, false, false, false)
        .expect("positive fallback cap");
    let mut pool = SegmentPool::new();

    let bundle = lower_recorded_agent_corpus(&corpus, &resolver, &mut pool)
        .expect("manifest recording lowers");
    let additional_body = bundle.programs[0].profiling.graph.nodes["llm_0"]
        .as_llm()
        .and_then(|node| node.request.as_ref())
        .and_then(|request| request.additional_body)
        .expect("manifest body is retained");
    let Payload::Raw { wire } = bundle
        .segments
        .get(additional_body)
        .expect("additional-body segment")
    else {
        panic!("additional body must retain raw JSON")
    };

    assert_eq!(wire.as_ref(), expected);
}

#[test]
fn lowerer_merges_profile_fields_without_normalizing_manifest_extra_body() {
    let extra = br#"{ "top_p" : 0.42, "custom" : { "second" : 2, "first" : 1 } }"#;
    let expected = br#"{"temperature":0.7,"top_k":20,"min_p":0,"repeat_penalty":1.05,"parallel_tool_calls":true, "top_p" : 0.42, "custom" : { "second" : 2, "first" : 1 } }"#;
    let mut corpus = fixture_corpus();
    corpus.manifest = Some(manifest_with_extra(
        std::str::from_utf8(extra).expect("fixture is UTF-8"),
    ));
    let resolver = BuiltinReplayRequestProfileResolver::new(true, 99, false, false, false, true)
        .expect("positive fallback cap");
    let mut pool = SegmentPool::new();

    let bundle = lower_recorded_agent_corpus(&corpus, &resolver, &mut pool)
        .expect("manifest recording lowers");
    let additional_body = bundle.programs[0].profiling.graph.nodes["llm_0"]
        .as_llm()
        .and_then(|node| node.request.as_ref())
        .and_then(|request| request.additional_body)
        .expect("merged body is retained");
    let Payload::Raw { wire } = bundle
        .segments
        .get(additional_body)
        .expect("additional-body segment")
    else {
        panic!("additional body must retain raw JSON")
    };

    assert_eq!(wire.as_ref(), expected);
}

#[test]
fn lowerer_reuses_a_shared_message_prefix_handle() {
    let mut corpus = fixture_corpus();
    corpus.traces[0].recording.events[0]
        .provider_request
        .as_mut()
        .expect("first model request")
        .messages = Some(vec![
        raw(r#"{ "role" : "system", "content" : "shared" }"#),
        raw(r#"{"role":"user","content":"first"}"#),
    ]);
    corpus.traces[0].recording.events[2]
        .provider_request
        .as_mut()
        .expect("second model request")
        .messages = Some(vec![
        raw(r#"{ "role" : "system", "content" : "shared" }"#),
        raw(r#"{"role":"user","content":"second"}"#),
    ]);
    let mut pool = SegmentPool::new();
    let bundle = lower_recorded_agent_corpus(
        &corpus,
        &FixtureProfileResolver {
            execute_tools: false,
        },
        &mut pool,
    )
    .expect("valid recording lowers");
    let first = segment_handle(&bundle.programs[0].profiling.graph.nodes["llm_0"]);
    let second = segment_handle(&bundle.programs[0].profiling.graph.nodes["llm_1"]);

    assert_eq!(first[0], second[0]);
    assert_eq!(
        bundle
            .segments
            .segment(first[1])
            .expect("first child")
            .parent,
        Some(first[0])
    );
    assert_eq!(
        bundle
            .segments
            .segment(second[1])
            .expect("second child")
            .parent,
        Some(second[0])
    );
}

#[test]
fn pinchbench_has_no_inferred_profile_body() {
    let mut corpus = fixture_corpus();
    corpus.traces[0].identity = Some(ReplayTaskIdentity {
        adapter: "pinchbench".into(),
        family: "fixture".into(),
        task_id: "fixture-task".into(),
        primary_role: None,
    });
    let resolver = BuiltinReplayRequestProfileResolver::new(true, 99, false, false, false, false)
        .expect("positive fallback cap");
    let mut pool = SegmentPool::new();

    let bundle = lower_recorded_agent_corpus(&corpus, &resolver, &mut pool)
        .expect("PinchBench recording lowers");

    assert!(
        bundle.programs[0].profiling.graph.nodes["llm_0"]
            .as_llm()
            .and_then(|node| node.request.as_ref())
            .is_some_and(|request| request.additional_body.is_none())
    );
}

#[test]
fn lowerer_returns_one_deterministic_unknown_adapter_warning_fact() {
    let mut corpus = fixture_corpus();
    let unknown = ReplayTaskIdentity {
        adapter: "unrecognized-adapter".into(),
        family: "fixture".into(),
        task_id: "first-task".into(),
        primary_role: None,
    };
    corpus.traces[0].identity = Some(unknown.clone());
    let mut second = corpus.traces[0].clone();
    second.trace_id = "second-trace".into();
    second.identity = Some(ReplayTaskIdentity {
        task_id: "second-task".into(),
        ..unknown
    });
    corpus.traces.push(second);
    let resolver = BuiltinReplayRequestProfileResolver::new(true, 99, false, false, false, false)
        .expect("positive fallback cap");
    let mut pool = SegmentPool::new();

    let bundle = lower_recorded_agent_corpus(&corpus, &resolver, &mut pool)
        .expect("unknown-family recordings still lower");

    assert_eq!(
        bundle.metadata.warning_facts,
        vec![GraphInputWarning::new(
            "agent_recording_unknown_adapter",
            BTreeMap::from([("adapter".into(), "unrecognized-adapter".into())]),
        )]
    );
}

#[test]
fn builtin_profile_applies_runner_sampling_and_optional_recorded_overrides() {
    let mut pool = SegmentPool::new();
    let resolver = BuiltinReplayRequestProfileResolver::new(true, 99, false, true, true, false)
        .expect("positive fallback cap");
    let bundle = lower_recorded_agent_corpus(&fixture_corpus(), &resolver, &mut pool)
        .expect("valid recording lowers");
    let llm = bundle.programs[0].profiling.graph.nodes["llm_0"]
        .as_llm()
        .expect("LLM node");
    let request = llm.request.as_ref().expect("typed request fields");
    let fields = request.additional_body.expect("SWE-Bench profile fields");
    let Payload::Raw { wire } = bundle.segments.get(fields).expect("request fields") else {
        panic!("request fields must remain raw bytes")
    };
    assert_eq!(
        serde_json::from_slice::<Value>(wire).expect("profile JSON"),
        json!({
            "temperature": 0.1,
            "top_p": 0.2,
            "top_k": 20,
            "min_p": 0,
            "repeat_penalty": 1.05,
            "parallel_tool_calls": true,
        })
    );
    assert_eq!(request.model.as_deref(), Some("recorded-model"));

    let standard = BuiltinReplayRequestProfileResolver::new(true, 99, false, true, true, true)
        .expect("positive fallback cap");
    let standard_bundle = lower_recorded_agent_corpus(&fixture_corpus(), &standard, &mut pool)
        .expect("valid standard scenario lowers");
    let standard_llm = standard_bundle.programs[0].profiling.graph.nodes["llm_0"]
        .as_llm()
        .expect("LLM node");
    let standard_fields = standard_llm
        .request
        .as_ref()
        .and_then(|request| request.additional_body)
        .expect("SWE-Bench profile fields");
    let Payload::Raw { wire } = standard_bundle
        .segments
        .get(standard_fields)
        .expect("request fields")
    else {
        panic!("request fields must remain raw bytes")
    };
    assert_eq!(
        serde_json::from_slice::<Value>(wire).expect("profile JSON"),
        json!({
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "min_p": 0,
            "repeat_penalty": 1.05,
            "parallel_tool_calls": true,
        })
    );
}

fn fixture_corpus() -> ValidatedRecordedAgentCorpus {
    let identity = ReplayTaskIdentity {
        adapter: "swebench".into(),
        family: "fixture".into(),
        task_id: "fixture-task".into(),
        primary_role: None,
    };
    let first_message = raw(
        r#"{ "z" : 0, "role" : "user", "content" : [{ "type" : "text", "text" : "one" }], "extra" : { "b" : 2, "a" : 1 } }"#,
    );
    let tools = raw(
        r#"[ { "type" : "function", "function" : { "name" : "bash", "parameters" : { "z" : 1, "a" : 2 } } } ]"#,
    );
    let second_message = raw(r#"{"role":"user","content":"two"}"#);
    let events = vec![
        model_event(
            1,
            10.0,
            2_000_000_000,
            vec![first_message],
            Some(tools),
            Some(5),
        ),
        tool_event(2, 11.0, "pwd"),
        model_event(3, 20.0, 1_000_000_000, vec![second_message], None, Some(0)),
        tool_event(4, 21.0, "git status --short"),
    ];
    ValidatedRecordedAgentCorpus {
        manifest: None,
        manifest_digest: None,
        traces: vec![ValidatedRecordedAgentTrace {
            trace_id: "fixture-trace".into(),
            identity: Some(identity),
            path: PathBuf::from("fixture.json"),
            digest: "fixture-digest".into(),
            image: None,
            recording: RecordedAgentRecording {
                format: "mini-swe-agent-recording-1.0".into(),
                metadata: Default::default(),
                events,
            },
        }],
        shape: ExpectedCorpusShape {
            total_isl: 0,
            isl_delta: 0,
            peak_isl: 0,
            total_osl: 5,
            model_calls: 2,
            tool_calls: 2,
            tool_duration_ms: 0.0,
            max_tool_call_duration_ms: 0.0,
            timed_out_tool_calls: 0,
        },
        recording_digests: BTreeMap::from([(
            "swebench:fixture-task".into(),
            "fixture-digest".into(),
        )]),
    }
}

fn segment_handle(node: &ExecutableGraphNode) -> Vec<Handle> {
    node.as_llm()
        .expect("LLM node")
        .items
        .iter()
        .map(|item| match item {
            PromptItem::Seg { seg } => *seg,
            _ => panic!("recorded message must be a direct segment"),
        })
        .collect()
}

fn manifest_with_extra(extra_request_body: &str) -> RecordedAgentReplayManifest {
    serde_json::from_str(&format!(
        r#"{{
            "name":"fixture", "mode":"replay",
            "defaults":{{
                "config":"mixed", "step_limit":1, "cost_limit":0.0,
                "environment_class":"mixed", "docker_network":"none", "per_inference_timeout":1.0,
                "fallback_max_output_tokens":99, "temperature":0.7, "top_p":0.8, "top_k":20,
                "min_p":0.0, "stream_for_timing":true, "raw_openai_stream_for_replay_timing":true,
                "replay_max_tokens_from_recording":true, "replay_max_tokens_margin":0,
                "extra_request_body":{extra_request_body}, "cross_run_cache_isolation":true,
                "warmup":true, "measurement_scope":"agent_run_only"
            }},
            "aggregate":{{
                "total_isl":0, "isl_delta":0, "peak_isl":0, "total_osl":0,
                "model_calls":0, "tool_calls":0, "tool_duration_ms":0.0,
                "max_tool_call_duration_ms":0.0, "timed_out_tool_calls":0
            }},
            "tasks":[], "attribution":{{}}
        }}"#
    ))
    .expect("valid manifest fixture")
}

fn model_event(
    id: u64,
    timestamp: f64,
    duration_ns: u64,
    messages: Vec<Box<RawValue>>,
    tools: Option<Box<RawValue>>,
    completion_tokens: Option<u64>,
) -> RecordedAgentEvent {
    RecordedAgentEvent {
        id,
        event_type: "model_call".into(),
        timestamp,
        duration_ns: Some(duration_ns),
        step: None,
        provider_request: Some(RecordedProviderRequest {
            messages: Some(messages),
            tools,
            model: Some("recorded-model".into()),
            temperature: Some(0.1),
            top_p: Some(0.2),
            max_tokens: Some(123),
        }),
        response_message: completion_tokens
            .map(|tokens| json!({"extra":{"response":{"usage":{"completion_tokens":tokens}}}})),
        action: None,
        error: None,
    }
}

fn raw(value: &str) -> Box<RawValue> {
    serde_json::from_str(value).expect("fixture JSON")
}

fn tool_event(id: u64, timestamp: f64, command: &str) -> RecordedAgentEvent {
    RecordedAgentEvent {
        id,
        event_type: "tool_call".into(),
        timestamp,
        duration_ns: Some(100),
        step: None,
        provider_request: None,
        response_message: None,
        action: Some(json!({"command": command})),
        error: None,
    }
}
