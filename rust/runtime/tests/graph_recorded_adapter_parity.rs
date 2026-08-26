// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Byte-exact replay parity across recorded trace formats.
//!
//! Both fixtures pin the same `content_root_seed`, and recorded content
//! synthesis derives its stream from that seed alone (the SHA-256 child seed and
//! CPython-parity window draw in `graph::recorded::content`). With that stream
//! held common, both source formats must lower to the same topology and exact
//! materialized message bytes.
#![cfg(feature = "engine")]

use std::collections::BTreeMap;

use aiperf_runtime::dataset::{
    DatasetSource, Handle, LoadConfig, Payload, SegmentStore, TiktokenTokenizer,
};
use aiperf_runtime::graph::materialize::{PromptMaterializer, SegmentItemsMaterializer};
use aiperf_runtime::graph::model::{GraphRecord, LlmNode};
use aiperf_runtime::graph::recorded::{
    PromptCorpus, RecordedTraceInputConfig, compile_dynamo_trace_input, compile_weka_trace_input,
};
use bytes::Bytes;
use serde_json::{Value, json};

fn config(records: Value) -> RecordedTraceInputConfig {
    RecordedTraceInputConfig {
        load: LoadConfig::new(DatasetSource::Inline(records)),
        root_limit: None,
        max_context_length: None,
        max_osl: None,
        idle_gap_cap_seconds: Some(60.0),
        prompt_corpus: PromptCorpus::Sonnet,
        content_root_seed: 20_260_707,
    }
}

fn weka_fixture() -> Value {
    json!({
        "id": "root",
        "models": ["recorded-model"],
        "block_size": 16,
        "hash_id_scope": "global",
        "requests": [
            {
                "t": 0.0,
                "type": "s",
                "model": "recorded-model",
                "in": 37,
                "out": 3,
                "hash_ids": [101, 102],
                "api_time": 1.5,
                "ttft": 0.25
            },
            {
                "t": 0.5,
                "type": "subagent",
                "agent_id": "child",
                "subagent_type": "Explore",
                "duration_ms": null,
                "total_tokens": null,
                "tool_use_count": null,
                "status": "completed",
                "models": ["recorded-model"],
                "requests": [{
                    "t": 0.5,
                    "type": "n",
                    "model": "recorded-model",
                    "in": 16,
                    "out": 2,
                    "hash_ids": [101],
                    "api_time": 0.5
                }]
            },
            {
                "t": 2.0,
                "type": "n",
                "model": "recorded-model",
                "in": 48,
                "out": 0,
                "hash_ids": [101, 102, 103],
                "api_time": 0.4
            }
        ]
    })
}

fn dynamo_fixture() -> Value {
    Value::Array(vec![
        json!({
            "schema": "dynamo.request.trace.v1",
            "event_type": "request_end",
            "event_time_unix_ms": 1_001_500,
            "event_source": "dynamo",
            "agent_context": {"session_id": "root"},
            "request": {
                "request_id": "root-0",
                "model": "recorded-model",
                "input_tokens": 37,
                "output_tokens": 3,
                "cached_tokens": 0,
                "request_received_ms": 1_000_000,
                "total_time_ms": 1500,
                "ttft_ms": 250,
                "replay": {
                    "trace_block_size": 16,
                    "input_length": 37,
                    "input_sequence_hashes": [101, 102, 900001]
                }
            }
        }),
        json!({
            "schema": "dynamo.request.trace.v1",
            "event_type": "request_end",
            "event_time_unix_ms": 1_001_000,
            "event_source": "dynamo",
            "agent_context": {
                "session_id": "child",
                "parent_session_id": "root"
            },
            "request": {
                "request_id": "child-0",
                "model": "recorded-model",
                "input_tokens": 16,
                "output_tokens": 2,
                "cached_tokens": 0,
                "request_received_ms": 1_000_500,
                "total_time_ms": 500,
                "replay": {
                    "trace_block_size": 16,
                    "input_length": 16,
                    "input_sequence_hashes": [101]
                }
            }
        }),
        json!({
            "schema": "dynamo.request.trace.v1",
            "event_type": "request_end",
            "event_time_unix_ms": 1_002_400,
            "event_source": "dynamo",
            "agent_context": {"session_id": "root"},
            "request": {
                "request_id": "root-1",
                "model": "recorded-model",
                "input_tokens": 48,
                "output_tokens": 0,
                "cached_tokens": 0,
                "request_received_ms": 1_002_000,
                "total_time_ms": 400,
                "replay": {
                    "trace_block_size": 16,
                    "input_length": 48,
                    "input_sequence_hashes": [101, 102, 103]
                }
            }
        }),
    ])
}

fn materialized_messages(
    store: std::sync::Arc<dyn aiperf_runtime::dataset::SegmentStore>,
    node: &LlmNode,
) -> Vec<Bytes> {
    SegmentItemsMaterializer::new(store)
        .build(node, &BTreeMap::new())
        .expect("materialize recorded prompt")
}

fn assert_topology_parity(weka: &GraphRecord, dynamo: &GraphRecord) {
    assert_eq!(
        weka.nodes.keys().collect::<Vec<_>>(),
        dynamo.nodes.keys().collect::<Vec<_>>()
    );
    assert_eq!(
        serde_json::to_value(&weka.edges).unwrap(),
        serde_json::to_value(&dynamo.edges).unwrap()
    );
    assert_eq!(
        serde_json::to_value(&weka.state).unwrap(),
        serde_json::to_value(&dynamo.state).unwrap()
    );
    for node_id in weka.nodes.keys() {
        let left = weka.nodes[node_id].as_llm().unwrap();
        let right = dynamo.nodes[node_id].as_llm().unwrap();
        assert_eq!(left.streaming, right.streaming, "streaming: {node_id}");
        assert_eq!(left.max_tokens, right.max_tokens, "max_tokens: {node_id}");
        assert_eq!(
            serde_json::to_value(&left.inputs).unwrap(),
            serde_json::to_value(&right.inputs).unwrap(),
            "fan-in: {node_id}"
        );
        assert_eq!(
            left.min_start_delay_us, right.min_start_delay_us,
            "start delay: {node_id}"
        );
        for key in [
            "arrival_offset_us",
            "input_tokens",
            "model",
            "recorded_output_tokens",
            "theoretical_prefix_cache_hit_blocks",
            "theoretical_prefix_cache_total_blocks",
        ] {
            assert_eq!(
                left.metadata.get(key),
                right.metadata.get(key),
                "metadata {key}: {node_id}"
            );
        }
    }
}

fn message_segments(store: &dyn SegmentStore) -> Vec<(String, String, Vec<u8>, Vec<u32>)> {
    let mut messages = (0..store.len())
        .filter_map(|index| {
            let handle = Handle::new(u32::try_from(index).expect("test segment index"));
            let segment = store.segment(handle)?;
            let Payload::Message { role, wire, tokens } = &segment.payload else {
                return None;
            };
            Some((
                segment.id.to_hex(),
                role.as_str().to_string(),
                wire.to_vec(),
                tokens.to_vec(),
            ))
        })
        .collect::<Vec<_>>();
    messages.sort_by(|left, right| left.0.cmp(&right.0));
    messages
}

#[tokio::test]
async fn logical_weka_and_dynamo_traces_materialize_byte_identical_prompts() {
    let tokenizer = TiktokenTokenizer::builtin();
    let weka = compile_weka_trace_input(config(weka_fixture()), &tokenizer)
        .await
        .expect("compile WEKA fixture");
    let dynamo = compile_dynamo_trace_input(config(dynamo_fixture()), &tokenizer)
        .await
        .expect("compile Dynamo fixture");

    assert_eq!(weka.programs.len(), 1);
    assert_eq!(dynamo.programs.len(), 1);
    let weka_graph = &weka.programs[0].profiling.graph;
    let dynamo_graph = &dynamo.programs[0].profiling.graph;
    assert_topology_parity(weka_graph, dynamo_graph);
    assert_eq!(
        message_segments(weka.segments.as_ref()),
        message_segments(dynamo.segments.as_ref()),
        "all prompt and synthesized response message segments"
    );

    for node_id in weka_graph.nodes.keys() {
        let weka_messages = materialized_messages(
            weka.segments.clone(),
            weka_graph.nodes[node_id].as_llm().unwrap(),
        );
        let dynamo_messages = materialized_messages(
            dynamo.segments.clone(),
            dynamo_graph.nodes[node_id].as_llm().unwrap(),
        );
        assert_eq!(weka_messages, dynamo_messages, "wire messages: {node_id}");
        for message in weka_messages {
            let value: Value = serde_json::from_slice(&message).expect("valid message JSON");
            assert!(
                value["content"]
                    .as_str()
                    .is_some_and(|text| !text.is_empty())
            );
        }
    }
}
