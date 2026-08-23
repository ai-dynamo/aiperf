// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Public behavior coverage for typed graph request materialization.
#![cfg(feature = "engine")]

use std::collections::BTreeMap;
use std::sync::Arc;

use aiperf_runtime::graph::materialize::{GraphRequestMaterializer, SegmentItemsMaterializer};
use aiperf_runtime::graph::model::{LlmNode, LlmRequestSpec};
use aiperf_runtime::graph::segment::SegmentPool;
use bytes::Bytes;
use serde_json::json;

#[test]
fn materializer_keeps_typed_request_fields_and_message_wires() {
    let mut pool = SegmentPool::new();
    let tools = pool
        .intern_raw(
            None,
            Bytes::from_static(br#"[{"type":"function","function":{"name":"bash"}}]"#),
        )
        .expect("tools segment");
    let additional_body = pool
        .intern_raw(
            None,
            Bytes::from_static(br#"{"temperature":0.7,"parallel_tool_calls":true}"#),
        )
        .expect("additional-body segment");
    let materializer = SegmentItemsMaterializer::new(Arc::new(pool.freeze()));
    let node = LlmNode {
        output: "reply".into(),
        streaming: true,
        inputs: Vec::new(),
        min_start_delay_us: None,
        max_tokens: Some(8),
        items: Vec::new(),
        request: Some(LlmRequestSpec {
            tools: Some(tools),
            model: Some("selected-model".into()),
            additional_body: Some(additional_body),
        }),
        metadata: BTreeMap::new(),
    };
    let messages = vec![Bytes::from_static(
        br#"{"role":"user","content":"keep-me"}"#,
    )];

    let request = materializer
        .materialize(&node, messages.clone())
        .expect("materialized request");

    assert_eq!(request.messages, messages);
    assert_eq!(
        request.tools,
        Some(Bytes::from_static(
            br#"[{"type":"function","function":{"name":"bash"}}]"#
        ))
    );
    assert_eq!(request.model.as_deref(), Some("selected-model"));
    assert_eq!(
        request.additional_body,
        Some(Bytes::from_static(
            br#"{"temperature":0.7,"parallel_tool_calls":true}"#
        ))
    );
    assert_eq!(request.max_tokens, Some(8));
    assert!(request.streaming);
}

#[test]
fn materializer_rejects_reserved_extra_body_keys() {
    let mut pool = SegmentPool::new();
    let additional_body = pool
        .intern_raw(
            None,
            Bytes::from(serde_json::to_vec(&json!({"messages": []})).expect("JSON")),
        )
        .expect("additional-body segment");
    let materializer = SegmentItemsMaterializer::new(Arc::new(pool.freeze()));
    let node = LlmNode {
        output: "reply".into(),
        streaming: true,
        inputs: Vec::new(),
        min_start_delay_us: None,
        max_tokens: Some(8),
        items: Vec::new(),
        request: Some(LlmRequestSpec {
            tools: None,
            model: None,
            additional_body: Some(additional_body),
        }),
        metadata: BTreeMap::new(),
    };

    assert!(
        materializer
            .materialize(&node, vec![Bytes::from_static(br#"{"role":"user"}"#)])
            .is_err()
    );
}
