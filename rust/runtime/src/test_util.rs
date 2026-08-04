// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Test-only helpers: an in-process mock OpenAI SSE server.

use axum::{Router, http::header, response::IntoResponse, routing::post};

/// Streams a realistic chat-completions response: a role-only opening chunk,
/// two content chunks (`a`, `b`), a finish-only chunk, authoritative usage,
/// then `[DONE]`. Only the two content chunks should be timed as output tokens.
async fn chat_handler() -> impl IntoResponse {
    let body = concat!(
        "data: {\"id\":\"x\",\"object\":\"chat.completion.chunk\",\"created\":0,\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}\n\n",
        "data: {\"id\":\"x\",\"object\":\"chat.completion.chunk\",\"created\":0,\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"a\"},\"finish_reason\":null}]}\n\n",
        "data: {\"id\":\"x\",\"object\":\"chat.completion.chunk\",\"created\":0,\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"b\"},\"finish_reason\":null}]}\n\n",
        "data: {\"id\":\"x\",\"object\":\"chat.completion.chunk\",\"created\":0,\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
        "data: {\"id\":\"x\",\"object\":\"chat.completion.chunk\",\"created\":0,\"model\":\"m\",\"choices\":[],\"usage\":{\"prompt_tokens\":3,\"completion_tokens\":2}}\n\n",
        "data: [DONE]\n\n",
    );
    ([(header::CONTENT_TYPE, "text/event-stream")], body)
}

/// Spawn the mock server on an ephemeral port and return its base URL.
pub async fn spawn_mock() -> String {
    let app = Router::new().route("/v1/chat/completions", post(chat_handler));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    format!("http://{addr}")
}

use crate::multiturn::ConversationSource;

/// A prepared-endpoint table holding the builtin streaming `chat` endpoint at
/// key 0, matching the endpoint the synthetic/native sources bind. Attach it to
/// a dispatching [`crate::transport::http::TransportSink`] via `with_prepared_endpoints` so
/// prepared turns resolve their dense endpoint key.
pub fn chat_dispatch_table() -> std::rc::Rc<crate::endpoints::PreparedEndpointTable> {
    use crate::endpoints::{
        EndpointId, EndpointRegistry, PreparedEndpointTable, RawEndpointConfig,
    };
    let endpoint = EndpointRegistry::builtin()
        .unwrap()
        .prepare(
            &EndpointId::new("chat").unwrap(),
            RawEndpointConfig {
                streaming: true,
                use_server_token_count: true,
                ..RawEndpointConfig::default()
            },
        )
        .unwrap();
    let mut table = PreparedEndpointTable::new();
    table.push(endpoint).unwrap();
    std::rc::Rc::new(table)
}

/// Build a conversation source over the live native dataset + prepared chat
/// endpoint path from inline `multi_turn` conversation JSON.
async fn build_prepared_source(
    conversations: serde_json::Value,
    model: &str,
    output_tokens: usize,
) -> Box<dyn ConversationSource> {
    use crate::dataset::{
        ComposeConfig, DatasetSource, LoadConfig, LoaderRegistry, TiktokenTokenizer,
    };
    use crate::endpoints::{
        EndpointId, EndpointRegistry, PreparedEndpointTable, RawEndpointConfig,
    };
    use crate::multiturn::{NativeDatasetConversationSource, PreparedEndpointReference};
    use crate::rng::RngRoot;
    let dataset = LoaderRegistry::with_builtin_formats()
        .unwrap()
        .build_dataset(
            Some("multi_turn"),
            &LoadConfig::new(DatasetSource::Inline(conversations)),
            &ComposeConfig::new(model, RngRoot::new(Some(1))),
            &TiktokenTokenizer::builtin(),
        )
        .await
        .unwrap();
    let registry = EndpointRegistry::builtin().unwrap();
    let endpoint_id = EndpointId::new("chat").unwrap();
    let endpoint = registry
        .prepare(
            &endpoint_id,
            RawEndpointConfig {
                streaming: true,
                use_server_token_count: true,
                ..RawEndpointConfig::default()
            },
        )
        .unwrap();
    let mut table = PreparedEndpointTable::new();
    let key = table.push(endpoint).unwrap();
    Box::new(
        NativeDatasetConversationSource::sequential_with_prepared_endpoint(
            dataset,
            model,
            output_tokens,
            std::rc::Rc::new(table),
            PreparedEndpointReference { key, endpoint_id },
        )
        .unwrap(),
    )
}

/// Single-turn corpus of `entries` conversations (`session_0000`..) wired the way
/// ONE thread of a `workers > 1` `global`-dispatch cell is: the source owns the
/// `cell_id`-mod-`cell_count` residue class for enumeration but addresses
/// absolute corpus positions when drawing.
pub async fn partitioned_single_turn_source(
    entries: usize,
    cell_id: u32,
    cell_count: u32,
    model: &str,
) -> Box<dyn ConversationSource> {
    use crate::cellular::ModuloCellPartition;
    use crate::dataset::{
        ComposeConfig, DatasetSource, LoadConfig, LoaderRegistry, TiktokenTokenizer,
    };
    use crate::endpoints::EndpointId;
    use crate::multiturn::{NativeDatasetConversationSource, PreparedEndpointReference};
    use crate::rng::RngRoot;
    let conversations = (0..entries)
        .map(|index| {
            serde_json::json!({
                "session_id": format!("session_{index:04}"),
                "turns": [{
                    "text": format!("prompt {index}"),
                    "input_length": 4,
                    "output_length": 1,
                }],
            })
        })
        .collect::<Vec<_>>();
    let dataset = LoaderRegistry::with_builtin_formats()
        .unwrap()
        .build_dataset(
            Some("multi_turn"),
            &LoadConfig::new(DatasetSource::Inline(serde_json::json!(conversations))),
            &ComposeConfig::new(model, RngRoot::new(Some(1))),
            &TiktokenTokenizer::builtin(),
        )
        .await
        .unwrap();
    let endpoint_id = EndpointId::new("chat").unwrap();
    let prepared = crate::endpoints::EndpointRegistry::builtin()
        .unwrap()
        .prepare(
            &endpoint_id,
            crate::endpoints::RawEndpointConfig {
                streaming: true,
                use_server_token_count: true,
                ..crate::endpoints::RawEndpointConfig::default()
            },
        )
        .unwrap();
    let mut table = crate::endpoints::PreparedEndpointTable::new();
    let key = table.push(prepared).unwrap();
    let resolver: std::rc::Rc<dyn crate::multiturn::PreparedTurnEndpointResolver> = std::rc::Rc::new(
        crate::multiturn::PreparedEndpointTableResolver::single(
            std::rc::Rc::new(table),
            PreparedEndpointReference { key, endpoint_id },
        )
        .unwrap(),
    );
    Box::new(
        NativeDatasetConversationSource::sequential_with_prepared_resolver_for_partition(
            dataset,
            model,
            1,
            resolver,
            Some(ModuloCellPartition::new(cell_id, cell_count).unwrap()),
            true,
        )
        .unwrap(),
    )
}

/// Synthetic multi-turn source with `turns` `input_tokens`-word prompts.
pub async fn synthetic_prepared_source(
    turns: usize,
    input_tokens: usize,
    output_tokens: usize,
    think_time_ms: Option<u64>,
    model: &str,
) -> Box<dyn ConversationSource> {
    let mut turn_objs = Vec::new();
    for index in 0..turns.max(1) {
        let mut turn = serde_json::json!({
            "text": format!("turn {index}: {}", vec!["lorem"; input_tokens].join(" ")),
            "input_length": input_tokens,
            "output_length": output_tokens,
        });
        if index > 0 {
            turn["delay"] = serde_json::json!(think_time_ms.unwrap_or(0));
        }
        turn_objs.push(turn);
    }
    build_prepared_source(
        serde_json::json!([{"session_id":"synthetic","turns": turn_objs}]),
        model,
        output_tokens,
    )
    .await
}

/// Trace-timestamped single-turn conversation source for fixed-schedule tests.
pub async fn timestamped_prepared_source(
    entries: &[(&str, f64)],
    model: &str,
) -> Box<dyn ConversationSource> {
    let convs = entries
        .iter()
        .map(|(id, timestamp)| {
            serde_json::json!({
                "session_id": id,
                "turns": [{"text": "hello", "timestamp": timestamp, "input_length": 2, "output_length": 1}],
            })
        })
        .collect::<Vec<_>>();
    build_prepared_source(serde_json::json!(convs), model, 1).await
}
