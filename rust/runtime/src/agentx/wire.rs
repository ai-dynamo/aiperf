// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Wire-request formatting for the legacy AgentX path: turns the reconstructed
//! chat prefix ([`crate::agentx::synth::ChatMessage`]) into the OpenAI
//! `/v1/chat/completions` request body an inference server receives.
//!
//! This is the bridge from reconstruction to transport — the exact bytes the
//! online engine would send at each turn's dispatch instant. Held separate from
//! the transport so the wire shape is unit-testable without a live client.

use serde_json::{Map, Value, json};

use crate::agentx::synth::ChatMessage;

/// Options controlling the chat request body.
#[derive(Debug, Clone, Default)]
pub struct ChatRequestOptions {
    /// Streaming (`stream: true`), as the AgentX scenario requires.
    pub streaming: bool,
    /// Inject `ignore_eos: true` into the body (scenario `require_ignore_eos`).
    pub ignore_eos: bool,
    /// A cache-bust marker prepended to the first message content (or appended,
    /// per its own leading/trailing whitespace).
    pub cache_bust_marker: Option<String>,
    /// Place the marker on the first user turn instead of the first message.
    pub cache_bust_first_user_turn: bool,
}

fn message_json(m: &ChatMessage) -> Value {
    let mut obj = Map::new();
    obj.insert("role".into(), json!(m.role));
    obj.insert("content".into(), json!(m.content));
    if let Some(tc) = &m.tool_calls {
        obj.insert(
            "tool_calls".into(),
            Value::Array(
                tc.iter()
                    .map(|c| {
                        json!({
                            "id": c.id,
                            "type": "function",
                            "function": { "name": c.name, "arguments": c.arguments },
                        })
                    })
                    .collect(),
            ),
        );
    }
    if let Some(id) = &m.tool_call_id {
        obj.insert("tool_call_id".into(), json!(id));
    }
    Value::Object(obj)
}

/// Build just the accumulated `messages` array value (role/content objects),
/// applying the cache-bust marker to the first message with the same placement
/// rule as [`chat_request_body`]. Used by the agentic composer to intern each
/// per-turn delta as a message-array segment for the history-accumulating
/// dispatch (so the runtime materializer concatenates deltas + live replies).
pub(crate) fn chat_messages_array(
    messages: &[ChatMessage],
    cache_bust_marker: Option<&str>,
) -> Value {
    let mut msgs: Vec<ChatMessage> = messages.to_vec();
    apply_cache_bust(&mut msgs, cache_bust_marker, false);
    Value::Array(msgs.iter().map(message_json).collect())
}

/// Build the OpenAI `/v1/chat/completions` request body for one dispatched turn.
///
/// `messages` is the full accumulated chat prefix for this turn (the endpoint
/// concatenates deltas at request time; here the caller passes the resolved
/// prefix). `max_tokens` is the recorded output cap; `model` the mapped name.
pub fn chat_request_body(
    model: &str,
    messages: &[ChatMessage],
    max_tokens: i64,
    opts: &ChatRequestOptions,
) -> Value {
    // Apply the cache-bust marker to the first message if requested.
    let mut msgs: Vec<ChatMessage> = messages.to_vec();
    apply_cache_bust(
        &mut msgs,
        opts.cache_bust_marker.as_deref(),
        opts.cache_bust_first_user_turn,
    );

    let mut body = Map::new();
    body.insert("model".into(), json!(model));
    body.insert(
        "messages".into(),
        Value::Array(msgs.iter().map(message_json).collect()),
    );
    body.insert("max_tokens".into(), json!(max_tokens.max(1)));
    body.insert("stream".into(), json!(opts.streaming));
    if opts.ignore_eos {
        body.insert("ignore_eos".into(), json!(true));
    }
    Value::Object(body)
}

fn apply_cache_bust(messages: &mut [ChatMessage], marker: Option<&str>, first_user_turn: bool) {
    let Some(marker) = marker else { return };
    let target = if first_user_turn {
        messages.iter_mut().find(|message| message.role == "user")
    } else {
        messages.first_mut()
    };
    let Some(target) = target else { return };
    if marker.starts_with('\n') {
        target.content.push_str(marker);
    } else {
        target.content = format!("{marker}{}", target.content);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_streaming_ignore_eos_body() {
        let msgs = vec![
            ChatMessage::plain("system", "sys"),
            ChatMessage::plain("user", "hello"),
        ];
        let body = chat_request_body(
            "my-model",
            &msgs,
            128,
            &ChatRequestOptions {
                streaming: true,
                ignore_eos: true,
                cache_bust_marker: None,
                cache_bust_first_user_turn: false,
            },
        );
        assert_eq!(body["model"], json!("my-model"));
        assert_eq!(body["stream"], json!(true));
        assert_eq!(body["ignore_eos"], json!(true));
        assert_eq!(body["max_tokens"], json!(128));
        assert_eq!(body["messages"][1]["content"], json!("hello"));
    }

    #[test]
    fn prefix_cache_bust_marker_prepended() {
        let msgs = vec![ChatMessage::plain("user", "hi")];
        let body = chat_request_body(
            "m",
            &msgs,
            4,
            &ChatRequestOptions {
                streaming: false,
                ignore_eos: false,
                cache_bust_marker: Some("[rid:abc]\n\n".into()),
                cache_bust_first_user_turn: false,
            },
        );
        assert_eq!(body["messages"][0]["content"], json!("[rid:abc]\n\nhi"));
        // max_tokens floors at 1.
        let z = chat_request_body("m", &msgs, 0, &ChatRequestOptions::default());
        assert_eq!(z["max_tokens"], json!(1));
    }

    #[test]
    fn first_user_turn_cache_bust_skips_system_message() {
        let msgs = vec![
            ChatMessage::plain("system", "sys"),
            ChatMessage::plain("user", "hi"),
        ];
        let body = chat_request_body(
            "m",
            &msgs,
            4,
            &ChatRequestOptions {
                cache_bust_marker: Some("[warmup]\n\n".into()),
                cache_bust_first_user_turn: true,
                ..ChatRequestOptions::default()
            },
        );
        assert_eq!(body["messages"][0]["content"], json!("sys"));
        assert_eq!(body["messages"][1]["content"], json!("[warmup]\n\nhi"));
    }
}
