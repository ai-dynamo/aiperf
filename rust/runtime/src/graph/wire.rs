// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! The wire message shape is extensible so graph dataflow remains
//! endpoint-agnostic. Each dialect supplies a `WireMessage` implementation and
//! matching [`GraphSink`](super::sink) for body encoding and reply parsing.
//!
//! The executor and its sink are generic over `M: WireMessage` — a message is
//! whatever a dialect says it is (role + content blocks, tool calls,
//! system-vs-turn split, …). The segment store, prompt materializer, and
//! channels stay dialect-agnostic by carrying each message as its serialized
//! JSON bytes.

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

/// One unit of conversation content in a dialect's request. Content-addressed in
/// the segment store (via its `Serialize` form) and round-tripped through graph
/// channels (via `Serialize`/`Deserialize`).
pub trait WireMessage:
    Clone + Serialize + DeserializeOwned + std::fmt::Debug + PartialEq + 'static
{
    /// Conversation role (`"user"`/`"assistant"`/`"system"`), used for segment
    /// framing and history merging. Role-less dialects may return `""`.
    fn role(&self) -> &str;

    /// Build an assistant message carrying `text` (for splicing a reply into
    /// successor prompts). Structured-only dialects can wrap `text` as they see fit.
    fn assistant(text: String) -> Self;
}

/// The OpenAI chat-completions message: `{ "role", "content" }`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OpenAiChatMessage {
    pub role: String,
    pub content: String,
}

impl OpenAiChatMessage {
    pub fn new(role: impl Into<String>, content: impl Into<String>) -> Self {
        OpenAiChatMessage {
            role: role.into(),
            content: content.into(),
        }
    }
}

impl WireMessage for OpenAiChatMessage {
    fn role(&self) -> &str {
        &self.role
    }
    fn assistant(text: String) -> Self {
        OpenAiChatMessage {
            role: "assistant".to_string(),
            content: text,
        }
    }
}
