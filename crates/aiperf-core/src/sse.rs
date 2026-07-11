// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Client-owned OpenAI chat-completion SSE types + a minimal line decoder.
//!
//! AIPerf owns its wire layer (a stable external spec), so it can benchmark any
//! OpenAI-compatible server without depending on a specific server's internal
//! protocol types. These structs deserialize the streaming chunk shape; unknown
//! fields are ignored by serde.

use serde::Deserialize;

/// One `chat.completion.chunk` streamed over SSE.
#[derive(Debug, Deserialize)]
pub struct ChatChunk {
    /// Per-choice deltas (usually one).
    #[serde(default)]
    pub choices: Vec<ChatChoice>,
    /// Authoritative usage counts, present on the final chunk when
    /// `stream_options.include_usage` is set.
    #[serde(default)]
    pub usage: Option<Usage>,
}

impl ChatChunk {
    /// Concatenated text of this chunk's content (and reasoning) deltas across
    /// all choices. Empty for role-only or finish-only chunks, which are not
    /// counted as output tokens. Reasoning-model output (`reasoning_content`,
    /// e.g. Qwen3/DeepSeek-R1) counts as output the same as regular content.
    pub fn delta_text(&self) -> String {
        let mut out = String::new();
        for choice in &self.choices {
            if let Some(text) = &choice.delta.content {
                out.push_str(text);
            }
            if let Some(reasoning) = &choice.delta.reasoning_content {
                out.push_str(reasoning);
            }
        }
        out
    }
}

/// One choice within a chunk.
#[derive(Debug, Deserialize)]
pub struct ChatChoice {
    /// Incremental content for this choice.
    #[serde(default)]
    pub delta: Delta,
    /// Finish reason, e.g. `"stop"` or `"length"`, on the terminal choice chunk.
    #[serde(default)]
    pub finish_reason: Option<String>,
}

/// The incremental delta of a choice.
#[derive(Debug, Default, Deserialize)]
pub struct Delta {
    /// Content text delta (`None` for role-only / finish-only chunks).
    #[serde(default)]
    pub content: Option<String>,
    /// Reasoning-model output delta (`reasoning_content`, e.g. Qwen3/DeepSeek-R1);
    /// counts as output the same as regular content.
    #[serde(default)]
    pub reasoning_content: Option<String>,
}

/// Server-reported token usage.
#[derive(Debug, Deserialize)]
pub struct Usage {
    /// Authoritative input (prompt) token count.
    #[serde(default)]
    pub prompt_tokens: u32,
    /// Authoritative output (completion) token count.
    #[serde(default)]
    pub completion_tokens: u32,
}

/// One decoded SSE line.
pub enum SseEvent {
    /// A chat-completion chunk payload (boxed; the struct is large).
    Data(Box<ChatChunk>),
    /// The terminal `[DONE]` sentinel.
    Done,
    /// A comment, blank, or otherwise-ignored line.
    Other,
}

/// Parse a single SSE line into an [`SseEvent`].
///
/// Handles `data:` payloads and the `[DONE]` sentinel; comment (`:`), blank,
/// and `event:`/`id:` lines return [`SseEvent::Other`], as does a payload that
/// fails to deserialize.
pub fn parse_sse_line(line: &str) -> SseEvent {
    let line = line.trim_end();
    let Some(rest) = line.strip_prefix("data:") else {
        return SseEvent::Other;
    };
    let payload = rest.trim();
    if payload == "[DONE]" {
        return SseEvent::Done;
    }
    match serde_json::from_str::<ChatChunk>(payload) {
        Ok(chunk) => SseEvent::Data(Box::new(chunk)),
        Err(_) => SseEvent::Other,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_done_sentinel() {
        assert!(matches!(parse_sse_line("data: [DONE]"), SseEvent::Done));
    }

    #[test]
    fn parses_chat_delta() {
        let line = r#"data: {"id":"x","object":"chat.completion.chunk","created":0,"model":"m","choices":[{"index":0,"delta":{"content":"hi"},"finish_reason":null}]}"#;
        match parse_sse_line(line) {
            SseEvent::Data(chunk) => {
                assert_eq!(chunk.choices[0].delta.content.as_deref(), Some("hi"));
            }
            _ => panic!("expected Data"),
        }
    }

    #[test]
    fn ignores_comment_line() {
        assert!(matches!(parse_sse_line(": ping"), SseEvent::Other));
    }
}
