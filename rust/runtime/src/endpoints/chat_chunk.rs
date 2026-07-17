// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Typed OpenAI chat-completion SSE chunk codec.
//!
//! The transport frames SSE bytes; callers deserialize each `data:` payload
//! into [`ChatChunk`]. Unknown provider fields are ignored.

use serde::Deserialize;

/// One `chat.completion.chunk` streamed over SSE.
#[derive(Debug, Deserialize)]
pub struct ChatChunk {
    /// Provider response identifier repeated on streamed chunks.
    #[serde(default)]
    pub id: Option<String>,
    /// Per-choice deltas (usually one).
    #[serde(default)]
    pub choices: Vec<ChatChoice>,
    /// Authoritative usage counts, present on the final chunk when
    /// `stream_options.include_usage` is set.
    #[serde(default)]
    pub usage: Option<Usage>,
}

impl ChatChunk {
    /// Returns true when any choice carries non-empty user-visible content.
    pub fn has_output_delta(&self) -> bool {
        self.choices.iter().any(|choice| {
            choice
                .delta
                .content
                .as_deref()
                .is_some_and(|content| !content.is_empty())
        })
    }

    /// Returns true when any choice carries non-empty reasoning-only content.
    pub fn has_reasoning_delta(&self) -> bool {
        self.choices.iter().any(|choice| {
            choice
                .delta
                .reasoning_content
                .as_deref()
                .is_some_and(|content| !content.is_empty())
        })
    }

    /// Concatenated text of this chunk's content (and reasoning) deltas across
    /// all choices. Empty for role-only or finish-only chunks, which are not
    /// counted as token-like deltas. Reasoning-model output
    /// (`reasoning_content`, e.g. Qwen3/DeepSeek-R1) is retained in the reply;
    /// callers use [`Self::has_output_delta`] and [`Self::has_reasoning_delta`]
    /// when metrics need the two classes separately.
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
    /// OpenAI-compatible prompt-token detail object.
    #[serde(default)]
    pub prompt_tokens_details: Option<TokenDetails>,
    /// Responses-compatible input-token detail object.
    #[serde(default)]
    pub input_tokens_details: Option<TokenDetails>,
    /// Provider-specific top-level cache-read count.
    #[serde(default)]
    pub cache_read_input_tokens: Option<u32>,
}

impl Usage {
    /// Return a cache-hit count from supported OpenAI-compatible usage shapes.
    pub fn cached_tokens(&self) -> Option<u32> {
        self.prompt_tokens_details
            .as_ref()
            .and_then(|details| details.cached_tokens)
            .or_else(|| {
                self.input_tokens_details
                    .as_ref()
                    .and_then(|details| details.cached_tokens)
            })
            .or(self.cache_read_input_tokens)
    }
}

/// Token-detail fields nested under prompt/input usage.
#[derive(Debug, Deserialize)]
pub struct TokenDetails {
    /// Tokens served from a provider prompt cache.
    #[serde(default)]
    pub cached_tokens: Option<u32>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(payload: &str) -> ChatChunk {
        serde_json::from_str::<ChatChunk>(payload).expect("valid chunk")
    }

    #[test]
    fn delta_text_concatenates_content_and_reasoning() {
        let chunk = parse(
            r#"{"choices":[{"index":0,"delta":{"content":"hi","reasoning_content":" think"}}]}"#,
        );
        assert_eq!(chunk.delta_text(), "hi think");
        assert!(chunk.has_output_delta());
        assert!(chunk.has_reasoning_delta());
    }

    #[test]
    fn role_only_chunk_has_empty_delta_text() {
        let chunk = parse(r#"{"choices":[{"index":0,"delta":{"role":"assistant"}}]}"#);
        assert!(chunk.delta_text().is_empty());
    }

    #[test]
    fn usage_chunk_parses_authoritative_counts() {
        let chunk = parse(
            r#"{"id":"resp-1","choices":[],"usage":{"prompt_tokens":7,"completion_tokens":3,"prompt_tokens_details":{"cached_tokens":5}}}"#,
        );
        assert_eq!(chunk.id.as_deref(), Some("resp-1"));
        let usage = chunk.usage.expect("usage present");
        assert_eq!(usage.prompt_tokens, 7);
        assert_eq!(usage.completion_tokens, 3);
        assert_eq!(usage.cached_tokens(), Some(5));
    }
}
