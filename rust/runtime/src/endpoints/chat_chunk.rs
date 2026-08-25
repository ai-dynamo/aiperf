// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Typed OpenAI chat-completion SSE chunk codec.
//!
//! The transport frames SSE bytes; callers deserialize each `data:` payload
//! into [`ChatChunk`]. Unknown provider fields are ignored.

use serde::Deserialize;

use crate::endpoints::models::ResponseData;

/// One `chat.completion.chunk` streamed over SSE.
#[derive(Debug, Deserialize)]
pub struct ChatChunk {
    /// Provider response identifier repeated on streamed chunks.
    #[serde(default)]
    pub id: Option<String>,
    /// Alternate identifier key used by some providers; read only when `id` is
    /// absent, matching the generic metadata path's fallback order.
    #[serde(default)]
    pub request_id: Option<String>,
    /// Payload discriminator. The generic extractor refuses to interpret a body
    /// whose `object` is neither `chat.completion` nor `chat.completion.chunk`,
    /// so the typed path must see it to make the same decision.
    #[serde(default)]
    pub object: Option<String>,
    /// Per-choice deltas (usually one).
    #[serde(default)]
    pub choices: Vec<ChatChoice>,
    /// Authoritative usage counts, present on the final chunk when
    /// `stream_options.include_usage` is set.
    #[serde(default)]
    pub usage: Option<Usage>,
}

impl ChatChunk {
    /// Remove per-request stats from the sole choice, suppressing multi-choice responses.
    pub fn take_speculative_decoding_stats(&mut self) -> Option<serde_json::Value> {
        if self.choices.len() != 1 {
            return None;
        }
        self.choices.first_mut()?.speculative_decoding_stats.take()
    }

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

    /// Endpoint-normalized response data for a streamed chunk, equivalent to
    /// the generic `serde_json::Value` extractor but without building a `Value`.
    ///
    /// Deliberately narrower than [`Self::delta_text`]: the generic extractor
    /// reads only the FIRST choice, and its precedence is reasoning, then tool
    /// calls, then content. Both are reproduced exactly — a differential test
    /// pins them together, because a divergence here silently changes exported
    /// records rather than failing.
    ///
    /// An absent `object` is accepted as a chunk, matching the compatibility
    /// shim the dispatch path applies for the chat endpoint: a body with a
    /// `choices` envelope and no `object` is treated as a
    /// `chat.completion.chunk` while streaming. Any other `object` value is
    /// declined so non-streaming and unknown bodies keep using the generic
    /// extractor rather than being guessed at.
    pub fn into_stream_response_data(self) -> Option<ResponseData> {
        match self.object.as_deref() {
            Some("chat.completion.chunk") | None => {}
            Some(_) => return None,
        }
        // Consumes the chunk so the delta's strings move into the result. The
        // chunk is dropped immediately after this call on every path, so
        // borrowing it here only bought a clone of the content and reasoning
        // text -- one extra allocation and copy per streamed token.
        let delta = self.choices.into_iter().next()?.delta;
        let content = delta.content;
        if let Some(reasoning) = delta
            .reasoning_content
            .or(delta.reasoning)
            .filter(|value| !value.is_empty())
        {
            return Some(ResponseData::Reasoning { content, reasoning });
        }
        let mut parts: Vec<String> = Vec::new();
        for call in delta.tool_calls {
            let Some(function) = call.function else {
                continue;
            };
            if let Some(name) = function.name.filter(|value| !value.is_empty()) {
                parts.push(name);
            }
            if let Some(arguments) = function.arguments.filter(|value| !value.is_empty()) {
                parts.push(arguments);
            }
        }
        let tool_call_text = parts.concat();
        if !tool_call_text.is_empty() {
            return Some(ResponseData::ToolCall {
                tool_call_text,
                content: content.filter(|value| !value.is_empty()),
            });
        }
        content
            .filter(|value| !value.is_empty())
            .map(|text| ResponseData::Text { text })
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
    /// vLLM per-request acceptance data carried by the finish-reason chunk.
    #[serde(default)]
    pub speculative_decoding_stats: Option<serde_json::Value>,
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
    /// Alternate reasoning key; read only when `reasoning_content` is absent.
    #[serde(default)]
    pub reasoning: Option<String>,
    /// Incremental tool-call deltas.
    #[serde(default)]
    pub tool_calls: Vec<ToolCall>,
}

/// One tool-call entry inside a delta.
#[derive(Debug, Deserialize)]
pub struct ToolCall {
    /// The invoked function; entries without one contribute no text.
    #[serde(default)]
    pub function: Option<ToolFunction>,
}

/// The function payload of a tool call.
#[derive(Debug, Deserialize)]
pub struct ToolFunction {
    /// Function name delta.
    #[serde(default)]
    pub name: Option<String>,
    /// Serialized argument delta.
    #[serde(default)]
    pub arguments: Option<String>,
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

    /// Every shape the two implementations must agree on. A payload added here
    /// is checked against the generic extractor, so a field the typed struct
    /// forgets to model shows up as a test failure rather than as silently
    /// altered export records.
    const DIFFERENTIAL_CORPUS: &[&str] = &[
        // plain content delta
        r#"{"object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"hi"}}]}"#,
        // role-only opener carries no data
        r#"{"object":"chat.completion.chunk","choices":[{"index":0,"delta":{"role":"assistant"}}]}"#,
        // empty content is not a Text response
        r#"{"object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":""}}]}"#,
        // reasoning wins over content
        r#"{"object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"a","reasoning_content":"why"}}]}"#,
        // the `reasoning` spelling is the fallback key
        r#"{"object":"chat.completion.chunk","choices":[{"index":0,"delta":{"reasoning":"why"}}]}"#,
        // empty reasoning falls through to content
        r#"{"object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"a","reasoning_content":""}}]}"#,
        // tool call name + arguments concatenate
        r#"{"object":"chat.completion.chunk","choices":[{"index":0,"delta":{"tool_calls":[{"function":{"name":"f","arguments":"{}"}}]}}]}"#,
        // tool call with content present
        r#"{"object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"c","tool_calls":[{"function":{"name":"f"}}]}}]}"#,
        // tool call entry without a function contributes nothing
        r#"{"object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"c","tool_calls":[{"id":"x"}]}}]}"#,
        // only the FIRST choice is read
        r#"{"object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"first"}},{"index":1,"delta":{"content":"second"}}]}"#,
        // no choices at all
        r#"{"object":"chat.completion.chunk","choices":[]}"#,
        // terminal usage-only chunk
        r#"{"object":"chat.completion.chunk","choices":[],"usage":{"prompt_tokens":7,"completion_tokens":3}}"#,
        // a missing `object` must not be interpreted
        r#"{"choices":[{"index":0,"delta":{"content":"hi"}}]}"#,
    ];

    /// The generic result a streamed chat body actually produces, which is the
    /// extractor PLUS the dispatch shim: for the chat endpoint, a `choices`
    /// envelope with no `object` is read as a chunk while streaming. Comparing
    /// against the bare extractor would assert the wrong contract -- that
    /// mistake shipped once and dropped every delta from servers that omit
    /// `object`.
    fn generic_stream_response_data(payload: &str) -> Option<ResponseData> {
        let value: serde_json::Value =
            serde_json::from_str(payload).expect("corpus payload is valid JSON");
        let mut object = value
            .as_object()
            .expect("corpus payload is an object")
            .clone();
        if !object.contains_key("object") && object.contains_key("choices") {
            object.insert(
                "object".into(),
                serde_json::Value::String("chat.completion.chunk".into()),
            );
        }
        crate::endpoints::implementation::extract_chat_response_data(&object)
    }

    #[test]
    fn typed_response_data_matches_the_generic_value_extractor() {
        for payload in DIFFERENTIAL_CORPUS {
            let typed = parse(payload).into_stream_response_data();
            assert_eq!(
                typed,
                generic_stream_response_data(payload),
                "typed path diverged for payload: {payload}"
            );
        }
    }

    /// The fast path must decline anything it does not model, so those bodies
    /// keep flowing through the generic extractor.
    #[test]
    fn typed_response_data_declines_non_chunk_objects() {
        let non_streaming =
            r#"{"object":"chat.completion","choices":[{"index":0,"message":{"content":"hi"}}]}"#;
        assert_eq!(parse(non_streaming).into_stream_response_data(), None);
    }

    /// The regression the first cut shipped: servers that omit `object` had
    /// every delta silently dropped.
    #[test]
    fn typed_response_data_accepts_a_chunk_without_an_object_field() {
        let no_object = r#"{"id":"response","choices":[{"index":0,"delta":{"content":"hel"}}]}"#;
        assert_eq!(
            parse(no_object).into_stream_response_data(),
            Some(ResponseData::Text {
                text: "hel".to_string()
            })
        );
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

    #[test]
    fn finish_only_chunk_retains_speculative_decoding_stats() {
        let chunk = parse(
            r#"{"object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop","speculative_decoding_stats":{"acceptance_histogram":{"0":1},"num_spec_steps":1}}]}"#,
        );
        assert!(chunk.choices[0].delta.content.is_none());
        assert!(chunk.usage.is_none());
        assert_eq!(
            chunk.choices[0]
                .speculative_decoding_stats
                .as_ref()
                .and_then(|stats| stats["num_spec_steps"].as_u64()),
            Some(1)
        );
    }
}
