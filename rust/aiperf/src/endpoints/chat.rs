// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared OpenAI chat-completions request-body construction, so every live HTTP
//! sink emits the identical streaming wire contract in one place.
//!
//! This is the standalone body builder consumed by the runner-library online
//! path's legacy turn binding; the dialect-driven convergence target is
//! [`ChatEndpoint`](crate::endpoints::ChatEndpoint), which owns the full prepared-endpoint
//! request pipeline.

use serde_json::{Value, json};

/// Build a streaming `/v1/chat/completions` request body. `messages` is a slice
/// of `(role, content)` pairs. `stream` and `stream_options.include_usage` are
/// always set so the server returns authoritative prompt/completion token counts.
pub fn chat_request_body(model: &str, messages: &[(&str, &str)], max_tokens: usize) -> Value {
    json!({
        "model": model,
        "stream": true,
        "stream_options": {"include_usage": true},
        "max_tokens": max_tokens,
        "messages": messages
            .iter()
            .map(|(role, content)| json!({"role": role, "content": content}))
            .collect::<Vec<_>>(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_streaming_body_with_usage() {
        let body = chat_request_body("m", &[("user", "hi")], 8);
        assert_eq!(body["model"], "m");
        assert_eq!(body["stream"], true);
        assert_eq!(body["stream_options"]["include_usage"], true);
        assert_eq!(body["max_tokens"], 8);
        assert_eq!(body["messages"][0]["role"], "user");
        assert_eq!(body["messages"][0]["content"], "hi");
    }
}
