// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared endpoint request, response, and error models.

use std::fmt::{self, Display};

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::config::EndpointConfig;

/// Result type returned by endpoint operations.
pub type EndpointResult<T> = Result<T, EndpointError>;

/// Endpoint formatting, parsing, and validation errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EndpointError {
    /// A request shape is incompatible with the selected endpoint.
    InvalidRequest(String),
    /// An endpoint configuration value is invalid.
    InvalidConfig(String),
    /// A response shape is hard-invalid for this endpoint.
    InvalidResponse(String),
}

impl Display for EndpointError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidRequest(message) => write!(f, "invalid endpoint request: {message}"),
            Self::InvalidConfig(message) => write!(f, "invalid endpoint config: {message}"),
            Self::InvalidResponse(message) => write!(f, "invalid endpoint response: {message}"),
        }
    }
}

impl std::error::Error for EndpointError {}

/// The selected endpoint and primary model for a request.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelEndpoint {
    /// Primary model name used unless a turn overrides it.
    pub primary_model_name: String,
    /// Endpoint configuration.
    pub endpoint: EndpointConfig,
}

/// Credit phase used by formatters for warmup-only behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CreditPhase {
    /// Warmup phase.
    Warmup,
    /// Profiling phase.
    #[default]
    Profiling,
}

/// Text, image, audio, or video contents carried by a turn.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct Media {
    /// Optional media field name.
    pub name: String,
    /// Batched content strings.
    pub contents: Vec<String>,
}

impl Media {
    /// Construct an unnamed media item.
    pub fn new(contents: impl Into<Vec<String>>) -> Self {
        Self {
            name: String::new(),
            contents: contents.into(),
        }
    }
}

/// A single workload turn.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct Turn {
    /// Optional model override for this turn.
    pub model: Option<String>,
    /// Optional role; synthetic messages default to `user`.
    pub role: Option<String>,
    /// Optional generation cap.
    pub max_tokens: Option<u32>,
    /// Preformatted OpenAI-compatible messages spliced verbatim when non-empty.
    pub raw_messages: Option<Vec<Value>>,
    /// Preformatted OpenAI-compatible tools.
    pub raw_tools: Option<Vec<Value>>,
    /// Text items.
    pub texts: Vec<Media>,
    /// Image items.
    pub images: Vec<Media>,
    /// Audio items.
    pub audios: Vec<Media>,
    /// Video items.
    pub videos: Vec<Media>,
    /// Per-turn extra body fields merged after endpoint extras.
    pub extra_body: Option<Map<String, Value>>,
}

/// Full request context consumed by endpoint formatters.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RequestInfo {
    /// Selected model endpoint.
    pub model_endpoint: ModelEndpoint,
    /// Turns in this dispatch.
    pub turns: Vec<Turn>,
    /// Optional shared system message.
    pub system_message: Option<String>,
    /// Optional per-conversation user context message.
    pub user_context_message: Option<String>,
    /// Credit phase.
    pub credit_phase: CreditPhase,
}

/// Decoded server response with a performance timestamp.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ServerResponse {
    /// Performance timestamp in nanoseconds.
    pub perf_ns: u64,
    /// Decoded JSON body, if JSON parsing succeeded.
    pub json: Option<Value>,
    /// Raw text body, if retained.
    pub raw: Option<String>,
}

impl ServerResponse {
    /// Construct a JSON response fixture.
    pub fn from_json(perf_ns: u64, value: Value) -> Self {
        Self {
            perf_ns,
            json: Some(value),
            raw: None,
        }
    }
}

/// All responses observed for one request.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct RequestRecord {
    /// Response chunks or full responses in arrival order.
    pub responses: Vec<ServerResponse>,
}

/// Parsed endpoint response data.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponseData {
    /// Text response.
    Text { text: String },
    /// Reasoning response, optionally with normal content.
    Reasoning {
        content: Option<String>,
        reasoning: String,
    },
    /// Tool-call response, optionally with normal content.
    ToolCall {
        tool_call_text: String,
        content: Option<String>,
    },
    /// Embedding vectors.
    Embeddings { embeddings: Vec<Vec<f64>> },
    /// Ranking result objects.
    Rankings { rankings: Vec<Value> },
}

impl ResponseData {
    /// Return the generated text counted for output-length metrics.
    pub fn get_text(&self) -> String {
        match self {
            Self::Text { text } => text.clone(),
            Self::Reasoning { content, reasoning } => {
                let mut out = reasoning.clone();
                if let Some(content) = content {
                    out.push_str(content);
                }
                out
            }
            Self::ToolCall {
                tool_call_text,
                content,
            } => {
                let mut out = content.clone().unwrap_or_default();
                out.push_str(tool_call_text);
                out
            }
            Self::Embeddings { .. } | Self::Rankings { .. } => String::new(),
        }
    }
}

/// Parsed response plus optional provider usage.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ParsedResponse {
    /// Performance timestamp in nanoseconds.
    pub perf_ns: u64,
    /// Parsed response data, absent for usage-only streaming frames.
    pub data: Option<ResponseData>,
    /// Provider usage object.
    pub usage: Option<Value>,
}

/// Payload input extraction result for input token and media accounting.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct ExtractedPayload {
    /// Bare text strings to tokenize.
    pub texts: Vec<String>,
    /// Tool-derived strings counted separately by chat-template tokenization.
    pub tool_texts: Vec<String>,
    /// Number of image content parts.
    pub image_count: u32,
    /// Number of audio content parts.
    pub audio_count: u32,
    /// Number of video content parts.
    pub video_count: u32,
    /// Token count contributed by pre-tokenized token-id lists.
    pub pretokenised_token_count: u64,
    /// Role/content view for tokenizer chat templates.
    pub messages: Option<Vec<Value>>,
}
