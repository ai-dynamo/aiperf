// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared endpoint request, response, and error models.

use std::fmt::{self, Display};

use bytes::Bytes;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use smallvec::SmallVec;

use crate::endpoints::config::EndpointConfig;

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
    /// Request serialization failed. The rendered error keeps this type
    /// cloneable and comparable.
    Serialization(String),
}

impl Display for EndpointError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidRequest(message) => write!(f, "invalid endpoint request: {message}"),
            Self::InvalidConfig(message) => write!(f, "invalid endpoint config: {message}"),
            Self::InvalidResponse(message) => write!(f, "invalid endpoint response: {message}"),
            Self::Serialization(message) => {
                write!(f, "endpoint body serialization error: {message}")
            }
        }
    }
}

impl std::error::Error for EndpointError {}

impl From<serde_json::Error> for EndpointError {
    fn from(value: serde_json::Error) -> Self {
        Self::Serialization(value.to_string())
    }
}

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
    /// Exact validated input token IDs for a token-native endpoint.
    #[serde(default)]
    pub raw_token_ids: Option<Vec<u32>>,
    /// Preformatted OpenAI-compatible messages spliced verbatim when non-empty.
    pub raw_messages: Option<Vec<Value>>,
    /// Preformatted OpenAI-compatible tools.
    pub raw_tools: Option<Vec<Value>>,
    /// Preformatted vendor-shaped top-level system content blocks.
    pub raw_system: Option<Vec<Value>>,
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
    /// Prebuilt raw request body used by the raw endpoint reconstruction path.
    pub raw_payload: Option<Value>,
    /// Message wires serialized at load and spliced verbatim at dispatch; never
    /// serialized as part of the turn.
    #[serde(skip)]
    pub lowered: Option<SmallVec<[Bytes; 1]>>,
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
    /// Optional request correlation identifier exposed to templates.
    #[serde(default)]
    pub x_request_id: Option<String>,
    /// Optional session correlation identifier exposed to templates.
    #[serde(default)]
    pub x_correlation_id: Option<String>,
    /// Optional authored conversation identifier exposed to templates.
    #[serde(default)]
    pub conversation_id: Option<String>,
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
    /// Exact generated token IDs returned by a token-native endpoint.
    TokenIds { token_ids: Vec<u32> },
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
    /// Image-retrieval result objects.
    ImageRetrieval { data: Vec<Value> },
    /// Generated or edited images and their response metadata.
    Images(ImageResponseData),
    /// Synthesized audio and its output geometry.
    Audio(AudioResponseData),
    /// Async video-job state.
    Video(Box<VideoResponseData>),
}

impl ResponseData {
    /// Return the generated text counted for output-length metrics.
    pub fn get_text(&self) -> String {
        match self {
            Self::Text { text } => text.clone(),
            Self::TokenIds { .. } => String::new(),
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
            Self::Embeddings { .. }
            | Self::Rankings { .. }
            | Self::ImageRetrieval { .. }
            | Self::Images(_)
            | Self::Audio(_)
            | Self::Video(_) => String::new(),
        }
    }

    /// Whether this value carries a generated token output even when it has no text.
    pub fn has_token_output(&self) -> bool {
        match self {
            Self::Text { text } => !text.is_empty(),
            Self::TokenIds { token_ids } => !token_ids.is_empty(),
            Self::Reasoning { content, reasoning } => {
                !reasoning.is_empty() || content.as_ref().is_some_and(|text| !text.is_empty())
            }
            Self::ToolCall {
                tool_call_text,
                content,
            } => {
                !tool_call_text.is_empty() || content.as_ref().is_some_and(|text| !text.is_empty())
            }
            Self::Embeddings { .. }
            | Self::Rankings { .. }
            | Self::ImageRetrieval { .. }
            | Self::Images(_)
            | Self::Audio(_)
            | Self::Video(_) => false,
        }
    }

    /// Return the exact output-token count when the response is token-native.
    pub fn raw_token_count(&self) -> Option<u64> {
        match self {
            Self::TokenIds { token_ids } => u64::try_from(token_ids.len()).ok(),
            _ => None,
        }
    }
}

/// Synthesized-audio response data.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AudioResponseData {
    /// Raw audio bytes returned by the service.
    pub audio_bytes: Vec<u8>,
    /// Output sample rate in hertz.
    pub sample_rate_hz: u32,
    /// Riva audio encoding name.
    pub encoding: String,
    /// Derived duration for mono 16-bit linear PCM, when computable.
    pub duration_ms: Option<f64>,
}

/// One generated-image item returned by image generation or edit endpoints.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct ImageDataItem {
    /// Provider-hosted image URL.
    pub url: Option<String>,
    /// Base64-encoded image body.
    pub b64_json: Option<String>,
    /// Provider-revised prompt.
    pub revised_prompt: Option<String>,
    /// Streaming partial-image index.
    pub partial_image_index: Option<u64>,
}

/// Generated-image response data shared by image generation and edit.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct ImageResponseData {
    /// Generated image items in provider order.
    pub images: Vec<ImageDataItem>,
    /// Provider-reported image dimensions.
    pub size: Option<String>,
    /// Provider-reported quality setting.
    pub quality: Option<String>,
    /// Provider-reported output format.
    pub output_format: Option<String>,
    /// Provider-reported background setting.
    pub background: Option<String>,
}

/// Async video-job state returned by submission and polling responses.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct VideoResponseData {
    /// Unique video job identifier.
    pub video_id: Option<String>,
    /// Provider object discriminator.
    pub object: Option<String>,
    /// Current job status.
    pub status: Option<String>,
    /// Completion percentage or provider progress scalar.
    pub progress: Option<Value>,
    /// Completed video content URL.
    pub url: Option<String>,
    /// Requested video dimensions.
    pub size: Option<String>,
    /// Requested video duration.
    pub seconds: Option<Value>,
    /// Provider quality setting.
    pub quality: Option<String>,
    /// Model used for generation.
    pub model: Option<String>,
    /// Provider creation timestamp.
    pub created_at: Option<Value>,
    /// Provider completion timestamp.
    pub completed_at: Option<Value>,
    /// Provider expiry timestamp.
    pub expires_at: Option<Value>,
    /// Server-reported inference duration in seconds.
    pub inference_time_s: Option<f64>,
    /// Server-reported peak memory in MiB.
    pub peak_memory_mb: Option<f64>,
    /// Provider error object or scalar.
    pub error: Option<Value>,
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
    /// RAG source objects, when supplied by the endpoint.
    #[serde(default)]
    pub sources: Option<Value>,
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
