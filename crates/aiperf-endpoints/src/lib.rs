// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Endpoint adapters for request formatting, response parsing, and input extraction.
//!
//! The crate owns decoded JSON body construction and decoded JSON response parsing for
//! OpenAI-compatible endpoint dialects. Transport concerns such as URL assembly, headers,
//! SSE framing, and cancellation remain outside this crate.

mod anthropic;
mod config;
mod endpoints;
mod extraction;
mod metadata;
mod models;
mod tier2;
mod usage;

pub use anthropic::MessagesEndpoint;
pub use config::{EndpointConfig, RequestContentType};
pub use endpoints::{
    ChatEmbeddingsEndpoint, ChatEndpoint, CompletionsEndpoint, EmbeddingsEndpoint, Endpoint,
    ResponsesEndpoint, WARMUP_SYSTEM_MESSAGE_PREFIX,
};
pub use extraction::extract_payload;
pub use metadata::{EndpointMetadata, EndpointType, metadata_for};
pub use models::{
    CreditPhase, EndpointError, EndpointResult, ExtractedPayload, ImageDataItem, ImageResponseData,
    Media, ModelEndpoint, ParsedResponse, RequestInfo, RequestRecord, ResponseData, ServerResponse,
    Turn, VideoResponseData,
};
pub use tier2::{
    CohereRankingsEndpoint, HfTeiRankingsEndpoint, HuggingFaceGenerateEndpoint, ImageEditEndpoint,
    ImageGenerationEndpoint, ImageRetrievalEndpoint, NimEmbeddingsEndpoint, NimRankingsEndpoint,
    RawEndpoint, SolidoRagEndpoint, TemplateEndpoint, VideoGenerationEndpoint,
};
pub use usage::UsageView;
