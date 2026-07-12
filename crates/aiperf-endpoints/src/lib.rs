// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Endpoint adapters for request formatting, response parsing, and input extraction.
//!
//! The crate owns decoded JSON body construction and decoded JSON response parsing for
//! provider endpoint dialects, including OpenAI-compatible APIs and Anthropic Messages.
//! Transport concerns such as URL assembly, SSE framing, and cancellation remain outside
//! this crate; dialect-owned authentication headers are exposed through [`Endpoint`].
//! KServe HTTP and inference-protocol dialects are open-registry, protocol-v2-only
//! factories; their native gRPC wire bindings live in `aiperf-transport-grpc`.

mod anthropic;
mod config;
mod endpoints;
mod extraction;
mod kserve;
mod metadata;
mod models;
mod registry;
mod tier2;
mod usage;

pub use anthropic::MessagesEndpoint;
pub use config::{EffectiveEndpointConfig, EndpointConfig, RawEndpointConfig, RequestContentType};
pub use endpoints::{
    ChatEmbeddingsEndpoint, ChatEndpoint, CompletionsEndpoint, EmbeddingsEndpoint, Endpoint,
    ResponsesEndpoint, WARMUP_SYSTEM_MESSAGE_PREFIX,
};
pub use extraction::extract_payload;
pub use kserve::{
    KServeChatFactory, KServeCompletionsFactory, KServeEmbeddingsFactory, KServeV1PredictFactory,
    KServeV2EmbeddingsFactory, KServeV2ImagesFactory, KServeV2InferFactory,
    KServeV2RankingsFactory, KServeV2VlmFactory,
};
pub use metadata::{EndpointDescriptor, EndpointMetadata, EndpointType, Modality, metadata_for};
pub use models::{
    CreditPhase, EndpointError, EndpointResult, ExtractedPayload, ImageDataItem, ImageResponseData,
    Media, ModelEndpoint, ParsedResponse, RequestInfo, RequestRecord, ResponseData, ServerResponse,
    Turn, VideoResponseData,
};
pub use registry::{
    EndpointFactory, EndpointId, EndpointIdError, EndpointKey, EndpointRegistry,
    EndpointRegistryBuilder, EndpointRegistryError, EndpointResolver, PreparedEndpoint,
    PreparedEndpointBehavior, PreparedEndpointTable, PreparedReadinessRequest, PreparedRequest,
    ReadinessMethod, ReadinessPolicy, StatelessEndpointFactory,
};
pub use tier2::{
    CohereRankingsEndpoint, HfTeiRankingsEndpoint, HuggingFaceGenerateEndpoint, ImageEditEndpoint,
    ImageGenerationEndpoint, ImageRetrievalEndpoint, NimEmbeddingsEndpoint, NimRankingsEndpoint,
    RawEndpoint, RawEndpointFactory, SolidoRagEndpoint, TemplateEndpoint, TemplateEndpointFactory,
    VideoGenerationEndpoint,
};
pub use usage::UsageView;
