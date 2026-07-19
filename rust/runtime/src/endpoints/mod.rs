// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Endpoint adapters for request formatting, response parsing, and input extraction.
//!
//! The module owns decoded JSON body construction and decoded JSON response parsing for
//! provider endpoint dialects, including OpenAI-compatible APIs and Anthropic Messages.
//! Transport concerns such as URL assembly, SSE framing, and cancellation remain outside
//! this module; dialect-owned authentication headers are exposed through [`Endpoint`].
//! KServe HTTP/inference-protocol and NVIDIA Riva ASR/TTS/NLP dialects are
//! open-registry, protocol-v2-only factories; their native gRPC wire bindings
//! live in `aiperf_runtime::transport::grpc`.

mod anthropic;
mod chat;
pub mod chat_chunk;
mod config;
mod dynosim;
mod endpoints;
mod extraction;
mod kserve;
mod metadata;
mod models;
mod registry;
mod riva;
mod sagemaker;
mod tier2;
mod usage;
mod vllm_generate;

pub use anthropic::MessagesEndpoint;
pub use chat::chat_request_body;
pub use config::{EffectiveEndpointConfig, EndpointConfig, RawEndpointConfig, RequestContentType};
pub use dynosim::DynosimEndpointFactory;
pub use endpoints::{
    ChatEmbeddingsEndpoint, ChatEndpoint, CompletionsEndpoint, EmbeddingsEndpoint, Endpoint,
    ResponsesEndpoint, ShapeLowerer, TurnMessageLowerer, WARMUP_SYSTEM_MESSAGE_PREFIX,
};
pub use extraction::extract_payload;
pub use kserve::{
    KServeChatFactory, KServeCompletionsFactory, KServeEmbeddingsFactory, KServeV1PredictFactory,
    KServeV2EmbeddingsFactory, KServeV2ImagesFactory, KServeV2InferFactory,
    KServeV2RankingsFactory, KServeV2VlmFactory,
};
pub use metadata::{EndpointDescriptor, EndpointType, Modality};
pub use models::{
    AudioResponseData, CreditPhase, EndpointError, EndpointResult, ExtractedPayload, ImageDataItem,
    ImageResponseData, Media, ModelEndpoint, ParsedResponse, RequestInfo, RequestRecord,
    ResponseData, ServerResponse, Turn, VideoResponseData,
};
pub use registry::{
    EndpointFactory, EndpointId, EndpointIdError, EndpointKey, EndpointRegistry,
    EndpointRegistryBuilder, EndpointRegistryError, EndpointResolver, PreparedEndpoint,
    PreparedEndpointBehavior, PreparedEndpointTable, PreparedReadinessRequest, PreparedRequest,
    ReadinessMethod, ReadinessPolicy, ReadinessSuccess, StatelessEndpointFactory,
};
pub use riva::{
    RivaAnalyzeEntitiesFactory, RivaAnalyzeIntentFactory, RivaAsrFactory, RivaNaturalQueryFactory,
    RivaPunctuateTextFactory, RivaTextClassifyFactory, RivaTokenClassifyFactory,
    RivaTransformTextFactory, RivaTtsFactory,
};
pub use sagemaker::SageMakerFactory;
pub use tier2::{
    CohereRankingsEndpoint, HfTeiRankingsEndpoint, HuggingFaceGenerateEndpoint, ImageEditEndpoint,
    ImageGenerationEndpoint, ImageRetrievalEndpoint, NimEmbeddingsEndpoint, NimRankingsEndpoint,
    RawEndpoint, RawEndpointFactory, SolidoRagEndpoint, TemplateEndpoint, TemplateEndpointFactory,
    VideoGenerationEndpoint,
};
pub use usage::UsageView;
pub use vllm_generate::VllmGenerateFactory;
