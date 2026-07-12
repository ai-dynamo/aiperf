// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Static endpoint capability metadata ported from `src/aiperf/plugin/plugins.yaml`.

use serde::{Deserialize, Serialize};

/// Registered endpoint types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EndpointType {
    /// OpenAI Chat Completions.
    Chat,
    /// Legacy OpenAI Completions.
    Completions,
    /// OpenAI Responses API.
    Responses,
    /// Anthropic Messages API.
    Messages,
    /// OpenAI Embeddings.
    Embeddings,
    /// Chat-shaped embeddings.
    ChatEmbeddings,
    /// NVIDIA NIM embeddings.
    NimEmbeddings,
    /// Cohere rerank endpoint.
    CohereRankings,
    /// Hugging Face TEI rerank endpoint.
    HfTeiRankings,
    /// NVIDIA NIM rankings.
    NimRankings,
    /// Hugging Face text generation.
    HuggingfaceGenerate,
    /// Image generation.
    ImageGeneration,
    /// Image edit.
    ImageEdit,
    /// Video generation.
    VideoGeneration,
    /// Image retrieval.
    ImageRetrieval,
    /// Solido RAG.
    SolidoRag,
    /// Raw passthrough.
    Raw,
    /// Template passthrough.
    Template,
}

/// Static endpoint capability metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EndpointMetadata {
    /// Endpoint type.
    pub endpoint_type: EndpointType,
    /// Default endpoint path, if any.
    pub endpoint_path: Option<&'static str>,
    /// Streaming endpoint path override, if any.
    pub streaming_path: Option<&'static str>,
    /// Whether streaming is supported.
    pub supports_streaming: bool,
    /// Whether output tokens are produced.
    pub produces_tokens: bool,
    /// Whether input tokenization is enabled.
    pub tokenizes_input: bool,
    /// Whether multipart form data is required.
    pub requires_form_data: bool,
    /// Whether submit/poll lifecycle is required.
    pub requires_polling: bool,
    /// Whether media must be inlined before dispatch.
    pub requires_inline_media: bool,
    /// Whether audio input is supported.
    pub supports_audio: bool,
    /// Whether image input is supported.
    pub supports_images: bool,
    /// Whether video input is supported.
    pub supports_videos: bool,
    /// Whether image output is produced.
    pub produces_images: bool,
    /// Whether video output is produced.
    pub produces_videos: bool,
    /// Metrics group title.
    pub metrics_title: &'static str,
    /// Presentation service kind.
    pub service_kind: &'static str,
}

macro_rules! m {
    ($ty:ident, $path:expr, $stream_path:expr, $stream:expr, $out_tok:expr, $in_tok:expr, $form:expr, $poll:expr, $inline:expr, $audio:expr, $image:expr, $video:expr, $out_img:expr, $out_vid:expr, $title:expr, $kind:expr) => {
        EndpointMetadata {
            endpoint_type: EndpointType::$ty,
            endpoint_path: $path,
            streaming_path: $stream_path,
            supports_streaming: $stream,
            produces_tokens: $out_tok,
            tokenizes_input: $in_tok,
            requires_form_data: $form,
            requires_polling: $poll,
            requires_inline_media: $inline,
            supports_audio: $audio,
            supports_images: $image,
            supports_videos: $video,
            produces_images: $out_img,
            produces_videos: $out_vid,
            metrics_title: $title,
            service_kind: $kind,
        }
    };
}

const METADATA: [EndpointMetadata; 18] = [
    m!(
        Chat,
        Some("/v1/chat/completions"),
        None,
        true,
        true,
        true,
        false,
        false,
        false,
        true,
        true,
        true,
        false,
        false,
        "LLM Metrics",
        "llm"
    ),
    m!(
        Completions,
        Some("/v1/completions"),
        None,
        true,
        true,
        true,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        "LLM Metrics",
        "llm"
    ),
    m!(
        Responses,
        Some("/v1/responses"),
        None,
        true,
        true,
        true,
        false,
        false,
        false,
        true,
        true,
        false,
        false,
        false,
        "LLM Metrics",
        "llm"
    ),
    m!(
        Messages,
        Some("/v1/messages"),
        None,
        true,
        true,
        true,
        false,
        false,
        false,
        false,
        true,
        false,
        false,
        false,
        "LLM Metrics",
        "llm"
    ),
    m!(
        Embeddings,
        Some("/v1/embeddings"),
        None,
        false,
        false,
        true,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        "Embeddings Metrics",
        "embeddings"
    ),
    m!(
        ChatEmbeddings,
        Some("/v1/embeddings"),
        None,
        false,
        false,
        true,
        false,
        false,
        false,
        false,
        true,
        false,
        false,
        false,
        "Embeddings Metrics",
        "embeddings"
    ),
    m!(
        NimEmbeddings,
        Some("/v1/embeddings"),
        None,
        false,
        false,
        true,
        false,
        false,
        false,
        false,
        true,
        false,
        false,
        false,
        "NIM Embeddings Metrics",
        "embeddings"
    ),
    m!(
        CohereRankings,
        Some("/v2/rerank"),
        None,
        false,
        false,
        true,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        "Ranking Metrics",
        "rankings"
    ),
    m!(
        HfTeiRankings,
        Some("/rerank"),
        None,
        false,
        false,
        true,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        "Ranking Metrics",
        "rankings"
    ),
    m!(
        NimRankings,
        Some("/v1/ranking"),
        None,
        false,
        false,
        true,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        "Rankings Metrics",
        "rankings"
    ),
    m!(
        HuggingfaceGenerate,
        Some("/generate"),
        Some("/generate_stream"),
        true,
        true,
        true,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        "LLM Metrics",
        "llm"
    ),
    m!(
        ImageGeneration,
        Some("/v1/images/generations"),
        None,
        true,
        false,
        true,
        false,
        false,
        false,
        false,
        false,
        false,
        true,
        false,
        "Image Generation Metrics",
        "image_generation"
    ),
    m!(
        ImageEdit,
        Some("/v1/images/edits"),
        None,
        false,
        false,
        true,
        true,
        false,
        false,
        false,
        true,
        false,
        true,
        false,
        "Image Edit Metrics",
        "image_edit"
    ),
    m!(
        VideoGeneration,
        Some("/v1/videos"),
        None,
        false,
        false,
        true,
        true,
        true,
        false,
        false,
        false,
        false,
        false,
        true,
        "Video Generation Metrics",
        "video_generation"
    ),
    m!(
        ImageRetrieval,
        Some("/v1/infer"),
        None,
        false,
        false,
        false,
        false,
        false,
        true,
        false,
        true,
        false,
        false,
        false,
        "Image Retrieval Metrics",
        "image_retrieval"
    ),
    m!(
        SolidoRag,
        Some("/rag/api/prompt"),
        None,
        true,
        true,
        true,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        "SOLIDO RAG Metrics",
        "llm"
    ),
    m!(
        Raw,
        None,
        None,
        true,
        true,
        true,
        false,
        false,
        false,
        true,
        true,
        true,
        false,
        false,
        "LLM Metrics",
        "llm"
    ),
    m!(
        Template,
        None,
        None,
        true,
        true,
        true,
        false,
        false,
        false,
        true,
        true,
        true,
        false,
        false,
        "LLM Metrics",
        "llm"
    ),
];

/// Return static metadata for an endpoint type.
pub fn metadata_for(endpoint_type: EndpointType) -> &'static EndpointMetadata {
    METADATA
        .iter()
        .find(|metadata| metadata.endpoint_type == endpoint_type)
        .expect("metadata table covers all EndpointType variants")
}
