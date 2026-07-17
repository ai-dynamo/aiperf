// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Static endpoint capability metadata.

use serde::{Deserialize, Serialize};

/// Stable input/output modality advertised by an endpoint descriptor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Modality {
    /// Natural-language text.
    Text,
    /// Token-stream output.
    Tokens,
    /// Image input or output.
    Image,
    /// Audio input or output.
    Audio,
    /// Video input or output.
    Video,
    /// Dense vector output.
    Embeddings,
    /// Ranked result output.
    Rankings,
}

/// Declarative endpoint facts published by the runner capability catalog.
///
/// Conditional validation and request/response behavior deliberately remain on
/// [`crate::endpoints::EndpointFactory`] and [`crate::endpoints::PreparedEndpoint`]; this descriptor
/// is not a metadata-driven behavior language.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct EndpointDescriptor {
    /// Canonical open endpoint identifier.
    pub id: &'static str,
    /// Accepted compatibility spellings, never advertised as separate entries.
    pub aliases: &'static [&'static str],
    /// Concise human-readable description.
    pub description: &'static str,
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
    /// Whether every dataset turn must carry validated exact input token IDs.
    ///
    /// This capability is consumed during dataset composition and validation;
    /// request formatters therefore receive typed IDs and never re-parse an
    /// arbitrary JSON array on the dispatch path.
    pub requires_raw_token_ids: bool,
    /// Whether multipart form data is required.
    pub requires_form_data: bool,
    /// Whether submit/poll lifecycle is required.
    pub requires_polling: bool,
    /// Whether media must be inlined before dispatch.
    pub requires_inline_media: bool,
    /// Supported input modalities.
    pub input_modalities: &'static [Modality],
    /// Produced output modalities.
    pub output_modalities: &'static [Modality],
    /// Metrics group title.
    pub metrics_title: &'static str,
    /// Presentation service kind.
    pub service_kind: &'static str,
}

/// Registered endpoint types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EndpointType {
    /// OpenAI Chat Completions.
    Chat,
    /// OpenAI Completions.
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

impl EndpointType {
    /// Return the canonical open identifier used by the frozen registry.
    ///
    /// This mapping exists only for protocol-v1 compatibility. New endpoint
    /// factories are identified solely by their descriptor and do not add an
    /// enum variant here.
    pub const fn canonical_id(self) -> &'static str {
        match self {
            Self::Chat => "chat",
            Self::Completions => "completions",
            Self::Responses => "responses",
            Self::Messages => "messages",
            Self::Embeddings => "embeddings",
            Self::ChatEmbeddings => "chat_embeddings",
            Self::NimEmbeddings => "nim_embeddings",
            Self::CohereRankings => "cohere_rankings",
            Self::HfTeiRankings => "hf_tei_rankings",
            Self::NimRankings => "nim_rankings",
            Self::HuggingfaceGenerate => "huggingface_generate",
            Self::ImageGeneration => "image_generation",
            Self::ImageEdit => "image_edit",
            Self::VideoGeneration => "video_generation",
            Self::ImageRetrieval => "image_retrieval",
            Self::SolidoRag => "solido_rag",
            Self::Raw => "raw",
            Self::Template => "template",
        }
    }

    /// Resolve a closed-enum type from a canonical open id.
    ///
    /// Open-registry-only factories (KServe, Riva, `vllm_generate`, dynosim) have
    /// no [`EndpointType`] variant and return `None`.
    pub fn from_canonical_id(id: &str) -> Option<Self> {
        Some(match id {
            "chat" | "chat_completions" => Self::Chat,
            "completions" => Self::Completions,
            "responses" => Self::Responses,
            "messages" => Self::Messages,
            "embeddings" => Self::Embeddings,
            "chat_embeddings" => Self::ChatEmbeddings,
            "nim_embeddings" => Self::NimEmbeddings,
            "cohere_rankings" => Self::CohereRankings,
            "hf_tei_rankings" => Self::HfTeiRankings,
            "nim_rankings" => Self::NimRankings,
            "huggingface_generate" => Self::HuggingfaceGenerate,
            "image_generation" => Self::ImageGeneration,
            "image_edit" => Self::ImageEdit,
            "video_generation" => Self::VideoGeneration,
            "image_retrieval" => Self::ImageRetrieval,
            "solido_rag" => Self::SolidoRag,
            "raw" => Self::Raw,
            "template" => Self::Template,
            _ => return None,
        })
    }
}

impl EndpointDescriptor {
    /// Closed-enum view for protocol-v1 [`EndpointConfig`] and dataset paths.
    pub fn compatibility_type(self) -> Option<EndpointType> {
        EndpointType::from_canonical_id(self.id)
    }

    /// Whether this descriptor accepts the given input modality.
    pub fn supports_input(self, modality: Modality) -> bool {
        self.input_modalities.contains(&modality)
    }

    /// Whether this descriptor produces the given output modality.
    pub fn supports_output(self, modality: Modality) -> bool {
        self.output_modalities.contains(&modality)
    }
}
