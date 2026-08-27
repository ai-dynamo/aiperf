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

/// Protocol-v1 source-compatible name for the open endpoint-kind ID.
pub type EndpointType = crate::endpoints::type_id::EndpointTypeId;

impl EndpointDescriptor {
    /// Closed-enum view for protocol-v1 [`crate::endpoints::EndpointConfig`] and dataset paths.
    pub fn legacy_type(self) -> Option<EndpointType> {
        match self.id {
            // Realtime lowers the same authored turn shape as Responses before
            // translating it into WebSocket conversation events.
            "realtime" => Some(EndpointType::Responses),
            id => EndpointType::from_canonical_id(id),
        }
    }

    /// Whether this endpoint emits `conversation.system` as an on-the-wire
    /// system message (chat, responses, realtime, messages, chat_embeddings —
    /// realtime through its Responses [`legacy_type`](Self::legacy_type)). Dataset
    /// composition uses this to gate hoisting a leading authored `system` turn
    /// into the conversation-level system prompt: only these endpoints would
    /// carry it on the wire, so on the others the system turn is left in place
    /// rather than silently dropped. Mirrors Python
    /// `EndpointMetadata.consumes_system_message`.
    pub fn consumes_system_message(self) -> bool {
        matches!(
            self.id,
            "chat" | "responses" | "messages" | "chat_embeddings"
        )
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

#[cfg(test)]
mod tests {
    use super::EndpointType;

    #[test]
    fn audio_transcription_is_a_registered_legacy_type() {
        assert_eq!(
            EndpointType::from_canonical_id("audio_transcription"),
            Some(EndpointType::AudioTranscription)
        );
    }
}
