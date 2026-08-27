// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Open endpoint-kind identity.

use serde::{Deserialize, Deserializer, Serialize, Serializer, de};
use std::collections::{HashMap, HashSet};
use std::sync::{LazyLock, RwLock};

/// Dense index of one registered endpoint kind.
///
/// Built-in indices retain the declaration order of the former closed enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct EndpointTypeId(u32);

const BUILTIN_NAMES: [&str; 19] = [
    "audio_transcription",
    "chat",
    "completions",
    "responses",
    "messages",
    "embeddings",
    "chat_embeddings",
    "nim_embeddings",
    "cohere_rankings",
    "hf_tei_rankings",
    "nim_rankings",
    "huggingface_generate",
    "image_generation",
    "image_edit",
    "video_generation",
    "image_retrieval",
    "solido_rag",
    "raw",
    "template",
];

struct EndpointTypeInterner {
    names: Vec<&'static str>,
    ids: HashMap<&'static str, EndpointTypeId>,
}

static INTERNER: LazyLock<RwLock<EndpointTypeInterner>> = LazyLock::new(|| {
    let names = BUILTIN_NAMES.to_vec();
    let ids = names
        .iter()
        .enumerate()
        .map(|(index, name)| (*name, EndpointTypeId(index as u32)))
        .collect();
    RwLock::new(EndpointTypeInterner { names, ids })
});

#[allow(non_upper_case_globals)]
impl EndpointTypeId {
    /// OpenAI audio transcription.
    pub const AudioTranscription: Self = Self(0);
    /// OpenAI Chat Completions.
    pub const Chat: Self = Self(1);
    /// OpenAI Completions.
    pub const Completions: Self = Self(2);
    /// OpenAI Responses API.
    pub const Responses: Self = Self(3);
    /// Anthropic Messages API.
    pub const Messages: Self = Self(4);
    /// OpenAI Embeddings.
    pub const Embeddings: Self = Self(5);
    /// Chat-shaped embeddings.
    pub const ChatEmbeddings: Self = Self(6);
    /// NVIDIA NIM embeddings.
    pub const NimEmbeddings: Self = Self(7);
    /// Cohere rerank endpoint.
    pub const CohereRankings: Self = Self(8);
    /// Hugging Face TEI rerank endpoint.
    pub const HfTeiRankings: Self = Self(9);
    /// NVIDIA NIM rankings.
    pub const NimRankings: Self = Self(10);
    /// Hugging Face text generation.
    pub const HuggingfaceGenerate: Self = Self(11);
    /// Image generation.
    pub const ImageGeneration: Self = Self(12);
    /// Image edit.
    pub const ImageEdit: Self = Self(13);
    /// Video generation.
    pub const VideoGeneration: Self = Self(14);
    /// Image retrieval.
    pub const ImageRetrieval: Self = Self(15);
    /// Solido RAG.
    pub const SolidoRag: Self = Self(16);
    /// Raw passthrough.
    pub const Raw: Self = Self(17);
    /// Template passthrough.
    pub const Template: Self = Self(18);

    /// Return the canonical endpoint-kind spelling.
    pub fn as_str(&self) -> &'static str {
        if let Some(name) = BUILTIN_NAMES.get(self.0 as usize) {
            return name;
        }
        let interner = match INTERNER.read() {
            Ok(interner) => interner,
            Err(poisoned) => poisoned.into_inner(),
        };
        interner
            .names
            .get(self.0 as usize)
            .copied()
            .unwrap_or("invalid_endpoint_type")
    }

    /// Resolve a process-interned endpoint kind by canonical spelling.
    pub fn resolve(name: &str) -> Option<Self> {
        let interner = match INTERNER.read() {
            Ok(interner) => interner,
            Err(poisoned) => poisoned.into_inner(),
        };
        interner.ids.get(name).copied()
    }

    /// Resolve a name only when it belongs to `registry`.
    pub fn resolve_in(registry: &EndpointTypeRegistry, name: &str) -> Option<Self> {
        Self::resolve(name).filter(|id| registry.ids.contains(id))
    }

    /// Return the canonical open identifier used by protocol-v1 callers.
    pub fn canonical_id(self) -> &'static str {
        self.as_str()
    }

    /// Resolve a protocol-v1 endpoint type from its canonical ID or legacy alias.
    pub fn from_canonical_id(id: &str) -> Option<Self> {
        Self::resolve(match id {
            "chat_completions" => "chat",
            other => other,
        })
    }
}

impl Serialize for EndpointTypeId {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for EndpointTypeId {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let name = String::deserialize(deserializer)?;
        Self::from_canonical_id(&name)
            .ok_or_else(|| de::Error::custom(format!("unknown endpoint type {name:?}")))
    }
}

/// Set of endpoint kinds available to one host registry.
#[derive(Debug, Clone)]
pub struct EndpointTypeRegistry {
    ids: HashSet<EndpointTypeId>,
}

impl EndpointTypeRegistry {
    /// Construct a registry containing every built-in endpoint kind.
    pub fn builtin() -> Self {
        Self {
            ids: (0..BUILTIN_NAMES.len())
                .map(|index| EndpointTypeId(index as u32))
                .collect(),
        }
    }

    /// Register one canonical endpoint-kind spelling.
    pub fn register(&mut self, name: &str) -> Result<EndpointTypeId, String> {
        if name.is_empty() {
            return Err("endpoint type name must not be empty".to_string());
        }
        if Self::contains_name(self, name) {
            return Err(format!("endpoint type {name:?} is already registered"));
        }

        let mut interner = match INTERNER.write() {
            Ok(interner) => interner,
            Err(poisoned) => poisoned.into_inner(),
        };
        let id = if let Some(id) = interner.ids.get(name).copied() {
            id
        } else {
            let index = u32::try_from(interner.names.len())
                .map_err(|_| "more than u32::MAX endpoint types were registered".to_string())?;
            let name: &'static str = Box::leak(name.to_owned().into_boxed_str());
            let id = EndpointTypeId(index);
            interner.names.push(name);
            interner.ids.insert(name, id);
            id
        };
        self.ids.insert(id);
        Ok(id)
    }

    fn contains_name(&self, name: &str) -> bool {
        EndpointTypeId::resolve(name).is_some_and(|id| self.ids.contains(&id))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_new_endpoint_type_registers_without_touching_the_enum() {
        let mut registry = EndpointTypeRegistry::builtin();
        let id = registry
            .register("plugin_endpoint")
            .expect("register a new endpoint type");

        assert_eq!(id.as_str(), "plugin_endpoint");
        assert_eq!(
            EndpointTypeId::resolve_in(&registry, "plugin_endpoint"),
            Some(id)
        );
    }

    #[test]
    fn duplicate_endpoint_type_registration_is_rejected() {
        let mut registry = EndpointTypeRegistry::builtin();

        assert!(registry.register("chat").is_err());
    }

    #[test]
    fn built_in_endpoint_types_keep_their_wire_spelling() {
        let names = [
            "audio_transcription",
            "chat",
            "completions",
            "responses",
            "messages",
            "embeddings",
            "chat_embeddings",
            "nim_embeddings",
            "cohere_rankings",
            "hf_tei_rankings",
            "nim_rankings",
            "huggingface_generate",
            "image_generation",
            "image_edit",
            "video_generation",
            "image_retrieval",
            "solido_rag",
            "raw",
            "template",
        ];

        for name in names {
            let id = EndpointTypeId::resolve(name).expect("built-in endpoint type");
            assert_eq!(id.as_str(), name);
            assert_eq!(serde_json::to_string(&id).expect("serialize id"), format!("\"{name}\""));
            assert_eq!(
                serde_json::from_str::<EndpointTypeId>(&format!("\"{name}\""))
                    .expect("deserialize id"),
                id
            );
        }
    }
}
