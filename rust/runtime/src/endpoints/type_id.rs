// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Open endpoint-kind identity.

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
