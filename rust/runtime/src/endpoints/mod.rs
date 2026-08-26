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
mod extraction;
mod implementation;
mod kserve;
mod metadata;
mod models;
mod registry;
mod riva;
mod sagemaker;
mod spec_decode;
mod tier2;
mod usage;
mod vllm_generate;

pub use anthropic::MessagesEndpoint;
pub use chat::chat_request_body;
pub use config::{
    EffectiveEndpointConfig, EndpointConfig, RawEndpointConfig, RequestContentType,
    ResetKvCacheConfig, ServerProfilerConfig,
};
pub use dynosim::DynosimEndpointFactory;
pub use extraction::extract_payload;
pub(crate) use implementation::capture_endpoint_policy;
pub use implementation::{
    ChatEmbeddingsEndpoint, ChatEndpoint, CompletionsEndpoint, EmbeddingsEndpoint, Endpoint,
    RealtimeEndpoint, ResponsesEndpoint, ShapeLowerer, TurnMessageLowerer,
    WARMUP_SYSTEM_MESSAGE_PREFIX,
};
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
    WebSocketCapabilities, WebSocketConnectionModel, WebSocketDialect,
};
pub use riva::{
    RivaAnalyzeEntitiesFactory, RivaAnalyzeIntentFactory, RivaAsrFactory, RivaNaturalQueryFactory,
    RivaPunctuateTextFactory, RivaTextClassifyFactory, RivaTokenClassifyFactory,
    RivaTransformTextFactory, RivaTtsFactory,
};
pub use sagemaker::SageMakerFactory;
pub(crate) use spec_decode::{extract_vllm_spec_decode_stats, parse_vllm_spec_decode_stats};
pub use tier2::{
    AudioTranscriptionEndpoint, CohereRankingsEndpoint, HfTeiRankingsEndpoint,
    HuggingFaceGenerateEndpoint, ImageEditEndpoint, ImageGenerationEndpoint,
    ImageRetrievalEndpoint, NimEmbeddingsEndpoint, NimRankingsEndpoint, RawEndpoint,
    RawEndpointFactory, SolidoRagEndpoint, TemplateEndpoint, TemplateEndpointFactory,
    VideoGenerationEndpoint,
};
pub use usage::UsageView;
pub use vllm_generate::VllmGenerateFactory;

#[cfg(test)]
mod spec_decode_tests {
    use super::spec_decode::{extract_vllm_spec_decode_stats, parse_vllm_spec_decode_stats};
    use std::collections::BTreeMap;

    fn worked_payload() -> serde_json::Value {
        serde_json::json!({
            "mean_acceptance_length": 3.25,
            "draft_acceptance_rate": 0.5625,
            "acceptance_histogram": [1, 1, 2, 3, 1],
            "num_accepted_draft_tokens": 18,
            "num_draft_tokens": 32,
            "num_spec_steps": 8,
            "num_spec_tokens": 4,
            "per_step_accepted": [2, 3, 1, 4, 2, 0, 3, 3],
            "per_step_drafted": [4, 4, 4, 4, 4, 4, 4, 4]
        })
    }

    #[test]
    fn vllm_worked_example_normalizes_to_the_canonical_record() {
        let record = parse_vllm_spec_decode_stats(worked_payload(), Some(26))
            .expect("worked payload is canonical");
        assert_eq!(record.engine, "vllm");
        assert_eq!(
            record.acceptance_histogram,
            BTreeMap::from([(0, 1), (1, 1), (2, 2), (3, 3), (4, 1)])
        );
        assert_eq!(record.num_spec_steps, 8);
        assert_eq!(record.num_accepted_draft_tokens, 18);
        assert_eq!(record.num_draft_tokens, 32);
        assert_eq!(record.completion_tokens, Some(26));
    }

    #[test]
    fn inconsistent_vllm_aggregate_is_rejected() {
        let mut payload = worked_payload();
        payload["num_accepted_draft_tokens"] = serde_json::json!(19);
        assert!(parse_vllm_spec_decode_stats(payload, None).is_err());
    }

    #[test]
    fn vllm_draft_acceptance_rate_must_be_a_fraction() {
        for invalid_rate in [-0.01, 1.01] {
            let mut payload = worked_payload();
            payload["draft_acceptance_rate"] = serde_json::json!(invalid_rate);
            assert!(parse_vllm_spec_decode_stats(payload, None).is_err());
        }
    }

    #[test]
    fn response_root_stats_are_independent_of_choice_count() {
        let response = serde_json::json!({
            "choices": [
                {"speculative_decoding_stats": worked_payload()},
                {"speculative_decoding_stats": worked_payload()}
            ],
            "metrics": {"speculative_decoding": worked_payload()}
        });
        assert_eq!(
            extract_vllm_spec_decode_stats(&response),
            Some(&response["metrics"]["speculative_decoding"])
        );
    }

    #[test]
    fn obsolete_per_choice_stats_are_not_accepted() {
        let response = serde_json::json!({
            "choices": [{"speculative_decoding_stats": worked_payload()}]
        });
        assert_eq!(extract_vllm_spec_decode_stats(&response), None);
    }

    #[test]
    fn dense_zero_step_and_fully_rejected_histograms_normalize_sparsely() {
        let mut zero_step = worked_payload();
        zero_step["mean_acceptance_length"] = serde_json::json!(1.0);
        zero_step["draft_acceptance_rate"] = serde_json::json!(0.0);
        zero_step["acceptance_histogram"] = serde_json::json!([0, 0, 0, 0, 0]);
        zero_step["num_spec_steps"] = serde_json::json!(0);
        zero_step["num_accepted_draft_tokens"] = serde_json::json!(0);
        zero_step["num_draft_tokens"] = serde_json::json!(0);
        zero_step["per_step_accepted"] = serde_json::json!([]);
        zero_step["per_step_drafted"] = serde_json::json!([]);
        let zero = parse_vllm_spec_decode_stats(zero_step, None).unwrap();
        assert!(zero.acceptance_histogram.is_empty());

        let mut rejected = worked_payload();
        rejected["mean_acceptance_length"] = serde_json::json!(1.0);
        rejected["draft_acceptance_rate"] = serde_json::json!(0.0);
        rejected["acceptance_histogram"] = serde_json::json!([20, 0, 0, 0, 0]);
        rejected["num_spec_steps"] = serde_json::json!(20);
        rejected["num_accepted_draft_tokens"] = serde_json::json!(0);
        rejected["num_draft_tokens"] = serde_json::json!(80);
        rejected
            .as_object_mut()
            .unwrap()
            .remove("per_step_accepted");
        rejected.as_object_mut().unwrap().remove("per_step_drafted");
        let rejected = parse_vllm_spec_decode_stats(rejected, None).unwrap();
        assert_eq!(rejected.acceptance_histogram, BTreeMap::from([(0, 20)]));
    }

    #[test]
    fn dense_histogram_rejects_obsolete_and_impossible_shapes() {
        let mut obsolete = worked_payload();
        obsolete["acceptance_histogram"] = serde_json::json!({"0": 1, "4": 1});
        assert!(parse_vllm_spec_decode_stats(obsolete, None).is_err());

        for histogram in [
            serde_json::json!([1, 1, 2, 4]),
            serde_json::json!([1, 1, 2, 3, 0, 1]),
            serde_json::json!([1, 1, 2, false, 1]),
        ] {
            let mut payload = worked_payload();
            payload["acceptance_histogram"] = histogram;
            assert!(parse_vllm_spec_decode_stats(payload, None).is_err());
        }
    }

    #[test]
    fn missing_fixed_draft_width_accepts_the_observed_dense_histogram() {
        let mut payload = worked_payload();
        payload.as_object_mut().unwrap().remove("num_spec_tokens");
        let record = parse_vllm_spec_decode_stats(payload, None).unwrap();
        assert_eq!(record.num_spec_tokens, None);
    }
}
