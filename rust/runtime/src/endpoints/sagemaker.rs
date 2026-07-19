// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! AWS SageMaker Runtime `InvokeEndpoint` / `InvokeEndpointWithResponseStream`.
//!
//! Runner-protocol-v2-only factories, following the same alias pattern as the
//! KServe OpenAI-compatible routes: request body construction and response/
//! chunk parsing are delegated entirely to [`ChatEndpoint`], since the mock
//! server (and real SageMaker JumpStart/DJL LMI containers hosting an
//! OpenAI-chat-shaped model) accept the identical `messages` wire body.
//! SageMaker has no models-list diagnostic route, so readiness uses a
//! minimal `POST` inference probe against the invoke path itself, matching
//! [`ChatEndpoint::readiness_policy`]'s `"inference"` mode.

use std::collections::BTreeMap;
use std::sync::Arc;

use serde_json::{Value, json};

use crate::body_plan::BodyPlan;
use crate::endpoints::config::{EffectiveEndpointConfig, EndpointConfig};
use crate::endpoints::endpoints::{ChatEndpoint, Endpoint};
use crate::endpoints::metadata::{EndpointDescriptor, EndpointType, Modality};
use crate::endpoints::models::{
    EndpointResult, ExtractedPayload, ParsedResponse, RequestRecord, ServerResponse, Turn,
};
use crate::endpoints::registry::{
    EndpointFactory, PreparedEndpoint, PreparedEndpointBehavior, PreparedReadinessRequest,
    PreparedRequest, ReadinessMethod, ReadinessPolicy, ReadinessSuccess,
};

const SAGEMAKER_INVOKE_DESCRIPTOR: EndpointDescriptor = EndpointDescriptor {
    id: "sagemaker_invoke",
    aliases: &["sagemaker"],
    description: "AWS SageMaker Runtime InvokeEndpoint API",
    endpoint_path: Some("/endpoints/{model_name}/invocations"),
    streaming_path: None,
    supports_streaming: false,
    produces_tokens: true,
    tokenizes_input: true,
    requires_raw_token_ids: false,
    requires_form_data: false,
    requires_polling: false,
    requires_inline_media: false,
    input_modalities: &[Modality::Text],
    output_modalities: &[Modality::Tokens],
    metrics_title: "LLM Metrics",
    service_kind: "sagemaker",
};

const SAGEMAKER_INVOKE_STREAM_DESCRIPTOR: EndpointDescriptor = EndpointDescriptor {
    id: "sagemaker_invoke_stream",
    aliases: &["sagemaker_stream"],
    description: "AWS SageMaker Runtime InvokeEndpointWithResponseStream API",
    endpoint_path: None,
    streaming_path: Some("/endpoints/{model_name}/invocations-response-stream"),
    supports_streaming: true,
    produces_tokens: true,
    tokenizes_input: true,
    requires_raw_token_ids: false,
    requires_form_data: false,
    requires_polling: false,
    requires_inline_media: false,
    input_modalities: &[Modality::Text],
    output_modalities: &[Modality::Tokens],
    metrics_title: "LLM Metrics",
    service_kind: "sagemaker",
};

/// Protocol-v2-only factory for AWS SageMaker Runtime `InvokeEndpoint`.
#[derive(Clone, Copy, Debug, Default)]
pub struct SageMakerInvokeFactory;

/// Protocol-v2-only factory for AWS SageMaker Runtime
/// `InvokeEndpointWithResponseStream`.
#[derive(Clone, Copy, Debug, Default)]
pub struct SageMakerInvokeStreamFactory;

macro_rules! sagemaker_factory {
    ($factory:ty, $descriptor:expr) => {
        impl EndpointFactory for $factory {
            fn descriptor(&self) -> &'static EndpointDescriptor {
                &$descriptor
            }

            fn prepare(
                &self,
                config: EffectiveEndpointConfig,
            ) -> EndpointResult<Box<dyn PreparedEndpoint>> {
                prepare_sagemaker(&$descriptor, config)
            }
        }
    };
}

sagemaker_factory!(SageMakerInvokeFactory, SAGEMAKER_INVOKE_DESCRIPTOR);
sagemaker_factory!(
    SageMakerInvokeStreamFactory,
    SAGEMAKER_INVOKE_STREAM_DESCRIPTOR
);

fn prepare_sagemaker(
    descriptor: &'static EndpointDescriptor,
    config: EffectiveEndpointConfig,
) -> EndpointResult<Box<dyn PreparedEndpoint>> {
    let endpoint = Arc::new(ChatEndpoint);
    let legacy_config = EndpointConfig::from_raw(EndpointType::Chat, config.to_raw());
    let headers = endpoint.format_headers(&legacy_config);
    Ok(Box::new(PreparedSageMakerEndpoint {
        endpoint,
        descriptor,
        config,
        legacy_config,
        headers,
    }))
}

#[derive(Debug)]
struct PreparedSageMakerEndpoint {
    endpoint: Arc<ChatEndpoint>,
    descriptor: &'static EndpointDescriptor,
    config: EffectiveEndpointConfig,
    legacy_config: EndpointConfig,
    headers: BTreeMap<String, String>,
}

impl PreparedEndpoint for PreparedSageMakerEndpoint {
    fn descriptor(&self) -> &'static EndpointDescriptor {
        self.descriptor
    }

    fn config(&self) -> &EffectiveEndpointConfig {
        &self.config
    }

    fn format_payload(&self, request: &PreparedRequest<'_>) -> EndpointResult<BodyPlan> {
        self.endpoint
            .format_prepared_payload(request, self.config.as_raw())
    }

    fn headers(&self) -> &BTreeMap<String, String> {
        &self.headers
    }

    fn readiness_policy(&self, model: &str) -> EndpointResult<ReadinessPolicy> {
        let path = self
            .config
            .as_raw()
            .path
            .clone()
            .unwrap_or_else(|| "/endpoints/{model_name}/invocations".to_string())
            .replace("{model_name}", model);
        Ok(ReadinessPolicy::Request(PreparedReadinessRequest {
            method: ReadinessMethod::Post,
            path,
            headers: self.headers.clone(),
            body: Some(json!({
                "messages": [{"role": "user", "content": "Lo"}],
                "max_tokens": 1,
            })),
            success: ReadinessSuccess::NonServerError,
        }))
    }

    fn parse_response(&self, response: &ServerResponse) -> EndpointResult<Option<ParsedResponse>> {
        self.endpoint
            .parse_response_with_config(response, &self.legacy_config)
    }

    fn extract_payload_inputs(&self, body: &Value) -> ExtractedPayload {
        self.endpoint.extract_payload_inputs(body)
    }

    fn extract_response_data(&self, record: &RequestRecord) -> EndpointResult<Vec<ParsedResponse>> {
        self.endpoint
            .extract_response_data_with_config(record, &self.legacy_config)
    }

    fn build_assistant_turn(&self, record: &RequestRecord) -> EndpointResult<Option<Turn>> {
        self.endpoint.build_assistant_turn(record)
    }

    fn captures_assistant_turn(&self) -> bool {
        self.endpoint.captures_assistant_turn()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::endpoints::config::{EffectiveEndpointConfig, RawEndpointConfig};

    fn default_config() -> EffectiveEndpointConfig {
        EffectiveEndpointConfig::from_validated(RawEndpointConfig::default())
    }

    #[test]
    fn invoke_descriptor_uses_non_streaming_invocations_path() {
        let descriptor = SageMakerInvokeFactory.descriptor();
        assert_eq!(
            descriptor.endpoint_path,
            Some("/endpoints/{model_name}/invocations")
        );
        assert!(!descriptor.supports_streaming);
    }

    #[test]
    fn invoke_stream_descriptor_uses_streaming_path() {
        let descriptor = SageMakerInvokeStreamFactory.descriptor();
        assert_eq!(
            descriptor.streaming_path,
            Some("/endpoints/{model_name}/invocations-response-stream")
        );
        assert!(descriptor.supports_streaming);
    }

    #[test]
    fn readiness_policy_posts_minimal_chat_body_to_model_scoped_path() {
        let prepared = SageMakerInvokeFactory.prepare(default_config()).unwrap();
        let ReadinessPolicy::Request(request) = prepared.readiness_policy("my-model").unwrap()
        else {
            panic!("expected a single readiness request");
        };
        assert_eq!(request.method, ReadinessMethod::Post);
        assert_eq!(request.path, "/endpoints/my-model/invocations");
        assert_eq!(
            request.body.as_ref().and_then(|b| b.get("messages")),
            Some(&json!([{"role": "user", "content": "Lo"}]))
        );
    }

    #[test]
    fn parses_chat_shaped_response_via_delegated_chat_endpoint() {
        let prepared = SageMakerInvokeFactory.prepare(default_config()).unwrap();
        let response = ServerResponse::from_json(
            0,
            json!({
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}}],
                "usage": {"completion_tokens": 1}
            }),
        );
        let parsed = prepared.parse_response(&response).unwrap();
        assert!(parsed.is_some());
    }
}
