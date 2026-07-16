// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Materialization-only `dynosim` endpoint dialect.
//!
//! Selected by default whenever the run's transport is `dynosim_offline` or
//! `dynosim_online` (see the Python `BenchmarkConfig` transport/endpoint
//! defaulting). Both dynosim transports drive Dynamo's passive in-process engine
//! and open no sockets: the scheduled path materializes requests through
//! [`crate`'s consumer] `TraceHashAwareRequestMaterializer` (an empty HTTP body
//! plus store-derived input-token accounting), and the direct Graph-IR path
//! dispatches authored tokens straight into the simulator. The endpoint
//! therefore never has its [`PreparedEndpoint::format_payload`] dialed on the
//! product path; it exists so the run can name a first-class `dynosim` dialect
//! (reported as `dynosim://offline`) instead of a fake chat-over-HTTP URL.
//!
//! Material composition semantics mirror the chat dialect (text input,
//! token output, no raw-token requirement) so that a trace-hash dataset composes
//! and validates byte-identically to the pre-transport `endpoint.type: chat`
//! offline form. Setting `requires_raw_token_ids` would reject recorded
//! `mooncake_trace` turns (they carry `trace_hash_ids`, not exact `raw_token_ids`)
//! at [`crate`'s] post-compose `Dataset::validate_for_endpoint` pass.

use std::collections::BTreeMap;

use serde_json::{Value, json};

use crate::body_plan::BodyPlan;
use crate::endpoints::config::{EffectiveEndpointConfig, RawEndpointConfig};
use crate::endpoints::metadata::{EndpointDescriptor, Modality};
use crate::endpoints::models::{
    EndpointResult, ExtractedPayload, ParsedResponse, RequestRecord, ServerResponse, Turn,
};
use crate::endpoints::registry::{
    EndpointFactory, PreparedEndpoint, PreparedRequest, ReadinessPolicy,
};

const DYNOSIM_DESCRIPTOR: EndpointDescriptor = EndpointDescriptor {
    id: "dynosim",
    aliases: &[],
    description: "Dynamo passive-engine co-simulation materialization dialect",
    // Offline/online dynosim open no sockets; the request body is materialized
    // as empty bytes and no path is ever dialed.
    endpoint_path: None,
    streaming_path: None,
    supports_streaming: true,
    produces_tokens: true,
    tokenizes_input: true,
    // Recorded traces carry trace_hash_ids, not exact raw token IDs; requiring
    // raw token IDs here would reject the mooncake scheduled path.
    requires_raw_token_ids: false,
    requires_form_data: false,
    requires_polling: false,
    requires_inline_media: false,
    input_modalities: &[Modality::Text],
    output_modalities: &[Modality::Tokens],
    metrics_title: "LLM Metrics",
    service_kind: "llm",
};

/// Protocol-v2-only factory for the in-process Dynamo materialization dialect.
#[derive(Clone, Copy, Debug, Default)]
pub struct DynosimEndpointFactory;

impl EndpointFactory for DynosimEndpointFactory {
    fn descriptor(&self) -> &'static EndpointDescriptor {
        &DYNOSIM_DESCRIPTOR
    }

    fn validate_config(&self, _config: &mut RawEndpointConfig) -> EndpointResult<()> {
        Ok(())
    }

    fn prepare(
        &self,
        config: EffectiveEndpointConfig,
    ) -> EndpointResult<Box<dyn PreparedEndpoint>> {
        let mut headers = config.as_raw().headers.clone();
        if let Some(api_key) = &config.as_raw().api_key {
            headers.insert("Authorization".into(), format!("Bearer {api_key}"));
        }
        Ok(Box::new(PreparedDynosim { config, headers }))
    }
}

#[derive(Debug)]
struct PreparedDynosim {
    config: EffectiveEndpointConfig,
    headers: BTreeMap<String, String>,
}

impl PreparedEndpoint for PreparedDynosim {
    fn descriptor(&self) -> &'static EndpointDescriptor {
        &DYNOSIM_DESCRIPTOR
    }

    fn config(&self) -> &EffectiveEndpointConfig {
        &self.config
    }

    fn format_payload(&self, request: &PreparedRequest<'_>) -> EndpointResult<BodyPlan> {
        // Never dialed on the product path: the in-process engine consumes an
        // empty materialized body. This best-effort shape keeps the dialect
        // preparable and inspectable without pretending to be a real API.
        let model = request
            .turns()
            .first()
            .and_then(|turn| turn.model.clone())
            .unwrap_or_else(|| request.primary_model_name().to_string());
        let payload = json!({ "model": model, "engine": "dynosim://offline" });
        Ok(BodyPlan::from_object(
            payload.as_object().expect("dynosim payload is an object"),
        )?)
    }

    fn headers(&self) -> &BTreeMap<String, String> {
        &self.headers
    }

    fn readiness_policy(&self, _model: &str) -> EndpointResult<ReadinessPolicy> {
        Ok(ReadinessPolicy::Unsupported {
            reason: "dynosim co-simulation opens no server; readiness is not dialed",
        })
    }

    fn parse_response(&self, _response: &ServerResponse) -> EndpointResult<Option<ParsedResponse>> {
        Ok(None)
    }

    fn extract_payload_inputs(&self, _body: &Value) -> ExtractedPayload {
        ExtractedPayload::default()
    }

    fn extract_response_data(
        &self,
        _record: &RequestRecord,
    ) -> EndpointResult<Vec<ParsedResponse>> {
        Ok(Vec::new())
    }

    fn build_assistant_turn(&self, _record: &RequestRecord) -> EndpointResult<Option<Turn>> {
        Ok(None)
    }

    fn captures_assistant_turn(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn descriptor_is_materialization_only_and_not_raw_token_native() {
        let descriptor = DynosimEndpointFactory.descriptor();
        assert_eq!(descriptor.id, "dynosim");
        assert!(descriptor.endpoint_path.is_none());
        assert!(!descriptor.requires_raw_token_ids);
    }
}
