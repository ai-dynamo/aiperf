// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Open endpoint-to-gRPC binding registry.

use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;
use std::sync::Arc;

use bytes::Bytes;
use http::uri::PathAndQuery;
use serde_json::Value;

use crate::endpoints::EndpointId;

use crate::transport::grpc::codec::{
    CodecError, decode_model_infer_response, decode_model_ready_response,
    decode_model_stream_infer_response, encode_model_infer_request, encode_model_ready_request,
};

const MODEL_INFER: &str = "/inference.GRPCInferenceService/ModelInfer";
const MODEL_STREAM_INFER: &str = "/inference.GRPCInferenceService/ModelStreamInfer";
const MODEL_READY: &str = "/inference.GRPCInferenceService/ModelReady";

/// One decoded server- or bidirectional-streaming response message.
#[derive(Clone, Debug, PartialEq)]
pub struct GrpcStreamChunk {
    /// In-band server error.
    pub error_message: Option<String>,
    /// Canonical endpoint response body.
    pub response: Option<Value>,
    /// Unframed protobuf message size.
    pub response_size: usize,
}

/// Worker-local binding between one endpoint dialect and gRPC wire protocol.
pub trait GrpcEndpointBinding: fmt::Debug {
    /// Canonical endpoint ID supported by this binding.
    fn endpoint_id(&self) -> &EndpointId;
    /// Unary inference method path.
    fn unary_method(&self) -> &'static PathAndQuery;
    /// Streaming method path, if supported by this endpoint.
    fn streaming_method(&self) -> Option<&'static PathAndQuery> {
        None
    }
    /// Bidirectional streaming method path, if supported by this endpoint.
    fn bidi_streaming_method(&self) -> Option<&'static PathAndQuery> {
        None
    }
    /// Readiness method path, if this protocol defines one.
    fn readiness_method(&self) -> Option<&'static PathAndQuery> {
        None
    }
    /// Encode one canonical endpoint payload.
    fn encode_request(
        &self,
        payload: &Value,
        model_name: &str,
        request_id: &str,
    ) -> Result<Bytes, CodecError>;
    /// Encode the ordered config-first messages for a bidirectional request.
    fn encode_bidi_requests(
        &self,
        _payload: &Value,
        _model_name: &str,
        _request_id: &str,
    ) -> Result<Vec<Bytes>, CodecError> {
        Err(CodecError::new(format!(
            "endpoint {} does not support bidirectional gRPC requests",
            self.endpoint_id()
        )))
    }
    /// Decode one unary response.
    fn decode_response(&self, bytes: &[u8]) -> Result<Value, CodecError>;
    /// Decode one streaming response.
    fn decode_stream_response(&self, bytes: &[u8]) -> Result<GrpcStreamChunk, CodecError>;
    /// Encode a model-readiness request.
    fn encode_readiness_request(&self, _model_name: &str) -> Bytes {
        Bytes::new()
    }
    /// Decode a model-readiness response.
    fn decode_readiness_response(&self, _bytes: &[u8]) -> Result<bool, CodecError> {
        Err(CodecError::new(format!(
            "endpoint {} does not define a gRPC readiness response",
            self.endpoint_id()
        )))
    }
}

/// Startup factory for worker-local gRPC endpoint bindings.
pub trait GrpcEndpointBindingFactory: fmt::Debug + Send + Sync {
    /// Canonical endpoint IDs registered by this factory.
    fn endpoint_ids(&self) -> &'static [&'static str];
    /// Prepare one worker-local binding.
    fn prepare(&self, endpoint_id: &EndpointId)
    -> Result<Box<dyn GrpcEndpointBinding>, CodecError>;
}

/// Registry construction or lookup failure.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum GrpcEndpointBindingRegistryError {
    /// Two factories claim the same endpoint ID.
    Duplicate(EndpointId),
    /// No compiled gRPC binding supports the endpoint.
    Unsupported {
        /// Requested endpoint.
        requested: EndpointId,
        /// Deterministically ordered supported endpoints.
        available: Vec<EndpointId>,
    },
    /// Binding preparation failed.
    Preparation(String),
}

impl fmt::Display for GrpcEndpointBindingRegistryError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Duplicate(id) => write!(formatter, "duplicate gRPC binding for endpoint {id}"),
            Self::Unsupported {
                requested,
                available,
            } => write!(
                formatter,
                "endpoint {requested} has no compiled gRPC binding; supported endpoints: {}",
                available
                    .iter()
                    .map(EndpointId::as_str)
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            Self::Preparation(message) => formatter.write_str(message),
        }
    }
}

impl Error for GrpcEndpointBindingRegistryError {}

/// Mutable startup-only gRPC binding registry.
#[derive(Clone, Default)]
pub struct GrpcBindingRegistryBuilder {
    entries: BTreeMap<EndpointId, Arc<dyn GrpcEndpointBindingFactory>>,
}

impl fmt::Debug for GrpcBindingRegistryBuilder {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("GrpcBindingRegistryBuilder")
            .field("endpoint_ids", &self.entries.keys().collect::<Vec<_>>())
            .finish()
    }
}

impl GrpcBindingRegistryBuilder {
    /// Construct an empty builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Construct the stock KServe v2 and NVIDIA Riva catalog.
    pub fn with_builtins() -> Result<Self, GrpcEndpointBindingRegistryError> {
        let mut builder = Self::new();
        builder.register(KServeV2GrpcBindingFactory)?;
        crate::transport::grpc::riva_binding::register_builtins(&mut builder)?;
        Ok(builder)
    }

    /// Register one factory transactionally for all its endpoint IDs.
    pub fn register<F>(&mut self, factory: F) -> Result<(), GrpcEndpointBindingRegistryError>
    where
        F: GrpcEndpointBindingFactory + 'static,
    {
        let factory: Arc<dyn GrpcEndpointBindingFactory> = Arc::new(factory);
        let ids = factory
            .endpoint_ids()
            .iter()
            .map(|id| {
                EndpointId::new(id).map_err(|error| {
                    GrpcEndpointBindingRegistryError::Preparation(error.to_string())
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut claimed = BTreeSet::new();
        for id in &ids {
            if !claimed.insert(id.clone()) || self.entries.contains_key(id) {
                return Err(GrpcEndpointBindingRegistryError::Duplicate(id.clone()));
            }
        }
        for id in ids {
            self.entries.insert(id, factory.clone());
        }
        Ok(())
    }

    /// Freeze the deterministic registry.
    pub fn freeze(self) -> GrpcBindingRegistry {
        GrpcBindingRegistry {
            entries: Arc::new(self.entries),
        }
    }
}

/// Immutable gRPC endpoint-binding catalog.
#[derive(Clone)]
pub struct GrpcBindingRegistry {
    entries: Arc<BTreeMap<EndpointId, Arc<dyn GrpcEndpointBindingFactory>>>,
}

impl fmt::Debug for GrpcBindingRegistry {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("GrpcBindingRegistry")
            .field("endpoint_ids", &self.entries.keys().collect::<Vec<_>>())
            .finish()
    }
}

impl GrpcBindingRegistry {
    /// Construct the built-in catalog.
    pub fn builtin() -> Result<Self, GrpcEndpointBindingRegistryError> {
        Ok(GrpcBindingRegistryBuilder::with_builtins()?.freeze())
    }

    /// Iterate supported endpoint IDs.
    pub fn endpoint_ids(&self) -> impl ExactSizeIterator<Item = &EndpointId> {
        self.entries.keys()
    }

    /// Prepare a worker-local binding for one endpoint.
    pub fn prepare(
        &self,
        endpoint_id: &EndpointId,
    ) -> Result<Box<dyn GrpcEndpointBinding>, GrpcEndpointBindingRegistryError> {
        let factory = self.entries.get(endpoint_id).ok_or_else(|| {
            GrpcEndpointBindingRegistryError::Unsupported {
                requested: endpoint_id.clone(),
                available: self.entries.keys().cloned().collect(),
            }
        })?;
        factory
            .prepare(endpoint_id)
            .map_err(|error| GrpcEndpointBindingRegistryError::Preparation(error.to_string()))
    }
}

/// Factory for all KServe v2 OIP bindings.
#[derive(Clone, Copy, Debug, Default)]
pub struct KServeV2GrpcBindingFactory;

impl GrpcEndpointBindingFactory for KServeV2GrpcBindingFactory {
    fn endpoint_ids(&self) -> &'static [&'static str] {
        &[
            "kserve_v2_embeddings",
            "kserve_v2_images",
            "kserve_v2_infer",
            "kserve_v2_rankings",
            "kserve_v2_vlm",
        ]
    }

    fn prepare(
        &self,
        endpoint_id: &EndpointId,
    ) -> Result<Box<dyn GrpcEndpointBinding>, CodecError> {
        let streaming = matches!(endpoint_id.as_str(), "kserve_v2_infer" | "kserve_v2_vlm");
        Ok(Box::new(KServeV2GrpcBinding {
            endpoint_id: endpoint_id.clone(),
            streaming,
        }))
    }
}

#[derive(Clone, Debug)]
struct KServeV2GrpcBinding {
    endpoint_id: EndpointId,
    streaming: bool,
}

impl GrpcEndpointBinding for KServeV2GrpcBinding {
    fn endpoint_id(&self) -> &EndpointId {
        &self.endpoint_id
    }

    fn unary_method(&self) -> &'static PathAndQuery {
        static PATH: PathAndQuery = PathAndQuery::from_static(MODEL_INFER);
        &PATH
    }

    fn streaming_method(&self) -> Option<&'static PathAndQuery> {
        static PATH: PathAndQuery = PathAndQuery::from_static(MODEL_STREAM_INFER);
        self.streaming.then_some(&PATH)
    }

    fn readiness_method(&self) -> Option<&'static PathAndQuery> {
        static PATH: PathAndQuery = PathAndQuery::from_static(MODEL_READY);
        Some(&PATH)
    }

    fn encode_request(
        &self,
        payload: &Value,
        model_name: &str,
        request_id: &str,
    ) -> Result<Bytes, CodecError> {
        encode_model_infer_request(payload, model_name, "", request_id)
    }

    fn decode_response(&self, bytes: &[u8]) -> Result<Value, CodecError> {
        decode_model_infer_response(bytes)
    }

    fn decode_stream_response(&self, bytes: &[u8]) -> Result<GrpcStreamChunk, CodecError> {
        let response_size = bytes.len();
        let (error_message, response) = decode_model_stream_infer_response(bytes)?;
        Ok(GrpcStreamChunk {
            error_message,
            response,
            response_size,
        })
    }

    fn encode_readiness_request(&self, model_name: &str) -> Bytes {
        encode_model_ready_request(model_name, "")
    }

    fn decode_readiness_response(&self, bytes: &[u8]) -> Result<bool, CodecError> {
        decode_model_ready_response(bytes)
    }
}
