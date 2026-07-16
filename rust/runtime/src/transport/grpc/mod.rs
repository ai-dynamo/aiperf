// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Clock-injected native gRPC transport for runner protocol v2.
//!
//! Endpoint JSON remains transport-neutral in `aiperf-endpoints`. Open
//! [`GrpcEndpointBinding`] factories lower that canonical JSON to protobuf
//! bytes and decode protobuf responses back to the same JSON shape consumed by
//! endpoint parsers. The transport itself is protobuf-agnostic and operates on
//! raw framed messages over Tonic.

pub mod binding;
pub mod codec;
pub mod models;
pub mod proto;
#[cfg(feature = "grpc")]
pub mod sink;
mod raw_codec;
mod riva_binding;
mod riva_codec;
pub mod riva_proto;
pub mod transport;

pub use binding::{
    GrpcBindingRegistry, GrpcBindingRegistryBuilder, GrpcEndpointBinding,
    GrpcEndpointBindingFactory, GrpcEndpointBindingRegistryError, GrpcStreamChunk,
    KServeV2GrpcBindingFactory,
};
pub use codec::{
    CodecError, decode_model_infer_response, decode_model_ready_response,
    decode_model_stream_infer_response, encode_model_infer_request, encode_model_ready_request,
};
pub use models::{
    ConnectionReuseStrategy, GrpcClientConfig, GrpcErrorDetails, GrpcErrorKind, GrpcRequestConfig,
    GrpcRequestRecord, GrpcResponse, GrpcTraceData,
};
pub use riva_binding::{
    RivaAnalyzeEntitiesGrpcBindingFactory, RivaAnalyzeIntentGrpcBindingFactory,
    RivaAsrGrpcBindingFactory, RivaNaturalQueryGrpcBindingFactory,
    RivaPunctuateTextGrpcBindingFactory, RivaTextClassifyGrpcBindingFactory,
    RivaTokenClassifyGrpcBindingFactory, RivaTransformTextGrpcBindingFactory,
    RivaTtsGrpcBindingFactory,
};
pub use transport::{GrpcTransport, GrpcTransportError, grpc_status_to_http};
#[cfg(feature = "grpc")]
pub use sink::*;
