// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Checked-in Prost representation of the vendored KServe OIP v2 schema.
//!
//! The authoritative schema is
//! `proto/grpc_predict_v2.proto`, vendored from KServe's
//! `open_inference_grpc/grpc_predict_v2.proto`. Keeping generated-equivalent
//! Rust checked in avoids a build-time `protoc` dependency.

use std::collections::BTreeMap;

use prost::Message;

/// Model readiness request.
#[derive(Clone, PartialEq, Message)]
pub struct ModelReadyRequest {
    /// Model name.
    #[prost(string, tag = "1")]
    pub name: String,
    /// Optional model version.
    #[prost(string, tag = "2")]
    pub version: String,
}

/// Model readiness response.
#[derive(Clone, Copy, PartialEq, Message)]
pub struct ModelReadyResponse {
    /// Whether the model is ready.
    #[prost(bool, tag = "1")]
    pub ready: bool,
}

/// Scalar inference parameter.
#[derive(Clone, PartialEq, Message)]
pub struct InferParameter {
    /// Selected scalar value.
    #[prost(oneof = "infer_parameter::ParameterChoice", tags = "1, 2, 3, 4")]
    pub parameter_choice: Option<infer_parameter::ParameterChoice>,
}

/// Nested types for [`InferParameter`].
pub mod infer_parameter {
    use prost::Oneof;

    /// One supported scalar parameter representation.
    #[derive(Clone, PartialEq, Oneof)]
    pub enum ParameterChoice {
        /// Boolean value.
        #[prost(bool, tag = "1")]
        BoolParam(bool),
        /// Signed integer value.
        #[prost(int64, tag = "2")]
        Int64Param(i64),
        /// String value.
        #[prost(string, tag = "3")]
        StringParam(String),
        /// Double-precision value.
        #[prost(double, tag = "4")]
        DoubleParam(f64),
    }
}

/// Typed flattened tensor contents.
#[derive(Clone, PartialEq, Message)]
pub struct InferTensorContents {
    /// BOOL values.
    #[prost(bool, repeated, tag = "1")]
    pub bool_contents: Vec<bool>,
    /// INT8/INT16/INT32 values.
    #[prost(int32, repeated, tag = "2")]
    pub int_contents: Vec<i32>,
    /// INT64 values.
    #[prost(int64, repeated, tag = "3")]
    pub int64_contents: Vec<i64>,
    /// UINT8/UINT16/UINT32 values.
    #[prost(uint32, repeated, tag = "4")]
    pub uint_contents: Vec<u32>,
    /// UINT64 values.
    #[prost(uint64, repeated, tag = "5")]
    pub uint64_contents: Vec<u64>,
    /// FP16/FP32 values (the typed OIP field is FP32).
    #[prost(float, repeated, tag = "6")]
    pub fp32_contents: Vec<f32>,
    /// FP64 values.
    #[prost(double, repeated, tag = "7")]
    pub fp64_contents: Vec<f64>,
    /// BYTES values.
    #[prost(bytes = "vec", repeated, tag = "8")]
    pub bytes_contents: Vec<Vec<u8>>,
}

/// OIP v2 model inference request.
#[derive(Clone, PartialEq, Message)]
pub struct ModelInferRequest {
    /// Model name.
    #[prost(string, tag = "1")]
    pub model_name: String,
    /// Optional model version.
    #[prost(string, tag = "2")]
    pub model_version: String,
    /// Optional request ID.
    #[prost(string, tag = "3")]
    pub id: String,
    /// Request parameters.
    #[prost(btree_map = "string, message", tag = "4")]
    pub parameters: BTreeMap<String, InferParameter>,
    /// Input tensors.
    #[prost(message, repeated, tag = "5")]
    pub inputs: Vec<model_infer_request::InferInputTensor>,
    /// Requested output tensors.
    #[prost(message, repeated, tag = "6")]
    pub outputs: Vec<model_infer_request::InferRequestedOutputTensor>,
    /// Raw input contents, parallel to `inputs`.
    #[prost(bytes = "vec", repeated, tag = "7")]
    pub raw_input_contents: Vec<Vec<u8>>,
}

/// Nested request tensor messages.
pub mod model_infer_request {
    use std::collections::BTreeMap;

    use prost::Message;

    use super::{InferParameter, InferTensorContents};

    /// Input tensor.
    #[derive(Clone, PartialEq, Message)]
    pub struct InferInputTensor {
        /// Tensor name.
        #[prost(string, tag = "1")]
        pub name: String,
        /// OIP datatype name.
        #[prost(string, tag = "2")]
        pub datatype: String,
        /// Tensor shape.
        #[prost(int64, repeated, tag = "3")]
        pub shape: Vec<i64>,
        /// Tensor parameters.
        #[prost(btree_map = "string, message", tag = "4")]
        pub parameters: BTreeMap<String, InferParameter>,
        /// Typed contents.
        #[prost(message, optional, tag = "5")]
        pub contents: Option<InferTensorContents>,
    }

    /// Requested output tensor.
    #[derive(Clone, PartialEq, Message)]
    pub struct InferRequestedOutputTensor {
        /// Tensor name.
        #[prost(string, tag = "1")]
        pub name: String,
        /// Output parameters.
        #[prost(btree_map = "string, message", tag = "2")]
        pub parameters: BTreeMap<String, InferParameter>,
    }
}

/// OIP v2 model inference response.
#[derive(Clone, PartialEq, Message)]
pub struct ModelInferResponse {
    /// Model name.
    #[prost(string, tag = "1")]
    pub model_name: String,
    /// Model version.
    #[prost(string, tag = "2")]
    pub model_version: String,
    /// Correlated request ID.
    #[prost(string, tag = "3")]
    pub id: String,
    /// Response parameters.
    #[prost(btree_map = "string, message", tag = "4")]
    pub parameters: BTreeMap<String, InferParameter>,
    /// Output tensors.
    #[prost(message, repeated, tag = "5")]
    pub outputs: Vec<model_infer_response::InferOutputTensor>,
    /// Raw output contents, parallel to `outputs`.
    #[prost(bytes = "vec", repeated, tag = "6")]
    pub raw_output_contents: Vec<Vec<u8>>,
}

/// Nested response tensor messages.
pub mod model_infer_response {
    use std::collections::BTreeMap;

    use prost::Message;

    use super::{InferParameter, InferTensorContents};

    /// Output tensor.
    #[derive(Clone, PartialEq, Message)]
    pub struct InferOutputTensor {
        /// Tensor name.
        #[prost(string, tag = "1")]
        pub name: String,
        /// OIP datatype name.
        #[prost(string, tag = "2")]
        pub datatype: String,
        /// Tensor shape.
        #[prost(int64, repeated, tag = "3")]
        pub shape: Vec<i64>,
        /// Tensor parameters.
        #[prost(btree_map = "string, message", tag = "4")]
        pub parameters: BTreeMap<String, InferParameter>,
        /// Typed contents.
        #[prost(message, optional, tag = "5")]
        pub contents: Option<InferTensorContents>,
    }
}

/// One server-streaming inference envelope.
#[derive(Clone, PartialEq, Message)]
pub struct ModelStreamInferResponse {
    /// In-band stream error, empty for success.
    #[prost(string, tag = "1")]
    pub error_message: String,
    /// Inference response.
    #[prost(message, optional, tag = "2")]
    pub infer_response: Option<ModelInferResponse>,
}
