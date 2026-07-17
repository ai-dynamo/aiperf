// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! KServe OIP v2 wire-compatibility tests.

use std::collections::BTreeMap;

use aiperf_runtime::endpoints::EndpointId;
use aiperf_runtime::transport::grpc::proto::infer_parameter::ParameterChoice;
use aiperf_runtime::transport::grpc::proto::model_infer_response::InferOutputTensor;
use aiperf_runtime::transport::grpc::proto::{
    InferTensorContents, ModelInferRequest, ModelInferResponse, ModelStreamInferResponse,
};
use aiperf_runtime::transport::grpc::{
    CodecError, GrpcBindingRegistry, GrpcBindingRegistryBuilder, GrpcEndpointBinding,
    GrpcEndpointBindingFactory, GrpcEndpointBindingRegistryError, decode_model_infer_response,
    decode_model_stream_infer_response, encode_model_infer_request, grpc_status_to_http,
};
use prost::Message;
use serde_json::json;
use tonic::Code;

#[test]
fn canonical_json_encodes_every_typed_tensor_and_parameter_variant() {
    let payload = json!({
        "parameters": {
            "bool": true,
            "int": 7,
            "float": 1.5,
            "string": "value",
        },
        "inputs": [
            {
                "name": "bytes",
                "shape": [2],
                "datatype": "BYTES",
                "data": ["a", "b"],
                "parameters": {"binary_data_size": 2}
            },
            {"name": "i8", "shape": [2], "datatype": "INT8", "data": [-1, 2]},
            {"name": "i64", "shape": [1], "datatype": "INT64", "data": [3]},
            {"name": "u32", "shape": [1], "datatype": "UINT32", "data": [4]},
            {"name": "u64", "shape": [1], "datatype": "UINT64", "data": [5]},
            {"name": "f16", "shape": [1], "datatype": "FP16", "data": [1.25]},
            {"name": "f64", "shape": [1], "datatype": "FP64", "data": [2.5]},
            {"name": "bool", "shape": [3], "datatype": "BOOL", "data": [false, 1, "x"]},
            {"name": "unknown", "shape": [1], "datatype": "CUSTOM", "data": ["raw"]},
        ],
    });
    let bytes = encode_model_infer_request(&payload, "model", "version", "request").unwrap();
    let request = ModelInferRequest::decode(bytes).unwrap();
    assert_eq!(request.model_name, "model");
    assert_eq!(request.model_version, "version");
    assert_eq!(request.id, "request");
    assert_eq!(
        request.parameters["bool"].parameter_choice,
        Some(ParameterChoice::BoolParam(true))
    );
    assert_eq!(
        request.parameters["int"].parameter_choice,
        Some(ParameterChoice::Int64Param(7))
    );
    assert_eq!(
        request.parameters["float"].parameter_choice,
        Some(ParameterChoice::DoubleParam(1.5))
    );
    assert_eq!(
        request.parameters["string"].parameter_choice,
        Some(ParameterChoice::StringParam("value".to_string()))
    );
    assert_eq!(
        request.inputs[0].contents.as_ref().unwrap().bytes_contents,
        vec![b"a".to_vec(), b"b".to_vec()]
    );
    assert_eq!(
        request.inputs[0].parameters["binary_data_size"].parameter_choice,
        Some(ParameterChoice::Int64Param(2))
    );
    assert_eq!(
        request.inputs[1].contents.as_ref().unwrap().int_contents,
        vec![-1, 2]
    );
    assert_eq!(
        request.inputs[2].contents.as_ref().unwrap().int64_contents,
        vec![3]
    );
    assert_eq!(
        request.inputs[3].contents.as_ref().unwrap().uint_contents,
        vec![4]
    );
    assert_eq!(
        request.inputs[4].contents.as_ref().unwrap().uint64_contents,
        vec![5]
    );
    assert_eq!(
        request.inputs[5].contents.as_ref().unwrap().fp32_contents,
        vec![1.25]
    );
    assert_eq!(
        request.inputs[6].contents.as_ref().unwrap().fp64_contents,
        vec![2.5]
    );
    assert_eq!(
        request.inputs[7].contents.as_ref().unwrap().bool_contents,
        vec![false, true, true]
    );
    assert_eq!(
        request.inputs[8].contents.as_ref().unwrap().bytes_contents,
        vec![b"raw".to_vec()]
    );
}

#[test]
fn request_bytes_match_kserve_v2_serializer_exactly() {
    let payload = json!({
        "inputs": [{
            "name": "text", "shape": [1], "datatype": "BYTES", "data": ["hello"]
        }]
    });
    let bytes = encode_model_infer_request(&payload, "model", "", "request").unwrap();
    assert_eq!(
        bytes.as_ref(),
        b"\x0a\x05model\x1a\x07request\x2a\x19\x0a\x04text\x12\x05BYTES\x1a\x01\x01\x2a\x07B\x05hello"
    );
}

#[test]
fn response_prefers_typed_contents_and_preserves_identity_fields() {
    let response = ModelInferResponse {
        model_name: "model".to_string(),
        model_version: "1".to_string(),
        id: "request".to_string(),
        outputs: vec![InferOutputTensor {
            name: "text".to_string(),
            datatype: "BYTES".to_string(),
            shape: vec![1],
            contents: Some(InferTensorContents {
                bytes_contents: vec![b"typed".to_vec()],
                ..InferTensorContents::default()
            }),
            ..InferOutputTensor::default()
        }],
        raw_output_contents: vec![raw_bytes(&["raw"])],
        ..ModelInferResponse::default()
    };
    assert_eq!(
        decode_model_infer_response(&response.encode_to_vec()).unwrap(),
        json!({
            "outputs": [{
                "name": "text", "datatype": "BYTES", "shape": [1], "data": ["typed"]
            }],
            "model_name": "model",
            "model_version": "1",
            "id": "request",
        })
    );
}

#[test]
fn raw_output_decoding_matches_kserve_bytes_and_numeric_layouts() {
    let outputs = vec![
        output("bytes", "BYTES", vec![2]),
        output("i16", "INT16", vec![2]),
        output("i32", "INT32", vec![2]),
        output("i64", "INT64", vec![1]),
        output("u16", "UINT16", vec![2]),
        output("u64", "UINT64", vec![1]),
        output("f16", "FP16", vec![1]),
        output("f32", "FP32", vec![1]),
        output("f64", "FP64", vec![1]),
        output("bool", "BOOL", vec![2]),
        output("unknown", "CUSTOM", vec![1]),
    ];
    let raw_output_contents = vec![
        raw_bytes(&["hello", "world"]),
        [(-2i16).to_le_bytes(), 3i16.to_le_bytes()].concat(),
        [(-4i32).to_le_bytes(), 5i32.to_le_bytes()].concat(),
        (-6i64).to_le_bytes().to_vec(),
        [7u16.to_le_bytes(), 8u16.to_le_bytes()].concat(),
        9u64.to_le_bytes().to_vec(),
        half::f16::from_f32(1.5).to_bits().to_le_bytes().to_vec(),
        2.5f32.to_le_bytes().to_vec(),
        3.5f64.to_le_bytes().to_vec(),
        vec![0, 1],
        b"opaque".to_vec(),
    ];
    let response = ModelInferResponse {
        outputs,
        raw_output_contents,
        ..ModelInferResponse::default()
    };
    let decoded = decode_model_infer_response(&response.encode_to_vec()).unwrap();
    let data = decoded["outputs"]
        .as_array()
        .unwrap()
        .iter()
        .map(|output| output["data"].clone())
        .collect::<Vec<_>>();
    assert_eq!(data[0], json!(["hello", "world"]));
    assert_eq!(data[1], json!([-2, 3]));
    assert_eq!(data[2], json!([-4, 5]));
    assert_eq!(data[3], json!([-6]));
    assert_eq!(data[4], json!([7, 8]));
    assert_eq!(data[5], json!([9]));
    assert_eq!(data[6], json!([1.5]));
    assert_eq!(data[7], json!([2.5]));
    assert_eq!(data[8], json!([3.5]));
    assert_eq!(data[9], json!([false, true]));
    assert_eq!(data[10], json!(["opaque"]));
}

#[test]
fn stream_envelope_retains_in_band_errors_and_default_responses() {
    let error = ModelStreamInferResponse {
        error_message: "backend failed".to_string(),
        infer_response: None,
    };
    assert_eq!(
        decode_model_stream_infer_response(&error.encode_to_vec()).unwrap(),
        (Some("backend failed".to_string()), None)
    );

    let empty = ModelStreamInferResponse::default();
    assert_eq!(
        decode_model_stream_infer_response(&empty.encode_to_vec()).unwrap(),
        (None, Some(json!({"outputs": []})))
    );
}

#[test]
fn binding_registry_is_open_and_only_advertises_grpc_capable_v2_dialects() {
    let registry = GrpcBindingRegistry::builtin().unwrap();
    let ids = registry
        .endpoint_ids()
        .map(EndpointId::as_str)
        .collect::<Vec<_>>();
    assert_eq!(
        ids,
        vec![
            "kserve_v2_embeddings",
            "kserve_v2_images",
            "kserve_v2_infer",
            "kserve_v2_rankings",
            "kserve_v2_vlm",
            "riva_analyze_entities",
            "riva_analyze_intent",
            "riva_asr",
            "riva_natural_query",
            "riva_punctuate_text",
            "riva_text_classify",
            "riva_token_classify",
            "riva_transform_text",
            "riva_tts",
        ]
    );
    assert!(
        registry
            .prepare(&EndpointId::new("kserve_v1_predict").unwrap())
            .is_err()
    );
    let infer = registry
        .prepare(&EndpointId::new("kserve_v2_infer").unwrap())
        .unwrap();
    assert!(infer.streaming_method().is_some());
    let embeddings = registry
        .prepare(&EndpointId::new("kserve_v2_embeddings").unwrap())
        .unwrap();
    assert!(embeddings.streaming_method().is_none());
}

#[derive(Debug)]
struct DuplicateFactory;

impl GrpcEndpointBindingFactory for DuplicateFactory {
    fn endpoint_ids(&self) -> &'static [&'static str] {
        &["duplicate", "duplicate"]
    }

    fn prepare(
        &self,
        _endpoint_id: &EndpointId,
    ) -> Result<Box<dyn GrpcEndpointBinding>, CodecError> {
        unreachable!("duplicate registration must fail before preparation")
    }
}

#[test]
fn binding_registry_rejects_duplicates_within_one_factory_transactionally() {
    let mut builder = GrpcBindingRegistryBuilder::new();
    assert_eq!(
        builder.register(DuplicateFactory).unwrap_err(),
        GrpcEndpointBindingRegistryError::Duplicate(EndpointId::new("duplicate").unwrap())
    );
    assert_eq!(builder.freeze().endpoint_ids().count(), 0);
}

#[test]
fn all_native_status_codes_map_to_http_equivalents() {
    let cases = [
        (Code::Ok, 200),
        (Code::Cancelled, 499),
        (Code::Unknown, 500),
        (Code::InvalidArgument, 400),
        (Code::DeadlineExceeded, 504),
        (Code::NotFound, 404),
        (Code::AlreadyExists, 409),
        (Code::PermissionDenied, 403),
        (Code::ResourceExhausted, 429),
        (Code::FailedPrecondition, 400),
        (Code::Aborted, 409),
        (Code::OutOfRange, 400),
        (Code::Unimplemented, 501),
        (Code::Internal, 500),
        (Code::Unavailable, 503),
        (Code::DataLoss, 500),
        (Code::Unauthenticated, 401),
    ];
    for (code, expected) in cases {
        assert_eq!(grpc_status_to_http(code), expected);
    }
}

fn output(name: &str, datatype: &str, shape: Vec<i64>) -> InferOutputTensor {
    InferOutputTensor {
        name: name.to_string(),
        datatype: datatype.to_string(),
        shape,
        parameters: BTreeMap::new(),
        contents: None,
    }
}

fn raw_bytes(values: &[&str]) -> Vec<u8> {
    let mut output = Vec::new();
    for value in values {
        output.extend_from_slice(&(value.len() as u32).to_le_bytes());
        output.extend_from_slice(value.as_bytes());
    }
    output
}
