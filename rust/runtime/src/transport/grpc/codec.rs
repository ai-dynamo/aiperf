// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! KServe OIP v2 canonical-JSON/protobuf conversion.

use std::error::Error;
use std::fmt;

use bytes::Bytes;
use half::f16;
use prost::Message;
use serde_json::{Map, Number, Value};

use crate::transport::grpc::proto::infer_parameter::ParameterChoice;
use crate::transport::grpc::proto::model_infer_request::{
    InferInputTensor, InferRequestedOutputTensor,
};
use crate::transport::grpc::proto::{
    InferParameter, InferTensorContents, ModelInferRequest, ModelInferResponse, ModelReadyRequest,
    ModelReadyResponse, ModelStreamInferResponse,
};

/// Canonical JSON/protobuf conversion failure.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CodecError {
    message: String,
}

impl CodecError {
    /// Construct an extension-owned codec failure.
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for CodecError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl Error for CodecError {}

impl From<prost::DecodeError> for CodecError {
    fn from(error: prost::DecodeError) -> Self {
        Self::new(format!("decode KServe protobuf: {error}"))
    }
}

/// Encode a canonical KServe v2 endpoint body as `ModelInferRequest` bytes.
pub fn encode_model_infer_request(
    payload: &Value,
    model_name: &str,
    model_version: &str,
    request_id: &str,
) -> Result<Bytes, CodecError> {
    let object = payload
        .as_object()
        .ok_or_else(|| CodecError::new("KServe v2 payload must be a JSON object"))?;
    let parameters = object
        .get("parameters")
        .map(parameter_map)
        .transpose()?
        .unwrap_or_default();
    let inputs = object
        .get("inputs")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .map(input_tensor)
        .collect::<Result<Vec<_>, _>>()?;
    let outputs = object
        .get("outputs")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .map(requested_output_tensor)
        .collect::<Result<Vec<_>, _>>()?;
    let request = ModelInferRequest {
        model_name: model_name.to_string(),
        model_version: model_version.to_string(),
        id: request_id.to_string(),
        parameters,
        inputs,
        outputs,
        raw_input_contents: Vec::new(),
    };
    Ok(Bytes::from(request.encode_to_vec()))
}

/// Encode a KServe model-readiness request.
pub fn encode_model_ready_request(model_name: &str, model_version: &str) -> Bytes {
    Bytes::from(
        ModelReadyRequest {
            name: model_name.to_string(),
            version: model_version.to_string(),
        }
        .encode_to_vec(),
    )
}

/// Decode a KServe model-readiness response.
pub fn decode_model_ready_response(bytes: &[u8]) -> Result<bool, CodecError> {
    Ok(ModelReadyResponse::decode(bytes)?.ready)
}

/// Decode `ModelInferResponse` bytes to the canonical KServe v2 JSON shape.
pub fn decode_model_infer_response(bytes: &[u8]) -> Result<Value, CodecError> {
    response_to_json(&ModelInferResponse::decode(bytes)?)
}

/// Decode one server-streaming envelope.
pub fn decode_model_stream_infer_response(
    bytes: &[u8],
) -> Result<(Option<String>, Option<Value>), CodecError> {
    let response = ModelStreamInferResponse::decode(bytes)?;
    if !response.error_message.is_empty() {
        return Ok((Some(response.error_message), None));
    }
    let infer_response = response.infer_response.unwrap_or_default();
    Ok((None, Some(response_to_json(&infer_response)?)))
}

fn input_tensor(value: &Value) -> Result<InferInputTensor, CodecError> {
    let object = value
        .as_object()
        .ok_or_else(|| CodecError::new("KServe input tensor must be an object"))?;
    let name = required_string(object, "name")?;
    let datatype = required_string(object, "datatype")?.to_ascii_uppercase();
    let shape = required_shape(object)?;
    let parameters = object
        .get("parameters")
        .map(parameter_map)
        .transpose()?
        .unwrap_or_default();
    let contents = object
        .get("data")
        .map(|data| tensor_contents(&datatype, data))
        .transpose()?;
    Ok(InferInputTensor {
        name,
        datatype,
        shape,
        parameters,
        contents,
    })
}

fn requested_output_tensor(value: &Value) -> Result<InferRequestedOutputTensor, CodecError> {
    let object = value
        .as_object()
        .ok_or_else(|| CodecError::new("KServe requested output tensor must be an object"))?;
    Ok(InferRequestedOutputTensor {
        name: required_string(object, "name")?,
        parameters: object
            .get("parameters")
            .map(parameter_map)
            .transpose()?
            .unwrap_or_default(),
    })
}

fn required_string(object: &Map<String, Value>, name: &str) -> Result<String, CodecError> {
    object
        .get(name)
        .and_then(Value::as_str)
        .map(ToString::to_string)
        .ok_or_else(|| CodecError::new(format!("KServe tensor field {name:?} must be a string")))
}

fn required_shape(object: &Map<String, Value>) -> Result<Vec<i64>, CodecError> {
    object
        .get("shape")
        .and_then(Value::as_array)
        .ok_or_else(|| CodecError::new("KServe tensor field \"shape\" must be an array"))?
        .iter()
        .map(|dimension| {
            dimension
                .as_i64()
                .ok_or_else(|| CodecError::new("KServe tensor dimensions must be integers"))
        })
        .collect()
}

fn parameter_map(
    value: &Value,
) -> Result<std::collections::BTreeMap<String, InferParameter>, CodecError> {
    value
        .as_object()
        .ok_or_else(|| CodecError::new("KServe parameters must be an object"))?
        .iter()
        .map(|(name, value)| Ok((name.clone(), infer_parameter(value)?)))
        .collect()
}

fn infer_parameter(value: &Value) -> Result<InferParameter, CodecError> {
    let parameter_choice = match value {
        Value::Bool(value) => ParameterChoice::BoolParam(*value),
        Value::Number(value) if value.is_i64() => {
            ParameterChoice::Int64Param(value.as_i64().expect("checked integer"))
        }
        Value::Number(value) if value.is_u64() => {
            let value = value
                .as_u64()
                .and_then(|value| i64::try_from(value).ok())
                .ok_or_else(|| CodecError::new("KServe integer parameter exceeds int64"))?;
            ParameterChoice::Int64Param(value)
        }
        Value::Number(value) => ParameterChoice::DoubleParam(
            value
                .as_f64()
                .ok_or_else(|| CodecError::new("invalid KServe numeric parameter"))?,
        ),
        Value::String(value) => ParameterChoice::StringParam(value.clone()),
        Value::Null | Value::Array(_) | Value::Object(_) => {
            ParameterChoice::StringParam(python_string(value))
        }
    };
    Ok(InferParameter {
        parameter_choice: Some(parameter_choice),
    })
}

fn tensor_contents(datatype: &str, value: &Value) -> Result<InferTensorContents, CodecError> {
    let data = value
        .as_array()
        .ok_or_else(|| CodecError::new("KServe tensor data must be an array"))?;
    let mut contents = InferTensorContents::default();
    match datatype {
        "BYTES" => {
            contents.bytes_contents = data
                .iter()
                .map(|value| match value {
                    Value::String(value) => value.as_bytes().to_vec(),
                    _ => python_string(value).into_bytes(),
                })
                .collect();
        }
        "INT8" | "INT16" | "INT32" => {
            contents.int_contents = data
                .iter()
                .map(|value| json_i64(value).and_then(|value| i32::try_from(value).ok()))
                .map(|value| value.ok_or_else(|| CodecError::new("invalid INT32 tensor value")))
                .collect::<Result<_, _>>()?;
        }
        "INT64" => {
            contents.int64_contents = data
                .iter()
                .map(json_i64)
                .map(|value| value.ok_or_else(|| CodecError::new("invalid INT64 tensor value")))
                .collect::<Result<_, _>>()?;
        }
        "UINT8" | "UINT16" | "UINT32" => {
            contents.uint_contents = data
                .iter()
                .map(|value| json_u64(value).and_then(|value| u32::try_from(value).ok()))
                .map(|value| value.ok_or_else(|| CodecError::new("invalid UINT32 tensor value")))
                .collect::<Result<_, _>>()?;
        }
        "UINT64" => {
            contents.uint64_contents = data
                .iter()
                .map(json_u64)
                .map(|value| value.ok_or_else(|| CodecError::new("invalid UINT64 tensor value")))
                .collect::<Result<_, _>>()?;
        }
        "FP16" | "FP32" => {
            contents.fp32_contents = data
                .iter()
                .map(json_f64)
                .map(|value| value.map(|value| value as f32))
                .map(|value| value.ok_or_else(|| CodecError::new("invalid FP32 tensor value")))
                .collect::<Result<_, _>>()?;
        }
        "FP64" => {
            contents.fp64_contents = data
                .iter()
                .map(json_f64)
                .map(|value| value.ok_or_else(|| CodecError::new("invalid FP64 tensor value")))
                .collect::<Result<_, _>>()?;
        }
        "BOOL" => {
            contents.bool_contents = data.iter().map(python_bool).collect();
        }
        _ => {
            contents.bytes_contents = data
                .iter()
                .map(|value| python_string(value).into_bytes())
                .collect();
        }
    }
    Ok(contents)
}

fn response_to_json(response: &ModelInferResponse) -> Result<Value, CodecError> {
    let mut outputs = Vec::with_capacity(response.outputs.len());
    for (index, output) in response.outputs.iter().enumerate() {
        let datatype = output.datatype.to_ascii_uppercase();
        let mut data = output
            .contents
            .as_ref()
            .map(|contents| typed_tensor_data(contents, &datatype))
            .transpose()?
            .unwrap_or_default();
        if data.is_empty()
            && let Some(raw) = response.raw_output_contents.get(index)
        {
            data = raw_tensor_data(raw, &datatype, &output.shape)?;
        }
        outputs.push(serde_json::json!({
            "name": output.name,
            "datatype": output.datatype,
            "shape": output.shape,
            "data": data,
        }));
    }
    let mut result = Map::new();
    result.insert("outputs".to_string(), Value::Array(outputs));
    if !response.model_name.is_empty() {
        result.insert(
            "model_name".to_string(),
            Value::String(response.model_name.clone()),
        );
    }
    if !response.model_version.is_empty() {
        result.insert(
            "model_version".to_string(),
            Value::String(response.model_version.clone()),
        );
    }
    if !response.id.is_empty() {
        result.insert("id".to_string(), Value::String(response.id.clone()));
    }
    Ok(Value::Object(result))
}

fn typed_tensor_data(
    contents: &InferTensorContents,
    datatype: &str,
) -> Result<Vec<Value>, CodecError> {
    Ok(match datatype {
        "BYTES" => contents
            .bytes_contents
            .iter()
            .map(|value| Value::String(String::from_utf8_lossy(value).into_owned()))
            .collect(),
        "INT8" | "INT16" | "INT32" => contents
            .int_contents
            .iter()
            .copied()
            .map(Number::from)
            .map(Value::Number)
            .collect(),
        "INT64" => contents
            .int64_contents
            .iter()
            .copied()
            .map(Number::from)
            .map(Value::Number)
            .collect(),
        "UINT8" | "UINT16" | "UINT32" => contents
            .uint_contents
            .iter()
            .copied()
            .map(Number::from)
            .map(Value::Number)
            .collect(),
        "UINT64" => contents
            .uint64_contents
            .iter()
            .copied()
            .map(Number::from)
            .map(Value::Number)
            .collect(),
        "FP16" | "FP32" => contents
            .fp32_contents
            .iter()
            .map(|value| json_number(f64::from(*value)))
            .collect::<Result<_, _>>()?,
        "FP64" => contents
            .fp64_contents
            .iter()
            .map(|value| json_number(*value))
            .collect::<Result<_, _>>()?,
        "BOOL" => contents
            .bool_contents
            .iter()
            .copied()
            .map(Value::Bool)
            .collect(),
        _ => contents
            .bytes_contents
            .iter()
            .map(|value| Value::String(String::from_utf8_lossy(value).into_owned()))
            .collect(),
    })
}

fn raw_tensor_data(raw: &[u8], datatype: &str, shape: &[i64]) -> Result<Vec<Value>, CodecError> {
    if datatype == "BYTES" {
        let mut values = Vec::new();
        let mut offset = 0usize;
        while offset < raw.len() {
            let Some(length_bytes) = raw.get(offset..offset.saturating_add(4)) else {
                break;
            };
            let length =
                u32::from_le_bytes(length_bytes.try_into().expect("slice length checked above"))
                    as usize;
            offset += 4;
            let value = &raw[offset..raw.len().min(offset.saturating_add(length))];
            values.push(Value::String(String::from_utf8_lossy(value).into_owned()));
            offset += length;
        }
        return Ok(values);
    }
    let count = shape
        .iter()
        .try_fold(1i64, |count, dimension| count.checked_mul(*dimension));
    let Some(count) = count.filter(|count| *count > 0) else {
        return Ok(Vec::new());
    };
    let count = usize::try_from(count).map_err(|_| CodecError::new("tensor shape is too large"))?;
    let width = datatype_width(datatype);
    let Some(width) = width else {
        return Ok(vec![Value::String(
            String::from_utf8_lossy(raw).into_owned(),
        )]);
    };
    let required = count
        .checked_mul(width)
        .ok_or_else(|| CodecError::new("tensor byte length overflows usize"))?;
    if raw.len() < required {
        return Err(CodecError::new(format!(
            "raw {datatype} tensor needs {required} bytes for shape {shape:?}, got {}",
            raw.len()
        )));
    }
    (0..count)
        .map(|index| raw_scalar(&raw[index * width..(index + 1) * width], datatype))
        .collect()
}

fn datatype_width(datatype: &str) -> Option<usize> {
    match datatype {
        "BOOL" | "INT8" | "UINT8" => Some(1),
        "INT16" | "UINT16" | "FP16" => Some(2),
        "INT32" | "UINT32" | "FP32" => Some(4),
        "INT64" | "UINT64" | "FP64" => Some(8),
        _ => None,
    }
}

fn raw_scalar(bytes: &[u8], datatype: &str) -> Result<Value, CodecError> {
    Ok(match datatype {
        "BOOL" => Value::Bool(bytes[0] != 0),
        "INT8" => Value::Number(Number::from(i8::from_le_bytes([bytes[0]]))),
        "UINT8" => Value::Number(Number::from(bytes[0])),
        "INT16" => Value::Number(Number::from(i16::from_le_bytes(array(bytes)?))),
        "UINT16" => Value::Number(Number::from(u16::from_le_bytes(array(bytes)?))),
        "INT32" => Value::Number(Number::from(i32::from_le_bytes(array(bytes)?))),
        "UINT32" => Value::Number(Number::from(u32::from_le_bytes(array(bytes)?))),
        "INT64" => Value::Number(Number::from(i64::from_le_bytes(array(bytes)?))),
        "UINT64" => Value::Number(Number::from(u64::from_le_bytes(array(bytes)?))),
        "FP16" => json_number(f64::from(f16::from_bits(u16::from_le_bytes(array(bytes)?))))?,
        "FP32" => json_number(f64::from(f32::from_le_bytes(array(bytes)?)))?,
        "FP64" => json_number(f64::from_le_bytes(array(bytes)?))?,
        _ => Value::String(String::from_utf8_lossy(bytes).into_owned()),
    })
}

fn array<const N: usize>(bytes: &[u8]) -> Result<[u8; N], CodecError> {
    bytes
        .try_into()
        .map_err(|_| CodecError::new(format!("expected {N} tensor bytes, got {}", bytes.len())))
}

fn json_number(value: f64) -> Result<Value, CodecError> {
    Number::from_f64(value)
        .map(Value::Number)
        .ok_or_else(|| CodecError::new("KServe tensor contains non-finite float"))
}

fn json_i64(value: &Value) -> Option<i64> {
    value
        .as_i64()
        .or_else(|| value.as_bool().map(i64::from))
        .or_else(|| value.as_str().and_then(|value| value.parse().ok()))
}

fn json_u64(value: &Value) -> Option<u64> {
    value
        .as_u64()
        .or_else(|| value.as_bool().map(u64::from))
        .or_else(|| value.as_str().and_then(|value| value.parse().ok()))
}

fn json_f64(value: &Value) -> Option<f64> {
    value
        .as_f64()
        .or_else(|| value.as_str().and_then(|value| value.parse().ok()))
}

fn python_bool(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(value) => *value,
        Value::Number(value) => value.as_f64().is_some_and(|value| value != 0.0),
        Value::String(value) => !value.is_empty(),
        Value::Array(value) => !value.is_empty(),
        Value::Object(value) => !value.is_empty(),
    }
}

fn python_string(value: &Value) -> String {
    match value {
        Value::Null => "None".to_string(),
        Value::Bool(true) => "True".to_string(),
        Value::Bool(false) => "False".to_string(),
        Value::String(value) => value.clone(),
        Value::Number(value) => value.to_string(),
        Value::Array(_) | Value::Object(_) => value.to_string(),
    }
}
