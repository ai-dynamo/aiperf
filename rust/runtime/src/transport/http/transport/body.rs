// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Pluggable request-body encoding, including multipart file descriptors.
//!
//! Endpoint formatters keep binary data JSON-safe as
//! `{b64_data, filename, content_type}` until this final wire boundary.

use std::hash::{Hash, Hasher};

use base64::Engine;
use base64::engine::general_purpose::STANDARD;
use bytes::Bytes;
use serde_json::Value;

use crate::transport::core::ErrorDetails;

/// An encoded request body and the exact wire content type it requires.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EncodedRequestBody {
    /// Complete request bytes.
    pub bytes: Bytes,
    /// Content-Type header, including a multipart boundary when applicable.
    pub content_type: String,
}

/// Extensible conversion from a decoded payload into transport-ready bytes.
pub trait RequestBodyEncoder {
    /// Encode one request payload.
    fn encode(&self, payload: &Value) -> Result<EncodedRequestBody, ErrorDetails>;
}

/// Compact application/json request encoder.
#[derive(Debug, Clone, Copy, Default)]
pub struct JsonBodyEncoder;

impl RequestBodyEncoder for JsonBodyEncoder {
    fn encode(&self, payload: &Value) -> Result<EncodedRequestBody, ErrorDetails> {
        serde_json::to_vec(payload)
            .map(Bytes::from)
            .map(|bytes| EncodedRequestBody {
                bytes,
                content_type: "application/json".into(),
            })
            .map_err(|error| ErrorDetails::other(format!("serialize JSON body: {error}")))
    }
}

/// RFC 7578 multipart/form-data encoder for endpoint file descriptors.
#[derive(Debug, Clone, Copy, Default)]
pub struct MultipartBodyEncoder;

impl RequestBodyEncoder for MultipartBodyEncoder {
    fn encode(&self, payload: &Value) -> Result<EncodedRequestBody, ErrorDetails> {
        let object = payload.as_object().ok_or_else(|| {
            ErrorDetails::other("multipart request payload must be a JSON object")
        })?;
        let boundary = boundary_for(payload)?;
        let mut body = Vec::new();
        for (name, value) in object {
            if value.is_null() {
                continue;
            }
            validate_disposition_value(name, "field name")?;
            body.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
            if let Some(file) = file_descriptor(value)? {
                validate_disposition_value(&file.filename, "filename")?;
                validate_header_value(&file.content_type, "content type")?;
                body.extend_from_slice(
                    format!(
                        "Content-Disposition: form-data; name=\"{}\"; filename=\"{}\"\r\nContent-Type: {}\r\n\r\n",
                        escape_quoted(name),
                        escape_quoted(&file.filename),
                        file.content_type
                    )
                    .as_bytes(),
                );
                body.extend_from_slice(&file.bytes);
                body.extend_from_slice(b"\r\n");
                continue;
            }
            body.extend_from_slice(
                format!(
                    "Content-Disposition: form-data; name=\"{}\"\r\n\r\n",
                    escape_quoted(name)
                )
                .as_bytes(),
            );
            body.extend_from_slice(form_value(value)?.as_bytes());
            body.extend_from_slice(b"\r\n");
        }
        body.extend_from_slice(format!("--{boundary}--\r\n").as_bytes());
        Ok(EncodedRequestBody {
            bytes: Bytes::from(body),
            content_type: format!("multipart/form-data; boundary={boundary}"),
        })
    }
}

struct FilePart {
    bytes: Vec<u8>,
    filename: String,
    content_type: String,
}

fn file_descriptor(value: &Value) -> Result<Option<FilePart>, ErrorDetails> {
    let Some(object) = value.as_object() else {
        return Ok(None);
    };
    let Some(encoded) = object.get("b64_data").and_then(Value::as_str) else {
        return Ok(None);
    };
    let bytes = STANDARD.decode(encoded).map_err(|error| {
        ErrorDetails::other(format!("multipart b64_data is not valid base64: {error}"))
    })?;
    Ok(Some(FilePart {
        bytes,
        filename: object
            .get("filename")
            .and_then(Value::as_str)
            .unwrap_or("file")
            .to_string(),
        content_type: object
            .get("content_type")
            .and_then(Value::as_str)
            .unwrap_or("application/octet-stream")
            .to_string(),
    }))
}

fn form_value(value: &Value) -> Result<String, ErrorDetails> {
    match value {
        Value::Bool(value) => Ok(value.to_string()),
        Value::String(value) => Ok(value.clone()),
        Value::Number(value) => Ok(value.to_string()),
        Value::Array(_) | Value::Object(_) => serde_json::to_string(value)
            .map_err(|error| ErrorDetails::other(format!("serialize multipart field: {error}"))),
        Value::Null => Ok(String::new()),
    }
}

fn boundary_for(payload: &Value) -> Result<String, ErrorDetails> {
    let serialized = serde_json::to_vec(payload).map_err(|error| {
        ErrorDetails::other(format!("serialize multipart boundary seed: {error}"))
    })?;
    let mut first = std::collections::hash_map::DefaultHasher::new();
    serialized.hash(&mut first);
    let mut second = std::collections::hash_map::DefaultHasher::new();
    0xa1_5e_2f_u64.hash(&mut second);
    serialized.hash(&mut second);
    Ok(format!(
        "aiperf-{:016x}{:016x}",
        first.finish(),
        second.finish()
    ))
}

fn validate_disposition_value(value: &str, label: &str) -> Result<(), ErrorDetails> {
    validate_header_value(value, label)
}

fn validate_header_value(value: &str, label: &str) -> Result<(), ErrorDetails> {
    if value.contains(['\r', '\n']) {
        Err(ErrorDetails::other(format!(
            "multipart {label} must not contain CR or LF"
        )))
    } else {
        Ok(())
    }
}

fn escape_quoted(value: &str) -> String {
    value.replace('\\', "\\\\").replace('"', "\\\"")
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn json_encoder_is_compact_and_typed() {
        let encoded = JsonBodyEncoder.encode(&json!({"a":1})).unwrap();
        assert_eq!(encoded.bytes, br#"{"a":1}"#[..]);
        assert_eq!(encoded.content_type, "application/json");
    }

    #[test]
    fn multipart_encoder_writes_text_bool_and_binary_parts() {
        let encoded = MultipartBodyEncoder
            .encode(&json!({
                "prompt":"edit",
                "enabled":true,
                "skipped":null,
                "image":{"b64_data":"iVBORw0KGgo=","filename":"ref.png","content_type":"image/png"}
            }))
            .unwrap();
        assert!(
            encoded
                .content_type
                .starts_with("multipart/form-data; boundary=aiperf-")
        );
        let body = encoded.bytes;
        assert!(
            body.windows(b"name=\"prompt\"\r\n\r\nedit".len())
                .any(|window| window == b"name=\"prompt\"\r\n\r\nedit")
        );
        assert!(
            body.windows(b"name=\"enabled\"\r\n\r\ntrue".len())
                .any(|window| window == b"name=\"enabled\"\r\n\r\ntrue")
        );
        assert!(
            !body
                .windows(b"skipped".len())
                .any(|window| window == b"skipped")
        );
        assert!(
            body.windows(b"filename=\"ref.png\"".len())
                .any(|window| window == b"filename=\"ref.png\"")
        );
        assert!(body.windows(8).any(|window| window == b"\x89PNG\r\n\x1a\n"));
    }

    #[test]
    fn multipart_encoder_rejects_invalid_base64_and_header_injection() {
        assert!(
            MultipartBodyEncoder
                .encode(&json!({"image":{"b64_data":"!!!"}}))
                .is_err()
        );
        assert!(
            MultipartBodyEncoder
                .encode(&json!({"bad\r\nname":"x"}))
                .is_err()
        );
    }
}
