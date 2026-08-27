// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Boundary-owned endpoint values and the narrow views a formatter needs.
//!
//! An endpoint plugin composes a request from three boundary facts: the dense
//! [`Handle`] that names one frozen pre-serialized content segment, the
//! [`SegmentReader`] view that resolves a handle to its exact wire bytes, and
//! the [`Overrides`] set of authored per-dispatch top-level fields. The
//! WebSocket operation values are here for the same reason: they are store-free
//! by construction, so a dialect can prepare a complete operation without the
//! runtime's segment arena.
//!
//! The full body-planning machinery stays runtime-owned: it closes over the
//! mutable segment arena, its payload kinds, and the dataset error taxonomy,
//! none of which belong on the plugin boundary.

use std::fmt::{self, Display, Formatter};

use base64::Engine;
use base64::engine::general_purpose::STANDARD;
use bytes::Bytes;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

/// Dense opaque index into a frozen segment arena.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Handle(u32);

impl Handle {
    /// Construct a handle from its arena index.
    pub const fn new(index: u32) -> Self {
        Self(index)
    }

    /// Return the underlying arena index.
    pub const fn index(self) -> u32 {
        self.0
    }

    /// Return the arena index as `usize` for slice access.
    pub const fn as_usize(self) -> usize {
        self.0 as usize
    }
}

impl Display for Handle {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        Display::fmt(&self.0, f)
    }
}

/// The narrow read view an endpoint formatter needs over a frozen segment store.
///
/// This is deliberately smaller than the runtime's segment-store seam: a
/// formatter resolves handles to exact pre-serialized wire bytes and never sees
/// the arena, its payload kinds, its content identifiers, or its interner.
pub trait SegmentReader {
    /// Return one segment's exact pre-serialized wire bytes, or `None` for an
    /// unknown handle.
    fn wire(&self, handle: Handle) -> Option<Bytes>;
}

/// Per-dispatch top-level request fields spliced after the static message array.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Overrides {
    fields: Map<String, Value>,
}

impl Overrides {
    /// Construct an empty override set.
    pub fn new() -> Self {
        Self::default()
    }

    /// Construct overrides from an insertion-ordered JSON object.
    pub fn from_map(fields: Map<String, Value>) -> Self {
        Self { fields }
    }

    /// Whether no per-dispatch fields will be inserted.
    pub fn is_empty(&self) -> bool {
        self.fields.is_empty()
    }

    /// Number of top-level fields.
    pub fn len(&self) -> usize {
        self.fields.len()
    }

    /// Insert or replace one field; later inserts win within the override tail.
    pub fn insert(&mut self, key: impl Into<String>, value: Value) -> Option<Value> {
        self.fields.insert(key.into(), value)
    }

    /// Set a model override.
    pub fn set_model(&mut self, model: impl Into<String>) {
        self.insert("model", Value::String(model.into()));
    }

    /// Set the endpoint-selected generation-cap field.
    pub fn set_max_tokens(&mut self, field_name: impl Into<String>, max_tokens: u32) {
        self.insert(field_name, Value::from(max_tokens));
    }

    /// Set streaming behavior.
    pub fn set_stream(&mut self, stream: bool) {
        self.insert("stream", Value::Bool(stream));
    }

    /// Request authoritative usage in streaming responses.
    pub fn set_include_usage(&mut self, include_usage: bool) {
        let mut stream_options = Map::new();
        stream_options.insert("include_usage".into(), Value::Bool(include_usage));
        self.insert("stream_options", Value::Object(stream_options));
    }

    /// Borrow the decoded fields for endpoint-specific augmentation.
    pub fn fields(&self) -> &Map<String, Value> {
        &self.fields
    }

    /// Serialize the override fields as a spliceable tail with the enclosing
    /// braces stripped, so callers can insert them into an existing object.
    /// Public so callers can pre-serialize a reusable tail once and feed it to
    /// the wire-parts body builder instead of re-serializing per dispatch.
    pub fn inner_bytes(&self) -> serde_json::Result<Vec<u8>> {
        if self.fields.is_empty() {
            return Ok(Vec::new());
        }
        let encoded = serde_json::to_vec(&self.fields)?;
        debug_assert_eq!(encoded.first(), Some(&b'{'));
        debug_assert_eq!(encoded.last(), Some(&b'}'));
        Ok(encoded[1..encoded.len() - 1].to_vec())
    }
}

/// WebSocket application opcode selected by an endpoint dialect.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PreparedWsOpcode {
    /// A UTF-8 text application message.
    Text,
    /// An opaque binary application message.
    Binary,
}

/// Logical role of one complete WebSocket application message.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PreparedWsMessageRole {
    /// Request-scoped input that contributes to application-event timing.
    MeasuredInput,
    /// Session or protocol control message excluded from timing.
    Control,
    /// Terminal acknowledgement excluded from timing.
    TerminalAck,
}

/// One immutable complete application message.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PreparedWsMessage {
    opcode: PreparedWsOpcode,
    payload: Bytes,
    role: PreparedWsMessageRole,
}

impl PreparedWsMessage {
    /// Build an application message with the endpoint-selected opcode.
    pub fn new(opcode: PreparedWsOpcode, payload: Bytes, role: PreparedWsMessageRole) -> Self {
        Self {
            opcode,
            payload,
            role,
        }
    }

    /// Build a text application message.
    pub fn text(payload: Bytes, role: PreparedWsMessageRole) -> Self {
        Self::new(PreparedWsOpcode::Text, payload, role)
    }

    /// Return the endpoint-selected application opcode.
    pub fn opcode(&self) -> PreparedWsOpcode {
        self.opcode
    }

    /// Borrow the complete immutable message payload.
    pub fn payload(&self) -> &Bytes {
        &self.payload
    }

    /// Return how this message participates in request timing.
    pub fn role(&self) -> PreparedWsMessageRole {
        self.role
    }
}

/// Failure to serialize one prepared WebSocket operation for an artifact.
#[derive(Debug)]
pub enum PreparedWsArtifactError {
    /// A message declared as text does not contain UTF-8.
    InvalidText(std::str::Utf8Error),
    /// The canonical artifact envelope could not be serialized.
    Serialization(serde_json::Error),
}

impl Display for PreparedWsArtifactError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidText(error) => write!(
                formatter,
                "websocket text application message is not UTF-8: {error}"
            ),
            Self::Serialization(error) => {
                write!(
                    formatter,
                    "serializing websocket operation artifact: {error}"
                )
            }
        }
    }
}

impl std::error::Error for PreparedWsArtifactError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::InvalidText(error) => Some(error),
            Self::Serialization(error) => Some(error),
        }
    }
}

/// Immutable, store-free application messages for one WebSocket operation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PreparedWsOperation {
    messages: Box<[PreparedWsMessage]>,
    http_sse_fallback_body: Option<Bytes>,
    input_projection: Option<Bytes>,
    requires_affinity_state: bool,
}

impl PreparedWsOperation {
    /// Freeze application messages and an optional independently prepared HTTP/SSE body.
    pub fn new(
        messages: impl IntoIterator<Item = PreparedWsMessage>,
        http_sse_fallback_body: Option<Bytes>,
    ) -> Self {
        Self {
            messages: messages.into_iter().collect(),
            http_sse_fallback_body,
            input_projection: None,
            requires_affinity_state: false,
        }
    }

    /// Retain the endpoint request body used for input extraction and counting.
    pub fn with_input_projection(mut self, input_projection: Bytes) -> Self {
        self.input_projection = Some(input_projection);
        self
    }

    /// Borrow the complete application messages in send order.
    pub fn messages(&self) -> &[PreparedWsMessage] {
        &self.messages
    }

    /// Borrow the equivalent HTTP/SSE request body when the dialect prepared one.
    pub fn http_sse_fallback_body(&self) -> Option<&Bytes> {
        self.http_sse_fallback_body.as_ref()
    }

    /// Borrow the endpoint request body used for input extraction and counting.
    pub fn input_projection(&self) -> Option<&Bytes> {
        self.input_projection.as_ref()
    }

    /// Mark that this operation contains only a continuation turn.
    pub fn requiring_affinity_state(mut self) -> Self {
        self.requires_affinity_state = true;
        self
    }

    /// Whether dispatch must use the logical session's affinity-bound socket.
    pub fn requires_affinity_state(&self) -> bool {
        self.requires_affinity_state
    }

    /// Replace the application messages while retaining operation metadata.
    ///
    /// Crate-public in the runtime before the move; public here because the
    /// WebSocket dialect that rewrites an operation's messages now lives on the
    /// far side of the crate boundary.
    pub fn with_messages(&self, messages: impl IntoIterator<Item = PreparedWsMessage>) -> Self {
        Self {
            messages: messages.into_iter().collect(),
            http_sse_fallback_body: self.http_sse_fallback_body.clone(),
            input_projection: self.input_projection.clone(),
            requires_affinity_state: self.requires_affinity_state,
        }
    }

    /// Serialize the complete ordered operation into the canonical artifact envelope.
    pub fn to_artifact_bytes(&self) -> std::result::Result<Bytes, PreparedWsArtifactError> {
        let messages = self
            .messages
            .iter()
            .map(|message| {
                let opcode = match message.opcode {
                    PreparedWsOpcode::Text => "text",
                    PreparedWsOpcode::Binary => "binary",
                };
                let role = match message.role {
                    PreparedWsMessageRole::MeasuredInput => "measured_input",
                    PreparedWsMessageRole::Control => "control",
                    PreparedWsMessageRole::TerminalAck => "terminal_ack",
                };
                let payload = match message.opcode {
                    PreparedWsOpcode::Text => Value::String(
                        std::str::from_utf8(&message.payload)
                            .map_err(PreparedWsArtifactError::InvalidText)?
                            .to_owned(),
                    ),
                    PreparedWsOpcode::Binary => Value::String(STANDARD.encode(&message.payload)),
                };
                Ok(serde_json::json!({
                    "opcode": opcode,
                    "role": role,
                    "payload": payload,
                }))
            })
            .collect::<std::result::Result<Vec<Value>, PreparedWsArtifactError>>()?;
        serde_json::to_vec(&serde_json::json!({
            "transport":"websocket",
            "messages":messages,
        }))
        .map(Bytes::from)
        .map_err(PreparedWsArtifactError::Serialization)
    }
}
