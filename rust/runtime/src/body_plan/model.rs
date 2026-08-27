// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Boundary values for declarative request bodies.

use std::borrow::Cow;

use bytes::Bytes;
use serde_json::Value;
use smallvec::SmallVec;

use crate::dataset::segment::Handle;

/// A JSON object field name. `Cow` so endpoints pass `&'static str` literals for
/// free while a runtime decompose can carry owned keys (user `extra_body`).
pub type FieldName = Cow<'static, str>;

/// An endpoint-generated literal paired with its serialized wire bytes.
///
/// Binding serializes once so cached plans can splice the wire without
/// per-dispatch serialization. The wire is absent only when serialization
/// failed; materialization retries and returns that error.
#[derive(Debug, Clone)]
pub struct LiteralValue {
    value: Value,
    wire: Option<Bytes>,
}

/// The value bound to one [`BodyPlan`] field.
///
/// Content remains a segment reference or pre-serialized wire. Values generated
/// by an endpoint without a content segment use [`Literal`].
///
/// [`Literal`]: FieldValue::Literal
#[derive(Debug, Clone, PartialEq)]
pub enum FieldValue {
    /// An endpoint-generated value with no content segment, serialized when bound.
    Literal(LiteralValue),
    /// One pre-serialized content segment (system block, tools, a nested body).
    Segment(Handle),
    /// An ordered array of message segments, comma-joined inside `[` `]`.
    Segments(SmallVec<[Handle; 1]>),
    /// An ordered array of already-serialized message wires not interned in the
    /// frozen store, including dynamic or live-continuation content. Spliced
    /// identically to [`Segments`](FieldValue::Segments); the materializer needs
    /// no store lookup. Serialized exactly once by the producer, never here.
    Wires(SmallVec<[Bytes; 1]>),
    /// A slot that holds a field's ordinal position until
    /// [`fill_reserved`](BodyPlan::fill_reserved) supplies its wires.
    ///
    /// A dialect that builds its body as a `serde_json::Map` gets field order
    /// from insertion order, so the message field has to be inserted *before*
    /// the fields that follow it — but its wires are assembled separately and
    /// are not a `Value`. `Reserved` is that position, and it has no
    /// serialization: materializing one is an error rather than an empty array,
    /// so a forgotten or misspelled fill fails loudly instead of dispatching a
    /// body with no messages.
    Reserved,
}

/// An ordered named-field program plus the exact serialized length of the body
/// it materializes to.
///
/// `exact_len` lets the emitter reserve the finished body once. It is absent
/// while a segment-store lookup or unfilled reserved slot makes the length
/// unknown; per-dispatch overrides are not part of the cached value.
#[derive(Debug, Clone)]
pub struct FieldProgram {
    fields: SmallVec<[(FieldName, FieldValue); 8]>,
    exact_len: Option<usize>,
}

/// A declarative, wire-agnostic description of a request body.
///
/// Built once per turn at lowering (the run's endpoint is known at config
/// time); dispatch only *materializes* it. A [`Raw`](BodyPlan::Raw) plan is the
/// degenerate whole-body case (recorded `raw_payload` replay / a complete
/// prebuilt body); a [`Fields`](BodyPlan::Fields) plan is an ordered named-field
/// JSON object assembled by the shared materializer.
#[derive(Debug, Clone, PartialEq)]
// A `BodyPlan` is built once per turn at lowering, never in a hot per-dispatch
// loop, so the size gap between the tiny `Raw` handle and the inline field list
// is immaterial; the `;8` inline capacity avoids reallocation for typical bodies.
#[allow(clippy::large_enum_variant)]
pub enum BodyPlan {
    /// A complete prebuilt body spliced/cloned whole (the raw fast path).
    Raw(Handle),
    /// An ordered named-field JSON object.
    Fields(FieldProgram),
    /// A complete body serialized once at lowering into inline bytes, cloned
    /// whole at dispatch. Produced by [`BodyPlan::prebuilt_if_static`] for turns
    /// whose materialization carries no per-dispatch field (no `model`/`stream`/
    /// `max_tokens` tail and no overrides on the scheduled path), so dispatch is
    /// a refcount clone — no buffer allocation, no content memcpy. Distinct from
    /// [`Raw`](BodyPlan::Raw) only in that it holds inline bytes rather than a
    /// frozen-store handle, so it can be built after the segment store is frozen.
    Prebuilt(Bytes),
}

/// One request body crossing the dataset → transport boundary.
///
/// Replaces the mutually-exclusive pair `Request.request_body: Option<Value>`
/// and `Request.request_body_bytes: Option<Bytes>`, deleting the two runtime
/// exclusivity checks the HTTP and gRPC sinks used to carry: the illegal state
/// is no longer representable.
///
/// `Send + Sync` by construction — every member is. That is load-bearing:
/// [`Dispatchable`](crate::dispatch::sink::Dispatchable) is `Send + Sync` and
/// [`Request`](crate::transport::core::Request) implements it. There is no
/// interior mutability here.
#[derive(Clone, Debug, PartialEq)]
pub enum RequestBody {
    /// Assembled wire bytes. Every JSON-over-HTTP transport, every raw-payload
    /// replay, every [`Prebuilt`](BodyPlan::Prebuilt) plan, dynosim, dry-run,
    /// and the cellular request fan-out see only this.
    Wire(Bytes),
    /// A store-free field program retained for a transport that consumes
    /// structure rather than bytes (gRPC KServe/Riva, multipart form
    /// endpoints). Produced only via [`RequestBody::planned`], which rejects
    /// handle-bearing plans, so no [`Handle`] ever crosses the boundary without
    /// its store.
    Plan(std::sync::Arc<BodyPlan>),
    /// Complete application messages consumed only by the WebSocket transport.
    WebSocket(std::sync::Arc<PreparedWsOperation>),
    /// A decoded body supplied by a caller that never had a plan (accuracy
    /// benchmarks, the skeleton workload). Boxed to keep the enum small;
    /// `size_of::<serde_json::Value>()` is several times a `Bytes`.
    Value(Box<Value>),
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

/// Immutable, store-free application messages for one WebSocket operation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PreparedWsOperation {
    messages: Box<[PreparedWsMessage]>,
    http_sse_fallback_body: Option<Bytes>,
    input_projection: Option<Bytes>,
    requires_affinity_state: bool,
}

#[path = "plan.rs"]
mod plan;

pub use plan::*;
