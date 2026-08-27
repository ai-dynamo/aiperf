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
/// The wire is produced once when the value is *bound to a field*, not once per
/// dispatch: the materializer splices it verbatim, exactly as it splices segment
/// and message wires. That keeps literal serialization off the dispatch path for
/// a cached plan entirely, and makes it exactly one pass for a plan rebuilt per
/// dispatch (`precomputable_body() == false` endpoints, graph nodes, warmup) —
/// the same count as re-serializing at materialization, but the bytes are then
/// also free to measure, which is what lets [`FieldProgram`] reserve exactly.
///
/// This matters most where a literal is large. `from_object` converts a
/// top-level array to spliceable [`Wires`](FieldValue::Wires) only when every
/// element is an *object*, so the prompt-sized `input` string arrays that the
/// embeddings endpoints bind stay literals — and those plans also carry `model`,
/// so [`prebuilt_if_static`](BodyPlan::prebuilt_if_static) cannot collapse them
/// and the program stays alive for the whole run.
///
/// The wire is absent only when the value would not serialize, in which case
/// materialization re-attempts it and surfaces the error.
#[derive(Debug, Clone)]
pub struct LiteralValue {
    value: Value,
    wire: Option<Bytes>,
}

/// The value bound to one [`BodyPlan`] field.
///
/// Content values are *segment references*, never inline bytes: the endpoint
/// declares which stored segment fills a slot and the materializer splices its
/// pre-serialized wire bytes. Everything endpoint-generated that has no content
/// segment is a [`Literal`] — usually a scalar or a small struct (`model`,
/// `max_tokens`, `stream`, `sampling_params`, …), but also prompt-sized values
/// that are not object arrays, such as an embeddings `input` string array or
/// `vllm_generate`'s token-ID array.
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
/// `exact_len` becomes the [`SizeHint`] that lets [`JsonEmitter`] reserve the
/// finished body in one allocation instead of growing a guessed buffer per
/// dispatch. It counts the enclosing braces, every `"name":` frame, every
/// separating comma, and each value's serialized bytes — but **not** the
/// per-dispatch override tail, which is not part of the program.
///
/// It is `None` whenever a [`Segment`](FieldValue::Segment) or
/// [`Segments`](FieldValue::Segments) value makes the length depend on a segment
/// store the program does not hold, whenever a literal fails to serialize, and
/// whenever a [`Reserved`](FieldValue::Reserved) slot is still unfilled — such a
/// program has no serialized length at all, since materializing it is an error.
/// A `None` hint costs only the old capacity heuristic; a *wrong* hint would be
/// a silent regression, so every mutator either maintains it exactly or clears
/// it, and a debug assertion checks it against the finished buffer.
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
