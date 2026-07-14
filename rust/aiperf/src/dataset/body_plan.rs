// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Declarative request-body plans and the shared JSON materializer.
//!
//! This is the dispatch-side companion to the segment-unification storage model
//! (`specs/2026-07-13-endpoint-body-construction-design.md`). An endpoint's
//! formatter stops returning a fully-inline [`serde_json::Value`] and instead
//! declares a [`BodyPlan`]: an ordered list of named fields whose values are
//! either endpoint-generated literal scalars/structs or *segment handles* into
//! the frozen [`SegmentStore`]. A single shared [`JsonBodyMaterializer`] walks
//! the plan and concatenates pre-serialized segment bytes into the one
//! contiguous `Full<Bytes>` request body — **zero content re-serialize**.
//!
//! The materializer is a strict generalization of
//! [`build_message_body_from_wires`](crate::dataset::materialize::build_message_body_from_wires):
//! a plan of `Fields([("messages", Segments(handles))])` materialized with the
//! same [`Overrides`] produces byte-identical output to the legacy
//! `{"messages":[...],<override tail>}` splice path, and a [`BodyPlan::Raw`]
//! plan reproduces the `raw_payload` fast path exactly. The endpoint declares
//! shape with segment slots; it never touches commas, brackets, or content
//! serialization. Protobuf-wire endpoints (KServe V2 / Riva) read the same
//! segments through their codec rather than splicing bytes — see
//! `transport_grpc` — so this materializer is intentionally JSON-only.

use bytes::{BufMut, Bytes, BytesMut};
use serde_json::Value;
use smallvec::SmallVec;

use crate::dataset::error::{DatasetError, Result};
use crate::dataset::materialize::{
    Overrides, message_wire, splice_raw_object, validate_object_slice,
};
use crate::dataset::segment::{Handle, Payload, SegmentStore};

/// A JSON object field name authored by an endpoint formatter.
pub type FieldName = &'static str;

/// The value bound to one [`BodyPlan`] field.
///
/// Content values are *segment references*, never inline bytes: the endpoint
/// declares which stored segment fills a slot and the materializer splices its
/// pre-serialized wire bytes. Only endpoint-generated scalars/structs that have
/// no content segment (`model`, `max_tokens`, `stream`, …) are [`Literal`].
///
/// [`Literal`]: FieldValue::Literal
#[derive(Debug, Clone, PartialEq)]
pub enum FieldValue {
    /// An endpoint-generated scalar or struct, serialized once (small).
    Literal(Value),
    /// One pre-serialized content segment (system block, tools, a nested body).
    Segment(Handle),
    /// An ordered array of message segments, comma-joined inside `[` `]`.
    Segments(SmallVec<[Handle; 1]>),
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
    Fields(SmallVec<[(FieldName, FieldValue); 8]>),
}

impl BodyPlan {
    /// Start an empty named-field plan.
    pub fn new() -> Self {
        Self::Fields(SmallVec::new())
    }

    /// A one-field degenerate plan wrapping a complete prebuilt body.
    pub fn raw(handle: Handle) -> Self {
        Self::Raw(handle)
    }

    // Field builders are only reachable off `new()` (a `Fields` plan); a `Raw`
    // plan is a no-op sink since it carries no named fields.
    fn push(mut self, name: FieldName, value: FieldValue) -> Self {
        if let Self::Fields(fields) = &mut self {
            fields.push((name, value));
        }
        self
    }

    /// Declare an ordered message array (`"name":[seg,seg,…]`).
    pub fn array(self, name: FieldName, handles: impl IntoIterator<Item = Handle>) -> Self {
        self.push(name, FieldValue::Segments(handles.into_iter().collect()))
    }

    /// Declare a single content segment field (`"name":<segment wire>`).
    pub fn segment(self, name: FieldName, handle: Handle) -> Self {
        self.push(name, FieldValue::Segment(handle))
    }

    /// Declare a single content segment field only when present.
    pub fn opt_segment(self, name: FieldName, handle: Option<Handle>) -> Self {
        match handle {
            Some(handle) => self.segment(name, handle),
            None => self,
        }
    }

    /// Declare a literal endpoint-generated value.
    pub fn literal(self, name: FieldName, value: Value) -> Self {
        self.push(name, FieldValue::Literal(value))
    }

    /// Declare a literal string field.
    pub fn str(self, name: FieldName, value: impl Into<String>) -> Self {
        self.literal(name, Value::String(value.into()))
    }

    /// Declare a literal integer field.
    pub fn int(self, name: FieldName, value: u32) -> Self {
        self.literal(name, Value::from(value))
    }

    /// Declare a literal boolean field.
    pub fn bool(self, name: FieldName, value: bool) -> Self {
        self.literal(name, Value::Bool(value))
    }
}

impl Default for BodyPlan {
    fn default() -> Self {
        Self::new()
    }
}

/// The shared JSON splicer: `BodyPlan` + dispatch [`Overrides`] → one `Bytes`.
///
/// Walks the plan in field order, concatenating literal bytes and pre-serialized
/// segment bytes from the store, then appends the small per-dispatch override
/// tail (`model`/`max_tokens`/`stream`/…) exactly as the legacy splice path
/// does. Content is never re-serialized; only [`Literal`](FieldValue::Literal)
/// scalars and the override tail are serialized (both small, once).
pub struct JsonBodyMaterializer;

impl JsonBodyMaterializer {
    /// Materialize a plan against a store into the single request body buffer.
    pub fn materialize<S: SegmentStore + ?Sized>(
        plan: &BodyPlan,
        store: &S,
        overrides: &Overrides,
    ) -> Result<Bytes> {
        match plan {
            BodyPlan::Raw(handle) => match store.get(*handle)? {
                Payload::Raw { wire } => splice_raw_object(wire, overrides),
                payload => Err(DatasetError::PayloadKind {
                    handle: *handle,
                    expected: "raw",
                    actual: payload.kind_name(),
                }),
            },
            BodyPlan::Fields(fields) => materialize_fields(fields, store, overrides),
        }
    }
}

fn materialize_fields<S: SegmentStore + ?Sized>(
    fields: &[(FieldName, FieldValue)],
    store: &S,
    overrides: &Overrides,
) -> Result<Bytes> {
    let override_inner = overrides.inner_bytes()?;
    let mut body = BytesMut::with_capacity(fields.len() * 32 + override_inner.len() + 2);
    body.put_u8(b'{');
    for (index, (name, value)) in fields.iter().enumerate() {
        if index > 0 {
            body.put_u8(b',');
        }
        body.put_u8(b'"');
        body.put_slice(name.as_bytes());
        body.put_slice(b"\":");
        match value {
            // Endpoint-generated scalars/structs serialize straight into the
            // buffer — the only serialization on this path, and small.
            FieldValue::Literal(literal) => serde_json::to_writer((&mut body).writer(), literal)?,
            FieldValue::Segment(handle) => body.put_slice(&segment_field_wire(store, *handle)?),
            FieldValue::Segments(handles) => {
                body.put_u8(b'[');
                for (element, handle) in handles.iter().enumerate() {
                    if element > 0 {
                        body.put_u8(b',');
                    }
                    let wire = message_wire(store, *handle)?;
                    validate_object_slice(&wire).map_err(|error| {
                        DatasetError::InvalidWire(format!("message at index {element}: {error}"))
                    })?;
                    body.put_slice(&wire);
                }
                body.put_u8(b']');
            }
        }
    }
    if !override_inner.is_empty() {
        if !fields.is_empty() {
            body.put_u8(b',');
        }
        body.put_slice(&override_inner);
    }
    body.put_u8(b'}');
    Ok(body.freeze())
}

/// Resolve one non-array content segment to its exact wire bytes. Message and
/// raw segments carry a complete JSON value; text/token/media segments are not
/// spliceable as a JSON field here and are a construction error.
fn segment_field_wire<S: SegmentStore + ?Sized>(store: &S, handle: Handle) -> Result<Bytes> {
    match store.get(handle)? {
        Payload::Message { wire, .. } | Payload::Raw { wire } => Ok(wire.clone()),
        payload => Err(DatasetError::PayloadKind {
            handle,
            expected: "message-or-raw",
            actual: payload.kind_name(),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::materialize::build_message_body_from_wires;
    use crate::dataset::segment::SegmentPool;

    fn message(pool: &mut SegmentPool, parent: Option<Handle>, wire: &'static [u8]) -> Handle {
        pool.intern_message(parent, "user", Bytes::from_static(wire), vec![1_u32])
            .unwrap()
    }

    #[test]
    fn messages_plan_is_byte_identical_to_legacy_splice() {
        let mut pool = SegmentPool::new();
        let system = message(
            &mut pool,
            None,
            br#"{"role":"system","content":"S"}"#,
        );
        let user = message(
            &mut pool,
            Some(system),
            br#"{"content":"hi","role":"user","x":1}"#,
        );
        let mut overrides = Overrides::new();
        overrides.set_model("m");
        overrides.set_stream(true);
        let store = pool.freeze();

        let plan = BodyPlan::new().array("messages", [system, user]);
        let planned = JsonBodyMaterializer::materialize(&plan, &store, &overrides).unwrap();

        // The legacy oracle: the same message wires spliced by the existing path.
        let wires = [
            message_wire(&store, system).unwrap(),
            message_wire(&store, user).unwrap(),
        ];
        let legacy = build_message_body_from_wires(&wires, &overrides).unwrap();

        assert_eq!(planned, legacy);
        assert_eq!(
            planned,
            Bytes::from_static(
                br#"{"messages":[{"role":"system","content":"S"},{"content":"hi","role":"user","x":1}],"model":"m","stream":true}"#
            )
        );
    }

    #[test]
    fn raw_plan_reproduces_raw_payload_fast_path() {
        let mut pool = SegmentPool::new();
        let authored = Bytes::from_static(b" \t{\"z\":1, \"messages\":[]}\n");
        let raw = pool.intern_raw(None, authored.clone()).unwrap();
        let store = pool.freeze();

        let plan = BodyPlan::raw(raw);
        assert_eq!(
            JsonBodyMaterializer::materialize(&plan, &store, &Overrides::new()).unwrap(),
            authored
        );

        let mut overrides = Overrides::new();
        overrides.set_model("new");
        assert_eq!(
            JsonBodyMaterializer::materialize(&plan, &store, &overrides).unwrap(),
            Bytes::from_static(b" \t{\"z\":1, \"messages\":[],\"model\":\"new\"}\n")
        );
    }

    #[test]
    fn mixed_literal_segment_and_array_fields_concatenate_in_order() {
        let mut pool = SegmentPool::new();
        let msg = message(&mut pool, None, br#"{"role":"user","content":"hi"}"#);
        let tools = pool
            .intern_raw(None, Bytes::from_static(br#"[{"type":"function"}]"#))
            .unwrap();
        let store = pool.freeze();

        let plan = BodyPlan::new()
            .str("model", "gpt")
            .array("messages", [msg])
            .segment("tools", tools)
            .int("max_tokens", 7)
            .bool("stream", false);
        let body = JsonBodyMaterializer::materialize(&plan, &store, &Overrides::new()).unwrap();

        assert_eq!(
            body,
            Bytes::from_static(
                br#"{"model":"gpt","messages":[{"role":"user","content":"hi"}],"tools":[{"type":"function"}],"max_tokens":7,"stream":false}"#
            )
        );
        // Must be valid JSON with the expected field values.
        let decoded: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(decoded["model"], "gpt");
        assert_eq!(decoded["max_tokens"], 7);
        assert_eq!(decoded["messages"][0]["content"], "hi");
    }

    #[test]
    fn opt_segment_omits_absent_fields() {
        let mut pool = SegmentPool::new();
        let msg = message(&mut pool, None, br#"{"role":"user","content":"q"}"#);
        let store = pool.freeze();
        let plan = BodyPlan::new()
            .array("messages", [msg])
            .opt_segment("tools", None);
        let body = JsonBodyMaterializer::materialize(&plan, &store, &Overrides::new()).unwrap();
        assert_eq!(
            body,
            Bytes::from_static(br#"{"messages":[{"role":"user","content":"q"}]}"#)
        );
    }

    #[test]
    fn non_spliceable_segment_domains_are_rejected() {
        let mut pool = SegmentPool::new();
        let tokens = pool.intern_token_ids(None, [1_u32, 2]).unwrap();
        let store = pool.freeze();
        let plan = BodyPlan::new().segment("field", tokens);
        assert!(JsonBodyMaterializer::materialize(&plan, &store, &Overrides::new()).is_err());
    }
}
