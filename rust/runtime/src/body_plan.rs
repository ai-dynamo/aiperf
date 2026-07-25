// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Declarative request-body plans and the shared JSON materializer.
//!
//! An endpoint declares a [`BodyPlan`]: an ordered list of named fields whose values are
//! either endpoint-generated literal scalars/structs or *segment handles* into
//! the frozen [`SegmentStore`]. A single shared [`JsonBodyMaterializer`] walks
//! the plan and concatenates pre-serialized segment bytes into the one
//! contiguous `Full<Bytes>` request body — **zero content re-serialize**.
//!
//! A plan of `Fields([("messages", Segments(handles))])` produces
//! `{"messages":[...],<override tail>}` byte-identically to
//! [`build_message_body_from_wires`](crate::dataset::materialize::build_message_body_from_wires),
//! and a [`BodyPlan::Raw`] plan preserves a complete `raw_payload`. The endpoint
//! declares shape with segment slots; it never touches commas, brackets, or content
//! serialization. Protobuf-wire endpoints (KServe V2 / Riva) read the same
//! segments through their codec rather than splicing bytes — see
//! `transport::grpc` — so this materializer is intentionally JSON-only.

use std::borrow::Cow;

use bytes::{BufMut, Bytes, BytesMut};
use serde_json::Value;
use smallvec::SmallVec;

use crate::dataset::error::{DatasetError, Result};
use crate::dataset::materialize::{
    Overrides, message_wire, splice_raw_object, validate_object_slice,
};
use crate::dataset::segment::{Handle, Payload, SegmentStore};

/// A JSON object field name. `Cow` so endpoints pass `&'static str` literals for
/// free while a runtime decompose can carry owned keys (user `extra_body`).
pub type FieldName = Cow<'static, str>;

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
    /// An ordered array of already-serialized message wires not interned in the
    /// frozen store, including dynamic or live-continuation content. Spliced
    /// identically to [`Segments`](FieldValue::Segments); the materializer needs
    /// no store lookup. Serialized exactly once by the producer, never here.
    Wires(SmallVec<[Bytes; 1]>),
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
    /// A complete body serialized once at lowering into inline bytes, cloned
    /// whole at dispatch. Produced by [`BodyPlan::prebuilt_if_static`] for turns
    /// whose materialization carries no per-dispatch field (no `model`/`stream`/
    /// `max_tokens` tail and no overrides on the scheduled path), so dispatch is
    /// a refcount clone — no buffer allocation, no content memcpy. Distinct from
    /// [`Raw`](BodyPlan::Raw) only in that it holds inline bytes rather than a
    /// frozen-store handle, so it can be built after the segment store is frozen.
    Prebuilt(Bytes),
}

impl BodyPlan {
    /// Start an empty named-field plan.
    pub fn new() -> Self {
        Self::Fields(SmallVec::new())
    }

    /// Construct a plan wrapping a complete prebuilt body.
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

    /// Declare an ordered message array of stored segments (`"name":[seg,…]`).
    pub fn array(
        self,
        name: impl Into<FieldName>,
        handles: impl IntoIterator<Item = Handle>,
    ) -> Self {
        self.push(
            name.into(),
            FieldValue::Segments(handles.into_iter().collect()),
        )
    }

    /// Declare an ordered message array of already-serialized wires — dynamic or
    /// live-continuation content not interned in the frozen store.
    pub fn wire_array(
        self,
        name: impl Into<FieldName>,
        wires: impl IntoIterator<Item = Bytes>,
    ) -> Self {
        self.push(name.into(), FieldValue::Wires(wires.into_iter().collect()))
    }

    /// Declare a single content segment field (`"name":<segment wire>`).
    pub fn segment(self, name: impl Into<FieldName>, handle: Handle) -> Self {
        self.push(name.into(), FieldValue::Segment(handle))
    }

    /// Declare a single content segment field only when present.
    pub fn opt_segment(self, name: impl Into<FieldName>, handle: Option<Handle>) -> Self {
        match handle {
            Some(handle) => self.segment(name, handle),
            None => self,
        }
    }

    /// Declare a literal endpoint-generated value.
    pub fn literal(self, name: impl Into<FieldName>, value: Value) -> Self {
        self.push(name.into(), FieldValue::Literal(value))
    }

    /// Declare a literal string field.
    pub fn str(self, name: impl Into<FieldName>, value: impl Into<String>) -> Self {
        self.literal(name, Value::String(value.into()))
    }

    /// Declare a literal integer field.
    pub fn int(self, name: impl Into<FieldName>, value: u32) -> Self {
        self.literal(name, Value::from(value))
    }

    /// Declare a literal boolean field.
    pub fn bool(self, name: impl Into<FieldName>, value: bool) -> Self {
        self.literal(name, Value::Bool(value))
    }

    /// Convert non-empty top-level arrays of objects to spliceable wires; retain
    /// all other values as literals.
    ///
    /// Returns serialization errors from array elements unchanged.
    pub fn from_object(
        object: &serde_json::Map<String, Value>,
    ) -> std::result::Result<Self, serde_json::Error> {
        let mut plan = Self::new();
        for (key, value) in object {
            let name = Cow::Owned(key.clone());
            match value.as_array() {
                Some(elements) if !elements.is_empty() && elements.iter().all(Value::is_object) => {
                    let wires = elements
                        .iter()
                        .map(|element| serde_json::to_vec(element).map(Bytes::from))
                        .collect::<std::result::Result<SmallVec<[Bytes; 1]>, _>>()?;
                    plan = plan.push(name, FieldValue::Wires(wires));
                }
                _ => plan = plan.push(name, FieldValue::Literal(value.clone())),
            }
        }
        Ok(plan)
    }

    /// Replace an existing message-array field with pre-serialized wires while
    /// preserving field order. Empty wire lists leave the field unchanged.
    pub fn splice_message_wires(&mut self, name: &str, wires: SmallVec<[Bytes; 1]>) {
        if wires.is_empty() {
            return;
        }
        if let Self::Fields(fields) = self
            && let Some(slot) = fields.iter_mut().find(|(field, _)| field == name)
        {
            slot.1 = FieldValue::Wires(wires);
        }
    }

    /// Borrow a top-level literal field's value by name.
    pub fn literal_field(&self, name: &str) -> Option<&Value> {
        match self {
            Self::Fields(fields) => fields.iter().find_map(|(field, value)| match value {
                FieldValue::Literal(literal) if field == name => Some(literal),
                _ => None,
            }),
            Self::Raw(_) | Self::Prebuilt(_) => None,
        }
    }

    /// Field names whose value can differ per dispatch (`effective_from_plan`
    /// reads these out of the plan literals). A plan carrying any of them cannot
    /// be collapsed to a single prebuilt body.
    const PER_DISPATCH_LITERALS: [&'static str; 5] = [
        "model",
        "stream",
        "max_tokens",
        "max_completion_tokens",
        "max_output_tokens",
    ];

    /// Collapse a fully-static `Fields` plan into a single [`Prebuilt`](BodyPlan::Prebuilt)
    /// body, serialized once here, so dispatch clones it instead of re-splicing.
    ///
    /// Returns `self` unchanged when collapsing would not be byte-exact: a
    /// streaming-capable endpoint (whose `stream` flag `effective_from_plan` may
    /// toggle per dispatch), a plan carrying any [`PER_DISPATCH_LITERALS`]
    /// (`model`/`max_tokens`/…), or a non-`Fields` plan. The scheduled path that
    /// consumes precomputed plans always dispatches with empty overrides, so a
    /// collapsed body needs no per-dispatch mutation.
    pub fn prebuilt_if_static(self, supports_streaming: bool) -> Self {
        let collapsible = !supports_streaming
            && matches!(&self, Self::Fields(_))
            && Self::PER_DISPATCH_LITERALS
                .iter()
                .all(|field| self.literal_field(field).is_none());
        if !collapsible {
            return self;
        }
        match self.materialize_standalone() {
            Ok(bytes) => Self::Prebuilt(bytes),
            // A body that will not materialize now would fail identically at
            // dispatch; leave the plan so the live path surfaces the error.
            Err(_) => self,
        }
    }

    /// Set a top-level literal field: replace in place if the name already
    /// exists (position preserved), else append — the insertion-order semantics
    /// of `serde_json::Map::insert`, so dispatch overrides fold in byte-for-byte.
    pub fn set_literal(&mut self, name: impl Into<FieldName>, value: Value) {
        if let Self::Fields(fields) = self {
            let name = name.into();
            if let Some(slot) = fields.iter_mut().find(|(field, _)| *field == name) {
                slot.1 = FieldValue::Literal(value);
            } else {
                fields.push((name, FieldValue::Literal(value)));
            }
        }
    }

    /// Fold per-dispatch [`Overrides`] into the plan's literal fields with the
    /// same in-place/append semantics `merge_overrides` applies to a JSON object,
    /// so materialization matches an object merge byte-for-byte.
    pub fn merge_overrides(&mut self, overrides: &Overrides) {
        for (name, value) in overrides.fields() {
            self.set_literal(Cow::Owned(name.clone()), value.clone());
        }
    }

    /// Materialize a plan that references no stored segments (only literals and
    /// inline wires).
    pub fn materialize_standalone(&self) -> Result<Bytes> {
        let store = crate::dataset::segment::InMemorySegmentStore::default();
        JsonBodyMaterializer::materialize(self, &store, &Overrides::new())
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
/// tail (`model`/`max_tokens`/`stream`/…). Content is never re-serialized; only [`Literal`](FieldValue::Literal)
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
            // Already a complete object; splice applies any (rare) override tail
            // and otherwise clones the prebuilt bytes without a store lookup.
            BodyPlan::Prebuilt(bytes) => splice_raw_object(bytes, overrides),
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
                    push_message(&mut body, &wire, element)?;
                }
                body.put_u8(b']');
            }
            FieldValue::Wires(wires) => {
                body.put_u8(b'[');
                for (element, wire) in wires.iter().enumerate() {
                    if element > 0 {
                        body.put_u8(b',');
                    }
                    push_message(&mut body, wire, element)?;
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

/// Append one validated message-object wire as an array element.
fn push_message(body: &mut BytesMut, wire: &[u8], index: usize) -> Result<()> {
    validate_object_slice(wire)
        .map_err(|error| DatasetError::InvalidWire(format!("message at index {index}: {error}")))?;
    body.put_slice(wire);
    Ok(())
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
    fn messages_plan_is_byte_identical_to_message_splice() {
        let mut pool = SegmentPool::new();
        let system = message(&mut pool, None, br#"{"role":"system","content":"S"}"#);
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

        let wires = [
            message_wire(&store, system).unwrap(),
            message_wire(&store, user).unwrap(),
        ];
        let spliced = build_message_body_from_wires(&wires, &overrides).unwrap();

        assert_eq!(planned, spliced);
        assert_eq!(
            planned,
            Bytes::from_static(
                br#"{"messages":[{"role":"system","content":"S"},{"content":"hi","role":"user","x":1}],"model":"m","stream":true}"#
            )
        );
    }

    #[test]
    fn prebuilt_if_static_collapses_static_body_byte_identically() {
        // An image-retrieval-shaped body: one wire array, no per-dispatch tail.
        let images = [
            Bytes::from_static(br#"{"type":"image_url","url":"data:image/png;base64,AA=="}"#),
            Bytes::from_static(br#"{"type":"image_url","url":"data:image/png;base64,BB=="}"#),
        ];
        let plan = BodyPlan::new().wire_array("input", images);
        let baseline = plan.materialize_standalone().unwrap();

        // Non-streaming + no model/stream/max_tokens literal => collapses.
        let collapsed = plan.prebuilt_if_static(false);
        assert!(matches!(collapsed, BodyPlan::Prebuilt(_)));
        // Dispatch (empty overrides) yields byte-identical bytes, now via clone.
        assert_eq!(collapsed.materialize_standalone().unwrap(), baseline);
    }

    #[test]
    fn prebuilt_if_static_leaves_per_dispatch_and_streaming_bodies_alone() {
        // A chat-shaped body carrying a `model` literal must NOT collapse: the
        // effective-request path rewrites it per dispatch.
        let with_model = BodyPlan::new()
            .wire_array(
                "messages",
                [Bytes::from_static(br#"{"role":"user","content":"hi"}"#)],
            )
            .literal("model", Value::String("m".into()));
        assert!(matches!(
            with_model.prebuilt_if_static(false),
            BodyPlan::Fields(_)
        ));

        // A streaming-capable endpoint may toggle `stream` per dispatch, so even a
        // literal-free body stays a live plan.
        let streamable = BodyPlan::new().wire_array(
            "input",
            [Bytes::from_static(br#"{"type":"image_url","url":"x"}"#)],
        );
        assert!(matches!(
            streamable.prebuilt_if_static(true),
            BodyPlan::Fields(_)
        ));
    }

    #[test]
    fn wire_array_splices_identically_to_stored_segments() {
        let mut pool = SegmentPool::new();
        let a = message(&mut pool, None, br#"{"role":"user","content":"one"}"#);
        let b = message(
            &mut pool,
            Some(a),
            br#"{"role":"assistant","content":"two"}"#,
        );
        let store = pool.freeze();
        let mut overrides = Overrides::new();
        overrides.set_model("m");

        let wire_a = message_wire(&store, a).unwrap();
        let wire_b = message_wire(&store, b).unwrap();

        let segment_plan = BodyPlan::new().array("messages", [a, b]);
        let wire_plan = BodyPlan::new().wire_array("messages", [wire_a.clone(), wire_b.clone()]);

        let from_segments =
            JsonBodyMaterializer::materialize(&segment_plan, &store, &overrides).unwrap();
        let from_wires = JsonBodyMaterializer::materialize(&wire_plan, &store, &overrides).unwrap();
        let spliced = build_message_body_from_wires(&[wire_a, wire_b], &overrides).unwrap();

        assert_eq!(from_wires, from_segments);
        assert_eq!(from_wires, spliced);
    }

    #[test]
    fn merged_object_bridge_is_byte_identical_to_to_vec() {
        // Covers messages-array splicing plus scalar, nested-object, string-array,
        // and user extra keys — every top-level shape a formatter emits.
        let object = serde_json::json!({
            "messages": [
                {"role": "system", "content": "S"},
                {"role": "user", "content": "hi"}
            ],
            "model": "m",
            "stream": true,
            "max_completion_tokens": 8,
            "stream_options": {"include_usage": true},
            "input": ["a", "b"],
            "user_extra_key": {"nested": [1, 2, 3]}
        });
        let map = object.as_object().unwrap();
        let bridged = BodyPlan::from_object(map)
            .unwrap()
            .materialize_standalone()
            .unwrap();
        assert_eq!(bridged, Bytes::from(serde_json::to_vec(&object).unwrap()));
    }

    #[test]
    fn plan_merge_overrides_matches_object_insert_then_to_vec() {
        let object = serde_json::json!({
            "messages": [{"role": "user", "content": "q"}],
            "model": "old",
            "stream": true
        });
        let mut overrides = Overrides::new();
        overrides.set_model("new"); // existing key -> in-place
        overrides.set_stream(false); // existing key -> in-place
        overrides.insert("seed", Value::from(7)); // new key -> append

        let mut merged = object.as_object().unwrap().clone();
        for (key, value) in overrides.fields() {
            merged.insert(key.clone(), value.clone());
        }
        let merged_bytes = Bytes::from(serde_json::to_vec(&Value::Object(merged)).unwrap());

        let mut plan = BodyPlan::from_object(object.as_object().unwrap()).unwrap();
        plan.merge_overrides(&overrides);
        assert_eq!(plan.materialize_standalone().unwrap(), merged_bytes);
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
