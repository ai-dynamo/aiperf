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
//! serialization.
//!
//! Materialization is a push: [`emit`] plays a plan into a [`BodyEmitter`], one
//! implementation per wire format, and [`JsonEmitter`] is the JSON one. That is
//! the seam a transport which does not speak JSON uses to consume the plan's
//! structure directly — protobuf-wire endpoints (KServe V2 / Riva) read the same
//! segments through their codec rather than splicing bytes, see
//! `transport::grpc` — instead of assembling JSON only to parse it back.

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

impl LiteralValue {
    /// Bind a value, serializing its wire once.
    pub fn new(value: Value) -> Self {
        // `to_vec` starts at a 128-byte `Vec` and grows by doubling, so its
        // capacity almost never equals its length. `Bytes::from(Vec)` takes the
        // free `into_boxed_slice` path *only* on that equality; otherwise it both
        // allocates a `Shared` and retains the slack — which a cached plan then
        // holds for the whole run. Right-sizing first takes the promotable
        // `Box<[u8]>` path: no allocation here, and `Shared` only if it is cloned.
        let wire = serde_json::to_vec(&value)
            .ok()
            .map(|bytes| Bytes::from(bytes.into_boxed_slice()));
        Self { value, wire }
    }

    /// Borrow the bound value.
    pub fn value(&self) -> &Value {
        &self.value
    }

    /// The pre-serialized wire, absent only for a value that will not serialize.
    pub fn wire(&self) -> Option<&Bytes> {
        self.wire.as_ref()
    }
}

/// Two literals are equal when they bind the same value; the wire is derived.
impl PartialEq for LiteralValue {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}

impl From<Value> for LiteralValue {
    fn from(value: Value) -> Self {
        Self::new(value)
    }
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
/// `exact_len` lets [`materialize_fields`] reserve the finished body in one
/// allocation instead of growing a guessed buffer per dispatch. It counts the
/// enclosing braces, every `"name":` frame, every separating comma, and each
/// value's serialized bytes — but **not** the per-dispatch override tail, which
/// is not part of the program.
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

/// Two programs are equal when they declare the same fields; `exact_len` is a
/// derived cache of the field list, not part of the plan's identity.
impl PartialEq for FieldProgram {
    fn eq(&self, other: &Self) -> bool {
        self.fields == other.fields
    }
}

impl Default for FieldProgram {
    fn default() -> Self {
        Self::new()
    }
}

impl FieldProgram {
    /// The `{`/`}` an empty program still materializes to.
    const BRACE_BYTES: usize = 2;

    /// Start an empty program.
    pub fn new() -> Self {
        Self {
            fields: SmallVec::new(),
            exact_len: Some(Self::BRACE_BYTES),
        }
    }

    /// Borrow the declared fields in order.
    pub fn fields(&self) -> &[(FieldName, FieldValue)] {
        &self.fields
    }

    /// The exact serialized length of this program's body without any
    /// per-dispatch override tail, when it does not depend on a segment store.
    pub fn exact_len(&self) -> Option<usize> {
        self.exact_len
    }

    /// Append a field, extending the hint by the new entry plus its separator.
    fn push(&mut self, name: FieldName, value: FieldValue) {
        let separator = usize::from(!self.fields.is_empty());
        self.exact_len = self
            .exact_len
            .zip(entry_len(&name, &value))
            .map(|(current, entry)| current + separator + entry);
        self.fields.push((name, value));
    }

    /// Replace an existing field's value in place, or append when absent.
    fn set(&mut self, name: FieldName, value: FieldValue) {
        match self.position(&name) {
            Some(index) => self.replace_at(index, value),
            None => self.push(name, value),
        }
    }

    /// Index of a declared field by name.
    fn position(&self, name: &str) -> Option<usize> {
        self.fields.iter().position(|(field, _)| field == name)
    }

    /// Replace one field's value in place, preserving field order.
    ///
    /// The hint moves by the difference between the two values. When the
    /// replaced value's length was unknown (a store-dependent segment) the
    /// difference is unknowable, so the whole hint is rebuilt — that value may
    /// have been the only thing making the program store-dependent.
    fn replace_at(&mut self, index: usize, value: FieldValue) {
        let previous = value_len(&self.fields[index].1);
        let replacement = value_len(&value);
        self.fields[index].1 = value;
        match (previous, replacement) {
            (Some(previous), Some(replacement)) => {
                // `current` folded `previous` in, so the subtraction is in range;
                // checked arithmetic degrades a future invariant break to a
                // dropped hint rather than a wrapped, absurd reservation.
                self.exact_len = self
                    .exact_len
                    .and_then(|current| current.checked_add(replacement))
                    .and_then(|total| total.checked_sub(previous));
            }
            (Some(_), None) => self.exact_len = None,
            (None, _) => self.recompute(),
        }
    }

    /// Rebuild `exact_len` from the current field list.
    fn recompute(&mut self) {
        let separators = self.fields.len().saturating_sub(1);
        self.exact_len = self
            .fields
            .iter()
            .try_fold(Self::BRACE_BYTES + separators, |total, (name, value)| {
                entry_len(name, value).map(|entry| total + entry)
            });
    }
}

/// The bytes one field contributes: `"`, the unescaped name, `":`, and the value.
fn entry_len(name: &FieldName, value: &FieldValue) -> Option<usize> {
    value_len(value).map(|value| 1 + name.len() + 2 + value)
}

/// The serialized bytes one field value contributes, or `None` when the length
/// depends on a segment store this program does not hold.
fn value_len(value: &FieldValue) -> Option<usize> {
    match value {
        // The wire was serialized when the value was bound, so its exact length
        // is free here and the write path splices it rather than re-serializing.
        FieldValue::Literal(literal) => literal.wire().map(Bytes::len),
        // A reserved slot has no serialization, so a program still holding one
        // has no length — materializing it errors rather than producing a body.
        FieldValue::Segment(_) | FieldValue::Segments(_) | FieldValue::Reserved => None,
        FieldValue::Wires(wires) => Some(array_len(wires.iter().map(Bytes::len), wires.len())),
    }
}

/// `[`, the elements, the commas between them, and `]`.
fn array_len(elements: impl Iterator<Item = usize>, count: usize) -> usize {
    2 + elements.sum::<usize>() + count.saturating_sub(1)
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

impl BodyPlan {
    /// Start an empty named-field plan.
    pub fn new() -> Self {
        Self::Fields(FieldProgram::new())
    }

    /// Construct a plan wrapping a complete prebuilt body.
    pub fn raw(handle: Handle) -> Self {
        Self::Raw(handle)
    }

    // Field builders are only reachable off `new()` (a `Fields` plan); a `Raw`
    // plan is a no-op sink since it carries no named fields.
    fn push(mut self, name: FieldName, value: FieldValue) -> Self {
        if let Self::Fields(program) = &mut self {
            program.push(name, value);
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
        self.push(name.into(), FieldValue::Literal(LiteralValue::new(value)))
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
                    // Right-size before `Bytes::from`, exactly as `LiteralValue::new`
                    // does: `to_vec` leaves capacity slack that the `Vec` path would
                    // both retain and pay a `Shared` allocation for. These wires are
                    // held for the run by every cached chat/embeddings plan, so the
                    // slack is retained per message rather than per body.
                    let wires = elements
                        .iter()
                        .map(|element| {
                            serde_json::to_vec(element)
                                .map(|bytes| Bytes::from(bytes.into_boxed_slice()))
                        })
                        .collect::<std::result::Result<SmallVec<[Bytes; 1]>, _>>()?;
                    plan = plan.push(name, FieldValue::Wires(wires));
                }
                _ => plan = plan.push(name, FieldValue::Literal(LiteralValue::new(value.clone()))),
            }
        }
        Ok(plan)
    }

    /// Build a plan from an object, converting each named field to an unfilled
    /// [`Reserved`](FieldValue::Reserved) slot.
    ///
    /// The object's entry for a reserved name exists only to fix that field's
    /// ordinal position; its value is discarded. Every reserved name must be
    /// present — a name the object never declared is an error here rather than
    /// a body silently missing that field, which is what a dialect that forgot
    /// or misspelled its position marker used to dispatch.
    pub fn from_object_reserving(
        object: &serde_json::Map<String, Value>,
        reserved: &[&str],
    ) -> Result<Self> {
        let mut program = match Self::from_object(object)? {
            Self::Fields(program) => program,
            // `from_object` builds only `Fields`; the other plans carry no
            // named fields and so can reserve nothing.
            other => return Ok(other),
        };
        for name in reserved {
            let Some(index) = program.position(name) else {
                return Err(DatasetError::ReservedField(format!(
                    "cannot reserve {name:?}: the payload declares no such field, \
                     so it has no position to hold"
                )));
            };
            program.replace_at(index, FieldValue::Reserved);
        }
        Ok(Self::Fields(program))
    }

    /// Fill a [`Reserved`](FieldValue::Reserved) slot with pre-serialized wires,
    /// in the position the slot was reserved at.
    ///
    /// Errors when the plan reserved no slot under `name`, or when the named
    /// field is not a reserved slot (already filled, or a plain value). That is
    /// a dialect that dropped, misspelled, or reordered its reservation — a code
    /// defect, unreachable for any correct dialect and independent of the
    /// dataset, so failing hard is safe.
    ///
    /// An **empty** `wires` list is deliberately *not* an error, because unlike
    /// the above it is data-dependent. A Responses turn whose recorded output
    /// items are all replay-unsafe (`reasoning`, `web_search_call`, …) lowers to
    /// zero wires, and one such conversation inside an otherwise healthy dataset
    /// must not take the whole run down. The field materializes as `[]` —
    /// byte-identical to what the preceding empty-array placeholder dispatched —
    /// so the request is recorded as the server rejection it has always been and
    /// issuance continues. The warning is what makes that case loud; it fires
    /// once per affected turn, bounded by the failures already being recorded.
    #[must_use = "an unhandled fill failure leaves the slot unfilled and the body unbuildable"]
    pub fn fill_reserved(&mut self, name: &str, wires: SmallVec<[Bytes; 1]>) -> Result<()> {
        let Self::Fields(program) = self else {
            return Err(DatasetError::ReservedField(format!(
                "cannot fill reserved field {name:?}: this plan carries no named fields"
            )));
        };
        let index = program
            .position(name)
            .filter(|index| matches!(program.fields()[*index], (_, FieldValue::Reserved)));
        let Some(index) = index else {
            return Err(DatasetError::ReservedField(format!(
                "cannot fill {name:?}: the plan reserved no slot under that name"
            )));
        };
        if wires.is_empty() {
            tracing::warn!(
                field = name,
                "message array lowered to zero elements; dispatching an empty array, \
                 which the endpoint will reject"
            );
        }
        program.replace_at(index, FieldValue::Wires(wires));
        Ok(())
    }

    /// Borrow a top-level literal field's value by name.
    pub fn literal_field(&self, name: &str) -> Option<&Value> {
        match self {
            Self::Fields(program) => program
                .fields
                .iter()
                .find_map(|(field, value)| match value {
                    FieldValue::Literal(literal) if field == name => Some(literal.value()),
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
        if let Self::Fields(program) = self {
            program.set(name.into(), FieldValue::Literal(LiteralValue::new(value)));
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

/// What the driver knows about a body's finished length before it is emitted.
///
/// A byte-buffer emitter reserves from this and allocates exactly once; an
/// emitter that builds a structured value ignores it entirely.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SizeHint {
    /// The exact serialized byte length of the finished body, override tail
    /// included. A JSON emitter that reserves this never reallocates and
    /// retains no slack, and can assert the finished buffer against it.
    Exact(usize),
    /// A rough guess, used when a store-dependent
    /// [`Segment`](FieldValue::Segment) value makes the exact length
    /// unknowable without resolving the store.
    Estimated(usize),
}

impl SizeHint {
    /// The byte count to reserve, exact or estimated.
    pub fn capacity(self) -> usize {
        match self {
            Self::Exact(bytes) | Self::Estimated(bytes) => bytes,
        }
    }

    /// The exact length when known, for a post-write assertion.
    pub fn exact(self) -> Option<usize> {
        match self {
            Self::Exact(bytes) => Some(bytes),
            Self::Estimated(_) => None,
        }
    }
}

/// A wire-format sink a [`BodyPlan`] is played into.
///
/// One implementation per wire format. [`emit`] drives the plan through it in
/// field order, pushing each value exactly once; nothing intermediate is
/// buffered, so a transport that does not speak JSON never assembles JSON it
/// will not send. Statically dispatched — no vtable on the dispatch path.
///
/// The call protocol is one of two shapes, and an implementation may rely on
/// that:
///
/// - **Named fields.** [`begin`](BodyEmitter::begin), then per declared field a
///   [`field`](BodyEmitter::field) naming it followed by exactly one of
///   [`literal`](BodyEmitter::literal), [`wire`](BodyEmitter::wire), or
///   [`array`](BodyEmitter::array); then [`overrides`](BodyEmitter::overrides);
///   then [`finish`](BodyEmitter::finish).
/// - **Whole body.** [`whole`](BodyEmitter::whole) alone, then
///   [`finish`](BodyEmitter::finish). No `begin`, no fields.
pub trait BodyEmitter {
    /// What this emitter produces — request bytes, a decoded value, protobuf
    /// message parts, multipart form parts.
    type Output;

    /// Start a named-field body. Called once, before any field.
    fn begin(&mut self, size: SizeHint) -> Result<()>;

    /// Open the field `name`; exactly one value call follows.
    fn field(&mut self, name: &str) -> Result<()>;

    /// An endpoint-generated value bound to the open field. Both forms are
    /// offered because they cost different things to different emitters: a JSON
    /// emitter splices [`LiteralValue::wire`] verbatim, while an emitter
    /// building a structured value clones [`LiteralValue::value`] and parses
    /// nothing.
    fn literal(&mut self, literal: &LiteralValue) -> Result<()>;

    /// One complete pre-serialized JSON value occupying the open field.
    fn wire(&mut self, wire: &Bytes) -> Result<()>;

    /// An ordered array of pre-serialized object wires occupying the open
    /// field. The iterator resolves store handles lazily, so a handle-addressed
    /// plan never materializes an intermediate `Vec<Bytes>`, and it borrows
    /// inline wires rather than bumping a refcount per message per dispatch.
    fn array<'wire>(
        &mut self,
        elements: &mut dyn Iterator<Item = Result<Cow<'wire, Bytes>>>,
    ) -> Result<()>;

    /// A complete authored object occupying the whole body
    /// ([`BodyPlan::Raw`], [`BodyPlan::Prebuilt`]).
    ///
    /// `overrides` is passed here rather than folded into
    /// [`overrides`](BodyEmitter::overrides) because raw override semantics —
    /// always-append tail, authored whitespace preserved, a duplicate key
    /// permitted — are not the in-place-or-append semantics of a named-field
    /// program. Collapsing the two rewrites authored bytes.
    fn whole(&mut self, wire: &Bytes, overrides: &Overrides) -> Result<()>;

    /// Per-dispatch top-level fields, appended after the program's own fields.
    ///
    /// `inner_wire` is `overrides.inner_bytes()` serialized once by the driver
    /// (the driver needs its length for the [`SizeHint`] regardless): the
    /// enclosing braces are stripped, so it splices directly into an open
    /// object. A JSON emitter appends it; an emitter that builds a structured
    /// value reads `overrides.fields()` and ignores it.
    fn overrides(&mut self, overrides: &Overrides, inner_wire: &[u8]) -> Result<()>;

    /// Complete the body.
    fn finish(self) -> Result<Self::Output>;
}

/// Play `plan` into `emitter`, resolving segment handles against `store` and
/// applying the per-dispatch `overrides`.
pub fn emit<E: BodyEmitter, S: SegmentStore + ?Sized>(
    plan: &BodyPlan,
    store: &S,
    overrides: &Overrides,
    mut emitter: E,
) -> Result<E::Output> {
    match plan {
        BodyPlan::Raw(handle) => match store.get(*handle)? {
            Payload::Raw { wire } => emitter.whole(wire, overrides)?,
            payload => {
                return Err(DatasetError::PayloadKind {
                    handle: *handle,
                    expected: "raw",
                    actual: payload.kind_name(),
                });
            }
        },
        // Already a complete object; the emitter applies any (rare) override
        // tail and otherwise takes the prebuilt bytes without a store lookup.
        BodyPlan::Prebuilt(bytes) => emitter.whole(bytes, overrides)?,
        BodyPlan::Fields(program) => emit_fields(program, store, overrides, &mut emitter)?,
    }
    emitter.finish()
}

/// Play one named-field program, field by field.
fn emit_fields<E: BodyEmitter, S: SegmentStore + ?Sized>(
    program: &FieldProgram,
    store: &S,
    overrides: &Overrides,
    emitter: &mut E,
) -> Result<()> {
    let fields = program.fields();
    let override_inner = overrides.inner_bytes()?;
    // The tail is per-dispatch, so the program's hint does not cover it.
    let tail = if override_inner.is_empty() {
        0
    } else {
        override_inner.len() + usize::from(!fields.is_empty())
    };
    // An exact hint makes the body one allocation with no slack; without one
    // (a store-dependent segment value) fall back to a rough guess.
    let size = match program.exact_len() {
        Some(exact) => SizeHint::Exact(exact + tail),
        None => SizeHint::Estimated(fields.len() * 32 + override_inner.len() + 2),
    };
    emitter.begin(size)?;
    for (name, value) in fields {
        emitter.field(name)?;
        match value {
            FieldValue::Literal(literal) => emitter.literal(literal)?,
            FieldValue::Segment(handle) => emitter.wire(&segment_field_wire(store, *handle)?)?,
            FieldValue::Segments(handles) => {
                let mut elements = handles
                    .iter()
                    .map(|handle| message_wire(store, *handle).map(Cow::Owned));
                emitter.array(&mut elements)?;
            }
            FieldValue::Wires(wires) => {
                let mut elements = wires.iter().map(|wire| Ok(Cow::Borrowed(wire)));
                emitter.array(&mut elements)?;
            }
            // Whatever the emitter has written so far is dropped with the
            // error, so an unfilled slot yields no body at all rather than a
            // body missing the field the endpoint meant to put here.
            FieldValue::Reserved => {
                return Err(DatasetError::ReservedField(format!(
                    "field {name:?} was reserved but never filled; the endpoint must \
                     call fill_reserved before the body is materialized"
                )));
            }
        }
    }
    emitter.overrides(overrides, &override_inner)
}

/// The JSON [`BodyEmitter`]: concatenates literal bytes, pre-serialized segment
/// bytes, and message wires into the one contiguous request body, then appends
/// the small per-dispatch override tail. Content is never re-serialized.
#[derive(Debug, Default)]
pub struct JsonEmitter {
    buf: BytesMut,
    /// Set by [`BodyEmitter::whole`] instead of `buf`, so a whole-body plan with
    /// no overrides hands back the authored bytes as a refcount clone rather
    /// than copying them through the buffer. Mutually exclusive with the
    /// named-field path by the call protocol.
    whole: Option<Bytes>,
    /// The exact finished length when the program knew it, checked against the
    /// buffer before it is frozen.
    expected: Option<usize>,
    fields_written: usize,
}

impl BodyEmitter for JsonEmitter {
    type Output = Bytes;

    fn begin(&mut self, size: SizeHint) -> Result<()> {
        self.buf = BytesMut::with_capacity(size.capacity());
        self.expected = size.exact();
        self.buf.put_u8(b'{');
        Ok(())
    }

    fn field(&mut self, name: &str) -> Result<()> {
        if self.fields_written > 0 {
            self.buf.put_u8(b',');
        }
        self.fields_written += 1;
        self.buf.put_u8(b'"');
        self.buf.put_slice(name.as_bytes());
        self.buf.put_slice(b"\":");
        Ok(())
    }

    fn literal(&mut self, literal: &LiteralValue) -> Result<()> {
        // Spliced from the wire bound with the value; the fallback re-attempts
        // the serialization that failed at bind time so the error surfaces here.
        match literal.wire() {
            Some(wire) => self.buf.put_slice(wire),
            None => serde_json::to_writer((&mut self.buf).writer(), literal.value())?,
        }
        Ok(())
    }

    fn wire(&mut self, wire: &Bytes) -> Result<()> {
        self.buf.put_slice(wire);
        Ok(())
    }

    fn array<'wire>(
        &mut self,
        elements: &mut dyn Iterator<Item = Result<Cow<'wire, Bytes>>>,
    ) -> Result<()> {
        self.buf.put_u8(b'[');
        for (index, element) in elements.enumerate() {
            if index > 0 {
                self.buf.put_u8(b',');
            }
            let wire = element?;
            validate_object_slice(&wire).map_err(|error| {
                DatasetError::InvalidWire(format!("message at index {index}: {error}"))
            })?;
            self.buf.put_slice(&wire);
        }
        self.buf.put_u8(b']');
        Ok(())
    }

    fn whole(&mut self, wire: &Bytes, overrides: &Overrides) -> Result<()> {
        self.whole = Some(splice_raw_object(wire, overrides)?);
        Ok(())
    }

    fn overrides(&mut self, _overrides: &Overrides, inner_wire: &[u8]) -> Result<()> {
        if !inner_wire.is_empty() {
            if self.fields_written > 0 {
                self.buf.put_u8(b',');
            }
            self.buf.put_slice(inner_wire);
        }
        Ok(())
    }

    fn finish(mut self) -> Result<Bytes> {
        if let Some(whole) = self.whole {
            return Ok(whole);
        }
        self.buf.put_u8(b'}');
        if let Some(expected) = self.expected {
            debug_assert_eq!(
                expected,
                self.buf.len(),
                "exact_len drifted; a mutator failed to update it"
            );
        }
        Ok(self.buf.freeze())
    }
}

/// The shared JSON splicer: `BodyPlan` + dispatch [`Overrides`] → one `Bytes`.
///
/// A thin driver over [`JsonEmitter`], retained as the name every materializing
/// caller already uses.
pub struct JsonBodyMaterializer;

impl JsonBodyMaterializer {
    /// Materialize a plan against a store into the single request body buffer.
    pub fn materialize<S: SegmentStore + ?Sized>(
        plan: &BodyPlan,
        store: &S,
        overrides: &Overrides,
    ) -> Result<Bytes> {
        emit(plan, store, overrides, JsonEmitter::default())
    }
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
    use crate::dataset::segment::{InMemorySegmentStore, SegmentPool};

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

    fn hint(plan: &BodyPlan) -> Option<usize> {
        match plan {
            BodyPlan::Fields(program) => program.exact_len(),
            BodyPlan::Raw(_) | BodyPlan::Prebuilt(_) => None,
        }
    }

    #[test]
    fn exact_len_matches_materialized_length() {
        // Literals only, covering escaping and number formatting.
        let literals = BodyPlan::new()
            .str("model", "a \"quoted\" \u{2028} model")
            .int("max_tokens", 1024)
            .bool("stream", false)
            .literal("stream_options", serde_json::json!({"include_usage": true}));
        assert_eq!(
            hint(&literals),
            Some(literals.materialize_standalone().unwrap().len())
        );

        // Literals plus a spliced wire array.
        let with_wires = BodyPlan::new()
            .wire_array(
                "messages",
                [
                    Bytes::from_static(br#"{"role":"system","content":"S"}"#),
                    Bytes::from_static(br#"{"role":"user","content":"hi"}"#),
                ],
            )
            .str("model", "m")
            .bool("stream", true);
        assert_eq!(
            hint(&with_wires),
            Some(with_wires.materialize_standalone().unwrap().len())
        );

        // A folded-in override tail: in-place replacement and append together.
        let mut overrides = Overrides::new();
        overrides.set_model("a-much-longer-model-name");
        overrides.set_stream(false);
        overrides.insert("seed", Value::from(7));
        let mut merged = with_wires.clone();
        merged.merge_overrides(&overrides);
        assert_eq!(
            hint(&merged),
            Some(merged.materialize_standalone().unwrap().len())
        );

        // The hint excludes the per-dispatch tail, so a dispatch carrying an
        // unmerged override set materializes to more than the program's length.
        let dispatched = JsonBodyMaterializer::materialize(
            &with_wires,
            &InMemorySegmentStore::default(),
            &overrides,
        )
        .unwrap();
        assert!(hint(&with_wires).unwrap() < dispatched.len());
    }

    #[test]
    fn literal_wire_is_bound_once_and_matches_the_serializer() {
        // The wire is spliced verbatim at materialization, so it must be exactly
        // what `to_writer` would have emitted — escaping, float and large-integer
        // formatting included.
        for value in [
            Value::String("esc \"q\" \\ \n \u{2028} \u{1f600}".into()),
            Value::from(1.0e-7_f64),
            Value::from(u64::MAX),
            serde_json::json!({"include_usage": true, "nested": [1, 2, 3]}),
            serde_json::json!((0..4096).collect::<Vec<u32>>()),
        ] {
            let literal = LiteralValue::new(value.clone());
            assert_eq!(
                literal.wire().map(|wire| wire.as_ref()),
                Some(serde_json::to_vec(&value).unwrap().as_slice()),
                "cached literal wire drifted from the serializer"
            );
        }

        // And the spliced body matches a whole-object serialization.
        let object = serde_json::json!({
            "token_ids": (0..2048).collect::<Vec<u32>>(),
            "sampling_params": {"max_tokens": 16},
            "stream": false
        });
        let plan = BodyPlan::from_object(object.as_object().unwrap()).unwrap();
        assert_eq!(
            plan.materialize_standalone().unwrap(),
            Bytes::from(serde_json::to_vec(&object).unwrap())
        );
    }

    #[test]
    fn literal_wire_retains_no_capacity_slack() {
        // `serde_json::to_vec` hands back a 128-byte-minimum, doubling-grown Vec.
        // Handing that to `Bytes::from` directly would retain the slack for the
        // life of a cached plan and allocate a `Shared` eagerly; right-sizing
        // first takes the promotable path. `BytesMut::from` reclaims a unique
        // buffer without copying, so its capacity reports what was retained.
        for value in [
            Value::Bool(false),
            Value::String("m".into()),
            serde_json::json!((0..4096).collect::<Vec<u32>>()),
        ] {
            let expected = serde_json::to_vec(&value).unwrap().len();
            let wire = LiteralValue::new(value).wire.expect("value serializes");
            assert_eq!(wire.len(), expected);
            assert_eq!(
                BytesMut::from(wire).capacity(),
                expected,
                "literal wire retained capacity slack; \
                 was `Bytes::from(Vec)` restored in place of the boxed slice?"
            );
        }

        // The probe must be able to see slack, or the assertions above are
        // vacuous: the un-right-sized shape has to report the Vec's capacity.
        let slack = serde_json::to_vec(&Value::Bool(false)).unwrap();
        let retained = slack.capacity();
        assert!(retained > slack.len(), "to_vec no longer over-allocates");
        assert_eq!(BytesMut::from(Bytes::from(slack)).capacity(), retained);
    }

    #[test]
    fn exact_len_is_absent_for_store_dependent_values() {
        let mut pool = SegmentPool::new();
        let msg = message(&mut pool, None, br#"{"role":"user","content":"hi"}"#);
        let store = pool.freeze();

        let mut plan = BodyPlan::new().array("messages", [msg]).str("model", "m");
        assert_eq!(hint(&plan), None);
        // Materialization still succeeds on the fallback capacity path.
        assert!(JsonBodyMaterializer::materialize(&plan, &store, &Overrides::new()).is_ok());

        // Replacing the segment array with wires makes the length knowable again.
        let BodyPlan::Fields(program) = &mut plan else {
            panic!("field plan");
        };
        let index = program.position("messages").unwrap();
        program.replace_at(
            index,
            FieldValue::Wires(smallvec::smallvec![Bytes::from_static(
                br#"{"role":"user","content":"hi"}"#
            )]),
        );
        assert_eq!(
            hint(&plan),
            Some(plan.materialize_standalone().unwrap().len())
        );
    }

    /// The payload key a dialect inserts purely to hold a field's position.
    fn reserving_payload() -> serde_json::Map<String, Value> {
        let mut payload = serde_json::Map::new();
        payload.insert("messages".into(), Value::Null);
        payload.insert("model".into(), Value::String("m".into()));
        payload
    }

    fn wires(count: usize) -> SmallVec<[Bytes; 1]> {
        (0..count)
            .map(|_| Bytes::from_static(br#"{"role":"user","content":"hi"}"#))
            .collect()
    }

    #[test]
    fn unfilled_reserved_field_is_an_error_not_an_empty_array() {
        let plan = BodyPlan::from_object_reserving(&reserving_payload(), &["messages"]).unwrap();
        assert!(
            plan.materialize_standalone().is_err(),
            "an unfilled Reserved field must fail loudly, not emit []"
        );
        // And it must not be paying for a capacity hint it cannot honor.
        assert_eq!(hint(&plan), None);
    }

    #[test]
    fn filling_an_absent_field_is_an_error() {
        let mut plan = BodyPlan::from_object_reserving(&serde_json::Map::new(), &[]).unwrap();
        assert!(
            plan.fill_reserved("messages", wires(1)).is_err(),
            "filling a field the plan does not declare must error"
        );

        // Reserving a name the payload never declared fails at construction —
        // the failure mode of a dialect that forgot or misspelled its marker.
        assert!(BodyPlan::from_object_reserving(&reserving_payload(), &["mesages"]).is_err());

        // A declared-but-not-reserved field is not a fill target either: that is
        // a dialect that dropped the reservation but kept the fill.
        let mut unreserved = BodyPlan::from_object_reserving(&reserving_payload(), &[]).unwrap();
        assert!(unreserved.fill_reserved("messages", wires(1)).is_err());

        // Nor is a slot that was already filled.
        let mut filled =
            BodyPlan::from_object_reserving(&reserving_payload(), &["messages"]).unwrap();
        filled.fill_reserved("messages", wires(1)).unwrap();
        assert!(filled.fill_reserved("messages", wires(1)).is_err());
    }

    #[test]
    fn empty_fill_still_dispatches_an_empty_array_rather_than_failing_the_run() {
        // Data-dependent, unlike a missing reservation: a Responses turn whose
        // recorded output is entirely replay-unsafe lowers to zero wires. The old
        // placeholder shipped `[]` and the server rejected that one request while
        // the run continued; erroring here would instead `?` out of
        // `Workload::execute` and kill the phase, so one bad conversation in a
        // healthy dataset would take the whole benchmark down.
        let mut plan =
            BodyPlan::from_object_reserving(&reserving_payload(), &["messages"]).unwrap();
        plan.fill_reserved("messages", SmallVec::new()).unwrap();
        assert_eq!(
            plan.materialize_standalone().unwrap(),
            Bytes::from_static(br#"{"messages":[],"model":"m"}"#),
            "an empty fill must stay byte-identical to the placeholder it replaced"
        );
    }

    #[test]
    fn filled_reservation_is_byte_identical_to_the_placeholder_splice() {
        // The reserved slot must hold the field's ordinal position exactly: a
        // slot that drifted to the end would still be valid JSON and still
        // carry every message, so only the bytes catch it.
        let message = br#"{"role":"user","content":"hi"}"#;
        let expected = Bytes::from_static(
            br#"{"messages":[{"role":"user","content":"hi"}],"model":"m","stream":true}"#,
        );

        let mut payload = serde_json::Map::new();
        payload.insert("messages".into(), Value::Null);
        payload.insert("model".into(), Value::String("m".into()));
        payload.insert("stream".into(), Value::Bool(true));
        let mut plan = BodyPlan::from_object_reserving(&payload, &["messages"]).unwrap();
        plan.fill_reserved("messages", smallvec::smallvec![Bytes::from_static(message)])
            .unwrap();

        let body = plan.materialize_standalone().unwrap();
        assert_eq!(body, expected);
        // Filling restores an exact hint: the slot's length became knowable.
        assert_eq!(hint(&plan), Some(body.len()));
    }

    #[test]
    fn non_spliceable_segment_domains_are_rejected() {
        let mut pool = SegmentPool::new();
        let tokens = pool.intern_token_ids(None, [1_u32, 2]).unwrap();
        let store = pool.freeze();
        let plan = BodyPlan::new().segment("field", tokens);
        assert!(JsonBodyMaterializer::materialize(&plan, &store, &Overrides::new()).is_err());
    }

    /// The pre-emitter materializer, frozen verbatim as the other side of the
    /// byte-identity differential below.
    ///
    /// Do not evolve this to track new behavior: a divergence from it is
    /// precisely what the differential exists to report. A new
    /// [`FieldValue`] variant makes its `match` non-exhaustive, which is the
    /// intended prompt to decide deliberately what the reference should say.
    fn reference_materialize<S: SegmentStore + ?Sized>(
        plan: &BodyPlan,
        store: &S,
        overrides: &Overrides,
    ) -> Result<Bytes> {
        let program = match plan {
            BodyPlan::Raw(handle) => {
                return match store.get(*handle)? {
                    Payload::Raw { wire } => splice_raw_object(wire, overrides),
                    payload => Err(DatasetError::PayloadKind {
                        handle: *handle,
                        expected: "raw",
                        actual: payload.kind_name(),
                    }),
                };
            }
            BodyPlan::Prebuilt(bytes) => return splice_raw_object(bytes, overrides),
            BodyPlan::Fields(program) => program,
        };
        let fields = program.fields();
        let override_inner = overrides.inner_bytes()?;
        let tail = if override_inner.is_empty() {
            0
        } else {
            override_inner.len() + usize::from(!fields.is_empty())
        };
        let capacity = match program.exact_len() {
            Some(exact) => exact + tail,
            None => fields.len() * 32 + override_inner.len() + 2,
        };
        let mut body = BytesMut::with_capacity(capacity);
        body.put_u8(b'{');
        for (index, (name, value)) in fields.iter().enumerate() {
            if index > 0 {
                body.put_u8(b',');
            }
            body.put_u8(b'"');
            body.put_slice(name.as_bytes());
            body.put_slice(b"\":");
            match value {
                FieldValue::Literal(literal) => match literal.wire() {
                    Some(wire) => body.put_slice(wire),
                    None => serde_json::to_writer((&mut body).writer(), literal.value())?,
                },
                FieldValue::Segment(handle) => body.put_slice(&segment_field_wire(store, *handle)?),
                FieldValue::Segments(handles) => {
                    body.put_u8(b'[');
                    for (element, handle) in handles.iter().enumerate() {
                        if element > 0 {
                            body.put_u8(b',');
                        }
                        let wire = message_wire(store, *handle)?;
                        reference_push_message(&mut body, &wire, element)?;
                    }
                    body.put_u8(b']');
                }
                FieldValue::Wires(wires) => {
                    body.put_u8(b'[');
                    for (element, wire) in wires.iter().enumerate() {
                        if element > 0 {
                            body.put_u8(b',');
                        }
                        reference_push_message(&mut body, wire, element)?;
                    }
                    body.put_u8(b']');
                }
                FieldValue::Reserved => {
                    return Err(DatasetError::ReservedField(format!(
                        "field {name:?} was reserved but never filled; the endpoint must \
                         call fill_reserved before the body is materialized"
                    )));
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

    fn reference_push_message(body: &mut BytesMut, wire: &[u8], index: usize) -> Result<()> {
        validate_object_slice(wire).map_err(|error| {
            DatasetError::InvalidWire(format!("message at index {index}: {error}"))
        })?;
        body.put_slice(wire);
        Ok(())
    }

    #[test]
    fn emitter_is_byte_identical_to_the_pre_emitter_materializer() {
        let mut pool = SegmentPool::new();
        let system = message(&mut pool, None, br#"{"role":"system","content":"S"}"#);
        let user = message(
            &mut pool,
            Some(system),
            br#"{"content":"hi","role":"user","x":1}"#,
        );
        let tools = pool
            .intern_raw(None, Bytes::from_static(br#"[{"type":"function"}]"#))
            .unwrap();
        let raw = pool
            .intern_raw(None, Bytes::from_static(b" \t{\"z\":1, \"messages\":[]}\n"))
            .unwrap();
        let tokens = pool.intern_token_ids(None, [1_u32, 2]).unwrap();
        let store = pool.freeze();

        let wire = |bytes: &'static [u8]| Bytes::from_static(bytes);
        let plans: Vec<(&str, BodyPlan)> = vec![
            ("empty", BodyPlan::new()),
            (
                "literals only",
                BodyPlan::new()
                    .str("model", "a \"quoted\" \u{2028} model")
                    .int("max_tokens", 1024)
                    .bool("stream", false)
                    .literal("stream_options", serde_json::json!({"include_usage": true})),
            ),
            (
                "wire array plus literals",
                BodyPlan::new()
                    .wire_array(
                        "messages",
                        [
                            wire(br#"{"role":"system","content":"S"}"#),
                            wire(br#"{"role":"user","content":"hi"}"#),
                        ],
                    )
                    .str("model", "m")
                    .bool("stream", true),
            ),
            (
                "empty wire array",
                BodyPlan::new().wire_array("messages", []).str("model", "m"),
            ),
            (
                "stored segments",
                BodyPlan::new()
                    .array("messages", [system, user])
                    .str("model", "m"),
            ),
            (
                "mixed segment, array, and literals",
                BodyPlan::new()
                    .str("model", "gpt")
                    .array("messages", [user])
                    .segment("tools", tools)
                    .int("max_tokens", 7)
                    .bool("stream", false),
            ),
            ("raw", BodyPlan::raw(raw)),
            (
                "prebuilt",
                BodyPlan::new()
                    .wire_array("input", [wire(br#"{"type":"image_url","url":"x"}"#)])
                    .prebuilt_if_static(false),
            ),
            (
                "unfilled reserved slot",
                BodyPlan::from_object_reserving(&reserving_payload(), &["messages"]).unwrap(),
            ),
            (
                "non-spliceable segment",
                BodyPlan::new().segment("f", tokens),
            ),
        ];

        let mut tail = Overrides::new();
        tail.set_model("a-much-longer-model-name");
        tail.set_stream(false);
        tail.insert("seed", Value::from(7));

        for (name, plan) in &plans {
            for (variant, overrides) in [("no overrides", &Overrides::new()), ("tail", &tail)] {
                let expected = reference_materialize(plan, &store, overrides);
                let actual = JsonBodyMaterializer::materialize(plan, &store, overrides);
                match (expected, actual) {
                    (Ok(expected), Ok(actual)) => assert_eq!(
                        actual, expected,
                        "emitter diverged from the reference for {name} / {variant}"
                    ),
                    (Err(_), Err(_)) => {}
                    (expected, actual) => panic!(
                        "emitter and reference disagreed on failure for {name} / {variant}: \
                         reference={expected:?} emitter={actual:?}"
                    ),
                }
            }
        }
    }

    #[test]
    fn raw_and_prebuilt_bodies_are_handed_over_without_copying() {
        // The whole-body plans exist to make dispatch a refcount bump; routing
        // them through the emitter's buffer would still be byte-identical, so
        // only pointer identity catches the regression.
        let authored = Bytes::from_static(br#"{"messages":[],"z":1}"#);
        let mut pool = SegmentPool::new();
        let raw = pool.intern_raw(None, authored.clone()).unwrap();
        let store = pool.freeze();

        let from_raw =
            JsonBodyMaterializer::materialize(&BodyPlan::raw(raw), &store, &Overrides::new())
                .unwrap();
        assert_eq!(from_raw.as_ptr(), authored.as_ptr());

        let prebuilt = BodyPlan::Prebuilt(authored.clone());
        let from_prebuilt =
            JsonBodyMaterializer::materialize(&prebuilt, &store, &Overrides::new()).unwrap();
        assert_eq!(from_prebuilt.as_ptr(), authored.as_ptr());
    }

    #[test]
    fn exact_hint_reserves_the_body_without_slack() {
        // The emitter's whole reason for taking a `SizeHint::Exact` is one
        // allocation with nothing retained; `BytesMut::from` reclaims a unique
        // buffer without copying, so its capacity reports what was retained.
        let plan = BodyPlan::new()
            .wire_array(
                "messages",
                [Bytes::from_static(br#"{"role":"user","content":"hi"}"#)],
            )
            .str("model", "m")
            .bool("stream", true);
        let body = plan.materialize_standalone().unwrap();
        assert_eq!(BytesMut::from(body.clone()).capacity(), body.len());
    }
}
