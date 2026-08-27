// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Zero-parse request-body and dynamic-splice materialization.
//!
//! Static messages are pre-serialized at composition time. The hot path clones
//! their ref-counted [`Bytes`] handles and concatenates them into a request body;
//! it never decodes and re-encodes those messages. Raw request bodies are returned
//! byte-identically when no override is present. With overrides, only the small
//! override tail is serialized and inserted immediately before the raw object's
//! closing brace, so authored bytes and key order remain untouched.

use std::sync::Arc;

use bytes::{BufMut, Bytes, BytesMut};
use serde_json::{Map, Value};

use crate::dataset::error::{DatasetError, Result};
use crate::dataset::segment::{Handle, Payload, SegmentStore};

const MESSAGE_HEAD: &[u8] = b"{\"messages\":[";

/// Per-dispatch top-level request fields spliced after the static message array.
///
/// Boundary-owned: defined in `aiperf_core::endpoint` because endpoint
/// formatters author the override tail across the plugin boundary.
pub use aiperf_core::endpoint::Overrides;

/// One instruction in a prompt assembly program.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AssemblyItem {
    /// Append one static pre-serialized message from the segment store.
    Segment(Handle),
    /// Append the captured messages currently bound to a dynamic channel key.
    Splice(String),
}

/// Dynamic message source used by graph and multi-turn assembly.
pub trait MessageSpliceResolver {
    /// Resolve a key to ordered pre-serialized message objects.
    fn resolve(&self, key: &str) -> Result<Vec<Bytes>>;
}

/// Shared segment-store materializer for static and dynamic prompt programs.
#[derive(Clone)]
pub struct SegmentItemsMaterializer {
    store: Arc<dyn SegmentStore>,
}

impl SegmentItemsMaterializer {
    /// Bind a materializer to a frozen shared store.
    pub fn new(store: Arc<dyn SegmentStore>) -> Self {
        Self { store }
    }

    /// Borrow the backing store.
    pub fn store(&self) -> &Arc<dyn SegmentStore> {
        &self.store
    }

    /// Resolve an assembly program into ordered message-wire slices.
    pub fn materialize_messages(
        &self,
        items: &[AssemblyItem],
        splices: &dyn MessageSpliceResolver,
    ) -> Result<Vec<Bytes>> {
        let mut messages = Vec::with_capacity(items.len());
        for item in items {
            match item {
                AssemblyItem::Segment(handle) => {
                    messages.push(message_wire(self.store.as_ref(), *handle)?);
                }
                AssemblyItem::Splice(key) => messages.extend(splices.resolve(key)?),
            }
        }
        Ok(messages)
    }

    /// Resolve and build the complete request body.
    pub fn build(
        &self,
        items: &[AssemblyItem],
        splices: &dyn MessageSpliceResolver,
        overrides: &Overrides,
    ) -> Result<Bytes> {
        let messages = self.materialize_messages(items, splices)?;
        build_message_body_from_wires(&messages, overrides)
    }
}

/// Build a body from static handles through the [`SegmentStore`] trait method.
pub(crate) fn build_body_from_handles<S: SegmentStore + ?Sized>(
    store: &S,
    handles: &[Handle],
    overrides: &Overrides,
) -> Result<Bytes> {
    if let [handle] = handles
        && let Payload::Raw { wire } = store.get(*handle)?
    {
        return splice_raw_object(wire, overrides);
    }

    let messages = handles
        .iter()
        .map(|handle| message_wire(store, *handle))
        .collect::<Result<Vec<_>>>()?;
    build_message_body_from_wires(&messages, overrides)
}

pub(crate) fn message_wire<S: SegmentStore + ?Sized>(store: &S, handle: Handle) -> Result<Bytes> {
    match store.get(handle)? {
        Payload::Message { wire, .. } => Ok(wire.clone()),
        payload => Err(DatasetError::PayloadKind {
            handle,
            expected: "message",
            actual: payload.kind_name(),
        }),
    }
}

/// Build a request body from already-serialized message objects.
///
/// This is the graph sink fast path: materialization clones static/dynamic
/// message slices, then the sink adds its model/stream/token overrides without
/// decoding any message.
pub fn build_message_body_from_wires(messages: &[Bytes], overrides: &Overrides) -> Result<Bytes> {
    let override_inner = overrides.inner_bytes()?;
    Ok(build_message_body_from_wire_parts(
        messages,
        &override_inner,
    ))
}

/// Assemble a request body from message wires plus a **pre-serialized** override
/// tail — the override object's inner bytes with the enclosing braces stripped
/// (see [`Overrides::inner_bytes`]). Byte-identical to
/// [`build_message_body_from_wires`] for the same inputs.
///
/// This is the hottest allocation on the graph dispatch path, so the override
/// tail is serialized once by the caller (per distinct value) and reused here
/// rather than re-serialized per request. There is **no validation on this
/// path**: static message wires are produced by the dataset composer
/// serializing parsed messages, and dynamic splices are replies serialized from
/// a typed value, so every wire reaching here is a well-formed JSON object by
/// construction. Re-scanning each slice per dispatch would be pure overhead.
pub fn build_message_body_from_wire_parts(messages: &[Bytes], override_inner: &[u8]) -> Bytes {
    let message_bytes = messages.iter().map(Bytes::len).sum::<usize>();
    let commas = messages.len().saturating_sub(1);
    let tail = if override_inner.is_empty() { 2 } else { 3 };
    let mut body = BytesMut::with_capacity(
        MESSAGE_HEAD.len() + message_bytes + commas + tail + override_inner.len(),
    );
    body.put_slice(MESSAGE_HEAD);
    for (index, message) in messages.iter().enumerate() {
        if index > 0 {
            body.put_u8(b',');
        }
        body.put_slice(message);
    }
    body.put_u8(b']');
    if !override_inner.is_empty() {
        body.put_u8(b',');
        body.put_slice(override_inner);
    }
    body.put_u8(b'}');
    body.freeze()
}

pub(crate) fn splice_raw_object(wire: &Bytes, overrides: &Overrides) -> Result<Bytes> {
    validate_object_slice(wire).map_err(DatasetError::InvalidWire)?;
    if overrides.is_empty() {
        return Ok(wire.clone());
    }
    let override_inner = overrides.inner_bytes()?;
    let first = wire
        .iter()
        .position(|byte| !byte.is_ascii_whitespace())
        .expect("validated non-empty object");
    let last = wire
        .iter()
        .rposition(|byte| !byte.is_ascii_whitespace())
        .expect("validated non-empty object");
    let has_existing_fields = wire[first + 1..last]
        .iter()
        .any(|byte| !byte.is_ascii_whitespace());
    let mut body = BytesMut::with_capacity(wire.len() + override_inner.len() + 1);
    body.put_slice(&wire[..last]);
    if has_existing_fields {
        body.put_u8(b',');
    }
    body.put_slice(&override_inner);
    body.put_slice(&wire[last..]);
    Ok(body.freeze())
}

pub(crate) fn validate_object_slice(wire: &[u8]) -> std::result::Result<(), String> {
    let Some(first) = wire.iter().find(|byte| !byte.is_ascii_whitespace()) else {
        return Err("empty byte slice".into());
    };
    let Some(last) = wire.iter().rfind(|byte| !byte.is_ascii_whitespace()) else {
        return Err("empty byte slice".into());
    };
    if *first != b'{' || *last != b'}' {
        return Err("expected a top-level JSON object".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use crate::dataset::segment::SegmentPool;

    struct Splices(HashMap<String, Vec<Bytes>>);

    impl MessageSpliceResolver for Splices {
        fn resolve(&self, key: &str) -> Result<Vec<Bytes>> {
            self.0
                .get(key)
                .cloned()
                .ok_or_else(|| DatasetError::MissingSplice(key.to_string()))
        }
    }

    #[test]
    fn static_body_is_exact_concat_with_override_tail() {
        let mut pool = SegmentPool::new();
        let system = pool
            .intern_message(
                None,
                "system",
                Bytes::from_static(br#"{"role":"system","content":"S"}"#),
                vec![1_u32].into_boxed_slice(),
            )
            .unwrap();
        let user = pool
            .intern_message(
                Some(system),
                "user",
                Bytes::from_static(br#"{"content":"hi","role":"user","x":1}"#),
                vec![2_u32].into_boxed_slice(),
            )
            .unwrap();
        let mut overrides = Overrides::new();
        overrides.set_model("m");
        overrides.set_stream(true);

        let body = pool.build_body(&[system, user], &overrides).unwrap();
        assert_eq!(
            body,
            Bytes::from_static(
                br#"{"messages":[{"role":"system","content":"S"},{"content":"hi","role":"user","x":1}],"model":"m","stream":true}"#
            )
        );
    }

    #[test]
    fn raw_body_is_byte_identical_without_overrides_and_tail_spliced_with_them() {
        let mut pool = SegmentPool::new();
        let authored = Bytes::from_static(b" \t{\"z\":1, \"messages\":[]}\n");
        let raw = pool.intern_raw(None, authored.clone()).unwrap();
        assert_eq!(
            pool.build_body(&[raw], &Overrides::new()).unwrap(),
            authored
        );

        let mut overrides = Overrides::new();
        overrides.set_model("new");
        let body = pool.build_body(&[raw], &overrides).unwrap();
        assert_eq!(
            body,
            Bytes::from_static(b" \t{\"z\":1, \"messages\":[],\"model\":\"new\"}\n")
        );
    }

    #[test]
    fn raw_override_uses_last_key_semantics_without_rewriting_authored_bytes() {
        let mut pool = SegmentPool::new();
        let raw = pool
            .intern_raw(None, Bytes::from_static(br#"{"model":"old","x":1}"#))
            .unwrap();
        let mut overrides = Overrides::new();
        overrides.set_model("new");
        let body = pool.build_body(&[raw], &overrides).unwrap();
        assert_eq!(
            body,
            Bytes::from_static(br#"{"model":"old","x":1,"model":"new"}"#)
        );
        let decoded: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(decoded["model"], "new");
    }

    #[test]
    fn dynamic_splice_interleaves_preencoded_messages() {
        let mut pool = SegmentPool::new();
        let first = pool
            .intern_message(
                None,
                "user",
                Bytes::from_static(br#"{"role":"user","content":"one"}"#),
                vec![1_u32].into_boxed_slice(),
            )
            .unwrap();
        let second = pool
            .intern_message(
                Some(first),
                "user",
                Bytes::from_static(br#"{"role":"user","content":"two"}"#),
                vec![2_u32].into_boxed_slice(),
            )
            .unwrap();
        let materializer = SegmentItemsMaterializer::new(Arc::new(pool.freeze()));
        let splices = Splices(HashMap::from([(
            "reply".into(),
            vec![Bytes::from_static(
                br#"{"role":"assistant","content":"answer"}"#,
            )],
        )]));
        let body = materializer
            .build(
                &[
                    AssemblyItem::Segment(first),
                    AssemblyItem::Splice("reply".into()),
                    AssemblyItem::Segment(second),
                ],
                &splices,
                &Overrides::new(),
            )
            .unwrap();
        assert_eq!(
            body,
            Bytes::from_static(
                br#"{"messages":[{"role":"user","content":"one"},{"role":"assistant","content":"answer"},{"role":"user","content":"two"}]}"#
            )
        );
    }
}
