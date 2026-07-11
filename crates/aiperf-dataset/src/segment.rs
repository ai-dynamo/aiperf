// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Dense, prefix-dependent content-addressed segment storage.
//!
//! IDs use four disjoint BLAKE3 domains (`message`, `text-only`, `raw`, and
//! `media`). A child hash includes its parent's content hash rather than the
//! parent's insertion index, so IDs remain deterministic when unrelated rows are
//! loaded in a different order. The public address is a dense [`Handle`]; the
//! hash-to-handle map exists only while [`SegmentPool`] is mutable.

use std::collections::HashMap;
use std::fmt::{self, Display};

use bytes::Bytes;
use serde::{Deserialize, Serialize};

use crate::error::{DatasetError, Result};
use crate::materialize::{Overrides, build_body_from_handles};
use crate::model::MediaKind;

const HASH_VERSION: &[u8] = b"aiperf-dataset-segment-v1\0";

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
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Display::fmt(&self.0, f)
    }
}

/// Opaque deterministic BLAKE3 content identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SegmentId([u8; 32]);

impl SegmentId {
    /// Borrow the digest bytes used when hashing a child segment.
    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    /// Return the lowercase hexadecimal digest.
    pub fn to_hex(self) -> String {
        self.0.iter().map(|byte| format!("{byte:02x}")).collect()
    }
}

impl Display for SegmentId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for byte in self.0 {
            write!(f, "{byte:02x}")?;
        }
        Ok(())
    }
}

/// Open-ended message role retained alongside a pre-serialized message.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Role(String);

impl Role {
    /// Construct a role without restricting endpoint-specific role names.
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    /// Borrow the role string.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl From<&str> for Role {
    fn from(value: &str) -> Self {
        Self::new(value)
    }
}

impl From<String> for Role {
    fn from(value: String) -> Self {
        Self::new(value)
    }
}

impl Display for Role {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// One interned wire payload.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Payload {
    /// Pre-serialized endpoint message keyed by its authoritative token IDs.
    Message {
        /// Message role, framed into the content hash.
        role: Role,
        /// Exact compact wire object, including authored key order and extra keys.
        wire: Bytes,
        /// Token IDs produced at composition time.
        tokens: Box<[u32]>,
    },
    /// Plain text content for endpoint fields that are not message objects.
    Text {
        /// Text role or field role used for prefix framing.
        role: Role,
        /// Exact UTF-8 bytes.
        bytes: Bytes,
        /// Token IDs produced at composition time.
        tokens: Box<[u32]>,
    },
    /// Exact JSON wire for a raw body, raw message list, tools, headers, or extras.
    Raw {
        /// Key-order-sensitive bytes; consumers interpret the field by its handle slot.
        wire: Bytes,
    },
    /// Binary or encoded multimodal content.
    Media {
        /// Media category used by endpoint formatting and accounting.
        kind: MediaKind,
        /// Exact content bytes, stored once even when referenced by many turns.
        bytes: Bytes,
    },
}

impl Payload {
    /// Stable kind name used in validation errors.
    pub const fn kind_name(&self) -> &'static str {
        match self {
            Self::Message { .. } => "message",
            Self::Text { .. } => "text-only",
            Self::Raw { .. } => "raw",
            Self::Media { .. } => "media",
        }
    }

    /// Number of authoritative input tokens carried by this payload, when tokenized.
    pub fn token_count(&self) -> Option<usize> {
        match self {
            Self::Message { tokens, .. } | Self::Text { tokens, .. } => Some(tokens.len()),
            Self::Raw { .. } | Self::Media { .. } => None,
        }
    }
}

/// One arena entry with its deterministic hash and prefix parent.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Segment {
    /// Deterministic content identifier; consumers treat it as opaque.
    pub id: SegmentId,
    /// Prefix segment this entry extends.
    pub parent: Option<Handle>,
    /// Interned content.
    pub payload: Payload,
}

/// Read-only segment-store seam shared by datasets and graph materializers.
pub trait SegmentStore: Send + Sync {
    /// Return an arena entry, or `None` for an unknown handle.
    fn segment(&self, handle: Handle) -> Option<&Segment>;

    /// Number of unique interned segments.
    fn len(&self) -> usize;

    /// Whether the store contains no segments.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Resolve one payload with a contextual unknown-handle error.
    fn get(&self, handle: Handle) -> Result<&Payload> {
        self.segment(handle)
            .map(|segment| &segment.payload)
            .ok_or(DatasetError::UnknownHandle(handle))
    }

    /// Resolve one deterministic content identifier.
    fn id(&self, handle: Handle) -> Result<SegmentId> {
        self.segment(handle)
            .map(|segment| segment.id)
            .ok_or(DatasetError::UnknownHandle(handle))
    }

    /// Assemble a JSON request body by concatenating stored wire slices and only
    /// serializing the per-dispatch override tail.
    fn build_body(&self, handles: &[Handle], overrides: &Overrides) -> Result<Bytes> {
        build_body_from_handles(self, handles, overrides)
    }
}

/// Mutable write-side interner.
#[derive(Debug, Clone, Default)]
pub struct SegmentPool {
    arena: Vec<Segment>,
    ids: HashMap<SegmentId, Handle>,
}

impl SegmentPool {
    /// Create an empty segment pool.
    pub fn new() -> Self {
        Self::default()
    }

    /// Intern one payload under an optional prefix parent.
    pub fn intern(&mut self, parent: Option<Handle>, payload: Payload) -> Result<Handle> {
        let parent_id = match parent {
            Some(handle) => Some(
                self.arena
                    .get(handle.as_usize())
                    .ok_or(DatasetError::UnknownParent(handle))?
                    .id,
            ),
            None => None,
        };
        let id = payload_id(parent_id, &payload);
        if let Some(handle) = self.ids.get(&id) {
            return Ok(*handle);
        }
        let index =
            u32::try_from(self.arena.len()).map_err(|_| DatasetError::SegmentCapacityExceeded)?;
        let handle = Handle::new(index);
        self.arena.push(Segment {
            id,
            parent,
            payload,
        });
        self.ids.insert(id, handle);
        Ok(handle)
    }

    /// Intern a pre-serialized message keyed on token IDs.
    pub fn intern_message(
        &mut self,
        parent: Option<Handle>,
        role: impl Into<Role>,
        wire: impl Into<Bytes>,
        tokens: impl Into<Box<[u32]>>,
    ) -> Result<Handle> {
        self.intern(
            parent,
            Payload::Message {
                role: role.into(),
                wire: wire.into(),
                tokens: tokens.into(),
            },
        )
    }

    /// Intern plain text under the disjoint `text-only` hash domain.
    pub fn intern_text(
        &mut self,
        parent: Option<Handle>,
        role: impl Into<Role>,
        bytes: impl Into<Bytes>,
        tokens: impl Into<Box<[u32]>>,
    ) -> Result<Handle> {
        self.intern(
            parent,
            Payload::Text {
                role: role.into(),
                bytes: bytes.into(),
                tokens: tokens.into(),
            },
        )
    }

    /// Intern exact raw wire bytes under the disjoint `raw` hash domain.
    pub fn intern_raw(&mut self, parent: Option<Handle>, wire: impl Into<Bytes>) -> Result<Handle> {
        self.intern(parent, Payload::Raw { wire: wire.into() })
    }

    /// Intern media content under the disjoint `media` hash domain.
    pub fn intern_media(
        &mut self,
        parent: Option<Handle>,
        kind: MediaKind,
        bytes: impl Into<Bytes>,
    ) -> Result<Handle> {
        self.intern(
            parent,
            Payload::Media {
                kind,
                bytes: bytes.into(),
            },
        )
    }

    /// Freeze the arena and discard the write-only hash map.
    pub fn freeze(self) -> InMemorySegmentStore {
        InMemorySegmentStore {
            arena: self.arena.into_boxed_slice(),
        }
    }
}

impl SegmentStore for SegmentPool {
    fn segment(&self, handle: Handle) -> Option<&Segment> {
        self.arena.get(handle.as_usize())
    }

    fn len(&self) -> usize {
        self.arena.len()
    }
}

/// Frozen in-memory arena shared across worker threads.
#[derive(Debug, Clone, Default)]
pub struct InMemorySegmentStore {
    arena: Box<[Segment]>,
}

impl InMemorySegmentStore {
    /// Borrow the dense arena in insertion order.
    pub fn segments(&self) -> &[Segment] {
        &self.arena
    }
}

impl SegmentStore for InMemorySegmentStore {
    fn segment(&self, handle: Handle) -> Option<&Segment> {
        self.arena.get(handle.as_usize())
    }

    fn len(&self) -> usize {
        self.arena.len()
    }
}

fn payload_id(parent: Option<SegmentId>, payload: &Payload) -> SegmentId {
    let mut hasher = blake3::Hasher::new();
    hasher.update(HASH_VERSION);
    match payload {
        Payload::Message { role, tokens, .. } => {
            hasher.update(b"message\0");
            hash_parent(&mut hasher, parent);
            hasher.update(role.as_str().as_bytes());
            hasher.update(b"\0");
            for token in tokens {
                hasher.update(&token.to_le_bytes());
            }
        }
        Payload::Text { role, bytes, .. } => {
            hasher.update(b"text-only\0");
            hash_parent(&mut hasher, parent);
            hasher.update(role.as_str().as_bytes());
            hasher.update(b"\0");
            hasher.update(bytes);
        }
        Payload::Raw { wire } => {
            hasher.update(b"raw\0");
            hash_parent(&mut hasher, parent);
            hasher.update(wire);
        }
        Payload::Media { kind, bytes } => {
            hasher.update(b"media\0");
            hash_parent(&mut hasher, parent);
            hasher.update(kind.as_str().as_bytes());
            hasher.update(b"\0");
            hasher.update(bytes);
        }
    }
    SegmentId(*hasher.finalize().as_bytes())
}

fn hash_parent(hasher: &mut blake3::Hasher, parent: Option<SegmentId>) {
    if let Some(parent) = parent {
        hasher.update(parent.as_bytes());
    }
    hasher.update(b"\0");
}

#[cfg(test)]
mod tests {
    use super::*;

    fn msg(content: &str) -> Bytes {
        Bytes::from(format!(r#"{{"role":"user","content":"{content}"}}"#))
    }

    #[test]
    fn handles_are_dense_and_dedup_by_prefix_dependent_id() {
        let mut pool = SegmentPool::new();
        let root = pool
            .intern_message(
                None,
                "system",
                msg("system"),
                vec![1_u32, 2].into_boxed_slice(),
            )
            .unwrap();
        let child = pool
            .intern_message(
                Some(root),
                "user",
                msg("hello"),
                vec![3_u32].into_boxed_slice(),
            )
            .unwrap();
        let duplicate = pool
            .intern_message(
                Some(root),
                "user",
                msg("hello"),
                vec![3_u32].into_boxed_slice(),
            )
            .unwrap();
        let different_prefix = pool
            .intern_message(None, "user", msg("hello"), vec![3_u32].into_boxed_slice())
            .unwrap();

        assert_eq!(root.index(), 0);
        assert_eq!(child.index(), 1);
        assert_eq!(duplicate, child);
        assert_eq!(different_prefix.index(), 2);
        assert_ne!(pool.id(child).unwrap(), pool.id(different_prefix).unwrap());
        assert_eq!(pool.len(), 3);
    }

    #[test]
    fn domains_cannot_alias() {
        let mut pool = SegmentPool::new();
        let bytes = Bytes::from_static(b"same");
        let message = pool
            .intern_message(None, "user", bytes.clone(), vec![1_u32].into_boxed_slice())
            .unwrap();
        let text = pool
            .intern_text(None, "user", bytes.clone(), vec![1_u32].into_boxed_slice())
            .unwrap();
        let raw = pool.intern_raw(None, bytes.clone()).unwrap();
        let media = pool
            .intern_media(None, MediaKind::Image, bytes.clone())
            .unwrap();

        let ids = [message, text, raw, media].map(|handle| pool.id(handle).unwrap());
        for (index, id) in ids.iter().enumerate() {
            assert!(!ids[..index].contains(id));
        }
    }

    #[test]
    fn frozen_store_discards_writer_map_but_preserves_handles() {
        let mut pool = SegmentPool::new();
        let handle = pool.intern_raw(None, Bytes::from_static(b"{}")).unwrap();
        let id = pool.id(handle).unwrap();
        let store = pool.freeze();
        assert_eq!(store.id(handle).unwrap(), id);
        assert_eq!(store.len(), 1);
    }
}
