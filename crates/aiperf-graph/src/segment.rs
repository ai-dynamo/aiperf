// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Content-addressed segment store — the prompt-materialization substrate,
//! generic over the dialect's [`WireMessage`] so segment shape is endpoint-
//! agnostic.
//!
//! Segments are **prefix-dependent, content-addressed** message fragments: a
//! segment's id folds in its `parent_id` plus the message's serialized bytes, so
//! identical content under different prefixes gets distinct ids and shared
//! prefixes dedup to one id — which lets a downstream KV-cache simulator reason
//! about prefix reuse. Ids are opaque `blake3` digests, so only determinism +
//! dedup-consistency matter.

use std::collections::HashMap;
use std::marker::PhantomData;

use crate::wire::WireMessage;

/// A content-addressed, prefix-dependent message segment.
#[derive(Debug, Clone, PartialEq)]
pub struct Segment<M> {
    pub id: String,
    /// Prefix segment this one extends, or `None` at a path root.
    pub parent_id: Option<String>,
    /// The dialect message this segment carries.
    pub message: M,
}

/// The read side every prompt materializer consumes — an **extensible trait** so
/// an in-memory pool, an mmap store, or a remote store all plug in.
pub trait SegmentStore<M: WireMessage> {
    /// The segment for `id`, if present.
    fn get(&self, id: &str) -> Option<&Segment<M>>;

    /// Reconstruct the message list for a segment path.
    fn materialize(&self, path_ids: &[String]) -> Vec<M> {
        path_ids
            .iter()
            .filter_map(|id| self.get(id))
            .map(|s| s.message.clone())
            .collect()
    }
}

/// In-memory content-addressed pool.
#[derive(Debug, Clone)]
pub struct SegmentPool<M> {
    by_id: HashMap<String, Segment<M>>,
    _marker: PhantomData<M>,
}

impl<M: WireMessage> Default for SegmentPool<M> {
    fn default() -> Self {
        SegmentPool {
            by_id: HashMap::new(),
            _marker: PhantomData,
        }
    }
}

impl<M: WireMessage> SegmentPool<M> {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn len(&self) -> usize {
        self.by_id.len()
    }
    pub fn is_empty(&self) -> bool {
        self.by_id.is_empty()
    }

    /// Intern a message under `parent_id`; returns its content-addressed id (dedup).
    pub fn add(&mut self, message: M, parent_id: Option<&str>) -> String {
        let id = segment_id(parent_id, message.role(), &message);
        self.by_id.entry(id.clone()).or_insert_with(|| Segment {
            id: id.clone(),
            parent_id: parent_id.map(str::to_string),
            message,
        });
        id
    }
}

impl<M: WireMessage> SegmentStore<M> for SegmentPool<M> {
    fn get(&self, id: &str) -> Option<&Segment<M>> {
        self.by_id.get(id)
    }
}

/// Content-addressed id: `blake3(parent \0 role \0 serialize(message))`, hex-16.
pub fn segment_id<M: WireMessage>(parent_id: Option<&str>, role: &str, message: &M) -> String {
    let mut h = blake3::Hasher::new();
    h.update(parent_id.unwrap_or("").as_bytes());
    h.update(b"\x00");
    h.update(role.as_bytes());
    h.update(b"\x00");
    h.update(
        serde_json::to_vec(message)
            .expect("serialize message")
            .as_slice(),
    );
    h.finalize().to_hex()[..32].to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wire::OpenAiChatMessage as Msg;

    #[test]
    fn prefix_dependence_and_dedup() {
        let mut pool: SegmentPool<Msg> = SegmentPool::new();
        let sys = pool.add(Msg::new("system", "You are concise."), None);
        let u1 = pool.add(Msg::new("user", "hello"), Some(&sys));
        let u1_root = pool.add(Msg::new("user", "hello"), None);
        assert_ne!(u1, u1_root); // same text, different prefix -> different id
        let u1_again = pool.add(Msg::new("user", "hello"), Some(&sys));
        assert_eq!(u1, u1_again); // dedup
        assert_eq!(pool.len(), 3);

        let msgs = pool.materialize(&[sys, u1]);
        assert_eq!(
            msgs,
            vec![
                Msg::new("system", "You are concise."),
                Msg::new("user", "hello")
            ]
        );
    }
}
