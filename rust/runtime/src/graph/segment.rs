// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Graph-facing exports for the universal dataset segment store.
//!
//! Graph prompts and ordinary dataset turns use exactly the same dense arena,
//! BLAKE3 hash domains, prefix-parent identity, and byte-concatenation path. This
//! module contains only the graph message convenience function; storage lives in
//! `crate::dataset` so a second graph-only implementation cannot drift.

use bytes::Bytes;

use crate::dataset::{Result, TextTokenizer};
use crate::graph::wire::WireMessage;

pub use crate::dataset::{
    Handle, InMemorySegmentStore, Payload, Role, Segment, SegmentId, SegmentPool, SegmentStore,
};

/// Serialize and tokenize one dialect message once, then intern it under its
/// prefix parent in the shared store.
pub fn intern_message<M: WireMessage>(
    pool: &mut SegmentPool,
    message: &M,
    parent: Option<Handle>,
    tokenizer: &dyn TextTokenizer,
) -> Result<Handle> {
    let wire = serde_json::to_vec(message)?;
    let text = std::str::from_utf8(&wire).expect("serde_json always emits UTF-8");
    let tokens = tokenizer.encode(text)?;
    pool.intern_message(
        parent,
        message.role(),
        Bytes::from(wire),
        tokens.into_boxed_slice(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::TiktokenTokenizer;
    use crate::graph::wire::OpenAiChatMessage as Msg;

    #[test]
    fn graph_messages_use_dense_prefix_dependent_dataset_handles() {
        let tokenizer = TiktokenTokenizer::builtin();
        let mut pool = SegmentPool::new();
        let system = intern_message(&mut pool, &Msg::new("system", "S"), None, &tokenizer).unwrap();
        let child =
            intern_message(&mut pool, &Msg::new("user", "hi"), Some(system), &tokenizer).unwrap();
        let duplicate =
            intern_message(&mut pool, &Msg::new("user", "hi"), Some(system), &tokenizer).unwrap();
        let root = intern_message(&mut pool, &Msg::new("user", "hi"), None, &tokenizer).unwrap();
        assert_eq!(system.index(), 0);
        assert_eq!(child, duplicate);
        assert_ne!(child, root);
    }
}
