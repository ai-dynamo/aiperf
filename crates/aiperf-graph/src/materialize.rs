// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Graph prompt materialization over pre-serialized message slices.
//!
//! Static items resolve dense handles through `aiperf-dataset`; dynamic items
//! clone the encoded replies retained by message channels. Neither path parses or
//! reserializes a message while building a successor prompt.

use std::collections::BTreeMap;
use std::sync::Arc;

use aiperf_dataset::{
    AssemblyItem, DatasetError, MessageSpliceResolver,
    SegmentItemsMaterializer as SharedMaterializer, SegmentStore,
};
use bytes::Bytes;

use crate::model::{LlmNode, PromptItem};
use crate::reducers::ChanVal;

/// Build a node's ordered wire-message slices from its program and channel state.
pub trait PromptMaterializer {
    /// Materialize without decoding or reserializing static messages.
    fn build(
        &self,
        node: &LlmNode,
        inputs: &BTreeMap<String, ChanVal>,
    ) -> Result<Vec<Bytes>, DatasetError>;
}

/// Shared-store plus dynamic-splice materializer.
pub struct SegmentItemsMaterializer {
    inner: SharedMaterializer,
}

impl SegmentItemsMaterializer {
    /// Bind the graph materializer to the universal segment store.
    pub fn new(store: Arc<dyn SegmentStore>) -> Self {
        Self {
            inner: SharedMaterializer::new(store),
        }
    }
}

struct InputSplices<'a>(&'a BTreeMap<String, ChanVal>);

impl MessageSpliceResolver for InputSplices<'_> {
    fn resolve(&self, key: &str) -> Result<Vec<Bytes>, DatasetError> {
        match self.0.get(key) {
            Some(ChanVal::EncodedMessages { wires, .. }) => Ok(wires.clone()),
            Some(ChanVal::Val(serde_json::Value::Array(messages))) => messages
                .iter()
                .map(|message| {
                    serde_json::to_vec(message)
                        .map(Bytes::from)
                        .map_err(Into::into)
                })
                .collect(),
            Some(ChanVal::Unset) | None => Ok(Vec::new()),
            Some(ChanVal::Val(_)) => Ok(Vec::new()),
        }
    }
}

impl PromptMaterializer for SegmentItemsMaterializer {
    fn build(
        &self,
        node: &LlmNode,
        inputs: &BTreeMap<String, ChanVal>,
    ) -> Result<Vec<Bytes>, DatasetError> {
        let items: Vec<_> = node
            .items
            .iter()
            .map(|item| match item {
                PromptItem::Seg { seg } => AssemblyItem::Segment(*seg),
                PromptItem::Splice { splice } => AssemblyItem::Splice(splice.clone()),
            })
            .collect();
        self.inner
            .materialize_messages(&items, &InputSplices(inputs))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::segment::{SegmentPool, intern_message};
    use crate::wire::OpenAiChatMessage as Msg;
    use aiperf_dataset::TiktokenTokenizer;
    use serde_json::json;

    #[test]
    fn interleaves_static_segments_and_retained_dynamic_wire() {
        let tokenizer = TiktokenTokenizer::builtin();
        let mut pool = SegmentPool::new();
        let system = intern_message(&mut pool, &Msg::new("system", "S"), None, &tokenizer).unwrap();
        let user = intern_message(
            &mut pool,
            &Msg::new("user", "again"),
            Some(system),
            &tokenizer,
        )
        .unwrap();
        let materializer = SegmentItemsMaterializer::new(Arc::new(pool.freeze()));
        let node: LlmNode = serde_json::from_value(json!({
            "node_type": "llm", "prompt": [], "output": "o",
            "items": [{"seg": system}, {"splice": "pred"}, {"seg": user}]
        }))
        .unwrap();
        let reply = json!({"role":"assistant","content":"prior reply"});
        let reply_wire = Bytes::from_static(br#"{"role":"assistant","content":"prior reply"}"#);
        let mut inputs = BTreeMap::new();
        inputs.insert(
            "pred".to_string(),
            ChanVal::EncodedMessages {
                value: serde_json::Value::Array(vec![reply]),
                wires: vec![reply_wire.clone()],
            },
        );

        let messages = materializer.build(&node, &inputs).unwrap();
        assert_eq!(messages.len(), 3);
        assert_eq!(messages[1], reply_wire);
    }
}
