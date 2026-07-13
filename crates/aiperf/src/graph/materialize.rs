// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Graph prompt materialization over pre-serialized message slices.
//!
//! Static items resolve dense handles through `aiperf-dataset`; dynamic items
//! clone the encoded replies retained by message channels. Neither path parses or
//! reserializes a message while building a successor prompt.

use std::collections::BTreeMap;
use std::sync::Arc;

use crate::dataset::{
    AssemblyItem, DatasetError, MessageSpliceResolver, Payload,
    SegmentItemsMaterializer as SharedMaterializer, SegmentStore,
};
use bytes::Bytes;
use serde_json::value::RawValue;

use crate::graph::model::{LlmNode, PromptItem};
use crate::graph::reducers::ChanVal;

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
        let splices = InputSplices(inputs);
        let mut messages = Vec::with_capacity(node.items.len());
        for item in &node.items {
            match item {
                PromptItem::Seg { seg } => messages.extend(
                    self.inner
                        .materialize_messages(&[AssemblyItem::Segment(*seg)], &splices)?,
                ),
                PromptItem::RawMessages { raw_messages } => {
                    messages.extend(raw_message_wires(
                        self.inner.store().as_ref(),
                        *raw_messages,
                    )?);
                }
                PromptItem::Text { text, role } => {
                    messages.push(text_message_wire(self.inner.store().as_ref(), *text, role)?);
                }
                PromptItem::Splice { splice } => messages.extend(splices.resolve(splice)?),
            }
        }
        Ok(messages)
    }
}

fn raw_message_wires(
    store: &dyn SegmentStore,
    handle: crate::dataset::Handle,
) -> Result<Vec<Bytes>, DatasetError> {
    let Payload::Raw { wire } = store.get(handle)? else {
        return Err(DatasetError::PayloadKind {
            handle,
            expected: "raw message array",
            actual: store.get(handle)?.kind_name(),
        });
    };
    let messages: Vec<Box<RawValue>> = serde_json::from_slice(wire)?;
    if messages.is_empty() {
        return Err(DatasetError::InvalidWire(format!(
            "raw message-array handle {handle} is empty"
        )));
    }
    messages
        .into_iter()
        .enumerate()
        .map(|(index, message)| {
            let value: serde_json::Value = serde_json::from_str(message.get())?;
            if !value.is_object() {
                return Err(DatasetError::InvalidWire(format!(
                    "raw message-array handle {handle} entry {index} is not an object"
                )));
            }
            Ok(Bytes::copy_from_slice(message.get().as_bytes()))
        })
        .collect()
}

fn text_message_wire(
    store: &dyn SegmentStore,
    handle: crate::dataset::Handle,
    role: &str,
) -> Result<Bytes, DatasetError> {
    let Payload::Text { bytes, .. } = store.get(handle)? else {
        return Err(DatasetError::PayloadKind {
            handle,
            expected: "text-only",
            actual: store.get(handle)?.kind_name(),
        });
    };
    let content = std::str::from_utf8(bytes).map_err(|error| {
        DatasetError::InvalidWire(format!("text handle {handle} is not UTF-8: {error}"))
    })?;
    serde_json::to_vec(&serde_json::json!({"role": role, "content": content}))
        .map(Bytes::from)
        .map_err(Into::into)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::{TextTokenizer, TiktokenTokenizer};
    use crate::graph::segment::{SegmentPool, intern_message};
    use crate::graph::wire::OpenAiChatMessage as Msg;
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

    #[test]
    fn expands_raw_message_arrays_without_reencoding_the_objects() {
        let mut pool = SegmentPool::new();
        let raw = pool
            .intern_raw(
                None,
                Bytes::from_static(
                    br#"[ { "role" : "system", "content" : "S" },{"role":"user","content":"U"}]"#,
                ),
            )
            .unwrap();
        let materializer = SegmentItemsMaterializer::new(Arc::new(pool.freeze()));
        let node: LlmNode = serde_json::from_value(json!({
            "output": "o",
            "items": [{"raw_messages": raw}]
        }))
        .unwrap();

        let messages = materializer.build(&node, &BTreeMap::new()).unwrap();
        assert_eq!(messages.len(), 2);
        assert_eq!(
            messages[0],
            Bytes::from_static(br#"{ "role" : "system", "content" : "S" }"#)
        );
        assert_eq!(
            messages[1],
            Bytes::from_static(br#"{"role":"user","content":"U"}"#)
        );
    }

    #[test]
    fn projects_text_segments_with_the_authored_role() {
        let tokenizer = TiktokenTokenizer::builtin();
        let mut pool = SegmentPool::new();
        let text = pool
            .intern_text(
                None,
                "system",
                Bytes::from_static(b"shared"),
                tokenizer.encode("shared").unwrap(),
            )
            .unwrap();
        let materializer = SegmentItemsMaterializer::new(Arc::new(pool.freeze()));
        let node: LlmNode = serde_json::from_value(json!({
            "output": "o",
            "items": [{"text": text, "role": "system"}]
        }))
        .unwrap();

        let messages = materializer.build(&node, &BTreeMap::new()).unwrap();
        assert_eq!(
            serde_json::from_slice::<serde_json::Value>(&messages[0]).unwrap(),
            json!({"role": "system", "content": "shared"})
        );
    }
}
