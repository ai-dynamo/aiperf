// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Prompt materialization — an **extensible trait** (generic over dialect message
//! `M`) turning a node into the `messages` list dispatched on the wire.
//!
//! The default impl ([`SegmentItemsMaterializer`]) walks a node's assembly
//! program: static, prefix-cached segments come from a [`SegmentStore`], and
//! dynamic slots splice a predecessor's captured reply, read from that
//! predecessor's output channel. Swapping the trait plugs in a different grammar.

use std::collections::BTreeMap;
use std::rc::Rc;

use serde_json::Value;

use crate::model::{LlmNode, PromptItem};
use crate::reducers::ChanVal;
use crate::segment::SegmentStore;
use crate::wire::WireMessage;

/// Build a node's wire messages from its program + the channel state at fire time.
pub trait PromptMaterializer<M: WireMessage> {
    fn build(&self, node: &LlmNode, inputs: &BTreeMap<String, ChanVal>) -> Vec<M>;
}

/// Segment-store + dynamic-splice materializer.
pub struct SegmentItemsMaterializer<M: WireMessage> {
    store: Rc<dyn SegmentStore<M>>,
}

impl<M: WireMessage> SegmentItemsMaterializer<M> {
    pub fn new(store: Rc<dyn SegmentStore<M>>) -> Self {
        SegmentItemsMaterializer { store }
    }
}

impl<M: WireMessage> PromptMaterializer<M> for SegmentItemsMaterializer<M> {
    fn build(&self, node: &LlmNode, inputs: &BTreeMap<String, ChanVal>) -> Vec<M> {
        let mut out: Vec<M> = Vec::new();
        for item in &node.items {
            match item {
                PromptItem::Seg { seg } => {
                    out.extend(self.store.materialize(std::slice::from_ref(seg)));
                }
                PromptItem::Splice { splice } => {
                    if let Some(ChanVal::Val(Value::Array(msgs))) = inputs.get(splice) {
                        for m in msgs {
                            if let Ok(msg) = serde_json::from_value::<M>(m.clone()) {
                                out.push(msg);
                            }
                        }
                    }
                }
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::segment::SegmentPool;
    use crate::wire::OpenAiChatMessage as Msg;
    use serde_json::json;

    #[test]
    fn interleaves_static_segments_and_dynamic_splices() {
        let mut pool: SegmentPool<Msg> = SegmentPool::new();
        let sys = pool.add(Msg::new("system", "S"), None);
        let u1 = pool.add(Msg::new("user", "hi"), Some(&sys));
        let u2 = pool.add(Msg::new("user", "again"), Some(&u1));
        let store: Rc<dyn SegmentStore<Msg>> = Rc::new(pool);
        let mat = SegmentItemsMaterializer::new(store);

        let node: LlmNode = serde_json::from_value(json!({
            "node_type": "llm", "prompt": [], "output": "o",
            "items": [{"seg": sys}, {"seg": u1}, {"splice": "pred"}, {"seg": u2}]
        }))
        .unwrap();

        let mut inputs = BTreeMap::new();
        inputs.insert(
            "pred".to_string(),
            ChanVal::Val(json!([{"role": "assistant", "content": "prior reply"}])),
        );

        assert_eq!(
            mat.build(&node, &inputs),
            vec![
                Msg::new("system", "S"),
                Msg::new("user", "hi"),
                Msg::new("assistant", "prior reply"),
                Msg::new("user", "again"),
            ]
        );
    }
}
