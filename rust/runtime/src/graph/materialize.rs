// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Graph prompt materialization over pre-serialized message slices.
//!
//! Static items resolve dense handles through `aiperf-dataset`; dynamic items
//! clone the encoded replies retained by message channels. Neither path parses or
//! reserializes a message while building a successor prompt.

use std::collections::BTreeMap;
use std::error::Error;
use std::fmt::{self, Display};
use std::sync::Arc;

use crate::dataset::{
    AssemblyItem, DatasetError, MessageSpliceResolver, Payload,
    SegmentItemsMaterializer as SharedMaterializer, SegmentStore,
};
use bytes::Bytes;
use serde_json::{Map, Value, value::RawValue};

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

    /// Compose typed request fields after dynamic prompt assembly.
    ///
    /// Lightweight materializers retain the message-only contract; the shared
    /// segment materializer overrides this to resolve node-owned raw fields.
    fn materialize_request(
        &self,
        node: &LlmNode,
        messages: Vec<Bytes>,
    ) -> Result<MaterializedGraphRequest, GraphRequestMaterializationError> {
        Ok(MaterializedGraphRequest {
            messages,
            tools: None,
            model: None,
            additional_body: None,
            max_tokens: node.max_tokens,
            streaming: node.streaming,
        })
    }
}

/// Fully typed core request assembled for one graph LLM node.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MaterializedGraphRequest {
    /// Ordered pre-serialized message objects.
    pub messages: Vec<Bytes>,
    /// Optional pre-serialized tool array.
    pub tools: Option<Bytes>,
    /// Optional typed model override.
    pub model: Option<String>,
    /// Optional pre-serialized non-protected request fields.
    pub additional_body: Option<Bytes>,
    /// Typed generation cap.
    pub max_tokens: Option<usize>,
    /// Typed streaming flag.
    pub streaming: bool,
}

/// Failure while resolving typed request fields from the shared segment store.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraphRequestMaterializationError(pub String);

impl Display for GraphRequestMaterializationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl Error for GraphRequestMaterializationError {}

impl From<DatasetError> for GraphRequestMaterializationError {
    fn from(error: DatasetError) -> Self {
        Self(error.to_string())
    }
}

/// Resolve node-owned request fields after dynamic message assembly.
pub trait GraphRequestMaterializer {
    /// Compose prebuilt message wires with the typed fields owned by `node`.
    fn materialize(
        &self,
        node: &LlmNode,
        messages: Vec<Bytes>,
    ) -> Result<MaterializedGraphRequest, GraphRequestMaterializationError>;
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

    fn materialize_request(
        &self,
        node: &LlmNode,
        messages: Vec<Bytes>,
    ) -> Result<MaterializedGraphRequest, GraphRequestMaterializationError> {
        GraphRequestMaterializer::materialize(self, node, messages)
    }
}

impl GraphRequestMaterializer for SegmentItemsMaterializer {
    fn materialize(
        &self,
        node: &LlmNode,
        messages: Vec<Bytes>,
    ) -> Result<MaterializedGraphRequest, GraphRequestMaterializationError> {
        let Some(spec) = node.request.as_ref() else {
            return Ok(MaterializedGraphRequest {
                messages,
                tools: None,
                model: None,
                additional_body: None,
                max_tokens: node.max_tokens,
                streaming: node.streaming,
            });
        };
        let tools = spec
            .tools
            .map(|handle| raw_wire(self.inner.store().as_ref(), handle, "tools"))
            .transpose()?;
        let additional_body = spec
            .additional_body
            .map(|handle| raw_wire(self.inner.store().as_ref(), handle, "additional body"))
            .transpose()?;
        if let Some(body) = additional_body.as_ref() {
            decode_additional_body_wire(body, "additional body")?;
        }
        Ok(MaterializedGraphRequest {
            messages,
            tools,
            model: spec.model.clone(),
            additional_body,
            max_tokens: node.max_tokens,
            streaming: node.streaming,
        })
    }
}

/// Validate one preserved JSON body and return its fields without altering its wire bytes.
pub(crate) fn decode_additional_body_wire(
    wire: &[u8],
    origin: &str,
) -> Result<Map<String, Value>, GraphRequestMaterializationError> {
    let value: Value = serde_json::from_slice(wire).map_err(|error| {
        GraphRequestMaterializationError(format!("{origin} is not JSON: {error}"))
    })?;
    let object = value.as_object().ok_or_else(|| {
        GraphRequestMaterializationError(format!("{origin} must be a JSON object"))
    })?;
    validate_additional_body(object, origin)?;
    Ok(object.clone())
}

/// Reject fields that remain under typed request ownership.
pub(crate) fn validate_additional_body(
    body: &Map<String, Value>,
    origin: &str,
) -> Result<(), GraphRequestMaterializationError> {
    const PROTECTED: &[&str] = &[
        "api_base",
        "api_key",
        "custom_llm_provider",
        "max_tokens",
        "messages",
        "model",
        "stream",
        "stream_options",
        "timeout",
        "tools",
    ];
    if let Some(key) = body.keys().find(|key| PROTECTED.contains(&key.as_str())) {
        return Err(GraphRequestMaterializationError(format!(
            "{origin} may not override protected request field {key:?}"
        )));
    }
    Ok(())
}

fn raw_wire(
    store: &dyn SegmentStore,
    handle: crate::dataset::Handle,
    label: &str,
) -> Result<Bytes, GraphRequestMaterializationError> {
    let Payload::Raw { wire } = store.get(handle)? else {
        return Err(GraphRequestMaterializationError(format!(
            "{label} handle {handle} must reference raw bytes"
        )));
    };
    Ok(wire.clone())
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
