// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! One-pass parsing of the canonical agentic DAG JSONL source schema.
//!
//! This leaf parser validates one authored `dag_jsonl` source, preserving owned
//! exact JSON wires and normalizing FORK/SPAWN branch declarations, then hands
//! the validated program to [`crate::graph::input::compile_dag_jsonl_input`] for direct
//! Graph-IR lowering. It deliberately lives in the graph crate — beside its sole
//! consumer — rather than in the linear dataset loader registry: `dag_jsonl` is a
//! graph source, never a linear `Dataset`, conversation, or credit-orchestrator
//! program.
//!
//! Source acquisition reuses the dataset crate's registry-free
//! [`crate::dataset::load_raw_rows`] seam; no [`crate::dataset::DatasetLoader`]
//! registration or linear composer is involved.

use std::collections::{HashMap, HashSet};

use crate::dataset::{
    DatasetError, DatasetSource, LoadConfig, Result, RowOrigin, TextTokenizer, load_raw_rows,
};
use crate::endpoints::{ExtractedPayload, extract_payload};
use bytes::Bytes;
use serde::Deserialize;
use serde_json::value::RawValue;
use serde_json::{Map, Value};

/// One validated `dag_jsonl` source before any linear-dataset composition.
///
/// Graph runtimes consume this owned program directly. It deliberately contains
/// no linear conversation, DAG-metadata, sampler, or session-manager state.
#[derive(Debug, Clone)]
pub struct DagJsonlProgram {
    /// Conversations in authored source order.
    pub conversations: Vec<DagJsonlConversation>,
}

/// One authored DAG conversation retained for direct Graph-IR lowering.
#[derive(Debug, Clone)]
pub struct DagJsonlConversation {
    /// Source coordinate used by validation diagnostics.
    pub origin: RowOrigin,
    /// Stable authored identifier.
    pub session_id: String,
    /// Turns in authored order.
    pub turns: Vec<DagJsonlTurn>,
    /// Fresh-context children dispatched before root turn zero.
    pub pre_session_spawns: Vec<String>,
}

/// One authored DAG turn with exact JSON object/array wires.
#[derive(Debug, Clone)]
pub struct DagJsonlTurn {
    /// Exact authored message-array wire.
    pub messages: Bytes,
    /// Optional per-turn model override.
    pub model: Option<String>,
    /// Optional endpoint-adapter name.
    pub endpoint: Option<String>,
    /// Optional streaming override.
    pub streaming: Option<bool>,
    /// Optional generation cap.
    pub max_tokens: Option<u32>,
    /// Exact optional tools-array wire.
    pub tools: Option<Bytes>,
    /// Exact optional vendor system-array wire.
    pub raw_system: Option<Bytes>,
    /// Exact optional extra-body object wire.
    pub extra: Option<Bytes>,
    /// Exact optional extra-header object wire.
    pub extra_headers: Option<Bytes>,
    /// Exact optional query-parameter object wire.
    pub request_parameters: Option<Bytes>,
    /// FORK descriptors attached after this turn.
    pub forks: Vec<DagJsonlFork>,
    /// SPAWN descriptors attached after this turn.
    pub spawns: Vec<DagJsonlSpawn>,
    /// Authored delay in milliseconds.
    pub delay_ms: f64,
}

/// One normalized FORK descriptor.
#[derive(Debug, Clone)]
pub struct DagJsonlFork {
    /// Child conversation identifier.
    pub child: String,
    /// Whether the parent may continue without joining the child.
    pub background: bool,
}

/// One normalized SPAWN group.
#[derive(Debug, Clone)]
pub struct DagJsonlSpawn {
    /// Fresh-context child conversation identifiers.
    pub children: Vec<String>,
    /// Optional parent turn index that waits for every child terminal.
    pub join_at: Option<usize>,
}

/// Parse and validate a `dag_jsonl` source exactly once for direct Graph-IR use.
pub async fn load_dag_jsonl_program(config: &LoadConfig) -> Result<DagJsonlProgram> {
    // A row is a graph vertex, not a sampling unit. Fetch and validate the
    // complete program before the graph layer applies any limit to root plans;
    // truncating here could silently delete a referenced child conversation.
    let mut complete = config.clone();
    complete.max_rows = None;
    if let DatasetSource::HuggingFace { max_rows, .. } = &mut complete.source {
        *max_rows = None;
    }
    let rows = load_raw_rows(&complete).await?;
    if rows.is_empty() {
        return Err(DatasetError::Validation(
            "DAG JSONL source contains no conversations".into(),
        ));
    }
    let mut ids = HashSet::new();
    let mut conversations = Vec::with_capacity(rows.len());
    for row in rows {
        let retained_wire;
        let wire = if let Some(wire) = row.wire.as_deref() {
            wire
        } else {
            retained_wire = serde_json::to_vec(&row.value)?;
            retained_wire.as_slice()
        };
        let parsed: DagConversationRaw<'_> = serde_json::from_slice(wire).map_err(|error| {
            DatasetError::Validation(format!("{}: invalid DAG conversation: {error}", row.origin))
        })?;
        parsed.validate(&row.origin)?;
        if !ids.insert(parsed.session_id.clone()) {
            return Err(DatasetError::DuplicateSession(parsed.session_id));
        }
        conversations.push(parsed.into_program(row.origin)?);
    }
    Ok(DagJsonlProgram { conversations })
}

/// Count authored non-tool input tokens and tool tokens for one direct DAG turn.
pub fn dag_jsonl_turn_token_counts(
    messages: &[u8],
    tools: Option<&[u8]>,
    tokenizer: &dyn TextTokenizer,
) -> Result<(u64, u64)> {
    let messages: Value = serde_json::from_slice(messages)?;
    let tools = tools.map(serde_json::from_slice).transpose()?;
    let mut payload = Map::new();
    payload.insert("messages".into(), messages);
    if let Some(tools) = tools {
        payload.insert("tools".into(), tools);
    }
    let extracted = extract_payload(&Value::Object(payload));
    let input_tokens = input_tokens_excluding_tools(&extracted, tokenizer)?;
    let tool_tokens = extracted
        .tool_texts
        .iter()
        .try_fold(0_u64, |count, text| add_token_count(count, text, tokenizer))?;
    Ok((input_tokens, tool_tokens))
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct DagConversationRaw<'a> {
    session_id: String,
    #[serde(borrow)]
    turns: Vec<DagTurnRaw<'a>>,
    #[serde(default)]
    pre_session_spawns: Vec<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct DagTurnRaw<'a> {
    #[serde(borrow)]
    messages: &'a RawValue,
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    endpoint: Option<String>,
    #[serde(default)]
    streaming: Option<bool>,
    #[serde(default)]
    max_tokens: Option<u32>,
    #[serde(default, borrow)]
    tools: Option<&'a RawValue>,
    #[serde(default, borrow)]
    raw_system: Option<&'a RawValue>,
    #[serde(default, borrow)]
    extra: Option<&'a RawValue>,
    #[serde(default, borrow)]
    extra_headers: Option<&'a RawValue>,
    #[serde(default, borrow)]
    request_parameters: Option<&'a RawValue>,
    #[serde(default)]
    forks: Vec<ForkEntry>,
    #[serde(default)]
    spawns: Vec<SpawnEntry>,
    #[serde(default)]
    delay: f64,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
enum ForkEntry {
    Child(String),
    Config {
        child: String,
        #[serde(default)]
        background: bool,
    },
}

impl ForkEntry {
    fn child(&self) -> &str {
        match self {
            Self::Child(child) | Self::Config { child, .. } => child,
        }
    }

    fn background(&self) -> bool {
        matches!(
            self,
            Self::Config {
                background: true,
                ..
            }
        )
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
enum SpawnEntry {
    Child(String),
    Config {
        children: Vec<String>,
        #[serde(default)]
        join_at: Option<usize>,
    },
}

impl DagConversationRaw<'_> {
    fn validate(&self, origin: &impl std::fmt::Display) -> Result<()> {
        if self.session_id.is_empty() || self.turns.is_empty() {
            return Err(DatasetError::Validation(format!(
                "{origin}: DAG session_id and turns must be non-empty"
            )));
        }
        reject_duplicates(
            &self.pre_session_spawns,
            &format!("{origin}: pre_session_spawns"),
        )?;
        for (index, turn) in self.turns.iter().enumerate() {
            if turn.max_tokens == Some(0) {
                return Err(DatasetError::Validation(format!(
                    "{origin}: turn {index} max_tokens must be positive"
                )));
            }
            if !turn.delay.is_finite() || turn.delay < 0.0 {
                return Err(DatasetError::Validation(format!(
                    "{origin}: turn {index} delay must be finite and non-negative"
                )));
            }
            for (field, value) in [("model", &turn.model), ("endpoint", &turn.endpoint)] {
                if value.as_ref().is_some_and(|value| value.trim().is_empty()) {
                    return Err(DatasetError::Validation(format!(
                        "{origin}: turn {index} {field} must be non-empty when configured"
                    )));
                }
            }
            validate_messages(turn.messages, origin, index)?;
            validate_optional_array(turn.tools, "tools", origin, index)?;
            validate_optional_array(turn.raw_system, "raw_system", origin, index)?;
            validate_optional_object(turn.extra, "extra", origin, index)?;
            validate_optional_string_object(turn.extra_headers, "extra_headers", origin, index)?;
            validate_optional_string_object(
                turn.request_parameters,
                "request_parameters",
                origin,
                index,
            )?;
            let fork_children = turn
                .forks
                .iter()
                .map(|fork| fork.child().to_string())
                .collect::<Vec<_>>();
            reject_duplicates(&fork_children, &format!("{origin}: turn {index} forks"))?;
            for fork in &turn.forks {
                if fork.child().is_empty() {
                    return Err(DatasetError::Validation(format!(
                        "{origin}: turn {index} fork child cannot be empty"
                    )));
                }
            }
            let spawn_groups = spawn_groups(&turn.spawns)?;
            let mut all_spawn_children = Vec::new();
            for (children, join_at) in &spawn_groups {
                if children.is_empty() || children.iter().any(String::is_empty) {
                    return Err(DatasetError::Validation(format!(
                        "{origin}: turn {index} spawn children must be non-empty"
                    )));
                }
                reject_duplicates(children, &format!("{origin}: turn {index} spawn group"))?;
                if let Some(join_at) = join_at
                    && (*join_at <= index || *join_at >= self.turns.len())
                {
                    return Err(DatasetError::Validation(format!(
                        "{origin}: turn {index} spawn join_at={join_at} must be after the spawning turn and inside the conversation"
                    )));
                }
                all_spawn_children.extend(children.iter().cloned());
            }
            reject_duplicates(
                &all_spawn_children,
                &format!("{origin}: turn {index} spawn groups"),
            )?;
            if index + 1 < self.turns.len() && turn.forks.iter().any(|fork| !fork.background()) {
                return Err(DatasetError::Validation(format!(
                    "{origin}: turn {index} has a foreground fork before the terminal turn"
                )));
            }
        }
        Ok(())
    }

    fn into_program(self, origin: RowOrigin) -> Result<DagJsonlConversation> {
        let turns = self
            .turns
            .into_iter()
            .map(|turn| {
                let spawns = spawn_groups(&turn.spawns)?
                    .into_iter()
                    .map(|(children, join_at)| DagJsonlSpawn { children, join_at })
                    .collect();
                Ok(DagJsonlTurn {
                    messages: Bytes::copy_from_slice(turn.messages.get().as_bytes()),
                    model: turn.model,
                    endpoint: turn.endpoint,
                    streaming: turn.streaming,
                    max_tokens: turn.max_tokens,
                    tools: turn
                        .tools
                        .map(|value| Bytes::copy_from_slice(value.get().as_bytes())),
                    raw_system: turn
                        .raw_system
                        .map(|value| Bytes::copy_from_slice(value.get().as_bytes())),
                    extra: turn
                        .extra
                        .map(|value| Bytes::copy_from_slice(value.get().as_bytes())),
                    extra_headers: turn
                        .extra_headers
                        .map(|value| Bytes::copy_from_slice(value.get().as_bytes())),
                    request_parameters: turn
                        .request_parameters
                        .map(|value| Bytes::copy_from_slice(value.get().as_bytes())),
                    forks: turn
                        .forks
                        .into_iter()
                        .map(|fork| DagJsonlFork {
                            child: fork.child().to_string(),
                            background: fork.background(),
                        })
                        .collect(),
                    spawns,
                    delay_ms: turn.delay,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(DagJsonlConversation {
            origin,
            session_id: self.session_id,
            turns,
            pre_session_spawns: self.pre_session_spawns,
        })
    }
}

fn spawn_groups(entries: &[SpawnEntry]) -> Result<Vec<(Vec<String>, Option<usize>)>> {
    let mut groups = Vec::new();
    let mut bare = Vec::new();
    for entry in entries {
        match entry {
            SpawnEntry::Child(child) => bare.push(child.clone()),
            SpawnEntry::Config { children, join_at } => {
                if !bare.is_empty() {
                    groups.push((std::mem::take(&mut bare), None));
                }
                groups.push((children.clone(), *join_at));
            }
        }
    }
    if !bare.is_empty() {
        groups.push((bare, None));
    }
    Ok(groups)
}

fn reject_duplicates(values: &[String], context: &str) -> Result<()> {
    let mut seen = HashSet::new();
    if let Some(value) = values.iter().find(|value| !seen.insert(value.as_str())) {
        return Err(DatasetError::Validation(format!(
            "{context}: duplicate child {value:?}"
        )));
    }
    Ok(())
}

fn validate_messages(raw: &RawValue, origin: &impl std::fmt::Display, turn: usize) -> Result<()> {
    let messages: Value = serde_json::from_str(raw.get())?;
    let Some(messages) = messages.as_array() else {
        return Err(DatasetError::Validation(format!(
            "{origin}: turn {turn} messages must be an array"
        )));
    };
    if messages.is_empty()
        || messages.iter().any(|message| {
            !message.is_object() || !message.get("role").is_some_and(Value::is_string)
        })
    {
        return Err(DatasetError::Validation(format!(
            "{origin}: turn {turn} messages must be non-empty objects with roles"
        )));
    }
    Ok(())
}

fn validate_optional_array(
    raw: Option<&RawValue>,
    field: &str,
    origin: &impl std::fmt::Display,
    turn: usize,
) -> Result<()> {
    if let Some(raw) = raw
        && !serde_json::from_str::<Value>(raw.get())?.is_array()
    {
        return Err(DatasetError::Validation(format!(
            "{origin}: turn {turn} {field} must be an array"
        )));
    }
    Ok(())
}

fn validate_optional_object(
    raw: Option<&RawValue>,
    field: &str,
    origin: &impl std::fmt::Display,
    turn: usize,
) -> Result<()> {
    if let Some(raw) = raw
        && !serde_json::from_str::<Value>(raw.get())?.is_object()
    {
        return Err(DatasetError::Validation(format!(
            "{origin}: turn {turn} {field} must be an object"
        )));
    }
    Ok(())
}

fn validate_optional_string_object(
    raw: Option<&RawValue>,
    field: &str,
    origin: &impl std::fmt::Display,
    turn: usize,
) -> Result<()> {
    let Some(raw) = raw else { return Ok(()) };
    let value: Value = serde_json::from_str(raw.get())?;
    let Some(object) = value.as_object() else {
        return Err(DatasetError::Validation(format!(
            "{origin}: turn {turn} {field} must be an object"
        )));
    };
    if object.values().any(|value| !value.is_string()) {
        return Err(DatasetError::Validation(format!(
            "{origin}: turn {turn} {field} values must be strings"
        )));
    }
    Ok(())
}

fn input_tokens_excluding_tools(
    extracted: &ExtractedPayload,
    tokenizer: &dyn TextTokenizer,
) -> Result<u64> {
    let mut excluded = HashMap::<&str, usize>::new();
    for text in &extracted.tool_texts {
        *excluded.entry(text).or_default() += 1;
    }
    let mut count = extracted.pretokenised_token_count;
    for text in &extracted.texts {
        if let Some(remaining) = excluded.get_mut(text.as_str())
            && *remaining > 0
        {
            *remaining -= 1;
            continue;
        }
        count = add_token_count(count, text, tokenizer)?;
    }
    Ok(count)
}

fn add_token_count(count: u64, text: &str, tokenizer: &dyn TextTokenizer) -> Result<u64> {
    count
        .checked_add(tokenizer.encode(text)?.len() as u64)
        .ok_or_else(|| DatasetError::Validation("DAG token count overflowed u64".into()))
}

#[cfg(test)]
mod tests {
    use serde_json::{Value, json};

    use super::*;

    async fn program(value: Value) -> Result<DagJsonlProgram> {
        load_dag_jsonl_program(&LoadConfig::new(DatasetSource::Inline(value))).await
    }

    #[tokio::test]
    async fn parser_normalizes_forks_spawns_and_exact_turn_wires() {
        let parsed = program(json!([
            {
                "session_id":"root",
                "pre_session_spawns":["pre"],
                "turns":[{
                    "messages":[{"role":"user","content":"q"}],
                    "forks":[{"child":"fork","background":true}],
                    "spawns":[{"children":["spawn"],"join_at":1}]
                }, {"messages":[{"role":"user","content":"joined"}]}]
            },
            {"session_id":"fork","turns":[{"messages":[{"role":"user","content":"f"}]}]},
            {"session_id":"spawn","turns":[{"messages":[{"role":"user","content":"s"}]}]},
            {"session_id":"pre","turns":[{"messages":[{"role":"user","content":"p"}]}]}
        ]))
        .await
        .unwrap();
        let root = &parsed.conversations[0];
        assert_eq!(root.session_id, "root");
        assert_eq!(root.pre_session_spawns, ["pre"]);
        assert_eq!(root.turns[0].forks[0].child, "fork");
        assert!(root.turns[0].forks[0].background);
        assert_eq!(root.turns[0].spawns[0].children, ["spawn"]);
        assert_eq!(root.turns[0].spawns[0].join_at, Some(1));
        let messages: Value = serde_json::from_slice(&root.turns[0].messages).unwrap();
        assert_eq!(messages[0]["content"], "q");
    }

    #[tokio::test]
    async fn empty_and_locally_malformed_sources_fail_in_the_parser() {
        assert!(program(json!([])).await.is_err());
        assert!(
            program(json!([{"session_id":"x","turns":[{"messages":[]}]}]))
                .await
                .is_err()
        );
        assert!(
            program(json!([{
                "session_id":"x",
                "turns":[{"messages":[{"role":"user"}],"spawns":["a","a"]}]
            }]))
            .await
            .is_err()
        );
    }

    #[tokio::test]
    async fn row_cap_never_hides_an_invalid_dag_vertex() {
        let mut config = LoadConfig::new(DatasetSource::Inline(json!([
            {"session_id":"root","turns":[{"messages":[{"role":"user"}]}]},
            {"session_id":"hidden-by-a-row-cap","turns":[{"messages":[]}]}
        ])));
        config.max_rows = Some(1);
        assert!(load_dag_jsonl_program(&config).await.is_err());
    }
}
