// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Channel reducers for `overwrite` and `add_messages` channels.
//!
//! [`ChanVal::Unset`] is the sentinel for a channel that has never been written
//! (distinct from a written JSON `null`).

use crate::model::ReducerName;
use serde::{Serialize, Serializer};
use serde_json::Value;
use std::collections::HashMap;

/// A channel value: either the never-written sentinel or a concrete JSON value.
#[derive(Debug, Clone, PartialEq)]
pub enum ChanVal {
    /// Never written.
    Unset,
    /// A concrete value; stored by clone, never mutated in place.
    Val(Value),
}

impl Serialize for ChanVal {
    /// `Unset` serializes to the `{"$unset": true}` sentinel so a never-written
    /// channel round-trips distinctly from a written `null`; `Val` serializes
    /// verbatim.
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        match self {
            ChanVal::Unset => serde_json::json!({ "$unset": true }).serialize(s),
            ChanVal::Val(v) => v.serialize(s),
        }
    }
}

impl ChanVal {
    /// The underlying `Value`, or `None` when unset.
    pub fn as_value(&self) -> Option<&Value> {
        match self {
            ChanVal::Unset => None,
            ChanVal::Val(v) => Some(v),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReducerError {
    /// Two or more nodes wrote an overwrite-typed channel concurrently.
    OverwriteConflict(String),
    /// `add_messages` received a non-list write value.
    NonListMessages { writer_id: String, got: String },
    /// A reducer name is unknown.
    UnknownReducer(String),
}

impl std::fmt::Display for ReducerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ReducerError::OverwriteConflict(ids) => write!(
                f,
                "overwrite-typed channel written by multiple nodes concurrently: {ids}"
            ),
            ReducerError::NonListMessages { writer_id, got } => write!(
                f,
                "add_messages reducer expected a list for writer '{writer_id}', got {got}"
            ),
            ReducerError::UnknownReducer(name) => write!(f, "unknown reducer '{name}'"),
        }
    }
}

impl std::error::Error for ReducerError {}

/// One committed write: `(writer_node_id, value)`.
pub type Write = (String, Value);

/// Return the single writer's value; reject concurrent multi-writer.
pub fn overwrite_reducer(current: &ChanVal, writes: &[Write]) -> Result<ChanVal, ReducerError> {
    if writes.is_empty() {
        return Ok(current.clone());
    }
    if writes.len() > 1 {
        let ids = writes
            .iter()
            .map(|(id, _)| id.as_str())
            .collect::<Vec<_>>()
            .join(", ");
        return Err(ReducerError::OverwriteConflict(ids));
    }
    Ok(ChanVal::Val(writes[0].1.clone()))
}

/// The JSON type name for a value, used in the non-array `add_messages` error.
fn value_type_name(v: &Value) -> &'static str {
    match v {
        Value::Null => "null",
        Value::Bool(_) => "bool",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

/// Extract a message's replacement key: `msg["id"]` when present and non-null.
fn message_id(msg: &Value) -> Option<&Value> {
    let obj = msg.as_object()?;
    match obj.get("id") {
        Some(Value::Null) | None => None,
        Some(id) => Some(id),
    }
}

/// Append writer values; replace prior messages whose `id` matches a new message.
///
/// Maintains an `id -> index` map keyed by each id's canonical JSON form
/// (see [`id_key`]), so every lookup and update is O(1): a new message whose
/// `id` matches an existing one overwrites in place (last write to that id
/// wins, keeping the original position), otherwise it appends in encounter
/// order. Building the result is therefore O(n) in the total message count.
/// The ordering and replacement semantics are identical to a linear scan.
pub fn add_messages_reducer(current: &ChanVal, writes: &[Write]) -> Result<ChanVal, ReducerError> {
    let mut acc: Vec<Value> = match current {
        ChanVal::Val(Value::Array(a)) => a.clone(),
        ChanVal::Unset => Vec::new(),
        // A messages channel only ever holds an array or Unset; a scalar can't
        // occur here, so start from empty for that unreached case.
        ChanVal::Val(_) => Vec::new(),
    };
    // Canonical id form -> index over the accumulator (last write to an id wins).
    let mut id_index: HashMap<String, usize> = HashMap::new();
    for (i, msg) in acc.iter().enumerate() {
        if let Some(id) = message_id(msg) {
            id_index.insert(id_key(id), i);
        }
    }
    for (writer_id, value) in writes {
        let list = match value {
            Value::Array(items) => items,
            other => {
                return Err(ReducerError::NonListMessages {
                    writer_id: writer_id.clone(),
                    got: value_type_name(other).to_string(),
                });
            }
        };
        for msg in list {
            if let Some(id) = message_id(msg) {
                let key = id_key(id);
                if let Some(&idx) = id_index.get(&key) {
                    acc[idx] = msg.clone();
                    continue;
                }
                id_index.insert(key, acc.len());
            }
            acc.push(msg.clone());
        }
    }
    Ok(ChanVal::Val(Value::Array(acc)))
}

/// Canonical, hashable form of a message id: its compact JSON encoding.
///
/// `Value` is not `Hash` (it can hold floats), so we key the index on the
/// compact JSON string instead. Structurally-equal ids encode to the same
/// string and distinct ids encode to distinct strings for the scalar/array/
/// object ids that occur as message ids, so this reproduces `Value` equality
/// (the linear-scan `k == id` test) exactly for those inputs.
fn id_key(id: &Value) -> String {
    id.to_string()
}

/// Look up a reducer by name and apply it.
pub fn apply_reducer(
    name: ReducerName,
    current: &ChanVal,
    writes: &[Write],
) -> Result<ChanVal, ReducerError> {
    match name {
        ReducerName::Overwrite => overwrite_reducer(current, writes),
        ReducerName::AddMessages => add_messages_reducer(current, writes),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn overwrite_empty_returns_current() {
        assert_eq!(
            overwrite_reducer(&ChanVal::Val(json!("a")), &[]).unwrap(),
            ChanVal::Val(json!("a"))
        );
        assert_eq!(
            overwrite_reducer(&ChanVal::Unset, &[]).unwrap(),
            ChanVal::Unset
        );
    }

    #[test]
    fn overwrite_single_and_conflict() {
        assert_eq!(
            overwrite_reducer(&ChanVal::Unset, &[("n0".into(), json!("x"))]).unwrap(),
            ChanVal::Val(json!("x"))
        );
        let err = overwrite_reducer(
            &ChanVal::Unset,
            &[("n0".into(), json!("x")), ("n1".into(), json!("y"))],
        )
        .unwrap_err();
        assert_eq!(err, ReducerError::OverwriteConflict("n0, n1".into()));
    }

    #[test]
    fn add_messages_append() {
        let out = add_messages_reducer(
            &ChanVal::Unset,
            &[("n0".into(), json!([{"role": "user", "content": "hi"}]))],
        )
        .unwrap();
        assert_eq!(
            out,
            ChanVal::Val(json!([{"role": "user", "content": "hi"}]))
        );
    }

    #[test]
    fn add_messages_id_replace() {
        let current = ChanVal::Val(json!([{"id": "m1", "content": "old"}]));
        let out = add_messages_reducer(
            &current,
            &[(
                "n0".into(),
                json!([{"id": "m1", "content": "new"}, {"content": "plain"}]),
            )],
        )
        .unwrap();
        assert_eq!(
            out,
            ChanVal::Val(json!([{"id": "m1", "content": "new"}, {"content": "plain"}]))
        );
    }

    #[test]
    fn add_messages_repeated_ids_preserve_order() {
        // Existing accumulator carries ids "a" and "b" at positions 0 and 1.
        let current = ChanVal::Val(json!([{"id": "a", "v": 1}, {"id": "b", "v": 1}]));
        let out = add_messages_reducer(
            &current,
            &[
                (
                    "n0".into(),
                    json!([
                        {"id": "b", "v": 2},   // replace-in-place at index 1
                        {"id": "c", "v": 1},   // new id, appends at index 2
                        {"content": "plain"}   // no id, appends at index 3
                    ]),
                ),
                (
                    "n1".into(),
                    json!([
                        {"id": "a", "v": 3},   // replace-in-place at index 0
                        {"id": "c", "v": 2},   // replace-in-place at index 2
                        {"id": "d", "v": 1},   // new id, appends at index 4
                        {"id": "c", "v": 3}    // replace-in-place at index 2 again
                    ]),
                ),
            ],
        )
        .unwrap();
        // Explicitly encoded expected order: id-based replacements keep their
        // original position; new ids and id-less messages append in encounter
        // order; last write to a given id wins its slot.
        assert_eq!(
            out,
            ChanVal::Val(json!([
                {"id": "a", "v": 3},
                {"id": "b", "v": 2},
                {"id": "c", "v": 3},
                {"content": "plain"},
                {"id": "d", "v": 1}
            ]))
        );
    }

    #[test]
    fn add_messages_non_list_errors() {
        let err =
            add_messages_reducer(&ChanVal::Unset, &[("n0".into(), json!("bad"))]).unwrap_err();
        assert_eq!(
            err,
            ReducerError::NonListMessages {
                writer_id: "n0".into(),
                got: "string".into()
            }
        );
    }
}
