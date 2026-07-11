// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Channel reducers for `overwrite` and `add_messages` channels.
//!
//! [`ChanVal::Unset`] is the sentinel for a channel that has never been written
//! (distinct from a written JSON `null`).

use crate::model::ReducerName;
use serde::{Serialize, Serializer};
use serde_json::Value;

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
/// Maintains an `id -> index` association as a `Vec<(id, index)>` scanned
/// linearly (via `get_index`/`set_index`): a new message whose `id` matches
/// an existing one overwrites in place (last-index-wins on duplicate ids),
/// otherwise appends. Each id lookup/update is O(k) in the number of distinct
/// ids seen so far, so building the result is O(n·k) — fine for the small
/// message counts on this replay path; the ordering is what must stay exact.
pub fn add_messages_reducer(current: &ChanVal, writes: &[Write]) -> Result<ChanVal, ReducerError> {
    let mut acc: Vec<Value> = match current {
        ChanVal::Val(Value::Array(a)) => a.clone(),
        ChanVal::Unset => Vec::new(),
        // A messages channel only ever holds an array or Unset; a scalar can't
        // occur here, so start from empty for that unreached case.
        ChanVal::Val(_) => Vec::new(),
    };
    // id -> index over the existing accumulator (last occurrence wins).
    let mut id_index: Vec<(Value, usize)> = Vec::new();
    for (i, msg) in acc.iter().enumerate() {
        if let Some(id) = message_id(msg) {
            set_index(&mut id_index, id.clone(), i);
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
                if let Some(idx) = get_index(&id_index, id) {
                    acc[idx] = msg.clone();
                    continue;
                }
                set_index(&mut id_index, id.clone(), acc.len());
            }
            acc.push(msg.clone());
        }
    }
    Ok(ChanVal::Val(Value::Array(acc)))
}

fn get_index(index: &[(Value, usize)], id: &Value) -> Option<usize> {
    index.iter().rev().find(|(k, _)| k == id).map(|(_, i)| *i)
}

fn set_index(index: &mut Vec<(Value, usize)>, id: Value, i: usize) {
    if let Some(slot) = index.iter_mut().find(|(k, _)| *k == id) {
        slot.1 = i;
    } else {
        index.push((id, i));
    }
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
