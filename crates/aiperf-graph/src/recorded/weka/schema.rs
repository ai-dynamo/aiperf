// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Strict recursive WEKA trace schema and Pydantic-compatible scalar coercions.

use std::collections::HashSet;

use num_bigint::BigInt;
use serde_json::{Map, Value};

use crate::recorded::RecordedTraceError;

#[derive(Debug)]
pub(super) struct WekaTrace {
    pub id: String,
    pub block_size: usize,
    pub global_hash_scope: bool,
    pub requests: Vec<WekaEntry>,
}

#[derive(Debug)]
pub(super) enum WekaEntry {
    Leaf(WekaLeaf),
    Subagent(WekaSubagent),
}

#[derive(Debug)]
pub(super) struct WekaLeaf {
    pub start_seconds: f64,
    pub model: String,
    pub input_tokens: usize,
    pub output_tokens: usize,
    pub hashes: Vec<BigInt>,
    pub duration_seconds: f64,
    pub streaming: bool,
    pub ttft_seconds: Option<f64>,
}

#[derive(Debug)]
pub(super) struct WekaSubagent {
    pub agent_id: String,
    pub status: String,
    pub requests: Vec<WekaEntry>,
}

pub(super) fn parse_trace(value: Value) -> Result<WekaTrace, RecordedTraceError> {
    let object = into_object(value, "WEKA trace")?;
    reject_unknown(
        &object,
        &[
            "id",
            "models",
            "block_size",
            "hash_id_scope",
            "tool_tokens",
            "system_tokens",
            "requests",
            "totals",
        ],
        "WEKA trace",
    )?;
    let id = required_string(&object, "id", "WEKA trace")?;
    validate_scope_id(&id, "trace id")?;
    string_list(required(&object, "models", "WEKA trace")?, "models")?;
    let block_size = positive_usize(required(&object, "block_size", "WEKA trace")?, "block_size")?;
    let global_hash_scope = match required_string(&object, "hash_id_scope", "WEKA trace")?.as_str()
    {
        "local" => false,
        "global" => true,
        other => {
            return Err(RecordedTraceError(format!(
                "WEKA hash_id_scope must be \"local\" or \"global\", got {other:?}"
            )));
        }
    };
    optional_nonnegative(&object, "tool_tokens")?;
    optional_nonnegative(&object, "system_tokens")?;
    if let Some(totals) = object.get("totals")
        && !totals.is_null()
        && !totals.is_object()
    {
        return Err(RecordedTraceError(
            "WEKA totals must be an object or null".into(),
        ));
    }
    let requests = entry_list(required(&object, "requests", "WEKA trace")?, "requests")?;
    if requests.is_empty() {
        return Err(RecordedTraceError(
            "WEKA trace requests cannot be empty".into(),
        ));
    }
    let mut scopes = HashSet::new();
    scopes.insert(id.clone());
    validate_subagent_scopes(&requests, &mut scopes)?;
    Ok(WekaTrace {
        id,
        block_size,
        global_hash_scope,
        requests,
    })
}

fn entry_list(value: &Value, label: &str) -> Result<Vec<WekaEntry>, RecordedTraceError> {
    value
        .as_array()
        .ok_or_else(|| RecordedTraceError(format!("WEKA {label} must be a list")))?
        .iter()
        .enumerate()
        .map(|(index, value)| parse_entry(value.clone(), &format!("{label}[{index}]")))
        .collect()
}

fn parse_entry(value: Value, label: &str) -> Result<WekaEntry, RecordedTraceError> {
    let object = into_object(value, label)?;
    match required_string(&object, "type", label)?.as_str() {
        "n" => parse_leaf(object, label, false).map(WekaEntry::Leaf),
        "s" => parse_leaf(object, label, true).map(WekaEntry::Leaf),
        "subagent" => parse_subagent(object, label).map(WekaEntry::Subagent),
        other => Err(RecordedTraceError(format!(
            "WEKA {label}.type must be n, s, or subagent, got {other:?}"
        ))),
    }
}

fn parse_leaf(
    object: Map<String, Value>,
    label: &str,
    streaming: bool,
) -> Result<WekaLeaf, RecordedTraceError> {
    let mut allowed = vec![
        "t",
        "type",
        "model",
        "in",
        "out",
        "input_length",
        "output_length",
        "hash_ids",
        "input_types",
        "output_types",
        "stop",
        "api_time",
        "think_time",
    ];
    if streaming {
        allowed.push("ttft");
    }
    reject_unknown(&object, &allowed, label)?;
    let input = aliased_required(&object, "in", "input_length", label)?;
    let output = aliased_required(&object, "out", "output_length", label)?;
    let hashes = object
        .get("hash_ids")
        .map(|value| nonnegative_bigint_list(value, &format!("{label}.hash_ids")))
        .transpose()?
        .unwrap_or_default();
    string_list_default(&object, "input_types", label)?;
    string_list_default(&object, "output_types", label)?;
    if let Some(stop) = object.get("stop") {
        scalar_string(stop, &format!("{label}.stop"))?;
    }
    optional_finite_nonnegative(&object, "think_time", label)?;
    let duration_seconds = optional_finite_nonnegative(&object, "api_time", label)?.unwrap_or(0.0);
    let ttft_seconds = if streaming {
        optional_finite_nonnegative(&object, "ttft", label)?
    } else {
        None
    };
    Ok(WekaLeaf {
        start_seconds: finite_nonnegative(required(&object, "t", label)?, &format!("{label}.t"))?,
        model: required_string(&object, "model", label)?,
        input_tokens: nonnegative_usize(input, &format!("{label}.in"))?,
        output_tokens: nonnegative_usize(output, &format!("{label}.out"))?,
        hashes,
        duration_seconds,
        streaming,
        ttft_seconds,
    })
}

fn parse_subagent(
    object: Map<String, Value>,
    label: &str,
) -> Result<WekaSubagent, RecordedTraceError> {
    reject_unknown(
        &object,
        &[
            "t",
            "type",
            "agent_id",
            "subagent_type",
            "duration_ms",
            "total_tokens",
            "tool_use_count",
            "status",
            "requests",
            "models",
            "tool_tokens",
            "system_tokens",
        ],
        label,
    )?;
    finite_nonnegative(required(&object, "t", label)?, &format!("{label}.t"))?;
    required_string(&object, "subagent_type", label)?;
    required_string(&object, "status", label)?;
    string_list(
        required(&object, "models", label)?,
        &format!("{label}.models"),
    )?;
    for field in ["duration_ms", "total_tokens", "tool_use_count"] {
        optional_nullable_nonnegative(&object, field)?;
    }
    for field in ["tool_tokens", "system_tokens"] {
        optional_nonnegative(&object, field)?;
    }
    Ok(WekaSubagent {
        agent_id: required_string(&object, "agent_id", label)?,
        status: required_string(&object, "status", label)?,
        requests: entry_list(
            required(&object, "requests", label)?,
            &format!("{label}.requests"),
        )?,
    })
}

fn validate_subagent_scopes(
    entries: &[WekaEntry],
    scopes: &mut HashSet<String>,
) -> Result<(), RecordedTraceError> {
    for entry in entries {
        let WekaEntry::Subagent(subagent) = entry else {
            continue;
        };
        validate_scope_id(&subagent.agent_id, "subagent agent_id")?;
        if !scopes.insert(subagent.agent_id.clone()) {
            return Err(RecordedTraceError(format!(
                "WEKA scope id {:?} is duplicated or collides with the trace id",
                subagent.agent_id
            )));
        }
        validate_subagent_scopes(&subagent.requests, scopes)?;
    }
    Ok(())
}

fn validate_scope_id(value: &str, label: &str) -> Result<(), RecordedTraceError> {
    if value.is_empty() || value.contains(':') {
        return Err(RecordedTraceError(format!(
            "WEKA {label} must be non-empty and cannot contain ':'"
        )));
    }
    Ok(())
}

fn reject_unknown(
    object: &Map<String, Value>,
    allowed: &[&str],
    label: &str,
) -> Result<(), RecordedTraceError> {
    if let Some(key) = object.keys().find(|key| !allowed.contains(&key.as_str())) {
        return Err(RecordedTraceError(format!(
            "WEKA {label} contains unknown field {key:?}"
        )));
    }
    Ok(())
}

fn into_object(value: Value, label: &str) -> Result<Map<String, Value>, RecordedTraceError> {
    value
        .as_object()
        .cloned()
        .ok_or_else(|| RecordedTraceError(format!("WEKA {label} must be an object")))
}

fn required<'a>(
    object: &'a Map<String, Value>,
    field: &str,
    label: &str,
) -> Result<&'a Value, RecordedTraceError> {
    object
        .get(field)
        .ok_or_else(|| RecordedTraceError(format!("WEKA {label} is missing {field:?}")))
}

fn required_string(
    object: &Map<String, Value>,
    field: &str,
    label: &str,
) -> Result<String, RecordedTraceError> {
    scalar_string(required(object, field, label)?, &format!("{label}.{field}"))
}

fn scalar_string(value: &Value, label: &str) -> Result<String, RecordedTraceError> {
    value
        .as_str()
        .map(str::to_string)
        .ok_or_else(|| RecordedTraceError(format!("WEKA {label} must be a string")))
}

fn aliased_required<'a>(
    object: &'a Map<String, Value>,
    alias: &str,
    name: &str,
    label: &str,
) -> Result<&'a Value, RecordedTraceError> {
    match (object.get(alias), object.get(name)) {
        (Some(value), None) | (None, Some(value)) => Ok(value),
        (Some(_), Some(_)) => Err(RecordedTraceError(format!(
            "WEKA {label} cannot provide both {alias:?} and {name:?}"
        ))),
        (None, None) => Err(RecordedTraceError(format!(
            "WEKA {label} is missing {alias:?}"
        ))),
    }
}

fn string_list(value: &Value, label: &str) -> Result<(), RecordedTraceError> {
    let values = value
        .as_array()
        .ok_or_else(|| RecordedTraceError(format!("WEKA {label} must be a list")))?;
    if values.iter().any(|value| !value.is_string()) {
        return Err(RecordedTraceError(format!(
            "WEKA {label} entries must be strings"
        )));
    }
    Ok(())
}

fn string_list_default(
    object: &Map<String, Value>,
    field: &str,
    label: &str,
) -> Result<(), RecordedTraceError> {
    object
        .get(field)
        .map(|value| string_list(value, &format!("{label}.{field}")))
        .transpose()
        .map(drop)
}

fn nonnegative_bigint_list(value: &Value, label: &str) -> Result<Vec<BigInt>, RecordedTraceError> {
    value
        .as_array()
        .ok_or_else(|| RecordedTraceError(format!("WEKA {label} must be a list")))?
        .iter()
        .enumerate()
        .map(|(index, value)| {
            let integer = bigint(value, &format!("{label}[{index}]"))?;
            if integer < BigInt::from(0) {
                return Err(RecordedTraceError(format!(
                    "WEKA {label}[{index}] must be non-negative"
                )));
            }
            Ok(integer)
        })
        .collect()
}

fn bigint(value: &Value, label: &str) -> Result<BigInt, RecordedTraceError> {
    super::super::scalar::integer(value)
        .ok_or_else(|| RecordedTraceError(format!("WEKA {label} must be an integer")))
}

fn positive_usize(value: &Value, label: &str) -> Result<usize, RecordedTraceError> {
    let value = nonnegative_usize(value, label)?;
    if value == 0 {
        return Err(RecordedTraceError(format!("WEKA {label} must be positive")));
    }
    Ok(value)
}

fn nonnegative_usize(value: &Value, label: &str) -> Result<usize, RecordedTraceError> {
    bigint(value, label)?
        .try_into()
        .map_err(|_| RecordedTraceError(format!("WEKA {label} is outside usize range")))
}

fn optional_nonnegative(
    object: &Map<String, Value>,
    field: &str,
) -> Result<Option<usize>, RecordedTraceError> {
    object
        .get(field)
        .map(|value| nonnegative_usize(value, field))
        .transpose()
}

fn optional_nullable_nonnegative(
    object: &Map<String, Value>,
    field: &str,
) -> Result<Option<usize>, RecordedTraceError> {
    match object.get(field) {
        None | Some(Value::Null) => Ok(None),
        Some(value) => nonnegative_usize(value, field).map(Some),
    }
}

fn finite_nonnegative(value: &Value, label: &str) -> Result<f64, RecordedTraceError> {
    let parsed = super::super::scalar::float(value)
        .filter(|value| value.is_finite() && *value >= 0.0)
        .ok_or_else(|| {
            RecordedTraceError(format!("WEKA {label} must be finite and non-negative"))
        })?;
    Ok(parsed)
}

fn optional_finite_nonnegative(
    object: &Map<String, Value>,
    field: &str,
    label: &str,
) -> Result<Option<f64>, RecordedTraceError> {
    match object.get(field) {
        None | Some(Value::Null) => Ok(None),
        Some(value) => finite_nonnegative(value, &format!("{label}.{field}")).map(Some),
    }
}

#[cfg(test)]
mod tests {
    use num_bigint::BigInt;
    use serde_json::json;

    use super::*;

    #[test]
    fn schema_accepts_python_integer_coercions_nullable_subagent_stats_and_wide_hashes() {
        let value: Value = serde_json::from_str(
            r#"{
                "id":"trace","models":["m"],"block_size":"16.0",
                "hash_id_scope":"global","requests":[{
                    "t":false,"type":"subagent","agent_id":"child",
                    "subagent_type":"Explore","duration_ms":null,
                    "total_tokens":null,"tool_use_count":null,
                    "status":"async_launched","requests":[{
                        "t":"0.5","type":"s","model":"m","input_length":"16.0",
                        "output_length":true,
                        "hash_ids":[184467440737095516170],"ttft":"0.25"
                    }],"models":["m"]
                }]
            }"#,
        )
        .unwrap();
        let trace = parse_trace(value).unwrap();
        assert_eq!(trace.block_size, 16);
        let WekaEntry::Subagent(subagent) = &trace.requests[0] else {
            panic!("subagent")
        };
        assert_eq!(subagent.status, "async_launched");
        let WekaEntry::Leaf(leaf) = &subagent.requests[0] else {
            panic!("leaf")
        };
        assert_eq!(leaf.input_tokens, 16);
        assert_eq!(leaf.output_tokens, 1);
        assert_eq!(leaf.start_seconds, 0.5);
        assert_eq!(leaf.ttft_seconds, Some(0.25));
        assert_eq!(
            leaf.hashes,
            ["184467440737095516170".parse::<BigInt>().unwrap()]
        );
    }

    #[test]
    fn schema_rejects_unknown_fields_alias_conflicts_and_scope_collisions() {
        let base = json!({
            "id": "trace", "models": ["m"], "block_size": 16,
            "hash_id_scope": "local", "requests": [{
                "t": 0, "type": "n", "model": "m", "in": 16,
                "out": 1, "hash_ids": [1]
            }]
        });
        let mut unknown = base.clone();
        unknown["foreign"] = json!(true);
        assert!(
            parse_trace(unknown)
                .unwrap_err()
                .to_string()
                .contains("unknown field")
        );

        let mut aliases = base.clone();
        aliases["requests"][0]["input_length"] = json!(16);
        assert!(
            parse_trace(aliases)
                .unwrap_err()
                .to_string()
                .contains("cannot provide both")
        );

        let mut collision = base;
        collision["requests"] = json!([{
            "t": 0, "type": "subagent", "agent_id": "trace",
            "subagent_type": "x", "status": "completed", "requests": [], "models": []
        }]);
        assert!(
            parse_trace(collision)
                .unwrap_err()
                .to_string()
                .contains("collides")
        );
    }

    #[test]
    fn nullable_and_nonnullable_optional_integer_fields_remain_distinct() {
        let accepted = json!({
            "id": "trace", "models": ["m"], "block_size": 16,
            "hash_id_scope": "local", "requests": [{
                "t": 0, "type": "subagent", "agent_id": "child",
                "subagent_type": "x", "status": "async_launched",
                "duration_ms": null, "total_tokens": null, "tool_use_count": null,
                "requests": [{"t":0,"type":"n","model":"m","in":1,"out":1}],
                "models": []
            }]
        });
        parse_trace(accepted.clone()).unwrap();
        let mut rejected = accepted;
        rejected["requests"][0]["tool_tokens"] = Value::Null;
        assert!(parse_trace(rejected).is_err());
    }
}
