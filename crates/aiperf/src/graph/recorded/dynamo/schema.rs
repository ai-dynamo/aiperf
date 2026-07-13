// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Typed Dynamo request-trace record schema with ignored forward fields.

use num_bigint::BigInt;
use serde_json::value::RawValue;
use serde_json::{Map, Value};

use crate::graph::recorded::RecordedTraceError;

/// Untouched hash skeleton parsed alongside the [`Value`] tree.
///
/// Dynamo `input_sequence_hashes` routinely exceed `u64::MAX`, so they are
/// captured as raw JSON tokens and coerced through
/// `scalar::hash_i128_from_raw_text` rather than the lossy `Value` number path.
/// Both the direct record shape (`request.replay`) and the sink-envelope shape
/// (`event.request.replay`) are covered so the extraction matches
/// [`unwrap_sink_envelope`].
#[derive(serde::Deserialize)]
struct DynamoReplayRaw {
    #[serde(default, rename = "input_sequence_hashes")]
    hashes: Vec<Box<RawValue>>,
}

#[derive(serde::Deserialize)]
struct DynamoRequestRaw {
    replay: Option<DynamoReplayRaw>,
}

#[derive(serde::Deserialize)]
struct DynamoEventRaw {
    request: Option<DynamoRequestRaw>,
}

#[derive(serde::Deserialize)]
struct DynamoRecordRaw {
    request: Option<DynamoRequestRaw>,
    event: Option<DynamoEventRaw>,
}

/// Pull the raw `input_sequence_hashes` tokens out of an untouched record,
/// preferring the direct `request` and falling back to the sink-envelope's
/// `event.request` exactly as [`unwrap_sink_envelope`] resolves the `Value` side.
fn extract_raw_hashes(raw: &RawValue) -> Result<Vec<Box<RawValue>>, RecordedTraceError> {
    let record: DynamoRecordRaw = serde_json::from_str(raw.get()).map_err(|error| {
        RecordedTraceError(format!(
            "Dynamo trace record has invalid structure: {error}"
        ))
    })?;
    let replay = record
        .request
        .or_else(|| record.event.and_then(|event| event.request))
        .and_then(|request| request.replay);
    Ok(replay.map(|replay| replay.hashes).unwrap_or_default())
}

#[derive(Debug, Clone)]
pub(super) struct TraceRecord {
    pub source_order: usize,
    pub event_type: EventType,
    pub event_time_ms: i64,
    pub context: Option<AgentContext>,
    pub request: Option<RequestMetrics>,
    pub tool: Option<ToolEvent>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(super) enum EventType {
    RequestEnd,
    ToolStart,
    ToolEnd,
    ToolError,
}

#[derive(Debug, Clone)]
pub(super) struct AgentContext {
    pub session_id: String,
    pub parent_session_id: Option<String>,
    pub parent_trajectory_id: Option<String>,
}

#[derive(Debug, Clone)]
pub(super) struct RequestMetrics {
    pub request_id: String,
    pub model: Option<String>,
    pub input_tokens: Option<i64>,
    pub output_tokens: Option<i64>,
    pub cached_tokens: Option<i64>,
    pub request_received_ms: Option<i64>,
    pub total_time_ms: Option<f64>,
    pub ttft_ms: Option<f64>,
    pub replay: Option<ReplayMetrics>,
}

#[derive(Debug, Clone)]
pub(super) struct ReplayMetrics {
    pub block_size: usize,
    pub input_length: i64,
    pub hashes: Vec<i128>,
}

#[derive(Debug, Clone)]
pub(super) struct ToolEvent {
    pub tool_call_id: String,
    pub status: Option<String>,
}

pub(super) fn unwrap_sink_envelope(value: Value) -> Value {
    match value {
        Value::Object(mut object) if !object.contains_key("schema") => object
            .remove("event")
            .filter(Value::is_object)
            .unwrap_or(Value::Object(object)),
        value => value,
    }
}

pub(super) fn parse_record(
    raw: &RawValue,
    source_order: usize,
) -> Result<Option<TraceRecord>, RecordedTraceError> {
    let value: Value = serde_json::from_str(raw.get()).map_err(|error| {
        RecordedTraceError(format!(
            "Dynamo trace line {} is invalid JSON: {error}",
            source_order + 1
        ))
    })?;
    let raw_hashes = extract_raw_hashes(raw)?;
    let value = unwrap_sink_envelope(value);
    let Some(object) = value.as_object() else {
        return Err(RecordedTraceError(format!(
            "Dynamo trace line {} must be an object",
            source_order + 1
        )));
    };
    let Some(schema) = object.get("schema") else {
        return Ok(None);
    };
    if string(schema, "schema")? != "dynamo.request.trace.v1" {
        return Err(RecordedTraceError(format!(
            "Dynamo trace line {} has unsupported schema {:?}",
            source_order + 1,
            schema
        )));
    }
    let event_type = match string(required(object, "event_type")?, "event_type")?.as_str() {
        "request_end" => EventType::RequestEnd,
        "tool_start" => EventType::ToolStart,
        "tool_end" => EventType::ToolEnd,
        "tool_error" => EventType::ToolError,
        other => {
            return Err(RecordedTraceError(format!(
                "Dynamo event_type {other:?} is not supported"
            )));
        }
    };
    if let Some(source) = object.get("event_source")
        && !source.is_null()
    {
        let source = string(source, "event_source")?;
        if source != "dynamo" && source != "harness" {
            return Err(RecordedTraceError(format!(
                "Dynamo event_source must be dynamo or harness, got {source:?}"
            )));
        }
    }
    let context = object
        .get("agent_context")
        .filter(|value| !value.is_null())
        .map(parse_context)
        .transpose()?;
    let request = object
        .get("request")
        .filter(|value| !value.is_null())
        .map(|value| parse_request(value, &raw_hashes))
        .transpose()?;
    let tool = object
        .get("tool")
        .filter(|value| !value.is_null())
        .map(parse_tool)
        .transpose()?;
    Ok(Some(TraceRecord {
        source_order,
        event_type,
        event_time_ms: integer_i64(
            required(object, "event_time_unix_ms")?,
            "event_time_unix_ms",
        )?,
        context,
        request,
        tool,
    }))
}

fn parse_context(value: &Value) -> Result<AgentContext, RecordedTraceError> {
    let object = value
        .as_object()
        .ok_or_else(|| RecordedTraceError("Dynamo agent_context must be an object".into()))?;
    optional_string(object, "trajectory_id")?;
    optional_string(object, "session_type_id")?;
    Ok(AgentContext {
        session_id: string(required(object, "session_id")?, "agent_context.session_id")?,
        parent_session_id: optional_string(object, "parent_session_id")?,
        parent_trajectory_id: optional_string(object, "parent_trajectory_id")?,
    })
}

fn parse_request(
    value: &Value,
    raw_hashes: &[Box<RawValue>],
) -> Result<RequestMetrics, RecordedTraceError> {
    let object = value
        .as_object()
        .ok_or_else(|| RecordedTraceError("Dynamo request must be an object".into()))?;
    optional_string(object, "x_request_id")?;
    optional_float(object, "prefill_wait_time_ms", false)?;
    optional_float(object, "prefill_time_ms", false)?;
    let total_time_ms = optional_float(object, "total_time_ms", false)?;
    let ttft_ms = optional_float(object, "ttft_ms", true)?;
    optional_float(object, "avg_itl_ms", true)?;
    optional_float(object, "kv_hit_rate", false)?;
    optional_float(object, "kv_transfer_estimated_latency_ms", true)?;
    optional_integer(object, "queue_depth")?;
    if let Some(worker) = object.get("worker").filter(|value| !value.is_null()) {
        parse_worker(worker)?;
    }
    Ok(RequestMetrics {
        request_id: string(required(object, "request_id")?, "request.request_id")?,
        model: optional_string(object, "model")?,
        input_tokens: optional_i64(object, "input_tokens")?,
        output_tokens: optional_i64(object, "output_tokens")?,
        cached_tokens: optional_i64(object, "cached_tokens")?,
        request_received_ms: optional_i64(object, "request_received_ms")?,
        total_time_ms,
        ttft_ms,
        replay: object
            .get("replay")
            .filter(|value| !value.is_null())
            .map(|value| parse_replay(value, raw_hashes))
            .transpose()?,
    })
}

fn parse_replay(
    value: &Value,
    raw_hashes: &[Box<RawValue>],
) -> Result<ReplayMetrics, RecordedTraceError> {
    let object = value
        .as_object()
        .ok_or_else(|| RecordedTraceError("Dynamo request.replay must be an object".into()))?;
    let block_size = usize_value(
        required(object, "trace_block_size")?,
        "request.replay.trace_block_size",
    )?;
    if block_size == 0 {
        return Err(RecordedTraceError(
            "Dynamo replay trace_block_size must be positive".into(),
        ));
    }
    let input_length = integer_i64(
        required(object, "input_length")?,
        "request.replay.input_length",
    )?;
    let values = required(object, "input_sequence_hashes")?
        .as_array()
        .ok_or_else(|| {
            RecordedTraceError("Dynamo replay input_sequence_hashes must be a list".into())
        })?;
    // The raw skeleton reads the same `input_sequence_hashes` array, so lengths
    // match; guard rather than index blindly. The raw tokens preserve digits past
    // `u64::MAX` that the `Value` array above has already coerced through `f64`.
    if values.len() != raw_hashes.len() {
        return Err(RecordedTraceError(format!(
            "Dynamo replay input_sequence_hashes raw/value counts diverged ({} vs {})",
            raw_hashes.len(),
            values.len()
        )));
    }
    let hashes = raw_hashes
        .iter()
        .enumerate()
        .map(|(index, raw)| {
            let hash =
                super::super::scalar::hash_i128_from_raw_text(raw.get()).ok_or_else(|| {
                    RecordedTraceError(format!(
                        "Dynamo request.replay.input_sequence_hashes[{index}] must be an integer"
                    ))
                })?;
            if hash < 0 {
                return Err(RecordedTraceError(
                    "Dynamo recorded replay hashes must be non-negative".into(),
                ));
            }
            Ok(hash)
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(ReplayMetrics {
        block_size,
        input_length,
        hashes,
    })
}

fn parse_tool(value: &Value) -> Result<ToolEvent, RecordedTraceError> {
    let object = value
        .as_object()
        .ok_or_else(|| RecordedTraceError("Dynamo tool must be an object".into()))?;
    let tool_call_id = string(required(object, "tool_call_id")?, "tool.tool_call_id")?;
    string(required(object, "tool_class")?, "tool.tool_class")?;
    let status = optional_string(object, "status")?;
    if status.as_deref().is_some_and(|status| {
        !matches!(
            status,
            "running"
                | "succeeded"
                | "ok"
                | "success"
                | "error"
                | "failed"
                | "cancelled"
                | "canceled"
                | "timeout"
        )
    }) {
        return Err(RecordedTraceError(format!(
            "Dynamo tool.status has unsupported value {:?}",
            status.as_deref().unwrap_or_default()
        )));
    }
    optional_float(object, "duration_ms", false)?;
    Ok(ToolEvent {
        tool_call_id,
        status,
    })
}

fn parse_worker(value: &Value) -> Result<(), RecordedTraceError> {
    let object = value
        .as_object()
        .ok_or_else(|| RecordedTraceError("Dynamo request.worker must be an object".into()))?;
    for field in [
        "prefill_worker_id",
        "prefill_dp_rank",
        "decode_worker_id",
        "decode_dp_rank",
    ] {
        optional_integer(object, field)?;
    }
    Ok(())
}

fn required<'a>(
    object: &'a Map<String, Value>,
    field: &str,
) -> Result<&'a Value, RecordedTraceError> {
    object
        .get(field)
        .ok_or_else(|| RecordedTraceError(format!("Dynamo record is missing {field:?}")))
}

fn string(value: &Value, label: &str) -> Result<String, RecordedTraceError> {
    value
        .as_str()
        .map(str::to_string)
        .ok_or_else(|| RecordedTraceError(format!("Dynamo {label} must be a string")))
}

fn optional_string(
    object: &Map<String, Value>,
    field: &str,
) -> Result<Option<String>, RecordedTraceError> {
    match object.get(field) {
        None | Some(Value::Null) => Ok(None),
        Some(value) => string(value, field).map(Some),
    }
}

fn optional_i64(
    object: &Map<String, Value>,
    field: &str,
) -> Result<Option<i64>, RecordedTraceError> {
    match object.get(field) {
        None | Some(Value::Null) => Ok(None),
        Some(value) => integer_i64(value, field).map(Some),
    }
}

fn optional_integer(
    object: &Map<String, Value>,
    field: &str,
) -> Result<Option<BigInt>, RecordedTraceError> {
    match object.get(field) {
        None | Some(Value::Null) => Ok(None),
        Some(value) => bigint(value, field).map(Some),
    }
}

fn optional_float(
    object: &Map<String, Value>,
    field: &str,
    require_finite: bool,
) -> Result<Option<f64>, RecordedTraceError> {
    let Some(value) = object.get(field) else {
        return Ok(None);
    };
    if value.is_null() {
        return Ok(None);
    }
    let value = super::super::scalar::float(value)
        .ok_or_else(|| RecordedTraceError(format!("Dynamo {field} must be a number")))?;
    if require_finite && !value.is_finite() {
        return Err(RecordedTraceError(format!("Dynamo {field} must be finite")));
    }
    Ok(Some(value))
}

fn usize_value(value: &Value, label: &str) -> Result<usize, RecordedTraceError> {
    bigint(value, label)?
        .try_into()
        .map_err(|_| RecordedTraceError(format!("Dynamo {label} is outside usize range")))
}

fn integer_i64(value: &Value, label: &str) -> Result<i64, RecordedTraceError> {
    let integer = bigint(value, label)?;
    integer
        .try_into()
        .map_err(|_| RecordedTraceError(format!("Dynamo {label} is outside i64 range")))
}

fn bigint(value: &Value, label: &str) -> Result<BigInt, RecordedTraceError> {
    super::super::scalar::integer(value)
        .ok_or_else(|| RecordedTraceError(format!("Dynamo {label} must be an integer")))
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    /// Re-serialize a `json!`-built `Value` into a raw token for `parse_record`.
    /// Only safe for fixtures whose hashes already fit through `Value` decoding.
    fn raw(value: &Value) -> Box<RawValue> {
        serde_json::value::to_raw_value(value).unwrap()
    }

    #[test]
    fn current_schema_accepts_envelopes_extras_coercions_and_u64_hash_domain() {
        // Build the RawValue from text so the >u64::MAX hash keeps every digit.
        let raw = RawValue::from_string(
            r#"{
                "timestamp": 1,
                "event": {
                    "schema":"dynamo.request.trace.v1",
                    "event_type":"request_end",
                    "event_time_unix_ms":"1000.0",
                    "event_source":"dynamo",
                    "forward_compatible": {"ignored": true},
                    "agent_context":{"session_id":"s","future":"ok"},
                    "request":{
                        "request_id":"r","input_tokens":"16.0","output_tokens":true,
                        "queue_depth":184467440737095516170,
                        "worker":{"decode_worker_id":184467440737095516170},
                        "replay":{
                            "trace_block_size":"16.0","input_length":"16.0",
                            "input_sequence_hashes":[184467440737095516170],
                            "future":"ok"
                        }
                    }
                }
            }"#
            .to_string(),
        )
        .unwrap();
        let record = parse_record(&raw, 0).unwrap().unwrap();
        assert_eq!(record.event_time_ms, 1000);
        let request = record.request.unwrap();
        assert_eq!(request.input_tokens, Some(16));
        assert_eq!(request.output_tokens, Some(1));
        assert_eq!(request.replay.as_ref().unwrap().block_size, 16);
        assert_eq!(
            request.replay.unwrap().hashes,
            ["184467440737095516170".parse::<i128>().unwrap()]
        );
    }

    #[test]
    fn marker_lines_skip_and_replay_only_records_remain_typed() {
        assert!(
            parse_record(&raw(&json!({"verification": "trace-s3-uploader"})), 0)
                .unwrap()
                .is_none()
        );
        let record = parse_record(
            &raw(&json!({
                "schema": "dynamo.request.trace.v1",
                "event_type": "request_end",
                "event_time_unix_ms": 1,
                "request": {
                    "request_id": "r",
                    "replay": {
                        "trace_block_size": 2,
                        "input_length": 4,
                        "input_sequence_hashes": [1, 2]
                    }
                }
            })),
            1,
        )
        .unwrap()
        .unwrap();
        assert!(record.context.is_none());
        assert!(record.request.is_some());
    }

    #[test]
    fn tool_status_aliases_are_lossless_and_invalid_status_fails() {
        for status in [
            "running",
            "succeeded",
            "ok",
            "success",
            "error",
            "failed",
            "cancelled",
            "canceled",
            "timeout",
        ] {
            let record = parse_record(
                &raw(&json!({
                    "schema": "dynamo.request.trace.v1",
                    "event_type": "tool_end",
                    "event_time_unix_ms": 1,
                    "agent_context": {"session_id": "s"},
                    "tool": {"tool_call_id": "c", "tool_class": "shell", "status": status}
                })),
                0,
            )
            .unwrap()
            .unwrap();
            assert_eq!(record.tool.unwrap().status.as_deref(), Some(status));
        }
        let error = parse_record(
            &raw(&json!({
                "schema": "dynamo.request.trace.v1",
                "event_type": "tool_error",
                "event_time_unix_ms": 1,
                "tool": {"tool_call_id": "c", "tool_class": "shell", "status": "unknown"}
            })),
            0,
        )
        .unwrap_err();
        assert!(error.to_string().contains("unsupported value"));
    }

    #[test]
    fn finite_metric_fields_and_recorded_hash_namespace_fail_closed() {
        let invalid_ttft = json!({
            "schema": "dynamo.request.trace.v1", "event_type": "request_end",
            "event_time_unix_ms": 1,
            "request": {"request_id": "r", "ttft_ms": "nan"}
        });
        assert!(
            parse_record(&raw(&invalid_ttft), 0)
                .unwrap_err()
                .to_string()
                .contains("finite")
        );

        let negative_hash = json!({
            "schema": "dynamo.request.trace.v1", "event_type": "request_end",
            "event_time_unix_ms": 1,
            "request": {"request_id": "r", "replay": {
                "trace_block_size": 16, "input_length": 1,
                "input_sequence_hashes": [-1]
            }}
        });
        assert!(
            parse_record(&raw(&negative_hash), 0)
                .unwrap_err()
                .to_string()
                .contains("non-negative")
        );
    }

    #[test]
    fn declared_optional_context_fields_are_validated_even_when_not_retained() {
        let invalid = json!({
            "schema": "dynamo.request.trace.v1",
            "event_type": "request_end",
            "event_time_unix_ms": 1,
            "agent_context": {
                "session_id": "s",
                "trajectory_id": 7,
                "session_type_id": "agent"
            }
        });
        assert!(
            parse_record(&raw(&invalid), 0)
                .unwrap_err()
                .to_string()
                .contains("trajectory_id must be a string")
        );
    }
}
