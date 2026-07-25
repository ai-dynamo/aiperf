// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Raw per-record export serialization for the AgentX legacy path.
//!
//! Assembles the byte-exact reconstruction ([`crate::agentx::loader`]) — content
//! (`raw_messages`), per-turn timing (`timestamp_ms`/`delay_ms`/`api_time_ms`),
//! provenance, prefix-cache tallies, and the agentic-replay dispatch schedule
//! ([`crate::agentx::trajectory_source::replay_schedule`]) — into the
//! `export.records` raw record shape: one record per reconstructed turn carrying
//! the exact wire timing and content. This is the export-level artifact whose
//! byte-exactness the real-corpus + schedule e2es establish field-by-field.

use serde_json::{json, Value};

use crate::agentx::loader::ReconstructedConversation;
use crate::agentx::synth::ChatMessage;
use crate::agentx::trajectory_source::{replay_schedule, ReplayPhase};

fn messages_json(messages: &[ChatMessage]) -> Vec<Value> {
    messages
        .iter()
        .map(|m| {
            let mut obj = json!({ "role": m.role, "content": m.content });
            if let Some(tc) = &m.tool_calls {
                obj["tool_calls"] = Value::Array(
                    tc.iter()
                        .map(|c| {
                            json!({
                                "id": c.id,
                                "type": "function",
                                "function": { "name": c.name, "arguments": c.arguments },
                            })
                        })
                        .collect(),
                );
            }
            if let Some(id) = &m.tool_call_id {
                obj["tool_call_id"] = json!(id);
            }
            obj
        })
        .collect()
}

fn phase_str(phase: ReplayPhase) -> &'static str {
    match phase {
        ReplayPhase::History => "history",
        ReplayPhase::Warmup => "warmup",
        ReplayPhase::Profiling => "profiling",
    }
}

/// Serialize one reconstructed conversation into raw per-turn export records.
///
/// When `t_star_ms` is provided, each record is annotated with its agentic-replay
/// `phase` and PROFILING `dispatch_offset_ms` computed from the turns'
/// `timestamp_ms` via [`replay_schedule`] (the execution-order timing). When
/// `None`, only the recorded per-turn timing is emitted.
pub fn raw_export_records(conv: &ReconstructedConversation, t_star_ms: Option<f64>) -> Vec<Value> {
    let schedule = t_star_ms.map(|t| {
        let ts: Vec<Option<f64>> = conv.turns.iter().map(|turn| turn.timestamp_ms).collect();
        replay_schedule(&ts, t)
    });

    conv.turns
        .iter()
        .enumerate()
        .map(|(k, turn)| {
            let mut record = json!({
                "session_id": conv.session_id,
                "parent_conversation_id": conv.parent_conversation_id,
                "replay_scope_id": conv.replay_scope_id,
                "turn_index": k,
                "source_trace_id": turn.source_trace_id,
                "source_outer_idx": turn.source_outer_idx,
                "source_kind": turn.source_kind,
                "timestamp_ms": turn.timestamp_ms,
                "delay_ms": turn.delay_ms,
                "api_time_ms": turn.api_time_ms,
                "model": turn.model,
                "max_tokens": turn.max_tokens,
                "reset_context": turn.reset_context,
                "input_kind": turn.input_kind.map(|ik| ik.as_str()),
                "theoretical_prefix_cache_hit_blocks": turn.theoretical_prefix_cache_hit_blocks,
                "theoretical_prefix_cache_total_blocks": turn.theoretical_prefix_cache_total_blocks,
                "raw_messages": messages_json(&turn.raw_messages),
            });
            if let Some(sched) = &schedule {
                let st = &sched[k];
                record["phase"] = json!(phase_str(st.phase));
                record["dispatch_offset_ms"] = json!(st.offset_ms);
            }
            record
        })
        .collect()
}

/// Serialize a whole trace's conversations (root + children) into a flat list of
/// raw export records, in conversation then turn order.
pub fn raw_export_trace(convs: &[ReconstructedConversation], t_star_ms: Option<f64>) -> Vec<Value> {
    convs
        .iter()
        .flat_map(|c| raw_export_records(c, t_star_ms))
        .collect()
}

/// Emit a conversation's export records in **dispatch order** for an
/// agentic-replay run at `t_star_ms`: warmup turn first, then profiling turns by
/// ascending dispatch offset from t* (stable on ties), history turns omitted
/// (back-seeded, not dispatched). This is the deterministic run-time emission
/// order the async loop would produce, carrying byte-exact timing + content.
pub fn dispatch_ordered_records(conv: &ReconstructedConversation, t_star_ms: f64) -> Vec<Value> {
    let ts: Vec<Option<f64>> = conv.turns.iter().map(|t| t.timestamp_ms).collect();
    let schedule = replay_schedule(&ts, t_star_ms);
    let records = raw_export_records(conv, Some(t_star_ms));

    // (dispatch rank, offset, original index) -> stable dispatch order.
    let mut order: Vec<(u8, f64, usize)> = schedule
        .iter()
        .enumerate()
        .filter_map(|(i, st)| match st.phase {
            ReplayPhase::Warmup => Some((0u8, 0.0, i)),
            ReplayPhase::Profiling => Some((1u8, st.offset_ms.unwrap_or(0.0), i)),
            ReplayPhase::History => None,
        })
        .collect();
    order.sort_by(|a, b| {
        a.0.cmp(&b.0)
            .then(a.1.partial_cmp(&b.1).unwrap())
            .then(a.2.cmp(&b.2))
    });
    order.into_iter().map(|(_, _, i)| records[i].clone()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agentx::loader::{ReconstructedTurn, TurnInputKind};
    use crate::agentx::synth::ChatMessage;

    fn conv() -> ReconstructedConversation {
        ReconstructedConversation {
            session_id: "t".into(),
            replay_scope_id: "t".into(),
            parent_conversation_id: None,
            turns: vec![
                ReconstructedTurn {
                    timestamp_ms: Some(0.0),
                    delay_ms: None,
                    api_time_ms: Some(100.0),
                    source_trace_id: "t".into(),
                    source_outer_idx: 0,
                    source_kind: "weka_main".into(),
                    model: "m".into(),
                    max_tokens: 4,
                    raw_messages: vec![ChatMessage::plain("user", "hi")],
                    reset_context: false,
                    theoretical_prefix_cache_hit_blocks: 0,
                    theoretical_prefix_cache_total_blocks: 2,
                    input_kind: None,
                    spawn_branch: None,
                    join_prerequisite: None,
                },
                ReconstructedTurn {
                    timestamp_ms: Some(1000.0),
                    delay_ms: Some(900.0),
                    api_time_ms: Some(200.0),
                    source_trace_id: "t".into(),
                    source_outer_idx: 1,
                    source_kind: "weka_main".into(),
                    model: "m".into(),
                    max_tokens: 4,
                    raw_messages: vec![ChatMessage::plain("assistant", "yo")],
                    reset_context: true,
                    theoretical_prefix_cache_hit_blocks: 2,
                    theoretical_prefix_cache_total_blocks: 2,
                    input_kind: Some(TurnInputKind::UserInput),
                    spawn_branch: None,
                    join_prerequisite: None,
                },
            ],
        }
    }

    #[test]
    fn raw_records_carry_timing_and_content() {
        let recs = raw_export_records(&conv(), None);
        assert_eq!(recs.len(), 2);
        assert_eq!(recs[0]["timestamp_ms"], json!(0.0));
        assert_eq!(recs[0]["delay_ms"], Value::Null);
        assert_eq!(recs[0]["raw_messages"][0]["content"], json!("hi"));
        assert_eq!(recs[1]["delay_ms"], json!(900.0));
        assert_eq!(recs[1]["reset_context"], json!(true));
        assert_eq!(recs[1]["input_kind"], json!("user_input"));
    }

    #[test]
    fn dispatch_order_warmup_then_profiling_by_offset() {
        // 4 turns at [0, 500, 300, 900] ms; t* = 250.
        let mut c = conv();
        c.turns[0].timestamp_ms = Some(0.0);
        c.turns[1].timestamp_ms = Some(500.0);
        c.turns.push(c.turns[1].clone());
        c.turns.push(c.turns[1].clone());
        c.turns[2].timestamp_ms = Some(300.0);
        c.turns[2].source_outer_idx = 2;
        c.turns[3].timestamp_ms = Some(900.0);
        c.turns[3].source_outer_idx = 3;
        // t*=250 -> resume 1 (t=500), warmup 0. Profiling turns: 1(500->250),
        // 2(300->50), 3(900->650). Dispatch order: warmup(0), then by offset:
        // turn2(50), turn1(250), turn3(650).
        let recs = dispatch_ordered_records(&c, 250.0);
        let outer: Vec<i64> = recs
            .iter()
            .map(|r| r["source_outer_idx"].as_i64().unwrap())
            .collect();
        assert_eq!(outer, vec![0, 2, 1, 3]);
        assert_eq!(recs[0]["phase"], json!("warmup"));
        assert_eq!(recs[1]["dispatch_offset_ms"], json!(50.0));
        assert_eq!(recs[3]["dispatch_offset_ms"], json!(650.0));
    }

    #[test]
    fn schedule_annotation_when_t_star_given() {
        // timestamps [0, 1000], t* = 500 -> resume 1, warmup 0.
        let recs = raw_export_records(&conv(), Some(500.0));
        assert_eq!(recs[0]["phase"], json!("warmup"));
        assert_eq!(recs[0]["dispatch_offset_ms"], Value::Null);
        assert_eq!(recs[1]["phase"], json!("profiling"));
        assert_eq!(recs[1]["dispatch_offset_ms"], json!(500.0));
    }
}
