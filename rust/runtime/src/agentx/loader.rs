// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! WEKA trace → conversation reconstruction hub, ported from
//! `src/aiperf/dataset/loader/weka_trace.py`.
//!
//! This is the integrative layer that walks a conversation's turns through the
//! [`super::synth::ConversationReconstructor`], assembling per-turn
//! `raw_messages` + timing + prefix-cache tallies into
//! [`ReconstructedConversation`]s.
//!
//! Scope so far: the **main-conversation** path (a trace's top-level normal
//! requests, `source_kind = "weka_main"`), no subagent/flat-chain expansion and
//! no idle-gap time-warp. Those layer on next. The token generator is injected
//! via [`super::synth::TokenSynth`] (a [`super::corpus::CorpusTokenSynth`] in
//! production; a stub in parity tests).

use std::collections::HashMap;

use crate::agentx::prepass::{compute_shared_prefix_cache_metrics, MetricRecord, SortKey};
use crate::agentx::synth::{
    compute_asst_block_caps, ChatMessage, ConversationReconstructor, PrefixTooTruncated, TokenSynth,
};

/// What produced a turn's new input (Python `TurnInputKind`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TurnInputKind {
    /// A human-paced input turn.
    UserInput,
    /// A machine-paced tool-result continuation.
    ToolResult,
}

impl TurnInputKind {
    /// Wire string (`user_input` / `tool_result`).
    pub fn as_str(self) -> &'static str {
        match self {
            TurnInputKind::UserInput => "user_input",
            TurnInputKind::ToolResult => "tool_result",
        }
    }
}

/// A normalized top-level request view (projection of `WekaNormalRequest` /
/// `WekaStreamingRequest`) carrying every field the loader loop reads.
#[derive(Debug, Clone, PartialEq)]
pub struct NormalReq {
    /// Request timestamp in seconds from conversation start.
    pub t: f64,
    /// Server processing time in seconds.
    pub api_time: Option<f64>,
    /// Client delay in seconds before this request.
    pub think_time: Option<f64>,
    /// Model identifier.
    pub model: String,
    /// KV-cache block hash ids.
    pub hash_ids: Vec<i64>,
    /// Input token count.
    pub input_length: i64,
    /// Output token count.
    pub output_length: i64,
    /// Content-type annotations for input.
    pub input_types: Vec<String>,
    /// Stop reason: `""`, `"tool_use"`, `"end_turn"`.
    pub stop: String,
}

/// One reconstructed turn (a projection of AIPerf `Turn` limited to the fields
/// the byte-exact raw export compares).
#[derive(Debug, Clone, PartialEq)]
pub struct ReconstructedTurn {
    /// Absolute timestamp in ms (None when delays are ignored).
    pub timestamp_ms: Option<f64>,
    /// Inter-turn delay in ms (None on turn 0 or when delays are ignored).
    pub delay_ms: Option<f64>,
    /// Server processing duration in ms.
    pub api_time_ms: Option<f64>,
    /// Originating trace id.
    pub source_trace_id: String,
    /// Outer index in the trace's request list.
    pub source_outer_idx: i64,
    /// Provenance tag (`weka_main` for the main conversation).
    pub source_kind: String,
    /// Mapped model name.
    pub model: String,
    /// Sendable `max_tokens` (recorded output, capped, floored at 1).
    pub max_tokens: i64,
    /// Delta-encoded chat messages for this turn.
    pub raw_messages: Vec<ChatMessage>,
    /// Whether this emission resets previously-sent context.
    pub reset_context: bool,
    /// Theoretical prefix-cache hit blocks.
    pub theoretical_prefix_cache_hit_blocks: i64,
    /// Theoretical prefix-cache total blocks.
    pub theoretical_prefix_cache_total_blocks: i64,
    /// Classified input kind (None for legacy traces with no signal).
    pub input_kind: Option<TurnInputKind>,
}

/// A reconstructed conversation (projection of AIPerf `Conversation`).
#[derive(Debug, Clone, PartialEq)]
pub struct ReconstructedConversation {
    /// Conversation / session id (the trace id for the main conversation, the
    /// child session id for a subagent/flat chain).
    pub session_id: String,
    /// Replay scope id (the root trace id).
    pub replay_scope_id: String,
    /// Parent conversation id (`None` for the main conversation).
    pub parent_conversation_id: Option<String>,
    /// Turns in order.
    pub turns: Vec<ReconstructedTurn>,
}

/// Classify what produced a turn's new input (Python `_classify_turn_input`).
pub fn classify_turn_input(req: &NormalReq, prev: Option<&NormalReq>) -> Option<TurnInputKind> {
    if !req.input_types.is_empty() {
        if req.input_types.iter().any(|t| t == "tool_result") {
            return Some(TurnInputKind::ToolResult);
        }
        return Some(TurnInputKind::UserInput);
    }
    if let Some(p) = prev {
        if !p.stop.is_empty() {
            if p.stop == "tool_use" {
                return Some(TurnInputKind::ToolResult);
            }
            return Some(TurnInputKind::UserInput);
        }
    }
    None
}

/// Per-turn server-processing duration in ms (Python `_api_time_ms`).
/// Missing/non-finite/negative → None (distinct from a recorded 0.0).
pub fn api_time_ms(api_time: Option<f64>) -> Option<f64> {
    match api_time {
        Some(d) if d.is_finite() => Some(d.max(0.0) * 1000.0),
        _ => None,
    }
}

/// Convert a start-to-start inter-request delay to end-to-start (Python
/// `_end_to_start_delay_ms`): subtract the previous request's server time,
/// floored at 0. `None` when there is no prior turn.
pub fn end_to_start_delay_ms(
    start_to_start_ms: Option<f64>,
    prev_api_seconds: Option<f64>,
) -> Option<f64> {
    let s2s = start_to_start_ms?;
    let api_ms = match prev_api_seconds {
        Some(a) if a.is_finite() => a * 1000.0,
        _ => 0.0,
    };
    Some((s2s - api_ms).max(0.0))
}

/// Clamp a delay to at most `cap_seconds * 1000` ms (Python `_clamp_delay_ms`).
/// Non-finite → None; negatives pass through; only the upper bound is enforced.
pub fn clamp_delay_ms(delay_ms: f64, cap_seconds: Option<f64>) -> Option<f64> {
    if !delay_ms.is_finite() {
        return None;
    }
    match cap_seconds {
        None => Some(delay_ms),
        Some(cap) => {
            let cap_ms = cap * 1000.0;
            Some(if delay_ms > cap_ms { cap_ms } else { delay_ms })
        }
    }
}

/// Resolve recorded `out` to a sendable `max_tokens` (Python `_cap_output`):
/// honor `max_osl`, upgrade a recorded 0 to 1.
pub fn cap_output(output_length: i64, max_osl: Option<i64>) -> i64 {
    let mut capped = output_length;
    if let Some(m) = max_osl {
        if capped > m {
            capped = m;
        }
    }
    if capped >= 1 {
        capped
    } else {
        1
    }
}

/// Options for [`reconstruct_main_conversation`].
#[derive(Debug, Clone, Default)]
pub struct MainReconstructOptions {
    /// `--synthesis-max-osl` cap.
    pub max_osl: Option<i64>,
    /// Drop all timing (timestamp/delay/api_time set None).
    pub ignore_delays: bool,
    /// Use recorded `think_time` as the inter-turn delay when present.
    pub think_time_only: bool,
    /// Inter-turn delay cap in seconds.
    pub delay_cap_seconds: Option<f64>,
    /// Whether `turn_delta` emits tool-shaped messages.
    pub tool_shaped_messages: bool,
}

/// Reconstruct a trace's main conversation from its top-level normal requests.
///
/// `normals` is `(outer_idx, request)` in the plan's `(t, outer_idx)` order.
/// `synth` must already be scoped to `trace_id` (e.g.
/// [`super::corpus::CorpusTokenSynth::set_scope`]). `model_map` renames recorded
/// models. Returns `Err` only when turn 0's system prefix is too truncated.
pub fn reconstruct_main_conversation(
    trace_id: &str,
    block_size: i64,
    tool_tokens: i64,
    system_tokens: i64,
    normals: &[(i64, NormalReq)],
    synth: &mut dyn TokenSynth,
    model_map: &HashMap<String, String>,
    opts: &MainReconstructOptions,
) -> Result<ReconstructedConversation, PrefixTooTruncated> {
    // Single-conversation shared prefix-cache metrics (seen-set in time order).
    let metric_records: Vec<MetricRecord> = normals
        .iter()
        .enumerate()
        .map(|(k, (outer_idx, r))| MetricRecord {
            sort_key: SortKey {
                absolute_t: r.t,
                outer_idx: *outer_idx,
                stream_idx: 0,
                k: k as i64,
            },
            session_id: trace_id.to_string(),
            k: k as i64,
            hash_ids: r.hash_ids.clone(),
        })
        .collect();
    let metric_values = compute_shared_prefix_cache_metrics(metric_records);

    reconstruct_conversation(
        trace_id,
        trace_id,
        None,
        trace_id,
        "weka_main",
        block_size,
        tool_tokens,
        system_tokens,
        normals,
        synth,
        model_map,
        &metric_values,
        opts,
    )
}

/// Reconstruct one conversation (main or child) by driving the synth over its
/// requests. Shared by the main-conversation and subagent/flat-chain paths.
///
/// `requests` is `(source_outer_idx, request)`; the outer index is stamped on
/// each turn (per-request for the main conversation, the spawn-marker index for
/// a child). `metric_values` is the trace-wide prefix-cache map (keyed by
/// `(session_id, k)`). `synth` must already be scoped to the root trace id.
#[allow(clippy::too_many_arguments)]
pub fn reconstruct_conversation(
    session_id: &str,
    replay_scope_id: &str,
    parent_conversation_id: Option<&str>,
    source_trace_id: &str,
    source_kind: &str,
    block_size: i64,
    init_tool_tokens: i64,
    init_system_tokens: i64,
    requests: &[(i64, NormalReq)],
    synth: &mut dyn TokenSynth,
    model_map: &HashMap<String, String>,
    metric_values: &HashMap<(String, i64), (i64, i64)>,
    opts: &MainReconstructOptions,
) -> Result<ReconstructedConversation, PrefixTooTruncated> {
    let cap_input: Vec<(Vec<i64>, i64)> = requests
        .iter()
        .map(|(_, r)| (r.hash_ids.clone(), r.input_length))
        .collect();
    let asst_block_caps = compute_asst_block_caps(&cap_input, block_size);

    let mut recon = ConversationReconstructor::new(block_size, opts.tool_shaped_messages);
    let mut turns: Vec<ReconstructedTurn> = Vec::with_capacity(requests.len());

    for (k, (outer_idx, req)) in requests.iter().enumerate() {
        let seed = format!("{session_id}:turn_{k}:partial_tail");
        let prev = if k > 0 { Some(&requests[k - 1].1) } else { None };
        let input_kind = classify_turn_input(req, prev);
        let is_tool_result = input_kind == Some(TurnInputKind::ToolResult);

        if k == 0 {
            recon.init_turn_0(
                synth,
                &req.hash_ids,
                req.input_length,
                init_tool_tokens,
                init_system_tokens,
                &seed,
            )?;
        } else {
            let p = prev.unwrap();
            recon.advance_turn(
                synth,
                &p.hash_ids,
                p.output_length,
                &req.hash_ids,
                req.input_length,
                &seed,
                is_tool_result,
                asst_block_caps[k],
            );
        }

        // Timing (no idle-gap warp in this path).
        let t_ms = req.t * 1000.0;
        let mut delay_ms: Option<f64> = if k == 0 {
            None
        } else if opts.think_time_only && req.think_time.is_some() {
            Some(req.think_time.unwrap() * 1000.0)
        } else {
            let p = prev.unwrap();
            end_to_start_delay_ms(Some(t_ms - p.t * 1000.0), p.api_time)
        };
        if let Some(d) = delay_ms {
            delay_ms = clamp_delay_ms(d, opts.delay_cap_seconds).map(|x| x.max(0.0));
        }

        let delta = recon.turn_delta();
        let (hit, total) = metric_values
            .get(&(session_id.to_string(), k as i64))
            .copied()
            .unwrap_or((0, 0));

        turns.push(ReconstructedTurn {
            timestamp_ms: if opts.ignore_delays { None } else { Some(t_ms) },
            delay_ms: if opts.ignore_delays { None } else { delay_ms },
            api_time_ms: if opts.ignore_delays {
                None
            } else {
                api_time_ms(req.api_time)
            },
            source_trace_id: source_trace_id.to_string(),
            source_outer_idx: *outer_idx,
            source_kind: source_kind.to_string(),
            model: model_map.get(&req.model).cloned().unwrap_or(req.model.clone()),
            max_tokens: cap_output(req.output_length, opts.max_osl),
            raw_messages: delta.delta_messages,
            reset_context: delta.reset_context,
            theoretical_prefix_cache_hit_blocks: hit,
            theoretical_prefix_cache_total_blocks: total,
            input_kind,
        });
    }

    Ok(ReconstructedConversation {
        session_id: session_id.to_string(),
        replay_scope_id: replay_scope_id.to_string(),
        parent_conversation_id: parent_conversation_id.map(str::to_string),
        turns,
    })
}

/// Convert a parsed `WekaTrace` into its reconstructed conversations (the root
/// `weka_main` conversation plus one `weka_subagent` child per active subagent
/// chain). Ports the no-flat-chain, no-idle-warp path of
/// `WekaTraceLoader.convert_to_conversations`.
///
/// `synth` must be freshly scoped to `trace_id` (its per-scope block cache is
/// shared across the trace's conversations under the `local` hash namespace).
/// Flat-chain splitting and the idle-gap time-warp are not yet applied.
pub fn convert_trace_to_conversations(
    trace_id: &str,
    trace: &crate::agentx::trace::WekaTrace,
    synth: &mut dyn TokenSynth,
    model_map: &HashMap<String, String>,
    cfg: &crate::agentx::config::WekaConfig,
    opts: &MainReconstructOptions,
) -> Result<Vec<ReconstructedConversation>, PrefixTooTruncated> {
    use crate::agentx::plan::{build_shared_metric_values, dropped_subagent_indices, ParentPlan};
    use crate::agentx::subagent::expand_subagent_to_child_plans;
    use crate::agentx::trace::WekaRequest;

    let block_size = trace.block_size;

    // Split top-level requests into normals + subagents; expand each subagent.
    let mut normals: Vec<(i64, NormalReq)> = Vec::new();
    let mut subagent_outer_indices: Vec<i64> = Vec::new();
    let mut children: Vec<crate::agentx::subagent::ChildPlan> = Vec::new();
    for (outer_idx, req) in trace.requests.iter().enumerate() {
        let outer_idx = outer_idx as i64;
        match req {
            WekaRequest::Normal(n) => normals.push((outer_idx, normal_req_from_normal(n))),
            WekaRequest::Streaming(s) => normals.push((outer_idx, normal_req_from_streaming(s))),
            WekaRequest::Subagent(entry) => {
                let sa_index = subagent_outer_indices.len();
                subagent_outer_indices.push(outer_idx);
                children.extend(expand_subagent_to_child_plans(
                    trace_id, sa_index, outer_idx, entry, block_size, cfg,
                ));
            }
        }
    }

    let parent = ParentPlan {
        trace_id: trace_id.to_string(),
        normals: normals.clone(),
        subagent_outer_indices,
        block_size,
    };
    let dropped = dropped_subagent_indices(&parent);
    let metrics_by_trace = build_shared_metric_values(std::slice::from_ref(&parent), &children);
    let metric_values = metrics_by_trace
        .get(trace_id)
        .cloned()
        .unwrap_or_default();

    let mut out: Vec<ReconstructedConversation> = Vec::new();
    // Root conversation.
    out.push(reconstruct_conversation(
        trace_id,
        trace_id,
        None,
        trace_id,
        "weka_main",
        block_size,
        trace.tool_tokens,
        trace.system_tokens,
        &normals,
        synth,
        model_map,
        &metric_values,
        opts,
    )?);

    // Active child conversations.
    for cp in &children {
        if dropped.contains(&cp.subagent_index) {
            continue;
        }
        let child_requests: Vec<(i64, NormalReq)> = cp
            .requests
            .iter()
            .cloned()
            .map(|r| (cp.source_outer_idx, r))
            .collect();
        out.push(reconstruct_conversation(
            &cp.session_id,
            trace_id,
            Some(trace_id),
            trace_id,
            "weka_subagent",
            cp.block_size,
            cp.init_tool_tokens,
            cp.init_system_tokens,
            &child_requests,
            synth,
            model_map,
            &metric_values,
            opts,
        )?);
    }
    Ok(out)
}

/// Build a [`NormalReq`] from a wire normal request.
pub fn normal_req_from_normal(n: &crate::agentx::trace::WekaNormalRequest) -> NormalReq {
    NormalReq {
        t: n.t,
        api_time: n.api_time,
        think_time: n.think_time,
        model: n.model.clone(),
        hash_ids: n.hash_ids.clone(),
        input_length: n.input_length,
        output_length: n.output_length,
        input_types: n.input_types.clone(),
        stop: n.stop.clone(),
    }
}

/// Build a [`NormalReq`] from a wire streaming request.
pub fn normal_req_from_streaming(s: &crate::agentx::trace::WekaStreamingRequest) -> NormalReq {
    NormalReq {
        t: s.t,
        api_time: s.api_time,
        think_time: s.think_time,
        model: s.model.clone(),
        hash_ids: s.hash_ids.clone(),
        input_length: s.input_length,
        output_length: s.output_length,
        input_types: s.input_types.clone(),
        stop: s.stop.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_prefers_input_types_then_prev_stop() {
        let user = NormalReq {
            t: 0.0,
            api_time: None,
            think_time: None,
            model: "m".into(),
            hash_ids: vec![],
            input_length: 4,
            output_length: 4,
            input_types: vec!["text".into()],
            stop: String::new(),
        };
        assert_eq!(classify_turn_input(&user, None), Some(TurnInputKind::UserInput));
        let tool = NormalReq {
            input_types: vec!["tool_result".into()],
            ..user.clone()
        };
        assert_eq!(classify_turn_input(&tool, None), Some(TurnInputKind::ToolResult));
        let prev_tooluse = NormalReq {
            stop: "tool_use".into(),
            ..user.clone()
        };
        let bare = NormalReq {
            input_types: vec![],
            ..user.clone()
        };
        assert_eq!(
            classify_turn_input(&bare, Some(&prev_tooluse)),
            Some(TurnInputKind::ToolResult)
        );
        assert_eq!(classify_turn_input(&bare, None), None);
    }

    struct StubSynth {
        bs: i64,
    }
    impl TokenSynth for StubSynth {
        fn decode_block_tokens(&mut self, hash_ids: &[i64]) -> Vec<u32> {
            hash_ids
                .iter()
                .flat_map(|&h| (0..self.bs).map(move |i| (h as u32) * 1000 + i as u32))
                .collect()
        }
        fn sample_partial_tail_tokens(&mut self, n: usize, _seed: &str) -> Vec<u32> {
            (0..n as u32).map(|i| 900_000 + i).collect()
        }
        fn decode_tokens_to_text(&self, tokens: &[u32]) -> String {
            tokens.iter().map(|t| t.to_string()).collect::<Vec<_>>().join(" ")
        }
    }

    #[test]
    fn convert_trace_emits_root_and_child_conversations() {
        use crate::agentx::config::WekaConfig;
        use crate::agentx::trace::{
            WekaInnerRequest, WekaNormalRequest, WekaRequest, WekaSubagentEntry, WekaTrace,
            HashIdScope,
        };
        let norm = |t: f64, hs: &[i64], in_len: i64| WekaNormalRequest {
            t,
            model: "m".into(),
            input_length: in_len,
            output_length: 4,
            hash_ids: hs.to_vec(),
            input_types: vec![],
            output_types: vec![],
            stop: String::new(),
            api_time: Some(0.1),
            think_time: None,
        };
        let trace = WekaTrace {
            id: "t".into(),
            models: vec!["m".into()],
            block_size: 4,
            hash_id_scope: HashIdScope::Local,
            tool_tokens: 0,
            system_tokens: 0,
            requests: vec![
                WekaRequest::Normal(norm(0.0, &[1, 2], 8)),
                WekaRequest::Subagent(WekaSubagentEntry {
                    t: 1.0,
                    agent_id: "a1".into(),
                    subagent_type: "Explore".into(),
                    duration_ms: Some(500),
                    total_tokens: None,
                    tool_use_count: None,
                    status: "completed".into(),
                    requests: vec![WekaInnerRequest::Normal(norm(1.0, &[5, 6], 8))],
                    models: vec!["m".into()],
                    tool_tokens: 0,
                    system_tokens: 0,
                }),
                WekaRequest::Normal(norm(2.0, &[1, 2, 3], 12)),
            ],
            totals: None,
        };
        let mut synth = StubSynth { bs: 4 };
        let convs = convert_trace_to_conversations(
            "t",
            &trace,
            &mut synth,
            &HashMap::new(),
            &WekaConfig::default(),
            &MainReconstructOptions::default(),
        )
        .unwrap();
        // Root + one active subagent child.
        assert_eq!(convs.len(), 2);
        assert_eq!(convs[0].session_id, "t");
        assert_eq!(convs[0].parent_conversation_id, None);
        assert_eq!(convs[0].turns.len(), 2); // two top-level normals
        assert_eq!(convs[1].session_id, "t::sa:a1");
        assert_eq!(convs[1].parent_conversation_id.as_deref(), Some("t"));
        assert_eq!(convs[1].turns[0].source_kind, "weka_subagent");
    }

    #[test]
    fn delay_and_cap_helpers() {
        // end-to-start subtracts prev api time (0.1s = 100ms) from the 1000ms gap.
        assert_eq!(end_to_start_delay_ms(Some(1000.0), Some(0.1)), Some(900.0));
        assert_eq!(end_to_start_delay_ms(None, Some(0.1)), None);
        assert_eq!(clamp_delay_ms(5000.0, Some(2.0)), Some(2000.0));
        assert_eq!(clamp_delay_ms(f64::NAN, None), None);
        assert_eq!(cap_output(0, None), 1);
        assert_eq!(cap_output(500, Some(100)), 100);
    }
}
