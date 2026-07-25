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

use std::collections::{HashMap, HashSet};

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
    use crate::agentx::plan::{
        build_shared_metric_values, detect_and_split_flat_chains, dropped_subagent_indices,
        ParentPlan,
    };
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

    // Flat-chain splitting: partition the top-level requests into the main
    // chain + detected flat worker chains (only when enabled and >1 normal).
    let (main_normals, flat_plans) = if cfg.split_flattened_agents && normals.len() > 1 {
        detect_and_split_flat_chains(
            trace_id,
            &normals,
            trace.tool_tokens,
            trace.system_tokens,
            block_size,
            cfg,
        )
    } else {
        (normals.clone(), Vec::new())
    };

    let parent = ParentPlan {
        trace_id: trace_id.to_string(),
        normals: main_normals.clone(),
        subagent_outer_indices,
        block_size,
    };
    let dropped = dropped_subagent_indices(&parent);
    let metrics_by_trace =
        build_shared_metric_values(std::slice::from_ref(&parent), &children, &flat_plans);
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
        &main_normals,
        synth,
        model_map,
        &metric_values,
        opts,
    )?);

    // Flat worker-chain conversations.
    for fp in &flat_plans {
        out.push(reconstruct_conversation(
            &fp.session_id,
            trace_id,
            Some(trace_id),
            trace_id,
            "weka_flat",
            fp.block_size,
            fp.init_tool_tokens,
            fp.init_system_tokens,
            &fp.requests,
            synth,
            model_map,
            &metric_values,
            opts,
        )?);
    }

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

/// One trace's conversion result: its trace id and the reconstructed
/// conversations (or the turn-0 prefix error).
pub type TraceConversions = Result<Vec<ReconstructedConversation>, PrefixTooTruncated>;

/// Reconstruct many traces **serially** (Slice 2 reference path). `make_synth`
/// builds a fresh, trace-scoped [`TokenSynth`] for each `(trace_id, block_size)`.
pub fn convert_traces_serial<S, MK>(
    traces: &[(String, crate::agentx::trace::WekaTrace)],
    model_map: &HashMap<String, String>,
    cfg: &crate::agentx::config::WekaConfig,
    opts: &MainReconstructOptions,
    make_synth: MK,
) -> Vec<TraceConversions>
where
    S: TokenSynth,
    MK: Fn(&str, i64) -> S,
{
    traces
        .iter()
        .map(|(tid, trace)| {
            let mut synth = make_synth(tid, trace.block_size);
            convert_trace_to_conversations(tid, trace, &mut synth, model_map, cfg, opts)
        })
        .collect()
}

/// Reconstruct many traces **in parallel** (Slice 2, `rayon`). Each trace is
/// self-contained (own trace-scoped synth + hash namespace), so the output is
/// **byte-identical to [`convert_traces_serial`]** regardless of thread count.
pub fn convert_traces_parallel<S, MK>(
    traces: &[(String, crate::agentx::trace::WekaTrace)],
    model_map: &HashMap<String, String>,
    cfg: &crate::agentx::config::WekaConfig,
    opts: &MainReconstructOptions,
    make_synth: MK,
) -> Vec<TraceConversions>
where
    S: TokenSynth + Send,
    MK: Fn(&str, i64) -> S + Sync,
{
    use rayon::prelude::*;
    traces
        .par_iter()
        .map(|(tid, trace)| {
            let mut synth = make_synth(tid, trace.block_size);
            convert_trace_to_conversations(tid, trace, &mut synth, model_map, cfg, opts)
        })
        .collect()
}

/// Map trace-side model names to configured `endpoint.model_names` (Python
/// `_build_model_map`).
///
/// The trace's main model (first parent request, falling back to the first
/// subagent inner request) maps to `configured[0]`; other distinct trace models
/// map to `configured[1..]` in first-appearance order with modulo wrap. Empty
/// when `configured` is empty or no model is found.
pub fn build_model_map(
    trace: &crate::agentx::trace::WekaTrace,
    configured: &[String],
) -> HashMap<String, String> {
    use crate::agentx::trace::{WekaInnerRequest, WekaRequest};
    if configured.is_empty() {
        return HashMap::new();
    }

    let inner_model = |r: &WekaInnerRequest| match r {
        WekaInnerRequest::Normal(n) => n.model.clone(),
        WekaInnerRequest::Streaming(s) => s.model.clone(),
    };

    // Main model: first top-level normal/streaming, else first subagent inner.
    let mut main_model: Option<String> = None;
    for req in &trace.requests {
        match req {
            WekaRequest::Normal(n) => {
                main_model = Some(n.model.clone());
                break;
            }
            WekaRequest::Streaming(s) => {
                main_model = Some(s.model.clone());
                break;
            }
            WekaRequest::Subagent(_) => {}
        }
    }
    if main_model.is_none() {
        for req in &trace.requests {
            if let WekaRequest::Subagent(entry) = req {
                if let Some(first) = entry.requests.first() {
                    main_model = Some(inner_model(first));
                    break;
                }
            }
        }
    }
    let main_model = match main_model {
        Some(m) => m,
        None => return HashMap::new(),
    };

    let mut ordered: Vec<String> = vec![main_model.clone()];
    let mut seen: HashSet<String> = HashSet::new();
    seen.insert(main_model);
    let push = |m: String, ordered: &mut Vec<String>, seen: &mut HashSet<String>| {
        if seen.insert(m.clone()) {
            ordered.push(m);
        }
    };
    for req in &trace.requests {
        match req {
            WekaRequest::Normal(n) => push(n.model.clone(), &mut ordered, &mut seen),
            WekaRequest::Streaming(s) => push(s.model.clone(), &mut ordered, &mut seen),
            WekaRequest::Subagent(entry) => {
                for inner in &entry.requests {
                    push(inner_model(inner), &mut ordered, &mut seen);
                }
            }
        }
    }

    let n = configured.len();
    ordered
        .into_iter()
        .enumerate()
        .map(|(i, m)| (m, configured[i % n].clone()))
        .collect()
}

/// Peak requested context length across a trace's parent and subagent requests
/// (Python `_trace_peak_context_length`): `input + capped_output`. Parent turns
/// honor `max_osl`; subagent child turns use the uncapped recorded output.
pub fn trace_peak_context_length(
    trace: &crate::agentx::trace::WekaTrace,
    max_osl: Option<i64>,
) -> i64 {
    use crate::agentx::trace::{WekaInnerRequest, WekaRequest};
    let mut peak = 0i64;
    for req in &trace.requests {
        match req {
            WekaRequest::Normal(n) => {
                peak = peak.max(n.input_length + cap_output(n.output_length, max_osl));
            }
            WekaRequest::Streaming(s) => {
                peak = peak.max(s.input_length + cap_output(s.output_length, max_osl));
            }
            WekaRequest::Subagent(entry) => {
                for child in &entry.requests {
                    let (in_len, out_len) = match child {
                        WekaInnerRequest::Normal(n) => (n.input_length, n.output_length),
                        WekaInnerRequest::Streaming(s) => (s.input_length, s.output_length),
                    };
                    // Subagent children are NOT subject to max_osl (uncapped output).
                    peak = peak.max(in_len + cap_output(out_len, None));
                }
            }
        }
    }
    peak
}

/// Filter-then-cap selection over HF/loaded traces by peak context length
/// (Python `SemiAnalysisCCTracesWekaLoader._validate_rows` selection, delegating
/// to `filter_then_cap`). Drops traces whose peak context exceeds
/// `max_context_length` without consuming a slot, then keeps the first
/// `num_dataset_entries` eligible.
pub fn hf_select_traces(
    traces: Vec<(String, crate::agentx::trace::WekaTrace)>,
    num_dataset_entries: Option<usize>,
    max_context_length: Option<i64>,
    max_osl: Option<i64>,
) -> (
    Vec<(String, crate::agentx::trace::WekaTrace)>,
    crate::agentx::selection::SelectionStats,
) {
    let candidates = traces.into_iter().map(|(id, trace)| {
        let peak = trace_peak_context_length(&trace, max_osl);
        ((id, trace), peak)
    });
    crate::agentx::selection::filter_then_cap(candidates, num_dataset_entries, max_context_length)
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

    /// The real-corpus end-to-end gate: reconstruct `simple.json`'s main
    /// conversation with the ACTUAL Qwen3-0.6B tokenizer + `build_coding_corpus`
    /// + `CorpusTokenSynth` + `convert_trace_to_conversations`, and diff every
    /// turn's `raw_messages` (real decoded text) + timing + prefix-cache against
    /// the real-Python golden (`tools/agentx_realcorpus_golden.py`). Skips when
    /// Qwen or the golden is absent.
    #[test]
    fn realcorpus_main_conversation_matches_python() {
        use crate::agentx::config::WekaConfig;
        use crate::agentx::corpus::CorpusTokenSynth;
        use crate::agentx::trace::WekaTrace;
        use crate::dataset::tokenizer::TextTokenizer;
        use crate::rng::compat::python_random::PythonRandomGenerator;

        let manifest = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let repo = manifest.join("../..");
        let golden_path = repo.join("tests/fixtures/agentx/realcorpus_golden.json");
        let golden_raw = match std::fs::read(&golden_path) {
            Ok(r) => r,
            Err(_) => {
                eprintln!("skip: realcorpus golden absent");
                return;
            }
        };
        let golden: serde_json::Value = serde_json::from_slice(&golden_raw).unwrap();

        // Load Qwen from the HF cache (skip if absent).
        let home = std::env::var("HOME").unwrap_or_default();
        let base =
            format!("{home}/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots");
        let snap = std::fs::read_dir(&base).ok().and_then(|d| {
            d.filter_map(|e| e.ok())
                .map(|e| e.path())
                .find(|p| p.join("tokenizer.json").exists())
        });
        let snap = match snap {
            Some(s) => s,
            None => {
                eprintln!("skip: Qwen not cached");
                return;
            }
        };
        let tok = crate::dataset::tokenizer::HuggingFaceTokenizer::from_directory(&snap)
            .expect("load qwen");
        let corpus = crate::dataset::coding::build_coding_corpus(&tok, 42).expect("corpus");

        // Derive the hash-id base seed exactly as Python's CodingContentGenerator.
        let hash_base_seed =
            PythonRandomGenerator::derive_child_seed(42, crate::rng::namespace::DATASET_CODING_CONTENT_CORPUS);

        // build_coding_corpus is deterministic given seed 42 -> reuse across fixtures.
        let cfg = WekaConfig {
            split_flattened_agents: false,
            ..WekaConfig::default()
        };
        for fixture in golden.as_array().unwrap() {
            assert_eq!(
                hash_base_seed,
                fixture["hash_base_seed"].as_u64().unwrap(),
                "hash_base_seed"
            );
            let trace_id = fixture["trace_id"].as_str().unwrap();
            let bs = fixture["block_size"].as_i64().unwrap();
            let fpath = repo.join(fixture["fixture"].as_str().unwrap());
            let trace = WekaTrace::from_json_bytes(&std::fs::read(&fpath).unwrap()).unwrap();

            let mut synth =
                CorpusTokenSynth::new(corpus.clone(), bs, hash_base_seed, trace_id, |t: &[u32]| {
                    tok.decode(t).unwrap()
                });
            let convs = convert_trace_to_conversations(
                trace_id,
                &trace,
                &mut synth,
                &HashMap::new(),
                &cfg,
                &MainReconstructOptions::default(),
            )
            .unwrap();
            let by_sid: HashMap<&str, &ReconstructedConversation> =
                convs.iter().map(|c| (c.session_id.as_str(), c)).collect();

            for wc in fixture["conversations"].as_array().unwrap() {
                let sid = wc["session_id"].as_str().unwrap();
                let conv = by_sid
                    .get(sid)
                    .unwrap_or_else(|| panic!("{fixture:?}: missing conversation {sid}"));
                let want_turns = wc["turns"].as_array().unwrap();
                assert_eq!(conv.turns.len(), want_turns.len(), "{sid} turn count");
                for (i, (t, w)) in conv.turns.iter().zip(want_turns).enumerate() {
                    assert_eq!(t.timestamp_ms, w["timestamp_ms"].as_f64(), "{sid} t{i} timestamp");
                    assert_eq!(t.delay_ms, w["delay_ms"].as_f64(), "{sid} t{i} delay");
                    assert_eq!(t.source_kind, wc["source_kind"].as_str().unwrap(), "{sid} t{i} kind");
                    assert_eq!(
                        t.reset_context,
                        w["reset_context"].as_bool().unwrap(),
                        "{sid} t{i} reset"
                    );
                    assert_eq!(t.max_tokens, w["max_tokens"].as_i64().unwrap(), "{sid} t{i} max_tokens");
                    assert_eq!(
                        t.theoretical_prefix_cache_hit_blocks,
                        w["hit"].as_i64().unwrap(),
                        "{sid} t{i} hit"
                    );
                    let wm = w["raw_messages"].as_array().unwrap();
                    assert_eq!(t.raw_messages.len(), wm.len(), "{sid} t{i} msg count");
                    for (m, w) in t.raw_messages.iter().zip(wm) {
                        assert_eq!(m.role, w["role"].as_str().unwrap(), "{sid} t{i} role");
                        assert_eq!(
                            m.content,
                            w["content"].as_str().unwrap(),
                            "{sid} t{i} content (real Qwen text)"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn hf_peak_context_and_selection() {
        use crate::agentx::trace::{HashIdScope, WekaNormalRequest, WekaRequest, WekaTrace};
        let mk = |id: &str, in_out: &[(i64, i64)]| {
            (
                id.to_string(),
                WekaTrace {
                    id: id.into(),
                    models: vec!["m".into()],
                    block_size: 4,
                    hash_id_scope: HashIdScope::Local,
                    tool_tokens: 0,
                    system_tokens: 0,
                    requests: in_out
                        .iter()
                        .map(|&(i, o)| {
                            WekaRequest::Normal(WekaNormalRequest {
                                t: 0.0,
                                model: "m".into(),
                                input_length: i,
                                output_length: o,
                                hash_ids: vec![1],
                                input_types: vec![],
                                output_types: vec![],
                                stop: String::new(),
                                api_time: None,
                                think_time: None,
                            })
                        })
                        .collect(),
                    totals: None,
                },
            )
        };
        // peak = max(input + capped_output). a: 100+4=104; b: 900+50=950; c: 200+4=204.
        let a = mk("a", &[(100, 4)]);
        assert_eq!(trace_peak_context_length(&a.1, None), 104);
        let traces = vec![a, mk("b", &[(900, 50)]), mk("c", &[(200, 4)])];
        // max_context 500 drops b; cap 2 keeps a, c.
        let (kept, stats) = hf_select_traces(traces, Some(2), Some(500), None);
        assert_eq!(kept.iter().map(|(id, _)| id.as_str()).collect::<Vec<_>>(), vec!["a", "c"]);
        assert_eq!(stats.rejected_by_maxctx, 1);
        assert_eq!(stats.largest_observed, 950);
    }

    #[test]
    fn parallel_reconstruction_is_byte_identical_to_serial() {
        use crate::agentx::config::WekaConfig;
        use crate::agentx::trace::{HashIdScope, WekaNormalRequest, WekaRequest, WekaTrace};

        let norm = |t: f64, hs: &[i64], in_len: i64| {
            WekaRequest::Normal(WekaNormalRequest {
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
            })
        };
        // Several distinct traces (varying turns/prefixes).
        let mut traces: Vec<(String, WekaTrace)> = Vec::new();
        for i in 0..8 {
            let base = (i as i64) * 100;
            traces.push((
                format!("trace_{i}"),
                WekaTrace {
                    id: format!("trace_{i}"),
                    models: vec!["m".into()],
                    block_size: 4,
                    hash_id_scope: HashIdScope::Local,
                    tool_tokens: 0,
                    system_tokens: 0,
                    requests: vec![
                        norm(0.0, &[base, base + 1], 8),
                        norm(1.0, &[base, base + 1, base + 2], 12),
                        norm(2.0, &[base, base + 1, base + 2, base + 3], 16),
                    ],
                    totals: None,
                },
            ));
        }

        let make = |_tid: &str, bs: i64| StubSynth { bs };
        let cfg = WekaConfig::default();
        let opts = MainReconstructOptions::default();
        let serial = convert_traces_serial(&traces, &HashMap::new(), &cfg, &opts, make);
        let parallel = convert_traces_parallel(&traces, &HashMap::new(), &cfg, &opts, make);
        assert_eq!(serial, parallel, "parallel must be byte-identical to serial");
        assert_eq!(serial.len(), 8);
    }

    #[test]
    fn model_map_main_first_then_appearance_with_wrap() {
        use crate::agentx::trace::{HashIdScope, WekaNormalRequest, WekaRequest, WekaTrace};
        let norm = |model: &str| {
            WekaRequest::Normal(WekaNormalRequest {
                t: 0.0,
                model: model.into(),
                input_length: 4,
                output_length: 4,
                hash_ids: vec![1],
                input_types: vec![],
                output_types: vec![],
                stop: String::new(),
                api_time: None,
                think_time: None,
            })
        };
        let trace = WekaTrace {
            id: "t".into(),
            models: vec![],
            block_size: 4,
            hash_id_scope: HashIdScope::Local,
            tool_tokens: 0,
            system_tokens: 0,
            requests: vec![norm("opus"), norm("haiku"), norm("sonnet"), norm("opus")],
            totals: None,
        };
        // 3 distinct models, 2 configured -> wrap: opus->A, haiku->B, sonnet->A.
        let m = build_model_map(&trace, &["A".to_string(), "B".to_string()]);
        assert_eq!(m["opus"], "A");
        assert_eq!(m["haiku"], "B");
        assert_eq!(m["sonnet"], "A");
        // Empty configured -> identity (empty map).
        assert!(build_model_map(&trace, &[]).is_empty());
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
