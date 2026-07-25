// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Compose reconstructed WEKA trajectories into a linear scheduled [`Dataset`].
//!
//! The agentic-replay timing mode runs on the shared scheduled runtime, which
//! reads turns through a [`ConversationSource`](crate::multiturn::ConversationSource)
//! over a [`Dataset`]. This module builds that dataset directly from the
//! byte-exact AgentX reconstruction: each turn's exact OpenAI `/v1/chat/completions`
//! body (from [`crate::agentx::wire::chat_request_body`], with the byte-exact
//! cache-bust marker on the first turn of each trajectory tree) is interned as an
//! opaque `Raw` segment so the transport replays it verbatim — no re-tokenization
//! or prompt re-synthesis — and the recorded per-turn `timestamp_ms`/`delay_ms`
//! are carried so the workload can compute t\*-relative dispatch times.

use std::collections::HashSet;
use std::sync::Arc;

use anyhow::Result;
use bytes::Bytes;

use crate::agentx::cache_bust::{resolve_tree_marker, CacheBustLedger, CacheBustTarget};
use crate::agentx::loader::ReconstructedConversation;
use crate::agentx::wire::{chat_request_body, ChatRequestOptions};
use crate::dataset::Dataset;
use crate::dataset::model::{
    BranchId, Conversation, ConversationBranch, ConversationBranchMode, ConversationContextMode,
    DagMetadata, DispatchTiming, ModelId, PrerequisiteKind, SessionId, Turn, TurnPrerequisite,
};
use crate::dataset::segment::{Role, SegmentPool};

/// Wire-body options applied when composing (scenario-derived).
#[derive(Debug, Clone)]
pub struct WekaComposeOptions {
    /// Emit `stream: true` (scenario requires streaming).
    pub streaming: bool,
    /// Inject `ignore_eos: true`.
    pub ignore_eos: bool,
    /// Benchmark id for the cache-bust digest.
    pub benchmark_id: String,
    /// Cache-bust placement (scenario-locked to first-turn-prefix for the MVP).
    pub cache_bust_target: CacheBustTarget,
}

/// Apply the per-lane t\* snapshot slice to reconstructed trajectories, porting
/// Python `AgenticReplayStrategy`'s snapshot construction.
///
/// For each lane (in stable order, the seed's lane index): sample t\* uniformly
/// over `[min_ts + min_ratio·dur, min_ts + max_ratio·dur)` from the lane's
/// recorded turn timestamps (numpy PCG64, matched to Python), find the first turn
/// at/after t\* (the PROFILING resume point), DROP the earlier history turns, and
/// rebase each retained turn's `timestamp_ms` to its t\*-relative dispatch offset
/// (`max(0, ts − t*)`). Lanes with no post-t\* turn are dropped. The result feeds
/// [`compose_weka_agentic_dataset`]; the workload then only aligns lanes and
/// dispatches — history exclusion and t\* live here.
pub fn slice_trajectories_at_tstar(
    convs: Vec<ReconstructedConversation>,
    base_seed: u64,
    start_min_ratio: f64,
    start_max_ratio: f64,
    idle_gap_cap_ms: Option<f64>,
) -> Vec<ReconstructedConversation> {
    use crate::agentx::trajectory_source::{
        capped_warmup_lead_ms, next_turn_index_at_or_after, seed_for_trace_lane,
        timestamped_t_star_ms,
    };
    let mut out = Vec::with_capacity(convs.len());
    for (lane_index, conv) in convs.into_iter().enumerate() {
        let ts: Vec<Option<f64>> = conv.turns.iter().map(|t| t.timestamp_ms).collect();
        let known: Vec<f64> = ts.iter().filter_map(|x| *x).collect();
        let t_star = if known.is_empty() {
            0.0
        } else {
            let mn = known.iter().copied().fold(f64::INFINITY, f64::min);
            let mx = known.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let dur = mx - mn;
            let seed = seed_for_trace_lane(base_seed, &conv.session_id, lane_index as i64);
            timestamped_t_star_ms(seed, mn + start_min_ratio * dur, mn + start_max_ratio * dur)
        };
        let Some(next_idx) = next_turn_index_at_or_after(&ts, t_star) else {
            continue; // no post-t* (profiling) turn on this lane
        };
        let base_id = conv.session_id.clone();

        // WARMUP slice: the last pre-t* turn (n-1), a single 1-token prime whose
        // dispatch offset is its capped lead (t* − warm_ts). Shares the tree's
        // root correlation (via parent id) so it reuses the same cache-bust marker.
        if next_idx >= 1 {
            let warm_i = (next_idx - 1) as usize;
            let mut warm_turn = conv.turns[warm_i].clone();
            let lead = capped_warmup_lead_ms(t_star - warm_turn.timestamp_ms.unwrap_or(t_star), idle_gap_cap_ms);
            warm_turn.timestamp_ms = Some(lead.max(0.0));
            warm_turn.max_tokens = 1; // Python `_WARMUP_MAX_TOKENS`
            // The warmup prime is a standalone 1-token request; it is NOT a
            // spawn/join point. Strip any inherited branch/join metadata so the
            // warmup conversation is not misread as a Spawn parent by validate_dag.
            warm_turn.spawn_branch = None;
            warm_turn.join_prerequisite = None;
            out.push(ReconstructedConversation {
                session_id: format!("{base_id}{WARMUP_SUFFIX}"),
                replay_scope_id: conv.replay_scope_id.clone(),
                parent_conversation_id: Some(base_id.clone()),
                turns: vec![warm_turn],
            });
        }

        // PROFILING slice: turns from the resume point, rebased to t*-relative
        // offsets (history excluded).
        let mut prof = conv;
        prof.turns.drain(0..next_idx as usize);
        for turn in &mut prof.turns {
            turn.timestamp_ms = Some(turn.timestamp_ms.map_or(0.0, |ts| (ts - t_star).max(0.0)));
        }
        if !prof.turns.is_empty() {
            out.push(prof);
        }
    }
    out
}

/// Session-id suffix marking a warmup-phase conversation (turn n-1 prime). The
/// `agentic_replay` warmup phase dispatches only these; profiling skips them.
pub const WARMUP_SUFFIX: &str = "::warmup";

/// Compose reconstructed conversations into a verbatim-replay [`Dataset`].
///
/// Turns carry their exact chat body as a `Raw` segment (byte-for-byte replay,
/// preserving tool messages) plus the recorded `timestamp_ms`/`delay_ms`. The
/// cache-bust marker (resolved once per trajectory tree via a shared ledger) is
/// baked into the first turn's body. Context mode is `MessageArrayWithResponses`
/// (each turn is a complete self-contained messages array).
pub fn compose_weka_agentic_dataset(
    convs: &[ReconstructedConversation],
    opts: &WekaComposeOptions,
) -> Result<Dataset> {
    let mut pool = SegmentPool::new();
    let mut ledger = CacheBustLedger::default();
    let mut conversations = Vec::with_capacity(convs.len());

    // Session ids that some spawn branch targets as a child. Such conversations
    // need lineage `DagMetadata` even when they declare no branches of their own,
    // so the DAG validation (`branch references unknown child` / `child not
    // referenced by parent`) is satisfied. Unrelated conversations stay `dag: None`.
    let spawn_child_ids: HashSet<&str> = convs
        .iter()
        .flat_map(|c| c.turns.iter())
        .filter_map(|t| t.spawn_branch.as_ref())
        .flat_map(|sb| sb.child_session_ids.iter().map(String::as_str))
        .collect();

    for (traj_index, conv) in convs.iter().enumerate() {
        // The trajectory tree's root correlation (subagent/flat children share it).
        let correlation = conv
            .parent_conversation_id
            .clone()
            .unwrap_or_else(|| conv.session_id.clone());
        let marker = resolve_tree_marker(
            &mut ledger,
            &correlation,
            &opts.benchmark_id,
            traj_index as i64,
            &conv.session_id,
            opts.cache_bust_target,
        );

        let mut turns = Vec::with_capacity(conv.turns.len());
        let mut conv_branches: Vec<ConversationBranch> = Vec::new();
        let mut parent = None;
        for (i, t) in conv.turns.iter().enumerate() {
            let req_opts = ChatRequestOptions {
                streaming: opts.streaming,
                ignore_eos: opts.ignore_eos,
                // The marker rides the first turn only; it is then part of the
                // self-contained prefix of every later turn's replayed body.
                cache_bust_marker: if i == 0 { marker.clone() } else { None },
            };
            let body = chat_request_body(&t.model, &t.raw_messages, t.max_tokens, &req_opts);
            let wire = Bytes::from(serde_json::to_vec(&body)?);
            let handle = pool.intern_raw(parent, wire)?;
            parent = Some(handle);

            let mut turn = Turn {
                role: Some(Role::from("user")),
                model: Some(ModelId::from(t.model.as_str())),
                max_tokens: u32::try_from(t.max_tokens.max(1)).ok(),
                streaming: Some(opts.streaming),
                timestamp_ms: t.timestamp_ms,
                delay_ms: t.delay_ms,
                body: Turn::dispatch_body(Some(handle), None, &[]),
                ..Turn::default()
            };

            // Surface the reconstructed subagent spawn/join structure onto the
            // composed turn: a spawn declares a branch (recorded on the turn and
            // the conversation DAG); a join gates the turn on child completion.
            if let Some(sb) = &t.spawn_branch {
                let bid = BranchId::from(sb.branch_id.as_str());
                turn.branch_ids.push(bid.clone());
                conv_branches.push(ConversationBranch {
                    branch_id: bid,
                    child_conversation_ids: sb
                        .child_session_ids
                        .iter()
                        .map(|s| SessionId::from(s.clone()))
                        .collect(),
                    mode: if sb.mode_fork {
                        ConversationBranchMode::Fork
                    } else {
                        ConversationBranchMode::Spawn
                    },
                    dispatch_timing: DispatchTiming::Post,
                    background: sb.background,
                });
            }
            if let Some(jp) = &t.join_prerequisite {
                turn.prerequisites.push(TurnPrerequisite {
                    kind: PrerequisiteKind::SpawnJoin,
                    branch_id: Some(BranchId::from(jp.branch_id.as_str())),
                    child_conversation_ids: jp
                        .child_session_ids
                        .iter()
                        .map(|s| SessionId::from(s.clone()))
                        .collect(),
                    barrier_id: None,
                    timer_seconds: None,
                    event_name: None,
                });
            }

            turns.push(turn);
        }

        let session_id = SessionId::from(conv.session_id.clone());
        let is_root = conv.parent_conversation_id.is_none();
        let parent_id = conv
            .parent_conversation_id
            .as_ref()
            .map(|p| SessionId::from(p.clone()));
        let root_id = SessionId::from(conv.replay_scope_id.clone());
        // Attach a DAG when the conversation either declares spawn branches or is
        // itself a spawn-branch child (needs lineage); linear conversations with
        // no subagent relationship leave `dag` absent.
        let dag = if !conv_branches.is_empty() {
            Some(DagMetadata {
                branches: conv_branches.into_iter().collect(),
                is_root,
                agent_depth: if is_root { 0 } else { 1 },
                parent_conversation_id: parent_id,
                root_conversation_id: root_id,
            })
        } else if spawn_child_ids.contains(conv.session_id.as_str()) {
            Some(DagMetadata {
                branches: Default::default(),
                is_root: false,
                agent_depth: 1,
                parent_conversation_id: parent_id,
                root_conversation_id: root_id,
            })
        } else {
            None
        };

        conversations.push(Conversation {
            session_id,
            turns,
            system: None,
            user_context: None,
            context_mode: None,
            accuracy: None,
            dag,
        });
    }

    Dataset::new(
        conversations,
        Arc::new(pool.freeze()),
        "sequential",
        ConversationContextMode::MessageArrayWithResponses,
    )
    .map_err(|error| anyhow::anyhow!(error.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agentx::loader::{ReconstructedConversation, ReconstructedTurn};
    use crate::agentx::synth::ChatMessage;

    fn turn(ts: f64, content: &str) -> ReconstructedTurn {
        ReconstructedTurn {
            timestamp_ms: Some(ts),
            delay_ms: None,
            api_time_ms: None,
            source_trace_id: "t".into(),
            source_outer_idx: 0,
            source_kind: "weka_main".into(),
            model: "m".into(),
            max_tokens: 8,
            raw_messages: vec![ChatMessage::plain("user", content.to_string())],
            reset_context: false,
            theoretical_prefix_cache_hit_blocks: 0,
            theoretical_prefix_cache_total_blocks: 1,
            input_kind: None,
            spawn_branch: None,
            join_prerequisite: None,
        }
    }

    #[test]
    fn slice_excludes_history_and_emits_warmup_at_n_minus_1() {
        // Turns at ts 0/100/200; ratios 0.5/0.5 -> t* = min + 0.5*dur = 100
        // (hi==lo, so no RNG). next_idx = first ts>=100 = 1. warmup = turn 0.
        let conv = ReconstructedConversation {
            session_id: "t".into(),
            replay_scope_id: "t".into(),
            parent_conversation_id: None,
            turns: vec![turn(0.0, "a"), turn(100.0, "b"), turn(200.0, "c")],
        };
        let out = slice_trajectories_at_tstar(vec![conv], 0, 0.5, 0.5, None);
        // One warmup conv (turn n-1) + one profiling conv (turns from t*).
        assert_eq!(out.len(), 2);
        let warm = out.iter().find(|c| c.session_id.ends_with(WARMUP_SUFFIX)).unwrap();
        assert_eq!(warm.turns.len(), 1);
        assert_eq!(warm.turns[0].max_tokens, 1); // _WARMUP_MAX_TOKENS
        assert_eq!(warm.turns[0].timestamp_ms, Some(100.0)); // lead = t* - warm_ts = 100
        assert_eq!(warm.parent_conversation_id.as_deref(), Some("t")); // shared tree marker
        let prof = out.iter().find(|c| !c.session_id.ends_with(WARMUP_SUFFIX)).unwrap();
        // History (turn 0) excluded; profiling turns rebased to t*-relative offsets.
        assert_eq!(prof.turns.len(), 2);
        assert_eq!(prof.turns[0].timestamp_ms, Some(0.0)); // ts 100 - t* 100
        assert_eq!(prof.turns[1].timestamp_ms, Some(100.0)); // ts 200 - t* 100
    }

    fn opts() -> WekaComposeOptions {
        WekaComposeOptions {
            streaming: true,
            ignore_eos: true,
            benchmark_id: "bench".into(),
            cache_bust_target: CacheBustTarget::FirstTurnPrefix,
        }
    }

    #[test]
    fn composer_preserves_spawn_join_metadata() {
        use crate::agentx::loader::{JoinPrerequisite, SpawnBranch};
        let mut t0 = turn(0.0, "a");
        t0.spawn_branch = Some(SpawnBranch {
            branch_id: "br:a".into(),
            child_session_ids: vec!["t::sa:a".into()],
            background: false,
            mode_fork: false,
        });
        let t1 = turn(1000.0, "b");
        let mut t2 = turn(2000.0, "c");
        t2.join_prerequisite = Some(JoinPrerequisite {
            branch_id: "br:a".into(),
            child_session_ids: vec!["t::sa:a".into()],
        });
        let root = ReconstructedConversation {
            session_id: "t".into(),
            replay_scope_id: "t".into(),
            parent_conversation_id: None,
            turns: vec![t0, t1, t2],
        };
        // The spawned child conversation the branch targets; its lineage is
        // surfaced onto the composed dataset so the DAG validates.
        let child = ReconstructedConversation {
            session_id: "t::sa:a".into(),
            replay_scope_id: "t".into(),
            parent_conversation_id: Some("t".into()),
            turns: vec![turn(0.0, "child")],
        };
        let ds = compose_weka_agentic_dataset(&[root, child], &opts()).unwrap();
        let conv = &ds.conversations()[0];
        assert!(!conv.turns[0].branch_ids.is_empty());
        assert_eq!(conv.turns[0].branch_ids[0].as_str(), "br:a");
        // The spawn branch is attached to the conversation DAG.
        let dag = conv.dag.as_ref().expect("dag present");
        assert_eq!(dag.branches.len(), 1);
        assert_eq!(dag.branches[0].mode, crate::dataset::model::ConversationBranchMode::Spawn);
        let pre = &conv.turns[2].prerequisites[0];
        assert_eq!(pre.kind, crate::dataset::model::PrerequisiteKind::SpawnJoin);
        assert_eq!(
            pre.child_conversation_ids.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
            vec!["t::sa:a"]
        );
    }

    #[test]
    fn composes_verbatim_turns_with_timestamps_and_marker() {
        let conv = ReconstructedConversation {
            session_id: "t".into(),
            replay_scope_id: "t".into(),
            parent_conversation_id: None,
            turns: vec![turn(0.0, "hello"), turn(1000.0, "again")],
        };
        let ds = compose_weka_agentic_dataset(
            std::slice::from_ref(&conv),
            &WekaComposeOptions {
                streaming: true,
                ignore_eos: true,
                benchmark_id: "bench".into(),
                cache_bust_target: CacheBustTarget::FirstTurnPrefix,
            },
        )
        .unwrap();
        assert_eq!(ds.conversations().len(), 1);
        let c = &ds.conversations()[0];
        assert_eq!(c.turns.len(), 2);
        assert_eq!(c.turns[0].timestamp_ms, Some(0.0));
        assert_eq!(c.turns[1].timestamp_ms, Some(1000.0));
        assert_eq!(c.turns[0].max_tokens, Some(8));
    }
}
