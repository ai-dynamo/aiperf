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

use std::sync::Arc;

use anyhow::Result;
use bytes::Bytes;

use crate::agentx::cache_bust::{resolve_tree_marker, CacheBustLedger, CacheBustTarget};
use crate::agentx::loader::{ReconstructedConversation, ReconstructedTurn};
use crate::dataset::Dataset;
use crate::dataset::model::{Conversation, ConversationContextMode, SessionId, Turn};
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
        capped_warmup_lead_ms, next_turn_index_at_or_after, seed_for_trace,
        timestamped_t_star_ms,
    };
    use std::collections::BTreeMap;
    // Group conversations into trace-trees keyed by `replay_scope_id` (the root
    // trace id; subagents reconstruct with the same scope). The Python oracle
    // samples ONE t* per trace-tree and snapshots the whole tree together, so
    // subagents share the root's t* and dispatch at their parent-relative spawn
    // offsets (child turn timestamps are already in root-trace coordinates) —
    // join-gated behind the parent instead of firing free from an independent t*.
    let mut trees: BTreeMap<String, Vec<ReconstructedConversation>> = BTreeMap::new();
    for conv in convs {
        trees.entry(conv.replay_scope_id.clone()).or_default().push(conv);
    }

    let mut out = Vec::new();
    for (_tree_index, (scope, members)) in trees.into_iter().enumerate() {
        // One t* for the whole tree, sampled from the ROOT's recorded turn span.
        // Seed on the root trace id only (Python `_seed_for_trace(random_seed,
        // root_id)`) so the sampled instant byte-matches the oracle's per-trace t*.
        let root_id = members
            .iter()
            .find(|c| c.parent_conversation_id.is_none())
            .map(|r| r.session_id.clone())
            .unwrap_or_else(|| scope.clone());
        let root_ts: Vec<f64> = members
            .iter()
            .find(|c| c.parent_conversation_id.is_none())
            .map(|r| r.turns.iter().filter_map(|t| t.timestamp_ms).collect())
            .unwrap_or_default();
        let t_star = if root_ts.is_empty() {
            0.0
        } else {
            let mn = root_ts.iter().copied().fold(f64::INFINITY, f64::min);
            let mx = root_ts.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let dur = mx - mn;
            let seed = seed_for_trace(base_seed, &root_id);
            timestamped_t_star_ms(seed, mn + start_min_ratio * dur, mn + start_max_ratio * dur)
        };

        // ONE warmup prime per tree: the root's last pre-t* boundary turn carrying
        // the full flattened root prefix. Subagents share the primed prefix under
        // the shared hash-id scope, so no per-subagent warmup is emitted.
        if let Some(root) = members.iter().find(|c| c.parent_conversation_id.is_none()) {
            let rts: Vec<Option<f64>> = root.turns.iter().map(|t| t.timestamp_ms).collect();
            if let Some(root_next) = next_turn_index_at_or_after(&rts, t_star)
                && root_next >= 1
            {
                let warm_i = (root_next - 1) as usize;
                let mut warm_turn = root.turns[warm_i].clone();
                warm_turn.raw_messages = flatten_prefix(&root.turns[0..=warm_i]);
                let lead = capped_warmup_lead_ms(
                    t_star - warm_turn.timestamp_ms.unwrap_or(t_star),
                    idle_gap_cap_ms,
                );
                warm_turn.timestamp_ms = Some(lead.max(0.0));
                warm_turn.max_tokens = 1; // Python `_WARMUP_MAX_TOKENS`
                warm_turn.spawn_branch = None;
                warm_turn.join_prerequisite = None;
                out.push(ReconstructedConversation {
                    session_id: format!("{}{WARMUP_SUFFIX}", root.session_id),
                    replay_scope_id: root.replay_scope_id.clone(),
                    parent_conversation_id: Some(root.session_id.clone()),
                    turns: vec![warm_turn],
                });
            }
        }

        // PROFILING slice: every tree member (root + subagents) resumes at the
        // SHARED t*. Back-seed the flattened pre-t* prefix into the first retained
        // turn (folding a subagent's inherited parent context into its resume turn)
        // and rebase timestamps to the shared t*, so a subagent dispatches at
        // `spawn_ms − t*` — after the parent reaches the spawn point (join-gated).
        for conv in members {
            let ts: Vec<Option<f64>> = conv.turns.iter().map(|t| t.timestamp_ms).collect();
            let Some(next_idx) = next_turn_index_at_or_after(&ts, t_star) else {
                continue; // entirely pre-t* (already-drained history)
            };
            let mut prof = conv;
            let seed_prefix = flatten_prefix(&prof.turns[0..=next_idx as usize]);
            prof.turns.drain(0..next_idx as usize);
            if let Some(first) = prof.turns.first_mut() {
                first.raw_messages = seed_prefix;
            }
            for turn in &mut prof.turns {
                turn.timestamp_ms =
                    Some(turn.timestamp_ms.map_or(0.0, |ts| (ts - t_star).max(0.0)));
            }
            if !prof.turns.is_empty() {
                out.push(prof);
            }
        }
    }
    out
}

/// Flatten the per-turn deltas of `turns` into the full accumulated OpenAI
/// message array, exactly as the runtime's `DeltasWithoutResponses` accumulator
/// does (naive concatenation of each turn's delta messages; live replies are
/// folded in only at dispatch). This reproduces the recorded conversation prefix
/// the Python oracle back-seeds on resume.
fn flatten_prefix(turns: &[ReconstructedTurn]) -> Vec<crate::agentx::synth::ChatMessage> {
    turns
        .iter()
        .flat_map(|t| t.raw_messages.iter().cloned())
        .collect()
}

/// Session-id suffix marking a warmup-phase conversation (turn n-1 prime). The
/// `agentic_replay` warmup phase dispatches only these; profiling skips them.
pub const WARMUP_SUFFIX: &str = "::warmup";

/// Compose reconstructed conversations into a verbatim-replay [`Dataset`].
///
/// Turns carry their DELTA chat messages as a message-array segment (interned,
/// not an opaque `Raw` body) plus the recorded `timestamp_ms`/`delay_ms`. The
/// cache-bust marker (resolved once per trajectory tree via a shared ledger) is
/// baked into turn 0's first message. Context mode is `DeltasWithoutResponses`,
/// so the runtime materializer concatenates each turn's delta with the captured
/// live replies into the full accumulated history — matching the Python oracle.
/// `count_tokens` supplies each delta's client input-token count.
pub fn compose_weka_agentic_dataset(
    convs: &[ReconstructedConversation],
    opts: &WekaComposeOptions,
    count_tokens: &dyn Fn(&str) -> usize,
) -> Result<Dataset> {
    let mut pool = SegmentPool::new();
    let mut ledger = CacheBustLedger::default();
    let mut conversations = Vec::with_capacity(convs.len());

    // NOTE: the reconstructed subagent spawn/join structure
    // (`ReconstructedTurn.spawn_branch` / `.join_prerequisite`) is intentionally
    // NOT propagated onto the composed `Dataset` (no `ConversationBranch`, no
    // `TurnPrerequisite`, no `DagMetadata`). The dataset's DAG validator cannot be
    // satisfied once history-slicing separates a spawn declaration from its join
    // turn, and carrying the DAG would force subagent children to be non-sampleable
    // (breaking multi-worker partitioning). Join gating instead consumes a side
    // `Vec<TreeSpec>` built directly from the reconstruction and threaded into the
    // `agentic_replay` workload — so children stay plain, sampleable conversations
    // and the dataset stays DAG-free and valid under any slice.
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
        for (i, t) in conv.turns.iter().enumerate() {
            // Intern this turn's DELTA messages as a message-array segment (not an
            // opaque Raw body) so the runtime materializer accumulates the full
            // conversation prefix + captured live replies under
            // `DeltasWithoutResponses` — matching the Python oracle, whose worker
            // sends the whole growing history each turn. The cache-bust marker
            // rides turn 0's first message and then persists as the permanent
            // accumulated prefix.
            let marker_for_turn = if i == 0 { marker.as_deref() } else { None };
            let msgs_value = crate::agentx::wire::chat_messages_array(&t.raw_messages, marker_for_turn);
            let handle = pool.intern_raw(None, Bytes::from(serde_json::to_vec(&msgs_value)?))?;
            // Per-turn delta input-token count; the delta-accumulating materializer
            // sums these (plus captured reply tokens) for the ISL accounting.
            let input_tokens: u64 = t
                .raw_messages
                .iter()
                .map(|m| count_tokens(&m.content) as u64)
                .sum();
            let extra_body = opts
                .ignore_eos
                .then(|| pool.intern_raw(None, Bytes::from_static(b"{\"ignore_eos\":true}")))
                .transpose()?;

            let turn = Turn {
                role: Some(Role::from("user")),
                // `None` => the chat composer authors the run's primary `--model`,
                // not the trace's recorded per-turn model (oracle parity).
                model: None,
                max_tokens: u32::try_from(t.max_tokens.max(1)).ok(),
                input_tokens: Some(input_tokens),
                streaming: Some(opts.streaming),
                timestamp_ms: t.timestamp_ms,
                delay_ms: t.delay_ms,
                raw_messages: Some(handle),
                extra_body,
                ..Turn::default()
            };
            turns.push(turn);
        }

        conversations.push(Conversation {
            session_id: SessionId::from(conv.session_id.clone()),
            turns,
            system: None,
            user_context: None,
            context_mode: None,
            accuracy: None,
            // No DAG: subagent gating rides a side TreeSpec map, not the dataset.
            dag: None,
        });
    }

    Dataset::new(
        conversations,
        Arc::new(pool.freeze()),
        "sequential",
        ConversationContextMode::DeltasWithoutResponses,
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
    fn composer_leaves_dataset_dag_free_despite_spawn_join_metadata() {
        // Subagent spawn/join structure on the reconstruction MUST NOT be
        // propagated onto the composed Dataset: it would trip the DAG validator
        // once history-slicing separates a spawn from its join, and would make
        // children non-sampleable (breaking multi-worker). Gating rides a side
        // TreeSpec map instead — the composed dataset stays clean and valid.
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
        let child = ReconstructedConversation {
            session_id: "t::sa:a".into(),
            replay_scope_id: "t".into(),
            parent_conversation_id: Some("t".into()),
            turns: vec![turn(0.0, "child")],
        };
        // Composes without a DAG-validation error, and every conversation is
        // DAG-free with no branch/prerequisite metadata on its turns.
        let ds = compose_weka_agentic_dataset(&[root, child], &opts(), &|s| s.len()).unwrap();
        for conv in ds.conversations() {
            assert!(conv.dag.is_none(), "dataset must stay DAG-free");
            for t in &conv.turns {
                assert!(t.branch_ids.is_empty());
                assert!(t.prerequisites.is_empty());
            }
        }
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
            &|s| s.len(),
        )
        .unwrap();
        assert_eq!(ds.conversations().len(), 1);
        let c = &ds.conversations()[0];
        assert_eq!(c.turns.len(), 2);
        assert_eq!(c.turns[0].timestamp_ms, Some(0.0));
        assert_eq!(c.turns[1].timestamp_ms, Some(1000.0));
        assert_eq!(c.turns[0].max_tokens, Some(8));
        // Delta messages are interned as a message-array segment (not a raw body);
        // the dataset is `DeltasWithoutResponses` so dispatch accumulates history.
        assert!(c.turns[0].raw_messages.is_some());
    }
}
