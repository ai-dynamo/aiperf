// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Transport-neutral inputs and the statistics primitive for dataset analysis.
//!
//! This module holds the pure-logic foundation for the `--dry-run` dataset
//! report: neutral per-record and per-turn observation structs, plus
//! [`stat_summary`], a distribution summarizer used across the report sections.

pub mod prefix_cache;

pub use prefix_cache::{IdealReuse, IdentitySource, RequestBlocks, ideal_reuse};

/// A single observed request record, expressed in transport-neutral terms.
#[derive(Debug, Clone)]
pub struct AnalyzedRecord {
    /// Conversation the record belongs to.
    pub conversation_id: String,
    /// Zero-based turn index within the conversation.
    pub turn_index: usize,
    /// Request start time in nanoseconds.
    pub start_ns: i64,
    /// Request end time in nanoseconds.
    pub end_ns: i64,
    /// Admission time in nanoseconds, when available.
    pub admit_ns: Option<i64>,
    /// First-token time in nanoseconds, when available.
    pub first_token_ns: Option<i64>,
    /// Number of input tokens.
    pub input_tokens: u64,
    /// Number of output tokens.
    pub output_tokens: u64,
    /// Per-token arrival times in nanoseconds.
    pub token_arrival_ns: Vec<i64>,
}

/// A single planned turn, expressed in transport-neutral terms.
#[derive(Debug, Clone)]
pub struct AnalyzedTurn {
    /// Conversation the turn belongs to.
    pub conversation_id: String,
    /// Zero-based turn index within the conversation.
    pub turn_index: usize,
    /// Number of input tokens.
    pub input_tokens: u64,
    /// Maximum number of output tokens requested.
    pub max_output_tokens: u64,
    /// Inter-turn delay in milliseconds, when specified.
    pub delay_ms: Option<f64>,
    /// Content-block identifiers, when available.
    pub block_ids: Option<Vec<i64>>,
    /// System-prompt handle, when available.
    pub system_handle: Option<u64>,
}

/// Summary statistics over a set of finite samples.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct StatSummary {
    /// Number of samples.
    pub count: u64,
    /// Arithmetic mean.
    pub mean: f64,
    /// Population standard deviation (divide by N).
    pub std: f64,
    /// Minimum sample.
    pub min: f64,
    /// Maximum sample.
    pub max: f64,
    /// Sum of all samples.
    pub sum: f64,
    /// 1st percentile.
    pub p1: f64,
    /// 5th percentile.
    pub p5: f64,
    /// 10th percentile.
    pub p10: f64,
    /// 25th percentile.
    pub p25: f64,
    /// 50th percentile (median).
    pub p50: f64,
    /// 75th percentile.
    pub p75: f64,
    /// 90th percentile.
    pub p90: f64,
    /// 95th percentile.
    pub p95: f64,
    /// 99th percentile.
    pub p99: f64,
}

/// Compute summary statistics over `values`. Percentiles use linear
/// interpolation between closest ranks on the sorted samples. Returns `None`
/// when `values` is empty. `values` must be finite.
pub fn stat_summary(values: &[f64]) -> Option<StatSummary> {
    if values.is_empty() {
        return None;
    }
    let mut sorted: Vec<f64> = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).expect("finite samples"));
    let n = sorted.len();
    let sum: f64 = sorted.iter().sum();
    let mean = sum / n as f64;
    let var = sorted.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
    let pct = |p: f64| -> f64 {
        if n == 1 {
            return sorted[0];
        }
        let rank = p * (n - 1) as f64;
        let lo = rank.floor() as usize;
        let hi = rank.ceil() as usize;
        sorted[lo] + (rank - lo as f64) * (sorted[hi] - sorted[lo])
    };
    Some(StatSummary {
        count: n as u64,
        mean,
        std: var.sqrt(),
        min: sorted[0],
        max: sorted[n - 1],
        sum,
        p1: pct(0.01),
        p5: pct(0.05),
        p10: pct(0.10),
        p25: pct(0.25),
        p50: pct(0.50),
        p75: pct(0.75),
        p90: pct(0.90),
        p95: pct(0.95),
        p99: pct(0.99),
    })
}

/// Structural summary of a planned dataset: how conversations and turns are
/// distributed.
#[derive(Debug, Clone, serde::Serialize)]
pub struct DatasetShape {
    /// Number of distinct conversations.
    pub conversations: u64,
    /// Total number of turns across all conversations.
    pub total_turns: u64,
    /// Number of conversations with exactly one turn.
    pub single_turn_conversations: u64,
    /// Number of conversations with more than one turn.
    pub multi_turn_conversations: u64,
    /// Distribution of turn counts per conversation, when non-empty.
    pub turns_per_conversation: Option<StatSummary>,
    /// Distinct model names referenced by the dataset. Populated by a later
    /// adapter; empty until then.
    pub models: Vec<String>,
    /// Largest zero-based turn index observed.
    pub max_turn_index: usize,
}

/// Compute the [`DatasetShape`] over a set of planned turns. Turns are grouped
/// by `conversation_id`; per-conversation turn counts feed
/// `turns_per_conversation`.
pub fn dataset_shape(turns: &[AnalyzedTurn]) -> DatasetShape {
    use std::collections::BTreeMap;

    let mut per_conversation: BTreeMap<&str, usize> = BTreeMap::new();
    let mut max_turn_index = 0usize;
    for turn in turns {
        *per_conversation
            .entry(turn.conversation_id.as_str())
            .or_insert(0) += 1;
        max_turn_index = max_turn_index.max(turn.turn_index);
    }

    let counts: Vec<f64> = per_conversation.values().map(|&c| c as f64).collect();
    let single_turn_conversations = per_conversation.values().filter(|&&c| c == 1).count() as u64;
    let multi_turn_conversations = per_conversation.values().filter(|&&c| c > 1).count() as u64;

    DatasetShape {
        conversations: per_conversation.len() as u64,
        total_turns: turns.len() as u64,
        single_turn_conversations,
        multi_turn_conversations,
        turns_per_conversation: stat_summary(&counts),
        models: vec![],
        max_turn_index,
    }
}

/// Sequence-length summary of a planned dataset: input, output, combined, and
/// ratio distributions plus aggregate token budgets.
#[derive(Debug, Clone, serde::Serialize)]
pub struct LengthStats {
    /// Input-sequence-length distribution, when non-empty.
    pub isl: Option<StatSummary>,
    /// Output-sequence-length distribution, when non-empty.
    pub osl: Option<StatSummary>,
    /// Combined input-plus-output length distribution, when non-empty.
    pub total: Option<StatSummary>,
    /// Distribution of input/output ratios over turns with positive output,
    /// when non-empty.
    pub isl_osl_ratio: Option<StatSummary>,
    /// Sum of input tokens across all turns.
    pub total_prompt_tokens: u64,
    /// Sum of maximum output tokens across all turns.
    pub total_completion_tokens: u64,
    /// Sum of input and output token budgets across all turns.
    pub grand_total_tokens: u64,
}

/// Compute the [`LengthStats`] over a set of planned turns. Ratios are computed
/// only for turns with a positive output budget.
pub fn length_stats(turns: &[AnalyzedTurn]) -> LengthStats {
    let mut isl = Vec::with_capacity(turns.len());
    let mut osl = Vec::with_capacity(turns.len());
    let mut total = Vec::with_capacity(turns.len());
    let mut ratio = Vec::new();
    let mut total_prompt_tokens = 0u64;
    let mut total_completion_tokens = 0u64;
    for turn in turns {
        let input = turn.input_tokens as f64;
        let output = turn.max_output_tokens as f64;
        isl.push(input);
        osl.push(output);
        total.push(input + output);
        if turn.max_output_tokens > 0 {
            ratio.push(input / output);
        }
        total_prompt_tokens += turn.input_tokens;
        total_completion_tokens += turn.max_output_tokens;
    }

    LengthStats {
        isl: stat_summary(&isl),
        osl: stat_summary(&osl),
        total: stat_summary(&total),
        isl_osl_ratio: stat_summary(&ratio),
        total_prompt_tokens,
        total_completion_tokens,
        grand_total_tokens: total_prompt_tokens + total_completion_tokens,
    }
}

/// Statistics for a single turn index across all conversations that reach it.
#[derive(Debug, Clone, serde::Serialize)]
pub struct TurnIndexStat {
    /// Zero-based turn index this row describes.
    pub turn_index: usize,
    /// Number of conversations that have a turn at this index.
    pub conversations_reaching: u64,
    /// Input-sequence-length distribution at this index, when non-empty.
    pub isl: Option<StatSummary>,
    /// Output-sequence-length distribution at this index, when non-empty.
    pub osl: Option<StatSummary>,
    /// Mean input-token growth from the previous index, averaged over
    /// conversations present at both this index and the prior one. `None` at
    /// index 0 or when no conversation spans both indices.
    pub mean_history_growth: Option<f64>,
    /// Authored inter-turn think-time distribution (from `delay_ms`) at this
    /// index, when any delays are specified.
    pub authored_think_time_ms: Option<StatSummary>,
}

/// Per-turn-index breakdown of a planned dataset.
#[derive(Debug, Clone, serde::Serialize)]
pub struct TurnStats {
    /// One row per observed turn index, ordered ascending.
    pub by_index: Vec<TurnIndexStat>,
}

/// Compute the [`TurnStats`] over a set of planned turns. Turns are grouped by
/// `turn_index`. `mean_history_growth` at index `i` (`i > 0`) is the mean over
/// conversations present at both `i` and `i - 1` of `isl_i - isl_{i-1}`.
pub fn turn_stats(turns: &[AnalyzedTurn]) -> TurnStats {
    use std::collections::BTreeMap;

    let mut by_index: BTreeMap<usize, Vec<&AnalyzedTurn>> = BTreeMap::new();
    // ISL keyed by (conversation, turn index) for cross-index growth deltas.
    let mut isl_by_key: BTreeMap<(&str, usize), u64> = BTreeMap::new();
    for turn in turns {
        by_index.entry(turn.turn_index).or_default().push(turn);
        isl_by_key.insert(
            (turn.conversation_id.as_str(), turn.turn_index),
            turn.input_tokens,
        );
    }

    let rows = by_index
        .into_iter()
        .map(|(turn_index, group)| {
            let isl: Vec<f64> = group.iter().map(|t| t.input_tokens as f64).collect();
            let osl: Vec<f64> = group.iter().map(|t| t.max_output_tokens as f64).collect();
            let think: Vec<f64> = group.iter().filter_map(|t| t.delay_ms).collect();

            let mean_history_growth = if turn_index == 0 {
                None
            } else {
                let deltas: Vec<f64> = group
                    .iter()
                    .filter_map(|t| {
                        let prev = isl_by_key.get(&(t.conversation_id.as_str(), turn_index - 1))?;
                        Some(t.input_tokens as f64 - *prev as f64)
                    })
                    .collect();
                if deltas.is_empty() {
                    None
                } else {
                    Some(deltas.iter().sum::<f64>() / deltas.len() as f64)
                }
            };

            TurnIndexStat {
                turn_index,
                conversations_reaching: group.len() as u64,
                isl: stat_summary(&isl),
                osl: stat_summary(&osl),
                mean_history_growth,
                authored_think_time_ms: stat_summary(&think),
            }
        })
        .collect();

    TurnStats { by_index: rows }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stat_summary_matches_hand_computed() {
        let v = [1.0, 2.0, 3.0, 4.0, 5.0];
        let s = stat_summary(&v).expect("non-empty");
        assert_eq!(s.count, 5);
        assert_eq!(s.min, 1.0);
        assert_eq!(s.max, 5.0);
        assert_eq!(s.sum, 15.0);
        assert_eq!(s.mean, 3.0);
        assert_eq!(s.p50, 3.0);
        // population variance of 1..5 = 2.0 → std = sqrt(2)
        assert!((s.std - 2.0_f64.sqrt()).abs() < 1e-9);
        // p90 by linear interpolation on rank: idx = 0.90*(5-1)=3.6 → 4.0 + 0.6*(5-4)=4.6
        assert!((s.p90 - 4.6).abs() < 1e-9);
    }

    #[test]
    fn stat_summary_empty_is_none() {
        assert!(stat_summary(&[]).is_none());
    }

    fn fixture_turns() -> Vec<AnalyzedTurn> {
        // conversation "a": 2 turns; conversation "b": 1 turn
        vec![
            AnalyzedTurn {
                conversation_id: "a".into(),
                turn_index: 0,
                input_tokens: 100,
                max_output_tokens: 20,
                delay_ms: None,
                block_ids: None,
                system_handle: None,
            },
            AnalyzedTurn {
                conversation_id: "a".into(),
                turn_index: 1,
                input_tokens: 150,
                max_output_tokens: 30,
                delay_ms: Some(500.0),
                block_ids: None,
                system_handle: None,
            },
            AnalyzedTurn {
                conversation_id: "b".into(),
                turn_index: 0,
                input_tokens: 200,
                max_output_tokens: 40,
                delay_ms: None,
                block_ids: None,
                system_handle: None,
            },
        ]
    }

    #[test]
    fn shape_counts_conversations_and_turns() {
        let s = dataset_shape(&fixture_turns());
        assert_eq!(s.conversations, 2);
        assert_eq!(s.total_turns, 3);
        assert_eq!(s.single_turn_conversations, 1);
        assert_eq!(s.multi_turn_conversations, 1);
        assert_eq!(s.max_turn_index, 1);
        assert_eq!(s.turns_per_conversation.unwrap().mean, 1.5);
    }

    #[test]
    fn length_stats_sums_token_budgets() {
        let l = length_stats(&fixture_turns());
        assert_eq!(l.total_prompt_tokens, 450);
        assert_eq!(l.total_completion_tokens, 90);
        assert_eq!(l.grand_total_tokens, 540);
        assert_eq!(l.isl.unwrap().max, 200.0);
        assert_eq!(l.osl.unwrap().min, 20.0);
    }

    #[test]
    fn turn_stats_tracks_history_growth() {
        let t = turn_stats(&fixture_turns());
        assert_eq!(t.by_index.len(), 2);
        assert_eq!(t.by_index[0].turn_index, 0);
        assert_eq!(t.by_index[0].conversations_reaching, 2);
        assert_eq!(t.by_index[1].conversations_reaching, 1);
        // turn 1 ISL 150 minus turn 0 ISL 100 in conversation "a" = 50
        assert_eq!(t.by_index[1].mean_history_growth.unwrap(), 50.0);
        assert_eq!(
            t.by_index[1].authored_think_time_ms.as_ref().unwrap().mean,
            500.0
        );
    }
}
