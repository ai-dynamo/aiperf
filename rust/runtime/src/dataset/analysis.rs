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
/// only for turns with a positive output budget. Accepts anything iterable as
/// `&AnalyzedTurn` so callers can pass a slice or a filtered borrow without
/// cloning.
pub fn length_stats<'a>(turns: impl IntoIterator<Item = &'a AnalyzedTurn>) -> LengthStats {
    let mut isl = Vec::new();
    let mut osl = Vec::new();
    let mut total = Vec::new();
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

/// Concurrency (inflight-request) statistics over an execution timeline.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ConcurrencyStats {
    /// Peak number of simultaneously inflight requests.
    pub peak: u64,
    /// Time-weighted average inflight count over the run duration.
    pub time_weighted_avg: f64,
    /// Inflight count at each change point, as `(rel_seconds, inflight)` pairs
    /// where `rel_seconds` is measured from the first request start.
    pub samples: Vec<(f64, u64)>,
}

/// Throughput statistics over an execution timeline.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ThroughputStats {
    /// Completed requests per second over the run duration.
    pub requests_per_s: f64,
    /// Output tokens per second over the run duration.
    pub output_tokens_per_s: f64,
    /// Wall-clock run duration in seconds (`max end - min start`).
    pub run_duration_s: f64,
}

/// Queue/backlog statistics over an execution timeline.
#[derive(Debug, Clone, serde::Serialize)]
pub struct QueueStats {
    /// Per-record `start_ns - admit_ns` queue delay in milliseconds, over
    /// records with an admission time, when non-empty.
    pub queue_delay_ms: Option<StatSummary>,
}

/// Execution-timeline summary: concurrency, throughput, and queue backlog.
#[derive(Debug, Clone, serde::Serialize)]
pub struct TimelineStats {
    /// Inflight-concurrency statistics.
    pub concurrency: ConcurrencyStats,
    /// Throughput statistics.
    pub throughput: ThroughputStats,
    /// Queue/backlog statistics.
    pub queue: QueueStats,
}

/// Compute the [`TimelineStats`] over a set of observed records. Concurrency is
/// derived by a sweep line over `[start_ns, end_ns)` intervals; on ties an end
/// event is processed before a start event so a back-to-back handoff does not
/// inflate the peak. Returns `None` for empty input. When the run duration is
/// zero, rate outputs are `0.0` rather than non-finite.
pub fn timeline_stats(records: &[AnalyzedRecord]) -> Option<TimelineStats> {
    if records.is_empty() {
        return None;
    }

    let min_start = records.iter().map(|r| r.start_ns).min().unwrap();
    let max_end = records.iter().map(|r| r.end_ns).max().unwrap();
    let duration_ns = (max_end - min_start).max(0) as f64;
    let run_duration_s = duration_ns / 1e9;

    // Sweep-line events: +1 at start, -1 at end. Ties process -1 before +1.
    let mut events: Vec<(i64, i64)> = Vec::with_capacity(records.len() * 2);
    for r in records {
        events.push((r.start_ns, 1));
        events.push((r.end_ns, -1));
    }
    events.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));

    let mut inflight: i64 = 0;
    let mut peak: i64 = 0;
    let mut integral = 0.0f64;
    let mut prev_ns = min_start;
    let mut samples: Vec<(f64, u64)> = Vec::new();
    for (ns, delta) in events {
        // Accumulate time-weighted integral over the segment just ended.
        integral += inflight as f64 * (ns - prev_ns) as f64;
        prev_ns = ns;
        inflight += delta;
        peak = peak.max(inflight);
        let rel_s = (ns - min_start) as f64 / 1e9;
        // Record inflight at this change point, coalescing same-instant events.
        if let Some(last) = samples.last_mut() {
            if (last.0 - rel_s).abs() < f64::EPSILON {
                last.1 = inflight as u64;
                continue;
            }
        }
        samples.push((rel_s, inflight as u64));
    }

    let time_weighted_avg = if duration_ns > 0.0 {
        integral / duration_ns
    } else {
        0.0
    };

    let count = records.len() as f64;
    let total_output: u64 = records.iter().map(|r| r.output_tokens).sum();
    let (requests_per_s, output_tokens_per_s) = if run_duration_s > 0.0 {
        (count / run_duration_s, total_output as f64 / run_duration_s)
    } else {
        (0.0, 0.0)
    };

    let queue_delays: Vec<f64> = records
        .iter()
        .filter_map(|r| r.admit_ns.map(|a| (r.start_ns - a) as f64 / 1e6))
        .collect();

    Some(TimelineStats {
        concurrency: ConcurrencyStats {
            peak: peak.max(0) as u64,
            time_weighted_avg,
            samples,
        },
        throughput: ThroughputStats {
            requests_per_s,
            output_tokens_per_s,
            run_duration_s,
        },
        queue: QueueStats {
            queue_delay_ms: stat_summary(&queue_delays),
        },
    })
}

/// Realized/ideal prefix-cache reuse analysis for the dataset report.
#[derive(Debug, Clone, serde::Serialize)]
pub struct CacheReuseAnalysis {
    /// How the block identities feeding the analysis were derived.
    pub identity_source: prefix_cache::IdentitySource,
    /// Ideal (unbounded, no-eviction) reuse statistics.
    pub ideal: prefix_cache::IdealReuse,
    /// Realized finite-capacity LRU hit-rate curve in ascending capacity order.
    pub realized: Vec<prefix_cache::CacheCurvePoint>,
    /// Block size in tokens used to derive the identities.
    pub block_size: u32,
}

/// Full `--dry-run` dataset analysis: structural, length, per-turn, cache-reuse,
/// and execution-timeline sections.
#[derive(Debug, Clone, serde::Serialize)]
pub struct DatasetAnalysis {
    /// Structural summary (conversations, turns).
    pub shape: DatasetShape,
    /// Sequence-length summary.
    pub lengths: LengthStats,
    /// Per-turn-index breakdown.
    pub turns: TurnStats,
    /// Prefix-cache reuse analysis, when usable block identities exist.
    pub cache: Option<CacheReuseAnalysis>,
    /// Execution-timeline summary, when records are present.
    pub timeline: Option<TimelineStats>,
    /// Per-conversation length breakdown, emitted only when
    /// [`AnalysisOptions::per_conversation`] is set. `None` when the breakdown
    /// was not requested; empty vector when requested over an empty dataset.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub conversations: Option<Vec<ConversationSummary>>,
}

/// Length breakdown for a single conversation, emitted under the
/// `--dataset-analysis-per-conversation` knob.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ConversationSummary {
    /// Stable conversation identity (endpoint `conversation_id` or a synthesized
    /// session key), matching the ids used elsewhere in the analysis.
    pub conversation_id: String,
    /// Number of turns planned for this conversation.
    pub turns: u64,
    /// Sequence-length summary restricted to this conversation's turns.
    pub lengths: LengthStats,
}

/// Options controlling dataset analysis.
#[derive(Debug, Clone)]
pub struct AnalysisOptions {
    /// Block size in tokens for length-structure identity synthesis.
    pub block_size: u32,
    /// When set, an additional realized-curve point at this explicit LRU
    /// capacity (in blocks) is appended to the sweep.
    pub explicit_cache_blocks: Option<u64>,
    /// When set, [`analyze`] emits a per-conversation length breakdown
    /// ([`DatasetAnalysis::conversations`]). Off by default.
    pub per_conversation: bool,
}

impl Default for AnalysisOptions {
    fn default() -> Self {
        Self {
            block_size: 16,
            explicit_cache_blocks: None,
            per_conversation: false,
        }
    }
}

/// Extract per-request block-id sequences and the identity source they were
/// derived from.
///
/// If any turn carries precomputed `block_ids`, those are used directly and the
/// source is [`IdentitySource::HashIds`]. Otherwise block ids are synthesized
/// from sequence-length structure ([`IdentitySource::LengthStructure`]): within
/// a conversation, turn `i` reuses turn `i - 1`'s leading
/// `floor(prev_isl / block_size)` blocks so multi-turn prefixes chain, and a
/// shared `system_handle` maps to a shared leading block run across
/// conversations. The [`IdentitySource::TokenBlocks`] variant is reserved for a
/// later materialized-token path and is not produced here.
fn request_blocks(
    turns: &[AnalyzedTurn],
    opts: &AnalysisOptions,
) -> (
    prefix_cache::IdentitySource,
    Vec<prefix_cache::RequestBlocks>,
) {
    use std::collections::{BTreeMap, HashMap};

    let block_size = (opts.block_size as u64).max(1);

    // Direct path: precomputed content hashes.
    if turns.iter().any(|t| t.block_ids.is_some()) {
        let blocks = turns
            .iter()
            .map(|t| prefix_cache::RequestBlocks {
                conversation_id: t.conversation_id.clone(),
                turn_index: t.turn_index,
                block_ids: t.block_ids.clone().unwrap_or_default(),
            })
            .collect();
        return (prefix_cache::IdentitySource::HashIds, blocks);
    }

    // Fallback: synthesize block ids from length structure. Ids are drawn from a
    // single monotone id space so distinct block runs never collide.
    let mut by_conversation: BTreeMap<&str, Vec<&AnalyzedTurn>> = BTreeMap::new();
    for turn in turns {
        by_conversation
            .entry(turn.conversation_id.as_str())
            .or_default()
            .push(turn);
    }

    let mut next_id: i64 = 1;
    // Shared leading block run per system-prompt handle.
    let mut system_runs: HashMap<u64, Vec<i64>> = HashMap::new();
    let mut out: Vec<prefix_cache::RequestBlocks> = Vec::with_capacity(turns.len());

    for (_conversation, mut group) in by_conversation {
        group.sort_by_key(|t| t.turn_index);
        let mut prev_ids: Vec<i64> = Vec::new();
        let mut prev_isl: u64 = 0;

        for (position, turn) in group.iter().enumerate() {
            let n_blocks = turn.input_tokens.div_ceil(block_size) as usize;
            let mut ids: Vec<i64> = Vec::with_capacity(n_blocks);

            if position == 0 {
                // Leading shared run from the system handle, when present.
                if let Some(handle) = turn.system_handle {
                    let run = system_runs.entry(handle).or_insert_with(|| {
                        let id = next_id;
                        next_id += 1;
                        vec![id]
                    });
                    let take = run.len().min(n_blocks);
                    ids.extend_from_slice(&run[..take]);
                }
            } else {
                // Reuse the previous turn's leading whole blocks.
                let reuse = (prev_isl / block_size) as usize;
                let reuse = reuse.min(prev_ids.len()).min(n_blocks);
                ids.extend_from_slice(&prev_ids[..reuse]);
            }

            while ids.len() < n_blocks {
                ids.push(next_id);
                next_id += 1;
            }

            prev_ids = ids.clone();
            prev_isl = turn.input_tokens;
            out.push(prefix_cache::RequestBlocks {
                conversation_id: turn.conversation_id.clone(),
                turn_index: turn.turn_index,
                block_ids: ids,
            });
        }
    }

    (prefix_cache::IdentitySource::LengthStructure, out)
}

/// Assemble the full [`DatasetAnalysis`] from planned turns and observed records.
///
/// Section builders (`dataset_shape`, `length_stats`, `turn_stats`,
/// `timeline_stats`) are combined with a prefix-cache reuse analysis. Block
/// identities are extracted by [`request_blocks`], then joined to record
/// `start_ns` on `(conversation_id, turn_index)` and sorted into arrival order
/// for both the ideal and realized reuse computations. When
/// `opts.explicit_cache_blocks` is set, an extra realized point at that capacity
/// is appended. `cache` is `None` when no usable block identities exist. Every
/// serialized `f64` is guarded to a finite value (non-finite becomes `0.0`).
pub fn analyze(
    turns: &[AnalyzedTurn],
    records: &[AnalyzedRecord],
    opts: &AnalysisOptions,
) -> DatasetAnalysis {
    use std::collections::HashMap;

    let shape = dataset_shape(turns);
    let lengths = length_stats(turns);
    let turns_stats = turn_stats(turns);
    let timeline = timeline_stats(records);

    let (identity_source, blocks) = request_blocks(turns, opts);
    let cache = if blocks.iter().any(|b| !b.block_ids.is_empty()) {
        // Join to record start times to establish arrival order.
        let mut start_by_key: HashMap<(&str, usize), i64> = HashMap::new();
        for r in records {
            start_by_key.insert((r.conversation_id.as_str(), r.turn_index), r.start_ns);
        }
        let mut arrival = blocks;
        arrival.sort_by_key(|b| {
            start_by_key
                .get(&(b.conversation_id.as_str(), b.turn_index))
                .copied()
                .unwrap_or(i64::MAX)
        });

        let ideal = prefix_cache::ideal_reuse(&arrival);
        let mut realized = prefix_cache::realized_sweep(&arrival, ideal.unique_blocks);
        if let Some(capacity) = opts.explicit_cache_blocks {
            realized.push(prefix_cache::realized_reuse(&arrival, capacity));
        }

        Some(CacheReuseAnalysis {
            identity_source,
            ideal,
            realized,
            block_size: opts.block_size,
        })
    } else {
        None
    };

    let mut analysis = DatasetAnalysis {
        shape,
        lengths,
        turns: turns_stats,
        cache,
        timeline,
        conversations: opts
            .per_conversation
            .then(|| per_conversation_summaries(turns)),
    };
    sanitize_analysis(&mut analysis);
    analysis
}

/// Group `turns` by conversation id (ascending) and compute a [`LengthStats`]
/// per group, restricted to that conversation's turns.
fn per_conversation_summaries(turns: &[AnalyzedTurn]) -> Vec<ConversationSummary> {
    use std::collections::BTreeMap;

    let mut by_conversation: BTreeMap<&str, Vec<&AnalyzedTurn>> = BTreeMap::new();
    for turn in turns {
        by_conversation
            .entry(turn.conversation_id.as_str())
            .or_default()
            .push(turn);
    }

    by_conversation
        .into_iter()
        .map(|(conversation_id, group)| ConversationSummary {
            conversation_id: conversation_id.to_string(),
            turns: group.len() as u64,
            lengths: length_stats(group.iter().copied()),
        })
        .collect()
}

/// Replace a non-finite value (NaN/Inf) with `0.0`; pass finite values through.
fn finite(x: f64) -> f64 {
    if x.is_finite() { x } else { 0.0 }
}

/// Guard every `f64` in a [`StatSummary`] to a finite value.
fn sanitize_stat(s: &mut StatSummary) {
    s.mean = finite(s.mean);
    s.std = finite(s.std);
    s.min = finite(s.min);
    s.max = finite(s.max);
    s.sum = finite(s.sum);
    s.p1 = finite(s.p1);
    s.p5 = finite(s.p5);
    s.p10 = finite(s.p10);
    s.p25 = finite(s.p25);
    s.p50 = finite(s.p50);
    s.p75 = finite(s.p75);
    s.p90 = finite(s.p90);
    s.p95 = finite(s.p95);
    s.p99 = finite(s.p99);
}

/// Guard every serialized `f64` reachable from a [`DatasetAnalysis`].
fn sanitize_analysis(a: &mut DatasetAnalysis) {
    if let Some(s) = a.shape.turns_per_conversation.as_mut() {
        sanitize_stat(s);
    }
    for opt in [
        &mut a.lengths.isl,
        &mut a.lengths.osl,
        &mut a.lengths.total,
        &mut a.lengths.isl_osl_ratio,
    ] {
        if let Some(s) = opt.as_mut() {
            sanitize_stat(s);
        }
    }
    if let Some(conversations) = a.conversations.as_mut() {
        for summary in conversations.iter_mut() {
            for opt in [
                &mut summary.lengths.isl,
                &mut summary.lengths.osl,
                &mut summary.lengths.total,
                &mut summary.lengths.isl_osl_ratio,
            ] {
                if let Some(s) = opt.as_mut() {
                    sanitize_stat(s);
                }
            }
        }
    }
    for row in &mut a.turns.by_index {
        if let Some(s) = row.isl.as_mut() {
            sanitize_stat(s);
        }
        if let Some(s) = row.osl.as_mut() {
            sanitize_stat(s);
        }
        if let Some(v) = row.mean_history_growth.as_mut() {
            *v = finite(*v);
        }
        if let Some(s) = row.authored_think_time_ms.as_mut() {
            sanitize_stat(s);
        }
    }
    if let Some(c) = a.cache.as_mut() {
        c.ideal.hit_rate = finite(c.ideal.hit_rate);
        for p in &mut c.realized {
            p.hit_rate = finite(p.hit_rate);
        }
    }
    if let Some(t) = a.timeline.as_mut() {
        t.concurrency.time_weighted_avg = finite(t.concurrency.time_weighted_avg);
        for sample in &mut t.concurrency.samples {
            sample.0 = finite(sample.0);
        }
        t.throughput.requests_per_s = finite(t.throughput.requests_per_s);
        t.throughput.output_tokens_per_s = finite(t.throughput.output_tokens_per_s);
        t.throughput.run_duration_s = finite(t.throughput.run_duration_s);
        if let Some(s) = t.queue.queue_delay_ms.as_mut() {
            sanitize_stat(s);
        }
    }
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

    fn rec(conv: &str, ti: usize, start: i64, end: i64, out: u64) -> AnalyzedRecord {
        AnalyzedRecord {
            conversation_id: conv.into(),
            turn_index: ti,
            start_ns: start,
            end_ns: end,
            admit_ns: Some(start),
            first_token_ns: Some(start),
            input_tokens: 10,
            output_tokens: out,
            token_arrival_ns: vec![],
        }
    }

    #[test]
    fn timeline_concurrency_and_throughput() {
        // r0 [0,2s), r1 [1,3s) overlap → peak concurrency 2. 2 requests over 3s.
        let recs = vec![
            rec("a", 0, 0, 2_000_000_000, 5),
            rec("b", 0, 1_000_000_000, 3_000_000_000, 5),
        ];
        let t = timeline_stats(&recs).unwrap();
        assert_eq!(t.concurrency.peak, 2);
        assert!((t.throughput.run_duration_s - 3.0).abs() < 1e-9);
        assert!((t.throughput.requests_per_s - 2.0 / 3.0).abs() < 1e-9);
        assert!((t.throughput.output_tokens_per_s - 10.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn timeline_empty_is_none() {
        assert!(timeline_stats(&[]).is_none());
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

    #[test]
    fn analyze_end_to_end_with_hash_ids() {
        let turns = vec![
            AnalyzedTurn {
                conversation_id: "a".into(),
                turn_index: 0,
                input_tokens: 32,
                max_output_tokens: 8,
                delay_ms: None,
                block_ids: Some(vec![1, 2]),
                system_handle: None,
            },
            AnalyzedTurn {
                conversation_id: "a".into(),
                turn_index: 1,
                input_tokens: 48,
                max_output_tokens: 8,
                delay_ms: None,
                block_ids: Some(vec![1, 2, 3]),
                system_handle: None,
            },
        ];
        let records = vec![
            rec("a", 0, 0, 1_000_000_000, 8),
            rec("a", 1, 1_000_000_000, 2_000_000_000, 8),
        ];
        let a = analyze(&turns, &records, &AnalysisOptions::default());
        assert_eq!(a.shape.conversations, 1);
        let cache = a.cache.clone().unwrap();
        assert!(matches!(
            cache.identity_source,
            prefix_cache::IdentitySource::HashIds
        ));
        // blocks: r0 [1,2], r1 [1,2,3] → 2 cached of 5.
        assert_eq!(cache.ideal.cached_blocks, 2);
        assert!(!cache.realized.is_empty());
        assert!(a.timeline.is_some());
        // serde round-trips without NaN
        let json = serde_json::to_string(&a).unwrap();
        assert!(!json.contains("NaN"));
    }
}
