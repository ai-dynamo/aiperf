// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Console rendering for the `--dry-run` dataset analysis.
//!
//! [`render_analysis_txt`] turns a [`DatasetAnalysis`] into a set of
//! heavy-header tables (reusing [`crate::export::console_txt::render_table`])
//! covering dataset shape, sequence lengths, the turn-by-turn breakdown, prefix
//! cache reuse (including the realized-capacity sweep), and the execution
//! timeline. Distributions include a compact fixed-width ASCII sparkline so the
//! shape reads at a glance without a plotting dependency.

use crate::dataset::analysis::{DatasetAnalysis, StatSummary};
use crate::export::console_txt::{Justify, render_table};

/// Console export width, in cells, for the analysis tables.
const WIDTH: usize = 100;

/// Stat columns shown per distribution row.
const STAT_HEADERS: &[&str] = &[
    "metric", "count", "avg", "min", "p50", "p90", "p99", "max", "std",
];

/// Render the full dataset analysis as a stack of console tables.
pub fn render_analysis_txt(a: &DatasetAnalysis) -> String {
    let mut blocks: Vec<String> = Vec::new();
    blocks.push(shape_table(a));
    blocks.push(lengths_table(a));
    blocks.push(turns_table(a));
    if let Some(cache) = cache_tables(a) {
        blocks.push(cache);
    }
    if let Some(timeline) = timeline_table(a) {
        blocks.push(timeline);
    }
    if let Some(conversations) = per_conversation_table(a) {
        blocks.push(conversations);
    }
    blocks.join("\n")
}

/// Format an `f64` with grouped magnitude and two decimals.
fn num(value: f64) -> String {
    format!("{value:.2}")
}

/// Build the stat cells (excluding the leading name) for one distribution.
fn stat_cells(name: &str, s: &StatSummary) -> Vec<String> {
    vec![
        name.to_string(),
        s.count.to_string(),
        num(s.mean),
        num(s.min),
        num(s.p50),
        num(s.p90),
        num(s.p99),
        num(s.max),
        num(s.std),
    ]
}

/// Right-justify every stat column except the leading metric name.
fn stat_justify() -> Vec<Justify> {
    let mut j = vec![Justify::Left];
    j.extend(std::iter::repeat_n(Justify::Right, STAT_HEADERS.len() - 1));
    j
}

/// Structural summary: conversation and turn counts.
fn shape_table(a: &DatasetAnalysis) -> String {
    let s = &a.shape;
    let rows = vec![
        vec!["conversations".to_string(), s.conversations.to_string()],
        vec!["total turns".to_string(), s.total_turns.to_string()],
        vec![
            "single-turn conversations".to_string(),
            s.single_turn_conversations.to_string(),
        ],
        vec![
            "multi-turn conversations".to_string(),
            s.multi_turn_conversations.to_string(),
        ],
        vec!["max turn index".to_string(), s.max_turn_index.to_string()],
    ];
    render_table(
        "NVIDIA AIPerf | Dataset Shape",
        &["property", "value"],
        &rows,
        &[Justify::Left, Justify::Right],
        WIDTH,
    )
}

/// Sequence-length distributions plus aggregate token budgets.
fn lengths_table(a: &DatasetAnalysis) -> String {
    let l = &a.lengths;
    let mut rows: Vec<Vec<String>> = Vec::new();
    for (name, opt) in [
        ("input (ISL)", &l.isl),
        ("output (OSL)", &l.osl),
        ("total", &l.total),
        ("ISL/OSL ratio", &l.isl_osl_ratio),
    ] {
        if let Some(s) = opt.as_ref() {
            rows.push(stat_cells(name, s));
        }
    }
    let lengths = render_table(
        "NVIDIA AIPerf | Sequence Lengths",
        STAT_HEADERS,
        &rows,
        &stat_justify(),
        WIDTH,
    );

    let budget_rows = vec![
        vec![
            "total prompt tokens".to_string(),
            l.total_prompt_tokens.to_string(),
        ],
        vec![
            "total completion tokens".to_string(),
            l.total_completion_tokens.to_string(),
        ],
        vec![
            "grand total tokens".to_string(),
            l.grand_total_tokens.to_string(),
        ],
    ];
    let budget = render_table(
        "NVIDIA AIPerf | Token Budget",
        &["property", "value"],
        &budget_rows,
        &[Justify::Left, Justify::Right],
        WIDTH,
    );
    format!("{lengths}\n{budget}")
}

/// Per-conversation length breakdown, rendered only when the analysis carries
/// one (i.e. `--dataset-analysis-per-conversation`). One row per conversation
/// with its turn count and input/output length means.
fn per_conversation_table(a: &DatasetAnalysis) -> Option<String> {
    let conversations = a.conversations.as_ref()?;
    let rows: Vec<Vec<String>> = conversations
        .iter()
        .map(|c| {
            let isl_avg = c
                .lengths
                .isl
                .as_ref()
                .map_or_else(|| "-".to_string(), |s| num(s.mean));
            let osl_avg = c
                .lengths
                .osl
                .as_ref()
                .map_or_else(|| "-".to_string(), |s| num(s.mean));
            vec![
                c.conversation_id.clone(),
                c.turns.to_string(),
                isl_avg,
                osl_avg,
            ]
        })
        .collect();
    Some(render_table(
        "NVIDIA AIPerf | Per-Conversation Lengths",
        &["conversation", "turns", "isl avg", "osl avg"],
        &rows,
        &[
            Justify::Left,
            Justify::Right,
            Justify::Right,
            Justify::Right,
        ],
        WIDTH,
    ))
}

/// Per-turn-index breakdown: reach, ISL/OSL, history growth, think time.
fn turns_table(a: &DatasetAnalysis) -> String {
    let headers = &[
        "turn",
        "reaching",
        "isl avg",
        "osl avg",
        "history growth",
        "think ms avg",
    ];
    let rows: Vec<Vec<String>> = a
        .turns
        .by_index
        .iter()
        .map(|row| {
            let isl = row.isl.as_ref().map_or("-".to_string(), |s| num(s.mean));
            let osl = row.osl.as_ref().map_or("-".to_string(), |s| num(s.mean));
            let growth = row.mean_history_growth.map_or("-".to_string(), num);
            let think = row
                .authored_think_time_ms
                .as_ref()
                .map_or("-".to_string(), |s| num(s.mean));
            vec![
                row.turn_index.to_string(),
                row.conversations_reaching.to_string(),
                isl,
                osl,
                growth,
                think,
            ]
        })
        .collect();
    let justify = std::iter::repeat_n(Justify::Right, headers.len()).collect::<Vec<_>>();
    render_table(
        "NVIDIA AIPerf | Turn-by-Turn",
        headers,
        &rows,
        &justify,
        WIDTH,
    )
}

/// Ideal and realized prefix-cache reuse, including the capacity sweep.
fn cache_tables(a: &DatasetAnalysis) -> Option<String> {
    let cache = a.cache.as_ref()?;
    let ideal = &cache.ideal;
    let ideal_rows = vec![
        vec![
            "identity source".to_string(),
            format!("{:?}", cache.identity_source),
        ],
        vec!["block size".to_string(), cache.block_size.to_string()],
        vec!["total blocks".to_string(), ideal.total_blocks.to_string()],
        vec!["cached blocks".to_string(), ideal.cached_blocks.to_string()],
        vec!["hit rate".to_string(), num(ideal.hit_rate)],
        vec!["unique blocks".to_string(), ideal.unique_blocks.to_string()],
        vec!["unique roots".to_string(), ideal.unique_roots.to_string()],
        vec![
            "intra-conversation cached".to_string(),
            ideal.intra_conversation_cached.to_string(),
        ],
        vec![
            "cross-conversation cached".to_string(),
            ideal.cross_conversation_cached.to_string(),
        ],
    ];
    let ideal_table = render_table(
        "NVIDIA AIPerf | Prefix Cache (Ideal)",
        &["property", "value"],
        &ideal_rows,
        &[Justify::Left, Justify::Right],
        WIDTH,
    );

    let sweep_rows: Vec<Vec<String>> = cache
        .realized
        .iter()
        .map(|p| {
            vec![
                p.capacity_blocks.to_string(),
                num(p.hit_rate),
                sparkbar(p.hit_rate, 1.0, 20),
                p.evictions.to_string(),
            ]
        })
        .collect();
    let sweep_table = render_table(
        "NVIDIA AIPerf | Prefix Cache (Realized Sweep)",
        &["capacity blocks", "hit rate", "curve", "evictions"],
        &sweep_rows,
        &[
            Justify::Right,
            Justify::Right,
            Justify::Left,
            Justify::Right,
        ],
        WIDTH,
    );

    Some(format!("{ideal_table}\n{sweep_table}"))
}

/// Execution timeline: concurrency, throughput, and queue backlog.
fn timeline_table(a: &DatasetAnalysis) -> Option<String> {
    let t = a.timeline.as_ref()?;
    let rows = vec![
        vec![
            "peak concurrency".to_string(),
            t.concurrency.peak.to_string(),
        ],
        vec![
            "time-weighted avg concurrency".to_string(),
            num(t.concurrency.time_weighted_avg),
        ],
        vec![
            "run duration (s)".to_string(),
            num(t.throughput.run_duration_s),
        ],
        vec!["requests / s".to_string(), num(t.throughput.requests_per_s)],
        vec![
            "output tokens / s".to_string(),
            num(t.throughput.output_tokens_per_s),
        ],
    ];
    let summary = render_table(
        "NVIDIA AIPerf | Execution Timeline",
        &["property", "value"],
        &rows,
        &[Justify::Left, Justify::Right],
        WIDTH,
    );

    let sparkline = concurrency_sparkline(&t.concurrency.samples);
    if sparkline.is_empty() {
        Some(summary)
    } else {
        Some(format!("{summary}\nconcurrency: {sparkline}"))
    }
}

/// Fixed-width proportional bar for a value in `[0, max]`.
fn sparkbar(value: f64, max: f64, width: usize) -> String {
    if max <= 0.0 || width == 0 {
        return String::new();
    }
    let filled = ((value / max) * width as f64)
        .round()
        .clamp(0.0, width as f64) as usize;
    let mut bar = String::with_capacity(width);
    for _ in 0..filled {
        bar.push('\u{2588}'); // █
    }
    for _ in filled..width {
        bar.push('\u{2591}'); // ░
    }
    bar
}

/// Compact Unicode sparkline of inflight concurrency over the run.
fn concurrency_sparkline(samples: &[(f64, u64)]) -> String {
    if samples.is_empty() {
        return String::new();
    }
    const RAMP: [char; 8] = [
        '\u{2581}', '\u{2582}', '\u{2583}', '\u{2584}', '\u{2585}', '\u{2586}', '\u{2587}',
        '\u{2588}',
    ];
    let peak = samples.iter().map(|(_, c)| *c).max().unwrap_or(0);
    if peak == 0 {
        return String::new();
    }
    samples
        .iter()
        .map(|(_, c)| {
            let idx = ((*c as f64 / peak as f64) * (RAMP.len() - 1) as f64).round() as usize;
            RAMP[idx.min(RAMP.len() - 1)]
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::analysis::*;

    #[test]
    fn renders_all_sections() {
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
                delay_ms: Some(250.0),
                block_ids: Some(vec![1, 2, 3]),
                system_handle: None,
            },
        ];
        let records = vec![
            AnalyzedRecord {
                conversation_id: "a".into(),
                turn_index: 0,
                start_ns: 0,
                end_ns: 1_000_000_000,
                admit_ns: Some(0),
                first_token_ns: Some(0),
                input_tokens: 32,
                output_tokens: 8,
                token_arrival_ns: vec![],
            },
            AnalyzedRecord {
                conversation_id: "a".into(),
                turn_index: 1,
                start_ns: 1_000_000_000,
                end_ns: 2_000_000_000,
                admit_ns: Some(1_000_000_000),
                first_token_ns: Some(1_000_000_000),
                input_tokens: 48,
                output_tokens: 8,
                token_arrival_ns: vec![],
            },
        ];
        let a = analyze(&turns, &records, &AnalysisOptions::default());
        let out = render_analysis_txt(&a);
        assert!(out.contains("Dataset Shape"));
        assert!(out.contains("Sequence Lengths"));
        assert!(out.contains("Turn-by-Turn"));
        assert!(out.contains("Prefix Cache"));
        assert!(out.contains("Execution Timeline"));
    }
}
