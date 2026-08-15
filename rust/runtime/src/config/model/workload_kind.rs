// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Workload-kind classification derived from the typed benchmark config.
//!
//! The protocol-v2 projection selects a workload implementation (`scheduled` vs
//! `graph`) from the run's dataset. Historically that decision was a hand-rolled
//! string match duplicated in `engine::protocol_v2`; this module makes it a
//! single typed primitive computed from [`BenchmarkConfig`], so the graph-format
//! set has exactly one source of truth and cannot drift between the typed model
//! and the wire projection.

use super::config::BenchmarkConfig;
use super::dataset::Dataset;

/// Dataset formats that select the direct Graph-IR workload.
///
/// Any dataset whose native format token is one of these routes the run through
/// the `graph` workload; every other dataset uses the `scheduled` workload.
pub const GRAPH_FORMATS: [&str; 5] = [
    "dag_jsonl",
    "conditional_graph",
    "weka_trace",
    "dynamo_trace",
    "agent_recording",
];

/// The workload implementation a run projects onto.
///
/// `StaticAccuracy` is intentionally not represented: today's projection selects
/// a static-accuracy run through the dataset *plan*
/// ([`NativeDatasetPlan::StaticAccuracy`](crate::engine)), not a distinct
/// workload id — the emitted workload id is still `scheduled`.
// TODO(step-1): accuracy path — add a `StaticAccuracy` variant only once the
// projection emits a distinct static-accuracy workload id rather than reusing
// `scheduled`. Encoding it now would invent behavior the wire does not carry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WorkloadKind {
    /// Time/rate/concurrency scheduled execution (the `scheduled` workload id).
    Scheduled,
    /// Direct Graph-IR whole-trace execution (the `graph` workload id).
    Graph,
}

impl WorkloadKind {
    /// The built-in workload identifier this kind projects to.
    pub fn workload_id(self) -> &'static str {
        match self {
            WorkloadKind::Scheduled => "scheduled",
            WorkloadKind::Graph => "graph",
        }
    }
}

/// Return whether a dataset format/type token selects the graph workload.
///
/// The token is a dataset's native format (`dag_jsonl`, `weka_trace`, …) or, for
/// the loose wire shortcut, its `type` discriminant. Shared with the protocol-v2
/// wire projection so both consult one graph-format list.
pub fn is_graph_format(token: Option<&str>) -> bool {
    token.is_some_and(|token| GRAPH_FORMATS.contains(&token))
}

/// The native format token of a typed dataset, if it carries one.
///
/// Synthetic datasets have no format token (they are never graph datasets);
/// file and public datasets carry their loader format.
fn dataset_format_token(dataset: &Dataset) -> Option<&str> {
    match dataset {
        Dataset::Synthetic(_) => None,
        Dataset::File(file) => file.format.as_deref(),
        Dataset::Public(public) => Some(public.format.as_str()),
    }
}

/// Classify a benchmark config's workload from its datasets.
///
/// A run whose (single) dataset is a graph format yields [`WorkloadKind::Graph`];
/// every other configuration — including an absent dataset list — yields
/// [`WorkloadKind::Scheduled`].
pub fn workload_kind(cfg: &BenchmarkConfig) -> WorkloadKind {
    let is_graph = cfg
        .datasets
        .as_deref()
        .unwrap_or_default()
        .iter()
        .any(|dataset| is_graph_format(dataset_format_token(dataset)));
    if is_graph {
        WorkloadKind::Graph
    } else {
        WorkloadKind::Scheduled
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::model::dataset::{Distribution, FileDataset, Prompts, Sampling, Synthetic};

    fn synthetic_dataset() -> Dataset {
        Dataset::Synthetic(Synthetic {
            prompts: Prompts {
                batch_size: 1,
                isl: Distribution {
                    mean: Some(128.0),
                    ..Default::default()
                },
                osl: None,
                num_prefix_prompts: None,
                prefix_prompt_length: None,
                block_size: None,
                corpus: None,
                sequence_distribution: None,
                prefix_reuse_fraction: None,
                prefix_reuse_ratio: None,
                cache_bust: None,
            },
            prefix_prompts: None,
            images: None,
            audio: None,
            video: None,
            rankings: None,
            sampling: Sampling("sequential".into()),
            turns: None,
            turn_delay_ratio: 1.0,
            entries: Some(1),
            random_seed: None,
            num_conversations: None,
            turn_delay_ms: None,
        })
    }

    fn file_dataset(format: &str) -> Dataset {
        Dataset::File(FileDataset {
            format: Some(format.to_string()),
            sampling: Sampling("sequential".into()),
            options: serde_json::Map::new(),
            path: Some("/tmp/trace.jsonl".into()),
            entries: None,
            random_seed: None,
            osl: None,
            prompts: None,
            records: None,
            synthesis: None,
            graph: None,
            cache_bust: None,
            prefetch_media_urls: false,
        })
    }

    fn cfg_with(dataset: Dataset) -> BenchmarkConfig {
        BenchmarkConfig {
            datasets: Some(vec![dataset]),
            ..BenchmarkConfig::default()
        }
    }

    #[test]
    fn synthetic_dataset_is_scheduled() {
        assert_eq!(
            workload_kind(&cfg_with(synthetic_dataset())),
            WorkloadKind::Scheduled
        );
    }

    #[test]
    fn dag_jsonl_dataset_is_graph() {
        assert_eq!(
            workload_kind(&cfg_with(file_dataset("dag_jsonl"))),
            WorkloadKind::Graph
        );
    }

    #[test]
    fn weka_trace_dataset_is_graph() {
        assert_eq!(
            workload_kind(&cfg_with(file_dataset("weka_trace"))),
            WorkloadKind::Graph
        );
    }

    #[test]
    fn non_graph_file_format_is_scheduled() {
        assert_eq!(
            workload_kind(&cfg_with(file_dataset("single_turn"))),
            WorkloadKind::Scheduled
        );
    }

    #[test]
    fn absent_datasets_default_to_scheduled() {
        assert_eq!(
            workload_kind(&BenchmarkConfig::default()),
            WorkloadKind::Scheduled
        );
    }
}
