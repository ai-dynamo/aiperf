// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Executor adapter that projects real run data into the neutral dataset-analysis
//! inputs and writes the `--dry-run` dataset-analysis artifacts.
//!
//! The pure-logic analysis ([`crate::dataset::analysis::analyze`]) and the four
//! sink writers ([`crate::export::analysis_txt`],
//! [`crate::export::dataset_analysis`], [`crate::export::analysis_html`]) are
//! transport-neutral: they consume [`AnalyzedTurn`]/[`AnalyzedRecord`] and know
//! nothing about the engine. This module owns the mapping from captured metric
//! records and the compiled graph input bundle into those neutral inputs, then
//! emits `dataset_analysis.{txt,json,csv,html}` beside the requested base path.
//!
//! On the graph path (recorded `weka_trace`/`dynamo_trace` or authored
//! `dag_jsonl`) the per-request structure is present *statically* in the compiled
//! [`GraphInputBundle`], so a `--dry-run` that dispatches zero records still
//! yields a fully populated dataset characterization: each
//! [`crate::graph::model::LlmNode`] carries its input token count
//! (`metadata["input_tokens"]`), generation cap (`LlmNode::max_tokens`), stable
//! conversation id and turn index, and — most importantly — its ordered,
//! prefix-dependent, content-addressed prompt-segment handles
//! ([`crate::graph::model::PromptItem::Seg`], mirrored in
//! `metadata["prompt_segment_handles"]`). Those BLAKE3 prefix-chained handles are
//! exactly the chained block identifiers the reuse analysis wants: an identical
//! leading handle means the whole prefix matched, so cross- and intra-turn KV
//! reuse is exact and the identity source is
//! [`crate::dataset::analysis::prefix_cache::IdentitySource::HashIds`]. The raw
//! per-block WEKA `hash_ids` (`i128`) are *not* preserved through lowering, but
//! the content-addressed segment handles stand in with equivalent prefix-exact
//! semantics at message granularity.
//!
//! The scheduled path has no [`GraphInputBundle`]; there turn observations are
//! derived from the retained [`CapturedRecord`] set (block ids absent → the
//! analysis synthesizes identities from length structure).

use std::path::PathBuf;

use anyhow::{Context, Result};

use crate::dataset::analysis::{AnalysisOptions, AnalyzedRecord, AnalyzedTurn, analyze};
use crate::engine::records::CapturedRecord;
use crate::graph::input::GraphInputBundle;
use crate::graph::model::{LlmNode, PromptItem};

/// A request to emit the dataset-analysis artifact family.
pub struct DatasetAnalysisRequest {
    /// Base output path; the four artifacts are written beside it using the fixed
    /// `dataset_analysis.*` file names (so `path`'s directory is what matters).
    pub path: PathBuf,
    /// Analysis tuning (block size, explicit LRU capacity point, and the
    /// per-conversation-breakdown toggle honored by
    /// [`crate::dataset::analysis::analyze`]).
    pub options: AnalysisOptions,
}

/// Stable conversation identity for a captured record.
///
/// Prefers the endpoint-provided `conversation_id`; falls back to the workload
/// session number so single-turn runs without an explicit conversation id still
/// group into distinct conversations rather than collapsing to one.
fn conversation_key(record: &CapturedRecord) -> String {
    record
        .ingest
        .conversation_id
        .clone()
        .unwrap_or_else(|| format!("session-{}", record.ingest.session_num))
}

/// Map captured metric records into neutral [`AnalyzedRecord`] observations.
///
/// Input-token count is the locally measured ISL; output-token count is the
/// output sequence length (output plus reasoning), preserving absent-as-zero.
pub fn analyzed_from_records(records: &[CapturedRecord]) -> Vec<AnalyzedRecord> {
    records
        .iter()
        .map(|record| {
            let ingest = &record.ingest;
            AnalyzedRecord {
                conversation_id: conversation_key(record),
                turn_index: ingest.turn_index as usize,
                start_ns: ingest.start_ns,
                end_ns: ingest.end_ns,
                admit_ns: ingest.admit_ns,
                first_token_ns: ingest.first_token_ns,
                input_tokens: ingest.tokens.input.unwrap_or(0),
                output_tokens: ingest.tokens.output_sequence_length().unwrap_or(0),
                token_arrival_ns: ingest.token_arrival_ns.clone(),
            }
        })
        .collect()
}

/// Collect a graph node's ordered prompt-segment handles as chained block ids.
///
/// The prompt-assembly program interleaves static content-addressed segment
/// handles ([`PromptItem::Seg`]/[`PromptItem::RawMessages`]/[`PromptItem::Text`])
/// with dynamic reply splices ([`PromptItem::Splice`]). Only the static handles
/// carry stable content identity; splices resolve from live channel state and are
/// skipped. Each retained handle is BLAKE3 prefix-dependent (interned against its
/// predecessor), so a shared leading handle across two prompts means the whole
/// prefix up to it is byte-identical — the exact chained-hash reuse test the
/// analysis performs. Handle indices are globally unique within the merged
/// segment store, so ids never collide across conversations.
fn node_block_ids(node: &LlmNode) -> Vec<i64> {
    node.items
        .iter()
        .filter_map(|item| match item {
            PromptItem::Seg { seg } => Some(i64::from(seg.index())),
            PromptItem::RawMessages { raw_messages } => Some(i64::from(raw_messages.index())),
            PromptItem::Text { text, .. } => Some(i64::from(text.index())),
            PromptItem::Splice { .. } => None,
        })
        .collect()
}

/// Map the compiled graph input's per-request structure into neutral
/// [`AnalyzedTurn`] observations — the ground truth on the graph `--dry-run`
/// path, where zero records are dispatched.
///
/// Each [`LlmNode`] across every trace plan becomes one turn: the conversation id
/// and turn index come from node metadata (falling back to the trace id / node
/// order), the input token count from `metadata["input_tokens"]`, the maximum
/// output budget from the node's generation cap ([`LlmNode::max_tokens`], falling
/// back to the recorded output token count), and the chained `block_ids` from the
/// node's ordered content-addressed prompt-segment handles (see
/// [`node_block_ids`]). Because those handles are real content hashes, the
/// analysis reports [`IdentitySource::HashIds`] with exact prefix reuse rather
/// than length-structure synthesis.
///
/// Turns are emitted in arrival order (`metadata["arrival_offset_us"]` when
/// present) so the ideal-reuse pass — which seeds its cache from earlier requests
/// — observes prefixes before the turns that reuse them. The pure analysis's
/// later stable re-sort by record `start_ns` preserves this order when records
/// are empty.
///
/// [`IdentitySource::HashIds`]: crate::dataset::analysis::prefix_cache::IdentitySource::HashIds
pub fn analyzed_turns_from_graph_input(input: &GraphInputBundle) -> Vec<AnalyzedTurn> {
    let mut ordered: Vec<(u64, AnalyzedTurn)> = Vec::new();
    for program in &input.programs {
        let plan = &program.profiling;
        for (order, node) in plan.graph.nodes.values().enumerate() {
            let crate::graph::model::ExecutableGraphNode::Llm(node) = node else {
                continue;
            };
            let metadata = &node.metadata;
            let conversation_id = metadata
                .get("conversation_id")
                .and_then(serde_json::Value::as_str)
                .map(str::to_string)
                .unwrap_or_else(|| plan.trace.id.clone());
            let turn_index = metadata
                .get("turn_index")
                .and_then(serde_json::Value::as_u64)
                .map_or(order, |value| value as usize);
            let input_tokens = metadata
                .get("input_tokens")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0);
            let max_output_tokens = node
                .max_tokens
                .map(|tokens| tokens as u64)
                .or_else(|| {
                    metadata
                        .get("recorded_output_tokens")
                        .and_then(serde_json::Value::as_u64)
                })
                .unwrap_or(0);
            let block_ids = node_block_ids(node);
            let block_ids = (!block_ids.is_empty()).then_some(block_ids);
            let arrival = metadata
                .get("arrival_offset_us")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(order as u64);
            ordered.push((
                arrival,
                AnalyzedTurn {
                    conversation_id,
                    turn_index,
                    input_tokens,
                    max_output_tokens,
                    delay_ms: None,
                    block_ids,
                    system_handle: None,
                },
            ));
        }
    }
    // Stable sort by arrival so shared prefixes are introduced before the turns
    // that reuse them; ties keep trace/node emission order.
    ordered.sort_by_key(|(arrival, _)| *arrival);
    ordered.into_iter().map(|(_, turn)| turn).collect()
}

/// Derive neutral [`AnalyzedTurn`] observations from the captured record set.
///
/// One planned turn produces one record on the dry-run paths, and the captured
/// records carry the authoritative per-turn ISL (`tokens.input`) and requested
/// output cap. The maximum output budget prefers the request-declared cap
/// (`tokens.requested_output`) and falls back to the realized output sequence
/// length. `block_ids` and `system_handle` are left absent (see the module docs);
/// the analysis then falls back to length-structure identity synthesis.
///
/// This is the shared source for both the graph path
/// ([`write_dataset_analysis`]) and the scheduled path
/// ([`write_dataset_analysis_from_records`]) — neither the compiled
/// [`GraphInputBundle`] nor the scheduled conversation source exposes per-turn
/// block hashes, so the records are the ground truth in both cases.
pub fn analyzed_turns_from_records(records: &[CapturedRecord]) -> Vec<AnalyzedTurn> {
    records
        .iter()
        .map(|record| {
            let ingest = &record.ingest;
            let max_output_tokens = ingest
                .tokens
                .requested_output
                .or_else(|| ingest.tokens.output_sequence_length())
                .unwrap_or(0);
            AnalyzedTurn {
                conversation_id: conversation_key(record),
                turn_index: ingest.turn_index as usize,
                input_tokens: ingest.tokens.input.unwrap_or(0),
                max_output_tokens,
                delay_ms: None,
                block_ids: None,
                system_handle: None,
            }
        })
        .collect()
}

/// Analyze the run and write `dataset_analysis.{txt,json,csv,html}` beside
/// [`DatasetAnalysisRequest::path`].
///
/// The parent directory of `req.path` is created if needed. Each artifact is a
/// distinct rendering of the same [`crate::dataset::analysis::DatasetAnalysis`]:
/// a console-table text report, pretty JSON, a stat-key CSV, and a self-contained
/// HTML single-page report.
pub fn write_dataset_analysis(
    req: &DatasetAnalysisRequest,
    records: &[CapturedRecord],
    input: &GraphInputBundle,
) -> Result<()> {
    let turns = analyzed_turns_from_graph_input(input);
    let analyzed_records = analyzed_from_records(records);
    let analysis = analyze(&turns, &analyzed_records, &req.options);
    write_analysis_artifacts(req, &analysis)
}

/// Analyze the run from the captured records alone and write
/// `dataset_analysis.{txt,json,csv,html}` beside [`DatasetAnalysisRequest::path`].
///
/// This is the scheduled-path entry point: the scheduled executor has no
/// [`GraphInputBundle`], so both the per-turn observations and the per-record
/// observations are derived from the retained [`CapturedRecord`] set (the ground
/// truth for a `--dry-run`; the requesting artifact forces the retain path so the
/// records are the full clean + errored set). The output is identical in shape to
/// [`write_dataset_analysis`].
pub fn write_dataset_analysis_from_records(
    req: &DatasetAnalysisRequest,
    records: &[CapturedRecord],
) -> Result<()> {
    let turns = analyzed_turns_from_records(records);
    let analyzed_records = analyzed_from_records(records);
    let analysis = analyze(&turns, &analyzed_records, &req.options);
    write_analysis_artifacts(req, &analysis)
}

/// Render the four dataset-analysis artifacts beside [`DatasetAnalysisRequest::path`].
///
/// The parent directory of `req.path` is created if needed. Each artifact is a
/// distinct rendering of the same [`crate::dataset::analysis::DatasetAnalysis`]:
/// a console-table text report, pretty JSON, a stat-key CSV, and a self-contained
/// HTML single-page report.
fn write_analysis_artifacts(
    req: &DatasetAnalysisRequest,
    analysis: &crate::dataset::analysis::DatasetAnalysis,
) -> Result<()> {
    if let Some(parent) = req.path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating dataset-analysis directory {}", parent.display()))?;
    }

    let txt_path = req.path.with_file_name("dataset_analysis.txt");
    let text = crate::export::analysis_txt::render_analysis_txt(analysis);
    std::fs::write(&txt_path, text)
        .with_context(|| format!("writing dataset-analysis text {}", txt_path.display()))?;

    let json_path = req.path.with_file_name("dataset_analysis.json");
    crate::export::dataset_analysis::write_dataset_analysis_json(analysis, &json_path)
        .with_context(|| format!("writing dataset-analysis JSON {}", json_path.display()))?;

    let csv_path = req.path.with_file_name("dataset_analysis.csv");
    crate::export::dataset_analysis::write_dataset_analysis_csv(analysis, &csv_path)
        .with_context(|| format!("writing dataset-analysis CSV {}", csv_path.display()))?;

    let html_path = req.path.with_file_name("dataset_analysis.html");
    crate::export::analysis_html::write_dataset_analysis_html(analysis, &html_path)
        .with_context(|| format!("writing dataset-analysis HTML {}", html_path.display()))?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::{Handle, SegmentPool, SegmentStore};
    use crate::engine::records::{CapturedModelOutput, CapturedRecord};
    use crate::graph::input::{GraphInputBundle, GraphInputMetadata};
    use crate::graph::model::{
        ExecutableGraphNode, GraphRecord, GraphTracePlan, GraphTraceProgram, LlmNode, PromptItem,
        TraceRecord,
    };
    use crate::metrics_core::{Phase, RecordIngest, TokenCounts};
    use std::collections::BTreeMap;
    use std::sync::Arc;

    /// Build an `LlmNode` carrying the metadata and content-addressed prompt
    /// handles the graph-input analysis reads: conversation id, turn index, input
    /// token count, generation cap, and a chained handle run.
    fn llm_node(conversation: &str, turn: usize, isl: u64, osl: usize, handles: &[u32]) -> LlmNode {
        let mut metadata = BTreeMap::new();
        metadata.insert(
            "conversation_id".into(),
            serde_json::Value::String(conversation.to_string()),
        );
        metadata.insert("turn_index".into(), serde_json::Value::from(turn));
        metadata.insert("input_tokens".into(), serde_json::Value::from(isl));
        metadata.insert(
            "arrival_offset_us".into(),
            serde_json::Value::from(turn as u64),
        );
        LlmNode {
            output: format!("{conversation}:{turn}_out"),
            streaming: true,
            inputs: Vec::new(),
            min_start_delay_us: None,
            max_tokens: Some(osl),
            items: handles
                .iter()
                .map(|index| PromptItem::Seg {
                    seg: Handle::new(*index),
                })
                .collect(),
            request: None,
            metadata,
        }
    }

    /// Build a captured record with the fields the analysis consumes populated.
    fn captured(
        conversation: &str,
        turn: u32,
        start: i64,
        end: i64,
        isl: u64,
        osl: u64,
    ) -> CapturedRecord {
        let mut ingest = RecordIngest::minimal(start, end, Phase::Profiling);
        ingest.conversation_id = Some(conversation.to_string());
        ingest.turn_index = turn;
        ingest.admit_ns = Some(start);
        ingest.first_token_ns = Some(start + 1_000_000);
        ingest.token_arrival_ns = vec![start + 1_000_000, start + 2_000_000];
        ingest.tokens = TokenCounts {
            input: Some(isl),
            output: Some(osl),
            reasoning: None,
            requested_output: Some(osl),
            first_content_chunk_tokens: None,
        };
        CapturedRecord {
            uuid: uuid::Uuid::nil(),
            x_correlation_id: format!("{conversation}-{turn}"),
            output: CapturedModelOutput::default(),
            raw: None,
            ingest,
        }
    }

    /// A minimal graph bundle: one trace plan with two chained turns whose leading
    /// content-addressed handle is shared, so the graph-input analysis reports
    /// real hash-id prefix reuse.
    fn bundle() -> GraphInputBundle {
        let mut nodes = BTreeMap::new();
        // Turn 1 reuses turn 0's leading handle (1) — a byte-identical prefix.
        nodes.insert(
            "n0".to_string(),
            ExecutableGraphNode::Llm(llm_node("conv-a", 0, 64, 16, &[1, 2])),
        );
        nodes.insert(
            "n1".to_string(),
            ExecutableGraphNode::Llm(llm_node("conv-a", 1, 96, 16, &[1, 2, 3])),
        );
        let graph = GraphRecord {
            nodes,
            ..GraphRecord::default()
        };
        let programs = vec![GraphTraceProgram::static_graph(GraphTracePlan {
            graph,
            trace: TraceRecord {
                id: "conv-a".into(),
                graph_ref: None,
                initial_state: Default::default(),
            },
            arrival_offset_ns: None,
        })];
        GraphInputBundle {
            programs,
            segments: Arc::new(SegmentPool::new().freeze()) as Arc<dyn SegmentStore>,
            metadata: GraphInputMetadata {
                format: "dag_jsonl".into(),
                root_count: 1,
                node_count: 2,
                warning_facts: Vec::new(),
            },
        }
    }

    #[test]
    fn writes_all_four_artifacts_with_cache_section() {
        let records = vec![
            captured("conv-a", 0, 0, 1_000_000_000, 64, 16),
            captured("conv-a", 1, 1_000_000_000, 2_000_000_000, 96, 16),
        ];
        let input = bundle();

        let dir = tempfile::tempdir().expect("tempdir");
        let req = DatasetAnalysisRequest {
            path: dir.path().join("dataset_analysis"),
            options: AnalysisOptions::default(),
        };

        write_dataset_analysis(&req, &records, &input).expect("write dataset analysis");

        for name in [
            "dataset_analysis.txt",
            "dataset_analysis.json",
            "dataset_analysis.csv",
            "dataset_analysis.html",
        ] {
            let path = dir.path().join(name);
            assert!(path.exists(), "expected {name} to be written");
        }

        // Length-structure identity synthesis (from positive ISL) yields a cache
        // section, so the JSON carries the reuse hit-rate curve.
        let json =
            std::fs::read_to_string(dir.path().join("dataset_analysis.json")).expect("read json");
        assert!(
            json.contains("hit_rate"),
            "cache reuse hit_rate must be present"
        );
        assert!(!json.contains("NaN"), "no non-finite tokens in JSON");
    }

    #[test]
    fn writes_four_artifacts_from_records_only() {
        // Scheduled path: no GraphInputBundle — both turns and records come from the
        // captured record set alone.
        let records = vec![
            captured("conv-a", 0, 0, 1_000_000_000, 64, 16),
            captured("conv-a", 1, 1_000_000_000, 2_000_000_000, 96, 16),
            captured("conv-b", 0, 0, 1_000_000_000, 64, 16),
        ];

        let dir = tempfile::tempdir().expect("tempdir");
        let req = DatasetAnalysisRequest {
            path: dir.path().join("dataset_analysis"),
            options: AnalysisOptions::default(),
        };

        write_dataset_analysis_from_records(&req, &records)
            .expect("write dataset analysis from records");

        for name in [
            "dataset_analysis.txt",
            "dataset_analysis.json",
            "dataset_analysis.csv",
            "dataset_analysis.html",
        ] {
            let path = dir.path().join(name);
            assert!(path.exists(), "expected {name} to be written");
        }

        let json =
            std::fs::read_to_string(dir.path().join("dataset_analysis.json")).expect("read json");
        assert!(
            json.contains("hit_rate"),
            "cache reuse hit_rate must be present"
        );
        assert!(!json.contains("NaN"), "no non-finite tokens in JSON");
    }

    #[test]
    fn graph_input_turns_carry_isl_cap_and_chained_hash_block_ids() {
        let input = bundle();
        let turns = analyzed_turns_from_graph_input(&input);
        assert_eq!(turns.len(), 2);
        // Emitted in arrival order (turn 0 before turn 1).
        assert_eq!(turns[0].turn_index, 0);
        assert_eq!(turns[0].input_tokens, 64);
        assert_eq!(turns[0].max_output_tokens, 16);
        assert_eq!(turns[0].block_ids, Some(vec![1, 2]));
        assert_eq!(turns[1].block_ids, Some(vec![1, 2, 3]));

        // Real content handles drive the HashIds identity path with exact reuse.
        let analysis = analyze(&turns, &[], &AnalysisOptions::default());
        let cache = analysis
            .cache
            .expect("cache section from hash-id block ids");
        assert_eq!(
            cache.identity_source,
            crate::dataset::analysis::prefix_cache::IdentitySource::HashIds
        );
        // Turn 1 reuses turn 0's two leading blocks (ids 1 and 2).
        assert_eq!(cache.ideal.cached_blocks, 2);
    }

    #[test]
    fn record_mapping_preserves_tokens_and_timing() {
        let records = vec![captured("c", 3, 10, 20, 42, 7)];
        let mapped = analyzed_from_records(&records);
        assert_eq!(mapped.len(), 1);
        assert_eq!(mapped[0].conversation_id, "c");
        assert_eq!(mapped[0].turn_index, 3);
        assert_eq!(mapped[0].input_tokens, 42);
        assert_eq!(mapped[0].output_tokens, 7);
        assert_eq!(mapped[0].admit_ns, Some(10));
    }

    #[test]
    fn per_conversation_flag_controls_breakdown_emission() {
        // Two conversations so a per-conversation breakdown is non-trivial.
        let records = vec![
            captured("conv-a", 0, 0, 1_000_000_000, 64, 16),
            captured("conv-a", 1, 1_000_000_000, 2_000_000_000, 96, 16),
            captured("conv-b", 0, 0, 1_000_000_000, 32, 8),
        ];

        // Flag off (default): no per-conversation section.
        let off = crate::dataset::analysis::analyze(
            &analyzed_turns_from_records(&records),
            &analyzed_from_records(&records),
            &AnalysisOptions::default(),
        );
        assert!(
            off.conversations.is_none(),
            "breakdown must be absent unless requested"
        );

        // Flag on: one summary per distinct conversation, each carrying its own
        // length distribution and turn count.
        let on = crate::dataset::analysis::analyze(
            &analyzed_turns_from_records(&records),
            &analyzed_from_records(&records),
            &AnalysisOptions {
                per_conversation: true,
                ..AnalysisOptions::default()
            },
        );
        let conversations = on.conversations.expect("breakdown requested");
        assert_eq!(conversations.len(), 2, "one summary per conversation");
        assert_eq!(conversations[0].conversation_id, "conv-a");
        assert_eq!(conversations[0].turns, 2);
        assert_eq!(conversations[1].conversation_id, "conv-b");
        assert_eq!(conversations[1].turns, 1);
        // conv-b's ISL sum (32) is scoped to conv-b alone, not the dataset total.
        let isl = conversations[1].lengths.isl.as_ref().expect("isl summary");
        assert_eq!(isl.sum, 32.0);
    }
}
