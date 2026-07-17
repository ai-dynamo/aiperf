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
//! Turn observations are derived from the retained [`CapturedRecord`] set. For a
//! `--dry-run` the captured records carry the authored per-turn input/output
//! sequence lengths faithfully (input token count and requested output cap), so
//! they are the ground-truth source for the length and per-turn-index sections.
//! The runtime [`GraphInputBundle`] model exposes neither per-turn precomputed
//! block hashes (`hash_ids`) nor the shared system-prompt handle on its
//! [`crate::graph::model::GraphTracePlan`] nodes, so `block_ids` and
//! `system_handle` are left absent; the analysis then synthesizes block
//! identities from length structure (see
//! [`crate::dataset::analysis::CacheReuseAnalysis`]).

use std::path::PathBuf;

use anyhow::{Context, Result};

use crate::dataset::analysis::{AnalysisOptions, AnalyzedRecord, AnalyzedTurn, analyze};
use crate::engine::records::CapturedRecord;
use crate::graph::input::GraphInputBundle;

/// A request to emit the dataset-analysis artifact family.
pub struct DatasetAnalysisRequest {
    /// Base output path; the four artifacts are written beside it using the fixed
    /// `dataset_analysis.*` file names (so `path`'s directory is what matters).
    pub path: PathBuf,
    /// Analysis tuning (block size, explicit LRU capacity point).
    pub options: AnalysisOptions,
    /// Reserved: emit per-conversation breakdowns. Not yet consumed by the pure
    /// analysis; retained so the request shape is stable across wiring tasks.
    pub per_conversation: bool,
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

/// Map the run's turns into neutral [`AnalyzedTurn`] observations.
///
/// Turns are derived from the captured records — one planned turn produces one
/// record on the dry-run graph path — because the compiled [`GraphInputBundle`]
/// does not expose per-turn ISL or block hashes on its trace-plan nodes. The
/// maximum output budget prefers the request-declared cap
/// (`tokens.requested_output`) and falls back to the realized output sequence
/// length. `block_ids` and `system_handle` are left absent (see the module
/// docs); the analysis then falls back to length-structure identity synthesis.
///
/// `input` is accepted for signature stability and future enrichment (e.g.
/// mapping trace-plan `hash_ids` once the runtime model surfaces them); it does
/// not currently contribute per-turn fields.
pub fn analyzed_from_graph(
    input: &GraphInputBundle,
    records: &[CapturedRecord],
) -> Vec<AnalyzedTurn> {
    let _ = input;
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
    let turns = analyzed_from_graph(input, records);
    let analyzed_records = analyzed_from_records(records);
    let analysis = analyze(&turns, &analyzed_records, &req.options);

    if let Some(parent) = req.path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating dataset-analysis directory {}", parent.display()))?;
    }

    let txt_path = req.path.with_file_name("dataset_analysis.txt");
    let text = crate::export::analysis_txt::render_analysis_txt(&analysis);
    std::fs::write(&txt_path, text)
        .with_context(|| format!("writing dataset-analysis text {}", txt_path.display()))?;

    let json_path = req.path.with_file_name("dataset_analysis.json");
    crate::export::dataset_analysis::write_dataset_analysis_json(&analysis, &json_path)
        .with_context(|| format!("writing dataset-analysis JSON {}", json_path.display()))?;

    let csv_path = req.path.with_file_name("dataset_analysis.csv");
    crate::export::dataset_analysis::write_dataset_analysis_csv(&analysis, &csv_path)
        .with_context(|| format!("writing dataset-analysis CSV {}", csv_path.display()))?;

    let html_path = req.path.with_file_name("dataset_analysis.html");
    crate::export::analysis_html::write_dataset_analysis_html(&analysis, &html_path)
        .with_context(|| format!("writing dataset-analysis HTML {}", html_path.display()))?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::{SegmentPool, SegmentStore};
    use crate::engine::records::{CapturedModelOutput, CapturedRecord};
    use crate::graph::input::{GraphInputBundle, GraphInputMetadata};
    use crate::graph::model::{GraphRecord, GraphTracePlan, TraceRecord};
    use crate::metrics_core::{Phase, RecordIngest, TokenCounts};
    use std::sync::Arc;

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
        };
        CapturedRecord {
            uuid: uuid::Uuid::nil(),
            x_correlation_id: format!("{conversation}-{turn}"),
            output: CapturedModelOutput::default(),
            raw: None,
            ingest,
        }
    }

    /// A minimal graph bundle: one trace plan, an empty frozen segment store.
    fn bundle() -> GraphInputBundle {
        let plans = vec![GraphTracePlan {
            graph: GraphRecord::default(),
            trace: TraceRecord {
                id: "conv-a".into(),
                graph_ref: None,
                initial_state: Default::default(),
            },
            arrival_offset_ns: None,
        }];
        GraphInputBundle {
            plans,
            segments: Arc::new(SegmentPool::new().freeze()) as Arc<dyn SegmentStore>,
            metadata: GraphInputMetadata {
                format: "dag_jsonl".into(),
                root_count: 1,
                node_count: 1,
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
            per_conversation: false,
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
}
