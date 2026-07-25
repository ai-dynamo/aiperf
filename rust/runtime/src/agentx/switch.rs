// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Switchable WEKA reconstruction semantics: **legacy** (this byte-exact AgentX
//! port) vs **graph-ir** (the next-gen `graph::recorded` path).
//!
//! A run selects one semantics via a flag; the two are mutually exclusive and
//! share no logic (the AgentX port is transitional and deleted once graph-ir
//! supersedes it). This module is the selection seam: it resolves the flag and,
//! for the **legacy** arm, drives the ported pipeline
//! ([`crate::agentx::loader`]). The **graph-ir** arm is owned by the runtime's
//! `graph::recorded` weka loader (a different output model), so this seam only
//! records the selection and hands off — it never reimplements graph-ir.

use std::collections::HashMap;

use crate::agentx::config::WekaConfig;
use crate::agentx::loader::{convert_traces_parallel, convert_traces_serial, MainReconstructOptions, TraceConversions};
use crate::agentx::synth::TokenSynth;
use crate::agentx::trace::WekaTrace;

/// The two switchable WEKA reconstruction semantics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WekaSemantics {
    /// The byte-exact AgentX legacy port (this module tree).
    Legacy,
    /// The next-gen graph-ir recorded path (`graph::recorded`).
    GraphIr,
}

impl WekaSemantics {
    /// Resolve the semantics from an optional flag string. `"legacy"`/`"agentx"`
    /// select the legacy port; `"graph-ir"`/`"graphir"`/`"graph_ir"` select
    /// graph-ir. Unset defaults to graph-ir (the next-gen path); unknown values
    /// are rejected.
    pub fn resolve(flag: Option<&str>) -> Result<Self, String> {
        match flag.map(|s| s.trim().to_ascii_lowercase()).as_deref() {
            None | Some("") | Some("graph-ir") | Some("graphir") | Some("graph_ir") => {
                Ok(WekaSemantics::GraphIr)
            }
            Some("legacy") | Some("agentx") => Ok(WekaSemantics::Legacy),
            Some(other) => Err(format!(
                "unknown weka semantics {other:?}; expected 'legacy' or 'graph-ir'"
            )),
        }
    }

    /// True when this run uses the AgentX legacy port.
    pub fn is_legacy(self) -> bool {
        matches!(self, WekaSemantics::Legacy)
    }
}

/// Drive WEKA reconstruction under the selected semantics.
///
/// For [`WekaSemantics::Legacy`] this runs the ported pipeline (serial or
/// parallel). For [`WekaSemantics::GraphIr`] it returns `Err` directing the
/// caller to the runtime's `graph::recorded` weka loader — the graph-ir output
/// model is distinct and owned there, so this seam does not fabricate it.
pub fn reconstruct_legacy<S, MK>(
    traces: &[(String, WekaTrace)],
    model_map: &HashMap<String, String>,
    cfg: &WekaConfig,
    opts: &MainReconstructOptions,
    parallel: bool,
    make_synth: MK,
) -> Vec<TraceConversions>
where
    S: TokenSynth + Send,
    MK: Fn(&str, i64) -> S + Sync,
{
    if parallel {
        convert_traces_parallel(traces, model_map, cfg, opts, make_synth)
    } else {
        convert_traces_serial(traces, model_map, cfg, opts, make_synth)
    }
}

/// Run the complete **legacy** WEKA replay pipeline for one trace and emit its
/// export-level raw records: reconstruct (root + subagent children + flat chains)
/// → annotate with the agentic-replay dispatch schedule (when `t_star_ms` is
/// given) → serialize to the `export.records` shape. This is the end-to-end
/// legacy path from trace to export artifact, composed from byte-exact units.
pub fn run_legacy_pipeline<S>(
    trace_id: &str,
    trace: &WekaTrace,
    synth: &mut S,
    model_map: &HashMap<String, String>,
    cfg: &WekaConfig,
    opts: &MainReconstructOptions,
    t_star_ms: Option<f64>,
) -> Result<Vec<serde_json::Value>, crate::agentx::synth::PrefixTooTruncated>
where
    S: TokenSynth,
{
    let convs = crate::agentx::loader::convert_trace_to_conversations(
        trace_id, trace, synth, model_map, cfg, opts,
    )?;
    Ok(crate::agentx::export::raw_export_trace(&convs, t_star_ms))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agentx::synth::TokenSynth;
    use crate::agentx::trace::{HashIdScope, WekaNormalRequest, WekaRequest};

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
    fn resolves_flags() {
        assert_eq!(WekaSemantics::resolve(None).unwrap(), WekaSemantics::GraphIr);
        assert_eq!(WekaSemantics::resolve(Some("graph-ir")).unwrap(), WekaSemantics::GraphIr);
        assert_eq!(WekaSemantics::resolve(Some("Legacy")).unwrap(), WekaSemantics::Legacy);
        assert_eq!(WekaSemantics::resolve(Some("agentx")).unwrap(), WekaSemantics::Legacy);
        assert!(WekaSemantics::resolve(Some("nope")).is_err());
    }

    #[test]
    fn legacy_dispatch_reconstructs() {
        let trace = WekaTrace {
            id: "t".into(),
            models: vec!["m".into()],
            block_size: 4,
            hash_id_scope: HashIdScope::Local,
            tool_tokens: 0,
            system_tokens: 0,
            requests: vec![WekaRequest::Normal(WekaNormalRequest {
                t: 0.0,
                model: "m".into(),
                input_length: 8,
                output_length: 4,
                hash_ids: vec![1, 2],
                input_types: vec![],
                output_types: vec![],
                stop: String::new(),
                api_time: Some(0.1),
                think_time: None,
            })],
            totals: None,
        };
        let sem = WekaSemantics::resolve(Some("legacy")).unwrap();
        assert!(sem.is_legacy());
        let out = reconstruct_legacy(
            &[("t".into(), trace)],
            &HashMap::new(),
            &WekaConfig::default(),
            &MainReconstructOptions::default(),
            false,
            |_tid, bs| StubSynth { bs },
        );
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].as_ref().unwrap()[0].turns.len(), 1);
    }

    #[test]
    fn full_legacy_pipeline_emits_scheduled_export_records() {
        // Two timestamped turns; the pipeline reconstructs, schedules at t*, and
        // serializes to export records carrying timing + content + phase.
        let trace = WekaTrace {
            id: "t".into(),
            models: vec!["m".into()],
            block_size: 4,
            hash_id_scope: HashIdScope::Local,
            tool_tokens: 0,
            system_tokens: 0,
            requests: vec![
                WekaRequest::Normal(WekaNormalRequest {
                    t: 0.0,
                    model: "m".into(),
                    input_length: 8,
                    output_length: 4,
                    hash_ids: vec![1, 2],
                    input_types: vec![],
                    output_types: vec![],
                    stop: String::new(),
                    api_time: Some(0.1),
                    think_time: None,
                }),
                WekaRequest::Normal(WekaNormalRequest {
                    t: 1.0,
                    model: "m".into(),
                    input_length: 12,
                    output_length: 4,
                    hash_ids: vec![1, 2, 3],
                    input_types: vec![],
                    output_types: vec![],
                    stop: String::new(),
                    api_time: Some(0.1),
                    think_time: None,
                }),
            ],
            totals: None,
        };
        let mut synth = StubSynth { bs: 4 };
        // t* = 500ms -> turn0 (t=0) warmup, turn1 (t=1000) profiling @ offset 500.
        let records = run_legacy_pipeline(
            "t",
            &trace,
            &mut synth,
            &HashMap::new(),
            &WekaConfig { split_flattened_agents: false, ..WekaConfig::default() },
            &MainReconstructOptions::default(),
            Some(500.0),
        )
        .unwrap();
        assert_eq!(records.len(), 2);
        assert_eq!(records[0]["phase"], serde_json::json!("warmup"));
        assert_eq!(records[1]["phase"], serde_json::json!("profiling"));
        assert_eq!(records[1]["dispatch_offset_ms"], serde_json::json!(500.0));
        // Content + timing present on the export records.
        assert_eq!(records[1]["delay_ms"], serde_json::json!(900.0));
        assert!(records[0]["raw_messages"].as_array().unwrap().len() >= 1);
    }
}
