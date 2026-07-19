// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Wall-clock timing harness for the native WEKA trace compiler.
//!
//! Loads the full `semianalysisai/cc-traces-weka-062126` corpus from a local
//! JSONL file (fully offline, no HF metadata fetch) and times
//! `compile_weka_trace_input` end to end with the production knobs
//! (coding corpus, seed 1234, builtin tokenizer, 60 s idle-gap cap).
//!
//! Run:
//! ```bash
//! cargo run -p aiperf-runtime --release --example weka_bench -- \
//!   /home/anthony/.cache/huggingface/hub/datasets--semianalysisai--cc-traces-weka-062126/snapshots/23f152f6f0f9399a85901b89a6458def0ef16729/traces.jsonl
//! ```

use std::path::PathBuf;
use std::time::Instant;

// Match the production `aiperf` binary's global allocator: the recorded-trace
// lowering does many small allocations (per-token decode bytes, message wires),
// which mimalloc handles far better than the system allocator.
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use aiperf_runtime::dataset::{DatasetSource, LoadConfig, TiktokenTokenizer};
use aiperf_runtime::graph::recorded::{
    PromptCorpus, RecordedTraceInputConfig, compile_weka_trace_input,
};

fn main() {
    let path = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .expect("usage: weka_bench <path-to-traces.jsonl>");

    let config = RecordedTraceInputConfig {
        load: LoadConfig::new(DatasetSource::Path(path.clone())),
        root_limit: None,
        max_context_length: None,
        max_osl: None,
        idle_gap_cap_seconds: Some(60.0),
        prompt_corpus: PromptCorpus::Coding,
        content_root_seed: 1234,
    };

    let tokenizer = TiktokenTokenizer::builtin();

    // Current-thread runtime: the compile fans its CPU work out to the rayon
    // global pool, so the async layer only awaits the one-shot document load.
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("build tokio runtime");

    let threads = rayon::current_num_threads();
    let started = Instant::now();
    let bundle = runtime
        .block_on(compile_weka_trace_input(config, &tokenizer))
        .expect("compile weka corpus");
    let elapsed = started.elapsed();

    // WIRE-level content digest: for every node, in trace/node order, fold the
    // actual bytes of each prompt segment it references (resolved through the
    // handle to the stored payload) plus the node's sent parameters. This is
    // invariant to internal handle numbering and segment-store dedup, so it is
    // the right equivalence check across a re-parallelization of the lowering:
    // two runs — or a sequential vs a within-trace-parallel build — that send the
    // byte-identical requests print the same digest.
    use aiperf_runtime::dataset::Payload;
    use aiperf_runtime::graph::model::PromptItem;
    let mut hasher = blake3::Hasher::new();
    for plan in &bundle.plans {
        hasher.update(plan.trace.id.as_bytes());
        for (node_id, node) in &plan.graph.nodes {
            hasher.update(node_id.as_bytes());
            hasher.update(&[node.streaming as u8]);
            hasher.update(&node.max_tokens.unwrap_or(0).to_le_bytes());
            for item in &node.items {
                if let PromptItem::Seg { seg } = item
                    && let Some(segment) = bundle.segments.segment(*seg)
                {
                    match &segment.payload {
                        Payload::Message { role, wire, .. } => {
                            hasher.update(role.as_str().as_bytes());
                            hasher.update(wire);
                        }
                        Payload::Text { bytes, .. } => {
                            hasher.update(bytes);
                        }
                        Payload::Raw { wire } => {
                            hasher.update(wire);
                        }
                        Payload::TokenIds { token_ids } => {
                            for token in token_ids.iter() {
                                hasher.update(&token.to_le_bytes());
                            }
                        }
                        Payload::Media { bytes, .. } => {
                            hasher.update(bytes);
                        }
                        Payload::TraceHashIds { hash_ids, .. } => {
                            for id in hash_ids.iter() {
                                hasher.update(&id.to_le_bytes());
                            }
                        }
                    }
                }
            }
        }
    }
    let digest = hasher.finalize();

    let node_count: usize = bundle.plans.iter().map(|p| p.graph.nodes.len()).sum();
    println!("content_digest={}", digest.to_hex());
    println!(
        "weka full corpus: {plans} traces, {nodes} nodes, {segs} segments in {elapsed:.3?} \
         ({threads} rayon threads)",
        plans = bundle.plans.len(),
        nodes = node_count,
        segs = bundle.segments.len(),
    );
    println!("elapsed_seconds={:.3}", elapsed.as_secs_f64());
}
