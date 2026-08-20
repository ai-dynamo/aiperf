// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Lower-once coverage for every built-in graph-inspection adapter.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

use aiperf_runtime::dataset::{TextTokenizer, TiktokenTokenizer};
use aiperf_runtime::engine::graph_input::{
    BuiltinRunnerGraphInputAdapterResolver, GraphInputAdapterResolver, GraphInputContext,
    PreparedRunnerGraphInput, prepare_local_graph_inspection_input,
};
use anyhow::Result;
use async_trait::async_trait;
use serde_json::value::RawValue;

struct GraphInspectionSource {
    format: &'static str,
    relative_path: &'static str,
}

const SOURCES: [GraphInspectionSource; 7] = [
    GraphInspectionSource {
        format: "dag_jsonl",
        relative_path: "../../tests/fixtures/dag/small.dag.jsonl",
    },
    GraphInspectionSource {
        format: "conditional_graph",
        relative_path: "../e2e-tests/tests/fixtures/conditional/conditional_shopping.yaml",
    },
    GraphInspectionSource {
        format: "weka_trace",
        relative_path: "../../tests/fixtures/weka_traces/simple.json",
    },
    GraphInspectionSource {
        format: "dynamo_trace",
        relative_path: "tests/fixtures/graph_inspection/dynamo-trace.jsonl",
    },
    GraphInspectionSource {
        format: "aiperf_trace",
        relative_path: "tests/fixtures/graph_inspection/aiperf-trace.json",
    },
    GraphInspectionSource {
        format: "agent_recording",
        relative_path: "tests/fixtures/recorded_agent_replay/recordings",
    },
    GraphInspectionSource {
        format: "otlp_genai",
        relative_path: "../cli/tests/fixtures/graph_tools/collapsed-replay.otlp.json",
    },
];

#[derive(Debug)]
struct CountingResolver {
    stock: BuiltinRunnerGraphInputAdapterResolver,
    load_calls: AtomicUsize,
}

impl CountingResolver {
    fn new() -> Self {
        Self {
            stock: BuiltinRunnerGraphInputAdapterResolver::new(),
            load_calls: AtomicUsize::new(0),
        }
    }

    fn load_calls(&self) -> usize {
        self.load_calls.load(Ordering::SeqCst)
    }
}

#[async_trait(?Send)]
impl GraphInputAdapterResolver for CountingResolver {
    fn validate_identity(&self, raw: &RawValue) -> Result<()> {
        self.stock.validate_identity(raw)
    }

    async fn load(
        &self,
        raw: &RawValue,
        context: &GraphInputContext<'_>,
    ) -> Result<PreparedRunnerGraphInput> {
        self.load_calls.fetch_add(1, Ordering::SeqCst);
        self.stock.load(raw, context).await
    }
}

fn fixture(relative_path: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join(relative_path)
        .canonicalize()
        .expect("canonical inspection fixture")
}

#[tokio::test]
async fn built_in_formats_lower_their_real_inspection_sources_once() {
    assert_eq!(
        SOURCES
            .iter()
            .map(|source| source.format)
            .collect::<Vec<_>>(),
        BuiltinRunnerGraphInputAdapterResolver::new().supported_formats()
    );
    let tokenizer = TiktokenTokenizer::builtin();

    for source in SOURCES {
        let resolver = CountingResolver::new();
        let prepared = prepare_local_graph_inspection_input(
            &resolver,
            &fixture(source.relative_path),
            source.format,
            &tokenizer as &dyn TextTokenizer,
            "chat",
            None,
            0,
        )
        .await
        .unwrap_or_else(|error| panic!("{} inspection fixture: {error:#}", source.format));

        assert_eq!(prepared.bundle.metadata.format, source.format);
        assert!(!prepared.bundle.programs.is_empty());
        assert_eq!(resolver.load_calls(), 1, "{}", source.format);
    }
}
