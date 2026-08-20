// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Downstream source-compatibility fixture for public graph-input extensions.

#![cfg(feature = "engine")]

use std::sync::atomic::{AtomicUsize, Ordering};

use aiperf_runtime::config::model::workload_kind::GRAPH_FORMATS;
use aiperf_runtime::dataset::{TextTokenizer, TiktokenTokenizer};
use aiperf_runtime::engine::graph_input::{
    BuiltinRunnerGraphInputAdapterResolver, GraphInputAdapterResolver, GraphInputContext,
    PreparedRunnerGraphInput,
};
use anyhow::{Result, anyhow};
use async_trait::async_trait;
use serde_json::value::RawValue;

#[derive(Debug, Default)]
struct ExternalResolver {
    loads: AtomicUsize,
}

#[async_trait(?Send)]
impl GraphInputAdapterResolver for ExternalResolver {
    fn validate_identity(&self, _raw: &RawValue) -> Result<()> {
        Ok(())
    }

    async fn load(
        &self,
        _raw: &RawValue,
        _context: &GraphInputContext<'_>,
    ) -> Result<PreparedRunnerGraphInput> {
        self.loads.fetch_add(1, Ordering::Relaxed);
        Err(anyhow!("compile-only compatibility fixture"))
    }
}

fn old_context_literal(tokenizer: &dyn TextTokenizer) -> GraphInputContext<'_> {
    GraphInputContext {
        tokenizer,
        run_random_seed: Some(7),
    }
}

#[tokio::test]
async fn legacy_public_graph_input_extensions_still_compile_and_forward() {
    let tokenizer = TiktokenTokenizer::builtin();
    let context = old_context_literal(&tokenizer);
    let raw = serde_json::value::to_raw_value(&serde_json::json!({"format": "dag_jsonl"}))
        .expect("raw graph input");
    let resolver = ExternalResolver::default();
    let legacy: &[&str; 6] = &GRAPH_FORMATS;

    assert!(!legacy.contains(&"aiperf_trace"));
    assert!(
        resolver
            .load_for_endpoint(&raw, &context, "chat")
            .await
            .is_err()
    );
    assert_eq!(resolver.loads.load(Ordering::Relaxed), 1);
}

#[test]
fn aiperf_trace_remains_a_built_in_graph_input() {
    assert!(
        BuiltinRunnerGraphInputAdapterResolver::new()
            .supported_formats()
            .contains(&"aiperf_trace")
    );
}
