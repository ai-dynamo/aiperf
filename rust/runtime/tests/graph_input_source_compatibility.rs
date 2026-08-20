// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Downstream source-compatibility fixture for public graph-input extensions.

#![cfg(feature = "engine")]

use aiperf_runtime::config::model::workload_kind::GRAPH_FORMATS;
use aiperf_runtime::dataset::TextTokenizer;
use aiperf_runtime::engine::graph_input::{
    GraphInputAdapterResolver, GraphInputContext, PreparedRunnerGraphInput,
};
use anyhow::Result;
use async_trait::async_trait;
use serde_json::value::RawValue;

#[derive(Debug)]
struct ExternalResolver;

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
        unreachable!("compile-only compatibility fixture")
    }
}

fn old_context_literal(tokenizer: &dyn TextTokenizer) -> GraphInputContext<'_> {
    GraphInputContext {
        tokenizer,
        run_random_seed: Some(7),
    }
}

#[test]
fn pre_extension_context_and_resolver_source_still_compile() {
    let _resolver: &dyn GraphInputAdapterResolver = &ExternalResolver;
    let _constructor = old_context_literal;
}

#[test]
fn graph_format_inventory_has_a_length_independent_type() {
    let formats: &'static [&'static str] = GRAPH_FORMATS;
    assert!(formats.contains(&"aiperf_trace"));
}
