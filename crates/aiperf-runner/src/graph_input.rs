// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Runner-owned direct Graph-IR input adapters.
//!
//! Python projects the authored file source without acquisition or parsing in
//! `src/aiperf/orchestrator/rust_wire.py:220-262`. This module performs an
//! identity-only format lookup, then gives the untouched object to exactly one
//! selected adapter. The adapter owns the sole strict full decode and lowers
//! directly to [`GraphInputBundle`]; no protocol-v1 DTO, linear
//! [`aiperf_dataset::Dataset`],
//! conversation, or second graph-source representation exists in this path.
//!
//! A future graph format registers another [`RunnerGraphInputAdapter`]. Its
//! private authored fields remain invisible to the resolver and coordinator.

use std::collections::BTreeMap;
use std::fmt;
use std::path::PathBuf;
use std::sync::Arc;

use aiperf_dataset::{DatasetSource, LoadConfig, TextTokenizer};
use aiperf_graph::input::{
    DagJsonlGraphInputAdapter, GraphInputAdapter, GraphInputBundle, GraphInputConfig,
};
use anyhow::{Context, Result, anyhow, ensure};
use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{Map, Value, value::RawValue};

use crate::execute::distribution;
use crate::protocol::DistributionSpec;

/// Canonical result retained after one selected graph-input adapter load.
pub struct PreparedRunnerGraphInput {
    /// Complete executable Graph-IR roots plus their frozen segment arena.
    pub bundle: GraphInputBundle,
    /// Dataset-local seed overriding the run root.
    pub random_seed: Option<u64>,
    /// Fallback output-token limit for nodes without an authored value.
    pub default_output_tokens: usize,
}

/// Inputs shared by every direct graph-source adapter.
pub struct RunnerGraphInputContext<'a> {
    /// Fully prepared tokenizer used during segment interning and token counts.
    pub tokenizer: &'a dyn TextTokenizer,
}

/// One direct authored graph-source adapter.
#[async_trait(?Send)]
pub trait RunnerGraphInputAdapter: fmt::Debug + Send + Sync {
    /// Stable authored format discriminator.
    fn format(&self) -> &'static str;

    /// Strictly decode and load one authored source exactly once.
    async fn load(
        &self,
        raw: &RawValue,
        context: &RunnerGraphInputContext<'_>,
    ) -> Result<PreparedRunnerGraphInput>;
}

/// Injected open resolver for direct graph-input adapters.
#[async_trait(?Send)]
pub trait RunnerGraphInputAdapterResolver: fmt::Debug + Send + Sync {
    /// Validate only that the open format identity selects a linked adapter.
    ///
    /// Adapter-owned fields remain untouched. Full strict decoding is deferred
    /// to [`Self::load`], which is invoked exactly once during preparation.
    fn validate_identity(&self, raw: &RawValue) -> Result<()>;

    /// Select the format adapter and retain its canonical Graph-IR output.
    async fn load(
        &self,
        raw: &RawValue,
        context: &RunnerGraphInputContext<'_>,
    ) -> Result<PreparedRunnerGraphInput>;
}

/// Deterministic built-in graph-input adapter composition.
pub struct BuiltinRunnerGraphInputAdapterResolver {
    adapters: BTreeMap<&'static str, Arc<dyn RunnerGraphInputAdapter>>,
}

impl fmt::Debug for BuiltinRunnerGraphInputAdapterResolver {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("BuiltinRunnerGraphInputAdapterResolver")
            .field("formats", &self.adapters.keys().collect::<Vec<_>>())
            .finish_non_exhaustive()
    }
}

impl Default for BuiltinRunnerGraphInputAdapterResolver {
    fn default() -> Self {
        Self::new()
    }
}

impl BuiltinRunnerGraphInputAdapterResolver {
    /// Compose the built-in direct Graph-IR formats.
    pub fn new() -> Self {
        let adapters: [Arc<dyn RunnerGraphInputAdapter>; 1] = [Arc::new(
            DagJsonlRunnerGraphInputAdapter::new(Arc::new(DagJsonlGraphInputAdapter)),
        )];
        Self {
            adapters: adapters
                .into_iter()
                .map(|adapter| (adapter.format(), adapter))
                .collect(),
        }
    }

    fn selected(&self, raw: &RawValue) -> Result<&dyn RunnerGraphInputAdapter> {
        // This intentionally reads only the open discriminator. The selected
        // adapter below remains the sole owner of the full authored object.
        let identity: GraphInputIdentity = serde_json::from_str(raw.get())
            .context("decoding graph-input adapter discriminator")?;
        self.adapters
            .get(identity.format.as_str())
            .map(Arc::as_ref)
            .ok_or_else(|| {
                anyhow!(
                    "no direct Graph-IR input adapter is registered for format {:?}",
                    identity.format
                )
            })
    }
}

#[async_trait(?Send)]
impl RunnerGraphInputAdapterResolver for BuiltinRunnerGraphInputAdapterResolver {
    fn validate_identity(&self, raw: &RawValue) -> Result<()> {
        self.selected(raw).map(drop)
    }

    async fn load(
        &self,
        raw: &RawValue,
        context: &RunnerGraphInputContext<'_>,
    ) -> Result<PreparedRunnerGraphInput> {
        self.selected(raw)?.load(raw, context).await
    }
}

#[derive(Deserialize)]
// Keeping only the discriminator makes Serde skip unknown fields through
// `IgnoredAny` instead of allocating an adapter-owned `Value` tree.
struct GraphInputIdentity {
    format: String,
}

/// Built-in `dag_jsonl` authored wrapper plus canonical lowering adapter.
pub struct DagJsonlRunnerGraphInputAdapter {
    lowerer: Arc<dyn GraphInputAdapter>,
}

impl fmt::Debug for DagJsonlRunnerGraphInputAdapter {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("DagJsonlRunnerGraphInputAdapter")
            .field("lowerer", &self.lowerer.name())
            .finish_non_exhaustive()
    }
}

impl DagJsonlRunnerGraphInputAdapter {
    /// Bind the process-wire adapter to one canonical Graph-IR lowerer.
    pub fn new(lowerer: Arc<dyn GraphInputAdapter>) -> Self {
        Self { lowerer }
    }
}

#[async_trait(?Send)]
impl RunnerGraphInputAdapter for DagJsonlRunnerGraphInputAdapter {
    fn format(&self) -> &'static str {
        "dag_jsonl"
    }

    async fn load(
        &self,
        raw: &RawValue,
        context: &RunnerGraphInputContext<'_>,
    ) -> Result<PreparedRunnerGraphInput> {
        let DagJsonlDatasetInput::File(spec) =
            serde_json::from_str(raw.get()).context("decoding direct dag_jsonl graph input")?;
        spec.validate(self.format())?;
        let default_output_tokens = spec.default_output_tokens()?;
        let random_seed = spec.random_seed;
        let input = spec.into_graph_input_config();
        let bundle = self
            .lowerer
            .load(input, context.tokenizer)
            .await
            .map_err(|error| anyhow!(error.to_string()))
            .context("loading and lowering direct authored dag_jsonl Graph-IR input")?;
        ensure!(
            !bundle.plans.is_empty(),
            "authored Graph-IR input contains no root traces after root limiting"
        );
        ensure!(
            bundle.metadata.format == self.format(),
            "Graph-IR adapter {:?} returned bundle format {:?}",
            self.format(),
            bundle.metadata.format
        );
        Ok(PreparedRunnerGraphInput {
            bundle,
            random_seed,
            default_output_tokens,
        })
    }
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
enum DagJsonlDatasetInput {
    File(DagJsonlFileInput),
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct DagJsonlFileInput {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    path: Option<PathBuf>,
    #[serde(default)]
    records: Option<Value>,
    format: String,
    #[serde(default = "default_sequential")]
    sampling: String,
    #[serde(default)]
    synthesis: Option<Value>,
    #[serde(default)]
    entries: Option<usize>,
    #[serde(default)]
    random_seed: Option<u64>,
    #[serde(default)]
    osl: Option<DistributionSpec>,
    #[serde(default)]
    options: Map<String, Value>,
}

impl DagJsonlFileInput {
    fn validate(&self, expected_format: &str) -> Result<()> {
        ensure!(
            self.name
                .as_ref()
                .is_none_or(|name| !name.trim().is_empty()),
            "graph dataset name must be non-empty when present"
        );
        ensure!(
            self.format == expected_format,
            "direct graph adapter {expected_format:?} received dataset.format={:?}",
            self.format
        );
        ensure!(
            self.path.is_some() ^ self.records.is_some(),
            "direct graph input requires exactly one of path or records"
        );
        ensure!(
            self.sampling.eq_ignore_ascii_case("sequential"),
            "direct Graph-IR input currently requires sequential root selection"
        );
        ensure!(
            self.synthesis.is_none(),
            "direct Graph-IR input does not accept linear trace synthesis"
        );
        ensure!(
            self.entries != Some(0),
            "direct graph root limit must be positive when configured"
        );
        for name in self.options.keys() {
            ensure!(
                name == "inter_turn_delay_cap_seconds",
                "dag_jsonl Graph-IR input does not support option {name:?}"
            );
        }
        if let Some(delay) = self.options.get("inter_turn_delay_cap_seconds") {
            ensure!(
                delay
                    .as_f64()
                    .is_some_and(|value| value.is_finite() && value >= 0.0),
                "inter_turn_delay_cap_seconds must be finite and non-negative"
            );
        }
        Ok(())
    }

    fn default_output_tokens(&self) -> Result<usize> {
        let expected = self
            .osl
            .as_ref()
            .map(distribution)
            .transpose()?
            .map(|value| value.expected_value().ceil())
            .unwrap_or(1.0);
        ensure!(
            expected.is_finite() && expected > 0.0 && expected <= usize::MAX as f64,
            "graph dataset.osl expected value is outside the native usize range"
        );
        Ok(expected as usize)
    }

    fn into_graph_input_config(self) -> GraphInputConfig {
        let source = match (self.path, self.records) {
            (Some(path), None) => DatasetSource::Path(path),
            (None, Some(records)) => DatasetSource::Inline(records),
            _ => unreachable!("source exclusivity validated"),
        };
        let mut load = LoadConfig::new(source);
        load.options = self.options;
        GraphInputConfig {
            load,
            root_limit: self.entries,
        }
    }
}

fn default_sequential() -> String {
    "sequential".into()
}

#[cfg(test)]
mod tests {
    use aiperf_dataset::TiktokenTokenizer;
    use serde_json::json;

    use super::*;

    fn raw(value: Value) -> Box<RawValue> {
        serde_json::value::to_raw_value(&value).unwrap()
    }

    #[test]
    fn identity_decode_skips_adapter_owned_fields_without_retaining_them() {
        assert_eq!(
            std::mem::size_of::<GraphInputIdentity>(),
            std::mem::size_of::<String>(),
            "the selector DTO must retain only its discriminator"
        );
        let resolver = BuiltinRunnerGraphInputAdapterResolver::new();
        resolver
            .validate_identity(&raw(json!({
                "type": "file",
                "format": "dag_jsonl",
                "future_adapter_field": {
                    "opaque": "x".repeat(1 << 20),
                    "nested": [{"owned_by": "selected adapter"}]
                }
            })))
            .unwrap();
    }

    #[tokio::test]
    async fn selected_adapter_owns_the_only_strict_decode_and_direct_load() {
        let resolver = BuiltinRunnerGraphInputAdapterResolver::new();
        let input = raw(json!({
            "type": "file",
            "format": "dag_jsonl",
            "sampling": "sequential",
            "records": [{
                "session_id": "root",
                "turns": [{"messages": [{"role": "user", "content": "hello"}]}]
            }],
            "osl": {"value": 3.0},
            "options": {"inter_turn_delay_cap_seconds": 0.5}
        }));
        let prepared = resolver
            .load(
                &input,
                &RunnerGraphInputContext {
                    tokenizer: &TiktokenTokenizer::builtin(),
                },
            )
            .await
            .unwrap();

        assert_eq!(prepared.bundle.metadata.format, "dag_jsonl");
        assert_eq!(prepared.bundle.metadata.root_count, 1);
        assert_eq!(prepared.bundle.metadata.node_count, 1);
        assert_eq!(prepared.default_output_tokens, 3);
    }

    #[tokio::test]
    async fn selected_adapter_rejects_unknown_full_shape_fields() {
        let resolver = BuiltinRunnerGraphInputAdapterResolver::new();
        let input = raw(json!({
            "type": "file",
            "format": "dag_jsonl",
            "records": [],
            "future_adapter_field": true
        }));
        let error = resolver
            .load(
                &input,
                &RunnerGraphInputContext {
                    tokenizer: &TiktokenTokenizer::builtin(),
                },
            )
            .await
            .err()
            .expect("unknown adapter fields must fail");
        assert!(format!("{error:#}").contains("future_adapter_field"));
    }
}
