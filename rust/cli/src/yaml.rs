// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! The YAML Config-v2 input surface: parse a config file into normalized
//! [`crate::load::Inputs`] and reuse the shared [`crate::load::build`] core.
//!
//! Config v2 accepts shorthand forms (`model:` → `models.items[0]`, `dataset:`
//! → `datasets[0]`, a flat `phases:` → one phase) and both snake_case and
//! camelCase keys. Single-word keys are case-identical; multi-word keys carry a
//! `serde(alias)` for the snake_case spelling. Only the common single-run
//! synthetic surface is modeled today; unknown keys are ignored (no
//! `deny_unknown_fields`) and richer keys are added incrementally like the flag
//! surface, each gated by a golden fixture.

use std::path::PathBuf;

use serde::Deserialize;

use crate::load::{self, Inputs, Warmup, default_isl};
use crate::model::dataset::Distribution;
use crate::model::transport::Transport;

/// Parse a YAML config file into one native run.
pub fn resolve(
    path: &std::path::Path,
    artifact_dir: Option<PathBuf>,
) -> anyhow::Result<crate::model::BenchmarkRun> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| anyhow::anyhow!("failed to read config {}: {e}", path.display()))?;
    let file: ConfigFile = serde_yaml::from_str(&text)
        .map_err(|e| anyhow::anyhow!("failed to parse config {}: {e}", path.display()))?;
    let inputs = file.benchmark.into_inputs(artifact_dir)?;
    load::build(inputs)
}

/// A string or a list of strings (Config shorthand for single-vs-many).
#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum StringOrVec {
    One(String),
    Many(Vec<String>),
}

impl StringOrVec {
    fn into_vec(self) -> Vec<String> {
        match self {
            StringOrVec::One(s) => vec![s],
            StringOrVec::Many(v) => v,
        }
    }
}

/// A scalar or a parametric distribution (Config shorthand, e.g. `isl: 512`).
#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum NumOrDist {
    Num(f64),
    Dist(DistFields),
}

#[derive(Debug, Deserialize)]
struct DistFields {
    mean: Option<f64>,
    stddev: Option<f64>,
    min: Option<f64>,
    max: Option<f64>,
}

#[derive(Debug, Deserialize)]
struct ConfigFile {
    benchmark: Benchmark,
}

#[derive(Debug, Deserialize)]
struct Benchmark {
    /// `model:` shorthand (single string or list).
    model: Option<StringOrVec>,
    /// Expanded `models:` block.
    models: Option<ModelsSection>,
    endpoint: EndpointSection,
    /// Orthogonal transport selection (`http` default; `grpc`/`dynosim_*`).
    transport: Option<TransportSection>,
    /// `dataset:` shorthand (single entry).
    dataset: Option<DatasetSection>,
    /// Expanded `datasets:` list (first entry used on the single-run path).
    datasets: Option<Vec<DatasetSection>>,
    tokenizer: Option<TokenizerSection>,
    phases: Phases,
}

#[derive(Debug, Deserialize)]
struct ModelsSection {
    items: Vec<ModelItem>,
}

#[derive(Debug, Deserialize)]
struct ModelItem {
    name: String,
}

#[derive(Debug, Deserialize)]
struct TransportSection {
    #[serde(rename = "type")]
    transport_type: String,
}

#[derive(Debug, Deserialize)]
struct EndpointSection {
    #[serde(rename = "type")]
    endpoint_type: Option<String>,
    url: StringOrVec,
    #[serde(default)]
    streaming: bool,
}

#[derive(Debug, Deserialize)]
struct DatasetSection {
    entries: Option<u32>,
    #[serde(alias = "num_conversations")]
    num_conversations: Option<u32>,
    prompts: Option<PromptsSection>,
}

#[derive(Debug, Deserialize)]
struct PromptsSection {
    isl: Option<NumOrDist>,
    osl: Option<NumOrDist>,
    #[serde(alias = "batch_size")]
    batch_size: Option<u32>,
}

#[derive(Debug, Deserialize)]
struct TokenizerSection {
    name: Option<String>,
    revision: Option<String>,
    #[serde(default, alias = "trust_remote_code")]
    trust_remote_code: bool,
}

/// A flat single phase (shorthand) or a list of phases.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum Phases {
    One(PhaseSection),
    Many(Vec<PhaseSection>),
}

#[derive(Debug, Deserialize)]
struct PhaseSection {
    concurrency: Option<u32>,
    rate: Option<f64>,
    requests: Option<u64>,
    sessions: Option<u64>,
    duration: Option<f64>,
    #[serde(alias = "grace_period")]
    grace_period: Option<f64>,
}

impl Benchmark {
    /// Normalize the parsed config into shared [`Inputs`].
    fn into_inputs(self, artifact_dir: Option<PathBuf>) -> anyhow::Result<Inputs> {
        let model_names = self.resolve_model_names()?;

        let endpoint_type = self
            .endpoint
            .endpoint_type
            .ok_or_else(|| anyhow::anyhow!("endpoint.type is required"))?;

        // Single-run: the shorthand `dataset:` or the first `datasets:` entry.
        let dataset = self
            .dataset
            .or_else(|| self.datasets.and_then(|d| d.into_iter().next()));
        let (isl, osl, batch_size) = extract_prompts(dataset.as_ref());
        let num_conversations = dataset.as_ref().and_then(|d| d.num_conversations);
        let dataset_entries = dataset.as_ref().and_then(|d| d.entries);

        // Single phase on the single-run path.
        let phase = match self.phases {
            Phases::One(p) => p,
            Phases::Many(mut v) => v
                .drain(..)
                .next()
                .ok_or_else(|| anyhow::anyhow!("phases must have at least one entry"))?,
        };

        let entries = num_conversations
            .or(dataset_entries)
            .or(phase.requests.map(|n| n as u32))
            .unwrap_or(load::DEFAULT_ENTRIES);

        let (tokenizer_name, tokenizer_revision, tokenizer_trust) = match self.tokenizer {
            Some(t) => (t.name, t.revision, t.trust_remote_code),
            None => (None, None, false),
        };

        Ok(Inputs {
            model_names,
            urls: self.endpoint.url.into_vec(),
            endpoint_type,
            transport: parse_transport(self.transport.as_ref())?,
            streaming: self.endpoint.streaming,
            timeout_seconds: None,
            use_legacy_max_tokens: false,
            use_server_token_count: false,
            connection_reuse: None,
            request_content_type: None,
            wait_for_model_timeout: None,
            wait_for_model_mode: None,
            wait_for_model_interval: None,
            apply_chat_template: false,
            prefill_concurrency: None,
            prefill_ramp: None,
            gpu_telemetry_enabled: true,
            server_metrics_enabled: true,
            server_metrics_formats: None,
            slos: serde_json::Map::new(),
            network_latency_mean: None,
            network_latency_probe: None,
            otel_url: None,
            api_key: None,
            headers: std::collections::BTreeMap::new(),
            tokenizer_name,
            tokenizer_revision,
            tokenizer_trust,
            isl,
            osl,
            turns: None,
            turn_delay_ratio: 1.0,
            turn_delay_ms: None,
            session_header: None,
            batch_size: batch_size.unwrap_or(1),
            sampling: "sequential".to_string(),
            entries,
            sessions: num_conversations.map(u64::from).or(phase.sessions),
            concurrency: phase.concurrency,
            request_rate: phase.rate,
            rate_mode: None,
            smoothness: None,
            concurrency_ramp: None,
            rate_ramp: None,
            cancellation: None,
            user_centric: None,
            request_count: phase.requests,
            benchmark_duration: phase.duration,
            grace_period: phase.grace_period,
            warmup: None::<Warmup>,
            random_seed: None,
            input_file: None,
            custom_dataset_type: None,
            public_dataset: None,
            fixed_schedule: None,
            fixed_schedule_start_offset: None,
            fixed_schedule_end_offset: None,
            model_strategy: None,
            slice_duration: None,
            isl_block_size: None,
            sketch_metrics: false,
            image_spec: None,
            artifact_dir: artifact_dir.unwrap_or_else(|| PathBuf::from("artifacts")),
        })
    }

    /// Resolve the model list from the `model:` shorthand or `models:` block.
    fn resolve_model_names(&self) -> anyhow::Result<Vec<String>> {
        if let Some(models) = &self.models {
            let names: Vec<String> = models.items.iter().map(|m| m.name.clone()).collect();
            anyhow::ensure!(!names.is_empty(), "models.items must not be empty");
            return Ok(names);
        }
        match &self.model {
            Some(m) => Ok(clone_string_or_vec(m)),
            None => anyhow::bail!("a model is required (set `model:` or `models:`)"),
        }
    }
}

/// Map a YAML `transport.type` string to the typed [`Transport`] (default HTTP).
fn parse_transport(section: Option<&TransportSection>) -> anyhow::Result<Transport> {
    let Some(section) = section else {
        return Ok(Transport::Http);
    };
    Ok(match section.transport_type.as_str() {
        "http" => Transport::Http,
        "grpc" => Transport::Grpc,
        "dynosim_offline" => Transport::DynosimOffline,
        "dynosim_online" => Transport::DynosimOnline,
        other => anyhow::bail!("unknown transport.type {other:?}"),
    })
}

/// Extract the ISL distribution, optional OSL, and batch size from a dataset.
fn extract_prompts(
    dataset: Option<&DatasetSection>,
) -> (Distribution, Option<Distribution>, Option<u32>) {
    let Some(prompts) = dataset.and_then(|d| d.prompts.as_ref()) else {
        return (default_isl(), None, None);
    };
    let isl = match &prompts.isl {
        Some(n) => clone_num_or_dist(n),
        None => default_isl(),
    };
    let osl = prompts.osl.as_ref().map(clone_num_or_dist);
    (isl, osl, prompts.batch_size)
}

/// Clone a `StringOrVec` into a `Vec<String>` without consuming it.
fn clone_string_or_vec(v: &StringOrVec) -> Vec<String> {
    match v {
        StringOrVec::One(s) => vec![s.clone()],
        StringOrVec::Many(list) => list.clone(),
    }
}

/// Clone a `NumOrDist` into a `Distribution` without consuming it.
fn clone_num_or_dist(n: &NumOrDist) -> Distribution {
    match n {
        NumOrDist::Num(value) => Distribution {
            value: Some(*value),
            ..Default::default()
        },
        NumOrDist::Dist(d) => Distribution {
            mean: d.mean,
            stddev: d.stddev,
            min: d.min,
            max: d.max,
            ..Default::default()
        },
    }
}
