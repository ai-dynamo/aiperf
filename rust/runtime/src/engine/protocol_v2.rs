// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Strict authored protocol-v2 request and response types.
//!
//! The native profile layer performs structural Config-v2 expansion and
//! serializes one authored run.
//! The selected runner owns component discovery, strict factory-specific config
//! decoding, preparation, execution, and reporting. Factory-owned objects stay
//! as [`RawValue`] until their registered implementation decodes them; this is
//! what keeps transport and workload identities open without weakening the outer
//! process contract.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::path::Component;
use std::path::PathBuf;
use std::str::FromStr;

use anyhow::{Result, anyhow, ensure};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::{Map, Value, value::RawValue};

use crate::config::model::workload_kind::{is_graph_format, WorkloadKind};
use crate::engine::protocol::{
    DispatchMode, MetricsSpec, ModelSelectionStrategy, ModelsSpec, VariationSpec,
};
use crate::engine::sidecar_input::{
    AuthoredSidecarInput, CONTENT_SERVER_SIDECAR_ID, GPU_TELEMETRY_SIDECAR_ID,
    LIVE_STREAMING_SIDECAR_ID, NETWORK_LATENCY_SIDECAR_ID, SERVER_METRICS_SIDECAR_ID,
};

/// Authored runner protocol version.
pub const PROTOCOL_V2: u32 = 2;

/// Open identifier resolved through a frozen runner registry.
///
/// IDs deliberately use a small wire-safe grammar so they can be used as
/// deterministic registry keys, report values, and extension namespaces.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ComponentId(String);

impl ComponentId {
    /// Return the normalized identifier.
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Consume the wrapper and return its owned identifier.
    pub fn into_string(self) -> String {
        self.0
    }
}

impl fmt::Display for ComponentId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl FromStr for ComponentId {
    type Err = String;

    fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
        let mut bytes = value.bytes();
        let Some(first) = bytes.next() else {
            return Err("runner component ID cannot be empty".into());
        };
        if !first.is_ascii_lowercase() {
            return Err(format!(
                "runner component ID {value:?} must start with a lowercase ASCII letter"
            ));
        }
        if !bytes.all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'_') {
            return Err(format!(
                "runner component ID {value:?} may contain only lowercase ASCII letters, digits, and underscores"
            ));
        }
        Ok(Self(value.to_owned()))
    }
}

impl Serialize for ComponentId {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for ComponentId {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        value.parse().map_err(serde::de::Error::custom)
    }
}

/// Operation performed by one fresh runner process.
#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum OperationV2 {
    /// Perform side-effect-free structural and static semantic validation.
    Validate,
    /// Repeat validation, prepare the run, execute it, and commit its report.
    Execute,
}

/// One strict protocol-v2 execution envelope reconstructed around the bare
/// [`BenchmarkRunWireV2`] stdin payload.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EnvelopeV2 {
    /// Wire protocol discriminator; it must equal [`PROTOCOL_V2`].
    pub protocol_version: u32,
    /// Requested process operation.
    pub operation: OperationV2,
    /// Exact Config-v2 run, including resolved bindings.
    pub run: BenchmarkRunWireV2,
}

impl EnvelopeV2 {
    /// Validate invariants owned by the versioned outer protocol.
    ///
    /// Component-specific config is intentionally not inspected here. Frozen
    /// factories own that strict decode during registry validation.
    pub fn validate_outer(&self) -> Result<()> {
        ensure!(
            self.protocol_version == PROTOCOL_V2,
            "runner protocol {} is unsupported; expected {PROTOCOL_V2}",
            self.protocol_version
        );
        self.run.validate_outer()
    }
}

/// Default thread-per-core worker count when `runtime.workers` is unset: the
/// machine's available parallelism, so a load run uses every core. Falls back to
/// `1` when the platform cannot report parallelism.
fn default_worker_count() -> u64 {
    std::thread::available_parallelism()
        .map(|n| n.get() as u64)
        .unwrap_or(1)
}

/// Decode the optional `runtime.dispatch` admission-strategy selector.
///
/// An explicit `runtime.dispatch` always wins. When it is absent, the default is
/// **cellular-aware**:
/// - single-process (`cells <= 1`) → [`DispatchMode::Global`] (the byte-exact,
///   parity-preserving default), and
/// - cellular (`cells > 1`) → [`DispatchMode::Sharded`].
///
/// The cellular default is `Sharded` because a cellular run has *already* forfeited
/// single-process byte-exact determinism: each cell partitions its slice of the
/// dispatch stream autonomously and the per-cell records merge at the end, so
/// `Global`'s shared cross-thread admission gate *inside* a cell buys parity that is
/// already gone — pure cross-thread overhead (measured ~7-8x slower than `Sharded`
/// in cellular mode on a c4-144). Cell subprocesses see the same `runtime.cells > 1`
/// in their envelope, so they resolve `Sharded` here too. An unrecognized string is a
/// hard decode error rather than a silent fallback.
fn parse_dispatch_mode(runtime: &Value) -> Result<DispatchMode> {
    match runtime.get("dispatch") {
        None | Some(Value::Null) => {
            let cells = runtime.get("cells").and_then(Value::as_u64).unwrap_or(1);
            if cells > 1 {
                Ok(DispatchMode::Sharded)
            } else {
                Ok(DispatchMode::default())
            }
        }
        Some(value) => serde_json::from_value(value.clone())
            .map_err(|error| anyhow!("run.cfg.runtime.dispatch: {error}")),
    }
}

/// Exact outer BenchmarkRun shape accepted by the product wire.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BenchmarkRunWireV2 {
    /// Stable benchmark identifier.
    pub benchmark_id: String,
    /// Runner-owned artifact directory.
    pub artifact_dir: PathBuf,
    /// Canonical benchmark configuration.
    pub cfg: BenchmarkConfigWireV2,
    /// Resolution facts computed before runner execution.
    #[serde(default)]
    pub resolved: Value,
    /// Optional sweep metadata retained without runner interpretation.
    #[serde(default)]
    pub sweep_id: Option<String>,
    /// Optional outer-loop variation retained without runner interpretation.
    #[serde(default)]
    pub variation: Option<VariationSpec>,
    /// Zero-based trial number.
    #[serde(default)]
    pub trial: usize,
    /// Human-readable run label.
    #[serde(default)]
    pub label: String,
    /// Redacted invoking command.
    #[serde(default)]
    pub cli_command: Option<String>,
    /// Deterministic root seed when authored.
    #[serde(default)]
    pub random_seed: Option<u64>,
    /// Envelope-level template variables.
    #[serde(default)]
    pub variables: BTreeMap<String, Value>,
}

/// Runner-relevant subset of the canonical BenchmarkConfig dump.
///
/// Unknown Config keys are ignored because the canonical dump includes fields this
/// runner does not consume, such as `model`, `warmup`, and `profiling`.
#[derive(Deserialize)]
pub struct BenchmarkConfigWireV2 {
    /// Model-selection policy.
    pub models: Value,
    /// Default endpoint profile.
    pub endpoint: Value,
    /// Additional named endpoint profiles.
    #[serde(default)]
    pub endpoint_profiles: BTreeMap<String, Value>,
    /// Canonical single-dataset list.
    #[serde(default)]
    pub datasets: Vec<Value>,
    /// Ordered phase policy.
    pub phases: Vec<Value>,
    /// Optional tokenizer policy.
    #[serde(default)]
    pub tokenizer: Option<Value>,
    /// Inline Config transport selection.
    pub transport: Value,
    /// Worker-count configuration.
    #[serde(default)]
    pub runtime: Value,
    /// Native output policy.
    #[serde(default)]
    pub artifacts: Value,
    /// Native metrics policy.
    #[serde(default)]
    pub metrics: Value,
    /// Optional run-failure policy (`"continue"` / `"abort"`); absent selects
    /// resilient scheduled execution or fail-fast graph execution.
    #[serde(default)]
    pub failure_policy: Value,
    /// Goodput/SLO policy.
    #[serde(default)]
    pub slos: Value,
    /// Explicit goodput policy retained for Config revisions that expose it separately.
    #[serde(default)]
    pub goodput: Value,
    /// GPU telemetry sidecar configuration.
    #[serde(default)]
    pub gpu_telemetry: Value,
    /// Server-metrics sidecar configuration.
    #[serde(default)]
    pub server_metrics: Value,
    /// Network-latency sidecar configuration.
    #[serde(default)]
    pub network_latency: Value,
    /// Generated-content sidecar configuration.
    #[serde(default)]
    pub content_server: Value,
    /// Prepared sidecar bag, when present.
    #[serde(default)]
    pub sidecars: Value,
    /// Native post-report export policy; absence decodes to all-disabled
    /// defaults.
    #[serde(default)]
    pub export: Value,
    /// Resolved WEKA reconstruction semantics (`legacy`|`graph-ir`); authored into
    /// the graph workload so the engine selects the legacy agentic path. Absent
    /// defers to the graph-ir default.
    #[serde(default)]
    pub weka_semantics: Option<String>,
}

impl BenchmarkRunWireV2 {
    /// Validate wire-only invariants before adapting to linked factories.
    pub fn validate_outer(&self) -> Result<()> {
        ensure!(
            !self.benchmark_id.trim().is_empty(),
            "run.benchmark_id cannot be empty"
        );
        ensure!(
            !self.artifact_dir.as_os_str().is_empty(),
            "run.artifact_dir cannot be empty"
        );
        ensure!(
            !self.cfg.datasets.is_empty(),
            "run.cfg.datasets must contain exactly one dataset"
        );
        Ok(())
    }

    /// Adapt canonical Config nesting to the linked preparation seam.
    pub(crate) fn into_authored(self) -> Result<AuthoredRunSpecV2> {
        self.validate_outer()?;
        let dataset = self
            .cfg
            .datasets
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("run.cfg.datasets must contain one dataset"))?;
        let workload_kind = if is_graph_format(dataset_type(&dataset)) {
            WorkloadKind::Graph
        } else {
            WorkloadKind::Scheduled
        };
        let workload_id = workload_kind.workload_id();
        let transport = component_from_inline(self.cfg.transport, "run.cfg.transport")?;
        let worker_count = self
            .cfg
            .runtime
            .get("workers")
            .and_then(Value::as_u64)
            // Unset `runtime.workers` auto-selects the machine's parallelism so a
            // load run uses every core (thread-per-core), instead of a single
            // worker. An explicit `runtime.workers` (including `1`) always wins.
            .unwrap_or_else(default_worker_count);
        ensure!(
            worker_count > 0 && worker_count <= usize::MAX as u64,
            "run.cfg.runtime.workers must be a positive usize"
        );
        let dispatch = parse_dispatch_mode(&self.cfg.runtime)?;
        let mut workload_config = serde_json::json!({
            "worker_count": worker_count,
            "dataset": dataset,
            "tokenizer": self.cfg.tokenizer.unwrap_or_else(|| serde_json::json!({})),
            "phases": self.cfg.phases,
            "failure_policy": self.cfg.failure_policy,
        });
        // `weka_semantics` selects the graph reconstruction pipeline and exists
        // only on the graph workload's config DTO. The scheduled workload DTO is
        // strict (`deny_unknown_fields`), so emitting the field there — even as
        // `null` — fails decode. Only attach it to the graph workload.
        if workload_kind == WorkloadKind::Graph {
            workload_config["weka_semantics"] = serde_json::json!(self.cfg.weka_semantics);
        }
        let workload = NamedRunnerComponentSpecV2 {
            id: workload_id.parse().expect("built-in workload ID is valid"),
            config: raw_value(workload_config)?,
        };
        let (sidecars, sidecars_present) = if !self.cfg.sidecars.is_null()
            && self
                .cfg
                .sidecars
                .as_object()
                .is_some_and(|object| !object.is_empty())
        {
            (
                serde_json::from_value(self.cfg.sidecars)
                    .map_err(|error| anyhow!("run.cfg.sidecars: {error}"))?,
                true,
            )
        } else if self.cfg.content_server.is_null() {
            (SidecarSpecV2::default(), false)
        } else {
            (
                SidecarSpecV2 {
                    content_server: Some(raw_value(self.cfg.content_server)?),
                    ..SidecarSpecV2::default()
                },
                true,
            )
        };
        Ok(AuthoredRunSpecV2 {
            identity: RunIdentitySpecV2 {
                benchmark_id: self.benchmark_id,
                sweep_id: self.sweep_id,
                label: self.label,
                trial: self.trial,
                random_seed: self.random_seed,
                variation: self.variation,
            },
            artifact_target: self.artifact_dir,
            models: models_from_config(self.cfg.models)?,
            endpoints: endpoint_profiles(self.cfg.endpoint, self.cfg.endpoint_profiles)?,
            transport,
            workload,
            metrics: serde_json::from_value(self.cfg.metrics).unwrap_or_default(),
            artifacts: serde_json::from_value(self.cfg.artifacts).unwrap_or_default(),
            export: serde_json::from_value(self.cfg.export).unwrap_or_default(),
            sidecars,
            dispatch,
            resource_presence: ResourcePresenceV2 {
                models: true,
                endpoints: true,
                metrics: true,
                artifacts: true,
                sidecars: sidecars_present,
            },
        })
    }
}

fn models_from_config(value: Value) -> Result<ModelsSpec> {
    let mut models = value
        .as_object()
        .cloned()
        .ok_or_else(|| anyhow!("run.cfg.models must be an object"))?;
    if let Some(Value::Array(items)) = models.get_mut("items") {
        for item in items {
            if let Some(item) = item.as_object_mut() {
                item.retain(|key, _| matches!(key.as_str(), "name" | "weight"));
            }
        }
    }
    serde_json::from_value(Value::Object(models))
        .map_err(|error| anyhow!("run.cfg.models: {error}"))
}

fn dataset_type(dataset: &Value) -> Option<&str> {
    dataset
        .get("format")
        .and_then(Value::as_str)
        .or_else(|| dataset.get("type").and_then(Value::as_str))
}

fn component_from_inline(value: Value, field: &str) -> Result<NamedRunnerComponentSpecV2> {
    let mut object = value
        .as_object()
        .cloned()
        .ok_or_else(|| anyhow!("{field} must be an object"))?;
    let id = object
        .remove("type")
        .and_then(|value| value.as_str().map(str::to_owned))
        .ok_or_else(|| anyhow!("{field}.type must be a string"))?
        .parse()
        .map_err(|error: String| anyhow!("{field}.type: {error}"))?;
    Ok(NamedRunnerComponentSpecV2 {
        id,
        config: raw_value(Value::Object(object))?,
    })
}

fn endpoint_profiles(
    default: Value,
    additional: BTreeMap<String, Value>,
) -> Result<EndpointProfilesSpecV2> {
    let mut profiles = Vec::with_capacity(additional.len() + 1);
    profiles.push(raw_value(endpoint_profile("default", default)?)?);
    for (id, config) in additional {
        profiles.push(raw_value(endpoint_profile(&id, config)?)?);
    }
    Ok(EndpointProfilesSpecV2 { profiles })
}

fn endpoint_profile(id: &str, value: Value) -> Result<Value> {
    let mut profile = value
        .as_object()
        .cloned()
        .ok_or_else(|| anyhow!("run.cfg.endpoint must be an object"))?;
    profile.insert("id".to_owned(), Value::String(id.to_owned()));
    if let Some(timeout) = profile.remove("timeout") {
        profile.insert("timeout_seconds".to_owned(), timeout);
    }
    profile.remove("url_strategy");
    Ok(Value::Object(profile))
}

fn raw_value(value: Value) -> Result<Box<RawValue>> {
    RawValue::from_string(serde_json::to_string(&value)?).map_err(Into::into)
}

/// Authored identity and runner-owned execution inputs for one run.
///
/// The wire shape places optional product resources under one strict
/// `resources` object. Concrete values are retained in flat resolved fields so
/// established prepared pair implementations cannot accidentally decode the
/// authored JSON a second time; [`Self::resource_is_present`] preserves the
/// required/optional/forbidden distinction for workload validation.
pub struct AuthoredRunSpecV2 {
    /// Stable identity projected from the outer orchestrator.
    pub identity: RunIdentitySpecV2,
    /// Exclusive artifact target selected but not yet created.
    pub artifact_target: PathBuf,
    /// Resolved model-selection policy; empty only when the resource was absent.
    pub models: ModelsSpec,
    /// Resolved endpoint profiles; empty only when the resource was absent.
    pub endpoints: EndpointProfilesSpecV2,
    /// Open transport selection (the `{transport, clock}` execution axis).
    pub transport: NamedRunnerComponentSpecV2,
    /// Open workload selection.
    pub workload: NamedRunnerComponentSpecV2,
    /// Resolved native metrics policy.
    pub metrics: MetricsSpec,
    /// Runner-owned artifact policy.
    pub artifacts: ArtifactSpecV2,
    /// Optional supervised sidecars, retained raw until their native factory
    /// performs its strict decode.
    pub sidecars: SidecarSpecV2,
    /// Native post-report export policy driving the [`crate::export`] plane.
    pub export: crate::export::ExportConfig,
    /// Admission strategy for `workers>1` scheduled execution
    /// (`runtime.dispatch`; defaults to [`DispatchMode::Global`]). Config
    /// surface only: not yet wired into execution behavior.
    pub dispatch: DispatchMode,
    resource_presence: ResourcePresenceV2,
}

/// Optional authored resources classified by the selected workload factory.
#[derive(Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AuthoredRunResourcesV2 {
    /// Inference model-selection policy.
    #[serde(default)]
    pub models: Option<ModelsSpec>,
    /// Inference endpoint profiles.
    #[serde(default)]
    pub endpoints: Option<EndpointProfilesSpecV2>,
    /// Native metric aggregation policy.
    #[serde(default)]
    pub metrics: Option<MetricsSpec>,
    /// Generic runner-owned artifact policy.
    #[serde(default)]
    pub artifacts: Option<ArtifactSpecV2>,
    /// Optional prepared telemetry/process sidecars.
    #[serde(default)]
    pub sidecars: Option<SidecarSpecV2>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct AuthoredRunWireV2 {
    identity: RunIdentitySpecV2,
    artifact_target: PathBuf,
    transport: NamedRunnerComponentSpecV2,
    workload: NamedRunnerComponentSpecV2,
    #[serde(default)]
    resources: AuthoredRunResourcesV2,
    #[serde(default)]
    dispatch: DispatchMode,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct ResourcePresenceV2 {
    models: bool,
    endpoints: bool,
    metrics: bool,
    artifacts: bool,
    sidecars: bool,
}

impl<'de> Deserialize<'de> for AuthoredRunSpecV2 {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = AuthoredRunWireV2::deserialize(deserializer)?;
        let resource_presence = ResourcePresenceV2 {
            models: wire.resources.models.is_some(),
            endpoints: wire.resources.endpoints.is_some(),
            metrics: wire.resources.metrics.is_some(),
            artifacts: wire.resources.artifacts.is_some(),
            sidecars: wire.resources.sidecars.is_some(),
        };
        Ok(Self {
            identity: wire.identity,
            artifact_target: wire.artifact_target,
            models: wire.resources.models.unwrap_or_else(empty_models),
            endpoints: wire.resources.endpoints.unwrap_or_default(),
            transport: wire.transport,
            workload: wire.workload,
            metrics: wire.resources.metrics.unwrap_or_default(),
            artifacts: wire.resources.artifacts.unwrap_or_default(),
            sidecars: wire.resources.sidecars.unwrap_or_default(),
            export: crate::export::ExportConfig::default(),
            dispatch: wire.dispatch,
            resource_presence,
        })
    }
}

fn empty_models() -> ModelsSpec {
    ModelsSpec {
        strategy: ModelSelectionStrategy::RoundRobin,
        items: Vec::new(),
    }
}

/// Resource fields whose presence is workload-classified.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum RunResourceV2 {
    /// Model-selection policy.
    Models,
    /// Endpoint profiles.
    Endpoints,
    /// Native metrics policy.
    Metrics,
    /// Generic artifact policy.
    Artifacts,
    /// Sidecar resources.
    Sidecars,
}

impl RunResourceV2 {
    /// Stable authored field name.
    #[must_use]
    pub const fn field_name(self) -> &'static str {
        match self {
            Self::Models => "models",
            Self::Endpoints => "endpoints",
            Self::Metrics => "metrics",
            Self::Artifacts => "artifacts",
            Self::Sidecars => "sidecars",
        }
    }
}

impl AuthoredRunSpecV2 {
    /// Validate common authored invariants without performing IO.
    pub fn validate_outer(&self) -> Result<()> {
        ensure!(
            !self.identity.benchmark_id.trim().is_empty(),
            "benchmark_id cannot be empty"
        );
        ensure!(
            !self.artifact_target.as_os_str().is_empty(),
            "artifact_target cannot be empty"
        );
        self.transport.validate_outer("transport")?;
        self.workload.validate_outer("workload")?;
        if self.resource_is_present(RunResourceV2::Models) {
            validate_models(&self.models)?;
        }
        if self.resource_is_present(RunResourceV2::Endpoints) {
            self.endpoints.validate_outer()?;
        }
        if self.resource_is_present(RunResourceV2::Metrics) {
            validate_metrics(&self.metrics)?;
        }
        if self.resource_is_present(RunResourceV2::Artifacts) {
            self.artifacts.validate_outer()?;
        }
        if self.resource_is_present(RunResourceV2::Sidecars) {
            self.sidecars.validate_outer()?;
        }
        Ok(())
    }

    /// Whether one resource block was explicitly present on the wire.
    #[must_use]
    pub const fn resource_is_present(&self, resource: RunResourceV2) -> bool {
        match resource {
            RunResourceV2::Models => self.resource_presence.models,
            RunResourceV2::Endpoints => self.resource_presence.endpoints,
            RunResourceV2::Metrics => self.resource_presence.metrics,
            RunResourceV2::Artifacts => self.resource_presence.artifacts,
            RunResourceV2::Sidecars => self.resource_presence.sidecars,
        }
    }
}

fn validate_models(models: &ModelsSpec) -> Result<()> {
    let mut total_weight = 0.0;
    for (index, item) in models.items.iter().enumerate() {
        ensure!(
            !item.name.trim().is_empty(),
            "models.items[{index}].name cannot be empty"
        );
        match (models.strategy, item.weight) {
            (ModelSelectionStrategy::Weighted, Some(weight)) => {
                ensure!(
                    weight.is_finite() && weight > 0.0,
                    "models.items[{index}].weight must be finite and positive"
                );
                total_weight += weight;
            }
            (ModelSelectionStrategy::Weighted, None) => {
                return Err(anyhow!(
                    "models.items[{index}].weight is required for weighted selection"
                ));
            }
            (_, Some(weight)) => ensure!(
                weight.is_finite() && weight >= 0.0,
                "models.items[{index}].weight must be finite and non-negative"
            ),
            (_, None) => {}
        }
    }
    if matches!(models.strategy, ModelSelectionStrategy::Weighted) {
        ensure!(
            total_weight.is_finite() && total_weight > 0.0,
            "weighted model selection requires a positive finite total weight"
        );
    }
    Ok(())
}

fn validate_metrics(metrics: &MetricsSpec) -> Result<()> {
    if let Some(duration) = metrics.slice_duration_seconds {
        ensure!(
            duration.is_finite() && duration > 0.0,
            "metrics.slice_duration_seconds must be finite and positive"
        );
    }
    for (name, threshold) in &metrics.slos {
        ensure!(!name.trim().is_empty(), "metrics SLO name cannot be empty");
        ensure!(
            threshold.is_finite() && *threshold >= 0.0,
            "metrics SLO {name:?} must be finite and non-negative"
        );
    }
    if let Some(fraction) = metrics.steady_state.fraction {
        ensure!(
            fraction.is_finite() && fraction > 0.0 && fraction <= 1.0,
            "metrics.steady_state.fraction must be finite and in (0, 1]"
        );
    }
    Ok(())
}

/// Stable run identity retained across validation, execution, and reports.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RunIdentitySpecV2 {
    /// Stable benchmark identifier.
    pub benchmark_id: String,
    /// Optional outer sweep identifier.
    #[serde(default)]
    pub sweep_id: Option<String>,
    /// Human-readable run label.
    #[serde(default)]
    pub label: String,
    /// Zero-based trial number.
    #[serde(default)]
    pub trial: usize,
    /// Deterministic root seed when authored.
    #[serde(default)]
    pub random_seed: Option<u64>,
    /// Optional outer-loop variation coordinates.
    #[serde(default)]
    pub variation: Option<VariationSpec>,
}

/// An open registered component plus its implementation-owned config object.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NamedRunnerComponentSpecV2 {
    /// Frozen-registry identifier.
    #[serde(rename = "type")]
    pub id: ComponentId,
    /// Strictly decoded by the selected factory.
    pub config: Box<RawValue>,
}

impl NamedRunnerComponentSpecV2 {
    /// Require the factory-owned payload to be a JSON object while leaving its
    /// keys entirely to the selected implementation.
    pub fn validate_outer(&self, field: &str) -> Result<()> {
        let value: Value = serde_json::from_str(self.config.get())
            .map_err(|error| anyhow!("{field}.config is invalid JSON: {error}"))?;
        ensure!(value.is_object(), "{field}.config must be a JSON object");
        Ok(())
    }
}

/// Authored endpoint profiles shared by every transport/workload pair.
#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EndpointProfilesSpecV2 {
    /// Non-empty raw profiles. Each object must carry `id` and `type`; the
    /// selected endpoint factory owns every remaining key.
    pub profiles: Vec<Box<RawValue>>,
}

impl EndpointProfilesSpecV2 {
    /// Parse profile identities and reject duplicate profile names.
    pub fn validate_outer(&self) -> Result<()> {
        ensure!(
            !self.profiles.is_empty(),
            "at least one endpoint profile is required"
        );
        let mut seen = BTreeSet::new();
        for (index, profile) in self.profiles.iter().enumerate() {
            let identity = endpoint_profile_identity(profile)
                .map_err(|error| anyhow!("endpoint profile {index}: {error}"))?;
            ensure!(
                seen.insert(identity.profile_id.clone()),
                "duplicate endpoint profile ID {:?}",
                identity.profile_id
            );
        }
        Ok(())
    }

    /// Return validated profile identities in authored order.
    pub fn identities(&self) -> Result<Vec<EndpointProfileIdentityV2>> {
        self.profiles
            .iter()
            .map(|profile| endpoint_profile_identity(profile))
            .collect()
    }
}

/// Identity fields common to every endpoint profile implementation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EndpointProfileIdentityV2 {
    /// Run-local profile name used by workloads.
    pub profile_id: String,
    /// Open endpoint factory ID.
    pub endpoint_id: ComponentId,
}

fn endpoint_profile_identity(profile: &RawValue) -> Result<EndpointProfileIdentityV2> {
    let object: Map<String, Value> = serde_json::from_str(profile.get())
        .map_err(|error| anyhow!("must be a JSON object: {error}"))?;
    let profile_id = object
        .get("id")
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow!("id must be a string"))?;
    ensure!(
        !profile_id.is_empty() && profile_id.trim() == profile_id,
        "id must be non-empty and contain no surrounding whitespace"
    );
    let endpoint_id = object
        .get("type")
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow!("type must be a string"))?
        .parse()
        .map_err(|error: String| anyhow!(error))?;
    Ok(EndpointProfileIdentityV2 {
        profile_id: profile_id.to_owned(),
        endpoint_id,
    })
}

/// Artifact outputs committed only after complete preparation succeeds.
#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArtifactSpecV2 {
    /// Per-request metric records path relative to the artifact target.
    #[serde(default)]
    pub records_path: Option<PathBuf>,
    /// Wide per-request metric Parquet sidecar path relative to the artifact
    /// target, or absent when the columnar export is disabled.
    #[serde(default)]
    pub records_parquet_path: Option<PathBuf>,
    /// Per-request metric CSV sidecar path relative to the artifact target, or
    /// absent when the CSV export is disabled.
    #[serde(default)]
    pub records_csv_path: Option<PathBuf>,
    /// Raw request/response records path relative to the artifact target.
    #[serde(default)]
    pub raw_path: Option<PathBuf>,
    /// Aggregated response-text output path relative to the artifact target.
    #[serde(default)]
    pub outputs_path: Option<PathBuf>,
    /// Per-session formatted request payloads (`inputs.json`) path relative to
    /// the artifact target.
    #[serde(default)]
    pub inputs_path: Option<PathBuf>,
    /// Include transport trace details in records.
    #[serde(default)]
    pub trace: bool,
    /// Authored user files materialized by the runner after validation.
    #[serde(default)]
    pub user_files: Vec<UserFileSpecV2>,
    /// Base path for the `--dry-run` dataset-analysis artifact family, relative to
    /// the artifact target. Present only when the dry-run analysis is requested.
    #[serde(default)]
    pub dataset_analysis_path: Option<PathBuf>,
    /// KV-cache block size (tokens) for the dry-run cache-reuse analysis. Absent →
    /// the analysis default (16).
    #[serde(default)]
    pub dataset_analysis_block_size: Option<u32>,
    /// Explicit realized-LRU cache capacity (blocks) sweep point for the dry-run
    /// analysis. Absent → the capacity sweep only.
    #[serde(default)]
    pub dataset_analysis_cache_blocks: Option<u64>,
    /// Request per-conversation breakdowns in the dry-run analysis.
    #[serde(default)]
    pub dataset_analysis_per_conversation: bool,
}

impl ArtifactSpecV2 {
    fn validate_outer(&self) -> Result<()> {
        let mut paths = BTreeSet::new();
        for (field, path) in [
            ("artifacts.records_path", self.records_path.as_ref()),
            (
                "artifacts.records_parquet_path",
                self.records_parquet_path.as_ref(),
            ),
            ("artifacts.records_csv_path", self.records_csv_path.as_ref()),
            ("artifacts.raw_path", self.raw_path.as_ref()),
            ("artifacts.outputs_path", self.outputs_path.as_ref()),
            ("artifacts.inputs_path", self.inputs_path.as_ref()),
        ] {
            if let Some(path) = path {
                validate_relative_artifact_path(path, field)?;
                ensure!(
                    paths.insert(path.clone()),
                    "duplicate artifact output path {path:?}"
                );
            }
        }
        for (index, file) in self.user_files.iter().enumerate() {
            let path = PathBuf::from(&file.path);
            validate_relative_artifact_path(&path, &format!("artifacts.user_files[{index}].path"))?;
            ensure!(
                paths.insert(path.clone()),
                "duplicate artifact output path {path:?}"
            );
        }
        Ok(())
    }
}

fn validate_relative_artifact_path(path: &std::path::Path, field: &str) -> Result<()> {
    ensure!(!path.as_os_str().is_empty(), "{field} cannot be empty");
    ensure!(!path.is_absolute(), "{field} must be relative");
    ensure!(
        path.components()
            .all(|component| matches!(component, Component::Normal(_))),
        "{field} must contain only normal relative path components"
    );
    Ok(())
}

/// One user-authored file to materialize under the run artifact target.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UserFileSpecV2 {
    /// POSIX-style relative output path.
    pub path: String,
    /// Selected serialization format.
    pub format: UserFileFormatV2,
    /// Rendered and serialized UTF-8 content.
    pub content: String,
}

/// Supported runner-side user-file encodings.
#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum UserFileFormatV2 {
    /// Pre-serialized pretty JSON.
    Json,
    /// Pre-serialized YAML.
    Yaml,
    /// UTF-8 text.
    Text,
}

/// Optional ancillary process/collector intent.
#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SidecarSpecV2 {
    /// Run-owned HTTP content-server factory config.
    #[serde(default)]
    pub content_server: Option<Box<RawValue>>,
    /// GPU telemetry factory-owned config.
    #[serde(default)]
    pub gpu_telemetry: Option<Box<RawValue>>,
    /// Network-latency factory-owned config.
    #[serde(default)]
    pub network_latency: Option<Box<RawValue>>,
    /// Server-metrics factory-owned config.
    #[serde(default)]
    pub server_metrics: Option<Box<RawValue>>,
    /// Live Python OTel/MLflow worker config.
    #[serde(default)]
    pub live_streaming: Option<Box<RawValue>>,
}

impl SidecarSpecV2 {
    /// Retain each present body beside the open adapter ID selected by its key.
    ///
    /// This is a borrowed structural projection only: the coordinator passes
    /// each raw body directly to the selected adapter, which owns the sole
    /// full decode.
    pub(crate) fn authored_inputs(&self) -> Vec<AuthoredSidecarInput<'_>> {
        [
            (CONTENT_SERVER_SIDECAR_ID, self.content_server.as_deref()),
            (GPU_TELEMETRY_SIDECAR_ID, self.gpu_telemetry.as_deref()),
            (NETWORK_LATENCY_SIDECAR_ID, self.network_latency.as_deref()),
            (SERVER_METRICS_SIDECAR_ID, self.server_metrics.as_deref()),
            (LIVE_STREAMING_SIDECAR_ID, self.live_streaming.as_deref()),
        ]
        .into_iter()
        .filter_map(|(id, config)| config.map(|config| AuthoredSidecarInput { id, config }))
        .collect()
    }

    fn validate_outer(&self) -> Result<()> {
        for (field, raw) in [
            ("sidecars.content_server", self.content_server.as_deref()),
            ("sidecars.gpu_telemetry", self.gpu_telemetry.as_deref()),
            ("sidecars.network_latency", self.network_latency.as_deref()),
            ("sidecars.server_metrics", self.server_metrics.as_deref()),
            ("sidecars.live_streaming", self.live_streaming.as_deref()),
        ] {
            let Some(raw) = raw else { continue };
            let value: Value = serde_json::from_str(raw.get())
                .map_err(|error| anyhow!("{field} is invalid JSON: {error}"))?;
            ensure!(value.is_object(), "{field} must be a JSON object");
        }
        Ok(())
    }
}

/// Stage reported by a typed protocol-v2 failure.
#[derive(Clone, Copy, Debug, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum FailureStageV2 {
    /// Envelope or wire-version failure.
    Protocol,
    /// Side-effect-free static validation failure.
    Validation,
    /// Dataset, endpoint, evaluator, or transport preparation failure.
    Preparation,
    /// Workload execution failure.
    Execution,
    /// Native report finalization or persistence failure.
    Reporting,
}

/// One stable, typed validation diagnostic.
#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct DiagnosticV2 {
    /// Stable machine-readable error code.
    pub code: String,
    /// Redacted human-readable explanation.
    pub message: String,
    /// Optional JSON-pointer-like authored field path.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,
}

/// Side-effect-free validation completeness state.
#[derive(Clone, Copy, Debug, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ValidationCompletenessV2 {
    /// Every check possible without external IO passed, but listed checks need
    /// execution preparation.
    Static,
    /// Dataset/profile references and every other deferred rule were checked.
    Complete,
}

/// One check intentionally deferred until networkful or filesystem-backed
/// execution preparation.
#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct DeferredCheckV2 {
    /// Stable machine-readable check identifier.
    pub code: String,
    /// JSON-pointer-like authored field path.
    pub path: String,
    /// Why static validation cannot complete this check.
    pub reason: String,
}

/// Exactly-one-line response for a protocol-v2 `validate` operation.
#[derive(Debug, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RunValidationV2 {
    /// Wire protocol version.
    pub protocol_version: u32,
    /// Stable response discriminator.
    pub event: &'static str,
    /// Decoded run identity when available.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub benchmark_id: Option<String>,
    /// Whether every static validation rule passed.
    pub success: bool,
    /// Whether preparation-time checks remain.
    pub completeness: ValidationCompletenessV2,
    /// Deferred checks in deterministic order.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub deferred_checks: Vec<DeferredCheckV2>,
    /// Typed validation diagnostics.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub errors: Vec<DiagnosticV2>,
}

/// Exactly-one-line response for a protocol-v2 `execute` operation.
#[derive(Debug, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RunTerminalV2 {
    /// Wire protocol version.
    pub protocol_version: u32,
    /// Stable response discriminator.
    pub event: &'static str,
    /// Decoded run identity when available.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub benchmark_id: Option<String>,
    /// Whether execution completed and committed its report.
    pub success: bool,
    /// Authoritative native-v2 report path on success.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub report_path: Option<PathBuf>,
    /// Stable failed stage.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stage: Option<FailureStageV2>,
    /// Typed failure diagnostics.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub errors: Vec<DiagnosticV2>,
    /// Non-authoritative diagnostic evidence emitted for failed executions.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub diagnostic_artifacts: Vec<RunDiagnosticArtifactV2>,
    /// Additive transport/workload run metadata returned before Python opens the report.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub run_metadata: BTreeMap<String, String>,
}

/// One non-authoritative diagnostic artifact returned by a failed execution.
///
/// The relative path is resolved below the run's authored artifact target. It
/// must never name `native-v2.json`: failed executions do not expose a report.
#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct RunDiagnosticArtifactV2 {
    /// Stable lowercase-snake-case artifact kind.
    pub kind: String,
    /// Relative path below the run artifact target.
    pub relative_path: PathBuf,
    /// Tagged cryptographic digest of the exact durable artifact bytes.
    pub content_hash: String,
}

#[cfg(any())]
mod tests {
    use super::*;

    fn request() -> Value {
        serde_json::json!({
            "protocol_version": 2,
            "operation": "validate",
            "run": {
                "benchmark_id": "run-1",
                "artifact_dir": "/tmp/not-created",
                "cfg": {
                    "models": {"items": [{"name": "model"}]},
                    "endpoint": {
                        "type": "future_endpoint",
                        "extension_field": {"kept": true}
                    },
                    "datasets": [{"type": "synthetic", "entries": 1}],
                    "phases": [{"type": "concurrency", "concurrency": 1}],
                    "transport": {"type": "future_transport"},
                    "runtime": {"workers": 1}
                }
            }
        })
    }

    #[test]
    fn authored_envelope_preserves_factory_owned_objects() {
        let decoded: EnvelopeV2 = serde_json::from_value(request()).unwrap();
        decoded.validate_outer().unwrap();
        assert_eq!(decoded.run.transport.id.as_str(), "future_transport");
        assert_eq!(decoded.run.transport.config.get(), r#"{"node":7}"#);
        assert_eq!(decoded.run.workload.id.as_str(), "future_workload");
        let identities = decoded.run.endpoints.identities().unwrap();
        assert_eq!(identities[0].profile_id, "default");
        assert_eq!(identities[0].endpoint_id.as_str(), "future_endpoint");
        assert!(
            decoded.run.endpoints.profiles[0]
                .get()
                .contains("extension_field")
        );
    }

    #[test]
    fn outer_contract_rejects_unknown_fields() {
        let mut value = request();
        value["run"]["unexpected"] = serde_json::json!(true);
        let error = serde_json::from_value::<EnvelopeV2>(value)
            .err()
            .expect("unknown outer field must fail")
            .to_string();
        assert!(error.contains("unknown field `unexpected`"), "{error}");
    }

    #[test]
    fn component_ids_are_open_but_wire_safe() {
        for valid in ["http", "acme_zmq4", "x"] {
            assert_eq!(valid.parse::<ComponentId>().unwrap().as_str(), valid);
        }
        for invalid in ["", " Online_http", "Online", "a-b", "a.b", "a/b"] {
            assert!(invalid.parse::<ComponentId>().is_err(), "{invalid:?}");
        }
    }

    #[test]
    fn duplicate_endpoint_profile_ids_fail_static_validation() {
        let mut value = request();
        let duplicate = value["run"]["resources"]["endpoints"]["profiles"][0].clone();
        value["run"]["resources"]["endpoints"]["profiles"]
            .as_array_mut()
            .unwrap()
            .push(duplicate);
        let decoded: EnvelopeV2 = serde_json::from_value(value).unwrap();
        let error = decoded.validate_outer().unwrap_err().to_string();
        assert!(error.contains("duplicate endpoint profile ID"), "{error}");
    }

    #[test]
    fn factory_owned_config_must_still_be_an_object() {
        let mut value = request();
        value["run"]["transport"]["config"] = serde_json::json!(null);
        let decoded: EnvelopeV2 = serde_json::from_value(value).unwrap();
        let error = decoded.validate_outer().unwrap_err().to_string();
        assert!(
            error.contains("transport.config must be a JSON object"),
            "{error}"
        );
    }

    #[test]
    fn artifact_paths_reject_dot_components_and_alias_collisions() {
        let mut value = request();
        value["run"]["resources"]["artifacts"] = serde_json::json!({
            "records_path": "./records.jsonl"
        });
        let decoded: EnvelopeV2 = serde_json::from_value(value).unwrap();
        let error = decoded.validate_outer().unwrap_err().to_string();
        assert!(error.contains("normal relative path components"), "{error}");
    }

    #[test]
    fn empty_resource_block_is_intentional_and_flat_fields_fail() {
        let mut value = request();
        value["run"]["resources"] = serde_json::json!({});
        let decoded: EnvelopeV2 = serde_json::from_value(value).unwrap();
        decoded.validate_outer().unwrap();
        for resource in [
            RunResourceV2::Models,
            RunResourceV2::Endpoints,
            RunResourceV2::Metrics,
            RunResourceV2::Artifacts,
            RunResourceV2::Sidecars,
        ] {
            assert!(!decoded.run.resource_is_present(resource));
        }

        let mut flat = request();
        flat["run"]["models"] = flat["run"]["resources"]["models"].take();
        let error = serde_json::from_value::<EnvelopeV2>(flat)
            .err()
            .unwrap()
            .to_string();
        assert!(error.contains("unknown field `models`"), "{error}");
    }
}

#[cfg(test)]
mod dispatch_mode_tests {
    use super::*;

    fn minimal_wire(runtime: Value) -> BenchmarkRunWireV2 {
        serde_json::from_value(serde_json::json!({
            "benchmark_id": "run-1",
            "artifact_dir": "/tmp/not-created",
            "cfg": {
                "models": {"items": [{"name": "model"}]},
                "endpoint": {"type": "future_endpoint"},
                "datasets": [{"type": "synthetic", "entries": 1}],
                "phases": [{"type": "concurrency", "concurrency": 1}],
                "transport": {"type": "future_transport"},
                "runtime": runtime,
            }
        }))
        .unwrap()
    }

    #[test]
    fn wire_contract_pins_into_authored_component_selection() {
        // Step-1 (config-model-unification) regression net: pin the observable
        // shape `into_authored` projects from a representative CLI-serialized wire
        // run, so the planned move to a shared typed BenchmarkRun stays byte-behavior
        // identical. Asserts the component-selection contract: transport id
        // (passthrough of `type`), workload id (derived from dataset), and the
        // cellular-aware dispatch default.
        fn project(dataset_type: &str, transport_type: &str, runtime: serde_json::Value) -> AuthoredRunSpecV2 {
            let wire: BenchmarkRunWireV2 = serde_json::from_value(serde_json::json!({
                "benchmark_id": "run-1",
                "artifact_dir": "/tmp/not-created",
                "cfg": {
                    "models": {"items": [{"name": "model"}]},
                    "endpoint": {"type": "chat", "urls": ["http://127.0.0.1:8000"]},
                    "datasets": [{"type": dataset_type, "entries": 1}],
                    "phases": [{"name": "profiling", "type": "concurrency", "concurrency": 1}],
                    "transport": {"type": transport_type},
                    "runtime": runtime,
                }
            }))
            .unwrap();
            wire.into_authored().unwrap()
        }

        // Synthetic dataset + http transport, single-process → scheduled/http/Global.
        let a = project("synthetic", "http", serde_json::json!({"workers": 1}));
        assert_eq!(a.workload.id.as_str(), "scheduled");
        assert_eq!(a.transport.id.as_str(), "http");
        assert_eq!(a.dispatch, DispatchMode::Global);

        // Graph dataset → graph workload; grpc transport id passes through.
        let b = project("dag_jsonl", "grpc", serde_json::json!({"workers": 4}));
        assert_eq!(b.workload.id.as_str(), "graph");
        assert_eq!(b.transport.id.as_str(), "grpc");
        assert_eq!(b.dispatch, DispatchMode::Global);

        // Cellular (cells>1) with absent dispatch → Sharded default.
        let c = project("synthetic", "http", serde_json::json!({"workers": 4, "cells": 4}));
        assert_eq!(c.dispatch, DispatchMode::Sharded);
    }

    #[test]
    fn scheduled_projection_omits_graph_only_weka_semantics() {
        // `weka_semantics` exists only on the graph workload's config DTO; the
        // scheduled DTO is strict (`deny_unknown_fields`). Emitting the key into
        // a scheduled config — even as `null` — fails decode with "unknown field
        // `weka_semantics`". The projection must attach it only for graph runs.
        fn project(dataset_type: &str) -> (String, serde_json::Value) {
            let wire: BenchmarkRunWireV2 = serde_json::from_value(serde_json::json!({
                "benchmark_id": "run-1",
                "artifact_dir": "/tmp/not-created",
                "cfg": {
                    "models": {"items": [{"name": "model"}]},
                    "endpoint": {"type": "future_endpoint"},
                    "datasets": [{"type": dataset_type, "entries": 1}],
                    "phases": [{"name": "profiling", "type": "concurrency", "concurrency": 1}],
                    "transport": {"type": "future_transport"},
                    "runtime": {"workers": 1},
                    "weka_semantics": "legacy",
                }
            }))
            .unwrap();
            let authored = wire.into_authored().unwrap();
            let config: serde_json::Value =
                serde_json::from_str(authored.workload.config.get()).unwrap();
            (authored.workload.id.as_str().to_owned(), config)
        }

        // Synthetic → scheduled workload: the key must be absent entirely.
        let (id, scheduled_config) = project("synthetic");
        assert_eq!(id, "scheduled");
        assert!(
            scheduled_config.get("weka_semantics").is_none(),
            "scheduled config must not carry graph-only weka_semantics: {scheduled_config}"
        );

        // A graph dataset → graph workload: the key still round-trips (unchanged).
        let (id, graph_config) = project("dag_jsonl");
        assert_eq!(id, "graph");
        assert_eq!(
            graph_config.get("weka_semantics").and_then(|v| v.as_str()),
            Some("legacy"),
            "graph config must still carry weka_semantics: {graph_config}"
        );
    }

    #[test]
    fn runtime_dispatch_defaults_to_global_when_absent() {        let runtime = serde_json::json!({"workers": 4, "cells": 1});
        assert_eq!(parse_dispatch_mode(&runtime).unwrap(), DispatchMode::Global);

        let authored = minimal_wire(runtime).into_authored().unwrap();
        assert_eq!(authored.dispatch, DispatchMode::Global);
    }

    #[test]
    fn runtime_dispatch_defaults_to_sharded_for_cellular() {
        // Absent dispatch + cells > 1 defaults to Sharded (cellular already forfeits the
        // byte-exact single-process guarantee, so Global's shared gate is pure overhead).
        let runtime = serde_json::json!({"workers": 4, "cells": 4});
        assert_eq!(
            parse_dispatch_mode(&runtime).unwrap(),
            DispatchMode::Sharded
        );
        let authored = minimal_wire(runtime).into_authored().unwrap();
        assert_eq!(authored.dispatch, DispatchMode::Sharded);
    }

    #[test]
    fn runtime_explicit_dispatch_wins_over_cellular_default() {
        // An explicit dispatch always wins, even in cellular mode.
        let runtime = serde_json::json!({"workers": 4, "cells": 4, "dispatch": "global"});
        assert_eq!(parse_dispatch_mode(&runtime).unwrap(), DispatchMode::Global);

        // And absent cells (treated as single-process) keeps the Global default.
        let runtime = serde_json::json!({"workers": 4});
        assert_eq!(parse_dispatch_mode(&runtime).unwrap(), DispatchMode::Global);
    }

    #[test]
    fn runtime_dispatch_rejects_unknown_variant() {
        let runtime = serde_json::json!({"workers": 1, "cells": 1, "dispatch": "bogus"});
        assert!(parse_dispatch_mode(&runtime).is_err());

        let error = match minimal_wire(runtime).into_authored() {
            Ok(_) => panic!("expected an unknown-dispatch-variant error"),
            Err(error) => error.to_string(),
        };
        assert!(error.contains("run.cfg.runtime.dispatch"), "{error}");
    }

    #[test]
    fn runtime_dispatch_accepts_every_kebab_case_variant() {
        for (wire_value, expected) in [
            ("sharded", DispatchMode::Sharded),
            ("global", DispatchMode::Global),
            ("global-hop", DispatchMode::GlobalHop),
        ] {
            let runtime = serde_json::json!({"workers": 1, "cells": 1, "dispatch": wire_value});
            assert_eq!(
                parse_dispatch_mode(&runtime).unwrap(),
                expected,
                "{wire_value}"
            );
            let authored = minimal_wire(runtime).into_authored().unwrap();
            assert_eq!(authored.dispatch, expected, "{wire_value}");
        }
    }
}
