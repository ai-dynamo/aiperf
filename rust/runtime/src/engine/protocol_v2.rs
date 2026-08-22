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

use crate::config::model::BenchmarkConfig;
use crate::config::model::transport::Transport;
use crate::config::model::workload_kind::{WorkloadKind, workload_kind};
use crate::engine::protocol::{
    DispatchMode, HopRouting, MetricsSpec, ModelSelectionStrategy, ModelsSpec, VariationSpec,
};
use crate::engine::sidecar_input::{
    AuthoredSidecarInput, CONTENT_SERVER_SIDECAR_ID, GPU_TELEMETRY_SIDECAR_ID,
    LIVE_STREAMING_SIDECAR_ID, NETWORK_LATENCY_SIDECAR_ID, SERVER_METRICS_SIDECAR_ID,
};
use crate::graph::supplement::PlannedReplayTraceInstance;

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
        // Normalize case and `-`→`_` through the shared seam (spec: every
        // discriminant matches Python's `_normalize_name`), then enforce the
        // strict runner-id grammar on the normalized form.
        let normalized = crate::extensions::normalize_ident(value);
        let mut bytes = normalized.bytes();
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
        Ok(Self(normalized))
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

/// Authoring-tagged execute-mode stdin payload.
///
/// The single-run and sweep/YAML-sweep profile paths serialize normalized authoring
/// [`Inputs`] under an `authoring` tag so the runtime performs the authoritative
/// `Inputs -> BenchmarkRun` resolution at `--execute`, rather than the CLI resolving
/// before the child launch. The sweep paths additionally carry the per-cell sweep
/// envelope (`sweep_id`/`variation`/`trial`) alongside the inputs — the runtime
/// overlays these onto the resolved run so the resolved wire the runner consumes is
/// byte-identical to what the CLI-side resolution would have produced (per-cell
/// `artifact_dir` and `random_seed` ride inside the `Inputs` themselves). Absent
/// envelope fields (the single-run case) default to a bare run.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct AuthoringWireV2 {
    /// Normalized profile inputs the runtime resolves through the shared `resolve`.
    authoring: crate::config::resolve::Inputs,
    /// Outer sweep identifier (absent for a bare single run).
    #[serde(default)]
    sweep_id: Option<String>,
    /// Sweep variation metadata (`{index,label,values}`); absent for a single run.
    #[serde(default)]
    variation: Option<Value>,
    /// Zero-based trial number (`0` for a single run).
    #[serde(default)]
    trial: u32,
}

/// Decode the execute-mode stdin payload into a resolved [`BenchmarkRunWireV2`].
///
/// Two wire shapes are accepted:
///
/// - The **authoring envelope** `{"authoring": <Inputs>}` (single runs, flag/YAML
///   sweeps, and adaptive-search / recipe sweeps alike): every CLI-side profile path
///   ships this shape, and the runtime performs the sole authoritative resolution
///   here through the shared [`crate::config::resolve::resolve`], re-projecting the
///   resolved [`crate::config::model::BenchmarkRun`] onto the wire shape. The
///   re-projection round-trips the resolved run through bytes, which preserves
///   factory-owned [`RawValue`] config that `serde_json::from_value` cannot
///   reconstruct.
/// - A **bare resolved run** (`BenchmarkRunWireV2` directly, with no `authoring`
///   tag): the pre-collapse internal `--execute` contract. No CLI sender emits this,
///   but the documented `--execute` stdin protocol and external harnesses (and the
///   e2e reachability test) still feed an already-resolved run directly. The runtime
///   accepts and returns it unchanged.
///
/// The two shapes are disjoint at the top level: the authoring envelope is tagged by
/// an `authoring` key, which the bare run's `deny_unknown_fields` contract forbids.
/// We branch on the presence of that key so a malformed authoring payload reports an
/// authoring error rather than being misread as a bare run.
pub fn decode_execute_wire(input: &[u8]) -> Result<BenchmarkRunWireV2> {
    let bytes = resolved_run_bytes(input)?;
    serde_json::from_slice(&bytes)
        .map_err(|error| anyhow!("resolved run failed the wire contract: {error}"))
}

/// Resolve an execute-mode stdin payload to the **resolved-run JSON bytes**.
///
/// This is the single authoritative `Inputs -> BenchmarkRun` resolution described on
/// [`decode_execute_wire`], exposed separately because callers that dispatch on
/// resolved facts (cellular promotion reads `cfg.runtime.cells`) must resolve before
/// they can read them — the authoring envelope carries no `cfg` at all.
///
/// Bytes, not a `Value`: the resolved run holds factory-owned
/// [`serde_json::value::RawValue`] config that survives a byte round-trip but cannot
/// be reconstructed through `serde_json::from_value`.
pub fn resolved_run_bytes(input: &[u8]) -> Result<Vec<u8>> {
    // Probe the top-level JSON for the `authoring` tag; a bare resolved run has no
    // such key (and `BenchmarkRunWireV2` rejects it via `deny_unknown_fields`).
    let has_authoring = serde_json::from_slice::<Value>(input)
        .ok()
        .and_then(|value| value.get("authoring").map(|_| ()))
        .is_some();
    if !has_authoring {
        // Pre-collapse contract: an already-resolved run projected onto the wire.
        // Decode to validate the shape, then hand back the caller's bytes unchanged.
        serde_json::from_slice::<BenchmarkRunWireV2>(input)
            .map_err(|error| anyhow!("invalid resolved run: {error}"))?;
        return Ok(input.to_vec());
    }
    let envelope: AuthoringWireV2 = serde_json::from_slice(input)
        .map_err(|error| anyhow!("invalid authoring inputs: {error}"))?;
    let mut run = crate::config::resolve::resolve(envelope.authoring)?;
    // Overlay the per-cell sweep envelope so the resolved run the runner consumes is
    // byte-identical to the CLI-side resolution (the sweep/search paths carry these;
    // a single run leaves them at their `None`/`0` defaults).
    run.sweep_id = envelope.sweep_id;
    run.trial = envelope.trial;
    if envelope.variation.is_some() {
        run.variation = envelope.variation;
    }
    serde_json::to_vec(&run).map_err(|error| anyhow!("re-serializing the resolved run: {error}"))
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

/// Decode the optional `runtime.hop_routing` worker-assignment selector.
///
/// Only meaningful for a [`DispatchMode::GlobalHop`] run with `workers > 1`,
/// where it chooses which worker thread executes each already-issued request
/// (see [`HopRouting`]). Absent (`None`) leaves the runtime on its
/// [`HopRouting::default`] (`RoundRobin`) placement; an unrecognized string is a
/// hard decode error rather than a silent fallback. The value is inert under any
/// other dispatch mode or `workers == 1`.
fn parse_hop_routing(runtime: &Value) -> Result<Option<HopRouting>> {
    match runtime.get("hop_routing") {
        None | Some(Value::Null) => Ok(None),
        Some(value) => serde_json::from_value(value.clone())
            .map(Some)
            .map_err(|error| anyhow!("run.cfg.runtime.hop_routing: {error}")),
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
    pub cfg: BenchmarkConfig,
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
    /// Controller-authored recorded-replay assignments for this cell.
    #[serde(default)]
    pub planned_replay_traces: BTreeSet<PlannedReplayTraceInstance>,
    /// Envelope-level template variables.
    #[serde(default)]
    pub variables: BTreeMap<String, Value>,
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
            self.cfg
                .datasets
                .as_ref()
                .is_some_and(|datasets| !datasets.is_empty()),
            "run.cfg.datasets must contain exactly one dataset"
        );
        Ok(())
    }

    /// Adapt canonical Config nesting to the linked preparation seam.
    pub fn into_authored(self) -> Result<AuthoredRunSpecV2> {
        self.validate_outer()?;
        // Classify the workload from the typed config before consuming it, so the
        // graph-format set stays sourced from `config::model::workload_kind`.
        let workload_kind = workload_kind(&self.cfg);
        let cfg = self.cfg;
        // The runner builds exactly one dataset; take the first and re-serialize it
        // as the dataset-factory-owned authored object.
        let dataset = cfg
            .datasets
            .and_then(|datasets| datasets.into_iter().next())
            .ok_or_else(|| anyhow!("run.cfg.datasets must contain one dataset"))?;
        let dataset = serde_json::to_value(&dataset)
            .map_err(|error| anyhow!("run.cfg.datasets[0]: {error}"))?;
        let workload_id = workload_kind.workload_id();
        let transport = transport_component(cfg.transport.as_ref())?;
        // Re-serialize the typed runtime policy so the worker-count and dispatch
        // resolution keep reading the same wire shape (`Null` when unset).
        let runtime = serde_json::to_value(&cfg.runtime)
            .map_err(|error| anyhow!("run.cfg.runtime: {error}"))?;
        let worker_count = runtime
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
        let dispatch = parse_dispatch_mode(&runtime)?;
        let hop_routing = parse_hop_routing(&runtime)?;
        let tokenizer = match cfg.tokenizer {
            Some(tokenizer) => serde_json::to_value(&tokenizer)
                .map_err(|error| anyhow!("run.cfg.tokenizer: {error}"))?,
            None => serde_json::json!({}),
        };
        let phases = serde_json::to_value(&cfg.phases)
            .map_err(|error| anyhow!("run.cfg.phases: {error}"))?;
        let failure_policy = serde_json::to_value(&cfg.failure_policy)
            .map_err(|error| anyhow!("run.cfg.failure_policy: {error}"))?;
        let mut workload_config = serde_json::json!({
            "worker_count": worker_count,
            "dataset": dataset,
            "tokenizer": tokenizer,
            "phases": phases,
            "failure_policy": failure_policy,
        });
        // `weka_semantics` selects the graph reconstruction pipeline and exists
        // only on the graph workload's config DTO. The scheduled workload DTO is
        // strict (`deny_unknown_fields`), so emitting the field there — even as
        // `null` — fails decode. Only attach it to the graph workload.
        if workload_kind == WorkloadKind::Graph {
            workload_config["weka_semantics"] = serde_json::json!(cfg.weka_semantics);
            workload_config["ignore_trace_delays"] = serde_json::json!(cfg.ignore_trace_delays);
            workload_config["recorded_agent_default"] =
                serde_json::json!(cfg.scenario.as_deref() == Some("recorded-agent-default"));
            workload_config["planned_replay_traces"] =
                serde_json::to_value(&self.planned_replay_traces)
                    .map_err(|error| anyhow!("run.planned_replay_traces: {error}"))?;
            if matches!(
                cfg.weka_semantics.as_deref(),
                Some("legacy") | Some("agentx")
            ) && let Some(cap) = cfg.system_idle_gap_cap_seconds
            {
                workload_config["system_idle_gap_cap_seconds"] = serde_json::json!(cap);
            }
        }
        let workload = NamedRunnerComponentSpecV2 {
            id: workload_id.parse().expect("built-in workload ID is valid"),
            config: raw_value(workload_config)?,
        };
        // Content-server config is folded into `cfg.sidecars` by the typed model,
        // so the sidecar bag is the single source of prepared sidecars.
        let sidecars_value = serde_json::to_value(&cfg.sidecars)
            .map_err(|error| anyhow!("run.cfg.sidecars: {error}"))?;
        let (sidecars, sidecars_present) = if sidecars_value
            .as_object()
            .is_some_and(|object| !object.is_empty())
        {
            (
                serde_json::from_value(sidecars_value)
                    .map_err(|error| anyhow!("run.cfg.sidecars: {error}"))?,
                true,
            )
        } else {
            (SidecarSpecV2::default(), false)
        };
        // Lower the authoring models to the runner spec via the typed `From`
        // (no `Value` round-trip); a missing models section is a hard error, as
        // before.
        let models = cfg
            .models
            .map(ModelsSpec::from)
            .ok_or_else(|| anyhow!("run.cfg.models must be an object"))?;
        let endpoint = serde_json::to_value(&cfg.endpoint)
            .map_err(|error| anyhow!("run.cfg.endpoint: {error}"))?;
        let additional_profiles = cfg
            .endpoint_profiles
            .into_iter()
            .collect::<BTreeMap<_, _>>();
        // Lower the authoring metrics to the runner spec via the typed
        // `TryFrom` (no untyped `Value` round-trip); default on absence or a
        // non-numeric SLO, matching the prior `from_value(...).unwrap_or_default()`.
        let metrics = cfg
            .metrics
            .and_then(|metrics| MetricsSpec::try_from(metrics).ok())
            .unwrap_or_default();
        let artifacts_spec: ArtifactSpecV2 = serde_json::from_value(
            serde_json::to_value(&cfg.artifacts)
                .map_err(|error| anyhow!("run.cfg.artifacts: {error}"))?,
        )
        .unwrap_or_default();
        let mut export_cfg: crate::export::ExportConfig = serde_json::from_value(
            serde_json::to_value(&cfg.export)
                .map_err(|error| anyhow!("run.cfg.export: {error}"))?,
        )
        .unwrap_or_default();
        // Derive the summary stem from the per-record path so
        // `--profile-export-prefix` / `artifacts.prefix` renames
        // `*_aiperf.{json,csv}` together with the jsonl.
        if let Some(path) = artifacts_spec.records_path.as_ref()
            && let Some(name) = path.file_name().and_then(|s| s.to_str())
        {
            let stem = name.strip_suffix(".jsonl").unwrap_or(name);
            if !stem.is_empty() {
                export_cfg.genai_perf.stem = stem.to_string();
                export_cfg.timeslice.stem = Some(format!("{stem}_aiperf"));
            }
        }
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
            models,
            endpoints: endpoint_profiles(endpoint, additional_profiles)?,
            transport,
            workload,
            metrics,
            artifacts: artifacts_spec,
            export: export_cfg,
            sidecars,
            dispatch,
            hop_routing,
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

/// Build the runner's transport component from the typed [`Transport`].
///
/// The id comes from `canonical_id()` rather than from a string lifted out of
/// the serialized body, so it cannot drift from the variant, and the
/// factory-owned config is the serialized body minus the `type` tag. Built-in
/// ids decode typed downstream; the plugin tail keeps the config opaque.
///
/// This is the typed consumer that replaces reading the transport back out of
/// an untyped `Value`. It is deliberately kept equivalent to routing
/// `serde_json::to_value(&transport)` through [`component_from_inline`], and
/// `transport_component_matches_inline_projection` pins that equivalence for
/// every variant.
fn transport_component(transport: Option<&Transport>) -> Result<NamedRunnerComponentSpecV2> {
    let Some(transport) = transport else {
        // Preserve the prior "transport must be an object" failure when unset.
        return component_from_inline(Value::Null, "run.cfg.transport");
    };
    let id: ComponentId = transport
        .canonical_id()
        .parse()
        .map_err(|error: String| anyhow!("run.cfg.transport.type: {error}"))?;
    let mut object = serde_json::to_value(transport)
        .map_err(|error| anyhow!("run.cfg.transport: {error}"))?
        .as_object()
        .cloned()
        .ok_or_else(|| anyhow!("run.cfg.transport must be an object"))?;
    object.remove("type");
    Ok(NamedRunnerComponentSpecV2 {
        id,
        config: raw_value(Value::Object(object))?,
    })
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
    /// Worker-assignment policy for `DispatchMode::GlobalHop` with `workers > 1`
    /// (`runtime.hop_routing`). `None` leaves the runtime on its
    /// [`HopRouting::default`] (`RoundRobin`) placement; inert under any other
    /// dispatch mode or `workers == 1`.
    pub hop_routing: Option<HopRouting>,
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
    #[serde(default)]
    hop_routing: Option<HopRouting>,
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
            hop_routing: wire.hop_routing,
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
    /// Recorded-agent tool timing output path.
    #[serde(default)]
    pub graph_tool_time_path: Option<PathBuf>,
    /// Recorded-agent trace summary output path.
    #[serde(default)]
    pub graph_trace_summary_path: Option<PathBuf>,
    /// Recorded-agent normalized replay metrics JSON output path.
    #[serde(default)]
    pub graph_replay_metrics_path: Option<PathBuf>,
    /// Optional recorded-agent normalized replay metrics CSV output path.
    #[serde(default)]
    pub graph_replay_metrics_csv_path: Option<PathBuf>,
    /// Recorded-agent replay failure output path.
    #[serde(default)]
    pub graph_replay_failures_path: Option<PathBuf>,
    /// Recorded-agent replay provenance output path.
    #[serde(default)]
    pub graph_replay_provenance_path: Option<PathBuf>,
    /// Recorded-agent backend metadata output path.
    #[serde(default)]
    pub graph_replay_backend_metadata_path: Option<PathBuf>,
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
            (
                "artifacts.graph_tool_time_path",
                self.graph_tool_time_path.as_ref(),
            ),
            (
                "artifacts.graph_trace_summary_path",
                self.graph_trace_summary_path.as_ref(),
            ),
            (
                "artifacts.graph_replay_metrics_path",
                self.graph_replay_metrics_path.as_ref(),
            ),
            (
                "artifacts.graph_replay_metrics_csv_path",
                self.graph_replay_metrics_csv_path.as_ref(),
            ),
            (
                "artifacts.graph_replay_failures_path",
                self.graph_replay_failures_path.as_ref(),
            ),
            (
                "artifacts.graph_replay_provenance_path",
                self.graph_replay_provenance_path.as_ref(),
            ),
            (
                "artifacts.graph_replay_backend_metadata_path",
                self.graph_replay_backend_metadata_path.as_ref(),
            ),
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
    fn component_ids_are_open_normalized_and_wire_safe() {
        // Canonical forms round-trip unchanged.
        for valid in ["http", "acme_zmq4", "x"] {
            assert_eq!(valid.parse::<ComponentId>().unwrap().as_str(), valid);
        }
        // Non-canonical spellings normalize (trim + lowercase + `-`→`_`) rather
        // than being rejected — the discriminant follows Python's
        // `_normalize_name` convention.
        for (input, normalized) in [
            (" Online_http", "online_http"),
            ("Online", "online"),
            ("a-b", "a_b"),
            ("DYNOSIM-OFFLINE", "dynosim_offline"),
        ] {
            assert_eq!(input.parse::<ComponentId>().unwrap().as_str(), normalized);
        }
        // Structurally invalid ids (empty, or characters no fold can rescue)
        // still fail closed.
        for invalid in ["", "   ", "a.b", "a/b", "1abc"] {
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

    /// A fully-serialized `chat` endpoint matching the CLI's typed `Endpoint`
    /// projection (every required field present), so `cfg` deserializes into the
    /// typed `BenchmarkConfig`.
    fn endpoint() -> Value {
        serde_json::json!({
            "type": "chat",
            "urls": ["http://127.0.0.1:8000"],
            "streaming": true,
            "use_legacy_max_tokens": false,
            "use_server_token_count": false,
            "timeout_seconds": 21600.0,
            "connection_reuse": "pooled",
            "ssl_verify": true,
            "connection_limit": 2500,
            "keepalive_timeout": 300.0,
            "download_video_content": false,
            "extra": {},
            "headers": {},
            "http2": false,
            "wait_for_model_timeout": 0.0,
            "wait_for_model_interval": 5.0,
            "wait_for_model_mode": "inference"
        })
    }

    /// The real serialized synthetic `Dataset` (scheduled workload).
    fn synthetic_ds() -> Value {
        serde_json::json!({
            "type": "synthetic",
            "entries": 1,
            "prompts": {"batch_size": 1, "isl": {"mean": 550.0, "stddev": 0.0}},
            "sampling": "sequential",
            "turn_delay_ratio": 1.0
        })
    }

    /// The real serialized graph `Dataset` (a `dag_jsonl` file → graph workload).
    fn graph_ds() -> Value {
        serde_json::json!({
            "type": "file",
            "format": "dag_jsonl",
            "sampling": "sequential",
            "options": {}
        })
    }

    /// Build a `cfg` object that deserializes into the typed `BenchmarkConfig`.
    fn base_cfg(dataset: Value, transport: Value, runtime: Value) -> Value {
        serde_json::json!({
            "models": {"strategy": "round_robin", "items": [{"name": "model"}]},
            "endpoint": endpoint(),
            "tokenizer": {
                "name": "model",
                "revision": "main",
                "trust_remote_code": false,
                "apply_chat_template": false
            },
            "transport": transport,
            "runtime": runtime,
            "datasets": [dataset],
            "phases": [{
                "name": "profiling",
                "type": "concurrency",
                "concurrency": 1,
                "exclude_from_results": false,
                "seamless": false,
                "requests": 1
            }],
        })
    }

    /// The real serialized `Runtime` policy (`workers_min` is a required wire key).
    fn rt(workers: u32, cells: u32) -> Value {
        serde_json::json!({"workers": workers, "workers_min": null, "cells": cells})
    }

    fn minimal_wire(runtime: Value) -> BenchmarkRunWireV2 {
        serde_json::from_value(serde_json::json!({
            "benchmark_id": "run-1",
            "artifact_dir": "/tmp/not-created",
            "cfg": base_cfg(synthetic_ds(), serde_json::json!({"type": "http"}), runtime),
        }))
        .unwrap()
    }

    #[test]
    fn execute_wire_accepts_bare_run_and_authoring_envelope_shapes() {
        // A **bare** resolved `BenchmarkRunWireV2` (no `authoring` tag) is the
        // pre-collapse internal `--execute` contract that the documented stdin
        // protocol, external harnesses, and the e2e reachability test still feed.
        // The runtime must accept it unchanged and project it to the same
        // `AuthoredRunSpecV2` as a direct decode + `into_authored`.
        let bare = serde_json::json!({
            "benchmark_id": "run-1",
            "artifact_dir": "/tmp/not-created",
            "cfg": base_cfg(synthetic_ds(), serde_json::json!({"type": "http"}), rt(1, 1)),
        });
        let bare_bytes = serde_json::to_vec(&bare).unwrap();
        let via_execute = decode_execute_wire(&bare_bytes)
            .expect("bare resolved run must be accepted")
            .into_authored()
            .expect("project bare run");
        let via_direct = serde_json::from_value::<BenchmarkRunWireV2>(bare)
            .expect("decode bare run")
            .into_authored()
            .expect("project bare run directly");
        assert_eq!(
            via_execute.identity.benchmark_id,
            via_direct.identity.benchmark_id
        );
        assert_eq!(via_execute.transport.id, via_direct.transport.id);
        assert_eq!(via_execute.workload.id, via_direct.workload.id);
        assert_eq!(via_execute.dispatch, via_direct.dispatch);

        // An `{"authoring": ...}` payload is routed to the authoring resolver, not
        // misread as a bare run: an inner-inputs failure surfaces the authoring
        // decode error (proving the envelope branch is taken). The positive
        // resolve-to-identical-run path is pinned end-to-end in `aiperf-cli`'s
        // `authoring_wire_matches_cli_resolved`.
        let envelope = serde_json::to_vec(&serde_json::json!({ "authoring": {} })).unwrap();
        let error = decode_execute_wire(&envelope)
            .err()
            .expect("empty authoring inputs must fail")
            .to_string();
        assert!(
            error.contains("invalid authoring inputs"),
            "authoring payload must route to the authoring path, got: {error}"
        );
    }

    #[test]
    fn wire_contract_pins_into_authored_component_selection() {
        // Step-1 (config-model-unification) regression net: pin the observable
        // shape `into_authored` projects from a representative CLI-serialized wire
        // run, so the move to the typed `BenchmarkConfig` stays byte-behavior
        // identical. Asserts the component-selection contract: transport id
        // (passthrough of `type`), workload id (derived from dataset), and the
        // cellular-aware dispatch default.
        fn project(dataset: Value, transport_type: &str, runtime: Value) -> AuthoredRunSpecV2 {
            let wire: BenchmarkRunWireV2 = serde_json::from_value(serde_json::json!({
                "benchmark_id": "run-1",
                "artifact_dir": "/tmp/not-created",
                "cfg": base_cfg(dataset, serde_json::json!({"type": transport_type}), runtime),
            }))
            .unwrap();
            wire.into_authored().unwrap()
        }

        // Synthetic dataset + http transport, single-process → scheduled/http/Global.
        let a = project(synthetic_ds(), "http", rt(1, 1));
        assert_eq!(a.workload.id.as_str(), "scheduled");
        assert_eq!(a.transport.id.as_str(), "http");
        assert_eq!(a.dispatch, DispatchMode::Global);

        // Graph dataset → graph workload; grpc transport id passes through.
        let b = project(graph_ds(), "grpc", rt(4, 1));
        assert_eq!(b.workload.id.as_str(), "graph");
        assert_eq!(b.transport.id.as_str(), "grpc");
        assert_eq!(b.dispatch, DispatchMode::Global);

        // Cellular (cells>1) with absent dispatch → Sharded default.
        let c = project(synthetic_ds(), "http", rt(4, 4));
        assert_eq!(c.dispatch, DispatchMode::Sharded);
    }

    #[test]
    fn scheduled_projection_omits_graph_only_weka_semantics() {
        // `weka_semantics` exists only on the graph workload's config DTO; the
        // scheduled DTO is strict (`deny_unknown_fields`). Emitting the key into
        // a scheduled config — even as `null` — fails decode with "unknown field
        // `weka_semantics`". The projection must attach it only for graph runs.
        fn project(dataset: Value) -> (String, serde_json::Value) {
            let mut cfg = base_cfg(dataset, serde_json::json!({"type": "http"}), rt(1, 1));
            cfg["weka_semantics"] = serde_json::json!("legacy");
            let wire: BenchmarkRunWireV2 = serde_json::from_value(serde_json::json!({
                "benchmark_id": "run-1",
                "artifact_dir": "/tmp/not-created",
                "cfg": cfg,
            }))
            .unwrap();
            let authored = wire.into_authored().unwrap();
            let config: serde_json::Value =
                serde_json::from_str(authored.workload.config.get()).unwrap();
            (authored.workload.id.as_str().to_owned(), config)
        }

        // Synthetic → scheduled workload: the key must be absent entirely.
        let (id, scheduled_config) = project(synthetic_ds());
        assert_eq!(id, "scheduled");
        assert!(
            scheduled_config.get("weka_semantics").is_none(),
            "scheduled config must not carry graph-only weka_semantics: {scheduled_config}"
        );

        // A graph dataset → graph workload: the key still round-trips (unchanged).
        let (id, graph_config) = project(graph_ds());
        assert_eq!(id, "graph");
        assert_eq!(
            graph_config.get("weka_semantics").and_then(|v| v.as_str()),
            Some("legacy"),
            "graph config must still carry weka_semantics: {graph_config}"
        );
    }

    #[test]
    fn ignore_trace_delays_is_graph_only_in_projection() {
        // `ignore_trace_delays` is graph-only, exactly like `weka_semantics`:
        // attaching it to the strict scheduled DTO would fail decode. The
        // projection must emit it only for graph workloads and round-trip the
        // authored value there.
        fn project(dataset: Value) -> (String, serde_json::Value) {
            let mut cfg = base_cfg(dataset, serde_json::json!({"type": "http"}), rt(1, 1));
            cfg["ignore_trace_delays"] = serde_json::json!(true);
            let wire: BenchmarkRunWireV2 = serde_json::from_value(serde_json::json!({
                "benchmark_id": "run-1",
                "artifact_dir": "/tmp/not-created",
                "cfg": cfg,
            }))
            .unwrap();
            let authored = wire.into_authored().unwrap();
            let config: serde_json::Value =
                serde_json::from_str(authored.workload.config.get()).unwrap();
            (authored.workload.id.as_str().to_owned(), config)
        }

        let (id, scheduled_config) = project(synthetic_ds());
        assert_eq!(id, "scheduled");
        assert!(
            scheduled_config.get("ignore_trace_delays").is_none(),
            "scheduled config must not carry graph-only ignore_trace_delays: {scheduled_config}"
        );

        let (id, graph_config) = project(graph_ds());
        assert_eq!(id, "graph");
        assert_eq!(
            graph_config
                .get("ignore_trace_delays")
                .and_then(|v| v.as_bool()),
            Some(true),
            "graph config must carry ignore_trace_delays: {graph_config}"
        );
    }

    #[test]
    fn runtime_dispatch_defaults_to_global_when_absent() {
        let runtime = serde_json::json!({"workers": 4, "cells": 1});
        assert_eq!(parse_dispatch_mode(&runtime).unwrap(), DispatchMode::Global);

        let authored = minimal_wire(rt(4, 1)).into_authored().unwrap();
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
        let authored = minimal_wire(rt(4, 4)).into_authored().unwrap();
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
        // The raw admission-strategy parser still rejects an unknown selector.
        let runtime = serde_json::json!({"workers": 1, "cells": 1, "dispatch": "bogus"});
        assert!(parse_dispatch_mode(&runtime).is_err());

        // With the typed wire, `runtime.dispatch` is a strict `DispatchMode`, so an
        // unknown variant now fails closed at wire-decode (before `into_authored`).
        let mut runtime = rt(1, 1);
        runtime["dispatch"] = serde_json::json!("bogus");
        let wire = serde_json::from_value::<BenchmarkRunWireV2>(serde_json::json!({
            "benchmark_id": "run-1",
            "artifact_dir": "/tmp/not-created",
            "cfg": base_cfg(synthetic_ds(), serde_json::json!({"type": "http"}), runtime),
        }));
        assert!(
            wire.is_err(),
            "unknown runtime.dispatch variant must be rejected at wire decode"
        );
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
            let mut runtime = rt(1, 1);
            runtime["dispatch"] = serde_json::json!(wire_value);
            let authored = minimal_wire(runtime).into_authored().unwrap();
            assert_eq!(authored.dispatch, expected, "{wire_value}");
        }
    }

    #[test]
    fn runtime_hop_routing_absent_is_none() {
        let runtime = serde_json::json!({"workers": 4, "cells": 1});
        assert_eq!(parse_hop_routing(&runtime).unwrap(), None);
        let authored = minimal_wire(runtime).into_authored().unwrap();
        assert_eq!(authored.hop_routing, None);
    }

    #[test]
    fn runtime_hop_routing_parses_sticky() {
        let runtime = serde_json::json!({"workers": 4, "cells": 1, "hop_routing": "sticky"});
        assert_eq!(
            parse_hop_routing(&runtime).unwrap(),
            Some(HopRouting::Sticky)
        );
        let authored = minimal_wire(runtime).into_authored().unwrap();
        assert_eq!(authored.hop_routing, Some(HopRouting::Sticky));
    }

    #[test]
    fn runtime_hop_routing_accepts_every_kebab_case_variant() {
        for (wire_value, expected) in [
            ("round-robin", HopRouting::RoundRobin),
            ("sticky", HopRouting::Sticky),
            ("least-loaded", HopRouting::LeastLoaded),
        ] {
            let runtime = serde_json::json!({"workers": 1, "cells": 1, "hop_routing": wire_value});
            assert_eq!(
                parse_hop_routing(&runtime).unwrap(),
                Some(expected),
                "{wire_value}"
            );
        }
    }

    #[test]
    fn runtime_hop_routing_rejects_unknown_variant() {
        // The raw hop-routing parser still rejects an unknown selector.
        let runtime = serde_json::json!({"workers": 1, "cells": 1, "hop_routing": "bogus"});
        assert!(parse_hop_routing(&runtime).is_err());

        // With the typed wire, `runtime.hop_routing` is a strict `HopRouting`, so an
        // unknown variant now fails closed at wire-decode (before `into_authored`).
        let mut runtime = rt(1, 1);
        runtime["hop_routing"] = serde_json::json!("bogus");
        let wire = serde_json::from_value::<BenchmarkRunWireV2>(serde_json::json!({
            "benchmark_id": "run-1",
            "artifact_dir": "/tmp/not-created",
            "cfg": base_cfg(synthetic_ds(), serde_json::json!({"type": "http"}), runtime),
        }));
        assert!(
            wire.is_err(),
            "unknown runtime.hop_routing variant must be rejected at wire decode"
        );
    }
}

#[cfg(test)]
mod transport_component_tests {
    use super::*;
    use crate::config::model::transport::{
        DryRunConfig, DynosimConfig, WebSocketTransportConfig,
    };

    /// Every `Transport` variant, so the payload-bearing arms are covered.
    ///
    /// The existing projection assertions only compare transport *ids*, which
    /// `Http`/`Grpc` satisfy trivially — they carry no payload, so an id-only
    /// check cannot detect a config body that the typed path drops or reshapes.
    fn all_variants() -> Vec<Transport> {
        vec![
            Transport::Http,
            Transport::Grpc,
            Transport::DynosimOffline(DynosimConfig::default()),
            Transport::DynosimOnline(DynosimConfig::default()),
            Transport::DryRun(DryRunConfig::default()),
            Transport::Websocket(WebSocketTransportConfig::default()),
        ]
    }

    /// Migration step 1: the typed `cfg.transport` consumer and the untyped
    /// projection it replaces must produce identical bindings.
    ///
    /// Identical means both halves of the component: the `id` the runner keys
    /// factory selection on, and the byte-exact `config` the factory decodes.
    #[test]
    fn transport_component_matches_inline_projection() {
        for transport in all_variants() {
            let typed = transport_component(Some(&transport))
                .unwrap_or_else(|error| panic!("typed component for {transport:?}: {error}"));
            let value = serde_json::to_value(&transport)
                .unwrap_or_else(|error| panic!("serialize {transport:?}: {error}"));
            let inline = component_from_inline(value, "run.cfg.transport")
                .unwrap_or_else(|error| panic!("inline component for {transport:?}: {error}"));

            assert_eq!(
                typed.id.as_str(),
                inline.id.as_str(),
                "transport id diverged for {transport:?}"
            );
            assert_eq!(
                typed.config.get(),
                inline.config.get(),
                "transport config diverged for {transport:?}"
            );
        }
    }

    /// The derived id must equal `canonical_id()`, and the config must never
    /// retain the `type` tag — a factory decoding a strict typed config would
    /// reject it as an unknown field.
    #[test]
    fn transport_component_drops_the_type_tag() {
        for transport in all_variants() {
            let component = transport_component(Some(&transport))
                .unwrap_or_else(|error| panic!("typed component for {transport:?}: {error}"));
            assert_eq!(component.id.as_str(), transport.canonical_id());
            let config: Value = serde_json::from_str(component.config.get())
                .unwrap_or_else(|error| panic!("component config is JSON: {error}"));
            assert!(
                config.get("type").is_none(),
                "`type` survived into the factory config for {transport:?}: {config}"
            );
        }
    }

    /// An unset `cfg.transport` keeps failing, and keeps failing with the
    /// message that names the field.
    #[test]
    fn absent_transport_is_rejected() {
        let error = transport_component(None)
            .expect_err("an absent transport has no component")
            .to_string();
        assert!(
            error.contains("run.cfg.transport"),
            "error does not name the field: {error}"
        );
    }
}
