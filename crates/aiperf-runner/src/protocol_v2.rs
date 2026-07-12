// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Strict authored protocol-v2 request and response types.
//!
//! Python owns structural Config-v2 expansion and serializes one authored run.
//! The selected runner owns component discovery, strict factory-specific config
//! decoding, preparation, execution, and reporting. Factory-owned objects stay
//! as [`RawValue`] until their registered implementation decodes them; this is
//! what keeps backend and workload identities open without weakening the outer
//! process contract.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::path::Component;
use std::path::PathBuf;
use std::str::FromStr;

use anyhow::{Result, anyhow, ensure};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::{Map, Value, value::RawValue};

use crate::protocol::{MetricsSpec, ModelSelectionStrategy, ModelsSpec, VariationSpec};

/// Authored runner protocol version.
pub const RUNNER_PROTOCOL_V2: u32 = 2;

/// Open identifier resolved through a frozen runner registry.
///
/// IDs deliberately use a small wire-safe grammar so they can be used as
/// deterministic registry keys, report values, and extension namespaces.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RunnerComponentId(String);

impl RunnerComponentId {
    /// Return the normalized identifier.
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Consume the wrapper and return its owned identifier.
    pub fn into_string(self) -> String {
        self.0
    }
}

impl fmt::Display for RunnerComponentId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl FromStr for RunnerComponentId {
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

impl Serialize for RunnerComponentId {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for RunnerComponentId {
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
pub enum RunnerOperationV2 {
    /// Perform side-effect-free structural and static semantic validation.
    Validate,
    /// Repeat validation, prepare the run, execute it, and commit its report.
    Execute,
}

/// One strict protocol-v2 stdin envelope.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RunnerEnvelopeV2 {
    /// Wire protocol discriminator; it must equal [`RUNNER_PROTOCOL_V2`].
    pub protocol_version: u32,
    /// Requested process operation.
    pub operation: RunnerOperationV2,
    /// BLAKE3 identity of the exact executable selected by Python.
    pub expected_distribution_id: String,
    /// Authored, not Python-resolved, single-run input.
    pub run: AuthoredRunSpecV2,
}

impl RunnerEnvelopeV2 {
    /// Validate invariants owned by the versioned outer protocol.
    ///
    /// Component-specific config is intentionally not inspected here. Frozen
    /// factories own that strict decode during registry validation.
    pub fn validate_outer(&self) -> Result<()> {
        ensure!(
            self.protocol_version == RUNNER_PROTOCOL_V2,
            "runner protocol {} is unsupported; expected {RUNNER_PROTOCOL_V2}",
            self.protocol_version
        );
        ensure!(
            !self.expected_distribution_id.is_empty()
                && self.expected_distribution_id.trim() == self.expected_distribution_id,
            "expected_distribution_id must be non-empty and contain no surrounding whitespace"
        );
        let digest = self
            .expected_distribution_id
            .strip_prefix("blake3:")
            .ok_or_else(|| anyhow!("expected_distribution_id must use the blake3: prefix"))?;
        ensure!(
            digest.len() == 64
                && digest
                    .bytes()
                    .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase()),
            "expected_distribution_id must contain exactly 64 lowercase hexadecimal digits"
        );
        self.run.validate_outer()
    }
}

/// Authored identity and runner-owned execution inputs for one run.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AuthoredRunSpecV2 {
    /// Stable identity projected from the outer orchestrator.
    pub identity: RunIdentitySpecV2,
    /// Exclusive artifact target selected but not yet created by Python.
    pub artifact_target: PathBuf,
    /// Model-selection policy.
    pub models: ModelsSpec,
    /// Raw endpoint profiles decoded by registered endpoint factories.
    pub endpoints: EndpointProfilesSpecV2,
    /// Open backend selection.
    pub backend: NamedRunnerComponentSpecV2,
    /// Open workload selection.
    pub workload: NamedRunnerComponentSpecV2,
    /// Native metrics policy shared by workload implementations.
    #[serde(default)]
    pub metrics: MetricsSpec,
    /// Runner-owned artifact policy.
    #[serde(default)]
    pub artifacts: ArtifactSpecV2,
    /// Optional supervised sidecars, retained raw until their native factory
    /// performs its strict decode.
    #[serde(default)]
    pub sidecars: SidecarSpecV2,
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
        ensure!(
            !self.models.items.is_empty(),
            "at least one model is required"
        );
        self.endpoints.validate_outer()?;
        self.backend.validate_outer("backend")?;
        self.workload.validate_outer("workload")?;
        validate_models(&self.models)?;
        validate_metrics(&self.metrics)?;
        self.artifacts.validate_outer()?;
        self.sidecars.validate_outer()?;
        Ok(())
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
    pub id: RunnerComponentId,
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

/// Authored endpoint profiles shared by every backend/workload pair.
#[derive(Debug, Deserialize)]
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
    pub endpoint_id: RunnerComponentId,
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
    /// Raw request/response records path relative to the artifact target.
    #[serde(default)]
    pub raw_path: Option<PathBuf>,
    /// Aggregated response-text output path relative to the artifact target.
    #[serde(default)]
    pub outputs_path: Option<PathBuf>,
    /// Include transport trace details in records.
    #[serde(default)]
    pub trace: bool,
    /// Authored user files materialized by the runner after validation.
    #[serde(default)]
    pub user_files: Vec<UserFileSpecV2>,
}

impl ArtifactSpecV2 {
    fn validate_outer(&self) -> Result<()> {
        let mut paths = BTreeSet::new();
        for (field, path) in [
            ("artifacts.records_path", self.records_path.as_ref()),
            ("artifacts.raw_path", self.raw_path.as_ref()),
            ("artifacts.outputs_path", self.outputs_path.as_ref()),
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
            if file.format == UserFileFormatV2::Text {
                ensure!(
                    file.content.is_string(),
                    "artifacts.user_files[{index}].content must be a string for text format"
                );
            }
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
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UserFileSpecV2 {
    /// POSIX-style relative output path.
    pub path: String,
    /// Selected serialization format.
    pub format: UserFileFormatV2,
    /// Authored structured or text content.
    pub content: Value,
}

/// Supported runner-side user-file encodings.
#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum UserFileFormatV2 {
    /// Pretty JSON.
    Json,
    /// YAML rendered by the presentation worker.
    Yaml,
    /// UTF-8 text.
    Text,
}

/// Optional ancillary process/collector intent.
#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SidecarSpecV2 {
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
    fn validate_outer(&self) -> Result<()> {
        for (field, raw) in [
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
pub enum RunnerFailureStageV2 {
    /// Envelope or wire-version failure.
    Protocol,
    /// Side-effect-free static validation failure.
    Validation,
    /// Dataset, endpoint, evaluator, or backend preparation failure.
    Preparation,
    /// Workload execution failure.
    Execution,
    /// Native report finalization or persistence failure.
    Reporting,
}

/// One stable, typed validation diagnostic.
#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct RunnerDiagnosticV2 {
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
    /// Exact executing-image identity.
    pub distribution_id: String,
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
    pub errors: Vec<RunnerDiagnosticV2>,
}

/// Exactly-one-line response for a protocol-v2 `execute` operation.
#[derive(Debug, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RunTerminalV2 {
    /// Wire protocol version.
    pub protocol_version: u32,
    /// Stable response discriminator.
    pub event: &'static str,
    /// Exact executing-image identity.
    pub distribution_id: String,
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
    pub stage: Option<RunnerFailureStageV2>,
    /// Typed failure diagnostics.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub errors: Vec<RunnerDiagnosticV2>,
    /// Additive backend/workload provenance returned before Python opens the report.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub provenance: BTreeMap<String, String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request() -> Value {
        serde_json::json!({
            "protocol_version": 2,
            "operation": "validate",
            "expected_distribution_id": format!("blake3:{}", "a".repeat(64)),
            "run": {
                "identity": {"benchmark_id": "run-1"},
                "artifact_target": "/tmp/not-created",
                "models": {"items": [{"name": "model"}]},
                "endpoints": {"profiles": [{
                    "id": "default",
                    "type": "future_endpoint",
                    "extension_field": {"kept": true}
                }]},
                "backend": {"type": "future_backend", "config": {"node": 7}},
                "workload": {"type": "future_workload", "config": {"mode": "x"}},
                "metrics": {},
                "artifacts": {},
                "sidecars": {}
            }
        })
    }

    #[test]
    fn authored_envelope_preserves_factory_owned_objects() {
        let decoded: RunnerEnvelopeV2 = serde_json::from_value(request()).unwrap();
        decoded.validate_outer().unwrap();
        assert_eq!(decoded.run.backend.id.as_str(), "future_backend");
        assert_eq!(decoded.run.backend.config.get(), r#"{"node":7}"#);
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
        let error = serde_json::from_value::<RunnerEnvelopeV2>(value)
            .err()
            .expect("unknown outer field must fail")
            .to_string();
        assert!(error.contains("unknown field `unexpected`"), "{error}");
    }

    #[test]
    fn component_ids_are_open_but_wire_safe() {
        for valid in ["online_http", "acme_zmq4", "x"] {
            assert_eq!(valid.parse::<RunnerComponentId>().unwrap().as_str(), valid);
        }
        for invalid in ["", " Online_http", "Online", "a-b", "a.b", "a/b"] {
            assert!(invalid.parse::<RunnerComponentId>().is_err(), "{invalid:?}");
        }
    }

    #[test]
    fn duplicate_endpoint_profile_ids_fail_static_validation() {
        let mut value = request();
        let duplicate = value["run"]["endpoints"]["profiles"][0].clone();
        value["run"]["endpoints"]["profiles"]
            .as_array_mut()
            .unwrap()
            .push(duplicate);
        let decoded: RunnerEnvelopeV2 = serde_json::from_value(value).unwrap();
        let error = decoded.validate_outer().unwrap_err().to_string();
        assert!(error.contains("duplicate endpoint profile ID"), "{error}");
    }

    #[test]
    fn factory_owned_config_must_still_be_an_object() {
        let mut value = request();
        value["run"]["backend"]["config"] = serde_json::json!(null);
        let decoded: RunnerEnvelopeV2 = serde_json::from_value(value).unwrap();
        let error = decoded.validate_outer().unwrap_err().to_string();
        assert!(
            error.contains("backend.config must be a JSON object"),
            "{error}"
        );
    }

    #[test]
    fn artifact_paths_reject_dot_components_and_alias_collisions() {
        let mut value = request();
        value["run"]["artifacts"] = serde_json::json!({
            "records_path": "./records.jsonl"
        });
        let decoded: RunnerEnvelopeV2 = serde_json::from_value(value).unwrap();
        let error = decoded.validate_outer().unwrap_err().to_string();
        assert!(error.contains("normal relative path components"), "{error}");
    }
}
