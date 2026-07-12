// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Strict protocol-v2 telemetry-watch and attached-archive authored DTOs.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::{self, Display, Formatter};
use std::path::{Component, Path, PathBuf};
use std::str::FromStr;

use anyhow::{Result, anyhow, ensure};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::{Value, value::RawValue};
use url::Url;
use uuid::Uuid;

use crate::protocol_v2::{NamedRunnerComponentSpecV2, RunnerComponentId};

/// Normalized immutable archive target URI.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct NormalizedArchiveUri(String);

impl NormalizedArchiveUri {
    /// Returns the normalized credential-free URI.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Returns the normalized storage scheme.
    #[must_use]
    pub fn scheme(&self) -> &str {
        self.0.split_once(':').map_or("", |(scheme, _)| scheme)
    }
}

impl Display for NormalizedArchiveUri {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl FromStr for NormalizedArchiveUri {
    type Err = String;

    fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
        if value.is_empty() || value.trim() != value {
            return Err(
                "archive target URI must be non-empty without surrounding whitespace".into(),
            );
        }
        let mut parsed =
            Url::parse(value).map_err(|error| format!("invalid archive URI: {error}"))?;
        if !matches!(parsed.scheme(), "file" | "s3" | "gs" | "az") {
            return Err(format!(
                "unsupported archive target scheme {:?}; expected file, s3, gs, or az",
                parsed.scheme()
            ));
        }
        if !parsed.username().is_empty() || parsed.password().is_some() {
            return Err("archive target URI must not contain credentials".into());
        }
        if parsed.query().is_some() || parsed.fragment().is_some() {
            return Err("archive target URI must not contain query or fragment data".into());
        }
        if parsed.scheme() == "file" {
            if parsed
                .host_str()
                .is_some_and(|host| !host.is_empty() && host != "localhost")
            {
                return Err("file archive target must be local".into());
            }
            if !Path::new(parsed.path()).is_absolute() {
                return Err("file archive target must contain an absolute path".into());
            }
            if parsed.host_str() == Some("localhost") {
                parsed
                    .set_host(None)
                    .map_err(|_| "invalid local file URI")?;
            }
        } else {
            if parsed.host_str().is_none_or(str::is_empty) {
                return Err("object archive target must name a bucket/container".into());
            }
            if parsed.path().is_empty() || parsed.path() == "/" {
                return Err("object archive target must include an archive prefix".into());
            }
        }
        parsed.set_query(None);
        parsed.set_fragment(None);
        Ok(Self(parsed.to_string()))
    }
}

impl Serialize for NormalizedArchiveUri {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for NormalizedArchiveUri {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        String::deserialize(deserializer)?
            .parse()
            .map_err(serde::de::Error::custom)
    }
}

/// One physical telemetry source selected by a frozen source-factory ID.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TelemetrySourceSpecV2 {
    /// Stable run-local source identity.
    pub id: String,
    /// Open source-factory ID.
    #[serde(rename = "type")]
    pub source_type: RunnerComponentId,
    /// Positive fixed-deadline cadence interval.
    pub interval_ns: i64,
    /// Positive per-call request lifetime ceiling.
    pub request_timeout_ns: i64,
    /// Strict source-factory-owned configuration.
    pub config: Box<RawValue>,
    /// Static additive source attributes.
    #[serde(default)]
    pub attributes: BTreeMap<String, String>,
}

impl TelemetrySourceSpecV2 {
    fn validate_static(&self, index: usize) -> Result<()> {
        validate_stable_id(&self.id, &format!("sources[{index}].id"))?;
        ensure!(
            self.interval_ns > 0,
            "sources[{index}].interval_ns must be positive"
        );
        ensure!(
            self.request_timeout_ns > 0,
            "sources[{index}].request_timeout_ns must be positive"
        );
        validate_raw_object(&self.config, &format!("sources[{index}].config"))?;
        for (key, value) in &self.attributes {
            validate_stable_id(key, &format!("sources[{index}].attributes key"))?;
            ensure!(
                !key.starts_with("aiperf."),
                "sources[{index}].attributes key {key:?} uses reserved aiperf. prefix"
            );
            ensure!(
                !value.contains('\0'),
                "sources[{index}].attributes[{key:?}] contains a NUL byte"
            );
        }
        Ok(())
    }
}

/// Reusable persistent archive configuration for collect/attachment.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TelemetryArchiveSpecV2 {
    /// Immutable local/object-store archive target.
    pub target: NormalizedArchiveUri,
    /// Absolute qualified local spool directory.
    pub local_spool: PathBuf,
    /// Positive byte quota including transaction reserve.
    pub spool_quota_bytes: u64,
    /// Positive file/inode quota including transaction reserve.
    pub spool_quota_files: u64,
    /// Whether degradation fails the terminal product outcome.
    pub required: bool,
    /// Persistent layout/writer policy.
    pub writer: NamedRunnerComponentSpecV2,
    /// Invocation-only store access/credential policy.
    pub store_access: NamedRunnerComponentSpecV2,
    /// Persistent segment/partition rotation policy.
    pub rotation: NamedRunnerComponentSpecV2,
    /// Primary or attached admission policy.
    pub admission: NamedRunnerComponentSpecV2,
    /// Create-new or exact-resume recovery policy.
    pub recovery: NamedRunnerComponentSpecV2,
    /// Archive-key provider selector without inline secret material.
    pub archive_key: NamedRunnerComponentSpecV2,
    /// Ordered additive enrichment policies.
    #[serde(default)]
    pub enrichers: Vec<NamedRunnerComponentSpecV2>,
    /// Ordered optional sanitizers after the mandatory credential baseline.
    #[serde(default)]
    pub sanitizers: Vec<NamedRunnerComponentSpecV2>,
    /// Exact raw-body retention/encryption policy.
    pub raw_body: NamedRunnerComponentSpecV2,
}

impl TelemetryArchiveSpecV2 {
    /// Validates common secret-free static archive structure.
    pub fn validate_static(&self) -> Result<()> {
        validate_absolute_normal_path(&self.local_spool, "archive.local_spool")?;
        ensure!(
            self.spool_quota_bytes > 0,
            "archive.spool_quota_bytes must be positive"
        );
        ensure!(
            self.spool_quota_files > 0,
            "archive.spool_quota_files must be positive"
        );
        for (field, component) in [
            ("archive.writer", &self.writer),
            ("archive.store_access", &self.store_access),
            ("archive.rotation", &self.rotation),
            ("archive.admission", &self.admission),
            ("archive.recovery", &self.recovery),
            ("archive.archive_key", &self.archive_key),
            ("archive.raw_body", &self.raw_body),
        ] {
            component.validate_outer(field)?;
        }
        validate_component_list("archive.enrichers", &self.enrichers)?;
        validate_component_list("archive.sanitizers", &self.sanitizers)?;
        Ok(())
    }
}

/// Source-free remote-finalization selectors.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TelemetryArchiveSyncSpecV2 {
    /// Exact stored archive identity.
    pub archive_id: Uuid,
    /// Exact stored archive target.
    pub target: NormalizedArchiveUri,
    /// Absolute existing local spool.
    pub local_spool: PathBuf,
    /// Invocation-only store access config.
    pub store_access: NamedRunnerComponentSpecV2,
    /// Must select the source-free `finalize_remote` policy.
    pub recovery: NamedRunnerComponentSpecV2,
    /// Key provider selector used to verify stored identity.
    pub archive_key: NamedRunnerComponentSpecV2,
}

impl TelemetryArchiveSyncSpecV2 {
    fn validate_static(&self) -> Result<()> {
        ensure!(
            !self.archive_id.is_nil(),
            "archive.archive_id cannot be nil"
        );
        validate_absolute_normal_path(&self.local_spool, "archive.local_spool")?;
        for (field, component) in [
            ("archive.store_access", &self.store_access),
            ("archive.recovery", &self.recovery),
            ("archive.archive_key", &self.archive_key),
        ] {
            component.validate_outer(field)?;
        }
        ensure!(
            self.recovery.id.as_str() == "finalize_remote",
            "finalize_remote archive.recovery.type must be \"finalize_remote\""
        );
        Ok(())
    }
}

/// Standalone watch collect or source-free remote-finalize workload.
#[derive(Debug)]
pub enum TelemetryWatchConfigV2 {
    /// Run fixed-deadline sources and incrementally archive every outcome.
    Collect {
        /// Optional positive run duration; absent runs until signal.
        duration_ns: Option<i64>,
        /// Positive ordered shutdown/finalization budget.
        shutdown_timeout_ns: i64,
        /// Non-empty unique physical source set.
        sources: Vec<TelemetrySourceSpecV2>,
        /// Persistent archive configuration.
        archive: Box<TelemetryArchiveSpecV2>,
    },
    /// Resume receipt/synchronization work without preparing any source.
    FinalizeRemote {
        /// Positive synchronization/finalization budget.
        shutdown_timeout_ns: i64,
        /// Source-free selectors verified against durable genesis.
        archive: TelemetryArchiveSyncSpecV2,
    },
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct TelemetryWatchCollectWireV2 {
    mode: String,
    duration_ns: Option<i64>,
    shutdown_timeout_ns: i64,
    sources: Vec<TelemetrySourceSpecV2>,
    archive: TelemetryArchiveSpecV2,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct TelemetryWatchFinalizeWireV2 {
    mode: String,
    shutdown_timeout_ns: i64,
    archive: TelemetryArchiveSyncSpecV2,
}

#[derive(Deserialize)]
struct TelemetryWatchModeWireV2 {
    mode: String,
}

impl<'de> Deserialize<'de> for TelemetryWatchConfigV2 {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let raw = Box::<RawValue>::deserialize(deserializer)?;
        let header: TelemetryWatchModeWireV2 =
            serde_json::from_str(raw.get()).map_err(serde::de::Error::custom)?;
        match header.mode.as_str() {
            "collect" => {
                let wire: TelemetryWatchCollectWireV2 =
                    serde_json::from_str(raw.get()).map_err(serde::de::Error::custom)?;
                debug_assert_eq!(wire.mode, "collect");
                Ok(Self::Collect {
                    duration_ns: wire.duration_ns,
                    shutdown_timeout_ns: wire.shutdown_timeout_ns,
                    sources: wire.sources,
                    archive: Box::new(wire.archive),
                })
            }
            "finalize_remote" => {
                let wire: TelemetryWatchFinalizeWireV2 =
                    serde_json::from_str(raw.get()).map_err(serde::de::Error::custom)?;
                debug_assert_eq!(wire.mode, "finalize_remote");
                Ok(Self::FinalizeRemote {
                    shutdown_timeout_ns: wire.shutdown_timeout_ns,
                    archive: wire.archive,
                })
            }
            mode => Err(serde::de::Error::custom(format!(
                "unsupported telemetry_watch mode {mode:?}"
            ))),
        }
    }
}

impl TelemetryWatchConfigV2 {
    /// Performs side-effect-free strict workload-local validation.
    pub fn validate_workload_static(&self) -> Result<()> {
        match self {
            Self::Collect {
                duration_ns,
                shutdown_timeout_ns,
                sources,
                archive,
            } => {
                ensure!(
                    duration_ns.is_none_or(|duration| duration > 0),
                    "telemetry_watch collect duration_ns must be positive when present"
                );
                ensure!(
                    *shutdown_timeout_ns > 0,
                    "telemetry_watch shutdown_timeout_ns must be positive"
                );
                ensure!(
                    !sources.is_empty(),
                    "telemetry_watch collect requires at least one source"
                );
                let mut source_ids = BTreeSet::new();
                for (index, source) in sources.iter().enumerate() {
                    source.validate_static(index)?;
                    ensure!(
                        source_ids.insert(source.id.clone()),
                        "duplicate telemetry source ID {:?}",
                        source.id
                    );
                }
                archive.validate_static()?;
                ensure!(
                    archive.admission.id.as_str() == "primary_durable",
                    "standalone telemetry_watch requires primary_durable admission"
                );
            }
            Self::FinalizeRemote {
                shutdown_timeout_ns,
                archive,
            } => {
                ensure!(
                    *shutdown_timeout_ns > 0,
                    "telemetry_watch shutdown_timeout_ns must be positive"
                );
                archive.validate_static()?;
            }
        }
        Ok(())
    }

    /// Validates run-level path safety once the common artifact target is known.
    pub fn validate_run_paths(&self, artifact_target: &Path) -> Result<()> {
        validate_absolute_normal_path(artifact_target, "artifact_target")?;
        let local_spool = match self {
            Self::Collect { archive, .. } => &archive.local_spool,
            Self::FinalizeRemote { archive, .. } => &archive.local_spool,
        };
        ensure!(
            local_spool != artifact_target,
            "archive.local_spool must not alias artifact_target"
        );
        Ok(())
    }

    /// Performs the complete strict workload and run-path validation.
    pub fn validate_static(&self, artifact_target: &Path) -> Result<()> {
        self.validate_workload_static()?;
        self.validate_run_paths(artifact_target)
    }

    /// Whether this mode must prepare source/control-plane capabilities.
    #[must_use]
    pub const fn collects_sources(&self) -> bool {
        matches!(self, Self::Collect { .. })
    }
}

/// Attached archive selection over already prepared physical sidecar sources.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TelemetryArchiveAttachmentSpecV2 {
    /// Non-empty unique prepared physical source IDs.
    pub source_ids: Vec<String>,
    /// Shared archive configuration; source config is never repeated here.
    pub archive: TelemetryArchiveSpecV2,
}

impl TelemetryArchiveAttachmentSpecV2 {
    /// Validates attachment-local structure before resolving source references.
    pub fn validate_static(&self) -> Result<()> {
        ensure!(
            !self.source_ids.is_empty(),
            "telemetry archive attachment source_ids cannot be empty"
        );
        let mut seen = BTreeSet::new();
        for (index, source) in self.source_ids.iter().enumerate() {
            validate_stable_id(source, &format!("telemetry_archive.source_ids[{index}]"))?;
            ensure!(
                seen.insert(source),
                "duplicate telemetry archive source ID {source:?}"
            );
        }
        self.archive.validate_static()?;
        ensure!(
            self.archive.admission.id.as_str() == "attached_best_effort",
            "attached telemetry archive requires attached_best_effort admission"
        );
        Ok(())
    }
}

fn validate_component_list(field: &str, components: &[NamedRunnerComponentSpecV2]) -> Result<()> {
    for (index, component) in components.iter().enumerate() {
        component.validate_outer(&format!("{field}[{index}]"))?;
    }
    Ok(())
}

fn validate_raw_object(raw: &RawValue, field: &str) -> Result<()> {
    let value: Value =
        serde_json::from_str(raw.get()).map_err(|error| anyhow!("{field}: {error}"))?;
    ensure!(value.is_object(), "{field} must be a JSON object");
    Ok(())
}

fn validate_stable_id(value: &str, field: &str) -> Result<()> {
    ensure!(
        !value.is_empty() && value.trim() == value && !value.chars().any(char::is_control),
        "{field} must be non-empty without surrounding whitespace or control characters"
    );
    Ok(())
}

fn validate_absolute_normal_path(path: &Path, field: &str) -> Result<()> {
    ensure!(!path.as_os_str().is_empty(), "{field} cannot be empty");
    ensure!(path.is_absolute(), "{field} must be absolute");
    ensure!(
        path.components()
            .all(|component| matches!(component, Component::RootDir | Component::Normal(_))),
        "{field} must contain only root and normal path components"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use serde::de::DeserializeOwned;

    use super::*;

    fn decode<T: DeserializeOwned>(value: &Value) -> std::result::Result<T, serde_json::Error> {
        serde_json::from_str(&value.to_string())
    }

    fn collect_json() -> Value {
        serde_json::json!({
            "mode": "collect",
            "duration_ns": null,
            "shutdown_timeout_ns": 30_000_000_000_i64,
            "sources": [{
                "id": "node-a",
                "type": "prometheus_http",
                "interval_ns": 1_000_000_000_i64,
                "request_timeout_ns": 5_000_000_000_i64,
                "config": {
                    "url": "https://node-a:9100/metrics",
                    "credential_provider": "node-metrics"
                },
                "attributes": {"role": "node"}
            }],
            "archive": {
                "target": "s3://benchmarks/watch/archive-id/",
                "local_spool": "/var/tmp/aiperf/archive-id",
                "spool_quota_bytes": 107374182400_u64,
                "spool_quota_files": 100000_u64,
                "required": true,
                "writer": {"type": "parquet_archive_v1", "config": {}},
                "store_access": {"type": "object_store", "config": {"credential_provider": "archive-store"}},
                "rotation": {"type": "rows_bytes_age", "config": {}},
                "admission": {"type": "primary_durable", "config": {}},
                "recovery": {"type": "create_new", "config": {}},
                "archive_key": {"type": "secret_provider", "config": {"id": "archive-identity"}},
                "enrichers": [],
                "sanitizers": [],
                "raw_body": {"type": "none", "config": {}}
            }
        })
    }

    #[test]
    fn complete_collect_projection_is_strict_and_valid() {
        let config: TelemetryWatchConfigV2 = decode(&collect_json()).unwrap();
        config
            .validate_static(Path::new("/var/lib/aiperf/run"))
            .unwrap();
        assert!(config.collects_sources());
    }

    #[test]
    fn unknown_fields_and_duplicate_sources_fail_before_side_effects() {
        let mut unknown = collect_json();
        unknown["unexpected"] = Value::Bool(true);
        assert!(decode::<TelemetryWatchConfigV2>(&unknown).is_err());

        let mut duplicate = collect_json();
        let source = duplicate["sources"][0].clone();
        duplicate["sources"].as_array_mut().unwrap().push(source);
        let config: TelemetryWatchConfigV2 = decode(&duplicate).unwrap();
        assert!(
            config
                .validate_static(Path::new("/var/lib/aiperf/run"))
                .unwrap_err()
                .to_string()
                .contains("duplicate telemetry source ID")
        );
    }

    #[test]
    fn source_free_finalize_authors_no_persistent_writer_fields() {
        let value = serde_json::json!({
            "mode": "finalize_remote",
            "shutdown_timeout_ns": 30_000_000_000_i64,
            "archive": {
                "archive_id": "018f84a7-1f3c-7c21-8be2-7e8dbf9536b1",
                "target": "s3://benchmarks/watch/archive-id/",
                "local_spool": "/var/tmp/aiperf/archive-id",
                "store_access": {"type": "object_store", "config": {}},
                "recovery": {"type": "finalize_remote", "config": {}},
                "archive_key": {"type": "secret_provider", "config": {"id": "archive-identity"}}
            }
        });
        let config: TelemetryWatchConfigV2 = decode(&value).unwrap();
        config
            .validate_static(Path::new("/var/lib/aiperf/run"))
            .unwrap();
        assert!(!config.collects_sources());
    }

    #[test]
    fn credentialed_targets_and_relative_spools_are_rejected() {
        assert!(
            "s3://user:secret@bucket/path"
                .parse::<NormalizedArchiveUri>()
                .is_err()
        );
        let mut value = collect_json();
        value["archive"]["local_spool"] = Value::String("relative/spool".to_owned());
        let config: TelemetryWatchConfigV2 = decode(&value).unwrap();
        assert!(
            config
                .validate_static(Path::new("/var/lib/aiperf/run"))
                .is_err()
        );
    }

    #[test]
    fn attached_archive_cannot_repeat_primary_admission() {
        let archive = collect_json()["archive"].clone();
        let attachment: TelemetryArchiveAttachmentSpecV2 =
            decode(&serde_json::json!({"source_ids": ["node-a"], "archive": archive})).unwrap();
        assert!(attachment.validate_static().is_err());
    }
}
