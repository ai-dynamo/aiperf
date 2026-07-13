// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Protocol-v2 telemetry-watch workload composition.
//!
//! Validation freezes source factories before any spool, credential, or
//! transport preparation. Collection and source-free finalization derive
//! different transport requirements from their already validated mode.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::{self, Debug, Formatter};
use std::sync::Arc;

use aiperf_telemetry_archive::CanonicalJsonValue;
use anyhow::{Result, anyhow};
use serde_json::value::RawValue;

use crate::registry::{
    ResourceRequirementsV2, RunnerClockKind, RunnerRegistryBuilder, RunnerWorkloadDescriptor,
    RunnerWorkloadFactory, ValidatedWorkloadConfig, WorkloadRequirements,
};
use crate::telemetry_archive_components::{
    ArchiveCollectionPlacement, TelemetryArchiveComponentRegistries,
    ValidatedTelemetryArchiveCollectComponents, ValidatedTelemetryArchiveSyncComponents,
};
use crate::telemetry_source::{
    ArchiveSourceFactoryRegistry, ArchiveSourceValidationContext, ValidatedArchiveSource,
};
use crate::telemetry_watch::TelemetryWatchConfigV2;

/// Built-in telemetry-watch workload capability facts.
pub static TELEMETRY_WATCH_WORKLOAD_DESCRIPTOR: RunnerWorkloadDescriptor =
    RunnerWorkloadDescriptor {
        id: "telemetry_watch",
        description: "fixed-deadline durable telemetry collection or source-free remote finalization",
        requires_semantic_responses: false,
        clock_kinds: &[RunnerClockKind::Real],
        required_transport_features: &[],
    };

/// One source whose concrete factory configuration is frozen without `Any`.
pub struct ValidatedTelemetrySourceV2 {
    /// Stable physical source identity.
    pub id: String,
    /// Frozen source-factory wire ID.
    pub source_type: String,
    /// Positive fixed-deadline cadence.
    pub interval_ns: i64,
    /// Positive per-call lifetime.
    pub request_timeout_ns: i64,
    /// Static additive attributes.
    pub attributes: BTreeMap<String, String>,
    /// Factory-produced canonical persistent source configuration.
    pub persistent_identity: CanonicalJsonValue,
    /// Factory-owned validated source ready for preparation.
    pub prepared: Box<dyn ValidatedArchiveSource>,
}

impl Debug for ValidatedTelemetrySourceV2 {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ValidatedTelemetrySourceV2")
            .field("id", &self.id)
            .field("source_type", &self.source_type)
            .field("interval_ns", &self.interval_ns)
            .field("request_timeout_ns", &self.request_timeout_ns)
            .field("attributes", &self.attributes)
            .field("persistent_identity", &self.persistent_identity)
            .field("prepared", &self.prepared)
            .finish()
    }
}

/// Fully validated mode-specific workload retained until preparation.
#[derive(Debug)]
pub enum ValidatedTelemetryWatchWorkloadV2 {
    /// Live source collection and durable local archive.
    Collect {
        /// Optional positive duration relative to execution start.
        duration_ns: Option<i64>,
        /// Positive ordered finalization budget.
        shutdown_timeout_ns: i64,
        /// Non-empty physical source set in authored order.
        sources: Vec<ValidatedTelemetrySourceV2>,
        /// Strictly selected persistent archive component graph.
        archive: ValidatedTelemetryArchiveCollectComponents,
    },
    /// Source-free receipt/publication completion.
    FinalizeRemote {
        /// Positive sync/finalization budget.
        shutdown_timeout_ns: i64,
        /// Strict source-free selectors checked against durable genesis.
        archive: ValidatedTelemetryArchiveSyncComponents,
    },
}

impl ValidatedTelemetryWatchWorkloadV2 {
    /// Whether this mode prepares source/control-plane capability.
    #[must_use]
    pub const fn collects_sources(&self) -> bool {
        matches!(self, Self::Collect { .. })
    }
}

/// Strict workload factory over one frozen source-factory universe.
#[derive(Clone, Debug)]
pub struct TelemetryWatchWorkloadFactoryV2 {
    sources: ArchiveSourceFactoryRegistry,
    archive_components: TelemetryArchiveComponentRegistries,
}

impl TelemetryWatchWorkloadFactoryV2 {
    /// Build from an explicitly composed source registry.
    #[must_use]
    pub fn new(
        sources: ArchiveSourceFactoryRegistry,
        archive_components: TelemetryArchiveComponentRegistries,
    ) -> Self {
        Self {
            sources,
            archive_components,
        }
    }

    /// Stock factory linked into the native runner distribution.
    #[must_use]
    pub fn stock() -> Self {
        Self::new(
            ArchiveSourceFactoryRegistry::stock(),
            TelemetryArchiveComponentRegistries::stock(),
        )
    }
}

impl RunnerWorkloadFactory for TelemetryWatchWorkloadFactoryV2 {
    fn descriptor(&self) -> &'static RunnerWorkloadDescriptor {
        &TELEMETRY_WATCH_WORKLOAD_DESCRIPTOR
    }

    fn validate(&self, authored: &RawValue) -> Result<Box<dyn ValidatedWorkloadConfig>> {
        let config: TelemetryWatchConfigV2 = serde_json::from_str(authored.get())
            .map_err(|error| anyhow!("telemetry_watch workload config: {error}"))?;
        config.validate_workload_static()?;
        match config {
            TelemetryWatchConfigV2::Collect {
                duration_ns,
                shutdown_timeout_ns,
                sources,
                archive,
            } => {
                let mut validated = Vec::with_capacity(sources.len());
                for source in sources {
                    let prepared = self.sources.validate(
                        source.source_type.as_str(),
                        &source.config,
                        ArchiveSourceValidationContext {
                            request_timeout_ns: source.request_timeout_ns,
                        },
                    )?;
                    let persistent_identity = prepared.persistent_identity().clone();
                    validated.push(ValidatedTelemetrySourceV2 {
                        id: source.id,
                        source_type: source.source_type.into_string(),
                        interval_ns: source.interval_ns,
                        request_timeout_ns: source.request_timeout_ns,
                        attributes: source.attributes,
                        persistent_identity,
                        prepared,
                    });
                }
                let archive = self
                    .archive_components
                    .validate_collect(*archive, ArchiveCollectionPlacement::StandalonePrimary)?;
                Ok(Box::new(ValidatedTelemetryWatchWorkloadV2::Collect {
                    duration_ns,
                    shutdown_timeout_ns,
                    sources: validated,
                    archive,
                }))
            }
            TelemetryWatchConfigV2::FinalizeRemote {
                shutdown_timeout_ns,
                archive,
            } => {
                let archive = self.archive_components.validate_sync(archive)?;
                Ok(Box::new(
                    ValidatedTelemetryWatchWorkloadV2::FinalizeRemote {
                        shutdown_timeout_ns,
                        archive,
                    },
                ))
            }
        }
    }

    fn requirements(&self, config: &dyn ValidatedWorkloadConfig) -> Result<WorkloadRequirements> {
        let config = config
            .as_any()
            .downcast_ref::<ValidatedTelemetryWatchWorkloadV2>()
            .ok_or_else(|| anyhow!("telemetry_watch factory received another workload config"))?;
        let mut transport_features = BTreeSet::new();
        if config.collects_sources() {
            transport_features.insert("control_plane_http".to_owned());
        }
        Ok(WorkloadRequirements {
            semantic_responses: false,
            clock_kinds: BTreeSet::from([RunnerClockKind::Real]),
            transport_features,
            resources: ResourceRequirementsV2::telemetry_watch(),
        })
    }
}

/// Register only the strict workload factory; a pair is advertised separately
/// after its product execution and qualification gates pass.
pub fn register_telemetry_watch_workload(builder: &mut RunnerRegistryBuilder) -> Result<()> {
    builder.register_workload(Arc::new(TelemetryWatchWorkloadFactoryV2::stock()))
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn raw(value: serde_json::Value) -> Box<RawValue> {
        RawValue::from_string(value.to_string()).unwrap()
    }

    fn archive() -> serde_json::Value {
        json!({
            "target": "file:///tmp/aiperf-watch-archive",
            "local_spool": "/tmp/aiperf-watch-spool",
            "spool_quota_bytes": 1_000_000,
            "spool_quota_files": 1000,
            "required": true,
            "writer": {"type": "parquet_archive_v1", "config": {}},
            "store_access": {"type": "local_filesystem", "config": {}},
            "rotation": {"type": "rows_bytes_age", "config": {}},
            "admission": {"type": "primary_durable", "config": {}},
            "recovery": {"type": "create_new", "config": {}},
            "archive_key": {"type": "secret_provider", "config": {"id": "archive-identity"}},
            "raw_body": {"type": "none", "config": {}}
        })
    }

    fn collect() -> Box<RawValue> {
        raw(json!({
            "mode": "collect",
            "duration_ns": 1_000_000_000_i64,
            "shutdown_timeout_ns": 1_000_000_000_i64,
            "sources": [{
                "id": "node-a",
                "type": "prometheus_http",
                "interval_ns": 100_000_000_i64,
                "request_timeout_ns": 50_000_000_i64,
                "config": {
                    "url": "http://127.0.0.1:9000/metrics",
                    "connect_timeout_ns": 10_000_000_i64,
                    "redirects": "disabled",
                    "proxy": "disabled",
                    "accepted_formats": ["prometheus_text_0_0_4"],
                    "max_compressed_bytes": 1024,
                    "max_decompressed_bytes": 4096
                }
            }],
            "archive": archive()
        }))
    }

    #[test]
    fn collect_requires_control_plane_but_source_free_finalize_does_not() {
        let factory = TelemetryWatchWorkloadFactoryV2::stock();
        let collect = factory.validate(&collect()).unwrap();
        let requirements = factory.requirements(collect.as_ref()).unwrap();
        assert_eq!(
            requirements.transport_features,
            BTreeSet::from(["control_plane_http".to_owned()])
        );
        assert_eq!(
            requirements.resources,
            ResourceRequirementsV2::telemetry_watch()
        );

        let finalize = raw(json!({
            "mode": "finalize_remote",
            "shutdown_timeout_ns": 1_000_000_000_i64,
            "archive": {
                "archive_id": "018f84a7-1f3c-7c21-8be2-7e8dbf9536b1",
                "target": "file:///tmp/aiperf-watch-archive",
                "local_spool": "/tmp/aiperf-watch-spool",
                "store_access": {"type": "local_filesystem", "config": {}},
                "recovery": {"type": "finalize_remote", "config": {}},
                "archive_key": {"type": "secret_provider", "config": {"id": "archive-identity"}}
            }
        }));
        let finalize = factory.validate(&finalize).unwrap();
        assert!(
            factory
                .requirements(finalize.as_ref())
                .unwrap()
                .transport_features
                .is_empty()
        );
    }

    #[test]
    fn unknown_source_fails_during_static_workload_validation() {
        let mut value: serde_json::Value = serde_json::from_str(collect().get()).unwrap();
        value["sources"][0]["type"] = json!("private_source");
        let error = TelemetryWatchWorkloadFactoryV2::stock()
            .validate(&raw(value))
            .unwrap_err()
            .to_string();
        assert!(error.contains("private_source"), "{error}");
    }
}
