// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Protocol-v2 composition for provider-neutral evaluation workloads.
//!
//! Python projects only registered provider/distribution identities, an opaque
//! provider-owned evaluation object, logical service routes, explicit resource
//! bindings, and unit concurrency. The exact runner distribution owns strict
//! provider/resource decoding and every worker launch coordinate. This module
//! deliberately never accepts an executable, module, argv, environment, URL,
//! credential, or upstream request target inside the workload object.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::Arc;

use aiperf::evaluation::host::{
    CompatibilityProxyIngress, EvaluationRoute, EvaluationRouteTable, HostExecutorRegistryBuilder,
    HostExecutorRuntime, HostOperationDescriptor, HostOperationFamily, RegisteredOperationId,
};
use aiperf::evaluation::inference::{
    DatasetEvaluationInferenceMaterializer, EvaluationInferenceMaterializer,
    PreparedEvaluationRoute, register_scheduled_inference_host_executors,
};
use aiperf::evaluation::proxy::{
    CompatibilityProxyDialect, CompatibilityProxyDialectRegistry, CompatibilityProxyRoute,
    EVALUATOR_PROXY_LOCAL_LOCATOR, EvaluatorCompatibilityProxy, LinuxProcessSubtreeAuthorizer,
    OpenAiChatCompatibilityDialect, ProxyOperationReceiver, ProxyProcessScopeAuthorizer,
    ProxyTimingRuntime, start_evaluator_compatibility_proxy,
};
use aiperf::evaluation::report::build_evaluation_report;
use aiperf::evaluation::retry::OperationCancellation;
use aiperf::evaluation::workload::{
    EvaluationAssetResolver, EvaluationHostCapabilityInventory, EvaluationReportCommit,
    EvaluationWorkload, EvaluationWorkloadLimits, SealingEvaluationArtifactFinalizer,
};
use aiperf::http::{HttpTurnExecutionBackend, PreparedHttpTurn, TransportSinkConfig};
use aiperf::multiturn::{PreparedEndpointReference, TurnToSend};
use aiperf::scheduled::{
    ScheduledRuntime, TurnDispatchOutcome, TurnDispatcher, TurnResponseObserver,
};
use aiperf_accuracy::{
    ArtifactProjectionPolicy, ArtifactSealLimits, CanonicalJson, EvaluationArtifactSealer,
    EvaluationAssetRequirement, EvaluationDistributionId, EvaluationExecutionGranularity,
    EvaluationHostBinding, EvaluationHostIdentity, EvaluationProvider, EvaluationProviderId,
    EvaluationProviderRegistry, EvaluationQueueCredits, EvaluationSchedulingMode,
    EvaluationSessionId, EvaluatorProtocolLimits, HostCapabilityId, LogicalServiceId,
    OperationPurpose, ProviderLaunchContext, ResolvedEvaluationAsset,
    STOCK_EVALUATION_OPERATION_SCHEMAS, ScopedProxyBinding, ScopedProxyGrant, ScopedProxySecret,
    SemanticOperationId, ValidatedProviderConfig, artifact_content_sha256,
};
use aiperf_clock::{Clock, RealClock, RealClockAnchor};
use aiperf_dataset::{TextTokenizer, TiktokenTokenizer};
use aiperf_endpoints::{
    EndpointDescriptor, EndpointId, EndpointKey, EndpointRegistry, PreparedEndpointTable,
    RawEndpointConfig,
};
use aiperf_metrics::{
    MetricsConfig, NativeReport, ReportEvaluationCompatibilityGrantLimits,
    ReportEvaluationCompatibilityInfo, ReportPairRunFacts, ReportRunInfo, RunOutcome,
};
use aiperf_timing::StopConfig;
use anyhow::{Context, Result, anyhow, bail, ensure};
use async_trait::async_trait;
use loadgen_core::sink::RequestObserver;
use serde::Deserialize;
use serde_json::{Value, json, value::RawValue};
use uuid::Uuid;

use crate::protocol::{
    EvaluationCapabilityInventory, EvaluationDistributionCapability,
    EvaluationHostOperationCapability, EvaluationProviderCapability,
    SupportedEvaluationCombination,
};
use crate::protocol_v2::RunnerComponentId;
use crate::readiness::{PreparedOnlineReadiness, ReadinessTransportFactory};
use crate::registry::{
    OnlineHttpBackendConfigV2, PreparedReportCommit, PreparedRunOutcome, PreparedRunnerOperation,
    ResourceRequirementV2, ResourceRequirementsV2, RunnerClockKind, RunnerPairFactory,
    RunnerRegistryBuilder, RunnerRunContext, RunnerWorkloadDescriptor, RunnerWorkloadFactory,
    ValidatedBackendConfig, ValidatedWorkloadConfig, WorkloadRequirements,
};
use crate::turn_execution::{
    HttpExecutionBackendConfig, HttpExecutionBackendFactory, HttpPreparedEndpointTableFactory,
};

/// Registered evaluator implementation and immutable worker distribution.
#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct EvaluationProviderSelectionV2 {
    /// Open evaluator-provider factory ID.
    #[serde(rename = "type")]
    pub provider_id: EvaluationProviderId,
    /// Immutable distribution ID registered by that factory.
    pub distribution: EvaluationDistributionId,
}

/// One logical evaluator service mapped to a Rust-owned endpoint profile.
#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct EvaluationRouteSpecV2 {
    /// Server-facing model identity from the run's model table.
    pub model: String,
    /// Run-local endpoint profile name.
    pub endpoint_profile: RunnerComponentId,
    /// Exact provider-plan purpose permitted on this route.
    #[serde(default = "default_evaluation_route_purpose")]
    pub purpose: OperationPurpose,
}

fn default_evaluation_route_purpose() -> OperationPurpose {
    OperationPurpose::new("primary").expect("built-in evaluation purpose is valid")
}

/// One explicitly named Rust-hosted resource implementation.
#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvaluationResourceSpecV2 {
    /// Open host-resource factory ID.
    #[serde(rename = "type")]
    pub resource_id: HostCapabilityId,
    /// Strictly decoded by the selected resource factory.
    pub config: Box<RawValue>,
}

impl fmt::Debug for EvaluationResourceSpecV2 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("EvaluationResourceSpecV2")
            .field("resource_id", &self.resource_id)
            .finish_non_exhaustive()
    }
}

/// Strict outer shape owned by the built-in `evaluation` workload factory.
#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvaluationWorkloadConfigV2 {
    /// Selected process-protocol provider and registered distribution.
    pub provider: EvaluationProviderSelectionV2,
    /// Opaque provider-authored value decoded by the selected provider factory.
    pub evaluation: Box<RawValue>,
    /// Logical service routes keyed by provider-visible `service_id`.
    pub routes: BTreeMap<LogicalServiceId, EvaluationRouteSpecV2>,
    /// Explicit host-resource bindings keyed by provider-visible resource name.
    #[serde(default)]
    pub resources: BTreeMap<RunnerComponentId, EvaluationResourceSpecV2>,
    /// Maximum concurrently started evaluation units.
    pub unit_concurrency: usize,
}

impl fmt::Debug for EvaluationWorkloadConfigV2 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("EvaluationWorkloadConfigV2")
            .field("provider", &self.provider)
            .field("routes", &self.routes.keys().collect::<Vec<_>>())
            .field("resources", &self.resources.keys().collect::<Vec<_>>())
            .field("unit_concurrency", &self.unit_concurrency)
            .finish_non_exhaustive()
    }
}

impl EvaluationWorkloadConfigV2 {
    /// Validate provider-neutral structure without IO or provider imports.
    pub fn validate_structure(&self) -> Result<()> {
        ensure!(
            self.unit_concurrency > 0,
            "evaluation unit_concurrency must be positive"
        );
        ensure!(
            !self.routes.is_empty(),
            "evaluation routes must contain at least one logical service"
        );
        let evaluation: Value = serde_json::from_str(self.evaluation.get())
            .map_err(|error| anyhow!("evaluation provider config is invalid JSON: {error}"))?;
        ensure!(
            evaluation.is_object(),
            "evaluation provider config must be a JSON object"
        );
        for (service_id, route) in &self.routes {
            ensure!(
                !route.model.is_empty() && route.model.trim() == route.model,
                "evaluation route {service_id:?}.model must be non-empty and contain no surrounding whitespace"
            );
            ensure!(
                !route.endpoint_profile.as_str().is_empty(),
                "evaluation route {service_id:?}.endpoint_profile must not be empty"
            );
        }
        for (resource_name, resource) in &self.resources {
            let config: Value = serde_json::from_str(resource.config.get()).map_err(|error| {
                anyhow!("evaluation resource {resource_name:?}.config is invalid JSON: {error}")
            })?;
            ensure!(
                config.is_object(),
                "evaluation resource {resource_name:?}.config must be a JSON object"
            );
        }
        Ok(())
    }
}

/// Strict provider-neutral workload descriptor.
pub static EVALUATION_WORKLOAD_DESCRIPTOR: RunnerWorkloadDescriptor = RunnerWorkloadDescriptor {
    id: "evaluation",
    description: "provider-neutral evaluator session over Rust-owned host effects",
    requires_semantic_responses: true,
    clock_kinds: &[RunnerClockKind::Real],
    required_backend_features: &["http"],
};

/// Provider-factory validated workload input retained through pair preparation.
#[derive(Clone)]
pub struct ValidatedEvaluationWorkloadConfigV2 {
    authored: EvaluationWorkloadConfigV2,
    provider_config: ValidatedProviderConfig,
}

impl fmt::Debug for ValidatedEvaluationWorkloadConfigV2 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ValidatedEvaluationWorkloadConfigV2")
            .field("authored", &self.authored)
            .field("provider_id", self.provider_config.provider_id())
            .field(
                "config_schema_version",
                &self.provider_config.schema_version(),
            )
            .field(
                "config_schema_sha256",
                &self.provider_config.schema_sha256(),
            )
            .field("config_sha256", &self.provider_config.config_sha256())
            .finish()
    }
}

impl ValidatedEvaluationWorkloadConfigV2 {
    /// Borrow the structural authored configuration.
    pub fn authored(&self) -> &EvaluationWorkloadConfigV2 {
        &self.authored
    }

    /// Borrow the factory-minted strict provider configuration.
    pub fn provider_config(&self) -> &ValidatedProviderConfig {
        &self.provider_config
    }
}

/// Workload factory bound to the exact frozen provider registry.
#[derive(Clone)]
pub struct EvaluationRunnerWorkloadFactory {
    providers: Arc<EvaluationProviderRegistry>,
}

impl fmt::Debug for EvaluationRunnerWorkloadFactory {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("EvaluationRunnerWorkloadFactory")
            .field("provider_count", &self.providers.len())
            .finish()
    }
}

impl EvaluationRunnerWorkloadFactory {
    /// Bind strict workload validation to one frozen provider universe.
    pub fn new(providers: Arc<EvaluationProviderRegistry>) -> Self {
        Self { providers }
    }
}

impl RunnerWorkloadFactory for EvaluationRunnerWorkloadFactory {
    fn descriptor(&self) -> &'static RunnerWorkloadDescriptor {
        &EVALUATION_WORKLOAD_DESCRIPTOR
    }

    fn validate(&self, authored: &RawValue) -> Result<Box<dyn ValidatedWorkloadConfig>> {
        let config: EvaluationWorkloadConfigV2 = serde_json::from_str(authored.get())
            .map_err(|error| anyhow!("evaluation workload config: {error}"))?;
        config.validate_structure()?;
        let provider_config = self
            .providers
            .validate_authored_bytes(
                &config.provider.provider_id,
                &config.provider.distribution,
                config.evaluation.get().as_bytes(),
            )
            .map_err(|error| anyhow!(error.to_string()))?;
        Ok(Box::new(ValidatedEvaluationWorkloadConfigV2 {
            authored: config,
            provider_config,
        }))
    }

    fn requirements(&self, config: &dyn ValidatedWorkloadConfig) -> Result<WorkloadRequirements> {
        let config = validated_evaluation_workload(config)?;
        config.authored.validate_structure()?;
        Ok(WorkloadRequirements {
            semantic_responses: true,
            clock_kinds: BTreeSet::from([RunnerClockKind::Real]),
            backend_features: BTreeSet::from(["http".to_owned()]),
            resources: ResourceRequirementsV2 {
                sidecars: ResourceRequirementV2::Forbidden,
                ..ResourceRequirementsV2::inference()
            },
        })
    }
}

/// Register the evaluation workload validator against an injected provider registry.
pub fn register_evaluation_workload(
    builder: &mut RunnerRegistryBuilder,
    providers: Arc<EvaluationProviderRegistry>,
) -> Result<()> {
    builder.register_workload(Arc::new(EvaluationRunnerWorkloadFactory::new(providers)))
}

fn validated_evaluation_workload(
    config: &dyn ValidatedWorkloadConfig,
) -> Result<&ValidatedEvaluationWorkloadConfigV2> {
    ValidatedWorkloadConfig::as_any(config)
        .downcast_ref::<ValidatedEvaluationWorkloadConfigV2>()
        .ok_or_else(|| anyhow!("evaluation workload factory received the wrong config type"))
}

/// Project capability truth from the exact frozen provider and host registries.
///
/// The caller may set `executable_pair` only after linking the real pair
/// factory and proving the selected isolation implementation is available.
/// Provider descriptors may declare more operations than this image can host;
/// supported combinations contain only the executable intersection.
pub fn build_evaluation_capability_inventory<'a>(
    providers: &EvaluationProviderRegistry,
    host_operations: impl IntoIterator<Item = &'a HostOperationDescriptor>,
    resource_ids: impl IntoIterator<Item = &'a str>,
    isolation_profile_id: &str,
    executable_pair: bool,
) -> Result<EvaluationCapabilityInventory> {
    ensure!(
        !isolation_profile_id.trim().is_empty(),
        "evaluation isolation profile ID must not be empty"
    );
    let mut host_operations = host_operations
        .into_iter()
        .map(|descriptor| {
            descriptor.validate()?;
            Ok((
                descriptor.operation_id.as_str().to_owned(),
                EvaluationHostOperationCapability {
                    id: descriptor.operation_id.as_str().to_owned(),
                    family: descriptor.family.as_str().to_owned(),
                    request_schema_sha256: descriptor.request_schema_fingerprint.clone(),
                    response_schema_sha256: descriptor.response_schema_fingerprint.clone(),
                    stream_schema_sha256: descriptor.stream_schema_fingerprint.clone(),
                    true_streaming: descriptor.true_streaming,
                    endpoint_capabilities: descriptor
                        .endpoint_capabilities
                        .iter()
                        .cloned()
                        .collect(),
                },
            ))
        })
        .collect::<Result<Vec<_>>>()?;
    host_operations.sort_by(|left, right| left.0.cmp(&right.0));
    let executable_operations = host_operations
        .iter()
        .map(|(id, capability)| (id.clone(), capability.clone()))
        .collect::<BTreeMap<_, _>>();
    let host_operation_capabilities = host_operations
        .into_iter()
        .map(|(_, capability)| capability)
        .collect::<Vec<_>>();
    let mut resources = resource_ids
        .into_iter()
        .map(str::to_owned)
        .collect::<Vec<_>>();
    resources.sort();
    resources.dedup();

    let mut provider_capabilities = Vec::with_capacity(providers.len());
    let mut supported_combinations = Vec::new();
    for descriptor in providers.descriptors() {
        descriptor
            .validate()
            .map_err(|error| anyhow!(error.to_string()))?;
        let available_distributions = providers.available_distributions(&descriptor.provider_id);
        if available_distributions.is_empty() {
            continue;
        }
        let declared_operations = descriptor
            .operations
            .iter()
            .map(|operation| operation.operation_id.as_str().to_owned())
            .collect::<Vec<_>>();
        let operations = descriptor
            .operations
            .iter()
            .filter(|operation| {
                executable_operations
                    .get(operation.operation_id.as_str())
                    .is_some_and(|host| operation_matches_host(operation, host))
            })
            .map(|operation| operation.operation_id.as_str().to_owned())
            .collect::<Vec<_>>();
        let distributions = available_distributions
            .iter()
            .map(|distribution| EvaluationDistributionCapability {
                id: distribution.distribution_id.as_str().to_owned(),
                package: distribution.package.clone(),
                package_version: distribution.package_version.clone(),
                provider_source_sha256: distribution.provider_source_sha256.clone(),
                worker_source_sha256: distribution.worker_source_sha256.clone(),
                dependency_lock_sha256: distribution.dependency_lock_sha256.clone(),
                launch_closure_sha256: distribution.launch_closure_sha256.clone(),
                oci_digest: distribution.oci_digest.clone(),
            })
            .collect::<Vec<_>>();
        provider_capabilities.push(EvaluationProviderCapability {
            id: descriptor.provider_id.as_str().to_owned(),
            display_name: descriptor.display_name.clone(),
            worker_protocol_versions: descriptor.worker_protocol_versions.clone(),
            execution_granularities: descriptor
                .execution_granularities
                .iter()
                .map(|granularity| match granularity {
                    EvaluationExecutionGranularity::Case => "case",
                    EvaluationExecutionGranularity::HostBatch => "host_batch",
                })
                .map(str::to_owned)
                .collect(),
            scheduling_modes: descriptor
                .scheduling_modes
                .iter()
                .map(|mode| match mode {
                    EvaluationSchedulingMode::Finite => "finite",
                    EvaluationSchedulingMode::RustOccurrences => "rust_occurrences",
                })
                .map(str::to_owned)
                .collect(),
            config_schema_version: descriptor.config_schema_version,
            config_schema_sha256: descriptor.config_schema_sha256.clone(),
            isolation_profile_id: isolation_profile_id.to_owned(),
            declared_operations,
            distributions,
        });
        if executable_pair && !operations.is_empty() {
            for distribution in available_distributions {
                supported_combinations.push(SupportedEvaluationCombination {
                    backend: "online_http".to_owned(),
                    workload: EVALUATION_WORKLOAD_DESCRIPTOR.id.to_owned(),
                    provider: descriptor.provider_id.as_str().to_owned(),
                    distribution: distribution.distribution_id.as_str().to_owned(),
                    operations: operations.clone(),
                    resources: resources.clone(),
                    isolation_profile_id: isolation_profile_id.to_owned(),
                });
            }
        }
    }
    Ok(EvaluationCapabilityInventory {
        providers: provider_capabilities,
        host_operations: host_operation_capabilities,
        supported_combinations,
    })
}

fn operation_matches_host(
    provider: &aiperf_accuracy::EvaluationOperationDescriptor,
    host: &EvaluationHostOperationCapability,
) -> bool {
    provider.input_schema_sha256 == host.request_schema_sha256
        && provider.output_schema_sha256 == host.response_schema_sha256
        && provider.stream_schema_sha256 == host.stream_schema_sha256
        && provider
            .endpoint_capabilities
            .iter()
            .cloned()
            .collect::<BTreeSet<_>>()
            == host
                .endpoint_capabilities
                .iter()
                .cloned()
                .collect::<BTreeSet<_>>()
}

pub(crate) fn stock_evaluation_host_operation_descriptors(
    operation_ids: &BTreeSet<String>,
) -> Result<Vec<HostOperationDescriptor>> {
    STOCK_EVALUATION_OPERATION_SCHEMAS
        .iter()
        .filter(|schema| operation_ids.contains(schema.operation_id))
        .map(|schema| {
            Ok(HostOperationDescriptor {
                operation_id: RegisteredOperationId::new(schema.operation_id)?,
                family: HostOperationFamily::new("inference")?,
                request_schema_fingerprint: schema.request_schema_sha256.to_string(),
                response_schema_fingerprint: schema.response_schema_sha256.to_string(),
                stream_schema_fingerprint: schema
                    .true_streaming
                    .then(|| schema.canonical_stream_schema_sha256.to_string()),
                true_streaming: schema.true_streaming,
                max_request_bytes: 8 * 1024 * 1024,
                max_response_bytes: 8 * 1024 * 1024,
                endpoint_capabilities: BTreeSet::from([schema.endpoint_capability.to_string()]),
            })
        })
        .collect()
}

/// Immutable provider-asset catalog selected by the runner distribution.
///
/// Production implementations bind only files already materialized inside the
/// attested evaluator rootfs. Future remote stores implement this seam without
/// changing provider planning or the evaluation workload.
pub trait EvaluationAssetCatalog: fmt::Debug + Send + Sync {
    /// Resolve one exact provider requirement to its contained worker path.
    fn resolve(&self, requirement: &EvaluationAssetRequirement) -> Result<ResolvedEvaluationAsset>;
}

const GSM8K_CANARY_ASSET_ID: &str = "openai_gsm8k_main_test_canary";
const GSM8K_CANARY_REVISION: &str =
    "openai/gsm8k@740312add88f781978c0658806c59bc2815b9866:main:test:first5";
const GSM8K_CANARY_SHA256: &str =
    "fc9b5c03206d193c0013baf2d6344a133fe0096a2b47cd1eafdcee297dfd398a";
const GSM8K_CANARY_MEDIA_TYPE: &str = "application/x-ndjson";
const GSM8K_CANARY_CONTAINED_PATH: &str = "/assets/gsm8k_canary.jsonl";
const GSM8K_CANARY_BYTES: &[u8] =
    include_bytes!("../../../src/aiperf/accuracy/evaluation/manifests/assets/gsm8k_canary.jsonl");

/// Stock immutable task-package catalog embedded in the exact runner image.
///
/// The relocatable distribution materializer copies the same bytes into the
/// attested rootfs at [`GSM8K_CANARY_CONTAINED_PATH`]. This catalog exposes
/// only that contained identity; it never gives the provider a host path.
#[derive(Clone, Copy, Debug, Default)]
pub struct StockEvaluationAssetCatalog;

impl EvaluationAssetCatalog for StockEvaluationAssetCatalog {
    fn resolve(&self, requirement: &EvaluationAssetRequirement) -> Result<ResolvedEvaluationAsset> {
        ensure!(
            artifact_content_sha256(GSM8K_CANARY_BYTES) == GSM8K_CANARY_SHA256,
            "embedded GSM8K evaluator asset digest drifted"
        );
        ensure!(
            requirement.asset_id == GSM8K_CANARY_ASSET_ID
                && requirement.source_kind == "task_package"
                && requirement.immutable_revision == GSM8K_CANARY_REVISION
                && requirement.content_sha256 == GSM8K_CANARY_SHA256
                && requirement.media_type == GSM8K_CANARY_MEDIA_TYPE,
            "evaluator requested an unregistered or drifted immutable asset"
        );
        Ok(ResolvedEvaluationAsset {
            asset_id: requirement.asset_id.clone(),
            contained_path: GSM8K_CANARY_CONTAINED_PATH.to_string(),
            content_sha256: requirement.content_sha256.clone(),
            immutable_revision: requirement.immutable_revision.clone(),
            media_type: requirement.media_type.clone(),
        })
    }
}

struct CatalogEvaluationAssetResolver {
    catalog: Arc<dyn EvaluationAssetCatalog>,
}

#[async_trait(?Send)]
impl EvaluationAssetResolver for CatalogEvaluationAssetResolver {
    async fn resolve(
        &self,
        requirements: &[EvaluationAssetRequirement],
    ) -> Result<Vec<ResolvedEvaluationAsset>> {
        requirements
            .iter()
            .map(|requirement| self.catalog.resolve(requirement))
            .collect()
    }
}

/// Endpoint-dialect to semantic evaluator-operation capability seam.
///
/// Endpoint descriptors intentionally remain transport/presentation metadata.
/// This separate resolver lets a future extension declare evaluator operation
/// compatibility without adding another closed endpoint enum to the workload.
pub trait EvaluationEndpointCapabilityResolver: fmt::Debug + Send + Sync {
    /// Return the exact semantic capability IDs executable by one endpoint.
    fn capabilities(&self, descriptor: &EndpointDescriptor) -> Result<BTreeSet<String>>;
}

/// Manifest-owned maximum local authority for one exact stock distribution.
#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct EvaluationCompatibilityGrantLimits {
    pub(crate) max_operations: u64,
    pub(crate) max_concurrent_operations: u64,
    pub(crate) max_request_bytes: u64,
    pub(crate) max_response_bytes: u64,
    pub(crate) max_stream_events: u64,
    pub(crate) expires_after_ms: u64,
}

impl EvaluationCompatibilityGrantLimits {
    pub(crate) fn validate(self) -> Result<()> {
        ensure!(
            self.max_operations > 0
                && self.max_operations <= 10_000_000
                && self.max_concurrent_operations > 0
                && self.max_concurrent_operations <= self.max_operations
                && self.max_concurrent_operations <= 255
                && self.max_request_bytes > 0
                && self.max_request_bytes <= 16 * 1024 * 1024 * 1024
                && self.max_response_bytes > 0
                && self.max_response_bytes <= 16 * 1024 * 1024 * 1024
                && self.max_stream_events > 0
                && self.max_stream_events <= 10_000_000
                && self.expires_after_ms > 0
                && self.expires_after_ms <= 24 * 60 * 60 * 1_000,
            "evaluation compatibility grant limits exceed the Rust proxy envelope"
        );
        Ok(())
    }
}

/// Manifest-owned compatibility requirement for one exact stock distribution.
#[derive(Clone, Debug, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(deny_unknown_fields)]
pub(crate) struct EvaluationCompatibilityRouteGrant {
    #[serde(rename = "dialect")]
    pub(crate) dialect_id: String,
    pub(crate) selector: String,
    pub(crate) service_id: LogicalServiceId,
    pub(crate) purpose: OperationPurpose,
    pub(crate) semantic_operation_id: SemanticOperationId,
    pub(crate) restricted_payload: bool,
}

impl EvaluationCompatibilityRouteGrant {
    pub(crate) fn validate(&self) -> Result<()> {
        ensure!(
            !self.dialect_id.is_empty() && self.dialect_id.trim() == self.dialect_id,
            "evaluation compatibility route has an invalid dialect ID"
        );
        self.to_proxy_route()?;
        Ok(())
    }

    fn to_proxy_route(&self) -> Result<CompatibilityProxyRoute> {
        if self.restricted_payload {
            CompatibilityProxyRoute::restricted(
                self.selector.clone(),
                self.service_id.clone(),
                self.purpose.clone(),
                self.semantic_operation_id.clone(),
            )
        } else {
            CompatibilityProxyRoute::new(
                self.selector.clone(),
                self.service_id.clone(),
                self.purpose.clone(),
                self.semantic_operation_id.clone(),
            )
        }
    }
}

/// Manifest-owned compatibility requirement for one exact stock distribution.
#[derive(Clone, Debug)]
pub(crate) struct EvaluationCompatibilityPolicyEntry {
    pub(crate) provider_id: EvaluationProviderId,
    pub(crate) distribution_id: EvaluationDistributionId,
    pub(crate) dialect_ids: Vec<String>,
    pub(crate) route_grants: Vec<EvaluationCompatibilityRouteGrant>,
    pub(crate) grant_limits: Option<EvaluationCompatibilityGrantLimits>,
}

/// Open factory for one local compatibility dialect family.
pub(crate) trait EvaluationCompatibilityDialectFactory: fmt::Debug + Send + Sync {
    /// Stable manifest-facing dialect identity.
    fn id(&self) -> &'static str;

    /// Bind one adapter to exact prepared logical routes.
    fn prepare(
        &self,
        routes: &EvaluationRouteTable,
        grants: &[EvaluationCompatibilityRouteGrant],
    ) -> Result<Arc<dyn CompatibilityProxyDialect>>;
}

/// Immutable manifest-driven proxy policy injected into pair preparation.
pub trait EvaluationCompatibilityPolicy: fmt::Debug + Send + Sync {
    /// Prepare no proxy for an explicitly pipe-only distribution, or one exact
    /// frozen adapter registry for a distribution that requires compatibility.
    fn prepare(
        &self,
        selection: &EvaluationProviderSelectionV2,
        routes: &EvaluationRouteTable,
    ) -> Result<Option<PreparedEvaluationCompatibility>>;
}

/// Frozen per-run compatibility composition and safe provenance identity.
#[derive(Clone, Debug)]
pub struct PreparedEvaluationCompatibility {
    dialects: CompatibilityProxyDialectRegistry,
    dialect_ids: Vec<String>,
    descriptor_sha256: String,
    grant_limits: EvaluationCompatibilityGrantLimits,
}

impl PreparedEvaluationCompatibility {
    fn dialects(&self) -> CompatibilityProxyDialectRegistry {
        self.dialects.clone()
    }
}

#[derive(Debug)]
struct FrozenEvaluationCompatibilityPolicy {
    entries: BTreeMap<(String, String), FrozenEvaluationCompatibilityEntry>,
    factories: BTreeMap<String, Arc<dyn EvaluationCompatibilityDialectFactory>>,
}

#[derive(Debug)]
struct FrozenEvaluationCompatibilityEntry {
    dialect_ids: Vec<String>,
    route_grants: Vec<EvaluationCompatibilityRouteGrant>,
    grant_limits: Option<EvaluationCompatibilityGrantLimits>,
}

impl FrozenEvaluationCompatibilityPolicy {
    fn new(
        entries: impl IntoIterator<Item = EvaluationCompatibilityPolicyEntry>,
        factories: impl IntoIterator<Item = Arc<dyn EvaluationCompatibilityDialectFactory>>,
    ) -> Result<Self> {
        let mut by_factory = BTreeMap::new();
        for factory in factories {
            let id = factory.id().to_string();
            ensure!(
                by_factory.insert(id.clone(), factory).is_none(),
                "duplicate evaluation compatibility dialect factory {id:?}"
            );
        }
        let mut by_distribution = BTreeMap::new();
        for entry in entries {
            let key = (
                entry.provider_id.as_str().to_string(),
                entry.distribution_id.as_str().to_string(),
            );
            let dialects = entry.dialect_ids.into_iter().collect::<BTreeSet<_>>();
            let route_grants = entry
                .route_grants
                .into_iter()
                .map(|grant| {
                    grant.validate()?;
                    Ok(grant)
                })
                .collect::<Result<BTreeSet<_>>>()?;
            if let Some(limits) = entry.grant_limits {
                limits.validate()?;
            }
            ensure!(
                dialects.is_empty() == route_grants.is_empty()
                    && dialects.is_empty() == entry.grant_limits.is_none(),
                "pipe-only and proxy-enabled compatibility policy fields disagree"
            );
            ensure!(
                dialects.iter().all(|id| by_factory.contains_key(id)),
                "evaluation compatibility policy references an unregistered dialect"
            );
            ensure!(
                route_grants
                    .iter()
                    .all(|grant| dialects.contains(&grant.dialect_id)),
                "evaluation compatibility route references an undeclared dialect"
            );
            ensure!(
                dialects.iter().all(|dialect| route_grants
                    .iter()
                    .any(|grant| &grant.dialect_id == dialect)),
                "evaluation compatibility dialect has no exact route grant"
            );
            ensure!(
                by_distribution
                    .insert(
                        key.clone(),
                        FrozenEvaluationCompatibilityEntry {
                            dialect_ids: dialects.into_iter().collect(),
                            route_grants: route_grants.into_iter().collect(),
                            grant_limits: entry.grant_limits,
                        },
                    )
                    .is_none(),
                "duplicate evaluation compatibility policy for {key:?}"
            );
        }
        ensure!(
            !by_distribution.is_empty(),
            "evaluation compatibility policy cannot be empty"
        );
        Ok(Self {
            entries: by_distribution,
            factories: by_factory,
        })
    }
}

impl EvaluationCompatibilityPolicy for FrozenEvaluationCompatibilityPolicy {
    fn prepare(
        &self,
        selection: &EvaluationProviderSelectionV2,
        routes: &EvaluationRouteTable,
    ) -> Result<Option<PreparedEvaluationCompatibility>> {
        let key = (
            selection.provider_id.as_str().to_string(),
            selection.distribution.as_str().to_string(),
        );
        let entry = self.entries.get(&key).ok_or_else(|| {
            anyhow!("stock evaluator distribution has no compatibility-policy entry")
        })?;
        if entry.dialect_ids.is_empty() {
            return Ok(None);
        }
        let grant_limits = entry
            .grant_limits
            .expect("nonempty policy requires validated grant limits");
        let dialects = entry
            .dialect_ids
            .iter()
            .map(|id| {
                let grants = entry
                    .route_grants
                    .iter()
                    .filter(|grant| &grant.dialect_id == id)
                    .cloned()
                    .collect::<Vec<_>>();
                self.factories
                    .get(id)
                    .expect("policy constructor validated dialect IDs")
                    .prepare(routes, &grants)
            })
            .collect::<Result<Vec<_>>>()?;
        let dialects = CompatibilityProxyDialectRegistry::new(dialects)?;
        let descriptor_sha256 = canonical_sha256(json!({
            "dialect_ids": entry.dialect_ids,
            "allowed_routes": entry.route_grants.iter().map(|grant| json!({
                "dialect_id": grant.dialect_id,
                "purpose": grant.purpose.as_str(),
                "restricted_payload": grant.restricted_payload,
                "selector": grant.selector,
                "semantic_operation_id": grant.semantic_operation_id.as_str(),
                "service_id": grant.service_id.as_str(),
            })).collect::<Vec<_>>(),
            "dialects": dialects
                .descriptors()
                .into_iter()
                .map(|(path, routes)| json!({
                    "path": path,
                    "routes": routes.into_iter().map(|route| json!({
                        "purpose": route.purpose.as_str(),
                        "selector": route.selector,
                        "semantic_operation_id": route.semantic_operation_id.as_str(),
                        "service_id": route.service_id.as_str(),
                    })).collect::<Vec<_>>(),
                }))
                .collect::<Vec<_>>(),
            "grant_limits": {
                "expires_after_ms": grant_limits.expires_after_ms,
                "max_concurrent_operations": grant_limits.max_concurrent_operations,
                "max_operations": grant_limits.max_operations,
                "max_request_bytes": grant_limits.max_request_bytes,
                "max_response_bytes": grant_limits.max_response_bytes,
                "max_stream_events": grant_limits.max_stream_events,
            },
        }))?;
        Ok(Some(PreparedEvaluationCompatibility {
            dialects,
            dialect_ids: entry.dialect_ids.clone(),
            descriptor_sha256,
            grant_limits,
        }))
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct OpenAiChatCompatibilityDialectFactory;

impl EvaluationCompatibilityDialectFactory for OpenAiChatCompatibilityDialectFactory {
    fn id(&self) -> &'static str {
        "openai_chat_completions"
    }

    fn prepare(
        &self,
        routes: &EvaluationRouteTable,
        grants: &[EvaluationCompatibilityRouteGrant],
    ) -> Result<Arc<dyn CompatibilityProxyDialect>> {
        let routes = grants
            .iter()
            .map(|grant| {
                ensure!(
                    grant.dialect_id == self.id()
                        && grant.semantic_operation_id.as_str() == "model.generate",
                    "OpenAI chat compatibility grant has an incompatible dialect or semantic operation"
                );
                let route = routes.resolve(grant.service_id.as_str())?;
                ensure!(
                    route.purpose == grant.purpose.as_str(),
                    "OpenAI chat compatibility route {:?} purpose drifted from the manifest grant",
                    route.service_id
                );
                ensure!(
                    route.endpoint_capabilities.contains("chat"),
                    "OpenAI chat compatibility route {:?} requires a chat-capable endpoint",
                    route.service_id
                );
                grant.to_proxy_route()
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Arc::new(OpenAiChatCompatibilityDialect::new(routes)?))
    }
}

pub(crate) fn stock_evaluation_compatibility_policy(
    entries: impl IntoIterator<Item = EvaluationCompatibilityPolicyEntry>,
) -> Result<Arc<dyn EvaluationCompatibilityPolicy>> {
    Ok(Arc::new(FrozenEvaluationCompatibilityPolicy::new(
        entries,
        [Arc::new(OpenAiChatCompatibilityDialectFactory)
            as Arc<dyn EvaluationCompatibilityDialectFactory>],
    )?))
}

/// Built-in endpoint capability projection for stock evaluator operations.
#[derive(Clone, Copy, Debug, Default)]
pub struct NativeEvaluationEndpointCapabilityResolver;

impl EvaluationEndpointCapabilityResolver for NativeEvaluationEndpointCapabilityResolver {
    fn capabilities(&self, descriptor: &EndpointDescriptor) -> Result<BTreeSet<String>> {
        let capability = match descriptor.id {
            "chat" | "messages" | "kserve_chat" => "chat",
            "completions" | "kserve_completions" => "completion",
            "responses" => "responses",
            "embeddings"
            | "chat_embeddings"
            | "nim_embeddings"
            | "kserve_embeddings"
            | "kserve_v2_embeddings" => "embedding",
            _ => {
                bail!(
                    "endpoint {:?} has no registered evaluator inference capability adapter",
                    descriptor.id
                )
            }
        };
        Ok(BTreeSet::from([capability.to_string()]))
    }
}

/// Register the real `online_http + evaluation` adapter against the same
/// provider registry used for side-effect-free workload validation.
pub fn register_online_http_evaluation_pair(
    builder: &mut RunnerRegistryBuilder,
    providers: Arc<EvaluationProviderRegistry>,
    assets: Arc<dyn EvaluationAssetCatalog>,
    compatibility: Arc<dyn EvaluationCompatibilityPolicy>,
) -> Result<()> {
    register_evaluation_workload(builder, providers.clone())?;
    builder.register_pair(Arc::new(OnlineHttpEvaluationPairFactory {
        providers,
        assets,
        compatibility,
        endpoint_capabilities: Arc::new(NativeEvaluationEndpointCapabilityResolver),
    }))
}

#[derive(Clone)]
struct OnlineHttpEvaluationPairFactory {
    providers: Arc<EvaluationProviderRegistry>,
    assets: Arc<dyn EvaluationAssetCatalog>,
    compatibility: Arc<dyn EvaluationCompatibilityPolicy>,
    endpoint_capabilities: Arc<dyn EvaluationEndpointCapabilityResolver>,
}

impl fmt::Debug for OnlineHttpEvaluationPairFactory {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("OnlineHttpEvaluationPairFactory")
            .field("provider_count", &self.providers.len())
            .finish_non_exhaustive()
    }
}

impl RunnerPairFactory for OnlineHttpEvaluationPairFactory {
    fn backend_id(&self) -> &'static str {
        "online_http"
    }

    fn workload_id(&self) -> &'static str {
        EVALUATION_WORKLOAD_DESCRIPTOR.id
    }

    fn validate_pair(
        &self,
        backend: &dyn ValidatedBackendConfig,
        workload: &dyn ValidatedWorkloadConfig,
    ) -> Result<()> {
        validated_evaluation_backend(backend)?;
        validated_evaluation_workload(workload)?
            .authored()
            .validate_structure()
    }

    fn validate_run(
        &self,
        run: &crate::protocol_v2::AuthoredRunSpecV2,
        context: &RunnerRunContext,
        backend: &dyn ValidatedBackendConfig,
        workload: &dyn ValidatedWorkloadConfig,
    ) -> Result<()> {
        self.validate_pair(backend, workload)?;
        let workload = validated_evaluation_workload(workload)?;
        validate_evaluation_authored_run(run, context, workload)?;
        let routes =
            prepare_evaluation_routes(context, workload, self.endpoint_capabilities.as_ref())?;
        self.compatibility
            .prepare(&workload.authored().provider, &routes.routes)?;
        Ok(())
    }

    fn prepare(
        &self,
        _run: &crate::protocol_v2::AuthoredRunSpecV2,
        _backend: Box<dyn ValidatedBackendConfig>,
        _workload: Box<dyn ValidatedWorkloadConfig>,
    ) -> Result<Box<dyn PreparedRunnerOperation>> {
        bail!("online_http + evaluation preparation requires the coordinator-owned runner context")
    }

    fn prepare_with_context(
        &self,
        run: &crate::protocol_v2::AuthoredRunSpecV2,
        context: &RunnerRunContext,
        backend: Box<dyn ValidatedBackendConfig>,
        workload: Box<dyn ValidatedWorkloadConfig>,
    ) -> Result<Box<dyn PreparedRunnerOperation>> {
        self.validate_run(run, context, backend.as_ref(), workload.as_ref())?;
        let workload = validated_evaluation_workload(workload.as_ref())?.clone();
        let routes =
            prepare_evaluation_routes(context, &workload, self.endpoint_capabilities.as_ref())?;
        let compatibility = self
            .compatibility
            .prepare(&workload.authored().provider, &routes.routes)?;
        let readiness = crate::online_execution::prepare_online_readiness(run, context)?;
        let metrics = crate::execute::metrics_config(&run.metrics)?;
        Ok(Box::new(PreparedOnlineEvaluationOperation {
            benchmark_id: run.identity.benchmark_id.clone(),
            artifact_target: run.artifact_target.clone(),
            workload,
            providers: self.providers.clone(),
            assets: self.assets.clone(),
            routes,
            compatibility,
            backend_factory: context.execution_factories().http_handle(),
            readiness,
            readiness_transport: context.execution_factories().readiness_transport_handle(),
            metrics,
        }))
    }
}

fn validated_evaluation_backend(
    config: &dyn ValidatedBackendConfig,
) -> Result<&OnlineHttpBackendConfigV2> {
    ValidatedBackendConfig::as_any(config)
        .downcast_ref::<OnlineHttpBackendConfigV2>()
        .ok_or_else(|| anyhow!("evaluation pair received the wrong online_http backend config"))
}

fn validate_evaluation_authored_run(
    run: &crate::protocol_v2::AuthoredRunSpecV2,
    context: &RunnerRunContext,
    workload: &ValidatedEvaluationWorkloadConfigV2,
) -> Result<()> {
    ensure!(
        workload.authored().resources.is_empty(),
        "this runner distribution registers no evaluator host-resource capabilities"
    );
    crate::execute::metrics_config(&run.metrics)?;
    ensure!(
        run.artifacts.records_path.is_none()
            && run.artifacts.raw_path.is_none()
            && run.artifacts.outputs_path.is_none()
            && !run.artifacts.trace
            && run.artifacts.user_files.is_empty(),
        "evaluation execution writes only canonical native-v2 and sealed provider artifacts"
    );
    ensure!(
        run.artifact_target.is_absolute(),
        "evaluation artifact target must be absolute"
    );
    let models = run
        .models
        .items
        .iter()
        .map(|model| model.name.as_str())
        .collect::<BTreeSet<_>>();
    for (service_id, route) in &workload.authored().routes {
        ensure!(
            models.contains(route.model.as_str()),
            "evaluation route {service_id} references undeclared model {:?}",
            route.model
        );
        context.endpoint_profile(route.endpoint_profile.as_str())?;
    }
    Ok(())
}

#[derive(Clone)]
struct EvaluationPreparedEndpointProfile {
    profile_id: String,
    endpoint_id: EndpointId,
    config: RawEndpointConfig,
}

#[derive(Clone)]
struct EvaluationPreparedEndpointTableFactory {
    registry: EndpointRegistry,
    profiles: Vec<EvaluationPreparedEndpointProfile>,
}

impl fmt::Debug for EvaluationPreparedEndpointTableFactory {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("EvaluationPreparedEndpointTableFactory")
            .field(
                "profiles",
                &self
                    .profiles
                    .iter()
                    .map(|profile| (&profile.profile_id, &profile.endpoint_id))
                    .collect::<Vec<_>>(),
            )
            .finish()
    }
}

impl EvaluationPreparedEndpointTableFactory {
    fn prepare_table(&self) -> Result<PreparedEndpointTable> {
        let mut table = PreparedEndpointTable::new();
        for profile in &self.profiles {
            let endpoint = self
                .registry
                .prepare(&profile.endpoint_id, profile.config.clone())
                .with_context(|| {
                    format!(
                        "preparing evaluation endpoint profile {:?}",
                        profile.profile_id
                    )
                })?;
            table.push(endpoint)?;
        }
        Ok(table)
    }
}

impl HttpPreparedEndpointTableFactory for EvaluationPreparedEndpointTableFactory {
    fn prepare_worker(&self) -> Result<PreparedEndpointTable> {
        self.prepare_table()
    }
}

struct PreparedEvaluationRoutes {
    routes: EvaluationRouteTable,
    prepared: Vec<PreparedEvaluationRoute>,
    table_factory: Arc<dyn HttpPreparedEndpointTableFactory>,
    base_urls: Vec<String>,
    transport: TransportSinkConfig,
    default_model: String,
    route_map_sha256: String,
    prepared_endpoints_sha256: String,
}

fn prepare_evaluation_routes(
    context: &RunnerRunContext,
    workload: &ValidatedEvaluationWorkloadConfigV2,
    capabilities: &dyn EvaluationEndpointCapabilityResolver,
) -> Result<PreparedEvaluationRoutes> {
    let referenced = workload
        .authored()
        .routes
        .values()
        .map(|route| route.endpoint_profile.as_str())
        .collect::<BTreeSet<_>>();
    let mut profiles = Vec::with_capacity(referenced.len());
    let mut policies = BTreeMap::new();
    for (profile_id, profile) in context.endpoint_profiles() {
        if !referenced.contains(profile_id) {
            continue;
        }
        let factory = context
            .product_registry()
            .endpoints()
            .resolve_factory(&profile.endpoint_id)
            .with_context(|| format!("resolving evaluation endpoint profile {profile_id:?}"))?;
        let mut config = profile.config.clone();
        if factory.descriptor().supports_streaming {
            config.streaming = true;
            config.use_server_token_count = true;
        }
        profiles.push(EvaluationPreparedEndpointProfile {
            profile_id: profile_id.to_string(),
            endpoint_id: profile.endpoint_id.clone(),
            config,
        });
        policies.insert(
            profile_id.to_string(),
            (
                profile.connection_reuse,
                profile.client.clone(),
                profile.session_header.clone(),
            ),
        );
    }
    ensure!(
        profiles.len() == referenced.len(),
        "one or more evaluation endpoint profiles were not prepared"
    );
    let table_factory = Arc::new(EvaluationPreparedEndpointTableFactory {
        registry: context.product_registry().endpoints().clone(),
        profiles,
    });
    let table = Rc::new(table_factory.prepare_table()?);

    let mut references = BTreeMap::new();
    let mut endpoint_identities = Vec::new();
    for (index, profile) in table_factory.profiles.iter().enumerate() {
        let index =
            u32::try_from(index).context("evaluation endpoint profile index exceeds u32")?;
        let key = EndpointKey::from_index(index);
        let endpoint = table.get(key)?;
        let endpoint_id = EndpointId::new(endpoint.descriptor().id)?;
        let identity =
            endpoint_identity_sha256(&profile.profile_id, endpoint_id.as_str(), endpoint)?;
        ensure!(
            references
                .insert(
                    profile.profile_id.clone(),
                    (
                        PreparedEndpointReference {
                            key,
                            endpoint_id: endpoint_id.clone(),
                        },
                        identity.clone(),
                        capabilities.capabilities(endpoint.descriptor())?,
                    ),
                )
                .is_none(),
            "duplicate prepared evaluation endpoint profile {:?}",
            profile.profile_id
        );
        endpoint_identities.push(json!({
            "endpoint_id": endpoint_id.as_str(),
            "prepared_identity_sha256": identity,
            "profile_id": profile.profile_id,
        }));
    }

    let mut routes = Vec::with_capacity(workload.authored().routes.len());
    let mut prepared = Vec::with_capacity(workload.authored().routes.len());
    for (service_id, authored) in &workload.authored().routes {
        let (reference, prepared_identity_sha256, endpoint_capabilities) = references
            .get(authored.endpoint_profile.as_str())
            .ok_or_else(|| {
                anyhow!(
                    "evaluation route {service_id} references unprepared endpoint profile {:?}",
                    authored.endpoint_profile.as_str()
                )
            })?;
        let route = EvaluationRoute {
            service_id: service_id.as_str().to_string(),
            purpose: authored.purpose.as_str().to_string(),
            model: authored.model.clone(),
            endpoint_profile: authored.endpoint_profile.as_str().to_string(),
            prepared_identity_sha256: prepared_identity_sha256.clone(),
            endpoint_capabilities: endpoint_capabilities.clone(),
        };
        route.validate()?;
        routes.push(route.clone());
        prepared.push(PreparedEvaluationRoute {
            route,
            endpoint_table: table.clone(),
            endpoint: reference.clone(),
        });
    }
    let routes = EvaluationRouteTable::new(routes)?;
    let route_map_sha256 = canonical_sha256(json!(
        routes
            .routes()
            .map(|route| json!({
                "endpoint_capabilities": route.endpoint_capabilities,
                "endpoint_profile": route.endpoint_profile,
                "model": route.model,
                "prepared_identity_sha256": route.prepared_identity_sha256,
                "purpose": route.purpose,
                "service_id": route.service_id,
            }))
            .collect::<Vec<_>>()
    ))?;
    let prepared_endpoints_sha256 = canonical_sha256(Value::Array(endpoint_identities))?;
    let first_profile = table_factory
        .profiles
        .first()
        .ok_or_else(|| anyhow!("evaluation has no prepared endpoint profile"))?;
    let first_policy = policies
        .get(&first_profile.profile_id)
        .expect("prepared profile retained transport policy");
    for profile in &table_factory.profiles {
        let policy = policies
            .get(&profile.profile_id)
            .expect("prepared profile retained transport policy");
        ensure!(
            policy == first_policy,
            "evaluation endpoint profiles must share HTTP client, reuse, and session-header policy"
        );
    }
    ensure!(
        !first_profile.config.urls.is_empty(),
        "evaluation endpoint profile {:?} has no URL",
        first_profile.profile_id
    );
    let transport = TransportSinkConfig {
        client: first_policy.1.clone(),
        connection_reuse: first_policy.0,
        session_header: first_policy.2.clone(),
    };
    let base_urls = first_profile.config.urls.clone();
    let default_model = workload
        .authored()
        .routes
        .values()
        .next()
        .expect("structural validation requires a route")
        .model
        .clone();
    Ok(PreparedEvaluationRoutes {
        routes,
        prepared,
        table_factory,
        base_urls,
        transport,
        default_model,
        route_map_sha256,
        prepared_endpoints_sha256,
    })
}

fn endpoint_identity_sha256(
    profile_id: &str,
    endpoint_id: &str,
    endpoint: &dyn aiperf_endpoints::PreparedEndpoint,
) -> Result<String> {
    let raw = endpoint.config().to_raw();
    let public_policy = serde_json::to_value(&raw)?;
    canonical_sha256(json!({
        "endpoint_id": endpoint_id,
        "has_api_key": raw.api_key.is_some(),
        "header_names": raw.headers.keys().collect::<Vec<_>>(),
        "policy": public_policy,
        "profile_id": profile_id,
    }))
}

fn canonical_sha256(value: Value) -> Result<String> {
    Ok(CanonicalJson::new(value)
        .map_err(|error| anyhow!(error.to_string()))?
        .normalized_result_sha256())
}

struct PreparedOnlineEvaluationOperation {
    benchmark_id: String,
    artifact_target: PathBuf,
    workload: ValidatedEvaluationWorkloadConfigV2,
    providers: Arc<EvaluationProviderRegistry>,
    assets: Arc<dyn EvaluationAssetCatalog>,
    routes: PreparedEvaluationRoutes,
    compatibility: Option<PreparedEvaluationCompatibility>,
    backend_factory: Arc<dyn HttpExecutionBackendFactory>,
    readiness: Box<dyn PreparedOnlineReadiness>,
    readiness_transport: Arc<dyn ReadinessTransportFactory>,
    metrics: MetricsConfig,
}

impl fmt::Debug for PreparedOnlineEvaluationOperation {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PreparedOnlineEvaluationOperation")
            .field("benchmark_id", &self.benchmark_id)
            .field("artifact_target", &self.artifact_target)
            .field("provider", &self.workload.authored().provider)
            .field(
                "routes",
                &self
                    .routes
                    .routes
                    .routes()
                    .map(|route| route.service_id.as_str())
                    .collect::<Vec<_>>(),
            )
            .finish_non_exhaustive()
    }
}

impl PreparedRunnerOperation for PreparedOnlineEvaluationOperation {
    fn execute(self: Box<Self>) -> Result<PreparedRunOutcome> {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .context("creating evaluation runner Tokio runtime")?;
        tokio::task::LocalSet::new().block_on(&runtime, execute_online_evaluation(*self))
    }
}

async fn execute_online_evaluation(
    operation: PreparedOnlineEvaluationOperation,
) -> Result<PreparedRunOutcome> {
    let real_clock_anchor = RealClockAnchor::now();
    let clock: Rc<dyn Clock> = RealClock::from_anchor(real_clock_anchor);
    let authored = operation.workload.authored();
    let factory = operation
        .providers
        .get(&authored.provider.provider_id)
        .ok_or_else(|| anyhow!("validated evaluation provider disappeared before execute"))?
        .clone();
    ensure!(
        operation.providers.distribution_is_available(
            &authored.provider.provider_id,
            &authored.provider.distribution,
        ),
        "validated evaluator distribution is no longer available"
    );
    let public_config = factory
        .project_public_config(operation.workload.provider_config().config())
        .map_err(|error| anyhow!(error.to_string()))
        .context("projecting factory-reviewed public evaluator config")?;

    let artifact_roots = prepare_evaluation_artifact_roots(
        &operation.artifact_target,
        operation.readiness.as_ref(),
        operation.readiness_transport.as_ref(),
        clock.clone(),
    )
    .await?;
    let staging_root = artifact_roots.staging;
    let promoted_root = artifact_roots.promoted;

    let session_id = EvaluationSessionId::new(format!("evaluation_{}", Uuid::new_v4().simple()))
        .map_err(|error| anyhow!(error.to_string()))?;
    let start_ns = clock.now_ns();
    let workers = std::thread::available_parallelism()
        .map(usize::from)
        .unwrap_or(1)
        .min(authored.unit_concurrency)
        .max(1);
    let execution_backend = operation
        .backend_factory
        .build(HttpExecutionBackendConfig {
            workers,
            coordinator_clock: clock.clone(),
            real_clock_anchor,
            base_urls: operation.routes.base_urls.clone(),
            model: operation.routes.default_model.clone(),
            transport: operation.routes.transport.clone(),
            prepared_endpoints: Some(operation.routes.table_factory.clone()),
        })
        .context("building evaluation HTTP execution placement")?;
    execution_backend
        .set_run_origin(start_ns)
        .context("setting evaluation HTTP run origin")?;
    let dispatcher: Rc<dyn TurnDispatcher> = Rc::new(EvaluationDispatcher {
        execution_backend: execution_backend.clone(),
        default_model: operation.routes.default_model.clone(),
    });
    let scheduled = ScheduledRuntime::new_with_metrics_config(
        clock.clone(),
        start_ns,
        dispatcher,
        StopConfig::default(),
        false,
        operation.metrics.clone(),
    );
    let materializer: Rc<dyn EvaluationInferenceMaterializer> =
        Rc::new(DatasetEvaluationInferenceMaterializer::new(
            Arc::new(TiktokenTokenizer::default()) as Arc<dyn TextTokenizer>,
            operation.routes.prepared.clone(),
        )?);
    let mut host_builder = HostExecutorRegistryBuilder::default();
    register_scheduled_inference_host_executors(&mut host_builder, materializer)?;
    let host_executors = host_builder.freeze()?;
    let host_runtime = HostExecutorRuntime::scheduled(scheduled.clone());
    let asset_resolver: Rc<dyn EvaluationAssetResolver> = Rc::new(CatalogEvaluationAssetResolver {
        catalog: operation.assets.clone(),
    });
    let sealer = EvaluationArtifactSealer::new(
        ArtifactSealLimits::default(),
        ArtifactProjectionPolicy::restricted_only(),
    )
    .map_err(|error| anyhow!(error.to_string()))?;
    let finalizer = Rc::new(SealingEvaluationArtifactFinalizer::new(
        sealer,
        &staging_root,
        &promoted_root,
    )?);
    let limits = EvaluationWorkloadLimits {
        unit_concurrency: authored.unit_concurrency,
        credit_ceiling: EvaluationQueueCredits {
            units: 65_536,
            host_operations: 1_000_000,
            host_operations_per_unit: 1_000_000,
            stream_events: 1_000_000,
            sandboxes: 0,
            processes: 0,
            artifacts: ArtifactSealLimits::default().max_artifacts,
            artifact_bytes: ArtifactSealLimits::default().max_total_bytes,
        },
        unit_page_size: 4_096,
        event_batch_size: 4_096,
        idle_poll_ns: 1_000_000,
    };
    limits.validate()?;
    let (host_capabilities, capability_inventory_sha256, schema_inventory_sha256) =
        evaluation_host_capabilities()?;
    let runner_sha256 = current_executable_sha256()?;

    let mut proxy_binding = None;
    let mut process_authorizer = None;
    let mut proxy_server: Option<EvaluatorCompatibilityProxy> = None;
    let mut proxy_receiver: Option<ProxyOperationReceiver> = None;
    if let Some(compatibility) = &operation.compatibility {
        let routes = compatibility.dialects.routes();
        let scope_nonce = Uuid::new_v4().simple().to_string();
        let process_scope_sha256 = canonical_sha256(json!({
            "compatibility_descriptor_sha256": compatibility.descriptor_sha256,
            "scope_nonce": scope_nonce,
            "session_id": session_id.as_str(),
        }))?;
        let grant = ScopedProxyGrant {
            grant_id: format!("grant-{}", Uuid::new_v4().simple()),
            session_id: session_id.clone(),
            secret: ScopedProxySecret::new(format!(
                "{}{}",
                Uuid::new_v4().simple(),
                Uuid::new_v4().simple()
            ))
            .map_err(|error| anyhow!(error.to_string()))?,
            service_ids: routes
                .iter()
                .map(|route| route.service_id.clone())
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect(),
            semantic_operation_ids: routes
                .iter()
                .map(|route| route.semantic_operation_id.clone())
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect(),
            purposes: routes
                .iter()
                .map(|route| route.purpose.clone())
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect(),
            process_scope_sha256: process_scope_sha256.clone(),
            max_operations: compatibility.grant_limits.max_operations,
            max_concurrent_operations: compatibility.grant_limits.max_concurrent_operations,
            max_request_bytes: compatibility.grant_limits.max_request_bytes,
            max_response_bytes: compatibility.grant_limits.max_response_bytes,
            max_stream_events: compatibility.grant_limits.max_stream_events,
            expires_after_ms: compatibility.grant_limits.expires_after_ms,
        };
        let binding = ScopedProxyBinding {
            local_locator: EVALUATOR_PROXY_LOCAL_LOCATOR.to_string(),
            host_socket_path: staging_root
                .parent()
                .expect("evaluation staging root has a parent")
                .join("evaluator-proxy.sock"),
            grant,
        };
        binding
            .validate()
            .map_err(|error| anyhow!(error.to_string()))?;
        let authorizer = Arc::new(LinuxProcessSubtreeAuthorizer::new(process_scope_sha256)?);
        let started = start_evaluator_compatibility_proxy(
            binding.clone(),
            clock.now_ns(),
            authorizer.clone() as Arc<dyn ProxyProcessScopeAuthorizer>,
            compatibility.dialects(),
            ProxyTimingRuntime::from_clock(clock.clone()),
        )
        .await;
        let (server, receiver) = match started {
            Ok(started) => started,
            Err(error) => {
                let shutdown = execution_backend.shutdown();
                return match shutdown {
                    Ok(()) => Err(error).context("starting evaluator compatibility proxy"),
                    Err(shutdown) => Err(error).context(format!(
                        "starting evaluator compatibility proxy; HTTP backend shutdown also failed: {shutdown:#}"
                    )),
                };
            }
        };
        proxy_binding = Some(binding);
        process_authorizer = Some(authorizer);
        proxy_server = Some(server);
        proxy_receiver = Some(receiver);
    }

    let launch_nonce = format!("{}{}", Uuid::new_v4().simple(), Uuid::new_v4().simple());
    let launch_context = ProviderLaunchContext {
        session_id: session_id.clone(),
        staging_dir: staging_root.clone(),
        proxy: proxy_binding.clone(),
        process_root_binder: process_authorizer
            .clone()
            .map(|authorizer| authorizer as Arc<dyn aiperf_accuracy::EvaluatorProcessRootBinder>),
        protocol_limits: EvaluatorProtocolLimits::default(),
        launch_nonce,
    };
    let prepared_launch = match factory
        .prepare_launch(&authored.provider.distribution, launch_context)
        .map_err(|error| anyhow!(error.to_string()))
        .context("preparing attested evaluator launch")
    {
        Ok(prepared) => prepared,
        Err(error) => {
            return Err(cleanup_evaluation_setup_failure(
                error,
                None,
                &mut proxy_server,
                &execution_backend,
            )
            .await);
        }
    };
    if prepared_launch.distribution_id() != &authored.provider.distribution
        || !aiperf_accuracy::is_sha256(prepared_launch.prelaunch_context_sha256())
    {
        return Err(cleanup_evaluation_setup_failure(
            anyhow!("prepared evaluator launch returned invalid distribution/digest evidence"),
            None,
            &mut proxy_server,
            &execution_backend,
        )
        .await);
    }
    let host_binding = EvaluationHostBinding {
        host: EvaluationHostIdentity {
            runner_sha256,
            capability_inventory_sha256,
            schema_inventory_sha256,
            isolation_proof_sha256: prepared_launch.isolation_evidence().proof_sha256.clone(),
        },
        route_map_sha256: operation.routes.route_map_sha256.clone(),
        prepared_endpoints_sha256: operation.routes.prepared_endpoints_sha256.clone(),
        sandbox_sha256: None,
    };
    if let Err(error) = host_binding.validate() {
        return Err(cleanup_evaluation_setup_failure(
            anyhow!(error.to_string()).context("validating evaluator host binding"),
            None,
            &mut proxy_server,
            &execution_backend,
        )
        .await);
    }
    let provider = match factory
        .launch(
            prepared_launch,
            operation.workload.provider_config(),
            host_binding,
        )
        .await
        .map_err(|error| anyhow!(error.to_string()))
        .context("launching supervised evaluator provider")
    {
        Ok(provider) => provider,
        Err(error) => {
            return Err(cleanup_evaluation_setup_failure(
                error,
                None,
                &mut proxy_server,
                &execution_backend,
            )
            .await);
        }
    };

    let plan_request = operation.workload.provider_config().plan_request(
        session_id,
        authored.provider.distribution.clone(),
        true,
    );
    let public_score_projection_policy = factory.public_score_projection_policy().clone();
    let public_aggregate_projection_policy = factory.public_aggregate_projection_policy().clone();
    let public_metadata_projector = factory.public_metadata_projector();
    let mut workload = EvaluationWorkload::new(
        provider,
        plan_request,
        operation.routes.routes.clone(),
        host_executors,
        host_runtime,
        asset_resolver,
        host_capabilities,
        public_score_projection_policy,
        public_aggregate_projection_policy,
        public_metadata_projector,
        public_config,
        None,
        finalizer,
        limits,
        OperationCancellation::default(),
    )
    .expect("evaluation workload limits were prevalidated");
    if let Some(server) = proxy_server.take() {
        workload = workload
            .with_compatibility_proxy(
                server,
                proxy_receiver
                    .take()
                    .expect("proxy server and receiver are prepared together"),
            )
            .expect("prepared proxy server and receiver share one authority");
    }
    let execution = workload.execute().await;
    let shutdown = execution_backend.shutdown();
    let execution = match (execution, shutdown) {
        (Ok(execution), Ok(())) => execution,
        (Err(error), Ok(())) => return Err(error),
        (Ok(_), Err(error)) => return Err(error).context("shutting down evaluation HTTP backend"),
        (Err(error), Err(shutdown)) => {
            return Err(error.context(format!(
                "evaluation HTTP backend also failed during shutdown: {shutdown:#}"
            )));
        }
    };
    let scheduled_report = scheduled.finish("evaluation", None);
    let evaluation_report = build_evaluation_report(
        &execution.candidate,
        &execution.sealed_artifacts,
        &operation.routes.routes,
        &execution.report_facts,
    )?;
    let case_count = evaluation_report.case_count;
    let operation_count = execution.operations.len();
    let native_report = NativeReport::from_outcome(
        &scheduled_report.native_metrics,
        &RunOutcome {
            run: ReportRunInfo {
                mode: Some("online".to_string()),
                model: Some(operation.routes.default_model.clone()),
            },
            evaluation: Some(evaluation_report),
            ..RunOutcome::default()
        },
    );
    let mut provenance = BTreeMap::from([
        ("backend".to_string(), "online_http".to_string()),
        (
            "provider".to_string(),
            authored.provider.provider_id.to_string(),
        ),
        (
            "distribution".to_string(),
            authored.provider.distribution.to_string(),
        ),
        ("evaluation_cases".to_string(), case_count.to_string()),
        (
            "evaluation_host_operations".to_string(),
            operation_count.to_string(),
        ),
    ]);
    match &operation.compatibility {
        Some(compatibility) => {
            provenance.insert(
                "evaluation_compatibility_proxy".to_string(),
                compatibility.dialect_ids.join(","),
            );
            provenance.insert(
                "evaluation_compatibility_descriptor_sha256".to_string(),
                compatibility.descriptor_sha256.clone(),
            );
        }
        None => {
            provenance.insert(
                "evaluation_compatibility_proxy".to_string(),
                "none".to_string(),
            );
        }
    }
    let report_facts = match &operation.compatibility {
        Some(compatibility) => {
            let grant = execution.compatibility_grant.ok_or_else(|| {
                anyhow!("proxy-enabled evaluation omitted its post-plan effective grant")
            })?;
            ReportPairRunFacts::new().with_evaluation_compatibility(
                ReportEvaluationCompatibilityInfo::new(
                    compatibility.dialect_ids.clone(),
                    compatibility.descriptor_sha256.clone(),
                    ReportEvaluationCompatibilityGrantLimits {
                        max_operations: grant.max_operations,
                        max_concurrent_operations: grant.max_concurrent_operations,
                        max_request_bytes: grant.max_request_bytes,
                        max_response_bytes: grant.max_response_bytes,
                        max_stream_events: grant.max_stream_events,
                        expires_after_ms: grant.expires_after_ms,
                    },
                )?,
            )
        }
        None => {
            ensure!(
                execution.compatibility_grant.is_none(),
                "pipe-only evaluation unexpectedly installed a compatibility grant"
            );
            ReportPairRunFacts::new()
        }
    };
    Ok(PreparedRunOutcome {
        native_report,
        report_facts,
        provenance,
        report_commit: Some(Box::new(PendingEvaluationReportCommit(
            execution.report_commit,
        ))),
    })
}

async fn cleanup_evaluation_setup_failure(
    primary: anyhow::Error,
    provider: Option<&mut dyn EvaluationProvider>,
    proxy: &mut Option<EvaluatorCompatibilityProxy>,
    execution_backend: &Rc<dyn HttpTurnExecutionBackend>,
) -> anyhow::Error {
    let abort_error = match provider {
        Some(provider) => provider
            .abort()
            .await
            .map_err(|error| anyhow!(error.to_string()))
            .err(),
        None => None,
    };
    let proxy_error = match proxy.take() {
        Some(proxy) => proxy
            .shutdown()
            .await
            .context("shutting down evaluator compatibility proxy during setup cleanup")
            .err(),
        None => None,
    };
    let backend_error = execution_backend.shutdown().err();
    if let Some(abort_error) = abort_error {
        return anyhow!(
            "evaluator provider abort/quiescence failed: {abort_error:#}; original setup failure: {primary:#}"
        );
    }
    let mut cleanup = Vec::new();
    if let Some(error) = proxy_error {
        cleanup.push(format!("compatibility proxy: {error:#}"));
    }
    if let Some(error) = backend_error {
        cleanup.push(format!("HTTP backend: {error:#}"));
    }
    if cleanup.is_empty() {
        primary
    } else {
        anyhow!(
            "{primary:#}; evaluator setup cleanup also failed: {}",
            cleanup.join("; ")
        )
    }
}

struct EvaluationArtifactRoots {
    staging: PathBuf,
    promoted: PathBuf,
}

/// Activate endpoint-owned readiness before any run artifact path exists.
///
/// This preserves the ordinary online lifecycle invariant: a readiness
/// failure cannot leave behind a directory that looks like a started run.
async fn prepare_evaluation_artifact_roots(
    artifact_target: &Path,
    readiness: &dyn PreparedOnlineReadiness,
    readiness_transport: &dyn ReadinessTransportFactory,
    clock: Rc<dyn Clock>,
) -> Result<EvaluationArtifactRoots> {
    if !readiness.is_empty() {
        let transport = readiness_transport.build(clock.clone());
        readiness
            .wait(clock, transport)
            .await
            .context("waiting for endpoint readiness")?;
    }
    std::fs::create_dir_all(artifact_target).with_context(|| {
        format!(
            "creating evaluation artifact target {}",
            artifact_target.display()
        )
    })?;
    let evaluation_root = artifact_target.join("evaluation");
    std::fs::create_dir(&evaluation_root).with_context(|| {
        format!(
            "creating exclusive evaluation artifact root {}",
            evaluation_root.display()
        )
    })?;
    let staging = evaluation_root.join("staging");
    std::fs::create_dir(&staging)
        .with_context(|| format!("creating evaluation staging root {}", staging.display()))?;
    Ok(EvaluationArtifactRoots {
        staging,
        promoted: evaluation_root.join("artifacts"),
    })
}

struct PendingEvaluationReportCommit(EvaluationReportCommit);

impl fmt::Debug for PendingEvaluationReportCommit {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_tuple("PendingEvaluationReportCommit")
            .field(&self.0)
            .finish()
    }
}

impl PreparedReportCommit for PendingEvaluationReportCommit {
    fn commit(self: Box<Self>) -> Result<()> {
        let Self(commit) = *self;
        commit.commit()
    }
}

struct EvaluationDispatcher {
    execution_backend: Rc<dyn HttpTurnExecutionBackend>,
    default_model: String,
}

#[async_trait(?Send)]
impl TurnDispatcher for EvaluationDispatcher {
    fn supports_response_streaming(&self) -> bool {
        self.execution_backend.supports_response_streaming()
    }

    fn inference_dimensions(&self, turn: &TurnToSend) -> aiperf_metrics::InferenceDimensions {
        self.execution_backend.inference_dimensions(turn)
    }

    async fn dispatch_turn(
        &self,
        turn: TurnToSend,
        observer: &dyn RequestObserver,
        on_first_token: &dyn Fn(i64),
    ) -> Result<TurnDispatchOutcome> {
        Ok(self
            .execution_backend
            .execute_turn(
                PreparedHttpTurn::from_turn(turn, &self.default_model),
                observer,
                on_first_token,
            )
            .await?
            .outcome)
    }

    async fn dispatch_turn_streaming(
        &self,
        turn: TurnToSend,
        observer: &dyn RequestObserver,
        on_first_token: &dyn Fn(i64),
        responses: &dyn TurnResponseObserver,
    ) -> Result<TurnDispatchOutcome> {
        Ok(self
            .execution_backend
            .execute_turn_streaming(
                PreparedHttpTurn::from_turn(turn, &self.default_model),
                observer,
                on_first_token,
                responses,
            )
            .await?
            .outcome)
    }
}

fn evaluation_host_capabilities() -> Result<(EvaluationHostCapabilityInventory, String, String)> {
    let capabilities = STOCK_EVALUATION_OPERATION_SCHEMAS
        .iter()
        .map(|schema| {
            Ok((
                HostCapabilityId::new(format!("inference.{}.v1", schema.operation_id))?,
                schema.combined_schema_sha256.to_string(),
            ))
        })
        .collect::<std::result::Result<Vec<_>, aiperf_accuracy::EvaluationProtocolError>>()
        .map_err(|error| anyhow!(error.to_string()))?;
    let capability_inventory_sha256 = canonical_sha256(json!(
        capabilities
            .iter()
            .map(|(id, schema)| json!({"capability_id": id, "schema_sha256": schema}))
            .collect::<Vec<_>>()
    ))?;
    let schema_inventory_sha256 = canonical_sha256(json!(
        STOCK_EVALUATION_OPERATION_SCHEMAS
            .iter()
            .map(|schema| json!({
                "endpoint_capability": schema.endpoint_capability,
                "operation_id": schema.operation_id,
                "request_schema_sha256": schema.request_schema_sha256,
                "response_schema_sha256": schema.response_schema_sha256,
                "stream_schema_sha256": schema.true_streaming.then_some(schema.canonical_stream_schema_sha256),
            }))
            .collect::<Vec<_>>()
    ))?;
    Ok((
        EvaluationHostCapabilityInventory::new(capabilities)?,
        capability_inventory_sha256,
        schema_inventory_sha256,
    ))
}

fn current_executable_sha256() -> Result<String> {
    let path = std::env::current_exe().context("resolving current runner executable")?;
    let bytes = std::fs::read(&path)
        .with_context(|| format!("reading current runner executable {}", path.display()))?;
    Ok(artifact_content_sha256(&bytes))
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use aiperf::evaluation::host::{HostOperationFamily, RegisteredOperationId};
    use aiperf_accuracy::{
        EvaluationDistributionDescriptor, EvaluationHostBinding, EvaluationIdentityComponent,
        EvaluationProvider, EvaluationProviderDescriptor, EvaluationProviderError,
        EvaluationProviderLauncher, EvaluationProviderRegistryBuilder,
        NemoEvaluatorProviderFactory, PreparedEvaluationProviderLaunch, ProviderLaunchContext,
    };
    use async_trait::async_trait;

    use crate::readiness::{
        ReadinessAttemptRequest, ReadinessAttemptResponse, ReadinessReport, ReadinessTransport,
    };

    use super::*;

    #[derive(Debug)]
    struct OrderingReadiness {
        artifact_target: PathBuf,
        waited: Rc<Cell<bool>>,
    }

    #[async_trait(?Send)]
    impl PreparedOnlineReadiness for OrderingReadiness {
        fn is_empty(&self) -> bool {
            false
        }

        fn target_count(&self) -> usize {
            1
        }

        async fn wait(
            &self,
            _clock: Rc<dyn Clock>,
            _transport: Rc<dyn ReadinessTransport>,
        ) -> Result<ReadinessReport> {
            ensure!(
                !self.artifact_target.exists(),
                "artifact target existed before evaluator readiness"
            );
            self.waited.set(true);
            Ok(ReadinessReport {
                targets_ready: 1,
                attempts: 1,
            })
        }
    }

    #[derive(Debug)]
    struct UnusedReadinessTransport;

    #[async_trait(?Send)]
    impl ReadinessTransport for UnusedReadinessTransport {
        async fn execute(&self, _request: ReadinessAttemptRequest) -> ReadinessAttemptResponse {
            unreachable!("ordering readiness does not issue a transport attempt")
        }
    }

    #[derive(Debug)]
    struct UnusedReadinessTransportFactory;

    impl ReadinessTransportFactory for UnusedReadinessTransportFactory {
        fn build(&self, _clock: Rc<dyn Clock>) -> Rc<dyn ReadinessTransport> {
            Rc::new(UnusedReadinessTransport)
        }
    }

    struct NeverLaunch;

    #[async_trait(?Send)]
    impl EvaluationProviderLauncher for NeverLaunch {
        fn check_distribution_available(
            &self,
            _distribution: &EvaluationDistributionDescriptor,
        ) -> std::result::Result<(), EvaluationProviderError> {
            Ok(())
        }

        fn prepare_launch(
            &self,
            _descriptor: &EvaluationProviderDescriptor,
            _distribution: &EvaluationDistributionDescriptor,
            _context: ProviderLaunchContext,
        ) -> std::result::Result<Box<dyn PreparedEvaluationProviderLaunch>, EvaluationProviderError>
        {
            unreachable!("side-effect-free workload validation cannot prepare a provider")
        }

        async fn launch(
            &self,
            _descriptor: &EvaluationProviderDescriptor,
            _config: &ValidatedProviderConfig,
            _host_binding: EvaluationHostBinding,
            _prepared: Box<dyn PreparedEvaluationProviderLaunch>,
        ) -> std::result::Result<Box<dyn EvaluationProvider>, EvaluationProviderError> {
            unreachable!("side-effect-free workload validation cannot launch a provider")
        }
    }

    struct UnavailableLaunch;

    #[async_trait(?Send)]
    impl EvaluationProviderLauncher for UnavailableLaunch {
        fn check_distribution_available(
            &self,
            _distribution: &EvaluationDistributionDescriptor,
        ) -> std::result::Result<(), EvaluationProviderError> {
            Err(EvaluationProviderError::Launch(
                "fixture isolation mechanism is unavailable".to_owned(),
            ))
        }

        fn prepare_launch(
            &self,
            _descriptor: &EvaluationProviderDescriptor,
            _distribution: &EvaluationDistributionDescriptor,
            _context: ProviderLaunchContext,
        ) -> std::result::Result<Box<dyn PreparedEvaluationProviderLaunch>, EvaluationProviderError>
        {
            unreachable!("an unavailable distribution cannot prepare")
        }

        async fn launch(
            &self,
            _descriptor: &EvaluationProviderDescriptor,
            _config: &ValidatedProviderConfig,
            _host_binding: EvaluationHostBinding,
            _prepared: Box<dyn PreparedEvaluationProviderLaunch>,
        ) -> std::result::Result<Box<dyn EvaluationProvider>, EvaluationProviderError> {
            unreachable!("an unavailable distribution cannot launch")
        }
    }

    fn providers() -> Arc<EvaluationProviderRegistry> {
        providers_with_launcher(Arc::new(NeverLaunch))
    }

    fn providers_with_launcher(
        launcher: Arc<dyn EvaluationProviderLauncher>,
    ) -> Arc<EvaluationProviderRegistry> {
        let digest = "a".repeat(64);
        let distribution = EvaluationDistributionDescriptor {
            distribution_id: EvaluationDistributionId::new("nvidia_nemo_evaluator_0_4_locked")
                .unwrap(),
            package: "nemo-evaluator".to_owned(),
            package_version: "0.4.0".to_owned(),
            provider_source_sha256: digest.clone(),
            worker_source_sha256: digest.clone(),
            dependency_lock_sha256: digest.clone(),
            identity_components: vec![EvaluationIdentityComponent {
                name: "fixture-worker".to_owned(),
                version: "1".to_owned(),
                source_sha256: digest.clone(),
                source_commit: None,
                base_source_sha256: None,
                overlay_policy: None,
                overlays: Vec::new(),
            }],
            oci_digest: None,
            launch_closure_sha256: digest,
        };
        let factory = NemoEvaluatorProviderFactory::new(vec![distribution], launcher).unwrap();
        let mut builder = EvaluationProviderRegistryBuilder::new();
        builder.register(Arc::new(factory)).unwrap();
        Arc::new(builder.freeze().unwrap())
    }

    fn decode(value: Value) -> Result<EvaluationWorkloadConfigV2> {
        Ok(serde_json::from_value(value)?)
    }

    fn authored() -> Value {
        serde_json::json!({
            "provider": {
                "type": "nemo_evaluator",
                "distribution": "nvidia_nemo_evaluator_0_4_locked"
            },
            "evaluation": {
                "environment": "gsm8k",
                "solver": "chat",
                "solver_config": {"max_tokens": 64},
                "selection": {"limit": 1, "seed": 0}
            },
            "routes": {
                "primary": {"model": "candidate", "endpoint_profile": "candidate"},
                "judge": {"model": "judge", "endpoint_profile": "judge_anthropic", "purpose": "judge"}
            },
            "resources": {
                "workspace": {"type": "contained_directory", "config": {"quota": 4096}}
            },
            "unit_concurrency": 2
        })
    }

    fn matching_generate_descriptor(
        providers: &EvaluationProviderRegistry,
    ) -> HostOperationDescriptor {
        let operation = providers
            .descriptors()
            .next()
            .expect("fixture provider")
            .operations
            .iter()
            .find(|operation| operation.operation_id.as_str() == "model.generate")
            .expect("stock provider declares model.generate");
        HostOperationDescriptor {
            operation_id: RegisteredOperationId::new(operation.operation_id.as_str()).unwrap(),
            family: HostOperationFamily::new("inference").unwrap(),
            request_schema_fingerprint: operation.input_schema_sha256.clone(),
            response_schema_fingerprint: operation.output_schema_sha256.clone(),
            stream_schema_fingerprint: operation.stream_schema_sha256.clone(),
            true_streaming: operation.stream_schema_sha256.is_some(),
            max_request_bytes: 1024,
            max_response_bytes: 1024,
            endpoint_capabilities: operation.endpoint_capabilities.iter().cloned().collect(),
        }
    }

    #[test]
    fn strict_shape_retains_only_identities_and_opaque_factory_objects() {
        let config = decode(authored()).unwrap();
        config.validate_structure().unwrap();
        assert_eq!(config.provider.provider_id.as_str(), "nemo_evaluator");
        assert_eq!(
            config.provider.distribution.as_str(),
            "nvidia_nemo_evaluator_0_4_locked"
        );
        assert_eq!(config.routes.len(), 2);
        assert_eq!(
            config.routes[&LogicalServiceId::new("primary").unwrap()]
                .purpose
                .as_str(),
            "primary"
        );
        assert_eq!(
            config.routes[&LogicalServiceId::new("judge").unwrap()]
                .purpose
                .as_str(),
            "judge"
        );
        assert_eq!(config.resources.len(), 1);
        let debug = format!("{config:?}");
        assert!(!debug.contains("fixture/exact@locked"));
        assert!(!debug.contains("quota"));
    }

    #[test]
    fn authored_launch_coordinates_fail_outer_decode() {
        for field in ["executable", "module", "argv", "environment", "current_dir"] {
            let mut value = authored();
            value["provider"][field] = Value::String("attacker-controlled".to_owned());
            let error = decode(value).unwrap_err().to_string();
            assert!(error.contains("unknown field"), "{field}: {error}");
        }
    }

    #[test]
    fn route_resource_and_provider_objects_fail_closed() {
        let mut empty_routes = authored();
        empty_routes["routes"] = serde_json::json!({});
        assert!(
            decode(empty_routes)
                .unwrap()
                .validate_structure()
                .unwrap_err()
                .to_string()
                .contains("at least one")
        );

        let mut non_object = authored();
        non_object["evaluation"] = serde_json::json!(["not", "an", "object"]);
        assert!(
            decode(non_object)
                .unwrap()
                .validate_structure()
                .unwrap_err()
                .to_string()
                .contains("JSON object")
        );

        let mut unknown_route = authored();
        unknown_route["routes"]["primary"]["url"] =
            Value::String("https://forbidden.invalid".to_owned());
        assert!(decode(unknown_route).is_err());

        let mut untyped_resource = authored();
        untyped_resource["resources"]["workspace"] = serde_json::json!({"config": {}});
        assert!(decode(untyped_resource).is_err());
    }

    #[test]
    fn compatibility_policy_grants_only_the_manifest_route() {
        let grant = EvaluationCompatibilityRouteGrant {
            dialect_id: "openai_chat_completions".to_string(),
            selector: "candidate".to_string(),
            service_id: LogicalServiceId::new("candidate").unwrap(),
            purpose: OperationPurpose::new("primary").unwrap(),
            semantic_operation_id: SemanticOperationId::new("model.generate").unwrap(),
            restricted_payload: false,
        };
        assert!(!grant.to_proxy_route().unwrap().carries_restricted_payload());
        let mut restricted = grant.clone();
        restricted.restricted_payload = true;
        assert!(
            restricted
                .to_proxy_route()
                .unwrap()
                .carries_restricted_payload()
        );
        let policy = stock_evaluation_compatibility_policy([EvaluationCompatibilityPolicyEntry {
            provider_id: EvaluationProviderId::new("openbench").unwrap(),
            distribution_id: EvaluationDistributionId::new("openbench-locked").unwrap(),
            dialect_ids: vec!["openai_chat_completions".to_string()],
            route_grants: vec![grant],
            grant_limits: Some(EvaluationCompatibilityGrantLimits {
                max_operations: 1,
                max_concurrent_operations: 1,
                max_request_bytes: 1024,
                max_response_bytes: 1024,
                max_stream_events: 1,
                expires_after_ms: 1000,
            }),
        }])
        .unwrap();
        let routes = EvaluationRouteTable::new([
            EvaluationRoute {
                service_id: "candidate".to_string(),
                purpose: "primary".to_string(),
                model: "server-candidate".to_string(),
                endpoint_profile: "candidate".to_string(),
                prepared_identity_sha256: "a".repeat(64),
                endpoint_capabilities: BTreeSet::from(["chat".to_string()]),
            },
            EvaluationRoute {
                service_id: "judge".to_string(),
                purpose: "judge".to_string(),
                model: "server-judge".to_string(),
                endpoint_profile: "judge".to_string(),
                prepared_identity_sha256: "b".repeat(64),
                endpoint_capabilities: BTreeSet::from(["completion".to_string()]),
            },
        ])
        .unwrap();
        let prepared = policy
            .prepare(
                &EvaluationProviderSelectionV2 {
                    provider_id: EvaluationProviderId::new("openbench").unwrap(),
                    distribution: EvaluationDistributionId::new("openbench-locked").unwrap(),
                },
                &routes,
            )
            .unwrap()
            .unwrap();
        let granted = prepared.dialects.routes();
        assert_eq!(granted.len(), 1);
        assert_eq!(granted[0].selector, "candidate");
        assert_eq!(granted[0].service_id.as_str(), "candidate");
        assert_eq!(prepared.dialect_ids, ["openai_chat_completions"]);
        assert_eq!(prepared.descriptor_sha256.len(), 64);
    }

    #[test]
    fn workload_factory_uses_the_selected_provider_schema_without_effects() {
        let factory = EvaluationRunnerWorkloadFactory::new(providers());
        let raw = RawValue::from_string(authored().to_string()).unwrap();
        let validated = factory.validate(&raw).unwrap();
        let validated = validated_evaluation_workload(validated.as_ref()).unwrap();
        assert_eq!(
            validated.provider_config().provider_id().as_str(),
            "nemo_evaluator"
        );
        assert_eq!(validated.provider_config().schema_version(), 1);
    }

    #[test]
    fn workload_structurally_forbids_generic_sidecars() {
        let factory = EvaluationRunnerWorkloadFactory::new(providers());
        let raw = RawValue::from_string(authored().to_string()).unwrap();
        let validated = factory.validate(&raw).unwrap();
        let requirements = factory.requirements(validated.as_ref()).unwrap();
        assert_eq!(
            requirements.resources.sidecars,
            ResourceRequirementV2::Forbidden
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn readiness_completes_before_evaluation_artifact_creation() {
        let temporary = tempfile::tempdir().unwrap();
        let artifact_target = temporary.path().join("run");
        let waited = Rc::new(Cell::new(false));
        let readiness = OrderingReadiness {
            artifact_target: artifact_target.clone(),
            waited: waited.clone(),
        };
        let clock: Rc<dyn Clock> = RealClock::from_anchor(RealClockAnchor::now());

        let roots = prepare_evaluation_artifact_roots(
            &artifact_target,
            &readiness,
            &UnusedReadinessTransportFactory,
            clock,
        )
        .await
        .unwrap();

        assert!(waited.get());
        assert!(roots.staging.is_dir());
        assert!(!roots.promoted.exists());
    }

    #[test]
    fn workload_factory_rejects_secret_authority_duplicate_keys_and_distribution_drift() {
        let factory = EvaluationRunnerWorkloadFactory::new(providers());

        let mut secret = authored();
        secret["evaluation"]["base_url"] = Value::String("https://forbidden.invalid".to_owned());
        let raw = RawValue::from_string(secret.to_string()).unwrap();
        let error = factory.validate(&raw).unwrap_err().to_string();
        assert!(error.contains("connection authority"), "{error}");
        assert!(!error.contains("forbidden.invalid"), "{error}");

        let duplicate = RawValue::from_string(
            r#"{
                "provider":{"type":"nemo_evaluator","distribution":"nvidia_nemo_evaluator_0_4_locked"},
                "evaluation":{"environment":"gsm8k","environment":"gsm8k","solver":"chat","solver_config":{"max_tokens":64},"selection":{"limit":1,"seed":0}},
                "routes":{"primary":{"model":"candidate","endpoint_profile":"candidate"}},
                "resources":{},
                "unit_concurrency":1
            }"#
            .to_owned(),
        )
        .unwrap();
        let error = factory.validate(&duplicate).unwrap_err().to_string();
        assert!(error.contains("duplicate"), "{error}");

        let mut drift = authored();
        drift["provider"]["distribution"] = Value::String("unregistered_lock".to_owned());
        let raw = RawValue::from_string(drift.to_string()).unwrap();
        let error = factory.validate(&raw).unwrap_err().to_string();
        assert!(error.contains("unknown evaluator distribution"), "{error}");
    }

    #[test]
    fn capabilities_publish_only_linked_operations_and_proven_distributions() {
        let providers = providers();
        let generate = matching_generate_descriptor(providers.as_ref());
        let unavailable = build_evaluation_capability_inventory(
            providers.as_ref(),
            [&generate],
            std::iter::empty(),
            "strict_process_tree_v1",
            false,
        )
        .unwrap();
        assert_eq!(unavailable.providers.len(), 1);
        assert_eq!(unavailable.host_operations.len(), 1);
        assert!(unavailable.supported_combinations.is_empty());

        let available = build_evaluation_capability_inventory(
            providers.as_ref(),
            [&generate],
            ["contained_directory"],
            "strict_process_tree_v1",
            true,
        )
        .unwrap();
        let combination = &available.supported_combinations[0];
        assert_eq!(combination.operations, ["model.generate"]);
        assert_eq!(combination.resources, ["contained_directory"]);
        assert_eq!(combination.distribution, "nvidia_nemo_evaluator_0_4_locked");
        assert!(
            !combination
                .operations
                .contains(&"model.complete".to_owned())
        );
        assert_eq!(
            available.host_operations[0].stream_schema_sha256,
            generate.stream_schema_fingerprint
        );
    }

    #[test]
    fn capabilities_reject_id_only_or_partial_schema_matches() {
        let providers = providers();
        let matching = matching_generate_descriptor(providers.as_ref());

        let mut mismatches = Vec::new();
        let mut request = matching.clone();
        request.request_schema_fingerprint = "1".repeat(64);
        mismatches.push(request);
        let mut response = matching.clone();
        response.response_schema_fingerprint = "2".repeat(64);
        mismatches.push(response);
        let mut stream = matching.clone();
        stream.stream_schema_fingerprint = Some("3".repeat(64));
        mismatches.push(stream);
        let mut endpoint_capabilities = matching;
        endpoint_capabilities.endpoint_capabilities = BTreeSet::from(["embedding".to_owned()]);
        mismatches.push(endpoint_capabilities);

        for mismatch in &mismatches {
            let inventory = build_evaluation_capability_inventory(
                providers.as_ref(),
                [mismatch],
                std::iter::empty(),
                "strict_process_tree_v1",
                true,
            )
            .unwrap();
            assert!(
                inventory.supported_combinations.is_empty(),
                "ID-only match overclaimed executable capability for {mismatch:?}"
            );
        }
    }

    #[test]
    fn capabilities_omit_unattested_provider_distributions() {
        let providers = providers_with_launcher(Arc::new(UnavailableLaunch));
        let inventory = build_evaluation_capability_inventory(
            providers.as_ref(),
            std::iter::empty(),
            std::iter::empty(),
            "strict_process_tree_v1",
            true,
        )
        .unwrap();

        assert!(inventory.providers.is_empty());
        assert!(inventory.supported_combinations.is_empty());
        assert!(
            providers
                .available_distributions(&EvaluationProviderId::new("nemo_evaluator").unwrap())
                .is_empty()
        );
    }
}
