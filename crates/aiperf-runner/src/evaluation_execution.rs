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
use std::sync::Arc;

use aiperf::evaluation::host::HostOperationDescriptor;
use aiperf_accuracy::{
    EvaluationDistributionId, EvaluationExecutionGranularity, EvaluationProviderId,
    EvaluationProviderRegistry, EvaluationSchedulingMode, HostCapabilityId, LogicalServiceId,
    ValidatedProviderConfig,
};
use anyhow::{Result, anyhow, ensure};
use serde::Deserialize;
use serde_json::{Value, value::RawValue};

use crate::protocol::{
    EvaluationCapabilityInventory, EvaluationDistributionCapability,
    EvaluationHostOperationCapability, EvaluationProviderCapability,
    SupportedEvaluationCombination,
};
use crate::protocol_v2::RunnerComponentId;
use crate::registry::{
    RunnerClockKind, RunnerRegistryBuilder, RunnerWorkloadDescriptor, RunnerWorkloadFactory,
    ValidatedWorkloadConfig, WorkloadRequirements,
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
        let distributions = descriptor
            .distributions
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
            for distribution in &descriptor.distributions {
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

#[cfg(test)]
mod tests {
    use aiperf::evaluation::host::{HostOperationFamily, RegisteredOperationId};
    use aiperf_accuracy::{
        EvaluationDistributionDescriptor, EvaluationProvider, EvaluationProviderDescriptor,
        EvaluationProviderError, EvaluationProviderLauncher, EvaluationProviderRegistryBuilder,
        NemoEvaluatorProviderFactory, ProviderLaunchContext,
    };
    use async_trait::async_trait;

    use super::*;

    struct NeverLaunch;

    #[async_trait(?Send)]
    impl EvaluationProviderLauncher for NeverLaunch {
        async fn launch(
            &self,
            _descriptor: &EvaluationProviderDescriptor,
            _distribution: &EvaluationDistributionDescriptor,
            _config: &ValidatedProviderConfig,
            _context: &ProviderLaunchContext,
        ) -> std::result::Result<Box<dyn EvaluationProvider>, EvaluationProviderError> {
            unreachable!("side-effect-free workload validation cannot launch a provider")
        }
    }

    fn providers() -> Arc<EvaluationProviderRegistry> {
        let digest = "a".repeat(64);
        let distribution = EvaluationDistributionDescriptor {
            distribution_id: EvaluationDistributionId::new("nvidia_nemo_evaluator_0_4_locked")
                .unwrap(),
            package: "nemo-evaluator".to_owned(),
            package_version: "0.4.0".to_owned(),
            provider_source_sha256: digest.clone(),
            worker_source_sha256: digest.clone(),
            dependency_lock_sha256: digest.clone(),
            oci_digest: None,
            launch_closure_sha256: digest,
        };
        let factory =
            NemoEvaluatorProviderFactory::new(vec![distribution], Arc::new(NeverLaunch)).unwrap();
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
                "benchmark": "fixture/exact@locked",
                "provider_options": {"opaque": true}
            },
            "routes": {
                "primary": {"model": "candidate", "endpoint_profile": "candidate"},
                "judge": {"model": "judge", "endpoint_profile": "judge_anthropic"}
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
                "evaluation":{"benchmark":"first","benchmark":"second"},
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
        endpoint_capabilities.endpoint_capabilities = BTreeSet::from(["model.complete".to_owned()]);
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
}
