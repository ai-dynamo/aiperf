// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

//! Strict native normalization for the executable Harbor task package contract.

use std::{
    collections::{BTreeMap, BTreeSet},
    path::{Component, Path},
    time::Duration,
};

use serde::Deserialize;
use tempfile::TempDir;

use crate::eval::{
    ArtifactDigest, ArtifactSpec, BenchmarkExecutionPlan, BenchmarkStepPlan, ComposeProjectPlan,
    ComposeServiceName, ContainerResources, EnvBinding, EnvironmentPlan, EvalExecutionError,
    HealthcheckPlan, ImageSource, MultiStepRewardStrategy, NativeGraphPackagePlan, NetworkPolicy,
    PhasePlan, VerifierCollectHook, VerifierMode, VerifierPlan,
    artifact_source_overlaps_reserved_verifier_path,
    shared_workdir_conflicts_reserved_verifier_path, verifier_artifact_target_collision,
};

use super::{AcquiredSource, HarborImportError, source_snapshot::ExecutableSourceView};
use crate::eval::native_graph::{
    NativeGraphPackageDraft, NativeGraphSectionDto, resolve_native_graph_package,
};

/// Executable material retained from one strict Harbor task package.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HarborTaskPackage {
    id: String,
    instruction: String,
    environment: String,
    verifier: String,
    agent_command: Vec<String>,
    verifier_command: Vec<String>,
    verifier_mode: VerifierMode,
    declared_artifacts: Vec<String>,
    identity_digest: ArtifactDigest,
    source: AcquiredSource,
    is_standard_directory: bool,
    container_resources: Option<(u64, u64)>,
    timeouts: Option<(Duration, Duration)>,
    execution_plan: BenchmarkExecutionPlan,
    native_graph: Option<NativeGraphPackagePlan>,
}

impl HarborTaskPackage {
    /// Returns the authored task identifier.
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Returns the authored instruction presented to the agent.
    pub fn instruction(&self) -> &str {
        &self.instruction
    }

    /// Returns the immutable environment artifact identity.
    pub fn environment(&self) -> &str {
        &self.environment
    }

    /// Returns the immutable verifier artifact identity.
    pub fn verifier(&self) -> &str {
        &self.verifier
    }

    /// Returns the exact argv used to invoke the agent.
    pub fn agent_command(&self) -> &[String] {
        &self.agent_command
    }

    /// Returns the exact argv used to invoke the verifier.
    pub fn verifier_command(&self) -> &[String] {
        &self.verifier_command
    }

    /// Returns the task-authored verifier topology.
    pub const fn verifier_mode(&self) -> VerifierMode {
        self.verifier_mode
    }

    /// Returns normalized absolute artifact paths in authored order.
    pub fn declared_artifacts(&self) -> &[String] {
        &self.declared_artifacts
    }

    /// Returns the digest of the complete authored package bytes.
    pub fn source_digest(&self) -> ArtifactDigest {
        self.source.source_digest()
    }

    /// Returns the immutable, exactly acquired package bytes.
    pub fn source_bytes(&self) -> &[u8] {
        self.source.primary_bytes()
    }

    /// Returns the canonical normalized package identity used by the task reference.
    pub fn identity_digest(&self) -> ArtifactDigest {
        self.identity_digest.clone()
    }

    /// Reports whether this package originated from a standard task directory.
    pub const fn is_standard_directory(&self) -> bool {
        self.is_standard_directory
    }

    /// Returns authored CPU and memory limits for a standard task container.
    pub const fn container_resources(&self) -> Option<(u64, u64)> {
        self.container_resources
    }

    /// Returns authored agent and verifier execution timeouts, when configured.
    pub const fn timeouts(&self) -> Option<(Duration, Duration)> {
        self.timeouts
    }

    /// Returns the immutable execution plan compiled during import.
    pub fn execution_plan(&self) -> &BenchmarkExecutionPlan {
        &self.execution_plan
    }

    /// Returns the resolved schema-1.1 NativeGraph plan when this task selects it.
    pub fn native_graph(&self) -> Option<&NativeGraphPackagePlan> {
        self.native_graph.as_ref()
    }

    pub(crate) fn materialize_source(&self) -> Result<MaterializedSource, EvalExecutionError> {
        let lease = tempfile::tempdir()
            .map_err(|error| EvalExecutionError::Materialization(error.to_string()))?;
        self.materialize_source_into(lease.path())?;
        Ok(MaterializedSource { lease })
    }

    pub(crate) fn materialize_source_into(
        &self,
        destination: &Path,
    ) -> Result<(), EvalExecutionError> {
        self.source
            .materialize_into(destination)
            .map_err(|error| EvalExecutionError::Materialization(error.to_string()))
    }
}

pub(crate) struct MaterializedSource {
    lease: TempDir,
}

impl MaterializedSource {
    pub(crate) fn root(&self) -> &Path {
        self.lease.path()
    }
}

pub(super) struct NormalizedPackageDraft {
    id: String,
    instruction: String,
    environment: String,
    verifier: String,
    agent_command: Vec<String>,
    verifier_command: Vec<String>,
    verifier_mode: VerifierMode,
    declared_artifacts: Vec<String>,
    is_standard_directory: bool,
    container_resources: Option<(u64, u64)>,
    timeouts: Option<(Duration, Duration)>,
    execution_plan: BenchmarkExecutionPlan,
    native_graph: Option<NativeGraphPackagePlan>,
}

impl NormalizedPackageDraft {
    pub(super) fn id(&self) -> &str {
        &self.id
    }

    pub(super) fn agent_command(&self) -> &[String] {
        &self.agent_command
    }

    pub(super) fn verifier_command(&self) -> &[String] {
        &self.verifier_command
    }

    pub(super) fn execution_plan(&self) -> &BenchmarkExecutionPlan {
        &self.execution_plan
    }

    pub(super) fn native_graph(&self) -> Option<&NativeGraphPackagePlan> {
        self.native_graph.as_ref()
    }

    pub(super) fn into_package(
        self,
        source: AcquiredSource,
        identity_digest: ArtifactDigest,
    ) -> HarborTaskPackage {
        HarborTaskPackage {
            id: self.id,
            instruction: self.instruction,
            environment: self.environment,
            verifier: self.verifier,
            agent_command: self.agent_command,
            verifier_command: self.verifier_command,
            verifier_mode: self.verifier_mode,
            declared_artifacts: self.declared_artifacts,
            identity_digest,
            source,
            is_standard_directory: self.is_standard_directory,
            container_resources: self.container_resources,
            timeouts: self.timeouts,
            execution_plan: self.execution_plan,
            native_graph: self.native_graph,
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct PackageTaskDto {
    id: String,
    instruction: String,
    environment: String,
    verifier: String,
    agent_command: Vec<String>,
    verifier_command: Vec<String>,
    declared_artifacts: Vec<String>,
}

pub(super) fn normalize(
    source: &AcquiredSource,
) -> Result<(NormalizedPackageDraft, ExecutableSourceView), HarborImportError> {
    let bytes = source.primary_bytes();
    let task = serde_json::from_slice::<PackageTaskDto>(bytes)
        .map_err(|error| HarborImportError::InvalidPackage(error.to_string()))?;
    if task.instruction.trim().is_empty() {
        return Err(HarborImportError::InvalidPackage(
            "instruction must not be empty".to_owned(),
        ));
    }
    validate_command("agent_command", &task.agent_command)?;
    validate_command("verifier_command", &task.verifier_command)?;
    let declared_artifacts = normalize_declared_artifacts(task.declared_artifacts)?;
    let environment = ArtifactDigest::parse(task.environment.clone())
        .map_err(|error| HarborImportError::InvalidPackage(error.to_string()))?;
    ArtifactDigest::parse(task.verifier.clone())
        .map_err(|error| HarborImportError::InvalidPackage(error.to_string()))?;
    let environment_plan = EnvironmentPlan {
        image_source: ImageSource::legacy_artifact(environment.clone()),
        resources: None,
        workdir: None,
        user: None,
        env: BTreeMap::new(),
        network: NetworkPolicy::public(),
        healthcheck: None,
    };
    let agent = PhasePlan {
        user: None,
        env: BTreeMap::new(),
        network: NetworkPolicy::public(),
        timeout: None,
    };
    let verifier = VerifierPlan {
        phase: PhasePlan {
            user: None,
            env: BTreeMap::new(),
            network: NetworkPolicy::public(),
            timeout: None,
        },
        mode: VerifierMode::Separate,
        environment: environment_plan.clone(),
    };
    let artifacts = declared_artifacts
        .iter()
        .cloned()
        .map(ArtifactSpec::exact_file)
        .collect::<Vec<_>>();
    let execution_plan = BenchmarkExecutionPlan {
        environment: environment_plan.clone(),
        agent: agent.clone(),
        verifier: verifier.clone(),
        artifacts: artifacts.clone(),
        compose: None,
        steps: vec![BenchmarkStepPlan::new(
            "default".to_owned(),
            task.instruction.clone(),
            "tests".to_owned(),
            agent,
            verifier,
            artifacts,
        )],
        has_explicit_steps: false,
        multi_step_reward_strategy: None,
    };
    let draft = NormalizedPackageDraft {
        id: task.id,
        instruction: task.instruction,
        environment: task.environment,
        verifier: task.verifier,
        agent_command: task.agent_command,
        verifier_command: task.verifier_command,
        verifier_mode: VerifierMode::Separate,
        declared_artifacts,
        is_standard_directory: false,
        container_resources: None,
        timeouts: None,
        execution_plan,
        native_graph: None,
    };
    let view = if source.is_tree() {
        ExecutableSourceView::WholeTree
    } else {
        ExecutableSourceView::PrimaryFile
    };
    Ok((draft, view))
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct StandardTaskManifest {
    schema_version: String,
    task: StandardTaskSection,
    agent: Option<StandardAgentSection>,
    verifier: Option<StandardVerifierSection>,
    #[serde(default)]
    artifacts: Vec<StandardArtifactDto>,
    environment: Option<StandardEnvironmentSection>,
    #[serde(default)]
    steps: Vec<StandardStepSection>,
    multi_step_reward_strategy: Option<String>,
    native_graph: Option<NativeGraphSectionDto>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct StandardTaskSection {
    name: String,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct StandardAgentSection {
    timeout_sec: Option<f64>,
    user: Option<String>,
    network: Option<String>,
    #[serde(default)]
    allowed_hosts: Vec<String>,
    #[serde(default)]
    env: BTreeMap<String, String>,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct StandardVerifierSection {
    environment_mode: Option<String>,
    timeout_sec: Option<f64>,
    user: Option<String>,
    network: Option<String>,
    #[serde(default)]
    allowed_hosts: Vec<String>,
    #[serde(default)]
    env: BTreeMap<String, String>,
    environment: Option<StandardEnvironmentSection>,
    collection_timeout_sec: Option<f64>,
    #[serde(default)]
    collect: Vec<StandardVerifierCollectSection>,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct StandardEnvironmentSection {
    cpus: Option<u64>,
    memory_mb: Option<u64>,
    workdir: Option<String>,
    user: Option<String>,
    network: Option<String>,
    #[serde(default)]
    allowed_hosts: Vec<String>,
    #[serde(default)]
    env: BTreeMap<String, String>,
    healthcheck: Option<StandardHealthcheckSection>,
    build_timeout_sec: Option<f64>,
    startup_timeout_sec: Option<f64>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(untagged)]
enum StandardArtifactDto {
    ExactFile(String),
    Collected(StandardArtifactTable),
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct StandardArtifactTable {
    source: String,
    destination: Option<String>,
    #[serde(default)]
    exclude: Vec<String>,
    service: Option<String>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct StandardVerifierCollectSection {
    service: Option<String>,
    command: Vec<String>,
    timeout_sec: Option<f64>,
    user: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct StandardStepSection {
    name: String,
    #[serde(default)]
    artifacts: Vec<StandardArtifactDto>,
    agent: Option<StandardAgentSection>,
    verifier: Option<StandardVerifierSection>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct StandardHealthcheckSection {
    command: Vec<String>,
    start_period_sec: Option<f64>,
    start_interval_sec: Option<f64>,
    interval_sec: Option<f64>,
    timeout_sec: Option<f64>,
    retries: Option<u32>,
}

/// Normalizes a standard task directory without executing its contents.
pub(super) fn normalize_standard_directory(
    source: &AcquiredSource,
) -> Result<(NormalizedPackageDraft, ExecutableSourceView), HarborImportError> {
    let manifest_bytes = source.primary_bytes();
    let manifest_value = std::str::from_utf8(manifest_bytes)
        .map_err(|error| HarborImportError::InvalidPackage(error.to_string()))?
        .parse::<toml::Value>()
        .map_err(|error| HarborImportError::InvalidPackage(error.to_string()))?;
    let manifest = manifest_value
        .try_into::<StandardTaskManifest>()
        .map_err(|error| HarborImportError::InvalidPackage(error.to_string()))?;
    if !matches!(manifest.schema_version.as_str(), "1.0" | "1.1") {
        return Err(HarborImportError::InvalidPackage(format!(
            "unsupported task schema version {:?}",
            manifest.schema_version
        )));
    }
    if manifest.task.name.trim().is_empty() {
        return Err(HarborImportError::InvalidPackage(
            "task.name must not be empty".to_owned(),
        ));
    }
    let native_graph = match (manifest.schema_version.as_str(), manifest.native_graph) {
        ("1.0", Some(_)) => {
            return Err(HarborImportError::InvalidPackage(
                "native_graph requires schema_version \"1.1\"".to_owned(),
            ));
        }
        ("1.1", None) => {
            return Err(HarborImportError::InvalidPackage(
                "schema_version \"1.1\" requires native_graph".to_owned(),
            ));
        }
        ("1.1", Some(section)) => {
            if !manifest.steps.is_empty() {
                return Err(HarborImportError::InvalidPackage(
                    "native_graph packages must not declare steps".to_owned(),
                ));
            }
            Some(resolve_native_graph_package(source, section)?)
        }
        ("1.0", None) => None,
        _ => {
            return Err(HarborImportError::InvalidPackage(
                "unsupported task schema version".to_owned(),
            ));
        }
    };
    read_required_source_file(source, "environment/Dockerfile")?;
    let mut environment_section = manifest.environment.unwrap_or_default();
    let compose = normalize_compose_project(source, &environment_section)?;
    environment_section.build_timeout_sec = None;
    environment_section.startup_timeout_sec = None;
    let environment_digest =
        source.executable_source_digest(&ExecutableSourceView::selected_roots(["environment"])?)?;
    let image_source = ImageSource::task_dockerfile(environment_digest.clone());
    let environment = normalize_environment(environment_section, image_source.clone())?;
    let root_agent =
        normalize_agent_phase(manifest.agent.unwrap_or_default(), &environment.network)?;
    let root_verifier_section = manifest.verifier.unwrap_or_default();
    let root_collect_hooks =
        normalize_collect_hooks(&root_verifier_section.collect, compose.as_ref())?;
    let root_collection_timeout = normalize_collection_timeout(
        "verifier.collection_timeout_sec",
        root_verifier_section.collection_timeout_sec,
        Duration::from_secs(120),
    )?;
    let root_verifier = normalize_verifier_plan(root_verifier_section, &environment, image_source)?;
    let root_artifacts = normalize_standard_artifacts(manifest.artifacts, compose.as_ref())?;
    let (steps, has_explicit_steps, multi_step_reward_strategy, verifier_bytes) =
        if manifest.steps.is_empty() {
            if manifest.multi_step_reward_strategy.is_some() {
                return Err(HarborImportError::InvalidPackage(
                    "multi_step_reward_strategy requires explicit steps".to_owned(),
                ));
            }
            let instruction = read_required_source_file(source, "instruction.md")?;
            if instruction.trim().is_empty() {
                return Err(HarborImportError::InvalidPackage(
                    "instruction.md must not be empty".to_owned(),
                ));
            }
            let verifier_bytes = read_required_source_bytes(source, "tests/test.sh")?;
            validate_phase_timeout_pair(&root_agent, root_verifier.phase())?;
            validate_step_artifacts(&root_artifacts, &root_verifier)?;
            validate_compose_step_policy(
                compose.as_ref(),
                &environment,
                &root_agent,
                &root_verifier,
                &root_artifacts,
                &root_collect_hooks,
                true,
            )?;
            (
                vec![
                    BenchmarkStepPlan::new(
                        "default".to_owned(),
                        instruction,
                        "tests".to_owned(),
                        root_agent,
                        root_verifier,
                        root_artifacts,
                    )
                    .with_collection(root_collect_hooks, root_collection_timeout),
                ],
                false,
                None,
                vec![verifier_bytes],
            )
        } else {
            let strategy =
                normalize_multi_step_reward_strategy(manifest.multi_step_reward_strategy)?;
            let mut names = BTreeSet::new();
            let step_count = manifest.steps.len();
            let mut steps = Vec::with_capacity(step_count);
            let mut verifier_bytes = Vec::with_capacity(step_count);
            for (step_index, step) in manifest.steps.into_iter().enumerate() {
                validate_step_name(&step.name)?;
                if !names.insert(step.name.clone()) {
                    return Err(HarborImportError::InvalidPackage(format!(
                        "step name is duplicated: {:?}",
                        step.name
                    )));
                }
                let instruction_path = format!("steps/{}/instruction.md", step.name);
                let instruction = read_required_source_file(source, &instruction_path)?;
                if instruction.trim().is_empty() {
                    return Err(HarborImportError::InvalidPackage(format!(
                        "{instruction_path} must not be empty"
                    )));
                }
                let step_agent = overlay_agent_phase(&root_agent, step.agent.unwrap_or_default())?;
                let step_verifier_section = step.verifier.unwrap_or_default();
                let mut collect_hooks = root_collect_hooks.clone();
                collect_hooks.extend(normalize_collect_hooks(
                    &step_verifier_section.collect,
                    compose.as_ref(),
                )?);
                let collection_timeout = normalize_collection_timeout(
                    "steps.verifier.collection_timeout_sec",
                    step_verifier_section.collection_timeout_sec,
                    root_collection_timeout,
                )?;
                let step_verifier = overlay_verifier_plan(
                    &root_verifier,
                    step_verifier_section,
                    &environment,
                    &environment_digest,
                )?;
                validate_phase_timeout_pair(&step_agent, step_verifier.phase())?;
                let mut artifacts = root_artifacts.clone();
                artifacts.extend(normalize_standard_artifacts(
                    step.artifacts,
                    compose.as_ref(),
                )?);
                validate_step_artifacts(&artifacts, &step_verifier)?;
                validate_compose_step_policy(
                    compose.as_ref(),
                    &environment,
                    &step_agent,
                    &step_verifier,
                    &artifacts,
                    &collect_hooks,
                    step_index + 1 == step_count,
                )?;
                let step_test = format!("steps/{}/tests/test.sh", step.name);
                let (verifier_test_root, bytes) = if source.contains_file(&step_test) {
                    (
                        format!("steps/{}/tests", step.name),
                        read_required_source_bytes(source, &step_test)?,
                    )
                } else {
                    (
                        "tests".to_owned(),
                        read_required_source_bytes(source, "tests/test.sh")?,
                    )
                };
                verifier_bytes.push(bytes);
                steps.push(
                    BenchmarkStepPlan::new(
                        step.name,
                        instruction,
                        verifier_test_root,
                        step_agent,
                        step_verifier,
                        artifacts,
                    )
                    .with_collection(collect_hooks, collection_timeout),
                );
            }
            (steps, true, Some(strategy), verifier_bytes)
        };
    let first_step = steps.first().ok_or_else(|| {
        HarborImportError::InvalidPackage("task must contain a logical step".to_owned())
    })?;
    let instruction = first_step.instruction().to_owned();
    let verifier_digest = ArtifactDigest::from_bytes(&verifier_bytes[0]);
    let verifier_mode = first_step.verifier().mode();
    let agent = first_step.agent().clone();
    let verifier = first_step.verifier().clone();
    let artifacts = first_step.artifacts().to_vec();
    let declared_artifacts = artifacts
        .iter()
        .map(|artifact| artifact.source().to_owned())
        .collect();
    let timeouts = timeout_pair(&agent, verifier.phase())?;
    let container_resources = environment
        .resources
        .map(|resources| (resources.cpus, resources.memory_mb));
    let execution_plan = BenchmarkExecutionPlan {
        environment,
        agent,
        verifier: VerifierPlan {
            phase: verifier.phase().clone(),
            mode: verifier_mode,
            environment: verifier.environment().clone(),
        },
        artifacts,
        compose,
        steps: steps.clone(),
        has_explicit_steps,
        multi_step_reward_strategy,
    };
    validate_shared_verifier_workdir(&execution_plan)?;
    let view = ExecutableSourceView::selected_roots(
        std::iter::once("environment")
            .chain(steps.iter().map(BenchmarkStepPlan::verifier_test_root))
            .chain(
                native_graph
                    .as_ref()
                    .into_iter()
                    .flat_map(NativeGraphPackageDraft::executable_source_paths),
            ),
    )?;
    let native_graph = native_graph
        .map(|native_graph| {
            source
                .executable_source_digest(&view)
                .map(|digest| native_graph.into_plan(digest))
        })
        .transpose()?;
    let draft = NormalizedPackageDraft {
        id: manifest.task.name,
        instruction,
        environment: environment_digest.as_str().to_owned(),
        verifier: verifier_digest.as_str().to_owned(),
        agent_command: vec!["aiperf-task-agent".to_owned()],
        verifier_command: vec!["/bin/sh".to_owned(), "tests/test.sh".to_owned()],
        verifier_mode,
        declared_artifacts,
        is_standard_directory: true,
        container_resources,
        timeouts,
        execution_plan,
        native_graph,
    };
    Ok((draft, view))
}

fn normalize_compose_project(
    source: &AcquiredSource,
    environment: &StandardEnvironmentSection,
) -> Result<Option<ComposeProjectPlan>, HarborImportError> {
    const DEFINITION_PATH: &str = "environment/docker-compose.yaml";
    const ALTERNATE_PATHS: [&str; 3] = [
        "environment/docker-compose.yml",
        "environment/compose.yaml",
        "environment/compose.yml",
    ];

    if let Some(path) = ALTERNATE_PATHS
        .into_iter()
        .find(|path| source.contains_path(path))
    {
        return Err(HarborImportError::InvalidPackage(format!(
            "unsupported Compose filename: {path:?}; expected {DEFINITION_PATH:?}"
        )));
    }
    if !source.contains_path(DEFINITION_PATH) {
        if environment.build_timeout_sec.is_some() || environment.startup_timeout_sec.is_some() {
            return Err(HarborImportError::InvalidPackage(
                "environment build/startup timeouts require environment/docker-compose.yaml"
                    .to_owned(),
            ));
        }
        return Ok(None);
    }

    let bytes = read_required_source_bytes(source, DEFINITION_PATH)?;
    let document = serde_yaml::from_slice::<serde_yaml::Value>(&bytes).map_err(|error| {
        HarborImportError::InvalidPackage(format!("{DEFINITION_PATH}: {error}"))
    })?;
    let document = document.as_mapping().ok_or_else(|| {
        HarborImportError::InvalidPackage(format!("{DEFINITION_PATH} must contain a YAML mapping"))
    })?;
    let services = document
        .get(serde_yaml::Value::String("services".to_owned()))
        .and_then(serde_yaml::Value::as_mapping)
        .filter(|services| !services.is_empty())
        .ok_or_else(|| {
            HarborImportError::InvalidPackage(format!(
                "{DEFINITION_PATH}.services must be a nonempty mapping"
            ))
        })?;

    let mut normalized_services = BTreeSet::new();
    for (name, value) in services {
        let authored_name = name.as_str().ok_or_else(|| {
            HarborImportError::InvalidPackage(format!(
                "{DEFINITION_PATH}.services keys must be strings"
            ))
        })?;
        let service =
            ComposeServiceName::parse(authored_name).map_err(HarborImportError::InvalidPackage)?;
        if !normalized_services.insert(service.clone()) {
            return Err(HarborImportError::InvalidPackage(format!(
                "Compose service name is duplicated after normalization: {:?}",
                service.as_str()
            )));
        }
        if service.as_str() == "main" {
            let main = value.as_mapping().ok_or_else(|| {
                HarborImportError::InvalidPackage(
                    "services.main must be a mapping containing only depends_on".to_owned(),
                )
            })?;
            for key in main.keys() {
                if key.as_str() != Some("depends_on") {
                    return Err(HarborImportError::InvalidPackage(
                        "services.main may contain only depends_on".to_owned(),
                    ));
                }
            }
        }
    }
    normalized_services.insert(ComposeServiceName::main());

    Ok(Some(ComposeProjectPlan {
        definition_path: DEFINITION_PATH.to_owned(),
        services: normalized_services,
        build_timeout: normalize_timeout(
            "environment.build_timeout_sec",
            environment.build_timeout_sec.unwrap_or(600.0),
        )?,
        startup_timeout: normalize_timeout(
            "environment.startup_timeout_sec",
            environment.startup_timeout_sec.unwrap_or(120.0),
        )?,
    }))
}

fn normalize_service_reference(
    service: Option<&str>,
    compose: Option<&ComposeProjectPlan>,
) -> Result<ComposeServiceName, HarborImportError> {
    let service = match service {
        Some(service) => {
            ComposeServiceName::parse(service).map_err(HarborImportError::InvalidPackage)?
        }
        None => ComposeServiceName::main(),
    };
    if service.as_str() != "main" {
        let Some(compose) = compose else {
            return Err(HarborImportError::InvalidPackage(format!(
                "non-main service {:?} requires environment/docker-compose.yaml",
                service.as_str()
            )));
        };
        if !compose.services().contains(&service) {
            return Err(HarborImportError::InvalidPackage(format!(
                "unknown Compose service: {:?}",
                service.as_str()
            )));
        }
    }
    Ok(service)
}

fn normalize_collect_hooks(
    hooks: &[StandardVerifierCollectSection],
    compose: Option<&ComposeProjectPlan>,
) -> Result<Vec<VerifierCollectHook>, HarborImportError> {
    hooks
        .iter()
        .map(|hook| {
            validate_command("verifier.collect.command", &hook.command)?;
            Ok(VerifierCollectHook {
                service: normalize_service_reference(hook.service.as_deref(), compose)?,
                command: hook.command.clone(),
                timeout: normalize_timeout(
                    "verifier.collect.timeout_sec",
                    hook.timeout_sec.unwrap_or(60.0),
                )?,
                user: hook.user.as_deref().map(normalize_user).transpose()?,
            })
        })
        .collect()
}

fn normalize_collection_timeout(
    field: &str,
    authored: Option<f64>,
    inherited: Duration,
) -> Result<Duration, HarborImportError> {
    authored
        .map(|seconds| normalize_timeout(field, seconds))
        .transpose()
        .map(|timeout| timeout.unwrap_or(inherited))
}

fn validate_compose_step_policy(
    compose: Option<&ComposeProjectPlan>,
    environment: &EnvironmentPlan,
    agent: &PhasePlan,
    verifier: &VerifierPlan,
    artifacts: &[ArtifactSpec],
    hooks: &[VerifierCollectHook],
    allows_non_main_evidence: bool,
) -> Result<(), HarborImportError> {
    let mut saw_non_main_hook = false;
    for hook in hooks {
        if hook.service().as_str() == "main" {
            if saw_non_main_hook {
                return Err(HarborImportError::InvalidPackage(
                    "main collection hooks must precede non-main collection hooks".to_owned(),
                ));
            }
        } else {
            saw_non_main_hook = true;
        }
    }

    let has_non_main_evidence = artifacts
        .iter()
        .any(|artifact| artifact.service() != "main")
        || hooks.iter().any(|hook| hook.service().as_str() != "main");
    if has_non_main_evidence && verifier.mode() != VerifierMode::Separate {
        return Err(HarborImportError::InvalidPackage(
            "non-main evidence requires verifier.environment_mode = \"separate\"".to_owned(),
        ));
    }
    if has_non_main_evidence && !allows_non_main_evidence {
        return Err(HarborImportError::InvalidPackage(
            "non-main evidence is allowed only on the final explicit step".to_owned(),
        ));
    }

    if compose.is_some()
        && [
            environment.network(),
            agent.network(),
            verifier.environment().network(),
            verifier.phase().network(),
        ]
        .into_iter()
        .any(|network| !network.is_public())
    {
        return Err(HarborImportError::InvalidPackage(
            "Compose tasks require public environment and phase networks".to_owned(),
        ));
    }
    Ok(())
}

fn validate_shared_verifier_workdir(
    execution_plan: &BenchmarkExecutionPlan,
) -> Result<(), HarborImportError> {
    if execution_plan.uses_shared_verifier()
        && let Some(workdir) = execution_plan.environment().workdir()
        && shared_workdir_conflicts_reserved_verifier_path(workdir)
            .map_err(HarborImportError::InvalidPackage)?
    {
        return Err(HarborImportError::InvalidPackage(format!(
            "shared verifier workdir occupies a reserved verifier path: {workdir:?}"
        )));
    }
    Ok(())
}

fn normalize_environment(
    environment: StandardEnvironmentSection,
    image_source: ImageSource,
) -> Result<EnvironmentPlan, HarborImportError> {
    if environment.build_timeout_sec.is_some() || environment.startup_timeout_sec.is_some() {
        return Err(HarborImportError::InvalidPackage(
            "build_timeout_sec and startup_timeout_sec are valid only in the root environment"
                .to_owned(),
        ));
    }
    let resources = match (environment.cpus, environment.memory_mb) {
        (None, None) => None,
        (Some(cpus), Some(memory_mb)) if cpus > 0 && memory_mb > 0 => {
            Some(ContainerResources { cpus, memory_mb })
        }
        (Some(_), Some(_)) => {
            return Err(HarborImportError::InvalidPackage(
                "environment.cpus and environment.memory_mb must be positive".to_owned(),
            ));
        }
        _ => {
            return Err(HarborImportError::InvalidPackage(
                "environment.cpus and environment.memory_mb must be configured together".to_owned(),
            ));
        }
    };
    let workdir = environment
        .workdir
        .map(|workdir| normalize_workdir(&workdir))
        .transpose()?;
    let user = environment
        .user
        .map(|user| normalize_user(&user))
        .transpose()?;
    let network = normalize_network(
        environment.network.as_deref(),
        &environment.allowed_hosts,
        &NetworkPolicy::public(),
    )?;
    Ok(EnvironmentPlan {
        image_source,
        resources,
        workdir,
        user,
        env: normalize_environment_bindings(environment.env)?,
        network,
        healthcheck: environment
            .healthcheck
            .map(normalize_healthcheck)
            .transpose()?,
    })
}

fn normalize_agent_phase(
    agent: StandardAgentSection,
    inherited_network: &NetworkPolicy,
) -> Result<PhasePlan, HarborImportError> {
    Ok(PhasePlan {
        user: agent.user.map(|user| normalize_user(&user)).transpose()?,
        env: normalize_environment_bindings(agent.env)?,
        network: normalize_network(
            agent.network.as_deref(),
            &agent.allowed_hosts,
            inherited_network,
        )?,
        timeout: agent
            .timeout_sec
            .map(|seconds| normalize_timeout("agent.timeout_sec", seconds))
            .transpose()?,
    })
}

fn normalize_verifier_plan(
    verifier: StandardVerifierSection,
    environment: &EnvironmentPlan,
    image_source: ImageSource,
) -> Result<VerifierPlan, HarborImportError> {
    let mode = normalize_verifier_mode(verifier.environment_mode.as_deref())?;
    let verifier_environment = match verifier.environment.clone() {
        Some(verifier_environment) if mode == VerifierMode::Separate => {
            normalize_environment(verifier_environment, image_source)?
        }
        Some(_) => {
            return Err(HarborImportError::InvalidPackage(
                "verifier.environment is only valid when verifier.environment_mode = \"separate\""
                    .to_owned(),
            ));
        }
        None => environment.clone(),
    };
    Ok(VerifierPlan {
        phase: normalize_verifier_phase(verifier, verifier_environment.network())?,
        mode,
        environment: verifier_environment,
    })
}

fn overlay_agent_phase(
    root: &PhasePlan,
    override_: StandardAgentSection,
) -> Result<PhasePlan, HarborImportError> {
    let mut env = root.env().clone();
    env.extend(normalize_environment_bindings(override_.env)?);
    Ok(PhasePlan {
        user: override_
            .user
            .map(|user| normalize_user(&user))
            .transpose()?
            .or_else(|| root.user().map(str::to_owned)),
        env,
        network: normalize_network(
            override_.network.as_deref(),
            &override_.allowed_hosts,
            root.network(),
        )?,
        timeout: override_
            .timeout_sec
            .map(|seconds| normalize_step_timeout("steps.agent.timeout_sec", seconds))
            .transpose()?
            .or(root.timeout()),
    })
}

fn overlay_verifier_plan(
    root: &VerifierPlan,
    override_: StandardVerifierSection,
    environment: &EnvironmentPlan,
    environment_digest: &ArtifactDigest,
) -> Result<VerifierPlan, HarborImportError> {
    let mode = match override_.environment_mode.as_deref() {
        Some(value) => normalize_verifier_mode(Some(value))?,
        None => root.mode(),
    };
    let verifier_environment = match override_.environment {
        Some(verifier_environment) if mode == VerifierMode::Separate => normalize_environment(
            verifier_environment,
            ImageSource::task_dockerfile(environment_digest.clone()),
        )?,
        Some(_) => {
            return Err(HarborImportError::InvalidPackage(
                "steps.verifier.environment requires environment_mode = \"separate\"".to_owned(),
            ));
        }
        None if mode == VerifierMode::Separate && root.mode() == VerifierMode::Separate => {
            root.environment().clone()
        }
        None => environment.clone(),
    };
    let mut env = root.phase().env().clone();
    env.extend(normalize_environment_bindings(override_.env)?);
    Ok(VerifierPlan {
        phase: PhasePlan {
            user: override_
                .user
                .map(|user| normalize_user(&user))
                .transpose()?
                .or_else(|| root.phase().user().map(str::to_owned)),
            env,
            network: normalize_network(
                override_.network.as_deref(),
                &override_.allowed_hosts,
                root.phase().network(),
            )?,
            timeout: override_
                .timeout_sec
                .map(|seconds| normalize_step_timeout("steps.verifier.timeout_sec", seconds))
                .transpose()?
                .or(root.phase().timeout()),
        },
        mode,
        environment: verifier_environment,
    })
}

fn normalize_verifier_mode(value: Option<&str>) -> Result<VerifierMode, HarborImportError> {
    match value {
        None | Some("shared") => Ok(VerifierMode::Shared),
        Some("separate") => Ok(VerifierMode::Separate),
        Some(value) => Err(HarborImportError::InvalidPackage(format!(
            "unsupported verifier environment_mode {value:?}"
        ))),
    }
}

fn timeout_pair(
    agent: &PhasePlan,
    verifier: &PhasePlan,
) -> Result<Option<(Duration, Duration)>, HarborImportError> {
    match (agent.timeout(), verifier.timeout()) {
        (None, None) => Ok(None),
        (Some(agent), Some(verifier)) => Ok(Some((agent, verifier))),
        (None, Some(_)) => Err(HarborImportError::InvalidPackage(
            "agent.timeout_sec must be configured with verifier.timeout_sec".to_owned(),
        )),
        (Some(_), None) => Err(HarborImportError::InvalidPackage(
            "verifier.timeout_sec must be configured with agent.timeout_sec".to_owned(),
        )),
    }
}

fn validate_phase_timeout_pair(
    agent: &PhasePlan,
    verifier: &PhasePlan,
) -> Result<(), HarborImportError> {
    timeout_pair(agent, verifier).map(|_| ())
}

fn normalize_multi_step_reward_strategy(
    value: Option<String>,
) -> Result<MultiStepRewardStrategy, HarborImportError> {
    match value.as_deref() {
        None | Some("mean") => Ok(MultiStepRewardStrategy::Mean),
        Some("final") => Ok(MultiStepRewardStrategy::Final),
        Some(value) => Err(HarborImportError::InvalidPackage(format!(
            "unsupported multi_step_reward_strategy {value:?}"
        ))),
    }
}

fn validate_step_name(name: &str) -> Result<(), HarborImportError> {
    if name.is_empty()
        || !name
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-'))
    {
        return Err(HarborImportError::InvalidPackage(format!(
            "step name must be a safe path component: {name:?}"
        )));
    }
    Ok(())
}

fn validate_effective_artifact_targets(
    artifacts: &[ArtifactSpec],
) -> Result<(), HarborImportError> {
    let mut targets = Vec::with_capacity(artifacts.len());
    let mut sources = BTreeSet::new();
    for artifact in artifacts {
        if !sources.insert((artifact.service().to_owned(), artifact.source().to_owned())) {
            return Err(HarborImportError::InvalidPackage(format!(
                "artifact service/source pair is duplicated: ({:?}, {:?})",
                artifact.service(),
                artifact.source()
            )));
        }
        let target = artifact.output_target();
        if targets.iter().any(|existing: &String| {
            existing == &target
                || existing.strip_prefix(&format!("{target}/")).is_some()
                || target.strip_prefix(&format!("{existing}/")).is_some()
        }) {
            return Err(HarborImportError::InvalidPackage(format!(
                "artifact output target is duplicated or overlaps: {target:?}"
            )));
        }
        targets.push(target);
    }
    Ok(())
}

fn validate_step_artifacts(
    artifacts: &[ArtifactSpec],
    verifier: &VerifierPlan,
) -> Result<(), HarborImportError> {
    validate_effective_artifact_targets(artifacts)?;
    if let Some(artifact) = artifacts
        .iter()
        .find(|artifact| artifact_source_overlaps_reserved_verifier_path(artifact))
    {
        return Err(HarborImportError::InvalidPackage(format!(
            "artifact source overlaps a reserved verifier path: {:?}",
            artifact.source()
        )));
    }
    if verifier.mode() == VerifierMode::Separate
        && let Some(workdir) = verifier.environment().workdir()
    {
        let collision = verifier_artifact_target_collision(workdir, artifacts)
            .map_err(HarborImportError::InvalidPackage)?;
        if let Some(target) = collision {
            return Err(HarborImportError::InvalidPackage(format!(
                "artifact destination overlaps a reserved verifier path: {target:?}"
            )));
        }
    }
    Ok(())
}

fn normalize_verifier_phase(
    verifier: StandardVerifierSection,
    inherited_network: &NetworkPolicy,
) -> Result<PhasePlan, HarborImportError> {
    Ok(PhasePlan {
        user: verifier
            .user
            .map(|user| normalize_user(&user))
            .transpose()?,
        env: normalize_environment_bindings(verifier.env)?,
        network: normalize_network(
            verifier.network.as_deref(),
            &verifier.allowed_hosts,
            inherited_network,
        )?,
        timeout: verifier
            .timeout_sec
            .map(|seconds| normalize_timeout("verifier.timeout_sec", seconds))
            .transpose()?,
    })
}

fn normalize_network(
    authored: Option<&str>,
    allowed_hosts: &[String],
    inherited: &NetworkPolicy,
) -> Result<NetworkPolicy, HarborImportError> {
    let Some(authored) = authored else {
        if allowed_hosts.is_empty() {
            return Ok(inherited.clone());
        }
        return Err(HarborImportError::InvalidPackage(
            "allowed_hosts requires network = \"allowlist\"".to_owned(),
        ));
    };
    match authored {
        "public" if allowed_hosts.is_empty() => Ok(NetworkPolicy::public()),
        "no-network" if allowed_hosts.is_empty() => Ok(NetworkPolicy::no_network()),
        "allowlist" => {
            NetworkPolicy::allowlist(allowed_hosts).map_err(HarborImportError::InvalidPackage)
        }
        "public" | "no-network" => Err(HarborImportError::InvalidPackage(
            "allowed_hosts requires network = \"allowlist\"".to_owned(),
        )),
        _ => Err(HarborImportError::InvalidPackage(format!(
            "unsupported network policy {authored:?}"
        ))),
    }
}

fn normalize_environment_bindings(
    bindings: BTreeMap<String, String>,
) -> Result<BTreeMap<String, EnvBinding>, HarborImportError> {
    bindings
        .into_iter()
        .map(|(name, value)| {
            crate::eval::validate_env_name(&name).map_err(HarborImportError::InvalidPackage)?;
            let value = EnvBinding::parse(&value).map_err(HarborImportError::InvalidPackage)?;
            Ok((name, value))
        })
        .collect()
}

fn normalize_user(user: &str) -> Result<String, HarborImportError> {
    crate::eval::validate_user(user).map_err(HarborImportError::InvalidPackage)?;
    Ok(user.to_owned())
}

fn normalize_workdir(workdir: &str) -> Result<String, HarborImportError> {
    let parsed = Path::new(workdir);
    if !parsed.is_absolute()
        || parsed.components().any(|component| {
            matches!(
                component,
                Component::ParentDir | Component::CurDir | Component::Prefix(_)
            )
        })
    {
        return Err(HarborImportError::InvalidPackage(format!(
            "workdir must be an absolute isolated path: {workdir:?}"
        )));
    }
    Ok(format!(
        "/{}",
        parsed
            .components()
            .filter_map(|component| match component {
                Component::Normal(segment) => Some(segment.to_string_lossy().into_owned()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("/")
    ))
}

fn normalize_healthcheck(
    healthcheck: StandardHealthcheckSection,
) -> Result<HealthcheckPlan, HarborImportError> {
    validate_command("environment.healthcheck.command", &healthcheck.command)?;
    if healthcheck.retries == Some(0) {
        return Err(HarborImportError::InvalidPackage(
            "environment.healthcheck.retries must be positive".to_owned(),
        ));
    }
    Ok(HealthcheckPlan {
        command: healthcheck.command,
        start_period: normalize_optional_duration(
            "environment.healthcheck.start_period_sec",
            healthcheck.start_period_sec,
        )?,
        start_interval: normalize_optional_duration(
            "environment.healthcheck.start_interval_sec",
            healthcheck.start_interval_sec,
        )?,
        interval: normalize_optional_duration(
            "environment.healthcheck.interval_sec",
            healthcheck.interval_sec,
        )?,
        timeout: normalize_optional_duration(
            "environment.healthcheck.timeout_sec",
            healthcheck.timeout_sec,
        )?,
        retries: healthcheck.retries,
    })
}

fn normalize_optional_duration(
    field: &str,
    seconds: Option<f64>,
) -> Result<Option<Duration>, HarborImportError> {
    seconds
        .map(|seconds| normalize_timeout(field, seconds))
        .transpose()
}

fn normalize_standard_artifacts(
    artifacts: Vec<StandardArtifactDto>,
    compose: Option<&ComposeProjectPlan>,
) -> Result<Vec<ArtifactSpec>, HarborImportError> {
    artifacts
        .into_iter()
        .map(|artifact| match artifact {
            StandardArtifactDto::ExactFile(source) => {
                Ok(ArtifactSpec::exact_file(normalize_artifact_path(&source)?))
            }
            StandardArtifactDto::Collected(artifact) => {
                let service = normalize_service_reference(artifact.service.as_deref(), compose)?;
                let source = normalize_artifact_path(&artifact.source)?;
                let destination = artifact
                    .destination
                    .map(|destination| normalize_artifact_destination(&destination))
                    .transpose()?;
                let exclude = normalize_artifact_exclusions(artifact.exclude)?;
                if service.as_str() == "main" {
                    Ok(ArtifactSpec::collected(source, destination, exclude))
                } else {
                    Ok(ArtifactSpec::collected_for_service(
                        source,
                        destination,
                        exclude,
                        service,
                    ))
                }
            }
        })
        .collect()
}

fn normalize_artifact_exclusions(
    mut patterns: Vec<String>,
) -> Result<Vec<String>, HarborImportError> {
    for pattern in &patterns {
        if pattern.is_empty()
            || pattern.starts_with('/')
            || pattern
                .split('/')
                .any(|component| component.is_empty() || component == "." || component == "..")
        {
            return Err(HarborImportError::InvalidPackage(format!(
                "artifact exclusion must be a nonempty relative glob with normal path components: {pattern:?}"
            )));
        }
    }
    patterns.sort();
    patterns.dedup();
    Ok(patterns)
}

fn normalize_timeout(field: &str, seconds: f64) -> Result<Duration, HarborImportError> {
    if !seconds.is_finite() || seconds <= 0.0 {
        return Err(HarborImportError::InvalidPackage(format!(
            "{field} must be finite and positive"
        )));
    }
    let duration = Duration::try_from_secs_f64(seconds).map_err(|error| {
        HarborImportError::InvalidPackage(format!("{field} is invalid: {error}"))
    })?;
    if duration.is_zero() {
        return Err(HarborImportError::InvalidPackage(format!(
            "{field} is below nanosecond precision"
        )));
    }
    Ok(duration)
}

fn normalize_step_timeout(field: &str, seconds: f64) -> Result<Duration, HarborImportError> {
    if seconds.fract() != 0.0 {
        return Err(HarborImportError::InvalidPackage(format!(
            "{field} must be a whole number of seconds"
        )));
    }
    normalize_timeout(field, seconds)
}

fn read_required_source_file(
    source: &AcquiredSource,
    relative_path: &str,
) -> Result<String, HarborImportError> {
    String::from_utf8(read_required_source_bytes(source, relative_path)?)
        .map_err(|error| HarborImportError::InvalidPackage(format!("{relative_path}: {error}")))
}

fn read_required_source_bytes(
    source: &AcquiredSource,
    relative_path: &str,
) -> Result<Vec<u8>, HarborImportError> {
    Ok(source.read(relative_path)?.to_vec())
}

fn validate_command(field: &'static str, command: &[String]) -> Result<(), HarborImportError> {
    if command.is_empty() || command.iter().any(|part| part.trim().is_empty()) {
        return Err(HarborImportError::InvalidPackage(format!(
            "{field} must be a nonempty argv"
        )));
    }
    Ok(())
}

fn normalize_declared_artifacts(artifacts: Vec<String>) -> Result<Vec<String>, HarborImportError> {
    let mut normalized = Vec::with_capacity(artifacts.len());
    let mut paths = BTreeSet::new();
    for path in artifacts {
        let path = normalize_artifact_path(&path)?;
        if !paths.insert(path.clone()) {
            return Err(HarborImportError::InvalidPackage(format!(
                "declared artifact path is duplicated: {path:?}"
            )));
        }
        normalized.push(path);
    }
    Ok(normalized)
}

fn normalize_artifact_path(path: &str) -> Result<String, HarborImportError> {
    let parsed = Path::new(path);
    if !parsed.is_absolute() || parsed == Path::new("/") {
        return Err(invalid_artifact_path(path));
    }
    if parsed.components().any(|component| {
        matches!(
            component,
            Component::ParentDir | Component::CurDir | Component::Prefix(_)
        )
    }) {
        return Err(invalid_artifact_path(path));
    }
    Ok(format!(
        "/{}",
        parsed
            .components()
            .filter_map(|component| match component {
                Component::Normal(segment) => Some(segment.to_string_lossy().into_owned()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("/")
    ))
}

fn normalize_artifact_destination(path: &str) -> Result<String, HarborImportError> {
    let parsed = Path::new(path);
    if parsed.is_absolute()
        || parsed.as_os_str().is_empty()
        || parsed.components().any(|component| {
            matches!(
                component,
                Component::ParentDir
                    | Component::CurDir
                    | Component::Prefix(_)
                    | Component::RootDir
            )
        })
    {
        return Err(HarborImportError::InvalidPackage(format!(
            "artifact destination must be a relative isolated path: {path:?}"
        )));
    }
    Ok(parsed
        .components()
        .filter_map(|component| match component {
            Component::Normal(segment) => Some(segment.to_string_lossy().into_owned()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("/"))
}

fn invalid_artifact_path(path: &str) -> HarborImportError {
    HarborImportError::InvalidPackage(format!(
        "declared artifact path must be absolute and isolated: {path:?}"
    ))
}
