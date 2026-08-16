// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

//! Strict policy for task-authored Compose sidecars and the generated `main` service.

#![cfg_attr(not(test), allow(dead_code))]

use std::{
    collections::{BTreeMap, BTreeSet},
    fmt, fs,
    path::{Component, Path, PathBuf},
};

use serde::{
    Deserialize, Deserializer, Serialize,
    de::{MapAccess, SeqAccess, Visitor, value::MapAccessDeserializer},
};

use super::{
    ComposeProjectPlan, ComposeServiceName, EnvironmentPlan, EvalExecutionError, validate_env_name,
};

const DEFAULT_NETWORK: &str = "default";
const DEFAULT_WORKDIR: &str = "/work";

/// The static service identity and generated-main dependency material accepted by policy.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ValidatedComposeProject {
    pub(crate) services: BTreeSet<ComposeServiceName>,
    pub(crate) main_dependencies: Vec<ComposeServiceName>,
    main_dependency_contract: MainDependencyContract,
}

impl ValidatedComposeProject {
    /// Returns the static main dependency contract for canonical validation.
    pub(crate) fn main_dependency_contract(&self) -> &MainDependencyContract {
        &self.main_dependency_contract
    }
}

/// Compose-normalized dependency authority captured from the authored overlay.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct MainDependencyContract {
    canonical: Option<BTreeMap<String, DependencyCondition>>,
    names: Vec<ComposeServiceName>,
}

/// Private base bytes paired with the exact generated-main authority they encode.
pub(crate) struct RenderedGeneratedMainCompose {
    bytes: Vec<u8>,
    expected_main: ExpectedGeneratedMain,
}

impl RenderedGeneratedMainCompose {
    /// Returns the generated base file bytes for materialization.
    pub(crate) fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Consumes the rendering and returns its generated base file bytes.
    pub(crate) fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }

    /// Returns the trusted main authority required for canonical validation.
    pub(crate) fn expected_main(&self) -> &ExpectedGeneratedMain {
        &self.expected_main
    }
}

/// Exact runtime-owned fields expected after Compose canonicalization.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub(crate) struct ExpectedGeneratedMain {
    image: String,
    command: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    working_dir: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    user: Option<String>,
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    environment: BTreeMap<String, String>,
    labels: BTreeMap<String, String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    cpus: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    mem_limit: Option<String>,
    volumes: Vec<ExpectedGeneratedMount>,
    networks: Vec<String>,
    restart: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
struct ExpectedGeneratedMount {
    #[serde(rename = "type")]
    mount_type: String,
    source: String,
    target: String,
}

/// Strictly validates the task-authored sidecar overlay without consulting Docker.
pub(crate) fn validate_authored_compose(
    yaml: &[u8],
    plan: &ComposeProjectPlan,
    environment_root: &Path,
) -> Result<ValidatedComposeProject, EvalExecutionError> {
    reject_dotenv_files(environment_root)?;
    let document = serde_yaml::from_slice::<AuthoredCompose>(yaml)
        .map_err(|error| decode_error("compose", &error.to_string()))?;
    reject_interpolation_in(&document)?;
    validate_authored_document(document, plan, environment_root)
}

/// Validates Docker Compose's read-only normalized JSON before any provider mutation.
pub(crate) fn validate_canonical_compose_json(
    json: &[u8],
    plan: &ComposeProjectPlan,
    environment_root: &Path,
    expected_main: &ExpectedGeneratedMain,
    expected_main_dependencies: &MainDependencyContract,
) -> Result<ValidatedComposeProject, EvalExecutionError> {
    reject_dotenv_files(environment_root)?;
    let document = serde_json::from_slice::<CanonicalCompose>(json)
        .map_err(|error| decode_error("compose", &error.to_string()))?;
    reject_interpolation_in(&document)?;
    validate_canonical_document(
        document,
        plan,
        environment_root,
        expected_main,
        expected_main_dependencies,
    )
}

/// Validates provider-produced JSON against the private generated-main model.
///
/// The model is retained by this module so a provider result cannot be accepted
/// merely by presenting a plausible canonical `main` service.
pub(crate) fn validate_provider_compose_config(
    json: &[u8],
    plan: &ComposeProjectPlan,
    environment_root: &Path,
    rendered: &RenderedGeneratedMainCompose,
    authored: &ValidatedComposeProject,
) -> Result<ValidatedComposeProject, EvalExecutionError> {
    validate_canonical_compose_json(
        json,
        plan,
        environment_root,
        rendered.expected_main(),
        authored.main_dependency_contract(),
    )
}

/// Renders the private base file whose `main` service is controlled only by AIPerf.
pub(crate) fn render_generated_main_compose(
    image_tag: &str,
    project_labels: &BTreeMap<String, String>,
    environment: &EnvironmentPlan,
    workspace: &Path,
) -> Result<RenderedGeneratedMainCompose, EvalExecutionError> {
    if image_tag.trim().is_empty() {
        return Err(policy_error(
            "services.main.image",
            "image tag must not be empty",
        ));
    }
    if !environment.network().is_public() {
        return Err(policy_error(
            "services.main.networks",
            "generated Compose main requires the public default network",
        ));
    }
    let workspace = workspace.canonicalize().map_err(|error| {
        policy_error(
            "services.main.volumes[0].source",
            format!("workspace cannot be resolved: {error}"),
        )
    })?;
    if !workspace.is_dir() {
        return Err(policy_error(
            "services.main.volumes[0].source",
            "workspace must be a directory",
        ));
    }
    let target = environment.workdir().unwrap_or(DEFAULT_WORKDIR);
    validate_container_path(target, "services.main.volumes[0].target")?;

    let mut labels = BTreeMap::new();
    for (name, value) in project_labels {
        if !is_reserved_label(name) {
            return Err(policy_error(
                format!("services.main.labels.{name}"),
                "generated project labels must use the AIPerf namespace",
            ));
        }
        labels.insert(name.clone(), escape_compose_dollars(value));
    }
    let public_environment = environment
        .env()
        .iter()
        .filter_map(|(name, binding)| {
            binding
                .literal()
                .map(|value| (name.clone(), escape_compose_dollars(value)))
        })
        .collect();
    let resources = environment.resources();
    let expected_main = ExpectedGeneratedMain {
        image: escape_compose_dollars(image_tag),
        command: vec!["sleep".to_owned(), "infinity".to_owned()],
        working_dir: environment.workdir().map(escape_compose_dollars),
        user: environment.user().map(escape_compose_dollars),
        environment: public_environment,
        labels,
        cpus: resources.map(|limits| limits.cpus()),
        mem_limit: resources.map(|limits| format!("{}m", limits.memory_mb())),
        volumes: vec![ExpectedGeneratedMount {
            mount_type: "bind".to_owned(),
            source: escape_compose_dollars(workspace.to_string_lossy().as_ref()),
            target: escape_compose_dollars(target),
        }],
        networks: vec![DEFAULT_NETWORK.to_owned()],
        restart: "no".to_owned(),
    };
    let bytes = serde_yaml::to_string(&GeneratedCompose {
        services: BTreeMap::from([("main", &expected_main)]),
        networks: BTreeMap::from([(DEFAULT_NETWORK, GeneratedNetwork {})]),
    })
    .map(String::into_bytes)
    .map_err(|error| policy_error("compose", format!("cannot render generated main: {error}")))?;
    Ok(RenderedGeneratedMainCompose {
        bytes,
        expected_main,
    })
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct AuthoredCompose {
    services: BTreeMap<String, AuthoredService>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    volumes: BTreeMap<String, AuthoredVolumeDefinition>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct AuthoredVolumeDefinition {}

#[derive(Debug, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct AuthoredService {
    #[serde(skip_serializing_if = "Option::is_none")]
    image: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    build: Option<BuildSpec>,
    #[serde(skip_serializing_if = "Option::is_none")]
    command: Option<CommandSpec>,
    #[serde(skip_serializing_if = "Option::is_none")]
    entrypoint: Option<CommandSpec>,
    #[serde(skip_serializing_if = "Option::is_none")]
    working_dir: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    user: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    environment: Option<EnvironmentSpec>,
    #[serde(skip_serializing_if = "Option::is_none")]
    depends_on: Option<DependsOn>,
    #[serde(skip_serializing_if = "Option::is_none")]
    healthcheck: Option<ComposeHealthcheck>,
    #[serde(skip_serializing_if = "Option::is_none")]
    expose: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    volumes: Option<Vec<VolumeMount>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    read_only: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tmpfs: Option<OneOrManyStrings>,
    #[serde(skip_serializing_if = "Option::is_none")]
    init: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop_grace_period: Option<LiteralScalar>,
    #[serde(skip_serializing_if = "Option::is_none")]
    labels: Option<Labels>,
    #[serde(skip_serializing_if = "Option::is_none")]
    cpus: Option<LiteralScalar>,
    #[serde(skip_serializing_if = "Option::is_none")]
    mem_limit: Option<LiteralScalar>,
}

#[derive(Debug, Serialize)]
#[serde(untagged)]
enum BuildSpec {
    Context(String),
    Detailed(BuildDetails),
}

impl<'de> Deserialize<'de> for BuildSpec {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_any(BuildSpecVisitor)
    }
}

struct BuildSpecVisitor;

impl<'de> Visitor<'de> for BuildSpecVisitor {
    type Value = BuildSpec;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a local build context string or strict build mapping")
    }

    fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        Ok(BuildSpec::Context(value.to_owned()))
    }

    fn visit_string<E>(self, value: String) -> Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        Ok(BuildSpec::Context(value))
    }

    fn visit_map<A>(self, map: A) -> Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        BuildDetails::deserialize(MapAccessDeserializer::new(map)).map(BuildSpec::Detailed)
    }
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct BuildDetails {
    context: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    dockerfile: Option<String>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    args: BTreeMap<String, LiteralScalar>,
    #[serde(skip_serializing_if = "Option::is_none")]
    target: Option<String>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(untagged)]
enum CommandSpec {
    String(String),
    List(Vec<String>),
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(untagged)]
enum OneOrManyStrings {
    One(String),
    Many(Vec<String>),
}

#[derive(Debug, Serialize)]
#[serde(untagged)]
enum EnvironmentSpec {
    Map(BTreeMap<String, LiteralScalar>),
    List(Vec<String>),
}

impl<'de> Deserialize<'de> for EnvironmentSpec {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_any(EnvironmentSpecVisitor)
    }
}

struct EnvironmentSpecVisitor;

impl<'de> Visitor<'de> for EnvironmentSpecVisitor {
    type Value = EnvironmentSpec;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a literal environment mapping or NAME=value list")
    }

    fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        let mut bindings = BTreeMap::new();
        while let Some(name) = map.next_key::<String>()? {
            let value = map.next_value::<Option<LiteralScalar>>()?.ok_or_else(|| {
                serde::de::Error::custom(format!("null environment value for `{name}`"))
            })?;
            bindings.insert(name, value);
        }
        Ok(EnvironmentSpec::Map(bindings))
    }

    fn visit_seq<A>(self, mut sequence: A) -> Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        let mut bindings = Vec::new();
        while let Some(binding) = sequence.next_element::<String>()? {
            bindings.push(binding);
        }
        Ok(EnvironmentSpec::List(bindings))
    }
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(untagged)]
enum Labels {
    Map(BTreeMap<String, LiteralScalar>),
    List(Vec<String>),
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(untagged)]
enum DependsOn {
    List(Vec<String>),
    Map(BTreeMap<String, DependencyCondition>),
}

impl DependsOn {
    fn names(&self) -> impl Iterator<Item = &str> {
        match self {
            Self::List(names) => DependsOnNames::List(names.iter()),
            Self::Map(dependencies) => DependsOnNames::Map(dependencies.keys()),
        }
    }
}

enum DependsOnNames<'a> {
    List(std::slice::Iter<'a, String>),
    Map(std::collections::btree_map::Keys<'a, String, DependencyCondition>),
}

impl<'a> Iterator for DependsOnNames<'a> {
    type Item = &'a str;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::List(names) => names.next().map(String::as_str),
            Self::Map(names) => names.next().map(String::as_str),
        }
    }
}

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct DependencyCondition {
    #[serde(skip_serializing_if = "Option::is_none")]
    condition: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    restart: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    required: Option<bool>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct ComposeHealthcheck {
    #[serde(skip_serializing_if = "Option::is_none")]
    test: Option<CommandSpec>,
    #[serde(skip_serializing_if = "Option::is_none")]
    interval: Option<LiteralScalar>,
    #[serde(skip_serializing_if = "Option::is_none")]
    timeout: Option<LiteralScalar>,
    #[serde(skip_serializing_if = "Option::is_none")]
    retries: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    start_period: Option<LiteralScalar>,
    #[serde(skip_serializing_if = "Option::is_none")]
    start_interval: Option<LiteralScalar>,
    #[serde(skip_serializing_if = "Option::is_none")]
    disable: Option<bool>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(untagged)]
enum VolumeMount {
    Short(String),
    Long(LongVolumeMount),
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct LongVolumeMount {
    #[serde(rename = "type", skip_serializing_if = "Option::is_none")]
    mount_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    source: Option<String>,
    target: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    read_only: Option<bool>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(untagged)]
enum LiteralScalar {
    String(String),
    Signed(i64),
    Unsigned(u64),
    Float(f64),
    Bool(bool),
}

impl LiteralScalar {
    fn is_valid(&self) -> bool {
        !matches!(self, Self::Float(value) if !value.is_finite())
    }

    fn is_positive(&self) -> bool {
        match self {
            Self::String(value) => !value.trim().is_empty() && !value.starts_with('-'),
            Self::Signed(value) => *value > 0,
            Self::Unsigned(value) => *value > 0,
            Self::Float(value) => value.is_finite() && *value > 0.0,
            Self::Bool(_) => false,
        }
    }
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct CanonicalCompose {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    name: Option<String>,
    services: BTreeMap<String, CanonicalService>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    networks: BTreeMap<String, CanonicalNetworkDefinition>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    volumes: BTreeMap<String, CanonicalVolumeDefinition>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct CanonicalNetworkDefinition {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    name: Option<String>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct CanonicalVolumeDefinition {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    name: Option<String>,
}

#[derive(Debug, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct CanonicalService {
    #[serde(skip_serializing_if = "Option::is_none")]
    image: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    build: Option<BuildSpec>,
    #[serde(skip_serializing_if = "Option::is_none")]
    command: Option<CommandSpec>,
    #[serde(skip_serializing_if = "Option::is_none")]
    entrypoint: Option<CommandSpec>,
    #[serde(skip_serializing_if = "Option::is_none")]
    working_dir: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    user: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    environment: Option<EnvironmentSpec>,
    #[serde(skip_serializing_if = "Option::is_none")]
    depends_on: Option<DependsOn>,
    #[serde(skip_serializing_if = "Option::is_none")]
    healthcheck: Option<ComposeHealthcheck>,
    #[serde(skip_serializing_if = "Option::is_none")]
    expose: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    volumes: Option<Vec<CanonicalVolumeMount>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    read_only: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tmpfs: Option<OneOrManyStrings>,
    #[serde(skip_serializing_if = "Option::is_none")]
    init: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop_grace_period: Option<LiteralScalar>,
    #[serde(skip_serializing_if = "Option::is_none")]
    labels: Option<Labels>,
    #[serde(skip_serializing_if = "Option::is_none")]
    cpus: Option<LiteralScalar>,
    #[serde(skip_serializing_if = "Option::is_none")]
    mem_limit: Option<LiteralScalar>,
    #[serde(skip_serializing_if = "Option::is_none")]
    networks: Option<BTreeMap<String, Option<CanonicalServiceNetwork>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    restart: Option<String>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct CanonicalServiceNetwork {}

#[derive(Debug, Deserialize, Serialize)]
#[serde(untagged)]
enum CanonicalVolumeMount {
    Short(String),
    Long(CanonicalLongVolumeMount),
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct CanonicalLongVolumeMount {
    #[serde(rename = "type")]
    mount_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    source: Option<String>,
    target: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    read_only: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    bind: Option<CanonicalBindOptions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    volume: Option<CanonicalVolumeOptions>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct CanonicalBindOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    create_host_path: Option<bool>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct CanonicalVolumeOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    nocopy: Option<bool>,
}

#[derive(Serialize)]
struct GeneratedCompose<'a> {
    services: BTreeMap<&'a str, &'a ExpectedGeneratedMain>,
    networks: BTreeMap<&'a str, GeneratedNetwork>,
}

#[derive(Serialize)]
struct GeneratedNetwork {}

fn validate_authored_document(
    document: AuthoredCompose,
    plan: &ComposeProjectPlan,
    environment_root: &Path,
) -> Result<ValidatedComposeProject, EvalExecutionError> {
    if document.services.is_empty() {
        return Err(policy_error("services", "must be a nonempty mapping"));
    }
    validate_volume_definitions(document.volumes.keys())?;
    let declared_volumes = document.volumes.keys().cloned().collect::<BTreeSet<_>>();
    let mut services = BTreeSet::from([ComposeServiceName::main()]);
    let mut main_dependencies = Vec::new();
    let mut main_dependency_contract = MainDependencyContract {
        canonical: None,
        names: Vec::new(),
    };
    for (name, service) in document.services {
        let service_name = ComposeServiceName::parse(&name)
            .map_err(|reason| policy_error(format!("services.{name}"), reason))?;
        let path = format!("services.{name}");
        if name == "main" {
            validate_authored_main(&service, &path)?;
            main_dependency_contract = validate_authored_main_dependencies(
                service.depends_on.as_ref(),
                &path,
                plan.services(),
            )?;
            main_dependencies = main_dependency_contract.names.clone();
        } else {
            validate_sidecar_service(
                &service,
                &path,
                environment_root,
                &declared_volumes,
                plan.services(),
            )?;
        }
        services.insert(service_name);
    }
    validate_service_set(&services, plan.services())?;
    Ok(ValidatedComposeProject {
        services,
        main_dependencies,
        main_dependency_contract,
    })
}

fn validate_canonical_document(
    document: CanonicalCompose,
    plan: &ComposeProjectPlan,
    environment_root: &Path,
    expected_main: &ExpectedGeneratedMain,
    expected_main_dependencies: &MainDependencyContract,
) -> Result<ValidatedComposeProject, EvalExecutionError> {
    if document.services.is_empty() {
        return Err(policy_error("services", "must be a nonempty mapping"));
    }
    if document
        .networks
        .keys()
        .map(String::as_str)
        .ne([DEFAULT_NETWORK])
    {
        return Err(policy_error(
            "networks",
            "canonical project must contain only the default public network",
        ));
    }
    validate_volume_definitions(document.volumes.keys())?;
    let declared_volumes = document.volumes.keys().cloned().collect::<BTreeSet<_>>();
    let mut services = BTreeSet::new();
    let mut main_dependencies = Vec::new();
    for (name, service) in document.services {
        let service_name = ComposeServiceName::parse(&name)
            .map_err(|reason| policy_error(format!("services.{name}"), reason))?;
        let path = format!("services.{name}");
        validate_default_networks(service.networks.as_ref(), &path)?;
        if name == "main" {
            main_dependencies = validate_canonical_main(
                &service,
                &path,
                expected_main,
                expected_main_dependencies,
            )?;
        } else {
            validate_canonical_sidecar(
                &service,
                &path,
                environment_root,
                &declared_volumes,
                plan.services(),
            )?;
        }
        services.insert(service_name);
    }
    validate_service_set(&services, plan.services())?;
    Ok(ValidatedComposeProject {
        services,
        main_dependencies,
        main_dependency_contract: expected_main_dependencies.clone(),
    })
}

fn validate_authored_main(service: &AuthoredService, path: &str) -> Result<(), EvalExecutionError> {
    let forbidden = [
        (service.image.is_some(), "image"),
        (service.build.is_some(), "build"),
        (service.command.is_some(), "command"),
        (service.entrypoint.is_some(), "entrypoint"),
        (service.working_dir.is_some(), "working_dir"),
        (service.user.is_some(), "user"),
        (service.environment.is_some(), "environment"),
        (service.healthcheck.is_some(), "healthcheck"),
        (service.expose.is_some(), "expose"),
        (service.volumes.is_some(), "volumes"),
        (service.read_only.is_some(), "read_only"),
        (service.tmpfs.is_some(), "tmpfs"),
        (service.init.is_some(), "init"),
        (service.stop_grace_period.is_some(), "stop_grace_period"),
        (service.labels.is_some(), "labels"),
        (service.cpus.is_some(), "cpus"),
        (service.mem_limit.is_some(), "mem_limit"),
    ];
    if let Some((_, field)) = forbidden.into_iter().find(|(present, _)| *present) {
        return Err(policy_error(
            format!("{path}.{field}"),
            "authored main may contain only depends_on",
        ));
    }
    Ok(())
}

fn validate_sidecar_service(
    service: &AuthoredService,
    path: &str,
    environment_root: &Path,
    declared_volumes: &BTreeSet<String>,
    services: &BTreeSet<ComposeServiceName>,
) -> Result<(), EvalExecutionError> {
    if service.image.is_none() && service.build.is_none() {
        return Err(policy_error(
            path,
            "sidecar must declare an image or local build",
        ));
    }
    if let Some(image) = &service.image
        && image.trim().is_empty()
    {
        return Err(policy_error(format!("{path}.image"), "must not be empty"));
    }
    if let Some(build) = &service.build {
        validate_build(build, &format!("{path}.build"), environment_root)?;
    }
    validate_common_service(
        service.command.as_ref(),
        service.entrypoint.as_ref(),
        service.working_dir.as_deref(),
        service.user.as_deref(),
        service.environment.as_ref(),
        service.depends_on.as_ref(),
        service.expose.as_deref(),
        service.volumes.as_deref(),
        service.tmpfs.as_ref(),
        service.labels.as_ref(),
        service.cpus.as_ref(),
        service.mem_limit.as_ref(),
        path,
        declared_volumes,
        services,
    )
}

fn validate_canonical_sidecar(
    service: &CanonicalService,
    path: &str,
    environment_root: &Path,
    declared_volumes: &BTreeSet<String>,
    services: &BTreeSet<ComposeServiceName>,
) -> Result<(), EvalExecutionError> {
    if service.restart.is_some() {
        return Err(policy_error(
            format!("{path}.restart"),
            "sidecar restart policies are forbidden",
        ));
    }
    if service.image.is_none() && service.build.is_none() {
        return Err(policy_error(
            path,
            "sidecar must declare an image or local build",
        ));
    }
    if let Some(build) = &service.build {
        validate_build(build, &format!("{path}.build"), environment_root)?;
    }
    validate_common_canonical_service(service, path, declared_volumes, services)
}

fn validate_canonical_main(
    service: &CanonicalService,
    path: &str,
    expected: &ExpectedGeneratedMain,
    expected_dependencies: &MainDependencyContract,
) -> Result<Vec<ComposeServiceName>, EvalExecutionError> {
    if service.image.as_ref() != Some(&expected.image) {
        return Err(policy_error(
            format!("{path}.image"),
            "generated main image differs from runtime authority",
        ));
    }
    if service.build.is_some() {
        return Err(policy_error(
            format!("{path}.build"),
            "generated main must not be replaced by a build",
        ));
    }
    match &service.command {
        Some(CommandSpec::List(command)) if command == &expected.command => {}
        _ => {
            return Err(policy_error(
                format!("{path}.command"),
                "generated main command differs from runtime authority",
            ));
        }
    }
    for (present, field) in [
        (service.entrypoint.is_some(), "entrypoint"),
        (service.healthcheck.is_some(), "healthcheck"),
        (service.expose.is_some(), "expose"),
        (service.read_only.is_some(), "read_only"),
        (service.tmpfs.is_some(), "tmpfs"),
        (service.init.is_some(), "init"),
        (service.stop_grace_period.is_some(), "stop_grace_period"),
    ] {
        if present {
            return Err(policy_error(
                format!("{path}.{field}"),
                "generated main contains an unsupported field",
            ));
        }
    }
    if service.working_dir != expected.working_dir {
        return Err(policy_error(
            format!("{path}.working_dir"),
            "generated main workdir differs from runtime authority",
        ));
    }
    if service.user != expected.user {
        return Err(policy_error(
            format!("{path}.user"),
            "generated main user differs from runtime authority",
        ));
    }
    if service.restart.as_ref() != Some(&expected.restart) {
        return Err(policy_error(
            format!("{path}.restart"),
            "generated main restart policy differs from runtime authority",
        ));
    }
    if !canonical_environment_matches(service.environment.as_ref(), &expected.environment) {
        return Err(policy_error(
            format!("{path}.environment"),
            "generated main environment differs from runtime authority",
        ));
    }
    if canonical_label_map(service.labels.as_ref()).as_ref() != Some(&expected.labels) {
        return Err(policy_error(
            format!("{path}.labels"),
            "generated main labels differ from runtime authority",
        ));
    }
    if canonical_u64(service.cpus.as_ref()) != expected.cpus {
        return Err(policy_error(
            format!("{path}.cpus"),
            "generated main CPU limit differs from runtime authority",
        ));
    }
    if canonical_string(service.mem_limit.as_ref()) != expected.mem_limit.as_deref() {
        return Err(policy_error(
            format!("{path}.mem_limit"),
            "generated main memory limit differs from runtime authority",
        ));
    }
    if let Some(mounts) = service.volumes.as_deref() {
        if mounts.len() != expected.volumes.len() {
            return Err(policy_error(
                format!("{path}.volumes"),
                "generated main mounts differ from runtime authority",
            ));
        }
        validate_generated_main_mount(
            &mounts[0],
            &expected.volumes[0],
            &format!("{path}.volumes[0]"),
        )?;
    } else {
        return Err(policy_error(
            format!("{path}.volumes"),
            "generated main is missing its workspace mount",
        ));
    }
    if !canonical_networks_match(service.networks.as_ref(), &expected.networks) {
        return Err(policy_error(
            format!("{path}.networks"),
            "generated main networks differ from runtime authority",
        ));
    }
    validate_canonical_main_dependencies(service.depends_on.as_ref(), expected_dependencies, path)?;
    Ok(expected_dependencies.names.clone())
}

#[allow(clippy::too_many_arguments)]
fn validate_common_service(
    command: Option<&CommandSpec>,
    entrypoint: Option<&CommandSpec>,
    workdir: Option<&str>,
    user: Option<&str>,
    environment: Option<&EnvironmentSpec>,
    depends_on: Option<&DependsOn>,
    expose: Option<&[String]>,
    volumes: Option<&[VolumeMount]>,
    tmpfs: Option<&OneOrManyStrings>,
    labels: Option<&Labels>,
    cpus: Option<&LiteralScalar>,
    mem_limit: Option<&LiteralScalar>,
    path: &str,
    declared_volumes: &BTreeSet<String>,
    services: &BTreeSet<ComposeServiceName>,
) -> Result<(), EvalExecutionError> {
    validate_command(command, &format!("{path}.command"))?;
    validate_command(entrypoint, &format!("{path}.entrypoint"))?;
    if let Some(workdir) = workdir {
        validate_container_path(workdir, &format!("{path}.working_dir"))?;
    }
    if user.is_some_and(str::is_empty) {
        return Err(policy_error(format!("{path}.user"), "must not be empty"));
    }
    validate_environment(environment, &format!("{path}.environment"))?;
    validate_dependencies(depends_on, path, services)?;
    if let Some(exposed) = expose {
        for (index, value) in exposed.iter().enumerate() {
            if value.is_empty() || value.contains(':') {
                return Err(policy_error(
                    format!("{path}.expose[{index}]"),
                    "must be a container port without a host mapping",
                ));
            }
        }
    }
    if let Some(mounts) = volumes {
        for (index, mount) in mounts.iter().enumerate() {
            validate_authored_mount(mount, &format!("{path}.volumes[{index}]"), declared_volumes)?;
        }
    }
    validate_tmpfs(tmpfs, &format!("{path}.tmpfs"))?;
    validate_labels(labels, &format!("{path}.labels"), false)?;
    validate_positive_scalar(cpus, &format!("{path}.cpus"))?;
    validate_positive_scalar(mem_limit, &format!("{path}.mem_limit"))
}

fn validate_common_canonical_service(
    service: &CanonicalService,
    path: &str,
    declared_volumes: &BTreeSet<String>,
    services: &BTreeSet<ComposeServiceName>,
) -> Result<(), EvalExecutionError> {
    validate_command(service.command.as_ref(), &format!("{path}.command"))?;
    validate_command(service.entrypoint.as_ref(), &format!("{path}.entrypoint"))?;
    if let Some(workdir) = service.working_dir.as_deref() {
        validate_container_path(workdir, &format!("{path}.working_dir"))?;
    }
    validate_environment(service.environment.as_ref(), &format!("{path}.environment"))?;
    validate_dependencies(service.depends_on.as_ref(), path, services)?;
    if let Some(mounts) = service.volumes.as_deref() {
        for (index, mount) in mounts.iter().enumerate() {
            validate_canonical_sidecar_mount(
                mount,
                &format!("{path}.volumes[{index}]"),
                declared_volumes,
            )?;
        }
    }
    validate_tmpfs(service.tmpfs.as_ref(), &format!("{path}.tmpfs"))?;
    validate_labels(service.labels.as_ref(), &format!("{path}.labels"), false)?;
    validate_positive_scalar(service.cpus.as_ref(), &format!("{path}.cpus"))?;
    validate_positive_scalar(service.mem_limit.as_ref(), &format!("{path}.mem_limit"))
}

fn validate_command(command: Option<&CommandSpec>, path: &str) -> Result<(), EvalExecutionError> {
    match command {
        Some(CommandSpec::String(value)) if value.trim().is_empty() => {
            Err(policy_error(path, "must not be empty"))
        }
        Some(CommandSpec::List(values))
            if values.is_empty() || values.iter().any(|value| value.trim().is_empty()) =>
        {
            Err(policy_error(path, "must be a nonempty command"))
        }
        _ => Ok(()),
    }
}

fn validate_environment(
    environment: Option<&EnvironmentSpec>,
    path: &str,
) -> Result<(), EvalExecutionError> {
    match environment {
        Some(EnvironmentSpec::Map(bindings)) => {
            for (name, value) in bindings {
                validate_env_name(name)
                    .map_err(|reason| policy_error(format!("{path}.{name}"), reason))?;
                if !value.is_valid() {
                    return Err(policy_error(
                        format!("{path}.{name}"),
                        "must be a finite literal",
                    ));
                }
            }
        }
        Some(EnvironmentSpec::List(bindings)) => {
            for (index, binding) in bindings.iter().enumerate() {
                let Some((name, _)) = binding.split_once('=') else {
                    return Err(policy_error(
                        format!("{path}[{index}]"),
                        "must contain a literal value and may not inherit from the host",
                    ));
                };
                validate_env_name(name)
                    .map_err(|reason| policy_error(format!("{path}[{index}]"), reason))?;
            }
        }
        None => {}
    }
    Ok(())
}

fn validate_dependencies(
    dependencies: Option<&DependsOn>,
    path: &str,
    services: &BTreeSet<ComposeServiceName>,
) -> Result<Vec<ComposeServiceName>, EvalExecutionError> {
    let mut normalized = Vec::new();
    let mut seen = BTreeSet::new();
    if let Some(dependencies) = dependencies {
        for dependency in dependencies.names() {
            let name = ComposeServiceName::parse(dependency).map_err(|reason| {
                policy_error(format!("{path}.depends_on.{dependency}"), reason)
            })?;
            if name.as_str() == "main" || !services.contains(&name) {
                return Err(policy_error(
                    format!("{path}.depends_on.{dependency}"),
                    "dependency must name another declared service",
                ));
            }
            if !seen.insert(name.clone()) {
                return Err(policy_error(
                    format!("{path}.depends_on.{dependency}"),
                    "dependency is duplicated",
                ));
            }
            normalized.push(name);
        }
    }
    Ok(normalized)
}

fn validate_authored_main_dependencies(
    dependencies: Option<&DependsOn>,
    path: &str,
    services: &BTreeSet<ComposeServiceName>,
) -> Result<MainDependencyContract, EvalExecutionError> {
    let names = validate_dependencies(dependencies, path, services)?;
    let canonical = match dependencies {
        None => None,
        Some(DependsOn::List(dependencies)) => Some(
            dependencies
                .iter()
                .map(|name| (name.clone(), default_dependency_condition()))
                .collect(),
        ),
        Some(DependsOn::Map(dependencies)) => {
            let mut canonical = BTreeMap::new();
            for (name, condition) in dependencies {
                if condition.condition.is_some()
                    || condition.restart.is_some()
                    || condition.required.is_some()
                {
                    return Err(policy_error(
                        format!("{path}.depends_on.{name}"),
                        "authored main dependencies may name services only",
                    ));
                }
                canonical.insert(name.clone(), default_dependency_condition());
            }
            Some(canonical)
        }
    };
    Ok(MainDependencyContract { canonical, names })
}

fn validate_canonical_main_dependencies(
    dependencies: Option<&DependsOn>,
    expected: &MainDependencyContract,
    path: &str,
) -> Result<(), EvalExecutionError> {
    let actual = match dependencies {
        None => None,
        Some(DependsOn::Map(dependencies)) => Some(dependencies),
        Some(DependsOn::List(_)) => {
            return Err(policy_error(
                format!("{path}.depends_on"),
                "canonical main dependencies are not normalized",
            ));
        }
    };
    if actual != expected.canonical.as_ref() {
        return Err(policy_error(
            format!("{path}.depends_on"),
            "canonical main dependencies differ from the authored overlay",
        ));
    }
    Ok(())
}

fn default_dependency_condition() -> DependencyCondition {
    DependencyCondition {
        condition: Some("service_started".to_owned()),
        restart: None,
        required: Some(true),
    }
}

fn validate_build(
    build: &BuildSpec,
    path: &str,
    environment_root: &Path,
) -> Result<(), EvalExecutionError> {
    let (context, dockerfile, args, target) = match build {
        BuildSpec::Context(context) => (context.as_str(), "Dockerfile", None, None),
        BuildSpec::Detailed(details) => (
            details.context.as_str(),
            details.dockerfile.as_deref().unwrap_or("Dockerfile"),
            Some(&details.args),
            details.target.as_deref(),
        ),
    };
    if looks_remote(context) {
        return Err(policy_error(
            format!("{path}.context"),
            "remote build contexts are forbidden",
        ));
    }
    let root = environment_root.canonicalize().map_err(|error| {
        policy_error(
            format!("{path}.context"),
            format!("environment root cannot be resolved: {error}"),
        )
    })?;
    let context = resolve_local_path(&root, Path::new(context), &format!("{path}.context"))?;
    if !context.is_dir() {
        return Err(policy_error(
            format!("{path}.context"),
            "build context must be a directory",
        ));
    }
    let dockerfile = resolve_local_path(
        &context,
        Path::new(dockerfile),
        &format!("{path}.dockerfile"),
    )?;
    if !dockerfile.is_file() {
        return Err(policy_error(
            format!("{path}.dockerfile"),
            "Dockerfile must be a regular file",
        ));
    }
    if let Some(args) = args {
        for (name, value) in args {
            if name.is_empty() || !value.is_valid() {
                return Err(policy_error(
                    format!("{path}.args.{name}"),
                    "build argument must be a finite named literal",
                ));
            }
        }
    }
    if target.is_some_and(str::is_empty) {
        return Err(policy_error(format!("{path}.target"), "must not be empty"));
    }
    Ok(())
}

fn resolve_local_path(
    base: &Path,
    candidate: &Path,
    path: &str,
) -> Result<PathBuf, EvalExecutionError> {
    if candidate.components().any(|component| {
        matches!(
            component,
            Component::ParentDir | Component::Prefix(_) | Component::RootDir
        )
    }) && !candidate.is_absolute()
    {
        return Err(policy_error(path, "path must not escape its owned root"));
    }
    let joined = if candidate.is_absolute() {
        candidate.to_owned()
    } else {
        base.join(candidate)
    };
    let resolved = joined
        .canonicalize()
        .map_err(|error| policy_error(path, format!("path cannot be resolved: {error}")))?;
    if !resolved.starts_with(base) {
        return Err(policy_error(path, "path escapes its owned root"));
    }
    Ok(resolved)
}

fn looks_remote(value: &str) -> bool {
    value.contains("://") || value.starts_with("git@") || value.starts_with("github.com/")
}

fn validate_authored_mount(
    mount: &VolumeMount,
    path: &str,
    declared_volumes: &BTreeSet<String>,
) -> Result<(), EvalExecutionError> {
    match mount {
        VolumeMount::Short(value) => {
            let parts = value.split(':').collect::<Vec<_>>();
            let (source, target, mode) = match parts.as_slice() {
                [target] => (None, *target, None),
                [source, target] => (Some(*source), *target, None),
                [source, target, mode] => (Some(*source), *target, Some(*mode)),
                _ => return Err(policy_error(path, "invalid volume mount syntax")),
            };
            if let Some(source) = source {
                validate_named_volume(source, &format!("{path}.source"), declared_volumes)?;
            }
            validate_container_path(target, &format!("{path}.target"))?;
            if mode.is_some_and(|mode| !matches!(mode, "ro" | "rw")) {
                return Err(policy_error(
                    format!("{path}.mode"),
                    "only ro or rw volume modes are accepted",
                ));
            }
        }
        VolumeMount::Long(mount) => {
            if mount.mount_type.as_deref().unwrap_or("volume") != "volume" {
                return Err(policy_error(path, "bind mounts are forbidden"));
            }
            if let Some(source) = mount.source.as_deref() {
                validate_named_volume(source, &format!("{path}.source"), declared_volumes)?;
            }
            validate_container_path(&mount.target, &format!("{path}.target"))?;
        }
    }
    Ok(())
}

fn validate_canonical_sidecar_mount(
    mount: &CanonicalVolumeMount,
    path: &str,
    declared_volumes: &BTreeSet<String>,
) -> Result<(), EvalExecutionError> {
    match mount {
        CanonicalVolumeMount::Short(value) => {
            validate_authored_mount(&VolumeMount::Short(value.clone()), path, declared_volumes)
        }
        CanonicalVolumeMount::Long(mount) => {
            if mount.mount_type != "volume" || mount.bind.is_some() {
                return Err(policy_error(path, "sidecar bind mounts are forbidden"));
            }
            if let Some(source) = mount.source.as_deref() {
                validate_named_volume(source, &format!("{path}.source"), declared_volumes)?;
            }
            validate_container_path(&mount.target, &format!("{path}.target"))
        }
    }
}

fn validate_generated_main_mount(
    mount: &CanonicalVolumeMount,
    expected: &ExpectedGeneratedMount,
    path: &str,
) -> Result<(), EvalExecutionError> {
    let CanonicalVolumeMount::Long(mount) = mount else {
        return Err(policy_error(
            path,
            "generated workspace mount must use long syntax",
        ));
    };
    if mount.mount_type != expected.mount_type {
        return Err(policy_error(
            format!("{path}.type"),
            "generated workspace mount type differs from runtime authority",
        ));
    }
    if mount.source.as_ref() != Some(&expected.source) {
        return Err(policy_error(
            format!("{path}.source"),
            "generated workspace source differs from runtime authority",
        ));
    }
    if mount.target != expected.target {
        return Err(policy_error(
            format!("{path}.target"),
            "generated workspace target differs from runtime authority",
        ));
    }
    if mount.read_only.is_some() {
        return Err(policy_error(
            format!("{path}.read_only"),
            "generated workspace mode differs from runtime authority",
        ));
    }
    if mount.bind.is_some() {
        return Err(policy_error(
            format!("{path}.bind"),
            "generated workspace bind options differ from runtime authority",
        ));
    }
    if mount.volume.is_some() {
        return Err(policy_error(
            format!("{path}.volume"),
            "generated workspace volume options differ from runtime authority",
        ));
    }
    Ok(())
}

fn canonical_environment_matches(
    actual: Option<&EnvironmentSpec>,
    expected: &BTreeMap<String, String>,
) -> bool {
    if expected.is_empty() {
        return actual.is_none();
    }
    canonical_string_map(actual).as_ref() == Some(expected)
}

fn canonical_string_map(values: Option<&EnvironmentSpec>) -> Option<BTreeMap<String, String>> {
    let EnvironmentSpec::Map(values) = values? else {
        return None;
    };
    values
        .iter()
        .map(|(name, value)| Some((name.clone(), canonical_string(Some(value))?.to_owned())))
        .collect()
}

fn canonical_label_map(values: Option<&Labels>) -> Option<BTreeMap<String, String>> {
    let Labels::Map(values) = values? else {
        return None;
    };
    values
        .iter()
        .map(|(name, value)| Some((name.clone(), canonical_string(Some(value))?.to_owned())))
        .collect()
}

fn canonical_string(value: Option<&LiteralScalar>) -> Option<&str> {
    match value? {
        LiteralScalar::String(value) => Some(value),
        _ => None,
    }
}

fn canonical_u64(value: Option<&LiteralScalar>) -> Option<u64> {
    match value? {
        LiteralScalar::Signed(value) => u64::try_from(*value).ok(),
        LiteralScalar::Unsigned(value) => Some(*value),
        LiteralScalar::Float(value)
            if value.is_finite() && value.fract() == 0.0 && *value >= 0.0 =>
        {
            Some(*value as u64)
        }
        _ => None,
    }
}

fn canonical_networks_match(
    actual: Option<&BTreeMap<String, Option<CanonicalServiceNetwork>>>,
    expected: &[String],
) -> bool {
    let Some(actual) = actual else {
        return false;
    };
    actual.len() == expected.len()
        && expected
            .iter()
            .all(|name| matches!(actual.get(name), Some(None)))
}

fn validate_named_volume(
    name: &str,
    path: &str,
    declared_volumes: &BTreeSet<String>,
) -> Result<(), EvalExecutionError> {
    if !is_safe_resource_name(name) || !declared_volumes.contains(name) {
        return Err(policy_error(
            path,
            "volume source must be a declared project-owned volume",
        ));
    }
    Ok(())
}

fn validate_volume_definitions<'a>(
    names: impl Iterator<Item = &'a String>,
) -> Result<(), EvalExecutionError> {
    for name in names {
        if !is_safe_resource_name(name) {
            return Err(policy_error(
                format!("volumes.{name}"),
                "invalid project volume name",
            ));
        }
    }
    Ok(())
}

fn is_safe_resource_name(value: &str) -> bool {
    let mut bytes = value.bytes();
    matches!(bytes.next(), Some(b'a'..=b'z' | b'A'..=b'Z' | b'0'..=b'9'))
        && bytes.all(
            |byte| matches!(byte, b'a'..=b'z' | b'A'..=b'Z' | b'0'..=b'9' | b'_' | b'.' | b'-'),
        )
}

fn validate_container_path(value: &str, path: &str) -> Result<(), EvalExecutionError> {
    let candidate = Path::new(value);
    if value == "/"
        || !candidate.is_absolute()
        || candidate
            .components()
            .any(|component| !matches!(component, Component::RootDir | Component::Normal(_)))
    {
        return Err(policy_error(
            path,
            "must be an absolute non-root container path without . or ..",
        ));
    }
    Ok(())
}

fn validate_tmpfs(tmpfs: Option<&OneOrManyStrings>, path: &str) -> Result<(), EvalExecutionError> {
    let values = match tmpfs {
        Some(OneOrManyStrings::One(value)) => std::slice::from_ref(value),
        Some(OneOrManyStrings::Many(values)) => values.as_slice(),
        None => return Ok(()),
    };
    for (index, value) in values.iter().enumerate() {
        let target = value.split(':').next().unwrap_or_default();
        validate_container_path(target, &format!("{path}[{index}]"))?;
    }
    Ok(())
}

fn validate_labels(
    labels: Option<&Labels>,
    path: &str,
    allow_reserved: bool,
) -> Result<(), EvalExecutionError> {
    match labels {
        Some(Labels::Map(labels)) => {
            for name in labels.keys() {
                if !allow_reserved && is_reserved_label(name) {
                    return Err(policy_error(
                        format!("{path}.{name}"),
                        "AIPerf labels are reserved",
                    ));
                }
            }
        }
        Some(Labels::List(labels)) => {
            for (index, label) in labels.iter().enumerate() {
                let name = label
                    .split_once('=')
                    .map_or(label.as_str(), |(name, _)| name);
                if !allow_reserved && is_reserved_label(name) {
                    return Err(policy_error(
                        format!("{path}[{index}]"),
                        "AIPerf labels are reserved",
                    ));
                }
            }
        }
        None => {}
    }
    Ok(())
}

fn is_reserved_label(name: &str) -> bool {
    let name = name.to_ascii_lowercase();
    name.starts_with("aiperf.") || name.starts_with("com.nvidia.aiperf.")
}

fn validate_positive_scalar(
    value: Option<&LiteralScalar>,
    path: &str,
) -> Result<(), EvalExecutionError> {
    if value.is_some_and(|value| !value.is_positive()) {
        return Err(policy_error(path, "must be a positive finite limit"));
    }
    Ok(())
}

fn validate_default_networks(
    networks: Option<&BTreeMap<String, Option<CanonicalServiceNetwork>>>,
    path: &str,
) -> Result<(), EvalExecutionError> {
    let Some(networks) = networks else {
        return Err(policy_error(
            format!("{path}.networks"),
            "canonical service is missing the default network",
        ));
    };
    if networks.keys().map(String::as_str).ne([DEFAULT_NETWORK]) {
        return Err(policy_error(
            format!("{path}.networks"),
            "service may use only the default project network",
        ));
    }
    Ok(())
}

fn validate_service_set(
    actual: &BTreeSet<ComposeServiceName>,
    expected: &BTreeSet<ComposeServiceName>,
) -> Result<(), EvalExecutionError> {
    if actual != expected {
        return Err(policy_error(
            "services",
            format!(
                "canonical services do not match the imported plan: expected {:?}, found {:?}",
                expected
                    .iter()
                    .map(ComposeServiceName::as_str)
                    .collect::<Vec<_>>(),
                actual
                    .iter()
                    .map(ComposeServiceName::as_str)
                    .collect::<Vec<_>>()
            ),
        ));
    }
    Ok(())
}

fn reject_dotenv_files(environment_root: &Path) -> Result<(), EvalExecutionError> {
    fn visit(root: &Path, current: &Path) -> Result<(), EvalExecutionError> {
        let entries = fs::read_dir(current).map_err(|error| {
            policy_error(
                "environment",
                format!("cannot inspect environment tree: {error}"),
            )
        })?;
        for entry in entries {
            let entry = entry.map_err(|error| {
                policy_error(
                    "environment",
                    format!("cannot inspect environment tree: {error}"),
                )
            })?;
            let path = entry.path();
            if entry.file_name() == ".env" {
                let relative = path.strip_prefix(root).unwrap_or(&path);
                return Err(policy_error(
                    format!("environment/{}", relative.display()),
                    ".env files are forbidden",
                ));
            }
            let file_type = entry.file_type().map_err(|error| {
                policy_error(
                    "environment",
                    format!("cannot inspect environment entry: {error}"),
                )
            })?;
            if file_type.is_dir() {
                visit(root, &path)?;
            }
        }
        Ok(())
    }

    visit(environment_root, environment_root)
}

fn reject_interpolation_in<T: Serialize>(value: &T) -> Result<(), EvalExecutionError> {
    let value = serde_yaml::to_value(value)
        .map_err(|error| policy_error("compose", format!("cannot inspect literals: {error}")))?;
    reject_interpolation_value(&value, "compose")
}

fn reject_interpolation_value(
    value: &serde_yaml::Value,
    path: &str,
) -> Result<(), EvalExecutionError> {
    match value {
        serde_yaml::Value::String(value) => reject_interpolation(value, path),
        serde_yaml::Value::Sequence(values) => {
            for (index, value) in values.iter().enumerate() {
                reject_interpolation_value(value, &format!("{path}[{index}]"))?;
            }
            Ok(())
        }
        serde_yaml::Value::Mapping(values) => {
            for (key, value) in values {
                let key = key.as_str().unwrap_or("<non-string-key>");
                reject_interpolation(key, &format!("{path}.{key}"))?;
                reject_interpolation_value(value, &format!("{path}.{key}"))?;
            }
            Ok(())
        }
        _ => Ok(()),
    }
}

fn reject_interpolation(value: &str, path: &str) -> Result<(), EvalExecutionError> {
    let bytes = value.as_bytes();
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] != b'$' {
            index += 1;
            continue;
        }
        if bytes.get(index + 1) == Some(&b'$') {
            index += 2;
            continue;
        }
        let next = bytes.get(index + 1).copied();
        if next == Some(b'{') || next.is_some_and(|byte| byte == b'_' || byte.is_ascii_alphabetic())
        {
            return Err(policy_error(
                path,
                "unescaped Compose interpolation is forbidden",
            ));
        }
        index += 1;
    }
    Ok(())
}

fn escape_compose_dollars(value: &str) -> String {
    value.replace('$', "$$")
}

fn decode_error(root: &str, error: &str) -> EvalExecutionError {
    let first_line = error.lines().next().unwrap_or(error);
    if let Some((prefix, remainder)) = first_line.split_once(": unknown field `")
        && let Some((field, _)) = remainder.split_once('`')
    {
        let prefix = if prefix == "." || prefix.is_empty() {
            root.to_owned()
        } else {
            prefix.to_owned()
        };
        return policy_error(format!("{prefix}.{field}"), first_line);
    }
    if let Some((prefix, remainder)) = first_line.split_once(": null environment value for `")
        && let Some((name, _)) = remainder.split_once('`')
    {
        return policy_error(format!("{prefix}.{name}"), first_line);
    }
    policy_error(root, first_line)
}

fn policy_error(path: impl AsRef<str>, reason: impl AsRef<str>) -> EvalExecutionError {
    EvalExecutionError::InvalidWorkspace(format!(
        "Compose policy at {}: {}",
        path.as_ref(),
        reason.as_ref()
    ))
}

#[cfg(test)]
mod tests {
    use std::{
        collections::{BTreeMap, BTreeSet},
        fs,
        path::Path,
        time::Duration,
    };

    use serde_yaml::Value;

    use super::{
        render_generated_main_compose, validate_authored_compose, validate_canonical_compose_json,
        validate_provider_compose_config,
    };
    use crate::eval::{
        ArtifactDigest, ComposeProjectPlan, ComposeServiceName, ContainerResources, EnvBinding,
        EnvironmentPlan, ImageSource, NetworkPolicy,
    };

    #[test]
    fn compose_policy_accepts_literal_sidecars_and_records_main_dependencies() {
        let temporary = tempfile::tempdir().unwrap();
        let environment_root = temporary.path().join("environment");
        fs::create_dir_all(environment_root.join("worker")).unwrap();
        fs::write(
            environment_root.join("worker/Dockerfile.worker"),
            "FROM scratch\n",
        )
        .unwrap();
        let plan = compose_plan(&["api", "main", "worker"]);
        let yaml = br#"
services:
  main:
    depends_on: [api, worker]
  api:
    image: example/api:fixture
    command: ["serve", "--port", "8080"]
    environment:
      MODE: fixture
    expose: ["8080"]
    labels:
      fixture.role: api
  worker:
    build:
      context: ./worker
      dockerfile: Dockerfile.worker
      args:
        MODE: fixture
    depends_on: [api]
    volumes:
      - data:/var/lib/worker:ro
volumes:
  data: {}
"#;

        let validated = validate_authored_compose(yaml, &plan, &environment_root).unwrap();

        assert_eq!(
            validated
                .services
                .iter()
                .map(ComposeServiceName::as_str)
                .collect::<Vec<_>>(),
            ["api", "main", "worker"]
        );
        assert_eq!(
            validated
                .main_dependencies
                .iter()
                .map(ComposeServiceName::as_str)
                .collect::<Vec<_>>(),
            ["api", "worker"]
        );
    }

    #[test]
    fn compose_policy_accepts_equivalent_canonical_json_and_rejects_service_drift() {
        let temporary = tempfile::tempdir().unwrap();
        let environment_root = temporary.path().join("environment");
        let workspace = temporary.path().join("agent-workspace");
        fs::create_dir_all(environment_root.join("worker")).unwrap();
        fs::create_dir(&workspace).unwrap();
        fs::write(
            environment_root.join("worker/Dockerfile.worker"),
            "FROM scratch\n",
        )
        .unwrap();
        let plan = compose_plan(&["api", "main", "worker"]);
        let context = environment_root.join("worker");
        let labels = BTreeMap::from([
            ("aiperf.project".to_owned(), "compose-fixture".to_owned()),
            ("aiperf.run".to_owned(), "run-17".to_owned()),
        ]);
        let generated = render_generated_main_compose(
            "aiperf-main:abc",
            &labels,
            &environment_plan(),
            &workspace,
        )
        .unwrap();
        let authored = validate_authored_compose(
            br#"
services:
  main:
    depends_on: [api]
  api:
    image: example/api:fixture
  worker:
    build:
      context: ./worker
      dockerfile: Dockerfile.worker
volumes:
  data: {}
"#,
            &plan,
            &environment_root,
        )
        .unwrap();
        let canonical = format!(
            r#"{{
  "services": {{
    "main": {{
      "image": "aiperf-main:abc",
      "command": ["sleep", "infinity"],
      "depends_on": {{"api": {{"condition": "service_started", "required": true}}}},
      "working_dir": "/workspace",
      "user": "1000:1000",
      "environment": {{"PUBLIC": "fixture"}},
      "labels": {{"aiperf.project": "compose-fixture", "aiperf.run": "run-17"}},
      "cpus": 2,
      "mem_limit": "512m",
      "networks": {{"default": null}},
      "volumes": [{{"type": "bind", "source": {}, "target": "/workspace"}}],
      "restart": "no"
    }},
    "api": {{"image": "example/api:fixture", "networks": {{"default": null}}}},
    "worker": {{
      "build": {{"context": {}, "dockerfile": "Dockerfile.worker", "args": {{"MODE": "fixture"}}}},
      "networks": {{"default": null}},
      "volumes": [{{"type": "volume", "source": "data", "target": "/var/lib/worker", "read_only": true}}]
    }}
  }},
  "networks": {{"default": {{"name": "fixture_default"}}}},
  "volumes": {{"data": {{"name": "fixture_data"}}}}
}}"#,
            serde_json::to_string(&workspace).unwrap(),
            serde_json::to_string(&context).unwrap(),
        );

        let validated = validate_provider_compose_config(
            canonical.as_bytes(),
            &plan,
            &environment_root,
            &generated,
            &authored,
        )
        .unwrap();
        assert_eq!(validated.services, plan.services().clone());
        assert_eq!(validated.main_dependencies[0].as_str(), "api");

        let drifted = canonical.replace("\"api\": {\"image\"", "\"unexpected\": {\"image\"");
        let error = validate_canonical_compose_json(
            drifted.as_bytes(),
            &plan,
            &environment_root,
            generated.expected_main(),
            authored.main_dependency_contract(),
        )
        .unwrap_err();
        assert!(error.to_string().contains("services"), "{error}");
    }

    #[test]
    fn compose_policy_rejects_each_generated_main_authority_drift() {
        let temporary = tempfile::tempdir().unwrap();
        let environment_root = temporary.path().join("environment");
        let workspace = temporary.path().join("agent-workspace");
        fs::create_dir(&environment_root).unwrap();
        fs::create_dir(&workspace).unwrap();
        let plan = compose_plan(&["api", "main"]);
        let labels = BTreeMap::from([
            ("aiperf.project".to_owned(), "compose-fixture".to_owned()),
            ("aiperf.run".to_owned(), "run-17".to_owned()),
        ]);
        let generated = render_generated_main_compose(
            "aiperf-main:abc",
            &labels,
            &environment_plan(),
            &workspace,
        )
        .unwrap();
        let authored = validate_authored_compose(
            b"services:\n  main: {}\n  api:\n    image: example/api:fixture\n",
            &plan,
            &environment_root,
        )
        .unwrap();
        let canonical = serde_json::json!({
            "services": {
                "main": {
                    "image": "aiperf-main:abc",
                    "command": ["sleep", "infinity"],
                    "working_dir": "/workspace",
                    "user": "1000:1000",
                    "environment": {"PUBLIC": "fixture"},
                    "labels": {
                        "aiperf.project": "compose-fixture",
                        "aiperf.run": "run-17"
                    },
                    "cpus": 2,
                    "mem_limit": "512m",
                    "networks": {"default": null},
                    "volumes": [{
                        "type": "bind",
                        "source": workspace,
                        "target": "/workspace"
                    }],
                    "restart": "no"
                },
                "api": {
                    "image": "example/api:fixture",
                    "networks": {"default": null}
                }
            },
            "networks": {"default": {"name": "fixture_default"}}
        });
        let drifts = [
            (
                "services.main.image",
                "/services/main/image",
                serde_json::json!("attacker:latest"),
            ),
            (
                "services.main.command",
                "/services/main/command",
                serde_json::json!(["mine"]),
            ),
            (
                "services.main.user",
                "/services/main/user",
                serde_json::json!("root"),
            ),
            (
                "services.main.working_dir",
                "/services/main/working_dir",
                serde_json::Value::Null,
            ),
            (
                "services.main.environment",
                "/services/main/environment",
                serde_json::json!({"PUBLIC": "changed"}),
            ),
            (
                "services.main.labels",
                "/services/main/labels",
                serde_json::json!({"aiperf.project": "attacker", "aiperf.run": "run-17"}),
            ),
            (
                "services.main.cpus",
                "/services/main/cpus",
                serde_json::json!(8),
            ),
            (
                "services.main.mem_limit",
                "/services/main/mem_limit",
                serde_json::json!("8g"),
            ),
            (
                "services.main.volumes[0].source",
                "/services/main/volumes/0/source",
                serde_json::json!("/"),
            ),
            (
                "services.main.volumes[0].target",
                "/services/main/volumes/0/target",
                serde_json::json!("/attacker"),
            ),
            (
                "services.main.volumes[0].read_only",
                "/services/main/volumes",
                serde_json::json!([{
                    "type": "bind",
                    "source": workspace,
                    "target": "/workspace",
                    "read_only": true
                }]),
            ),
            (
                "services.main.networks",
                "/services/main/networks",
                serde_json::json!({"other": null}),
            ),
            (
                "services.main.restart",
                "/services/main/restart",
                serde_json::json!("always"),
            ),
        ];

        for (path, pointer, replacement) in drifts {
            let mut drifted = canonical.clone();
            *drifted.pointer_mut(pointer).unwrap() = replacement;
            let error = validate_canonical_compose_json(
                &serde_json::to_vec(&drifted).unwrap(),
                &plan,
                &environment_root,
                generated.expected_main(),
                authored.main_dependency_contract(),
            )
            .expect_err("generated main authority drift must be rejected");
            assert!(
                error.to_string().contains(path),
                "expected {path:?} in {error} for {pointer}"
            );
        }
    }

    #[test]
    fn compose_policy_rejects_added_removed_or_modified_main_dependencies() {
        let temporary = tempfile::tempdir().unwrap();
        let environment_root = temporary.path().join("environment");
        let workspace = temporary.path().join("agent-workspace");
        fs::create_dir(&environment_root).unwrap();
        fs::create_dir(&workspace).unwrap();
        let plan = compose_plan(&["api", "main", "worker"]);
        let generated = render_generated_main_compose(
            "aiperf-main:abc",
            &BTreeMap::from([("aiperf.project".to_owned(), "fixture".to_owned())]),
            &environment_plan(),
            &workspace,
        )
        .unwrap();
        let authored = validate_authored_compose(
            b"services:\n  main:\n    depends_on: [api]\n  api:\n    image: example/api:fixture\n  worker:\n    image: example/worker:fixture\n",
            &plan,
            &environment_root,
        )
        .unwrap();
        let canonical = serde_json::json!({
            "services": {
                "main": {
                    "image": "aiperf-main:abc",
                    "command": ["sleep", "infinity"],
                    "depends_on": {
                        "api": {"condition": "service_started", "required": true}
                    },
                    "working_dir": "/workspace",
                    "user": "1000:1000",
                    "environment": {"PUBLIC": "fixture"},
                    "labels": {"aiperf.project": "fixture"},
                    "cpus": 2,
                    "mem_limit": "512m",
                    "networks": {"default": null},
                    "volumes": [{
                        "type": "bind",
                        "source": workspace,
                        "target": "/workspace"
                    }],
                    "restart": "no"
                },
                "api": {
                    "image": "example/api:fixture",
                    "networks": {"default": null}
                },
                "worker": {
                    "image": "example/worker:fixture",
                    "networks": {"default": null}
                }
            },
            "networks": {"default": {"name": "fixture_default"}}
        });
        let drifts = [
            Some(serde_json::json!({
                "api": {"condition": "service_started", "required": true},
                "worker": {"condition": "service_started", "required": true}
            })),
            None,
            Some(serde_json::json!({
                "api": {"condition": "service_healthy", "required": true}
            })),
            Some(serde_json::json!({
                "api": {"condition": "service_started", "required": true, "restart": true}
            })),
            Some(serde_json::json!({
                "api": {"condition": "service_started", "required": false}
            })),
        ];

        for dependencies in drifts {
            let mut drifted = canonical.clone();
            let main = drifted["services"]["main"].as_object_mut().unwrap();
            if let Some(dependencies) = dependencies {
                main.insert("depends_on".to_owned(), dependencies);
            } else {
                main.remove("depends_on");
            }
            let error = validate_canonical_compose_json(
                &serde_json::to_vec(&drifted).unwrap(),
                &plan,
                &environment_root,
                generated.expected_main(),
                authored.main_dependency_contract(),
            )
            .expect_err("canonical main dependencies must match the authored overlay");
            assert!(
                error.to_string().contains("services.main.depends_on"),
                "{error}"
            );
        }
    }

    #[test]
    fn compose_policy_renders_generated_main_from_runtime_authority_without_secrets() {
        let temporary = tempfile::tempdir().unwrap();
        let workspace = temporary.path().join("agent-workspace");
        fs::create_dir(&workspace).unwrap();
        let environment = environment_plan();
        let labels = BTreeMap::from([
            ("aiperf.project".to_owned(), "compose-fixture".to_owned()),
            ("aiperf.run".to_owned(), "run-17".to_owned()),
        ]);

        let rendered =
            render_generated_main_compose("aiperf-main:abc", &labels, &environment, &workspace)
                .unwrap();
        let document: Value = serde_yaml::from_slice(rendered.bytes()).unwrap();
        let main = &document["services"]["main"];

        assert_eq!(main["image"].as_str(), Some("aiperf-main:abc"));
        assert_eq!(main["command"][0].as_str(), Some("sleep"));
        assert_eq!(main["command"][1].as_str(), Some("infinity"));
        assert_eq!(main["working_dir"].as_str(), Some("/workspace"));
        assert_eq!(main["user"].as_str(), Some("1000:1000"));
        assert_eq!(main["environment"]["PUBLIC"].as_str(), Some("fixture"));
        assert!(main["environment"].get("TOKEN").is_none());
        assert_eq!(main["cpus"].as_u64(), Some(2));
        assert_eq!(main["mem_limit"].as_str(), Some("512m"));
        assert_eq!(main["restart"].as_str(), Some("no"));
        assert_eq!(main["networks"][0].as_str(), Some("default"));
        assert_eq!(main["volumes"][0]["type"].as_str(), Some("bind"));
        assert_eq!(main["volumes"][0]["source"].as_str(), workspace.to_str());
        assert_eq!(main["volumes"][0]["target"].as_str(), Some("/workspace"));
        assert_eq!(
            main["labels"]["aiperf.project"].as_str(),
            Some("compose-fixture")
        );
        let rendered = String::from_utf8(rendered.into_bytes()).unwrap();
        assert!(!rendered.contains("HOST_TOKEN"));
        assert!(!rendered.contains("agent-secret-value"));
    }

    #[test]
    fn compose_policy_rejects_unknown_and_host_facing_fields_at_their_yaml_path() {
        let cases = [
            ("services.api.unknown", "unknown: true"),
            ("services.api.env_file", "env_file: [.env]"),
            ("services.api.ports", "ports: [\"8080:8080\"]"),
            ("services.api.container_name", "container_name: fixed"),
            ("services.api.profiles", "profiles: [debug]"),
            ("services.api.deploy", "deploy: { replicas: 2 }"),
            ("services.api.network_mode", "network_mode: host"),
            ("services.api.dns", "dns: [8.8.8.8]"),
            (
                "services.api.extra_hosts",
                "extra_hosts: [\"host:host-gateway\"]",
            ),
            ("services.api.privileged", "privileged: true"),
            ("services.api.devices", "devices: [\"/dev/null:/dev/null\"]"),
            ("services.api.cap_add", "cap_add: [SYS_ADMIN]"),
            ("services.api.pid", "pid: host"),
            ("services.api.ipc", "ipc: host"),
            ("services.api.uts", "uts: host"),
            (
                "services.api.security_opt",
                "security_opt: [seccomp=unconfined]",
            ),
            ("services.api.userns_mode", "userns_mode: host"),
            ("services.api.restart", "restart: always"),
        ];
        for (path, field) in cases {
            assert_invalid_authored(
                path,
                &format!("services:\n  api:\n    image: api:fixture\n    {field}\n"),
            );
        }

        for (path, field) in [
            ("name", "name: caller-project"),
            ("networks", "networks: { host: { external: true } }"),
            ("secrets", "secrets: { token: { file: ./token } }"),
            ("configs", "configs: { app: { file: ./config } }"),
            ("include", "include: [other.yaml]"),
        ] {
            assert_invalid_authored(
                path,
                &format!("{field}\nservices:\n  api:\n    image: api:fixture\n"),
            );
        }
    }

    #[test]
    fn compose_policy_rejects_main_authority_and_reserved_labels() {
        for (path, field) in [
            ("services.main.image", "image: attacker:latest"),
            ("services.main.build", "build: ."),
            ("services.main.command", "command: [mine]"),
            ("services.main.entrypoint", "entrypoint: [mine]"),
            ("services.main.working_dir", "working_dir: /attacker"),
            ("services.main.user", "user: root"),
            ("services.main.environment", "environment: { OWNED: yes }"),
            ("services.main.volumes", "volumes: [data:/work]"),
            ("services.main.cpus", "cpus: 8"),
            ("services.main.mem_limit", "mem_limit: 8g"),
            (
                "services.main.healthcheck",
                "healthcheck: { test: [CMD, true] }",
            ),
            ("services.main.restart", "restart: always"),
        ] {
            assert_invalid_authored(
                path,
                &format!(
                    "services:\n  main:\n    depends_on: [api]\n    {field}\n  api:\n    image: api:fixture\n"
                ),
            );
        }

        assert_invalid_authored(
            "services.api.labels.aiperf.run",
            "services:\n  api:\n    image: api:fixture\n    labels:\n      aiperf.run: attacker\n",
        );
        assert_invalid_authored(
            "services.main.depends_on.api",
            "services:\n  main:\n    depends_on:\n      api:\n        condition: service_healthy\n  api:\n    image: api:fixture\n",
        );
    }

    #[test]
    fn compose_policy_rejects_interpolation_and_dotenv_but_allows_escaped_dollars() {
        for value in ["$TOKEN", "${TOKEN}", "prefix-${TOKEN}"] {
            assert_invalid_authored(
                "services.api.environment.TOKEN",
                &format!(
                    "services:\n  api:\n    image: api:fixture\n    environment:\n      TOKEN: {value:?}\n"
                ),
            );
        }
        assert_invalid_authored(
            "services.api.environment.TOKEN",
            "services:\n  api:\n    image: api:fixture\n    environment:\n      TOKEN:\n",
        );
        assert_invalid_authored(
            "services.api.environment[0]",
            "services:\n  api:\n    image: api:fixture\n    environment: [TOKEN]\n",
        );

        let temporary = tempfile::tempdir().unwrap();
        let environment_root = temporary.path().join("environment");
        fs::create_dir_all(environment_root.join("nested")).unwrap();
        fs::write(environment_root.join("nested/.env"), "TOKEN=secret\n").unwrap();
        let error = validate_authored_compose(
            b"services:\n  api:\n    image: api:fixture\n",
            &compose_plan(&["api", "main"]),
            &environment_root,
        )
        .unwrap_err();
        assert!(
            error.to_string().contains("environment/nested/.env"),
            "{error}"
        );

        let escaped_root = tempfile::tempdir().unwrap();
        let escaped = validate_authored_compose(
            b"services:\n  api:\n    image: api:fixture\n    command: [\"/bin/sh\", \"-c\", \"echo $$TOKEN\"]\n",
            &compose_plan(&["api", "main"]),
            escaped_root.path(),
        );
        assert!(escaped.is_ok(), "{escaped:?}");
    }

    #[test]
    fn compose_policy_rejects_bind_mounts_and_unowned_or_custom_volumes() {
        for (path, yaml) in [
            (
                "services.api.volumes[0]",
                "services:\n  api:\n    image: api:fixture\n    volumes: [\"./host:/data\"]\n",
            ),
            (
                "services.api.volumes[0].target",
                "services:\n  api:\n    image: api:fixture\n    volumes: [\"data:relative\"]\nvolumes:\n  data: {}\n",
            ),
            (
                "services.api.volumes[0].source",
                "services:\n  api:\n    image: api:fixture\n    volumes: [\"missing:/data\"]\n",
            ),
            (
                "volumes.data.external",
                "services:\n  api:\n    image: api:fixture\n    volumes: [\"data:/data\"]\nvolumes:\n  data:\n    external: true\n",
            ),
            (
                "volumes.data.driver",
                "services:\n  api:\n    image: api:fixture\nvolumes:\n  data:\n    driver: local\n",
            ),
        ] {
            assert_invalid_authored(path, yaml);
        }
    }

    #[test]
    fn compose_policy_rejects_remote_or_escaping_build_inputs() {
        let temporary = tempfile::tempdir().unwrap();
        let environment_root = temporary.path().join("environment");
        fs::create_dir_all(&environment_root).unwrap();
        for (path, build) in [
            (
                "services.api.build.context",
                "https://example.com/context.git",
            ),
            ("services.api.build.context", "../outside"),
            ("services.api.build.dockerfile", "../Dockerfile"),
            ("services.api.build.additional_contexts", "."),
            ("services.api.build.ssh", "."),
            ("services.api.build.secrets", "."),
            ("services.api.build.dockerfile_inline", "FROM scratch"),
        ] {
            let yaml = if path == "services.api.build.context" {
                format!(
                    "services:\n  api:\n    build:\n      {}: {build:?}\n",
                    path.rsplit('.').next().unwrap()
                )
            } else {
                format!(
                    "services:\n  api:\n    build:\n      context: .\n      {}: {build:?}\n",
                    path.rsplit('.').next().unwrap()
                )
            };
            assert_invalid_authored_at(path, &yaml, &environment_root);
        }
    }

    fn assert_invalid_authored(path: &str, yaml: &str) {
        let temporary = tempfile::tempdir().unwrap();
        assert_invalid_authored_at(path, yaml, temporary.path());
    }

    fn assert_invalid_authored_at(path: &str, yaml: &str, environment_root: &Path) {
        let error = validate_authored_compose(
            yaml.as_bytes(),
            &compose_plan(&["api", "main"]),
            environment_root,
        )
        .expect_err("unsafe Compose input must be rejected");
        assert!(
            error.to_string().contains(path),
            "expected {path:?} in {error} for:\n{yaml}"
        );
    }

    fn compose_plan(services: &[&str]) -> ComposeProjectPlan {
        ComposeProjectPlan {
            definition_path: "environment/docker-compose.yaml".to_owned(),
            services: services
                .iter()
                .map(|service| ComposeServiceName::parse(service).unwrap())
                .collect::<BTreeSet<_>>(),
            build_timeout: Duration::from_secs(600),
            startup_timeout: Duration::from_secs(120),
        }
    }

    fn environment_plan() -> EnvironmentPlan {
        EnvironmentPlan {
            image_source: ImageSource::task_dockerfile(ArtifactDigest::from_bytes(b"context")),
            resources: Some(ContainerResources {
                cpus: 2,
                memory_mb: 512,
            }),
            workdir: Some("/workspace".to_owned()),
            user: Some("1000:1000".to_owned()),
            env: BTreeMap::from([
                ("PUBLIC".to_owned(), EnvBinding::parse("fixture").unwrap()),
                (
                    "TOKEN".to_owned(),
                    EnvBinding::parse("${HOST_TOKEN}").unwrap(),
                ),
            ]),
            network: NetworkPolicy::public(),
            healthcheck: None,
        }
    }
}
