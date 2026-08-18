// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Resolution of immutable NativeGraph model bindings through product seams.

use std::{
    cell::RefCell,
    collections::BTreeMap,
    fmt::{self, Display, Formatter},
    rc::Rc,
    sync::Arc,
};

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::value::RawValue;
use url::Url;

use crate::{
    endpoints::RawEndpointConfig,
    engine::registry::ValidatedEndpointProfileV2,
    engine::{
        application::Application,
        execute::load_tokenizer,
        graph_execution::{NativeGraphTransportEvidence, execute_native_graph_trace},
        record_lane::EvalNodeRecordArtifact,
        registry::{NativeTransportExecution, WorkloadRequirements},
    },
    eval::GraphLoweringRequest,
    eval::{
        EnvName, EvalExecutionError, NativeGraphEpisodeCallback, NativeGraphEpisodeLease,
        SecretProvider, SecretValue,
    },
    extensions::AIPerfRegistry,
    multiturn::{EndpointInputTokenCounter, InputTokenCounter},
    transport::{core::ConnectionReuseStrategy, http::config::ClientConfig},
};

use super::{
    GenerationDefaults, ModelBindingId, ModelBindingSpec, ModelCapturePolicy, ModelSecretId,
    NativeGraphPackagePlan, TokenizerBindingSpec, lower_native_graph,
    lowering::{NATIVE_GRAPH_EXECUTION_PROFILE, NATIVE_GRAPH_SOURCE_SCHEMA},
};

/// Strict host-owned configuration for NativeGraph model-secret resolution.
///
/// Package bytes name only logical secret identifiers. This configuration maps
/// each such identifier to a host-secret name immediately before model runtime
/// construction; it deliberately contains neither endpoint policy nor adapter
/// environment variables.
#[derive(Clone, Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelRuntimeConfig {
    /// Version of this strict host-owned configuration document.
    pub version: u32,
    /// Host secret names keyed by package-declared logical model secret IDs.
    #[serde(default)]
    pub secrets: BTreeMap<ModelSecretId, EnvName>,
}

/// Resolves immutable package model bindings through the product registry.
///
/// Implementations must obtain all endpoint and transport behavior from the
/// supplied product composition. They must not construct an evaluation-local
/// HTTP or gRPC client.
pub trait NativeGraphModelBindingResolver: Send + Sync {
    /// Resolves every model binding selected by one immutable package.
    fn resolve(
        &self,
        bindings: &[ModelBindingSpec],
        runtime: &ModelRuntimeConfig,
        secrets: &dyn SecretProvider,
    ) -> Result<ResolvedModelBindingSet, ModelRuntimeError>;
}

/// Current resolver backed by the frozen product registry.
#[derive(Clone)]
pub struct CurrentNativeGraphModelBindingResolver {
    registry: AIPerfRegistry,
}

impl CurrentNativeGraphModelBindingResolver {
    /// Creates a resolver from the process's already frozen product registry.
    pub fn from_registry(registry: AIPerfRegistry) -> Self {
        Self { registry }
    }
}

impl NativeGraphModelBindingResolver for CurrentNativeGraphModelBindingResolver {
    fn resolve(
        &self,
        bindings: &[ModelBindingSpec],
        runtime: &ModelRuntimeConfig,
        secrets: &dyn SecretProvider,
    ) -> Result<ResolvedModelBindingSet, ModelRuntimeError> {
        if runtime.version != 1 {
            return Err(ModelRuntimeError::UnsupportedVersion(runtime.version));
        }

        let mut resolved = BTreeMap::new();
        for binding in bindings {
            let binding_id = binding.id.as_str().to_string();
            if resolved.contains_key(&binding.id) {
                return Err(ModelRuntimeError::DuplicateBinding(binding_id));
            }

            let model_headers = resolve_model_headers(binding, runtime, secrets)?;
            let mut config = RawEndpointConfig {
                urls: binding.urls.clone(),
                streaming: binding.streaming,
                timeout_seconds: binding.request_timeout_ms.get() as f64 / 1_000.0,
                ..RawEndpointConfig::default()
            };
            for (header, value) in &model_headers {
                // The prepared endpoint is the only consumer of this transient
                // request configuration. `ResolvedModelBinding` never exposes
                // these headers to supervised adapter environment construction.
                config
                    .headers
                    .insert(header.clone(), value.exposed().to_string());
            }

            let canonical_endpoint = self
                .registry
                .endpoints()
                .canonical_id(binding.endpoint_factory_id())
                .map_err(|error| ModelRuntimeError::Endpoint {
                    binding: binding_id.clone(),
                    reason: error.to_string(),
                })?
                .clone();
            let prepared = self
                .registry
                .endpoints()
                .prepare(&canonical_endpoint, config)
                .map_err(|error| ModelRuntimeError::Endpoint {
                    binding: binding_id.clone(),
                    reason: error.to_string(),
                })?;

            let transport = self
                .registry
                .transport_factory(binding.transport_factory_id())
                .ok_or_else(|| ModelRuntimeError::UnknownTransport {
                    binding: binding_id.clone(),
                    transport: binding.transport_factory_id().to_string(),
                })?;
            validate_transport_urls(binding, transport.descriptor().url_schemes)?;

            let timeout_ns = binding
                .request_timeout_ms
                .get()
                .checked_mul(1_000_000)
                .and_then(|value| i64::try_from(value).ok())
                .ok_or_else(|| ModelRuntimeError::InvalidTimeout(binding_id.clone()))?;
            let profile = ValidatedEndpointProfileV2 {
                profile_id: binding.endpoint_profile_id.clone(),
                endpoint_id: canonical_endpoint,
                config: prepared.config().to_raw(),
                connection_reuse: ConnectionReuseStrategy::default(),
                client: ClientConfig {
                    max_connect_retries: binding.max_connect_retries,
                    total_timeout_ns: Some(timeout_ns),
                    ..ClientConfig::default()
                },
                session_header: None,
            };
            let tokenizer = ResolvedTokenizerBinding::from_spec(&binding.tokenizer);
            let resolved_binding = ResolvedModelBinding {
                profile,
                transport_id: binding.transport_factory_id.clone(),
                tokenizer,
                model: binding.model.clone(),
                model_headers,
                adapter_environment: BTreeMap::new(),
                generation: binding.generation.clone(),
                capture: binding.capture,
            };
            resolved.insert(binding.id.clone(), resolved_binding);
        }
        Ok(ResolvedModelBindingSet { bindings: resolved })
    }
}

fn resolve_model_headers(
    binding: &ModelBindingSpec,
    runtime: &ModelRuntimeConfig,
    secrets: &dyn SecretProvider,
) -> Result<BTreeMap<String, SecretValue>, ModelRuntimeError> {
    let mut headers = BTreeMap::new();
    for reference in binding.authentication() {
        let name = runtime.secrets.get(&reference.secret).ok_or_else(|| {
            ModelRuntimeError::MissingSecret {
                binding: binding.id.as_str().to_string(),
                secret: reference.secret.as_str().to_string(),
            }
        })?;
        let value = secrets
            .resolve(name)
            .map_err(|error| ModelRuntimeError::Secret {
                binding: binding.id.as_str().to_string(),
                secret: reference.secret.as_str().to_string(),
                reason: error.to_string(),
            })?;
        if headers.insert(reference.header.clone(), value).is_some() {
            return Err(ModelRuntimeError::DuplicateHeader {
                binding: binding.id.as_str().to_string(),
                header: reference.header.clone(),
            });
        }
    }
    Ok(headers)
}

fn validate_transport_urls(
    binding: &ModelBindingSpec,
    schemes: &[&str],
) -> Result<(), ModelRuntimeError> {
    for raw_url in &binding.urls {
        let url = Url::parse(raw_url).map_err(|error| ModelRuntimeError::InvalidUrl {
            binding: binding.id.as_str().to_string(),
            url: raw_url.clone(),
            reason: error.to_string(),
        })?;
        if !schemes.contains(&url.scheme()) {
            return Err(ModelRuntimeError::UnsupportedTransportUrl {
                binding: binding.id.as_str().to_string(),
                transport: binding.transport_factory_id.clone(),
                scheme: url.scheme().to_string(),
            });
        }
    }
    Ok(())
}

/// Resolved immutable model bindings keyed by package model-binding ID.
#[derive(Clone)]
pub struct ResolvedModelBindingSet {
    bindings: BTreeMap<ModelBindingId, ResolvedModelBinding>,
}

impl ResolvedModelBindingSet {
    /// Returns one resolved binding by its package-defined identifier.
    pub fn binding(&self, id: &str) -> Option<&ResolvedModelBinding> {
        self.bindings
            .iter()
            .find_map(|(binding_id, binding)| (binding_id.as_str() == id).then_some(binding))
    }

    /// Iterates bindings in canonical package-binding identifier order.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = (&ModelBindingId, &ResolvedModelBinding)> {
        self.bindings.iter()
    }

    fn engine_inputs(
        &self,
        application: &Application,
    ) -> Result<NativeGraphModelStageInputs, ModelRuntimeError> {
        let profiles = self
            .bindings
            .values()
            .map(|binding| binding.profile.clone())
            .collect::<Vec<_>>();
        let context = application
            .native_graph_context(profiles.clone())
            .map_err(|error| ModelRuntimeError::Application(error.to_string()))?;
        let mut token_counters = BTreeMap::new();
        let mut transports = BTreeMap::new();
        let mut raw_enabled = false;
        let mut default_model = None;
        for (binding_id, binding) in &self.bindings {
            if token_counters
                .insert(
                    binding.profile.profile_id.clone(),
                    binding
                        .tokenizer
                        .input_token_counter(&binding.model)
                        .map_err(|reason| ModelRuntimeError::Tokenizer {
                            binding: binding_id.as_str().to_owned(),
                            reason,
                        })?,
                )
                .is_some()
            {
                return Err(ModelRuntimeError::DuplicateEndpointProfile(
                    binding.profile.profile_id.clone(),
                ));
            }
            let factory = application
                .product_registry()
                .transport_factory(&binding.transport_id)
                .ok_or_else(|| ModelRuntimeError::UnknownTransport {
                    binding: binding_id.as_str().to_owned(),
                    transport: binding.transport_id.clone(),
                })?;
            let config = RawValue::from_string("{}".to_owned()).map_err(|error| {
                ModelRuntimeError::Transport {
                    binding: binding_id.as_str().to_owned(),
                    reason: error.to_string(),
                }
            })?;
            let config = factory
                .validate(&config, &WorkloadRequirements::default())
                .map_err(|error| ModelRuntimeError::Transport {
                    binding: binding_id.as_str().to_owned(),
                    reason: error.to_string(),
                })?;
            let transport = factory
                .native_execution(config.as_ref(), &context)
                .map_err(|error| ModelRuntimeError::Transport {
                    binding: binding_id.as_str().to_owned(),
                    reason: error.to_string(),
                })?
                .ok_or_else(|| ModelRuntimeError::NoNativeTransport {
                    binding: binding_id.as_str().to_owned(),
                    transport: binding.transport_id.clone(),
                })?;
            transports.insert(binding.profile.profile_id.clone(), transport);
            raw_enabled |= binding.capture == ModelCapturePolicy::RedactedRaw;
            default_model.get_or_insert_with(|| binding.model.clone());
        }
        let default_model = default_model.ok_or(ModelRuntimeError::MissingBindings)?;
        Ok(NativeGraphModelStageInputs {
            context,
            profiles: Arc::new(profiles),
            token_counters,
            transports,
            default_model,
            raw_enabled,
        })
    }
}

/// Endpoint, transport, tokenizer, and capture policy resolved for one model.
#[derive(Clone)]
pub struct ResolvedModelBinding {
    profile: ValidatedEndpointProfileV2,
    transport_id: String,
    tokenizer: ResolvedTokenizerBinding,
    model: String,
    model_headers: BTreeMap<String, SecretValue>,
    adapter_environment: BTreeMap<String, String>,
    generation: GenerationDefaults,
    capture: ModelCapturePolicy,
}

impl ResolvedModelBinding {
    /// Returns the current endpoint profile selected from the product registry.
    pub fn profile(&self) -> &ValidatedEndpointProfileV2 {
        &self.profile
    }

    /// Returns the registered transport identifier selected by this binding.
    pub fn transport_id(&self) -> &str {
        &self.transport_id
    }

    /// Returns the resolved tokenizer identity for diagnostics and construction.
    pub fn tokenizer_id(&self) -> &str {
        self.tokenizer.id()
    }

    /// Returns the provider model identifier retained by this binding.
    pub fn model(&self) -> &str {
        &self.model
    }

    /// Returns host-resolved headers reserved for model dispatch.
    pub fn model_headers(&self) -> &BTreeMap<String, SecretValue> {
        &self.model_headers
    }

    /// Returns the adapter environment. Model secrets are never inserted here.
    pub fn adapter_environment(&self) -> &BTreeMap<String, String> {
        &self.adapter_environment
    }

    /// Returns the immutable package generation settings retained for graph dispatch.
    pub fn generation(&self) -> &GenerationDefaults {
        &self.generation
    }

    /// Returns the package-pinned raw-exchange capture policy.
    pub const fn capture(&self) -> ModelCapturePolicy {
        self.capture
    }

    /// Returns the package-pinned pre-send connect retry count.
    pub const fn max_connect_retries(&self) -> u32 {
        self.profile.client.max_connect_retries
    }
}

/// Tokenizer policy retained after model runtime resolution.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ResolvedTokenizerBinding {
    /// A local tokenizer selected by its package-pinned name and revision.
    Local {
        /// Local tokenizer identity.
        name: String,
        /// Pinned tokenizer revision.
        revision: String,
        /// Whether endpoint chat templating applies before tokenization.
        apply_chat_template: bool,
    },
    /// A server tokenizer selected by its package-pinned URL.
    Server {
        /// Server tokenizer origin.
        url: String,
        /// Whether endpoint chat templating applies before tokenization.
        apply_chat_template: bool,
    },
}

impl ResolvedTokenizerBinding {
    fn from_spec(spec: &TokenizerBindingSpec) -> Self {
        match spec {
            TokenizerBindingSpec::Local {
                name,
                revision,
                apply_chat_template,
            } => Self::Local {
                name: name.clone(),
                revision: revision.clone(),
                apply_chat_template: *apply_chat_template,
            },
            TokenizerBindingSpec::Server {
                url,
                apply_chat_template,
            } => Self::Server {
                url: url.clone(),
                apply_chat_template: *apply_chat_template,
            },
        }
    }

    fn id(&self) -> &str {
        match self {
            Self::Local { name, .. } => name,
            Self::Server { url, .. } => url,
        }
    }

    fn input_token_counter(&self, model: &str) -> Result<Arc<dyn InputTokenCounter>, String> {
        let (tokenizer, apply_chat_template) = match self {
            Self::Local {
                name,
                apply_chat_template,
                ..
            } => {
                let name = (name == "tiktoken").then_some("builtin").unwrap_or(name);
                (
                    load_tokenizer(Some(name)).map_err(|error| error.to_string())?,
                    *apply_chat_template,
                )
            }
            Self::Server {
                url,
                apply_chat_template,
            } => (
                Arc::new(
                    crate::dataset::ServerTokenizer::new(url, Some(model.to_owned()))
                        .map_err(|error| error.to_string())?,
                ) as Arc<dyn crate::dataset::TextTokenizer>,
                *apply_chat_template,
            ),
        };
        Ok(Arc::new(EndpointInputTokenCounter::new(
            tokenizer,
            apply_chat_template,
        )))
    }
}

struct NativeGraphModelStageInputs {
    context: crate::engine::registry::RunContext,
    profiles: Arc<Vec<ValidatedEndpointProfileV2>>,
    token_counters: BTreeMap<String, Arc<dyn InputTokenCounter>>,
    transports: BTreeMap<String, Arc<dyn NativeTransportExecution>>,
    default_model: String,
    raw_enabled: bool,
}

/// Rust-owned callback that executes one immutable NativeGraph through AIPerf's
/// registered endpoint, transport, tokenizer, graph, and observer seams.
pub struct EngineNativeGraphEpisodeCallback {
    inputs: NativeGraphModelStageInputs,
    program: crate::graph::model::GraphTraceProgram,
    record_artifact: Option<Rc<RefCell<EvalNodeRecordArtifact>>>,
    evidence: Option<NativeGraphTransportEvidence>,
}

impl EngineNativeGraphEpisodeCallback {
    /// Resolves one imported package against an already frozen application.
    pub fn new(
        application: &Application,
        package: &NativeGraphPackagePlan,
        runtime: &ModelRuntimeConfig,
        secrets: &dyn SecretProvider,
    ) -> Result<Self, NativeGraphModelStageError> {
        let registry = application.product_registry();
        let source =
            package
                .program_source()
                .ok_or_else(|| NativeGraphModelStageError::Factory {
                    seam: "lowerer source",
                    reason: "imported NativeGraph package has no program source".to_owned(),
                })?;
        let lowerer = registry
            .native_graph_lowerer("native_graph")
            .ok_or_else(|| NativeGraphModelStageError::Factory {
                seam: "lowerer",
                reason: "no linked NativeGraph lowerer provider named \"native_graph\"".to_owned(),
            })?;
        let lowerer =
            lowerer
                .bind(package)
                .map_err(|error| NativeGraphModelStageError::Factory {
                    seam: "lowerer",
                    reason: error.to_string(),
                })?;
        let program = lowerer
            .lower(GraphLoweringRequest {
                source_schema: NATIVE_GRAPH_SOURCE_SCHEMA,
                execution_profile: NATIVE_GRAPH_EXECUTION_PROFILE,
                source: source.bytes(),
            })
            .map_err(|error| NativeGraphModelStageError::Lowering(error.to_string()))?;
        let (_, report) = lower_native_graph(package)
            .map_err(|error| NativeGraphModelStageError::Lowering(error.to_string()))?;
        let fidelity = registry
            .native_graph_fidelity_observer("exact")
            .ok_or_else(|| NativeGraphModelStageError::Factory {
                seam: "fidelity observer",
                reason: "no linked NativeGraph fidelity observer named \"exact\"".to_owned(),
            })?
            .create();
        fidelity
            .observe(&report)
            .map_err(|error| NativeGraphModelStageError::Factory {
                seam: "fidelity observer",
                reason: error.to_string(),
            })?;
        let resolver = CurrentNativeGraphModelBindingResolver::from_registry(registry.clone());
        let bindings = resolver
            .resolve(package.model_bindings(), runtime, secrets)
            .map_err(NativeGraphModelStageError::Resolution)?;
        let inputs = bindings
            .engine_inputs(application)
            .map_err(NativeGraphModelStageError::Resolution)?;
        Ok(Self {
            inputs,
            program,
            record_artifact: None,
            evidence: None,
        })
    }

    /// Attaches a suite-owned coordinator artifact for completed model-node records.
    pub(crate) fn with_record_artifact(
        mut self,
        artifact: Rc<RefCell<EvalNodeRecordArtifact>>,
    ) -> Self {
        self.record_artifact = Some(artifact);
        self
    }

    /// Returns the completed graph's observed transport facts.
    pub fn transport_evidence(&self) -> Option<ObservedNativeGraphTransportEvidence> {
        self.evidence
            .map(ObservedNativeGraphTransportEvidence::from)
    }
}

/// Public immutable summary of the request records emitted by `EngineGraphSink`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ObservedNativeGraphTransportEvidence {
    model_records: usize,
    completed_traces: usize,
}

impl ObservedNativeGraphTransportEvidence {
    /// Returns the number of complete native model records observed for the trace.
    pub const fn model_records(&self) -> usize {
        self.model_records
    }

    /// Returns the number of complete graph traces observed for this callback.
    pub const fn completed_traces(&self) -> usize {
        self.completed_traces
    }
}

impl From<NativeGraphTransportEvidence> for ObservedNativeGraphTransportEvidence {
    fn from(value: NativeGraphTransportEvidence) -> Self {
        Self {
            model_records: value.model_records,
            completed_traces: value.completed_traces,
        }
    }
}

#[async_trait(?Send)]
impl NativeGraphEpisodeCallback for EngineNativeGraphEpisodeCallback {
    async fn run(&mut self, _: &mut dyn NativeGraphEpisodeLease) -> Result<(), EvalExecutionError> {
        let evidence = execute_native_graph_trace(
            &self.inputs.context,
            self.inputs.profiles.clone(),
            self.inputs.token_counters.clone(),
            self.inputs.transports.clone(),
            self.inputs.default_model.clone(),
            self.inputs.raw_enabled,
            self.program.clone(),
            self.record_artifact.clone(),
        )
        .await
        .map_err(|error| EvalExecutionError::NativeGraphModel(error.to_string()))?;
        self.evidence = Some(evidence);
        Ok(())
    }
}

/// Failure while constructing or dispatching an EngineGraphSink NativeGraph stage.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NativeGraphModelStageError {
    /// A required NativeGraph-only factory was unavailable or rejected binding.
    Factory {
        /// The factory seam selected by the frozen product registry.
        seam: &'static str,
        /// Redacted binding or selection failure.
        reason: String,
    },
    /// Immutable binding resolution failed before graph dispatch.
    Resolution(ModelRuntimeError),
    /// Source-faithful graph lowering failed before graph dispatch.
    Lowering(String),
}

impl Display for NativeGraphModelStageError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Factory { seam, reason } => {
                write!(formatter, "native graph {seam} factory failed: {reason}")
            }
            Self::Resolution(error) => error.fmt(formatter),
            Self::Lowering(error) => write!(
                formatter,
                "native graph model-stage lowering failed: {error}"
            ),
        }
    }
}

impl std::error::Error for NativeGraphModelStageError {}

/// NativeGraph model-runtime resolution failure.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ModelRuntimeError {
    /// The host configuration version is not supported by this binary.
    UnsupportedVersion(u32),
    /// The package declared a duplicate binding identifier.
    DuplicateBinding(String),
    /// Endpoint lookup or preparation failed.
    Endpoint { binding: String, reason: String },
    /// A package transport identifier is not registered by this binary.
    UnknownTransport { binding: String, transport: String },
    /// Multiple bindings attempted to install one endpoint profile identity.
    DuplicateEndpointProfile(String),
    /// No model binding was available to supply the graph's fallback model value.
    MissingBindings,
    /// Constructing the frozen application context failed.
    Application(String),
    /// A binding's tokenizer could not be prepared by the current tokenizer seam.
    Tokenizer { binding: String, reason: String },
    /// A binding's transport configuration could not be prepared by its factory.
    Transport { binding: String, reason: String },
    /// A registered transport has no current graph-native execution binding.
    NoNativeTransport { binding: String, transport: String },
    /// A package URL could not be parsed at runtime.
    InvalidUrl {
        binding: String,
        url: String,
        reason: String,
    },
    /// A binding selected a URL scheme unavailable on its registered transport.
    UnsupportedTransportUrl {
        binding: String,
        transport: String,
        scheme: String,
    },
    /// A binding's finite timeout cannot be represented by the current client.
    InvalidTimeout(String),
    /// A package logical secret was absent from host-owned runtime configuration.
    MissingSecret { binding: String, secret: String },
    /// Host secret resolution failed without exposing the secret value.
    Secret {
        binding: String,
        secret: String,
        reason: String,
    },
    /// A binding declared more than one value for an HTTP header.
    DuplicateHeader { binding: String, header: String },
}

impl Display for ModelRuntimeError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedVersion(version) => {
                write!(
                    formatter,
                    "unsupported NativeGraph model runtime version {version}"
                )
            }
            Self::DuplicateBinding(binding) => {
                write!(formatter, "duplicate NativeGraph model binding {binding:?}")
            }
            Self::Endpoint { binding, reason } => {
                write!(
                    formatter,
                    "model binding {binding:?} endpoint resolution failed: {reason}"
                )
            }
            Self::UnknownTransport { binding, transport } => write!(
                formatter,
                "model binding {binding:?} selected unavailable transport {transport:?}"
            ),
            Self::DuplicateEndpointProfile(profile) => write!(
                formatter,
                "NativeGraph bindings selected duplicate endpoint profile {profile:?}"
            ),
            Self::MissingBindings => {
                formatter.write_str("NativeGraph package declares no model bindings")
            }
            Self::Application(reason) => {
                write!(
                    formatter,
                    "NativeGraph application composition failed: {reason}"
                )
            }
            Self::Tokenizer { binding, reason } => write!(
                formatter,
                "model binding {binding:?} tokenizer preparation failed: {reason}"
            ),
            Self::Transport { binding, reason } => write!(
                formatter,
                "model binding {binding:?} transport preparation failed: {reason}"
            ),
            Self::NoNativeTransport { binding, transport } => write!(
                formatter,
                "model binding {binding:?} transport {transport:?} has no native graph execution binding"
            ),
            Self::InvalidUrl {
                binding,
                url,
                reason,
            } => write!(
                formatter,
                "model binding {binding:?} has invalid URL {url:?}: {reason}"
            ),
            Self::UnsupportedTransportUrl {
                binding,
                transport,
                scheme,
            } => write!(
                formatter,
                "model binding {binding:?} transport {transport:?} does not support {scheme:?} URLs"
            ),
            Self::InvalidTimeout(binding) => write!(
                formatter,
                "model binding {binding:?} request timeout exceeds the native client limit"
            ),
            Self::MissingSecret { binding, secret } => write!(
                formatter,
                "model binding {binding:?} requires host mapping for logical secret {secret:?}"
            ),
            Self::Secret {
                binding,
                secret,
                reason,
            } => write!(
                formatter,
                "model binding {binding:?} could not resolve logical secret {secret:?}: {reason}"
            ),
            Self::DuplicateHeader { binding, header } => write!(
                formatter,
                "model binding {binding:?} declared duplicate model header {header:?}"
            ),
        }
    }
}

impl std::error::Error for ModelRuntimeError {}
