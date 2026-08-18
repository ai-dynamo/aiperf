// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Resolution of immutable NativeGraph model bindings through product seams.

use std::{
    cell::RefCell,
    collections::BTreeMap,
    fmt::{self, Display, Formatter},
    io::Read,
    rc::Rc,
    sync::Arc,
};

use async_trait::async_trait;
use bytes::Bytes;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, value::RawValue};
use url::Url;

use crate::{
    endpoints::{RawEndpointConfig, Turn},
    engine::registry::ValidatedEndpointProfileV2,
    engine::{
        application::Application,
        execute::load_tokenizer,
        graph_execution::{
            NativeGraphLivePolicyCallSummary, NativeGraphPolicyEndpointRuntime,
            NativeGraphTransportEvidence, execute_native_graph_trace,
        },
        record_lane::EvalNodeRecordArtifact,
        registry::{NativeTransportExecution, WorkloadRequirements},
    },
    eval::GraphLoweringRequest,
    eval::{
        EnvName, EvalExecutionError, NativeGraphEpisodeCallback, NativeGraphEpisodeLease,
        NativeGraphLeaseRolloutStart, SecretProvider, SecretValue,
    },
    extensions::AIPerfRegistry,
    graph::agent::{
        AgentInvocationLease, AgentLoopError, AgentResponseStore, AgentTrajectory,
        AgentTrajectorySink, AgentTurnCoordinator, InvocationLeaseFactory,
        LiveAgentPolicyDecisionCollector, LiveAgentPolicyDecisionReader,
        LiveAgentPolicyDecisionRequest,
    },
    multiturn::{EndpointInputTokenCounter, InputTokenCounter},
    transport::{core::ConnectionReuseStrategy, http::config::ClientConfig},
};

use super::package::NativeGraphRolloutPlan;
use super::{
    ActionEncoderFactoryId, ActionEncodingLimits, ArtifactError,
    BoundNativeGraphEnvironmentStepper, DeclaredPolicyDecision, EpisodeActionEncodingError,
    EpisodeArtifactStore, FrozenArtifact, FrozenRolloutEvidence, GenerationDefaults,
    ModelBindingId, ModelBindingSpec, ModelCapturePolicy, ModelSecretId,
    NativeGraphAttemptAuthority, NativeGraphPackagePlan, TokenizerBindingSpec, lower_native_graph,
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

    /// Prepares the exact package-selected model binding for one bounded live policy decision.
    ///
    /// The reader is assembled through the application's frozen endpoint and transport seams;
    /// this method never constructs an evaluation-local HTTP or gRPC client. Only the selected
    /// binding is prepared, so an unrelated declared binding cannot become an implicit fallback.
    pub fn engine_selected_policy_runtime(
        &self,
        application: &Application,
        selected: &ModelBindingId,
    ) -> Result<Box<dyn NativeGraphPolicyModelRuntime>, ModelRuntimeError> {
        self.engine_selected_policy_runtime_with_summary(
            application,
            selected,
            Rc::new(RefCell::new(NativeGraphLivePolicyCallSummary::new())),
        )
    }

    fn engine_selected_policy_runtime_with_summary(
        &self,
        application: &Application,
        selected: &ModelBindingId,
        live_policy_summary: Rc<RefCell<NativeGraphLivePolicyCallSummary>>,
    ) -> Result<Box<dyn NativeGraphPolicyModelRuntime>, ModelRuntimeError> {
        let binding = self.bindings.get(selected).ok_or_else(|| {
            ModelRuntimeError::SelectedBindingMissing(selected.as_str().to_owned())
        })?;
        let context = application
            .native_graph_context(vec![binding.profile.clone()])
            .map_err(|error| ModelRuntimeError::Application(error.to_string()))?;
        let input_token_counter = binding
            .tokenizer
            .input_token_counter(&binding.model)
            .map_err(|reason| ModelRuntimeError::Tokenizer {
                binding: selected.as_str().to_owned(),
                reason,
            })?;
        let factory = context
            .product_registry()
            .transport_factory(&binding.transport_id)
            .ok_or_else(|| ModelRuntimeError::UnknownTransport {
                binding: selected.as_str().to_owned(),
                transport: binding.transport_id.clone(),
            })?;
        let config = RawValue::from_string("{}".to_owned()).map_err(|error| {
            ModelRuntimeError::Transport {
                binding: selected.as_str().to_owned(),
                reason: error.to_string(),
            }
        })?;
        let config = factory
            .validate(&config, &WorkloadRequirements::default())
            .map_err(|error| ModelRuntimeError::Transport {
                binding: selected.as_str().to_owned(),
                reason: error.to_string(),
            })?;
        let transport = factory
            .native_execution(config.as_ref(), &context)
            .map_err(|error| ModelRuntimeError::Transport {
                binding: selected.as_str().to_owned(),
                reason: error.to_string(),
            })?
            .ok_or_else(|| ModelRuntimeError::NoNativeTransport {
                binding: selected.as_str().to_owned(),
                transport: binding.transport_id.clone(),
            })?;
        let endpoint = NativeGraphPolicyEndpointRuntime::new(
            context.product_registry().endpoints().clone(),
            binding.profile.clone(),
            input_token_counter,
            transport,
            binding.model.clone(),
            binding.capture == ModelCapturePolicy::RedactedRaw,
            live_policy_summary,
        )
        .map_err(|error| ModelRuntimeError::Application(error.to_string()))?;
        Ok(Box::new(EngineSelectedNativeGraphPolicyModelRuntime {
            binding: selected.clone(),
            generation: binding.generation.clone(),
            endpoint,
        }))
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

/// One model-runtime failure while collecting a raw live-rollout policy decision.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NativeGraphModelDecisionError(String);

impl NativeGraphModelDecisionError {
    /// Creates a redacted model-decision runtime failure.
    pub fn new(reason: impl Into<String>) -> Self {
        Self(reason.into())
    }
}

impl Display for NativeGraphModelDecisionError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for NativeGraphModelDecisionError {}

/// Existing model-runtime boundary for one bounded policy decision.
///
/// The runtime receives no collector and cannot return an owned decision frame. It instead opens
/// a reader that the host drives through buffers capped by the imported policy limit. The rollout
/// coordinator owns lexical JSON admission and action encoding after this seam.
#[async_trait(?Send)]
pub trait NativeGraphPolicyModelRuntime {
    /// Returns the exact imported model binding resolved by this runtime.
    fn binding(&self) -> &ModelBindingId;

    /// Opens one package-bound prompt and frozen-observation decision reader.
    async fn open_decision(
        &mut self,
        request: &LiveAgentPolicyDecisionRequest,
    ) -> Result<Box<dyn LiveAgentPolicyDecisionReader>, NativeGraphModelDecisionError>;
}

/// Engine-backed runtime for one immutable selected NativeGraph model binding.
///
/// The endpoint runtime is prepared from the application's already frozen graph profile path and
/// exposes only a [`LiveAgentPolicyDecisionReader`]. It deliberately owns no graph evidence or
/// response-capture sink.
struct EngineSelectedNativeGraphPolicyModelRuntime {
    binding: ModelBindingId,
    generation: GenerationDefaults,
    endpoint: NativeGraphPolicyEndpointRuntime,
}

#[async_trait(?Send)]
impl NativeGraphPolicyModelRuntime for EngineSelectedNativeGraphPolicyModelRuntime {
    fn binding(&self) -> &ModelBindingId {
        &self.binding
    }

    async fn open_decision(
        &mut self,
        request: &LiveAgentPolicyDecisionRequest,
    ) -> Result<Box<dyn LiveAgentPolicyDecisionReader>, NativeGraphModelDecisionError> {
        let turn = policy_decision_turn(request, &self.generation)?;
        let max_output_tokens = self
            .generation
            .max_tokens
            .map(usize::try_from)
            .transpose()
            .map_err(|_| NativeGraphModelDecisionError::new("policy max_tokens exceeds usize"))?
            .unwrap_or(256);
        let reader = self
            .endpoint
            .open_bounded_decision(turn, max_output_tokens, request.max_decision_bytes())
            .await
            .map_err(|error| NativeGraphModelDecisionError::new(error.to_string()))?;
        Ok(Box::new(EngineNativeGraphPolicyDecisionReader { reader }))
    }
}

/// Reader wrapper that prevents the engine transport from returning an owned policy frame.
struct EngineNativeGraphPolicyDecisionReader {
    reader: crate::transport::core::BoundedDecisionReader,
}

#[async_trait(?Send)]
impl LiveAgentPolicyDecisionReader for EngineNativeGraphPolicyDecisionReader {
    async fn read(&mut self, destination: &mut [u8]) -> Result<usize, AgentLoopError> {
        self.reader.read(destination).map_err(|error| {
            AgentLoopError::new(format!("reading bounded policy decision: {error}"))
        })
    }
}

#[derive(Serialize)]
struct PolicyDecisionMessage<'a> {
    role: &'static str,
    content: &'a str,
}

fn policy_decision_turn(
    request: &LiveAgentPolicyDecisionRequest,
    generation: &GenerationDefaults,
) -> Result<Turn, NativeGraphModelDecisionError> {
    let prompt = std::str::from_utf8(request.prompt())
        .map_err(|_| NativeGraphModelDecisionError::new("imported policy prompt is not UTF-8"))?;
    let observation = std::str::from_utf8(request.observation()).map_err(|_| {
        NativeGraphModelDecisionError::new("frozen environment observation is not UTF-8")
    })?;
    let messages = [
        policy_decision_message("system", prompt)?,
        policy_decision_message("user", observation)?,
    ];
    let extra_body = generation_body(generation)?;
    Ok(Turn {
        max_tokens: generation.max_tokens,
        lowered: Some(messages.into_iter().collect()),
        extra_body: (!extra_body.is_empty()).then_some(extra_body),
        ..Turn::default()
    })
}

fn policy_decision_message(
    role: &'static str,
    content: &str,
) -> Result<Bytes, NativeGraphModelDecisionError> {
    serde_json::to_vec(&PolicyDecisionMessage { role, content })
        .map(Bytes::from)
        .map_err(|_| NativeGraphModelDecisionError::new("serializing sealed policy prompt facts"))
}

fn generation_body(
    generation: &GenerationDefaults,
) -> Result<Map<String, Value>, NativeGraphModelDecisionError> {
    let mut body = Map::new();
    if let Some(value) = generation.min_tokens {
        body.insert("min_tokens".to_owned(), Value::from(value));
    }
    if let Some(value) = generation.temperature {
        insert_finite_generation(&mut body, "temperature", value)?;
    }
    if let Some(value) = generation.top_p {
        insert_finite_generation(&mut body, "top_p", value)?;
    }
    if let Some(value) = generation.top_k {
        body.insert("top_k".to_owned(), Value::from(value));
    }
    if let Some(value) = generation.seed {
        body.insert("seed".to_owned(), Value::from(value));
    }
    if let Some(value) = generation.presence_penalty {
        insert_finite_generation(&mut body, "presence_penalty", value)?;
    }
    if let Some(value) = generation.frequency_penalty {
        insert_finite_generation(&mut body, "frequency_penalty", value)?;
    }
    if let Some(value) = generation.repetition_penalty {
        insert_finite_generation(&mut body, "repetition_penalty", value)?;
    }
    Ok(body)
}

fn insert_finite_generation(
    body: &mut Map<String, Value>,
    field: &'static str,
    value: f64,
) -> Result<(), NativeGraphModelDecisionError> {
    let value = serde_json::Number::from_f64(value).ok_or_else(|| {
        NativeGraphModelDecisionError::new("imported policy generation value is not finite")
    })?;
    body.insert(field.to_owned(), Value::Number(value));
    Ok(())
}

/// Opaque authority binding prepared live-rollout facts to one immutable environment binding.
#[derive(Clone)]
pub(crate) struct NativeGraphLiveRolloutBindingAuthority(Rc<()>);

impl NativeGraphLiveRolloutBindingAuthority {
    pub(crate) fn new() -> Self {
        Self(Rc::new(()))
    }

    fn matches(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.0, &other.0)
    }
}

/// Opaque authority minted only after one worker-local environment stepper starts.
#[derive(Clone)]
pub(crate) struct NativeGraphLiveRolloutSessionAuthority(Rc<()>);

impl NativeGraphLiveRolloutSessionAuthority {
    pub(crate) fn new() -> Self {
        Self(Rc::new(()))
    }

    fn matches(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.0, &other.0)
    }
}

/// A policy decision issued by the exact worker-local rollout coordinator.
///
/// The public type intentionally has no public constructor: a started environment stepper accepts
/// only decisions stamped by the coordinator bound from the same imported rollout session.
pub struct IssuedNativeGraphPolicyDecision {
    decision: DeclaredPolicyDecision,
    authority: NativeGraphLiveRolloutSessionAuthority,
}

impl fmt::Debug for IssuedNativeGraphPolicyDecision {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("IssuedNativeGraphPolicyDecision")
            .field("encoder", self.decision.encoder())
            .finish_non_exhaustive()
    }
}

impl IssuedNativeGraphPolicyDecision {
    pub(crate) fn new(
        decision: DeclaredPolicyDecision,
        authority: NativeGraphLiveRolloutSessionAuthority,
    ) -> Self {
        Self {
            decision,
            authority,
        }
    }

    pub(crate) fn belongs_to(&self, authority: &NativeGraphLiveRolloutSessionAuthority) -> bool {
        self.authority.matches(authority)
    }

    pub(crate) fn into_decision(self) -> DeclaredPolicyDecision {
        self.decision
    }
}

/// Prepared immutable live-rollout facts waiting for exactly one started worker session.
///
/// This object owns the selected runtime and has no session authority. It can therefore never
/// issue a decision until a matching started stepper consumes it.
pub struct PreparedNativeGraphLiveRolloutCoordinator {
    binding: NativeGraphLiveRolloutBindingAuthority,
    model_binding: ModelBindingId,
    prompt: super::NativeGraphRolloutPolicyPromptSource,
    action_encoder_id: ActionEncoderFactoryId,
    limits: ActionEncodingLimits,
    max_prompt_bytes: usize,
    runtime: Box<dyn NativeGraphPolicyModelRuntime>,
}

impl PreparedNativeGraphLiveRolloutCoordinator {
    pub(crate) fn into_coordinator(
        self,
        expected_binding: &NativeGraphLiveRolloutBindingAuthority,
        authority: NativeGraphLiveRolloutSessionAuthority,
    ) -> Result<NativeGraphLiveRolloutCoordinator, NativeGraphLiveRolloutError> {
        if !self.binding.matches(expected_binding) {
            return Err(NativeGraphLiveRolloutError::SessionBindingMismatch);
        }
        Ok(NativeGraphLiveRolloutCoordinator {
            model_binding: self.model_binding,
            prompt: self.prompt,
            action_encoder_id: self.action_encoder_id,
            limits: self.limits,
            max_prompt_bytes: self.max_prompt_bytes,
            authority,
            runtime: self.runtime,
        })
    }
}

/// Package-bound coordinator for one live rollout policy decision.
///
/// The coordinator is deliberately not a second executor: it extends the
/// existing [`AgentTurnCoordinator`] policy-decision path, while the existing
/// environment stepper retains all action-dispatch and child-lifecycle control.
pub struct NativeGraphLiveRolloutCoordinator {
    model_binding: ModelBindingId,
    prompt: super::NativeGraphRolloutPolicyPromptSource,
    action_encoder_id: ActionEncoderFactoryId,
    limits: ActionEncodingLimits,
    max_prompt_bytes: usize,
    authority: NativeGraphLiveRolloutSessionAuthority,
    runtime: Box<dyn NativeGraphPolicyModelRuntime>,
}

impl NativeGraphLiveRolloutCoordinator {
    /// Prepares immutable rollout selectors before any environment adapter is provisioned.
    pub(crate) fn prepare(
        rollout: &NativeGraphRolloutPlan,
        binding: NativeGraphLiveRolloutBindingAuthority,
        runtime: Box<dyn NativeGraphPolicyModelRuntime>,
    ) -> Result<PreparedNativeGraphLiveRolloutCoordinator, NativeGraphLiveRolloutError> {
        let policy = rollout.policy();
        if runtime.binding() != policy.model_binding_id() {
            return Err(NativeGraphLiveRolloutError::MissingModelBinding {
                binding: policy.model_binding_id().clone(),
            });
        }
        let max_decision_bytes = usize::try_from(policy.max_decision_bytes()).map_err(|_| {
            NativeGraphLiveRolloutError::DecisionLimitOutOfRange {
                field: "rollout.policy.max_decision_bytes",
            }
        })?;
        let max_action_bytes = usize::try_from(
            rollout.environment().artifact_limits().max_artifact_bytes(),
        )
        .map_err(|_| NativeGraphLiveRolloutError::DecisionLimitOutOfRange {
            field: "rollout.artifacts.max_artifact_bytes",
        })?;
        let limits = ActionEncodingLimits::new(max_decision_bytes, max_action_bytes)
            .map_err(NativeGraphLiveRolloutError::ActionEncoding)?;
        let max_prompt_bytes =
            usize::try_from(rollout.limits().max_prompt_bytes()).map_err(|_| {
                NativeGraphLiveRolloutError::DecisionLimitOutOfRange {
                    field: "rollout.limits.max_prompt_bytes",
                }
            })?;
        if policy.prompt_source().bytes().len() > max_prompt_bytes {
            return Err(NativeGraphLiveRolloutError::PromptLimitExceeded);
        }
        Ok(PreparedNativeGraphLiveRolloutCoordinator {
            model_binding: policy.model_binding_id().clone(),
            prompt: policy.prompt_source().clone(),
            action_encoder_id: rollout.environment().action_encoder_id().clone(),
            limits,
            max_prompt_bytes,
            binding,
            runtime,
        })
    }

    /// Dispatches one bounded model decision and returns only its sealed policy decision.
    ///
    /// The worker-local package-selected environment stepper owns the encoder authority that may
    /// turn this decision into a single-use action capability.
    pub async fn decide_policy_decision(
        &mut self,
        observation: &FrozenArtifact,
        store: &mut EpisodeArtifactStore,
    ) -> Result<IssuedNativeGraphPolicyDecision, NativeGraphLiveRolloutError> {
        let observation = store
            .read_frozen(observation)
            .map(Bytes::from)
            .map_err(NativeGraphLiveRolloutError::Artifact)?;
        self.decide_policy_decision_bytes(observation).await
    }

    /// Converts trusted, already-bounded frozen observation bytes into one sealed decision.
    ///
    /// The started environment stepper reads its own artifact store before this async call so a
    /// worker-local `RefCell` borrow cannot outlive a model-runtime await point.
    pub(crate) async fn decide_policy_decision_bytes(
        &mut self,
        observation: Bytes,
    ) -> Result<IssuedNativeGraphPolicyDecision, NativeGraphLiveRolloutError> {
        if self.prompt.bytes().len() > self.max_prompt_bytes {
            return Err(NativeGraphLiveRolloutError::PromptLimitExceeded);
        }
        let request = LiveAgentPolicyDecisionRequest::new(
            Bytes::copy_from_slice(self.prompt.bytes()),
            observation,
            self.limits.max_decision_bytes(),
        )
        .map_err(NativeGraphLiveRolloutError::Coordinator)?;
        let mut collector = LiveAgentPolicyDecisionCollector::new(self.limits.max_decision_bytes())
            .map_err(NativeGraphLiveRolloutError::Coordinator)?;
        let mut reader = AgentTurnCoordinator::next_live_policy_decision(self, &request)
            .await
            .map_err(NativeGraphLiveRolloutError::Coordinator)?;
        collector
            .collect_from(reader.as_mut())
            .await
            .map_err(NativeGraphLiveRolloutError::Coordinator)?;
        let decision = DeclaredPolicyDecision::from_json_bytes(
            self.action_encoder_id.clone(),
            &collector.into_bytes(),
            self.limits,
        )
        .map_err(NativeGraphLiveRolloutError::ActionEncoding)?;
        Ok(IssuedNativeGraphPolicyDecision::new(
            decision,
            self.authority.clone(),
        ))
    }
}

#[async_trait(?Send)]
impl AgentTurnCoordinator for NativeGraphLiveRolloutCoordinator {
    async fn next_live_policy_decision(
        &mut self,
        request: &LiveAgentPolicyDecisionRequest,
    ) -> Result<Box<dyn LiveAgentPolicyDecisionReader>, AgentLoopError> {
        if self.runtime.binding() != &self.model_binding {
            return Err(AgentLoopError::new(
                "live rollout model runtime changed its selected binding",
            ));
        }
        if request.prompt() != self.prompt.bytes() {
            return Err(AgentLoopError::new(
                "live rollout policy request did not retain the imported prompt snapshot",
            ));
        }
        if request.max_decision_bytes() != self.limits.max_decision_bytes() {
            return Err(AgentLoopError::new(
                "live rollout policy request did not retain the selected decision bound",
            ));
        }
        self.runtime
            .open_decision(request)
            .await
            .map_err(|error| AgentLoopError::new(error.to_string()))
    }

    async fn run(
        &mut self,
        _: &mut dyn AgentResponseStore,
        _: &mut dyn AgentTrajectorySink,
        _: &dyn InvocationLeaseFactory,
        _: &dyn AgentInvocationLease,
        _: &dyn crate::graph::tools::AgentToolCallDecoder,
        _: &dyn crate::graph::tools::AgentObservationFormatter,
    ) -> Result<AgentTrajectory, AgentLoopError> {
        Err(AgentLoopError::new(
            "NativeGraph live rollout coordinator does not execute recorded trajectories",
        ))
    }
}

/// Typed live-rollout decision failure without raw prompt, observation, or model bytes.
#[derive(Debug)]
pub enum NativeGraphLiveRolloutError {
    /// The selected model binding was not resolved before environment provisioning.
    MissingModelBinding {
        /// The package-selected binding identifier.
        binding: ModelBindingId,
    },
    /// A package-selected byte limit could not be represented on this platform.
    DecisionLimitOutOfRange {
        /// The selected package field.
        field: &'static str,
    },
    /// The existing agent coordinator rejected a sealed model-decision exchange.
    Coordinator(AgentLoopError),
    /// The imported prompt source exceeded its selected immutable cap.
    PromptLimitExceeded,
    /// Prepared live-rollout facts belonged to a different immutable environment binding.
    SessionBindingMismatch,
    /// The Rust-owned artifact store could not read or freeze a fact.
    Artifact(ArtifactError),
    /// The selected action encoder refused the raw model decision.
    ActionEncoding(EpisodeActionEncodingError),
}

impl Display for NativeGraphLiveRolloutError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingModelBinding { binding } => write!(
                formatter,
                "NativeGraph live rollout selected unavailable model binding {binding:?}"
            ),
            Self::DecisionLimitOutOfRange { field } => write!(
                formatter,
                "NativeGraph live rollout limit {field} does not fit this platform"
            ),
            Self::Coordinator(error) => Display::fmt(error, formatter),
            Self::PromptLimitExceeded => formatter.write_str(
                "NativeGraph live rollout prompt exceeds the selected imported byte limit",
            ),
            Self::SessionBindingMismatch => formatter.write_str(
                "NativeGraph live rollout coordinator belongs to another environment binding",
            ),
            Self::Artifact(error) => Display::fmt(error, formatter),
            Self::ActionEncoding(error) => Display::fmt(error, formatter),
        }
    }
}

impl std::error::Error for NativeGraphLiveRolloutError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Coordinator(error) => Some(error),
            Self::Artifact(error) => Some(error),
            Self::ActionEncoding(error) => Some(error),
            Self::MissingModelBinding { .. }
            | Self::DecisionLimitOutOfRange { .. }
            | Self::PromptLimitExceeded
            | Self::SessionBindingMismatch => None,
        }
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
    record_artifact: Option<EvalNodeRecordArtifact>,
    evidence: Option<NativeGraphTransportEvidence>,
    live_policy_runtime: Option<Box<dyn NativeGraphPolicyModelRuntime>>,
    live_policy_summary: Option<Rc<RefCell<NativeGraphLivePolicyCallSummary>>>,
    lease_rollout_start: Option<NativeGraphLeaseRolloutStart>,
    needs_live_rollout: bool,
    rollout_evidence: Option<FrozenRolloutEvidence>,
}

impl EngineNativeGraphEpisodeCallback {
    /// Resolves one imported package against an already frozen application.
    pub fn new(
        application: &Application,
        package: &NativeGraphPackagePlan,
        runtime: &ModelRuntimeConfig,
        secrets: &dyn SecretProvider,
        record_artifact: Option<EvalNodeRecordArtifact>,
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
        let (live_policy_runtime, live_policy_summary) = match package.rollout() {
            Some(rollout) => {
                let summary = Rc::new(RefCell::new(NativeGraphLivePolicyCallSummary::new()));
                let runtime = bindings
                    .engine_selected_policy_runtime_with_summary(
                        application,
                        rollout.policy().model_binding_id(),
                        Rc::clone(&summary),
                    )
                    .map_err(NativeGraphModelStageError::Resolution)?;
                (Some(runtime), Some(summary))
            }
            None => (None, None),
        };
        let inputs = bindings
            .engine_inputs(application)
            .map_err(NativeGraphModelStageError::Resolution)?;
        Ok(Self {
            inputs,
            program,
            record_artifact,
            evidence: None,
            live_policy_runtime,
            live_policy_summary,
            lease_rollout_start: None,
            needs_live_rollout: false,
            rollout_evidence: None,
        })
    }

    /// Binds one package-selected supervised rollout before Docker can provision its adapter.
    pub fn bind_live_rollout(
        &mut self,
        stepper: BoundNativeGraphEnvironmentStepper,
        authority: NativeGraphAttemptAuthority,
    ) -> Result<(), NativeGraphModelStageError> {
        if self.lease_rollout_start.is_some() || self.needs_live_rollout {
            return Err(NativeGraphModelStageError::Factory {
                seam: "live rollout",
                reason: "NativeGraph callback already has a bound live rollout".to_owned(),
            });
        }
        let runtime =
            self.live_policy_runtime
                .take()
                .ok_or_else(|| NativeGraphModelStageError::Factory {
                    seam: "live rollout",
                    reason: "NativeGraph package has no selected live rollout model runtime"
                        .to_owned(),
                })?;
        let summary =
            self.live_policy_summary
                .take()
                .ok_or_else(|| NativeGraphModelStageError::Factory {
                    seam: "live rollout",
                    reason: "NativeGraph package has no selected live rollout policy summary"
                        .to_owned(),
                })?;
        let coordinator = stepper
            .prepare_live_rollout_coordinator(runtime)
            .map_err(|error| NativeGraphModelStageError::Factory {
                seam: "live rollout",
                reason: error.to_string(),
            })?;
        self.lease_rollout_start = Some(NativeGraphLeaseRolloutStart::new(
            stepper,
            coordinator,
            authority,
            summary,
        ));
        self.needs_live_rollout = true;
        Ok(())
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
    live_policy_calls: u64,
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

    /// Returns the number of non-raw selected-policy calls used by a live rollout.
    pub const fn live_policy_calls(&self) -> u64 {
        self.live_policy_calls
    }
}

impl From<NativeGraphTransportEvidence> for ObservedNativeGraphTransportEvidence {
    fn from(value: NativeGraphTransportEvidence) -> Self {
        Self {
            model_records: value.model_records,
            completed_traces: value.completed_traces,
            live_policy_calls: value.live_policy_calls,
        }
    }
}

#[async_trait(?Send)]
impl NativeGraphEpisodeCallback for EngineNativeGraphEpisodeCallback {
    fn take_lease_rollout_start(&mut self) -> Option<NativeGraphLeaseRolloutStart> {
        self.lease_rollout_start.take()
    }

    async fn run(
        &mut self,
        lease: &mut dyn NativeGraphEpisodeLease,
    ) -> Result<(), EvalExecutionError> {
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
        if self.needs_live_rollout {
            let operation = lease.environment_adapter_start()?;
            operation.start_rollout().await?;
            let session = operation.rollout_session()?;
            let mut observation = session.reset().await?;
            loop {
                let transition = session.step(&observation).await?;
                if transition.is_terminal() {
                    break;
                }
                observation = transition.transition().observation().clone();
            }
            let rollout = session.freeze()?;
            let policy_calls = rollout
                .live_policy_calls()
                .map_or(0, |evidence| evidence.call_count());
            let evidence = self
                .evidence
                .as_mut()
                .ok_or(EvalExecutionError::InvalidRecipe(
                    "NativeGraph transport evidence",
                ))?;
            evidence.live_policy_calls = policy_calls;
            self.rollout_evidence = Some(rollout);
        }
        Ok(())
    }

    fn take_rollout_evidence(&mut self) -> Option<FrozenRolloutEvidence> {
        self.rollout_evidence.take()
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
    /// The live rollout selected a binding absent from the immutable resolved package set.
    SelectedBindingMissing(String),
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
            Self::SelectedBindingMissing(binding) => write!(
                formatter,
                "NativeGraph selected model binding {binding:?} is unavailable after immutable resolution"
            ),
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
