// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Frozen endpoint composition and worker-local preparation.
//!
//! The registry is mutable only through [`EndpointRegistryBuilder`] during
//! process composition. Runtime workers resolve authored IDs before scheduling,
//! prepare one binding per profile, and dispatch through an [`EndpointKey`]
//! indexed [`PreparedEndpointTable`]. This keeps catalog discovery and string
//! lookup out of the request path while leaving endpoint construction open to
//! statically linked factories.

use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt::{self, Display};
use std::str::FromStr;
use std::sync::Arc;

use serde::{Deserialize, Deserializer, Serialize};
use serde_json::Value;

use crate::body_plan::BodyPlan;
use crate::endpoints::DynosimEndpointFactory;
use crate::endpoints::MessagesEndpoint;
use crate::endpoints::VllmGenerateFactory;
use crate::endpoints::config::{EffectiveEndpointConfig, EndpointConfig, RawEndpointConfig};
use crate::endpoints::endpoints::{
    ChatEmbeddingsEndpoint, ChatEndpoint, CompletionsEndpoint, EmbeddingsEndpoint, Endpoint,
    ResponsesEndpoint,
};
use crate::endpoints::kserve::{
    KServeChatFactory, KServeCompletionsFactory, KServeEmbeddingsFactory, KServeV1PredictFactory,
    KServeV2EmbeddingsFactory, KServeV2ImagesFactory, KServeV2InferFactory,
    KServeV2RankingsFactory, KServeV2VlmFactory,
};
use crate::endpoints::metadata::{EndpointDescriptor, EndpointType};
use crate::endpoints::models::{
    CreditPhase, EndpointError, EndpointResult, ExtractedPayload, ParsedResponse, RequestInfo,
    RequestRecord, ServerResponse, Turn,
};
use crate::endpoints::riva::{
    RivaAnalyzeEntitiesFactory, RivaAnalyzeIntentFactory, RivaAsrFactory, RivaNaturalQueryFactory,
    RivaPunctuateTextFactory, RivaTextClassifyFactory, RivaTokenClassifyFactory,
    RivaTransformTextFactory, RivaTtsFactory,
};
use crate::endpoints::tier2::{
    CohereRankingsEndpoint, HfTeiRankingsEndpoint, HuggingFaceGenerateEndpoint, ImageEditEndpoint,
    ImageGenerationEndpoint, ImageRetrievalEndpoint, NimEmbeddingsEndpoint, NimRankingsEndpoint,
    RawEndpointFactory, SolidoRagEndpoint, TemplateEndpointFactory, VideoGenerationEndpoint,
};

pub(crate) fn legacy_descriptor_for(endpoint_type: EndpointType) -> &'static EndpointDescriptor {
    match endpoint_type {
        EndpointType::Chat => ChatEndpoint.descriptor(),
        EndpointType::Completions => CompletionsEndpoint.descriptor(),
        EndpointType::Responses => ResponsesEndpoint.descriptor(),
        EndpointType::Messages => MessagesEndpoint.descriptor(),
        EndpointType::Embeddings => EmbeddingsEndpoint.descriptor(),
        EndpointType::ChatEmbeddings => ChatEmbeddingsEndpoint.descriptor(),
        EndpointType::NimEmbeddings => NimEmbeddingsEndpoint.descriptor(),
        EndpointType::CohereRankings => CohereRankingsEndpoint.descriptor(),
        EndpointType::HfTeiRankings => HfTeiRankingsEndpoint.descriptor(),
        EndpointType::NimRankings => NimRankingsEndpoint.descriptor(),
        EndpointType::HuggingfaceGenerate => HuggingFaceGenerateEndpoint.descriptor(),
        EndpointType::ImageGeneration => ImageGenerationEndpoint.descriptor(),
        EndpointType::ImageEdit => ImageEditEndpoint.descriptor(),
        EndpointType::VideoGeneration => VideoGenerationEndpoint.descriptor(),
        EndpointType::ImageRetrieval => ImageRetrievalEndpoint.descriptor(),
        EndpointType::SolidoRag => SolidoRagEndpoint.descriptor(),
        EndpointType::Raw => crate::endpoints::RawEndpoint.descriptor(),
        EndpointType::Template => crate::endpoints::TemplateEndpoint.descriptor(),
    }
}

/// Open canonical endpoint identifier.
///
/// Construction validates only the stable `[a-z][a-z0-9_]*` grammar. Registry
/// membership is deliberately checked later so a syntactically valid ID from a
/// custom runner distribution survives wire deserialization.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(transparent)]
pub struct EndpointId(Box<str>);

impl EndpointId {
    /// Validate and construct an endpoint identifier.
    pub fn new(value: impl AsRef<str>) -> Result<Self, EndpointIdError> {
        let value = value.as_ref();
        let mut chars = value.chars();
        if !chars.next().is_some_and(|first| first.is_ascii_lowercase())
            || !chars.all(|character| {
                character.is_ascii_lowercase() || character.is_ascii_digit() || character == '_'
            })
        {
            return Err(EndpointIdError {
                value: value.to_string(),
            });
        }
        Ok(Self(value.into()))
    }

    /// Borrow the canonical identifier.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Display for EndpointId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

impl FromStr for EndpointId {
    type Err = EndpointIdError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        Self::new(value)
    }
}

impl TryFrom<&str> for EndpointId {
    type Error = EndpointIdError;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}

impl<'de> Deserialize<'de> for EndpointId {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Self::new(&value).map_err(serde::de::Error::custom)
    }
}

/// Invalid open endpoint identifier.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EndpointIdError {
    value: String,
}

impl EndpointIdError {
    /// Return the rejected spelling.
    pub fn value(&self) -> &str {
        &self.value
    }
}

impl Display for EndpointIdError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "invalid endpoint ID {:?}; expected [a-z][a-z0-9_]*",
            self.value
        )
    }
}

impl Error for EndpointIdError {}

/// Endpoint registry construction, lookup, or preparation failure.
#[derive(Debug)]
pub enum EndpointRegistryError {
    /// A descriptor contains an invalid canonical ID or alias.
    InvalidId(EndpointIdError),
    /// A canonical ID or alias collides with an existing registration.
    DuplicateName {
        /// Colliding name.
        name: EndpointId,
        /// Existing canonical owner.
        existing: EndpointId,
        /// Canonical owner being registered.
        incoming: EndpointId,
    },
    /// An authored ID is valid but absent from this compiled runner.
    UnknownEndpoint {
        /// Authored canonical ID or alias.
        requested: EndpointId,
        /// Deterministically ordered canonical catalog.
        available: Vec<EndpointId>,
    },
    /// Endpoint-specific validation or preparation failed.
    Preparation(EndpointError),
    /// More dense prepared bindings were added than `u32` can address.
    TooManyPreparedEndpoints,
    /// A dense endpoint key is outside the worker-local table.
    UnknownEndpointKey(EndpointKey),
    /// A protocol-v1 caller selected a factory without an [`Endpoint`] adapter.
    NoLegacyAdapter(EndpointId),
}

impl Display for EndpointRegistryError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidId(error) => Display::fmt(error, formatter),
            Self::DuplicateName {
                name,
                existing,
                incoming,
            } => write!(
                formatter,
                "endpoint name {name:?} for {incoming:?} conflicts with registered endpoint {existing:?}"
            ),
            Self::UnknownEndpoint {
                requested,
                available,
            } => write!(
                formatter,
                "unknown endpoint {requested:?}; compiled endpoints: {}",
                available
                    .iter()
                    .map(EndpointId::as_str)
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            Self::Preparation(error) => Display::fmt(error, formatter),
            Self::TooManyPreparedEndpoints => {
                formatter.write_str("more than u32::MAX endpoint bindings were prepared")
            }
            Self::UnknownEndpointKey(key) => {
                write!(formatter, "endpoint key {} is not prepared", key.index())
            }
            Self::NoLegacyAdapter(id) => write!(
                formatter,
                "endpoint {id:?} has no protocol-v1 Endpoint compatibility adapter"
            ),
        }
    }
}

impl Error for EndpointRegistryError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::InvalidId(error) => Some(error),
            Self::Preparation(error) => Some(error),
            Self::DuplicateName { .. }
            | Self::UnknownEndpoint { .. }
            | Self::TooManyPreparedEndpoints
            | Self::UnknownEndpointKey(_)
            | Self::NoLegacyAdapter(_) => None,
        }
    }
}

impl From<EndpointIdError> for EndpointRegistryError {
    fn from(error: EndpointIdError) -> Self {
        Self::InvalidId(error)
    }
}

impl From<EndpointError> for EndpointRegistryError {
    fn from(error: EndpointError) -> Self {
        Self::Preparation(error)
    }
}

/// Startup factory for one endpoint dialect.
///
/// Factories are immutable registry values. Each worker calls [`Self::prepare`]
/// once per selected profile and owns the returned binding locally.
pub trait EndpointFactory: fmt::Debug + Send + Sync {
    /// Return the canonical descriptor registered with this factory.
    fn descriptor(&self) -> &'static EndpointDescriptor;

    /// Apply endpoint-specific validation after generic descriptor policy.
    fn validate_config(&self, _config: &mut RawEndpointConfig) -> EndpointResult<()> {
        Ok(())
    }

    /// Construct one worker-local prepared binding.
    fn prepare(&self, config: EffectiveEndpointConfig)
    -> EndpointResult<Box<dyn PreparedEndpoint>>;

    /// Return the decoded-JSON adapter for protocol-v1 callers.
    fn legacy_endpoint(&self) -> Option<Arc<dyn Endpoint>> {
        None
    }
}

/// Borrowed request view consumed by prepared endpoints.
///
/// Endpoint policy is intentionally absent: a prepared endpoint can read only
/// its own validated [`EffectiveEndpointConfig`]. The protocol-v1
/// [`RequestInfo`] conversion is a compatibility wrapper outside the prepared
/// request path.
#[derive(Clone, Copy, Debug)]
pub struct PreparedRequest<'a> {
    primary_model_name: &'a str,
    turns: &'a [Turn],
    system_message: Option<&'a str>,
    user_context_message: Option<&'a str>,
    credit_phase: CreditPhase,
    x_request_id: Option<&'a str>,
    x_correlation_id: Option<&'a str>,
    conversation_id: Option<&'a str>,
}

impl<'a> PreparedRequest<'a> {
    /// Construct a borrowed request view with no endpoint configuration field.
    #[allow(clippy::too_many_arguments)]
    pub const fn new(
        primary_model_name: &'a str,
        turns: &'a [Turn],
        system_message: Option<&'a str>,
        user_context_message: Option<&'a str>,
        credit_phase: CreditPhase,
        x_request_id: Option<&'a str>,
        x_correlation_id: Option<&'a str>,
        conversation_id: Option<&'a str>,
    ) -> Self {
        Self {
            primary_model_name,
            turns,
            system_message,
            user_context_message,
            credit_phase,
            x_request_id,
            x_correlation_id,
            conversation_id,
        }
    }

    /// Primary model used unless a turn overrides it.
    pub const fn primary_model_name(&self) -> &'a str {
        self.primary_model_name
    }

    /// Authored and reconstructed turns for this dispatch.
    pub const fn turns(&self) -> &'a [Turn] {
        self.turns
    }

    /// Shared system message.
    pub const fn system_message(&self) -> Option<&'a str> {
        self.system_message
    }

    /// Per-conversation user context message.
    pub const fn user_context_message(&self) -> Option<&'a str> {
        self.user_context_message
    }

    /// Warmup or profiling phase.
    pub const fn credit_phase(&self) -> CreditPhase {
        self.credit_phase
    }

    /// Request correlation ID exposed to templates.
    pub const fn x_request_id(&self) -> Option<&'a str> {
        self.x_request_id
    }

    /// Session correlation ID exposed to templates.
    pub const fn x_correlation_id(&self) -> Option<&'a str> {
        self.x_correlation_id
    }

    /// Authored conversation ID exposed to templates.
    pub const fn conversation_id(&self) -> Option<&'a str> {
        self.conversation_id
    }
}

impl<'a> From<&'a RequestInfo> for PreparedRequest<'a> {
    fn from(request: &'a RequestInfo) -> Self {
        Self::new(
            &request.model_endpoint.primary_model_name,
            &request.turns,
            request.system_message.as_deref(),
            request.user_context_message.as_deref(),
            request.credit_phase,
            request.x_request_id.as_deref(),
            request.x_correlation_id.as_deref(),
            request.conversation_id.as_deref(),
        )
    }
}

/// Allocation-free formatter seam for protocol-v1 endpoint adapters.
pub trait PreparedEndpointBehavior: Endpoint {
    /// Format through a borrowed request and identity-free validated policy.
    fn format_prepared_payload(
        &self,
        request: &PreparedRequest<'_>,
        config: &RawEndpointConfig,
    ) -> EndpointResult<BodyPlan>;
}

pub(crate) fn format_legacy_payload<E>(
    endpoint: &E,
    request: &RequestInfo,
) -> EndpointResult<BodyPlan>
where
    E: PreparedEndpointBehavior + ?Sized,
{
    let config = RawEndpointConfig::from(&request.model_endpoint.endpoint);
    endpoint.format_prepared_payload(&PreparedRequest::from(request), &config)
}

/// HTTP method for one dialect-owned readiness request.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReadinessMethod {
    /// Readiness is queried without a request body.
    Get,
    /// Readiness is queried with the endpoint's decoded JSON body.
    Post,
}

/// Endpoint-owned predicate applied to a completed readiness response.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ReadinessSuccess {
    /// Any successful HTTP response proves the endpoint is ready.
    SuccessfulStatus,
    /// A model-listing response must contain this exact model id.
    ModelListed(String),
    /// Any response below a server error proves the inference stack is reachable.
    NonServerError,
}

/// Fully prepared dialect-owned readiness request.
#[derive(Clone, Debug, PartialEq)]
pub struct PreparedReadinessRequest {
    /// HTTP method selected by the dialect.
    pub method: ReadinessMethod,
    /// Absolute endpoint path selected by the dialect.
    pub path: String,
    /// Endpoint-owned authentication and content headers.
    pub headers: BTreeMap<String, String>,
    /// Optional decoded JSON request body.
    pub body: Option<Value>,
    /// Response condition required before the target is considered ready.
    pub success: ReadinessSuccess,
}

/// Explicit readiness behavior for one prepared endpoint profile.
#[derive(Clone, Debug, PartialEq)]
pub enum ReadinessPolicy {
    /// This dialect does not define a readiness request.
    Unsupported {
        /// Actionable reason surfaced by runner validation.
        reason: &'static str,
    },
    /// Exact request to send through the ordinary transport stack.
    Request(PreparedReadinessRequest),
    /// Ordered readiness requests that must all complete for one target.
    Requests(Vec<PreparedReadinessRequest>),
}

/// Worker-local endpoint binding prepared from one factory and effective config.
pub trait PreparedEndpoint: fmt::Debug {
    /// Canonical descriptor selected for this binding.
    fn descriptor(&self) -> &'static EndpointDescriptor;

    /// Validated immutable effective configuration.
    fn config(&self) -> &EffectiveEndpointConfig;

    /// Build a request-body plan the shared materializer splices into wire bytes.
    fn format_payload(&self, request: &PreparedRequest<'_>) -> EndpointResult<BodyPlan>;

    /// Whether static bind-time inputs fully determine the body, allowing the
    /// [`BodyPlan`] to be cached.
    ///
    /// Dialects using dispatch identity, raw payloads, or token IDs return
    /// `false`. This runtime property is absent from the capability wire.
    fn precomputable_body(&self) -> bool {
        true
    }

    /// Borrow endpoint-owned request headers prepared once per worker/profile.
    fn headers(&self) -> &BTreeMap<String, String>;

    /// Build an exact endpoint-owned readiness policy for one model.
    fn readiness_policy(&self, model: &str) -> EndpointResult<ReadinessPolicy>;

    /// Parse one decoded server response using prepared configuration.
    fn parse_response(&self, response: &ServerResponse) -> EndpointResult<Option<ParsedResponse>>;

    /// Extract tokenizable input and media counts from a built body.
    fn extract_payload_inputs(&self, body: &Value) -> ExtractedPayload;

    /// Parse every response in a request record.
    fn extract_response_data(&self, record: &RequestRecord) -> EndpointResult<Vec<ParsedResponse>>;

    /// Build an assistant turn for context replay.
    fn build_assistant_turn(&self, record: &RequestRecord) -> EndpointResult<Option<Turn>>;

    /// Whether successful responses should be reconstructed for later turns.
    fn captures_assistant_turn(&self) -> bool;
}

/// Factory wrapper for an existing immutable [`Endpoint`] implementation.
#[derive(Debug, Clone)]
pub struct StatelessEndpointFactory<E> {
    endpoint: Arc<E>,
}

impl<E> StatelessEndpointFactory<E> {
    /// Wrap one immutable decoded-JSON endpoint adapter.
    pub fn new(endpoint: E) -> Self {
        Self {
            endpoint: Arc::new(endpoint),
        }
    }
}

impl<E> EndpointFactory for StatelessEndpointFactory<E>
where
    E: PreparedEndpointBehavior + 'static,
{
    fn descriptor(&self) -> &'static EndpointDescriptor {
        self.endpoint.descriptor()
    }

    fn prepare(
        &self,
        config: EffectiveEndpointConfig,
    ) -> EndpointResult<Box<dyn PreparedEndpoint>> {
        let legacy_config = EndpointConfig::from_raw(
            self.endpoint
                .descriptor()
                .legacy_type()
                .expect("protocol-v1 Endpoint factory must map to EndpointType"),
            config.to_raw(),
        );
        let headers = self.endpoint.format_headers(&legacy_config);
        Ok(Box::new(StatelessPreparedEndpoint {
            endpoint: self.endpoint.clone(),
            config,
            legacy_config,
            headers,
        }))
    }

    fn legacy_endpoint(&self) -> Option<Arc<dyn Endpoint>> {
        Some(self.endpoint.clone())
    }
}

#[derive(Debug)]
struct StatelessPreparedEndpoint<E> {
    endpoint: Arc<E>,
    config: EffectiveEndpointConfig,
    legacy_config: EndpointConfig,
    headers: BTreeMap<String, String>,
}

impl<E> PreparedEndpoint for StatelessPreparedEndpoint<E>
where
    E: PreparedEndpointBehavior + 'static,
{
    fn descriptor(&self) -> &'static EndpointDescriptor {
        self.endpoint.descriptor()
    }

    fn config(&self) -> &EffectiveEndpointConfig {
        &self.config
    }

    fn format_payload(&self, request: &PreparedRequest<'_>) -> EndpointResult<BodyPlan> {
        self.endpoint
            .format_prepared_payload(request, self.config.as_raw())
    }

    fn headers(&self) -> &BTreeMap<String, String> {
        &self.headers
    }

    fn readiness_policy(&self, model: &str) -> EndpointResult<ReadinessPolicy> {
        self.endpoint.readiness_policy(&self.legacy_config, model)
    }

    fn parse_response(&self, response: &ServerResponse) -> EndpointResult<Option<ParsedResponse>> {
        self.endpoint
            .parse_response_with_config(response, &self.legacy_config)
    }

    fn extract_payload_inputs(&self, body: &Value) -> ExtractedPayload {
        self.endpoint.extract_payload_inputs(body)
    }

    fn extract_response_data(&self, record: &RequestRecord) -> EndpointResult<Vec<ParsedResponse>> {
        self.endpoint
            .extract_response_data_with_config(record, &self.legacy_config)
    }

    fn build_assistant_turn(&self, record: &RequestRecord) -> EndpointResult<Option<Turn>> {
        self.endpoint.build_assistant_turn(record)
    }

    fn captures_assistant_turn(&self) -> bool {
        self.endpoint.captures_assistant_turn()
    }
}

/// Mutable startup-only endpoint registry.
#[derive(Clone, Default)]
pub struct EndpointRegistryBuilder {
    entries: BTreeMap<EndpointId, Arc<dyn EndpointFactory>>,
    owners: BTreeMap<EndpointId, EndpointId>,
}

impl fmt::Debug for EndpointRegistryBuilder {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("EndpointRegistryBuilder")
            .field("canonical_ids", &self.entries.keys().collect::<Vec<_>>())
            .finish()
    }
}

impl EndpointRegistryBuilder {
    /// Construct an empty startup builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Construct the stock built-in endpoint catalog.
    pub fn with_builtins() -> Result<Self, EndpointRegistryError> {
        let mut builder = Self::new();
        builder.register_builtins()?;
        Ok(builder)
    }

    /// Register the stock endpoint dialects into an existing builder.
    pub fn register_builtins(&mut self) -> Result<(), EndpointRegistryError> {
        let builder = self;
        builder.register_endpoint(ChatEndpoint)?;
        builder.register_endpoint(CompletionsEndpoint)?;
        builder.register_endpoint(ResponsesEndpoint)?;
        builder.register_endpoint(MessagesEndpoint)?;
        builder.register_endpoint(EmbeddingsEndpoint)?;
        builder.register_endpoint(ChatEmbeddingsEndpoint)?;
        builder.register_endpoint(NimEmbeddingsEndpoint)?;
        builder.register_endpoint(CohereRankingsEndpoint)?;
        builder.register_endpoint(HfTeiRankingsEndpoint)?;
        builder.register_endpoint(NimRankingsEndpoint)?;
        builder.register_endpoint(HuggingFaceGenerateEndpoint)?;
        builder.register_endpoint(ImageGenerationEndpoint)?;
        builder.register_endpoint(ImageEditEndpoint)?;
        builder.register_endpoint(VideoGenerationEndpoint)?;
        builder.register_endpoint(ImageRetrievalEndpoint)?;
        builder.register_endpoint(SolidoRagEndpoint)?;
        builder.register_factory(RawEndpointFactory)?;
        builder.register_factory(TemplateEndpointFactory)?;
        builder.register_factory(VllmGenerateFactory)?;
        builder.register_factory(DynosimEndpointFactory)?;
        builder.register_factory(KServeChatFactory)?;
        builder.register_factory(KServeCompletionsFactory)?;
        builder.register_factory(KServeEmbeddingsFactory)?;
        builder.register_factory(KServeV1PredictFactory)?;
        builder.register_factory(KServeV2InferFactory)?;
        builder.register_factory(KServeV2EmbeddingsFactory)?;
        builder.register_factory(KServeV2RankingsFactory)?;
        builder.register_factory(KServeV2VlmFactory)?;
        builder.register_factory(KServeV2ImagesFactory)?;
        builder.register_factory(RivaAsrFactory)?;
        builder.register_factory(RivaTtsFactory)?;
        builder.register_factory(RivaTextClassifyFactory)?;
        builder.register_factory(RivaTokenClassifyFactory)?;
        builder.register_factory(RivaTransformTextFactory)?;
        builder.register_factory(RivaPunctuateTextFactory)?;
        builder.register_factory(RivaNaturalQueryFactory)?;
        builder.register_factory(RivaAnalyzeIntentFactory)?;
        builder.register_factory(RivaAnalyzeEntitiesFactory)?;
        Ok(())
    }

    /// Register an immutable protocol-v1 endpoint through the stateless factory.
    pub fn register_endpoint<E>(&mut self, endpoint: E) -> Result<(), EndpointRegistryError>
    where
        E: PreparedEndpointBehavior + 'static,
    {
        self.register_factory(StatelessEndpointFactory::new(endpoint))
    }

    /// Register one concrete factory atomically with its descriptor aliases.
    pub fn register_factory<F>(&mut self, factory: F) -> Result<(), EndpointRegistryError>
    where
        F: EndpointFactory + 'static,
    {
        self.register_factory_arc(Arc::new(factory))
    }

    /// Register one shared factory atomically with its descriptor aliases.
    pub fn register_factory_arc(
        &mut self,
        factory: Arc<dyn EndpointFactory>,
    ) -> Result<(), EndpointRegistryError> {
        let descriptor = factory.descriptor();
        let canonical = EndpointId::new(descriptor.id)?;
        let aliases = descriptor
            .aliases
            .iter()
            .map(EndpointId::new)
            .collect::<Result<Vec<_>, _>>()?;

        let mut proposed = BTreeSet::from([canonical.clone()]);
        for alias in &aliases {
            if !proposed.insert(alias.clone()) {
                return Err(EndpointRegistryError::DuplicateName {
                    name: alias.clone(),
                    existing: canonical.clone(),
                    incoming: canonical.clone(),
                });
            }
        }
        for name in &proposed {
            if let Some(existing) = self.owners.get(name) {
                return Err(EndpointRegistryError::DuplicateName {
                    name: name.clone(),
                    existing: existing.clone(),
                    incoming: canonical.clone(),
                });
            }
        }

        self.entries.insert(canonical.clone(), factory);
        for name in proposed {
            self.owners.insert(name, canonical.clone());
        }
        Ok(())
    }

    /// Freeze the deterministic catalog for capability, validation, and run use.
    pub fn freeze(self) -> EndpointRegistry {
        let entries = self
            .entries
            .into_iter()
            .map(|(id, factory)| RegistryEntry { id, factory })
            .collect::<Vec<_>>();
        let indices = entries
            .iter()
            .enumerate()
            .map(|(index, entry)| (entry.id.clone(), index))
            .collect::<BTreeMap<_, _>>();
        EndpointRegistry {
            inner: Arc::new(RegistryInner {
                entries,
                owners: self.owners,
                indices,
            }),
        }
    }
}

#[derive(Clone)]
struct RegistryEntry {
    id: EndpointId,
    factory: Arc<dyn EndpointFactory>,
}

struct RegistryInner {
    entries: Vec<RegistryEntry>,
    owners: BTreeMap<EndpointId, EndpointId>,
    indices: BTreeMap<EndpointId, usize>,
}

/// Immutable deterministic endpoint catalog.
#[derive(Clone)]
pub struct EndpointRegistry {
    inner: Arc<RegistryInner>,
}

impl fmt::Debug for EndpointRegistry {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("EndpointRegistry")
            .field("canonical_ids", &self.canonical_ids().collect::<Vec<_>>())
            .finish()
    }
}

impl EndpointRegistry {
    /// Construct and freeze the stock built-in endpoint catalog.
    pub fn builtin() -> Result<Self, EndpointRegistryError> {
        Ok(EndpointRegistryBuilder::with_builtins()?.freeze())
    }

    /// Clone startup factories into a mutable extension builder.
    pub fn to_builder(&self) -> EndpointRegistryBuilder {
        EndpointRegistryBuilder {
            entries: self
                .inner
                .entries
                .iter()
                .map(|entry| (entry.id.clone(), entry.factory.clone()))
                .collect(),
            owners: self.inner.owners.clone(),
        }
    }

    /// Number of canonical entries; aliases are not counted.
    pub fn len(&self) -> usize {
        self.inner.entries.len()
    }

    /// Whether the canonical catalog is empty.
    pub fn is_empty(&self) -> bool {
        self.inner.entries.is_empty()
    }

    /// Iterate canonical IDs in deterministic lexical order.
    pub fn canonical_ids(&self) -> impl ExactSizeIterator<Item = &EndpointId> {
        self.inner.entries.iter().map(|entry| &entry.id)
    }

    /// Iterate canonical descriptors in deterministic lexical order.
    pub fn descriptors(&self) -> impl ExactSizeIterator<Item = &'static EndpointDescriptor> + '_ {
        self.inner
            .entries
            .iter()
            .map(|entry| entry.factory.descriptor())
    }

    /// Resolve a canonical ID or explicit alias to its canonical ID.
    pub fn canonical_id(&self, id: &EndpointId) -> Result<&EndpointId, EndpointRegistryError> {
        self.inner.owners.get(id).ok_or_else(|| self.unknown(id))
    }

    /// Resolve one startup factory by canonical ID or explicit alias.
    pub fn resolve_factory(
        &self,
        id: &EndpointId,
    ) -> Result<Arc<dyn EndpointFactory>, EndpointRegistryError> {
        let canonical = self.canonical_id(id)?;
        let index = self.inner.indices[canonical];
        Ok(self.inner.entries[index].factory.clone())
    }

    /// Validate and prepare one worker-local endpoint binding.
    pub fn prepare(
        &self,
        id: &EndpointId,
        config: RawEndpointConfig,
    ) -> Result<Box<dyn PreparedEndpoint>, EndpointRegistryError> {
        let factory = self.resolve_factory(id)?;
        let descriptor = factory.descriptor();
        let mut normalized = config.validate_for_descriptor(descriptor)?;
        factory.validate_config(&mut normalized)?;
        let normalized = normalized.validate_for_descriptor(descriptor)?;
        factory
            .prepare(EffectiveEndpointConfig::from_validated(normalized))
            .map_err(EndpointRegistryError::from)
    }

    /// Resolve the adapter used by protocol-v1 materializers.
    pub fn legacy_endpoint(
        &self,
        id: &EndpointId,
    ) -> Result<Arc<dyn Endpoint>, EndpointRegistryError> {
        let canonical = self.canonical_id(id)?.clone();
        self.resolve_factory(&canonical)?
            .legacy_endpoint()
            .ok_or(EndpointRegistryError::NoLegacyAdapter(canonical))
    }

    fn unknown(&self, id: &EndpointId) -> EndpointRegistryError {
        EndpointRegistryError::UnknownEndpoint {
            requested: id.clone(),
            available: self.canonical_ids().cloned().collect(),
        }
    }
}

/// Object-safe lookup seam for an injected frozen endpoint catalog.
pub trait EndpointResolver: fmt::Debug + Send + Sync {
    /// Resolve one canonical ID or alias to its immutable startup factory.
    fn resolve_factory(
        &self,
        id: &EndpointId,
    ) -> Result<Arc<dyn EndpointFactory>, EndpointRegistryError>;
}

impl EndpointResolver for EndpointRegistry {
    fn resolve_factory(
        &self,
        id: &EndpointId,
    ) -> Result<Arc<dyn EndpointFactory>, EndpointRegistryError> {
        EndpointRegistry::resolve_factory(self, id)
    }
}

/// Dense worker-local endpoint binding key carried by scheduled turns.
#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
#[serde(transparent)]
pub struct EndpointKey(u32);

impl EndpointKey {
    /// Rehydrate a dense key carried over an authenticated execution channel.
    ///
    /// The receiving worker must still resolve the key through its own
    /// [`PreparedEndpointTable`], which rejects out-of-range values.
    pub const fn from_index(index: u32) -> Self {
        Self(index)
    }

    /// Return the worker-local vector index.
    pub const fn index(self) -> u32 {
        self.0
    }
}

/// Worker-local dense prepared endpoint bindings.
#[derive(Debug, Default)]
pub struct PreparedEndpointTable {
    entries: Vec<Box<dyn PreparedEndpoint>>,
}

impl PreparedEndpointTable {
    /// Construct an empty worker-local table.
    pub const fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Append one prepared profile and return its copyable dense key.
    pub fn push(
        &mut self,
        endpoint: Box<dyn PreparedEndpoint>,
    ) -> Result<EndpointKey, EndpointRegistryError> {
        let index = u32::try_from(self.entries.len())
            .map_err(|_| EndpointRegistryError::TooManyPreparedEndpoints)?;
        self.entries.push(endpoint);
        Ok(EndpointKey(index))
    }

    /// Borrow one prepared endpoint by dense key without name lookup.
    pub fn get(&self, key: EndpointKey) -> Result<&dyn PreparedEndpoint, EndpointRegistryError> {
        self.entries
            .get(key.0 as usize)
            .map(Box::as_ref)
            .ok_or(EndpointRegistryError::UnknownEndpointKey(key))
    }

    /// Number of worker-local prepared bindings.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the worker-local table is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}
