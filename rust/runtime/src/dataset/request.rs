// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Stateful conversation reconstruction and endpoint request materialization.
//!
//! `message_array_without_responses` prefix-diffs successive authored snapshots
//! and interleaves each captured live reply.

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use crate::endpoints::{
    ChatEmbeddingsEndpoint, ChatEndpoint, CohereRankingsEndpoint, CompletionsEndpoint, CreditPhase,
    EmbeddingsEndpoint, Endpoint, ExtractedPayload, HfTeiRankingsEndpoint,
    HuggingFaceGenerateEndpoint, ImageEditEndpoint, ImageGenerationEndpoint,
    ImageRetrievalEndpoint, Media, MessagesEndpoint, ModelEndpoint, NimEmbeddingsEndpoint,
    NimRankingsEndpoint, PreparedEndpoint, PreparedRequest, RawEndpoint, RequestInfo,
    ResponsesEndpoint, SolidoRagEndpoint, TemplateEndpoint, Turn as EndpointTurn,
    VideoGenerationEndpoint,
};
use bytes::Bytes;
use serde_json::{Map, Value};

use crate::body_plan::{BodyPlan, JsonBodyMaterializer, RequestBody, WireSplice};
use crate::dataset::dataset::{CachedTurnPlan, Dataset};
use crate::dataset::error::{DatasetError, Result};
use crate::dataset::materialize::{Overrides, message_wire};
use crate::dataset::model::{
    AccuracyAssociation, Conversation, ConversationContextMode, MediaKind, SessionId, Turn,
};
use crate::dataset::segment::{Handle, Payload, SegmentDomain, SegmentStore};
use smallvec::SmallVec;

/// One fully built dispatch request and its media-free accounting metadata.
#[derive(Debug, Clone, PartialEq)]
pub struct MaterializedRequest {
    /// The request body crossing to the transport. Raw payloads retain their
    /// authored bytes exactly unless explicit dispatch overrides were supplied.
    pub body: RequestBody,
    /// Per-turn headers, after validating every value as a string.
    pub headers: BTreeMap<String, String>,
    /// Per-turn URL query parameters.
    pub parameters: BTreeMap<String, String>,
    /// Authored endpoint/dialect override, when present.
    pub endpoint: Option<String>,
    /// Resolved endpoint path after endpoint configuration and capability metadata.
    pub endpoint_path: Option<String>,
    /// Effective model name.
    pub model: String,
    /// Requested output-token cap.
    pub max_tokens: Option<u32>,
    /// Effective response streaming mode.
    pub streaming: bool,
    /// Precomputed input-token count including selected history and captured replies.
    ///
    /// `None` when the composed turn did not establish a client token count
    /// (opaque raw request bodies that are not token-native).
    pub input_tokens: Option<u64>,
    /// Segment handle for exact raw token IDs, retained for token-native
    /// backends that bypass the serialized HTTP body.
    pub raw_token_ids: Option<Handle>,
    /// Audio duration used by ASR metrics.
    pub audio_duration_seconds: Option<f64>,
    /// Exact wire image count known from the composed turn content, when it is
    /// sound to trust without re-parsing the serialized body (see
    /// [`known_image_count`]). `None` means "unknown here — derive it at dispatch",
    /// so raw payloads and history-accumulating continuation turns stay correct.
    pub image_count: Option<u32>,
    /// Opaque evaluator association propagated without positional matching.
    pub accuracy: Option<AccuracyAssociation>,
    /// Zero-based authored turn index.
    pub turn_index: usize,
    /// Whether this is the final authored turn.
    pub is_final_turn: bool,
    /// Input structure the endpoint reported for this exact body, when it can
    /// supply one without re-parsing (see [`PreparedEndpoint::extracted`]).
    ///
    /// `None` means "derive it by parsing `body`", which is always correct.
    pub extracted: Option<ExtractedPayload>,
}

/// Endpoint-independent request-materialization extension point.
pub trait RequestMaterializer: Send + Sync {
    /// Build the current session turn through `endpoint` and apply explicit
    /// dispatch overrides after authored/endpoint fields.
    fn materialize(
        &self,
        session: &ConversationSession,
        endpoint: &dyn Endpoint,
        model_endpoint: &ModelEndpoint,
        phase: CreditPhase,
        overrides: &Overrides,
    ) -> Result<MaterializedRequest>;

    /// Build the current turn directly through a worker-local prepared
    /// endpoint binding. The endpoint owns all normalized configuration;
    /// callers provide only the primary model and per-dispatch overrides.
    fn materialize_prepared(
        &self,
        session: &ConversationSession,
        endpoint: &dyn PreparedEndpoint,
        primary_model_name: &str,
        phase: CreditPhase,
        overrides: &Overrides,
    ) -> Result<MaterializedRequest>;
}

/// Lookup seam for authored per-turn endpoint/dialect overrides.
pub trait EndpointResolver: Send + Sync {
    /// Resolve an optional authored endpoint name, falling back to the registry default.
    fn resolve(&self, name: Option<&str>) -> Result<Arc<dyn Endpoint>>;
    /// Resolve the typed endpoint selected by run configuration when a dataset
    /// turn has no authored name override.
    fn resolve_type(
        &self,
        endpoint_type: crate::endpoints::EndpointType,
    ) -> Result<Arc<dyn Endpoint>>;
}

/// Extensible name-to-endpoint registry containing built-in
/// [`crate::endpoints`] implementations and statically linked extensions.
#[derive(Clone)]
pub struct BuiltinEndpointResolver {
    default_name: String,
    endpoints: HashMap<String, Arc<dyn Endpoint>>,
    endpoint_types: HashMap<crate::endpoints::EndpointType, Arc<dyn Endpoint>>,
}

impl std::fmt::Debug for BuiltinEndpointResolver {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut names = self.endpoints.keys().collect::<Vec<_>>();
        names.sort();
        f.debug_struct("BuiltinEndpointResolver")
            .field("default_name", &self.default_name)
            .field("names", &names)
            .finish()
    }
}

impl BuiltinEndpointResolver {
    /// Construct the registry with Chat Completions as its default.
    pub fn chat_default() -> Self {
        let mut resolver = Self {
            default_name: "chat".into(),
            endpoints: HashMap::new(),
            endpoint_types: HashMap::new(),
        };
        resolver
            .register("chat", ChatEndpoint)
            .expect("built-in endpoint names are unique");
        resolver
            .register("chat_completions", ChatEndpoint)
            .expect("built-in endpoint names are unique");
        resolver
            .register("completions", CompletionsEndpoint)
            .expect("built-in endpoint names are unique");
        resolver
            .register("responses", ResponsesEndpoint)
            .expect("built-in endpoint names are unique");
        resolver
            .register("messages", MessagesEndpoint)
            .expect("built-in endpoint names are unique");
        resolver
            .register("embeddings", EmbeddingsEndpoint)
            .expect("built-in endpoint names are unique");
        resolver
            .register("chat_embeddings", ChatEmbeddingsEndpoint)
            .expect("built-in endpoint names are unique");
        resolver
            .register("nim_embeddings", NimEmbeddingsEndpoint)
            .expect("built-in endpoint names are unique");
        resolver
            .register("cohere_rankings", CohereRankingsEndpoint)
            .expect("built-in endpoint names are unique");
        resolver
            .register("hf_tei_rankings", HfTeiRankingsEndpoint)
            .expect("built-in endpoint names are unique");
        resolver
            .register("nim_rankings", NimRankingsEndpoint)
            .expect("built-in endpoint names are unique");
        resolver
            .register("huggingface_generate", HuggingFaceGenerateEndpoint)
            .expect("built-in endpoint names are unique");
        resolver
            .register("image_generation", ImageGenerationEndpoint)
            .expect("built-in endpoint names are unique");
        resolver
            .register("image_edit", ImageEditEndpoint)
            .expect("built-in endpoint names are unique");
        resolver
            .register("video_generation", VideoGenerationEndpoint)
            .expect("built-in endpoint names are unique");
        resolver
            .register("image_retrieval", ImageRetrievalEndpoint)
            .expect("built-in endpoint names are unique");
        resolver
            .register("solido_rag", SolidoRagEndpoint)
            .expect("built-in endpoint names are unique");
        resolver
            .register("raw", RawEndpoint)
            .expect("built-in endpoint names are unique");
        resolver
            .register("template", TemplateEndpoint)
            .expect("built-in endpoint names are unique");
        resolver
    }

    /// Select the fallback endpoint name after verifying it is registered.
    pub fn with_default(mut self, name: &str) -> Result<Self> {
        self.set_default(name)?;
        Ok(self)
    }

    /// Change the fallback endpoint after verifying it is registered.
    pub fn set_default(&mut self, name: &str) -> Result<()> {
        let normalized = normalize_endpoint_name(name);
        if !self.endpoints.contains_key(&normalized) {
            return Err(DatasetError::Validation(format!(
                "default endpoint {name:?} is not registered"
            )));
        }
        self.default_name = normalized;
        Ok(())
    }

    /// Register one endpoint implementation, rejecting duplicate normalized names.
    pub fn register(
        &mut self,
        name: impl Into<String>,
        endpoint: impl Endpoint + 'static,
    ) -> Result<()> {
        let authored = name.into();
        let normalized = normalize_endpoint_name(&authored);
        if normalized.is_empty() {
            return Err(DatasetError::Validation(
                "endpoint registration name cannot be empty".into(),
            ));
        }
        if self.endpoints.contains_key(&normalized) {
            return Err(DatasetError::Validation(format!(
                "duplicate endpoint registration {authored:?}"
            )));
        }
        let endpoint_type = endpoint.descriptor().legacy_type().expect("endpoint type");
        let endpoint: Arc<dyn Endpoint> = Arc::new(endpoint);
        self.endpoint_types
            .entry(endpoint_type)
            .or_insert_with(|| endpoint.clone());
        self.endpoints.insert(normalized, endpoint);
        Ok(())
    }
}

impl Default for BuiltinEndpointResolver {
    fn default() -> Self {
        Self::chat_default()
    }
}

impl EndpointResolver for BuiltinEndpointResolver {
    fn resolve(&self, name: Option<&str>) -> Result<Arc<dyn Endpoint>> {
        let normalized = name
            .map(normalize_endpoint_name)
            .unwrap_or_else(|| self.default_name.clone());
        self.endpoints.get(&normalized).cloned().ok_or_else(|| {
            let mut available = self.endpoints.keys().cloned().collect::<Vec<_>>();
            available.sort();
            DatasetError::Validation(format!(
                "unknown dataset endpoint {normalized:?}; registered endpoints: {}",
                available.join(", ")
            ))
        })
    }

    fn resolve_type(
        &self,
        endpoint_type: crate::endpoints::EndpointType,
    ) -> Result<Arc<dyn Endpoint>> {
        self.endpoint_types
            .get(&endpoint_type)
            .cloned()
            .ok_or_else(|| {
                DatasetError::Validation(format!(
                    "endpoint type {endpoint_type:?} is not registered"
                ))
            })
    }
}

fn normalize_endpoint_name(name: &str) -> String {
    name.trim().to_ascii_lowercase().replace(['-', '/'], "_")
}

/// Native materializer backed by [`crate::endpoints`] formatters.
#[derive(Debug, Clone, Copy, Default)]
pub struct EndpointRequestMaterializer;

/// Build one handle-free WebSocket operation while its segment store is available.
#[derive(Debug, Clone, Copy, Default)]
pub struct WsRequestMaterializer;

/// Select a complete prebuilt body via [`Turn::body`] and its segment domain.
/// A `raw`-domain `body[0]` is dispatched byte-for-byte without endpoint
/// formatting; a `message`- or
/// `token-ids`-domain body falls through to the formatter / token-native path.
pub(crate) fn raw_body_handle<S: SegmentStore + ?Sized>(
    current: &Turn,
    store: &S,
) -> Result<Option<Handle>> {
    match current.body.first() {
        Some(&handle) if store.domain(handle)? == SegmentDomain::Raw => Ok(Some(handle)),
        _ => Ok(None),
    }
}

/// The token-native handle carried in [`Turn::body`], if any. A turn holds at most one `TokenIds`
/// segment; when a raw body coexists (`[raw, token]`) the raw body wins
/// dispatch and this handle stays reachable for token-count validation and
/// token-native backends.
pub(crate) fn token_ids_handle<S: SegmentStore + ?Sized>(
    current: &Turn,
    store: &S,
) -> Result<Option<Handle>> {
    for &handle in &current.body {
        if store.domain(handle)? == SegmentDomain::TokenIds {
            return Ok(Some(handle));
        }
    }
    Ok(None)
}

/// The ordered `Message`-domain handles carried in [`Turn::body`].
pub(crate) fn body_message_handles<S: SegmentStore + ?Sized>(
    current: &Turn,
    store: &S,
) -> Result<SmallVec<[Handle; 1]>> {
    let mut handles = SmallVec::new();
    for &handle in &current.body {
        if store.domain(handle)? == SegmentDomain::Message {
            handles.push(handle);
        }
    }
    Ok(handles)
}

impl RequestMaterializer for EndpointRequestMaterializer {
    fn materialize(
        &self,
        session: &ConversationSession,
        endpoint: &dyn Endpoint,
        model_endpoint: &ModelEndpoint,
        phase: CreditPhase,
        overrides: &Overrides,
    ) -> Result<MaterializedRequest> {
        let (conversation, current, turn_index) = session.current()?;
        let store = session.dataset.segments().as_ref();
        let (body, effective) = if let Some(raw) = raw_body_handle(current, store)? {
            (
                JsonBodyMaterializer::materialize(&BodyPlan::raw(raw), store, overrides)?,
                EffectiveRequest {
                    model: effective_model(current, &model_endpoint.primary_model_name, overrides)?,
                    max_tokens: effective_max_tokens(current, overrides)?,
                    streaming: effective_streaming(
                        current,
                        model_endpoint.endpoint.streaming,
                        endpoint.descriptor().supports_streaming,
                        overrides,
                    )?,
                },
            )
        } else {
            let streaming = effective_streaming(
                current,
                model_endpoint.endpoint.streaming,
                endpoint.descriptor().supports_streaming,
                overrides,
            )?;
            let mut effective_model_endpoint = model_endpoint.clone();
            effective_model_endpoint.endpoint.streaming = streaming;
            // The legacy `Endpoint` path keeps every turn's composed media; only
            // the prepared dispatch path is hot enough to justify the narrower
            // spliced resolution.
            let turns = session.endpoint_turns(store, false)?;
            let request_info = RequestInfo {
                model_endpoint: effective_model_endpoint,
                turns,
                system_message: resolve_prompt(store, conversation.system)?,
                user_context_message: resolve_prompt(store, conversation.user_context)?,
                credit_phase: phase,
                x_request_id: None,
                x_correlation_id: None,
                conversation_id: Some(session.conversation_id().as_str().to_string()),
            };
            let mut plan = endpoint.format_payload(&request_info)?;
            let (effective, stream_fix) = effective_from_plan(
                &plan,
                current,
                &model_endpoint.primary_model_name,
                model_endpoint.endpoint.streaming,
                endpoint.descriptor().supports_streaming,
                overrides,
            )?;
            apply_dispatch_mutations(&mut plan, overrides, stream_fix);
            (plan.materialize_standalone()?, effective)
        };

        let endpoint_path = model_endpoint.endpoint.path.clone().or_else(|| {
            if effective.streaming {
                endpoint.descriptor().streaming_path
            } else {
                None
            }
            .or(endpoint.descriptor().endpoint_path)
            .map(str::to_string)
        });
        let mut headers = endpoint.format_headers(&model_endpoint.endpoint);
        headers.extend(raw_string_map(
            store,
            current.extra_headers,
            "extra_headers",
        )?);
        Ok(MaterializedRequest {
            body: RequestBody::wire(body),
            headers,
            parameters: raw_string_map(store, current.request_parameters, "request_parameters")?,
            endpoint: current.endpoint.clone(),
            endpoint_path,
            model: effective.model,
            max_tokens: effective.max_tokens,
            streaming: effective.streaming,
            input_tokens: session.input_tokens(store)?,
            raw_token_ids: None,
            audio_duration_seconds: current.audio_duration_seconds,
            // Established up front by dataset precompute over the same turn set
            // this body assembles, plus the replies captured so far; `None`
            // leaves dispatch to derive it by parsing the body.
            image_count: session.known_image_count(turn_index, overrides),
            accuracy: conversation.accuracy.clone(),
            turn_index,
            is_final_turn: turn_index + 1 == conversation.turns.len(),
            extracted: None,
        })
    }

    fn materialize_prepared(
        &self,
        session: &ConversationSession,
        endpoint: &dyn PreparedEndpoint,
        primary_model_name: &str,
        phase: CreditPhase,
        overrides: &Overrides,
    ) -> Result<MaterializedRequest> {
        let (conversation, current, turn_index) = session.current()?;
        let store = session.dataset.segments().as_ref();
        let configured_streaming = endpoint.config().streaming();
        let supports_streaming = endpoint.descriptor().supports_streaming;
        // Endpoint-reported input structure, offered only for a body this call
        // formatted itself with no dispatch override applied. Raw bodies, cached
        // plans, and overridden plans leave it absent so the counter parses.
        let mut extracted = None;
        let (body, effective) = if let Some(raw) = raw_body_handle(current, store)? {
            (
                JsonBodyMaterializer::materialize(&BodyPlan::raw(raw), store, overrides)?,
                EffectiveRequest {
                    model: effective_model(current, primary_model_name, overrides)?,
                    max_tokens: effective_max_tokens(current, overrides)?,
                    streaming: effective_streaming(
                        current,
                        configured_streaming,
                        supports_streaming,
                        overrides,
                    )?,
                },
            )
        } else {
            // Reuse the cached profiling-phase plan when the dataset precomputed
            // one for this exact `(conversation, turn)`. The
            // cache is only populated for the default endpoint, eligible context
            // modes, and eligible dialects, so a hit is byte-identical to a fresh
            // `format_payload` here; anything else returns `None` and falls back.
            // Warmup folds the system prompt inside the formatter, so it never
            // reuses the profiling-phase plan.
            let cached = (phase != CreditPhase::Warmup)
                .then(|| {
                    session
                        .dataset
                        .cached_body_plan(session.conversation_id(), turn_index)
                })
                .flatten();
            // A continuation turn's cached plan holds only its authored turns;
            // its captured replies are supplied here, borrowed from the session,
            // and interleaved as the body is emitted. `reply_wire_groups`
            // rejects a cache entry this dispatch's replies do not match, which
            // drops the whole turn back to live formatting.
            // Inline capacity is the conversation depth a continuation dispatch
            // stays allocation-free to; past it the groups spill to the heap for
            // one extra allocation per request. 16 covers the multi-turn depths
            // benchmarks actually run.
            let mut groups: SmallVec<[(u32, &[Bytes]); 16]> = SmallVec::new();
            let cached = cached.filter(|cached| session.reply_wire_groups(cached, &mut groups));
            let mut plan = match cached {
                // Shared, not copied: a dispatch that mutates nothing dispatches
                // straight off the dataset's plan.
                Some(cached) => {
                    #[cfg(test)]
                    CACHE_HITS.with(|count| count.set(count.get() + 1));
                    Arc::clone(&cached.plan)
                }
                None => {
                    let turns =
                        session.endpoint_turns(store, splices_lowered_wires(endpoint, phase))?;
                    let system_message = resolve_prompt(store, conversation.system)?;
                    let user_context_message = resolve_prompt(store, conversation.user_context)?;
                    let request = PreparedRequest::new(
                        primary_model_name,
                        &turns,
                        system_message.as_deref(),
                        user_context_message.as_deref(),
                        phase,
                        None,
                        None,
                        // Borrowed from the session, which outlives `request`;
                        // the identifier needs no per-dispatch copy.
                        Some(session.conversation_id().as_str()),
                    );
                    let plan = endpoint.format_payload(&request)?;
                    if overrides.is_empty() {
                        extracted = endpoint.extracted(&request, &plan);
                    }
                    Arc::new(plan)
                }
            };
            let splice = cached
                .and_then(|cached| cached.replies.as_ref())
                .filter(|replies| !overrides_replace_field(&plan, replies.field, overrides))
                .map(|replies| WireSplice::new(replies.field, &groups));
            #[cfg(test)]
            if splice.is_some() {
                SPLICED_DISPATCHES.with(|count| count.set(count.get() + 1));
            }
            let (effective, stream_fix) = effective_from_plan(
                &plan,
                current,
                primary_model_name,
                configured_streaming,
                supports_streaming,
                overrides,
            )?;
            // Copy the shared plan only for a dispatch that actually writes it.
            // The scheduled path dispatches with neither an override nor a stream
            // correction, so its normal case never copies.
            if !overrides.is_empty() || stream_fix.is_some() {
                apply_dispatch_mutations(Arc::make_mut(&mut plan), overrides, stream_fix);
                // The structure was captured from the plan the endpoint
                // formatted. A stream correction rewrites the `stream` literal
                // independently of `overrides`, so the dispatched body can
                // differ from the one described; drop the report rather than
                // let it describe a body that was not sent. Clearing here
                // rather than capturing later is forced: `request` is dropped
                // at the end of the arm above. The branch is not taken on
                // normal scheduled dispatch, so this costs the hot path
                // nothing.
                extracted = None;
            }
            (plan.materialize_spliced(splice.as_ref())?, effective)
        };

        let endpoint_path = endpoint.config().as_raw().path.clone().or_else(|| {
            if effective.streaming {
                endpoint.descriptor().streaming_path
            } else {
                None
            }
            .or(endpoint.descriptor().endpoint_path)
            .map(str::to_string)
        });
        let mut headers = endpoint.headers().clone();
        headers.extend(raw_string_map(
            store,
            current.extra_headers,
            "extra_headers",
        )?);
        Ok(MaterializedRequest {
            body: RequestBody::wire(body),
            headers,
            parameters: raw_string_map(store, current.request_parameters, "request_parameters")?,
            endpoint: current.endpoint.clone(),
            endpoint_path,
            model: effective.model,
            max_tokens: effective.max_tokens,
            streaming: effective.streaming,
            input_tokens: session.input_tokens(store)?,
            raw_token_ids: if endpoint.descriptor().requires_raw_token_ids {
                token_ids_handle(current, store)?
            } else {
                None
            },
            audio_duration_seconds: current.audio_duration_seconds,
            // Established up front by dataset precompute over the same turn set
            // this body assembles, plus the replies captured so far; `None`
            // leaves dispatch to derive it by parsing the body.
            image_count: session.known_image_count(turn_index, overrides),
            accuracy: conversation.accuracy.clone(),
            turn_index,
            is_final_turn: turn_index + 1 == conversation.turns.len(),
            extracted,
        })
    }
}

impl RequestMaterializer for WsRequestMaterializer {
    fn materialize(
        &self,
        _session: &ConversationSession,
        _endpoint: &dyn Endpoint,
        _model_endpoint: &ModelEndpoint,
        _phase: CreditPhase,
        _overrides: &Overrides,
    ) -> Result<MaterializedRequest> {
        Err(DatasetError::Validation(
            "WebSocket execution requires a prepared endpoint binding".to_owned(),
        ))
    }

    fn materialize_prepared(
        &self,
        session: &ConversationSession,
        endpoint: &dyn PreparedEndpoint,
        primary_model_name: &str,
        phase: CreditPhase,
        overrides: &Overrides,
    ) -> Result<MaterializedRequest> {
        materialize_websocket_prepared(session, endpoint, primary_model_name, phase, overrides)
    }
}

fn materialize_websocket_prepared(
    session: &ConversationSession,
    endpoint: &dyn PreparedEndpoint,
    primary_model_name: &str,
    phase: CreditPhase,
    overrides: &Overrides,
) -> Result<MaterializedRequest> {
    let (conversation, current, turn_index) = session.current()?;
    let store = session.dataset.segments().as_ref();
    if raw_body_handle(current, store)?.is_some() {
        return Err(DatasetError::Validation(
            "WebSocket request materialization does not accept opaque raw request bodies"
                .to_owned(),
        ));
    }
    let turns = session.endpoint_turns(store, splices_lowered_wires(endpoint, phase))?;
    let system_message = resolve_prompt(store, conversation.system)?;
    let user_context_message = resolve_prompt(store, conversation.user_context)?;
    let request = PreparedRequest::new(
        primary_model_name,
        &turns,
        system_message.as_deref(),
        user_context_message.as_deref(),
        phase,
        None,
        None,
        Some(session.conversation_id().as_str()),
    );
    let plan = endpoint.format_payload(&request)?;
    let (effective, _) = effective_from_plan(
        &plan,
        current,
        primary_model_name,
        endpoint.config().streaming(),
        endpoint.descriptor().supports_streaming,
        overrides,
    )?;
    let operation = endpoint.prepare_ws_operation(&request, &plan, store, overrides)?;
    let extracted = operation
        .input_projection()
        .map(|body| {
            serde_json::from_slice(body).map(|value| endpoint.extract_payload_inputs(&value))
        })
        .transpose()
        .map_err(|error| {
            DatasetError::Validation(format!(
                "WebSocket input-counting projection is not valid JSON: {error}"
            ))
        })?;
    let mut headers = endpoint.headers().clone();
    headers.extend(raw_string_map(
        store,
        current.extra_headers,
        "extra_headers",
    )?);
    Ok(MaterializedRequest {
        body: RequestBody::WebSocket(Arc::new(operation)),
        headers,
        parameters: raw_string_map(store, current.request_parameters, "request_parameters")?,
        endpoint: current.endpoint.clone(),
        endpoint_path: endpoint
            .config()
            .as_raw()
            .path
            .clone()
            .or_else(|| endpoint.descriptor().endpoint_path.map(str::to_string)),
        model: effective.model,
        max_tokens: effective.max_tokens,
        streaming: effective.streaming,
        input_tokens: session.input_tokens(store)?,
        raw_token_ids: None,
        audio_duration_seconds: current.audio_duration_seconds,
        image_count: session.known_image_count(turn_index, overrides),
        accuracy: conversation.accuracy.clone(),
        turn_index,
        is_final_turn: turn_index + 1 == conversation.turns.len(),
        extracted,
    })
}

/// Request materializer for simulator backends that consume stored trace hash
/// identities or exact raw token IDs instead of wire bytes.
///
/// Turns without a trace identity or an endpoint-required raw-token handle
/// delegate to [`EndpointRequestMaterializer`] unchanged. Native turns retain endpoint,
/// model, header, and query metadata but skip message reconstruction, endpoint
/// payload formatting, and JSON serialization. A caller must pair this
/// materializer with a dispatch adapter that resolves the stored handle; the
/// empty body is not a valid HTTP request.
#[derive(Debug, Clone, Copy, Default)]
pub struct TraceHashAwareRequestMaterializer;

impl RequestMaterializer for TraceHashAwareRequestMaterializer {
    fn materialize(
        &self,
        session: &ConversationSession,
        endpoint: &dyn Endpoint,
        model_endpoint: &ModelEndpoint,
        phase: CreditPhase,
        overrides: &Overrides,
    ) -> Result<MaterializedRequest> {
        let (conversation, current, turn_index) = session.current()?;
        if current.trace_hash_ids.is_none() {
            return EndpointRequestMaterializer.materialize(
                session,
                endpoint,
                model_endpoint,
                phase,
                overrides,
            );
        }
        let store = session.dataset.segments().as_ref();
        let effective = EffectiveRequest {
            model: effective_model(current, &model_endpoint.primary_model_name, overrides)?,
            max_tokens: effective_max_tokens(current, overrides)?,
            streaming: effective_streaming(
                current,
                model_endpoint.endpoint.streaming,
                endpoint.descriptor().supports_streaming,
                overrides,
            )?,
        };
        let endpoint_path = model_endpoint.endpoint.path.clone().or_else(|| {
            if effective.streaming {
                endpoint.descriptor().streaming_path
            } else {
                None
            }
            .or(endpoint.descriptor().endpoint_path)
            .map(str::to_string)
        });
        let mut headers = endpoint.format_headers(&model_endpoint.endpoint);
        headers.extend(raw_string_map(
            store,
            current.extra_headers,
            "extra_headers",
        )?);
        Ok(MaterializedRequest {
            body: RequestBody::wire(Bytes::new()),
            headers,
            parameters: raw_string_map(store, current.request_parameters, "request_parameters")?,
            endpoint: current.endpoint.clone(),
            endpoint_path,
            model: effective.model,
            max_tokens: effective.max_tokens,
            streaming: effective.streaming,
            input_tokens: session.input_tokens(store)?,
            raw_token_ids: None,
            audio_duration_seconds: current.audio_duration_seconds,
            // The dispatched body is empty here, so only a first turn's authored
            // content is describable; later turns parse.
            image_count: raw_body_handle(current, store)
                .ok()
                .flatten()
                .is_none()
                .then(|| trace_identity_image_count(current, turn_index))
                .flatten(),
            accuracy: conversation.accuracy.clone(),
            turn_index,
            is_final_turn: turn_index + 1 == conversation.turns.len(),
            extracted: None,
        })
    }

    fn materialize_prepared(
        &self,
        session: &ConversationSession,
        endpoint: &dyn PreparedEndpoint,
        primary_model_name: &str,
        phase: CreditPhase,
        overrides: &Overrides,
    ) -> Result<MaterializedRequest> {
        let (conversation, current, turn_index) = session.current()?;
        let store = session.dataset.segments().as_ref();
        let has_native_raw_tokens = endpoint.descriptor().requires_raw_token_ids
            && token_ids_handle(current, store)?.is_some();
        if current.trace_hash_ids.is_none() && !has_native_raw_tokens {
            return EndpointRequestMaterializer.materialize_prepared(
                session,
                endpoint,
                primary_model_name,
                phase,
                overrides,
            );
        }
        let effective = EffectiveRequest {
            model: effective_model(current, primary_model_name, overrides)?,
            max_tokens: effective_max_tokens(current, overrides)?,
            streaming: effective_streaming(
                current,
                endpoint.config().streaming(),
                endpoint.descriptor().supports_streaming,
                overrides,
            )?,
        };
        let endpoint_path = endpoint.config().as_raw().path.clone().or_else(|| {
            if effective.streaming {
                endpoint.descriptor().streaming_path
            } else {
                None
            }
            .or(endpoint.descriptor().endpoint_path)
            .map(str::to_string)
        });
        let mut headers = endpoint.headers().clone();
        headers.extend(raw_string_map(
            store,
            current.extra_headers,
            "extra_headers",
        )?);
        Ok(MaterializedRequest {
            body: RequestBody::wire(Bytes::new()),
            headers,
            parameters: raw_string_map(store, current.request_parameters, "request_parameters")?,
            endpoint: current.endpoint.clone(),
            endpoint_path,
            model: effective.model,
            max_tokens: effective.max_tokens,
            streaming: effective.streaming,
            input_tokens: session.input_tokens(store)?,
            raw_token_ids: if endpoint.descriptor().requires_raw_token_ids {
                token_ids_handle(current, store)?
            } else {
                None
            },
            audio_duration_seconds: current.audio_duration_seconds,
            // The dispatched body is empty here, so only a first turn's authored
            // content is describable; later turns parse.
            image_count: raw_body_handle(current, store)
                .ok()
                .flatten()
                .is_none()
                .then(|| trace_identity_image_count(current, turn_index))
                .flatten(),
            accuracy: conversation.accuracy.clone(),
            turn_index,
            is_final_turn: turn_index + 1 == conversation.turns.len(),
            extracted: None,
        })
    }
}

struct EffectiveRequest {
    model: String,
    max_tokens: Option<u32>,
    streaming: bool,
}

/// Read effective model/max-tokens/streaming for one dispatch without touching
/// the plan, and report the `stream` literal the plan must be corrected to when
/// the endpoint cannot stream (`None` when no correction applies).
///
/// This reads the plan as it *would* look once [`BodyPlan::merge_overrides`] has
/// folded `overrides` in — see [`merged_literal`] — because the plan literal, not
/// the override, is what the dispatched body carries. Reading before the merge
/// rather than after keeps an unmutated plan shareable, so a dispatch with no
/// override and no stream correction copies nothing.
fn effective_from_plan(
    plan: &BodyPlan,
    turn: &Turn,
    primary_model_name: &str,
    configured_streaming: bool,
    supports_streaming: bool,
    overrides: &Overrides,
) -> Result<(EffectiveRequest, Option<bool>)> {
    let model = match merged_literal(plan, overrides, "model") {
        Some(Value::String(model)) => model.clone(),
        Some(_) => {
            return Err(DatasetError::Validation(
                "effective request model must be a string".into(),
            ));
        }
        None => effective_model(turn, primary_model_name, overrides)?,
    };
    let mut max_tokens = effective_max_tokens(turn, overrides)?;
    for field in ["max_tokens", "max_completion_tokens", "max_output_tokens"] {
        if let Some(value) = merged_literal(plan, overrides, field) {
            max_tokens = Some(positive_u32(value, field)?);
        }
    }
    let requested_streaming = match merged_literal(plan, overrides, "stream") {
        Some(Value::Bool(streaming)) => *streaming,
        Some(_) => {
            return Err(DatasetError::Validation(
                "effective request stream must be boolean".into(),
            ));
        }
        None => effective_streaming(turn, configured_streaming, supports_streaming, overrides)?,
    };
    let streaming = requested_streaming && supports_streaming;
    let stream_fix = (requested_streaming != streaming).then_some(streaming);
    Ok((
        EffectiveRequest {
            model,
            max_tokens,
            streaming,
        },
        stream_fix,
    ))
}

/// What [`BodyPlan::literal_field`] would return *after*
/// [`BodyPlan::merge_overrides`] folded `overrides` into the plan.
///
/// `merge_overrides` writes every override name as a top-level literal and
/// touches no other field, so the post-merge literal for a name is the override
/// when the override set carries it and the plan's own literal otherwise. On a
/// non-`Fields` plan the merge is a no-op and `literal_field` is always `None`,
/// so the override is invisible here exactly as it is post-merge — a raw or
/// prebuilt body applies its overrides as a spliced tail instead. The match is
/// exhaustive so a new variant has to restate its answer rather than inherit a
/// wrong one.
fn merged_literal<'a>(
    plan: &'a BodyPlan,
    overrides: &'a Overrides,
    name: &str,
) -> Option<&'a Value> {
    match plan {
        BodyPlan::Fields(_) => overrides
            .fields()
            .get(name)
            .or_else(|| plan.literal_field(name)),
        BodyPlan::Raw(_) | BodyPlan::Prebuilt(_) => None,
    }
}

/// Fold this dispatch's overrides and stream correction into the plan, in the
/// order [`effective_from_plan`] read them: overrides first (in place for a name
/// the plan already declares, appended otherwise), then the stream correction
/// overwriting whatever `stream` ended up as.
///
/// A no-op exactly when `overrides` is empty and `stream_fix` is `None`, which
/// is what lets a shared plan skip its copy on that dispatch.
fn apply_dispatch_mutations(plan: &mut BodyPlan, overrides: &Overrides, stream_fix: Option<bool>) {
    plan.merge_overrides(overrides);
    if let Some(streaming) = stream_fix {
        plan.set_literal("stream", Value::Bool(streaming));
    }
}

fn effective_model(turn: &Turn, primary_model_name: &str, overrides: &Overrides) -> Result<String> {
    match overrides.fields().get("model") {
        Some(Value::String(model)) => Ok(model.clone()),
        Some(_) => Err(DatasetError::Validation(
            "request override model must be a string".into(),
        )),
        None => Ok(turn
            .model
            .as_ref()
            .map(|model| model.as_str().to_string())
            .unwrap_or_else(|| primary_model_name.to_string())),
    }
}

fn effective_max_tokens(turn: &Turn, overrides: &Overrides) -> Result<Option<u32>> {
    let mut effective = turn.max_tokens;
    for field in ["max_tokens", "max_completion_tokens", "max_output_tokens"] {
        let Some(value) = overrides.fields().get(field) else {
            continue;
        };
        effective = Some(positive_u32(value, field)?);
    }
    Ok(effective)
}

fn positive_u32(value: &Value, field: &str) -> Result<u32> {
    value
        .as_u64()
        .and_then(|value| u32::try_from(value).ok())
        .filter(|value| *value > 0)
        .ok_or_else(|| {
            DatasetError::Validation(format!("effective request {field} must be a positive u32"))
        })
}

fn effective_streaming(
    turn: &Turn,
    configured_streaming: bool,
    supports_streaming: bool,
    overrides: &Overrides,
) -> Result<bool> {
    let requested = match overrides.fields().get("stream") {
        Some(Value::Bool(streaming)) => *streaming,
        Some(_) => {
            return Err(DatasetError::Validation(
                "request override stream must be boolean".into(),
            ));
        }
        None => turn.streaming.unwrap_or(configured_streaming),
    };
    Ok(requested && supports_streaming)
}

#[derive(Debug, Clone, PartialEq)]
struct CapturedReply {
    after_turn: usize,
    turn: EndpointTurn,
    tokens: u64,
}

/// Per-runtime-session state over one immutable, reusable dataset conversation.
///
/// The session keeps only small dynamic assistant turns. Authored request bytes
/// remain in the dataset's shared segment store and are resolved on dispatch.
#[derive(Clone)]
pub struct ConversationSession {
    dataset: Arc<Dataset>,
    conversation_id: SessionId,
    context_mode: ConversationContextMode,
    current_turn: Option<usize>,
    replies: Vec<CapturedReply>,
    /// Running number of wire image parts contributed by the replies captured so
    /// far, or `None` once any reply's contribution could not be established.
    /// Only the `*WithoutResponses` modes splice replies, so only they read it.
    reply_images: Option<u32>,
}

impl std::fmt::Debug for ConversationSession {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ConversationSession")
            .field("conversation_id", &self.conversation_id)
            .field("context_mode", &self.context_mode)
            .field("current_turn", &self.current_turn)
            .field("captured_replies", &self.replies.len())
            .finish()
    }
}

impl ConversationSession {
    /// Bind a new runtime session to one authored conversation.
    pub fn new(dataset: Arc<Dataset>, conversation_id: SessionId) -> Result<Self> {
        let conversation = dataset.get(&conversation_id)?;
        let context_mode = dataset.context_mode(conversation);
        Ok(Self {
            dataset,
            conversation_id,
            context_mode,
            current_turn: None,
            replies: Vec::new(),
            reply_images: Some(0),
        })
    }

    /// Authored conversation identifier.
    pub fn conversation_id(&self) -> &SessionId {
        &self.conversation_id
    }

    /// Effective conversation context mode.
    pub const fn context_mode(&self) -> ConversationContextMode {
        self.context_mode
    }

    /// Authored endpoint override on the current turn, if any.
    pub fn endpoint_override(&self) -> Result<Option<&str>> {
        let (_, turn, _) = self.current()?;
        Ok(turn.endpoint.as_deref())
    }

    /// Advance sequentially to the requested authored turn.
    pub fn advance_to(&mut self, turn_index: usize) -> Result<&Turn> {
        let conversation = self.dataset.get(&self.conversation_id)?;
        let expected = self.current_turn.map_or(0, |current| current + 1);
        if turn_index != expected {
            return Err(DatasetError::Validation(format!(
                "conversation {:?} expected turn {expected}, got {turn_index}",
                self.conversation_id.as_str()
            )));
        }
        let turn = conversation.turns.get(turn_index).ok_or_else(|| {
            DatasetError::Validation(format!(
                "turn {turn_index} is out of range for conversation {:?} with {} turns",
                self.conversation_id.as_str(),
                conversation.turns.len()
            ))
        })?;
        self.current_turn = Some(turn_index);
        Ok(turn)
    }

    /// Jump-resume to an arbitrary authored turn, reconstructing the recorded
    /// 0..=`turn_index` context directly from the dataset.
    ///
    /// Unlike [`advance_to`](Self::advance_to), which enforces strictly sequential
    /// `current + 1` stepping, this seeks straight to `turn_index` so a lane can
    /// resume at a runtime-determined frontier without replaying every prior turn's
    /// dispatch. It is only faithful for context modes whose materialized context is
    /// self-contained in the recorded conversation DTO
    /// ([`MessageArrayWithResponses`](ConversationContextMode::MessageArrayWithResponses)
    /// and [`DeltasWithResponses`](ConversationContextMode::DeltasWithResponses),
    /// where [`endpoint_turns`](Self::endpoint_turns) reads the recorded turns
    /// directly). Modes that splice live captured replies
    /// ([`should_capture_response`](Self::should_capture_response) is `true`) cannot
    /// reconstruct a non-zero turn's context without those replies, so a jump past
    /// turn 0 fails closed. Seeking to turn 0 is always permitted and is identical to
    /// `advance_to(0)`.
    pub fn seek_to(&mut self, turn_index: usize) -> Result<&Turn> {
        let conversation = self.dataset.get(&self.conversation_id)?;
        let turn = conversation.turns.get(turn_index).ok_or_else(|| {
            DatasetError::Validation(format!(
                "turn {turn_index} is out of range for conversation {:?} with {} turns",
                self.conversation_id.as_str(),
                conversation.turns.len()
            ))
        })?;
        if turn_index > 0 && self.should_capture_response() {
            return Err(DatasetError::Validation(format!(
                "context mode {:?} reconstructs context from live captured replies; \
                 jump-resume to turn {turn_index} for conversation {:?} is unsupported",
                self.context_mode,
                self.conversation_id.as_str()
            )));
        }
        self.current_turn = Some(turn_index);
        Ok(turn)
    }

    /// Whether a successful response must be retained for later context.
    pub const fn should_capture_response(&self) -> bool {
        matches!(
            self.context_mode,
            ConversationContextMode::DeltasWithoutResponses
                | ConversationContextMode::MessageArrayWithoutResponses
        )
    }

    /// Store one endpoint-normalized assistant turn after the current request.
    /// `tokens` should be the server's authoritative completion count when
    /// available, so later request accounting needs no hot-path tokenization.
    ///
    /// `images` is the number of wire image parts this reply adds to later turns'
    /// bodies (see [`reply_image_count`]); `None` means it could not be
    /// established, which makes every later turn's count unknown so dispatch
    /// falls back to parsing.
    ///
    /// A reply the caller already lowered (`turn.lowered` is set) is retained as
    /// those wires alone: every later turn splices them verbatim, so the reply's
    /// composed media would only be deep-cloned once per later dispatch and then
    /// discarded by the formatter. Dropping it here bounds a session's retained
    /// reply state to the bytes it actually sends.
    pub fn capture_response(
        &mut self,
        mut turn: EndpointTurn,
        tokens: u64,
        images: Option<u32>,
    ) -> Result<()> {
        if !self.should_capture_response() {
            return Err(DatasetError::Validation(format!(
                "context mode {:?} does not accept live assistant responses",
                self.context_mode
            )));
        }
        let after_turn = self.current_turn.ok_or_else(|| {
            DatasetError::Validation("cannot capture a response before advancing a turn".into())
        })?;
        if self
            .replies
            .last()
            .is_some_and(|reply| reply.after_turn >= after_turn)
        {
            return Err(DatasetError::Validation(format!(
                "conversation {:?} already captured a response for turn {after_turn}",
                self.conversation_id.as_str()
            )));
        }
        self.reply_images = self
            .reply_images
            .zip(images)
            .and_then(|(total, added)| total.checked_add(added));
        // Dropping the composed content is sound only because nothing downstream
        // can read it back, which rests on two invariants held elsewhere. Both
        // are load-bearing and neither is enforced by a type:
        //
        // 1. No lowerable dialect's formatter inspects a reply's content. Every
        //    dialect `ShapeLowerer` can lower splices `lowered` verbatim
        //    (`rendered_turn_messages`, `endpoints.rs`), and every dialect that
        //    does read `texts`/`role`/`raw_messages` directly either rejects
        //    multiple turns or selects `first()`/`last()` — neither of which a
        //    captured reply can ever be. A new lowerable id whose formatter
        //    inspects content makes this silent data loss.
        // 2. A per-turn endpoint override cannot route this reply to some other
        //    dialect: the only `PreparedTurnEndpointResolver` registers just the
        //    default endpoint's id and aliases (`multiturn.rs`,
        //    `PreparedEndpointTableResolver::single`) and errors on any other
        //    name. A second resolver that registered other profiles by name would
        //    break this the same way.
        if turn.lowered.is_some() {
            turn.texts.clear();
            turn.images.clear();
            turn.audios.clear();
            turn.videos.clear();
            turn.raw_messages = None;
            turn.role = None;
        }
        self.replies.push(CapturedReply {
            after_turn,
            turn,
            tokens,
        });
        Ok(())
    }

    /// Bind this dispatch's captured replies to the splice positions a cached
    /// continuation plan reserved for them, answering whether the cached plan
    /// describes this dispatch at all.
    ///
    /// The plan fixes every field but the message array — model, generation cap,
    /// tools, system block, extra-body tail — so reusing it is only sound when
    /// the replies contribute nothing beyond wires at the recorded positions.
    /// Each condition below is a way that could stop holding, and each answers
    /// `false` so the turn reformats live rather than dispatching a body that
    /// silently disagrees with its history:
    ///
    /// - a plan with no reserved positions paired with a session that captured
    ///   replies (or the reverse) is describing a different turn shape;
    /// - a reply captured after some turn other than the one its position was
    ///   measured for would land in the wrong place;
    /// - an unlowered reply has no wires to splice at all;
    /// - a reply carrying a model, cap, tools, system block, or extra body would
    ///   change a field the plan already fixed.
    ///
    /// `groups` is left empty on a `false` answer.
    fn reply_wire_groups<'session>(
        &'session self,
        cached: &CachedTurnPlan,
        groups: &mut SmallVec<[(u32, &'session [Bytes]); 16]>,
    ) -> bool {
        let Some(replies) = cached.replies.as_ref() else {
            return self.replies.is_empty();
        };
        if replies.positions.len() != self.replies.len() {
            return false;
        }
        for (index, (position, reply)) in replies.positions.iter().zip(&self.replies).enumerate() {
            match reply.turn.lowered.as_deref() {
                Some(wires)
                    if reply.after_turn == index && reply_splices_only_wires(&reply.turn) =>
                {
                    groups.push((*position, wires));
                }
                _ => {
                    groups.clear();
                    return false;
                }
            }
        }
        true
    }

    /// Exact number of wire image parts the current turn's body carries, or
    /// `None` when it is not established and dispatch must parse the body.
    ///
    /// This is the authored-dataset count the dataset precomputed for this
    /// `(conversation, turn)` plus, for the modes that splice them, the replies
    /// captured so far. Both halves are absent by default — an un-precomputed
    /// dataset yields `None` and the parse fallback, which is always correct.
    fn known_image_count(&self, turn_index: usize, overrides: &Overrides) -> Option<u32> {
        if overrides_replace_items(overrides) {
            return None;
        }
        let authored = self
            .dataset
            .cached_image_count(&self.conversation_id, turn_index)?;
        if !self.should_capture_response() {
            return Some(authored);
        }
        self.reply_images?.checked_add(authored)
    }

    /// Materialize the current request through an injected implementation.
    pub fn materialize(
        &self,
        materializer: &dyn RequestMaterializer,
        endpoint: &dyn Endpoint,
        model_endpoint: &ModelEndpoint,
        phase: CreditPhase,
        overrides: &Overrides,
    ) -> Result<MaterializedRequest> {
        materializer.materialize(self, endpoint, model_endpoint, phase, overrides)
    }

    /// Materialize the current request through a prepared endpoint binding.
    pub fn materialize_prepared(
        &self,
        materializer: &dyn RequestMaterializer,
        endpoint: &dyn PreparedEndpoint,
        primary_model_name: &str,
        phase: CreditPhase,
        overrides: &Overrides,
    ) -> Result<MaterializedRequest> {
        materializer.materialize_prepared(self, endpoint, primary_model_name, phase, overrides)
    }

    fn current(&self) -> Result<(&Conversation, &Turn, usize)> {
        let conversation = self.dataset.get(&self.conversation_id)?;
        let turn_index = self.current_turn.ok_or_else(|| {
            DatasetError::Validation("cannot materialize before advancing a turn".into())
        })?;
        let turn = conversation
            .turns
            .get(turn_index)
            .expect("advance_to validates the current turn");
        Ok((conversation, turn, turn_index))
    }

    /// Assemble the ordered endpoint turns this dispatch's body carries.
    ///
    /// `spliced` selects [`resolve_turn_spliced`] for the authored turns, which
    /// the caller sets only when the bound dialect renders lowered wires and the
    /// phase is not warmup.
    fn endpoint_turns(&self, store: &dyn SegmentStore, spliced: bool) -> Result<Vec<EndpointTurn>> {
        let (conversation, _, current) = self.current()?;
        let resolve = |turn: &Turn| {
            if spliced {
                resolve_turn_spliced(store, turn)
            } else {
                resolve_turn(store, turn)
            }
        };
        match self.context_mode {
            ConversationContextMode::DeltasWithoutResponses => {
                let mut out = Vec::with_capacity(current + 1 + self.replies.len());
                for index in 0..=current {
                    out.push(resolve(&conversation.turns[index])?);
                    if let Some(reply) = self.replies.iter().find(|reply| reply.after_turn == index)
                    {
                        out.push(reply.turn.clone());
                    }
                }
                Ok(out)
            }
            ConversationContextMode::DeltasWithResponses => {
                conversation.turns[..=current].iter().map(resolve).collect()
            }
            ConversationContextMode::MessageArrayWithResponses => {
                Ok(vec![resolve(&conversation.turns[current])?])
            }
            ConversationContextMode::MessageArrayWithoutResponses => {
                self.merge_message_array_snapshots(store, conversation, current, spliced)
            }
        }
    }

    fn merge_message_array_snapshots(
        &self,
        store: &dyn SegmentStore,
        conversation: &Conversation,
        current: usize,
        spliced: bool,
    ) -> Result<Vec<EndpointTurn>> {
        let mut previous = Vec::<EndpointTurn>::new();
        let mut out = Vec::new();
        for index in 0..=current {
            let turn = &conversation.turns[index];
            let resolved = if spliced {
                resolve_turn_spliced(store, turn)?
            } else {
                resolve_turn(store, turn)?
            };
            let snapshot = split_snapshot(resolved);
            if !snapshot.starts_with(&previous) {
                return Err(DatasetError::Validation(format!(
                    "conversation {:?} turn {index} is not a prefix-extending message-array snapshot",
                    conversation.session_id.as_str()
                )));
            }
            out.extend(snapshot[previous.len()..].iter().cloned());
            if index < current {
                let reply = self
                    .replies
                    .iter()
                    .find(|reply| reply.after_turn == index)
                    .ok_or_else(|| {
                        DatasetError::Validation(format!(
                            "conversation {:?} is missing the live response after snapshot turn {index}",
                            conversation.session_id.as_str()
                        ))
                    })?;
                out.push(reply.turn.clone());
            }
            previous = snapshot;
        }
        Ok(out)
    }

    fn input_tokens(&self, store: &dyn SegmentStore) -> Result<Option<u64>> {
        let (conversation, current_turn, current) = self.current()?;
        if raw_body_handle(current_turn, store)?.is_some()
            || token_ids_handle(current_turn, store)?.is_some()
        {
            return Ok(current_turn.input_tokens);
        }
        let mut count = match self.context_mode {
            ConversationContextMode::DeltasWithoutResponses
            | ConversationContextMode::DeltasWithResponses => conversation.turns[..=current]
                .iter()
                .try_fold(0_u64, |count, turn| {
                    checked_add(
                        count,
                        turn.input_tokens.ok_or_else(|| {
                            DatasetError::Validation(
                                "delta context turn is missing a composed input token count".into(),
                            )
                        })?,
                    )
                })?,
            ConversationContextMode::MessageArrayWithResponses
            | ConversationContextMode::MessageArrayWithoutResponses => {
                current_turn.input_tokens.ok_or_else(|| {
                    DatasetError::Validation(
                        "message-array turn is missing a composed input token count".into(),
                    )
                })?
            }
        };
        if matches!(
            self.context_mode,
            ConversationContextMode::DeltasWithoutResponses
                | ConversationContextMode::MessageArrayWithoutResponses
        ) {
            for reply in self
                .replies
                .iter()
                .filter(|reply| reply.after_turn < current)
            {
                count = checked_add(count, reply.tokens)?;
            }
        }
        if let Some(tool_tokens) = selected_tool_tokens(conversation, current, self.context_mode) {
            count = checked_add(count, tool_tokens)?;
        }
        for handle in [conversation.system, conversation.user_context]
            .into_iter()
            .flatten()
        {
            if let Some(tokens) = store.get(handle)?.token_count() {
                count = checked_add(count, tokens as u64)?;
            }
        }
        Ok(Some(count))
    }
}

fn selected_tool_tokens(
    conversation: &Conversation,
    current: usize,
    mode: ConversationContextMode,
) -> Option<u64> {
    let turns = match mode {
        ConversationContextMode::MessageArrayWithResponses
        | ConversationContextMode::MessageArrayWithoutResponses => {
            &conversation.turns[current..=current]
        }
        ConversationContextMode::DeltasWithResponses
        | ConversationContextMode::DeltasWithoutResponses => &conversation.turns[..=current],
    };
    turns
        .iter()
        .rev()
        .find(|turn| turn.tools.is_some())
        .map(|turn| turn.tool_tokens)
}

fn checked_add(left: u64, right: u64) -> Result<u64> {
    left.checked_add(right)
        .ok_or_else(|| DatasetError::Validation("request input token count overflowed u64".into()))
}

pub(crate) fn split_snapshot(mut turn: EndpointTurn) -> Vec<EndpointTurn> {
    match turn.raw_messages.take() {
        Some(messages) if !messages.is_empty() => messages
            .into_iter()
            .map(|message| EndpointTurn {
                raw_messages: Some(vec![message]),
                ..EndpointTurn::default()
            })
            .collect(),
        _ => vec![turn],
    }
}

#[cfg(test)]
thread_local! {
    /// Dispatches on this thread that materialized through a cached continuation
    /// plan's wire splice.
    ///
    /// Declining the cache is always *safe* — the turn reformats live and the
    /// body is identical — which is exactly why the byte-identity matrix cannot
    /// see it: a build that declined every splice would compare a live body
    /// against a live body and pass. Without an observable count the entire
    /// measured saving could regress to zero with every gate green.
    ///
    /// Thread-local rather than a global counter so concurrently running tests
    /// cannot perturb each other's reading.
    pub(crate) static SPLICED_DISPATCHES: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };

    /// Dispatches on this thread that materialized through a precomputed plan
    /// instead of formatting the turn live.
    ///
    /// The same blind spot as [`SPLICED_DISPATCHES`], one level up and wider: a
    /// hit on turn 0, on a `*WithResponses` mode, or on a dialect that splices
    /// nothing carries no splice, so the splice counter cannot see it, and
    /// declining the cache reformats to identical bytes. The cache is what moves
    /// `format_payload` out of the timed loop for the input-array dialects —
    /// embeddings, rankings, image retrieval, where a single body inlines a whole
    /// image batch as data URLs — so a build that declined every hit would give
    /// the entire saving back with every byte assertion still passing.
    pub(crate) static CACHE_HITS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };

    /// Composed content handles resolved on this thread during dispatch.
    pub(crate) static COMPOSED_CONTENT_RESOLUTIONS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// Whether a dispatch override rewrites the very field a cached plan's captured
/// replies splice into.
///
/// Such an override supersedes the entire array — the authored turns and the
/// replies alike — exactly as it does on the live path, where `merge_overrides`
/// runs after the array is fully assembled. So the splice is dropped rather than
/// applied to a field the merge has already turned into a literal.
///
/// Compared against the plan's own field name rather than the message-array
/// names the dialects use, since an override naming `input` must not cancel a
/// splice into a `messages` field, or vice versa: that would dispatch a
/// continuation body silently missing its history.
fn overrides_replace_field(plan: &BodyPlan, field: usize, overrides: &Overrides) -> bool {
    match plan {
        BodyPlan::Fields(program) if !overrides.is_empty() => program
            .fields()
            .get(field)
            .is_some_and(|(name, _)| overrides.fields().contains_key(name.as_ref())),
        _ => false,
    }
}

/// Whether a captured reply contributes nothing to the body but its own message
/// wires — the precondition for splicing it into a cached plan, whose other
/// fields were fixed without it.
///
/// The media, role, and raw-message fields are already cleared by
/// [`ConversationSession::capture_response`] for a lowered reply and are checked
/// anyway: this is the one place the assumption is load-bearing, and a reply
/// reaching here by some future path must not depend on that clearing.
fn reply_splices_only_wires(reply: &EndpointTurn) -> bool {
    reply.model.is_none()
        && reply.role.is_none()
        && reply.max_tokens.is_none()
        && reply.raw_tools.is_none()
        && reply.raw_system.is_none()
        && reply.extra_body.is_none()
        && reply.raw_payload.is_none()
        && reply.raw_token_ids.is_none()
        && reply.raw_messages.is_none()
        && reply.texts.is_empty()
        && reply.images.is_empty()
        && reply.audios.is_empty()
        && reply.videos.is_empty()
}

/// Whether this dispatch renders its message array by splicing lowered wires,
/// making a turn's composed media unreachable from the formatter.
///
/// The dialect half is the endpoint's own
/// [`PreparedEndpoint::splices_lowered_wires`] answer rather than a list of ids
/// kept here, so a new dialect declares its own capability where it is defined
/// instead of being silently omitted from an enumeration in another module.
/// `lowerable_dialects_declare_that_they_splice_lowered_wires` pins that answer
/// against [`ShapeLowerer`], the predicate
/// [`Dataset::lower_messages_for_endpoint`](crate::dataset::Dataset::lower_messages_for_endpoint)
/// used to produce those wires.
///
/// Warmup is excluded: it re-renders the first turn from its media so the system
/// prompt can be folded into that message.
fn splices_lowered_wires(endpoint: &dyn PreparedEndpoint, phase: CreditPhase) -> bool {
    phase != CreditPhase::Warmup && endpoint.splices_lowered_wires()
}

/// Resolve one authored dataset turn into the endpoint-facing turn, including
/// its composed media content.
pub(crate) fn resolve_turn(store: &dyn SegmentStore, turn: &Turn) -> Result<EndpointTurn> {
    resolve_turn_inner(store, turn, false)
}

/// Resolve one authored dataset turn for a dialect that splices lowered message
/// wires, skipping the composed media a spliced turn cannot reach.
///
/// A turn whose content was lowered at load carries its rendered messages in
/// [`EndpointTurn::lowered`], and `rendered_turn_messages` splices those bytes
/// and never looks at `texts`/`images`/`audios`/`videos`. Re-resolving the
/// composed content therefore copies every prompt string out of the segment
/// store — the whole accumulated history, on every dispatch — to build media
/// the formatter discards.
///
/// Two conditions make the skip exact, and the caller owns both (see
/// [`splices_lowered_wires`]): the dialect must be one that renders through
/// `rendered_turn_messages`, and the phase must not be warmup, whose
/// `render_first` re-renders the first turn from its media so the system prompt
/// can be folded into it. A turn that was never lowered keeps its content
/// regardless, since there is no wire to splice.
pub(crate) fn resolve_turn_spliced(store: &dyn SegmentStore, turn: &Turn) -> Result<EndpointTurn> {
    resolve_turn_inner(store, turn, true)
}

fn resolve_turn_inner(
    store: &dyn SegmentStore,
    turn: &Turn,
    skip_lowered_content: bool,
) -> Result<EndpointTurn> {
    // Lowered content uses stored message wires; authored message arrays continue
    // through raw_messages. Validation guarantees the representations do not coexist.
    let message_handles = body_message_handles(turn, store)?;
    let lowered_content = !turn.content.is_empty() && !message_handles.is_empty();
    let mut raw_messages = Vec::new();
    let mut has_raw_messages = false;
    let mut lowered: Option<SmallVec<[Bytes; 1]>> = None;
    if lowered_content {
        let mut wires: SmallVec<[Bytes; 1]> = SmallVec::with_capacity(message_handles.len());
        for handle in &message_handles {
            wires.push(message_wire(store, *handle)?);
        }
        lowered = Some(wires);
    } else {
        for handle in &message_handles {
            raw_messages.push(raw_value(store, *handle, "message")?);
        }
    }
    if let Some(handle) = turn.raw_messages {
        has_raw_messages = true;
        match raw_value(store, handle, "raw_messages")? {
            Value::Array(messages) => raw_messages.extend(messages),
            _ => {
                return Err(DatasetError::Validation(format!(
                    "raw_messages handle {handle} must contain a JSON array"
                )));
            }
        }
    }

    let mut resolved = EndpointTurn {
        model: turn.model.as_ref().map(|model| model.as_str().to_string()),
        role: turn.role.as_ref().map(|role| role.as_str().to_string()),
        max_tokens: turn.max_tokens,
        raw_messages: (has_raw_messages || !raw_messages.is_empty()).then_some(raw_messages),
        raw_tools: raw_array(store, turn.tools, "tools")?,
        raw_system: raw_array(store, turn.raw_system, "raw_system")?,
        extra_body: raw_object(store, turn.extra_body, "extra_body")?,
        raw_token_ids: raw_token_ids(store, token_ids_handle(turn, store)?)?,
        lowered,
        ..EndpointTurn::default()
    };
    if skip_lowered_content && resolved.lowered.is_some() {
        // `role` shares the media's fate: the only reader is
        // `render_turn_message`, which a spliced turn never reaches, and the
        // role is already baked into the lowered wire.
        resolved.role = None;
        return Ok(resolved);
    }
    for group in &turn.content {
        let contents = group
            .handles
            .iter()
            .map(|handle| {
                #[cfg(test)]
                COMPOSED_CONTENT_RESOLUTIONS.with(|count| count.set(count.get() + 1));
                content_string(store, *handle, group.kind)
            })
            .collect::<Result<Vec<_>>>()?;
        let media = Media {
            name: group.name.clone(),
            contents,
            uuids: group.uuids.to_vec(),
        };
        match group.kind {
            MediaKind::Text => resolved.texts.push(media),
            MediaKind::Image => resolved.images.push(media),
            MediaKind::Audio => resolved.audios.push(media),
            MediaKind::Video => resolved.videos.push(media),
        }
    }
    Ok(resolved)
}

/// Wire image count for a trace-identity turn, derived from the composed turn
/// content without re-parsing the serialized body.
///
/// The trace-hash and token-native materializers dispatch an empty body, so this
/// only ever describes the authored content of a first turn; later turns return
/// `None`. It is deliberately separate from the precomputed dataset counts
/// ([`Dataset::cached_image_count`]), which describe a real serialized wire body.
fn trace_identity_image_count(current: &Turn, turn_index: usize) -> Option<u32> {
    if turn_index != 0 {
        return None;
    }
    composed_image_count(current)
}

/// Exact number of wire image parts one authored dataset turn contributes to the
/// `messages`/`input` array a dispatch body carries, or `None` when it cannot be
/// established without parsing the dispatched body.
///
/// Called only from dataset precompute, never on the timed path.
///
/// `image_count` is incremented in exactly one place — the image arm of
/// `walk_content_part` in `endpoints::extraction`, for a part inside an item's
/// `content` array whose `type` names an image part for the dialect. So a turn's
/// contribution is zero unless one of these reaches that arm:
///
/// - a non-text content group (`images`/`audios`/`videos` on the resolved turn),
///   which the endpoint renders into media content parts;
/// - an authored `raw_messages` item spliced verbatim, whose `content` may be an
///   array of arbitrary parts.
///
/// Text content groups render text parts, and `raw_tools`/`raw_system` land in
/// `tools`/`system`, which `extract_inputs` does not walk. When neither applies
/// the count is provably zero and this returns `Some(0)` without building
/// anything. Otherwise it builds this one turn's body through the same endpoint
/// and counts it with the same extractor dispatch would use, so the answer is
/// exact by construction rather than by enumerating render paths.
///
/// This describes ONE turn in isolation. Composing per-turn counts into a body's
/// count is only valid for a dialect that concatenates every turn's items; see
/// [`Dataset::precompute_image_counts`], which owns that decision.
pub(crate) fn dataset_turn_image_count(
    store: &dyn SegmentStore,
    endpoint: &dyn PreparedEndpoint,
    primary_model_name: &str,
    turn: &Turn,
) -> Option<u32> {
    // A raw payload is dispatched verbatim and has no structured content groups.
    if raw_body_handle(turn, store).ok().flatten().is_some() {
        return None;
    }
    // Establish the ordinary text-only answer without resolving the turn. This
    // runs over every turn of every conversation, once per conversation source,
    // and `resolve_turn` materializes every text handle into a fresh `String`.
    if authored_turn_is_text_only(turn) {
        return Some(0);
    }
    let resolved = resolve_turn(store, turn).ok()?;
    // Only the LAST turn's `extra_body` is merged into a message-array body
    // (`merge_extra(&mut payload, last.extra_body…)`), and `split_snapshot` drops
    // it entirely, so whether an item-bearing `extra_body` reaches the wire
    // depends on position and mode rather than on the turn alone. Fail closed.
    if resolved
        .extra_body
        .as_ref()
        .is_some_and(|extra| extra.contains_key("messages") || extra.contains_key("input"))
    {
        return None;
    }
    if is_provably_image_free(&resolved) {
        return Some(0);
    }
    let request = PreparedRequest::new(
        primary_model_name,
        std::slice::from_ref(&resolved),
        None,
        None,
        CreditPhase::Profiling,
        None,
        None,
        None,
    );
    let body = endpoint
        .format_payload(&request)
        .ok()?
        .materialize_standalone()
        .ok()?;
    let extracted = serde_json::from_slice::<Value>(&body)
        .ok()
        .map(|payload| endpoint.extract_payload_inputs(&payload));
    match extracted {
        // The endpoint's own extractor is the function dispatch would run on
        // this same body, and it reports whether the count it produced is this
        // dialect's authoritative answer. Take it whenever it is — INCLUDING an
        // exact zero. Conflating "the extractor said zero" with "the extractor
        // said nothing" is precisely what `composed_image_count` gets wrong: the
        // formatters drop empty content strings (`!content.is_empty()`) and
        // `handles.len()` does not, so a turn whose image contents are all empty
        // sends no image and must report none.
        Some(extracted) if extracted.owns_image_count => Some(extracted.image_count),
        // The extractor established nothing: the body is not JSON, or it is a
        // flat shape whose dialect carries images somewhere no wire walk can see
        // (image edit posts the turn's image as multipart form data). Fall back
        // to the media the turn composed, which is what the pre-existing
        // first-turn path reported. Sound only with nothing preformatted in play
        // — reaching here
        // means `is_provably_image_free` found media groups or array-content raw
        // messages, and the latter can hide an image the content groups do not
        // enumerate.
        _ => resolved
            .raw_messages
            .is_none()
            .then(|| composed_image_count(turn))
            .flatten(),
    }
}

/// Cheap authored-turn test establishing a zero without resolving the turn.
///
/// Sound because a turn whose content groups are all text, with no preformatted
/// message array and no per-turn extras, can only render text parts: its lowered
/// body wires are `resolve_turn`'s render of that same all-text content, and
/// nothing else reaches the counted roots. `content` must be non-empty — an empty
/// one means any `Message` body handles become `raw_messages` at resolve instead
/// of being this turn's own lowered text.
fn authored_turn_is_text_only(turn: &Turn) -> bool {
    !turn.content.is_empty()
        && turn
            .content
            .iter()
            .all(|group| group.kind == MediaKind::Text)
        && turn.raw_messages.is_none()
        && turn.extra_body.is_none()
}

/// Number of image content items one authored turn composed, independent of how
/// a dialect renders them.
fn composed_image_count(turn: &Turn) -> Option<u32> {
    let count: usize = turn
        .content
        .iter()
        .filter(|group| group.kind == MediaKind::Image)
        .map(|group| group.handles.len())
        .sum();
    u32::try_from(count).ok()
}

/// Exact number of wire image parts one captured assistant reply contributes, or
/// `None` when it cannot be established.
///
/// A reply is spliced into the same `messages`/`input` array as an authored turn
/// and the extractor counts image parts in every item regardless of role, so a
/// reply whose content is an array of parts must be inspected rather than assumed
/// empty. It is already a decoded `Value` tree, so the common assistant shape
/// (`content` is a string) resolves without touching the extractor at all.
pub(crate) fn reply_image_count(
    reply: &EndpointTurn,
    endpoint: &dyn PreparedEndpoint,
) -> Option<u32> {
    if !reply.images.is_empty() || !reply.audios.is_empty() || !reply.videos.is_empty() {
        return None;
    }
    // No preformatted items: the reply renders from its text media alone, which
    // can only produce text parts.
    let Some(messages) = reply
        .raw_messages
        .as_ref()
        .filter(|items| !items.is_empty())
    else {
        return Some(0);
    };
    // The Responses lowerer drops replay-unsafe output items, so a lowered wire
    // count below the item count means the dispatched array is a subset of what
    // is inspected here and counting the items would over-report.
    if reply
        .lowered
        .as_ref()
        .is_some_and(|lowered| lowered.len() != messages.len())
    {
        return None;
    }
    if messages
        .iter()
        .all(|item| !matches!(item.get("content"), Some(Value::Array(_))))
    {
        return Some(0);
    }
    // `walk_items_arrays` walks both `messages` and `input`, so this reaches the
    // items for every message-array dialect.
    let payload = Value::Object(Map::from_iter([(
        "messages".to_string(),
        Value::Array(messages.clone()),
    )]));
    // Same guard `dataset_turn_image_count` applies to the same extractor: take
    // the count only where the dialect reports it is the authoritative answer.
    // Without it a dialect whose extractor found nothing in this shape reports an
    // exact zero, and every later turn's count is silently short.
    let extracted = endpoint.extract_payload_inputs(&payload);
    extracted.owns_image_count.then_some(extracted.image_count)
}

/// Whether a resolved turn provably contributes no wire image part. See
/// [`dataset_turn_image_count`] for the enumeration this encodes.
fn is_provably_image_free(resolved: &EndpointTurn) -> bool {
    resolved.images.is_empty()
        && resolved.audios.is_empty()
        && resolved.videos.is_empty()
        && resolved
            .raw_messages
            .as_ref()
            .is_none_or(|items| items.iter().all(|item| !has_array_content(item)))
}

/// Whether a message-array item carries a content-part array, the only shape the
/// extractor can find an image part in.
fn has_array_content(item: &Value) -> bool {
    matches!(item.get("content"), Some(Value::Array(_)))
}

/// Whether a per-dispatch override set could rewrite the item array the image
/// count was established over.
fn overrides_replace_items(overrides: &Overrides) -> bool {
    !overrides.is_empty()
        && (overrides.fields().contains_key("messages") || overrides.fields().contains_key("input"))
}

fn raw_token_ids(store: &dyn SegmentStore, handle: Option<Handle>) -> Result<Option<Vec<u32>>> {
    let Some(handle) = handle else {
        return Ok(None);
    };
    match store.get(handle)? {
        Payload::TokenIds { token_ids } if !token_ids.is_empty() => Ok(Some(token_ids.to_vec())),
        payload => Err(DatasetError::PayloadKind {
            handle,
            expected: "non-empty token-ids",
            actual: payload.kind_name(),
        }),
    }
}

fn content_string(store: &dyn SegmentStore, handle: Handle, kind: MediaKind) -> Result<String> {
    let bytes = match (kind, store.get(handle)?) {
        (MediaKind::Text, Payload::Text { bytes, .. }) => bytes,
        (expected, Payload::Media { kind, bytes }) if expected == *kind => bytes,
        (_, payload) => {
            return Err(DatasetError::PayloadKind {
                handle,
                expected: "matching text/media content",
                actual: payload.kind_name(),
            });
        }
    };
    std::str::from_utf8(bytes)
        .map(str::to_string)
        .map_err(|error| DatasetError::InvalidWire(format!("handle {handle}: {error}")))
}

pub(crate) fn resolve_prompt(
    store: &dyn SegmentStore,
    handle: Option<Handle>,
) -> Result<Option<String>> {
    let Some(handle) = handle else {
        return Ok(None);
    };
    match store.get(handle)? {
        Payload::Text { bytes, .. } => std::str::from_utf8(bytes)
            .map(|text| Some(text.to_string()))
            .map_err(|error| DatasetError::InvalidWire(format!("handle {handle}: {error}"))),
        Payload::Message { wire, .. } => {
            let value: Value = serde_json::from_slice(wire)?;
            value
                .get("content")
                .and_then(Value::as_str)
                .map(|text| Some(text.to_string()))
                .ok_or_else(|| {
                    DatasetError::InvalidWire(format!(
                        "prompt message handle {handle} has no string content"
                    ))
                })
        }
        payload => Err(DatasetError::PayloadKind {
            handle,
            expected: "text or message",
            actual: payload.kind_name(),
        }),
    }
}

fn raw_value(store: &dyn SegmentStore, handle: Handle, field: &str) -> Result<Value> {
    let wire = match store.get(handle)? {
        Payload::Raw { wire } | Payload::Message { wire, .. } => wire,
        payload => {
            return Err(DatasetError::PayloadKind {
                handle,
                expected: "raw or message",
                actual: payload.kind_name(),
            });
        }
    };
    serde_json::from_slice(wire)
        .map_err(|error| DatasetError::InvalidWire(format!("{field} handle {handle}: {error}")))
}

fn raw_array(
    store: &dyn SegmentStore,
    handle: Option<Handle>,
    field: &str,
) -> Result<Option<Vec<Value>>> {
    let Some(handle) = handle else {
        return Ok(None);
    };
    match raw_value(store, handle, field)? {
        Value::Array(values) => Ok(Some(values)),
        _ => Err(DatasetError::InvalidWire(format!(
            "{field} handle {handle} must contain a JSON array"
        ))),
    }
}

fn raw_object(
    store: &dyn SegmentStore,
    handle: Option<Handle>,
    field: &str,
) -> Result<Option<Map<String, Value>>> {
    let Some(handle) = handle else {
        return Ok(None);
    };
    match raw_value(store, handle, field)? {
        Value::Object(values) => Ok(Some(values)),
        _ => Err(DatasetError::InvalidWire(format!(
            "{field} handle {handle} must contain a JSON object"
        ))),
    }
}

fn raw_string_map(
    store: &dyn SegmentStore,
    handle: Option<Handle>,
    field: &str,
) -> Result<BTreeMap<String, String>> {
    let Some(values) = raw_object(store, handle, field)? else {
        return Ok(BTreeMap::new());
    };
    values
        .into_iter()
        .map(|(key, value)| match value {
            Value::String(value) => Ok((key, value)),
            _ => Err(DatasetError::Validation(format!(
                "{field}.{key} must be a string"
            ))),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use crate::endpoints::{
        ChatEndpoint, EndpointConfig, EndpointId, EndpointRegistry, EndpointType, RawEndpointConfig,
    };
    use bytes::Bytes;
    use smallvec::smallvec;

    use super::*;
    use crate::dataset::model::{ContentGroup, CorrelationId, ModelId};
    use crate::dataset::segment::{Role, SegmentPool};

    fn model_endpoint() -> ModelEndpoint {
        ModelEndpoint {
            primary_model_name: "default-model".into(),
            endpoint: EndpointConfig {
                endpoint_type: EndpointType::Chat,
                streaming: true,
                use_server_token_count: true,
                ..EndpointConfig::default()
            },
        }
    }

    #[test]
    fn endpoint_registry_rejects_normalized_duplicates() {
        let mut resolver = BuiltinEndpointResolver::default();
        let error = resolver.register("CHAT", ChatEndpoint).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("duplicate endpoint registration")
        );
    }

    #[test]
    fn endpoint_registry_contains_every_native_dialect() {
        let resolver = BuiltinEndpointResolver::default();
        for name in [
            "chat",
            "chat_completions",
            "completions",
            "responses",
            "messages",
            "embeddings",
            "chat_embeddings",
            "nim_embeddings",
            "cohere_rankings",
            "hf_tei_rankings",
            "nim_rankings",
            "huggingface_generate",
            "image_generation",
            "image_edit",
            "video_generation",
            "image_retrieval",
            "solido_rag",
            "raw",
            "template",
        ] {
            assert!(resolver.resolve(Some(name)).is_ok(), "missing {name}");
        }
        assert_eq!(
            resolver
                .resolve_type(EndpointType::Messages)
                .unwrap()
                .descriptor()
                .legacy_type()
                .expect("endpoint type"),
            EndpointType::Messages
        );
    }

    fn message(pool: &mut SegmentPool, parent: Option<Handle>, role: &str, text: &str) -> Handle {
        pool.intern_message(
            parent,
            role,
            Bytes::from(
                serde_json::to_vec(&serde_json::json!({
                    "role": role,
                    "content": text
                }))
                .unwrap(),
            ),
            vec![text.len() as u32],
        )
        .unwrap()
    }

    fn dataset(mode: ConversationContextMode, turns: Vec<Turn>, pool: SegmentPool) -> Arc<Dataset> {
        let mut conversation = Conversation::new("session");
        conversation.turns = turns;
        conversation.context_mode = Some(mode);
        conversation.accuracy = Some(AccuracyAssociation {
            correlation_id: CorrelationId::from("corr"),
            task: "task".into(),
        });
        Arc::new(
            Dataset::new(
                vec![conversation],
                Arc::new(pool.freeze()),
                "sequential",
                ConversationContextMode::DeltasWithoutResponses,
            )
            .unwrap(),
        )
    }

    #[test]
    fn raw_payload_is_byte_exact_and_explicit_overrides_are_tail_spliced() {
        let mut pool = SegmentPool::new();
        let wire = Bytes::from_static(b"{ \"messages\" : [ ], \"model\":\"authored\" }\n");
        let raw = pool.intern_raw(None, wire.clone()).unwrap();
        let data = dataset(
            ConversationContextMode::MessageArrayWithResponses,
            vec![Turn {
                body: Turn::dispatch_body(Some(raw), None, &[]),
                model: Some(ModelId::from("metadata-only")),
                input_tokens: Some(7),
                ..Turn::default()
            }],
            pool,
        );
        let mut session = ConversationSession::new(data, SessionId::from("session")).unwrap();
        session.advance_to(0).unwrap();
        let exact = session
            .materialize(
                &EndpointRequestMaterializer,
                &ChatEndpoint,
                &model_endpoint(),
                CreditPhase::Profiling,
                &Overrides::new(),
            )
            .unwrap();
        assert_eq!(exact.body.to_wire().unwrap(), wire);
        assert_eq!(exact.input_tokens, Some(7));

        let mut overrides = Overrides::new();
        overrides.set_stream(true);
        let spliced = session
            .materialize(
                &EndpointRequestMaterializer,
                &ChatEndpoint,
                &model_endpoint(),
                CreditPhase::Profiling,
                &overrides,
            )
            .unwrap();
        assert_eq!(
            spliced.body.to_wire().unwrap(),
            b"{ \"messages\" : [ ], \"model\":\"authored\" ,\"stream\":true}\n"[..]
        );
    }

    #[test]
    fn resolve_turn_preserves_explicit_empty_raw_messages() {
        let mut pool = SegmentPool::new();
        let raw = pool.intern_raw(None, Bytes::from_static(b"[]")).unwrap();
        let turn = Turn {
            raw_messages: Some(raw),
            ..Turn::default()
        };
        let store = pool.freeze();

        let resolved = resolve_turn(&store, &turn).unwrap();

        assert_eq!(resolved.raw_messages, Some(Vec::new()));
    }

    #[test]
    fn trace_hash_materializer_skips_wire_body_but_preserves_dispatch_metadata() {
        let mut pool = SegmentPool::new();
        let hashes = pool
            .intern_trace_hash_ids(vec![11_i64, 12].into_boxed_slice(), 128)
            .unwrap();
        let data = dataset(
            ConversationContextMode::DeltasWithResponses,
            vec![Turn {
                model: Some(ModelId::from("trace-model")),
                max_tokens: Some(9),
                streaming: Some(true),
                input_tokens: Some(17),
                trace_hash_ids: Some(hashes),
                ..Turn::default()
            }],
            pool,
        );
        let mut session = ConversationSession::new(data, SessionId::from("session")).unwrap();
        session.advance_to(0).unwrap();

        let request = session
            .materialize(
                &TraceHashAwareRequestMaterializer,
                &ChatEndpoint,
                &model_endpoint(),
                CreditPhase::Profiling,
                &Overrides::new(),
            )
            .unwrap();

        assert!(request.body.to_wire().unwrap().is_empty());
        assert_eq!(request.model, "trace-model");
        assert_eq!(request.max_tokens, Some(9));
        assert!(request.streaming);
        assert_eq!(request.input_tokens, Some(17));
        assert_eq!(
            request.endpoint_path.as_deref(),
            Some("/v1/chat/completions")
        );
    }

    #[test]
    fn seek_to_jump_resume_equals_sequential_advance_for_message_array_responses() {
        // A recorded MessageArrayWithResponses conversation carries the full context
        // for each turn inside that turn's own message array (turn k embeds turns
        // 0..=k). Jump-resuming to turn k with `seek_to` must therefore reconstruct
        // byte-identical context to sequentially advancing 0 -> ... -> k.
        let build_data = || {
            let mut pool = SegmentPool::new();
            let q0 = message(&mut pool, None, "user", "q0");
            let a0 = message(&mut pool, Some(q0), "assistant", "a0");
            let q1 = message(&mut pool, Some(a0), "user", "q1");
            dataset(
                ConversationContextMode::MessageArrayWithResponses,
                vec![
                    Turn {
                        body: smallvec![q0],
                        input_tokens: Some(2),
                        max_tokens: Some(4),
                        ..Turn::default()
                    },
                    Turn {
                        body: smallvec![q0, a0, q1],
                        input_tokens: Some(5),
                        max_tokens: Some(4),
                        ..Turn::default()
                    },
                ],
                pool,
            )
        };

        // Sequential oracle: advance 0 -> 1 (no captured live reply is needed for a
        // with-responses mode) and materialize turn 1.
        let mut seq = ConversationSession::new(build_data(), SessionId::from("session")).unwrap();
        seq.advance_to(0).unwrap();
        seq.advance_to(1).unwrap();
        let seq_turn1 = seq
            .materialize(
                &EndpointRequestMaterializer,
                &ChatEndpoint,
                &model_endpoint(),
                CreditPhase::Profiling,
                &Overrides::new(),
            )
            .unwrap();

        // Jump-resume: a fresh session seeks straight to turn 1.
        let mut jump = ConversationSession::new(build_data(), SessionId::from("session")).unwrap();
        jump.seek_to(1).unwrap();
        let jump_turn1 = jump
            .materialize(
                &EndpointRequestMaterializer,
                &ChatEndpoint,
                &model_endpoint(),
                CreditPhase::Profiling,
                &Overrides::new(),
            )
            .unwrap();

        assert_eq!(jump_turn1.turn_index, 1);
        assert_eq!(
            jump_turn1.body, seq_turn1.body,
            "jump-resume to turn 1 is byte-identical to sequential advance to turn 1"
        );
        assert_eq!(jump_turn1.input_tokens, seq_turn1.input_tokens);

        // The reconstructed context contains the recorded prior turns 0..=1.
        let body: Value = serde_json::from_slice(&jump_turn1.body.to_wire().unwrap()).unwrap();
        let messages = body["messages"].to_string();
        assert!(
            messages.contains("q0"),
            "context carries turn 0: {messages}"
        );
        assert!(
            messages.contains("a0"),
            "context carries reply 0: {messages}"
        );
        assert!(
            messages.contains("q1"),
            "context carries turn 1: {messages}"
        );
        assert_eq!(body["messages"].as_array().unwrap().len(), 3);

        // seek_to(0) is identical to advance_to(0): the sequential contract is intact.
        let mut seek0 = ConversationSession::new(build_data(), SessionId::from("session")).unwrap();
        seek0.seek_to(0).unwrap();
        let mut adv0 = ConversationSession::new(build_data(), SessionId::from("session")).unwrap();
        adv0.advance_to(0).unwrap();
        let ov = Overrides::new();
        let seek0_body = seek0
            .materialize(
                &EndpointRequestMaterializer,
                &ChatEndpoint,
                &model_endpoint(),
                CreditPhase::Profiling,
                &ov,
            )
            .unwrap()
            .body;
        let adv0_body = adv0
            .materialize(
                &EndpointRequestMaterializer,
                &ChatEndpoint,
                &model_endpoint(),
                CreditPhase::Profiling,
                &ov,
            )
            .unwrap()
            .body;
        assert_eq!(seek0_body, adv0_body);
    }

    #[test]
    fn seek_to_rejects_jump_for_capture_dependent_context_modes() {
        // A without-responses mode reconstructs context from live captured replies,
        // which a jump cannot supply; seeking past turn 0 must fail closed rather than
        // silently drop the missing prior responses.
        let mut pool = SegmentPool::new();
        let q0 = message(&mut pool, None, "user", "q0");
        let q1 = message(&mut pool, Some(q0), "user", "q1");
        let data = dataset(
            ConversationContextMode::DeltasWithoutResponses,
            vec![
                Turn {
                    body: smallvec![q0],
                    input_tokens: Some(2),
                    ..Turn::default()
                },
                Turn {
                    body: smallvec![q1],
                    input_tokens: Some(3),
                    ..Turn::default()
                },
            ],
            pool,
        );
        let mut session = ConversationSession::new(data, SessionId::from("session")).unwrap();
        let error = session.seek_to(1).unwrap_err().to_string();
        assert!(
            error.contains("jump-resume"),
            "expected a jump-resume rejection, got: {error}"
        );
        // Seeking to turn 0 is still allowed for every mode.
        session.seek_to(0).unwrap();
    }

    #[test]
    fn raw_token_materializer_skips_endpoint_json_for_simulators() {
        let mut pool = SegmentPool::new();
        let raw_token_ids = pool.intern_token_ids(None, [11_u32, 22, 33]).unwrap();
        let data = dataset(
            ConversationContextMode::MessageArrayWithResponses,
            vec![Turn {
                model: Some(ModelId::from("token-model")),
                max_tokens: Some(9),
                input_tokens: Some(3),
                body: Turn::dispatch_body(None, Some(raw_token_ids), &[]),
                ..Turn::default()
            }],
            pool,
        );
        let registry = EndpointRegistry::builtin().unwrap();
        let endpoint = registry
            .prepare(
                &EndpointId::new("vllm_generate").unwrap(),
                RawEndpointConfig::default(),
            )
            .unwrap();
        data.validate_for_endpoint(endpoint.descriptor()).unwrap();
        let mut session = ConversationSession::new(data, SessionId::from("session")).unwrap();
        session.advance_to(0).unwrap();

        let request = session
            .materialize_prepared(
                &TraceHashAwareRequestMaterializer,
                endpoint.as_ref(),
                "default-model",
                CreditPhase::Profiling,
                &Overrides::new(),
            )
            .unwrap();

        assert!(request.body.to_wire().unwrap().is_empty());
        assert_eq!(request.raw_token_ids, Some(raw_token_ids));
        assert_eq!(request.model, "token-model");
        assert_eq!(request.max_tokens, Some(9));
        assert!(!request.streaming);
        assert_eq!(request.input_tokens, Some(3));
        assert_eq!(
            request.endpoint_path.as_deref(),
            Some("/inference/v1/generate")
        );
    }

    #[test]
    fn token_ids_inside_an_ordinary_raw_body_do_not_select_native_dispatch() {
        let mut pool = SegmentPool::new();
        let wire = Bytes::from_static(br#"{"messages":[],"token_ids":[11,22,33]}"#);
        let raw_payload = pool.intern_raw(None, wire.clone()).unwrap();
        let raw_token_ids = pool
            .intern_token_ids(Some(raw_payload), [11_u32, 22, 33])
            .unwrap();
        let data = dataset(
            ConversationContextMode::MessageArrayWithResponses,
            vec![Turn {
                input_tokens: Some(3),
                body: Turn::dispatch_body(Some(raw_payload), Some(raw_token_ids), &[]),
                ..Turn::default()
            }],
            pool,
        );
        let endpoint = EndpointRegistry::builtin()
            .unwrap()
            .prepare(
                &EndpointId::new("chat").unwrap(),
                RawEndpointConfig::default(),
            )
            .unwrap();
        let mut session = ConversationSession::new(data, SessionId::from("session")).unwrap();
        session.advance_to(0).unwrap();

        let request = session
            .materialize_prepared(
                &TraceHashAwareRequestMaterializer,
                endpoint.as_ref(),
                "default-model",
                CreditPhase::Profiling,
                &Overrides::new(),
            )
            .unwrap();

        assert_eq!(request.body.to_wire().unwrap(), wire);
        assert_eq!(request.raw_token_ids, None);
        assert_eq!(request.input_tokens, Some(3));
    }

    #[test]
    fn prepared_materializer_uses_the_bound_open_endpoint_directly() {
        let mut pool = SegmentPool::new();
        let text = pool
            .intern_text(
                None,
                Role::from("user"),
                Bytes::from_static(b"hello"),
                vec![1, 2],
            )
            .unwrap();
        let data = dataset(
            ConversationContextMode::DeltasWithResponses,
            vec![Turn {
                role: Some(Role::from("user")),
                content: smallvec![ContentGroup {
                    kind: MediaKind::Text,
                    name: String::new(),
                    handles: smallvec![text],
                    uuids: smallvec![],
                }],
                input_tokens: Some(2),
                max_tokens: Some(7),
                ..Turn::default()
            }],
            pool,
        );
        let registry = EndpointRegistry::builtin().unwrap();
        let endpoint = registry
            .prepare(
                &EndpointId::new("chat").unwrap(),
                RawEndpointConfig {
                    streaming: true,
                    use_server_token_count: true,
                    headers: BTreeMap::from([("x-profile".into(), "bound".into())]),
                    ..RawEndpointConfig::default()
                },
            )
            .unwrap();
        let mut session = ConversationSession::new(data, SessionId::from("session")).unwrap();
        session.advance_to(0).unwrap();

        let request = session
            .materialize_prepared(
                &EndpointRequestMaterializer,
                endpoint.as_ref(),
                "direct-model",
                CreditPhase::Profiling,
                &Overrides::new(),
            )
            .unwrap();

        let body: Value = serde_json::from_slice(&request.body.to_wire().unwrap()).unwrap();
        assert_eq!(body["model"], "direct-model");
        assert_eq!(body["messages"][0]["content"], "hello");
        assert_eq!(body["max_completion_tokens"], 7);
        assert_eq!(body["stream"], true);
        assert_eq!(request.headers["x-profile"], "bound");
        assert_eq!(
            request.endpoint_path.as_deref(),
            Some("/v1/chat/completions")
        );
    }

    #[test]
    fn deltas_without_responses_interleaves_live_assistant_and_accounts_tokens() {
        let mut pool = SegmentPool::new();
        let q0 = message(&mut pool, None, "user", "q0");
        let q1 = message(&mut pool, Some(q0), "user", "q1");
        let data = dataset(
            ConversationContextMode::DeltasWithoutResponses,
            vec![
                Turn {
                    body: smallvec![q0],
                    input_tokens: Some(2),
                    max_tokens: Some(4),
                    ..Turn::default()
                },
                Turn {
                    body: smallvec![q1],
                    input_tokens: Some(3),
                    max_tokens: Some(5),
                    audio_duration_seconds: Some(2.5),
                    ..Turn::default()
                },
            ],
            pool,
        );
        let mut session = ConversationSession::new(data, SessionId::from("session")).unwrap();
        session.advance_to(0).unwrap();
        session
            .capture_response(
                EndpointTurn {
                    role: Some("assistant".into()),
                    texts: vec![Media::new(vec!["a0".into()])],
                    ..EndpointTurn::default()
                },
                11,
                Some(0),
            )
            .unwrap();
        session.advance_to(1).unwrap();
        let request = session
            .materialize(
                &EndpointRequestMaterializer,
                &ChatEndpoint,
                &model_endpoint(),
                CreditPhase::Profiling,
                &Overrides::new(),
            )
            .unwrap();
        let body: Value = serde_json::from_slice(&request.body.to_wire().unwrap()).unwrap();
        assert_eq!(body["messages"][0]["content"], "q0");
        assert_eq!(body["messages"][1]["content"], "a0");
        assert_eq!(body["messages"][2]["content"], "q1");
        assert_eq!(request.input_tokens, Some(16));
        assert_eq!(request.max_tokens, Some(5));
        assert_eq!(request.audio_duration_seconds, Some(2.5));
        assert_eq!(request.accuracy.unwrap().correlation_id.as_str(), "corr");
    }

    #[test]
    fn message_array_without_responses_prefix_diffs_and_interleaves() {
        let mut pool = SegmentPool::new();
        let q0 = message(&mut pool, None, "user", "q0");
        let q1 = message(&mut pool, Some(q0), "user", "q1");
        let data = dataset(
            ConversationContextMode::MessageArrayWithoutResponses,
            vec![
                Turn {
                    body: smallvec![q0],
                    input_tokens: Some(2),
                    ..Turn::default()
                },
                Turn {
                    body: smallvec![q0, q1],
                    input_tokens: Some(5),
                    ..Turn::default()
                },
            ],
            pool,
        );
        let mut session = ConversationSession::new(data, SessionId::from("session")).unwrap();
        session.advance_to(0).unwrap();
        session
            .capture_response(
                EndpointTurn {
                    role: Some("assistant".into()),
                    texts: vec![Media::new(vec!["a0".into()])],
                    ..EndpointTurn::default()
                },
                7,
                Some(0),
            )
            .unwrap();
        session.advance_to(1).unwrap();
        let request = session
            .materialize(
                &EndpointRequestMaterializer,
                &ChatEndpoint,
                &model_endpoint(),
                CreditPhase::Profiling,
                &Overrides::new(),
            )
            .unwrap();
        let body: Value = serde_json::from_slice(&request.body.to_wire().unwrap()).unwrap();
        assert_eq!(body["messages"].as_array().unwrap().len(), 3);
        assert_eq!(body["messages"][0]["content"], "q0");
        assert_eq!(body["messages"][1]["content"], "a0");
        assert_eq!(body["messages"][2]["content"], "q1");
        assert_eq!(request.input_tokens, Some(12));
    }

    #[test]
    fn structured_content_headers_and_parameters_are_resolved_from_handles() {
        let mut pool = SegmentPool::new();
        let text = pool
            .intern_text(
                None,
                Role::from("user"),
                Bytes::from_static(b"hello"),
                vec![1, 2],
            )
            .unwrap();
        let headers = pool
            .intern_raw(None, Bytes::from_static(br#"{"x-custom":"yes"}"#))
            .unwrap();
        let params = pool
            .intern_raw(None, Bytes::from_static(br#"{"api-version":"2026-01"}"#))
            .unwrap();
        let data = dataset(
            ConversationContextMode::DeltasWithResponses,
            vec![Turn {
                role: Some(Role::from("user")),
                content: smallvec![ContentGroup {
                    kind: MediaKind::Text,
                    name: String::new(),
                    handles: smallvec![text],
                    uuids: smallvec![],
                }],
                input_tokens: Some(2),
                streaming: Some(false),
                extra_headers: Some(headers),
                request_parameters: Some(params),
                ..Turn::default()
            }],
            pool,
        );
        let mut session = ConversationSession::new(data, SessionId::from("session")).unwrap();
        session.advance_to(0).unwrap();
        let request = session
            .materialize(
                &EndpointRequestMaterializer,
                &ChatEndpoint,
                &model_endpoint(),
                CreditPhase::Profiling,
                &Overrides::new(),
            )
            .unwrap();
        assert_eq!(request.headers["x-custom"], "yes");
        assert_eq!(request.parameters["api-version"], "2026-01");
        let body: Value = serde_json::from_slice(&request.body.to_wire().unwrap()).unwrap();
        assert_eq!(body["messages"][0]["content"], "hello");
        assert_eq!(body["stream"], false);
        assert!(!request.streaming);
    }

    #[test]
    fn explicit_overrides_update_wire_and_effective_request_metadata() {
        let mut pool = SegmentPool::new();
        let text = pool
            .intern_text(
                None,
                Role::from("user"),
                Bytes::from_static(b"hello"),
                vec![1, 2],
            )
            .unwrap();
        let data = dataset(
            ConversationContextMode::DeltasWithResponses,
            vec![Turn {
                role: Some(Role::from("user")),
                model: Some(ModelId::from("authored-model")),
                max_tokens: Some(4),
                content: smallvec![ContentGroup {
                    kind: MediaKind::Text,
                    name: String::new(),
                    handles: smallvec![text],
                    uuids: smallvec![],
                }],
                input_tokens: Some(2),
                ..Turn::default()
            }],
            pool,
        );
        let mut session = ConversationSession::new(data, SessionId::from("session")).unwrap();
        session.advance_to(0).unwrap();
        let mut overrides = Overrides::new();
        overrides.set_model("dispatch-model");
        overrides.set_max_tokens("max_completion_tokens", 13);
        overrides.set_stream(false);
        let request = session
            .materialize(
                &EndpointRequestMaterializer,
                &ChatEndpoint,
                &model_endpoint(),
                CreditPhase::Profiling,
                &overrides,
            )
            .unwrap();
        let body: Value = serde_json::from_slice(&request.body.to_wire().unwrap()).unwrap();
        assert_eq!(body["model"], "dispatch-model");
        assert_eq!(body["max_completion_tokens"], 13);
        assert_eq!(body["stream"], false);
        assert_eq!(request.model, "dispatch-model");
        assert_eq!(request.max_tokens, Some(13));
        assert!(!request.streaming);
    }

    #[test]
    fn authored_extra_body_updates_effective_request_metadata() {
        let mut pool = SegmentPool::new();
        let text = pool
            .intern_text(
                None,
                Role::from("user"),
                Bytes::from_static(b"hello"),
                vec![1, 2],
            )
            .unwrap();
        let extra = pool
            .intern_raw(
                Some(text),
                Bytes::from_static(
                    br#"{"model":"body-model","stream":false,"max_completion_tokens":15}"#,
                ),
            )
            .unwrap();
        let data = dataset(
            ConversationContextMode::DeltasWithResponses,
            vec![Turn {
                role: Some(Role::from("user")),
                model: Some(ModelId::from("turn-model")),
                max_tokens: Some(4),
                content: smallvec![ContentGroup {
                    kind: MediaKind::Text,
                    name: String::new(),
                    handles: smallvec![text],
                    uuids: smallvec![],
                }],
                extra_body: Some(extra),
                input_tokens: Some(2),
                ..Turn::default()
            }],
            pool,
        );
        let mut session = ConversationSession::new(data, SessionId::from("session")).unwrap();
        session.advance_to(0).unwrap();
        let request = session
            .materialize(
                &EndpointRequestMaterializer,
                &ChatEndpoint,
                &model_endpoint(),
                CreditPhase::Profiling,
                &Overrides::new(),
            )
            .unwrap();
        let body: Value = serde_json::from_slice(&request.body.to_wire().unwrap()).unwrap();
        assert_eq!(body["model"], "body-model");
        assert_eq!(body["stream"], false);
        assert_eq!(body["max_completion_tokens"], 15);
        assert_eq!(request.model, "body-model");
        assert_eq!(request.max_tokens, Some(15));
        assert!(!request.streaming);
    }

    #[test]
    fn responses_override_resolves_dialect_body_and_default_path() {
        let mut pool = SegmentPool::new();
        let text = pool
            .intern_text(
                None,
                Role::from("user"),
                Bytes::from_static(b"hello responses"),
                vec![1, 2],
            )
            .unwrap();
        let data = dataset(
            ConversationContextMode::DeltasWithResponses,
            vec![Turn {
                role: Some(Role::from("user")),
                endpoint: Some("responses".into()),
                max_tokens: Some(9),
                content: smallvec![ContentGroup {
                    kind: MediaKind::Text,
                    name: String::new(),
                    handles: smallvec![text],
                    uuids: smallvec![],
                }],
                input_tokens: Some(2),
                ..Turn::default()
            }],
            pool,
        );
        let mut session = ConversationSession::new(data, SessionId::from("session")).unwrap();
        session.advance_to(0).unwrap();
        let resolver = BuiltinEndpointResolver::default();
        let endpoint = resolver
            .resolve(session.endpoint_override().unwrap())
            .unwrap();
        let mut configured = model_endpoint();
        configured.endpoint.endpoint_type =
            endpoint.descriptor().legacy_type().expect("endpoint type");
        let request = session
            .materialize(
                &EndpointRequestMaterializer,
                endpoint.as_ref(),
                &configured,
                CreditPhase::Profiling,
                &Overrides::new(),
            )
            .unwrap();
        let body: Value = serde_json::from_slice(&request.body.to_wire().unwrap()).unwrap();
        assert!(body.get("messages").is_none());
        assert_eq!(body["input"][0]["content"], "hello responses");
        assert_eq!(body["max_output_tokens"], 9);
        assert_eq!(request.endpoint.as_deref(), Some("responses"));
        assert_eq!(request.endpoint_path.as_deref(), Some("/v1/responses"));
    }

    use crate::endpoints::{ShapeLowerer, TurnMessageLowerer};

    fn content_turn(text: Handle, image: Option<Handle>) -> Turn {
        let mut content = smallvec![ContentGroup {
            kind: MediaKind::Text,
            name: String::new(),
            handles: smallvec![text],
            uuids: smallvec![],
        }];
        if let Some(image) = image {
            content.push(ContentGroup {
                kind: MediaKind::Image,
                name: String::new(),
                handles: smallvec![image],
                uuids: smallvec![],
            });
        }
        Turn {
            role: Some(Role::from("user")),
            content,
            input_tokens: Some(2),
            max_tokens: Some(7),
            ..Turn::default()
        }
    }

    fn prepared_chat() -> Box<dyn PreparedEndpoint> {
        EndpointRegistry::builtin()
            .unwrap()
            .prepare(
                &EndpointId::new("chat").unwrap(),
                RawEndpointConfig::default(),
            )
            .unwrap()
    }

    fn dispatch_body(dataset: Arc<Dataset>, endpoint: &dyn PreparedEndpoint) -> Bytes {
        let mut session = ConversationSession::new(dataset, SessionId::from("session")).unwrap();
        session.advance_to(0).unwrap();
        session
            .materialize_prepared(
                &EndpointRequestMaterializer,
                endpoint,
                "primary-model",
                CreditPhase::Profiling,
                &Overrides::new(),
            )
            .unwrap()
            .body
            .to_wire()
            .unwrap()
    }

    fn one_content_turn_dataset(text: Handle, image: Option<Handle>, pool: SegmentPool) -> Dataset {
        let mut conversation = Conversation::new("session");
        conversation.context_mode = Some(ConversationContextMode::MessageArrayWithResponses);
        conversation.turns = vec![content_turn(text, image)];
        Dataset::new(
            vec![conversation],
            Arc::new(pool.freeze()),
            "sequential",
            ConversationContextMode::MessageArrayWithResponses,
        )
        .unwrap()
    }

    #[test]
    fn lowered_dispatch_body_preserves_bytes() {
        let mut pool = SegmentPool::new();
        let text = pool
            .intern_text(
                None,
                Role::from("user"),
                Bytes::from_static(b"hello"),
                vec![1, 2],
            )
            .unwrap();
        let image = pool
            .intern_media(
                None,
                MediaKind::Image,
                Bytes::from_static(b"http://example/a.png"),
            )
            .unwrap();
        let base = one_content_turn_dataset(text, Some(image), pool);
        let endpoint = prepared_chat();

        let unlowered = Arc::new(base.clone());
        let mut lowered_ds = base;
        let lowerer = ShapeLowerer::for_descriptor_id(endpoint.descriptor().id).unwrap();
        lowered_ds.lower_messages_for_endpoint(&lowerer).unwrap();
        lowered_ds.lower_messages_for_endpoint(&lowerer).unwrap();
        let lowered = Arc::new(lowered_ds);

        let before = dispatch_body(unlowered, endpoint.as_ref());
        let after = dispatch_body(lowered.clone(), endpoint.as_ref());
        assert_eq!(before, after);
        assert_eq!(lowered.conversations()[0].turns[0].body.len(), 1);
    }

    #[test]
    fn identical_content_turns_dedup_to_one_segment_when_lowered() {
        let mut pool = SegmentPool::new();
        let text = pool
            .intern_text(
                None,
                Role::from("user"),
                Bytes::from_static(b"same"),
                vec![9],
            )
            .unwrap();
        let mut conversation = Conversation::new("session");
        conversation.context_mode = Some(ConversationContextMode::MessageArrayWithResponses);
        conversation.turns = vec![content_turn(text, None), content_turn(text, None)];
        let mut dataset = Dataset::new(
            vec![conversation],
            Arc::new(pool.freeze()),
            "sequential",
            ConversationContextMode::MessageArrayWithResponses,
        )
        .unwrap();
        let lowerer = ShapeLowerer::for_descriptor_id("chat").unwrap();
        dataset.lower_messages_for_endpoint(&lowerer).unwrap();

        let turns = &dataset.conversations()[0].turns;
        assert_eq!(turns[0].body[0], turns[1].body[0]);
    }

    /// Resolves every turn to the one endpoint under test.
    struct SingleEndpointLookup<'a>(&'a dyn PreparedEndpoint);

    impl crate::dataset::TurnEndpointLookup for SingleEndpointLookup<'_> {
        fn endpoint_for(&self, _name: Option<&str>) -> Option<&dyn PreparedEndpoint> {
            Some(self.0)
        }
    }

    /// The optimization itself, not just its answer: after precompute, a
    /// continuation turn of the default live-chat context mode must carry an
    /// established `image_count`, so dispatch never deserializes the body it
    /// just built. Asserting only the value would pass on the parse fallback.
    #[test]
    fn continuation_turns_carry_an_established_image_count() {
        let mut pool = SegmentPool::new();
        let text = pool
            .intern_text(
                None,
                Role::from("user"),
                Bytes::from_static(b"hello"),
                vec![1, 2],
            )
            .unwrap();
        let image = pool
            .intern_media(
                None,
                MediaKind::Image,
                Bytes::from_static(b"http://example/a.png"),
            )
            .unwrap();
        let mut conversation = Conversation::new("session");
        conversation.context_mode = Some(ConversationContextMode::DeltasWithoutResponses);
        // One image in turn 0; the deltas resend it under every later turn.
        conversation.turns = vec![
            content_turn(text, Some(image)),
            content_turn(text, None),
            content_turn(text, None),
        ];
        let mut dataset = Dataset::new(
            vec![conversation],
            Arc::new(pool.freeze()),
            "sequential",
            ConversationContextMode::DeltasWithoutResponses,
        )
        .unwrap();
        let endpoint = prepared_chat();
        let lowerer = ShapeLowerer::for_descriptor_id(endpoint.descriptor().id).unwrap();
        dataset.lower_messages_for_endpoint(&lowerer).unwrap();
        dataset
            .precompute_image_counts(&SingleEndpointLookup(endpoint.as_ref()), "primary-model")
            .unwrap();
        let dataset = Arc::new(dataset);

        let mut session = ConversationSession::new(dataset, SessionId::from("session")).unwrap();
        for turn_index in 0..3 {
            session.advance_to(turn_index).unwrap();
            let request = session
                .materialize_prepared(
                    &EndpointRequestMaterializer,
                    endpoint.as_ref(),
                    "primary-model",
                    CreditPhase::Profiling,
                    &Overrides::new(),
                )
                .unwrap();
            let body = request.body.clone().to_wire().unwrap();
            let parsed: Value = serde_json::from_slice(&body).unwrap();
            let on_the_wire = endpoint.extract_payload_inputs(&parsed).image_count;
            assert_eq!(
                request.image_count,
                Some(on_the_wire),
                "turn {turn_index} must establish the count the body actually carries"
            );
            let reply = EndpointTurn {
                role: Some("assistant".into()),
                texts: vec![Media::new(vec!["reply".to_string()])],
                ..EndpointTurn::default()
            };
            let images = reply_image_count(&reply, endpoint.as_ref());
            assert_eq!(images, Some(0), "a text-only reply carries no image");
            session.capture_response(reply, 4, images).unwrap();
        }
    }

    /// A dialect that selects one turn out of the list rather than concatenating
    /// them all must not be prefix-summed: `kserve_v2_vlm` formats
    /// `request.turns().first()`, so under a delta mode the wire always carries
    /// turn 0's images and a sum would over-report every later turn. Those turns
    /// must stay unestablished so dispatch parses the body and gets the truth.
    #[test]
    fn a_turn_selecting_dialect_is_never_prefix_summed() {
        let mut pool = SegmentPool::new();
        let text = pool
            .intern_text(
                None,
                Role::from("user"),
                Bytes::from_static(b"hello"),
                vec![1, 2],
            )
            .unwrap();
        let image = pool
            .intern_media(
                None,
                MediaKind::Image,
                Bytes::from_static(b"http://example/a.png"),
            )
            .unwrap();
        let mut conversation = Conversation::new("session");
        conversation.context_mode = Some(ConversationContextMode::DeltasWithoutResponses);
        // Every turn carries an image, so a prefix sum would report 1, 2, 3
        // while the wire only ever carries turn 0's one image.
        conversation.turns = vec![
            content_turn(text, Some(image)),
            content_turn(text, Some(image)),
            content_turn(text, Some(image)),
        ];
        let mut dataset = Dataset::new(
            vec![conversation],
            Arc::new(pool.freeze()),
            "sequential",
            ConversationContextMode::DeltasWithoutResponses,
        )
        .unwrap();
        let endpoint = prepare_endpoint("kserve_v2_vlm");
        dataset
            .precompute_image_counts(&SingleEndpointLookup(endpoint.as_ref()), "primary-model")
            .unwrap();

        let id = SessionId::from("session");
        assert_eq!(
            dataset.cached_image_count(&id, 0),
            Some(1),
            "turn 0 is the one index every context mode renders in full"
        );
        for turn_index in 1..3 {
            assert_eq!(
                dataset.cached_image_count(&id, turn_index),
                None,
                "turn {turn_index} must fall back to the parse, not a sum"
            );
        }
    }

    /// An extractor that owns its dialect's image count must be believed when it
    /// reports zero, not just when it reports some. `kserve_v2_vlm` omits the
    /// image tensor entirely when every image content is empty, so the wire
    /// carries no image and the count is exactly zero — while
    /// `composed_image_count` would say one, because `handles.len()` cannot see
    /// the formatter's `!content.is_empty()` filter.
    #[test]
    fn an_extractor_owning_its_count_is_believed_when_it_reports_zero() {
        let mut pool = SegmentPool::new();
        let text = pool
            .intern_text(
                None,
                Role::from("user"),
                Bytes::from_static(b"hello"),
                vec![1, 2],
            )
            .unwrap();
        let empty_image = pool
            .intern_media(None, MediaKind::Image, Bytes::from_static(b""))
            .unwrap();
        let mut conversation = Conversation::new("session");
        conversation.context_mode = Some(ConversationContextMode::MessageArrayWithResponses);
        conversation.turns = vec![content_turn(text, Some(empty_image))];
        let mut dataset = Dataset::new(
            vec![conversation],
            Arc::new(pool.freeze()),
            "sequential",
            ConversationContextMode::MessageArrayWithResponses,
        )
        .unwrap();
        let endpoint = prepare_endpoint("kserve_v2_vlm");
        dataset
            .precompute_image_counts(&SingleEndpointLookup(endpoint.as_ref()), "primary-model")
            .unwrap();

        assert_eq!(
            dataset.cached_image_count(&SessionId::from("session"), 0),
            Some(0),
            "an empty image content sends no image, so the count is exactly zero"
        );
    }

    /// The concatenation capability must follow the dialect, not its descriptor
    /// id: `kserve_chat` and `sagemaker` wrap `ChatEndpoint` under their own ids
    /// and would otherwise send every continuation turn back to the parse.
    #[test]
    fn chat_wrapping_dialects_report_that_they_render_all_turns() {
        for id in ["chat", "responses", "messages", "kserve_chat", "sagemaker"] {
            assert!(
                prepare_endpoint(id).renders_all_turns(),
                "{id} concatenates every turn and must say so"
            );
        }
        for id in ["kserve_v2_vlm", "kserve_v2_infer", "completions"] {
            assert!(
                !prepare_endpoint(id).renders_all_turns(),
                "{id} selects turns and must not be prefix-summed"
            );
        }
    }

    /// `MessageArrayWithoutResponses` is the one mode where turn 0 is not a
    /// single-turn request: `merge_message_array_snapshots` emits
    /// `split_snapshot(t0)`, which fans a turn holding N authored `raw_messages`
    /// into N formatter turns. A selecting dialect then renders one of those N
    /// while a count built from the unsplit turn covers all N, so the slot must
    /// stay unestablished and let dispatch parse.
    ///
    /// `template` is the reachable case: its body is user-authored and can put
    /// the whole `raw_messages` array into a counted item array, so unlike the
    /// other selectors its count is established rather than `None` by accident.
    #[test]
    fn a_selecting_dialect_describes_no_turn_under_split_snapshots() {
        let messages = serde_json::json!([
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "http://example/a.png"}}
            ]},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "http://example/b.png"}}
            ]},
        ]);
        let mut pool = SegmentPool::new();
        let raw_messages = pool
            .intern_raw(None, Bytes::from(serde_json::to_vec(&messages).unwrap()))
            .unwrap();
        let turn = Turn {
            role: Some(Role::from("user")),
            raw_messages: Some(raw_messages),
            input_tokens: Some(2),
            ..Turn::default()
        };
        // Renders every authored message into a counted `messages` array, which
        // is exactly what `split_snapshot` then splits apart at dispatch.
        let template = r#"{"model":"m","messages":{{ turn.raw_messages | tojson }}}"#;
        let endpoint = EndpointRegistry::builtin()
            .unwrap()
            .prepare(
                &EndpointId::new("template").unwrap(),
                RawEndpointConfig {
                    template: Some(template.to_string()),
                    ..RawEndpointConfig::default()
                },
            )
            .unwrap();

        let counts = |mode: ConversationContextMode| {
            let mut conversation = Conversation::new("session");
            conversation.context_mode = Some(mode);
            conversation.turns = vec![turn.clone()];
            let mut dataset = Dataset::new(
                vec![conversation],
                Arc::new(pool.clone().freeze()),
                "sequential",
                mode,
            )
            .unwrap();
            dataset
                .precompute_image_counts(&SingleEndpointLookup(endpoint.as_ref()), "m")
                .unwrap();
            dataset.cached_image_count(&SessionId::from("session"), 0)
        };

        // Control: the same turn under a mode that does not split establishes a
        // count, so the assertion below is about the splitting, not about the
        // fixture failing to produce one.
        assert_eq!(
            counts(ConversationContextMode::MessageArrayWithResponses),
            Some(2),
            "an unsplit turn 0 is one formatter turn and is describable"
        );
        assert_eq!(
            counts(ConversationContextMode::MessageArrayWithoutResponses),
            None,
            "split snapshots hand a selecting dialect one of N messages, so a \
             count over all N describes a body that is never sent"
        );
    }

    fn prepare_endpoint(id: &str) -> Box<dyn PreparedEndpoint> {
        EndpointRegistry::builtin()
            .unwrap()
            .prepare(&EndpointId::new(id).unwrap(), RawEndpointConfig::default())
            .unwrap()
    }

    fn text_turn(
        pool: &mut SegmentPool,
        text: &'static [u8],
        with_max_tokens: bool,
        with_extra_body: bool,
    ) -> Turn {
        // Text-segment identity keys on role + tokens (authoritative tokens), not
        // bytes; derive distinct tokens per content so segments don't mis-dedup.
        let tokens: Vec<u32> = text.iter().map(|&byte| byte as u32).collect();
        let text_handle = pool
            .intern_text(None, Role::from("user"), Bytes::from_static(text), tokens)
            .unwrap();
        let extra_body = with_extra_body.then(|| {
            pool.intern_raw(None, Bytes::from_static(br#"{"temperature":0.5}"#))
                .unwrap()
        });
        Turn {
            role: Some(Role::from("user")),
            content: smallvec![ContentGroup {
                kind: MediaKind::Text,
                name: String::new(),
                handles: smallvec![text_handle],
                uuids: smallvec![],
            }],
            input_tokens: Some(2),
            max_tokens: with_max_tokens.then_some(7),
            extra_body,
            ..Turn::default()
        }
    }

    fn single_conversation_dataset(
        mode: ConversationContextMode,
        turns: Vec<Turn>,
        pool: SegmentPool,
    ) -> Dataset {
        let mut conversation = Conversation::new("session");
        conversation.context_mode = Some(mode);
        conversation.turns = turns;
        Dataset::new(
            vec![conversation],
            Arc::new(pool.freeze()),
            "sequential",
            mode,
        )
        .unwrap()
    }

    /// A captured assistant reply, lowered before capture exactly as
    /// `multiturn::NativeSessionBackend::build_next_turn` lowers one.
    fn captured_reply(endpoint: &dyn PreparedEndpoint, index: usize) -> EndpointTurn {
        let mut reply = EndpointTurn {
            role: Some("assistant".into()),
            texts: vec![Media::new(vec![format!("reply {index}")])],
            ..EndpointTurn::default()
        };
        if let Some(lowerer) = ShapeLowerer::for_descriptor_id(endpoint.descriptor().id) {
            reply.lowered = Some(lowerer.lower_turn(&reply).unwrap());
        }
        reply
    }

    fn dispatch_turn(
        dataset: Arc<Dataset>,
        endpoint: &dyn PreparedEndpoint,
        turn_index: usize,
        overrides: &Overrides,
    ) -> Bytes {
        let mut session = ConversationSession::new(dataset, SessionId::from("session")).unwrap();
        for index in 0..=turn_index {
            session.advance_to(index).unwrap();
            // A mode that splices live replies gets one after every turn but the
            // current, which is what makes a continuation body more than its
            // authored turns.
            if index < turn_index && session.should_capture_response() {
                session
                    .capture_response(captured_reply(endpoint, index), 3, Some(0))
                    .unwrap();
            }
        }
        session
            .materialize_prepared(
                &EndpointRequestMaterializer,
                endpoint,
                "primary-model",
                CreditPhase::Profiling,
                overrides,
            )
            .unwrap()
            .body
            .to_wire()
            .unwrap()
    }

    #[test]
    fn websocket_materialization_resolves_handles_and_applies_overrides_before_dispatch() {
        let mut pool = SegmentPool::new();
        let turn = text_turn(&mut pool, b"hello world", true, false);
        let dataset = Arc::new(single_conversation_dataset(
            ConversationContextMode::MessageArrayWithResponses,
            vec![turn],
            pool,
        ));
        let endpoint = prepare_endpoint("responses");
        let mut session = ConversationSession::new(dataset, SessionId::from("session")).unwrap();
        session.advance_to(0).unwrap();
        let mut overrides = Overrides::new();
        overrides.set_model("override-model");

        let request = session
            .materialize_prepared(
                &WsRequestMaterializer,
                endpoint.as_ref(),
                "primary-model",
                CreditPhase::Profiling,
                &overrides,
            )
            .unwrap();
        let operation = request.body.websocket().unwrap();
        let event: Value = serde_json::from_slice(operation.messages()[0].payload()).unwrap();
        let fallback: Value = serde_json::from_slice(
            operation
                .http_sse_fallback_body()
                .expect("Responses prepares an equivalent HTTP/SSE request"),
        )
        .unwrap();

        assert_eq!(request.model, "override-model");
        assert_eq!(event["type"], "response.create");
        assert_eq!(event["model"], "override-model");
        assert_eq!(event["input"][0]["content"], "hello world");
        assert_eq!(fallback["model"], "override-model");
        assert_eq!(fallback["input"][0]["content"], "hello world");
        assert_eq!(fallback["stream"], true);
        assert_eq!(
            request
                .extracted
                .as_ref()
                .map(|input| input.texts.as_slice()),
            Some(["hello world".to_owned()].as_slice()),
            "WebSocket token counting must retain endpoint input rather than parse the artifact envelope"
        );
    }

    #[test]
    fn websocket_materialization_reuses_lowered_wires() {
        let mut pool = SegmentPool::new();
        let turn = text_turn(&mut pool, b"hello world", false, false);
        let mut dataset = single_conversation_dataset(
            ConversationContextMode::MessageArrayWithResponses,
            vec![turn],
            pool,
        );
        let endpoint = prepare_endpoint("responses");
        let lowerer = ShapeLowerer::for_descriptor_id(endpoint.descriptor().id).unwrap();
        dataset.lower_messages_for_endpoint(&lowerer).unwrap();
        let mut session =
            ConversationSession::new(Arc::new(dataset), SessionId::from("session")).unwrap();
        session.advance_to(0).unwrap();
        COMPOSED_CONTENT_RESOLUTIONS.with(|count| count.set(0));

        session
            .materialize_prepared(
                &WsRequestMaterializer,
                endpoint.as_ref(),
                "primary-model",
                CreditPhase::Profiling,
                &Overrides::new(),
            )
            .unwrap();

        COMPOSED_CONTENT_RESOLUTIONS.with(|count| assert_eq!(count.get(), 0));
    }

    #[test]
    fn cached_plan_is_byte_identical_to_per_dispatch_format_across_matrix() {
        // Cache and per-dispatch formatting must remain byte-identical across the
        // endpoint, context, override, max-token, and extra-body matrix.
        //
        // `DeltasWithoutResponses` is here for its *continuation* turns, whose
        // cached plan holds only the authored turns and splices the live replies
        // in at dispatch. Byte identity against the fully formatted body is the
        // only thing that establishes those splice positions are right; the
        // per-turn assertions below additionally establish that the spliced path
        // is the one being compared, since a cache miss would also compare equal.
        for endpoint_id in ["chat", "responses", "messages"] {
            for mode in [
                ConversationContextMode::MessageArrayWithResponses,
                ConversationContextMode::DeltasWithResponses,
                ConversationContextMode::DeltasWithoutResponses,
            ] {
                for with_max_tokens in [false, true] {
                    for with_extra_body in [false, true] {
                        for overrides_variant in 0..4 {
                            let mut pool = SegmentPool::new();
                            let turns = vec![
                                text_turn(
                                    &mut pool,
                                    b"hello world",
                                    with_max_tokens,
                                    with_extra_body,
                                ),
                                text_turn(
                                    &mut pool,
                                    b"second turn",
                                    with_max_tokens,
                                    with_extra_body,
                                ),
                                text_turn(
                                    &mut pool,
                                    b"third turn",
                                    with_max_tokens,
                                    with_extra_body,
                                ),
                            ];
                            let base = single_conversation_dataset(mode, turns, pool);
                            let endpoint = prepare_endpoint(endpoint_id);
                            let lowerer =
                                ShapeLowerer::for_descriptor_id(endpoint.descriptor().id).unwrap();

                            let mut cached_ds = base.clone();
                            cached_ds.lower_messages_for_endpoint(&lowerer).unwrap();
                            cached_ds
                                .precompute_body_plans(endpoint.as_ref(), "primary-model")
                                .unwrap();

                            let mut uncached_ds = base;
                            uncached_ds.lower_messages_for_endpoint(&lowerer).unwrap();

                            let cached = Arc::new(cached_ds);
                            let uncached = Arc::new(uncached_ds);

                            let mut overrides = Overrides::new();
                            if overrides_variant > 0 {
                                overrides.set_stream(true);
                                overrides.set_model("override-model");
                            }
                            // This dialect's message-array field name, and the
                            // one a *different* dialect would use.
                            let (own, other) = if endpoint_id == "responses" {
                                ("input", "messages")
                            } else {
                                ("messages", "input")
                            };
                            if overrides_variant == 2 {
                                // Naming this plan's own array field replaces the
                                // whole array, replies included, so the splice
                                // must stand down rather than be applied to a
                                // field `merge_overrides` rewrote to a literal.
                                overrides.insert(
                                    own,
                                    serde_json::json!([{"role": "user", "content": "replaced"}]),
                                );
                            }
                            if overrides_variant == 3 {
                                // Naming only the *other* dialect's field must
                                // NOT stand the splice down: that field is an
                                // ordinary appended extra here. An implementation
                                // testing a set of known array names instead of
                                // this plan's own field would decline, and
                                // dispatch a continuation body with no history at
                                // all — which variant 2 cannot catch, since there
                                // the splice is meant to stand down either way.
                                overrides.insert(other, Value::from(7));
                            }

                            for turn_index in 0..3 {
                                let entry = cached
                                    .cached_body_plan(&SessionId::from("session"), turn_index)
                                    .unwrap_or_else(|| {
                                        panic!(
                                            "expected cached plan: endpoint={endpoint_id} mode={mode:?} ti={turn_index}"
                                        )
                                    });
                                let splices = mode
                                    == ConversationContextMode::DeltasWithoutResponses
                                    && turn_index > 0;
                                assert_eq!(
                                    entry
                                        .replies
                                        .as_ref()
                                        .map(|replies| replies.positions.len()),
                                    splices.then_some(turn_index),
                                    "wrong splice reservation: endpoint={endpoint_id} mode={mode:?} ti={turn_index}"
                                );
                                // Declining the cache is safe and invisible to the
                                // comparison below — a build that spliced nothing
                                // would reformat live and still agree — so count
                                // the dispatches that actually took the spliced
                                // path. Without this the whole saving could be
                                // gone with every assertion still passing.
                                let before = SPLICED_DISPATCHES.get();
                                // The splice counter above is blind to the hits
                                // that splice nothing — turn 0, and every turn of
                                // the static modes — which is most of this matrix
                                // and all of the input-array dialects' saving.
                                // Count the cache hit itself so declining is
                                // visible on its own.
                                let hits_before = CACHE_HITS.get();
                                let from_cache = dispatch_turn(
                                    cached.clone(),
                                    endpoint.as_ref(),
                                    turn_index,
                                    &overrides,
                                );
                                assert_eq!(
                                    CACHE_HITS.get() - hits_before,
                                    1,
                                    "dispatch did not take the precomputed plan: \
                                     endpoint={endpoint_id} mode={mode:?} \
                                     overrides={overrides_variant} ti={turn_index}"
                                );
                                assert_eq!(
                                    SPLICED_DISPATCHES.get() - before,
                                    // Everything but an override replacing this
                                    // plan's own array field must splice.
                                    u64::from(splices && overrides_variant != 2),
                                    "spliced-dispatch count: endpoint={endpoint_id} mode={mode:?} \
                                     overrides={overrides_variant} ti={turn_index}"
                                );
                                let from_format = dispatch_turn(
                                    uncached.clone(),
                                    endpoint.as_ref(),
                                    turn_index,
                                    &overrides,
                                );
                                assert_eq!(
                                    from_cache, from_format,
                                    "byte divergence: endpoint={endpoint_id} mode={mode:?} max_tokens={with_max_tokens} extra_body={with_extra_body} overrides={overrides_variant} turn={turn_index}"
                                );
                                // The cached plan on its own carries only the
                                // authored turns, so a continuation body that
                                // equals it never spliced anything and the row
                                // above would have compared two unspliced bodies.
                                // Not asserted for the array-replacing override,
                                // which is meant to leave nothing to splice.
                                if splices && overrides_variant != 2 {
                                    assert_ne!(
                                        entry.plan.materialize_standalone().unwrap().len(),
                                        from_cache.len(),
                                        "continuation body was not spliced: endpoint={endpoint_id} ti={turn_index}"
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn reported_metadata_matches_the_dispatched_body_across_overrides() {
        // `effective_from_plan` reports what the *plan* carries once overrides are
        // folded in, not what the override set asked for — the two differ whenever
        // a later cap name shadows an earlier one, or the plan declares a field the
        // overrides do not. Reading the override set instead would make the runtime
        // dispatch one body and record different metadata, silently.
        for endpoint_id in ["chat", "responses", "messages"] {
            for with_max_tokens in [false, true] {
                for overrides_variant in 0..3 {
                    let mut pool = SegmentPool::new();
                    let turns = vec![text_turn(&mut pool, b"hello", with_max_tokens, false)];
                    let mut data = single_conversation_dataset(
                        ConversationContextMode::MessageArrayWithResponses,
                        turns,
                        pool,
                    );
                    let endpoint = prepare_endpoint(endpoint_id);
                    let lowerer =
                        ShapeLowerer::for_descriptor_id(endpoint.descriptor().id).unwrap();
                    data.lower_messages_for_endpoint(&lowerer).unwrap();
                    data.precompute_body_plans(endpoint.as_ref(), "primary-model")
                        .unwrap();

                    let overrides = match overrides_variant {
                        0 => Overrides::new(),
                        1 => {
                            let mut overrides = Overrides::new();
                            overrides.set_model("override-model");
                            overrides.set_stream(true);
                            overrides
                        }
                        // `max_tokens` is read before `max_completion_tokens`, so a
                        // plan carrying the latter shadows this override.
                        _ => {
                            let mut overrides = Overrides::new();
                            overrides.set_max_tokens("max_tokens", 99);
                            overrides
                        }
                    };

                    let mut session =
                        ConversationSession::new(Arc::new(data), SessionId::from("session"))
                            .unwrap();
                    session.advance_to(0).unwrap();
                    let request = session
                        .materialize_prepared(
                            &EndpointRequestMaterializer,
                            endpoint.as_ref(),
                            "primary-model",
                            CreditPhase::Profiling,
                            &overrides,
                        )
                        .unwrap();
                    let body: Value =
                        serde_json::from_slice(&request.body.to_wire().unwrap()).unwrap();
                    let case = format!(
                        "endpoint={endpoint_id} max_tokens={with_max_tokens} overrides={overrides_variant}"
                    );

                    if let Some(model) = body.get("model") {
                        assert_eq!(
                            *model,
                            Value::String(request.model.clone()),
                            "model: {case}"
                        );
                    }
                    if let Some(stream) = body.get("stream") {
                        assert_eq!(*stream, Value::Bool(request.streaming), "stream: {case}");
                    }
                    // The reported cap is the last of these the body declares.
                    let dispatched_cap =
                        ["max_tokens", "max_completion_tokens", "max_output_tokens"]
                            .iter()
                            .filter_map(|field| body.get(*field))
                            .next_back();
                    if let Some(cap) = dispatched_cap {
                        assert_eq!(
                            *cap,
                            Value::from(request.max_tokens.unwrap()),
                            "max tokens: {case}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn warmup_phase_falls_back_to_live_format_not_profiling_cache() {
        // Warmup folds the conversation system prompt while live-formatting the
        // first message. Profiling now preserves the same system-prompt semantics,
        // so byte inequality no longer distinguishes the two paths; the test-only
        // hit counter proves warmup still never consumes the profiling cache.
        let mut pool = SegmentPool::new();
        let system = pool
            .intern_text(
                None,
                Role::from("system"),
                Bytes::from_static(b"be terse"),
                vec![1],
            )
            .unwrap();
        let system_turn_text = pool
            .intern_text(
                None,
                Role::from("system"),
                Bytes::from_static(b"base"),
                vec![2, 3],
            )
            .unwrap();
        let system_turn = Turn {
            role: Some(Role::from("system")),
            content: smallvec![ContentGroup {
                kind: MediaKind::Text,
                name: String::new(),
                handles: smallvec![system_turn_text],
                uuids: smallvec![],
            }],
            input_tokens: Some(1),
            ..Turn::default()
        };
        let mut conversation = Conversation::new("session");
        conversation.context_mode = Some(ConversationContextMode::MessageArrayWithResponses);
        conversation.system = Some(system);
        conversation.turns = vec![system_turn];
        let mut dataset = Dataset::new(
            vec![conversation],
            Arc::new(pool.freeze()),
            "sequential",
            ConversationContextMode::MessageArrayWithResponses,
        )
        .unwrap();
        let endpoint = prepare_endpoint("chat");
        let lowerer = ShapeLowerer::for_descriptor_id("chat").unwrap();
        dataset.lower_messages_for_endpoint(&lowerer).unwrap();
        dataset
            .precompute_body_plans(endpoint.as_ref(), "primary-model")
            .unwrap();
        assert!(
            dataset
                .cached_body_plan(&SessionId::from("session"), 0)
                .is_some()
        );
        let dataset = Arc::new(dataset);

        let mut warmup_session =
            ConversationSession::new(dataset.clone(), SessionId::from("session")).unwrap();
        warmup_session.advance_to(0).unwrap();
        let hits_before = CACHE_HITS.get();
        let warmup_body = warmup_session
            .materialize_prepared(
                &EndpointRequestMaterializer,
                endpoint.as_ref(),
                "primary-model",
                CreditPhase::Warmup,
                &Overrides::new(),
            )
            .unwrap()
            .body
            .to_wire()
            .unwrap();
        assert_eq!(CACHE_HITS.get(), hits_before);
        let profiling_body = dispatch_turn(dataset, endpoint.as_ref(), 0, &Overrides::new());
        assert_eq!(CACHE_HITS.get(), hits_before + 1);
        assert_eq!(warmup_body, profiling_body);
        let warmup: Value = serde_json::from_slice(&warmup_body).unwrap();
        assert_eq!(warmup["messages"][0]["content"], "be terse\n\nbase");
    }

    #[test]
    fn ineligible_turns_and_endpoints_are_never_cached() {
        // Token-native dispatch has no reusable message-array body plan to cache
        // (`precomputable_body() == false`).
        let endpoint_id = "vllm_generate";
        let mut pool = SegmentPool::new();
        let turn = text_turn(&mut pool, b"x", true, false);
        let mut dataset = single_conversation_dataset(
            ConversationContextMode::MessageArrayWithResponses,
            vec![turn],
            pool,
        );
        let endpoint = prepare_endpoint(endpoint_id);
        dataset
            .precompute_body_plans(endpoint.as_ref(), "primary-model")
            .unwrap();
        assert!(
            dataset
                .cached_body_plan(&SessionId::from("session"), 0)
                .is_none(),
            "endpoint {endpoint_id} must not be cached"
        );

        let mut pool = SegmentPool::new();
        let raw = pool
            .intern_raw(None, Bytes::from_static(br#"{"messages":[]}"#))
            .unwrap();
        let override_turn = Turn {
            endpoint: Some("responses".into()),
            content: smallvec![ContentGroup {
                kind: MediaKind::Text,
                name: String::new(),
                handles: smallvec![
                    pool.intern_text(None, Role::from("user"), Bytes::from_static(b"ov"), vec![1])
                        .unwrap()
                ],
                uuids: smallvec![],
            }],
            input_tokens: Some(1),
            ..Turn::default()
        };
        let raw_turn = Turn {
            body: Turn::dispatch_body(Some(raw), None, &[]),
            input_tokens: Some(1),
            ..Turn::default()
        };
        let mut dataset = single_conversation_dataset(
            ConversationContextMode::MessageArrayWithResponses,
            vec![override_turn, raw_turn],
            pool,
        );
        let endpoint = prepare_endpoint("chat");
        let lowerer = ShapeLowerer::for_descriptor_id("chat").unwrap();
        dataset.lower_messages_for_endpoint(&lowerer).unwrap();
        dataset
            .precompute_body_plans(endpoint.as_ref(), "primary-model")
            .unwrap();
        assert!(
            dataset
                .cached_body_plan(&SessionId::from("session"), 0)
                .is_none(),
            "per-turn endpoint override must not be cached"
        );
        assert!(
            dataset
                .cached_body_plan(&SessionId::from("session"), 1)
                .is_none(),
            "raw body turn must not be cached"
        );

        let mut pool = SegmentPool::new();
        let dag_turn = text_turn(&mut pool, b"graph", false, false);
        let mut conversation = Conversation::new("session");
        conversation.context_mode = Some(ConversationContextMode::MessageArrayWithResponses);
        conversation.turns = vec![dag_turn];
        conversation.dag = Some(crate::dataset::model::DagMetadata {
            branches: Default::default(),
            is_root: true,
            agent_depth: 0,
            parent_conversation_id: None,
            root_conversation_id: SessionId::from("session"),
        });
        let mut dataset = Dataset::new(
            vec![conversation],
            Arc::new(pool.freeze()),
            "sequential",
            ConversationContextMode::MessageArrayWithResponses,
        )
        .unwrap();
        let endpoint = prepare_endpoint("chat");
        let lowerer = ShapeLowerer::for_descriptor_id("chat").unwrap();
        dataset.lower_messages_for_endpoint(&lowerer).unwrap();
        dataset
            .precompute_body_plans(endpoint.as_ref(), "primary-model")
            .unwrap();
        assert!(
            dataset
                .cached_body_plan(&SessionId::from("session"), 0)
                .is_none(),
            "graph/DAG conversation must not be cached"
        );
    }

    #[test]
    fn cached_first_turn_byte_identical_for_input_array_and_without_responses_modes() {
        // Input-array dialects (embeddings, image_retrieval) and both WithoutResponses
        // context modes now precompute their response-independent first turn. The
        // cached plan must dispatch byte-identically to per-request formatting.
        fn check(endpoint_id: &str, mode: ConversationContextMode, pool: SegmentPool, turn: Turn) {
            let base = single_conversation_dataset(mode, vec![turn], pool);
            let endpoint = prepare_endpoint(endpoint_id);

            let mut cached_ds = base.clone();
            if let Some(lowerer) = ShapeLowerer::for_descriptor_id(endpoint.descriptor().id) {
                cached_ds.lower_messages_for_endpoint(&lowerer).unwrap();
            }
            cached_ds
                .precompute_body_plans(endpoint.as_ref(), "primary-model")
                .unwrap();

            let mut uncached_ds = base;
            if let Some(lowerer) = ShapeLowerer::for_descriptor_id(endpoint.descriptor().id) {
                uncached_ds.lower_messages_for_endpoint(&lowerer).unwrap();
            }

            let cached = Arc::new(cached_ds);
            let uncached = Arc::new(uncached_ds);
            assert!(
                cached
                    .cached_body_plan(&SessionId::from("session"), 0)
                    .is_some(),
                "expected cached first-turn plan: endpoint={endpoint_id} mode={mode:?}"
            );
            assert_eq!(
                dispatch_turn(cached, endpoint.as_ref(), 0, &Overrides::new()),
                dispatch_turn(uncached, endpoint.as_ref(), 0, &Overrides::new()),
                "cached first-turn body must match live formatting: endpoint={endpoint_id} mode={mode:?}"
            );
        }

        for mode in [
            ConversationContextMode::MessageArrayWithResponses,
            ConversationContextMode::DeltasWithoutResponses,
            ConversationContextMode::MessageArrayWithoutResponses,
        ] {
            let mut pool = SegmentPool::new();
            let turn = text_turn(&mut pool, b"embed me", false, false);
            check("embeddings", mode, pool, turn);

            let mut pool = SegmentPool::new();
            let text = pool
                .intern_text(
                    None,
                    Role::from("user"),
                    Bytes::from_static(b"look"),
                    vec![9],
                )
                .unwrap();
            let image = pool
                .intern_media(None, MediaKind::Image, Bytes::from_static(b"http://a"))
                .unwrap();
            check(
                "image_retrieval",
                mode,
                pool,
                content_turn(text, Some(image)),
            );

            let mut pool = SegmentPool::new();
            let turn = text_turn(&mut pool, b"chat me", false, false);
            check("chat", mode, pool, turn);
        }
    }

    #[test]
    fn same_text_different_media_turns_lower_to_distinct_segments() {
        let mut pool = SegmentPool::new();
        let text = pool
            .intern_text(
                None,
                Role::from("user"),
                Bytes::from_static(b"look"),
                vec![9],
            )
            .unwrap();
        let image_a = pool
            .intern_media(None, MediaKind::Image, Bytes::from_static(b"http://a"))
            .unwrap();
        let image_b = pool
            .intern_media(None, MediaKind::Image, Bytes::from_static(b"http://b"))
            .unwrap();
        let mut conversation = Conversation::new("session");
        conversation.context_mode = Some(ConversationContextMode::MessageArrayWithResponses);
        conversation.turns = vec![
            content_turn(text, Some(image_a)),
            content_turn(text, Some(image_b)),
        ];
        let mut dataset = Dataset::new(
            vec![conversation],
            Arc::new(pool.freeze()),
            "sequential",
            ConversationContextMode::MessageArrayWithResponses,
        )
        .unwrap();
        let lowerer = ShapeLowerer::for_descriptor_id("chat").unwrap();
        dataset.lower_messages_for_endpoint(&lowerer).unwrap();

        let turns = &dataset.conversations()[0].turns;
        // Same text, different media must not mis-dedup to one wire.
        assert_ne!(turns[0].body[0], turns[1].body[0]);
    }

    /// Build a `*WithoutResponses` continuation body, optionally with the
    /// dataset (and the captured reply) lowered to spliceable wires.
    ///
    /// `lower == false` is the reference: nothing carries `lowered`, so every
    /// turn — authored and captured alike — is rendered from its composed media
    /// on the live path, which is what the spliced path must reproduce byte for
    /// byte.
    ///
    /// `expect_lowered` is the control. `lower_messages_for_endpoint` answers
    /// `Ok` however many turns it actually lowered — including none — so without
    /// pinning the authored turn's post-lowering state, both arms could be the
    /// same unlowered path and every comparison below would hold vacuously.
    fn continuation_body(
        endpoint: &dyn PreparedEndpoint,
        mode: ConversationContextMode,
        turns: Vec<Turn>,
        pool: SegmentPool,
        phase: CreditPhase,
        lower: bool,
        expect_lowered: bool,
    ) -> Bytes {
        let mut dataset = single_conversation_dataset(mode, turns, pool);
        let lowerer = ShapeLowerer::for_descriptor_id(endpoint.descriptor().id).unwrap();
        if lower {
            dataset.lower_messages_for_endpoint(&lowerer).unwrap();
            assert_eq!(
                !dataset.conversations()[0].turns[0].body.is_empty(),
                expect_lowered,
                "lowering left the authored turn in the wrong state, so this \
                 fixture is not exercising the path it names"
            );
        }
        let mut session =
            ConversationSession::new(Arc::new(dataset), SessionId::from("session")).unwrap();
        session.advance_to(0).unwrap();
        let mut reply = EndpointTurn {
            role: Some("assistant".into()),
            texts: vec![Media::new(vec!["prior reply".to_string()])],
            ..EndpointTurn::default()
        };
        if lower {
            reply.lowered = Some(lowerer.lower_turn(&reply).unwrap());
        }
        session.capture_response(reply, 3, Some(0)).unwrap();
        session.advance_to(1).unwrap();
        session
            .materialize_prepared(
                &EndpointRequestMaterializer,
                endpoint,
                "primary-model",
                phase,
                &Overrides::new(),
            )
            .unwrap()
            .body
            .to_wire()
            .unwrap()
    }

    /// The spliced resolution drops a lowered turn's composed media and role
    /// because the formatter cannot reach them. If that were ever untrue the
    /// dispatched body would silently differ from the rendered one, so pin the
    /// two against each other across the message-array dialects, both phases,
    /// and both authored-turn shapes.
    ///
    /// Warmup is the load-bearing half: it re-renders the first turn from its
    /// media to fold the system prompt in, so it must stay off the spliced path.
    #[test]
    fn spliced_continuation_body_matches_the_rendered_one() {
        for endpoint_id in ["chat", "responses", "messages"] {
            let endpoint = prepare_endpoint(endpoint_id);
            for phase in [CreditPhase::Profiling, CreditPhase::Warmup] {
                for with_max_tokens in [false, true] {
                    let build = |lower: bool| {
                        let mut pool = SegmentPool::new();
                        let turns = vec![
                            text_turn(&mut pool, b"hello world", with_max_tokens, false),
                            text_turn(&mut pool, b"second turn", with_max_tokens, false),
                        ];
                        continuation_body(
                            endpoint.as_ref(),
                            ConversationContextMode::DeltasWithoutResponses,
                            turns,
                            pool,
                            phase,
                            lower,
                            true,
                        )
                    };
                    assert_eq!(
                        build(true),
                        build(false),
                        "spliced/rendered divergence: endpoint={endpoint_id} phase={phase:?} \
                         max_tokens={with_max_tokens}"
                    );
                }
            }
        }
    }

    /// `MessageArrayWithoutResponses` carries authored `raw_messages` snapshots,
    /// which `turn_is_lowerable` excludes from lowering — so its continuation
    /// turns never reach the spliced resolution and `merge_message_array_snapshots`
    /// still prefix-diffs the same rendered turns it always did.
    #[test]
    fn authored_snapshot_turns_are_unaffected_by_lowering() {
        let endpoint = prepare_endpoint("chat");
        let build = |lower: bool| {
            let mut pool = SegmentPool::new();
            let mut snapshot = |messages: Value| {
                let handle = pool
                    .intern_raw(None, Bytes::from(serde_json::to_vec(&messages).unwrap()))
                    .unwrap();
                Turn {
                    role: Some(Role::from("user")),
                    raw_messages: Some(handle),
                    input_tokens: Some(2),
                    ..Turn::default()
                }
            };
            let turns = vec![
                snapshot(serde_json::json!([{"role": "user", "content": "first"}])),
                snapshot(serde_json::json!([
                    {"role": "user", "content": "first"},
                    {"role": "user", "content": "second"},
                ])),
            ];
            continuation_body(
                endpoint.as_ref(),
                ConversationContextMode::MessageArrayWithoutResponses,
                turns,
                pool,
                CreditPhase::Profiling,
                lower,
                // The control, inverted: these authored `raw_messages` snapshots
                // must come through the lowering pass untouched, which is the
                // whole claim this test makes.
                false,
            )
        };
        let body = build(true);
        assert_eq!(body, build(false));
        let parsed: Value = serde_json::from_slice(&body).unwrap();
        // Turn 0's snapshot, the captured reply, then turn 1's one-message delta.
        assert_eq!(parsed["messages"].as_array().unwrap().len(), 3);
        assert_eq!(parsed["messages"][1]["content"], "prior reply");
    }
}
