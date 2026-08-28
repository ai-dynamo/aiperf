// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Single-turn construction seam for streaming action bindings.
//!
//! A streaming `session_request.v1` action already carries the exact request
//! bytes its format decoded, so the streaming action binding has no
//! conversation template to sample and no history to walk — it needs only to
//! wrap those bytes in the [`TurnToSend`] the scheduled issuer accepts.
//!
//! That wrapping cannot happen in the streaming module: `TurnToSend::session`
//! is `pub(super)` and [`RuntimeSessionBackend`] is private to `multiturn`, so
//! the type is constructible only from inside this subtree. This module is the
//! one narrow, explicitly single-turn entry point rather than a widening of
//! either visibility, which would let any caller fabricate a session handle.
//!
//! The backend deliberately refuses every multi-turn operation. A streaming
//! session's continuation is authored by its session program and arrives as a
//! separate action with its own stable identity; silently walking a second turn
//! here would issue a request the session plane never ordered.

use std::{collections::BTreeMap, rc::Rc};

use anyhow::{Result, bail};
use uuid::Uuid;

use super::model::{
    RuntimeSessionBackend, SampledSession, TurnDataPolicy, TurnEndpoint, TurnMetadata,
    TurnResponse, TurnToSend,
};
use crate::body_plan::RequestBody;

/// Authored inputs for one streaming action turn.
#[derive(Debug)]
pub struct StreamingActionTurn {
    /// Stable session identity, reused as the conversation identity.
    pub session_id: String,
    /// Correlation identity sent to the backend.
    pub correlation_id: String,
    /// Prepared endpoint the action is bound to.
    pub endpoint: TurnEndpoint,
    /// Exact request bytes the streaming format decoded.
    pub body: RequestBody,
    /// Effective wire model, when the action selected one.
    pub effective_model: Option<String>,
    /// Authored input-token accounting for issuer-side admission.
    pub input_length: usize,
    /// Requested output-token cap.
    pub max_output_tokens: usize,
    /// Whether the endpoint should stream its response.
    pub is_streaming: bool,
}

/// Session backend for a streaming action, which is always exactly one turn.
#[derive(Debug)]
struct StreamingActionBackend;

impl RuntimeSessionBackend for StreamingActionBackend {
    fn available_turns(&self) -> usize {
        1
    }

    fn build_first_turn(
        &self,
        _owner: &SampledSession,
        _max_turns: Option<usize>,
    ) -> Result<TurnToSend> {
        bail!("a streaming action turn is materialized by its action binding, not rebuilt")
    }

    fn build_turn_at(
        &self,
        _owner: &SampledSession,
        _start_index: usize,
        _max_turns: Option<usize>,
    ) -> Result<TurnToSend> {
        bail!("a streaming action turn is materialized by its action binding, not rebuilt")
    }

    fn next_metadata(&self, _turn_index: usize) -> Result<TurnMetadata> {
        bail!("a streaming action carries no successor turn metadata")
    }

    fn build_next_turn(
        &self,
        _owner: &SampledSession,
        _current: &TurnToSend,
        _response: TurnResponse,
    ) -> Result<TurnToSend> {
        bail!("a streaming session's continuation arrives as its own ordered action")
    }
}

/// Build one schedulable turn from a streaming action's decoded request.
///
/// The turn is complete: `deferred_body` is false because the bytes are already
/// present, so no worker-side materialization step can substitute a different
/// body for the one the action's content lease was charged for.
#[must_use]
pub fn streaming_action_turn(authored: StreamingActionTurn) -> TurnToSend {
    let session = SampledSession {
        conversation_id: authored.session_id.clone(),
        x_correlation_id: authored.correlation_id.clone(),
        backend: Rc::new(StreamingActionBackend),
    };
    TurnToSend {
        uuid: Uuid::new_v4(),
        effective_model: authored.effective_model,
        conversation_id: authored.session_id,
        x_correlation_id: authored.correlation_id.clone(),
        request_correlation_id: authored.correlation_id,
        turn_index: 0,
        num_turns: 1,
        input_length: authored.input_length,
        max_output_tokens: authored.max_output_tokens,
        messages: Vec::new(),
        request_body: Some(authored.body),
        request_headers: BTreeMap::new(),
        request_parameters: BTreeMap::new(),
        endpoint_path: None,
        endpoint: authored.endpoint,
        streaming: authored.is_streaming,
        audio_duration_seconds: None,
        image_count: None,
        timestamp_ms: None,
        delay_ms: None,
        trace_hash_ids: None,
        raw_token_ids: None,
        data_policy: TurnDataPolicy::ordinary(),
        cancel_after_ns: None,
        url_index: None,
        deferred_body: false,
        session,
    }
}
