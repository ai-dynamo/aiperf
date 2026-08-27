// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Boundary observation values for scheduled turn dispatch.

use std::task::{Context, Poll};

use anyhow::Result;
use serde_json::Value;

use crate::dispatch::collector::ReplayTerminalStatus;
use crate::endpoints::ParsedResponse;
use crate::metrics_core::RequestTrace;

/// Backpressured endpoint-normalized response-frame consumer.
///
/// HTTP invokes this callback on the local reactor as each decoded SSE event
/// arrives. The poll/send split reserves bounded downstream capacity without
/// blocking a current-thread reactor or allocating a future per frame. Raw SSE
/// bytes never cross this seam.
pub trait TurnResponseObserver {
    /// Reserve capacity for the next endpoint-parsed frame.
    fn poll_ready(&self, context: &mut Context<'_>) -> Poll<Result<()>>;

    /// Send one frame after [`Self::poll_ready`] returned ready.
    fn start_send(&self, response: ParsedResponse) -> Result<()>;
}

/// Endpoint-normalized assistant and terminal metadata retained by the normal
/// dispatch path.
///
/// Scheduled workloads that only need continuation text can keep using
/// [`TurnDispatchOutcome::response_text`]. Stateful consumers use this richer
/// record to preserve reasoning, truncation, provider correlation, cache usage,
/// and infrastructure failures without reparsing transport payloads.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ModelResponseMetadata {
    /// User-visible assistant content without a separate reasoning channel.
    pub content: Option<String>,
    /// Provider-emitted reasoning content, when the endpoint distinguishes it.
    pub reasoning: Option<String>,
    /// Prompt tokens served from a provider cache.
    pub cached_prompt_tokens: Option<u64>,
    /// Provider response identifier used by stateful APIs and artifacts.
    pub response_id: Option<String>,
    /// Endpoint-normalized finish reason, such as `stop` or `length`.
    pub finish_reason: Option<String>,
    /// Exact generated token IDs from a token-native non-text response.
    pub output_token_ids: Option<Vec<u32>>,
    /// Reassembled OpenAI-compatible assistant message, including tool calls.
    pub assistant_message: Option<Value>,
    /// Stable transport/provider failure category for non-completed requests.
    pub error_kind: Option<String>,
    /// Human-readable transport/provider failure detail.
    pub error_message: Option<String>,
}

/// Terminal result returned by a scheduled turn dispatcher.
#[derive(Clone, Debug)]
pub struct TurnDispatchOutcome {
    /// Clock timestamp at which transport/backend dispatch began.
    pub start_ns: i64,
    /// Clock timestamp at which dispatch reached terminal.
    pub end_ns: i64,
    /// Terminal classification emitted to the measurement observer.
    pub terminal: ReplayTerminalStatus,
    /// Assistant text captured for the next turn's dynamic prompt splice.
    pub response_text: String,
    /// Rich model-response metadata captured by the ordinary endpoint parser.
    pub model_response: ModelResponseMetadata,
    /// Authoritative server prompt-token usage, when available.
    pub prompt_tokens: Option<u64>,
    /// Authoritative server completion-token usage, when available.
    pub completion_tokens: Option<u64>,
    /// Fine-grained transport metrics, when the backend supplies them.
    pub http: RequestTrace,
}
