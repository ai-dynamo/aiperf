// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Handle-only conversation and turn domain model.
//!
//! The fields preserve every Python dataset value that affects dispatch,
//! scheduling, context reconstruction, multimodal metrics,
//! or DAG reporting. Large or wire-sensitive values are always [`Handle`]s;
//! projection into metadata therefore never copies or strips media bytes.

use std::fmt::{self, Display};

use serde::{Deserialize, Serialize};
use smallvec::SmallVec;

use crate::segment::{Handle, Role};

macro_rules! string_id {
    ($(#[$meta:meta])* $name:ident) => {
        $(#[$meta])*
        #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
        #[serde(transparent)]
        pub struct $name(String);

        impl $name {
            /// Construct an identifier from an owned or borrowed string.
            pub fn new(value: impl Into<String>) -> Self {
                Self(value.into())
            }

            /// Borrow the identifier text.
            pub fn as_str(&self) -> &str {
                &self.0
            }
        }

        impl From<String> for $name {
            fn from(value: String) -> Self {
                Self::new(value)
            }
        }

        impl From<&str> for $name {
            fn from(value: &str) -> Self {
                Self::new(value)
            }
        }

        impl Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str(&self.0)
            }
        }
    };
}

string_id!(
    /// Authored dataset conversation identifier.
    SessionId
);
string_id!(
    /// Per-turn model override.
    ModelId
);
string_id!(
    /// Graph node identifier retained from DAG authoring.
    NodeId
);
string_id!(
    /// Opaque DAG branch identifier.
    BranchId
);

/// Supported content categories.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MediaKind {
    /// Plain text.
    Text,
    /// Image data or URL.
    Image,
    /// Audio data or URL.
    Audio,
    /// Video data or URL.
    Video,
}

impl MediaKind {
    /// Stable lowercase name used in hashing and endpoint conversion.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Text => "text",
            Self::Image => "image",
            Self::Audio => "audio",
            Self::Video => "video",
        }
    }
}

/// How authored turns and live responses accumulate into request context.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConversationContextMode {
    /// Turns are deltas; live inference replies are appended between later turns.
    #[default]
    DeltasWithoutResponses,
    /// Turns are deltas containing authored responses; live replies are not appended.
    DeltasWithResponses,
    /// Every turn is a self-contained message array containing authored responses.
    MessageArrayWithResponses,
    /// Every turn is a self-contained user-only message array and live replies are
    /// merged before the next dispatch.
    MessageArrayWithoutResponses,
}

/// DAG child-context behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConversationBranchMode {
    /// Child inherits its parent's materialized context.
    Fork,
    /// Child starts with fresh context.
    Spawn,
}

/// When a DAG branch begins relative to its declaring parent turn.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DispatchTiming {
    /// Dispatch before the parent's first turn.
    Pre,
    /// Dispatch after the declaring turn completes.
    #[default]
    Post,
}

/// Condition that can gate an authored turn.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PrerequisiteKind {
    /// Every selected child from a spawn branch has completed.
    SpawnJoin,
    /// One or more named child sessions have completed.
    ChildSessionComplete,
    /// A relative timer has elapsed.
    Timer,
    /// A named external event was emitted.
    ExternalEvent,
    /// Participants sharing a barrier identifier have arrived.
    Barrier,
}

/// A condition attached to the turn that consumes it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TurnPrerequisite {
    /// Prerequisite behavior.
    pub kind: PrerequisiteKind,
    /// Branch whose children participate, when branch-scoped.
    pub branch_id: Option<BranchId>,
    /// Optional child subset; absent means every child in the branch.
    pub child_conversation_ids: SmallVec<[SessionId; 1]>,
    /// Shared barrier identifier for multi-parent synchronization.
    pub barrier_id: Option<String>,
    /// Timer duration in seconds.
    pub timer_seconds: Option<f64>,
    /// External event name.
    pub event_name: Option<String>,
}

/// A branch from one parent turn to child conversations.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConversationBranch {
    /// Opaque branch identifier referenced by turns and prerequisites.
    pub branch_id: BranchId,
    /// Authored child conversation identifiers.
    pub child_conversation_ids: SmallVec<[SessionId; 1]>,
    /// Inherited-context fork or fresh-context spawn.
    pub mode: ConversationBranchMode,
    /// Pre-session or post-turn dispatch.
    pub dispatch_timing: DispatchTiming,
    /// Whether a forked parent continues after creating its children.
    pub background: bool,
}

/// One named batch of same-kind content handles.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContentGroup {
    /// Endpoint field category.
    pub kind: MediaKind,
    /// Endpoint field name, such as `text`, `image_url`, or `input_audio`.
    pub name: String,
    /// Ordered batched content handles.
    pub handles: SmallVec<[Handle; 1]>,
}

/// Per-turn dispatch data; every potentially large value is a segment handle.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Turn {
    /// Optional authored role; endpoint adapters default missing roles as needed.
    pub role: Option<Role>,
    /// Per-turn model override.
    pub model: Option<ModelId>,
    /// Per-turn endpoint/dialect override.
    pub endpoint: Option<String>,
    /// Maximum generated tokens.
    pub max_tokens: Option<u32>,
    /// Authored streaming override. Raw payload replay derives this directly
    /// from the preserved body without rewriting it.
    pub streaming: Option<bool>,
    /// Load-time token count for this turn's authored request content, excluding
    /// tools inherited from another turn and live assistant replies.
    pub input_tokens: u64,
    /// Load-time token count for this turn's tool definitions, when present.
    pub tool_tokens: u64,
    /// Absolute authored timestamp in milliseconds.
    pub timestamp_ms: Option<f64>,
    /// Relative authored delay in milliseconds.
    pub delay_ms: Option<f64>,
    /// Source-trace cache identities, when the loader received `hash_ids`.
    #[serde(default)]
    pub trace_hash_ids: Option<Handle>,
    /// Ordered pre-serialized message handles for message-array inputs.
    pub messages: SmallVec<[Handle; 1]>,
    /// Named text and multimodal groups for endpoint formatting.
    pub content: SmallVec<[ContentGroup; 1]>,
    /// Complete prebuilt request body sent through the raw fast path.
    pub raw_payload: Option<Handle>,
    /// Exact validated input token IDs for token-native endpoint/backends.
    ///
    /// The IDs remain in the shared segment arena; this handle is the native
    /// replacement for the deprecated Python mmap-serialized list.
    #[serde(default)]
    pub raw_token_ids: Option<Handle>,
    /// Complete preformatted messages array.
    pub raw_messages: Option<Handle>,
    /// Preformatted tool definitions.
    pub tools: Option<Handle>,
    /// Preformatted vendor-shaped top-level system content blocks.
    ///
    /// Python parity: `src/aiperf/common/models/dataset_models.py:184-190`
    /// from PR 731.
    pub raw_system: Option<Handle>,
    /// Earliest prior turn included when walking backward for the most recent tools.
    pub tool_walk_start: Option<u32>,
    /// Per-turn extra request-body fields.
    pub extra_body: Option<Handle>,
    /// Per-turn HTTP headers.
    pub extra_headers: Option<Handle>,
    /// Endpoint-specific request parameters kept separate from body extras.
    pub request_parameters: Option<Handle>,
    /// DAG prerequisites attached to this consuming turn.
    pub prerequisites: SmallVec<[TurnPrerequisite; 0]>,
    /// DAG branches declared by this turn.
    pub branch_ids: SmallVec<[BranchId; 0]>,
    /// Audio duration used by ASR metrics such as RTFx.
    pub audio_duration_seconds: Option<f64>,
}

impl Default for Turn {
    fn default() -> Self {
        Self {
            role: None,
            model: None,
            endpoint: None,
            max_tokens: None,
            streaming: None,
            input_tokens: 0,
            tool_tokens: 0,
            timestamp_ms: None,
            delay_ms: None,
            trace_hash_ids: None,
            messages: SmallVec::new(),
            content: SmallVec::new(),
            raw_payload: None,
            raw_token_ids: None,
            raw_messages: None,
            tools: None,
            raw_system: None,
            tool_walk_start: None,
            extra_body: None,
            extra_headers: None,
            request_parameters: None,
            prerequisites: SmallVec::new(),
            branch_ids: SmallVec::new(),
            audio_duration_seconds: None,
        }
    }
}

/// Media-free scheduling and reporting projection of one turn.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TurnMetadata {
    /// Absolute authored timestamp in milliseconds.
    pub timestamp_ms: Option<f64>,
    /// Relative authored delay in milliseconds.
    pub delay_ms: Option<f64>,
    /// Source-trace cache identities retained without resolving prompt bytes.
    #[serde(default)]
    pub trace_hash_ids: Option<Handle>,
    /// Load-time token count for authored request content.
    pub input_tokens: u64,
    /// Declared branch identifiers.
    pub branch_ids: SmallVec<[BranchId; 0]>,
    /// Whether any declared branch is a fork.
    pub has_forks: bool,
    /// Turn prerequisites.
    pub prerequisites: SmallVec<[TurnPrerequisite; 0]>,
    /// Audio duration retained for ASR metrics without resolving audio bytes.
    pub audio_duration_seconds: Option<f64>,
}

/// DAG topology and lineage attached to a conversation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DagMetadata {
    /// Branch descriptors declared in this conversation.
    pub branches: SmallVec<[ConversationBranch; 0]>,
    /// Whether this conversation can be sampled as a root.
    pub is_root: bool,
    /// Static nesting depth, with roots at zero.
    pub agent_depth: u32,
    /// Direct parent conversation, if any.
    pub parent_conversation_id: Option<SessionId>,
    /// Root conversation for lineage-wide reporting.
    pub root_conversation_id: SessionId,
}

/// One complete authored conversation backed by a shared segment store.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Conversation {
    /// Authored stable dataset identifier.
    pub session_id: SessionId,
    /// Ordered authored turns.
    pub turns: Vec<Turn>,
    /// Optional shared system prompt.
    pub system: Option<Handle>,
    /// Optional per-conversation user context prompt.
    pub user_context: Option<Handle>,
    /// Conversation-specific context behavior; absent inherits dataset default.
    pub context_mode: Option<ConversationContextMode>,
    /// DAG topology and lineage.
    pub dag: Option<DagMetadata>,
}

impl Conversation {
    /// Construct an empty non-DAG conversation.
    pub fn new(session_id: impl Into<SessionId>) -> Self {
        Self {
            session_id: session_id.into(),
            turns: Vec::new(),
            system: None,
            user_context: None,
            context_mode: None,
            dag: None,
        }
    }

    /// Build the media-free metadata projection without resolving any handles.
    pub fn metadata(&self) -> ConversationMetadata {
        let branch_modes = self.dag.as_ref().map(|dag| {
            dag.branches
                .iter()
                .map(|branch| (&branch.branch_id, branch.mode))
                .collect::<std::collections::HashMap<_, _>>()
        });
        let turns = self
            .turns
            .iter()
            .map(|turn| TurnMetadata {
                timestamp_ms: turn.timestamp_ms,
                delay_ms: turn.delay_ms,
                trace_hash_ids: turn.trace_hash_ids,
                input_tokens: turn.input_tokens,
                has_forks: branch_modes.as_ref().is_some_and(|modes| {
                    turn.branch_ids
                        .iter()
                        .any(|id| modes.get(id).copied() == Some(ConversationBranchMode::Fork))
                }),
                branch_ids: turn.branch_ids.clone(),
                prerequisites: turn.prerequisites.clone(),
                audio_duration_seconds: turn.audio_duration_seconds,
            })
            .collect();
        ConversationMetadata {
            conversation_id: self.session_id.clone(),
            turns,
            context_mode: self.context_mode,
            dag: self.dag.clone(),
        }
    }
}

/// Media-free dataset view consumed by samplers, timing, and reporting.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConversationMetadata {
    /// Authored conversation identifier.
    pub conversation_id: SessionId,
    /// Per-turn timing and DAG shape.
    pub turns: Vec<TurnMetadata>,
    /// Conversation-specific context behavior.
    pub context_mode: Option<ConversationContextMode>,
    /// DAG topology and lineage.
    pub dag: Option<DagMetadata>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metadata_projects_forks_without_payload_bytes() {
        let branch_id = BranchId::from("root:0");
        let mut conversation = Conversation::new("root");
        conversation.turns.push(Turn {
            branch_ids: smallvec::smallvec![branch_id.clone()],
            audio_duration_seconds: Some(1.25),
            ..Turn::default()
        });
        conversation.dag = Some(DagMetadata {
            branches: smallvec::smallvec![ConversationBranch {
                branch_id,
                child_conversation_ids: smallvec::smallvec![SessionId::from("child")],
                mode: ConversationBranchMode::Fork,
                dispatch_timing: DispatchTiming::Post,
                background: false,
            }],
            is_root: true,
            agent_depth: 0,
            parent_conversation_id: None,
            root_conversation_id: SessionId::from("root"),
        });

        let metadata = conversation.metadata();
        assert!(metadata.turns[0].has_forks);
        assert_eq!(metadata.turns[0].audio_duration_seconds, Some(1.25));
    }
}
