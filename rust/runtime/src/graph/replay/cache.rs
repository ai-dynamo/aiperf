// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Profiling-only cache isolation for recorded-agent replay.

use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;

use crate::rng::RngRoot;
use serde_json::Value;

const NAMESPACE_SUFFIX: &str = " Performance replay cache namespace. Ignore the digits above.\n\n";

/// Opaque controller-minted identity for one replay run.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReplayRunIdentity {
    root: RngRoot,
    label: String,
    checkpoint: Option<ReplayCheckpointIdentity>,
}

/// Secret-bearing replay identity retained only by protected controller state.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ReplayCheckpointIdentity {
    pub(crate) run_id: String,
    pub(crate) replay_root_digest: String,
    pub(crate) manifest_digest: String,
    pub(crate) recording_digests: BTreeMap<String, String>,
    pub(crate) request_profile_digests: BTreeMap<String, String>,
    pub(crate) cache_namespace: String,
}

impl ReplayRunIdentity {
    /// Mint an opaque, nonempty run identity from the controller's run label.
    #[must_use]
    pub fn mint(root: RngRoot, run_label: &str) -> Self {
        let label = run_label.trim();
        Self {
            root,
            label: if label.is_empty() {
                "recorded-agent-replay".to_string()
            } else {
                label.to_string()
            },
            checkpoint: None,
        }
    }

    /// Construct the controller-owned identity persisted in a replay checkpoint.
    ///
    /// The cache namespace remains opaque here; callers must use provenance
    /// redaction before publishing any identity-derived data.
    #[must_use]
    pub fn for_checkpoint(
        run_id: impl Into<String>,
        replay_root_digest: impl Into<String>,
        manifest_digest: impl Into<String>,
        recording_digests: BTreeMap<String, String>,
        request_profile_digests: BTreeMap<String, String>,
        cache_namespace: impl Into<String>,
    ) -> Self {
        let run_id = run_id.into();
        Self {
            root: RngRoot::new(None),
            label: run_id.clone(),
            checkpoint: Some(ReplayCheckpointIdentity {
                run_id,
                replay_root_digest: replay_root_digest.into(),
                manifest_digest: manifest_digest.into(),
                recording_digests,
                request_profile_digests,
                cache_namespace: cache_namespace.into(),
            }),
        }
    }

    /// Return the persisted opaque label that scopes replay-owned cleanup.
    #[must_use]
    pub fn label(&self) -> &str {
        &self.label
    }

    /// Return replay-root digest retained by a persistent run identity.
    #[must_use]
    pub fn replay_root_digest(&self) -> Option<&str> {
        self.checkpoint
            .as_ref()
            .map(|identity| identity.replay_root_digest.as_str())
    }

    /// Return manifest digest retained by a persistent run identity.
    #[must_use]
    pub fn manifest_digest(&self) -> Option<&str> {
        self.checkpoint
            .as_ref()
            .map(|identity| identity.manifest_digest.as_str())
    }

    /// Return source recording digests retained by a persistent run identity.
    #[must_use]
    pub fn recording_digests(&self) -> &BTreeMap<String, String> {
        static EMPTY: std::sync::OnceLock<BTreeMap<String, String>> = std::sync::OnceLock::new();
        self.checkpoint.as_ref().map_or_else(
            || EMPTY.get_or_init(BTreeMap::new),
            |identity| &identity.recording_digests,
        )
    }

    /// Return resolved request-profile digests retained by a persistent run identity.
    #[must_use]
    pub fn request_profile_digests(&self) -> &BTreeMap<String, String> {
        static EMPTY: std::sync::OnceLock<BTreeMap<String, String>> = std::sync::OnceLock::new();
        self.checkpoint.as_ref().map_or_else(
            || EMPTY.get_or_init(BTreeMap::new),
            |identity| &identity.request_profile_digests,
        )
    }

    /// Return the publication-safe digest of the protected cache namespace.
    #[must_use]
    pub fn cache_namespace_digest(&self) -> Option<String> {
        self.checkpoint_identity().map(|identity| {
            blake3::hash(identity.cache_namespace.as_bytes())
                .to_hex()
                .to_string()
        })
    }

    pub(crate) fn checkpoint_identity(&self) -> Option<&ReplayCheckpointIdentity> {
        self.checkpoint.as_ref()
    }
}

/// Wire message encoding used by the endpoint formatter.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReplayMessageDialect {
    /// OpenAI Chat Completions messages.
    OpenAiChat,
    /// OpenAI Responses input items.
    OpenAiResponses,
}

/// A profiling-only cache isolation policy.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum CacheIsolationPolicy {
    /// Preserve recorded request bytes.
    #[default]
    None,
    /// Prefix exactly the first profiling message without mutating stored messages.
    FirstMessagePrefix { namespace: String },
}

impl CacheIsolationPolicy {
    /// Create one replay-run-scoped first-message prefix policy.
    #[must_use]
    pub fn first_message_prefix(identity: ReplayRunIdentity) -> Self {
        let mut generator = identity.root.derive(&format!(
            "recorded-agent-replay-cache-namespace:{}",
            identity.label
        ));
        let digits = (0..32)
            .map(|_| (generator.random_u64() % 10).to_string())
            .collect::<Vec<_>>()
            .join(" ");
        Self::FirstMessagePrefix {
            namespace: format!("{digits}{NAMESPACE_SUFFIX}"),
        }
    }

    /// Return the profiling prefix when cache isolation is enabled.
    #[must_use]
    pub fn namespace(&self) -> Option<&str> {
        match self {
            Self::None => None,
            Self::FirstMessagePrefix { namespace } => Some(namespace),
        }
    }

    /// Return warmup messages unchanged.
    pub fn apply_warmup(
        &self,
        messages: &[Value],
        _dialect: ReplayMessageDialect,
    ) -> Result<Vec<Value>, ReplayCacheError> {
        Ok(messages.to_vec())
    }

    /// Apply the run-scoped prefix to profiling messages only.
    pub fn apply_profiling(
        &self,
        messages: &[Value],
        dialect: ReplayMessageDialect,
    ) -> Result<Vec<Value>, ReplayCacheError> {
        match self.namespace() {
            Some(namespace) => apply_first_message_prefix(messages, namespace, dialect),
            None => Ok(messages.to_vec()),
        }
    }
}

/// Failure to prefix the first replay message without changing its wire shape.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReplayCacheError(String);

impl fmt::Display for ReplayCacheError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl Error for ReplayCacheError {}

/// Clone `messages` and prefix the first message's supported content shape.
pub fn apply_first_message_prefix(
    messages: &[Value],
    namespace: &str,
    dialect: ReplayMessageDialect,
) -> Result<Vec<Value>, ReplayCacheError> {
    let Some(first) = messages.first() else {
        return Err(ReplayCacheError(
            "recorded replay has no messages to prefix".to_string(),
        ));
    };
    let mut first = first.clone();
    let object = first.as_object_mut().ok_or_else(|| {
        ReplayCacheError("recorded replay's first message must be an object".to_string())
    })?;
    let content = object.get_mut("content").ok_or_else(|| {
        ReplayCacheError("recorded replay's first message has no content field".to_string())
    })?;
    match content {
        Value::String(text) => *text = format!("{namespace}{text}"),
        Value::Null => *content = Value::String(namespace.to_string()),
        Value::Array(items) if dialect == ReplayMessageDialect::OpenAiResponses => {
            items.insert(
                0,
                serde_json::json!({"type": "input_text", "text": namespace}),
            );
        }
        Value::Array(_) => {
            return Err(ReplayCacheError(
                "structured recorded replay content requires the OpenAI Responses dialect"
                    .to_string(),
            ));
        }
        _ => {
            return Err(ReplayCacheError(
                "recorded replay's first message content must be string, array, or null"
                    .to_string(),
            ));
        }
    }
    let mut transformed = Vec::with_capacity(messages.len());
    transformed.push(first);
    transformed.extend_from_slice(&messages[1..]);
    Ok(transformed)
}
