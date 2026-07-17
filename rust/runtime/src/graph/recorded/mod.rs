// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Native recorded-trace adapters shared by WEKA and Dynamo inputs.
//!
//! Both formats normalize into one [`RecordedRequest`] list and traverse the
//! same content-parent, idle-warp, interval-order, message, and segment path.
//! Format selection remains runner-owned; this module contains no registry.

mod aiperf_trace;
mod coding;
mod content;
mod dynamo;
mod scalar;
mod source;
mod trie;
mod weka;

use std::error::Error;
use std::fmt::{self, Display};

use crate::dataset::LoadConfig;

pub use aiperf_trace::compile_aiperf_trace_input;
pub use dynamo::compile_dynamo_trace_input;
pub use source::{RecordedTracePathKind, enumerate_recorded_trace_files};
pub use weka::compile_weka_trace_input;

/// Opaque recorded cache-block identity.
///
/// Dynamo/WEKA captures record u64 cache-block hashes, and Dynamo mints small
/// negative virtual identities for non-replay turns; `i128` losslessly covers
/// that entire domain as an allocation-free `Copy` machine integer.
/// Decimal display preserves byte-exact content-seed derivation.
pub(crate) type BlockHash = i128;

/// Corpus used to reconstruct recorded token blocks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PromptCorpus {
    /// Procedural coding/tool transcript corpus used by recorded production traces.
    Coding,
    /// Deterministically chunked Shakespeare corpus used by parity fixtures.
    Sonnet,
}

impl PromptCorpus {
    /// Parse the strict Config-v2 spelling.
    pub fn parse(value: &str) -> Result<Self, RecordedTraceError> {
        match value {
            "coding" => Ok(Self::Coding),
            "sonnet" => Ok(Self::Sonnet),
            other => Err(RecordedTraceError(format!(
                "recorded graph prompt corpus must be \"coding\" or \"sonnet\", got {other:?}"
            ))),
        }
    }
}

/// Fully resolved source and replay policy for one recorded format compiler.
pub struct RecordedTraceInputConfig {
    /// Local, inline, URL, or Hugging Face source.
    pub load: LoadConfig,
    /// First-N eligible trace/tree cap, applied after context filtering.
    pub root_limit: Option<usize>,
    /// Peak per-request input-plus-output ceiling.
    pub max_context_length: Option<usize>,
    /// WEKA top-level output cap; Dynamo deliberately ignores this value.
    pub max_osl: Option<usize>,
    /// True-idle gap cap. `None` replays raw recorded gaps.
    pub idle_gap_cap_seconds: Option<f64>,
    /// Content corpus selected by Config v2.
    pub prompt_corpus: PromptCorpus,
    /// Concrete run/dataset seed used for byte-stable content synthesis.
    pub content_root_seed: u64,
}

impl RecordedTraceInputConfig {
    pub(crate) fn validate(&self) -> Result<(), RecordedTraceError> {
        if self.root_limit == Some(0) {
            return Err(RecordedTraceError(
                "recorded graph root limit must be positive when configured".into(),
            ));
        }
        if self.max_context_length == Some(0) || self.max_osl == Some(0) {
            return Err(RecordedTraceError(
                "recorded graph token caps must be positive when configured".into(),
            ));
        }
        if self
            .idle_gap_cap_seconds
            .is_some_and(|value| !value.is_finite() || value < 0.0)
        {
            return Err(RecordedTraceError(
                "recorded graph idle-gap cap must be finite and non-negative".into(),
            ));
        }
        Ok(())
    }
}

/// Focused parse, validation, synthesis, or lowering failure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecordedTraceError(pub String);

impl Display for RecordedTraceError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl Error for RecordedTraceError {}

impl From<crate::dataset::DatasetError> for RecordedTraceError {
    fn from(error: crate::dataset::DatasetError) -> Self {
        Self(error.to_string())
    }
}

impl From<serde_json::Error> for RecordedTraceError {
    fn from(error: serde_json::Error) -> Self {
        Self(error.to_string())
    }
}
