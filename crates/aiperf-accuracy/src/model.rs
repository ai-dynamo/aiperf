// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Transport-neutral accuracy benchmark models.

use std::collections::BTreeMap;

use aiperf_metrics::{CorrelationId, TaskId};
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// A benchmark dataset split.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DatasetSplit {
    /// Training examples, used by benchmarks whose upstream evaluation split is train.
    Train,
    /// Development/few-shot examples.
    Dev,
    /// Few-shot/development examples.
    Validation,
    /// Scored examples.
    Test,
}

impl DatasetSplit {
    /// Stable split name used by files and remote dataset APIs.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Train => "train",
            Self::Dev => "dev",
            Self::Validation => "validation",
            Self::Test => "test",
        }
    }
}

/// One OpenAI-compatible chat message.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChatMessage {
    /// Message role (`system`, `user`, or `assistant`).
    pub role: String,
    /// Text content.
    pub content: String,
}

impl ChatMessage {
    /// Builds a system message.
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: "system".to_string(),
            content: content.into(),
        }
    }

    /// Builds a user message.
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: "user".to_string(),
            content: content.into(),
        }
    }

    /// Builds an assistant message used by multi-turn few-shot prompts.
    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: "assistant".to_string(),
            content: content.into(),
        }
    }
}

/// Generation settings required by an accuracy problem.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GenerationConfig {
    /// Maximum generated tokens.
    pub max_tokens: usize,
    /// Sampling temperature.
    pub temperature: f64,
    /// Nucleus-sampling probability.
    pub top_p: f64,
    /// Stop strings.
    pub stop: Vec<String>,
}

/// One fully materialized benchmark problem.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct BenchmarkProblem {
    /// Stable benchmark item id, used for logs and per-record export.
    pub id: String,
    /// Stable correlation id carried from workload construction through grading.
    pub correlation_id: CorrelationId,
    /// Task/category used for per-task rollups.
    pub task: TaskId,
    /// Preformatted chat messages placed on the wire.
    pub messages: Vec<ChatMessage>,
    /// Expected answer accepted by the grader.
    pub ground_truth: String,
    /// Generation settings for this problem.
    pub generation: GenerationConfig,
    /// Benchmark-specific metadata retained for per-record export.
    pub metadata: BTreeMap<String, Value>,
}
