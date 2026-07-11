// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Accuracy benchmark errors.

use std::fmt::{self, Display};
use std::path::PathBuf;

/// Errors produced while loading, validating, or formatting an accuracy benchmark.
#[derive(Debug)]
pub enum AccuracyError {
    /// A dataset file could not be read.
    ReadDataset {
        /// Dataset file path.
        path: PathBuf,
        /// Underlying error text.
        message: String,
    },
    /// A dataset file was not valid JSON or JSON Lines.
    ParseDataset {
        /// Dataset file path or source label.
        source: String,
        /// Underlying parser error text.
        message: String,
    },
    /// A row violates the benchmark's schema or invariants.
    InvalidRow {
        /// Question id when it was available.
        question_id: Option<u64>,
        /// Actionable validation message.
        message: String,
    },
    /// A requested category/task is not present in the benchmark.
    UnknownTask {
        /// User-provided task.
        task: String,
        /// Supported task names.
        available: Vec<String>,
    },
    /// A requested benchmark is not registered.
    UnknownBenchmark {
        /// User-provided name.
        name: String,
        /// Canonical registered names.
        available: Vec<String>,
    },
    /// A requested grader is not registered.
    UnknownGrader {
        /// User-provided name.
        name: String,
        /// Canonical registered names.
        available: Vec<String>,
    },
    /// A registry entry has an empty, repeated, or otherwise invalid name.
    InvalidRegistration {
        /// Registry category being populated.
        category: &'static str,
        /// Actionable validation message.
        message: String,
    },
    /// A canonical name or alias conflicts with an existing registry entry.
    DuplicateRegistration {
        /// Registry category being populated.
        category: &'static str,
        /// Normalized conflicting name.
        name: String,
    },
    /// No problems remained after applying the benchmark configuration.
    EmptySelection(String),
    /// A grader received invalid ground truth.
    InvalidGroundTruth(String),
    /// A benchmark configuration would diverge from its reference protocol.
    UnsupportedConfiguration(String),
    /// A grader's isolated execution backend failed.
    GraderExecution(String),
}

impl Display for AccuracyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ReadDataset { path, message } => {
                write!(
                    f,
                    "could not read accuracy dataset {}: {message}",
                    path.display()
                )
            }
            Self::ParseDataset { source, message } => {
                write!(f, "could not parse accuracy dataset {source}: {message}")
            }
            Self::InvalidRow {
                question_id,
                message,
            } => match question_id {
                Some(id) => write!(
                    f,
                    "invalid accuracy dataset row question_id={id}: {message}"
                ),
                None => write!(f, "invalid accuracy dataset row: {message}"),
            },
            Self::UnknownTask { task, available } => write!(
                f,
                "unknown accuracy task {task:?}; available tasks: {}",
                available.join(", ")
            ),
            Self::UnknownBenchmark { name, available } => write!(
                f,
                "unknown accuracy benchmark {name:?}; available benchmarks: {}",
                available.join(", ")
            ),
            Self::UnknownGrader { name, available } => write!(
                f,
                "unknown accuracy grader {name:?}; available graders: {}",
                available.join(", ")
            ),
            Self::InvalidRegistration { category, message } => {
                write!(f, "invalid {category} registration: {message}")
            }
            Self::DuplicateRegistration { category, name } => {
                write!(f, "duplicate {category} registration {name:?}")
            }
            Self::EmptySelection(message) => {
                write!(f, "accuracy benchmark selected no problems: {message}")
            }
            Self::InvalidGroundTruth(message) => {
                write!(f, "invalid accuracy ground truth: {message}")
            }
            Self::UnsupportedConfiguration(message) => {
                write!(f, "unsupported accuracy configuration: {message}")
            }
            Self::GraderExecution(message) => {
                write!(f, "accuracy grader execution failed: {message}")
            }
        }
    }
}

impl std::error::Error for AccuracyError {}
