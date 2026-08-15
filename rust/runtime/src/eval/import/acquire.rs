// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

//! Immutable source locations and injected source acquisition.

use super::HarborImportError;

/// An immutable Harbor-compatible package source reference.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum HarborSource {
    /// A package supplied from a caller-controlled local source location.
    Local(String),
    /// A package at a pinned Git revision.
    PinnedGit {
        repository: String,
        revision: String,
    },
    /// An immutable registry package reference.
    Registry(String),
}

impl HarborSource {
    /// Creates a nonempty local source reference.
    pub fn local(location: impl Into<String>) -> Result<Self, HarborImportError> {
        let location = location.into();
        if location.trim().is_empty() {
            return Err(HarborImportError::InvalidSource("local location"));
        }
        Ok(Self::Local(location))
    }

    /// Returns the stable source location key passed to an acquirer.
    pub fn location(&self) -> &str {
        match self {
            Self::Local(location) | Self::Registry(location) => location,
            Self::PinnedGit { repository, .. } => repository,
        }
    }
}

/// Copies source package bytes into the native importer without provider coupling.
pub trait SourceAcquirer {
    /// Acquires the exact source bytes identified by a source reference.
    fn acquire(&self, source: &HarborSource) -> Result<Vec<u8>, HarborImportError>;
}
