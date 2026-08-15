// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

//! Offline-safe references to immutable evaluation manifests.

use crate::eval::EvalTaskRef;

/// An immutable local evaluation manifest that requires no registry connection.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RegistryReference {
    /// Local manifest identity.
    pub id: String,
    /// Exact selected tasks.
    pub tasks: Vec<EvalTaskRef>,
}

impl RegistryReference {
    /// Creates a local manifest reference.
    pub fn local(id: impl Into<String>, tasks: Vec<EvalTaskRef>) -> Result<Self, RegistryError> {
        let id = id.into();
        if id.trim().is_empty() || tasks.is_empty() {
            return Err(RegistryError::InvalidLocalManifest);
        }
        Ok(Self { id, tasks })
    }

    /// Local references are valid without an online registry.
    pub const fn is_offline_valid(&self) -> bool {
        true
    }
}

/// Invalid local registry-free manifest reference.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RegistryError {
    /// A local reference needs an identity and at least one selected task.
    InvalidLocalManifest,
}

impl std::fmt::Display for RegistryError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("local registry reference requires an id and task selection")
    }
}

impl std::error::Error for RegistryError {}
