// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

//! Offline-safe references to immutable evaluation manifests.

use std::collections::BTreeSet;

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
        let reference = Self {
            id: id.into(),
            tasks,
        };
        reference.validate_offline()?;
        Ok(reference)
    }

    /// Validates that this mutable manifest can run without registry access.
    pub fn validate_offline(&self) -> Result<(), RegistryError> {
        if self.id.trim().is_empty() {
            return Err(RegistryError::EmptyLocalManifestId);
        }
        if self.tasks.is_empty() {
            return Err(RegistryError::EmptyTaskSelection);
        }

        let mut task_ids = BTreeSet::new();
        for task in &self.tasks {
            if !task_ids.insert(task.id.as_str()) {
                return Err(RegistryError::DuplicateTaskId(task.id.as_str().to_owned()));
            }
        }
        Ok(())
    }

    /// Reports whether this local manifest remains valid without registry access.
    pub fn is_offline_valid(&self) -> bool {
        self.validate_offline().is_ok()
    }
}

/// Invalid local registry-free manifest reference.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RegistryError {
    /// A local reference needs an identity and at least one selected task.
    InvalidLocalManifest,
    /// The local manifest identity was empty.
    EmptyLocalManifestId,
    /// A local manifest must select at least one task.
    EmptyTaskSelection,
    /// A local manifest selected one mutable task identity more than once.
    DuplicateTaskId(String),
}

impl std::fmt::Display for RegistryError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidLocalManifest => {
                formatter.write_str("local registry reference requires an id and task selection")
            }
            Self::EmptyLocalManifestId => {
                formatter.write_str("local registry reference requires an id")
            }
            Self::EmptyTaskSelection => {
                formatter.write_str("local registry reference requires a task selection")
            }
            Self::DuplicateTaskId(task_id) => {
                write!(
                    formatter,
                    "local registry reference selects task {task_id:?} more than once"
                )
            }
        }
    }
}

impl std::error::Error for RegistryError {}
