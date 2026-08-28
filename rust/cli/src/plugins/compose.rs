// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Plugin universe composition before process effects.
//!
//! [`compose_plugin_universe`] is called once, early in the bootstrap sequence,
//! before `Application` construction and before any sockets, file descriptors,
//! or platform resources are opened.  It resolves the plugin lock, loads each
//! winning package's shared library, calls the plugin entry point, and returns
//! a [`FrozenPluginUniverse`] that captures the committed registry state.
//!
//! An absent lock file is a valid zero-plugin configuration: the function
//! returns an empty universe without error so that non-plugin deployments are
//! completely unaffected by the machinery.
//!
//! # Relation to Tasks 15 and 16
//!
//! `FrozenPluginUniverse` and `LockedCatalogBundle` are defined here as stubs.
//! They will be replaced with the canonical types from `aiperf-plugin-api` and
//! `aiperf-plugin-host` once Tasks 15 and 16 land and are integrated.  The
//! function signature and the zero-plugin behaviour will remain stable.

use std::path::Path;

/// A committed, immutable snapshot of the registered plugin universe.
///
/// **Stub** — replaced by the `aiperf-plugin-api` type when Task 15 lands.
/// The stable contract is `is_empty()` and the guarantee that construction
/// consumes and commits the registry so no further registration is possible.
#[derive(Debug)]
pub struct FrozenPluginUniverse {
    /// Number of winner packages that were activated.
    package_count: usize,
}

impl FrozenPluginUniverse {
    /// Construct an empty (zero-plugin) frozen universe.
    pub fn empty() -> Self {
        Self { package_count: 0 }
    }

    /// Return `true` when no plugins were activated.
    pub fn is_empty(&self) -> bool {
        self.package_count == 0
    }

    /// Number of winner packages that were activated.
    pub fn package_count(&self) -> usize {
        self.package_count
    }
}

/// Error returned when plugin composition fails.
#[derive(Debug, thiserror::Error)]
pub enum ComposeError {
    /// The lock file exists but failed verification (digest mismatch, truncated,
    /// or corrupt JSON).
    #[error("plugin lock verification failed: {0}")]
    LockVerification(String),

    /// A winner package's shared library could not be loaded.
    #[error("plugin '{package_id}' load failed: {reason}")]
    PluginLoad {
        /// The normalised package identifier from the lock.
        package_id: String,
        /// Human-readable load failure detail (e.g. `dlerror(3)` text).
        reason: String,
    },

    /// The plugin entry point ran but rejected the host ABI or returned an
    /// error during registration.
    #[error("plugin '{package_id}' registration failed: {reason}")]
    PluginRegistration {
        /// The normalised package identifier from the lock.
        package_id: String,
        /// Human-readable registration failure detail.
        reason: String,
    },
}

/// Resolve the plugin universe from `lock_path` and freeze the registry.
///
/// This function must be called before any process effect (socket, file
/// descriptor, platform resource) is opened.
///
/// # Behaviour
///
/// - If `lock_path` does not exist, returns [`FrozenPluginUniverse::empty`]
///   without error.  This is the common case for non-plugin deployments.
/// - If `lock_path` exists, loads and verifies the lock (digest check), then
///   for each `PackageStatus::Winner` entry: opens the shared library, calls
///   the plugin entry point, and registers the plugin.  Finally, freezes the
///   registry and returns the committed universe.
/// - Any failure after the lock file is found returns a typed [`ComposeError`].
///
/// # Stub note
///
/// The load / registration steps are stubs pending Task 14 (`dlopen` loader)
/// and Task 15 (transactional registry + freeze).  The file-existence check and
/// the empty-universe fast path are fully implemented and tested.
pub fn compose_plugin_universe(lock_path: &Path) -> Result<FrozenPluginUniverse, ComposeError> {
    if !lock_path.exists() {
        tracing::debug!(
            path = %lock_path.display(),
            "no plugin lock file found; starting with empty plugin universe"
        );
        return Ok(FrozenPluginUniverse::empty());
    }

    // --- Stub: real implementation pending Task 15/16 ---
    //
    // When Tasks 15 and 16 land, this block will:
    //   1. Call `LockedCatalogBundle::load_and_verify(lock_path)` → map to
    //      `ComposeError::LockVerification`.
    //   2. For each `LockedPackageV1` with `status == PackageStatus::Winner`:
    //      - Call the Task 14 loader to `dlopen` the artifact.
    //      - Call the plugin entry point via the Task 9 declaration ABI.
    //      - Register with the Task 15 transactional registry builder.
    //   3. Call `freeze_universe(registry_builder)` → `FrozenPluginUniverse`.
    //
    // For now, reading an existing lock file logs a warning and returns an
    // empty universe so nothing breaks until the integration lands.
    tracing::warn!(
        lock_path = %lock_path.display(),
        "plugin lock file found but plugin loading is not yet implemented (Tasks 14-16 pending integration)"
    );

    Ok(FrozenPluginUniverse::empty())
}
