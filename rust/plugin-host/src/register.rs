// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Plugin registration context and the single registration entry point.
//!
//! The host uses [`PluginRegistrationContext`] to bind a manifest-resolved
//! package identity before calling the plugin entry symbol, then calls
//! [`register_plugin`] to drive `AIPerfExtension::register` and convert the
//! result to an immutable [`FrozenPluginUniverse`].

use std::fmt;

use aiperf_plugin_api::{
    AIPerfExtension, ExtensionError, FrozenPluginUniverse, PluginPackageDescriptor,
    PluginRegistrar, RegistryId,
};

/// Error returned when a plugin's registration attempt fails.
#[derive(Debug)]
pub enum RegistrationError {
    /// A capability identifier was registered more than once by the same package.
    DuplicateCapability(RegistryId),
    /// The plugin's `register` call returned a typed extension error.
    Extension(ExtensionError),
}

impl fmt::Display for RegistrationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DuplicateCapability(id) => {
                write!(f, "capability {id:?} registered more than once by the same package")
            }
            Self::Extension(e) => write!(f, "plugin registration failed: {e}"),
        }
    }
}

impl std::error::Error for RegistrationError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Extension(e) => Some(e),
            Self::DuplicateCapability(_) => None,
        }
    }
}

impl From<ExtensionError> for RegistrationError {
    fn from(e: ExtensionError) -> Self {
        // Preserve the duplicate-capability detail when it is a re-registration.
        if let Some(id) = e.registry_id().cloned()
            && e.reason().contains("already registered")
        {
            return Self::DuplicateCapability(id);
        }
        Self::Extension(e)
    }
}

/// Host-side context that binds a manifest-resolved package identity to a
/// registration session.
///
/// The package pointer must be `'static` because [`PluginRegistrar`] borrows
/// it for `'static`; it typically points into a `LazyLock` or to plugin-owned
/// storage that the library handle keeps alive for the whole process lifetime.
#[derive(Debug)]
pub struct PluginRegistrationContext {
    package: &'static PluginPackageDescriptor,
}

impl PluginRegistrationContext {
    /// Bind a manifest-resolved package to a registration session.
    pub fn new(package: &'static PluginPackageDescriptor) -> Self {
        Self { package }
    }

    /// The manifest-bound package identity.
    pub fn package(&self) -> &'static PluginPackageDescriptor {
        self.package
    }
}

/// Drive one plugin's [`AIPerfExtension::register`] call and freeze the result.
///
/// A new [`PluginRegistrar`] is bound to `ctx.package()` and passed to the
/// extension.  On success the registrar is consumed by `freeze()` and the
/// immutable snapshot is returned.  On failure the partially staged state is
/// discarded and the error is returned.
pub fn register_plugin(
    ctx: &PluginRegistrationContext,
    extension: &dyn AIPerfExtension,
) -> Result<FrozenPluginUniverse, RegistrationError> {
    let mut registrar = PluginRegistrar::new(ctx.package);
    extension
        .register(&mut registrar)
        .map_err(RegistrationError::from)?;
    Ok(registrar.freeze())
}
