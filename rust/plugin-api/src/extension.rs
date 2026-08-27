// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The plugin entry shape and the registration seam it drives.
//!
//! A plugin is a Rust `cdylib` exporting exactly one symbol,
//! [`PLUGIN_ENTRY_SYMBOL_V1`], typed as [`PluginEntryV1`]. It is a Rust-ABI
//! function, not `extern "C"`: source-level ABI identity is established by the
//! host ABI universe record, not by a C calling convention. Calling the symbol
//! yields a [`PluginDeclarationV1`] holding two `'static` borrows, so the entry
//! call transfers no ownership. The host then drives
//! [`AIPerfExtension::register`] exactly once with a manifest-bound
//! [`PluginRegistrar`].
//!
//! The generation-1 registrar records observed registrations and rejects
//! duplicates. The endpoint, transport, and exporter category methods land on
//! top of this seam without changing the entry shape.

use std::{collections::HashSet, fmt, marker::PhantomData};

use crate::{
    descriptor::{PluginCategoryDescriptor, PluginPackageDescriptor},
    error::ExtensionError,
    id::RegistryId,
};

/// The exported symbol name every version-1 plugin library must provide.
///
/// Lookup is always scoped to the exact library handle the manifest selected,
/// so an unrelated library exporting the same name is never a second entry
/// point into a loaded package.
pub const PLUGIN_ENTRY_SYMBOL_V1: &str = "aiperf_plugin_entry_v1";

/// The type of the version-1 plugin entry symbol.
///
/// # Safety
///
/// The host may call this only on a function resolved from a library handle
/// that passed manifest, embedded-record, and build-identity validation. The
/// call is `unsafe` because the resolved address is trusted to have this exact
/// Rust-ABI signature; nothing in the type system proves that of a symbol read
/// out of a dynamic library.
pub type PluginEntryV1 = unsafe fn() -> PluginDeclarationV1;

/// What the version-1 entry symbol returns.
///
/// Both fields are `'static` borrows into plugin-owned storage. The plugin
/// allocates them and the plugin's library — which stays resident for the whole
/// process lifetime — owns dropping them, so no plugin-allocated storage is
/// ever freed by the host allocator.
pub struct PluginDeclarationV1 {
    /// Package identity, re-checked by the host against the manifest.
    pub package: &'static PluginPackageDescriptor,
    /// The registration entry point the host drives once.
    pub extension: &'static dyn AIPerfExtension,
}

impl fmt::Debug for PluginDeclarationV1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // `dyn AIPerfExtension` is deliberately not `Debug`: plugin-authored
        // formatting must not run inside host diagnostics.
        f.debug_struct("PluginDeclarationV1")
            .field("package", self.package)
            .finish_non_exhaustive()
    }
}

/// The one plugin-authored call the host makes during the entry transaction.
///
/// Implementations register ordinary factory trait objects through the
/// registrar. `register` returns a typed error rather than panicking: every
/// boundary artifact is built `panic = abort`, so an unwind out of this call
/// would terminate the process instead of failing the package.
pub trait AIPerfExtension: Send + Sync {
    /// Register every capability this package provides.
    ///
    /// The host calls this exactly once per loaded package. Returning an error
    /// rejects the whole package; partially staged registrations are discarded
    /// rather than committed.
    fn register(&self, registrar: &mut PluginRegistrar<'_>) -> Result<(), ExtensionError>;
}

/// A manifest-bound facade over the host's private staged registry.
///
/// The registrar supplies package identity from the manifest rather than from
/// the plugin, observes every actual registration, and never exposes the
/// aggregate registry. A plugin therefore cannot claim an origin other than the
/// package whose manifest selected its library: every descriptor the registrar
/// mints reads the package bound at [`PluginRegistrar::new`], and no method
/// accepts a caller-supplied origin.
///
/// The `'a` parameter reserves the borrow of host-private staged registry state
/// the generation-2 category methods take. It is not the package lifetime: the
/// package is `&'static` so a minted [`PluginCategoryDescriptor`] can outlive
/// the registrar without carrying a plugin-chosen origin.
pub struct PluginRegistrar<'a> {
    package: &'static PluginPackageDescriptor,
    observed: Vec<RegistryId>,
    // Registration order is the reportable fact, but duplicate detection must
    // not be quadratic once generation-2 category methods multiply the call
    // count, so the identifiers are also held in a set.
    seen: HashSet<RegistryId>,
    host_state: PhantomData<&'a ()>,
}

impl PluginRegistrar<'_> {
    /// Bind a registrar to the package identity the manifest resolved.
    pub fn new(package: &'static PluginPackageDescriptor) -> Self {
        Self {
            package,
            observed: Vec::new(),
            seen: HashSet::new(),
            host_state: PhantomData,
        }
    }

    /// The manifest-bound package identity.
    pub fn package(&self) -> &'static PluginPackageDescriptor {
        self.package
    }

    /// Every identifier registered so far, in registration order.
    pub fn observed(&self) -> &[RegistryId] {
        &self.observed
    }

    /// Record one registration against the bound package.
    ///
    /// This is the single staging seam the generation-2 endpoint, transport,
    /// and exporter methods call. Registering the same identifier twice from
    /// one package is a plugin authoring bug, so it is rejected here rather
    /// than silently resolved by priority later.
    pub fn record_registration(&mut self, id: RegistryId) -> Result<(), ExtensionError> {
        if !self.seen.insert(id.clone()) {
            return Err(ExtensionError::for_id(
                id,
                "identifier already registered by this package",
            ));
        }
        self.observed.push(id);
        Ok(())
    }

    /// Bind an observed identifier to the registrar's own package descriptor.
    ///
    /// The host uses this when it promotes staged registrations into the
    /// aggregate registry, where every entry must carry its origin. The origin
    /// is read from the manifest-bound package, never from the caller, so a
    /// plugin holding a registrar cannot mint a descriptor asserting another
    /// package's identity.
    pub fn describe(&self, id: RegistryId) -> PluginCategoryDescriptor {
        PluginCategoryDescriptor::new(id, self.package)
    }
}

impl fmt::Debug for PluginRegistrar<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PluginRegistrar")
            .field("package", self.package)
            .field("observed", &self.observed)
            .finish()
    }
}
