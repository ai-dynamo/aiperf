// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Frozen plugin universe — immutable snapshot after all plugins register.
//!
//! Calling [`PluginRegistrar::freeze`] consumes a registrar after
//! [`AIPerfExtension::register`] returns and produces a
//! [`FrozenPluginUniverse`] recording what that one package contributed.  The
//! host then merges multiple per-package snapshots into a
//! [`FrozenAIPerfRegistry`] through `freeze_universe` in `aiperf-plugin-host`.

use std::collections::HashMap;

use crate::descriptor::{PluginCategoryDescriptor, PluginPackageDescriptor};

/// Immutable snapshot of all registrations produced by one plugin package.
///
/// Created by [`PluginRegistrar::freeze`](crate::PluginRegistrar::freeze) after
/// `AIPerfExtension::register` completes without error.  The snapshot is
/// process-lifetime: the `'static` borrow to the package descriptor binds
/// the host-owned storage that the library handle keeps alive.
#[derive(Debug)]
pub struct FrozenPluginUniverse {
    pub(crate) inner: FrozenInner,
}

/// The data fields behind [`FrozenPluginUniverse`].
///
/// Exposed as a named struct so the host can destructure it in diagnostics.
/// Construction is `pub(crate)` — only [`PluginRegistrar::freeze`] mints one.
#[derive(Debug)]
pub struct FrozenInner {
    /// The manifest-bound package identity.
    pub package: &'static PluginPackageDescriptor,
    /// Every category descriptor registered, in registration order.
    pub registrations: Vec<PluginCategoryDescriptor>,
}

impl FrozenPluginUniverse {
    /// Construct from a registrar's final state.  Called only by
    /// [`PluginRegistrar::freeze`](crate::PluginRegistrar::freeze).
    pub(crate) fn from_parts(
        package: &'static PluginPackageDescriptor,
        registrations: Vec<PluginCategoryDescriptor>,
    ) -> Self {
        Self {
            inner: FrozenInner {
                package,
                registrations,
            },
        }
    }

    /// The manifest-bound package identity.
    pub fn package(&self) -> &'static PluginPackageDescriptor {
        self.inner.package
    }

    /// All registrations this package contributed, in registration order.
    pub fn registrations(&self) -> &[PluginCategoryDescriptor] {
        &self.inner.registrations
    }

    /// Number of registrations this package contributed.
    pub fn len(&self) -> usize {
        self.inner.registrations.len()
    }

    /// True when the package registered nothing.
    pub fn is_empty(&self) -> bool {
        self.inner.registrations.is_empty()
    }
}

/// Aggregate frozen state after every plugin in a universe has registered.
///
/// Built by `freeze_universe` in `aiperf-plugin-host` from the ordered
/// sequence of per-package [`FrozenPluginUniverse`] values.  After
/// construction the set of entries is fixed; no further mutation is possible.
#[derive(Debug)]
pub struct FrozenAIPerfRegistry {
    universes: Vec<FrozenPluginUniverse>,
    // Capability identifier to (universe index, registration index). Built once
    // at construction because the entry set is fixed afterwards, so resolving a
    // capability never walks the flattened registration list.
    index: HashMap<String, (usize, usize)>,
}

impl FrozenAIPerfRegistry {
    /// Merge an ordered set of per-package frozen universes into one
    /// immutable registry snapshot.
    ///
    /// The lookup index is built here.  When two packages register the same
    /// capability identifier the first in load order is indexed; deciding which
    /// of the two wins is the host's priority resolution, not this snapshot's.
    pub fn new(universes: Vec<FrozenPluginUniverse>) -> Self {
        let mut index =
            HashMap::with_capacity(universes.iter().map(FrozenPluginUniverse::len).sum());
        for (universe_index, universe) in universes.iter().enumerate() {
            for (registration_index, descriptor) in universe.registrations().iter().enumerate() {
                index
                    .entry(descriptor.id().as_str().to_string())
                    .or_insert((universe_index, registration_index));
            }
        }
        Self { universes, index }
    }

    /// Resolve a capability descriptor by its normalized identifier.
    ///
    /// Returns the load-order-first registration of `id`, or `None` when no
    /// package in the universe registered it.
    pub fn lookup_by_id(&self, id: &str) -> Option<&PluginCategoryDescriptor> {
        let (universe_index, registration_index) = *self.index.get(id)?;
        self.universes
            .get(universe_index)?
            .registrations()
            .get(registration_index)
    }

    /// All per-package frozen universes in load order.
    pub fn universes(&self) -> &[FrozenPluginUniverse] {
        &self.universes
    }

    /// Flat iterator over every registered category descriptor, in load order.
    pub fn all_registrations(&self) -> impl Iterator<Item = &PluginCategoryDescriptor> {
        self.universes.iter().flat_map(|u| u.registrations().iter())
    }

    /// Total number of registrations across all packages.
    pub fn registration_count(&self) -> usize {
        self.universes.iter().map(|u| u.len()).sum()
    }
}
