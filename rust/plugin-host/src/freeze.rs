// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Universe-level freeze: merge per-package frozen snapshots into a single
//! immutable [`FrozenAIPerfRegistry`].

use aiperf_plugin_api::{FrozenAIPerfRegistry, FrozenPluginUniverse};

/// Merge an ordered sequence of per-package frozen universes into one
/// immutable registry snapshot.
///
/// The caller supplies the universes in plugin load order; the registry
/// preserves that order in [`FrozenAIPerfRegistry::universes`] and in the flat
/// `all_registrations` iterator.  After this call no further plugin
/// registrations are accepted.
pub fn freeze_universe(universes: Vec<FrozenPluginUniverse>) -> FrozenAIPerfRegistry {
    FrozenAIPerfRegistry::new(universes)
}
