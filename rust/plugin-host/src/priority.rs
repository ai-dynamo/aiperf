// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Priority rules for plugin discovery sources (Task 13).
//!
//! Each `DiscoverySource` variant is assigned a stable kind ordinal that
//! determines its priority tier during catalog resolution.  Higher ordinals
//! win over lower ordinals; within the same ordinal, authored priority from
//! the manifest package entry breaks ties.

use crate::discovery::DiscoverySource;

/// Return the stable kind ordinal for a discovery source variant.
///
/// The ordinal encodes priority tier: a higher value beats a lower value
/// regardless of authored `priority` field in the manifest.
///
/// | Ordinal | Source kind                     |
/// |--------:|---------------------------------|
/// |       0 | Distribution (lowest)           |
/// |       1 | PlatformSystem                  |
/// |       2 | PlatformUser                    |
/// |       3 | Environment                     |
/// |       4 | ExplicitDirectory               |
/// |       5 | ExplicitManifest                |
/// |       6 | HermeticBundle (highest)        |
pub fn source_kind_ordinal(source: &DiscoverySource) -> u8 {
    match source {
        DiscoverySource::Distribution => 0,
        DiscoverySource::PlatformSystem => 1,
        DiscoverySource::PlatformUser => 2,
        DiscoverySource::Environment(_) => 3,
        DiscoverySource::ExplicitDirectory(_) => 4,
        DiscoverySource::ExplicitManifest(_) => 5,
        DiscoverySource::HermeticBundle(_) => 6,
    }
}

/// Compute the effective catalog priority for a discovered package.
///
/// `source_ordinal` is the kind ordinal from `source_kind_ordinal`.
/// `authored_priority` is the `priority` field from the manifest package entry.
/// The effective priority is `(source_ordinal as i32) * 1000 + authored_priority`.
/// This ensures source tier dominates over authored priority.
pub fn effective_priority(source_ordinal: u8, authored_priority: i32) -> i32 {
    (source_ordinal as i32) * 1000 + authored_priority
}
