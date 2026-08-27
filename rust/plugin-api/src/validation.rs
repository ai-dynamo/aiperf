// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Category selection, digests, host resources, and validation failures.
//!
//! A factory validates authored configuration and hands back an opaque prepared
//! value plus a receipt. The receipt is only meaningful against a fixed
//! vocabulary: which category was selected, which digests bind the plan, and
//! which host capabilities the plugin asked for. That vocabulary is here, and it
//! is host-owned and sealed — a plugin selects from it and cannot extend it.
//!
//! Digests are carried as raw 32-byte values rather than hex strings. The host
//! computes them; this crate only transports and compares them, so it needs no
//! hashing dependency and no allocation to hold one.

use core::fmt::{self, Display, Formatter};

use crate::descriptor::PluginCategoryDescriptor;

/// A 32-byte content digest computed by the host.
///
/// Comparison is byte-wise and total, so a sorted collection of digests has a
/// deterministic order independent of how they were rendered.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ContentDigest([u8; 32]);

impl ContentDigest {
    /// Wrap 32 digest bytes.
    pub const fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    /// Borrow the digest bytes.
    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    /// Render the digest as 64 lowercase hex characters, without allocating.
    pub fn to_hex(self) -> [u8; 64] {
        const HEX: &[u8; 16] = b"0123456789abcdef";
        let mut rendered = [0_u8; 64];
        for (index, byte) in self.0.iter().enumerate() {
            rendered[index * 2] = HEX[usize::from(byte >> 4)];
            rendered[index * 2 + 1] = HEX[usize::from(byte & 0x0f)];
        }
        rendered
    }
}

impl Display for ContentDigest {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        for byte in self.0 {
            write!(formatter, "{byte:02x}")?;
        }
        Ok(())
    }
}

/// The plugin categories generation one defines.
///
/// Sealed on purpose: a category is a host execution position, not a plugin
/// name. Adding one changes what the host will call and when, so it is a host
/// change rather than a plugin declaration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PluginCategory {
    /// Composes request bodies and reads responses for one endpoint dialect.
    Endpoint,
    /// Issues requests, either through the shared worker kernel or directly.
    Transport,
    /// Writes run output after the report is finalized.
    Exporter,
}

/// Every category in canonical order.
pub const PLUGIN_CATEGORIES: &[PluginCategory] = &[
    PluginCategory::Endpoint,
    PluginCategory::Transport,
    PluginCategory::Exporter,
];

impl PluginCategory {
    /// The lowercase label used in receipts and diagnostics.
    pub const fn label(self) -> &'static str {
        match self {
            Self::Endpoint => "endpoint",
            Self::Transport => "transport",
            Self::Exporter => "exporter",
        }
    }

    /// Parse a category label, refusing anything outside the sealed set.
    pub fn parse(label: &str) -> Result<Self, ValidationError> {
        PLUGIN_CATEGORIES
            .iter()
            .copied()
            .find(|category| category.label() == label)
            .ok_or(ValidationError::UnknownCategory)
    }
}

impl Display for PluginCategory {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.label())
    }
}

/// One host capability a factory asked to be given.
///
/// These name the narrow `aiperf_core::services` traits, not runtime values: a
/// plugin requests `Graph` and receives a
/// `aiperf_core::services::GraphService`, never a runtime graph program.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum HostResourceV1 {
    /// `aiperf_core::services::ClockService`.
    Clock,
    /// `aiperf_core::services::GraphService`.
    Graph,
    /// `aiperf_core::services::MetricsService`.
    Metrics,
    /// `aiperf_core::services::ArtifactService`.
    Artifacts,
    /// `aiperf_core::services::CancellationService`.
    Cancellation,
}

/// Every host resource in canonical (sorted) order.
pub const HOST_RESOURCES_V1: &[HostResourceV1] = &[
    HostResourceV1::Clock,
    HostResourceV1::Graph,
    HostResourceV1::Metrics,
    HostResourceV1::Artifacts,
    HostResourceV1::Cancellation,
];

impl HostResourceV1 {
    /// The lowercase label used in receipts and diagnostics.
    pub const fn label(self) -> &'static str {
        match self {
            Self::Clock => "clock",
            Self::Graph => "graph",
            Self::Metrics => "metrics",
            Self::Artifacts => "artifacts",
            Self::Cancellation => "cancellation",
        }
    }
}

impl Display for HostResourceV1 {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.label())
    }
}

/// A sorted, deduplicated host-resource request.
///
/// Sorting is what makes a receipt comparable: two factories that ask for the
/// same capabilities in different declaration orders must produce the same
/// receipt bytes, or the receipt cannot be used to detect drift.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct HostResourceSetV1 {
    resources: Vec<HostResourceV1>,
}

impl HostResourceSetV1 {
    /// Build a sorted, deduplicated set from any declaration order.
    pub fn new(resources: impl IntoIterator<Item = HostResourceV1>) -> Self {
        let mut resources: Vec<HostResourceV1> = resources.into_iter().collect();
        resources.sort_unstable();
        resources.dedup();
        Self { resources }
    }

    /// The requested resources in canonical order.
    pub fn as_slice(&self) -> &[HostResourceV1] {
        &self.resources
    }

    /// Whether a resource was requested.
    pub fn contains(&self, resource: HostResourceV1) -> bool {
        self.resources.contains(&resource)
    }

    /// Whether nothing was requested.
    pub fn is_empty(&self) -> bool {
        self.resources.is_empty()
    }
}

/// One category descriptor bound to its selected category and digest.
///
/// [`PluginCategoryDescriptor`] alone says which registry identifier a package
/// registered. This adds the two facts validation needs: which category
/// position it was registered into, and the digest of the descriptor bytes the
/// host validated.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidatedCategoryDescriptorV1 {
    category: PluginCategory,
    descriptor: PluginCategoryDescriptor,
    descriptor_digest: ContentDigest,
}

impl ValidatedCategoryDescriptorV1 {
    /// Bind one descriptor to its category and digest.
    pub const fn new(
        category: PluginCategory,
        descriptor: PluginCategoryDescriptor,
        descriptor_digest: ContentDigest,
    ) -> Self {
        Self {
            category,
            descriptor,
            descriptor_digest,
        }
    }

    /// The category position the descriptor was registered into.
    pub const fn category(&self) -> PluginCategory {
        self.category
    }

    /// The registered descriptor.
    pub const fn descriptor(&self) -> &PluginCategoryDescriptor {
        &self.descriptor
    }

    /// Digest of the descriptor bytes the host validated.
    pub const fn descriptor_digest(&self) -> ContentDigest {
        self.descriptor_digest
    }
}

/// Why a boundary value was refused before it could be used.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ValidationError {
    /// A category label outside the sealed [`PLUGIN_CATEGORIES`] set.
    UnknownCategory,
    /// A capture projection identifier the host does not define.
    UnknownCaptureProjection(String),
    /// A transport declared more or fewer than one execution shape.
    AmbiguousExecutionShape,
    /// A receipt's category does not match the factory that produced it.
    CategoryMismatch {
        /// The category the factory occupies.
        expected: PluginCategory,
        /// The category the receipt claims.
        found: PluginCategory,
    },
    /// The value was rejected for a category-specific reason.
    Rejected(String),
}

impl Display for ValidationError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownCategory => formatter.write_str("unknown plugin category"),
            Self::UnknownCaptureProjection(id) => {
                write!(formatter, "unknown capture projection {id:?}")
            }
            Self::AmbiguousExecutionShape => {
                formatter.write_str("a transport must declare exactly one execution shape")
            }
            Self::CategoryMismatch { expected, found } => write!(
                formatter,
                "receipt category {found} does not match factory category {expected}"
            ),
            Self::Rejected(reason) => write!(formatter, "validation rejected: {reason}"),
        }
    }
}

impl core::error::Error for ValidationError {}
