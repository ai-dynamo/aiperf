// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Static descriptors a plugin hands the host at the entry call.
//!
//! Descriptors are borrowed for `'static` by [`PluginDeclarationV1`], so a
//! plugin builds them once — typically in a `LazyLock` the SDK macro emits —
//! and the host never takes ownership of plugin-allocated descriptor storage.
//! Every field is host-readable and host-comparable: the loader re-checks the
//! descriptor against the manifest after the entry call, so an accidental
//! manifest/library pairing error rejects the whole package.
//!
//! [`PluginDeclarationV1`]: crate::extension::PluginDeclarationV1

use std::fmt;

use crate::{
    error::SourceApiVersionError,
    id::{REGISTRY_ID_NORMALIZATION_VERSION, RegistryId},
};

/// The source-compatibility version of the plugin API a package was built
/// against.
///
/// Parsed from a canonical `major.minor.patch` decimal triple. Compatibility is
/// semver-style: a plugin loads into a host that shares its major version and
/// is at least as new in minor version.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PluginSourceApiVersion {
    major: u16,
    minor: u16,
    patch: u16,
}

impl PluginSourceApiVersion {
    /// The source API version this crate implements.
    pub const CURRENT: Self = Self::new(1, 0, 0);

    /// Build a version from its three components.
    pub const fn new(major: u16, minor: u16, patch: u16) -> Self {
        Self {
            major,
            minor,
            patch,
        }
    }

    /// Parse a canonical `major.minor.patch` decimal triple.
    ///
    /// Non-canonical equivalent spellings — a missing component, a leading `+`
    /// or `-`, or redundant zero padding such as `01.0.0` — are rejected so one
    /// version has exactly one accepted spelling.
    pub fn parse(value: &str) -> Result<Self, SourceApiVersionError> {
        let mut components = value.split('.');
        let mut parsed = [0_u16; 3];
        for (index, slot) in parsed.iter_mut().enumerate() {
            let Some(component) = components.next() else {
                return Err(SourceApiVersionError::ComponentCount {
                    found: index + components.count(),
                });
            };
            *slot = parse_component(component).ok_or(SourceApiVersionError::Component { index })?;
        }
        let extra = components.count();
        if extra > 0 {
            return Err(SourceApiVersionError::ComponentCount { found: 3 + extra });
        }
        Ok(Self::new(parsed[0], parsed[1], parsed[2]))
    }

    /// The major component.
    pub const fn major(&self) -> u16 {
        self.major
    }

    /// The minor component.
    pub const fn minor(&self) -> u16 {
        self.minor
    }

    /// The patch component.
    pub const fn patch(&self) -> u16 {
        self.patch
    }

    /// Whether a package built against `self` may load into a `host` API.
    ///
    /// The major versions must match exactly and the host must not be older in
    /// minor version than the package it loads.
    pub const fn is_compatible_with(&self, host: &Self) -> bool {
        self.major == host.major && self.minor <= host.minor
    }
}

/// Parse one canonical decimal `u16` component: digits only, no sign, and no
/// redundant leading zero.
fn parse_component(component: &str) -> Option<u16> {
    if component.is_empty() || !component.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    if component.len() > 1 && component.starts_with('0') {
        return None;
    }
    component.parse().ok()
}

impl fmt::Display for PluginSourceApiVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

/// Identity a plugin package repeats to the host at the entry call.
///
/// The host compares these fields against the manifest that selected the
/// library. The descriptor is borrowed, never moved across the boundary, so the
/// plugin remains both the allocation and the drop owner of its storage.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PluginPackageDescriptor {
    /// The normalized package identifier.
    pub id: RegistryId,
    /// The package's own release version, as authored in its manifest.
    pub version: &'static str,
    /// One-line human-readable description, used only in diagnostics and
    /// catalog output.
    pub description: &'static str,
}

impl PluginPackageDescriptor {
    /// Build a package descriptor from an already-normalized identifier.
    ///
    /// Not `const`: `RegistryId` owns a `Box<str>`, so no value of this type
    /// can be produced in a const context. Plugins build their descriptor once
    /// in a `LazyLock` instead.
    pub fn new(id: RegistryId, version: &'static str, description: &'static str) -> Self {
        Self {
            id,
            version,
            description,
        }
    }

    /// Normalize `id` under [`REGISTRY_ID_NORMALIZATION_VERSION`] and build the
    /// descriptor.
    pub fn from_authored(
        id: &str,
        version: &'static str,
        description: &'static str,
    ) -> Result<Self, crate::error::RegistryIdError> {
        Ok(Self::new(
            RegistryId::new(id, REGISTRY_ID_NORMALIZATION_VERSION)?,
            version,
            description,
        ))
    }

    /// The normalized package identifier.
    pub fn id(&self) -> &RegistryId {
        &self.id
    }
}

/// Identity of one capability a package registers, bound to the package that
/// owns it.
///
/// The category descriptor exists so a registration observed by the host always
/// carries its origin: a plugin cannot claim a package identity other than the
/// one the manifest bound to its library. Both fields are private and the
/// constructor is `pub(crate)`, so the only way to obtain one is
/// [`PluginRegistrar::describe`](crate::PluginRegistrar::describe), which reads
/// the origin from the manifest-bound registrar.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PluginCategoryDescriptor {
    /// The normalized capability identifier within its category.
    id: RegistryId,
    /// The package that registered this capability.
    package: &'static PluginPackageDescriptor,
}

impl PluginCategoryDescriptor {
    /// Bind a normalized capability identifier to its owning package.
    ///
    /// `pub(crate)` on purpose: naming an origin is a host act, so the only
    /// plugin-reachable path to a descriptor is through a manifest-bound
    /// registrar.
    ///
    /// Not `const` for the same reason as
    /// [`PluginPackageDescriptor::new`]: the identifier is heap-backed.
    pub(crate) fn new(id: RegistryId, package: &'static PluginPackageDescriptor) -> Self {
        Self { id, package }
    }

    /// The normalized capability identifier.
    pub fn id(&self) -> &RegistryId {
        &self.id
    }

    /// The package that registered this capability.
    pub fn package(&self) -> &'static PluginPackageDescriptor {
        self.package
    }
}
