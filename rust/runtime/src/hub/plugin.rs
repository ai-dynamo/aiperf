// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The [`HubPlugin`] trait and hub error type.
//!
//! A hub plugin contributes two surfaces backed by one shared handler function:
//! an [`axum::Router`] (HTTP) and a set of velo handlers registered on the shared
//! instance. Registration is validated by the [`Hub`](super::Hub) host, which
//! rejects a duplicate prefix before installing the plugin's velo handlers.

use std::error::Error;
use std::fmt::{self, Display};
use std::sync::Arc;

use velo::Velo;

/// The hub's current ABI version.
///
/// This is the contract version of the [`HubPlugin`] surface — the trait shape,
/// the `HubError` vocabulary, and the registration semantics a plugin binds to.
/// A plugin declares the range it was built against via
/// [`HubPlugin::required_abi`]; [`Hub::register`](super::Hub::register) rejects a
/// plugin whose range does not include this value. Bump it when a change to the
/// plugin contract would silently misbehave against an older plugin (not for
/// additive, backward-compatible changes an old plugin tolerates).
pub const HUB_ABI_VERSION: u32 = 1;

/// The hub ABI version range a plugin was built against, declared via
/// [`HubPlugin::required_abi`] and checked at registration.
///
/// The range is inclusive on both ends. A plugin is compatible with the hub when
/// [`HUB_ABI_VERSION`] falls within `[min, max]`. The default —
/// [`HubAbiRequirement::current`] — pins a plugin to exactly the ABI it compiled
/// against, which is the safe default: a plugin that has thought about
/// forward/backward compatibility opts into a wider range explicitly.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HubAbiRequirement {
    /// The lowest hub ABI version the plugin supports (inclusive).
    pub min: u32,
    /// The highest hub ABI version the plugin supports (inclusive).
    pub max: u32,
}

impl HubAbiRequirement {
    /// Require exactly the ABI the plugin compiled against ([`HUB_ABI_VERSION`]).
    pub const fn current() -> Self {
        Self {
            min: HUB_ABI_VERSION,
            max: HUB_ABI_VERSION,
        }
    }

    /// Require exactly `version`.
    pub const fn exact(version: u32) -> Self {
        Self {
            min: version,
            max: version,
        }
    }

    /// Require any hub ABI in the inclusive range `[min, max]`.
    pub const fn range(min: u32, max: u32) -> Self {
        Self { min, max }
    }

    /// Whether `version` satisfies this requirement.
    pub const fn accepts(&self, version: u32) -> bool {
        self.min <= version && version <= self.max
    }
}

impl Default for HubAbiRequirement {
    fn default() -> Self {
        Self::current()
    }
}

impl Display for HubAbiRequirement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.min == self.max {
            write!(f, "{}", self.min)
        } else {
            write!(f, "{}..={}", self.min, self.max)
        }
    }
}

/// A unit of hub behavior contributing both an HTTP router and velo handlers.
///
/// Implementations are expected to route both surfaces into a single shared
/// handler function so the HTTP and velo paths cannot diverge (see
/// [`DiscoveryPlugin`](super::DiscoveryPlugin)).
pub trait HubPlugin: Send + Sync {
    /// The plugin's stable HTTP mount point and diagnostic identity, e.g.
    /// `"/discovery"`. Must be non-empty and begin with `/`; it is unique within
    /// a hub (a duplicate is rejected at registration).
    fn prefix(&self) -> &str;

    /// The hub ABI version range this plugin was built against. The hub rejects
    /// the plugin at registration unless [`HUB_ABI_VERSION`] falls within it.
    ///
    /// The default pins the plugin to exactly the ABI it compiled against
    /// ([`HubAbiRequirement::current`]); a plugin that has verified it tolerates a
    /// wider range overrides this to declare that range explicitly. Existing
    /// plugins compile unchanged and get the safe exact-current requirement.
    fn required_abi(&self) -> HubAbiRequirement {
        HubAbiRequirement::current()
    }

    /// The plugin's HTTP surface, mounted under [`prefix`](Self::prefix) by the
    /// hub. Each route the returned router declares is reached at
    /// `{prefix}{route}`.
    fn router(&self) -> axum::Router;

    /// Install this plugin's velo handlers on the shared `velo` instance. Called
    /// once, at registration, after the hub has accepted the plugin's prefix.
    fn register_velo_handlers(&self, velo: &Arc<Velo>) -> Result<(), HubError>;
}

/// An error constructing, registering into, or serving a [`Hub`](super::Hub).
#[derive(Debug)]
pub enum HubError {
    /// A plugin's prefix was empty or did not begin with `/`.
    InvalidPrefix(String),
    /// A plugin was registered whose prefix collides with an already-registered
    /// plugin (mirrors the duplicate-name rejection of `AIPerfRegistry`).
    DuplicatePrefix(String),
    /// A plugin declared a required hub ABI range that does not include the hub's
    /// [`HUB_ABI_VERSION`]. Carries the plugin prefix, the range it required, and
    /// the version the hub supports so the mismatch is legible.
    IncompatibleAbi {
        /// The registering plugin's prefix.
        prefix: String,
        /// The ABI range the plugin declared via [`HubPlugin::required_abi`].
        required: HubAbiRequirement,
        /// The hub's supported ABI version ([`HUB_ABI_VERSION`]).
        supported: u32,
    },
    /// A plugin's `register_velo_handlers` failed (e.g. a duplicate velo handler
    /// name). Carries the plugin prefix and the underlying message.
    VeloHandler {
        /// The registering plugin's prefix.
        prefix: String,
        /// The underlying velo/handler failure, stringified.
        message: String,
    },
    /// Binding or serving the hub's HTTP surface failed.
    Http(String),
}

impl Display for HubError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidPrefix(prefix) => {
                write!(
                    f,
                    "hub plugin prefix {prefix:?} must be non-empty and begin with '/'"
                )
            }
            Self::DuplicatePrefix(prefix) => {
                write!(f, "duplicate hub plugin prefix {prefix:?}")
            }
            Self::IncompatibleAbi {
                prefix,
                required,
                supported,
            } => {
                write!(
                    f,
                    "hub plugin {prefix:?} requires hub ABI {required} but the hub supports {supported}"
                )
            }
            Self::VeloHandler { prefix, message } => {
                write!(
                    f,
                    "hub plugin {prefix:?} failed to register velo handlers: {message}"
                )
            }
            Self::Http(message) => write!(f, "hub HTTP surface error: {message}"),
        }
    }
}

impl Error for HubError {}
