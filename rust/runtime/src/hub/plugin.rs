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
