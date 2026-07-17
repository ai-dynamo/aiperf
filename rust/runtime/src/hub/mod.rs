// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The per-experiment velo hub: one velo instance and one axum service, hosting
//! a set of [`HubPlugin`]s whose behavior is reachable identically over HTTP or a
//! velo message.
//!
//! One [`Hub`] per benchmark run co-binds a single velo messaging instance (built
//! with the shared [`build_velo`](crate::cellular::transport::connect::build_velo))
//! and a single [`axum`] HTTP service. Each registered plugin contributes an HTTP
//! router (mounted under its prefix) and velo handlers (installed on the shared
//! instance); the two surfaces are expected to share one handler function so they
//! cannot diverge. See [`specs/velo-hub.md`](../../../../specs/velo-hub.md) for the
//! design record and the deferred controller/artifact-plane fold-in.
//!
//! The velo and HTTP surfaces bind distinct sockets (as the cell↔controller and
//! artifact planes do today); "co-bound" means one [`Hub`]/[`HubServer`] owns and
//! lifecycle-manages both, not that they share a port.

mod discovery;
mod plugin;

use std::collections::BTreeSet;
use std::net::SocketAddr;
use std::sync::Arc;

use tokio::sync::oneshot;
use tokio::task::JoinHandle;
use velo::Velo;

pub use discovery::{
    DiscoveryPlugin, DiscoveryReply, DiscoveryRequest, DiscoveryState, HUB_DISCOVERY,
    handle_discovery,
};
pub use plugin::{HUB_ABI_VERSION, HubAbiRequirement, HubError, HubPlugin};

/// Composes [`HubPlugin`]s over one velo instance and serves both their HTTP and
/// velo surfaces.
///
/// Build with an already-bound velo instance, [`register`](Self::register) each
/// plugin (duplicate prefixes are rejected before the plugin's velo handlers are
/// installed), then [`serve`](Self::serve) to bring up the HTTP surface. The velo
/// surface is already live from the moment its handlers are registered.
pub struct Hub {
    velo: Arc<Velo>,
    plugins: Vec<Box<dyn HubPlugin>>,
    prefixes: BTreeSet<String>,
}

impl Hub {
    /// Create a hub over an already-bound velo instance (built with
    /// [`build_velo`](crate::cellular::transport::connect::build_velo), so the hub
    /// does not duplicate transport construction).
    pub fn new(velo: Arc<Velo>) -> Self {
        Self {
            velo,
            plugins: Vec::new(),
            prefixes: BTreeSet::new(),
        }
    }

    /// The shared velo instance, e.g. to read its [`peer_info`](velo::Velo::peer_info)
    /// when constructing a plugin's discovery state.
    pub fn velo(&self) -> &Arc<Velo> {
        &self.velo
    }

    /// Register `plugin`: negotiate its declared hub ABI, validate its prefix,
    /// reject a duplicate, and only then install its velo handlers on the shared
    /// instance. On success the plugin is retained for HTTP mounting by
    /// [`serve`](Self::serve).
    ///
    /// Registration is transactional in spirit — a plugin that declares an
    /// incompatible ABI, or whose prefix collides, is rejected before any of its
    /// velo handlers are installed and before any hub state (prefixes, retained
    /// plugins) is touched, so the hub is left exactly as it was. The ABI check
    /// runs first because it inspects no hub state and so needs no rollback.
    pub fn register(&mut self, plugin: Box<dyn HubPlugin>) -> Result<(), HubError> {
        let prefix = plugin.prefix().to_owned();
        let required = plugin.required_abi();
        if !required.accepts(HUB_ABI_VERSION) {
            return Err(HubError::IncompatibleAbi {
                prefix,
                required,
                supported: HUB_ABI_VERSION,
            });
        }
        if prefix.is_empty() || !prefix.starts_with('/') {
            return Err(HubError::InvalidPrefix(prefix));
        }
        if !self.prefixes.insert(prefix.clone()) {
            return Err(HubError::DuplicatePrefix(prefix));
        }
        // Install the velo surface only after the prefix is accepted. A failure
        // here rolls the prefix back so the hub is not left with a half-registered
        // plugin.
        if let Err(error) = plugin.register_velo_handlers(&self.velo) {
            self.prefixes.remove(&prefix);
            return Err(error);
        }
        self.plugins.push(plugin);
        Ok(())
    }

    /// The prefixes of every registered plugin, in deterministic order.
    pub fn prefixes(&self) -> impl ExactSizeIterator<Item = &str> {
        self.plugins.iter().map(|plugin| plugin.prefix())
    }

    /// Merge every registered plugin's router, each nested under its prefix, into
    /// one [`axum::Router`].
    pub fn router(&self) -> axum::Router {
        let mut app = axum::Router::new();
        for plugin in &self.plugins {
            app = app.nest(plugin.prefix(), plugin.router());
        }
        app
    }

    /// Bind the HTTP surface on `http_bind` and start serving both surfaces.
    /// `127.0.0.1:0` (in-process / test) or `0.0.0.0:PORT` (k8s). The returned
    /// [`HubServer`] keeps the velo instance and the HTTP task alive.
    pub async fn serve(self, http_bind: SocketAddr) -> Result<HubServer, HubError> {
        let app = self.router();
        let listener = tokio::net::TcpListener::bind(http_bind)
            .await
            .map_err(|error| HubError::Http(format!("binding hub HTTP to {http_bind}: {error}")))?;
        let http_addr = listener
            .local_addr()
            .map_err(|error| HubError::Http(format!("reading hub HTTP address: {error}")))?;
        let (shutdown_tx, shutdown_rx) = oneshot::channel();
        let task = tokio::spawn(async move {
            let _ = axum::serve(listener, app)
                .with_graceful_shutdown(async move {
                    let _ = shutdown_rx.await;
                })
                .await;
        });
        Ok(HubServer {
            _velo: self.velo,
            http_addr,
            shutdown_tx: Some(shutdown_tx),
            task: Some(task),
        })
    }
}

/// A running hub: the live velo instance plus the spawned HTTP server, with
/// graceful shutdown. Dropping it (or [`shutdown`](Self::shutdown)) stops the HTTP
/// server and tears down the velo messaging plane.
pub struct HubServer {
    /// Held so the velo instance (and its registered handlers) outlives the HTTP
    /// server; dropping it tears the messaging plane down.
    _velo: Arc<Velo>,
    http_addr: SocketAddr,
    shutdown_tx: Option<oneshot::Sender<()>>,
    task: Option<JoinHandle<()>>,
}

impl HubServer {
    /// The bound HTTP address (host + OS-assigned port when bound to `:0`).
    pub fn http_addr(&self) -> SocketAddr {
        self.http_addr
    }

    /// Stop the HTTP server and join its task.
    pub async fn shutdown(mut self) {
        if let Some(tx) = self.shutdown_tx.take() {
            let _ = tx.send(());
        }
        if let Some(task) = self.task.take() {
            let _ = task.await;
        }
    }
}

impl Drop for HubServer {
    fn drop(&mut self) {
        if let Some(tx) = self.shutdown_tx.take() {
            let _ = tx.send(());
        }
    }
}
