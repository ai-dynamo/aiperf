// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The `/artifact` hub plugin: folds the velo artifact-streaming plane
//! ([`ArtifactVeloReceiver`]) onto the shared hub velo instance, so per-record
//! artifact bytes ride the one hub anchor instead of a separate `:9600` server.
//!
//! The plugin reuses the streaming-zstd bounded-memory machinery
//! ([`crate::engine::artifact_stream_velo`]) verbatim — the OPEN/CLOSE/DONE velo
//! handlers and their per-file [`velo::StreamAnchor`] consumers are unchanged; only
//! the mount point moves onto the hub. Its [`register_velo_handlers`](HubPlugin::register_velo_handlers)
//! installs exactly those handlers via [`ArtifactVeloReceiver::register`], and the
//! bound receiver (which exposes the cell-completion barrier) is captured into a
//! take-once slot the bootstrap owns after `Hub::register` returns.
//!
//! The HTTP surface is a **dual-surface diagnostic**: `GET {prefix}/allowed` returns
//! the exact fail-closed allowlist the velo OPEN handler enforces, so the set that
//! gates the velo path is observable over HTTP from the same plugin state. The bulk
//! byte movement stays on the velo stream primitive (ordered + backpressured); the
//! streaming OPEN/CLOSE/DONE protocol has no faithful plain-axum mirror and is not
//! duplicated.

use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use axum::Json;
use axum::extract::State;
use axum::routing::get;
use serde_json::json;
use velo::Velo;

use super::plugin::{HubAbiRequirement, HubError, HubPlugin};
use crate::engine::artifact_stream_velo::ArtifactVeloReceiver;
use crate::engine::cellular_registration::CellRegistrationAuthority;

/// The default HTTP mount point / diagnostic identity for the artifact plugin.
pub const ARTIFACT_PREFIX: &str = "/artifact";

/// A shared, take-once slot for the [`ArtifactVeloReceiver`] the plugin binds at
/// registration. The bootstrap holds a clone and takes the receiver out after the hub
/// has installed the plugin's velo handlers, then awaits its cell-completion barrier.
pub type ReceiverSlot = Arc<Mutex<Option<ArtifactVeloReceiver>>>;

/// The `/artifact` hub plugin. Registers the velo artifact OPEN/CLOSE/DONE handlers on
/// the shared hub velo instance and captures the resulting receiver for the bootstrap.
pub struct ArtifactHubPlugin {
    prefix: String,
    temp_root: PathBuf,
    allowed: HashSet<String>,
    registration_authority: Arc<CellRegistrationAuthority>,
    receiver: ReceiverSlot,
}

impl ArtifactHubPlugin {
    /// Build the plugin over the controller's cellular scratch root (files land at
    /// `temp_root/cell-{id}/{rel}`) and the exact set of relative artifact paths the
    /// run may ship (fail-closed). Uses the default [`ARTIFACT_PREFIX`].
    pub(crate) fn new(
        temp_root: PathBuf,
        allowed: HashSet<String>,
        registration_authority: Arc<CellRegistrationAuthority>,
    ) -> Self {
        Self {
            prefix: ARTIFACT_PREFIX.to_owned(),
            temp_root,
            allowed,
            registration_authority,
            receiver: Arc::new(Mutex::new(None)),
        }
    }

    /// A clone of the take-once slot the bound [`ArtifactVeloReceiver`] lands in. The
    /// bootstrap clones this BEFORE boxing the plugin into `Hub::register`, then
    /// [`Option::take`]s the receiver out once registration has installed the velo
    /// handlers so it can await the cell-completion barrier.
    pub fn receiver_slot(&self) -> ReceiverSlot {
        self.receiver.clone()
    }
}

/// `GET {prefix}/allowed` — the dual-surface diagnostic: the fail-closed allowlist the
/// velo OPEN handler enforces, sorted for a deterministic response.
async fn http_allowed(State(allowed): State<Arc<Vec<String>>>) -> Json<serde_json::Value> {
    Json(json!({ "allowed": &*allowed }))
}

impl HubPlugin for ArtifactHubPlugin {
    fn prefix(&self) -> &str {
        &self.prefix
    }

    fn required_abi(&self) -> HubAbiRequirement {
        HubAbiRequirement::current()
    }

    fn router(&self) -> axum::Router {
        let mut allowed: Vec<String> = self.allowed.iter().cloned().collect();
        allowed.sort();
        axum::Router::new()
            .route("/allowed", get(http_allowed))
            .with_state(Arc::new(allowed))
    }

    fn register_velo_handlers(&self, velo: &Arc<Velo>) -> Result<(), HubError> {
        let receiver = ArtifactVeloReceiver::register(
            velo.clone(),
            self.temp_root.clone(),
            self.allowed.clone(),
            self.registration_authority.clone(),
        )
        .map_err(|error| HubError::VeloHandler {
            prefix: self.prefix.clone(),
            message: error.to_string(),
        })?;
        *self
            .receiver
            .lock()
            .expect("artifact receiver slot poisoned") = Some(receiver);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cellular::transport::connect::{BindSpec, build_velo};
    use crate::engine::artifact_stream_velo::ship_cell_artifacts_velo;
    use crate::hub::Hub;
    use std::time::Duration;

    /// The plugin mounts on a hub, its velo OPEN/CLOSE/DONE handlers stream a real
    /// file end-to-end over the hub anchor, and the captured receiver's barrier
    /// releases — proving the artifact plane rides the one hub instance.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn plugin_streams_an_artifact_over_the_hub_anchor() {
        let dir = tempfile::tempdir().unwrap();
        let src_dir = dir.path().join("cell-src");
        std::fs::create_dir_all(&src_dir).unwrap();
        let rel = "inputs.json";
        std::fs::write(src_dir.join(rel), vec![7u8; 200_000]).unwrap();
        let source_bytes = std::fs::read(src_dir.join(rel)).unwrap();

        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let velo = build_velo(BindSpec::TcpListener(listener))
            .await
            .expect("hub velo");
        let controller_peer = velo.peer_info();
        let landing = dir.path().join("landing");
        let allowed: HashSet<String> = [rel.to_owned()].into_iter().collect();

        let (authority, credentials) = CellRegistrationAuthority::mint(1).expect("authority");
        let plugin = ArtifactHubPlugin::new(landing.clone(), allowed, Arc::new(authority));
        let slot = plugin.receiver_slot();
        let mut hub = Hub::new(velo);
        hub.register(Box::new(plugin))
            .expect("register artifact plugin");
        assert_eq!(hub.prefixes().collect::<Vec<_>>(), vec!["/artifact"]);
        let receiver = slot
            .lock()
            .unwrap()
            .take()
            .expect("receiver captured by registration");

        // A real cell uses the trusted hub peer and its provisioned credential.
        let cell_velo = build_velo(BindSpec::TcpLoopback).await.expect("cell velo");
        cell_velo
            .register_peer(controller_peer.clone())
            .expect("register controller");
        ship_cell_artifacts_velo(
            &cell_velo,
            &controller_peer,
            0,
            &credentials[0],
            &src_dir,
            &[rel.to_owned()],
        )
        .await
        .expect("ship over hub");
        receiver
            .wait_for_cells(1, Duration::from_secs(30))
            .await
            .expect("barrier");

        assert_eq!(
            std::fs::read(landing.join("cell-0").join(rel)).unwrap(),
            source_bytes,
            "hub-mounted artifact plane landed the file byte-identical"
        );
    }
}
