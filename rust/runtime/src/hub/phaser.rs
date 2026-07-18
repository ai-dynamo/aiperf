// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The `/phaser` hub plugin: folds the velo monotonic-phaser control plane
//! ([`PhaserServer`]) onto the shared hub velo instance, so cells subscribe to the
//! controller's phaser broadcast over the one hub anchor instead of a plane bound
//! directly on the control-plane velo.
//!
//! The plugin reuses the phaser distribution verbatim — its
//! [`register_velo_handlers`](HubPlugin::register_velo_handlers) installs exactly the
//! `aiperf.phaser.subscribe` / `aiperf.phaser.event` handlers via [`PhaserServer::bind`],
//! serving the same [`Phaser`] the bootstrap `advance`s; only the mount point moves onto
//! the hub. The bound [`PhaserServer`] is retained inside the plugin (it only holds a
//! clone of the hub velo instance the handlers are registered on, so the control plane
//! survives the plugin's drop at [`Hub::serve`](super::Hub::serve) — the handlers and
//! their per-cell pump tasks live on the hub velo the [`HubServer`](super::HubServer)
//! holds).
//!
//! The HTTP surface is a **dual-surface diagnostic**: `GET {prefix}/status` reports the
//! plane's handler names plus the phaser's live `current_generation`, so the same phaser
//! state the velo subscribe handler replays is observable over HTTP from the same plugin.

use std::sync::{Arc, Mutex};

use axum::Json;
use axum::extract::State;
use axum::routing::get;
use serde_json::json;
use velo::Velo;

use super::plugin::{HubAbiRequirement, HubError, HubPlugin};
use crate::cellular::phaser::Phaser;
use crate::cellular::transport::phaser_velo::{
    HANDLER_PHASER_EVENT, HANDLER_PHASER_SUBSCRIBE, PhaserServer,
};

/// The default HTTP mount point / diagnostic identity for the phaser plugin.
pub const PHASER_PREFIX: &str = "/phaser";

/// The `/phaser` hub plugin. Registers the phaser control-plane velo handlers on the
/// shared hub velo instance, serving the bootstrap's [`Phaser`].
pub struct PhaserHubPlugin {
    prefix: String,
    phaser: Phaser,
    /// The bound server, retained so its (redundant) velo clone lives with the plugin.
    /// The handlers themselves ride the hub velo and outlive this.
    server: Mutex<Option<PhaserServer>>,
}

impl PhaserHubPlugin {
    /// Build the plugin over the controller's [`Phaser`] (the same instance the bootstrap
    /// `advance`s). Uses the default [`PHASER_PREFIX`].
    pub fn new(phaser: Phaser) -> Self {
        Self {
            prefix: PHASER_PREFIX.to_owned(),
            phaser,
            server: Mutex::new(None),
        }
    }
}

/// The immutable + live facts the diagnostic HTTP surface reports.
#[derive(Clone)]
struct PhaserStatus {
    prefix: String,
    phaser: Phaser,
}

/// `GET {prefix}/status` — the dual-surface diagnostic: the control-plane handler names
/// plus the phaser's live generation (the same state the velo subscribe handler replays).
async fn http_status(State(status): State<Arc<PhaserStatus>>) -> Json<serde_json::Value> {
    Json(json!({
        "plugin": status.prefix,
        "generation": status.phaser.current_generation(),
        "subscribe_handler": HANDLER_PHASER_SUBSCRIBE,
        "event_handler": HANDLER_PHASER_EVENT,
    }))
}

impl HubPlugin for PhaserHubPlugin {
    fn prefix(&self) -> &str {
        &self.prefix
    }

    fn required_abi(&self) -> HubAbiRequirement {
        HubAbiRequirement::current()
    }

    fn router(&self) -> axum::Router {
        let status = Arc::new(PhaserStatus {
            prefix: self.prefix.clone(),
            phaser: self.phaser.clone(),
        });
        axum::Router::new()
            .route("/status", get(http_status))
            .with_state(status)
    }

    fn register_velo_handlers(&self, velo: &Arc<Velo>) -> Result<(), HubError> {
        let server =
            PhaserServer::bind(velo.clone(), self.phaser.clone()).map_err(|error| {
                HubError::VeloHandler {
                    prefix: self.prefix.clone(),
                    message: error.to_string(),
                }
            })?;
        *self
            .server
            .lock()
            .expect("phaser server slot poisoned") = Some(server);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cellular::phaser::PhaseTransition;
    use crate::cellular::transport::connect::{BindSpec, build_velo, connect_controller};
    use crate::cellular::transport::phaser_velo::PhaserClient;
    use crate::hub::Hub;

    /// The plugin mounts on a hub, registers the phaser velo handlers, and a real cell
    /// subscribing over the hub anchor observes replay-then-live generations — proving the
    /// control plane rides the one hub instance.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn plugin_mounts_and_serves_phaser_over_the_hub() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let coordinate = format!("tcp://{addr}");
        let velo = build_velo(BindSpec::TcpListener(listener))
            .await
            .expect("hub velo");

        let phaser = Phaser::new();
        // Advance twice BEFORE the cell subscribes — these must arrive as replay.
        phaser.advance(PhaseTransition::Started);
        phaser.advance(PhaseTransition::ShardsAvailable(4));

        let plugin = PhaserHubPlugin::new(phaser.clone());
        let mut hub = Hub::new(velo);
        hub.register(Box::new(plugin))
            .expect("register phaser plugin");
        assert_eq!(hub.prefixes().collect::<Vec<_>>(), vec!["/phaser"]);

        let server = hub
            .serve(std::net::SocketAddr::from(([127, 0, 0, 1], 0)))
            .await
            .expect("serve hub");

        // A real cell subscribes over the hub's velo anchor and observes replay + live.
        let cell_velo = build_velo(BindSpec::TcpLoopback).await.expect("cell velo");
        let controller = connect_controller(&cell_velo, &coordinate)
            .await
            .expect("connect");
        let mut sub = PhaserClient::subscribe(cell_velo, &controller)
            .await
            .expect("subscribe");
        sub.await_generation(2).await.expect("replayed to gen 2");
        assert!(sub.seen_generation() >= 2);

        phaser.advance(PhaseTransition::PhaseAdvance("profiling".into())); // gen 3
        sub.await_generation(3).await.expect("live to gen 3");
        assert!(sub.seen_generation() >= 3);

        // The dual-surface diagnostic reports the phaser's live generation. Raw hyper GET
        // (direct `TcpStream::connect`, so ambient proxy settings are never consulted).
        let body = http_get_json(server.http_addr(), "/phaser/status").await;
        assert_eq!(body["generation"], 3, "diagnostic reports the live generation");
        assert_eq!(body["subscribe_handler"], HANDLER_PHASER_SUBSCRIBE);

        server.shutdown().await;
    }

    /// GET `path` over a raw hyper client (no ambient proxy) and decode the JSON body.
    async fn http_get_json(addr: std::net::SocketAddr, path: &str) -> serde_json::Value {
        use http_body_util::{BodyExt, Empty};
        use hyper_util::rt::TokioIo;

        let stream = tokio::net::TcpStream::connect(addr).await.expect("connect");
        let io = TokioIo::new(stream);
        let (mut sender, conn) = hyper::client::conn::http1::handshake(io)
            .await
            .expect("handshake");
        tokio::spawn(async move {
            let _ = conn.await;
        });
        let request = hyper::Request::builder()
            .method("GET")
            .uri(path)
            .header(hyper::header::HOST, "localhost")
            .body(Empty::<bytes::Bytes>::new())
            .expect("build request");
        let response = sender.send_request(request).await.expect("send request");
        assert!(response.status().is_success(), "http status {}", response.status());
        let bytes = response
            .into_body()
            .collect()
            .await
            .expect("collect body")
            .to_bytes();
        serde_json::from_slice(&bytes).expect("decode json")
    }
}
