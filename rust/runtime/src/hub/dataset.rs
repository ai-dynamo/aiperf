// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The `/dataset` hub plugin: folds the velo dataset fan-out data plane
//! ([`DatasetServer`]) onto the shared hub velo instance, so cells subscribe to the
//! controller's dataset broadcast over the one hub anchor instead of a plane bound
//! directly on the control-plane velo.
//!
//! The plugin reuses the fan-out machinery verbatim — its
//! [`register_velo_handlers`](HubPlugin::register_velo_handlers) installs exactly the
//! `aiperf.dataset.subscribe` / `aiperf.dataset.chunk` handlers via
//! [`DatasetServer::bind`], serving the same [`DatasetPublisher`] the bootstrap fills
//! and finalizes; only the mount point moves onto the hub. The bound [`DatasetServer`]
//! is retained inside the plugin (it only holds a clone of the hub velo instance the
//! handlers are registered on, so the fan-out plane survives the plugin's drop at
//! [`Hub::serve`](super::Hub::serve) — the handlers and their per-cell pump tasks live
//! on the hub velo the [`HubServer`](super::HubServer) holds).
//!
//! The HTTP surface is a **dual-surface diagnostic**: `GET {prefix}/status` reports the
//! plane's fixed facts (handler names) plus the publisher's live `chunk_count`, so the
//! same publisher state the velo subscribe handler replays is observable over HTTP from
//! the same plugin.

use std::sync::{Arc, Mutex};

use axum::Json;
use axum::extract::State;
use axum::routing::get;
use serde_json::json;
use velo::Velo;

use super::plugin::{HubAbiRequirement, HubError, HubPlugin};
use crate::cellular::dataset_session::DatasetPublisher;
use crate::cellular::transport::dataset_velo::{
    DatasetServer, HANDLER_DATASET_CHUNK, HANDLER_DATASET_SUBSCRIBE, WirePayload,
};

/// The default HTTP mount point / diagnostic identity for the dataset fan-out plugin.
pub const DATASET_PREFIX: &str = "/dataset";

/// The `/dataset` hub plugin. Registers the dataset fan-out velo handlers on the shared
/// hub velo instance, serving the bootstrap's [`DatasetPublisher`].
pub struct DatasetHubPlugin {
    prefix: String,
    publisher: DatasetPublisher<WirePayload>,
    /// The bound server, retained so its (redundant) velo clone lives with the plugin.
    /// The handlers themselves ride the hub velo and outlive this.
    server: Mutex<Option<DatasetServer>>,
}

impl DatasetHubPlugin {
    /// Build the plugin over the controller's dataset [`DatasetPublisher`] (the same
    /// instance the bootstrap `add`s chunks to and `finalize`s). Uses the default
    /// [`DATASET_PREFIX`].
    pub fn new(publisher: DatasetPublisher<WirePayload>) -> Self {
        Self {
            prefix: DATASET_PREFIX.to_owned(),
            publisher,
            server: Mutex::new(None),
        }
    }
}

/// The immutable + live facts the diagnostic HTTP surface reports.
#[derive(Clone)]
struct DatasetStatus {
    prefix: String,
    publisher: DatasetPublisher<WirePayload>,
}

/// `GET {prefix}/status` — the dual-surface diagnostic: the fan-out handler names plus
/// the publisher's live chunk count (the same state the velo subscribe handler replays).
async fn http_status(State(status): State<Arc<DatasetStatus>>) -> Json<serde_json::Value> {
    Json(json!({
        "plugin": status.prefix,
        "chunk_count": status.publisher.chunk_count(),
        "subscribe_handler": HANDLER_DATASET_SUBSCRIBE,
        "chunk_handler": HANDLER_DATASET_CHUNK,
    }))
}

impl HubPlugin for DatasetHubPlugin {
    fn prefix(&self) -> &str {
        &self.prefix
    }

    fn required_abi(&self) -> HubAbiRequirement {
        HubAbiRequirement::current()
    }

    fn router(&self) -> axum::Router {
        let status = Arc::new(DatasetStatus {
            prefix: self.prefix.clone(),
            publisher: self.publisher.clone(),
        });
        axum::Router::new()
            .route("/status", get(http_status))
            .with_state(status)
    }

    fn register_velo_handlers(&self, velo: &Arc<Velo>) -> Result<(), HubError> {
        let server =
            DatasetServer::bind(velo.clone(), self.publisher.clone()).map_err(|error| {
                HubError::VeloHandler {
                    prefix: self.prefix.clone(),
                    message: error.to_string(),
                }
            })?;
        *self
            .server
            .lock()
            .expect("dataset server slot poisoned") = Some(server);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cellular::dataset_session::DatasetRequest;
    use crate::cellular::transport::connect::{BindSpec, build_velo, connect_controller};
    use crate::cellular::transport::dataset_velo::DatasetClient;
    use crate::hub::Hub;

    /// The plugin mounts on a hub, registers the fan-out velo handlers, and a real cell
    /// subscribes over the hub anchor and builds its owned index — proving the dataset
    /// plane rides the one hub instance.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn plugin_mounts_and_serves_fanout_over_the_hub() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let coordinate = format!("tcp://{addr}");
        let velo = build_velo(BindSpec::TcpListener(listener))
            .await
            .expect("hub velo");

        let publisher = DatasetPublisher::<WirePayload>::new();
        let chunk = |ids: std::ops::Range<u64>| {
            ids.map(|request_id| DatasetRequest {
                request_id,
                payload: format!("req-{request_id}").into_bytes(),
            })
            .collect::<Vec<_>>()
        };
        publisher.add(chunk(0..6));
        publisher.add(chunk(6..12));
        publisher.finalize();

        let plugin = DatasetHubPlugin::new(publisher.clone());
        let mut hub = Hub::new(velo);
        hub.register(Box::new(plugin))
            .expect("register dataset plugin");
        assert_eq!(hub.prefixes().collect::<Vec<_>>(), vec!["/dataset"]);

        // Serve so the diagnostic HTTP surface is reachable too.
        let server = hub
            .serve(std::net::SocketAddr::from(([127, 0, 0, 1], 0)))
            .await
            .expect("serve hub");

        // A real cell subscribes over the hub's velo anchor and builds its owned index.
        let cell_velo = build_velo(BindSpec::TcpLoopback).await.expect("cell velo");
        let controller = connect_controller(&cell_velo, &coordinate)
            .await
            .expect("connect");
        let index = DatasetClient::build_owned_index(cell_velo, &controller, |id| id % 3 == 0)
            .await
            .expect("build index");
        let expected: Vec<u64> = (0..12).filter(|id| id % 3 == 0).collect();
        assert_eq!(index.owned_ids(), expected, "cell 0 of 3 owned set over hub");

        // The dual-surface diagnostic reports the same publisher chunk count. Raw hyper
        // GET (direct `TcpStream::connect`, so ambient proxy settings are never consulted).
        let body = http_get_json(server.http_addr(), "/dataset/status").await;
        assert_eq!(body["chunk_count"], 2, "diagnostic reports the two chunks");
        assert_eq!(body["subscribe_handler"], HANDLER_DATASET_SUBSCRIBE);

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
