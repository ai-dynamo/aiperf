// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The discovery plugin: the hub's connect-by-endpoint anchor and the concrete
//! proof of the dual-surface handler property.
//!
//! One function, [`handle_discovery`], is the single source of truth. Both the
//! velo unary handler (on [`HUB_DISCOVERY`], `rmp`-encoded raw payloads matching
//! the cellular numeric-fidelity convention) and the HTTP route
//! (`POST {prefix}/hello`, JSON) call it, so a client reaching the hub by endpoint
//! over velo and a client POSTing over HTTP observe identical replies.

use std::sync::Arc;

use axum::Json;
use axum::extract::State;
use axum::routing::post;
use bytes::Bytes;
use serde::{Deserialize, Serialize};
use velo::{Context, Handler, Velo};

use super::plugin::{HubError, HubPlugin};

/// The velo unary handler name the discovery plugin registers.
pub const HUB_DISCOVERY: &str = "hub.discovery";

/// A client's discovery request: who is asking.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DiscoveryRequest {
    /// A caller-supplied client identity, echoed into the reply's greeting.
    pub client: String,
}

/// The hub's discovery reply: its identity, dial-able endpoint, and mounted
/// plugins.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DiscoveryReply {
    /// The hub's velo instance identity (opaque string).
    pub hub_instance: String,
    /// The hub's dial-able velo endpoint coordinate (`tcp://HOST:PORT` or
    /// `uds://PATH`), the same coordinate `connect_controller` accepts.
    pub endpoint: String,
    /// The prefixes of the hub's mounted plugins.
    pub plugins: Vec<String>,
    /// A greeting naming the hub and echoing the request's client.
    pub greeting: String,
}

/// The immutable facts the discovery handler serves. Constructed by the hub
/// bootstrap (which knows the bound velo endpoint and instance id) and shared,
/// via [`Arc`], between the velo and HTTP surfaces.
#[derive(Clone, Debug)]
pub struct DiscoveryState {
    /// The hub's velo instance identity (opaque string).
    pub hub_instance: String,
    /// The hub's dial-able velo endpoint coordinate.
    pub endpoint: String,
    /// The prefixes of the hub's mounted plugins.
    pub plugins: Vec<String>,
}

/// The single source of truth for a discovery reply. Both the velo handler and
/// the HTTP route call this, so the two surfaces cannot diverge.
pub fn handle_discovery(state: &DiscoveryState, request: DiscoveryRequest) -> DiscoveryReply {
    DiscoveryReply {
        hub_instance: state.hub_instance.clone(),
        endpoint: state.endpoint.clone(),
        plugins: state.plugins.clone(),
        greeting: format!("hub {} welcomes {}", state.hub_instance, request.client),
    }
}

/// The hub's first plugin: the connect-by-endpoint discovery anchor, exposing
/// [`handle_discovery`] over both a velo unary and an HTTP route.
pub struct DiscoveryPlugin {
    state: Arc<DiscoveryState>,
}

impl DiscoveryPlugin {
    /// Create the discovery plugin over the given immutable [`DiscoveryState`].
    pub fn new(state: DiscoveryState) -> Self {
        Self {
            state: Arc::new(state),
        }
    }
}

/// `POST {prefix}/hello` — decode a JSON [`DiscoveryRequest`], call the shared
/// [`handle_discovery`], and return the JSON [`DiscoveryReply`].
async fn http_hello(
    State(state): State<Arc<DiscoveryState>>,
    Json(request): Json<DiscoveryRequest>,
) -> Json<DiscoveryReply> {
    Json(handle_discovery(&state, request))
}

impl HubPlugin for DiscoveryPlugin {
    fn prefix(&self) -> &str {
        "/discovery"
    }

    fn router(&self) -> axum::Router {
        axum::Router::new()
            .route("/hello", post(http_hello))
            .with_state(self.state.clone())
    }

    fn register_velo_handlers(&self, velo: &Arc<Velo>) -> Result<(), HubError> {
        let state = self.state.clone();
        velo.register_handler(
            Handler::unary_handler_async(HUB_DISCOVERY, move |ctx: Context| {
                let state = state.clone();
                async move {
                    let request: DiscoveryRequest = rmp_serde::from_slice(&ctx.payload)
                        .map_err(|error| anyhow::anyhow!("decode DiscoveryRequest: {error}"))?;
                    let reply = handle_discovery(&state, request);
                    let bytes = rmp_serde::to_vec(&reply)
                        .map_err(|error| anyhow::anyhow!("encode DiscoveryReply: {error}"))?;
                    Ok(Some(Bytes::from(bytes)))
                }
            })
            .build(),
        )
        .map_err(|error| HubError::VeloHandler {
            prefix: self.prefix().to_owned(),
            message: error.to_string(),
        })
    }
}

#[cfg(test)]
mod tests {
    use std::net::SocketAddr;

    use super::*;
    use crate::cellular::transport::connect::{BindSpec, build_velo, connect_controller};
    use crate::hub::Hub;

    /// A minimal second plugin (no velo handlers) so registration/mount tests can
    /// exercise more than one prefix and the duplicate-prefix guard.
    struct NoopPlugin(&'static str);
    impl HubPlugin for NoopPlugin {
        fn prefix(&self) -> &str {
            self.0
        }
        fn router(&self) -> axum::Router {
            axum::Router::new().route("/ping", axum::routing::get(|| async { "pong" }))
        }
        fn register_velo_handlers(&self, _velo: &Arc<Velo>) -> Result<(), HubError> {
            Ok(())
        }
    }

    fn discovery_state(endpoint: String) -> DiscoveryState {
        DiscoveryState {
            hub_instance: "hub-test".to_owned(),
            endpoint,
            plugins: vec!["/discovery".to_owned()],
        }
    }

    /// A hub with a fresh velo instance and the discovery plugin, plus the velo
    /// endpoint coordinate a client dials. Returns (hub, endpoint).
    async fn build_hub_with_discovery() -> (Hub, String) {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let endpoint = format!("tcp://{addr}");
        let velo = build_velo(BindSpec::TcpListener(listener))
            .await
            .expect("hub velo");
        let mut hub = Hub::new(velo);
        hub.register(Box::new(DiscoveryPlugin::new(discovery_state(
            endpoint.clone(),
        ))))
        .expect("register discovery");
        (hub, endpoint)
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn registration_rejects_duplicate_and_invalid_prefixes() {
        let velo = build_velo(BindSpec::TcpLoopback).await.expect("velo");
        let mut hub = Hub::new(velo);

        hub.register(Box::new(NoopPlugin("/a"))).expect("first ok");
        hub.register(Box::new(NoopPlugin("/b"))).expect("second ok");
        assert_eq!(hub.prefixes().collect::<Vec<_>>(), vec!["/a", "/b"]);

        // A duplicate prefix is rejected and the plugin set is unchanged.
        let dup = hub.register(Box::new(NoopPlugin("/a")));
        assert!(matches!(dup, Err(HubError::DuplicatePrefix(_))), "{dup:?}");
        assert_eq!(hub.prefixes().len(), 2);

        // An invalid (no leading slash) prefix is rejected.
        let bad = hub.register(Box::new(NoopPlugin("nope")));
        assert!(matches!(bad, Err(HubError::InvalidPrefix(_))), "{bad:?}");
        assert_eq!(hub.prefixes().len(), 2);
    }

    /// POST the discovery request over HTTP with a raw hyper client (direct
    /// `TcpStream::connect`, so ambient proxy settings are never consulted).
    async fn http_discovery(addr: SocketAddr, request: &DiscoveryRequest) -> DiscoveryReply {
        use http_body_util::{BodyExt, Full};
        use hyper_util::rt::TokioIo;

        let stream = tokio::net::TcpStream::connect(addr).await.expect("connect");
        let io = TokioIo::new(stream);
        let (mut sender, conn) = hyper::client::conn::http1::handshake(io)
            .await
            .expect("handshake");
        tokio::spawn(async move {
            let _ = conn.await;
        });
        let json = serde_json::to_vec(request).expect("encode request");
        let http_request = hyper::Request::builder()
            .method("POST")
            .uri("/discovery/hello")
            .header(hyper::header::HOST, "localhost")
            .header(hyper::header::CONTENT_TYPE, "application/json")
            .body(Full::new(Bytes::from(json)))
            .expect("build request");
        let response = sender
            .send_request(http_request)
            .await
            .expect("send request");
        assert!(
            response.status().is_success(),
            "http status {}",
            response.status()
        );
        let bytes = response
            .into_body()
            .collect()
            .await
            .expect("collect body")
            .to_bytes();
        serde_json::from_slice(&bytes).expect("decode reply")
    }

    /// Reach the discovery handler over a velo unary: connect to the hub by
    /// endpoint alone (`_hello` bootstrap), then send the `rmp`-encoded request.
    async fn velo_discovery(endpoint: &str, request: &DiscoveryRequest) -> DiscoveryReply {
        let client = build_velo(BindSpec::TcpLoopback)
            .await
            .expect("client velo");
        let peer = connect_controller(&client, endpoint)
            .await
            .expect("connect to hub");
        let body = rmp_serde::to_vec(request).expect("encode request");
        let reply: Bytes = client
            .unary(HUB_DISCOVERY)
            .expect("unary builder")
            .raw_payload(Bytes::from(body))
            .instance(peer.instance_id())
            .send()
            .await
            .expect("unary send");
        rmp_serde::from_slice(&reply).expect("decode reply")
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn discovery_is_identical_over_velo_and_http() {
        let (hub, endpoint) = build_hub_with_discovery().await;
        let server = hub
            .serve("127.0.0.1:0".parse().unwrap())
            .await
            .expect("serve hub");
        let http_addr = server.http_addr();

        let request = DiscoveryRequest {
            client: "cell-7".to_owned(),
        };

        let over_velo = velo_discovery(&endpoint, &request).await;
        let over_http = http_discovery(http_addr, &request).await;

        // The dual-surface property: the same handler backs both paths.
        assert_eq!(
            over_velo, over_http,
            "velo and HTTP discovery replies must be identical"
        );
        // And the shared handler actually ran (not two empty defaults).
        assert_eq!(over_velo.endpoint, endpoint);
        assert_eq!(over_velo.plugins, vec!["/discovery".to_owned()]);
        assert_eq!(over_velo.greeting, "hub hub-test welcomes cell-7");

        server.shutdown().await;
    }
}
