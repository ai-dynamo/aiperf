// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Discovery-free connection seam for the velo cell transport.
//!
//! A cell reaches the controller with **`velo.connect(Endpoint)`** — velo's
//! address-first bootstrap handshake (`ajcasagrande/velo` `feat/connect-by-endpoint`):
//! it dials the controller's operator/launcher-injected endpoint, learns the
//! controller's real `PeerInfo` via the `_hello` handshake, and mutually registers.
//! No discovery backend, no bootstrap side-channel, no forged identities — the only
//! a-priori fact a cell needs is the one endpoint (`AIPERF_CELL_CONTROLLER_ADDR`,
//! `tcp://HOST:PORT`; also `uds://PATH` for a pure-local run without HTTP artifact
//! shipping).
//!
//! The coordinate stays a `tcp://HOST:PORT` string in every shipping deployment so
//! the HTTP artifact plane (`runner_protocol::artifact_shipping`, which derives its
//! authority by swapping the port on the same coordinate) keeps working.

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use velo::transports::tcp::TcpTransportBuilder;
use velo::{Endpoint, PeerInfo, Transport, Velo};

/// How long a cell keeps retrying `connect` before giving up (the controller may
/// not have bound its listener yet when a k8s cell pod starts first).
const CONNECT_TIMEOUT: Duration = Duration::from_secs(60);

/// One retry interval for `connect`.
const CONNECT_RETRY_INTERVAL: Duration = Duration::from_millis(200);

/// How the controller or a cell binds its velo messaging transport.
pub enum BindSpec {
    /// Unix domain socket at this path (pure-local unix run without HTTP artifact
    /// shipping — the coordinate is then `uds://PATH`).
    #[cfg(unix)]
    UdsPath(PathBuf),
    /// TCP on an OS-assigned loopback port.
    TcpLoopback,
    /// TCP bound to a fixed address (the k8s controller at its operator-known port).
    TcpBind(SocketAddr),
    /// TCP over a caller-provided, already-bound listener — lets the caller read the
    /// OS-assigned port back (for the endpoint string) before handing it to velo.
    TcpListener(std::net::TcpListener),
}

/// Build a velo instance bound per `bind`. The cellular control plane is off the
/// per-request hot path, so a single messaging transport is sufficient.
pub async fn build_velo(bind: BindSpec) -> Result<Arc<Velo>> {
    let transport: Arc<dyn Transport> = match bind {
        #[cfg(unix)]
        BindSpec::UdsPath(path) => Arc::new(
            velo::transports::uds::UdsTransportBuilder::new()
                .socket_path(path)
                .build()
                .context("build uds transport")?,
        ),
        BindSpec::TcpLoopback => build_tcp_transport("127.0.0.1:0")?,
        BindSpec::TcpBind(addr) => build_tcp_transport(addr)?,
        BindSpec::TcpListener(listener) => Arc::new(
            TcpTransportBuilder::new()
                .from_listener(listener)
                .context("velo tcp from_listener")?
                .build()
                .context("build velo tcp transport")?,
        ),
    };
    Velo::builder()
        .add_transport(transport)
        .build()
        .await
        .context("build velo instance")
}

fn build_tcp_transport(addr: impl std::net::ToSocketAddrs) -> Result<Arc<dyn Transport>> {
    let listener = std::net::TcpListener::bind(addr).context("bind velo tcp listener")?;
    Ok(Arc::new(
        TcpTransportBuilder::new()
            .from_listener(listener)
            .context("velo tcp from_listener")?
            .build()
            .context("build velo tcp transport")?,
    ))
}

/// Parse a controller endpoint coordinate into a velo [`Endpoint`]: `tcp://HOST:PORT`
/// or `uds://PATH` (unix, pure-local).
pub fn parse_endpoint(coordinate: &str) -> Result<Endpoint> {
    if let Some(addr) = coordinate.strip_prefix("tcp://") {
        let socket: SocketAddr = addr
            .parse()
            .with_context(|| format!("parsing tcp endpoint {addr:?}"))?;
        return Ok(Endpoint::Tcp(socket));
    }
    if let Some(path) = coordinate.strip_prefix("uds://") {
        #[cfg(unix)]
        {
            return Ok(Endpoint::Uds(PathBuf::from(path)));
        }
        #[cfg(not(unix))]
        {
            let _ = path;
            bail!("uds endpoints are unix-only: {coordinate:?}");
        }
    }
    bail!("unrecognized controller endpoint {coordinate:?}; expected tcp://HOST:PORT or uds://PATH")
}

/// Connect to the controller at `coordinate`, retrying until it is reachable or
/// [`CONNECT_TIMEOUT`] elapses, and return its `PeerInfo`. Wraps `velo.connect`.
pub async fn connect_controller(velo: &Velo, coordinate: &str) -> Result<PeerInfo> {
    let endpoint = parse_endpoint(coordinate)?;
    let deadline = tokio::time::Instant::now() + CONNECT_TIMEOUT;
    loop {
        match velo.connect(endpoint.clone()).await {
            Ok(peer) => return Ok(peer),
            Err(error) => {
                if tokio::time::Instant::now() >= deadline {
                    return Err(error).context("connecting to controller (timed out)");
                }
            }
        }
        tokio::time::sleep(CONNECT_RETRY_INTERVAL).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_recognizes_tcp_and_uds_endpoints() {
        assert!(matches!(
            parse_endpoint("tcp://127.0.0.1:9500").unwrap(),
            Endpoint::Tcp(_)
        ));
        assert!(parse_endpoint("http://nope").is_err());
        assert!(parse_endpoint("tcp://not-an-addr").is_err());
        #[cfg(unix)]
        assert!(matches!(
            parse_endpoint("uds:///tmp/controller.sock").unwrap(),
            Endpoint::Uds(_)
        ));
    }

    /// A cell `connect`s the controller by TCP address alone (no PeerInfo), and the
    /// returned peer is the controller's real identity.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn connect_controller_bootstraps_by_endpoint() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let controller = build_velo(BindSpec::TcpListener(listener))
            .await
            .expect("controller velo");

        let cell = build_velo(BindSpec::TcpLoopback).await.expect("cell velo");
        let peer = connect_controller(&cell, &format!("tcp://{addr}"))
            .await
            .expect("connect");
        assert_eq!(peer.instance_id(), controller.instance_id());
    }
}
