// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Discovery-free connection seam for the velo cell transport.
//!
//! velo targets peers by a random per-run `InstanceId`, so a cell cannot address
//! the controller from a hardcoded `host:port` alone. This module implements
//! **mechanism B** (bootstrap-PeerInfo fetch, verified in
//! `examples/velo_cell_spike.rs`): the controller publishes its real, fully
//! public, serde `PeerInfo` at one operator-hardcoded coordinate; a cell fetches
//! it, `register_peer`s it, and then speaks velo. No etcd/NATS/velo-discovery
//! backend, and no velo change — the only a-priori fact a cell needs is the one
//! coordinate (`file:PATH` for a co-located launcher, `tcp://HOST:PORT` for k8s,
//! injected as `AIPERF_CELL_CONTROLLER_ADDR`).
//!
//! The seam is deliberately transport-neutral about *how* the bytes move: a
//! future coordinate scheme (a shared object store, a k8s ConfigMap projection)
//! is a new [`BootstrapSource`] variant, not a redesign.

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::task::JoinHandle;
use velo::transports::tcp::TcpTransportBuilder;
use velo::{PeerInfo, Transport, Velo};

/// A bootstrap payload larger than this is rejected rather than allocated — a
/// serialized `PeerInfo` is a few hundred bytes; this is generous headroom.
const BOOTSTRAP_MAX_LEN: u32 = 8 * 1024 * 1024;

/// How long a cell keeps retrying the bootstrap fetch before giving up (the
/// controller may not have published yet when a k8s cell pod starts first).
const BOOTSTRAP_FETCH_TIMEOUT: Duration = Duration::from_secs(60);

/// One retry interval for the bootstrap fetch.
const BOOTSTRAP_RETRY_INTERVAL: Duration = Duration::from_millis(200);

/// How the controller or a cell binds its velo messaging transport.
pub enum BindSpec {
    /// Unix domain socket at this path (local launcher, unix — lower overhead).
    #[cfg(unix)]
    UdsPath(PathBuf),
    /// TCP on an OS-assigned loopback port (local launcher on non-unix, e.g.
    /// Windows, or when UDS is unavailable).
    TcpLoopback,
    /// TCP bound to a fixed address (the k8s controller at its known port).
    TcpBind(SocketAddr),
}

/// Where the controller publishes, and a cell fetches, the controller's
/// serialized `PeerInfo` — parsed from the operator-hardcoded coordinate string.
pub enum BootstrapSource {
    /// A local file the controller writes (co-located launcher, same host).
    File(PathBuf),
    /// A TCP endpoint the controller serves (k8s cross-pod). For a cell this is
    /// the controller's `host:port`; for the controller's own listener it is the
    /// bind address (e.g. `0.0.0.0:PORT`).
    Tcp(String),
}

impl BootstrapSource {
    /// Parse a coordinate string: `file:PATH` or `tcp://HOST:PORT`.
    pub fn parse(coordinate: &str) -> Result<Self> {
        if let Some(path) = coordinate.strip_prefix("file:") {
            Ok(Self::File(PathBuf::from(path)))
        } else if let Some(addr) = coordinate.strip_prefix("tcp://") {
            Ok(Self::Tcp(addr.to_owned()))
        } else {
            bail!(
                "unrecognized controller bootstrap coordinate {coordinate:?}; \
                 expected `file:PATH` or `tcp://HOST:PORT`"
            )
        }
    }
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

/// Publish the controller's `PeerInfo` at `source` so cells reach it by the one
/// coordinate. `File` writes the bytes once and returns a completed handle;
/// `Tcp` spawns a listener that serves `u32` BE length + rmp(`PeerInfo`) to each
/// connection (a cell fetches once at startup). The returned handle is aborted
/// on drop by the caller (the controller holds it for the run).
pub async fn serve_bootstrap(source: &BootstrapSource, peer: &PeerInfo) -> Result<JoinHandle<()>> {
    let bytes = rmp_serde::to_vec(peer).context("encode controller PeerInfo")?;
    match source {
        BootstrapSource::File(path) => {
            tokio::fs::write(path, &bytes)
                .await
                .with_context(|| format!("write bootstrap file {}", path.display()))?;
            Ok(tokio::spawn(async {}))
        }
        BootstrapSource::Tcp(addr) => {
            let listener = TcpListener::bind(addr)
                .await
                .with_context(|| format!("bind bootstrap listener {addr}"))?;
            let bytes = Arc::new(bytes);
            Ok(tokio::spawn(async move {
                loop {
                    match listener.accept().await {
                        Ok((mut socket, _)) => {
                            let bytes = bytes.clone();
                            tokio::spawn(async move {
                                let len = (bytes.len() as u32).to_be_bytes();
                                let _ = socket.write_all(&len).await;
                                let _ = socket.write_all(&bytes).await;
                                let _ = socket.flush().await;
                            });
                        }
                        Err(_) => break,
                    }
                }
            }))
        }
    }
}

/// Fetch + decode the controller's `PeerInfo` from `source`, retrying until the
/// controller has published or [`BOOTSTRAP_FETCH_TIMEOUT`] elapses (a k8s cell
/// pod may start before the controller's listener is up).
pub async fn resolve_controller_peer(source: &BootstrapSource) -> Result<PeerInfo> {
    let deadline = tokio::time::Instant::now() + BOOTSTRAP_FETCH_TIMEOUT;
    let mut last_error: Option<anyhow::Error> = None;
    loop {
        match fetch_bootstrap(source).await {
            Ok(bytes) => {
                return rmp_serde::from_slice(&bytes).context("decode controller PeerInfo");
            }
            Err(error) => last_error = Some(error),
        }
        if tokio::time::Instant::now() >= deadline {
            return Err(last_error.unwrap_or_else(|| anyhow::anyhow!("bootstrap fetch failed")))
                .context("resolve controller PeerInfo (timed out)");
        }
        tokio::time::sleep(BOOTSTRAP_RETRY_INTERVAL).await;
    }
}

async fn fetch_bootstrap(source: &BootstrapSource) -> Result<Vec<u8>> {
    match source {
        BootstrapSource::File(path) => tokio::fs::read(path)
            .await
            .with_context(|| format!("read bootstrap file {}", path.display())),
        BootstrapSource::Tcp(addr) => {
            let mut socket = TcpStream::connect(addr)
                .await
                .with_context(|| format!("connect bootstrap {addr}"))?;
            let mut len_buf = [0_u8; 4];
            socket
                .read_exact(&mut len_buf)
                .await
                .context("read bootstrap length")?;
            let len = u32::from_be_bytes(len_buf);
            if len > BOOTSTRAP_MAX_LEN {
                bail!("bootstrap payload of {len} bytes exceeds the limit");
            }
            let mut body = vec![0_u8; len as usize];
            socket
                .read_exact(&mut body)
                .await
                .context("read bootstrap body")?;
            Ok(body)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a throwaway velo instance and round-trip its PeerInfo through both
    /// bootstrap sources, proving `serve_bootstrap` + `resolve_controller_peer`
    /// reconstruct the identical peer a cell would `register_peer`.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn bootstrap_round_trips_controller_peer_over_file_and_tcp() {
        let controller = build_velo(BindSpec::TcpLoopback).await.expect("build velo");
        let peer = controller.peer_info();
        let expected = rmp_serde::to_vec(&peer).unwrap();

        // File source.
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("controller-peer.rmp");
        let source = BootstrapSource::File(file.clone());
        let _h = serve_bootstrap(&source, &peer).await.expect("serve file");
        let got = resolve_controller_peer(&source).await.expect("resolve file");
        assert_eq!(rmp_serde::to_vec(&got).unwrap(), expected);

        // Tcp source.
        let source = BootstrapSource::Tcp("127.0.0.1:0".to_owned());
        // Bind an ephemeral port ourselves so we know the address to fetch from.
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        drop(listener);
        let serve_source = BootstrapSource::Tcp(addr.to_string());
        let _h2 = serve_bootstrap(&serve_source, &peer).await.expect("serve tcp");
        let fetch_source = BootstrapSource::Tcp(addr.to_string());
        let got = resolve_controller_peer(&fetch_source).await.expect("resolve tcp");
        assert_eq!(rmp_serde::to_vec(&got).unwrap(), expected);
        let _ = source; // silence unused in case the ephemeral bind path changes
    }

    #[test]
    fn parse_recognizes_file_and_tcp_coordinates() {
        assert!(matches!(
            BootstrapSource::parse("file:/tmp/x").unwrap(),
            BootstrapSource::File(_)
        ));
        assert!(matches!(
            BootstrapSource::parse("tcp://host:9500").unwrap(),
            BootstrapSource::Tcp(_)
        ));
        assert!(BootstrapSource::parse("http://nope").is_err());
    }
}
