// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `aiperf kube dashboard`: the local dashboard server pointed at the operator.
//!
//! The command starts the same in-process dashboard server (`crate::server`) and
//! the same SPA that `aiperf serve` starts, and swaps only its historical results
//! source: instead of walking a local artifact directory it reads the operator's
//! results API through the authenticated Kubernetes Service proxy. No `kubectl`
//! process is spawned and no cluster port is opened — the upstream leg is the
//! native client seam and the downstream leg binds `127.0.0.1` exclusively, so a
//! dashboard session is never reachable off the operator's machine.
//!
//! [`LoopbackForwarder`] is the raw loopback-only accept seam that enforces the
//! same locality guarantee for byte-forwarding callers.

use std::io::{ErrorKind, Read, Write};
use std::net::{IpAddr, Ipv4Addr, SocketAddr, TcpListener, TcpStream};
use std::sync::{Arc, Mutex};
use std::thread::sleep;
use std::time::Duration;

use serde_json::Value;

use super::client::KubeClient;
use super::error::KubeError;
use crate::server::index::{RunEntry, id_for};
use crate::server::{HistoricalSource, ServerConfig};

/// Bytes copied in one direction before the forwarder yields to its peer leg.
const RELAY_CHUNK_BYTES: usize = 64 * 1024;

/// A bound loopback listener that accepts only local dashboard clients.
pub struct LoopbackForwarder {
    listener: TcpListener,
}

impl LoopbackForwarder {
    /// Bind `127.0.0.1:<port>`; port `0` selects an ephemeral port.
    pub fn bind(port: u16) -> Result<Self, KubeError> {
        let address = SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), port);
        let listener = TcpListener::bind(address)?;
        Ok(Self { listener })
    }

    /// Return the exact bound loopback address.
    pub fn local_address(&self) -> Result<SocketAddr, KubeError> {
        Ok(self.listener.local_addr()?)
    }

    /// Accept one connection, refusing any peer that is not loopback.
    pub fn accept_local(&self) -> Result<TcpStream, KubeError> {
        let (stream, peer) = self.listener.accept()?;
        if !is_loopback(&peer) {
            // Dropping the stream closes it before any payload is exchanged.
            return Err(KubeError::Transport(format!(
                "refused non-loopback dashboard client {peer}"
            )));
        }
        Ok(stream)
    }

    /// Keep accepting loopback dashboard connections until the owner cancels the session.
    pub fn serve_until_cancelled(&self, is_cancelled: impl Fn() -> bool) -> Result<(), KubeError> {
        self.listener.set_nonblocking(true)?;
        while !is_cancelled() {
            match self.listener.accept() {
                Ok((stream, peer)) if is_loopback(&peer) => drop(stream),
                Ok((_stream, peer)) => {
                    tracing::warn!(peer = %peer, "refused non-loopback dashboard client");
                }
                Err(error) if error.kind() == ErrorKind::WouldBlock => {
                    sleep(Duration::from_millis(10));
                }
                Err(error) => return Err(KubeError::Io(error)),
            }
        }
        Ok(())
    }
}

/// Return whether a peer address is on the loopback interface.
pub fn is_loopback(peer: &SocketAddr) -> bool {
    match peer.ip() {
        IpAddr::V4(address) => address.is_loopback(),
        IpAddr::V6(address) => address.is_loopback(),
    }
}

/// Copy one direction of a forwarded connection until the source closes.
pub fn relay(source: &mut impl Read, sink: &mut impl Write) -> Result<u64, KubeError> {
    let mut buffer = vec![0_u8; RELAY_CHUNK_BYTES];
    let mut copied = 0_u64;
    loop {
        let read = source.read(&mut buffer)?;
        if read == 0 {
            sink.flush()?;
            return Ok(copied);
        }
        sink.write_all(&buffer[..read])?;
        copied += read as u64;
    }
}

/// The dashboard's historical results source when it browses a cluster: the
/// operator's retained result index, reached over the same authenticated
/// Kubernetes Service proxy `aiperf kube index` and `aiperf kube results` use.
///
/// A run's `artifact_dir` is the virtual operator coordinate
/// `"<namespace>/<job-id>/<run-id>"`, not a local path: it is what the run list's
/// stable id hashes and what [`HistoricalSource::read_report`] decomposes to
/// address the run's report artifact.
pub struct OperatorSource {
    // The blocking client is `Send + Sync` (credentials are plain data and the
    // transport is an `Arc<dyn KubeTransport: Send + Sync>`), so the source needs
    // no interior synchronization to be shared across the server's threads.
    client: KubeClient,
    namespace: String,
    operator_prefix: String,
}

impl OperatorSource {
    /// Bind a source to one namespace behind one operator Service proxy prefix.
    pub fn new(client: KubeClient, namespace: String, operator_prefix: String) -> Self {
        Self {
            client,
            namespace,
            operator_prefix,
        }
    }
}

impl HistoricalSource for OperatorSource {
    fn list(&self) -> Vec<RunEntry> {
        todo!("read the operator result index")
    }

    fn read_report(&self, _run: &RunEntry) -> Option<Value> {
        todo!("read the operator report artifact")
    }
}

/// Serve the local dashboard against one namespace's operator-retained results.
pub fn run(_args: &[String]) -> anyhow::Result<i32> {
    todo!("serve the operator-backed dashboard")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    };

    #[test]
    fn forwarder_binds_loopback_only() {
        let forwarder = LoopbackForwarder::bind(0).expect("bind");
        let address = forwarder.local_address().expect("address");
        assert_eq!(address.ip(), IpAddr::V4(Ipv4Addr::LOCALHOST));
        assert!(address.port() > 0);
    }

    #[test]
    fn non_loopback_peers_are_rejected() {
        assert!(!is_loopback(&"10.0.0.5:9000".parse().expect("peer")));
        assert!(is_loopback(&"127.0.0.1:9000".parse().expect("peer")));
        assert!(is_loopback(&"[::1]:9000".parse().expect("peer")));
    }

    #[test]
    fn relay_copies_every_byte_without_reframing() {
        let payload = vec![7_u8; RELAY_CHUNK_BYTES + 11];
        let mut source = payload.as_slice();
        let mut sink = Vec::new();
        let copied = relay(&mut source, &mut sink).expect("relay");
        assert_eq!(copied, payload.len() as u64);
        assert_eq!(sink, payload);
    }

    #[test]
    fn listener_remains_bound_until_the_dashboard_session_is_cancelled() {
        let forwarder = LoopbackForwarder::bind(0).expect("bind");
        let address = forwarder.local_address().expect("address");
        let cancelled = Arc::new(AtomicBool::new(false));
        let stop = cancelled.clone();
        let serving = std::thread::spawn(move || {
            forwarder.serve_until_cancelled(|| stop.load(Ordering::Relaxed))
        });

        TcpStream::connect(address).expect("first connection");
        TcpStream::connect(address).expect("second connection");
        cancelled.store(true, Ordering::Relaxed);
        serving
            .join()
            .expect("server thread")
            .expect("dashboard session");
    }
}
