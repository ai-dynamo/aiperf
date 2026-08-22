// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! In-process loopback-only forwarding for `aiperf kube dashboard`.
//!
//! The listener binds `127.0.0.1` exclusively so a dashboard session is never
//! reachable off the operator's machine, and every accepted connection is
//! re-checked to be loopback before any byte is proxied. No `kubectl` process
//! is spawned; the upstream leg uses the same authenticated native client seam.

use std::io::{ErrorKind, Read, Write};
use std::net::{IpAddr, Ipv4Addr, SocketAddr, TcpListener, TcpStream};
use std::thread::sleep;
use std::time::Duration;

use super::error::KubeError;

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
