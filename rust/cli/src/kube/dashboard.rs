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

    /// Fetch one bounded operator results document, or `None` when it is not
    /// retrievable. Browsing must degrade rather than fail the whole request, so
    /// an unreachable or unsuccessful operator yields no document.
    fn fetch(&self, path: &str) -> Option<Value> {
        let response = match self.client.execute("GET", path, "", Vec::new()) {
            Ok(response) => response,
            Err(error) => {
                tracing::warn!(error = %error, path, "operator results request failed");
                return None;
            }
        };
        if !response.is_success() {
            tracing::warn!(
                status = response.status,
                path,
                "operator results request was unsuccessful"
            );
            return None;
        }
        match serde_json::from_slice(&response.body) {
            Ok(document) => Some(document),
            Err(error) => {
                tracing::warn!(error = %error, path, "operator results response is not JSON");
                None
            }
        }
    }
}

/// Map one retained result-index item onto the dashboard's run entry.
///
/// The operator index carries no metric report, so the headline map stays empty;
/// the dashboard fills a run's metrics from its report when the detail endpoints
/// resolve it.
fn run_entry(namespace: &str, item: &Value) -> Option<RunEntry> {
    let run_id = item.pointer("/metadata/name").and_then(Value::as_str)?;
    let job_id = item.get("jobId").and_then(Value::as_str)?;
    let artifact_dir = format!("{namespace}/{job_id}/{run_id}");
    Some(RunEntry {
        id: id_for(&format!("{job_id}/{run_id}")),
        label: run_id.to_string(),
        artifact_dir,
        // The report is not a local file; the detail endpoints resolve it through
        // `read_report` instead.
        report_path: None,
        success: item.get("ready").and_then(Value::as_bool).unwrap_or(false),
        trial: 0,
        sweep_id: None,
        headline: std::collections::BTreeMap::new(),
        source: "operator",
    })
}

impl HistoricalSource for OperatorSource {
    fn list(&self) -> Vec<RunEntry> {
        let path = format!(
            "{}/api/results/{}",
            self.operator_prefix,
            super::command::encode_segment(&self.namespace)
        );
        let Some(document) = self.fetch(&path) else {
            return Vec::new();
        };
        document
            .get("items")
            .and_then(Value::as_array)
            .map(|items| {
                items
                    .iter()
                    .filter_map(|item| run_entry(&self.namespace, item))
                    .collect()
            })
            .unwrap_or_default()
    }

    fn read_report(&self, run: &RunEntry) -> Option<Value> {
        // `artifact_dir` is the virtual `"<namespace>/<job-id>/<run-id>"` coordinate
        // `run_entry` wrote; anything else did not come from this source.
        let mut parts = run.artifact_dir.split('/');
        let (namespace, job_id, run_id) = (parts.next()?, parts.next()?, parts.next()?);
        if parts.next().is_some() {
            return None;
        }
        let path = format!(
            "{}/api/results/{}/{}/{}/artifacts/{}",
            self.operator_prefix,
            super::command::encode_segment(namespace),
            super::command::encode_segment(job_id),
            super::command::encode_segment(run_id),
            crate::server::index::NATIVE_REPORT_NAME
        );
        self.fetch(&path)
    }
}

/// Serve the local dashboard against one namespace's operator-retained results.
///
/// The listener is loopback-only, so the session is reachable from this machine
/// alone; the upstream leg is the authenticated Kubernetes Service proxy, so no
/// cluster port is opened and no `kubectl` process is spawned.
pub fn run(args: &[String]) -> anyhow::Result<i32> {
    if args.iter().any(|argument| argument == "--help") {
        println!(
            "aiperf kube dashboard [--namespace <namespace>] [--port <port>] [--operator-service <name>] [--operator-namespace <namespace>]"
        );
        return Ok(0);
    }
    let port = match super::command::flag_value(args, "--port") {
        Some(value) => value
            .parse::<u16>()
            .map_err(|error| anyhow::anyhow!("--port must be a TCP port: {error}"))?,
        None => 0,
    };
    let client = KubeClient::from_options(&super::command::auth_options(args)?)?;
    let namespace = super::command::namespace(args)?.to_string();
    let operator_prefix = super::command::operator_service_proxy(args)?;
    let source = OperatorSource::new(client, namespace.clone(), operator_prefix);

    // Must precede `server::start`: threads inherit this thread's signal mask, and
    // a server thread with SIGINT/SIGTERM unblocked terminates the process before
    // the graceful path below can run.
    crate::serve::block_shutdown_signals();
    let handle = crate::server::start(
        ServerConfig {
            bind: SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), port),
            // Nothing is browsed from disk; the operator source owns the history.
            results_root: None,
            historical: Some(Arc::new(source)),
        },
        // A cluster dashboard has no local run loop, so its session index stays
        // empty and its live slot stays unset.
        Arc::new(Mutex::new(Vec::new())),
        Arc::new(Mutex::new(None)),
    )?;

    println!("Dashboard: http://{}", handle.local_addr());
    println!("aiperf: browsing namespace {namespace} through the operator results API");
    println!("aiperf: press Ctrl-C to stop");
    crate::serve::wait_for_shutdown();
    handle.shutdown();
    Ok(0)
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
