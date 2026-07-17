// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Multi-process round-robin load balancer for `aiperf-mock-server`.
//!
//! With `--processes N`, the parent re-executes `N` single-process children on
//! loopback ports, health-gates them, and forwards public TCP connections in
//! round-robin order. [`CONFIG_JSON_ENV`] carries the resolved configuration.
//!
//! The balancer is deliberately transport-transparent: it never parses HTTP, so
//! HTTP/1.1 keep-alive, HTTP/2, and SSE streaming all pass through untouched, and
//! clients see one OpenAI-compatible frontend.
//!
//! Round-robin is applied **per connection**, not per request — the cheapest
//! distribution that needs no backend connection pool or HTTP awareness. A
//! benchmark driving concurrency `C` opens ~`C` keep-alive connections, which
//! spread evenly across the `N` backends as long as `C >= N` (the intended
//! regime); below that some backends idle.

use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::process::{Child, Command};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use anyhow::{Context, bail};
use parking_lot::Mutex;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;
use tokio::sync::Notify;

use crate::config::MockServerConfig;
use crate::listener::build_listener;

/// Environment variable carrying the resolved [`MockServerConfig`] to children.
pub const CONFIG_JSON_ENV: &str = "MOCK_SERVER_CONFIG_JSON";

const HEALTH_GATE_TIMEOUT: Duration = Duration::from_secs(30);

const HEALTH_POLL_INTERVAL: Duration = Duration::from_millis(50);

const SUPERVISE_INTERVAL: Duration = Duration::from_millis(250);

/// Owns the spawned child processes and hard-kills them on drop, so the backends
/// never outlive the balancer (a crash, `?` bail, or normal shutdown all reap
/// them). Shared with the supervisor task via the inner `Arc<Mutex<..>>`.
struct ChildGroup {
    children: Arc<Mutex<Vec<Child>>>,
}

impl Drop for ChildGroup {
    fn drop(&mut self) {
        let mut children = self.children.lock();
        for child in children.iter_mut() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

/// Run until Ctrl-C or a child process exits.
pub fn run(parent: MockServerConfig, processes: usize) -> anyhow::Result<()> {
    debug_assert!(processes > 1);
    let ncpu = num_cpus::get().max(1);

    let child_workers = if parent.workers > 0 {
        parent.workers
    } else {
        (ncpu / processes).max(1)
    };
    let balancer_workers = (ncpu / processes).max(2);

    let ports = pick_free_ports(processes).context("selecting free backend ports")?;
    let exe = std::env::current_exe().context("resolving current executable path")?;

    let children = Arc::new(Mutex::new(Vec::with_capacity(processes)));
    for &port in &ports {
        let mut child_cfg = parent.clone();
        child_cfg.processes = 1;
        child_cfg.host = Ipv4Addr::LOCALHOST.to_string();
        child_cfg.port = port;
        child_cfg.workers = child_workers;
        // The balancer is HTTP-only; children must not each bind the gRPC port.
        child_cfg.grpc_port = None;

        let json = serde_json::to_string(&child_cfg).context("serializing child config")?;
        let mut cmd = Command::new(&exe);
        cmd.env(CONFIG_JSON_ENV, json);
        set_parent_death_signal(&mut cmd);
        let child = cmd
            .spawn()
            .with_context(|| format!("spawning backend child on 127.0.0.1:{port}"))?;
        children.lock().push(child);
    }
    // The guard reaps children on every return path.
    let _group = ChildGroup {
        children: children.clone(),
    };

    tracing::info!(
        processes,
        balancer_workers,
        child_workers,
        backends = ?ports,
        "Starting AIPerf Mock Server round-robin balancer"
    );

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .worker_threads(balancer_workers)
        .thread_name("aiperf-mock-lb")
        .build()?;

    runtime.block_on(serve_balancer(parent, ports, children))
}

async fn serve_balancer(
    parent: MockServerConfig,
    ports: Vec<u16>,
    children: Arc<Mutex<Vec<Child>>>,
) -> anyhow::Result<()> {
    for &port in &ports {
        wait_healthy(port, HEALTH_GATE_TIMEOUT)
            .await
            .with_context(|| format!("backend on 127.0.0.1:{port} did not become healthy"))?;
    }
    tracing::info!("All {} backends healthy", ports.len());

    let host: IpAddr = parent
        .host
        .parse()
        .unwrap_or(IpAddr::V4(Ipv4Addr::LOCALHOST));
    let addr = SocketAddr::new(host, parent.port);
    let listener = build_listener(addr).with_context(|| format!("binding balancer on {addr}"))?;
    tracing::info!(%addr, "Balancer listening; round-robin over {} backends", ports.len());

    let backends: Arc<Vec<SocketAddr>> = Arc::new(
        ports
            .iter()
            .map(|&p| SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), p))
            .collect(),
    );
    let counter = Arc::new(AtomicUsize::new(0));

    let shutdown = Arc::new(Notify::new());
    spawn_supervisor(children, shutdown.clone());
    spawn_signal_watch(shutdown.clone());

    loop {
        tokio::select! {
            biased;
            _ = shutdown.notified() => {
                tracing::info!("Balancer shutting down");
                break;
            }
            accepted = listener.accept() => {
                let (inbound, _peer) = match accepted {
                    Ok(v) => v,
                    Err(e) => {
                        tracing::warn!("accept error: {e}");
                        continue;
                    }
                };
                let _ = inbound.set_nodelay(true);
                let backend = pick_backend(&backends, &counter);
                tokio::spawn(async move {
                    proxy_connection(inbound, backend).await;
                });
            }
        }
    }
    Ok(())
}

/// `Relaxed` is sufficient because the counter does not synchronize other state.
fn pick_backend(backends: &[SocketAddr], counter: &AtomicUsize) -> SocketAddr {
    let idx = counter.fetch_add(1, Ordering::Relaxed) % backends.len();
    backends[idx]
}

async fn proxy_connection(mut inbound: TcpStream, backend: SocketAddr) {
    let mut outbound = match TcpStream::connect(backend).await {
        Ok(s) => s,
        Err(e) => {
            tracing::warn!(%backend, "backend connect failed: {e}");
            return;
        }
    };
    let _ = outbound.set_nodelay(true);
    if let Err(e) = tokio::io::copy_bidirectional(&mut inbound, &mut outbound).await {
        tracing::debug!(%backend, "connection closed: {e}");
    }
}

/// Shut down if a dead child would otherwise drop its share of traffic.
fn spawn_supervisor(children: Arc<Mutex<Vec<Child>>>, shutdown: Arc<Notify>) {
    tokio::spawn(async move {
        loop {
            tokio::time::sleep(SUPERVISE_INTERVAL).await;
            let dead = {
                let mut guard = children.lock();
                guard.iter_mut().find_map(|child| match child.try_wait() {
                    Ok(Some(status)) => Some(status),
                    _ => None,
                })
            };
            if let Some(status) = dead {
                tracing::error!(?status, "a backend child exited; shutting down balancer");
                shutdown.notify_one();
                break;
            }
        }
    });
}

fn spawn_signal_watch(shutdown: Arc<Notify>) {
    tokio::spawn(async move {
        if tokio::signal::ctrl_c().await.is_ok() {
            tracing::info!("received Ctrl-C");
            shutdown.notify_one();
        }
    });
}

/// Poll `GET /health` on a backend over a raw TCP connection (no HTTP client
/// dependency) until it returns `200` or the timeout elapses.
async fn wait_healthy(port: u16, timeout: Duration) -> anyhow::Result<()> {
    let deadline = tokio::time::Instant::now() + timeout;
    loop {
        if health_probe(port).await {
            return Ok(());
        }
        if tokio::time::Instant::now() >= deadline {
            bail!("timed out after {timeout:?}");
        }
        tokio::time::sleep(HEALTH_POLL_INTERVAL).await;
    }
}

async fn health_probe(port: u16) -> bool {
    let addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), port);
    let Ok(mut stream) = TcpStream::connect(addr).await else {
        return false;
    };
    let req = b"GET /health HTTP/1.0\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n";
    if stream.write_all(req).await.is_err() {
        return false;
    }
    let mut buf = [0u8; 128];
    match stream.read(&mut buf).await {
        Ok(n) if n > 0 => {
            let head = &buf[..n];
            // The status-code token is unambiguous in a well-formed status line.
            head.windows(5).any(|w| w == b" 200 ")
        }
        _ => false,
    }
}

/// Configure Linux to kill children if the parent dies before cleanup runs.
#[cfg(target_os = "linux")]
fn set_parent_death_signal(cmd: &mut Command) {
    use std::os::unix::process::CommandExt;
    // SAFETY: `pre_exec` runs in the forked child before `exec`. `prctl` is
    // async-signal-safe and touches no shared state, satisfying the pre_exec
    // contract.
    unsafe {
        cmd.pre_exec(|| {
            if libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGKILL) != 0 {
                return Err(std::io::Error::last_os_error());
            }
            // Close the race between `fork` and installing the parent-death signal.
            if libc::getppid() == 1 {
                libc::raise(libc::SIGKILL);
            }
            Ok(())
        });
    }
}

#[cfg(not(target_os = "linux"))]
fn set_parent_death_signal(_cmd: &mut Command) {}

/// Reserve `n` distinct free loopback ports by binding ephemeral sockets, reading
/// the kernel-assigned port, then releasing. There is a tiny window between
/// release and the child re-binding, but the subsequent health-gate turns any
/// lost race into a clear startup error rather than silent misbehavior.
fn pick_free_ports(n: usize) -> anyhow::Result<Vec<u16>> {
    let mut ports = Vec::with_capacity(n);
    // Holding probes open prevents duplicate assignments within this selection.
    let mut held = Vec::with_capacity(n);
    for _ in 0..n {
        let listener = std::net::TcpListener::bind((Ipv4Addr::LOCALHOST, 0))
            .context("binding an ephemeral port")?;
        ports.push(listener.local_addr()?.port());
        held.push(listener);
    }
    drop(held);
    Ok(ports)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_robin_cycles_backends() {
        let backends: Vec<SocketAddr> = [8001u16, 8002, 8003]
            .iter()
            .map(|&p| SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), p))
            .collect();
        let counter = AtomicUsize::new(0);
        let picked: Vec<u16> = (0..7)
            .map(|_| pick_backend(&backends, &counter).port())
            .collect();
        assert_eq!(picked, vec![8001, 8002, 8003, 8001, 8002, 8003, 8001]);
    }

    #[test]
    fn free_ports_are_distinct_and_nonzero() {
        let ports = pick_free_ports(5).unwrap();
        assert_eq!(ports.len(), 5);
        assert!(ports.iter().all(|&p| p != 0));
        let mut sorted = ports.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), 5, "ports must be distinct: {ports:?}");
    }

    #[test]
    fn child_config_json_round_trips() {
        let mut cfg = MockServerConfig {
            fast: true,
            no_tokenizer: true,
            ttft: 12.5,
            processes: 4,
            ..MockServerConfig::default()
        };
        cfg.processes = 1;
        cfg.port = 34567;
        let json = serde_json::to_string(&cfg).unwrap();
        let back: MockServerConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(back.port, 34567);
        assert_eq!(back.processes, 1);
        assert_eq!(back.ttft, 12.5);
        assert!(back.fast);
        assert!(back.no_tokenizer);
    }
}
