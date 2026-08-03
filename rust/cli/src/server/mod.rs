// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Always-on, cross-run, in-process dashboard server for the native orchestrator.
//!
//! The `aiperf` orchestrator (`rust/cli`) drives many runs — single, multi-trial,
//! and sweep cells — and is the one process with cross-run visibility. This module
//! embeds an axum HTTP server there so a browser can browse every run's
//! `native-v2.json` (and, in a follow-on, watch the in-flight run live). It is the
//! native successor to the retired per-run Python `api` service: no ZMQ, no mesh,
//! no Python — the orchestrator serves its own runs directly.
//!
//! Threading: the orchestrator's main thread is a *blocking* child-spawn loop
//! (`profile::run_cells` → `execute::run_once` → `child.wait()`) with **no** tokio
//! runtime. So the server runs on its own `std::thread` owning a multi-thread tokio
//! runtime (modelled on `aiperf::runner_protocol::artifact_shipping`'s server). It
//! shares only `Send` state with the loop — an `Arc<Mutex<Vec<RunEntry>>>` the loop
//! pushes each completed cell into — and never touches the run's `!Send` state.

pub mod assets;
pub mod index;
pub mod live;
pub mod routes;

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{SystemTime, UNIX_EPOCH};

use index::RunEntry;

/// How deep to walk the results root for historical `native-v2.json` reports.
const DEFAULT_SCAN_MAX_DEPTH: usize = 8;

/// The live session index the orchestrator pushes completed runs into. Cloned into
/// the server's state; both sides share the same `Vec` behind the mutex.
pub type SessionRuns = Arc<Mutex<Vec<RunEntry>>>;

/// The run currently in flight, that the dashboard streams live. Set by the run loop
/// while a child runs, cleared when it exits. The `heartbeat_path` is the NDJSON the
/// child's heartbeat lane writes (TTFT/ITL/latency sketches + counters per progress
/// tick); the SSE endpoint tails it. `id` matches the completed run's [`RunEntry::id`].
#[derive(Clone, serde::Serialize)]
pub struct LiveRun {
    pub id: String,
    pub label: String,
    pub artifact_dir: String,
    pub heartbeat_path: String,
}

/// Shared slot the run loop sets while a child is in flight; the server reads it.
pub type LiveSlot = Arc<Mutex<Option<LiveRun>>>;

/// Configuration for [`start`].
pub struct ServerConfig {
    /// Address to bind (`127.0.0.1:0` picks a free port; `0.0.0.0:PORT` for remote).
    pub bind: SocketAddr,
    /// Results root scanned for historical runs (and where the live session's runs
    /// live). `None` serves only the in-memory session.
    pub results_root: Option<PathBuf>,
}

/// State shared with every request handler (cheap to clone: `Arc` + small fields).
#[derive(Clone)]
pub struct AppState {
    session: SessionRuns,
    live: LiveSlot,
    results_root: Option<PathBuf>,
    started_unix: u64,
    scan_max_depth: usize,
}

/// A running dashboard server. Dropping it (or calling [`shutdown`](Self::shutdown))
/// gracefully stops the server and joins its thread.
pub struct ServerHandle {
    shutdown: Option<tokio::sync::oneshot::Sender<()>>,
    thread: Option<JoinHandle<()>>,
    local_addr: SocketAddr,
}

impl ServerHandle {
    /// The address the server actually bound (resolves an OS-assigned `:0` port).
    pub fn local_addr(&self) -> SocketAddr {
        self.local_addr
    }

    /// Signal graceful shutdown and join the server thread.
    pub fn shutdown(mut self) {
        self.stop();
    }

    fn stop(&mut self) {
        if let Some(tx) = self.shutdown.take() {
            let _ = tx.send(());
        }
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

impl Drop for ServerHandle {
    fn drop(&mut self) {
        self.stop();
    }
}

fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Start the dashboard server on its own thread, sharing `session` with the run loop.
/// Returns once the listener is bound (so the caller can log the real address) or
/// errors if the bind fails. The returned [`ServerHandle`] owns the server's lifetime.
pub fn start(
    config: ServerConfig,
    session: SessionRuns,
    live: LiveSlot,
) -> anyhow::Result<ServerHandle> {
    let state = AppState {
        session,
        live,
        results_root: config.results_root.clone(),
        started_unix: now_unix(),
        scan_max_depth: DEFAULT_SCAN_MAX_DEPTH,
    };
    let bind = config.bind;

    // The server thread reports its bound address back over this channel so the
    // caller learns the real port before the runtime blocks on `serve`.
    let (addr_tx, addr_rx) = std::sync::mpsc::channel::<Result<SocketAddr, String>>();
    let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel::<()>();

    let thread = std::thread::Builder::new()
        .name("aiperf-dashboard".to_string())
        .spawn(move || {
            let runtime = match tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .worker_threads(2)
                .build()
            {
                Ok(rt) => rt,
                Err(e) => {
                    let _ = addr_tx.send(Err(format!("build dashboard runtime: {e}")));
                    return;
                }
            };
            runtime.block_on(async move {
                let listener = match tokio::net::TcpListener::bind(bind).await {
                    Ok(l) => l,
                    Err(e) => {
                        let _ = addr_tx.send(Err(format!("bind {bind}: {e}")));
                        return;
                    }
                };
                let local = listener.local_addr().unwrap_or(bind);
                let _ = addr_tx.send(Ok(local));
                let app = routes::router(state);
                if let Err(e) = axum::serve(listener, app)
                    .with_graceful_shutdown(async move {
                        let _ = shutdown_rx.await;
                    })
                    .await
                {
                    tracing::warn!(error = %e, "aiperf dashboard server exited with error");
                }
            });
        })?;

    let local_addr = addr_rx
        .recv()
        .map_err(|_| anyhow::anyhow!("dashboard server thread exited before binding"))?
        .map_err(|e| anyhow::anyhow!("dashboard server: {e}"))?;

    Ok(ServerHandle {
        shutdown: Some(shutdown_tx),
        thread: Some(thread),
        local_addr,
    })
}

impl RunEntry {
    /// Build a session [`RunEntry`] from a completed sweep cell outcome.
    pub fn from_cell_outcome(o: &crate::sweep::aggregate::CellOutcome) -> Self {
        let label = if o.label.is_empty() {
            o.artifact_dir
                .file_name()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_default()
        } else {
            o.label.clone()
        };
        let sweep_id = o
            .values
            .as_ref()
            .and_then(|v| v.get("sweep_id"))
            .and_then(|v| v.as_str())
            .map(str::to_owned);
        RunEntry::build(
            &o.artifact_dir,
            o.report_path.clone(),
            label,
            o.success,
            o.trial,
            sweep_id,
            "session",
        )
    }
}
