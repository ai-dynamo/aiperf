// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Bounded-memory cross-host artifact shipping over HTTP and streaming zstd.
//!
//! - **Send** ([`FileCompressor`]): read the file in [`CHUNK_SIZE`]-byte chunks
//!   through a `zstd::stream::read::Encoder` and yield each compressed chunk. The
//!   client streams those chunks into the request body over a bounded channel
//!   so backpressure caps in-flight bytes.
//! - **Receive** (`decode_channel_to_file`): stream the request body frames
//!   into a `zstd::stream::raw::Decoder` writing to a `.part` file, then atomic
//!   `rename` to the final path (crash-safe: a partial upload never leaves a
//!   truncated final file). Bounded [`CHUNK_SIZE`] chunks throughout.
//!
//! Metrics summaries use velo; this HTTP plane carries per-record artifacts and
//! `inputs.json`.

use std::collections::{HashMap, HashSet};
use std::io::{self, Read, Write};
use std::net::SocketAddr;
use std::path::{Component, Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result, bail, ensure};
use axum::Router;
use axum::body::Body;
use axum::extract::{DefaultBodyLimit, Extension, Path as AxumPath, Request, State};
use axum::http::{HeaderMap, StatusCode};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use bytes::Bytes;
use hyper_util::rt::TokioIo;
use hyper_util::service::TowerToHyperService;
use rand::TryRngCore;
use rustls::pki_types::{CertificateDer, PrivateKeyDer, ServerName};
use tokio::sync::{Semaphore, mpsc, oneshot, watch};
use tokio::task::{JoinHandle, JoinSet};
use tokio_rustls::{TlsAcceptor, TlsConnector};
use tokio_stream::StreamExt;

use crate::cellular::transport::ArtifactChannelServerConfig;

/// The read/compress/write chunk size. Both the compressor's read buffer and the decoder's
/// per-frame write are bounded by this, so peak per-transfer memory is O(chunk),
/// independent of the artifact file size.
pub const CHUNK_SIZE: usize = 65536;

/// The zstd compression level. Level 3 balances compression ratio and speed for
/// line-oriented JSONL/CSV artifacts this ships.
pub const ZSTD_LEVEL: i32 = 3;

/// The `Content-Encoding` value a cell sets when it streams a zstd-compressed
/// artifact body; the controller streaming-decompresses only when it matches.
pub const ZSTD_CONTENT_ENCODING: &str = "zstd";

const DATASET_SERVE_EVENT_LEVEL: tracing::Level = tracing::Level::DEBUG;
const ARTIFACT_TLS_SERVER_NAME: &str = "aiperf-cellular-artifact.invalid";
const TLS_HANDSHAKE_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(5);
const TLS_HANDSHAKE_MAX_CONCURRENT: usize = 32;
const SERVER_SHUTDOWN_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(1);
const SERVER_FORCE_SHUTDOWN_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(1);

async fn bind_artifact_listener(bind: SocketAddr) -> Result<std::net::TcpListener> {
    tokio::net::TcpListener::bind(bind)
        .await
        .with_context(|| format!("binding artifact upload server to {bind}"))?
        .into_std()
        .context("converting artifact upload listener to standard listener")
}

/// One cell-local bearer capability for the per-run artifact channel.
///
/// It deliberately does not implement serialization, display, or string
/// conversion. The controller receives only [`Self::digest_bytes`].
pub(crate) struct ArtifactBearer([u8; 32]);

impl ArtifactBearer {
    pub(crate) fn generate() -> Result<Self> {
        let mut bytes = [0_u8; 32];
        rand::rngs::OsRng
            .try_fill_bytes(&mut bytes)
            .map_err(|_| anyhow::anyhow!("OS RNG could not mint artifact capability"))?;
        Ok(Self(bytes))
    }

    pub(crate) fn digest_bytes(&self) -> [u8; 32] {
        *blake3::hash(&self.hex_bytes()).as_bytes()
    }

    fn hex_bytes(&self) -> [u8; 64] {
        const HEX: &[u8; 16] = b"0123456789abcdef";
        let mut hex = [0_u8; 64];
        for (index, byte) in self.0.iter().copied().enumerate() {
            hex[index * 2] = HEX[(byte >> 4) as usize];
            hex[index * 2 + 1] = HEX[(byte & 0x0F) as usize];
        }
        hex
    }

    #[cfg(test)]
    pub(crate) fn from_test_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }
}

impl std::fmt::Debug for ArtifactBearer {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("ArtifactBearer([REDACTED])")
    }
}

/// A pinned-TLS, bearer-authenticated client for one controller artifact channel.
pub(crate) struct ArtifactChannelClient {
    authority: String,
    authorization: axum::http::HeaderValue,
    tls: Arc<rustls::ClientConfig>,
}

impl ArtifactChannelClient {
    pub(crate) fn new(
        authority: String,
        server_config: ArtifactChannelServerConfig,
        bearer: ArtifactBearer,
    ) -> Result<Self> {
        let mut roots = rustls::RootCertStore::empty();
        roots
            .add(CertificateDer::from(
                server_config.server_certificate_der().to_vec(),
            ))
            .context("installing pinned artifact TLS certificate")?;
        let provider = Arc::new(rustls::crypto::aws_lc_rs::default_provider());
        let tls = rustls::ClientConfig::builder_with_provider(provider)
            .with_safe_default_protocol_versions()
            .map_err(|_| anyhow::anyhow!("aws-lc cannot provide artifact TLS protocols"))?
            .with_root_certificates(roots)
            .with_no_client_auth();
        let mut raw = b"Bearer ".to_vec();
        raw.extend_from_slice(&bearer.hex_bytes());
        let authorization = axum::http::HeaderValue::from_bytes(&raw)
            .context("constructing artifact authorization header")?;
        Ok(Self {
            authority,
            authorization,
            tls: Arc::new(tls),
        })
    }

    fn request<B>(&self, method: &str, path: &str, body: B) -> Result<hyper::Request<B>> {
        hyper::Request::builder()
            .method(method)
            .uri(path)
            .header(hyper::header::HOST, &self.authority)
            .header(hyper::header::AUTHORIZATION, &self.authorization)
            .body(body)
            .context("building authenticated artifact request")
    }

    async fn connect_and_send<B>(
        &self,
        request: hyper::Request<B>,
        label: &str,
    ) -> Result<hyper::Response<hyper::body::Incoming>>
    where
        B: hyper::body::Body + Send + 'static,
        B::Data: Send,
        B::Error: Into<Box<dyn std::error::Error + Send + Sync>>,
    {
        let stream = tokio::net::TcpStream::connect(&self.authority)
            .await
            .with_context(|| format!("connecting to {label} server"))?;
        let server_name = ServerName::try_from(ARTIFACT_TLS_SERVER_NAME.to_owned())
            .map_err(|_| anyhow::anyhow!("invalid fixed artifact TLS server name"))?;
        let stream = TlsConnector::from(self.tls.clone())
            .connect(server_name, stream)
            .await
            .with_context(|| format!("artifact TLS handshake for {label}"))?;
        let io = hyper_util::rt::TokioIo::new(stream);
        let (mut sender, conn) = hyper::client::conn::http1::handshake(io)
            .await
            .with_context(|| format!("{label} HTTP handshake"))?;
        tokio::spawn(async move {
            let _ = conn.await;
        });
        sender
            .send_request(request)
            .await
            .with_context(|| format!("sending {label} request"))
    }
}

#[derive(Clone)]
pub(crate) struct ArtifactChannelRegistrar {
    authorization_state: Arc<parking_lot::RwLock<ArtifactAuthorizationState>>,
    server_config: ArtifactChannelServerConfig,
}

pub(crate) struct ArtifactRegistrationPlan {
    registrar: ArtifactChannelRegistrar,
    cell_id: usize,
    digest: [u8; 32],
    is_active: bool,
}

impl ArtifactRegistrationPlan {
    pub(crate) fn server_config(&self) -> ArtifactChannelServerConfig {
        self.registrar.server_config()
    }

    pub(crate) fn commit(mut self) {
        {
            let mut authorization = self.registrar.authorization_state.write();
            authorization.slots[self.cell_id] = ArtifactAuthorizationSlot::Authorized(self.digest);
            #[cfg(test)]
            {
                authorization.authorization_publication_count += 1;
            }
        }
        self.is_active = false;
    }
}

impl Drop for ArtifactRegistrationPlan {
    fn drop(&mut self) {
        if self.is_active {
            self.registrar.authorization_state.write().slots[self.cell_id] =
                ArtifactAuthorizationSlot::Vacant;
        }
    }
}

#[derive(Clone, Copy)]
enum ArtifactAuthorizationSlot {
    Vacant,
    Preparing([u8; 32]),
    Authorized([u8; 32]),
}

struct ArtifactAuthorizationState {
    slots: Box<[ArtifactAuthorizationSlot]>,
    #[cfg(test)]
    authorization_publication_count: usize,
}

impl ArtifactChannelRegistrar {
    pub(crate) fn prepare(
        &self,
        cell_id: u32,
        digest: [u8; 32],
    ) -> Result<ArtifactRegistrationPlan> {
        let cell_id = cell_id as usize;
        let mut authorization = self.authorization_state.write();
        ensure!(
            cell_id < authorization.slots.len(),
            "artifact registration cell is out of range"
        );
        ensure!(
            authorization
                .slots
                .iter()
                .enumerate()
                .all(|(other_id, slot)| other_id == cell_id
                    || !matches!(
                        slot,
                        ArtifactAuthorizationSlot::Preparing(existing)
                            | ArtifactAuthorizationSlot::Authorized(existing)
                            if *existing == digest
                    )),
            "artifact capability is already bound to another cell"
        );
        ensure!(
            matches!(
                authorization.slots[cell_id],
                ArtifactAuthorizationSlot::Vacant
            ),
            "cell already has an artifact registration"
        );
        authorization.slots[cell_id] = ArtifactAuthorizationSlot::Preparing(digest);
        Ok(ArtifactRegistrationPlan {
            registrar: self.clone(),
            cell_id,
            digest,
            is_active: true,
        })
    }

    pub(crate) fn server_config(&self) -> ArtifactChannelServerConfig {
        self.server_config.clone()
    }

    #[cfg(test)]
    pub(crate) fn authorization_publication_count(&self) -> usize {
        self.authorization_state
            .read()
            .authorization_publication_count
    }
}

#[derive(Clone, Copy)]
struct AuthenticatedCellId(u32);

struct TlsListener {
    completed: mpsc::Receiver<(
        tokio_rustls::server::TlsStream<tokio::net::TcpStream>,
        SocketAddr,
    )>,
    local_addr: SocketAddr,
}

struct ArtifactUploadLifecycle {
    runtime: tokio::runtime::Handle,
    /// A lifecycle moved into the origin-runtime reaper must never schedule a
    /// second reaper if that runtime has already stopped accepting work.
    can_spawn_reaper: bool,
    application_shutdown_tx: Option<oneshot::Sender<()>>,
    application_force_tx: Option<oneshot::Sender<()>>,
    admission_shutdown_tx: Option<oneshot::Sender<()>>,
    application_task: Option<JoinHandle<()>>,
    admission_task: Option<JoinHandle<()>>,
    application_error: Option<String>,
    admission_error: Option<String>,
    blocking_tasks: Option<ArtifactBlockingTasks>,
    #[cfg(test)]
    task_monitor: TestTaskMonitor,
}

#[derive(Clone)]
struct ArtifactBlockingTaskSender {
    requests: mpsc::Sender<ArtifactBlockingTask>,
}

enum ArtifactBlockingTask {
    Upload {
        rx: mpsc::Receiver<UploadChunk>,
        dest: PathBuf,
        zstd: bool,
        completed: oneshot::Sender<std::result::Result<(), String>>,
        #[cfg(test)]
        task_monitor: TestTaskMonitor,
    },
    Dataset {
        source: DatasetSource,
        chunks: mpsc::Sender<std::result::Result<Bytes, io::Error>>,
        #[cfg(test)]
        task_monitor: TestTaskMonitor,
    },
}

struct ArtifactBlockingTasks {
    sender: ArtifactBlockingTaskSender,
    shutdown: watch::Sender<()>,
    task: Option<JoinHandle<()>>,
}

/// A blocking operation whose handle remains owned if its caller is cancelled.
///
/// Tokio cannot abort a running `spawn_blocking` closure. Dropping its handle
/// would detach that closure, so cancellation moves the handle to the runtime
/// that created it and keeps joining there.
struct OwnedBlockingIoTask {
    runtime: tokio::runtime::Handle,
    task: Option<JoinHandle<io::Result<()>>>,
    route_class: &'static str,
    operation: &'static str,
}

impl OwnedBlockingIoTask {
    fn new(
        task: JoinHandle<io::Result<()>>,
        route_class: &'static str,
        operation: &'static str,
    ) -> Self {
        Self {
            runtime: tokio::runtime::Handle::current(),
            task: Some(task),
            route_class,
            operation,
        }
    }

    async fn join(&mut self) -> Result<()> {
        let result = match self.task.as_mut() {
            Some(task) => task.await,
            None => return Ok(()),
        };
        self.task = None;
        match result {
            Ok(Ok(())) => Ok(()),
            Ok(Err(error)) => Err(error).context(self.operation),
            Err(error) => Err(anyhow::anyhow!("{} task panicked: {error}", self.operation)),
        }
    }
}

impl Drop for OwnedBlockingIoTask {
    fn drop(&mut self) {
        let Some(task) = self.task.take() else {
            return;
        };
        let runtime = self.runtime.clone();
        let route_class = self.route_class;
        let operation = self.operation;
        runtime.spawn(async move {
            match task.await {
                Ok(Ok(())) => {}
                Ok(Err(error)) => tracing::debug!(
                    target: "aiperf_cellular_artifact",
                    error = %error,
                    route_class,
                    outcome = "reaper_task_failed",
                    operation,
                    "reaped artifact blocking task failed"
                ),
                Err(error) => tracing::error!(
                    target: "aiperf_cellular_artifact",
                    error = %error,
                    route_class,
                    outcome = "reaper_task_panicked",
                    operation,
                    "reaped artifact blocking task panicked"
                ),
            }
        });
    }
}

impl ArtifactBlockingTaskSender {
    async fn start_upload(
        &self,
        rx: mpsc::Receiver<UploadChunk>,
        dest: PathBuf,
        zstd: bool,
        #[cfg(test)] task_monitor: TestTaskMonitor,
    ) -> Result<oneshot::Receiver<std::result::Result<(), String>>> {
        let (completed, receiver) = oneshot::channel();
        self.requests
            .send(ArtifactBlockingTask::Upload {
                rx,
                dest,
                zstd,
                completed,
                #[cfg(test)]
                task_monitor,
            })
            .await
            .map_err(|_| anyhow::anyhow!("artifact blocking task supervisor stopped"))?;
        Ok(receiver)
    }

    async fn start_dataset(
        &self,
        source: DatasetSource,
        chunks: mpsc::Sender<std::result::Result<Bytes, io::Error>>,
        #[cfg(test)] task_monitor: TestTaskMonitor,
    ) -> Result<()> {
        self.requests
            .send(ArtifactBlockingTask::Dataset {
                source,
                chunks,
                #[cfg(test)]
                task_monitor,
            })
            .await
            .map_err(|_| anyhow::anyhow!("artifact blocking task supervisor stopped"))
    }
}

impl ArtifactBlockingTasks {
    fn start(#[cfg(test)] task_monitor: TestTaskMonitor) -> Self {
        let (requests_tx, mut requests_rx) = mpsc::channel(32);
        let (shutdown, mut shutdown_rx) = watch::channel(());
        #[cfg(test)]
        let supervisor = task_monitor.enter(TestTaskKind::BlockingSupervisor);
        let task = tokio::spawn(async move {
            #[cfg(test)]
            let _supervisor = supervisor;
            let mut tasks = JoinSet::new();
            loop {
                tokio::select! {
                    _ = shutdown_rx.changed() => break,
                    joined = tasks.join_next(), if !tasks.is_empty() => {
                        if let Some(Err(error)) = joined {
                            tracing::error!(target: "aiperf_cellular_artifact", error = %error, route_class = "artifact", outcome = "blocking_task_panicked", "artifact blocking task panicked");
                        }
                    }
                    request = requests_rx.recv() => match request {
                        Some(ArtifactBlockingTask::Upload { rx, dest, zstd, completed, #[cfg(test)] task_monitor }) => {
                            #[cfg(test)]
                            let writer = task_monitor.enter(TestTaskKind::UploadWriter);
                            tasks.spawn(async move {
                                let result = tokio::task::spawn_blocking(move || {
                                    #[cfg(test)] let _writer = writer;
                                    decode_channel_to_file(rx, &dest, zstd)
                                }).await;
                                let result = match result {
                                    Ok(Ok(())) => Ok(()),
                                    Ok(Err(error)) => Err(error.to_string()),
                                    Err(error) => Err(format!("artifact upload writer task panicked: {error}")),
                                };
                                if let Err(error) = &result {
                                    tracing::debug!(target: "aiperf_cellular_artifact", error = %error, route_class = "artifact", outcome = "writer_failed", "artifact upload writer stopped");
                                }
                                let _ = completed.send(result);
                            });
                        }
                        Some(ArtifactBlockingTask::Dataset { source, chunks, #[cfg(test)] task_monitor }) => {
                            #[cfg(test)]
                            let compressor = task_monitor.enter(TestTaskKind::DatasetCompressor);
                            tasks.spawn(async move {
                                let result = tokio::task::spawn_blocking(move || {
                                    #[cfg(test)] let _compressor = compressor;
                                    let DatasetSource::Path(path) = source;
                                    let result = FileCompressor::open(&path).and_then(|mut compressor| {
                                        while let Some(chunk) = compressor.next_chunk()? {
                                            if chunks.blocking_send(Ok(Bytes::from(chunk))).is_err() { break; }
                                        }
                                        Ok(())
                                    });
                                    if let Err(error) = &result { let _ = chunks.blocking_send(Err(io::Error::new(error.kind(), error.to_string()))); }
                                    result
                                }).await;
                                match result {
                                    Ok(Ok(())) => {}
                                    Ok(Err(error)) => tracing::debug!(
                                        target: "aiperf_cellular_artifact",
                                        error = %error,
                                        route_class = "dataset",
                                        outcome = "compressor_failed",
                                        "dataset compressor stopped"
                                    ),
                                    Err(error) => tracing::error!(
                                        target: "aiperf_cellular_artifact",
                                        error = %error,
                                        route_class = "dataset",
                                        outcome = "compressor_panicked",
                                        "dataset compressor task panicked"
                                    ),
                                }
                            });
                        }
                        None => break,
                    },
                }
            }
            while let Some(result) = tasks.join_next().await {
                if let Err(error) = result {
                    tracing::error!(target: "aiperf_cellular_artifact", error = %error, route_class = "artifact", outcome = "blocking_task_panicked", "artifact blocking task panicked");
                }
            }
        });
        Self {
            sender: ArtifactBlockingTaskSender {
                requests: requests_tx,
            },
            shutdown,
            task: Some(task),
        }
    }

    fn signal_shutdown(&self) {
        self.shutdown.send_modify(|_| {});
    }

    async fn join(&mut self) -> Result<()> {
        self.signal_shutdown();
        let result = match self.task.as_mut() {
            Some(task) => Some(task.await),
            None => None,
        };
        self.task = None;
        if let Some(result) = result {
            result.context("joining artifact blocking task supervisor")?;
        }
        Ok(())
    }
}

#[cfg(test)]
#[derive(Clone)]
struct TestTaskMonitor {
    state: Arc<parking_lot::Mutex<TestTaskMonitorState>>,
}

#[cfg(test)]
struct TestTaskMonitorState {
    live_tasks: usize,
    upload_writers: usize,
    live_updates: watch::Sender<usize>,
    upload_writer_updates: watch::Sender<usize>,
}

#[cfg(test)]
#[derive(Clone, Copy, PartialEq, Eq)]
enum TestTaskKind {
    TlsSupervisor,
    TlsHandshake,
    ApplicationSupervisor,
    Connection,
    BlockingSupervisor,
    UploadWriter,
    DatasetCompressor,
    Reaper,
}

#[cfg(test)]
struct TestTaskGuard {
    monitor: TestTaskMonitor,
    kind: TestTaskKind,
}

#[cfg(test)]
impl TestTaskMonitor {
    fn new() -> Self {
        Self {
            state: Arc::new(parking_lot::Mutex::new(TestTaskMonitorState {
                live_tasks: 0,
                upload_writers: 0,
                live_updates: watch::Sender::new(0),
                upload_writer_updates: watch::Sender::new(0),
            })),
        }
    }

    fn enter(&self, kind: TestTaskKind) -> TestTaskGuard {
        let mut state = self.state.lock();
        state.live_tasks += 1;
        state.live_updates.send_replace(state.live_tasks);
        if kind == TestTaskKind::UploadWriter {
            state.upload_writers += 1;
            state
                .upload_writer_updates
                .send_replace(state.upload_writers);
        }
        TestTaskGuard {
            monitor: self.clone(),
            kind,
        }
    }

    async fn wait_for_at_least(&self, expected: usize) {
        let mut updates = self.state.lock().live_updates.subscribe();
        loop {
            if *updates.borrow_and_update() >= expected {
                return;
            }
            updates.changed().await.unwrap();
        }
    }

    async fn wait_for_idle(&self) {
        let mut updates = self.state.lock().live_updates.subscribe();
        loop {
            if *updates.borrow_and_update() == 0 {
                return;
            }
            updates.changed().await.unwrap();
        }
    }

    async fn wait_for_upload_writer(&self) {
        let mut updates = self.state.lock().upload_writer_updates.subscribe();
        loop {
            if *updates.borrow_and_update() > 0 {
                return;
            }
            updates.changed().await.unwrap();
        }
    }
}

#[cfg(test)]
impl Drop for TestTaskGuard {
    fn drop(&mut self) {
        let mut state = self.monitor.state.lock();
        state.live_tasks = state
            .live_tasks
            .checked_sub(1)
            .expect("test task monitor guard must be balanced");
        state.live_updates.send_replace(state.live_tasks);
        if self.kind == TestTaskKind::UploadWriter {
            state.upload_writers = state
                .upload_writers
                .checked_sub(1)
                .expect("test upload writer monitor guard must be balanced");
            state
                .upload_writer_updates
                .send_replace(state.upload_writers);
        }
    }
}

impl TlsListener {
    fn admit(
        listener: tokio::net::TcpListener,
        acceptor: TlsAcceptor,
        local_addr: SocketAddr,
        #[cfg(test)] task_monitor: TestTaskMonitor,
    ) -> (Self, oneshot::Sender<()>, JoinHandle<()>) {
        let (completed_tx, completed) = mpsc::channel(TLS_HANDSHAKE_MAX_CONCURRENT);
        let (shutdown_tx, mut shutdown_rx) = oneshot::channel();
        #[cfg(test)]
        let supervisor = task_monitor.enter(TestTaskKind::TlsSupervisor);
        let task = tokio::spawn(async move {
            #[cfg(test)]
            let _supervisor = supervisor;
            let permits = Arc::new(Semaphore::new(TLS_HANDSHAKE_MAX_CONCURRENT));
            let mut handshakes = JoinSet::new();
            loop {
                tokio::select! {
                    _ = &mut shutdown_rx => break,
                    joined = handshakes.join_next(), if !handshakes.is_empty() => {
                        if let Some(Err(error)) = joined {
                            tracing::debug!(
                                target: "aiperf_cellular_artifact",
                                error = %error,
                                route_class = "tls",
                                outcome = "task_failed",
                                "artifact TLS handshake task failed"
                            );
                        }
                    }
                    accepted = listener.accept() => match accepted {
                        Ok((stream, address)) => {
                            let Ok(permit) = permits.clone().try_acquire_owned() else {
                                tracing::debug!(
                                    target: "aiperf_cellular_artifact",
                                    route_class = "tls",
                                    outcome = "admission_full",
                                    "artifact TLS connection rejected"
                                );
                                continue;
                            };
                            let acceptor = acceptor.clone();
                            let completed_tx = completed_tx.clone();
                            #[cfg(test)]
                            let task_monitor = task_monitor.clone();
                            #[cfg(test)]
                            let handshake = task_monitor.enter(TestTaskKind::TlsHandshake);
                            handshakes.spawn(async move {
                                #[cfg(test)]
                                let _handshake = handshake;
                                let _permit = permit;
                                match tokio::time::timeout(TLS_HANDSHAKE_TIMEOUT, acceptor.accept(stream)).await {
                                    Ok(Ok(stream)) => {
                                        if completed_tx.send((stream, address)).await.is_err() {
                                            tracing::debug!(
                                                target: "aiperf_cellular_artifact",
                                                route_class = "tls",
                                                outcome = "server_stopped",
                                                "artifact TLS handshake completed after server shutdown"
                                            );
                                        }
                                    }
                                    Ok(Err(_)) => tracing::debug!(
                                        target: "aiperf_cellular_artifact",
                                        route_class = "tls",
                                        outcome = "rejected",
                                        "artifact TLS connection rejected"
                                    ),
                                    Err(_) => tracing::debug!(
                                        target: "aiperf_cellular_artifact",
                                        route_class = "tls",
                                        outcome = "timeout",
                                        "artifact TLS handshake timed out"
                                    ),
                                }
                            });
                        }
                        Err(error) => tracing::debug!(
                            target: "aiperf_cellular_artifact",
                            error = %error,
                            route_class = "accept",
                            "artifact listener accept failed"
                        ),
                    },
                }
            }
            handshakes.abort_all();
            while handshakes.join_next().await.is_some() {}
        });
        (
            Self {
                completed,
                local_addr,
            },
            shutdown_tx,
            task,
        )
    }
}

impl axum::serve::Listener for TlsListener {
    type Io = tokio_rustls::server::TlsStream<tokio::net::TcpStream>;
    type Addr = SocketAddr;

    async fn accept(&mut self) -> (Self::Io, Self::Addr) {
        match self.completed.recv().await {
            Some(connection) => connection,
            None => std::future::pending().await,
        }
    }

    fn local_addr(&self) -> io::Result<Self::Addr> {
        Ok(self.local_addr)
    }
}

async fn serve_artifact_uploads(
    mut listener: TlsListener,
    app: Router,
    mut shutdown_rx: oneshot::Receiver<()>,
    mut force_rx: oneshot::Receiver<()>,
    #[cfg(test)] task_monitor: TestTaskMonitor,
    #[cfg(test)] supervisor: TestTaskGuard,
) {
    #[cfg(test)]
    let _supervisor = supervisor;
    let app = app.into_service::<hyper::body::Incoming>();
    let mut connections = JoinSet::new();
    let (connection_shutdown, _) = watch::channel(());
    let mut needs_force = false;
    loop {
        tokio::select! {
            _ = &mut shutdown_rx => {
                break;
            }
            _ = &mut force_rx => {
                needs_force = true;
                break;
            }
            joined = connections.join_next(), if !connections.is_empty() => {
                if let Some(Err(error)) = joined {
                    tracing::debug!(
                        target: "aiperf_cellular_artifact",
                        error = %error,
                        route_class = "http",
                        outcome = "task_failed",
                        "artifact HTTP connection task failed"
                    );
                }
            }
            (stream, address) = axum::serve::Listener::accept(&mut listener) => {
                let app = app.clone();
                let mut connection_shutdown = connection_shutdown.subscribe();
                #[cfg(test)]
                let task_monitor = task_monitor.clone();
                #[cfg(test)]
                let connection = task_monitor.enter(TestTaskKind::Connection);
                connections.spawn(async move {
                    #[cfg(test)]
                    let _connection = connection;
                    let io = TokioIo::new(stream);
                    let connection = hyper::server::conn::http1::Builder::new()
                        .serve_connection(io, TowerToHyperService::new(app))
                        .with_upgrades();
                    tokio::pin!(connection);
                    let result = tokio::select! {
                        result = &mut connection => result,
                        _ = connection_shutdown.changed() => {
                            connection.as_mut().graceful_shutdown();
                            connection.await
                        }
                    };
                    if let Err(error) = result {
                        tracing::debug!(
                            target: "aiperf_cellular_artifact",
                            error = %error,
                            %address,
                            route_class = "http",
                            outcome = "connection_closed",
                            "artifact HTTP connection closed"
                        );
                    }
                });
            }
        }
    }
    connection_shutdown.send_modify(|_| {});
    if needs_force {
        connections.abort_all();
    }
    while !connections.is_empty() {
        if needs_force {
            let _ = connections.join_next().await;
            continue;
        }
        tokio::select! {
            joined = connections.join_next() => {
                if let Some(Err(error)) = joined {
                    tracing::debug!(
                        target: "aiperf_cellular_artifact",
                        error = %error,
                        route_class = "http",
                        outcome = "task_failed",
                        "artifact HTTP connection task failed"
                    );
                }
            }
            _ = &mut force_rx => {
                needs_force = true;
                connections.abort_all();
            }
        }
    }
}

impl ArtifactUploadLifecycle {
    fn signal_shutdown(&mut self) {
        if let Some(tx) = self.application_shutdown_tx.take() {
            let _ = tx.send(());
        }
        if let Some(tx) = self.admission_shutdown_tx.take() {
            let _ = tx.send(());
        }
    }

    fn force_shutdown(&mut self) {
        if let Some(tx) = self.application_force_tx.take() {
            let _ = tx.send(());
        }
    }

    async fn join(&mut self) -> Result<()> {
        if self.application_error.is_none()
            && let Some(task) = self.application_task.as_mut()
        {
            let result = task.await;
            self.application_task = None;
            if let Err(error) = result {
                self.application_error = Some(error.to_string());
            }
        }
        if self.admission_error.is_none()
            && let Some(task) = self.admission_task.as_mut()
        {
            let result = task.await;
            self.admission_task = None;
            if let Err(error) = result {
                self.admission_error = Some(error.to_string());
            }
        }
        if let Some(blocking_tasks) = self.blocking_tasks.as_mut() {
            blocking_tasks.join().await?;
        }
        self.blocking_tasks = None;
        if let Some(error) = self.application_error.take() {
            bail!("joining artifact upload server: {error}");
        }
        if let Some(error) = self.admission_error.take() {
            bail!("joining artifact TLS admission: {error}");
        }
        Ok(())
    }

    async fn shutdown(mut self) -> Result<()> {
        self.signal_shutdown();
        match tokio::time::timeout(SERVER_SHUTDOWN_TIMEOUT, self.join()).await {
            Ok(result) => return result,
            Err(_) => {}
        }
        self.force_shutdown();
        tokio::time::timeout(SERVER_FORCE_SHUTDOWN_TIMEOUT, self.join())
            .await
            .map_err(|_| anyhow::anyhow!("timed out reaping forced artifact upload tasks"))?
    }
}

impl Drop for ArtifactUploadLifecycle {
    fn drop(&mut self) {
        if self.application_task.is_none()
            && self.admission_task.is_none()
            && self.blocking_tasks.is_none()
        {
            return;
        }
        self.signal_shutdown();
        self.force_shutdown();
        if !self.can_spawn_reaper {
            tracing::error!(
                target: "aiperf_cellular_artifact",
                route_class = "artifact",
                outcome = "origin_runtime_closed",
                has_blocking_tasks = self.blocking_tasks.is_some(),
                "artifact lifecycle could not be reaped because its origin runtime stopped"
            );
            return;
        }
        #[cfg(test)]
        let reaper = self.task_monitor.enter(TestTaskKind::Reaper);
        let lifecycle = Self {
            runtime: self.runtime.clone(),
            can_spawn_reaper: false,
            application_shutdown_tx: self.application_shutdown_tx.take(),
            application_force_tx: self.application_force_tx.take(),
            admission_shutdown_tx: self.admission_shutdown_tx.take(),
            application_task: self.application_task.take(),
            admission_task: self.admission_task.take(),
            application_error: self.application_error.take(),
            admission_error: self.admission_error.take(),
            blocking_tasks: self.blocking_tasks.take(),
            #[cfg(test)]
            task_monitor: self.task_monitor.clone(),
        };
        lifecycle.runtime.clone().spawn(async move {
            #[cfg(test)]
            let _reaper = reaper;
            if let Err(error) = lifecycle.shutdown().await {
                tracing::error!(target: "aiperf_cellular_artifact", error = %error, route_class = "artifact", outcome = "reaper_failed", "artifact upload lifecycle reaper failed");
            }
        });
    }
}

/// The wire manifest a cross-host cell fetches (`GET /dataset-manifest`) to learn
/// the exact file set that makes up a directory or segmented-prefix graph trace,
/// before streaming each one over the same HTTP+zstd plane a single file uses.
///
/// The controller derives this from the graph loader's OWN enumeration
/// (`crate::graph::recorded::enumerate_recorded_trace_files`), so the shipped set
/// is byte-for-byte the set a 1-cell run reads. The cell reconstructs the files
/// under a cell-local directory preserving their relative names, then
/// rewrites `datasets/0.path` per [`kind`](Self::kind):
/// - `"file"` / `"prefix"`: `path` = `<dest>/<base_name>` (a single file, or the
///   re-globbable segmented-prefix stem);
/// - `"dir"`: `path` = `<dest>` (the reconstructed directory the loader scans).
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct DatasetManifest {
    /// Layout kind: `"file"`, `"dir"`, or `"prefix"`.
    pub kind: String,
    /// The original trace path's file name — the stem the cell rewrites
    /// `datasets/0.path` around for the `file`/`prefix` kinds.
    pub base_name: String,
    /// The relative file names to fetch, in loader order. Graph shards remain
    /// flat; rooted replay packs may contain validated nested paths, served by
    /// `GET /dataset/{name}` and re-fetched by the cell in this order.
    pub files: Vec<String>,
}

/// A controller-owned source served to a cross-host cell.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DatasetSource {
    /// A regular dataset source that remains on the controller filesystem.
    Path(PathBuf),
}

/// A bounded streaming compressor over one artifact file: each
/// [`next_chunk`](Self::next_chunk) yields at most [`CHUNK_SIZE`] bytes of the
/// zstd frame, reading the source incrementally so the whole file is never
/// resident.
pub struct FileCompressor {
    encoder: zstd::stream::read::Encoder<'static, io::BufReader<std::fs::File>>,
}

impl FileCompressor {
    /// Open `path` for streaming compression at [`ZSTD_LEVEL`].
    pub fn open(path: &Path) -> io::Result<Self> {
        let file = std::fs::File::open(path)?;
        let encoder = zstd::stream::read::Encoder::new(file, ZSTD_LEVEL)?;
        Ok(Self { encoder })
    }

    /// The next compressed chunk (at most [`CHUNK_SIZE`] bytes), or `None` at the
    /// end of the frame. A short read mid-stream is normal — callers loop until
    /// `None`.
    ///
    /// Reads straight into the owned buffer that is returned: every caller
    /// consumes the `Vec` by move (`Bytes::from` on the HTTP ship path,
    /// `blocking_send` on the velo path), so a reusable scratch buffer would only
    /// add a redundant up-to-[`CHUNK_SIZE`] memcpy per chunk over a potentially
    /// multi-GB records artifact.
    pub fn next_chunk(&mut self) -> io::Result<Option<Vec<u8>>> {
        let mut chunk = vec![0_u8; CHUNK_SIZE];
        let n = self.encoder.read(&mut chunk)?;
        if n == 0 {
            return Ok(None);
        }
        chunk.truncate(n);
        Ok(Some(chunk))
    }
}

/// The `.part` staging path for a final artifact path: the final path with a
/// `.part` suffix APPENDED (not a replaced extension, so `x.jsonl` → `x.jsonl.part`
/// keeps its real extension). The receiver decompresses into this, then atomically
/// renames it onto the final path.
fn part_path_for(final_path: &Path) -> PathBuf {
    let mut raw = final_path.as_os_str().to_owned();
    raw.push(".part");
    PathBuf::from(raw)
}

/// A bounded streaming decompressor writing to a `.part` file, atomically renamed
/// onto the final path by [`finish`](Self::finish). A failed transfer leaves a
/// `.part` file, never a truncated final artifact.
pub struct DecompressToFile {
    decoder: zstd::stream::raw::Decoder<'static>,
    writer: io::BufWriter<std::fs::File>,
    output: Vec<u8>,
    finished_frame: bool,
    part_path: PathBuf,
    final_path: PathBuf,
}

/// Create the `.part` staging file for `final_path` (and any missing parent
/// dirs), returning the open file and its `.part` path. Shared by every
/// `.part`-file sink so the crash-safe staging convention lives in one place.
fn create_part_file(final_path: &Path) -> io::Result<(std::fs::File, PathBuf)> {
    if let Some(parent) = final_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let part_path = part_path_for(final_path);
    let file = std::fs::File::create(&part_path)?;
    Ok((file, part_path))
}

/// fsync the finished `.part` file and atomically rename it onto the final path,
/// so a crashed transfer leaves a `.part`, never a truncated final artifact.
fn commit_part_file(file: std::fs::File, part_path: &Path, final_path: &Path) -> io::Result<()> {
    file.sync_all()?;
    std::fs::rename(part_path, final_path)?;
    Ok(())
}

impl DecompressToFile {
    /// Create the `.part` staging file and a raw zstd decoder. The raw decoder
    /// exposes whether it consumed a complete frame, which the writer adapter
    /// deliberately hides when it is merely flushed.
    pub fn create(final_path: &Path) -> io::Result<Self> {
        let (file, part_path) = create_part_file(final_path)?;
        Ok(Self {
            decoder: zstd::stream::raw::Decoder::new()?,
            writer: io::BufWriter::new(file),
            output: vec![0; CHUNK_SIZE],
            finished_frame: false,
            part_path,
            final_path: final_path.to_path_buf(),
        })
    }

    /// Feed one compressed chunk; its decompressed bytes stream straight to the
    /// `.part` file (nothing buffered whole).
    pub fn write_chunk(&mut self, compressed: &[u8]) -> io::Result<()> {
        use zstd::stream::raw::{InBuffer, Operation, OutBuffer};

        let mut input = InBuffer::around(compressed);
        while input.pos() < compressed.len() {
            if self.finished_frame {
                self.decoder.reinit()?;
                self.finished_frame = false;
            }

            let input_before = input.pos();
            let mut output = OutBuffer::around(&mut self.output);
            let hint = self.decoder.run(&mut input, &mut output)?;
            let written = output.pos();
            self.writer.write_all(&self.output[..written])?;
            self.finished_frame = hint == 0;

            if written == 0 && input.pos() == input_before {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "zstd decoder made no progress",
                ));
            }
        }

        Ok(())
    }

    /// Validate the terminal zstd frame, then fsync and atomically commit the
    /// `.part` file. A truncated stream therefore remains only at `.part`.
    pub fn finish(mut self) -> io::Result<()> {
        use zstd::stream::raw::{Operation, OutBuffer};

        loop {
            let mut output = OutBuffer::around(&mut self.output);
            let hint = self.decoder.finish(&mut output, self.finished_frame)?;
            let written = output.pos();
            self.writer.write_all(&self.output[..written])?;
            if hint == 0 {
                break;
            }
        }
        self.writer.flush()?;
        let file = self
            .writer
            .into_inner()
            .map_err(std::io::IntoInnerError::into_error)?;
        commit_part_file(file, &self.part_path, &self.final_path)
    }
}

/// A plain (non-zstd) streaming sink to a `.part` file, for a client that did not
/// set `Content-Encoding: zstd`. Symmetric to [`DecompressToFile`] so the upload
/// handler picks one by the header without buffering either way.
struct PlainToFile {
    writer: io::BufWriter<std::fs::File>,
    part_path: PathBuf,
    final_path: PathBuf,
}

impl PlainToFile {
    fn create(final_path: &Path) -> io::Result<Self> {
        let (file, part_path) = create_part_file(final_path)?;
        Ok(Self {
            writer: io::BufWriter::new(file),
            part_path,
            final_path: final_path.to_path_buf(),
        })
    }

    fn write_chunk(&mut self, bytes: &[u8]) -> io::Result<()> {
        self.writer.write_all(bytes)
    }

    fn finish(mut self) -> io::Result<()> {
        self.writer.flush()?;
        let file = self
            .writer
            .into_inner()
            .map_err(std::io::IntoInnerError::into_error)?;
        commit_part_file(file, &self.part_path, &self.final_path)
    }
}

/// One frame of an in-flight upload: a body chunk, or the sender's explicit
/// end-of-body marker (a dropped channel instead means a cancelled transfer).
enum UploadChunk {
    Data(Bytes),
    Complete,
}

/// Drain `rx` (compressed-or-plain chunks the async handler forwards from the
/// request body) into `dest`, streaming-decompressing when `zstd` is set, then
/// atomic-rename. Runs on a blocking task so the file/zstd work never stalls the
/// async runtime. Bounded memory: one [`CHUNK_SIZE`] chunk at a time.
fn decode_channel_to_file(
    mut rx: mpsc::Receiver<UploadChunk>,
    dest: &Path,
    zstd: bool,
) -> io::Result<()> {
    if zstd {
        let mut sink = DecompressToFile::create(dest)?;
        while let Some(chunk) = rx.blocking_recv() {
            match chunk {
                UploadChunk::Data(chunk) => sink.write_chunk(&chunk)?,
                UploadChunk::Complete => return sink.finish(),
            }
        }
        Err(io::Error::new(
            io::ErrorKind::Interrupted,
            "artifact upload cancelled",
        ))
    } else {
        let mut sink = PlainToFile::create(dest)?;
        while let Some(chunk) = rx.blocking_recv() {
            match chunk {
                UploadChunk::Data(chunk) => sink.write_chunk(&chunk)?,
                UploadChunk::Complete => return sink.finish(),
            }
        }
        Err(io::Error::new(
            io::ErrorKind::Interrupted,
            "artifact upload cancelled",
        ))
    }
}

/// Validate a client-supplied relative artifact path against a cell-scoped
/// allowlist, rejecting anything that is not exactly one of the run's known
/// artifact relative paths. Fails closed on:
/// - an absolute path (`/etc/passwd`),
/// - any non-`Normal` component (`..`, `.`, a Windows prefix, a root),
/// - a path not present in `allowed` verbatim.
///
/// The allowlist is the exact set of relative artifact paths the controller
/// derived from `cfg.artifacts` ([`shippable_relatives`]: records/raw/CSV/parquet/
/// outputs, `inputs.json`, and the graph replay artifacts),
/// so a cell can only ever land bytes at a known per-record artifact location
/// inside its own `cell-{id}` dir — never traverse out of it.
pub(crate) fn validate_artifact_relpath(rel: &str, allowed: &HashSet<String>) -> Result<PathBuf> {
    ensure!(!rel.is_empty(), "empty artifact relative path");
    let path = Path::new(rel);
    ensure!(
        !path.is_absolute(),
        "artifact path {rel:?} must be relative"
    );
    for component in path.components() {
        ensure!(
            matches!(component, Component::Normal(_)),
            "artifact path {rel:?} has a non-normal component (traversal rejected)"
        );
    }
    ensure!(
        allowed.contains(rel),
        "artifact path {rel:?} is not an allowed per-record artifact for this run"
    );
    Ok(path.to_path_buf())
}

/// Shared state for the controller's artifact upload server: where cell files land
/// (`temp_root/cell-{id}/{rel}`), the per-run allowlist of relative artifact paths,
/// and the set of cells that have signaled upload completion.
struct UploadState {
    temp_root: PathBuf,
    allowed: HashSet<String>,
    blocking_tasks: ArtifactBlockingTaskSender,
    #[cfg(test)]
    task_monitor: TestTaskMonitor,
    /// The non-synthetic dataset SOURCE files the controller may serve to cells
    /// over `GET /dataset/{name}`, keyed by the file name a cell
    /// requests. A cross-host cell cannot read the controller-local dataset path,
    /// so the controller streams the source over the same HTTP + zstd plane the
    /// per-record artifact uploads use; the cell then recompiles it locally. Empty
    /// for a synthetic run (each cell regenerates the dataset from the shared seed)
    /// and for a same-host run (cells read the controller-local path directly).
    datasets: HashMap<String, DatasetSource>,
    /// The multi-file dataset manifest served at `GET /dataset-manifest`.
    /// `None` for a synthetic / same-host /
    /// no-dataset run (the route then `404`s); `Some` even for a single file, so a
    /// cell always learns the layout kind and file set from one place.
    manifest: Option<DatasetManifest>,
    /// The set of cells that have signaled `/done`, published through a
    /// [`watch`] channel. `watch` is version-tracked, so a `send` that lands
    /// between the barrier's set-size check and its `changed().await` bumps the
    /// version and wakes the waiter anyway — unlike a bare [`tokio::sync::Notify`]
    /// (whose `notify_waiters()` stores no permit), which could drop that wakeup
    /// and hang the barrier until the upload timeout. See
    /// [`ArtifactUploadServer::wait_for_cells`].
    done: watch::Sender<HashSet<u32>>,
}

/// The controller-side HTTP server cells POST their zstd-compressed artifact files
/// to. Bind, per-cell dir landing, streaming decode, and a cell-completion barrier
/// (`/cell/{id}/done`) the controller awaits before concatenation.
pub struct ArtifactUploadServer {
    local_addr: SocketAddr,
    state: Arc<UploadState>,
    registrar: ArtifactChannelRegistrar,
    lifecycle: Option<ArtifactUploadLifecycle>,
    #[cfg(test)]
    task_monitor: TestTaskMonitor,
}

impl ArtifactUploadServer {
    /// Bind the upload server on `bind` and start serving. `temp_root` is the
    /// controller's cellular scratch root (files land at `temp_root/cell-{id}/…`);
    /// `allowed` is the exact set of relative artifact paths the run may ship.
    /// Bind `0.0.0.0:PORT` (k8s) or `127.0.0.1:0` (in-process test).
    pub async fn start(
        bind: SocketAddr,
        temp_root: PathBuf,
        allowed: HashSet<String>,
        cell_count: u32,
    ) -> Result<Self> {
        Self::start_with_datasets(bind, temp_root, allowed, HashMap::new(), cell_count).await
    }

    /// Like [`start`](Self::start), plus a map of dataset SOURCE files the server
    /// may stream to cells over `GET /dataset/{name}` (cross-host
    /// non-synthetic datasets). `datasets` maps the requested file name to the
    /// controller-local absolute source path; a name absent from the map is served
    /// `404`, so a cell can only ever fetch a source file the run explicitly
    /// registered. An empty map disables dataset serving (the same route still
    /// exists but always returns `404`).
    pub async fn start_with_datasets(
        bind: SocketAddr,
        temp_root: PathBuf,
        allowed: HashSet<String>,
        datasets: HashMap<String, DatasetSource>,
        cell_count: u32,
    ) -> Result<Self> {
        Self::start_with_dataset_plan(bind, temp_root, allowed, datasets, None, cell_count).await
    }

    /// Like [`start_with_datasets`](Self::start_with_datasets), plus the multi-file
    /// [`DatasetManifest`] served at `GET /dataset-manifest`. A directory or
    /// segmented-prefix graph trace registers every shard in `datasets` (keyed by
    /// its flat relative name) AND a `manifest` describing the layout kind and file
    /// set, so a cell can fetch the manifest, then each shard, and reconstruct the
    /// tree. `None` makes the manifest route return `404`.
    pub async fn start_with_dataset_plan(
        bind: SocketAddr,
        temp_root: PathBuf,
        allowed: HashSet<String>,
        datasets: HashMap<String, DatasetSource>,
        manifest: Option<DatasetManifest>,
        cell_count: u32,
    ) -> Result<Self> {
        let listener = bind_artifact_listener(bind).await?;
        Self::start_with_dataset_plan_on_listener(
            listener, temp_root, allowed, datasets, manifest, cell_count,
        )
        .await
    }

    /// Start from a caller-reserved listener, preserving its public address.
    pub async fn start_with_dataset_plan_on_listener(
        listener: std::net::TcpListener,
        temp_root: PathBuf,
        allowed: HashSet<String>,
        datasets: HashMap<String, DatasetSource>,
        manifest: Option<DatasetManifest>,
        cell_count: u32,
    ) -> Result<Self> {
        listener
            .set_nonblocking(true)
            .context("setting artifact upload listener nonblocking")?;
        let listener = tokio::net::TcpListener::from_std(listener)
            .context("adopting artifact upload listener")?;
        #[cfg(test)]
        let task_monitor = TestTaskMonitor::new();
        let rcgen::CertifiedKey { cert, key_pair } =
            rcgen::generate_simple_self_signed(vec![ARTIFACT_TLS_SERVER_NAME.to_owned()])
                .context("generating per-run artifact TLS certificate")?;
        let certificate = CertificateDer::from(cert.der().to_vec());
        let private_key = PrivateKeyDer::try_from(key_pair.serialize_der())
            .map_err(|_| anyhow::anyhow!("constructing per-run artifact TLS private key"))?;
        let provider = Arc::new(rustls::crypto::aws_lc_rs::default_provider());
        let server_config = rustls::ServerConfig::builder_with_provider(provider)
            .with_safe_default_protocol_versions()
            .map_err(|_| anyhow::anyhow!("aws-lc cannot provide artifact TLS protocols"))?
            .with_no_client_auth()
            .with_single_cert(vec![certificate.clone()], private_key)
            .context("installing per-run artifact TLS identity")?;
        let registrar = ArtifactChannelRegistrar {
            authorization_state: Arc::new(parking_lot::RwLock::new(ArtifactAuthorizationState {
                slots: vec![ArtifactAuthorizationSlot::Vacant; cell_count as usize]
                    .into_boxed_slice(),
                #[cfg(test)]
                authorization_publication_count: 0,
            })),
            server_config: ArtifactChannelServerConfig::new(certificate.to_vec()),
        };
        let local_addr = listener
            .local_addr()
            .context("reading artifact upload server address")?;
        // All fallible setup is complete before the supervisor exists, so a
        // failed bind/configuration cannot detach its blocking children.
        let blocking_tasks = ArtifactBlockingTasks::start(
            #[cfg(test)]
            task_monitor.clone(),
        );
        let state = Arc::new(UploadState {
            temp_root,
            allowed,
            blocking_tasks: blocking_tasks.sender.clone(),
            #[cfg(test)]
            task_monitor: task_monitor.clone(),
            datasets,
            manifest,
            done: watch::Sender::new(HashSet::new()),
        });
        let app = Router::new()
            .route("/cell/{cell_id}/artifact/{*file}", post(upload_artifact))
            .route("/cell/{cell_id}/done", post(cell_done))
            .route("/dataset/{*name}", get(serve_dataset))
            .route("/dataset-manifest", get(serve_dataset_manifest))
            // The body is streamed frame by frame (bounded); lift the default 2 MB
            // request-body cap so a large records.jsonl upload is not truncated.
            .layer(DefaultBodyLimit::disable())
            .layer(axum::middleware::from_fn_with_state(
                registrar.authorization_state.clone(),
                authenticate_artifact_request,
            ))
            .with_state(state.clone());
        let runtime = tokio::runtime::Handle::current();
        let (application_shutdown_tx, application_shutdown_rx) = oneshot::channel();
        let (application_force_tx, application_force_rx) = oneshot::channel();
        #[cfg(test)]
        let application_supervisor = task_monitor.enter(TestTaskKind::ApplicationSupervisor);
        let (tls_listener, admission_shutdown_tx, admission_task) = TlsListener::admit(
            listener,
            TlsAcceptor::from(Arc::new(server_config)),
            local_addr,
            #[cfg(test)]
            task_monitor.clone(),
        );
        let application_task = tokio::spawn(serve_artifact_uploads(
            tls_listener,
            app,
            application_shutdown_rx,
            application_force_rx,
            #[cfg(test)]
            task_monitor.clone(),
            #[cfg(test)]
            application_supervisor,
        ));
        Ok(Self {
            local_addr,
            state,
            registrar,
            lifecycle: Some(ArtifactUploadLifecycle {
                runtime,
                can_spawn_reaper: true,
                application_shutdown_tx: Some(application_shutdown_tx),
                application_force_tx: Some(application_force_tx),
                admission_shutdown_tx: Some(admission_shutdown_tx),
                application_task: Some(application_task),
                admission_task: Some(admission_task),
                application_error: None,
                admission_error: None,
                blocking_tasks: Some(blocking_tasks),
                #[cfg(test)]
                task_monitor: task_monitor.clone(),
            }),
            #[cfg(test)]
            task_monitor,
        })
    }

    /// The bound address (host + OS-assigned port when bound to `:0`).
    pub fn local_addr(&self) -> SocketAddr {
        self.local_addr
    }

    /// A cheap clone that binds one registered bearer digest to each expected cell.
    pub(crate) fn registrar(&self) -> ArtifactChannelRegistrar {
        self.registrar.clone()
    }

    #[cfg(test)]
    fn test_task_monitor(&self) -> TestTaskMonitor {
        self.task_monitor.clone()
    }

    /// Wait until `cell_count` distinct cells have signaled `/done`, or `timeout`
    /// elapses. This is the controller's artifact barrier: every cell POSTs all its
    /// files THEN posts `/done`, so once all cells are done the per-cell dirs are
    /// complete and concatenation can run.
    pub async fn wait_for_cells(
        &self,
        cell_count: u32,
        timeout: std::time::Duration,
    ) -> Result<()> {
        // A `watch` receiver is version-tracked: `borrow_and_update` marks the
        // currently-seen version, and `changed()` returns immediately if a cell's
        // `/done` bumped the version since — even one that landed between the
        // set-size check and the `.await`. That closes the lost-wakeup race a bare
        // `Notify::notify_waiters()` (no stored permit) would open, where the final
        // cell's notification fired before the barrier registered a waiter and the
        // run then hung until the timeout despite every upload succeeding.
        let mut rx = self.state.done.subscribe();
        let wait = async {
            loop {
                if rx.borrow_and_update().len() >= cell_count as usize {
                    return;
                }
                if rx.changed().await.is_err() {
                    // All senders dropped (server shutting down): no further cell can
                    // signal, so stop waiting and let the caller's completeness check
                    // (concat over the landed dirs) decide.
                    return;
                }
            }
        };
        tokio::time::timeout(timeout, wait).await.map_err(|_| {
            let done = self.state.done.borrow();
            let missing: Vec<u32> = (0..cell_count).filter(|id| !done.contains(id)).collect();
            anyhow::anyhow!(
                "artifact upload barrier timed out after {timeout:?} with {} of {cell_count} \
                 cells done (missing cells: {missing:?})",
                done.len(),
            )
        })
    }

    /// Stop serving and reap all server tasks within a finite deadline.
    pub async fn shutdown(mut self) -> Result<()> {
        match self.lifecycle.take() {
            Some(lifecycle) => lifecycle.shutdown().await,
            None => Ok(()),
        }
    }
}

impl Drop for ArtifactUploadServer {
    fn drop(&mut self) {
        let Some(mut lifecycle) = self.lifecycle.take() else {
            return;
        };
        lifecycle.signal_shutdown();
        drop(lifecycle);
    }
}

fn unauthorized_artifact_response() -> Response {
    StatusCode::UNAUTHORIZED.into_response()
}

async fn authenticate_artifact_request(
    State(authorization_state): State<Arc<parking_lot::RwLock<ArtifactAuthorizationState>>>,
    mut request: Request,
    next: Next,
) -> Response {
    let Some(value) = request
        .headers()
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.strip_prefix("Bearer "))
        .filter(|token| {
            token.len() == 64
                && token.bytes().all(|byte| {
                    byte.is_ascii_digit() || (byte.is_ascii_lowercase() && byte.is_ascii_hexdigit())
                })
        })
    else {
        tracing::debug!(
            target: "aiperf_cellular_artifact",
            route_class = "artifact",
            outcome = "unauthorized",
            "artifact authorization rejected"
        );
        return unauthorized_artifact_response();
    };
    let digest = *blake3::hash(value.as_bytes()).as_bytes();
    let cell_id = authorization_state
        .read()
        .slots
        .iter()
        .position(|slot| {
            matches!(slot, ArtifactAuthorizationSlot::Authorized(expected) if *expected == digest)
        })
        .and_then(|cell_id| u32::try_from(cell_id).ok());
    let Some(cell_id) = cell_id else {
        tracing::debug!(
            target: "aiperf_cellular_artifact",
            route_class = "artifact",
            outcome = "unauthorized",
            "artifact authorization rejected"
        );
        return unauthorized_artifact_response();
    };
    request
        .extensions_mut()
        .insert(AuthenticatedCellId(cell_id));
    next.run(request).await
}

/// `POST /cell/{cell_id}/artifact/{*file}` — stream the (zstd) request body into
/// `temp_root/cell-{cell_id}/{file}` via `.part` + atomic rename. Path-validated
/// against the run allowlist; bounded memory throughout.
async fn upload_artifact(
    State(state): State<Arc<UploadState>>,
    Extension(AuthenticatedCellId(authenticated_cell_id)): Extension<AuthenticatedCellId>,
    AxumPath((cell_id, file)): AxumPath<(u32, String)>,
    headers: HeaderMap,
    body: Body,
) -> Result<StatusCode, (StatusCode, String)> {
    if authenticated_cell_id != cell_id {
        return Err((StatusCode::UNAUTHORIZED, String::new()));
    }
    let rel = validate_artifact_relpath(&file, &state.allowed)
        .map_err(|error| (StatusCode::BAD_REQUEST, error.to_string()))?;
    let dest = state.temp_root.join(format!("cell-{cell_id}")).join(&rel);
    let zstd = headers
        .get(axum::http::header::CONTENT_ENCODING)
        .and_then(|value| value.to_str().ok())
        .map(|value| value.eq_ignore_ascii_case(ZSTD_CONTENT_ENCODING))
        .unwrap_or(false);

    // Async body frames → bounded channel → blocking decode/rename task. The
    // channel bound (4) is the backpressure that keeps in-flight bytes O(chunk).
    let (tx, rx) = mpsc::channel::<UploadChunk>(4);
    let writer = state
        .blocking_tasks
        .start_upload(
            rx,
            dest,
            zstd,
            #[cfg(test)]
            state.task_monitor.clone(),
        )
        .await
        .map_err(|error| (StatusCode::INTERNAL_SERVER_ERROR, error.to_string()))?;

    // Count the on-wire (post-compression) body bytes so the completion log below is
    // an unambiguous observable that this artifact really crossed the HTTP socket —
    // and, with `content_encoding=zstd`, that it did so compressed, not via a
    // same-host shared filesystem. A multi-process cellular test greps the
    // controller's stderr for one such line per cell × file.
    let mut received_bytes: u64 = 0;
    let mut stream = body.into_data_stream();
    let mut body_error = None;
    while let Some(frame) = stream.next().await {
        let bytes = match frame {
            Ok(bytes) => bytes,
            Err(error) => {
                body_error = Some((
                    StatusCode::BAD_REQUEST,
                    format!("reading upload body: {error}"),
                ));
                break;
            }
        };
        received_bytes += bytes.len() as u64;
        if tx.send(UploadChunk::Data(bytes)).await.is_err() {
            // The writer task failed and dropped rx; surface its error below.
            break;
        }
    }
    if body_error.is_none() {
        let _ = tx.send(UploadChunk::Complete).await;
    }
    drop(tx);
    let writer_result = writer.await.map_err(|_| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("artifact writer supervisor for cell {cell_id} stopped"),
        )
    })?;
    if let Some(error) = body_error {
        let _ = writer_result;
        return Err(error);
    }
    match writer_result {
        Ok(()) => {
            // Emit wire encoding and byte count on a dedicated target so operators
            // can enable this event
            // to `info` (`AIPERF_RUNNER_LOG=warn,aiperf_cellular_artifact=info`)
            // without unmuting the whole runner. See
            // [`crate::engine::cellular_cell::CELL_ARTIFACT_HTTP_FORCE_ENV`].
            tracing::info!(
                target: "aiperf_cellular_artifact",
                cell_id,
                artifact = %file,
                content_encoding = if zstd {
                    ZSTD_CONTENT_ENCODING
                } else {
                    "identity"
                },
                bytes = received_bytes,
                "received artifact upload over HTTP"
            );
            Ok(StatusCode::OK)
        }
        Err(error) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("writing cell {cell_id} artifact {file:?}: {error}"),
        )),
    }
}

/// `POST /cell/{cell_id}/done` — record that this cell has uploaded all its files
/// and wake the controller's [`wait_for_cells`](ArtifactUploadServer::wait_for_cells)
/// barrier.
async fn cell_done(
    State(state): State<Arc<UploadState>>,
    Extension(AuthenticatedCellId(authenticated_cell_id)): Extension<AuthenticatedCellId>,
    AxumPath(cell_id): AxumPath<u32>,
) -> StatusCode {
    if authenticated_cell_id != cell_id {
        return StatusCode::UNAUTHORIZED;
    }
    // `send_modify` mutates the set and bumps the watch version even when no
    // receiver is currently parked, so a barrier that has not yet registered still
    // observes this cell on its next `borrow_and_update` — the wakeup cannot be lost.
    state.done.send_modify(|done| {
        done.insert(cell_id);
    });
    StatusCode::OK
}

/// `GET /dataset/{name}` — stream the registered dataset source file `name`
/// to the cell with `Content-Encoding: zstd`, bounded memory
/// ([`FileCompressor`], one [`CHUNK_SIZE`] chunk at a time). A name absent from
/// the run's dataset allowlist is `404` (a cell can only fetch a source the
/// controller registered). The whole file is never resident on either end.
async fn serve_dataset(
    State(state): State<Arc<UploadState>>,
    AxumPath(name): AxumPath<String>,
) -> Result<Response, (StatusCode, String)> {
    let source = state
        .datasets
        .get(&name)
        .cloned()
        .ok_or_else(|| (StatusCode::NOT_FOUND, format!("no dataset source {name:?}")))?;

    // Blocking chunked read + zstd on a blocking task → bounded channel; the axum
    // response body streams from that channel (whole-file never resident). A read
    // error mid-stream is forwarded as a stream error, truncating the body so the
    // cell's decoder fails rather than landing a partial file.
    let (tx, rx) = mpsc::channel::<Result<Bytes, io::Error>>(4);
    state
        .blocking_tasks
        .start_dataset(
            source,
            tx,
            #[cfg(test)]
            state.task_monitor.clone(),
        )
        .await
        .map_err(|error| (StatusCode::INTERNAL_SERVER_ERROR, error.to_string()))?;

    // Use the shared artifact target so this event can be enabled independently.
    tracing::event!(
        target: "aiperf_cellular_artifact",
        DATASET_SERVE_EVENT_LEVEL,
        dataset = %name,
        content_encoding = ZSTD_CONTENT_ENCODING,
        "served dataset source over TLS/authenticated transfer"
    );

    let body = Body::from_stream(tokio_stream::wrappers::ReceiverStream::new(rx));
    Response::builder()
        .header(axum::http::header::CONTENT_ENCODING, ZSTD_CONTENT_ENCODING)
        .body(body)
        .map_err(|error| (StatusCode::INTERNAL_SERVER_ERROR, error.to_string()))
}

/// `GET /dataset-manifest` returns the registered multi-file [`DatasetManifest`]
/// as JSON, so a cell learns the layout kind and file set before fetching each
/// shard via `GET /dataset/{name}`. `404` when no manifest is registered (a
/// synthetic / same-host / no-dataset run).
async fn serve_dataset_manifest(
    State(state): State<Arc<UploadState>>,
) -> Result<axum::Json<DatasetManifest>, (StatusCode, String)> {
    state.manifest.clone().map(axum::Json).ok_or_else(|| {
        (
            StatusCode::NOT_FOUND,
            "no dataset manifest registered".to_owned(),
        )
    })
}

/// Ship one cell's per-record artifact files (+ `inputs.json`) to the controller
/// over HTTP with streaming zstd, then POST the per-cell `/done` marker.
///
/// `client` carries the controller's artifact `host:port` (DNS-resolved, so a k8s
/// service name works) and this cell's bearer; `cell_dir` is this cell's own artifact dir; `relatives`
/// is the set of relative artifact paths to ship (only those that exist on disk
/// are sent — a metrics-only or lazy-CSV run legitimately omits some). Bounded
/// memory: each file streams through a [`FileCompressor`] over a bounded channel.
pub(crate) async fn ship_cell_artifacts(
    client: &ArtifactChannelClient,
    cell_id: u32,
    cell_dir: &Path,
    relatives: &[String],
) -> Result<()> {
    for rel in relatives {
        let src = cell_dir.join(rel);
        if !src.exists() {
            continue;
        }
        upload_one(client, cell_id, rel, &src)
            .await
            .with_context(|| format!("cell {cell_id} shipping artifact {rel:?}"))?;
    }
    post_done(client, cell_id)
        .await
        .with_context(|| format!("cell {cell_id} posting artifact done"))?;
    Ok(())
}

/// Stream one file to `POST /cell/{cell_id}/artifact/{rel}` with
/// `Content-Encoding: zstd`. Compression runs on a blocking task feeding a bounded
/// channel; the request body streams from that channel (whole-file never resident).
async fn upload_one(
    client: &ArtifactChannelClient,
    cell_id: u32,
    rel: &str,
    src: &Path,
) -> Result<()> {
    use http_body_util::StreamBody;
    use hyper::body::Frame;
    use tokio_stream::wrappers::ReceiverStream;

    // Producer: chunked read + zstd on a blocking task → bounded channel.
    let (tx, rx) = mpsc::channel::<std::result::Result<Bytes, io::Error>>(4);
    let path = src.to_path_buf();
    let mut producer = OwnedBlockingIoTask::new(
        tokio::task::spawn_blocking(move || -> io::Result<()> {
            let result: io::Result<()> = (|| {
                let mut compressor = FileCompressor::open(&path)?;
                while let Some(chunk) = compressor.next_chunk()? {
                    if tx.blocking_send(Ok(Bytes::from(chunk))).is_err() {
                        break; // receiver dropped (request failed); stop early
                    }
                }
                Ok(())
            })();
            if let Err(error) = &result {
                let _ = tx.blocking_send(Err(io::Error::new(error.kind(), error.to_string())));
            }
            result
        }),
        "artifact",
        "streaming artifact compressor",
    );

    let body_stream = ReceiverStream::new(rx).map(|chunk| chunk.map(Frame::data));
    let body = StreamBody::new(body_stream);

    let mut request = client.request("POST", &format!("/cell/{cell_id}/artifact/{rel}"), body)?;
    request.headers_mut().insert(
        hyper::header::CONTENT_ENCODING,
        hyper::header::HeaderValue::from_static(ZSTD_CONTENT_ENCODING),
    );

    // Always join the producer, including when the request itself failed. That
    // keeps a compressor error from becoming a clean EOF and keeps cancellation
    // ownership with `OwnedBlockingIoTask`.
    let response = send_request(client, request).await;
    let producer_result = producer.join().await;
    let status = response?;
    producer_result?;
    ensure!(
        status.is_success(),
        "artifact upload returned HTTP {status}"
    );
    Ok(())
}

/// POST the empty-bodied `/cell/{cell_id}/done` completion marker.
async fn post_done(client: &ArtifactChannelClient, cell_id: u32) -> Result<()> {
    use http_body_util::Empty;

    let request = client.request(
        "POST",
        &format!("/cell/{cell_id}/done"),
        Empty::<Bytes>::new(),
    )?;
    let status = send_request(client, request).await?;
    ensure!(status.is_success(), "done marker returned HTTP {status}");
    Ok(())
}

/// Open a pinned-TLS HTTP/1.1 connection to the client's authority (DNS-resolved),
/// send `request`, and return its status after draining the response. Uses a raw
/// hyper client connection.
async fn send_request<B>(
    client: &ArtifactChannelClient,
    request: hyper::Request<B>,
) -> Result<StatusCode>
where
    B: hyper::body::Body + Send + 'static,
    B::Data: Send,
    B::Error: Into<Box<dyn std::error::Error + Send + Sync>>,
{
    use http_body_util::BodyExt;

    let response = client.connect_and_send(request, "artifact").await?;
    let status = response.status();
    // Drain the (small) response body so the connection closes cleanly.
    let _ = response.into_body().collect().await;
    Ok(status)
}

/// Fetch a controller dataset source over `GET /dataset/{name}` and
/// stream it to `dest`, decompressing when the response is `Content-Encoding:
/// zstd`. Bounded memory: response frames flow through a bounded channel into a
/// blocking streaming decode (`.part` + atomic rename), so a crashed transfer
/// never leaves a truncated final file — the same discipline as the upload leg.
///
/// `client` carries the controller's artifact `host:port`; a same-host cell never
/// calls this (it reads the controller-local path directly).
pub(crate) async fn fetch_dataset_to_file(
    client: &ArtifactChannelClient,
    name: &str,
    dest: &Path,
) -> Result<()> {
    use http_body_util::{BodyExt, Empty};

    let request = client.request("GET", &dataset_request_path(name), Empty::<Bytes>::new())?;

    let response = client.connect_and_send(request, "dataset").await?;
    ensure!(
        response.status().is_success(),
        "dataset fetch for {name:?} returned HTTP {}",
        response.status()
    );
    let zstd = response
        .headers()
        .get(hyper::header::CONTENT_ENCODING)
        .and_then(|value| value.to_str().ok())
        .map(|value| value.eq_ignore_ascii_case(ZSTD_CONTENT_ENCODING))
        .unwrap_or(false);

    // Response frames → bounded channel → blocking decode/rename task.
    let (tx, rx) = mpsc::channel::<UploadChunk>(4);
    let dest_buf = dest.to_path_buf();
    let mut writer = OwnedBlockingIoTask::new(
        tokio::task::spawn_blocking(move || decode_channel_to_file(rx, &dest_buf, zstd)),
        "dataset",
        "streaming dataset writer",
    );

    let mut body = response.into_body();
    let mut body_error = None;
    while let Some(frame) = body.frame().await {
        let frame = match frame {
            Ok(frame) => frame,
            Err(error) => {
                body_error = Some(anyhow::anyhow!("reading dataset body: {error}"));
                break;
            }
        };
        if let Ok(data) = frame.into_data()
            && tx.send(UploadChunk::Data(data)).await.is_err()
        {
            break; // writer task failed and dropped rx; surface its error below
        }
    }
    if body_error.is_none() {
        let _ = tx.send(UploadChunk::Complete).await;
    }
    drop(tx);
    let writer_result = writer.join().await;
    if let Some(error) = body_error {
        let _ = writer_result;
        return Err(error);
    }
    writer_result.with_context(|| format!("writing dataset {name:?}"))
}

fn dataset_request_path(name: &str) -> String {
    let encoded = name
        .split('/')
        .map(|component| {
            percent_encoding::utf8_percent_encode(component, percent_encoding::NON_ALPHANUMERIC)
                .to_string()
        })
        .collect::<Vec<_>>()
        .join("/");
    format!("/dataset/{encoded}")
}

/// Fetch the multi-file dataset [`DatasetManifest`] over `GET /dataset-manifest`.
/// The response is a small JSON body containing file names, not file bytes,
/// so it is collected whole; each named file is then streamed by
/// [`reconstruct_shipped_dataset`] with the bounded-memory [`fetch_dataset_to_file`].
pub(crate) async fn fetch_dataset_manifest(
    client: &ArtifactChannelClient,
) -> Result<DatasetManifest> {
    use http_body_util::{BodyExt, Empty};

    let request = client.request("GET", "/dataset-manifest", Empty::<Bytes>::new())?;
    let response = client.connect_and_send(request, "dataset manifest").await?;
    ensure!(
        response.status().is_success(),
        "dataset manifest returned HTTP {}",
        response.status()
    );
    let bytes = response
        .into_body()
        .collect()
        .await
        .context("reading dataset manifest body")?
        .to_bytes();
    serde_json::from_slice(&bytes).context("decoding dataset manifest")
}

/// Validate a manifest-supplied relative path without allowing traversal.
fn validate_dataset_relname(name: &str) -> Result<()> {
    ensure!(!name.is_empty(), "empty dataset file name");
    let path = Path::new(name);
    let normalized = path.components().collect::<PathBuf>();
    ensure!(
        !path.is_absolute()
            && name.split('/').all(|component| !component.is_empty())
            && path
                .components()
                .all(|part| matches!(part, Component::Normal(_)))
            && normalized == path,
        "dataset file name {name:?} is not a safe relative path (traversal rejected)"
    );
    Ok(())
}

fn prepare_dataset_destination(dest_dir: &Path, relative: &Path) -> Result<PathBuf> {
    match std::fs::symlink_metadata(dest_dir) {
        Ok(metadata) => ensure!(
            metadata.is_dir() && !metadata.file_type().is_symlink(),
            "cell dataset root {} is not a real directory",
            dest_dir.display()
        ),
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            std::fs::create_dir_all(dest_dir)
                .with_context(|| format!("creating cell dataset dir {}", dest_dir.display()))?;
        }
        Err(error) => {
            return Err(error)
                .with_context(|| format!("inspecting cell dataset dir {}", dest_dir.display()));
        }
    }
    let destination = dest_dir.join(relative);
    let mut current = dest_dir.to_path_buf();
    if let Some(parent) = relative.parent() {
        for component in parent.components() {
            let Component::Normal(component) = component else {
                bail!("dataset destination contains a non-normal component")
            };
            current.push(component);
            match std::fs::symlink_metadata(&current) {
                Ok(metadata) => ensure!(
                    metadata.is_dir() && !metadata.file_type().is_symlink(),
                    "cell dataset parent {} is not a real directory",
                    current.display()
                ),
                Err(error) if error.kind() == io::ErrorKind::NotFound => {
                    std::fs::create_dir(&current).with_context(|| {
                        format!("creating cell dataset parent {}", current.display())
                    })?;
                }
                Err(error) => {
                    return Err(error).with_context(|| {
                        format!("inspecting cell dataset parent {}", current.display())
                    });
                }
            }
        }
    }
    if std::fs::symlink_metadata(&destination)
        .is_ok_and(|metadata| metadata.file_type().is_symlink())
    {
        bail!(
            "cell dataset destination {} is a symlink",
            destination.display()
        );
    }
    Ok(destination)
}

// Validate every manifest path before making any dataset request.
fn validate_dataset_manifest(manifest: &DatasetManifest) -> Result<()> {
    let mut seen = HashSet::with_capacity(manifest.files.len());
    for name in &manifest.files {
        validate_dataset_relname(name)
            .with_context(|| format!("validating shipped dataset file {name:?}"))?;
        ensure!(
            seen.insert(name),
            "duplicate dataset manifest path {name:?}"
        );
    }
    match manifest.kind.as_str() {
        "dir" => Ok(()),
        "file" | "prefix" | "replay_root" | "agent_session_set" => {
            if matches!(manifest.kind.as_str(), "replay_root" | "agent_session_set")
                && manifest.base_name.is_empty()
            {
                Ok(())
            } else {
                validate_dataset_relname(&manifest.base_name).with_context(|| {
                    format!("validating dataset base name {:?}", manifest.base_name)
                })
            }
        }
        other => bail!("unknown dataset manifest kind {other:?}"),
    }
}

/// Reconstruct a shipped directory / segmented-prefix / single-file graph trace
/// under `dest_dir` and return the local path `datasets/0.path` should
/// point at.
///
/// Fetches every file named in `manifest` (in order) via the bounded-memory
/// [`fetch_dataset_to_file`], landing each at `dest_dir/<name>` preserving its
/// relative name (each name is path-validated). Each file streams independently, so
/// peak memory is O(chunk) regardless of the number or size of shards. Returns:
/// - `"dir"` → `dest_dir` (the loader scans the reconstructed directory);
/// - `"file"` / `"prefix"` / `"replay_root"` / `"agent_session_set"` →
///   `dest_dir/base_name` (a single file, segmented-prefix stem, recording beneath
///   its replay root, or an imported session selected from its discovery root).
pub(crate) async fn reconstruct_shipped_dataset(
    client: &ArtifactChannelClient,
    manifest: &DatasetManifest,
    dest_dir: &Path,
) -> Result<PathBuf> {
    validate_dataset_manifest(manifest)?;
    for name in &manifest.files {
        let dest = prepare_dataset_destination(dest_dir, Path::new(name))?;
        fetch_dataset_to_file(client, name, &dest)
            .await
            .with_context(|| format!("fetching shipped dataset file {name:?}"))?;
    }
    match manifest.kind.as_str() {
        "dir" => Ok(dest_dir.to_path_buf()),
        "file" | "prefix" | "replay_root" | "agent_session_set" => {
            if matches!(manifest.kind.as_str(), "replay_root" | "agent_session_set")
                && manifest.base_name.is_empty()
            {
                Ok(dest_dir.to_path_buf())
            } else {
                Ok(dest_dir.join(&manifest.base_name))
            }
        }
        other => bail!("unknown dataset manifest kind {other:?}"),
    }
}

/// The relative artifact paths a run may ship over HTTP, derived from its
/// `ArtifactSpec` — every per-record file (records/raw/CSV/parquet/outputs), the
/// per-session `inputs.json`, and the per-cell graph replay artifacts (tool time,
/// trace summary, replay metrics JSON/CSV, failures, provenance, backend
/// metadata). Both the controller's server allowlist and the
/// cell's client shipping list come from this single function, so they can never
/// disagree on which files cross the wire.
pub fn shippable_relatives(artifacts: &crate::engine::protocol::ArtifactSpec) -> Vec<String> {
    let mut relatives = Vec::new();
    for path in [
        artifacts.records_path.as_ref(),
        artifacts.raw_path.as_ref(),
        artifacts.records_csv_path.as_ref(),
        artifacts.records_parquet_path.as_ref(),
        artifacts.outputs_path.as_ref(),
        artifacts.inputs_path.as_ref(),
        artifacts.graph_tool_time_path.as_ref(),
        artifacts.graph_trace_summary_path.as_ref(),
        artifacts.graph_replay_metrics_path.as_ref(),
        artifacts.graph_replay_metrics_csv_path.as_ref(),
        artifacts.graph_replay_failures_path.as_ref(),
        artifacts.graph_replay_provenance_path.as_ref(),
        artifacts.graph_replay_backend_metadata_path.as_ref(),
    ]
    .into_iter()
    .flatten()
    {
        relatives.push(path.to_string_lossy().into_owned());
    }
    relatives
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cellular::transport::CellRegister;

    #[tokio::test]
    async fn address_bind_keeps_the_existing_large_pending_queue() {
        const PENDING_CONNECTIONS: usize = 192;
        let listener = bind_artifact_listener("127.0.0.1:0".parse().unwrap())
            .await
            .unwrap();
        let address = listener.local_addr().unwrap();
        let mut connections = JoinSet::new();
        for _ in 0..PENDING_CONNECTIONS {
            connections.spawn(async move {
                tokio::time::timeout(
                    std::time::Duration::from_secs(1),
                    tokio::net::TcpStream::connect(address),
                )
                .await
            });
        }
        let mut connected = 0;
        while let Some(result) = connections.join_next().await {
            if matches!(result, Ok(Ok(Ok(_)))) {
                connected += 1;
            }
        }
        assert_eq!(
            connected, PENDING_CONNECTIONS,
            "address binding reduced the existing pending connection queue"
        );
    }

    struct SecureServerFixture {
        _temporary: tempfile::TempDir,
        server: ArtifactUploadServer,
        clients: Vec<ArtifactChannelClient>,
    }

    impl SecureServerFixture {
        fn client(&self, cell_id: usize) -> &ArtifactChannelClient {
            &self.clients[cell_id]
        }
    }

    struct UnregisteredSecureServerFixture {
        _temporary: tempfile::TempDir,
        server: ArtifactUploadServer,
        registrar: ArtifactChannelRegistrar,
    }

    async fn unregistered_secure_server_fixture(
        cell_count: u32,
    ) -> UnregisteredSecureServerFixture {
        let temporary = tempfile::tempdir().unwrap();
        let server = ArtifactUploadServer::start(
            "127.0.0.1:0".parse().unwrap(),
            temporary.path().join("landed"),
            HashSet::new(),
            cell_count,
        )
        .await
        .unwrap();
        let registrar = server.registrar();
        UnregisteredSecureServerFixture {
            _temporary: temporary,
            server,
            registrar,
        }
    }

    async fn secure_server_fixture(cell_count: u32) -> SecureServerFixture {
        let fixture = unregistered_secure_server_fixture(cell_count).await;
        let authority = fixture.server.local_addr().to_string();
        let mut clients = Vec::with_capacity(cell_count as usize);
        for cell_id in 0..cell_count {
            let bearer = ArtifactBearer::from_test_bytes([cell_id as u8 + 1; 32]);
            let plan = fixture
                .registrar
                .prepare(cell_id, bearer.digest_bytes())
                .unwrap();
            plan.commit();
            clients.push(
                ArtifactChannelClient::new(
                    authority.clone(),
                    fixture.registrar.server_config(),
                    bearer,
                )
                .unwrap(),
            );
        }
        SecureServerFixture {
            _temporary: fixture._temporary,
            server: fixture.server,
            clients,
        }
    }

    fn test_client(server: &ArtifactUploadServer, cell_id: u32) -> ArtifactChannelClient {
        let bearer = ArtifactBearer::from_test_bytes([cell_id as u8 + 1; 32]);
        let registrar = server.registrar();
        registrar
            .prepare(cell_id, bearer.digest_bytes())
            .unwrap()
            .commit();
        ArtifactChannelClient::new(
            server.local_addr().to_string(),
            registrar.server_config(),
            bearer,
        )
        .unwrap()
    }

    fn unreachable_test_client() -> ArtifactChannelClient {
        let provider = Arc::new(rustls::crypto::aws_lc_rs::default_provider());
        let tls = rustls::ClientConfig::builder_with_provider(provider)
            .with_safe_default_protocol_versions()
            .unwrap()
            .with_root_certificates(rustls::RootCertStore::empty())
            .with_no_client_auth();
        ArtifactChannelClient {
            authority: "127.0.0.1:1".to_owned(),
            authorization: axum::http::HeaderValue::from_static(
                "Bearer 0000000000000000000000000000000000000000000000000000000000000000",
            ),
            tls: Arc::new(tls),
        }
    }

    async fn fetch_status(client: &ArtifactChannelClient, path: &str) -> StatusCode {
        use http_body_util::{BodyExt, Empty};

        let request = client.request("GET", path, Empty::<Bytes>::new()).unwrap();
        let response = client.connect_and_send(request, "test").await.unwrap();
        let status = response.status();
        let _ = response.into_body().collect().await;
        status
    }

    async fn fetch_status_without_authorization(
        client: &ArtifactChannelClient,
        path: &str,
    ) -> StatusCode {
        use http_body_util::{BodyExt, Empty};

        let request = hyper::Request::builder()
            .method("GET")
            .uri(path)
            .header(hyper::header::HOST, &client.authority)
            .body(Empty::<Bytes>::new())
            .unwrap();
        let response = client.connect_and_send(request, "test").await.unwrap();
        let status = response.status();
        let _ = response.into_body().collect().await;
        status
    }

    async fn post_done_status(client: &ArtifactChannelClient, cell_id: u32) -> Result<StatusCode> {
        use http_body_util::{BodyExt, Empty};

        let request = client.request(
            "POST",
            &format!("/cell/{cell_id}/done"),
            Empty::<Bytes>::new(),
        )?;
        let response = client.connect_and_send(request, "test").await?;
        let status = response.status();
        let _ = response.into_body().collect().await;
        Ok(status)
    }

    async fn assert_plain_http_cannot_read_manifest(address: SocketAddr) {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        let mut stream = tokio::net::TcpStream::connect(address).await.unwrap();
        stream
            .write_all(b"GET /dataset-manifest HTTP/1.1\r\nHost: test\r\n\r\n")
            .await
            .unwrap();
        let mut bytes = Vec::new();
        let _ = tokio::time::timeout(
            std::time::Duration::from_millis(250),
            stream.read_to_end(&mut bytes),
        )
        .await;
        assert!(!bytes.windows(12).any(|window| window == b"HTTP/1.1 200"));
        assert!(!bytes.windows(8).any(|window| window == b"manifest"));
    }

    #[test]
    fn dataset_serve_event_is_debug_level() {
        assert_eq!(DATASET_SERVE_EVENT_LEVEL, tracing::Level::DEBUG);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn artifact_channel_rejects_plaintext_and_missing_credentials() {
        let fixture = secure_server_fixture(2).await;
        assert_plain_http_cannot_read_manifest(fixture.server.local_addr()).await;
        assert_eq!(
            fetch_status_without_authorization(fixture.client(0), "/dataset/does-not-exist").await,
            StatusCode::UNAUTHORIZED,
        );
        assert_eq!(
            fetch_status(&fixture.client(0), "/dataset/does-not-exist").await,
            StatusCode::NOT_FOUND,
        );
        fixture.server.shutdown().await.unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn stalled_tls_handshake_does_not_serialize_authenticated_transfers() {
        let fixture = secure_server_fixture(1).await;
        let stalled = tokio::net::TcpStream::connect(fixture.server.local_addr())
            .await
            .expect("connect stalled TCP client");
        tokio::time::sleep(std::time::Duration::from_millis(25)).await;

        let status = tokio::time::timeout(
            std::time::Duration::from_secs(1),
            fetch_status(fixture.client(0), "/dataset/does-not-exist"),
        )
        .await
        .expect("stalled TLS handshake must not block an authenticated transfer");
        assert_eq!(status, StatusCode::NOT_FOUND);

        tokio::time::sleep(TLS_HANDSHAKE_TIMEOUT + std::time::Duration::from_millis(25)).await;
        drop(stalled);
        assert_eq!(
            fetch_status(fixture.client(0), "/dataset/does-not-exist").await,
            StatusCode::NOT_FOUND
        );
        fixture.server.shutdown().await.unwrap();
    }

    mod artifact_upload_server {
        use super::*;
        use http_body_util::{BodyExt, Empty};

        #[tokio::test]
        async fn task_monitor_retains_updates_without_subscribers() {
            let monitor = TestTaskMonitor::new();
            let guard = monitor.enter(TestTaskKind::ApplicationSupervisor);
            tokio::time::timeout(
                std::time::Duration::from_millis(50),
                monitor.wait_for_at_least(1),
            )
            .await
            .expect("monitor must retain the live-task count before subscription");
            drop(guard);
            monitor.wait_for_idle().await;
        }

        #[tokio::test]
        async fn blocking_supervisor_join_clears_a_failed_handle() {
            let monitor = TestTaskMonitor::new();
            let mut tasks = ArtifactBlockingTasks::start(monitor);
            tasks
                .task
                .as_ref()
                .expect("blocking supervisor must have a join handle")
                .abort();

            assert!(tasks.join().await.is_err());
            assert!(
                tasks.join().await.is_ok(),
                "a completed failed handle must not be polled a second time"
            );
        }

        #[tokio::test]
        async fn lifecycle_retains_early_supervisor_failure_across_cancelled_join() {
            let application_task = tokio::spawn(std::future::pending::<()>());
            application_task.abort();
            let admission_task = tokio::spawn(std::future::pending::<()>());
            let mut lifecycle = ArtifactUploadLifecycle {
                runtime: tokio::runtime::Handle::current(),
                can_spawn_reaper: true,
                application_shutdown_tx: None,
                application_force_tx: None,
                admission_shutdown_tx: None,
                application_task: Some(application_task),
                admission_task: Some(admission_task),
                application_error: None,
                admission_error: None,
                blocking_tasks: None,
                task_monitor: TestTaskMonitor::new(),
            };

            assert!(
                tokio::time::timeout(std::time::Duration::from_millis(25), lifecycle.join())
                    .await
                    .is_err(),
                "the later stalled supervisor must cancel the join after the application error"
            );
            assert!(lifecycle.application_error.is_some());

            lifecycle
                .admission_task
                .as_ref()
                .expect("stalled admission handle must remain owned")
                .abort();
            let error = lifecycle
                .join()
                .await
                .expect_err("the earlier application failure must not be lost");
            assert!(error.to_string().contains("artifact upload server"));
        }

        #[tokio::test]
        async fn monitor_reserves_server_work_before_spawned_tasks_run() {
            let server = secure_server_fixture(1).await.server;
            let monitor = server.test_task_monitor();

            assert!(
                tokio::time::timeout(
                    std::time::Duration::from_millis(25),
                    monitor.wait_for_idle()
                )
                .await
                .is_err(),
                "server ownership must be visible before its spawned supervisors first poll"
            );

            server.shutdown().await.unwrap();
            monitor.wait_for_idle().await;
        }

        #[tokio::test]
        async fn failed_bind_returns_before_artifact_supervisors_start() {
            let temporary = tempfile::tempdir().unwrap();
            let held = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let result = ArtifactUploadServer::start(
                held.local_addr().unwrap(),
                temporary.path().to_path_buf(),
                HashSet::new(),
                1,
            )
            .await;
            let Err(error) = result else {
                panic!("an already-bound address must fail before lifecycle startup");
            };
            assert!(error.to_string().contains("binding artifact upload server"));
        }

        #[tokio::test]
        async fn shutdown_is_bounded_with_a_stalled_tls_client() {
            let server = secure_server_fixture(1).await.server;
            let _stalled = tokio::net::TcpStream::connect(server.local_addr())
                .await
                .unwrap();
            tokio::time::timeout(std::time::Duration::from_secs(2), server.shutdown())
                .await
                .expect("shutdown must finish within its deadline")
                .expect("shutdown must reap the server tasks");
        }

        #[tokio::test]
        async fn shutdown_is_bounded_with_an_idle_hyper_client() {
            let fixture = secure_server_fixture(1).await;
            let client = fixture.client(0);
            let stream = tokio::net::TcpStream::connect(fixture.server.local_addr())
                .await
                .unwrap();
            let server_name = ServerName::try_from(ARTIFACT_TLS_SERVER_NAME.to_owned()).unwrap();
            let stream = TlsConnector::from(client.tls.clone())
                .connect(server_name, stream)
                .await
                .unwrap();
            let (mut sender, connection) =
                hyper::client::conn::http1::handshake(hyper_util::rt::TokioIo::new(stream))
                    .await
                    .unwrap();
            let connection = tokio::spawn(async move {
                let _ = connection.await;
            });
            let request = client
                .request("GET", "/dataset/does-not-exist", Empty::<Bytes>::new())
                .unwrap();
            let response = sender.send_request(request).await.unwrap();
            assert_eq!(response.status(), StatusCode::NOT_FOUND);
            let _ = response.into_body().collect().await.unwrap();

            tokio::time::timeout(std::time::Duration::from_secs(2), fixture.server.shutdown())
                .await
                .expect("shutdown must finish within its deadline")
                .expect("shutdown must reap the server tasks");

            drop(sender);
            connection.abort();
        }

        #[tokio::test]
        async fn drop_refuses_connections_after_reaping_tls_tasks() {
            let server = secure_server_fixture(1).await.server;
            let task_monitor = server.test_task_monitor();
            let address = server.local_addr();
            let stalled = tokio::net::TcpStream::connect(address).await.unwrap();
            task_monitor.wait_for_at_least(2).await;

            drop(server);

            tokio::time::timeout(
                std::time::Duration::from_secs(2),
                task_monitor.wait_for_idle(),
            )
            .await
            .expect("Drop must reap TLS supervisor and handshake tasks");
            assert!(tokio::net::TcpStream::connect(address).await.is_err());
            drop(stalled);
        }

        #[tokio::test]
        async fn shutdown_reaps_a_stalled_upload_connection() {
            use http_body_util::StreamBody;
            use hyper::body::Frame;
            use tokio_stream::wrappers::ReceiverStream;

            let temporary = tempfile::tempdir().unwrap();
            let server = ArtifactUploadServer::start(
                "127.0.0.1:0".parse().unwrap(),
                temporary.path().join("landed"),
                HashSet::from(["records.jsonl".to_owned()]),
                1,
            )
            .await
            .unwrap();
            let task_monitor = server.test_task_monitor();
            let client = test_client(&server, 0);
            let (body_tx, body_rx) = mpsc::channel(1);
            let body = StreamBody::new(
                ReceiverStream::new(body_rx)
                    .map(|bytes| Ok::<_, std::convert::Infallible>(Frame::data(bytes))),
            );
            let request = client
                .request("POST", "/cell/0/artifact/records.jsonl", body)
                .unwrap();
            let upload =
                tokio::spawn(async move { client.connect_and_send(request, "test").await });
            body_tx.send(Bytes::from_static(b"partial")).await.unwrap();
            task_monitor.wait_for_upload_writer().await;
            let final_path = temporary.path().join("landed/cell-0/records.jsonl");
            let part_path = part_path_for(&final_path);
            tokio::time::timeout(std::time::Duration::from_secs(2), async {
                while !part_path.exists() {
                    tokio::task::yield_now().await;
                }
            })
            .await
            .expect("upload handler must start its writer before shutdown");

            server.shutdown().await.unwrap();

            tokio::time::timeout(
                std::time::Duration::from_secs(2),
                task_monitor.wait_for_idle(),
            )
            .await
            .expect("shutdown must reap the stalled upload connection");
            assert!(upload.await.unwrap().is_err());
            assert!(!final_path.exists());
        }

        #[tokio::test]
        async fn cancelled_shutdown_rehomes_the_stalled_upload_writer() {
            use http_body_util::StreamBody;
            use hyper::body::Frame;
            use tokio_stream::wrappers::ReceiverStream;

            let temporary = tempfile::tempdir().unwrap();
            let server = ArtifactUploadServer::start(
                "127.0.0.1:0".parse().unwrap(),
                temporary.path().join("landed"),
                HashSet::from(["records.jsonl".to_owned()]),
                1,
            )
            .await
            .unwrap();
            let task_monitor = server.test_task_monitor();
            let client = test_client(&server, 0);
            let (body_tx, body_rx) = mpsc::channel(1);
            let body = StreamBody::new(
                ReceiverStream::new(body_rx)
                    .map(|bytes| Ok::<_, std::convert::Infallible>(Frame::data(bytes))),
            );
            let request = client
                .request("POST", "/cell/0/artifact/records.jsonl", body)
                .unwrap();
            let upload =
                tokio::spawn(async move { client.connect_and_send(request, "test").await });
            body_tx.send(Bytes::from_static(b"partial")).await.unwrap();
            task_monitor.wait_for_upload_writer().await;

            assert!(
                tokio::time::timeout(std::time::Duration::from_millis(25), server.shutdown(),)
                    .await
                    .is_err()
            );

            tokio::time::timeout(
                std::time::Duration::from_secs(2),
                task_monitor.wait_for_idle(),
            )
            .await
            .expect("cancelling shutdown must rehome and reap every live task");
            assert!(upload.await.unwrap().is_err());
            assert!(
                !temporary
                    .path()
                    .join("landed/cell-0/records.jsonl")
                    .exists(),
                "a reaped incomplete upload must not be published"
            );
        }

        #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        async fn drop_reaps_tasks_outside_the_entered_runtime() {
            let server = secure_server_fixture(1).await.server;
            let task_monitor = server.test_task_monitor();
            let address = server.local_addr();
            let stalled = tokio::net::TcpStream::connect(address).await.unwrap();
            task_monitor.wait_for_at_least(2).await;

            std::thread::spawn(move || drop(server)).join().unwrap();

            tokio::time::timeout(
                std::time::Duration::from_secs(2),
                task_monitor.wait_for_idle(),
            )
            .await
            .expect("Drop must use the origin runtime to reap its tasks");
            drop(stalled);
        }

        #[test]
        fn drop_after_origin_runtime_closes_does_not_recursively_reap() {
            let temporary = tempfile::tempdir().unwrap();
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();
            let server = runtime.block_on(async {
                ArtifactUploadServer::start(
                    "127.0.0.1:0".parse().unwrap(),
                    temporary.path().to_path_buf(),
                    HashSet::new(),
                    1,
                )
                .await
                .unwrap()
            });
            runtime.shutdown_background();

            drop(server);
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn artifact_channel_binds_credentials_to_cell_routes() {
        let fixture = secure_server_fixture(2).await;
        let cell_zero = fixture.client(0);
        assert_eq!(
            post_done_status(&cell_zero, 1).await.unwrap(),
            StatusCode::UNAUTHORIZED
        );
        assert_eq!(
            post_done_status(&cell_zero, 0).await.unwrap(),
            StatusCode::OK
        );
        fixture.server.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn artifact_channel_registrar_binds_each_digest_once() {
        let fixture = unregistered_secure_server_fixture(2).await;
        let digest_zero = ArtifactBearer::from_test_bytes([0x11; 32]).digest_bytes();
        let digest_one = ArtifactBearer::from_test_bytes([0x22; 32]).digest_bytes();
        fixture.registrar.prepare(0, digest_zero).unwrap().commit();
        assert!(fixture.registrar.prepare(0, digest_zero).is_err());
        assert!(fixture.registrar.prepare(0, digest_one).is_err());
        assert!(fixture.registrar.prepare(1, digest_zero).is_err());
        assert!(fixture.registrar.prepare(2, digest_one).is_err());
        fixture.server.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn artifact_registration_reservation_blocks_competing_cell() {
        let fixture = unregistered_secure_server_fixture(2).await;
        let bearer = ArtifactBearer::from_test_bytes([0x39; 32]);
        let owner = ArtifactChannelClient::new(
            fixture.server.local_addr().to_string(),
            fixture.registrar.server_config(),
            ArtifactBearer::from_test_bytes([0x39; 32]),
        )
        .unwrap();
        let plan = fixture.registrar.prepare(0, bearer.digest_bytes()).unwrap();

        assert!(fixture.registrar.prepare(1, bearer.digest_bytes()).is_err());
        assert_eq!(
            post_done_status(&owner, 0).await.unwrap(),
            StatusCode::UNAUTHORIZED
        );
        plan.commit();
        assert_eq!(post_done_status(&owner, 0).await.unwrap(), StatusCode::OK);
        assert_eq!(
            post_done_status(&owner, 1).await.unwrap(),
            StatusCode::UNAUTHORIZED
        );
        fixture.server.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn dropping_artifact_registration_reservation_allows_new_owner() {
        let fixture = unregistered_secure_server_fixture(2).await;
        let bearer = ArtifactBearer::from_test_bytes([0x3A; 32]);
        let new_owner = ArtifactChannelClient::new(
            fixture.server.local_addr().to_string(),
            fixture.registrar.server_config(),
            ArtifactBearer::from_test_bytes([0x3A; 32]),
        )
        .unwrap();
        let abandoned = fixture.registrar.prepare(0, bearer.digest_bytes()).unwrap();

        drop(abandoned);
        let replacement = fixture.registrar.prepare(1, bearer.digest_bytes()).unwrap();
        replacement.commit();

        assert_eq!(
            post_done_status(&new_owner, 0).await.unwrap(),
            StatusCode::UNAUTHORIZED
        );
        assert_eq!(
            post_done_status(&new_owner, 1).await.unwrap(),
            StatusCode::OK
        );
        fixture.server.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn registration_transaction_artifact_plan_is_invisible_until_commit() {
        let fixture = unregistered_secure_server_fixture(1).await;
        let bearer = ArtifactBearer::from_test_bytes([0x31; 32]);
        let client = ArtifactChannelClient::new(
            fixture.server.local_addr().to_string(),
            fixture.registrar.server_config(),
            ArtifactBearer::from_test_bytes([0x31; 32]),
        )
        .unwrap();
        let plan = fixture.registrar.prepare(0, bearer.digest_bytes()).unwrap();

        assert_eq!(
            post_done_status(&client, 0).await.unwrap(),
            StatusCode::UNAUTHORIZED
        );
        plan.commit();
        assert_eq!(post_done_status(&client, 0).await.unwrap(), StatusCode::OK);
        fixture.server.shutdown().await.unwrap();
    }

    #[test]
    fn artifact_channel_secrets_are_redacted() {
        let bearer = ArtifactBearer::from_test_bytes([0xA5; 32]);
        assert_eq!(format!("{bearer:?}"), "ArtifactBearer([REDACTED])");
        let register = CellRegister {
            cell_id: 0,
            cell_peer: Vec::new(),
            artifact_capability_digest: Some(bearer.digest_bytes()),
            registration_proof: None,
        };
        let encoded = rmp_serde::to_vec(&register).unwrap();
        assert!(!encoded.windows(32).any(|window| window == [0xA5; 32]));
    }

    fn sample_bytes(len: usize) -> Vec<u8> {
        let mut out = Vec::with_capacity(len);
        let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
        while out.len() < len {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            out.extend_from_slice(&state.to_le_bytes());
        }
        out.truncate(len);
        out
    }

    fn compress_to_chunks(path: &Path) -> Vec<Vec<u8>> {
        let mut compressor = FileCompressor::open(path).unwrap();
        let mut chunks = Vec::new();
        while let Some(chunk) = compressor.next_chunk().unwrap() {
            assert!(
                chunk.len() <= CHUNK_SIZE,
                "compressed chunk {} exceeds CHUNK_SIZE {CHUNK_SIZE}",
                chunk.len()
            );
            chunks.push(chunk);
        }
        chunks
    }

    #[test]
    fn streaming_zstd_round_trips_byte_for_byte() {
        let dir = tempfile::tempdir().unwrap();
        let src = dir.path().join("source.bin");
        let payload = sample_bytes(CHUNK_SIZE * 5 + 12345);
        std::fs::write(&src, &payload).unwrap();

        let chunks = compress_to_chunks(&src);
        assert!(chunks.len() >= 2, "payload must span multiple chunks");

        let dest = dir.path().join("nested").join("dest.bin");
        let mut sink = DecompressToFile::create(&dest).unwrap();
        assert!(
            part_path_for(&dest).exists(),
            ".part staged during transfer"
        );
        assert!(!dest.exists(), "final file not created until finish()");
        for chunk in &chunks {
            sink.write_chunk(chunk).unwrap();
        }
        sink.finish().unwrap();

        assert_eq!(std::fs::read(&dest).unwrap(), payload);
        assert!(
            !part_path_for(&dest).exists(),
            ".part renamed away on finish"
        );
    }

    #[test]
    fn empty_file_round_trips() {
        let dir = tempfile::tempdir().unwrap();
        let src = dir.path().join("empty.bin");
        std::fs::File::create(&src).unwrap();
        let chunks = compress_to_chunks(&src);
        let dest = dir.path().join("empty-out.bin");
        let mut sink = DecompressToFile::create(&dest).unwrap();
        for chunk in &chunks {
            sink.write_chunk(chunk).unwrap();
        }
        sink.finish().unwrap();
        assert!(std::fs::read(&dest).unwrap().is_empty());
    }

    #[test]
    fn path_validation_rejects_traversal_and_unknown() {
        let mut allowed = HashSet::new();
        allowed.insert("profile_export.jsonl".to_owned());
        allowed.insert("inputs.json".to_owned());

        assert!(validate_artifact_relpath("profile_export.jsonl", &allowed).is_ok());
        assert!(validate_artifact_relpath("inputs.json", &allowed).is_ok());

        for bad in [
            "../etc/passwd",
            "/etc/passwd",
            "a/../b",
            "./profile_export.jsonl",
            "profile_export.parquet", // valid shape but not in this run's allowlist
            "",
        ] {
            assert!(
                validate_artifact_relpath(bad, &allowed).is_err(),
                "should reject {bad:?}"
            );
        }
    }

    #[test]
    fn plain_uncompressed_sink_round_trips() {
        let dir = tempfile::tempdir().unwrap();
        let dest = dir.path().join("plain.bin");
        let payload = sample_bytes(1000);
        let mut sink = PlainToFile::create(&dest).unwrap();
        sink.write_chunk(&payload[..400]).unwrap();
        sink.write_chunk(&payload[400..]).unwrap();
        sink.finish().unwrap();
        assert_eq!(std::fs::read(&dest).unwrap(), payload);
    }

    #[test]
    fn truncated_zstd_frame_never_renames_the_part_file() {
        let dir = tempfile::tempdir().unwrap();
        let dest = dir.path().join("truncated.bin");
        let (tx, rx) = mpsc::channel(2);
        tx.blocking_send(UploadChunk::Data(Bytes::from_static(b"\x28\xb5\x2f\xfd")))
            .unwrap();
        tx.blocking_send(UploadChunk::Complete).unwrap();
        drop(tx);

        assert!(decode_channel_to_file(rx, &dest, true).is_err());
        assert!(!dest.exists(), "truncated zstd input must not be published");
    }

    #[tokio::test]
    async fn failed_artifact_compressor_never_completes_the_upload() {
        let temporary = tempfile::tempdir().unwrap();
        let server = ArtifactUploadServer::start(
            "127.0.0.1:0".parse().unwrap(),
            temporary.path().join("landed"),
            HashSet::from(["records.jsonl".to_owned()]),
            1,
        )
        .await
        .unwrap();
        let task_monitor = server.test_task_monitor();
        let missing = temporary.path().join("missing-records.jsonl");
        let result = upload_one(&test_client(&server, 0), 0, "records.jsonl", &missing).await;

        assert!(result.is_err(), "a failed compressor must fail the upload");
        server.shutdown().await.unwrap();
        task_monitor.wait_for_idle().await;
        assert!(
            !temporary
                .path()
                .join("landed/cell-0/records.jsonl")
                .exists(),
            "producer failure must not become a successful atomic rename"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn wait_for_cells_returns_when_completed_before_waiter_registers() {
        use std::time::{Duration, Instant};

        let root = tempfile::tempdir().unwrap();
        let server = ArtifactUploadServer::start(
            "127.0.0.1:0".parse().unwrap(),
            root.path().to_path_buf(),
            HashSet::new(),
            3,
        )
        .await
        .unwrap();

        let cell_count = 3u32;
        server.state.done.send_modify(|done| {
            for id in 0..cell_count {
                done.insert(id);
            }
        });

        let start = Instant::now();
        server
            .wait_for_cells(cell_count, Duration::from_secs(30))
            .await
            .expect("barrier must release when all cells completed before the waiter registered");
        assert!(
            start.elapsed() < Duration::from_secs(5),
            "wait_for_cells returned after {:?}, i.e. blocked on a lost wakeup",
            start.elapsed()
        );

        server.shutdown().await.unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn wait_for_cells_times_out_and_names_missing_cells() {
        use std::time::Duration;

        let root = tempfile::tempdir().unwrap();
        let server = ArtifactUploadServer::start(
            "127.0.0.1:0".parse().unwrap(),
            root.path().to_path_buf(),
            HashSet::new(),
            3,
        )
        .await
        .unwrap();

        server.state.done.send_modify(|done| {
            done.insert(0);
            done.insert(1);
        });

        let error = server
            .wait_for_cells(3, Duration::from_millis(200))
            .await
            .expect_err("barrier must time out when a cell never signals done");
        let message = error.to_string();
        assert!(
            message.contains("2 of 3") && message.contains('2'),
            "timeout error should report progress and the missing cell: {message}"
        );

        server.shutdown().await.unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn in_process_ship_then_concat_matches_batch_over_union() {
        use crate::engine::protocol::ArtifactSpec;
        use crate::engine::shard_artifacts::concatenate_cell_artifacts;

        let root = tempfile::tempdir().unwrap();
        let cell_count = 2u32;
        let source_root = root.path().join("sources");
        let records_rel = "profile_export.jsonl";
        let inputs_rel = "inputs.json";
        let source_dirs: Vec<PathBuf> = (0..cell_count)
            .map(|id| source_root.join(format!("cell-{id}")))
            .collect();
        for (id, dir) in source_dirs.iter().enumerate() {
            std::fs::create_dir_all(dir).unwrap();
            let mut records = std::fs::File::create(dir.join(records_rel)).unwrap();
            for row in 0..2000 {
                writeln!(
                    records,
                    "{{\"cell\":{id},\"row\":{row},\"pad\":\"{}\"}}",
                    "x".repeat(40)
                )
                .unwrap();
            }
            std::fs::write(
                dir.join(inputs_rel),
                b"{\"schema\":\"inputs\",\"data\":[1,2,3]}",
            )
            .unwrap();
        }

        let artifacts = ArtifactSpec {
            records_path: Some(PathBuf::from(records_rel)),
            raw_path: None,
            records_csv_path: None,
            records_parquet_path: None,
            outputs_path: None,
            inputs_path: Some(PathBuf::from(inputs_rel)),
            trace: false,
            dataset_analysis_path: None,
            ..Default::default()
        };
        let relatives = shippable_relatives(&artifacts);
        let allowed: HashSet<String> = relatives.iter().cloned().collect();
        let controller_root = root.path().join("controller-temp");
        std::fs::create_dir_all(&controller_root).unwrap();

        let server = ArtifactUploadServer::start(
            "127.0.0.1:0".parse().unwrap(),
            controller_root.clone(),
            allowed,
            cell_count,
        )
        .await
        .unwrap();
        for (id, dir) in source_dirs.iter().enumerate() {
            let client = test_client(&server, id as u32);
            ship_cell_artifacts(&client, id as u32, dir, &relatives)
                .await
                .unwrap();
        }
        server
            .wait_for_cells(cell_count, std::time::Duration::from_secs(10))
            .await
            .unwrap();

        for (id, src_dir) in source_dirs.iter().enumerate() {
            let landed_dir = controller_root.join(format!("cell-{id}"));
            for rel in [records_rel, inputs_rel] {
                assert_eq!(
                    std::fs::read(landed_dir.join(rel)).unwrap(),
                    std::fs::read(src_dir.join(rel)).unwrap(),
                    "cell {id} {rel} landed byte-identical"
                );
                assert!(
                    !part_path_for(&landed_dir.join(rel)).exists(),
                    "no lingering .part for cell {id} {rel}"
                );
            }
        }

        let controller_dirs: Vec<PathBuf> = (0..cell_count)
            .map(|id| controller_root.join(format!("cell-{id}")))
            .collect();
        let merged_from_uploaded = root.path().join("merged-uploaded");
        std::fs::create_dir_all(&merged_from_uploaded).unwrap();
        concatenate_cell_artifacts(&controller_dirs, &merged_from_uploaded, &artifacts).unwrap();

        let merged_from_source = root.path().join("merged-source");
        std::fs::create_dir_all(&merged_from_source).unwrap();
        concatenate_cell_artifacts(&source_dirs, &merged_from_source, &artifacts).unwrap();

        let line_set = |path: &Path| -> std::collections::BTreeMap<String, usize> {
            let mut set = std::collections::BTreeMap::new();
            let text = std::fs::read_to_string(path).unwrap();
            for line in text.lines() {
                *set.entry(line.to_owned()).or_insert(0) += 1;
            }
            set
        };
        assert_eq!(
            line_set(&merged_from_uploaded.join(records_rel)),
            line_set(&merged_from_source.join(records_rel)),
            "merged-from-uploaded records line SET == batch over source union"
        );
        assert_eq!(
            std::fs::read_to_string(merged_from_uploaded.join(records_rel))
                .unwrap()
                .lines()
                .count(),
            4000,
        );

        server.shutdown().await.unwrap();
    }

    #[test]
    fn dataset_relname_validation_rejects_traversal() {
        assert!(validate_dataset_relname("shard-000.jsonl").is_ok());
        assert!(validate_dataset_relname("sub/deep.json").is_ok());
        for bad in ["", "../x", "a/../b", "/etc/passwd", ".", "a//b"] {
            assert!(
                validate_dataset_relname(bad).is_err(),
                "must reject {bad:?}"
            );
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn manifest_directory_reconstructs_identical_tree() {
        let dir = tempfile::tempdir().unwrap();
        let src_dir = dir.path().join("shards");
        std::fs::create_dir_all(&src_dir).unwrap();
        let mut datasets = HashMap::new();
        let mut names = Vec::new();
        for shard in ["a.json", "b.json"] {
            let path = src_dir.join(shard);
            let mut file = std::fs::File::create(&path).unwrap();
            for row in 0..2000 {
                writeln!(
                    file,
                    "{{\"shard\":\"{shard}\",\"row\":{row},\"pad\":\"{}\"}}",
                    "z".repeat(40)
                )
                .unwrap();
            }
            datasets.insert(shard.to_owned(), DatasetSource::Path(path));
            names.push(shard.to_owned());
        }
        let manifest = DatasetManifest {
            kind: "dir".to_owned(),
            base_name: "shards".to_owned(),
            files: names.clone(),
        };
        let server = ArtifactUploadServer::start_with_dataset_plan(
            "127.0.0.1:0".parse().unwrap(),
            dir.path().join("controller-temp"),
            HashSet::new(),
            datasets,
            Some(manifest.clone()),
            1,
        )
        .await
        .unwrap();
        let client = test_client(&server, 0);

        let fetched = fetch_dataset_manifest(&client).await.unwrap();
        assert_eq!(fetched, manifest, "manifest round-trips over HTTP");

        let cell_dir = dir.path().join("cell").join("dataset");
        let rewritten = reconstruct_shipped_dataset(&client, &fetched, &cell_dir)
            .await
            .unwrap();
        assert_eq!(
            rewritten, cell_dir,
            "dir kind points path at the reconstructed dir"
        );

        for shard in &names {
            assert_eq!(
                std::fs::read(cell_dir.join(shard)).unwrap(),
                std::fs::read(src_dir.join(shard)).unwrap(),
                "shard {shard} landed byte-identical"
            );
        }
        let mut landed: Vec<String> = std::fs::read_dir(&cell_dir)
            .unwrap()
            .map(|e| e.unwrap().file_name().to_str().unwrap().to_owned())
            .collect();
        landed.sort();
        assert_eq!(landed, names, "reconstructed dir == shipped set");

        server.shutdown().await.unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn manifest_prefix_points_path_at_stem_beside_shards() {
        let dir = tempfile::tempdir().unwrap();
        let src_dir = dir.path().join("src");
        std::fs::create_dir_all(&src_dir).unwrap();
        let mut datasets = HashMap::new();
        let shard_names = ["trace.000000.jsonl.gz", "trace.000001.jsonl.gz"];
        for shard in shard_names {
            let path = src_dir.join(shard);
            std::fs::write(&path, format!("{shard}\n")).unwrap();
            datasets.insert(shard.to_owned(), DatasetSource::Path(path));
        }
        let manifest = DatasetManifest {
            kind: "prefix".to_owned(),
            base_name: "trace.jsonl.gz".to_owned(),
            files: shard_names.iter().map(|s| s.to_string()).collect(),
        };
        let server = ArtifactUploadServer::start_with_dataset_plan(
            "127.0.0.1:0".parse().unwrap(),
            dir.path().join("controller-temp"),
            HashSet::new(),
            datasets,
            Some(manifest.clone()),
            1,
        )
        .await
        .unwrap();
        let client = test_client(&server, 0);
        let cell_dir = dir.path().join("cell").join("dataset");
        let rewritten = reconstruct_shipped_dataset(&client, &manifest, &cell_dir)
            .await
            .unwrap();
        assert_eq!(
            rewritten,
            cell_dir.join("trace.jsonl.gz"),
            "prefix kind points path at the stem beside the shards"
        );
        for shard in shard_names {
            assert!(cell_dir.join(shard).is_file(), "shard {shard} landed");
        }
        server.shutdown().await.unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn manifest_reconstructs_nested_replay_assets_without_flattening() {
        let dir = tempfile::tempdir().unwrap();
        let source = dir.path().join("source");
        let recording = source.join("recordings/trace.json");
        let asset = source.join("benchmark/pinchbench/assets/input.txt");
        std::fs::create_dir_all(recording.parent().unwrap()).unwrap();
        std::fs::create_dir_all(asset.parent().unwrap()).unwrap();
        std::fs::write(&recording, b"recording").unwrap();
        std::fs::write(&asset, b"workspace asset").unwrap();
        let datasets = HashMap::from([
            (
                "recordings/trace.json".to_owned(),
                DatasetSource::Path(recording),
            ),
            (
                "benchmark/pinchbench/assets/input.txt".to_owned(),
                DatasetSource::Path(asset),
            ),
        ]);
        let manifest = DatasetManifest {
            kind: "replay_root".to_owned(),
            base_name: "recordings/trace.json".to_owned(),
            files: vec![
                "recordings/trace.json".to_owned(),
                "benchmark/pinchbench/assets/input.txt".to_owned(),
            ],
        };
        let server = ArtifactUploadServer::start_with_dataset_plan(
            "127.0.0.1:0".parse().unwrap(),
            dir.path().join("controller-temp"),
            HashSet::new(),
            datasets,
            Some(manifest.clone()),
            1,
        )
        .await
        .unwrap();
        let landed = dir.path().join("landed");
        let client = test_client(&server, 0);
        let rewritten = reconstruct_shipped_dataset(&client, &manifest, &landed)
            .await
            .unwrap();

        assert_eq!(rewritten, landed.join("recordings/trace.json"));
        assert_eq!(
            std::fs::read(landed.join("benchmark/pinchbench/assets/input.txt")).unwrap(),
            b"workspace asset"
        );
        server.shutdown().await.unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn agent_session_exact_set_reconstructs_only_nested_discovered_sources() {
        let dir = tempfile::tempdir().unwrap();
        let source = dir.path().join("source");
        let selected = source.join("selected");
        let main = selected.join("main.jsonl");
        let subagent = selected.join("main/subagents/agent-aaa.jsonl");
        for (path, bytes) in [
            (
                &main,
                b"{\"sessionId\":\"main\",\"parentUuid\":null}\n".as_slice(),
            ),
            (
                &subagent,
                b"{\"sessionId\":\"subagent\",\"parentUuid\":null}\n".as_slice(),
            ),
        ] {
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(path, bytes).unwrap();
        }
        let manifest = DatasetManifest {
            kind: "agent_session_set".to_owned(),
            base_name: "selected".to_owned(),
            files: vec![
                "selected/main.jsonl".to_owned(),
                "selected/main/subagents/agent-aaa.jsonl".to_owned(),
            ],
        };
        let server = ArtifactUploadServer::start_with_dataset_plan(
            "127.0.0.1:0".parse().unwrap(),
            dir.path().join("controller-temp"),
            HashSet::new(),
            HashMap::from([
                (
                    "selected/main.jsonl".to_owned(),
                    DatasetSource::Path(main.clone()),
                ),
                (
                    "selected/main/subagents/agent-aaa.jsonl".to_owned(),
                    DatasetSource::Path(subagent.clone()),
                ),
            ]),
            Some(manifest.clone()),
            1,
        )
        .await
        .unwrap();
        let landed = dir.path().join("landed");
        let client = test_client(&server, 0);
        let rewritten = reconstruct_shipped_dataset(&client, &manifest, &landed)
            .await
            .unwrap();

        assert_eq!(rewritten, landed.join("selected"));
        assert!(!landed.join("secret.jsonl").exists());
        let source_read_set =
            crate::graph::recorded::agent_recording::discover_imported_agent_read_set(
                &selected,
                Some(&source),
                crate::config::model::dataset::RecordedAgentSourceFormat::ClaudeCode,
                Some(true),
            )
            .unwrap();
        let rediscovered =
            crate::graph::recorded::agent_recording::discover_imported_agent_read_set(
                &rewritten,
                Some(&landed),
                crate::config::model::dataset::RecordedAgentSourceFormat::ClaudeCode,
                Some(true),
            )
            .unwrap();
        let read_set_bytes =
            |read_set: crate::graph::recorded::agent_recording::ImportedAgentReadSet| {
                read_set
                    .files
                    .into_iter()
                    .map(|file| (file.relative_path, std::fs::read(file.path).unwrap()))
                    .collect::<std::collections::BTreeMap<_, _>>()
            };
        assert_eq!(
            read_set_bytes(rediscovered),
            read_set_bytes(source_read_set),
            "the landed selected directory must rediscover the exact complete session set"
        );
        server.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn agent_session_exact_set_rejects_duplicate_paths_before_fetching() {
        let dest = tempfile::tempdir().unwrap();
        let manifest = DatasetManifest {
            kind: "agent_session_set".to_owned(),
            base_name: "main.jsonl".to_owned(),
            files: vec!["main.jsonl".to_owned(), "main.jsonl".to_owned()],
        };
        let client = unreachable_test_client();
        let error = reconstruct_shipped_dataset(&client, &manifest, dest.path())
            .await
            .expect_err("a duplicate must reject before attempting the unreachable server");
        assert!(
            format!("{error:#}").contains("duplicate dataset manifest path \"main.jsonl\""),
            "duplicate rejection must win over an unreachable fetch: {error:#}"
        );
    }

    #[tokio::test]
    async fn agent_session_exact_set_rejects_traversal_paths() {
        let dest = tempfile::tempdir().unwrap();
        let manifest = DatasetManifest {
            kind: "agent_session_set".to_owned(),
            base_name: "main.jsonl".to_owned(),
            files: vec!["../secret.jsonl".to_owned()],
        };
        assert!(
            reconstruct_shipped_dataset(&unreachable_test_client(), &manifest, dest.path(),)
                .await
                .is_err()
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn server_rejects_unallowed_upload() {
        let root = tempfile::tempdir().unwrap();
        let mut allowed = HashSet::new();
        allowed.insert("profile_export.jsonl".to_owned());
        let server = ArtifactUploadServer::start(
            "127.0.0.1:0".parse().unwrap(),
            root.path().to_path_buf(),
            allowed,
            1,
        )
        .await
        .unwrap();
        let client = test_client(&server, 0);
        let src = root.path().join("payload.bin");
        std::fs::write(&src, b"hello").unwrap();

        let unknown = upload_one(&client, 0, "secret.parquet", &src).await;
        assert!(unknown.is_err(), "unallowed artifact must be rejected");
        server.shutdown().await.unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn serve_then_download_round_trips_dataset_bytes() {
        let dir = tempfile::tempdir().unwrap();
        let src = dir.path().join("prompts.jsonl");
        let mut file = std::fs::File::create(&src).unwrap();
        for row in 0..2000 {
            writeln!(file, "{{\"text\":\"prompt {row} {}\"}}", "y".repeat(40)).unwrap();
        }
        drop(file);
        let source_bytes = std::fs::read(&src).unwrap();
        assert!(source_bytes.len() > CHUNK_SIZE, "source spans many chunks");

        let mut datasets = HashMap::new();
        datasets.insert(
            "prompts.jsonl".to_owned(),
            super::DatasetSource::Path(src.clone()),
        );
        let server = ArtifactUploadServer::start_with_datasets(
            "127.0.0.1:0".parse().unwrap(),
            dir.path().join("controller-temp"),
            HashSet::new(),
            datasets,
            1,
        )
        .await
        .unwrap();
        let client = test_client(&server, 0);

        let dest = dir.path().join("cell").join("prompts.jsonl");
        fetch_dataset_to_file(&client, "prompts.jsonl", &dest)
            .await
            .unwrap();
        assert_eq!(
            std::fs::read(&dest).unwrap(),
            source_bytes,
            "downloaded dataset is byte-identical to the source"
        );
        assert!(
            !part_path_for(&dest).exists(),
            "no lingering .part after atomic rename"
        );

        let missing = dir.path().join("cell").join("unknown.jsonl");
        assert!(
            fetch_dataset_to_file(&client, "unknown.jsonl", &missing)
                .await
                .is_err(),
            "an unregistered dataset name must be rejected"
        );

        server.shutdown().await.unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn compiled_dataset_matches_between_original_and_shipped_file() {
        use crate::dataset::{
            ComposeConfig, DatasetSource, LoadConfig, LoaderRegistry, TiktokenTokenizer,
        };
        use crate::rng::RngRoot;

        let dir = tempfile::tempdir().unwrap();
        let src = dir.path().join("prompts.jsonl");
        std::fs::write(
            &src,
            b"{\"text\":\"alpha\",\"output_length\":7}\n\
              {\"text\":\"beta\"}\n\
              {\"session_id\":\"s\",\"text\":\"gamma\"}\n",
        )
        .unwrap();

        let mut datasets = HashMap::new();
        datasets.insert(
            "prompts.jsonl".to_owned(),
            super::DatasetSource::Path(src.clone()),
        );
        let server = ArtifactUploadServer::start_with_datasets(
            "127.0.0.1:0".parse().unwrap(),
            dir.path().join("controller-temp"),
            HashSet::new(),
            datasets,
            1,
        )
        .await
        .unwrap();
        let client = test_client(&server, 0);
        let shipped = dir.path().join("cell").join("prompts.jsonl");
        fetch_dataset_to_file(&client, "prompts.jsonl", &shipped)
            .await
            .unwrap();
        server.shutdown().await.unwrap();

        let compile = |path: &Path| {
            let path = path.to_path_buf();
            async move {
                LoaderRegistry::with_builtin_formats()
                    .unwrap()
                    .build_dataset(
                        Some("single_turn"),
                        &LoadConfig::new(DatasetSource::Path(path)),
                        &ComposeConfig::new("model", RngRoot::new(Some(4242))),
                        &TiktokenTokenizer::builtin(),
                    )
                    .await
                    .unwrap()
            }
        };
        let original = compile(&src).await;
        let from_shipped = compile(&shipped).await;
        assert_eq!(
            original.conversations(),
            from_shipped.conversations(),
            "recompiling shipped bytes yields the identical conversation list"
        );
    }
}
