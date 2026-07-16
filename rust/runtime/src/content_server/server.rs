// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Run-owned HTTP server and full-response lifecycle instrumentation.
//!
//! Routes, startup, directory validation, traversal confinement, captured
//! fields, and interval definitions are implemented here.

use std::collections::BTreeMap;
use std::fmt;
use std::net::{Ipv6Addr, SocketAddr};
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context as TaskContext, Poll};

use async_trait::async_trait;
use axum::Router;
use axum::body::Body;
use axum::extract::{ConnectInfo, State};
use axum::http::{HeaderMap, Request, Response, StatusCode, Version, header};
use axum::middleware::{self, Next};
use axum::routing::get;
use bytes::Bytes;
use http_body::{Body as HttpBody, Frame, SizeHint};
use percent_encoding::percent_decode_str;
use tempfile::TempDir;
use tokio::net::TcpListener;
use tokio::sync::oneshot;
use tokio::task::JoinHandle;
use tower_http::services::ServeDir;

use crate::content_server::model::{
    ContentRequestRecord, ContentServerStatus, RequestTrackerSnapshot,
};
use crate::content_server::tracker::{
    ContentServerClock, RequestTracker, SystemContentServerClock,
};
use crate::content_server::{ContentServerError, Result};

/// Listener, serving-root, and bounded tracking policy for one run.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ContentServerConfig {
    /// Host/interface to bind and advertise.
    pub host: String,
    /// TCP port. `0` is accepted by the library for isolated tests; the strict
    /// runner adapter accepts only `1..=65535` like the Python environment.
    pub port: u16,
    /// Existing directory to serve, or `None` for a run-scoped temporary root.
    pub content_dir: Option<PathBuf>,
    /// Maximum number of recent request records to retain.
    pub max_tracked_records: usize,
}

/// Running server boundary retained by the runner's resource bundle.
#[async_trait]
pub trait ContentServerRuntime: fmt::Debug + Send {
    /// Current listener status.
    fn status(&self) -> ContentServerStatus;
    /// Actual bound socket address (useful when port `0` was requested).
    fn local_addr(&self) -> SocketAddr;
    /// Clone an internally consistent request-tracker snapshot.
    fn request_snapshot(&self) -> RequestTrackerSnapshot;
    /// Gracefully stop accepting requests and drain active connections.
    async fn shutdown(&mut self) -> Result<()>;
}

/// Injectable construction seam for one run-owned content server.
#[async_trait]
pub trait ContentServerFactory: fmt::Debug + Send + Sync {
    /// Bind the listener and return only after the socket is ready.
    async fn start(&self, config: ContentServerConfig) -> Result<Box<dyn ContentServerRuntime>>;
}

/// Axum/tower-http server factory with an injectable two-clock source.
#[derive(Clone)]
pub struct NativeContentServerFactory {
    clock: Arc<dyn ContentServerClock>,
}

impl fmt::Debug for NativeContentServerFactory {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("NativeContentServerFactory")
            .field("clock", &self.clock)
            .finish()
    }
}

impl Default for NativeContentServerFactory {
    fn default() -> Self {
        Self {
            clock: Arc::new(SystemContentServerClock::default()),
        }
    }
}

impl NativeContentServerFactory {
    /// Bind request tracking to an alternate wall/monotonic clock.
    pub fn new(clock: Arc<dyn ContentServerClock>) -> Self {
        Self { clock }
    }
}

#[derive(Clone)]
struct ServerState {
    tracker: Arc<RequestTracker>,
    clock: Arc<dyn ContentServerClock>,
    content_dir: PathBuf,
}

struct NativeContentServerRuntime {
    status: ContentServerStatus,
    local_addr: SocketAddr,
    tracker: Arc<RequestTracker>,
    shutdown_tx: Option<oneshot::Sender<()>>,
    task: Option<JoinHandle<std::io::Result<()>>>,
    _temporary_directory: Option<TempDir>,
}

impl fmt::Debug for NativeContentServerRuntime {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("NativeContentServerRuntime")
            .field("status", &self.status)
            .field("local_addr", &self.local_addr)
            .finish_non_exhaustive()
    }
}

#[async_trait]
impl ContentServerRuntime for NativeContentServerRuntime {
    fn status(&self) -> ContentServerStatus {
        self.status.clone()
    }

    fn local_addr(&self) -> SocketAddr {
        self.local_addr
    }

    fn request_snapshot(&self) -> RequestTrackerSnapshot {
        self.tracker.snapshot()
    }

    async fn shutdown(&mut self) -> Result<()> {
        if let Some(shutdown) = self.shutdown_tx.take() {
            let _ = shutdown.send(());
        }
        // Capture the serving outcome without short-circuiting: a join/serve
        // failure must still mark the server disabled and drop the run-scoped
        // temporary directory, otherwise a failed shutdown leaves
        // `status.enabled == true` and leaks the temp dir on disk.
        let serving_result = match self.task.take() {
            Some(task) => match task.await {
                Ok(serving) => serving.map_err(|source| {
                    ContentServerError::io("serving content-server connections", source)
                }),
                Err(error) => Err(ContentServerError::Task(error.to_string())),
            },
            None => Ok(()),
        };
        self.status.enabled = false;
        let cleanup_result = match self._temporary_directory.take() {
            Some(temporary_directory) => temporary_directory.close().map_err(|source| {
                ContentServerError::io("removing temporary content-server directory", source)
            }),
            None => Ok(()),
        };
        serving_result.and(cleanup_result)
    }
}

impl Drop for NativeContentServerRuntime {
    fn drop(&mut self) {
        if let Some(shutdown) = self.shutdown_tx.take() {
            let _ = shutdown.send(());
        }
        if let Some(task) = self.task.take() {
            task.abort();
        }
    }
}

#[async_trait]
impl ContentServerFactory for NativeContentServerFactory {
    async fn start(&self, config: ContentServerConfig) -> Result<Box<dyn ContentServerRuntime>> {
        if config.host.trim().is_empty() || config.host.trim() != config.host {
            return Err(ContentServerError::invalid(
                "content-server host must be non-empty and contain no surrounding whitespace",
            ));
        }
        let (content_dir, temporary_directory) = prepare_content_directory(config.content_dir)?;
        let listener = TcpListener::bind((config.host.as_str(), config.port))
            .await
            .map_err(|source| {
                ContentServerError::io(
                    format!("binding content server to {}:{}", config.host, config.port),
                    source,
                )
            })?;
        let local_addr = listener.local_addr().map_err(|source| {
            ContentServerError::io("reading content-server listener address", source)
        })?;
        let base_url = format!(
            "http://{}:{}",
            advertised_host(&config.host),
            local_addr.port()
        );
        let tracker = Arc::new(RequestTracker::new(config.max_tracked_records));
        let state = Arc::new(ServerState {
            tracker: tracker.clone(),
            clock: self.clock.clone(),
            content_dir: content_dir.clone(),
        });
        let app = Router::new()
            .route("/healthz", get(|| async { "ok" }))
            .nest_service(
                "/content",
                ServeDir::new(content_dir.clone()).append_index_html_on_directories(false),
            )
            .layer(middleware::from_fn_with_state(state.clone(), track_request));
        let (shutdown_tx, shutdown_rx) = oneshot::channel();
        let task = tokio::spawn(async move {
            axum::serve(
                listener,
                app.into_make_service_with_connect_info::<SocketAddr>(),
            )
            .with_graceful_shutdown(async move {
                let _ = shutdown_rx.await;
            })
            .await
        });

        Ok(Box::new(NativeContentServerRuntime {
            status: ContentServerStatus {
                enabled: true,
                base_url,
                content_dir,
                reason: None,
            },
            local_addr,
            tracker,
            shutdown_tx: Some(shutdown_tx),
            task: Some(task),
            _temporary_directory: temporary_directory,
        }))
    }
}

fn prepare_content_directory(authored: Option<PathBuf>) -> Result<(PathBuf, Option<TempDir>)> {
    match authored {
        Some(path) => {
            let canonical = path.canonicalize().map_err(|source| {
                ContentServerError::io(
                    format!("resolving content directory {}", path.display()),
                    source,
                )
            })?;
            if !canonical.is_dir() {
                return Err(ContentServerError::invalid(format!(
                    "content directory is not a directory: {}",
                    canonical.display()
                )));
            }
            Ok((canonical, None))
        }
        None => {
            let temporary = tempfile::Builder::new()
                .prefix("aiperf_content_")
                .tempdir()
                .map_err(|source| {
                    ContentServerError::io("creating temporary content-server directory", source)
                })?;
            let canonical = temporary.path().canonicalize().map_err(|source| {
                ContentServerError::io("resolving temporary content-server directory", source)
            })?;
            Ok((canonical, Some(temporary)))
        }
    }
}

fn advertised_host(host: &str) -> String {
    if host.parse::<Ipv6Addr>().is_ok() {
        format!("[{host}]")
    } else {
        host.to_owned()
    }
}

async fn track_request(
    State(state): State<Arc<ServerState>>,
    request: Request<Body>,
    next: Next,
) -> Response<Body> {
    let arrival_wall_ns = state.clock.wall_time_ns();
    let arrival_mono_ns = state.clock.monotonic_ns();
    let method = request.method().to_string();
    let raw_path = request.uri().path().to_owned();
    let path = percent_decode_str(&raw_path)
        .decode_utf8_lossy()
        .into_owned();
    let query_string = request.uri().query().unwrap_or_default().to_owned();
    let http_version = http_version(request.version());
    let client = request
        .extensions()
        .get::<ConnectInfo<SocketAddr>>()
        .map(|info| info.0);
    let request_headers = headers_to_map(request.headers());

    let response = match content_path_rejection(&state.content_dir, &raw_path).await {
        Some(response) => response,
        None => next.run(request).await,
    };
    let response_ready_ns = state.clock.monotonic_ns();
    let status_code = response.status().as_u16();
    let response_headers = headers_to_map(response.headers());
    let content_type = response_headers
        .get("content-type")
        .cloned()
        .unwrap_or_else(|| "application/octet-stream".into());
    let (parts, body) = response.into_parts();
    let pending = PendingRecord {
        tracker: state.tracker.clone(),
        clock: state.clock.clone(),
        arrival_wall_ns,
        arrival_mono_ns,
        response_ready_ns,
        method,
        path,
        query_string,
        http_version,
        client_host: client
            .map(|address| address.ip().to_string())
            .unwrap_or_default(),
        client_port: client.map(|address| address.port()).unwrap_or_default(),
        request_headers,
        status_code,
        content_type,
        response_headers,
        body_bytes: 0,
        body_chunk_count: 0,
        first_body_ns: None,
        last_body_ns: None,
        finished: false,
    };
    Response::from_parts(parts, Body::new(TrackedBody::new(body, pending)))
}

async fn content_path_rejection(content_dir: &Path, uri_path: &str) -> Option<Response<Body>> {
    let authored = if uri_path == "/content" || uri_path == "/content/" {
        ""
    } else {
        uri_path.strip_prefix("/content/")?
    };
    let decoded = match percent_decode_str(authored).decode_utf8() {
        Ok(decoded) => decoded,
        Err(_) => return Some(plain_response(StatusCode::NOT_FOUND, "Not Found")),
    };
    let mut target = content_dir.to_path_buf();
    for component in Path::new(decoded.as_ref()).components() {
        match component {
            std::path::Component::Normal(component) => target.push(component),
            std::path::Component::CurDir => {}
            std::path::Component::Prefix(_)
            | std::path::Component::RootDir
            | std::path::Component::ParentDir => {
                return Some(plain_response(StatusCode::FORBIDDEN, "Forbidden"));
            }
        }
    }
    match tokio::fs::canonicalize(&target).await {
        Ok(canonical) if !canonical.starts_with(content_dir) => {
            Some(plain_response(StatusCode::FORBIDDEN, "Forbidden"))
        }
        Ok(canonical) if canonical.is_file() => None,
        Ok(_) => Some(plain_response(StatusCode::NOT_FOUND, "Not Found")),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            Some(plain_response(StatusCode::NOT_FOUND, "Not Found"))
        }
        Err(error) if error.kind() == std::io::ErrorKind::PermissionDenied => {
            Some(plain_response(StatusCode::FORBIDDEN, "Forbidden"))
        }
        Err(_) => Some(plain_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            "Internal Server Error",
        )),
    }
}

fn plain_response(status: StatusCode, body: &'static str) -> Response<Body> {
    Response::builder()
        .status(status)
        .header(header::CONTENT_TYPE, "text/plain; charset=utf-8")
        .body(Body::from(body))
        .expect("static content-server response is valid")
}

fn http_version(version: Version) -> String {
    match version {
        Version::HTTP_09 => "0.9".into(),
        Version::HTTP_10 => "1.0".into(),
        Version::HTTP_11 => "1.1".into(),
        Version::HTTP_2 => "2".into(),
        Version::HTTP_3 => "3".into(),
        other => format!("{other:?}"),
    }
}

fn headers_to_map(headers: &HeaderMap) -> BTreeMap<String, String> {
    headers
        .keys()
        .map(|name| {
            let values = headers
                .get_all(name)
                .iter()
                .map(|value| latin1(value.as_bytes()))
                .collect::<Vec<_>>()
                .join(", ");
            (name.as_str().to_ascii_lowercase(), values)
        })
        .collect()
}

fn latin1(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| char::from(*byte)).collect()
}

struct PendingRecord {
    tracker: Arc<RequestTracker>,
    clock: Arc<dyn ContentServerClock>,
    arrival_wall_ns: u64,
    arrival_mono_ns: u64,
    response_ready_ns: u64,
    method: String,
    path: String,
    query_string: String,
    http_version: String,
    client_host: String,
    client_port: u16,
    request_headers: BTreeMap<String, String>,
    status_code: u16,
    content_type: String,
    response_headers: BTreeMap<String, String>,
    body_bytes: u64,
    body_chunk_count: u64,
    first_body_ns: Option<u64>,
    last_body_ns: Option<u64>,
    finished: bool,
}

impl PendingRecord {
    fn observe_body(&mut self, len: usize) {
        if len == 0 {
            return;
        }
        let now = self.clock.monotonic_ns();
        self.first_body_ns.get_or_insert(now);
        self.last_body_ns = Some(now);
        self.body_bytes = self
            .body_bytes
            .saturating_add(u64::try_from(len).unwrap_or(u64::MAX));
        self.body_chunk_count = self.body_chunk_count.saturating_add(1);
    }

    fn finish(&mut self, error: Option<String>) {
        if self.finished {
            return;
        }
        self.finished = true;
        let completed_ns = self.clock.monotonic_ns();
        let first_body_ns = self.first_body_ns.unwrap_or(0);
        let transfer_duration_ns = match (self.first_body_ns, self.last_body_ns) {
            (Some(first), Some(last)) => last.saturating_sub(first),
            _ => 0,
        };
        self.tracker.record(ContentRequestRecord {
            timestamp_ns: self.arrival_wall_ns,
            method: std::mem::take(&mut self.method),
            path: std::mem::take(&mut self.path),
            query_string: std::mem::take(&mut self.query_string),
            http_version: std::mem::take(&mut self.http_version),
            client_host: std::mem::take(&mut self.client_host),
            client_port: self.client_port,
            request_headers: std::mem::take(&mut self.request_headers),
            status_code: self.status_code,
            content_type: std::mem::take(&mut self.content_type),
            response_headers: std::mem::take(&mut self.response_headers),
            body_bytes: self.body_bytes,
            body_chunk_count: self.body_chunk_count,
            latency_ns: completed_ns.saturating_sub(self.arrival_mono_ns),
            time_to_first_byte_ns: self.response_ready_ns.saturating_sub(self.arrival_mono_ns),
            time_to_first_body_byte_ns: first_body_ns.saturating_sub(self.arrival_mono_ns),
            transfer_duration_ns,
            error,
        });
    }
}

struct TrackedBody {
    inner: Pin<Box<Body>>,
    pending: PendingRecord,
    expected_body_bytes: Option<u64>,
}

impl TrackedBody {
    fn new(inner: Body, pending: PendingRecord) -> Self {
        let expected_body_bytes = pending
            .response_headers
            .get("content-length")
            .and_then(|value| value.parse().ok())
            .or_else(|| inner.size_hint().exact());
        let mut pending = pending;
        if inner.is_end_stream() {
            pending.finish(None);
        }
        Self {
            inner: Box::pin(inner),
            pending,
            expected_body_bytes,
        }
    }
}

impl HttpBody for TrackedBody {
    type Data = Bytes;
    type Error = axum::Error;

    fn poll_frame(
        self: Pin<&mut Self>,
        context: &mut TaskContext<'_>,
    ) -> Poll<Option<std::result::Result<Frame<Self::Data>, Self::Error>>> {
        let this = self.get_mut();
        match this.inner.as_mut().poll_frame(context) {
            Poll::Ready(Some(Ok(frame))) => {
                if let Some(data) = frame.data_ref() {
                    this.pending.observe_body(data.len());
                }
                Poll::Ready(Some(Ok(frame)))
            }
            Poll::Ready(Some(Err(error))) => {
                this.pending.finish(Some(error.to_string()));
                Poll::Ready(Some(Err(error)))
            }
            Poll::Ready(None) => {
                this.pending.finish(None);
                Poll::Ready(None)
            }
            Poll::Pending => Poll::Pending,
        }
    }

    fn is_end_stream(&self) -> bool {
        self.inner.is_end_stream()
    }

    fn size_hint(&self) -> SizeHint {
        self.inner.size_hint()
    }
}

impl Drop for TrackedBody {
    fn drop(&mut self) {
        if !self.pending.finished {
            let complete = self.inner.is_end_stream()
                || self
                    .expected_body_bytes
                    .is_some_and(|expected| expected == self.pending.body_bytes);
            let error = (!complete).then(|| "response body dropped before completion".into());
            self.pending.finish(error);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicU64, Ordering};

    use axum::http::HeaderValue;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    use super::*;

    #[derive(Debug)]
    struct ManualClock {
        wall_ns: u64,
        monotonic_ns: AtomicU64,
    }

    impl ContentServerClock for ManualClock {
        fn wall_time_ns(&self) -> u64 {
            self.wall_ns
        }

        fn monotonic_ns(&self) -> u64 {
            self.monotonic_ns.load(Ordering::Relaxed)
        }
    }

    fn pending_record(tracker: Arc<RequestTracker>, clock: Arc<ManualClock>) -> PendingRecord {
        PendingRecord {
            tracker,
            arrival_wall_ns: clock.wall_time_ns(),
            clock,
            arrival_mono_ns: 100,
            response_ready_ns: 120,
            method: "GET".into(),
            path: "/content/test.bin".into(),
            query_string: String::new(),
            http_version: "1.1".into(),
            client_host: "127.0.0.1".into(),
            client_port: 1234,
            request_headers: BTreeMap::new(),
            status_code: 200,
            content_type: "application/octet-stream".into(),
            response_headers: BTreeMap::new(),
            body_bytes: 0,
            body_chunk_count: 0,
            first_body_ns: None,
            last_body_ns: None,
            finished: false,
        }
    }

    #[test]
    fn pending_record_uses_wall_time_only_for_identity_and_monotonic_intervals() {
        let tracker = Arc::new(RequestTracker::new(10));
        let clock = Arc::new(ManualClock {
            wall_ns: 1_000,
            monotonic_ns: AtomicU64::new(150),
        });
        let mut pending = pending_record(tracker.clone(), clock.clone());

        pending.observe_body(3);
        clock.monotonic_ns.store(170, Ordering::Relaxed);
        pending.observe_body(2);
        clock.monotonic_ns.store(200, Ordering::Relaxed);
        pending.finish(None);
        pending.finish(Some("ignored duplicate finish".into()));

        let snapshot = tracker.snapshot();
        assert_eq!(snapshot.total_requests, 1);
        let record = &snapshot.records[0];
        assert_eq!(record.timestamp_ns, 1_000);
        assert_eq!(record.body_bytes, 5);
        assert_eq!(record.body_chunk_count, 2);
        assert_eq!(record.time_to_first_byte_ns, 20);
        assert_eq!(record.time_to_first_body_byte_ns, 50);
        assert_eq!(record.transfer_duration_ns, 20);
        assert_eq!(record.latency_ns, 100);
        assert_eq!(record.error, None);
    }

    #[test]
    fn dropping_an_unconsumed_response_body_records_a_terminal_error() {
        let tracker = Arc::new(RequestTracker::new(10));
        let clock = Arc::new(ManualClock {
            wall_ns: 1_000,
            monotonic_ns: AtomicU64::new(200),
        });
        let tracked = TrackedBody::new(
            Body::from("not-consumed"),
            pending_record(tracker.clone(), clock),
        );

        drop(tracked);

        let snapshot = tracker.snapshot();
        assert_eq!(snapshot.total_requests, 1);
        assert_eq!(snapshot.records[0].body_bytes, 0);
        assert_eq!(
            snapshot.records[0].error.as_deref(),
            Some("response body dropped before completion")
        );
    }

    #[test]
    fn duplicate_headers_are_lowercased_and_joined_with_http_separator() {
        let mut headers = HeaderMap::new();
        headers.append("Set-Cookie", HeaderValue::from_static("a=1"));
        headers.append("set-cookie", HeaderValue::from_static("b=2"));
        headers.insert("X-Custom", HeaderValue::from_bytes(&[0x80]).unwrap());

        let captured = headers_to_map(&headers);

        assert_eq!(captured["set-cookie"], "a=1, b=2");
        assert_eq!(captured["x-custom"], "\u{80}");
    }

    async fn request_with_headers(
        address: SocketAddr,
        target: &str,
        headers: &[(&str, &str)],
    ) -> Vec<u8> {
        let mut stream = tokio::net::TcpStream::connect(address).await.unwrap();
        let extra_headers = headers
            .iter()
            .map(|(name, value)| format!("{name}: {value}\r\n"))
            .collect::<String>();
        stream
            .write_all(
                format!(
                    "GET {target} HTTP/1.1\r\nHost: localhost\r\nUser-Agent: content-test\r\n{extra_headers}Connection: close\r\n\r\n"
                )
                .as_bytes(),
            )
            .await
            .unwrap();
        let mut response = Vec::new();
        stream.read_to_end(&mut response).await.unwrap();
        response
    }

    async fn request(address: SocketAddr, target: &str) -> Vec<u8> {
        request_with_headers(address, target, &[]).await
    }

    async fn head_request(address: SocketAddr, target: &str) -> Vec<u8> {
        let mut stream = tokio::net::TcpStream::connect(address).await.unwrap();
        stream
            .write_all(
                format!("HEAD {target} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")
                    .as_bytes(),
            )
            .await
            .unwrap();
        let mut response = Vec::new();
        stream.read_to_end(&mut response).await.unwrap();
        response
    }

    #[tokio::test]
    async fn serves_health_nested_files_ranges_and_tracks_terminal_bytes() {
        let directory = tempfile::tempdir().unwrap();
        std::fs::create_dir(directory.path().join("images")).unwrap();
        std::fs::create_dir(directory.path().join("video")).unwrap();
        std::fs::write(directory.path().join("images/test.png"), b"0123456789").unwrap();
        std::fs::write(directory.path().join("video/test.webm"), b"webm").unwrap();
        std::fs::write(directory.path().join("opaque.unknown"), b"opaque").unwrap();
        let factory = NativeContentServerFactory::default();
        let mut server = factory
            .start(ContentServerConfig {
                host: "127.0.0.1".into(),
                port: 0,
                content_dir: Some(directory.path().to_owned()),
                max_tracked_records: 10,
            })
            .await
            .unwrap();

        let health = request(server.local_addr(), "/healthz").await;
        assert!(health.starts_with(b"HTTP/1.1 200 OK"));
        assert!(health.ends_with(b"ok"));
        let image = request(server.local_addr(), "/content/images/test.png?raw=1").await;
        assert!(image.starts_with(b"HTTP/1.1 200 OK"));
        assert!(
            image
                .windows(b"content-type: image/png".len())
                .any(|window| { window.eq_ignore_ascii_case(b"content-type: image/png") })
        );
        assert!(image.ends_with(b"0123456789"));
        let video = request(server.local_addr(), "/content/video/test.webm").await;
        assert!(video.starts_with(b"HTTP/1.1 200 OK"));
        assert!(
            video
                .windows(b"content-type: video/webm".len())
                .any(|window| window.eq_ignore_ascii_case(b"content-type: video/webm"))
        );
        let opaque = request(server.local_addr(), "/content/opaque.unknown").await;
        assert!(opaque.starts_with(b"HTTP/1.1 200 OK"));
        assert!(
            opaque
                .windows(b"content-type: application/octet-stream".len())
                .any(|window| {
                    window.eq_ignore_ascii_case(b"content-type: application/octet-stream")
                })
        );
        let range = request_with_headers(
            server.local_addr(),
            "/content/images/test.png",
            &[("Range", "bytes=2-5")],
        )
        .await;
        assert!(range.starts_with(b"HTTP/1.1 206 Partial Content"));
        assert!(range.ends_with(b"2345"));
        let head = head_request(server.local_addr(), "/content/images/test.png").await;
        assert!(head.starts_with(b"HTTP/1.1 200 OK"));
        assert!(head.ends_with(b"\r\n\r\n"));
        let directory_response = request(server.local_addr(), "/content/images").await;
        assert!(directory_response.starts_with(b"HTTP/1.1 404 Not Found"));
        assert!(directory_response.ends_with(b"Not Found"));
        let traversal = request(server.local_addr(), "/content/%2e%2e/Cargo.toml").await;
        assert!(traversal.starts_with(b"HTTP/1.1 403 Forbidden"));
        #[cfg(unix)]
        {
            use std::os::unix::fs::symlink;

            let outside = tempfile::tempdir().unwrap();
            let secret = outside.path().join("secret.txt");
            std::fs::write(&secret, b"must-not-leak").unwrap();
            symlink(&secret, directory.path().join("images/escape.txt")).unwrap();
            let escaped = request(server.local_addr(), "/content/images/escape.txt").await;
            assert!(escaped.starts_with(b"HTTP/1.1 403 Forbidden"));
            assert!(!escaped.ends_with(b"must-not-leak"));
        }
        let missing = request(server.local_addr(), "/content/missing.txt").await;
        assert!(missing.starts_with(b"HTTP/1.1 404 Not Found"));

        let snapshot = server.request_snapshot();
        assert_eq!(snapshot.total_requests, if cfg!(unix) { 10 } else { 9 });
        assert_eq!(
            snapshot.total_bytes_served,
            snapshot
                .records
                .iter()
                .map(|record| record.body_bytes)
                .sum::<u64>()
        );
        let image_record = snapshot
            .records
            .iter()
            .find(|record| record.path == "/content/images/test.png")
            .unwrap();
        assert_eq!(image_record.query_string, "raw=1");
        assert_eq!(image_record.status_code, 200);
        assert_eq!(image_record.content_type, "image/png");
        assert_eq!(image_record.body_bytes, 10);
        assert!(image_record.body_chunk_count >= 1);
        assert_eq!(image_record.error, None);
        assert_eq!(image_record.request_headers["user-agent"], "content-test");
        assert!(image_record.time_to_first_byte_ns <= image_record.latency_ns);
        assert!(image_record.time_to_first_body_byte_ns <= image_record.latency_ns);
        let head_record = snapshot
            .records
            .iter()
            .find(|record| record.method == "HEAD" && record.path == "/content/images/test.png")
            .unwrap();
        assert_eq!(head_record.body_bytes, 0);
        assert_eq!(head_record.body_chunk_count, 0);
        assert_eq!(head_record.error, None);
        let traversal_record = snapshot
            .records
            .iter()
            .find(|record| record.path == "/content/../Cargo.toml")
            .expect("tracked paths use the ASGI-compatible decoded representation");
        assert_eq!(traversal_record.status_code, 403);

        server.shutdown().await.unwrap();
        assert!(!server.status().enabled);
    }

    #[tokio::test]
    async fn missing_directory_fails_before_binding_and_temp_directory_is_owned() {
        let parent = tempfile::tempdir().unwrap();
        let missing = parent.path().join("missing");
        let factory = NativeContentServerFactory::default();
        let error = factory
            .start(ContentServerConfig {
                host: "127.0.0.1".into(),
                port: 0,
                content_dir: Some(missing),
                max_tracked_records: 10,
            })
            .await
            .unwrap_err();
        assert!(error.to_string().contains("resolving content directory"));

        let mut server = factory
            .start(ContentServerConfig {
                host: "127.0.0.1".into(),
                port: 0,
                content_dir: None,
                max_tracked_records: 10,
            })
            .await
            .unwrap();
        let content_dir = server.status().content_dir;
        assert!(content_dir.is_dir());
        server.shutdown().await.unwrap();
        assert!(!content_dir.exists());
    }
}
