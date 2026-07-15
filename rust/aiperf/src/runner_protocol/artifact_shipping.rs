// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Cross-host cellular per-record artifact shipping over HTTP + streaming zstd
//! (Stage E, reopened).
//!
//! # Why this exists
//!
//! Same-host `--cells N` writes each cell's per-record artifacts into a
//! controller-local `temp_root/cell-{id}` dir and concatenates them at finalize
//! ([`crate::runner_protocol::shard_artifacts::concatenate_cell_artifacts`],
//! Stage D). A cross-host (k8s) pod writes to its OWN pod filesystem, unreachable
//! by the controller, so Stage E originally DROPPED those files (the
//! "shared-storage-only" product boundary).
//!
//! That conclusion was too conservative. The Python aiperf reference
//! (`../new-config-kube`) ships per-node records BETWEEN nodes over HTTP with
//! **streaming zstd**, bounded-memory, proven at 250k–1M request scale. This
//! module ports that memory discipline to the Rust runner: a controller-side HTTP
//! upload server and a cell-side streaming-zstd HTTP client. A cell POSTs each of
//! its per-record artifact files (+ `inputs.json`) to the controller with
//! `Content-Encoding: zstd`; the controller streaming-decompresses each to
//! `temp_root/cell-{id}/{file}` via `.part` + atomic rename; then the EXISTING
//! Stage D concat runs unchanged (the files are now controller-local).
//!
//! # The load-bearing property: bounded memory on BOTH ends
//!
//! Neither side ever holds a whole artifact file (or its whole compressed image)
//! in RAM. Ported from the Python reference (`compression.py:131-161`,
//! `results.py:195-224`, `progress_download.py:88-116`):
//!
//! - **Send** ([`FileCompressor`]): read the file in [`CHUNK_SIZE`]-byte chunks
//!   through a `zstd::stream::read::Encoder` and yield each compressed chunk. The
//!   client streams those chunks into the request body over a bounded channel
//!   (backpressure caps in-flight bytes) — improving on the Python leg, which
//!   uploaded the artifact bytes uncompressed.
//! - **Receive** ([`decode_channel_to_file`]): stream the request body frames
//!   into a `zstd::stream::write::Decoder` writing to a `.part` file, then atomic
//!   `rename` to the final path (crash-safe: a partial upload never leaves a
//!   truncated final file). Bounded [`CHUNK_SIZE`] chunks throughout.
//!
//! # Transport is for artifact FILES only
//!
//! Metrics summaries still ship via the velo `StorePartition` (Stage C). This HTTP
//! plane carries ONLY the per-record artifact files (records/raw/CSV/parquet/
//! outputs) and the per-session `inputs.json`. The controller derives the cell→
//! controller address from the same k8s bootstrap/DNS coordinate the velo
//! controller already publishes (see [`crate::runner_protocol::cellular_controller`]).

use std::collections::HashSet;
use std::io::{self, Read, Write};
use std::net::SocketAddr;
use std::path::{Component, Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result, bail, ensure};
use axum::Router;
use axum::body::Body;
use axum::extract::{DefaultBodyLimit, Path as AxumPath, State};
use axum::http::{HeaderMap, StatusCode};
use axum::routing::post;
use bytes::Bytes;
use tokio::sync::{Mutex, Notify, mpsc, oneshot};
use tokio::task::JoinHandle;
use tokio_stream::StreamExt;

/// The read/compress/write chunk size, in bytes — mirrors the Python reference
/// (`CHUNK_SIZE = 65536`). Both the compressor's read buffer and the decoder's
/// per-frame write are bounded by this, so peak per-transfer memory is O(chunk),
/// independent of the artifact file size.
pub const CHUNK_SIZE: usize = 65536;

/// The zstd compression level — mirrors the Python reference (`ZSTD_LEVEL = 3`).
/// Level 3 is zstd's default: a good ratio/speed balance for the large,
/// line-oriented JSONL/CSV artifacts this ships.
pub const ZSTD_LEVEL: i32 = 3;

/// The `Content-Encoding` value a cell sets when it streams a zstd-compressed
/// artifact body; the controller streaming-decompresses only when it matches.
pub const ZSTD_CONTENT_ENCODING: &str = "zstd";

// -- streaming zstd core ----------------------------------------------------------

/// A bounded streaming compressor over one artifact file: each
/// [`next_chunk`](Self::next_chunk) yields at most [`CHUNK_SIZE`] bytes of the
/// zstd frame, reading the source incrementally so the whole file is never
/// resident. Ported from the Python `ZstdCompressor(level=3).compressobj()` chunk
/// loop (`compression.py:131-161`).
pub struct FileCompressor {
    encoder: zstd::stream::read::Encoder<'static, io::BufReader<std::fs::File>>,
    buf: Box<[u8]>,
}

impl FileCompressor {
    /// Open `path` for streaming compression at [`ZSTD_LEVEL`].
    pub fn open(path: &Path) -> io::Result<Self> {
        let file = std::fs::File::open(path)?;
        let encoder = zstd::stream::read::Encoder::new(file, ZSTD_LEVEL)?;
        Ok(Self {
            encoder,
            buf: vec![0_u8; CHUNK_SIZE].into_boxed_slice(),
        })
    }

    /// The next compressed chunk (at most [`CHUNK_SIZE`] bytes), or `None` at the
    /// end of the frame. A short read mid-stream is normal — callers loop until
    /// `None`.
    pub fn next_chunk(&mut self) -> io::Result<Option<Vec<u8>>> {
        let n = self.encoder.read(&mut self.buf)?;
        Ok((n > 0).then(|| self.buf[..n].to_vec()))
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
/// onto the final path by [`finish`](Self::finish). Ported from the Python
/// `ZstdDecompressor().decompressobj()` → `.part` → `os.replace` leg
/// (`progress_download.py:88-116`): a crashed/partial transfer leaves a `.part`
/// file, never a truncated final artifact.
pub struct DecompressToFile {
    decoder: zstd::stream::write::Decoder<'static, io::BufWriter<std::fs::File>>,
    part_path: PathBuf,
    final_path: PathBuf,
}

impl DecompressToFile {
    /// Create the `.part` staging file (and any missing parent dirs) and wrap it
    /// in a streaming zstd write-decoder.
    pub fn create(final_path: &Path) -> io::Result<Self> {
        if let Some(parent) = final_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let part_path = part_path_for(final_path);
        let file = std::fs::File::create(&part_path)?;
        let decoder = zstd::stream::write::Decoder::new(io::BufWriter::new(file))?;
        Ok(Self {
            decoder,
            part_path,
            final_path: final_path.to_path_buf(),
        })
    }

    /// Feed one compressed chunk; its decompressed bytes stream straight to the
    /// `.part` file (nothing buffered whole).
    pub fn write_chunk(&mut self, compressed: &[u8]) -> io::Result<()> {
        self.decoder.write_all(compressed)
    }

    /// Flush the decoder, fsync the `.part` file, and atomically rename it onto
    /// the final path.
    pub fn finish(mut self) -> io::Result<()> {
        self.decoder.flush()?;
        let writer = self.decoder.into_inner();
        let file = writer
            .into_inner()
            .map_err(std::io::IntoInnerError::into_error)?;
        file.sync_all()?;
        std::fs::rename(&self.part_path, &self.final_path)?;
        Ok(())
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
        if let Some(parent) = final_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let part_path = part_path_for(final_path);
        let file = std::fs::File::create(&part_path)?;
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
        file.sync_all()?;
        std::fs::rename(&self.part_path, &self.final_path)?;
        Ok(())
    }
}

/// Drain `rx` (compressed-or-plain chunks the async handler forwards from the
/// request body) into `dest`, streaming-decompressing when `zstd` is set, then
/// atomic-rename. Runs on a blocking task so the file/zstd work never stalls the
/// async runtime. Bounded memory: one [`CHUNK_SIZE`] chunk at a time.
fn decode_channel_to_file(
    mut rx: mpsc::Receiver<Bytes>,
    dest: &Path,
    zstd: bool,
) -> io::Result<()> {
    if zstd {
        let mut sink = DecompressToFile::create(dest)?;
        while let Some(chunk) = rx.blocking_recv() {
            sink.write_chunk(&chunk)?;
        }
        sink.finish()
    } else {
        let mut sink = PlainToFile::create(dest)?;
        while let Some(chunk) = rx.blocking_recv() {
            sink.write_chunk(&chunk)?;
        }
        sink.finish()
    }
}

// -- path / filename validation ---------------------------------------------------

/// Validate a client-supplied relative artifact path against a cell-scoped
/// allowlist, rejecting anything that is not exactly one of the run's known
/// artifact relative paths. Fails closed on:
/// - an absolute path (`/etc/passwd`),
/// - any non-`Normal` component (`..`, `.`, a Windows prefix, a root),
/// - a path not present in `allowed` verbatim.
///
/// The allowlist is the exact set of relative artifact paths the controller
/// derived from `cfg.artifacts` (records/raw/CSV/parquet/outputs + `inputs.json`),
/// so a cell can only ever land bytes at a known per-record artifact location
/// inside its own `cell-{id}` dir — never traverse out of it.
fn validate_artifact_relpath(rel: &str, allowed: &HashSet<String>) -> Result<PathBuf> {
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

// -- controller-side upload server ------------------------------------------------

/// Shared state for the controller's artifact upload server: where cell files land
/// (`temp_root/cell-{id}/{rel}`), the per-run allowlist of relative artifact paths,
/// and the set of cells that have signaled upload completion.
struct UploadState {
    temp_root: PathBuf,
    allowed: HashSet<String>,
    done: Mutex<HashSet<u32>>,
    done_notify: Notify,
}

/// The controller-side HTTP server cells POST their zstd-compressed artifact files
/// to. Bind, per-cell dir landing, streaming decode, and a cell-completion barrier
/// (`/cell/{id}/done`) the controller awaits before running the Stage D concat.
pub struct ArtifactUploadServer {
    local_addr: SocketAddr,
    state: Arc<UploadState>,
    shutdown_tx: Option<oneshot::Sender<()>>,
    task: Option<JoinHandle<()>>,
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
    ) -> Result<Self> {
        let state = Arc::new(UploadState {
            temp_root,
            allowed,
            done: Mutex::new(HashSet::new()),
            done_notify: Notify::new(),
        });
        let app = Router::new()
            .route("/cell/{cell_id}/artifact/{*file}", post(upload_artifact))
            .route("/cell/{cell_id}/done", post(cell_done))
            // The body is streamed frame by frame (bounded); lift the default 2 MB
            // request-body cap so a large records.jsonl upload is not truncated.
            .layer(DefaultBodyLimit::disable())
            .with_state(state.clone());
        let listener = tokio::net::TcpListener::bind(bind)
            .await
            .with_context(|| format!("binding artifact upload server to {bind}"))?;
        let local_addr = listener
            .local_addr()
            .context("reading artifact upload server address")?;
        let (shutdown_tx, shutdown_rx) = oneshot::channel();
        let task = tokio::spawn(async move {
            let _ = axum::serve(listener, app)
                .with_graceful_shutdown(async move {
                    let _ = shutdown_rx.await;
                })
                .await;
        });
        Ok(Self {
            local_addr,
            state,
            shutdown_tx: Some(shutdown_tx),
            task: Some(task),
        })
    }

    /// The bound address (host + OS-assigned port when bound to `:0`).
    pub fn local_addr(&self) -> SocketAddr {
        self.local_addr
    }

    /// Wait until `cell_count` distinct cells have signaled `/done`, or `timeout`
    /// elapses. This is the controller's artifact barrier: every cell POSTs all its
    /// files THEN posts `/done`, so once all cells are done the per-cell dirs are
    /// complete and the Stage D concat can run.
    pub async fn wait_for_cells(
        &self,
        cell_count: u32,
        timeout: std::time::Duration,
    ) -> Result<()> {
        let wait = async {
            loop {
                {
                    let done = self.state.done.lock().await;
                    if done.len() >= cell_count as usize {
                        return;
                    }
                }
                self.state.done_notify.notified().await;
            }
        };
        tokio::time::timeout(timeout, wait).await.map_err(|_| {
            let done = self.state.done.try_lock().map(|d| d.len()).unwrap_or(0);
            anyhow::anyhow!(
                "artifact upload barrier timed out with {done} of {cell_count} cells done"
            )
        })
    }

    /// Stop serving and join the server task.
    pub async fn shutdown(mut self) {
        if let Some(tx) = self.shutdown_tx.take() {
            let _ = tx.send(());
        }
        if let Some(task) = self.task.take() {
            let _ = task.await;
        }
    }
}

impl Drop for ArtifactUploadServer {
    fn drop(&mut self) {
        if let Some(tx) = self.shutdown_tx.take() {
            let _ = tx.send(());
        }
    }
}

/// `POST /cell/{cell_id}/artifact/{*file}` — stream the (zstd) request body into
/// `temp_root/cell-{cell_id}/{file}` via `.part` + atomic rename. Path-validated
/// against the run allowlist; bounded memory throughout.
async fn upload_artifact(
    State(state): State<Arc<UploadState>>,
    AxumPath((cell_id, file)): AxumPath<(u32, String)>,
    headers: HeaderMap,
    body: Body,
) -> Result<StatusCode, (StatusCode, String)> {
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
    let (tx, rx) = mpsc::channel::<Bytes>(4);
    let writer = tokio::task::spawn_blocking(move || decode_channel_to_file(rx, &dest, zstd));

    let mut stream = body.into_data_stream();
    while let Some(frame) = stream.next().await {
        let bytes = frame.map_err(|error| {
            (
                StatusCode::BAD_REQUEST,
                format!("reading upload body: {error}"),
            )
        })?;
        if tx.send(bytes).await.is_err() {
            // The writer task failed and dropped rx; surface its error below.
            break;
        }
    }
    drop(tx);
    match writer.await {
        Ok(Ok(())) => Ok(StatusCode::OK),
        Ok(Err(error)) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("writing cell {cell_id} artifact {file:?}: {error}"),
        )),
        Err(join) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("artifact writer task for cell {cell_id} panicked: {join}"),
        )),
    }
}

/// `POST /cell/{cell_id}/done` — record that this cell has uploaded all its files
/// and wake the controller's [`wait_for_cells`](ArtifactUploadServer::wait_for_cells)
/// barrier.
async fn cell_done(
    State(state): State<Arc<UploadState>>,
    AxumPath(cell_id): AxumPath<u32>,
) -> StatusCode {
    state.done.lock().await.insert(cell_id);
    state.done_notify.notify_waiters();
    StatusCode::OK
}

// -- cell-side HTTP client --------------------------------------------------------

/// Ship one cell's per-record artifact files (+ `inputs.json`) to the controller
/// over HTTP with streaming zstd, then POST the per-cell `/done` marker.
///
/// `authority` is the controller's artifact `host:port` (DNS-resolved, so a k8s
/// service name works); `cell_dir` is this cell's own artifact dir; `relatives`
/// is the set of relative artifact paths to ship (only those that exist on disk
/// are sent — a metrics-only or lazy-CSV run legitimately omits some). Bounded
/// memory: each file streams through a [`FileCompressor`] over a bounded channel.
pub async fn ship_cell_artifacts(
    authority: &str,
    cell_id: u32,
    cell_dir: &Path,
    relatives: &[String],
) -> Result<()> {
    for rel in relatives {
        let src = cell_dir.join(rel);
        if !src.exists() {
            continue;
        }
        upload_one(authority, cell_id, rel, &src)
            .await
            .with_context(|| format!("cell {cell_id} shipping artifact {rel:?}"))?;
    }
    post_done(authority, cell_id)
        .await
        .with_context(|| format!("cell {cell_id} posting artifact done"))?;
    Ok(())
}

/// Stream one file to `POST /cell/{cell_id}/artifact/{rel}` with
/// `Content-Encoding: zstd`. Compression runs on a blocking task feeding a bounded
/// channel; the request body streams from that channel (whole-file never resident).
async fn upload_one(authority: &str, cell_id: u32, rel: &str, src: &Path) -> Result<()> {
    use http_body_util::StreamBody;
    use hyper::body::Frame;
    use tokio_stream::wrappers::ReceiverStream;

    // Producer: chunked read + zstd on a blocking task → bounded channel.
    let (tx, rx) = mpsc::channel::<Bytes>(4);
    let path = src.to_path_buf();
    let producer = tokio::task::spawn_blocking(move || -> io::Result<()> {
        let mut compressor = FileCompressor::open(&path)?;
        while let Some(chunk) = compressor.next_chunk()? {
            if tx.blocking_send(Bytes::from(chunk)).is_err() {
                break; // receiver dropped (request failed); stop early
            }
        }
        Ok(())
    });

    let body_stream =
        ReceiverStream::new(rx).map(|bytes| Ok::<_, std::convert::Infallible>(Frame::data(bytes)));
    let body = StreamBody::new(body_stream);

    let request = hyper::Request::builder()
        .method("POST")
        .uri(format!("/cell/{cell_id}/artifact/{rel}"))
        .header(hyper::header::HOST, authority)
        .header(hyper::header::CONTENT_ENCODING, ZSTD_CONTENT_ENCODING)
        .body(body)
        .context("building artifact upload request")?;

    let status = send_request(authority, request).await?;
    // Join the producer so a compression/IO error is not silently swallowed.
    match producer.await {
        Ok(Ok(())) => {}
        Ok(Err(error)) => return Err(error).context("streaming compress"),
        Err(join) => bail!("compression task panicked: {join}"),
    }
    ensure!(
        status.is_success(),
        "artifact upload returned HTTP {status}"
    );
    Ok(())
}

/// POST the empty-bodied `/cell/{cell_id}/done` completion marker.
async fn post_done(authority: &str, cell_id: u32) -> Result<()> {
    use http_body_util::Empty;

    let request = hyper::Request::builder()
        .method("POST")
        .uri(format!("/cell/{cell_id}/done"))
        .header(hyper::header::HOST, authority)
        .body(Empty::<Bytes>::new())
        .context("building done request")?;
    let status = send_request(authority, request).await?;
    ensure!(status.is_success(), "done marker returned HTTP {status}");
    Ok(())
}

/// Open an HTTP/1.1 connection to `authority` (DNS-resolved), send `request`, and
/// return its status after draining the response. Uses a raw hyper client
/// connection (no `hyper-util` legacy `Client` feature needed).
async fn send_request<B>(authority: &str, request: hyper::Request<B>) -> Result<StatusCode>
where
    B: hyper::body::Body + Send + 'static,
    B::Data: Send,
    B::Error: Into<Box<dyn std::error::Error + Send + Sync>>,
{
    use http_body_util::BodyExt;

    let stream = tokio::net::TcpStream::connect(authority)
        .await
        .with_context(|| format!("connecting to artifact server {authority}"))?;
    let io = hyper_util::rt::TokioIo::new(stream);
    let (mut sender, conn) = hyper::client::conn::http1::handshake(io)
        .await
        .context("artifact HTTP handshake")?;
    tokio::spawn(async move {
        let _ = conn.await;
    });
    let response = sender
        .send_request(request)
        .await
        .context("sending artifact request")?;
    let status = response.status();
    // Drain the (small) response body so the connection closes cleanly.
    let _ = response.into_body().collect().await;
    Ok(status)
}

// -- allowlist / relative-path derivation -----------------------------------------

/// The relative artifact paths a run may ship over HTTP, derived from its
/// `ArtifactSpec` — every per-record file (records/raw/CSV/parquet/outputs) plus
/// the per-session `inputs.json`. Both the controller's server allowlist and the
/// cell's client shipping list come from this single function, so they can never
/// disagree on which files cross the wire.
pub fn shippable_relatives(
    artifacts: &crate::runner_protocol::protocol::ArtifactSpec,
) -> Vec<String> {
    let mut relatives = Vec::new();
    for path in [
        artifacts.records_path.as_ref(),
        artifacts.raw_path.as_ref(),
        artifacts.records_csv_path.as_ref(),
        artifacts.records_parquet_path.as_ref(),
        artifacts.outputs_path.as_ref(),
        artifacts.inputs_path.as_ref(),
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

    /// Deterministic pseudo-random bytes spanning several [`CHUNK_SIZE`] windows,
    /// so the round-trip exercises multi-chunk streaming (not a single read).
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

    /// Collect a file's whole zstd frame as bounded chunks, asserting each chunk
    /// respects [`CHUNK_SIZE`] (the memory property) — the streaming compressor
    /// never emits an unbounded buffer.
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
        // A file larger than several chunks so both the compressor and the decoder
        // stream across many bounded windows.
        let dir = tempfile::tempdir().unwrap();
        let src = dir.path().join("source.bin");
        let payload = sample_bytes(CHUNK_SIZE * 5 + 12345);
        std::fs::write(&src, &payload).unwrap();

        let chunks = compress_to_chunks(&src);
        assert!(chunks.len() >= 2, "payload must span multiple chunks");

        // Decompress the chunks into a .part-staged final file.
        let dest = dir.path().join("nested").join("dest.bin");
        let mut sink = DecompressToFile::create(&dest).unwrap();
        // The .part exists mid-transfer; the final does not yet.
        assert!(
            part_path_for(&dest).exists(),
            ".part staged during transfer"
        );
        assert!(!dest.exists(), "final file not created until finish()");
        for chunk in &chunks {
            sink.write_chunk(chunk).unwrap();
        }
        sink.finish().unwrap();

        // Byte-for-byte round trip; .part cleaned up by the rename.
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

        // Accept an exact allowed relative path.
        assert!(validate_artifact_relpath("profile_export.jsonl", &allowed).is_ok());
        assert!(validate_artifact_relpath("inputs.json", &allowed).is_ok());

        // Reject traversal, absolute, and non-allowlisted names.
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
        // The non-zstd receive leg (a client that omitted Content-Encoding) still
        // stages via .part and atomic-renames.
        let dir = tempfile::tempdir().unwrap();
        let dest = dir.path().join("plain.bin");
        let payload = sample_bytes(1000);
        let mut sink = PlainToFile::create(&dest).unwrap();
        sink.write_chunk(&payload[..400]).unwrap();
        sink.write_chunk(&payload[400..]).unwrap();
        sink.finish().unwrap();
        assert_eq!(std::fs::read(&dest).unwrap(), payload);
    }

    /// End-to-end in-process integration (no k8s): stand up the controller upload
    /// server on localhost, have two "cells" ship real per-record JSONL artifacts
    /// with streaming zstd, then assert (1) the controller's per-cell dirs are
    /// byte-identical to the source files, and (2) the existing Stage D concat over
    /// the uploaded dirs equals the batch concat over the SOURCE union (set parity).
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn in_process_ship_then_concat_matches_batch_over_union() {
        use crate::runner_protocol::protocol::ArtifactSpec;
        use crate::runner_protocol::shard_artifacts::concatenate_cell_artifacts;

        let root = tempfile::tempdir().unwrap();
        // Two source "cell" dirs (as if each cell wrote to its OWN pod fs).
        let cell_count = 2u32;
        let source_root = root.path().join("sources");
        let records_rel = "profile_export.jsonl";
        let inputs_rel = "inputs.json";
        // Cell 0 and cell 1 write disjoint record rows + an identical inputs.json.
        let source_dirs: Vec<PathBuf> = (0..cell_count)
            .map(|id| source_root.join(format!("cell-{id}")))
            .collect();
        for (id, dir) in source_dirs.iter().enumerate() {
            std::fs::create_dir_all(dir).unwrap();
            let mut records = std::fs::File::create(dir.join(records_rel)).unwrap();
            // A few JSONL rows, larger than a chunk to exercise multi-chunk streaming.
            for row in 0..2000 {
                writeln!(
                    records,
                    "{{\"cell\":{id},\"row\":{row},\"pad\":\"{}\"}}",
                    "x".repeat(40)
                )
                .unwrap();
            }
            // Identical inputs.json in every cell (per-session full-dataset doc).
            std::fs::write(
                dir.join(inputs_rel),
                b"{\"schema\":\"inputs\",\"data\":[1,2,3]}",
            )
            .unwrap();
        }

        // Controller: allowlist + landing root.
        let artifacts = ArtifactSpec {
            records_path: Some(PathBuf::from(records_rel)),
            raw_path: None,
            records_csv_path: None,
            records_parquet_path: None,
            outputs_path: None,
            inputs_path: Some(PathBuf::from(inputs_rel)),
            trace: false,
        };
        let relatives = shippable_relatives(&artifacts);
        let allowed: HashSet<String> = relatives.iter().cloned().collect();
        let controller_root = root.path().join("controller-temp");
        std::fs::create_dir_all(&controller_root).unwrap();

        let server = ArtifactUploadServer::start(
            "127.0.0.1:0".parse().unwrap(),
            controller_root.clone(),
            allowed,
        )
        .await
        .unwrap();
        let authority = server.local_addr().to_string();

        // Every cell ships its files over HTTP + zstd.
        for (id, dir) in source_dirs.iter().enumerate() {
            ship_cell_artifacts(&authority, id as u32, dir, &relatives)
                .await
                .unwrap();
        }
        // The controller's barrier releases once all cells posted /done.
        server
            .wait_for_cells(cell_count, std::time::Duration::from_secs(10))
            .await
            .unwrap();

        // (1) Per-cell landed files are byte-identical to the sources.
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

        // (2) Stage D concat over the UPLOADED dirs == concat over the SOURCE dirs.
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
        // 2 cells * 2000 rows == 4000 merged rows.
        assert_eq!(
            std::fs::read_to_string(merged_from_uploaded.join(records_rel))
                .unwrap()
                .lines()
                .count(),
            4000,
        );

        server.shutdown().await;
    }

    /// Shipping an unknown/traversal path is rejected by the server (defense in
    /// depth): even a malicious cell cannot land bytes outside its allowlisted set.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn server_rejects_unallowed_upload() {
        let root = tempfile::tempdir().unwrap();
        let mut allowed = HashSet::new();
        allowed.insert("profile_export.jsonl".to_owned());
        let server = ArtifactUploadServer::start(
            "127.0.0.1:0".parse().unwrap(),
            root.path().to_path_buf(),
            allowed,
        )
        .await
        .unwrap();
        let authority = server.local_addr().to_string();
        let src = root.path().join("payload.bin");
        std::fs::write(&src, b"hello").unwrap();

        // Not in the allowlist → the upload fails.
        let unknown = upload_one(&authority, 0, "secret.parquet", &src).await;
        assert!(unknown.is_err(), "unallowed artifact must be rejected");
        server.shutdown().await;
    }
}
