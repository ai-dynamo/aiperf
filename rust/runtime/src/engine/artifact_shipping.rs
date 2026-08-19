// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Bounded-memory cross-host artifact shipping over HTTP and streaming zstd.
//!
//! - **Send** ([`FileCompressor`]): read the file in [`CHUNK_SIZE`]-byte chunks
//!   through a `zstd::stream::read::Encoder` and yield each compressed chunk. The
//!   client streams those chunks into the request body over a bounded channel
//!   so backpressure caps in-flight bytes.
//! - **Receive** (`decode_channel_to_file`): stream the request body frames
//!   into a `zstd::stream::write::Decoder` writing to a `.part` file, then atomic
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
use axum::extract::{DefaultBodyLimit, Path as AxumPath, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::Response;
use axum::routing::{get, post};
use bytes::Bytes;
use tokio::sync::{mpsc, oneshot, watch};
use tokio::task::JoinHandle;
use tokio_stream::StreamExt;

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
    decoder: zstd::stream::write::Decoder<'static, io::BufWriter<std::fs::File>>,
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
    /// Create the `.part` staging file and wrap it in a streaming zstd
    /// write-decoder.
    pub fn create(final_path: &Path) -> io::Result<Self> {
        let (file, part_path) = create_part_file(final_path)?;
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

    /// Flush the decoder, then fsync and atomically commit the `.part` file.
    pub fn finish(mut self) -> io::Result<()> {
        self.decoder.flush()?;
        let writer = self.decoder.into_inner();
        let file = writer
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
    /// The non-synthetic dataset SOURCE files the controller may serve to cells
    /// over `GET /dataset/{name}`, keyed by the file name a cell
    /// requests. A cross-host cell cannot read the controller-local dataset path,
    /// so the controller streams the source over the same HTTP + zstd plane the
    /// per-record artifact uploads use; the cell then recompiles it locally. Empty
    /// for a synthetic run (each cell regenerates the dataset from the shared seed)
    /// and for a same-host run (cells read the controller-local path directly).
    datasets: HashMap<String, PathBuf>,
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
        Self::start_with_datasets(bind, temp_root, allowed, HashMap::new()).await
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
        datasets: HashMap<String, PathBuf>,
    ) -> Result<Self> {
        Self::start_with_dataset_plan(bind, temp_root, allowed, datasets, None).await
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
        datasets: HashMap<String, PathBuf>,
        manifest: Option<DatasetManifest>,
    ) -> Result<Self> {
        let state = Arc::new(UploadState {
            temp_root,
            allowed,
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

    // Count the on-wire (post-compression) body bytes so the completion log below is
    // an unambiguous observable that this artifact really crossed the HTTP socket —
    // and, with `content_encoding=zstd`, that it did so compressed, not via a
    // same-host shared filesystem. A multi-process cellular test greps the
    // controller's stderr for one such line per cell × file.
    let mut received_bytes: u64 = 0;
    let mut stream = body.into_data_stream();
    while let Some(frame) = stream.next().await {
        let bytes = frame.map_err(|error| {
            (
                StatusCode::BAD_REQUEST,
                format!("reading upload body: {error}"),
            )
        })?;
        received_bytes += bytes.len() as u64;
        if tx.send(bytes).await.is_err() {
            // The writer task failed and dropped rx; surface its error below.
            break;
        }
    }
    drop(tx);
    match writer.await {
        Ok(Ok(())) => {
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
    let path = state
        .datasets
        .get(&name)
        .cloned()
        .ok_or_else(|| (StatusCode::NOT_FOUND, format!("no dataset source {name:?}")))?;

    // Blocking chunked read + zstd on a blocking task → bounded channel; the axum
    // response body streams from that channel (whole-file never resident). A read
    // error mid-stream is forwarded as a stream error, truncating the body so the
    // cell's decoder fails rather than landing a partial file.
    let (tx, rx) = mpsc::channel::<Result<Bytes, io::Error>>(4);
    tokio::task::spawn_blocking(move || match FileCompressor::open(&path) {
        Ok(mut compressor) => loop {
            match compressor.next_chunk() {
                Ok(Some(chunk)) => {
                    if tx.blocking_send(Ok(Bytes::from(chunk))).is_err() {
                        break; // receiver dropped (client disconnected)
                    }
                }
                Ok(None) => break,
                Err(error) => {
                    let _ = tx.blocking_send(Err(error));
                    break;
                }
            }
        },
        Err(error) => {
            let _ = tx.blocking_send(Err(error));
        }
    });

    // Use the shared artifact target so this event can be enabled independently.
    tracing::info!(
        target: "aiperf_cellular_artifact",
        dataset = %name,
        content_encoding = ZSTD_CONTENT_ENCODING,
        "served dataset source over HTTP"
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
/// connection.
async fn send_request<B>(authority: &str, request: hyper::Request<B>) -> Result<StatusCode>
where
    B: hyper::body::Body + Send + 'static,
    B::Data: Send,
    B::Error: Into<Box<dyn std::error::Error + Send + Sync>>,
{
    use http_body_util::BodyExt;

    let response = connect_and_send(authority, request, "artifact").await?;
    let status = response.status();
    // Drain the (small) response body so the connection closes cleanly.
    let _ = response.into_body().collect().await;
    Ok(status)
}

/// Open an HTTP/1.1 connection to `authority` (DNS-resolved), send `request`, and
/// return the raw response (the connection driver is spawned; the caller consumes
/// the body). `label` names the server role in connection/handshake errors
/// (`"artifact"` vs `"dataset"`). Shared by the status-only [`send_request`] upload
/// leg and the body-streaming [`fetch_dataset_to_file`] download leg.
async fn connect_and_send<B>(
    authority: &str,
    request: hyper::Request<B>,
    label: &str,
) -> Result<hyper::Response<hyper::body::Incoming>>
where
    B: hyper::body::Body + Send + 'static,
    B::Data: Send,
    B::Error: Into<Box<dyn std::error::Error + Send + Sync>>,
{
    let stream = tokio::net::TcpStream::connect(authority)
        .await
        .with_context(|| format!("connecting to {label} server {authority}"))?;
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

/// Fetch a controller dataset source over `GET /dataset/{name}` and
/// stream it to `dest`, decompressing when the response is `Content-Encoding:
/// zstd`. Bounded memory: response frames flow through a bounded channel into a
/// blocking streaming decode (`.part` + atomic rename), so a crashed transfer
/// never leaves a truncated final file — the same discipline as the upload leg.
///
/// `authority` is the controller's artifact `host:port`; a same-host cell never
/// calls this (it reads the controller-local path directly).
pub async fn fetch_dataset_to_file(authority: &str, name: &str, dest: &Path) -> Result<()> {
    use http_body_util::{BodyExt, Empty};

    let request = hyper::Request::builder()
        .method("GET")
        .uri(dataset_request_path(name))
        .header(hyper::header::HOST, authority)
        .body(Empty::<Bytes>::new())
        .context("building dataset fetch request")?;

    let response = connect_and_send(authority, request, "dataset").await?;
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
    let (tx, rx) = mpsc::channel::<Bytes>(4);
    let dest_buf = dest.to_path_buf();
    let writer = tokio::task::spawn_blocking(move || decode_channel_to_file(rx, &dest_buf, zstd));

    let mut body = response.into_body();
    while let Some(frame) = body.frame().await {
        let frame = frame.map_err(|error| anyhow::anyhow!("reading dataset body: {error}"))?;
        if let Ok(data) = frame.into_data()
            && tx.send(data).await.is_err()
        {
            break; // writer task failed and dropped rx; surface its error below
        }
    }
    drop(tx);
    match writer.await {
        Ok(Ok(())) => Ok(()),
        Ok(Err(error)) => Err(error).with_context(|| format!("writing dataset {name:?}")),
        Err(join) => bail!("dataset writer task panicked: {join}"),
    }
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
pub async fn fetch_dataset_manifest(authority: &str) -> Result<DatasetManifest> {
    use http_body_util::{BodyExt, Empty};

    let request = hyper::Request::builder()
        .method("GET")
        .uri("/dataset-manifest")
        .header(hyper::header::HOST, authority)
        .body(Empty::<Bytes>::new())
        .context("building dataset manifest request")?;

    let stream = tokio::net::TcpStream::connect(authority)
        .await
        .with_context(|| format!("connecting to dataset server {authority}"))?;
    let io = hyper_util::rt::TokioIo::new(stream);
    let (mut sender, conn) = hyper::client::conn::http1::handshake(io)
        .await
        .context("dataset manifest HTTP handshake")?;
    tokio::spawn(async move {
        let _ = conn.await;
    });
    let response = sender
        .send_request(request)
        .await
        .context("sending dataset manifest request")?;
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
pub async fn reconstruct_shipped_dataset(
    authority: &str,
    manifest: &DatasetManifest,
    dest_dir: &Path,
) -> Result<PathBuf> {
    validate_dataset_manifest(manifest)?;
    for name in &manifest.files {
        let dest = prepare_dataset_destination(dest_dir, Path::new(name))?;
        fetch_dataset_to_file(authority, name, &dest)
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
/// `ArtifactSpec` — every per-record file (records/raw/CSV/parquet/outputs) plus
/// the per-session `inputs.json`. Both the controller's server allowlist and the
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

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn wait_for_cells_returns_when_completed_before_waiter_registers() {
        use std::time::{Duration, Instant};

        let root = tempfile::tempdir().unwrap();
        let server = ArtifactUploadServer::start(
            "127.0.0.1:0".parse().unwrap(),
            root.path().to_path_buf(),
            HashSet::new(),
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

        server.shutdown().await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn wait_for_cells_times_out_and_names_missing_cells() {
        use std::time::Duration;

        let root = tempfile::tempdir().unwrap();
        let server = ArtifactUploadServer::start(
            "127.0.0.1:0".parse().unwrap(),
            root.path().to_path_buf(),
            HashSet::new(),
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

        server.shutdown().await;
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
        )
        .await
        .unwrap();
        let authority = server.local_addr().to_string();

        for (id, dir) in source_dirs.iter().enumerate() {
            ship_cell_artifacts(&authority, id as u32, dir, &relatives)
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

        server.shutdown().await;
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
            datasets.insert(shard.to_owned(), path);
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
        )
        .await
        .unwrap();
        let authority = server.local_addr().to_string();

        let fetched = fetch_dataset_manifest(&authority).await.unwrap();
        assert_eq!(fetched, manifest, "manifest round-trips over HTTP");

        let cell_dir = dir.path().join("cell").join("dataset");
        let rewritten = reconstruct_shipped_dataset(&authority, &fetched, &cell_dir)
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

        server.shutdown().await;
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
            datasets.insert(shard.to_owned(), path);
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
        )
        .await
        .unwrap();
        let authority = server.local_addr().to_string();
        let cell_dir = dir.path().join("cell").join("dataset");
        let rewritten = reconstruct_shipped_dataset(&authority, &manifest, &cell_dir)
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
        server.shutdown().await;
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
            ("recordings/trace.json".to_owned(), recording),
            ("benchmark/pinchbench/assets/input.txt".to_owned(), asset),
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
        )
        .await
        .unwrap();
        let landed = dir.path().join("landed");
        let rewritten =
            reconstruct_shipped_dataset(&server.local_addr().to_string(), &manifest, &landed)
                .await
                .unwrap();

        assert_eq!(rewritten, landed.join("recordings/trace.json"));
        assert_eq!(
            std::fs::read(landed.join("benchmark/pinchbench/assets/input.txt")).unwrap(),
            b"workspace asset"
        );
        server.shutdown().await;
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
                ("selected/main.jsonl".to_owned(), main.clone()),
                (
                    "selected/main/subagents/agent-aaa.jsonl".to_owned(),
                    subagent.clone(),
                ),
            ]),
            Some(manifest.clone()),
        )
        .await
        .unwrap();
        let landed = dir.path().join("landed");
        let rewritten =
            reconstruct_shipped_dataset(&server.local_addr().to_string(), &manifest, &landed)
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
        server.shutdown().await;
    }

    #[tokio::test]
    async fn agent_session_exact_set_rejects_duplicate_paths_before_fetching() {
        let dest = tempfile::tempdir().unwrap();
        let manifest = DatasetManifest {
            kind: "agent_session_set".to_owned(),
            base_name: "main.jsonl".to_owned(),
            files: vec!["main.jsonl".to_owned(), "main.jsonl".to_owned()],
        };
        let error = reconstruct_shipped_dataset("127.0.0.1:1", &manifest, dest.path())
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
            reconstruct_shipped_dataset("127.0.0.1:1", &manifest, dest.path())
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
        )
        .await
        .unwrap();
        let authority = server.local_addr().to_string();
        let src = root.path().join("payload.bin");
        std::fs::write(&src, b"hello").unwrap();

        let unknown = upload_one(&authority, 0, "secret.parquet", &src).await;
        assert!(unknown.is_err(), "unallowed artifact must be rejected");
        server.shutdown().await;
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
        datasets.insert("prompts.jsonl".to_owned(), src.clone());
        let server = ArtifactUploadServer::start_with_datasets(
            "127.0.0.1:0".parse().unwrap(),
            dir.path().join("controller-temp"),
            HashSet::new(),
            datasets,
        )
        .await
        .unwrap();
        let authority = server.local_addr().to_string();

        let dest = dir.path().join("cell").join("prompts.jsonl");
        fetch_dataset_to_file(&authority, "prompts.jsonl", &dest)
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
            fetch_dataset_to_file(&authority, "unknown.jsonl", &missing)
                .await
                .is_err(),
            "an unregistered dataset name must be rejected"
        );

        server.shutdown().await;
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
        datasets.insert("prompts.jsonl".to_owned(), src.clone());
        let server = ArtifactUploadServer::start_with_datasets(
            "127.0.0.1:0".parse().unwrap(),
            dir.path().join("controller-temp"),
            HashSet::new(),
            datasets,
        )
        .await
        .unwrap();
        let authority = server.local_addr().to_string();
        let shipped = dir.path().join("cell").join("prompts.jsonl");
        fetch_dataset_to_file(&authority, "prompts.jsonl", &shipped)
            .await
            .unwrap();
        server.shutdown().await;

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
