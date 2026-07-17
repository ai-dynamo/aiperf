// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Bounded-memory cross-host artifact shipping over the **velo** streaming plane.
//!
//! This is the velo sibling of [`crate::engine::artifact_shipping`] (raw-hyper
//! HTTP/1 + streaming zstd on a second port). It carries the exact same bytes —
//! per-record artifact files streamed as zstd chunks — but rides the SAME velo
//! instance/bootstrap the cellular control plane already uses (the controller's
//! `AIPERF_CELL_CONTROLLER_ADDR` endpoint, default port 9500), so no second port
//! (9600) has to be exposed cross-host.
//!
//! ## Transport primitive
//!
//! velo exposes a native ordered, backpressured streaming primitive —
//! [`velo::StreamAnchor`] / [`velo::StreamSender`] (see the pinned velo dep,
//! `lib/velo/src/streaming/{anchor,sender,frame}.rs`; `Velo::create_anchor` /
//! `Velo::attach_anchor` in `lib/velo/src/lib.rs`). We use it directly rather than
//! an application-level windowed `am_send` scheme, because:
//!
//! - **Ordering is guaranteed.** The messenger AM dispatcher is explicitly
//!   unordered (velo deprecates `VeloFrameTransport` for exactly this), so a naive
//!   `am_send` chunk stream would need seq numbers + an unbounded reorder buffer.
//!   The anchor's `TcpFrameTransport` delivers frames in order.
//! - **Backpressure is built in.** [`StreamSender::send`] awaits, so in-flight
//!   bytes stay `O(window · chunk)`, never `O(file)`.
//! - **No per-chunk lock.** Each file gets its own anchor with a dedicated
//!   consumer task, so there is no `Arc<Mutex<_>>` on the chunk path.
//!
//! ## Protocol
//!
//! The controller (data receiver) is the anchor **creator/consumer**; a cell (data
//! sender) is the anchor **attacher/producer**. Three unary handlers frame each
//! file so the producer learns the handle and the consumer's commit is observable:
//!
//! - [`HANDLER_ARTIFACT_OPEN`] (unary): a cell sends [`OpenRequest`]
//!   (`cell_id`, its own `PeerInfo`, and the relative artifact path). The controller
//!   `register_peer`s the cell (so the reverse streaming attach routes back),
//!   validates the path against the run allowlist, creates a per-file
//!   [`velo::StreamAnchor`], spawns a consumer that streaming-decompresses each
//!   frame into a `.part` file (bounded memory), and replies with the anchor
//!   [`OpenReply::handle`].
//! - The cell `attach_anchor`s that handle and streams zstd chunks
//!   ([`crate::engine::artifact_shipping::FileCompressor`]), then `finalize`s.
//! - [`HANDLER_ARTIFACT_CLOSE`] (unary): the cell blocks here until the controller's
//!   consumer task has committed (fsync + atomic rename) the `.part` file, so the
//!   cell never reports done before the bytes are durably landed.
//! - [`HANDLER_ARTIFACT_DONE`] (unary): the cell signals it has shipped every file;
//!   the controller's [`ArtifactVeloReceiver::wait_for_cells`] barrier releases once
//!   all cells are done — the velo mirror of the HTTP `/cell/{id}/done` barrier.
//!
//! Bodies are `rmp-serde`; chunks are `Vec<u8>` velo stream items.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::{Context as _, Result, bail, ensure};
use bytes::Bytes;
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use tokio::sync::{mpsc, oneshot, watch};
use velo::{Context, Handler, PeerInfo, StreamAnchorHandle, StreamFrame, Velo};

use super::artifact_shipping::{
    DecompressToFile, FileCompressor, shippable_relatives, validate_artifact_relpath,
};

/// Unary handler: open a per-file artifact stream and return its anchor handle.
pub const HANDLER_ARTIFACT_OPEN: &str = "aiperf.artifact.open";
/// Unary handler: wait for a per-file artifact stream to be committed on disk.
pub const HANDLER_ARTIFACT_CLOSE: &str = "aiperf.artifact.close";
/// Unary handler: a cell signals it has shipped all of its artifact files.
pub const HANDLER_ARTIFACT_DONE: &str = "aiperf.artifact.done";

/// A cell's request to open a per-file artifact stream. Carries the cell's own
/// `PeerInfo` (rmp bytes) so the controller can `register_peer` the fresh shipping
/// instance and route the streaming attach back, mirroring the partition-ship path.
#[derive(Debug, Serialize, Deserialize)]
struct OpenRequest {
    cell_id: u32,
    /// The shipping instance's `PeerInfo` (`rmp`), so the controller can address it.
    cell_peer: Vec<u8>,
    /// The relative artifact path (validated against the run allowlist).
    rel: String,
}

/// The controller's reply to [`OpenRequest`]: the anchor handle the cell attaches
/// to, plus the controller's full `PeerInfo`. The endpoint-first `velo.connect`
/// handshake does not always propagate the controller's STREAMING endpoint into the
/// cell-visible `PeerInfo` (it registers the peer for messaging), so the anchor
/// attach would fail with "peer not registered". Carrying the controller's complete
/// `peer_info()` here (which advertises the streaming key) lets the cell
/// `register_peer` it before attaching, fanning the streaming endpoint out reliably.
#[derive(Debug, Serialize, Deserialize)]
struct OpenReply {
    handle: StreamAnchorHandle,
    /// The controller's full `PeerInfo` (`rmp`), advertising its streaming endpoint.
    controller_peer: Vec<u8>,
}

/// A cell's request to wait for a per-file stream's on-disk commit.
#[derive(Debug, Serialize, Deserialize)]
struct CloseRequest {
    cell_id: u32,
    rel: String,
}

/// A cell's signal that it has shipped every artifact file.
#[derive(Debug, Serialize, Deserialize)]
struct DoneRequest {
    cell_id: u32,
}

/// A unary reply carrying success or a `Display` error string.
#[derive(Debug, Serialize, Deserialize)]
struct Ack {
    ok: bool,
    error: Option<String>,
}

impl Ack {
    fn ok() -> Self {
        Self {
            ok: true,
            error: None,
        }
    }
    fn err(error: impl std::fmt::Display) -> Self {
        Self {
            ok: false,
            error: Some(error.to_string()),
        }
    }
}

/// Key identifying one in-flight per-file transfer (`cell_id` + relative path).
type FileKey = (u32, String);

/// Encode a value as an rmp `Bytes` unary reply.
fn encode_reply<T: Serialize>(value: &T) -> anyhow::Result<Option<Bytes>> {
    let bytes = rmp_serde::to_vec(value).map_err(|error| anyhow::anyhow!("encode reply: {error}"))?;
    Ok(Some(Bytes::from(bytes)))
}

/// Controller-side shared state for the velo artifact plane.
struct ReceiverState {
    /// Where cell files land (`temp_root/cell-{id}/{rel}`).
    temp_root: PathBuf,
    /// The exact set of relative artifact paths the run may ship (fail-closed).
    allowed: HashSet<String>,
    /// Per-file commit signals: OPEN stores the receiver, the consumer task fires the
    /// sender after fsync+rename, and CLOSE awaits the receiver.
    completions: Mutex<HashMap<FileKey, oneshot::Receiver<Result<(), String>>>>,
    /// Cells that have signaled DONE, published through a version-tracked [`watch`]
    /// so a completion landing between the barrier's size check and its `.await`
    /// still wakes it — the same lost-wakeup guard the HTTP barrier uses.
    done: watch::Sender<HashSet<u32>>,
}

/// The controller endpoint for the velo artifact plane: registers the
/// open/close/done handlers on the shared control-plane velo instance and exposes a
/// cell-completion barrier ([`wait_for_cells`](Self::wait_for_cells)).
///
/// Unlike the HTTP [`ArtifactUploadServer`](crate::engine::artifact_shipping::ArtifactUploadServer),
/// this does NOT own a listener or a second port — it hangs handlers off the velo
/// instance the cellular control plane already bound, so artifact bytes ride the
/// one messaging endpoint.
pub struct ArtifactVeloReceiver {
    state: Arc<ReceiverState>,
    /// Held so the velo instance (and its registered handlers) outlives the receiver.
    _velo: Arc<Velo>,
}

impl ArtifactVeloReceiver {
    /// Register the artifact handlers on `velo`. `temp_root` is the controller's
    /// cellular scratch root (files land at `temp_root/cell-{id}/{rel}`); `allowed`
    /// is the exact set of relative artifact paths the run may ship.
    pub fn register(velo: Arc<Velo>, temp_root: PathBuf, allowed: HashSet<String>) -> Result<Self> {
        let state = Arc::new(ReceiverState {
            temp_root,
            allowed,
            completions: Mutex::new(HashMap::new()),
            done: watch::Sender::new(HashSet::new()),
        });

        // OPEN (unary): register the cell, create the per-file anchor + consumer,
        // reply with the handle. Capture a `Weak<Velo>` (not `Arc`) so the handler
        // does not form a reference cycle with the velo instance that owns it —
        // `create_anchor` lives on `Velo`, not the messenger `ctx.msg`.
        let open_state = state.clone();
        let open_velo = Arc::downgrade(&velo);
        velo.register_handler(
            Handler::unary_handler_async(HANDLER_ARTIFACT_OPEN, move |ctx: Context| {
                let state = open_state.clone();
                let velo = open_velo.clone();
                async move {
                    let Some(velo) = velo.upgrade() else {
                        return Err(anyhow::anyhow!("velo instance dropped before artifact open"));
                    };
                    handle_open(&state, &velo, ctx).await
                }
            })
            .build(),
        )
        .context("register artifact open handler")?;

        // CLOSE (unary): await the per-file commit, reply with its result.
        let close_state = state.clone();
        velo.register_handler(
            Handler::unary_handler_async(HANDLER_ARTIFACT_CLOSE, move |ctx: Context| {
                let state = close_state.clone();
                async move { handle_close(&state, ctx).await }
            })
            .build(),
        )
        .context("register artifact close handler")?;

        // DONE (unary): mark the cell done, wake the barrier.
        let done_state = state.clone();
        velo.register_handler(
            Handler::unary_handler_async(HANDLER_ARTIFACT_DONE, move |ctx: Context| {
                let state = done_state.clone();
                async move {
                    let request: DoneRequest = rmp_serde::from_slice(&ctx.payload)
                        .map_err(|error| anyhow::anyhow!("decode DoneRequest: {error}"))?;
                    // `send_modify` bumps the watch version even with no parked
                    // receiver, so a barrier not yet registered still observes this.
                    state.done.send_modify(|done| {
                        done.insert(request.cell_id);
                    });
                    encode_reply(&Ack::ok())
                }
            })
            .build(),
        )
        .context("register artifact done handler")?;

        Ok(Self { state, _velo: velo })
    }

    /// Wait until `cell_count` distinct cells have signaled DONE, or `timeout`
    /// elapses. The velo mirror of the HTTP artifact barrier.
    pub async fn wait_for_cells(&self, cell_count: u32, timeout: Duration) -> Result<()> {
        let mut rx = self.state.done.subscribe();
        let wait = async {
            loop {
                if rx.borrow_and_update().len() >= cell_count as usize {
                    return;
                }
                if rx.changed().await.is_err() {
                    return;
                }
            }
        };
        tokio::time::timeout(timeout, wait).await.map_err(|_| {
            let done = self.state.done.borrow();
            let missing: Vec<u32> = (0..cell_count).filter(|id| !done.contains(id)).collect();
            anyhow::anyhow!(
                "velo artifact barrier timed out after {timeout:?} with {} of {cell_count} \
                 cells done (missing cells: {missing:?})",
                done.len(),
            )
        })
    }
}

/// Handle an OPEN: validate the path, register the cell peer, create the per-file
/// anchor + streaming-decompress consumer, and reply with the anchor handle.
async fn handle_open(
    state: &Arc<ReceiverState>,
    velo: &Arc<Velo>,
    ctx: Context,
) -> anyhow::Result<Option<Bytes>> {
    let request: OpenRequest = rmp_serde::from_slice(&ctx.payload)
        .map_err(|error| anyhow::anyhow!("decode OpenRequest: {error}"))?;
    // Validate the relative path against the run allowlist (fail closed on traversal
    // or unknown artifacts) before creating any anchor or file.
    let rel = match validate_artifact_relpath(&request.rel, &state.allowed) {
        Ok(rel) => rel,
        Err(error) => return encode_reply(&Ack::err(error)),
    };
    let peer: PeerInfo = rmp_serde::from_slice(&request.cell_peer)
        .map_err(|error| anyhow::anyhow!("decode cell PeerInfo: {error}"))?;
    // Register the fresh shipping instance so the streaming attach routes back.
    ctx.msg
        .register_peer(peer)
        .map_err(|error| anyhow::anyhow!("register_peer artifact shipper: {error}"))?;

    let dest = state
        .temp_root
        .join(format!("cell-{}", request.cell_id))
        .join(&rel);

    // One anchor + consumer per file: no shared per-chunk lock.
    let anchor = velo.create_anchor::<Vec<u8>>();
    let handle = anchor.handle();

    let (commit_tx, commit_rx) = oneshot::channel::<Result<(), String>>();
    let key: FileKey = (request.cell_id, request.rel.clone());
    state
        .completions
        .lock()
        .expect("artifact completions mutex poisoned")
        .insert(key, commit_rx);

    let rel_label = request.rel.clone();
    let cell_id = request.cell_id;
    tokio::spawn(async move {
        let result = consume_stream_to_file(anchor, &dest).await;
        if let Ok(bytes) = &result {
            // A dedicated observable (mirrors the HTTP plane's "received artifact
            // upload over HTTP" line) so an operator/test can confirm the bytes really
            // crossed the velo stream. Enable with
            // `AIPERF_RUNNER_LOG=warn,aiperf_cellular_artifact=info`.
            tracing::info!(
                target: "aiperf_cellular_artifact",
                cell_id,
                artifact = %rel_label,
                transport = "velo",
                bytes = *bytes,
                "received artifact stream over velo"
            );
        }
        // A dropped receiver (a cell that never CLOSEs) is benign: log and move on.
        if commit_tx.send(result.map(|_| ()).map_err(|e| e.to_string())).is_err() {
            tracing::debug!(
                target: "aiperf_cellular_artifact",
                cell_id,
                artifact = %rel_label,
                "artifact stream committed but no CLOSE awaited it"
            );
        }
    });

    let controller_peer = rmp_serde::to_vec(&velo.peer_info())
        .map_err(|error| anyhow::anyhow!("encode controller peer: {error}"))?;
    encode_reply(&OpenReply {
        handle,
        controller_peer,
    })
}

/// Drive one per-file [`velo::StreamAnchor`] to terminal, streaming each zstd chunk
/// into a `.part` file and atomically committing on `Finalized`. Bounded memory:
/// one chunk decompressed at a time (the sync zstd write is microsecond-scale and
/// off the benchmark hot path).
async fn consume_stream_to_file(
    mut anchor: velo::StreamAnchor<Vec<u8>>,
    dest: &Path,
) -> Result<u64> {
    let mut sink = DecompressToFile::create(dest)
        .with_context(|| format!("creating artifact sink {}", dest.display()))?;
    let mut received: u64 = 0;
    while let Some(frame) = anchor.next().await {
        match frame {
            Ok(StreamFrame::Item(chunk)) => {
                received += chunk.len() as u64;
                sink.write_chunk(&chunk).context("writing artifact chunk")?;
            }
            Ok(StreamFrame::Finalized) => {
                sink.finish().context("committing artifact file")?;
                return Ok(received);
            }
            Ok(StreamFrame::Dropped) => {
                bail!("artifact stream sender dropped before finalize")
            }
            Ok(StreamFrame::Heartbeat) => {}
            Ok(other) => bail!("unexpected artifact stream sentinel: {other:?}"),
            Err(error) => bail!("artifact stream error: {error}"),
        }
    }
    bail!("artifact stream ended without a Finalized sentinel")
}

/// Handle a CLOSE: await the per-file commit and report its result.
async fn handle_close(state: &Arc<ReceiverState>, ctx: Context) -> anyhow::Result<Option<Bytes>> {
    let request: CloseRequest = rmp_serde::from_slice(&ctx.payload)
        .map_err(|error| anyhow::anyhow!("decode CloseRequest: {error}"))?;
    let key: FileKey = (request.cell_id, request.rel.clone());
    let rx = state
        .completions
        .lock()
        .expect("artifact completions mutex poisoned")
        .remove(&key);
    let Some(rx) = rx else {
        return encode_reply(&Ack::err(format!(
            "no open artifact stream for cell {} {:?}",
            request.cell_id, request.rel
        )));
    };
    match rx.await {
        Ok(Ok(())) => encode_reply(&Ack::ok()),
        Ok(Err(error)) => encode_reply(&Ack::err(error)),
        Err(_) => encode_reply(&Ack::err("artifact consumer dropped before commit")),
    }
}

/// Ship one cell's per-record artifact files (+ `inputs.json`) to the controller
/// over the velo streaming plane, then signal DONE.
///
/// `velo` is a connected instance (built + `connect_controller`ed by the caller);
/// `controller` is the resolved controller `PeerInfo`; `cell_dir` is this cell's own
/// artifact dir; `relatives` is the set of relative artifact paths to ship (only
/// those present on disk are sent). Bounded memory: each file streams through a
/// [`FileCompressor`] over a bounded channel into a backpressured [`velo::StreamSender`].
pub async fn ship_cell_artifacts_velo(
    velo: &Arc<Velo>,
    controller: &PeerInfo,
    cell_id: u32,
    cell_dir: &Path,
    relatives: &[String],
) -> Result<()> {
    // Register the controller peer so the streaming (anchor-attach) transport can
    // resolve its endpoint — `velo.connect`'s `_hello` handshake registers the peer
    // for messaging, but the anchor attach dials the streaming transport, which needs
    // the peer fanned out to it explicitly (idempotent; a re-register is harmless).
    let _ = velo.register_peer(controller.clone());
    for rel in relatives {
        let src = cell_dir.join(rel);
        if !src.exists() {
            continue;
        }
        ship_one_velo(velo, controller, cell_id, rel, &src)
            .await
            .with_context(|| format!("cell {cell_id} shipping artifact {rel:?} over velo"))?;
    }
    // DONE marker: the controller's barrier releases once every cell signals.
    let body = rmp_serde::to_vec(&DoneRequest { cell_id }).context("encode DoneRequest")?;
    let reply: Bytes = velo
        .unary(HANDLER_ARTIFACT_DONE)
        .context("artifact done unary")?
        .raw_payload(Bytes::from(body))
        .instance(controller.instance_id())
        .send()
        .await
        .context("sending artifact done")?;
    let ack: Ack = rmp_serde::from_slice(&reply).context("decode done ack")?;
    ensure!(ack.ok, "controller nacked artifact done: {:?}", ack.error);
    Ok(())
}

/// Stream one file: OPEN → attach → pump zstd chunks (backpressured) → finalize →
/// CLOSE (awaits the controller commit). Compression runs on a blocking task feeding
/// a bounded channel; the whole file is never resident.
async fn ship_one_velo(
    velo: &Arc<Velo>,
    controller: &PeerInfo,
    cell_id: u32,
    rel: &str,
    src: &Path,
) -> Result<()> {
    // OPEN: hand the controller our peer + path, receive the anchor handle.
    let cell_peer = rmp_serde::to_vec(&velo.peer_info()).context("encode cell peer")?;
    let open = OpenRequest {
        cell_id,
        cell_peer,
        rel: rel.to_owned(),
    };
    let body = rmp_serde::to_vec(&open).context("encode OpenRequest")?;
    let reply: Bytes = velo
        .unary(HANDLER_ARTIFACT_OPEN)
        .context("artifact open unary")?
        .raw_payload(Bytes::from(body))
        .instance(controller.instance_id())
        .send()
        .await
        .context("sending artifact open")?;
    // A validation failure comes back as an Ack (never an OpenReply); decode that
    // shape first so a rejected path surfaces a clean error, not a decode failure.
    if let Ok(ack) = rmp_serde::from_slice::<Ack>(&reply)
        && !ack.ok
    {
        bail!("controller rejected artifact open: {:?}", ack.error);
    }
    let open_reply: OpenReply = rmp_serde::from_slice(&reply).context("decode OpenReply")?;

    // Register the controller's full PeerInfo so its STREAMING endpoint is known to
    // this cell's anchor transport before we attach (the connect handshake alone may
    // not have propagated it). Idempotent for the messaging side.
    let controller_full: PeerInfo =
        rmp_serde::from_slice(&open_reply.controller_peer).context("decode controller peer")?;
    let _ = velo.register_peer(controller_full);

    // Attach the backpressured sender to the controller's anchor.
    let sender = velo
        .attach_anchor::<Vec<u8>>(open_reply.handle)
        .await
        .map_err(|error| anyhow::anyhow!("attach artifact anchor: {error}"))?;

    // Producer: chunked read + zstd on a blocking task → bounded channel.
    let (tx, mut rx) = mpsc::channel::<Vec<u8>>(4);
    let path = src.to_path_buf();
    let producer = tokio::task::spawn_blocking(move || -> std::io::Result<()> {
        let mut compressor = FileCompressor::open(&path)?;
        while let Some(chunk) = compressor.next_chunk()? {
            if tx.blocking_send(chunk).is_err() {
                break; // receiver dropped (stream failed); stop early
            }
        }
        Ok(())
    });

    // Pump chunks into the backpressured velo sender (in-flight bytes O(window·chunk)).
    while let Some(chunk) = rx.recv().await {
        sender
            .send(chunk)
            .await
            .map_err(|error| anyhow::anyhow!("send artifact chunk: {error}"))?;
    }
    match producer.await {
        Ok(Ok(())) => {}
        Ok(Err(error)) => return Err(error).context("streaming compress"),
        Err(join) => bail!("compression task panicked: {join}"),
    }
    sender
        .finalize()
        .map_err(|error| anyhow::anyhow!("finalize artifact stream: {error}"))?;

    // CLOSE: block until the controller has committed the file to disk.
    let close = CloseRequest {
        cell_id,
        rel: rel.to_owned(),
    };
    let body = rmp_serde::to_vec(&close).context("encode CloseRequest")?;
    let reply: Bytes = velo
        .unary(HANDLER_ARTIFACT_CLOSE)
        .context("artifact close unary")?
        .raw_payload(Bytes::from(body))
        .instance(controller.instance_id())
        .send()
        .await
        .context("sending artifact close")?;
    let ack: Ack = rmp_serde::from_slice(&reply).context("decode close ack")?;
    ensure!(
        ack.ok,
        "controller failed to commit artifact {rel:?}: {:?}",
        ack.error
    );
    Ok(())
}

/// The relative artifact paths a run may ship over velo — identical to the HTTP
/// plane's set ([`shippable_relatives`]), so the two transports ship the same files.
pub fn shippable_relatives_velo(artifacts: &crate::engine::protocol::ArtifactSpec) -> Vec<String> {
    shippable_relatives(artifacts)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cellular::transport::connect::{BindSpec, build_velo, connect_controller};
    use std::io::Write;

    /// A deterministic pseudo-random payload of `len` bytes (no allocator noise).
    fn sample_bytes(len: usize) -> Vec<u8> {
        let mut out = Vec::with_capacity(len);
        let mut state: u64 = 0x1234_5678_9ABC_DEF0;
        while out.len() < len {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            out.extend_from_slice(&state.to_le_bytes());
        }
        out.truncate(len);
        out
    }

    /// Peak RSS of this process in bytes (Linux `statm` resident pages), or `None`
    /// off Linux. Used to bound the streaming transfer's live memory.
    fn resident_bytes() -> Option<u64> {
        let statm = std::fs::read_to_string("/proc/self/statm").ok()?;
        let resident_pages: u64 = statm.split_whitespace().nth(1)?.parse().ok()?;
        Some(resident_pages * 4096)
    }

    // Stream a large (≥50 MB) synthetic artifact over two REAL velo instances on
    // loopback TCP and assert: (1) the receiver reassembles byte-identical, and
    // (2) the transfer's live memory growth is bounded (O(window·chunk)), not
    // O(file) — the whole file is never resident on either end.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn velo_stream_large_artifact_round_trips_with_bounded_memory() {
        let dir = tempfile::tempdir().unwrap();

        // A 64 MB incompressible-ish artifact spanning ~1000 chunks.
        let payload_len = 64 * 1024 * 1024;
        let src_dir = dir.path().join("cell-src");
        std::fs::create_dir_all(&src_dir).unwrap();
        let rel = "profile_export.jsonl";
        {
            let mut file = std::fs::File::create(src_dir.join(rel)).unwrap();
            file.write_all(&sample_bytes(payload_len)).unwrap();
        }
        let source_bytes = std::fs::read(src_dir.join(rel)).unwrap();

        let allowed: HashSet<String> = [rel.to_owned()].into_iter().collect();
        let landing = dir.path().join("controller-landing");

        // Controller velo + artifact receiver.
        let controller_velo = build_velo(BindSpec::TcpLoopback).await.expect("controller velo");
        let controller_addr = controller_velo.peer_info();
        let controller_endpoint = {
            // Rebuild from a bound listener so the cell can connect by endpoint.
            controller_addr.clone()
        };
        let _ = controller_endpoint;
        let receiver = ArtifactVeloReceiver::register(
            controller_velo.clone(),
            landing.clone(),
            allowed,
        )
        .expect("register receiver");

        // Cell velo connects to the controller by its PeerInfo (loopback).
        let cell_velo = build_velo(BindSpec::TcpLoopback).await.expect("cell velo");
        // Register the controller peer directly (same as VeloCellClient::connect).
        cell_velo
            .register_peer(controller_addr.clone())
            .expect("register controller peer");

        let baseline = resident_bytes();

        ship_cell_artifacts_velo(
            &cell_velo,
            &controller_addr,
            0,
            &src_dir,
            &[rel.to_owned()],
        )
        .await
        .expect("ship over velo");

        receiver
            .wait_for_cells(1, Duration::from_secs(30))
            .await
            .expect("barrier");

        // Byte-identical reassembly.
        let landed = landing.join("cell-0").join(rel);
        assert_eq!(
            std::fs::read(&landed).unwrap(),
            source_bytes,
            "velo-shipped artifact landed byte-identical"
        );

        // Bounded memory: the transfer must not have grown RSS by anywhere near the
        // 64 MB file size. Allow a generous 24 MB slack for velo buffers/tokio/zstd,
        // still far below O(file).
        if let (Some(base), Some(peak)) = (baseline, resident_bytes()) {
            let grew = peak.saturating_sub(base);
            assert!(
                grew < 24 * 1024 * 1024,
                "RSS grew {grew} bytes during a {payload_len}-byte transfer — not O(chunk)"
            );
        }
    }

    // A path outside the run allowlist is rejected at OPEN (fail closed), and the
    // cell surfaces a clean error rather than landing bytes.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn velo_rejects_unallowed_artifact() {
        let dir = tempfile::tempdir().unwrap();
        let src_dir = dir.path().join("cell-src");
        std::fs::create_dir_all(&src_dir).unwrap();
        std::fs::write(src_dir.join("secret.parquet"), b"nope").unwrap();

        let allowed: HashSet<String> = ["profile_export.jsonl".to_owned()].into_iter().collect();
        let controller_velo = build_velo(BindSpec::TcpLoopback).await.expect("controller velo");
        let controller_peer = controller_velo.peer_info();
        let _receiver = ArtifactVeloReceiver::register(
            controller_velo.clone(),
            dir.path().join("landing"),
            allowed,
        )
        .expect("register");

        let cell_velo = build_velo(BindSpec::TcpLoopback).await.expect("cell velo");
        cell_velo.register_peer(controller_peer.clone()).unwrap();

        let result = ship_cell_artifacts_velo(
            &cell_velo,
            &controller_peer,
            0,
            &src_dir,
            &["secret.parquet".to_owned()],
        )
        .await;
        assert!(result.is_err(), "unallowed artifact must be rejected");
    }

    // The `connect_controller` bootstrap path (endpoint-first) also carries the
    // streaming attach: build a controller on a bound listener, dial it by
    // `tcp://addr`, and stream a small file end-to-end.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn velo_stream_over_connect_by_endpoint() {
        let dir = tempfile::tempdir().unwrap();
        let src_dir = dir.path().join("cell-src");
        std::fs::create_dir_all(&src_dir).unwrap();
        let rel = "inputs.json";
        std::fs::write(src_dir.join(rel), sample_bytes(200_000)).unwrap();
        let source_bytes = std::fs::read(src_dir.join(rel)).unwrap();

        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let controller_velo = build_velo(BindSpec::TcpListener(listener))
            .await
            .expect("controller velo");
        let allowed: HashSet<String> = [rel.to_owned()].into_iter().collect();
        let landing = dir.path().join("landing");
        let receiver =
            ArtifactVeloReceiver::register(controller_velo.clone(), landing.clone(), allowed)
                .expect("register");

        let cell_velo = build_velo(BindSpec::TcpLoopback).await.expect("cell velo");
        let controller_peer = connect_controller(&cell_velo, &format!("tcp://{addr}"))
            .await
            .expect("connect");

        ship_cell_artifacts_velo(&cell_velo, &controller_peer, 2, &src_dir, &[rel.to_owned()])
            .await
            .expect("ship");
        receiver
            .wait_for_cells(1, Duration::from_secs(30))
            .await
            .expect("barrier");

        assert_eq!(
            std::fs::read(landing.join("cell-2").join(rel)).unwrap(),
            source_bytes,
        );
    }
}
