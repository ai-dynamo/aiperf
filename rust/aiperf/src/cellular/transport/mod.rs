// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The cross-node communication seam — heartbeats and partitions from cells to the
//! controller.
//!
//! The roadmap keeps a *cell* transport- and deployment-neutral: the concept is
//! "get a serialized [`CellMessage`] from a cell to the aggregator", not any fixed
//! wire (`specs/2026-07-12-cellular-ready-seams-and-roadmap.md`, S2/S3 "Later").
//! Every message is length-prefixed MessagePack (`u32` big-endian length + body),
//! which preserves the NaN/`+inf` sketch sentinels JSON cannot and round-trips the
//! untagged `MetricValue` a non-self-describing format cannot.
//!
//! Two sides behind two traits:
//! - [`CellClient`] (the cell) sends messages. [`TcpCellClient`] is a blocking
//!   `std::net::TcpStream` — a cell sends a handful of heartbeats plus one final
//!   partition, never on its per-request hot path, so blocking writes are fine and
//!   keep the cell off an async reactor.
//! - [`ControllerTransport`] (the controller) receives a merged stream of every
//!   cell's messages. [`TcpControllerTransport`] accepts `expected_cells`
//!   connections on a Tokio listener, reads each concurrently, and merges them into
//!   one channel; `recv` yields `None` once every cell has closed.
//!
//! A cell that is a thread rather than a process would implement the same two
//! traits over an in-process channel — the controller and merge logic are unchanged.

use std::net::SocketAddr;

use serde::{Deserialize, Serialize};
use tokio::io::{AsyncRead, AsyncReadExt};
use tokio::net::{TcpListener, TcpStream, ToSocketAddrs};
use tokio::sync::mpsc;
use tokio::task::JoinHandle;

use crate::cellular::heartbeat::MetricsHeartbeat;
use crate::cellular::shard::RecordsShardPartition;

/// Discovery-free connection seam: velo transport construction + the
/// bootstrap-PeerInfo exchange that lets a cell reach the controller from one
/// operator-hardcoded coordinate. Gated on the `velo` feature.
#[cfg(feature = "velo")]
pub mod connect;

/// A frame body larger than this is rejected as corrupt/hostile rather than
/// allocated — defense-in-depth against a bad length prefix.
const MAX_FRAME_LEN: u32 = 512 * 1024 * 1024;

/// One self-attributing message from a cell to the controller. A cell sends its
/// final heartbeat then its partition, and closes; the controller counts partitions
/// to termination and detects a cell failure by the child's non-zero exit code, so
/// no explicit hello/goodbye framing is needed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CellMessage {
    /// The cell's final counters + saturation + latency sketches. Boxed so it does
    /// not inflate the smaller `Partition` variant's footprint in the channel buffer.
    Heartbeat {
        /// The reporting cell's identifier.
        cell_id: u32,
        /// The cell's counters + saturation + latency sketches.
        heartbeat: Box<MetricsHeartbeat>,
    },
    /// The cell's records-shard partition, sent once at run end. The partition
    /// carries its own `cell_id`.
    Partition(RecordsShardPartition),
}

/// velo handler name: cell → controller registration. The reply carries the
/// cell's serialized `CellLaunchSpec` (rmp), replacing the stdin pipe, and the
/// call ticks the controller's readiness barrier.
pub const HANDLER_REGISTER: &str = "aiperf.cell.register";
/// velo handler name: cell → controller heartbeat (fire-and-forget `am_send`).
pub const HANDLER_HEARTBEAT: &str = "aiperf.cell.heartbeat";
/// velo handler name: cell → controller records-shard partition ship. The reply
/// is a [`CellAck`]; the body is transferred by a rendezvous handle.
pub const HANDLER_PARTITION: &str = "aiperf.cell.partition";

/// The cell's registration request: its `cell_id` plus its own serialized
/// `velo::PeerInfo` (rmp-encoded) so the controller can `register_peer` it and
/// route the reply (and later messages) back. The reply is the cell's
/// `CellLaunchSpec` bytes (rmp).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CellRegister {
    /// Zero-based cell identifier — the barrier key.
    pub cell_id: u32,
    /// `rmp_serde`-encoded `velo::PeerInfo` of the registering cell.
    pub cell_peer: Vec<u8>,
}

/// A cell's records-shard partition ship: the `cell_id` plus the rendezvous
/// `DataHandle` (as `u128`) of the rmp-encoded [`RecordsShardPartition`], so a
/// large partition is pulled out-of-band rather than inflating an AM frame.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CellPartitionShip {
    /// The shipping cell's identifier.
    pub cell_id: u32,
    /// The rendezvous data handle (`velo::DataHandle` as `u128`) of the partition body.
    pub data_handle: u128,
}

/// Generic controller acknowledgement reply.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CellAck {
    /// Whether the controller accepted the message.
    pub ok: bool,
}

/// Encodes a message as a length-prefixed MessagePack frame.
pub fn encode_frame(message: &CellMessage) -> Result<Vec<u8>, CellTransportError> {
    let body = rmp_serde::to_vec(message)
        .map_err(|error| CellTransportError::Encode(error.to_string()))?;
    // Enforce the same cap the reader applies, so an oversized partition fails on the
    // cell before the transfer rather than being rejected after it on the controller.
    if body.len() > MAX_FRAME_LEN as usize {
        return Err(CellTransportError::FrameTooLarge(body.len()));
    }
    let len =
        u32::try_from(body.len()).map_err(|_| CellTransportError::FrameTooLarge(body.len()))?;
    let mut frame = Vec::with_capacity(4 + body.len());
    frame.extend_from_slice(&len.to_be_bytes());
    frame.extend_from_slice(&body);
    Ok(frame)
}

/// Reads one length-prefixed frame, or `None` at a clean end of stream.
async fn read_frame<R>(reader: &mut R) -> Result<Option<CellMessage>, CellTransportError>
where
    R: AsyncRead + Unpin,
{
    let mut len_buf = [0_u8; 4];
    match reader.read_exact(&mut len_buf).await {
        Ok(_) => {}
        // A cell that closed after its last frame ends the stream cleanly.
        Err(error) if error.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(None),
        Err(error) => return Err(CellTransportError::Io(error.to_string())),
    }
    let len = u32::from_be_bytes(len_buf);
    if len > MAX_FRAME_LEN {
        return Err(CellTransportError::FrameTooLarge(len as usize));
    }
    let mut body = vec![0_u8; len as usize];
    reader
        .read_exact(&mut body)
        .await
        .map_err(|error| CellTransportError::Io(error.to_string()))?;
    let message = rmp_serde::from_slice(&body)
        .map_err(|error| CellTransportError::Decode(error.to_string()))?;
    Ok(Some(message))
}

/// The cell side of the seam: sends [`CellMessage`]s to the controller.
pub trait CellClient {
    /// Sends one message, blocking until it is written.
    fn send(&mut self, message: &CellMessage) -> Result<(), CellTransportError>;
}

/// The controller side of the seam: a merged stream of every cell's messages.
#[async_trait::async_trait]
pub trait ControllerTransport {
    /// Receives the next message from any cell, or `None` once all cells closed.
    async fn recv(&mut self) -> Result<Option<CellMessage>, CellTransportError>;
}

/// A cell's blocking TCP connection to the controller.
pub struct TcpCellClient {
    stream: std::net::TcpStream,
}

impl TcpCellClient {
    /// Connects to the controller at `addr`, disabling Nagle so heartbeats flush
    /// promptly.
    pub fn connect(addr: impl std::net::ToSocketAddrs) -> Result<Self, CellTransportError> {
        let stream = std::net::TcpStream::connect(addr)
            .map_err(|error| CellTransportError::Io(error.to_string()))?;
        let _ = stream.set_nodelay(true);
        Ok(Self { stream })
    }
}

impl CellClient for TcpCellClient {
    fn send(&mut self, message: &CellMessage) -> Result<(), CellTransportError> {
        use std::io::Write;
        let frame = encode_frame(message)?;
        self.stream
            .write_all(&frame)
            .and_then(|()| self.stream.flush())
            .map_err(|error| CellTransportError::Io(error.to_string()))
    }
}

/// The controller's TCP endpoint: accepts `expected_cells` connections and merges
/// their framed messages into one channel.
pub struct TcpControllerTransport {
    receiver: mpsc::Receiver<Result<CellMessage, CellTransportError>>,
    local_addr: SocketAddr,
    accept: JoinHandle<()>,
}

impl Drop for TcpControllerTransport {
    fn drop(&mut self) {
        // Stop accepting/reading as soon as the controller drops the endpoint (e.g.
        // on an aborted run), rather than leaving the accept task blocked on a
        // connection that will never arrive.
        self.accept.abort();
    }
}

impl TcpControllerTransport {
    /// Binds `addr` and starts accepting exactly `expected_cells` cell connections.
    /// Returns once bound; the concrete [`local_addr`](Self::local_addr) (e.g. the
    /// OS-assigned port for `:0`) is what cells connect to.
    pub async fn bind(
        addr: impl ToSocketAddrs,
        expected_cells: usize,
    ) -> Result<Self, CellTransportError> {
        let listener = TcpListener::bind(addr)
            .await
            .map_err(|error| CellTransportError::Io(error.to_string()))?;
        let local_addr = listener
            .local_addr()
            .map_err(|error| CellTransportError::Io(error.to_string()))?;
        let (sender, receiver) = mpsc::channel(1024);
        let accept = tokio::spawn(async move {
            let mut readers = Vec::with_capacity(expected_cells);
            for _ in 0..expected_cells {
                match listener.accept().await {
                    Ok((socket, _)) => {
                        let sender = sender.clone();
                        readers.push(tokio::spawn(read_connection(socket, sender)));
                    }
                    Err(error) => {
                        let _ = sender
                            .send(Err(CellTransportError::Io(error.to_string())))
                            .await;
                        break;
                    }
                }
            }
            for reader in readers {
                let _ = reader.await;
            }
            // Dropping the last `sender` here closes the channel → `recv` yields None.
        });
        Ok(Self {
            receiver,
            local_addr,
            accept,
        })
    }

    /// The bound address cells connect to.
    pub fn local_addr(&self) -> SocketAddr {
        self.local_addr
    }
}

#[async_trait::async_trait]
impl ControllerTransport for TcpControllerTransport {
    async fn recv(&mut self) -> Result<Option<CellMessage>, CellTransportError> {
        match self.receiver.recv().await {
            Some(Ok(message)) => Ok(Some(message)),
            Some(Err(error)) => Err(error),
            None => Ok(None),
        }
    }
}

async fn read_connection(
    mut socket: TcpStream,
    sender: mpsc::Sender<Result<CellMessage, CellTransportError>>,
) {
    loop {
        match read_frame(&mut socket).await {
            Ok(Some(message)) => {
                if sender.send(Ok(message)).await.is_err() {
                    return;
                }
            }
            Ok(None) => return,
            Err(error) => {
                let _ = sender.send(Err(error)).await;
                return;
            }
        }
    }
}

/// Error encoding, decoding, or transporting a [`CellMessage`].
///
/// A plain enum with a hand-written [`Display`](std::fmt::Display) per the crate's
/// error convention.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CellTransportError {
    /// A message could not be encoded.
    Encode(String),
    /// A message could not be decoded.
    Decode(String),
    /// A socket read/write failed.
    Io(String),
    /// A length prefix exceeded [`MAX_FRAME_LEN`].
    FrameTooLarge(usize),
}

impl std::fmt::Display for CellTransportError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Encode(error) => write!(f, "failed to encode cell message: {error}"),
            Self::Decode(error) => write!(f, "failed to decode cell message: {error}"),
            Self::Io(error) => write!(f, "cell transport io error: {error}"),
            Self::FrameTooLarge(len) => {
                write!(f, "cell message frame of {len} bytes exceeds the limit")
            }
        }
    }
}

impl std::error::Error for CellTransportError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cellular::heartbeat::HeartbeatAccumulator;
    use crate::metrics_core::ingest::RecordIngest;
    use crate::metrics_core::window::Phase;

    fn sample_partition(cell_id: u32) -> RecordsShardPartition {
        let mut record = RecordIngest::minimal(1_000, 5_000, Phase::Profiling);
        record.request_index = Some(cell_id as usize);
        RecordsShardPartition::new(cell_id, vec![record])
    }

    fn sample_heartbeat() -> MetricsHeartbeat {
        let mut accumulator = HeartbeatAccumulator::new();
        accumulator.observe(Some(20.0), Some(5.0), Some(50.0));
        accumulator.snapshot(1, Default::default(), Default::default())
    }

    #[test]
    fn frame_round_trips_every_message_variant() {
        for message in [
            CellMessage::Heartbeat {
                cell_id: 1,
                heartbeat: Box::new(sample_heartbeat()),
            },
            CellMessage::Partition(sample_partition(2)),
        ] {
            let frame = encode_frame(&message).expect("encode");
            // The 4-byte prefix equals the body length.
            let declared = u32::from_be_bytes(frame[..4].try_into().unwrap()) as usize;
            assert_eq!(declared, frame.len() - 4);
            let decoded: CellMessage = rmp_serde::from_slice(&frame[4..]).expect("decode");
            assert_eq!(format!("{decoded:?}"), format!("{message:?}"));
        }
    }

    #[tokio::test]
    async fn tcp_loopback_delivers_all_cells_messages_then_closes() {
        let cell_count = 3;
        let mut controller = TcpControllerTransport::bind("127.0.0.1:0", cell_count)
            .await
            .expect("bind");
        let addr = controller.local_addr();

        // Each "cell" connects on a blocking thread and ships its heartbeat then its
        // partition, matching the real cell's ship order, and closes.
        let mut handles = Vec::new();
        for cell_id in 0..cell_count as u32 {
            handles.push(std::thread::spawn(move || {
                let mut client = TcpCellClient::connect(addr).expect("connect");
                client
                    .send(&CellMessage::Heartbeat {
                        cell_id,
                        heartbeat: Box::new(sample_heartbeat()),
                    })
                    .unwrap();
                client
                    .send(&CellMessage::Partition(sample_partition(cell_id)))
                    .unwrap();
            }));
        }

        let mut heartbeats = 0;
        let mut partitions = 0;
        while let Some(message) = controller.recv().await.expect("recv") {
            match message {
                CellMessage::Partition(partition) => {
                    assert_eq!(partition.len(), 1);
                    partitions += 1;
                }
                CellMessage::Heartbeat { .. } => heartbeats += 1,
            }
        }
        for handle in handles {
            handle.join().unwrap();
        }
        assert_eq!((heartbeats, partitions), (cell_count, cell_count));
    }
}
