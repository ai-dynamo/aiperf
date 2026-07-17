// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The cross-node communication seam — heartbeats and partitions from cells to the
//! controller.
//!
//! The roadmap keeps a *cell* transport- and deployment-neutral: the concept is
//! "get a serialized [`CellMessage`] from a cell to the aggregator", not any fixed
//! wire (`specs/2026-07-12-cellular-ready-seams-and-roadmap.md`, S2/S3 "Later"; the
//! velo realization is `specs/2026-07-15-velo-cell-transport-design.md`).
//!
//! Two sides behind two traits:
//! - [`CellClient`] (the cell) sends messages. It is **async** — the velo impl
//!   ([`velo_transport::VeloCellClient`]) drives an async messaging client. A cell
//!   sends a handful of heartbeats plus one final partition, never on its
//!   per-request hot path.
//! - [`ControllerTransport`] (the controller) receives a merged stream of every
//!   cell's messages. The velo impl ([`velo_transport::VeloControllerTransport`])
//!   registers named velo handlers that push each decoded message into one channel.
//!
//! **Wire encoding.** Message bodies are MessagePack (`rmp-serde`) carried as
//! velo *raw* payloads — NOT velo's typed (JSON) payloads — because the sketches
//! in [`MetricsHeartbeat`] anchor `min = +inf` and the records carry NaN metric
//! values, neither of which JSON can round-trip. velo owns framing and (for a large
//! partition) transparent large-payload staging, so this module no longer does its
//! own length-prefixing.
//!
//! A cell that is a thread rather than a process would implement the same two
//! traits over an in-process channel — the controller and merge logic are unchanged.

use serde::{Deserialize, Serialize};

use crate::cellular::heartbeat::MetricsHeartbeat;
use crate::cellular::shard::{ColumnStorePartition, RecordsShardPartition};

/// Discovery-free connection seam: velo transport construction + the
/// bootstrap-PeerInfo exchange that lets a cell reach the controller from one
/// operator-hardcoded coordinate. Gated on the `velo` feature.
#[cfg(feature = "cellular")]
pub mod connect;
/// Velo distribution for the dataset fan-out data plane (ultimate spec §3).
#[cfg(feature = "cellular")]
pub mod dataset_velo;
/// Velo distribution for the monotonic phaser control plane (ultimate spec §4).
#[cfg(feature = "cellular")]
pub mod phaser_velo;
/// The velo-backed cell↔controller transport (cell client + controller
/// endpoint), gated on the `velo` feature.
#[cfg(feature = "cellular")]
pub mod velo_transport;

/// One self-attributing message from a cell to the controller. A cell sends its
/// final heartbeat then its partition, and closes; the controller counts partitions
/// to termination, so no explicit hello/goodbye framing is needed.
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
    /// The cell's pre-accumulated column-store partition, sent once at run end in
    /// place of [`Partition`] when the cell ran metrics-only exact-fold (Stage C): it
    /// folded its records into its own EXACT accumulator and dropped them, so it has
    /// no record `Vec` to ship — it ships the folded store instead. The controller
    /// appends every cell's store ([`merge_store_partitions`](crate::cellular::merge_store_partitions))
    /// into the merged report (within-tolerance summary, not byte-exact — see
    /// [`ColumnStorePartition`]). The partition carries its own `cell_id`. Boxed (like
    /// [`Heartbeat`](Self::Heartbeat)) so this — the largest variant by far, a whole
    /// folded store — does not inflate every slot of the controller's message channel.
    StorePartition(Box<ColumnStorePartition>),
}

/// velo handler name: cell → controller registration. The reply carries the
/// cell's serialized `CellLaunchSpec` (rmp), replacing the stdin pipe, and the
/// call ticks the controller's readiness barrier.
pub const HANDLER_REGISTER: &str = "aiperf.cell.register";
/// velo handler name: cell → controller heartbeat (fire-and-forget `am_send`).
pub const HANDLER_HEARTBEAT: &str = "aiperf.cell.heartbeat";
/// velo handler name: cell → controller records-shard partition ship (unary; the
/// reply is an rmp [`CellAck`]).
pub const HANDLER_PARTITION: &str = "aiperf.cell.partition";
/// velo handler name: cell → controller column-store partition ship (unary; the
/// reply is an rmp [`CellAck`]). The Stage-C exact-fold sibling of
/// [`HANDLER_PARTITION`]: a metrics-only cell ships its folded store, not a record
/// `Vec`.
pub const HANDLER_STORE_PARTITION: &str = "aiperf.cell.store_partition";

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

/// A cell's records-shard partition ship. Carries the shipping velo instance's
/// own serialized `PeerInfo` alongside the partition so the controller can
/// `register_peer` it and route the ack back — a cell ships from a *fresh* velo
/// instance (its spec-fetch instance is already gone), which the controller has
/// not yet seen, so the register-time peer does not suffice.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CellPartitionShip {
    /// `rmp_serde`-encoded `velo::PeerInfo` of the shipping cell instance.
    pub cell_peer: Vec<u8>,
    /// The cell's records-shard partition.
    pub partition: RecordsShardPartition,
}

/// A cell's column-store partition ship — the Stage-C exact-fold sibling of
/// [`CellPartitionShip`]. Carries the shipping velo instance's own serialized
/// `velo::PeerInfo` (rmp-encoded) alongside the folded store so the controller can
/// `register_peer` it and route the ack back (the cell ships from a *fresh* velo
/// instance the controller has not yet seen).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CellStorePartitionShip {
    /// `rmp_serde`-encoded `velo::PeerInfo` of the shipping cell instance.
    pub cell_peer: Vec<u8>,
    /// The cell's folded column-store partition.
    pub partition: ColumnStorePartition,
}

/// Generic controller acknowledgement reply (rmp), returned from the partition
/// handler so the cell knows its shard was received before exiting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CellAck {
    /// Whether the controller accepted the message.
    pub ok: bool,
}

/// The cell side of the seam: sends [`CellMessage`]s to the controller. Async
/// because the velo messaging client is async (a cell sends only a few control
/// messages, off its per-request hot path).
#[async_trait::async_trait]
pub trait CellClient {
    /// Sends one message, awaiting until it is handed to the transport.
    async fn send(&mut self, message: &CellMessage) -> Result<(), CellTransportError>;
}

/// The controller side of the seam: a merged stream of every cell's messages.
#[async_trait::async_trait]
pub trait ControllerTransport {
    /// Receives the next message from any cell, or `None` once the stream closes.
    async fn recv(&mut self) -> Result<Option<CellMessage>, CellTransportError>;
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
    /// A transport (velo send/receive, connection) failure.
    Io(String),
}

impl std::fmt::Display for CellTransportError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Encode(error) => write!(f, "failed to encode cell message: {error}"),
            Self::Decode(error) => write!(f, "failed to decode cell message: {error}"),
            Self::Io(error) => write!(f, "cell transport io error: {error}"),
        }
    }
}

impl std::error::Error for CellTransportError {}
