// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The cross-node communication seam — heartbeats, phase signals, and partitions from
//! cells to the controller.
//!
//! - [`CellClient`] (the cell) sends messages. It is **async** — the velo impl
//!   ([`velo_transport::VeloCellClient`]) drives an async messaging client. A cell
//!   sends a handful of control messages plus one final partition, never on its
//!   per-request hot path.
//! - [`ControllerTransport`] (the controller) receives a merged stream of every
//!   cell's messages. The velo impl ([`velo_transport::VeloControllerTransport`])
//!   registers named velo handlers that push each decoded message into one channel.
//!
//! **Wire encoding.** Message bodies are MessagePack (`rmp-serde`) carried as
//! velo raw payloads rather than velo's typed JSON payloads because the sketches
//! in [`MetricsHeartbeat`] anchor `min = +inf` and the records carry NaN metric
//! values, neither of which JSON can round-trip.

use serde::{Deserialize, Serialize};

use crate::cellular::heartbeat::MetricsHeartbeat;
use crate::cellular::shard::{ColumnStorePartition, RecordsShardPartition};

/// Public, secret-free identity of a controller's per-run artifact TLS channel.
///
/// The certificate is delivered beside the cell envelope over Velo. The matching
/// private key never leaves the controller process.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArtifactChannelServerConfig {
    server_certificate_der: Vec<u8>,
}

impl ArtifactChannelServerConfig {
    pub(crate) fn new(server_certificate_der: Vec<u8>) -> Self {
        Self {
            server_certificate_der,
        }
    }

    pub(crate) fn server_certificate_der(&self) -> &[u8] {
        &self.server_certificate_der
    }
}

/// Discovery-free Velo construction and endpoint-based controller connection.
#[cfg(feature = "cellular")]
pub mod connect;
/// Velo distribution for the dataset fan-out data plane.
#[cfg(all(feature = "cellular", feature = "engine"))]
pub mod dataset_velo;
/// Velo distribution for the monotonic phaser control plane.
#[cfg(all(feature = "cellular", feature = "engine"))]
pub mod phaser_velo;
/// The velo-backed cell↔controller transport (cell client + controller
/// endpoint), gated on the `cellular` + `engine` features.
#[cfg(all(feature = "cellular", feature = "engine"))]
pub mod velo_transport;

/// One per-cell phase-barrier signal surfaced to the controller.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CellPhaseSignal {
    /// The cell has reached the named phase barrier and is waiting for controller action.
    Ready,
    /// The cell has completed the named phase and is acknowledging it to the controller.
    Complete,
}

/// One self-attributing message from a cell to the controller. A cell sends a small
/// number of fire-and-forget control messages plus its terminal partition, and closes;
/// the controller counts partitions to termination, so no explicit hello/goodbye
/// framing is needed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CellMessage {
    /// Result of a cell-local replay capability check, sent after envelope fetch
    /// and before the controller may release START.
    Preflight {
        /// Reporting cell identity.
        cell_id: u32,
        /// Successful capability check or a diagnostic refusal.
        result: Result<(), String>,
    },
    /// The cell's final counters + saturation + latency sketches. Boxed so it does
    /// not inflate the smaller `Partition` variant's footprint in the channel buffer.
    Heartbeat {
        /// The reporting cell's identifier.
        cell_id: u32,
        /// The cell's counters + saturation + latency sketches.
        heartbeat: Box<MetricsHeartbeat>,
    },
    /// A control-plane phase signal the controller later aggregates into exact
    /// cross-cell phase barriers.
    PhaseSignal {
        /// The reporting cell's identifier.
        cell_id: u32,
        /// The named phase gate the signal belongs to.
        phase: String,
        /// Which barrier point the cell is reporting.
        signal: CellPhaseSignal,
    },
    /// The cell's records-shard partition, sent once at run end. The partition
    /// carries its own `cell_id`.
    Partition(RecordsShardPartition),
    /// The cell's pre-accumulated column-store partition, sent once at run end in
    /// place of [`Self::Partition`] when the cell ran metrics-only exact folding: it
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
/// cell's sliced execute envelope (rmp-wrapped protocol-v2 JSON bytes), and the
/// call ticks the controller's readiness barrier.
pub const HANDLER_REGISTER: &str = "aiperf.cell.register";
/// velo handler name: cell → controller heartbeat (fire-and-forget `am_send`).
pub const HANDLER_HEARTBEAT: &str = "aiperf.cell.heartbeat";
/// velo handler name: cell → controller preflight result (fire-and-forget).
pub const HANDLER_PREFLIGHT: &str = "aiperf.cell.preflight";
/// velo handler name: cell → controller phase signal (fire-and-forget `am_send`).
pub const HANDLER_PHASE_SIGNAL: &str = "aiperf.cell.phase_signal";
/// velo handler name: cell → controller records-shard partition ship (unary; the
/// reply is an rmp [`CellAck`]).
pub const HANDLER_PARTITION: &str = "aiperf.cell.partition";
/// velo handler name: cell → controller column-store partition ship (unary; the
/// reply is an rmp [`CellAck`]). The exact-fold sibling of
/// [`HANDLER_PARTITION`]: a metrics-only cell ships its folded store, not a record
/// `Vec`.
pub const HANDLER_STORE_PARTITION: &str = "aiperf.cell.store_partition";

/// The cell's registration request: its `cell_id` plus its own serialized
/// `velo::PeerInfo` (rmp-encoded) so the controller can `register_peer` it and
/// route the reply (and later messages) back. The reply is an rmp
/// `RegisterReply` carrying the cell's sliced execute envelope bytes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CellRegister {
    /// Zero-based cell identifier — the barrier key.
    pub cell_id: u32,
    /// `rmp_serde`-encoded `velo::PeerInfo` of the registering cell.
    pub cell_peer: Vec<u8>,
    /// BLAKE3 digest of the cell-local artifact bearer, when the controller
    /// enabled the per-run artifact channel. The raw bearer never crosses Velo.
    pub artifact_capability_digest: Option<[u8; 32]>,
    /// Controller-verifiable proof that this registration came from the launched cell.
    #[serde(default)]
    pub registration_proof: Option<CellRegistrationProof>,
}

/// Wire-visible Ed25519 proof for a single cellular registration transcript.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CellRegistrationProof {
    /// Fixed protocol version for the signed registration transcript.
    pub version: u8,
    /// Per-run nonce selected by the controller.
    pub run_nonce: [u8; 32],
    /// Exact controller peer and resolved dial address observed by the cell.
    #[cfg(feature = "cellular")]
    pub(crate) controller_binding: connect::ControllerPeerBinding,
    /// Ed25519 signature over the canonical registration transcript.
    pub signature: Vec<u8>,
}

/// A cell's records-shard partition payload inside an authenticated frame.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CellPartitionShip {
    /// Identity repeated in the payload and checked against the authenticated role.
    pub cell_id: u32,
    /// The cell's records-shard partition.
    pub partition: RecordsShardPartition,
}

/// A cell's column-store partition payload inside an authenticated frame.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CellStorePartitionShip {
    /// Identity repeated in the payload and checked against the authenticated role.
    pub cell_id: u32,
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
    /// A controller handler was not published before the registration deadline.
    ReadinessTimeout {
        /// Handler whose typed publication was awaited.
        handler: &'static str,
    },
    /// A constant, redacted authentication failure.
    Authentication(&'static str),
}

impl std::fmt::Display for CellTransportError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Encode(error) => write!(f, "failed to encode cell message: {error}"),
            Self::Decode(error) => write!(f, "failed to decode cell message: {error}"),
            Self::Io(error) => write!(f, "cell transport io error: {error}"),
            Self::ReadinessTimeout { handler } => {
                write!(
                    f,
                    "cell transport readiness timed out for handler {handler}"
                )
            }
            Self::Authentication(detail) => {
                write!(f, "cell transport authentication failed: {detail}")
            }
        }
    }
}

impl std::error::Error for CellTransportError {}
