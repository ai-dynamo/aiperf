// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Cellular execution and measurement seams.
//!
//! A *cell* is a unit of autonomous scale: a thread, a process, or a node. The
//! single process is a cell of one and a cluster is `N` cells. Issuance assigns
//! dense global dispatch ordinals, partitions retain byte-exact records or
//! mergeable column stores, heartbeats carry mergeable t-digests, and transports
//! carry registrations, heartbeats, and partitions between cells and controllers.

pub mod broadcast;
pub mod dataset_session;
pub mod dispatch_state;
pub mod heartbeat;
pub mod issuance;
pub mod partition;
pub mod phaser;
pub mod shard;
pub mod sketch;
pub mod transport;

pub use heartbeat::{
    HeartbeatAccumulator, HeartbeatCounters, HeartbeatSaturation, MetricsHeartbeat,
};
pub use issuance::{CellularAutonomousIssuer, DirectIssuanceAuthority, IssuanceAuthority};
pub use partition::{CellPartition, CellPartitionError, ModuloCellPartition};
pub use shard::{
    ColumnStorePartition, DirectRecordsShard, PartitionCodecError, RecordsMergeError, RecordsShard,
    RecordsShardPartition, merge_records_by_concatenation, merge_records_in_global_order,
    merge_store_partitions,
};
pub use sketch::TDigest;
#[cfg(feature = "cellular")]
pub use transport::velo_transport::{VeloCellClient, VeloControllerTransport};
pub use transport::{
    CellAck, CellClient, CellMessage, CellPartitionShip, CellRegister, CellStorePartitionShip,
    CellTransportError, ControllerTransport, HANDLER_HEARTBEAT, HANDLER_PARTITION,
    HANDLER_REGISTER, HANDLER_STORE_PARTITION,
};
