// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Velo distribution for the dataset data plane.
//!
//! Makes the in-process
//! [`DatasetPublisher`](crate::cellular::dataset_session::DatasetPublisher) /
//! [`DatasetIndex`](crate::cellular::dataset_session::DatasetIndex) a distributed
//! fan-out: the controller generates the dataset once and `add`s chunks; each cell
//! subscribes over velo and receives **replay-on-attach** (every chunk so far in the
//! subscribe reply) then the **live tail** (pushed as more chunks land), and builds
//! its owned index. Two handlers, identical in shape to
//! [`phaser_velo`](crate::cellular::transport::phaser_velo):
//!
//! - `aiperf.dataset.subscribe` (unary, cell → controller): the cell sends its
//!   `PeerInfo`; the controller atomically attaches a broadcast consumer, returns
//!   the snapshot, and spawns a pump forwarding the tail.
//! - `aiperf.dataset.chunk` (fire-and-forget, controller → cell): one pushed chunk event.
//!
//! The payload is opaque `Vec<u8>` (the cell decodes its own request shape).
//!
//! # Trust boundary
//!
//! The local benchmark deployment trusts the controller/cell routing plane: raw Velo
//! controller pushes provide no per-push authenticity or replay rejection. Generation
//! ordering remains the delivery invariant, while this codec enforces its independent
//! resource bound before decoding a payload.

use std::collections::HashMap;
use std::io::Read;

use bytes::Bytes;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use velo::{Context, Handler, PeerInfo, Velo};

use crate::cellular::broadcast::{BroadcastEvent, Subscription};
use crate::cellular::dataset_session::{
    DatasetChunk, DatasetIndex, DatasetPublisher, DatasetRequest,
};
use crate::engine::cellular_registration::{
    AdmissionPurpose, CellRegistrationAuthority, CellRegistrationCredential,
};

/// The opaque dataset payload on the wire (the cell decodes its own request shape).
pub type WirePayload = Vec<u8>;

/// zstd level for the fan-out wire (matches the artifact-shipping plane's `ZSTD_LEVEL`).
const ZSTD_LEVEL: i32 = 3;

// The controller currently emits 16 requests per fan-out chunk. Velo TCP
// limits the header plus payload to 16 MiB; reserve its 11-byte preamble, the
// 22-byte ActiveMessage header, and the longest dataset handler name. This is a
// benchmark codec resource bound, not a transport-security guarantee.
const MAX_VELO_FRAME_BYTES: usize = 16 * 1024 * 1024;
const VELO_TCP_PREAMBLE_BYTES: usize = 11;
const VELO_ACTIVE_MESSAGE_FIXED_HEADER_BYTES: usize = 22;
const MAX_DATASET_HANDLER_BYTES: usize = "aiperf.dataset.subscribe".len();

fn dataset_compressed_limit() -> anyhow::Result<usize> {
    MAX_VELO_FRAME_BYTES
        .checked_sub(VELO_TCP_PREAMBLE_BYTES)
        .and_then(|limit| limit.checked_sub(VELO_ACTIVE_MESSAGE_FIXED_HEADER_BYTES))
        .and_then(|limit| limit.checked_sub(MAX_DATASET_HANDLER_BYTES))
        .ok_or_else(|| anyhow::anyhow!("dataset Velo payload limit underflow"))
}

fn dataset_wire_output_limit() -> anyhow::Result<usize> {
    let compressed_limit = dataset_compressed_limit()?;
    let mut lower = 0;
    let mut upper = compressed_limit;
    while lower < upper {
        let span = upper
            .checked_sub(lower)
            .ok_or_else(|| anyhow::anyhow!("dataset wire output limit overflow"))?;
        let midpoint = lower
            .checked_add(span / 2)
            .and_then(|midpoint| midpoint.checked_add(span % 2))
            .ok_or_else(|| anyhow::anyhow!("dataset wire output limit overflow"))?;
        if zstd::zstd_safe::compress_bound(midpoint) <= compressed_limit {
            lower = midpoint;
        } else {
            upper = midpoint
                .checked_sub(1)
                .ok_or_else(|| anyhow::anyhow!("dataset wire output limit underflow"))?;
        }
    }
    Ok(lower)
}

/// Serialize `value` as MessagePack then zstd-compress it — the fan-out wire form. The
/// dataset broadcast replays whole chunks to every cell, so compressing the redundant
/// request bodies (same model/structure, only content differs) is a real win over the
/// uncompressed rmp the phaser control plane can use (its events are tiny).
fn zpack<T: Serialize>(value: &T) -> anyhow::Result<Vec<u8>> {
    let packed = rmp_serde::to_vec(value)
        .map_err(|error| anyhow::anyhow!("encode dataset wire value: {error}"))?;
    if packed.len() > dataset_wire_output_limit()? {
        anyhow::bail!("dataset wire value exceeds the protocol output limit");
    }
    let compressed = zstd::encode_all(packed.as_slice(), ZSTD_LEVEL)
        .map_err(|error| anyhow::anyhow!("zstd-compress dataset wire value: {error}"))?;
    if compressed.len() > dataset_compressed_limit()? {
        anyhow::bail!("dataset wire value exceeds Velo payload capacity");
    }
    Ok(compressed)
}

/// Inverse of [`zpack`]: zstd-decompress then MessagePack-decode.
fn zunpack<T: serde::de::DeserializeOwned>(bytes: &[u8]) -> anyhow::Result<T> {
    zunpack_with_limit(bytes, dataset_wire_output_limit()?)
}

fn zunpack_with_limit<T: serde::de::DeserializeOwned>(
    bytes: &[u8],
    max_output_bytes: usize,
) -> anyhow::Result<T> {
    let decoder = zstd::stream::read::Decoder::new(bytes)
        .map_err(|error| anyhow::anyhow!("zstd-decompress dataset wire value: {error}"))?;
    let read_limit = u64::try_from(max_output_bytes)
        .map_err(|_| anyhow::anyhow!("dataset wire output limit does not fit u64"))?
        .checked_add(1)
        .ok_or_else(|| anyhow::anyhow!("dataset wire output limit overflow"))?;
    let initial_capacity = max_output_bytes.min(64 * 1024);
    let mut packed = Vec::with_capacity(initial_capacity);
    decoder
        .take(read_limit)
        .read_to_end(&mut packed)
        .map_err(|error| anyhow::anyhow!("zstd-decompress dataset wire value: {error}"))?;
    if packed.len() > max_output_bytes {
        anyhow::bail!("dataset wire value exceeds the protocol output limit");
    }
    rmp_serde::from_slice(&packed)
        .map_err(|error| anyhow::anyhow!("decode dataset wire value: {error}"))
}

/// Handler: a cell subscribes to the dataset broadcast and gets the replay snapshot.
pub const HANDLER_DATASET_SUBSCRIBE: &str = "aiperf.dataset.subscribe";
/// Handler: the controller pushes one live dataset chunk to a subscribed cell.
pub const HANDLER_DATASET_CHUNK: &str = "aiperf.dataset.chunk";

#[derive(Serialize, Deserialize)]
struct DatasetSubscribeRequest {
    cell_id: u32,
}

#[derive(Serialize, Deserialize)]
struct DatasetSubscribeReply {
    replay: Vec<BroadcastEvent<DatasetChunk<WirePayload>>>,
}

#[derive(Serialize, Deserialize)]
enum DatasetWireEvent {
    Item(DatasetChunk<WirePayload>),
    Finalized,
    Failed(String),
}

async fn build_owned_wire_index(
    replay: Vec<BroadcastEvent<DatasetChunk<WirePayload>>>,
    mut live: mpsc::UnboundedReceiver<DatasetWireEvent>,
    owns: impl Fn(u64) -> bool,
) -> anyhow::Result<DatasetIndex<WirePayload>> {
    let mut owned = HashMap::new();
    let mut accept = |event: BroadcastEvent<DatasetChunk<WirePayload>>| {
        if let BroadcastEvent::Item(chunk) = event {
            for request in chunk.requests {
                if owns(request.request_id) {
                    owned.insert(request.request_id, request.payload);
                }
            }
        }
    };
    for event in replay {
        if matches!(event, BroadcastEvent::Finalized) {
            return Ok(DatasetIndex::from_owned(owned));
        }
        accept(event);
    }
    while let Some(event) = live.recv().await {
        match event {
            DatasetWireEvent::Item(chunk) => accept(BroadcastEvent::Item(chunk)),
            DatasetWireEvent::Finalized => return Ok(DatasetIndex::from_owned(owned)),
            DatasetWireEvent::Failed(error) => anyhow::bail!(error),
        }
    }
    anyhow::bail!("dataset Velo live stream ended before finalization")
}

/// Controller-owned admission boundary for Velo request fan-out replay.
pub(crate) struct DatasetWirePublisher {
    publisher: DatasetPublisher<WirePayload>,
    replay: Vec<BroadcastEvent<DatasetChunk<WirePayload>>>,
}

impl DatasetWirePublisher {
    /// Create an empty checked publisher.
    pub(crate) fn new() -> Self {
        Self {
            publisher: DatasetPublisher::new(),
            replay: Vec::new(),
        }
    }

    /// Clone the publisher for the Velo service.
    pub(crate) fn publisher(&self) -> DatasetPublisher<WirePayload> {
        self.publisher.clone()
    }

    /// Number of accepted chunks.
    pub(crate) fn chunk_count(&self) -> u64 {
        self.publisher.chunk_count()
    }

    /// Admit a live event and its complete candidate replay before mutation.
    pub(crate) fn add(
        &mut self,
        requests: Vec<DatasetRequest<WirePayload>>,
    ) -> anyhow::Result<u64> {
        let chunk = DatasetChunk {
            chunk_id: self.publisher.chunk_count(),
            requests,
        };
        let event = BroadcastEvent::Item(chunk.clone());
        zpack(&event)?;
        let mut replay = self.replay.clone();
        replay.push(event.clone());
        zpack(&DatasetSubscribeReply { replay })?;

        let chunk_id = self.publisher.add(chunk.requests);
        self.replay.push(event);
        Ok(chunk_id)
    }

    /// Seal only when the terminal replay reply fits too.
    pub(crate) fn finalize(&self) -> anyhow::Result<()> {
        let mut replay = self.replay.clone();
        replay.push(BroadcastEvent::Finalized);
        zpack(&DatasetSubscribeReply { replay })?;
        self.publisher.finalize();
        Ok(())
    }

    /// Consume this boundary after the controller has sealed it.
    pub(crate) fn into_publisher(self) -> DatasetPublisher<WirePayload> {
        self.publisher
    }
}

/// Controller-side dataset service. Holds the velo instance so the handler + per-cell
/// pumps outlive it.
pub struct DatasetServer {
    _velo: std::sync::Arc<Velo>,
}

impl DatasetServer {
    /// Register the `subscribe` handler on `velo`, serving the given `publisher`. Each
    /// subscribing cell gets the current replay and a pump forwarding the live tail.
    pub(crate) fn bind(
        velo: std::sync::Arc<Velo>,
        publisher: DatasetPublisher<WirePayload>,
        registration_authority: std::sync::Arc<CellRegistrationAuthority>,
    ) -> anyhow::Result<Self> {
        let push_velo = velo.clone();
        velo.register_handler(
            Handler::unary_handler_async(HANDLER_DATASET_SUBSCRIBE, move |ctx: Context| {
                let publisher = publisher.clone();
                let push_velo = push_velo.clone();
                let registration_authority = registration_authority.clone();
                async move {
                    let opened = registration_authority
                        .open_payload::<DatasetSubscribeRequest>(
                            AdmissionPurpose::DatasetSubscribe,
                            &ctx.payload,
                        )
                        .map_err(|_| anyhow::anyhow!("AdmissionRejected"))?;
                    let (role, _, authenticated_peer, request) = opened.into_parts();
                    anyhow::ensure!(
                        role == crate::engine::cellular_bootstrap::CellularRole::Cell(
                            request.cell_id,
                        ),
                        "AdmissionRejected"
                    );
                    let peer: PeerInfo = rmp_serde::from_slice(&authenticated_peer)
                        .map_err(|_| anyhow::anyhow!("AdmissionRejected"))?;
                    let cell = peer.instance_id();
                    push_velo
                        .register_peer(peer)
                        .map_err(|error| anyhow::anyhow!("register_peer cell: {error}"))?;

                    let Subscription { replay, mut live } = publisher.attach_raw();

                    let pump_velo = push_velo.clone();
                    tokio::spawn(async move {
                        while let Some(event) = live.recv().await {
                            let terminal = matches!(event, BroadcastEvent::Finalized);
                            let wire_event = match event {
                                BroadcastEvent::Item(chunk) => DatasetWireEvent::Item(chunk),
                                BroadcastEvent::Finalized => DatasetWireEvent::Finalized,
                            };
                            let body = match zpack(&wire_event) {
                                Ok(body) => body,
                                Err(error) => {
                                    let failed = DatasetWireEvent::Failed(error.to_string());
                                    let Ok(body) = zpack(&failed) else {
                                        break;
                                    };
                                    if let Ok(builder) = pump_velo.am_send(HANDLER_DATASET_CHUNK) {
                                        let _ = builder
                                            .raw_payload(Bytes::from(body))
                                            .instance(cell)
                                            .send()
                                            .await;
                                    }
                                    break;
                                }
                            };
                            let sent = match pump_velo.am_send(HANDLER_DATASET_CHUNK) {
                                Ok(builder) => {
                                    builder
                                        .raw_payload(Bytes::from(body))
                                        .instance(cell)
                                        .send()
                                        .await
                                }
                                Err(_) => break,
                            };
                            if sent.is_err() || terminal {
                                break;
                            }
                        }
                    });

                    let reply = DatasetSubscribeReply { replay };
                    let bytes = zpack(&reply)?;
                    Ok(Some(Bytes::from(bytes)))
                }
            })
            .build(),
        )
        .map_err(|error| anyhow::anyhow!("registering dataset subscribe handler: {error}"))?;
        Ok(Self { _velo: velo })
    }
}

/// Cell-side dataset client.
pub struct DatasetClient;

impl DatasetClient {
    /// Subscribe to the controller's dataset broadcast and build this cell's owned index
    /// (the requests where `owns(request_id)`), draining to `finalize`. RAM is O(owned)
    /// even though every chunk is observed.
    pub(crate) async fn build_owned_index(
        velo: std::sync::Arc<Velo>,
        controller: &PeerInfo,
        cell_id: u32,
        credential: &CellRegistrationCredential,
        owns: impl Fn(u64) -> bool,
    ) -> anyhow::Result<DatasetIndex<WirePayload>> {
        let (tx, live) = mpsc::unbounded_channel::<DatasetWireEvent>();
        velo.register_handler(
            Handler::am_handler_async(HANDLER_DATASET_CHUNK, move |ctx: Context| {
                let tx = tx.clone();
                async move {
                    let event: DatasetWireEvent = zunpack(&ctx.payload)?;
                    let _ = tx.send(event);
                    Ok(())
                }
            })
            .build(),
        )
        .map_err(|error| anyhow::anyhow!("registering dataset chunk handler: {error}"))?;

        velo.register_peer(controller.clone())
            .map_err(|error| anyhow::anyhow!("register_peer controller: {error}"))?;

        if credential.cell_id() != cell_id {
            anyhow::bail!("dataset credential does not match the cell identity");
        }
        let request = DatasetSubscribeRequest { cell_id };
        let body = credential.seal_payload(
            AdmissionPurpose::DatasetSubscribe,
            &velo.peer_info(),
            &request,
        )?;
        let reply_bytes: Bytes = velo
            .unary(HANDLER_DATASET_SUBSCRIBE)
            .map_err(|error| anyhow::anyhow!("dataset subscribe builder: {error}"))?
            .raw_payload(Bytes::from(body))
            .instance(controller.instance_id())
            .send()
            .await
            .map_err(|error| anyhow::anyhow!("dataset subscribe send: {error}"))?;
        let reply: DatasetSubscribeReply = zunpack(&reply_bytes)?;

        build_owned_wire_index(reply.replay, live, owns).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cellular::dataset_session::DatasetRequest;
    use crate::cellular::transport::connect::{BindSpec, build_velo};
    use crate::engine::cellular_registration::CellRegistrationAuthority;

    #[test]
    fn zunpack_rejects_expansion_beyond_protocol_cap() {
        let packed = rmp_serde::to_vec(&vec![0_u8; 62]).expect("encode fixture");
        assert_eq!(packed.len(), 65, "fixture must exceed the supplied cap");
        let compressed = zstd::encode_all(packed.as_slice(), ZSTD_LEVEL).expect("compress fixture");

        let error = zunpack_with_limit::<Vec<u8>>(&compressed, 64).expect_err("must reject");

        assert!(error.to_string().contains("dataset wire value exceeds"));
    }

    #[test]
    fn zunpack_accepts_protocol_maximum() {
        let value = vec![0_u8; 61];
        let packed = rmp_serde::to_vec(&value).expect("encode fixture");
        assert_eq!(
            packed.len(),
            64,
            "fixture must exactly reach the supplied cap"
        );
        let compressed = zstd::encode_all(packed.as_slice(), ZSTD_LEVEL).expect("compress fixture");

        assert_eq!(
            zunpack_with_limit::<Vec<u8>>(&compressed, 64).expect("round trip"),
            value
        );
    }

    fn deterministic_incompressible_bytes(len: usize) -> Vec<u8> {
        let mut state = 0x4d59_5df4_d0f3_3173_u64;
        (0..len)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                state as u8
            })
            .collect()
    }

    struct BinaryPayload<'a>(&'a [u8]);

    impl Serialize for BinaryPayload<'_> {
        fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where
            S: serde::Serializer,
        {
            serializer.serialize_bytes(self.0)
        }
    }

    #[test]
    fn zpack_rejects_incompressible_value_at_old_logical_limit() {
        let old_limit = MAX_VELO_FRAME_BYTES;
        let value = deterministic_incompressible_bytes(old_limit - 5);
        assert_eq!(
            rmp_serde::to_vec(&BinaryPayload(&value))
                .expect("encode exact old-limit fixture")
                .len(),
            old_limit
        );

        let error = zpack(&BinaryPayload(&value)).expect_err("must reject before publish");

        assert!(error.to_string().contains("protocol output limit"));
    }

    #[test]
    fn zpack_round_trips_incompressible_value_at_safe_logical_limit() {
        let value = deterministic_incompressible_bytes(
            dataset_wire_output_limit().expect("safe limit") - 5,
        );

        let compressed = zpack(&BinaryPayload(&value)).expect("safe value must publish");

        assert_eq!(zunpack::<Vec<u8>>(&compressed).expect("round trip"), value);
    }

    #[test]
    fn zpack_rejects_incompressible_value_beyond_safe_logical_limit() {
        let value = deterministic_incompressible_bytes(
            dataset_wire_output_limit().expect("safe limit") - 4,
        );

        let error = zpack(&BinaryPayload(&value)).expect_err("must reject before publish");

        assert!(error.to_string().contains("protocol output limit"));
    }

    #[test]
    fn oversized_producer_chunk_does_not_advance_history() {
        let mut publisher = DatasetWirePublisher::new();
        let subscription = publisher.publisher().attach_raw();
        let oversized = vec![DatasetRequest {
            request_id: 0,
            payload: vec![u8::MAX; dataset_wire_output_limit().expect("safe limit") / 2],
        }];

        let error = publisher
            .add(oversized)
            .expect_err("must reject oversized chunk");

        assert!(error.to_string().contains("protocol output limit"));
        assert_eq!(publisher.chunk_count(), 0);
        assert!(subscription.replay.is_empty());
        assert!(subscription.live.is_empty());
    }

    #[test]
    fn checked_publisher_rejects_replay_growth_without_advancing_history() {
        let mut publisher = DatasetWirePublisher::new();
        let payload = vec![u8::MAX; dataset_wire_output_limit().expect("safe limit") / 4];

        publisher
            .add(vec![DatasetRequest {
                request_id: 0,
                payload: payload.clone(),
            }])
            .expect("first event fits");
        let error = publisher
            .add(vec![DatasetRequest {
                request_id: 1,
                payload,
            }])
            .expect_err("complete replay must reject");

        assert!(error.to_string().contains("protocol output limit"));
        assert_eq!(publisher.chunk_count(), 1);
    }

    #[tokio::test]
    async fn failed_live_wire_event_returns_promptly() {
        let (tx, rx) = mpsc::unbounded_channel();
        tx.send(DatasetWireEvent::Failed("controller encode failed".into()))
            .expect("test receiver is live");

        let result = tokio::time::timeout(
            std::time::Duration::from_millis(100),
            build_owned_wire_index(Vec::new(), rx, |_| true),
        )
        .await
        .expect("failed event must not hang");
        let Err(error) = result else {
            panic!("failed event must reach the cell");
        };

        assert!(error.to_string().contains("controller encode failed"));
    }

    #[tokio::test]
    async fn cells_build_disjoint_owned_indexes_over_velo() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind");
        let controller_velo = build_velo(BindSpec::TcpListener(listener))
            .await
            .expect("controller velo");
        let publisher = DatasetPublisher::<WirePayload>::new();
        let (authority, credentials) = CellRegistrationAuthority::mint(3).expect("authority");
        let _server = DatasetServer::bind(
            controller_velo.clone(),
            publisher.clone(),
            std::sync::Arc::new(authority),
        )
        .expect("bind");

        // Publish 12 requests in two chunks, then finalize.
        let chunk = |ids: std::ops::Range<u64>| {
            ids.map(|request_id| DatasetRequest {
                request_id,
                payload: format!("req-{request_id}").into_bytes(),
            })
            .collect::<Vec<_>>()
        };
        publisher.add(chunk(0..6));
        publisher.add(chunk(6..12));
        publisher.finalize();

        // Two cells (of 3) build their owned indexes over velo.
        let mut owned_all = Vec::new();
        for cell_id in [0u64, 1] {
            let cell_velo = build_velo(BindSpec::TcpLoopback).await.expect("cell velo");
            let controller = controller_velo.peer_info();
            let index = DatasetClient::build_owned_index(
                cell_velo,
                &controller,
                cell_id as u32,
                &credentials[cell_id as usize],
                move |id| id % 3 == cell_id,
            )
            .await
            .expect("build index");
            let expected: Vec<u64> = (0..12).filter(|id| id % 3 == cell_id).collect();
            assert_eq!(
                index.owned_ids(),
                expected,
                "cell {cell_id} owned set over velo"
            );
            assert_eq!(
                index
                    .get(cell_id)
                    .map(|p| String::from_utf8_lossy(p).into_owned()),
                Some(format!("req-{cell_id}"))
            );
            owned_all.extend(index.owned_ids());
        }
        assert_eq!(
            owned_all.len(),
            8,
            "cells 0 and 1 of 3 own 4 each, disjoint"
        );
    }
}
