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

use std::io::Read;

use bytes::Bytes;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use velo::{Context, Handler, PeerInfo, Velo};

use crate::cellular::broadcast::{BroadcastEvent, Subscription};
use crate::cellular::dataset_session::{DatasetChunk, DatasetIndex, DatasetPublisher};
use crate::engine::cellular_registration::{
    AdmissionPurpose, CellRegistrationAuthority, CellRegistrationCredential,
};

/// The opaque dataset payload on the wire (the cell decodes its own request shape).
pub type WirePayload = Vec<u8>;

/// zstd level for the fan-out wire (matches the artifact-shipping plane's `ZSTD_LEVEL`).
const ZSTD_LEVEL: i32 = 3;

// The controller currently emits 16 requests per fan-out chunk. Velo's TCP framing
// defaults to a 16 MiB maximum frame; use that as this benchmark protocol's logical
// codec cap, not as a claim that every transport enforces it. The codec applies the
// same cap before compression and after bounded decompression.
const MAX_DATASET_CHUNK_REQUESTS: u64 = 16;
const MAX_VELO_FRAME_BYTES: u64 = 16 * 1024 * 1024;
const MAX_DATASET_REQUEST_PAYLOAD_BYTES: u64 = MAX_VELO_FRAME_BYTES / MAX_DATASET_CHUNK_REQUESTS;

// Largest MessagePack dataset wire value accepted by this fan-out protocol.
const MAX_DATASET_WIRE_OUTPUT_BYTES: u64 =
    MAX_DATASET_REQUEST_PAYLOAD_BYTES * MAX_DATASET_CHUNK_REQUESTS;

fn dataset_wire_output_limit() -> anyhow::Result<u64> {
    MAX_DATASET_REQUEST_PAYLOAD_BYTES
        .checked_mul(MAX_DATASET_CHUNK_REQUESTS)
        .ok_or_else(|| anyhow::anyhow!("dataset wire output limit overflow"))
}

/// Serialize `value` as MessagePack then zstd-compress it — the fan-out wire form. The
/// dataset broadcast replays whole chunks to every cell, so compressing the redundant
/// request bodies (same model/structure, only content differs) is a real win over the
/// uncompressed rmp the phaser control plane can use (its events are tiny).
fn zpack<T: Serialize>(value: &T) -> anyhow::Result<Vec<u8>> {
    let packed = rmp_serde::to_vec(value)
        .map_err(|error| anyhow::anyhow!("encode dataset wire value: {error}"))?;
    let packed_len = u64::try_from(packed.len())
        .map_err(|_| anyhow::anyhow!("dataset wire value length does not fit u64"))?;
    if packed_len > dataset_wire_output_limit()? {
        anyhow::bail!("dataset wire value exceeds the protocol output limit");
    }
    zstd::encode_all(packed.as_slice(), ZSTD_LEVEL)
        .map_err(|error| anyhow::anyhow!("zstd-compress dataset wire value: {error}"))
}

/// Inverse of [`zpack`]: zstd-decompress then MessagePack-decode.
fn zunpack<T: serde::de::DeserializeOwned>(bytes: &[u8]) -> anyhow::Result<T> {
    zunpack_with_limit(bytes, MAX_DATASET_WIRE_OUTPUT_BYTES)
}

fn zunpack_with_limit<T: serde::de::DeserializeOwned>(
    bytes: &[u8],
    max_output_bytes: u64,
) -> anyhow::Result<T> {
    let decoder = zstd::stream::read::Decoder::new(bytes)
        .map_err(|error| anyhow::anyhow!("zstd-decompress dataset wire value: {error}"))?;
    let read_limit = max_output_bytes
        .checked_add(1)
        .ok_or_else(|| anyhow::anyhow!("dataset wire output limit overflow"))?;
    let initial_capacity = usize::try_from(max_output_bytes.min(64 * 1024))
        .map_err(|_| anyhow::anyhow!("dataset wire output limit does not fit usize"))?;
    let mut packed = Vec::with_capacity(initial_capacity);
    decoder
        .take(read_limit)
        .read_to_end(&mut packed)
        .map_err(|error| anyhow::anyhow!("zstd-decompress dataset wire value: {error}"))?;
    let packed_len = u64::try_from(packed.len())
        .map_err(|_| anyhow::anyhow!("dataset wire value length does not fit u64"))?;
    if packed_len > max_output_bytes {
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
                            let Ok(body) = zpack(&event) else {
                                break;
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
        let (tx, live) = mpsc::unbounded_channel::<BroadcastEvent<DatasetChunk<WirePayload>>>();
        velo.register_handler(
            Handler::am_handler_async(HANDLER_DATASET_CHUNK, move |ctx: Context| {
                let tx = tx.clone();
                async move {
                    let event: BroadcastEvent<DatasetChunk<WirePayload>> = zunpack(&ctx.payload)?;
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

        let sub = Subscription {
            replay: reply.replay,
            live,
        };
        Ok(DatasetIndex::build_owned(sub, owns).await)
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

    #[test]
    fn zpack_rejects_wire_value_beyond_protocol_cap() {
        let bytes =
            vec![0_u8; usize::try_from(MAX_DATASET_WIRE_OUTPUT_BYTES).expect("cap fits usize")];

        let error = zpack(&bytes).expect_err("must reject");

        assert!(error.to_string().contains("dataset wire value exceeds"));
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
