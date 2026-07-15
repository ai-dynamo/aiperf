// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Spike: prove the discovery-free cell↔controller reach used by the velo cell
//! transport, mechanism B (bootstrap PeerInfo fetch), against official velo
//! v0.5.0 — two separate `Velo` instances, no discovery backend.
//!
//! The controller knows only its own bind address; the cell obtains the
//! controller's real `PeerInfo` out-of-band (here handed directly, simulating a
//! bootstrap fetch from the operator-hardcoded coordinate), `register_peer`s it,
//! and calls the `register` typed-unary handler carrying its OWN `peer_info()` so
//! the controller can reach it back. Success = the register reply round-trips.
//!
//! Run: `cargo run -p aiperf --features velo --example velo_cell_spike`

use std::sync::Arc;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use velo::transports::tcp::TcpTransportBuilder;
use velo::{Handler, TypedContext, Velo};

/// The cell's registration payload: its `cell_id` plus its own serialized
/// `PeerInfo` (rmp) so the controller can `register_peer` and reach it back.
#[derive(Serialize, Deserialize, Clone)]
struct Register {
    cell_id: u32,
    cell_peer: Vec<u8>,
}

/// Stand-in for the `CellLaunchSpec` reply.
#[derive(Serialize, Deserialize, Clone, Debug)]
struct SpecReply {
    cell_id: u32,
    ok: bool,
}

async fn build_tcp_velo() -> Result<Arc<Velo>> {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").context("bind loopback")?;
    let transport = Arc::new(
        TcpTransportBuilder::new()
            .from_listener(listener)
            .context("from_listener")?
            .build()
            .context("tcp build")?,
    );
    Velo::builder()
        .add_transport(transport)
        .build()
        .await
        .context("velo build")
}

#[tokio::main(flavor = "multi_thread", worker_threads = 2)]
async fn main() -> Result<()> {
    // --- controller: knows only its own bind address ---
    let controller = build_tcp_velo().await?;
    controller
        .register_handler(
            Handler::typed_unary_async(
                "aiperf.cell.register",
                |ctx: TypedContext<Register>| async move {
                    // Learn the cell's real PeerInfo from the payload so replies route back.
                    let peer: velo::PeerInfo = rmp_serde::from_slice(&ctx.input.cell_peer)
                        .context("decode cell PeerInfo")?;
                    ctx.msg.register_peer(peer).context("register_peer cell")?;
                    Ok(SpecReply {
                        cell_id: ctx.input.cell_id,
                        ok: true,
                    })
                },
            )
            .build(),
        )
        .context("register handler")?;

    // Mechanism B: the controller's REAL PeerInfo, transferred out-of-band.
    // In production the controller serves these bytes at its hardcoded bootstrap
    // coordinate; here we serialize/deserialize directly to prove the velo path.
    let controller_peer_bytes =
        rmp_serde::to_vec(&controller.peer_info()).context("encode controller PeerInfo")?;

    // --- cell: knows only the controller's PeerInfo bytes (the "bootstrap") ---
    let cell = build_tcp_velo().await?;
    let controller_peer: velo::PeerInfo =
        rmp_serde::from_slice(&controller_peer_bytes).context("decode controller PeerInfo")?;
    cell.register_peer(controller_peer.clone())
        .context("register_peer controller")?;

    let reply: SpecReply = cell
        .typed_unary::<SpecReply>("aiperf.cell.register")
        .context("build typed_unary")?
        .payload(&Register {
            cell_id: 7,
            cell_peer: rmp_serde::to_vec(&cell.peer_info()).context("encode cell PeerInfo")?,
        })
        .context("attach payload")?
        .instance(controller_peer.instance_id())
        .send()
        .await
        .context("register send")?;

    assert!(
        reply.ok && reply.cell_id == 7,
        "unexpected reply: {reply:?}"
    );
    println!("SPIKE RESULT: mechanism B (bootstrap-peerinfo) WORKS — reply {reply:?}");
    Ok(())
}
