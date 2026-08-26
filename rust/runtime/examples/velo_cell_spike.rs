// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Discovery-free Velo peer registration between two local instances.
//!
//! The cell registers the controller's `PeerInfo` and sends its own `PeerInfo`
//! through the typed registration handler so replies can route back.
//!
//! Run: `cargo run -p aiperf-runtime --features cellular --example velo_cell_spike`

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
    let controller = build_tcp_velo().await?;
    controller
        .register_handler(
            Handler::typed_unary_async(
                "aiperf.cell.register",
                |ctx: TypedContext<Register>| async move {
                    // Register the advertised cell identity before replying.
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

    let controller_peer_bytes =
        rmp_serde::to_vec(&controller.peer_info()).context("encode controller PeerInfo")?;

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
    println!("registration reply: {reply:?}");
    Ok(())
}
