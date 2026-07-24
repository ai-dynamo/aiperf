// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! io_uring thread-per-core HTTP engine for the request hot path (`--uring`).
//!
//! One monoio io_uring runtime per core, each with its own `SO_REUSEPORT`
//! listener, running the real non-streaming chat path with no cross-core
//! scheduling. Request framing/routing is shared with the blocking engine via
//! [`crate::http_core`]. Implies `--fast` semantics. See `http_core` for the
//! served routes.

use std::sync::Arc;

use monoio::io::{AsyncReadRent, AsyncWriteRentExt};
use monoio::net::{ListenerOpts, TcpListener, TcpStream};

use crate::config::MockServerConfig;
use crate::http_core::{build_engine_state, parse_head, route};
use crate::state::AppState;

/// Launch one io_uring runtime per core, each serving the shared `AppState`.
pub fn run(config: &MockServerConfig) -> anyhow::Result<()> {
    let cores = if config.workers > 0 {
        config.workers
    } else {
        num_cpus::get().max(1)
    };
    let host = config.host.clone();
    let port = config.port;
    let state = build_engine_state(config);

    tracing::info!(
        %host, port, cores,
        "Starting AIPerf Mock Server (io_uring engine); non-streaming chat path, --fast semantics"
    );

    let mut handles = Vec::with_capacity(cores);
    for _ in 0..cores {
        let state = state.clone();
        let host = host.clone();
        handles.push(std::thread::spawn(move || {
            let mut rt = monoio::RuntimeBuilder::<monoio::IoUringDriver>::new()
                .with_entries(4096)
                .build()
                .expect("build io_uring runtime");
            rt.block_on(serve(host, port, state));
        }));
    }
    for h in handles {
        let _ = h.join();
    }
    Ok(())
}

async fn serve(host: String, port: u16, state: Arc<AppState>) {
    let addr = format!("{host}:{port}");
    let opts = ListenerOpts::default().reuse_port(true).reuse_addr(true);
    let listener = match TcpListener::bind_with_config(addr.as_str(), &opts) {
        Ok(l) => l,
        Err(e) => {
            tracing::error!(%addr, "io_uring bind failed: {e}");
            return;
        }
    };
    loop {
        match listener.accept().await {
            Ok((stream, _)) => {
                let _ = stream.set_nodelay(true);
                monoio::spawn(handle_conn(stream, state.clone()));
            }
            Err(_) => continue,
        }
    }
}

async fn handle_conn(mut stream: TcpStream, state: Arc<AppState>) {
    let mut acc: Vec<u8> = Vec::with_capacity(16384);
    let mut buf: Vec<u8> = vec![0u8; 65536];
    loop {
        loop {
            let head = match parse_head(&acc) {
                Ok(Some(h)) => h,
                Ok(None) => break,
                Err(_) => return,
            };
            let total = head.head_len + head.body_len;
            if acc.len() < total {
                break;
            }
            let resp = route(&state, &head, &acc);
            let close = !head.keep_alive;
            let (wres, _) = stream.write_all(resp).await;
            if wres.is_err() || close {
                return;
            }
            acc.drain(..total);
        }
        let (res, b) = stream.read(buf).await;
        buf = b;
        match res {
            Ok(0) => return,
            Ok(n) => acc.extend_from_slice(&buf[..n]),
            Err(_) => return,
        }
    }
}
