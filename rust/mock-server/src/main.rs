// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `aiperf-mock-server` binary - Rust rewrite of the Python mock server.

use std::net::SocketAddr;

use aiperf_mock_server::listener::{LISTEN_BACKLOG, build_listener};
use aiperf_mock_server::{MockServerConfig, balancer, build_router};
use clap::Parser;
use hyper_util::rt::{TokioExecutor, TokioIo};
use hyper_util::server::conn::auto::Builder as ConnBuilder;
use tower::Service;
use tracing_subscriber::EnvFilter;
use tracing_subscriber::filter::LevelFilter;

fn main() -> anyhow::Result<()> {
    let config = load_config()?;

    init_tracing(&config.log_level);

    // Multi-process mode: become a round-robin balancer over `processes` child
    // servers. `0` = auto = nproc; `1` falls through to the single-process path.
    let processes = if config.processes == 0 {
        num_cpus::get().max(1)
    } else {
        config.processes
    };
    if processes > 1 {
        // The L4 balancer is HTTP-only; the KServe gRPC listener is not spliced
        // across children (each would bind the same port). Warn and skip it.
        if config.grpc_port.is_some() {
            tracing::warn!(
                "--grpc-port is ignored with --processes > 1 (the balancer is HTTP-only); \
                 gRPC is not served in multi-process mode"
            );
        }
        if config.uds.is_some() {
            tracing::warn!(
                "--uds is ignored with --processes > 1 (the balancer is TCP-only); \
                 the Unix-domain socket listener is not served in multi-process mode"
            );
        }
        return balancer::run(config, processes);
    }

    // Build a tuned multi-threaded tokio runtime:
    //   - worker_threads = nproc by default (override via --workers)
    //   - max_blocking_threads large enough for bursty I/O
    //   - enable_all: timers + I/O drivers
    let worker_threads = if config.workers > 0 {
        config.workers
    } else {
        num_cpus::get()
    };
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .worker_threads(worker_threads)
        .max_blocking_threads(1024)
        .thread_name("aiperf-mock")
        .build()?;

    runtime.block_on(async move { serve(config, worker_threads).await })
}

async fn serve(config: MockServerConfig, worker_threads: usize) -> anyhow::Result<()> {
    tracing::info!(
        host = %config.host,
        port = config.port,
        fast = config.fast,
        workers = worker_threads,
        "Starting AIPerf Mock Server (Rust)"
    );

    if !config.no_tokenizer {
        aiperf_mock_server::tokens::load_corpus();
    }

    let host: std::net::IpAddr = config
        .host
        .parse()
        .unwrap_or(std::net::IpAddr::V4(std::net::Ipv4Addr::LOCALHOST));
    let addr = SocketAddr::new(host, config.port);
    let state = aiperf_mock_server::app::build_state(config.clone());

    // Optional KServe OIP v2 gRPC listener on its own port, sharing this run's
    // AppState (recorder/prefix-cache/scheduler) with the HTTP frontend.
    if let Some(grpc_port) = config.grpc_port {
        let grpc_addr = SocketAddr::new(host, grpc_port);
        let grpc_state = state.clone();
        tokio::spawn(async move {
            if let Err(error) = aiperf_mock_server::grpc::serve_grpc(grpc_addr, grpc_state).await {
                tracing::error!(%grpc_addr, "gRPC server exited: {error}");
            }
        });
    }

    let router = build_router(state);

    // Optional Unix-domain socket listener serving the SAME router over HTTP/1.1
    // (the runner's UDS transport is h1-only). Runs alongside the TCP frontend.
    if let Some(uds_path) = config.uds.clone() {
        let uds_router = router.clone();
        tokio::spawn(async move {
            if let Err(error) =
                aiperf_mock_server::listener::serve_router_uds(uds_router, &uds_path).await
            {
                tracing::error!(%uds_path, "UDS server exited: {error}");
            }
        });
    }

    tracing::info!(%addr, backlog = LISTEN_BACKLOG, "Listening");
    let listener = build_listener(addr)?;

    // Manual accept loop — enables TCP_NODELAY on every accepted socket (axum's
    // default `serve` leaves it off, which defeated our streaming throughput
    // because Nagle's algorithm was holding small SSE chunks for ~40 ms). Each
    // connection gets its own tokio task driving hyper's auto HTTP/1+2
    // handshake.
    let make_service = router.into_make_service();
    // 0 = leave hyper's default; otherwise advertise this h2 stream ceiling so a
    // single connection can hold very large concurrent-request counts.
    let max_concurrent_streams = config.max_concurrent_streams;
    loop {
        let (stream, peer) = match listener.accept().await {
            Ok(v) => v,
            Err(e) => {
                tracing::warn!("accept error: {e}");
                continue;
            }
        };
        // Disable Nagle for low-latency streaming.
        let _ = stream.set_nodelay(true);

        let tower_service = match make_service.clone().call(peer).await {
            Ok(svc) => svc,
            Err(e) => {
                tracing::warn!("make_service error: {e}");
                continue;
            }
        };

        tokio::spawn(async move {
            let io = TokioIo::new(stream);
            let hyper_service =
                hyper::service::service_fn(move |req: hyper::Request<hyper::body::Incoming>| {
                    tower_service.clone().call(req)
                });
            let mut builder = ConnBuilder::new(TokioExecutor::new());
            if max_concurrent_streams > 0 {
                builder
                    .http2()
                    .max_concurrent_streams(max_concurrent_streams);
            }
            if let Err(e) = builder
                .serve_connection_with_upgrades(io, hyper_service)
                .await
            {
                // A clean keep-alive idle close returns Ok(()); reaching here means
                // the peer dropped the connection abnormally (broken pipe / reset /
                // incomplete message) — exactly the client-side "Connection lost"
                // errors a benchmark counts. Surface it at WARN so it is visible at
                // the default INFO log level instead of being buried at DEBUG.
                tracing::warn!(%peer, "connection error: {e}");
            }
        });
    }
}

/// Load the effective config. A balancer-spawned child carries its exact config
/// as JSON in [`balancer::CONFIG_JSON_ENV`] — deserialize that verbatim (already
/// final, `apply_flags` was run by the parent) rather than re-parsing argv.
/// Otherwise parse the CLI/env surface and apply the `--fast`/`--verbose`
/// post-processing.
fn load_config() -> anyhow::Result<MockServerConfig> {
    if let Ok(json) = std::env::var(balancer::CONFIG_JSON_ENV) {
        let config: MockServerConfig = serde_json::from_str(&json)
            .map_err(|e| anyhow::anyhow!("invalid {}: {e}", balancer::CONFIG_JSON_ENV))?;
        return Ok(config);
    }
    Ok(MockServerConfig::parse().apply_flags())
}

fn init_tracing(level: &str) {
    let filter = EnvFilter::builder()
        .with_default_directive(level_to_filter(level).into())
        .with_env_var("AIPERF_MOCK_LOG")
        .from_env_lossy();
    let _ = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(false)
        .try_init();
}

fn level_to_filter(level: &str) -> LevelFilter {
    match level.to_ascii_uppercase().as_str() {
        "DEBUG" => LevelFilter::DEBUG,
        "INFO" => LevelFilter::INFO,
        "WARNING" | "WARN" => LevelFilter::WARN,
        "ERROR" => LevelFilter::ERROR,
        "CRITICAL" => LevelFilter::ERROR,
        _ => LevelFilter::INFO,
    }
}
