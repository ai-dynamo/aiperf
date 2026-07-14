// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `aiperf-mock-server` binary - Rust rewrite of the Python mock server.

use std::net::SocketAddr;

use aiperf_mock_server::{MockServerConfig, build_router};
use clap::Parser;
use hyper_util::rt::{TokioExecutor, TokioIo};
use hyper_util::server::conn::auto::Builder as ConnBuilder;
use socket2::{Domain, Protocol, Socket, Type};
use tower::Service;
use tracing_subscriber::EnvFilter;
use tracing_subscriber::filter::LevelFilter;

/// Listen backlog. Large enough that a C10K burst of SYNs doesn't get dropped;
/// kernel silently clamps to /proc/sys/net/core/somaxconn (4096 on modern Linux).
const LISTEN_BACKLOG: i32 = 16_384;

fn main() -> anyhow::Result<()> {
    let config = MockServerConfig::parse().apply_flags();

    init_tracing(&config.log_level);

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
    let router = build_router(state);

    tracing::info!(%addr, backlog = LISTEN_BACKLOG, "Listening");
    let listener = build_listener(addr)?;

    // Manual accept loop — enables TCP_NODELAY on every accepted socket (axum's
    // default `serve` leaves it off, which defeated our streaming throughput
    // because Nagle's algorithm was holding small SSE chunks for ~40 ms). Each
    // connection gets its own tokio task driving hyper's auto HTTP/1+2
    // handshake.
    let make_service = router.into_make_service();
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
            if let Err(e) = ConnBuilder::new(TokioExecutor::new())
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

fn build_listener(addr: SocketAddr) -> anyhow::Result<tokio::net::TcpListener> {
    let domain = if addr.is_ipv4() {
        Domain::IPV4
    } else {
        Domain::IPV6
    };
    let socket = Socket::new(domain, Type::STREAM, Some(Protocol::TCP))?;
    socket.set_nonblocking(true)?;
    socket.set_reuse_address(true)?;
    // SO_REUSEPORT on Linux lets multiple sockets bind the same port so the
    // kernel load-balances accepts across them. Safe fallback if unsupported.
    #[cfg(target_os = "linux")]
    let _ = socket.set_reuse_port(true);
    socket.bind(&addr.into())?;
    socket.listen(LISTEN_BACKLOG)?;
    let std_listener: std::net::TcpListener = socket.into();
    let listener = tokio::net::TcpListener::from_std(std_listener)?;
    Ok(listener)
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
