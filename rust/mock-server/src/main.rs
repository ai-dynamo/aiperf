// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `aiperf-mock-server` process entry point.

use std::net::SocketAddr;

use aiperf_mock_server::listener::{LISTEN_BACKLOG, build_listener};
use aiperf_mock_server::{MockServerConfig, balancer, build_router, tls};
use clap::Parser;
use tracing_subscriber::EnvFilter;
use tracing_subscriber::filter::LevelFilter;

fn main() -> anyhow::Result<()> {
    let config = load_config()?;

    init_tracing(&config.log_level);

    // `0` selects one child per available CPU; `1` serves directly.
    let processes = if config.processes == 0 {
        num_cpus::get().max(1)
    } else {
        config.processes
    };
    if processes > 1 {
        // The HTTP-only balancer cannot splice the separate gRPC or UDS listeners.
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

    // HTTP and gRPC share the certificate and ALPN policy.
    let acceptor = tls::build_acceptor(&config)?;
    if acceptor.is_some() {
        let mode = if config.tls_self_signed && config.tls_cert.is_none() {
            "self-signed"
        } else {
            "cert/key"
        };
        tracing::info!(%addr, tls = mode, "TLS enabled (ALPN h2 + http/1.1)");
    }

    // The gRPC listener shares state and TLS configuration with HTTP.
    if let Some(grpc_port) = config.grpc_port {
        let grpc_addr = SocketAddr::new(host, grpc_port);
        let grpc_state = state.clone();
        let grpc_acceptor = acceptor.clone();
        tokio::spawn(async move {
            if let Err(error) =
                aiperf_mock_server::grpc::serve_grpc_with_tls(grpc_addr, grpc_state, grpc_acceptor)
                    .await
            {
                tracing::error!(%grpc_addr, "gRPC server exited: {error}");
            }
        });
    }

    let router = build_router(state);

    // The UDS transport requires HTTP/1.1.
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

    tls::serve_http(listener, router, acceptor, config.max_concurrent_streams).await
}

/// Load CLI configuration or the fully resolved child-process configuration.
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
