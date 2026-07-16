// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

mod common;

use std::cell::Cell;
use std::convert::Infallible;
use std::rc::Rc;
use std::time::Duration;

use aiperf_runtime::transport_http::client::connection::LocalExec;
use aiperf_runtime::transport_http::client::pool::{ConnectionManager, ConnectionPool};
use aiperf_runtime::transport_http::config::ClientConfig;
use aiperf_runtime::transport_http::models::{ConnectionReuseStrategy, HttpVersion, RequestConfig};
use aiperf_runtime::transport_http::transport::http_transport::HttpTransport;
use aiperf_runtime::transport_http::{Clock, RealClock, SimClock};
use bytes::Bytes;
use common::run_local;
use http_body_util::Full;
use hyper::service::service_fn;
use hyper::{Request, Response};
use hyper_util::rt::TokioIo;
use tokio::net::TcpListener;
use tokio::task::JoinHandle;

#[derive(Clone, Copy)]
enum ServerProtocol {
    H1,
    H2,
}

struct LoopbackServer {
    base_url: String,
    accepted: Rc<Cell<usize>>,
    active: Rc<Cell<usize>>,
    max_active: Rc<Cell<usize>>,
    accept_task: JoinHandle<()>,
}

impl LoopbackServer {
    async fn spawn(protocol: ServerProtocol, response_delay: Duration) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let accepted = Rc::new(Cell::new(0));
        let active = Rc::new(Cell::new(0));
        let max_active = Rc::new(Cell::new(0));
        let accepted_task = accepted.clone();
        let active_task = active.clone();
        let max_active_task = max_active.clone();

        let accept_task = tokio::task::spawn_local(async move {
            loop {
                let Ok((stream, _)) = listener.accept().await else {
                    break;
                };
                accepted_task.set(accepted_task.get() + 1);
                let active = active_task.clone();
                let max_active = max_active_task.clone();
                let service = service_fn(move |_request: Request<hyper::body::Incoming>| {
                    let active = active.clone();
                    let max_active = max_active.clone();
                    async move {
                        let now = active.get() + 1;
                        active.set(now);
                        max_active.set(max_active.get().max(now));
                        if !response_delay.is_zero() {
                            tokio::time::sleep(response_delay).await;
                        }
                        active.set(active.get() - 1);
                        Ok::<_, Infallible>(
                            Response::builder()
                                .header("content-type", "application/json")
                                .body(Full::new(Bytes::from_static(b"{\"ok\":true}")))
                                .unwrap(),
                        )
                    }
                });
                tokio::task::spawn_local(async move {
                    let io = TokioIo::new(stream);
                    match protocol {
                        ServerProtocol::H1 => {
                            let _ = hyper::server::conn::http1::Builder::new()
                                .serve_connection(io, service)
                                .await;
                        }
                        ServerProtocol::H2 => {
                            let _ = hyper::server::conn::http2::Builder::new(LocalExec)
                                .serve_connection(io, service)
                                .await;
                        }
                    }
                });
            }
        });

        Self {
            base_url: format!("http://{address}"),
            accepted,
            active,
            max_active,
            accept_task,
        }
    }
}

impl Drop for LoopbackServer {
    fn drop(&mut self) {
        self.accept_task.abort();
        assert_eq!(
            self.active.get(),
            0,
            "server requests must drain before drop"
        );
    }
}

fn request(base_url: &str) -> RequestConfig {
    RequestConfig::new(format!("{base_url}/health"))
}

#[test]
fn ordinary_transport_multiplexes_concurrent_h2_on_one_connection() {
    run_local(async {
        let server = LoopbackServer::spawn(ServerProtocol::H2, Duration::from_millis(20)).await;
        let clock: Rc<dyn Clock> = RealClock::new();
        let transport = Rc::new(HttpTransport::new(
            clock,
            ClientConfig {
                http_version: HttpVersion::Http2PriorKnowledge,
                max_connections_per_origin: 1,
                ..ClientConfig::default()
            },
        ));
        let config = request(&server.base_url);

        let mut tasks = Vec::new();
        for _ in 0..24 {
            let transport = transport.clone();
            let config = config.clone();
            tasks.push(tokio::task::spawn_local(async move {
                transport.get(&config).await
            }));
        }

        let mut reused = 0;
        for task in tasks {
            let record = task.await.unwrap();
            assert!(!record.has_error(), "H2 request failed: {:?}", record.error);
            assert_eq!(record.status, Some(200));
            if record
                .trace
                .as_ref()
                .is_some_and(|trace| trace.connection_reused_ns.is_some())
            {
                reused += 1;
            }
        }

        assert_eq!(server.accepted.get(), 1, "all H2 streams share one socket");
        assert!(server.max_active.get() > 1, "H2 streams must overlap");
        assert_eq!(reused, 23, "only the connection-opening request is fresh");
    });
}

#[test]
fn h1_pool_is_bounded_queues_and_collects_wire_chunks() {
    run_local(async {
        let server = LoopbackServer::spawn(ServerProtocol::H1, Duration::from_millis(20)).await;
        let clock: Rc<dyn Clock> = RealClock::new();
        let transport = Rc::new(HttpTransport::new(
            clock,
            ClientConfig {
                http_version: HttpVersion::Http1Only,
                max_connections_per_origin: 2,
                collect_trace_chunks: true,
                ..ClientConfig::default()
            },
        ));
        let config = request(&server.base_url);

        let mut tasks = Vec::new();
        for _ in 0..12 {
            let transport = transport.clone();
            let config = config.clone();
            tasks.push(tokio::task::spawn_local(async move {
                transport.get(&config).await
            }));
        }

        let mut queued = 0;
        for task in tasks {
            let record = task.await.unwrap();
            assert!(!record.has_error(), "H1 request failed: {:?}", record.error);
            let trace = record.trace.unwrap();
            if trace.connection_pool_wait_start_ns.is_some() {
                queued += 1;
                assert!(trace.connection_pool_wait_end_ns.is_some());
            }
            assert_eq!(
                trace.response_chunks.len(),
                trace.response_chunks_count as usize
            );
            assert_eq!(
                trace
                    .response_chunks
                    .iter()
                    .map(|(_, size)| *size)
                    .sum::<u64>(),
                trace.response_bytes_total
            );
            assert_eq!(trace.request_chunks.len(), 1);
        }

        assert_eq!(server.accepted.get(), 2, "H1 pool must stop at its bound");
        assert_eq!(server.max_active.get(), 2);
        assert!(
            queued >= 10,
            "requests beyond capacity must record pool wait"
        );
    });
}

#[test]
fn sticky_h1_serializes_one_session_and_releases_on_final_turn() {
    run_local(async {
        let server = LoopbackServer::spawn(ServerProtocol::H1, Duration::from_millis(10)).await;
        let clock: Rc<dyn Clock> = RealClock::new();
        let transport = Rc::new(HttpTransport::new(
            clock,
            ClientConfig {
                http_version: HttpVersion::Http1Only,
                max_connections_per_origin: 8,
                ..ClientConfig::default()
            },
        ));
        let config = request(&server.base_url)
            .reuse(ConnectionReuseStrategy::StickyUserSessions)
            .correlation_id("session-a")
            .final_turn(false);

        let mut tasks = Vec::new();
        for _ in 0..6 {
            let transport = transport.clone();
            let config = config.clone();
            tasks.push(tokio::task::spawn_local(async move {
                transport.get(&config).await
            }));
        }
        for task in tasks {
            let record = task.await.unwrap();
            assert!(
                !record.has_error(),
                "sticky request failed: {:?}",
                record.error
            );
        }
        assert_eq!(server.accepted.get(), 1);
        assert_eq!(server.max_active.get(), 1, "one H1 lease serializes turns");

        let final_turn = config.clone().final_turn(true);
        let final_record = transport.get(&final_turn).await;
        assert!(!final_record.has_error());
        assert_eq!(server.accepted.get(), 1, "final turn reuses its lease");

        let restarted = transport.get(&config).await;
        assert!(!restarted.has_error());
        assert_eq!(
            server.accepted.get(),
            2,
            "a post-final session gets a new connection"
        );
    });
}

#[test]
fn sim_clock_expires_idle_h1_and_reuses_dns_cache() {
    run_local(async {
        let server = LoopbackServer::spawn(ServerProtocol::H1, Duration::ZERO).await;
        let clock = Rc::new(SimClock::new());
        let clock_dyn: Rc<dyn Clock> = clock.clone();
        let transport = HttpTransport::new(
            clock_dyn,
            ClientConfig {
                http_version: HttpVersion::Http1Only,
                keepalive_ns: Some(5),
                dns_cache_ttl_ns: Some(100),
                ..ClientConfig::default()
            },
        );
        let config = request(&server.base_url);

        let first = transport.get(&config).await;
        assert!(!first.has_error());
        assert_eq!(server.accepted.get(), 1);
        assert!(first.trace.unwrap().dns_cache_miss_ns.is_some());

        clock.advance_to(5);
        let second = transport.get(&config).await;
        assert!(!second.has_error());
        let trace = second.trace.unwrap();
        assert_eq!(server.accepted.get(), 2, "expired H1 socket is replaced");
        assert!(trace.connection_reused_ns.is_none());
        assert_eq!(trace.dns_cache_hit_ns, Some(5));
        assert!(trace.dns_lookup_start_ns.is_none());
    });
}

#[test]
fn sim_clock_expires_idle_h2_root_connection() {
    run_local(async {
        let server = LoopbackServer::spawn(ServerProtocol::H2, Duration::ZERO).await;
        let clock = Rc::new(SimClock::new());
        let clock_dyn: Rc<dyn Clock> = clock.clone();
        let transport = HttpTransport::new(
            clock_dyn,
            ClientConfig {
                http_version: HttpVersion::Http2PriorKnowledge,
                keepalive_ns: Some(5),
                ..ClientConfig::default()
            },
        );
        let config = request(&server.base_url);

        let first = transport.get(&config).await;
        assert!(!first.has_error());
        assert_eq!(server.accepted.get(), 1);

        clock.advance_to(5);
        let second = transport.get(&config).await;
        assert!(!second.has_error());
        assert_eq!(server.accepted.get(), 2, "expired H2 root is replaced");
        assert!(second.trace.unwrap().connection_reused_ns.is_none());
    });
}

#[test]
fn zero_keepalive_disables_idle_h1_reuse() {
    run_local(async {
        let server = LoopbackServer::spawn(ServerProtocol::H1, Duration::ZERO).await;
        let clock: Rc<dyn Clock> = RealClock::new();
        let transport = HttpTransport::new(
            clock,
            ClientConfig {
                http_version: HttpVersion::Http1Only,
                keepalive_ns: Some(0),
                ..ClientConfig::default()
            },
        );
        let config = request(&server.base_url);

        let first = transport.get(&config).await;
        let second = transport.get(&config).await;
        assert!(!first.has_error());
        assert!(!second.has_error());
        assert_eq!(server.accepted.get(), 2);
        assert!(second.trace.unwrap().connection_reused_ns.is_none());
    });
}

#[test]
fn sim_clock_total_timeout_includes_h1_pool_wait() {
    run_local(async {
        let server = LoopbackServer::spawn(ServerProtocol::H1, Duration::from_millis(20)).await;
        let clock = Rc::new(SimClock::new());
        let manager: Rc<dyn ConnectionManager> = Rc::new(ConnectionPool::new());
        let unlimited = Rc::new(HttpTransport::with_connection_manager(
            clock.clone(),
            ClientConfig {
                http_version: HttpVersion::Http1Only,
                max_connections_per_origin: 1,
                ..ClientConfig::default()
            },
            manager.clone(),
        ));
        let bounded = Rc::new(HttpTransport::with_connection_manager(
            clock.clone(),
            ClientConfig {
                http_version: HttpVersion::Http1Only,
                max_connections_per_origin: 1,
                total_timeout_ns: Some(5),
                ..ClientConfig::default()
            },
            manager,
        ));
        let config = request(&server.base_url);

        let first_transport = unlimited.clone();
        let first_config = config.clone();
        let first =
            tokio::task::spawn_local(async move { first_transport.get(&first_config).await });
        while server.active.get() == 0 {
            tokio::task::yield_now().await;
        }

        let second_transport = bounded.clone();
        let second_config = config.clone();
        let second =
            tokio::task::spawn_local(async move { second_transport.get(&second_config).await });
        tokio::task::yield_now().await;
        clock.advance_to(5);

        let timed_out = second.await.unwrap();
        let error = timed_out.error.expect("pool waiter must hit total timeout");
        assert_eq!(
            error.kind,
            aiperf_runtime::transport_http::models::ErrorKind::Timeout
        );
        assert_eq!(error.message, "request timeout after 5ns");
        let trace = timed_out.trace.unwrap();
        assert_eq!(trace.connection_pool_wait_start_ns, Some(0));
        assert!(trace.connection_pool_wait_end_ns.is_none());

        let completed = first.await.unwrap();
        assert!(!completed.has_error());
        assert_eq!(server.accepted.get(), 1);
    });
}
