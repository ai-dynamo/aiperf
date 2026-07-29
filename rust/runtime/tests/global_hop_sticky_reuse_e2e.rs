// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Global-hop sticky worker-routing e2e: proves per-session single-connection
//! reuse under `HopRouting::Sticky`, and its absence under `HopRouting::RoundRobin`.
//!
//! The `workers > 1` [`ThreadPerCoreExecutor`] (built by `HttpExecutionFactory`
//! for the [`DispatchMode::GlobalHop`] placement) routes each dispatched turn to
//! a worker OS thread by `hop_routing`. Each worker owns a worker-local HTTP
//! connection pool keyed by `correlation_id` (see
//! `transport/http/client/pool.rs`): under [`ConnectionReuseStrategy::StickyUserSessions`]
//! a session holds ONE connection for the lifetime of its non-final turns.
//!
//! Consequences this test asserts from raw per-record output (drained
//! [`RecordIngest`]s carrying the executing `worker_id`, the `correlation_id`,
//! and `http.connection_reused`):
//!
//! 1. **Sticky honored** — with `hop_routing = Sticky` every turn of a session
//!    executes on ONE worker (`worker_id` constant per `correlation_id`) AND
//!    every non-first turn reports `connection_reused = true`: one connection
//!    per session.
//! 2. **Contrast** — the SAME workload under `hop_routing = RoundRobin` scatters
//!    a session's turns across workers (`worker_id` not constant per
//!    `correlation_id`), so per-session single-connection reuse does not hold.
//!
//! A real in-process HTTP server on `127.0.0.1` (NOT `localhost`: the runtime
//! client is IPv4-only on loopback) serves a fixed streaming SSE completion over
//! keep-alive connections, so a worker CAN reuse a pooled connection.

#![cfg(feature = "engine")]

use std::collections::BTreeMap;
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::Arc;

use aiperf_runtime::clock::{Clock, RealClock, RealClockAnchor};
use aiperf_runtime::endpoints::{
    EndpointId, EndpointKey, EndpointRegistry, PreparedEndpointTable, RawEndpointConfig,
};
use aiperf_runtime::engine::protocol::HopRouting;
use aiperf_runtime::engine::turn_execution::{
    ExecutionBackendConfig, HttpExecutionFactory, PreparedEndpointTableFactory,
    RequestExecutorFactory,
};
use aiperf_runtime::metrics::RequestMetricMetadata;
use aiperf_runtime::metrics_core::{MetricsConfig, RecordIngest};
use aiperf_runtime::multiturn::{PreparedEndpointReference, TurnDataPolicy};
use aiperf_runtime::transport::core::{
    ConnectionReuseStrategy, MeasuredContext, PreparedEndpointBinding, PreparedTurn, Request,
    RequestExecutor,
};
use aiperf_runtime::transport::http::TransportSinkConfig;

use axum::response::sse::{Event, Sse};
use axum::{Router, routing::post};
use uuid::Uuid;

const WORKERS: usize = 2;
const CONVERSATIONS: usize = 3;
const TURNS_PER_CONVERSATION: u32 = 3;

/// One conversation's turn as observed at the record boundary.
#[derive(Debug)]
struct ObservedTurn {
    correlation_id: String,
    turn_index: u32,
    worker_id: String,
    connection_reused: Option<bool>,
    errored: bool,
}

/// A keep-alive in-process streaming SSE inference endpoint on `127.0.0.1`.
///
/// Runs on its own OS thread with a multi-thread runtime so it never contends
/// with the coordinator's current-thread reactor. Persistent HTTP/1 keep-alive
/// (no `Connection: close`) is what lets a worker reuse a pooled connection for
/// consecutive same-session turns.
struct KeepAliveMock {
    base_url: String,
    shutdown: Option<tokio::sync::oneshot::Sender<()>>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl KeepAliveMock {
    fn spawn() -> Self {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let addr = listener.local_addr().unwrap();
        let base_url = format!("http://{addr}");
        let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel::<()>();
        let thread = std::thread::Builder::new()
            .name("keepalive-mock".into())
            .spawn(move || {
                let runtime = tokio::runtime::Builder::new_multi_thread()
                    .worker_threads(2)
                    .enable_all()
                    .build()
                    .unwrap();
                runtime.block_on(async move {
                    let listener = tokio::net::TcpListener::from_std(listener).unwrap();
                    let app = Router::new().route("/v1/chat/completions", post(handler));
                    axum::serve(listener, app)
                        .with_graceful_shutdown(async move {
                            let _ = shutdown_rx.await;
                        })
                        .await
                        .unwrap();
                });
            })
            .unwrap();
        Self {
            base_url,
            shutdown: Some(shutdown_tx),
            thread: Some(thread),
        }
    }
}

impl Drop for KeepAliveMock {
    fn drop(&mut self) {
        if let Some(shutdown) = self.shutdown.take() {
            let _ = shutdown.send(());
        }
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

/// Fixed two-token streaming completion with authoritative usage and `[DONE]`.
async fn handler() -> Sse<impl futures::Stream<Item = Result<Event, std::convert::Infallible>>> {
    let frames = vec![
        Event::default().data(
            serde_json::json!({
                "choices": [{"index": 0, "delta": {"role": "assistant", "content": "hi"}}]
            })
            .to_string(),
        ),
        Event::default().data(
            serde_json::json!({
                "choices": [{"index": 0, "delta": {"content": " there"}, "finish_reason": "stop"}]
            })
            .to_string(),
        ),
        Event::default().data(
            serde_json::json!({
                "choices": [],
                "usage": {"prompt_tokens": 5, "completion_tokens": 2}
            })
            .to_string(),
        ),
        Event::default().data("[DONE]"),
    ];
    Sse::new(futures::stream::iter(
        frames.into_iter().map(Ok::<_, std::convert::Infallible>),
    ))
}

/// Worker-local `chat` endpoint table bound to the mock's base URL, configured
/// for streaming with server-authoritative token counts.
#[derive(Clone)]
struct ChatEndpointTableFactory {
    registry: EndpointRegistry,
    url: String,
}

impl PreparedEndpointTableFactory for ChatEndpointTableFactory {
    fn prepare_worker(&self) -> anyhow::Result<PreparedEndpointTable> {
        let endpoint = self.registry.prepare(
            &EndpointId::new("chat")?,
            RawEndpointConfig {
                urls: vec![self.url.clone()],
                streaming: true,
                use_server_token_count: true,
                ..RawEndpointConfig::default()
            },
        )?;
        let mut table = PreparedEndpointTable::new();
        assert_eq!(table.push(endpoint)?, EndpointKey::from_index(0));
        Ok(table)
    }
}

/// Build a `workers > 1` global-hop backend under `routing` and
/// `StickyUserSessions` connection reuse.
fn build_backend(base_url: &str, routing: HopRouting) -> Rc<dyn RequestExecutor> {
    let anchor = RealClockAnchor::now();
    let clock: Rc<dyn Clock> = RealClock::from_anchor(anchor);
    let table_factory = Arc::new(ChatEndpointTableFactory {
        registry: EndpointRegistry::builtin().unwrap(),
        url: base_url.to_string(),
    });
    let transport = TransportSinkConfig {
        connection_reuse: ConnectionReuseStrategy::StickyUserSessions,
        ..TransportSinkConfig::default()
    };
    let backend = HttpExecutionFactory
        .build(ExecutionBackendConfig {
            workers: WORKERS,
            coordinator_clock: clock.clone(),
            real_clock_anchor: anchor,
            base_urls: vec![base_url.to_string()],
            model: "fixture-model".to_string(),
            transport,
            raw_enabled: false,
            prepared_endpoints: Some(table_factory),
            hop_routing: routing,
        })
        .unwrap();
    let origin_ns = clock.now_ns();
    backend.set_run_origin(origin_ns).unwrap();
    backend
        .configure_measurement(MetricsConfig::default(), origin_ns)
        .unwrap();
    backend
}

/// One streaming chat turn for `correlation_id`/`turn_index`; `is_final_turn`
/// releases the sticky connection so only the last turn of a session drops it.
fn streaming_turn(correlation_id: &str, is_final_turn: bool) -> PreparedTurn {
    PreparedTurn {
        request: Request {
            uuid: Uuid::new_v4(),
            input_length: 5,
            max_output_tokens: 2,
            prompt_text: None,
            request_body: Some(serde_json::json!({
                "model": "fixture-model",
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 2,
                "stream": true,
                "stream_options": {"include_usage": true}
            })),
            request_body_bytes: None,
            headers: BTreeMap::new(),
            parameters: BTreeMap::new(),
            endpoint_path: None,
            streaming: true,
            x_correlation_id: Some(correlation_id.to_string()),
            is_final_turn,
            cancel_after_ns: None,
            url_index: None,
            image_count: None,
            recorded_api_time_ns: None,
            recorded_ttft_ns: None,
        },
        model: "fixture-model".to_string(),
        endpoint: PreparedEndpointBinding::Prepared(PreparedEndpointReference {
            key: EndpointKey::from_index(0),
            endpoint_id: EndpointId::new("chat").unwrap(),
        }),
        endpoint_aware: true,
        data_policy: TurnDataPolicy::ordinary(),
    }
}

/// Coordinator-known arrival facts carrying the session correlation + turn index.
fn measured_context(correlation_id: &str, turn_index: u32) -> MeasuredContext {
    MeasuredContext {
        arrival_ms: 0.0,
        input_length: 5,
        requested_output_length: 2,
        metadata: RequestMetricMetadata {
            turn_index,
            correlation_id: Some(correlation_id.to_string()),
            conversation_id: Some(correlation_id.to_string()),
            ..RequestMetricMetadata::default()
        },
        wants_live_record: false,
        consume_record: false,
    }
}

fn run_local<F: std::future::Future>(fut: F) -> F::Output {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let local = tokio::task::LocalSet::new();
    local.block_on(&rt, fut)
}

/// Drive the multi-turn workload through the backend and return the drained
/// per-record observations.
fn run_workload(base_url: &str, routing: HopRouting) -> Vec<ObservedTurn> {
    run_local(async {
        let backend = build_backend(base_url, routing);
        // Each session issues its turns in order; the coordinator awaits every
        // turn before the next, so issuance order is deterministic. The final
        // turn releases the sticky connection.
        for conversation in 0..CONVERSATIONS {
            let correlation = format!("session-{conversation}");
            for turn_index in 0..TURNS_PER_CONVERSATION {
                let is_final = turn_index + 1 == TURNS_PER_CONVERSATION;
                let outcome = backend
                    .execute_measured(
                        streaming_turn(&correlation, is_final),
                        measured_context(&correlation, turn_index),
                        &|_| {},
                    )
                    .await
                    .expect("turn dispatch must complete");
                assert_eq!(
                    outcome.result.record.status,
                    Some(200),
                    "server delivered turn {turn_index} of {correlation}"
                );
            }
        }
        let end_ns = RealClock::from_anchor(RealClockAnchor::now()).now_ns();
        let records = backend.drain_records(end_ns).expect("drain records");
        backend.shutdown().unwrap();
        records
            .into_iter()
            .map(|(_uuid, ingest): (Uuid, RecordIngest)| ObservedTurn {
                correlation_id: ingest.correlation_id,
                turn_index: ingest.turn_index,
                worker_id: ingest
                    .worker_id
                    .expect("hop worker stamps the executing worker id"),
                connection_reused: ingest.http.connection_reused,
                errored: ingest.errored,
            })
            .collect()
    })
}

/// Group observations by correlation id, each sorted by turn index.
fn by_correlation(observed: Vec<ObservedTurn>) -> HashMap<String, Vec<ObservedTurn>> {
    let mut grouped: HashMap<String, Vec<ObservedTurn>> = HashMap::new();
    for turn in observed {
        grouped
            .entry(turn.correlation_id.clone())
            .or_default()
            .push(turn);
    }
    for turns in grouped.values_mut() {
        turns.sort_by_key(|turn| turn.turn_index);
    }
    grouped
}

#[test]
fn sticky_global_hop_keeps_one_connection_per_session() {
    let mock = KeepAliveMock::spawn();
    let observed = run_workload(&mock.base_url, HopRouting::Sticky);
    assert_eq!(
        observed.len(),
        CONVERSATIONS * TURNS_PER_CONVERSATION as usize,
        "every dispatched turn produced a record"
    );
    let grouped = by_correlation(observed);
    assert_eq!(grouped.len(), CONVERSATIONS, "one group per session");

    for (correlation, turns) in &grouped {
        assert_eq!(
            turns.len(),
            TURNS_PER_CONVERSATION as usize,
            "session {correlation} has all turns"
        );
        assert!(
            turns.iter().all(|turn| !turn.errored),
            "session {correlation} turns all succeeded"
        );

        // Sticky honored: every turn of the session ran on ONE worker.
        let worker = &turns[0].worker_id;
        assert!(
            turns.iter().all(|turn| &turn.worker_id == worker),
            "session {correlation} must stay on one worker under sticky routing, got {:?}",
            turns.iter().map(|t| &t.worker_id).collect::<Vec<_>>()
        );

        // One connection per session: the first turn establishes it, every later
        // turn reuses it (worker-local pool keyed by correlation id).
        assert_eq!(
            turns[0].connection_reused,
            Some(false),
            "first turn of {correlation} establishes a fresh connection"
        );
        for turn in &turns[1..] {
            assert_eq!(
                turn.connection_reused,
                Some(true),
                "turn {} of {correlation} reuses the session connection",
                turn.turn_index
            );
        }
    }
}

#[test]
fn round_robin_global_hop_scatters_a_session_across_workers() {
    let mock = KeepAliveMock::spawn();
    let observed = run_workload(&mock.base_url, HopRouting::RoundRobin);
    assert_eq!(
        observed.len(),
        CONVERSATIONS * TURNS_PER_CONVERSATION as usize,
        "every dispatched turn produced a record"
    );
    let grouped = by_correlation(observed);
    assert_eq!(grouped.len(), CONVERSATIONS, "one group per session");

    // Deterministic core: with round-robin over 2 workers and 3 sequentially
    // issued turns per session, each session's turns land on workers {i, i+1, i}
    // — never a single worker. Per-session single-connection reuse cannot hold.
    let mut any_scatter = false;
    for (correlation, turns) in &grouped {
        assert!(
            turns.iter().all(|turn| !turn.errored),
            "session {correlation} turns all succeeded"
        );
        let distinct: std::collections::BTreeSet<&String> =
            turns.iter().map(|turn| &turn.worker_id).collect();
        assert!(
            distinct.len() > 1,
            "session {correlation} must scatter across workers under round-robin, got {:?}",
            turns.iter().map(|t| &t.worker_id).collect::<Vec<_>>()
        );
        any_scatter = true;
    }
    assert!(any_scatter, "at least one session was observed");
}
