// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Per-dispatch body-path cost for ordinary HTTP chat.
//!
//! Measures where time and heap allocations actually go on the plain-chat fast
//! path: cached-plan dispatch, uncached (`format_payload`) dispatch, the
//! image-count body re-parse at `transport/http/sink/endpoint_dispatch.rs`, and
//! the `TurnToSend` deep clone in `IssuedCredit::from_issued_turn`.
//!
//! # Harness
//!
//! This target is auto-discovered by Cargo with the default `harness = true`
//! (libtest). Adding a `[[bench]] harness = false` entry to `runtime/Cargo.toml`
//! was out of scope for the measurement task that produced this file, so the
//! measurements are plain `#[test]` entry points driving a hand-rolled
//! wall-clock + allocation-counting loop rather than criterion. That is also the
//! only way to get the number that matters most here — allocations per dispatch
//! — which criterion does not report.
//!
//! Run with:
//!
//! ```text
//! cargo test --release -p aiperf-runtime --bench chat_dispatch_bench -- --nocapture --test-threads=1
//! ```
//!
//! `--test-threads=1` is required: the allocator hook is process-global, while
//! allocation accounting is thread-local and RAII-scoped.

use std::alloc::{GlobalAlloc, Layout};
use std::cell::Cell;
use std::collections::BTreeMap;
use std::io::{Read, Write};
use std::net::TcpListener;
use std::rc::Rc;
use std::sync::Arc;
use std::thread;
use std::time::Instant;

use bytes::Bytes;
use serde_json::Value;
use smallvec::smallvec;

use aiperf_runtime::body_plan::RequestBody;
use aiperf_runtime::dataset::materialize::Overrides;
use aiperf_runtime::dataset::model::{
    ContentGroup, Conversation, ConversationContextMode, MediaKind, SessionId, Turn,
};
use aiperf_runtime::dataset::request::{ConversationSession, EndpointRequestMaterializer};
use aiperf_runtime::dataset::segment::{Role, SegmentPool};
use aiperf_runtime::dataset::{Dataset, TurnEndpointLookup};
use aiperf_runtime::dispatch::collector::ReplayTerminalStatus;
use aiperf_runtime::endpoints::{
    CreditPhase, EndpointId, EndpointRegistry, PreparedEndpoint, PreparedEndpointTable,
    RawEndpointConfig, ShapeLowerer, TurnMessageLowerer,
};
use aiperf_runtime::multiturn::{
    ConversationSource, IssuedCredit, NativeDatasetConversationSource, PreparedEndpointReference,
    TurnResponse,
};
use aiperf_runtime::transport::http::RealClock;
use aiperf_runtime::transport::http::client::http_client::HttpClient;
use aiperf_runtime::transport::http::config::ClientConfig;

// ---------------------------------------------------------------------------
// Counting allocator
// ---------------------------------------------------------------------------

thread_local! {
    static COUNTING: Cell<bool> = const { Cell::new(false) };
    static ALLOC_COUNT: Cell<u64> = const { Cell::new(0) };
    static ALLOC_BYTES: Cell<u64> = const { Cell::new(0) };
}

struct CountingAlloc;

unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let pointer = unsafe { mimalloc::MiMalloc.alloc(layout) };
        if !pointer.is_null() && COUNTING.get() {
            ALLOC_COUNT.with(|count| count.set(count.get().saturating_add(1)));
            ALLOC_BYTES.with(|total| {
                total.set(total.get().saturating_add(layout.size() as u64));
            });
        }
        pointer
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let pointer = unsafe { mimalloc::MiMalloc.alloc_zeroed(layout) };
        if !pointer.is_null() && COUNTING.get() {
            ALLOC_COUNT.with(|count| count.set(count.get().saturating_add(1)));
            ALLOC_BYTES.with(|total| {
                total.set(total.get().saturating_add(layout.size() as u64));
            });
        }
        pointer
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { mimalloc::MiMalloc.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let replacement = unsafe { mimalloc::MiMalloc.realloc(ptr, layout, new_size) };
        if !replacement.is_null() && COUNTING.get() {
            ALLOC_COUNT.with(|count| count.set(count.get().saturating_add(1)));
            ALLOC_BYTES.with(|total| total.set(total.get().saturating_add(new_size as u64)));
        }
        replacement
    }
}

#[global_allocator]
static GLOBAL: CountingAlloc = CountingAlloc;

struct AllocationScope {
    is_active: bool,
}

impl AllocationScope {
    fn start() -> Self {
        assert!(!COUNTING.replace(true), "allocation probes may not nest");
        ALLOC_COUNT.set(0);
        ALLOC_BYTES.set(0);
        Self { is_active: true }
    }

    fn finish(mut self) -> (u64, u64) {
        self.disable();
        (ALLOC_COUNT.get(), ALLOC_BYTES.get())
    }

    fn disable(&mut self) {
        if self.is_active {
            assert!(COUNTING.replace(false), "allocation probe was not active");
            self.is_active = false;
        }
    }
}

impl Drop for AllocationScope {
    fn drop(&mut self) {
        self.disable();
    }
}

/// One measured scenario.
struct Sample {
    label: String,
    body_bytes: usize,
    ns_per_iter: f64,
    allocs_per_iter: f64,
    alloc_bytes_per_iter: f64,
}

impl Sample {
    fn print(&self) {
        println!(
            "{:<52} body={:>7}B  {:>10.3} us  {:>8.1} allocs  {:>10.0} alloc-B",
            self.label,
            self.body_bytes,
            self.ns_per_iter / 1000.0,
            self.allocs_per_iter,
            self.alloc_bytes_per_iter,
        );
    }

    fn print_allocation_json(&self, path: &str, iterations: u64) {
        println!(
            "AIPERF_ALLOCATION_SAMPLE {}",
            serde_json::json!({
                "path": path,
                "iterations": iterations,
                "allocation_count_per_request": self.allocs_per_iter,
                "allocated_bytes_per_request": self.alloc_bytes_per_iter,
                "nanoseconds_per_request": self.ns_per_iter,
            })
        );
    }
}

/// Time and count allocations for `iters` runs of `body`, after a warm-up pass.
fn measure<T>(
    label: impl Into<String>,
    body_bytes: usize,
    iters: u64,
    mut body: impl FnMut() -> T,
) -> Sample {
    // Warm caches, lazy state, and any one-shot interning.
    for _ in 0..(iters / 10).max(16) {
        std::hint::black_box(body());
    }

    let allocation_scope = AllocationScope::start();
    let start = Instant::now();
    for _ in 0..iters {
        std::hint::black_box(body());
    }
    let elapsed = start.elapsed();
    let (allocs, bytes) = allocation_scope.finish();

    Sample {
        label: label.into(),
        body_bytes,
        ns_per_iter: elapsed.as_nanos() as f64 / iters as f64,
        allocs_per_iter: allocs as f64 / iters as f64,
        alloc_bytes_per_iter: bytes as f64 / iters as f64,
    }
}

// ---------------------------------------------------------------------------
// Dataset construction
// ---------------------------------------------------------------------------

/// Deterministic ASCII filler of `len` bytes, so bodies are exactly sized and
/// JSON-escape-free (an escape would change parse cost for reasons unrelated to
/// size).
fn filler(len: usize, seed: u8) -> String {
    const ALPHABET: &[u8] = b"abcdefghijklmnopqrstuvwxyz ";
    (0..len)
        .map(|index| ALPHABET[(index + seed as usize) % ALPHABET.len()] as char)
        .collect()
}

fn user_turn(pool: &mut SegmentPool, text: String) -> Turn {
    let tokens: Vec<u32> = (0..(text.len() / 4).max(1) as u32).collect();
    let handle = pool
        .intern_text(None, Role::from("user"), Bytes::from(text), tokens)
        .expect("intern user text");
    Turn {
        role: Some(Role::from("user")),
        content: smallvec![ContentGroup {
            kind: MediaKind::Text,
            name: String::new(),
            handles: smallvec![handle],
            uuids: smallvec![],
        }],
        input_tokens: Some(8),
        max_tokens: Some(128),
        ..Turn::default()
    }
}

/// A recorded self-contained message-array snapshot for turn `index`: every
/// prior user message, every prior authored assistant reply, then this turn's
/// user message. This is the `MessageArrayWithResponses` shape, whose every turn
/// is response-independent and so caches outright — the reference the live
/// `DeltasWithoutResponses` row below is measured against.
fn snapshot_turn(pool: &mut SegmentPool, index: usize, per_message: usize) -> Turn {
    let mut handles = smallvec![];
    for prior in 0..index {
        let user = pool
            .intern_text(
                None,
                Role::from("user"),
                Bytes::from(filler(per_message, prior as u8)),
                vec![prior as u32],
            )
            .expect("intern prior user");
        handles.push(user);
        let assistant = pool
            .intern_text(
                None,
                Role::from("assistant"),
                Bytes::from(filler(per_message, 100 + prior as u8)),
                vec![100 + prior as u32],
            )
            .expect("intern prior assistant");
        handles.push(assistant);
    }
    let current = pool
        .intern_text(
            None,
            Role::from("user"),
            Bytes::from(filler(per_message, index as u8)),
            vec![index as u32],
        )
        .expect("intern current user");
    handles.push(current);

    Turn {
        role: Some(Role::from("user")),
        content: smallvec![ContentGroup {
            kind: MediaKind::Text,
            name: String::new(),
            handles,
            uuids: smallvec![],
        }],
        input_tokens: Some(8),
        max_tokens: Some(128),
        ..Turn::default()
    }
}

fn prepared_chat_endpoint() -> Box<dyn PreparedEndpoint> {
    prepared_chat_endpoint_with_streaming(true)
}

fn prepared_chat_endpoint_with_streaming(streaming: bool) -> Box<dyn PreparedEndpoint> {
    EndpointRegistry::builtin()
        .expect("builtin endpoint registry")
        .prepare(
            &EndpointId::new("chat").expect("chat endpoint id"),
            RawEndpointConfig {
                streaming,
                ..RawEndpointConfig::default()
            },
        )
        .expect("prepare chat endpoint")
}

fn deterministic_http_server(requests: usize) -> (url::Url, thread::JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind allocation-probe server");
    let address = listener.local_addr().expect("allocation-probe address");
    let server = thread::spawn(move || {
        let body = br#"{"id":"chatcmpl-allocation","object":"chat.completion","created":0,"model":"mock-model","choices":[{"index":0,"message":{"role":"assistant","content":"deterministic response"},"finish_reason":"stop"}],"usage":{"prompt_tokens":4,"completion_tokens":2,"total_tokens":6}}"#;
        for _ in 0..requests {
            let (mut stream, _) = listener.accept().expect("accept allocation-probe request");
            let mut request = Vec::new();
            let mut chunk = [0_u8; 4096];
            loop {
                let read = stream
                    .read(&mut chunk)
                    .expect("read allocation-probe request");
                if read == 0 {
                    break;
                }
                request.extend_from_slice(&chunk[..read]);
                if request.windows(4).any(|window| window == b"\r\n\r\n") {
                    break;
                }
            }
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                body.len()
            );
            stream
                .write_all(response.as_bytes())
                .expect("write allocation-probe response headers");
            stream
                .write_all(body)
                .expect("write allocation-probe response body");
        }
    });
    (
        url::Url::parse(&format!("http://{address}/v1/chat/completions"))
            .expect("allocation-probe URL"),
        server,
    )
}

/// Every turn resolves to the one endpoint under measurement.
struct SingleEndpointLookup<'a>(&'a dyn PreparedEndpoint);

impl TurnEndpointLookup for SingleEndpointLookup<'_> {
    fn endpoint_for(&self, _name: Option<&str>) -> Option<&dyn PreparedEndpoint> {
        Some(self.0)
    }
}

fn build_dataset(
    mode: ConversationContextMode,
    turns: Vec<Turn>,
    pool: SegmentPool,
    endpoint: &dyn PreparedEndpoint,
    precompute: bool,
) -> Arc<Dataset> {
    let mut conversation = Conversation::new("session");
    conversation.context_mode = Some(mode);
    conversation.turns = turns;
    let mut dataset = Dataset::new(
        vec![conversation],
        Arc::new(pool.freeze()),
        "sequential",
        mode,
    )
    .expect("dataset");
    if let Some(lowerer) = ShapeLowerer::for_descriptor_id(endpoint.descriptor().id) {
        dataset
            .lower_messages_for_endpoint(&lowerer)
            .expect("lower messages");
    }
    if precompute {
        dataset
            .precompute_body_plans(endpoint, "primary-model")
            .expect("precompute body plans");
    }
    // Always established, independent of body-plan caching: this is what makes a
    // continuation turn's `image_count` known so dispatch never re-parses.
    dataset
        .precompute_image_counts(&SingleEndpointLookup(endpoint), "primary-model")
        .expect("precompute image counts");
    Arc::new(dataset)
}

/// Drive a session to `turn_index`, splicing a synthetic assistant reply after
/// each prior turn when the mode captures live replies.
///
/// The reply is lowered through the endpoint's [`ShapeLowerer`] before capture,
/// exactly as `multiturn::NativeSessionBackend::build_next_turn` does. Capturing
/// an unlowered reply here would measure a re-render the runtime never performs.
fn session_at(
    dataset: Arc<Dataset>,
    endpoint: &dyn PreparedEndpoint,
    turn_index: usize,
    reply_len: usize,
) -> ConversationSession {
    let lowerer = ShapeLowerer::for_descriptor_id(endpoint.descriptor().id);
    let mut session =
        ConversationSession::new(dataset, SessionId::from("session")).expect("session");
    for index in 0..=turn_index {
        session.advance_to(index).expect("advance");
        if index < turn_index && session.should_capture_response() {
            let mut reply = endpoint_assistant_turn(filler(reply_len, 200 + index as u8));
            if let Some(lowerer) = &lowerer {
                reply.lowered = Some(lowerer.lower_turn(&reply).expect("lower reply"));
            }
            session
                // A text-only assistant reply contributes no wire image part.
                .capture_response(reply, 64, Some(0))
                .expect("capture response");
        }
    }
    session
}

fn endpoint_assistant_turn(text: String) -> aiperf_runtime::endpoints::Turn {
    aiperf_runtime::endpoints::Turn {
        role: Some("assistant".to_string()),
        texts: vec![aiperf_runtime::endpoints::Media::new(vec![text])],
        ..aiperf_runtime::endpoints::Turn::default()
    }
}

fn dispatch_body(
    session: &ConversationSession,
    endpoint: &dyn PreparedEndpoint,
    overrides: &Overrides,
) -> Bytes {
    session
        .materialize_prepared(
            &EndpointRequestMaterializer,
            endpoint,
            "primary-model",
            CreditPhase::Profiling,
            overrides,
        )
        .expect("materialize")
        .body
        .to_wire()
        .expect("to_wire")
}

/// What `endpoint_dispatch.rs` actually runs: materialize the body, then take
/// the established image count when there is one and parse the body only when
/// there is not.
fn dispatch_body_and_image_count(
    session: &ConversationSession,
    endpoint: &dyn PreparedEndpoint,
    overrides: &Overrides,
) -> (Bytes, usize) {
    let request = session
        .materialize_prepared(
            &EndpointRequestMaterializer,
            endpoint,
            "primary-model",
            CreditPhase::Profiling,
            overrides,
        )
        .expect("materialize");
    let known = request.image_count;
    let body = request.body.to_wire().expect("to_wire");
    let count = known.map_or_else(
        || {
            let value = serde_json::from_slice::<Value>(&body).expect("parse");
            endpoint.extract_payload_inputs(&value).image_count as usize
        },
        |count| count as usize,
    );
    (body, count)
}

// ---------------------------------------------------------------------------
// L1/L2: body path + image-count parse
// ---------------------------------------------------------------------------

/// Per-message text lengths chosen so the finished turn-3 body lands near the
/// three requested sizes (~500 B, ~4 KB, ~32 KB of accumulated history).
const SIZES: [(&str, usize); 3] = [("0.5K", 55), ("4K", 490), ("32K", 4000)];

const ITERS: u64 = 20_000;

#[test]
fn required_plugin_allocation_baselines() {
    const REQUEST_ITERATIONS: u64 = 10_000;
    let endpoint_construction = measure("endpoint construction", 0, 10_000, || {
        prepared_chat_endpoint_with_streaming(false)
    });
    endpoint_construction.print_allocation_json("endpoint_preparation", 10_000);

    let formatting_endpoint = prepared_chat_endpoint_with_streaming(false);
    let mut pool = SegmentPool::new();
    let formatting_turn = user_turn(&mut pool, "deterministic request".to_owned());
    let formatting_dataset = build_dataset(
        ConversationContextMode::MessageArrayWithResponses,
        vec![formatting_turn],
        pool,
        formatting_endpoint.as_ref(),
        false,
    );
    let formatting_session = session_at(formatting_dataset, formatting_endpoint.as_ref(), 0, 0);
    let formatting_overrides = Overrides::new();
    let endpoint_formatting = measure("endpoint request formatting", 0, 10_000, || {
        dispatch_body(
            &formatting_session,
            formatting_endpoint.as_ref(),
            &formatting_overrides,
        )
    });
    endpoint_formatting.print_allocation_json("endpoint_formatting", 10_000);

    let server_requests = usize::try_from(1 + 2 * REQUEST_ITERATIONS)
        .expect("allocation-probe iteration count fits usize");
    let (url, server) = deterministic_http_server(server_requests);
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("allocation-probe runtime");
    let local = tokio::task::LocalSet::new();
    local.block_on(&runtime, async {
        let clock: Rc<dyn aiperf_runtime::transport::http::Clock> = RealClock::new();
        let client = HttpClient::new(clock, ClientConfig::default());
        let endpoint = prepared_chat_endpoint_with_streaming(false);
        let mut full_path_pool = SegmentPool::new();
        let full_path_turn = user_turn(&mut full_path_pool, "deterministic request".to_owned());
        let full_path_dataset = build_dataset(
            ConversationContextMode::MessageArrayWithResponses,
            vec![full_path_turn],
            full_path_pool,
            endpoint.as_ref(),
            false,
        );
        let full_path_session = session_at(full_path_dataset, endpoint.as_ref(), 0, 0);
        let full_path_overrides = Overrides::new();
        let headers = BTreeMap::from([
            ("Content-Type".to_owned(), "application/json".to_owned()),
            ("Accept".to_owned(), "application/json".to_owned()),
        ]);
        let body = Bytes::from_static(
            br#"{"model":"mock-model","stream":false,"max_tokens":2,"messages":[{"role":"user","content":"deterministic request"}]}"#,
        );

        let warm = client
            .request(&url, &headers, body.clone(), false, |_| {})
            .await;
        assert!(warm.is_valid(), "warm request failed: {:?}", warm.error);

        let allocation_scope = AllocationScope::start();
        let dispatch_start = Instant::now();
        for _ in 0..REQUEST_ITERATIONS {
            let dispatch = client
                .request(&url, &headers, body.clone(), false, |_| {})
                .await;
            assert!(dispatch.is_valid(), "dispatch failed: {:?}", dispatch.error);
        }
        let dispatch_elapsed = dispatch_start.elapsed();
        let (dispatch_allocations, dispatch_bytes) = allocation_scope.finish();
        println!(
            "AIPERF_ALLOCATION_SAMPLE {}",
            serde_json::json!({
                "path": "transport_dispatch",
                "iterations": REQUEST_ITERATIONS,
                "allocation_count": dispatch_allocations,
                "allocated_bytes": dispatch_bytes,
                "allocation_count_per_request": dispatch_allocations as f64 / REQUEST_ITERATIONS as f64,
                "allocated_bytes_per_request": dispatch_bytes as f64 / REQUEST_ITERATIONS as f64,
                "nanoseconds_per_request": dispatch_elapsed.as_nanos() as f64 / REQUEST_ITERATIONS as f64,
            })
        );

        let allocation_scope = AllocationScope::start();
        let request_start = Instant::now();
        for _ in 0..REQUEST_ITERATIONS {
            let body = dispatch_body(
                &full_path_session,
                endpoint.as_ref(),
                &full_path_overrides,
            );
            let successful = client
                .request(&url, &headers, body, false, |_| {})
                .await;
            assert!(successful.is_valid(), "request failed: {:?}", successful.error);
            let endpoint_record = aiperf_runtime::endpoints::RequestRecord {
                responses: successful
                    .responses
                    .iter()
                    .filter_map(|response| match response {
                        aiperf_runtime::transport::core::Response::Text(text) => {
                            Some(aiperf_runtime::endpoints::ServerResponse {
                                perf_ns: u64::try_from(text.perf_ns).unwrap_or(u64::MAX),
                                json: text.json(),
                                raw: Some(text.text.clone()),
                            })
                        }
                        aiperf_runtime::transport::core::Response::Sse(_) => None,
                    })
                    .collect(),
            };
            let parsed = endpoint
                .extract_response_data(&endpoint_record)
                .expect("parse successful response");
            let assistant = endpoint
                .build_assistant_turn(&endpoint_record)
                .expect("build successful assistant turn");
            assert_eq!(parsed.len(), 1);
            assert!(assistant.is_some());
        }
        let request_elapsed = request_start.elapsed();
        let (request_allocations, request_bytes) = allocation_scope.finish();
        println!(
            "AIPERF_ALLOCATION_SAMPLE {}",
            serde_json::json!({
                "path": "full_successful_request",
                "iterations": REQUEST_ITERATIONS,
                "allocation_count": request_allocations,
                "allocated_bytes": request_bytes,
                "allocation_count_per_request": request_allocations as f64 / REQUEST_ITERATIONS as f64,
                "allocated_bytes_per_request": request_bytes as f64 / REQUEST_ITERATIONS as f64,
                "nanoseconds_per_request": request_elapsed.as_nanos() as f64 / REQUEST_ITERATIONS as f64,
            })
        );
    });
    server.join().expect("allocation-probe server exits");
}

#[test]
fn chat_dispatch_body_path_profile() {
    println!("\n=== L1: per-dispatch body path (materialize_prepared -> Bytes) ===");
    let mut samples = Vec::new();
    let endpoint = prepared_chat_endpoint();
    let overrides = Overrides::new();

    for (size_label, per_message) in SIZES {
        // --- Turn 0, cached plan. The best case: Arc::clone + materialize. ---
        {
            let mut pool = SegmentPool::new();
            let turns = vec![user_turn(&mut pool, filler(per_message * 7, 0))];
            let dataset = build_dataset(
                ConversationContextMode::MessageArrayWithResponses,
                turns,
                pool,
                endpoint.as_ref(),
                true,
            );
            let session = session_at(dataset, endpoint.as_ref(), 0, 0);
            let len = dispatch_body(&session, endpoint.as_ref(), &overrides).len();
            samples.push(measure(
                format!("[{size_label}] turn 0  CACHED plan"),
                len,
                ITERS,
                || dispatch_body(&session, endpoint.as_ref(), &overrides),
            ));
        }

        // --- Turn 0, uncached (format_payload). ---
        {
            let mut pool = SegmentPool::new();
            let turns = vec![user_turn(&mut pool, filler(per_message * 7, 0))];
            let dataset = build_dataset(
                ConversationContextMode::MessageArrayWithResponses,
                turns,
                pool,
                endpoint.as_ref(),
                false,
            );
            let session = session_at(dataset, endpoint.as_ref(), 0, 0);
            let len = dispatch_body(&session, endpoint.as_ref(), &overrides).len();
            samples.push(measure(
                format!("[{size_label}] turn 0  UNCACHED format_payload"),
                len,
                ITERS,
                || dispatch_body(&session, endpoint.as_ref(), &overrides),
            ));
        }

        // --- Turn 3, cached plan (recorded snapshot mode). ---
        {
            let mut pool = SegmentPool::new();
            let turns: Vec<Turn> = (0..4)
                .map(|index| snapshot_turn(&mut pool, index, per_message))
                .collect();
            let dataset = build_dataset(
                ConversationContextMode::MessageArrayWithResponses,
                turns,
                pool,
                endpoint.as_ref(),
                true,
            );
            let session = session_at(dataset, endpoint.as_ref(), 3, 0);
            let len = dispatch_body(&session, endpoint.as_ref(), &overrides).len();
            samples.push(measure(
                format!("[{size_label}] turn 3  CACHED plan (recorded)"),
                len,
                ITERS,
                || dispatch_body(&session, endpoint.as_ref(), &overrides),
            ));
        }

        // --- Turn 3, live multi-turn chat: 3 prior user turns + 3 captured
        // assistant replies. `DeltasWithoutResponses` is the default context
        // mode and the ordinary chat case. Its continuation turns cache the
        // authored half of the body and splice the captured replies in at
        // dispatch, so this row measures a cached plan plus a live splice — it
        // was a full per-dispatch `format_payload` before that landed. ---
        {
            let mut pool = SegmentPool::new();
            let turns: Vec<Turn> = (0..4)
                .map(|index| user_turn(&mut pool, filler(per_message, index as u8)))
                .collect();
            let dataset = build_dataset(
                ConversationContextMode::DeltasWithoutResponses,
                turns,
                pool,
                endpoint.as_ref(),
                true,
            );
            let session = session_at(dataset, endpoint.as_ref(), 3, per_message);
            let len = dispatch_body(&session, endpoint.as_ref(), &overrides).len();
            samples.push(measure(
                format!("[{size_label}] turn 3  LIVE chat (cached plan + reply splice)"),
                len,
                ITERS,
                || dispatch_body(&session, endpoint.as_ref(), &overrides),
            ));
        }
    }

    for sample in &samples {
        sample.print();
    }

    // --- Conversation depth against the reply-group inline bound. ---
    //
    // A continuation dispatch collects one `(position, wires)` group per
    // captured reply into a `SmallVec` with 16 inline slots. Up to that depth
    // the collection is stack-only; past it every request heap-allocates it
    // once more. Both sides are measured rather than argued, since "history
    // depth is free" is exactly the kind of claim that is only true up to a
    // bound nobody wrote down.
    println!("\n=== L1b: conversation depth vs. the reply-group inline bound ===");
    let mut depth_samples = Vec::new();
    for depth in [16_usize, 24] {
        let mut pool = SegmentPool::new();
        let turns: Vec<Turn> = (0..=depth)
            .map(|index| user_turn(&mut pool, filler(55, index as u8)))
            .collect();
        let dataset = build_dataset(
            ConversationContextMode::DeltasWithoutResponses,
            turns,
            pool,
            endpoint.as_ref(),
            true,
        );
        let session = session_at(dataset, endpoint.as_ref(), depth, 55);
        let len = dispatch_body(&session, endpoint.as_ref(), &overrides).len();
        depth_samples.push(measure(
            format!("[0.5K/msg] turn {depth:<2} ({depth} replies, inline bound 16)"),
            len,
            ITERS,
            || dispatch_body(&session, endpoint.as_ref(), &overrides),
        ));
    }
    for sample in &depth_samples {
        sample.print();
    }
    samples.extend(depth_samples);

    println!("\n=== L2: image-count re-parse on the dispatched body ===");
    println!(
        "(endpoint_dispatch.rs: serde_json::from_slice::<Value>(&body) when image_count is None)"
    );
    let mut parse_samples = Vec::new();
    for (size_label, per_message) in SIZES {
        let mut pool = SegmentPool::new();
        let turns: Vec<Turn> = (0..4)
            .map(|index| user_turn(&mut pool, filler(per_message, index as u8)))
            .collect();
        let dataset = build_dataset(
            ConversationContextMode::DeltasWithoutResponses,
            turns,
            pool,
            endpoint.as_ref(),
            true,
        );
        let session = session_at(dataset, endpoint.as_ref(), 3, per_message);
        let body = dispatch_body(&session, endpoint.as_ref(), &overrides);
        let len = body.len();

        parse_samples.push(measure(
            format!("[{size_label}] serde_json::from_slice::<Value>"),
            len,
            ITERS,
            || serde_json::from_slice::<Value>(&body).ok(),
        ));

        let parsed: Value = serde_json::from_slice(&body).expect("parse body");
        parse_samples.push(measure(
            format!("[{size_label}] parse + extract_payload_inputs"),
            len,
            ITERS,
            || {
                let value = serde_json::from_slice::<Value>(&body).expect("parse");
                endpoint.extract_payload_inputs(&value).image_count
            },
        ));
        parse_samples.push(measure(
            format!("[{size_label}] extract_payload_inputs only"),
            len,
            ITERS,
            || endpoint.extract_payload_inputs(&parsed).image_count,
        ));

        // Composite: what dispatch actually pays for a turn>=1 chat request —
        // materialize the body, then re-parse it and walk it just to learn that
        // `num_images == 0`.
        parse_samples.push(measure(
            format!("[{size_label}] TOTAL t3 dispatch = materialize + image count"),
            len,
            ITERS,
            || {
                let body = dispatch_body(&session, endpoint.as_ref(), &overrides);
                let value = serde_json::from_slice::<Value>(&body).expect("parse");
                (body, endpoint.extract_payload_inputs(&value).image_count)
            },
        ));

        // Counterfactual: the same dispatch with a known image count, which is
        // what turn 0 already gets.
        parse_samples.push(measure(
            format!("[{size_label}] TOTAL t3 dispatch, image count KNOWN"),
            len,
            ITERS,
            || dispatch_body(&session, endpoint.as_ref(), &overrides),
        ));

        // The real dispatch decision, end to end: whatever the runtime
        // establishes up front decides whether the body is parsed here.
        parse_samples.push(measure(
            format!("[{size_label}] TOTAL t3 dispatch, REAL image-count path"),
            len,
            ITERS,
            || dispatch_body_and_image_count(&session, endpoint.as_ref(), &overrides),
        ));
    }
    for sample in &parse_samples {
        sample.print();
    }
}

// ---------------------------------------------------------------------------
// L2b: borrowed vs consumed wire assembly
// ---------------------------------------------------------------------------

/// The same body path as [`dispatch_body`], consuming the materialized
/// `Request` the way the three owned-destructure dispatch sites do.
fn dispatch_body_consumed(
    session: &ConversationSession,
    endpoint: &dyn PreparedEndpoint,
    overrides: &Overrides,
) -> Bytes {
    session
        .materialize_prepared(
            &EndpointRequestMaterializer,
            endpoint,
            "primary-model",
            CreditPhase::Profiling,
            overrides,
        )
        .expect("materialize")
        .body
        .into_wire()
        .expect("into_wire")
}

/// `RequestBody::to_wire(&self)` against `RequestBody::into_wire(self)` on a
/// freshly materialized body, which is what dispatch holds.
///
/// A `BytesMut::freeze()`-derived body has `len == capacity` and so rides the
/// promotable vtable: the first clone heap-allocates a 24-byte `Shared` block
/// instead of bumping a refcount. Each iteration materializes a new body, so
/// that first-clone promotion is paid every time, exactly as in a run.
///
/// **Two shapes, and they do not give the same answer.** Exactly one promotion
/// is required per body as soon as two handles exist, so what `into_wire` saves
/// depends entirely on how many handles the call site takes:
///
/// - *One handle, then drop* — `grpc/sink.rs`, which `into_wire`s a non-decoded
///   body, parses the bytes and drops them. Consuming removes the only clone,
///   so this banks −1 alloc.
/// - *Two handles* — any HTTP dispatch that keeps a second handle live:
///   `endpoint_dispatch.rs`'s `request_payload` clone when an artifact consumes
///   it, or the multipart re-encode in `prepare_request`, which retains the
///   canonical JSON alongside the form-encoded wire bytes. The promotion merely
///   relocates to that second clone: 1 alloc before, 1 alloc after. What is
///   saved is a refcount inc/dec pair.
///
/// The `SOLE HANDLE` rows model gRPC; the `SECOND HANDLE` rows model HTTP.
///
/// The `HTTP CHAIN` rows model the *whole* endpoint-aware chat path rather than
/// one call site, which is the only way to see whether removing any single
/// clone removes the allocation or merely relocates it. Measured, they do not:
/// `before` and `after` both report 4.0 allocs / 551 alloc-B on the 0.5 KB body
/// while `floor` reports 3.0 / 527, so the promotion is one 24-byte `Shared`
/// block that survives as long as *any* second handle does.
///
/// The three clones those rows model are gone from the JSON chat path:
/// `endpoint_dispatch.rs` clones `canonical_body()` into `request_payload` only
/// when `captures_request_payload()` reports that an artifact consumes it,
/// `HttpTransport::send_body` takes the wire `Bytes` by value and keeps no copy
/// on the record, and `dispatch`/`dispatch_backpressured` consume `self` so
/// `wire_body` moves to the transport instead of being cloned out of a live
/// `prepared`. `prepare_request` retains a second handle only for the multipart
/// re-encode, so an artifact-free JSON dispatch sits at the modelled `floor`.
///
/// These rows are a *model* — synthetic `.clone()` calls in the shape of the
/// real chain, not a call into `prepare_request`/`dispatch`. That is sound for
/// "does removing one clone remove an allocation", but it cannot catch a future
/// allocation regression introduced *inside* those functions.
#[test]
fn chat_dispatch_wire_ownership_profile() {
    println!("\n=== L2b: to_wire(&self) vs into_wire(self) on an owned request ===");
    let endpoint = prepared_chat_endpoint();
    let overrides = Overrides::new();
    let mut samples = Vec::new();

    for (size_label, per_message) in SIZES {
        // Turn 0 with a cached plan: the ordinary-chat fast path, where the
        // clone is the largest share of what is left.
        let mut pool = SegmentPool::new();
        let turns = vec![user_turn(&mut pool, filler(per_message * 7, 0))];
        let dataset = build_dataset(
            ConversationContextMode::MessageArrayWithResponses,
            turns,
            pool,
            endpoint.as_ref(),
            true,
        );
        let session = session_at(dataset, endpoint.as_ref(), 0, 0);
        let len = dispatch_body(&session, endpoint.as_ref(), &overrides).len();
        assert_eq!(
            dispatch_body(&session, endpoint.as_ref(), &overrides),
            dispatch_body_consumed(&session, endpoint.as_ref(), &overrides),
            "into_wire must be byte-identical to to_wire"
        );

        // gRPC shape: the assembled bytes are the only handle taken.
        samples.push(measure(
            format!("[{size_label}] SOLE HANDLE, to_wire   (gRPC, before)"),
            len,
            ITERS,
            || dispatch_body(&session, endpoint.as_ref(), &overrides),
        ));
        samples.push(measure(
            format!("[{size_label}] SOLE HANDLE, into_wire (gRPC, after)"),
            len,
            ITERS,
            || dispatch_body_consumed(&session, endpoint.as_ref(), &overrides),
        ));

        // HTTP shape: a second handle is taken immediately afterwards, the way
        // `endpoint_dispatch.rs`'s gated `request_payload` clone and
        // `prepare_request`'s multipart arm both do. One promotion is owed
        // either way; consuming only moves who pays it.
        samples.push(measure(
            format!("[{size_label}] SECOND HANDLE, to_wire   (HTTP, before)"),
            len,
            ITERS,
            || {
                let body = dispatch_body(&session, endpoint.as_ref(), &overrides);
                let payload = body.clone();
                (body, payload)
            },
        ));
        samples.push(measure(
            format!("[{size_label}] SECOND HANDLE, into_wire (HTTP, after)"),
            len,
            ITERS,
            || {
                let body = dispatch_body_consumed(&session, endpoint.as_ref(), &overrides);
                let payload = body.clone();
                (body, payload)
            },
        ));

        // The full endpoint-aware chat chain, every handle that is live at once.
        //
        // The four-handle shape this row models: an assembled body kept as
        // `canonical_body`, a `request_payload` clone off it, a `wire_body`
        // clone for the transport, and one more clone onto the record. Only the
        // first clone allocates — it installs the shared control block — so a
        // removed clone buys an allocation only when no later clone survives
        // it, which is why the `after` row below still costs what `before` did.
        samples.push(measure(
            format!("[{size_label}] HTTP CHAIN before: canonical+payload+wire+record"),
            len,
            ITERS,
            || {
                let body = dispatch_body_consumed(&session, endpoint.as_ref(), &overrides);
                let canonical = body.clone();
                let wire = body;
                let payload = canonical.clone();
                let sent = wire.clone();
                let recorded = sent.clone();
                (canonical, wire, payload, sent, recorded)
            },
        ));
        samples.push(measure(
            format!("[{size_label}] HTTP CHAIN after:  payload+wire+record"),
            len,
            ITERS,
            || {
                let wire = dispatch_body_consumed(&session, endpoint.as_ref(), &overrides);
                let payload = wire.clone();
                let sent = wire.clone();
                let recorded = sent.clone();
                (wire, payload, sent, recorded)
            },
        ));
        // The floor the chain reaches when the wire handle is the only one: no
        // promotion at all. The gap to the two rows above is not attributable to
        // any single clone — it takes `request_payload` gated, no record clone,
        // *and* a dispatch handoff that consumes `wire_body` rather than cloning
        // it out of a live `prepared`, which is where the path stands now.
        samples.push(measure(
            format!("[{size_label}] HTTP CHAIN floor:  sole handle (no second handle)"),
            len,
            ITERS,
            || dispatch_body_consumed(&session, endpoint.as_ref(), &overrides),
        ));
    }

    for sample in &samples {
        sample.print();
    }
}

// ---------------------------------------------------------------------------
// L3/L4: full turn build and the IssuedCredit deep clone
// ---------------------------------------------------------------------------

fn source_for(
    mode: ConversationContextMode,
    turns: Vec<Turn>,
    pool: SegmentPool,
) -> NativeDatasetConversationSource {
    let endpoint = prepared_chat_endpoint();
    let mut conversation = Conversation::new("session");
    conversation.context_mode = Some(mode);
    conversation.turns = turns;
    let dataset = Dataset::new(
        vec![conversation],
        Arc::new(pool.freeze()),
        "sequential",
        mode,
    )
    .expect("dataset");
    let mut table = PreparedEndpointTable::new();
    let endpoint_id = EndpointId::new("chat").expect("chat id");
    let key = table.push(endpoint).expect("push endpoint");
    NativeDatasetConversationSource::sequential_with_prepared_endpoint(
        dataset,
        "primary-model",
        128,
        Rc::new(table),
        PreparedEndpointReference { key, endpoint_id },
    )
    .expect("source")
}

fn turn_response(text: String) -> TurnResponse {
    TurnResponse {
        text,
        assistant_message: None,
        completion_tokens: Some(64),
        terminal: ReplayTerminalStatus::Completed,
    }
}

#[test]
fn chat_turn_build_and_credit_clone_profile() {
    println!("\n=== L3/L4: full turn build + IssuedCredit deep clone ===");
    let mut samples = Vec::new();

    for (size_label, per_message) in SIZES {
        let mut pool = SegmentPool::new();
        let turns: Vec<Turn> = (0..4)
            .map(|index| user_turn(&mut pool, filler(per_message, index as u8)))
            .collect();
        let mut source = source_for(ConversationContextMode::DeltasWithoutResponses, turns, pool);

        // Walk to turn 2, keeping its credit: `next_turn` short-circuits to
        // `None` on a final credit, so the turn-3 build must be driven from the
        // turn-2 credit, not the turn-3 one.
        let (turn2, turn3) = {
            let session = source.next(None).expect("sample session");
            let mut turn = session.build_first_turn(None).expect("first turn");
            for index in 0..2u64 {
                let credit = IssuedCredit::from_turn(index, 0, &turn);
                turn = source
                    .next_turn(
                        &credit,
                        turn_response(filler(per_message, 210 + index as u8)),
                    )
                    .expect("next turn")
                    .expect("non-final");
            }
            let turn2 = turn;
            let credit2 = IssuedCredit::from_turn(2, 0, &turn2);
            let turn3 = source
                .next_turn(&credit2, turn_response(filler(per_message, 212)))
                .expect("next turn")
                .expect("non-final");
            (turn2, turn3)
        };
        println!(
            "  (context: turn3.messages.len()={}, headers={}, params={})",
            turn3.messages.len(),
            turn3.request_headers.len(),
            turn3.request_parameters.len(),
        );
        let body_len = match &turn3.request_body {
            Some(RequestBody::Wire(bytes)) => bytes.len(),
            _ => 0,
        };

        samples.push(measure(
            format!("[{size_label}] IssuedCredit::from_turn (TurnToSend clone) t3"),
            body_len,
            ITERS,
            || IssuedCredit::from_turn(0, 0, &turn3),
        ));

        let turn0 = {
            let session = source.next(None).expect("sample session");
            session.build_first_turn(None).expect("first turn")
        };
        let turn0_len = match &turn0.request_body {
            Some(RequestBody::Wire(bytes)) => bytes.len(),
            _ => 0,
        };
        samples.push(measure(
            format!("[{size_label}] IssuedCredit::from_turn (TurnToSend clone) t0"),
            turn0_len,
            ITERS,
            || IssuedCredit::from_turn(0, 0, &turn0),
        ));

        samples.push(measure(
            format!("[{size_label}] build_first_turn (full turn 0 build)"),
            turn0_len,
            ITERS / 4,
            || {
                let session = source.next(None).expect("sample session");
                session.build_first_turn(None).expect("first turn")
            },
        ));

        // `next_turn` advances the sampled session's interior state, so it
        // cannot be replayed against one credit. Measure a whole fresh 4-turn
        // chain instead; subtract the `build_first_turn` row and divide by 3 for
        // the per-continuation cost.
        let _ = &turn2;
        samples.push(measure(
            format!("[{size_label}] full 4-turn chain (t0 + 3 continuations)"),
            body_len,
            ITERS / 8,
            || {
                let session = source.next(None).expect("sample session");
                let mut turn = session.build_first_turn(None).expect("first turn");
                for index in 0..3u64 {
                    let credit = IssuedCredit::from_turn(index, 0, &turn);
                    turn = source
                        .next_turn(
                            &credit,
                            turn_response(filler(per_message, 210 + index as u8)),
                        )
                        .expect("next turn")
                        .expect("non-final");
                }
                turn
            },
        ));
    }

    for sample in &samples {
        sample.print();
    }
}
