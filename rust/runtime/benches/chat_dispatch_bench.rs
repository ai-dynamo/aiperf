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
//! `--test-threads=1` is required: the counting allocator is process-global.

use std::alloc::{GlobalAlloc, Layout, System};
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
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

// ---------------------------------------------------------------------------
// Counting allocator
// ---------------------------------------------------------------------------

static ALLOC_COUNT: AtomicU64 = AtomicU64::new(0);
static ALLOC_BYTES: AtomicU64 = AtomicU64::new(0);
static COUNTING: AtomicBool = AtomicBool::new(false);

struct CountingAlloc;

unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if COUNTING.load(Ordering::Relaxed) {
            ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
            ALLOC_BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
        }
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if COUNTING.load(Ordering::Relaxed) {
            ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
            ALLOC_BYTES.fetch_add(
                new_size.saturating_sub(layout.size()) as u64,
                Ordering::Relaxed,
            );
        }
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: CountingAlloc = CountingAlloc;

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
}

/// Time and count allocations for `iters` runs of `body`, after a warm-up pass.
fn measure<T>(
    label: impl Into<String>,
    body_bytes: usize,
    iters: u64,
    mut body: impl FnMut() -> T,
) -> Sample {
    // Warm caches, lazy statics, and any one-shot interning.
    for _ in 0..(iters / 10).max(16) {
        std::hint::black_box(body());
    }

    ALLOC_COUNT.store(0, Ordering::Relaxed);
    ALLOC_BYTES.store(0, Ordering::Relaxed);
    COUNTING.store(true, Ordering::Relaxed);
    let start = Instant::now();
    for _ in 0..iters {
        std::hint::black_box(body());
    }
    let elapsed = start.elapsed();
    COUNTING.store(false, Ordering::Relaxed);
    let allocs = ALLOC_COUNT.load(Ordering::Relaxed);
    let bytes = ALLOC_BYTES.load(Ordering::Relaxed);

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
/// user message. This is the `MessageArrayWithResponses` shape, and the only
/// shape whose non-zero turns are served from the precomputed plan cache.
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
    EndpointRegistry::builtin()
        .expect("builtin endpoint registry")
        .prepare(
            &EndpointId::new("chat").expect("chat endpoint id"),
            RawEndpointConfig {
                streaming: true,
                ..RawEndpointConfig::default()
            },
        )
        .expect("prepare chat endpoint")
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
        // mode and the ordinary chat case; only its turn 0 is ever cached, so
        // this always runs the live format_payload path. ---
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
                format!("[{size_label}] turn 3  LIVE chat (always uncached)"),
                len,
                ITERS,
                || dispatch_body(&session, endpoint.as_ref(), &overrides),
            ));
        }
    }

    for sample in &samples {
        sample.print();
    }

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
/// - *One handle, then drop* — `grpc/sink.rs:344`, which parses the bytes and
///   drops them. Consuming removes the only clone, so this banks −1 alloc.
/// - *Two handles* — both HTTP sites. `http/sink.rs:391` takes `body.clone()`
///   for the record payload, and the endpoint-aware path reaches
///   `HttpTransport::send_body`, which clones into `RequestRecord::request_body`
///   unconditionally. The promotion merely relocates to that second clone:
///   1 alloc before, 1 alloc after. What is saved is a refcount inc/dec pair.
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
/// Post-change the promotion is paid by the *first* clone in the chain,
/// `endpoint_dispatch.rs:281`'s `canonical_body().clone()` for `request_payload`.
/// Everything after it is a refcount bump, so deleting any single later clone —
/// including the record clone in `HttpTransport::send_body` — buys nothing. The
/// floor needs all three gone at once: gate `request_payload`, drop the record
/// clone, and let `dispatch`/`dispatch_backpressured` consume `self` instead of
/// cloning `wire_body` out of a still-live `prepared`. That last one is a plain
/// ownership artifact of the transport handoff, not a raw-artifact cost.
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
        // `http/sink.rs:391` and `endpoint_binding.rs:297` both do. One
        // promotion is owed either way; consuming only moves who pays it.
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
        // Four handles were taken per dispatch: `prepare_request` cloned the
        // assembled body into `canonical_body`, `endpoint_dispatch.rs` cloned
        // that into `request_payload`, `PreparedHttpEndpointRequest::dispatch`
        // cloned `wire_body` for the transport, and `HttpTransport::send_body`
        // cloned it once more into `RequestRecord::request_body`. Only the first
        // clone allocates — it installs the shared control block — so a removed
        // clone buys an allocation only when no later clone survives it.
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
        // The floor the chain would reach if the wire handle were the only one:
        // no promotion at all. The gap to the two rows above is not attributable
        // to any single clone — it needs `request_payload` gated, the record
        // clone dropped, *and* the dispatch handoff to consume `wire_body`
        // rather than clone it out of a live `prepared`.
        samples.push(measure(
            format!("[{size_label}] HTTP CHAIN floor:  sole handle (needs all 3 gone)"),
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
