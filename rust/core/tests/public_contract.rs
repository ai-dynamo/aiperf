// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Public contract of the boundary-owned core.
//!
//! Every import here is from `aiperf_core` alone. The test crate declares no
//! dependency on `aiperf-runtime`, so a regression that pulls a runtime type
//! back into the boundary fails this file at compile time rather than at
//! review time.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::future::Future;
use std::pin::Pin;
use std::rc::Rc;

use aiperf_core::artifact::{ArtifactAccess, ArtifactEntry, ArtifactError};
use aiperf_core::clock::{Clock, RunOutcome};
use aiperf_core::dispatch::{
    Dispatchable, ObservedTokenKind, ObservedUsage, ReplayTerminalStatus, RequestObserver,
    RequestSink,
};
use aiperf_core::endpoint::{
    Handle, Overrides, PreparedWsMessage, PreparedWsMessageRole, PreparedWsOpcode,
    PreparedWsOperation, SegmentReader,
};
use aiperf_core::measure::{ErrorDetails, ErrorKind, Response, SseMessage, TextResponse};
use aiperf_core::report::write_finalized_report_json;
use bytes::Bytes;
use uuid::Uuid;

/// A plugin-authored clock that needs no runtime crate.
struct FakeClock {
    now_ns: RefCell<i64>,
}

impl Clock for FakeClock {
    fn now_ns(&self) -> i64 {
        *self.now_ns.borrow()
    }

    fn sleep(self: Rc<Self>, duration_ns: i64) -> Pin<Box<dyn Future<Output = ()>>> {
        *self.now_ns.borrow_mut() += duration_ns.max(0);
        Box::pin(std::future::ready(()))
    }

    fn is_virtual(&self) -> bool {
        true
    }

    fn drive(self: Rc<Self>, _body: Pin<Box<dyn Future<Output = ()> + '_>>) -> RunOutcome {
        RunOutcome { deadlocked: false }
    }
}

/// An in-memory capability-limited artifact store.
#[derive(Default)]
struct FakeArtifacts {
    entries: RefCell<BTreeMap<String, Vec<u8>>>,
}

impl ArtifactAccess for FakeArtifacts {
    fn list(&self) -> Result<Vec<ArtifactEntry>, ArtifactError> {
        Ok(self
            .entries
            .borrow()
            .iter()
            .map(|(relative_path, bytes)| ArtifactEntry {
                relative_path: relative_path.clone(),
                len: bytes.len() as u64,
            })
            .collect())
    }

    fn read(&self, relative_path: &str) -> Result<Vec<u8>, ArtifactError> {
        aiperf_core::artifact::check_relative(relative_path)?;
        self.entries
            .borrow()
            .get(relative_path)
            .cloned()
            .ok_or_else(|| ArtifactError::Rejected(format!("no artifact {relative_path}")))
    }

    fn create(&self, relative_path: &str, contents: &[u8]) -> Result<(), ArtifactError> {
        aiperf_core::artifact::check_relative(relative_path)?;
        self.entries
            .borrow_mut()
            .insert(relative_path.to_owned(), contents.to_vec());
        Ok(())
    }

    fn append(&self, relative_path: &str, contents: &[u8]) -> Result<(), ArtifactError> {
        aiperf_core::artifact::check_relative(relative_path)?;
        self.entries
            .borrow_mut()
            .entry(relative_path.to_owned())
            .or_default()
            .extend_from_slice(contents);
        Ok(())
    }
}

/// Compile-time proof that `ArtifactAccess` exposes no `raw_path()`.
///
/// This blanket extension is the only `raw_path` in scope. If the trait or an
/// inherent implementation ever grew one, the call below would become
/// ambiguous and this file would stop compiling.
trait NoRawPathEscapeHatch {
    fn raw_path(&self) -> &'static str {
        "aiperf-core exposes no raw artifact path"
    }
}

impl<T: ArtifactAccess> NoRawPathEscapeHatch for T {}

struct TinyRequest {
    uuid: Uuid,
}

impl Dispatchable for TinyRequest {
    fn uuid(&self) -> Uuid {
        self.uuid
    }
    fn input_length(&self) -> usize {
        3
    }
    fn max_output_tokens(&self) -> usize {
        2
    }
}

#[derive(Default)]
struct CountingObserver {
    tokens: RefCell<Vec<f64>>,
    terminal: RefCell<Option<ReplayTerminalStatus>>,
}

impl RequestObserver for CountingObserver {
    fn on_arrival(&self, _uuid: Uuid, _arrival_ms: f64, _input: usize, _requested: usize) {}
    fn on_admit(&self, _uuid: Uuid, _admit_ms: f64, _reused: usize) {}
    fn on_token(&self, _uuid: Uuid, at_ms: f64) {
        self.tokens.borrow_mut().push(at_ms);
    }
    fn on_usage(&self, _uuid: Uuid, _usage: ObservedUsage) {}
    fn on_terminal(&self, _uuid: Uuid, status: ReplayTerminalStatus) {
        *self.terminal.borrow_mut() = Some(status);
    }
}

struct EchoSink;

#[async_trait::async_trait(?Send)]
impl RequestSink<TinyRequest> for EchoSink {
    async fn dispatch(&self, req: TinyRequest, obs: &dyn RequestObserver) -> anyhow::Result<()> {
        obs.on_arrival(req.uuid(), 0.0, req.input_length(), req.max_output_tokens());
        obs.on_classified_token(req.uuid(), 1.0, ObservedTokenKind::Output);
        obs.on_output_tokens(req.uuid(), &[2.0]);
        obs.on_terminal(req.uuid(), ReplayTerminalStatus::Completed);
        Ok(())
    }
}

/// A plugin-side segment reader that owns its own storage.
struct FakeSegments(Vec<Bytes>);

impl SegmentReader for FakeSegments {
    fn wire(&self, handle: Handle) -> Option<Bytes> {
        self.0.get(handle.as_usize()).cloned()
    }
}

#[test]
fn a_plugin_clock_satisfies_the_core_clock_contract() {
    let clock: Rc<dyn Clock> = Rc::new(FakeClock {
        now_ns: RefCell::new(7),
    });
    assert_eq!(clock.now_ns(), 7);
    assert!(clock.is_virtual());
    let outcome = clock.drive(Box::pin(std::future::ready(())));
    assert!(!outcome.deadlocked);
}

#[test]
fn artifact_access_is_capability_limited() {
    let artifacts = FakeArtifacts::default();
    artifacts.create("summary.json", b"{}").unwrap();
    artifacts.append("summary.json", b"\n").unwrap();
    assert_eq!(artifacts.read("summary.json").unwrap(), b"{}\n");
    assert_eq!(artifacts.list().unwrap().len(), 1);

    // The trait admits only approved relative paths.
    for rejected in ["/etc/passwd", "../escape.json", "", "a/../../b"] {
        assert!(
            artifacts.read(rejected).is_err(),
            "artifact access accepted {rejected:?}"
        );
    }

    // No `raw_path()` exists on the capability, so the blanket extension wins.
    assert_eq!(
        artifacts.raw_path(),
        "aiperf-core exposes no raw artifact path"
    );
}

#[test]
fn the_dispatch_seam_is_worker_local() {
    let observer = CountingObserver::default();
    let request = TinyRequest {
        uuid: Uuid::from_u128(11),
    };
    futures_lite_block_on(EchoSink.dispatch(request, &observer)).unwrap();
    assert_eq!(*observer.tokens.borrow(), vec![1.0, 2.0]);
    assert_eq!(
        *observer.terminal.borrow(),
        Some(ReplayTerminalStatus::Completed)
    );
}

#[test]
fn measurement_values_are_boundary_owned() {
    let response = Response::Text(TextResponse {
        perf_ns: 42,
        text: "hello".to_owned(),
        body: Bytes::from_static(b"hello"),
        content_type: Some("text/plain".to_owned()),
    });
    assert_eq!(response.perf_ns(), 42);
    assert!(matches!(response, Response::Text(_)));

    let message = SseMessage::parse("data: {\"a\":1}", 7);
    assert_eq!(message.perf_ns, 7);
    assert!(!message.packets.is_empty());

    let error = ErrorDetails::http(503, "boom");
    assert_eq!(error.kind, ErrorKind::Http);
    assert_eq!(error.code, Some(503));
}

#[test]
fn endpoint_values_are_store_free() {
    let mut overrides = Overrides::new();
    overrides.set_model("m");
    overrides.set_stream(true);
    assert!(!overrides.is_empty());
    assert!(!overrides.inner_bytes().unwrap().is_empty());

    let operation = PreparedWsOperation::new(
        [PreparedWsMessage::new(
            PreparedWsOpcode::Text,
            Bytes::from_static(b"{}"),
            PreparedWsMessageRole::MeasuredInput,
        )],
        None,
    );
    assert_eq!(operation.messages().len(), 1);
    assert!(operation.to_artifact_bytes().is_ok());

    let segments = FakeSegments(vec![Bytes::from_static(b"{\"role\":\"user\"}")]);
    assert!(segments.wire(Handle::new(0)).is_some());
    assert!(segments.wire(Handle::new(1)).is_none());
}

#[test]
fn a_finalized_report_projection_commits_exactly_once() {
    let directory = std::env::temp_dir().join(format!(
        "aiperf_core_public_contract_{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&directory).unwrap();
    let path = directory.join("report.json");
    let _ = std::fs::remove_file(&path);

    write_finalized_report_json(&serde_json::json!({"schema_version": "2.0"}), &path).unwrap();
    let committed = std::fs::read_to_string(&path).unwrap();
    assert!(committed.contains("\"schema_version\": \"2.0\""));

    // The commit never replaces an existing authority.
    assert!(write_finalized_report_json(&serde_json::json!({}), &path).is_err());

    let _ = std::fs::remove_dir_all(&directory);
}

/// Minimal executor: the dispatch seam is `?Send` and must stay drivable
/// without pulling a runtime into the contract test.
fn futures_lite_block_on<T>(future: impl Future<Output = T>) -> T {
    use std::sync::Arc;
    use std::task::{Context, Poll, Wake, Waker};

    struct NoopWaker;
    impl Wake for NoopWaker {
        fn wake(self: Arc<Self>) {}
    }

    let mut future = std::pin::pin!(future);
    let waker = Waker::from(Arc::new(NoopWaker));
    let mut context = Context::from_waker(&waker);
    loop {
        match future.as_mut().poll(&mut context) {
            Poll::Ready(value) => return value,
            Poll::Pending => std::thread::yield_now(),
        }
    }
}
