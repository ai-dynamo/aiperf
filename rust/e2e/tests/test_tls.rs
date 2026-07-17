// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Full-stack e2e for the mock server's TLS/HTTPS frontend.
//!
//! Runs the real `python -m aiperf profile` CLI (native `aiperf` + its
//! production Clock-injected hyper HTTPS client) against an
//! `aiperf-mock-server` HTTPS listener terminated by a fresh in-memory
//! self-signed certificate ([`aiperf_mock_server::tls::self_signed_acceptor`] —
//! the exact server side `--tls-self-signed` stands up). The client trusts the
//! self-signed cert by disabling verification via `endpoint.ssl_verify: false`,
//! which the runner's HTTP client honors by installing a
//! `NoCertificateVerification` verifier
//! (`aiperf_runtime::transport::http::client::connection::rustls_config`,
//! `rust/aiperf/src/transport::http/client/connection.rs:326-347`) — the Rust
//! equivalent of the Python connector's `ssl=False`. The listener advertises
//! ALPN `h2`+`http/1.1`; the client negotiates one and streams SSE over TLS.
//!
//! # `grpcs` through `aiperf profile`
//!
//! `grpcs` is also driven end to end here
//! ([`grpcs_kserve_infer_via_aiperf_profile_raw_records`]). The runner's tonic
//! client now honors `endpoint.ssl_verify=false` by installing the SAME
//! `NoCertificateVerification` verifier the HTTP transport uses, via tonic's
//! `Endpoint::tls_config_with_verifier`
//! (`rust/aiperf/src/transport::grpc/transport.rs`), so a `grpcs://` run against
//! the self-signed mock completes instead of failing the handshake against
//! system roots.
//!
//! # The tuned-mock raw-record bar
//!
//! The mock is tuned to fixed, jitter-free per-token latency (analytic mode) so
//! every `profile_export_raw.jsonl` record's on-the-wire timing (TTFT / ITL /
//! request_latency) and data (OSL / model / status) reproduces the tuned model
//! within a tight transport tolerance — the same `assert_raw_records_*` bar the
//! cleartext core-path e2es use, now proven end-to-end over TLS.
//!
//! # Environment caveat (must run un-sandboxed)
//!
//! Like the other tuned-timing e2es, the mock injects latency through the
//! RealClock `timerfd`; a timer-virtualizing sandbox collapses the sleeps and
//! the timing assertions self-skip via [`common::timing_fast_forwarded`].

mod common;
use common::*;

use std::net::{TcpListener as StdTcpListener, TcpStream};
use std::time::Duration;

use aiperf_mock_server::{AppState, MockServerConfig, build_router, tls};

/// Tuned mock TTFT (ms). Matches the cleartext tuned-timing reference point.
const TTFT_MS: f64 = 100.0;
/// Tuned mock ITL (ms).
const ITL_MS: f64 = 10.0;
/// Fixed output cap (exact generation via `osl: {mean, stddev: 0}`).
const OSL: usize = 8;
/// Request count for the single-phase run.
const REQUESTS: u32 = 6;
/// Concurrency for the single-phase run.
const CONCURRENCY: u32 = 2;

/// An in-process `aiperf-mock-server` HTTPS listener bound to a random loopback
/// port, terminated by a fresh self-signed cert. Mirrors the harness's cleartext
/// `MockServer` but wraps every accepted stream in the shared
/// [`tls::serve_http`] loop with a self-signed acceptor — the identical code
/// path the `--tls-self-signed` binary runs.
struct TlsMockServer {
    /// Base URL, e.g. `https://127.0.0.1:<port>`.
    url: String,
    /// Owned runtime driving the TLS accept loop; dropping it stops the server.
    runtime: Option<tokio::runtime::Runtime>,
}

impl TlsMockServer {
    fn start(cfg: MockServerConfig) -> Self {
        let cfg = cfg.apply_flags();

        // Bind synchronously so the port is known and listening before the
        // accept loop adopts the socket.
        let std_listener = StdTcpListener::bind("127.0.0.1:0").expect("bind https mock listener");
        let port = std_listener.local_addr().expect("listener addr").port();
        std_listener
            .set_nonblocking(true)
            .expect("set listener nonblocking");

        let state: std::sync::Arc<AppState> = AppState::build(cfg);
        let router = build_router(state);
        let acceptor = tls::self_signed_acceptor().expect("build self-signed acceptor");

        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .expect("build https mock runtime");
        runtime.spawn(async move {
            let listener = tokio::net::TcpListener::from_std(std_listener)
                .expect("adopt std listener into tokio");
            // Same accept loop main.rs uses; a serve error surfaces to the test
            // as a connection failure in the aiperf subprocess.
            let _ = tls::serve_http(listener, router, Some(acceptor), 0).await;
        });

        wait_for_tcp(port);
        Self {
            url: format!("https://127.0.0.1:{port}"),
            runtime: Some(runtime),
        }
    }
}

impl Drop for TlsMockServer {
    fn drop(&mut self) {
        if let Some(rt) = self.runtime.take() {
            rt.shutdown_background();
        }
    }
}

/// Poll a raw TCP connect until the TLS listener accepts (the rustls handshake
/// itself is exercised by the aiperf run). 50 tries, 100 ms apart.
fn wait_for_tcp(port: u16) {
    for _ in 0..50 {
        if TcpStream::connect(("127.0.0.1", port)).is_ok() {
            // Small grace so the spawned accept task is polling before the first
            // real TLS handshake races it.
            std::thread::sleep(Duration::from_millis(100));
            return;
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    panic!("https mock server on port {port} never became reachable");
}

/// A Config-v2 YAML selecting the native HTTP transport against an `https://`
/// URL with certificate verification DISABLED, tuned synthetic ISL/OSL, and raw
/// per-record export. The harness appends `--artifact-dir` and `--tokenizer`.
fn https_config(url: &str) -> String {
    format!(
        "schemaVersion: \"2.0\"\n\
         benchmark:\n\
        \x20 model: gpt-4\n\
        \x20 endpoint:\n\
        \x20   url: {url}\n\
        \x20   type: chat\n\
        \x20   streaming: true\n\
        \x20   sslVerify: false\n\
        \x20   waitForModelTimeout: 0.0\n\
        \x20 dataset:\n\
        \x20   type: synthetic\n\
        \x20   entries: {REQUESTS}\n\
        \x20   prompts:\n\
        \x20     isl: {{mean: 64, stddev: 0}}\n\
        \x20     osl: {{mean: {OSL}, stddev: 0}}\n\
        \x20 phases:\n\
        \x20   - name: profiling\n\
        \x20     type: concurrency\n\
        \x20     requests: {REQUESTS}\n\
        \x20     concurrency: {CONCURRENCY}\n\
        \x20 gpuTelemetry: {{enabled: false}}\n\
        \x20 serverMetrics: {{enabled: false}}\n\
        \x20 artifacts:\n\
        \x20   raw: true\n\
        \x20   records:\n\
        \x20     - jsonl\n\
        \x20 runtime:\n\
        \x20   ui: none\n"
    )
}

/// HTTPS single-turn tuned raw-record e2e: `aiperf profile` streams SSE over TLS
/// against the self-signed mock with `ssl_verify=false`, and every raw record
/// reproduces the tuned TTFT/ITL/latency + OSL/model/status.
#[tokio::test]
async fn tuned_https_single_turn_raw_timing() {
    if cfg!(target_os = "macos") {
        return; // artifact e2es are flaky on macOS CI
    }

    // The harness supplies the subprocess machinery; its own cleartext mock is
    // unused — the config points the run at the separate HTTPS listener below.
    let h = AIPerfHarness::new().await;
    let tls_mock = TlsMockServer::start(tuned_mock_config(TTFT_MS, ITL_MS));

    let cfg_file = h.artifact_path().join("https_tuned.yaml");
    std::fs::write(&cfg_file, https_config(&tls_mock.url)).expect("write https config");

    let r = h.run(&format!("--config {}", cfg_file.display()));
    assert!(
        r.success(),
        "https tuned run failed (exit {}):\nstdout:\n{}\nstderr:\n{}",
        r.exit_code,
        r.stdout,
        r.stderr
    );

    let records = r.artifacts.raw_records();
    assert_eq!(
        records.len(),
        REQUESTS as usize,
        "expected {REQUESTS} raw records over HTTPS"
    );

    // DATA proof independent of the timer environment: every record is an HTTP
    // 200 over TLS carrying exactly OSL streamed content chunks for `gpt-4`.
    for (index, record) in records.iter().enumerate() {
        let timing = extract_timing(record);
        assert_eq!(
            timing.status,
            Some(200),
            "record {index}: HTTPS status {:?} != 200",
            timing.status
        );
        assert_eq!(
            timing.osl, OSL,
            "record {index}: OSL {} != {OSL} over TLS",
            timing.osl
        );
        assert_eq!(
            timing.model.as_deref(),
            Some("gpt-4"),
            "record {index}: model {:?} != gpt-4",
            timing.model
        );
    }

    if timing_fast_forwarded(&records, TTFT_MS) {
        return;
    }
    // TIMING proof: the full tuned TTFT/ITL/latency bar, now over TLS. TTFT gets
    // more headroom than the cleartext path because it folds in the extra rustls
    // handshake round-trip on connection setup (measured ~19 ms on loopback);
    // ITL is steady-state per-token pacing unaffected by TLS, so it stays tight.
    assert_raw_records_timing_and_data(
        &records,
        &TunedExpectations::new(TTFT_MS, ITL_MS, OSL)
            .model("gpt-4")
            .tol_ms(40.0, 2.0),
    );
}

// ============================================================================
// grpcs — the runner's tonic client with ssl_verify=false against a self-signed
// KServe gRPC listener, driven through the full product path.
// ============================================================================

/// Requests for the single-phase grpcs run.
const GRPCS_REQUESTS: u32 = 6;

/// Start an in-process `grpcs` KServe gRPC listener (TLS self-signed) on its OWN
/// runtime thread — the caller's `#[tokio::test]` runtime is blocked on the
/// synchronous `aiperf` subprocess during the run, so the accept loop must live
/// elsewhere. Returns the `grpcs://127.0.0.1:PORT` URL.
fn start_grpcs_mock() -> String {
    let reserved = StdTcpListener::bind("127.0.0.1:0").expect("reserve grpcs port");
    let addr = reserved.local_addr().unwrap();
    drop(reserved);

    let cfg = MockServerConfig {
        fast: true,
        no_tokenizer: true,
        ..MockServerConfig::default()
    }
    .apply_flags();
    aiperf_mock_server::tokens::load_corpus();

    std::thread::spawn(move || {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .expect("grpcs mock runtime");
        rt.block_on(async move {
            let acceptor = tls::self_signed_acceptor().expect("self-signed acceptor");
            let state = AppState::build(cfg);
            let _ =
                aiperf_mock_server::grpc::serve_grpc_with_tls(addr, state, Some(acceptor)).await;
        });
    });
    std::thread::sleep(Duration::from_millis(400));
    format!("grpcs://127.0.0.1:{}", addr.port())
}

/// Config-v2 YAML: a KServe v2 infer endpoint over `grpcs://` with cert
/// verification disabled and readiness probing off.
fn grpcs_config(url: &str) -> String {
    format!(
        "schemaVersion: \"2.0\"\n\
         benchmark:\n\
        \x20 models: [{DEFAULT_MODEL}]\n\
        \x20 endpoint:\n\
        \x20   urls: [\"{url}\"]\n\
        \x20   type: kserve_v2_infer\n\
        \x20   streaming: false\n\
        \x20   sslVerify: false\n\
        \x20   waitForModelTimeout: 0.0\n\
        \x20 dataset:\n\
        \x20   type: synthetic\n\
        \x20   entries: {GRPCS_REQUESTS}\n\
        \x20   prompts:\n\
        \x20     isl: 16\n\
        \x20     osl: 8\n\
        \x20 phases:\n\
        \x20   - name: profiling\n\
        \x20     type: concurrency\n\
        \x20     requests: {GRPCS_REQUESTS}\n\
        \x20     concurrency: 2\n\
        \x20 gpuTelemetry: {{enabled: false}}\n\
        \x20 serverMetrics: {{enabled: false}}\n\
        \x20 transport:\n\
        \x20   type: grpc\n\
        \x20 runtime:\n\
        \x20   ui: none\n"
    )
}

/// `aiperf profile` runs KServe v2 infer over `grpcs://` against the self-signed
/// mock with `sslVerify: false`. A successful raw-record export proves the tonic
/// client completed the TLS handshake with the insecure verifier installed.
#[tokio::test]
async fn grpcs_kserve_infer_via_aiperf_profile_raw_records() {
    if cfg!(target_os = "macos") {
        return;
    }
    let url = start_grpcs_mock();

    let h = AIPerfHarness::new().await; // subprocess machinery; its cleartext mock is unused
    let cfg_file = h.artifact_path().join("grpcs.yaml");
    std::fs::write(&cfg_file, grpcs_config(&url)).expect("write grpcs config");

    let r = h.run(&format!(
        "--config {} --export-level raw",
        cfg_file.display()
    ));
    assert!(
        r.success(),
        "grpcs run failed (exit {}):\nstdout:\n{}\nstderr:\n{}",
        r.exit_code,
        r.stdout,
        r.stderr
    );

    let records = r.artifacts.raw_records();
    assert_eq!(
        records.len(),
        GRPCS_REQUESTS as usize,
        "one raw record per grpcs request"
    );
    for (i, rec) in records.iter().enumerate() {
        let errored = rec.get("error").map(|e| !e.is_null()).unwrap_or(false);
        assert!(
            !errored,
            "record {i} errored over grpcs: {:?}",
            rec.get("error")
        );
        let has_response = rec
            .get("responses")
            .and_then(|v| v.as_array())
            .map(|a| !a.is_empty())
            .unwrap_or(false);
        assert!(has_response, "record {i} has no gRPC response: {rec}");
    }
}
