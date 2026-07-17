// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

mod common;
use common::*;

use std::net::{TcpListener as StdTcpListener, TcpStream};
use std::time::Duration;

use aiperf_mock_server::{AppState, MockServerConfig, build_router, tls};

const TTFT_MS: f64 = 100.0;
const ITL_MS: f64 = 10.0;
const OSL: usize = 8;
const REQUESTS: u32 = 6;
const CONCURRENCY: u32 = 2;

struct TlsMockServer {
    url: String,
    runtime: Option<tokio::runtime::Runtime>,
}

impl TlsMockServer {
    fn start(cfg: MockServerConfig) -> Self {
        let cfg = cfg.apply_flags();

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
        // A separate runtime keeps the listener live while profile execution blocks.
        runtime.spawn(async move {
            let listener = tokio::net::TcpListener::from_std(std_listener)
                .expect("adopt std listener into tokio");
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

fn wait_for_tcp(port: u16) {
    for _ in 0..50 {
        if TcpStream::connect(("127.0.0.1", port)).is_ok() {
            std::thread::sleep(Duration::from_millis(100));
            return;
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    panic!("https mock server on port {port} never became reachable");
}

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

#[tokio::test]
async fn tuned_https_single_turn_raw_timing() {
    if cfg!(target_os = "macos") {
        return;
    }

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
    // TLS setup affects TTFT but not steady-state token pacing.
    assert_raw_records_timing_and_data(
        &records,
        &TunedExpectations::new(TTFT_MS, ITL_MS, OSL)
            .model("gpt-4")
            .tol_ms(40.0, 2.0),
    );
}

const GRPCS_REQUESTS: u32 = 6;

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

    // The listener must outlive blocking profile execution on the test runtime.
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

#[tokio::test]
async fn grpcs_kserve_infer_via_aiperf_profile_raw_records() {
    if cfg!(target_os = "macos") {
        return;
    }
    let url = start_grpcs_mock();

    let h = AIPerfHarness::new().await;
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
