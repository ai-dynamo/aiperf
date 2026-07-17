// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Protocol-v2 HTTP client policy coverage.
//!
//! These tests execute scheduled runs through an `aiperf --execute` subprocess.
//! Defaults and insecure TLS behavior are verified without protocol v1 or a
//! leaf-only transport API.

use std::convert::Infallible;
use std::io::Write;
use std::process::{Command, Output, Stdio};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use bytes::Bytes;
use http_body_util::Full;
use hyper::service::service_fn;
use hyper::{Request, Response};
use hyper_util::rt::{TokioExecutor, TokioIo};
use rustls::pki_types::{CertificateDer, PrivateKeyDer, pem::PemObject};
use serde_json::{Map, Value, json};
use tokio::net::TcpListener;
use tokio_rustls::TlsAcceptor;

#[derive(Default)]
struct ServerCounters {
    accepted: AtomicUsize,
    active: AtomicUsize,
    max_active: AtomicUsize,
    completed: AtomicUsize,
}

impl ServerCounters {
    fn enter(&self) {
        let active = self.active.fetch_add(1, Ordering::SeqCst) + 1;
        self.max_active.fetch_max(active, Ordering::SeqCst);
    }

    fn leave(&self) {
        self.active.fetch_sub(1, Ordering::SeqCst);
        self.completed.fetch_add(1, Ordering::SeqCst);
    }
}

#[derive(Clone, Copy)]
enum ServerProtocol {
    H1,
    H2,
}

struct LoopbackServer {
    base_url: String,
    counters: Arc<ServerCounters>,
    task: tokio::task::JoinHandle<()>,
}

impl LoopbackServer {
    async fn spawn(protocol: ServerProtocol, response_delay: Duration) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let counters = Arc::new(ServerCounters::default());
        let accept_counters = counters.clone();
        let task = tokio::spawn(async move {
            loop {
                let Ok((stream, _)) = listener.accept().await else {
                    break;
                };
                accept_counters.accepted.fetch_add(1, Ordering::SeqCst);
                let request_counters = accept_counters.clone();
                let service = service_fn(move |_request: Request<hyper::body::Incoming>| {
                    let counters = request_counters.clone();
                    async move {
                        counters.enter();
                        tokio::time::sleep(response_delay).await;
                        counters.leave();
                        Ok::<_, Infallible>(chat_response())
                    }
                });
                tokio::spawn(async move {
                    let io = TokioIo::new(stream);
                    match protocol {
                        ServerProtocol::H1 => {
                            let _ = hyper::server::conn::http1::Builder::new()
                                .serve_connection(io, service)
                                .await;
                        }
                        ServerProtocol::H2 => {
                            let _ = hyper::server::conn::http2::Builder::new(TokioExecutor::new())
                                .serve_connection(io, service)
                                .await;
                        }
                    }
                });
            }
        });
        Self {
            base_url: format!("http://{address}"),
            counters,
            task,
        }
    }
}

impl Drop for LoopbackServer {
    fn drop(&mut self) {
        self.task.abort();
    }
}

fn chat_response() -> Response<Full<Bytes>> {
    let body = concat!(
        "data: {\"choices\":[{\"delta\":{\"content\":\"answer\"}}]}\n\n",
        "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":1,\"completion_tokens\":1}}\n\n",
        "data: [DONE]\n\n",
    );
    Response::builder()
        .header("content-type", "text/event-stream")
        .body(Full::new(Bytes::from_static(body.as_bytes())))
        .unwrap()
}

fn run_child(request: Value) -> Output {
    let mut child = Command::new(env!("CARGO_BIN_EXE_aiperf"))
        .arg("--execute")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(&serde_json::to_vec(&request["run"]).unwrap())
        .unwrap();
    child.wait_with_output().unwrap()
}

fn scheduled_request(
    artifact_dir: &std::path::Path,
    endpoint_url: &str,
    requests: usize,
    concurrency: usize,
    policy: Map<String, Value>,
) -> Value {
    let mut profile = Map::from_iter([
        ("type".into(), json!("chat")),
        ("urls".into(), json!([endpoint_url])),
        ("streaming".into(), json!(true)),
        ("use_server_token_count".into(), json!(true)),
        ("wait_for_model_timeout".into(), json!(0.0)),
    ]);
    profile.extend(policy);
    json!({
        "protocol_version": 2,
        "operation": "execute",
        "run": {
            "benchmark_id": "http-policy-v2",
            "artifact_dir": artifact_dir,
            "cfg": {
                "models": {"items": [{"name": "fixture-model"}]},
                "endpoint": profile,
                "datasets": [{
                    "type": "synthetic",
                    "entries": requests,
                    "sampling": "sequential",
                    "prompts": {"isl": {"value": 4.0}, "osl": {"value": 1.0}}
                }],
                "tokenizer": {
                    "name": "cl100k_base",
                    "revision": "main",
                    "trust_remote_code": false,
                    "apply_chat_template": false
                },
                "phases": [{
                    "type": "concurrency",
                    "name": "profiling",
                    "exclude_from_results": false,
                    "requests": requests,
                    "concurrency": concurrency
                }],
                "transport": {"type": "http"},
                "runtime": {"workers": 1}
            }
        }
    })
}

fn assert_success(output: &Output) {
    assert!(
        output.status.success(),
        "stdout={}\nstderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let terminal: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(terminal["success"], true, "{terminal}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn ordinary_runner_multiplexes_scheduled_h2_on_one_connection() {
    let server = LoopbackServer::spawn(ServerProtocol::H2, Duration::from_millis(25)).await;
    let artifacts = tempfile::tempdir().unwrap();
    let request = scheduled_request(
        &artifacts.path().join("h2"),
        &server.base_url,
        16,
        16,
        Map::from_iter([
            ("http2".into(), json!(true)),
            ("connection_limit".into(), json!(1)),
        ]),
    );

    let output = tokio::task::spawn_blocking(move || run_child(request))
        .await
        .unwrap();
    assert_success(&output);
    assert_eq!(server.counters.accepted.load(Ordering::SeqCst), 1);
    assert!(server.counters.max_active.load(Ordering::SeqCst) > 1);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn ordinary_runner_honors_h1_capacity_and_zero_keepalive() {
    let bounded = LoopbackServer::spawn(ServerProtocol::H1, Duration::from_millis(25)).await;
    let artifacts = tempfile::tempdir().unwrap();
    let request = scheduled_request(
        &artifacts.path().join("bounded"),
        &bounded.base_url,
        12,
        12,
        Map::from_iter([("connection_limit".into(), json!(2))]),
    );
    let output = tokio::task::spawn_blocking(move || run_child(request))
        .await
        .unwrap();
    assert_success(&output);
    assert_eq!(bounded.counters.accepted.load(Ordering::SeqCst), 2);
    assert_eq!(bounded.counters.max_active.load(Ordering::SeqCst), 2);

    let no_idle_reuse = LoopbackServer::spawn(ServerProtocol::H1, Duration::ZERO).await;
    let request = scheduled_request(
        &artifacts.path().join("zero-keepalive"),
        &no_idle_reuse.base_url,
        3,
        1,
        Map::from_iter([("keepalive_timeout".into(), json!(0.0))]),
    );
    let output = tokio::task::spawn_blocking(move || run_child(request))
        .await
        .unwrap();
    assert_success(&output);
    assert_eq!(no_idle_reuse.counters.accepted.load(Ordering::SeqCst), 3);
}

// Public tokio-rustls test certificate. It is intentionally outside the web
// PKI trust store and does not match the loopback IP.
const CHAIN: &str = r#"-----BEGIN CERTIFICATE-----
MIIBszCCAVmgAwIBAgIUUg3keFcU1xXWK8BNVb1KynPulV8wCgYIKoZIzj0EAwIw
JjEkMCIGA1UEAwwbUnVzdGxzIFJvYnVzdCBSb290IC0gUnVuZyAyMCAXDTc1MDEw
MTAwMDAwMFoYDzQwOTYwMTAxMDAwMDAwWjAhMR8wHQYDVQQDDBZyY2dlbiBzZWxm
IHNpZ25lZCBjZXJ0MFkwEwYHKoZIzj0CAQYIKoZIzj0DAQcDQgAEud6w4gtZ0xbw
J3E69SSMy5TZfdIifl9L5ZY+hgEe4UiUsBWS32f6Y5NR5Jo8FO1f6o13b3+FvVHR
EHCGdvppL6NoMGYwFQYDVR0RBA4wDIIKZm9vYmFyLmNvbTAdBgNVHSUEFjAUBggr
BgEFBQcDAQYIKwYBBQUHAwIwHQYDVR0OBBYEFELvxbj5tD75n4pYFvJyr+c8qVEi
MA8GA1UdEwEB/wQFMAMBAQAwCgYIKoZIzj0EAwIDSAAwRQIhALxSSdUsrRFnwNMu
/doBqI8i8u5HdohVAheFTDwObkOMAiASSjULUtkWSD15u/7Sr01Wm9J1MpqW1pob
BVqU3CNRlA==
-----END CERTIFICATE-----
-----BEGIN CERTIFICATE-----
MIIBiTCCATCgAwIBAgIUHWiVYIvMMWoZEFYvSz46COf2FqowCgYIKoZIzj0EAwIw
HTEbMBkGA1UEAwwSUnVzdGxzIFJvYnVzdCBSb290MCAXDTc1MDEwMTAwMDAwMFoY
DzQwOTYwMTAxMDAwMDAwWjAmMSQwIgYDVQQDDBtSdXN0bHMgUm9idXN0IFJvb3Qg
LSBSdW5nIDIwWTATBgcqhkjOPQIBBggqhkjOPQMBBwNCAATAOCcBD7dXjmAZ3te5
D47cCJ9ec93PWv7BKYIL826CJsKfXQOGrBTthLm77hXLhHu6uv8E5QXNLZpfowLQ
Do1ao0MwQTAPBgNVHQ8BAf8EBQMDB4QAMB0GA1UdDgQWBBRdza76r11Ok9vRmlg6
Nn/wL/N+jTAPBgNVHRMBAf8EBTADAQH/MAoGCCqGSM49BAMCA0cAMEQCIFmZrXeK
hnfkahocvkhhNT3cDv1LWf6WBoFaCiBwZXFPAiARaKRiSCMG7PCHmSqFe82TBVmL
odHGogAVax1Dh/aYAA==
-----END CERTIFICATE-----
"#;

const KEY: &str = r#"-----BEGIN PRIVATE KEY-----
MIGHAgEAMBMGByqGSM49AgEGCCqGSM49AwEHBG0wawIBAQQgTbAQpfjAT46fgF4B
mP15n37woNG5ZNJmwcqsred/7tmhRANCAAS53rDiC1nTFvAncTr1JIzLlNl90iJ+
X0vllj6GAR7hSJSwFZLfZ/pjk1HkmjwU7V/qjXdvf4W9UdEQcIZ2+mkv
-----END PRIVATE KEY-----
"#;

async fn spawn_untrusted_https() -> LoopbackServer {
    let certificates = CertificateDer::pem_slice_iter(CHAIN.as_bytes())
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    let key = PrivateKeyDer::from_pem_slice(KEY.as_bytes()).unwrap();
    let provider = Arc::new(rustls::crypto::aws_lc_rs::default_provider());
    let mut config = rustls::ServerConfig::builder_with_provider(provider)
        .with_safe_default_protocol_versions()
        .unwrap()
        .with_no_client_auth()
        .with_single_cert(certificates, key)
        .unwrap();
    config.alpn_protocols = vec![b"http/1.1".to_vec()];
    let acceptor = TlsAcceptor::from(Arc::new(config));
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let counters = Arc::new(ServerCounters::default());
    let accept_counters = counters.clone();
    let task = tokio::spawn(async move {
        loop {
            let Ok((stream, _)) = listener.accept().await else {
                break;
            };
            accept_counters.accepted.fetch_add(1, Ordering::SeqCst);
            let acceptor = acceptor.clone();
            let request_counters = accept_counters.clone();
            tokio::spawn(async move {
                let Ok(stream) = acceptor.accept(stream).await else {
                    return;
                };
                let service = service_fn(move |request: Request<hyper::body::Incoming>| {
                    let counters = request_counters.clone();
                    async move {
                        counters.enter();
                        let response = if request.method() == hyper::Method::GET {
                            Response::builder()
                                .header("content-type", "application/json")
                                .body(Full::new(Bytes::from_static(
                                    b"{\"data\":[{\"id\":\"fixture-model\"}]}",
                                )))
                                .unwrap()
                        } else {
                            Response::builder()
                                .header("content-type", "application/json")
                                .body(Full::new(Bytes::from_static(
                                    b"{\"predictions\":[{\"output\":\"answer\"}]}",
                                )))
                                .unwrap()
                        };
                        counters.leave();
                        Ok::<_, Infallible>(response)
                    }
                });
                let _ = hyper::server::conn::http1::Builder::new()
                    .serve_connection(TokioIo::new(stream), service)
                    .await;
            });
        }
    });
    LoopbackServer {
        base_url: format!("https://{address}"),
        counters,
        task,
    }
}

fn tls_request(artifact_dir: &std::path::Path, endpoint_url: &str, ssl_verify: bool) -> Value {
    json!({
        "protocol_version": 2,
        "operation": "execute",
        "run": {
            "benchmark_id": "http-tls-policy-v2",
            "artifact_dir": artifact_dir,
            "cfg": {
                "models": {"items": [{"name": "fixture-model"}]},
                "endpoint": {
                    "type": "kserve_v1_predict",
                    "urls": [endpoint_url],
                    "streaming": false,
                    "ssl_verify": ssl_verify,
                    // Generous readiness budget so this process-spawning test does
                    // not flake under parallel CPU load: the TLS handshake +
                    // readiness probe can exceed a 100ms budget when many test
                    // children contend for cores. The asserted behavior is
                    // unchanged — verified TLS still fails with UnknownIssuer and
                    // the insecure request still reaches the server.
                    "wait_for_model_timeout": 5.0,
                    "wait_for_model_interval": 0.05,
                    "wait_for_model_mode": "models"
                },
                "datasets": [{
                    "type": "synthetic",
                    "entries": 1,
                    "sampling": "sequential",
                    "prompts": {"isl": {"value": 4.0}, "osl": {"value": 1.0}}
                }],
                "tokenizer": {
                    "name": "cl100k_base",
                    "revision": "main",
                    "trust_remote_code": false,
                    "apply_chat_template": false
                },
                "phases": [{
                    "type": "concurrency",
                    "name": "profiling",
                    "exclude_from_results": false,
                    "requests": 1,
                    "concurrency": 1
                }],
                "transport": {"type": "http"},
                "runtime": {"workers": 1}
            }
        }
    })
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn ordinary_runner_requires_explicit_insecure_tls_policy() {
    let server = spawn_untrusted_https().await;
    let artifacts = tempfile::tempdir().unwrap();
    let verified = tls_request(&artifacts.path().join("verified"), &server.base_url, true);
    let output = tokio::task::spawn_blocking(move || run_child(verified))
        .await
        .unwrap();
    assert!(!output.status.success(), "verified TLS unexpectedly ran");
    let terminal: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(terminal["stage"], "execution", "{terminal}");
    assert!(
        terminal["errors"][0]["message"]
            .as_str()
            .is_some_and(|message| message.contains("waiting for endpoint readiness")
                && message.contains("UnknownIssuer")),
        "{terminal}"
    );
    assert_eq!(server.counters.completed.load(Ordering::SeqCst), 0);

    let insecure = tls_request(
        &artifacts.path().join("explicit-insecure"),
        &server.base_url,
        false,
    );
    let output = tokio::task::spawn_blocking(move || run_child(insecure))
        .await
        .unwrap();
    assert_success(&output);
    assert!(
        server.counters.completed.load(Ordering::SeqCst) >= 2,
        "readiness and scheduled inference must both reach the TLS server"
    );
}
