// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Crate-level TLS proofs for the mock server's HTTPS and `grpcs` listeners.
//!
//! These exercise the exact server-side code `--tls-cert`/`--tls-key`/
//! `--tls-self-signed` stand up ([`aiperf_mock_server::tls`] +
//! [`aiperf_mock_server::grpc::serve_grpc_with_tls`]) with real TLS clients:
//!
//!   * `https_serves_streamed_chat_over_tls` — a native-TLS reqwest client with
//!     verification disabled (`danger_accept_invalid_certs`, the client-side
//!     analogue of the runner's `ssl_verify=false`) POSTs a streaming chat
//!     completion over HTTPS and reads back SSE content. This proves the HTTPS
//!     path the `aiperf profile` e2e (`rust/e2e/tests/test_tls.rs`) also drives.
//!
//!   * `grpcs_listener_negotiates_h2_alpn` — a raw rustls client with a
//!     no-verify verifier completes the TLS handshake against the `grpcs`
//!     listener and asserts the negotiated ALPN is `h2`. This is the ONLY TLS
//!     coverage the `grpcs` listener can get from a fresh self-signed cert: the
//!     runner's tonic `grpcs` client verifies against the system trust roots
//!     with no accept-invalid toggle
//!     (`rust/aiperf/src/transport::grpc/transport.rs:783-793`), so a self-signed
//!     mock is unreachable through `aiperf profile` `grpcs://` by design — hence
//!     this direct-handshake proof of the listener's TLS termination + h2 ALPN.

use std::net::SocketAddr;
use std::sync::Arc;

use aiperf_mock_server::config::MockServerConfig;
use aiperf_mock_server::grpc::serve_grpc_with_tls;
use aiperf_mock_server::{AppState, build_router, tls};
use tokio::net::{TcpListener, TcpStream};

/// A fast, tokenizer-free config for TLS coverage — latency and tokenizer are
/// irrelevant to the handshake/transport assertions here.
fn fast_state() -> Arc<AppState> {
    let cfg = MockServerConfig {
        fast: true,
        no_tokenizer: true,
        ..MockServerConfig::default()
    }
    .apply_flags();
    AppState::build(cfg)
}

/// Bind an ephemeral loopback port and return the listener + its address.
async fn bind_ephemeral() -> (TcpListener, SocketAddr) {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind ephemeral");
    let addr = listener.local_addr().expect("listener addr");
    (listener, addr)
}

/// A native-TLS reqwest client (HTTP/1+2) that accepts any certificate — the
/// client analogue of the runner's `ssl_verify=false`.
fn insecure_https_client() -> reqwest::Client {
    reqwest::Client::builder()
        .danger_accept_invalid_certs(true)
        .build()
        .expect("build insecure https client")
}

/// HTTPS: a streaming chat completion over the self-signed TLS frontend returns
/// 200 and streamed SSE content.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn https_serves_streamed_chat_over_tls() {
    let (listener, addr) = bind_ephemeral().await;
    let router = build_router(fast_state());
    let acceptor = tls::self_signed_acceptor().expect("self-signed acceptor");
    tokio::spawn(async move {
        let _ = tls::serve_http(listener, router, Some(acceptor), 0).await;
    });

    let client = insecure_https_client();
    let url = format!("https://127.0.0.1:{}/v1/chat/completions", addr.port());
    let body = serde_json::json!({
        "model": "gpt-4",
        "stream": true,
        "max_tokens": 4,
        "messages": [{"role": "user", "content": "hello over tls"}],
    });

    let resp = client
        .post(&url)
        .json(&body)
        .send()
        .await
        .expect("HTTPS request should succeed against the self-signed listener");
    assert_eq!(resp.status().as_u16(), 200, "HTTPS status must be 200");

    let text = resp.text().await.expect("read streamed body");
    assert!(
        text.contains("chat.completion.chunk"),
        "streamed HTTPS body should carry SSE chat.completion.chunk frames, got:\n{text}"
    );
    assert!(
        text.contains("[DONE]"),
        "streamed HTTPS body should terminate with [DONE], got:\n{text}"
    );
}

/// HTTPS: a non-streaming health check confirms the plain GET path is also
/// served over TLS (not just the streaming POST).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn https_serves_health_over_tls() {
    let (listener, addr) = bind_ephemeral().await;
    let router = build_router(fast_state());
    let acceptor = tls::self_signed_acceptor().expect("self-signed acceptor");
    tokio::spawn(async move {
        let _ = tls::serve_http(listener, router, Some(acceptor), 0).await;
    });

    let client = insecure_https_client();
    let url = format!("https://127.0.0.1:{}/health", addr.port());
    let resp = client.get(&url).send().await.expect("HTTPS health request");
    assert_eq!(resp.status().as_u16(), 200, "HTTPS /health must be 200");
}

/// `grpcs`: the TLS-wrapped gRPC listener completes a handshake and negotiates
/// the `h2` ALPN protocol tonic clients require. See the module docs for why
/// this is a raw-handshake proof rather than an `aiperf profile` run.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn grpcs_listener_negotiates_h2_alpn() {
    let (listener, addr) = bind_ephemeral().await;
    // `serve_grpc_with_tls` binds its own listener from the addr; drop ours so
    // the port is free for it to re-bind (SO_REUSEADDR/REUSEPORT).
    drop(listener);
    let acceptor = tls::self_signed_acceptor().expect("self-signed acceptor");
    tokio::spawn(async move {
        let _ = serve_grpc_with_tls(addr, fast_state(), Some(acceptor)).await;
    });

    // Wait for the grpcs listener to accept.
    for _ in 0..50 {
        if TcpStream::connect(addr).await.is_ok() {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }

    let alpn = tls_handshake_alpn(addr).await;
    assert_eq!(
        alpn.as_deref(),
        Some(&b"h2"[..]),
        "grpcs listener must negotiate the h2 ALPN protocol, got {alpn:?}"
    );
}

/// Complete a rustls client handshake (verification disabled, requesting ALPN
/// `h2` then `http/1.1`) against `addr` and return the negotiated ALPN protocol.
async fn tls_handshake_alpn(addr: SocketAddr) -> Option<Vec<u8>> {
    let provider = Arc::new(rustls::crypto::aws_lc_rs::default_provider());
    let mut config = rustls::ClientConfig::builder_with_provider(provider.clone())
        .with_safe_default_protocol_versions()
        .expect("safe default protocol versions")
        .dangerous()
        .with_custom_certificate_verifier(Arc::new(NoVerify { provider }))
        .with_no_client_auth();
    config.alpn_protocols = vec![b"h2".to_vec(), b"http/1.1".to_vec()];

    let connector = tokio_rustls::TlsConnector::from(Arc::new(config));
    let server_name = rustls::pki_types::ServerName::try_from("127.0.0.1")
        .expect("valid server name")
        .to_owned();
    let tcp = TcpStream::connect(addr)
        .await
        .expect("tcp connect to grpcs");
    let tls = connector
        .connect(server_name, tcp)
        .await
        .expect("TLS handshake against grpcs listener should succeed");
    let (_, conn) = tls.get_ref();
    conn.alpn_protocol().map(|p| p.to_vec())
}

/// A `ServerCertVerifier` that accepts any certificate, mirroring the runner's
/// `ssl_verify=false` (`aiperf_runtime::transport::http::client::connection`) — used only
/// to reach the ALPN result of the handshake against a self-signed listener.
#[derive(Debug)]
struct NoVerify {
    provider: Arc<rustls::crypto::CryptoProvider>,
}

impl rustls::client::danger::ServerCertVerifier for NoVerify {
    fn verify_server_cert(
        &self,
        _end_entity: &rustls::pki_types::CertificateDer<'_>,
        _intermediates: &[rustls::pki_types::CertificateDer<'_>],
        _server_name: &rustls::pki_types::ServerName<'_>,
        _ocsp_response: &[u8],
        _now: rustls::pki_types::UnixTime,
    ) -> Result<rustls::client::danger::ServerCertVerified, rustls::Error> {
        Ok(rustls::client::danger::ServerCertVerified::assertion())
    }

    fn verify_tls12_signature(
        &self,
        message: &[u8],
        cert: &rustls::pki_types::CertificateDer<'_>,
        signed: &rustls::DigitallySignedStruct,
    ) -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        rustls::crypto::verify_tls12_signature(
            message,
            cert,
            signed,
            &self.provider.signature_verification_algorithms,
        )
    }

    fn verify_tls13_signature(
        &self,
        message: &[u8],
        cert: &rustls::pki_types::CertificateDer<'_>,
        signed: &rustls::DigitallySignedStruct,
    ) -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        rustls::crypto::verify_tls13_signature(
            message,
            cert,
            signed,
            &self.provider.signature_verification_algorithms,
        )
    }

    fn supported_verify_schemes(&self) -> Vec<rustls::SignatureScheme> {
        self.provider
            .signature_verification_algorithms
            .supported_schemes()
    }
}
