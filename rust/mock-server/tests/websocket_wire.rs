// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Raw-frame integration coverage for mock WebSocket listeners.

use std::net::SocketAddr;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::task::{Context, Poll};
use std::time::Duration;

use aiperf_mock_server::config::{
    MockServerConfig, WebSocketControl, WebSocketMode, WebSocketScenario,
};
use aiperf_mock_server::{AppState, build_router, listener, tls};
use serde_json::Value;
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt, ReadBuf};
use tokio::net::{TcpListener, TcpStream, UnixStream};

const TEST_WEBSOCKET_KEY: &str = "dGhlIHNhbXBsZSBub25jZQ==";

trait TestIo: AsyncRead + AsyncWrite + Send + Unpin {}

impl<T> TestIo for T where T: AsyncRead + AsyncWrite + Send + Unpin {}

struct BoxIo(Box<dyn TestIo>);

impl AsyncRead for BoxIo {
    fn poll_read(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut ReadBuf<'_>,
    ) -> Poll<std::io::Result<()>> {
        Pin::new(&mut *self.0).poll_read(cx, buf)
    }
}

impl AsyncWrite for BoxIo {
    fn poll_write(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<std::io::Result<usize>> {
        Pin::new(&mut *self.0).poll_write(cx, buf)
    }

    fn poll_flush(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<std::io::Result<()>> {
        Pin::new(&mut *self.0).poll_flush(cx)
    }

    fn poll_shutdown(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<std::io::Result<()>> {
        Pin::new(&mut *self.0).poll_shutdown(cx)
    }
}

struct TestServer {
    address: SocketAddr,
    state: Arc<AppState>,
    task: tokio::task::JoinHandle<()>,
}

impl TestServer {
    async fn start(config: MockServerConfig, is_tls: bool) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind WebSocket test listener");
        let address = listener.local_addr().expect("WebSocket test address");
        let state = AppState::build(config);
        let router = build_router(state.clone());
        let acceptor = if is_tls {
            Some(tls::self_signed_acceptor().expect("self-signed WebSocket acceptor"))
        } else {
            None
        };
        let task = tokio::spawn(async move {
            let _ = tls::serve_http(listener, router, acceptor, 0).await;
        });
        Self {
            address,
            state,
            task,
        }
    }

    async fn connect(&self, is_tls: bool, path: &str) -> RawWebSocketClient {
        let tcp = TcpStream::connect(self.address)
            .await
            .expect("connect WebSocket test listener");
        let io = if is_tls {
            BoxIo(Box::new(connect_insecure_tls(tcp).await))
        } else {
            BoxIo(Box::new(tcp))
        };
        RawWebSocketClient::upgrade(io, path).await
    }

    async fn wait_for_captures(&self, count: usize) -> Vec<Value> {
        for _ in 0..100 {
            let request = http::Request::builder()
                .uri("/mock/websocket/captures")
                .body(axum::body::Body::empty())
                .expect("capture request");
            use tower::ServiceExt;
            let response = build_router(self.state.clone())
                .oneshot(request)
                .await
                .expect("capture response");
            use http_body_util::BodyExt;
            let body = response
                .into_body()
                .collect()
                .await
                .expect("collect capture response")
                .to_bytes();
            let captures: Vec<Value> =
                serde_json::from_slice(&body).expect("capture response is JSON");
            if captures.len() >= count {
                return captures;
            }
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
        panic!("mock did not publish {count} WebSocket captures");
    }
}

impl Drop for TestServer {
    fn drop(&mut self) {
        self.task.abort();
    }
}

#[derive(Debug)]
struct WireFrame {
    is_final: bool,
    opcode: u8,
    payload: Vec<u8>,
}

struct RawWebSocketClient {
    io: BoxIo,
}

impl RawWebSocketClient {
    async fn upgrade(mut io: BoxIo, path: &str) -> Self {
        let request = format!(
            "GET {path} HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: keep-alive, Upgrade\r\nUpgrade: websocket\r\nSec-WebSocket-Version: 13\r\nSec-WebSocket-Key: {TEST_WEBSOCKET_KEY}\r\n\r\n"
        );
        io.write_all(request.as_bytes())
            .await
            .expect("write WebSocket upgrade");
        io.flush().await.expect("flush WebSocket upgrade");
        let headers = read_http_headers(&mut io).await;
        assert!(
            headers.starts_with("HTTP/1.1 101 "),
            "upgrade must switch protocols: {headers:?}"
        );
        assert!(
            headers
                .to_ascii_lowercase()
                .contains("sec-websocket-accept:"),
            "upgrade must sign the client key: {headers:?}"
        );
        Self { io }
    }

    async fn send_text(&mut self, text: &str) {
        self.send_data_frame(true, 0x1, text.as_bytes()).await;
    }

    async fn send_data_frame(&mut self, is_final: bool, opcode: u8, payload: &[u8]) {
        let mask = [0x13, 0x37, 0x42, 0x99];
        let mut frame = vec![(u8::from(is_final) << 7) | opcode];
        match payload.len() {
            len @ 0..=125 => frame.push(0x80 | len as u8),
            len @ 126..=65_535 => {
                frame.push(0x80 | 126);
                frame.extend_from_slice(&(len as u16).to_be_bytes());
            }
            len => {
                frame.push(0x80 | 127);
                frame.extend_from_slice(&(len as u64).to_be_bytes());
            }
        }
        frame.extend_from_slice(&mask);
        frame.extend(
            payload
                .iter()
                .enumerate()
                .map(|(index, byte)| byte ^ mask[index % mask.len()]),
        );
        self.io
            .write_all(&frame)
            .await
            .expect("write masked client text frame");
        self.io.flush().await.expect("flush client text frame");
    }

    async fn read_frame(&mut self) -> WireFrame {
        tokio::time::timeout(Duration::from_secs(5), read_server_frame(&mut self.io))
            .await
            .expect("server frame timeout")
            .expect("read server frame")
    }
}

async fn read_http_headers(io: &mut (impl AsyncRead + Unpin)) -> String {
    let mut bytes = Vec::new();
    loop {
        bytes.push(io.read_u8().await.expect("read WebSocket upgrade response"));
        if bytes.ends_with(b"\r\n\r\n") {
            break;
        }
        assert!(bytes.len() < 16_384, "upgrade headers must be bounded");
    }
    String::from_utf8(bytes).expect("upgrade response headers are UTF-8")
}

async fn read_server_frame(io: &mut (impl AsyncRead + Unpin)) -> std::io::Result<WireFrame> {
    let first = io.read_u8().await?;
    let second = io.read_u8().await?;
    let is_final = first & 0x80 != 0;
    let opcode = first & 0x0f;
    assert_eq!(second & 0x80, 0, "server frames must not be masked");
    let payload_len = match second & 0x7f {
        len @ 0..=125 => u64::from(len),
        126 => u64::from(io.read_u16().await?),
        127 => io.read_u64().await?,
        _ => unreachable!(),
    };
    let payload_len = usize::try_from(payload_len).expect("test frame length fits usize");
    let mut payload = vec![0; payload_len];
    io.read_exact(&mut payload).await?;
    Ok(WireFrame {
        is_final,
        opcode,
        payload,
    })
}

async fn assert_fragmented_turn(mut client: RawWebSocketClient) {
    client
        .send_text(
            r#"{"type":"response.create","model":"m😀","input":[{"role":"user","content":"hello"}]}"#,
        )
        .await;

    let mut messages = Vec::new();
    let mut message = Vec::new();
    let mut is_fragment_in_progress = false;
    let mut has_nonfinal_data = false;
    let mut has_interjected_ping = false;
    loop {
        let frame = client.read_frame().await;
        match frame.opcode {
            0x1 => {
                assert!(!is_fragment_in_progress, "text starts only one message");
                assert!(
                    frame.payload.len() <= 5,
                    "text frame exceeds fragment bound"
                );
                assert!(
                    std::str::from_utf8(&frame.payload).is_ok(),
                    "fragment must not split a UTF-8 scalar"
                );
                message.extend_from_slice(&frame.payload);
                is_fragment_in_progress = !frame.is_final;
                has_nonfinal_data |= !frame.is_final;
            }
            0x0 => {
                assert!(
                    is_fragment_in_progress,
                    "continuation requires open message"
                );
                assert!(
                    frame.payload.len() <= 5,
                    "continuation exceeds fragment bound"
                );
                assert!(
                    std::str::from_utf8(&frame.payload).is_ok(),
                    "continuation must not split a UTF-8 scalar"
                );
                message.extend_from_slice(&frame.payload);
                is_fragment_in_progress = !frame.is_final;
            }
            0x9 => {
                has_interjected_ping |= is_fragment_in_progress;
                assert_eq!(frame.payload, b"mock");
            }
            0xa => {}
            opcode => panic!("unexpected server opcode {opcode:#x}"),
        }
        if !is_fragment_in_progress && !message.is_empty() {
            let event: Value = serde_json::from_slice(&message).expect("reassembled event is JSON");
            let is_terminal = event["type"] == "response.completed";
            messages.push(event);
            message.clear();
            if is_terminal {
                break;
            }
        }
    }

    assert!(
        has_nonfinal_data,
        "positive fragment size must author fragments"
    );
    assert!(
        has_interjected_ping,
        "configured Ping must be interjected between data fragments"
    );
    assert!(
        messages
            .iter()
            .any(|event| event["type"] == "response.output_text.delta"),
        "reassembled content event must be preserved"
    );
    assert_eq!(
        messages
            .iter()
            .find(|event| event["type"] == "response.completed")
            .expect("terminal response")["response"]["model"],
        "m😀"
    );
}

fn fragmented_config() -> MockServerConfig {
    MockServerConfig {
        websocket_mode: WebSocketMode::TurnSerialized,
        websocket_scenario: WebSocketScenario::Normal,
        websocket_fragment_bytes: 5,
        websocket_control_before_content: WebSocketControl::Ping,
        websocket_first_content_delay_ms: 0.0,
        websocket_content_interval_ms: 0.0,
        no_tokenizer: true,
        ..MockServerConfig::default()
    }
}

#[tokio::test]
async fn ws_authors_utf8_safe_fragments_with_interjected_control() {
    let server = TestServer::start(fragmented_config(), false).await;
    assert_fragmented_turn(server.connect(false, "/mock/websocket/turns").await).await;
}

#[tokio::test]
async fn wss_authors_the_same_fragments_after_tls_upgrade() {
    let server = TestServer::start(fragmented_config(), true).await;
    assert_fragmented_turn(server.connect(true, "/mock/websocket/turns").await).await;
}

#[tokio::test]
async fn raw_upgrade_enforces_frame_and_reassembled_message_limits() {
    let server = TestServer::start(
        MockServerConfig {
            websocket_mode: WebSocketMode::TurnSerialized,
            websocket_fragment_bytes: 4,
            websocket_max_message_bytes: 64,
            websocket_first_content_delay_ms: 0.0,
            websocket_content_interval_ms: 0.0,
            no_tokenizer: true,
            ..MockServerConfig::default()
        },
        false,
    )
    .await;

    let mut oversized_frame = server.connect(false, "/mock/websocket/turns").await;
    oversized_frame
        .send_data_frame(true, 0x1, &[b'x'; 65])
        .await;
    drop(oversized_frame);

    let mut oversized_message = server.connect(false, "/mock/websocket/turns").await;
    oversized_message
        .send_data_frame(false, 0x1, &[b'x'; 40])
        .await;
    oversized_message
        .send_data_frame(true, 0x0, &[b'y'; 40])
        .await;
    drop(oversized_message);

    let captures = server.wait_for_captures(2).await;
    assert_eq!(
        captures
            .iter()
            .map(|capture| capture["close"].as_str())
            .collect::<Vec<_>>(),
        vec![Some("receive_error"), Some("receive_error")]
    );
}

#[tokio::test]
async fn raw_upgrade_rejects_a_handshake_without_a_client_key() {
    let server = TestServer::start(fragmented_config(), false).await;
    let mut stream = TcpStream::connect(server.address)
        .await
        .expect("connect raw-upgrade listener");
    stream
        .write_all(
            b"GET /mock/websocket/turns HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: Upgrade\r\nUpgrade: websocket\r\nSec-WebSocket-Version: 13\r\n\r\n",
        )
        .await
        .expect("write invalid upgrade");
    let headers = read_http_headers(&mut stream).await;
    assert!(
        headers.starts_with("HTTP/1.1 400 "),
        "missing key must fail before upgrade: {headers:?}"
    );
}

#[cfg(unix)]
#[tokio::test]
async fn uds_serves_the_raw_fragmentation_route() {
    let path = temp_socket_path("websocket");
    let config = fragmented_config();
    let router = build_router(AppState::build(config));
    let path_str = path.to_string_lossy().into_owned();
    let server_path = path_str.clone();
    let task = tokio::spawn(async move {
        let _ = listener::serve_router_uds(router, &server_path).await;
    });
    for _ in 0..100 {
        if path.exists() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(1)).await;
    }
    let stream = UnixStream::connect(&path_str)
        .await
        .expect("connect WebSocket UDS listener");
    let client =
        RawWebSocketClient::upgrade(BoxIo(Box::new(stream)), "/mock/websocket/turns").await;
    assert_fragmented_turn(client).await;
    task.abort();
    let _ = std::fs::remove_file(path);
}

#[cfg(unix)]
fn temp_socket_path(tag: &str) -> PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let count = COUNTER.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "aiperf-mock-{tag}-{}-{count}.sock",
        std::process::id()
    ))
}

async fn connect_insecure_tls(tcp: TcpStream) -> tokio_rustls::client::TlsStream<TcpStream> {
    let provider = Arc::new(rustls::crypto::aws_lc_rs::default_provider());
    let mut config = rustls::ClientConfig::builder_with_provider(provider.clone())
        .with_safe_default_protocol_versions()
        .expect("safe TLS versions")
        .dangerous()
        .with_custom_certificate_verifier(Arc::new(NoVerify { provider }))
        .with_no_client_auth();
    config.alpn_protocols = vec![b"http/1.1".to_vec()];
    let connector = tokio_rustls::TlsConnector::from(Arc::new(config));
    let server_name = rustls::pki_types::ServerName::try_from("127.0.0.1")
        .expect("valid test server name")
        .to_owned();
    connector
        .connect(server_name, tcp)
        .await
        .expect("connect self-signed WSS listener")
}

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
