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
use tokio::sync::watch;

const TEST_WEBSOCKET_KEY: &str = "dGhlIHNhbXBsZSBub25jZQ==";
const TEST_WEBSOCKET_ACCEPT: &str = "s3pPLMBiTxaQ9kYGzzhZRbK+xOo=";
const MAX_TEST_FRAME_BYTES: usize = 8 * 1024 * 1024;

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
        RawWebSocketClient::connect(self.address, is_tls, path).await
    }

    async fn raw_request(&self, request: &[u8]) -> String {
        let mut stream = TcpStream::connect(self.address)
            .await
            .expect("connect raw-upgrade listener");
        stream
            .write_all(request)
            .await
            .expect("write raw upgrade request");
        stream.flush().await.expect("flush raw upgrade request");
        read_http_headers(&mut stream).await
    }

    async fn wait_for_captures(&self, count: usize) -> Vec<Value> {
        self.wait_for_captures_for(count, Duration::from_millis(250))
            .await
    }

    async fn wait_for_captures_for(&self, count: usize, timeout: Duration) -> Vec<Value> {
        let deadline = tokio::time::Instant::now() + timeout;
        loop {
            let captures = self.captures().await;
            if captures.len() >= count {
                return captures;
            }
            assert!(
                tokio::time::Instant::now() < deadline,
                "mock did not publish {count} WebSocket captures within {timeout:?}"
            );
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
    }

    async fn captures(&self) -> Vec<Value> {
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
        serde_json::from_slice(&body).expect("capture response is JSON")
    }

    fn request_total(&self, status: &str) -> u64 {
        self.state
            .recorder
            .metrics
            .aiperf
            .REQUESTS_TOTAL
            .with_label_values(&["mock_websocket_turns", "POST", status])
            .get()
    }
}

struct PausedProxy {
    address: SocketAddr,
    is_paused: watch::Sender<bool>,
    has_server_data: watch::Sender<bool>,
    task: tokio::task::JoinHandle<()>,
}

impl PausedProxy {
    async fn start(target: SocketAddr) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind backpressure proxy");
        let address = listener.local_addr().expect("backpressure proxy address");
        let (is_paused, mut pause_rx) = watch::channel(false);
        let (has_server_data, _) = watch::channel(false);
        let server_data_tx = has_server_data.clone();
        let task = tokio::spawn(async move {
            let (client, _) = listener.accept().await.expect("accept proxy client");
            let upstream_socket = tokio::net::TcpSocket::new_v4().expect("create proxy socket");
            upstream_socket
                .set_recv_buffer_size(64 * 1024)
                .expect("constrain proxy receive buffer before connect");
            let upstream = upstream_socket
                .connect(target)
                .await
                .expect("connect proxy upstream");
            let (mut client_read, mut client_write) = client.into_split();
            let (mut upstream_read, mut upstream_write) = upstream.into_split();
            let client_to_server = tokio::spawn(async move {
                let _ = tokio::io::copy(&mut client_read, &mut upstream_write).await;
            });
            let mut buffer = [0u8; 64 * 1024];
            loop {
                if *pause_rx.borrow() {
                    tokio::select! {
                        changed = pause_rx.changed() => {
                            if changed.is_err() {
                                break;
                            }
                        }
                        ready = upstream_read.readable(), if !*server_data_tx.borrow() => {
                            if ready.is_err() {
                                break;
                            }
                            server_data_tx.send_replace(true);
                        }
                    }
                    continue;
                }
                tokio::select! {
                    biased;
                    changed = pause_rx.changed() => {
                        if changed.is_err() {
                            break;
                        }
                    }
                    read = upstream_read.read(&mut buffer) => match read {
                        Ok(0) | Err(_) => break,
                        Ok(read) => {
                            if client_write.write_all(&buffer[..read]).await.is_err() {
                                break;
                            }
                        }
                    }
                }
            }
            client_to_server.abort();
        });
        Self {
            address,
            is_paused,
            has_server_data,
            task,
        }
    }

    fn pause(&self) {
        self.is_paused.send_replace(true);
    }

    fn resume(&self) {
        self.is_paused.send_replace(false);
    }

    async fn wait_for_server_data(&self) {
        let mut data_rx = self.has_server_data.subscribe();
        while !*data_rx.borrow_and_update() {
            data_rx
                .changed()
                .await
                .expect("backpressure proxy data signal");
        }
    }
}

impl Drop for PausedProxy {
    fn drop(&mut self) {
        self.task.abort();
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
    async fn connect(address: SocketAddr, is_tls: bool, path: &str) -> Self {
        let tcp = TcpStream::connect(address)
            .await
            .expect("connect WebSocket test listener");
        let io = if is_tls {
            BoxIo(Box::new(connect_insecure_tls(tcp).await))
        } else {
            BoxIo(Box::new(tcp))
        };
        Self::upgrade(io, path).await
    }

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
            headers.lines().any(|line| {
                line.eq_ignore_ascii_case(&format!("Sec-WebSocket-Accept: {TEST_WEBSOCKET_ACCEPT}"))
            }),
            "upgrade must return the exact RFC accept value: {headers:?}"
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

    async fn send_ping(&mut self, payload: &[u8]) {
        self.send_data_frame(true, 0x9, payload).await;
    }

    async fn send_close(&mut self) {
        self.send_data_frame(true, 0x8, &[]).await;
    }

    async fn read_frame(&mut self) -> WireFrame {
        self.read_frame_result().await.expect("read server frame")
    }

    async fn read_frame_result(&mut self) -> std::io::Result<WireFrame> {
        tokio::time::timeout(Duration::from_secs(5), read_server_frame(&mut self.io))
            .await
            .expect("server frame timeout")
    }
}

#[derive(Default)]
struct TextReassembler {
    payload: Vec<u8>,
    is_open: bool,
}

impl TextReassembler {
    fn push(&mut self, frame: &WireFrame) -> Option<Value> {
        match frame.opcode {
            0x1 => {
                assert!(!self.is_open, "text frame cannot overlap a message");
                self.payload.extend_from_slice(&frame.payload);
                self.is_open = !frame.is_final;
            }
            0x0 => {
                assert!(self.is_open, "continuation requires an open message");
                self.payload.extend_from_slice(&frame.payload);
                self.is_open = !frame.is_final;
            }
            _ => return None,
        }
        if self.is_open {
            return None;
        }
        let event = serde_json::from_slice(&self.payload).expect("reassembled event is JSON");
        self.payload.clear();
        Some(event)
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
    if first & 0x70 != 0 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "server frame sets an unsupported RSV bit",
        ));
    }
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
    if payload_len > MAX_TEST_FRAME_BYTES {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "server frame exceeds the test reader bound",
        ));
    }
    if opcode & 0x08 != 0 && (!is_final || payload_len > 125) {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "server emitted an invalid control frame",
        ));
    }
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

fn large_turn_request(model_bytes: usize) -> String {
    format!(
        r#"{{"type":"response.create","model":"{}","input":[]}}"#,
        "m".repeat(model_bytes)
    )
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
    let headers = server
        .raw_request(
            b"GET /mock/websocket/turns HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: Upgrade\r\nUpgrade: websocket\r\nSec-WebSocket-Version: 13\r\n\r\n",
        )
        .await;
    assert!(
        headers.starts_with("HTTP/1.1 400 "),
        "missing key must fail before upgrade: {headers:?}"
    );
}

#[tokio::test]
async fn raw_upgrade_rejects_ambiguous_or_body_framed_handshakes() {
    let server = TestServer::start(fragmented_config(), false).await;
    let invalid_headers = [
        "Sec-WebSocket-Key: invalid\r\n",
        "Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\nSec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n",
        "Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\nSec-WebSocket-Version: 13\r\n",
        "Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\nUpgrade: websocket\r\n",
        "Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\nContent-Length: 1\r\n",
        "Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\nTransfer-Encoding: chunked\r\n",
    ];
    for (index, extra) in invalid_headers.into_iter().enumerate() {
        let mut request = format!(
            "GET /mock/websocket/turns HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: Upgrade\r\nUpgrade: websocket\r\nSec-WebSocket-Version: 13\r\n{extra}\r\n"
        );
        if index == 4 {
            request.push('x');
        } else if index == 5 {
            request.push_str("0\r\n\r\n");
        }
        let headers = server.raw_request(request.as_bytes()).await;
        assert!(
            headers.starts_with("HTTP/1.1 400 "),
            "invalid handshake case {index} must fail before upgrade: {headers:?}"
        );
    }
}

#[tokio::test]
async fn realtime_reads_upload_and_ping_while_output_is_scheduled() {
    let server = TestServer::start(
        MockServerConfig {
            websocket_mode: WebSocketMode::Realtime,
            websocket_scenario: WebSocketScenario::Normal,
            websocket_fragment_bytes: 5,
            websocket_first_content_delay_ms: 250.0,
            websocket_content_interval_ms: 0.0,
            no_tokenizer: true,
            ..MockServerConfig::default()
        },
        false,
    )
    .await;
    let mut client = server.connect(false, "/mock/websocket/realtime").await;
    client
        .send_text(r#"{"type":"conversation.item.create"}"#)
        .await;
    client
        .send_text(r#"{"type":"input_audio_buffer.commit"}"#)
        .await;
    client.send_text(r#"{"type":"response.create"}"#).await;
    client
        .send_text(r#"{"type":"input_audio_buffer.append","audio":"AA=="}"#)
        .await;
    client.send_ping(b"duplex").await;

    let pong = tokio::time::timeout(Duration::from_millis(100), client.read_frame())
        .await
        .expect("Realtime must keep reading control while content is scheduled");
    assert_eq!(pong.opcode, 0xa);
    assert_eq!(pong.payload, b"duplex");
}

#[tokio::test]
async fn backpressured_fragmented_output_prioritizes_pong_over_remaining_data_on_ws_and_wss() {
    for is_tls in [false, true] {
        let server = TestServer::start(
            MockServerConfig {
                websocket_mode: WebSocketMode::TurnSerialized,
                websocket_scenario: WebSocketScenario::Normal,
                websocket_fragment_bytes: 64 * 1024,
                websocket_first_content_delay_ms: 0.0,
                websocket_content_interval_ms: 0.0,
                no_tokenizer: true,
                ..MockServerConfig::default()
            },
            is_tls,
        )
        .await;
        let proxy = PausedProxy::start(server.address).await;
        let mut client =
            RawWebSocketClient::connect(proxy.address, is_tls, "/mock/websocket/turns").await;
        proxy.pause();
        client.send_text(&large_turn_request(5 * 1024 * 1024)).await;
        tokio::time::timeout(Duration::from_secs(1), proxy.wait_for_server_data())
            .await
            .expect("server output must reach the paused proxy");
        client.send_ping(b"backpressure").await;
        tokio::time::sleep(Duration::from_millis(10)).await;
        proxy.resume();

        tokio::time::timeout(Duration::from_secs(5), async {
            let mut is_message_open = false;
            loop {
                let frame = client.read_frame().await;
                match frame.opcode {
                    0x1 => {
                        assert!(!is_message_open, "text message cannot overlap");
                        is_message_open = !frame.is_final;
                    }
                    0x0 => {
                        assert!(is_message_open, "continuation requires an open message");
                        is_message_open = !frame.is_final;
                    }
                    0xa => {
                        assert_eq!(frame.payload, b"backpressure");
                        assert!(
                            is_message_open,
                            "Pong must overtake the remaining fragments after pressure releases"
                        );
                        break;
                    }
                    opcode => panic!("unexpected server opcode {opcode:#x}"),
                }
            }
        })
        .await
        .expect("backpressured Pong proof must finish within five seconds");
    }
}

#[tokio::test]
async fn server_close_deadline_includes_blocked_output_on_ws_and_wss() {
    for is_tls in [false, true] {
        let server = TestServer::start(
            MockServerConfig {
                websocket_mode: WebSocketMode::TurnSerialized,
                websocket_scenario: WebSocketScenario::CloseBeforeTerminal,
                websocket_fragment_bytes: 64 * 1024,
                websocket_first_content_delay_ms: 0.0,
                websocket_content_interval_ms: 0.0,
                no_tokenizer: true,
                ..MockServerConfig::default()
            },
            is_tls,
        )
        .await;
        let proxy = PausedProxy::start(server.address).await;
        let mut client =
            RawWebSocketClient::connect(proxy.address, is_tls, "/mock/websocket/turns").await;
        proxy.pause();
        client.send_text(&large_turn_request(5 * 1024 * 1024)).await;

        let captures = server
            .wait_for_captures_for(1, Duration::from_millis(1500))
            .await;
        assert_eq!(captures[0]["close"], "close_handshake_timeout");
    }
}

#[tokio::test]
async fn client_close_is_answered_over_ws_and_wss() {
    for is_tls in [false, true] {
        let server = TestServer::start(fragmented_config(), is_tls).await;
        let mut client = server.connect(is_tls, "/mock/websocket/turns").await;
        client.send_close().await;
        assert_eq!(client.read_frame().await.opcode, 0x8);
        let captures = server.wait_for_captures(1).await;
        assert_eq!(captures[0]["close"], "client_close");
    }
}

#[tokio::test]
async fn terminal_then_immediate_client_close_remains_successful_over_ws_and_wss() {
    for is_tls in [false, true] {
        let server = TestServer::start(fragmented_config(), is_tls).await;
        let mut client = server.connect(is_tls, "/mock/websocket/turns").await;
        client
            .send_text(r#"{"type":"response.create","model":"mock","input":"hello"}"#)
            .await;
        let mut messages = TextReassembler::default();
        loop {
            let frame = client.read_frame().await;
            if messages
                .push(&frame)
                .is_some_and(|event| event["type"] == "response.completed")
            {
                client.send_close().await;
                break;
            }
        }
        loop {
            if client.read_frame().await.opcode == 0x8 {
                break;
            }
        }

        let captures = server.wait_for_captures(1).await;
        assert_eq!(captures[0]["terminal"], "completed");
        assert_eq!(captures[0]["close"], "client_close");
        assert_eq!(server.request_total("200"), 1);
        assert_eq!(server.request_total("500"), 0);
    }
}

#[tokio::test]
async fn preterminal_server_close_waits_for_peer_over_ws_and_wss() {
    for is_tls in [false, true] {
        let server = TestServer::start(
            MockServerConfig {
                websocket_mode: WebSocketMode::TurnSerialized,
                websocket_scenario: WebSocketScenario::CloseBeforeTerminal,
                websocket_fragment_bytes: 5,
                websocket_first_content_delay_ms: 0.0,
                websocket_content_interval_ms: 0.0,
                no_tokenizer: true,
                ..MockServerConfig::default()
            },
            is_tls,
        )
        .await;
        let mut client = server.connect(is_tls, "/mock/websocket/turns").await;
        client
            .send_text(r#"{"type":"response.create","model":"mock","input":"hello"}"#)
            .await;
        let mut messages = TextReassembler::default();
        let mut has_terminal = false;
        loop {
            let frame = client.read_frame().await;
            if frame.opcode == 0x8 {
                break;
            }
            has_terminal |= messages
                .push(&frame)
                .is_some_and(|event| event["type"] == "response.completed");
        }
        assert!(!has_terminal, "pre-terminal close must not complete");
        client.send_close().await;
        let captures = server.wait_for_captures(1).await;
        assert_eq!(captures[0]["close"], "clean_server_close");
        assert!(captures[0]["events"].as_array().is_some_and(|events| {
            events
                .iter()
                .any(|event| event["direction"] == "in" && event["opcode"] == "close")
        }));
    }
}

#[tokio::test]
async fn postterminal_dirty_drop_has_no_close_frame_over_ws_and_wss() {
    for is_tls in [false, true] {
        let server = TestServer::start(
            MockServerConfig {
                websocket_mode: WebSocketMode::TurnSerialized,
                websocket_scenario: WebSocketScenario::DirtyCloseAfterTerminal,
                websocket_fragment_bytes: 5,
                websocket_first_content_delay_ms: 0.0,
                websocket_content_interval_ms: 0.0,
                no_tokenizer: true,
                ..MockServerConfig::default()
            },
            is_tls,
        )
        .await;
        let mut client = server.connect(is_tls, "/mock/websocket/turns").await;
        client
            .send_text(r#"{"type":"response.create","model":"mock","input":"hello"}"#)
            .await;
        let mut wire = Vec::new();
        loop {
            match client.read_frame_result().await {
                Ok(frame) => {
                    assert_ne!(frame.opcode, 0x8, "dirty drop must not send Close");
                    wire.extend_from_slice(&frame.payload);
                }
                Err(_) => break,
            }
        }
        assert!(
            wire.windows(b"response.completed".len())
                .any(|window| window == b"response.completed"),
            "dirty drop must follow terminal completion"
        );
        let captures = server.wait_for_captures(1).await;
        assert_eq!(captures[0]["close"], "dirty_transport_drop");
        assert_eq!(captures[0]["terminal"], "completed");
    }
}

#[tokio::test]
async fn raw_reader_rejects_oversized_or_invalid_server_frames_before_allocation() {
    let (mut writer, mut reader) = tokio::io::duplex(64);
    writer
        .write_all(&[0x82, 0x7f, 0, 0, 0, 0, 0, 128, 0, 1])
        .await
        .expect("write oversized frame header");
    let error = read_server_frame(&mut reader)
        .await
        .expect_err("oversized frame must fail before payload allocation");
    assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
}

#[test]
fn text_reassembler_detects_an_event_type_split_across_fragments() {
    let mut messages = TextReassembler::default();
    assert!(
        messages
            .push(&WireFrame {
                is_final: false,
                opcode: 0x1,
                payload: br#"{"type":"response."#.to_vec(),
            })
            .is_none()
    );
    let event = messages
        .push(&WireFrame {
            is_final: true,
            opcode: 0x0,
            payload: br#"completed"}"#.to_vec(),
        })
        .expect("final fragment completes the event");
    assert_eq!(event["type"], "response.completed");
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
