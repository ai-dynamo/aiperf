// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! WebSocket upgrade and frame emission shared by the mock scenario loop.

use std::fmt::{self, Display, Formatter};
use std::future::Future;

use axum::body::Body;
use axum::extract::Request;
use axum::extract::ws::{Message as AxumMessage, WebSocket};
use axum::response::Response;
use base64::Engine as _;
use base64::engine::general_purpose::STANDARD as BASE64_STANDARD;
use bytes::Bytes;
use futures::stream::{SplitSink, SplitStream};
use futures::{SinkExt, StreamExt};
use http::header::{
    CONNECTION, CONTENT_LENGTH, SEC_WEBSOCKET_ACCEPT, SEC_WEBSOCKET_KEY, SEC_WEBSOCKET_VERSION,
    TRANSFER_ENCODING, UPGRADE,
};
use http::{HeaderMap, HeaderValue, Method, StatusCode, Version};
use hyper::upgrade::OnUpgrade;
use hyper_util::rt::TokioIo;
use tokio_tungstenite::WebSocketStream;
use tokio_tungstenite::tungstenite::handshake::derive_accept_key;
use tokio_tungstenite::tungstenite::protocol::frame::Frame;
use tokio_tungstenite::tungstenite::protocol::frame::coding::{Data, OpCode};
use tokio_tungstenite::tungstenite::protocol::{
    Message as TungsteniteMessage, Role, WebSocketConfig,
};

type RawWebSocket = WebSocketStream<TokioIo<hyper::upgrade::Upgraded>>;

/// One control frame optionally placed inside the following fragmented message.
#[derive(Clone, Debug)]
pub(super) enum OutboundControl {
    Ping(Bytes),
    Pong(Bytes),
}

impl OutboundControl {
    pub(super) fn opcode(&self) -> &'static str {
        match self {
            Self::Ping(_) => "ping",
            Self::Pong(_) => "pong",
        }
    }

    pub(super) fn payload(&self) -> &Bytes {
        match self {
            Self::Ping(payload) | Self::Pong(payload) => payload,
        }
    }

    fn tungstenite_message(&self) -> TungsteniteMessage {
        match self {
            Self::Ping(payload) => TungsteniteMessage::Ping(payload.clone()),
            Self::Pong(payload) => TungsteniteMessage::Pong(payload.clone()),
        }
    }

    fn axum_message(&self) -> AxumMessage {
        match self {
            Self::Ping(payload) => AxumMessage::Ping(payload.clone()),
            Self::Pong(payload) => AxumMessage::Pong(payload.clone()),
        }
    }
}

/// Reassembled inbound messages consumed by the protocol loop.
pub(super) enum InboundMessage {
    Text(Bytes),
    Binary(Bytes),
    Ping(Bytes),
    Pong(Bytes),
    Close,
}

/// Ordinary Axum sockets and raw-frame sockets behind one scenario-loop API.
pub(super) enum ConnectionSocket {
    Axum(WebSocket),
    Raw(RawWebSocket),
}

impl ConnectionSocket {
    pub(super) fn from_axum(socket: WebSocket) -> Self {
        Self::Axum(socket)
    }

    fn from_raw(socket: RawWebSocket) -> Self {
        Self::Raw(socket)
    }

    pub(super) fn split(self) -> (ConnectionReader, ConnectionWriter) {
        match self {
            Self::Axum(socket) => {
                let (writer, reader) = socket.split();
                (
                    ConnectionReader::Axum(reader),
                    ConnectionWriter::Axum(writer),
                )
            }
            Self::Raw(socket) => {
                let (writer, reader) = socket.split();
                (ConnectionReader::Raw(reader), ConnectionWriter::Raw(writer))
            }
        }
    }
}

/// Read half continuously driven by the connection coordinator.
pub(super) enum ConnectionReader {
    Axum(SplitStream<WebSocket>),
    Raw(SplitStream<RawWebSocket>),
}

impl ConnectionReader {
    pub(super) async fn recv(&mut self) -> Option<Result<InboundMessage, WireError>> {
        match self {
            Self::Axum(socket) => socket.next().await.map(|result| {
                result
                    .map(axum_inbound)
                    .map_err(|error| WireError::new(error.to_string()))
            }),
            Self::Raw(socket) => socket.next().await.map(|result| {
                result
                    .map_err(|error| WireError::new(error.to_string()))
                    .and_then(tungstenite_inbound)
            }),
        }
    }
}

/// Write half exclusively owned by the bounded connection writer task.
pub(super) enum ConnectionWriter {
    Axum(SplitSink<WebSocket, AxumMessage>),
    Raw(SplitSink<RawWebSocket, TungsteniteMessage>),
}

/// One frame in a validated outbound text message.
#[derive(Clone)]
pub(super) enum TextFrame {
    Complete(String),
    Fragment {
        payload: Bytes,
        opcode: OpCode,
        is_final: bool,
    },
}

impl TextFrame {
    pub(super) fn is_final(&self) -> bool {
        match self {
            Self::Complete(_) => true,
            Self::Fragment { is_final, .. } => *is_final,
        }
    }
}

/// Lazily authored frames for one outbound text message.
pub(super) enum TextMessage {
    Complete(Option<String>),
    Fragmented {
        payload: Bytes,
        fragment_bytes: usize,
        start: usize,
        has_emitted_empty: bool,
    },
}

impl TextMessage {
    pub(super) fn next_frame(&mut self) -> Result<Option<TextFrame>, WireError> {
        match self {
            Self::Complete(text) => Ok(text.take().map(TextFrame::Complete)),
            Self::Fragmented {
                payload,
                fragment_bytes,
                start,
                has_emitted_empty,
            } => {
                if payload.is_empty() {
                    if *has_emitted_empty {
                        return Ok(None);
                    }
                    *has_emitted_empty = true;
                    return Ok(Some(TextFrame::Fragment {
                        payload: Bytes::new(),
                        opcode: OpCode::Data(Data::Text),
                        is_final: true,
                    }));
                }
                if *start == payload.len() {
                    return Ok(None);
                }
                let mut end = start.saturating_add(*fragment_bytes).min(payload.len());
                while end > *start
                    && end < payload.len()
                    && payload[end] & 0b1100_0000 == 0b1000_0000
                {
                    end -= 1;
                }
                if end == *start {
                    return Err(WireError::new(
                        "fragment size cannot hold the next UTF-8 scalar".to_owned(),
                    ));
                }
                let is_final = end == payload.len();
                let opcode = if *start == 0 {
                    OpCode::Data(Data::Text)
                } else {
                    OpCode::Data(Data::Continue)
                };
                let frame = TextFrame::Fragment {
                    payload: payload.slice(*start..end),
                    opcode,
                    is_final,
                };
                *start = end;
                Ok(Some(frame))
            }
        }
    }
}

impl ConnectionWriter {
    pub(super) fn prepare_text(
        &self,
        payload: Bytes,
        fragment_bytes: usize,
        max_message_bytes: usize,
    ) -> Result<TextMessage, WireError> {
        if payload.len() > max_message_bytes {
            return Err(WireError::new(format!(
                "outbound application message is {} bytes, exceeding configured maximum {max_message_bytes}",
                payload.len()
            )));
        }
        if fragment_bytes != 0 && fragment_bytes < 4 {
            return Err(WireError::new(
                "fragment size must be zero or at least four bytes".to_owned(),
            ));
        }
        if matches!(self, Self::Raw(_)) && fragment_bytes > 0 {
            std::str::from_utf8(&payload)
                .map_err(|error| WireError::new(format!("outbound text is not UTF-8: {error}")))?;
            Ok(TextMessage::Fragmented {
                payload,
                fragment_bytes,
                start: 0,
                has_emitted_empty: false,
            })
        } else {
            Ok(TextMessage::Complete(Some(outbound_text(&payload)?)))
        }
    }

    pub(super) async fn feed_text_frame(&mut self, frame: &TextFrame) -> Result<(), WireError> {
        match (self, frame) {
            (Self::Axum(socket), TextFrame::Complete(text)) => socket
                .feed(AxumMessage::Text(text.clone().into()))
                .await
                .map_err(|error| WireError::new(error.to_string())),
            (Self::Raw(socket), TextFrame::Complete(text)) => socket
                .feed(TungsteniteMessage::Text(text.clone().into()))
                .await
                .map_err(|error| WireError::new(error.to_string())),
            (
                Self::Raw(socket),
                TextFrame::Fragment {
                    payload,
                    opcode,
                    is_final,
                },
            ) => socket
                .feed(TungsteniteMessage::Frame(Frame::message(
                    payload.clone(),
                    *opcode,
                    *is_final,
                )))
                .await
                .map_err(|error| WireError::new(error.to_string())),
            (Self::Axum(_), TextFrame::Fragment { .. }) => Err(WireError::new(
                "raw text fragment selected for an Axum socket".to_owned(),
            )),
        }
    }

    pub(super) async fn send_control(
        &mut self,
        control: &OutboundControl,
    ) -> Result<(), WireError> {
        match self {
            Self::Axum(socket) => socket
                .send(control.axum_message())
                .await
                .map_err(|error| WireError::new(error.to_string())),
            Self::Raw(socket) => socket
                .send(control.tungstenite_message())
                .await
                .map_err(|error| WireError::new(error.to_string())),
        }
    }

    pub(super) async fn send_close(&mut self) -> Result<(), WireError> {
        match self {
            Self::Axum(socket) => socket
                .send(AxumMessage::Close(None))
                .await
                .map_err(|error| WireError::new(error.to_string())),
            Self::Raw(socket) => socket
                .send(TungsteniteMessage::Close(None))
                .await
                .map_err(|error| WireError::new(error.to_string())),
        }
    }

    pub(super) async fn flush(&mut self) -> Result<(), WireError> {
        match self {
            Self::Axum(socket) => socket
                .flush()
                .await
                .map_err(|error| WireError::new(error.to_string())),
            Self::Raw(socket) => socket
                .flush()
                .await
                .map_err(|error| WireError::new(error.to_string())),
        }
    }
}

fn outbound_text(payload: &Bytes) -> Result<String, WireError> {
    std::str::from_utf8(payload)
        .map(str::to_owned)
        .map_err(|error| WireError::new(format!("outbound text is not UTF-8: {error}")))
}

fn axum_inbound(message: AxumMessage) -> InboundMessage {
    match message {
        AxumMessage::Text(text) => InboundMessage::Text(Bytes::copy_from_slice(text.as_bytes())),
        AxumMessage::Binary(payload) => InboundMessage::Binary(payload),
        AxumMessage::Ping(payload) => InboundMessage::Ping(payload),
        AxumMessage::Pong(payload) => InboundMessage::Pong(payload),
        AxumMessage::Close(_) => InboundMessage::Close,
    }
}

fn tungstenite_inbound(message: TungsteniteMessage) -> Result<InboundMessage, WireError> {
    Ok(match message {
        TungsteniteMessage::Text(text) => {
            InboundMessage::Text(Bytes::copy_from_slice(text.as_bytes()))
        }
        TungsteniteMessage::Binary(payload) => InboundMessage::Binary(payload),
        TungsteniteMessage::Ping(payload) => InboundMessage::Ping(payload),
        TungsteniteMessage::Pong(payload) => InboundMessage::Pong(payload),
        TungsteniteMessage::Close(_) => InboundMessage::Close,
        TungsteniteMessage::Frame(_) => {
            return Err(WireError::new(
                "tungstenite exposed an unreassembled inbound frame".to_owned(),
            ));
        }
    })
}

/// Validated HTTP/1 WebSocket upgrade used only when raw framing is authored.
pub(super) struct RawUpgrade {
    on_upgrade: OnUpgrade,
    response: Response,
    config: WebSocketConfig,
}

impl RawUpgrade {
    pub(super) fn from_request(
        mut request: Request,
        max_message_bytes: usize,
    ) -> Result<Self, Response> {
        if request.version() != Version::HTTP_11 || request.method() != Method::GET {
            return Err(rejection(StatusCode::METHOD_NOT_ALLOWED));
        }
        let headers = request.headers();
        if !header_has_token(headers, CONNECTION, b"upgrade")
            || !single_header_eq(headers, UPGRADE, b"websocket")
            || !single_header_eq(headers, SEC_WEBSOCKET_VERSION, b"13")
            || headers.contains_key(CONTENT_LENGTH)
            || headers.contains_key(TRANSFER_ENCODING)
        {
            return Err(rejection(StatusCode::BAD_REQUEST));
        }
        let Some(key) = single_header(headers, SEC_WEBSOCKET_KEY).cloned() else {
            return Err(rejection(StatusCode::BAD_REQUEST));
        };
        let Ok(decoded_key) = BASE64_STANDARD.decode(key.as_bytes()) else {
            return Err(rejection(StatusCode::BAD_REQUEST));
        };
        if decoded_key.len() != 16 {
            return Err(rejection(StatusCode::BAD_REQUEST));
        }
        let Some(on_upgrade) = request.extensions_mut().remove::<OnUpgrade>() else {
            return Err(rejection(StatusCode::UPGRADE_REQUIRED));
        };
        let mut response = rejection(StatusCode::SWITCHING_PROTOCOLS);
        response
            .headers_mut()
            .insert(CONNECTION, HeaderValue::from_static("upgrade"));
        response
            .headers_mut()
            .insert(UPGRADE, HeaderValue::from_static("websocket"));
        let accept = HeaderValue::from_str(&derive_accept_key(key.as_bytes()))
            .map_err(|_| rejection(StatusCode::BAD_REQUEST))?;
        response.headers_mut().insert(SEC_WEBSOCKET_ACCEPT, accept);
        let mut config = WebSocketConfig::default();
        config.max_message_size = Some(max_message_bytes);
        config.max_frame_size = Some(max_message_bytes);
        Ok(Self {
            on_upgrade,
            response,
            config,
        })
    }

    pub(super) fn on_upgrade<C, Fut>(self, callback: C) -> Response
    where
        C: FnOnce(ConnectionSocket) -> Fut + Send + 'static,
        Fut: Future<Output = ()> + Send + 'static,
    {
        tokio::spawn(async move {
            let upgraded = match self.on_upgrade.await {
                Ok(upgraded) => upgraded,
                Err(error) => {
                    tracing::debug!(component = "websocket_mock", error = %error, "raw WebSocket upgrade failed");
                    return;
                }
            };
            let socket = WebSocketStream::from_raw_socket(
                TokioIo::new(upgraded),
                Role::Server,
                Some(self.config),
            )
            .await;
            callback(ConnectionSocket::from_raw(socket)).await;
        });
        self.response
    }
}

fn rejection(status: StatusCode) -> Response {
    let mut response = Response::new(Body::empty());
    *response.status_mut() = status;
    response
}

fn single_header_eq(headers: &HeaderMap, name: http::header::HeaderName, expected: &[u8]) -> bool {
    single_header(headers, name)
        .is_some_and(|value| value.as_bytes().eq_ignore_ascii_case(expected))
}

fn single_header(headers: &HeaderMap, name: http::header::HeaderName) -> Option<&HeaderValue> {
    let mut values = headers.get_all(name).iter();
    let value = values.next()?;
    values.next().is_none().then_some(value)
}

fn header_has_token(headers: &HeaderMap, name: http::header::HeaderName, expected: &[u8]) -> bool {
    headers.get_all(name).iter().any(|value| {
        value
            .as_bytes()
            .split(|byte| *byte == b',')
            .any(|token| token.trim_ascii().eq_ignore_ascii_case(expected))
    })
}

#[derive(Debug)]
pub(super) struct WireError {
    message: String,
}

impl WireError {
    fn new(message: String) -> Self {
        Self { message }
    }
}

impl Display for WireError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for WireError {}
