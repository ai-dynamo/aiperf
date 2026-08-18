// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! WebSocket upgrade and frame emission shared by the mock scenario loop.

use std::fmt::{self, Display, Formatter};
use std::future::Future;

use axum::body::Body;
use axum::extract::Request;
use axum::extract::ws::{Message as AxumMessage, WebSocket};
use axum::response::Response;
use bytes::Bytes;
use futures::{SinkExt, StreamExt};
use http::header::{
    CONNECTION, SEC_WEBSOCKET_ACCEPT, SEC_WEBSOCKET_KEY, SEC_WEBSOCKET_VERSION, UPGRADE,
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

    pub(super) async fn send_text(
        &mut self,
        payload: Bytes,
        fragment_bytes: usize,
        max_message_bytes: usize,
        interjected: Option<&OutboundControl>,
    ) -> Result<(), WireError> {
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
        let text = std::str::from_utf8(&payload)
            .map_err(|error| WireError::new(format!("outbound text is not UTF-8: {error}")))?;
        match self {
            Self::Axum(socket) => {
                if let Some(control) = interjected {
                    socket
                        .send(control.axum_message())
                        .await
                        .map_err(|error| WireError::new(error.to_string()))?;
                }
                socket
                    .send(AxumMessage::Text(text.to_owned().into()))
                    .await
                    .map_err(|error| WireError::new(error.to_string()))
            }
            Self::Raw(socket) if fragment_bytes > 0 => {
                send_fragmented_text(socket, payload, text.len(), fragment_bytes, interjected).await
            }
            Self::Raw(socket) => {
                if let Some(control) = interjected {
                    socket
                        .send(control.tungstenite_message())
                        .await
                        .map_err(|error| WireError::new(error.to_string()))?;
                }
                socket
                    .send(TungsteniteMessage::Text(text.to_owned().into()))
                    .await
                    .map_err(|error| WireError::new(error.to_string()))
            }
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

async fn send_fragmented_text(
    socket: &mut RawWebSocket,
    payload: Bytes,
    text_len: usize,
    fragment_bytes: usize,
    interjected: Option<&OutboundControl>,
) -> Result<(), WireError> {
    if text_len <= fragment_bytes {
        if let Some(control) = interjected {
            socket
                .send(control.tungstenite_message())
                .await
                .map_err(|error| WireError::new(error.to_string()))?;
        }
    }
    if text_len == 0 {
        return socket
            .send(TungsteniteMessage::Frame(Frame::message(
                Bytes::new(),
                OpCode::Data(Data::Text),
                true,
            )))
            .await
            .map_err(|error| WireError::new(error.to_string()));
    }
    let mut start = 0;
    while start < text_len {
        let mut end = start.saturating_add(fragment_bytes).min(text_len);
        while end > start && end < text_len && payload[end] & 0b1100_0000 == 0b1000_0000 {
            end -= 1;
        }
        if end == start {
            return Err(WireError::new(
                "fragment size cannot hold the next UTF-8 scalar".to_owned(),
            ));
        }
        let is_final = end == text_len;
        let opcode = if start == 0 {
            OpCode::Data(Data::Text)
        } else {
            OpCode::Data(Data::Continue)
        };
        socket
            .send(TungsteniteMessage::Frame(Frame::message(
                payload.slice(start..end),
                opcode,
                is_final,
            )))
            .await
            .map_err(|error| WireError::new(error.to_string()))?;
        if start == 0
            && !is_final
            && let Some(control) = interjected
        {
            socket
                .send(control.tungstenite_message())
                .await
                .map_err(|error| WireError::new(error.to_string()))?;
        }
        start = end;
    }
    Ok(())
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
            || !header_eq(headers, UPGRADE, b"websocket")
            || !header_eq(headers, SEC_WEBSOCKET_VERSION, b"13")
        {
            return Err(rejection(StatusCode::BAD_REQUEST));
        }
        let Some(key) = headers.get(SEC_WEBSOCKET_KEY).cloned() else {
            return Err(rejection(StatusCode::BAD_REQUEST));
        };
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

fn header_eq(headers: &HeaderMap, name: http::header::HeaderName, expected: &[u8]) -> bool {
    headers
        .get(name)
        .is_some_and(|value| value.as_bytes().eq_ignore_ascii_case(expected))
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
