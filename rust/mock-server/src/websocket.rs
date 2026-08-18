// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Typed protocol vocabulary and scenario state for mock WebSocket routes.

use std::collections::VecDeque;
use std::fmt::{self, Display, Formatter};
use std::ops::{Deref, DerefMut};
use std::sync::Arc;

use aiperf_runtime::clock::sleep_ns;
use axum::Json;
use axum::extract::ws::WebSocketUpgrade;
use axum::extract::{Request, State};
use axum::response::Response;
use base64::Engine as _;
use base64::engine::general_purpose::STANDARD as BASE64_STANDARD;
use bytes::Bytes;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::config::{MockServerConfig, WebSocketControl, WebSocketScenario};
use crate::metrics::MetricRecorder;
use crate::models::Usage;
use crate::state::AppState;

mod wire;

use wire::{ConnectionSocket, InboundMessage, OutboundControl, RawUpgrade};

const MAX_CAPTURE_EVENTS_PER_CONNECTION: usize = 16_384;

/// Sanitized metadata for one completed WebSocket connection.
#[derive(Clone, Debug, Default, Serialize)]
pub(crate) struct WebSocketCapture {
    connection_id: u64,
    route: &'static str,
    scenario: WebSocketScenario,
    terminal: TerminalClassification,
    close: CloseClassification,
    dropped_events: usize,
    events: Vec<WebSocketCaptureEvent>,
}

impl WebSocketCapture {
    fn new(
        connection_id: u64,
        route: RouteKind,
        scenario: WebSocketScenario,
        event_capacity: usize,
    ) -> Self {
        Self {
            connection_id,
            route: route.endpoint_label(),
            scenario,
            terminal: TerminalClassification::None,
            close: CloseClassification::Open,
            dropped_events: 0,
            events: Vec::with_capacity(event_capacity.min(MAX_CAPTURE_EVENTS_PER_CONNECTION)),
        }
    }

    fn push_event(&mut self, event: WebSocketCaptureEvent) {
        if self.events.len() == MAX_CAPTURE_EVENTS_PER_CONNECTION {
            self.dropped_events = self.dropped_events.saturating_add(1);
            return;
        }
        self.events.push(event);
    }
}

#[derive(Clone, Copy, Debug, Default, Serialize)]
#[serde(rename_all = "snake_case")]
enum TerminalClassification {
    #[default]
    None,
    Completed,
    ContinuationRejected,
    ProtocolError,
}

#[derive(Clone, Copy, Debug, Default, Serialize)]
#[serde(rename_all = "snake_case")]
enum CloseClassification {
    #[default]
    Open,
    ClientClose,
    CleanServerClose,
    DirtyTransportDrop,
    ReceiveError,
    SendError,
    ProtocolError,
    Cancelled,
}

/// Sanitized metadata for one direction of a WebSocket message.
#[derive(Clone, Debug, Serialize)]
pub(crate) struct WebSocketCaptureEvent {
    direction: &'static str,
    opcode: &'static str,
    event_type: Option<&'static str>,
    turn: Option<u64>,
    operation_id: Option<String>,
    bytes: usize,
    payload_digest: String,
    relative_ns: i64,
}

/// Bounded completed-connection capture retention.
pub(crate) struct WebSocketCaptureStore {
    capacity: usize,
    entries: Mutex<VecDeque<WebSocketCapture>>,
}

impl WebSocketCaptureStore {
    /// Create a store whose capacity was validated during configuration loading.
    pub(crate) fn new(capacity: usize) -> Self {
        Self {
            capacity,
            entries: Mutex::new(VecDeque::new()),
        }
    }

    fn push(&self, capture: WebSocketCapture) {
        if self.capacity == 0 {
            return;
        }
        let mut entries = self.entries.lock();
        if entries.len() >= self.capacity {
            entries.pop_front();
        }
        entries.push_back(capture);
    }

    fn snapshot(&self) -> Vec<WebSocketCapture> {
        self.entries.lock().iter().cloned().collect()
    }
}

struct CapturePublication<'a> {
    store: &'a WebSocketCaptureStore,
    capture: WebSocketCapture,
}

impl<'a> CapturePublication<'a> {
    fn new(store: &'a WebSocketCaptureStore, capture: WebSocketCapture) -> Self {
        Self { store, capture }
    }
}

impl Deref for CapturePublication<'_> {
    type Target = WebSocketCapture;

    fn deref(&self) -> &Self::Target {
        &self.capture
    }
}

impl DerefMut for CapturePublication<'_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.capture
    }
}

impl Drop for CapturePublication<'_> {
    fn drop(&mut self) {
        let mut capture = std::mem::take(&mut self.capture);
        if matches!(capture.close, CloseClassification::Open) {
            capture.close = CloseClassification::Cancelled;
        }
        self.store.push(capture);
    }
}

#[derive(Clone, Debug)]
struct CaptureAttribution {
    turn: u64,
    operation_id: String,
}

fn attribute_event(
    event: &ClientEvent,
    connection_id: u64,
    next_turn: &mut u64,
    attribution: &mut Option<CaptureAttribution>,
) {
    let starts_turn = matches!(event, ClientEvent::StartTurn { .. })
        || attribution.is_none()
            && matches!(
                event,
                ClientEvent::AppendAudio { .. }
                    | ClientEvent::AddConversationItem
                    | ClientEvent::RequestResponse
            );
    if starts_turn {
        *next_turn = next_turn.saturating_add(1);
        *attribution = Some(CaptureAttribution {
            turn: *next_turn,
            operation_id: format!("mock-ws-operation-{connection_id}-{next_turn}"),
        });
    }
}

/// Upgrade the serialized Responses mock route.
pub(crate) async fn turns_upgrade(
    State(state): State<Arc<AppState>>,
    websocket: WebSocketUpgrade,
) -> Response {
    let max_message_bytes = state.config.websocket_max_message_bytes;
    websocket
        .max_message_size(max_message_bytes)
        .max_frame_size(max_message_bytes)
        .on_upgrade(move |socket| {
            serve_connection(state, ConnectionSocket::from_axum(socket), RouteKind::Turns)
        })
}

/// Upgrade the serialized Responses route through the raw-frame writer.
pub(crate) async fn turns_raw_upgrade(
    State(state): State<Arc<AppState>>,
    request: Request,
) -> Response {
    raw_upgrade(state, request, RouteKind::Turns)
}

/// Upgrade the duplex Realtime mock route.
pub(crate) async fn realtime_upgrade(
    State(state): State<Arc<AppState>>,
    websocket: WebSocketUpgrade,
) -> Response {
    let max_message_bytes = state.config.websocket_max_message_bytes;
    websocket
        .max_message_size(max_message_bytes)
        .max_frame_size(max_message_bytes)
        .on_upgrade(move |socket| {
            serve_connection(
                state,
                ConnectionSocket::from_axum(socket),
                RouteKind::Realtime,
            )
        })
}

/// Upgrade the duplex Realtime route through the raw-frame writer.
pub(crate) async fn realtime_raw_upgrade(
    State(state): State<Arc<AppState>>,
    request: Request,
) -> Response {
    raw_upgrade(state, request, RouteKind::Realtime)
}

fn raw_upgrade(state: Arc<AppState>, request: Request, route: RouteKind) -> Response {
    let max_message_bytes = state.config.websocket_max_message_bytes;
    match RawUpgrade::from_request(request, max_message_bytes) {
        Ok(upgrade) => upgrade.on_upgrade(move |socket| serve_connection(state, socket, route)),
        Err(response) => response,
    }
}

/// Return sanitized completed WebSocket connection captures.
pub(crate) async fn captures(State(state): State<Arc<AppState>>) -> Json<Vec<WebSocketCapture>> {
    Json(state.websocket_captures.snapshot())
}

async fn serve_connection(state: Arc<AppState>, mut socket: ConnectionSocket, route: RouteKind) {
    let connection_id = state
        .websocket_connections
        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let started_ns = state.clock_anchor.now_ns();
    let mut capture = CapturePublication::new(
        &state.websocket_captures,
        WebSocketCapture::new(
            connection_id,
            route,
            state.config.websocket_scenario,
            usize::try_from(state.config.websocket_content_events)
                .unwrap_or(MAX_CAPTURE_EVENTS_PER_CONNECTION)
                .saturating_add(8),
        ),
    );
    let mut scenario = ConnectionScenario::new(route, connection_id, &state.config);
    let mut operation: Option<OperationAccounting<'_>> = None;
    let mut pending_request_bytes = 0usize;
    let mut next_capture_turn = 0u64;
    let mut attribution: Option<CaptureAttribution> = None;

    while let Some(message) = socket.recv().await {
        let now_ns = state.clock_anchor.now_ns();
        let message = match message {
            Ok(message) => message,
            Err(error) => {
                tracing::debug!(component = "websocket_mock", error = %error, "WebSocket receive failed");
                capture.close = CloseClassification::ReceiveError;
                break;
            }
        };
        match message {
            InboundMessage::Text(payload) => {
                if let Some(operation) = operation.as_mut() {
                    operation.add_request_bytes(payload.len());
                } else {
                    pending_request_bytes = pending_request_bytes.saturating_add(payload.len());
                }
                if payload.len() > state.config.websocket_max_message_bytes {
                    capture.push_event(capture_event_with_attribution(
                        "in",
                        "text",
                        &payload,
                        attribution.as_ref(),
                        now_ns,
                        started_ns,
                    ));
                    let response_bytes = protocol_error(
                        &mut socket,
                        &mut capture,
                        state.clock_anchor,
                        started_ns,
                        attribution.as_ref(),
                        state.config.websocket_fragment_bytes,
                        state.config.websocket_max_message_bytes,
                        "application message exceeds configured size",
                    )
                    .await;
                    if let Some(operation) = operation.as_mut() {
                        operation.add_response_bytes(response_bytes);
                    }
                    break;
                }
                let event = match parse_client_event(&payload, route) {
                    Ok(event) => event,
                    Err(error) => {
                        capture.push_event(capture_event_with_attribution(
                            "in",
                            "text",
                            &payload,
                            attribution.as_ref(),
                            now_ns,
                            started_ns,
                        ));
                        let response_bytes = protocol_error(
                            &mut socket,
                            &mut capture,
                            state.clock_anchor,
                            started_ns,
                            attribution.as_ref(),
                            state.config.websocket_fragment_bytes,
                            state.config.websocket_max_message_bytes,
                            &error.to_string(),
                        )
                        .await;
                        if let Some(operation) = operation.as_mut() {
                            operation.add_response_bytes(response_bytes);
                        }
                        break;
                    }
                };
                attribute_event(
                    &event,
                    connection_id,
                    &mut next_capture_turn,
                    &mut attribution,
                );
                capture.push_event(capture_event_with_attribution(
                    "in",
                    "text",
                    &payload,
                    attribution.as_ref(),
                    now_ns,
                    started_ns,
                ));
                let operation_model = event.operation_model(route).map(str::to_owned);
                if let Some(model) = operation_model {
                    if operation.is_some() {
                        let response_bytes = protocol_error(
                            &mut socket,
                            &mut capture,
                            state.clock_anchor,
                            started_ns,
                            attribution.as_ref(),
                            state.config.websocket_fragment_bytes,
                            state.config.websocket_max_message_bytes,
                            "connection already has an in-flight operation",
                        )
                        .await;
                        if let Some(operation) = operation.as_mut() {
                            operation.add_response_bytes(response_bytes);
                        }
                        break;
                    }
                    operation = Some(OperationAccounting::begin(
                        &state.recorder,
                        route,
                        model,
                        now_ns,
                        pending_request_bytes,
                    ));
                    pending_request_bytes = 0;
                }
                let actions = match scenario.on_event(event, now_ns) {
                    Ok(actions) => actions,
                    Err(error) => {
                        let response_bytes = protocol_error(
                            &mut socket,
                            &mut capture,
                            state.clock_anchor,
                            started_ns,
                            attribution.as_ref(),
                            state.config.websocket_fragment_bytes,
                            state.config.websocket_max_message_bytes,
                            &error.to_string(),
                        )
                        .await;
                        if let Some(operation) = operation.as_mut() {
                            operation.add_response_bytes(response_bytes);
                        }
                        break;
                    }
                };
                let result = send_actions(
                    &state,
                    &mut socket,
                    &mut capture,
                    started_ns,
                    attribution.as_ref(),
                    actions,
                )
                .await;
                if let Some(operation) = operation.as_mut() {
                    operation.add_response_bytes(result.response_bytes);
                }
                let operation_finished = result.operation.is_some();
                match result.operation {
                    Some(OperationResult::Completed { completion_tokens }) => {
                        capture.terminal = TerminalClassification::Completed;
                        if let Some(operation) = operation.take() {
                            operation.complete(state.clock_anchor.now_ns(), completion_tokens);
                        }
                    }
                    Some(OperationResult::ContinuationRejected) => {
                        capture.terminal = TerminalClassification::ContinuationRejected;
                        if let Some(operation) = operation.take() {
                            operation.reject();
                        }
                    }
                    None => {}
                }
                if let Some(close) = result.close {
                    capture.close = close;
                }
                if operation_finished {
                    attribution = None;
                }
                if !result.keep_connection {
                    break;
                }
            }
            InboundMessage::Binary(payload) => {
                capture.push_event(capture_event_with_attribution(
                    "in",
                    "binary",
                    &payload,
                    attribution.as_ref(),
                    now_ns,
                    started_ns,
                ));
                let response_bytes = protocol_error(
                    &mut socket,
                    &mut capture,
                    state.clock_anchor,
                    started_ns,
                    attribution.as_ref(),
                    state.config.websocket_fragment_bytes,
                    state.config.websocket_max_message_bytes,
                    "binary application messages are not supported",
                )
                .await;
                if let Some(operation) = operation.as_mut() {
                    operation.add_request_bytes(payload.len());
                    operation.add_response_bytes(response_bytes);
                }
                break;
            }
            InboundMessage::Ping(payload) => {
                capture.push_event(capture_event_with_attribution(
                    "in",
                    "ping",
                    &payload,
                    attribution.as_ref(),
                    now_ns,
                    started_ns,
                ));
                let mut event = capture_event_with_attribution(
                    "out",
                    "pong",
                    &payload,
                    attribution.as_ref(),
                    state.clock_anchor.now_ns(),
                    started_ns,
                );
                if socket
                    .send_control(&OutboundControl::Pong(payload))
                    .await
                    .is_err()
                {
                    capture.close = CloseClassification::SendError;
                    break;
                }
                event.relative_ns = state.clock_anchor.now_ns().saturating_sub(started_ns);
                capture.push_event(event);
            }
            InboundMessage::Pong(payload) => {
                capture.push_event(capture_event_with_attribution(
                    "in",
                    "pong",
                    &payload,
                    attribution.as_ref(),
                    now_ns,
                    started_ns,
                ));
            }
            InboundMessage::Close => {
                capture.push_event(capture_event_with_attribution(
                    "in",
                    "close",
                    &[],
                    attribution.as_ref(),
                    now_ns,
                    started_ns,
                ));
                capture.close = CloseClassification::ClientClose;
                break;
            }
        }
    }
    if matches!(capture.close, CloseClassification::Open) {
        capture.close = CloseClassification::ReceiveError;
    }
    drop(operation);
}

#[derive(Clone, Copy)]
enum OperationResult {
    Completed { completion_tokens: usize },
    ContinuationRejected,
}

struct ActionResult {
    keep_connection: bool,
    operation: Option<OperationResult>,
    response_bytes: usize,
    close: Option<CloseClassification>,
}

impl Default for ActionResult {
    fn default() -> Self {
        Self {
            keep_connection: true,
            operation: None,
            response_bytes: 0,
            close: None,
        }
    }
}

async fn send_actions(
    state: &AppState,
    socket: &mut ConnectionSocket,
    capture: &mut WebSocketCapture,
    started_ns: i64,
    attribution: Option<&CaptureAttribution>,
    actions: Vec<ServerAction>,
) -> ActionResult {
    let mut result = ActionResult::default();
    let mut pending_control: Option<OutboundControl> = None;
    for action in actions {
        match action {
            ServerAction::SendText { at_ns, payload } => {
                let delay_ns = at_ns.saturating_sub(state.clock_anchor.now_ns());
                if delay_ns > 0 {
                    sleep_ns(delay_ns).await;
                }
                let mut event = capture_event_with_attribution(
                    "out",
                    "text",
                    &payload,
                    attribution,
                    state.clock_anchor.now_ns(),
                    started_ns,
                );
                let payload_len = payload.len();
                if let Err(error) = socket
                    .send_text(
                        payload,
                        state.config.websocket_fragment_bytes,
                        state.config.websocket_max_message_bytes,
                        pending_control.as_ref(),
                    )
                    .await
                {
                    tracing::debug!(component = "websocket_mock", error = %error, "WebSocket send failed");
                    result.keep_connection = false;
                    result.close = Some(CloseClassification::SendError);
                    return result;
                }
                event.relative_ns = state.clock_anchor.now_ns().saturating_sub(started_ns);
                result.response_bytes = result.response_bytes.saturating_add(payload_len);
                if let Some(control) = pending_control.take() {
                    capture.push_event(capture_event_with_attribution(
                        "out",
                        control.opcode(),
                        control.payload(),
                        attribution,
                        state.clock_anchor.now_ns(),
                        started_ns,
                    ));
                }
                capture.push_event(event);
            }
            ServerAction::SendPing(payload) => {
                if state.config.websocket_fragment_bytes > 0 {
                    pending_control = Some(OutboundControl::Ping(payload));
                    continue;
                }
                let mut event = capture_event_with_attribution(
                    "out",
                    "ping",
                    &payload,
                    attribution,
                    state.clock_anchor.now_ns(),
                    started_ns,
                );
                if socket
                    .send_control(&OutboundControl::Ping(payload))
                    .await
                    .is_err()
                {
                    result.keep_connection = false;
                    result.close = Some(CloseClassification::SendError);
                    return result;
                }
                event.relative_ns = state.clock_anchor.now_ns().saturating_sub(started_ns);
                capture.push_event(event);
            }
            ServerAction::SendPong(payload) => {
                if state.config.websocket_fragment_bytes > 0 {
                    pending_control = Some(OutboundControl::Pong(payload));
                    continue;
                }
                let mut event = capture_event_with_attribution(
                    "out",
                    "pong",
                    &payload,
                    attribution,
                    state.clock_anchor.now_ns(),
                    started_ns,
                );
                if socket
                    .send_control(&OutboundControl::Pong(payload))
                    .await
                    .is_err()
                {
                    result.keep_connection = false;
                    result.close = Some(CloseClassification::SendError);
                    return result;
                }
                event.relative_ns = state.clock_anchor.now_ns().saturating_sub(started_ns);
                capture.push_event(event);
            }
            ServerAction::Close => {
                result.keep_connection = false;
                result.close = Some(if socket.send_close().await.is_ok() {
                    capture.push_event(capture_event_with_attribution(
                        "out",
                        "close",
                        &[],
                        attribution,
                        state.clock_anchor.now_ns(),
                        started_ns,
                    ));
                    CloseClassification::CleanServerClose
                } else {
                    CloseClassification::SendError
                });
                return result;
            }
            ServerAction::DropTransport => {
                result.keep_connection = false;
                result.close = Some(CloseClassification::DirtyTransportDrop);
                return result;
            }
            ServerAction::CompleteOperation { completion_tokens } => {
                result.operation = Some(OperationResult::Completed { completion_tokens });
            }
            ServerAction::RejectContinuation => {
                result.operation = Some(OperationResult::ContinuationRejected);
            }
        }
    }
    if let Some(control) = pending_control {
        if socket.send_control(&control).await.is_err() {
            result.keep_connection = false;
            result.close = Some(CloseClassification::SendError);
        } else {
            capture.push_event(capture_event_with_attribution(
                "out",
                control.opcode(),
                control.payload(),
                attribution,
                state.clock_anchor.now_ns(),
                started_ns,
            ));
        }
    }
    result
}

async fn protocol_error(
    socket: &mut ConnectionSocket,
    capture: &mut WebSocketCapture,
    clock_anchor: aiperf_runtime::clock::RealClockAnchor,
    started_ns: i64,
    attribution: Option<&CaptureAttribution>,
    fragment_bytes: usize,
    max_message_bytes: usize,
    message: &str,
) -> usize {
    let payload = serde_json::json!({"type":"error","error":{"message":message}}).to_string();
    let bytes = payload.len();
    let mut event = capture_event_with_attribution(
        "out",
        "text",
        payload.as_bytes(),
        attribution,
        clock_anchor.now_ns(),
        started_ns,
    );
    if socket
        .send_text(
            Bytes::from(payload),
            fragment_bytes,
            max_message_bytes,
            None,
        )
        .await
        .is_ok()
    {
        event.relative_ns = clock_anchor.now_ns().saturating_sub(started_ns);
        capture.push_event(event);
    } else {
        capture.close = CloseClassification::SendError;
        return 0;
    }
    let is_close_sent = socket.send_close().await.is_ok();
    if is_close_sent {
        capture.push_event(capture_event_with_attribution(
            "out",
            "close",
            &[],
            attribution,
            clock_anchor.now_ns(),
            started_ns,
        ));
    }
    capture.terminal = TerminalClassification::ProtocolError;
    capture.close = if is_close_sent {
        CloseClassification::ProtocolError
    } else {
        CloseClassification::SendError
    };
    bytes
}

#[cfg(test)]
fn capture_event(
    direction: &'static str,
    opcode: &'static str,
    payload: &[u8],
    now_ns: i64,
    started_ns: i64,
) -> WebSocketCaptureEvent {
    capture_event_with_attribution(direction, opcode, payload, None, now_ns, started_ns)
}

fn capture_event_with_attribution(
    direction: &'static str,
    opcode: &'static str,
    payload: &[u8],
    attribution: Option<&CaptureAttribution>,
    now_ns: i64,
    started_ns: i64,
) -> WebSocketCaptureEvent {
    let event_type = canonical_event_type(payload);
    WebSocketCaptureEvent {
        direction,
        opcode,
        event_type,
        turn: attribution.map(|attribution| attribution.turn),
        operation_id: attribution.map(|attribution| attribution.operation_id.clone()),
        bytes: payload.len(),
        payload_digest: blake3::hash(payload).to_hex().to_string(),
        relative_ns: now_ns.saturating_sub(started_ns),
    }
}

fn canonical_event_type(payload: &[u8]) -> Option<&'static str> {
    let value = serde_json::from_slice::<Value>(payload).ok()?;
    let event_type = value.get("type")?.as_str()?;
    Some(match event_type {
        "response.create" => "response.create",
        "session.update" => "session.update",
        "input_audio_buffer.append" => "input_audio_buffer.append",
        "conversation.item.create" => "conversation.item.create",
        "input_audio_buffer.commit" => "input_audio_buffer.commit",
        "response.created" => "response.created",
        "response.output_text.delta" => "response.output_text.delta",
        "response.completed" => "response.completed",
        "response.continuation_rejected" => "response.continuation_rejected",
        "response.text.delta" => "response.text.delta",
        "response.audio.delta" => "response.audio.delta",
        "response.done" => "response.done",
        "error" => "error",
        _ => "unknown",
    })
}

/// The mock WebSocket route selected during HTTP upgrade.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum RouteKind {
    /// Serialized OpenAI Responses turns.
    Turns,
    /// Duplex OpenAI Realtime events.
    Realtime,
}

impl RouteKind {
    /// Stable mock endpoint label.
    pub(crate) const fn endpoint_label(self) -> &'static str {
        match self {
            Self::Turns => "mock_websocket_turns",
            Self::Realtime => "mock_websocket_realtime",
        }
    }
}

/// One validated client application event.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum ClientEvent {
    /// Starts a serialized Responses operation.
    StartTurn {
        model: String,
        continuation: Option<String>,
    },
    /// Configures a Realtime session.
    ConfigureSession,
    /// Adds audio to a Realtime input buffer.
    AppendAudio { bytes: usize },
    /// Adds a conversational input item to a Realtime session.
    AddConversationItem,
    /// Commits Realtime input.
    CommitInput,
    /// Requests a Realtime response.
    RequestResponse,
}

impl ClientEvent {
    fn operation_model(&self, route: RouteKind) -> Option<&str> {
        match (route, self) {
            (RouteKind::Turns, Self::StartTurn { model, .. }) => Some(model),
            (RouteKind::Realtime, Self::RequestResponse) => Some("mock-realtime"),
            _ => None,
        }
    }
}

/// A protocol violation detected before scenario execution.
#[derive(Debug, Eq, PartialEq)]
pub(crate) struct ProtocolError {
    message: String,
}

impl ProtocolError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl Display for ProtocolError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for ProtocolError {}

/// Parse a complete JSON application message under its route-specific grammar.
pub(crate) fn parse_client_event(
    payload: &[u8],
    route: RouteKind,
) -> Result<ClientEvent, ProtocolError> {
    let value: Value = serde_json::from_slice(payload)
        .map_err(|error| ProtocolError::new(format!("invalid WebSocket JSON: {error}")))?;
    let object = value
        .as_object()
        .ok_or_else(|| ProtocolError::new("WebSocket event must be a JSON object"))?;
    let event_type = object
        .get("type")
        .and_then(Value::as_str)
        .ok_or_else(|| ProtocolError::new("WebSocket event has no string type"))?
        .to_owned();

    match (route, event_type.as_str()) {
        (RouteKind::Turns, "response.create") => {
            #[derive(Deserialize)]
            struct ResponsesCreate {
                model: String,
                input: Value,
                #[serde(default)]
                previous_response_id: Option<String>,
            }

            let create: ResponsesCreate = serde_json::from_value(value).map_err(|error| {
                ProtocolError::new(format!("invalid response.create event: {error}"))
            })?;
            if create.model.is_empty() {
                return Err(ProtocolError::new(
                    "response.create model must not be empty",
                ));
            }
            if matches!(create.input, Value::Null) {
                return Err(ProtocolError::new("response.create input must not be null"));
            }
            if create.previous_response_id.as_deref() == Some("") {
                return Err(ProtocolError::new(
                    "response.create previous_response_id must not be empty",
                ));
            }
            Ok(ClientEvent::StartTurn {
                model: create.model,
                continuation: create.previous_response_id,
            })
        }
        (RouteKind::Realtime, "session.update") => Ok(ClientEvent::ConfigureSession),
        (RouteKind::Realtime, "input_audio_buffer.append") => {
            let audio = object
                .get("audio")
                .and_then(Value::as_str)
                .ok_or_else(|| ProtocolError::new("audio append requires audio"))?;
            let audio = BASE64_STANDARD
                .decode(audio)
                .map_err(|error| ProtocolError::new(format!("invalid base64 audio: {error}")))?;
            Ok(ClientEvent::AppendAudio { bytes: audio.len() })
        }
        (RouteKind::Realtime, "conversation.item.create") => Ok(ClientEvent::AddConversationItem),
        (RouteKind::Realtime, "input_audio_buffer.commit") => Ok(ClientEvent::CommitInput),
        (RouteKind::Realtime, "response.create") => Ok(ClientEvent::RequestResponse),
        _ => Err(ProtocolError::new(format!(
            "event type {event_type:?} is not valid for {}",
            route.endpoint_label()
        ))),
    }
}

/// One server action at an absolute mock-clock target.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum ServerAction {
    /// Send one complete JSON text application message.
    SendText { at_ns: i64, payload: Bytes },
    /// Send one Ping control frame.
    SendPing(Bytes),
    /// Send one Pong control frame.
    SendPong(Bytes),
    /// Close with a protocol-error application message already emitted.
    Close,
    /// End the underlying transport without a close handshake.
    DropTransport,
    /// Mark the preceding terminal event as a successful logical operation.
    CompleteOperation { completion_tokens: usize },
    /// Mark the preceding event as a recoverable continuation rejection.
    RejectContinuation,
}

impl ServerAction {
    #[cfg(test)]
    fn is_terminal(&self) -> bool {
        matches!(self, Self::CompleteOperation { .. })
    }

    #[cfg(test)]
    fn is_scheduled_content(&self) -> bool {
        matches!(self, Self::SendText { payload, .. } if payload.windows(b"delta".len()).any(|window| window == b"delta"))
    }

    #[cfg(test)]
    fn is_text_delta(&self) -> bool {
        matches!(self, Self::SendText { payload, .. } if payload.windows(b"response.text.delta".len()).any(|window| window == b"response.text.delta"))
    }
}

struct OperationAccounting<'a> {
    recorder: &'a MetricRecorder,
    route: RouteKind,
    model: String,
    started_ns: i64,
    request_bytes: u64,
    response_bytes: u64,
    is_active: bool,
}

impl<'a> OperationAccounting<'a> {
    fn begin(
        recorder: &'a MetricRecorder,
        route: RouteKind,
        model: String,
        started_ns: i64,
        request_bytes: usize,
    ) -> Self {
        recorder.init_model_config(&model);
        recorder.record_streaming_start(route.endpoint_label(), &model);
        recorder.record_request_start(route.endpoint_label(), &model);
        Self {
            recorder,
            route,
            model,
            started_ns,
            request_bytes: usize_to_u64(request_bytes),
            response_bytes: 0,
            is_active: true,
        }
    }

    fn add_request_bytes(&mut self, bytes: usize) {
        self.request_bytes = self.request_bytes.saturating_add(usize_to_u64(bytes));
    }

    fn add_response_bytes(&mut self, bytes: usize) {
        self.response_bytes = self.response_bytes.saturating_add(usize_to_u64(bytes));
    }

    fn complete(mut self, completed_ns: i64, completion_tokens: usize) {
        let endpoint = self.route.endpoint_label();
        let usage = websocket_usage(completion_tokens);
        self.recorder
            .record_request_bytes(endpoint, self.request_bytes, self.response_bytes);
        self.recorder
            .record_token_metrics(endpoint, &self.model, &usage);
        self.recorder.record_basic_success(
            endpoint,
            completed_ns.saturating_sub(self.started_ns).max(0) as f64 / 1_000_000_000.0,
        );
        self.recorder.record_request_end(endpoint);
        self.is_active = false;
    }

    fn reject(mut self) {
        let endpoint = self.route.endpoint_label();
        self.recorder
            .record_request_bytes(endpoint, self.request_bytes, self.response_bytes);
        self.recorder
            .record_error(endpoint, "websocket_continuation_rejected");
        self.recorder.record_request_end(endpoint);
        self.is_active = false;
    }
}

impl Drop for OperationAccounting<'_> {
    fn drop(&mut self) {
        if !self.is_active {
            return;
        }
        let endpoint = self.route.endpoint_label();
        self.recorder
            .record_request_bytes(endpoint, self.request_bytes, self.response_bytes);
        self.recorder.record_error(endpoint, "websocket_incomplete");
        self.recorder.record_request_end(endpoint);
        self.is_active = false;
    }
}

fn websocket_usage(completion_tokens: usize) -> Usage {
    Usage {
        prompt_tokens: 1,
        completion_tokens,
        total_tokens: completion_tokens.saturating_add(1),
        completion_tokens_details: None,
        prompt_tokens_details: None,
        cache_creation_input_tokens: None,
        prompt_cache_miss_tokens: None,
        tool_use_prompt_token_count: None,
        prompt_audio_seconds: None,
        cache_read_input_tokens: None,
    }
}

fn usize_to_u64(value: usize) -> u64 {
    u64::try_from(value).unwrap_or(u64::MAX)
}

/// Connection-local deterministic scenario state.
pub(crate) struct ConnectionScenario {
    route: RouteKind,
    scenario: WebSocketScenario,
    connection_id: u64,
    next_turn: u64,
    last_completed_response_id: Option<String>,
    has_in_flight_turn: bool,
    has_realtime_input: bool,
    realtime_commit_ns: Option<i64>,
    has_interleaved_output: bool,
    content_events: u32,
    first_content_delay_ns: i64,
    content_interval_ns: i64,
    control_before_content: WebSocketControl,
}

impl ConnectionScenario {
    /// Construct state for one accepted socket.
    pub(crate) fn new(route: RouteKind, connection_id: u64, config: &MockServerConfig) -> Self {
        Self {
            route,
            scenario: config.websocket_scenario,
            connection_id,
            next_turn: 0,
            last_completed_response_id: None,
            has_in_flight_turn: false,
            has_realtime_input: false,
            realtime_commit_ns: None,
            has_interleaved_output: false,
            content_events: config.websocket_content_events,
            first_content_delay_ns: ms_to_ns(config.websocket_first_content_delay_ms),
            content_interval_ns: ms_to_ns(config.websocket_content_interval_ms),
            control_before_content: config.websocket_control_before_content,
        }
    }

    /// Apply one validated client event and return its deterministic effects.
    pub(crate) fn on_event(
        &mut self,
        event: ClientEvent,
        input_complete_ns: i64,
    ) -> Result<Vec<ServerAction>, ProtocolError> {
        match self.route {
            RouteKind::Turns => self.on_turn_event(event, input_complete_ns),
            RouteKind::Realtime => self.on_realtime_event(event, input_complete_ns),
        }
    }

    fn on_turn_event(
        &mut self,
        event: ClientEvent,
        input_complete_ns: i64,
    ) -> Result<Vec<ServerAction>, ProtocolError> {
        match event {
            ClientEvent::StartTurn {
                model,
                continuation,
            } => {
                if self.has_in_flight_turn {
                    return Err(ProtocolError::new(
                        "serialized turn already has an in-flight operation",
                    ));
                }
                if self.scenario == WebSocketScenario::StaleReuse && self.next_turn > 0 {
                    return Ok(vec![ServerAction::DropTransport]);
                }
                if self.scenario == WebSocketScenario::RejectContinuation && continuation.is_some()
                {
                    return Ok(vec![
                        self.text_action(
                            input_complete_ns,
                            serde_json::json!({
                                "type":"response.continuation_rejected",
                                "error":{"message":"previous response is unavailable"},
                            }),
                        ),
                        ServerAction::RejectContinuation,
                    ]);
                }
                if let Some(continuation) = continuation
                    && self.last_completed_response_id.as_deref() != Some(continuation.as_str())
                {
                    return Err(ProtocolError::new(
                        "continuation does not match completed response",
                    ));
                }
                self.has_in_flight_turn = true;
                self.next_turn = self.next_turn.saturating_add(1);
                let response_id = format!("mock-ws-{}-{}", self.connection_id, self.next_turn);
                let mut actions = vec![self.text_action(
                    input_complete_ns,
                    serde_json::json!({
                        "type":"response.created",
                        "response":{
                            "id":response_id,
                            "object":"response",
                            "status":"in_progress",
                            "model":model,
                        },
                    }),
                )];
                self.append_control(&mut actions);
                let first_content_ns =
                    input_complete_ns.saturating_add(self.first_content_delay_ns);
                if self.scenario != WebSocketScenario::DoneOnly {
                    for index in 0..self.content_events {
                        actions.push(self.text_action_at(
                            first_content_ns.saturating_add(
                                self.content_interval_ns.saturating_mul(i64::from(index)),
                            ),
                            serde_json::json!({"type":"response.output_text.delta","delta":"mock"}),
                        ));
                    }
                }
                if self.scenario == WebSocketScenario::CloseBeforeTerminal {
                    actions.push(ServerAction::Close);
                    self.has_in_flight_turn = false;
                    return Ok(actions);
                }
                let terminal_at_ns = first_content_ns.saturating_add(
                    self.content_interval_ns
                        .saturating_mul(i64::from(self.content_events)),
                );
                let output_tokens = self.content_events.max(1) as usize;
                let output_text = "mock".repeat(output_tokens);
                actions.push(self.text_action_at(
                    terminal_at_ns,
                    serde_json::json!({
                        "type":"response.completed",
                        "response":{
                            "id":response_id,
                            "object":"response",
                            "status":"completed",
                            "model":model,
                            "output":[{
                                "type":"message",
                                "role":"assistant",
                                "content":[{"type":"output_text","text":output_text}],
                            }],
                            "usage":{
                                "input_tokens":1,
                                "output_tokens":output_tokens,
                                "total_tokens":output_tokens + 1,
                            },
                        },
                    }),
                ));
                actions.push(ServerAction::CompleteOperation {
                    completion_tokens: output_tokens,
                });
                self.last_completed_response_id = Some(response_id);
                self.has_in_flight_turn = false;
                if self.scenario == WebSocketScenario::DirtyCloseAfterTerminal {
                    actions.push(ServerAction::DropTransport);
                }
                Ok(actions)
            }
            _ => Err(ProtocolError::new("event is invalid for serialized turns")),
        }
    }

    fn on_realtime_event(
        &mut self,
        event: ClientEvent,
        input_complete_ns: i64,
    ) -> Result<Vec<ServerAction>, ProtocolError> {
        match event {
            ClientEvent::ConfigureSession => Ok(Vec::new()),
            ClientEvent::AddConversationItem => {
                self.has_realtime_input = true;
                Ok(Vec::new())
            }
            ClientEvent::AppendAudio { .. } => {
                self.has_realtime_input = true;
                if self.scenario == WebSocketScenario::InterleavedRealtime
                    && !self.has_interleaved_output
                {
                    self.has_interleaved_output = true;
                    return Ok(vec![self.text_action(
                        input_complete_ns,
                        serde_json::json!({"type":"response.text.delta","delta":"mock"}),
                    )]);
                }
                Ok(Vec::new())
            }
            ClientEvent::CommitInput => {
                if !self.has_realtime_input {
                    return Err(ProtocolError::new("cannot commit empty Realtime input"));
                }
                self.realtime_commit_ns = Some(input_complete_ns);
                Ok(Vec::new())
            }
            ClientEvent::RequestResponse => {
                let commit_ns = self.realtime_commit_ns.take().ok_or_else(|| {
                    ProtocolError::new("Realtime response requires committed input")
                })?;
                let first_content_ns = commit_ns.saturating_add(self.first_content_delay_ns);
                let mut actions = Vec::new();
                self.append_control(&mut actions);
                if self.scenario != WebSocketScenario::DoneOnly {
                    for index in 0..self.content_events.max(1) {
                        actions.push(self.text_action_at(
                            first_content_ns.saturating_add(
                                self.content_interval_ns.saturating_mul(i64::from(index)),
                            ),
                            serde_json::json!({"type":"response.text.delta","delta":"mock"}),
                        ));
                    }
                    actions.push(
                        self.text_action_at(
                            first_content_ns.saturating_add(
                                self.content_interval_ns
                                    .saturating_mul(i64::from(self.content_events.max(1))),
                            ),
                            serde_json::json!({"type":"response.audio.delta","delta":"AAE="}),
                        ),
                    );
                }
                let output_tokens = self.content_events.max(1) as usize;
                let output_text = "mock".repeat(output_tokens);
                let terminal_at_ns = if self.scenario == WebSocketScenario::DoneOnly {
                    first_content_ns
                } else {
                    first_content_ns.saturating_add(
                        self.content_interval_ns
                            .saturating_mul(i64::from(self.content_events.max(1))),
                    )
                };
                actions.push(self.text_action_at(
                    terminal_at_ns,
                    serde_json::json!({
                        "type":"response.done",
                        "response":{
                            "id":"mock-realtime",
                            "object":"realtime.response",
                            "status":"completed",
                            "output":[{
                                "type":"message",
                                "role":"assistant",
                                "content":[{"type":"output_text","text":output_text}],
                            }],
                            "usage":{
                                "input_tokens":1,
                                "output_tokens":output_tokens,
                                "total_tokens":output_tokens + 1,
                            },
                        },
                    }),
                ));
                actions.push(ServerAction::CompleteOperation {
                    completion_tokens: output_tokens,
                });
                self.has_realtime_input = false;
                self.has_interleaved_output = false;
                Ok(actions)
            }
            _ => Err(ProtocolError::new("event is invalid for Realtime")),
        }
    }

    fn append_control(&self, actions: &mut Vec<ServerAction>) {
        match self.control_before_content {
            WebSocketControl::None => {}
            WebSocketControl::Ping => {
                actions.push(ServerAction::SendPing(Bytes::from_static(b"mock")))
            }
            WebSocketControl::Pong => {
                actions.push(ServerAction::SendPong(Bytes::from_static(b"mock")))
            }
        }
    }

    fn text_action(&self, at_ns: i64, value: Value) -> ServerAction {
        self.text_action_at(at_ns, value)
    }

    fn text_action_at(&self, at_ns: i64, value: Value) -> ServerAction {
        ServerAction::SendText {
            at_ns,
            payload: Bytes::from(value.to_string()),
        }
    }
}

fn ms_to_ns(milliseconds: f64) -> i64 {
    (milliseconds * 1_000_000.0).round() as i64
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::SinkExt;
    use tokio_tungstenite::WebSocketStream;
    use tokio_tungstenite::connect_async;
    use tokio_tungstenite::tungstenite::Message as ClientMessage;

    type ClientSocket = WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>;

    struct TestServer {
        address: std::net::SocketAddr,
        state: Arc<AppState>,
        task: tokio::task::JoinHandle<()>,
    }

    impl TestServer {
        async fn start(config: MockServerConfig) -> Self {
            let state = AppState::build(config);
            let router = crate::app::build_router(state.clone());
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
                .await
                .expect("bind test listener");
            let address = listener.local_addr().expect("test listener address");
            let task = tokio::spawn(async move {
                let _ = axum::serve(listener, router).await;
            });
            Self {
                address,
                state,
                task,
            }
        }

        async fn connect(&self, path: &str) -> ClientSocket {
            let url = format!("ws://{}{path}", self.address);
            connect_async(url).await.expect("upgrade mock route").0
        }

        async fn wait_for_captures(&self, count: usize) -> Vec<WebSocketCapture> {
            for _ in 0..100 {
                let captures = self.state.websocket_captures.snapshot();
                if captures.len() >= count {
                    return captures;
                }
                tokio::time::sleep(std::time::Duration::from_millis(1)).await;
            }
            self.state.websocket_captures.snapshot()
        }
    }

    impl Drop for TestServer {
        fn drop(&mut self) {
            self.task.abort();
        }
    }

    async fn send_json(socket: &mut ClientSocket, payload: &str) {
        socket
            .send(ClientMessage::Text(payload.into()))
            .await
            .expect("send mock request event");
    }

    async fn read_json_event(socket: &mut ClientSocket, expected_type: &str) -> Value {
        while let Some(message) = socket.next().await {
            let message = message.expect("read mock response");
            let ClientMessage::Text(payload) = message else {
                continue;
            };
            let event: Value = serde_json::from_str(&payload).expect("mock output is JSON");
            if event["type"] == expected_type {
                return event;
            }
        }
        panic!("socket ended before {expected_type}");
    }

    #[test]
    fn turn_codec_accepts_response_create_and_rejects_invented_acknowledgement() {
        assert!(matches!(
            parse_client_event(
                br#"{"type":"response.create","model":"mock-model","input":[]}"#,
                RouteKind::Turns
            ),
            Ok(ClientEvent::StartTurn {
                continuation: None,
                ..
            })
        ));
        assert!(
            parse_client_event(
                br#"{"type":"response.acknowledge","response_id":"r1"}"#,
                RouteKind::Turns
            )
            .is_err()
        );
    }

    #[test]
    fn turn_codec_requires_the_responses_model_and_input() {
        assert!(
            parse_client_event(
                br#"{"type":"response.create","input":[]}"#,
                RouteKind::Turns
            )
            .is_err()
        );
        assert!(
            parse_client_event(
                br#"{"type":"response.create","model":"mock-model"}"#,
                RouteKind::Turns
            )
            .is_err()
        );
    }

    #[test]
    fn realtime_codec_rejects_non_json_input() {
        assert!(parse_client_event(b"not json", RouteKind::Realtime).is_err());
    }

    #[test]
    fn realtime_codec_rejects_invalid_base64_audio() {
        assert!(
            parse_client_event(
                br#"{"type":"input_audio_buffer.append","audio":"not base64"}"#,
                RouteKind::Realtime,
            )
            .is_err()
        );
    }

    #[test]
    fn realtime_response_requires_input_commit_and_anchors_to_it() {
        let config = MockServerConfig {
            websocket_first_content_delay_ms: 2.0,
            ..MockServerConfig::default()
        };
        let mut scenario = ConnectionScenario::new(RouteKind::Realtime, 1, &config);
        scenario
            .on_event(ClientEvent::ConfigureSession, 100)
            .expect("session configuration is valid");
        assert!(
            scenario
                .on_event(ClientEvent::RequestResponse, 200)
                .is_err()
        );
        scenario
            .on_event(ClientEvent::AppendAudio { bytes: 2 }, 300)
            .expect("audio append is valid");
        scenario
            .on_event(ClientEvent::CommitInput, 400)
            .expect("input commit is valid");
        let first_text_at = scenario
            .on_event(ClientEvent::RequestResponse, 10_000)
            .expect("response after commit is valid")
            .into_iter()
            .find_map(|action| match action {
                ServerAction::SendText { at_ns, payload }
                    if payload
                        .windows(b"response.text.delta".len())
                        .any(|window| window == b"response.text.delta") =>
                {
                    Some(at_ns)
                }
                _ => None,
            })
            .expect("response has a text delta");
        assert_eq!(first_text_at, 2_000_400);
    }

    #[test]
    fn normal_turn_schedules_content_then_completed() {
        let config = MockServerConfig::default();
        let mut scenario = ConnectionScenario::new(RouteKind::Turns, 1, &config);
        let actions = scenario
            .on_event(
                ClientEvent::StartTurn {
                    model: "mock-model".to_owned(),
                    continuation: None,
                },
                1_000,
            )
            .expect("normal turn is valid");
        assert!(actions.iter().any(ServerAction::is_scheduled_content));
        assert!(actions.iter().any(ServerAction::is_terminal));
    }

    #[test]
    fn normal_turn_terminal_repeats_the_complete_streamed_text() {
        let config = MockServerConfig {
            websocket_content_events: 2,
            ..MockServerConfig::default()
        };
        let mut scenario = ConnectionScenario::new(RouteKind::Turns, 1, &config);
        let terminal = scenario
            .on_event(
                ClientEvent::StartTurn {
                    model: "mock-model".to_owned(),
                    continuation: None,
                },
                1_000,
            )
            .expect("normal turn is valid")
            .into_iter()
            .find_map(|action| match action {
                ServerAction::SendText { payload, .. } => serde_json::from_slice::<Value>(&payload)
                    .ok()
                    .filter(|event| event["type"] == "response.completed"),
                _ => None,
            })
            .expect("normal turn has a terminal event");
        assert_eq!(
            terminal["response"]["output"][0]["content"][0]["text"],
            "mockmock"
        );
    }

    #[test]
    fn reject_continuation_scenario_rejects_the_matching_previous_response() {
        let config = MockServerConfig {
            websocket_scenario: WebSocketScenario::RejectContinuation,
            ..MockServerConfig::default()
        };
        let mut scenario = ConnectionScenario::new(RouteKind::Turns, 1, &config);
        let first = scenario
            .on_event(
                ClientEvent::StartTurn {
                    model: "mock-model".to_owned(),
                    continuation: None,
                },
                1_000,
            )
            .expect("first turn is valid");
        let response_id = first
            .iter()
            .find_map(|action| match action {
                ServerAction::SendText { payload, .. } => serde_json::from_slice::<Value>(payload)
                    .ok()
                    .filter(|event| event["type"] == "response.completed")
                    .and_then(|event| event["response"]["id"].as_str().map(str::to_owned)),
                _ => None,
            })
            .expect("first turn has a response identity");
        let second = scenario
            .on_event(
                ClientEvent::StartTurn {
                    model: "mock-model".to_owned(),
                    continuation: Some(response_id),
                },
                2_000,
            )
            .expect("configured rejection is a server event");
        assert!(second.iter().any(|action| {
            matches!(action, ServerAction::SendText { payload, .. }
                if payload.windows(b"response.continuation_rejected".len())
                    .any(|window| window == b"response.continuation_rejected"))
        }));
    }

    #[test]
    fn done_only_carries_content_in_the_terminal_response_output() {
        let config = MockServerConfig {
            websocket_scenario: WebSocketScenario::DoneOnly,
            websocket_content_events: 0,
            ..MockServerConfig::default()
        };
        let mut scenario = ConnectionScenario::new(RouteKind::Turns, 1, &config);
        let terminal = scenario
            .on_event(
                ClientEvent::StartTurn {
                    model: "mock-model".to_owned(),
                    continuation: None,
                },
                1_000,
            )
            .expect("done-only turn is valid")
            .into_iter()
            .find_map(|action| match action {
                ServerAction::SendText { payload, .. } => serde_json::from_slice::<Value>(&payload)
                    .ok()
                    .filter(|event| event["type"] == "response.completed"),
                _ => None,
            })
            .expect("done-only turn has a terminal event");
        assert_eq!(
            terminal["response"]["output"][0]["content"][0]["text"],
            "mock"
        );
    }

    #[test]
    fn realtime_done_only_terminal_carries_content_usage_and_uses_commit_target() {
        let config = MockServerConfig {
            websocket_scenario: WebSocketScenario::DoneOnly,
            websocket_content_events: 0,
            websocket_first_content_delay_ms: 2.0,
            websocket_content_interval_ms: 3.0,
            ..MockServerConfig::default()
        };
        let mut scenario = ConnectionScenario::new(RouteKind::Realtime, 1, &config);
        scenario
            .on_event(ClientEvent::AppendAudio { bytes: 2 }, 100)
            .expect("audio append is valid");
        scenario
            .on_event(ClientEvent::CommitInput, 1_000)
            .expect("input commit is valid");
        let (terminal_at, terminal) = scenario
            .on_event(ClientEvent::RequestResponse, 10_000)
            .expect("response after commit is valid")
            .into_iter()
            .find_map(|action| match action {
                ServerAction::SendText { at_ns, payload } => {
                    serde_json::from_slice::<Value>(&payload)
                        .ok()
                        .filter(|event| event["type"] == "response.done")
                        .map(|event| (at_ns, event))
                }
                _ => None,
            })
            .expect("done-only response has a terminal event");
        assert_eq!(terminal_at, 2_001_000);
        assert_eq!(terminal["response"]["status"], "completed");
        assert_eq!(
            terminal["response"]["output"][0]["content"][0]["text"],
            "mock"
        );
        assert_eq!(terminal["response"]["usage"]["output_tokens"], 1);
    }

    #[test]
    fn realtime_interleaving_emits_output_after_audio_before_commit() {
        let config = MockServerConfig {
            websocket_scenario: WebSocketScenario::InterleavedRealtime,
            ..MockServerConfig::default()
        };
        let mut scenario = ConnectionScenario::new(RouteKind::Realtime, 1, &config);
        let actions = scenario
            .on_event(ClientEvent::AppendAudio { bytes: 4 }, 1_000)
            .expect("audio append is valid");
        assert!(actions.iter().any(ServerAction::is_text_delta));
    }

    #[test]
    fn content_targets_use_one_absolute_anchor() {
        let config = MockServerConfig {
            websocket_content_events: 2,
            websocket_first_content_delay_ms: 2.0,
            websocket_content_interval_ms: 3.0,
            ..MockServerConfig::default()
        };
        let mut scenario = ConnectionScenario::new(RouteKind::Turns, 1, &config);
        let actions = scenario
            .on_event(
                ClientEvent::StartTurn {
                    model: "mock-model".to_owned(),
                    continuation: None,
                },
                10,
            )
            .expect("normal turn is valid");
        let content_times = actions
            .iter()
            .filter_map(|action| match action {
                ServerAction::SendText { at_ns, payload }
                    if payload
                        .windows(b"response.output_text.delta".len())
                        .any(|window| window == b"response.output_text.delta") =>
                {
                    Some(*at_ns)
                }
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(content_times, vec![2_000_010, 5_000_010]);
    }

    #[tokio::test]
    async fn routes_are_absent_when_websocket_mode_is_disabled() {
        use axum::body::Body;
        use http::Request;
        use tower::ServiceExt;

        let router =
            crate::app::build_router(crate::state::AppState::build(MockServerConfig::default()));
        let response = router
            .oneshot(
                Request::builder()
                    .uri("/mock/websocket/turns")
                    .body(Body::empty())
                    .expect("request is valid"),
            )
            .await
            .expect("router response");
        assert_eq!(response.status(), http::StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn realtime_wire_requires_commit_and_done_only_carries_terminal_content() {
        let server = TestServer::start(MockServerConfig {
            websocket_mode: crate::config::WebSocketMode::Realtime,
            websocket_scenario: WebSocketScenario::DoneOnly,
            websocket_content_events: 0,
            websocket_first_content_delay_ms: 0.0,
            websocket_content_interval_ms: 0.0,
            no_tokenizer: true,
            ..MockServerConfig::default()
        })
        .await;

        let mut rejected = server.connect("/mock/websocket/realtime").await;
        send_json(&mut rejected, r#"{"type":"session.update"}"#).await;
        send_json(&mut rejected, r#"{"type":"response.create"}"#).await;
        let error = read_json_event(&mut rejected, "error").await;
        assert!(
            error["error"]["message"]
                .as_str()
                .is_some_and(|message| message.contains("committed input"))
        );

        let mut socket = server.connect("/mock/websocket/realtime").await;
        send_json(
            &mut socket,
            r#"{"type":"input_audio_buffer.append","audio":"AAE="}"#,
        )
        .await;
        send_json(&mut socket, r#"{"type":"input_audio_buffer.commit"}"#).await;
        send_json(&mut socket, r#"{"type":"response.create"}"#).await;
        let done = read_json_event(&mut socket, "response.done").await;
        assert_eq!(done["response"]["status"], "completed");
        assert_eq!(done["response"]["output"][0]["content"][0]["text"], "mock");
        assert_eq!(done["response"]["usage"]["output_tokens"], 1);

        let endpoint = RouteKind::Realtime.endpoint_label();
        assert_eq!(
            server
                .state
                .recorder
                .metrics
                .aiperf
                .REQUESTS_TOTAL
                .with_label_values(&[endpoint, "POST", "500"])
                .get(),
            1
        );
        assert_eq!(
            server
                .state
                .recorder
                .metrics
                .aiperf
                .REQUESTS_TOTAL
                .with_label_values(&[endpoint, "POST", "200"])
                .get(),
            1
        );
    }

    #[tokio::test]
    async fn reused_turns_wire_attributes_rejection_and_full_history_recovery() {
        let server = TestServer::start(MockServerConfig {
            websocket_mode: crate::config::WebSocketMode::TurnSerialized,
            websocket_scenario: WebSocketScenario::RejectContinuation,
            websocket_first_content_delay_ms: 0.0,
            websocket_content_interval_ms: 0.0,
            no_tokenizer: true,
            ..MockServerConfig::default()
        })
        .await;
        let mut socket = server.connect("/mock/websocket/turns").await;

        send_json(
            &mut socket,
            r#"{"type":"response.create","model":"mock-model","input":[]}"#,
        )
        .await;
        let first = read_json_event(&mut socket, "response.completed").await;
        let response_id = first["response"]["id"]
            .as_str()
            .expect("completed response has an id");
        send_json(
            &mut socket,
            &format!(
                r#"{{"type":"response.create","model":"mock-model","input":[],"previous_response_id":"{response_id}"}}"#
            ),
        )
        .await;
        read_json_event(&mut socket, "response.continuation_rejected").await;
        send_json(
            &mut socket,
            r#"{"type":"response.create","model":"mock-model","input":[]}"#,
        )
        .await;
        read_json_event(&mut socket, "response.completed").await;
        socket
            .send(ClientMessage::Close(None))
            .await
            .expect("close reused connection");

        let captures = server.wait_for_captures(1).await;
        let value = serde_json::to_value(&captures[0]).expect("capture serializes");
        let requests = value["events"]
            .as_array()
            .expect("capture events are an array")
            .iter()
            .filter(|event| event["direction"] == "in" && event["event_type"] == "response.create")
            .collect::<Vec<_>>();
        assert_eq!(
            requests
                .iter()
                .map(|event| event["turn"].as_u64())
                .collect::<Vec<_>>(),
            vec![Some(1), Some(2), Some(3)]
        );
        let operation_ids = requests
            .iter()
            .filter_map(|event| event["operation_id"].as_str())
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(operation_ids.len(), 3);
    }

    #[tokio::test]
    async fn wire_close_boundaries_finalize_request_accounting() {
        for (scenario, status, terminal, close) in [
            (
                WebSocketScenario::CloseBeforeTerminal,
                "500",
                None,
                "clean_server_close",
            ),
            (
                WebSocketScenario::DirtyCloseAfterTerminal,
                "200",
                Some("response.completed"),
                "dirty_transport_drop",
            ),
        ] {
            let server = TestServer::start(MockServerConfig {
                websocket_mode: crate::config::WebSocketMode::TurnSerialized,
                websocket_scenario: scenario,
                websocket_first_content_delay_ms: 0.0,
                websocket_content_interval_ms: 0.0,
                no_tokenizer: true,
                ..MockServerConfig::default()
            })
            .await;
            let mut socket = server.connect("/mock/websocket/turns").await;
            send_json(
                &mut socket,
                r#"{"type":"response.create","model":"mock-model","input":[]}"#,
            )
            .await;
            if let Some(terminal) = terminal {
                read_json_event(&mut socket, terminal).await;
            } else {
                while let Some(message) = socket.next().await {
                    match message {
                        Ok(ClientMessage::Close(_)) | Err(_) => break,
                        _ => {}
                    }
                }
            }
            let endpoint = RouteKind::Turns.endpoint_label();
            assert_eq!(
                server
                    .state
                    .recorder
                    .metrics
                    .aiperf
                    .REQUESTS_TOTAL
                    .with_label_values(&[endpoint, "POST", status])
                    .get(),
                1
            );
            assert_eq!(
                server
                    .state
                    .recorder
                    .metrics
                    .aiperf
                    .REQUESTS_IN_PROGRESS
                    .with_label_values(&[endpoint])
                    .get(),
                0
            );
            let captures = server.wait_for_captures(1).await;
            let value = serde_json::to_value(&captures[0]).expect("capture serializes");
            assert_eq!(value["close"], close);
        }
    }

    #[tokio::test]
    async fn wire_enforces_message_limit_and_delivers_configured_control() {
        let limited = TestServer::start(MockServerConfig {
            websocket_mode: crate::config::WebSocketMode::TurnSerialized,
            websocket_max_message_bytes: 64,
            no_tokenizer: true,
            ..MockServerConfig::default()
        })
        .await;
        let mut oversized = limited.connect("/mock/websocket/turns").await;
        oversized
            .send(ClientMessage::Text("x".repeat(65).into()))
            .await
            .expect("send oversized message");
        while let Some(message) = oversized.next().await {
            match message {
                Ok(ClientMessage::Close(_)) | Err(_) => break,
                _ => {}
            }
        }
        let captures = limited.wait_for_captures(1).await;
        let value = serde_json::to_value(&captures[0]).expect("capture serializes");
        assert_eq!(value["close"], "receive_error");

        let controlled = TestServer::start(MockServerConfig {
            websocket_mode: crate::config::WebSocketMode::TurnSerialized,
            websocket_control_before_content: WebSocketControl::Ping,
            websocket_first_content_delay_ms: 0.0,
            websocket_content_interval_ms: 0.0,
            no_tokenizer: true,
            ..MockServerConfig::default()
        })
        .await;
        let mut socket = controlled.connect("/mock/websocket/turns").await;
        send_json(
            &mut socket,
            r#"{"type":"response.create","model":"mock-model","input":[]}"#,
        )
        .await;
        let mut has_ping = false;
        while let Some(message) = socket.next().await {
            match message.expect("read controlled response") {
                ClientMessage::Ping(_) => has_ping = true,
                ClientMessage::Text(payload) => {
                    let event: Value = serde_json::from_str(&payload).expect("mock output is JSON");
                    if event["type"] == "response.completed" {
                        break;
                    }
                }
                _ => {}
            }
        }
        assert!(has_ping);
    }

    #[test]
    fn captures_evict_the_oldest_completed_connection() {
        let store = WebSocketCaptureStore::new(1);
        store.push(WebSocketCapture::new(
            1,
            RouteKind::Turns,
            WebSocketScenario::Normal,
            0,
        ));
        store.push(WebSocketCapture::new(
            2,
            RouteKind::Turns,
            WebSocketScenario::Normal,
            0,
        ));
        let entries = store.snapshot();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].connection_id, 2);
    }

    #[test]
    fn capture_event_exposes_digest_and_length_without_payload() {
        let capture = capture_event(
            "in",
            "text",
            br#"{"type":"response.create","secret":"x"}"#,
            20,
            10,
        );
        let value = serde_json::to_value(capture).expect("capture serializes");
        assert_eq!(value["bytes"], 39);
        assert_eq!(value["event_type"], "response.create");
        assert!(value["payload_digest"].as_str().is_some());
        assert!(value.get("payload").is_none());
    }

    #[test]
    fn capture_event_canonicalizes_unknown_types() {
        let capture = capture_event(
            "in",
            "text",
            br#"{"type":"secret-controlled-value"}"#,
            20,
            10,
        );
        let value = serde_json::to_value(capture).expect("capture serializes");
        assert_eq!(value["event_type"], "unknown");
    }

    #[test]
    fn one_connection_capture_has_a_hard_event_metadata_bound() {
        let mut capture = WebSocketCapture::new(1, RouteKind::Turns, WebSocketScenario::Normal, 0);
        for index in 0..=MAX_CAPTURE_EVENTS_PER_CONNECTION {
            capture.push_event(capture_event(
                "in",
                "text",
                format!(r#"{{"type":"event-{index}"}}"#).as_bytes(),
                index as i64,
                0,
            ));
        }
        assert_eq!(capture.events.len(), MAX_CAPTURE_EVENTS_PER_CONNECTION);
        assert_eq!(capture.dropped_events, 1);
    }

    #[test]
    fn dropped_operation_accounting_releases_inflight_once() {
        let recorder = crate::metrics::MetricRecorder::new();
        let endpoint = RouteKind::Turns.endpoint_label();
        {
            let _operation = OperationAccounting::begin(
                &recorder,
                RouteKind::Turns,
                "mock-model".to_owned(),
                10,
                39,
            );
            assert_eq!(
                recorder
                    .metrics
                    .aiperf
                    .REQUESTS_IN_PROGRESS
                    .with_label_values(&[endpoint])
                    .get(),
                1
            );
        }
        assert_eq!(
            recorder
                .metrics
                .aiperf
                .REQUESTS_IN_PROGRESS
                .with_label_values(&[endpoint])
                .get(),
            0
        );
        assert_eq!(
            recorder
                .metrics
                .aiperf
                .REQUESTS_TOTAL
                .with_label_values(&[endpoint, "POST", "500"])
                .get(),
            1
        );
    }

    #[tokio::test]
    async fn turns_route_upgrades_and_emits_a_completed_response() {
        use futures::SinkExt;
        use tokio_tungstenite::connect_async;
        use tokio_tungstenite::tungstenite::Message as ClientMessage;

        let config = MockServerConfig {
            websocket_mode: crate::config::WebSocketMode::TurnSerialized,
            no_tokenizer: true,
            ..MockServerConfig::default()
        };
        let state = crate::state::AppState::build(config);
        let router = crate::app::build_router(state.clone());
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind test listener");
        let address = listener.local_addr().expect("test listener address");
        let server = tokio::spawn(async move {
            let _ = axum::serve(listener, router).await;
        });

        let url = format!("ws://{address}/mock/websocket/turns");
        let (mut socket, _) = connect_async(url).await.expect("upgrade turns route");
        socket
            .send(ClientMessage::Text(
                r#"{"type":"response.create","model":"mock-model","input":[]}"#.into(),
            ))
            .await
            .expect("send turn request");

        let mut event_types = Vec::new();
        while let Some(message) = socket.next().await {
            let message = message.expect("read mock response");
            let ClientMessage::Text(payload) = message else {
                continue;
            };
            let event: Value = serde_json::from_str(&payload).expect("mock output is JSON");
            let event_type = event["type"]
                .as_str()
                .expect("mock output has an event type")
                .to_owned();
            event_types.push(event_type.clone());
            if event_type == "response.completed" {
                break;
            }
        }
        assert_eq!(
            event_types,
            vec![
                "response.created".to_owned(),
                "response.output_text.delta".to_owned(),
                "response.completed".to_owned(),
            ]
        );
        let endpoint = RouteKind::Turns.endpoint_label();
        assert_eq!(
            state
                .recorder
                .metrics
                .aiperf
                .REQUESTS_TOTAL
                .with_label_values(&[endpoint, "POST", "200"])
                .get(),
            1
        );
        assert_eq!(
            state
                .recorder
                .metrics
                .aiperf
                .REQUESTS_IN_PROGRESS
                .with_label_values(&[endpoint])
                .get(),
            0
        );
        socket
            .send(ClientMessage::Close(None))
            .await
            .expect("close turns connection");
        for _ in 0..100 {
            if !state.websocket_captures.snapshot().is_empty() {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(1)).await;
        }
        assert_eq!(state.websocket_captures.snapshot().len(), 1);
        server.abort();
    }
}
