// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Worker-local WebSocket connection-driver primitives.

use std::cell::RefCell;
use std::collections::VecDeque;
use std::fmt::{self, Display, Formatter};
use std::rc::Rc;

use futures::stream::{SplitSink, SplitStream};
use futures::{Sink, SinkExt, Stream, StreamExt};
use tokio::sync::{Notify, mpsc};
use tokio::task::JoinHandle;
use tokio_tungstenite::tungstenite::{
    Error as TungsteniteError, Message,
    protocol::frame::{
        Frame,
        coding::{Data, OpCode},
    },
};

use crate::body_plan::{
    PreparedWsMessage, PreparedWsMessageRole, PreparedWsOpcode, PreparedWsOperation,
};
use crate::clock::Clock;

const CONTROL_QUEUE_CAPACITY: usize = 4;
const PEER_CLOSE_HANDSHAKE_NS: i64 = 1_000_000_000;

/// Count and byte bounds for one socket's outbound application queue.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ApplicationQueueLimits {
    max_commands: usize,
    max_bytes: usize,
    max_frame_bytes: usize,
}

impl ApplicationQueueLimits {
    /// Create queue limits already validated by configuration loading.
    pub(crate) const fn new(max_commands: usize, max_bytes: usize) -> Self {
        Self {
            max_commands,
            max_bytes,
            max_frame_bytes: max_bytes,
        }
    }

    /// Bound each data fragment so a control frame retains writer capacity.
    pub(crate) const fn with_max_frame_bytes(mut self, max_frame_bytes: usize) -> Self {
        self.max_frame_bytes = max_frame_bytes;
        self
    }
}

/// Stable reason an equivalent HTTP/SSE operation replaced WebSocket before send.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum FallbackReason {
    /// DNS, TCP, or pre-application handshake I/O failed.
    NetworkConnect,
    /// The peer explicitly does not expose the requested WebSocket upgrade.
    UnsupportedUpgrade,
}

impl FallbackReason {
    /// Stable artifact value.
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::NetworkConnect => "network_connect",
            Self::UnsupportedUpgrade => "unsupported_upgrade",
        }
    }
}

/// Classify only failures eligible for a dialect-declared pre-send fallback.
pub(crate) fn classify_upgrade_failure(error: &TungsteniteError) -> Option<FallbackReason> {
    match error {
        TungsteniteError::Io(_) => Some(FallbackReason::NetworkConnect),
        TungsteniteError::Http(response) if matches!(response.status().as_u16(), 426 | 501) => {
            Some(FallbackReason::UnsupportedUpgrade)
        }
        _ => None,
    }
}

/// Validate the complete immutable operation before its first message is queued.
pub(crate) fn validate_application_queue(
    operation: &PreparedWsOperation,
    limits: ApplicationQueueLimits,
) -> Result<usize, WsDriverError> {
    if operation.messages().len() > limits.max_commands {
        return Err(WsDriverError::QueueCommandLimit {
            actual: operation.messages().len(),
            limit: limits.max_commands,
        });
    }
    let bytes = operation
        .messages()
        .iter()
        .try_fold(0_usize, |total, message| {
            total.checked_add(message.payload().len())
        });
    match bytes {
        Some(actual) if actual <= limits.max_bytes => Ok(actual),
        Some(actual) => Err(WsDriverError::QueueByteLimit {
            actual,
            limit: limits.max_bytes,
        }),
        None => Err(WsDriverError::QueueByteOverflow),
    }
}

/// Clock-driven operation limits applied by the split socket driver.
#[derive(Clone, Copy, Debug)]
pub(crate) struct DriverTiming {
    /// Absolute end-to-end operation deadline.
    pub(crate) deadline_ns: Option<i64>,
    /// Absolute age-based connection rotation boundary.
    pub(crate) rotation_deadline_ns: i64,
    /// Interval between keepalive pings.
    pub(crate) ping_interval_ns: i64,
    /// Maximum silence between complete application messages.
    pub(crate) stream_idle_timeout_ns: i64,
    /// Relative cancellation delay armed after measured input is flushed.
    pub(crate) cancel_after_ns: Option<i64>,
}

/// One event produced by the socket driver.
#[derive(Debug)]
pub(crate) enum DriverEvent {
    /// An application message was flushed successfully.
    Flushed {
        /// Logical role assigned before dispatch.
        role: PreparedWsMessageRole,
        /// Timestamp sampled immediately after `SinkExt::send` completed.
        timestamp_ns: i64,
    },
    /// A complete reassembled application message was received.
    Application {
        /// Message payload.
        payload: bytes::Bytes,
        /// Whether the WebSocket opcode was text.
        is_text: bool,
        /// Clock timestamp after the complete message was yielded.
        timestamp_ns: i64,
    },
}

/// Explicit driver failures suitable for a library hot path.
#[derive(Debug)]
pub(crate) enum WsDriverError {
    QueueCommandLimit { actual: usize, limit: usize },
    QueueByteLimit { actual: usize, limit: usize },
    QueueByteOverflow,
    MissingMeasuredInput,
    InvalidTextMessage,
    ApplicationQueueClosed,
    UnexpectedMeasuredInputFlush,
    ControlQueueFull,
    WriterStopped,
    Write(TungsteniteError),
    Read(TungsteniteError),
    PeerClosed,
    ResponseByteLimit { actual: usize, limit: usize },
    ResponseByteOverflow,
    StreamIdleTimeout,
    OperationDeadline,
    RequestCancellation,
    ConnectionRotation,
    CloseHandshakeTimeout,
    Reunite,
}

impl Display for WsDriverError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::QueueCommandLimit { actual, limit } => write!(
                formatter,
                "websocket operation queues {actual} commands; limit is {limit}"
            ),
            Self::QueueByteLimit { actual, limit } => write!(
                formatter,
                "websocket operation queues {actual} payload bytes; limit is {limit}"
            ),
            Self::QueueByteOverflow => {
                formatter.write_str("websocket application queue byte count overflowed")
            }
            Self::MissingMeasuredInput => {
                formatter.write_str("websocket operation has no measured input message")
            }
            Self::InvalidTextMessage => {
                formatter.write_str("websocket text application message is not UTF-8")
            }
            Self::ApplicationQueueClosed => {
                formatter.write_str("websocket application writer stopped before enqueue")
            }
            Self::UnexpectedMeasuredInputFlush => formatter.write_str(
                "websocket writer reported more measured-input flushes than were queued",
            ),
            Self::ControlQueueFull => formatter.write_str("websocket control queue is full"),
            Self::WriterStopped => {
                formatter.write_str("websocket writer stopped before returning its socket half")
            }
            Self::Write(error) => write!(formatter, "websocket write failed: {error}"),
            Self::Read(error) => write!(formatter, "websocket read failed: {error}"),
            Self::PeerClosed => {
                formatter.write_str("websocket peer closed before a terminal event")
            }
            Self::ResponseByteLimit { actual, limit } => write!(
                formatter,
                "websocket response consumed {actual} bytes; limit is {limit}"
            ),
            Self::ResponseByteOverflow => {
                formatter.write_str("websocket response byte count overflowed")
            }
            Self::StreamIdleTimeout => {
                formatter.write_str("websocket response stream reached its idle timeout")
            }
            Self::OperationDeadline => {
                formatter.write_str("websocket operation reached its deadline")
            }
            Self::RequestCancellation => formatter.write_str("websocket request was cancelled"),
            Self::ConnectionRotation => {
                formatter.write_str("websocket connection reached its rotation boundary")
            }
            Self::CloseHandshakeTimeout => {
                formatter.write_str("websocket reciprocal close handshake reached its deadline")
            }
            Self::Reunite => formatter.write_str("websocket split halves could not be reunited"),
        }
    }
}

impl std::error::Error for WsDriverError {}

enum ControlCommand {
    Send(Message),
    FlushPeerClose,
    Finish,
}

#[derive(Default)]
struct ControlProgress {
    needs_finish: bool,
    has_flushed_peer_close: bool,
}

impl ControlProgress {
    fn merge(&mut self, other: Self) {
        self.needs_finish |= other.needs_finish;
        self.has_flushed_peer_close |= other.has_flushed_peer_close;
    }
}

enum WriterNotice<S>
where
    S: Sink<Message, Error = TungsteniteError> + Unpin,
{
    Flushed {
        role: PreparedWsMessageRole,
        timestamp_ns: i64,
    },
    PeerCloseFlushed,
    Finished(SplitSink<S, Message>),
    Failed(WsDriverError),
}

/// One split, turn-serialized socket operation.
///
/// The writer half lives on a worker-local task. The owner continues polling the
/// read half, deadlines, and control traffic while a write is backpressured.
pub(crate) struct SocketOperationDriver<S>
where
    S: Stream<Item = Result<Message, TungsteniteError>>
        + Sink<Message, Error = TungsteniteError>
        + Unpin
        + 'static,
{
    clock: Rc<dyn Clock>,
    read: Option<SplitStream<S>>,
    application_tx: Option<mpsc::Sender<PreparedWsMessage>>,
    control_tx: mpsc::Sender<ControlCommand>,
    notices: Rc<RefCell<VecDeque<WriterNotice<S>>>>,
    notice: Rc<Notify>,
    writer: JoinHandle<()>,
    pending: VecDeque<DriverEvent>,
    timing: DriverTiming,
    next_ping_ns: i64,
    remaining_measured_inputs: usize,
    has_input_completed: bool,
    cancellation_deadline_ns: Option<i64>,
    last_application_receive_ns: i64,
    response_bytes: usize,
    max_response_bytes: usize,
    finished_sink: Option<SplitSink<S, Message>>,
    peer_close_deadline_ns: Option<i64>,
    has_flushed_peer_close: bool,
}

impl<S> SocketOperationDriver<S>
where
    S: Stream<Item = Result<Message, TungsteniteError>>
        + Sink<Message, Error = TungsteniteError>
        + Unpin
        + 'static,
{
    /// Split a socket, start its independent writer, and enqueue one operation.
    pub(crate) fn start(
        socket: S,
        clock: Rc<dyn Clock>,
        operation: &PreparedWsOperation,
        queue_limits: ApplicationQueueLimits,
        timing: DriverTiming,
        max_response_bytes: usize,
    ) -> Result<Self, WsDriverError> {
        validate_application_queue(operation, queue_limits)?;
        let remaining_measured_inputs = operation
            .messages()
            .iter()
            .filter(|message| message.role() == PreparedWsMessageRole::MeasuredInput)
            .count();
        if remaining_measured_inputs == 0 {
            return Err(WsDriverError::MissingMeasuredInput);
        }
        let (sink, read) = socket.split();
        let (application_tx, application_rx) = mpsc::channel(queue_limits.max_commands);
        let (control_tx, control_rx) = mpsc::channel(CONTROL_QUEUE_CAPACITY);
        let notices = Rc::new(RefCell::new(VecDeque::with_capacity(
            operation.messages().len() + 1,
        )));
        let notice = Rc::new(Notify::new());
        let writer = tokio::task::spawn_local(writer_loop(
            sink,
            application_rx,
            control_rx,
            clock.clone(),
            queue_limits.max_frame_bytes,
            notices.clone(),
            notice.clone(),
        ));
        for message in operation.messages() {
            application_tx
                .try_send(message.clone())
                .map_err(|_| WsDriverError::ApplicationQueueClosed)?;
        }
        let now_ns = clock.now_ns();
        Ok(Self {
            clock,
            read: Some(read),
            application_tx: Some(application_tx),
            control_tx,
            notices,
            notice,
            writer,
            pending: VecDeque::with_capacity(operation.messages().len()),
            timing,
            next_ping_ns: now_ns.saturating_add(timing.ping_interval_ns),
            remaining_measured_inputs,
            has_input_completed: false,
            cancellation_deadline_ns: None,
            last_application_receive_ns: now_ns,
            response_bytes: 0,
            max_response_bytes,
            finished_sink: None,
            peer_close_deadline_ns: None,
            has_flushed_peer_close: false,
        })
    }

    /// Progress reads, successful flush notifications, keepalive, and deadlines.
    pub(crate) async fn next(&mut self) -> Result<DriverEvent, WsDriverError> {
        loop {
            self.drain_notices()?;
            if self.has_flushed_peer_close {
                return Err(WsDriverError::PeerClosed);
            }
            if let Some(event) = self.pending.pop_front() {
                return Ok(event);
            }
            let now_ns = self.clock.now_ns();
            if let Some(error) = self.boundary_error(now_ns) {
                return Err(error);
            }
            let can_read_or_ping = self.peer_close_deadline_ns.is_none();
            let read = self.read.as_mut().ok_or(WsDriverError::WriterStopped)?;
            let notice = self.notice.notified();
            let ping = self
                .clock
                .clone()
                .sleep(self.next_ping_ns.saturating_sub(now_ns));
            let idle = deadline_sleep(
                self.clock.clone(),
                self.has_input_completed.then_some(
                    self.last_application_receive_ns
                        .saturating_add(self.timing.stream_idle_timeout_ns),
                ),
                now_ns,
            );
            let deadline = deadline_sleep(self.clock.clone(), self.timing.deadline_ns, now_ns);
            let cancellation =
                deadline_sleep(self.clock.clone(), self.cancellation_deadline_ns, now_ns);
            let rotation = self
                .clock
                .clone()
                .sleep(self.timing.rotation_deadline_ns.saturating_sub(now_ns));
            let peer_close =
                deadline_sleep(self.clock.clone(), self.peer_close_deadline_ns, now_ns);
            tokio::select! {
                biased;
                () = deadline => return Err(WsDriverError::OperationDeadline),
                () = cancellation => return Err(WsDriverError::RequestCancellation),
                () = rotation => return Err(WsDriverError::ConnectionRotation),
                () = idle => return Err(WsDriverError::StreamIdleTimeout),
                () = peer_close => return Err(WsDriverError::CloseHandshakeTimeout),
                () = notice => {}
                message = read.next(), if can_read_or_ping => {
                    if let Some(error) = self.boundary_error(self.clock.now_ns()) {
                        return Err(error);
                    }
                    self.on_message(message)?;
                }
                () = ping, if can_read_or_ping => {
                    self.control_tx
                        .try_send(ControlCommand::Send(Message::Ping(bytes::Bytes::new())))
                        .map_err(|_| WsDriverError::ControlQueueFull)?;
                    self.next_ping_ns = self.clock.now_ns().saturating_add(self.timing.ping_interval_ns);
                }
            }
        }
    }

    fn boundary_error(&self, now_ns: i64) -> Option<WsDriverError> {
        if self
            .timing
            .deadline_ns
            .is_some_and(|deadline_ns| now_ns >= deadline_ns)
        {
            return Some(WsDriverError::OperationDeadline);
        }
        if self
            .cancellation_deadline_ns
            .is_some_and(|deadline_ns| now_ns >= deadline_ns)
        {
            return Some(WsDriverError::RequestCancellation);
        }
        if now_ns >= self.timing.rotation_deadline_ns {
            return Some(WsDriverError::ConnectionRotation);
        }
        if self
            .peer_close_deadline_ns
            .is_some_and(|deadline_ns| now_ns >= deadline_ns)
        {
            return Some(WsDriverError::CloseHandshakeTimeout);
        }
        if self.has_input_completed
            && now_ns
                >= self
                    .last_application_receive_ns
                    .saturating_add(self.timing.stream_idle_timeout_ns)
        {
            return Some(WsDriverError::StreamIdleTimeout);
        }
        None
    }

    fn on_message(
        &mut self,
        message: Option<Result<Message, TungsteniteError>>,
    ) -> Result<(), WsDriverError> {
        let message = message
            .ok_or(WsDriverError::PeerClosed)?
            .map_err(WsDriverError::Read)?;
        match message {
            Message::Text(text) => {
                self.on_application(bytes::Bytes::copy_from_slice(text.as_bytes()), true)
            }
            Message::Binary(bytes) => self.on_application(bytes, false),
            Message::Ping(payload) => self
                .control_tx
                .try_send(ControlCommand::Send(Message::Pong(payload)))
                .map_err(|_| WsDriverError::ControlQueueFull),
            Message::Pong(_) => Ok(()),
            Message::Close(_) => {
                self.control_tx
                    .try_send(ControlCommand::FlushPeerClose)
                    .map_err(|_| WsDriverError::ControlQueueFull)?;
                self.peer_close_deadline_ns =
                    Some(self.clock.now_ns().saturating_add(PEER_CLOSE_HANDSHAKE_NS));
                Ok(())
            }
            Message::Frame(_) => Ok(()),
        }
    }

    fn on_application(
        &mut self,
        payload: bytes::Bytes,
        is_text: bool,
    ) -> Result<(), WsDriverError> {
        self.response_bytes = self
            .response_bytes
            .checked_add(payload.len())
            .ok_or(WsDriverError::ResponseByteOverflow)?;
        if self.response_bytes > self.max_response_bytes {
            return Err(WsDriverError::ResponseByteLimit {
                actual: self.response_bytes,
                limit: self.max_response_bytes,
            });
        }
        let timestamp_ns = self.clock.now_ns();
        self.last_application_receive_ns = timestamp_ns;
        self.drain_notices()?;
        self.pending.push_back(DriverEvent::Application {
            payload,
            is_text,
            timestamp_ns,
        });
        Ok(())
    }

    fn drain_notices(&mut self) -> Result<(), WsDriverError> {
        while let Some(notice) = self.notices.borrow_mut().pop_front() {
            match notice {
                WriterNotice::Flushed { role, timestamp_ns } => {
                    if role == PreparedWsMessageRole::MeasuredInput {
                        self.remaining_measured_inputs = self
                            .remaining_measured_inputs
                            .checked_sub(1)
                            .ok_or(WsDriverError::UnexpectedMeasuredInputFlush)?;
                        if self.remaining_measured_inputs == 0 {
                            self.has_input_completed = true;
                            self.last_application_receive_ns = timestamp_ns;
                            self.cancellation_deadline_ns = self
                                .timing
                                .cancel_after_ns
                                .map(|delay_ns| timestamp_ns.saturating_add(delay_ns.max(0)));
                        }
                    }
                    self.pending
                        .push_back(DriverEvent::Flushed { role, timestamp_ns });
                }
                WriterNotice::PeerCloseFlushed => self.has_flushed_peer_close = true,
                WriterNotice::Finished(sink) => self.finished_sink = Some(sink),
                WriterNotice::Failed(error) => return Err(error),
            }
        }
        Ok(())
    }

    /// Stop the writer after all queued application commands and reunite halves.
    pub(crate) async fn finish(mut self) -> Result<(S, Vec<DriverEvent>), WsDriverError> {
        self.application_tx.take();
        self.control_tx
            .try_send(ControlCommand::Finish)
            .map_err(|_| WsDriverError::ControlQueueFull)?;
        loop {
            self.drain_notices()?;
            if let Some(sink) = self.finished_sink.take() {
                let read = self.read.take().ok_or(WsDriverError::WriterStopped)?;
                let socket = sink.reunite(read).map_err(|_| WsDriverError::Reunite)?;
                self.writer.abort();
                return Ok((socket, self.pending.drain(..).collect()));
            }
            let now_ns = self.clock.now_ns();
            if let Some(error) = self.boundary_error(now_ns) {
                return Err(error);
            }
            tokio::select! {
                biased;
                () = deadline_sleep(self.clock.clone(), self.timing.deadline_ns, now_ns) => {
                    return Err(WsDriverError::OperationDeadline);
                }
                () = deadline_sleep(self.clock.clone(), self.cancellation_deadline_ns, now_ns) => {
                    return Err(WsDriverError::RequestCancellation);
                }
                () = self.clock.clone().sleep(self.timing.rotation_deadline_ns.saturating_sub(now_ns)) => {
                    return Err(WsDriverError::ConnectionRotation);
                }
                () = self.notice.notified() => {}
            }
        }
    }
}

impl<S> Drop for SocketOperationDriver<S>
where
    S: Stream<Item = Result<Message, TungsteniteError>>
        + Sink<Message, Error = TungsteniteError>
        + Unpin
        + 'static,
{
    fn drop(&mut self) {
        // Abort is nonblocking and does not share capacity with either data queue.
        // Dropping the read half then tears down the socket if the operation did
        // not explicitly reunite it.
        self.writer.abort();
    }
}

async fn writer_loop<S>(
    mut sink: SplitSink<S, Message>,
    mut application_rx: mpsc::Receiver<PreparedWsMessage>,
    mut control_rx: mpsc::Receiver<ControlCommand>,
    clock: Rc<dyn Clock>,
    max_frame_bytes: usize,
    notices: Rc<RefCell<VecDeque<WriterNotice<S>>>>,
    notice: Rc<Notify>,
) where
    S: Stream<Item = Result<Message, TungsteniteError>>
        + Sink<Message, Error = TungsteniteError>
        + Unpin
        + 'static,
{
    let mut needs_finish = false;
    loop {
        if needs_finish && application_rx.is_closed() && application_rx.is_empty() {
            if let Err(error) = sink.flush().await {
                push_notice(
                    &notices,
                    &notice,
                    WriterNotice::Failed(WsDriverError::Write(error)),
                );
            } else {
                push_notice(&notices, &notice, WriterNotice::Finished(sink));
            }
            return;
        }
        let result = tokio::select! {
            biased;
            command = control_rx.recv() => match command {
                Some(ControlCommand::Send(message)) => sink.send(message).await.map(|_| None).map_err(WsDriverError::Write),
                Some(ControlCommand::FlushPeerClose) => {
                    match sink.flush().await {
                        Ok(()) => {
                            push_notice(&notices, &notice, WriterNotice::PeerCloseFlushed);
                            Ok(None)
                        }
                        Err(error) => Err(WsDriverError::Write(error)),
                    }
                }
                Some(ControlCommand::Finish) => {
                    needs_finish = true;
                    Ok(None)
                }
                None => return,
            },
            message = application_rx.recv() => match message {
                Some(message) => send_application(
                    &mut sink,
                    message,
                    &mut control_rx,
                    &clock,
                    max_frame_bytes,
                )
                .await
                .map(Some),
                None => {
                    needs_finish = true;
                    Ok(None)
                }
            }
        };
        match result {
            Ok(Some((role, timestamp_ns, progress))) => {
                needs_finish |= progress.needs_finish;
                push_notice(
                    &notices,
                    &notice,
                    WriterNotice::Flushed { role, timestamp_ns },
                );
                if progress.has_flushed_peer_close {
                    push_notice(&notices, &notice, WriterNotice::PeerCloseFlushed);
                }
            }
            Ok(None) => {}
            Err(error) => {
                push_notice(&notices, &notice, WriterNotice::Failed(error));
                return;
            }
        }
    }
}

async fn send_application<S>(
    sink: &mut SplitSink<S, Message>,
    message: PreparedWsMessage,
    control_rx: &mut mpsc::Receiver<ControlCommand>,
    clock: &Rc<dyn Clock>,
    max_frame_bytes: usize,
) -> Result<(PreparedWsMessageRole, i64, ControlProgress), WsDriverError>
where
    S: Sink<Message, Error = TungsteniteError> + Unpin,
{
    let role = message.role();
    if message.opcode() == PreparedWsOpcode::Text {
        std::str::from_utf8(message.payload()).map_err(|_| WsDriverError::InvalidTextMessage)?;
    }
    let mut progress = ControlProgress::default();
    if message.payload().is_empty() {
        let data = match message.opcode() {
            PreparedWsOpcode::Text => Data::Text,
            PreparedWsOpcode::Binary => Data::Binary,
        };
        sink.feed(Message::Frame(Frame::message(
            bytes::Bytes::new(),
            OpCode::Data(data),
            true,
        )))
        .await
        .map_err(WsDriverError::Write)?;
        progress.merge(flush_servicing_controls(sink, control_rx).await?);
        return Ok((role, clock.now_ns(), progress));
    }
    let payload = message.payload().clone();
    let mut offset = 0_usize;
    let mut is_first = true;
    while offset < payload.len() {
        let end = offset
            .saturating_add(max_frame_bytes.max(1))
            .min(payload.len());
        let data = if is_first {
            match message.opcode() {
                PreparedWsOpcode::Text => Data::Text,
                PreparedWsOpcode::Binary => Data::Binary,
            }
        } else {
            Data::Continue
        };
        is_first = false;
        sink.feed(Message::Frame(Frame::message(
            payload.slice(offset..end),
            OpCode::Data(data),
            end == payload.len(),
        )))
        .await
        .map_err(WsDriverError::Write)?;
        progress.merge(flush_servicing_controls(sink, control_rx).await?);
        offset = end;
    }
    Ok((role, clock.now_ns(), progress))
}

async fn flush_servicing_controls<S>(
    sink: &mut SplitSink<S, Message>,
    control_rx: &mut mpsc::Receiver<ControlCommand>,
) -> Result<ControlProgress, WsDriverError>
where
    S: Sink<Message, Error = TungsteniteError> + Unpin,
{
    let mut progress = ControlProgress::default();
    loop {
        tokio::select! {
            biased;
            command = control_rx.recv() => match command {
                Some(ControlCommand::Send(control)) => {
                    sink.feed(control).await.map_err(WsDriverError::Write)?;
                }
                Some(ControlCommand::FlushPeerClose) => {
                    sink.flush().await.map_err(WsDriverError::Write)?;
                    progress.has_flushed_peer_close = true;
                }
                Some(ControlCommand::Finish) => progress.needs_finish = true,
                None => return Err(WsDriverError::WriterStopped),
            },
            result = sink.flush() => {
                result.map_err(WsDriverError::Write)?;
                return Ok(progress);
            }
        }
    }
}

fn push_notice<S>(
    notices: &Rc<RefCell<VecDeque<WriterNotice<S>>>>,
    notify: &Notify,
    value: WriterNotice<S>,
) where
    S: Sink<Message, Error = TungsteniteError> + Unpin,
{
    notices.borrow_mut().push_back(value);
    notify.notify_one();
}

async fn deadline_sleep(clock: Rc<dyn Clock>, deadline_ns: Option<i64>, now_ns: i64) {
    match deadline_ns {
        Some(deadline_ns) => clock.sleep(deadline_ns.saturating_sub(now_ns)).await,
        None => std::future::pending().await,
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::collections::BTreeMap;
    use std::pin::Pin;
    use std::rc::Rc;
    use std::task::{Context, Poll, Waker};
    use std::time::Duration;

    use futures::{Sink, SinkExt, Stream, StreamExt};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;
    use tokio::sync::{oneshot, watch};
    use tokio_tungstenite::accept_async;
    use tokio_tungstenite::tungstenite::{Message, protocol::WebSocketConfig};

    use super::{
        ApplicationQueueLimits, DriverEvent, DriverTiming, SocketOperationDriver, WsDriverError,
    };
    use crate::body_plan::{PreparedWsMessage, PreparedWsMessageRole, PreparedWsOperation};
    use crate::clock::{Clock, RealClock};
    use crate::transport::http::config::ClientConfig;
    use crate::transport::ws::connector;

    #[derive(Default)]
    struct BackpressureState {
        has_application_frame: bool,
        has_ping: bool,
        has_pong: bool,
        reader: Option<Waker>,
        writer: Option<Waker>,
    }

    struct BackpressureSocket {
        state: Rc<RefCell<BackpressureState>>,
    }

    #[derive(Default)]
    struct SequencedWriteState {
        application_frames: usize,
        is_second_ready: bool,
        writer: Option<Waker>,
    }

    struct SequencedWriteSocket {
        state: Rc<RefCell<SequencedWriteState>>,
    }

    struct PausedUploadProxy {
        address: std::net::SocketAddr,
        is_paused: watch::Sender<bool>,
        has_upload_data: watch::Sender<bool>,
        task: tokio::task::JoinHandle<()>,
    }

    impl PausedUploadProxy {
        async fn start(target: std::net::SocketAddr) -> Self {
            let listener = TcpListener::bind("127.0.0.1:0")
                .await
                .expect("bind upload proxy");
            let address = listener.local_addr().expect("upload proxy address");
            let (is_paused, mut pause_rx) = watch::channel(false);
            let (has_upload_data, _) = watch::channel(false);
            let upload_data_tx = has_upload_data.clone();
            let task = tokio::task::spawn_local(async move {
                let (client, _) = listener.accept().await.expect("accept proxy client");
                let upstream = tokio::net::TcpStream::connect(target)
                    .await
                    .expect("connect proxy upstream");
                let (mut client_read, mut client_write) = client.into_split();
                let (mut upstream_read, mut upstream_write) = upstream.into_split();
                let download = tokio::task::spawn_local(async move {
                    let _ = tokio::io::copy(&mut upstream_read, &mut client_write).await;
                });
                let mut buffer = [0_u8; 64 * 1024];
                loop {
                    if *pause_rx.borrow() {
                        tokio::select! {
                            changed = pause_rx.changed() => {
                                if changed.is_err() {
                                    break;
                                }
                            }
                            ready = client_read.readable(), if !*upload_data_tx.borrow() => {
                                if ready.is_err() {
                                    break;
                                }
                                upload_data_tx.send_replace(true);
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
                        read = client_read.read(&mut buffer) => match read {
                            Ok(0) | Err(_) => break,
                            Ok(read) => {
                                if upstream_write.write_all(&buffer[..read]).await.is_err() {
                                    break;
                                }
                            }
                        }
                    }
                }
                download.abort();
            });
            Self {
                address,
                is_paused,
                has_upload_data,
                task,
            }
        }

        fn pause(&self) {
            self.is_paused.send_replace(true);
        }

        fn resume(&self) {
            self.is_paused.send_replace(false);
        }

        async fn wait_for_upload_data(&self) {
            let mut data_rx = self.has_upload_data.subscribe();
            while !*data_rx.borrow_and_update() {
                data_rx.changed().await.expect("upload proxy data signal");
            }
        }
    }

    impl Drop for PausedUploadProxy {
        fn drop(&mut self) {
            self.task.abort();
        }
    }

    impl Stream for SequencedWriteSocket {
        type Item = Result<Message, tokio_tungstenite::tungstenite::Error>;

        fn poll_next(self: Pin<&mut Self>, _context: &mut Context<'_>) -> Poll<Option<Self::Item>> {
            Poll::Pending
        }
    }

    impl Sink<Message> for SequencedWriteSocket {
        type Error = tokio_tungstenite::tungstenite::Error;

        fn poll_ready(
            self: Pin<&mut Self>,
            context: &mut Context<'_>,
        ) -> Poll<Result<(), Self::Error>> {
            let mut state = self.state.borrow_mut();
            if state.application_frames == 0 || state.is_second_ready {
                Poll::Ready(Ok(()))
            } else {
                state.writer = Some(context.waker().clone());
                Poll::Pending
            }
        }

        fn start_send(self: Pin<&mut Self>, message: Message) -> Result<(), Self::Error> {
            if matches!(message, Message::Frame(_)) {
                self.state.borrow_mut().application_frames += 1;
            }
            Ok(())
        }

        fn poll_flush(
            self: Pin<&mut Self>,
            _context: &mut Context<'_>,
        ) -> Poll<Result<(), Self::Error>> {
            Poll::Ready(Ok(()))
        }

        fn poll_close(
            self: Pin<&mut Self>,
            _context: &mut Context<'_>,
        ) -> Poll<Result<(), Self::Error>> {
            Poll::Ready(Ok(()))
        }
    }

    impl Stream for BackpressureSocket {
        type Item = Result<Message, tokio_tungstenite::tungstenite::Error>;

        fn poll_next(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Option<Self::Item>> {
            let mut state = self.state.borrow_mut();
            if state.has_application_frame && !state.has_ping {
                state.has_ping = true;
                return Poll::Ready(Some(Ok(Message::Ping(bytes::Bytes::from_static(
                    b"health",
                )))));
            }
            state.reader = Some(context.waker().clone());
            Poll::Pending
        }
    }

    impl Sink<Message> for BackpressureSocket {
        type Error = tokio_tungstenite::tungstenite::Error;

        fn poll_ready(
            self: Pin<&mut Self>,
            _context: &mut Context<'_>,
        ) -> Poll<Result<(), Self::Error>> {
            Poll::Ready(Ok(()))
        }

        fn start_send(self: Pin<&mut Self>, message: Message) -> Result<(), Self::Error> {
            let mut state = self.state.borrow_mut();
            match message {
                Message::Frame(_) => {
                    state.has_application_frame = true;
                    if let Some(reader) = state.reader.take() {
                        reader.wake();
                    }
                }
                Message::Pong(payload) => {
                    assert_eq!(payload, bytes::Bytes::from_static(b"health"));
                    state.has_pong = true;
                    if let Some(writer) = state.writer.take() {
                        writer.wake();
                    }
                }
                _ => {}
            }
            Ok(())
        }

        fn poll_flush(
            self: Pin<&mut Self>,
            context: &mut Context<'_>,
        ) -> Poll<Result<(), Self::Error>> {
            let mut state = self.state.borrow_mut();
            if state.has_pong {
                Poll::Ready(Ok(()))
            } else {
                state.writer = Some(context.waker().clone());
                Poll::Pending
            }
        }

        fn poll_close(
            self: Pin<&mut Self>,
            _context: &mut Context<'_>,
        ) -> Poll<Result<(), Self::Error>> {
            Poll::Ready(Ok(()))
        }
    }

    #[test]
    fn control_frames_progress_while_application_flush_is_backpressured() {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("test runtime builds");
        let local = tokio::task::LocalSet::new();
        local.block_on(&runtime, async {
            let state = Rc::new(RefCell::new(BackpressureState::default()));
            let socket = BackpressureSocket {
                state: state.clone(),
            };
            let operation = PreparedWsOperation::new(
                [PreparedWsMessage::text(
                    bytes::Bytes::from_static(b"data"),
                    PreparedWsMessageRole::MeasuredInput,
                )],
                None,
            );
            let clock: Rc<dyn Clock> = RealClock::new();
            let now_ns = clock.now_ns();
            let mut driver = SocketOperationDriver::start(
                socket,
                clock,
                &operation,
                ApplicationQueueLimits::new(1, 4).with_max_frame_bytes(2),
                DriverTiming {
                    deadline_ns: Some(now_ns.saturating_add(1_000_000_000)),
                    rotation_deadline_ns: now_ns.saturating_add(1_000_000_000),
                    ping_interval_ns: 1_000_000_000,
                    stream_idle_timeout_ns: 1_000_000_000,
                    cancel_after_ns: None,
                },
                64,
            )
            .expect("driver starts");

            let event = tokio::time::timeout(Duration::from_secs(1), driver.next())
                .await
                .expect("writer progresses under backpressure")
                .expect("writer stays healthy");
            assert!(matches!(event, DriverEvent::Flushed { .. }));
            assert!(state.borrow().has_pong, "peer ping must be answered");
            let _ = driver.finish().await.expect("driver reunites socket");
        });
    }

    #[test]
    fn peer_initiated_close_is_flushed_before_driver_reports_peer_closed() {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("test runtime builds");
        tokio::task::LocalSet::new().block_on(&runtime, async {
            let listener = TcpListener::bind("127.0.0.1:0")
                .await
                .expect("test listener binds");
            let address = listener.local_addr().expect("listener has an address");
            let (reply_seen, reply_result) = oneshot::channel();
            tokio::task::spawn_local(async move {
                let (stream, _) = listener.accept().await.expect("server accepts client");
                let mut socket = accept_async(stream).await.expect("server upgrades client");
                loop {
                    match socket.next().await {
                        Some(Ok(Message::Text(_))) => break,
                        Some(Ok(_)) => {}
                        Some(Err(error)) => panic!("server read failed: {error}"),
                        None => panic!("client closed before its application message"),
                    }
                }
                socket
                    .send(Message::Close(None))
                    .await
                    .expect("server sends Close");
                let saw_reply = matches!(socket.next().await, Some(Ok(Message::Close(_))));
                let _ = reply_seen.send(saw_reply);
            });

            let clock: Rc<dyn Clock> = RealClock::new();
            let url = url::Url::parse(&format!("ws://{address}/v1/responses"))
                .expect("test URL is valid");
            let socket = connector::connect(
                &url,
                &BTreeMap::new(),
                &ClientConfig::default(),
                WebSocketConfig::default(),
                clock.clone(),
                None,
            )
            .await
            .expect("client upgrades server");
            let operation = PreparedWsOperation::new(
                [PreparedWsMessage::text(
                    bytes::Bytes::from_static(b"request"),
                    PreparedWsMessageRole::MeasuredInput,
                )],
                None,
            );
            let now_ns = clock.now_ns();
            let mut driver = SocketOperationDriver::start(
                socket,
                clock,
                &operation,
                ApplicationQueueLimits::new(1, 64),
                DriverTiming {
                    deadline_ns: Some(now_ns.saturating_add(1_000_000_000)),
                    rotation_deadline_ns: now_ns.saturating_add(1_000_000_000),
                    ping_interval_ns: 1_000_000_000,
                    stream_idle_timeout_ns: 1_000_000_000,
                    cancel_after_ns: None,
                },
                64,
            )
            .expect("driver starts");
            let error = loop {
                match driver.next().await {
                    Ok(_) => {}
                    Err(error) => break error,
                }
            };

            assert!(matches!(error, WsDriverError::PeerClosed));
            assert!(
                tokio::time::timeout(Duration::from_secs(1), reply_result)
                    .await
                    .expect("server observes reciprocal Close")
                    .expect("client flushed reciprocal Close")
            );
        });
    }

    #[test]
    fn real_tcp_backpressure_allows_pong_before_remaining_max_sized_frames() {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("test runtime builds");
        let local = tokio::task::LocalSet::new();
        local.block_on(&runtime, async {
            const FRAME_BYTES: usize = 1024 * 1024;
            const PAYLOAD_BYTES: usize = 5 * FRAME_BYTES;
            const WRITER_RESERVE_BYTES: usize = 14 + 131;

            let listener = TcpListener::bind("127.0.0.1:0")
                .await
                .expect("test listener binds");
            let address = listener.local_addr().expect("listener has an address");
            let (send_ping, receive_ping) = oneshot::channel();
            let (pong_before_message, pong_result) = oneshot::channel();
            tokio::task::spawn_local(async move {
                let (stream, _) = listener.accept().await.expect("server accepts client");
                let mut socket = accept_async(stream).await.expect("server upgrades client");
                receive_ping.await.expect("test requests ping");
                socket
                    .send(Message::Ping(bytes::Bytes::from_static(b"pressure")))
                    .await
                    .expect("server sends ping");
                let is_pong_first = loop {
                    match socket
                        .next()
                        .await
                        .expect("client responds")
                        .expect("client frame is valid")
                    {
                        Message::Pong(payload) => {
                            assert_eq!(payload, bytes::Bytes::from_static(b"pressure"));
                            break true;
                        }
                        Message::Text(_) | Message::Binary(_) => break false,
                        _ => {}
                    }
                };
                if is_pong_first {
                    loop {
                        match socket
                            .next()
                            .await
                            .expect("client completes the upload")
                            .expect("client frame is valid")
                        {
                            Message::Text(_) | Message::Binary(_) => break,
                            _ => {}
                        }
                    }
                }
                let _ = pong_before_message.send(is_pong_first);
                socket
                    .send(Message::Text("done".into()))
                    .await
                    .expect("server keeps the upgraded socket live");
                let _ = socket.next().await;
            });

            let proxy = PausedUploadProxy::start(address).await;
            let clock: Rc<dyn Clock> = RealClock::new();
            let url = url::Url::parse(&format!("ws://{}/v1/responses", proxy.address))
                .expect("test URL is valid");
            let websocket_config = WebSocketConfig::default()
                .write_buffer_size(64 * 1024)
                .max_write_buffer_size(PAYLOAD_BYTES + WRITER_RESERVE_BYTES)
                .max_frame_size(Some(FRAME_BYTES));
            let socket = connector::connect(
                &url,
                &BTreeMap::new(),
                &ClientConfig::default(),
                websocket_config,
                clock.clone(),
                None,
            )
            .await
            .expect("client upgrades server");
            proxy.pause();
            let operation = PreparedWsOperation::new(
                [PreparedWsMessage::text(
                    bytes::Bytes::from(vec![b'x'; PAYLOAD_BYTES]),
                    PreparedWsMessageRole::MeasuredInput,
                )],
                None,
            );
            let now_ns = clock.now_ns();
            let mut driver = SocketOperationDriver::start(
                socket,
                clock,
                &operation,
                ApplicationQueueLimits::new(1, PAYLOAD_BYTES).with_max_frame_bytes(FRAME_BYTES),
                DriverTiming {
                    deadline_ns: Some(now_ns.saturating_add(5_000_000_000)),
                    rotation_deadline_ns: now_ns.saturating_add(5_000_000_000),
                    ping_interval_ns: 5_000_000_000,
                    stream_idle_timeout_ns: 5_000_000_000,
                    cancel_after_ns: None,
                },
                64,
            )
            .expect("driver starts");

            tokio::time::timeout(Duration::from_secs(1), proxy.wait_for_upload_data())
                .await
                .expect("client upload reaches the paused proxy");
            send_ping.send(()).expect("server ping request is live");
            tokio::time::sleep(Duration::from_millis(10)).await;
            proxy.resume();

            tokio::time::timeout(Duration::from_secs(5), async {
                loop {
                    if matches!(
                        driver.next().await.expect("writer stays healthy"),
                        DriverEvent::Flushed { .. }
                    ) {
                        break;
                    }
                }
            })
            .await
            .expect("writer progresses after pressure releases");
            assert!(
                tokio::time::timeout(Duration::from_secs(1), pong_result)
                    .await
                    .expect("server observes a client message")
                    .expect("server reports ordering"),
                "Pong must overtake the remaining application fragments"
            );
            let _ = driver.finish().await.expect("driver reunites socket");
        });
    }

    #[test]
    fn cancellation_arms_only_after_every_measured_input_is_flushed() {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("test runtime builds");
        let local = tokio::task::LocalSet::new();
        local.block_on(&runtime, async {
            let state = Rc::new(RefCell::new(SequencedWriteState::default()));
            let socket = SequencedWriteSocket {
                state: state.clone(),
            };
            let operation = PreparedWsOperation::new(
                [
                    PreparedWsMessage::text(
                        bytes::Bytes::from_static(b"first"),
                        PreparedWsMessageRole::MeasuredInput,
                    ),
                    PreparedWsMessage::text(
                        bytes::Bytes::from_static(b"second"),
                        PreparedWsMessageRole::MeasuredInput,
                    ),
                ],
                None,
            );
            let clock: Rc<dyn Clock> = RealClock::new();
            let now_ns = clock.now_ns();
            let mut driver = SocketOperationDriver::start(
                socket,
                clock,
                &operation,
                ApplicationQueueLimits::new(2, 11),
                DriverTiming {
                    deadline_ns: Some(now_ns.saturating_add(1_000_000_000)),
                    rotation_deadline_ns: now_ns.saturating_add(1_000_000_000),
                    ping_interval_ns: 1_000_000_000,
                    stream_idle_timeout_ns: 1_000_000_000,
                    cancel_after_ns: Some(0),
                },
                64,
            )
            .expect("driver starts");

            assert!(matches!(
                driver.next().await,
                Ok(DriverEvent::Flushed {
                    role: PreparedWsMessageRole::MeasuredInput,
                    ..
                })
            ));
            {
                let mut state = state.borrow_mut();
                state.is_second_ready = true;
                if let Some(writer) = state.writer.take() {
                    writer.wake();
                }
            }
            assert!(matches!(
                driver.next().await,
                Ok(DriverEvent::Flushed {
                    role: PreparedWsMessageRole::MeasuredInput,
                    ..
                })
            ));
            assert!(matches!(
                driver.next().await,
                Err(super::WsDriverError::RequestCancellation)
            ));
        });
    }

    #[test]
    fn split_driver_flushes_input_while_receiving_application_events() {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("test runtime builds");
        let local = tokio::task::LocalSet::new();
        local.block_on(&runtime, async {
            let listener = TcpListener::bind("127.0.0.1:0")
                .await
                .expect("test listener binds");
            let address = listener.local_addr().expect("listener has an address");
            tokio::task::spawn_local(async move {
                let (stream, _) = listener.accept().await.expect("server accepts client");
                let mut socket = accept_async(stream).await.expect("server upgrades client");
                let request = socket
                    .next()
                    .await
                    .expect("client sends request")
                    .expect("request frame is valid");
                assert_eq!(request.into_text().expect("request is text"), "request");
                socket
                    .send(Message::Ping(bytes::Bytes::from_static(b"health")))
                    .await
                    .expect("server sends ping");
                let pong = socket
                    .next()
                    .await
                    .expect("client answers ping")
                    .expect("pong frame is valid");
                assert_eq!(pong, Message::Pong(bytes::Bytes::from_static(b"health")));
                socket
                    .send(Message::Text("reply".into()))
                    .await
                    .expect("server sends reply");
                let _ = socket.next().await;
            });

            let clock: Rc<dyn Clock> = RealClock::new();
            let url = url::Url::parse(&format!("ws://{address}/v1/responses"))
                .expect("test URL is valid");
            let socket = connector::connect(
                &url,
                &BTreeMap::new(),
                &ClientConfig::default(),
                WebSocketConfig::default(),
                clock.clone(),
                None,
            )
            .await
            .expect("client upgrades server");
            let operation = PreparedWsOperation::new(
                [PreparedWsMessage::text(
                    bytes::Bytes::from_static(b"request"),
                    PreparedWsMessageRole::MeasuredInput,
                )],
                None,
            );
            let now_ns = clock.now_ns();
            let mut driver = SocketOperationDriver::start(
                socket,
                clock,
                &operation,
                ApplicationQueueLimits::new(1, 7),
                DriverTiming {
                    deadline_ns: Some(now_ns.saturating_add(1_000_000_000)),
                    rotation_deadline_ns: now_ns.saturating_add(1_000_000_000),
                    ping_interval_ns: 1_000_000_000,
                    stream_idle_timeout_ns: 1_000_000_000,
                    cancel_after_ns: None,
                },
                64,
            )
            .expect("driver starts");

            let mut flushed = false;
            let mut received = false;
            for _ in 0..2 {
                match driver.next().await.expect("driver progresses") {
                    DriverEvent::Flushed {
                        role: PreparedWsMessageRole::MeasuredInput,
                        ..
                    } => flushed = true,
                    DriverEvent::Application {
                        payload, is_text, ..
                    } => {
                        assert!(is_text);
                        assert_eq!(payload, bytes::Bytes::from_static(b"reply"));
                        received = true;
                    }
                    DriverEvent::Flushed { .. } => {}
                }
                if flushed && received {
                    break;
                }
            }
            assert!(flushed, "measured input must flush");
            assert!(received, "reader must receive the server application event");
            let _ = driver.finish().await.expect("driver reunites socket");
        });
    }
}
