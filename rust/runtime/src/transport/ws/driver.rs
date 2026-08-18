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
use tokio_tungstenite::tungstenite::{Error as TungsteniteError, Message};

use crate::body_plan::{
    PreparedWsMessage, PreparedWsMessageRole, PreparedWsOpcode, PreparedWsOperation,
};
use crate::clock::Clock;

const CONTROL_QUEUE_CAPACITY: usize = 4;

/// Count and byte bounds for one socket's outbound application queue.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ApplicationQueueLimits {
    max_commands: usize,
    max_bytes: usize,
}

impl ApplicationQueueLimits {
    /// Create queue limits already validated by configuration loading.
    pub(crate) const fn new(max_commands: usize, max_bytes: usize) -> Self {
        Self {
            max_commands,
            max_bytes,
        }
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
    /// Interval between keepalive pings.
    pub(crate) ping_interval_ns: i64,
    /// Maximum silence between complete application messages.
    pub(crate) stream_idle_timeout_ns: i64,
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
    InvalidTextMessage,
    ApplicationQueueClosed,
    ControlQueueFull,
    WriterStopped,
    Write(TungsteniteError),
    Read(TungsteniteError),
    PeerClosed,
    ResponseByteLimit { actual: usize, limit: usize },
    ResponseByteOverflow,
    StreamIdleTimeout,
    OperationDeadline,
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
            Self::InvalidTextMessage => {
                formatter.write_str("websocket text application message is not UTF-8")
            }
            Self::ApplicationQueueClosed => {
                formatter.write_str("websocket application writer stopped before enqueue")
            }
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
            Self::Reunite => formatter.write_str("websocket split halves could not be reunited"),
        }
    }
}

impl std::error::Error for WsDriverError {}

enum ControlCommand {
    Send(Message),
    Finish,
}

enum WriterNotice<S>
where
    S: Sink<Message, Error = TungsteniteError> + Unpin,
{
    Flushed {
        role: PreparedWsMessageRole,
        timestamp_ns: i64,
    },
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
    last_application_receive_ns: i64,
    response_bytes: usize,
    max_response_bytes: usize,
    finished_sink: Option<SplitSink<S, Message>>,
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
            last_application_receive_ns: now_ns,
            response_bytes: 0,
            max_response_bytes,
            finished_sink: None,
        })
    }

    /// Progress reads, successful flush notifications, keepalive, and deadlines.
    pub(crate) async fn next(&mut self) -> Result<DriverEvent, WsDriverError> {
        loop {
            self.drain_notices()?;
            if let Some(event) = self.pending.pop_front() {
                return Ok(event);
            }
            let now_ns = self.clock.now_ns();
            if self
                .timing
                .deadline_ns
                .is_some_and(|deadline_ns| now_ns >= deadline_ns)
            {
                return Err(WsDriverError::OperationDeadline);
            }
            if now_ns
                >= self
                    .last_application_receive_ns
                    .saturating_add(self.timing.stream_idle_timeout_ns)
            {
                return Err(WsDriverError::StreamIdleTimeout);
            }
            let read = self.read.as_mut().ok_or(WsDriverError::WriterStopped)?;
            let notice = self.notice.notified();
            let ping = self
                .clock
                .clone()
                .sleep(self.next_ping_ns.saturating_sub(now_ns));
            let idle = self.clock.clone().sleep(
                self.last_application_receive_ns
                    .saturating_add(self.timing.stream_idle_timeout_ns)
                    .saturating_sub(now_ns),
            );
            let deadline = deadline_sleep(self.clock.clone(), self.timing.deadline_ns, now_ns);
            tokio::select! {
                biased;
                () = notice => {}
                message = read.next() => self.on_message(message)?,
                () = deadline => return Err(WsDriverError::OperationDeadline),
                () = idle => return Err(WsDriverError::StreamIdleTimeout),
                () = ping => {
                    self.control_tx
                        .try_send(ControlCommand::Send(Message::Ping(bytes::Bytes::new())))
                        .map_err(|_| WsDriverError::ControlQueueFull)?;
                    self.next_ping_ns = self.clock.now_ns().saturating_add(self.timing.ping_interval_ns);
                }
            }
        }
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
            Message::Close(_) => Err(WsDriverError::PeerClosed),
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
                    self.pending
                        .push_back(DriverEvent::Flushed { role, timestamp_ns });
                }
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
            if self
                .timing
                .deadline_ns
                .is_some_and(|deadline_ns| now_ns >= deadline_ns)
            {
                return Err(WsDriverError::OperationDeadline);
            }
            tokio::select! {
                () = self.notice.notified() => {}
                () = deadline_sleep(self.clock.clone(), self.timing.deadline_ns, now_ns) => {
                    return Err(WsDriverError::OperationDeadline);
                }
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
        let result = if needs_finish {
            match application_rx.recv().await {
                Some(message) => send_application(&mut sink, message, &clock).await.map(Some),
                None => {
                    if let Err(error) = sink.flush().await {
                        Err(WsDriverError::Write(error))
                    } else {
                        push_notice(&notices, &notice, WriterNotice::Finished(sink));
                        return;
                    }
                }
            }
        } else {
            tokio::select! {
                biased;
                command = control_rx.recv() => match command {
                    Some(ControlCommand::Send(message)) => sink.send(message).await.map(|_| None).map_err(WsDriverError::Write),
                    Some(ControlCommand::Finish) => {
                        needs_finish = true;
                        Ok(None)
                    }
                    None => return,
                },
                message = application_rx.recv() => match message {
                    Some(message) => send_application(&mut sink, message, &clock).await.map(Some),
                    None => {
                        needs_finish = true;
                        Ok(None)
                    }
                }
            }
        };
        match result {
            Ok(Some((role, timestamp_ns))) => push_notice(
                &notices,
                &notice,
                WriterNotice::Flushed { role, timestamp_ns },
            ),
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
    clock: &Rc<dyn Clock>,
) -> Result<(PreparedWsMessageRole, i64), WsDriverError>
where
    S: Sink<Message, Error = TungsteniteError> + Unpin,
{
    let role = message.role();
    let wire = match message.opcode() {
        PreparedWsOpcode::Text => {
            let text = std::str::from_utf8(message.payload())
                .map_err(|_| WsDriverError::InvalidTextMessage)?;
            Message::Text(text.to_owned().into())
        }
        PreparedWsOpcode::Binary => Message::Binary(message.payload().clone()),
    };
    sink.send(wire).await.map_err(WsDriverError::Write)?;
    Ok((role, clock.now_ns()))
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
