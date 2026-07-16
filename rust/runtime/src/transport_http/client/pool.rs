// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Bounded connection management and reuse leases.
//!
//! Python's shared connector is bounded and queues when exhausted, while its
//! sticky connector owns exactly one connection per correlation id. This module
//! preserves that policy without putting synchronization on the token path:
//! state is local `Rc`/`RefCell`, a request acquires one RAII lease, HTTP/1
//! leases are exclusive, and HTTP/2 leases clone one live multiplexed sender.
//!
//! Waiters queue FIFO on the per-origin `Notify`. Freeing one slot (an HTTP/1
//! lease returning, or a connect reservation rolling back) wakes exactly one
//! waiter with [`Notify::notify_one`]; only transitions that admit many at once
//! — a first connect revealing multi-slot HTTP/1 or unbounded HTTP/2, and a
//! session retire — broadcast with `notify_waiters`. A fresh request can still
//! take a just-freed idle slot ahead of a woken waiter, which then re-queues.

use std::cell::{Cell, RefCell};
use std::collections::{HashMap, VecDeque};
use std::rc::Rc;

use crate::clock::Clock;
use async_trait::async_trait;
use tokio::sync::Notify;
use url::Url;

use crate::transport_http::client::connection::{Sender, SocketInfo, establish_with_resolver};
use crate::transport_http::client::resolver::{CachingDnsResolver, DnsResolver};
use crate::transport_http::config::ClientConfig;
use crate::transport_http::models::{ConnectionReuseStrategy, ErrorDetails, TraceData};

fn origin_key(url: &Url) -> String {
    format!(
        "{}://{}:{}",
        url.scheme(),
        url.host_str().unwrap_or(""),
        url.port_or_known_default().unwrap_or(0)
    )
}

fn expired(idle_since_ns: i64, now_ns: i64, keepalive_ns: Option<i64>) -> bool {
    keepalive_ns
        .filter(|timeout| *timeout >= 0)
        .is_some_and(|timeout| now_ns.saturating_sub(idle_since_ns) >= timeout)
}

fn copy_socket(trace: &mut TraceData, socket: SocketInfo) {
    trace.local_ip = Some(socket.local.ip().to_string());
    trace.local_port = Some(socket.local.port());
    trace.remote_ip = Some(socket.remote.ip().to_string());
    trace.remote_port = Some(socket.remote.port());
}

struct IdleH1 {
    sender: Sender,
    socket: SocketInfo,
    idle_since_ns: i64,
}

struct SharedH2 {
    sender: Sender,
    socket: SocketInfo,
    generation: u64,
    active_streams: usize,
    idle_since_ns: Option<i64>,
}

enum ProtocolState {
    Unknown { connecting: bool },
    H1 { open: usize, idle: VecDeque<IdleH1> },
    H2(SharedH2),
}

struct PoolEntry {
    state: RefCell<ProtocolState>,
    notify: Notify,
    retired: Cell<bool>,
}

impl PoolEntry {
    fn new() -> Self {
        Self {
            state: RefCell::new(ProtocolState::Unknown { connecting: false }),
            notify: Notify::new(),
            retired: Cell::new(false),
        }
    }

    fn prune(&self, now_ns: i64, keepalive_ns: Option<i64>) {
        let mut state = self.state.borrow_mut();
        let reset = match &mut *state {
            ProtocolState::Unknown { .. } => false,
            ProtocolState::H1 { open, idle } => {
                let before = idle.len();
                idle.retain(|connection| {
                    connection.sender.is_ready()
                        && !connection.sender.is_closed()
                        && !expired(connection.idle_since_ns, now_ns, keepalive_ns)
                });
                *open = open.saturating_sub(before - idle.len());
                *open == 0
            }
            ProtocolState::H2(shared) => {
                shared.sender.is_closed()
                    || (shared.active_streams == 0
                        && shared
                            .idle_since_ns
                            .is_some_and(|idle| expired(idle, now_ns, keepalive_ns)))
            }
        };
        if reset {
            *state = ProtocolState::Unknown { connecting: false };
        }
    }
}

struct PoolInner {
    origins: RefCell<HashMap<String, Rc<PoolEntry>>>,
    sticky: RefCell<HashMap<String, (String, Rc<PoolEntry>)>>,
    resolver: Rc<dyn DnsResolver>,
    next_generation: Cell<u64>,
}

impl PoolInner {
    fn generation(&self) -> u64 {
        let next = self.next_generation.get();
        self.next_generation.set(next.wrapping_add(1));
        next
    }
}

/// Connection-acquisition policy seam used by [`HttpTransport`](crate::transport_http::transport::http_transport::HttpTransport).
#[async_trait(?Send)]
pub trait ConnectionManager {
    /// Acquire one request lease under the selected reuse strategy.
    async fn acquire(
        &self,
        url: &Url,
        cfg: &ClientConfig,
        clock: Rc<dyn Clock>,
        reuse: ConnectionReuseStrategy,
        correlation_id: Option<&str>,
        trace: &mut TraceData,
    ) -> Result<ConnectionLease, ErrorDetails>;

    /// Retire every connection and waiter belonging to a sticky session.
    fn release_session(&self, correlation_id: &str);
}

/// Local, bounded pool used by the native HTTP transport.
#[derive(Clone)]
pub struct ConnectionPool {
    inner: Rc<PoolInner>,
}

impl ConnectionPool {
    /// Build a pool with a shared caching system resolver.
    pub fn new() -> Self {
        Self::with_resolver(Rc::new(CachingDnsResolver::default()))
    }

    /// Build a pool over an injected DNS policy.
    pub fn with_resolver(resolver: Rc<dyn DnsResolver>) -> Self {
        Self {
            inner: Rc::new(PoolInner {
                origins: RefCell::new(HashMap::new()),
                sticky: RefCell::new(HashMap::new()),
                resolver,
                next_generation: Cell::new(1),
            }),
        }
    }

    fn entry(
        &self,
        url: &Url,
        reuse: ConnectionReuseStrategy,
        correlation_id: Option<&str>,
    ) -> Result<Rc<PoolEntry>, ErrorDetails> {
        let origin = origin_key(url);
        match reuse {
            ConnectionReuseStrategy::Never => {
                Err(ErrorDetails::other("never strategy has no pool entry"))
            }
            ConnectionReuseStrategy::Pooled => Ok(self
                .inner
                .origins
                .borrow_mut()
                .entry(origin)
                .or_insert_with(|| Rc::new(PoolEntry::new()))
                .clone()),
            ConnectionReuseStrategy::StickyUserSessions => {
                let correlation_id = correlation_id.ok_or_else(|| {
                    ErrorDetails::other("sticky strategy requires correlation_id")
                })?;
                let mut sticky = self.inner.sticky.borrow_mut();
                match sticky.get(correlation_id) {
                    Some((bound_origin, _)) if bound_origin != &origin => {
                        Err(ErrorDetails::other(format!(
                            "sticky session {correlation_id:?} is bound to {bound_origin}, not {origin}"
                        )))
                    }
                    Some((_, entry)) => Ok(entry.clone()),
                    None => {
                        let entry = Rc::new(PoolEntry::new());
                        sticky.insert(correlation_id.to_string(), (origin, entry.clone()));
                        Ok(entry)
                    }
                }
            }
        }
    }

    async fn acquire_managed(
        &self,
        entry: Rc<PoolEntry>,
        url: &Url,
        cfg: &ClientConfig,
        clock: Rc<dyn Clock>,
        trace: &mut TraceData,
        h1_limit: usize,
    ) -> Result<ConnectionLease, ErrorDetails> {
        if h1_limit == 0 {
            return Err(ErrorDetails::other(
                "max_connections_per_origin must be positive",
            ));
        }

        loop {
            if entry.retired.get() {
                return Err(ErrorDetails::other(
                    "sticky connection lease was released while waiting",
                ));
            }

            let notified = entry.notify.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();

            let now_ns = clock.now_ns();
            entry.prune(now_ns, cfg.keepalive_ns);
            let decision = {
                let mut state = entry.state.borrow_mut();
                match &mut *state {
                    ProtocolState::Unknown { connecting } => {
                        if *connecting {
                            AcquireDecision::Wait
                        } else {
                            *connecting = true;
                            AcquireDecision::Establish(ReservationKind::First)
                        }
                    }
                    ProtocolState::H1 { open, idle } => {
                        if let Some(connection) = idle.pop_front() {
                            AcquireDecision::ReuseH1(connection)
                        } else if *open < h1_limit {
                            *open += 1;
                            AcquireDecision::Establish(ReservationKind::AdditionalH1)
                        } else {
                            AcquireDecision::Wait
                        }
                    }
                    ProtocolState::H2(shared) => {
                        // invariant: state is H2 only after storing a multiplexed root, so the sender is always clonable.
                        let sender = shared
                            .sender
                            .clone_multiplex()
                            .expect("H2 pool state must contain an H2 sender");
                        shared.active_streams += 1;
                        shared.idle_since_ns = None;
                        AcquireDecision::ReuseH2 {
                            sender,
                            socket: shared.socket,
                            generation: shared.generation,
                        }
                    }
                }
            };

            match decision {
                AcquireDecision::Wait => {
                    if trace.connection_pool_wait_start_ns.is_none() {
                        trace.connection_pool_wait_start_ns = Some(now_ns);
                    }
                    notified.await;
                }
                AcquireDecision::ReuseH1(connection) => {
                    finish_wait(trace, &clock);
                    trace.connection_reused_ns = Some(clock.now_ns());
                    copy_socket(trace, connection.socket);
                    return Ok(ConnectionLease::managed_h1(
                        connection.sender,
                        connection.socket,
                        entry.clone(),
                        clock.clone(),
                    ));
                }
                AcquireDecision::ReuseH2 {
                    sender,
                    socket,
                    generation,
                } => {
                    finish_wait(trace, &clock);
                    trace.connection_reused_ns = Some(clock.now_ns());
                    copy_socket(trace, socket);
                    return Ok(ConnectionLease::managed_h2(
                        sender,
                        entry.clone(),
                        clock.clone(),
                        generation,
                    ));
                }
                AcquireDecision::Establish(reservation) => {
                    finish_wait(trace, &clock);
                    let mut guard = ReservationGuard::new(entry.clone(), reservation);
                    let established = establish_with_resolver(
                        url,
                        cfg,
                        clock.clone(),
                        trace,
                        self.inner.resolver.as_ref(),
                    )
                    .await;
                    let (sender, socket) = match established {
                        Ok(connection) => connection,
                        Err(error) => return Err(error),
                    };
                    if entry.retired.get() {
                        return Err(ErrorDetails::other(
                            "sticky connection lease was released during connect",
                        ));
                    }

                    let lease = match reservation {
                        ReservationKind::First if sender.is_multiplexed() => {
                            // invariant: `is_multiplexed()` was just checked true, so the root sender is clonable.
                            let root = sender
                                .clone_multiplex()
                                .expect("multiplexed sender must be clonable");
                            let generation = self.inner.generation();
                            *entry.state.borrow_mut() = ProtocolState::H2(SharedH2 {
                                sender: root,
                                socket,
                                generation,
                                active_streams: 1,
                                idle_since_ns: None,
                            });
                            ConnectionLease::managed_h2(
                                sender,
                                entry.clone(),
                                clock.clone(),
                                generation,
                            )
                        }
                        ReservationKind::First => {
                            *entry.state.borrow_mut() = ProtocolState::H1 {
                                open: 1,
                                idle: VecDeque::new(),
                            };
                            ConnectionLease::managed_h1(
                                sender,
                                socket,
                                entry.clone(),
                                clock.clone(),
                            )
                        }
                        ReservationKind::AdditionalH1 if !sender.is_multiplexed() => {
                            ConnectionLease::managed_h1(
                                sender,
                                socket,
                                entry.clone(),
                                clock.clone(),
                            )
                        }
                        ReservationKind::AdditionalH1 => {
                            return Err(ErrorDetails::other(
                                "origin changed protocol from HTTP/1 to HTTP/2",
                            ));
                        }
                    };
                    guard.disarm();
                    // First connect can admit many waiters (multi-slot H1 / H2),
                    // so broadcast, not a single-slot handoff.
                    entry.notify.notify_waiters();
                    return Ok(lease);
                }
            }
        }
    }
}

impl Default for ConnectionPool {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait(?Send)]
impl ConnectionManager for ConnectionPool {
    async fn acquire(
        &self,
        url: &Url,
        cfg: &ClientConfig,
        clock: Rc<dyn Clock>,
        reuse: ConnectionReuseStrategy,
        correlation_id: Option<&str>,
        trace: &mut TraceData,
    ) -> Result<ConnectionLease, ErrorDetails> {
        if reuse == ConnectionReuseStrategy::Never {
            let (sender, _) = establish_with_resolver(
                url,
                cfg,
                clock.clone(),
                trace,
                self.inner.resolver.as_ref(),
            )
            .await?;
            return Ok(ConnectionLease::never(sender));
        }

        let entry = self.entry(url, reuse, correlation_id)?;
        let limit = if reuse == ConnectionReuseStrategy::StickyUserSessions {
            1
        } else {
            cfg.max_connections_per_origin
        };
        self.acquire_managed(entry, url, cfg, clock, trace, limit)
            .await
    }

    fn release_session(&self, correlation_id: &str) {
        if let Some((_, entry)) = self.inner.sticky.borrow_mut().remove(correlation_id) {
            entry.retired.set(true);
            *entry.state.borrow_mut() = ProtocolState::Unknown { connecting: false };
            entry.notify.notify_waiters();
        }
    }
}

fn finish_wait(trace: &mut TraceData, clock: &Rc<dyn Clock>) {
    if trace.connection_pool_wait_start_ns.is_some() {
        trace.connection_pool_wait_end_ns = Some(clock.now_ns());
    }
}

enum AcquireDecision {
    Wait,
    Establish(ReservationKind),
    ReuseH1(IdleH1),
    ReuseH2 {
        sender: Sender,
        socket: SocketInfo,
        generation: u64,
    },
}

#[derive(Clone, Copy)]
enum ReservationKind {
    First,
    AdditionalH1,
}

struct ReservationGuard {
    entry: Rc<PoolEntry>,
    kind: ReservationKind,
    armed: bool,
}

impl ReservationGuard {
    fn new(entry: Rc<PoolEntry>, kind: ReservationKind) -> Self {
        Self {
            entry,
            kind,
            armed: true,
        }
    }

    fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for ReservationGuard {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        let mut state = self.entry.state.borrow_mut();
        match (self.kind, &mut *state) {
            (ReservationKind::First, ProtocolState::Unknown { connecting }) => {
                *connecting = false;
            }
            (ReservationKind::AdditionalH1, ProtocolState::H1 { open, .. }) => {
                *open = open.saturating_sub(1);
                if *open == 0 {
                    *state = ProtocolState::Unknown { connecting: false };
                }
            }
            _ => {}
        }
        drop(state);
        // Rollback frees one slot: FIFO-wake one waiter (see module docs).
        self.entry.notify.notify_one();
    }
}

enum LeaseKind {
    Never,
    H1 {
        entry: Rc<PoolEntry>,
        clock: Rc<dyn Clock>,
        socket: SocketInfo,
    },
    H2 {
        entry: Rc<PoolEntry>,
        clock: Rc<dyn Clock>,
        generation: u64,
    },
}

/// RAII ownership of one request's connection or HTTP/2 stream capability.
///
/// A lease is non-reusable by default. Call [`mark_reusable`](Self::mark_reusable)
/// only after the complete response has drained successfully; cancellation and
/// early-return paths then discard HTTP/1 capacity automatically.
pub struct ConnectionLease {
    sender: Option<Sender>,
    kind: LeaseKind,
    reusable: bool,
}

impl ConnectionLease {
    fn never(sender: Sender) -> Self {
        Self {
            sender: Some(sender),
            kind: LeaseKind::Never,
            reusable: false,
        }
    }

    fn managed_h1(
        sender: Sender,
        socket: SocketInfo,
        entry: Rc<PoolEntry>,
        clock: Rc<dyn Clock>,
    ) -> Self {
        Self {
            sender: Some(sender),
            kind: LeaseKind::H1 {
                entry,
                clock,
                socket,
            },
            reusable: false,
        }
    }

    fn managed_h2(
        sender: Sender,
        entry: Rc<PoolEntry>,
        clock: Rc<dyn Clock>,
        generation: u64,
    ) -> Self {
        Self {
            sender: Some(sender),
            kind: LeaseKind::H2 {
                entry,
                clock,
                generation,
            },
            reusable: false,
        }
    }

    /// Borrow the protocol sender for the request lifecycle.
    pub fn sender_mut(&mut self) -> &mut Sender {
        self.sender
            .as_mut()
            .expect("connection lease sender already consumed")
    }

    /// Return the HTTP/1 connection to its pool when this lease drops.
    ///
    /// HTTP/2 stream leases always release their active-stream accounting; the
    /// shared connection remains only while its root sender is open.
    pub fn mark_reusable(&mut self) {
        self.reusable = true;
    }
}

impl Drop for ConnectionLease {
    fn drop(&mut self) {
        let Some(sender) = self.sender.take() else {
            return;
        };
        match &self.kind {
            LeaseKind::Never => {}
            LeaseKind::H1 {
                entry,
                clock,
                socket,
            } => {
                let mut state = entry.state.borrow_mut();
                let mut reset = false;
                if let ProtocolState::H1 { open, idle } = &mut *state {
                    if self.reusable
                        && !entry.retired.get()
                        && sender.is_ready()
                        && !sender.is_closed()
                    {
                        idle.push_back(IdleH1 {
                            sender,
                            socket: *socket,
                            idle_since_ns: clock.now_ns(),
                        });
                    } else {
                        *open = open.saturating_sub(1);
                        reset = *open == 0;
                    }
                }
                if reset {
                    *state = ProtocolState::Unknown { connecting: false };
                }
                drop(state);
                // One HTTP/1 slot freed: FIFO-wake one waiter (see module docs).
                entry.notify.notify_one();
            }
            LeaseKind::H2 {
                entry,
                clock,
                generation,
            } => {
                let mut state = entry.state.borrow_mut();
                let mut reset = false;
                if let ProtocolState::H2(shared) = &mut *state
                    && shared.generation == *generation
                {
                    shared.active_streams = shared.active_streams.saturating_sub(1);
                    if sender.is_closed() || shared.sender.is_closed() {
                        reset = true;
                    } else if shared.active_streams == 0 {
                        shared.idle_since_ns = Some(clock.now_ns());
                    }
                }
                if reset {
                    *state = ProtocolState::Unknown { connecting: false };
                    drop(state);
                    // Only a reset leaves a waiter to wake (to reconnect); an open
                    // H2 sender never parks a request, so waking there would just
                    // store an unused permit.
                    entry.notify.notify_one();
                }
            }
        }
    }
}
