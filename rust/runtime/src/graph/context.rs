// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Per-trace mutable state and node-result shapes.
//!
//! The mutable fields are interior-mutable (`RefCell`) so every fire task (each
//! holding an `Rc<TraceContext>`) shares them lock-free — one trace's tasks all
//! run on a single thread, so no synchronization is needed on the hot path.

use crate::graph::channel_store::VersionedChannelStore;
use crate::graph::errors::TraceError;
use crate::graph::model::TraceRecord;
use std::cell::RefCell;
use std::collections::{BTreeMap, HashSet};
use std::rc::Rc;
use tokio::sync::Notify;

/// The success marker returned by a node's execute (a node writes exactly one
/// channel, published directly to the store rather than carried here).
#[derive(Debug, Clone, Default)]
pub struct NodeExecutionResult;

/// Per-trace mutable state passed into every node's fire path.
pub struct TraceContext {
    pub trace: TraceRecord,
    /// `Rc<str>` copy of `trace.id`, derived once here so the per-node fire path
    /// clones a pointer instead of reallocating the id string on every node.
    pub trace_id: Rc<str>,
    pub store: Rc<VersionedChannelStore>,
    pub scheduled_node_ids: RefCell<HashSet<String>>,
    /// Node ids that finished firing. The cycle guard uses this to distinguish
    /// an in-flight re-schedule (AND-fan-in dedup) from re-entering a completed
    /// node (a genuine cycle).
    pub completed_node_ids: RefCell<HashSet<String>>,
    pub node_finish_wall_us: RefCell<BTreeMap<String, f64>>,
    pub node_dispatch_wall_us: RefCell<BTreeMap<String, f64>>,
    pub node_first_token_wall_us: RefCell<BTreeMap<String, f64>>,
    /// Woken whenever any node resolves its first-token status (a first token
    /// was observed, or the node finished without one). Successors gated on
    /// `delay_after_predecessor_first_token_us` park here and re-check.
    first_token_notify: Rc<Notify>,
    /// First trace-stop error observed; makes the run halt.
    pub abort: RefCell<Option<TraceError>>,
    /// Wakes node-policy admission futures when fail-fast aborts the trace.
    abort_notify: Rc<Notify>,
}

impl TraceContext {
    pub fn new(trace: TraceRecord, store: Rc<VersionedChannelStore>) -> Rc<Self> {
        let trace_id: Rc<str> = Rc::from(trace.id.as_str());
        Rc::new(TraceContext {
            trace,
            trace_id,
            store,
            scheduled_node_ids: RefCell::new(HashSet::new()),
            completed_node_ids: RefCell::new(HashSet::new()),
            node_finish_wall_us: RefCell::new(BTreeMap::new()),
            node_dispatch_wall_us: RefCell::new(BTreeMap::new()),
            node_first_token_wall_us: RefCell::new(BTreeMap::new()),
            first_token_notify: Rc::new(Notify::new()),
            abort: RefCell::new(None),
            abort_notify: Rc::new(Notify::new()),
        })
    }

    /// Signal that a node's first-token status is now resolved (observed a
    /// first token, or finished without one). Callers must update the backing
    /// state (`node_first_token_wall_us` / `completed_node_ids`) BEFORE calling
    /// this, so a woken waiter's re-check succeeds.
    pub fn notify_first_token(&self) {
        self.first_token_notify.notify_waiters();
    }

    /// Park until `node_id`'s first-token status resolves — either a first token
    /// was observed, the node finished, or the whole trace aborted. Re-checks
    /// existing state on each wake (single-threaded, so the signal can't be lost
    /// between check and await).
    pub async fn await_first_token(&self, node_id: &str) {
        loop {
            {
                if self.node_first_token_wall_us.borrow().contains_key(node_id)
                    || self.completed_node_ids.borrow().contains(node_id)
                    || self.is_aborted()
                {
                    return;
                }
            }
            let notify = self.first_token_notify.clone();
            notify.notified().await;
        }
    }

    /// Record a node's first-token wall time (first wins) and latch its
    /// first-token event, so successors gated on
    /// `delay_after_predecessor_first_token_us` unblock at the predecessor's
    /// first token rather than at its completion.
    pub fn set_first_token(&self, node_id: &str, wall_us: f64) {
        self.node_first_token_wall_us
            .borrow_mut()
            .entry(node_id.to_string())
            .or_insert(wall_us);
        self.notify_first_token();
    }

    /// Record the first trace-stop; later ones are ignored (first error wins).
    pub fn set_abort(&self, err: TraceError) {
        let mut slot = self.abort.borrow_mut();
        if slot.is_none() {
            let reason = err.to_string();
            *slot = Some(err);
            drop(slot);
            self.store.abort_all(reason);
            self.notify_first_token();
            self.abort_notify.notify_waiters();
        }
    }

    /// Park until the trace abort latch is set.
    pub async fn await_abort(&self) {
        loop {
            if self.is_aborted() {
                return;
            }
            self.abort_notify.notified().await;
        }
    }

    /// True once a trace-stop has been recorded.
    pub fn is_aborted(&self) -> bool {
        self.abort.borrow().is_some()
    }
}
