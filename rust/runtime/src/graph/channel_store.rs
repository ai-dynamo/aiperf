// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Per-trace channel state for the async-dataflow executor.
//!
//! A versioned append-only log per channel; writes are linearized by a single
//! monotonic sequence counter (`last_seq`); readers capture per-channel versions and reducers
//! consume them in `(write_seq, writer_node_id)` order. Values are stored by
//! clone of the `serde_json::Value`, never mutated in place.

use crate::graph::model::{ChannelSpec, Count, ReducerName};
use crate::graph::reducers::{ChanVal, ReducerError, apply_reducer};
use serde_json::Value;
use std::cell::RefCell;
use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::rc::Rc;
use tokio::sync::Notify;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StoreError {
    UnknownChannel(String),
    Orphaned { channel: String, reason: String },
    Reducer(ReducerError),
}

impl std::fmt::Display for StoreError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StoreError::UnknownChannel(c) => write!(f, "unknown channel: {c:?}"),
            StoreError::Orphaned { channel, reason } => {
                write!(f, "channel {channel:?} orphaned: {reason}")
            }
            StoreError::Reducer(e) => write!(f, "{e}"),
        }
    }
}
impl std::error::Error for StoreError {}

impl From<ReducerError> for StoreError {
    fn from(e: ReducerError) -> Self {
        StoreError::Reducer(e)
    }
}

/// One committed write to a value channel.
#[derive(Debug, Clone)]
struct LogEntry {
    write_seq: i64,
    writer_node_id: String,
    value: ChanVal,
}

/// Canonical reducer-consumption order for two log entries: by `write_seq`, then
/// by `writer_node_id` as a stable tie-break for same-seq concurrent writes.
fn entry_order(a: &LogEntry, b: &LogEntry) -> Ordering {
    a.write_seq
        .cmp(&b.write_seq)
        .then_with(|| a.writer_node_id.cmp(&b.writer_node_id))
}

/// Immutable snapshot of which channel versions count for one firing.
#[derive(Debug, Clone, Default)]
pub struct VersionCapture {
    pub captured_seqs: Vec<i64>,
}

struct StoreInner {
    /// Channel definitions — immutable for the run, so this is `Rc`-shared from
    /// the executor rather than deep-cloned into every per-trace store.
    specs: Rc<BTreeMap<String, ChannelSpec>>,
    logs: BTreeMap<String, Vec<LogEntry>>,
    arrival_count: BTreeMap<String, i64>,
    producers_remaining: BTreeMap<String, i64>,
    /// Declared producer counts — immutable for the run, `Rc`-shared like `specs`.
    producers_declared: Rc<BTreeMap<String, i64>>,
    /// The last committed write sequence number (0 when only init seeds exist).
    last_seq: i64,
    /// One `Notify` per channel; readers park on it and re-check their arrival
    /// count / orphan state on wake. `notify_waiters` wakes all parked readers
    /// of a channel (single-threaded, so a notify can't be lost between a
    /// reader's synchronous check and its `.await`).
    notifiers: BTreeMap<String, Rc<Notify>>,
    orphaned: BTreeMap<String, String>,
    overwrite_writer: BTreeMap<String, String>,
}

/// Per-trace channel store backed by per-channel append-only logs.
pub struct VersionedChannelStore {
    inner: RefCell<StoreInner>,
}

impl VersionedChannelStore {
    /// Build a per-trace store over `Rc`-shared, run-immutable channel `specs`
    /// and `producers_declared`. Only the mutable per-trace state (logs, arrival
    /// counts, remaining producers, notifiers) is allocated here; the shared maps
    /// are cloned as `Rc` pointer bumps rather than deep-copied every trace.
    pub fn new(
        initial: &BTreeMap<String, Value>,
        specs: Rc<BTreeMap<String, ChannelSpec>>,
        producers_declared: Rc<BTreeMap<String, i64>>,
    ) -> Result<Self, StoreError> {
        let mut logs: BTreeMap<String, Vec<LogEntry>> =
            specs.keys().map(|c| (c.clone(), Vec::new())).collect();
        let arrival_count: BTreeMap<String, i64> = specs.keys().map(|c| (c.clone(), 0)).collect();
        let producers_remaining: BTreeMap<String, i64> = specs
            .keys()
            .map(|c| (c.clone(), *producers_declared.get(c).unwrap_or(&0)))
            .collect();
        let notifiers: BTreeMap<String, Rc<Notify>> = specs
            .keys()
            .map(|c| (c.clone(), Rc::new(Notify::new())))
            .collect();

        // Seed initial state at write_seq 0 (reducer seed; not an arrival).
        for (ch, value) in initial {
            if !specs.contains_key(ch) {
                return Err(StoreError::UnknownChannel(ch.clone()));
            }
            let value = channel_value(&specs[ch], value);
            logs.get_mut(ch).unwrap().push(LogEntry {
                write_seq: 0,
                writer_node_id: "__init__".to_string(),
                value,
            });
        }

        Ok(VersionedChannelStore {
            inner: RefCell::new(StoreInner {
                specs,
                logs,
                arrival_count,
                producers_remaining,
                producers_declared,
                last_seq: 0,
                notifiers,
                orphaned: BTreeMap::new(),
                overwrite_writer: BTreeMap::new(),
            }),
        })
    }

    // ---- write ----------------------------------------------------------

    /// Commit `value` to every channel in `channel_names`.
    pub fn write(
        &self,
        channel_names: &[String],
        value: &Value,
        writer_node_id: &str,
    ) -> Result<(), StoreError> {
        if channel_names.is_empty() {
            return Ok(());
        }
        {
            let inner = self.inner.borrow();
            for ch in channel_names {
                validate_write_channel(&inner, ch, writer_node_id)?;
            }
        }
        for ch in channel_names {
            let value = {
                let inner = self.inner.borrow();
                channel_value(&inner.specs[ch], value)
            };
            {
                let mut inner = self.inner.borrow_mut();
                commit_write_channel(&mut inner, ch, &value, writer_node_id);
            }
            self.notify_channel(ch);
        }
        Ok(())
    }

    /// Commit a value that already retains encoded message wires.
    pub fn write_channel_value(
        &self,
        channel_names: &[String],
        value: &ChanVal,
        writer_node_id: &str,
    ) -> Result<(), StoreError> {
        if channel_names.is_empty() {
            return Ok(());
        }
        {
            let inner = self.inner.borrow();
            for channel in channel_names {
                validate_write_channel(&inner, channel, writer_node_id)?;
            }
        }
        for channel in channel_names {
            {
                let mut inner = self.inner.borrow_mut();
                commit_write_channel(&mut inner, channel, value, writer_node_id);
            }
            self.notify_channel(channel);
        }
        Ok(())
    }

    // ---- await_inputs ---------------------------------------------------

    /// Block until every requirement is satisfied; return frozen captures.
    pub async fn await_inputs(
        &self,
        requirements: &[(String, Count)],
    ) -> Result<BTreeMap<String, VersionCapture>, StoreError> {
        let mut captures: BTreeMap<String, VersionCapture> = BTreeMap::new();
        for (channel, count) in requirements {
            {
                let inner = self.inner.borrow();
                if !inner.specs.contains_key(channel) {
                    return Err(StoreError::UnknownChannel(channel.clone()));
                }
                if let Some(reason) = inner.orphaned.get(channel) {
                    return Err(StoreError::Orphaned {
                        channel: channel.clone(),
                        reason: reason.clone(),
                    });
                }
            }
            let target = self.resolve_count(channel, count);
            self.await_count(channel, target).await?;
            captures.insert(channel.clone(), self.capture(channel, target));
        }
        Ok(captures)
    }

    fn resolve_count(&self, channel: &str, count: &Count) -> i64 {
        match count.as_int() {
            Some(n) => n,
            None => *self
                .inner
                .borrow()
                .producers_declared
                .get(channel)
                .unwrap_or(&0),
        }
    }

    async fn await_count(&self, channel: &str, target: i64) -> Result<(), StoreError> {
        if target <= 0 {
            return Ok(());
        }
        loop {
            // Check-then-park. Single-threaded: nothing runs between this
            // synchronous check and the `.notified().await` below, so a
            // `notify_waiters` from a concurrent writer can't be lost.
            let notify = {
                let inner = self.inner.borrow();
                if *inner.arrival_count.get(channel).unwrap_or(&0) >= target {
                    return Ok(());
                }
                if let Some(reason) = inner.orphaned.get(channel) {
                    return Err(StoreError::Orphaned {
                        channel: channel.to_string(),
                        reason: reason.clone(),
                    });
                }
                let arrival = *inner.arrival_count.get(channel).unwrap_or(&0);
                let remaining = *inner.producers_remaining.get(channel).unwrap_or(&0);
                // This reader's count can no longer be met: orphan THIS reader
                // only (a lower-count reader on the same channel may still be
                // satisfiable, so the channel itself is not poisoned here).
                if arrival + remaining < target {
                    return Err(StoreError::Orphaned {
                        channel: channel.to_string(),
                        reason: "insufficient_producers_remaining".to_string(),
                    });
                }
                inner
                    .notifiers
                    .get(channel)
                    .cloned()
                    .expect("channel has a notifier")
            };
            notify.notified().await;
        }
    }

    fn capture(&self, channel: &str, target: i64) -> VersionCapture {
        if target <= 0 {
            return VersionCapture::default();
        }
        let inner = self.inner.borrow();
        let entries = &inner.logs[channel];
        let mut non_init: Vec<&LogEntry> = entries.iter().filter(|e| e.write_seq != 0).collect();
        non_init.sort_by(|a, b| entry_order(a, b));
        let chosen = non_init.iter().take(target as usize);
        VersionCapture {
            captured_seqs: chosen.map(|e| e.write_seq).collect(),
        }
    }

    /// Wake every reader parked on `channel`; each re-checks its own count and
    /// orphan state and either proceeds or re-parks.
    fn notify_channel(&self, channel: &str) {
        let notify = self.inner.borrow().notifiers.get(channel).cloned();
        if let Some(n) = notify {
            n.notify_waiters();
        }
    }

    // ---- read -----------------------------------------------------------

    /// Return the reduced value per channel at the captured versions.
    pub fn read(
        &self,
        requirements: &[(String, Count)],
        capture: &BTreeMap<String, VersionCapture>,
    ) -> Result<BTreeMap<String, ChanVal>, StoreError> {
        let inner = self.inner.borrow();
        let mut out = BTreeMap::new();
        for (channel, _count) in requirements {
            if !inner.specs.contains_key(channel) {
                return Err(StoreError::UnknownChannel(channel.clone()));
            }
            let cap = capture.get(channel).ok_or_else(|| {
                StoreError::UnknownChannel(format!("no capture for channel {channel:?}"))
            })?;
            out.insert(
                channel.clone(),
                reduce_value_channel(&inner, channel, &cap.captured_seqs)?,
            );
        }
        Ok(out)
    }

    // ---- snapshots ------------------------------------------------------

    /// Final user-visible view of every channel.
    pub fn snapshot(&self) -> Result<BTreeMap<String, ChanVal>, StoreError> {
        let inner = self.inner.borrow();
        let mut out = BTreeMap::new();
        for ch in inner.specs.keys() {
            let entries = &inner.logs[ch];
            if entries.is_empty() {
                out.insert(ch.clone(), ChanVal::Unset);
                continue;
            }
            out.insert(ch.clone(), reduce_all(&inner, ch, entries)?);
        }
        Ok(out)
    }

    /// Latest committed write sequence number (0 when only init seeds exist).
    pub fn current_seq(&self) -> i64 {
        self.inner.borrow().last_seq
    }

    /// Reduce every channel considering only entries with `write_seq <= max_seq`.
    /// Unset channels are omitted from the result (absent == unwritten).
    pub fn snapshot_at_seq(&self, max_seq: i64) -> Result<BTreeMap<String, ChanVal>, StoreError> {
        let inner = self.inner.borrow();
        let mut out = BTreeMap::new();
        for ch in inner.specs.keys() {
            if let Some(value) = reduce_channel_at_seq(&inner, ch, max_seq)? {
                out.insert(ch.clone(), value);
            }
        }
        Ok(out)
    }

    /// Reduce only the named channels considering entries with `write_seq <= max_seq`.
    ///
    /// This is the executor's per-node input snapshot: a node materializes its
    /// prompt from just its `PromptItem::Splice` channels, so reducing every
    /// channel in the store on every fire is O(channels × history) wasted work on
    /// the profiled allocation hot path. The result over the covered channels is
    /// identical to [`snapshot_at_seq`] restricted to those keys — reducers are
    /// strictly per-channel, and a splice key that is not declared state (or has
    /// no visible write) is omitted here exactly as the full-store snapshot omits
    /// it, so the materializer's `.get(key)` sees the same value or the same
    /// absence. Repeated keys are reduced once.
    pub fn snapshot_selected_at_seq(
        &self,
        channels: &[&str],
        max_seq: i64,
    ) -> Result<BTreeMap<String, ChanVal>, StoreError> {
        let inner = self.inner.borrow();
        let mut out = BTreeMap::new();
        for &ch in channels {
            // Only declared channels have a log; the full-store snapshot iterates
            // `specs`, so an undeclared splice key is absent there too.
            if !inner.specs.contains_key(ch) || out.contains_key(ch) {
                continue;
            }
            if let Some(value) = reduce_channel_at_seq(&inner, ch, max_seq)? {
                out.insert(ch.to_string(), value);
            }
        }
        Ok(out)
    }

    // ---- producer accounting / orphan propagation ----------------------

    /// Inform the store that one producer of `channel` has terminated.
    pub fn mark_producer_done(&self, channel: &str, success: bool) -> Result<(), StoreError> {
        {
            let mut inner = self.inner.borrow_mut();
            if !inner.specs.contains_key(channel) {
                return Err(StoreError::UnknownChannel(channel.to_string()));
            }
            if *inner.producers_remaining.get(channel).unwrap_or(&0) > 0 {
                *inner.producers_remaining.get_mut(channel).unwrap() -= 1;
            }
            let arrival = *inner.arrival_count.get(channel).unwrap_or(&0);
            let remaining = *inner.producers_remaining.get(channel).unwrap_or(&0);
            let has_init_seed = inner
                .logs
                .get(channel)
                .map(|es| es.iter().any(|e| e.write_seq == 0))
                .unwrap_or(false);
            // The channel is truly dead — no producer will ever write, nothing
            // was written, no init seed — so poison it: every reader errors.
            if !success && remaining == 0 && arrival == 0 && !has_init_seed {
                inner
                    .orphaned
                    .insert(channel.to_string(), "all_producers_cancelled".to_string());
            }
        }
        // Wake parked readers: an over-target reader re-checks reachability and
        // orphans itself; a lower-count reader re-parks or proceeds.
        self.notify_channel(channel);
        Ok(())
    }

    /// Abort every parked reader in this trace and wake it to observe the abort.
    ///
    /// Fail-fast can prevent a downstream producer from ever being scheduled.
    /// Static producer accounting cannot decrement a node that never started,
    /// so poisoning every channel is the only complete trace-wide wakeup edge.
    pub fn abort_all(&self, reason: impl Into<String>) {
        let reason = reason.into();
        let channels = {
            let mut inner = self.inner.borrow_mut();
            let channels = inner.specs.keys().cloned().collect::<Vec<_>>();
            for channel in &channels {
                inner
                    .orphaned
                    .entry(channel.clone())
                    .or_insert_with(|| reason.clone());
            }
            channels
        };
        for channel in channels {
            self.notify_channel(&channel);
        }
    }
}

fn validate_write_channel(
    inner: &StoreInner,
    channel: &str,
    writer_node_id: &str,
) -> Result<(), StoreError> {
    let spec = inner
        .specs
        .get(channel)
        .ok_or_else(|| StoreError::UnknownChannel(channel.to_string()))?;
    if spec.reducer != ReducerName::Overwrite {
        return Ok(());
    }
    if let Some(prior) = inner.overwrite_writer.get(channel) {
        return Err(StoreError::Reducer(ReducerError::OverwriteConflict(
            format!("{prior} already; rejecting second writer {writer_node_id}"),
        )));
    }
    Ok(())
}

fn commit_write_channel(
    inner: &mut StoreInner,
    channel: &str,
    value: &ChanVal,
    writer_node_id: &str,
) {
    inner.last_seq += 1;
    let seq = inner.last_seq;
    inner.logs.get_mut(channel).unwrap().push(LogEntry {
        write_seq: seq,
        writer_node_id: writer_node_id.to_string(),
        value: value.clone(),
    });
    *inner.arrival_count.get_mut(channel).unwrap() += 1;
    if inner.specs[channel].reducer == ReducerName::Overwrite {
        inner
            .overwrite_writer
            .entry(channel.to_string())
            .or_insert_with(|| writer_node_id.to_string());
    }
}

fn reduce_value_channel(
    inner: &StoreInner,
    channel: &str,
    captured_seqs: &[i64],
) -> Result<ChanVal, StoreError> {
    let spec = &inner.specs[channel];
    let entries = &inner.logs[channel];
    let init = entries.iter().find(|e| e.write_seq == 0);
    let current = match init {
        Some(e) => e.value.clone(),
        None => ChanVal::Unset,
    };
    if captured_seqs.is_empty() {
        return Ok(current);
    }
    let seq_set: std::collections::HashSet<i64> = captured_seqs.iter().copied().collect();
    let mut chosen: Vec<&LogEntry> = entries
        .iter()
        .filter(|e| seq_set.contains(&e.write_seq))
        .collect();
    chosen.sort_by(|a, b| entry_order(a, b));
    let tuples: Vec<(String, ChanVal)> = chosen
        .iter()
        .map(|e| (e.writer_node_id.clone(), e.value.clone()))
        .collect();
    Ok(apply_reducer(spec.reducer, &current, &tuples)?)
}

/// Reduce one channel over just the entries with `write_seq <= max_seq`, reading
/// the log by reference.
///
/// Mirrors [`reduce_all`]'s ordering and init/tuple split exactly, but folds the
/// sequence filter in and never materializes an intermediate `Vec<LogEntry>` — it
/// borrows each entry and clones only the reducer inputs the reducer consumes.
/// Returns `None` when no entry is visible at `max_seq` (the caller omits the
/// channel, matching `snapshot_at_seq`'s empty-skip).
fn reduce_channel_at_seq(
    inner: &StoreInner,
    channel: &str,
    max_seq: i64,
) -> Result<Option<ChanVal>, StoreError> {
    let mut sorted: Vec<&LogEntry> = inner.logs[channel]
        .iter()
        .filter(|e| e.write_seq <= max_seq)
        .collect();
    if sorted.is_empty() {
        return Ok(None);
    }
    sorted.sort_by(|a, b| entry_order(a, b));
    let init = sorted.iter().find(|e| e.write_seq == 0);
    let current = match init {
        Some(e) => e.value.clone(),
        None => ChanVal::Unset,
    };
    let tuples: Vec<(String, ChanVal)> = sorted
        .iter()
        .filter(|e| e.write_seq != 0)
        .map(|e| (e.writer_node_id.clone(), e.value.clone()))
        .collect();
    let spec = &inner.specs[channel];
    if tuples.is_empty() {
        Ok(Some(current))
    } else {
        Ok(Some(apply_reducer(spec.reducer, &current, &tuples)?))
    }
}

fn reduce_all(
    inner: &StoreInner,
    channel: &str,
    entries: &[LogEntry],
) -> Result<ChanVal, StoreError> {
    let mut sorted: Vec<&LogEntry> = entries.iter().collect();
    sorted.sort_by(|a, b| entry_order(a, b));
    let init = sorted.iter().find(|e| e.write_seq == 0);
    let current = match init {
        Some(e) => e.value.clone(),
        None => ChanVal::Unset,
    };
    let tuples: Vec<(String, ChanVal)> = sorted
        .iter()
        .filter(|e| e.write_seq != 0)
        .map(|e| (e.writer_node_id.clone(), e.value.clone()))
        .collect();
    let spec = &inner.specs[channel];
    if tuples.is_empty() {
        Ok(current)
    } else {
        Ok(apply_reducer(spec.reducer, &current, &tuples)?)
    }
}

fn channel_value(spec: &ChannelSpec, value: &Value) -> ChanVal {
    if spec.channel_type == crate::graph::model::ChannelType::Messages
        && let Value::Array(messages) = value
    {
        return ChanVal::encoded_messages(
            messages
                .iter()
                .map(|message| {
                    let wire = serde_json::to_vec(message)
                        .expect("serde_json::Value serialization is infallible");
                    (message.clone(), bytes::Bytes::from(wire))
                })
                .collect(),
        );
    }
    ChanVal::Val(value.clone())
}
