// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Per-trace channel state for the async-dataflow executor.
//!
//! A versioned append-only log per channel; writes are linearized by a single
//! monotonic sequence counter (`last_seq`); readers capture per-channel versions and reducers
//! consume them in `(write_seq, writer_node_id)` order. Values are stored by
//! clone of the `serde_json::Value`, never mutated in place.

use crate::model::{ChannelSpec, Count, ReducerName};
use crate::reducers::{ChanVal, ReducerError, apply_reducer};
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
    value: Value,
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
    specs: BTreeMap<String, ChannelSpec>,
    logs: BTreeMap<String, Vec<LogEntry>>,
    arrival_count: BTreeMap<String, i64>,
    producers_remaining: BTreeMap<String, i64>,
    producers_declared: BTreeMap<String, i64>,
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
    pub fn new(
        initial: &BTreeMap<String, Value>,
        channel_specs: &BTreeMap<String, ChannelSpec>,
        producers_per_channel: &BTreeMap<String, i64>,
    ) -> Result<Self, StoreError> {
        let specs = channel_specs.clone();
        let mut logs: BTreeMap<String, Vec<LogEntry>> = channel_specs
            .keys()
            .map(|c| (c.clone(), Vec::new()))
            .collect();
        let arrival_count: BTreeMap<String, i64> =
            channel_specs.keys().map(|c| (c.clone(), 0)).collect();
        let producers_remaining: BTreeMap<String, i64> = channel_specs
            .keys()
            .map(|c| (c.clone(), *producers_per_channel.get(c).unwrap_or(&0)))
            .collect();
        let producers_declared = producers_remaining.clone();
        let notifiers: BTreeMap<String, Rc<Notify>> = channel_specs
            .keys()
            .map(|c| (c.clone(), Rc::new(Notify::new())))
            .collect();

        // Seed initial state at write_seq 0 (reducer seed; not an arrival).
        for (ch, value) in initial {
            if !specs.contains_key(ch) {
                return Err(StoreError::UnknownChannel(ch.clone()));
            }
            logs.get_mut(ch).unwrap().push(LogEntry {
                write_seq: 0,
                writer_node_id: "__init__".to_string(),
                value: value.clone(),
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
            {
                let mut inner = self.inner.borrow_mut();
                commit_write_channel(&mut inner, ch, value, writer_node_id);
            }
            self.notify_channel(ch);
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
            let entries: Vec<&LogEntry> = inner.logs[ch]
                .iter()
                .filter(|e| e.write_seq <= max_seq)
                .collect();
            if entries.is_empty() {
                continue;
            }
            let owned: Vec<LogEntry> = entries.into_iter().cloned().collect();
            out.insert(ch.clone(), reduce_all(&inner, ch, &owned)?);
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
    value: &Value,
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
        Some(e) => ChanVal::Val(e.value.clone()),
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
    let tuples: Vec<(String, Value)> = chosen
        .iter()
        .map(|e| (e.writer_node_id.clone(), e.value.clone()))
        .collect();
    Ok(apply_reducer(spec.reducer, &current, &tuples)?)
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
        Some(e) => ChanVal::Val(e.value.clone()),
        None => ChanVal::Unset,
    };
    let tuples: Vec<(String, Value)> = sorted
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
