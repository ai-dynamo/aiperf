// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Live in-flight streaming: the `/api/live` SSE endpoint.
//!
//! The running child writes `MetricsHeartbeat` NDJSON to the path in its
//! `AIPERF_CELLULAR_HEARTBEAT_LOG` (set per child by the run loop) — each line a full
//! cumulative snapshot: counters (issued/completed/errored), saturation (in_flight),
//! and percentile-projected TTFT/ITL/latency sketches, emitted per phase-progress
//! tick. The orchestrator server tails that file and pushes the latest snapshot to
//! the browser over Server-Sent Events. No ZMQ, no second pipe: the parent reads the
//! NDJSON the child already writes.
//!
//! Events emitted:
//! - `run`       — the live run's identity ([`super::LiveRun`]) when a run starts.
//! - `heartbeat` — the latest `metrics_heartbeat` line, whenever the file grows.
//! - `idle`      — once, when a run finishes and none is in flight.

use std::convert::Infallible;
use std::io::{Read, Seek, SeekFrom};
use std::time::Duration;

use axum::extract::State;
use axum::response::sse::{Event, KeepAlive, Sse};
use futures::stream::{self, Stream};

use super::AppState;

/// Poll cadence for the heartbeat file (the runner's own cadence is coarser).
const POLL: Duration = Duration::from_millis(400);

/// `GET /api/live` — Server-Sent Events of the in-flight run's heartbeat stream.
pub async fn live_stream(
    State(state): State<AppState>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let stream = stream::unfold(LiveTail::new(state), |mut tail| async move {
        // Loop until there is something to send; SSE keep-alive covers idle gaps.
        loop {
            tokio::time::sleep(POLL).await;
            if let Some(event) = tail.step() {
                return Some((Ok(event), tail));
            }
        }
    });
    Sse::new(stream).keep_alive(KeepAlive::new().interval(Duration::from_secs(10)))
}

/// Per-connection tail state: which run we're following and how far into its
/// heartbeat file we've read.
struct LiveTail {
    state: AppState,
    /// The run id currently being tailed (`None` when idle).
    current_id: Option<String>,
    /// Bytes consumed from the current heartbeat file.
    offset: u64,
    /// Buffered trailing bytes that did not yet end in a newline.
    partial: Vec<u8>,
}

impl LiveTail {
    fn new(state: AppState) -> Self {
        Self {
            state,
            current_id: None,
            offset: 0,
            partial: Vec::new(),
        }
    }

    /// One poll step: emit a `run` event on a new run, a `heartbeat` event when the
    /// file grew, an `idle` event once when a run ends, else `None` (keep polling).
    fn step(&mut self) -> Option<Event> {
        let live = self.state.live.lock().expect("live mutex").clone();
        match live {
            None => {
                // A run just ended → announce idle once, then stay quiet.
                if self.current_id.take().is_some() {
                    self.reset();
                    return Some(Event::default().event("idle").data("{}"));
                }
                None
            }
            Some(run) => {
                if self.current_id.as_deref() != Some(run.id.as_str()) {
                    // New run in flight → reset the tail and announce it.
                    self.current_id = Some(run.id.clone());
                    self.reset();
                    let meta = serde_json::to_string(&run).unwrap_or_else(|_| "{}".to_string());
                    return Some(Event::default().event("run").data(meta));
                }
                // Same run — stream the newest heartbeat snapshot if the file grew.
                self.latest_line(&run.heartbeat_path)
                    .map(|line| Event::default().event("heartbeat").data(line))
            }
        }
    }

    fn reset(&mut self) {
        self.offset = 0;
        self.partial.clear();
    }

    /// Read any bytes appended since `offset` and return the LAST complete line (each
    /// heartbeat is a full cumulative snapshot, so earlier appended lines are
    /// superseded). Buffers a trailing partial line for the next poll. `None` when
    /// the file is absent or produced no new complete line.
    fn latest_line(&mut self, path: &str) -> Option<String> {
        let mut file = std::fs::File::open(path).ok()?;
        file.seek(SeekFrom::Start(self.offset)).ok()?;
        let mut fresh = Vec::new();
        let read = file.read_to_end(&mut fresh).ok()?;
        if read == 0 {
            return None;
        }
        self.offset += read as u64;
        self.partial.extend_from_slice(&fresh);
        let last_newline = self.partial.iter().rposition(|&b| b == b'\n')?;
        // Everything up to and including the last newline is complete; keep the rest.
        let remainder = self.partial.split_off(last_newline + 1);
        let complete = std::mem::replace(&mut self.partial, remainder);
        // The newest non-empty line in the complete region.
        let line = complete
            .split(|&b| b == b'\n')
            .rev()
            .find(|l| !l.is_empty())?;
        Some(String::from_utf8_lossy(line).into_owned())
    }
}
