// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Worker-local measurement plumbing shared by every `RequestExecutor` sink.
//!
//! Each worker owns one [`NativeMetricsObserver`]. It accumulates arrival,
//! admission, token, usage, terminal, and response facts into one
//! [`RecordIngest`] per request.

use std::cell::RefCell;
use std::future::Future;
use std::rc::Rc;

use anyhow::{Result, anyhow};
use uuid::Uuid;

use crate::clock::Clock;
use crate::dispatch::collector::ReplayTerminalStatus;
use crate::dispatch::sink::RequestObserver;
use crate::metrics::{NativeMetricsObserver, NativeResponseMetadata};
use crate::metrics_core::{MetricsConfig, RecordIngest};
use crate::transport::core::{DispatchResult, MeasuredContext};

/// A worker-local metric accumulator.
#[derive(Default)]
pub(crate) struct WorkerMeasurement {
    cell: RefCell<Option<Rc<NativeMetricsObserver>>>,
}

impl WorkerMeasurement {
    /// Install an observer for this worker's run.
    pub(crate) fn configure(&self, clock: Rc<dyn Clock>, config: MetricsConfig, origin_ns: i64) {
        let observer = NativeMetricsObserver::new(clock, origin_ns, config);
        *self.cell.borrow_mut() = Some(Rc::new(observer));
    }

    /// Access the worker-local observer, erroring if the measured execution path
    /// is used before `configure`.
    pub(crate) fn observer(&self) -> Result<Rc<NativeMetricsObserver>> {
        self.cell
            .borrow()
            .clone()
            .ok_or_else(|| anyhow!("worker-local measurement was not configured before dispatch"))
    }

    /// Finalize and remove the observer at `end_ns`.
    pub(crate) fn drain(&self, end_ns: i64) -> Vec<(Uuid, RecordIngest)> {
        match self.cell.borrow_mut().take() {
            Some(observer) => {
                observer
                    .take_finalizer_at(end_ns)
                    .finish_with_records()
                    .records
            }
            None => Vec::new(),
        }
    }
}

/// Derive the optional live record for a just-dispatched request.
///
/// Metrics-only (sketch) mode moves the record out of the observer so its token
/// storage is freed as the run streams; every other mode clones it and leaves
/// the authoritative copy for the drain.
pub(crate) fn live_record(
    observer: &NativeMetricsObserver,
    uuid: Uuid,
    context: &MeasuredContext,
) -> Option<RecordIngest> {
    context
        .wants_live_record
        .then(|| {
            if context.consume_record {
                observer.drain_terminal_record(uuid, 0)
            } else {
                observer.snapshot_record(uuid, 0)
            }
        })
        .flatten()
}

/// Register coordinator-known arrival facts on `observer`, drive `dispatch` to
/// terminal, and record the terminal transport facts.
///
/// The observer accumulates the complete record so the end-of-run drain yields
/// one authoritative [`RecordIngest`] per request. `phase`, `session_num`, the
/// global `request_index`, and the credit-issued `admit_ns` are patched onto the
/// drained record coordinator-side; they are intentionally not set here. On a
/// dispatch error the worker still records a complete failed terminal so the
/// drain has one record for this identity (the coordinator's fallback only
/// covers identities no worker ever touched).
pub(crate) async fn measure_dispatch<F>(
    observer: &NativeMetricsObserver,
    clock: &dyn Clock,
    uuid: Uuid,
    context: &MeasuredContext,
    dispatch: F,
) -> Result<DispatchResult>
where
    F: Future<Output = Result<DispatchResult>>,
{
    observer.register_metadata(uuid, context.metadata.clone());
    observer.on_arrival(
        uuid,
        context.arrival_ms,
        context.input_length,
        context.requested_output_length,
    );
    let result = dispatch.await;
    match &result {
        Ok(collected) => {
            let outcome = &collected.outcome;
            observer.record_response(
                uuid,
                NativeResponseMetadata {
                    start_ns: Some(outcome.start_ns),
                    end_ns: Some(outcome.end_ns),
                    prompt_tokens: outcome.prompt_tokens,
                    completion_tokens: outcome.completion_tokens,
                    http: outcome.http,
                },
            );
        }
        Err(_) => {
            let now = clock.now_ns();
            observer.on_terminal(uuid, ReplayTerminalStatus::Failed);
            observer.record_response(
                uuid,
                NativeResponseMetadata {
                    start_ns: Some(now),
                    end_ns: Some(now),
                    ..NativeResponseMetadata::default()
                },
            );
        }
    }
    result
}
