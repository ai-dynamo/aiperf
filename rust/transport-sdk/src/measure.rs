// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Worker-local measurement plumbing shared by every request-transport sink.
//!
//! Each worker owns one observer and accumulates arrival, admission, token,
//! usage, and terminal facts into it. Accumulating per worker and merging once at
//! the drain boundary is what keeps the per-request and per-token paths free of
//! shared-state contention.
//!
//! [`measure_dispatch`] is applied **exactly once** per request, by the execution
//! capsule, around the sink's dispatch. A sink that also wrapped itself would
//! register arrival twice and record two terminals, changing observed TTFT and
//! latency.

use std::cell::RefCell;
use std::future::Future;
use std::rc::Rc;

use aiperf_core::dispatch::{ReplayTerminalStatus, RequestObserver};
use anyhow::{Result, anyhow};
use uuid::Uuid;

use crate::direct::WorkerRequest;

/// The coordinator-known facts registered before a request is dispatched.
#[derive(Clone, Copy, Debug, Default)]
pub struct ArrivalFacts {
    /// Per-request measurement identity.
    pub uuid: Uuid,
    /// Run-relative arrival instant in milliseconds.
    pub arrival_ms: f64,
    /// Prompt length in tokens.
    pub input_length: usize,
    /// Requested output length in tokens.
    pub requested_output_length: usize,
}

impl ArrivalFacts {
    /// Project the arrival facts a worker request already carries.
    pub fn from_request(request: &WorkerRequest) -> Self {
        Self {
            uuid: request.uuid(),
            arrival_ms: 0.0,
            input_length: request.input_length(),
            requested_output_length: request.max_output_tokens(),
        }
    }

    /// Stamp the run-relative arrival instant.
    #[must_use]
    pub const fn at_ms(mut self, arrival_ms: f64) -> Self {
        self.arrival_ms = arrival_ms;
        self
    }
}

/// A worker-local metric accumulator, generic over the host's observer type.
///
/// Generic rather than holding a concrete observer so no host-private
/// measurement type is named from a plugin. The observer is installed once per
/// run and taken at drain.
pub struct WorkerMeasurement<O> {
    cell: RefCell<Option<Rc<O>>>,
}

impl<O> Default for WorkerMeasurement<O> {
    fn default() -> Self {
        Self {
            cell: RefCell::new(None),
        }
    }
}

impl<O> WorkerMeasurement<O> {
    /// Install this worker's observer for the run.
    pub fn configure(&self, observer: Rc<O>) {
        *self.cell.borrow_mut() = Some(observer);
    }

    /// Access the worker-local observer.
    ///
    /// Errors rather than panics when the measured path is entered before
    /// `configure`: a misordered bootstrap is a run-configuration bug the caller
    /// can report, not an invariant this crate may abort the process on.
    pub fn observer(&self) -> Result<Rc<O>> {
        self.cell
            .borrow()
            .clone()
            .ok_or_else(|| anyhow!("worker-local measurement was not configured before dispatch"))
    }

    /// Whether an observer is installed.
    pub fn is_configured(&self) -> bool {
        self.cell.borrow().is_some()
    }

    /// Take the observer at end of run, leaving this worker unconfigured.
    pub fn take(&self) -> Option<Rc<O>> {
        self.cell.borrow_mut().take()
    }
}

/// Register arrival, drive one dispatch to terminal, and record the outcome.
///
/// On a dispatch error the worker still records a complete failed terminal, so
/// the drain has one record for this identity. A coordinator-side fallback only
/// covers identities no worker ever touched, so dropping the terminal here would
/// silently lose the request from the record set.
///
/// The terminal *instant* is not stamped here: on success it comes from the
/// transport's own measured window, and on failure the observer's own clock
/// supplies it. Reading a second clock here would be a second time source.
pub async fn measure_dispatch<F, T>(
    observer: &dyn RequestObserver,
    arrival: ArrivalFacts,
    dispatch: F,
) -> Result<T>
where
    F: Future<Output = Result<T>>,
{
    observer.on_arrival(
        arrival.uuid,
        arrival.arrival_ms,
        arrival.input_length,
        arrival.requested_output_length,
    );
    let result = dispatch.await;
    if result.is_err() {
        observer.on_terminal(arrival.uuid, ReplayTerminalStatus::Failed);
    }
    result
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;

    use super::*;

    #[derive(Default)]
    struct RecordingObserver {
        arrivals: RefCell<Vec<Uuid>>,
        terminals: RefCell<Vec<(Uuid, ReplayTerminalStatus)>>,
    }

    impl RequestObserver for RecordingObserver {
        fn on_arrival(&self, uuid: Uuid, _ms: f64, _input: usize, _requested: usize) {
            self.arrivals.borrow_mut().push(uuid);
        }
        fn on_admit(&self, _uuid: Uuid, _ms: f64, _reused: usize) {}
        fn on_token(&self, _uuid: Uuid, _at_ms: f64) {}
        fn on_terminal(&self, uuid: Uuid, status: ReplayTerminalStatus) {
            self.terminals.borrow_mut().push((uuid, status));
        }
    }

    /// Minimal executor: these futures never yield to a reactor.
    fn drive<T>(future: impl Future<Output = T>) -> T {
        use std::task::{Context, Poll, Wake, Waker};
        struct NoopWake;
        impl Wake for NoopWake {
            fn wake(self: std::sync::Arc<Self>) {}
        }
        let waker = Waker::from(std::sync::Arc::new(NoopWake));
        let mut context = Context::from_waker(&waker);
        let mut future = Box::pin(future);
        loop {
            match future.as_mut().poll(&mut context) {
                Poll::Ready(value) => return value,
                Poll::Pending => std::thread::yield_now(),
            }
        }
    }

    #[test]
    fn success_registers_arrival_and_no_synthetic_terminal() {
        let observer = RecordingObserver::default();
        let facts = ArrivalFacts {
            uuid: Uuid::from_u128(7),
            ..ArrivalFacts::default()
        };
        let outcome: Result<u8> = drive(measure_dispatch(&observer, facts, async { Ok(9u8) }));
        assert_eq!(outcome.unwrap(), 9);
        assert_eq!(observer.arrivals.borrow().as_slice(), &[Uuid::from_u128(7)]);
        assert!(observer.terminals.borrow().is_empty());
    }

    #[test]
    fn dispatch_error_still_records_a_failed_terminal() {
        let observer = RecordingObserver::default();
        let facts = ArrivalFacts {
            uuid: Uuid::from_u128(11),
            ..ArrivalFacts::default()
        };
        let outcome: Result<u8> = drive(measure_dispatch(&observer, facts, async {
            Err(anyhow!("connect refused"))
        }));
        assert!(outcome.is_err());
        assert_eq!(
            observer.terminals.borrow().as_slice(),
            &[(Uuid::from_u128(11), ReplayTerminalStatus::Failed)]
        );
    }

    #[test]
    fn measurement_reports_before_configure_instead_of_panicking() {
        let measurement: WorkerMeasurement<RecordingObserver> = WorkerMeasurement::default();
        assert!(!measurement.is_configured());
        assert!(measurement.observer().is_err());
        measurement.configure(Rc::new(RecordingObserver::default()));
        assert!(measurement.observer().is_ok());
        assert!(measurement.take().is_some());
        assert!(measurement.observer().is_err());
    }
}
