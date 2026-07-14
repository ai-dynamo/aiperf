// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Live measurement observer that tees into the adaptive window sampler.

use std::rc::Rc;

use crate::clock::Clock;
use loadgen_core::collector::ReplayTerminalStatus;
use loadgen_core::sink::{ObservedTokenKind, ObservedUsage, RequestObserver};
use uuid::Uuid;

use crate::adaptive_core::runtime::SharedWindowSampler;

/// Forwards every measurement event to the ordinary run observer while feeding
/// returned-request state to a [`crate::adaptive_core::WindowSampler`].
///
/// The observer converts the sink's relative millisecond timestamps back onto
/// the injected clock's integer-nanosecond timeline. Terminal time is sampled
/// directly from that same clock because the transport-neutral observer seam
/// carries terminal status but no terminal timestamp.
pub struct AdaptiveObserver {
    delegate: Rc<dyn RequestObserver>,
    sampler: SharedWindowSampler,
    clock: Rc<dyn Clock>,
    start_ns: i64,
}

impl AdaptiveObserver {
    /// Build a tee observer sharing the run clock and its relative-time origin.
    pub fn new(
        delegate: Rc<dyn RequestObserver>,
        sampler: SharedWindowSampler,
        clock: Rc<dyn Clock>,
        start_ns: i64,
    ) -> Self {
        Self {
            delegate,
            sampler,
            clock,
            start_ns,
        }
    }

    fn absolute_ns(&self, relative_ms: f64) -> i64 {
        if !relative_ms.is_finite() {
            return self.clock.now_ns();
        }
        self.start_ns
            .saturating_add((relative_ms * 1_000_000.0).round() as i64)
    }
}

impl RequestObserver for AdaptiveObserver {
    fn on_arrival(
        &self,
        uuid: Uuid,
        arrival_ms: f64,
        input_length: usize,
        requested_output_length: usize,
    ) {
        self.delegate
            .on_arrival(uuid, arrival_ms, input_length, requested_output_length);
        self.sampler
            .borrow_mut()
            .on_arrival(uuid, self.absolute_ns(arrival_ms));
    }

    fn on_admit(&self, uuid: Uuid, admit_ms: f64, reused_input_tokens: usize) {
        self.delegate.on_admit(uuid, admit_ms, reused_input_tokens);
        self.sampler
            .borrow_mut()
            .on_admit(uuid, self.absolute_ns(admit_ms));
    }

    fn on_token(&self, uuid: Uuid, at_ms: f64) {
        self.delegate.on_token(uuid, at_ms);
        self.sampler
            .borrow_mut()
            .on_token(uuid, self.absolute_ns(at_ms));
    }

    fn on_classified_token(&self, uuid: Uuid, at_ms: f64, kind: ObservedTokenKind) {
        self.delegate.on_classified_token(uuid, at_ms, kind);
        self.sampler
            .borrow_mut()
            .on_token(uuid, self.absolute_ns(at_ms));
    }

    fn on_output_tokens(&self, uuid: Uuid, at_ms: &[f64]) {
        self.delegate.on_output_tokens(uuid, at_ms);
        let mut sampler = self.sampler.borrow_mut();
        for &timestamp in at_ms {
            sampler.on_token(uuid, self.absolute_ns(timestamp));
        }
    }

    fn on_usage(&self, uuid: Uuid, usage: ObservedUsage) {
        self.delegate.on_usage(uuid, usage);
        self.sampler.borrow_mut().on_usage(uuid, usage);
    }

    fn on_terminal(&self, uuid: Uuid, status: ReplayTerminalStatus) {
        self.delegate.on_terminal(uuid, status);
        self.sampler
            .borrow_mut()
            .on_terminal(uuid, status, self.clock.now_ns());
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;

    use crate::clock::SimClock;

    use super::*;
    use crate::adaptive_core::window::TumblingWindowSampler;

    #[derive(Default)]
    struct CountObserver(RefCell<usize>);

    impl RequestObserver for CountObserver {
        fn on_arrival(&self, _: Uuid, _: f64, _: usize, _: usize) {}
        fn on_admit(&self, _: Uuid, _: f64, _: usize) {}
        fn on_token(&self, _: Uuid, _: f64) {
            *self.0.borrow_mut() += 1;
        }
        fn on_terminal(&self, _: Uuid, _: ReplayTerminalStatus) {}
    }

    #[test]
    fn forwards_and_joins_relative_token_times_on_the_clock_timeline() {
        let clock = Rc::new(SimClock::new());
        let sampler: SharedWindowSampler =
            Rc::new(RefCell::new(Box::new(TumblingWindowSampler::new(100))));
        let delegate = Rc::new(CountObserver::default());
        let observer = AdaptiveObserver::new(delegate.clone(), sampler.clone(), clock.clone(), 100);
        let uuid = Uuid::new_v4();
        observer.on_arrival(uuid, 0.0, 1, 2);
        observer.on_token(uuid, 1.0);
        clock.advance_to(2_000_100);
        observer.on_terminal(uuid, ReplayTerminalStatus::Completed);
        let stats = sampler.borrow_mut().take(clock.now_ns());
        assert_eq!(*delegate.0.borrow(), 1);
        assert_eq!(stats.successful_requests[0].ttft_ns, Some(1_000_000));
        assert_eq!(stats.successful_requests[0].request_latency_ns, 1_000_000);
    }

    #[test]
    fn classified_tokens_are_forwarded_and_sampled_exactly_once() {
        let clock = Rc::new(SimClock::new());
        let sampler: SharedWindowSampler =
            Rc::new(RefCell::new(Box::new(TumblingWindowSampler::new(0))));
        let delegate = Rc::new(CountObserver::default());
        let observer = AdaptiveObserver::new(delegate.clone(), sampler.clone(), clock.clone(), 0);
        let uuid = Uuid::new_v4();
        observer.on_arrival(uuid, 0.0, 1, 1);
        observer.on_classified_token(uuid, 1.0, ObservedTokenKind::Reasoning);
        observer.on_usage(
            uuid,
            ObservedUsage {
                prompt_tokens: Some(1),
                completion_tokens: Some(1),
                ..ObservedUsage::default()
            },
        );
        clock.advance_to(2_000_000);
        observer.on_terminal(uuid, ReplayTerminalStatus::Completed);

        let stats = sampler.borrow_mut().take(clock.now_ns());
        assert_eq!(*delegate.0.borrow(), 1);
        assert_eq!(stats.successful_requests[0].output_sequence_length, Some(1));
        assert_eq!(stats.successful_requests[0].ttft_ns, Some(1_000_000));
    }
}
