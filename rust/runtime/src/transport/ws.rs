// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! WebSocket application-event lag measurement.

use crate::dispatch::sink::ObservedRoundTripMetrics;

/// Constant-size post-flush timing state for one logical WebSocket operation.
#[derive(Debug, Default)]
pub struct RoundTripTimingState {
    last_send_ns: Option<i64>,
    send_timestamp_sum_ns: i128,
    send_count: u64,
    last_content_receive_ns: Option<i64>,
    content_receive_timestamp_sum_ns: i128,
    content_receive_count: u64,
    last_observation_ns: Option<i64>,
    is_invalid: bool,
}

impl RoundTripTimingState {
    /// Record a measured input timestamp immediately after its flush completes.
    pub fn on_measured_input_flushed(&mut self, timestamp_ns: i64) {
        if self.is_invalid
            || timestamp_ns < 0
            || self.last_content_receive_ns.is_some()
            || self
                .last_observation_ns
                .is_some_and(|previous| timestamp_ns < previous)
        {
            self.is_invalid = true;
            return;
        }
        let Some(sum) = self
            .send_timestamp_sum_ns
            .checked_add(i128::from(timestamp_ns))
        else {
            self.is_invalid = true;
            return;
        };
        let Some(count) = self.send_count.checked_add(1) else {
            self.is_invalid = true;
            return;
        };
        self.send_timestamp_sum_ns = sum;
        self.send_count = count;
        self.last_send_ns = Some(timestamp_ns);
        self.last_observation_ns = Some(timestamp_ns);
    }

    /// Record one decoded event carrying non-empty user-visible content.
    pub fn on_content_received(&mut self, timestamp_ns: i64) {
        if self.is_invalid
            || timestamp_ns < 0
            || self
                .last_observation_ns
                .is_some_and(|previous| timestamp_ns < previous)
        {
            self.is_invalid = true;
            return;
        }
        let Some(sum) = self
            .content_receive_timestamp_sum_ns
            .checked_add(i128::from(timestamp_ns))
        else {
            self.is_invalid = true;
            return;
        };
        let Some(count) = self.content_receive_count.checked_add(1) else {
            self.is_invalid = true;
            return;
        };
        self.content_receive_timestamp_sum_ns = sum;
        self.content_receive_count = count;
        self.last_content_receive_ns = Some(timestamp_ns);
        self.last_observation_ns = Some(timestamp_ns);
    }

    /// Finish the two application-event lag estimators, preserving invalidity as absence.
    pub fn finish(&self) -> ObservedRoundTripMetrics {
        if self.is_invalid || self.send_count == 0 || self.content_receive_count == 0 {
            return ObservedRoundTripMetrics::default();
        }
        let (Some(last_send_ns), Some(last_content_receive_ns)) =
            (self.last_send_ns, self.last_content_receive_ns)
        else {
            return ObservedRoundTripMetrics::default();
        };
        let Some(last_send_to_last_content_ns) = last_content_receive_ns.checked_sub(last_send_ns)
        else {
            return ObservedRoundTripMetrics::default();
        };
        if last_send_to_last_content_ns < 0 {
            return ObservedRoundTripMetrics::default();
        }
        let mean_timestamp_lag_ns = self.content_receive_timestamp_sum_ns as f64
            / self.content_receive_count as f64
            - self.send_timestamp_sum_ns as f64 / self.send_count as f64;
        if !mean_timestamp_lag_ns.is_finite() || mean_timestamp_lag_ns < 0.0 {
            return ObservedRoundTripMetrics::default();
        }
        ObservedRoundTripMetrics {
            last_send_to_last_content_ns: Some(last_send_to_last_content_ns),
            mean_timestamp_lag_ns: Some(mean_timestamp_lag_ns),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::RoundTripTimingState;

    #[test]
    fn unequal_populations_use_the_difference_of_timestamp_means() {
        let mut timing = RoundTripTimingState::default();
        timing.on_measured_input_flushed(100);
        timing.on_measured_input_flushed(300);
        timing.on_content_received(500);

        let fact = timing.finish();
        assert_eq!(fact.last_send_to_last_content_ns, Some(200));
        assert_eq!(fact.mean_timestamp_lag_ns, Some(300.0));
    }

    #[test]
    fn missing_or_invalid_ordering_is_absent() {
        assert_eq!(RoundTripTimingState::default().finish(), Default::default());

        let mut timing = RoundTripTimingState::default();
        timing.on_measured_input_flushed(20);
        timing.on_content_received(10);
        assert_eq!(timing.finish(), Default::default());
    }

    #[test]
    fn regressing_send_or_content_timestamp_invalidates_operation() {
        let mut send_regression = RoundTripTimingState::default();
        send_regression.on_measured_input_flushed(300);
        send_regression.on_measured_input_flushed(100);
        send_regression.on_content_received(500);
        assert_eq!(send_regression.finish(), Default::default());

        let mut content_regression = RoundTripTimingState::default();
        content_regression.on_measured_input_flushed(100);
        content_regression.on_content_received(500);
        content_regression.on_content_received(300);
        assert_eq!(content_regression.finish(), Default::default());
    }

    #[test]
    fn measured_input_after_content_invalidates_operation() {
        let mut timing = RoundTripTimingState::default();
        timing.on_measured_input_flushed(300);
        timing.on_content_received(500);
        timing.on_measured_input_flushed(400);

        assert_eq!(timing.finish(), Default::default());
    }
}
