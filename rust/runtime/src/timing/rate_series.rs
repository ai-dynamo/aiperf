// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Piecewise-linear request-rate schedule driven through [`Clock`](crate::clock::Clock).

use std::cell::RefCell;
use std::rc::Rc;

use crate::ancillary::RATE_RAMP_UPDATE_INTERVAL_NS;
use crate::clock::Clock;
use crate::phase_runtime::ScheduledPhaseController;
use crate::timing::IntervalGenerator;
use crate::timing::LocalPhaseFuture;

/// One request-rate control point.
#[derive(Clone, Debug, PartialEq)]
pub struct RateSeriesPoint {
    /// Elapsed seconds from series start (after any rate ramp completes).
    pub time_s: f64,
    /// Request rate in queries per second at this point.
    pub qps: f64,
}

/// Piecewise-linear request-rate schedule.
#[derive(Clone, Debug, PartialEq)]
pub struct RateSeriesSchedule {
    /// Strictly increasing control points (≥2).
    pub points: Vec<RateSeriesPoint>,
}

impl RateSeriesSchedule {
    /// Build from `(time_s, qps)` pairs.
    pub fn from_points(points: impl IntoIterator<Item = (f64, f64)>) -> Self {
        Self {
            points: points
                .into_iter()
                .map(|(time_s, qps)| RateSeriesPoint { time_s, qps })
                .collect(),
        }
    }
}

/// Interpolate the configured request rate at elapsed phase seconds.
pub fn value_at(series: &RateSeriesSchedule, elapsed_sec: f64) -> f64 {
    let times: Vec<f64> = series.points.iter().map(|p| p.time_s).collect();
    let qps: Vec<f64> = series.points.iter().map(|p| p.qps).collect();
    if elapsed_sec <= times[0] {
        return qps[0];
    }
    if elapsed_sec >= *times.last().expect("validated series") {
        return *qps.last().expect("validated series");
    }
    let right = times.partition_point(|time| *time <= elapsed_sec);
    let left = right.saturating_sub(1);
    let left_time = times[left];
    let right_time = times[right];
    let progress = (elapsed_sec - left_time) / (right_time - left_time);
    qps[left] + (qps[right] - qps[left]) * progress
}

/// Clock-driven controller applying a rate series through an interval generator.
pub struct RateSeriesDriver {
    clock: Rc<dyn Clock>,
    series: RateSeriesSchedule,
    intervals: Rc<RefCell<Box<dyn IntervalGenerator>>>,
    start_delay_ns: u64,
}

impl RateSeriesDriver {
    /// Build a driver that updates `intervals` on a fixed cadence after an optional delay.
    pub fn new(
        clock: Rc<dyn Clock>,
        series: RateSeriesSchedule,
        intervals: Rc<RefCell<Box<dyn IntervalGenerator>>>,
        start_delay_ns: u64,
    ) -> Self {
        Self {
            clock,
            series,
            intervals,
            start_delay_ns,
        }
    }

    /// Apply the initial rate synchronously, then spawn the update loop locally.
    pub fn spawn_local(self) -> tokio::task::JoinHandle<()> {
        let initial = value_at(&self.series, 0.0);
        self.intervals.borrow_mut().set_rate(initial);
        tokio::task::spawn_local(async move { self.run().await })
    }

    async fn run(self) {
        if self.start_delay_ns > 0 {
            self.clock.clone().sleep(self.start_delay_ns as i64).await;
        }
        let started_at_ns = self.clock.now_ns();
        let final_time_s = self.series.points.last().map(|p| p.time_s).unwrap_or(0.0);
        loop {
            self.clock
                .clone()
                .sleep(RATE_RAMP_UPDATE_INTERVAL_NS as i64)
                .await;
            let elapsed_sec =
                (self.clock.now_ns().saturating_sub(started_at_ns).max(0) as f64) / 1e9;
            let rate = value_at(&self.series, elapsed_sec);
            self.intervals.borrow_mut().set_rate(rate);
            if elapsed_sec >= final_time_s {
                break;
            }
        }
    }
}

/// Phase controller owning one rate-series background task.
pub struct RateSeriesScheduledPhaseController {
    driver: RefCell<Option<RateSeriesDriver>>,
    handle: RefCell<Option<tokio::task::JoinHandle<()>>>,
}

impl RateSeriesScheduledPhaseController {
    /// Take ownership of the prepared driver for one phase.
    pub fn new(driver: RateSeriesDriver) -> Self {
        Self {
            driver: RefCell::new(Some(driver)),
            handle: RefCell::new(None),
        }
    }
}

impl ScheduledPhaseController for RateSeriesScheduledPhaseController {
    fn start(&self) -> anyhow::Result<()> {
        let driver = self
            .driver
            .borrow_mut()
            .take()
            .ok_or_else(|| anyhow::anyhow!("rate-series controller was already started"))?;
        *self.handle.borrow_mut() = Some(driver.spawn_local());
        Ok(())
    }

    fn stop(&self) -> LocalPhaseFuture<anyhow::Result<()>> {
        self.driver.borrow_mut().take();
        let handle = self.handle.borrow_mut().take();
        Box::pin(async move {
            if let Some(handle) = handle {
                handle.abort();
                let _ = handle.await;
            }
            Ok(())
        })
    }

    // Intentionally inherits the default `pending` wait_until_stop: rate series
    // only updates arrival rate and never independently ends issuance. An
    // immediately-ready future would cancel the workload in
    // `ScheduledPhaseExecution::execute`'s select.
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::timing::{ArrivalPattern, make_interval_generator};

    fn sample_series() -> RateSeriesSchedule {
        RateSeriesSchedule::from_points([(10.0, 5.0), (20.0, 15.0), (30.0, 10.0)])
    }

    #[test]
    fn value_at_interpolates_and_holds_edges() {
        let series = sample_series();
        assert!((value_at(&series, 0.0) - 5.0).abs() < f64::EPSILON);
        assert!((value_at(&series, 10.0) - 5.0).abs() < f64::EPSILON);
        assert!((value_at(&series, 15.0) - 10.0).abs() < f64::EPSILON);
        assert!((value_at(&series, 20.0) - 15.0).abs() < f64::EPSILON);
        assert!((value_at(&series, 25.0) - 12.5).abs() < f64::EPSILON);
        assert!((value_at(&series, 40.0) - 10.0).abs() < f64::EPSILON);
    }

    #[test]
    fn driver_applies_initial_rate_before_spawn() {
        let intervals = Rc::new(RefCell::new(make_interval_generator(
            ArrivalPattern::Poisson,
            Some(1.0),
            None,
            0,
        )));
        let initial = value_at(&sample_series(), 0.0);
        intervals.borrow_mut().set_rate(initial);
        assert!((intervals.borrow().rate() - 5.0).abs() < f64::EPSILON);
    }
}
