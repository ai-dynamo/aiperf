// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Application wiring for ancillary timing policy.
//!
//! Curve math and policy traits live in [`crate::timing`]; this module owns only
//! CLI/runtime composition: phase ramp durations, the canonical 100ms rate-ramp
//! cadence, seeded cancellation construction, and comma-separated endpoint
//! normalization.

use crate::timing::{BernoulliFixedDelay, CancellationPolicy, RoundRobinUrlSelector, UrlSelector};
use anyhow::{Result, bail};

/// Default rate-ramp update interval, in nanoseconds.
pub const RATE_RAMP_UPDATE_INTERVAL_NS: u64 = 100_000_000;

/// Ancillary policy configuration for one run phase.
#[derive(Clone, Debug)]
pub struct AncillaryTimingConfig {
    /// Duration of the discrete session-concurrency ramp from one to target.
    pub concurrency_ramp_duration_ns: Option<u64>,
    /// Duration of the discrete prefill-concurrency ramp from one to target.
    pub prefill_concurrency_ramp_duration_ns: Option<u64>,
    /// Duration of the continuous request-rate ramp to target.
    pub request_rate_ramp_duration_ns: Option<u64>,
    /// Continuous request-rate update cadence.
    pub rate_ramp_update_interval_ns: u64,
    /// Percentage of profiling requests selected for client disconnect.
    pub cancellation_rate_percent: Option<f64>,
    /// Fixed delay after send completion for selected requests.
    pub cancellation_delay_ns: i64,
}

impl Default for AncillaryTimingConfig {
    fn default() -> Self {
        Self {
            concurrency_ramp_duration_ns: None,
            prefill_concurrency_ramp_duration_ns: None,
            request_rate_ramp_duration_ns: None,
            rate_ramp_update_interval_ns: RATE_RAMP_UPDATE_INTERVAL_NS,
            cancellation_rate_percent: None,
            cancellation_delay_ns: 0,
        }
    }
}

impl AncillaryTimingConfig {
    /// Validate duration and delay fields. Percentage validation remains owned
    /// by the concrete cancellation policy constructor.
    pub fn validate(&self) -> Result<()> {
        for (name, duration) in [
            (
                "concurrency ramp duration",
                self.concurrency_ramp_duration_ns,
            ),
            (
                "prefill concurrency ramp duration",
                self.prefill_concurrency_ramp_duration_ns,
            ),
            (
                "request-rate ramp duration",
                self.request_rate_ramp_duration_ns,
            ),
            (
                "rate-ramp update interval",
                Some(self.rate_ramp_update_interval_ns),
            ),
        ] {
            if duration == Some(0) || duration.is_some_and(|value| value > i64::MAX as u64) {
                bail!("{name} must be in 1..={}ns", i64::MAX);
            }
        }
        if self.cancellation_delay_ns < 0 {
            bail!("cancellation delay must be non-negative");
        }
        // Construct once for validation without exposing a second set of rate rules.
        let _ = BernoulliFixedDelay::from_delay_ns_seed(
            self.cancellation_rate_percent,
            self.cancellation_delay_ns,
            Some(0),
        )?;
        Ok(())
    }

    /// Whether at least one actuator ramp is configured.
    pub fn has_ramps(&self) -> bool {
        self.concurrency_ramp_duration_ns.is_some()
            || self.prefill_concurrency_ramp_duration_ns.is_some()
            || self.request_rate_ramp_duration_ns.is_some()
    }

    /// Build a seeded cancellation policy, omitting the disabled zero/None case.
    pub fn cancellation_policy(&self, seed: u64) -> Result<Option<Box<dyn CancellationPolicy>>> {
        let policy = BernoulliFixedDelay::from_delay_ns_seed(
            self.cancellation_rate_percent,
            self.cancellation_delay_ns,
            Some(seed),
        )?;
        Ok(policy
            .is_enabled()
            .then_some(Box::new(policy) as Box<dyn CancellationPolicy>))
    }
}

/// Parse a comma-separated positional base URL into a non-empty ordered list.
pub fn parse_base_urls(value: &str) -> Result<Vec<String>> {
    let urls: Vec<_> = value
        .split(',')
        .map(str::trim)
        .filter(|url| !url.is_empty())
        .map(ToString::to_string)
        .collect();
    if urls.is_empty() {
        bail!("at least one non-empty base URL is required");
    }
    Ok(urls)
}

/// Build the default round-robin selector only when more than one URL exists.
pub fn url_selector(urls: &[String]) -> Result<Option<Box<dyn UrlSelector>>> {
    if urls.is_empty() {
        bail!("at least one endpoint is required for URL selection");
    }
    if urls.len() == 1 {
        return Ok(None);
    }
    if urls.len() > u32::MAX as usize {
        bail!("endpoint count exceeds the u32 request-index representation");
    }
    Ok(Some(Box::new(RoundRobinUrlSelector::new(urls.to_vec())?)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_and_trims_multiple_urls() {
        assert_eq!(
            parse_base_urls(" http://a ,http://b ").unwrap(),
            vec!["http://a", "http://b"]
        );
        assert!(parse_base_urls(" , ").is_err());
    }

    #[test]
    fn selector_is_only_created_for_multi_url_runs() {
        assert!(url_selector(&[]).is_err());
        assert!(url_selector(&["a".into()]).unwrap().is_none());
        let mut selector = url_selector(&["a".into(), "b".into()]).unwrap().unwrap();
        assert_eq!(selector.next_index(), 0);
        assert_eq!(selector.next_index(), 1);
    }

    #[test]
    fn validates_all_nonzero_durations_and_cancellation() {
        let mut config = AncillaryTimingConfig {
            concurrency_ramp_duration_ns: Some(0),
            ..AncillaryTimingConfig::default()
        };
        assert!(config.validate().is_err());
        config.concurrency_ramp_duration_ns = Some(1);
        config.cancellation_rate_percent = Some(101.0);
        assert!(config.validate().is_err());
    }
}
