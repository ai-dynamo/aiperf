// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Typed `phases` section of the native `BenchmarkConfig`.
//!
//! Wire shape ported from `src/aiperf/orchestrator/rust_wire.py::_phase`,
//! `_rate_phase`, `_ramp`. Every phase carries a common block (`name`,
//! `exclude_from_results`, `seamless`, plus optional bounds/ramps/cancellation/
//! adaptive) and a `type`-discriminated body. Optionals use `_set_optional`
//! semantics (absent when unset).

use serde::{Deserialize, Serialize};

/// A concurrency/prefill/rate ramp (`_ramp`).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Ramp {
    /// Ramp duration, seconds.
    pub duration: f64,
    /// Ramp strategy id.
    pub strategy: String,
}

/// Post-send cancellation policy.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Cancellation {
    /// Fraction of requests cancelled.
    pub rate: f64,
    /// Delay before cancellation, seconds.
    pub delay: f64,
}

/// Fields common to every phase.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PhaseCommon {
    /// Phase name (`warmup` / `profiling`).
    pub name: String,
    /// Exclude this phase's records from results.
    pub exclude_from_results: bool,
    /// Run seamlessly into the next phase (no drain barrier).
    pub seamless: bool,
    /// Request-count bound (present when set).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub requests: Option<u64>,
    /// Session-count bound (present when set).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sessions: Option<u64>,
    /// Duration bound, seconds (present when set).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub duration: Option<f64>,
    /// Prefill concurrency (present when set).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prefill_concurrency: Option<u32>,
    /// Grace period, seconds (present when set).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub grace_period: Option<f64>,
    /// Concurrency ramp (present when set).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub concurrency_ramp: Option<Ramp>,
    /// Prefill ramp (present when set).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prefill_ramp: Option<Ramp>,
    /// Rate ramp (present when set).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rate_ramp: Option<Ramp>,
    /// Cancellation policy (present when set).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cancellation: Option<Cancellation>,
    /// Agentic cache-warmup duration, seconds (present on scenario configs).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agentic_cache_warmup_duration: Option<f64>,
    // NOTE: `adaptive_scale` is intentionally deferred — its projection is a
    // large nested block (`_adaptive_scale`) exercised only by adaptive configs,
    // added when that path is ported.
}

/// The `type`-discriminated phase body.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum PhaseKind {
    /// Fixed-concurrency arrivals.
    Concurrency {
        /// Concurrent in-flight requests.
        concurrency: u32,
    },
    /// Poisson-distributed request rate.
    Poisson {
        /// Requests per second.
        rate: f64,
        /// Optional concurrency cap.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        concurrency: Option<u32>,
    },
    /// Gamma-distributed request rate.
    Gamma {
        /// Requests per second.
        rate: f64,
        /// Optional concurrency cap.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        concurrency: Option<u32>,
        /// Optional burstiness/smoothness shape.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        smoothness: Option<f64>,
    },
    /// Constant (deterministic) request rate.
    Constant {
        /// Requests per second.
        rate: f64,
        /// Optional concurrency cap.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        concurrency: Option<u32>,
    },
    /// User-centric closed-loop arrivals.
    UserCentric {
        /// Per-user request rate.
        rate: f64,
        /// Number of users.
        users: u32,
        /// Optional concurrency cap.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        concurrency: Option<u32>,
    },
    /// Replay of a fixed timestamp schedule.
    FixedSchedule {
        /// Auto-offset the schedule to start at zero.
        auto_offset: bool,
        /// Optional explicit start offset.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        start_offset: Option<i64>,
        /// Optional explicit end offset.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        end_offset: Option<i64>,
    },
}

/// One typed phase (common block flattened with the discriminated body).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Phase {
    /// Fields common to every phase.
    #[serde(flatten)]
    pub common: PhaseCommon,
    /// The phase-type-specific body.
    #[serde(flatten)]
    pub kind: PhaseKind,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn concurrency_phase_flattens_type_and_common() {
        let p = Phase {
            common: PhaseCommon {
                name: "profiling".into(),
                exclude_from_results: false,
                seamless: false,
                requests: Some(1),
                sessions: None,
                duration: None,
                prefill_concurrency: None,
                grace_period: None,
                concurrency_ramp: None,
                prefill_ramp: None,
                rate_ramp: None,
                cancellation: None,
                agentic_cache_warmup_duration: None,
            },
            kind: PhaseKind::Concurrency { concurrency: 1 },
        };
        let v = serde_json::to_value(&p).unwrap();
        assert_eq!(v["type"], serde_json::json!("concurrency"));
        assert_eq!(v["concurrency"], serde_json::json!(1));
        assert_eq!(v["name"], serde_json::json!("profiling"));
        assert_eq!(v["requests"], serde_json::json!(1));
        assert_eq!(v.get("duration"), None);
    }
}
