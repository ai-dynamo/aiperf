// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Reusable timing assertions and profile-argument builders for native end-to-end tests.
//!
//! Python's timing harness inspected the credit bus directly. The native execution
//! path intentionally has no test-only credit bus; every terminal
//! `profile_export.jsonl` record is instead the authoritative completed-credit
//! observation. Its metadata timestamps share the wall-clock timeline required for
//! concurrency and per-session checks, so these helpers do not consume raw exports.

use std::collections::{BTreeMap, HashMap};

use serde_json::Value;

/// Default model used by timing profile-argument builders.
pub const DEFAULT_MODEL: &str = "openai/gpt-oss-120b";

/// A completed profile whose `profile_export.jsonl` records can be inspected.
pub trait ProfileExport {
    /// Load the terminal `profile_export.jsonl` records.
    fn profile_export_records(&self) -> Vec<Value>;
}

/// Default deterministic seed for timing tests with a stochastic arrival pattern.
pub const DEFAULT_RANDOM_SEED: u64 = 42;

/// Latencies used to predict whether a timing-test configuration exercises a cap.
#[derive(Debug, Clone, Copy)]
pub struct RealisticLatencyConfig {
    /// Time to first token in milliseconds.
    pub ttft_ms: f64,
    /// Inter-token latency in milliseconds.
    pub itl_ms: f64,
}

impl Default for RealisticLatencyConfig {
    fn default() -> Self {
        Self {
            ttft_ms: 5.0,
            itl_ms: 1.0,
        }
    }
}

impl RealisticLatencyConfig {
    /// Expected request duration for an output of `osl` tokens.
    pub fn request_duration_secs(self, osl: u32) -> f64 {
        (self.ttft_ms + f64::from(osl) * self.itl_ms) / 1_000.0
    }

    /// Expected steady-state total concurrency at `qps`.
    pub fn expected_max_concurrent(self, qps: f64, osl: u32) -> f64 {
        if qps <= 0.0 {
            f64::INFINITY
        } else {
            qps * self.request_duration_secs(osl)
        }
    }

    /// Expected steady-state prefill concurrency at `qps`.
    pub fn expected_max_prefill_concurrent(self, qps: f64) -> f64 {
        if qps <= 0.0 {
            f64::INFINITY
        } else {
            qps * self.ttft_ms / 1_000.0
        }
    }
}

/// A timing scenario used by the profile-argument builders and cap predictors.
#[derive(Debug, Clone)]
pub struct TimingTestConfig {
    /// Number of conversations to run.
    pub num_sessions: u32,
    /// Requested arrival rate. Zero selects burst behavior.
    pub qps: f64,
    /// Fixed turns in each conversation.
    pub turns_per_session: u32,
    /// Optional total in-flight cap.
    pub concurrency: Option<u32>,
    /// Optional prefill in-flight cap.
    pub prefill_concurrency: Option<u32>,
    /// Requested output length.
    pub osl: u32,
    /// Model latency used only for test-design predictions.
    pub latency: RealisticLatencyConfig,
}

impl TimingTestConfig {
    /// Construct a single-turn scenario with the Python harness defaults.
    pub fn new(num_sessions: u32, qps: f64) -> Self {
        Self {
            num_sessions,
            qps,
            turns_per_session: 1,
            concurrency: None,
            prefill_concurrency: None,
            osl: 50,
            latency: RealisticLatencyConfig::default(),
        }
    }

    /// Total expected terminal records for a session-bounded run.
    pub fn expected_requests(&self) -> u32 {
        self.num_sessions * self.turns_per_session
    }

    /// Expected request gap in seconds, when rate-limited.
    pub fn expected_gap_secs(&self) -> Option<f64> {
        (self.qps > 0.0).then(|| 1.0 / self.qps)
    }

    /// Expected per-user gap in seconds for user-centric timing.
    pub fn expected_user_gap_secs(&self) -> Option<f64> {
        (self.qps > 0.0).then(|| f64::from(self.num_sessions) / self.qps)
    }

    /// Predicted total concurrency before an authored cap applies.
    pub fn expected_max_concurrent(&self) -> f64 {
        self.latency.expected_max_concurrent(self.qps, self.osl)
    }

    /// Predicted prefill concurrency before an authored cap applies.
    pub fn expected_max_prefill_concurrent(&self) -> f64 {
        self.latency.expected_max_prefill_concurrent(self.qps)
    }

    /// Whether this configuration should saturate its total-concurrency cap.
    pub fn will_hit_concurrency_limit(&self) -> bool {
        let Some(limit) = self.concurrency else {
            return false;
        };
        if self.qps <= 0.0 {
            self.num_sessions > limit
        } else {
            self.expected_max_concurrent() >= f64::from(limit)
        }
    }

    /// Whether this configuration should saturate its prefill-concurrency cap.
    pub fn will_hit_prefill_limit(&self) -> bool {
        let Some(limit) = self.prefill_concurrency else {
            return false;
        };
        if self.qps <= 0.0 {
            self.num_sessions > limit
        } else {
            self.expected_max_prefill_concurrent() >= f64::from(limit)
        }
    }
}

/// Optional timing-mode arguments for [`build_timing_command`].
#[derive(Debug, Clone)]
pub struct TimingCommandOptions<'a> {
    /// Native arrival pattern such as `constant`, `poisson`, or `gamma`.
    pub arrival_pattern: Option<&'a str>,
    /// Select user-centric timing at this rate.
    pub user_centric_rate: Option<f64>,
    /// Deterministic seed; `None` leaves the run unseeded.
    pub random_seed: Option<u64>,
    /// Additional, already shell-split-compatible profile arguments.
    pub extra_args: &'a str,
}

impl Default for TimingCommandOptions<'_> {
    fn default() -> Self {
        Self {
            arrival_pattern: None,
            user_centric_rate: None,
            random_seed: Some(DEFAULT_RANDOM_SEED),
            extra_args: "",
        }
    }
}

/// Build profile arguments consumable by either native harness, not a shell command.
///
/// This is the native equivalent of Python's `build_timing_command`; the
/// harness itself supplies the `aiperf profile` prefix.
pub fn build_timing_command(
    config: &TimingTestConfig,
    options: TimingCommandOptions<'_>,
) -> String {
    let mut args = format!(
        "--model {DEFAULT_MODEL} --streaming --output-tokens-mean {} --ui none",
        config.osl
    );
    let turns = options
        .user_centric_rate
        .map_or(config.turns_per_session, |_| {
            config.turns_per_session.max(2)
        });
    if turns > 1 {
        args.push_str(&format!(
            " --session-turns-mean {turns} --session-turns-stddev 0"
        ));
    }
    if let Some(limit) = config.concurrency {
        args.push_str(&format!(" --concurrency {limit}"));
    }
    if let Some(limit) = config.prefill_concurrency {
        args.push_str(&format!(" --prefill-concurrency {limit}"));
    }
    if let Some(rate) = options.user_centric_rate {
        args.push_str(&format!(
            " --num-users {} --user-centric-rate {rate} --benchmark-duration 1.0 --benchmark-grace-period 0.0",
            config.num_sessions
        ));
    } else {
        args.push_str(&format!(" --num-sessions {}", config.num_sessions));
        if config.qps > 0.0 {
            args.push_str(&format!(" --request-rate {}", config.qps));
            if let Some(pattern) = options.arrival_pattern {
                args.push_str(&format!(" --arrival-pattern {pattern}"));
            }
        }
    }
    if let Some(seed) = options.random_seed {
        args.push_str(&format!(" --random-seed {seed}"));
    }
    if !options.extra_args.is_empty() {
        args.push(' ');
        args.push_str(options.extra_args);
    }
    args
}

/// Build burst-mode profile arguments consumable by either native harness.
///
/// This is the native equivalent of Python's `build_burst_command`; the
/// harness itself supplies the `aiperf profile` prefix.
pub fn build_burst_command(config: &TimingTestConfig) -> String {
    let mut args = format!(
        "--model {DEFAULT_MODEL} --streaming --num-sessions {} --output-tokens-mean {} --ui none",
        config.num_sessions, config.osl
    );
    if let Some(limit) = config.concurrency {
        args.push_str(&format!(" --concurrency {limit}"));
    }
    if config.turns_per_session > 1 {
        args.push_str(&format!(
            " --session-turns-mean {} --session-turns-stddev 0",
            config.turns_per_session
        ));
    }
    if let Some(limit) = config.prefill_concurrency {
        args.push_str(&format!(" --prefill-concurrency {limit}"));
    }
    args
}

/// Assert that the completed request count matches `expected`.
pub fn assert_request_count<R: ProfileExport>(
    result: &R,
    expected: usize,
    message: &str,
) -> Result<(), String> {
    let actual = result.profile_export_records().len();
    (actual == expected).then_some(()).ok_or_else(|| {
        let context = (!message.is_empty())
            .then(|| format!("{message}: "))
            .unwrap_or_default();
        format!("{context}expected {expected} requests, got {actual}")
    })
}

/// Assert every issued credit reached a terminal profile-export record.
pub fn assert_credits_balanced<R: ProfileExport>(result: &R) -> Result<(), String> {
    let rows = result.profile_export_records();
    if rows.is_empty() {
        return Err(
            "no terminal profile-export records available to validate credit balance".to_string(),
        );
    }
    let incomplete = rows
        .iter()
        .filter(|row| raw_i64(row, "request_end_ns").is_none())
        .count();
    (incomplete == 0).then_some(()).ok_or_else(|| {
        format!(
            "{incomplete} of {} issued credits have no terminal request_end_ns",
            rows.len()
        )
    })
}

/// Assert the maximum request or prefill overlap never exceeds `limit`.
pub fn assert_concurrency_limit_respected<R: ProfileExport>(
    result: &R,
    limit: usize,
    prefill: bool,
) -> Result<(), String> {
    let peak = peak_concurrency(&result.profile_export_records(), prefill)?;
    (peak <= limit).then_some(()).ok_or_else(|| {
        format!(
            "max {} concurrency {peak} exceeded limit {limit}",
            if prefill { "prefill" } else { "total" }
        )
    })
}

/// Assert the configured request or prefill cap was exercised.
pub fn assert_concurrency_limit_hit<R: ProfileExport>(
    result: &R,
    limit: usize,
    prefill: bool,
) -> Result<(), String> {
    let peak = peak_concurrency(&result.profile_export_records(), prefill)?;
    (peak == limit).then_some(()).ok_or_else(|| {
        format!(
            "max {} concurrency {peak} did not reach limit {limit}",
            if prefill { "prefill" } else { "total" }
        )
    })
}

/// Assert worker counts remain within `tolerance_pct` of a perfect split.
pub fn assert_fair_load_distribution<R: ProfileExport>(
    result: &R,
    tolerance_pct: f64,
) -> Result<(), String> {
    let mut counts: HashMap<String, usize> = HashMap::new();
    for row in result.profile_export_records() {
        let Some(worker) = row.pointer("/metadata/worker_id").and_then(Value::as_str) else {
            continue;
        };
        *counts.entry(worker.to_owned()).or_default() += 1;
    }
    if counts.len() < 2 {
        return Err(format!(
            "need records from at least two workers, got {}",
            counts.len()
        ));
    }
    let mean = counts.values().sum::<usize>() as f64 / counts.len() as f64;
    let permitted = mean * tolerance_pct / 100.0;
    let unfair: Vec<_> = counts
        .iter()
        .filter(|(_, count)| (**count as f64 - mean).abs() > permitted)
        .collect();
    unfair.is_empty().then_some(()).ok_or_else(|| {
        format!(
            "unfair worker distribution: {counts:?}, mean {mean:.2}, tolerance {tolerance_pct}%"
        )
    })
}

/// Assert every conversation has exactly `expected_turns` terminal records.
pub fn assert_session_credits_match<R: ProfileExport>(
    result: &R,
    expected_turns: usize,
) -> Result<(), String> {
    let rows = result.profile_export_records();
    let sessions = sessions(&rows)?;
    let bad: Vec<_> = sessions
        .iter()
        .filter(|(_, rows)| rows.len() != expected_turns)
        .collect();
    bad.is_empty()
        .then_some(())
        .ok_or_else(|| format!("sessions do not all have {expected_turns} credits: {sessions:?}"))
}

/// Assert each conversation's turn indices are `0, 1, …` in credit-issue order.
pub fn assert_turn_indices_sequential<R: ProfileExport>(result: &R) -> Result<(), String> {
    let rows = result.profile_export_records();
    let sessions = sessions(&rows)?;
    verify_no_interleaving_within_session_records(&sessions)?;
    for (session, rows) in sessions {
        let mut rows = rows;
        rows.sort_unstable_by_key(|row| raw_i64(row, "credit_issued_ns").unwrap_or(i64::MIN));
        let indices: Vec<_> = rows
            .iter()
            .map(|row| raw_u64(row, "turn_index"))
            .collect::<Option<_>>()
            .ok_or_else(|| format!("session {session} is missing turn_index"))?;
        if indices
            .iter()
            .enumerate()
            .any(|(expected, actual)| *actual != expected as u64)
        {
            return Err(format!(
                "session {session} has non-sequential turn indices: {indices:?}"
            ));
        }
    }
    Ok(())
}

/// Assert a configuration should exercise its total-concurrency limit.
pub fn assert_test_will_hit_concurrency_limit(
    config: &TimingTestConfig,
    message: &str,
) -> Result<(), String> {
    config
        .will_hit_concurrency_limit()
        .then_some(())
        .ok_or_else(|| {
            format!(
                "{message}test will not hit concurrency limit {:?}; predicted {:.1}",
                config.concurrency,
                config.expected_max_concurrent()
            )
        })
}

/// Assert a configuration should exercise its prefill-concurrency limit.
pub fn assert_test_will_hit_prefill_limit(
    config: &TimingTestConfig,
    message: &str,
) -> Result<(), String> {
    config
        .will_hit_prefill_limit()
        .then_some(())
        .ok_or_else(|| {
            format!(
                "{message}test will not hit prefill limit {:?}; predicted {:.1}",
                config.prefill_concurrency,
                config.expected_max_prefill_concurrent()
            )
        })
}

/// Verify credits within each session have strictly increasing issue timestamps.
pub fn verify_no_interleaving_within_session<R: ProfileExport>(result: &R) -> Result<(), String> {
    let rows = result.profile_export_records();
    verify_no_interleaving_within_session_records(&sessions(&rows)?)
}

/// Verify that credit issue order switches between conversations at least once.
pub fn verify_sessions_can_interleave<R: ProfileExport>(result: &R) -> Result<(), String> {
    let rows = result.profile_export_records();
    let sessions = sessions(&rows)?;
    if sessions.len() < 2 {
        return Ok(());
    }
    let mut events = Vec::new();
    for (session, rows) in &sessions {
        for row in rows {
            events.push((
                raw_i64(row, "credit_issued_ns")
                    .ok_or_else(|| format!("session {session} is missing credit_issued_ns"))?,
                session.as_str(),
            ));
        }
    }
    events.sort_unstable_by_key(|(time, _)| *time);
    let transitions = events
        .windows(2)
        .filter(|pair| pair[0].1 != pair[1].1)
        .count();
    (transitions >= sessions.len() - 1)
        .then_some(())
        .ok_or_else(|| {
            format!(
                "only {transitions} session transitions, expected at least {}",
                sessions.len() - 1
            )
        })
}

fn sessions(rows: &[Value]) -> Result<BTreeMap<String, Vec<&Value>>, String> {
    let mut sessions = BTreeMap::new();
    for row in rows {
        let session = row
            .pointer("/metadata/x_correlation_id")
            .and_then(Value::as_str)
            .ok_or_else(|| {
                "profile-export record is missing metadata.x_correlation_id".to_string()
            })?;
        sessions
            .entry(session.to_owned())
            .or_insert_with(Vec::new)
            .push(row);
    }
    Ok(sessions)
}

fn verify_no_interleaving_within_session_records(
    sessions: &BTreeMap<String, Vec<&Value>>,
) -> Result<(), String> {
    for (session, rows) in sessions {
        let mut rows = rows.clone();
        rows.sort_unstable_by_key(|row| raw_i64(row, "credit_issued_ns").unwrap_or(i64::MIN));
        for pair in rows.windows(2) {
            let previous = raw_i64(pair[0], "credit_issued_ns")
                .ok_or_else(|| format!("session {session} is missing credit_issued_ns"))?;
            let current = raw_i64(pair[1], "credit_issued_ns")
                .ok_or_else(|| format!("session {session} is missing credit_issued_ns"))?;
            if current <= previous {
                return Err(format!(
                    "session {session} credit issue timestamp {current} is before or equal to {previous}"
                ));
            }
        }
    }
    Ok(())
}

fn peak_concurrency(rows: &[Value], prefill: bool) -> Result<usize, String> {
    let mut events = Vec::with_capacity(rows.len() * 2);
    for row in rows {
        let start_key = "credit_issued_ns";
        let end_key = if prefill {
            "request_ack_ns"
        } else {
            "request_end_ns"
        };
        let start = raw_i64(row, start_key)
            .ok_or_else(|| format!("profile-export record is missing {start_key}"))?;
        let end = raw_i64(row, end_key)
            .ok_or_else(|| format!("profile-export record is missing {end_key}"))?;
        events.push((start, 1_i32));
        events.push((end, -1_i32));
    }
    // Match the Python analyzer's `(timestamp, -delta)` sort: an issuance at
    // the same timestamp as a return is counted before the return.
    events.sort_unstable_by_key(|(timestamp, change)| (*timestamp, -*change));
    let mut active = 0_i32;
    let mut peak = 0_i32;
    for (_, change) in events {
        active += change;
        peak = peak.max(active);
    }
    Ok(peak as usize)
}

fn raw_i64(row: &Value, key: &str) -> Option<i64> {
    row.pointer(&format!("/metadata/{key}"))
        .and_then(Value::as_i64)
}
fn raw_u64(row: &Value, key: &str) -> Option<u64> {
    row.pointer(&format!("/metadata/{key}"))
        .and_then(Value::as_u64)
}
