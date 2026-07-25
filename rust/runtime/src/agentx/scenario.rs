// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Scenario-lock runtime helpers (Slice 4), ported from
//! `src/aiperf/common/scenario/`.
//!
//! So far: the context-overflow classifier (`context_overflow.py`). The
//! substring allowlist is passed in (Python reads it from
//! `Environment.AGENTX.CONTEXT_OVERFLOW_SUBSTRINGS`).

use crate::agentx::cache_bust::CacheBustTarget;

/// A named benchmark-scenario invariant lock (Python `ScenarioSpec`). Frozen
/// truth; the resolver auto-fills defaults and validates user conflicts.
#[derive(Debug, Clone, PartialEq)]
pub struct ScenarioSpec {
    /// Scenario identifier, e.g. `"inferencex-agentx-mvp"`.
    pub name: String,
    /// Required timing mode (e.g. `"agentic_replay"`).
    pub timing_mode: String,
    /// Require `ignore_eos`.
    pub require_ignore_eos: bool,
    /// Require `use_think_time_only`.
    pub require_use_think_time_only: bool,
    /// Require streaming.
    pub require_streaming: bool,
    /// Forbid `ignore_trace_delays`.
    pub forbid_ignore_trace_delays: bool,
    /// Forbid input truncation.
    pub forbid_input_truncation: bool,
    /// Allowed loader identifiers.
    pub require_loader: Vec<String>,
    /// Minimum benchmark duration (seconds).
    pub min_benchmark_duration_seconds: i64,
    /// Default benchmark duration (seconds).
    pub default_benchmark_duration_seconds: Option<i64>,
    /// Default trajectory-start min ratio.
    pub default_trajectory_start_min_ratio: Option<f64>,
    /// Default trajectory-start max ratio.
    pub default_trajectory_start_max_ratio: Option<f64>,
    /// Inter-turn delay cap (seconds).
    pub inter_turn_delay_cap_seconds: Option<f64>,
    /// Trace idle-gap cap (seconds).
    pub trace_idle_gap_cap_seconds: Option<f64>,
    /// Required cache-bust target.
    pub require_cache_bust: Option<CacheBustTarget>,
}

/// The `inferencex-agentx-mvp` scenario (Python `INFERENCEX_AGENTX_MVP`).
pub fn inferencex_agentx_mvp() -> ScenarioSpec {
    let loaders = [
        "semianalysis_cc_traces_weka_with_subagents",
        "semianalysis_cc_traces_weka_with_subagents_256k",
        "semianalysis_cc_traces_weka_with_subagents_060226",
        "semianalysis_cc_traces_weka_with_subagents_060226_256k",
        "semianalysis_cc_traces_weka_with_subagents_060526",
        "semianalysis_cc_traces_weka_with_subagents_060526_256k",
        "semianalysis_cc_traces_weka_with_subagents_060826",
        "semianalysis_cc_traces_weka_with_subagents_060826_256k",
        "semianalysis_cc_traces_weka_061326",
        "semianalysis_cc_traces_weka_061326_256k",
        "semianalysis_cc_traces_weka_061526",
        "semianalysis_cc_traces_weka_061526_256k",
        "semianalysis_cc_traces_weka_062126",
        "semianalysis_cc_traces_weka_062126_256k",
        "weka_trace",
        "weka_hf",
    ];
    ScenarioSpec {
        name: "inferencex-agentx-mvp".to_string(),
        timing_mode: "agentic_replay".to_string(),
        require_ignore_eos: true,
        require_use_think_time_only: false,
        require_streaming: true,
        forbid_ignore_trace_delays: true,
        forbid_input_truncation: true,
        require_loader: loaders.iter().map(|s| s.to_string()).collect(),
        min_benchmark_duration_seconds: 900,
        default_benchmark_duration_seconds: Some(1800),
        default_trajectory_start_min_ratio: Some(0.0),
        default_trajectory_start_max_ratio: Some(1.0),
        inter_turn_delay_cap_seconds: None,
        trace_idle_gap_cap_seconds: Some(10.0),
        require_cache_bust: Some(CacheBustTarget::FirstTurnPrefix),
    }
}

/// A specific scenario-lock conflict (Python `ScenarioViolation`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScenarioViolation {
    /// The flag/field in conflict (e.g. `--streaming`).
    pub flag: String,
    /// The value the user provided.
    pub current_value: String,
    /// The value the scenario requires.
    pub required_value: String,
    /// Human-readable explanation.
    pub message: String,
}

/// Outcome of applying one boolean invariant lock (Python `_apply_*` shape).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LockResult {
    /// Already satisfied by the user's config.
    Satisfied,
    /// The scenario default was applied (the field was unset).
    Applied,
    /// The user explicitly set a conflicting value.
    Violated(ScenarioViolation),
}

/// Apply a `require_<x> == true` lock (streaming / ignore-eos pattern): already
/// true → satisfied; explicitly false → violation; unset false → apply default.
pub fn apply_require_true(
    current: bool,
    explicitly_set: bool,
    flag: &str,
    message: &str,
) -> LockResult {
    if current {
        LockResult::Satisfied
    } else if explicitly_set {
        LockResult::Violated(ScenarioViolation {
            flag: flag.to_string(),
            current_value: "false".to_string(),
            required_value: "true".to_string(),
            message: message.to_string(),
        })
    } else {
        LockResult::Applied
    }
}

/// Apply a `forbid_<x>` lock (`x` must be false): already false → satisfied;
/// explicitly true → violation; unset true → apply default (force false).
pub fn apply_forbid_true(
    current: bool,
    explicitly_set: bool,
    flag: &str,
    message: &str,
) -> LockResult {
    if !current {
        LockResult::Satisfied
    } else if explicitly_set {
        LockResult::Violated(ScenarioViolation {
            flag: flag.to_string(),
            current_value: "true".to_string(),
            required_value: "false".to_string(),
            message: message.to_string(),
        })
    } else {
        LockResult::Applied
    }
}

/// True when `loader` is in the scenario's allowed set (Python
/// `_apply_require_loader` membership). `None` loader is not allowed.
pub fn loader_allowed(loader: Option<&str>, allowed: &[String]) -> bool {
    match loader {
        Some(l) => allowed.iter().any(|a| a == l),
        None => false,
    }
}

/// Look up a registered scenario by name (Python `get_scenario`). `None` when
/// unknown.
pub fn get_scenario(name: &str) -> Option<ScenarioSpec> {
    match name {
        "inferencex-agentx-mvp" => Some(inferencex_agentx_mvp()),
        _ => None,
    }
}

/// Classify whether an error response indicates a context-overflow (Python
/// `is_context_overflow_response`).
///
/// Case-insensitive substring match against (1) the raw body text and (2) the
/// OpenAI-style nested `error.message` field when the body parses as JSON.
/// Callers pre-filter to error responses and pre-decode bytes bodies. `None` /
/// empty body or empty allowlist → false.
pub fn is_context_overflow_response(body: Option<&str>, substrings: &[String]) -> bool {
    let text = match body {
        Some(t) if !t.is_empty() => t,
        _ => return false,
    };
    let needles: Vec<String> = substrings
        .iter()
        .filter(|s| !s.is_empty())
        .map(|s| s.to_lowercase())
        .collect();
    if needles.is_empty() {
        return false;
    }

    let lowered = text.to_lowercase();
    if needles.iter().any(|n| lowered.contains(n)) {
        return true;
    }

    if let Some(msg) = extract_openai_error_message(text) {
        let nested = msg.to_lowercase();
        if needles.iter().any(|n| nested.contains(n)) {
            return true;
        }
    }
    false
}

/// Return the OpenAI-style `error.message` from a JSON body (Python
/// `_extract_openai_error_message`). Tolerates a string-shaped `error` field.
fn extract_openai_error_message(text: &str) -> Option<String> {
    let parsed: serde_json::Value = serde_json::from_str(text).ok()?;
    let obj = parsed.as_object()?;
    match obj.get("error") {
        Some(serde_json::Value::Object(err)) => {
            err.get("message").and_then(|m| m.as_str()).map(String::from)
        }
        Some(serde_json::Value::String(s)) => Some(s.clone()),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn subs() -> Vec<String> {
        vec!["context length".into(), "maximum context".into()]
    }

    #[test]
    fn invariant_lock_decisions() {
        // require_streaming: already on -> satisfied.
        assert_eq!(apply_require_true(true, false, "--streaming", "m"), LockResult::Satisfied);
        // unset -> apply default.
        assert_eq!(apply_require_true(false, false, "--streaming", "m"), LockResult::Applied);
        // explicitly off -> violation.
        assert!(matches!(
            apply_require_true(false, true, "--streaming", "m"),
            LockResult::Violated(_)
        ));
        // forbid_ignore_trace_delays: explicitly true -> violation.
        assert!(matches!(
            apply_forbid_true(true, true, "--ignore-trace-delays", "m"),
            LockResult::Violated(_)
        ));
        assert_eq!(apply_forbid_true(false, false, "x", "m"), LockResult::Satisfied);
        // loader allowlist.
        let allowed = vec!["weka_trace".to_string(), "weka_hf".to_string()];
        assert!(loader_allowed(Some("weka_trace"), &allowed));
        assert!(!loader_allowed(Some("mooncake_trace"), &allowed));
        assert!(!loader_allowed(None, &allowed));
    }

    #[test]
    fn mvp_scenario_registered() {
        let s = get_scenario("inferencex-agentx-mvp").unwrap();
        assert_eq!(s.timing_mode, "agentic_replay");
        assert!(s.require_streaming && s.require_ignore_eos);
        assert_eq!(s.min_benchmark_duration_seconds, 900);
        assert_eq!(s.require_cache_bust, Some(CacheBustTarget::FirstTurnPrefix));
        assert!(s.require_loader.contains(&"weka_trace".to_string()));
        assert!(get_scenario("nope").is_none());
    }

    #[test]
    fn matches_raw_body_case_insensitive() {
        assert!(is_context_overflow_response(
            Some("Error: Maximum Context exceeded"),
            &subs()
        ));
    }

    #[test]
    fn matches_openai_error_message() {
        let body = r#"{"error": {"message": "This model's maximum context length is 8192 tokens"}}"#;
        assert!(is_context_overflow_response(Some(body), &subs()));
    }

    #[test]
    fn string_error_field_is_used() {
        let body = r#"{"error": "maximum context reached"}"#;
        assert!(is_context_overflow_response(Some(body), &subs()));
    }

    #[test]
    fn no_match_and_empty_cases() {
        assert!(!is_context_overflow_response(Some("rate limit exceeded"), &subs()));
        assert!(!is_context_overflow_response(None, &subs()));
        assert!(!is_context_overflow_response(Some(""), &subs()));
        assert!(!is_context_overflow_response(Some("context length"), &[]));
    }
}
