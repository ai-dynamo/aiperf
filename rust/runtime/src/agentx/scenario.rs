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
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
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

/// Result of applying a scenario's invariant locks (Python `ScenarioOutcome`).
#[derive(Debug, Clone, PartialEq, Default, serde::Serialize)]
pub struct ScenarioOutcome {
    /// The applied scenario name (`None` for a no-op / no scenario).
    pub scenario_name: Option<String>,
    /// Locks auto-filled or already satisfied.
    pub applied_locks: Vec<String>,
    /// Conflicts (empty on success; populated only under `unsafe_override`).
    pub violations: Vec<ScenarioViolation>,
    /// Submission validity (`Some(false)` under override, `Some(true)` on clean
    /// apply, `None` for the no-op outcome).
    pub submission_valid: Option<bool>,
    /// Reasons submission is invalid.
    pub submission_invalid_reasons: Vec<String>,
}

/// A hard scenario-lock failure (Python `ScenarioLockError`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScenarioLockError {
    /// The conflicting violations.
    pub violations: Vec<ScenarioViolation>,
    /// Whether `--unsafe-override` could bypass it.
    pub bypassable: bool,
}

impl std::fmt::Display for ScenarioLockError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "scenario lock failure ({} violation(s){}): {}",
            self.violations.len(),
            if self.bypassable {
                ", bypassable with --unsafe-override"
            } else {
                ""
            },
            self.violations
                .iter()
                .map(|v| v.message.as_str())
                .collect::<Vec<_>>()
                .join("; ")
        )
    }
}

impl std::error::Error for ScenarioLockError {}

/// The run config fields the scenario invariants read (a projection of
/// `BenchmarkRun.cfg`), with explicit-set flags where the check distinguishes
/// user-set from default.
#[derive(Debug, Clone, Default)]
pub struct RunLockInputs {
    /// `endpoint.streaming` + whether the user set it.
    pub streaming: bool,
    /// Whether `--streaming` was explicitly set.
    pub streaming_explicit: bool,
    /// `extra_inputs.ignore_eos`: `None` absent, `Some(false/true)`.
    pub ignore_eos: Option<bool>,
    /// `dataset.ignore_trace_delays` + explicit flag.
    pub ignore_trace_delays: bool,
    /// Whether `--ignore-trace-delays` was explicitly set.
    pub ignore_trace_delays_explicit: bool,
    /// The detected loader identifier.
    pub loader: Option<String>,
    /// The user's cache-bust target (`None` = unset).
    pub cache_bust: Option<CacheBustTarget>,
    /// Whether cache-bust was explicitly set.
    pub cache_bust_explicit: bool,
    /// `--unsafe-override`.
    pub unsafe_override: bool,
    /// Whether the dataset is the synthetic CLI default (non-overridable under
    /// `require_loader`).
    pub synthetic_default_dataset: bool,
}

/// Apply a scenario's invariant locks to `inputs`, composing the decision core
/// (Python `apply_scenario`, config-projection). Returns the outcome, or a hard
/// `ScenarioLockError` when a non-overridable violation exists or violations
/// remain without `unsafe_override`.
pub fn apply_scenario_locks(
    spec: &ScenarioSpec,
    inputs: &RunLockInputs,
) -> Result<ScenarioOutcome, ScenarioLockError> {
    let mut violations: Vec<ScenarioViolation> = Vec::new();
    let mut applied: Vec<String> = Vec::new();
    let record = |r: LockResult, lock: &str, v: &mut Vec<ScenarioViolation>, a: &mut Vec<String>| {
        match r {
            LockResult::Satisfied | LockResult::Applied => a.push(lock.to_string()),
            LockResult::Violated(viol) => v.push(viol),
        }
    };

    if spec.require_streaming {
        let r = apply_require_true(
            inputs.streaming,
            inputs.streaming_explicit,
            "--streaming",
            &format!("scenario {:?} requires --streaming", spec.name),
        );
        record(r, "streaming", &mut violations, &mut applied);
    }
    if spec.require_ignore_eos {
        // None -> inject (applied); Some(false) -> violation; Some(true) -> applied.
        match inputs.ignore_eos {
            None | Some(true) => applied.push("ignore_eos".to_string()),
            Some(false) => violations.push(ScenarioViolation {
                flag: "extra_inputs.ignore_eos".into(),
                current_value: "false".into(),
                required_value: "true".into(),
                message: format!("scenario {:?} requires ignore_eos=true", spec.name),
            }),
        }
    }
    if spec.forbid_ignore_trace_delays {
        let r = apply_forbid_true(
            inputs.ignore_trace_delays,
            inputs.ignore_trace_delays_explicit,
            "--ignore-trace-delays",
            &format!("scenario {:?} forbids --ignore-trace-delays", spec.name),
        );
        record(r, "forbid_ignore_trace_delays", &mut violations, &mut applied);
    }
    if !spec.require_loader.is_empty() {
        if loader_allowed(inputs.loader.as_deref(), &spec.require_loader) {
            applied.push("require_loader".to_string());
        } else {
            violations.push(ScenarioViolation {
                flag: "--input-file/--public-dataset".into(),
                current_value: inputs.loader.clone().unwrap_or_else(|| "<none>".into()),
                required_value: spec.require_loader.join(" | "),
                message: format!("scenario {:?} requires an allowed weka loader", spec.name),
            });
        }
    }
    if let Some(required) = spec.require_cache_bust {
        match inputs.cache_bust {
            Some(cb) if cb == required => applied.push("require_cache_bust".to_string()),
            None => applied.push("require_cache_bust".to_string()), // auto-filled
            Some(cb) if inputs.cache_bust_explicit => violations.push(ScenarioViolation {
                flag: "--cache-bust".into(),
                current_value: format!("{cb:?}"),
                required_value: format!("{required:?}"),
                message: format!("scenario {:?} requires a specific cache-bust", spec.name),
            }),
            Some(_) => applied.push("require_cache_bust".to_string()),
        }
    }

    // Non-overridable: a synthetic default dataset under require_loader always fails.
    let hard = !spec.require_loader.is_empty()
        && inputs.synthetic_default_dataset
        && violations.iter().any(|v| v.flag.contains("input-file"));
    if hard {
        return Err(ScenarioLockError {
            violations,
            bypassable: false,
        });
    }
    if !violations.is_empty() && !inputs.unsafe_override {
        return Err(ScenarioLockError {
            violations,
            bypassable: true,
        });
    }
    if !violations.is_empty() {
        return Ok(ScenarioOutcome {
            scenario_name: Some(spec.name.clone()),
            applied_locks: applied,
            violations,
            submission_valid: Some(false),
            submission_invalid_reasons: vec!["unsafe_override".to_string()],
        });
    }
    Ok(ScenarioOutcome {
        scenario_name: Some(spec.name.clone()),
        applied_locks: applied,
        violations: Vec::new(),
        submission_valid: Some(true),
        submission_invalid_reasons: Vec::new(),
    })
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
    fn apply_scenario_resolver_outcomes() {
        let spec = inferencex_agentx_mvp();
        // Clean config: streaming on, ignore_eos set, no trace-delay-ignore,
        // allowed loader, correct cache-bust -> submission_valid true.
        let ok = RunLockInputs {
            streaming: true,
            ignore_eos: Some(true),
            ignore_trace_delays: false,
            loader: Some("weka_trace".into()),
            cache_bust: Some(CacheBustTarget::FirstTurnPrefix),
            ..Default::default()
        };
        let out = apply_scenario_locks(&spec, &ok).unwrap();
        assert_eq!(out.submission_valid, Some(true));
        assert!(out.violations.is_empty());
        assert!(out.applied_locks.contains(&"streaming".to_string()));

        // Explicit --no-streaming -> hard lock error (bypassable).
        let bad = RunLockInputs {
            streaming: false,
            streaming_explicit: true,
            ignore_eos: Some(true),
            loader: Some("weka_trace".into()),
            cache_bust: Some(CacheBustTarget::FirstTurnPrefix),
            ..Default::default()
        };
        let err = apply_scenario_locks(&spec, &bad).unwrap_err();
        assert!(err.bypassable);
        assert!(!err.violations.is_empty());

        // Same conflict + unsafe_override -> outcome with submission_valid false.
        let overridden = RunLockInputs {
            unsafe_override: true,
            ..bad
        };
        let out = apply_scenario_locks(&spec, &overridden).unwrap();
        assert_eq!(out.submission_valid, Some(false));
        assert_eq!(out.submission_invalid_reasons, vec!["unsafe_override".to_string()]);
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
