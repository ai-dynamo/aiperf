// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Offline cross-field validation of the typed [`BenchmarkConfig`].
//!
//! These are the raise-only cross-field invariants ported from the Python
//! `BenchmarkConfig` `@model_validator(mode="after")` methods in
//! `src/aiperf/config/config.py`. They inspect an already-lowered typed config
//! and return an error naming the first violated invariant. Mutating
//! default-fillers (tokenizer defaulting, seed defaulting, …) are intentionally
//! not ported here — this pass only rejects, it never rewrites.

use anyhow::{bail, Result};

use super::model::config::BenchmarkConfig;
use super::model::dataset::Dataset;
use super::model::phase::{Phase, PhaseKind, PhaseRole};

/// Validate the cross-field invariants of a lowered [`BenchmarkConfig`].
///
/// Returns `Ok(())` when every invariant holds, or the first `Err` describing
/// the violated invariant. Ordering mirrors the Python validator ordering so
/// error messages are stable across the two implementations.
pub fn validate(cfg: &BenchmarkConfig) -> Result<()> {
    // NOTE: the Python `validate_cache_bust_compatibility` and
    // `validate_agentic_cache_warmup` raise-only invariants are not portable
    // until the typed model grows (a) a per-phase `timing_mode` override field,
    // (b) a `cache_bust` target on the config, and (c) scenario-registry /
    // agentic-timing-mode resolution — none of which exist on the typed
    // `BenchmarkConfig`/`Phase` today. Port them once those land.
    validate_phase_names_unique(cfg)?;
    validate_profiling_phase_required(cfg)?;
    validate_phase_stops(cfg)?;
    validate_seamless_not_on_first_phase(cfg)?;
    validate_prefill_requires_streaming(cfg)?;
    validate_endpoint_profile_names(cfg)?;
    validate_phase_dataset_compatibility(cfg)?;
    Ok(())
}

/// The phases list, or an empty slice when the section is absent.
fn phases(cfg: &BenchmarkConfig) -> &[Phase] {
    cfg.phases.as_deref().unwrap_or(&[])
}

/// Whether a phase resolves to the `profiling` role.
///
/// Mirrors `infer_legacy_phase_kind`: an explicit kind wins; otherwise the
/// canonical name `profiling` infers the profiling role, and any other
/// non-canonical name without a kind is not treated as profiling.
fn is_profiling(phase: &Phase) -> bool {
    match phase.common.kind {
        Some(PhaseRole::Profiling) => true,
        Some(PhaseRole::Warmup) => false,
        None => phase.common.name.eq_ignore_ascii_case("profiling"),
    }
}

/// Reject duplicate phase names, case-insensitively.
fn validate_phase_names_unique(cfg: &BenchmarkConfig) -> Result<()> {
    let mut seen: std::collections::HashMap<String, String> = std::collections::HashMap::new();
    for phase in phases(cfg) {
        let key = phase.common.name.to_lowercase();
        if let Some(prev) = seen.get(&key) {
            bail!(
                "duplicate phase name '{}' conflicts with '{}' — names must be unique \
                 case-insensitively",
                phase.common.name,
                prev
            );
        }
        seen.insert(key, phase.common.name.clone());
    }
    Ok(())
}

/// Require at least one profiling-kind phase — warmup alone is not a benchmark.
fn validate_profiling_phase_required(cfg: &BenchmarkConfig) -> Result<()> {
    if !phases(cfg).iter().any(is_profiling) {
        bail!("a 'profiling' phase is required; got no profiling-kind phase");
    }
    Ok(())
}

/// Require each phase to declare a generic stop condition.
///
/// Every phase needs one of `requests`/`duration`/`sessions`, unless a named
/// `scenario` supplies the bound later, or the phase is a self-bounding
/// agentic-cache warmup. The requirement is gated on the phase type: it applies
/// to every phase EXCEPT `fixed_schedule`, whose replay schedule is itself the
/// stop condition. This mirrors the Python `phase._stop_condition_required`
/// ClassVar (`True` for every phase type except `FixedSchedulePhase`), which is
/// unrelated to whether the phase carries an adaptive-scale controller.
fn validate_phase_stops(cfg: &BenchmarkConfig) -> Result<()> {
    // A named scenario owns the benchmark invariants and auto-fills the phase
    // stop condition at resolution time, which runs after this pass.
    if cfg.scenario.is_some() {
        return Ok(());
    }
    for phase in phases(cfg) {
        let c = &phase.common;
        if !matches!(phase.kind, PhaseKind::FixedSchedule { .. })
            && c.requests.is_none()
            && c.duration.is_none()
            && c.sessions.is_none()
            && c.agentic_cache_warmup_duration.is_none()
        {
            bail!(
                "Phase '{}': at least one of 'requests', 'duration', or 'sessions' \
                 must be specified",
                c.name
            );
        }
    }
    Ok(())
}

/// Ensure seamless is not enabled on the first phase.
fn validate_seamless_not_on_first_phase(cfg: &BenchmarkConfig) -> Result<()> {
    if let Some(first) = phases(cfg).first() {
        if first.common.seamless {
            bail!(
                "Phase config '{}' cannot have seamless=true because it is first; \
                 seamless transitions only apply to subsequent phase configs",
                first.common.name
            );
        }
    }
    Ok(())
}

/// Prefill concurrency requires streaming to measure TTFT boundaries.
fn validate_prefill_requires_streaming(cfg: &BenchmarkConfig) -> Result<()> {
    let streaming = cfg.endpoint.as_ref().map(|e| e.streaming).unwrap_or(false);
    for phase in phases(cfg) {
        if phase.common.prefill_concurrency.is_some() && !streaming {
            bail!(
                "Phase '{}': prefill_concurrency requires endpoint.streaming=true",
                phase.common.name
            );
        }
    }
    Ok(())
}

/// Keep named endpoint-profile references structural: keys must be non-empty,
/// carry no surrounding whitespace, and never redefine reserved `default`.
fn validate_endpoint_profile_names(cfg: &BenchmarkConfig) -> Result<()> {
    for profile_id in cfg.endpoint_profiles.keys() {
        if profile_id.trim().is_empty() || profile_id != profile_id.trim() {
            bail!("endpoint_profiles keys must be non-empty and contain no surrounding whitespace");
        }
        if profile_id == "default" {
            bail!(
                "endpoint_profiles cannot redefine reserved profile 'default'; \
                 use benchmark.endpoint for that profile"
            );
        }
    }
    Ok(())
}

/// Whether the discriminated phase body requires sequential dataset sampling.
fn requires_sequential_sampling(phase: &Phase) -> bool {
    matches!(phase.kind, PhaseKind::FixedSchedule { .. })
}

/// Whether the discriminated phase body requires a multi-turn dataset.
fn requires_multi_turn(phase: &Phase) -> bool {
    matches!(phase.kind, PhaseKind::UserCentric { .. })
}

/// Validate that each phase is compatible with the default dataset.
///
/// `fixed_schedule` requires sequential sampling and `user_centric` requires a
/// multi-turn dataset — both checks apply only to file-backed datasets, exactly
/// as the Python `check_phase_dataset_compatibility` predicate does.
fn validate_phase_dataset_compatibility(cfg: &BenchmarkConfig) -> Result<()> {
    // The default dataset is the first authored dataset; nothing to check when
    // no dataset section is present.
    let Some(dataset) = cfg.datasets.as_ref().and_then(|d| d.first()) else {
        return Ok(());
    };
    let Dataset::File(file) = dataset else {
        return Ok(());
    };
    for phase in phases(cfg) {
        if requires_sequential_sampling(phase) && file.sampling.0 != "sequential" {
            bail!(
                "Phase '{}' requires sequential sampling, but the dataset uses '{}' sampling",
                phase.common.name,
                file.sampling.0
            );
        }
        if requires_multi_turn(phase) {
            let is_multi_turn = file.format.as_deref() == Some("multi_turn");
            if !is_multi_turn {
                bail!(
                    "Phase '{}' requires multi_turn dataset format, but the dataset uses '{}' format",
                    phase.common.name,
                    file.format.as_deref().unwrap_or("none")
                );
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// Build a `BenchmarkConfig` from a JSON value, panicking on decode error.
    fn cfg(value: serde_json::Value) -> BenchmarkConfig {
        serde_json::from_value(value).expect("decode BenchmarkConfig")
    }

    /// A minimal valid single-profiling-phase config.
    fn valid_value() -> serde_json::Value {
        json!({
            "endpoint": {
                "urls": ["http://localhost:8000"],
                "type": "chat",
                "streaming": true,
                "use_legacy_max_tokens": false,
                "use_server_token_count": false,
                "timeout_seconds": 60.0,
                "connection_reuse": "pooled",
                "ssl_verify": true,
                "connection_limit": 100,
                "keepalive_timeout": 60.0,
                "download_video_content": false,
                "extra": {},
                "headers": {},
                "http2": false,
                "wait_for_model_timeout": 60.0,
                "wait_for_model_interval": 1.0,
                "wait_for_model_mode": "models"
            },
            "phases": [
                {
                    "name": "profiling",
                    "kind": "profiling",
                    "exclude_from_results": false,
                    "seamless": false,
                    "requests": 100,
                    "type": "concurrency",
                    "concurrency": 8
                }
            ]
        })
    }

    #[test]
    fn valid_config_passes() {
        assert!(validate(&cfg(valid_value())).is_ok());
    }

    #[test]
    fn duplicate_phase_name_case_insensitive_rejected() {
        let mut v = valid_value();
        v["phases"] = json!([
            {"name": "P", "kind": "profiling", "exclude_from_results": false,
             "seamless": false, "requests": 1, "type": "concurrency", "concurrency": 1},
            {"name": "p", "kind": "profiling", "exclude_from_results": false,
             "seamless": false, "requests": 1, "type": "concurrency", "concurrency": 1}
        ]);
        let err = validate(&cfg(v)).unwrap_err().to_string();
        assert!(err.contains("duplicate phase name"), "{err}");
    }

    #[test]
    fn missing_profiling_phase_rejected() {
        let mut v = valid_value();
        v["phases"] = json!([
            {"name": "warmup", "kind": "warmup", "exclude_from_results": true,
             "seamless": false, "requests": 1, "type": "concurrency", "concurrency": 1}
        ]);
        let err = validate(&cfg(v)).unwrap_err().to_string();
        assert!(err.contains("profiling"), "{err}");
    }

    #[test]
    fn profiling_phase_without_stop_rejected() {
        let mut v = valid_value();
        v["phases"] = json!([
            {"name": "profiling", "kind": "profiling", "exclude_from_results": false,
             "seamless": false, "type": "concurrency", "concurrency": 8}
        ]);
        let err = validate(&cfg(v)).unwrap_err().to_string();
        assert!(err.contains("requests"), "{err}");
    }

    #[test]
    fn phase_without_stop_allowed_when_scenario_set() {
        let mut v = valid_value();
        v["scenario"] = json!("some-scenario");
        v["phases"] = json!([
            {"name": "profiling", "kind": "profiling", "exclude_from_results": false,
             "seamless": false, "type": "concurrency", "concurrency": 8}
        ]);
        assert!(validate(&cfg(v)).is_ok());
    }

    #[test]
    fn fixed_schedule_profiling_phase_without_stop_allowed() {
        // FixedSchedule's replay schedule is its own stop condition, so a
        // profiling fixed_schedule phase with no requests/duration/sessions is
        // valid (Python `_stop_condition_required` is False for this type).
        let mut v = valid_value();
        v["datasets"] = json!([
            {"type": "file", "sampling": "sequential", "options": {}, "path": "/tmp/t.jsonl"}
        ]);
        v["phases"] = json!([
            {"name": "profiling", "kind": "profiling", "exclude_from_results": false,
             "seamless": false, "type": "fixed_schedule", "auto_offset": true}
        ]);
        assert!(validate(&cfg(v)).is_ok());
    }

    #[test]
    fn adaptive_scale_phase_without_stop_rejected() {
        // adaptive_scale does NOT exempt a phase from the stop-condition
        // requirement; the gate is purely the phase type (Python invariant).
        let mut v = valid_value();
        v["phases"] = json!([
            {"name": "profiling", "kind": "profiling", "exclude_from_results": false,
             "seamless": false, "type": "concurrency", "concurrency": 8,
             "adaptive_scale": {
                 "control_variable": "concurrency",
                 "minimum": 1, "maximum": 64,
                 "assessment_period_seconds": 10.0,
                 "sustain_duration_seconds": 5.0,
                 "min_completed_requests": 100,
                 "strategy_type": "sla",
                 "step_policy": "linear",
                 "base_step": 1,
                 "max_step_multiplier": 4,
                 "step_percent": 10.0,
                 "sla_filters": []
             }}
        ]);
        let err = validate(&cfg(v)).unwrap_err().to_string();
        assert!(err.contains("requests"), "{err}");
    }

    #[test]
    fn seamless_on_first_phase_rejected() {
        let mut v = valid_value();
        v["phases"][0]["seamless"] = json!(true);
        let err = validate(&cfg(v)).unwrap_err().to_string();
        assert!(err.contains("seamless"), "{err}");
    }

    #[test]
    fn prefill_without_streaming_rejected() {
        let mut v = valid_value();
        v["endpoint"]["streaming"] = json!(false);
        v["phases"][0]["prefill_concurrency"] = json!(4);
        let err = validate(&cfg(v)).unwrap_err().to_string();
        assert!(err.contains("streaming"), "{err}");
    }

    #[test]
    fn empty_endpoint_profile_key_rejected() {
        let mut v = valid_value();
        v["endpoint_profiles"] = json!({"  ": {}});
        let err = validate(&cfg(v)).unwrap_err().to_string();
        assert!(err.contains("non-empty"), "{err}");
    }

    #[test]
    fn default_endpoint_profile_key_rejected() {
        let mut v = valid_value();
        v["endpoint_profiles"] = json!({"default": {}});
        let err = validate(&cfg(v)).unwrap_err().to_string();
        assert!(err.contains("default"), "{err}");
    }

    #[test]
    fn fixed_schedule_requires_sequential_sampling() {
        let mut v = valid_value();
        v["datasets"] = json!([
            {"type": "file", "sampling": "random", "options": {}, "path": "/tmp/t.jsonl"}
        ]);
        v["phases"] = json!([
            {"name": "profiling", "kind": "profiling", "exclude_from_results": false,
             "seamless": false, "requests": 1, "type": "fixed_schedule", "auto_offset": true}
        ]);
        let err = validate(&cfg(v)).unwrap_err().to_string();
        assert!(err.contains("sequential"), "{err}");
    }

    #[test]
    fn fixed_schedule_ok_with_sequential_sampling() {
        let mut v = valid_value();
        v["datasets"] = json!([
            {"type": "file", "sampling": "sequential", "options": {}, "path": "/tmp/t.jsonl"}
        ]);
        v["phases"] = json!([
            {"name": "profiling", "kind": "profiling", "exclude_from_results": false,
             "seamless": false, "requests": 1, "type": "fixed_schedule", "auto_offset": true}
        ]);
        assert!(validate(&cfg(v)).is_ok());
    }

    #[test]
    fn user_centric_requires_multi_turn_dataset() {
        let mut v = valid_value();
        v["datasets"] = json!([
            {"type": "file", "format": "single_turn", "sampling": "sequential",
             "options": {}, "path": "/tmp/t.jsonl"}
        ]);
        v["phases"] = json!([
            {"name": "profiling", "kind": "profiling", "exclude_from_results": false,
             "seamless": false, "requests": 1, "type": "user_centric", "rate": 1.0, "users": 4}
        ]);
        let err = validate(&cfg(v)).unwrap_err().to_string();
        assert!(err.contains("multi_turn"), "{err}");
    }

    #[test]
    fn user_centric_ok_with_multi_turn_dataset() {
        let mut v = valid_value();
        v["datasets"] = json!([
            {"type": "file", "format": "multi_turn", "sampling": "sequential",
             "options": {}, "path": "/tmp/t.jsonl"}
        ]);
        v["phases"] = json!([
            {"name": "profiling", "kind": "profiling", "exclude_from_results": false,
             "seamless": false, "requests": 1, "type": "user_centric", "rate": 1.0, "users": 4}
        ]);
        assert!(validate(&cfg(v)).is_ok());
    }
}
