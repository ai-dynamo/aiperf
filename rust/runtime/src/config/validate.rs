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

use anyhow::{Result, bail};

use super::model::config::BenchmarkConfig;
use super::model::dataset::{CacheBustTarget, Dataset};
use super::model::phase::{Phase, PhaseKind, PhaseRole};

/// Validate the cross-field invariants of a lowered [`BenchmarkConfig`].
///
/// Returns `Ok(())` when every invariant holds, or the first `Err` describing
/// the violated invariant. Ordering mirrors the Python validator ordering so
/// error messages are stable across the two implementations.
pub fn validate(cfg: &BenchmarkConfig) -> Result<()> {
    validate_phase_names_unique(cfg)?;
    validate_profiling_phase_required(cfg)?;
    validate_phase_stops(cfg)?;
    validate_seamless_not_on_first_phase(cfg)?;
    validate_prefill_requires_streaming(cfg)?;
    validate_endpoint_profile_names(cfg)?;
    validate_phase_dataset_compatibility(cfg)?;
    validate_cache_bust_compatibility(cfg)?;
    validate_recorded_agent_replay(cfg)?;
    validate_agentic_cache_warmup(cfg)?;
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

/// The agentic-replay timing-mode identifier (Python `TimingMode.AGENTIC_REPLAY`).
const AGENTIC_REPLAY: &str = "agentic_replay";

/// Resolve the default dataset's cache-bust target.
///
/// Mirrors Python `BenchmarkConfig.get_cache_bust_target`: for a synthetic
/// dataset the target lives at `prompts.cache_bust.target`; for a file-backed
/// dataset it lives at the dataset-level `cache_bust.target`. Any other dataset
/// kind (or an absent dataset section) resolves to `None`.
fn cache_bust_target(cfg: &BenchmarkConfig) -> CacheBustTarget {
    let Some(dataset) = cfg.datasets.as_ref().and_then(|d| d.first()) else {
        return CacheBustTarget::None;
    };
    match dataset {
        Dataset::Synthetic(s) => s
            .prompts
            .cache_bust
            .map(|cb| cb.target)
            .unwrap_or(CacheBustTarget::None),
        Dataset::File(f) => f
            .cache_bust
            .map(|cb| cb.target)
            .unwrap_or(CacheBustTarget::None),
        Dataset::Public(p) => p
            .cache_bust
            .map(|cb| cb.target)
            .unwrap_or(CacheBustTarget::None),
    }
}

/// The profiling-kind phases, in authored order.
fn profiling_phases(cfg: &BenchmarkConfig) -> impl Iterator<Item = &Phase> {
    phases(cfg).iter().filter(|p| is_profiling(p))
}

/// Refuse cache-bust on incompatible timing modes / endpoint types.
///
/// Ported from Python `validate_cache_bust_compatibility`. Marker minting only
/// fires in the agentic-replay strategy, and only the chat / responses endpoint
/// formatters consume the system-message field that hosts the marker; any other
/// combination silently drops the marker, so we refuse loudly at config time.
///
/// Deferral cases (the lockdown does NOT fire, exactly as in Python):
/// * a config carrying a `scenario` is governed by that scenario's own
///   invariant locks (applied post-construction, which never re-triggers this
///   pass), so we skip;
/// * no profiling phase carries an EXPLICIT `timing_mode` override — the
///   effective mode is derived at runtime from the phase type, so agentic-ness
///   is not yet knowable here. The lockdown only fires once a phase has
///   explicitly declared a (non-agentic) timing mode incompatible with the
///   requested cache-bust.
fn validate_cache_bust_compatibility(cfg: &BenchmarkConfig) -> Result<()> {
    if cfg.scenario.is_some() {
        return Ok(());
    }
    if cache_bust_target(cfg) == CacheBustTarget::None {
        return Ok(());
    }

    let explicit: Vec<&str> = profiling_phases(cfg)
        .filter_map(|p| p.common.timing_mode.as_deref())
        .collect();
    if explicit.is_empty() {
        return Ok(());
    }
    if !explicit
        .iter()
        .any(|m| m.eq_ignore_ascii_case(AGENTIC_REPLAY))
    {
        bail!(
            "cache-bust requires the agentic_replay timing mode \
             (set today by --scenario inferencex-agentx-mvp); the profiling \
             phase(s) are not agentic_replay. Cache-bust marker minting is only \
             implemented for agentic_replay."
        );
    }

    let endpoint_type = cfg
        .endpoint
        .as_ref()
        .map(|e| e.endpoint_type.0.as_str())
        .unwrap_or("");
    if !endpoint_type.eq_ignore_ascii_case("chat")
        && !endpoint_type.eq_ignore_ascii_case("responses")
    {
        bail!(
            "cache-bust requires --endpoint-type chat or responses; got {}. \
             Other endpoint formatters do not consume the system message field \
             that hosts the marker.",
            endpoint_type
        );
    }
    Ok(())
}

/// Validate recorded-agent graph authoring before worker construction.
fn validate_recorded_agent_replay(cfg: &BenchmarkConfig) -> Result<()> {
    for dataset in cfg.datasets.as_deref().unwrap_or_default() {
        let Dataset::File(file) = dataset else {
            continue;
        };
        let Some(graph) = file.graph.as_ref() else {
            continue;
        };
        if file.format.as_deref() != Some("agent_recording") {
            bail!("dataset.graph is only supported with dataset.format=agent_recording");
        }
        for (field, value) in [
            ("command_timeout_seconds", graph.command_timeout_seconds),
            (
                "container_stop_timeout_seconds",
                graph.container_stop_timeout_seconds,
            ),
            (
                "session_close_grace_seconds",
                graph.session_close_grace_seconds,
            ),
        ] {
            if !value.is_finite() || value <= 0.0 {
                bail!("dataset.graph.{field} must be a positive finite number of seconds");
            }
        }
        if graph.tool_image.is_some() && !graph.execute_tools {
            bail!("dataset.graph.tool_image requires dataset.graph.execute_tools=true");
        }
    }
    if let Some(metadata) = cfg.metadata.as_ref()
        && !matches!(
            metadata.endpoint_placement.as_str(),
            "co_located" | "remote" | "unknown"
        )
    {
        bail!(
            "metadata.endpoint_placement must be co_located, remote, or unknown"
        );
    }
    Ok(())
}

/// Whether the profiling phases resolve to the agentic-replay timing mode.
///
/// Mirrors Python `_is_agentic_replay` / `_phase_timing_mode`: the first
/// profiling phase's EXPLICIT `timing_mode` override wins; otherwise the
/// phase-type mapping is consulted — and no phase type ever maps to
/// agentic_replay, so a config without an explicit override is never agentic.
fn profiling_is_agentic_replay(cfg: &BenchmarkConfig) -> bool {
    match profiling_phases(cfg).next() {
        Some(first) => match first.common.timing_mode.as_deref() {
            Some(mode) => mode.eq_ignore_ascii_case(AGENTIC_REPLAY),
            // The phase-type mapping never yields agentic_replay.
            None => false,
        },
        None => false,
    }
}

/// Restrict accelerated cache warmup to a weka reconstruction run.
///
/// Ported from Python `validate_agentic_cache_warmup`, and kept keyed the same
/// way as the `resolve.rs` twin: BOTH weka arms consume the accelerated
/// cache-warmup substage — the legacy arm through `lower_legacy_agentic`, the
/// graph-ir arm through `build_pressure_recycle` in `graph_phase_runtime` — so
/// any resolved `weka_semantics` accepts. Outside weka the value reaches no
/// consumer and is silently dropped, so an unguarded flag there is an invisible
/// no-op; reject it instead.
///
/// NOTE (ported-partial): Python resolves a named scenario's declared
/// `timing_mode` through the scenario registry (`get_scenario(...).timing_mode`)
/// and rejects the flag when that lock is not agentic_replay. That resolution is
/// a runtime-only registry lookup with no representation in the typed config, so
/// the scenario branch is DEFERRED here: when a `scenario` is set we return
/// `Ok`, leaving the scenario-lock check to the runtime scenario resolver. The
/// no-scenario branch (the fully config-time-checkable case) is ported exactly.
fn validate_agentic_cache_warmup(cfg: &BenchmarkConfig) -> Result<()> {
    let uses_warmup =
        profiling_phases(cfg).any(|p| p.common.agentic_cache_warmup_duration.is_some());
    if !uses_warmup {
        return Ok(());
    }
    // Scenario-locked timing_mode is resolved from the scenario registry at
    // runtime, not at config time; defer that branch to the runtime resolver.
    if cfg.scenario.is_some() {
        return Ok(());
    }
    // An explicit `weka_semantics` selects a weka arm outright. Absent that, an
    // explicit agentic_replay phase override still reaches the legacy lowering.
    if cfg.weka_semantics.is_none() && !profiling_is_agentic_replay(cfg) {
        bail!(
            "--agentic-cache-warmup-duration requires a weka reconstruction run \
             (--weka-semantics, or a scenario that locks one); neither weka arm \
             lowers this run, so the accelerated cache-warmup substage reaches no \
             consumer."
        );
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

    // ---- validate_cache_bust_compatibility ---------------------------------

    /// A synthetic dataset carrying a cache-bust target.
    fn synthetic_with_cache_bust(target: &str) -> serde_json::Value {
        json!([{
            "type": "synthetic",
            "prompts": {
                "batch_size": 1,
                "isl": {"mean": 128.0},
                "cache_bust": {"target": target}
            },
            "sampling": "sequential",
            "turn_delay_ratio": 1.0
        }])
    }

    #[test]
    fn cache_bust_with_non_agentic_explicit_timing_mode_rejected() {
        let mut v = valid_value();
        v["datasets"] = synthetic_with_cache_bust("system_prefix");
        // Explicit non-agentic timing mode + cache-bust is the loud-refuse case.
        v["phases"][0]["timing_mode"] = json!("request_rate");
        let err = validate(&cfg(v)).unwrap_err().to_string();
        assert!(err.contains("agentic_replay"), "{err}");
    }

    #[test]
    fn cache_bust_agentic_but_wrong_endpoint_rejected() {
        let mut v = valid_value();
        v["endpoint"]["type"] = json!("completions");
        v["datasets"] = synthetic_with_cache_bust("system_prefix");
        v["phases"][0]["timing_mode"] = json!("agentic_replay");
        let err = validate(&cfg(v)).unwrap_err().to_string();
        assert!(err.contains("chat or responses"), "{err}");
    }

    #[test]
    fn cache_bust_agentic_chat_endpoint_ok() {
        let mut v = valid_value();
        v["datasets"] = synthetic_with_cache_bust("system_prefix");
        v["phases"][0]["timing_mode"] = json!("agentic_replay");
        assert!(validate(&cfg(v)).is_ok());
    }

    #[test]
    fn cache_bust_deferred_without_explicit_timing_mode() {
        // No explicit timing_mode → agentic-ness resolved later; do not refuse.
        let mut v = valid_value();
        v["datasets"] = synthetic_with_cache_bust("system_prefix");
        assert!(validate(&cfg(v)).is_ok());
    }

    #[test]
    fn cache_bust_skipped_when_scenario_set() {
        let mut v = valid_value();
        v["scenario"] = json!("inferencex-agentx-mvp");
        v["datasets"] = synthetic_with_cache_bust("system_prefix");
        v["phases"][0]["timing_mode"] = json!("request_rate");
        assert!(validate(&cfg(v)).is_ok());
    }

    #[test]
    fn cache_bust_none_target_ok() {
        let mut v = valid_value();
        v["datasets"] = synthetic_with_cache_bust("none");
        v["phases"][0]["timing_mode"] = json!("request_rate");
        assert!(validate(&cfg(v)).is_ok());
    }

    // ---- validate_agentic_cache_warmup -------------------------------------

    #[test]
    fn agentic_warmup_without_weka_rejected() {
        let mut v = valid_value();
        v["phases"][0]["agentic_cache_warmup_duration"] = json!(30.0);
        let err = validate(&cfg(v)).unwrap_err().to_string();
        assert!(err.contains("requires a weka reconstruction run"), "{err}");
    }

    /// Both weka arms consume the substage, so either spelling accepts. This
    /// regressed once in the `resolve.rs` twin, which accepted only `legacy`.
    #[test]
    fn agentic_warmup_accepted_under_any_weka_semantics() {
        for mode in ["legacy", "graph-ir"] {
            let mut v = valid_value();
            v["weka_semantics"] = json!(mode);
            v["phases"][0]["agentic_cache_warmup_duration"] = json!(30.0);
            assert!(validate(&cfg(v)).is_ok(), "rejected under {mode}");
        }
    }

    #[test]
    fn agentic_warmup_with_agentic_timing_mode_ok() {
        let mut v = valid_value();
        v["phases"][0]["agentic_cache_warmup_duration"] = json!(30.0);
        v["phases"][0]["timing_mode"] = json!("agentic_replay");
        assert!(validate(&cfg(v)).is_ok());
    }

    #[test]
    fn agentic_warmup_deferred_when_scenario_set() {
        // Scenario-locked timing_mode is resolved at runtime; defer (do not raise).
        let mut v = valid_value();
        v["scenario"] = json!("inferencex-agentx-mvp");
        v["phases"][0]["agentic_cache_warmup_duration"] = json!(30.0);
        assert!(validate(&cfg(v)).is_ok());
    }

    fn recorded_agent_dataset(graph: serde_json::Value, options: serde_json::Value) -> serde_json::Value {
        json!([{
            "type": "file",
            "format": "agent_recording",
            "sampling": "sequential",
            "options": options,
            "path": "/tmp/recording.json",
            "graph": graph
        }])
    }

    #[test]
    fn recorded_agent_tool_execution_rejects_virtual_transports() {
        for transport in ["dry_run", "dynosim_offline"] {
            let mut v = valid_value();
            v["transport"] = json!({"type": transport});
            v["datasets"] = recorded_agent_dataset(json!({"execute_tools": true}), json!({}));
            let err = validate(&cfg(v)).expect_err("virtual transport must be rejected");
            assert!(err.to_string().contains(transport), "{err}");
        }
    }

    #[test]
    fn recorded_agent_tool_config_rejects_images_without_tools_and_open_loop() {
        let mut image_without_tools = valid_value();
        image_without_tools["datasets"] = recorded_agent_dataset(
            json!({"pinch_image": "pinch:latest"}),
            json!({}),
        );
        let err = validate(&cfg(image_without_tools))
            .expect_err("tool images require tool execution")
            .to_string();
        assert!(err.contains("pinch_image") && err.contains("execute_tools"), "{err}");

        let mut open_loop = valid_value();
        open_loop["datasets"] = recorded_agent_dataset(
            json!({"execute_tools": true}),
            json!({"open_loop_replay": true}),
        );
        let err = validate(&cfg(open_loop))
            .expect_err("open-loop replay cannot retain trace-local tool ordering")
            .to_string();
        assert!(err.contains("open_loop_replay"), "{err}");
    }
}
