// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Scenario-lock runtime helpers (Slice 4), ported from
//! `src/aiperf/common/scenario/`.
//!
//! So far: the context-overflow classifier (`context_overflow.py`). The
//! substring allowlist is passed in (Python reads it from
//! `Environment.AGENTX.CONTEXT_OVERFLOW_SUBSTRINGS`).

use crate::agentx::cache_bust::CacheBustTarget;
use crate::graph::driver::ReplayTaskIdentity;
use crate::graph::recorded::agent_recording::CanonicalReplayFixture;

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
    /// Generic canonical recorded-agent replay policy when this scenario has one.
    pub recorded_agent: Option<RecordedAgentScenarioLock>,
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
        recorded_agent: None,
    }
}

/// Generic lock data for the registry-owned recorded-agent workload.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecordedAgentScenarioLock {
    /// Required fixture workload name.
    pub workload_name: String,
    /// Required PinchBench image reference.
    pub pinch_image: String,
    /// Exact worker and cell cardinality.
    pub workers: u32,
    /// Exact cell cardinality.
    pub cells: u32,
}

/// The comparable canonical recorded-agent replay scenario.
#[must_use]
pub fn recorded_agent_default() -> ScenarioSpec {
    ScenarioSpec {
        name: "recorded-agent-default".to_string(),
        timing_mode: "agentic_replay".to_string(),
        require_ignore_eos: false,
        require_use_think_time_only: false,
        require_streaming: true,
        forbid_ignore_trace_delays: true,
        forbid_input_truncation: true,
        require_loader: vec!["agent_recording".to_string()],
        min_benchmark_duration_seconds: 0,
        default_benchmark_duration_seconds: None,
        default_trajectory_start_min_ratio: None,
        default_trajectory_start_max_ratio: None,
        inter_turn_delay_cap_seconds: None,
        trace_idle_gap_cap_seconds: None,
        require_cache_bust: None,
        recorded_agent: Some(RecordedAgentScenarioLock {
            workload_name: "recorded-agent-eight-v1".to_string(),
            pinch_image: "aiperf-recorded-agent-pinchbench:v1".to_string(),
            workers: 1,
            cells: 1,
        }),
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
        "recorded-agent-default" => Some(recorded_agent_default()),
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
    /// Whether every manifest task completed successfully when completeness is required.
    pub complete: Option<bool>,
    /// Stable reasons a result is incomplete.
    pub incomplete_reasons: Vec<String>,
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
    /// Recorded-agent projection for generic registry-backed lock application.
    pub recorded_agent: Option<RecordedAgentScenarioInputs>,
}

/// Apply a scenario's invariant locks to `inputs`, composing the decision core
/// (Python `apply_scenario`, config-projection). Returns the outcome, or a hard
/// `ScenarioLockError` when a non-overridable violation exists or violations
/// remain without `unsafe_override`.
pub fn apply_scenario_locks(
    spec: &ScenarioSpec,
    inputs: &RunLockInputs,
) -> Result<ScenarioOutcome, ScenarioLockError> {
    if spec.recorded_agent.is_some() {
        let fixture = CanonicalReplayFixture::load().map_err(|error| ScenarioLockError {
            violations: vec![ScenarioViolation {
                flag: "recorded_agent.fixture".to_string(),
                current_value: "unavailable".to_string(),
                required_value: "canonical fixture".to_string(),
                message: error.to_string(),
            }],
            bypassable: false,
        })?;
        let recorded = inputs
            .recorded_agent
            .as_ref()
            .ok_or_else(|| ScenarioLockError {
                violations: vec![ScenarioViolation {
                    flag: "dataset.format".to_string(),
                    current_value: "<none>".to_string(),
                    required_value: "agent_recording".to_string(),
                    message:
                        "scenario \"recorded-agent-default\" requires an agent recording dataset"
                            .to_string(),
                }],
                bypassable: false,
            })?;
        return apply_recorded_agent_scenario_locks(recorded, &fixture);
    }
    let mut violations: Vec<ScenarioViolation> = Vec::new();
    let mut applied: Vec<String> = Vec::new();
    let record =
        |r: LockResult, lock: &str, v: &mut Vec<ScenarioViolation>, a: &mut Vec<String>| match r {
            LockResult::Satisfied | LockResult::Applied => a.push(lock.to_string()),
            LockResult::Violated(viol) => v.push(viol),
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
        record(
            r,
            "forbid_ignore_trace_delays",
            &mut violations,
            &mut applied,
        );
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
            complete: None,
            incomplete_reasons: Vec::new(),
        });
    }
    Ok(ScenarioOutcome {
        scenario_name: Some(spec.name.clone()),
        applied_locks: applied,
        violations: Vec::new(),
        submission_valid: Some(true),
        submission_invalid_reasons: Vec::new(),
        complete: None,
        incomplete_reasons: Vec::new(),
    })
}

/// The resolved fields that the canonical replay lock checks.
#[derive(Debug, Clone)]
pub struct RecordedAgentScenarioInputs {
    /// Ordered source task identity vector.
    pub task_order: Vec<ReplayTaskIdentity>,
    /// Canonical fixture manifest digest.
    pub manifest_digest: String,
    /// Canonical fixture recording digests.
    pub recording_digests: std::collections::BTreeMap<String, String>,
    /// Dataset format identity.
    pub dataset_format: String,
    /// Whether task tools execute on the real host.
    pub execute_tools: bool,
    /// Whether the active transport uses virtual time.
    pub virtual_clock: bool,
    /// Runtime worker count.
    pub workers: u32,
    /// Runtime cell count.
    pub cells: u32,
    /// Whether a trace can wrap/recycle.
    pub allow_wrap: bool,
    /// Whether sampling shuffles task order.
    pub shuffle: bool,
    /// Maximum concurrently active traces.
    pub active_traces: u32,
    /// Whether endpoint streaming is enabled.
    pub streaming: bool,
    /// Whether endpoint usage is authoritative for token counts.
    pub use_server_token_count: bool,
    /// Whether client input truncation is enabled.
    pub input_truncation: bool,
    /// Whether the metrics path is sketch-only.
    pub sketch_metrics: bool,
    /// Selected cache isolation mode.
    pub cache_isolation_mode: String,
    /// PinchBench environment image.
    pub pinch_image: String,
    /// Whether per-task warmup is enabled.
    pub warmup: bool,
    /// Free-form hardware description, with `unknown` allowed.
    pub hardware_description: Option<String>,
    /// Whether a prior run requested persistent resume.
    pub resume: bool,
    /// Whether the complete manifest reached successful terminal cleanup.
    pub complete: bool,
    /// User-authorized bypass for non-hard scenario conflicts.
    pub unsafe_override: bool,
}

impl RecordedAgentScenarioInputs {
    /// Construct the exact canonical lock projection for a fixture.
    #[must_use]
    pub fn canonical(fixture: &CanonicalReplayFixture) -> Self {
        Self {
            task_order: fixture
                .manifest
                .tasks
                .iter()
                .map(|task| task.identity.clone())
                .collect(),
            manifest_digest: fixture.manifest_digest.clone(),
            recording_digests: fixture.digest_index.recordings.clone(),
            dataset_format: "agent_recording".to_string(),
            execute_tools: true,
            virtual_clock: false,
            workers: 1,
            cells: 1,
            allow_wrap: false,
            shuffle: false,
            active_traces: 1,
            streaming: true,
            use_server_token_count: true,
            input_truncation: false,
            sketch_metrics: false,
            cache_isolation_mode: "first_message_prefix".to_string(),
            pinch_image: "aiperf-recorded-agent-pinchbench:v1".to_string(),
            warmup: true,
            hardware_description: Some("unknown".to_string()),
            resume: false,
            complete: true,
            unsafe_override: false,
        }
    }
}

/// Apply the registry-owned canonical replay policy without widening AgentX semantics.
pub fn apply_recorded_agent_scenario_locks(
    inputs: &RecordedAgentScenarioInputs,
    fixture: &CanonicalReplayFixture,
) -> Result<ScenarioOutcome, ScenarioLockError> {
    let lock = match recorded_agent_default().recorded_agent {
        Some(lock) => lock,
        None => {
            return Err(ScenarioLockError {
                violations: vec![ScenarioViolation {
                    flag: "scenario.registry".to_string(),
                    current_value: "missing recorded-agent lock".to_string(),
                    required_value: "recorded-agent lock".to_string(),
                    message: "recorded-agent-default registry entry is incomplete".to_string(),
                }],
                bypassable: false,
            });
        }
    };
    let mut violations = Vec::new();
    let mut applied = Vec::new();
    let mut record = |is_valid: bool, flag: &str, current: String, required: String| {
        if is_valid {
            applied.push(flag.to_string());
        } else {
            violations.push(ScenarioViolation {
                flag: flag.to_string(),
                current_value: current,
                required_value: required,
                message: format!("scenario \"recorded-agent-default\" requires {flag}"),
            });
        }
    };
    record(
        inputs.dataset_format == "agent_recording",
        "dataset.format",
        inputs.dataset_format.clone(),
        "agent_recording".to_string(),
    );
    record(
        inputs.manifest_digest == fixture.manifest_digest,
        "dataset.manifest_digest",
        inputs.manifest_digest.clone(),
        fixture.manifest_digest.clone(),
    );
    record(
        inputs.recording_digests == fixture.digest_index.recordings,
        "dataset.recording_digests",
        "different".to_string(),
        "canonical digests".to_string(),
    );
    let canonical_order = fixture
        .manifest
        .tasks
        .iter()
        .map(|task| task.identity.clone())
        .collect::<Vec<_>>();
    record(
        inputs.task_order == canonical_order,
        "dataset.task_order",
        "different".to_string(),
        "canonical manifest order".to_string(),
    );
    record(
        inputs.workers == lock.workers,
        "runtime.workers",
        inputs.workers.to_string(),
        lock.workers.to_string(),
    );
    record(
        inputs.cells == lock.cells,
        "runtime.cells",
        inputs.cells.to_string(),
        lock.cells.to_string(),
    );
    record(
        inputs.active_traces == 1,
        "runtime.active_traces",
        inputs.active_traces.to_string(),
        "1".to_string(),
    );
    record(
        !inputs.allow_wrap,
        "dataset.allow_wrap",
        inputs.allow_wrap.to_string(),
        "false".to_string(),
    );
    record(
        !inputs.shuffle,
        "dataset.shuffle",
        inputs.shuffle.to_string(),
        "false".to_string(),
    );
    record(
        inputs.streaming,
        "endpoint.streaming",
        inputs.streaming.to_string(),
        "true".to_string(),
    );
    record(
        inputs.use_server_token_count,
        "endpoint.use_server_token_count",
        inputs.use_server_token_count.to_string(),
        "true".to_string(),
    );
    record(
        !inputs.input_truncation,
        "dataset.input_truncation",
        inputs.input_truncation.to_string(),
        "false".to_string(),
    );
    record(
        !inputs.sketch_metrics,
        "metrics.sketch",
        inputs.sketch_metrics.to_string(),
        "false".to_string(),
    );
    record(
        inputs.cache_isolation_mode == "first_message_prefix",
        "dataset.cache_isolation",
        inputs.cache_isolation_mode.clone(),
        "first_message_prefix".to_string(),
    );
    record(
        inputs.execute_tools,
        "dataset.graph.execute_tools",
        inputs.execute_tools.to_string(),
        "true".to_string(),
    );
    record(
        inputs.pinch_image == lock.pinch_image,
        "dataset.graph.pinch_image",
        inputs.pinch_image.clone(),
        lock.pinch_image,
    );
    record(
        inputs.warmup,
        "dataset.graph.emit_warmup",
        inputs.warmup.to_string(),
        "true".to_string(),
    );
    record(
        inputs
            .hardware_description
            .as_deref()
            .is_some_and(|value| !value.trim().is_empty()),
        "metadata.hardware",
        inputs
            .hardware_description
            .clone()
            .unwrap_or_else(|| "<none>".to_string()),
        "non-empty".to_string(),
    );
    record(
        inputs.complete,
        "result.complete",
        inputs.complete.to_string(),
        "true".to_string(),
    );
    if inputs.virtual_clock {
        return Err(ScenarioLockError {
            violations: vec![ScenarioViolation {
                flag: "transport.clock".to_string(),
                current_value: "virtual".to_string(),
                required_value: "real".to_string(),
                message: "recorded-agent tools cannot run on a virtual clock".to_string(),
            }],
            bypassable: false,
        });
    }
    if inputs.resume && inputs.cells > 1 {
        return Err(ScenarioLockError {
            violations: vec![ScenarioViolation {
                flag: "runtime.cells".to_string(),
                current_value: inputs.cells.to_string(),
                required_value: "1 when resume is enabled".to_string(),
                message: "recorded-agent resume does not support cells > 1".to_string(),
            }],
            bypassable: false,
        });
    }
    if !violations.is_empty() && !inputs.unsafe_override {
        return Err(ScenarioLockError {
            violations,
            bypassable: true,
        });
    }
    let incomplete_reasons = if !inputs.complete {
        vec!["incomplete_replay".to_string()]
    } else {
        Vec::new()
    };
    let mut invalid_reasons = Vec::new();
    if !violations.is_empty() {
        invalid_reasons.extend(["unsafe_override".to_string(), "non_comparable".to_string()]);
    }
    if !incomplete_reasons.is_empty() {
        invalid_reasons.extend(incomplete_reasons.clone());
    }
    Ok(ScenarioOutcome {
        scenario_name: Some("recorded-agent-default".to_string()),
        applied_locks: applied,
        violations,
        submission_valid: Some(invalid_reasons.is_empty()),
        submission_invalid_reasons: invalid_reasons,
        complete: Some(inputs.complete),
        incomplete_reasons,
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
        Some(serde_json::Value::Object(err)) => err
            .get("message")
            .and_then(|m| m.as_str())
            .map(String::from),
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
        assert_eq!(
            out.submission_invalid_reasons,
            vec!["unsafe_override".to_string()]
        );
    }

    #[test]
    fn invariant_lock_decisions() {
        // require_streaming: already on -> satisfied.
        assert_eq!(
            apply_require_true(true, false, "--streaming", "m"),
            LockResult::Satisfied
        );
        // unset -> apply default.
        assert_eq!(
            apply_require_true(false, false, "--streaming", "m"),
            LockResult::Applied
        );
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
        assert_eq!(
            apply_forbid_true(false, false, "x", "m"),
            LockResult::Satisfied
        );
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
        let body =
            r#"{"error": {"message": "This model's maximum context length is 8192 tokens"}}"#;
        assert!(is_context_overflow_response(Some(body), &subs()));
    }

    #[test]
    fn string_error_field_is_used() {
        let body = r#"{"error": "maximum context reached"}"#;
        assert!(is_context_overflow_response(Some(body), &subs()));
    }

    #[test]
    fn no_match_and_empty_cases() {
        assert!(!is_context_overflow_response(
            Some("rate limit exceeded"),
            &subs()
        ));
        assert!(!is_context_overflow_response(None, &subs()));
        assert!(!is_context_overflow_response(Some(""), &subs()));
        assert!(!is_context_overflow_response(Some("context length"), &[]));
    }
}
