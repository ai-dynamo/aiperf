// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Benchmark phase identity and loadgen validation.

use std::collections::HashSet;

use crate::model::phase::{Phase, PhaseKind, PhaseRole};
use crate::model::rate_series::RateSeries;

const PHASE_NAME_PATTERN: &str = r"^[A-Za-z_][A-Za-z0-9_-]*$";

/// Reserved Windows path component names (case-insensitive).
const WINDOWS_RESERVED: &[&str] = &[
    "CON", "PRN", "AUX", "NUL", "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8",
    "COM9", "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9",
];

/// Infer semantic role from reserved canonical names when `kind` is omitted.
pub fn infer_phase_role(name: &str, kind: Option<PhaseRole>) -> anyhow::Result<PhaseRole> {
    if let Some(kind) = kind {
        return Ok(kind);
    }
    match name {
        "warmup" => Ok(PhaseRole::Warmup),
        "profiling" => Ok(PhaseRole::Profiling),
        _ => anyhow::bail!("phase {name:?} requires explicit kind (warmup or profiling)"),
    }
}

/// Normalize every phase's `kind` and `exclude_from_results` and validate the workflow.
pub fn normalize_and_validate_phases(phases: &mut [Phase]) -> anyhow::Result<()> {
    if phases.is_empty() {
        anyhow::bail!("at least one phase is required");
    }

    let mut seen_names = HashSet::new();
    let mut profiling_count = 0usize;

    for (index, phase) in phases.iter_mut().enumerate() {
        let name = phase.common.name.trim();
        if name.is_empty() {
            anyhow::bail!("phase {index} name must be a non-empty string");
        }
        validate_phase_name(name)?;

        let lower = name.to_ascii_lowercase();
        if !seen_names.insert(lower) {
            anyhow::bail!("duplicate phase name {name:?} (names are unique case-insensitively)");
        }

        let role = infer_phase_role(name, phase.common.kind)?;
        if name == "warmup" && role != PhaseRole::Warmup {
            anyhow::bail!("phase name warmup cannot be paired with kind profiling");
        }
        if name == "profiling" && role != PhaseRole::Profiling {
            anyhow::bail!("phase name profiling cannot be paired with kind warmup");
        }

        phase.common.kind = Some(role);
        let expected_exclude = role == PhaseRole::Warmup;
        if phase.common.exclude_from_results != expected_exclude {
            anyhow::bail!(
                "phase {name:?} exclude_from_results must be {expected_exclude} for kind {role:?}"
            );
        }

        if role == PhaseRole::Profiling {
            profiling_count += 1;
        }

        validate_phase_loadgen(phase)?;
    }

    if profiling_count == 0 {
        anyhow::bail!("a profiling phase is required");
    }

    if phases.first().is_some_and(|p| p.common.seamless) {
        anyhow::bail!("seamless cannot be set on the first phase");
    }

    Ok(())
}

fn validate_phase_name(name: &str) -> anyhow::Result<()> {
    let re = regex::Regex::new(PHASE_NAME_PATTERN).expect("valid phase name regex");
    if !re.is_match(name) {
        anyhow::bail!(
            "phase name {name:?} must match {PHASE_NAME_PATTERN} (letters, digits, _ and -)"
        );
    }
    let upper = name.to_ascii_uppercase();
    if WINDOWS_RESERVED.contains(&upper.as_str()) {
        anyhow::bail!("phase name {name:?} is a reserved Windows path component");
    }
    Ok(())
}

fn validate_phase_loadgen(phase: &Phase) -> anyhow::Result<()> {
    let is_rate_phase = matches!(
        phase.kind,
        PhaseKind::Poisson { .. } | PhaseKind::Gamma { .. } | PhaseKind::Constant { .. }
    );
    let scalar_rate = match phase.kind {
        PhaseKind::Poisson { rate, .. }
        | PhaseKind::Gamma { rate, .. }
        | PhaseKind::Constant { rate, .. } => Some(rate),
        _ => None,
    };
    let has_series = phase.common.rate_series.is_some();

    if matches!(phase.kind, PhaseKind::UserCentric { .. }) && has_series {
        anyhow::bail!("user-centric phases do not support rate_series");
    }

    if let Some(adaptive) = phase.common.adaptive_scale.as_ref() {
        // Adaptive scale drives its own control axis; a fixed-schedule replay
        // has no controllable knob to sweep, so the combination is rejected.
        if matches!(phase.kind, PhaseKind::FixedSchedule { .. }) {
            anyhow::bail!("adaptive_scale cannot be combined with fixed_schedule phases");
        }
        // A request_rate-controlled adaptive sweep and an authored rate_series
        // both own the phase's request rate; they cannot coexist.
        if adaptive.control_variable == "request_rate" && has_series {
            anyhow::bail!(
                "adaptive_scale control.variable=request_rate cannot be combined with rate_series"
            );
        }
    }

    if has_series {
        let series = phase
            .common
            .rate_series
            .as_ref()
            .expect("rate_series present");
        series.validate()?;
        if !is_rate_phase {
            anyhow::bail!("rate_series requires a rate-controlled phase type");
        }
        // Projection fills `rate` from `rate_series.initial_qps` for interval
        // bootstrap. Authored mutual exclusion is rate != initial bootstrap.
        if let Some(rate) = scalar_rate
            && (rate - series.initial_qps()).abs() > 1e-9
        {
            anyhow::bail!("rate and rate_series are mutually exclusive");
        }
    } else if is_rate_phase && scalar_rate.is_none_or(|rate| !(rate.is_finite() && rate > 0.0)) {
        anyhow::bail!("rate-controlled phases require rate or rate_series");
    }

    Ok(())
}

/// Count profiling phases for CLI overlay ambiguity checks.
pub fn profiling_phase_count(phases: &[Phase]) -> usize {
    phases
        .iter()
        .filter(|p| p.common.kind == Some(PhaseRole::Profiling))
        .count()
}

/// CLI loadgen fields overlaid onto the unique profiling phase.
#[derive(Default)]
pub(crate) struct LoadgenOverlay {
    concurrency: Option<u32>,
    request_rate: Option<f64>,
    request_count: Option<u64>,
    benchmark_duration: Option<f64>,
    grace_period: Option<f64>,
    prefill_concurrency: Option<u32>,
    sessions: Option<u64>,
    request_rate_series: Option<RateSeries>,
}

impl LoadgenOverlay {
    pub(crate) fn from_inputs(inputs: &crate::load::Inputs) -> Self {
        Self {
            concurrency: inputs.concurrency,
            request_rate: inputs.request_rate,
            request_count: inputs.request_count,
            benchmark_duration: inputs.benchmark_duration,
            grace_period: inputs.grace_period,
            prefill_concurrency: inputs.prefill_concurrency,
            sessions: inputs.sessions,
            request_rate_series: inputs.request_rate_series.clone(),
        }
    }
}
pub fn find_unique_profiling_phase_index(phases: &[Phase]) -> anyhow::Result<usize> {
    let profiling_indices: Vec<usize> = phases
        .iter()
        .enumerate()
        .filter(|(_, phase)| phase.common.kind == Some(PhaseRole::Profiling))
        .map(|(index, _)| index)
        .collect();
    if profiling_indices.len() != 1 {
        let names: Vec<&str> = profiling_indices
            .iter()
            .map(|index| phases[*index].common.name.as_str())
            .collect();
        anyhow::bail!(
            "CLI loadgen flags target the profiling phase, but this config has \
             {} profiling phases: {}. Set the value in YAML or use an explicit phase path.",
            profiling_indices.len(),
            names.join(", ")
        );
    }
    Ok(profiling_indices[0])
}

/// Overlay explicit CLI loadgen flags onto the unique profiling phase in a
/// multi-phase YAML workflow.
pub(crate) fn apply_cli_loadgen_overlays(
    phases: &mut [Phase],
    overlay: &LoadgenOverlay,
) -> anyhow::Result<()> {
    let has_overlay = overlay.concurrency.is_some()
        || overlay.request_rate.is_some()
        || overlay.request_count.is_some()
        || overlay.benchmark_duration.is_some()
        || overlay.grace_period.is_some()
        || overlay.prefill_concurrency.is_some()
        || overlay.sessions.is_some()
        || overlay.request_rate_series.is_some();
    if !has_overlay {
        return Ok(());
    }

    let index = find_unique_profiling_phase_index(phases)?;
    let phase = &mut phases[index];
    if let Some(requests) = overlay.request_count {
        phase.common.requests = Some(requests);
    }
    if let Some(duration) = overlay.benchmark_duration {
        phase.common.duration = Some(duration);
    }
    if let Some(grace) = overlay.grace_period {
        phase.common.grace_period = Some(grace);
    }
    if let Some(sessions) = overlay.sessions {
        phase.common.sessions = Some(sessions);
    }
    if let Some(prefill) = overlay.prefill_concurrency {
        phase.common.prefill_concurrency = Some(prefill);
    }
    if let Some(series) = &overlay.request_rate_series {
        phase.common.rate_series = Some(series.clone());
        let initial = series.initial_qps();
        match &mut phase.kind {
            PhaseKind::Poisson { rate, .. }
            | PhaseKind::Gamma { rate, .. }
            | PhaseKind::Constant { rate, .. } => *rate = initial,
            _ => {}
        }
    } else if let Some(rate) = overlay.request_rate {
        match &mut phase.kind {
            PhaseKind::Poisson { rate: slot, .. }
            | PhaseKind::Gamma { rate: slot, .. }
            | PhaseKind::Constant { rate: slot, .. } => *slot = rate,
            PhaseKind::UserCentric { rate: slot, .. } => *slot = rate,
            _ => {}
        }
    }
    if let Some(concurrency) = overlay.concurrency {
        match &mut phase.kind {
            PhaseKind::Concurrency { concurrency: slot } => *slot = concurrency,
            PhaseKind::Poisson {
                concurrency: slot, ..
            }
            | PhaseKind::Gamma {
                concurrency: slot, ..
            }
            | PhaseKind::Constant {
                concurrency: slot, ..
            } => *slot = Some(concurrency),
            PhaseKind::UserCentric {
                concurrency: slot, ..
            } => *slot = Some(concurrency),
            _ => {}
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::phase::{PhaseCommon, PhaseKind};

    fn concurrency_phase(name: &str, kind: Option<PhaseRole>) -> Phase {
        Phase {
            common: PhaseCommon {
                timing_mode: None,
                name: name.into(),
                kind,
                exclude_from_results: kind == Some(PhaseRole::Warmup),
                seamless: false,
                requests: Some(2),
                sessions: None,
                duration: None,
                prefill_concurrency: None,
                grace_period: None,
                concurrency_ramp: None,
                prefill_ramp: None,
                rate_ramp: None,
                cancellation: None,
                agentic_cache_warmup_duration: None,
                adaptive_scale: None,
                rate_series: None,
            },
            kind: PhaseKind::Concurrency { concurrency: 1 },
        }
    }

    fn adaptive_scale(control_variable: &str) -> crate::model::phase::AdaptiveScale {
        crate::model::phase::AdaptiveScale {
            control_variable: control_variable.into(),
            minimum: serde_json::Number::from(1),
            maximum: serde_json::Number::from(64),
            assessment_period_seconds: 5.0,
            sustain_duration_seconds: 10.0,
            min_completed_requests: 1,
            strategy_type: "ramp_until_fail".into(),
            step_policy: "sla_margin".into(),
            base_step: 2,
            max_step_multiplier: 4,
            step_percent: 0.0,
            sla_filters: vec![],
        }
    }

    #[test]
    fn adaptive_scale_rejected_on_fixed_schedule_phase() {
        let mut phase = concurrency_phase("fs", Some(PhaseRole::Profiling));
        phase.common.adaptive_scale = Some(adaptive_scale("concurrency"));
        phase.kind = PhaseKind::FixedSchedule {
            auto_offset: true,
            start_offset: None,
            end_offset: None,
        };
        let err = normalize_and_validate_phases(&mut vec![phase]).unwrap_err();
        assert!(err.to_string().contains("fixed_schedule"));
    }

    #[test]
    fn adaptive_request_rate_rejects_rate_series() {
        let mut phase = concurrency_phase("rr", Some(PhaseRole::Profiling));
        phase.common.adaptive_scale = Some(adaptive_scale("request_rate"));
        phase.common.rate_series =
            Some(crate::model::rate_series::RateSeries::from_json_str("[[0,1],[10,5]]").unwrap());
        phase.kind = PhaseKind::Constant {
            rate: 1.0,
            concurrency: None,
        };
        let err = normalize_and_validate_phases(&mut vec![phase]).unwrap_err();
        assert!(err.to_string().contains("rate_series"));
    }

    #[test]
    fn custom_name_requires_kind() {
        let mut phases = vec![concurrency_phase("storm_1", None)];
        assert!(normalize_and_validate_phases(&mut phases).is_err());
    }

    #[test]
    fn multiple_profiling_phases_allowed() {
        let mut phases = vec![
            concurrency_phase("low", Some(PhaseRole::Profiling)),
            concurrency_phase("storm", Some(PhaseRole::Profiling)),
        ];
        normalize_and_validate_phases(&mut phases).unwrap();
        assert_eq!(profiling_phase_count(&phases), 2);
    }

    #[test]
    fn unique_profiling_index_rejects_ambiguous_workflows() {
        let mut phases = vec![
            concurrency_phase("low", Some(PhaseRole::Profiling)),
            concurrency_phase("storm", Some(PhaseRole::Profiling)),
        ];
        normalize_and_validate_phases(&mut phases).unwrap();
        let err = find_unique_profiling_phase_index(&phases).unwrap_err();
        assert!(err.to_string().contains("2 profiling phases"));
        assert!(err.to_string().contains("low, storm"));
    }

    #[test]
    fn warmup_only_rejected() {
        let mut phases = vec![concurrency_phase("warmup", Some(PhaseRole::Warmup))];
        assert!(normalize_and_validate_phases(&mut phases).is_err());
    }
}
