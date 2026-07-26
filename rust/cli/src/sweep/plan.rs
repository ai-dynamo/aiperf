// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Typed benchmark-plan expansion over the shared config model.
//!
//! One unifying seam mirrors Python's `build_benchmark_plan`/`expand_sweep`: a
//! base [`BenchmarkConfig`] plus a typed [`Sweep`] expand into one
//! [`BenchmarkRun`] per combination. Each run carries the swept scalar applied
//! onto a clone of the typed config (by dotted config path), variation metadata,
//! and `random_seed = base_seed + index`.
//!
//! Axis ordering, combination order, and value rendering deliberately match the
//! flag-driven [`super::expand`] contract so this seam can subsume it without
//! shifting observable output (artifact dir names, combination order, seeds).

use serde_json::Value;

use crate::model::{BenchmarkConfig, BenchmarkRun, Resolved};
use crate::sweep::SweepType;

/// One typed sweep axis: a dotted config path and its ordered scalar values.
///
/// `path` addresses the config model in serialized form. Object keys index by
/// name; array elements are matched by their `name` field (so `phases.profiling`
/// selects the `profiling` phase and `datasets.main` the `main` dataset).
#[derive(Clone, Debug)]
pub struct SweepAxis {
    /// Dotted config path (e.g. `phases.profiling.concurrency`).
    pub path: String,
    /// Directory-name segment (the last meaningful path component).
    pub seg: String,
    /// Ordered typed values this axis takes across the sweep.
    pub values: Vec<Value>,
}

/// A typed sweep: its axes and how they combine.
#[derive(Clone, Debug)]
pub struct Sweep {
    /// The sweep axes (any order; expansion sorts by dotted path).
    pub axes: Vec<SweepAxis>,
    /// Grid (cartesian) or zip (lockstep) combination.
    pub sweep_type: SweepType,
}

impl Sweep {
    /// A grid sweep of the given axes.
    pub fn grid(axes: Vec<SweepAxis>) -> Self {
        Self {
            axes,
            sweep_type: SweepType::Grid,
        }
    }

    /// A lockstep (zip) sweep of the given axes.
    pub fn zip(axes: Vec<SweepAxis>) -> Self {
        Self {
            axes,
            sweep_type: SweepType::Zip,
        }
    }

    /// A single-axis sweep from a "magic list": a numeric list authored directly
    /// at a config path (e.g. a phase field carrying `[1, 2, 4]`) that expands to
    /// one run per element. The last path component is the directory segment.
    pub fn magic_list(path: impl Into<String>, values: Vec<Value>) -> Self {
        let path = path.into();
        let seg = path.rsplit('.').next().unwrap_or(path.as_str()).to_string();
        Self::grid(vec![SweepAxis { path, seg, values }])
    }
}

/// Render a JSON scalar for a variation label / directory name.
///
/// Integers render bare; integral floats retain a trailing `.0`; other values
/// use their shortest form. This matches the flag-path rendering contract.
fn render_value(v: &Value) -> String {
    match v {
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                i.to_string()
            } else if let Some(f) = n.as_f64() {
                if f.fract() == 0.0 && f.is_finite() {
                    format!("{f:.1}")
                } else {
                    format!("{f}")
                }
            } else {
                n.to_string()
            }
        }
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

/// Set a dotted config path onto a serialized config `Value`, creating nothing:
/// the path must already exist. Object segments index by key; array segments
/// match the element whose `name` equals the segment. Returns `true` when the
/// leaf was found and overwritten.
fn set_dotted(cfg: &mut Value, path: &str, value: Value) -> bool {
    let mut segments = path.split('.').peekable();
    let mut cur = cfg;
    while let Some(seg) = segments.next() {
        let is_leaf = segments.peek().is_none();
        // Resolve `cur` to the container the segment addresses.
        let next: Option<&mut Value> = match cur {
            Value::Object(map) => {
                if is_leaf {
                    // The leaf key must already exist: `BenchmarkConfig` is not
                    // `deny_unknown_fields`, so inserting a typo'd key would be
                    // silently dropped on re-deserialize, yielding duplicate runs
                    // with no error. Require pre-existence like the array path.
                    match map.get_mut(seg) {
                        Some(slot) => {
                            *slot = value;
                            return true;
                        }
                        None => return false,
                    }
                }
                map.get_mut(seg)
            }
            Value::Array(items) => {
                let found = items
                    .iter_mut()
                    .find(|item| item.get("name").and_then(Value::as_str) == Some(seg));
                match found {
                    Some(item) => {
                        if is_leaf {
                            // A named array element is a container, not a scalar
                            // slot; a leaf segment cannot target it.
                            return false;
                        }
                        Some(item)
                    }
                    None => return false,
                }
            }
            _ => return false,
        };
        match next {
            Some(n) => cur = n,
            None => return false,
        }
    }
    false
}

/// Cartesian product of per-axis value indices, last axis varying fastest.
fn cartesian(lengths: &[usize]) -> Vec<Vec<usize>> {
    let mut out = vec![vec![]];
    for &len in lengths {
        let mut next = Vec::new();
        for prefix in &out {
            for i in 0..len {
                let mut p = prefix.clone();
                p.push(i);
                next.push(p);
            }
        }
        out = next;
    }
    out
}

/// Expand a base config and typed sweep into one [`BenchmarkRun`] per
/// combination.
///
/// Axes are sorted by dotted path first (so combination order and labels are
/// deterministic and match the flag path). For each combination the base config
/// is cloned, every axis scalar applied by dotted path, and the result stamped
/// with variation metadata and `random_seed = base_seed + index`. The returned
/// runs carry no artifact directory or sweep id; the caller stamps those.
pub fn build_benchmark_plan(
    base: &BenchmarkConfig,
    sweep: &Sweep,
    base_seed: Option<u64>,
) -> anyhow::Result<Vec<BenchmarkRun>> {
    let mut axes = sweep.axes.clone();
    // Directory names and combination order depend on dotted-path ordering.
    axes.sort_by(|a, b| a.path.cmp(&b.path));

    let combos: Vec<Vec<usize>> = match sweep.sweep_type {
        SweepType::Grid => cartesian(&axes.iter().map(|a| a.values.len()).collect::<Vec<_>>()),
        SweepType::Zip => {
            let len = axes.first().map(|a| a.values.len()).unwrap_or(0);
            anyhow::ensure!(
                axes.iter().all(|a| a.values.len() == len),
                "zip sweep requires all axes to have the same number of values"
            );
            (0..len).map(|i| vec![i; axes.len()]).collect()
        }
    };

    let base_value = serde_json::to_value(base)?;
    let mut runs = Vec::with_capacity(combos.len());
    for (index, combo) in combos.into_iter().enumerate() {
        let mut cfg_value = base_value.clone();
        let mut label_parts = Vec::new();
        let mut values = serde_json::Map::new();
        for (axis, &vi) in axes.iter().zip(combo.iter()) {
            let raw = &axis.values[vi];
            anyhow::ensure!(
                set_dotted(&mut cfg_value, &axis.path, raw.clone()),
                "sweep path {:?} not found in config",
                axis.path
            );
            label_parts.push(format!("{}={}", axis.path, render_value(raw)));
            values.insert(axis.path.clone(), raw.clone());
        }
        let cfg: BenchmarkConfig = serde_json::from_value(cfg_value)?;
        let label = label_parts.join(", ");
        runs.push(BenchmarkRun {
            benchmark_id: String::new(),
            artifact_dir: std::path::PathBuf::new(),
            cfg,
            cli_command: None,
            label: label.clone(),
            random_seed: base_seed.map(|b| b + index as u64),
            sweep_id: None,
            trial: 0,
            variation: Some(serde_json::json!({
                "index": index,
                "label": label,
                "values": values,
            })),
            resolved: Resolved::default(),
            variables: serde_json::Map::new(),
        });
    }
    Ok(runs)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A base config with a single `profiling` concurrency phase.
    fn base_cfg() -> BenchmarkConfig {
        use crate::model::phase::{Phase, PhaseCommon, PhaseKind, PhaseRole};
        let phase = Phase {
            common: PhaseCommon {
                timing_mode: None,
                name: "profiling".into(),
                kind: Some(PhaseRole::Profiling),
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
                adaptive_scale: None,
                rate_series: None,
            },
            kind: PhaseKind::Concurrency { concurrency: 1 },
        };
        BenchmarkConfig {
            phases: Some(vec![phase]),
            ..BenchmarkConfig::default()
        }
    }

    #[test]
    fn grid_concurrency_expands_with_seeds_and_values() {
        let sweep = Sweep::grid(vec![SweepAxis {
            path: "phases.profiling.concurrency".to_string(),
            seg: "concurrency".to_string(),
            values: vec![Value::from(1), Value::from(2), Value::from(4)],
        }]);

        let runs = build_benchmark_plan(&base_cfg(), &sweep, Some(42)).expect("plan");

        assert_eq!(runs.len(), 3, "one run per axis value");

        let concurrency = |run: &BenchmarkRun| -> i64 {
            let phases = run.cfg.phases.as_ref().unwrap();
            let profiling = phases
                .iter()
                .find(|p| p.common.name == "profiling")
                .unwrap();
            serde_json::to_value(&profiling.kind).unwrap()["concurrency"]
                .as_i64()
                .unwrap()
        };

        assert_eq!(concurrency(&runs[0]), 1);
        assert_eq!(concurrency(&runs[1]), 2);
        assert_eq!(concurrency(&runs[2]), 4);

        assert_eq!(runs[0].random_seed, Some(42));
        assert_eq!(runs[1].random_seed, Some(43));
        assert_eq!(runs[2].random_seed, Some(44));

        let variation = runs[1].variation.as_ref().unwrap();
        assert_eq!(variation["index"], serde_json::json!(1));
        assert_eq!(
            variation["label"],
            serde_json::json!("phases.profiling.concurrency=2")
        );
        assert_eq!(
            variation["values"]["phases.profiling.concurrency"],
            serde_json::json!(2)
        );
    }

    /// A base config with `profiling` (concurrency) and `requests` phase fields
    /// plus a `main` dataset, exercising multi-axis and array-by-name paths.
    fn base_two_field_cfg() -> BenchmarkConfig {
        use crate::model::phase::{Phase, PhaseCommon, PhaseKind, PhaseRole};
        let common = PhaseCommon {
            timing_mode: None,
            name: "profiling".into(),
            kind: Some(PhaseRole::Profiling),
            exclude_from_results: false,
            seamless: false,
            requests: Some(10),
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
        };
        BenchmarkConfig {
            phases: Some(vec![Phase {
                common,
                kind: PhaseKind::Concurrency { concurrency: 1 },
            }]),
            ..BenchmarkConfig::default()
        }
    }

    #[test]
    fn zip_axes_expand_in_lockstep() {
        // Two axes of equal length combine lockstep, not cartesian.
        let sweep = Sweep::zip(vec![
            SweepAxis {
                path: "phases.profiling.concurrency".to_string(),
                seg: "concurrency".to_string(),
                values: vec![Value::from(1), Value::from(2)],
            },
            SweepAxis {
                path: "phases.profiling.requests".to_string(),
                seg: "requests".to_string(),
                values: vec![Value::from(100), Value::from(200)],
            },
        ]);

        let runs = build_benchmark_plan(&base_two_field_cfg(), &sweep, Some(7)).expect("plan");
        assert_eq!(runs.len(), 2, "lockstep yields one run per index, not 4");

        let cfg0 = serde_json::to_value(&runs[0].cfg).unwrap();
        assert_eq!(cfg0["phases"][0]["concurrency"], serde_json::json!(1));
        assert_eq!(cfg0["phases"][0]["requests"], serde_json::json!(100));
        let cfg1 = serde_json::to_value(&runs[1].cfg).unwrap();
        assert_eq!(cfg1["phases"][0]["concurrency"], serde_json::json!(2));
        assert_eq!(cfg1["phases"][0]["requests"], serde_json::json!(200));
        assert_eq!(runs[1].random_seed, Some(8));
    }

    #[test]
    fn magic_list_expands_numeric_list_at_phase_path() {
        // A numeric list authored at a phase path expands to one run per element.
        let sweep = Sweep::magic_list(
            "phases.profiling.concurrency",
            vec![Value::from(4), Value::from(8), Value::from(16)],
        );
        assert_eq!(sweep.axes[0].seg, "concurrency");

        let runs = build_benchmark_plan(&base_two_field_cfg(), &sweep, Some(42)).expect("plan");
        assert_eq!(runs.len(), 3);
        let got: Vec<i64> = runs
            .iter()
            .map(|r| {
                serde_json::to_value(&r.cfg).unwrap()["phases"][0]["concurrency"]
                    .as_i64()
                    .unwrap()
            })
            .collect();
        assert_eq!(got, vec![4, 8, 16]);
        assert_eq!(runs[2].random_seed, Some(44));
    }

    #[test]
    fn missing_path_is_an_error() {
        // A leaf key that does not exist on an existing object is a hard error:
        // set_dotted requires pre-existence so a typo'd path cannot silently
        // inject a key that re-deserialization drops.
        let sweep = Sweep::magic_list("phases.profiling.nonexistent_field", vec![Value::from(1)]);
        // A missing named phase (array element) cannot be traversed.
        let sweep_bad_phase =
            Sweep::magic_list("phases.no_such_phase.concurrency", vec![Value::from(1)]);
        assert!(build_benchmark_plan(&base_two_field_cfg(), &sweep, Some(1)).is_err());
        assert!(build_benchmark_plan(&base_two_field_cfg(), &sweep_bad_phase, Some(1)).is_err());
    }
}
