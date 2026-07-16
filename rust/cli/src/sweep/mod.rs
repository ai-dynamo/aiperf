// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Native sweep / multi-run engine.
//!
//! Ports Python's `config/sweep/expand.py` + `orchestrator/orchestrator.py`:
//! comma-list flags become sweep axes, expanded (grid = Cartesian product, zip =
//! lockstep) into `Variation`s, each of which clones the base flags, overrides
//! the swept scalar, and re-runs the single-run `load::resolve` builder — so the
//! sweep engine is a loop *around* the byte-exact single-run projection.
//!
//! Byte-exact contracts preserved from Python (proven vs `tools/parity/dump_sweep.py`):
//! - Axis keys sorted alphabetically by dotted path before producting/zipping
//!   (directory-name + combination order depend on it).
//! - `label = "<dotted_path>=<value>, ..."`; `dir_name = "<seg>_<value>__..."`.
//! - Value rendering: integer axes render the integer; float axes (rate/duration)
//!   render Python's `str(float)` (`2` → `2.0`).
//! - `sweep_id` is one UUID for the whole plan; `random_seed = base + index`.

use crate::flags::ProfileFlags;

pub mod aggregate;
pub mod artifact_dir;
pub mod run;
pub mod yaml_sweep;

/// How an axis value renders in labels / dir names (mirrors the config field type).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AxisKind {
    Int,
    Float,
}

/// A sweepable flag: its canonical dotted config path, the dir-name segment (the
/// last path component), the value kind, and the flag id used to override it.
struct AxisDef {
    id: &'static str,
    path: &'static str,
    seg: &'static str,
    kind: AxisKind,
}

const AXES: &[AxisDef] = &[
    AxisDef {
        id: "concurrency",
        path: "phases.profiling.concurrency",
        seg: "concurrency",
        kind: AxisKind::Int,
    },
    AxisDef {
        id: "request_count",
        path: "phases.profiling.requests",
        seg: "requests",
        kind: AxisKind::Int,
    },
    AxisDef {
        id: "request_rate",
        path: "phases.profiling.rate",
        seg: "rate",
        kind: AxisKind::Float,
    },
    AxisDef {
        id: "benchmark_duration",
        path: "phases.profiling.duration",
        seg: "duration",
        kind: AxisKind::Float,
    },
    AxisDef {
        id: "num_conversations",
        path: "phases.profiling.sessions",
        seg: "sessions",
        kind: AxisKind::Int,
    },
    AxisDef {
        id: "isl",
        path: "datasets.main.prompts.isl.mean",
        seg: "mean",
        kind: AxisKind::Int,
    },
    AxisDef {
        id: "osl",
        path: "datasets.main.prompts.osl.mean",
        seg: "mean",
        kind: AxisKind::Int,
    },
];

/// Read the raw flag string for an axis id.
fn axis_raw<'a>(flags: &'a ProfileFlags, id: &str) -> Option<&'a str> {
    let v = match id {
        "concurrency" => &flags.concurrency,
        "request_count" => &flags.request_count,
        "request_rate" => &flags.request_rate,
        "benchmark_duration" => &flags.benchmark_duration,
        "num_conversations" => &flags.num_conversations,
        "isl" => &flags.isl,
        "osl" => &flags.osl,
        _ => return None,
    };
    v.as_deref()
}

/// Overwrite the raw flag string for an axis id (used per-variation).
fn axis_set(flags: &mut ProfileFlags, id: &str, value: String) {
    let slot = match id {
        "concurrency" => &mut flags.concurrency,
        "request_count" => &mut flags.request_count,
        "request_rate" => &mut flags.request_rate,
        "benchmark_duration" => &mut flags.benchmark_duration,
        "num_conversations" => &mut flags.num_conversations,
        "isl" => &mut flags.isl,
        "osl" => &mut flags.osl,
        _ => return,
    };
    *slot = Some(value);
}

/// One collected axis: its definition plus the raw split values.
struct Axis {
    def: &'static AxisDef,
    values: Vec<String>,
}

/// Python `str(float)` for a whole/decimal float (`2` → `2.0`, `2.5` → `2.5`).
fn py_float(raw: &str) -> anyhow::Result<String> {
    let v: f64 = raw.trim().parse()?;
    // Rust's `{}` drops the trailing `.0`; re-add it for whole numbers to match
    // Python's `str(float)`. Non-whole values already render with a fraction.
    if v.fract() == 0.0 && v.is_finite() {
        Ok(format!("{v:.1}"))
    } else {
        Ok(format!("{v}"))
    }
}

/// Render an axis value for the label / dir name.
fn render(kind: AxisKind, raw: &str) -> anyhow::Result<String> {
    match kind {
        AxisKind::Int => {
            // Validate it parses; keep the integer text.
            let v: i64 = raw.trim().parse()?;
            Ok(v.to_string())
        }
        AxisKind::Float => py_float(raw),
    }
}

/// One expanded sweep variation.
#[derive(Clone, Debug)]
pub struct Variation {
    /// Zero-based combination index (defines the seed offset).
    pub index: usize,
    /// `"<dotted_path>=<value>, ..."` (sorted by path).
    pub label: String,
    /// `"<seg>_<value>__..."` directory name (sorted by path).
    pub dir_name: String,
    /// Ordered `(dotted_path, rendered_value)` pairs (sorted by path).
    pub values: Vec<(String, String)>,
    /// Per-axis raw flag overrides `(flag_id, raw_value)`.
    overrides: Vec<(&'static str, String)>,
}

/// The expansion outcome: either a single (degenerate) run or a sweep.
pub struct Expansion {
    /// One variation per cell (index 0 = base for a non-sweep).
    pub variations: Vec<Variation>,
    /// Whether any axis had >1 value (a real sweep).
    pub is_sweep: bool,
    /// The dataset entry-count pool shared by every cell when a count axis
    /// (`request_count`/`num_conversations`) is swept: `max(axis)` (Python
    /// `_resolve_entries` uses the max so every variation has its full set).
    /// `None` when no count axis is swept (each cell derives its own entries).
    pub entries_override: Option<i64>,
}

/// The sweep combination strategy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SweepType {
    Grid,
    Zip,
}

/// Collect sweep axes from comma-list flags and expand them.
pub fn expand(flags: &ProfileFlags, sweep_type: SweepType) -> anyhow::Result<Expansion> {
    // Collect axes: any sweepable flag whose value contains a comma.
    let mut axes: Vec<Axis> = Vec::new();
    for def in AXES {
        if let Some(raw) = axis_raw(flags, def.id)
            && raw.contains(',')
        {
            let values: Vec<String> = raw.split(',').map(|s| s.trim().to_string()).collect();
            axes.push(Axis { def, values });
        }
    }
    if axes.is_empty() {
        // No sweep: one base variation.
        return Ok(Expansion {
            variations: vec![Variation {
                index: 0,
                label: "base".to_string(),
                dir_name: String::new(),
                values: Vec::new(),
                overrides: Vec::new(),
            }],
            is_sweep: false,
            entries_override: None,
        });
    }

    // Dataset entry pool for a swept count axis: `max`, highest precedence
    // `num_conversations` then `request_count` (only when `--num-dataset-entries`
    // is not explicitly pinned). Mirrors Python `_resolve_entries`.
    let axis_max = |id: &str| -> Option<i64> {
        axes.iter().find(|a| a.def.id == id).map(|a| {
            a.values
                .iter()
                .filter_map(|v| v.trim().parse::<i64>().ok())
                .max()
                .unwrap_or(0)
        })
    };
    let entries_override = if flags.num_dataset_entries.is_some() {
        None
    } else {
        axis_max("num_conversations").or_else(|| axis_max("request_count"))
    };

    // Alphabetical sort by dotted path — dir-name + combination order depend on it.
    axes.sort_by(|a, b| a.def.path.cmp(b.def.path));

    // Build the combination index tuples.
    let combos: Vec<Vec<usize>> = match sweep_type {
        SweepType::Grid => cartesian(&axes),
        SweepType::Zip => {
            let len = axes[0].values.len();
            anyhow::ensure!(
                axes.iter().all(|a| a.values.len() == len),
                "zip sweep requires all axes to have the same number of values"
            );
            (0..len).map(|i| vec![i; axes.len()]).collect()
        }
    };

    let mut variations = Vec::with_capacity(combos.len());
    for (index, combo) in combos.into_iter().enumerate() {
        let mut label_parts = Vec::new();
        let mut dir_parts = Vec::new();
        let mut values = Vec::new();
        let mut overrides = Vec::new();
        for (axis, &vi) in axes.iter().zip(combo.iter()) {
            let raw = &axis.values[vi];
            let rendered = render(axis.def.kind, raw)?;
            label_parts.push(format!("{}={}", axis.def.path, rendered));
            dir_parts.push(format!("{}_{}", axis.def.seg, rendered));
            values.push((axis.def.path.to_string(), rendered));
            overrides.push((axis.def.id, raw.clone()));
        }
        variations.push(Variation {
            index,
            label: label_parts.join(", "),
            dir_name: dir_parts.join("__"),
            values,
            overrides,
        });
    }
    Ok(Expansion {
        variations,
        is_sweep: true,
        entries_override,
    })
}

/// Cartesian product of axis value indices (last axis varies fastest — matches
/// Python's `itertools.product` over the sorted axes).
fn cartesian(axes: &[Axis]) -> Vec<Vec<usize>> {
    let mut out = vec![vec![]];
    for axis in axes {
        let mut next = Vec::new();
        for prefix in &out {
            for i in 0..axis.values.len() {
                let mut p = prefix.clone();
                p.push(i);
                next.push(p);
            }
        }
        out = next;
    }
    out
}

impl Variation {
    /// Apply this variation's overrides onto a clone of the base flags.
    pub fn apply(&self, base: &ProfileFlags) -> ProfileFlags {
        let mut flags = base.clone();
        for (id, value) in &self.overrides {
            axis_set(&mut flags, id, value.clone());
        }
        flags
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn py_float_matches_python_str() {
        assert_eq!(py_float("2").unwrap(), "2.0");
        assert_eq!(py_float("2.0").unwrap(), "2.0");
        assert_eq!(py_float("2.5").unwrap(), "2.5");
        assert_eq!(py_float("10").unwrap(), "10.0");
    }
}
