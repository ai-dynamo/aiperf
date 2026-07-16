// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! YAML `sweep:` block expansion.
//!
//! Byte-for-byte port of Python's `aiperf.config.sweep.expand::_expand_grid_sweep`
//! / `_expand_zip_sweep` + the shared dotted-path helpers
//! (`aiperf.config.loader.dotted_path`). A config with a top-level
//!
//! ```yaml
//! sweep:
//!   type: grid            # or zip
//!   parameters:
//!     datasets.default.prompts.isl: [128, 512, 2048]
//!     rate: [10.0, 30.0]  # bare-name alias -> phases.profiling.rate
//! ```
//!
//! expands, per Python's plan build, in the order: `${ENV}` substitution on the
//! base config → sweep expansion (deep-clone the base, write each parameter value
//! into the `benchmark` subtree at its dotted path, drop the `sweep` key) →
//! per-variation Jinja render → resolve each variation as a single run. Parameter
//! names are alpha-sorted (stable variation order); grid = Cartesian product, zip
//! = lockstep. Label/dir-name match `SweepVariation` (`"path=value, ..."` /
//! `"seg_value__..."`).

use std::collections::BTreeMap;

use serde_json::{Map, Value};

/// Bare-name sugar for the 12 most-swept phase fields (`_SWEEP_PATH_ALIASES`).
const ALIASES: &[(&str, &str)] = &[
    ("concurrency", "phases.profiling.concurrency"),
    (
        "prefill_concurrency",
        "phases.profiling.prefill_concurrency",
    ),
    ("rate", "phases.profiling.rate"),
    ("requests", "phases.profiling.requests"),
    ("duration", "phases.profiling.duration"),
    ("sessions", "phases.profiling.sessions"),
    ("users", "phases.profiling.users"),
    ("smoothness", "phases.profiling.smoothness"),
    ("grace_period", "phases.profiling.grace_period"),
    ("concurrency_ramp", "phases.profiling.concurrency_ramp"),
    ("prefill_ramp", "phases.profiling.prefill_ramp"),
    ("rate_ramp", "phases.profiling.rate_ramp"),
];

/// First segments that target the envelope, not a `benchmark` field.
const NON_SWEEPABLE_FIRST: &[&str] = &["sweep", "multi_run", "random_seed"];

/// One expanded YAML-sweep variation.
pub struct YamlVariation {
    /// Zero-based combination index.
    pub index: usize,
    /// `"<dotted_path>=<value>, ..."` label (sorted by path).
    pub label: String,
    /// `"<leaf>_<value>__..."` directory name (sorted by path).
    pub dir_name: String,
    /// The base config with this variation's overrides applied and `sweep`
    /// removed (env-substituted, not yet Jinja-rendered).
    pub config: Value,
    /// `dotted_path -> value` (config-typed), the stamped `variation.values`.
    pub values: Vec<(String, Value)>,
}

/// The parsed `sweep:` block shape (grid or zip over dotted-path parameters).
pub struct YamlSweep {
    /// `grid` (Cartesian) or `zip` (lockstep).
    pub kind: SweepKind,
    /// Alpha-sorted `(resolved_path, values)` axes.
    axes: Vec<(String, Vec<Value>)>,
}

/// Grid vs zip combination strategy.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum SweepKind {
    Grid,
    Zip,
}

/// Normalize a config's `benchmark` subtree the way Python's
/// `normalize_benchmark_input` does at `AIPerfConfig` parse — BEFORE the sweep
/// override rewrites dotted paths, so a path like `datasets.default.prompts.isl`
/// resolves even when the config used the singular `dataset:` / `warmup:` /
/// `profiling:` / `model:` shorthands. Idempotent when already in list form.
pub fn normalize_benchmark(config: &mut Value) {
    let Some(bench) = config.get_mut("benchmark").and_then(Value::as_object_mut) else {
        return;
    };

    // warmup/profiling -> phases (each entry gets its `name`).
    if bench.contains_key("warmup") || bench.contains_key("profiling") {
        let mut phases = Vec::new();
        if let Some(mut w) = bench.remove("warmup") {
            if let Some(o) = w.as_object_mut() {
                o.entry("name")
                    .or_insert_with(|| Value::String("warmup".into()));
            }
            phases.push(w);
        }
        if let Some(mut p) = bench.remove("profiling") {
            if let Some(o) = p.as_object_mut() {
                o.entry("name")
                    .or_insert_with(|| Value::String("profiling".into()));
            }
            phases.push(p);
        }
        bench.insert("phases".into(), Value::Array(phases));
    }

    // model -> models; a bare string / string list -> {items:[{name}]}.
    if bench.contains_key("model") && !bench.contains_key("models") {
        let m = bench.remove("model").unwrap();
        bench.insert("models".into(), m);
    }
    if let Some(models) = bench.get_mut("models") {
        let items: Option<Vec<Value>> = match models {
            Value::String(s) => Some(vec![serde_json::json!({"name": s})]),
            Value::Array(a) if a.iter().all(Value::is_string) => {
                Some(a.iter().map(|s| serde_json::json!({"name": s})).collect())
            }
            _ => None,
        };
        if let Some(items) = items {
            *models = serde_json::json!({"items": items});
        }
    }

    // dataset -> datasets: [{name: default, ...}].
    if bench.contains_key("dataset") && !bench.contains_key("datasets") {
        let mut ds = bench.remove("dataset").unwrap();
        if let Some(o) = ds.as_object_mut() {
            let mut named = Map::new();
            named.insert("name".into(), Value::String("default".into()));
            for (k, v) in o.iter() {
                named.insert(k.clone(), v.clone());
            }
            ds = Value::Object(named);
        }
        bench.insert("datasets".into(), Value::Array(vec![ds]));
    }

    // A flat single-phase shorthand `phases: {type: ...}` -> a named list.
    if let Some(phases) = bench.get_mut("phases")
        && phases.is_object()
        && phases.get("type").is_some()
    {
        let mut named = Map::new();
        named.insert("name".into(), Value::String("profiling".into()));
        for (k, v) in phases.as_object().unwrap() {
            named.insert(k.clone(), v.clone());
        }
        *phases = Value::Array(vec![Value::Object(named)]);
    }
}

/// Read and validate the `sweep:` block off a config value, returning `None`
/// when the config has no sweep block.
pub fn parse(config: &Value) -> anyhow::Result<Option<YamlSweep>> {
    let Some(sweep) = config.get("sweep").and_then(Value::as_object) else {
        return Ok(None);
    };
    let kind = match sweep.get("type").and_then(Value::as_str) {
        Some("grid") | None => SweepKind::Grid,
        Some("zip") => SweepKind::Zip,
        Some(other) => anyhow::bail!("unknown sweep type {other:?} (expected grid/zip)"),
    };
    let params = sweep
        .get("parameters")
        .and_then(Value::as_object)
        .ok_or_else(|| anyhow::anyhow!("sweep block requires a `parameters:` map"))?;

    // Resolve each parameter path (alias + validation); alpha-sort by resolved
    // path for a stable variation order.
    let mut axes: BTreeMap<String, Vec<Value>> = BTreeMap::new();
    for (path, values) in params {
        let values = values.as_array().filter(|v| !v.is_empty()).ok_or_else(|| {
            anyhow::anyhow!("sweep parameter {path:?}: value list must be non-empty")
        })?;
        let resolved = validate_dotted_path(path)?;
        if axes.contains_key(&resolved) {
            anyhow::bail!("sweep parameter {path:?} resolves to {resolved:?}, already a parameter");
        }
        axes.insert(resolved, values.clone());
    }
    if axes.is_empty() {
        anyhow::bail!("sweep block has no parameters");
    }
    if kind == SweepKind::Zip {
        let len = axes.values().next().map(Vec::len).unwrap_or(0);
        anyhow::ensure!(
            axes.values().all(|v| v.len() == len),
            "zip sweep parameters must all have equal length"
        );
    }
    Ok(Some(YamlSweep {
        kind,
        axes: axes.into_iter().collect(),
    }))
}

impl YamlSweep {
    /// Expand into per-variation configs. `base` is the env-substituted config
    /// value (with the `sweep:` block still present; it is dropped per variation).
    pub fn expand(&self, base: &Value) -> anyhow::Result<Vec<YamlVariation>> {
        let combos: Vec<Vec<usize>> = match self.kind {
            SweepKind::Grid => cartesian(&self.axes),
            SweepKind::Zip => {
                let len = self.axes.first().map(|(_, v)| v.len()).unwrap_or(0);
                (0..len).map(|i| vec![i; self.axes.len()]).collect()
            }
        };

        let mut out = Vec::with_capacity(combos.len());
        for (index, combo) in combos.into_iter().enumerate() {
            let mut config = base.clone();
            // Overrides write into the `benchmark` subtree; drop the sweep key.
            if let Some(obj) = config.as_object_mut() {
                obj.remove("sweep");
            }
            let benchmark = config
                .as_object_mut()
                .and_then(|o| {
                    o.entry("benchmark")
                        .or_insert_with(|| Value::Object(Map::new()));
                    o.get_mut("benchmark")
                })
                .ok_or_else(|| anyhow::anyhow!("config root must be a mapping"))?;

            let mut label_parts = Vec::new();
            let mut dir_parts = Vec::new();
            let mut values = Vec::new();
            for ((path, axis_values), &vi) in self.axes.iter().zip(combo.iter()) {
                let value = axis_values[vi].clone();
                set_nested_value(benchmark, path, value.clone())?;
                label_parts.push(format!("{path}={}", render_value(&value)));
                let leaf = path.rsplit('.').next().unwrap_or(path);
                dir_parts.push(format!("{leaf}_{}", render_value(&value)));
                values.push((path.clone(), value));
            }
            out.push(YamlVariation {
                index,
                label: label_parts.join(", "),
                dir_name: dir_parts.join("__"),
                config,
                values,
            });
        }
        Ok(out)
    }
}

/// Cartesian product of axis value indices (last axis varies fastest — matches
/// Python's `itertools.product`).
fn cartesian(axes: &[(String, Vec<Value>)]) -> Vec<Vec<usize>> {
    let mut out = vec![vec![]];
    for (_, values) in axes {
        let mut next = Vec::new();
        for prefix in &out {
            for i in 0..values.len() {
                let mut p = prefix.clone();
                p.push(i);
                next.push(p);
            }
        }
        out = next;
    }
    out
}

/// Python `str(value)` for a config-typed sweep value: an integer renders bare,
/// a float via Python's `str(float)` (`10` → `10.0`), a string verbatim.
fn render_value(v: &Value) -> String {
    match v {
        Value::Number(n) if n.is_i64() || n.is_u64() => n.to_string(),
        Value::Number(n) => n.as_f64().map(py_float).unwrap_or_else(|| n.to_string()),
        Value::String(s) => s.clone(),
        Value::Bool(b) => {
            // Python str(True) == "True".
            if *b { "True".into() } else { "False".into() }
        }
        other => other.to_string(),
    }
}

/// Python `str(float)` (`10.0` for whole numbers, else the shortest repr).
fn py_float(v: f64) -> String {
    if v.fract() == 0.0 && v.is_finite() {
        format!("{v:.1}")
    } else {
        format!("{v}")
    }
}

/// Resolve a bare-name alias, then validate the dotted path (`_validate_dotted_path`).
fn validate_dotted_path(p: &str) -> anyhow::Result<String> {
    // Bare name (no dot) resolves through the alias table.
    let resolved = if !p.contains('.') {
        ALIASES
            .iter()
            .find(|(k, _)| *k == p)
            .map(|(_, v)| v.to_string())
            .unwrap_or_else(|| p.to_string())
    } else {
        p.to_string()
    };
    anyhow::ensure!(!resolved.is_empty(), "dimension path must be non-empty");
    anyhow::ensure!(
        !resolved.starts_with('.'),
        "dimension path {resolved:?} must not start with '.'"
    );
    anyhow::ensure!(
        !resolved.ends_with('.'),
        "dimension path {resolved:?} must not end with '.'"
    );
    anyhow::ensure!(
        !resolved.contains(".."),
        "dimension path {resolved:?} must not contain '..'"
    );
    let first = resolved.split('.').next().unwrap_or("");
    if first == "sweep" {
        anyhow::bail!("dimension path {resolved:?} targets the sweep config itself");
    }
    if NON_SWEEPABLE_FIRST.contains(&first) {
        anyhow::bail!(
            "dimension path {resolved:?} targets non-sweepable top-level field {first:?}"
        );
    }
    if first == "benchmark" {
        anyhow::bail!(
            "dimension path {resolved:?} must not include the redundant 'benchmark.' prefix"
        );
    }
    Ok(resolved)
}

/// True if every entry of `arr` is an object carrying a string `name`.
fn is_named_dict_list(arr: &[Value]) -> bool {
    !arr.is_empty()
        && arr
            .iter()
            .all(|it| it.get("name").and_then(Value::as_str).is_some())
}

/// Find the list entry named `name`, with the `phases.profiling` recipe fallback
/// (unique non-warmup phase). Returns the index into `arr`.
fn find_named_index(arr: &[Value], name: &str, parent_key: &str) -> Option<usize> {
    if let Some(i) = arr
        .iter()
        .position(|it| it.get("name").and_then(Value::as_str) == Some(name))
    {
        return Some(i);
    }
    if name != "profiling" || parent_key != "phases" {
        return None;
    }
    let candidates: Vec<usize> = arr
        .iter()
        .enumerate()
        .filter(|(_, it)| it.get("name").and_then(Value::as_str) != Some("warmup"))
        .map(|(i, _)| i)
        .collect();
    if candidates.len() == 1 {
        Some(candidates[0])
    } else {
        None
    }
}

/// Write `value` into `data` at a dotted path, traversing dicts by key and
/// lists-of-named-dicts by `name` (`_set_nested_value`).
fn set_nested_value(data: &mut Value, path: &str, value: Value) -> anyhow::Result<()> {
    let keys: Vec<&str> = path.split('.').collect();
    let mut current = data;
    for i in 0..keys.len() - 1 {
        let key = keys[i];
        if current.is_array() && is_named_dict_list(current.as_array().unwrap()) {
            let parent = if i > 0 { keys[i - 1] } else { "" };
            let arr = current.as_array().unwrap();
            let idx = find_named_index(arr, key, parent).ok_or_else(|| {
                anyhow::anyhow!("sweep path {path:?}: no entry named {key:?} found")
            })?;
            current = &mut current[idx];
            continue;
        }
        let obj = current
            .as_object_mut()
            .ok_or_else(|| anyhow::anyhow!("sweep path {path:?}: cannot descend at {key:?}"))?;
        current = obj.entry(key).or_insert_with(|| Value::Object(Map::new()));
    }
    let last = keys[keys.len() - 1];
    if current.is_array() && is_named_dict_list(current.as_array().unwrap()) {
        let parent = if keys.len() >= 2 {
            keys[keys.len() - 2]
        } else {
            ""
        };
        let arr = current.as_array().unwrap();
        let idx = find_named_index(arr, last, parent)
            .ok_or_else(|| anyhow::anyhow!("sweep path {path:?}: no entry named {last:?} found"))?;
        if let Some(obj) = current[idx].as_object_mut() {
            obj.insert(last.to_string(), value);
        }
    } else if let Some(obj) = current.as_object_mut() {
        obj.insert(last.to_string(), value);
    } else {
        anyhow::bail!("sweep path {path:?}: cannot assign at {last:?}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alias_and_grid_expand() {
        let base = serde_json::json!({
            "benchmark": {"phases": [{"name": "profiling", "type": "constant", "rate": 1.0}]},
        });
        let sweep = serde_json::json!({
            "type": "grid",
            "parameters": {"rate": [10.0, 30.0], "datasets.default.prompts.isl": [128, 512]},
        });
        let mut cfg = base.clone();
        cfg.as_object_mut().unwrap().insert("sweep".into(), sweep);
        let parsed = parse(&cfg).unwrap().unwrap();
        let vars = parsed.expand(&cfg).unwrap();
        assert_eq!(vars.len(), 4);
        // Alpha-sorted axes: datasets.* before phases.* (rate -> phases.profiling.rate).
        assert_eq!(
            vars[0].label,
            "datasets.default.prompts.isl=128, phases.profiling.rate=10.0"
        );
        assert_eq!(vars[0].dir_name, "isl_128__rate_10.0");
        // The rate override landed on the profiling phase.
        let phases = &vars[0].config["benchmark"]["phases"];
        assert_eq!(phases[0]["rate"], Value::from(10.0));
        // The sweep key is dropped from each variation config.
        assert!(vars[0].config.get("sweep").is_none());
    }
}
