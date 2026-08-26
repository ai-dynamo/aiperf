// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Config preprocessing for `${ENV}` substitution and Jinja2 rendering.
//!
//! The pipeline runs on the parsed YAML value tree before typed deserialization:
//!
//! 1. `${VAR}` / `${VAR:default}` substitution from the process environment. A
//!    string that is *entirely* one reference is coerced to bool/int/float; an
//!    embedded reference stays a string.
//! 2. The `variables:` block is resolved in dependency order into a flat
//!    template context. Each variable may reference any other variable.
//! 3. Jinja2 `{{ expr }}` / `{% ... %}` rendering over every string leaf, with
//!    `StrictUndefined` (a missing name is a hard error) and result coercion.
//!    `template`/`body`/`payload_template` fields are skipped because the
//!    endpoint renders them per request, and the `artifacts.user_files` subtree
//!    is skipped because its leaves are serialized verbatim.
//!
//! The Jinja engine is `minijinja`; its
//! expression semantics match Jinja2 for the arithmetic/attribute forms the
//! templates use.

use std::sync::LazyLock;

use minijinja::{Environment, UndefinedBehavior, Value as JinjaValue};
use regex::Regex;
use serde_json::{Map, Value};

/// `${VAR}` or `${VAR:default}` with any non-`}` bytes allowed in the default.
static ENV_VAR: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::([^}]*))?\}").unwrap());
/// A string consisting entirely of one `${...}` reference.
static WHOLE_ENV_VAR: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^\$\{[A-Za-z_][A-Za-z0-9_]*(?::[^}]*)?\}$").unwrap());
/// An unterminated `${...` opener.
static UNTERMINATED_ENV_VAR: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\$\{[^}\n]*$").unwrap());

/// Request-time Jinja fields that must not render during config loading.
const SKIP_TEMPLATE_FIELDS: &[&str] = &["template", "body", "payload_template"];
/// Subtrees whose string leaves are carried through verbatim, never rendered.
const SKIP_TEMPLATE_PATH_PREFIXES: &[&str] = &[
    "artifacts.user_files",
    "benchmark.artifacts.user_files",
    "artifacts.userFiles",
    "benchmark.artifacts.userFiles",
];

/// Apply env-var substitution then Jinja rendering to a parsed config value.
pub fn expand_config(root: Value) -> anyhow::Result<Value> {
    let root = substitute_env(root)?;
    render_with_context(root)
}

/// Substitute `${ENV}` before sweep expansion and per-variation Jinja rendering.
pub(crate) fn substitute_env(root: Value) -> anyhow::Result<Value> {
    substitute_env_vars(root)
}

/// Build the Jinja context and render each template string.
pub(crate) fn render_with_context(root: Value) -> anyhow::Result<Value> {
    let context = build_context(&root)?;
    render_jinja(root, &context, "")
}

fn substitute_env_vars(value: Value) -> anyhow::Result<Value> {
    match value {
        Value::String(s) => substitute_string(&s),
        Value::Array(a) => Ok(Value::Array(
            a.into_iter()
                .map(substitute_env_vars)
                .collect::<anyhow::Result<_>>()?,
        )),
        Value::Object(o) => {
            let mut out = Map::new();
            for (k, v) in o {
                out.insert(k, substitute_env_vars(v)?);
            }
            Ok(Value::Object(out))
        }
        other => Ok(other),
    }
}

fn substitute_string(text: &str) -> anyhow::Result<Value> {
    let mut missing: Option<String> = None;
    let substituted = ENV_VAR.replace_all(text, |caps: &regex::Captures| {
        let name = &caps[1];
        match std::env::var(name) {
            Ok(v) => v,
            Err(_) => match caps.get(2) {
                Some(default) => default.as_str().to_string(),
                None => {
                    if missing.is_none() {
                        missing = Some(name.to_string());
                    }
                    String::new()
                }
            },
        }
    });
    if let Some(name) = missing {
        anyhow::bail!("required environment variable ${{{name}}} is not set (no default)");
    }
    if UNTERMINATED_ENV_VAR.is_match(&substituted) {
        anyhow::bail!("unterminated environment variable reference in {substituted:?}");
    }
    // Embedded references remain strings; whole references adopt scalar types.
    if WHOLE_ENV_VAR.is_match(text) {
        Ok(coerce_scalar(&substituted))
    } else {
        Ok(Value::String(substituted.into_owned()))
    }
}

/// Coerce booleans, then integers, then floats; otherwise preserve the string.
fn coerce_scalar(s: &str) -> Value {
    let lower = s.to_ascii_lowercase();
    if lower == "true" {
        return Value::Bool(true);
    }
    if lower == "false" {
        return Value::Bool(false);
    }
    if let Ok(i) = s.parse::<i64>() {
        return Value::from(i);
    }
    if let Ok(fv) = s.parse::<f64>()
        && let Some(n) = serde_json::Number::from_f64(fv)
    {
        return Value::Number(n);
    }
    Value::String(s.to_string())
}

/// Build the Jinja context: the whole config at top level, the `benchmark`
/// subtree lifted to top level, and the resolved `variables:` block.
fn build_context(root: &Value) -> anyhow::Result<Map<String, Value>> {
    let mut ctx = Map::new();
    if let Some(obj) = root.as_object() {
        for (k, v) in obj {
            ctx.insert(k.clone(), v.clone());
        }
        // Top-level benchmark keys remain available to existing templates.
        if let Some(benchmark) = obj.get("benchmark").and_then(Value::as_object) {
            for (k, v) in benchmark {
                ctx.insert(k.clone(), v.clone());
            }
        }
        if let Some(variables) = obj.get("variables").and_then(Value::as_object) {
            let mut base = ctx.clone();
            for key in variables.keys() {
                base.remove(key);
            }
            let resolved = resolve_variables(variables, &base)?;
            for (k, v) in resolved {
                ctx.insert(k, v);
            }
        }
    }
    Ok(ctx)
}

/// Resolve the `variables:` block: each entry is a Jinja template that may
/// reference any other variable. Iterative fixpoint (a variable that references
/// an unresolved one raises `StrictUndefined` and is deferred to the next pass);
/// no progress in a pass means a cycle.
fn resolve_variables(
    variables: &Map<String, Value>,
    base: &Map<String, Value>,
) -> anyhow::Result<Map<String, Value>> {
    let mut resolved: Map<String, Value> = Map::new();
    let mut pending: Vec<(String, Value)> = variables
        .iter()
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();

    while !pending.is_empty() {
        // Stable ordering makes dependency resolution deterministic.
        pending.sort_by(|a, b| a.0.cmp(&b.0));
        let mut made_progress = false;
        let mut still_pending = Vec::new();
        for (name, template) in pending.into_iter() {
            let mut context = base.clone();
            for (k, v) in &resolved {
                context.insert(k.clone(), v.clone());
            }
            match render_value_strict(&template, &context, &format!("variables.{name}")) {
                Ok(v) => {
                    resolved.insert(name, v);
                    made_progress = true;
                }
                // Undefined names may resolve after another variable completes.
                Err(RenderError::Undefined) => still_pending.push((name, template)),
                Err(RenderError::Fatal(e)) => return Err(e),
            }
        }
        if !made_progress {
            let names: Vec<String> = still_pending.into_iter().map(|(n, _)| n).collect();
            anyhow::bail!("circular reference among variables: {}", names.join(", "));
        }
        pending = still_pending;
    }
    Ok(resolved)
}

fn path_is_skipped(path: &str) -> bool {
    SKIP_TEMPLATE_PATH_PREFIXES
        .iter()
        .any(|p| path == *p || path.starts_with(&format!("{p}.")))
}

fn render_jinja(value: Value, context: &Map<String, Value>, path: &str) -> anyhow::Result<Value> {
    if path_is_skipped(path) {
        return Ok(value);
    }
    match value {
        Value::String(s) => {
            let field = path.rsplit('.').next().unwrap_or("");
            if SKIP_TEMPLATE_FIELDS.contains(&field) {
                return Ok(Value::String(s));
            }
            match render_str_strict(&s, context, path) {
                Ok(v) => Ok(v),
                Err(RenderError::Undefined) => {
                    anyhow::bail!("Jinja2 template error at {path:?}: undefined variable in {s:?}")
                }
                Err(RenderError::Fatal(e)) => Err(e),
            }
        }
        Value::Array(a) => Ok(Value::Array(
            a.into_iter()
                .enumerate()
                .map(|(i, v)| render_jinja(v, context, &join_path(path, &i.to_string())))
                .collect::<anyhow::Result<_>>()?,
        )),
        Value::Object(o) => {
            let mut out = Map::new();
            for (k, v) in o {
                let child = join_path(path, &k);
                out.insert(k, render_jinja(v, context, &child)?);
            }
            Ok(Value::Object(out))
        }
        other => Ok(other),
    }
}

fn join_path(prefix: &str, key: &str) -> String {
    if prefix.is_empty() {
        key.to_string()
    } else {
        format!("{prefix}.{key}")
    }
}

enum RenderError {
    Undefined,
    Fatal(anyhow::Error),
}

fn render_value_strict(
    value: &Value,
    context: &Map<String, Value>,
    path: &str,
) -> Result<Value, RenderError> {
    match value {
        Value::String(s) => render_str_strict(s, context, path),
        other => Ok(other.clone()),
    }
}

/// Render a template with strict undefined names and coerce its result.
fn render_str_strict(
    data: &str,
    context: &Map<String, Value>,
    path: &str,
) -> Result<Value, RenderError> {
    if data.contains("{{") && !data.contains("}}") {
        return Err(RenderError::Fatal(anyhow::anyhow!(
            "Jinja2 template error at {path:?}: unbalanced '{{{{' with no closing '}}}}'"
        )));
    }
    if data.contains("{%") && !data.contains("%}") {
        return Err(RenderError::Fatal(anyhow::anyhow!(
            "Jinja2 template error at {path:?}: unbalanced '{{%' with no closing '%}}'"
        )));
    }
    let is_expr = data.contains("{{") && data.contains("}}");
    if !is_expr && !data.contains("{%") {
        return Ok(Value::String(data.to_string()));
    }

    let mut env = Environment::new();
    env.set_undefined_behavior(UndefinedBehavior::Strict);
    let tmpl = env.template_from_str(data).map_err(|e| {
        RenderError::Fatal(anyhow::anyhow!("Jinja2 template error at {path:?}: {e}"))
    })?;
    let jinja_ctx = JinjaValue::from_serialize(Value::Object(context.clone()));
    match tmpl.render(jinja_ctx) {
        Ok(rendered) => Ok(coerce_scalar(&rendered)),
        Err(e) if e.kind() == minijinja::ErrorKind::UndefinedError => Err(RenderError::Undefined),
        Err(e) => Err(RenderError::Fatal(anyhow::anyhow!(
            "Jinja2 template error at {path:?}: {e}"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn env_var_default_coerces_scalar() {
        let v = substitute_string("${AIPERF_TEST_UNSET:600.0}").unwrap();
        assert_eq!(v, Value::from(600.0));
        let v = substitute_string("${AIPERF_TEST_UNSET:42}").unwrap();
        assert_eq!(v, Value::from(42));
        let v = substitute_string("x-${AIPERF_TEST_UNSET:ab}-y").unwrap();
        assert_eq!(v, Value::String("x-ab-y".into()));
    }

    #[test]
    fn jinja_variables_resolve_in_dep_order() {
        let root = serde_json::json!({
            "variables": {"a": "{{ b }}", "b": 3, "c": "{{ a }}"},
            "benchmark": {"dataset": {"entries": "{{ c }}"}},
        });
        let out = expand_config(root).unwrap();
        assert_eq!(out["benchmark"]["dataset"]["entries"], Value::from(3));
    }

    #[test]
    fn jinja_arithmetic_coerces_to_int() {
        let root = serde_json::json!({
            "variables": {"base": 4, "mult": 3},
            "benchmark": {"phases": {"concurrency": "{{ base * mult }}"}},
        });
        let out = expand_config(root).unwrap();
        assert_eq!(out["benchmark"]["phases"]["concurrency"], Value::from(12));
    }

    #[test]
    fn user_files_subtree_is_skipped() {
        let root = serde_json::json!({
            "benchmark": {"artifacts": {"user_files": {"f.txt": "{{ runtime_only }}"}}},
        });
        let out = expand_config(root).unwrap();
        assert_eq!(
            out["benchmark"]["artifacts"]["user_files"]["f.txt"],
            Value::String("{{ runtime_only }}".into())
        );
    }
}
