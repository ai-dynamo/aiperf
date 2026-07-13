// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! CI/CD, config-file, markdown-doc, and test-output renderers. Port of `_coding_cicd_docs.py`.

use super::templates::TemplateRenderer;
use super::vocab::*;
use crate::recorded::RecordedTraceError;

/// `_gen_cicd_output`: a five-step CI pipeline transcript for a random toolchain.
pub(super) fn cicd_output(r: &mut TemplateRenderer) -> Result<String, RecordedTraceError> {
    let mod_ = r.pick(MODULES)?;
    let n_pass = r.number(20, 200)?;
    let n_fail = r.number(0, 5)?;
    let n_skip = r.number(0, 10)?;
    let n_pkgs = r.number(50, 300)?;
    let install_time = r.number(5, 100)? as f64 / 10.0;
    let n_lint_files = r.number(10, 100)?;
    let n_type_mods = r.number(100, 500)?;
    let coverage = r.number(700, 990)? as f64 / 10.0;
    let ver = format!(
        "{}.{}.{}",
        r.number(1, 9)?,
        r.number(0, 99)?,
        r.number(0, 99)?
    );
    let artifact_size = r.number(1, 500)? as f64 / 10.0;
    let status = if n_fail == 0 { "PASSED" } else { "FAILED" };
    let elapsed = r.number(30, 600)?;

    let (install_step, lint_step, typecheck_step, test_step, build_step) = match r.index(4)? {
        0 => (
            format!(
                "pip install -r requirements.txt\n  Resolved {n_pkgs} packages in {install_time:.1}s"
            ),
            format!(
                "ruff check . && ruff format --check .\n  All checks passed ({n_lint_files} files)"
            ),
            format!("mypy src/\n  Success: {n_type_mods} modules checked"),
            format!(
                "pytest tests/ -v\n  {n_pass} passed, {n_fail} failed, {n_skip} skipped\n  Coverage: {coverage:.1}%"
            ),
            format!("python -m build\n  Built {mod_}-{ver}.tar.gz ({artifact_size:.1} MB)"),
        ),
        1 => (
            format!("go mod download\n  Resolved {n_pkgs} packages in {install_time:.1}s"),
            format!("golangci-lint run ./...\n  All checks passed ({n_lint_files} files)"),
            format!("go vet ./...\n  Success: {n_type_mods} packages checked"),
            format!(
                "go test -v -race -coverprofile=coverage.out ./...\n  {n_pass} passed, {n_fail} failed, {n_skip} skipped\n  Coverage: {coverage:.1}%"
            ),
            format!(
                "go build -o bin/{mod_} ./cmd/{mod_}\n  Built bin/{mod_} ({artifact_size:.1} MB)"
            ),
        ),
        2 => (
            format!("cargo fetch\n  Resolved {n_pkgs} crates in {install_time:.1}s"),
            format!("cargo clippy -- -D warnings\n  All checks passed ({n_lint_files} files)"),
            format!("cargo check\n  Checked {n_type_mods} crates"),
            format!(
                "cargo test\n  {n_pass} passed, {n_fail} failed, {n_skip} ignored\n  Coverage: {coverage:.1}%"
            ),
            format!("cargo build --release\n  Built target/release/{mod_} ({artifact_size:.1} MB)"),
        ),
        _ => (
            format!("npm ci\n  Resolved {n_pkgs} packages in {install_time:.1}s"),
            format!(
                "eslint src/ && prettier --check src/\n  All checks passed ({n_lint_files} files)"
            ),
            format!("tsc --noEmit\n  Success: {n_type_mods} modules checked"),
            format!(
                "vitest run\n  {n_pass} passed, {n_fail} failed, {n_skip} skipped\n  Coverage: {coverage:.1}%"
            ),
            format!("npm run build\n  Built dist/{mod_}-{ver}.tgz ({artifact_size:.1} MB)"),
        ),
    };

    Ok(format!(
        r#"=== CI Pipeline: {mod_} ===
Step 1/5: Installing dependencies...
  {install_step}
Step 2/5: Linting...
  {lint_step}
Step 3/5: Type checking...
  {typecheck_step}
Step 4/5: Running tests...
  {test_step}
Step 5/5: Building artifacts...
  {build_step}
Pipeline {status} in {elapsed}s
"#
    ))
}

/// `_gen_config_file`: dispatch across the config-file kinds (yaml/toml/dockerfile/makefile).
pub(super) fn config_file(r: &mut TemplateRenderer) -> Result<String, RecordedTraceError> {
    let mod_ = r.pick(MODULES)?;
    let v = r.sample(VARS, 3)?;
    let (v1, v2, v3) = (v[0], v[1], v[2]);

    match r.index(4)? {
        0 => config_yaml(r, mod_, v1, v2, v3),
        1 => config_toml(r, mod_, v1, v2, v3),
        2 => config_dockerfile(r, mod_, v1, v2),
        _ => config_makefile(mod_),
    }
}

fn config_yaml(
    r: &mut TemplateRenderer,
    mod_: &str,
    v1: &str,
    v2: &str,
    v3: &str,
) -> Result<String, RecordedTraceError> {
    let port = r.number(3000, 9999)?;
    let workers = r.number(1, 16)?;
    let v2_val = r.number(1, 1000)?;
    let v3_val = r.pick(MODULES)?;
    let db_port = match r.index(4)? {
        0 => 5432,
        1 => 3306,
        2 => 27017,
        _ => 6379,
    };
    let pool = r.number(5, 50)?;
    Ok(format!(
        r#"# {mod_} configuration
service:
  name: {mod_}
  port: {port}
  workers: {workers}
  {v1}:
    enabled: true
    {v2}: {v2_val}
    {v3}: "{v3_val}"
  logging:
    level: info
    format: json
  database:
    host: localhost
    port: {db_port}
    pool_size: {pool}
"#
    ))
}

fn config_toml(
    r: &mut TemplateRenderer,
    mod_: &str,
    v1: &str,
    v2: &str,
    v3: &str,
) -> Result<String, RecordedTraceError> {
    let ver = format!(
        "{}.{}.{}",
        r.number(0, 9)?,
        r.number(0, 99)?,
        r.number(0, 99)?
    );
    let desc_cls = r.pick(CLASSES)?;
    let desc_method = r.pick(METHODS)?;
    let dep1 = r.pick(MODULES)?;
    let dep2 = r.pick(MODULES)?;
    let dep1_ver = format!("{}.{}", r.number(1, 5)?, r.number(0, 20)?);
    let dep2_ver = format!("{}.{}", r.number(0, 3)?, r.number(0, 40)?);
    let tool_mod = r.pick(MODULES)?;
    let v1_val = r.number(1, 100)?;
    let v2_val = r.pick(MODULES)?;
    Ok(format!(
        r#"[project]
name = "{mod_}"
version = "{ver}"
description = "{desc_cls} {desc_method} service"

[dependencies]
{dep1} = "{dep1_ver}"
{dep2} = "{dep2_ver}"

[tool.{tool_mod}]
{v1} = {v1_val}
{v2} = "{v2_val}"
{v3} = true
"#
    ))
}

fn config_dockerfile(
    r: &mut TemplateRenderer,
    mod_: &str,
    v1: &str,
    v2: &str,
) -> Result<String, RecordedTraceError> {
    let env1_val = r.number(1, 100)?;
    let env2_val = r.pick(MODULES)?;
    let port = r.number(3000, 9999)?;
    let py_ver = r.number(10, 13)?;
    let base_image = format!("python:3.{py_ver}-slim");
    let install_cmd = "COPY requirements.txt .\nRUN pip install --no-cache-dir -r requirements.txt";
    let run_cmd = format!(r#"CMD ["python", "-m", "{mod_}"]"#);
    let v1_upper = v1.to_uppercase();
    let v2_upper = v2.to_uppercase();
    Ok(format!(
        r#"FROM {base_image}

WORKDIR /app

{install_cmd}

COPY src/ ./src/

ENV {v1_upper}={env1_val}
ENV {v2_upper}={env2_val}

EXPOSE {port}

{run_cmd}
"#
    ))
}

fn config_makefile(mod_: &str) -> Result<String, RecordedTraceError> {
    Ok(format!(
        "\
.PHONY: build test lint clean

build:
\t@echo \"Building {mod_}...\"
\tgo build -o bin/{mod_} ./cmd/{mod_}

test:
\t@echo \"Testing {mod_}...\"
\tgo test -v -race ./...

lint:
\tgolangci-lint run ./...

clean:
\trm -rf bin/ dist/ *.egg-info
"
    ))
}

/// `_gen_markdown_doc`: an API-reference markdown page for a random class + language example.
pub(super) fn markdown_doc(r: &mut TemplateRenderer) -> Result<String, RecordedTraceError> {
    let cls = r.pick(CLASSES)?;
    let m = r.sample(METHODS, 2)?;
    let (m1, m2) = (m[0], m[1]);
    let mod_ = r.pick(MODULES)?;
    let v1 = r.pick(VARS)?;
    let err = r.pick(ERRORS)?;

    let (fence, code, param_type, return_type) = match r.index(4)? {
        0 => {
            let param_type = r
                .pick(&[
                    "str", "int", "float", "bool", "dict", "list", "Any", "Optional",
                ])?
                .to_string();
            let return_type = r
                .pick(&["str", "int", "bool", "dict", "list", "None", "Any"])?
                .to_string();
            let code = format!(
                "from {mod_} import {cls}\n\ninstance = {cls}({v1}=\"value\")\nresult = await instance.{m1}()"
            );
            ("python", code, param_type, return_type)
        }
        1 => {
            let param_type = r
                .pick(&[
                    "string",
                    "int",
                    "int64",
                    "bool",
                    "[]byte",
                    "error",
                    "context.Context",
                ])?
                .to_string();
            let m1_title = TemplateRenderer::title_case(m1);
            let return_type = match r.index(5)? {
                0 => "string".to_string(),
                1 => "int".to_string(),
                2 => "bool".to_string(),
                3 => "error".to_string(),
                _ => format!("*{cls}"),
            };
            let code = format!(
                "import \"{mod_}\"\n\nc := {mod_}.New{cls}(\"{v1}\")\nerr := c.{m1_title}(ctx)"
            );
            ("go", code, param_type, return_type)
        }
        2 => {
            let param_type = r
                .pick(&[
                    "&str",
                    "String",
                    "i64",
                    "bool",
                    "Vec<u8>",
                    "&[u8]",
                    "Option<String>",
                ])?
                .to_string();
            let return_type = r
                .pick(&[
                    "Result<()>",
                    "Result<String>",
                    "bool",
                    "Option<String>",
                    "&str",
                ])?
                .to_string();
            let code = format!(
                "use {mod_}::{cls};\n\nlet mut c = {cls}::new(\"{v1}\");\nc.{m1}().await?;"
            );
            ("rust", code, param_type, return_type)
        }
        _ => {
            let param_type = r
                .pick(&[
                    "string",
                    "number",
                    "boolean",
                    "Record<string, unknown>",
                    "unknown[]",
                ])?
                .to_string();
            let return_type = r
                .pick(&["string", "number", "boolean", "void", "Promise<void>"])?
                .to_string();
            let code = format!(
                "import {{ {cls} }} from './{mod_}';\n\nconst c = new {cls}({{ {v1}: 'value' }});\nawait c.{m1}();"
            );
            ("typescript", code, param_type, return_type)
        }
    };

    let v = r.sample(VARS, 2)?;
    let (v2, v3) = (v[0], v[1]);
    let err2 = r.pick(ERRORS)?;
    let mod_upper = mod_.to_uppercase();
    let v1_upper = v1.to_uppercase();
    let env_var = format!("AIPERF_{mod_upper}_{v1_upper}");

    Ok(format!(
        r#"# {cls}

## Overview

The `{cls}` class provides {m1} and {m2} operations for the `{mod_}` module.

## Usage

```{fence}
{code}
```

## Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `{v1}` | {param_type} | required | Primary {v1} identifier |
| `{v2}` | {param_type} | `None` | Optional {v2} override |
| `{v3}` | int | `10` | Maximum {v3} per batch |
| `timeout` | float | `30.0` | Operation timeout in seconds |

Environment variable override: `{env_var}`

## API Reference

### `{m1}({v1})`

Performs the {m1} operation.

**Parameters:**
- `{v1}` ({param_type}): The input {v1}.

**Returns:** {return_type}

### `{m2}()`

Performs the {m2} operation.

**Raises:** `ValueError` if {err}.

## Errors

| Error | Condition | Recovery |
|-------|-----------|----------|
| `ValueError` | {err} | Check {v1} parameter |
| `RuntimeError` | {err2} | Retry with backoff |
| `TimeoutError` | Operation exceeds timeout | Increase timeout or reduce {v3} |
"#
    ))
}

/// `_gen_test_output`: dispatch across test-runner transcripts (pytest/go/cargo).
pub(super) fn test_output(r: &mut TemplateRenderer) -> Result<String, RecordedTraceError> {
    let mod_ = r.pick(MODULES)?;
    let cls = r.pick(CLASSES)?;
    let methods = r.sample(METHODS, 5)?;

    match r.index(3)? {
        0 => test_output_pytest(r, mod_, cls, &methods),
        1 => test_output_go(r, mod_, &methods),
        _ => test_output_cargo(r, mod_, cls, &methods),
    }
}

fn test_output_pytest(
    r: &mut TemplateRenderer,
    mod_: &str,
    cls: &str,
    methods: &[&str],
) -> Result<String, RecordedTraceError> {
    let mut lines: Vec<String> = vec![
        "============================= test session starts ============================="
            .to_string(),
    ];
    lines.push(format!("collected {} items\n", r.number(10, 100)?));
    let mut n_pass = 0usize;
    for m in methods {
        let status = match r.index(4)? {
            3 => "FAILED",
            _ => "PASSED",
        };
        if status == "PASSED" {
            n_pass += 1;
        }
        lines.push(format!(
            "tests/test_{mod_}.py::Test{cls}::test_{m} {status}"
        ));
    }
    let n_fail = methods.len() - n_pass;
    let dur = r.number(50, 3000)? as f64 / 100.0;
    lines.push(format!("\n{}", "=".repeat(70)));
    lines.push(format!("{n_pass} passed, {n_fail} failed in {dur:.2}s"));
    Ok(lines.join("\n") + "\n")
}

fn test_output_go(
    r: &mut TemplateRenderer,
    mod_: &str,
    methods: &[&str],
) -> Result<String, RecordedTraceError> {
    let mut lines: Vec<String> = Vec::new();
    let mut status = "ok";
    for m in methods {
        status = match r.index(4)? {
            3 => "FAIL",
            _ => "ok",
        };
        let dur = r.number(1, 2000)? as f64 / 1000.0;
        let m_title = TemplateRenderer::title_case(m);
        lines.push(format!("--- {status}: Test{m_title} ({dur:.3}s)"));
    }
    let pkg = r.pick(MODULES)?;
    let total_dur = r.number(100, 5000)? as f64 / 1000.0;
    lines.push(format!("{status}  \t{mod_}/{pkg}\t{total_dur:.3}s"));
    Ok(lines.join("\n") + "\n")
}

fn test_output_cargo(
    r: &mut TemplateRenderer,
    mod_: &str,
    cls: &str,
    methods: &[&str],
) -> Result<String, RecordedTraceError> {
    let cls_lower = cls.to_lowercase();
    let mut lines: Vec<String> = vec![format!(
        "   Compiling {mod_} v0.{}.{}",
        r.number(1, 99)?,
        r.number(0, 9)?
    )];
    let finished = r.number(100, 3000)? as f64 / 100.0;
    lines.push(format!("    Finished test target(s) in {finished:.2}s"));
    lines.push("     Running unittests src/lib.rs\n".to_string());
    let mut n_pass = 0usize;
    for m in methods {
        let status = match r.index(4)? {
            3 => "FAILED",
            _ => "ok",
        };
        if status == "ok" {
            n_pass += 1;
        }
        lines.push(format!("test {mod_}::{cls_lower}::test_{m} ... {status}"));
    }
    let n_fail = methods.len() - n_pass;
    let result = if n_fail == 0 { "ok" } else { "FAILED" };
    lines.push(format!(
        "\ntest result: {result}. {n_pass} passed; {n_fail} failed; 0 ignored"
    ));
    Ok(lines.join("\n") + "\n")
}
