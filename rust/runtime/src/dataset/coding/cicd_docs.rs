// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! CI/CD, config-file, markdown-doc, and test-output renderers.

use super::CodingCorpusError;
use super::templates::TemplateRenderer;
use super::vocab::*;

/// A five-step CI pipeline transcript for a random toolchain.
///
/// `lang` selects the toolchain; the selection draw is consumed regardless.
pub(super) fn cicd_output(
    r: &mut TemplateRenderer,
    lang: Option<usize>,
) -> Result<String, CodingCorpusError> {
    let mod_ = r.pick(MODULES)?;
    let n_pass = r.number(20, 200)?;
    let n_fail = r.number(0, 5)?;
    let n_skip = r.number(0, 10)?;
    let n_pkgs = r.number(50, 300)?;
    let install_time = r.uniform(0.5, 10.0);
    let n_lint_files = r.number(10, 100)?;
    let n_type_mods = r.number(100, 500)?;
    let coverage = r.uniform(70.0, 99.0);
    let ver = format!(
        "{}.{}.{}",
        r.number(1, 9)?,
        r.number(0, 99)?,
        r.number(0, 99)?
    );
    let artifact_size = r.uniform(0.1, 50.0);
    let status = if n_fail == 0 { "PASSED" } else { "FAILED" };
    let elapsed = r.number(30, 600)?;

    let toolchain_draw = r.index(4)?;
    let sel = lang.unwrap_or(toolchain_draw);
    let (install_step, lint_step, typecheck_step, test_step, build_step) = match sel {
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

/// Dispatch across config-file kinds. The candidate kind
/// list depends on `lang` (`_lang_to_kinds`), so the `choice` index range and
/// mapping change with language; `None` uses the full 4-kind list.
pub(super) fn config_file(
    r: &mut TemplateRenderer,
    lang: Option<usize>,
) -> Result<String, CodingCorpusError> {
    let mod_ = r.pick(MODULES)?;
    let v = r.sample(VARS, 3)?;
    let (v1, v2, v3) = (v[0], v[1], v[2]);

    let choices: &[&str] = match lang {
        Some(0) => &["yaml", "toml", "dockerfile"],
        Some(1) => &["yaml", "makefile"],
        Some(2) => &["toml"],
        Some(3) => &["yaml", "dockerfile"],
        _ => &["yaml", "toml", "dockerfile", "makefile"],
    };
    let kind = choices[r.index(choices.len())?];
    match kind {
        "yaml" => config_yaml(r, mod_, v1, v2, v3),
        "toml" => config_toml(r, mod_, v1, v2, v3),
        "dockerfile" => config_dockerfile(r, mod_, v1, v2, lang),
        _ => config_makefile(mod_),
    }
}

fn config_yaml(
    r: &mut TemplateRenderer,
    mod_: &str,
    v1: &str,
    v2: &str,
    v3: &str,
) -> Result<String, CodingCorpusError> {
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
) -> Result<String, CodingCorpusError> {
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
    lang: Option<usize>,
) -> Result<String, CodingCorpusError> {
    let env1_val = r.number(1, 100)?;
    let env2_val = r.pick(MODULES)?;
    let port = r.number(3000, 9999)?;
    // `None` selects Python.
    let (base_image, install_cmd, run_cmd) = match lang {
        Some(1) => {
            let go_ver = format!("1.{}", r.number(21, 23)?);
            (
                format!("golang:{go_ver}-alpine"),
                "COPY go.mod go.sum ./\nRUN go mod download".to_string(),
                format!(r#"CMD ["./bin/{mod_}"]"#),
            )
        }
        Some(2) => (
            "rust:1-slim".to_string(),
            "COPY Cargo.toml Cargo.lock ./\nRUN cargo fetch".to_string(),
            format!(r#"CMD ["./target/release/{mod_}"]"#),
        ),
        Some(3) => {
            let node_ver = r.number(18, 22)?;
            (
                format!("node:{node_ver}-alpine"),
                "COPY package.json package-lock.json ./\nRUN npm ci".to_string(),
                format!(r#"CMD ["node", "dist/{mod_}/index.js"]"#),
            )
        }
        _ => {
            let py_ver = r.number(10, 13)?;
            (
                format!("python:3.{py_ver}-slim"),
                "COPY requirements.txt .\nRUN pip install --no-cache-dir -r requirements.txt"
                    .to_string(),
                format!(r#"CMD ["python", "-m", "{mod_}"]"#),
            )
        }
    };
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

fn config_makefile(mod_: &str) -> Result<String, CodingCorpusError> {
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

/// An API-reference page for a random class and language example.
pub(super) fn markdown_doc(r: &mut TemplateRenderer) -> Result<String, CodingCorpusError> {
    let cls = r.pick(CLASSES)?;
    let m = r.sample(METHODS, 2)?;
    let (m1, m2) = (m[0], m[1]);
    let mod_ = r.pick(MODULES)?;
    let v1 = r.pick(VARS)?;
    let err = r.pick(ERRORS)?;

    // Every language's parameter and return types are drawn before selection.
    let py_param = r
        .pick(&[
            "str", "int", "float", "bool", "dict", "list", "Any", "Optional",
        ])?
        .to_string();
    let py_return = r
        .pick(&["str", "int", "bool", "dict", "list", "None", "Any"])?
        .to_string();
    let go_param = r
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
    let go_return = match r.index(5)? {
        0 => "string".to_string(),
        1 => "int".to_string(),
        2 => "bool".to_string(),
        3 => "error".to_string(),
        _ => format!("*{cls}"),
    };
    let rust_param = r
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
    let rust_return = r
        .pick(&[
            "Result<()>",
            "Result<String>",
            "bool",
            "Option<String>",
            "&str",
        ])?
        .to_string();
    let ts_param = r
        .pick(&[
            "string",
            "number",
            "boolean",
            "Record<string, unknown>",
            "unknown[]",
        ])?
        .to_string();
    let ts_return = r
        .pick(&["string", "number", "boolean", "void", "Promise<void>"])?
        .to_string();
    let m1_title = TemplateRenderer::title_case(m1);
    let examples: [(&str, String, String, String); 4] = [
        (
            "python",
            format!(
                "from {mod_} import {cls}\n\ninstance = {cls}({v1}=\"value\")\nresult = await instance.{m1}()"
            ),
            py_param,
            py_return,
        ),
        (
            "go",
            format!(
                "import \"{mod_}\"\n\nc := {mod_}.New{cls}(\"{v1}\")\nerr := c.{m1_title}(ctx)"
            ),
            go_param,
            go_return,
        ),
        (
            "rust",
            format!("use {mod_}::{cls};\n\nlet mut c = {cls}::new(\"{v1}\");\nc.{m1}().await?;"),
            rust_param,
            rust_return,
        ),
        (
            "typescript",
            format!(
                "import {{ {cls} }} from './{mod_}';\n\nconst c = new {cls}({{ {v1}: 'value' }});\nawait c.{m1}();"
            ),
            ts_param,
            ts_return,
        ),
    ];
    let sel = r.index(4)?;
    let (fence, code, param_type, return_type) = {
        let e = &examples[sel];
        (e.0, e.1.clone(), e.2.clone(), e.3.clone())
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

/// Dispatch across pytest, Go, and Cargo test transcripts.
pub(super) fn test_output(r: &mut TemplateRenderer) -> Result<String, CodingCorpusError> {
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
) -> Result<String, CodingCorpusError> {
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
    let dur = r.uniform(0.5, 30.0);
    lines.push(format!("\n{}", "=".repeat(70)));
    lines.push(format!("{n_pass} passed, {n_fail} failed in {dur:.2}s"));
    Ok(lines.join("\n") + "\n")
}

fn test_output_go(
    r: &mut TemplateRenderer,
    mod_: &str,
    methods: &[&str],
) -> Result<String, CodingCorpusError> {
    let mut lines: Vec<String> = Vec::new();
    let mut status = "ok";
    for m in methods {
        status = match r.index(4)? {
            3 => "FAIL",
            _ => "ok",
        };
        let dur = r.uniform(0.001, 2.0);
        let m_title = TemplateRenderer::title_case(m);
        lines.push(format!("--- {status}: Test{m_title} ({dur:.3}s)"));
    }
    let pkg = r.pick(MODULES)?;
    let total_dur = r.uniform(0.1, 5.0);
    lines.push(format!("{status}  \t{mod_}/{pkg}\t{total_dur:.3}s"));
    Ok(lines.join("\n") + "\n")
}

fn test_output_cargo(
    r: &mut TemplateRenderer,
    mod_: &str,
    cls: &str,
    methods: &[&str],
) -> Result<String, CodingCorpusError> {
    let cls_lower = cls.to_lowercase();
    let mut lines: Vec<String> = vec![format!(
        "   Compiling {mod_} v0.{}.{}",
        r.number(1, 99)?,
        r.number(0, 9)?
    )];
    let finished = r.uniform(1.0, 30.0);
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
