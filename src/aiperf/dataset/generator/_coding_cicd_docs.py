# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CI/CD output, config-file, markdown-doc, and test-output generators.

Extracted from ``coding_content.py`` to keep that module under the
ergonomics file-size cap. Methods read ``self._template_rng`` and the
shared vocabulary tuples; behavior is unchanged.
"""

from __future__ import annotations

from aiperf.dataset.generator._coding_vocab import (
    _CLASSES,
    _ERROR_MESSAGES,
    _METHODS,
    _MODULES,
    _VARS,
)


class _CicdDocsMixin:
    def _gen_cicd_output(self, language: str | None = None) -> str:
        r = self._template_rng
        mod = r.choice(_MODULES)
        n_pass = r.randint(20, 200)
        n_fail = r.randint(0, 5)
        n_skip = r.randint(0, 10)
        n_pkgs = r.randint(50, 300)
        install_time = r.uniform(0.5, 10)
        n_lint_files = r.randint(10, 100)
        n_type_mods = r.randint(100, 500)
        coverage = r.uniform(70, 99)
        ver = f"{r.randint(1, 9)}.{r.randint(0, 99)}.{r.randint(0, 99)}"
        artifact_size = r.uniform(0.1, 50)
        status = "PASSED" if n_fail == 0 else "FAILED"
        elapsed = r.randint(30, 600)

        lang_toolchain = {
            "python": {
                "install": f"pip install -r requirements.txt\n  Resolved {n_pkgs} packages in {install_time:.1f}s",
                "lint": f"ruff check . && ruff format --check .\n  All checks passed ({n_lint_files} files)",
                "typecheck": f"mypy src/\n  Success: {n_type_mods} modules checked",
                "test": f"pytest tests/ -v\n  {n_pass} passed, {n_fail} failed, {n_skip} skipped\n  Coverage: {coverage:.1f}%",
                "build": f"python -m build\n  Built {mod}-{ver}.tar.gz ({artifact_size:.1f} MB)",
            },
            "go": {
                "install": f"go mod download\n  Resolved {n_pkgs} packages in {install_time:.1f}s",
                "lint": f"golangci-lint run ./...\n  All checks passed ({n_lint_files} files)",
                "typecheck": f"go vet ./...\n  Success: {n_type_mods} packages checked",
                "test": f"go test -v -race -coverprofile=coverage.out ./...\n  {n_pass} passed, {n_fail} failed, {n_skip} skipped\n  Coverage: {coverage:.1f}%",
                "build": f"go build -o bin/{mod} ./cmd/{mod}\n  Built bin/{mod} ({artifact_size:.1f} MB)",
            },
            "rust": {
                "install": f"cargo fetch\n  Resolved {n_pkgs} crates in {install_time:.1f}s",
                "lint": f"cargo clippy -- -D warnings\n  All checks passed ({n_lint_files} files)",
                "typecheck": f"cargo check\n  Checked {n_type_mods} crates",
                "test": f"cargo test\n  {n_pass} passed, {n_fail} failed, {n_skip} ignored\n  Coverage: {coverage:.1f}%",
                "build": f"cargo build --release\n  Built target/release/{mod} ({artifact_size:.1f} MB)",
            },
            "typescript": {
                "install": f"npm ci\n  Resolved {n_pkgs} packages in {install_time:.1f}s",
                "lint": f"eslint src/ && prettier --check src/\n  All checks passed ({n_lint_files} files)",
                "typecheck": f"tsc --noEmit\n  Success: {n_type_mods} modules checked",
                "test": f"vitest run\n  {n_pass} passed, {n_fail} failed, {n_skip} skipped\n  Coverage: {coverage:.1f}%",
                "build": f"npm run build\n  Built dist/{mod}-{ver}.tgz ({artifact_size:.1f} MB)",
            },
        }
        toolchain = lang_toolchain.get(
            language, r.choice(list(lang_toolchain.values()))
        )

        return f"""\
=== CI Pipeline: {mod} ===
Step 1/5: Installing dependencies...
  {toolchain["install"]}
Step 2/5: Linting...
  {toolchain["lint"]}
Step 3/5: Type checking...
  {toolchain["typecheck"]}
Step 4/5: Running tests...
  {toolchain["test"]}
Step 5/5: Building artifacts...
  {toolchain["build"]}
Pipeline {status} in {elapsed}s
"""

    def _gen_config_file(self, language: str | None = None) -> str:
        r = self._template_rng
        mod = r.choice(_MODULES)
        v1, v2, v3 = r.sample(_VARS, 3)

        lang_to_kinds: dict[str, list[str]] = {
            "python": ["yaml", "toml", "dockerfile"],
            "go": ["yaml", "makefile"],
            "rust": ["toml"],
            "typescript": ["yaml", "dockerfile"],
        }
        choices = (
            lang_to_kinds.get(language, ["yaml", "toml", "dockerfile", "makefile"])
            if language
            else ["yaml", "toml", "dockerfile", "makefile"]
        )
        kind = r.choice(choices)
        if kind == "yaml":
            return self._gen_config_yaml(mod, v1, v2, v3)
        if kind == "toml":
            return self._gen_config_toml(mod, v1, v2, v3)
        if kind == "dockerfile":
            return self._gen_config_dockerfile(mod, v1, v2, language)
        return self._gen_config_makefile(mod)

    def _gen_config_yaml(self, mod: str, v1: str, v2: str, v3: str) -> str:
        r = self._template_rng
        port = r.randint(3000, 9999)
        workers = r.randint(1, 16)
        v2_val = r.randint(1, 1000)
        v3_val = r.choice(_MODULES)
        db_port = r.choice([5432, 3306, 27017, 6379])
        pool = r.randint(5, 50)
        return f"""\
# {mod} configuration
service:
  name: {mod}
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
"""

    def _gen_config_toml(self, mod: str, v1: str, v2: str, v3: str) -> str:
        r = self._template_rng
        ver = f"{r.randint(0, 9)}.{r.randint(0, 99)}.{r.randint(0, 99)}"
        desc_cls = r.choice(_CLASSES)
        desc_method = r.choice(_METHODS)
        dep1, dep2 = r.choice(_MODULES), r.choice(_MODULES)
        dep1_ver = f"{r.randint(1, 5)}.{r.randint(0, 20)}"
        dep2_ver = f"{r.randint(0, 3)}.{r.randint(0, 40)}"
        tool_mod = r.choice(_MODULES)
        v1_val = r.randint(1, 100)
        v2_val = r.choice(_MODULES)
        return f"""\
[project]
name = "{mod}"
version = "{ver}"
description = "{desc_cls} {desc_method} service"

[dependencies]
{dep1} = "{dep1_ver}"
{dep2} = "{dep2_ver}"

[tool.{tool_mod}]
{v1} = {v1_val}
{v2} = "{v2_val}"
{v3} = true
"""

    def _gen_config_dockerfile(
        self, mod: str, v1: str, v2: str, language: str | None
    ) -> str:
        r = self._template_rng
        env1_val = r.randint(1, 100)
        env2_val = r.choice(_MODULES)
        port = r.randint(3000, 9999)
        docker_lang = language or "python"
        if docker_lang == "python":
            py_ver = r.randint(10, 13)
            base_image = f"python:3.{py_ver}-slim"
            install_cmd = "COPY requirements.txt .\nRUN pip install --no-cache-dir -r requirements.txt"
            run_cmd = f'CMD ["python", "-m", "{mod}"]'
        elif docker_lang == "go":
            go_ver = f"1.{r.randint(21, 23)}"
            base_image = f"golang:{go_ver}-alpine"
            install_cmd = "COPY go.mod go.sum ./\nRUN go mod download"
            run_cmd = f'CMD ["./bin/{mod}"]'
        elif docker_lang == "rust":
            base_image = "rust:1-slim"
            install_cmd = "COPY Cargo.toml Cargo.lock ./\nRUN cargo fetch"
            run_cmd = f'CMD ["./target/release/{mod}"]'
        else:
            node_ver = r.randint(18, 22)
            base_image = f"node:{node_ver}-alpine"
            install_cmd = "COPY package.json package-lock.json ./\nRUN npm ci"
            run_cmd = f'CMD ["node", "dist/{mod}/index.js"]'
        return f"""\
FROM {base_image}

WORKDIR /app

{install_cmd}

COPY src/ ./src/

ENV {v1.upper()}={env1_val}
ENV {v2.upper()}={env2_val}

EXPOSE {port}

{run_cmd}
"""

    def _gen_config_makefile(self, mod: str) -> str:
        return f"""\
.PHONY: build test lint clean

build:
\t@echo "Building {mod}..."
\tgo build -o bin/{mod} ./cmd/{mod}

test:
\t@echo "Testing {mod}..."
\tgo test -v -race ./...

lint:
\tgolangci-lint run ./...

clean:
\trm -rf bin/ dist/ *.egg-info
"""

    def _gen_markdown_doc(self, language: str | None = None) -> str:
        r = self._template_rng
        cls = r.choice(_CLASSES)
        m1, m2 = r.sample(_METHODS, 2)
        mod = r.choice(_MODULES)
        v1 = r.choice(_VARS)
        err = r.choice(_ERROR_MESSAGES)

        lang_examples = self._build_markdown_lang_examples(cls, m1, mod, v1)
        example = lang_examples.get(language, r.choice(list(lang_examples.values())))

        v2, v3 = r.sample(_VARS, 2)
        err2 = r.choice(_ERROR_MESSAGES)
        env_var = f"AIPERF_{mod.upper()}_{v1.upper()}"

        return f"""\
# {cls}

## Overview

The `{cls}` class provides {m1} and {m2} operations for the `{mod}` module.

## Usage

```{example["fence"]}
{example["code"]}
```

## Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `{v1}` | {example["param_type"]} | required | Primary {v1} identifier |
| `{v2}` | {example["param_type"]} | `None` | Optional {v2} override |
| `{v3}` | int | `10` | Maximum {v3} per batch |
| `timeout` | float | `30.0` | Operation timeout in seconds |

Environment variable override: `{env_var}`

## API Reference

### `{m1}({v1})`

Performs the {m1} operation.

**Parameters:**
- `{v1}` ({example["param_type"]}): The input {v1}.

**Returns:** {example["return_type"]}

### `{m2}()`

Performs the {m2} operation.

**Raises:** `ValueError` if {err}.

## Errors

| Error | Condition | Recovery |
|-------|-----------|----------|
| `ValueError` | {err} | Check {v1} parameter |
| `RuntimeError` | {err2} | Retry with backoff |
| `TimeoutError` | Operation exceeds timeout | Increase timeout or reduce {v3} |
"""

    def _build_markdown_lang_examples(
        self, cls: str, m1: str, mod: str, v1: str
    ) -> dict[str, dict[str, str]]:
        r = self._template_rng
        return {
            "python": {
                "fence": "python",
                "code": f'from {mod} import {cls}\n\ninstance = {cls}({v1}="value")\nresult = await instance.{m1}()',
                "param_type": r.choice(
                    ("str", "int", "float", "bool", "dict", "list", "Any", "Optional")
                ),
                "return_type": r.choice(
                    ("str", "int", "bool", "dict", "list", "None", "Any")
                ),
            },
            "go": {
                "fence": "go",
                "code": f'import "{mod}"\n\nc := {mod}.New{cls}("{v1}")\nerr := c.{m1.title()}(ctx)',
                "param_type": r.choice(
                    (
                        "string",
                        "int",
                        "int64",
                        "bool",
                        "[]byte",
                        "error",
                        "context.Context",
                    )
                ),
                "return_type": r.choice(("string", "int", "bool", "error", f"*{cls}")),
            },
            "rust": {
                "fence": "rust",
                "code": f'use {mod}::{cls};\n\nlet mut c = {cls}::new("{v1}");\nc.{m1}().await?;',
                "param_type": r.choice(
                    (
                        "&str",
                        "String",
                        "i64",
                        "bool",
                        "Vec<u8>",
                        "&[u8]",
                        "Option<String>",
                    )
                ),
                "return_type": r.choice(
                    ("Result<()>", "Result<String>", "bool", "Option<String>", "&str")
                ),
            },
            "typescript": {
                "fence": "typescript",
                "code": f"import {{ {cls} }} from './{mod}';\n\nconst c = new {cls}({{ {v1}: 'value' }});\nawait c.{m1}();",
                "param_type": r.choice(
                    (
                        "string",
                        "number",
                        "boolean",
                        "Record<string, unknown>",
                        "unknown[]",
                    )
                ),
                "return_type": r.choice(
                    ("string", "number", "boolean", "void", "Promise<void>")
                ),
            },
        }

    def _gen_test_output(self, language: str | None = None) -> str:
        r = self._template_rng
        mod = r.choice(_MODULES)
        cls = r.choice(_CLASSES)
        methods = r.sample(list(_METHODS), 5)

        lang_to_kind = {
            "python": "pytest",
            "go": "go",
            "rust": "cargo",
            "typescript": "jest",
        }
        kind = (
            lang_to_kind[language]
            if language in lang_to_kind
            else r.choice(["pytest", "go", "cargo"])
        )
        if kind == "pytest":
            lines = [
                "============================= test session starts ============================="
            ]
            lines.append(f"collected {r.randint(10, 100)} items\n")
            for m in methods:
                status = r.choice(["PASSED", "PASSED", "PASSED", "FAILED"])
                lines.append(f"tests/test_{mod}.py::Test{cls}::test_{m} {status}")
            n_pass = sum(1 for line in lines if "PASSED" in line)
            n_fail = len(methods) - n_pass
            dur = r.uniform(0.5, 30.0)
            lines.append(f"\n{'=' * 70}")
            lines.append(f"{n_pass} passed, {n_fail} failed in {dur:.2f}s")
            return "\n".join(lines) + "\n"
        elif kind == "jest":
            runner = r.choice(["JEST", "VITEST"])
            lines = [
                f" {runner}  v{r.randint(28, 30)}.{r.randint(0, 9)}.{r.randint(0, 9)}"
            ]
            lines.append("")
            results: list[str] = []
            for m in methods:
                passed = r.choice([True, True, True, False])
                mark = "\u2713" if passed else "\u2717"
                dur_ms = r.randint(1, 500)
                results.append(f"  {mark} {cls} > {m} ({dur_ms} ms)")
                lines.append(results[-1])
            n_pass = sum(1 for res in results if "\u2713" in res)
            n_fail = len(methods) - n_pass
            dur = r.uniform(0.5, 15.0)
            lines.append("")
            lines.append(
                f"Tests:       {n_fail} failed, {n_pass} passed, {len(methods)} total"
            )
            lines.append(f"Time:        {dur:.3f} s")
            lines.append(f"Ran all test suites matching /src/{mod}.test.ts/i.")
            return "\n".join(lines) + "\n"
        elif kind == "go":
            lines = []
            for m in methods:
                status = r.choice(["ok", "ok", "ok", "FAIL"])
                dur = r.uniform(0.001, 2.0)
                lines.append(f"--- {status}: Test{m.title()} ({dur:.3f}s)")
            lines.append(
                f"{status}  \t{mod}/{r.choice(_MODULES)}\t{r.uniform(0.1, 5.0):.3f}s"
            )
            return "\n".join(lines) + "\n"
        else:
            lines = [f"   Compiling {mod} v0.{r.randint(1, 99)}.{r.randint(0, 9)}"]
            lines.append(f"    Finished test target(s) in {r.uniform(1, 30):.2f}s")
            lines.append("     Running unittests src/lib.rs\n")
            for m in methods:
                status = r.choice(["ok", "ok", "ok", "FAILED"])
                lines.append(f"test {mod}::{cls.lower()}::test_{m} ... {status}")
            n_pass = sum(1 for line in lines if "... ok" in line)
            n_fail = len(methods) - n_pass
            lines.append(
                f"\ntest result: {'ok' if n_fail == 0 else 'FAILED'}. "
                f"{n_pass} passed; {n_fail} failed; 0 ignored"
            )
            return "\n".join(lines) + "\n"
