# `aiperf kube show` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `aiperf kube show --path <file>` — a read-only CLI command that prints an AIPerfJob CR with Jinja2 templates and `${ENV_VAR}` substitutions in `spec.benchmark` fully rendered, while passing through `metadata` and every non-benchmark `spec.*` key untouched.

**Architecture:** Single new cyclopts subcommand module (`src/aiperf/cli_commands/kube/show.py`) registered under the existing `aiperf kube` app. Core pipeline: `yaml.safe_load` → structural checks (reuse `kubernetes/validate.py::validate_yaml_structure`) → `operator.spec_converter.extract_benchmark_config()` (does `expand_config_dict` + `AIPerfConfig.model_validate` with no K8s runtime injection) → `config.dump_config()` → re-inject rendered benchmark into the original document → `yaml.safe_dump(..., sort_keys=False)` → stdout.

**Tech Stack:** Python 3.10+, cyclopts (CLI), pyyaml, pydantic (via existing helpers).

---

## File Structure

- **Create:** `src/aiperf/cli_commands/kube/show.py` (~70 lines incl. imports and docstring).
  - Owns: command flag parsing, orchestration, stdout output.
  - Pure; no network, no K8s API calls.
- **Create:** `tests/unit/cli_commands/kube/test_show.py`
  - Owns: unit tests for every branch of `show.py`.
- **Modify:** `src/aiperf/cli_commands/kube/_app.py`
  - Change: add one `app.command(...)` registration for `show`, alphabetical between `results` and `validate` (matches existing order pattern).

No other production files change. Docs regenerate via the `generate-cli-docs` pre-commit hook.

---

## Task 1: Scaffold the command and register it under `aiperf kube`

**Goal:** Create an empty `show` subcommand that is importable, appears in `aiperf kube --help`, and raises a clear message if invoked. This isolates the cyclopts wiring from the real logic so later tasks focus on behaviour.

**Files:**
- Create: `src/aiperf/cli_commands/kube/show.py`
- Modify: `src/aiperf/cli_commands/kube/_app.py` — add one command registration.
- Test: `tests/unit/cli_commands/kube/test_show.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/cli_commands/kube/test_show.py` with:

```python
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for aiperf kube show — render AIPerfJob CR with Jinja2/env-vars resolved."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml


def _minimal_cr() -> dict:
    """Minimal valid AIPerfJob CR dict."""
    return {
        "apiVersion": "aiperf.nvidia.com/v1alpha1",
        "kind": "AIPerfJob",
        "metadata": {"name": "test-job"},
        "spec": {
            "image": "nvcr.io/nvidia/aiperf:latest",
            "benchmark": {
                "models": ["test-model"],
                "endpoint": {"urls": ["http://localhost:8000/v1/chat/completions"]},
                "datasets": {
                    "main": {
                        "type": "synthetic",
                        "entries": 10,
                        "prompts": {"isl": 32, "osl": 16},
                    }
                },
                "phases": {
                    "default": {"type": "concurrency", "requests": 10, "concurrency": 1}
                },
            },
        },
    }


def _write(path: Path, doc: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.dump(doc, sort_keys=False))
    return path


def test_show_module_importable() -> None:
    """The show module must be importable and expose an `app` attribute."""
    from aiperf.cli_commands.kube import show

    assert hasattr(show, "app"), "show.app (cyclopts App) must be defined"


def test_show_registered_in_kube_app() -> None:
    """The `show` subcommand must be wired into `aiperf kube`."""
    from aiperf.cli_commands.kube._app import app

    # cyclopts exposes registered subcommands via app._commands or iteration.
    # We only care that a command named "show" exists.
    command_names = {cmd.name for cmd in app}  # cyclopts App is iterable
    assert "show" in command_names
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/cli_commands/kube/test_show.py -n auto -v`
Expected: both tests FAIL — `ModuleNotFoundError: aiperf.cli_commands.kube.show` for the first, and the second errors on the missing module import as well.

- [ ] **Step 3: Create the stub `show.py`**

Create `src/aiperf/cli_commands/kube/show.py`:

```python
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kube show command: render AIPerfJob CR with Jinja2/env-vars resolved."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from cyclopts import App, Parameter

app = App(name="show")


@app.default
def show(
    *,
    path: Annotated[
        Path,
        Parameter(
            name=["-p", "--path"],
            help="Path to an AIPerfJob YAML file.",
        ),
    ],
) -> None:
    """Render an AIPerfJob CR with Jinja2 and env-var templates resolved.

    Reads the CR, expands ``{{ ... }}`` expressions and ``${ENV_VAR}``
    substitutions inside ``spec.benchmark``, validates the result against
    ``AIPerfConfig``, re-wraps it in the original ``metadata`` and
    non-benchmark ``spec.*`` fields, and prints YAML to stdout.

    Examples:
        aiperf kube show --path recipes/qwen3-32b-fp8/trtllm/agg/perf.yaml
    """
    raise NotImplementedError("show command not yet implemented")
```

- [ ] **Step 4: Register `show` in the kube app**

Modify `src/aiperf/cli_commands/kube/_app.py`. Between the `results` and `debug` registrations (roughly alphabetical among the read-only inspection commands), add:

```python
app.command(
    "aiperf.cli_commands.kube.show:app",
    name="show",
    help="Render an AIPerfJob CR with Jinja2/env-vars resolved",
)
```

- [ ] **Step 5: Run test to verify first test passes, second passes**

Run: `uv run pytest tests/unit/cli_commands/kube/test_show.py -n auto -v`
Expected: both tests PASS.

If `test_show_registered_in_kube_app` fails because `app` is not iterable the way the test assumed, inspect with:

```python
>>> from aiperf.cli_commands.kube._app import app
>>> dir(app)
```

…and adjust the assertion to whatever cyclopts exposes (e.g. `app._commands`, `app.subapps`, etc.). Keep the test's intent: confirm a command named `show` is registered.

- [ ] **Step 6: Confirm CLI help includes `show`**

Run: `uv run aiperf kube --help 2>&1 | grep -E '^\│ show'`
Expected: one line like `│ show       Render an AIPerfJob CR with Jinja2/env-vars resolved              │`

- [ ] **Step 7: Commit**

```bash
git add src/aiperf/cli_commands/kube/show.py src/aiperf/cli_commands/kube/_app.py tests/unit/cli_commands/kube/test_show.py
git commit -s -m "feat(kube): scaffold \`aiperf kube show\` subcommand

Empty stub that raises NotImplementedError; later tasks add the
render pipeline. Registered under \`aiperf kube\`."
```

---

## Task 2: Render Jinja2 templates end-to-end (happy path)

**Goal:** The primary behaviour: `{{ concurrency_per_gpu * deployment_gpu_count }}` in the input becomes an integer in the output; `variables:` is stripped; non-benchmark `spec.*` keys and `metadata` pass through.

**Files:**
- Modify: `src/aiperf/cli_commands/kube/show.py` — replace the `NotImplementedError` with the real pipeline.
- Modify: `tests/unit/cli_commands/kube/test_show.py` — add rendering and pass-through tests.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/cli_commands/kube/test_show.py`:

```python
def _run_show(path: Path, capsys: pytest.CaptureFixture[str]) -> str:
    """Invoke the show command's default callable directly and return stdout."""
    from aiperf.cli_commands.kube.show import show as show_cmd

    show_cmd(path=path)
    return capsys.readouterr().out


def test_show_renders_jinja_templates(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`{{ a * b }}` inside phases must resolve to an int; variables section is stripped."""
    doc = _minimal_cr()
    doc["spec"]["benchmark"]["variables"] = {
        "concurrency_per_gpu": 2,
        "deployment_gpu_count": 16,
    }
    doc["spec"]["benchmark"]["phases"]["default"]["concurrency"] = (
        "{{ concurrency_per_gpu * deployment_gpu_count }}"
    )
    doc["spec"]["benchmark"]["phases"]["default"]["requests"] = (
        "{{ concurrency_per_gpu * deployment_gpu_count * 10 }}"
    )
    path = _write(tmp_path / "job.yaml", doc)

    out = _run_show(path, capsys)
    rendered = yaml.safe_load(out)

    phase = rendered["spec"]["benchmark"]["phases"]["default"]
    assert phase["concurrency"] == 32
    assert phase["requests"] == 320
    assert "variables" not in rendered["spec"]["benchmark"]
    assert "{{" not in out


def test_show_passes_through_non_benchmark_fields(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """metadata and non-benchmark spec.* keys must appear unchanged."""
    doc = _minimal_cr()
    doc["spec"]["image"] = "custom-image:v1"
    doc["spec"]["connectionsPerWorker"] = 200
    doc["spec"]["podTemplate"] = {
        "imagePullSecrets": ["mysecret"],
        "env": [{"name": "X", "value": "y"}],
    }
    path = _write(tmp_path / "job.yaml", doc)

    out = _run_show(path, capsys)
    rendered = yaml.safe_load(out)

    assert rendered["apiVersion"] == "aiperf.nvidia.com/v1alpha1"
    assert rendered["kind"] == "AIPerfJob"
    assert rendered["metadata"]["name"] == "test-job"
    assert rendered["spec"]["image"] == "custom-image:v1"
    assert rendered["spec"]["connectionsPerWorker"] == 200
    assert rendered["spec"]["podTemplate"]["imagePullSecrets"] == ["mysecret"]
    assert rendered["spec"]["podTemplate"]["env"] == [{"name": "X", "value": "y"}]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/cli_commands/kube/test_show.py::test_show_renders_jinja_templates tests/unit/cli_commands/kube/test_show.py::test_show_passes_through_non_benchmark_fields -n auto -v`
Expected: FAIL with `NotImplementedError: show command not yet implemented`.

- [ ] **Step 3: Implement the render pipeline**

Replace the body of `show()` in `src/aiperf/cli_commands/kube/show.py` (and add imports at module top):

```python
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kube show command: render AIPerfJob CR with Jinja2/env-vars resolved."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import yaml
from cyclopts import App, Parameter

app = App(name="show")


@app.default
def show(
    *,
    path: Annotated[
        Path,
        Parameter(
            name=["-p", "--path"],
            help="Path to an AIPerfJob YAML file.",
        ),
    ],
) -> None:
    """Render an AIPerfJob CR with Jinja2 and env-var templates resolved.

    Reads the CR, expands ``{{ ... }}`` expressions and ``${ENV_VAR}``
    substitutions inside ``spec.benchmark``, validates the result against
    ``AIPerfConfig``, re-wraps it in the original ``metadata`` and
    non-benchmark ``spec.*`` fields, and prints YAML to stdout.

    Examples:
        aiperf kube show --path recipes/qwen3-32b-fp8/trtllm/agg/perf.yaml
    """
    from aiperf.cli_utils import exit_on_error
    from aiperf.config import dump_config
    from aiperf.operator.spec_converter import extract_benchmark_config

    with exit_on_error(title="Error Rendering AIPerfJob"):
        doc = yaml.safe_load(path.read_text())

        if not isinstance(doc, dict):
            raise ValueError(f"{path}: document is not a YAML mapping")
        if doc.get("kind") != "AIPerfJob":
            raise ValueError(
                f"{path}: not an AIPerfJob manifest (kind={doc.get('kind')!r})"
            )
        spec = doc.get("spec")
        if not isinstance(spec, dict) or not isinstance(spec.get("benchmark"), dict):
            raise ValueError(
                f"{path}: spec.benchmark is required and must be a mapping"
            )

        # Render + validate the benchmark section. extract_benchmark_config
        # runs expand_config_dict (env vars + Jinja2) then AIPerfConfig
        # validation, and deliberately skips K8s runtime injection.
        config = extract_benchmark_config(spec)
        rendered_benchmark = yaml.safe_load(dump_config(config))

        doc["spec"]["benchmark"] = rendered_benchmark
        print(
            yaml.safe_dump(doc, sort_keys=False, default_flow_style=False),
            end="",
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/cli_commands/kube/test_show.py -n auto -v`
Expected: all 4 tests PASS (the 2 from Task 1 plus the 2 new ones).

- [ ] **Step 5: Smoke-test against a real recipe**

Run: `uv run aiperf kube show --path recipes/qwen3-235b-a22b-fp8/trtllm/agg/perf.yaml | yq '.spec.benchmark.phases.profiling.concurrency'`
Expected: `32`

- [ ] **Step 6: Commit**

```bash
git add src/aiperf/cli_commands/kube/show.py tests/unit/cli_commands/kube/test_show.py
git commit -s -m "feat(kube): render Jinja2/env-vars in \`aiperf kube show\`

Implements the happy path: reads an AIPerfJob CR, expands
{{ ... }} templates and \${ENV_VAR} substitutions in
spec.benchmark via extract_benchmark_config(), validates against
AIPerfConfig, and emits the full CR as YAML with non-benchmark
fields passed through unchanged."
```

---

## Task 3: Env var substitution resolves in benchmark fields

**Goal:** Prove `${ENV_VAR:default}` syntax is honoured (it's handled by `expand_config_dict`, but lock the behaviour in a test so future refactors can't silently break it).

**Files:**
- Modify: `tests/unit/cli_commands/kube/test_show.py` — add one test.

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/cli_commands/kube/test_show.py`:

```python
def test_show_resolves_env_var_default(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """${VAR:default} must resolve to `default` when VAR is unset."""
    monkeypatch.delenv("AIPERF_TEST_MODEL", raising=False)

    doc = _minimal_cr()
    doc["spec"]["benchmark"]["models"] = ["${AIPERF_TEST_MODEL:fallback-model}"]
    path = _write(tmp_path / "job.yaml", doc)

    out = _run_show(path, capsys)
    rendered = yaml.safe_load(out)

    # AIPerfConfig normalises string/list forms to {"items": [{"name": ...}]}.
    items = rendered["spec"]["benchmark"]["models"]["items"]
    assert items[0]["name"] == "fallback-model"
```

- [ ] **Step 2: Run the test**

Run: `uv run pytest tests/unit/cli_commands/kube/test_show.py::test_show_resolves_env_var_default -n auto -v`
Expected: PASS (no code change — `extract_benchmark_config` already does this).

If it FAILS: re-examine `expand_config_dict` behaviour; do not patch around it — the function is canonical.

- [ ] **Step 3: Commit**

```bash
git add tests/unit/cli_commands/kube/test_show.py
git commit -s -m "test(kube): lock env-var default resolution in \`show\`"
```

---

## Task 4: Error handling — missing file, wrong kind, missing benchmark

**Goal:** Every bad input exits non-zero with a clear message. Uses `cli_utils.exit_on_error`, which converts raised exceptions into `SystemExit(1)` and prints a rich error panel.

**Files:**
- Modify: `tests/unit/cli_commands/kube/test_show.py` — add three error-path tests.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/cli_commands/kube/test_show.py`:

```python
def test_show_missing_file_exits_nonzero(tmp_path: Path) -> None:
    from aiperf.cli_commands.kube.show import show as show_cmd

    with pytest.raises(SystemExit) as exc_info:
        show_cmd(path=tmp_path / "does-not-exist.yaml")
    assert exc_info.value.code != 0


def test_show_wrong_kind_exits_nonzero(tmp_path: Path) -> None:
    from aiperf.cli_commands.kube.show import show as show_cmd

    doc = _minimal_cr()
    doc["kind"] = "Pod"
    path = _write(tmp_path / "job.yaml", doc)

    with pytest.raises(SystemExit) as exc_info:
        show_cmd(path=path)
    assert exc_info.value.code != 0


def test_show_missing_benchmark_exits_nonzero(tmp_path: Path) -> None:
    from aiperf.cli_commands.kube.show import show as show_cmd

    doc = _minimal_cr()
    del doc["spec"]["benchmark"]
    path = _write(tmp_path / "job.yaml", doc)

    with pytest.raises(SystemExit) as exc_info:
        show_cmd(path=path)
    assert exc_info.value.code != 0
```

- [ ] **Step 2: Run the tests**

Run: `uv run pytest tests/unit/cli_commands/kube/test_show.py -k "exits_nonzero" -n auto -v`
Expected: all 3 PASS. `exit_on_error` converts the `ValueError`s raised by the guard clauses (Task 2 already added these) into `SystemExit(1)`. The missing-file case is an `OSError`/`FileNotFoundError` from `read_text()`; `exit_on_error` handles it the same way.

If any FAIL because `exit_on_error` does not raise `SystemExit` for `FileNotFoundError`, inspect `src/aiperf/cli_utils.py::exit_on_error` and confirm its exception net — adjust the guard in `show()` to wrap `path.exists()` or pre-check before `read_text()` if needed. Minimal preferred fix: add `if not path.is_file(): raise ValueError(f"{path}: file does not exist")` as the first line inside `exit_on_error`.

- [ ] **Step 3: Commit**

```bash
git add tests/unit/cli_commands/kube/test_show.py src/aiperf/cli_commands/kube/show.py
git commit -s -m "test(kube): cover error paths in \`aiperf kube show\`"
```
(If `show.py` was not changed, drop it from the `add`.)

---

## Task 5: Pydantic validation errors surface to the user

**Goal:** An input whose rendered benchmark fails `AIPerfConfig` validation exits non-zero and shows the pydantic error. This locks in the behaviour that the user gets the same clear error they would see from `aiperf kube validate`, not a silent bad YAML dump.

**Files:**
- Modify: `tests/unit/cli_commands/kube/test_show.py` — one more test.

- [ ] **Step 1: Write the failing test**

Append:

```python
def test_show_invalid_benchmark_exits_nonzero(tmp_path: Path) -> None:
    """First phase with seamless=True is a known AIPerfConfig invariant violation."""
    from aiperf.cli_commands.kube.show import show as show_cmd

    doc = _minimal_cr()
    # Seamless is only valid on non-first phases; AIPerfConfig rejects it here.
    doc["spec"]["benchmark"]["phases"]["default"]["seamless"] = True
    path = _write(tmp_path / "job.yaml", doc)

    with pytest.raises(SystemExit) as exc_info:
        show_cmd(path=path)
    assert exc_info.value.code != 0
```

- [ ] **Step 2: Run the test**

Run: `uv run pytest tests/unit/cli_commands/kube/test_show.py::test_show_invalid_benchmark_exits_nonzero -n auto -v`
Expected: PASS (no implementation change required; the pydantic error bubbles up through `extract_benchmark_config` into `exit_on_error`, which converts it to `SystemExit`).

- [ ] **Step 3: Commit**

```bash
git add tests/unit/cli_commands/kube/test_show.py
git commit -s -m "test(kube): surface pydantic errors in \`aiperf kube show\`"
```

---

## Task 6: Full unit suite + real-recipe smoke test

**Goal:** Run the whole unit suite once to confirm nothing regressed, then render every ported recipe to confirm they all round-trip.

**Files:** None modified.

- [ ] **Step 1: Run full unit suite**

Run: `uv run pytest tests/unit/ -n auto`
Expected: full suite passes. If anything outside `tests/unit/cli_commands/kube/test_show.py` fails, investigate — the new command should not affect other tests.

- [ ] **Step 2: Render every recipe and assert no unresolved templates**

Run the following shell one-liner (renders all 17 recipes, asserts no `{{` or `variables:` survives):

```bash
set -euo pipefail
for f in $(find recipes -name perf.yaml); do
  out=$(uv run aiperf kube show --path "$f")
  if echo "$out" | grep -qE '\{\{|^\s*variables:'; then
    echo "UNRENDERED: $f"
    exit 1
  fi
  echo "OK: $f"
done
echo "All recipes rendered cleanly."
```

Expected: 17 `OK:` lines followed by `All recipes rendered cleanly.` No `UNRENDERED:` lines.

- [ ] **Step 3: Spot-check math against Task 2's expected values**

Run:
```bash
uv run aiperf kube show --path recipes/qwen3-235b-a22b-fp8/trtllm/agg/perf.yaml \
  | yq '.spec.benchmark.phases.profiling.concurrency, .spec.benchmark.phases.profiling.requests'
```
Expected (two lines):
```
32
320
```

- [ ] **Step 4: Final commit if anything changed**

No code changes are expected here. If Step 1/2/3 exposed a regression, fix it inline, add a regression test to `test_show.py`, and commit. Otherwise this task has nothing to commit.

---

## Self-Review

**Spec coverage:**

| Spec section | Task |
|---|---|
| New command `aiperf kube show --path <file>` | Task 1 |
| Single required flag, YAML-only output | Task 1 + Task 2 |
| Renders `{{ ... }}` inside `spec.benchmark` | Task 2 |
| Strips `variables:` from output | Task 2 |
| Passes through `metadata`, non-benchmark `spec.*` | Task 2 |
| Env var substitution resolves | Task 3 |
| Missing file → exit 1 | Task 4 |
| Wrong kind → exit 1 | Task 4 |
| Missing `spec.benchmark` → exit 1 | Task 4 |
| Pydantic validation error → exit 1 | Task 5 |
| End-to-end render against real recipes | Task 6 |
| Validates via `AIPerfConfig` before dumping | Task 2 (implementation uses `extract_benchmark_config`) |
| No K8s runtime injection | Task 2 (ensured by using `extract_benchmark_config`, which per its docstring omits it) |
| YAML only (no JSON) | Task 1 (no `--format` flag) |

No gaps.

**Placeholder scan:** None. Every code block is concrete. The one fallback in Task 1 Step 5 ("if cyclopts App is not iterable, inspect and adjust") is a defensive note, not a placeholder — the intent is explicit and there's a concrete path forward.

**Type / name consistency:**

- `app` name is `show` everywhere.
- `path: Path` parameter is used identically in every test.
- `extract_benchmark_config(spec)` is called with the full `spec` dict (matching its signature in `src/aiperf/operator/spec_converter.py:315`).
- `dump_config(config)` signature matches `src/aiperf/config/loader/core.py:192` (keyword-only `exclude_defaults` and `exclude_none`, both default `True` — we accept the defaults).
- Test helpers `_minimal_cr`, `_write`, `_run_show` are defined in Task 1/Task 2 and reused unchanged in Tasks 3–5.
