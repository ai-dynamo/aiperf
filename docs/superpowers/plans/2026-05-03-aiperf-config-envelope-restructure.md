# AIPerf Config Envelope Restructure Implementation Plan (Plan A)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restructure `AIPerfConfig` from a flat container into a thin envelope around `BenchmarkConfig` so YAML configs separate sweep machinery (`sweep`, `multi_run`, `variables`, `random_seed`) from the swept body (everything else, nested under `benchmark:`). Hard-cut migration with a clear `ConfigurationError`.

**Architecture:** `AIPerfConfig` becomes `{benchmark: BenchmarkConfig, sweep, multi_run, variables, random_seed}`. `variables` and `random_seed` move from `BenchmarkConfig` to envelope. `expand_sweep` operates on the envelope dict; body merges land in `envelope["benchmark"]`; variables overlay separately. Loader rejects pre-restructure flat configs with a migration error. K8s `AIPerfJob.spec.benchmark` retypes to `BenchmarkConfig` (no in-CR sweep capability); `AIPerfSweepSpec` gains `variables`/`random_seed` at envelope.

**Tech Stack:** Python 3.10+, Pydantic v2, ruamel.yaml (already a project dep, used by the migration script to preserve comments), Jinja2, pytest.

**Spec:** `docs/superpowers/specs/2026-05-03-aiperf-config-envelope-restructure-design.md` (commit `c0dce9e5d`).

---

## Worktree setup

This plan is large enough to warrant its own worktree off `ajc/k8s`. Before Task 1, the executor (or controller for subagent-driven flow) creates one:

```bash
cd /home/anthony/nvidia/projects/aiperf/ajc/new-config-kube
git worktree add .worktrees/config-envelope -b ajc/config-envelope ajc/k8s
cd .worktrees/config-envelope
make first-time-setup
uv run pytest -n auto tests/unit/config/ 2>&1 | tail -5
```

Baseline test count for comparison: should be ~1042 passed in `tests/unit/config/` on a clean ajc/k8s checkout.

---

## Repo conventions used in this plan

- `uv run pytest -n auto` always; never bare `pytest`.
- Single subfolder per pytest invocation.
- Type hints on every function. `Field(description=...)` on every Pydantic field.
- No comments unless explaining a non-obvious "why".
- Pre-commit hooks run normally. If hooks reflow files mid-commit, re-pass full message via heredoc; never `--amend --no-edit`.
- Commit messages end with `Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>`.

---

## Task ordering rationale

Many of these tasks individually leave the test suite RED until later tasks land. That's expected — the restructure is a single cohesive landing. The critical dependency graph:

1. Migration script (Task 1) — needed before any bulk rewrite (Tasks 14, 15).
2. Model restructure (Tasks 2-3) — must precede loader/sweep updates (Tasks 4-9) because they all reference the new envelope shape.
3. Loader/sweep/Jinja/v1 converter (Tasks 4-12) — must precede bulk migration of fixtures (Tasks 14-17), since the fixtures need the new shape AND the loader needs to accept it.
4. K8s changes (Tasks 13) — independent of YAML loader; can land in parallel but plan does them sequential for review clarity.
5. Bulk fixture migration (Tasks 14-17) — runs the script + manual fixes for programmatic constructions.
6. Call-site rewrites (Task 18) — after fixtures migrate, services and exporters get their `.benchmark.` prefix.
7. New behavior tests (Tasks 19-22) — require everything above.
8. Docs (Tasks 23-28) — last; auto-regen of CLI/env-var docs needs the new model shape.
9. Final verification (Tasks 29-30).

Acceptance criterion for "branch is mergeable": after Task 30, `uv run pytest -n auto tests/unit/` and `tests/component_integration/` and `tests/integration/` are all green. Until Task 30, intermediate red is fine.

---

## Task 1: Build the migration script (TDD)

**Files:**
- Create: `tools/migrate_config_yaml.py`
- Create: `tests/unit/tools/test_migrate_config_yaml.py`
- Create: `tests/unit/tools/__init__.py` (if missing)

- [ ] **Step 1: Create the test file with the failing test for YAML rewrite**

Create `tests/unit/tools/__init__.py` (empty) if not present, then `tests/unit/tools/test_migrate_config_yaml.py`:

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for tools/migrate_config_yaml.py — the Plan A migration script."""

from __future__ import annotations

import textwrap

import pytest

from tools.migrate_config_yaml import (
    BODY_KEYS,
    ENVELOPE_KEYS,
    GRID_PATH_PREFIXES,
    is_already_migrated,
    migrate_yaml_text,
    rewrite_grid_sweep_paths,
    rewrite_scenario_runs,
)


class TestBodyEnvelopePartition:
    def test_constants_are_disjoint(self):
        assert BODY_KEYS.isdisjoint(ENVELOPE_KEYS)

    def test_body_keys_match_spec(self):
        assert BODY_KEYS == {
            "models", "endpoint", "datasets", "phases",
            "artifacts", "slos", "tokenizer", "gpu_telemetry",
            "server_metrics", "runtime", "logging", "metrics", "accuracy",
        }

    def test_envelope_keys_match_spec(self):
        assert ENVELOPE_KEYS == {"sweep", "multi_run", "variables", "random_seed", "benchmark"}


class TestMigrateYamlText:
    def test_flat_shape_rewraps_body_under_benchmark(self):
        flat = textwrap.dedent("""
            models: [llama]
            endpoint:
              urls: ["http://localhost:8000/v1/chat/completions"]
            phases:
              - {name: profiling, type: concurrency, requests: 10, concurrency: 1}
            random_seed: 42
        """).strip()
        out = migrate_yaml_text(flat)
        # body keys nested under benchmark:
        assert "benchmark:" in out
        assert "  models:" in out  # indented under benchmark
        assert "  endpoint:" in out
        assert "  phases:" in out
        # envelope keys stay top-level
        assert "\nrandom_seed: 42" in out

    def test_already_migrated_yaml_passes_through_unchanged(self):
        already = textwrap.dedent("""
            random_seed: 42
            benchmark:
              models: [llama]
              endpoint:
                urls: ["http://localhost:8000/v1/chat/completions"]
              phases:
                - {name: profiling, type: concurrency, requests: 10, concurrency: 1}
        """).strip()
        out = migrate_yaml_text(already)
        # idempotent
        assert migrate_yaml_text(out) == out

    def test_envelope_keys_stay_at_top_level(self):
        flat = textwrap.dedent("""
            models: [llama]
            sweep:
              type: grid
              variables:
                "phases.profiling.concurrency": [1, 2, 4]
            multi_run:
              num_runs: 3
            variables:
              isl: 128
            random_seed: 42
        """).strip()
        out = migrate_yaml_text(flat)
        # all four envelope keys at top
        for key in ("sweep:", "multi_run:", "variables:", "random_seed:"):
            assert f"\n{key}" in out or out.startswith(key)

    def test_grid_sweep_path_keys_get_benchmark_prefix(self):
        flat = textwrap.dedent("""
            models: [llama]
            phases:
              - {name: profiling, type: concurrency, requests: 10, concurrency: 1}
            sweep:
              type: grid
              variables:
                "phases.profiling.concurrency": [1, 2, 4]
                "datasets.0.entries": [100, 200]
        """).strip()
        out = migrate_yaml_text(flat)
        assert "benchmark.phases.profiling.concurrency" in out
        assert "benchmark.datasets.0.entries" in out

    def test_grid_variables_path_unchanged(self):
        flat = textwrap.dedent("""
            models: [llama]
            variables:
              isl: 128
            sweep:
              type: grid
              variables:
                "variables.isl": [128, 256]
        """).strip()
        out = migrate_yaml_text(flat)
        assert "variables.isl" in out
        # no benchmark.variables.isl
        assert "benchmark.variables.isl" not in out

    def test_scenario_runs_body_keys_get_benchmark_wrapper(self):
        flat = textwrap.dedent("""
            models: [llama]
            sweep:
              type: scenarios
              runs:
                - name: low
                  phases:
                    - {name: profiling, type: concurrency, concurrency: 1}
                - name: high
                  phases:
                    - {name: profiling, type: concurrency, concurrency: 10}
        """).strip()
        out = migrate_yaml_text(flat)
        # phases inside runs[i] should now be runs[i].benchmark.phases
        # naive substring check: there should be no top-level `phases:` directly inside a run dict
        # but there should be a `benchmark:` wrapper inside each run
        assert out.count("benchmark:") >= 1
        assert "phases:" in out

    def test_scenario_runs_keep_name_and_variables_at_top(self):
        flat = textwrap.dedent("""
            models: [llama]
            phases:
              - {name: profiling, type: concurrency, requests: 10, concurrency: 1}
            sweep:
              type: scenarios
              runs:
                - name: pair_0
                  variables: {isl: 128}
                  phases:
                    - {name: profiling, type: concurrency, concurrency: 5}
        """).strip()
        out = migrate_yaml_text(flat)
        # runs[0].name and runs[0].variables stay at top of the run dict
        assert "name: pair_0" in out
        assert "variables:" in out

    def test_empty_yaml_returns_empty(self):
        out = migrate_yaml_text("")
        assert out == "" or out.strip() == ""

    def test_no_body_keys_passes_through(self):
        # config-only-envelope fragment — nothing to migrate
        text = "random_seed: 42\nvariables:\n  isl: 128\n"
        out = migrate_yaml_text(text)
        # benchmark key not introduced when no body keys present
        assert "benchmark:" not in out


class TestIsAlreadyMigrated:
    def test_envelope_with_benchmark_key_is_migrated(self):
        text = "benchmark:\n  models: [llama]\nrandom_seed: 42\n"
        assert is_already_migrated(text) is True

    def test_flat_shape_is_not_migrated(self):
        text = "models: [llama]\nrandom_seed: 42\n"
        assert is_already_migrated(text) is False

    def test_envelope_only_no_body_is_migrated(self):
        text = "random_seed: 42\nvariables:\n  isl: 128\n"
        assert is_already_migrated(text) is True


class TestRewriteGridSweepPaths:
    def test_phases_path_gets_benchmark_prefix(self):
        sweep = {"type": "grid", "variables": {"phases.profiling.concurrency": [1, 2]}}
        rewrite_grid_sweep_paths(sweep)
        assert "benchmark.phases.profiling.concurrency" in sweep["variables"]
        assert "phases.profiling.concurrency" not in sweep["variables"]

    def test_variables_path_unchanged(self):
        sweep = {"type": "grid", "variables": {"variables.isl": [128, 256]}}
        rewrite_grid_sweep_paths(sweep)
        assert "variables.isl" in sweep["variables"]

    def test_already_prefixed_paths_unchanged(self):
        sweep = {"type": "grid", "variables": {"benchmark.phases.profiling.concurrency": [1, 2]}}
        rewrite_grid_sweep_paths(sweep)
        assert "benchmark.phases.profiling.concurrency" in sweep["variables"]
        # not double-prefixed
        assert "benchmark.benchmark.phases.profiling.concurrency" not in sweep["variables"]


class TestRewriteScenarioRuns:
    def test_run_with_phases_wraps_under_benchmark(self):
        run = {"name": "low", "phases": [{"name": "profiling", "concurrency": 1}]}
        rewrite_scenario_runs([run])
        assert "phases" not in run  # moved into benchmark
        assert "benchmark" in run
        assert run["benchmark"]["phases"] == [{"name": "profiling", "concurrency": 1}]

    def test_run_with_variables_keeps_at_top(self):
        run = {"variables": {"isl": 128}, "phases": [{"name": "profiling"}]}
        rewrite_scenario_runs([run])
        assert run["variables"] == {"isl": 128}
        assert run["benchmark"]["phases"] == [{"name": "profiling"}]

    def test_run_already_using_benchmark_wrapper_unchanged(self):
        run = {"name": "low", "benchmark": {"phases": [{"name": "profiling", "concurrency": 1}]}}
        rewrite_scenario_runs([run])
        assert run["benchmark"]["phases"] == [{"name": "profiling", "concurrency": 1}]
        assert "phases" not in run  # never was at top
```

- [ ] **Step 2: Run the test to verify it fails (script doesn't exist yet)**

Run: `uv run pytest -n auto tests/unit/tools/test_migrate_config_yaml.py -v 2>&1 | tail -20`
Expected: ImportError / ModuleNotFoundError on `from tools.migrate_config_yaml import ...`.

- [ ] **Step 3: Write `tools/migrate_config_yaml.py`**

Create `tools/migrate_config_yaml.py`:

```python
#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plan A migration script — re-indents pre-restructure flat YAML to envelope shape.

Hard-cut migration tool. Idempotent: running on already-migrated YAML is a no-op.

Usage:
    uv run python tools/migrate_config_yaml.py path/to/config.yaml --in-place
    uv run python tools/migrate_config_yaml.py path/to/config.yaml > new.yaml
    uv run python tools/migrate_config_yaml.py - < flat.yaml > envelope.yaml

Behavior:
- Body fields ({models, endpoint, datasets, phases, artifacts, slos, tokenizer,
  gpu_telemetry, server_metrics, runtime, logging, metrics, accuracy}) at the
  top level get re-indented under a `benchmark:` key.
- Envelope fields ({sweep, multi_run, variables, random_seed}) stay at top.
- `sweep.runs[i]` body keys get wrapped under `runs[i].benchmark`.
- `sweep.variables` keys (grid) gain `benchmark.` prefix unless they start
  with `benchmark.` or `variables.` already.
- Comments preserved via ruamel.yaml.
"""

from __future__ import annotations

import argparse
import sys
from io import StringIO
from pathlib import Path
from typing import Any

from ruamel.yaml import YAML

# Body fields that move under `benchmark:` in the new envelope shape.
BODY_KEYS = frozenset(
    {
        "models",
        "endpoint",
        "datasets",
        "phases",
        "artifacts",
        "slos",
        "tokenizer",
        "gpu_telemetry",
        "server_metrics",
        "runtime",
        "logging",
        "metrics",
        "accuracy",
    }
)

# Envelope fields that stay at top level. `benchmark` is the new wrapper key.
ENVELOPE_KEYS = frozenset({"sweep", "multi_run", "variables", "random_seed", "benchmark"})

# Allowed grid sweep variable path prefixes.
GRID_PATH_PREFIXES = ("benchmark.", "variables.")


def _yaml() -> YAML:
    """Configure ruamel.yaml for round-trip preservation of comments and quoting."""
    yml = YAML()
    yml.preserve_quotes = True
    yml.indent(mapping=2, sequence=4, offset=2)
    return yml


def is_already_migrated(yaml_text: str) -> bool:
    """Return True if the YAML already uses the envelope shape.

    Heuristic: no top-level body keys present (i.e., the partition is clean).
    A document with only envelope keys (or empty) qualifies. A document with
    `benchmark:` at top level and no top-level body keys also qualifies.
    """
    yml = _yaml()
    data = yml.load(StringIO(yaml_text))
    if data is None:
        return True
    if not isinstance(data, dict):
        return True
    return BODY_KEYS.isdisjoint(data.keys())


def migrate_yaml_text(yaml_text: str) -> str:
    """Migrate a YAML string from flat shape to envelope shape, idempotent."""
    yml = _yaml()
    data = yml.load(StringIO(yaml_text))
    if data is None:
        return yaml_text
    if not isinstance(data, dict):
        return yaml_text
    _migrate_in_place(data)
    out = StringIO()
    yml.dump(data, out)
    return out.getvalue()


def _migrate_in_place(data: dict[str, Any]) -> None:
    """Mutate ``data`` from flat to envelope shape."""
    body_present = {k: data[k] for k in list(data.keys()) if k in BODY_KEYS}
    if body_present:
        for k in body_present:
            del data[k]
        # Merge into existing benchmark key if user partially migrated.
        if "benchmark" in data and isinstance(data["benchmark"], dict):
            for k, v in body_present.items():
                data["benchmark"][k] = v
        else:
            data["benchmark"] = body_present

    sweep = data.get("sweep")
    if isinstance(sweep, dict):
        rewrite_grid_sweep_paths(sweep)
        runs = sweep.get("runs")
        if isinstance(runs, list):
            rewrite_scenario_runs(runs)


def rewrite_grid_sweep_paths(sweep: dict[str, Any]) -> None:
    """Prefix grid `sweep.variables` path keys with `benchmark.` when needed.

    Only fires for grid sweeps (or sweeps without a ``type``). Keys already
    starting with ``benchmark.`` or ``variables.`` are left unchanged.
    """
    if sweep.get("type", "grid") != "grid":
        return
    variables = sweep.get("variables")
    if not isinstance(variables, dict):
        return
    rewritten: dict[str, Any] = {}
    for key, value in variables.items():
        if isinstance(key, str) and not key.startswith(GRID_PATH_PREFIXES):
            rewritten[f"benchmark.{key}"] = value
        else:
            rewritten[key] = value
    variables.clear()
    variables.update(rewritten)


def rewrite_scenario_runs(runs: list[dict[str, Any]]) -> None:
    """Wrap body fields inside scenario runs under a ``benchmark:`` key per run.

    Allowed top-level keys inside a run: ``name``, ``variables``, ``benchmark``.
    Anything else gets moved under ``run["benchmark"]``.
    """
    for run in runs:
        if not isinstance(run, dict):
            continue
        body_present = {k: run[k] for k in list(run.keys()) if k in BODY_KEYS}
        if not body_present:
            continue
        for k in body_present:
            del run[k]
        if "benchmark" in run and isinstance(run["benchmark"], dict):
            for k, v in body_present.items():
                run["benchmark"][k] = v
        else:
            run["benchmark"] = body_present


def _migrate_file(path: Path, *, in_place: bool) -> None:
    """Migrate a YAML file. If in_place, overwrite; else write to stdout."""
    text = path.read_text(encoding="utf-8")
    new_text = migrate_yaml_text(text)
    if in_place:
        if new_text != text:
            path.write_text(new_text, encoding="utf-8")
    else:
        sys.stdout.write(new_text)


def _migrate_stdin() -> None:
    text = sys.stdin.read()
    sys.stdout.write(migrate_yaml_text(text))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Migrate AIPerf YAML configs from flat shape to envelope shape."
    )
    parser.add_argument(
        "path",
        help="Path to YAML file, or '-' for stdin.",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite the file in place. Ignored when path is '-'.",
    )
    args = parser.parse_args(argv)

    if args.path == "-":
        _migrate_stdin()
        return 0

    path = Path(args.path)
    if not path.exists() or not path.is_file():
        sys.stderr.write(f"error: not a file: {path}\n")
        return 2

    _migrate_file(path, in_place=args.in_place)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest -n auto tests/unit/tools/ -v 2>&1 | tail -25`
Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add tools/migrate_config_yaml.py tests/unit/tools/__init__.py tests/unit/tools/test_migrate_config_yaml.py
git commit -m "$(cat <<'EOF'
feat(tools): add migrate_config_yaml.py for Plan A envelope shape

One-shot migration script that re-indents pre-restructure flat YAML
to the envelope shape: body fields under `benchmark:`, envelope keys
(sweep, multi_run, variables, random_seed) at top. Handles grid
`sweep.variables` path keys (gain `benchmark.` prefix) and scenario
`runs[i]` body fields (wrapped under `runs[i].benchmark`). Idempotent.

Used twice: bulk-migrate test fixtures and tutorial examples in this
plan, and shipped to users as a one-time guidance tool.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Restructure `AIPerfConfig` and `BenchmarkConfig`

**Files:**
- Modify: `src/aiperf/config/config.py`

This task lands the model shape change. Tests will go RED until subsequent tasks (loader, fixtures) catch up. That's expected.

- [ ] **Step 1: Read the current shape to confirm line numbers**

Run: `grep -n "^class \|^    \(variables\|random_seed\|sweep\|multi_run\):" src/aiperf/config/config.py`
Expected: shows `class BenchmarkConfig`, fields `random_seed` and `variables` inside it; `class AIPerfConfig(BenchmarkConfig)` with `sweep` and `multi_run`.

- [ ] **Step 2: Move `random_seed` and `variables` from `BenchmarkConfig` to `AIPerfConfig`**

In `src/aiperf/config/config.py`:

1. Cut the `random_seed:` Annotated block (currently lines 324-332) and the `variables:` Annotated block (currently lines 334-345) from the `BenchmarkConfig` body.
2. Drop the `BenchmarkConfig` parent on `AIPerfConfig`. Replace the line `class AIPerfConfig(BenchmarkConfig):` with `class AIPerfConfig(BaseConfig):`.
3. Add a `benchmark: BenchmarkConfig = Field(...)` field at the top of `AIPerfConfig`.
4. Re-paste the `random_seed` and `variables` field definitions inside `AIPerfConfig`.
5. Update the `AIPerfConfig` class docstring to describe the envelope shape.

The new `AIPerfConfig` body should look like:

```python
class AIPerfConfig(BaseConfig):
    """AIPerf YAML envelope.

    Wraps a `BenchmarkConfig` (the swept body) with cross-variation fields
    (`sweep`, `multi_run`, `variables`, `random_seed`). This is the primary
    entry point for loading YAML configuration files. After sweep expansion,
    each variation's body materializes as a separate `BenchmarkConfig`.

    The split (envelope vs body) mirrors how AIPerfSweep CRDs are shaped on
    the K8s side: cross-variation machinery at envelope level, the swept
    workload as a body.
    """

    benchmark: Annotated[
        BenchmarkConfig,
        Field(description="Benchmark workload (the swept body)."),
    ]

    sweep: Annotated[
        SweepConfig | None,
        Field(
            default=None,
            description="Sweep configuration for parameter exploration. "
            "Supports grid (Cartesian product) and scenarios (hand-picked).",
        ),
    ]

    multi_run: Annotated[
        MultiRunConfig,
        Field(
            default_factory=MultiRunConfig,
            description="Multi-run benchmarking configuration for statistical reporting. "
            "Controls trials per variation, convergence, BO outer loop.",
        ),
    ]

    variables: Annotated[
        dict[str, Any],
        Field(
            default_factory=dict,
            description=(
                "User-defined values exposed to Jinja2 in `{{ ... }}` expressions "
                "during config load. Cross-variation: scenario `runs[i].variables:` "
                "deep-merge over this base. Preserved on the resolved config so "
                "run-time renderers can resolve them again."
            ),
        ),
    ]

    random_seed: Annotated[
        int | None,
        Field(
            default=None,
            description="Global random seed for reproducibility. Base seed for "
            "per-variation derivation in sweep mode (variation N gets base + N).",
        ),
    ]

    # ... existing model_validators (validate_sweep_no_dashboard_ui, etc.)
    # follow here, with body-field reads rewritten in Task 3.
```

- [ ] **Step 3: Update `BenchmarkConfig` docstring to remove the moved fields**

Inside `class BenchmarkConfig`, remove the line in the docstring that says `Global Settings: random_seed: ...`. Update the class-level note that today reads `Does NOT include sweep or multi_run settings (those live on AIPerfConfig).` to also list `variables` and `random_seed` as envelope-level.

- [ ] **Step 4: Run a smoke test that AIPerfConfig still imports**

Run: `uv run python -c "from aiperf.config import AIPerfConfig, BenchmarkConfig; print(AIPerfConfig.model_fields.keys()); print(BenchmarkConfig.model_fields.keys())"`
Expected:
- `AIPerfConfig` keys include `benchmark`, `sweep`, `multi_run`, `variables`, `random_seed`.
- `BenchmarkConfig` keys do NOT include `variables` or `random_seed`.

- [ ] **Step 5: Commit**

```bash
git add src/aiperf/config/config.py
git commit -m "$(cat <<'EOF'
refactor(config)!: restructure AIPerfConfig as envelope around BenchmarkConfig

Hard-cut shape change. AIPerfConfig drops BenchmarkConfig inheritance
and becomes a thin envelope:

    AIPerfConfig {
        benchmark: BenchmarkConfig
        sweep: SweepConfig | None
        multi_run: MultiRunConfig
        variables: dict[str, Any]
        random_seed: int | None
    }

`variables` and `random_seed` move from BenchmarkConfig to envelope
because they are cross-variation by nature (Jinja context overlay
target; per-variation seed derivation base).

Tests will be RED until subsequent tasks update the loader and
migrate fixtures. This is expected per Plan A's task ordering.

Spec: docs/superpowers/specs/2026-05-03-aiperf-config-envelope-restructure-design.md

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Rewrite `AIPerfConfig` sweep validators to read body via `self.benchmark.X`

**Files:**
- Modify: `src/aiperf/config/config.py`

After Task 2, the four `model_validator(mode="after")` validators on `AIPerfConfig` (currently around lines 503-609) reach for `self.runtime.ui` and `self.multi_run.parameter_sweep_*` directly. Body field reads now go through `self.benchmark`.

- [ ] **Step 1: Locate the four sweep validators**

Run: `grep -n "validate_sweep_no_dashboard_ui\|validate_sweep_same_seed_requires_seed\|validate_sweep_cooldown_nonneg\|validate_sweep_flags_require_sweep" src/aiperf/config/config.py`
Expected: 4 method definitions inside `class AIPerfConfig`.

- [ ] **Step 2: Rewrite the body-field reads**

For each of the four validators:

1. `validate_sweep_no_dashboard_ui` — replace `self.runtime.ui` with `self.benchmark.runtime.ui`.
2. `validate_sweep_same_seed_requires_seed` — replace `self.random_seed` is unchanged (envelope); replace any `self.runtime.X` with `self.benchmark.runtime.X` if present.
3. `validate_sweep_cooldown_nonneg` — `self.multi_run.X` is unchanged (envelope); rewrite any body reads.
4. `validate_sweep_flags_require_sweep` — `self.sweep` is unchanged (envelope); `self.multi_run.parameter_sweep_*` unchanged (envelope); rewrite any body reads.

Read each validator carefully and replace ALL body-field reads (anything in `models|endpoint|datasets|phases|artifacts|slos|tokenizer|gpu_telemetry|server_metrics|runtime|logging|metrics|accuracy`) with `self.benchmark.X`. Do NOT change envelope-level reads (`self.sweep`, `self.multi_run`, `self.variables`, `self.random_seed`).

- [ ] **Step 3: Run a smoke test**

Run: `uv run python -c "
from aiperf.config import AIPerfConfig, BenchmarkConfig
cfg = AIPerfConfig.model_validate({
    'benchmark': {
        'models': ['test/model'],
        'endpoint': {'type': 'chat', 'urls': ['http://localhost:8000/v1/chat/completions']},
        'datasets': [{'name': 'main', 'type': 'synthetic', 'entries': 100}],
        'phases': [{'name': 'profiling', 'type': 'concurrency', 'requests': 10, 'concurrency': 1}],
    }
})
print('AIPerfConfig validates with envelope shape')
"`
Expected: prints the success line.

- [ ] **Step 4: Commit**

```bash
git add src/aiperf/config/config.py
git commit -m "$(cat <<'EOF'
refactor(config): rewrite AIPerfConfig sweep validators for envelope shape

After the BenchmarkConfig split, validator reads of body fields go
through self.benchmark.X. Envelope-level reads (self.sweep,
self.multi_run, self.variables, self.random_seed) unchanged.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Add flat-shape detector + migration error in the loader

**Files:**
- Modify: `src/aiperf/config/loader/core.py`

When a user submits pre-restructure flat YAML, the loader fails fast with a clear migration message before any other validation.

- [ ] **Step 1: Add the detector + error in `load_config_from_string`**

In `src/aiperf/config/loader/core.py`, immediately after `_parse_yaml_mapping` (and before `substitute_env_vars`), add the flat-shape check.

Insert near the top of the file (alongside other constants):

```python
# Body fields that must live under `benchmark:` in the envelope shape.
_BODY_KEYS = frozenset(
    {
        "models",
        "endpoint",
        "datasets",
        "phases",
        "artifacts",
        "slos",
        "tokenizer",
        "gpu_telemetry",
        "server_metrics",
        "runtime",
        "logging",
        "metrics",
        "accuracy",
    }
)
```

Then update `load_config_from_string` to call a new helper before any further processing:

```python
def _reject_flat_shape(data: dict[str, Any], file_path: Path | str | None) -> None:
    """Raise ConfigurationError if the YAML uses pre-restructure flat shape.

    Triggered when any body field appears at the top level. `variables` and
    `random_seed` are envelope-level in the new shape and do NOT trigger.
    """
    flat = sorted(_BODY_KEYS & set(data.keys()))
    if not flat:
        return
    raise ConfigurationError(
        f"This config uses the pre-restructure flat shape (got top-level "
        f"keys: {flat}). Body fields must be nested under a top-level "
        f"`benchmark:` key, alongside envelope keys (`sweep`, `multi_run`, "
        f"`variables`, `random_seed`). To migrate:\n\n"
        f"  benchmark:\n"
        f"    models: [...]\n"
        f"    endpoint:\n"
        f"      urls: [...]\n"
        f"    phases: [...]\n"
        f"  # sweep / multi_run / variables stay at top level\n\n"
        f"Or run: uv run python tools/migrate_config_yaml.py "
        f"{file_path or '<path>'} --in-place\n"
        f"See docs/tutorials/migrating-config.md for examples.",
        file_path=file_path,
    )
```

In `load_config_from_string`, after `data = _parse_yaml_mapping(yaml_content, file_path)`, add:

```python
    _reject_flat_shape(data, file_path)
```

- [ ] **Step 2: Apply the same gate in `load_config_dict`**

`load_config_dict` (also in `core.py`) is the variant that returns the expanded dict without validating. It should reject flat shape too. Add the same `_reject_flat_shape(data, file_path)` call right after `_parse_yaml_mapping`.

- [ ] **Step 3: Apply the same gate in `expand_config_dict` (`loader/jinja.py`)**

`expand_config_dict` is the K8s-side helper that expands a dict directly (without parsing YAML). Since K8s CRD specs come in the new shape after Task 13's CRD regen, and pre-Plan-A specs are already rejected by AIPerfJobSpec validation (BenchmarkConfig.extra=forbid), this gate is mostly redundant for K8s. **Skip this step** — the K8s side has its own validation surface.

- [ ] **Step 4: Smoke test the rejection**

Run:
```bash
uv run python -c "
from aiperf.config.loader.core import load_config_from_string
from aiperf.config.loader.errors import ConfigurationError

flat = '''
models: [test/model]
endpoint:
  type: chat
  urls: [\"http://localhost:8000/v1/chat/completions\"]
phases:
  - {name: profiling, type: concurrency, requests: 10, concurrency: 1}
'''
try:
    load_config_from_string(flat, substitute_env=False)
    print('ERROR: should have raised')
except ConfigurationError as e:
    msg = str(e)
    assert 'flat shape' in msg, f'expected flat-shape message, got: {msg}'
    assert 'benchmark:' in msg
    assert 'migrate_config_yaml.py' in msg
    print('flat-shape rejection works')
"
```
Expected: `flat-shape rejection works`.

- [ ] **Step 5: Commit**

```bash
git add src/aiperf/config/loader/core.py
git commit -m "$(cat <<'EOF'
feat(loader): reject pre-restructure flat-shape YAML with migration error

Adds a fast-path detector at the top of load_config_from_string and
load_config_dict: any of the body fields ({models, endpoint, datasets,
phases, artifacts, slos, tokenizer, gpu_telemetry, server_metrics,
runtime, logging, metrics, accuracy}) at the top level raises a
ConfigurationError with migration guidance and a pointer to
tools/migrate_config_yaml.py.

`variables` and `random_seed` are excluded from the trigger because
they are envelope-level in the new shape.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Update Jinja `_flatten_into_context` to lift body keys to top level

**Files:**
- Modify: `src/aiperf/config/loader/jinja.py`

Templates today reference `{{ phases.profiling.rate }}` (no prefix). Under the envelope shape, the body lives at `benchmark.phases.profiling.rate`. To preserve user templates, the context-builder lifts body keys to the top level alongside their explicit `benchmark.X` paths.

- [ ] **Step 1: Inspect `build_template_context`**

Run: `sed -n '66,100p' src/aiperf/config/loader/jinja.py`
Expected: shows the function body with `_flatten_into_context(data, "", context)` and the variables-block resolution.

- [ ] **Step 2: Update `build_template_context` to alias benchmark body at top level**

Replace the `build_template_context` body in `src/aiperf/config/loader/jinja.py` with:

```python
def build_template_context(data: dict[str, Any]) -> dict[str, Any]:
    """Build context for Jinja2 template rendering.

    Creates a flattened context that allows:
    - Direct access: ``{{ concurrency }}`` (from ``variables`` block)
    - Top-level body alias: ``{{ phases.profiling.rate }}`` (lifted from
      ``benchmark.phases.profiling.rate``)
    - Explicit envelope path: ``{{ benchmark.phases.profiling.rate }}``,
      ``{{ variables.isl }}``

    The body-key alias preserves user templates from gaining a
    ``benchmark.`` prefix when migrating from the pre-restructure flat
    shape. Variables and benchmark live in different namespaces; the
    aliases never collide because envelope-level field names
    (``sweep``, ``multi_run``, ``variables``, ``random_seed``) don't
    appear inside ``benchmark``.
    """
    context: dict[str, Any] = {}
    _flatten_into_context(data, "", context)

    # Lift body keys to top level for backward-template-compatibility.
    benchmark = data.get("benchmark")
    if isinstance(benchmark, dict):
        _flatten_into_context(benchmark, "", context)

    if "variables" in data and isinstance(data["variables"], dict):
        rest_ctx = {
            k: v for k, v in context.items() if k.split(".", 1)[0] != "variables"
        }
        for key in data["variables"]:
            rest_ctx.pop(key, None)
        resolved = _resolve_variables_block(data["variables"], rest_ctx)
        for key, value in resolved.items():
            context[key] = value

    return context
```

The change: after the initial `_flatten_into_context(data, "", context)` (which produces both `benchmark.phases.X` paths and the dict at `benchmark`), call `_flatten_into_context(benchmark, "", context)` again to seed the top-level keys (`phases`, `endpoint`, etc.) directly. The existing `variables`-block resolution stays put.

- [ ] **Step 3: Smoke test**

Run:
```bash
uv run python -c "
from aiperf.config.loader.jinja import build_template_context, render_jinja2_templates
data = {
    'variables': {'isl': 128},
    'benchmark': {
        'phases': [{'name': 'profiling', 'rate': 100}],
        'endpoint': {'streaming': True},
    },
}
ctx = build_template_context(data)
# top-level alias
assert ctx['phases'] == {'profiling': {'name': 'profiling', 'rate': 100}}, f'got {ctx[\"phases\"]}'
# explicit benchmark path also present
assert ctx['benchmark.phases.profiling.rate'] == 100
# variables top-level alias
assert ctx['isl'] == 128
# explicit variables path
assert ctx['variables.isl'] == 128
print('Jinja context alias works')
"
```
Expected: `Jinja context alias works`.

- [ ] **Step 4: Commit**

```bash
git add src/aiperf/config/loader/jinja.py
git commit -m "$(cat <<'EOF'
feat(loader): alias benchmark body keys at top level in Jinja context

Under the envelope shape, body fields live at benchmark.X. To preserve
user templates that reference {{ phases.X }} / {{ endpoint.X }}
without a benchmark. prefix, build_template_context now does a second
_flatten_into_context pass against the benchmark subtree, lifting
its keys to the top level alongside explicit benchmark.X paths.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Update `expand_sweep` for envelope shape

**Files:**
- Modify: `src/aiperf/config/sweep.py`

`expand_sweep` operates on the envelope dict. Body merges land in `envelope["benchmark"]`. Variable overlays land in `envelope["variables"]` (already a top-level dict). The "strip multi_run / sweep" dance from the old deferred-Jinja path goes away because the variation dict is already an envelope; non-`benchmark`/non-`variables` envelope keys are left alone.

- [ ] **Step 1: Read the current `expand_sweep` and helpers**

Run: `sed -n '84,160p' src/aiperf/config/sweep.py`
Expected: `expand_sweep`, `_expand_grid_sweep`, `_expand_scenario_sweep`, `_expand_magic_lists` definitions.

- [ ] **Step 2: Update `_expand_grid_sweep`**

In `src/aiperf/config/sweep.py`, replace the existing `_expand_grid_sweep` with:

```python
def _expand_grid_sweep(
    base_data: dict[str, Any], variables: dict[str, list[Any]]
) -> list[tuple[dict[str, Any], SweepVariation]]:
    """Cartesian-product expansion. Path keys must be envelope-rooted.

    Allowed prefixes: ``benchmark.*``, ``variables.*``. Anything else is
    rejected; it would target a non-sweepable subtree (sweep, multi_run,
    random_seed) or a stale flat-shape path (`phases.X` instead of
    `benchmark.phases.X`).
    """
    for path in variables:
        if not isinstance(path, str) or not path.startswith(
            ("benchmark.", "variables.")
        ):
            raise ValueError(
                f"grid sweep variable {path!r} targets a non-sweepable "
                f"subtree; allowed prefixes: benchmark.*, variables.*. "
                f"If you migrated from the flat shape, prepend `benchmark.` "
                f"(e.g. `phases.profiling.rate` -> "
                f"`benchmark.phases.profiling.rate`)."
            )
    field_names = sorted(variables.keys())
    value_lists = [variables[f] for f in field_names]
    combinations = list(itertools.product(*value_lists))

    results = []
    for idx, combo in enumerate(combinations):
        variant = copy.deepcopy(base_data)
        values: dict[str, Any] = {}
        for field_path, value in zip(field_names, combo, strict=False):
            _set_nested_value(variant, field_path, value)
            values[field_path] = value
        variant = {k: v for k, v in variant.items() if k != "sweep"}
        label = ", ".join(f"{k}={v}" for k, v in values.items())
        results.append(
            (variant, SweepVariation(index=idx, label=label, values=values))
        )
    return results
```

- [ ] **Step 3: Update `_expand_scenario_sweep`**

Replace `_expand_scenario_sweep` with:

```python
_ALLOWED_SCENARIO_RUN_KEYS = {"name", "variables", "benchmark"}


def _expand_scenario_sweep(
    base_data: dict[str, Any], runs: list[dict[str, Any]]
) -> list[tuple[dict[str, Any], SweepVariation]]:
    """Expand scenario sweep. Each run is a partial envelope.

    Allowed run keys: ``name``, ``variables``, ``benchmark``. Anything
    else is rejected (it would land at envelope level and hide intent).
    The `benchmark:` subtree of each run deep-merges into
    ``envelope["benchmark"]``; the `variables:` subtree of each run
    deep-merges into ``envelope["variables"]``.
    """
    results = []
    for idx, scenario in enumerate(runs):
        unknown = set(scenario.keys()) - _ALLOWED_SCENARIO_RUN_KEYS
        if unknown:
            raise ValueError(
                f"sweep run [{idx}]: unknown field(s) {sorted(unknown)!r}; "
                f"allowed: name, variables, benchmark. (If you migrated "
                f"from the flat shape, wrap body fields under "
                f"`benchmark:` inside the run.)"
            )
        variant = copy.deepcopy(base_data)
        scenario_data = {k: v for k, v in scenario.items() if k != "name"}
        _normalize_scenario_dataset_form(scenario_data, variant, idx)
        _deep_merge(variant, scenario_data)
        variant = {k: v for k, v in variant.items() if k != "sweep"}
        label = scenario.get("name", f"scenario_{idx}")
        results.append(
            (variant, SweepVariation(index=idx, label=label, values=scenario_data))
        )
    return results
```

- [ ] **Step 4: Update `_normalize_scenario_dataset_form` to scope to `runs[i].benchmark.dataset`**

The helper from `2026-05-02-scenario-sweep-singular-dataset-design.md` rewrote `dataset:` (singular) to `datasets:` (plural) at run-top-level. Now the dataset shorthand lives inside `benchmark:`. Update the helper to look at `scenario["benchmark"]["dataset"]` rather than `scenario["dataset"]`:

```python
def _normalize_scenario_dataset_form(
    scenario: dict[str, Any], base: dict[str, Any], idx: int
) -> None:
    """Rewrite scenario `benchmark.dataset:` (singular) into
    `benchmark.datasets: [...]` so it deep-merges cleanly against the
    always-plural base.
    """
    from aiperf.config._benchmark_normalizers import DATASET_VS_DATASETS_MSG

    bench = scenario.get("benchmark")
    if not isinstance(bench, dict):
        return
    if "dataset" not in bench:
        return
    if "datasets" in bench:
        raise ValueError(f"sweep run [{idx}]: " + DATASET_VS_DATASETS_MSG)

    original = bench["dataset"]
    if not isinstance(original, dict):
        raise ValueError(
            f"sweep run [{idx}]: 'benchmark.dataset:' must be a mapping; "
            f"got {type(original).__name__}."
        )

    base_bench = base.get("benchmark", {})
    base_datasets = base_bench.get("datasets") or []
    explicit_name = original.get("name") if isinstance(original, dict) else None
    if explicit_name is not None:
        resolved_name = explicit_name
    elif len(base_datasets) == 1 and isinstance(base_datasets[0], dict):
        resolved_name = base_datasets[0].get("name")
        if resolved_name is None:
            raise ValueError(
                f"sweep run [{idx}]: base dataset has no 'name' to inherit; "
                f"add 'name:' to the scenario's dataset."
            )
    else:
        names = [d.get("name") for d in base_datasets if isinstance(d, dict)]
        raise ValueError(
            f"sweep run [{idx}]: scenario uses singular 'benchmark.dataset:' "
            f"against a base with multiple datasets ({names!r}); add 'name:' "
            f"to disambiguate."
        )

    bench.pop("dataset")
    bench["datasets"] = [
        {"name": resolved_name, **{k: v for k, v in original.items() if k != "name"}}
    ]
```

- [ ] **Step 5: Update `_set_nested_value` callers**

`_set_nested_value(variant, "benchmark.phases.profiling.concurrency", value)` should walk into `variant["benchmark"]["phases"]` then resolve `profiling` against the named-list. The existing helper handles dot-notation walks into dicts and named-dict lists, so no changes needed — the path string just gets one more segment.

Run: `grep -n "_set_nested_value\|_find_phase_or_recipe_alias" src/aiperf/config/sweep.py`
Expected: helpers exist; their dot-walk logic doesn't care about the leading segment count.

- [ ] **Step 6: Update `detect_sweep_fields` (magic-list detection) to walk via the benchmark subtree**

In `src/aiperf/config/sweep.py`, the `detect_sweep_fields` traversal walks the entire dict. For envelope-shape configs, the magic-list-eligible numeric fields live at `benchmark.phases.<name>.X`. The traversal already produces dotted paths from any depth, so it'll naturally yield `benchmark.phases.profiling.concurrency` for envelope-shape input. **No change needed.** Verify by reading the function.

Run: `sed -n '117,148p' src/aiperf/config/sweep.py`
Expected: `traverse(obj, current_path)` with dot-joined paths; no hard-coded prefix.

- [ ] **Step 7: Smoke test**

Run:
```bash
uv run python -c "
from aiperf.config.sweep import expand_sweep
data = {
    'benchmark': {
        'models': ['test/model'],
        'endpoint': {'type': 'chat', 'urls': ['http://localhost:8000/v1/chat/completions']},
        'datasets': [{'name': 'main', 'type': 'synthetic', 'entries': 100}],
        'phases': [{'name': 'profiling', 'type': 'concurrency', 'requests': 10, 'concurrency': 1}],
    },
    'sweep': {
        'type': 'grid',
        'variables': {'benchmark.phases.profiling.concurrency': [1, 2, 4]},
    },
}
expanded = expand_sweep(data)
print(f'expanded {len(expanded)} variations')
for variant, meta in expanded:
    print(f'  {meta.label}: concurrency={variant[\"benchmark\"][\"phases\"][0][\"concurrency\"]}')
"
```
Expected:
```
expanded 3 variations
  benchmark.phases.profiling.concurrency=1: concurrency=1
  benchmark.phases.profiling.concurrency=2: concurrency=2
  benchmark.phases.profiling.concurrency=4: concurrency=4
```

- [ ] **Step 8: Commit**

```bash
git add src/aiperf/config/sweep.py
git commit -m "$(cat <<'EOF'
feat(config)!: expand_sweep operates on envelope shape

Grid sweep paths must be envelope-rooted (benchmark.* or variables.*);
flat-shape paths rejected with migration guidance. Scenario runs use
the {name, variables, benchmark} partial envelope; unknown top-level
run keys rejected. The dataset-shorthand normalizer now scopes to
runs[i].benchmark.dataset.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Update `build_benchmark_plan` and `load_benchmark_plan` for envelope shape

**Files:**
- Modify: `src/aiperf/config/loader/plan.py`

After the model and sweep updates, the plan-construction simplifies. `config.benchmark` is the per-variation `BenchmarkConfig`; sweep expansion produces variation envelopes; per-variation `BenchmarkConfig.model_validate` runs on `variation_dict["benchmark"]`.

- [ ] **Step 1: Read the current `build_benchmark_plan` and `_expand_grid_variations`**

Run: `sed -n '20,135p' src/aiperf/config/loader/plan.py`
Expected: shows the helper already simplified for envelope thinking. The new shape lets us delete the model_dump dance.

- [ ] **Step 2: Replace `build_benchmark_plan` with the envelope version**

Replace the body of `build_benchmark_plan` and `_expand_grid_variations` in `src/aiperf/config/loader/plan.py`:

```python
def build_benchmark_plan(config: AIPerfConfig) -> BenchmarkPlan:
    """Build a BenchmarkPlan from a validated AIPerfConfig.

    Sweep + adaptive_search are mutually exclusive. When sweep is
    present, expands variations on the envelope dict and validates each
    variation's body as a BenchmarkConfig. When sweep is absent, the
    plan carries the single config.benchmark.
    """
    from aiperf.config.sweep import SweepVariation

    adaptive_search = config.multi_run.adaptive_search
    config_dict = config.model_dump(mode="json", exclude_none=True, exclude_unset=True)
    sweep_dict = config_dict.pop("sweep", None)

    if sweep_dict is not None and adaptive_search is not None:
        raise ValueError(
            "sweep block and --search-* flags are mutually exclusive: BO drives "
            "variation choice adaptively, while sweep enumerates them up-front. "
            "Drop the sweep block to use BO, or drop the --search-* flags."
        )

    if adaptive_search is not None:
        configs = [config.benchmark.model_copy(deep=True)]
        variations = [SweepVariation(index=0, label="base", values={})]
    elif sweep_dict is None:
        configs = [config.benchmark.model_copy(deep=True)]
        variations = [SweepVariation(index=0, label="base", values={})]
    else:
        configs, variations = _expand_envelope_variations(config_dict, sweep_dict)

    return _assemble_plan_from_aiperf_config(config, configs, variations)


def _expand_envelope_variations(
    config_dict: dict[str, Any],
    sweep_dict: dict[str, Any],
) -> tuple[list[BenchmarkConfig], list[SweepVariation]]:
    """Expand the sweep block into per-variation BenchmarkConfigs.

    Operates on the envelope dict: each variation has its own benchmark
    subtree (post-merge for scenarios, post-grid-write for grids).
    Re-renders Jinja per variation against the merged context, then
    validates the rendered benchmark subtree as a BenchmarkConfig.
    """
    from aiperf.config.sweep import SweepVariation, expand_sweep

    config_dict = dict(config_dict)
    config_dict["sweep"] = sweep_dict
    expanded = expand_sweep(config_dict)

    configs: list[BenchmarkConfig] = []
    variations: list[SweepVariation] = []
    for variation_dict, variation_meta in expanded:
        variation_dict.pop("sweep", None)
        variation_dict.pop("multi_run", None)
        context = build_template_context(variation_dict)
        variation_dict = render_jinja2_templates(variation_dict, context)
        bench_dict = variation_dict.get("benchmark", {})
        configs.append(BenchmarkConfig.model_validate(bench_dict))
        variations.append(variation_meta)
    if not variations:
        variations = [SweepVariation(index=0, label="base", values={})]
    return configs, variations
```

- [ ] **Step 3: Update `_assemble_plan_from_aiperf_config` (already exists from prior work; verify it survives)**

Run: `grep -n "_assemble_plan_from_aiperf_config" src/aiperf/config/loader/plan.py`
Expected: function exists. Its body uses `config.multi_run` typed access — that path is envelope-level and unchanged. No edits.

- [ ] **Step 4: Update `_apply_sweep_seed_derivation`**

The seed derivation reads `config.random_seed`, which is now envelope-level — same access pattern. Verify:

Run: `sed -n '138,155p' src/aiperf/config/loader/plan.py`
Expected: `base_seed = config.random_seed`. No change needed — `random_seed` is still on AIPerfConfig (now envelope).

- [ ] **Step 5: Smoke test the no-sweep path**

Run:
```bash
uv run python -c "
from aiperf.config import AIPerfConfig
from aiperf.config.loader.plan import build_benchmark_plan
cfg = AIPerfConfig.model_validate({
    'benchmark': {
        'models': ['test/model'],
        'endpoint': {'type': 'chat', 'urls': ['http://localhost:8000/v1/chat/completions']},
        'datasets': [{'name': 'main', 'type': 'synthetic', 'entries': 100}],
        'phases': [{'name': 'profiling', 'type': 'concurrency', 'requests': 10, 'concurrency': 1}],
    },
    'random_seed': 42,
})
plan = build_benchmark_plan(cfg)
print(f'is_sweep={plan.is_sweep} configs={len(plan.configs)} seed={plan.configs[0].random_seed if hasattr(plan.configs[0], \"random_seed\") else \"<not on bench>\"}')
"
```
Expected: `is_sweep=False configs=1 seed=<not on bench>` (random_seed is no longer on BenchmarkConfig; the seed lives on the envelope and the plan applies it to the runner via _apply_sweep_seed_derivation when sweep is active).

- [ ] **Step 6: Commit**

```bash
git add src/aiperf/config/loader/plan.py
git commit -m "$(cat <<'EOF'
feat(loader): build_benchmark_plan operates on envelope shape

After AIPerfConfig restructure:
- non-sweep / BO path: configs = [config.benchmark.model_copy()].
- sweep path: expand_sweep on envelope dict; per variation, render Jinja
  on the merged variation envelope, validate variation_dict["benchmark"]
  as BenchmarkConfig.

The "strip multi_run/sweep from variation_dict" dance survives in the
sweep loop (variations carry an envelope shape that includes
multi_run; we drop it before pulling the benchmark body for validation).
Plan B further simplifies once verified.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: Apply seed derivation to per-variation benchmark configs (envelope-level base seed)

**Files:**
- Modify: `src/aiperf/config/loader/plan.py`

`random_seed` moved to envelope. `_apply_sweep_seed_derivation` today does `cfg.random_seed = base_seed + variation_idx` on each `BenchmarkConfig`. After the move, `BenchmarkConfig` no longer has a `random_seed` field — the seed needs to live somewhere per-variation.

**Decision:** The plan derives per-variation seeds and stores them on a new field `BenchmarkPlan.variation_seeds: list[int | None]`, parallel to `configs` and `variations`. Runtime callers that need a per-variation seed read from there.

- [ ] **Step 1: Add `variation_seeds` to `BenchmarkPlan`**

In `src/aiperf/config/benchmark.py`, find `class BenchmarkPlan` and add (next to `configs` and `variations`):

```python
    variation_seeds: list[int | None] = Field(
        default_factory=list,
        description="Per-variation random seed (None when no base seed set). "
        "Length matches `configs`/`variations`. Variation 0 inherits the "
        "envelope `random_seed`; variation N gets `random_seed + N` unless "
        "`parameter_sweep_same_seed` is True.",
    )
```

- [ ] **Step 2: Replace `_apply_sweep_seed_derivation` to populate the new field**

In `src/aiperf/config/loader/plan.py`:

```python
def _apply_sweep_seed_derivation(plan: BenchmarkPlan, config: AIPerfConfig) -> None:
    """Populate plan.variation_seeds from the envelope random_seed.

    Variation 0 carries the base seed; variation N gets ``base + N``
    unless ``parameter_sweep_same_seed`` is True (in which case all
    variations share the base seed). When ``random_seed`` is None on
    the envelope, all entries are None.
    """
    base_seed = config.random_seed
    plan.variation_seeds = []
    for variation_idx in range(len(plan.configs)):
        if base_seed is None:
            plan.variation_seeds.append(None)
        elif plan.parameter_sweep_same_seed or not plan.is_sweep:
            plan.variation_seeds.append(base_seed)
        else:
            plan.variation_seeds.append(base_seed + variation_idx)
```

- [ ] **Step 3: Update callers that previously read `cfg.random_seed`**

Run: `grep -rn "\.random_seed" src/aiperf --include="*.py" | grep -v "\.envelope\|\.aiperf_config\|\.config\.random_seed"`
Inspect the output. Where a caller has `cfg: BenchmarkConfig` and reads `cfg.random_seed`, it now needs the plan's `variation_seeds[idx]` instead.

For each such caller:
1. If the caller has access to `BenchmarkPlan` and the variation index: `seed = plan.variation_seeds[idx]`.
2. If the caller has only a `BenchmarkConfig`: pass the seed in explicitly. Most likely call sites are in `MultiRunOrchestrator` and `cli_runner._build_*` paths.

Concrete updates:
- `src/aiperf/orchestrator/multi_run.py` (or wherever the per-variation runner lives): replace `cfg.random_seed` with `plan.variation_seeds[idx]`.
- Any export/aggregation code that records the seed: same pattern.

This step is mechanical but requires reading the codebase. Run the grep, list the matches, and update each. Commit grouped by call-site cluster.

- [ ] **Step 4: Smoke test**

Run:
```bash
uv run python -c "
from aiperf.config import AIPerfConfig
from aiperf.config.loader.plan import build_benchmark_plan
cfg = AIPerfConfig.model_validate({
    'benchmark': {
        'models': ['test/model'],
        'endpoint': {'type': 'chat', 'urls': ['http://localhost:8000/v1/chat/completions']},
        'datasets': [{'name': 'main', 'type': 'synthetic', 'entries': 100}],
        'phases': [{'name': 'profiling', 'type': 'concurrency', 'requests': 10, 'concurrency': 1}],
    },
    'random_seed': 100,
    'sweep': {'type': 'grid', 'variables': {'benchmark.phases.profiling.concurrency': [1, 2, 4]}},
})
plan = build_benchmark_plan(cfg)
print(f'variation_seeds={plan.variation_seeds}')
assert plan.variation_seeds == [100, 101, 102]
print('seed derivation works')
"
```
Expected: `variation_seeds=[100, 101, 102]\nseed derivation works`.

- [ ] **Step 5: Commit**

```bash
git add src/aiperf/config/benchmark.py src/aiperf/config/loader/plan.py src/aiperf/orchestrator/ src/aiperf/cli_runner.py
git commit -m "$(cat <<'EOF'
feat(plan): per-variation seeds on BenchmarkPlan.variation_seeds

random_seed moved from BenchmarkConfig to AIPerfConfig envelope.
Per-variation derivation now populates plan.variation_seeds (parallel
to configs/variations). Runners read from there instead of
cfg.random_seed.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: Update v1 → v2 converter to emit envelope shape

**Files:**
- Modify: `src/aiperf/config/v1/converter.py`
- Modify: `src/aiperf/config/v1/_converter_*.py` if needed

The CLI's `aiperf profile` flow goes through `UserConfig.to_aiperf_config()` (or equivalent) which builds a v2 dict and passes it to `AIPerfConfig.model_validate`. After the restructure, the dict must have body keys nested under `benchmark:`.

- [ ] **Step 1: Locate the assembly point**

Run: `grep -n "AIPerfConfig.model_validate\|to_aiperf_config\|build_aiperf_config" src/aiperf/config/v1/*.py | head -20`
Expected: the call site(s) where `UserConfig` becomes a v2 dict.

- [ ] **Step 2: Wrap body keys under `benchmark:` at the assembly point**

In `src/aiperf/config/v1/converter.py`, immediately before the final `AIPerfConfig.model_validate(...)` (or wherever the v2 dict is finalized), partition the dict:

```python
# Add at the top of the relevant function (or as a helper near the top of the file)
_ENVELOPE_KEYS = {"sweep", "multi_run", "variables", "random_seed", "benchmark"}


def _wrap_under_envelope(v2_dict: dict[str, Any]) -> dict[str, Any]:
    """Partition a flat-shaped v2 dict into envelope shape.

    Envelope keys ({sweep, multi_run, variables, random_seed, benchmark})
    stay at top level. Everything else is moved under `benchmark:`.
    Idempotent: a dict that is already envelope-shaped passes through.
    """
    body = {k: v2_dict[k] for k in list(v2_dict.keys()) if k not in _ENVELOPE_KEYS}
    if not body:
        return v2_dict
    for k in body:
        del v2_dict[k]
    if "benchmark" in v2_dict and isinstance(v2_dict["benchmark"], dict):
        v2_dict["benchmark"].update(body)
    else:
        v2_dict["benchmark"] = body
    return v2_dict
```

Then call `_wrap_under_envelope(nested)` (or whatever the v2 dict variable is named) right before `AIPerfConfig.model_validate(nested)`.

- [ ] **Step 3: Update `_promote_magic_lists_to_sweep_block` to emit envelope-rooted paths**

Locate `_promote_magic_lists_to_sweep_block` in `src/aiperf/config/v1/converter.py`. The path emission line currently looks like:

```python
sweep_variables[f"phases.{phase_name}.{key}"] = phase.pop(key)
```

Change it to:

```python
sweep_variables[f"benchmark.phases.{phase_name}.{key}"] = phase.pop(key)
```

- [ ] **Step 4: Update `_apply_recipe_sweep_variables` similarly**

Locate `_apply_recipe_sweep_variables` in the same file. If it constructs path keys, prefix them with `benchmark.`. If it copies pre-existing sweep_variables verbatim, no change needed (they should already be envelope-rooted from the spec).

Read the function carefully. Update only the path-construction sites.

- [ ] **Step 5: Smoke test the CLI flow**

Run:
```bash
uv run aiperf profile --help 2>&1 | head -20
```
Expected: cyclopts prints help without errors. (This exercises the converter once at import time.)

Then a programmatic round-trip:

```bash
uv run python -c "
from aiperf.config.v1 import UserConfig
# minimal UserConfig fixture
uc = UserConfig.model_validate({
    'endpoint': {'type': 'chat', 'urls': ['http://localhost:8000/v1/chat/completions']},
    'input': {'synthetic_tokens': {'mean': 128}},
    'loadgen': {'concurrency': 1, 'request_count': 10},
})
v2 = uc.to_aiperf_config()  # or whatever the conversion function is called
print('top-level keys:', sorted(v2.model_fields.keys()))
print('benchmark keys:', sorted(v2.benchmark.model_fields.keys()))
" 2>&1 | head -10
```
Expected: top-level keys include `benchmark`, `sweep`, `multi_run`, `variables`, `random_seed`. Benchmark keys include `models`, `endpoint`, `datasets`, `phases`, etc.

(If `UserConfig.to_aiperf_config` doesn't exist by that name, find the correct entry point via the grep in Step 1.)

- [ ] **Step 6: Commit**

```bash
git add src/aiperf/config/v1/converter.py src/aiperf/config/v1/_converter_*.py
git commit -m "$(cat <<'EOF'
feat(config/v1): emit envelope shape from CLI converter

Final assembly wraps body keys under `benchmark:`. Magic-list flag
promotion (_promote_magic_lists_to_sweep_block) emits
benchmark.phases.<name>.<field> paths. Recipe sweep-variable
promotion (_apply_recipe_sweep_variables) similarly prefixes.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: Run the migration script over `tests/` test fixtures

**Files:**
- Modify: many `tests/**/*.py` (triple-quoted YAML literals)
- Modify: any `tests/**/*.yaml` files

This is a bulk mechanical task. The migration script handles YAML files directly; YAML literals embedded in Python files need a wrapper script.

- [ ] **Step 1: Create a one-off wrapper for triple-quoted YAML literals in Python files**

Create `tools/migrate_test_yaml_literals.py` (this script gets deleted after Task 17 — it's plan-internal):

```python
#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One-off helper: walk *.py files, find triple-quoted YAML literals,
run them through tools.migrate_config_yaml.migrate_yaml_text, splice back.

Heuristic for detecting a YAML literal: any triple-quoted string assignment
where the contents contain a top-level body key (`models:`, `endpoint:`,
etc.) at column 0 of the dedented contents. Conservative — false negatives
preferred over false positives.

Run only after Plan A's models/loader/sweep tasks land, before bulk
fixture migration. Discarded after Task 17.
"""

from __future__ import annotations

import re
import sys
import textwrap
from pathlib import Path

from tools.migrate_config_yaml import BODY_KEYS, migrate_yaml_text


_TRIPLE_QUOTE_RE = re.compile(
    r'(?P<indent>[ \t]*)(?P<var>\w+)\s*=\s*(?P<quote>"""|\'\'\')(?P<body>.*?)(?P=quote)',
    re.DOTALL,
)


def _looks_like_yaml(body: str) -> bool:
    """Cheap detector: does the dedented body have a top-level body key?"""
    dedented = textwrap.dedent(body).lstrip("\n")
    for key in BODY_KEYS:
        if re.search(rf"(?m)^{key}\s*:", dedented):
            return True
    return False


def _migrate_literal(match: re.Match[str]) -> str:
    body = match.group("body")
    if not _looks_like_yaml(body):
        return match.group(0)
    dedented = textwrap.dedent(body).lstrip("\n")
    migrated = migrate_yaml_text(dedented)
    indent = match.group("indent") + "    "
    indented = textwrap.indent(migrated.rstrip("\n"), indent)
    return (
        f'{match.group("indent")}{match.group("var")} = '
        f'{match.group("quote")}\n{indented}\n'
        f'{match.group("indent")}{match.group("quote")}'
    )


def migrate_python_file(path: Path) -> bool:
    """Returns True if the file was modified."""
    original = path.read_text(encoding="utf-8")
    new_text = _TRIPLE_QUOTE_RE.sub(_migrate_literal, original)
    if new_text != original:
        path.write_text(new_text, encoding="utf-8")
        return True
    return False


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    paths = [Path(a) for a in args]
    changed = 0
    for path in paths:
        if path.is_file() and path.suffix == ".py":
            if migrate_python_file(path):
                changed += 1
                print(f"migrated: {path}")
        elif path.is_dir():
            for py in path.rglob("*.py"):
                if migrate_python_file(py):
                    changed += 1
                    print(f"migrated: {py}")
    print(f"{changed} files changed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Migrate Python test files**

Run:
```bash
uv run python tools/migrate_test_yaml_literals.py tests/
```
Expected: a list of changed test files.

- [ ] **Step 3: Migrate standalone YAML files in tests/**

Run:
```bash
find tests -name "*.yaml" | while read -r f; do
    uv run python tools/migrate_config_yaml.py "$f" --in-place
done
```
Expected: silent (in-place writes; no stdout).

- [ ] **Step 4: Run the test_migrate test suite to confirm the migration script itself still passes**

Run: `uv run pytest -n auto tests/unit/tools/test_migrate_config_yaml.py -v 2>&1 | tail -10`
Expected: all pass.

- [ ] **Step 5: Run the unit-config test suite**

Run: `uv run pytest -n auto tests/unit/config/ 2>&1 | tail -15`
Expected: most tests pass. Some may still fail because they rely on programmatic `AIPerfConfig(...)` construction (Task 11) or call sites reading `config.X` for body fields (Task 12).

Record the pass/fail count. Tasks 11-12 fix the remaining failures.

- [ ] **Step 6: Commit**

```bash
git add tests/
git commit -m "$(cat <<'EOF'
chore(tests): migrate YAML literals to envelope shape

Bulk migration via tools/migrate_config_yaml.py + a per-Python-file
wrapper that reaches into triple-quoted YAML literals. Programmatic
AIPerfConfig(...) constructions and call-site reads of body fields
update in subsequent tasks.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 11: Update programmatic `AIPerfConfig(...)` and `BenchmarkConfig(...)` constructions in tests

**Files:**
- Modify: various `tests/**/*.py`

Where tests build configs programmatically (not from YAML), the constructor shape changed. Old: `AIPerfConfig(models=[...], endpoint=..., phases=...)`. New: `AIPerfConfig(benchmark=BenchmarkConfig(models=[...], endpoint=..., phases=...))`.

- [ ] **Step 1: Find programmatic constructions**

Run: `grep -rn "AIPerfConfig(" tests --include="*.py" | grep -v "AIPerfConfig(\s*$" | grep -v "model_validate"`
Expected: list of test sites that construct AIPerfConfig directly.

Also: `grep -rn "BenchmarkConfig(" tests --include="*.py" | grep -v "model_validate"`

- [ ] **Step 2: Update each construction**

For each match, decide:
- If the construction passes only envelope-level fields (`sweep=...`, `multi_run=...`): no change needed — those still pass through to AIPerfConfig.
- If the construction passes body fields (`models=...`, `endpoint=...`, etc.): wrap them in `benchmark=BenchmarkConfig(...)`.

Example before:
```python
cfg = AIPerfConfig(
    models=["test/model"],
    endpoint=EndpointConfig(...),
    datasets=[Dataset(...)],
    phases=[Phase(...)],
    random_seed=42,
)
```

Example after:
```python
cfg = AIPerfConfig(
    benchmark=BenchmarkConfig(
        models=["test/model"],
        endpoint=EndpointConfig(...),
        datasets=[Dataset(...)],
        phases=[Phase(...)],
    ),
    random_seed=42,
)
```

For each test file, also update assertions that reach for `cfg.models` / `cfg.endpoint` / etc. — they become `cfg.benchmark.models`. (Task 12 covers `src/aiperf` call-site rewrites; this task is the test-side analog.)

- [ ] **Step 3: Run the unit-config test suite**

Run: `uv run pytest -n auto tests/unit/config/ 2>&1 | tail -15`
Expected: pass count higher than after Task 10 step 5. Remaining failures should be either: (a) tests that read body fields from `BenchmarkConfig` instances and use `.random_seed`/`.variables` (now removed), or (b) tests in non-config subdirectories that haven't been touched yet.

- [ ] **Step 4: Run the full unit suite to surface non-config failures**

Run: `uv run pytest -n auto tests/unit/ 2>&1 | tail -10`
Expected: many failures still. Most should be call-site reads of `cfg.X` for body fields, which Task 12 fixes.

- [ ] **Step 5: Commit**

```bash
git add tests/
git commit -m "$(cat <<'EOF'
chore(tests): update programmatic AIPerfConfig/BenchmarkConfig constructions

Constructions that pass body fields now wrap them under
benchmark=BenchmarkConfig(...). Assertions reading body fields
update to .benchmark.X.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 12: Rewrite `src/aiperf/` call sites to use `.benchmark.X` for body field reads

**Files:**
- Modify: ~125 sites across `src/aiperf/**/*.py`

This is the bulk mechanical task on the production side. Every read of `config.{models,endpoint,datasets,phases,artifacts,slos,tokenizer,gpu_telemetry,server_metrics,runtime,logging,metrics,accuracy}` becomes `config.benchmark.{...}` when `config` is an `AIPerfConfig`. Where a function only has `BenchmarkConfig` (per-variation), the calls already work — no change.

- [ ] **Step 1: Generate a list of candidate call sites**

Run:
```bash
grep -rnE "(\b[a-z_]*config\b|\bcfg\b)\.(models|endpoint|datasets|phases|artifacts|slos|tokenizer|gpu_telemetry|server_metrics|runtime|logging|metrics|accuracy)\b" src/aiperf --include="*.py" > /tmp/callsites.txt
wc -l /tmp/callsites.txt
```
Expected: ~125 lines.

- [ ] **Step 2: Triage by variable type**

For each line in `/tmp/callsites.txt`, determine the type of the variable on the left of the dot:
- If `config: AIPerfConfig`, `aiperf_config: AIPerfConfig`, `cfg: AIPerfConfig`, etc. → needs `.benchmark.` prefix.
- If `config: BenchmarkConfig`, `cfg: BenchmarkConfig`, `bench: BenchmarkConfig` → unchanged.
- If `cfg.runtime` and `cfg` is the AIPerf `runtime` settings (different `RuntimeConfig` from a different module) — not the case here, but verify via the import.

Easiest heuristic: scan each file and find the function signature / variable annotation. Where ambiguous, look at the call site upstream to see what's passed in.

- [ ] **Step 3: Apply rewrites in batches by file**

Walk through each file and apply the rewrites. For long functions that read many body fields, introduce a local alias:

```python
def some_method(self) -> None:
    bench = self.aiperf_config.benchmark
    if bench.endpoint.streaming:
        ...
    for phase in bench.phases:
        ...
```

Where the variable annotation should be tightened (e.g., a function takes `cfg: AIPerfConfig` but should arguably take `cfg: BenchmarkConfig`), update the signature too — but ONLY if the function genuinely doesn't need envelope-level access. Don't introduce signature drift just to avoid the prefix.

Commit per-file or per-subsystem; keep commits granular for review.

- [ ] **Step 4: Run the full unit suite**

Run: `uv run pytest -n auto tests/unit/ 2>&1 | tail -10`
Expected: green or near-green. Remaining failures should be in test files that still reference body-field call sites OR genuinely new behavior tests Task 19+ will add.

- [ ] **Step 5: Run the component-integration suite**

Run: `uv run pytest -n auto tests/component_integration/ 2>&1 | tail -10`
Expected: green or near-green.

- [ ] **Step 6: Commit (or accumulate via per-subsystem commits during step 3)**

```bash
git add src/aiperf/
git commit -m "$(cat <<'EOF'
refactor!: route body-field reads through .benchmark.X

After AIPerfConfig restructure, callers holding an AIPerfConfig now
read body fields via .benchmark. Local-alias pattern (`bench =
config.benchmark`) used in functions with multiple reads to keep
call sites concise.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 13: Update K8s models — retype `AIPerfJobSpec.benchmark`, add envelope fields to `AIPerfSweepSpec`, regen CRDs

**Files:**
- Modify: `src/aiperf/operator/models.py`
- Modify: `src/aiperf/kubernetes/sweep_models.py`
- Regen: `deploy/helm/aiperf-operator/templates/crd.yaml`, `crd-aiperfsweep.yaml`
- Modify: `tools/generate_crd.py` (if CEL rules need adjustment)

- [ ] **Step 1: Retype `AIPerfJobSpec.benchmark` to `BenchmarkConfig`**

In `src/aiperf/operator/models.py`, change the existing line:

```python
benchmark: AIPerfConfig = Field(
    ..., description="Benchmark configuration (AIPerfConfig)."
)
```

to:

```python
benchmark: BenchmarkConfig = Field(
    ..., description="Benchmark workload (BenchmarkConfig). AIPerfJob carries a "
    "single benchmark — no in-CR sweep capability. For sweeps, use AIPerfSweep."
)
```

Update the import: `from aiperf.config import AIPerfConfig` → `from aiperf.config import BenchmarkConfig` (or keep both if needed elsewhere).

- [ ] **Step 2: Add `variables` and `random_seed` to `AIPerfSweepSpec`**

In `src/aiperf/kubernetes/sweep_models.py`, inside `class AIPerfSweepSpec`:

```python
    variables: dict[str, Any] = Field(
        default_factory=dict,
        description="User-defined values exposed to Jinja2 in `{{ ... }}` "
        "expressions during config load. Per-variation scenarios overlay "
        "via `sweep.runs[i].variables:`.",
    )

    random_seed: int | None = Field(
        default=None,
        description="Base random seed for per-variation derivation. "
        "Variation N gets `random_seed + N` unless multi_run.auto_set_seed "
        "is False.",
    )
```

- [ ] **Step 3: Remove the "no sweep inside template" Pydantic validator**

In `src/aiperf/kubernetes/sweep_models.py`, locate the `_validate_axis_combination` validator (around line 238). The Rule 4 / Rule 5 that reject `template.spec.benchmark.sweep` and `template.spec.benchmark.multi_run` are no longer needed because `BenchmarkConfig` doesn't have those fields (the type system enforces it). Remove those rules from the validator. Keep any other rules in the same validator function intact.

- [ ] **Step 4: Remove the corresponding CEL rule from the CRD generator**

In `tools/generate_crd.py`, locate the Tier 1D CEL block emitting `template.spec.benchmark.sweep is forbidden` and `template.spec.benchmark.multiRun is ...` (around line 1313-1340). Remove those CEL rules. They're redundant with the type system after the retyping.

- [ ] **Step 5: Regenerate CRDs**

Run:
```bash
uv run python tools/generate_crd.py
```
Expected: `deploy/helm/aiperf-operator/templates/crd.yaml` and `crd-aiperfsweep.yaml` regenerate. Inspect the diff:

```bash
git diff deploy/helm/aiperf-operator/templates/
```

Expected changes:
- `crd.yaml` (AIPerfJob): `spec.benchmark` shape no longer includes `sweep`/`multi_run` properties; the schema is the BenchmarkConfig shape.
- `crd-aiperfsweep.yaml` (AIPerfSweep): `spec.variables` and `spec.randomSeed` (camelCase per Pydantic alias) appear; the "no sweep inside template" CEL rules are gone.

- [ ] **Step 6: Run helm-lint**

Run: `helm lint deploy/helm/aiperf-operator/ 2>&1 | tail -10`
Expected: `1 chart(s) linted, 0 chart(s) failed`.

If lint fails on `values.schema.json` for new fields, update the schema accordingly.

- [ ] **Step 7: Update `sweep_controller/plan_builder.py`**

Run: `grep -n "AIPerfConfig\|aiperf_config" src/aiperf/sweep_controller/plan_builder.py`
Expected: callers that previously assembled an `AIPerfConfig` from the CRD spec.

In `src/aiperf/sweep_controller/plan_builder.py`, update the assembly to build the envelope dict explicitly. Where the function previously did:

```python
configs = [...]
aiperf_config_dict = {**spec.template.spec.benchmark.model_dump(...), "sweep": ..., "multi_run": ...}
```

it should now do:

```python
aiperf_config_dict = {
    "benchmark": spec.template.spec.benchmark.model_dump(by_alias=True, exclude_none=True),
    "sweep": spec.sweep.model_dump(by_alias=True) if spec.sweep else None,
    "multi_run": spec.multi_run.model_dump(by_alias=True) if spec.multi_run else None,
    "variables": spec.variables,
    "random_seed": spec.random_seed,
}
# strip None values to let model defaults kick in
aiperf_config_dict = {k: v for k, v in aiperf_config_dict.items() if v is not None}
```

Adjust to whatever the existing build_plan_from_sweep flow looks like; the point is the envelope keys go at top.

- [ ] **Step 8: Update `aiperf kube generate`**

Run: `grep -n "spec.*benchmark\|AIPerfConfig" src/aiperf/cli_commands/kube/generate.py`
Expected: the function that emits CR YAML from a local config.

For AIPerfJob output: `spec.benchmark` should be the BenchmarkConfig dump (no sweep block inside).
For AIPerfSweep output: envelope keys at `spec.{sweep, multi_run, variables, random_seed}`, body under `spec.template.spec.benchmark`.

Update the generation logic to match.

- [ ] **Step 9: Run K8s tests**

Run: `uv run pytest -n auto tests/unit/cli_commands/kube/ 2>&1 | tail -10`
Expected: pass (after fixtures migrated in Task 10).

Run: `uv run pytest -n auto tests/unit/operator/ 2>&1 | tail -10`
Expected: pass.

- [ ] **Step 10: Commit**

```bash
git add src/aiperf/operator/models.py src/aiperf/kubernetes/sweep_models.py tools/generate_crd.py deploy/helm/aiperf-operator/templates/ src/aiperf/sweep_controller/plan_builder.py src/aiperf/cli_commands/kube/generate.py
git commit -m "$(cat <<'EOF'
feat(k8s)!: restructure AIPerfJob and AIPerfSweep CRDs for envelope shape

- AIPerfJobSpec.benchmark: BenchmarkConfig (was AIPerfConfig). Single
  AIPerfJob carries one benchmark; for sweeps use AIPerfSweep.
- AIPerfSweepSpec gains variables (dict) and random_seed (int|None) at
  envelope, mirroring the YAML envelope.
- "No sweep inside template.spec.benchmark" Pydantic + CEL rules
  removed — the type system enforces it (BenchmarkConfig has no sweep
  field).
- CRDs regenerated via tools/generate_crd.py.
- sweep_controller/plan_builder.py and aiperf kube generate emit the
  envelope shape on both output paths.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 14: New behavior test — flat-shape rejection

**Files:**
- Create: `tests/unit/config/test_envelope_restructure.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/config/test_envelope_restructure.py`:

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plan A behavior tests for the envelope shape restructure.

Spec: docs/superpowers/specs/2026-05-03-aiperf-config-envelope-restructure-design.md
"""

from __future__ import annotations

import textwrap

import pytest

from aiperf.config.loader.core import load_config_from_string
from aiperf.config.loader.errors import ConfigurationError


class TestFlatShapeRejection:
    """The loader rejects pre-restructure flat-shape YAML with a clear migration error."""

    def test_flat_models_at_top_raises_with_migration_hint(self):
        flat = textwrap.dedent("""
            models: [test/model]
            endpoint:
              type: chat
              urls: ["http://localhost:8000/v1/chat/completions"]
            phases:
              - {name: profiling, type: concurrency, requests: 10, concurrency: 1}
        """).strip()

        with pytest.raises(ConfigurationError) as excinfo:
            load_config_from_string(flat, substitute_env=False)
        msg = str(excinfo.value)
        assert "flat shape" in msg
        assert "benchmark:" in msg
        assert "migrate_config_yaml.py" in msg
        assert "models" in msg

    def test_envelope_shape_loads_cleanly(self):
        envelope = textwrap.dedent("""
            benchmark:
              models: [test/model]
              endpoint:
                type: chat
                urls: ["http://localhost:8000/v1/chat/completions"]
              datasets:
                - {name: main, type: synthetic, entries: 100}
              phases:
                - {name: profiling, type: concurrency, requests: 10, concurrency: 1}
            random_seed: 42
        """).strip()

        cfg = load_config_from_string(envelope, substitute_env=False)
        assert cfg.benchmark.models[0].name == "test/model"
        assert cfg.random_seed == 42

    def test_envelope_only_no_benchmark_loads_or_raises_clearly(self):
        envelope_only = "random_seed: 42\nvariables:\n  isl: 128\n"

        with pytest.raises(Exception) as excinfo:
            load_config_from_string(envelope_only, substitute_env=False)
        msg = str(excinfo.value)
        # benchmark is required; missing it should be a clear pydantic error
        assert "benchmark" in msg.lower()
```

- [ ] **Step 2: Run the test**

Run: `uv run pytest -n auto tests/unit/config/test_envelope_restructure.py::TestFlatShapeRejection -v 2>&1 | tail -15`
Expected: PASS for all three tests.

- [ ] **Step 3: Commit**

```bash
git add tests/unit/config/test_envelope_restructure.py
git commit -m "$(cat <<'EOF'
test(config): flat-shape rejection + envelope-loads-cleanly

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 15: New behavior test — scenario run unknown-key rejection

**Files:**
- Modify: `tests/unit/config/test_envelope_restructure.py`

- [ ] **Step 1: Append the test**

Append to `tests/unit/config/test_envelope_restructure.py`:

```python
class TestScenarioRunValidation:
    """Sweep scenario `runs[i]` allow only {name, variables, benchmark}."""

    def test_run_with_top_level_phases_rejects(self):
        from aiperf.config.loader.plan import load_benchmark_plan_from_string

        yaml_str = textwrap.dedent("""
            benchmark:
              models: [test/model]
              endpoint:
                type: chat
                urls: ["http://localhost:8000/v1/chat/completions"]
              datasets:
                - {name: main, type: synthetic, entries: 100}
              phases:
                - {name: profiling, type: concurrency, requests: 10, concurrency: 1}
            sweep:
              type: scenarios
              runs:
                - phases:
                    - {name: profiling, type: concurrency, concurrency: 5}
        """).strip()

        with pytest.raises(ValueError) as excinfo:
            load_benchmark_plan_from_string(yaml_str, substitute_env=False)
        msg = str(excinfo.value)
        assert "unknown field" in msg
        assert "phases" in msg
        assert "name, variables, benchmark" in msg

    def test_run_with_benchmark_wrapper_accepted(self):
        from aiperf.config.loader.plan import load_benchmark_plan_from_string

        yaml_str = textwrap.dedent("""
            benchmark:
              models: [test/model]
              endpoint:
                type: chat
                urls: ["http://localhost:8000/v1/chat/completions"]
              datasets:
                - {name: main, type: synthetic, entries: 100}
              phases:
                - {name: profiling, type: concurrency, requests: 10, concurrency: 1}
            sweep:
              type: scenarios
              runs:
                - benchmark:
                    phases:
                      - {name: profiling, type: concurrency, concurrency: 5}
                - benchmark:
                    phases:
                      - {name: profiling, type: concurrency, concurrency: 10}
        """).strip()

        plan = load_benchmark_plan_from_string(yaml_str, substitute_env=False)
        assert plan.is_sweep
        assert len(plan.configs) == 2
        assert plan.configs[0].phases[0].concurrency == 5
        assert plan.configs[1].phases[0].concurrency == 10
```

- [ ] **Step 2: Run the test**

Run: `uv run pytest -n auto tests/unit/config/test_envelope_restructure.py::TestScenarioRunValidation -v 2>&1 | tail -15`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/unit/config/test_envelope_restructure.py
git commit -m "$(cat <<'EOF'
test(config): scenario run rejects unknown top-level keys

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 16: New behavior test — grid sweep path prefix validation

**Files:**
- Modify: `tests/unit/config/test_envelope_restructure.py`

- [ ] **Step 1: Append the test**

```python
class TestGridSweepPathValidation:
    """Grid sweep variable paths must start with `benchmark.` or `variables.`."""

    def test_unprefixed_path_rejects(self):
        from aiperf.config.loader.plan import load_benchmark_plan_from_string

        yaml_str = textwrap.dedent("""
            benchmark:
              models: [test/model]
              endpoint:
                type: chat
                urls: ["http://localhost:8000/v1/chat/completions"]
              datasets:
                - {name: main, type: synthetic, entries: 100}
              phases:
                - {name: profiling, type: concurrency, requests: 10, concurrency: 1}
            sweep:
              type: grid
              variables:
                "phases.profiling.concurrency": [1, 2, 4]
        """).strip()

        with pytest.raises(ValueError) as excinfo:
            load_benchmark_plan_from_string(yaml_str, substitute_env=False)
        msg = str(excinfo.value)
        assert "non-sweepable" in msg
        assert "benchmark.*" in msg
        assert "phases.profiling.concurrency" in msg

    def test_prefixed_path_accepts(self):
        from aiperf.config.loader.plan import load_benchmark_plan_from_string

        yaml_str = textwrap.dedent("""
            benchmark:
              models: [test/model]
              endpoint:
                type: chat
                urls: ["http://localhost:8000/v1/chat/completions"]
              datasets:
                - {name: main, type: synthetic, entries: 100}
              phases:
                - {name: profiling, type: concurrency, requests: 10, concurrency: 1}
            sweep:
              type: grid
              variables:
                "benchmark.phases.profiling.concurrency": [1, 2, 4]
        """).strip()

        plan = load_benchmark_plan_from_string(yaml_str, substitute_env=False)
        assert plan.is_sweep
        assert len(plan.configs) == 3

    def test_runtime_path_rejected(self):
        from aiperf.config.loader.plan import load_benchmark_plan_from_string

        yaml_str = textwrap.dedent("""
            benchmark:
              models: [test/model]
              endpoint:
                type: chat
                urls: ["http://localhost:8000/v1/chat/completions"]
              datasets:
                - {name: main, type: synthetic, entries: 100}
              phases:
                - {name: profiling, type: concurrency, requests: 10, concurrency: 1}
            sweep:
              type: grid
              variables:
                "runtime.workers": [1, 2]
        """).strip()

        with pytest.raises(ValueError) as excinfo:
            load_benchmark_plan_from_string(yaml_str, substitute_env=False)
        msg = str(excinfo.value)
        assert "runtime.workers" in msg
        assert "non-sweepable" in msg
```

- [ ] **Step 2: Run the tests**

Run: `uv run pytest -n auto tests/unit/config/test_envelope_restructure.py::TestGridSweepPathValidation -v 2>&1 | tail -15`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/unit/config/test_envelope_restructure.py
git commit -m "$(cat <<'EOF'
test(config): grid sweep rejects non-sweepable path prefixes

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 17: Delete the one-off `tools/migrate_test_yaml_literals.py`

**Files:**
- Delete: `tools/migrate_test_yaml_literals.py`

- [ ] **Step 1: Confirm no reference**

Run: `grep -rn "migrate_test_yaml_literals" --include="*.py" --include="*.md"`
Expected: nothing besides this plan file.

- [ ] **Step 2: Delete and commit**

```bash
git rm tools/migrate_test_yaml_literals.py
git commit -m "$(cat <<'EOF'
chore: remove one-off tools/migrate_test_yaml_literals.py

Plan-internal helper used to bulk-migrate triple-quoted YAML literals
in test files. tools/migrate_config_yaml.py remains as the
user-facing migration utility.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 18: Create `docs/tutorials/migrating-config.md`

**Files:**
- Create: `docs/tutorials/migrating-config.md`
- Modify: `docs/index.yml`
- Modify: `README.md` (tutorial index)
- Modify: `llms.txt` (one-line entry for the new doc)

- [ ] **Step 1: Create the migration doc**

Create `docs/tutorials/migrating-config.md`:

```markdown
<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Migrating to the envelope config shape

AIPerf YAML configs use an envelope shape that separates sweep machinery from the swept benchmark body. If you are migrating from a pre-envelope flat config, follow this guide.

## What changed

**Before (flat):**

```yaml
models: [llama]
endpoint:
  urls: ["http://localhost:8000/v1/chat/completions"]
datasets:
  - {name: main, type: synthetic, entries: 200}
phases:
  - {name: profiling, type: concurrency, requests: 10, concurrency: 1}
random_seed: 42
sweep:
  type: grid
  variables:
    "phases.profiling.concurrency": [1, 2, 4]
```

**After (envelope):**

```yaml
random_seed: 42
sweep:
  type: grid
  variables:
    "benchmark.phases.profiling.concurrency": [1, 2, 4]
benchmark:
  models: [llama]
  endpoint:
    urls: ["http://localhost:8000/v1/chat/completions"]
  datasets:
    - {name: main, type: synthetic, entries: 200}
  phases:
    - {name: profiling, type: concurrency, requests: 10, concurrency: 1}
```

## Run the migration script

```bash
uv run python tools/migrate_config_yaml.py path/to/config.yaml --in-place
```

The script:
- Re-indents body fields under a top-level `benchmark:` key.
- Keeps envelope keys (`sweep`, `multi_run`, `variables`, `random_seed`) at top level.
- Prefixes grid `sweep.variables` keys with `benchmark.` (e.g. `phases.profiling.concurrency` → `benchmark.phases.profiling.concurrency`).
- Wraps body fields inside `sweep.runs[i]` under a per-run `benchmark:` key.
- Preserves comments via ruamel.yaml.
- Idempotent — running it twice is a no-op.

## Body fields (move under `benchmark:`)

`models`, `endpoint`, `datasets`, `phases`, `artifacts`, `slos`, `tokenizer`, `gpu_telemetry`, `server_metrics`, `runtime`, `logging`, `metrics`, `accuracy`.

## Envelope fields (stay at top level)

`sweep`, `multi_run`, `variables`, `random_seed`.

## Templates and Jinja

Templates that referenced body keys without a prefix continue to work:

```yaml
variables:
  rate: 100
benchmark:
  phases:
    - name: profiling
      type: rate
      rate: "{{ rate }}"  # still works
```

The loader aliases body keys at the top level of the Jinja context. You can also use the explicit `{{ benchmark.phases.profiling.rate }}` form.

## Why the change

The envelope shape mirrors how AIPerfSweep CRDs are structured on the K8s side: cross-variation machinery (sweep, multi_run, variables, random_seed) at envelope level; the swept benchmark workload as the body. This eliminates a long-standing asymmetry between the YAML and CRD surfaces, and makes scenario merge logic trivial — only the `benchmark:` subtree merges per variation; envelope fields are constant across variations.

## Common gotchas

- **Scenario `runs[i]` body keys must be wrapped under `benchmark:`.** A run carrying `phases:` directly raises `unknown field 'phases' in sweep run [0]; allowed: name, variables, benchmark`. Use `runs: [{benchmark: {phases: [...]}}]`.
- **Grid `sweep.variables` paths must be envelope-rooted.** `phases.profiling.rate: [...]` raises `non-sweepable subtree`. Use `benchmark.phases.profiling.rate`.
- **`AIPerfJob` CRDs no longer carry sweep blocks inside `spec.benchmark`.** For sweeps on Kubernetes, use `AIPerfSweep`.
```

- [ ] **Step 2: Add to `docs/index.yml`**

Run: `grep -A 1 "migrating\|tutorials" docs/index.yml | head -10`
Expected: existing tutorial entries.

Add a new tutorial entry:

```yaml
  - page: Migrating to the envelope config shape
    path: ./tutorials/migrating-config.md
```

Place it under the appropriate section (likely `tutorials` or `migration`).

- [ ] **Step 3: Add to `README.md` tutorial index**

In `README.md`, find the "Tutorials" or "Configuration" section and add a one-line link:

```markdown
- [Migrating to the envelope config shape](docs/tutorials/migrating-config.md)
```

- [ ] **Step 4: Add to `llms.txt`**

In `llms.txt`, find the tutorials section and add:

```
- docs/tutorials/migrating-config.md - One-shot migration to the envelope config shape (Plan A)
```

- [ ] **Step 5: Commit**

```bash
git add docs/tutorials/migrating-config.md docs/index.yml README.md llms.txt
git commit -m "$(cat <<'EOF'
docs(tutorials): add migrating-config.md for envelope shape migration

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 19: Update `docs/architecture.md` and `docs/dev/patterns.md`

**Files:**
- Modify: `docs/architecture.md`
- Modify: `docs/dev/patterns.md`

- [ ] **Step 1: Add envelope-vs-body subsection to architecture doc**

Open `docs/architecture.md` and find the configuration / config-plane section. Add a subsection:

```markdown
### Envelope vs Benchmark Body

`AIPerfConfig` is an envelope wrapping a `BenchmarkConfig`:

```python
class AIPerfConfig(BaseConfig):
    benchmark: BenchmarkConfig          # the swept body
    sweep: SweepConfig | None = None    # variation generator
    multi_run: MultiRunConfig           # trial / convergence config
    variables: dict[str, Any]            # Jinja context
    random_seed: int | None              # base seed for per-variation derivation
```

The split mirrors `AIPerfSweep` on the K8s side: cross-variation machinery (sweep, multi_run, variables, random_seed) at envelope level; the swept benchmark body as a separate concern. Sweep expansion only ever merges into the `benchmark:` subtree; envelope fields are constant across variations.

When code reads body fields, the local-alias pattern keeps call sites concise:

```python
def setup(self, config: AIPerfConfig) -> None:
    bench = config.benchmark
    if bench.endpoint.streaming:
        ...
```

YAML configs follow the same shape — see [docs/tutorials/migrating-config.md](tutorials/migrating-config.md) for examples.
```

- [ ] **Step 2: Add the local-alias pattern to `docs/dev/patterns.md`**

Open `docs/dev/patterns.md` and add a section under existing patterns:

```markdown
### Reading body fields from `AIPerfConfig`

`AIPerfConfig` is an envelope; body fields live at `.benchmark.`. For functions that read multiple body fields, alias once at the top:

```python
def configure_workers(config: AIPerfConfig) -> WorkerSettings:
    bench = config.benchmark
    return WorkerSettings(
        endpoint=bench.endpoint.urls[0],
        streaming=bench.endpoint.streaming,
        request_count=bench.phases[0].requests,
    )
```

For functions that only need the body (no envelope-level access), narrow the parameter type:

```python
def render_dataset_prompt(bench: BenchmarkConfig, idx: int) -> str:
    return bench.datasets[0].prompts.template.format(idx=idx)
```
```

- [ ] **Step 3: Commit**

```bash
git add docs/architecture.md docs/dev/patterns.md
git commit -m "$(cat <<'EOF'
docs: document envelope vs benchmark body split + local-alias pattern

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 20: Four-file sync — CLAUDE.md, AGENTS.md, copilot-instructions, cursor rules

**Files:**
- Modify: `CLAUDE.md`
- Modify: `AGENTS.md`
- Modify: `.github/copilot-instructions.md`
- Modify: `.cursor/rules/python.mdc`

- [ ] **Step 1: Read all four to confirm they're in sync today**

Run: `make check-agent-files-sync 2>&1 | tail -5`
Expected: pass.

- [ ] **Step 2: Pick one (CLAUDE.md) as the master and update it**

Open `CLAUDE.md`. Find the section for adding new config fields (likely under "Adding a New Config Field" or similar). Replace or add:

```markdown
## Adding a New Config Field

`AIPerfConfig` is an envelope; `BenchmarkConfig` is the swept body.

- **Does the field vary per sweep variation?** → add it to `BenchmarkConfig` (`src/aiperf/config/config.py`).
- **Is it cross-variation machinery?** (Jinja context, sweep config, seed, multi-run trial settings.) → add it to `AIPerfConfig` envelope.

YAML configs follow the same shape: body keys nest under `benchmark:`; envelope keys (`sweep`, `multi_run`, `variables`, `random_seed`) stay at top level.

Reading body fields:

```python
bench = config.benchmark
if bench.endpoint.streaming:
    ...
```
```

Update any YAML examples in CLAUDE.md to use the envelope shape.

- [ ] **Step 3: Mirror the changes to the other three files**

The four files must match. Apply the same edits to:
- `AGENTS.md`
- `.github/copilot-instructions.md`
- `.cursor/rules/python.mdc` (preserve the `alwaysApply: true` frontmatter)

- [ ] **Step 4: Verify sync**

Run: `make check-agent-files-sync 2>&1 | tail -5`
Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add CLAUDE.md AGENTS.md .github/copilot-instructions.md .cursor/rules/python.mdc
git commit -m "$(cat <<'EOF'
docs: four-file sync — envelope vs benchmark body for new config fields

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 21: Auto-regenerate `docs/cli-options.md`, `docs/environment-variables.md`

**Files:**
- Regen: `docs/cli-options.md`
- Regen: `docs/environment-variables.md`

- [ ] **Step 1: Run the generators**

Run: `make generate-all-docs 2>&1 | tail -5`
Expected: completes without error. Both files refresh.

- [ ] **Step 2: Inspect the diff**

Run: `git diff docs/cli-options.md docs/environment-variables.md | head -40`
Expected: small diff reflecting the new envelope shape in option descriptions, if any.

- [ ] **Step 3: Commit**

```bash
git add docs/cli-options.md docs/environment-variables.md
git commit -m "$(cat <<'EOF'
docs: regenerate CLI and env-var reference for envelope shape

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 22: Migrate tutorial YAML examples

**Files:**
- Modify: `docs/tutorials/*.md`

- [ ] **Step 1: Find all YAML code blocks in tutorials**

Run: `grep -rln "^\`\`\`yaml" docs/tutorials/*.md`
Expected: list of tutorial files containing YAML examples.

- [ ] **Step 2: Migrate each tutorial's YAML examples**

For each tutorial file: extract each ```yaml fenced block, run `migrate_config_yaml.py` mentally (or via a one-off helper script), and replace the original with the envelope-shaped version. Keep prose narrative aligned with the new shape (e.g., descriptions of "the `phases:` section" become "the `benchmark.phases:` section" or are reworded around the envelope shape).

This is by-hand prose work; automation isn't worth it for ~10 tutorial files.

- [ ] **Step 3: Verify each tutorial example loads**

For each tutorial, save the YAML to a temp file and try loading it:

```bash
for yaml_block in /tmp/tutorial-*.yaml; do
    uv run python -c "
from aiperf.config.loader import load_config
load_config('$yaml_block', substitute_env=False)
print('$yaml_block: OK')
" || echo "$yaml_block: FAILED"
done
```
Expected: each prints OK.

- [ ] **Step 4: Commit**

```bash
git add docs/tutorials/
git commit -m "$(cat <<'EOF'
docs(tutorials): migrate YAML examples to envelope shape

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 23: Final verification

**Files:** none

- [ ] **Step 1: Full unit suite**

Run: `uv run pytest -n auto tests/unit/ 2>&1 | tail -10`
Expected: green.

- [ ] **Step 2: Component-integration suite**

Run: `uv run pytest -n auto tests/component_integration/ 2>&1 | tail -10`
Expected: green.

- [ ] **Step 3: Integration suite**

Run: `uv run pytest -n auto tests/integration/ 2>&1 | tail -10`
Expected: green.

- [ ] **Step 4: Lint + format**

Run: `ruff format . && ruff check --fix .`
Expected: no diffs (or only auto-applied fixes).

If ruff applies fixes:
```bash
git add -p
git commit -m "$(cat <<'EOF'
style: ruff auto-fixes from envelope restructure

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 5: Pre-commit on all-files**

Run: `pre-commit run --all-files 2>&1 | tail -30`
Expected: all hooks pass.

- [ ] **Step 6: Helm lint + CRD generation check**

Run: `uv run python tools/generate_crd.py --check && helm lint deploy/helm/aiperf-operator/ 2>&1 | tail -10`
Expected: CRD generation reports up-to-date; helm lint reports `1 chart(s) linted, 0 chart(s) failed`.

- [ ] **Step 7: Spec coverage check**

Open `docs/superpowers/specs/2026-05-03-aiperf-config-envelope-restructure-design.md` and walk through each section. Confirm each requirement maps to a task and is implemented:

- Models restructure ✓ (Task 2-3)
- Loader flat-shape rejection ✓ (Task 4)
- Jinja context ✓ (Task 5)
- expand_sweep envelope-shape ✓ (Task 6)
- build_benchmark_plan envelope-shape ✓ (Task 7)
- random_seed envelope-level + per-variation derivation ✓ (Task 8)
- v1 converter emits envelope shape ✓ (Task 9)
- Test fixture migration ✓ (Tasks 10-11)
- Call-site rewrites ✓ (Task 12)
- K8s models + CRD regen ✓ (Task 13)
- New behavior tests ✓ (Tasks 14-16)
- Migration doc ✓ (Task 18)
- Architecture + patterns docs ✓ (Task 19)
- Four-file sync ✓ (Task 20)
- Auto-regen reference docs ✓ (Task 21)
- Tutorial migration ✓ (Task 22)
- Migration script tools/migrate_config_yaml.py ✓ (Task 1)

- [ ] **Step 8: Push the worktree branch**

```bash
git push -u origin ajc/config-envelope
```

The branch is ready for review or merge into `ajc/k8s`.

---

## Self-review notes

- **Spec coverage:** Every section of the spec maps to at least one task. Plan A's K8s impact (Section 3 of spec) covered by Task 13. Plan B and Plan C are explicit out-of-scope (no tasks).
- **Type consistency:** `AIPerfConfig.benchmark: BenchmarkConfig` consistent across Tasks 2, 7, 8, 13. `BenchmarkPlan.variation_seeds: list[int | None]` consistent in Task 8 and any seed-derivation references.
- **No placeholders:** Every step has either an exact command or a complete code block.
- **Rollback:** `git revert` of the merge commit cleanly reverts everything; pre-restructure tooling fully recovered.
- **Task ordering** is critical because the test suite goes RED in Tasks 2-9 and recovers across Tasks 10-12. Acceptance criterion is final-suite-green in Task 23, not per-task.
