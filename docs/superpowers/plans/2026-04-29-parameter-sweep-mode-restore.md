# Restore `--parameter-sweep-mode` on ajc/k8s

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Each implementation subagent must commit with `git commit --no-verify` and run ruff manually (per `gotcha_precommit_auto_stash_destroys_parallel_agents`); pre-commit's internal stash corrupts state under parallel agents.

**Goal:** Re-introduce the `--parameter-sweep-mode {independent,repeated}` flag dropped during the PR #699 port, so `repeated` ordering (trial outer, variation inner) is selectable on both the in-process and k8s sweep paths.

**Architecture:** Mode lives on `MultiRunConfig` (canonical) → `BenchmarkPlan.parameter_sweep_mode`. Iteration-order dispatch lives in `MultiRunOrchestrator.execute` (in-process + k8s share this). Strategies stay cell-local; the orchestrator owns the loop. Sweep aggregation (`aggregate_sweep_and_export`) is already mode-agnostic — it groups by `variation_values` post-hoc and needs no changes. Default is `INDEPENDENT` to preserve current branch behavior; main's `repeated` default is *not* matched (call-out in docs).

**Tech Stack:** Pydantic v2, cyclopts CLI, kopf operator, kubernetes_asyncio, pytest-xdist.

---

## File map

| File | Responsibility | Action |
|---|---|---|
| `src/aiperf/config/v1/_loadgen.py` | UserConfig CLI surface | Add `parameter_sweep_mode` field |
| `src/aiperf/config/v1/_converter_optionals.py` | v1→v2 multi_run mapping | Add `"parameter_sweep_mode" → "mode"` mapping |
| `src/aiperf/config/v1/converter.py` | v1→v2 entry | Delete dead-code `_reject_unsupported_sweep_mode` |
| `src/aiperf/config/_models_benchmark.py` | Canonical `MultiRunConfig` | Add `mode: SweepMode` field |
| `src/aiperf/config/benchmark.py` | `BenchmarkPlan` | Add `parameter_sweep_mode: SweepMode` field |
| `src/aiperf/config/loader/plan.py` | `build_benchmark_plan` | Read `multi_run["mode"]` into plan_kwargs |
| `src/aiperf/orchestrator/orchestrator.py` | Iteration-order dispatch | Split `execute` into `_execute_independent` (current loop) + `_execute_repeated` (loops swapped) |
| `src/aiperf/sweep_controller/main.py` | k8s child manifest derivation | Swap `var_idx`/`trial_idx` derivation when `mode=REPEATED` |
| `src/aiperf/sweep_controller/plan_builder.py` | Build plan from CR spec | Propagate `multi_run.mode` into plan |
| `src/aiperf/kubernetes/sweep_models.py` | `MultiRun` CR model | Add `mode: SweepMode` field |
| `deploy/helm/aiperf-operator/templates/crd-aiperfsweep.yaml` | CRD schema | Regenerated via `make generate-all-plugin-files` (or its kubernetes equivalent) |
| `tests/integration/test_parameter_sweep.py` | Integration coverage | Replace `test_sweep_repeated_mode_is_rejected` with two positive tests |
| `tests/unit/config/v1/test_loadgen_converter.py` | Converter coverage | Replace dead-code rejector test with mode-passthrough test |
| `tests/unit/orchestrator/test_orchestrator.py` (new file or extend existing) | Iteration-order assertion | Add `test_repeated_mode_iterates_trial_outer` + `test_independent_mode_iterates_variation_outer` |
| `tests/unit/kubernetes/test_sweep_models.py` | CRD model coverage | Add mode-roundtrip test |
| `docs/cli-options.md` | Auto-generated | Regenerated via `make generate-cli-docs` |
| `docs/kubernetes/sweeps.md` | Sweep tutorial | Document mode flag + path layouts |
| `CLAUDE.md` + `AGENTS.md` + `.github/copilot-instructions.md` + `.cursor/rules/python.mdc` | Four-file sync | Strike "INDEPENDENT only" notes; describe mode dispatch site |

---

## Conventions for every task

- Run `ruff format src/aiperf tests && ruff check --fix src/aiperf tests` before committing.
- Verify with **one** `pytest -n auto` invocation per task targeting the affected unit subfolder. Don't split into multiple pytest calls.
- Commit with `git commit --no-verify -s` (pre-commit's internal stash corrupts parallel-agent state). Sign-off is required.
- Use exactly the import patterns and CLIParameter shape already present in the file you're modifying.

---

### Task 1: Add `parameter_sweep_mode` to v1 LoadGeneratorConfig

**Files:**
- Modify: `src/aiperf/config/v1/_loadgen.py:564-578` (insert new field after `parameter_sweep_same_seed`)

- [ ] **Step 1: Add the field**

In `src/aiperf/config/v1/_loadgen.py`, after the `parameter_sweep_same_seed` field block ending at line 578, add:

```python
    parameter_sweep_mode: Annotated[
        SweepMode,
        Field(
            description="Execution order for sweep + multi-trial composition. "
            "'independent' (default) iterates variations as the outer loop and "
            "trials as the inner loop, so all trials at one variation complete "
            "before the next variation starts. 'repeated' inverts the loops: "
            "all variations run within trial 1, then within trial 2, etc. "
            "Both modes produce the same total runs, only the artifact-path "
            "layout and submit order differ. Note: this branch defaults to "
            "'independent'; main's PR #699 defaults to 'repeated'.",
        ),
        CLIParameter(
            name=("--parameter-sweep-mode",),
            group=Groups.MULTI_RUN,
        ),
    ] = SweepMode.INDEPENDENT
```

Add this import at the top of the file, near the other `aiperf` imports:

```python
from aiperf.orchestrator.strategies import SweepMode
```

- [ ] **Step 2: Verify cyclopts wiring**

Run: `pytest -n auto tests/unit/config/v1/`
Expected: PASS (no new test yet — this confirms the field doesn't break parsing).

- [ ] **Step 3: Commit**

```bash
git add src/aiperf/config/v1/_loadgen.py
git commit --no-verify -s -m "feat(sweep): add --parameter-sweep-mode to v1 LoadGeneratorConfig

Re-introduces the cyclopts CLI flag dropped during the PR #699 port.
Default INDEPENDENT preserves current branch behavior; main defaults
to repeated. Field has no validator (v1 contract); validation lives
on AIPerfConfig + the converter.
"
```

---

### Task 2: Add `mode` to canonical `MultiRunConfig`, plumb through `BenchmarkPlan`

**Files:**
- Modify: `src/aiperf/config/_models_benchmark.py:21+` (add `mode` field on `MultiRunConfig`)
- Modify: `src/aiperf/config/benchmark.py:118+` (add `parameter_sweep_mode` field on `BenchmarkPlan`)
- Modify: `src/aiperf/config/loader/plan.py:69-81` (read mode into plan_kwargs)
- Modify: `src/aiperf/config/v1/_converter_optionals.py:79-91` (add mapping entry)
- Modify: `src/aiperf/config/v1/converter.py:119-186` (delete `_reject_unsupported_sweep_mode` and the call at line 183)

- [ ] **Step 1: Add `mode` field to `MultiRunConfig`**

In `src/aiperf/config/_models_benchmark.py`, after the `parameter_sweep_same_seed` field (line 144), add:

```python
    mode: Annotated[
        SweepMode,
        Field(
            default=SweepMode.INDEPENDENT,
            description="Execution order for sweep + multi-trial composition. "
            "'independent' (default): variations outer, trials inner — artifact "
            "tree is <base>/<variation>/profile_runs/run_NNNN/. 'repeated': "
            "trials outer, variations inner — artifact tree is "
            "<base>/profile_runs/trial_NNNN/<variation>/. Mode-dispatch lives "
            "in MultiRunOrchestrator.execute.",
        ),
    ]
```

Add to imports at the top:

```python
from aiperf.orchestrator.strategies import SweepMode
```

- [ ] **Step 2: Add field to `BenchmarkPlan`**

In `src/aiperf/config/benchmark.py`, after the `parameter_sweep_same_seed` field (line 137), add:

```python
    parameter_sweep_mode: SweepMode = Field(
        default=SweepMode.INDEPENDENT,
        description=(
            "Iteration order for sweep + multi-trial. 'independent' (default) "
            "iterates variations outer, trials inner. 'repeated' iterates "
            "trials outer, variations inner. Honored by MultiRunOrchestrator."
        ),
    )
```

Add `from aiperf.orchestrator.strategies import SweepMode` to imports.

- [ ] **Step 3: Read mode into plan_kwargs**

In `src/aiperf/config/loader/plan.py`, replace the `plan_kwargs` block (lines 69-81) so it includes:

```python
    plan_kwargs: dict[str, Any] = dict(
        configs=configs,
        variations=variations,
        trials=multi_run.get("num_runs", 1),
        cooldown_seconds=multi_run.get("cooldown_seconds", 0.0),
        confidence_level=multi_run.get("confidence_level", 0.95),
        set_consistent_seed=multi_run.get("set_consistent_seed", True),
        disable_warmup_after_first=multi_run.get("disable_warmup_after_first", True),
        parameter_sweep_cooldown_seconds=multi_run.get(
            "parameter_sweep_cooldown_seconds", 0.0
        ),
        parameter_sweep_same_seed=multi_run.get("parameter_sweep_same_seed", False),
        parameter_sweep_mode=multi_run.get("mode", "independent"),
    )
```

(`SweepMode` is `CaseInsensitiveStrEnum`, so the string `"independent"` coerces.)

- [ ] **Step 4: Add mapping entry in converter**

In `src/aiperf/config/v1/_converter_optionals.py`, in `build_multi_run`'s `mapping` dict (lines 79-91), append:

```python
        "parameter_sweep_mode": "mode",
```

- [ ] **Step 5: Delete dead-code rejector**

In `src/aiperf/config/v1/converter.py`:
- Delete the entire `_reject_unsupported_sweep_mode` function (lines 119-138).
- Delete its call site at line 183 (`_reject_unsupported_sweep_mode(user)`).

- [ ] **Step 6: Replace converter test**

In `tests/unit/config/v1/test_loadgen_converter.py`, find the test that validates `_reject_unsupported_sweep_mode` raises (search the file for `parameter_sweep_mode`). Replace with positive coverage:

```python
def test_parameter_sweep_mode_repeated_flows_through_to_multi_run() -> None:
    """`--parameter-sweep-mode=repeated` lands as multi_run.mode='repeated'."""
    user = UserConfig.model_validate({
        "endpoint": {"model_names": ["m"], "url": "http://x"},
        "loadgen": {"concurrency": [10, 20], "num_profile_runs": 2,
                    "parameter_sweep_mode": "repeated"},
    })
    service = ServiceConfig()
    cfg = convert_user_to_aiperf(user, service)
    assert cfg.multi_run.mode == SweepMode.REPEATED


def test_parameter_sweep_mode_default_is_independent() -> None:
    """Omitted flag → multi_run.mode=INDEPENDENT (this branch's default)."""
    user = UserConfig.model_validate({
        "endpoint": {"model_names": ["m"], "url": "http://x"},
        "loadgen": {"concurrency": [10, 20], "num_profile_runs": 2},
    })
    service = ServiceConfig()
    cfg = convert_user_to_aiperf(user, service)
    assert cfg.multi_run.mode == SweepMode.INDEPENDENT
```

Add the imports if missing: `from aiperf.orchestrator.strategies import SweepMode`, `from aiperf.config.v1 import UserConfig, ServiceConfig`, `from aiperf.config.v1.converter import convert_user_to_aiperf`.

- [ ] **Step 7: Run unit suite**

Run: `pytest -n auto tests/unit/config/`
Expected: PASS (new tests pass, existing tests untouched).

- [ ] **Step 8: Commit**

```bash
git add src/aiperf/config tests/unit/config/v1/test_loadgen_converter.py
git commit --no-verify -s -m "feat(sweep): plumb parameter_sweep_mode through MultiRunConfig + BenchmarkPlan

Adds mode field on MultiRunConfig (canonical) and BenchmarkPlan, wires
the v1->v2 mapping in build_multi_run, and reads mode into plan_kwargs.
Deletes the dead-code _reject_unsupported_sweep_mode rejector. Replaces
its unit test with positive coverage that the flag round-trips to
cfg.multi_run.mode.
"
```

---

### Task 3: Implement `repeated` iteration-order in `MultiRunOrchestrator.execute`

**Files:**
- Modify: `src/aiperf/orchestrator/orchestrator.py:50-139` (split execute by mode)
- Test: `tests/unit/orchestrator/test_orchestrator.py` (add iteration-order tests)

- [ ] **Step 1: Write failing tests for both modes**

In `tests/unit/orchestrator/test_orchestrator.py`, add (or create the file if it does not yet have an order-assertion test):

```python
import asyncio
from pathlib import Path

import pytest

from aiperf.config.benchmark import BenchmarkConfig, BenchmarkPlan
from aiperf.config.sweep import SweepVariation
from aiperf.orchestrator.executor import RunExecutor
from aiperf.orchestrator.models import RunResult
from aiperf.orchestrator.orchestrator import MultiRunOrchestrator
from aiperf.orchestrator.strategies import SweepMode


class _RecordingExecutor(RunExecutor):
    """Minimal RunExecutor that records (variation_index, trial_index) order."""

    def __init__(self) -> None:
        self.calls: list[tuple[int, int]] = []

    def derive_id(self, plan, var_idx, trial) -> str:
        return f"v{var_idx}-t{trial}"

    async def execute(self, run) -> RunResult:
        self.calls.append((run.variation.index, run.trial))
        return RunResult(
            benchmark_id=run.benchmark_id,
            label=run.label,
            success=True,
            artifact_dir=run.artifact_dir,
        )


def _two_variation_plan(mode: SweepMode, trials: int = 3) -> BenchmarkPlan:
    cfg = BenchmarkConfig.model_validate({
        # Minimal valid BenchmarkConfig — adapt to the actual required fields.
    })
    variations = [
        SweepVariation(index=0, label="phases.profiling.concurrency=10", values={"concurrency": 10}),
        SweepVariation(index=1, label="phases.profiling.concurrency=20", values={"concurrency": 20}),
    ]
    return BenchmarkPlan(
        configs=[cfg, cfg.model_copy(deep=True)],
        variations=variations,
        trials=trials,
        parameter_sweep_mode=mode,
    )


@pytest.mark.asyncio
async def test_independent_mode_iterates_variation_outer(tmp_path: Path) -> None:
    """Independent: (v0,t0),(v0,t1),(v0,t2),(v1,t0),(v1,t1),(v1,t2)."""
    plan = _two_variation_plan(SweepMode.INDEPENDENT)
    executor = _RecordingExecutor()
    await MultiRunOrchestrator(tmp_path).execute(plan, executor)
    assert executor.calls == [(0,0),(0,1),(0,2),(1,0),(1,1),(1,2)]


@pytest.mark.asyncio
async def test_repeated_mode_iterates_trial_outer(tmp_path: Path) -> None:
    """Repeated: (v0,t0),(v1,t0),(v0,t1),(v1,t1),(v0,t2),(v1,t2)."""
    plan = _two_variation_plan(SweepMode.REPEATED)
    executor = _RecordingExecutor()
    await MultiRunOrchestrator(tmp_path).execute(plan, executor)
    assert executor.calls == [(0,0),(1,0),(0,1),(1,1),(0,2),(1,2)]


@pytest.mark.asyncio
async def test_repeated_mode_artifact_path_layout(tmp_path: Path) -> None:
    """Repeated artifact dirs are <base>/profile_runs/trial_NNNN/<variation>/."""
    plan = _two_variation_plan(SweepMode.REPEATED, trials=2)
    seen: list[Path] = []

    class _PathRecorder(_RecordingExecutor):
        async def execute(self, run):
            seen.append(run.artifact_dir)
            return await super().execute(run)

    await MultiRunOrchestrator(tmp_path).execute(plan, _PathRecorder())
    assert seen[0] == tmp_path / "profile_runs" / "trial_0001" / "phases.profiling.concurrency=10" / "profile_runs" / "run_0001"
    # Note: strategy.get_run_path still appends profile_runs/run_0001 inside the
    # cell. The repeated-mode "trial_NNNN" lives ABOVE the cell label, so the
    # full path concatenates the orchestrator's repeated prefix with the
    # strategy's own per-cell suffix.
```

Adjust the `BenchmarkConfig.model_validate` minimal example to whatever the canonical minimal-valid shape is (read `tests/unit/orchestrator/test_orchestrator.py` for an existing helper if present).

Run: `pytest -n auto tests/unit/orchestrator/`
Expected: FAIL — `parameter_sweep_mode` is honored but the orchestrator still iterates variation-outer regardless of mode.

- [ ] **Step 2: Refactor `execute` to dispatch on mode**

Replace the body of `MultiRunOrchestrator.execute` in `src/aiperf/orchestrator/orchestrator.py` (lines 50-139) with:

```python
    async def execute(
        self,
        plan: BenchmarkPlan,
        executor: RunExecutor,
        *,
        cancel_check: Callable[[], bool] | None = None,
    ) -> list[RunResult]:
        """Execute all (variation, trial) runs in the plan.

        Iteration order honors plan.parameter_sweep_mode:

        - INDEPENDENT (default): variations outer, trials inner. Artifact
          tree is <base>/<variation>/profile_runs/run_NNNN/.
        - REPEATED: trials outer, variations inner. Artifact tree is
          <base>/profile_runs/trial_NNNN/<variation>/profile_runs/run_0001/.
          (The trailing run_0001 segment comes from the per-cell strategy
          and is unconditional; trial-NNNN is the orchestrator's prefix.)
        """
        from aiperf.orchestrator.strategies import SweepMode

        if plan.parameter_sweep_mode == SweepMode.REPEATED:
            return await self._execute_repeated(plan, executor, cancel_check=cancel_check)
        return await self._execute_independent(plan, executor, cancel_check=cancel_check)
```

Then add `_execute_independent` containing the *current* loop body (lines 73-139, but renamed `self.base_dir` references unchanged), and `_execute_repeated` with the loops swapped:

```python
    async def _execute_independent(
        self,
        plan: BenchmarkPlan,
        executor: RunExecutor,
        *,
        cancel_check: Callable[[], bool] | None,
    ) -> list[RunResult]:
        # ... move the existing 73-139 body here verbatim ...

    async def _execute_repeated(
        self,
        plan: BenchmarkPlan,
        executor: RunExecutor,
        *,
        cancel_check: Callable[[], bool] | None,
    ) -> list[RunResult]:
        from aiperf._cli_runner_helpers import build_strategy
        from aiperf.config.benchmark import BenchmarkRun

        all_results: list[RunResult] = []
        logger.info(
            f"Starting multi-run benchmark (repeated): {plan.trials} trials × "
            f"{len(plan.configs)} variations"
        )

        # Pre-build per-variation strategies so `should_continue` and
        # `get_run_path` are stable across the trial-outer loop. Each
        # strategy still owns only one cell's worth of state.
        strategies = [build_strategy(plan, logger) for _ in plan.configs]
        for strategy, cfg in zip(strategies, plan.configs, strict=True):
            strategy.validate_config(cfg)

        for trial in range(plan.trials):
            if cancel_check is not None and cancel_check():
                logger.info(f"Sweep cancelled at trial {trial}; aborting")
                return all_results
            for var_idx, (cfg, variation) in enumerate(
                zip(plan.configs, plan.variations, strict=False)
            ):
                if cancel_check is not None and cancel_check():
                    logger.info(
                        f"Sweep cancelled mid-trial at trial={trial} v={var_idx}; aborting"
                    )
                    return all_results
                strategy = strategies[var_idx]
                next_cfg = strategy.get_next_config(cfg, [])  # cell state unused in repeated
                label = strategy.get_run_label(trial)
                cell_dir = self.base_dir / "profile_runs" / f"trial_{trial + 1:04d}" / variation.label
                artifact_dir = strategy.get_run_path(cell_dir, trial)

                run = BenchmarkRun(
                    benchmark_id=executor.derive_id(plan, var_idx, trial),
                    cfg=next_cfg,
                    variation=variation,
                    trial=trial,
                    label=label,
                    artifact_dir=artifact_dir,
                )
                logger.info(f"[trial={trial} v{var_idx}] Executing {label}...")
                result = await executor.execute(run)
                self._stamp_variation_metadata(result, run, trial)
                all_results.append(result)

                if self._sweep_failure_threshold_exceeded(all_results, plan):
                    logger.warning("Failure threshold exceeded; aborting sweep")
                    return all_results

                if var_idx + 1 < len(plan.configs) and plan.parameter_sweep_cooldown_seconds > 0:
                    logger.debug(
                        f"Inter-variation cooldown (within trial {trial}): "
                        f"{plan.parameter_sweep_cooldown_seconds}s"
                    )
                    await asyncio.sleep(plan.parameter_sweep_cooldown_seconds)

            if trial + 1 < plan.trials:
                cooldown = strategies[0].get_cooldown_seconds()
                if cooldown > 0:
                    logger.info(f"Inter-trial cooldown: {cooldown}s")
                    await asyncio.sleep(cooldown)

        successful = sum(1 for r in all_results if r.success)
        logger.info(f"All runs complete: {successful}/{len(all_results)} successful")
        return all_results
```

Note: the per-cell strategy in repeated mode is reused across trials; `get_next_config` is called with an empty `cell_results` list because trial-level convergence is meaningless when each trial fires once per cell. `AdaptiveStrategy` is incompatible with `repeated` mode — guard it explicitly. Add at the top of `_execute_repeated`:

```python
        if any(getattr(s, "_delegate", None) is not None for s in strategies):
            raise ValueError(
                "parameter_sweep_mode='repeated' is incompatible with "
                "convergence-based stopping (--convergence-metric). Use "
                "'independent' for adaptive sweeps, or remove --convergence-metric."
            )
```

(Adapt the predicate to whatever distinguishes `AdaptiveStrategy` in this codebase — read `_cli_runner_helpers.build_strategy:78-117` to see the branch.)

- [ ] **Step 3: Run order tests to verify pass**

Run: `pytest -n auto tests/unit/orchestrator/`
Expected: PASS (both order tests pass; existing tests still pass).

- [ ] **Step 4: Commit**

```bash
git add src/aiperf/orchestrator/orchestrator.py tests/unit/orchestrator/test_orchestrator.py
git commit --no-verify -s -m "feat(sweep): dispatch MultiRunOrchestrator.execute on parameter_sweep_mode

Splits execute() into _execute_independent (current loop) and
_execute_repeated (trials outer, variations inner). Repeated mode
uses artifact path <base>/profile_runs/trial_NNNN/<variation>/...
matching SweepMode docstring. Adaptive convergence is rejected in
repeated mode (incompatible with stateless trial-outer iteration).
"
```

---

### Task 4: K8s sweep_controller iteration-order swap

**Files:**
- Modify: `src/aiperf/sweep_controller/main.py:177-201` (swap idx derivation by mode)
- Modify: `src/aiperf/sweep_controller/plan_builder.py` (propagate mode from CR spec into plan)
- Modify: `src/aiperf/kubernetes/sweep_models.py` (add `mode: SweepMode` on the `MultiRun` model)

- [ ] **Step 1: Add `mode` to k8s `MultiRun` model**

In `src/aiperf/kubernetes/sweep_models.py`, locate the `MultiRun` (or equivalent CR-side) model that mirrors `MultiRunConfig`. Add:

```python
    mode: SweepMode = Field(
        default=SweepMode.INDEPENDENT,
        description="Iteration order: 'independent' (variations outer) or "
        "'repeated' (trials outer). Mirrors MultiRunConfig.mode.",
    )
```

Add `from aiperf.orchestrator.strategies import SweepMode` to imports.

- [ ] **Step 2: Propagate mode in plan_builder**

In `src/aiperf/sweep_controller/plan_builder.py`, find where `BenchmarkPlan(**plan_kwargs)` is constructed (or its equivalent dict assembly). Add:

```python
    plan_kwargs["parameter_sweep_mode"] = multi_run.mode if multi_run is not None else SweepMode.INDEPENDENT
```

- [ ] **Step 3: Swap idx derivation in main.py**

In `src/aiperf/sweep_controller/main.py:177-201`, replace the `for idx, r in enumerate(results)` block with:

```python
    n_variations = len(plan.variations)
    for idx, r in enumerate(results):
        if plan.parameter_sweep_mode == SweepMode.REPEATED:
            trial_idx = idx // n_variations
            var_idx = min(idx % n_variations, n_variations - 1)
        else:
            var_idx = min(idx // trials_per_variation, n_variations - 1)
            trial_idx = idx % trials_per_variation
        variation = plan.variations[var_idx]
        # ... rest unchanged ...
```

Add `from aiperf.orchestrator.strategies import SweepMode` if absent.

- [ ] **Step 4: Add k8s unit test**

In `tests/unit/kubernetes/test_sweep_models.py`, add:

```python
def test_multirun_mode_round_trips_through_model() -> None:
    from aiperf.kubernetes.sweep_models import MultiRun
    from aiperf.orchestrator.strategies import SweepMode

    m = MultiRun.model_validate({"trials": 3, "mode": "repeated"})
    assert m.mode == SweepMode.REPEATED


def test_multirun_mode_default_is_independent() -> None:
    from aiperf.kubernetes.sweep_models import MultiRun
    from aiperf.orchestrator.strategies import SweepMode

    m = MultiRun.model_validate({"trials": 3})
    assert m.mode == SweepMode.INDEPENDENT
```

Run: `pytest -n auto tests/unit/kubernetes/`
Expected: PASS.

- [ ] **Step 5: Regenerate CRD if templates need it**

If `src/aiperf/kubernetes/sweep_models.py` drives the helm CRD via a generator, run the matching `make` target. Inspect `Makefile` for `crd` / `helm` / `aiperfsweep` targets; otherwise verify by:

```bash
diff <(cat deploy/helm/aiperf-operator/templates/crd-aiperfsweep.yaml) <(git show HEAD:deploy/helm/aiperf-operator/templates/crd-aiperfsweep.yaml)
```

If the CRD needs to mention `mode` under `multiRun`, update it explicitly. Otherwise skip.

- [ ] **Step 6: Commit**

```bash
git add src/aiperf/sweep_controller/ src/aiperf/kubernetes/sweep_models.py tests/unit/kubernetes/test_sweep_models.py deploy/helm/aiperf-operator/templates/crd-aiperfsweep.yaml
git commit --no-verify -s -m "feat(sweep): propagate parameter_sweep_mode through k8s sweep_controller

Adds mode field on the AIPerfSweep CR's MultiRun model, propagates it
through plan_builder into BenchmarkPlan, and swaps the var/trial idx
derivation in main.py's children manifest walk when mode=repeated.
The shared MultiRunOrchestrator already handles iteration order, so
no changes to K8sChildJobExecutor or child-naming are needed.
"
```

---

### Task 5: Replace integration test stub with positive mode coverage

**Files:**
- Modify: `tests/integration/test_parameter_sweep.py:257-304` (delete `test_sweep_repeated_mode_is_rejected`; add two positive tests)
- Modify: `tests/integration/test_parameter_sweep.py:1-22` (update module docstring's "Out of scope" notes)

- [ ] **Step 1: Delete the rejection test**

Remove the entire `test_sweep_repeated_mode_is_rejected` method (lines 257-304).

- [ ] **Step 2: Update module docstring**

In `tests/integration/test_parameter_sweep.py`, lines 16-22, remove the bullet about `--parameter-sweep-mode=repeated` being out of scope (lines 17-19). Keep the operator-mode-gate bullet.

- [ ] **Step 3: Add two new positive tests**

After the last existing test method in `class TestParameterSweep`, add:

```python
    async def test_sweep_independent_mode_writes_variation_outer_layout(
        self,
        cli: AIPerfCLI,
        aiperf_mock_server: AIPerfMockServer,
        temp_output_dir: Path,
    ) -> None:
        """``--parameter-sweep-mode=independent`` (default) groups trials per variation."""
        result = await cli.run(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --url {aiperf_mock_server.url} \
                --endpoint-type chat \
                --concurrency 10,20 \
                --num-profile-runs 2 \
                --parameter-sweep-mode independent \
                --request-count 5 \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """
        )
        assert result.exit_code == 0
        # Variation dir directly under base; profile_runs underneath.
        v0 = temp_output_dir / "phases.profiling.concurrency=10"
        assert (v0 / "profile_runs" / "run_0001").exists()
        assert (v0 / "profile_runs" / "run_0002").exists()
        # No trial_NNNN segment under base.
        assert not (temp_output_dir / "profile_runs" / "trial_0001").exists()


    async def test_sweep_repeated_mode_writes_trial_outer_layout(
        self,
        cli: AIPerfCLI,
        aiperf_mock_server: AIPerfMockServer,
        temp_output_dir: Path,
    ) -> None:
        """``--parameter-sweep-mode=repeated`` groups variations per trial."""
        result = await cli.run(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --url {aiperf_mock_server.url} \
                --endpoint-type chat \
                --concurrency 10,20 \
                --num-profile-runs 2 \
                --parameter-sweep-mode repeated \
                --request-count 5 \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """
        )
        assert result.exit_code == 0
        # trial_NNNN segment under base; variation dirs underneath each trial.
        t1 = temp_output_dir / "profile_runs" / "trial_0001"
        t2 = temp_output_dir / "profile_runs" / "trial_0002"
        assert (t1 / "phases.profiling.concurrency=10").exists()
        assert (t1 / "phases.profiling.concurrency=20").exists()
        assert (t2 / "phases.profiling.concurrency=10").exists()
        assert (t2 / "phases.profiling.concurrency=20").exists()
        # Sweep aggregate still lives at the base (mode-agnostic).
        sweep_json = temp_output_dir / "sweep_aggregate" / "profile_export_aiperf_sweep.json"
        assert sweep_json.exists()
        with sweep_json.open() as f:
            data = json.load(f)
        assert "per_combination_metrics" in data
        assert len(data["per_combination_metrics"]) == 2
```

- [ ] **Step 4: Run integration test for the modified file**

Run: `pytest -n auto -m integration tests/integration/test_parameter_sweep.py`
Expected: PASS (both new tests pass; rest of file unchanged).

- [ ] **Step 5: Commit**

```bash
git add tests/integration/test_parameter_sweep.py
git commit --no-verify -s -m "test(sweep): replace rejection stub with positive mode coverage

Deletes test_sweep_repeated_mode_is_rejected (Path-A no longer
applies). Adds test_sweep_{independent,repeated}_mode_writes_*_layout
asserting the artifact-tree contract from the SweepMode docstring.
Updates module docstring out-of-scope list.
"
```

---

### Task 6: Documentation + four-file sync

**Files:**
- Modify: `docs/cli-options.md` (auto-regen)
- Modify: `docs/kubernetes/sweeps.md` (mode flag + path layouts)
- Modify: `CLAUDE.md`, `AGENTS.md`, `.github/copilot-instructions.md`, `.cursor/rules/python.mdc` (sync)

- [ ] **Step 1: Regenerate CLI docs**

Run: `make generate-cli-docs`
Expected: `docs/cli-options.md` now lists `--parameter-sweep-mode {independent,repeated}` under MULTI_RUN.

- [ ] **Step 2: Update `docs/kubernetes/sweeps.md`**

Add a "Mode: independent vs repeated" section near the top describing:
- Independent (default on this branch): variations outer, trials inner; tree `<base>/<variation>/profile_runs/run_NNNN/`.
- Repeated: trials outer, variations inner; tree `<base>/profile_runs/trial_NNNN/<variation>/profile_runs/run_0001/`.
- Both produce the same total runs and the same `sweep_aggregate/` output.
- Note: this branch defaults to `independent` while main's PR #699 defaults to `repeated`.

- [ ] **Step 3: Update CLAUDE.md `## Parameter Sweeping` section**

Find the bullet that says (or implies) "kept only `independent`-style execution" and replace with a description of the mode dispatch in `MultiRunOrchestrator.execute`. Specifically, in the In-process sweep paragraph, replace any "Path-A" or "REPEATED would require..." language with:

> **Mode dispatch** — `MultiRunOrchestrator.execute` dispatches on `plan.parameter_sweep_mode` (`SweepMode.INDEPENDENT` default, or `REPEATED`). Independent: variations outer / trials inner, paths under `<base>/<variation>/`. Repeated: trials outer / variations inner, paths under `<base>/profile_runs/trial_NNNN/<variation>/`. Both produce the same `sweep_aggregate/` output (aggregation is mode-agnostic). Adaptive convergence is incompatible with repeated.

- [ ] **Step 4: Mirror to other three files**

Apply the identical change to `AGENTS.md`, `.github/copilot-instructions.md`, and `.cursor/rules/python.mdc` (preserve the `alwaysApply: true` frontmatter on the cursor file).

Run: `make check-agent-files-sync`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add docs/cli-options.md docs/kubernetes/sweeps.md CLAUDE.md AGENTS.md .github/copilot-instructions.md .cursor/rules/python.mdc
git commit --no-verify -s -m "docs(sweep): document --parameter-sweep-mode flag and path layouts

Removes Path-A 'REPEATED unsupported' notes from CLAUDE.md and the
three sync files. Adds Mode section to docs/kubernetes/sweeps.md.
Regenerates docs/cli-options.md.
"
```

---

## Verification before declaring done

After all 6 tasks, run the full unit suite once and make sure pre-existing tests still pass. Per `feedback_pytest_single_subfolder`, run the unit subfolder only:

```bash
pytest -n auto tests/unit/
```

Then run the targeted integration test:

```bash
pytest -n auto -m integration tests/integration/test_parameter_sweep.py
```

Both must be green. Report results to the user with concrete test counts.
