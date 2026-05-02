# Cluster-Side Adaptive Search (BO under AIPerfSweep) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Allow `--search-*` flags (Bayesian Optimization adaptive outer loop) to run under the Kubernetes operator via `AIPerfSweep` CRDs, not just in-process. v1 of the BO feature (already on `ajc/k8s`) hard-rejects BO under `AIPERF_OPERATOR_MANAGED=1`. This plan lifts that restriction.

**Architecture:** Move the `BayesianSearchPlanner` instantiation from `_run_multi_benchmark` (in-process path) into `sweep_controller/main.py` (cluster path). The orchestrator's `execute_adaptive_search` already iterates one-at-a-time via `executor.execute()`, and the K8s executor (`K8sChildJobExecutor`) already creates one `AIPerfJob` per call — the BO loop is structurally compatible with the existing cluster path. The kopf operator side stays BO-agnostic; the controller pod owns the planner state. Status fields (`totalVariations`, `maxTotalRuns`) become upper bounds (analogous to how `convergence.max_runs` already plays that role); the rollup's `Aggregating` gate stays correct because the controller pod's terminal-phase write already supersedes it via the existing JSON-patch test-op guard.

**Tech Stack:** Python 3.10+, Pydantic v2, `kubernetes_asyncio`, kopf, the existing `AdaptiveSearchConfig` / `BayesianSearchPlanner` from v1.

**Spec:** Built on the investigation in this conversation (Explore agent reading the actual code, not the v1 design doc). Key finding: there is **no data-plane / correctness blocker**. Only the two hard-fail guards I added in v1 prevent it, plus a few status-field cosmetics.

---

## Pre-reads (skim before starting)

- `src/aiperf/sweep_controller/main.py:374-412` — controller pod's existing flow: `build_plan_from_sweep` → `K8sChildJobExecutor` → `orchestrator.execute(...)`. The seam where BO wires in.
- `src/aiperf/cli_runner.py:432-449` — the in-process equivalent: where `BayesianSearchPlanner` is instantiated and passed as `search_planner=` kwarg. Mirror this in `main.py`.
- `src/aiperf/orchestrator/orchestrator.py:235-293` — `execute_adaptive_search`. Already cluster-compatible; reuses `_run_independent_cell` which calls `executor.execute()` per iteration.
- `src/aiperf/sweep_controller/k8s_executor.py:200-231` — `K8sChildJobExecutor.execute`. Already creates one AIPerfJob per call; one-at-a-time is the only mode.
- `src/aiperf/sweep_controller/plan_builder.py:21-40, 59` — the v1 hard reject. Lift this.
- `src/aiperf/cli_runner.py:462-484` — the other v1 hard reject. Lift this.
- `src/aiperf/operator/handlers/sweep/create.py:54-70` — kopf handler that writes `status.totalVariations` and `status.maxTotalRuns`. Special-case adaptive-search.
- `src/aiperf/operator/handlers/sweep/child_rollup.py:130-164` — the `Aggregating` gate. Already permissive when `maxTotalRuns <= 0`; verify and document.
- `src/aiperf/kubernetes/sweep_models.py:52-127` — K8s-side `MultiRunConfig` and `ConvergenceConfig`. We mirror `adaptive_search` here.
- `deploy/helm/aiperf-operator/templates/crd-aiperfsweep.yaml` — auto-generated CRD. Will pick up the new field via `tools/generate_crd.py --check`.

## Out of scope (v2)

- A new `AIPerfAdaptiveSweep` CRD. The existing `AIPerfSweep` CRD is extended; no parallel CRD.
- Resume from a partial `search_history.json` mid-run (BO loop still starts fresh on controller-pod restart; the K8s child-name idempotency layer already covers per-iteration resume — see Task 4 below).
- Multi-objective Pareto BO. Single scalar objective only, same as v1.
- Cross-CR / cross-namespace adaptive search.

---

## Task 1: Mirror `AdaptiveSearchConfig` into K8s-side `MultiRunConfig`

**Files:**
- Modify: `src/aiperf/kubernetes/sweep_models.py:52-127` (add `adaptive_search` field to the K8s `MultiRunConfig`)
- Test: `tests/unit/kubernetes/test_sweep_models.py` (extend or create)

The in-process `MultiRunConfig` (in `src/aiperf/config/_models_benchmark.py`) already has `adaptive_search: Any` (Task 5a of v1). The K8s-side `MultiRunConfig` (used by the `AIPerfSweep` CR spec) is a separate Pydantic class and needs the same field.

- [ ] **Step 1: Write failing test**

```python
# tests/unit/kubernetes/test_sweep_models.py
from aiperf.kubernetes.sweep_models import MultiRunConfig

def test_multi_run_accepts_adaptive_search():
    cfg = MultiRunConfig.model_validate({
        "trials": 3,
        "adaptive_search": {
            "algorithm": "bayes",
            "search_space": [{"path": "phases.profiling.concurrency", "lo": 1, "hi": 1000, "kind": "int"}],
            "objective_metric": "output_token_throughput",
            "objective_stat": "avg",
            "objective_direction": "maximize",
            "max_iterations": 10,
        },
    })
    assert cfg.adaptive_search is not None
    assert cfg.adaptive_search.max_iterations == 10
```

- [ ] **Step 2: Run test, verify failure**

```bash
uv run pytest -n auto tests/unit/kubernetes/test_sweep_models.py -v
```

Expected: ValidationError on unknown `adaptive_search` key.

- [ ] **Step 3: Add field to `MultiRunConfig`**

In `src/aiperf/kubernetes/sweep_models.py`, after the existing fields on `MultiRunConfig`, append:

```python
    adaptive_search: Annotated[
        Any,
        Field(
            default=None,
            description=(
                "Adaptive outer-loop configuration (Bayesian Optimization). "
                "When set, the controller pod drives the sweep adaptively "
                "instead of grid-expanding. Mutually exclusive with the "
                "top-level `sweep` block (the controller hard-fails on both)."
            ),
        ),
    ] = None
```

Use the same `Any` typing trick as the in-process side (avoids circular import on `aiperf.config.adaptive_search`).

Add a `field_validator(mode="before")` that coerces `dict → AdaptiveSearchConfig` (mirroring the in-process side at `src/aiperf/config/_models_benchmark.py`).

- [ ] **Step 4: Re-run test, verify pass**

- [ ] **Step 5: Regenerate CRD**

```bash
uv run python tools/generate_crd.py
git diff deploy/helm/aiperf-operator/templates/crd-aiperfsweep.yaml
```

Expect a new `adaptive_search` property block in the AIPerfSweep CRD schema, with `x-kubernetes-preserve-unknown-fields: true` (since the typed shape on the operator side is `Any`).

- [ ] **Step 6: Commit**

```bash
git add src/aiperf/kubernetes/sweep_models.py tests/unit/kubernetes/test_sweep_models.py deploy/helm/aiperf-operator/templates/crd-aiperfsweep.yaml
git commit -s -m "feat(operator): mirror adaptive_search field into K8s MultiRunConfig + CRD"
```

---

## Task 2: Lift the two v1 hard-rejects

**Files:**
- Modify: `src/aiperf/sweep_controller/plan_builder.py:21-40, 59` (drop `_reject_adaptive_search_in_sweep_cr` call site; keep the helper for now as defensive scaffolding the next task removes)
- Modify: `src/aiperf/cli_runner.py:462-484` (replace the `is_adaptive_search` rejection branch with a permissive pass-through that lets the controller pod handle BO)
- Test: `tests/unit/cli_commands/test_search_reject_under_operator.py` (update — the operator-managed test should now PASS instead of raising)

The v1 rejects exist because we hadn't ported the code. Now we have. Remove them.

- [ ] **Step 1: Update test expectations**

In `tests/unit/cli_commands/test_search_reject_under_operator.py`, flip the operator-managed test:

```python
def test_bo_allowed_under_operator(monkeypatch):
    """v2: adaptive search is allowed under AIPERF_OPERATOR_MANAGED=1."""
    monkeypatch.setenv("AIPERF_OPERATOR_MANAGED", "1")
    # Should NOT raise — controller pod owns the BO loop in cluster mode.
    _reject_in_process_sweep_under_operator(_bo_plan())
```

- [ ] **Step 2: Run test, expect failure (the rejection still fires).**

- [ ] **Step 3: Drop the BO branch from `_reject_in_process_sweep_under_operator`**

In `src/aiperf/cli_runner.py`, remove the `if plan.is_adaptive_search:` branch. Leave the `is_sweep` branch as-is — grid sweeps under the operator still need the AIPerfSweep CRD.

- [ ] **Step 4: Drop `_reject_adaptive_search_in_sweep_cr` from `plan_builder.py`**

In `src/aiperf/sweep_controller/plan_builder.py`, remove the call to `_reject_adaptive_search_in_sweep_cr(spec)`. Keep the helper function definition with a docstring noting it's no longer wired (or delete entirely — preference: delete; defensive code that doesn't fire is dead weight).

- [ ] **Step 5: Re-run test, verify pass**

- [ ] **Step 6: Commit**

```bash
git commit -s -m "feat(operator): allow adaptive search under operator-managed runs"
```

---

## Task 3: Wire `BayesianSearchPlanner` into `sweep_controller/main.py`

**Files:**
- Modify: `src/aiperf/sweep_controller/main.py:374-412` (add planner instantiation parallel to `cli_runner.py:432-436`)
- Test: `tests/component_integration/test_search_cluster_e2e.py` (NEW — uses a stub K8s executor, mirrors `test_search_e2e.py` but exercises the cluster path)

The orchestrator's `execute_adaptive_search` already does the right thing when handed a `search_planner` kwarg — the only missing piece is wiring up the planner in the controller pod.

- [ ] **Step 1: Write the cluster-path E2E test**

Create `tests/component_integration/test_search_cluster_e2e.py`. Stub `K8sChildJobExecutor` to return synthetic results (mirror the in-process `_StubExecutor` from `test_search_e2e.py`). Drive `sweep_controller.main` with a `dict` AIPerfSweep CR carrying `adaptive_search` in `multi_run`. Assert that `len(results) == max_iterations` and `search_history.json` is written.

- [ ] **Step 2: Run test, expect failure**

- [ ] **Step 3: Modify `sweep_controller/main.py`**

In the section around line 392-404 where the orchestrator is constructed, mirror the in-process logic:

```python
search_planner = None
if plan.is_adaptive_search:
    from aiperf.orchestrator.search_planner.bayesian import BayesianSearchPlanner

    search_planner = BayesianSearchPlanner(plan.configs[0], plan.adaptive_search)
    logger.info(
        f"Cluster-side adaptive search active: "
        f"max_iterations={plan.adaptive_search.max_iterations}, "
        f"objective={plan.adaptive_search.objective_metric}:"
        f"{plan.adaptive_search.objective_stat}:{plan.adaptive_search.objective_direction}"
    )

results = await orchestrator.execute(plan, executor, search_planner=search_planner)
```

- [ ] **Step 4: Re-run test, verify pass**

- [ ] **Step 5: Commit**

```bash
git commit -s -m "feat(operator): drive Bayesian search from sweep_controller pod"
```

---

## Task 4: Status-field handling in the kopf create handler

**Files:**
- Modify: `src/aiperf/operator/handlers/sweep/create.py:54-70` (special-case adaptive-search for `totalVariations` / `maxTotalRuns`)
- Test: `tests/unit/operator/test_sweep_create.py` (extend)

For an adaptive sweep, the controller pod doesn't know the final variation count up front — only an upper bound (`max_iterations`). Two options:

**Option A** — write `totalVariations = max_iterations` (upper bound). UI progress bar may under-report on early convergence (cosmetic). Rollup's `Aggregating` gate may fire one tick too early; existing JSON-patch test-op guard at `child_rollup.py:373-443` defers to the controller's terminal write, so this is a transient phase wobble at most.

**Option B** — write `totalVariations = 0` and `maxTotalRuns = 0`. The rollup's permissive check (`isinstance(int) and max_total_runs > 0`) silently disables the gate; completion comes solely from the controller pod's terminal write. UI displays "indeterminate."

**Recommendation:** Option A. The upper bound is more informative for users watching the dashboard, and the test-op guard defends correctness.

- [ ] **Step 1: Write failing test**

```python
def test_create_handler_writes_max_iterations_for_adaptive(...):
    # construct a CR with multi_run.adaptive_search set
    # invoke the handler
    # assert patch.status["totalVariations"] == max_iterations
    # assert patch.status["maxTotalRuns"] == max_iterations * trials
```

- [ ] **Step 2: Run test, expect failure**

- [ ] **Step 3: Modify `create.py`**

After the existing `expand_sweep`-based computation:

```python
# Adaptive search: maxTotalRuns is an upper bound (max_iterations × trials).
# Early convergence routes through controller's terminal-phase write which
# supersedes any premature rollup-driven Aggregating phase via the existing
# _conditional_phase_set test-op guard.
adaptive = getattr(spec.multi_run, "adaptive_search", None) if spec.multi_run else None
if adaptive is not None:
    n_variations = adaptive.max_iterations
    max_total_runs = adaptive.max_iterations * (spec.multi_run.trials or 1)
else:
    expanded = expand_sweep(...)
    n_variations = len(expanded)
    max_total_runs = n_variations * max_trials

patch.status["totalVariations"] = n_variations
patch.status["maxTotalRuns"] = max_total_runs
```

- [ ] **Step 4: Re-run test, verify pass**

- [ ] **Step 5: Commit**

```bash
git commit -s -m "feat(operator): write upper-bound cardinality status fields for adaptive sweeps"
```

---

## Task 5: Make `children.json` manifest writer results-driven

**Files:**
- Modify: `src/aiperf/sweep_controller/main.py:230-268` (the `_write_children_manifest` helper or equivalent)
- Test: extend `test_search_cluster_e2e.py`

The current writer iterates `plan.variations[var_idx]` to derive child names. For adaptive search, `plan.variations` has length 1 (the placeholder); the actual variation set is in `results`. Switch to results-driven enumeration.

- [ ] **Step 1: Test that `children.json` lists exactly `max_iterations × trials` entries for an adaptive run**

- [ ] **Step 2: Run test, verify failure**

- [ ] **Step 3: Refactor**

Replace `for var_idx, variation in enumerate(plan.variations):` style with `for result in results:` style; pull `var_idx` and `trial_index` from `result.variation_values` and `result.trial_index` directly. The `build_child_name(sweep_name, var_idx, trial)` formula is unchanged; only the iteration source changes.

- [ ] **Step 4: Re-run test, verify pass**

- [ ] **Step 5: Commit**

```bash
git commit -s -m "feat(operator): make children.json manifest writer results-driven"
```

---

## Task 6: Documentation

**Files:**
- Modify: `docs/sweeping/bayesian-optimization.md` (drop the v1 "in-process only" caveat)
- Modify: `docs/kubernetes/sweeps.md` (add a section explaining adaptive sweeps)
- Modify: `docs/architecture.md` (the multi-run paragraph mentions cluster path now)
- Modify: AGENTS.md / CLAUDE.md / .github/copilot-instructions.md / .cursor/rules/python.mdc (4-file sync — update the Parameter Sweeping bullet to remove "in-process only" caveat)
- Modify: `llms.txt` (no new file; existing doc reference unchanged)

- [ ] **Step 1: Edit each doc file**

Remove every mention of "in-process only" / "BO is forbidden under the operator" in v1 wording. Replace with: "BO works in-process via `aiperf profile` AND under the operator via `AIPerfSweep` CRDs; the controller pod owns the planner state."

- [ ] **Step 2: Run sync verification**

```bash
make check-agent-files-sync
```

- [ ] **Step 3: Commit**

```bash
git commit -s -m "docs(search): document cluster-side Bayesian Optimization"
```

---

## Task 7: Final integration sweep

- [ ] **Step 1:** `uv run --active pytest -n auto tests/unit/` — pass.
- [ ] **Step 2:** `uv run --active pytest -n auto -m component_integration` — pass.
- [ ] **Step 3:** `pre-commit run --all-files` — pass.
- [ ] **Step 4:** `make check-ergonomics` and `make check-ruff-baselined` — zero new violations.
- [ ] **Step 5 (optional smoke):** apply an `AIPerfSweep` CR with `adaptive_search` to a kind cluster; observe the BO loop running cluster-side. Verify `bo_history.json` appears in the artifact dir and the operator's REST API reports `totalVariations` correctly.

---

## Effort estimate

- **Total LOC:** ~250 (most of the orchestration plumbing already exists in v1).
- **Files touched:** ~10.
- **Complexity:** 3/10 — every seam is already adaptive-shaped from v1; this plan is mostly removing guards and mirroring config fields.
- **Risk:** Low. The single soft constraint (rollup's `Aggregating` gate firing prematurely on early BO convergence) is defended by the existing test-op guard and falls back to the controller pod's terminal-phase write.

## Self-review notes

**What v1 got wrong:** The v1 plan called this a "fundamental incompatibility" with the operator. It's not. The orchestrator's `executor.execute()` seam was already one-at-a-time; the K8s executor already created one AIPerfJob per call; resume already worked via deterministic naming. The only actual blockers were two reject guards I added to defer the work, plus a few status-field cosmetics. Calling it "fundamental" was rhetoric.

**Why this plan is short:** The hard architectural work was done in v1. v2 is the port, not the rewrite.
