# AIPerfSweep Spec Type-Safety Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `AIPerfSweepSpec.template` fully type-safe (no `dict[str, Any]` escapes), so a malformed sweep YAML is rejected at submit time with a precise field error rather than swallowed and mutated downstream.

**Architecture:** Promote `AIPerfJobSpec` from a narrow 8-field validator into a full Pydantic model that mirrors the entire AIPerfJob CRD spec — composing `DeploymentConfig`, adding `benchmark: AIPerfConfig`, and using camelCase aliases so a raw CRD dict validates directly via `model_validate`. Type `AIPerfJobTemplate.metadata` and `AIPerfJobTemplate.spec` so the lazy round-trip in `_validate_axis_combination` becomes implicit.

**Tech Stack:** Python 3.10+, Pydantic v2, kopf, kubernetes_asyncio.

**Branch policy:** Work continues on the current branch (`ajc/k8s`). Do **not** branch off, do **not** open worktrees, do **not** open a PR.

**Test policy:**
- Each task: ONE `uv run pytest -n auto tests/unit/` invocation at the end. No subfolder splits, no parallel pytest invocations.
- Skip `pre-commit run --all-files`. Use `git commit --no-verify` and run `ruff format . && ruff check --fix .` manually before committing.
- Always pass `model="opus"` when dispatching subagents.

---

## File Map

**New files:**
- None. (`AIPerfJobSpec` stays in `src/aiperf/operator/models.py` to avoid blast-radius churn — it can already be imported eagerly from `aiperf.kubernetes.sweep_models` because no operator-only deps live in `operator/models.py`.)

**Modified files:**
- `src/aiperf/operator/models.py` — Restructure `AIPerfJobSpec`; keep `K8sEndpointConfig`/`OwnerReference`/`MetricsSummary`/`PhaseProgress`/`ControllerFetchResult`/`EndpointHealthResult` untouched.
- `src/aiperf/kubernetes/sweep_models.py` — Add `ObjectMetaPartial`; type `AIPerfJobTemplate.metadata` and `.spec`; remove the lazy round-trip from `_validate_axis_combination`.
- `src/aiperf/operator/handlers/create.py` — Replace `AIPerfJobSpec.from_crd_spec(spec)` with `AIPerfJobSpec.model_validate(spec)`. Adjust `get_endpoint_url()` callers if signature shifts.
- `tests/unit/operator/test_models.py` — Update tests for the new model surface.
- `tests/unit/kubernetes/test_sweep_models.py` — Update template-spec validation tests.
- `tests/unit/cli_commands/kube/test_profile.py` — Replace `from_crd_spec` calls with `model_validate`.

---

## Task 1: Restructure `AIPerfJobSpec` to cover the full CRD spec

**Goal:** Replace the narrow 8-field model with a full one composed of `DeploymentConfig` fields + `benchmark: AIPerfConfig` + `skip_endpoint_check`. Use Pydantic camelCase aliases so a raw CRD dict validates directly.

**Files:**
- Modify: `src/aiperf/operator/models.py:261-351`
- Modify: `src/aiperf/operator/handlers/create.py:72,93,97`
- Modify: `tests/unit/operator/test_models.py` (update test fixtures and assertions)
- Modify: `tests/unit/cli_commands/kube/test_profile.py:146-162` (replace `from_crd_spec` with `model_validate`)

### Step 1.1: Read existing structure

Read:
- `src/aiperf/operator/models.py:261-351` (current `AIPerfJobSpec`)
- `src/aiperf/config/deployment.py:159-220` (`DeploymentConfig` field list — image, image_pull_policy, resource_mode, connections_per_worker, timeout_seconds, ttl_seconds_after_finished, results_ttl_days, keep_failed_pods, cancel, pod_template)
- `src/aiperf/config/__init__.py` (confirm `AIPerfConfig` is exported)
- `tools/generate_crd.py:503-554` (canonical CRD spec field list)

- [ ] Confirm `DeploymentConfig` already has `extra="forbid"` and pydantic-friendly defaults.

### Step 1.2: Write the failing test FIRST

Add to `tests/unit/operator/test_models.py` (append to the existing file, near the existing `AIPerfJobSpec` tests):

```python
def test_aiperf_job_spec_validates_full_crd_dict_via_model_validate():
    """A complete CRD spec dict (camelCase, with benchmark) round-trips through model_validate."""
    crd_spec = {
        "image": "nvcr.io/nvidia/aiperf:latest",
        "imagePullPolicy": "IfNotPresent",
        "timeoutSeconds": 600,
        "skipEndpointCheck": True,
        "benchmark": {
            "endpoint": {
                "type": "chat",
                "url": "http://example:8000",
                "model_names": ["test-model"],
            },
        },
    }
    spec = AIPerfJobSpec.model_validate(crd_spec)
    assert spec.image == "nvcr.io/nvidia/aiperf:latest"
    assert spec.skip_endpoint_check is True
    assert spec.timeout_seconds == 600
    assert spec.benchmark.endpoint.url == "http://example:8000"


def test_aiperf_job_spec_rejects_unknown_top_level_keys():
    """Unknown camelCase keys at the spec top level must be rejected."""
    crd_spec = {
        "image": "nvcr.io/nvidia/aiperf:latest",
        "benchmark": {
            "endpoint": {
                "type": "chat",
                "url": "http://example:8000",
                "model_names": ["test-model"],
            },
        },
        "bogusField": "nope",
    }
    with pytest.raises(ValueError, match="bogusField|extra"):
        AIPerfJobSpec.model_validate(crd_spec)


def test_aiperf_job_spec_get_endpoint_url_reads_from_benchmark():
    """get_endpoint_url() reads benchmark.endpoint.url after restructure."""
    spec = AIPerfJobSpec.model_validate(
        {
            "benchmark": {
                "endpoint": {
                    "type": "chat",
                    "url": "http://example:8000",
                    "model_names": ["m"],
                },
            },
        }
    )
    assert spec.get_endpoint_url() == "http://example:8000"
```

### Step 1.3: Run the new tests — confirm they fail

Run: `uv run pytest -n auto tests/unit/operator/test_models.py::test_aiperf_job_spec_validates_full_crd_dict_via_model_validate tests/unit/operator/test_models.py::test_aiperf_job_spec_rejects_unknown_top_level_keys tests/unit/operator/test_models.py::test_aiperf_job_spec_get_endpoint_url_reads_from_benchmark`

Expected: FAIL — current `AIPerfJobSpec.model_validate` would not accept `benchmark` as a typed field, would not reject `bogusField` (it currently accepts arbitrary endpoint dict and silently ignores extras).

### Step 1.4: Implement the new `AIPerfJobSpec`

Replace the body of `class AIPerfJobSpec` in `src/aiperf/operator/models.py` (lines 261-351) with the structure below. Keep `OwnerReference`, `EndpointHealthResult`, `MetricsSummary`, `PhaseProgress`, `ControllerFetchResult`, `K8sEndpointConfig` untouched.

```python
class AIPerfJobSpec(AIPerfBaseModel):
    """Validated AIPerfJob spec mirroring the full CRD spec.

    Composes DeploymentConfig fields (camelCase on the wire via aliases) plus
    `benchmark: AIPerfConfig` plus `skip_endpoint_check`. Validate a raw CRD
    dict via `AIPerfJobSpec.model_validate(spec)` — the previous
    `from_crd_spec` helper is now a thin alias retained for back-compat.
    """

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        alias_generator=lambda f: _camel(f),
    )

    # Deployment fields (mirrored from DeploymentConfig — kept inline rather
    # than inherited so the field order matches the CRD schema and so the
    # operator can evolve the two independently).
    image: str = Field(
        default="nvcr.io/nvidia/aiperf:latest",
        description="Container image for AIPerf",
    )
    image_pull_policy: ImagePullPolicy | None = Field(
        default=None, description="Image pull policy (Always, Never, IfNotPresent)"
    )
    resource_mode: Literal["guaranteed", "burstable", "none"] = Field(
        default="guaranteed",
        description="CPU/memory resource mode for controller and worker pods.",
    )
    connections_per_worker: int = Field(
        default=100,
        ge=1,
        description="Maximum concurrent connections each worker handles.",
    )
    timeout_seconds: float = Field(
        default=0, ge=0, description="Job timeout in seconds (0 = no timeout)"
    )
    ttl_seconds_after_finished: int | None = Field(
        default=300, description="TTL after finished (seconds)"
    )
    results_ttl_days: int | None = Field(
        default=None, ge=1, le=365, description="TTL for results in PVC (days)"
    )
    keep_failed_pods: bool = Field(
        default=False,
        description="Preserve failed JobSet pod attempts for debugging.",
    )
    cancel: bool = Field(default=False, description="Set to true to cancel the job")
    pod_template: PodTemplateConfig = Field(
        default_factory=PodTemplateConfig,
        description="Pod template configuration",
    )

    # Operator-specific
    skip_endpoint_check: bool = Field(
        default=False,
        description="Skip the operator-side endpoint reachability probe before deploying.",
    )

    # Benchmark configuration (the entire AIPerfConfig).
    benchmark: AIPerfConfig = Field(
        ..., description="Benchmark configuration (AIPerfConfig)."
    )

    @field_validator("image")
    @classmethod
    def _validate_image_non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError(
                f"Image is required (got {v!r}); set image.repository and image.tag or pass --image."
            )
        return v

    @classmethod
    def from_crd_spec(cls, spec: dict[str, Any]) -> AIPerfJobSpec:
        """Validate a raw CRD spec dict.

        Retained for back-compat with existing call sites — equivalent to
        ``AIPerfJobSpec.model_validate(spec)`` since aliases handle camelCase.
        """
        return cls.model_validate(spec)

    def get_endpoint_url(self) -> str | None:
        """Extract primary endpoint URL from benchmark.endpoint."""
        endpoint = self.benchmark.endpoint
        if endpoint is None:
            return None
        url = getattr(endpoint, "url", None)
        if url:
            return url
        urls = getattr(endpoint, "urls", None) or []
        return urls[0] if urls else None
```

Add the `_camel` helper near the top of the file (just after the imports):

```python
def _camel(name: str) -> str:
    """snake_case → camelCase for Pydantic field aliases."""
    head, *tail = name.split("_")
    return head + "".join(part.capitalize() for part in tail)
```

Update imports at the top of the file:

```python
from typing import Any, Literal

from pydantic import ConfigDict, Field, field_validator

from aiperf.common.models import AIPerfBaseModel
from aiperf.config import AIPerfConfig
from aiperf.config.deployment import PodTemplateConfig
from aiperf.kubernetes.enums import ImagePullPolicy
from aiperf.kubernetes.k8s_models import K8sCamelModel
```

Remove the now-unused `re`, `model_validator` imports if nothing else needs them (check the rest of the file first; `OwnerReference` and friends still use `K8sCamelModel`, no `re` or `model_validator` references should remain at the top of `AIPerfJobSpec`).

### Step 1.5: Update `src/aiperf/operator/handlers/create.py`

Modify line 72:

```python
# Before
validated_spec = AIPerfJobSpec.from_crd_spec(spec)
# After
validated_spec = AIPerfJobSpec.model_validate(spec)
```

Lines 93 (`get_endpoint_url`) and 97 (`skip_endpoint_check`) need no change — both accessors survive the restructure.

### Step 1.6: Update existing tests in `tests/unit/operator/test_models.py`

Find every `AIPerfJobSpec.from_crd_spec({...})` call (or `AIPerfJobSpec(endpoint={...}, ...)` constructor call) and:
- Replace `from_crd_spec` with `model_validate` (semantically identical now, but new tests should use the canonical entrypoint).
- Replace the `endpoint=...` constructor kwarg with the nested `benchmark={"endpoint": {...}}` shape.

For tests that asserted `spec.endpoint == {...}`, change to `spec.benchmark.endpoint.url == "..."` etc.

### Step 1.7: Update `tests/unit/cli_commands/kube/test_profile.py:146-162`

Replace `AIPerfJobSpec.from_crd_spec(crd_spec)` with `AIPerfJobSpec.model_validate(crd_spec)` (the alias-call still works, but prefer canonical name in new code). Update assertions if any were peeking at `spec.endpoint`.

### Step 1.8: Format and lint

```bash
ruff format src/aiperf/operator/models.py src/aiperf/operator/handlers/create.py tests/unit/operator/test_models.py tests/unit/cli_commands/kube/test_profile.py
ruff check --fix src/aiperf/operator/models.py src/aiperf/operator/handlers/create.py tests/unit/operator/test_models.py tests/unit/cli_commands/kube/test_profile.py
```

### Step 1.9: Run the unit suite

Run: `uv run pytest -n auto tests/unit/`

Expected: all green. If `tests/unit/kubernetes/test_sweep_models.py` fails because `template.spec.benchmark` lacks required fields (e.g. shorthand benchmark dicts that AIPerfConfig now rejects with the field aliases), park those failures for Task 2 — but before parking, confirm each failure is the expected "now-validated" error and not a regression in the operator path.

### Step 1.10: Commit

```bash
git add -u
git commit --no-verify -s -m "$(cat <<'EOF'
refactor(operator): promote AIPerfJobSpec to full CRD-spec coverage

Replace the narrow 8-field validator with a full Pydantic model composing
DeploymentConfig fields + benchmark: AIPerfConfig + skip_endpoint_check.
Camel-case aliases via populate_by_name=True so a raw CRD dict validates
directly through model_validate. from_crd_spec is retained as a thin
alias for back-compat.

This catches malformed AIPerfJob specs (typo'd benchmark keys, wrong-
typed deployment fields) at the operator validation boundary instead of
silently dropping them and erroring downstream.
EOF
)"
```

---

## Task 2: Type `AIPerfJobTemplate.metadata` and `AIPerfJobTemplate.spec`

**Goal:** Replace `dict[str, Any]` on `AIPerfJobTemplate` with a typed `ObjectMetaPartial` and `AIPerfJobSpec`. Drop the lazy round-trip in `AIPerfSweepSpec._validate_axis_combination`.

**Files:**
- Modify: `src/aiperf/kubernetes/sweep_models.py` (add `ObjectMetaPartial`, retype `AIPerfJobTemplate`, simplify validator)
- Modify: `tests/unit/kubernetes/test_sweep_models.py` (update test fixtures)
- Verify: `src/aiperf/sweep_controller/k8s_executor.py:160-181` (still consumes a dict; ensure it dumps the typed spec back to dict via `model_dump(by_alias=True, mode="json")`)
- Verify: `src/aiperf/operator/handlers/sweep/create.py` (any direct dict access to `template.spec` should become `template.spec.model_dump(by_alias=True, mode="json")`)

### Step 2.1: Read consumers

Read:
- `src/aiperf/sweep_controller/k8s_executor.py:140-220` (consumes `template_spec` as a dict; copy.deepcopy on it)
- `src/aiperf/operator/handlers/sweep/create.py` (sweep CR creation handler — confirm template-spec usage)
- `tests/unit/kubernetes/test_sweep_models.py:1-50, 200-260` (test fixture shape)

### Step 2.2: Write the failing test

Append to `tests/unit/kubernetes/test_sweep_models.py`:

```python
def test_aiperf_job_template_metadata_rejects_unknown_keys():
    """ObjectMetaPartial rejects fields outside labels/annotations."""
    with pytest.raises(ValueError, match="extra|name"):
        AIPerfJobTemplate.model_validate(
            {
                "metadata": {"name": "should-not-be-here"},
                "spec": MINIMAL_VALID_TEMPLATE_SPEC,
            }
        )


def test_aiperf_job_template_metadata_typed_labels_and_annotations():
    """Labels and annotations both validate as dict[str, str]."""
    template = AIPerfJobTemplate.model_validate(
        {
            "metadata": {
                "labels": {"team": "perf"},
                "annotations": {"note": "rampA"},
            },
            "spec": MINIMAL_VALID_TEMPLATE_SPEC,
        }
    )
    assert template.metadata.labels == {"team": "perf"}
    assert template.metadata.annotations == {"note": "rampA"}


def test_aiperf_job_template_spec_is_typed_aiperf_job_spec():
    """template.spec is parsed as AIPerfJobSpec, not a raw dict."""
    template = AIPerfJobTemplate.model_validate(
        {"spec": MINIMAL_VALID_TEMPLATE_SPEC}
    )
    from aiperf.operator.models import AIPerfJobSpec  # local import to avoid cycle in test header

    assert isinstance(template.spec, AIPerfJobSpec)
    assert template.spec.skip_endpoint_check is False
```

(`MINIMAL_VALID_TEMPLATE_SPEC` already exists in this test file — see lines ~15-50. Reuse it.)

### Step 2.3: Run the new tests — confirm they fail

Run: `uv run pytest -n auto tests/unit/kubernetes/test_sweep_models.py -k "metadata_rejects_unknown_keys or metadata_typed_labels_and_annotations or template_spec_is_typed_aiperf_job_spec"`

Expected: FAIL — current `AIPerfJobTemplate.metadata` is `dict[str, Any]` (accepts anything) and `.spec` is `dict[str, Any]` (returns dict, not model).

### Step 2.4: Implement `ObjectMetaPartial` and retype `AIPerfJobTemplate`

In `src/aiperf/kubernetes/sweep_models.py`, near the top of the module just before `MultiRunConfig`:

```python
class ObjectMetaPartial(BaseConfig):
    """Subset of Kubernetes ObjectMeta safe to stamp onto child CRs.

    Only labels and annotations are merged into children; name/namespace/uid
    are managed by the controller, so accepting them here would silently lose
    user intent. extra='forbid' surfaces typos like `lables:` at submit time.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    labels: dict[str, str] = Field(
        default_factory=dict,
        description="Labels merged into every child AIPerfJob.",
    )
    annotations: dict[str, str] = Field(
        default_factory=dict,
        description="Annotations merged into every child AIPerfJob.",
    )
```

Add `ObjectMetaPartial` to the module's `__all__` list.

Replace `AIPerfJobTemplate` (lines 115-127) with:

```python
class AIPerfJobTemplate(BaseConfig):
    """Wrapper around an AIPerfJobSpec stamped onto every child."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    metadata: ObjectMetaPartial = Field(
        default_factory=ObjectMetaPartial,
        description="ObjectMeta partial merged into every child (labels, annotations).",
    )
    spec: AIPerfJobSpec = Field(
        ...,
        description="AIPerfJobSpec used as the child stamp. Must not contain sweep:/multi_run:.",
    )
```

Add the eager top-of-file import:

```python
from aiperf.operator.models import AIPerfJobSpec
```

Replace `_validate_axis_combination` lines 165-219 with the simplified version (the `template.spec` round-trip is now implicit; we still enforce the "no sweep keys nested in template.spec" rule):

```python
    @model_validator(mode="after")
    def _validate_axis_combination(self) -> AIPerfSweepSpec:
        # Rule 1: at least one of sweep, multi_run, convergence must be set.
        if self.sweep is None and self.multi_run is None and self.convergence is None:
            raise ValueError(
                "AIPerfSweep requires at least one of `sweep`, `multiRun`, or `convergence`. "
                "For a single benchmark, use AIPerfJob via `aiperf kube profile`."
            )
        # Rule 2: convergence requires multiRun set, with multi_run.trials unset.
        if self.convergence is not None:
            if self.multi_run is None:
                raise ValueError(
                    "`convergence` requires `multiRun` to be set "
                    "(for cooldown/seed/warmup config)."
                )
            if self.multi_run.trials is not None:
                raise ValueError(
                    "`multiRun.trials` must be unset when `convergence` is set; "
                    "convergence.maxRuns governs the per-cell trial cap."
                )
        # Rule 4: sweep-axis keys must not appear inside template.spec or
        # template.spec.benchmark — they belong at AIPerfSweep.spec. The
        # AIPerfJobSpec model has extra='forbid' so unknown keys at top-level
        # of template.spec already raise; we still need to guard the benchmark
        # subtree because AIPerfConfig may accept stray keys via shorthand
        # before-validators.
        forbidden_keys = ("sweep", "multi_run", "multiRun", "convergence")
        benchmark_dump = self.template.spec.benchmark.model_dump(by_alias=False)
        for forbidden in forbidden_keys:
            if forbidden in benchmark_dump:
                raise ValueError(
                    f"`template.spec.benchmark.{forbidden}` is not permitted on AIPerfSweep. "
                    f"Set `spec.{forbidden}` at the top level instead."
                )
        return self
```

Remove the lazy `from aiperf.operator.models import AIPerfJobSpec` and the surrounding `try/except` at the old lines 209-218 — they're no longer needed because the field type now does the validation.

### Step 2.5: Update consumers that read `template.spec` as a dict

`src/aiperf/sweep_controller/k8s_executor.py:160-181` does `copy.deepcopy(self.sweep["spec"]["template"]["spec"])` — this works on the **raw kopf-supplied dict**, not the Pydantic model, so no change needed (the executor never instantiates `AIPerfSweepSpec`; it consumes the dict the apiserver returned).

`src/aiperf/operator/handlers/sweep/create.py` — re-read after Step 2.4 to confirm the same pattern (kopf dict, not Pydantic). If any code path **does** instantiate `AIPerfSweepSpec` and then needs the wire-format dict for `template.spec`, replace with `template.spec.model_dump(by_alias=True, mode="json", exclude_none=True)`.

### Step 2.6: Update existing test fixtures in `test_sweep_models.py`

Find every fixture that builds a sweep dict where `template.metadata` includes anything other than `labels`/`annotations`. Most existing tests should already use the legal subset; if any pass `name`/`namespace`/etc., remove them (they'd be silently ignored before; now they error, which is the intended behavior).

For tests that previously expected the `AIPerfJobSpec.from_crd_spec` ValueError wrapping in Rule 5 (`"is not a valid AIPerfJobSpec"`), update the expected error to match the direct Pydantic ValidationError now raised at field-validation time.

### Step 2.7: Format and lint

```bash
ruff format src/aiperf/kubernetes/sweep_models.py tests/unit/kubernetes/test_sweep_models.py src/aiperf/sweep_controller/k8s_executor.py src/aiperf/operator/handlers/sweep/create.py
ruff check --fix src/aiperf/kubernetes/sweep_models.py tests/unit/kubernetes/test_sweep_models.py src/aiperf/sweep_controller/k8s_executor.py src/aiperf/operator/handlers/sweep/create.py
```

### Step 2.8: Run the unit suite

Run: `uv run pytest -n auto tests/unit/`

Expected: all green. If `tests/integration/test_aiperfsweep_e2e.py` is in the unit selection (it shouldn't be — it's `tests/integration/`), it will not run. Integration tests are out of scope for this branch's verification.

### Step 2.9: Commit

```bash
git add -u
git commit --no-verify -s -m "$(cat <<'EOF'
refactor(sweep): type AIPerfJobTemplate.metadata and .spec

Replace dict[str, Any] on AIPerfJobTemplate with a tiny ObjectMetaPartial
(labels + annotations, extra=forbid) and the now-full AIPerfJobSpec.
The lazy from_crd_spec round-trip in _validate_axis_combination is gone
— field-typed Pydantic validation does the same job at submit time, with
better error paths.
EOF
)"
```

---

## Task 3: Final verification

**Goal:** Confirm both refactors compose cleanly and no caller silently depends on the old dict shape.

### Step 3.1: Type-check and import-walk

Run: `uv run python -c "from aiperf.kubernetes.sweep_models import AIPerfSweepSpec, AIPerfJobTemplate, ObjectMetaPartial; from aiperf.operator.models import AIPerfJobSpec; print('OK')"`

Expected: prints `OK` with no ImportError or circular-import.

### Step 3.2: Validate CRD generation still works

Run: `uv run python tools/generate_crd.py --check`

Expected: zero diff against checked-in CRDs (the CRD wire schema is unchanged — only the in-memory Pydantic models are tightened).

### Step 3.3: Smoke-test the validate command

Run: `uv run aiperf kube validate --help`

Expected: prints CLI help, no import error.

### Step 3.4: Run unit suite one final time

Run: `uv run pytest -n auto tests/unit/`

Expected: all green.

### Step 3.5: Commit any cleanup-only changes (if any)

If steps 3.1-3.3 surfaced anything that needed a tweak (an alias mismatch, a stale doc-string), fix and commit:

```bash
git add -u
git commit --no-verify -s -m "fix(sweep): post-refactor cleanup from verification pass"
```

If nothing needed changing, skip this step.

---

## Out of Scope (do **not** touch in this branch)

- Moving `AIPerfJobSpec` out of `src/aiperf/operator/models.py` — left where it is to keep blast radius small.
- Restructuring `K8sEndpointConfig`, `MetricsSummary`, `PhaseProgress`, `EndpointHealthResult`, `ControllerFetchResult`, `OwnerReference` — out of scope.
- Tightening `template.spec.benchmark.endpoint` beyond what `AIPerfConfig.endpoint` already enforces — already typed via `AIPerfConfig`.
- CRD schema changes to `aiperfjobs.aiperf.nvidia.com` — wire schema unchanged.
- Updating `docs/superpowers/specs/2026-04-25-k8s-sweeps-design.md` — implementation detail, not user-facing.
