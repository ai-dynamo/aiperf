# Keep Failed JobSet Pods Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in `spec.keepFailedPods` debug mode that preserves all failed JobSet pod attempts by disabling worker retries and omitting Job and JobSet TTL cleanup for that run.

**Architecture:** Thread a new top-level CR field through the operator spec-conversion path into `JobSetSpec`, then let JobSet manifest generation switch worker `backoffLimit` and TTL emission only when the flag is enabled. Keep current behavior unchanged for normal runs and cover the change with CRD, conversion, and manifest tests.

**Tech Stack:** Python 3.10+, Pydantic, kopf/operator spec conversion, Kubernetes JobSet manifest generation, pytest, generated Helm CRD

---

## File structure

- Modify: `tools/generate_crd.py` — add generated CRD schema entry for `spec.keepFailedPods`
- Modify: `deploy/helm/aiperf-operator/templates/crd.yaml` — regenerated output only
- Modify: `src/aiperf/operator/spec_converter.py` — carry `keepFailedPods` from CR spec into deployment/jobset inputs
- Modify: `src/aiperf/kubernetes/jobset.py` — add the flag to `JobSetSpec`, switch worker backoff to `0`, and omit TTLs when enabled
- Test: `tests/unit/kubernetes/test_resources.py` and/or `tests/unit/kubernetes/test_client.py` — verify generated manifest behavior
- Test: `tests/unit/operator/...` or existing conversion tests — verify CR/spec conversion path if present

### Task 1: Add CRD schema support for keepFailedPods

**Files:**
- Modify: `tools/generate_crd.py`
- Modify: `deploy/helm/aiperf-operator/templates/crd.yaml`
- Test: generated output via `python -m tools.generate_crd --check`

- [ ] **Step 1: Write the failing schema assertion**

```python
assert "keepFailedPods" in spec_properties
assert spec_properties["keepFailedPods"] == {
    "type": "boolean",
    "description": "Preserve failed JobSet pod attempts for debugging by disabling retries and TTL cleanup.",
    "default": False,
}
```

- [ ] **Step 2: Run the generator check to verify it fails**

Run: `PYTHONUNBUFFERED=1 uv run python tools/generate_crd.py --check`
Expected: FAIL because the generated CRD does not include `keepFailedPods`.

- [ ] **Step 3: Add the new generated spec field**

```python
spec_properties["keepFailedPods"] = {
    "type": "boolean",
    "description": (
        "Preserve failed JobSet pod attempts for debugging by disabling "
        "retries and TTL cleanup."
    ),
    "default": False,
}
```

- [ ] **Step 4: Regenerate the CRD output**

Run: `PYTHONUNBUFFERED=1 uv run python tools/generate_crd.py`
Expected: `deploy/helm/aiperf-operator/templates/crd.yaml` is updated with `keepFailedPods` under `spec`.

- [ ] **Step 5: Run the generator check to verify it passes**

Run: `PYTHONUNBUFFERED=1 uv run python tools/generate_crd.py --check`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add tools/generate_crd.py deploy/helm/aiperf-operator/templates/crd.yaml
git commit -m "feat: add keepFailedPods CRD field"
```

### Task 2: Thread keepFailedPods through spec conversion

**Files:**
- Modify: `src/aiperf/operator/spec_converter.py`
- Test: existing operator/spec conversion test file if present

- [ ] **Step 1: Write the failing conversion test**

```python
def test_converts_keep_failed_pods_flag() -> None:
    raw_spec = {
        "keepFailedPods": True,
        "benchmark": {
            "models": {"items": [{"name": "mock"}]},
            "endpoint": {"urls": ["http://example/v1/chat/completions"]},
            "datasets": {"main": {"type": "synthetic", "prompts": {"isl": {"mean": 32}}}},
            "phases": {"profiling": {"type": "concurrency", "requests": 10, "concurrency": 1}},
        },
    }
    converted = convert_spec_to_run_config(raw_spec)
    assert converted.deployment.keep_failed_pods is True
```

- [ ] **Step 2: Run the targeted test to verify it fails**

Run: `PYTHONUNBUFFERED=1 uv run pytest tests/unit/operator/<conversion-test-file>.py::test_converts_keep_failed_pods_flag -v`
Expected: FAIL because `keepFailedPods` is ignored.

- [ ] **Step 3: Add the minimal conversion plumbing**

```python
keep_failed_pods = bool(raw_spec.get("keepFailedPods", False))
...
deployment_config = DeploymentConfig(
    ...,
    keep_failed_pods=keep_failed_pods,
)
```

If the deployment/jobset input model does not yet have this field, add it there with:

```python
keep_failed_pods: bool = Field(
    default=False,
    description="Preserve failed JobSet pod attempts for debugging.",
)
```

- [ ] **Step 4: Run the targeted test to verify it passes**

Run: `PYTHONUNBUFFERED=1 uv run pytest tests/unit/operator/<conversion-test-file>.py::test_converts_keep_failed_pods_flag -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/aiperf/operator/spec_converter.py src/aiperf/config/deployment.py tests/unit/operator/<conversion-test-file>.py
git commit -m "feat: carry keepFailedPods through spec conversion"
```

### Task 3: Change JobSet generation for debug retention mode

**Files:**
- Modify: `src/aiperf/kubernetes/jobset.py`
- Test: `tests/unit/kubernetes/test_resources.py` or the existing JobSet manifest test file

- [ ] **Step 1: Write the failing manifest test for debug retention mode**

```python
def test_jobset_manifest_keep_failed_pods_disables_worker_retries_and_ttls() -> None:
    spec = JobSetSpec(
        name="job",
        namespace="default",
        job_id="job",
        image="example:latest",
        worker_replicas=2,
        config=mock_run_config,
        keep_failed_pods=True,
    )

    manifest = spec.to_manifest()
    controller_job, worker_job = manifest["spec"]["replicatedJobs"]

    assert controller_job["template"]["spec"]["backoffLimit"] == 0
    assert worker_job["template"]["spec"]["backoffLimit"] == 0
    assert "ttlSecondsAfterFinished" not in worker_job["template"]["spec"]
    assert "ttlSecondsAfterFinished" not in manifest["spec"]
```

- [ ] **Step 2: Write the failing manifest test for default behavior**

```python
def test_jobset_manifest_default_keeps_existing_retry_and_ttl_behavior() -> None:
    spec = JobSetSpec(
        name="job",
        namespace="default",
        job_id="job",
        image="example:latest",
        worker_replicas=2,
        config=mock_run_config,
        keep_failed_pods=False,
    )

    manifest = spec.to_manifest()
    controller_job, worker_job = manifest["spec"]["replicatedJobs"]

    assert controller_job["template"]["spec"]["backoffLimit"] == 0
    assert worker_job["template"]["spec"]["backoffLimit"] == 3
    assert "ttlSecondsAfterFinished" in worker_job["template"]["spec"]
    assert "ttlSecondsAfterFinished" in manifest["spec"]
```

- [ ] **Step 3: Run the targeted manifest tests to verify they fail**

Run: `PYTHONUNBUFFERED=1 uv run pytest tests/unit/kubernetes/<jobset-test-file>.py -k keep_failed_pods -v`
Expected: FAIL because `JobSetSpec` does not yet support the flag.

- [ ] **Step 4: Implement the minimal JobSet changes**

```python
keep_failed_pods: bool = Field(
    default=False,
    description="Preserve failed JobSet pod attempts for debugging.",
)
```

```python
worker_job = ReplicatedJobSpec(
    ...,
    backoff_limit=0 if self.keep_failed_pods else jobset_config.WORKER_BACKOFF_LIMIT,
    job_ttl_seconds=None if self.keep_failed_pods else self.ttl_seconds,
    ...,
)
```

```python
ttl = None
if not self.keep_failed_pods:
    ttl = (
        self.ttl_seconds
        if self.ttl_seconds is not None
        else jobset_config.TTL_SECONDS_AFTER_FINISHED
    )
if ttl is not None:
    manifest["spec"]["ttlSecondsAfterFinished"] = ttl
```

- [ ] **Step 5: Run the targeted manifest tests to verify they pass**

Run: `PYTHONUNBUFFERED=1 uv run pytest tests/unit/kubernetes/<jobset-test-file>.py -k keep_failed_pods -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/aiperf/kubernetes/jobset.py tests/unit/kubernetes/<jobset-test-file>.py
git commit -m "fix: preserve failed jobset pods in debug mode"
```

### Task 4: Run focused verification and pre-commit

**Files:**
- Modify: any touched files from Tasks 1-3
- Test: focused touched-area commands only

- [ ] **Step 1: Run focused unit tests**

Run: `PYTHONUNBUFFERED=1 uv run pytest tests/unit/kubernetes/<jobset-test-file>.py tests/unit/operator/<conversion-test-file>.py -v`
Expected: PASS

- [ ] **Step 2: Re-check CRD generation**

Run: `PYTHONUNBUFFERED=1 uv run python tools/generate_crd.py --check`
Expected: PASS

- [ ] **Step 3: Run pre-commit on touched files**

Run: `pre-commit run --files tools/generate_crd.py deploy/helm/aiperf-operator/templates/crd.yaml src/aiperf/operator/spec_converter.py src/aiperf/config/deployment.py src/aiperf/kubernetes/jobset.py tests/unit/kubernetes/<jobset-test-file>.py tests/unit/operator/<conversion-test-file>.py`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add tools/generate_crd.py deploy/helm/aiperf-operator/templates/crd.yaml src/aiperf/operator/spec_converter.py src/aiperf/config/deployment.py src/aiperf/kubernetes/jobset.py tests/unit/kubernetes/<jobset-test-file>.py tests/unit/operator/<conversion-test-file>.py
git commit -m "feat: add keepFailedPods debug mode"
```
