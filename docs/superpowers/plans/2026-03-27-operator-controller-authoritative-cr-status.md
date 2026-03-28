# Operator Controller-Authoritative CR Status Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Kubernetes operator report `AIPerfJob` status from controller truth so the CR moves to `Running` as soon as profiling starts and `status.workers` reflects controller aggregate pod-local state.

**Architecture:** Extend the controller `/api/progress` response to include aggregate worker-pod status, teach the operator `ProgressClient` and `StatusBuilder` to consume and write that richer snapshot, and simplify `monitor_progress()` so controller progress is authoritative whenever available. Keep JobSet-derived readiness only for bootstrap and controller-unavailable fallback.

**Tech Stack:** Python 3.10+, FastAPI, Pydantic, kopf, kr8s, pytest, Helm CRD YAML

---

## File structure

- Modify: `src/aiperf/api/routers/progress.py` — add controller-authored aggregate worker status to `/api/progress`
- Modify: `src/aiperf/controller/system_controller.py` — add helper(s) that summarize `_pod_states` into aggregate controller status
- Modify: `src/aiperf/operator/progress_client.py` — parse enriched progress response and drop worker-startup-state-driven status logic
- Modify: `src/aiperf/operator/status.py` — support writing aggregate `status.workers`
- Modify: `src/aiperf/operator/handlers/monitor.py` — make controller progress authoritative for phase and worker status
- Modify: `deploy/helm/aiperf-operator/templates/crd.yaml` — expand `status.workers` schema for aggregate fields
- Modify: `tests/unit/api/routers/test_progress.py` — verify `/api/progress` includes aggregate worker status
- Modify: `tests/unit/operator/test_progress_client.py` — verify enriched progress parsing
- Modify: `tests/unit/operator/test_status.py` — verify aggregate worker status setter behavior
- Modify: `tests/unit/operator/test_main.py` — verify monitor transitions to `Running` from controller progress even with partial readiness
- Modify: `docs/architecture.md` — document controller-authoritative Kubernetes operator status semantics

### Task 1: Extend controller progress API with aggregate worker status

**Files:**
- Modify: `src/aiperf/controller/system_controller.py`
- Modify: `src/aiperf/api/routers/progress.py`
- Test: `tests/unit/api/routers/test_progress.py`

- [ ] **Step 1: Write the failing API test**

```python
from aiperf.common.messages import WorkerPodStateMessage


def test_progress_includes_aggregate_worker_status(
    progress_client: TestClient,
    progress_router: ProgressRouter,
) -> None:
    progress_router._pod_states = {
        "0": WorkerPodStateMessage(
            service_id="worker-pod-manager-0",
            pod_index="0",
            benchmark_generation="bench-1",
            dataset_generation="data-1",
            declared_workers=2,
            declared_record_processors=1,
            router_connected_workers=2,
            dispatchable_workers=1,
            ready_workers=1,
            ready_record_processors=1,
            degraded_workers=1,
            degraded_record_processors=0,
            pod_state="ready",
            admission_state="dispatchable",
        )
    }
    response = progress_client.get("/api/progress")
    data = response.json()
    assert data["workers"] == {
        "ready": 1,
        "total": 2,
        "dispatchable": 1,
        "router_connected": 2,
        "ready_record_processors": 1,
        "declared_record_processors": 1,
        "ready_pods": 1,
        "total_pods": 1,
        "degraded_pods": 0,
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/api/routers/test_progress.py::TestProgressEndpoint::test_progress_includes_aggregate_worker_status -v`
Expected: FAIL because `ProgressResponse` has no `workers` field.

- [ ] **Step 3: Add controller aggregate model and API response field**

```python
class AggregateWorkerStatus(AIPerfBaseModel):
    ready: int = Field(default=0, description="Dispatch-ready worker count.")
    total: int = Field(default=0, description="Declared worker count.")
    dispatchable: int = Field(default=0, description="Workers eligible to receive credits.")
    router_connected: int = Field(default=0, description="Workers connected to the credit router.")
    ready_record_processors: int = Field(
        default=0,
        description="Record processors currently available across worker pods.",
    )
    declared_record_processors: int = Field(
        default=0,
        description="Declared record-processor count across worker pods.",
    )
    ready_pods: int = Field(default=0, description="Pods with usable worker capacity.")
    total_pods: int = Field(default=0, description="Total worker pods seen by the controller.")
    degraded_pods: int = Field(default=0, description="Pods that are usable but degraded.")


class ProgressResponse(AIPerfBaseModel):
    phases: dict[CreditPhase, CombinedPhaseStats] = Field(
        default_factory=dict,
        description="Per-phase progress stats",
    )
    workers: AggregateWorkerStatus = Field(
        default_factory=AggregateWorkerStatus,
        description="Controller-authored aggregate worker-pod status.",
    )
```

```python
def _build_aggregate_worker_status(self) -> AggregateWorkerStatus:
    pods = list(self._pod_states.values())
    return AggregateWorkerStatus(
        ready=sum(pod.ready_workers for pod in pods),
        total=sum(pod.declared_workers for pod in pods),
        dispatchable=sum(pod.dispatchable_workers for pod in pods),
        router_connected=sum(pod.router_connected_workers for pod in pods),
        ready_record_processors=sum(pod.ready_record_processors for pod in pods),
        declared_record_processors=sum(
            pod.declared_record_processors for pod in pods
        ),
        ready_pods=sum(
            1
            for pod in pods
            if pod.dispatchable_workers >= 1 and pod.ready_record_processors >= 1
        ),
        total_pods=len(pods),
        degraded_pods=sum(
            1
            for pod in pods
            if pod.dispatchable_workers >= 1
            and pod.ready_record_processors >= 1
            and (
                pod.degraded_workers > 0 or pod.degraded_record_processors > 0
            )
        ),
    )
```

```python
@progress_router.get("/api/progress", response_model=ProgressResponse, tags=["API"])
async def get_progress(component: ProgressDep) -> ProgressResponse:
    return ProgressResponse(
        phases=component._progress_tracker._phases,
        workers=component._build_aggregate_worker_status(),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/api/routers/test_progress.py::TestProgressEndpoint::test_progress_includes_aggregate_worker_status -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/unit/api/routers/test_progress.py src/aiperf/api/routers/progress.py src/aiperf/controller/system_controller.py
git commit -m "feat: expose controller aggregate worker status"
```

### Task 2: Parse controller aggregate status in the operator client and status builder

**Files:**
- Modify: `src/aiperf/operator/progress_client.py`
- Modify: `src/aiperf/operator/status.py`
- Test: `tests/unit/operator/test_progress_client.py`
- Test: `tests/unit/operator/test_status.py`

- [ ] **Step 1: Write the failing progress-client test**

```python
def test_parse_with_worker_aggregate_status(
    progress_api_response_running: dict[str, Any],
) -> None:
    client = ProgressClient()
    progress = client._parse_progress_response(
        {
            **progress_api_response_running,
            "workers": {
                "ready": 4,
                "total": 8,
                "dispatchable": 3,
                "router_connected": 6,
                "ready_record_processors": 2,
                "declared_record_processors": 4,
                "ready_pods": 2,
                "total_pods": 4,
                "degraded_pods": 1,
            },
        }
    )
    assert progress.workers.dispatchable == 3
    assert progress.workers.total_pods == 4
```

- [ ] **Step 2: Write the failing status-builder test**

```python
def test_set_worker_aggregate_status(self) -> None:
    mock_patch = MagicMock()
    mock_patch.status = {}
    builder = StatusBuilder(mock_patch)

    builder.set_worker_aggregate_status(
        {
            "ready": 4,
            "total": 8,
            "dispatchable": 3,
            "routerConnected": 6,
            "readyRecordProcessors": 2,
            "declaredRecordProcessors": 4,
            "readyPods": 2,
            "totalPods": 4,
            "degradedPods": 1,
        }
    )

    assert mock_patch.status["workers"]["dispatchable"] == 3
    assert mock_patch.status["workers"]["readyPods"] == 2
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/unit/operator/test_progress_client.py::TestProgressClientParseResponse::test_parse_with_worker_aggregate_status tests/unit/operator/test_status.py::TestStatusBuilder::test_set_worker_aggregate_status -v`
Expected: FAIL because `JobProgress` has no `workers` field and `StatusBuilder` has no aggregate setter.

- [ ] **Step 4: Add aggregate worker status parsing and writing**

```python
class AggregateWorkerStatus(AIPerfBaseModel):
    ready: int = Field(default=0, description="Dispatch-ready worker count.")
    total: int = Field(default=0, description="Declared worker count.")
    dispatchable: int = Field(default=0, description="Workers eligible to receive credits.")
    router_connected: int = Field(default=0, description="Workers connected to the router.")
    ready_record_processors: int = Field(default=0, description="Ready record processors.")
    declared_record_processors: int = Field(default=0, description="Declared record processors.")
    ready_pods: int = Field(default=0, description="Usable worker pods.")
    total_pods: int = Field(default=0, description="Observed worker pods.")
    degraded_pods: int = Field(default=0, description="Usable but degraded worker pods.")


class JobProgress(AIPerfBaseModel):
    phases: dict[CreditPhase, CombinedPhaseStats] = Field(
        default_factory=dict,
        description="Progress stats for each benchmark phase",
    )
    workers: AggregateWorkerStatus = Field(
        default_factory=AggregateWorkerStatus,
        description="Controller-authored aggregate worker status.",
    )
    error: str | None = Field(default=None, description="Error message if job failed")
    connection_error: str | None = Field(
        default=None,
        description="Connection error if progress API was unreachable",
    )
```

```python
return JobProgress(
    phases=phases,
    workers=AggregateWorkerStatus(**data.get("workers", {})),
    error=data.get("error"),
)
```

```python
def set_worker_aggregate_status(
    self,
    workers: dict[str, int],
) -> StatusBuilder:
    self._patch.status["workers"] = workers
    return self


def set_workers(self, ready: int, total: int) -> StatusBuilder:
    return self.set_worker_aggregate_status({"ready": ready, "total": total})
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/unit/operator/test_progress_client.py::TestProgressClientParseResponse::test_parse_with_worker_aggregate_status tests/unit/operator/test_status.py::TestStatusBuilder::test_set_worker_aggregate_status -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add tests/unit/operator/test_progress_client.py tests/unit/operator/test_status.py src/aiperf/operator/progress_client.py src/aiperf/operator/status.py
git commit -m "refactor: parse controller aggregate worker status"
```

### Task 3: Make monitor_progress controller-authoritative for phase and workers

**Files:**
- Modify: `src/aiperf/operator/handlers/monitor.py`
- Test: `tests/unit/operator/test_main.py`

- [ ] **Step 1: Write the failing monitor test for profiling start**

```python
@pytest.mark.asyncio
async def test_monitor_progress_uses_controller_phase_when_profiling_started() -> None:
    from aiperf.operator.handlers.monitor import monitor_progress

    kopf_patch = MagicMock()
    kopf_patch.status = {}

    mock_jobset = MagicMock()
    mock_jobset.raw = {
        "status": {
            "replicatedJobsStatus": [
                {"name": "workers", "ready": 1, "active": 1, "succeeded": 0, "failed": 0, "suspended": 0}
            ],
            "conditions": [],
        }
    }

    mock_progress = JobProgress(
        phases={
            "profiling": CombinedPhaseStats(
                phase="profiling",
                total_expected_requests=10,
                requests_completed=1,
                start_ns=1,
                last_update_ns=2,
            )
        },
        workers=AggregateWorkerStatus(
            ready=1,
            total=2,
            dispatchable=1,
            router_connected=2,
            ready_record_processors=1,
            declared_record_processors=1,
            ready_pods=1,
            total_pods=2,
            degraded_pods=0,
        ),
    )

    with (
        mock_patch("aiperf.operator.handlers.monitor.AsyncJobSet.get", new=AsyncMock(return_value=mock_jobset)),
        mock_patch("aiperf.operator.handlers.monitor.get_api", new=AsyncMock(return_value=MagicMock())),
        mock_patch("aiperf.operator.handlers.monitor.get_or_create_progress_client", new=AsyncMock(return_value=mock_client := AsyncMock())),
        mock_patch("aiperf.operator.handlers.monitor._check_pod_restarts", new=AsyncMock()),
        mock_patch("aiperf.operator.handlers.monitor._maybe_recover_terminated_controller", new=AsyncMock(return_value=False)),
    ):
        mock_client.get_progress.return_value = mock_progress
        mock_client.get_metrics.return_value = {}
        mock_client.get_server_metrics.return_value = {}

        await monitor_progress(
            body={},
            status={"phase": Phase.INITIALIZING, "jobSetName": "test-jobset", "jobId": "job-1"},
            spec={},
            name="job",
            namespace="default",
            patch=kopf_patch,
        )

    assert kopf_patch.status["phase"] == Phase.RUNNING
    assert kopf_patch.status["currentPhase"] == "profiling"
    assert kopf_patch.status["workers"]["dispatchable"] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/operator/test_main.py::TestMonitorProgress::test_monitor_progress_uses_controller_phase_when_profiling_started -v`
Expected: FAIL because `monitor_progress()` still gates `Running` on all workers ready and does not write aggregate worker fields.

- [ ] **Step 3: Refactor monitor logic to use controller progress first**

```python
async def _apply_controller_progress_status(
    patch: kopf.Patch,
    sb: StatusBuilder,
    progress: JobProgress,
) -> None:
    if progress.current_phase is not None:
        patch.status["currentPhase"] = progress.current_phase
        if str(progress.current_phase) == "profiling":
            sb.set_phase(Phase.RUNNING)
            sb.conditions.set_true(
                ConditionType.BENCHMARK_RUNNING,
                "BenchmarkStarted",
                "Benchmark is running",
            )

    sb.set_worker_aggregate_status(
        {
            "ready": progress.workers.ready,
            "total": progress.workers.total,
            "dispatchable": progress.workers.dispatchable,
            "routerConnected": progress.workers.router_connected,
            "readyRecordProcessors": progress.workers.ready_record_processors,
            "declaredRecordProcessors": progress.workers.declared_record_processors,
            "readyPods": progress.workers.ready_pods,
            "totalPods": progress.workers.total_pods,
            "degradedPods": progress.workers.degraded_pods,
        }
    )
```

```python
progress = await client.get_progress(controller_host)
if not progress.connection_error:
    await _apply_controller_progress_status(patch, sb, progress)

# Delete the all-workers-ready gate that does this:
# if app_workers_ready == total_workers:
#     sb.set_phase(Phase.RUNNING)
```

- [ ] **Step 4: Run targeted tests to verify they pass**

Run: `uv run pytest tests/unit/operator/test_main.py::TestMonitorProgress::test_monitor_progress_uses_controller_phase_when_profiling_started tests/unit/operator/test_main.py::TestFetchProgress::test_updates_status_with_progress -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/unit/operator/test_main.py src/aiperf/operator/handlers/monitor.py
git commit -m "fix: derive operator phase from controller progress"
```

### Task 4: Expand CRD schema and document the new semantics

**Files:**
- Modify: `deploy/helm/aiperf-operator/templates/crd.yaml`
- Modify: `docs/architecture.md`
- Test: `tests/unit/operator/test_status.py`

- [ ] **Step 1: Write the failing schema/status test**

```python
def test_set_workers_preserves_simple_summary_and_aggregate_fields(self) -> None:
    mock_patch = MagicMock()
    mock_patch.status = {}
    builder = StatusBuilder(mock_patch)

    builder.set_worker_aggregate_status(
        {
            "ready": 4,
            "total": 8,
            "dispatchable": 3,
            "routerConnected": 6,
            "readyRecordProcessors": 2,
            "declaredRecordProcessors": 4,
            "readyPods": 2,
            "totalPods": 4,
            "degradedPods": 1,
        }
    )

    assert mock_patch.status["workers"] == {
        "ready": 4,
        "total": 8,
        "dispatchable": 3,
        "routerConnected": 6,
        "readyRecordProcessors": 2,
        "declaredRecordProcessors": 4,
        "readyPods": 2,
        "totalPods": 4,
        "degradedPods": 1,
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/operator/test_status.py::TestStatusBuilder::test_set_workers_preserves_simple_summary_and_aggregate_fields -v`
Expected: FAIL until the richer worker payload is fully supported.

- [ ] **Step 3: Expand the CRD schema and docs**

```yaml
workers:
  type: object
  properties:
    ready:
      type: integer
      description: Controller-authoritative count of ready workers.
    total:
      type: integer
      description: Controller-authoritative count of declared workers.
    dispatchable:
      type: integer
      description: Workers eligible to receive credits.
    routerConnected:
      type: integer
      description: Workers connected to the credit router.
    readyRecordProcessors:
      type: integer
      description: Ready record processors across worker pods.
    declaredRecordProcessors:
      type: integer
      description: Declared record processors across worker pods.
    readyPods:
      type: integer
      description: Worker pods with usable capacity.
    totalPods:
      type: integer
      description: Total worker pods seen by the controller.
    degradedPods:
      type: integer
      description: Worker pods that are usable but degraded.
```

```markdown
- In Kubernetes mode, the operator now mirrors controller truth for job lifecycle.
- `status.phase=Running` means the controller has started profiling.
- `status.workers` reflects controller aggregate worker-pod readiness and dispatchability, not raw JobSet pod readiness.
```

- [ ] **Step 4: Run targeted verification**

Run: `uv run pytest tests/unit/operator/test_status.py::TestStatusBuilder::test_set_workers_preserves_simple_summary_and_aggregate_fields tests/unit/api/routers/test_progress.py::TestProgressEndpoint::test_progress_includes_aggregate_worker_status -v`
Expected: PASS

- [ ] **Step 5: Run focused formatting and unit tests**

Run: `uv run pytest tests/unit/api/routers/test_progress.py tests/unit/operator/test_progress_client.py tests/unit/operator/test_status.py tests/unit/operator/test_main.py -v`
Expected: PASS

- [ ] **Step 6: Run pre-commit on changed files**

Run: `pre-commit run --files src/aiperf/api/routers/progress.py src/aiperf/controller/system_controller.py src/aiperf/operator/progress_client.py src/aiperf/operator/status.py src/aiperf/operator/handlers/monitor.py deploy/helm/aiperf-operator/templates/crd.yaml tests/unit/api/routers/test_progress.py tests/unit/operator/test_progress_client.py tests/unit/operator/test_status.py tests/unit/operator/test_main.py docs/architecture.md`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add deploy/helm/aiperf-operator/templates/crd.yaml docs/architecture.md tests/unit/operator/test_status.py
git commit -m "docs: align operator status schema with controller truth"
```
