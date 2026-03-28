# Kubernetes WorkerPodManager Controller Isolation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `WorkerPodManager` the only controller-facing service in Kubernetes mode, while allowing workers and whole worker pods to churn safely during active benchmarks.

**Architecture:** Replace worker/record-processor controller connectivity with pod-local state/query coordination owned by `WorkerPodManager`. Split router connectivity from dispatch eligibility, move Kubernetes benchmark start decisions to pod-level aggregate readiness, and replace rebroadcast-oriented dataset handoff with queryable current-state snapshots at both the pod-local and controller/API layers.

**Tech Stack:** Python 3.10+, asyncio, msgspec, Pydantic, ZeroMQ DEALER/ROUTER, FastAPI, pytest, aiohttp

---

## File map

### Core protocol and message files
- Modify: `src/aiperf/credit/messages.py` — replace `WorkerReady` with explicit routing-state messages.
- Modify: `src/aiperf/common/pod_lifecycle_structs.py` — add pod-local query/snapshot messages for workers and record processors.
- Modify: `src/aiperf/common/enums/enums.py` — add/replace command and message enums for pod snapshots and controller snapshot endpoints if needed.
- Modify: `src/aiperf/common/messages/worker_messages.py` — add pod-centric aggregate state message for Kubernetes mode, stop relying on worker-centric summary.

### Router and worker runtime
- Modify: `src/aiperf/credit/sticky_router.py` — track connected workers separately from dispatchable workers.
- Modify: `src/aiperf/workers/worker.py` — perform early router handshake, then become dispatchable only after pod-local query-driven convergence.
- Modify: `src/aiperf/records/record_processor_service.py` — remove controller-facing assumptions in Kubernetes mode, use pod-local lifecycle/query state.
- Modify: `src/aiperf/workers/worker_pod_manager.py` — become the authoritative pod-local state service and sole controller-facing authority.

### Controller and API
- Modify: `src/aiperf/controller/system_controller.py` — use pod-manager-only registration in Kubernetes mode, pod-level benchmark admission/start policy, and aggregate pod snapshots.
- Modify: `src/aiperf/controller/kubernetes_service_manager.py` — keep pod health checks, expose pod-state helpers used by the new start policy.
- Modify: `src/aiperf/api/routers/dataset.py` — add dataset-state snapshot endpoint(s).
- Modify: `src/aiperf/api/dataset_mixin.py` — store dataset metadata/state needed by snapshot endpoints.
- Modify: `src/aiperf/dataset/dataset_manager.py` — publish generation/versioned dataset state and stop relying on Kubernetes rebroadcast behavior.

### Tests
- Modify: `tests/unit/credit/test_sticky_router.py`
- Modify: `tests/unit/workers/test_worker.py`
- Modify: `tests/unit/workers/test_worker_pod_manager.py`
- Modify: `tests/unit/controller/test_system_controller.py`
- Add or modify: `tests/unit/api/test_dataset_router.py`
- Add or modify: `tests/kubernetes/test_benchmark.py`
- Add or modify: `tests/kubernetes/test_scaling.py`

### Docs
- Modify: `docs/architecture.md`
- Modify: `docs/dev/patterns.md`

---

### Task 1: Redefine router worker-state protocol

**Files:**
- Modify: `src/aiperf/credit/messages.py`
- Modify: `tests/unit/credit/test_sticky_router.py`

- [ ] **Step 1: Write the failing router protocol tests**

```python
from aiperf.credit.messages import (
    WorkerConnected,
    WorkerDispatchable,
    WorkerUndispatchable,
    WorkerShutdown,
)


async def test_connected_worker_is_not_routable(run) -> None:
    router = StickyCreditRouter(run=run, service_id="router")
    router._credit_router_client.send_to = AsyncMock()

    await router._handle_return_router_message(
        "worker-1", WorkerConnected(worker_id="worker-1")
    )

    credit = make_credit(corr_id="corr-1")
    with pytest.raises(RuntimeError, match="No dispatchable workers available"):
        await router.send_credit(credit)


async def test_dispatchable_worker_enters_routing_pool(run) -> None:
    router = StickyCreditRouter(run=run, service_id="router")
    router._credit_router_client.send_to = AsyncMock()

    await router._handle_return_router_message(
        "worker-1", WorkerConnected(worker_id="worker-1")
    )
    await router._handle_return_router_message(
        "worker-1", WorkerDispatchable(worker_id="worker-1")
    )

    await router.send_credit(make_credit(corr_id="corr-2"))
    router._credit_router_client.send_to.assert_called_once()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/credit/test_sticky_router.py -k "dispatchable or connected" -v`
Expected: FAIL with import errors for `WorkerConnected` / `WorkerDispatchable` and/or router behavior mismatch.

- [ ] **Step 3: Replace the protocol in `src/aiperf/credit/messages.py`**

```python
class WorkerConnected(Struct, frozen=True, kw_only=True, tag_field="t", tag="wc"):
    worker_id: str


class WorkerDispatchable(
    Struct, frozen=True, kw_only=True, tag_field="t", tag="wd"
):
    worker_id: str


class WorkerUndispatchable(
    Struct, frozen=True, kw_only=True, tag_field="t", tag="wu"
):
    worker_id: str
    reason: str | None = None


WorkerToRouterMessage: TypeAlias = (
    WorkerConnected
    | WorkerDispatchable
    | WorkerUndispatchable
    | WorkerShutdown
    | CreditReturn
    | FirstToken
    | TimePing
    | InFlightReport
)
```

- [ ] **Step 4: Update the router tests to import the new messages and stop using `WorkerReady`**

```python
from aiperf.credit.messages import (
    CreditReturn,
    FirstToken,
    TimePing,
    WorkerConnected,
    WorkerDispatchable,
    WorkerUndispatchable,
    WorkerShutdown,
)
```

- [ ] **Step 5: Run tests to verify the protocol layer passes**

Run: `uv run pytest tests/unit/credit/test_sticky_router.py -k "dispatchable or connected" -v`
Expected: PASS for new protocol imports; router behavior will still fail until Task 2 is complete.

- [ ] **Step 6: Commit**

```bash
git add tests/unit/credit/test_sticky_router.py src/aiperf/credit/messages.py
git commit -m "refactor: split router connectivity from dispatchability"
```

### Task 2: Refactor `StickyCreditRouter` to route only dispatchable workers

**Files:**
- Modify: `src/aiperf/credit/sticky_router.py`
- Test: `tests/unit/credit/test_sticky_router.py`

- [ ] **Step 1: Write the failing dispatch-pool tests**

```python
async def test_undispatchable_worker_leaves_routing_pool(run) -> None:
    router = StickyCreditRouter(run=run, service_id="router")
    router._credit_router_client.send_to = AsyncMock()

    await router._handle_return_router_message(
        "worker-1", WorkerConnected(worker_id="worker-1")
    )
    await router._handle_return_router_message(
        "worker-1", WorkerDispatchable(worker_id="worker-1")
    )
    await router._handle_return_router_message(
        "worker-1", WorkerUndispatchable(worker_id="worker-1")
    )

    with pytest.raises(RuntimeError, match="No dispatchable workers available"):
        await router.send_credit(make_credit(corr_id="corr-3"))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/credit/test_sticky_router.py -k "undispatchable or dispatchable" -v`
Expected: FAIL because `_workers` is still the single source of truth.

- [ ] **Step 3: Split connected vs dispatchable state in `src/aiperf/credit/sticky_router.py`**

```python
self._connected_workers: set[str] = set()
self._workers_cache: list[WorkerLoad] = []
self._workers: dict[str, WorkerLoad] = {}
self._workers_by_load: dict[int, set[str]] = defaultdict(set)


def _mark_worker_connected(self, worker_id: str) -> None:
    self._connected_workers.add(worker_id)
    self._initializing_workers.discard(worker_id)


def _mark_worker_dispatchable(self, worker_id: str) -> None:
    self._mark_worker_connected(worker_id)
    if worker_id not in self._workers:
        self._register_worker(worker_id)


def _mark_worker_undispatchable(self, worker_id: str) -> WorkerLoad | None:
    self._connected_workers.add(worker_id)
    return self._unregister_worker(worker_id)
```

- [ ] **Step 4: Update return-channel message handling to use the new state transitions**

```python
case TimePing():
    self._initializing_workers.add(worker_id)
    await self._credit_router_client.send_to(
        worker_id,
        TimePong(sequence=message.sequence, sent_at_ns=message.sent_at_ns),
    )
case WorkerConnected():
    self._mark_worker_connected(worker_id)
case WorkerDispatchable():
    self._mark_worker_dispatchable(worker_id)
case WorkerUndispatchable():
    detached = self._mark_worker_undispatchable(worker_id)
    self._detach_worker(worker_id, detached)
case WorkerShutdown():
    self._connected_workers.discard(worker_id)
```

- [ ] **Step 5: Tighten the routing error message and hot path**

```python
if not self._workers:
    raise RuntimeError("No dispatchable workers available for routing")
```

- [ ] **Step 6: Run the router tests**

Run: `uv run pytest tests/unit/credit/test_sticky_router.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add src/aiperf/credit/sticky_router.py tests/unit/credit/test_sticky_router.py
git commit -m "refactor: route credits only to dispatchable workers"
```

### Task 3: Make workers converge through pod-local query state before becoming dispatchable

**Files:**
- Modify: `src/aiperf/common/pod_lifecycle_structs.py`
- Modify: `src/aiperf/workers/worker.py`
- Test: `tests/unit/workers/test_worker.py`

- [ ] **Step 1: Write the failing worker churn tests**

```python
async def test_k8s_worker_connects_router_before_dataset_ready(k8s_worker: Worker) -> None:
    k8s_worker.return_dealer_client.send = AsyncMock()
    k8s_worker._query_pod_dataset_state = AsyncMock(return_value=None)

    await k8s_worker._send_worker_ready_message()

    k8s_worker.return_dealer_client.send.assert_awaited_once_with(
        WorkerConnected(worker_id="k8s-worker")
    )
    assert not k8s_worker._worker_ready_event.is_set()


async def test_k8s_worker_becomes_dispatchable_after_dataset_query(k8s_worker: Worker) -> None:
    k8s_worker.return_dealer_client.send = AsyncMock()
    k8s_worker._query_pod_dataset_state = AsyncMock(
        return_value=PodDatasetStateSnapshot(
            service_id="pod-manager",
            benchmark_generation="gen-1",
            dataset_generation="data-1",
            ready=True,
            data_file_path="/aiperf/datasets/dataset.dat",
            index_file_path="/aiperf/datasets/index.dat",
            conversation_count=4,
            total_size_bytes=1024,
        )
    )

    await k8s_worker._complete_k8s_startup_flow()

    assert any(
        isinstance(call.args[0], WorkerDispatchable)
        for call in k8s_worker.return_dealer_client.send.await_args_list
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/workers/test_worker.py -k "dispatchable or dataset_query" -v`
Expected: FAIL because the worker still waits on `PodDatasetReady` push semantics and sends `WorkerReady`.

- [ ] **Step 3: Add pod-local query/snapshot structs**

```python
class PodDatasetStateQuery(Struct, frozen=True, kw_only=True, tag_field="t", tag="dq"):
    service_id: str


class PodDatasetStateSnapshot(
    Struct, frozen=True, kw_only=True, tag_field="t", tag="ds"
):
    service_id: str
    benchmark_generation: str
    dataset_generation: str
    ready: bool
    data_file_path: str | None = None
    index_file_path: str | None = None
    conversation_count: int = 0
    total_size_bytes: int = 0
    error_message: str | None = None
```

- [ ] **Step 4: Replace K8s worker startup in `src/aiperf/workers/worker.py`**

```python
@on_start
async def _send_worker_ready_message(self) -> None:
    await self._publish_startup_state(WorkerStartupState.STARTING)
    if self._is_kubernetes_mode():
        await self._publish_startup_state(WorkerStartupState.ROUTER_PROBING)
        await self._measure_baseline_rtt()
        await self.return_dealer_client.send(WorkerConnected(worker_id=self.service_id))
        await self._publish_startup_state(WorkerStartupState.WAITING_FOR_DATASET)
        await self._complete_k8s_startup_flow()
        return
```

- [ ] **Step 5: Add a query-driven convergence helper in `src/aiperf/workers/worker.py`**

```python
async def _complete_k8s_startup_flow(self) -> None:
    snapshot = await self._query_pod_dataset_state()
    if snapshot is None or not snapshot.ready:
        return

    await self._initialize_dataset_client(
        MemoryMapClientMetadata(
            data_file_path=Path(snapshot.data_file_path),
            index_file_path=Path(snapshot.index_file_path),
            conversation_count=snapshot.conversation_count,
            total_size_bytes=snapshot.total_size_bytes,
        )
    )
    await self.return_dealer_client.send(
        WorkerDispatchable(worker_id=self.service_id)
    )
    await self._publish_startup_state(WorkerStartupState.READY)
    self._worker_ready_event.set()
```

- [ ] **Step 6: Make shutdown and drain revoke eligibility first**

```python
if self._is_kubernetes_mode():
    await self.return_dealer_client.send(
        WorkerUndispatchable(worker_id=self.service_id, reason="shutdown")
    )
```

- [ ] **Step 7: Run worker tests**

Run: `uv run pytest tests/unit/workers/test_worker.py -k "k8s or dispatchable or dataset_query" -v`
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add src/aiperf/common/pod_lifecycle_structs.py src/aiperf/workers/worker.py tests/unit/workers/test_worker.py
git commit -m "refactor: make k8s workers converge before dispatchability"
```

### Task 4: Turn `WorkerPodManager` into the authoritative pod-local state service

**Files:**
- Modify: `src/aiperf/workers/worker_pod_manager.py`
- Modify: `src/aiperf/records/record_processor_service.py`
- Test: `tests/unit/workers/test_worker_pod_manager.py`

- [ ] **Step 1: Write the failing pod-manager query tests**

```python
async def test_worker_pod_manager_answers_dataset_queries_from_current_state(
    worker_pod_manager: WorkerPodManager,
) -> None:
    worker_pod_manager._benchmark_generation = "gen-1"
    worker_pod_manager._dataset_generation = "data-1"
    worker_pod_manager._dataset_downloaded = True
    worker_pod_manager._dataset_client_metadata = MemoryMapClientMetadata(
        data_file_path=Path("/tmp/dataset.dat"),
        index_file_path=Path("/tmp/index.dat"),
        conversation_count=3,
        total_size_bytes=128,
    )

    response = await worker_pod_manager._on_pod_lifecycle_message(
        "identity-1",
        PodDatasetStateQuery(service_id="worker-1"),
    )

    assert response.ready is True
    assert response.dataset_generation == "data-1"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/workers/test_worker_pod_manager.py -k "dataset_queries or current_state" -v`
Expected: FAIL because pod queries are not implemented.

- [ ] **Step 3: Add pod-manager snapshot state fields**

```python
self._benchmark_generation: str | None = None
self._dataset_generation: str | None = None
self._pod_admission_state: str = "admitting"
self._controller_snapshot: BenchmarkStateSnapshot | None = None
```

- [ ] **Step 4: Answer pod-local dataset/state queries directly from current truth**

```python
case PodDatasetStateQuery():
    return PodDatasetStateSnapshot(
        service_id=self.service_id,
        benchmark_generation=self._benchmark_generation or "",
        dataset_generation=self._dataset_generation or "",
        ready=self._dataset_downloaded and self._dataset_client_metadata is not None,
        data_file_path=str(self._dataset_client_metadata.data_file_path)
        if self._dataset_client_metadata
        else None,
        index_file_path=str(self._dataset_client_metadata.index_file_path)
        if self._dataset_client_metadata
        else None,
        conversation_count=self._dataset_client_metadata.conversation_count
        if self._dataset_client_metadata
        else 0,
        total_size_bytes=self._dataset_client_metadata.total_size_bytes
        if self._dataset_client_metadata
        else 0,
    )
```

- [ ] **Step 5: Stop using rebroadcast semantics for late workers**

```python
if self._dataset_downloaded:
    self.debug("Dataset already downloaded; workers should query current state")
    return
```

- [ ] **Step 6: Remove direct controller assumptions from record processors**

```python
@on_start
async def _register_with_worker_pod_manager(self) -> None:
    if self.pod_lifecycle_dealer_client is None:
        return
    await self.pod_lifecycle_dealer_client.send(
        PodPeerHello(
            service_id=self.service_id,
            service_type=str(self.service_type),
            pod_index=self._pod_index,
        )
    )
```

- [ ] **Step 7: Run pod-manager and record-processor unit tests**

Run: `uv run pytest tests/unit/workers/test_worker_pod_manager.py tests/unit/workers/test_worker.py -k "pod_manager or dataset_queries" -v`
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add src/aiperf/workers/worker_pod_manager.py src/aiperf/records/record_processor_service.py tests/unit/workers/test_worker_pod_manager.py tests/unit/workers/test_worker.py
git commit -m "refactor: make worker pod manager the pod-local state authority"
```

### Task 5: Add controller/API benchmark snapshot and dataset generation state

**Files:**
- Modify: `src/aiperf/api/routers/dataset.py`
- Modify: `src/aiperf/api/dataset_mixin.py`
- Modify: `src/aiperf/dataset/dataset_manager.py`
- Test: `tests/unit/api/test_dataset_router.py`

- [ ] **Step 1: Write the failing dataset snapshot API tests**

```python
async def test_dataset_snapshot_returns_generation_and_api_urls(client) -> None:
    response = await client.get("/api/dataset/state")
    assert response.status_code == 200
    body = response.json()
    assert body["dataset_generation"] == "data-1"
    assert body["ready"] is True
    assert body["data_url"].endswith("/api/dataset/data")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/api/test_dataset_router.py -k "dataset_snapshot" -v`
Expected: FAIL with 404 for `/api/dataset/state`.

- [ ] **Step 3: Store versioned dataset state in the API mixin/router**

```python
self._dataset_generation: str | None = None
self._benchmark_generation: str | None = None

@on_message(MessageType.DATASET_CONFIGURED_NOTIFICATION)
async def _on_dataset_configured(self, message: DatasetConfiguredNotification) -> None:
    self._dataset_client_metadata = message.client_metadata
    self._dataset_generation = message.dataset_generation
    self._benchmark_generation = message.benchmark_generation
    self._dataset_configured.set()
```

- [ ] **Step 4: Extend `DatasetConfiguredNotification` production in `dataset_manager.py`**

```python
notification = DatasetConfiguredNotification(
    service_id=self.service_id,
    metadata=self.dataset_metadata,
    client_metadata=client_metadata,
    benchmark_generation=self.run.cfg.artifacts.benchmark_id,
    dataset_generation=f"{self.run.cfg.artifacts.benchmark_id}:dataset",
)
```

- [ ] **Step 5: Add a state endpoint in `src/aiperf/api/routers/dataset.py`**

```python
@dataset_router.get("/api/dataset/state")
async def get_dataset_state(component: DatasetDep) -> dict[str, object]:
    await _wait_for_dataset_metadata(component)
    metadata = component.dataset_client_metadata
    return {
        "ready": True,
        "benchmark_generation": component.benchmark_generation,
        "dataset_generation": component.dataset_generation,
        "conversation_count": metadata.conversation_count,
        "total_size_bytes": metadata.total_size_bytes,
        "data_url": "/api/dataset/data",
        "index_url": "/api/dataset/index",
    }
```

- [ ] **Step 6: Run dataset API tests**

Run: `uv run pytest tests/unit/api/test_dataset_router.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add src/aiperf/api/routers/dataset.py src/aiperf/api/dataset_mixin.py src/aiperf/dataset/dataset_manager.py tests/unit/api/test_dataset_router.py
git commit -m "feat: add versioned dataset snapshot api"
```

### Task 6: Move Kubernetes benchmark-start policy to pod-level readiness thresholds

**Files:**
- Modify: `src/aiperf/common/messages/worker_messages.py`
- Modify: `src/aiperf/controller/system_controller.py`
- Modify: `src/aiperf/workers/worker_pod_manager.py`
- Test: `tests/unit/controller/test_system_controller.py`

- [ ] **Step 1: Write the failing controller start-policy tests**

```python
async def test_k8s_controller_starts_after_grace_period_with_enough_ready_pods(
    system_controller: SystemController,
) -> None:
    system_controller._pod_start_grace_period = 5.0
    system_controller._pod_states = {
        "0": WorkerPodStateSnapshot(
            service_id="pod-0",
            pod_index="0",
            dispatchable_workers=2,
            ready_record_processors=1,
            pod_state="ready",
            admission_state="dispatchable",
            benchmark_generation="gen-1",
            dataset_generation="data-1",
        )
    }

    assert system_controller._has_sufficient_ready_worker_pods() is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/controller/test_system_controller.py -k "grace_period or ready_worker_pods" -v`
Expected: FAIL because the controller still waits on per-worker `READY` state.

- [ ] **Step 3: Replace worker-centric summary with a pod snapshot message**

```python
class WorkerPodStateMessage(BaseServiceMessage):
    message_type: MessageTypeT = MessageType.WORKER_POD_STATE
    pod_index: str = Field(..., description="Worker pod index")
    benchmark_generation: str = Field(..., description="Loaded benchmark generation")
    dataset_generation: str = Field(..., description="Loaded dataset generation")
    declared_workers: int = Field(..., description="Configured workers in this pod")
    declared_record_processors: int = Field(..., description="Configured record processors in this pod")
    router_connected_workers: int = Field(..., description="Workers connected to router")
    dispatchable_workers: int = Field(..., description="Workers eligible for credits")
    ready_record_processors: int = Field(..., description="Record processors ready for work")
    pod_state: str = Field(..., description="Aggregate pod state")
    admission_state: str = Field(..., description="Pod admission state")
```

- [ ] **Step 4: Publish pod snapshots from `WorkerPodManager` instead of worker-centric summaries**

```python
snapshot = WorkerPodStateMessage(
    service_id=self.service_id,
    pod_index=self._pod_index or "",
    benchmark_generation=self._benchmark_generation or "",
    dataset_generation=self._dataset_generation or "",
    declared_workers=self.workers_per_pod,
    declared_record_processors=self.record_processors_per_pod,
    router_connected_workers=self._router_connected_workers(),
    dispatchable_workers=self._dispatchable_workers(),
    ready_record_processors=self._ready_record_processors(),
    pod_state=self._current_pod_state(),
    admission_state=self._pod_admission_state,
)
await self.publish(snapshot)
```

- [ ] **Step 5: Replace `_wait_for_all_workers_ready()` in `system_controller.py` with pod-threshold gating**

```python
async def _wait_for_sufficient_worker_pods(self, timeout: float) -> None:
    begin = time.perf_counter()
    while True:
        if self._has_sufficient_ready_worker_pods():
            return
        if time.perf_counter() - begin >= self._pod_start_grace_period:
            if self._has_minimum_start_capacity():
                return
        if time.perf_counter() - begin >= timeout:
            raise ServiceRegistrationTimeoutError(
                "Timed out waiting for sufficient worker pod readiness",
                missing={},
            )
        await asyncio.sleep(1.0)
```

- [ ] **Step 6: Use the new gate in the post-configure startup flow**

```python
self.info("Post-configure startup flow: waiting for sufficient worker pod readiness")
async with self.try_operation_or_stop("Wait For Worker Pods Ready"):
    await self._wait_for_sufficient_worker_pods(
        timeout=Environment.SERVICE.PROFILE_START_TIMEOUT,
    )
```

- [ ] **Step 7: Run controller and pod-manager tests**

Run: `uv run pytest tests/unit/controller/test_system_controller.py tests/unit/workers/test_worker_pod_manager.py -v`
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add src/aiperf/common/messages/worker_messages.py src/aiperf/controller/system_controller.py src/aiperf/workers/worker_pod_manager.py tests/unit/controller/test_system_controller.py tests/unit/workers/test_worker_pod_manager.py
git commit -m "refactor: start k8s benchmarks from pod readiness thresholds"
```

### Task 7: Remove Kubernetes child-service controller registration and finish churn coverage

**Files:**
- Modify: `src/aiperf/controller/system_controller.py`
- Modify: `src/aiperf/workers/worker.py`
- Modify: `src/aiperf/records/record_processor_service.py`
- Test: `tests/kubernetes/test_benchmark.py`
- Test: `tests/kubernetes/test_scaling.py`

- [ ] **Step 1: Write the failing integration-level churn tests**

```python
async def test_late_worker_pod_join_does_not_receive_credits_until_dispatchable(
    system_controller: SystemController,
) -> None:
    system_controller._pod_states = {
        "0": WorkerPodStateSnapshot(
            service_id="pod-0",
            pod_index="0",
            benchmark_generation="gen-1",
            dataset_generation="data-1",
            declared_workers=4,
            declared_record_processors=1,
            router_connected_workers=4,
            dispatchable_workers=0,
            ready_record_processors=1,
            pod_state="starting",
            admission_state="admitting",
        )
    }

    assert system_controller._has_sufficient_ready_worker_pods() is False


def test_controller_does_not_expect_worker_registration_in_k8s_mode(
    k8s_system_controller: SystemController,
) -> None:
    assert ServiceType.WORKER not in k8s_system_controller.required_services
    assert ServiceType.RECORD_PROCESSOR not in k8s_system_controller.required_services
    assert ServiceType.WORKER_POD_MANAGER in k8s_system_controller.required_services
```

- [ ] **Step 2: Run the targeted tests to verify they fail**

Run: `uv run pytest tests/kubernetes/test_benchmark.py -k "late_worker_pod_join" -v`
Expected: FAIL because Kubernetes mode still expects child service registration and worker-centric readiness.

- [ ] **Step 3: Remove Kubernetes child registration expectations in `system_controller.py`**

```python
if is_k8s_mode:
    self._k8s_topology = self._build_k8s_service_topology()
    self.required_services[ServiceType.WORKER_POD_MANAGER] = (
        self._k8s_topology.num_worker_pods
    )
```

- [ ] **Step 4: Ensure worker and record-processor shutdown paths are pod-local only in K8s mode**

```python
if self._is_kubernetes_mode():
    await self.pod_lifecycle_dealer_client.send(
        PodPeerShutdown(
            service_id=self.service_id,
            service_type=str(self.service_type),
        )
    )
    return
```

- [ ] **Step 5: Run Kubernetes-focused tests**

Run: `uv run pytest tests/kubernetes/test_benchmark.py tests/kubernetes/test_scaling.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/aiperf/controller/system_controller.py src/aiperf/workers/worker.py src/aiperf/records/record_processor_service.py tests/kubernetes/test_benchmark.py tests/kubernetes/test_scaling.py
git commit -m "refactor: remove k8s child controller registration"
```

### Task 8: Update architecture docs and run repo verification

**Files:**
- Modify: `docs/architecture.md`
- Modify: `docs/dev/patterns.md`

- [ ] **Step 1: Update architecture documentation**

```markdown
- In Kubernetes mode, only `WorkerPodManager` connects to `SystemController`.
- Workers connect to the credit router before dataset availability but become dispatchable only after query-driven convergence.
- Worker and whole-pod churn are handled through queryable current-state snapshots.
```

- [ ] **Step 2: Update development patterns documentation**

```markdown
- Prefer state/query contracts over rebroadcast-only startup notifications for churn-safe Kubernetes services.
- Router presence and dispatchability are separate concepts.
```

- [ ] **Step 3: Run formatting, lint, and targeted tests**

Run: `ruff format . && ruff check --fix .`
Expected: PASS with formatting/lint fixes applied if needed.

- [ ] **Step 4: Run the focused Python test suite**

Run: `uv run pytest tests/unit/credit/test_sticky_router.py tests/unit/workers/test_worker.py tests/unit/workers/test_worker_pod_manager.py tests/unit/controller/test_system_controller.py tests/unit/api/test_dataset_router.py -n auto`
Expected: PASS

- [ ] **Step 5: Run Kubernetes-focused verification**

Run: `uv run pytest tests/kubernetes/test_benchmark.py tests/kubernetes/test_scaling.py -v`
Expected: PASS

- [ ] **Step 6: Run pre-commit**

Run: `pre-commit run --all-files`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add docs/architecture.md docs/dev/patterns.md
git commit -m "docs: describe k8s pod-manager readiness model"
```

## Self-review

### Spec coverage
- Controller-visible topology becomes pod-only: Tasks 6 and 7.
- Router connected vs dispatchable split: Tasks 1 and 2.
- Worker churn via query-driven convergence: Tasks 3 and 4.
- Whole-pod churn via controller/API current-state snapshots: Tasks 4, 5, and 6.
- Controller start policy based on enough ready pods after grace period: Task 6.
- Partial pod capacity tolerated: Tasks 4 and 6.
- No compatibility layer / delete old paths: Tasks 1, 2, 4, 6, and 7.
- Docs updates: Task 8.

### Placeholder scan
- No `TODO` / `TBD` placeholders remain.
- Each code-changing task includes concrete code blocks.
- Each verification step names an exact command and expected result.

### Type consistency
- Router messages consistently use `WorkerConnected`, `WorkerDispatchable`, `WorkerUndispatchable`, `WorkerShutdown`.
- Pod-local query path consistently uses `PodDatasetStateQuery` and `PodDatasetStateSnapshot`.
- Controller aggregate state consistently uses `WorkerPodStateMessage`.
