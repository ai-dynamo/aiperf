# WorkerGroupManager Clean Replacement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the Kubernetes-only WorkerPodManager direction with a transport-agnostic WorkerGroupManager used in Kubernetes and local mode, removing direct worker/record-processor controller lifecycle dependencies and worker PUB/SUB startup gating.

**Architecture:** Introduce a shared WorkerGroupManager core plus run-mode adapters/protocols for dataset authority, lifecycle transport, and child membership. The controller will see only group managers, workers and record processors will become group-local children, and dataset readiness will become a universal group-local snapshot contract across Kubernetes and local mode with no backward-compatibility layer.

**Tech Stack:** Python 3.10+, Pydantic config models, existing AIPerf lifecycle hooks, ZMQ/IPC transports, Kubernetes JobSet path, local multiprocess path, pytest

---

## File map

- Create: `src/aiperf/workers/worker_group_manager.py` — shared group-manager orchestration core replacing `WorkerPodManager`
- Create: `src/aiperf/workers/group_runtime.py` — run-mode protocol definitions and adapter selection
- Create: `src/aiperf/workers/group_dataset_authority.py` — shared group-local dataset snapshot contracts
- Create: `src/aiperf/workers/group_lifecycle_transport.py` — group-local lifecycle/query transport contracts
- Create: `tests/unit/workers/test_worker_group_manager.py` — shared contract tests for the new group manager
- Modify: `src/aiperf/workers/worker_pod_manager.py` — remove or replace with a thin compatibility-free redirect during implementation, then delete once references are updated
- Modify: `src/aiperf/workers/worker.py` — remove direct controller/event-bus startup dependence; consume group-local snapshots only
- Modify: `src/aiperf/records/record_processor_service.py` — report state group-locally instead of relying on controller-facing assumptions
- Modify: `src/aiperf/controller/system_controller.py` — treat groups as the controller-visible unit, not child services
- Modify: `src/aiperf/common/control_structs.py` — rename/reframe registration payloads from pod-specific to group-specific fields where needed
- Modify: `src/aiperf/common/pod_lifecycle_structs.py` — rename to group-local lifecycle/state structs or replace with transport-agnostic equivalents
- Modify: `src/aiperf/common/enums/enums.py` — rename pod-specific service/message concepts that become group-specific
- Modify: `src/aiperf/common/base_component_service.py` — remove controller-facing assumptions from child services where necessary
- Modify: `src/aiperf/common/mixins/message_bus_mixin.py` — remove worker startup requirement for controller PUB/SUB probes where child services no longer need it
- Modify: `src/aiperf/kubernetes/jobset.py` — launch `WorkerGroupManager` in Kubernetes worker pods and wire group-local env/config
- Modify: `src/aiperf/cli_commands/kube/profile.py` and/or `src/aiperf/cli_commands/kube/generate.py` — ensure generated manifests use the new group-manager world cleanly
- Modify: `src/aiperf/config/models.py` and `src/aiperf/config/config.py` — generalize `workers_per_pod` / `record_processors_per_pod` toward group semantics if needed
- Modify: `tests/unit/workers/test_worker.py` — worker startup tests for group-local dataset and readiness flow
- Modify: `tests/unit/controller/test_system_controller.py` — controller tests for group-only registration/readiness
- Modify: `tests/unit/kubernetes/test_jobset.py` — Kubernetes manifest tests for group manager
- Modify: `docs/architecture.md` — describe WorkerGroupManager as the universal unit-of-capacity
- Modify: `docs/dev/patterns.md` — update service/lifecycle patterns to reflect group-local child services
- Modify: `docs/dev/kubernetes-flow.md` — replace pod-manager-specific startup description with WorkerGroupManager
- Modify: `CLAUDE.md`, `.github/copilot-instructions.md`, `.cursor/rules/python.mdc` — if coding-standard or architecture guidance in these files mentions WorkerPodManager directly

### Task 1: Introduce WorkerGroupManager core and run-mode protocols

**Files:**
- Create: `src/aiperf/workers/worker_group_manager.py`
- Create: `src/aiperf/workers/group_runtime.py`
- Create: `src/aiperf/workers/group_dataset_authority.py`
- Create: `src/aiperf/workers/group_lifecycle_transport.py`
- Test: `tests/unit/workers/test_worker_group_manager.py`

- [ ] **Step 1: Write the failing shared contract tests**

```python
from __future__ import annotations

import pytest

from aiperf.common.enums import WorkerStartupState, WorkerStatus
from aiperf.workers.worker_group_manager import WorkerGroupManager
from aiperf.workers.group_dataset_authority import GroupDatasetSnapshot


def test_group_manager_reports_group_capacity_from_member_provider() -> None:
    manager = WorkerGroupManager(
        runtime_adapter=FakeRuntimeAdapter(num_workers=5, num_record_processors=5)
    )

    registration = manager.build_group_registration()

    assert registration.num_workers == 5
    assert registration.num_record_processors == 5


@pytest.mark.asyncio
async def test_group_manager_gates_dispatchability_on_dataset_ready() -> None:
    manager = WorkerGroupManager(runtime_adapter=FakeRuntimeAdapter())
    await manager.on_child_startup_state(
        service_id="worker-0",
        state=WorkerStartupState.READY,
    )

    assert manager.is_child_dispatchable("worker-0") is False

    await manager.update_dataset_snapshot(
        GroupDatasetSnapshot(dataset_ready=True, benchmark_generation="gen-1")
    )

    assert manager.is_child_dispatchable("worker-0") is True


@pytest.mark.asyncio
async def test_group_manager_aggregates_child_health() -> None:
    manager = WorkerGroupManager(runtime_adapter=FakeRuntimeAdapter())

    await manager.on_child_health(service_id="worker-0", status=WorkerStatus.READY)
    await manager.on_child_health(service_id="worker-1", status=WorkerStatus.DEGRADED)

    summary = manager.current_group_summary()
    assert summary.ready == 1
    assert summary.degraded == 1
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `uv run pytest tests/unit/workers/test_worker_group_manager.py -v`
Expected: FAIL because the new WorkerGroupManager and protocol files do not exist yet.

- [ ] **Step 3: Add the run-mode protocol layer**

```python
# src/aiperf/workers/group_runtime.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class GroupMemberCapacity:
    num_workers: int
    num_record_processors: int


class GroupRuntimeAdapter(Protocol):
    def group_capacity(self) -> GroupMemberCapacity: ...
    async def send_child_command(self, service_id: str, command: str) -> None: ...
    async def publish_group_snapshot(self) -> None: ...
```

```python
# src/aiperf/workers/group_dataset_authority.py
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GroupDatasetSnapshot:
    dataset_ready: bool
    benchmark_generation: str | None
    dataset_generation: str | None = None
    data_file_path: str | None = None
    index_file_path: str | None = None
```

- [ ] **Step 4: Add the minimal WorkerGroupManager core**

```python
# src/aiperf/workers/worker_group_manager.py
from __future__ import annotations

from dataclasses import dataclass, field

from aiperf.workers.group_dataset_authority import GroupDatasetSnapshot
from aiperf.workers.group_runtime import GroupRuntimeAdapter


@dataclass
class ChildState:
    startup_ready: bool = False
    healthy: bool = False
    dispatchable: bool = False


class WorkerGroupManager:
    def __init__(self, runtime_adapter: GroupRuntimeAdapter) -> None:
        self.runtime_adapter = runtime_adapter
        self.dataset_snapshot = GroupDatasetSnapshot(
            dataset_ready=False,
            benchmark_generation=None,
        )
        self.child_states: dict[str, ChildState] = {}

    def build_group_registration(self):
        capacity = self.runtime_adapter.group_capacity()
        return type(
            "GroupRegistration",
            (),
            {
                "num_workers": capacity.num_workers,
                "num_record_processors": capacity.num_record_processors,
            },
        )()
```

- [ ] **Step 5: Run the WorkerGroupManager tests to verify they pass**

Run: `uv run pytest tests/unit/workers/test_worker_group_manager.py -v`
Expected: PASS

- [ ] **Step 6: Commit the new group-manager core**

```bash
git add src/aiperf/workers/worker_group_manager.py src/aiperf/workers/group_runtime.py src/aiperf/workers/group_dataset_authority.py src/aiperf/workers/group_lifecycle_transport.py tests/unit/workers/test_worker_group_manager.py
git commit -m "feat: add WorkerGroupManager core"
```

### Task 2: Move controller topology and registration to group-only semantics

**Files:**
- Modify: `src/aiperf/controller/system_controller.py`
- Modify: `src/aiperf/common/control_structs.py`
- Modify: `tests/unit/controller/test_system_controller.py`

- [ ] **Step 1: Write the failing controller tests**

```python
def test_kubernetes_topology_requires_group_managers_not_child_services() -> None:
    controller = make_system_controller_for_k8s(groups=3)

    required = controller.required_service_types()

    assert "worker_group_manager" in required
    assert "worker" not in required
    assert "record_processor" not in required


def test_group_registration_child_counts_are_capacity_not_controller_expectations() -> None:
    registration = make_group_registration(num_workers=5, num_record_processors=5)
    controller = make_system_controller_for_k8s(groups=1)

    controller.on_group_registered(registration)

    assert controller.declared_group_capacity()[registration.sid].num_workers == 5
```

- [ ] **Step 2: Run the controller tests to verify they fail**

Run: `uv run pytest tests/unit/controller/test_system_controller.py -v`
Expected: FAIL because the controller still expands topology to workers and record processors.

- [ ] **Step 3: Change controller expectations to group-only**

```python
# src/aiperf/controller/system_controller.py
if self._is_group_managed_mode():
    expected_service_types = {
        ServiceType.WORKER_GROUP_MANAGER,
        ServiceType.SYSTEM_CONTROLLER,
        ServiceType.DATASET_MANAGER,
        ServiceType.TIMING_MANAGER,
        ServiceType.RECORDS_MANAGER,
        ServiceType.API,
    }
```

```python
# src/aiperf/common/control_structs.py
class Registration(...):
    group_name: str | None = None
    group_index: str | None = None
    num_workers: int | None = None
    num_record_processors: int | None = None
```

- [ ] **Step 4: Run the controller tests to verify they pass**

Run: `uv run pytest tests/unit/controller/test_system_controller.py -v`
Expected: PASS

- [ ] **Step 5: Commit the controller contract cleanup**

```bash
git add src/aiperf/controller/system_controller.py src/aiperf/common/control_structs.py tests/unit/controller/test_system_controller.py
git commit -m "refactor: make controller group-manager only"
```

### Task 3: Remove worker startup dependence on controller PUB/SUB and move dataset authority group-local

**Files:**
- Modify: `src/aiperf/workers/worker.py`
- Modify: `src/aiperf/common/mixins/message_bus_mixin.py`
- Modify: `tests/unit/workers/test_worker.py`

- [ ] **Step 1: Write the failing worker tests**

```python
@pytest.mark.asyncio
async def test_worker_in_group_managed_mode_does_not_wait_for_event_bus_probe() -> None:
    worker = make_worker(group_managed=True)

    await worker._run_startup_flow()

    assert worker.started_without_global_message_bus_probe is True


@pytest.mark.asyncio
async def test_worker_uses_group_dataset_snapshot_instead_of_dataset_broadcast() -> None:
    worker = make_worker(group_managed=True)

    await worker._on_group_dataset_snapshot(
        dataset_ready_snapshot(data_file_path="/tmp/data.bin")
    )

    assert worker.dataset_client is not None
```

- [ ] **Step 2: Run the worker tests to verify they fail**

Run: `uv run pytest tests/unit/workers/test_worker.py -v`
Expected: FAIL because workers still wait on global PUB/SUB connectivity and dataset rebroadcast handling.

- [ ] **Step 3: Remove worker startup gating on global PUB/SUB in group-managed mode**

```python
# src/aiperf/workers/worker.py
if self._is_group_managed_mode():
    await self._wait_for_group_snapshot()
else:
    await self._run_connection_probes()
```

```python
# src/aiperf/common/mixins/message_bus_mixin.py
async def _run_connection_probes(self) -> None:
    if getattr(self, "skip_global_message_bus_probe", False):
        return
    ...
```

- [ ] **Step 4: Route worker dataset readiness through group-local snapshots only**

```python
# src/aiperf/workers/worker.py
async def _on_group_dataset_snapshot(self, snapshot: GroupDatasetSnapshot) -> None:
    if not snapshot.dataset_ready:
        return
    await self._initialize_dataset_client_from_group_snapshot(snapshot)
```

- [ ] **Step 5: Run the worker tests to verify they pass**

Run: `uv run pytest tests/unit/workers/test_worker.py -v`
Expected: PASS

- [ ] **Step 6: Commit the worker startup simplification**

```bash
git add src/aiperf/workers/worker.py src/aiperf/common/mixins/message_bus_mixin.py tests/unit/workers/test_worker.py
git commit -m "refactor: make workers use group-local startup state"
```

### Task 4: Evolve WorkerPodManager into WorkerGroupManager and wire Kubernetes mode

**Files:**
- Modify: `src/aiperf/workers/worker_pod_manager.py`
- Modify: `src/aiperf/kubernetes/jobset.py`
- Modify: `tests/unit/kubernetes/test_jobset.py`
- Modify: `tests/unit/workers/test_worker_pod_manager.py`

- [ ] **Step 1: Write the failing Kubernetes wiring tests**

```python
def test_jobset_uses_worker_group_manager_service_type() -> None:
    manifest = build_jobset_manifest()

    assert "worker_group_manager" in str(manifest)
    assert "worker_pod_manager" not in str(manifest)


def test_k8s_group_manager_consumes_dataset_notification_and_publishes_group_snapshot() -> None:
    manager = make_k8s_group_manager()
    message = dataset_configured_notification()

    await manager._on_dataset_configured(message)

    assert manager.dataset_snapshot.dataset_ready is True
```

- [ ] **Step 2: Run the Kubernetes tests to verify they fail**

Run: `uv run pytest tests/unit/kubernetes/test_jobset.py tests/unit/workers/test_worker_pod_manager.py -v`
Expected: FAIL because Kubernetes still uses WorkerPodManager naming and pod-local assumptions.

- [ ] **Step 3: Rename and reframe the Kubernetes manager**

```python
# src/aiperf/workers/worker_pod_manager.py
class WorkerGroupManager(BaseComponentService):
    """Group-local orchestrator for child workers and record processors."""
```

```python
# src/aiperf/kubernetes/jobset.py
command = [
    "aiperf",
    "service",
    "--type",
    "worker_group_manager",
    ...,
]
```

- [ ] **Step 4: Run the Kubernetes tests to verify they pass**

Run: `uv run pytest tests/unit/kubernetes/test_jobset.py tests/unit/workers/test_worker_pod_manager.py -v`
Expected: PASS

- [ ] **Step 5: Commit the Kubernetes rename/evolution**

```bash
git add src/aiperf/workers/worker_pod_manager.py src/aiperf/kubernetes/jobset.py tests/unit/kubernetes/test_jobset.py tests/unit/workers/test_worker_pod_manager.py
git commit -m "refactor: evolve WorkerPodManager into WorkerGroupManager"
```

### Task 5: Introduce local-mode WorkerGroupManager adapter and remove local special-casing

**Files:**
- Modify: `src/aiperf/common/subprocess_manager.py`
- Modify: `src/aiperf/config/models.py`
- Modify: `src/aiperf/config/config.py`
- Modify: `tests/unit/common/test_subprocess_manager.py`
- Modify: `tests/unit/controller/conftest.py`

- [ ] **Step 1: Write the failing local-mode tests**

```python
def test_local_mode_starts_group_manager_before_children() -> None:
    runtime = make_local_runtime_config()

    services = build_local_services(runtime)

    assert services[0].service_type == "worker_group_manager"


def test_local_mode_workers_receive_group_capacity_from_adapter() -> None:
    adapter = make_local_group_runtime_adapter(workers=4, record_processors=1)

    assert adapter.group_capacity().num_workers == 4
    assert adapter.group_capacity().num_record_processors == 1
```

- [ ] **Step 2: Run the local-mode tests to verify they fail**

Run: `uv run pytest tests/unit/common/test_subprocess_manager.py -v`
Expected: FAIL because local mode still starts workers/record processors without a group-manager boundary.

- [ ] **Step 3: Add the local adapter and launch local children under WorkerGroupManager**

```python
# src/aiperf/common/subprocess_manager.py
services = [
    make_worker_group_manager_service(runtime_adapter=LocalGroupRuntimeAdapter(...)),
    *make_group_child_services(...),
]
```

- [ ] **Step 4: Run the local-mode tests to verify they pass**

Run: `uv run pytest tests/unit/common/test_subprocess_manager.py -v`
Expected: PASS

- [ ] **Step 5: Commit the local-mode conversion**

```bash
git add src/aiperf/common/subprocess_manager.py src/aiperf/config/models.py src/aiperf/config/config.py tests/unit/common/test_subprocess_manager.py tests/unit/controller/conftest.py
git commit -m "refactor: use WorkerGroupManager in local mode"
```

### Task 6: Update group-local message/state contracts and docs

**Files:**
- Modify: `src/aiperf/common/pod_lifecycle_structs.py`
- Modify: `src/aiperf/common/enums/enums.py`
- Modify: `docs/architecture.md`
- Modify: `docs/dev/patterns.md`
- Modify: `docs/dev/kubernetes-flow.md`
- Modify: `CLAUDE.md`
- Modify: `.github/copilot-instructions.md`
- Modify: `.cursor/rules/python.mdc`

- [ ] **Step 1: Write the failing documentation and message-contract tests**

```python
def test_group_local_struct_names_replace_pod_local_names() -> None:
    assert hasattr(group_lifecycle_structs, "GroupDatasetStateSnapshot")
    assert not hasattr(group_lifecycle_structs, "PodDatasetStateSnapshot")
```

- [ ] **Step 2: Run the contract tests to verify they fail**

Run: `uv run pytest tests/unit/common/messages/test_messages.py -v`
Expected: FAIL because naming and message contracts still use pod-local terminology.

- [ ] **Step 3: Rename the lifecycle/state structs and update docs**

```python
# src/aiperf/common/pod_lifecycle_structs.py
class GroupDatasetStateSnapshot(...):
    ...
```

```md
# docs/architecture.md
WorkerGroupManager is the universal unit of capacity and readiness across Kubernetes and local mode.
```

- [ ] **Step 4: Run the contract tests to verify they pass**

Run: `uv run pytest tests/unit/common/messages/test_messages.py -v`
Expected: PASS

- [ ] **Step 5: Verify the three-file sync rule after doc/rule updates**

Run: `git diff -- CLAUDE.md .github/copilot-instructions.md .cursor/rules/python.mdc`
Expected: Only header/frontmatter differences remain.

- [ ] **Step 6: Commit the naming and docs cleanup**

```bash
git add src/aiperf/common/pod_lifecycle_structs.py src/aiperf/common/enums/enums.py docs/architecture.md docs/dev/patterns.md docs/dev/kubernetes-flow.md CLAUDE.md .github/copilot-instructions.md .cursor/rules/python.mdc tests/unit/common/messages/test_messages.py
git commit -m "docs: describe WorkerGroupManager architecture"
```

### Task 7: Final targeted verification

**Files:**
- Modify: any files touched in Tasks 1-6
- Test: targeted files from Tasks 1-6

- [ ] **Step 1: Run the focused replacement suite**

Run: `uv run pytest tests/unit/workers/test_worker_group_manager.py tests/unit/workers/test_worker.py tests/unit/workers/test_worker_pod_manager.py tests/unit/controller/test_system_controller.py tests/unit/kubernetes/test_jobset.py tests/unit/common/test_subprocess_manager.py tests/unit/common/messages/test_messages.py -v`
Expected: PASS

- [ ] **Step 2: Run formatting and lint fixes**

Run: `ruff format . && ruff check --fix .`
Expected: PASS with no remaining formatting/lint issues in touched files

- [ ] **Step 3: Run pre-commit on changed files**

Run: `pre-commit run --files src/aiperf/workers/worker_group_manager.py src/aiperf/workers/group_runtime.py src/aiperf/workers/group_dataset_authority.py src/aiperf/workers/group_lifecycle_transport.py src/aiperf/workers/worker_pod_manager.py src/aiperf/workers/worker.py src/aiperf/controller/system_controller.py src/aiperf/common/control_structs.py src/aiperf/common/pod_lifecycle_structs.py src/aiperf/common/enums/enums.py src/aiperf/common/base_component_service.py src/aiperf/common/mixins/message_bus_mixin.py src/aiperf/kubernetes/jobset.py src/aiperf/common/subprocess_manager.py src/aiperf/config/models.py src/aiperf/config/config.py tests/unit/workers/test_worker_group_manager.py tests/unit/workers/test_worker.py tests/unit/workers/test_worker_pod_manager.py tests/unit/controller/test_system_controller.py tests/unit/kubernetes/test_jobset.py tests/unit/common/test_subprocess_manager.py tests/unit/common/messages/test_messages.py docs/architecture.md docs/dev/patterns.md docs/dev/kubernetes-flow.md CLAUDE.md .github/copilot-instructions.md .cursor/rules/python.mdc`
Expected: PASS

- [ ] **Step 4: Inspect the final diff**

Run: `git diff -- src/aiperf/workers/worker_group_manager.py src/aiperf/workers/group_runtime.py src/aiperf/workers/group_dataset_authority.py src/aiperf/workers/group_lifecycle_transport.py src/aiperf/workers/worker_pod_manager.py src/aiperf/workers/worker.py src/aiperf/controller/system_controller.py src/aiperf/common/control_structs.py src/aiperf/common/pod_lifecycle_structs.py src/aiperf/common/enums/enums.py src/aiperf/common/base_component_service.py src/aiperf/common/mixins/message_bus_mixin.py src/aiperf/kubernetes/jobset.py src/aiperf/common/subprocess_manager.py src/aiperf/config/models.py src/aiperf/config/config.py tests/unit/workers/test_worker_group_manager.py tests/unit/workers/test_worker.py tests/unit/workers/test_worker_pod_manager.py tests/unit/controller/test_system_controller.py tests/unit/kubernetes/test_jobset.py tests/unit/common/test_subprocess_manager.py tests/unit/common/messages/test_messages.py docs/architecture.md docs/dev/patterns.md docs/dev/kubernetes-flow.md CLAUDE.md .github/copilot-instructions.md .cursor/rules/python.mdc`
Expected: Only the WorkerGroupManager clean replacement changes appear

- [ ] **Step 5: Create the final implementation commit**

```bash
git add src/aiperf/workers/worker_group_manager.py src/aiperf/workers/group_runtime.py src/aiperf/workers/group_dataset_authority.py src/aiperf/workers/group_lifecycle_transport.py src/aiperf/workers/worker_pod_manager.py src/aiperf/workers/worker.py src/aiperf/controller/system_controller.py src/aiperf/common/control_structs.py src/aiperf/common/pod_lifecycle_structs.py src/aiperf/common/enums/enums.py src/aiperf/common/base_component_service.py src/aiperf/common/mixins/message_bus_mixin.py src/aiperf/kubernetes/jobset.py src/aiperf/common/subprocess_manager.py src/aiperf/config/models.py src/aiperf/config/config.py tests/unit/workers/test_worker_group_manager.py tests/unit/workers/test_worker.py tests/unit/workers/test_worker_pod_manager.py tests/unit/controller/test_system_controller.py tests/unit/kubernetes/test_jobset.py tests/unit/common/test_subprocess_manager.py tests/unit/common/messages/test_messages.py docs/architecture.md docs/dev/patterns.md docs/dev/kubernetes-flow.md CLAUDE.md .github/copilot-instructions.md .cursor/rules/python.mdc
git commit -m "refactor: replace WorkerPodManager with WorkerGroupManager"
```
