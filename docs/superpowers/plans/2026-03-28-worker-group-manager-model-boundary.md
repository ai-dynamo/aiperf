# WorkerGroupManager Model Boundary Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert newly introduced WorkerGroupManager wire models to `msgspec.Struct` and local-only coordination models to `@dataclass(slots=True)` without broad repo-wide model churn.

**Architecture:** Keep the conversion tightly scoped to the WorkerGroupManager work. Group-local wire contracts become `msgspec.Struct` types suitable for high-frequency runtime transport, while in-memory orchestration state stays lightweight `dataclass(slots=True)` data. Existing config/API/export models remain untouched.

**Tech Stack:** Python 3.10+, msgspec, dataclasses, existing worker/group-manager code, pytest

---

## File map

- Modify: `src/aiperf/workers/group_dataset_authority.py`
- Modify: `src/aiperf/workers/group_runtime.py`
- Modify: `src/aiperf/workers/group_lifecycle_transport.py`
- Modify: `src/aiperf/workers/worker_group_manager.py`
- Modify: `src/aiperf/common/pod_lifecycle_structs.py`
- Modify: `tests/unit/workers/test_worker_group_manager.py`
- Modify: `tests/unit/common/messages/test_messages.py`

### Task 1: Convert new wire contracts to msgspec.Struct

**Files:**
- Modify: `src/aiperf/workers/group_dataset_authority.py`
- Modify: `src/aiperf/common/pod_lifecycle_structs.py`
- Test: `tests/unit/common/messages/test_messages.py`

- [ ] **Step 1: Write failing contract tests**

```python
import msgspec

from aiperf.workers.group_dataset_authority import GroupDatasetSnapshot


def test_group_dataset_snapshot_is_msgspec_struct() -> None:
    assert issubclass(GroupDatasetSnapshot, msgspec.Struct)
```

- [ ] **Step 2: Run the focused tests to verify they fail**

Run: `uv run pytest tests/unit/common/messages/test_messages.py -v`
Expected: FAIL because the new group-local wire models are not yet msgspec structs.

- [ ] **Step 3: Convert the wire models**

```python
import msgspec


class GroupDatasetSnapshot(msgspec.Struct, kw_only=True):
    dataset_ready: bool
    benchmark_generation: str | None
    dataset_generation: str | None = None
    data_file_path: str | None = None
    index_file_path: str | None = None
```

- [ ] **Step 4: Run the focused tests to verify they pass**

Run: `uv run pytest tests/unit/common/messages/test_messages.py -v`
Expected: PASS

### Task 2: Convert local-only coordination models to dataclass(slots=True)

**Files:**
- Modify: `src/aiperf/workers/group_runtime.py`
- Modify: `src/aiperf/workers/worker_group_manager.py`
- Test: `tests/unit/workers/test_worker_group_manager.py`

- [ ] **Step 1: Write failing local-model tests**

```python
from dataclasses import is_dataclass

from aiperf.workers.group_runtime import GroupMemberCapacity
from aiperf.workers.worker_group_manager import GroupChildState


def test_group_member_capacity_is_slotted_dataclass() -> None:
    assert is_dataclass(GroupMemberCapacity)
    assert hasattr(GroupMemberCapacity, "__slots__")


def test_group_child_state_is_slotted_dataclass() -> None:
    assert is_dataclass(GroupChildState)
    assert hasattr(GroupChildState, "__slots__")
```

- [ ] **Step 2: Run the focused tests to verify they fail**

Run: `uv run pytest tests/unit/workers/test_worker_group_manager.py -v`
Expected: FAIL because the target local-only models are not yet slotted dataclasses.

- [ ] **Step 3: Convert the local-only models**

```python
from dataclasses import dataclass


@dataclass(slots=True)
class GroupMemberCapacity:
    num_workers: int
    num_record_processors: int
```

```python
@dataclass(slots=True)
class GroupChildState:
    ...
```

- [ ] **Step 4: Run the focused tests to verify they pass**

Run: `uv run pytest tests/unit/workers/test_worker_group_manager.py -v`
Expected: PASS

### Task 3: Verify consumers still work with the new model boundaries

**Files:**
- Modify: any of the files above as needed
- Test: `tests/unit/workers/test_worker_group_manager.py`
- Test: `tests/unit/common/messages/test_messages.py`

- [ ] **Step 1: Add focused round-trip and usage tests**

```python
def test_group_dataset_snapshot_round_trips_via_msgspec() -> None:
    encoded = msgspec.json.encode(GroupDatasetSnapshot(dataset_ready=True, benchmark_generation="g1"))
    decoded = msgspec.json.decode(encoded, type=GroupDatasetSnapshot)
    assert decoded.dataset_ready is True
```

- [ ] **Step 2: Run the focused verification tests**

Run: `uv run pytest tests/unit/workers/test_worker_group_manager.py tests/unit/common/messages/test_messages.py -v`
Expected: PASS

- [ ] **Step 3: Create the final implementation commit**

```bash
git add src/aiperf/workers/group_dataset_authority.py src/aiperf/workers/group_runtime.py src/aiperf/workers/group_lifecycle_transport.py src/aiperf/workers/worker_group_manager.py src/aiperf/common/pod_lifecycle_structs.py tests/unit/workers/test_worker_group_manager.py tests/unit/common/messages/test_messages.py
git commit -m "refactor: align WorkerGroupManager model boundaries"
```
