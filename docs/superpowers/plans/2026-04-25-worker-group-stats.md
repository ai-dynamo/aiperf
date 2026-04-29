# Worker Group Stats — Route Worker Reporting Through WGM Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the WorkerGroupManager (WGM) the single source of truth for per-worker stats/status reaching the controller, and switch the local web UI to render one row per worker-group manager (with a per-worker dropdown only when there is exactly one group).

**Architecture:**
- Today, in MP/k8s mode, workers send `GroupWorkerHealth` to the WGM via DEALER, and the WGM only republishes a status-enum-only `WorkerStatusSummaryMessage` plus a pod-level `WorkerPodStateMessage`. The controller's `WorkerTrackerMixin` keys by `worker_id` and never sees the per-worker health/task numbers — the local web UI's `worker-table.js` therefore renders mostly-empty per-worker rows.
- The fix: introduce a new `WorkerGroupStatsMessage` (with a `WorkerGroupStats` payload) that the WGM publishes each tick. It carries (a) aggregate health/task stats for the group, (b) per-worker `WorkerStats` children (used for the dropdown), and (c) status/startup-state. The controller tracks `dict[group_id → WorkerGroupStats]`. `/api/workers` keeps its path but returns the new shape. The static-v2 UI renders one row per group; if the map has exactly one group, the row is expandable and lists its worker children.
- The fake-in-process test path (`AIPERF_FAKE_IN_PROCESS_MODE=1`) has no real WGM; we keep it working by also accepting `WorkerHealthMessage` in the controller and synthesizing a single-group fallback (`group_id="local"`).

**Tech Stack:** Python 3.10+, msgspec/pydantic models, FastAPI (web API), Preact + htm (static-v2 UI), Textual (dashboard UI), pytest + asyncio.

---

## File Structure

**Create:**
- `tests/unit/common/messages/test_worker_group_stats_message.py` — message model tests
- `tests/unit/workers/test_build_worker_group_stats.py` — aggregation function tests

**Modify (per task):**
- `src/aiperf/common/enums/messages.py` — add `WORKER_GROUP_STATS` enum value
- `src/aiperf/common/models/progress_models.py` — add `WorkerGroupStats` dataclass
- `src/aiperf/common/models/__init__.py` — export `WorkerGroupStats`
- `src/aiperf/common/messages/worker_messages.py` — add `WorkerGroupStatsMessage`
- `src/aiperf/common/messages/__init__.py` — export new message
- `src/aiperf/workers/worker_pod_helpers.py` — add `build_worker_group_stats(...)`
- `src/aiperf/workers/worker_pod_manager.py` — publish `WorkerGroupStatsMessage` in `_publish_worker_summary`
- `src/aiperf/common/mixins/worker_tracker_mixin.py` — replace `WorkerTracker` semantics with group-keyed tracker; subscribe to `WORKER_GROUP_STATS`; keep degenerate `WORKER_HEALTH` fallback for fake-in-process mode
- `src/aiperf/common/hooks/_core.py` and `_decorators.py` — add `ON_WORKER_GROUP_UPDATE` hook
- `src/aiperf/api/models/responses.py` — change `WorkersResponse.workers` field to `worker_groups: dict[str, WorkerGroupStats]`
- `src/aiperf/api/routers/workers.py` — return new shape
- `src/aiperf/api/static-v2/lib/state.js` — replace `workers` signal with `workerGroups` signal
- `src/aiperf/api/static-v2/lib/ws-dispatch.js` — handle `worker_group_stats`; drop per-worker `worker_health` consumption (legacy `worker_status_summary` keeps updating top-level group statuses if present)
- `src/aiperf/api/static-v2/lib/ws.js` — subscribe to `worker_group_stats`
- `src/aiperf/api/static-v2/components/worker-table.js` — render group rows; expandable child list when exactly 1 group
- `src/aiperf/api/static-v2/style.css` — minimal styles for expand toggle
- `src/aiperf/ui/dashboard/worker_status_table.py` — render group rows
- `src/aiperf/ui/dashboard/worker_dashboard.py` — adapt hook signatures
- `src/aiperf/ui/dashboard/aiperf_textual_app.py` — adapt callbacks
- `src/aiperf/ui/dashboard/aiperf_dashboard_ui.py` — wire `ON_WORKER_GROUP_UPDATE` hook

**Modify (test files):**
- `tests/unit/common/mixins/test_worker_tracker.py` — rewrite around group-keyed tracker
- `tests/unit/api/routers/test_workers.py` — assert new payload shape
- `tests/unit/workers/test_worker_pod_manager.py` — add coverage for new published message

---

## Task 1: Add `WorkerGroupStats` model and `WorkerGroupStatsMessage`

**Files:**
- Modify: `src/aiperf/common/enums/messages.py` — add `WORKER_GROUP_STATS = "worker_group_stats"`
- Modify: `src/aiperf/common/models/progress_models.py` — append the new dataclass
- Modify: `src/aiperf/common/models/__init__.py` — export `WorkerGroupStats`
- Modify: `src/aiperf/common/messages/worker_messages.py` — append message class
- Modify: `src/aiperf/common/messages/__init__.py` — export `WorkerGroupStatsMessage`
- Test: `tests/unit/common/messages/test_worker_group_stats_message.py`

- [ ] **Step 1: Add `WORKER_GROUP_STATS` to MessageType**

In `src/aiperf/common/enums/messages.py`, add a new enum value alongside the existing worker message types:

```python
WORKER_GROUP_STATS = "worker_group_stats"
```

Place it next to `WORKER_STATUS_SUMMARY` and `WORKER_POD_STATE`.

- [ ] **Step 2: Add `WorkerGroupStats` dataclass**

Append to `src/aiperf/common/models/progress_models.py`:

```python
@dataclass(slots=True, kw_only=True)
class WorkerGroupStats:
    """Aggregate stats for one worker-group (one WorkerGroupManager).

    Mutable slotted dataclass, shared between msgspec (HTTP /api/workers
    payload encoded via msgspec) and Pydantic (``WorkersResponse``).

    ``workers`` is the per-child WorkerStats map, used by the local web UI
    when there is exactly one group (expandable dropdown).
    """

    __pydantic_config__: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    group_id: str
    status: WorkerStatus = WorkerStatus.IDLE
    startup_state: WorkerStartupState | None = None
    declared_workers: int = 0
    ready_workers: int = 0
    health: ProcessHealth | None = None
    task_stats: WorkerTaskStats = field(default_factory=WorkerTaskStats)
    workers: dict[str, WorkerStats] = field(default_factory=dict)
    last_update_ns: int | None = None
```

`ProcessHealth` here is the *aggregate* (avg cpu_usage, sum memory_usage across children). `task_stats` is the sum.

Export it from `src/aiperf/common/models/__init__.py` (add to both the `from progress_models import …` line and the `__all__` list).

- [ ] **Step 3: Add `WorkerGroupStatsMessage`**

Append to `src/aiperf/common/messages/worker_messages.py`:

```python
class WorkerGroupStatsMessage(
    BaseServiceMessage, kw_only=True, tag=MessageType.WORKER_GROUP_STATS.value
):
    """Aggregate WorkerGroupStats published by a WorkerGroupManager."""

    group: WorkerGroupStats
```

You will need `from aiperf.common.models import WorkerGroupStats` at the top — but msgspec Structs cannot embed Pydantic dataclasses with `__pydantic_config__` directly through union discrimination. To keep the wire format simple, declare the message as carrying the **pre-encoded fields**:

```python
class WorkerGroupStatsMessage(
    BaseServiceMessage, kw_only=True, tag=MessageType.WORKER_GROUP_STATS.value
):
    """Aggregate stats for a single worker-group manager."""

    group_id: str
    status: WorkerStatus
    startup_state: WorkerStartupState | None = None
    declared_workers: int = 0
    ready_workers: int = 0
    health: ProcessHealth | None = None
    task_stats: WorkerTaskStats
    worker_statuses: dict[str, WorkerStatus] = msgspec.field(default_factory=dict)
    worker_startup_states: dict[str, WorkerStartupState] = msgspec.field(default_factory=dict)
    worker_task_stats: dict[str, WorkerTaskStats] = msgspec.field(default_factory=dict)
    worker_health: dict[str, ProcessHealth] = msgspec.field(default_factory=dict)
    last_update_ns: int = msgspec.field(default_factory=time.time_ns)
```

(Per-worker health/task is carried inline so the controller can populate `WorkerGroupStats.workers` with full `WorkerStats` for the dropdown.)

Export it from `src/aiperf/common/messages/__init__.py`.

- [ ] **Step 4: Write the tests**

`tests/unit/common/messages/test_worker_group_stats_message.py`:

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for WorkerGroupStatsMessage round-trip and defaults."""

from __future__ import annotations

import msgspec

from aiperf.common.enums import MessageType, WorkerStartupState, WorkerStatus
from aiperf.common.messages import WorkerGroupStatsMessage
from aiperf.common.models import ProcessHealth, WorkerTaskStats


def _health() -> ProcessHealth:
    return ProcessHealth(
        pid=1, create_time=0.0, uptime=1.0, cpu_usage=12.5, memory_usage=2048
    )


def test_message_tag_matches_enum() -> None:
    msg = WorkerGroupStatsMessage(
        service_id="wgm-0",
        group_id="wgm-0",
        status=WorkerStatus.HEALTHY,
        task_stats=WorkerTaskStats(),
    )
    assert msg.message_type == MessageType.WORKER_GROUP_STATS


def test_round_trip_preserves_per_worker_maps() -> None:
    msg = WorkerGroupStatsMessage(
        service_id="wgm-0",
        group_id="wgm-0",
        status=WorkerStatus.HIGH_LOAD,
        startup_state=WorkerStartupState.READY,
        declared_workers=2,
        ready_workers=2,
        health=_health(),
        task_stats=WorkerTaskStats(total=10, failed=1),
        worker_statuses={"w-0": WorkerStatus.HEALTHY, "w-1": WorkerStatus.HIGH_LOAD},
        worker_startup_states={"w-0": WorkerStartupState.READY},
        worker_task_stats={"w-0": WorkerTaskStats(total=5)},
        worker_health={"w-0": _health()},
    )
    encoded = msgspec.json.encode(msg)
    decoded = msgspec.json.decode(encoded, type=WorkerGroupStatsMessage)
    assert decoded.group_id == "wgm-0"
    assert decoded.worker_statuses == msg.worker_statuses
    assert decoded.worker_task_stats["w-0"].total == 5
    assert decoded.health.cpu_usage == 12.5


def test_defaults_are_empty_maps() -> None:
    msg = WorkerGroupStatsMessage(
        service_id="wgm-0",
        group_id="wgm-0",
        status=WorkerStatus.IDLE,
        task_stats=WorkerTaskStats(),
    )
    assert msg.worker_statuses == {}
    assert msg.worker_task_stats == {}
    assert msg.worker_health == {}
```

- [ ] **Step 5: Run the test**

```bash
uv run pytest tests/unit/common/messages/test_worker_group_stats_message.py -n auto -v
```

Expected: all 3 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add src/aiperf/common/enums/messages.py \
        src/aiperf/common/models/progress_models.py \
        src/aiperf/common/models/__init__.py \
        src/aiperf/common/messages/worker_messages.py \
        src/aiperf/common/messages/__init__.py \
        tests/unit/common/messages/test_worker_group_stats_message.py
git commit -s -m "feat(messages): add WorkerGroupStatsMessage and WorkerGroupStats model"
```

---

## Task 2: WGM aggregation helper + publish call

**Files:**
- Modify: `src/aiperf/workers/worker_pod_helpers.py` — add `build_worker_group_stats(...)`
- Modify: `src/aiperf/workers/worker_pod_manager.py` — publish in `_publish_worker_summary`
- Test: `tests/unit/workers/test_build_worker_group_stats.py`

- [ ] **Step 1: Write the failing aggregation test**

`tests/unit/workers/test_build_worker_group_stats.py`:

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for build_worker_group_stats aggregation helper."""

from __future__ import annotations

from aiperf.common.enums import WorkerStartupState, WorkerStatus
from aiperf.common.models import ProcessHealth, WorkerTaskStats
from aiperf.workers.worker_group_state import WorkerStatusInfo
from aiperf.workers.worker_pod_helpers import build_worker_group_stats


def _info(
    *,
    worker_id: str,
    status: WorkerStatus,
    cpu: float,
    mem: int,
    total: int,
    failed: int = 0,
    startup: WorkerStartupState | None = WorkerStartupState.READY,
) -> WorkerStatusInfo:
    info = WorkerStatusInfo(worker_id=worker_id)
    info.status = status
    info.startup_state = startup
    info.health = ProcessHealth(
        pid=1, create_time=0.0, uptime=1.0, cpu_usage=cpu, memory_usage=mem
    )
    info.task_stats = WorkerTaskStats(total=total, failed=failed)
    return info


def test_aggregates_task_stats_as_sum() -> None:
    workers = {
        "w-0": _info(worker_id="w-0", status=WorkerStatus.HEALTHY, cpu=10.0, mem=100, total=5, failed=0),
        "w-1": _info(worker_id="w-1", status=WorkerStatus.HEALTHY, cpu=20.0, mem=200, total=7, failed=2),
    }
    msg = build_worker_group_stats(
        service_id="wgm-0",
        declared_workers=2,
        worker_infos=workers,
    )
    assert msg.task_stats.total == 12
    assert msg.task_stats.failed == 2


def test_aggregates_cpu_as_average_memory_as_sum() -> None:
    workers = {
        "w-0": _info(worker_id="w-0", status=WorkerStatus.HEALTHY, cpu=10.0, mem=100, total=0),
        "w-1": _info(worker_id="w-1", status=WorkerStatus.HEALTHY, cpu=30.0, mem=200, total=0),
    }
    msg = build_worker_group_stats(
        service_id="wgm-0", declared_workers=2, worker_infos=workers
    )
    assert msg.health is not None
    assert msg.health.cpu_usage == 20.0
    assert msg.health.memory_usage == 300


def test_group_status_uses_worst_child() -> None:
    workers = {
        "w-0": _info(worker_id="w-0", status=WorkerStatus.HEALTHY, cpu=0.0, mem=0, total=0),
        "w-1": _info(worker_id="w-1", status=WorkerStatus.ERROR, cpu=0.0, mem=0, total=0),
    }
    msg = build_worker_group_stats(
        service_id="wgm-0", declared_workers=2, worker_infos=workers
    )
    assert msg.status == WorkerStatus.ERROR


def test_per_worker_maps_populated() -> None:
    workers = {
        "w-0": _info(worker_id="w-0", status=WorkerStatus.HEALTHY, cpu=5.0, mem=10, total=3),
    }
    msg = build_worker_group_stats(
        service_id="wgm-0", declared_workers=1, worker_infos=workers
    )
    assert msg.worker_statuses == {"w-0": WorkerStatus.HEALTHY}
    assert msg.worker_task_stats["w-0"].total == 3
    assert msg.worker_health["w-0"].cpu_usage == 5.0
    assert msg.worker_startup_states["w-0"] == WorkerStartupState.READY


def test_ready_workers_counts_ready_startup_state_only() -> None:
    workers = {
        "w-0": _info(worker_id="w-0", status=WorkerStatus.HEALTHY, cpu=0.0, mem=0, total=0,
                     startup=WorkerStartupState.READY),
        "w-1": _info(worker_id="w-1", status=WorkerStatus.IDLE, cpu=0.0, mem=0, total=0,
                     startup=WorkerStartupState.WAITING_FOR_DATASET),
    }
    msg = build_worker_group_stats(
        service_id="wgm-0", declared_workers=2, worker_infos=workers
    )
    assert msg.ready_workers == 1


def test_empty_group_yields_idle_zeroed_message() -> None:
    msg = build_worker_group_stats(
        service_id="wgm-0", declared_workers=0, worker_infos={}
    )
    assert msg.status == WorkerStatus.IDLE
    assert msg.task_stats.total == 0
    assert msg.worker_statuses == {}
    assert msg.health is None
```

- [ ] **Step 2: Verify it fails**

```bash
uv run pytest tests/unit/workers/test_build_worker_group_stats.py -n auto -v
```

Expected: ImportError on `build_worker_group_stats`.

- [ ] **Step 3: Implement the helper**

Append to `src/aiperf/workers/worker_pod_helpers.py` (and add necessary imports near the top — `WorkerStatus`, `WorkerStartupState`, `WorkerGroupStatsMessage`, `WorkerStatusInfo`):

```python
def build_worker_group_stats(
    *,
    service_id: str,
    declared_workers: int,
    worker_infos: Mapping[str, "WorkerStatusInfo"],
) -> WorkerGroupStatsMessage:
    """Aggregate per-child status into a single WorkerGroupStatsMessage.

    - ``task_stats`` summed (total/failed/completed/in_progress).
    - ``health.cpu_usage`` averaged across children with a non-None health.
    - ``health.memory_usage`` summed.
    - Group status = worst child status (ERROR > HIGH_LOAD > STALE > HEALTHY > IDLE).
    - ``ready_workers`` = count of children with ``startup_state == READY``.
    """
    statuses = {wid: info.status for wid, info in worker_infos.items()}
    startup_states = {
        wid: info.startup_state
        for wid, info in worker_infos.items()
        if info.startup_state is not None
    }
    task_stats_map = {wid: info.task_stats for wid, info in worker_infos.items()}
    health_map = {
        wid: info.health for wid, info in worker_infos.items() if info.health is not None
    }

    total = sum(t.total for t in task_stats_map.values())
    failed = sum(t.failed for t in task_stats_map.values())
    completed = sum(getattr(t, "completed", 0) for t in task_stats_map.values())
    in_progress = sum(getattr(t, "in_progress", 0) for t in task_stats_map.values())
    aggregated_task_stats = WorkerTaskStats(
        total=total, failed=failed, completed=completed, in_progress=in_progress
    )

    aggregated_health: ProcessHealth | None = None
    if health_map:
        cpu_avg = sum(h.cpu_usage for h in health_map.values()) / len(health_map)
        mem_sum = sum(h.memory_usage for h in health_map.values())
        first = next(iter(health_map.values()))
        aggregated_health = ProcessHealth(
            pid=first.pid,
            create_time=first.create_time,
            uptime=max(h.uptime for h in health_map.values()),
            cpu_usage=cpu_avg,
            memory_usage=mem_sum,
        )

    group_status = _worst_status([info.status for info in worker_infos.values()])
    ready_workers = sum(
        1 for s in startup_states.values() if s == WorkerStartupState.READY
    )

    return WorkerGroupStatsMessage(
        service_id=service_id,
        group_id=service_id,
        status=group_status,
        startup_state=None,
        declared_workers=declared_workers,
        ready_workers=ready_workers,
        health=aggregated_health,
        task_stats=aggregated_task_stats,
        worker_statuses=statuses,
        worker_startup_states=startup_states,
        worker_task_stats=task_stats_map,
        worker_health={wid: h for wid, h in health_map.items()},
    )


_STATUS_RANK = {
    WorkerStatus.IDLE: 0,
    WorkerStatus.HEALTHY: 1,
    WorkerStatus.STALE: 2,
    WorkerStatus.HIGH_LOAD: 3,
    WorkerStatus.ERROR: 4,
}


def _worst_status(statuses: list[WorkerStatus]) -> WorkerStatus:
    if not statuses:
        return WorkerStatus.IDLE
    return max(statuses, key=lambda s: _STATUS_RANK.get(s, 0))
```

(The `WorkerStatusInfo` import goes inside `if TYPE_CHECKING:` to avoid circular imports; type annotation is a string.)

- [ ] **Step 4: Wire it into the WGM publish path**

In `src/aiperf/workers/worker_pod_manager.py`, in `_publish_worker_summary` (around line 436-453), publish the new message *in addition to* the existing two:

```python
async def _publish_worker_summary(self) -> None:
    """Publish worker-group, worker-summary, and pod-state snapshots."""
    summary = build_worker_status_summary(
        service_id=self.service_id,
        worker_infos=self.worker_health,
    )
    pod_summary = build_pod_summary(
        service_id=self.service_id,
        pod_index=self._pod_index,
        benchmark_generation=self._benchmark_generation,
        dataset_generation=self._dataset_generation,
        workers_per_pod=self.workers_per_pod,
        record_processors_per_pod=self.record_processors_per_pod,
        worker_startup_states=summary.worker_startup_states,
        peer_types=self._pod_peer_types,
    )
    group_stats = build_worker_group_stats(
        service_id=self.service_id,
        declared_workers=self.workers_per_pod,
        worker_infos=self.worker_health,
    )
    await self.publish(group_stats)
    await self.publish(summary)
    await self.publish(pod_summary)
```

Add the import at the top of `worker_pod_manager.py`:

```python
from aiperf.workers.worker_pod_helpers import (
    ...,  # existing imports
    build_worker_group_stats,
)
```

- [ ] **Step 5: Run the tests**

```bash
uv run pytest tests/unit/workers/test_build_worker_group_stats.py tests/unit/workers/test_worker_pod_manager.py -n auto -v
```

Expected: all 6 new aggregation tests PASS, existing pod-manager tests PASS.

- [ ] **Step 6: Commit**

```bash
git add src/aiperf/workers/worker_pod_helpers.py \
        src/aiperf/workers/worker_pod_manager.py \
        tests/unit/workers/test_build_worker_group_stats.py
git commit -s -m "feat(workers): WGM publishes WorkerGroupStatsMessage with per-child rollup"
```

---

## Task 3: Controller-side `WorkerGroupTracker` + mixin update

**Files:**
- Modify: `src/aiperf/common/hooks/_core.py` — add `ON_WORKER_GROUP_UPDATE = "@on_worker_group_update"`
- Modify: `src/aiperf/common/hooks/_decorators.py` — add `on_worker_group_update` decorator helper (mirror `on_worker_update`)
- Modify: `src/aiperf/common/mixins/worker_tracker_mixin.py` — replace `WorkerTracker` semantics
- Modify: `tests/unit/common/mixins/test_worker_tracker.py` — rewrite for new semantics

- [ ] **Step 1: Add new hook enum + decorator**

In `src/aiperf/common/hooks/_core.py`, add a new enum value:

```python
ON_WORKER_GROUP_UPDATE = "@on_worker_group_update"
```

In `src/aiperf/common/hooks/_decorators.py`, add a sibling helper to `on_worker_update`:

```python
def on_worker_group_update(func):
    """Decorator: invoked when a WorkerGroupStatsMessage updates a group."""
    MyPlugin._on_worker_group_update.__aiperf_hook_type__ = AIPerfHook.ON_WORKER_GROUP_UPDATE  # noqa: F821
    return _hook_decorator(AIPerfHook.ON_WORKER_GROUP_UPDATE, func)
```

(Match the exact body style of `on_worker_update` immediately above.)

- [ ] **Step 2: Rewrite `WorkerTracker` and `WorkerTrackerMixin` to be group-keyed**

Replace the body of `src/aiperf/common/mixins/worker_tracker_mixin.py` with:

```python
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

from aiperf.common.enums import MessageType, WorkerStartupState, WorkerStatus
from aiperf.common.hooks import AIPerfHook, on_message, provides_hooks
from aiperf.common.messages import (
    WorkerGroupStatsMessage,
    WorkerHealthMessage,
    WorkerStatusSummaryMessage,
)
from aiperf.common.mixins.message_bus_mixin import MessageBusClientMixin
from aiperf.common.models import (
    ProcessHealth,
    WorkerGroupStats,
    WorkerStats,
    WorkerTaskStats,
)

if TYPE_CHECKING:
    from aiperf.config import BenchmarkRun


_FAKE_GROUP_ID = "local"
"""Synthetic group id used in fake-in-process mode where no real WGM exists."""


class WorkerGroupTracker:
    """Standalone per-group worker tracker.

    Keyed by ``group_id`` (the WorkerGroupManager service_id). In fake-in-process
    mode (no WGM), per-worker WORKER_HEALTH messages are folded into a single
    synthetic group ``"local"`` so the dashboard still has a row to render.
    """

    def __init__(self) -> None:
        self._groups: dict[str, WorkerGroupStats] = {}

    def update_from_group_message(
        self, message: WorkerGroupStatsMessage
    ) -> WorkerGroupStats:
        """Replace a group entry from a freshly-published WGM snapshot."""
        children: dict[str, WorkerStats] = {}
        for wid, status in message.worker_statuses.items():
            children[wid] = WorkerStats(
                worker_id=wid,
                status=status,
                startup_state=message.worker_startup_states.get(wid),
                health=message.worker_health.get(wid),
                task_stats=message.worker_task_stats.get(wid, WorkerTaskStats()),
                last_update_ns=message.last_update_ns,
            )
        group = WorkerGroupStats(
            group_id=message.group_id,
            status=message.status,
            startup_state=message.startup_state,
            declared_workers=message.declared_workers,
            ready_workers=message.ready_workers,
            health=message.health,
            task_stats=message.task_stats,
            workers=children,
            last_update_ns=message.last_update_ns,
        )
        self._groups[message.group_id] = group
        return group

    def update_from_worker_health(
        self,
        worker_id: str,
        health: ProcessHealth,
        task_stats: WorkerTaskStats,
    ) -> WorkerGroupStats:
        """Fold a per-worker health message into the synthetic ``local`` group.

        Only used by fake-in-process tests where workers publish
        WORKER_HEALTH directly (no WorkerGroupManager exists).
        """
        group = self._groups.get(_FAKE_GROUP_ID) or WorkerGroupStats(
            group_id=_FAKE_GROUP_ID
        )
        children = dict(group.workers)
        children[worker_id] = WorkerStats(
            worker_id=worker_id,
            status=WorkerStatus.HEALTHY,
            health=health,
            task_stats=task_stats,
        )
        group.workers = children
        group.task_stats = WorkerTaskStats(
            total=sum(c.task_stats.total for c in children.values()),
            failed=sum(c.task_stats.failed for c in children.values()),
        )
        if children:
            group.health = ProcessHealth(
                pid=next(iter(children.values())).health.pid,
                create_time=next(iter(children.values())).health.create_time,
                uptime=max(
                    (c.health.uptime for c in children.values() if c.health), default=0.0
                ),
                cpu_usage=sum(
                    c.health.cpu_usage for c in children.values() if c.health
                ) / max(1, sum(1 for c in children.values() if c.health)),
                memory_usage=sum(
                    c.health.memory_usage for c in children.values() if c.health
                ),
            )
        group.declared_workers = max(group.declared_workers, len(children))
        self._groups[_FAKE_GROUP_ID] = group
        return group

    def update_worker_statuses(self, worker_statuses: dict[str, WorkerStatus]) -> None:
        """Update per-worker statuses (legacy WORKER_STATUS_SUMMARY path).

        Folded under the synthetic ``local`` group when there is no
        matching WGM-keyed group yet.
        """
        group = self._groups.get(_FAKE_GROUP_ID) or WorkerGroupStats(
            group_id=_FAKE_GROUP_ID
        )
        children = dict(group.workers)
        for worker_id, status in worker_statuses.items():
            child = children.get(worker_id) or WorkerStats(worker_id=worker_id)
            child.status = status
            children[worker_id] = child
        group.workers = children
        self._groups[_FAKE_GROUP_ID] = group

    def update_worker_startup_states(
        self, worker_startup_states: dict[str, WorkerStartupState]
    ) -> None:
        group = self._groups.get(_FAKE_GROUP_ID) or WorkerGroupStats(
            group_id=_FAKE_GROUP_ID
        )
        children = dict(group.workers)
        for worker_id, startup_state in worker_startup_states.items():
            child = children.get(worker_id) or WorkerStats(worker_id=worker_id)
            child.startup_state = startup_state
            children[worker_id] = child
        group.workers = children
        self._groups[_FAKE_GROUP_ID] = group

    def get_group(self, group_id: str) -> WorkerGroupStats | None:
        return self._groups.get(group_id)

    @property
    def worker_groups(self) -> dict[str, WorkerGroupStats]:
        return self._groups


@provides_hooks(
    AIPerfHook.ON_WORKER_GROUP_UPDATE,
    AIPerfHook.ON_WORKER_UPDATE,
    AIPerfHook.ON_WORKER_STATUS_SUMMARY,
)
class WorkerTrackerMixin(MessageBusClientMixin):
    """Tracks worker-group health/stats via the message bus."""

    def __init__(self, run: BenchmarkRun, **kwargs):
        super().__init__(run=run, **kwargs)
        self._worker_tracker = WorkerGroupTracker()

    @on_message(MessageType.WORKER_GROUP_STATS)
    async def _on_worker_group_stats(self, message: WorkerGroupStatsMessage) -> None:
        group = self._worker_tracker.update_from_group_message(message)
        await self.run_hooks(
            AIPerfHook.ON_WORKER_GROUP_UPDATE,
            group_id=group.group_id,
            group_stats=group,
        )

    @on_message(MessageType.WORKER_HEALTH)
    async def _on_worker_health(self, message: WorkerHealthMessage) -> None:
        """Fake-in-process fallback: worker publishes raw health, no WGM."""
        group = self._worker_tracker.update_from_worker_health(
            message.service_id, message.health, message.task_stats
        )
        await self.run_hooks(
            AIPerfHook.ON_WORKER_GROUP_UPDATE,
            group_id=group.group_id,
            group_stats=group,
        )
        # Keep legacy hook firing so any external listener still works.
        await self.run_hooks(
            AIPerfHook.ON_WORKER_UPDATE,
            worker_id=message.service_id,
            worker_stats=group.workers[message.service_id],
        )

    @on_message(MessageType.WORKER_STATUS_SUMMARY)
    async def _on_worker_status_summary(
        self, message: WorkerStatusSummaryMessage
    ) -> None:
        self._worker_tracker.update_worker_statuses(message.worker_statuses)
        self._worker_tracker.update_worker_startup_states(message.worker_startup_states)
        await self.run_hooks(
            AIPerfHook.ON_WORKER_STATUS_SUMMARY,
            worker_status_summary=message.worker_statuses,
            worker_startup_states=message.worker_startup_states,
        )
```

(Note: the legacy `WorkerTracker` class name is dropped, so any external caller that imports it breaks loudly — the test file is rewritten in step 3 to match.)

- [ ] **Step 3: Rewrite the mixin tests around groups**

Replace `tests/unit/common/mixins/test_worker_tracker.py` with:

```python
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the standalone WorkerGroupTracker."""

from __future__ import annotations

import pytest

from aiperf.common.enums import WorkerStartupState, WorkerStatus
from aiperf.common.messages import WorkerGroupStatsMessage
from aiperf.common.mixins.worker_tracker_mixin import WorkerGroupTracker
from aiperf.common.models import ProcessHealth, WorkerTaskStats


@pytest.fixture
def tracker() -> WorkerGroupTracker:
    return WorkerGroupTracker()


def _health(cpu: float = 10.0, mem: int = 1024) -> ProcessHealth:
    return ProcessHealth(pid=1, create_time=0.0, uptime=1.0, cpu_usage=cpu, memory_usage=mem)


def _group_msg(**overrides) -> WorkerGroupStatsMessage:
    base = dict(
        service_id="wgm-0",
        group_id="wgm-0",
        status=WorkerStatus.HEALTHY,
        declared_workers=2,
        ready_workers=2,
        health=_health(),
        task_stats=WorkerTaskStats(total=4),
        worker_statuses={
            "w-0": WorkerStatus.HEALTHY,
            "w-1": WorkerStatus.HEALTHY,
        },
        worker_startup_states={"w-0": WorkerStartupState.READY},
        worker_task_stats={"w-0": WorkerTaskStats(total=2), "w-1": WorkerTaskStats(total=2)},
        worker_health={"w-0": _health(), "w-1": _health()},
    )
    base.update(overrides)
    return WorkerGroupStatsMessage(**base)


class TestUpdateFromGroupMessage:
    def test_creates_group_entry(self, tracker: WorkerGroupTracker) -> None:
        group = tracker.update_from_group_message(_group_msg())
        assert group.group_id == "wgm-0"
        assert group.status == WorkerStatus.HEALTHY
        assert set(group.workers.keys()) == {"w-0", "w-1"}

    def test_replaces_group_entry_on_subsequent_update(
        self, tracker: WorkerGroupTracker
    ) -> None:
        tracker.update_from_group_message(_group_msg())
        tracker.update_from_group_message(
            _group_msg(
                status=WorkerStatus.HIGH_LOAD,
                worker_statuses={"w-0": WorkerStatus.HIGH_LOAD},
                worker_task_stats={"w-0": WorkerTaskStats(total=2)},
                worker_health={"w-0": _health()},
            )
        )
        group = tracker.get_group("wgm-0")
        assert group is not None
        assert group.status == WorkerStatus.HIGH_LOAD
        assert set(group.workers.keys()) == {"w-0"}

    def test_per_child_stats_populated(self, tracker: WorkerGroupTracker) -> None:
        group = tracker.update_from_group_message(_group_msg())
        assert group.workers["w-0"].task_stats.total == 2
        assert group.workers["w-0"].health is not None
        assert group.workers["w-0"].startup_state == WorkerStartupState.READY


class TestFakeInProcessFallback:
    def test_worker_health_creates_local_group(
        self, tracker: WorkerGroupTracker
    ) -> None:
        tracker.update_from_worker_health(
            "w-0", _health(cpu=20.0), WorkerTaskStats(total=3)
        )
        group = tracker.get_group("local")
        assert group is not None
        assert group.workers["w-0"].task_stats.total == 3

    def test_multiple_workers_aggregated_in_local_group(
        self, tracker: WorkerGroupTracker
    ) -> None:
        tracker.update_from_worker_health("w-0", _health(cpu=10.0, mem=1000), WorkerTaskStats(total=2))
        tracker.update_from_worker_health("w-1", _health(cpu=30.0, mem=2000), WorkerTaskStats(total=4))
        group = tracker.get_group("local")
        assert group is not None
        assert group.task_stats.total == 6
        assert group.health is not None
        assert group.health.memory_usage == 3000
        assert group.health.cpu_usage == 20.0


class TestWorkerGroupsProperty:
    def test_empty_initially(self, tracker: WorkerGroupTracker) -> None:
        assert tracker.worker_groups == {}

    def test_keyed_by_group_id(self, tracker: WorkerGroupTracker) -> None:
        tracker.update_from_group_message(_group_msg(service_id="wgm-0", group_id="wgm-0"))
        tracker.update_from_group_message(_group_msg(service_id="wgm-1", group_id="wgm-1"))
        assert set(tracker.worker_groups.keys()) == {"wgm-0", "wgm-1"}
```

- [ ] **Step 4: Run the tests**

```bash
uv run pytest tests/unit/common/mixins/test_worker_tracker.py tests/unit/common/hooks tests/unit/common/messages tests/unit/workers -n auto -v
```

Expected: all PASS. (If `tests/unit/common/test_worker_tracker.py` exists as a duplicate of the mixins one, delete the duplicate — keep only the mixins location.)

- [ ] **Step 5: Commit**

```bash
git add src/aiperf/common/hooks/_core.py \
        src/aiperf/common/hooks/_decorators.py \
        src/aiperf/common/mixins/worker_tracker_mixin.py \
        tests/unit/common/mixins/test_worker_tracker.py
git commit -s -m "refactor(worker-tracker): switch to group-keyed tracker fed by WorkerGroupStatsMessage"
```

---

## Task 4: API model + router shape change

**Files:**
- Modify: `src/aiperf/api/models/responses.py` — replace `WorkersResponse.workers` field with `worker_groups: dict[str, WorkerGroupStats]`
- Modify: `src/aiperf/api/routers/workers.py` — return new shape from `_worker_tracker.worker_groups`
- Modify: `tests/unit/api/routers/test_workers.py` — update assertions

- [ ] **Step 1: Update the response model**

In `src/aiperf/api/models/responses.py`, change the import to bring in `WorkerGroupStats` from `aiperf.common.models`, then update the class:

```python
class WorkersResponse(AIPerfBaseModel):
    """Per-worker-group stats payload for /api/workers."""

    worker_groups: dict[str, WorkerGroupStats] = Field(
        description="Per-worker-group aggregated stats keyed by group_id."
    )
```

(Drop the `WorkerStats` import if it's no longer used; otherwise keep it — `WorkerStats` is now nested inside `WorkerGroupStats.workers`.)

- [ ] **Step 2: Update the router**

In `src/aiperf/api/routers/workers.py`:

```python
@workers_router.get("/api/workers", response_model=WorkersResponse, tags=["API"])
async def get_workers(component: WorkersDep) -> WorkersResponse:
    """Get worker-group status with full per-group stats and per-child rollup."""
    return WorkersResponse(worker_groups=component._worker_tracker.worker_groups)
```

- [ ] **Step 3: Update the router tests**

Replace the body of `tests/unit/api/routers/test_workers.py`:

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for WorkersRouter."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from pytest import param
from starlette.testclient import TestClient

from aiperf.api.routers.workers import WorkersRouter
from aiperf.common.enums import WorkerStatus
from aiperf.common.models import WorkerGroupStats, WorkerStats
from aiperf.config import AIPerfConfig


@pytest.fixture
def workers_router(mock_zmq, router_config: AIPerfConfig) -> WorkersRouter:
    return WorkersRouter(run=router_config)


@pytest.fixture
def workers_client(workers_router: WorkersRouter) -> TestClient:
    app = FastAPI()
    app.state.workers = workers_router
    app.include_router(workers_router.get_router())
    return TestClient(app)


class TestWorkersEndpoint:
    """Test the /api/workers endpoint (group-keyed payload)."""

    def test_workers_empty(self, workers_client: TestClient) -> None:
        response = workers_client.get("/api/workers")
        assert response.status_code == 200
        assert response.json() == {"worker_groups": {}}

    @pytest.mark.parametrize(
        "statuses,expected_active",
        [
            param([WorkerStatus.HEALTHY], 1, id="one-healthy"),
            param([WorkerStatus.IDLE], 0, id="one-idle"),
            param([WorkerStatus.HIGH_LOAD], 1, id="one-high-load"),
            param([WorkerStatus.HEALTHY, WorkerStatus.HEALTHY], 2, id="two-healthy"),
            param([WorkerStatus.HEALTHY, WorkerStatus.IDLE], 1, id="one-healthy-one-idle"),
        ],
    )  # fmt: skip
    def test_single_group_active_count(
        self,
        workers_client: TestClient,
        workers_router: WorkersRouter,
        statuses: list[WorkerStatus],
        expected_active: int,
    ) -> None:
        children = {
            f"w-{i}": WorkerStats(worker_id=f"w-{i}", status=status)
            for i, status in enumerate(statuses)
        }
        group_status = (
            WorkerStatus.HIGH_LOAD
            if WorkerStatus.HIGH_LOAD in statuses
            else (WorkerStatus.HEALTHY if WorkerStatus.HEALTHY in statuses
                  else WorkerStatus.IDLE)
        )
        workers_router._worker_tracker._groups = {
            "wgm-0": WorkerGroupStats(
                group_id="wgm-0", status=group_status, workers=children
            )
        }
        response = workers_client.get("/api/workers")
        data = response.json()
        groups = data["worker_groups"]
        assert set(groups.keys()) == {"wgm-0"}
        active = sum(
            1
            for w in groups["wgm-0"]["workers"].values()
            if w["status"] in (WorkerStatus.HEALTHY, WorkerStatus.HIGH_LOAD)
        )
        assert active == expected_active

    def test_multiple_groups_render_separately(
        self, workers_client: TestClient, workers_router: WorkersRouter
    ) -> None:
        workers_router._worker_tracker._groups = {
            "wgm-0": WorkerGroupStats(group_id="wgm-0", status=WorkerStatus.HEALTHY),
            "wgm-1": WorkerGroupStats(group_id="wgm-1", status=WorkerStatus.HIGH_LOAD),
        }
        data = workers_client.get("/api/workers").json()
        assert set(data["worker_groups"].keys()) == {"wgm-0", "wgm-1"}
        assert data["worker_groups"]["wgm-1"]["status"] == WorkerStatus.HIGH_LOAD
```

- [ ] **Step 4: Run the tests**

```bash
uv run pytest tests/unit/api/routers/test_workers.py -n auto -v
```

Expected: all 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/aiperf/api/models/responses.py \
        src/aiperf/api/routers/workers.py \
        tests/unit/api/routers/test_workers.py
git commit -s -m "feat(api): /api/workers returns worker_groups map keyed by group_id"
```

---

## Task 5: Static-v2 frontend — group rows + single-group dropdown

**Files:**
- Modify: `src/aiperf/api/static-v2/lib/state.js` — replace `workers` signal with `workerGroups`
- Modify: `src/aiperf/api/static-v2/lib/ws-dispatch.js` — handle `worker_group_stats`; drop the `worker_health`/`worker_status_summary` per-worker logic
- Modify: `src/aiperf/api/static-v2/lib/ws.js` — subscribe to `worker_group_stats`
- Modify: `src/aiperf/api/static-v2/components/worker-table.js` — new layout
- Modify: `src/aiperf/api/static-v2/style.css` — minimal style additions
- Modify: `src/aiperf/api/static-v2/app.js` — no logic change, just verify import still works

- [ ] **Step 1: Update state.js**

In `src/aiperf/api/static-v2/lib/state.js`, replace the `workers` signal with:

```javascript
/** Map of groupId → WorkerGroupInfo. Each group contains a `workers` child map. */
export const workerGroups = signal({});
```

In `resetLiveState`, replace `workers.value = {};` with `workerGroups.value = {};`.

- [ ] **Step 2: Update ws-dispatch.js**

Replace the `applyWorkers` helper and the `worker_health` / `worker_status_summary` cases with a `worker_group_stats` case:

```javascript
import {
  phases, records, workerGroups, serverMetrics,
  realtimeMetrics, telemetryMetrics,
  recordTimeseriesSample,
  markRunStarted,
  log,
} from './state.js';

// ... existing applyPhase / applyRecords ...

/** Replace one group entry from a WorkerGroupStatsMessage. */
function applyGroupStats(msg) {
  const groupId = msg.group_id ?? msg.service_id;
  if (!groupId) return;
  const children = {};
  for (const [wid, status] of Object.entries(msg.worker_statuses ?? {})) {
    const ts = (msg.worker_task_stats ?? {})[wid] ?? {};
    const wh = (msg.worker_health ?? {})[wid] ?? null;
    children[wid] = {
      id: wid,
      status,
      startupState: (msg.worker_startup_states ?? {})[wid] ?? null,
      inFlight: ts.in_progress ?? 0,
      completed: ts.completed ?? 0,
      failed: ts.failed ?? 0,
      total: ts.total ?? 0,
      cpu: wh?.cpu_usage ?? null,
      memory: wh?.memory_usage ?? null,
    };
  }
  const group = {
    id: groupId,
    status: msg.status ?? 'idle',
    startupState: msg.startup_state ?? null,
    declaredWorkers: msg.declared_workers ?? 0,
    readyWorkers: msg.ready_workers ?? 0,
    inFlight: msg.task_stats?.in_progress ?? 0,
    completed: msg.task_stats?.completed ?? 0,
    failed: msg.task_stats?.failed ?? 0,
    total: msg.task_stats?.total ?? 0,
    cpu: msg.health?.cpu_usage ?? null,
    memory: msg.health?.memory_usage ?? null,
    workers: children,
  };
  workerGroups.value = { ...workerGroups.value, [groupId]: group };
}

export function handleWsMessage(msg) {
  // ... existing cases up through 'all_records_received' ...

  switch (type) {
    // ... unchanged cases ...

    case 'worker_group_stats':
      applyGroupStats(msg);
      return;

    // remove the 'worker_health' and 'worker_status_summary' cases entirely
  }
}
```

(Keep all non-worker cases identical. Remove the legacy `worker_health` and `worker_status_summary` handlers.)

- [ ] **Step 3: Update ws.js subscription list**

In `src/aiperf/api/static-v2/lib/ws.js`, replace `'worker_health',` and `'worker_status_summary',` in `SUBSCRIBE_TYPES` with a single line:

```javascript
'worker_group_stats',
```

- [ ] **Step 4: Update worker-table.js**

Rewrite `src/aiperf/api/static-v2/components/worker-table.js` to render groups, with a single-group dropdown:

```javascript
// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Worker-group roster table. One row per WorkerGroupManager.
 *
 * When there is exactly one group, the row is expandable and reveals a
 * nested per-worker table (the dropdown). With multiple groups, only the
 * group-level row is shown to keep the dashboard scannable.
 */

import { html } from 'htm/preact';
import { useState } from 'preact/hooks';
import { workerGroups } from '../lib/state.js';
import { fmtInt, fmtBytes, fmtPercent } from '../lib/format.js';

const KNOWN_STATUSES = ['healthy', 'high_load', 'error', 'idle', 'stale'];

function shortId(id) {
  if (!id) return '';
  const parts = id.split('-');
  return parts.length <= 2 ? id : parts.slice(-2).join('-');
}

function safeStatusClass(status) {
  return KNOWN_STATUSES.includes(status) ? status : 'idle';
}

function displayStatus(g) {
  const s = (g.status ?? 'idle').replace('_', ' ');
  if (g.startupState && g.startupState !== 'ready') {
    return `${s} (${String(g.startupState).replace(/_/g, ' ')})`;
  }
  return s;
}

function ChildTable({ children }) {
  const ids = Object.keys(children).sort();
  if (ids.length === 0) return html`<tr><td colspan="7" class="empty">No worker children yet.</td></tr>`;
  return ids.map((id) => {
    const w = children[id];
    return html`
      <tr key=${id} class="worker-child-row">
        <td><span class="worker-id" style="padding-left: 18px">↳ ${shortId(id)}</span></td>
        <td><span class=${'worker-status ' + safeStatusClass(w.status)}>${displayStatus(w)}</span></td>
        <td style="text-align: right">${fmtInt(w.inFlight ?? 0)}</td>
        <td style="text-align: right">${fmtInt(w.completed ?? 0)}</td>
        <td style="text-align: right">${fmtInt(w.failed ?? 0)}</td>
        <td style="text-align: right">${w.cpu != null ? fmtPercent(w.cpu) : '---'}</td>
        <td style="text-align: right">${fmtBytes(w.memory)}</td>
      </tr>
    `;
  });
}

export function WorkerTable() {
  const map = workerGroups.value;
  const groupIds = Object.keys(map).sort();
  const singleGroup = groupIds.length === 1;
  const [expanded, setExpanded] = useState(true);  // default open when single group

  return html`
    <div class="card">
      <div class="card-title">Worker Groups <span class="text-dim" style="margin-left: 6px; font-weight: 400">(${groupIds.length})</span></div>
      ${groupIds.length === 0
        ? html`<div class="empty">No worker-group reports yet.</div>`
        : html`
          <div style="overflow-x: auto">
            <table class="worker-table">
              <thead>
                <tr>
                  <th>${singleGroup ? html`<span class="group-toggle" onClick=${() => setExpanded(!expanded)}>${expanded ? '▾' : '▸'}</span> ` : ''}Group</th>
                  <th>Status</th>
                  <th style="text-align: right">In-flight</th>
                  <th style="text-align: right">Completed</th>
                  <th style="text-align: right">Failed</th>
                  <th style="text-align: right">CPU</th>
                  <th style="text-align: right">Memory</th>
                </tr>
              </thead>
              <tbody>
                ${groupIds.map((gid) => {
                  const g = map[gid];
                  const childCount = Object.keys(g.workers ?? {}).length;
                  return html`
                    <>
                      <tr key=${gid} class="worker-group-row">
                        <td><span class="worker-id">${shortId(gid)} <span class="text-dim">(${g.readyWorkers ?? 0}/${g.declaredWorkers ?? childCount} ready)</span></span></td>
                        <td><span class=${'worker-status ' + safeStatusClass(g.status)}>${displayStatus(g)}</span></td>
                        <td style="text-align: right">${fmtInt(g.inFlight ?? 0)}</td>
                        <td style="text-align: right">${fmtInt(g.completed ?? 0)}</td>
                        <td style="text-align: right">${fmtInt(g.failed ?? 0)}</td>
                        <td style="text-align: right">${g.cpu != null ? fmtPercent(g.cpu) : '---'}</td>
                        <td style="text-align: right">${fmtBytes(g.memory)}</td>
                      </tr>
                      ${singleGroup && expanded
                        ? html`<${ChildTable} children=${g.workers ?? {}} />`
                        : ''
                      }
                    </>
                  `;
                })}
              </tbody>
            </table>
          </div>
        `
      }
    </div>
  `;
}
```

- [ ] **Step 5: Add minimal style for the toggle and child rows**

Append to `src/aiperf/api/static-v2/style.css`:

```css
/* ───── Worker group toggle ───── */
.group-toggle {
  cursor: pointer;
  user-select: none;
  display: inline-block;
  width: 14px;
  color: var(--muted);
}
.group-toggle:hover { color: var(--text); }
.worker-child-row td { background: rgba(255, 255, 255, 0.02); }
```

- [ ] **Step 6: Manual smoke check**

Start the local API + open the dashboard, run a small benchmark, and confirm:
- The Worker Groups card shows exactly one row (the local WGM) with non-zero CPU/memory/task counts.
- Clicking ▾ collapses the per-worker dropdown; clicking ▸ expands it again.

(If you cannot run the dev server in this environment, log this step as "not verified in CLI" — the unit tests cover the data layer.)

- [ ] **Step 7: Commit**

```bash
git add src/aiperf/api/static-v2/lib/state.js \
        src/aiperf/api/static-v2/lib/ws-dispatch.js \
        src/aiperf/api/static-v2/lib/ws.js \
        src/aiperf/api/static-v2/components/worker-table.js \
        src/aiperf/api/static-v2/style.css
git commit -s -m "feat(ui): web dashboard renders worker-group rows with per-child dropdown"
```

---

## Task 6: Textual dashboard adapts to group hook

**Files:**
- Modify: `src/aiperf/ui/dashboard/aiperf_dashboard_ui.py` — wire `ON_WORKER_GROUP_UPDATE`
- Modify: `src/aiperf/ui/dashboard/aiperf_textual_app.py` — add `on_worker_group_update` callback
- Modify: `src/aiperf/ui/dashboard/worker_dashboard.py` — adapt to group payload
- Modify: `src/aiperf/ui/dashboard/worker_status_table.py` — switch row key to group_id

- [ ] **Step 1: Read existing dashboard to confirm callback shape**

Open these files and note the existing function signatures:
- `src/aiperf/ui/dashboard/aiperf_textual_app.py:283-299`
- `src/aiperf/ui/dashboard/worker_dashboard.py:73-102`
- `src/aiperf/ui/dashboard/worker_status_table.py:37-102`

- [ ] **Step 2: Add the new hook wiring**

In `src/aiperf/ui/dashboard/aiperf_dashboard_ui.py`, alongside the existing `attach_hook(AIPerfHook.ON_WORKER_UPDATE, ...)` (around line 58), add:

```python
self.attach_hook(
    AIPerfHook.ON_WORKER_GROUP_UPDATE, self.app.on_worker_group_update
)
```

Keep the existing `ON_WORKER_UPDATE` and `ON_WORKER_STATUS_SUMMARY` attachments (they remain for the fake-in-process fallback).

- [ ] **Step 3: Add the callback on the textual app**

In `src/aiperf/ui/dashboard/aiperf_textual_app.py`, add a method paired with `on_worker_update`:

```python
def on_worker_group_update(self, group_id: str, group_stats: WorkerGroupStats) -> None:
    """Called whenever a group rolled-up snapshot lands."""
    self.call_from_thread(
        self.query_one(WorkerStatusTable).update_group, group_id, group_stats
    )
```

Add the import: `from aiperf.common.models import WorkerGroupStats`.

- [ ] **Step 4: Update `WorkerStatusTable` to row-by-group**

In `src/aiperf/ui/dashboard/worker_status_table.py`, change the column headers + row format. Replace the existing `update_worker_stats(worker_id, stats)` API path with `update_group(group_id, group_stats: WorkerGroupStats)`:

```python
from aiperf.common.models import WorkerGroupStats

class WorkerStatusTable(DataTable):
    DEFAULT_COLUMNS = [
        ("Group ID", 24),
        ("Status", 14),
        ("Ready", 10),
        ("In-flight", 10),
        ("Completed", 12),
        ("Failed", 10),
        ("CPU%", 8),
        ("Memory", 12),
    ]

    def update_group(self, group_id: str, group: WorkerGroupStats) -> None:
        ready = f"{group.ready_workers}/{group.declared_workers or len(group.workers)}"
        cpu = f"{group.health.cpu_usage:.1f}" if group.health else "—"
        mem = format_bytes(group.health.memory_usage) if group.health else "—"
        row = [
            shorten(group_id),
            str(group.status).replace("_", " "),
            ready,
            str(group.task_stats.in_progress),
            str(group.task_stats.completed),
            str(group.task_stats.failed),
            cpu,
            mem,
        ]
        if group_id in self._row_keys:
            self.update_row(self._row_keys[group_id], row)
        else:
            key = self.add_row(*row)
            self._row_keys[group_id] = key
```

(Keep the existing per-worker-id codepath as a no-op or remove it; the mixin still fires `ON_WORKER_UPDATE` only in fake-in-process tests.)

- [ ] **Step 5: Update worker_dashboard.py**

Replace the `on_worker_update`/`on_worker_status_summary` hook bodies to delegate to `WorkerStatusTable.update_group` via the synthetic `local` group, OR (simpler) leave them as no-ops — the new `on_worker_group_update` already covers both real WGM mode and the fake-in-process synthetic group.

```python
@on_worker_group_update
async def _on_group_update(
    self, group_id: str, group_stats: WorkerGroupStats
) -> None:
    """Update the on-screen table from a group snapshot."""
    self._table.update_group(group_id, group_stats)
```

- [ ] **Step 6: Run the unit suite**

```bash
uv run pytest tests/unit -n auto
```

Expected: full unit suite passes (or matches the baseline of pre-existing flaky tests already documented in memory; do not introduce new failures).

- [ ] **Step 7: Commit**

```bash
git add src/aiperf/ui/dashboard/aiperf_dashboard_ui.py \
        src/aiperf/ui/dashboard/aiperf_textual_app.py \
        src/aiperf/ui/dashboard/worker_dashboard.py \
        src/aiperf/ui/dashboard/worker_status_table.py
git commit -s -m "feat(ui-textual): row-per-group dashboard fed by ON_WORKER_GROUP_UPDATE"
```

---

## Task 7: Final unit-suite + lint pass

- [ ] **Step 1: Format + lint**

```bash
ruff format . && ruff check --fix .
```

- [ ] **Step 2: Run the full unit suite once**

```bash
uv run pytest tests/unit -n auto
```

Expected: PASS (or match pre-existing baseline). Investigate and fix any new failures.

- [ ] **Step 3: Run pre-commit on staged files**

```bash
pre-commit run
```

Expected: PASS.

- [ ] **Step 4: Commit any auto-fixups**

```bash
git add -A
git commit -s -m "chore: ruff/pre-commit fixups for worker-group-stats"
```

(Skip this step if there are no changes.)

---

## Out of scope

- **Cross-pod aggregation** in the operator UI (the K8s dashboard already has its own job-level summary; this plan only fixes the local dev web UI under `static-v2/`).
- **Renaming `/api/workers` → `/api/worker_groups`** — the user kept the path; only the shape changed.
- **Removing `WORKER_STATUS_SUMMARY` / `WORKER_POD_STATE`** — those still drive other listeners (operator dashboard, attach command); leave them alone.
- **WebSocket re-subscription on already-connected clients** — clients reload on disconnect; sufficient.
