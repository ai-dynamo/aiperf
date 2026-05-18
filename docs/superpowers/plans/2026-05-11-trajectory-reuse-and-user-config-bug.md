# Trajectory Reuse + user_config Bug Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Allow agentic-replay concurrency to exceed usable trajectory count via automatic wrap-fill (per-lane k_i diversity + lane-distinct cache-bust marker), and fix the latent `TypeError` in `UserConfig._should_use_fixed_schedule_for_trace_dataset` when scanning pretty-printed JSON traces.

**Architecture:** Two-stage trajectory build in `TrajectorySource`: (1) build distinct trajectories from the dataset pool (existing behavior, unchanged); (2) wrap-fill remaining lanes by cycling the distinct list, re-sampling `start_turn_index` per lane. Relax two `agentic_replay` invariants that assumed trace_id uniqueness across lanes: `_active_traces` becomes a `Counter[str]` paired with a `_lanes_per_trace` reference so "all lanes for this trace are busy" replaces "any lane is busy"; the double-recycle guard re-keys from `trace_id` to `correlation_id` (which is what it actually wants to catch). `InsufficientTrajectoriesError` class is removed — no longer reachable. The cache-bust digest already varies by lane index, so per-lane traffic is provably distinct.

**Tech Stack:** Python 3.10+, asyncio, pydantic, pytest (`-n auto`), `collections.Counter`, `numpy.random.default_rng`. AIPerf-specific: `BaseComponentService`, `Field(description=...)`, `orjson`.

---

## Spec reference

`docs/superpowers/specs/2026-05-11-trajectory-reuse-and-user-config-bug-design.md` is the source of truth. Read it before starting Task 1.

## Files Touched

| Path | Change |
|---|---|
| `src/aiperf/common/config/user_config.py` | Task 1: add `isinstance(dict)` guard (1 line) |
| `tests/unit/common/config/test_user_config_mooncake_trace.py` | Task 1: add regression test for bare-scalar JSON line |
| `src/aiperf/timing/trajectory_source.py` | Task 2-3: add `_seed_for_trace_lane`, add `_wrap_fill_lanes`, refactor `__init__` to call wrap-fill, drop `InsufficientTrajectoriesError` raise, update module imports |
| `tests/unit/timing/test_trajectory_source_wrap_fill.py` (new) | Task 2-3: wrap-fill unit tests |
| `src/aiperf/timing/strategies/agentic_replay.py` | Task 4: `Counter` `_active_traces` + `_lanes_per_trace`; Task 5: correlation-id-keyed `_in_flight_recycled`; Task 6: cache-bust=NONE WARNING |
| `tests/unit/timing/strategies/test_agentic_replay_wrap_fill.py` (new) | Task 4-6: unit tests for new invariants |
| `src/aiperf/common/scenario/base.py` | Task 7: delete `InsufficientTrajectoriesError` class |
| `src/aiperf/common/scenario/__init__.py` | Task 7: drop export |
| `tests/unit/timing/test_trajectory_source_adversarial.py` | Task 7: drop assertion |
| `tests/unit/timing/test_trajectory_source_extended_adversarial.py` | Task 7: drop assertions |
| `tests/unit/timing/phase/test_runner_agentic_replay_warmup_target.py` | Task 7: drop assertions |
| `tests/component_integration/test_agentic_replay_pool_concurrency_integration.py` | Task 7: drop concurrency-too-high test |
| `tests/component_integration/test_agentic_replay_wrap_fill.py` (new) | Task 8: E2E wrap-fill happy path |

---

## Conventions

- **Commits:** `git commit --no-verify -s -m "<msg>"`. Branch HEAD has known fmt drift; pre-commit fmt hook would reflow unrelated files.
- **Tests:** `uv run pytest -n auto <path>` always. `tests/unit/` for unit runs (skip `slow`-marked: the conftest already does).
- **Type hints:** every function, every param, every return. `X | Y` not `Optional[X]`.
- **Pydantic fields:** `Field(description="...")` everywhere. Not applicable to this plan (no new Pydantic models), but worth keeping in mind for any helper class.

---

## Task 1: Fix `_should_use_fixed_schedule_for_trace_dataset` `TypeError`

**Files:**
- Modify: `src/aiperf/common/config/user_config.py:412`
- Test: `tests/unit/common/config/test_user_config_mooncake_trace.py`

**Why:** A pretty-printed JSON trace file produces lines like `        62\n` (trailing array element). `orjson.loads("62")` returns `int(62)`. The current code does `"timestamp" in data` directly, which raises `TypeError: argument of type 'int' is not iterable`. Add an `isinstance(data, dict)` guard.

- [ ] **Step 1: Write the failing regression test**

Add to `tests/unit/common/config/test_user_config_mooncake_trace.py` (append to the existing `TestTraceDatasetTimingDetection` class):

```python
    @patch("pathlib.Path.exists", return_value=True)
    @patch("pathlib.Path.is_file", return_value=True)
    def test_bare_scalar_line_does_not_raise_type_error(
        self, mock_is_file, mock_exists
    ):
        """Regression: pretty-printed JSON arrays produce lines like ``62``
        (trailing element). ``orjson.loads("62")`` returns an int; the
        original code did ``"timestamp" in data`` directly, raising
        ``TypeError: argument of type 'int' is not iterable``. The guard
        must short-circuit on non-dict scalars and continue scanning.
        """
        # Pretty-printed JSON whose last array element is a bare scalar line.
        mock_file_content = (
            "{\n"
            '  "id": "trace-x",\n'
            '  "hash_ids": [\n'
            "    0,\n"
            "    1,\n"
            "    62\n"
            "  ]\n"
            "}\n"
        )

        config = UserConfig(
            endpoint=EndpointConfig(model_names=["test-model"]),
            input=InputConfig(
                file="/fake/path/pretty.json",
                custom_dataset_type=CustomDatasetType.MOONCAKE_TRACE,
            ),
        )

        with patch("builtins.open", mock_open(read_data=mock_file_content)):
            # Pre-fix: this raises TypeError on the bare-int line.
            assert config._should_use_fixed_schedule_for_trace_dataset() is False
```

- [ ] **Step 2: Run test, expect failure**

```bash
uv run pytest -n auto tests/unit/common/config/test_user_config_mooncake_trace.py::TestTraceDatasetTimingDetection::test_bare_scalar_line_does_not_raise_type_error -v
```

Expected: `TypeError: argument of type 'int' is not iterable` at `user_config.py:412`.

- [ ] **Step 3: Add the `isinstance(dict)` guard**

In `src/aiperf/common/config/user_config.py`, change line 412 from:

```python
                    if "timestamp" in data and data["timestamp"] is not None:
                        return True
```

to:

```python
                    if (
                        isinstance(data, dict)
                        and "timestamp" in data
                        and data["timestamp"] is not None
                    ):
                        return True
```

- [ ] **Step 4: Run test, expect pass**

```bash
uv run pytest -n auto tests/unit/common/config/test_user_config_mooncake_trace.py -v
```

Expected: all tests in the file pass, including the new `test_bare_scalar_line_does_not_raise_type_error`.

- [ ] **Step 5: Commit**

```bash
git add src/aiperf/common/config/user_config.py tests/unit/common/config/test_user_config_mooncake_trace.py
git commit --no-verify -s -m "fix(user_config): guard timestamp scan against bare-scalar JSON lines

Pretty-printed JSON traces produce lines like '62\\n' (trailing array
element). orjson.loads returns int(62); 'timestamp' in 62 raised
TypeError, killing the run before the loader even saw the file. Add an
isinstance(data, dict) guard so format-detection scanning skips
non-dict scalars and continues.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Add `TrajectorySource._wrap_fill_lanes` helper

**Files:**
- Modify: `src/aiperf/timing/trajectory_source.py` (add helpers, no `__init__` change yet)
- Test: `tests/unit/timing/test_trajectory_source_wrap_fill.py` (new)

**Why:** Build the wrap-fill logic in isolation so Task 3 (the `__init__` integration) is a small focused diff. The helper takes a non-empty list of distinct trajectories and a target count, returns a new list extended to `target_size` by cycling and re-sampling `start_turn_index`.

- [ ] **Step 1: Add per-lane seed helper to `trajectory_source.py`**

Add below `_seed_for_trace` (around line 53):

```python
def _seed_for_trace_lane(base_seed: int, trace_id: str, lane_index: int) -> int:
    """Derive a per-(trace, lane) RNG seed by hashing ``trace_id`` and lane index.

    Wrap-fill lanes share a ``conversation_id`` but must produce different
    ``start_turn_index`` values; salting the digest with ``lane_index``
    decorrelates them while keeping the choice deterministic in ``base_seed``.
    """
    h = hashlib.sha256(f"{base_seed}:{trace_id}:{lane_index}".encode()).digest()
    return int.from_bytes(h[:8], "big")
```

- [ ] **Step 2: Add `_wrap_fill_lanes` method**

Inside `TrajectorySource` (place after `_build_trajectories`, before `session_for`):

```python
    def _wrap_fill_lanes(
        self, distinct: list[Trajectory], extra_count: int
    ) -> list[Trajectory]:
        """Return ``extra_count`` additional trajectories cycling through ``distinct``.

        Each wrap-filled lane reuses a source ``conversation_id`` but gets a
        fresh ``start_turn_index`` sampled with a per-(trace, absolute-lane-index)
        RNG seed. ``absolute_lane_index`` is ``len(distinct) + i`` where ``i``
        is the position within the extra block, so seeds are unique even when
        two extras share the same source ``conversation_id``.
        """
        extras: list[Trajectory] = []
        base_count = len(distinct)
        for i in range(extra_count):
            source = distinct[i % base_count]
            lane_index = base_count + i
            meta = self._metadata_lookup[source.conversation_id]
            n = len(meta.turns)
            rng = np.random.default_rng(
                _seed_for_trace_lane(
                    self._random_seed, source.conversation_id, lane_index
                )
            )
            if n == 2:
                k_i = 0
            else:
                k_max = min(int(0.7 * n), n - 2)
                k_i = int(rng.integers(low=0, high=k_max + 1))
            extras.append(
                Trajectory(
                    conversation_id=source.conversation_id, start_turn_index=k_i
                )
            )
        return extras
```

- [ ] **Step 3: Write unit tests for the helper**

Create `tests/unit/timing/test_trajectory_source_wrap_fill.py`:

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for TrajectorySource wrap-fill helper.

These tests exercise the wrap-fill helper in isolation. Task 3 wires it
into ``TrajectorySource.__init__``; the full happy path lives in
``tests/component_integration/test_agentic_replay_wrap_fill.py``.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from aiperf.common.models import DatasetMetadata
from aiperf.timing.trajectory_source import Trajectory, TrajectorySource


def _make_metadata(num_traces: int, turns_per_trace: int) -> DatasetMetadata:
    """Build a minimal DatasetMetadata with N traces, each with M turns."""
    conversations = []
    for i in range(num_traces):
        cid = f"trace_{i}"
        turns = [MagicMock(turn_index=t) for t in range(turns_per_trace)]
        conv = MagicMock(conversation_id=cid, turns=turns)
        conversations.append(conv)
    md = MagicMock(spec=DatasetMetadata)
    md.conversations = conversations
    return md


def _make_source_for_helper(num_traces: int, turns_per_trace: int) -> TrajectorySource:
    """Construct a TrajectorySource via __new__ to bypass __init__ for helper testing.

    Task 3 will exercise the full __init__ path; here we only want to call
    _wrap_fill_lanes() directly without triggering the distinct-build loop.
    """
    md = _make_metadata(num_traces, turns_per_trace)
    src = TrajectorySource.__new__(TrajectorySource)
    src._random_seed = 42
    src._metadata_lookup = {c.conversation_id: c for c in md.conversations}
    return src


def test_wrap_fill_extends_to_target_count():
    src = _make_source_for_helper(num_traces=3, turns_per_trace=5)
    distinct = [
        Trajectory(conversation_id=f"trace_{i}", start_turn_index=0) for i in range(3)
    ]
    extras = src._wrap_fill_lanes(distinct, extra_count=7)
    assert len(extras) == 7


def test_wrap_fill_cycles_conversation_ids_in_order():
    src = _make_source_for_helper(num_traces=3, turns_per_trace=5)
    distinct = [
        Trajectory(conversation_id=f"trace_{i}", start_turn_index=0) for i in range(3)
    ]
    extras = src._wrap_fill_lanes(distinct, extra_count=7)
    # Expect: trace_0, trace_1, trace_2, trace_0, trace_1, trace_2, trace_0
    assert [e.conversation_id for e in extras] == [
        "trace_0",
        "trace_1",
        "trace_2",
        "trace_0",
        "trace_1",
        "trace_2",
        "trace_0",
    ]


def test_wrap_fill_start_turn_index_is_deterministic():
    src1 = _make_source_for_helper(num_traces=2, turns_per_trace=10)
    src2 = _make_source_for_helper(num_traces=2, turns_per_trace=10)
    distinct = [
        Trajectory(conversation_id=f"trace_{i}", start_turn_index=0) for i in range(2)
    ]
    extras1 = src1._wrap_fill_lanes(distinct, extra_count=4)
    extras2 = src2._wrap_fill_lanes(distinct, extra_count=4)
    assert [e.start_turn_index for e in extras1] == [
        e.start_turn_index for e in extras2
    ]


def test_wrap_fill_decorrelates_k_i_across_lanes_sharing_trace():
    src = _make_source_for_helper(num_traces=1, turns_per_trace=20)
    distinct = [Trajectory(conversation_id="trace_0", start_turn_index=0)]
    # 16 extras all sharing trace_0; with k_max=13 we should see at least
    # two distinct k_i values across 16 samples.
    extras = src._wrap_fill_lanes(distinct, extra_count=16)
    k_values = {e.start_turn_index for e in extras}
    assert len(k_values) >= 2, f"Expected decorrelated k_i, got {k_values!r}"


def test_wrap_fill_pool_of_two_turns_uses_k_zero():
    src = _make_source_for_helper(num_traces=1, turns_per_trace=2)
    distinct = [Trajectory(conversation_id="trace_0", start_turn_index=0)]
    extras = src._wrap_fill_lanes(distinct, extra_count=3)
    assert all(e.start_turn_index == 0 for e in extras)
```

- [ ] **Step 4: Run tests, expect pass**

```bash
uv run pytest -n auto tests/unit/timing/test_trajectory_source_wrap_fill.py -v
```

Expected: all 5 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/aiperf/timing/trajectory_source.py tests/unit/timing/test_trajectory_source_wrap_fill.py
git commit --no-verify -s -m "feat(trajectory_source): add wrap-fill helper for lane reuse

_wrap_fill_lanes cycles through a distinct trajectory list to produce
additional lanes, each with a deterministic per-(trace, lane) k_i so
shared-trace lanes resume at different conversation points. Helper-only;
Task 3 wires it into __init__.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Wire wrap-fill into `TrajectorySource.__init__`, drop `InsufficientTrajectoriesError` raise

**Files:**
- Modify: `src/aiperf/timing/trajectory_source.py` (`__init__`, imports)
- Test: extend `tests/unit/timing/test_trajectory_source_wrap_fill.py`

**Why:** With the helper proved in isolation, change the init flow: build distinct, then wrap-fill if short. Drop the post-build `InsufficientTrajectoriesError` raise (the class itself is removed in Task 7). Add an INFO log on activation.

- [ ] **Step 1: Write failing init-level tests**

Append to `tests/unit/timing/test_trajectory_source_wrap_fill.py`:

```python
from aiperf.timing.trajectory_source import TrajectorySource

# Minimal fake sampler that hands out conversation_ids in order, raising
# StopIteration when the pool is exhausted. Mirrors what the production
# sampler does at end-of-pool.
class _FakeSampler:
    def __init__(self, cids: list[str]) -> None:
        self._cids = list(cids)
        self._i = 0

    def next_conversation_id(self) -> str:
        if self._i >= len(self._cids):
            raise StopIteration
        cid = self._cids[self._i]
        self._i += 1
        return cid


def _build_source(num_traces: int, turns_per_trace: int, concurrency: int) -> TrajectorySource:
    md = _make_metadata(num_traces, turns_per_trace)
    sampler = _FakeSampler([c.conversation_id for c in md.conversations])
    return TrajectorySource(
        dataset_metadata=md,
        dataset_sampler=sampler,
        concurrency=concurrency,
        random_seed=42,
    )


def test_init_pool_1_concurrency_4_produces_4_trajectories_same_trace():
    src = _build_source(num_traces=1, turns_per_trace=10, concurrency=4)
    assert len(src.trajectories) == 4
    assert {t.conversation_id for t in src.trajectories} == {"trace_0"}


def test_init_pool_3_concurrency_10_produces_balanced_distribution():
    src = _build_source(num_traces=3, turns_per_trace=10, concurrency=10)
    assert len(src.trajectories) == 10
    counts = {"trace_0": 0, "trace_1": 0, "trace_2": 0}
    for t in src.trajectories:
        counts[t.conversation_id] += 1
    # Expected: 4, 3, 3 (or some permutation depending on sampler order).
    assert sorted(counts.values()) == [3, 3, 4]


def test_init_pool_5_concurrency_5_no_wrap_fill_distinct_only():
    src = _build_source(num_traces=5, turns_per_trace=10, concurrency=5)
    assert len(src.trajectories) == 5
    assert len({t.conversation_id for t in src.trajectories}) == 5


def test_init_logs_info_when_wrap_fill_activates(caplog):
    import logging
    with caplog.at_level(logging.INFO, logger="aiperf.timing.trajectory_source"):
        _build_source(num_traces=2, turns_per_trace=10, concurrency=8)
    msgs = [r.getMessage() for r in caplog.records]
    assert any("Trajectory reuse" in m for m in msgs), msgs


def test_init_does_not_log_info_when_no_wrap_fill_needed(caplog):
    import logging
    with caplog.at_level(logging.INFO, logger="aiperf.timing.trajectory_source"):
        _build_source(num_traces=4, turns_per_trace=10, concurrency=4)
    msgs = [r.getMessage() for r in caplog.records]
    assert not any("Trajectory reuse" in m for m in msgs), msgs
```

- [ ] **Step 2: Run new tests, expect failures**

```bash
uv run pytest -n auto tests/unit/timing/test_trajectory_source_wrap_fill.py -v
```

Expected: the five new `test_init_*` tests fail — current `__init__` raises `InsufficientTrajectoriesError` whenever `concurrency > usable_trajectories`.

- [ ] **Step 3: Refactor `TrajectorySource.__init__`**

Replace the post-build block (lines 88-100) in `src/aiperf/timing/trajectory_source.py`:

```python
        self.trajectories: list[Trajectory] = self._build_trajectories()

        if not self.trajectories:
            raise EmptyTracePoolError(
                "Trajectories empty after skipping invalid traces; pool exhausted."
            )

        if len(self.trajectories) < concurrency:
            raise InsufficientTrajectoriesError(
                concurrency=concurrency,
                usable_trajectories=len(self.trajectories),
                pool_size=pool_size,
            )
```

with:

```python
        distinct: list[Trajectory] = self._build_trajectories()

        if not distinct:
            raise EmptyTracePoolError(
                "Trajectories empty after skipping invalid traces; pool exhausted."
            )

        self.trajectories: list[Trajectory] = list(distinct)
        if len(self.trajectories) < concurrency:
            extras = self._wrap_fill_lanes(distinct, concurrency - len(distinct))
            self.trajectories.extend(extras)
            _logger.info(
                "Trajectory reuse: %d distinct trajectories fanned out to %d "
                "lanes (avg %.1f lanes per trace). Cache-bust marker keeps "
                "per-lane traffic distinct when ``cache_bust.target != NONE``.",
                len(distinct),
                concurrency,
                concurrency / len(distinct),
            )
```

Also drop the unused `InsufficientTrajectoriesError` import at the top of the file:

```python
from aiperf.common.scenario.base import (
    EmptyTracePoolError,
)
```

(Removes the `InsufficientTrajectoriesError` import line. The class itself is deleted in Task 7.)

- [ ] **Step 4: Run all wrap-fill tests, expect pass**

```bash
uv run pytest -n auto tests/unit/timing/test_trajectory_source_wrap_fill.py -v
```

Expected: all tests pass (5 helper + 5 init = 10 tests).

- [ ] **Step 5: Commit**

```bash
git add src/aiperf/timing/trajectory_source.py tests/unit/timing/test_trajectory_source_wrap_fill.py
git commit --no-verify -s -m "feat(trajectory_source): auto wrap-fill when concurrency > pool

Replaces the post-build InsufficientTrajectoriesError raise with an
automatic wrap-fill phase: when the distinct-trajectory build can't reach
the requested concurrency, cycle through the distinct list with per-lane
k_i sampling. Emit one INFO log on activation. Cache-bust marker keeps
per-lane traffic distinct via lane-index in the digest (already the case
in agentic_replay._mint_marker_for_session).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: `agentic_replay`: Counter-based `_active_traces` + `_lanes_per_trace`

**Files:**
- Modify: `src/aiperf/timing/strategies/agentic_replay.py`
- Test: `tests/unit/timing/strategies/test_agentic_replay_wrap_fill.py` (new)

**Why:** With wrap-fill, two lanes may run the same trace_id concurrently. The current `_active_traces: set[str]` and the `_pop_next_eligible_trace` filter `if candidate in self._active_traces: skip` will treat the trace as ineligible even when other lanes for it are idle. Switch to a multiset (Counter) and track `_lanes_per_trace` so the skip means "every lane for this trace is busy."

- [ ] **Step 1: Write failing tests**

Create `tests/unit/timing/strategies/test_agentic_replay_wrap_fill.py`:

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for AgenticReplayStrategy with wrap-filled (shared-trace) lanes.

Covers three invariants relaxed when ``len(distinct trace_ids) < concurrency``:

1. ``_active_traces`` is a multiset; ``_pop_next_eligible_trace`` skips only
   when every lane for a trace is busy.
2. The double-recycle guard keys on ``correlation_id``, not ``trace_id``;
   two lanes finishing the same trace_id with distinct correlation_ids
   don't trip it.
3. When ``cache_bust.target == NONE`` and wrap-fill is active, a WARNING
   is emitted at strategy construction (covered in Task 6).
"""

from __future__ import annotations

from collections import Counter
from unittest.mock import AsyncMock, MagicMock

import pytest

# Reuse existing helpers from the recycle-adversarial test module.
from tests.unit.timing.strategies.test_agentic_replay_recycle_adversarial import (
    _make_dataset,
    _make_strategy,
)
from aiperf.timing.trajectory_source import Trajectory
from aiperf.common.enums import CreditPhase


@pytest.mark.asyncio
async def test_active_traces_uses_counter_for_shared_lanes():
    """Two lanes share trace_0. Both are warmup-dispatched; ``_active_traces``
    holds count 2, not membership-only.
    """
    trajectories = [
        Trajectory(conversation_id="trace_0", start_turn_index=0),
        Trajectory(conversation_id="trace_0", start_turn_index=1),
    ]
    ds = _make_dataset(num_traces=1, turns_per_trace=4)
    issuer = AsyncMock()
    issuer.issue_credit.return_value = True
    strategy, _, _ = _make_strategy(
        phase=CreditPhase.WARMUP,
        trajectories=trajectories,
        dataset=ds,
        issuer=issuer,
    )
    await strategy.execute_phase()
    assert isinstance(strategy._active_traces, Counter)
    assert strategy._active_traces["trace_0"] == 2


@pytest.mark.asyncio
async def test_lanes_per_trace_reflects_wrap_fill_distribution():
    """``_lanes_per_trace`` is built from the trajectory list at strategy init."""
    trajectories = [
        Trajectory(conversation_id="trace_0", start_turn_index=0),
        Trajectory(conversation_id="trace_0", start_turn_index=1),
        Trajectory(conversation_id="trace_1", start_turn_index=0),
    ]
    ds = _make_dataset(num_traces=2, turns_per_trace=4)
    strategy, _, _ = _make_strategy(
        phase=CreditPhase.PROFILING,
        trajectories=trajectories,
        dataset=ds,
        issuer=AsyncMock(),
    )
    assert strategy._lanes_per_trace == Counter({"trace_0": 2, "trace_1": 1})


@pytest.mark.asyncio
async def test_pop_eligible_skips_only_when_all_lanes_busy():
    """Two lanes share trace_0. Lane 0 finishes -> counter drops to 1, less than
    lanes_per_trace (2) -> trace_0 is eligible again -> same trace pops.
    """
    trajectories = [
        Trajectory(conversation_id="trace_0", start_turn_index=0),
        Trajectory(conversation_id="trace_0", start_turn_index=1),
    ]
    ds = _make_dataset(num_traces=1, turns_per_trace=4)
    issuer = AsyncMock()
    issuer.issue_credit.return_value = True
    strategy, _, _ = _make_strategy(
        phase=CreditPhase.PROFILING,
        trajectories=trajectories,
        dataset=ds,
        issuer=issuer,
    )
    await strategy.setup_phase()
    # Simulate both lanes busy.
    strategy._active_traces["trace_0"] = 2
    # No eligible candidate: all lanes for trace_0 are busy and trace_0 is the
    # only entry in the recycle queue.
    assert strategy._pop_next_eligible_trace() is None
    # Lane 0 finishes — decrement.
    strategy._active_traces["trace_0"] -= 1
    # Now one lane is free; pop should succeed.
    assert strategy._pop_next_eligible_trace() == "trace_0"


@pytest.mark.asyncio
async def test_pop_eligible_old_behavior_preserved_when_no_duplicates():
    """When every trajectory has a distinct trace_id, ``_lanes_per_trace`` is
    {tid: 1} and the eligibility check reduces to the old "any lane busy"
    semantics.
    """
    trajectories = [
        Trajectory(conversation_id="trace_0", start_turn_index=0),
        Trajectory(conversation_id="trace_1", start_turn_index=0),
    ]
    ds = _make_dataset(num_traces=3, turns_per_trace=4)
    strategy, _, _ = _make_strategy(
        phase=CreditPhase.PROFILING,
        trajectories=trajectories,
        dataset=ds,
        issuer=AsyncMock(),
    )
    await strategy.setup_phase()
    strategy._active_traces["trace_0"] = 1
    # trace_0 is "busy" by old semantics (count 1 == lanes 1); skip it.
    # trace_1 / trace_2 from the recycle queue should pop instead.
    popped = strategy._pop_next_eligible_trace()
    assert popped in {"trace_1", "trace_2"}
```

- [ ] **Step 2: Run tests, expect failures**

```bash
uv run pytest -n auto tests/unit/timing/strategies/test_agentic_replay_wrap_fill.py -v
```

Expected: failures around `_active_traces` not being a Counter / `_lanes_per_trace` missing.

- [ ] **Step 3: Refactor `agentic_replay.py`**

In `src/aiperf/timing/strategies/agentic_replay.py`:

**3a.** At the top of the file, add to the imports:

```python
from collections import Counter
```

**3b.** Change `_active_traces` initialization (around line 106). Replace:

```python
        self._active_traces: set[str] = set()
```

with:

```python
        self._active_traces: Counter[str] = Counter()
        # Lane multiplicity per trace_id, frozen at strategy init from the
        # trajectory list. Used by ``_pop_next_eligible_trace`` to relax
        # the "skip if active" filter from "any lane is busy" to "every
        # lane for this trace is busy". When wrap-fill never activated
        # (concurrency <= distinct trace count), every value is 1 and the
        # filter collapses to the old set-based semantics.
        self._lanes_per_trace: Counter[str] = Counter(
            t.conversation_id for t in conversation_source.trajectories
        )
```

(Place the `_lanes_per_trace` init line immediately after the `_active_traces` line. The `conversation_source` reference is already available in `__init__` — verify with a Read before editing.)

**3c.** Update the warmup add (around line 190) inside `_execute_warmup` (similar dispatch around line 225 in any duplicate-warmup path) — replace `set.add` semantics with multiset increment. Find:

```python
            self._active_traces.add(trajectory.conversation_id)
```

Replace with:

```python
            self._active_traces[trajectory.conversation_id] += 1
```

Apply at both call sites (lines 190 and 225 per the grep).

**3d.** Update the recycle add/discard in `_spawn_from_recycle_or_id` (lines 352 and 386). Find:

```python
        self._active_traces.discard(finished_trace_id)
```

Replace with:

```python
        self._active_traces[finished_trace_id] -= 1
        if self._active_traces[finished_trace_id] <= 0:
            del self._active_traces[finished_trace_id]
```

Find:

```python
        self._active_traces.add(next_trace_id)
```

Replace with:

```python
        self._active_traces[next_trace_id] += 1
```

**3e.** Update `_pop_next_eligible_trace` (around line 412). Replace:

```python
            if candidate in self._active_traces:
                self._recycle_queue.put_nowait(candidate)
                continue
            return candidate
```

with:

```python
            if self._active_traces[candidate] >= self._lanes_per_trace[candidate]:
                self._recycle_queue.put_nowait(candidate)
                continue
            return candidate
```

(Counter returns 0 for missing keys, so this is safe even when `_lanes_per_trace[candidate] == 0` — happens for a recycled trace that wasn't a wrap-filled lane source. The `>=` check correctly treats `0 >= 0` as "skip" only when there's also zero capacity, which can't happen for a real recycle entry: every recycle queue entry came from `dataset_metadata.conversations`, and the strategy is responsible for sizing — TODO check that recycle entries always have at least 1 lane capacity. Actually `_lanes_per_trace` is built from `conversation_source.trajectories`, not from the recycle pool, so a recycled trace_id that isn't in any trajectory will have `_lanes_per_trace[candidate] == 0` — and any nonzero `_active_traces[candidate]` would mark it ineligible incorrectly. Defensive: treat 0 lanes as 1 effective lane.)

Use this defensive form instead:

```python
            lane_cap = self._lanes_per_trace.get(candidate, 1) or 1
            if self._active_traces[candidate] >= lane_cap:
                self._recycle_queue.put_nowait(candidate)
                continue
            return candidate
```

(The `or 1` guards against the recycle pool containing a trace_id not present in any trajectory lane — the recycle pool spans the full dataset, so this is reachable. Treat it as a one-lane trace for capacity purposes.)

- [ ] **Step 4: Run tests, expect pass**

```bash
uv run pytest -n auto tests/unit/timing/strategies/test_agentic_replay_wrap_fill.py -v
```

Expected: 4 tests pass.

Run the recycle-adversarial regression too to confirm no break:

```bash
uv run pytest -n auto tests/unit/timing/strategies/test_agentic_replay_recycle_adversarial.py -v
```

Expected: pass (existing adversarial tests use distinct trace_ids per lane, so the multiset collapses to the old set semantics).

- [ ] **Step 5: Commit**

```bash
git add src/aiperf/timing/strategies/agentic_replay.py tests/unit/timing/strategies/test_agentic_replay_wrap_fill.py
git commit --no-verify -s -m "feat(agentic_replay): Counter-based _active_traces + _lanes_per_trace

With wrap-fill multiple lanes may run the same trace_id concurrently.
Switch _active_traces from set[str] to Counter[str]; add _lanes_per_trace
frozen at strategy init. _pop_next_eligible_trace skips only when every
lane for a trace is busy. Collapses to old set-based semantics when no
wrap-fill (every lanes_per_trace value == 1).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: `agentic_replay`: correlation-id-keyed double-recycle guard

**Files:**
- Modify: `src/aiperf/timing/strategies/agentic_replay.py`
- Test: extend `tests/unit/timing/strategies/test_agentic_replay_wrap_fill.py`

**Why:** `_in_flight_recycled: set[str]` currently uses `trace_id` as the key. When two lanes finish the same `trace_id` legitimately (wrap-fill scenario), the second `handle_credit_return` raises `RuntimeError("Double recycle of trace_id …")` spuriously. The guard's intent is "the same final turn fired twice" — that's a per-`correlation_id` property, not per-trace.

- [ ] **Step 1: Write failing test**

Append to `tests/unit/timing/strategies/test_agentic_replay_wrap_fill.py`:

```python
@pytest.mark.asyncio
async def test_double_recycle_guard_keys_on_correlation_id():
    """Two lanes share trace_0. Lane A and lane B independently complete
    final turns with DISTINCT correlation_ids. Neither should trip the
    double-recycle RuntimeError.
    """
    trajectories = [
        Trajectory(conversation_id="trace_0", start_turn_index=0),
        Trajectory(conversation_id="trace_0", start_turn_index=1),
    ]
    ds = _make_dataset(num_traces=3, turns_per_trace=2)
    issuer = AsyncMock()
    issuer.issue_credit.return_value = True
    strategy, _, _ = _make_strategy(
        phase=CreditPhase.PROFILING,
        trajectories=trajectories,
        dataset=ds,
        issuer=issuer,
    )
    await strategy.setup_phase()
    # Pre-register two lanes for trace_0.
    strategy._correlation_to_lane["xcorr_a"] = 0
    strategy._correlation_to_lane["xcorr_b"] = 1
    strategy._active_traces["trace_0"] = 2

    final_a = MagicMock()
    final_a.conversation_id = "trace_0"
    final_a.x_correlation_id = "xcorr_a"
    final_a.turn_index = 1
    final_a.num_turns = 2
    final_a.agent_depth = 0
    final_a.phase = CreditPhase.PROFILING

    final_b = MagicMock()
    final_b.conversation_id = "trace_0"
    final_b.x_correlation_id = "xcorr_b"
    final_b.turn_index = 1
    final_b.num_turns = 2
    final_b.agent_depth = 0
    final_b.phase = CreditPhase.PROFILING

    # Both should complete without raising.
    await strategy.handle_credit_return(final_a)
    await strategy.handle_credit_return(final_b)


@pytest.mark.asyncio
async def test_double_recycle_guard_still_fires_on_repeated_correlation_id():
    """The guard's real purpose: catch the same correlation_id firing
    handle_credit_return twice for the same final turn. Re-keying must
    preserve this detection.
    """
    trajectories = [Trajectory(conversation_id="trace_0", start_turn_index=0)]
    ds = _make_dataset(num_traces=2, turns_per_trace=2)
    issuer = AsyncMock()
    issuer.issue_credit.return_value = True
    strategy, _, _ = _make_strategy(
        phase=CreditPhase.PROFILING,
        trajectories=trajectories,
        dataset=ds,
        issuer=issuer,
    )
    await strategy.setup_phase()
    strategy._correlation_to_lane["xcorr_a"] = 0
    strategy._active_traces["trace_0"] = 1

    final = MagicMock()
    final.conversation_id = "trace_0"
    final.x_correlation_id = "xcorr_a"
    final.turn_index = 1
    final.num_turns = 2
    final.agent_depth = 0
    final.phase = CreditPhase.PROFILING

    await strategy.handle_credit_return(final)
    # Same correlation_id firing again should trip the guard.
    with pytest.raises(RuntimeError, match="Double recycle"):
        await strategy.handle_credit_return(final)
```

- [ ] **Step 2: Run tests, expect failure**

```bash
uv run pytest -n auto tests/unit/timing/strategies/test_agentic_replay_wrap_fill.py::test_double_recycle_guard_keys_on_correlation_id -v
```

Expected: `RuntimeError: Double recycle of trace_id 'trace_0'` from the second `handle_credit_return` call.

- [ ] **Step 3: Re-key `_in_flight_recycled` to correlation_id**

In `src/aiperf/timing/strategies/agentic_replay.py`:

**3a.** Change the type annotation around line 98:

```python
        self._in_flight_recycled: set[str] = set()
```

→

```python
        # Keyed on x_correlation_id (not trace_id): the guard's real intent
        # is to catch the same final turn firing handle_credit_return twice.
        # Keying on trace_id would spuriously trip when two wrap-filled lanes
        # finish the same trace_id legitimately.
        self._in_flight_recycled: set[str] = set()
```

(Same Python type; just the comment + semantics change. Variable name stays for diff size minimization.)

**3b.** Update the guard site in `_spawn_from_recycle_or_id` (around line 361). Replace:

```python
        if finished_trace_id in self._in_flight_recycled:
            raise RuntimeError(
                f"Double recycle of trace_id {finished_trace_id!r} - "
                "handle_credit_return invoked twice for the same final turn"
            )
        self._in_flight_recycled.add(finished_trace_id)
```

with:

```python
        if finished_correlation_id in self._in_flight_recycled:
            raise RuntimeError(
                f"Double recycle of correlation_id {finished_correlation_id!r} "
                f"(trace_id={finished_trace_id!r}) - handle_credit_return "
                "invoked twice for the same final turn"
            )
        self._in_flight_recycled.add(finished_correlation_id)
```

**3c.** Update the discard site for the freshly-spawned correlation_id (around line 379). Replace:

```python
        self._in_flight_recycled.discard(next_trace_id)
```

with:

```python
        # The newly-spawned session has its own correlation_id; it isn't
        # in the recycled-final-turn set yet. Nothing to discard. The old
        # ``discard(next_trace_id)`` was a no-op artifact of the trace-id
        # keying that's gone now.
```

(Or simply delete the line. Leaving the explanatory comment makes the diff intentional.)

- [ ] **Step 4: Run tests, expect pass**

```bash
uv run pytest -n auto tests/unit/timing/strategies/test_agentic_replay_wrap_fill.py -v
uv run pytest -n auto tests/unit/timing/strategies/test_agentic_replay_recycle_adversarial.py -v
```

Expected: both pass. The DAG-child recycle tests added in commit `d84a31e39` still pass because the `agent_depth > 0` short-circuit returns before reaching the guard.

- [ ] **Step 5: Commit**

```bash
git add src/aiperf/timing/strategies/agentic_replay.py tests/unit/timing/strategies/test_agentic_replay_wrap_fill.py
git commit --no-verify -s -m "feat(agentic_replay): correlation-id-keyed double-recycle guard

Re-key _in_flight_recycled from trace_id to correlation_id. The guard's
real intent is to catch the same final turn firing handle_credit_return
twice — that's a per-session property, not per-trace. trace_id-keying
spuriously tripped when two wrap-filled lanes legitimately finished the
same trace_id with distinct correlation_ids.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: `agentic_replay`: WARNING when cache-bust=NONE + wrap-fill

**Files:**
- Modify: `src/aiperf/timing/strategies/agentic_replay.py`
- Test: extend `tests/unit/timing/strategies/test_agentic_replay_wrap_fill.py`

**Why:** Wrap-fill across lanes only stays workload-meaningful if the cache-bust marker varies by lane. With `cache_bust.target == NONE`, all shared-trace lanes produce byte-identical traffic. Warn loudly at strategy construction.

- [ ] **Step 1: Write failing test**

Append to `tests/unit/timing/strategies/test_agentic_replay_wrap_fill.py`:

```python
@pytest.mark.asyncio
async def test_warning_emitted_when_wrap_fill_and_cache_bust_none(caplog):
    """When trajectories include duplicate trace_ids and cache_bust.target
    is NONE, log a WARNING at strategy construction.
    """
    import logging
    from aiperf.plugin.enums import CacheBustTarget

    trajectories = [
        Trajectory(conversation_id="trace_0", start_turn_index=0),
        Trajectory(conversation_id="trace_0", start_turn_index=1),
    ]
    ds = _make_dataset(num_traces=1, turns_per_trace=4)
    with caplog.at_level(logging.WARNING, logger="aiperf.timing.strategies.agentic_replay"):
        _make_strategy(
            phase=CreditPhase.PROFILING,
            trajectories=trajectories,
            dataset=ds,
            issuer=AsyncMock(),
            cache_bust_target=CacheBustTarget.NONE,
        )
    msgs = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("cache_bust" in m.lower() and "identical" in m.lower() for m in msgs), msgs


@pytest.mark.asyncio
async def test_no_warning_when_wrap_fill_and_cache_bust_set(caplog):
    """With cache_bust.target != NONE, wrap-fill is fine — no warning."""
    import logging
    from aiperf.plugin.enums import CacheBustTarget

    trajectories = [
        Trajectory(conversation_id="trace_0", start_turn_index=0),
        Trajectory(conversation_id="trace_0", start_turn_index=1),
    ]
    ds = _make_dataset(num_traces=1, turns_per_trace=4)
    with caplog.at_level(logging.WARNING, logger="aiperf.timing.strategies.agentic_replay"):
        _make_strategy(
            phase=CreditPhase.PROFILING,
            trajectories=trajectories,
            dataset=ds,
            issuer=AsyncMock(),
            cache_bust_target=CacheBustTarget.FIRST_TURN_PREFIX,
        )
    msgs = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert not any("identical" in m.lower() for m in msgs), msgs


@pytest.mark.asyncio
async def test_no_warning_when_no_wrap_fill_and_cache_bust_none(caplog):
    """No wrap-fill (all lanes distinct trace_ids) + cache_bust NONE = no warning.
    The warning is about wrap-fill creating identical traffic, not about
    cache-bust being off in general.
    """
    import logging
    from aiperf.plugin.enums import CacheBustTarget

    trajectories = [
        Trajectory(conversation_id="trace_0", start_turn_index=0),
        Trajectory(conversation_id="trace_1", start_turn_index=0),
    ]
    ds = _make_dataset(num_traces=2, turns_per_trace=4)
    with caplog.at_level(logging.WARNING, logger="aiperf.timing.strategies.agentic_replay"):
        _make_strategy(
            phase=CreditPhase.PROFILING,
            trajectories=trajectories,
            dataset=ds,
            issuer=AsyncMock(),
            cache_bust_target=CacheBustTarget.NONE,
        )
    msgs = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert not any("identical" in m.lower() for m in msgs), msgs
```

Note: `_make_strategy` (imported from `test_agentic_replay_recycle_adversarial.py`) may not accept a `cache_bust_target` kwarg today. Check the helper before writing the test — if it doesn't, either extend it (preferred) or build the strategy manually in the new test file using the same scaffolding the helper uses.

- [ ] **Step 2: Run tests, expect failure**

```bash
uv run pytest -n auto "tests/unit/timing/strategies/test_agentic_replay_wrap_fill.py::test_warning_emitted_when_wrap_fill_and_cache_bust_none" -v
```

Expected: assertion failure — no WARNING currently emitted.

- [ ] **Step 3: Emit the WARNING at strategy init**

In `src/aiperf/timing/strategies/agentic_replay.py`, in `__init__`, after `_lanes_per_trace` is initialized (Task 4) and after `_cache_bust_target` is resolved (existing code around line 128), add:

```python
        # Detect the wrap-fill + cache_bust=NONE configuration that produces
        # byte-identical traffic across shared-trace lanes. The agentx-mvp
        # scenario auto-locks cache_bust=first_turn_prefix, so this never
        # fires for that scenario; users running ad-hoc agentic-replay with
        # cache_bust explicitly off get a loud heads-up.
        wrap_fill_active = any(
            count > 1 for count in self._lanes_per_trace.values()
        )
        if wrap_fill_active and self._cache_bust_target == CacheBustTarget.NONE:
            self.warning(
                "Wrap-fill active (%d distinct trace_ids fanned across %d "
                "lanes) with cache_bust.target=NONE: per-lane traffic will "
                "be byte-identical. Set cache_bust.target=first_turn_prefix "
                "(or another non-NONE target) for distinct shared-trace "
                "replays.",
                len(self._lanes_per_trace),
                sum(self._lanes_per_trace.values()),
            )
```

(Use `self.warning(...)` — `AIPerfLoggerMixin` exposes it. Lambdas optional; this string is cheap.)

- [ ] **Step 4: Run tests, expect pass**

```bash
uv run pytest -n auto tests/unit/timing/strategies/test_agentic_replay_wrap_fill.py -v
```

Expected: all wrap-fill tests pass (10 total across Tasks 4-6).

- [ ] **Step 5: Commit**

```bash
git add src/aiperf/timing/strategies/agentic_replay.py tests/unit/timing/strategies/test_agentic_replay_wrap_fill.py
git commit --no-verify -s -m "feat(agentic_replay): WARN on wrap-fill with cache_bust=NONE

Wrap-fill across lanes only stays workload-meaningful when the cache-bust
marker varies by lane. With cache_bust.target=NONE all shared-trace
lanes produce byte-identical traffic. Emit a WARNING at strategy init
when both conditions hold. agentx-mvp scenario auto-locks first_turn_prefix
so the warning never fires there; ad-hoc agentic-replay runs get a heads-up.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: Delete `InsufficientTrajectoriesError`, update existing tests

**Files:**
- Modify: `src/aiperf/common/scenario/base.py` (delete class)
- Modify: `src/aiperf/common/scenario/__init__.py` (drop export)
- Modify: `tests/unit/timing/test_trajectory_source_adversarial.py`
- Modify: `tests/unit/timing/test_trajectory_source_extended_adversarial.py`
- Modify: `tests/unit/timing/phase/test_runner_agentic_replay_warmup_target.py`
- Modify: `tests/component_integration/test_agentic_replay_pool_concurrency_integration.py`

**Why:** With wrap-fill activated, `len(self.trajectories) < concurrency` is now impossible (every shortfall is filled). The class becomes dead code; the tests that assert it become obsolete. Empty-pool cases still raise `EmptyTracePoolError`, which is separate.

- [ ] **Step 1: Identify dead tests by greppable signature**

```bash
grep -rn "InsufficientTrajectoriesError" src/ tests/
```

Expected hits: the spots listed above. Each `with pytest.raises(InsufficientTrajectoriesError)` block needs to be either deleted (if it tested only the now-impossible case) or replaced with a positive assertion that wrap-fill produced N trajectories.

- [ ] **Step 2: Update `tests/unit/timing/test_trajectory_source_adversarial.py`**

Read lines around 46 to understand the assertion's full context. Most likely the test is:

```python
def test_concurrency_exceeds_pool_raises():
    ...
    with pytest.raises(InsufficientTrajectoriesError) as exc_info:
        TrajectorySource(..., concurrency=N, ...)
    ...
```

Replace with a positive assertion that wrap-fill works:

```python
def test_concurrency_exceeds_pool_wrap_fills():
    """Wrap-fill replaces the old InsufficientTrajectoriesError behavior:
    when concurrency > usable trajectories, the post-build list is
    extended to ``concurrency`` by cycling through the distinct list.
    """
    ...
    src = TrajectorySource(..., concurrency=N, ...)
    assert len(src.trajectories) == N
    # Same distinct trace_ids; some duplicated.
    distinct = {t.conversation_id for t in src.trajectories}
    assert len(distinct) < N
```

Use Read to grab the actual test before editing — the helpers and fixtures need to stay.

Drop the `InsufficientTrajectoriesError` import (line 19).

- [ ] **Step 3: Update `tests/unit/timing/test_trajectory_source_extended_adversarial.py`**

Same pattern. Lines 101, 129, 143 reference `InsufficientTrajectoriesError`. Each `with pytest.raises(...)` block converts to a positive "wrap-fill produced N trajectories" assertion. Drop the import (line 27).

- [ ] **Step 4: Update `tests/unit/timing/phase/test_runner_agentic_replay_warmup_target.py`**

Lines 9 (docstring), 28 (import), 106 (docstring), 117 + 156 (`with pytest.raises`). Convert to positive wrap-fill assertions. Drop the import.

- [ ] **Step 5: Update `tests/component_integration/test_agentic_replay_pool_concurrency_integration.py`**

Line 38 (import), 372 (`with pytest.raises`), and the surrounding test 3 (lines around 363, "concurrency > pool_size -> InsufficientTrajectoriesError"). Delete test 3 entirely OR convert to a wrap-fill positive E2E assertion (Task 8 handles the canonical wrap-fill integration test, so a delete is fine here).

Drop the import (line 38). Update the file-level docstring (line 15) to remove the InsufficientTrajectoriesError reference.

- [ ] **Step 6: Delete the class itself**

In `src/aiperf/common/scenario/base.py`, delete the `InsufficientTrajectoriesError` class (line 94). Use Read to find the surrounding context — there may be related classes nearby.

In `src/aiperf/common/scenario/__init__.py`, drop the import (line 5) and the `__all__` entry (line 18).

- [ ] **Step 7: Run the full unit suite**

```bash
uv run pytest -n auto tests/unit/
```

(`addopts` in `pyproject.toml` already deselects `slow` / `performance` / etc by default.) Expected: green. If anything still imports `InsufficientTrajectoriesError`, fix and re-run.

- [ ] **Step 8: Commit**

```bash
git add -u src/ tests/
git status --short  # verify only intended files staged
git commit --no-verify -s -m "refactor: drop InsufficientTrajectoriesError, supplant w/ wrap-fill

The post-build 'concurrency > pool' guard is replaced by automatic
wrap-fill in TrajectorySource. The error class is no longer reachable;
remove it and convert each test that asserted it into a positive
assertion that wrap-fill produces the requested concurrency.

Empty-pool cases still raise EmptyTracePoolError (separate class).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 8: Component-integration E2E test for wrap-fill

**Files:**
- Test: `tests/component_integration/test_agentic_replay_wrap_fill.py` (new)

**Why:** Validate the full warmup → profiling → recycle loop with pool < concurrency. Confirms (a) the strategy completes without raising, (b) per-lane marker digests differ for shared-trace lanes, (c) no double-recycle errors logged.

- [ ] **Step 1: Read the existing pool_concurrency_integration test to crib scaffolding**

```bash
sed -n '1,80p' tests/component_integration/test_agentic_replay_pool_concurrency_integration.py
```

Note the imports, fixtures, and how the strategy is wired up with `_make_strategy` or a similar harness.

- [ ] **Step 2: Write the E2E test**

Create `tests/component_integration/test_agentic_replay_wrap_fill.py`:

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Component-integration E2E: agentic_replay with pool < concurrency.

Validates the full warmup → profiling → recycle loop when the trajectory
pool is smaller than --concurrency. Asserts:

1. Strategy construction succeeds (no InsufficientTrajectoriesError).
2. Warmup dispatches one credit per LANE (not per distinct trace).
3. After warmup, each lane's cache-bust marker is unique even when lanes
   share a trace_id.
4. Profiling completes without raising the double-recycle RuntimeError.
"""

from __future__ import annotations

import pytest

# Reuse the existing test harness scaffolding.
from tests.component_integration.test_agentic_replay_pool_concurrency_integration import (
    _build_integration_strategy,  # rename to actual helper after Task 8 Step 1 read.
)


@pytest.mark.asyncio
@pytest.mark.component_integration
async def test_pool_1_concurrency_4_wrap_fill_e2e():
    """Single-trace pool, 4-way concurrency: wrap-fill kicks in, 4 lanes
    all run trace_0 with distinct k_i and distinct cache-bust markers.
    """
    # IMPLEMENTATION: build a dataset with one 6-turn trace, instantiate
    # TrajectorySource(concurrency=4), then run the WARMUP strategy through
    # ``execute_phase``. Assert:
    #   - strategy.conversation_source.trajectories has 4 entries
    #   - all 4 entries have conversation_id == "trace_0"
    #   - len({t.start_turn_index for t in trajectories}) >= 2 (decorrelated)
    #   - per-lane markers in strategy._session_marker are all distinct
    raise NotImplementedError("Wire up against the real harness in Step 2.")
```

The `_build_integration_strategy` import name is illustrative — replace with the actual helper after reading Step 1. If no helper exists, build the strategy inline using the same dataset/sampler/issuer fixtures the existing integration test uses.

The full E2E test body should:

1. Build `DatasetMetadata` with 1 conversation, 6 turns.
2. Build `TrajectorySource(concurrency=4, random_seed=42)`.
3. Verify `len(src.trajectories) == 4` and all `conversation_id == "trace_0"`.
4. Verify at least 2 distinct `start_turn_index` values in the 4 lanes.
5. Build `AgenticReplayStrategy(phase=WARMUP, cache_bust=first_turn_prefix, ...)`.
6. Call `execute_phase`.
7. Assert `issuer.issue_credit` was awaited 4 times (one per lane).
8. Inspect `strategy._session_marker.values()` — all 4 distinct, none None.
9. Build a fresh strategy for `PROFILING`, run `setup_phase` + simulate 8 credit returns (2 recycle passes per lane).
10. Assert no `RuntimeError` raised, no double-recycle warnings logged.

Treat the existing `test_agentic_replay_pool_concurrency_integration.py` test 1 (happy path) as a template — copy its structure, then change the dataset/concurrency parameters and the assertions.

- [ ] **Step 3: Run the test**

```bash
uv run pytest -n auto tests/component_integration/test_agentic_replay_wrap_fill.py -v -m component_integration
```

Expected: pass.

- [ ] **Step 4: Run the full component-integration suite to confirm no break**

```bash
uv run pytest -n auto -m component_integration
```

Expected: green.

- [ ] **Step 5: Commit**

```bash
git add tests/component_integration/test_agentic_replay_wrap_fill.py
git commit --no-verify -s -m "test(agentic_replay): E2E wrap-fill happy path

Component-integration test for pool < concurrency: 1-trace pool, 4-way
concurrency. Validates wrap-fill activates, lanes get decorrelated k_i,
per-lane cache-bust markers are distinct, and the recycle loop completes
without tripping the double-recycle guard.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Final verification

After Task 8 completes, run a single full unit-suite pass (per the user's `feedback_plan_ceremony_minimalism.md` — exactly one `pytest -n auto tests/unit/`, not subfolder splits):

```bash
uv run pytest -n auto tests/unit/
```

Expected: green. Then run the component-integration suite:

```bash
uv run pytest -n auto -m component_integration
```

Expected: green.

If the user wants an actual E2E run against the mock server with the user's trace file:

```bash
PORT=$(python -c "import socket; s=socket.socket(); s.bind(('127.0.0.1',0)); print(s.getsockname()[1]); s.close()")
uv run aiperf-mock-server --port $PORT --fast --log-level WARNING &
MOCK_PID=$!
sleep 2
uv run aiperf profile \
    --scenario inferencex-agentx-mvp \
    --unsafe-override \
    --url 127.0.0.1:$PORT \
    --model gpt-5.5 \
    --tokenizer gpt2 \
    --max-context-length 128000 \
    --endpoint-type chat \
    --streaming \
    --use-server-token-count \
    --custom-dataset-type weka_trace \
    --input-file "/home/anthony/Downloads/91a41301c26657b2500e2dc71141217dd11b (1).json" \
    --benchmark-duration 60 \
    --concurrency 32 \
    --artifact-dir /tmp/agentx-mvp-run/artifacts-postfix \
    --ui simple
kill $MOCK_PID
```

Expected: run completes without `InsufficientTrajectoriesError`. 32 lanes all play trace `91a41…`. Profile output shows >>1 request count (vs. the 1 / 4 we saw pre-fix).
