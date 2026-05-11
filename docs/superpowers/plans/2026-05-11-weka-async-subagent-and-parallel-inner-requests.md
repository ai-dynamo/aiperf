# Weka Async Subagent + Parallel Inner-Request Replay — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Always use `model="opus"` on subagent dispatches.

**Goal:** Replay weka traces where the subagent runs past the parent's next turn (async dispatch), and where a subagent's inner requests overlap in time (true parallelism), preserving the recorded concurrency shape instead of artificially serializing.

**Architecture:** Two changes in `weka_trace.py` (and a mirror in `weka_parallel_convert.py`):
1. **Async-aware topology:** at branch-build time, compute the subagent's recorded end time (`sa.t + sa.duration_ms/1000`, falling back to `max(ir.t + ir.api_time)` when duration is None). If it exceeds the *following* parent turn's `t`, the parent didn't wait — emit the branch with `is_background=True` and **no** `SPAWN_JOIN` prerequisite.
2. **Stream-clustering for inner requests:** partition `sa.requests` into the minimum number of non-overlapping streams via greedy interval-packing (earliest-fitting stream wins). Each stream becomes one child Conversation; multiple streams become siblings under the same multi-child SPAWN branch.

Both changes are derivable from existing schema fields (`sa.t`, `sa.duration_ms`, inner `t` + `api_time`); no upstream `kv-cache-tester` schema change. Existing `ConversationBranchInfo.is_background` + multi-child SPAWN machinery already supports the shape; the loader just emits it now.

**Tech Stack:** Python 3.10+, Pydantic, AIPerf `WekaTraceLoader` (`src/aiperf/dataset/loader/weka_trace.py`), `_reconstruct_serial` and `_reconstruct_parallel` paths.

---

## File Structure

- **Modify:** `src/aiperf/dataset/loader/weka_trace.py` — topology pass (~lines 660-710), child-conversation builder (~lines 722-784), and corresponding sections of `_reconstruct_parallel`. Also adds a small pure helper `_pack_into_streams(inner_requests) -> list[list[WekaNormalRequest]]` next to the existing module-level dataclasses.
- **Modify:** `src/aiperf/dataset/loader/weka_parallel_convert.py` — mirror the topology + stream-packing logic in the multiprocessing worker (~lines 344-end).
- **Create:** `tests/fixtures/weka_traces/async_subagent_with_parallel_inner.json` — regression fixture copied from `/home/anthony/Downloads/91a41301c26657b2500e2dc71141217dd11b.json`.
- **Create:** `tests/unit/dataset/loader/test_weka_async_subagent.py` — unit tests for async-detection and stream-clustering, using the existing `_subagent`/`_normal`/`_build_trace`/`_make_loader` helper pattern from `test_weka_trace_graph_adversarial.py`.

The plan does NOT touch the scheduler or `ConversationBranchInfo` — the existing `is_background=True` + multi-child SPAWN already do what we need.

---

## Task 1: Regression fixture

**Files:**
- Create: `tests/fixtures/weka_traces/async_subagent_with_parallel_inner.json`

- [ ] **Step 1: Copy the user's downloaded trace into the fixtures directory**

```bash
cp /home/anthony/Downloads/91a41301c26657b2500e2dc71141217dd11b.json \
   tests/fixtures/weka_traces/async_subagent_with_parallel_inner.json
```

- [ ] **Step 2: Verify it parses cleanly under the AIPerf schema**

```bash
uv run python -c "
import orjson
from aiperf.dataset.loader.weka_trace_models import WekaTrace
raw = orjson.loads(open('tests/fixtures/weka_traces/async_subagent_with_parallel_inner.json', 'rb').read())
t = WekaTrace.model_validate(raw)
sa = [r for r in t.requests if r.type == 'subagent'][0]
print(f'trace_id={t.id}')
print(f'num requests={len(t.requests)}')
print(f'subagent at idx 4: t={sa.t}, duration_ms={sa.duration_ms}, inner={len(sa.requests)}')
"
```

Expected output (verbatim values you should see):
```
trace_id=91a41301c26657b2500e2dc71141217dd11b
num requests=8
subagent at idx 4: t=33.166, duration_ms=246584, inner=2
```

- [ ] **Step 3: Commit**

```bash
git add tests/fixtures/weka_traces/async_subagent_with_parallel_inner.json
git commit -s -m "test(weka): add async-subagent + parallel-inner regression fixture"
```

---

## Task 2: Async-aware topology in serial path

**Files:**
- Modify: `src/aiperf/dataset/loader/weka_trace.py:660-710` (topology grouping inside `_reconstruct_serial`)
- Test: `tests/unit/dataset/loader/test_weka_async_subagent.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/dataset/loader/test_weka_async_subagent.py`:

```python
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for async-subagent and parallel-inner-request replay in WekaTraceLoader.

Reuses the helpers from test_weka_trace_graph_adversarial.py: same
``_subagent``/``_normal``/``_build_trace``/``_make_loader`` pattern, same fixture
loader path.
"""

from pathlib import Path
from unittest.mock import MagicMock

import orjson
import pytest

from aiperf.common.enums import ConversationBranchMode, PrerequisiteKind
from aiperf.dataset.loader.weka_trace import WekaTraceLoader


def _mk_user_config():
    uc = MagicMock()
    uc.input.random_seed = 0
    uc.input.fixed_schedule_start_offset = None
    uc.input.fixed_schedule_end_offset = None
    uc.input.ignore_trace_delays = False
    uc.input.use_think_time_only = False
    uc.loadgen.inter_turn_delay_cap_seconds = None
    uc.input.synthesis.max_isl = None
    uc.input.synthesis.max_osl = None
    uc.input.max_context_length = None
    uc.input.synthesis.should_synthesize.return_value = False
    uc.input.prompt.input_tokens.block_size = None
    uc.tokenizer.trust_remote_code = False
    uc.tokenizer.revision = None
    uc.tokenizer.name = "t"
    uc.endpoint.model_names = ["m"]
    return uc


def _make_loader(filename, uc, monkeypatch):
    loader = WekaTraceLoader(filename=str(filename), user_config=uc)
    monkeypatch.setattr(
        loader,
        "synthesize_prompts_from_hash_ids",
        lambda rs: {r.key: f"p-{r.key}" for r in rs},
    )
    loader.prompt_generator = MagicMock()
    loader.prompt_generator._cache = {}
    loader.prompt_generator._sample_tokens.side_effect = lambda n: [0] * n
    loader.prompt_generator._tokenized_corpus = list(range(10000, 11000))
    loader.prompt_generator._corpus_size = 1000
    from tests.unit.dataset.loader.conftest import stub_hash_id_corpus_rng

    stub_hash_id_corpus_rng(loader.prompt_generator)
    loader.prompt_generator.tokenizer.decode.side_effect = (
        lambda toks: f"<dec:{len(toks)}>"
    )
    loader._tokenizer_name = "t"
    loader._trust_remote_code = False
    loader._tokenizer_revision = None
    loader._block_size = 64
    return loader


def _subagent(agent_id, *, t, duration_ms, inner):
    """inner: list of (t_offset_seconds, api_time_seconds_or_None)."""
    inner_reqs = [
        {
            "t": t + dt,
            "type": "n",
            "model": "m",
            "in": 10,
            "out": 1,
            "api_time": api_t,
        }
        for dt, api_t in inner
    ]
    return {
        "t": t,
        "type": "subagent",
        "agent_id": agent_id,
        "subagent_type": "X",
        "duration_ms": duration_ms,
        "total_tokens": 0,
        "tool_use_count": 0,
        "status": "completed",
        "requests": inner_reqs,
        "models": ["m"],
    }


def _normal(t, model="m", in_=10, out=1):
    return {"t": t, "type": "n", "model": model, "in": in_, "out": out}


def _build_trace(trace_id, requests, models=("m",)):
    return {
        "id": trace_id,
        "models": list(models),
        "block_size": 64,
        "hash_id_scope": "local",
        "requests": requests,
    }


def _write_trace(tmp_path, data, name="t.json"):
    p = tmp_path / name
    p.write_bytes(orjson.dumps(data))
    return p


def test_subagent_running_past_following_parent_is_background(tmp_path, monkeypatch):
    """sa.t + duration_ms/1000 > following_parent.t → branch is_background=True,
    no SPAWN_JOIN prerequisite.
    """
    data = _build_trace(
        "t_async",
        [
            _normal(t=0.0),
            # sa starts at t=1, runs 100 seconds, ends at t=101.
            _subagent("a1", t=1.0, duration_ms=100_000, inner=[(0.0, 100.0)]),
            # following parent at t=2 — well before sa_end at t=101.
            _normal(t=2.0),
        ],
    )
    path = _write_trace(tmp_path, data)
    loader = _make_loader(path, _mk_user_config(), monkeypatch)
    convs = loader.convert_to_conversations()

    parent = next(c for c in convs if c.session_id == "t_async")
    assert len(parent.branches) == 1
    branch = parent.branches[0]
    assert branch.mode == ConversationBranchMode.SPAWN
    assert branch.is_background is True, (
        "Subagent runs past following parent turn — parent didn't wait. "
        "Expected is_background=True, got False."
    )
    # No SPAWN_JOIN prerequisite on any parent turn for this branch.
    for turn in parent.turns:
        for prereq in turn.prerequisites:
            assert not (
                prereq.kind == PrerequisiteKind.SPAWN_JOIN
                and prereq.branch_id == branch.branch_id
            ), "background branch should not have a SPAWN_JOIN prerequisite"


def test_subagent_finishing_before_following_parent_keeps_join(tmp_path, monkeypatch):
    """sa.t + duration_ms/1000 < following_parent.t → branch has SPAWN_JOIN,
    is_background=False (current behavior, regression guard).
    """
    data = _build_trace(
        "t_sync",
        [
            _normal(t=0.0),
            # sa runs 1s, ends at t=2.
            _subagent("a1", t=1.0, duration_ms=1000, inner=[(0.0, 1.0)]),
            # following parent at t=10 — well after sa_end at t=2.
            _normal(t=10.0),
        ],
    )
    path = _write_trace(tmp_path, data)
    loader = _make_loader(path, _mk_user_config(), monkeypatch)
    convs = loader.convert_to_conversations()

    parent = next(c for c in convs if c.session_id == "t_sync")
    branch = parent.branches[0]
    assert branch.is_background is False
    # SPAWN_JOIN must be on the following parent turn.
    following_turn = parent.turns[1]
    join_prereqs = [
        p for p in following_turn.prerequisites
        if p.kind == PrerequisiteKind.SPAWN_JOIN and p.branch_id == branch.branch_id
    ]
    assert len(join_prereqs) == 1


def test_subagent_duration_ms_none_falls_back_to_inner_api_time(tmp_path, monkeypatch):
    """When duration_ms is None (status='async_launched' style), end-time is
    inferred from max(inner.t + inner.api_time)."""
    data = _build_trace(
        "t_no_dur",
        [
            _normal(t=0.0),
            # duration_ms=None, but inner request runs from t=1 to t=51.
            _subagent("a1", t=1.0, duration_ms=None, inner=[(0.0, 50.0)]),
            _normal(t=2.0),  # well before sa_end at t=51.
        ],
    )
    path = _write_trace(tmp_path, data)
    loader = _make_loader(path, _mk_user_config(), monkeypatch)
    convs = loader.convert_to_conversations()

    parent = next(c for c in convs if c.session_id == "t_no_dur")
    branch = parent.branches[0]
    assert branch.is_background is True
```

- [ ] **Step 2: Run the failing test, confirm it fails**

```bash
uv run pytest tests/unit/dataset/loader/test_weka_async_subagent.py -v -n auto
```

Expected: the first and third tests FAIL with `assert False is True` (because today `is_background = (following is None)` regardless of timing). The middle test PASSES (regression baseline).

- [ ] **Step 3: Implement the topology fix in `_reconstruct_serial`**

Edit `src/aiperf/dataset/loader/weka_trace.py`. In `_reconstruct_serial`, locate the topology block (around lines 660-710). Make the following changes:

**3a.** Build an `outer_idx → t` map alongside `outer_to_turn_pos`. Right after the first pass that emits parent turns (around line 658, after `outer_to_turn_pos[outer_idx] = len(conv.turns) - 1`), add:

```python
outer_to_t: dict[int, float] = {
    outer_idx: req.t for outer_idx, req in plan.normals
}
```

**3b.** In the grouping block, while iterating `plan.subagents`, also record the *outer* following index alongside the turn-pos `following`, so we can look up its `t`. Change:

```python
for sa_outer_idx, sa_entry in plan.subagents:
    preceding = max(
        (pos for oi, pos in outer_to_turn_pos.items() if oi < sa_outer_idx),
        default=None,
    )
    following = min(
        (pos for oi, pos in outer_to_turn_pos.items() if oi > sa_outer_idx),
        default=None,
    )
    if preceding is None:
        _logger.info(
            f"Dropping subagent '{sa_entry.agent_id}' from trace "
            f"{plan.trace_id}: no preceding parent turn"
        )
        dropped_sa_agent_ids.add(sa_entry.agent_id)
        continue
    key = (preceding, following)
    if key not in groups:
        group_order.append(key)
    groups[key].append(sa_entry)
```

to also build a `key → following_outer_idx` map keyed on the same group key:

```python
group_following_outer: dict[tuple[int | None, int | None], int | None] = {}
for sa_outer_idx, sa_entry in plan.subagents:
    preceding = max(
        (pos for oi, pos in outer_to_turn_pos.items() if oi < sa_outer_idx),
        default=None,
    )
    following = min(
        (pos for oi, pos in outer_to_turn_pos.items() if oi > sa_outer_idx),
        default=None,
    )
    if preceding is None:
        _logger.info(
            f"Dropping subagent '{sa_entry.agent_id}' from trace "
            f"{plan.trace_id}: no preceding parent turn"
        )
        dropped_sa_agent_ids.add(sa_entry.agent_id)
        continue
    following_outer_idx = min(
        (oi for oi in outer_to_t.keys() if oi > sa_outer_idx),
        default=None,
    )
    key = (preceding, following)
    if key not in groups:
        group_order.append(key)
        group_following_outer[key] = following_outer_idx
    groups[key].append(sa_entry)
```

**3c.** In the branch-emission loop, compute `sa_end_t` and override `is_background`. Replace:

```python
for preceding, following in group_order:
    entries = groups[(preceding, following)]
    child_sids = [f"{plan.trace_id}::sa:{e.agent_id}" for e in entries]
    branch_id = f"{plan.trace_id}:spawn:{entries[0].agent_id}"
    is_background = following is None
    conv.branches.append(
        ConversationBranchInfo(
            branch_id=branch_id,
            child_conversation_ids=child_sids,
            mode=ConversationBranchMode.SPAWN,
            is_background=is_background,
        )
    )
    conv.turns[preceding].branch_ids.append(branch_id)
    if following is not None:
        conv.turns[following].prerequisites.append(
            TurnPrerequisite(
                kind=PrerequisiteKind.SPAWN_JOIN,
                branch_id=branch_id,
            )
        )
```

with:

```python
for preceding, following in group_order:
    entries = groups[(preceding, following)]
    child_sids = [f"{plan.trace_id}::sa:{e.agent_id}" for e in entries]
    branch_id = f"{plan.trace_id}:spawn:{entries[0].agent_id}"

    is_background = following is None
    if not is_background:
        following_outer_idx = group_following_outer[(preceding, following)]
        following_t = outer_to_t[following_outer_idx]
        sa_end_t = max(
            _sa_end_seconds(entry) for entry in entries
        )
        if sa_end_t > following_t:
            is_background = True
            _logger.info(
                f"Trace {plan.trace_id}: reclassifying subagent branch "
                f"'{branch_id}' as background — recorded subagent end "
                f"t={sa_end_t:.2f}s exceeds following parent turn t={following_t:.2f}s "
                f"(parent did not wait in the recording)."
            )

    conv.branches.append(
        ConversationBranchInfo(
            branch_id=branch_id,
            child_conversation_ids=child_sids,
            mode=ConversationBranchMode.SPAWN,
            is_background=is_background,
        )
    )
    conv.turns[preceding].branch_ids.append(branch_id)
    if following is not None and not is_background:
        conv.turns[following].prerequisites.append(
            TurnPrerequisite(
                kind=PrerequisiteKind.SPAWN_JOIN,
                branch_id=branch_id,
            )
        )
```

**3d.** At module scope (near the top, after the `_NormalRequestT` alias), add the helper:

```python
def _sa_end_seconds(entry: WekaSubagentEntry) -> float:
    """Recorded end time of a subagent, in seconds.

    Uses ``duration_ms`` when present. Falls back to ``max(inner.t + inner.api_time)``
    when ``duration_ms`` is None (recorded for ``status='async_launched'`` subagents).
    Falls back further to ``entry.t`` when both are unavailable.
    """
    if entry.duration_ms is not None:
        return entry.t + entry.duration_ms / 1000.0
    if entry.requests:
        return max(
            ir.t + (ir.api_time or 0.0) for ir in entry.requests
        )
    return entry.t
```

- [ ] **Step 4: Run the tests, confirm they pass**

```bash
uv run pytest tests/unit/dataset/loader/test_weka_async_subagent.py -v -n auto
```

Expected: all three tests PASS.

- [ ] **Step 5: Run the full unit suite to confirm no regressions**

```bash
uv run pytest tests/unit/ -n auto
```

Expected: no new failures attributable to this change. Pre-existing flakes (e.g. `gotcha_test_error_queue_flaky_under_xdist`) are acceptable; rerun once.

- [ ] **Step 6: Commit**

```bash
git add -p src/aiperf/dataset/loader/weka_trace.py \
        tests/unit/dataset/loader/test_weka_async_subagent.py
git commit -s -m "feat(weka): reclassify subagent branch as async when sa_end > following parent t"
```

---

## Task 3: Stream-packing for parallel inner requests (serial path)

**Files:**
- Modify: `src/aiperf/dataset/loader/weka_trace.py` — `_ChildPlan`, child-conversation builder (~lines 722-784), and add `_pack_into_streams` helper
- Test: `tests/unit/dataset/loader/test_weka_async_subagent.py` (extend with new tests)

- [ ] **Step 1: Add the failing tests**

Append to `tests/unit/dataset/loader/test_weka_async_subagent.py`:

```python
def test_subagent_with_overlapping_inner_requests_emits_separate_child_conversations(
    tmp_path, monkeypatch
):
    """Two inner requests with overlapping [t, t+api_time] become two child
    Conversations under one multi-child SPAWN branch.
    """
    data = _build_trace(
        "t_par",
        [
            _normal(t=0.0),
            # Two inner requests at t=1 and t=1.1, both running 100s — overlap ~99.9s.
            _subagent(
                "a1",
                t=1.0,
                duration_ms=100_000,
                inner=[(0.0, 100.0), (0.1, 100.0)],
            ),
            _normal(t=200.0),  # well after both inner ends; SPAWN_JOIN-eligible.
        ],
    )
    path = _write_trace(tmp_path, data)
    loader = _make_loader(path, _mk_user_config(), monkeypatch)
    convs = loader.convert_to_conversations()

    parent = next(c for c in convs if c.session_id == "t_par")
    branch = parent.branches[0]
    # Two streams → two child conversations as siblings in the branch.
    assert len(branch.child_conversation_ids) == 2, (
        f"Expected 2 sibling child conversations, got "
        f"{branch.child_conversation_ids}"
    )
    expected_sids = {"t_par::sa:a1:s0", "t_par::sa:a1:s1"}
    assert set(branch.child_conversation_ids) == expected_sids

    children = {c.session_id: c for c in convs if c.session_id.startswith("t_par::sa")}
    assert set(children.keys()) == expected_sids
    for sid in expected_sids:
        assert len(children[sid].turns) == 1, (
            f"each parallel stream is one inner request → one turn; "
            f"{sid} has {len(children[sid].turns)} turns"
        )


def test_subagent_with_sequential_inner_requests_emits_one_child_conversation(
    tmp_path, monkeypatch
):
    """Two non-overlapping inner requests stay in ONE child Conversation as two
    sequential turns (regression: don't fragment serial inners).
    """
    data = _build_trace(
        "t_seq",
        [
            _normal(t=0.0),
            # Inner 0: t=1, runs 1s (ends t=2). Inner 1: t=3, runs 1s (ends t=4).
            _subagent(
                "a1",
                t=1.0,
                duration_ms=3000,
                inner=[(0.0, 1.0), (2.0, 1.0)],
            ),
            _normal(t=10.0),
        ],
    )
    path = _write_trace(tmp_path, data)
    loader = _make_loader(path, _mk_user_config(), monkeypatch)
    convs = loader.convert_to_conversations()

    parent = next(c for c in convs if c.session_id == "t_seq")
    branch = parent.branches[0]
    assert branch.child_conversation_ids == ["t_seq::sa:a1"], (
        "single sequential stream keeps the legacy session-id shape (no :s0 suffix)"
    )
    child = next(c for c in convs if c.session_id == "t_seq::sa:a1")
    assert len(child.turns) == 2
```

- [ ] **Step 2: Run, confirm both new tests fail**

```bash
uv run pytest tests/unit/dataset/loader/test_weka_async_subagent.py::test_subagent_with_overlapping_inner_requests_emits_separate_child_conversations tests/unit/dataset/loader/test_weka_async_subagent.py::test_subagent_with_sequential_inner_requests_emits_one_child_conversation -v
```

Expected: parallel test FAILS (today's code emits one Conversation `t_par::sa:a1` with two turns). Sequential test PASSES (regression baseline).

- [ ] **Step 3: Add the `_pack_into_streams` helper**

At module scope in `src/aiperf/dataset/loader/weka_trace.py`, near `_sa_end_seconds`, add:

```python
def _pack_into_streams(
    requests: list[WekaNormalRequest],
) -> list[list[WekaNormalRequest]]:
    """Partition inner requests into the minimum number of non-overlapping
    sequential streams (interval-graph chromatic decomposition, greedy
    earliest-fit).

    Two requests ``A``, ``B`` overlap when ``[A.t, A.t + A.api_time)`` intersects
    ``[B.t, B.t + B.api_time)``. Each returned stream is a chain of
    non-overlapping requests in ``t``-order. The number of streams equals the
    maximum number of concurrent inner requests at any instant.

    A request with ``api_time = None`` is treated as zero-duration (the
    interval becomes the instant ``[t, t)``) — it never overlaps anything by
    itself, so it lands in the first stream by ``t``. This matches the
    behaviour of subagents whose telemetry was not captured.
    """
    sorted_reqs = sorted(requests, key=lambda r: r.t)
    streams: list[list[WekaNormalRequest]] = []
    stream_ends: list[float] = []
    for r in sorted_reqs:
        r_end = r.t + (r.api_time or 0.0)
        placed = False
        for i, end in enumerate(stream_ends):
            if end <= r.t:
                streams[i].append(r)
                stream_ends[i] = r_end
                placed = True
                break
        if not placed:
            streams.append([r])
            stream_ends.append(r_end)
    return streams
```

- [ ] **Step 4: Replace `_ChildPlan` and rebuild child plans per stream**

Change the `_ChildPlan` dataclass (around line 63) to carry the stream's request list and a `stream_index`:

```python
@dataclass
class _ChildPlan:
    session_id: str
    parent_trace_id: str
    subagent_index: int
    entry: WekaSubagentEntry
    stream_index: int
    stream_requests: list[WekaNormalRequest]
```

In `_reconstruct_serial` where child plans are built (around line 467), replace the single-`_ChildPlan` emission with one-per-stream:

```python
else:  # WekaSubagentEntry
    sa_index = len(subagents)
    subagents.append((idx, req))
    streams = _pack_into_streams(list(req.requests))
    for stream_idx, stream_reqs in enumerate(streams):
        if len(streams) == 1:
            child_sid = f"{trace_id}::sa:{req.agent_id}"
        else:
            child_sid = f"{trace_id}::sa:{req.agent_id}:s{stream_idx}"
        child_plans.append(
            _ChildPlan(
                session_id=child_sid,
                parent_trace_id=trace_id,
                subagent_index=sa_index,
                entry=req,
                stream_index=stream_idx,
                stream_requests=stream_reqs,
            )
        )
```

- [ ] **Step 5: Update the branch-emission to list all stream-children**

In the same `for preceding, following in group_order:` loop modified in Task 2, the `child_sids` line needs to enumerate streams per entry rather than one sid per entry. Replace:

```python
child_sids = [f"{plan.trace_id}::sa:{e.agent_id}" for e in entries]
```

with:

```python
child_sids: list[str] = []
for e in entries:
    e_streams = _pack_into_streams(list(e.requests))
    if len(e_streams) == 1:
        child_sids.append(f"{plan.trace_id}::sa:{e.agent_id}")
    else:
        for stream_idx in range(len(e_streams)):
            child_sids.append(
                f"{plan.trace_id}::sa:{e.agent_id}:s{stream_idx}"
            )
        _logger.info(
            f"Trace {plan.trace_id}: subagent '{e.agent_id}' has "
            f"{len(e_streams)} parallel inner-request streams; emitting "
            f"as sibling child conversations."
        )
```

- [ ] **Step 6: Update the child-Conversation builder to iterate the stream**

In the `for cp in child_plans:` loop (around line 722-784), change the iteration `for k, creq in enumerate(cp.entry.requests)` to use the stream:

```python
for k, creq in enumerate(cp.stream_requests):
    seed = f"{cp.session_id}:turn_{k}:partial_tail"
    if k == 0:
        child_recon.init_turn_0(
            hash_ids=creq.hash_ids,
            in_tokens=creq.input_length,
            tool_tokens=cp.entry.tool_tokens,
            system_tokens=cp.entry.system_tokens,
            seed=seed,
        )
    else:
        prev_creq = cp.stream_requests[k - 1]
        child_recon.advance_turn(
            prev_hash_ids=prev_creq.hash_ids,
            prev_in_tokens=prev_creq.input_length,
            prev_out_tokens=prev_creq.output_length,
            curr_hash_ids=creq.hash_ids,
            curr_in_tokens=creq.input_length,
            seed=seed,
        )
    t_ms = creq.t * 1000.0
    if k == 0:
        child_delay_ms: float | None = None
    elif think_time_only and creq.think_time is not None:
        child_delay_ms = creq.think_time * 1000.0
    else:
        child_delay_ms = t_ms - cp.stream_requests[k - 1].t * 1000.0
    if child_delay_ms is not None:
        child_delay_ms = self._delay_cap_tracker.clamp(child_delay_ms)
    child_delta = child_recon.turn_delta()
    child_conv.turns.append(
        Turn(
            timestamp=None if ignore_delays else t_ms,
            delay=None if ignore_delays else child_delay_ms,
            model=child_model_map.get(creq.model, creq.model),
            max_tokens=self._cap_output(creq),
            raw_messages=child_delta.delta_messages,
            reset_context=child_delta.reset_context,
        )
    )
```

(The only structural changes vs. today: `cp.entry.requests` → `cp.stream_requests`; everything else is identical.)

- [ ] **Step 7: Run the tests, confirm they pass**

```bash
uv run pytest tests/unit/dataset/loader/test_weka_async_subagent.py -v -n auto
```

Expected: all five tests PASS.

- [ ] **Step 8: Run the full unit suite**

```bash
uv run pytest tests/unit/ -n auto
```

Expected: no new failures. The existing `test_weka_trace_graph_adversarial.py` tests should still pass — they use sequential or zero inner requests, which fall in the `len(streams) == 1` path that preserves the legacy session-id shape.

- [ ] **Step 9: Commit**

```bash
git add -p src/aiperf/dataset/loader/weka_trace.py \
        tests/unit/dataset/loader/test_weka_async_subagent.py
git commit -s -m "feat(weka): split overlapping subagent inner requests into parallel sibling conversations"
```

---

## Task 4: Mirror the changes in the parallel reconstruction path

**Files:**
- Modify: `src/aiperf/dataset/loader/weka_parallel_convert.py` — topology block (~lines 344-380) and child-conversation worker block (~lines 382-end)

The parallel multiprocessing path duplicates the serial logic in a worker that does not import from `weka_trace`. It must mirror the same two fixes for byte-equivalence (already enforced by `test_weka_trace_byte_exact_corpus.py`).

- [ ] **Step 1: Write the failing test (force the parallel path)**

Append to `tests/unit/dataset/loader/test_weka_async_subagent.py`:

```python
def test_async_branch_detected_under_parallel_reconstruction(
    tmp_path, monkeypatch
):
    """Same async-detection under the multiprocessing path."""
    monkeypatch.setenv("AIPERF_WEKA_PARALLEL_THRESHOLD", "1")
    monkeypatch.setenv("AIPERF_WEKA_PARALLEL_WORKERS", "2")
    data = _build_trace(
        "t_par_async",
        [
            _normal(t=0.0),
            _subagent("a1", t=1.0, duration_ms=100_000, inner=[(0.0, 100.0)]),
            _normal(t=2.0),
        ],
    )
    path = _write_trace(tmp_path, data)
    loader = _make_loader(path, _mk_user_config(), monkeypatch)
    convs = loader.convert_to_conversations()
    parent = next(c for c in convs if c.session_id == "t_par_async")
    branch = parent.branches[0]
    assert branch.is_background is True


def test_parallel_inner_split_under_parallel_reconstruction(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("AIPERF_WEKA_PARALLEL_THRESHOLD", "1")
    monkeypatch.setenv("AIPERF_WEKA_PARALLEL_WORKERS", "2")
    data = _build_trace(
        "t_par_split",
        [
            _normal(t=0.0),
            _subagent(
                "a1",
                t=1.0,
                duration_ms=100_000,
                inner=[(0.0, 100.0), (0.1, 100.0)],
            ),
            _normal(t=200.0),
        ],
    )
    path = _write_trace(tmp_path, data)
    loader = _make_loader(path, _mk_user_config(), monkeypatch)
    convs = loader.convert_to_conversations()
    parent = next(c for c in convs if c.session_id == "t_par_split")
    branch = parent.branches[0]
    assert set(branch.child_conversation_ids) == {
        "t_par_split::sa:a1:s0",
        "t_par_split::sa:a1:s1",
    }
```

Verify the env-var names match the loader. Check with:

```bash
grep -nE "WEKA_PARALLEL" src/aiperf/dataset/loader/weka_trace.py | head
```

If the actual names differ (e.g. they may be `WEKA_PARALLEL_THRESHOLD` without the `AIPERF_` prefix, or read via `Environment.dataset.weka_parallel_threshold`), update the monkeypatch calls accordingly **before** running the test.

- [ ] **Step 2: Run the failing tests**

```bash
uv run pytest tests/unit/dataset/loader/test_weka_async_subagent.py::test_async_branch_detected_under_parallel_reconstruction tests/unit/dataset/loader/test_weka_async_subagent.py::test_parallel_inner_split_under_parallel_reconstruction -v
```

Expected: both FAIL with the same assertion errors as Tasks 2 and 3.

- [ ] **Step 3: Mirror the topology fix in `weka_parallel_convert.py`**

In `weka_parallel_convert.py`, locate the topology block (around lines 344-380). The dict payload `_WekaSubagentMarkerPayload` carries `t`, `duration_ms`, and inner `requests` (each with `t`, `api_time`). Apply the same logic:

- Build `outer_to_t` from the parent-normal-request iteration earlier in the worker (the `parent` task carries a list of normals; use `req["t"]` as the source).
- In the grouping section, record `group_following_outer[key] = following_outer_idx`.
- In the branch-emission loop, compute `sa_end_t` via a local helper that mirrors `_sa_end_seconds` but reads from dict payloads, and override `is_background` accordingly. Drop the SPAWN_JOIN when reclassified.

The dict-payload variant of the helper:

```python
def _sa_end_seconds_dict(entry: _WekaSubagentMarkerPayload) -> float:
    if entry.get("duration_ms") is not None:
        return entry["t"] + entry["duration_ms"] / 1000.0
    reqs = entry.get("requests") or []
    if reqs:
        return max(
            ir["t"] + (ir.get("api_time") or 0.0) for ir in reqs
        )
    return entry["t"]
```

Place this helper at module scope near the existing `_WekaSubagentMarkerPayload` definition.

- [ ] **Step 4: Mirror stream-packing**

Add a dict-payload variant of `_pack_into_streams`:

```python
def _pack_into_streams_dict(
    requests: list[_WekaNormalRequestPayload],
) -> list[list[_WekaNormalRequestPayload]]:
    sorted_reqs = sorted(requests, key=lambda r: r["t"])
    streams: list[list[_WekaNormalRequestPayload]] = []
    stream_ends: list[float] = []
    for r in sorted_reqs:
        r_end = r["t"] + (r.get("api_time") or 0.0)
        placed = False
        for i, end in enumerate(stream_ends):
            if end <= r["t"]:
                streams[i].append(r)
                stream_ends[i] = r_end
                placed = True
                break
        if not placed:
            streams.append([r])
            stream_ends.append(r_end)
    return streams
```

In the child-conversation builder section (the `for cp in task.children:` loop, ~lines 382-end), partition `cp["requests"]` into streams and emit one `_WekaChildDict` per stream. The session-id naming rule is the same as the serial path: 1 stream → unchanged sid; N streams → `<sid>:s0`, `:s1`, ...

The construction of `task.children` happens earlier (when the work task is built in `_reconstruct_parallel` of `weka_trace.py`). That is where `_WekaChildPayload`s are constructed today — modify it to emit one payload per stream, with a `stream_index` field and a `stream_requests` list mirroring the dataclass changes in Task 3.

(If the worker reads `cp["requests"]` directly and we instead pass per-stream-already-split payloads, the worker only needs to iterate `cp["requests"]` as before — the split happens before serialization. Prefer that: keep the worker dumb and split in the parent process.)

- [ ] **Step 5: Update the branch-side `child_session_ids` listing**

In the branch emission of `weka_parallel_convert.py`, replace the `child_sids = [f"{task.trace_id}::sa:{e['agent_id']}" for e in entries]` with the same stream-aware enumeration used in the serial path.

- [ ] **Step 6: Run the tests, confirm they pass**

```bash
uv run pytest tests/unit/dataset/loader/test_weka_async_subagent.py -v -n auto
```

Expected: all seven tests PASS.

- [ ] **Step 7: Run byte-exact regression for the parallel path**

```bash
uv run pytest tests/unit/dataset/loader/test_weka_trace_byte_exact_corpus.py -v -n auto
uv run pytest tests/component_integration/dataset/test_weka_trace_byte_exact_drift.py -v -n auto
```

Expected: PASS. (The corpus uses real traces without parallel inner requests under our current understanding; if any trace happens to trigger the new path, the bytes will change and we need to update the recorded baselines. If that happens, regenerate baselines and commit them as a separate "test: refresh weka byte-exact baselines after parallel-inner-request split" commit.)

- [ ] **Step 8: Run the full unit suite**

```bash
uv run pytest tests/unit/ -n auto
```

- [ ] **Step 9: Commit**

```bash
git add -p src/aiperf/dataset/loader/weka_parallel_convert.py \
        src/aiperf/dataset/loader/weka_trace.py \
        tests/unit/dataset/loader/test_weka_async_subagent.py
git commit -s -m "feat(weka): mirror async-detection and stream-splitting in parallel reconstruction"
```

---

## Task 5: Integration test against the real downloaded trace

**Files:**
- Test: `tests/unit/dataset/loader/test_weka_async_subagent.py` (append)

- [ ] **Step 1: Add the integration test**

Append to `tests/unit/dataset/loader/test_weka_async_subagent.py`:

```python
FIXTURES = Path(__file__).parents[3] / "fixtures" / "weka_traces"


def test_async_subagent_with_parallel_inner_real_trace(tmp_path, monkeypatch):
    """End-to-end regression against the real captured trace.

    Trace shape (verified by inspection):
      - 7 streaming parent turns at t=0, 13, 23.89, 32.36, 36.54, 271.10, 280.18
      - 1 subagent at outer index 4 (t=33.16, duration_ms=246584)
        with TWO overlapping inner requests (api_time ≈ 237s each)

    Expected loader output:
      - 1 SPAWN branch with is_background=True (sa_end ≈ 279.75 > 36.54)
      - 2 sibling child conversations with session ids
        '<trace>::sa:codex_subagent_001:s0' and ':s1'
      - No SPAWN_JOIN prerequisite on parent turn 4 (the t=36.54 turn)
    """
    src = FIXTURES / "async_subagent_with_parallel_inner.json"
    assert src.exists(), f"regression fixture missing: {src}"
    # Loader requires a single file path or directory; copy into tmp_path
    # so we don't depend on the fixture location at runtime.
    dst = tmp_path / src.name
    dst.write_bytes(src.read_bytes())

    uc = _mk_user_config()
    loader = _make_loader(dst, uc, monkeypatch)
    convs = loader.convert_to_conversations()

    parent = next(c for c in convs if c.session_id == "91a41301c26657b2500e2dc71141217dd11b")
    assert len(parent.branches) == 1
    branch = parent.branches[0]
    assert branch.mode == ConversationBranchMode.SPAWN
    assert branch.is_background is True
    assert set(branch.child_conversation_ids) == {
        "91a41301c26657b2500e2dc71141217dd11b::sa:codex_subagent_001:s0",
        "91a41301c26657b2500e2dc71141217dd11b::sa:codex_subagent_001:s1",
    }
    # No SPAWN_JOIN on any parent turn for this branch.
    for turn in parent.turns:
        for prereq in turn.prerequisites:
            assert not (
                prereq.kind == PrerequisiteKind.SPAWN_JOIN
                and prereq.branch_id == branch.branch_id
            )

    # Both children exist and each has exactly one turn.
    sid_s0 = "91a41301c26657b2500e2dc71141217dd11b::sa:codex_subagent_001:s0"
    sid_s1 = "91a41301c26657b2500e2dc71141217dd11b::sa:codex_subagent_001:s1"
    children_by_sid = {c.session_id: c for c in convs}
    assert sid_s0 in children_by_sid
    assert sid_s1 in children_by_sid
    assert len(children_by_sid[sid_s0].turns) == 1
    assert len(children_by_sid[sid_s1].turns) == 1
```

- [ ] **Step 2: Run the integration test**

```bash
uv run pytest tests/unit/dataset/loader/test_weka_async_subagent.py::test_async_subagent_with_parallel_inner_real_trace -v
```

Expected: PASS, given Tasks 2-4 are complete. If it fails, the assertion message identifies which property is wrong (branch shape, sibling ids, or SPAWN_JOIN presence).

- [ ] **Step 3: Full unit suite final pass**

```bash
uv run pytest tests/unit/ -n auto
```

- [ ] **Step 4: Commit**

```bash
git add tests/unit/dataset/loader/test_weka_async_subagent.py
git commit -s -m "test(weka): regression for real async-subagent trace with parallel inner requests"
```

---

## Self-review checklist (run after planning, before dispatch)

1. **Spec coverage:**
   - Change A (async-aware topology) → Task 2 ✓
   - Change B (parallel inner-request streams) → Task 3 ✓
   - Parallel-path mirror → Task 4 ✓
   - Real-file regression → Task 5 ✓
   - INFO logging → Task 2 step 3c + Task 3 step 5 ✓
2. **Placeholder scan:** no TBDs, no "add appropriate error handling," no "similar to Task N" without code. ✓
3. **Type/name consistency:** `_sa_end_seconds(entry)` is used in Task 2; the dict-payload variant `_sa_end_seconds_dict` introduced in Task 4 lives in the parallel module. `_pack_into_streams` (Task 3) and `_pack_into_streams_dict` (Task 4) have matching signatures. `_ChildPlan` gains two new fields (`stream_index`, `stream_requests`) in Task 3 and is read in Task 3's child builder. ✓

## Known limitations (out of scope for this plan)

- **Per-child start delay relative to branch fire-time.** Inner requests within a parallel group are launched at slightly different `t` values in the recording (0.06s offset in the real trace). This plan launches all stream-children simultaneously when the branch fires; the sub-second offset is not preserved. Acceptable for MVP; revisit if a trace shows >1s intra-group offset materially distorting metrics.
- **Hash-id prefix sharing across parallel siblings.** Each stream gets its own `ConversationReconstructor` with a cleared block cache (today's per-child behaviour preserved). Sibling streams that share the same prefix tokens will not reuse KV-cache from each other in the replay — which is correct, since they were independent sessions in the recording.
- **Multi-`status` enrichment.** The plan does not read `sa.status` ('completed', 'async_launched', etc.). The async heuristic (`sa_end > following_t`) is timing-based; `status='async_launched'` traces typically have `duration_ms=None`, and the `_sa_end_seconds` fallback to `max(ir.t + ir.api_time)` handles them naturally. Future work could add an explicit status-based override.
