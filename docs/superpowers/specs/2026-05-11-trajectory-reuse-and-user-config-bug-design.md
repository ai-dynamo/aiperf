# Design: Trajectory reuse + user_config trace-scan bug fix

**Date:** 2026-05-11
**Author:** Anthony Casagrande
**Status:** Approved (inline, before write)

## Motivation

Two issues surfaced while running the `inferencex-agentx-mvp` scenario against a
single-trace `weka_trace` JSON file:

1. **Trace-scan crash.** `UserConfig._should_use_fixed_schedule_for_trace_dataset`
   reads the input file line-by-line to detect a `timestamp` key. When the file
   is pretty-printed multi-line JSON (not JSONL), some lines parse as bare
   scalars (e.g. the last element of a `hash_ids` array, `62\n`, parses to
   `int(62)`). The check `"timestamp" in data` then raises
   `TypeError: argument of type 'int' is not iterable`. Repro: trace
   `91a41301c26657b2500e2dc71141217dd11b.json` with
   `--custom-dataset-type weka_trace --inter-turn-delay-cap-seconds=1
   --unsafe-override`.

2. **Concurrency capped at pool size.** The agentic-replay scheduler raises
   `InsufficientTrajectoriesError` when `--concurrency` exceeds the count of
   usable distinct trajectories (each trace produces one trajectory, traces
   with fewer than 2 turns are skipped). For a single-trace pool this caps
   concurrency at 1. The scenario already auto-injects a cache-bust marker
   that varies by lane index, so recycled plays of the same trace are
   provably distinct on the wire — the constraint is no longer load-bearing.

## Non-goals

- Changing how the recycle queue is sized or seeded at PROFILING start
  (already spans the full dataset).
- Reworking cache-bust digest inputs.
- Changing the `agentx-mvp` scenario's submission-validity rules beyond
  what's stated below.

## 1. Bug fix — `user_config.py`

**File:** `src/aiperf/common/config/user_config.py`
**Function:** `_should_use_fixed_schedule_for_trace_dataset`
**Line:** 412

Guard the `in` check with `isinstance(data, dict)`:

```python
if isinstance(data, dict) and "timestamp" in data and data["timestamp"] is not None:
    return True
```

Bare-scalar `orjson.loads("62")` returns `int`; `bool`/`str`/`list`/`None`
likewise lack the dict `in` semantics. The `isinstance` guard short-circuits
without changing the success path (a JSONL file with one JSON object per
line still hits `dict`).

**Test:** `tests/unit/common/config/test_user_config_trace_scan.py` (new) —
feed a multi-line pretty-printed `weka_trace`-style JSON file and assert
`_should_use_fixed_schedule_for_trace_dataset` returns `False` without
raising.

## 2. Trajectory reuse via wrap-fill

### Activation

**Automatic** when `concurrency > usable_trajectories`. No new CLI flag.
One `INFO`-level log line on activation. Submission validity stamp is
unchanged (`submission_valid=true` for agentx-mvp).

### TrajectorySource changes

**File:** `src/aiperf/timing/trajectory_source.py`

Add a wrap-fill phase after the existing distinct-build loop:

```python
distinct = self._build_trajectories()  # current behavior, may be < target_size
if not distinct:
    raise EmptyTracePoolError(...)

self.trajectories = distinct
if len(self.trajectories) < self._target_size:
    self.trajectories.extend(
        self._wrap_fill_lanes(distinct, self._target_size - len(distinct))
    )
    _logger.info(
        "Trajectory reuse: %d distinct trajectories fanned out to %d lanes "
        "(avg %.1f lanes per trace).",
        len(distinct), self._target_size, self._target_size / len(distinct),
    )
```

`_wrap_fill_lanes` cycles the distinct list and produces fresh
`Trajectory(conversation_id=src.conversation_id, start_turn_index=k_i)`
entries. `k_i` is sampled deterministically from
`np.random.default_rng(_seed_for_trace_lane(base_seed, conv_id, lane_index))`,
where `lane_index` is the absolute index of the new lane in
`self.trajectories`. This gives each shared-trace lane a distinct resume
point so they don't reduce to byte-identical replays.

Remove the post-build `InsufficientTrajectoriesError` raise. Keep
`EmptyTracePoolError` for the 0-trajectory degenerate case (all traces
have <2 turns, or pool is empty after filtering).

### AgenticReplayStrategy changes

**File:** `src/aiperf/timing/strategies/agentic_replay.py`

Two invariants assume `trace_id` uniqueness across lanes and must relax.
Both changes are unconditional — they remain correct when no wrap-fill
occurred (i.e. when every lane has a distinct trace_id, the new code paths
collapse to the old behavior).

1. **`_active_traces: set[str]` → `collections.Counter[str]`.**
   - `add(trace_id)` becomes `self._active_traces[trace_id] += 1`.
   - `discard(trace_id)` becomes a decrement with key removal at 0.
   - `_pop_next_eligible_trace`'s "skip if active" filter changes from
     `tid in _active_traces` to
     `self._active_traces[tid] >= self._lanes_per_trace[tid]`, where
     `self._lanes_per_trace` is a Counter built once at strategy init from
     the wrap-filled `trajectories` list. The skip now means "every lane
     for this trace is currently busy," not "any lane is busy."

2. **Double-recycle guard key: `trace_id` → `correlation_id`.**
   `_in_flight_recycled` currently raises `RuntimeError("Double recycle of
   trace_id …")` when two lanes legitimately finish the same trace.
   Re-key to `correlation_id` (or replace with
   `Set[tuple[str, str]]` of `(trace_id, correlation_id)`). The guard's
   real intent — catching the same `handle_credit_return` call firing
   twice for the same final turn — is preserved.

### Cache-bust dependency

The lane-distinctness relies on `_mint_marker_for_session` hashing
`(benchmark_id, recycle_pass, lane_index, trace_id)`. When wrap-fill is
active and `cache_bust.target == NONE`, traffic across shared-trace lanes
is byte-identical. Emit a `WARNING`-level log in that case. Do not
auto-promote (surprising) and do not error (some users may want that
behavior, e.g. for cache-saturation tests).

The `inferencex-agentx-mvp` scenario auto-locks
`cache_bust.target = first_turn_prefix`, so the warning never fires for
agentx-mvp runs.

### Submission validity

Unchanged. `submission_valid=true` even when wrap-fill is active. The
cache-bust marker preserves the per-replay distinctness that the AgentX
MVP recipe cares about; the number of *distinct conversation contexts*
is reduced, but that's a property of the input dataset, not of the
benchmark recipe.

## 3. Tests

### New

- `tests/unit/common/config/test_user_config_trace_scan.py` — bug-fix
  regression. Multi-line `weka_trace`-style JSON file →
  `_should_use_fixed_schedule_for_trace_dataset` returns `False`, no raise.
- `tests/unit/timing/test_trajectory_source_wrap_fill.py` — wrap-fill
  unit tests:
  - pool=1, conc=4 → 4 trajectories, same `conversation_id`, distinct
    `start_turn_index` across lanes (deterministic per seed).
  - pool=3, conc=10 → 10 trajectories, each distinct trace appears in
    3 or 4 lanes (balanced wrap).
  - pool=0 (all traces <2 turns) still raises `EmptyTracePoolError`.
- `tests/component_integration/test_agentic_replay_wrap_fill.py` — E2E
  smoke: pool=1, conc=4 → run completes; per-lane marker digests differ;
  no double-recycle errors logged.

### Update / delete

- `tests/component_integration/test_agentic_replay_pool_concurrency_integration.py`
  and the `*adversarial*` siblings: remove assertions that
  `concurrency > pool` raises `InsufficientTrajectoriesError`. Keep
  empty-pool / all-skipped-traces cases — those still raise
  `EmptyTracePoolError`.
- `tests/unit/timing/phase/test_runner_agentic_replay_warmup_target.py`:
  same — drop the concurrency-too-high cases, keep empty-pool.

## 4. Affected files (estimate)

| Path | Change kind |
|---|---|
| `src/aiperf/common/config/user_config.py` | 1-line guard |
| `src/aiperf/timing/trajectory_source.py` | Add wrap-fill, drop post-build raise |
| `src/aiperf/timing/strategies/agentic_replay.py` | Counter for `_active_traces`, correlation-id double-recycle guard, `_lanes_per_trace` |
| `src/aiperf/common/scenario/base.py` | Likely leave `InsufficientTrajectoriesError` class in place — still raised for empty pool? Decision deferred to plan step; favor delete if unused after refactor. |
| `tests/unit/common/config/test_user_config_trace_scan.py` | New |
| `tests/unit/timing/test_trajectory_source_wrap_fill.py` | New |
| `tests/component_integration/test_agentic_replay_wrap_fill.py` | New |
| `tests/unit/timing/test_trajectory_source_*adversarial*.py` | Drop concurrency-too-high cases |
| `tests/component_integration/test_agentic_replay_pool_concurrency_integration.py` | Drop concurrency-too-high cases |
| `tests/unit/timing/phase/test_runner_agentic_replay_warmup_target.py` | Drop concurrency-too-high cases |
| `docs/cli-options.md` | Regen (no CLI change; doc-gen idempotent) |
| `CHANGELOG.md` / scenario tutorial | Mention auto wrap-fill under "Notes" if a notes section exists for agentx-mvp |

## 5. Risks

- **Recycle-queue starvation with pool=1.** With a single-trace pool and
  every lane busy on the same trace, the next lane to finish will pop the
  same trace_id from the queue. Counter-based eligibility allows it. The
  scheduler ends up with all 32 lanes pinned to the one trace at all
  times. That's the intended outcome — but it's also what the original
  `InsufficientTrajectoriesError` was guarding against. Mitigation: the
  INFO log on activation makes the situation visible; users who want
  diverse traffic still need a larger trace pool.
- **Double-recycle guard semantic shift.** Re-keying to `correlation_id`
  changes what "the same final turn fired twice" means. There exists code
  that emits the same `correlation_id` on a deterministic retry path. The
  plan step should grep for `correlation_id` reuse before flipping the
  guard key, and add a unit test that pins the guard's behavior.
- **Cache-bust off + wrap-fill = identical traffic.** Documented and
  warned but not blocked. Users who set `cache_bust=NONE` explicitly are
  presumed to want this.
