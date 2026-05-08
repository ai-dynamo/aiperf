# `aiperf report weka-trace` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a CLI subcommand `aiperf report weka-trace <path>` that runs the existing `agentic_code_gen` HTML reporting pipeline against real Weka kv-cache-tester trace files, via a light reader that reuses `WekaTrace` pydantic models and skips the heavy loader path.

**Architecture:** A new light reader (`weka_input.py`) parses weka JSON files/directories using the existing `WekaTrace` pydantic models and emits `dict[session_id, list[ParsedTurn]]` — the exact shape the existing reporting pipeline already consumes. A new CLI subcommand wires it into `extract_metrics` / `extract_cache_metrics` / `render_plot_report` / `write_cache_structure` / `render_cache_explorer` / `render_simulation`. No tokenizer, no `UserConfig`, no `PromptGenerator`, no new renderers.

**Tech Stack:** Python 3.10+, pydantic v2, orjson, cyclopts (CLI), pytest.

**Spec:** `docs/superpowers/specs/2026-05-07-weka-trace-html-reports-design.md`

---

## File Structure

**Create:**
- `src/aiperf/dataset/agentic_code_gen/reporting/weka_input.py` — light reader: weka JSON file/dir → `dict[session_id, list[ParsedTurn]]` + `_parsed_to_sim_sessions` helper.
- `src/aiperf/cli_commands/report.py` — cyclopts `App(name="report")` with `weka-trace` default subcommand.
- `tests/unit/dataset/agentic_code_gen/test_weka_report_input.py` — unit tests for the reader.
- `tests/unit/cli_commands/test_report_weka_cli.py` — CLI smoke test.

**Modify:**
- `src/aiperf/cli.py` — register the new `report` subcommand.
- `src/aiperf/dataset/agentic_code_gen/reporting/report.py` — `_print_target_table` early-return when comparisons is empty.
- `src/aiperf/dataset/agentic_code_gen/reporting/cache_explorer.py` — `write_cache_structure` accepts a `block_size_override` kwarg so CLI-supplied block size flows into `cache_structure.json` even without a manifest.

**Existing test fixtures used (no changes):**
- `tests/fixtures/weka_traces/simple.json` — single normal-only trace, 2 turns.
- `tests/fixtures/weka_traces/one_subagent.json` — 1 parent + 1 subagent.
- `tests/fixtures/weka_traces/multi_model.json`, `terminal_subagent.json` — additional shapes.
- `tests/fixtures/weka_traces_small/trace_*.json` — directory of 5 traces for dir-mode tests.

---

## Task 1: Reader scaffold + parent-only sessions (no subagents yet)

**Files:**
- Create: `src/aiperf/dataset/agentic_code_gen/reporting/weka_input.py`
- Test: `tests/unit/dataset/agentic_code_gen/test_weka_report_input.py`

- [ ] **Step 1: Write the failing test for a single-file parent trace**

```python
# tests/unit/dataset/agentic_code_gen/test_weka_report_input.py
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the weka -> ParsedTurn light reader."""

from __future__ import annotations

from pathlib import Path

import pytest

from aiperf.dataset.agentic_code_gen.reporting.weka_input import load_weka_as_parsed

FIXTURES = Path(__file__).resolve().parents[3] / "fixtures" / "weka_traces"


def test_single_file_parent_normals_become_one_session() -> None:
    parsed = load_weka_as_parsed(FIXTURES / "simple.json")

    assert list(parsed.keys()) == ["trace_simple"]
    turns = parsed["trace_simple"]
    assert len(turns) == 2

    assert turns[0].session_id == "trace_simple"
    assert turns[0].input_length == 200
    assert turns[0].output_length == 30
    assert turns[0].hash_ids == [1, 2, 3]
    assert turns[0].delay_ms == 0.0
    assert turns[0].group_id is None
    assert turns[0].is_restart is False

    assert turns[1].input_length == 250
    assert turns[1].output_length == 40
    assert turns[1].hash_ids == [1, 2, 3, 4]
    # delay = (5.0 - 0.0) * 1000.0
    assert turns[1].delay_ms == pytest.approx(5000.0)
```

- [ ] **Step 2: Verify the test fails**

Run: `uv run pytest -n auto tests/unit/dataset/agentic_code_gen/test_weka_report_input.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'aiperf.dataset.agentic_code_gen.reporting.weka_input'`.

- [ ] **Step 3: Implement the parent-only reader**

```python
# src/aiperf/dataset/agentic_code_gen/reporting/weka_input.py
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Light reader: weka JSON file/dir -> ParsedTurn sessions for HTML reports.

Reuses the WekaTrace pydantic models from `weka_trace_models.py` and skips
the heavy WekaTraceLoader path entirely (no tokenizer, no UserConfig, no
PromptGenerator). Output shape matches what the existing reporting pipeline
already consumes: `dict[session_id, list[ParsedTurn]]`.
"""

from __future__ import annotations

from pathlib import Path

import orjson

from aiperf.dataset.agentic_code_gen.reporting.trace import ParsedTurn
from aiperf.dataset.loader.weka_trace_models import (
    WekaNormalRequest,
    WekaStreamingRequest,
    WekaTrace,
)


def _enumerate_files(path: Path) -> list[Path]:
    """Mirror WekaTraceLoader._enumerate_files: file or sorted *.json dir."""
    if path.is_dir():
        return sorted(path.glob("*.json"))
    return [path]


def _load_weka_traces(path: Path) -> list[WekaTrace]:
    """Parse every *.json under `path` (file or dir) into WekaTrace models."""
    traces: list[WekaTrace] = []
    for file_path in _enumerate_files(path):
        blob = orjson.loads(file_path.read_bytes())
        traces.append(WekaTrace.model_validate(blob))
    return traces


def _parent_session_turns(trace: WekaTrace) -> list[ParsedTurn]:
    """Build the ParsedTurn list for a parent trace's normal/streaming requests.

    delay_ms is computed between consecutive normal requests using their
    seconds-valued `t` field (subagent entries between them do not advance
    the previous-normal pointer; their `t` is on the parent's clock and what
    matters for report distributions is the gap between consecutive normals).
    """
    turns: list[ParsedTurn] = []
    prev_t: float | None = None
    for req in trace.requests:
        if not isinstance(req, WekaNormalRequest | WekaStreamingRequest):
            continue
        delay_ms = 0.0 if prev_t is None else (req.t - prev_t) * 1000.0
        turns.append(
            ParsedTurn(
                session_id=trace.id,
                input_length=req.input_length,
                output_length=req.output_length,
                hash_ids=req.hash_ids,
                delay_ms=delay_ms,
                group_id=None,
                is_restart=False,
            )
        )
        prev_t = req.t
    return turns


def load_weka_as_parsed(path: Path) -> dict[str, list[ParsedTurn]]:
    """Read a weka trace file or directory of *.json into ParsedTurn sessions.

    Each parent trace becomes one session keyed by `trace.id`. Subagent
    handling, max_context_length filtering, and other knobs are added in
    later tasks.
    """
    traces = _load_weka_traces(path)
    parsed: dict[str, list[ParsedTurn]] = {}
    for trace in traces:
        if trace.id in parsed:
            raise ValueError(
                f"Duplicate trace id '{trace.id}' across input files"
            )
        parsed[trace.id] = _parent_session_turns(trace)
    return parsed
```

Also create the test directory's `__init__.py` if missing:

```bash
test -f tests/unit/dataset/agentic_code_gen/__init__.py || touch tests/unit/dataset/agentic_code_gen/__init__.py
```

- [ ] **Step 4: Verify the test passes**

Run: `uv run pytest -n auto tests/unit/dataset/agentic_code_gen/test_weka_report_input.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/aiperf/dataset/agentic_code_gen/reporting/weka_input.py \
        tests/unit/dataset/agentic_code_gen/test_weka_report_input.py \
        tests/unit/dataset/agentic_code_gen/__init__.py
git commit -s -m "feat(reporting): add light weka -> ParsedTurn reader (parent-only)"
```

---

## Task 2: Directory mode + duplicate-id detection

**Files:**
- Modify: `tests/unit/dataset/agentic_code_gen/test_weka_report_input.py`

- [ ] **Step 1: Add directory-mode and duplicate-id tests**

Append to the test file:

```python
def test_directory_yields_one_session_per_trace() -> None:
    parsed = load_weka_as_parsed(
        Path(__file__).resolve().parents[3] / "fixtures" / "weka_traces_small"
    )
    # 5 trace files in this fixture dir
    assert len(parsed) == 5
    # Insertion order must match sorted(glob("*.json"))
    assert list(parsed.keys()) == sorted(parsed.keys())


def test_duplicate_trace_id_raises(tmp_path: Path) -> None:
    """Two files with the same trace.id in one dir is an error."""
    blob = (FIXTURES / "simple.json").read_bytes()
    (tmp_path / "a.json").write_bytes(blob)
    (tmp_path / "b.json").write_bytes(blob)

    with pytest.raises(ValueError, match="Duplicate trace id 'trace_simple'"):
        load_weka_as_parsed(tmp_path)
```

- [ ] **Step 2: Verify the directory test passes (logic already implemented in Task 1)**

Run: `uv run pytest -n auto tests/unit/dataset/agentic_code_gen/test_weka_report_input.py -v`
Expected: PASS for both new tests. The reader's `_enumerate_files` + duplicate-id check already cover this.

If `weka_traces_small` fixture filenames don't sort to identical insertion order, fix by replacing the assertion with `assert list(parsed.keys()) == [t.id for t in sorted_traces]` after manually loading; otherwise the simpler insertion-order check is sufficient because all five trace IDs in that fixture are distinct.

- [ ] **Step 3: Commit**

```bash
git add tests/unit/dataset/agentic_code_gen/test_weka_report_input.py
git commit -s -m "test(reporting): cover directory mode + duplicate trace id"
```

---

## Task 3: Subagent sessions

**Files:**
- Modify: `src/aiperf/dataset/agentic_code_gen/reporting/weka_input.py`
- Modify: `tests/unit/dataset/agentic_code_gen/test_weka_report_input.py`

- [ ] **Step 1: Write the failing tests for subagent handling**

Append to the test file:

```python
def test_subagent_becomes_separate_session() -> None:
    parsed = load_weka_as_parsed(FIXTURES / "one_subagent.json")

    # 1 parent + 1 subagent
    assert set(parsed.keys()) == {"trace_sa", "trace_sa::sa:agent_001"}

    parent = parsed["trace_sa"]
    # parent has two normals (the subagent entry between them is skipped)
    assert len(parent) == 2
    # delay between the two normals: (6.0 - 0.0) * 1000
    assert parent[0].delay_ms == 0.0
    assert parent[1].delay_ms == pytest.approx(6000.0)

    sub = parsed["trace_sa::sa:agent_001"]
    assert len(sub) == 1
    assert sub[0].input_length == 100
    assert sub[0].output_length == 50
    assert sub[0].hash_ids == [10, 11]
    assert sub[0].delay_ms == 0.0  # first turn of a session


def test_no_subagents_flag_omits_subagent_sessions() -> None:
    parsed = load_weka_as_parsed(
        FIXTURES / "one_subagent.json", include_subagents=False
    )
    assert set(parsed.keys()) == {"trace_sa"}
```

- [ ] **Step 2: Verify the new tests fail**

Run: `uv run pytest -n auto tests/unit/dataset/agentic_code_gen/test_weka_report_input.py::test_subagent_becomes_separate_session tests/unit/dataset/agentic_code_gen/test_weka_report_input.py::test_no_subagents_flag_omits_subagent_sessions -v`
Expected: FAIL — subagent sessions don't exist yet, and `include_subagents` is not a parameter.

- [ ] **Step 3: Add subagent support to the reader**

Edit `src/aiperf/dataset/agentic_code_gen/reporting/weka_input.py`:

Add `WekaSubagentEntry` to the imports:

```python
from aiperf.dataset.loader.weka_trace_models import (
    WekaNormalRequest,
    WekaStreamingRequest,
    WekaSubagentEntry,
    WekaTrace,
)
```

Add a helper for subagent sessions:

```python
def _subagent_session_turns(
    trace_id: str, entry: WekaSubagentEntry
) -> tuple[str, list[ParsedTurn]]:
    """Build (session_id, turns) for one subagent entry's nested normal requests.

    delay_ms is computed within the subagent's own request list, so the
    subagent's first turn always has delay_ms=0.0 (matches the convention
    used for parent-session turn 0).
    """
    session_id = f"{trace_id}::sa:{entry.agent_id}"
    turns: list[ParsedTurn] = []
    prev_t: float | None = None
    for req in entry.requests:
        delay_ms = 0.0 if prev_t is None else (req.t - prev_t) * 1000.0
        turns.append(
            ParsedTurn(
                session_id=session_id,
                input_length=req.input_length,
                output_length=req.output_length,
                hash_ids=req.hash_ids,
                delay_ms=delay_ms,
                group_id=None,
                is_restart=False,
            )
        )
        prev_t = req.t
    return session_id, turns
```

Update `load_weka_as_parsed`:

```python
def load_weka_as_parsed(
    path: Path,
    *,
    include_subagents: bool = True,
) -> dict[str, list[ParsedTurn]]:
    """Read a weka trace file or directory of *.json into ParsedTurn sessions.

    Each parent trace becomes one session keyed by `trace.id`. When
    include_subagents=True (default), each `WekaSubagentEntry` in the parent's
    request list also becomes a session keyed by `f"{trace.id}::sa:{agent_id}"`.
    """
    traces = _load_weka_traces(path)
    parsed: dict[str, list[ParsedTurn]] = {}
    for trace in traces:
        if trace.id in parsed:
            raise ValueError(
                f"Duplicate trace id '{trace.id}' across input files"
            )
        parsed[trace.id] = _parent_session_turns(trace)
        if include_subagents:
            for req in trace.requests:
                if isinstance(req, WekaSubagentEntry):
                    sid, turns = _subagent_session_turns(trace.id, req)
                    if sid in parsed:
                        raise ValueError(
                            f"Duplicate subagent session id '{sid}' in trace "
                            f"'{trace.id}'"
                        )
                    parsed[sid] = turns
    return parsed
```

- [ ] **Step 4: Verify all reader tests pass**

Run: `uv run pytest -n auto tests/unit/dataset/agentic_code_gen/test_weka_report_input.py -v`
Expected: PASS for all four tests.

- [ ] **Step 5: Commit**

```bash
git add src/aiperf/dataset/agentic_code_gen/reporting/weka_input.py \
        tests/unit/dataset/agentic_code_gen/test_weka_report_input.py
git commit -s -m "feat(reporting): emit subagent sessions from weka light reader"
```

---

## Task 4: `max_context_length` pre-filter

**Files:**
- Modify: `src/aiperf/dataset/agentic_code_gen/reporting/weka_input.py`
- Modify: `tests/unit/dataset/agentic_code_gen/test_weka_report_input.py`

- [ ] **Step 1: Write the failing test**

Append to the test file:

```python
def test_max_context_length_drops_oversized_traces() -> None:
    # simple.json has peak input_length=250; cap below that drops it.
    parsed = load_weka_as_parsed(FIXTURES / "simple.json", max_context_length=100)
    assert parsed == {}

    # Cap above the peak keeps it.
    parsed = load_weka_as_parsed(FIXTURES / "simple.json", max_context_length=1000)
    assert "trace_simple" in parsed


def test_max_context_length_drops_subagents_with_parent() -> None:
    # one_subagent.json parent peak input_length=400; cap=100 drops parent
    # and its subagent.
    parsed = load_weka_as_parsed(
        FIXTURES / "one_subagent.json", max_context_length=100
    )
    assert parsed == {}
```

- [ ] **Step 2: Verify the new tests fail**

Run: `uv run pytest -n auto tests/unit/dataset/agentic_code_gen/test_weka_report_input.py::test_max_context_length_drops_oversized_traces tests/unit/dataset/agentic_code_gen/test_weka_report_input.py::test_max_context_length_drops_subagents_with_parent -v`
Expected: FAIL — `max_context_length` is not a parameter yet.

- [ ] **Step 3: Add the filter**

Edit `weka_input.py`:

```python
def _parent_peak_input_length(trace: WekaTrace) -> int:
    """Peak `input_length` across the parent's normal/streaming requests.

    Mirrors WekaTraceLoader._filter_traces_by_max_context's rule.
    """
    peak = 0
    for req in trace.requests:
        if isinstance(req, WekaNormalRequest | WekaStreamingRequest):
            if req.input_length > peak:
                peak = req.input_length
    return peak
```

Update `load_weka_as_parsed`:

```python
def load_weka_as_parsed(
    path: Path,
    *,
    include_subagents: bool = True,
    max_context_length: int | None = None,
) -> dict[str, list[ParsedTurn]]:
    """Read a weka trace file or directory of *.json into ParsedTurn sessions.

    Each parent trace becomes one session keyed by `trace.id`. When
    include_subagents=True (default), each `WekaSubagentEntry` in the parent's
    request list also becomes a session keyed by `f"{trace.id}::sa:{agent_id}"`.

    When max_context_length is set, traces whose parent peak input_length
    exceeds the cap are dropped entirely (parent and subagents).
    """
    traces = _load_weka_traces(path)
    parsed: dict[str, list[ParsedTurn]] = {}
    for trace in traces:
        if (
            max_context_length is not None
            and _parent_peak_input_length(trace) > max_context_length
        ):
            continue
        if trace.id in parsed:
            raise ValueError(
                f"Duplicate trace id '{trace.id}' across input files"
            )
        parsed[trace.id] = _parent_session_turns(trace)
        if include_subagents:
            for req in trace.requests:
                if isinstance(req, WekaSubagentEntry):
                    sid, turns = _subagent_session_turns(trace.id, req)
                    if sid in parsed:
                        raise ValueError(
                            f"Duplicate subagent session id '{sid}' in trace "
                            f"'{trace.id}'"
                        )
                    parsed[sid] = turns
    return parsed
```

- [ ] **Step 4: Verify all reader tests pass**

Run: `uv run pytest -n auto tests/unit/dataset/agentic_code_gen/test_weka_report_input.py -v`
Expected: PASS for all six tests.

- [ ] **Step 5: Commit**

```bash
git add src/aiperf/dataset/agentic_code_gen/reporting/weka_input.py \
        tests/unit/dataset/agentic_code_gen/test_weka_report_input.py
git commit -s -m "feat(reporting): add max_context_length pre-filter to weka reader"
```

---

## Task 5: Simulation-shape converter

**Files:**
- Modify: `src/aiperf/dataset/agentic_code_gen/reporting/weka_input.py`
- Modify: `tests/unit/dataset/agentic_code_gen/test_weka_report_input.py`

The existing `simulation.render_simulation` consumes `list[dict]` (the same shape produced by `trace.load_simulation_sessions(jsonl_path)`). Since we don't have a JSONL, write an in-memory equivalent that converts our `dict[session_id, list[ParsedTurn]]` directly.

- [ ] **Step 1: Write the failing test**

Append to the test file:

```python
def test_parsed_to_sim_sessions_shape() -> None:
    from aiperf.dataset.agentic_code_gen.reporting.weka_input import (
        parsed_to_sim_sessions,
    )

    parsed = load_weka_as_parsed(FIXTURES / "simple.json")
    sim = parsed_to_sim_sessions(parsed)

    assert len(sim) == 1
    s = sim[0]
    assert s["session_id"] == "trace_simple"
    assert s["group_id"] == 0
    assert s["is_restart"] is False
    assert len(s["turns"]) == 2

    t0, t1 = s["turns"]
    assert t0["input_length"] == 200
    assert t0["output_length"] == 30
    assert t0["delay_ms"] == 0.0
    assert t0["hash_ids"] == [1, 2, 3]
    # cumulative_input_length = running sum of input + output prior to and
    # including the current input. Matches load_simulation_sessions's rule:
    # cumulative += input_length (before append), then cumulative += output_length.
    assert t0["cumulative_input_length"] == 200

    assert t1["input_length"] == 250
    assert t1["delay_ms"] == pytest.approx(5000.0)
    # 200 (in0) + 30 (out0) + 250 (in1) = 480
    assert t1["cumulative_input_length"] == 480
```

- [ ] **Step 2: Verify the test fails**

Run: `uv run pytest -n auto tests/unit/dataset/agentic_code_gen/test_weka_report_input.py::test_parsed_to_sim_sessions_shape -v`
Expected: FAIL — `parsed_to_sim_sessions` does not exist.

- [ ] **Step 3: Implement `parsed_to_sim_sessions`**

Append to `weka_input.py`:

```python
def parsed_to_sim_sessions(
    parsed: dict[str, list[ParsedTurn]],
) -> list[dict]:
    """Convert ParsedTurn sessions to the dict shape `render_simulation` expects.

    Mirrors `trace.load_simulation_sessions` but operates in-memory rather
    than reading a JSONL file. cumulative_input_length is the running sum
    of input + output tokens up to and including the current turn's input.
    """
    result: list[dict] = []
    for session_id, turns in parsed.items():
        cumulative = 0
        sim_turns: list[dict] = []
        for turn in turns:
            cumulative += turn.input_length
            sim_turns.append(
                {
                    "input_length": turn.input_length,
                    "output_length": turn.output_length,
                    "delay_ms": turn.delay_ms,
                    "hash_ids": turn.hash_ids,
                    "cumulative_input_length": cumulative,
                }
            )
            cumulative += turn.output_length

        first = turns[0] if turns else None
        result.append(
            {
                "session_id": session_id,
                "group_id": first.group_id if first and first.group_id else 0,
                "is_restart": first.is_restart if first else False,
                "turns": sim_turns,
            }
        )
    return result
```

- [ ] **Step 4: Verify the test passes**

Run: `uv run pytest -n auto tests/unit/dataset/agentic_code_gen/test_weka_report_input.py -v`
Expected: PASS for all seven tests.

- [ ] **Step 5: Commit**

```bash
git add src/aiperf/dataset/agentic_code_gen/reporting/weka_input.py \
        tests/unit/dataset/agentic_code_gen/test_weka_report_input.py
git commit -s -m "feat(reporting): add parsed_to_sim_sessions for simulation HTML"
```

---

## Task 6: Guard `_print_target_table` against empty comparisons

**Files:**
- Modify: `src/aiperf/dataset/agentic_code_gen/reporting/report.py`
- Test: `tests/unit/dataset/agentic_code_gen/test_report.py` (existing file — append)

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/dataset/agentic_code_gen/test_report.py`:

```python
def test_print_target_table_skips_when_no_comparisons() -> None:
    """When ReportData.comparisons is empty (no manifest), the target table
    should be omitted entirely rather than rendered as a header with no rows."""
    from rich.console import Console

    from aiperf.dataset.agentic_code_gen.reporting.metrics import (
        PercentileStats,
        ReportData,
    )
    from aiperf.dataset.agentic_code_gen.reporting.report import _print_target_table

    empty_stats = PercentileStats(
        mean=0.0, median=0.0, p05=0.0, p25=0.0, p75=0.0, p95=0.0, p99=0.0
    )
    data = ReportData(
        session_count=0,
        total_turns=0,
        comparisons=[],
        hash_id_block_stats=empty_stats,
        request_latency_stats=empty_stats,
        session_duration_min_stats=empty_stats,
    )

    console = Console(record=True, width=140)
    _print_target_table(console, data)
    assert "Target vs Observed" not in console.export_text()
```

(If `PercentileStats` requires more fields, run `python -c "from aiperf.dataset.agentic_code_gen.models import PercentileStats; print(PercentileStats.model_fields)"` and add them with zero values. The test only cares that the table is skipped.)

- [ ] **Step 2: Verify the test fails**

Run: `uv run pytest -n auto tests/unit/dataset/agentic_code_gen/test_report.py::test_print_target_table_skips_when_no_comparisons -v`
Expected: FAIL — current `_print_target_table` always prints the header.

- [ ] **Step 3: Add the guard**

Edit `src/aiperf/dataset/agentic_code_gen/reporting/report.py`:

```python
def _print_target_table(console: Console, data: ReportData) -> None:
    if not data.comparisons:
        return
    table = Table(title="Target vs Observed")
    # ... existing body unchanged
```

- [ ] **Step 4: Verify all reporting tests pass**

Run: `uv run pytest -n auto tests/unit/dataset/agentic_code_gen/test_report.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/aiperf/dataset/agentic_code_gen/reporting/report.py \
        tests/unit/dataset/agentic_code_gen/test_report.py
git commit -s -m "fix(reporting): skip target-vs-observed table when no comparisons"
```

---

## Task 7: `write_cache_structure` honors a `block_size` override

The current implementation defaults to `block_size=512` when manifest is None. We want the CLI's `--block-size` to flow into `cache_structure.json` even without a manifest. Add a kwarg.

**Files:**
- Modify: `src/aiperf/dataset/agentic_code_gen/reporting/cache_explorer.py`
- Test: `tests/unit/dataset/agentic_code_gen/test_report.py` (existing file — append) OR new file. Use existing for cohesion.

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/dataset/agentic_code_gen/test_report.py`:

```python
def test_write_cache_structure_block_size_override(tmp_path) -> None:
    """When manifest is None and a block_size override is provided, the
    written cache_structure.json must use the override (not the 512 default)."""
    import orjson

    from aiperf.dataset.agentic_code_gen.reporting.cache_explorer import (
        write_cache_structure,
    )
    from aiperf.dataset.agentic_code_gen.reporting.trace import ParsedTurn

    sessions = {
        "s1": [
            ParsedTurn(
                session_id="s1",
                input_length=100,
                output_length=10,
                hash_ids=[1, 2, 3],
                delay_ms=0.0,
            )
        ]
    }
    write_cache_structure(
        sessions, manifest=None, output_dir=tmp_path, block_size_override=64
    )
    payload = orjson.loads((tmp_path / "cache_structure.json").read_bytes())
    assert payload["block_size"] == 64
```

- [ ] **Step 2: Verify the test fails**

Run: `uv run pytest -n auto tests/unit/dataset/agentic_code_gen/test_report.py::test_write_cache_structure_block_size_override -v`
Expected: FAIL — `block_size_override` is not a parameter.

- [ ] **Step 3: Add the override kwarg**

Edit `src/aiperf/dataset/agentic_code_gen/reporting/cache_explorer.py`:

```python
def write_cache_structure(
    sessions: dict[str, list[ParsedTurn]],
    manifest: DatasetManifest | None,
    output_dir: Path,
    *,
    block_size_override: int | None = None,
) -> dict:
    """Generate cache_structure.json with per-session block classification.

    block_size_override takes precedence over manifest's block_size and the
    built-in default (512). Used by the weka real-trace report path where
    no manifest is written but the CLI's --block-size should still flow.
    """
    default_cache = CacheLayerConfig()
    l1_tokens = default_cache.layer1_tokens
    l15_tokens = default_cache.layer1_5_tokens
    block_size = 512
    if manifest:
        block_size = manifest.generation_params.block_size
        l1_tokens = manifest.generation_params.cache.layer1_tokens
        l15_tokens = manifest.generation_params.cache.layer1_5_tokens
    if block_size_override is not None:
        block_size = block_size_override
    l1_blocks = math.ceil(l1_tokens / block_size) if block_size > 0 else 0
    l15_blocks_count = math.ceil(l15_tokens / block_size) if block_size > 0 else 0
    # ... rest of the body unchanged
```

- [ ] **Step 4: Verify the test passes**

Run: `uv run pytest -n auto tests/unit/dataset/agentic_code_gen/test_report.py::test_write_cache_structure_block_size_override -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/aiperf/dataset/agentic_code_gen/reporting/cache_explorer.py \
        tests/unit/dataset/agentic_code_gen/test_report.py
git commit -s -m "feat(reporting): write_cache_structure accepts block_size override"
```

---

## Task 8: CLI command `aiperf report weka-trace`

**Files:**
- Create: `src/aiperf/cli_commands/report.py`
- Modify: `src/aiperf/cli.py`
- Test: `tests/unit/cli_commands/test_report_weka_cli.py`

- [ ] **Step 1: Write the failing CLI smoke test**

```python
# tests/unit/cli_commands/test_report_weka_cli.py
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end smoke test for `aiperf report weka-trace`."""

from __future__ import annotations

from pathlib import Path

FIXTURES_DIR = (
    Path(__file__).resolve().parents[2] / "fixtures" / "weka_traces_small"
)


def test_report_weka_trace_writes_three_html_files(tmp_path: Path) -> None:
    from aiperf.cli_commands.report import report_weka_trace

    report_weka_trace(
        path=FIXTURES_DIR,
        output=tmp_path,
        block_size=64,
    )

    run_dirs = list(tmp_path.glob("weka-report_*"))
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]

    for name in ("report.html", "cache_explorer.html", "simulation.html"):
        path = run_dir / name
        assert path.exists(), f"missing {name}"
        assert path.stat().st_size > 0, f"{name} is empty"

    cache_json = run_dir / "cache_structure.json"
    assert cache_json.exists() and cache_json.stat().st_size > 0
```

Create the test directory's `__init__.py` if it doesn't exist:

```bash
test -f tests/unit/cli_commands/__init__.py || touch tests/unit/cli_commands/__init__.py
```

- [ ] **Step 2: Verify the test fails**

Run: `uv run pytest -n auto tests/unit/cli_commands/test_report_weka_cli.py -v`
Expected: FAIL — `aiperf.cli_commands.report` does not exist.

- [ ] **Step 3: Create the CLI module**

```python
# src/aiperf/cli_commands/report.py
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CLI commands for generating HTML reports from real trace files."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Literal

from cyclopts import App, Parameter

app = App(name="report")


@app.default
def report(
    target: Annotated[
        Literal["weka-trace"],
        Parameter(help="Trace flavor to report on."),
    ],
    path: Path,
    *,
    output: Path = Path("."),
    block_size: int = 512,
    max_context_length: int | None = None,
    no_subagents: bool = False,
    prefill_tps: float = 20_000,
    decode_tps: float = 60,
) -> None:
    """Render HTML reports (report.html, cache_explorer.html, simulation.html)
    for a real trace file or directory.

    Examples:
        aiperf report weka-trace ./traces/
        aiperf report weka-trace ./traces/ --block-size 64
        aiperf report weka-trace ./traces/ --max-context-length 200000
        aiperf report weka-trace ./traces/ --no-subagents

    Args:
        target: Trace flavor (currently only `weka-trace`).
        path: Path to a trace file or a directory of *.json trace files.
        output: Parent directory for the auto-named run directory.
        block_size: KV cache block size for cache statistics.
        max_context_length: Drop traces whose peak input_length exceeds this.
        no_subagents: Skip subagent sessions; report only parent traces.
        prefill_tps: Synthetic prefill throughput for latency estimates.
        decode_tps: Synthetic decode throughput for latency estimates.
    """
    match target:
        case "weka-trace":
            report_weka_trace(
                path=path,
                output=output,
                block_size=block_size,
                max_context_length=max_context_length,
                no_subagents=no_subagents,
                prefill_tps=prefill_tps,
                decode_tps=decode_tps,
            )


def report_weka_trace(
    *,
    path: Path,
    output: Path = Path("."),
    block_size: int = 512,
    max_context_length: int | None = None,
    no_subagents: bool = False,
    prefill_tps: float = 20_000,
    decode_tps: float = 60,
) -> None:
    """Render HTML reports for a weka trace file or directory.

    Writes an auto-named run directory `weka-report_<basename>_<UTC-ts>/`
    containing report.html, cache_explorer.html, simulation.html, and
    cache_structure.json.
    """
    from rich.console import Console

    from aiperf.dataset.agentic_code_gen.reporting.cache_explorer import (
        render_cache_explorer,
        write_cache_structure,
    )
    from aiperf.dataset.agentic_code_gen.reporting.metrics import (
        build_report_data,
        extract_cache_metrics,
        extract_metrics,
    )
    from aiperf.dataset.agentic_code_gen.reporting.plot_report import (
        render_plot_report,
    )
    from aiperf.dataset.agentic_code_gen.reporting.report import (
        _print_report_to_console,
    )
    from aiperf.dataset.agentic_code_gen.reporting.simulation import (
        render_simulation,
    )
    from aiperf.dataset.agentic_code_gen.reporting.weka_input import (
        load_weka_as_parsed,
        parsed_to_sim_sessions,
    )

    console = Console()

    parsed = load_weka_as_parsed(
        path,
        include_subagents=not no_subagents,
        max_context_length=max_context_length,
    )
    if not parsed:
        console.print(
            "[yellow]No traces matched the input "
            "(empty directory or all dropped by --max-context-length).[/yellow]"
        )
        raise SystemExit(1)

    basename = path.stem if path.is_file() else path.name
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_dir = output / f"weka-report_{basename}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=False)

    metrics = extract_metrics(
        parsed, prefill_tps=prefill_tps, decode_tps=decode_tps
    )
    metrics.update(extract_cache_metrics(parsed, block_size=block_size))
    report_data = build_report_data(metrics, manifest=None)

    render_plot_report(metrics, parsed, run_dir)
    cache_payload = write_cache_structure(
        parsed, manifest=None, output_dir=run_dir, block_size_override=block_size
    )
    render_cache_explorer(run_dir, cache_payload)

    sim_sessions = parsed_to_sim_sessions(parsed)
    render_simulation(
        sim_sessions, run_dir / "simulation.html", block_size=block_size
    )

    _print_report_to_console(report_data)
    console.print(f"[green]Run directory: {run_dir}[/green]")
    console.print(f"  Report:          {run_dir / 'report.html'}")
    console.print(f"  Cache explorer:  {run_dir / 'cache_explorer.html'}")
    console.print(f"  Simulation:      {run_dir / 'simulation.html'}")
```

- [ ] **Step 4: Wire into the top-level CLI**

Edit `src/aiperf/cli.py`. Insert the new line in the registration block (alphabetical placement, between `plugins` and `service`):

```python
app.command("aiperf.cli_commands.analyze_trace:app", name="analyze-trace")
app.command("aiperf.cli_commands.profile:app", name="profile")
app.command("aiperf.cli_commands.plot:app", name="plot")
app.command("aiperf.cli_commands.plugins:app", name="plugins")
app.command("aiperf.cli_commands.report:app", name="report")
app.command("aiperf.cli_commands.service:app", name="service")
app.command("aiperf.cli_commands.speed_bench_report:app", name="speed-bench-report")
app.command("aiperf.cli_commands.synthesize:app", name="synthesize")
app.command("aiperf.cli_commands.validate:app", name="validate")
```

- [ ] **Step 5: Verify the smoke test passes**

Run: `uv run pytest -n auto tests/unit/cli_commands/test_report_weka_cli.py -v`
Expected: PASS.

- [ ] **Step 6: Manually exercise the CLI to verify end-to-end**

Run:

```bash
uv run aiperf report weka-trace tests/fixtures/weka_traces_small --output /tmp --block-size 64
```

Expected:
- Console prints "Dataset Report" header (no Target vs Observed table since comparisons is empty).
- Output ends with run-dir paths.
- `/tmp/weka-report_weka_traces_small_<ts>/{report,cache_explorer,simulation}.html` exist and are non-empty.

- [ ] **Step 7: Commit**

```bash
git add src/aiperf/cli_commands/report.py src/aiperf/cli.py \
        tests/unit/cli_commands/test_report_weka_cli.py \
        tests/unit/cli_commands/__init__.py
git commit -s -m "feat(cli): add \`aiperf report weka-trace\` HTML report command"
```

---

## Task 9: Regenerate CLI docs

The `generate-cli-docs` pre-commit hook auto-regenerates `docs/cli-options.md` from the cyclopts app. Run it explicitly so the doc change is part of the same PR rather than triggered by a downstream hook on someone else's commit.

**Files:**
- Modify: `docs/cli-options.md` (auto-regenerated)

- [ ] **Step 1: Regenerate the docs**

Run:

```bash
make generate-all-docs
```

(If `make generate-all-docs` does more than CLI docs and that introduces unrelated diffs, fall back to the narrower target: check the Makefile for `generate-cli-docs` and run that instead.)

- [ ] **Step 2: Verify the new command appears**

Run:

```bash
grep -n "aiperf report" docs/cli-options.md | head
```

Expected: at least one match showing the `report weka-trace` subcommand and its options.

- [ ] **Step 3: Commit**

```bash
git add docs/cli-options.md
git commit -s -m "docs: regenerate cli-options.md for \`aiperf report weka-trace\`"
```

---

## Task 10: Final verification

- [ ] **Step 1: Run the full unit suite**

Run: `uv run pytest -n auto tests/unit/`
Expected: PASS (no regressions introduced by `_print_target_table` guard or `write_cache_structure` signature change).

- [ ] **Step 2: Run pre-commit on all touched files**

Run:

```bash
pre-commit run --files \
  src/aiperf/cli.py \
  src/aiperf/cli_commands/report.py \
  src/aiperf/dataset/agentic_code_gen/reporting/weka_input.py \
  src/aiperf/dataset/agentic_code_gen/reporting/report.py \
  src/aiperf/dataset/agentic_code_gen/reporting/cache_explorer.py \
  tests/unit/dataset/agentic_code_gen/test_weka_report_input.py \
  tests/unit/dataset/agentic_code_gen/test_report.py \
  tests/unit/cli_commands/test_report_weka_cli.py \
  docs/cli-options.md
```

Expected: All hooks PASS. If `ruff` reformats anything, re-stage and amend the most recent commit (or, if multiple commits are touched, just `git commit -s -m "chore: ruff format"` with the deltas).

- [ ] **Step 3: Manual end-to-end check on a real-shape input**

Run:

```bash
uv run aiperf report weka-trace tests/fixtures/weka_traces --output /tmp/weka-report-check --block-size 64
ls /tmp/weka-report-check/weka-report_*/
xdg-open /tmp/weka-report-check/weka-report_*/report.html  # optional, visual check
```

Expected: All three HTML files exist; `report.html` shows ISL/OSL/cache histograms; no Target-vs-Observed table; subagent sessions are visible in the simulation.

- [ ] **Step 4: Verify no stash files, clean tree**

```bash
git status
```

Expected: clean working tree on the feature branch.

---

## Spec Coverage Self-Check

| Spec section | Implemented in |
|---|---|
| New `weka_input.py` light reader | Tasks 1, 3, 4, 5 |
| `WekaTrace` model reuse, file/dir auto-detect | Task 1 |
| Parent session = trace.id, normals + delays | Task 1 |
| Subagent sessions = `f"{trace.id}::sa:{agent_id}"` | Task 3 |
| `--no-subagents` flag | Task 3 + 8 |
| `--max-context-length` filter | Tasks 4 + 8 |
| `_parsed_to_sim_sessions` (renamed `parsed_to_sim_sessions`) | Task 5 |
| `_print_target_table` empty-comparisons guard | Task 6 |
| `write_cache_structure` honors CLI block_size when manifest absent | Task 7 |
| `aiperf report weka-trace` CLI with all flags | Task 8 |
| Auto-named output dir `weka-report_<basename>_<UTC-ts>/` | Task 8 |
| No `manifest.json`, no `comparison.txt` | Task 8 (we simply don't write them) |
| Console echo paths | Task 8 |
| Unit tests for reader (parent / dir / subagent / no-subagents / max-ctx / sim-shape) | Tasks 1–5 |
| CLI smoke test | Task 8 |
| `cli.py` registration | Task 8 |
| Regenerate `docs/cli-options.md` | Task 9 |

All spec requirements have a task. No placeholders. Method names consistent across tasks (`load_weka_as_parsed`, `parsed_to_sim_sessions`, `report_weka_trace`, `block_size_override`).
