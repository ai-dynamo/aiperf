<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# T-Digest Aggregator for `InterChunkLatencyMetric` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `MetricArray` with a `TDigestListMetricAggregator` for list-valued record metrics in `MetricResultsProcessor` so `InterChunkLatencyMetric` no longer accumulates a 40 GB run-level array at 1 M-request ramp scale.

**Architecture:** New module `src/aiperf/metrics/list_metric_aggregation.py` exposes a single aggregator class that mirrors `MetricArray`'s `to_result(tag, header, unit) -> MetricResult` contract. The class wraps `tdigest.TDigest` for percentiles and maintains five running side-channel scalars (`count`, `sum`, `sum_sq`, `min`, `max`) so `count`, `sum`, `min`, `max`, `avg`, `std` stay bit-exact. `MetricResultsProcessor.process_result` picks the aggregator type at first-touch based on whether the value is a `list`; `_create_metric_result` gains one `isinstance` branch. No flag, no mode, no config knob.

**Tech Stack:** Python 3.10+, `tdigest~=0.5.2.2` (new dependency), `pytest -n auto`, `numpy` (already present, used for parity tests), Pydantic `MetricResult` (existing).

**Spec:** `docs/superpowers/specs/2026-04-29-tdigest-icl-aggregator-design.md` (commit `d0775d9a6`).

**Working directory for all tasks:** `/home/anthony/nvidia/projects/aiperf/ajc/tdigest-icl-aggregator`

**Branch:** `ajc/tdigest-icl-aggregator` (already exists, tracking `origin/main`).

---

## File Map

| Action | Path | Responsibility |
|---|---|---|
| Modify | `pyproject.toml` | Add `tdigest~=0.5.2.2` to `dependencies`. |
| Create | `src/aiperf/metrics/list_metric_aggregation.py` | `TDigestListMetricAggregator` class — t-digest sketch + 5 running scalars; `append`, `extend`, `to_result`. |
| Create | `tests/unit/metrics/test_list_metric_aggregation.py` | Unit tests for the aggregator: empty, single-value, extend, min/max/count/sum/avg/std exactness, percentile tolerance, schema parity vs `MetricArray`. |
| Modify | `src/aiperf/post_processors/metric_results_processor.py` | (1) import `TDigestListMetricAggregator`; (2) at first-touch in `process_result`, choose aggregator type by `isinstance(value, list)`; (3) add isinstance branch in `_create_metric_result`. |
| Modify | `tests/unit/post_processors/test_metric_results_processor.py` | Retrofit `test_process_result_record_metric_list_values` to assert the new aggregator type and stat surface; scalar test stays untouched. |

**Size budget (acceptance criterion #7 from spec):** ≤200 LOC of production code total. Target: ~80 LOC aggregator + ~10 LOC processor wiring + 1 toml line.

---

## Task 1: Add `tdigest` dependency

**Files:**
- Modify: `pyproject.toml` (add one line under `dependencies`)

- [ ] **Step 1.1: Add the dependency via `uv`**

```bash
cd /home/anthony/nvidia/projects/aiperf/ajc/tdigest-icl-aggregator
uv add 'tdigest~=0.5.2.2'
```

Expected: `uv` adds the line to `[project].dependencies` in `pyproject.toml`, updates `uv.lock`, and installs into `.venv`. Stdout shows `+ tdigest==0.5.2.2`.

- [ ] **Step 1.2: Verify the dependency works**

```bash
cd /home/anthony/nvidia/projects/aiperf/ajc/tdigest-icl-aggregator
uv run python -c "import tdigest; td = tdigest.TDigest(); td.update(1.0); td.update(2.0); td.update(3.0); print('p50=', td.percentile(50)); print('n=', td.n)"
```

Expected: prints `p50= 2.0` (or close) and `n= 3.0`. No `ModuleNotFoundError`.

- [ ] **Step 1.3: Commit**

```bash
cd /home/anthony/nvidia/projects/aiperf/ajc/tdigest-icl-aggregator
git add pyproject.toml uv.lock
git commit -s -m "$(cat <<'EOF'
chore(deps): add tdigest~=0.5.2.2 for list-metric aggregation

Used by TDigestListMetricAggregator (next commit) to bound run-level
memory of list-valued record metrics. Pure-Python, no transitive C/Rust
deps.
EOF
)"
```

Expected: pre-commit hooks run; commit succeeds. (If pre-commit modifies `uv.lock` ordering, re-stage and retry — do NOT use `--amend`, do NOT use `--no-verify`.)

---

## Task 2: `TDigestListMetricAggregator` (TDD)

**Files:**
- Create: `src/aiperf/metrics/list_metric_aggregation.py`
- Create: `tests/unit/metrics/test_list_metric_aggregation.py`

The aggregator class is consumed by Task 3, so it must land first. Tests are written first per project policy (TDD).

- [ ] **Step 2.1: Write the test file (all 10 tests)**

Create `tests/unit/metrics/test_list_metric_aggregation.py` with exactly this content:

```python
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ``TDigestListMetricAggregator``.

The aggregator is the run-level storage for list-valued record metrics
(today only ``inter_chunk_latency``). It backs percentile reads with a
t-digest sketch but keeps ``count`` / ``sum`` / ``min`` / ``max`` /
``avg`` / ``std`` bit-exact via running side-channel scalars.
"""

from __future__ import annotations

import numpy as np
import pytest

from aiperf.metrics.list_metric_aggregation import TDigestListMetricAggregator
from aiperf.metrics.metric_dicts import MetricArray


class TestTDigestListMetricAggregator:
    """Behavioral contract for the aggregator."""

    def test_empty_aggregator_returns_count_zero_result(self) -> None:
        agg = TDigestListMetricAggregator()
        result = agg.to_result(tag="test", header="Test", unit="ms")
        assert result.tag == "test"
        assert result.header == "Test"
        assert result.unit == "ms"
        assert result.count == 0
        assert result.sum is None
        assert result.min is None
        assert result.max is None
        assert result.avg is None
        assert result.std is None
        assert result.p50 is None
        assert result.p99 is None

    def test_append_single_value_count_one(self) -> None:
        agg = TDigestListMetricAggregator()
        agg.append(7.0)
        result = agg.to_result(tag="t", header="T", unit="ms")
        assert result.count == 1
        assert result.sum == pytest.approx(7.0)
        assert result.min == pytest.approx(7.0)
        assert result.max == pytest.approx(7.0)
        assert result.avg == pytest.approx(7.0)
        assert result.std == pytest.approx(0.0)
        # Percentiles of a single point all collapse to that point.
        assert result.p50 == pytest.approx(7.0)

    def test_extend_with_list_count_matches_len(self) -> None:
        agg = TDigestListMetricAggregator()
        agg.extend([1.0, 2.0, 3.0, 4.0, 5.0])
        result = agg.to_result(tag="t", header="T", unit="ms")
        assert result.count == 5
        assert result.sum == pytest.approx(15.0)

    def test_repeated_extend_accumulates(self) -> None:
        agg = TDigestListMetricAggregator()
        agg.extend([1.0, 2.0, 3.0])
        agg.extend([4.0, 5.0])
        result = agg.to_result(tag="t", header="T", unit="ms")
        assert result.count == 5
        assert result.sum == pytest.approx(15.0)
        assert result.min == pytest.approx(1.0)
        assert result.max == pytest.approx(5.0)

    def test_mixed_int_and_float_values(self) -> None:
        agg = TDigestListMetricAggregator()
        agg.append(1)         # int
        agg.append(2.5)       # float
        agg.extend([3, 4.5])  # mixed list
        result = agg.to_result(tag="t", header="T", unit="ms")
        assert result.count == 4
        assert result.sum == pytest.approx(11.0)

    def test_min_max_exact_across_random_inputs(self) -> None:
        rng = np.random.default_rng(42)
        values = rng.uniform(low=-1000.0, high=1000.0, size=10_000)
        agg = TDigestListMetricAggregator()
        agg.extend(values.tolist())
        result = agg.to_result(tag="t", header="T", unit="ms")
        # Bit-exact min/max via running side-channel.
        assert result.min == float(values.min())
        assert result.max == float(values.max())

    def test_count_sum_exact_across_random_inputs(self) -> None:
        rng = np.random.default_rng(42)
        values = rng.uniform(low=0.0, high=1000.0, size=10_000)
        agg = TDigestListMetricAggregator()
        agg.extend(values.tolist())
        result = agg.to_result(tag="t", header="T", unit="ms")
        assert result.count == 10_000
        # Exact within float64 round-off (sum order matters slightly).
        assert result.sum == pytest.approx(float(values.sum()), rel=1e-12)

    def test_avg_std_exact_against_numpy(self) -> None:
        rng = np.random.default_rng(42)
        values = rng.normal(loc=100.0, scale=15.0, size=10_000)
        agg = TDigestListMetricAggregator()
        agg.extend(values.tolist())
        result = agg.to_result(tag="t", header="T", unit="ms")
        # avg = sum / count; std = sqrt(max(0, sum_sq/count - avg^2))
        # both within float64 round-off of numpy's reference.
        assert result.avg == pytest.approx(float(np.mean(values)), rel=1e-9)
        assert result.std == pytest.approx(float(np.std(values)), rel=1e-9)

    def test_percentiles_within_tolerance(self) -> None:
        rng = np.random.default_rng(42)
        values = rng.uniform(low=0.0, high=1000.0, size=100_000)
        agg = TDigestListMetricAggregator()
        agg.extend(values.tolist())
        result = agg.to_result(tag="t", header="T", unit="ms")
        # T-digest's documented relative error band on percentiles is
        # well under 1% at this sample size; we hold ourselves to 0.5%.
        assert result.p50 == pytest.approx(float(np.percentile(values, 50)), rel=0.005)
        assert result.p90 == pytest.approx(float(np.percentile(values, 90)), rel=0.005)
        assert result.p99 == pytest.approx(float(np.percentile(values, 99)), rel=0.005)

    def test_to_result_schema_matches_metric_array(self) -> None:
        # 100k samples — t-digest's relative error on percentile *values*
        # at extreme tails (e.g. p1 on uniform[0, 1000] where the true
        # value is ~9.66, small relative to the data range) only stays
        # within 0.5% at this sample size. Smaller N exhibits 1% relative
        # error at p1 even though rank accuracy is well within the
        # documented t-digest bound.
        rng = np.random.default_rng(42)
        values = rng.uniform(low=0.0, high=1000.0, size=100_000)

        digest_agg = TDigestListMetricAggregator()
        digest_agg.extend(values.tolist())
        digest_result = digest_agg.to_result(tag="t", header="T", unit="ms")

        array_agg = MetricArray()
        array_agg.extend(values.tolist())
        array_result = array_agg.to_result(tag="t", header="T", unit="ms")

        # Same field set on the Pydantic model.
        assert set(digest_result.model_fields_set) == set(
            array_result.model_fields_set
        )
        # Bit-exact stats.
        assert digest_result.count == array_result.count
        assert digest_result.min == pytest.approx(array_result.min)
        assert digest_result.max == pytest.approx(array_result.max)
        assert digest_result.sum == pytest.approx(array_result.sum, rel=1e-12)
        assert digest_result.avg == pytest.approx(array_result.avg, rel=1e-9)
        assert digest_result.std == pytest.approx(array_result.std, rel=1e-9)
        # Approximate percentiles within t-digest tolerance.
        for pct_field in ("p1", "p5", "p10", "p25", "p50", "p75", "p90", "p95", "p99"):
            assert getattr(digest_result, pct_field) == pytest.approx(
                getattr(array_result, pct_field), rel=0.005
            )
```

- [ ] **Step 2.2: Run tests to verify they fail with `ModuleNotFoundError`**

```bash
cd /home/anthony/nvidia/projects/aiperf/ajc/tdigest-icl-aggregator
uv run pytest tests/unit/metrics/test_list_metric_aggregation.py -v 2>&1 | head -30
```

Expected: collection error mentioning `from aiperf.metrics.list_metric_aggregation import TDigestListMetricAggregator` failing — module does not yet exist.

- [ ] **Step 2.3: Create the aggregator module**

Create `src/aiperf/metrics/list_metric_aggregation.py` with exactly this content:

```python
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run-level aggregator for list-valued record metrics.

Used by :class:`aiperf.post_processors.metric_results_processor.MetricResultsProcessor`
when a ``MetricType.RECORD`` metric arrives with a list value (today only
``inter_chunk_latency``, where each request contributes a list of inter-chunk
gap durations). At 1 M-request ramp scale the exact storage —
``records × (chunks-1) × 8 B`` — would dwarf the records-manager pod's
memory budget. T-digest bounds it to a few KB regardless of sample count.

Stats:
- ``count``, ``sum``, ``min``, ``max``, ``avg``, ``std`` are exact
  (running side-channel scalars).
- ``p1``..``p99`` are approximate via t-digest (~0.5% relative error).
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import tdigest as _tdigest

from aiperf.common.models import MetricResult
from aiperf.common.types import MetricTagT


class TDigestListMetricAggregator:
    """Bounded-memory aggregator backed by a t-digest sketch."""

    def __init__(self) -> None:
        self._td = _tdigest.TDigest()
        self._count: int = 0
        self._sum: float = 0.0
        self._sum_sq: float = 0.0
        self._min: float | None = None
        self._max: float | None = None

    def append(self, value: int | float) -> None:
        """Add a single sample."""
        v = float(value)
        self._td.update(v)
        self._count += 1
        self._sum += v
        self._sum_sq += v * v
        self._min = v if self._min is None else (v if v < self._min else self._min)
        self._max = v if self._max is None else (v if v > self._max else self._max)

    def extend(self, values: Iterable[int | float]) -> None:
        """Add many samples. Iterable is consumed once."""
        for v in values:
            self.append(v)

    def to_result(
        self, tag: MetricTagT, header: str, unit: str
    ) -> MetricResult:
        """Return a :class:`MetricResult` with the same field set as
        ``MetricArray.to_result``. Percentiles come from the t-digest;
        every other stat is exact."""
        if self._count == 0:
            return MetricResult(tag=tag, header=header, unit=unit, count=0)
        avg = self._sum / self._count
        # Population variance, matching numpy's default for np.std(arr).
        # max(0, ...) clamps tiny floating-point underflow when all samples
        # are equal (sum_sq/count == avg^2 to <1ulp).
        var = max(0.0, self._sum_sq / self._count - avg * avg)
        std = math.sqrt(var)
        return MetricResult(
            tag=tag,
            header=header,
            unit=unit,
            count=self._count,
            sum=self._sum,
            min=self._min,
            max=self._max,
            avg=avg,
            std=std,
            p1=self._td.percentile(1),
            p5=self._td.percentile(5),
            p10=self._td.percentile(10),
            p25=self._td.percentile(25),
            p50=self._td.percentile(50),
            p75=self._td.percentile(75),
            p90=self._td.percentile(90),
            p95=self._td.percentile(95),
            p99=self._td.percentile(99),
        )
```

- [ ] **Step 2.4: Run the new tests to verify they pass**

```bash
cd /home/anthony/nvidia/projects/aiperf/ajc/tdigest-icl-aggregator
uv run pytest tests/unit/metrics/test_list_metric_aggregation.py -v
```

Expected: all 10 tests pass. If percentile-tolerance test fails by a small margin, do NOT loosen the tolerance — investigate first (likely a bug). The 0.5% band is the documented t-digest accuracy at 100k samples and should comfortably hold.

- [ ] **Step 2.5: Commit**

```bash
cd /home/anthony/nvidia/projects/aiperf/ajc/tdigest-icl-aggregator
git add src/aiperf/metrics/list_metric_aggregation.py tests/unit/metrics/test_list_metric_aggregation.py
git commit -s -m "$(cat <<'EOF'
feat(metrics): TDigestListMetricAggregator for list-valued record metrics

Bounded-memory replacement for MetricArray when a record metric's value
is a list (today only inter_chunk_latency, which contributes one sample
per inter-chunk gap per request — billions of points at ramp scale and
~40 GB resident on the records-manager pod with exact storage).

Sketch-backed percentiles (~0.5% relative error); count/sum/min/max/avg/
std stay bit-exact via running side-channel scalars. Same to_result
contract as MetricArray, so the consuming processor can swap in with
one isinstance branch (next commit).
EOF
)"
```

Expected: pre-commit hooks pass; commit succeeds.

---

## Task 3: Wire `TDigestListMetricAggregator` into `MetricResultsProcessor` (TDD)

**Files:**
- Modify: `src/aiperf/post_processors/metric_results_processor.py:21,82-94,196-207`
- Modify: `tests/unit/post_processors/test_metric_results_processor.py` (one test method)

The aggregator from Task 2 is now plumbed in. The test retrofit comes first.

- [ ] **Step 3.1: Read the existing list-values test to understand the surface**

```bash
cd /home/anthony/nvidia/projects/aiperf/ajc/tdigest-icl-aggregator
sed -n '60,90p' tests/unit/post_processors/test_metric_results_processor.py
```

Expected: shows `test_process_result_record_metric_list_values` that asserts `isinstance(processor._results["test_record"], MetricArray)` and `list(processor._results["test_record"].data) == [10.0, 20.0, 30.0]`. We will replace the body of this test only.

- [ ] **Step 3.2: Retrofit the list-values test**

Use Edit to replace the body of `test_process_result_record_metric_list_values`. The exact `old_string` to replace:

```python
    @pytest.mark.asyncio
    async def test_process_result_record_metric_list_values(
        self, mock_metric_registry: Mock, mock_user_config: UserConfig
    ) -> None:
        """Test processing record metric with list values extends the array."""
        processor = MetricResultsProcessor(mock_user_config)
        processor._tags_to_types = {"test_record": MetricType.RECORD}

        # Process list of values
        message = create_metric_records_message(
            x_request_id="test-1",
            results=[{"test_record": [10.0, 20.0, 30.0]}],
        )
        await processor.process_result(message.to_data())

        assert "test_record" in processor._results
        assert isinstance(processor._results["test_record"], MetricArray)
        assert list(processor._results["test_record"].data) == [10.0, 20.0, 30.0]
```

The exact `new_string`:

```python
    @pytest.mark.asyncio
    async def test_process_result_record_metric_list_values(
        self, mock_metric_registry: Mock, mock_user_config: UserConfig
    ) -> None:
        """List-valued record metrics use the t-digest aggregator (not MetricArray).

        T-digest preserves count/sum/min/max exactly; percentiles are
        approximate but irrelevant to this test (3 samples).
        """
        from aiperf.metrics.list_metric_aggregation import (
            TDigestListMetricAggregator,
        )

        processor = MetricResultsProcessor(mock_user_config)
        processor._tags_to_types = {"test_record": MetricType.RECORD}

        message = create_metric_records_message(
            x_request_id="test-1",
            results=[{"test_record": [10.0, 20.0, 30.0]}],
        )
        await processor.process_result(message.to_data())

        assert "test_record" in processor._results
        assert isinstance(
            processor._results["test_record"], TDigestListMetricAggregator
        )
        # Stat-shape check (count/sum/min/max are bit-exact via side-channel).
        result = processor._results["test_record"].to_result(
            tag="test_record", header="Test Record", unit="ms"
        )
        assert result.count == 3
        assert result.sum == pytest.approx(60.0)
        assert result.min == pytest.approx(10.0)
        assert result.max == pytest.approx(30.0)
```

- [ ] **Step 3.3: Run the modified test to verify it fails**

```bash
cd /home/anthony/nvidia/projects/aiperf/ajc/tdigest-icl-aggregator
uv run pytest tests/unit/post_processors/test_metric_results_processor.py::TestMetricResultsProcessor::test_process_result_record_metric_list_values -v
```

Expected: FAIL because the processor still creates `MetricArray()` for list values (production code not yet updated). Failure message: `assert isinstance(<MetricArray ...>, TDigestListMetricAggregator)`.

- [ ] **Step 3.4: Update `MetricResultsProcessor` imports**

Use Edit on `src/aiperf/post_processors/metric_results_processor.py`. Replace `old_string`:

```python
from aiperf.metrics.metric_dicts import MetricArray, MetricResultsDict
from aiperf.metrics.metric_registry import MetricRegistry
```

With `new_string`:

```python
from aiperf.metrics.list_metric_aggregation import TDigestListMetricAggregator
from aiperf.metrics.metric_dicts import MetricArray, MetricResultsDict
from aiperf.metrics.metric_registry import MetricRegistry
```

- [ ] **Step 3.5: Update `process_result` first-touch logic**

Use Edit on `src/aiperf/post_processors/metric_results_processor.py`. Replace `old_string`:

```python
                if metric_type == MetricType.RECORD:
                    if tag not in results_dict:
                        results_dict[tag] = MetricArray()
                    if isinstance(value, list):
                        # NOTE: Right now we only support list-based metrics by extending the array.
                        #       In the future, we possibly could support having nested arrays.
                        results_dict[tag].extend(value)  # type: ignore
                    else:
                        results_dict[tag].append(value)  # type: ignore
```

With `new_string`:

```python
                if metric_type == MetricType.RECORD:
                    if tag not in results_dict:
                        # The metric class shape doesn't change mid-run, so the
                        # storage type can be picked at first-touch. List values
                        # go to the bounded t-digest aggregator (today only
                        # inter_chunk_latency would otherwise blow past pod RAM
                        # at ramp scale); scalar values stay in MetricArray.
                        results_dict[tag] = (
                            TDigestListMetricAggregator()
                            if isinstance(value, list)
                            else MetricArray()
                        )
                    if isinstance(value, list):
                        results_dict[tag].extend(value)  # type: ignore
                    else:
                        results_dict[tag].append(value)  # type: ignore
```

- [ ] **Step 3.6: Update `_create_metric_result` isinstance dispatch**

Use Edit on `src/aiperf/post_processors/metric_results_processor.py`. Replace `old_string`:

```python
        if isinstance(values, MetricArray):
            return values.to_result(tag, metric_class.header, str(metric_class.unit))

        if isinstance(values, int | float):
```

With `new_string`:

```python
        if isinstance(values, MetricArray | TDigestListMetricAggregator):
            return values.to_result(tag, metric_class.header, str(metric_class.unit))

        if isinstance(values, int | float):
```

- [ ] **Step 3.7: Run the targeted test to verify it passes**

```bash
cd /home/anthony/nvidia/projects/aiperf/ajc/tdigest-icl-aggregator
uv run pytest tests/unit/post_processors/test_metric_results_processor.py -v
```

Expected: all tests in this file pass, including `test_process_result_record_metric_list_values` (now asserts the t-digest aggregator) and `test_process_result_record_metric` (scalar path, MetricArray, untouched).

- [ ] **Step 3.8: Commit**

```bash
cd /home/anthony/nvidia/projects/aiperf/ajc/tdigest-icl-aggregator
git add src/aiperf/post_processors/metric_results_processor.py tests/unit/post_processors/test_metric_results_processor.py
git commit -s -m "$(cat <<'EOF'
feat(metrics): wire TDigestListMetricAggregator into MetricResultsProcessor

When a RECORD-type metric arrives with a list value (today only
inter_chunk_latency), pick the bounded t-digest aggregator at first-touch
instead of MetricArray. Scalar record metrics still use MetricArray —
unchanged.

ICL goes from O(records × chunks_per_record × 8 B) — ~40 GB on the
records-manager at 1 M req × 5 K chunks — to ~few KB constant.
Exporters (JsonMetricResult / MetricResult) consume the same to_result
contract; per-record JSONL paths are untouched.
EOF
)"
```

Expected: pre-commit hooks pass; commit succeeds.

---

## Task 4: Final verification (one shot, unit suite + ergonomics + ruff)

Per project policy: ONE pytest invocation against the full unit suite. No subfolder splits.

- [ ] **Step 4.1: Run ruff format and check**

```bash
cd /home/anthony/nvidia/projects/aiperf/ajc/tdigest-icl-aggregator
uv run ruff format . && uv run ruff check --fix .
```

Expected: no diff produced (or only whitespace), no remaining errors. If ruff modifies files, re-stage and amend the most recent commit only if changes are pure formatting; otherwise create a new fixup commit.

- [ ] **Step 4.2: Run ergonomics + ruff baseline guards**

```bash
cd /home/anthony/nvidia/projects/aiperf/ajc/tdigest-icl-aggregator
make check-ergonomics && make check-ruff-baselined
```

Expected: both pass with zero new violations. If a new violation appears in our two new files, **fix the code**, do NOT add an entry to the baseline.

- [ ] **Step 4.3: Run the full unit suite**

```bash
cd /home/anthony/nvidia/projects/aiperf/ajc/tdigest-icl-aggregator
set -o pipefail && PYTHONUNBUFFERED=1 uv run pytest tests/unit/ -n auto 2>&1 | tee /tmp/tdigest-final.log | tail -30
```

Expected: pass count ≥ baseline + 10 (we added 10 tests in `test_list_metric_aggregation.py`; we did not delete any). Baseline was `8551 passed, 8 skipped` on the fresh worktree, so target is `8561 passed, 8 skipped`. Allow ±1 from collection-time skips that depend on environment.

- [ ] **Step 4.4: Confirm spec acceptance criteria**

Manually verify each criterion from `docs/superpowers/specs/2026-04-29-tdigest-icl-aggregator-design.md` § Acceptance criteria:

1. `make check-ergonomics` passes — confirmed in 4.2.
2. `make check-ruff-baselined` passes — confirmed in 4.2.
3. `pre-commit run --all-files` — skip per project policy (committed via standard hooks already).
4. Pass count meets baseline+10 — confirmed in 4.3.
5. New module exposes only `TDigestListMetricAggregator` (no enum, no flag, no mode):

```bash
cd /home/anthony/nvidia/projects/aiperf/ajc/tdigest-icl-aggregator
grep -E "^(class |def |[A-Z_]+ ?=)" src/aiperf/metrics/list_metric_aggregation.py
```

Expected output: only `class TDigestListMetricAggregator:`.

6. `pyproject.toml` adds exactly one dependency line:

```bash
cd /home/anthony/nvidia/projects/aiperf/ajc/tdigest-icl-aggregator
git diff origin/main -- pyproject.toml | grep '^[-+][^+-]'
```

Expected: one `+` line for `tdigest~=0.5.2.2`. (Plus possibly auto-sorted neighboring lines from `uv add`; that's fine.)

7. Production-code diff under 200 LOC:

```bash
cd /home/anthony/nvidia/projects/aiperf/ajc/tdigest-icl-aggregator
git diff origin/main --stat -- 'src/' | tail -3
```

Expected: total `insertions(+)` for `src/` under 200.

- [ ] **Step 4.5: Push branch (optional — only if user asks)**

```bash
cd /home/anthony/nvidia/projects/aiperf/ajc/tdigest-icl-aggregator
git push -u origin ajc/tdigest-icl-aggregator
```

Do NOT push unless the user explicitly requests it.

---

## Self-Review

**Spec coverage:**

- Spec § In scope item 1 (new module) → Task 2.3.
- Spec § In scope item 2 (process_result first-touch) → Task 3.5.
- Spec § In scope item 3 (_create_metric_result isinstance) → Task 3.6.
- Spec § In scope item 4 (tdigest dep) → Task 1.1.
- Spec § In scope item 5 (test retrofit) → Task 3.2.
- Spec § In scope item 6 (new test file with all listed tests) → Task 2.1.
- Spec § Architecture / Aggregator contract (append/extend/to_result, no MetricArray inheritance) → Task 2.3.
- Spec § Implementation notes (5 side-channel scalars, exact mean/std formula, min/max from side-channel) → Task 2.3.
- Spec § Error handling (empty digest → count=0 None fields; empty extend no-op) → covered by `test_empty_aggregator_returns_count_zero_result`; no-op extend trivially handled by the empty-iterable for-loop.
- Spec § Testing all 10 named tests → Task 2.1 implements all 10.
- Spec § Acceptance criteria 1-7 → Task 4.

No gaps.

**Placeholder scan:** No "TBD", "TODO", or hand-wave language remains. Every code step shows the full code.

**Type consistency:** `TDigestListMetricAggregator` referenced identically in tests, module, processor import, processor first-touch, and `_create_metric_result` isinstance. Method names `append` / `extend` / `to_result` consistent throughout.

**One-shot test policy:** Only one full `pytest tests/unit/ -n auto` invocation, in Task 4.3. Per-task targeted test runs (2.4, 3.3, 3.7) are scoped to single files / single tests during TDD and do not violate the policy.

---

## Execution

**Recommended:** subagent-driven-development (per memory `feedback_always_subagent_driven_execution`). Each task above is one subagent dispatch with `model="opus"` (per memory `feedback_always_opus_4_7`). Tasks are sequential — Task 2 depends on Task 1's installed dep; Task 3 depends on Task 2's class; Task 4 depends on Tasks 1–3.

Do NOT execute inline.
