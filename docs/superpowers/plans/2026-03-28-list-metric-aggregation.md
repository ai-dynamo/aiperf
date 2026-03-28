# List Metric Aggregation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a first-class `metrics.listMetricAggregation` config option that switches list-valued record-metric summary aggregation between exact storage and t-digest while keeping summary exports unchanged.

**Architecture:** Introduce a small user-facing `MetricsConfig` section with an enum-backed `list_metric_aggregation` field, then route list-valued record metrics through a dedicated summary accumulator abstraction. Scalar metrics, per-record exports, and summary schemas stay unchanged; only the internal summary path for list metrics changes.

**Tech Stack:** Pydantic v2 config models, NumPy, existing `MetricResultsProcessor` / `TimesliceMetricResultsProcessor`, Python `tdigest` library, pytest

---

## File map

- Create: `src/aiperf/config/metrics.py` — user-facing metrics config section
- Create: `src/aiperf/metrics/list_metric_aggregation.py` — exact and t-digest list-metric summary accumulators
- Create: `tests/unit/config/test_metrics_config.py` — config parsing coverage for `metrics.listMetricAggregation`
- Create: `tests/unit/metrics/test_list_metric_aggregation.py` — unit tests for exact/t-digest accumulator behavior
- Modify: `src/aiperf/common/enums/enums.py` — add `ListMetricAggregationMode`
- Modify: `src/aiperf/common/enums/__init__.py` — export `ListMetricAggregationMode`
- Modify: `src/aiperf/common/enums/metric_enums.py` — widen `MetricDictValueTypeT` to include the new accumulator type
- Modify: `src/aiperf/config/config.py` — add `metrics: MetricsConfig` to `BenchmarkConfig`
- Modify: `src/aiperf/config/__init__.py` — export `MetricsConfig`
- Modify: `src/aiperf/metrics/metric_dicts.py` — update `MetricResultsDict` typing/docs if it stores the new accumulator type
- Modify: `src/aiperf/post_processors/metric_results_processor.py` — instantiate the configured list-metric accumulator instead of always using `MetricArray.extend(...)`
- Modify: `tests/unit/post_processors/test_metric_results_processor.py` — processor coverage for exact + tdigest behavior and unchanged summary shape
- Modify: `tests/unit/post_processors/test_record_export_results_processor.py` — prove `artifacts.per_chunk_data` still solely controls per-record JSONL list exports
- Modify: `tests/unit/post_processors/test_record_export_csv_processor.py` — prove `artifacts.per_chunk_data` still solely controls per-record CSV list exports
- Modify: `pyproject.toml` — add required `tdigest` dependency
- Modify: `docs/metrics-reference.md` — document exact vs tdigest summary behavior for list-valued metrics
- Modify: `docs/tutorials/working-with-profile-exports.md` — clarify that `per_chunk_data` controls raw list export, while `metrics.listMetricAggregation` controls summary aggregation only

### Task 1: Add the user-facing config surface

**Files:**
- Create: `src/aiperf/config/metrics.py`
- Modify: `src/aiperf/common/enums/enums.py`
- Modify: `src/aiperf/common/enums/__init__.py`
- Modify: `src/aiperf/config/config.py:43-67,262-336`
- Modify: `src/aiperf/config/__init__.py:38-59,108-121,184-280`
- Test: `tests/unit/config/test_metrics_config.py`

- [ ] **Step 1: Write the failing config tests**

```python
from __future__ import annotations

import pytest
from pydantic import ValidationError

from aiperf.common.enums import ListMetricAggregationMode
from aiperf.config.config import AIPerfConfig


def _base_config() -> dict:
    return {
        "models": ["test-model"],
        "endpoint": {"urls": ["http://localhost:8000/v1/chat/completions"]},
        "datasets": {
            "default": {
                "type": "synthetic",
                "entries": 1,
                "prompts": {"isl": 128, "osl": 64},
            }
        },
        "phases": {
            "profiling": {
                "type": "concurrency",
                "concurrency": 1,
                "requests": 1,
            }
        },
    }


def test_metrics_config_defaults_to_exact() -> None:
    config = AIPerfConfig(**_base_config())
    assert config.metrics.list_metric_aggregation is ListMetricAggregationMode.EXACT


def test_metrics_config_accepts_tdigest_alias() -> None:
    config = AIPerfConfig(
        **{
            **_base_config(),
            "metrics": {"listMetricAggregation": "tdigest"},
        }
    )
    assert config.metrics.list_metric_aggregation is ListMetricAggregationMode.TDIGEST


def test_metrics_config_rejects_invalid_mode() -> None:
    with pytest.raises(ValidationError, match="listMetricAggregation"):
        AIPerfConfig(
            **{
                **_base_config(),
                "metrics": {"listMetricAggregation": "approx"},
            }
        )
```

- [ ] **Step 2: Run the config tests to verify they fail**

Run: `uv run pytest tests/unit/config/test_metrics_config.py -v`
Expected: FAIL with import or validation errors because `MetricsConfig` and `ListMetricAggregationMode` do not exist yet.

- [ ] **Step 3: Add the enum and config model**

```python
# src/aiperf/common/enums/enums.py
class ListMetricAggregationMode(CaseInsensitiveStrEnum):
    """How list-valued record metrics are aggregated into summary statistics."""

    EXACT = "exact"
    TDIGEST = "tdigest"
```

```python
# src/aiperf/config/metrics.py
from __future__ import annotations

from typing import Annotated

from pydantic import ConfigDict, Field

from aiperf.common.enums import ListMetricAggregationMode
from aiperf.config._base import BaseConfig


class MetricsConfig(BaseConfig):
    """Metrics summary configuration."""

    model_config = ConfigDict(extra="forbid", validate_default=True)

    list_metric_aggregation: Annotated[
        ListMetricAggregationMode,
        Field(
            default=ListMetricAggregationMode.EXACT,
            description="How list-valued record metrics are aggregated into summary "
            "statistics. 'exact' retains all samples; 'tdigest' uses approximate "
            "percentiles while preserving the existing MetricResult output shape.",
        ),
    ]
```

```python
# src/aiperf/config/config.py
from aiperf.config.metrics import MetricsConfig

metrics: Annotated[
    MetricsConfig,
    Field(
        default_factory=MetricsConfig,
        description="Metrics summary configuration. Controls how list-valued "
        "record metrics are aggregated into summary statistics.",
    ),
]
```

- [ ] **Step 4: Export the new symbols from `aiperf.config` and `aiperf.common.enums`**

```python
# src/aiperf/common/enums/__init__.py
from aiperf.common.enums.enums import ListMetricAggregationMode

# Add ListMetricAggregationMode to the existing import block from
# aiperf.common.enums.enums and append "ListMetricAggregationMode" to the
# existing __all__ list. Do not replace the module's current exports.
```

```python
# src/aiperf/config/__init__.py
from aiperf.config.metrics import MetricsConfig

# Add the MetricsConfig import alongside the other config imports and append
# "MetricsConfig" to the existing __all__ list. Do not replace the module's
# current exports.
```

- [ ] **Step 5: Run the config tests to verify they pass**

Run: `uv run pytest tests/unit/config/test_metrics_config.py -v`
Expected: PASS

- [ ] **Step 6: Commit the config surface**

```bash
git add src/aiperf/common/enums/enums.py src/aiperf/common/enums/__init__.py src/aiperf/config/metrics.py src/aiperf/config/config.py src/aiperf/config/__init__.py tests/unit/config/test_metrics_config.py
git commit -m "feat: add list metric aggregation config"
```

### Task 2: Add exact and t-digest list-metric summary accumulators

**Files:**
- Create: `src/aiperf/metrics/list_metric_aggregation.py`
- Modify: `src/aiperf/common/enums/metric_enums.py:19-26`
- Modify: `src/aiperf/metrics/metric_dicts.py:121-214`
- Modify: `pyproject.toml:26-72`
- Test: `tests/unit/metrics/test_list_metric_aggregation.py`

- [ ] **Step 1: Write the failing accumulator tests**

```python
from __future__ import annotations

import pytest

from aiperf.common.enums import ListMetricAggregationMode
from aiperf.metrics.list_metric_aggregation import build_list_metric_aggregator


@pytest.mark.parametrize(
    "mode",
    [
        ListMetricAggregationMode.EXACT,
        ListMetricAggregationMode.TDIGEST,
    ],
)
def test_list_metric_aggregator_preserves_summary_shape(mode) -> None:
    aggregator = build_list_metric_aggregator(mode)
    aggregator.add_samples([1.0, 2.0, 3.0, 4.0, 5.0])

    result = aggregator.to_result(
        tag="inter_chunk_latency",
        header="Inter Chunk Latency",
        unit="ns",
    )

    assert result.count == 5
    assert result.min == 1.0
    assert result.max == 5.0
    assert result.avg == pytest.approx(3.0)
    assert result.std is not None
    assert result.p50 is not None
    assert result.p95 is not None
    assert result.p99 is not None


def test_tdigest_list_metric_aggregator_stays_close_to_exact() -> None:
    exact = build_list_metric_aggregator(ListMetricAggregationMode.EXACT)
    tdigest = build_list_metric_aggregator(ListMetricAggregationMode.TDIGEST)
    samples = [float(i) for i in range(1, 2001)]

    exact.add_samples(samples)
    tdigest.add_samples(samples)

    exact_result = exact.to_result("inter_chunk_latency", "Inter Chunk Latency", "ns")
    tdigest_result = tdigest.to_result(
        "inter_chunk_latency", "Inter Chunk Latency", "ns"
    )

    assert tdigest_result.p50 == pytest.approx(exact_result.p50, abs=5.0)
    assert tdigest_result.p95 == pytest.approx(exact_result.p95, abs=10.0)
    assert tdigest_result.p99 == pytest.approx(exact_result.p99, abs=10.0)
```

- [ ] **Step 2: Run the accumulator tests to verify they fail**

Run: `uv run pytest tests/unit/metrics/test_list_metric_aggregation.py -v`
Expected: FAIL because the accumulator module and dependency do not exist yet.

- [ ] **Step 3: Add the required dependency and implement the accumulators**

Run: `uv add tdigest`

```python
# src/aiperf/metrics/list_metric_aggregation.py
from __future__ import annotations

from abc import ABC, abstractmethod
from math import sqrt

from tdigest import TDigest

from aiperf.common.enums import ListMetricAggregationMode
from aiperf.common.models.record_models import MetricResult
from aiperf.metrics.metric_dicts import MetricArray


class ListMetricAggregator(ABC):
    @abstractmethod
    def add_samples(self, values: list[int] | list[float]) -> None:
        """Add a batch of list-valued metric samples."""

    @abstractmethod
    def to_result(self, tag: str, header: str, unit: str) -> MetricResult:
        """Return the standard summary MetricResult for the accumulated samples."""


class ExactListMetricAggregator(ListMetricAggregator):
    def __init__(self) -> None:
        self._values = MetricArray()

    def add_samples(self, values: list[int] | list[float]) -> None:
        self._values.extend(values)

    def to_result(self, tag: str, header: str, unit: str) -> MetricResult:
        return self._values.to_result(tag, header, unit)


class TDigestListMetricAggregator(ListMetricAggregator):
    def __init__(self) -> None:
        self._digest = TDigest()
        self._count = 0
        self._sum = 0.0
        self._sum_squares = 0.0
        self._min: float | None = None
        self._max: float | None = None

    def add_samples(self, values: list[int] | list[float]) -> None:
        for value in values:
            numeric = float(value)
            self._digest.update(numeric)
            self._count += 1
            self._sum += numeric
            self._sum_squares += numeric * numeric
            self._min = numeric if self._min is None else min(self._min, numeric)
            self._max = numeric if self._max is None else max(self._max, numeric)

    def to_result(self, tag: str, header: str, unit: str) -> MetricResult:
        avg = self._sum / self._count
        variance = max((self._sum_squares / self._count) - (avg * avg), 0.0)
        return MetricResult(
            tag=tag,
            header=header,
            unit=unit,
            min=self._min,
            max=self._max,
            avg=avg,
            sum=self._sum,
            std=sqrt(variance),
            p1=self._digest.percentile(1),
            p5=self._digest.percentile(5),
            p10=self._digest.percentile(10),
            p25=self._digest.percentile(25),
            p50=self._digest.percentile(50),
            p75=self._digest.percentile(75),
            p90=self._digest.percentile(90),
            p95=self._digest.percentile(95),
            p99=self._digest.percentile(99),
            count=self._count,
        )


def build_list_metric_aggregator(
    mode: ListMetricAggregationMode,
) -> ListMetricAggregator:
    if mode is ListMetricAggregationMode.TDIGEST:
        return TDigestListMetricAggregator()
    return ExactListMetricAggregator()
```

The important constraint is that `TDigestListMetricAggregator` must explicitly track
`count`, `sum`, `sum_squares`, `min`, and `max` alongside the digest. Do not claim those
statistics come from t-digest alone.

- [ ] **Step 4: Widen the run-level metric typing to allow the new accumulator**

```python
# src/aiperf/common/enums/metric_enums.py
if TYPE_CHECKING:
    from aiperf.metrics.list_metric_aggregation import ListMetricAggregator
    from aiperf.metrics.metric_dicts import MetricArray

MetricDictValueTypeT: TypeAlias = (
    "MetricValueTypeT | list[MetricValueTypeT] | MetricArray | ListMetricAggregator"
)
```

```python
# src/aiperf/metrics/metric_dicts.py
class MetricResultsDict(BaseMetricDict[MetricDictValueTypeT]):
    """
    A dict of metrics over an entire run.

    This will include:
    - Scalar record metrics as a MetricArray of their values
    - List-valued record metrics as a list-metric summary accumulator
    - The most recent value of each BaseAggregateMetric
    - The value of any BaseDerivedMetric that has already been computed
    """
```

- [ ] **Step 5: Run the accumulator tests to verify they pass**

Run: `uv run pytest tests/unit/metrics/test_list_metric_aggregation.py -v`
Expected: PASS

- [ ] **Step 6: Commit the accumulator layer**

```bash
git add pyproject.toml uv.lock src/aiperf/metrics/list_metric_aggregation.py src/aiperf/common/enums/metric_enums.py src/aiperf/metrics/metric_dicts.py tests/unit/metrics/test_list_metric_aggregation.py
git commit -m "feat: add list metric summary accumulators"
```

### Task 3: Wire the processor to the configured list-metric mode

**Files:**
- Modify: `src/aiperf/post_processors/metric_results_processor.py:37-212`
- Modify: `tests/unit/post_processors/test_metric_results_processor.py`

- [ ] **Step 1: Write the failing processor tests**

```python
@pytest.mark.asyncio
async def test_process_result_record_metric_list_values_uses_exact_mode_by_default(
    mock_metric_registry: Mock, mock_user_config: AIPerfConfig
) -> None:
    processor = MetricResultsProcessor(_make_run(mock_user_config))
    processor._tags_to_types = {"test_record": MetricType.RECORD}

    message = create_metric_records_message(
        x_request_id="test-1",
        results=[{"test_record": [10.0, 20.0, 30.0]}],
    )
    await processor.process_result(message.to_data())

    result = processor._create_metric_result("test_record", processor._results["test_record"])
    assert result.count == 3
    assert result.p50 == pytest.approx(20.0)


@pytest.mark.asyncio
async def test_process_result_record_metric_list_values_uses_tdigest_mode(
    mock_metric_registry: Mock, mock_user_config: AIPerfConfig
) -> None:
    mock_user_config.metrics.list_metric_aggregation = ListMetricAggregationMode.TDIGEST
    processor = MetricResultsProcessor(_make_run(mock_user_config))
    processor._tags_to_types = {"test_record": MetricType.RECORD}

    message = create_metric_records_message(
        x_request_id="test-1",
        results=[{"test_record": [10.0, 20.0, 30.0, 40.0]}],
    )
    await processor.process_result(message.to_data())

    result = processor._create_metric_result("test_record", processor._results["test_record"])
    assert result.count == 4
    assert result.p50 == pytest.approx(25.0, abs=5.0)
    assert result.avg == pytest.approx(25.0)


def test_create_metric_result_accepts_list_metric_aggregator(
    mock_metric_registry: Mock, mock_user_config: AIPerfConfig
) -> None:
    processor = MetricResultsProcessor(_make_run(mock_user_config))
    processor._instances_map = {RequestLatencyMetric.tag: RequestLatencyMetric()}
    aggregator = build_list_metric_aggregator(ListMetricAggregationMode.EXACT)
    aggregator.add_samples([10.0, 20.0, 30.0])

    result = processor._create_metric_result(RequestLatencyMetric.tag, aggregator)

    assert result.count == 3
    assert result.avg == pytest.approx(20.0)
```

- [ ] **Step 2: Run the processor tests to verify they fail**

Run: `uv run pytest tests/unit/post_processors/test_metric_results_processor.py -v`
Expected: FAIL because the processor still assumes all record metrics use `MetricArray`.

- [ ] **Step 3: Cache the selected mode and branch only for list-valued record metrics**

```python
# src/aiperf/post_processors/metric_results_processor.py
from aiperf.metrics.list_metric_aggregation import (
    ListMetricAggregator,
    build_list_metric_aggregator,
)
```

```python
# inside MetricResultsProcessor.__init__
self._list_metric_aggregation_mode = run.cfg.metrics.list_metric_aggregation
```

```python
# inside MetricResultsProcessor.process_result
if metric_type == MetricType.RECORD:
    if isinstance(value, list):
        current_values = results_dict.get(tag)
        if current_values is None:
            aggregator = build_list_metric_aggregator(
                self._list_metric_aggregation_mode
            )
            results_dict[tag] = aggregator
        else:
            aggregator = current_values
        assert isinstance(aggregator, ListMetricAggregator)
        aggregator.add_samples(value)
    else:
        if tag not in results_dict:
            results_dict[tag] = MetricArray()
        results_dict[tag].append(value)
```

- [ ] **Step 4: Teach `_create_metric_result(...)` to emit the existing summary shape from either storage mode**

```python
def _create_metric_result(
    self, tag: MetricTagT, values: MetricDictValueTypeT
) -> MetricResult:
    metric_class = self._instances_map[tag]

    if isinstance(values, MetricArray):
        return values.to_result(tag, metric_class.header, str(metric_class.unit))

    if isinstance(values, ListMetricAggregator):
        return values.to_result(tag, metric_class.header, str(metric_class.unit))

    if isinstance(values, int | float):
        return MetricResult(
            tag=metric_class.tag,
            header=metric_class.header,
            unit=str(metric_class.unit),
            avg=values,
            count=1,
        )

    raise ValueError(f"Unexpected values type: {type(values)}")
```

- [ ] **Step 5: Run the processor tests to verify they pass**

Run: `uv run pytest tests/unit/post_processors/test_metric_results_processor.py -v`
Expected: PASS

- [ ] **Step 6: Commit the processor wiring**

```bash
git add src/aiperf/post_processors/metric_results_processor.py tests/unit/post_processors/test_metric_results_processor.py
git commit -m "feat: wire list metric aggregation mode into summaries"
```

### Task 4: Lock down export behavior and docs

**Files:**
- Modify: `tests/unit/post_processors/test_record_export_results_processor.py`
- Modify: `tests/unit/post_processors/test_record_export_csv_processor.py`
- Modify: `docs/metrics-reference.md`
- Modify: `docs/tutorials/working-with-profile-exports.md`

- [ ] **Step 1: Write the failing regression tests for per-record exports**

```python
@pytest.mark.asyncio
async def test_jsonl_export_filters_list_metrics_when_per_chunk_data_disabled(
    tmp_artifact_dir: Path,
    mock_metric_registry: Mock,
) -> None:
    config = AIPerfConfig(
        models=["test-model"],
        endpoint={"urls": ["http://localhost:8000/v1/chat/completions"], "type": "chat"},
        datasets={
            "default": {
                "type": "synthetic",
                "entries": 1,
                "prompts": {"isl": 128, "osl": 64},
            }
        },
        phases={"default": {"type": "concurrency", "concurrency": 1, "requests": 1}},
        metrics={"listMetricAggregation": "tdigest"},
        artifacts={"dir": str(tmp_artifact_dir), "records": ["jsonl"]},
    )
    message = create_metric_records_message(results=[{"inter_chunk_latency": [1_000_000, 2_000_000]}])
    processor = RecordExportResultsProcessor(service_id="records-manager", run=_make_run(config))

    async with aiperf_lifecycle(processor):
        with patch.object(
            MetricRecordDict,
            "to_display_dict",
            return_value={
                "inter_chunk_latency": MetricValue(value=[1.0, 2.0], unit="ms"),
            },
        ):
            await processor.process_result(message.to_data())

    exported_record = orjson.loads(processor.output_file.read_bytes().splitlines()[0])
    assert "inter_chunk_latency" not in exported_record["metrics"]


@pytest.mark.asyncio
async def test_csv_export_filters_list_metrics_when_per_chunk_data_disabled(
    tmp_artifact_dir: Path,
    mock_metric_registry: Mock,
) -> None:
    config = _make_csv_config(
        tmp_artifact_dir,
        metrics={"listMetricAggregation": "tdigest"},
        records=["csv"],
    )
    message = create_metric_records_message(results=[{"inter_chunk_latency": [1_000_000, 2_000_000]}])
    processor = RecordExportCSVProcessor(service_id="records-manager", run=_make_run(config))

    async with aiperf_lifecycle(processor):
        with patch.object(
            MetricRecordDict,
            "to_display_dict",
            return_value={
                "inter_chunk_latency": MetricValue(value=[1.0, 2.0], unit="ms"),
            },
        ):
            await processor.process_result(message.to_data())

    row = _parse_csv_output(processor.output_file)[0]
    assert "inter_chunk_latency (ms)" not in row
```

- [ ] **Step 2: Run the export regression tests to verify their current baseline**

Run: `uv run pytest tests/unit/post_processors/test_record_export_results_processor.py tests/unit/post_processors/test_record_export_csv_processor.py -v`
Expected: PASS after minor test fixture updates, proving the new config option does not change export gating.

- [ ] **Step 3: Update the metric docs and profile-export tutorial**

```md
<!-- docs/metrics-reference.md -->
### Inter Chunk Latency

`inter_chunk_latency` is a list-valued record metric. Summary exports always emit the
same percentile fields (`p1`..`p99`, `min`, `max`, `avg`, `std`) regardless of the
configured aggregation mode.

- `metrics.listMetricAggregation: exact` retains all list samples for exact percentiles
- `metrics.listMetricAggregation: tdigest` uses t-digest for approximate percentiles
  while keeping exact `count`, `min`, `max`, `avg`, and `std`
```

```md
<!-- docs/tutorials/working-with-profile-exports.md -->
`artifacts.perChunkData` controls whether raw list-valued metrics such as
`inter_chunk_latency` are included in per-request JSONL/CSV exports.

`metrics.listMetricAggregation` is separate: it controls how list-valued metrics are
aggregated into run-level summary percentiles and does not change per-record export
inclusion.
```

- [ ] **Step 4: Run docs-adjacent and export regression tests**

Run: `uv run pytest tests/unit/post_processors/test_record_export_results_processor.py tests/unit/post_processors/test_record_export_csv_processor.py -v`
Expected: PASS

- [ ] **Step 5: Commit the regression coverage and docs**

```bash
git add tests/unit/post_processors/test_record_export_results_processor.py tests/unit/post_processors/test_record_export_csv_processor.py docs/metrics-reference.md docs/tutorials/working-with-profile-exports.md
git commit -m "docs: describe list metric aggregation behavior"
```

### Task 5: End-to-end verification and cleanup

**Files:**
- Modify: any files touched in Tasks 1-4
- Test: targeted files from Tasks 1-4

- [ ] **Step 1: Run the focused unit suite**

Run: `uv run pytest tests/unit/config/test_metrics_config.py tests/unit/metrics/test_list_metric_aggregation.py tests/unit/post_processors/test_metric_results_processor.py tests/unit/post_processors/test_record_export_results_processor.py tests/unit/post_processors/test_record_export_csv_processor.py -v`
Expected: PASS

- [ ] **Step 2: Run formatting and lint fixes**

Run: `ruff format . && ruff check --fix .`
Expected: PASS with no remaining formatting or lint errors in touched files

- [ ] **Step 3: Run pre-commit on all changed files**

Run: `pre-commit run --files src/aiperf/common/enums/enums.py src/aiperf/common/enums/__init__.py src/aiperf/common/enums/metric_enums.py src/aiperf/config/metrics.py src/aiperf/config/config.py src/aiperf/config/__init__.py src/aiperf/metrics/list_metric_aggregation.py src/aiperf/metrics/metric_dicts.py src/aiperf/post_processors/metric_results_processor.py tests/unit/config/test_metrics_config.py tests/unit/metrics/test_list_metric_aggregation.py tests/unit/post_processors/test_metric_results_processor.py tests/unit/post_processors/test_record_export_results_processor.py tests/unit/post_processors/test_record_export_csv_processor.py docs/metrics-reference.md docs/tutorials/working-with-profile-exports.md`
Expected: PASS

- [ ] **Step 4: Inspect the final diff before handing off**

Run: `git diff -- src/aiperf/common/enums/enums.py src/aiperf/common/enums/__init__.py src/aiperf/common/enums/metric_enums.py src/aiperf/config/metrics.py src/aiperf/config/config.py src/aiperf/config/__init__.py src/aiperf/metrics/list_metric_aggregation.py src/aiperf/metrics/metric_dicts.py src/aiperf/post_processors/metric_results_processor.py tests/unit/config/test_metrics_config.py tests/unit/metrics/test_list_metric_aggregation.py tests/unit/post_processors/test_metric_results_processor.py tests/unit/post_processors/test_record_export_results_processor.py tests/unit/post_processors/test_record_export_csv_processor.py docs/metrics-reference.md docs/tutorials/working-with-profile-exports.md`
Expected: Only the planned list-metric aggregation, dependency, test, and docs changes appear

- [ ] **Step 5: Create the final implementation commit**

```bash
git add src/aiperf/common/enums/enums.py src/aiperf/common/enums/__init__.py src/aiperf/common/enums/metric_enums.py src/aiperf/config/metrics.py src/aiperf/config/config.py src/aiperf/config/__init__.py src/aiperf/metrics/list_metric_aggregation.py src/aiperf/metrics/metric_dicts.py src/aiperf/post_processors/metric_results_processor.py tests/unit/config/test_metrics_config.py tests/unit/metrics/test_list_metric_aggregation.py tests/unit/post_processors/test_metric_results_processor.py tests/unit/post_processors/test_record_export_results_processor.py tests/unit/post_processors/test_record_export_csv_processor.py docs/metrics-reference.md docs/tutorials/working-with-profile-exports.md pyproject.toml uv.lock
git commit -m "feat: add configurable list metric aggregation"
```
