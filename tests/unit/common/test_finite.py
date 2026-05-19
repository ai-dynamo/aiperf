# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the centralized NaN/inf discipline module."""

from __future__ import annotations

import math

import numpy as np
import pytest
from pydantic import BaseModel, ValidationError
from pytest import param

from aiperf.common.finite import (
    FiniteFloat,
    is_finite_value,
    nan_safe_mean,
    nan_safe_std,
    scrub_non_finite,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        param(0.0, True, id="zero_float"),
        param(1.5, True, id="positive_float"),
        param(-3.7, True, id="negative_float"),
        param(0, True, id="zero_int"),
        param(42, True, id="positive_int"),
        param(float("nan"), False, id="python_nan"),
        param(float("inf"), False, id="python_pos_inf"),
        param(float("-inf"), False, id="python_neg_inf"),
        param(None, False, id="none"),
        param(True, False, id="bool_true_rejected"),
        param(False, False, id="bool_false_rejected"),
        param("abc", False, id="string_non_numeric"),
        param("1.5", True, id="string_numeric_coerces"),
        param([1.0, 2.0], False, id="list"),
        param({"a": 1.0}, False, id="dict"),
        param(np.float32(np.nan), False, id="numpy_float32_nan"),
        param(np.float64(np.inf), False, id="numpy_float64_inf"),
        param(np.float64(-np.inf), False, id="numpy_float64_neg_inf"),
        param(np.float64(1.5), True, id="numpy_float64_finite"),
        param(np.int64(0), True, id="numpy_int64_zero"),
        param(np.int64(-100), True, id="numpy_int64_negative"),
    ],
)
def test_is_finite_value(value: object, expected: bool) -> None:
    """is_finite_value must recognize finite Python and numpy scalars."""
    assert is_finite_value(value) is expected


class _M(BaseModel):
    """Tiny fixture model for FiniteFloat validation tests."""

    val: FiniteFloat
    opt: FiniteFloat | None = None


@pytest.mark.parametrize(
    "good_value",
    [
        param(0.0, id="zero"),
        param(1.0, id="positive"),
        param(-2.5, id="negative"),
        param(1e-300, id="tiny_positive"),
    ],
)
def test_FiniteFloat_pydantic_field_accepts_finite(good_value: float) -> None:
    """FiniteFloat must accept finite values without raising."""
    m = _M(val=good_value)
    assert m.val == good_value


@pytest.mark.parametrize(
    "bad_value",
    [
        param(float("nan"), id="nan"),
        param(float("inf"), id="pos_inf"),
        param(float("-inf"), id="neg_inf"),
    ],
)
def test_FiniteFloat_pydantic_field_rejects_non_finite(bad_value: float) -> None:
    """FiniteFloat must reject NaN/inf with a value-bearing error message."""
    with pytest.raises(ValidationError) as excinfo:
        _M(val=bad_value)
    msg = str(excinfo.value)
    assert "must be finite" in msg
    # message must include the rejected value so debug isn't a guessing game
    assert repr(bad_value) in msg or str(bad_value) in msg


def test_FiniteFloat_optional_none_passes_through() -> None:
    """`FiniteFloat | None` must accept None (validator only runs on values)."""
    m = _M(val=1.0, opt=None)
    assert m.opt is None


def test_FiniteFloat_optional_rejects_nan() -> None:
    """`FiniteFloat | None` still rejects NaN when value is provided."""
    with pytest.raises(ValidationError):
        _M(val=1.0, opt=float("nan"))


def test_scrub_non_finite_dict_list_nested() -> None:
    """Non-finite values in nested dict/list/tuple slots become None; structure preserved."""
    payload = {
        "metrics": {
            "ttft": {"mean": float("nan"), "std": 0.5, "min": float("-inf")},
            "rps": {"mean": 100.0, "max": float("inf")},
        },
        "history": [
            {"obj": float("nan"), "feasible": True},
            {"obj": 1.5, "feasible": False},
        ],
        "tuple_section": (1.0, float("inf"), "leave-me"),
        "string_field": "ok",
    }
    out = scrub_non_finite(payload)
    assert out["metrics"]["ttft"]["mean"] is None
    assert out["metrics"]["ttft"]["std"] == 0.5
    assert out["metrics"]["ttft"]["min"] is None
    assert out["metrics"]["rps"]["mean"] == 100.0
    assert out["metrics"]["rps"]["max"] is None
    assert out["history"][0]["obj"] is None
    assert out["history"][0]["feasible"] is True
    assert out["history"][1]["obj"] == 1.5
    # tuple stays tuple
    assert isinstance(out["tuple_section"], tuple)
    assert out["tuple_section"] == (1.0, None, "leave-me")
    assert out["string_field"] == "ok"


def test_scrub_non_finite_numpy_scalars() -> None:
    """numpy.float32(nan) and numpy.float64(inf) must be coerced to None."""
    payload = {
        "f32_nan": np.float32(np.nan),
        "f64_inf": np.float64(np.inf),
        "f64_finite": np.float64(3.14),
        "i64": np.int64(7),
    }
    out = scrub_non_finite(payload)
    assert out["f32_nan"] is None
    assert out["f64_inf"] is None
    # finite numpy float coerced through the same path -> python float, finite
    assert out["f64_finite"] == pytest.approx(3.14)
    # int passes through unchanged
    assert out["i64"] == np.int64(7)


def test_scrub_non_finite_does_not_recurse_into_str_bytes() -> None:
    """A literal 'nan' string is not a numeric NaN and must survive untouched."""
    payload = {
        "label": "nan",
        "raw": b"inf",
        "bytearr": bytearray(b"x"),
    }
    out = scrub_non_finite(payload)
    assert out["label"] == "nan"
    assert out["raw"] == b"inf"
    assert out["bytearr"] == bytearray(b"x")


def test_scrub_non_finite_preserves_bools() -> None:
    """Booleans are not metric values; they pass through unchanged."""
    out = scrub_non_finite({"flag": True, "other": False})
    assert out["flag"] is True
    assert out["other"] is False


def test_scrub_non_finite_orjson_roundtrip_distinguishes_null_and_nan() -> None:
    """End-to-end: scrub + orjson must produce real null for NaN, drop the NaN string entirely."""
    import orjson

    payload = {"observed": float("nan"), "expected": None, "good": 1.0}
    raw = orjson.dumps(scrub_non_finite(payload)).decode()
    parsed = orjson.loads(raw)
    assert parsed["observed"] is None
    assert parsed["expected"] is None
    assert parsed["good"] == 1.0


def test_nan_safe_mean_empty_returns_none() -> None:
    """Empty input -> None (not NaN, not 0)."""
    assert nan_safe_mean([]) is None


def test_nan_safe_mean_all_nan_returns_none() -> None:
    """All non-finite -> None."""
    assert nan_safe_mean([float("nan"), float("inf"), float("-inf"), None]) is None


def test_nan_safe_mean_mixed_returns_finite_only() -> None:
    """NaN entries are filtered before averaging."""
    result = nan_safe_mean([1.0, 2.0, float("nan"), 3.0])
    assert result == pytest.approx(2.0)


def test_nan_safe_mean_handles_numpy_inputs() -> None:
    """numpy arrays / scalars are filtered consistently with Python floats."""
    values = [np.float64(1.0), np.float64(np.nan), np.float64(3.0)]
    assert nan_safe_mean(values) == pytest.approx(2.0)


def test_nan_safe_std_empty_returns_none() -> None:
    """Empty input -> None."""
    assert nan_safe_std([]) is None


def test_nan_safe_std_too_few_finite_returns_none() -> None:
    """Fewer than (1 + ddof) finite values -> None."""
    assert nan_safe_std([1.0]) is None  # ddof=1 default needs >=2
    assert nan_safe_std([float("nan"), 2.0]) is None


def test_nan_safe_std_mixed_returns_finite_only() -> None:
    """NaN entries are filtered before stddev."""
    # finite values: [1, 2, 3, 4, 5]; sample stddev = sqrt(2.5)
    result = nan_safe_std([1.0, 2.0, float("nan"), 3.0, 4.0, 5.0, float("inf")])
    assert result == pytest.approx(math.sqrt(2.5))


def test_nan_safe_std_ddof_zero_population() -> None:
    """ddof=0 only requires 1 finite value (population stddev)."""
    assert nan_safe_std([1.0], ddof=0) == pytest.approx(0.0)


# --- Exporter regression tests ---------------------------------------------


def _make_aggregate_with_combos(combos):
    """Build an AggregateResult with the given per-combination payloads."""
    from aiperf.orchestrator.aggregation.base import AggregateResult

    return AggregateResult(
        aggregation_type="sweep",
        num_runs=len(combos),
        num_successful_runs=len(combos),
        failed_runs=[],
        metrics=combos,
        metadata={
            "sweep_parameters": [{"name": "concurrency", "values": [1, 2]}],
            "num_combinations": len(combos),
        },
    )


@pytest.mark.asyncio
async def test_aggregate_sweep_csv_header_is_union_across_combos(tmp_path) -> None:
    """R2-H6 regression: CSV header must include metrics that appear in any combo, not just combo[0]."""
    from aiperf.exporters.aggregate import (
        AggregateExporterConfig,
        AggregateSweepCsvExporter,
    )

    combos = [
        {
            "parameters": {"concurrency": 1},
            "metrics": {
                "rps": {"mean": 10.0, "std": 1.0, "min": 9.0, "max": 11.0, "cv": 0.1},
            },
        },
        {
            "parameters": {"concurrency": 2},
            "metrics": {
                "rps": {"mean": 20.0, "std": 2.0, "min": 18.0, "max": 22.0, "cv": 0.1},
                "extra": {"mean": 5.0, "std": 0.5, "min": 4.5, "max": 5.5, "cv": 0.1},
            },
        },
    ]
    cfg = AggregateExporterConfig(
        result=_make_aggregate_with_combos(combos), output_dir=tmp_path
    )
    csv_path = await AggregateSweepCsvExporter(cfg).export()
    text = csv_path.read_text()
    # Header should include both metrics
    assert "extra_mean" in text
    assert "rps_mean" in text


@pytest.mark.asyncio
async def test_aggregate_sweep_csv_header_when_first_combo_empty(tmp_path) -> None:
    """R2-H6 regression: if combo[0] has empty metrics, columns must still appear from later combos."""
    from aiperf.exporters.aggregate import (
        AggregateExporterConfig,
        AggregateSweepCsvExporter,
    )

    combos = [
        {"parameters": {"concurrency": 1}, "metrics": {}},  # failed combo
        {
            "parameters": {"concurrency": 2},
            "metrics": {
                "rps": {"mean": 20.0, "std": 2.0, "min": 18.0, "max": 22.0, "cv": 0.1}
            },
        },
    ]
    cfg = AggregateExporterConfig(
        result=_make_aggregate_with_combos(combos), output_dir=tmp_path
    )
    csv_path = await AggregateSweepCsvExporter(cfg).export()
    text = csv_path.read_text()
    assert "rps_mean" in text


def test_aggregate_sweep_csv_format_number_handles_nan_and_inf() -> None:
    """R2-M10 regression: NaN, +inf, -inf must all render as empty string (matching None)."""
    from aiperf.exporters.aggregate.aggregate_sweep_csv_exporter import (
        AggregateSweepCsvExporter,
    )

    # Bypass __init__ for a pure-function test
    exporter = AggregateSweepCsvExporter.__new__(AggregateSweepCsvExporter)
    assert exporter._format_number(float("nan")) == ""
    assert exporter._format_number(float("inf")) == ""
    assert exporter._format_number(float("-inf")) == ""
    assert exporter._format_number(None) == ""
    assert exporter._format_number(1.234) == "1.23"
    assert exporter._format_number(1.2345, decimals=4) == "1.2345"


@pytest.mark.asyncio
async def test_aggregate_sweep_json_scrubs_nan_to_null(tmp_path) -> None:
    """End-to-end: NaN/inf in per-combo metrics must serialize as JSON null, not the literal NaN."""
    import orjson

    from aiperf.exporters.aggregate import (
        AggregateExporterConfig,
        AggregateSweepJsonExporter,
    )

    combos = [
        {
            "parameters": {"concurrency": 1},
            "metrics": {
                "ttft": {
                    "mean": float("nan"),
                    "std": 1.0,
                    "min": 0.5,
                    "max": float("inf"),
                    "cv": 0.1,
                },
            },
        },
    ]
    cfg = AggregateExporterConfig(
        result=_make_aggregate_with_combos(combos), output_dir=tmp_path
    )
    json_path = await AggregateSweepJsonExporter(cfg).export()
    parsed = orjson.loads(json_path.read_bytes())
    metrics = parsed["per_combination_metrics"][0]["metrics"]["ttft"]
    assert metrics["mean"] is None
    assert metrics["max"] is None
    assert metrics["std"] == 1.0
