# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sweep-sampling edge-case regression tests.

Covers:
- ``SamplingDimension.choices`` rejects NaN/inf numeric entries.
- ``_map_dim`` clamps int output to respect declared ``[lo, hi]``
  (banker's rounding could otherwise produce values below ``lo`` for
  ``kind=int, scale=log, lo<1``).
- ``is_finite_value`` correctly handles
  ``numpy.float32`` NaN/inf (a stdlib-float-only check would
  miss it).
"""

from __future__ import annotations

import math

import pytest
from pydantic import ValidationError
from pytest import param

from aiperf.common.finite import is_finite_value
from aiperf.config.sweep.expand_qmc import _map_dim
from aiperf.config.sweep.sampling import SamplingDimension


class TestChoicesFiniteValidation:
    """SamplingDimension.choices must reject non-finite numeric entries.

    ``_validate_finite_bounds`` covers lo/hi only; without the same check on
    ``choices``, a non-finite entry slips through and crashes mid-flight at
    the orchestrator's defensive guard instead of blocking at validation time.
    """

    @pytest.mark.parametrize(
        "bad",
        [
            param(float("nan"), id="nan"),
            param(float("inf"), id="pos_inf"),
            param(float("-inf"), id="neg_inf"),
        ],
    )  # fmt: skip
    def test_choices_with_nonfinite_float_rejected(self, bad: float) -> None:
        with pytest.raises(ValidationError, match="finite"):
            SamplingDimension(path="x", choices=[bad, 1.0])

    def test_choices_mixed_numeric_finite_accepted(self) -> None:
        dim = SamplingDimension(path="x", choices=[1.0, 2.0, 3])
        assert dim.choices == [1.0, 2.0, 3]

    def test_choices_non_numeric_unaffected_strings(self) -> None:
        dim = SamplingDimension(path="x", choices=["a", "b", "c"])
        assert dim.choices == ["a", "b", "c"]

    def test_choices_non_numeric_unaffected_dicts(self) -> None:
        # Dicts are unhashable; the existing hashability validator
        # rejects them. Use tuples instead to verify the finite check
        # doesn't fire on non-numerics.
        dim = SamplingDimension(path="x", choices=[(1, "a"), (2, "b")])
        assert dim.choices == [(1, "a"), (2, "b")]

    def test_choices_mixed_str_and_finite_numeric_accepted(self) -> None:
        # Mixed numeric + non-numeric is weird but technically allowed.
        # Only the numeric entries get the finite check.
        dim = SamplingDimension(path="x", choices=["alpha", 1.0, "beta", 2])
        assert len(dim.choices) == 4

    def test_choices_with_nonfinite_among_strings_still_rejected(self) -> None:
        with pytest.raises(ValidationError, match="finite"):
            SamplingDimension(path="x", choices=["alpha", float("nan"), "beta"])


class TestIntLogScaleClamping:
    """``_map_dim`` clamps int output so it never violates declared bounds.

    Without the clamp,
    ``int(round(exp(log(0.5))))=0`` for ``lo=0.5, u=0.0``, which is
    below the user-declared ``lo``.
    """

    @pytest.mark.parametrize(
        ("lo", "hi", "scale", "warns_narrow"),
        [
            param(0.5, 1.0, "log", True, id="log_sub_one_lo"),
            param(0.5, 2.0, "log", True, id="log_lo_below_one_wider"),
            param(0.5, 1.5, "linear", True, id="linear_sub_one_lo"),
            param(1.0, 3.0, "log", False, id="log_one_to_three"),
            param(2.0, 5.0, "linear", False, id="linear_two_to_five"),
        ],
    )  # fmt: skip
    def test_int_mapping_never_below_lo_or_above_hi(
        self, lo: float, hi: float, scale: str, warns_narrow: bool
    ) -> None:
        if warns_narrow:
            with pytest.warns(UserWarning, match="too narrow for sampling diversity"):
                dim = SamplingDimension(path="x", lo=lo, hi=hi, kind="int", scale=scale)
        else:
            dim = SamplingDimension(path="x", lo=lo, hi=hi, kind="int", scale=scale)
        # Sweep the unit interval densely; all outputs must lie within
        # [ceil(lo), floor(hi)].
        lo_ceil = math.ceil(lo)
        hi_floor = math.floor(hi)
        for i in range(101):
            u = i / 100.0
            v = _map_dim(u, dim)
            assert isinstance(v, int)
            assert v >= lo_ceil, f"u={u}: produced {v}, below ceil(lo)={lo_ceil}"
            assert v <= hi_floor, f"u={u}: produced {v}, above floor(hi)={hi_floor}"

    def test_int_log_lo_half_u_zero_clamps_to_one(self) -> None:
        """Direct repro: lo=0.5, log scale,
        u=0.0 would map to 0 without the clamp; must clamp to 1."""
        with pytest.warns(UserWarning, match="too narrow for sampling diversity"):
            dim = SamplingDimension(path="x", lo=0.5, hi=1.0, kind="int", scale="log")
        assert _map_dim(0.0, dim) == 1


class TestIsFiniteValueNumpyFloat32:
    """Cross-cutting smoke for ``is_finite_value``.

    ``isinstance(x, float)`` misses
    ``numpy.float32``. ``is_finite_value`` duck-types via
    ``math.isfinite(float(x))`` so it handles all numpy float widths
    uniformly.
    """

    def test_numpy_float32_nan_is_not_finite(self) -> None:
        np = pytest.importorskip("numpy")
        assert is_finite_value(np.float32("nan")) is False

    def test_numpy_float32_pos_inf_is_not_finite(self) -> None:
        np = pytest.importorskip("numpy")
        assert is_finite_value(np.float32("inf")) is False

    def test_numpy_float32_neg_inf_is_not_finite(self) -> None:
        np = pytest.importorskip("numpy")
        assert is_finite_value(np.float32("-inf")) is False

    def test_numpy_float32_finite_value_is_finite(self) -> None:
        np = pytest.importorskip("numpy")
        assert is_finite_value(np.float32(1.5)) is True

    def test_numpy_float64_nan_is_not_finite(self) -> None:
        np = pytest.importorskip("numpy")
        assert is_finite_value(np.float64("nan")) is False

    def test_numpy_int64_finite(self) -> None:
        np = pytest.importorskip("numpy")
        assert is_finite_value(np.int64(7)) is True
