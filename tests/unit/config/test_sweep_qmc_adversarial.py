# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial regression tests for QMC sweep validators.

One test per hypothesis from the adversarial QMC report, named by
hypothesis ID (H1, H3, H4, H5, H6, etc.). Covers validator gaps that
allowed silent-wrong configs and crashes through to expansion time.
"""

from __future__ import annotations

import warnings

import pytest
from pydantic import ValidationError

from aiperf.config.sweep import (
    LatinHypercubeSweep,
    SamplingDimension,
    SobolSweep,
)
from aiperf.config.sweep.expand_qmc import expand_qmc_sweep


class TestH1EmptyChoicesRejected:
    """H1: SamplingDimension(choices=[]) used to pass validation and then
    crash at expansion with IndexError.
    """

    def test_empty_choices_rejected_at_validation(self) -> None:
        with pytest.raises(ValidationError, match="at least one entry"):
            SamplingDimension(path="x", choices=[])

    def test_singleton_choices_allowed(self) -> None:
        # Length-1 choices is degenerate but valid (always picks the one entry).
        dim = SamplingDimension(path="x", choices=["only"])
        assert dim.choices == ["only"]


class TestH3DuplicateDimPathsRejected:
    """H3 (formerly H4 in report): two dims sharing the same `path`
    silently overwrote earlier values in the variant dict.
    """

    def test_duplicate_paths_rejected_sobol(self) -> None:
        with pytest.raises(ValidationError, match=r"unique paths.*'x'"):
            SobolSweep(
                samples=4,
                dimensions=[
                    SamplingDimension(path="x", lo=1, hi=10),
                    SamplingDimension(path="x", lo=1000, hi=10000),
                ],
            )

    def test_duplicate_paths_rejected_lhs(self) -> None:
        with pytest.raises(ValidationError, match=r"unique paths"):
            LatinHypercubeSweep(
                samples=4,
                dimensions=[
                    SamplingDimension(path="a.b", lo=1, hi=2),
                    SamplingDimension(path="a.b", choices=[1, 2, 3]),
                ],
            )

    def test_distinct_paths_accepted(self) -> None:
        # Sanity check: we didn't break the happy path.
        sweep = SobolSweep(
            samples=4,
            dimensions=[
                SamplingDimension(path="x", lo=1, hi=10),
                SamplingDimension(path="y", lo=1, hi=10),
            ],
        )
        assert len(sweep.dimensions) == 2


class TestH5PhantomKeyPaths:
    """H5: pathological dotted paths used to silently create empty-string
    keys at the envelope root (e.g. path="" -> {"": value}).
    """

    @pytest.mark.parametrize(
        "bad_path",
        ["", ".", "..a", "a..b", ".a", "a."],
    )
    def test_phantom_path_rejected(self, bad_path: str) -> None:
        with pytest.raises(ValidationError):
            SamplingDimension(path=bad_path, lo=1, hi=10)

    def test_sweep_prefix_rejected(self) -> None:
        with pytest.raises(ValidationError, match="sweep config"):
            SamplingDimension(path="sweep.type", lo=1, hi=10)

    def test_sweep_substring_in_segment_allowed(self) -> None:
        # Only the first segment matches "sweep" — "sweepy" is fine.
        dim = SamplingDimension(path="sweepy.value", lo=1, hi=10)
        assert dim.path == "sweepy.value"


class TestH2NarrowIntRangeWarns:
    """H2 (also tracked as int-log collapse): narrow integer ranges
    cause sample dedup. The model now warns but does not raise.
    """

    def test_narrow_int_log_range_warns(self) -> None:
        with pytest.warns(UserWarning, match="too narrow for sampling diversity"):
            SamplingDimension(path="x", lo=1, hi=2, scale="log", kind="int")

    def test_narrow_int_linear_range_warns(self) -> None:
        # hi - lo == 1, < 2 -> too narrow.
        with pytest.warns(UserWarning, match="too narrow"):
            SamplingDimension(path="x", lo=5, hi=6, scale="linear", kind="int")

    def test_wide_int_log_range_no_warning(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            SamplingDimension(path="x", lo=1, hi=256, scale="log", kind="int")

    def test_real_kind_no_warning_on_narrow_range(self) -> None:
        # real kind doesn't collapse; no warning.
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            SamplingDimension(path="x", lo=1, hi=2, scale="log", kind="real")


class TestH3ScrambleFalseWarns:
    """H3 (renumbered to scramble corner): scramble=False starts at the
    origin, so the first variant is (lo, lo, ...). The expander now
    warns when this is requested.
    """

    def test_scramble_false_emits_warning(self) -> None:
        dims = [SamplingDimension(path="x", lo=1, hi=10)]
        with pytest.warns(UserWarning, match="scramble=False"):
            expand_qmc_sweep(
                {"benchmark": {}},
                sweep_type="sobol",
                samples=4,
                seed=42,
                dimensions=dims,
                options={"scramble": False},
            )

    def test_scramble_true_no_warning(self) -> None:
        dims = [SamplingDimension(path="x", lo=1, hi=10)]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            expand_qmc_sweep(
                {"benchmark": {}},
                sweep_type="sobol",
                samples=4,
                seed=42,
                dimensions=dims,
                options={"scramble": True},
            )
        scramble_warnings = [w for w in caught if "scramble=False" in str(w.message)]
        assert scramble_warnings == []


class TestH6SamplesUpperBound:
    """H6: `samples` had no upper bound; users could submit configs
    that would explode at expansion time.
    """

    def test_samples_above_cap_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SobolSweep(
                samples=2**21,
                dimensions=[SamplingDimension(path="x", lo=1, hi=10)],
            )

    def test_samples_at_cap_accepted(self) -> None:
        sweep = SobolSweep(
            samples=2**20,
            dimensions=[SamplingDimension(path="x", lo=1, hi=10)],
        )
        assert sweep.samples == 2**20

    def test_samples_huge_int_rejected(self) -> None:
        with pytest.raises(ValidationError):
            LatinHypercubeSweep(
                samples=10**9,
                dimensions=[SamplingDimension(path="x", lo=1, hi=10)],
            )


class TestH7LhsNoScrambleField:
    """H7 (cosmetic): LatinHypercubeSweep should not accept a
    `scramble` field — that's a Sobol-only knob.
    """

    def test_lhs_rejects_scramble_field(self) -> None:
        with pytest.raises(ValidationError):
            LatinHypercubeSweep(
                samples=4,
                scramble=True,  # type: ignore[call-arg]
                dimensions=[SamplingDimension(path="x", lo=1, hi=10)],
            )


class TestH8NonFiniteBoundsRejected:
    """H8 (renumbered): NaN/inf in lo/hi used to flow through and
    poison the variant dict / sampling_design.json artifact.
    """

    @pytest.mark.parametrize(
        "lo,hi",
        [
            (float("nan"), 10.0),
            (1.0, float("nan")),
            (float("-inf"), 10.0),
            (1.0, float("inf")),
            (float("inf"), float("inf")),
        ],
    )
    def test_non_finite_bounds_rejected(self, lo: float, hi: float) -> None:
        with pytest.raises(ValidationError, match="finite"):
            SamplingDimension(path="x", lo=lo, hi=hi)

    def test_finite_bounds_accepted(self) -> None:
        # Sanity: finite floats still pass.
        dim = SamplingDimension(path="x", lo=1.0, hi=10.0)
        assert dim.lo == 1.0
        assert dim.hi == 10.0
