# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Round-4 adversarial regressions for the sweep + config system.

Continues the H-series after ``test_sweep_round3_adversarial.py``:

- H20: ``SLAFilter`` rejects NaN/inf threshold and empty/whitespace
  ``metric_tag`` (was: silent feasibility false-fail when threshold is
  NaN; meaningless filter when metric_tag is empty).
- H21: ``PostProcessSpec`` rejects empty handler, NUL bytes in
  ``output_filename``, and dot-only stems like ``..json`` / ``.json``
  (was: hidden-file confusion + OS-level open failures at write time).
- H22: ``detect_sweep_fields`` is scoped to phase-rooted paths only --
  a list at ``datasets.X.prompts.isl.mean = [100, 200]`` no longer
  silently auto-sweeps and produces validation-failing variants.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from aiperf.config.sweep.adaptive import SLAFilter
from aiperf.config.sweep.expand import detect_sweep_fields, expand_sweep
from aiperf.search_recipes._post_process import PostProcessSpec

# -- H20: SLAFilter validation -----------------------------------------------


class TestH20SLAFilterValidation:
    """SLAFilter used to accept any float threshold and any string
    metric_tag, including degenerate inputs that silently broke filtering.
    """

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_threshold_rejected(self, bad: float) -> None:
        with pytest.raises(ValidationError, match="finite"):
            SLAFilter(metric_tag="ttft", op="lt", threshold=bad)

    @pytest.mark.parametrize("bad", ["", " ", "\t", "   \n"])
    def test_empty_or_whitespace_metric_tag_rejected(self, bad: str) -> None:
        with pytest.raises(ValidationError, match="non-empty"):
            SLAFilter(metric_tag=bad, op="lt", threshold=200.0)

    def test_normal_filter_allowed(self) -> None:
        f = SLAFilter(
            metric_tag="time_to_first_token",
            stat="p95",
            op="lt",
            threshold=200.0,
        )
        assert f.metric_tag == "time_to_first_token"
        assert f.threshold == 200.0


# -- H21: PostProcessSpec validation ----------------------------------------


class TestH21PostProcessSpecValidation:
    """PostProcessSpec used to accept hidden / dot-stem filenames, NUL
    bytes, and empty handler names.
    """

    def test_empty_handler_rejected(self) -> None:
        with pytest.raises(ValidationError, match="non-empty"):
            PostProcessSpec(handler="", output_filename="x.json")

    def test_whitespace_handler_rejected(self) -> None:
        with pytest.raises(ValidationError, match="non-empty"):
            PostProcessSpec(handler="   ", output_filename="x.json")

    @pytest.mark.parametrize(
        "bad_filename",
        [
            ".json",  # leading-dot only stem
            "..json",  # parent-resembling stem
            "...json",  # all-dot stem
        ],
    )
    def test_dot_stem_filename_rejected(self, bad_filename: str) -> None:
        with pytest.raises(ValidationError, match=r"non-dot stem"):
            PostProcessSpec(handler="h", output_filename=bad_filename)

    def test_nul_byte_filename_rejected(self) -> None:
        with pytest.raises(ValidationError, match="NUL bytes"):
            PostProcessSpec(handler="h", output_filename="a\x00b.json")

    def test_normal_filename_allowed(self) -> None:
        spec = PostProcessSpec(handler="h", output_filename="ttft_curve.json")
        assert spec.output_filename == "ttft_curve.json"

    def test_dot_in_stem_allowed(self) -> None:
        # A '.' in the middle of the stem is fine; only dot-only stems are
        # rejected.
        spec = PostProcessSpec(handler="h", output_filename="ttft.curve.json")
        assert spec.output_filename == "ttft.curve.json"


# -- H22: magic-list scope narrowed to phase-rooted paths -------------------


class TestH22MagicListScope:
    """detect_sweep_fields used to traverse the entire benchmark subtree;
    a list at any magic-named key (``mean``, ``count``, ...) anywhere in
    the tree got auto-swept. Now scoped to ``phases.*`` only.
    """

    def test_mean_inside_distribution_not_swept(self) -> None:
        # `mean` is in MAGIC_LIST_FIELDS but here it's the discriminator key
        # of a Normal distribution -- a list there is a user error, not a
        # sweep request.
        data = {"datasets": [{"prompts": {"isl": {"mean": [100, 200], "stddev": 10}}}]}
        assert detect_sweep_fields(data) == {}

    def test_count_in_unrelated_path_not_swept(self) -> None:
        data = {"datasets": [{"prompts": {"count": [1, 2, 3]}}]}
        assert detect_sweep_fields(data) == {}

    def test_phase_concurrency_still_swept(self) -> None:
        data = {
            "phases": [
                {"name": "profiling", "concurrency": [1, 2, 4]},
            ]
        }
        fields = detect_sweep_fields(data)
        assert fields == {"phases.profiling.concurrency": [1, 2, 4]}

    def test_phase_flat_shape_still_swept(self) -> None:
        # YAML flat-shape phases (single dict instead of list of dicts).
        data = {"phases": {"concurrency": [1, 2, 4]}}
        fields = detect_sweep_fields(data)
        assert fields == {"phases.concurrency": [1, 2, 4]}

    def test_e2e_no_phantom_sweep_from_distribution_list(self) -> None:
        # End-to-end: confirm the false-positive bug repro no longer expands.
        data = {
            "benchmark": {
                "datasets": [
                    {
                        "name": "d",
                        "type": "synthetic",
                        "entries": 1,
                        "prompts": {
                            "isl": {"mean": [100, 200], "stddev": 10},
                            "osl": 32,
                        },
                    }
                ],
                "phases": [
                    {
                        "name": "profiling",
                        "type": "concurrency",
                        "concurrency": 1,
                        "requests": 1,
                    }
                ],
            },
        }
        out = expand_sweep(data)
        # No magic-list expansion -> single base variation.
        assert len(out) == 1
        assert out[0][1].label == "base"
