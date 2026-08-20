# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The sweep_variations index must record the swept parameters, not the cell id.

``variation_values`` is what makes two rows comparable --
``_stable_variation_values`` strips ``trial_index`` from it so trials of one
parameter point collapse together. Before the children manifest carried the
values there was nothing to store but ``variation_label``, which for an
adaptive sweep is ``search_iter_NNNN``: an artifact-path cell id that says
nothing about what was tried and differs for every trial of the SAME point.
"""

from __future__ import annotations

import pytest

from aiperf.operator.runs_index import (
    _child_variation_values,
    _stable_variation_values,
)


def test_swept_parameters_are_recorded_from_the_manifest() -> None:
    values = _child_variation_values(
        {
            "variation_values": '{"phases.profiling.concurrency":17}',
            "variation_label": "search_iter_0008",
            "trial_index": None,
        }
    )

    assert values["phases.profiling.concurrency"] == 17
    # The label is retained so a row stays identifiable, but it is no longer
    # the only thing describing the variation.
    assert values["variation_label"] == "search_iter_0008"


def test_trials_of_one_point_collapse_once_values_are_present() -> None:
    """The grouping consequence, which is the whole reason this column exists."""
    trials = [
        _child_variation_values(
            {
                "variation_values": '{"phases.profiling.concurrency":17}',
                "variation_label": "search_iter_0008",
                "trial_index": trial,
            }
        )
        for trial in (0, 1, 2)
    ]

    stable = [_stable_variation_values(t) for t in trials]
    assert stable[0] == stable[1] == stable[2], "trials of one point must group"
    assert stable[0]["phases.profiling.concurrency"] == 17


@pytest.mark.parametrize(
    "raw",
    [
        '{"__aiperf_truncated__":true,"limitBytes":256,"reason":"too large"}',
        {"__aiperf_truncated__": True},
    ],
    ids=["json-string", "already-parsed"],
)  # fmt: skip
def test_the_truncation_marker_is_not_indexed(raw) -> None:
    """A marker means the values are UNKNOWN, not that they are these keys.

    Indexing it would mint one bogus group shared by every oversized variation.
    Mirrors the read-side rule at kubernetes/results.py:704.
    """
    values = _child_variation_values(
        {"variation_values": raw, "variation_label": "search_iter_0008"}
    )

    assert "__aiperf_truncated__" not in values
    assert "limitBytes" not in values
    assert values == {"variation_label": "search_iter_0008"}


@pytest.mark.parametrize(
    "raw",
    ["", "garbage", "[1,2,3]", None],
    ids=["empty", "unparseable", "json-but-not-an-object", "absent"],
)  # fmt: skip
def test_unusable_values_degrade_to_the_label(raw) -> None:
    values = _child_variation_values(
        {"variation_values": raw, "variation_label": "concurrency-8"}
    )

    assert values == {"variation_label": "concurrency-8"}
