# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``aiperf kube results`` must name the operating point it downloaded.

``v8-t0 (sweep-v08-t0)`` identifies a directory; under an adaptive planner even
the variation label is ``search_iter_0008``. ``_cell_values`` turns the
manifest's swept values into the descriptor that gets appended, and must return
"" rather than a partial descriptor whenever the values are unusable.
"""

from __future__ import annotations

import pytest
from pytest import param

from aiperf.kubernetes.results import _cell_values


@pytest.mark.parametrize(
    ("entry", "expected"),
    [
        param(
            {"variationValues": '{"phases.profiling.concurrency":17}'},
            "concurrency=17",
            id="camel-case-wire-shape",
        ),
        param(
            {"variation_values": '{"phases.profiling.concurrency":17}'},
            "concurrency=17",
            id="snake-case-live-shape",
        ),
        param(
            {"variationValues": '{"a.b":1,"c":2}'},
            "b=1, c=2",
            id="multiple-dimensions-shortened-to-leaves",
        ),
        param(
            {"variationValues": {"phases.profiling.concurrency": 17}},
            "concurrency=17",
            id="already-parsed-object",
        ),
    ],
)  # fmt: skip
def test_cell_values_describes_the_operating_point(entry: dict, expected: str) -> None:
    assert _cell_values(entry) == expected


@pytest.mark.parametrize(
    "entry",
    [
        param({}, id="older-archive-without-the-field"),
        param({"variationValues": ""}, id="empty-string"),
        param({"variationValues": "not json"}, id="unparseable"),
        param({"variationValues": "[1,2]"}, id="not-an-object"),
        param(
            {"variationValues": '{"__aiperf_truncated__":true,"limitBytes":256}'},
            id="writer-side-truncation-marker",
        ),
        param({"variationValues": '{"concurrency":null}'}, id="value-less-entry"),
        param({"variationValues": "{}"}, id="empty-object"),
        param(
            {"variationValues": '{"variables.tuning":{"a":1}}'},
            id="only-nested-values",
        ),
    ],
)  # fmt: skip
def test_cell_values_is_empty_rather_than_partial(entry: dict) -> None:
    """The caller prints the plain `v8-t0 (name): OK` line when this is "",
    so an unusable payload degrades to today's output instead of asserting a
    parameter that was never applied."""
    assert _cell_values(entry) == ""
