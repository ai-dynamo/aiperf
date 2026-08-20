# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The winner card's headline string is subject to the same guards as the table.

``formatTrialValues`` in ``components/sweep-winner-summary-helpers.js`` renders
the most prominent text on the sweep detail page, and it takes
``SearchBestTrial.variation_values``, which is ``dict[str, Any]`` -- a
truncation marker, a nested object, a null, and a list are all in-type. It used
to be a second, unguarded implementation of
``pages/sweep-detail-helpers.formatVariationValues`` and disagreed with it on
every one of those inputs, printing ``__aiperf_truncated__=true, limitBytes=256``
or ``tuning=[object Object]`` as the headline.

These tests assert the two helpers agree input-for-input, so a future guard
added to either side cannot silently apply to only one of them.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pytest import param

from tests.unit.ui.node_utils import run_node

_UI_ROOT = Path(__file__).resolve().parents[3] / "src" / "aiperf" / "operator" / "ui"
WINNER_HELPERS = _UI_ROOT / "components" / "sweep-winner-summary-helpers.js"
DETAIL_HELPERS = _UI_ROOT / "pages" / "sweep-detail-helpers.js"


def _format_both(values: object) -> dict[str, object]:
    """Run both helpers over one input and return their two answers."""
    script = f"""
        import {{ formatTrialValues }} from {WINNER_HELPERS.as_uri()!r};
        import {{ formatVariationValues }} from {DETAIL_HELPERS.as_uri()!r};
        const input = {json.dumps(values)};
        console.log(JSON.stringify({{
            trial: formatTrialValues(input),
            variation: formatVariationValues(input),
        }}));
    """
    return json.loads(run_node(script))


@pytest.mark.parametrize(
    "values,expected",
    [
        param(
            {"phases.profiling.concurrency": 17},
            "concurrency=17",
            id="scalar-is-rendered-by-leaf-name",
        ),
        param(
            {"__aiperf_truncated__": True, "limitBytes": 256},
            None,
            id="truncation-marker-is-not-a-swept-parameter",
        ),
        param(
            {"phases.profiling.concurrency": 17, "tuning": {"beta": 1}},
            "concurrency=17",
            id="nested-object-is-dropped-not-stringified",
        ),
        param(
            {"phases.profiling.concurrency": None},
            None,
            id="null-value-describes-nothing",
        ),
        param(
            {"input_sequence_length": [1, 2, 3]},
            None,
            id="list-value-has-no-honest-one-line-form",
        ),
        param({}, None, id="empty-map"),
        param(None, None, id="absent"),
    ],
)  # fmt: skip
def test_headline_matches_the_variations_table_formatting(
    values: object, expected: str | None
) -> None:
    both = _format_both(values)

    assert both["trial"] == expected
    assert both["trial"] == both["variation"], (
        "The winner card's headline and the variations table must describe a "
        "variation identically; two implementations of one concept diverge."
    )
