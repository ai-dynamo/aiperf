# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``--accuracy-enable-cot`` / ``--accuracy-no-enable-cot`` CLI tri-state.

``accuracy_enable_cot`` is ``bool | None``: unset defers to the benchmark's
``default_enable_cot`` metadata, ``--accuracy-enable-cot`` forces CoT on, and
``--accuracy-no-enable-cot`` forces it off. The negative flag is what makes the
non-CoT path reachable for CoT-default benchmarks (e.g. ``mmlu_pro``), so it is
covered here at the cyclopts-parse boundary rather than only at the field level.
"""

from __future__ import annotations

import pytest
from pytest import param

from aiperf.cli_commands.profile import app

_BASE_ARGV = [
    "--model",
    "m",
    "--url",
    "http://localhost:1/v1",
    "--accuracy-benchmark",
    "mmlu",
]


def _parse_enable_cot(extra_argv: list[str]) -> bool | None:
    _cmd, bound, _ignored = app.parse_args(
        _BASE_ARGV + extra_argv, exit_on_error=False, verbose=False
    )
    return bound.arguments["cli_config"].accuracy_enable_cot


@pytest.mark.parametrize(
    "extra_argv,expected",
    [
        param([], None, id="unset_defers_to_metadata"),
        param(["--accuracy-enable-cot"], True, id="enable_forces_cot_on"),
        param(["--accuracy-no-enable-cot"], False, id="no_enable_forces_cot_off"),
    ],
)  # fmt: skip
def test_accuracy_enable_cot_tristate(
    extra_argv: list[str], expected: bool | None
) -> None:
    assert _parse_enable_cot(extra_argv) is expected
