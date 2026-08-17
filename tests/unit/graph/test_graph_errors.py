# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The graph failure taxonomy that survives the credit-return wire."""

import pytest
from pytest import param

from aiperf.graph.errors import GraphErrorCode, format_graph_error, parse_graph_error


def test_format_then_parse_round_trips() -> None:
    wire = format_graph_error(GraphErrorCode.POOL_MISSING, "t-1::3f2a/7")
    assert parse_graph_error(wire) is GraphErrorCode.POOL_MISSING


def test_format_renders_the_value_not_the_member_repr() -> None:
    """A str-Enum whose __str__ is not overridden would emit 'GraphErrorCode.X'."""
    assert format_graph_error(GraphErrorCode.POOL_MISSING, "d").startswith(
        "aiperf.graph.pool_missing: "
    )


@pytest.mark.parametrize(
    "error,expected",
    [
        param(None, None, id="none"),
        param("", None, id="empty"),
        param("some upstream 500", None, id="unrelated-error"),
        param("aiperf.graph.pool_missing: t-1/3", GraphErrorCode.POOL_MISSING, id="pool"),
        param("aiperf.graph.capture_failed: boom", GraphErrorCode.CAPTURE_FAILED, id="capture"),
    ],
)  # fmt: skip
def test_parse_graph_error(error: str | None, expected: GraphErrorCode | None) -> None:
    assert parse_graph_error(error) is expected


def test_parse_ignores_error_details_repr_noise() -> None:
    """A stringified ErrorDetails must not be mistaken for a bare code.

    ErrorDetails.__str__ embeds code=/type=/cause= noise, so a prefix match
    against its repr would misclassify. Parsing requires the code to lead.
    """
    assert parse_graph_error("message='aiperf.graph.pool_missing: x' code=500") is None
