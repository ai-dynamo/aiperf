# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The dynamo ctx gate must fire on BOTH build entries, not just ``parse``.

The gate lived only in ``DynamoTraceAdapter.parse``, but the store builder's
streaming route -- the DEFAULT for dynamo -- calls
``stream_dynamo_trace_segment_payloads`` directly and forwards a hand-picked
subset of ctx fields, so both refused knobs were silently ignored there.
"""

from __future__ import annotations

import inspect

import pytest
from pytest import param

from aiperf.dataset.graph.adapters.dynamo.trace import assert_ctx_knobs_supported
from aiperf.dataset.graph.adapters.dynamo.trace_reader import DynamoTraceAdapterError
from aiperf.dataset.graph.parse_context import GraphParseContext


@pytest.mark.parametrize(
    "kwargs,flag",
    [
        param({"use_think_time_only": True}, "--use-think-time-only", id="think-time"),
        param(
            {"delay_cap_seconds": 5.0},
            "--inter-turn-delay-cap-seconds",
            id="inter-turn-cap",
        ),
    ],
)  # fmt: skip
def test_unsupported_ctx_knob_raises_naming_the_flag(
    kwargs: dict[str, object], flag: str
) -> None:
    """Each refused knob fails loud and names the CLI flag the operator passed."""
    with pytest.raises(DynamoTraceAdapterError) as exc:
        assert_ctx_knobs_supported(GraphParseContext(**kwargs))
    assert flag in str(exc.value)


@pytest.mark.parametrize(
    "ctx",
    [
        param(None, id="no-ctx"),
        param(GraphParseContext(), id="default-ctx"),
        param(GraphParseContext(idle_gap_cap_seconds=2.0), id="supported-knob"),
    ],
)  # fmt: skip
def test_supported_ctx_passes(ctx: GraphParseContext | None) -> None:
    """A ctx carrying only supported knobs is accepted."""
    assert_ctx_knobs_supported(ctx)


def test_streaming_route_applies_the_gate() -> None:
    """The store builder's streaming branch must call the gate.

    Source-text guard rather than a full build: the regression is structural
    (a code path that skips the check), and reproducing it end-to-end needs a
    real corpus plus a worker pool. If this method stops calling the gate, the
    default dynamo path silently ignores both flags again.
    """
    from aiperf.dataset.graph.store_build import GraphStoreBuilder

    src = inspect.getsource(GraphStoreBuilder)
    assert "assert_ctx_knobs_supported(ctx)" in src, (
        "the streaming dynamo route must apply the adapter's ctx gate; "
        "it does not call DynamoTraceAdapter.parse"
    )
