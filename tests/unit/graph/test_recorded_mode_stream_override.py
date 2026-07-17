# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-request stream override for graph credits (Task 2).

The recorded per-node wire mode (weka ``"n"``/``"s"``, dynamo ``ttft_ms``) wins
for graph credits over the global ``endpoint.streaming`` flag; a mode-less node
(override ``None``) follows the global. These tests pin the precedence at the
payload-stamp seam (``apply_run_level_payload_options``) and the ``RequestInfo``
carrier field; the transport-side wiring is covered in
``tests/unit/transports/test_aiohttp_transport.py``.
"""

from __future__ import annotations

import pytest
from pytest import param

from aiperf.common.enums import CreditPhase, ModelSelectionStrategy
from aiperf.common.models import EndpointInfo
from aiperf.common.models.model_endpoint_info import (
    ModelEndpointInfo,
    ModelInfo,
    ModelListInfo,
)
from aiperf.common.models.record_models import RequestInfo
from aiperf.graph.worker_materialize import apply_run_level_payload_options
from aiperf.plugin.enums import EndpointType


def _endpoint(
    *,
    streaming: bool,
    use_server_token_count: bool = False,
    extra: list[tuple[str, object]] | None = None,
) -> EndpointInfo:
    return EndpointInfo(
        type=EndpointType.CHAT,
        streaming=streaming,
        use_server_token_count=use_server_token_count,
        extra=extra or [],
    )


@pytest.mark.parametrize(
    "override,glob,expected",
    [
        param(True, False, True, id="node-stream-wins-over-global-off"),
        param(False, True, False, id="node-nonstream-wins-over-global-on"),
        param(None, True, True, id="no-override-follows-global-on"),
        param(None, False, False, id="no-override-follows-global-off"),
    ],
)  # fmt: skip
def test_apply_run_level_stream_precedence(override, glob, expected):
    """Recorded per-node mode wins; ``None`` follows the global flag."""
    payload = {"stream": "stale"}
    apply_run_level_payload_options(
        payload, _endpoint(streaming=glob), stream_override=override
    )
    assert payload["stream"] is expected


def test_include_usage_follows_final_stream():
    """``include_usage`` keys on the FINAL stamped ``stream``, not the global.

    ``stream_override=False`` + global True + ``use_server_token_count`` forces
    the wire to non-streaming, so no ``stream_options`` is layered; the converse
    (override True over global False) DOES layer it.
    """
    # Override False beats global True -> final stream False -> no usage forced.
    payload = {"messages": [{"role": "user", "content": "hi"}], "stream": True}
    apply_run_level_payload_options(
        payload,
        _endpoint(streaming=True, use_server_token_count=True),
        stream_override=False,
    )
    assert payload["stream"] is False
    assert "stream_options" not in payload

    # Override True beats global False -> final stream True -> usage forced.
    payload2 = {"messages": [{"role": "user", "content": "hi"}], "stream": False}
    apply_run_level_payload_options(
        payload2,
        _endpoint(streaming=False, use_server_token_count=True),
        stream_override=True,
    )
    assert payload2["stream"] is True
    assert payload2["stream_options"] == {"include_usage": True}


def test_request_info_stream_override_defaults_none():
    """A plain (non-graph) ``RequestInfo`` construction leaves the override None."""
    model_endpoint = ModelEndpointInfo(
        models=ModelListInfo(
            models=[ModelInfo(name="test-model")],
            model_selection_strategy=ModelSelectionStrategy.ROUND_ROBIN,
        ),
        endpoint=EndpointInfo(
            type=EndpointType.CHAT, base_url="http://localhost:8000/v1/chat"
        ),
    )
    request_info = RequestInfo(
        model_endpoint=model_endpoint,
        turns=[],
        turn_index=0,
        credit_num=0,
        credit_phase=CreditPhase.PROFILING,
        x_request_id="rid",
        x_correlation_id="cid",
        conversation_id="conv",
    )
    assert request_info.stream_override is None
