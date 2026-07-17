# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Every graph workload rejects non-chat endpoint types up front.

The graph dispatch path materializes a chat-completions body
(``{"messages": [...], "max_completion_tokens": N, "stream": bool}``) and sends
it verbatim, bypassing ``format_payload``. Pointing that body at a non-chat
endpoint (``completions`` expects ``prompt``; ``embeddings`` expects ``input``)
makes every request 422 with no actionable up-front error.
``validate_graph_endpoint_type`` guards the workload at configure time (the
DatasetManager runs it before the graph store is built), raising
``GraphEndpointUnsupportedError`` -- the guard is format-generic and serves
every graph workload (weka / dynamo / native / dag_jsonl). Chat-compatibility
is keyed on the endpoint metadata's ``endpoint_path`` ending in
``/chat/completions`` so any future chat-completions endpoint passes without an
allowlist edit.
"""

from __future__ import annotations

import pytest

from aiperf.config.flags.cli_config import CLIConfig
from aiperf.dataset.graph.workload_detect import (
    GraphEndpointUnsupportedError,
    validate_graph_endpoint_type,
)
from tests.unit.conftest import make_run_from_cli


def _run(endpoint_type: str):
    cfg = CLIConfig(model_names=["test-model"], endpoint_type=endpoint_type)
    return make_run_from_cli(cfg)


def test_chat_endpoint_passes_guard() -> None:
    """The chat endpoint (the supported shape) passes the guard silently."""
    validate_graph_endpoint_type(_run("chat"))  # no raise


@pytest.mark.parametrize(
    "endpoint_type",
    ["completions", "embeddings", "nim_embeddings"],
)
def test_non_chat_endpoint_rejected_with_actionable_error(endpoint_type: str) -> None:
    """Non-chat endpoint types raise a clear configure-time error.

    The message names the offending ``--endpoint-type`` and points at the chat
    endpoint -- a validator-gate style up-front rejection, not a per-request 422.
    """
    with pytest.raises(GraphEndpointUnsupportedError) as exc_info:
        validate_graph_endpoint_type(_run(endpoint_type))
    message = str(exc_info.value)
    assert endpoint_type in message
    assert "chat" in message.lower()
