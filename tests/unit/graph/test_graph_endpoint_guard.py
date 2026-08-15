# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Every graph workload rejects non-chat endpoint types up front."""

from __future__ import annotations

from typing import Any

import pytest
from pytest import param

from aiperf.config.flags.cli_config import CLIConfig
from aiperf.dataset.graph.workload_detect import (
    GraphEndpointUnsupportedError,
    validate_graph_endpoint_type,
)
from tests.unit.conftest import make_run_from_cli


def _run(endpoint_type: str) -> Any:
    """A minimal resolved run config pinned to the given endpoint type."""
    cfg = CLIConfig(model_names=["test-model"], endpoint_type=endpoint_type)
    return make_run_from_cli(cfg)


def test_chat_endpoint_passes_guard() -> None:
    """The chat endpoint (the supported shape) passes the guard silently."""
    validate_graph_endpoint_type(_run("chat"))  # no raise


@pytest.mark.parametrize(
    "endpoint_type",
    [
        param("completions", id="completions"),
        param("embeddings", id="embeddings"),
        param("nim_embeddings", id="nim-embeddings"),
    ],
)  # fmt: skip
def test_non_chat_endpoint_rejected_with_actionable_error(endpoint_type: str) -> None:
    """Non-chat endpoint types raise a configure-time error naming both types."""
    # The message must name the offending type AND the supported one, or the user
    # cannot tell what to change.
    with pytest.raises(GraphEndpointUnsupportedError) as exc_info:
        validate_graph_endpoint_type(_run(endpoint_type))
    message = str(exc_info.value)
    assert endpoint_type in message
    assert "chat" in message.lower()
