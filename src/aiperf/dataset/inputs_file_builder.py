# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Helpers for building the inputs.json payload from loaded conversations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from aiperf.common.models import (
    InputsFile,
    RequestInfo,
    SessionPayloads,
)
from aiperf.plugin import plugins
from aiperf.plugin.enums import PluginType

if TYPE_CHECKING:
    from aiperf.common.models import Conversation
    from aiperf.config import BenchmarkRun
    from aiperf.endpoints.protocols import EndpointProtocol


def build_inputs_file(
    run: BenchmarkRun, dataset: dict[str, Conversation]
) -> InputsFile:
    """Build the InputsFile payload from the loaded conversation dataset."""
    inputs = InputsFile()

    EndpointClass = plugins.get_class(PluginType.ENDPOINT, run.cfg.endpoint.type)
    endpoint: EndpointProtocol = EndpointClass(run=run)

    session_payloads_map: dict[str, list] = {}
    for conversation in dataset.values():
        session_id = conversation.session_id
        session_payloads_map.setdefault(session_id, [])

        for i, turn in enumerate(conversation.turns):
            request_info = RequestInfo(
                turns=[turn],
                turn_index=i,
                credit_num=i,
                credit_phase="profiling",
                x_request_id="",
                x_correlation_id="",
                conversation_id=conversation.session_id,
                system_message=conversation.system_message,
                user_context_message=conversation.user_context_message,
            )
            request_info.endpoint_headers = endpoint.get_endpoint_headers(request_info)
            request_info.endpoint_params = endpoint.get_endpoint_params(request_info)
            session_payloads_map[session_id].append(
                endpoint.format_payload(request_info)
            )

    for session_id, payloads in session_payloads_map.items():
        inputs.data.append(SessionPayloads(session_id=session_id, payloads=payloads))
    return inputs
