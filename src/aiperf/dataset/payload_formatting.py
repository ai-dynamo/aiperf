# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared payload formatting logic for dataset processing.

Provides a generator that creates formatted API request payloads from
conversations using an endpoint protocol. Used by both the dataset manager
(inputs.json generation) and the custom composer (payload pre-formatting).
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any

from aiperf.common.enums import CreditPhase
from aiperf.common.models import Conversation
from aiperf.common.models.model_endpoint_info import ModelEndpointInfo
from aiperf.common.models.record_models import RequestInfo
from aiperf.plugin import plugins
from aiperf.plugin.enums import PluginType


def format_conversation_payloads(
    conversations: Iterable[Conversation],
    model_endpoint: ModelEndpointInfo,
) -> Iterator[tuple[str, int, dict[str, Any]]]:
    """Yield formatted payloads for each turn in the given conversations.

    Creates an endpoint instance and iterates over all turns, producing
    (session_id, turn_index, payload) tuples.

    Args:
        conversations: Conversations to format payloads for.
        model_endpoint: Endpoint configuration for payload formatting.

    Yields:
        Tuples of (session_id, turn_index, formatted_payload_dict).

    Raises:
        NotImplementedError: If the endpoint does not support format_payload.
    """
    EndpointClass = plugins.get_class(PluginType.ENDPOINT, model_endpoint.endpoint.type)
    endpoint = EndpointClass(model_endpoint=model_endpoint)

    for conversation in conversations:
        for i, turn in enumerate(conversation.turns):
            request_info = RequestInfo(
                model_endpoint=model_endpoint,
                turns=[turn],
                turn_index=i,
                credit_num=i,
                credit_phase=CreditPhase.PROFILING,
                x_request_id="",
                x_correlation_id="",
                conversation_id=conversation.session_id,
            )
            request_info.endpoint_headers = endpoint.get_endpoint_headers(request_info)
            request_info.endpoint_params = endpoint.get_endpoint_params(request_info)
            yield conversation.session_id, i, endpoint.format_payload(request_info)
