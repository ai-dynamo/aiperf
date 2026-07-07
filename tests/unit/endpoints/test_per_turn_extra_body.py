# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-turn ``extra_body`` merge coverage for every endpoint that formats a payload.

Regression guard: the per-turn ``extra_body`` merge (sampling overrides such as
``top_p`` / ``nvext`` / ``encoding_format`` populated from a dataset row's
``extra``) must reach the TOP-LEVEL request payload, and must be the last writer
so it wins over endpoint-level ``--extra-inputs`` on a key collision. This lived
only for ``ChatEndpoint`` (``tests/unit/workers/test_per_request_extra.py``),
which is why CI missed the 7 endpoints below dropping the merge. Real ``Turn`` /
``RequestInfo`` objects are used (never ``MagicMock``) so attr-path drift fails
loudly instead of silently no-op'ing.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from pytest import param

from aiperf.common.models import Image, Text, Turn
from aiperf.endpoints.base_endpoint import BaseEndpoint
from aiperf.endpoints.huggingface_generate import HuggingFaceGenerateEndpoint
from aiperf.endpoints.nim_image_retrieval import ImageRetrievalEndpoint
from aiperf.endpoints.nim_rankings import NIMRankingsEndpoint
from aiperf.endpoints.openai_completions import CompletionsEndpoint
from aiperf.endpoints.openai_embeddings import EmbeddingsEndpoint
from aiperf.endpoints.solido_rag import SolidoEndpoint
from aiperf.endpoints.template_endpoint import TemplateEndpoint
from aiperf.plugin.enums import EndpointType
from tests.unit.endpoints.conftest import (
    create_config,
    create_endpoint_with_mock_transport,
    create_request_info,
)

# 1x1 transparent PNG data URL for the image-retrieval endpoint.
BASE64_PNG = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="

# A collision-safe key that no endpoint sets on its own base payload, so a
# collision test isolates last-writer semantics rather than a base-key clash.
COLLISION_KEY = "aiperf_extra_probe"

# Realistic per-turn sampling overrides that flow from a dataset row's ``extra``.
EXTRA_BODY: dict[str, Any] = {"top_p": 0.9, "nvext": {"priority": 7}}


def _completions_turn(extra_body: dict[str, Any] | None) -> Turn:
    return Turn(texts=[Text(contents=["hello"])], extra_body=extra_body)


def _embeddings_turn(extra_body: dict[str, Any] | None) -> Turn:
    return Turn(texts=[Text(contents=["hello"])], extra_body=extra_body)


def _rankings_turn(extra_body: dict[str, Any] | None) -> Turn:
    return Turn(
        texts=[
            Text(name="query", contents=["What is AIPerf?"]),
            Text(name="passages", contents=["A benchmarking tool", "For LLMs"]),
        ],
        model="test-model",
        extra_body=extra_body,
    )


def _template_turn(extra_body: dict[str, Any] | None) -> Turn:
    return Turn(
        texts=[Text(contents=["hello"])], model="test-model", extra_body=extra_body
    )


def _hf_turn(extra_body: dict[str, Any] | None) -> Turn:
    return Turn(texts=[Text(contents=["hello"])], extra_body=extra_body)


def _image_turn(extra_body: dict[str, Any] | None) -> Turn:
    return Turn(images=[Image(contents=[BASE64_PNG])], extra_body=extra_body)


def _solido_turn(extra_body: dict[str, Any] | None) -> Turn:
    return Turn(texts=[Text(contents=["hello"])], extra_body=extra_body)


# id -> (EndpointType, endpoint class, turn factory, template body | None)
_CASES = [
    param(
        EndpointType.COMPLETIONS,
        CompletionsEndpoint,
        _completions_turn,
        None,
        id="completions",
    ),
    param(
        EndpointType.EMBEDDINGS,
        EmbeddingsEndpoint,
        _embeddings_turn,
        None,
        id="embeddings",
    ),
    param(
        EndpointType.NIM_RANKINGS,
        NIMRankingsEndpoint,
        _rankings_turn,
        None,
        id="rankings",
    ),
    param(
        EndpointType.TEMPLATE,
        TemplateEndpoint,
        _template_turn,
        '{"model": "{{ model }}"}',
        id="template",
    ),
    param(
        EndpointType.HUGGINGFACE_GENERATE,
        HuggingFaceGenerateEndpoint,
        _hf_turn,
        None,
        id="huggingface_generate",
    ),
    param(
        EndpointType.IMAGE_RETRIEVAL,
        ImageRetrievalEndpoint,
        _image_turn,
        None,
        id="nim_image_retrieval",
    ),
    param(
        EndpointType.SOLIDO_RAG,
        SolidoEndpoint,
        _solido_turn,
        None,
        id="solido_rag",
    ),
]  # fmt: skip


def _build(
    endpoint_type: EndpointType,
    endpoint_class: type[BaseEndpoint],
    turn: Turn,
    *,
    extra: dict[str, Any] | None = None,
    template: str | None = None,
):
    """Build a real endpoint + RequestInfo for the given turn (no MagicMock)."""
    kwargs: dict[str, Any] = {}
    if template is not None:
        kwargs["template"] = {"body": template}
    if extra is not None:
        kwargs["extra"] = extra
    cfg = create_config(endpoint_type, **kwargs)
    endpoint = create_endpoint_with_mock_transport(endpoint_class, cfg)
    request_info = create_request_info(config=cfg, turns=[turn])
    return endpoint, request_info


@pytest.mark.parametrize("endpoint_type,endpoint_class,turn_factory,template", _CASES)
def test_per_turn_extra_body_lands_in_payload(
    endpoint_type: EndpointType,
    endpoint_class: type[BaseEndpoint],
    turn_factory: Callable[[dict[str, Any] | None], Turn],
    template: str | None,
):
    """Per-turn ``extra_body`` keys appear at the top level of the formatted payload."""
    endpoint, request_info = _build(
        endpoint_type, endpoint_class, turn_factory(EXTRA_BODY), template=template
    )

    payload = endpoint.format_payload(request_info)

    for key, value in EXTRA_BODY.items():
        assert payload[key] == value


@pytest.mark.parametrize("endpoint_type,endpoint_class,turn_factory,template", _CASES)
def test_per_turn_extra_body_overrides_endpoint_extra(
    endpoint_type: EndpointType,
    endpoint_class: type[BaseEndpoint],
    turn_factory: Callable[[dict[str, Any] | None], Turn],
    template: str | None,
):
    """Per-turn ``extra_body`` is the last writer, overriding endpoint-level extra."""
    endpoint, request_info = _build(
        endpoint_type,
        endpoint_class,
        turn_factory({COLLISION_KEY: "per_turn"}),
        extra={COLLISION_KEY: "endpoint_level"},
        template=template,
    )

    payload = endpoint.format_payload(request_info)

    assert payload[COLLISION_KEY] == "per_turn"


@pytest.mark.parametrize("endpoint_type,endpoint_class,turn_factory,template", _CASES)
def test_none_extra_body_produces_valid_payload(
    endpoint_type: EndpointType,
    endpoint_class: type[BaseEndpoint],
    turn_factory: Callable[[dict[str, Any] | None], Turn],
    template: str | None,
):
    """A turn with ``extra_body=None`` formats a payload without adding extra keys."""
    endpoint, request_info = _build(
        endpoint_type, endpoint_class, turn_factory(None), template=template
    )

    payload = endpoint.format_payload(request_info)

    assert isinstance(payload, dict)
    for key in EXTRA_BODY:
        assert key not in payload


def test_huggingface_extra_body_merges_top_level_not_parameters():
    """HF TGI ``extra_body`` must merge into the TOP-LEVEL payload, never ``parameters``."""
    endpoint, request_info = _build(
        EndpointType.HUGGINGFACE_GENERATE,
        HuggingFaceGenerateEndpoint,
        _hf_turn({"top_p": 0.9}),
    )

    payload = endpoint.format_payload(request_info)

    assert payload["top_p"] == 0.9
    assert "top_p" not in payload["parameters"]
