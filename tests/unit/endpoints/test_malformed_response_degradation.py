# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Differential tests: every endpoint's ``parse_response`` must DEGRADE (never
crash) on a malformed HTTP-200 body.

Wave 8 fixed ``ChatEndpoint`` (openai_chat.py) so a malformed ``choices`` shape
degrades to ``None`` instead of raising ``AttributeError`` at
``choices[0].get(...)``. Wave 9 found the same class — an unguarded ``.get(...)``
/ subscript on a list element or nested item whose type came from parsed server
JSON — left open in every OTHER endpoint.

The crash matters because the worker parses the response unconditionally
(``worker.py`` ``_request_latency_ns_for_record`` at line 1110, reached from
line 1088) BEFORE ``_send_inference_result_message`` (line 1095). A raised
``AttributeError`` there DROPS the ``RequestRecord`` from the metrics pipeline
and mislabels the credit with a raw exception, instead of the body being cleanly
accounted as an error record.

Each malformed body below previously crashed the named endpoint. The contract:
``parse_response`` returns cleanly (degrading to ``None`` or skipping the bad
item) — it must NEVER raise ``AttributeError`` / ``KeyError`` / ``TypeError`` /
``IndexError``. A valid body must still parse (no regression).
"""

from __future__ import annotations

from typing import Any

import pytest
from pytest import param

from aiperf.common.models.record_models import (
    EmbeddingResponseData,
    ImageResponseData,
    ParsedResponse,
    RankingsResponseData,
    TextResponseData,
)
from aiperf.endpoints.base_endpoint import BaseEndpoint
from aiperf.endpoints.chat_embeddings import ChatEmbeddingsEndpoint
from aiperf.endpoints.cohere_rankings import CohereRankingsEndpoint
from aiperf.endpoints.hf_tei_rankings import HFTeiRankingsEndpoint
from aiperf.endpoints.huggingface_generate import HuggingFaceGenerateEndpoint
from aiperf.endpoints.nim_image_retrieval import ImageRetrievalEndpoint
from aiperf.endpoints.nim_rankings import NIMRankingsEndpoint
from aiperf.endpoints.openai_completions import CompletionsEndpoint
from aiperf.endpoints.openai_embeddings import EmbeddingsEndpoint
from aiperf.endpoints.openai_image_edit import ImageEditEndpoint
from aiperf.endpoints.openai_image_generation import ImageGenerationEndpoint
from aiperf.endpoints.openai_responses import ResponsesEndpoint
from aiperf.endpoints.openai_video_generation import VideoGenerationEndpoint
from aiperf.endpoints.raw_endpoint import RawEndpoint
from aiperf.endpoints.solido_rag import SolidoEndpoint
from aiperf.endpoints.template_endpoint import TemplateEndpoint
from aiperf.plugin.enums import EndpointType
from tests.unit.endpoints.conftest import (
    create_config,
    create_endpoint_with_mock_transport,
    create_mock_response,
)

# Exceptions that signal the unguarded-``.get``/subscript bug class. A malformed
# body must degrade cleanly instead of raising any of these.
BUG_CLASS_EXCEPTIONS = (AttributeError, KeyError, TypeError, IndexError)

# Reusable malformed first-list-element values that previously crashed
# ``<element>.get(...)`` on parsed server JSON.
_NON_DICT_ELEMENTS = [None, "x", 5]


def _make_endpoint(cls, endpoint_type: EndpointType, template: dict | None = None):
    cfg = create_config(endpoint_type, template=template)
    return create_endpoint_with_mock_transport(cls, cfg)


def _parse_without_bug_crash(endpoint, body: Any) -> ParsedResponse | None:
    """Call ``parse_response`` and fail loudly if it raises the bug-class error."""
    response = create_mock_response(123456789, body)
    try:
        return endpoint.parse_response(response)
    except BUG_CLASS_EXCEPTIONS as exc:  # pragma: no cover - failure path
        pytest.fail(
            f"{type(endpoint).__name__}.parse_response raised "
            f"{type(exc).__name__} on malformed body {body!r}: {exc}"
        )


# ---------------------------------------------------------------------------
# Endpoints that MUST degrade a malformed body to ``None``.
# ---------------------------------------------------------------------------


def _completions_malformed() -> list[tuple[str, Any]]:
    base = {"object": "text_completion"}
    bodies = [(f"first-{e!r}", {**base, "choices": [e]}) for e in _NON_DICT_ELEMENTS]
    bodies += [
        ("choices-str", {**base, "choices": "oops"}),
        ("choices-dict", {**base, "choices": {"text": "hi"}}),
        ("choices-empty", {**base, "choices": []}),
        ("choices-missing", dict(base)),
    ]
    return bodies


def _cohere_malformed() -> list[tuple[str, Any]]:
    bodies = [(f"item-{e!r}", {"results": [e]}) for e in _NON_DICT_ELEMENTS]
    bodies += [
        ("results-str", {"results": "oops"}),
        ("results-empty", {"results": []}),
        ("results-missing", {}),
    ]
    return bodies


def _hf_malformed() -> list[tuple[str, Any]]:
    bodies = [(f"first-{e!r}", [e]) for e in _NON_DICT_ELEMENTS]
    bodies += [
        ("empty-list", []),
        ("empty-dict", {}),
        ("empty-text", {"generated_text": ""}),
    ]
    return bodies


def _responses_malformed() -> list[tuple[str, Any]]:
    return [
        ("response-str", {"type": "response.completed", "response": "oops"}),
        ("response-int", {"type": "response.completed", "response": 5}),
        ("output-item-none", {"object": "response", "output": [None]}),
        ("output-item-str", {"object": "response", "output": ["x"]}),
        ("output-str", {"object": "response", "output": "oops"}),
        ("output-missing", {"object": "response"}),
    ]


def _auto_detect_malformed() -> list[tuple[str, Any]]:
    """Bodies routed through ``BaseEndpoint.try_extract_*`` (template / raw)."""
    bodies = [(f"choices-first-{e!r}", {"choices": [e]}) for e in _NON_DICT_ELEMENTS]
    bodies += [
        ("message-not-dict", {"choices": [{"message": "hello"}]}),
        ("delta-not-dict", {"choices": [{"delta": "hello"}]}),
        ("embeddings-data-none", {"data": [None]}),
        ("empty-dict", {}),
    ]
    return bodies


def _cases(cls, endpoint_type, prefix, bodies, template=None):
    return [
        param(cls, endpoint_type, template, body, id=f"{prefix}-{suffix}")
        for suffix, body in bodies
    ]


DEGRADE_TO_NONE_CASES = (
    _cases(CompletionsEndpoint, EndpointType.COMPLETIONS, "completions", _completions_malformed())
    + _cases(CohereRankingsEndpoint, EndpointType.COHERE_RANKINGS, "cohere", _cohere_malformed())
    + _cases(HuggingFaceGenerateEndpoint, EndpointType.HUGGINGFACE_GENERATE, "hf", _hf_malformed())
    + _cases(ResponsesEndpoint, EndpointType.RESPONSES, "responses", _responses_malformed())
    + _cases(RawEndpoint, EndpointType.RAW, "raw", _auto_detect_malformed())
    + _cases(
        TemplateEndpoint,
        EndpointType.TEMPLATE,
        "tmpl",
        _auto_detect_malformed(),
        template={"body": "{}", "response_field": "text"},
    )
)  # fmt: skip


@pytest.mark.parametrize("cls, endpoint_type, template, body", DEGRADE_TO_NONE_CASES)
def test_malformed_body_degrades_to_none(cls, endpoint_type, template, body):
    """A malformed 200 body degrades to ``None`` without raising."""
    endpoint = _make_endpoint(cls, endpoint_type, template)
    assert _parse_without_bug_crash(endpoint, body) is None


# ---------------------------------------------------------------------------
# chat_embeddings shares the auto-detect embeddings path; a non-dict item must
# not crash. Some auto-detect bodies carry no embeddings, so allow None here.
# ---------------------------------------------------------------------------

CHAT_EMBEDDINGS_MALFORMED = [
    param({"data": [None]}, id="ce-data-none"),
    param({"data": ["x"]}, id="ce-data-str"),
    param({"data": [5]}, id="ce-data-int"),
    param({"data": [{"object": "embedding", "embedding": [0.1]}, None]}, id="ce-mixed"),
    param({"data": "oops"}, id="ce-data-str-top"),
    param({}, id="ce-empty"),
]  # fmt: skip


@pytest.mark.parametrize("body", CHAT_EMBEDDINGS_MALFORMED)
def test_chat_embeddings_malformed_does_not_crash(body):
    endpoint = _make_endpoint(ChatEmbeddingsEndpoint, EndpointType.CHAT_EMBEDDINGS)
    # Must not raise; a mixed body still surfaces the one valid embedding.
    _parse_without_bug_crash(endpoint, body)


# ---------------------------------------------------------------------------
# Image endpoints skip non-dict ``data`` items (degrade to empty images).
# ---------------------------------------------------------------------------

IMAGE_ENDPOINTS = [
    param(ImageEditEndpoint, EndpointType.IMAGE_EDIT, id="image_edit"),
    param(ImageGenerationEndpoint, EndpointType.IMAGE_GENERATION, id="image_generation"),
]  # fmt: skip


@pytest.mark.parametrize("cls, endpoint_type", IMAGE_ENDPOINTS)
@pytest.mark.parametrize(
    "body",
    [
        param({"data": [None]}, id="data-none"),
        param({"data": ["x"]}, id="data-str"),
        param({"data": [5]}, id="data-int"),
        param({"data": [None, {"b64_json": "abc"}]}, id="data-mixed"),
    ],
)  # fmt: skip
def test_image_endpoints_skip_non_dict_items(cls, endpoint_type, body):
    """Malformed ``data`` items are skipped instead of crashing ``item.get(...)``."""
    endpoint = _make_endpoint(cls, endpoint_type)
    result = _parse_without_bug_crash(endpoint, body)
    assert result is not None
    assert isinstance(result.data, ImageResponseData)
    # The lone valid item (mixed case) survives; pure-malformed yields empty.
    expected = 1 if any(isinstance(i, dict) for i in body["data"]) else 0
    assert len(result.data.images) == expected


# ---------------------------------------------------------------------------
# Passthrough-safe endpoints (no element deref): malformed dict bodies must not
# crash. These are documented as already-guarded; the test pins that.
# ---------------------------------------------------------------------------

PASSTHROUGH_SAFE = [
    param(NIMRankingsEndpoint, EndpointType.NIM_RANKINGS, {"rankings": [None]}, id="nim_rankings"),
    param(HFTeiRankingsEndpoint, EndpointType.HF_TEI_RANKINGS, {"results": [None]}, id="hf_tei_rankings"),
    param(ImageRetrievalEndpoint, EndpointType.IMAGE_RETRIEVAL, {"data": [None]}, id="image_retrieval"),
    param(SolidoEndpoint, EndpointType.SOLIDO_RAG, {"content": None, "sources": [None]}, id="solido"),
    param(VideoGenerationEndpoint, EndpointType.VIDEO_GENERATION, {"status": [None]}, id="video"),
    param(EmbeddingsEndpoint, EndpointType.EMBEDDINGS, {"data": []}, id="embeddings-empty"),
]  # fmt: skip


@pytest.mark.parametrize("cls, endpoint_type, body", PASSTHROUGH_SAFE)
def test_passthrough_safe_endpoints_do_not_crash(cls, endpoint_type, body):
    endpoint = _make_endpoint(cls, endpoint_type)
    _parse_without_bug_crash(endpoint, body)


# ---------------------------------------------------------------------------
# No regression: a valid body still parses correctly for every fixed endpoint.
# ---------------------------------------------------------------------------


def _assert_text(result: ParsedResponse | None, expected: str) -> bool:
    assert result is not None
    assert isinstance(result.data, TextResponseData)
    assert result.data.text == expected
    return True


VALID_BODY_CASES = [
    param(
        CompletionsEndpoint,
        EndpointType.COMPLETIONS,
        None,
        {"object": "text_completion", "choices": [{"text": "hello"}]},
        lambda r: _assert_text(r, "hello"),
        id="completions-valid",
    ),
    param(
        HuggingFaceGenerateEndpoint,
        EndpointType.HUGGINGFACE_GENERATE,
        None,
        [{"generated_text": "hi there"}],
        lambda r: _assert_text(r, "hi there"),
        id="hf-valid-list",
    ),
    param(
        HuggingFaceGenerateEndpoint,
        EndpointType.HUGGINGFACE_GENERATE,
        None,
        {"generated_text": "hi dict"},
        lambda r: _assert_text(r, "hi dict"),
        id="hf-valid-dict",
    ),
    param(
        ResponsesEndpoint,
        EndpointType.RESPONSES,
        None,
        {"type": "response.output_text.delta", "delta": "streamed"},
        lambda r: _assert_text(r, "streamed"),
        id="responses-valid",
    ),
    param(
        CohereRankingsEndpoint,
        EndpointType.COHERE_RANKINGS,
        None,
        {"results": [{"index": 0, "relevance_score": 0.9}]},
        lambda r: (
            r is not None
            and isinstance(r.data, RankingsResponseData)
            and r.data.rankings == [{"index": 0, "score": 0.9}]
        ),
        id="cohere-valid",
    ),
    param(
        ChatEmbeddingsEndpoint,
        EndpointType.CHAT_EMBEDDINGS,
        None,
        {"data": [{"object": "embedding", "embedding": [0.1, 0.2]}]},
        lambda r: (
            r is not None
            and isinstance(r.data, EmbeddingResponseData)
            and r.data.embeddings == [[0.1, 0.2]]
        ),
        id="chat-embeddings-valid",
    ),
    param(
        RawEndpoint,
        EndpointType.RAW,
        None,
        {"choices": [{"message": {"content": "raw hi"}}]},
        lambda r: _assert_text(r, "raw hi"),
        id="raw-valid",
    ),
    param(
        TemplateEndpoint,
        EndpointType.TEMPLATE,
        {"body": "{}", "response_field": "text"},
        {"choices": [{"message": {"content": "tmpl hi"}}]},
        lambda r: _assert_text(r, "tmpl hi"),
        id="template-valid",
    ),
    param(
        ImageEditEndpoint,
        EndpointType.IMAGE_EDIT,
        None,
        {"data": [{"b64_json": "abc"}]},
        lambda r: (
            r is not None
            and isinstance(r.data, ImageResponseData)
            and len(r.data.images) == 1
        ),
        id="image-edit-valid",
    ),
]  # fmt: skip


@pytest.mark.parametrize("cls, endpoint_type, template, body, check", VALID_BODY_CASES)
def test_valid_body_still_parses(cls, endpoint_type, template, body, check):
    """No regression: a well-formed body still parses to the expected data."""
    endpoint = _make_endpoint(cls, endpoint_type, template)
    result = endpoint.parse_response(create_mock_response(123456789, body))
    assert check(result)


def test_base_endpoint_try_extract_text_survives_non_dict_choice():
    """Direct guard check on the shared ``BaseEndpoint.try_extract_text`` helper."""
    endpoint: BaseEndpoint = _make_endpoint(
        CompletionsEndpoint, EndpointType.COMPLETIONS
    )
    for choice in _NON_DICT_ELEMENTS:
        assert endpoint.try_extract_text({"choices": [choice]}) is None
    # A truthy non-dict message/delta must not crash the nested ``.get``.
    assert endpoint.try_extract_text({"choices": [{"message": "hi"}]}) is None
    assert endpoint.try_extract_text({"choices": [{"delta": "hi"}]}) is None
    # Happy path still extracts.
    ok = endpoint.try_extract_text({"choices": [{"text": "yes"}]})
    assert isinstance(ok, TextResponseData) and ok.text == "yes"
