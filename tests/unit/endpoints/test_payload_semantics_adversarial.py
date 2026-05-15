# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial tests pinning the dispatch-turn vs inheritance contract.

Contract:
- Per-request fields (``extra_body``, ``max_tokens``, ``model``) are read
  from ``turns[-1]`` only; parent turns never leak into a child payload.
- Only ``raw_tools`` walks turn history (latest-non-None) so a FORK child
  that doesn't redeclare tools inherits the parent's tool config.
- Authored datasets use ``extra``; ``extra_body`` at the row level is
  rejected.
"""

from __future__ import annotations

import base64

import pytest

from aiperf.common.models import Image, Text, Turn
from aiperf.dataset.loader.dag_jsonl_models import DagTurn
from aiperf.dataset.loader.models import MooncakeTrace, SingleTurn
from aiperf.endpoints.openai_chat import ChatEndpoint
from aiperf.endpoints.openai_image_edit import ImageEditEndpoint
from aiperf.endpoints.openai_image_generation import ImageGenerationEndpoint
from aiperf.endpoints.openai_responses import ResponsesEndpoint
from aiperf.endpoints.openai_video_generation import VideoGenerationEndpoint
from aiperf.endpoints.template_endpoint import TemplateEndpoint
from aiperf.plugin.enums import EndpointType
from tests.unit.endpoints.conftest import (
    create_endpoint_with_mock_transport,
    create_model_endpoint,
    create_request_info,
)

# Tiny 1x1 PNG that survives the image-edit MIME sniffer.
_PNG_BYTES = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08"
    b"\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\rIDATx\x9cc\xfc\xcf\xc0\x00"
    b"\x00\x00\x03\x00\x01\x9a\xa3\x9eS\x00\x00\x00\x00IEND\xaeB`\x82"
)
_PNG_DATA_URL = f"data:image/png;base64,{base64.b64encode(_PNG_BYTES).decode()}"


# ---------------------------------------------------------------------------
# Multi-turn formatters: parent fields never leak into the child payload
# ---------------------------------------------------------------------------


class TestChatDispatchTurnScoping:
    @pytest.fixture
    def endpoint(self):
        me = create_model_endpoint(EndpointType.CHAT)
        return create_endpoint_with_mock_transport(ChatEndpoint, me)

    def _format(self, endpoint, parent_kwargs, child_kwargs):
        parent = Turn(role="user", texts=[Text(contents=["parent"])], **parent_kwargs)
        child = Turn(role="user", texts=[Text(contents=["child"])], **child_kwargs)
        request_info = create_request_info(
            model_endpoint=endpoint.model_endpoint, turns=[parent, child]
        )
        return endpoint.format_payload(request_info)

    def test_parent_extra_body_does_not_leak(self, endpoint):
        payload = self._format(
            endpoint,
            parent_kwargs={"extra_body": {"vendor": "parent"}},
            child_kwargs={},
        )
        assert "vendor" not in payload

    def test_parent_max_tokens_does_not_leak(self, endpoint):
        payload = self._format(
            endpoint,
            parent_kwargs={"max_tokens": 256},
            child_kwargs={},
        )
        assert "max_completion_tokens" not in payload
        assert "max_tokens" not in payload

    def test_parent_model_does_not_leak(self, endpoint):
        payload = self._format(
            endpoint,
            parent_kwargs={"model": "parent-model"},
            child_kwargs={},
        )
        assert payload["model"] == endpoint.model_endpoint.primary_model_name

    def test_empty_child_extra_body_does_not_pull_parent(self, endpoint):
        """Falsy empty dict on child still blocks parent extra_body leak."""
        payload = self._format(
            endpoint,
            parent_kwargs={"extra_body": {"vendor": "parent"}},
            child_kwargs={"extra_body": {}},
        )
        assert "vendor" not in payload

    def test_raw_tools_still_inherits_from_parent(self, endpoint):
        tools = [{"type": "function", "function": {"name": "lookup"}}]
        payload = self._format(
            endpoint,
            parent_kwargs={"raw_tools": tools},
            child_kwargs={},
        )
        assert payload["tools"] == tools

    def test_raw_tools_intermediate_none_is_transparent(self, endpoint):
        parent_tools = [{"type": "function", "function": {"name": "parent_fn"}}]
        mid_tools = [{"type": "function", "function": {"name": "mid_fn"}}]
        parent = Turn(
            role="user", texts=[Text(contents=["root"])], raw_tools=parent_tools
        )
        mid = Turn(role="user", texts=[Text(contents=["middle"])], raw_tools=mid_tools)
        leaf = Turn(role="user", texts=[Text(contents=["child"])], raw_tools=None)
        request_info = create_request_info(
            model_endpoint=endpoint.model_endpoint, turns=[parent, mid, leaf]
        )
        payload = endpoint.format_payload(request_info)
        assert payload["tools"] == mid_tools


class TestResponsesDispatchTurnScoping:
    @pytest.fixture
    def endpoint(self):
        me = create_model_endpoint(EndpointType.RESPONSES, streaming=True)
        return create_endpoint_with_mock_transport(ResponsesEndpoint, me)

    def _format(self, endpoint, parent_kwargs, child_kwargs):
        parent = Turn(role="user", texts=[Text(contents=["parent"])], **parent_kwargs)
        child = Turn(role="user", texts=[Text(contents=["child"])], **child_kwargs)
        return endpoint.format_payload(
            create_request_info(
                model_endpoint=endpoint.model_endpoint, turns=[parent, child]
            )
        )

    def test_parent_extra_body_does_not_leak(self, endpoint):
        payload = self._format(
            endpoint,
            parent_kwargs={"extra_body": {"vendor": "parent"}},
            child_kwargs={},
        )
        assert "vendor" not in payload

    def test_parent_max_tokens_does_not_leak(self, endpoint):
        payload = self._format(
            endpoint,
            parent_kwargs={"max_tokens": 256},
            child_kwargs={},
        )
        assert "max_output_tokens" not in payload

    def test_parent_model_does_not_leak(self, endpoint):
        payload = self._format(
            endpoint,
            parent_kwargs={"model": "parent-model"},
            child_kwargs={},
        )
        assert payload["model"] == endpoint.model_endpoint.primary_model_name

    def test_raw_tools_still_inherits_from_parent(self, endpoint):
        tools = [{"type": "function", "name": "lookup"}]
        payload = self._format(
            endpoint,
            parent_kwargs={"raw_tools": tools},
            child_kwargs={},
        )
        assert payload["tools"] == tools


class TestSingleTurnStyleFormattersUseDispatchingTurn:
    """For formatters that accept ``len(turns) >= 1`` but format only one turn,
    confirm they read from ``turns[-1]`` and never accidentally surface
    parent prompts/models/extras."""

    @pytest.fixture
    def image_gen(self):
        me = create_model_endpoint(EndpointType.IMAGE_GENERATION)
        return create_endpoint_with_mock_transport(ImageGenerationEndpoint, me)

    @pytest.fixture
    def video_gen(self):
        me = create_model_endpoint(EndpointType.VIDEO_GENERATION)
        return create_endpoint_with_mock_transport(VideoGenerationEndpoint, me)

    @pytest.fixture
    def image_edit(self):
        me = create_model_endpoint(EndpointType.IMAGE_EDIT)
        return create_endpoint_with_mock_transport(ImageEditEndpoint, me)

    @pytest.fixture
    def template(self):
        me = create_model_endpoint(
            EndpointType.TEMPLATE,
            extra=[
                (
                    "payload_template",
                    '{"text": {{ text|tojson }}, "model": {{ model|tojson }}}',
                )
            ],
        )
        return create_endpoint_with_mock_transport(TemplateEndpoint, me), me

    def test_image_generation_ignores_parent_turn(self, image_gen):
        parent = Turn(texts=[Text(contents=["parent prompt"])], model="parent-model")
        child = Turn(
            texts=[Text(contents=["child prompt"])],
            model="child-model",
            extra_body={"vendor": "child"},
        )
        payload = image_gen.format_payload(
            create_request_info(
                model_endpoint=image_gen.model_endpoint, turns=[parent, child]
            )
        )
        assert payload["prompt"] == "child prompt"
        assert payload["model"] == "child-model"
        assert payload["vendor"] == "child"

    def test_image_generation_parent_extra_body_does_not_leak(self, image_gen):
        parent = Turn(
            texts=[Text(contents=["parent prompt"])],
            extra_body={"vendor": "parent"},
        )
        child = Turn(texts=[Text(contents=["child prompt"])])
        payload = image_gen.format_payload(
            create_request_info(
                model_endpoint=image_gen.model_endpoint, turns=[parent, child]
            )
        )
        assert "vendor" not in payload

    def test_video_generation_ignores_parent_turn(self, video_gen):
        parent = Turn(texts=[Text(contents=["parent video"])], model="parent-model")
        child = Turn(
            texts=[Text(contents=["child video"])],
            model="child-model",
            extra_body={"vendor_fps": 24},
        )
        payload = video_gen.format_payload(
            create_request_info(
                model_endpoint=video_gen.model_endpoint, turns=[parent, child]
            )
        )
        assert payload["prompt"] == "child video"
        assert payload["model"] == "child-model"
        assert payload["vendor_fps"] == 24

    def test_image_edit_ignores_parent_turn(self, image_edit):
        parent = Turn(
            texts=[Text(contents=["parent edit"])],
            images=[Image(contents=[_PNG_DATA_URL])],
            model="parent-model",
            extra_body={"seed": 1},
        )
        child = Turn(
            texts=[Text(contents=["child edit"])],
            images=[Image(contents=["https://example.com/child.png"])],
            model="child-model",
            extra_body={"seed": 2},
        )
        payload = image_edit.format_payload(
            create_request_info(
                model_endpoint=image_edit.model_endpoint, turns=[parent, child]
            )
        )
        assert payload["prompt"] == "child edit"
        assert payload["model"] == "child-model"
        assert payload["url"] == "https://example.com/child.png"
        assert "image" not in payload
        assert payload["seed"] == 2

    def test_image_edit_extra_body_cannot_overwrite_reserved(self, image_edit):
        turn = Turn(
            texts=[Text(contents=["legit"])],
            images=[Image(contents=[_PNG_DATA_URL])],
            extra_body={"prompt": "HIJACKED", "size": "512x512"},
        )
        payload = image_edit.format_payload(
            create_request_info(model_endpoint=image_edit.model_endpoint, turns=[turn])
        )
        assert payload["prompt"] == "legit"
        assert payload["size"] == "512x512"

    def test_template_ignores_parent_turn(self, template):
        endpoint, me = template
        parent = Turn(texts=[Text(contents=["parent text"])], model="parent-model")
        child = Turn(texts=[Text(contents=["child text"])], model="child-model")
        payload = endpoint.format_payload(
            create_request_info(model_endpoint=me, turns=[parent, child])
        )
        assert payload == {"text": "child text", "model": "child-model"}


# ---------------------------------------------------------------------------
# Dataset schemas: authored extra propagates; extra_body is silently ignored
# on allow-extra schemas but rejected on extra="forbid" schemas (DagTurn).
# ---------------------------------------------------------------------------


class TestDatasetExtraSchemaContract:
    def test_dag_turn_rejects_extra_body(self):
        """DagTurn uses ``extra='forbid'``; typo'd extra_body is rejected."""
        with pytest.raises(ValueError):
            DagTurn(
                messages=[{"role": "user", "content": "hi"}],
                extra_body={"vendor": 1},
            )

    def test_dag_turn_accepts_extra(self):
        t = DagTurn(
            messages=[{"role": "user", "content": "hi"}],
            extra={"vendor": 1},
        )
        assert t.extra == {"vendor": 1}

    def test_single_turn_extra_round_trips(self):
        row = SingleTurn(text="hi", extra={"vendor": 1})
        restored = SingleTurn.model_validate_json(row.model_dump_json())
        assert restored.extra == {"vendor": 1}

    def test_mooncake_extra_round_trips(self):
        row = MooncakeTrace(text_input="hi", extra={"vendor": 1})
        restored = MooncakeTrace.model_validate_json(row.model_dump_json())
        assert restored.extra == {"vendor": 1}
