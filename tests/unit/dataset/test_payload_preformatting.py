# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ``aiperf.dataset.payload_formatting``.

Covers:
- ``is_preformat_eligible`` eligibility contract (single-turn, FORK, context
  modes).
- ``format_conversation_payloads`` yields only eligible conversations and
  propagates ``NotImplementedError`` from the endpoint.
- ``preformat_payloads`` stamps raw_payload on eligible turns, emits logger
  messages on success and on ``NotImplementedError`` fallback.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aiperf.common.enums import (
    ConversationBranchMode,
    ConversationContextMode,
    ModelSelectionStrategy,
    RequestContentType,
)
from aiperf.common.models import (
    Conversation,
    ConversationBranchInfo,
    Turn,
)
from aiperf.common.models.model_endpoint_info import (
    EndpointInfo,
    ModelEndpointInfo,
    ModelInfo,
    ModelListInfo,
)
from aiperf.dataset.payload_formatting import (
    format_conversation_payloads,
    is_preformat_eligible,
    preformat_payloads,
)


def _make_model_endpoint(endpoint_type: str = "chat") -> ModelEndpointInfo:
    """Build a minimal ModelEndpointInfo. Plugin lookup is patched in tests
    so the ``endpoint_type`` value is only carried for plumbing."""
    return ModelEndpointInfo(
        models=ModelListInfo(
            models=[ModelInfo(name="test")],
            model_selection_strategy=ModelSelectionStrategy.ROUND_ROBIN,
        ),
        endpoint=EndpointInfo(type=endpoint_type, base_urls=["http://localhost:8000"]),
    )


def _branch(branch_id: str, mode: ConversationBranchMode) -> ConversationBranchInfo:
    return ConversationBranchInfo(
        branch_id=branch_id, mode=mode, child_conversation_ids=["child"]
    )


def _make_conversation(
    *,
    num_turns: int = 1,
    context_mode: ConversationContextMode | None = None,
    branches: list[ConversationBranchInfo] | None = None,
    session_id: str = "conv-1",
) -> Conversation:
    return Conversation(
        session_id=session_id,
        turns=[Turn() for _ in range(num_turns)],
        context_mode=context_mode,
        branches=branches or [],
    )


class TestIsPreformatEligible:
    @pytest.mark.parametrize(
        ("conversation", "expected"),
        [
            pytest.param(_make_conversation(num_turns=1), True, id="single_turn"),
            pytest.param(
                _make_conversation(
                    num_turns=1,
                    branches=[_branch("b1", ConversationBranchMode.FORK)],
                ),
                False,
                id="fork_branch",
            ),
            pytest.param(
                _make_conversation(
                    num_turns=3,
                    context_mode=ConversationContextMode.DELTAS_WITHOUT_RESPONSES,
                ),
                False,
                id="deltas_without_responses_multi_turn",
            ),
            pytest.param(
                _make_conversation(
                    num_turns=1,
                    context_mode=ConversationContextMode.DELTAS_WITHOUT_RESPONSES,
                ),
                True,
                id="deltas_without_responses_single_turn",
            ),
            pytest.param(
                _make_conversation(num_turns=3, context_mode=None),
                False,
                id="none_context_mode_multi_turn",
            ),
            pytest.param(
                _make_conversation(
                    num_turns=3,
                    context_mode=ConversationContextMode.MESSAGE_ARRAY_WITH_RESPONSES,
                ),
                True,
                id="message_array_with_responses_multi_turn",
            ),
            pytest.param(
                _make_conversation(
                    num_turns=3,
                    context_mode=ConversationContextMode.DELTAS_WITH_RESPONSES,
                ),
                False,
                id="deltas_with_responses_multi_turn",
            ),
            pytest.param(
                _make_conversation(
                    num_turns=1,
                    branches=[_branch("b1", ConversationBranchMode.SPAWN)],
                ),
                True,
                id="spawn_branch",
            ),
        ],
    )
    def test_eligibility_cases(
        self, conversation: Conversation, expected: bool
    ) -> None:
        assert is_preformat_eligible(conversation) is expected

    def test_fork_child_rejected(self):
        """A non-root child with parent_conversation_id is seeded from the
        parent's live turn_list at dispatch — ineligible."""
        conv = Conversation(
            session_id="child",
            turns=[Turn()],
            is_root=False,
            parent_conversation_id="parent",
        )
        assert is_preformat_eligible(conv) is False


class TestFormatConversationPayloads:
    def _setup_endpoint(
        self, format_payload_side_effect: Any
    ) -> tuple[MagicMock, MagicMock]:
        """Patch the endpoint plugin and return (endpoint_instance, plugins_mock)."""
        endpoint_instance = MagicMock()
        endpoint_instance.format_payload.side_effect = format_payload_side_effect

        EndpointClass = MagicMock(return_value=endpoint_instance)
        return endpoint_instance, EndpointClass

    def test_skips_ineligible_conversations(self):
        """Mixed eligible+ineligible batch: only eligible turns yielded."""
        eligible = _make_conversation(
            num_turns=2,
            session_id="ok",
            context_mode=ConversationContextMode.MESSAGE_ARRAY_WITH_RESPONSES,
        )
        ineligible_mode = _make_conversation(
            num_turns=3,
            session_id="skip",
            context_mode=ConversationContextMode.DELTAS_WITHOUT_RESPONSES,
        )
        ineligible_fork = _make_conversation(
            num_turns=1,
            session_id="forked",
            branches=[_branch("b1", ConversationBranchMode.FORK)],
        )

        _endpoint_instance, EndpointClass = self._setup_endpoint(
            lambda req: {"session": req.conversation_id, "turn": req.turn_index}
        )
        model_endpoint = _make_model_endpoint("chat")

        with patch(
            "aiperf.dataset.payload_formatting.plugins.get_class",
            return_value=EndpointClass,
        ):
            yielded = list(
                format_conversation_payloads(
                    [eligible, ineligible_mode, ineligible_fork], model_endpoint
                )
            )

        assert [(sid, idx) for sid, idx, _ in yielded] == [
            ("ok", 0),
            ("ok", 1),
        ]

    def test_propagates_endpoint_notimplementederror(self):
        """Iterator surfaces NotImplementedError from format_payload."""
        conv = _make_conversation(num_turns=1, session_id="raw")

        def _raise(_req):
            raise NotImplementedError("no raw_payload set")

        _, EndpointClass = self._setup_endpoint(_raise)
        model_endpoint = _make_model_endpoint("raw")

        with (
            patch(
                "aiperf.dataset.payload_formatting.plugins.get_class",
                return_value=EndpointClass,
            ),
            pytest.raises(NotImplementedError),
        ):
            list(format_conversation_payloads([conv], model_endpoint))


class _RecordingLogger:
    """Captures ``logger.info(...)`` calls for assertion."""

    def __init__(self) -> None:
        self.messages: list[str] = []

    def info(self, msg: Any) -> None:
        self.messages.append(msg() if callable(msg) else str(msg))


class TestPreformatPayloads:
    def _setup_endpoint(
        self, format_payload_side_effect: Any
    ) -> tuple[MagicMock, MagicMock]:
        endpoint_instance = MagicMock()
        endpoint_instance.format_payload.side_effect = format_payload_side_effect
        EndpointClass = MagicMock(return_value=endpoint_instance)
        return endpoint_instance, EndpointClass

    def test_stamps_raw_payload_on_eligible_turns(self):
        """Every eligible turn gets ``raw_payload`` set to the formatter result."""
        conv_a = _make_conversation(num_turns=2, session_id="a")
        conv_b = _make_conversation(num_turns=1, session_id="b")
        # Make conv_a multi-turn-eligible by overriding context_mode
        conv_a.context_mode = ConversationContextMode.MESSAGE_ARRAY_WITH_RESPONSES

        _, EndpointClass = self._setup_endpoint(
            lambda req: {"session": req.conversation_id, "turn": req.turn_index}
        )
        logger = _RecordingLogger()
        model_endpoint = _make_model_endpoint("chat")

        with patch(
            "aiperf.dataset.payload_formatting.plugins.get_class",
            return_value=EndpointClass,
        ):
            preformat_payloads([conv_a, conv_b], model_endpoint, logger=logger)

        assert conv_a.turns[0].raw_payload == {"session": "a", "turn": 0}
        assert conv_a.turns[1].raw_payload == {"session": "a", "turn": 1}
        assert conv_b.turns[0].raw_payload == {"session": "b", "turn": 0}

    def test_deltas_with_responses_multi_turn_is_not_stamped_turn_local(self):
        """Prefix-accumulating multi-turn conversations must stay dispatch-formatted."""
        conv = _make_conversation(
            num_turns=2,
            context_mode=ConversationContextMode.DELTAS_WITH_RESPONSES,
            session_id="deltas-with-responses",
        )
        format_payload = MagicMock(return_value={"turn-local": True})
        endpoint_instance = MagicMock()
        endpoint_instance.format_payload.side_effect = format_payload
        EndpointClass = MagicMock(return_value=endpoint_instance)
        logger = _RecordingLogger()
        model_endpoint = _make_model_endpoint("chat")

        with patch(
            "aiperf.dataset.payload_formatting.plugins.get_class",
            return_value=EndpointClass,
        ):
            preformat_payloads([conv], model_endpoint, logger=logger)

        assert [turn.raw_payload for turn in conv.turns] == [None, None]
        format_payload.assert_not_called()

    def test_logger_info_called_on_success(self):
        """The success-path info message fires when at least one turn is stamped."""
        conv = _make_conversation(num_turns=1, session_id="ok")
        _, EndpointClass = self._setup_endpoint(lambda req: {"x": 1})
        logger = _RecordingLogger()
        model_endpoint = _make_model_endpoint("chat")

        with patch(
            "aiperf.dataset.payload_formatting.plugins.get_class",
            return_value=EndpointClass,
        ):
            preformat_payloads([conv], model_endpoint, logger=logger)

        assert any("Pre-formatted 1 payloads" in msg for msg in logger.messages), (
            f"Expected success info in {logger.messages!r}"
        )

    def test_skips_multipart_endpoints(self):
        """Multipart endpoints need dict payloads at dispatch so FormData can be built."""
        conv = _make_conversation(num_turns=1, session_id="multipart")
        format_payload = MagicMock(return_value={"prompt": "edit"})
        endpoint_instance = MagicMock()
        endpoint_instance.format_payload.side_effect = format_payload
        EndpointClass = MagicMock(return_value=endpoint_instance)
        logger = _RecordingLogger()
        model_endpoint = _make_model_endpoint("image_edit")
        model_endpoint.endpoint.request_content_type = (
            RequestContentType.MULTIPART_FORM_DATA
        )

        with patch(
            "aiperf.dataset.payload_formatting.plugins.get_class",
            return_value=EndpointClass,
        ):
            preformat_payloads([conv], model_endpoint, logger=logger)

        assert conv.turns[0].raw_payload is None
        format_payload.assert_not_called()
        assert any("multipart/form-data" in msg for msg in logger.messages), (
            f"Expected multipart skip info in {logger.messages!r}"
        )

    def test_multipart_skip_does_not_construct_endpoint_for_mixed_conversations(self):
        eligible = _make_conversation(num_turns=1, session_id="eligible")
        eligible_with_existing_raw = _make_conversation(num_turns=1, session_id="raw")
        eligible_with_existing_raw.turns[0].raw_payload = {"prompt": "preserve"}
        ineligible = _make_conversation(
            num_turns=2,
            context_mode=ConversationContextMode.DELTAS_WITHOUT_RESPONSES,
            session_id="ineligible",
        )
        EndpointClass = MagicMock()
        logger = _RecordingLogger()
        model_endpoint = _make_model_endpoint("image_edit")
        model_endpoint.endpoint.request_content_type = (
            RequestContentType.MULTIPART_FORM_DATA
        )

        with patch(
            "aiperf.dataset.payload_formatting.plugins.get_class",
            return_value=EndpointClass,
        ):
            preformat_payloads(
                [eligible, eligible_with_existing_raw, ineligible],
                model_endpoint,
                logger=logger,
            )

        assert eligible.turns[0].raw_payload is None
        assert eligible_with_existing_raw.turns[0].raw_payload == {"prompt": "preserve"}
        assert all(turn.raw_payload is None for turn in ineligible.turns)
        EndpointClass.assert_not_called()

    def test_logger_info_called_on_notimplementederror(self):
        """The fallback info message fires when format_payload raises."""
        conv = _make_conversation(num_turns=1, session_id="raw")

        def _raise(_req):
            raise NotImplementedError("no raw_payload")

        _, EndpointClass = self._setup_endpoint(_raise)
        logger = _RecordingLogger()
        model_endpoint = _make_model_endpoint("raw")

        with patch(
            "aiperf.dataset.payload_formatting.plugins.get_class",
            return_value=EndpointClass,
        ):
            preformat_payloads([conv], model_endpoint, logger=logger)

        assert conv.turns[0].raw_payload is None
        assert any(
            "does not support format_payload" in msg for msg in logger.messages
        ), f"Expected fallback info in {logger.messages!r}"

    def test_skips_turns_already_carrying_raw_payload(self):
        """Pre-existing raw_payload (raw_payload/inputs_json loaders) is preserved."""
        conv = _make_conversation(num_turns=1, session_id="pre")
        conv.turns[0].raw_payload = {"preserved": True}

        format_payload = MagicMock(return_value={"replaced": True})
        endpoint_instance = MagicMock()
        endpoint_instance.format_payload.side_effect = format_payload
        EndpointClass = MagicMock(return_value=endpoint_instance)
        logger = _RecordingLogger()
        model_endpoint = _make_model_endpoint("chat")

        with patch(
            "aiperf.dataset.payload_formatting.plugins.get_class",
            return_value=EndpointClass,
        ):
            preformat_payloads([conv], model_endpoint, logger=logger)

        assert conv.turns[0].raw_payload == {"preserved": True}
        format_payload.assert_not_called()
