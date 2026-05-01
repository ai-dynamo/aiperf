# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

import orjson
import pytest
from pydantic import ValidationError

from aiperf.common.enums import ConversationContextMode
from aiperf.dataset.loader.inputs_json import InputsJsonPayloadLoader
from aiperf.dataset.loader.models import InputsJsonSession


@pytest.fixture
def inputs_json_data():
    return {
        "data": [
            {
                "session_id": "sess-1",
                "payloads": [
                    {"messages": [{"role": "user", "content": "Hello"}], "model": "m1"},
                    {
                        "messages": [{"role": "user", "content": "Follow up"}],
                        "model": "m1",
                    },
                ],
            },
            {
                "session_id": "sess-2",
                "payloads": [
                    {
                        "messages": [{"role": "user", "content": "Question"}],
                        "model": "m2",
                    },
                ],
            },
        ]
    }


@pytest.fixture
def inputs_json_file(tmp_path, inputs_json_data):
    path = tmp_path / "inputs.json"
    path.write_bytes(orjson.dumps(inputs_json_data))
    return path


class TestCanLoad:
    def test_accepts_inputs_json_data(self, inputs_json_data):
        assert InputsJsonPayloadLoader.can_load(data=inputs_json_data) is True

    def test_rejects_empty_data_list(self):
        assert InputsJsonPayloadLoader.can_load(data={"data": []}) is False

    def test_rejects_data_without_payloads(self):
        assert (
            InputsJsonPayloadLoader.can_load(data={"data": [{"session_id": "x"}]})
            is False
        )

    def test_rejects_non_dict(self):
        assert InputsJsonPayloadLoader.can_load(data={"messages": []}) is False

    @pytest.mark.parametrize("data", [[1, 2, 3], "not a dict", 42])
    def test_non_dict_data_returns_false(self, data):
        """Non-dict data must return False, not raise AttributeError."""
        assert InputsJsonPayloadLoader.can_load(data=data) is False

    def test_file_containing_json_array_returns_false(self, tmp_path):
        """File whose root JSON value is an array must return False."""
        path = tmp_path / "array.json"
        path.write_bytes(orjson.dumps([1, 2, 3]))
        assert InputsJsonPayloadLoader.can_load(filename=path) is False

    def test_accepts_file(self, inputs_json_file):
        assert InputsJsonPayloadLoader.can_load(filename=inputs_json_file) is True

    def test_rejects_non_json_file(self, tmp_path):
        path = tmp_path / "data.txt"
        path.write_text("not json")
        assert InputsJsonPayloadLoader.can_load(filename=path) is False

    def test_returns_false_for_none(self):
        assert InputsJsonPayloadLoader.can_load() is False


class TestLoadDataset:
    def _make_loader(self, filename):
        loader = InputsJsonPayloadLoader.__new__(InputsJsonPayloadLoader)
        loader.filename = str(filename)
        loader.info = MagicMock()
        loader.debug = MagicMock()
        return loader

    def test_load_dataset(self, inputs_json_file):
        loader = self._make_loader(inputs_json_file)
        data = loader.load_dataset()
        assert len(data) == 2
        assert len(data["sess-1"][0].payloads) == 2
        assert len(data["sess-2"][0].payloads) == 1

    def test_convert_to_conversations(self, inputs_json_file):
        loader = self._make_loader(inputs_json_file)
        data = loader.load_dataset()
        conversations = loader.convert_to_conversations(data)
        assert len(conversations) == 2

        sess1_conv = next(c for c in conversations if c.session_id == "sess-1")
        assert len(sess1_conv.turns) == 2
        assert sess1_conv.turns[0].raw_payload is not None
        assert sess1_conv.turns[0].raw_payload["model"] == "m1"
        assert sess1_conv.turns[0].role == "user"

        sess2_conv = next(c for c in conversations if c.session_id == "sess-2")
        assert len(sess2_conv.turns) == 1

    def test_conversations_have_message_array_with_responses_context_mode(
        self, inputs_json_file
    ):
        loader = self._make_loader(inputs_json_file)
        data = loader.load_dataset()
        conversations = loader.convert_to_conversations(data)
        for conv in conversations:
            assert (
                conv.context_mode
                == ConversationContextMode.MESSAGE_ARRAY_WITH_RESPONSES
            )


class TestContextMode:
    def test_default_context_mode_is_message_array_with_responses(self):
        assert (
            InputsJsonPayloadLoader.get_default_context_mode()
            == ConversationContextMode.MESSAGE_ARRAY_WITH_RESPONSES
        )


class TestInputsJsonSessionValidation:
    def test_empty_payloads_raises_validation_error(self):
        """Sessions with zero payloads must be rejected at the schema boundary."""
        with pytest.raises(ValidationError):
            InputsJsonSession(session_id="test", payloads=[])

    def test_non_empty_payloads_accepted(self):
        session = InputsJsonSession(
            session_id="test", payloads=[{"model": "m", "messages": []}]
        )
        assert len(session.payloads) == 1
