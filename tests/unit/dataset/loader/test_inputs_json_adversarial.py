# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial coverage for InputsJsonPayloadLoader.

Pins current behavior for edge cases in `can_load` and `load_dataset`,
and directly asserts the Wave 2 fixes (duplicate session_id rejection,
ValueError on missing keys, non-object entry rejection) via
`pytest.raises`.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import orjson
import pytest
from pydantic import ValidationError
from pytest import param

from aiperf.dataset.loader.inputs_json import InputsJsonPayloadLoader
from aiperf.dataset.loader.models import InputsJsonSession


def _make_loader(filename):
    loader = InputsJsonPayloadLoader.__new__(InputsJsonPayloadLoader)
    loader.filename = str(filename)
    loader.info = MagicMock()
    loader.debug = MagicMock()
    return loader


class TestCanLoadAdversarial:
    @pytest.mark.parametrize(
        "bad_data",
        [param([]), param("s"), param(123)],
    )  # fmt: skip
    def test_can_load_non_dict_data_returns_false(self, bad_data):
        """`can_load` with non-dict `data` must return False, not raise."""
        assert InputsJsonPayloadLoader.can_load(data=bad_data) is False

    def test_can_load_dict_without_data_key_returns_false(self):
        """Dict without a top-level ``data`` key must return False."""
        assert InputsJsonPayloadLoader.can_load(data={"not_data": []}) is False

    def test_can_load_data_not_a_list_returns_false(self):
        """Top-level ``data`` value that isn't a list must return False."""
        assert InputsJsonPayloadLoader.can_load(data={"data": "str"}) is False

    def test_can_load_file_with_non_json_extension_returns_false(self, tmp_path):
        """Files with a non-``.json`` suffix must return False even if content is valid JSON."""
        path = tmp_path / "inputs.txt"
        path.write_bytes(
            orjson.dumps({"data": [{"session_id": "s", "payloads": [{"model": "m"}]}]})
        )
        assert InputsJsonPayloadLoader.can_load(filename=path) is False

    def test_can_load_zero_byte_file_returns_false(self, tmp_path):
        """Empty files must return False (orjson raises, caught)."""
        path = tmp_path / "empty.json"
        path.write_bytes(b"")
        assert InputsJsonPayloadLoader.can_load(filename=path) is False

    def test_can_load_file_with_json_array_top_level_returns_false(self, tmp_path):
        """Files whose root JSON value is an array must return False."""
        path = tmp_path / "array.json"
        path.write_bytes(orjson.dumps([{"session_id": "x", "payloads": [{"m": 1}]}]))
        assert InputsJsonPayloadLoader.can_load(filename=path) is False


class TestLoadDatasetAdversarial:
    def test_load_dataset_entry_with_empty_payloads_list_rejected_by_pydantic(
        self, tmp_path
    ):
        """``InputsJsonSession`` has ``min_length=1`` on payloads; empty list raises ValidationError."""
        path = tmp_path / "inputs.json"
        path.write_bytes(orjson.dumps({"data": [{"session_id": "s", "payloads": []}]}))
        loader = _make_loader(path)
        with pytest.raises(ValidationError):
            loader.load_dataset()

    def test_convert_to_conversations_session_id_passthrough(self, tmp_path):
        """The ``session_id`` from the file must appear verbatim on the resulting Conversation."""
        data = {
            "data": [
                {
                    "session_id": "custom-id",
                    "payloads": [{"model": "m", "messages": []}],
                }
            ]
        }
        path = tmp_path / "inputs.json"
        path.write_bytes(orjson.dumps(data))
        loader = _make_loader(path)
        conversations = loader.convert_to_conversations(loader.load_dataset())
        assert len(conversations) == 1
        assert conversations[0].session_id == "custom-id"

    def test_convert_to_conversations_emits_one_turn_per_payload(self, tmp_path):
        """Three payloads must produce three Turns, each with ``raw_payload`` set."""
        payloads = [
            {"model": "m", "turn": 1},
            {"model": "m", "turn": 2},
            {"model": "m", "turn": 3},
        ]
        data = {"data": [{"session_id": "s", "payloads": payloads}]}
        path = tmp_path / "inputs.json"
        path.write_bytes(orjson.dumps(data))
        loader = _make_loader(path)
        conversations = loader.convert_to_conversations(loader.load_dataset())
        assert len(conversations) == 1
        turns = conversations[0].turns
        assert len(turns) == 3
        for turn, expected in zip(turns, payloads, strict=True):
            assert turn.raw_payload == expected


class TestWave2FixForwardCompatibility:
    """Tests that pin the post-fix behavior after Wave 2 bug fixes landed."""

    def test_load_dataset_duplicate_session_id_rejected_post_fix(self, tmp_path):
        data = {
            "data": [
                {"session_id": "dup", "payloads": [{"model": "m1"}]},
                {"session_id": "dup", "payloads": [{"model": "m2"}]},
            ]
        }
        path = tmp_path / "inputs.json"
        path.write_bytes(orjson.dumps(data))
        loader = _make_loader(path)
        with pytest.raises(ValueError, match="duplicate"):
            loader.load_dataset()

    def test_load_dataset_missing_required_key_raises_value_error_post_fix(
        self, tmp_path
    ):
        path = tmp_path / "inputs.json"
        path.write_bytes(orjson.dumps({"data": [{"payloads": [{"x": 1}]}]}))
        loader = _make_loader(path)
        with pytest.raises(ValueError, match="session_id"):
            loader.load_dataset()

    @pytest.mark.parametrize(
        "bad_entry",
        [
            param(None, id="null"),
            param(42, id="number"),
            param([{"session_id": "s"}], id="list"),
            param("session", id="string"),
        ],
    )  # fmt: skip
    def test_load_dataset_non_object_entry_raises_value_error(
        self, tmp_path, bad_entry
    ):
        path = tmp_path / "inputs.json"
        path.write_bytes(orjson.dumps({"data": [bad_entry]}))
        loader = _make_loader(path)
        with pytest.raises(ValueError, match="must be an object"):
            loader.load_dataset()


def test_inputs_json_session_model_rejects_empty_payloads_directly():
    """Sanity check that the Pydantic constraint is on the model, not the loader."""
    with pytest.raises(ValidationError):
        InputsJsonSession(session_id="s", payloads=[])
