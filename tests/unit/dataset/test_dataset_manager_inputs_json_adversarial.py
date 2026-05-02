# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial unit tests for DatasetManager inputs.json / payload handling.

Targets three known-fragile surfaces in ``dataset_manager.py``:

- ``_profile_configure_command`` skip-logic for pre-built payloads.
- ``_generate_input_payloads`` raw-vs-formatted branch (mixed state).
- ``_preformat_payloads`` all-or-nothing gating plus NotImplementedError escape.
- ``_generate_inputs_json_file`` OSError swallow, re-raise, ``.tmp`` cleanup.

The Wave-2 fix targets now pass: MOONCAKE_TRACE payload mode is skipped for
inputs.json generation, and mixed raw_payload / non-raw turns in a single
conversation raise ``ValueError`` instead of silently dropping.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest

from aiperf.common.config import (
    EndpointConfig,
    InputConfig,
    OutputConfig,
    ServiceConfig,
    UserConfig,
)
from aiperf.common.enums import (
    CacheBustTarget,
    ConversationContextMode,
    ModelSelectionStrategy,
)
from aiperf.common.models import Conversation, Turn
from aiperf.common.models.dataset_models import Text
from aiperf.common.models.model_endpoint_info import (
    EndpointInfo,
    ModelEndpointInfo,
    ModelInfo,
    ModelListInfo,
)
from aiperf.dataset.dataset_manager import DatasetManager
from aiperf.plugin.enums import CustomDatasetType, EndpointType

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _raw() -> dict[str, Any]:
    return {"model": "m", "messages": [{"role": "user", "content": "hi"}]}


def _chat_endpoint() -> ModelEndpointInfo:
    return ModelEndpointInfo(
        models=ModelListInfo(
            models=[ModelInfo(name="test")],
            model_selection_strategy=ModelSelectionStrategy.ROUND_ROBIN,
        ),
        endpoint=EndpointInfo(type=EndpointType.CHAT, base_url="http://localhost"),
    )


def _raw_endpoint() -> ModelEndpointInfo:
    return ModelEndpointInfo(
        models=ModelListInfo(
            models=[ModelInfo(name="test")],
            model_selection_strategy=ModelSelectionStrategy.ROUND_ROBIN,
        ),
        endpoint=EndpointInfo(type=EndpointType.RAW, base_url="http://localhost"),
    )


def _stub_manager(dataset: dict[str, Conversation]) -> DatasetManager:
    """Cheap DatasetManager stub for methods that only touch ``self.dataset``."""
    mgr = object.__new__(DatasetManager)
    mgr.dataset = dataset
    return mgr


def _full_manager(
    tmp_path: Path,
    custom_dataset_type: str | None = None,
    endpoint_type: str = EndpointType.CHAT,
) -> DatasetManager:
    """Construct a real DatasetManager instance via the public constructor."""
    input_file = None
    if custom_dataset_type is not None:
        input_file = tmp_path / "fake_input.jsonl"
        input_file.touch()

    user_config = UserConfig(
        endpoint=EndpointConfig(
            model_names=["test-model"],
            type=endpoint_type,
            streaming=False,
            url="http://localhost:8000",
        ),
        input=InputConfig(
            custom_dataset_type=custom_dataset_type,
            file=str(input_file) if input_file else None,
        ),
        output=OutputConfig(artifact_directory=tmp_path),
    )
    return DatasetManager(
        service_config=ServiceConfig(),
        user_config=user_config,
        service_id="test_dm",
    )


# ---------------------------------------------------------------------------
# _generate_input_payloads: raw vs formatted branch
# ---------------------------------------------------------------------------


class TestGenerateInputPayloadsAdversarial:
    def test_generate_input_payloads_uniform_raw_payload_conversations_preserves_all_turns(
        self,
    ) -> None:
        r1, r2, r3 = _raw(), _raw(), _raw()
        r2["messages"][0]["content"] = "bye"
        r3["messages"][0]["content"] = "again"
        convs = {
            "s1": Conversation(
                session_id="s1",
                turns=[
                    Turn(role="user", raw_payload=r1),
                    Turn(role="user", raw_payload=r2),
                ],
            ),
            "s2": Conversation(
                session_id="s2", turns=[Turn(role="user", raw_payload=r3)]
            ),
        }
        mgr = _stub_manager(convs)

        inputs = mgr._generate_input_payloads(_raw_endpoint())
        by_session = {s.session_id: s.payloads for s in inputs.data}
        assert by_session["s1"] == [r1, r2]
        assert by_session["s2"] == [r3]

    def test_generate_input_payloads_uniform_non_raw_conversations_formats_via_format_conversation_payloads(
        self,
    ) -> None:
        convs = {
            "s1": Conversation(
                session_id="s1",
                turns=[Turn(role="user", texts=[Text(contents=["hello"])])],
            ),
            "s2": Conversation(
                session_id="s2",
                turns=[Turn(role="user", texts=[Text(contents=["world"])])],
            ),
        }
        mgr = _stub_manager(convs)

        with patch(
            "aiperf.dataset.payload_formatting.format_conversation_payloads"
        ) as mock_fmt:
            mock_fmt.return_value = iter(
                [("s1", 0, {"fmt": "a"}), ("s2", 0, {"fmt": "b"})]
            )
            inputs = mgr._generate_input_payloads(_chat_endpoint())

        by_session = {s.session_id: s.payloads for s in inputs.data}
        assert by_session == {"s1": [{"fmt": "a"}], "s2": [{"fmt": "b"}]}

    def test_generate_input_payloads_mixed_raw_and_non_raw_across_conversations(
        self,
    ) -> None:
        """Any raw_payload anywhere -> raw branch; non-raw conv yields no payloads."""
        convs = {
            "raw": Conversation(
                session_id="raw", turns=[Turn(role="user", raw_payload=_raw())]
            ),
            "non_raw": Conversation(
                session_id="non_raw",
                turns=[Turn(role="user", texts=[Text(contents=["x"])])],
            ),
        }
        mgr = _stub_manager(convs)

        inputs = mgr._generate_input_payloads(_raw_endpoint())
        by_session = {s.session_id: s.payloads for s in inputs.data}
        # non_raw conversation contributes nothing because it has no raw_payload
        assert "raw" in by_session
        assert "non_raw" not in by_session

    def test_generate_input_payloads_empty_conversations_list_no_crash(self) -> None:
        mgr = _stub_manager({})
        inputs = mgr._generate_input_payloads(_chat_endpoint())
        assert inputs.data == []

    def test_generate_input_payloads_raw_payload_none_on_all_turns_of_conversation_treats_as_non_raw(
        self,
    ) -> None:
        """All turns have raw_payload=None -> has_raw_payloads=False -> formatted branch."""
        convs = {
            "s1": Conversation(
                session_id="s1",
                turns=[
                    Turn(role="user", raw_payload=None, texts=[Text(contents=["a"])]),
                    Turn(role="user", raw_payload=None, texts=[Text(contents=["b"])]),
                ],
            ),
        }
        mgr = _stub_manager(convs)

        with patch(
            "aiperf.dataset.payload_formatting.format_conversation_payloads"
        ) as mock_fmt:
            mock_fmt.return_value = iter([("s1", 0, {"f": 0}), ("s1", 1, {"f": 1})])
            inputs = mgr._generate_input_payloads(_chat_endpoint())

        assert inputs.data[0].payloads == [{"f": 0}, {"f": 1}]


# ---------------------------------------------------------------------------
# _preformat_payloads: all-or-nothing + NotImplementedError escape
# ---------------------------------------------------------------------------


class TestPreformatPayloadsAdversarial:
    def _make_mgr(self, convs: list[Conversation]) -> DatasetManager:
        mgr = object.__new__(DatasetManager)
        # Minimal user_config: any non-None object is enough to pass the guard.
        mgr.user_config = Mock()
        # Disable cache-bust so the preformat path runs (the cache-bust early
        # return bails preformatting whenever target != NONE).
        mgr.user_config.input.prompt.cache_bust.target = CacheBustTarget.NONE
        # Stub the logger mixin attrs that _preformat_payloads uses.
        mgr.info = Mock()
        return mgr

    def test_preformat_payloads_all_convs_eligible_formats_in_place(self) -> None:
        convs = [
            Conversation(
                session_id="s1",
                turns=[Turn(role="user", texts=[Text(contents=["hello"])])],
            ),
            Conversation(
                session_id="s2",
                context_mode=ConversationContextMode.MESSAGE_ARRAY_WITH_RESPONSES,
                turns=[
                    Turn(role="user", texts=[Text(contents=["a"])]),
                    Turn(role="assistant", texts=[Text(contents=["b"])]),
                ],
            ),
        ]
        mgr = self._make_mgr(convs)

        with (
            patch(
                "aiperf.dataset.dataset_manager.format_conversation_payloads"
            ) as mock_fmt,
            patch("aiperf.dataset.dataset_manager.ModelEndpointInfo.from_user_config"),
        ):
            mock_fmt.return_value = iter(
                [
                    ("s1", 0, {"p": "s1_0"}),
                    ("s2", 0, {"p": "s2_0"}),
                    ("s2", 1, {"p": "s2_1"}),
                ]
            )
            mgr._preformat_payloads(convs)

        assert convs[0].turns[0].raw_payload == {"p": "s1_0"}
        assert convs[1].turns[0].raw_payload == {"p": "s2_0"}
        assert convs[1].turns[1].raw_payload == {"p": "s2_1"}

    def test_preformat_payloads_one_conv_ineligible_short_circuits_entirely_all_or_nothing(
        self,
    ) -> None:
        """DELTAS_WITH_RESPONSES multi-turn conv -> preformat aborts for ALL convs."""
        convs = [
            Conversation(
                session_id="ok",
                turns=[Turn(role="user", texts=[Text(contents=["x"])])],
            ),
            Conversation(
                session_id="bad",
                context_mode=ConversationContextMode.DELTAS_WITH_RESPONSES,
                turns=[
                    Turn(role="user", texts=[Text(contents=["a"])]),
                    Turn(role="user", texts=[Text(contents=["b"])]),
                ],
            ),
        ]
        mgr = self._make_mgr(convs)

        with patch(
            "aiperf.dataset.dataset_manager.format_conversation_payloads"
        ) as mock_fmt:
            mgr._preformat_payloads(convs)
            mock_fmt.assert_not_called()

        for conv in convs:
            for turn in conv.turns:
                assert turn.raw_payload is None

    def test_preformat_payloads_endpoint_raises_not_implemented_mid_iteration_rollback_or_skip(
        self,
    ) -> None:
        """NotImplementedError mid-iteration -> silently skip; no partial payloads pin current behavior."""
        convs = [
            Conversation(
                session_id=f"s{i}",
                turns=[Turn(role="user", texts=[Text(contents=[f"t{i}"])])],
            )
            for i in range(4)
        ]
        mgr = self._make_mgr(convs)

        def _gen():
            yield ("s0", 0, {"p": 0})
            yield ("s1", 0, {"p": 1})
            raise NotImplementedError("endpoint does not support format_payload")

        with (
            patch(
                "aiperf.dataset.dataset_manager.format_conversation_payloads",
                return_value=_gen(),
            ),
            patch("aiperf.dataset.dataset_manager.ModelEndpointInfo.from_user_config"),
        ):
            mgr._preformat_payloads(convs)

        # Partial state IS left behind -- s0 and s1 got payloads before the throw.
        # This pins current behavior (swallow, no rollback).
        assert convs[0].turns[0].raw_payload == {"p": 0}
        assert convs[1].turns[0].raw_payload == {"p": 1}
        assert convs[2].turns[0].raw_payload is None
        assert convs[3].turns[0].raw_payload is None


# ---------------------------------------------------------------------------
# Skip-logic in _profile_configure_command
# ---------------------------------------------------------------------------


class TestSkipInputsJsonAdversarial:
    @pytest.mark.asyncio
    async def test_skip_inputs_json_generation_for_raw_payload_dataset_type(
        self, tmp_path: Path
    ) -> None:
        mgr = _full_manager(tmp_path, CustomDatasetType.RAW_PAYLOAD)
        mgr._configure_dataset = AsyncMock()
        mgr._configure_tokenizer = AsyncMock()
        mgr._configure_dataset_client_and_free_memory = AsyncMock()

        with patch.object(
            mgr, "_generate_inputs_json_file", new_callable=AsyncMock
        ) as mock_gen:
            await mgr._profile_configure_command(Mock())
            mock_gen.assert_not_called()

    @pytest.mark.asyncio
    async def test_skip_inputs_json_generation_for_inputs_json_dataset_type(
        self, tmp_path: Path
    ) -> None:
        mgr = _full_manager(tmp_path, CustomDatasetType.INPUTS_JSON)
        mgr._configure_dataset = AsyncMock()
        mgr._configure_tokenizer = AsyncMock()
        mgr._configure_dataset_client_and_free_memory = AsyncMock()

        with patch.object(
            mgr, "_generate_inputs_json_file", new_callable=AsyncMock
        ) as mock_gen:
            await mgr._profile_configure_command(Mock())
            mock_gen.assert_not_called()


# ---------------------------------------------------------------------------
# _generate_inputs_json_file: error handling + cleanup
# ---------------------------------------------------------------------------


class TestGenerateInputsJsonFileAdversarial:
    def _mgr(self, tmp_path: Path) -> DatasetManager:
        mgr = _full_manager(tmp_path)
        mgr.dataset = {
            "s1": Conversation(
                session_id="s1",
                turns=[Turn(role="user", raw_payload=_raw())],
            ),
        }
        return mgr

    @pytest.mark.asyncio
    async def test_generate_inputs_json_file_oserror_during_replace_swallowed_logs(
        self, tmp_path: Path, caplog
    ) -> None:
        caplog.set_level(logging.ERROR)
        mgr = self._mgr(tmp_path)

        def boom(self: Path, target: Any) -> Any:
            raise OSError("disk full")

        with patch.object(Path, "replace", boom):
            # Must not raise: OSError branch is swallowed.
            await mgr._generate_inputs_json_file()

        # Untouched call is allowed elsewhere; sanity-check module still usable.
        assert True  # type-check noop

        assert any(
            "Error generating inputs.json file" in rec.message for rec in caplog.records
        )

    @pytest.mark.asyncio
    async def test_generate_inputs_json_file_other_exception_reraised(
        self, tmp_path: Path, caplog
    ) -> None:
        caplog.set_level(logging.ERROR)
        mgr = self._mgr(tmp_path)

        with (
            patch.object(
                mgr,
                "_generate_input_payloads",
                side_effect=RuntimeError("fatal"),
            ),
            pytest.raises(RuntimeError, match="fatal"),
        ):
            await mgr._generate_inputs_json_file()

        assert any(
            "Error generating inputs.json file" in rec.message for rec in caplog.records
        )

    @pytest.mark.asyncio
    async def test_generate_inputs_json_file_temp_file_cleaned_on_success_and_failure(
        self, tmp_path: Path
    ) -> None:
        mgr = self._mgr(tmp_path)
        tmp_file = tmp_path / "inputs.tmp"

        # Success path: no .tmp lingers after atomic replace.
        await mgr._generate_inputs_json_file()
        assert not tmp_file.exists()
        assert (tmp_path / "inputs.json").exists()

        # Failure path: .tmp written but replace raises -> finally unlink removes it.
        (tmp_path / "inputs.json").unlink()

        def boom_replace(self: Path, target: Any) -> Any:
            raise OSError("cannot replace")

        with patch.object(Path, "replace", boom_replace):
            await mgr._generate_inputs_json_file()

        # finally: if a .tmp was written it should be gone now.
        assert not tmp_file.exists()


# ---------------------------------------------------------------------------
# Wave-2 fix targets (xfail strict)
# ---------------------------------------------------------------------------


class TestWave2FixTargets:
    @pytest.mark.asyncio
    async def test_mooncake_trace_with_payload_mode_skips_inputs_json_post_fix(
        self, tmp_path: Path
    ) -> None:
        mgr = _full_manager(tmp_path, CustomDatasetType.MOONCAKE_TRACE)
        mgr._configure_dataset = AsyncMock()
        mgr._configure_tokenizer = AsyncMock()
        mgr._configure_dataset_client_and_free_memory = AsyncMock()

        # Simulate Mooncake loader having built raw_payload-backed turns.
        mgr.dataset = {
            "s1": Conversation(
                session_id="s1",
                context_mode=ConversationContextMode.MESSAGE_ARRAY_WITH_RESPONSES,
                turns=[Turn(role="user", raw_payload=_raw())],
            ),
        }
        # _configure_dataset is mocked out, so set the source-payload flag
        # it would normally compute before _preformat_payloads ran.
        mgr._all_turns_source_loaded_payloads = True

        with patch.object(
            mgr, "_generate_inputs_json_file", new_callable=AsyncMock
        ) as mock_gen:
            await mgr._profile_configure_command(Mock())
            mock_gen.assert_not_called()  # CURRENT: called; FIX: not called.

    def test_mixed_raw_and_non_raw_turns_raises_or_handles_consistently_post_fix(
        self,
    ) -> None:
        conv = Conversation(
            session_id="s1",
            turns=[
                Turn(role="user", raw_payload=_raw()),
                Turn(role="user", texts=[Text(contents=["should-not-be-dropped"])]),
            ],
        )
        mgr = _stub_manager({"s1": conv})

        # Expectation: either the call raises a ValueError mentioning mixed
        # raw_payload, or it returns all turns (2 payloads). CURRENT: silently
        # returns 1 payload.
        with pytest.raises(ValueError, match="mixed raw_payload"):
            mgr._generate_input_payloads(_raw_endpoint())
