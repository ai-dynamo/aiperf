# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``HFMooncakeTraceDatasetLoader`` and recorded trajectory conversion."""

from copy import deepcopy
from decimal import Decimal
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, Mock

import orjson
import pytest
from pytest import param

from aiperf.common.enums import ConversationContextMode
from aiperf.common.exceptions import DatasetLoaderError
from aiperf.config.dataset import PublicDataset
from aiperf.config.flags import CLIConfig
from aiperf.config.resolution.plan import BenchmarkRun
from aiperf.dataset.loader.base_hf_dataset import BaseHFDatasetLoader
from aiperf.dataset.loader.hf_mooncake_trace import (
    HFMooncakeTraceDatasetLoader,
    trajectory_to_mooncake,
)
from aiperf.dataset.loader.models import MooncakeTrace
from aiperf.dataset.loader.mooncake_trace import MooncakeTraceDatasetLoader
from aiperf.endpoints.openai_chat import ChatEndpoint
from aiperf.plugin import plugins
from aiperf.plugin.enums import DatasetSamplingStrategy, PluginType, PublicDatasetType
from aiperf.workers.session_manager import UserSessionManager
from tests.unit.conftest import make_run_from_cli


@pytest.fixture
def row() -> dict[str, Any]:
    return {
        "session_id": "swe:1",
        "model": "google/gemma-4-26B-A4B-it",
        "benchmark": "SWE-Bench Multilingual",
        "messages": [
            {"role": "system", "content": "system", "tool_calls": None},
            {
                "role": "user",
                "content": "question",
                "reasoning_content": None,
                "tool_call_id": None,
                "name": None,
            },
            {
                "role": "assistant",
                "content": None,
                "reasoning_content": "recorded reasoning",
                "tool_calls": [
                    {
                        "id": "call1",
                        "index": 0,
                        "type": "function",
                        "function": {"name": "bash", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "content": "result", "tool_call_id": "call1"},
            {"role": "assistant", "content": "recorded reply"},
            {"role": "tool", "content": "result2", "tool_call_id": "call2"},
            {"role": "assistant", "content": "recorded final reply"},
            {"role": "tool", "content": "trailing result", "tool_call_id": "call3"},
        ],
        "tools": '[{"type":"function","function":{"name":"bash"}}]',
        "output_length": 2048,
        "extra": {
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": -1,
            "seed": 42,
            "stream": True,
        },
    }


@pytest.fixture
def loader(
    benchmark_run: BenchmarkRun, mock_prompt_generator: Mock
) -> HFMooncakeTraceDatasetLoader:
    return HFMooncakeTraceDatasetLoader(
        run=benchmark_run,
        hf_dataset_name="example/trajectories",
        streaming=True,
        prompt_generator=mock_prompt_generator,
    )


@pytest.fixture
def mock_hf_dataset(monkeypatch: pytest.MonkeyPatch, row: dict[str, Any]) -> AsyncMock:
    mock = AsyncMock(return_value={"dataset": [row]})
    monkeypatch.setattr(BaseHFDatasetLoader, "load_dataset", mock)
    return mock


class TestConfiguration:
    def test_records_hf_source_defaults(
        self, loader: HFMooncakeTraceDatasetLoader
    ) -> None:
        assert loader.hf_dataset_name == "example/trajectories"
        assert loader.hf_split == "test"
        assert loader.hf_subset is None
        assert loader.hf_revision is None
        assert loader.streaming is True

    def test_explicit_subset(
        self, benchmark_run: BenchmarkRun, mock_prompt_generator: Mock
    ) -> None:
        loader = HFMooncakeTraceDatasetLoader(
            run=benchmark_run,
            hf_dataset_name="example/trajectories",
            prompt_generator=mock_prompt_generator,
            hf_subset="custom-config",
        )
        assert loader.hf_subset == "custom-config"


@pytest.mark.asyncio
class TestLoadDatasetRowValidation:
    async def test_returns_validated_traces_keyed_by_id(
        self,
        loader: HFMooncakeTraceDatasetLoader,
        mock_hf_dataset: AsyncMock,
        row: dict[str, Any],
    ) -> None:
        mock_hf_dataset.return_value = {
            "dataset": [row, row | {"session_id": "second"}]
        }
        data = await loader.load_dataset()
        mock_hf_dataset.assert_awaited_once_with()
        assert list(data) == ["swe:1", "second"]
        for session_id, traces in data.items():
            assert len(traces) == 3
            assert all(isinstance(trace, MooncakeTrace) for trace in traces)
            assert all(trace.session_id == session_id for trace in traces)
        assert loader._mooncake.inline_records == []

    async def test_empty_dataset_raises(
        self,
        loader: HFMooncakeTraceDatasetLoader,
        mock_hf_dataset: AsyncMock,
    ) -> None:
        mock_hf_dataset.return_value = {"dataset": []}
        with pytest.raises(DatasetLoaderError, match="no trajectories matched"):
            await loader.load_dataset()

    async def test_filters_require_model_and_benchmark_matches(
        self,
        benchmark_run: BenchmarkRun,
        mock_prompt_generator: Mock,
        mock_hf_dataset: AsyncMock,
        row: dict[str, Any],
    ) -> None:
        loader = HFMooncakeTraceDatasetLoader(
            run=benchmark_run,
            hf_dataset_name="example/trajectories",
            prompt_generator=mock_prompt_generator,
            filters={"model": row["model"], "benchmark": row["benchmark"]},
        )
        mock_hf_dataset.return_value = {
            "dataset": [row | {"model": "other"}, row | {"benchmark": "other"}, row]
        }
        assert list(await loader.load_dataset()) == [row["session_id"]]

    async def test_no_matching_rows_raises(
        self,
        benchmark_run: BenchmarkRun,
        mock_prompt_generator: Mock,
        mock_hf_dataset: AsyncMock,
    ) -> None:
        loader = HFMooncakeTraceDatasetLoader(
            run=benchmark_run,
            hf_dataset_name="example/trajectories",
            prompt_generator=mock_prompt_generator,
            filters={"benchmark": "other"},
        )
        with pytest.raises(DatasetLoaderError, match="no trajectories matched"):
            await loader.load_dataset()

    @pytest.mark.parametrize("value,spelling,other", [
        param(True, "true", 1, id="json_true"),
        param(False, "false", 0, id="json_false"),
        param(True, "True", False, id="python_true"),
        param(False, "False", True, id="python_false"),
        param(None, "null", "None", id="json_null"),
        param(None, "None", False, id="python_none"),
        param(42, "42", 43, id="integer"),
        param(1.5, "1.5", 2.5, id="float"),
        param("true", "true", "True", id="string"),
        param(Decimal("2.00"), "2.00", Decimal("1.50"), id="decimal"),
        param(b"match", "b'match'", b"other", id="binary"),
    ])  # fmt: skip
    async def test_scalar_filter_selects_matching_rows(
        self,
        loader: HFMooncakeTraceDatasetLoader,
        mock_hf_dataset: AsyncMock,
        row: dict[str, Any],
        value: bool | str | int | float | Decimal | bytes | None,
        spelling: str,
        other: bool | str | int | float | Decimal | bytes,
    ) -> None:
        loader.filters = {"success": spelling}
        mock_hf_dataset.return_value = {
            "dataset": [row | {"success": other}, row | {"success": value}]
        }
        assert list(await loader.load_dataset()) == [row["session_id"]]

    async def test_missing_filter_column_raises(
        self,
        benchmark_run: BenchmarkRun,
        mock_prompt_generator: Mock,
        mock_hf_dataset: AsyncMock,
    ) -> None:
        loader = HFMooncakeTraceDatasetLoader(
            run=benchmark_run,
            hf_dataset_name="example/trajectories",
            prompt_generator=mock_prompt_generator,
            filters={"missing": "x"},
        )
        with pytest.raises(DatasetLoaderError, match="row 0:.*source path.*missing"):
            await loader.load_dataset()

    async def test_nested_source_filter(
        self,
        benchmark_run: BenchmarkRun,
        mock_prompt_generator: Mock,
        mock_hf_dataset: AsyncMock,
        row: dict[str, Any],
    ) -> None:
        row["metadata"] = '{"model":"selected"}'
        loader = HFMooncakeTraceDatasetLoader(
            run=benchmark_run,
            hf_dataset_name="example/trajectories",
            prompt_generator=mock_prompt_generator,
            filters={"metadata.model": "selected"},
        )
        assert list(await loader.load_dataset()) == [row["session_id"]]

    @pytest.mark.parametrize("session_ids", [
        param(("swe:1", "swe:1"), id="strings"),
        param((1, "1"), id="normalized_integer"),
    ])  # fmt: skip
    async def test_duplicate_session_ids_raises(
        self,
        loader: HFMooncakeTraceDatasetLoader,
        mock_hf_dataset: AsyncMock,
        row: dict[str, Any],
        session_ids: tuple[str | int, str | int],
    ) -> None:
        mock_hf_dataset.return_value = {
            "dataset": [row | {"session_id": value} for value in session_ids]
        }
        with pytest.raises(DatasetLoaderError, match="row 1:.*duplicate session_id"):
            await loader.load_dataset()

    @pytest.mark.parametrize("session_id", [
        param("", id="empty"),
        param(True, id="boolean"),
        param(1.5, id="float"),
    ])  # fmt: skip
    async def test_invalid_session_id_raises_with_row_index(
        self,
        loader: HFMooncakeTraceDatasetLoader,
        mock_hf_dataset: AsyncMock,
        row: dict[str, Any],
        session_id: str | bool | float,
    ) -> None:
        mock_hf_dataset.return_value = {
            "dataset": [row, row | {"session_id": session_id}]
        }
        with pytest.raises(DatasetLoaderError, match="row 1:.*nonempty string"):
            await loader.load_dataset()

    @pytest.mark.parametrize("missing_id", [
        param({}, id="absent"),
        param({"session_id": None}, id="null"),
    ])  # fmt: skip
    async def test_absent_or_null_session_ids_are_generated(
        self,
        loader: HFMooncakeTraceDatasetLoader,
        mock_hf_dataset: AsyncMock,
        row: dict[str, Any],
        missing_id: dict[str, None],
    ) -> None:
        row.pop("session_id")
        mock_hf_dataset.return_value = {"dataset": [row | missing_id, row | missing_id]}
        assert list(await loader.load_dataset()) == ["hf-row-0", "hf-row-1"]

    async def test_duration_only_loads_all_rows(
        self,
        mock_prompt_generator: Mock,
        mock_hf_dataset: AsyncMock,
        row: dict[str, Any],
    ) -> None:
        benchmark_run = make_run_from_cli(
            CLIConfig(
                model_names=["target"],
                hf_mooncake_dataset="example/trajectories",
                benchmark_duration=10,
            )
        )
        mock_hf_dataset.return_value = {
            "dataset": iter([row, row | {"session_id": "second"}])
        }
        loader = HFMooncakeTraceDatasetLoader(
            run=benchmark_run,
            hf_dataset_name="example/trajectories",
            prompt_generator=mock_prompt_generator,
            streaming=True,
        )
        traces = await loader.load_dataset()
        assert list(traces) == ["swe:1", "second"]
        assert all(len(turns) == 3 for turns in traces.values())

    @pytest.mark.parametrize("flags,expected", [
        param({"conversation_num": 1}, 1, id="sessions"),
        param({"request_count": 1}, 1, id="requests"),
        param({"request_count": 2, "concurrency": 2}, 2, id="concurrent_requests"),
        param({"conversation_num_dataset_entries": 1, "benchmark_duration": 10}, 1, id="entries"),
    ])  # fmt: skip
    async def test_limit_stops_stream_after_complete_matching_trajectory(
        self,
        mock_prompt_generator: Mock,
        mock_hf_dataset: AsyncMock,
        row: dict[str, Any],
        flags: dict[str, int],
        expected: int,
    ) -> None:
        loader = HFMooncakeTraceDatasetLoader(
            run=make_run_from_cli(
                CLIConfig(
                    model_names=["target"],
                    hf_mooncake_dataset="example/trajectories",
                    **flags,
                )
            ),
            hf_dataset_name="example/trajectories",
            prompt_generator=mock_prompt_generator,
            filters={"model": row["model"]},
            streaming=True,
        )
        selected = [row | {"session_id": str(i)} for i in range(expected)]
        unread = row | {"session_id": "unread"}
        rows = iter([row | {"model": "other"}, *selected, unread])
        mock_hf_dataset.return_value = {"dataset": rows}

        traces = await loader.load_dataset()

        assert list(traces) == [source["session_id"] for source in selected]
        assert all(len(turns) == 3 for turns in traces.values())
        assert next(rows) is unread

    async def test_inline_records_released_after_load_failure(
        self,
        loader: HFMooncakeTraceDatasetLoader,
        mock_hf_dataset: AsyncMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            loader._mooncake,
            "load_dataset",
            Mock(side_effect=ValueError("invalid trace")),
        )
        with pytest.raises(ValueError, match="invalid trace"):
            await loader.load_dataset()
        assert loader._mooncake.inline_records == []


class TestFieldMapping:
    @pytest.mark.parametrize("present", [False, True])
    def test_whitespace_column_name_is_literal(
        self,
        loader: HFMooncakeTraceDatasetLoader,
        row: dict[str, Any],
        present: bool,
    ) -> None:
        dataset = PublicDataset(
            name="default",
            type="public",
            dataset=PublicDatasetType.MOONCAKE_HF,
            hf_mooncake_dataset="example/trajectories",
            mooncake_trace_to_hf_mapping={"messages": "nested.   "},
        )
        loader.field_mapping = dataset.mooncake_trace_to_hf_mapping
        source = row | {"nested": {"   ": row["messages"]} if present else {}}
        if present:
            assert list(loader._rows_to_traces([source])) == [row["session_id"]]
        else:
            with pytest.raises(DatasetLoaderError, match="source path.*does not exist"):
                loader._rows_to_traces([source])

    @pytest.mark.parametrize("session_id", [0, 123])
    def test_integer_session_id_mapping_normalizes_to_string(
        self,
        loader: HFMooncakeTraceDatasetLoader,
        row: dict[str, Any],
        session_id: int,
    ) -> None:
        loader.field_mapping = {"session_id": "question_id"}
        traces = loader._rows_to_traces([row | {"question_id": session_id}])
        assert list(traces) == [str(session_id)]
        assert all(
            trace.session_id == str(session_id) for trace in traces[str(session_id)]
        )

    @pytest.mark.parametrize("encoded", [False, True])
    def test_mapped_columns_match_canonical_traces(
        self,
        loader: HFMooncakeTraceDatasetLoader,
        benchmark_run: BenchmarkRun,
        mock_prompt_generator: Mock,
        row: dict[str, Any],
        encoded: bool,
    ) -> None:
        source = {
            "question_id": row["session_id"],
            "turns": row["messages"],
            "extras": {
                "tools": row["tools"],
                "max_tokens": row["output_length"],
                "request_params": orjson.dumps(
                    row["extra"] | {"temperature": 0.1}
                ).decode(),
                "temperature": 0.7,
                "top_p": None,
            },
        }
        if encoded:
            source["extras"] = orjson.dumps(source["extras"]).decode()
            source["turns"] = orjson.dumps(source["turns"]).decode()
        before = deepcopy(source)
        mapped = HFMooncakeTraceDatasetLoader(
            run=benchmark_run,
            hf_dataset_name="example/trajectories",
            prompt_generator=mock_prompt_generator,
            field_mapping={
                "session_id": "question_id",
                "messages": "turns",
                "tools": "extras.tools",
                "output_length": "extras.max_tokens",
                "extra": "extras.request_params",
                "extra.temperature": "extras.temperature",
                "extra.top_p": "extras.top_p",
            },
        )._rows_to_traces([source])
        assert mapped == loader._rows_to_traces([row])
        assert source == before

    @pytest.mark.parametrize("mapping,error", [
        param({"messages": "missing"}, "source path.*missing", id="missing_source"),
        param({"extra.model": "model"}, "must not override", id="mapped_model_override"),
    ])  # fmt: skip
    def test_invalid_mapped_row_fails_clearly(
        self,
        benchmark_run: BenchmarkRun,
        mock_prompt_generator: Mock,
        mapping: dict[str, str],
        error: str,
        row: dict[str, Any],
    ) -> None:
        with pytest.raises(DatasetLoaderError, match=error):
            HFMooncakeTraceDatasetLoader(
                run=benchmark_run,
                hf_dataset_name="example/trajectories",
                prompt_generator=mock_prompt_generator,
                field_mapping=mapping,
            )._rows_to_traces([row])


class TestTrajectoryConversion:
    @pytest.mark.parametrize(
        "roles,lengths",
        [
            param(["user", "assistant"], [1], id="final_assistant"),
            param(["system", "user", "assistant", "tool", "assistant", "tool", "assistant", "tool"], [2, 4, 6], id="original_example"),
            param(["user", "user", "assistant", "assistant", "tool", "tool", "assistant"], [2, 3, 6], id="consecutive_roles"),
        ],
    )  # fmt: skip
    def test_splits_before_every_assistant(
        self, roles: list[str], lengths: list[int], row: dict[str, Any]
    ) -> None:
        row["messages"] = [
            {"role": role, "content": str(i)} for i, role in enumerate(roles)
        ]
        traces = trajectory_to_mooncake(row)
        assert [trace.messages for trace in traces] == [
            row["messages"][:end] for end in lengths
        ]
        assert all(trace.assistant_responses == "recorded" for trace in traces)

    @pytest.mark.parametrize(
        "patch,error",
        [
            param({"messages": []}, "nonempty", id="empty"),
            param({"messages": [{"role": "assistant", "content": "x"}]}, "message 0", id="empty_prefix"),
            param({"messages": [{"role": "user", "content": "x"}]}, "no assistant", id="no_boundary"),
            param({"messages": [None]}, "message 0", id="invalid_message"),
            param({"tools": "invalid JSON"}, "session", id="tools_json"),
            param({"extra": '[]'}, "object", id="params_type"),
            param({"extra": '{"messages":[]}'}, "override", id="history_override"),
            param({"extra": '{"model":"other"}'}, "override", id="model_override"),
            param({"output_length": 0}, "output_length", id="invalid_cap"),
        ],
    )  # fmt: skip
    def test_invalid_rows_fail_with_session(
        self, patch: dict[str, Any], error: str, row: dict[str, Any]
    ) -> None:
        with pytest.raises(DatasetLoaderError, match=error) as caught:
            trajectory_to_mooncake(row | patch)
        assert "swe:1" in str(caught.value)

    @pytest.mark.parametrize("params", [
        param({"max_tokens": 8}, id="max_tokens"),
        param({"max_completion_tokens": 8}, id="max_completion_tokens"),
        param({"max_tokens": None, "max_completion_tokens": 8}, id="null_legacy"),
    ])  # fmt: skip
    def test_output_cap_falls_back_to_request_params(
        self, params: dict[str, Any], row: dict[str, Any]
    ) -> None:
        row["output_length"] = None
        row["extra"] = orjson.dumps(params).decode()
        for trace in trajectory_to_mooncake(row):
            assert trace.output_length == 8
            assert trace.extra == {}


@pytest.mark.asyncio
class TestConvertToConversationsDelegation:
    async def test_hf_pipeline_delegates_and_matches_mooncake_file(
        self,
        loader: HFMooncakeTraceDatasetLoader,
        mock_hf_dataset: AsyncMock,
        mock_prompt_generator: Mock,
        monkeypatch: pytest.MonkeyPatch,
        row: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        rows = [row, row | {"session_id": "second"}]
        mock_hf_dataset.return_value = {"dataset": rows}
        load = Mock(wraps=loader._mooncake.load_dataset)
        convert = Mock(wraps=loader._mooncake.convert_to_conversations)
        monkeypatch.setattr(loader._mooncake, "load_dataset", load)
        monkeypatch.setattr(loader._mooncake, "convert_to_conversations", convert)
        data = await loader.load_dataset()
        load.assert_called_once_with()
        conversations = await loader.convert_to_conversations(data)
        convert.assert_called_once_with(data)

        filename = tmp_path / "traces.jsonl"
        filename.write_bytes(
            b"".join(
                orjson.dumps(
                    trace.model_dump(exclude_none=True),
                    option=orjson.OPT_APPEND_NEWLINE,
                )
                for source in rows
                for trace in trajectory_to_mooncake(source)
            )
        )
        file_loader = MooncakeTraceDatasetLoader(
            filename=filename, run=loader.run, prompt_generator=mock_prompt_generator
        )
        assert conversations == file_loader.convert_to_conversations(
            file_loader.load_dataset()
        )

    async def test_recorded_prefixes_preserve_payload_and_disable_response_storage(
        self,
        loader: HFMooncakeTraceDatasetLoader,
        mock_hf_dataset: AsyncMock,
        row: dict[str, Any],
    ) -> None:
        before = deepcopy(row)
        expected_messages = [
            {key: value for key, value in message.items() if value is not None}
            for message in deepcopy(row["messages"])
        ]
        expected_messages[2]["tool_calls"][0].pop("index")
        data = await loader.load_dataset()
        conversation = (await loader.convert_to_conversations(data))[0]
        assert row == before
        assert (
            conversation.context_mode
            == ConversationContextMode.MESSAGE_ARRAY_WITH_RESPONSES
        )
        assert len(conversation.turns) == 3
        session = UserSessionManager().create_and_store(
            x_correlation_id="test", conversation=conversation, num_turns=3
        )
        endpoint = ChatEndpoint(model_endpoint=Mock())
        for index, end in enumerate([2, 4, 6]):
            session.advance_turn(index)
            assert not session.should_store_response()
            messages = endpoint.build_messages(session.turn_list)
            assert messages == expected_messages[:end]
            turn = conversation.turns[index]
            assert turn.max_tokens == 2048
            assert turn.raw_tools == orjson.loads(row["tools"])
            assert turn.extra_body == {
                "temperature": 0.7,
                "top_p": 0.8,
                "top_k": -1,
                "seed": 42,
            }
            assert turn.model is None


class TestSamplingStrategy:
    def test_preferred_sampling_strategy_is_sequential(self) -> None:
        assert (
            HFMooncakeTraceDatasetLoader.get_preferred_sampling_strategy()
            == DatasetSamplingStrategy.SEQUENTIAL
        )


class TestPluginRegistry:
    def test_class_registered_under_public_dataset_loader(self) -> None:
        assert (
            plugins.get_class(
                PluginType.PUBLIC_DATASET_LOADER, PublicDatasetType.MOONCAKE_HF
            )
            is HFMooncakeTraceDatasetLoader
        )

    def test_metadata_uses_runtime_repository_and_test_split(self) -> None:
        metadata = plugins.get_public_dataset_loader_metadata(
            PublicDatasetType.MOONCAKE_HF
        )
        assert metadata.is_trace is True
        assert metadata.streaming is True
        assert metadata.hf_dataset_name is None
        assert metadata.hf_split == "test"
