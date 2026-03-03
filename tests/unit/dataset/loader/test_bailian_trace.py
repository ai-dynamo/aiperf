# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import logging
from unittest.mock import Mock, patch

import pytest
from pydantic import ValidationError

from aiperf.common.config import (
    EndpointConfig,
    InputConfig,
    InputTokensConfig,
    PromptConfig,
    SynthesisConfig,
    UserConfig,
)
from aiperf.dataset.loader.bailian_trace import BailianTraceDatasetLoader
from aiperf.dataset.loader.models import BailianTrace

# ============================================================================
# BailianTrace Model Tests
# ============================================================================


class TestBailianTrace:
    """Validation and construction tests for the BailianTrace model."""

    def test_create_minimal(self):
        trace = BailianTrace(
            chat_id=1,
            timestamp=1700000000.0,
            input_length=100,
            output_length=40,
        )
        assert trace.chat_id == 1
        assert trace.parent_chat_id == -1
        assert trace.timestamp == 1700000000.0
        assert trace.input_length == 100
        assert trace.output_length == 40
        assert trace.request_type == ""
        assert trace.turn == 1
        assert trace.hash_ids == []

    def test_create_full(self):
        trace = BailianTrace(
            chat_id=42,
            parent_chat_id=10,
            timestamp=1700000001.5,
            input_length=256,
            output_length=64,
            request_type="chat",
            turn=3,
            hash_ids=[1, 2, 3, 4, 5],
        )
        assert trace.chat_id == 42
        assert trace.parent_chat_id == 10
        assert trace.turn == 3
        assert trace.hash_ids == [1, 2, 3, 4, 5]
        assert trace.request_type == "chat"

    def test_type_alias_deserialization(self):
        """The JSONL 'type' field maps to 'request_type' via alias."""
        raw = (
            '{"chat_id": 1, "timestamp": 1.0, "input_length": 10, '
            '"output_length": 5, "type": "inference"}'
        )
        trace = BailianTrace.model_validate_json(raw)
        assert trace.request_type == "inference"

    def test_missing_required_chat_id(self):
        with pytest.raises(ValidationError, match="chat_id"):
            BailianTrace(
                timestamp=1.0,
                input_length=10,
                output_length=5,
            )

    def test_missing_required_timestamp(self):
        with pytest.raises(ValidationError, match="timestamp"):
            BailianTrace(
                chat_id=1,
                input_length=10,
                output_length=5,
            )

    def test_missing_required_input_length(self):
        with pytest.raises(ValidationError, match="input_length"):
            BailianTrace(
                chat_id=1,
                timestamp=1.0,
                output_length=5,
            )

    def test_missing_required_output_length(self):
        with pytest.raises(ValidationError, match="output_length"):
            BailianTrace(
                chat_id=1,
                timestamp=1.0,
                input_length=10,
            )


# ============================================================================
# BailianTraceDatasetLoader Tests
# ============================================================================


class TestBailianTraceDatasetLoader:
    """Core loader functionality tests."""

    @pytest.fixture
    def mock_prompt_generator(self):
        generator = Mock()
        generator.generate.return_value = "Generated prompt text"
        generator._decoded_cache = {}
        generator._build_token_sequence.return_value = [1, 2, 3, 4, 5]
        return generator

    @pytest.fixture
    def default_user_config(self):
        return UserConfig(endpoint=EndpointConfig(model_names=["test-model"]))

    def _make_user_config(
        self,
        start_offset: int | None = None,
        end_offset: int | None = None,
        file: str | None = None,
    ) -> UserConfig:
        has_offsets = start_offset is not None or end_offset is not None
        input_config = (
            InputConfig(
                file=file,
                fixed_schedule=True,
                fixed_schedule_start_offset=start_offset,
                fixed_schedule_end_offset=end_offset,
            )
            if has_offsets
            else InputConfig()
        )
        return UserConfig(
            endpoint=EndpointConfig(model_names=["test-model"]),
            input=input_config,
        )

    # ---- basic loading ----

    def test_load_basic(
        self, create_jsonl_file, mock_prompt_generator, default_user_config
    ):
        content = [
            '{"chat_id": 1, "parent_chat_id": -1, "timestamp": 1.0, "input_length": 100, "output_length": 40, "type": "text", "turn": 1, "hash_ids": [10, 20]}',
            '{"chat_id": 2, "parent_chat_id": -1, "timestamp": 2.0, "input_length": 200, "output_length": 80, "type": "text", "turn": 1, "hash_ids": [30]}',
        ]
        filename = create_jsonl_file(content)

        loader = BailianTraceDatasetLoader(
            filename=filename,
            user_config=default_user_config,
            prompt_generator=mock_prompt_generator,
        )
        dataset = loader.load_dataset()

        assert len(dataset) == 2
        all_traces = [t for traces in dataset.values() for t in traces]
        assert all_traces[0].input_length == 100
        assert all_traces[1].input_length == 200

    def test_timestamps_converted_to_milliseconds(
        self, create_jsonl_file, mock_prompt_generator, default_user_config
    ):
        content = [
            '{"chat_id": 1, "timestamp": 1.5, "input_length": 10, "output_length": 5}',
        ]
        filename = create_jsonl_file(content)

        loader = BailianTraceDatasetLoader(
            filename=filename,
            user_config=default_user_config,
            prompt_generator=mock_prompt_generator,
        )
        dataset = loader.load_dataset()

        trace = list(dataset.values())[0][0]
        assert trace.timestamp == 1500.0

    def test_skips_empty_lines(
        self, create_jsonl_file, mock_prompt_generator, default_user_config
    ):
        content = [
            '{"chat_id": 1, "timestamp": 1.0, "input_length": 10, "output_length": 5}',
            "",
            '{"chat_id": 2, "timestamp": 2.0, "input_length": 20, "output_length": 10}',
        ]
        filename = create_jsonl_file(content)

        loader = BailianTraceDatasetLoader(
            filename=filename,
            user_config=default_user_config,
            prompt_generator=mock_prompt_generator,
        )
        dataset = loader.load_dataset()

        total = sum(len(v) for v in dataset.values())
        assert total == 2

    # ---- multi-turn grouping ----

    def test_groups_by_parent_chat_id(
        self, create_jsonl_file, mock_prompt_generator, default_user_config
    ):
        """Entries with the same root should be grouped into one session."""
        content = [
            '{"chat_id": 100, "parent_chat_id": -1, "timestamp": 1.0, "input_length": 50, "output_length": 20, "turn": 1}',
            '{"chat_id": 101, "parent_chat_id": 100, "timestamp": 2.0, "input_length": 60, "output_length": 25, "turn": 2}',
            '{"chat_id": 102, "parent_chat_id": 101, "timestamp": 3.0, "input_length": 70, "output_length": 30, "turn": 3}',
        ]
        filename = create_jsonl_file(content)

        loader = BailianTraceDatasetLoader(
            filename=filename,
            user_config=default_user_config,
            prompt_generator=mock_prompt_generator,
        )
        dataset = loader.load_dataset()

        # All three should be in one session rooted at chat_id=100
        assert len(dataset) == 1
        session = list(dataset.values())[0]
        assert len(session) == 3
        assert [t.turn for t in session] == [1, 2, 3]

    def test_separate_sessions(
        self, create_jsonl_file, mock_prompt_generator, default_user_config
    ):
        """Independent root entries form separate sessions."""
        content = [
            '{"chat_id": 1, "parent_chat_id": -1, "timestamp": 1.0, "input_length": 50, "output_length": 20, "turn": 1}',
            '{"chat_id": 2, "parent_chat_id": -1, "timestamp": 2.0, "input_length": 60, "output_length": 25, "turn": 1}',
        ]
        filename = create_jsonl_file(content)

        loader = BailianTraceDatasetLoader(
            filename=filename,
            user_config=default_user_config,
            prompt_generator=mock_prompt_generator,
        )
        dataset = loader.load_dataset()

        assert len(dataset) == 2

    def test_turns_sorted_within_session(
        self, create_jsonl_file, mock_prompt_generator, default_user_config
    ):
        """Turns are sorted even if JSONL order differs."""
        content = [
            '{"chat_id": 102, "parent_chat_id": 101, "timestamp": 3.0, "input_length": 70, "output_length": 30, "turn": 3}',
            '{"chat_id": 100, "parent_chat_id": -1, "timestamp": 1.0, "input_length": 50, "output_length": 20, "turn": 1}',
            '{"chat_id": 101, "parent_chat_id": 100, "timestamp": 2.0, "input_length": 60, "output_length": 25, "turn": 2}',
        ]
        filename = create_jsonl_file(content)

        loader = BailianTraceDatasetLoader(
            filename=filename,
            user_config=default_user_config,
            prompt_generator=mock_prompt_generator,
        )
        dataset = loader.load_dataset()

        session = list(dataset.values())[0]
        assert [t.turn for t in session] == [1, 2, 3]

    # ---- offset filtering ----

    @pytest.mark.parametrize(
        "start_offset,end_offset,expected_count,description",
        [
            (None, None, 3, "no filtering"),
            (1500, None, 2, "start offset only — keeps ts >= 1500 ms"),
            (None, 2500, 2, "end offset only — keeps ts <= 2500 ms"),
            (1500, 2500, 1, "both offsets — keeps ts in [1500, 2500]"),
        ],
    )  # fmt: skip
    def test_offset_filtering(
        self,
        create_jsonl_file,
        mock_prompt_generator,
        start_offset,
        end_offset,
        expected_count,
        description,
    ):
        """Timestamps are converted to ms before offset comparison."""
        content = [
            '{"chat_id": 1, "timestamp": 1.0, "input_length": 10, "output_length": 5}',
            '{"chat_id": 2, "timestamp": 2.0, "input_length": 10, "output_length": 5}',
            '{"chat_id": 3, "timestamp": 3.0, "input_length": 10, "output_length": 5}',
        ]
        filename = create_jsonl_file(content)

        user_config = self._make_user_config(start_offset, end_offset, file=filename)
        loader = BailianTraceDatasetLoader(
            filename=filename,
            user_config=user_config,
            prompt_generator=mock_prompt_generator,
        )
        dataset = loader.load_dataset()

        total = sum(len(v) for v in dataset.values())
        assert total == expected_count, f"Failed for {description}"

    def test_offset_filtering_logs_skipped(
        self, create_jsonl_file, mock_prompt_generator, caplog
    ):
        caplog.set_level(logging.INFO)

        content = [
            '{"chat_id": 1, "timestamp": 0.5, "input_length": 10, "output_length": 5}',
            '{"chat_id": 2, "timestamp": 2.0, "input_length": 10, "output_length": 5}',
            '{"chat_id": 3, "timestamp": 5.0, "input_length": 10, "output_length": 5}',
        ]
        filename = create_jsonl_file(content)

        user_config = self._make_user_config(1000, 3000, file=filename)
        loader = BailianTraceDatasetLoader(
            filename=filename,
            user_config=user_config,
            prompt_generator=mock_prompt_generator,
        )
        loader.load_dataset()

        assert "Skipped 2 traces" in caplog.text

    # ---- max_isl / max_osl ----

    @pytest.mark.parametrize(
        "max_isl,expected_count",
        [
            (None, 3),
            (500, 3),
            (150, 2),
            (50, 0),
        ],
    )  # fmt: skip
    def test_max_isl_filtering(
        self, create_jsonl_file, mock_prompt_generator, max_isl, expected_count
    ):
        content = [
            '{"chat_id": 1, "timestamp": 1.0, "input_length": 100, "output_length": 10}',
            '{"chat_id": 2, "timestamp": 2.0, "input_length": 150, "output_length": 10}',
            '{"chat_id": 3, "timestamp": 3.0, "input_length": 200, "output_length": 10}',
        ]
        filename = create_jsonl_file(content)

        user_config = UserConfig(
            endpoint=EndpointConfig(model_names=["test-model"]),
            input=InputConfig(synthesis=SynthesisConfig(max_isl=max_isl)),
        )
        loader = BailianTraceDatasetLoader(
            filename=filename,
            user_config=user_config,
            prompt_generator=mock_prompt_generator,
        )
        dataset = loader.load_dataset()

        total = sum(len(v) for v in dataset.values())
        assert total == expected_count

    @pytest.mark.parametrize(
        "max_osl,expected_output_lengths",
        [
            (None, [50, 100, 150]),
            (500, [50, 100, 150]),
            (100, [50, 100, 100]),
            (25, [25, 25, 25]),
        ],
    )  # fmt: skip
    def test_max_osl_capping(
        self, create_jsonl_file, mock_prompt_generator, max_osl, expected_output_lengths
    ):
        content = [
            '{"chat_id": 1, "timestamp": 1.0, "input_length": 10, "output_length": 50}',
            '{"chat_id": 2, "timestamp": 2.0, "input_length": 10, "output_length": 100}',
            '{"chat_id": 3, "timestamp": 3.0, "input_length": 10, "output_length": 150}',
        ]
        filename = create_jsonl_file(content)

        user_config = UserConfig(
            endpoint=EndpointConfig(model_names=["test-model"]),
            input=InputConfig(synthesis=SynthesisConfig(max_osl=max_osl)),
        )
        loader = BailianTraceDatasetLoader(
            filename=filename,
            user_config=user_config,
            prompt_generator=mock_prompt_generator,
        )
        dataset = loader.load_dataset()

        actual = [t.output_length for traces in dataset.values() for t in traces]
        assert actual == expected_output_lengths

    # ---- can_load ----

    @pytest.mark.parametrize(
        "data,expected",
        [
            ({"chat_id": 1, "timestamp": 1.0, "input_length": 10, "output_length": 5}, True),
            ({"chat_id": 1, "timestamp": 1.0, "input_length": 10, "output_length": 5, "type": "chat", "turn": 2, "hash_ids": [1]}, True),
            ({"input_length": 10, "hash_ids": [1]}, False),  # Mooncake, not Bailian
            ({"text": "hello"}, False),
            (None, False),
        ],
    )  # fmt: skip
    def test_can_load(self, data, expected):
        assert BailianTraceDatasetLoader.can_load(data=data) is expected

    # ---- convert_to_conversations ----

    @patch("aiperf.dataset.loader.base_trace_loader.parallel_decode")
    def test_convert_to_conversations(
        self, mock_parallel_decode, mock_prompt_generator, default_user_config
    ):
        mock_parallel_decode.return_value = ["decoded prompt 1", "decoded prompt 2"]

        trace_data = {
            "100": [
                BailianTrace(
                    chat_id=100,
                    timestamp=1000.0,
                    input_length=100,
                    output_length=50,
                    hash_ids=[1, 2, 3],
                ),
            ],
            "200": [
                BailianTrace(
                    chat_id=200,
                    timestamp=2000.0,
                    input_length=200,
                    output_length=80,
                    hash_ids=[4, 5],
                ),
            ],
        }

        loader = BailianTraceDatasetLoader(
            filename="dummy.jsonl",
            user_config=default_user_config,
            prompt_generator=mock_prompt_generator,
        )
        conversations = loader.convert_to_conversations(trace_data)

        assert len(conversations) == 2
        assert conversations[0].session_id == "100"
        assert conversations[0].turns[0].timestamp == 1000.0
        assert conversations[0].turns[0].max_tokens == 50

    def test_convert_empty_data(self, mock_prompt_generator, default_user_config):
        loader = BailianTraceDatasetLoader(
            filename="dummy.jsonl",
            user_config=default_user_config,
            prompt_generator=mock_prompt_generator,
        )
        assert loader.convert_to_conversations({}) == []

    def test_convert_without_hash_ids(self, mock_prompt_generator, default_user_config):
        """When hash_ids is empty, falls back to normal prompt generation."""
        trace_data = {
            "1": [
                BailianTrace(
                    chat_id=1,
                    timestamp=1000.0,
                    input_length=100,
                    output_length=50,
                ),
            ],
        }

        loader = BailianTraceDatasetLoader(
            filename="dummy.jsonl",
            user_config=default_user_config,
            prompt_generator=mock_prompt_generator,
        )
        conversations = loader.convert_to_conversations(trace_data)

        assert len(conversations) == 1
        mock_prompt_generator.generate.assert_called_once_with(
            mean=100, stddev=0, hash_ids=[]
        )

    @patch("aiperf.dataset.loader.base_trace_loader.parallel_decode")
    def test_parallel_decode_length_mismatch_raises(
        self, mock_parallel_decode, mock_prompt_generator, default_user_config
    ):
        """strict=True in zip guards against silent data loss."""
        mock_parallel_decode.return_value = ["only one"]  # expecting 2

        trace_data = {
            "1": [
                BailianTrace(
                    chat_id=1,
                    timestamp=1.0,
                    input_length=10,
                    output_length=5,
                    hash_ids=[1],
                )
            ],
            "2": [
                BailianTrace(
                    chat_id=2,
                    timestamp=2.0,
                    input_length=20,
                    output_length=10,
                    hash_ids=[2],
                )
            ],
        }

        loader = BailianTraceDatasetLoader(
            filename="dummy.jsonl",
            user_config=default_user_config,
            prompt_generator=mock_prompt_generator,
        )

        with pytest.raises(ValueError, match="zip"):
            loader.convert_to_conversations(trace_data)

    # ---- multi-turn conversation conversion ----

    @patch("aiperf.dataset.loader.base_trace_loader.parallel_decode")
    def test_multi_turn_conversation_ordering(
        self, mock_parallel_decode, mock_prompt_generator, default_user_config
    ):
        mock_parallel_decode.return_value = [
            "prompt turn 1",
            "prompt turn 2",
            "prompt turn 3",
        ]

        trace_data = {
            "100": [
                BailianTrace(
                    chat_id=100,
                    timestamp=1000.0,
                    input_length=50,
                    output_length=20,
                    turn=1,
                    hash_ids=[1],
                ),
                BailianTrace(
                    chat_id=101,
                    parent_chat_id=100,
                    timestamp=2000.0,
                    input_length=60,
                    output_length=25,
                    turn=2,
                    hash_ids=[2],
                ),
                BailianTrace(
                    chat_id=102,
                    parent_chat_id=101,
                    timestamp=3000.0,
                    input_length=70,
                    output_length=30,
                    turn=3,
                    hash_ids=[3],
                ),
            ],
        }

        loader = BailianTraceDatasetLoader(
            filename="dummy.jsonl",
            user_config=default_user_config,
            prompt_generator=mock_prompt_generator,
        )
        conversations = loader.convert_to_conversations(trace_data)

        assert len(conversations) == 1
        conv = conversations[0]
        assert len(conv.turns) == 3
        assert conv.turns[0].timestamp == 1000.0
        assert conv.turns[1].timestamp == 2000.0
        assert conv.turns[2].timestamp == 3000.0


# ============================================================================
# Synthesis Integration Tests
# ============================================================================


def _make_synthesis_config(
    speedup_ratio: float = 1.0,
    prefix_len_multiplier: float = 1.0,
    max_isl: int | None = None,
    block_size: int = 16,
) -> UserConfig:
    return UserConfig(
        endpoint=EndpointConfig(model_names=["test-model"]),
        input=InputConfig(
            synthesis=SynthesisConfig(
                speedup_ratio=speedup_ratio,
                prefix_len_multiplier=prefix_len_multiplier,
                max_isl=max_isl,
            ),
            prompt=PromptConfig(
                input_tokens=InputTokensConfig(block_size=block_size),
            ),
        ),
    )


class TestBailianTraceSynthesisIntegration:
    @pytest.fixture
    def mock_prompt_generator(self):
        generator = Mock()
        generator.generate.return_value = "Generated prompt text"
        generator._decoded_cache = {}
        generator._build_token_sequence.return_value = [1, 2, 3, 4, 5]
        return generator

    def test_speedup_ratio_scales_timestamps(self, mock_prompt_generator):
        data = {
            "1": [
                BailianTrace(
                    chat_id=1,
                    timestamp=1000.0,
                    input_length=16,
                    output_length=10,
                    hash_ids=[1],
                ),
                BailianTrace(
                    chat_id=2,
                    parent_chat_id=1,
                    timestamp=2000.0,
                    input_length=16,
                    output_length=10,
                    turn=2,
                    hash_ids=[2],
                ),
            ],
        }
        user_config = _make_synthesis_config(speedup_ratio=2.0)

        loader = BailianTraceDatasetLoader(
            filename="dummy.jsonl",
            user_config=user_config,
            prompt_generator=mock_prompt_generator,
        )
        result = loader._apply_synthesis(data)

        assert result["1"][0].timestamp == 500.0
        assert result["1"][1].timestamp == 1000.0

    def test_synthesis_preserves_session_structure(self, mock_prompt_generator):
        data = {
            "1": [
                BailianTrace(
                    chat_id=1,
                    timestamp=1000.0,
                    input_length=16,
                    output_length=10,
                    hash_ids=[1],
                ),
            ],
            "2": [
                BailianTrace(
                    chat_id=2,
                    timestamp=2000.0,
                    input_length=16,
                    output_length=10,
                    hash_ids=[2],
                ),
            ],
        }
        user_config = _make_synthesis_config(speedup_ratio=2.0)

        loader = BailianTraceDatasetLoader(
            filename="dummy.jsonl",
            user_config=user_config,
            prompt_generator=mock_prompt_generator,
        )
        result = loader._apply_synthesis(data)

        assert set(result.keys()) == {"1", "2"}
        assert len(result["1"]) == 1
        assert len(result["2"]) == 1

    def test_synthesis_returns_bailian_trace_objects(self, mock_prompt_generator):
        data = {
            "1": [
                BailianTrace(
                    chat_id=1,
                    timestamp=1000.0,
                    input_length=16,
                    output_length=10,
                    hash_ids=[1],
                ),
            ],
        }
        user_config = _make_synthesis_config(speedup_ratio=2.0)

        loader = BailianTraceDatasetLoader(
            filename="dummy.jsonl",
            user_config=user_config,
            prompt_generator=mock_prompt_generator,
        )
        result = loader._apply_synthesis(data)

        for traces in result.values():
            for trace in traces:
                assert isinstance(trace, BailianTrace)

    def test_empty_input(self, mock_prompt_generator):
        user_config = _make_synthesis_config(speedup_ratio=2.0)
        loader = BailianTraceDatasetLoader(
            filename="dummy.jsonl",
            user_config=user_config,
            prompt_generator=mock_prompt_generator,
        )
        assert loader._apply_synthesis({}) == {}

    def test_end_to_end_with_synthesis(self, create_jsonl_file, mock_prompt_generator):
        content = [
            '{"chat_id": 1, "timestamp": 1.0, "input_length": 16, "output_length": 10, "hash_ids": [1]}',
            '{"chat_id": 2, "timestamp": 2.0, "input_length": 16, "output_length": 10, "hash_ids": [2]}',
        ]
        filename = create_jsonl_file(content)

        user_config = _make_synthesis_config(speedup_ratio=2.0)
        loader = BailianTraceDatasetLoader(
            filename=filename,
            user_config=user_config,
            prompt_generator=mock_prompt_generator,
        )
        dataset = loader.load_dataset()

        traces = [t for ts in dataset.values() for t in ts]
        # Timestamps: 1.0s → 1000ms, 2.0s → 2000ms, then /2 = 500ms, 1000ms
        assert traces[0].timestamp == 500.0
        assert traces[1].timestamp == 1000.0
