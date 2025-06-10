#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

import io
from pathlib import Path
from unittest.mock import patch

import pytest

from aiperf.common.tokenizer import Tokenizer
from aiperf.services.dataset.composer import PayloadInputRetriever

# TODO: temporary
from aiperf.services.dataset.composer.file_input_retriever import (
    CustomDataConfig,
    SyntheticTokensConfig,
    TurnDelayConfig,
)
from aiperf.services.dataset.composer.generic_dataset import (
    DataRow,
    FileData,
    GenericDataset,
)
from aiperf.services.dataset.generator import PromptGenerator


class TestPayloadInputRetriever:
    @pytest.fixture
    def retriever(self) -> PayloadInputRetriever:
        config = CustomDataConfig(
            filename=Path("test_input.jsonl"),
            tokenizer=Tokenizer(),
            synthetic_tokens=SyntheticTokensConfig(mean=10, stddev=2),
            turn_delay=TurnDelayConfig(ratio=1.0),
        )
        return PayloadInputRetriever(config)

    @pytest.mark.parametrize(
        "input_data, expected_prompts, expected_payload_metadata_list, expected_optional_data_list",
        [
            (
                '{"text": "What is AI?", "timestamp": 123, "session_id": "abc"}\n'
                '{"text": "How does ML work?", "timestamp": 0, "custom_field": "value"}\n',
                ["What is AI?", "How does ML work?"],
                [{"timestamp": 123, "session_id": "abc"}, {"timestamp": 0}],
                [{}, {"custom_field": "value"}],
            ),
            (
                '{"text": "What is AI?", "timestamp": 123, "session_id": "abc"}',
                ["What is AI?"],
                [{"timestamp": 123, "session_id": "abc"}],
                [{}],
            ),
            ('{"timestamp": 789}\n', ["Synthetic prompt"], [{"timestamp": 789}], [{}]),
            (
                '{"text": "What is AI?", "delay": 123, "session_id": "abc"}\n'
                '{"text": "How does ML work?", "delay": 0, "custom_field": "value"}\n',
                ["What is AI?", "How does ML work?"],
                [{"delay": 123, "session_id": "abc"}, {"delay": 0}],
                [{}, {"custom_field": "value"}],
            ),
        ],
    )
    @patch("builtins.open")
    @patch.object(PromptGenerator, "create_synthetic_prompt")
    def test_get_content_from_input_file(
        self,
        mock_synthetic_prompt,
        mock_file,
        retriever,
        input_data,
        expected_prompts,
        expected_payload_metadata_list,
        expected_optional_data_list,
    ):
        mock_file.return_value = io.StringIO(input_data)
        mock_synthetic_prompt.return_value = "Synthetic prompt"

        data_dict = retriever._get_content_from_input_file(Path("test_input.jsonl"))

        assert data_dict["prompts"] == expected_prompts
        assert data_dict["payload_metadata_list"] == expected_payload_metadata_list
        assert data_dict["optional_data_list"] == expected_optional_data_list

        if "text" not in input_data:
            mock_synthetic_prompt.assert_called_once()

    def test_convert_content_to_data_file(self, retriever):
        prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]
        optional_data_list = [{"session_id": "123"}, {"custom_field": "value"}, {}]
        payload_metadata_list = [{"timestamp": 1}, {"timestamp": 2}, {"delay": 3}]
        data_dict = {
            "prompts": prompts,
            "payload_metadata_list": payload_metadata_list,
            "optional_data_list": optional_data_list,
        }

        file_data = retriever._convert_content_to_data_file(data_dict)

        assert len(file_data.rows) == 3
        assert file_data.rows[0].texts == ["Prompt 1"]
        assert file_data.rows[0].payload_metadata
        assert file_data.rows[0].payload_metadata.get("timestamp")
        assert file_data.rows[0].payload_metadata.get("timestamp") == 1
        assert file_data.rows[0].optional_data == {"session_id": "123"}
        assert file_data.rows[1].texts == ["Prompt 2"]
        assert file_data.rows[1].payload_metadata
        assert file_data.rows[1].payload_metadata.get("timestamp")
        assert file_data.rows[1].payload_metadata.get("timestamp") == 2
        assert file_data.rows[1].optional_data == {"custom_field": "value"}
        assert file_data.rows[2].texts == ["Prompt 3"]
        assert file_data.rows[2].payload_metadata
        assert file_data.rows[2].payload_metadata.get("delay")
        assert file_data.rows[2].payload_metadata.get("delay") == 3

    @pytest.mark.parametrize(
        "payload_metadata",
        [
            {"timestamp": 123, "session_id": "abc"},
            {"delay": 456, "session_id": "def"},
        ],
    )
    @patch.object(PayloadInputRetriever, "_get_input_dataset_from_file")
    def test_retrieve_data(self, mock_get_input, retriever, payload_metadata):
        mock_file_data = FileData(
            [
                DataRow(
                    texts=["Test prompt"],
                    optional_data={"key": "value"},
                    payload_metadata=payload_metadata,
                )
            ]
        )
        mock_get_input.return_value = mock_file_data

        dataset = retriever.retrieve_data()

        assert isinstance(dataset, GenericDataset)
        assert str(retriever.config.filename) in dataset.files_data
        assert dataset.files_data[str(retriever.config.filename)] == mock_file_data

    @pytest.mark.parametrize(
        "delay, delay_ratio, expected_delay",
        [
            (1000, 1.0, 1000),  # No change with ratio 1.0
            (1000, 0.5, 500),  # Half the delay
            (1000, 2.0, 2000),  # Double the delay
            (0, 1.5, 0),  # Zero delay stays zero
        ],
    )
    def test_delay_ratio_application(
        self, retriever, delay, delay_ratio, expected_delay
    ):
        # Set the delay ratio in the config
        retriever.config.turn_delay.ratio = delay_ratio

        # Create test data with a delay
        data = {"delay": delay}

        # Get the payload metadata
        metadata = retriever._get_payload_metadata(data)

        # Check that the delay was properly scaled
        assert metadata["delay"] == expected_delay
