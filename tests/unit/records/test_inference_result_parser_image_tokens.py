# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for image token counting in InferenceResultParser."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aiperf.common.models import ParsedResponse, RequestRecord, TextResponseData, Usage
from aiperf.common.models.dataset_models import Image, Text, Turn
from aiperf.common.models.modality_token_counts import ModalityTokenCounts
from tests.unit.records.conftest import create_test_request_info


def make_parsed_response(
    text: str = "response",
    perf_ns: int = 1000,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    include_usage: bool = True,
) -> ParsedResponse:
    """Create a ParsedResponse with optional usage data."""
    usage = None
    if include_usage and (prompt_tokens is not None or completion_tokens is not None):
        usage_data = {}
        if prompt_tokens is not None:
            usage_data["prompt_tokens"] = prompt_tokens
        if completion_tokens is not None:
            usage_data["completion_tokens"] = completion_tokens
        usage = Usage(usage_data)
    return ParsedResponse(
        perf_ns=perf_ns,
        data=TextResponseData(text=text),
        usage=usage,
    )


def setup_parser_responses(parser, responses):
    """Configure the parser's endpoint to return the given responses."""
    parser.endpoint.extract_response_data = MagicMock(return_value=responses)


def create_turn_with_images_and_text(
    num_images: int = 1,
    texts: list[str] | None = None,
) -> Turn:
    """Create a turn with both images and text."""
    texts = texts or ["Hello world"]
    return Turn(
        role="user",
        texts=[Text(name="prompt", contents=texts)],
        images=[
            Image(
                name="img",
                contents=[f"data:image/png;base64,fake_{i}" for i in range(num_images)],
            )
        ],
    )


@pytest.fixture
def spy_tokenizer():
    """Tokenizer mock that returns word count as token count."""
    tokenizer = MagicMock()
    tokenizer.encode.side_effect = lambda x: list(range(len(x.split())))
    return tokenizer


@pytest.fixture
def image_turn():
    """Turn with one image and 'Hello world' text."""
    return create_turn_with_images_and_text()


@pytest.fixture
def text_only_turn():
    """Turn with only text, no images."""
    return Turn(
        role="user",
        texts=[Text(name="prompt", contents=["Hello world"])],
    )


class TestRequestHasImages:
    def test_turn_with_images_detected(self, setup_inference_parser, image_turn):
        """_request_has_images returns True when images present."""
        record = RequestRecord(
            request_info=create_test_request_info(turns=[image_turn]),
            model_name="test-model",
            turns=[image_turn],
        )
        assert setup_inference_parser._request_has_images(record) is True

    def test_turn_without_images_not_detected(
        self, setup_inference_parser, text_only_turn
    ):
        """_request_has_images returns False when no images."""
        record = RequestRecord(
            request_info=create_test_request_info(turns=[text_only_turn]),
            model_name="test-model",
            turns=[text_only_turn],
        )
        assert setup_inference_parser._request_has_images(record) is False

    def test_empty_image_contents_not_detected(self, setup_inference_parser):
        """Empty image contents are not counted as images."""
        turn = Turn(
            role="user",
            texts=[Text(name="prompt", contents=["Hello"])],
            images=[Image(name="empty", contents=[])],
        )
        record = RequestRecord(
            request_info=create_test_request_info(turns=[turn]),
            model_name="test-model",
            turns=[turn],
        )
        assert setup_inference_parser._request_has_images(record) is False

    def test_multiple_turns_mixed(
        self, setup_inference_parser, text_only_turn, image_turn
    ):
        """Images detected across multiple turns."""
        record = RequestRecord(
            request_info=create_test_request_info(turns=[text_only_turn, image_turn]),
            model_name="test-model",
            turns=[text_only_turn, image_turn],
        )
        assert setup_inference_parser._request_has_images(record) is True


@pytest.mark.asyncio
class TestImageTokenCounting:
    """Tests for image+text token counting via modality breakdown."""

    @pytest.fixture(autouse=True)
    def _enable_tokenize_input(self, setup_inference_parser):
        setup_inference_parser.tokenize_input = True

    async def test_images_with_precomputed_estimates_scales_to_server_total(
        self, setup_inference_parser, spy_tokenizer, image_turn
    ):
        """When pre-computed modality estimates exist, they are scaled to server total."""
        setup_inference_parser.get_tokenizer = AsyncMock(return_value=spy_tokenizer)

        record = RequestRecord(
            request_info=create_test_request_info(turns=[image_turn]),
            model_name="test-model",
            turns=[image_turn],
            input_modalities_local=ModalityTokenCounts(text=120, image=480),
        )
        responses = [make_parsed_response(prompt_tokens=650, completion_tokens=50)]
        setup_parser_responses(setup_inference_parser, responses)

        result = await setup_inference_parser.process_valid_record(record)

        assert result.token_counts.input == 650
        assert result.token_counts.input_modalities_local == ModalityTokenCounts(
            text=120, image=480
        )
        # Scaled: text=round(650*120/600)=130, image=650-130=520
        assert result.token_counts.input_modalities.text == 130
        assert result.token_counts.input_modalities.image == 520
        assert (
            result.token_counts.input_modalities.text
            + result.token_counts.input_modalities.image
            == 650
        )

    async def test_images_without_estimates_uses_subtraction_fallback(
        self, setup_inference_parser, spy_tokenizer, image_turn
    ):
        """When no pre-computed estimates, falls back to subtraction approach."""
        setup_inference_parser.get_tokenizer = AsyncMock(return_value=spy_tokenizer)

        record = RequestRecord(
            request_info=create_test_request_info(turns=[image_turn]),
            model_name="test-model",
            turns=[image_turn],
        )
        responses = [make_parsed_response(prompt_tokens=1200, completion_tokens=50)]
        setup_parser_responses(setup_inference_parser, responses)

        result = await setup_inference_parser.process_valid_record(record)

        # input = server's prompt_tokens (1200)
        assert result.token_counts.input == 1200
        # Subtraction fallback: text=2 (word count), image=1200-2=1198
        assert result.token_counts.input_modalities.text == 2
        assert result.token_counts.input_modalities.image == 1198

    async def test_no_images_no_modality_breakdown(
        self, setup_inference_parser, spy_tokenizer, text_only_turn
    ):
        """When no images are present, no modality breakdown is computed."""
        setup_inference_parser.get_tokenizer = AsyncMock(return_value=spy_tokenizer)

        record = RequestRecord(
            request_info=create_test_request_info(turns=[text_only_turn]),
            model_name="test-model",
            turns=[text_only_turn],
        )
        responses = [make_parsed_response(prompt_tokens=999, completion_tokens=50)]
        setup_parser_responses(setup_inference_parser, responses)

        result = await setup_inference_parser.process_valid_record(record)

        assert result.token_counts.input == 999
        assert result.token_counts.input_modalities is None
        assert result.token_counts.input_modalities_local is None

    async def test_images_no_server_usage_no_modality_breakdown(
        self, setup_inference_parser, spy_tokenizer, image_turn
    ):
        """When images present but server has no usage, modality breakdown is unavailable."""
        setup_inference_parser.get_tokenizer = AsyncMock(return_value=spy_tokenizer)

        record = RequestRecord(
            request_info=create_test_request_info(turns=[image_turn]),
            model_name="test-model",
            turns=[image_turn],
        )
        responses = [make_parsed_response(include_usage=False)]
        setup_parser_responses(setup_inference_parser, responses)

        result = await setup_inference_parser.process_valid_record(record)

        # No server total, so modality breakdown is unavailable
        assert result.token_counts.input_modalities is None

    async def test_images_no_client_text_count_no_modality_breakdown(
        self, setup_inference_parser, image_turn
    ):
        """When images present and client text tokenization fails, modality breakdown unavailable."""
        setup_inference_parser.compute_input_token_count = AsyncMock(return_value=None)
        setup_inference_parser.get_tokenizer = AsyncMock(return_value=MagicMock())

        record = RequestRecord(
            request_info=create_test_request_info(turns=[image_turn]),
            model_name="test-model",
            turns=[image_turn],
        )
        responses = [make_parsed_response(prompt_tokens=1200, completion_tokens=50)]
        setup_parser_responses(setup_inference_parser, responses)

        result = await setup_inference_parser.process_valid_record(record)

        assert result.token_counts.input == 1200
        assert result.token_counts.input_modalities is None

    async def test_image_tokens_clamped_to_zero(
        self, setup_inference_parser, spy_tokenizer
    ):
        """Image tokens are clamped to 0 when server reports fewer tokens than client text count."""
        turn = create_turn_with_images_and_text(
            num_images=1,
            texts=["word " * 100],
        )
        setup_inference_parser.get_tokenizer = AsyncMock(return_value=spy_tokenizer)

        record = RequestRecord(
            request_info=create_test_request_info(turns=[turn]),
            model_name="test-model",
            turns=[turn],
        )
        responses = [make_parsed_response(prompt_tokens=50, completion_tokens=10)]
        setup_parser_responses(setup_inference_parser, responses)

        with patch.object(setup_inference_parser, "warning") as mock_warning:
            result = await setup_inference_parser.process_valid_record(record)

            mock_warning.assert_called_once()
            call_args = mock_warning.call_args[0][0]
            assert "Clamping" in call_args

        assert result.token_counts.input == 50
        assert result.token_counts.input_modalities.text == 100
        assert result.token_counts.input_modalities.image == 0

    async def test_multiple_images_across_turns(
        self, setup_inference_parser, spy_tokenizer
    ):
        """Multiple images and text across multiple turns."""
        turn1 = create_turn_with_images_and_text(num_images=2, texts=["First turn"])
        turn2 = create_turn_with_images_and_text(num_images=1, texts=["Second turn"])
        setup_inference_parser.get_tokenizer = AsyncMock(return_value=spy_tokenizer)

        record = RequestRecord(
            request_info=create_test_request_info(turns=[turn1, turn2]),
            model_name="test-model",
            turns=[turn1, turn2],
        )
        responses = [make_parsed_response(prompt_tokens=5000, completion_tokens=50)]
        setup_parser_responses(setup_inference_parser, responses)

        result = await setup_inference_parser.process_valid_record(record)

        assert result.token_counts.input == 5000
        # "First turn" (2 words) + " " + "Second turn" (2 words) = 4 words
        assert result.token_counts.input_modalities.text == 4
        assert result.token_counts.input_modalities.image == 5000 - 4

    async def test_precomputed_estimates_without_server_total(
        self, setup_inference_parser, spy_tokenizer, image_turn
    ):
        """When pre-computed estimates exist but no server total, local is set but scaled is None."""
        setup_inference_parser.get_tokenizer = AsyncMock(return_value=spy_tokenizer)

        estimates = ModalityTokenCounts(text=120, image=480)
        record = RequestRecord(
            request_info=create_test_request_info(turns=[image_turn]),
            model_name="test-model",
            turns=[image_turn],
            input_modalities_local=estimates,
        )
        responses = [make_parsed_response(include_usage=False)]
        setup_parser_responses(setup_inference_parser, responses)

        result = await setup_inference_parser.process_valid_record(record)

        assert result.token_counts.input_modalities_local == estimates
        assert result.token_counts.input_modalities is None

    async def test_output_and_reasoning_unaffected(
        self, setup_inference_parser, spy_tokenizer, image_turn
    ):
        """Output and reasoning tokens are unaffected by image token counting."""
        setup_inference_parser.get_tokenizer = AsyncMock(return_value=spy_tokenizer)

        record = RequestRecord(
            request_info=create_test_request_info(turns=[image_turn]),
            model_name="test-model",
            turns=[image_turn],
        )
        responses = [
            make_parsed_response(
                text="output word three", prompt_tokens=1000, completion_tokens=50
            )
        ]
        setup_parser_responses(setup_inference_parser, responses)

        result = await setup_inference_parser.process_valid_record(record)

        assert result.token_counts.output == 50
        assert result.token_counts.reasoning is None
