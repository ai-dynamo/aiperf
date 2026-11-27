# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from pytest import param

from aiperf.common.models import Text, Turn
from aiperf.endpoints.base_endpoint import BaseEndpoint
from aiperf.endpoints.cohere_rankings import CohereRankingsEndpoint
from aiperf.endpoints.hf_tei_rankings import HFTeiRankingsEndpoint
from aiperf.endpoints.huggingface_generate import HuggingFaceGenerateEndpoint
from aiperf.endpoints.nim_embeddings import NIMEmbeddingsEndpoint
from aiperf.endpoints.nim_image_retrieval import ImageRetrievalEndpoint
from aiperf.endpoints.nim_rankings import NIMRankingsEndpoint
from aiperf.endpoints.openai_audio_transcription import AudioTranscriptionEndpoint
from aiperf.endpoints.openai_completions import CompletionsEndpoint
from aiperf.endpoints.openai_embeddings import EmbeddingsEndpoint
from aiperf.endpoints.openai_image_edit import ImageEditEndpoint
from aiperf.endpoints.openai_image_generation import ImageGenerationEndpoint
from aiperf.endpoints.openai_video_generation import VideoGenerationEndpoint
from aiperf.endpoints.solido_rag import SolidoEndpoint
from aiperf.endpoints.template_endpoint import TemplateEndpoint
from aiperf.plugin.enums import EndpointType
from tests.unit.endpoints.conftest import (
    create_endpoint_with_mock_transport,
    create_model_endpoint,
    create_request_info,
)

_TEMPLATE_EXTRA = [("payload_template", '{"text": {{ texts|tojson }}}')]

TEXT_ONLY_ENDPOINTS = [
    param(CompletionsEndpoint, EndpointType.COMPLETIONS, None, id="completions"),
    param(EmbeddingsEndpoint, EndpointType.EMBEDDINGS, None, id="embeddings"),
    param(NIMEmbeddingsEndpoint, EndpointType.NIM_EMBEDDINGS, None, id="nim_embeddings"),
    param(CohereRankingsEndpoint, EndpointType.COHERE_RANKINGS, None, id="cohere_rankings"),
    param(HFTeiRankingsEndpoint, EndpointType.HF_TEI_RANKINGS, None, id="hf_tei_rankings"),
    param(NIMRankingsEndpoint, EndpointType.NIM_RANKINGS, None, id="nim_rankings"),
    param(
        HuggingFaceGenerateEndpoint,
        EndpointType.HUGGINGFACE_GENERATE,
        None,
        id="huggingface_generate",
    ),
    param(SolidoEndpoint, EndpointType.SOLIDO_RAG, None, id="solido_rag"),
    param(ImageRetrievalEndpoint, EndpointType.IMAGE_RETRIEVAL, None, id="image_retrieval"),
    param(
        AudioTranscriptionEndpoint,
        EndpointType.AUDIO_TRANSCRIPTION,
        None,
        id="audio_transcription",
    ),
    param(
        ImageGenerationEndpoint, EndpointType.IMAGE_GENERATION, None, id="image_generation"
    ),
    param(ImageEditEndpoint, EndpointType.IMAGE_EDIT, None, id="image_edit"),
    param(
        VideoGenerationEndpoint, EndpointType.VIDEO_GENERATION, None, id="video_generation"
    ),
    param(TemplateEndpoint, EndpointType.TEMPLATE, _TEMPLATE_EXTRA, id="template"),
]  # fmt: skip


class TestTextOnlyEndpointsRejectRawMessages:
    """Endpoints that never read turn.raw_messages must fail instead of dropping it."""

    @pytest.mark.parametrize(
        ("endpoint_class", "endpoint_type", "extra"), TEXT_ONLY_ENDPOINTS
    )
    def test_format_payload_raw_messages_row_raises_naming_endpoint(
        self,
        endpoint_class: type[BaseEndpoint],
        endpoint_type: EndpointType,
        extra: list[tuple[str, str]] | None,
    ) -> None:
        model_endpoint = create_model_endpoint(endpoint_type, extra=extra)
        endpoint = create_endpoint_with_mock_transport(endpoint_class, model_endpoint)
        turn = Turn(raw_messages=[{"role": "user", "content": "authored message"}])
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        with pytest.raises(
            ValueError, match=rf"{endpoint_type} endpoint does not support raw_messages"
        ):
            endpoint.format_payload(request_info)

    def test_format_payload_text_only_completions_row_still_sends_prompt(self) -> None:
        model_endpoint = create_model_endpoint(EndpointType.COMPLETIONS)
        endpoint = create_endpoint_with_mock_transport(
            CompletionsEndpoint, model_endpoint
        )
        turn = Turn(texts=[Text(contents=["Once upon a time"])])
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert payload["prompt"] == "Once upon a time"

    def test_format_payload_empty_raw_messages_completions_row_sends_text(
        self,
    ) -> None:
        model_endpoint = create_model_endpoint(EndpointType.COMPLETIONS)
        endpoint = create_endpoint_with_mock_transport(
            CompletionsEndpoint, model_endpoint
        )
        turn = Turn(texts=[Text(contents=["Once upon a time"])], raw_messages=[])
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert payload["prompt"] == "Once upon a time"
