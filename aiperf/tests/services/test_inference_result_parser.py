# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from aiperf.common.config.service_config import ServiceConfig
from aiperf.common.enums import CommunicationBackend, ServiceRunType
from aiperf.common.record_models import RequestRecord
from aiperf.common.tokenizer import Tokenizer
from aiperf.services.inference_result_parser.inference_result_parser import (
    InferenceResultParser,
)


def service_config():
    return ServiceConfig(
        service_run_type=ServiceRunType.MULTIPROCESSING,
        comm_backend=CommunicationBackend.ZMQ_TCP,
    )


@pytest.mark.asyncio
async def test_input_token_count_chat_completion_api():
    parser = InferenceResultParser(service_config=service_config())
    tokenizer = Tokenizer.from_pretrained("gpt2")

    request = RequestRecord(
        request={
            "messages": [
                {"role": "user", "content": "Hello, how are you?"},
                {"role": "assistant", "content": "I'm good, thank you!"},
            ]
        }
    )

    expected = len(tokenizer.encode("Hello, how are you? I'm good, thank you!"))
    actual = await parser._compute_input_token_count(request, tokenizer)
    assert actual == expected


@pytest.mark.asyncio
async def test_input_token_count_completion_api():
    parser = InferenceResultParser(service_config=service_config())
    tokenizer = Tokenizer.from_pretrained("gpt2")

    prompt = "The universe is vast and mysterious."
    request = RequestRecord(request={"prompt": prompt})
    expected = len(tokenizer.encode(prompt))
    assert await parser._compute_input_token_count(request, tokenizer) == expected

    prompt_list = ["The quick", "brown fox"]
    request = RequestRecord(request={"prompt": prompt_list})
    expected = len(tokenizer.encode(" ".join(prompt_list)))
    assert await parser._compute_input_token_count(request, tokenizer) == expected


@pytest.mark.asyncio
async def test_input_token_count_embedding_api():
    parser = InferenceResultParser(service_config=service_config())
    tokenizer = Tokenizer.from_pretrained("gpt2")

    input_text = "Encode this sentence."
    request = RequestRecord(request={"input": input_text})
    expected = len(tokenizer.encode(input_text))
    assert await parser._compute_input_token_count(request, tokenizer) == expected

    input_list = ["Encode", "this", "sentence"]
    request = RequestRecord(request={"input": input_list})
    expected = len(tokenizer.encode(" ".join(input_list)))
    assert await parser._compute_input_token_count(request, tokenizer) == expected


@pytest.mark.asyncio
async def test_input_token_count_response_api():
    parser = InferenceResultParser(service_config=service_config())
    tokenizer = Tokenizer.from_pretrained("gpt2")

    inputs_text = "Please respond to this prompt."
    request = RequestRecord(request={"inputs": inputs_text})
    expected = len(tokenizer.encode(inputs_text))
    assert await parser._compute_input_token_count(request, tokenizer) == expected


@pytest.mark.asyncio
async def test_input_token_count_chat_completion_chunk_api():
    parser = InferenceResultParser(service_config=service_config())
    tokenizer = Tokenizer.from_pretrained("gpt2")

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me a joke."},
    ]
    expected = len(tokenizer.encode(" ".join(m["content"] for m in messages)))

    request = RequestRecord(request={"messages": messages})
    assert await parser._compute_input_token_count(request, tokenizer) == expected
