# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Annotated, Any

import cyclopts
from pydantic import BeforeValidator, Field

from aiperf.common.config.base_config import BaseConfig
from aiperf.common.config.config_defaults import InputDefaults
from aiperf.common.config.config_validators import (
    parse_file,
    parse_goodput,
    parse_str_or_dict,
)
from aiperf.common.config.input.audio_config import AudioConfig
from aiperf.common.config.input.image_config import ImageConfig
from aiperf.common.config.input.prompt_config import PromptConfig
from aiperf.common.config.input.sessions_config import SessionsConfig


class InputConfig(BaseConfig):
    """
    A configuration class for defining input related settings.
    """

    batch_size: Annotated[
        int,
        Field(
            description="The batch size of text requests GenAI-Perf should send.\n"
            "This is currently supported with the embeddings and rankings endpoint types",
        ),
        cyclopts.Parameter(
            name=("--batch-size"),
        ),
    ] = InputDefaults.BATCH_SIZE

    extra: Annotated[
        dict[str, str] | None,
        Field(
            description="Provide additional inputs to include with every request.\n"
            "Inputs should be in an 'input_name:value' format.",
        ),
        cyclopts.Parameter(
            name=("--extra"),
        ),
        BeforeValidator(parse_str_or_dict),
    ] = InputDefaults.EXTRA

    goodput: Annotated[
        dict[str, Any],
        Field(
            description="An option to provide constraints in order to compute goodput.\n"
            "Specify goodput constraints as 'key:value' pairs,\n"
            "where the key is a valid metric name, and the value is a number representing\n"
            "either milliseconds or a throughput value per second.\n"
            "For example:\n"
            "  request_latency:300\n"
            "  output_token_throughput_per_user:600",
        ),
        cyclopts.Parameter(
            name=("--goodput"),
        ),
        BeforeValidator(parse_goodput),
    ] = InputDefaults.GOODPUT

    headers: Annotated[
        dict[str, str] | None,
        Field(
            description="Adds a custom header to the requests.\n"
            "Headers must be specified as 'Header:Value' pairs.",
        ),
        BeforeValidator(parse_str_or_dict),
        cyclopts.Parameter(
            name=("--header"),
        ),
    ] = InputDefaults.HEADER

    file: Annotated[
        Any,
        Field(
            description="The file or directory containing the content to use for profiling.\n"
            "Example:\n"
            '  text: "Your prompt here"\n'
            "To use synthetic files for a converter that needs multiple files,\n"
            "prefix the path with 'synthetic:' followed by a comma-separated list of file names.\n"
            "The synthetic filenames should not have extensions.\n"
            "Example:\n"
            "  synthetic: queries,passages",
        ),
        BeforeValidator(parse_file),
        cyclopts.Parameter(
            name=("--file"),
        ),
    ] = InputDefaults.FILE

    num_dataset_entries: Annotated[
        int,
        Field(
            ge=1,
            description="The number of unique payloads to sample from.\n"
            "These will be reused until benchmarking is complete.",
        ),
        cyclopts.Parameter(
            name=("--num-dataset-entries"),
        ),
    ] = InputDefaults.NUM_DATASET_ENTRIES

    random_seed: Annotated[
        int | None,
        Field(
            description="The seed used to generate random values.",
        ),
        cyclopts.Parameter(
            name=("--random-seed"),
        ),
    ] = InputDefaults.RANDOM_SEED

    audio: AudioConfig = AudioConfig()
    image: ImageConfig = ImageConfig()
    prompt: PromptConfig = PromptConfig()
    sessions: SessionsConfig = SessionsConfig()
