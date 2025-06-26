# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Annotated, Any

import cyclopts
from pydantic import BeforeValidator, Field

from aiperf.common.config.base_config import BaseConfig
from aiperf.common.config.config_defaults import InputDefaults
from aiperf.common.config.config_validators import parse_file, parse_goodput
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
            description="The batch size of text requests GenAI-Perf should send.\
            \nThis is currently supported with the embeddings and rankings endpoint types",
        ),
        cyclopts.Parameter(
            name=("--batch-size"),
        ),
    ] = InputDefaults.BATCH_SIZE

    extra: Annotated[
        Any,
        Field(
            description="Provide additional inputs to include with every request.\
            \nInputs should be in an 'input_name:value' format.",
        ),
        cyclopts.Parameter(
            name=("--extra"),
        ),
    ] = InputDefaults.EXTRA

    goodput: Annotated[
        dict[str, Any],
        Field(
            description="An option to provide constraints in order to compute goodput.\
            \nSpecify goodput constraints as 'key:value' pairs,\
            \nwhere the key is a valid metric name, and the value is a number representing\
            \neither milliseconds or a throughput value per second.\
            \nFor example:\
            \n  request_latency:300\
            \n  output_token_throughput_per_user:600",
        ),
        cyclopts.Parameter(
            name=("--goodput"),
        ),
        BeforeValidator(parse_goodput),
    ] = InputDefaults.GOODPUT

    header: Annotated[
        Any,
        Field(
            description="Adds a custom header to the requests.\
            \nHeaders must be specified as 'Header:Value' pairs.",
        ),
        cyclopts.Parameter(
            name=("--header"),
        ),
    ] = InputDefaults.HEADER

    file: Annotated[
        Any,
        Field(
            description="The file or directory containing the content to use for profiling.\
            \nExample:\
            \n  text: \"Your prompt here\"\
            \n\nTo use synthetic files for a converter that needs multiple files,\
            \nprefix the path with 'synthetic:' followed by a comma-separated list of file names.\
            \nThe synthetic filenames should not have extensions.\
            \nExample:\
            \n  synthetic: queries,passages",
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
            description="The number of unique payloads to sample from.\
            \nThese will be reused until benchmarking is complete.",
        ),
        cyclopts.Parameter(
            name=("--num-dataset-entries"),
        ),
    ] = InputDefaults.NUM_DATASET_ENTRIES

    random_seed: Annotated[
        int,
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
