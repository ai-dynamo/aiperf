#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0


from typing import Annotated, Any

from pydantic import BeforeValidator, Field

from aiperf.common.models.config.audio_config import AudioConfig
from aiperf.common.models.config.base_config import BaseConfig
from aiperf.common.models.config.config_defaults import InputDefaults
from aiperf.common.models.config.config_validators import parse_file, parse_goodput


class InputConfig(BaseConfig):
    """
    A configuration class for defining input related settings.
    """

    batch_size: Annotated[
        int,
        Field(
            default=InputDefaults.BATCH_SIZE,
            description="The batch size of text requests GenAI-Perf should send.\
            \nThis is currently supported with the embeddings and rankings endpoint types",
        ),
    ]
    extra: Annotated[
        Any,
        Field(
            default=InputDefaults.EXTRA,
            description="Provide additional inputs to include with every request.\
            \nInputs should be in an 'input_name:value' format.",
        ),
    ]
    goodput: Annotated[
        dict[str, Any],
        Field(
            default=InputDefaults.GOODPUT,
            description="An option to provide constraints in order to compute goodput.\
            \nSpecify goodput constraints as 'key:value' pairs,\
            \nwhere the key is a valid metric name, and the value is a number representing\
            \neither milliseconds or a throughput value per second.\
            \nFor example:\
            \n  request_latency:300\
            \n  output_token_throughput_per_user:600",
        ),
        BeforeValidator(parse_goodput),
    ]
    header: Annotated[
        Any,
        Field(
            default=InputDefaults.HEADER,
            description="Adds a custom header to the requests.\
            \nHeaders must be specified as 'Header:Value' pairs.",
        ),
    ]
    file: Annotated[
        Any,
        Field(
            default=InputDefaults.FILE,
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
    ]
    num_dataset_entries: Annotated[
        int,
        Field(
            default=InputDefaults.NUM_DATASET_ENTRIES,
            ge=1,
            description="The number of unique payloads to sample from.\
            \nThese will be reused until benchmarking is complete.",
        ),
    ]
    random_seed: Annotated[
        int,
        Field(
            default=InputDefaults.RANDOM_SEED,
            description="The seed used to generate random values.",
        ),
    ]

    audio: AudioConfig = AudioConfig()
    # image = ConfigImage()
    # output_tokens = ConfigOutputTokens()
    # synthetic_tokens = ConfigSyntheticTokens()
    # prefix_prompt = ConfigPrefixPrompt()
    # sessions = ConfigSessions()
