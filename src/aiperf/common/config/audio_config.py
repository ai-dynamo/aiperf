# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Annotated

from pydantic import BeforeValidator, Field

from aiperf.common.config.base_config import BaseConfig
from aiperf.common.config.cli_parameter import CLIParameter
from aiperf.common.config.config_defaults import AudioDefaults
from aiperf.common.config.config_validators import parse_str_or_list_of_positive_values
from aiperf.common.config.groups import Groups
from aiperf.common.enums import AudioFormat


class AudioLengthConfig(BaseConfig):
    """
    A configuration class for defining audio length related settings.
    """

    _CLI_GROUP = Groups.AUDIO_INPUT

    mean: Annotated[
        float,
        Field(
            ge=0,
            description="Mean duration in seconds for synthetically generated audio files. Audio lengths follow a normal distribution "
            "around this mean (±`--audio-length-stddev`). Used when `--audio-batch-size` > 0 for multimodal benchmarking. "
            "Generated audio is random noise with specified sample rate, bit depth, and format.",
        ),
        CLIParameter(
            name=(
                "--audio-length-mean",  # GenAI-Perf
            ),
            group=_CLI_GROUP,
        ),
    ] = AudioDefaults.LENGTH_MEAN

    stddev: Annotated[
        float,
        Field(
            ge=0,
            description="Standard deviation for synthetic audio duration in seconds. Creates variability in audio lengths when > 0, "
            "simulating mixed-duration audio inputs. Durations follow normal distribution. "
            "Set to 0 for uniform audio lengths.",
        ),
        CLIParameter(
            name=(
                "--audio-length-stddev",  # GenAI-Perf
            ),
            group=_CLI_GROUP,
        ),
    ] = AudioDefaults.LENGTH_STDDEV


class AudioConfig(BaseConfig):
    """
    A configuration class for defining audio related settings.
    """

    _CLI_GROUP = Groups.AUDIO_INPUT

    batch_size: Annotated[
        int,
        Field(
            ge=0,
            description="The number of audio inputs to include in each request. Supported with the `chat` endpoint type for multimodal models.",
        ),
        CLIParameter(
            name=(
                "--audio-batch-size",
                "--batch-size-audio",  # GenAI-Perf
            ),
            group=_CLI_GROUP,
        ),
    ] = AudioDefaults.BATCH_SIZE

    length: AudioLengthConfig = AudioLengthConfig()

    format: Annotated[
        AudioFormat,
        Field(
            description="File format for generated audio files. Supports `wav` (uncompressed PCM, larger files) and `mp3` (compressed, smaller files). "
            "Format choice affects file size in multimodal requests but not audio characteristics (sample rate, bit depth, duration).",
        ),
        CLIParameter(
            name=(
                "--audio-format",  # GenAI-Perf
            ),
            group=_CLI_GROUP,
        ),
    ] = AudioDefaults.FORMAT

    depths: Annotated[
        list[int],
        Field(
            min_length=1,
            description="List of audio bit depths in bits to randomly select from when generating audio files. Each audio file is assigned "
            "a random depth from this list. Common values: `8` (low quality), `16` (CD quality), `24` (professional), `32` (high-end). "
            "Specify multiple values (e.g., `--audio-depths 16 24`) for mixed-quality testing.",
        ),
        BeforeValidator(parse_str_or_list_of_positive_values),
        CLIParameter(
            name=(
                "--audio-depths",  # GenAI-Perf
            ),
            group=_CLI_GROUP,
        ),
    ] = AudioDefaults.DEPTHS

    sample_rates: Annotated[
        list[float],
        Field(
            min_length=1,
            description="A list of audio sample rates to randomly select from in kHz.\n"
            "Common sample rates are 16, 44.1, 48, 96, etc.",
        ),
        BeforeValidator(parse_str_or_list_of_positive_values),
        CLIParameter(
            name=(
                "--audio-sample-rates",  # GenAI-Perf
            ),
            group=_CLI_GROUP,
        ),
    ] = AudioDefaults.SAMPLE_RATES

    num_channels: Annotated[
        int,
        Field(
            ge=1,
            le=2,
            description="Number of audio channels for synthetic audio generation. `1` = mono (single channel), `2` = stereo (left/right channels). "
            "Stereo doubles file size but simulates realistic audio for models supporting spatial audio processing. "
            "Most speech models use mono.",
        ),
        CLIParameter(
            name=(
                "--audio-num-channels",  # GenAI-Perf
            ),
            group=_CLI_GROUP,
        ),
    ] = AudioDefaults.NUM_CHANNELS
