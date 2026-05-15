# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Annotated

from pydantic import BeforeValidator, Field, model_validator
from typing_extensions import Self

from aiperf.common.config.base_config import BaseConfig
from aiperf.common.config.cli_parameter import CLIParameter
from aiperf.common.config.config_validators import parse_str_or_list_of_positive_values
from aiperf.common.config.groups import Groups
from aiperf.common.enums import AudioFormat


class AudioLengthConfig(BaseConfig):
    """
    A configuration class for defining audio length related settings.
    """

    mean: Annotated[
        float,
        Field(
            default=0.0,
            ge=0,
            description="Mean duration in seconds for synthetically generated audio files. Audio lengths follow a normal distribution "
            "around this mean (±`--audio-length-stddev`). Used when `--audio-batch-size` > 0 for multimodal benchmarking. "
            "Generated audio is random noise with specified sample rate, bit depth, and format.",
        ),
        CLIParameter(
            name=(
                "--audio-length-mean",  # GenAI-Perf
            ),
            group=Groups.AUDIO_INPUT,
        ),
    ]

    stddev: Annotated[
        float,
        Field(
            default=0.0,
            ge=0,
            description="Standard deviation for synthetic audio duration in seconds. Creates variability in audio lengths when > 0, "
            "simulating mixed-duration audio inputs. Durations follow normal distribution. "
            "Set to 0 for uniform audio lengths.",
        ),
        CLIParameter(
            name=(
                "--audio-length-stddev",  # GenAI-Perf
            ),
            group=Groups.AUDIO_INPUT,
        ),
    ]


class AudioConfig(BaseConfig):
    """
    A configuration class for defining audio related settings.
    """

    batch_size: Annotated[
        int,
        Field(
            default=1,
            ge=0,
            description="The number of audio inputs to include in each request. Supported with the `chat` endpoint type for multimodal models.",
        ),
        CLIParameter(
            name=(
                "--audio-batch-size",
                "--batch-size-audio",  # GenAI-Perf
            ),
            group=Groups.AUDIO_INPUT,
        ),
    ]

    length: Annotated[
        AudioLengthConfig,
        Field(
            default_factory=AudioLengthConfig,
            description="Duration distribution for synthetic audio samples (mean and stddev in seconds).",
        ),
    ]

    format: Annotated[
        AudioFormat,
        Field(
            default=AudioFormat.WAV,
            description="File format for generated audio files. Supports `wav` (uncompressed PCM, larger files) and `mp3` (compressed, smaller files). "
            "Format choice affects file size in multimodal requests but not audio characteristics (sample rate, bit depth, duration).",
        ),
        CLIParameter(
            name=(
                "--audio-format",  # GenAI-Perf
            ),
            group=Groups.AUDIO_INPUT,
        ),
    ]

    depths: Annotated[
        list[int],
        Field(
            default=[16],
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
            group=Groups.AUDIO_INPUT,
        ),
    ]

    sample_rates: Annotated[
        list[float],
        Field(
            default=[16.0],
            min_length=1,
            description="A list of audio sample rates to randomly select from in kHz.\n"
            "Common sample rates are 16, 44.1, 48, 96, etc.",
        ),
        BeforeValidator(parse_str_or_list_of_positive_values),
        CLIParameter(
            name=(
                "--audio-sample-rates",  # GenAI-Perf
            ),
            group=Groups.AUDIO_INPUT,
        ),
    ]

    num_channels: Annotated[
        int,
        Field(
            default=1,
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
            group=Groups.AUDIO_INPUT,
        ),
    ]

    @model_validator(mode="after")
    def _validate_audio_options(self) -> Self:
        """Validate the audio options.

        Flag configs where the user supplied non-default length parameters but did not
        enable audio. `batch_size=0` is treated as an explicit disable and always allowed,
        and default-valued fields (e.g., from a round-tripped config file) do not trip
        the check.
        """
        if self.audio_enabled() or self.batch_size == 0:
            return self
        if self.length.mean != 0.0 or self.length.stddev != 0.0:
            raise ValueError(
                "Audio generation is disabled but audio length options were provided. Please set `--audio-batch-size` and `--audio-length-mean` to enable audio generation."
            )
        return self

    def audio_enabled(self) -> bool:
        """Check if audio is enabled."""
        return self.length.mean > 0 and self.batch_size > 0
