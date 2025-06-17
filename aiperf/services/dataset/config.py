# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from aiperf.common.enums import AudioFormat, CustomDatasetType, ImageFormat
from aiperf.common.tokenizer import Tokenizer


# TODO: Temporary. Remove after configurations are created.
########################################################
# Generator configurations
########################################################
class PrefixPromptConfig(BaseModel):
    pool_size: int = Field(default=0, description="Pool size of the prefix prompt.")
    length: int = Field(default=0, description="Length of the prefix prompt.")


class PromptConfig(BaseModel):
    # TODO: any better workaround? (needed because pydantic cannot serialize Tokenizer class)
    model_config = ConfigDict(arbitrary_types_allowed=True)

    tokenizer: Tokenizer | None = Field(
        default=None, description="Tokenizer to use for the prompt generation."
    )
    batch_size: int = Field(
        default=1, description="Batch size of the prompt generation."
    )
    mean: int = Field(default=550, description="Mean length of the prompt.")
    stddev: int = Field(
        default=250, description="Standard deviation of the prompt length."
    )
    block_size: int = Field(default=512, description="Block size of the prompt.")
    prefix_prompt: PrefixPromptConfig = Field(
        default=PrefixPromptConfig(),
        description="Prefix prompt to use for the prompt generation.",
    )


class ImageConfig(BaseModel):
    batch_size: int = Field(
        default=1, description="Batch size of the image generation."
    )
    width_mean: int = Field(default=0, description="Mean width of the image.")
    width_stddev: int = Field(
        default=0, description="Standard deviation of the image width."
    )
    height_mean: int = Field(default=0, description="Mean height of the image.")
    height_stddev: int = Field(
        default=0, description="Standard deviation of the image height."
    )
    format: ImageFormat | None = Field(default=None, description="Format of the image.")


class AudioConfig(BaseModel):
    batch_size: int = Field(
        default=1, description="Batch size of the audio generation."
    )
    length_mean: int = Field(default=0, description="Mean length of the audio.")
    length_stddev: int = Field(
        default=0, description="Standard deviation of the audio length."
    )
    num_channels: int = Field(default=1, description="Number of channels of the audio.")
    sample_rates: list[int] = Field(
        default_factory=lambda: [16], description="Sample rates of the audio."
    )
    depths: list[int] = Field(
        default_factory=lambda: [16], description="Depths of the audio."
    )
    format: AudioFormat = Field(
        default=AudioFormat.WAV, description="Format of the audio."
    )


########################################################
# Composer configurations
########################################################


class TurnDelayConfig(BaseModel):
    mean: int = Field(default=0, description="Mean delay of the turn in seconds.")
    stddev: int = Field(
        default=0, description="Standard deviation of the turn delay in seconds."
    )
    ratio: float = Field(default=1.0, description="Delay ratio of the turn.")


class TurnConfig(BaseModel):
    mean: int = Field(default=10, description="Mean number of turns in a session.")
    stddev: int = Field(
        default=0, description="Standard deviation of the number of turns in a session."
    )
    delay: TurnDelayConfig = Field(
        default=TurnDelayConfig(), description="Delay of the turn."
    )


class DatasetConfig(BaseModel):
    filename: Path | None = Field(default=None, description="Filename of the dataset.")
    custom_dataset_type: CustomDatasetType = Field(
        default=CustomDatasetType.SINGLE_TURN, description="Type of the custom dataset."
    )
    num_conversations: int = Field(
        default=100, description="Number of total conversations (or sessions)."
    )
    prompt: PromptConfig = Field(
        default=PromptConfig(), description="Prompt configuration."
    )
    image: ImageConfig = Field(
        default=ImageConfig(), description="Image configuration."
    )
    audio: AudioConfig = Field(
        default=AudioConfig(), description="Audio configuration."
    )
    turn: TurnConfig = Field(default=TurnConfig(), description="Turn configuration.")
