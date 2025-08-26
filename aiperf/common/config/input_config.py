# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Annotated, Any

from cyclopts import Parameter
from pydantic import BeforeValidator, Field, model_validator
from typing_extensions import Self

from aiperf.common.aiperf_logger import AIPerfLogger
from aiperf.common.config.audio_config import AudioConfig
from aiperf.common.config.base_config import BaseConfig
from aiperf.common.config.config_defaults import InputDefaults
from aiperf.common.config.config_validators import (
    parse_file,
    parse_str_or_dict_as_tuple_list,
)
from aiperf.common.config.conversation_config import ConversationConfig
from aiperf.common.config.groups import Groups
from aiperf.common.config.image_config import ImageConfig
from aiperf.common.config.prompt_config import PromptConfig
from aiperf.common.enums import CustomDatasetType, PublicDatasetType

logger = AIPerfLogger(__name__)


class InputConfig(BaseConfig):
    """
    A configuration class for defining input related settings.
    """

    _CLI_GROUP = Groups.INPUT

    @model_validator(mode="after")
    def validate_fixed_schedule(self) -> Self:
        """Validate the fixed schedule configuration."""
        if self.fixed_schedule and self.file is None:
            raise ValueError("Fixed schedule requires a file to be provided")
        if self.file is not None:
            self.fixed_schedule = True
            logger.debug("Fixed schedule is enabled because file is provided")
        return self

    @model_validator(mode="after")
    def validate_fixed_schedule_start_offset(self) -> Self:
        """Validate the fixed schedule start offset configuration."""
        if (
            self.fixed_schedule_start_offset is not None
            and self.fixed_schedule_auto_offset
        ):
            raise ValueError(
                "The --fixed-schedule-start-offset and --fixed-schedule-auto-offset options cannot be used together"
            )
        return self

    @model_validator(mode="after")
    def validate_fixed_schedule_start_and_end_offset(self) -> Self:
        """Validate the fixed schedule start and end offset configuration."""
        if (
            self.fixed_schedule_start_offset is not None
            and self.fixed_schedule_end_offset is not None
            and self.fixed_schedule_start_offset > self.fixed_schedule_end_offset
        ):
            raise ValueError(
                "The --fixed-schedule-start-offset must be less than or equal to the --fixed-schedule-end-offset"
            )
        return self

    extra: Annotated[
        Any,
        Field(
            description="Provide additional inputs to include with every request.\n"
            "Inputs should be in an 'input_name:value' format.\n"
            "Alternatively, a string representing a json formatted dict can be provided.",
        ),
        Parameter(
            name=(
                "--extra-inputs",  # GenAI-Perf
            ),
            consume_multiple=True,
            group=_CLI_GROUP,
        ),
        BeforeValidator(parse_str_or_dict_as_tuple_list),
    ] = InputDefaults.EXTRA

    headers: Annotated[
        Any,
        Field(
            description="Adds a custom header to the requests.\n"
            "Headers must be specified as 'Header:Value' pairs.\n"
            "Alternatively, a string representing a json formatted dict can be provided.",
        ),
        BeforeValidator(parse_str_or_dict_as_tuple_list),
        Parameter(
            name=(
                "--header",  # GenAI-Perf
                "-H",  # GenAI-Perf
            ),
            consume_multiple=True,
            group=_CLI_GROUP,
        ),
    ] = InputDefaults.HEADERS

    file: Annotated[
        Any,
        Field(
            description="The file or directory path that contains the dataset to use for profiling.\n"
            "This parameter is used in conjunction with the `custom_dataset_type` parameter\n"
            "to support different types of user provided datasets.",
        ),
        BeforeValidator(parse_file),
        Parameter(
            name=(
                "--input-file",  # GenAI-Perf,
            ),
            group=_CLI_GROUP,
        ),
    ] = InputDefaults.FILE

    fixed_schedule: Annotated[
        bool,
        Field(
            description="Specifies to run a fixed schedule of requests. This is normally inferred from the --input-file parameter, but can be set manually here."
        ),
        Parameter(
            name=(
                "--fixed-schedule",  # GenAI-Perf
            ),
            group=_CLI_GROUP,
        ),
    ] = InputDefaults.FIXED_SCHEDULE

    # NEW AIPerf Option
    fixed_schedule_auto_offset: Annotated[
        bool,
        Field(
            description="Specifies to automatically offset the timestamps in the fixed schedule, such that the first "
            "timestamp is considered 0, and the rest are shifted accordingly. If disabled, the timestamps will be "
            "assumed to be relative to 0."
        ),
        Parameter(
            name=("--fixed-schedule-auto-offset",),
            group=_CLI_GROUP,
        ),
    ] = InputDefaults.FIXED_SCHEDULE_AUTO_OFFSET

    # NEW AIPerf Option
    fixed_schedule_start_offset: Annotated[
        int | None,
        Field(
            ge=0,
            description="Specifies the offset in milliseconds to start the fixed schedule at. By default, the schedule "
            "starts at 0, but this option can be used to start at a reference point further in the schedule. This "
            "option cannot be used in conjunction with the --fixed-schedule-auto-offset. The schedule will include "
            "any requests at the start offset.",
        ),
        Parameter(
            name=("--fixed-schedule-start-offset",),
            group=_CLI_GROUP,
        ),
    ] = InputDefaults.FIXED_SCHEDULE_START_OFFSET

    # NEW AIPerf Option
    fixed_schedule_end_offset: Annotated[
        int | None,
        Field(
            ge=0,
            description="Specifies the offset in milliseconds to end the fixed schedule at. By default, the schedule "
            "ends at the last timestamp in the trace dataset, but this option can be used to only run a subset of the trace. "
            "The schedule will include any requests at the end offset.",
        ),
        Parameter(
            name=("--fixed-schedule-end-offset",),
            group=_CLI_GROUP,
        ),
    ] = InputDefaults.FIXED_SCHEDULE_END_OFFSET

    public_dataset: Annotated[
        PublicDatasetType | None,
        Field(description="The public dataset to use for the requests."),
        Parameter(
            name=("--public-dataset"),
            group=_CLI_GROUP,
        ),
    ] = InputDefaults.PUBLIC_DATASET

    # NEW AIPerf Option
    custom_dataset_type: Annotated[
        CustomDatasetType | None,
        Field(
            description="The type of custom dataset to use.\n"
            "This parameter is used in conjunction with the --file parameter.",
        ),
        Parameter(
            name=("--custom-dataset-type"),
            group=_CLI_GROUP,
        ),
    ] = InputDefaults.CUSTOM_DATASET_TYPE

    random_seed: Annotated[
        int | None,
        Field(
            default=None,
            description="The seed used to generate random values.\n"
            "Set to some value to make the synthetic data generation deterministic.\n"
            "It will use system default if not provided.",
        ),
        Parameter(
            name=(
                "--random-seed",  # GenAI-Perf
            ),
            group=_CLI_GROUP,
        ),
    ] = InputDefaults.RANDOM_SEED

    audio: AudioConfig = AudioConfig()
    image: ImageConfig = ImageConfig()
    prompt: PromptConfig = PromptConfig()
    conversation: ConversationConfig = ConversationConfig()
