# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Annotated, Any

import cyclopts
from pydantic import Field

from aiperf.common.config.base_config import BaseConfig
from aiperf.common.config.config_defaults import SweepDefaults, SweepParamDefaults
from aiperf.progress.progress_models import (
    SweepCompletionTrigger,
    SweepMultiParamOrder,
    SweepParamOrder,
    SweepParamType,
)


class SweepParam(BaseConfig):
    """A parameter to be swept."""

    name: Annotated[
        str,
        Field(description="The name of the parameter"),
        cyclopts.Parameter(
            name=("--sweep-param-name"),
        ),
    ]

    param_type: Annotated[
        SweepParamType,
        Field(description="The type of the parameter"),
        cyclopts.Parameter(
            name=("--sweep-param-type"),
        ),
    ]

    values: Annotated[
        list[Any] | None,
        Field(
            description="The values of the parameter. This is only applicable for string and boolean parameters.",
        ),
        cyclopts.Parameter(
            name=("--sweep-param-values"),
        ),
    ] = SweepParamDefaults.VALUES

    order: Annotated[
        SweepParamOrder,
        Field(
            description="The order of the parameter.\n"
            "ascending: The parameter will be swept over a range of values in ascending order.\n"
            "descending: The parameter will be swept over a range of values in descending order.\n"
            "random: The parameter will be swept over a range of values in random order.",
        ),
        cyclopts.Parameter(
            name=("--sweep-param-order"),
        ),
    ] = SweepParamDefaults.ORDER

    start: Annotated[
        float | None,
        Field(
            description="The start value of the parameter. This is only applicable for float and int parameters.",
        ),
        cyclopts.Parameter(
            name=("--sweep-param-start"),
        ),
    ] = SweepParamDefaults.START

    step: Annotated[
        float | None,
        Field(
            description="The step size of the parameter. This is only applicable for float and int parameters.",
        ),
        cyclopts.Parameter(
            name=("--sweep-param-step"),
        ),
    ] = SweepParamDefaults.STEP

    end: Annotated[
        float | None,
        Field(
            description="The end value of the parameter. This is only applicable for float and int parameters.",
        ),
        cyclopts.Parameter(
            name=("--sweep-param-end"),
        ),
    ] = SweepParamDefaults.END

    completion_trigger: Annotated[
        SweepCompletionTrigger,
        Field(description="The trigger for sweep completion"),
        cyclopts.Parameter(
            name=("--sweep-completion-trigger"),
        ),
    ] = SweepParamDefaults.COMPLETION_TRIGGER

    max_profiles: Annotated[
        int | None,
        Field(
            description="The maximum number of profiles to be run. If not specified, the sweep will run until the completion trigger is met.",
        ),
        cyclopts.Parameter(
            name=("--sweep-max-profiles"),
        ),
    ] = SweepParamDefaults.MAX_PROFILES


class SweepConfig(BaseConfig):
    """A sweep of parameters."""

    params: Annotated[
        list[SweepParam] | None,
        Field(
            description="The list of all parameters to be swept and their values.",
        ),
        cyclopts.Parameter(
            name=("--sweep-params"),
        ),
    ] = SweepDefaults.PARAMS

    order: Annotated[
        SweepMultiParamOrder,
        Field(description="The order of the parameters to be swept."),
        cyclopts.Parameter(
            name=("--multi-sweep-order"),
        ),
    ] = SweepDefaults.ORDER
