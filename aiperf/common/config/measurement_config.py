# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Annotated

from cyclopts import Parameter
from pydantic import Field

from aiperf.common.config.base_config import BaseConfig
from aiperf.common.config.config_defaults import MeasurementDefaults
from aiperf.common.config.groups import Groups


class MeasurementConfig(BaseConfig):
    """
    A configuration class for defining top-level measurement settings.
    """

    _CLI_GROUP = Groups.MEASUREMENT

    # TODO: Not implemented yet
    measurement_interval: Annotated[
        float,
        Field(
            ge=1,
            le=1_000_000,
            description="The time interval used for each measurement in milliseconds. "
            "AIPerf will sample a time interval specified and take "
            "measurement over the requests completed within that time interval. "
            "When using the default stability percentage, AIPerf will benchmark  "
            "for 3*(measurement_interval) milliseconds.",
        ),
        Parameter(
            name=(
                "--measurement-interval-ms",
                "--measurement-interval",  # GenAI-Perf
                "-p",  # GenAI-Perf
            ),
            parse=False,  # TODO: Not yet supported
            group=_CLI_GROUP,
        ),
    ] = MeasurementDefaults.MEASUREMENT_INTERVAL

    # TODO: Not implemented yet
    stability_percentage: Annotated[
        float,
        Field(
            gt=0.0,
            lt=1.0,
            description="The allowed variation in latency measurements when determining if a result is stable.\n"
            "The measurement is considered as stable if the ratio of max / min\n"
            "from the recent 3 measurements is within (stability percentage)\n"
            "in terms of both infer per second and latency.",
        ),
        Parameter(
            name=(
                "--stability-percentage",  # GenAI-Perf
                "-s",  # GenAI-Perf
            ),
            parse=False,  # TODO: Not yet supported
            group=_CLI_GROUP,
        ),
    ] = MeasurementDefaults.STABILITY_PERCENTAGE
