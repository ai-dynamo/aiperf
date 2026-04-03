# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import Annotated

from pydantic import Field, model_validator
from typing_extensions import Self

from aiperf.common.config.base_config import BaseConfig
from aiperf.common.config.cli_parameter import CLIParameter, DisableCLI
from aiperf.common.config.config_defaults import WorkersDefaults
from aiperf.common.config.groups import Groups


class WorkersConfig(BaseConfig):
    """Worker configuration."""

    _CLI_GROUP = Groups.WORKERS

    min: Annotated[
        int | None,
        Field(
            description="Minimum number of worker processes to maintain in the pool. Workers handle request execution and API communication. "
            "Currently not configurable via CLI - reserved for future dynamic scaling features.",
        ),
        DisableCLI(reason="Not currently supported"),
    ] = WorkersDefaults.MIN

    max: Annotated[
        int | None,
        Field(
            description="Maximum number of workers to create. If not specified, the number of"
            " workers will be determined by the formula `min(concurrency, (num CPUs * 0.75) - 1)`, "
            " with a default max cap of 32. Any value provided will still be capped by"
            " the concurrency value (if specified), but not by the max cap.",
        ),
        CLIParameter(
            name=("--workers-max", "--max-workers"),
            group=_CLI_GROUP,
        ),
    ] = WorkersDefaults.MAX

    direct: Annotated[
        bool | None,
        Field(
            description="Enable direct worker mode: co-locate a single Worker inside TimingManager's process "
            "and deliver credits via method calls instead of ZMQ. Eliminates IPC overhead for "
            "ultra-low-latency benchmarks. Some endpoints enable this automatically. "
            "Use --no-workers-direct to force standard multi-process workers.",
        ),
        CLIParameter(
            name=("--workers-direct",),
            negative=("--no-workers-direct",),
            group=_CLI_GROUP,
        ),
    ] = WorkersDefaults.DIRECT

    @model_validator(mode="after")
    def validate_direct_incompatible_with_max(self) -> Self:
        """Direct worker mode uses a single co-located worker, so --workers-max is meaningless."""
        if self.direct and "max" in self.model_fields_set:
            raise ValueError(
                "--workers-direct and --workers-max cannot be used together. "
                "Direct worker mode uses a single co-located worker."
            )
        return self
