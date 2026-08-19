# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Kubernetes-specific view of the flat benchmark CLI configuration."""

from typing import Annotated, TypeAlias

from cyclopts import Parameter

from aiperf.config.flags.cli_config import CLIConfig

# The local worker-process limit does not apply to distributed Kubernetes
# execution; KubeOptions.total_workers owns that surface instead.
KubeCLIConfig: TypeAlias = Annotated[
    CLIConfig, Parameter(parse=r"^(?!workers_max$).*$")
]
