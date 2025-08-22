# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from cyclopts import Parameter

from aiperf.common.config.groups import Groups
from aiperf.common.constants import AIPERF_DEV_MODE


class CLIParameter(Parameter):
    """Configuration for a CLI parameter.

    This is a subclass of the cyclopts.Parameter class that includes the default configuration AIPerf uses
    for all of its CLI parameters. This is used to ensure that the CLI parameters are consistent across all
    of the AIPerf config.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, show_env_var=False, negative=False, **kwargs)


class DisableCLI(CLIParameter):
    """Configuration for a CLI parameter that is disabled.

    This is a subclass of the CLIParameter class that is used to set a CLI parameter to disabled.
    """

    def __init__(self, reason: str, *args, **kwargs):
        super().__init__(*args, parse=False, **kwargs)


class DeveloperOnlyCLI(CLIParameter):
    """Configuration for a CLI parameter that is only available to developers.

    This is a subclass of the CLIParameter class that is used to set a CLI parameter to only be available to developers.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, parse=AIPERF_DEV_MODE, group=Groups.DEVELOPER, **kwargs)
