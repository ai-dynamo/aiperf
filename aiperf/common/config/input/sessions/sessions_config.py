#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

from typing import Annotated

from pydantic import Field

from aiperf.common.config.base_config import BaseConfig
from aiperf.common.config.config_defaults import SessionsDefaults
from aiperf.common.config.input.sessions.session_turn_delay_config import (
    SessionTurnDelayConfig,
)
from aiperf.common.config.input.sessions.session_turns_config import (
    SessionTurnsConfig,
)


class SessionsConfig(BaseConfig):
    """
    A configuration class for defining sessions related settings.
    """

    num: Annotated[
        int,
        Field(
            default=SessionsDefaults.NUM,
            ge=0,
            description="The number of sessions to simulate",
        ),
    ]
    turns: SessionTurnsConfig = SessionTurnsConfig()
    turn_delay: SessionTurnDelayConfig = SessionTurnDelayConfig()
