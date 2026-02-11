# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.environment import Environment

# ZMQ Constants
TOPIC_END = "$"
"""This is used to add to the end of each topic to prevent the topic from being a prefix of another topic.
This is required for the PUB/SUB pattern to work correctly, otherwise topics like "command_response" will be
received by the "command" subscriber as well.

For example:
- "command$"
- "command_response$"
"""

TOPIC_END_LENGTH = len(TOPIC_END)
"""The length of TOPIC_END. Computed once at runtime to avoid recomputing the length on every message."""

TOPIC_END_ENCODED = TOPIC_END.encode()
"""The encoded version of TOPIC_END. Computed once at runtime to avoid recomputing the encoded version on every message."""

TOPIC_DELIMITER = "."
"""The delimiter between topic parts.
This is used to create an inverted hierarchy of topics for filtering by service type or service id.

For example:
- "command$"
- "system_controller.command$"
- "timing_manager_eff34565.command$"
"""


class ZMQSocketDefaults:
    """Default values for ZMQ sockets.

    Socket options are loaded from the Environment configuration to allow runtime tuning.
    """

    # Socket Options (loaded from Environment)
    RCVTIMEO = Environment.ZMQ.RCVTIMEO
    SNDTIMEO = Environment.ZMQ.SNDTIMEO
    TCP_KEEPALIVE = 1
    TCP_KEEPALIVE_IDLE = Environment.ZMQ.TCP_KEEPALIVE_IDLE
    TCP_KEEPALIVE_INTVL = Environment.ZMQ.TCP_KEEPALIVE_INTVL
    TCP_KEEPALIVE_CNT = 3
    IMMEDIATE = 1  # Don't queue messages
    LINGER = 0  # Don't wait on close

    # High Water Mark
    # TODO: Investigate better ways to handle this
    # https://zeromq.org/socket-api/#high-water-mark
    # NOTE: We set these to 0 to allow for unlimited messages to be queued. This is important to
    #       ensure that the system does not lose messages. It does however mean that the system
    #       could run out of memory if too many messages are queued.
    SNDHWM = 0  # No send high water mark (unlimited)
    RCVHWM = 0  # No receive high water mark (unlimited)
