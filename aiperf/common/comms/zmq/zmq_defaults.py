#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: Apache-2.0


class ZMQSocketDefaults:
    """Default values for ZMQ sockets."""

    # Socket Options
    RCVTIMEO = 300000  # 5 minutes
    SNDTIMEO = 300000  # 5 minutes
    TCP_KEEPALIVE = 1
    TCP_KEEPALIVE_IDLE = 60
    TCP_KEEPALIVE_INTVL = 10
    TCP_KEEPALIVE_CNT = 3
    IMMEDIATE = 1  # Don't queue messages
    LINGER = 0  # Don't wait on close
