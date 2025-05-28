# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

__all__ = [
    "ZMQClient",
    "BaseZMQClient",
    "ZMQPubClient",
    "ZMQPullClient",
    "ZMQPushClient",
    "ZMQRepClient",
    "ZMQReqClient",
    "ZMQSubClient",
]

from typing import Union

from aiperf.common.comms.zmq.clients.base import BaseZMQClient
from aiperf.common.comms.zmq.clients.pub import ZMQPubClient
from aiperf.common.comms.zmq.clients.pull import ZMQPullClient
from aiperf.common.comms.zmq.clients.push import ZMQPushClient
from aiperf.common.comms.zmq.clients.rep import ZMQRepClient
from aiperf.common.comms.zmq.clients.req import ZMQReqClient
from aiperf.common.comms.zmq.clients.sub import ZMQSubClient

# Union of all the possible ZMQ client types for type checking
ZMQClient = Union[  # noqa: UP007
    ZMQPubClient,
    ZMQSubClient,
    ZMQPullClient,
    ZMQPushClient,
    ZMQRepClient,
    ZMQReqClient,
]
