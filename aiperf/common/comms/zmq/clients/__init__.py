#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

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
