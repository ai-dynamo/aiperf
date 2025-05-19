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

from aiperf.common.enums.base_enums import StrEnum


class CommunicationBackend(StrEnum):
    """Supported communication backends."""

    ZMQ_TCP = "zmq_tcp"
    """ZeroMQ backend using TCP sockets."""


class Topic(StrEnum):
    """Communication topics for the main messaging bus.
    Right now, there is some overlap between Topic and MessageType."""

    CREDIT_DROP = "credit_drop"
    """The topic for credit drop messages."""

    CREDIT_RETURN = "credit_return"
    """The topic for credit return messages."""

    REGISTRATION = "registration"
    """The topic for registration messages."""

    COMMAND = "command"
    """The topic for command messages."""

    RESPONSE = "response"
    """The topic for response messages."""

    STATUS = "status"
    """The topic for status messages."""

    HEARTBEAT = "heartbeat"
    """The topic for heartbeat messages."""


# TODO: Is this separation needed? Or should we just use the Topic enum?
class DataTopic(StrEnum):
    """TBD. Specific data topics for use in the future."""

    DATASET = "dataset_data"
    """The topic for dataset data."""

    RECORDS = "records_data"
    """The topic for records data."""

    WORKER = "worker_data"
    """The topic for worker data."""

    POST_PROCESSOR = "post_processor_data"
    """The topic for post processor data."""

    RESULTS = "results"
    """The topic for results data."""

    METRICS = "metrics"
    """The topic for metrics data."""

    CONVERSATION = "conversation_data"
    """The topic for conversation data."""


TopicType = Topic | DataTopic
"""TopicType is a union of all the various different topic types supported by
the system."""
