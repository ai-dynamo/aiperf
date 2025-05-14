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
from typing import Union

from aiperf.common.enums.base import StrEnum


class CommBackend(StrEnum):
    """Supported communication backends."""

    ZMQ_TCP = "zmq_tcp"


class CommType(StrEnum):
    """Communication type for response bus operations."""

    PUB = "pub"
    SUB = "sub"
    PUSH = "push"
    PULL = "pull"
    REP = "rep"
    REQ = "req"


class Topic(StrEnum):
    """Communication topics for the main response bus."""

    CREDIT_DROP = "credit_drop"
    CREDIT_RETURN = "credit_return"
    REGISTRATION = "registration"
    COMMAND = "command"
    RESPONSE = "response"
    STATUS = "status"
    HEARTBEAT = "heartbeat"


class DataTopic(StrEnum):
    """Specific data topics for different service domains."""

    DATASET = "dataset_data"
    TIMING = "timing_data"
    RECORDS = "records_data"
    WORKER = "worker_data"
    POST_PROCESSOR = "post_processor_data"
    CREDIT = "credit"
    RESULTS = "results"
    METRICS = "metrics"
    CONVERSATION = "conversation_data"


TopicType = Union[Topic, DataTopic]
