# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.enums.base_enums import CaseInsensitiveStrEnum


class CommunicationBackend(CaseInsensitiveStrEnum):
    ZMQ_TCP = "zmq_tcp"
    ZMQ_IPC = "zmq_ipc"


class CommClientType(CaseInsensitiveStrEnum):
    PUB = "pub"
    SUB = "sub"
    PUSH = "push"
    PULL = "pull"
    REQUEST = "request"
    REPLY = "reply"
    STREAMING_ROUTER = "streaming_router"
    STREAMING_DEALER = "streaming_dealer"


class CommAddress(CaseInsensitiveStrEnum):
    """Enum for specifying the address type for communication clients.
    This is used to lookup the address in the communication config."""

    EVENT_BUS_PROXY_FRONTEND = "event_bus_proxy_frontend"
    """Frontend address for services to publish messages to."""

    EVENT_BUS_PROXY_BACKEND = "event_bus_proxy_backend"
    """Backend address for services to subscribe to messages."""

    CREDIT_DROP = "credit_drop"
    """Address to send CreditDrop messages from the TimingManager to the Worker."""

    CREDIT_RETURN = "credit_return"
    """Address to send CreditReturn messages from the Worker to the TimingManager."""

    RECORDS = "records"
    """Address to send parsed records from InferenceParser to RecordManager."""

    DATASET_MANAGER_PROXY_FRONTEND = "dataset_manager_proxy_frontend"
    """Frontend address for sending requests to the DatasetManager."""

    DATASET_MANAGER_PROXY_BACKEND = "dataset_manager_proxy_backend"
    """Backend address for the DatasetManager to receive requests from clients."""

    RAW_INFERENCE_PROXY_FRONTEND = "raw_inference_proxy_frontend"
    """Frontend address for sending raw inference messages to the InferenceParser from Workers."""

    RAW_INFERENCE_PROXY_BACKEND = "raw_inference_proxy_backend"
    """Backend address for the InferenceParser to receive raw inference messages from Workers."""


class ZMQProxyType(CaseInsensitiveStrEnum):
    DEALER_ROUTER = "dealer_router"
    XPUB_XSUB = "xpub_xsub"
    PUSH_PULL = "push_pull"
