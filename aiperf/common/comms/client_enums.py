# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import Union

from aiperf.common.enums import DataTopic, StrEnum, Topic, TopicType
from aiperf.common.exceptions import (
    CommunicationClientNotFoundError,
)


class PubClientType(StrEnum):
    """
    Enum for specifying the client type for publishing messages. Includes a helper method
    for retrieving the appropriate client type based on the topic.
    """

    CONTROLLER = "controller_pub"
    COMPONENT = "component_pub"

    @classmethod
    def from_topic(cls, topic: TopicType) -> "PubClientType":
        """Determine the appropriate ClientType based on topic.

        Args:
            topic: The topic to communicate on

        Returns:
            The appropriate ClientType for the given topic
        """
        match topic:
            case Topic.HEARTBEAT | Topic.REGISTRATION | Topic.STATUS | Topic.RESPONSE:
                return cls.COMPONENT
            case Topic.COMMAND:
                return cls.CONTROLLER
            case _:
                raise CommunicationClientNotFoundError(
                    f"No client type found for topic {topic}"
                )


class SubClientType(StrEnum):
    """
    Enum for specifying the client type for subscribing to messages. Includes a helper method
    for retrieving the appropriate client type based on the topic.
    """

    CONTROLLER = "controller_sub"
    COMPONENT = "component_sub"

    @classmethod
    def from_topic(cls, topic: TopicType) -> "SubClientType":
        """Determine the appropriate ClientType based on topic.

        Args:
            topic: The topic to communicate on

        Returns:
            The appropriate ClientType for the given topic
        """
        match topic:
            case Topic.HEARTBEAT | Topic.REGISTRATION | Topic.STATUS | Topic.RESPONSE:
                return cls.COMPONENT
            case Topic.COMMAND:
                return cls.CONTROLLER
            case _:
                raise CommunicationClientNotFoundError(
                    f"No client type found for topic {topic}"
                )


class PushClientType(StrEnum):
    """
    Enum for specifying the client type for pushing messages. Includes a helper method
    for retrieving the appropriate client type based on the topic.
    """

    RECORDS = "records_push"
    INFERENCE_RESULTS = "inference_results_push"
    CREDIT_DROP = "credit_drop_push"
    CREDIT_RETURN = "credit_return_push"

    @classmethod
    def from_topic(cls, topic: TopicType) -> "PushClientType":
        """Determine the appropriate ClientType based on communication type and topic.

        Args:
            topic: The topic to communicate on

        Returns:
            The appropriate ClientType for the given topic
        """
        match topic:
            case Topic.CREDIT_DROP:
                return cls.CREDIT_DROP
            case Topic.CREDIT_RETURN:
                return cls.CREDIT_RETURN
            case DataTopic.RECORDS:
                return cls.RECORDS
            case DataTopic.RESULTS:
                return cls.INFERENCE_RESULTS
            case _:
                raise CommunicationClientNotFoundError(
                    f"No client type found for topic {topic}"
                )


class PullClientType(StrEnum):
    """
    Enum for specifying the client type for pulling messages. Includes a helper method
    for retrieving the appropriate client type based on the topic.
    """

    RECORDS = "records_pull"
    INFERENCE_RESULTS = "inference_results_pull"
    CREDIT_DROP = "credit_drop_pull"
    CREDIT_RETURN = "credit_return_pull"

    @classmethod
    def from_topic(cls, topic: TopicType) -> "PullClientType":
        """Determine the appropriate ClientType based on topic.

        Args:
            topic: The topic to communicate on

        Returns:
            The appropriate ClientType for the given topic
        """
        match topic:
            case Topic.CREDIT_DROP:
                return cls.CREDIT_DROP
            case Topic.CREDIT_RETURN:
                return cls.CREDIT_RETURN
            case DataTopic.RECORDS:
                return cls.RECORDS
            case DataTopic.RESULTS:
                return cls.INFERENCE_RESULTS
            case _:
                raise CommunicationClientNotFoundError(
                    f"No client type found for topic {topic}"
                )


class ReqClientType(StrEnum):
    """
    Enum for specifying the client type for requesting messages. Includes a helper method
    for retrieving the appropriate client type based on the topic.
    """

    CONVERSATION_DATA = "conversation_data_req"

    @classmethod
    def from_topic(cls, topic: TopicType) -> "ReqClientType":
        """Determine the appropriate ClientType based on topic.

        Args:
            topic: The topic to communicate on

        Returns:
            The appropriate ClientType for the given topic
        """
        match topic:
            case DataTopic.CONVERSATION:
                return cls.CONVERSATION_DATA
            case _:
                raise CommunicationClientNotFoundError(
                    f"No client type found for topic {topic}"
                )


class RepClientType(StrEnum):
    """
    Enum for specifying the client type for responding to messages. Includes a helper method
    for retrieving the appropriate client type based on the topic.
    """

    CONVERSATION_DATA = "conversation_data_rep"

    @classmethod
    def from_topic(cls, topic: TopicType) -> "RepClientType":
        """Determine the appropriate ClientType based on topic.

        Args:
            topic: The topic to communicate on

        Returns:
            The appropriate ClientType for the given topic
        """
        match topic:
            case DataTopic.CONVERSATION:
                return cls.CONVERSATION_DATA
            case _:
                raise CommunicationClientNotFoundError(
                    f"No client type found for topic {topic}"
                )


ClientType = Union[  # noqa: UP007
    PubClientType,
    SubClientType,
    PushClientType,
    PullClientType,
    ReqClientType,
    RepClientType,
]
"""Union of all client types."""
