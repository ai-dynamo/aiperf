# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.enums import CaseInsensitiveStrEnum, MessageType, Topic
from aiperf.common.exceptions import CommunicationError, CommunicationErrorReason


class PubClientType(CaseInsensitiveStrEnum):
    """
    Enum for specifying the client type for publishing messages. Includes a helper method
    for retrieving the appropriate client type based on the topic.
    """

    CONTROLLER = "controller_pub"
    COMPONENT = "component_pub"

    @classmethod
    def from_topic(cls, topic: Topic) -> "PubClientType":
        """Determine the appropriate ClientType based on topic.

        Args:
            topic: The topic to communicate on

        Returns:
            The appropriate ClientType for the given topic
        """
        match topic:
            case (
                Topic.HEARTBEAT
                | Topic.REGISTRATION
                | Topic.STATUS
                | Topic.RESPONSE
                | Topic.CREDITS_COMPLETE
                | Topic.PROFILE_PROGRESS
                | Topic.PROFILE_RESULTS
                | Topic.PROFILE_STATS
            ):
                return cls.COMPONENT
            case Topic.COMMAND:
                return cls.CONTROLLER
            case _:
                raise CommunicationError(
                    CommunicationErrorReason.CLIENT_NOT_FOUND,
                    f"No client type found for topic {topic}",
                )


class SubClientType(CaseInsensitiveStrEnum):
    """
    Enum for specifying the client type for subscribing to messages. Includes a helper method
    for retrieving the appropriate client type based on the topic.
    """

    CONTROLLER = "controller_sub"
    COMPONENT = "component_sub"

    @classmethod
    def from_topic(cls, topic: Topic) -> "SubClientType":
        """Determine the appropriate ClientType based on topic.

        Args:
            topic: The topic to communicate on

        Returns:
            The appropriate ClientType for the given topic
        """
        match topic:
            case (
                Topic.HEARTBEAT
                | Topic.REGISTRATION
                | Topic.STATUS
                | Topic.RESPONSE
                | Topic.CREDITS_COMPLETE
                | Topic.PROFILE_PROGRESS
                | Topic.PROFILE_RESULTS
                | Topic.PROFILE_STATS
            ):
                return cls.COMPONENT
            case Topic.COMMAND:
                return cls.CONTROLLER
            case _:
                raise CommunicationError(
                    CommunicationErrorReason.CLIENT_NOT_FOUND,
                    f"No client type found for topic {topic}",
                )


class PushClientType(CaseInsensitiveStrEnum):
    """
    Enum for specifying the client type for pushing messages. Includes a helper method
    for retrieving the appropriate client type based on the topic.
    """

    INFERENCE_RESULTS = "inference_results_push"
    CREDIT_DROP = "credit_drop_push"
    CREDIT_RETURN = "credit_return_push"

    @classmethod
    def from_topic(cls, topic: Topic) -> "PushClientType":
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
            case Topic.INFERENCE_RESULTS:
                return cls.INFERENCE_RESULTS
            case _:
                raise CommunicationError(
                    CommunicationErrorReason.CLIENT_NOT_FOUND,
                    f"No client type found for topic {topic}",
                )


class PullClientType(CaseInsensitiveStrEnum):
    """
    Enum for specifying the client type for pulling messages. Includes a helper method
    for retrieving the appropriate client type based on the topic.
    """

    INFERENCE_RESULTS = "inference_results_pull"
    CREDIT_DROP = "credit_drop_pull"
    CREDIT_RETURN = "credit_return_pull"

    @classmethod
    def from_message_type(cls, message_type: MessageType) -> "PullClientType":
        """Determine the appropriate ClientType based on message type.

        Args:
            message_type: The message type to communicate on

        Returns:
            The appropriate ClientType for the given message type
        """
        match message_type:
            case MessageType.CREDIT_DROP:
                return cls.CREDIT_DROP
            case MessageType.CREDIT_RETURN:
                return cls.CREDIT_RETURN
            case MessageType.INFERENCE_RESULTS:
                return cls.INFERENCE_RESULTS
            case _:
                raise CommunicationError(
                    CommunicationErrorReason.CLIENT_NOT_FOUND,
                    f"No client type found for message type {message_type}",
                )


class ReqClientType(CaseInsensitiveStrEnum):
    """
    Enum for specifying the client type for requesting messages. Includes a helper method
    for retrieving the appropriate client type based on the topic.
    """

    CONVERSATION_DATA = "conversation_data_req"

    @classmethod
    def from_topic(cls, topic: Topic) -> "ReqClientType":
        """Determine the appropriate ClientType based on topic.

        Args:
            topic: The topic to communicate on

        Returns:
            The appropriate ClientType for the given topic
        """
        match topic:
            case Topic.CONVERSATION_DATA:
                return cls.CONVERSATION_DATA
            case _:
                raise CommunicationError(
                    CommunicationErrorReason.CLIENT_NOT_FOUND,
                    f"No client type found for topic {topic}",
                )


class RepClientType(CaseInsensitiveStrEnum):
    """
    Enum for specifying the client type for responding to messages. Includes a helper method
    for retrieving the appropriate client type based on the topic.
    """

    CONVERSATION_DATA = "conversation_data_rep"

    @classmethod
    def from_topic(cls, topic: Topic) -> "RepClientType":
        """Determine the appropriate ClientType based on topic.

        Args:
            topic: The topic to communicate on

        Returns:
            The appropriate ClientType for the given topic
        """
        match topic:
            case Topic.CONVERSATION_DATA:
                return cls.CONVERSATION_DATA
            case _:
                raise CommunicationError(
                    CommunicationErrorReason.CLIENT_NOT_FOUND,
                    f"No client type found for topic {topic}",
                )


ClientType = (
    PubClientType
    | SubClientType
    | PushClientType
    | PullClientType
    | ReqClientType
    | RepClientType
)
"""Union of all client types."""
