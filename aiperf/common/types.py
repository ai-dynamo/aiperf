# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
This module defines common used alias types for AIPerf. This both helps prevent circular imports and
helps with type hinting.
"""

from collections.abc import Awaitable, Callable
from types import UnionType
from typing import TYPE_CHECKING, Any, TypeAlias, TypeVar, Union

from aiperf.common.enums import (
    CaseInsensitiveStrEnum,
    CommAddress,
    CommandType,
    MediaType,
    MessageType,
    MetricType,
    ServiceType,
)

if TYPE_CHECKING:
    from pydantic import BaseModel

    from aiperf.clients.model_endpoint_info import ModelEndpointInfo
    from aiperf.common.messages.base_messages import Message
    from aiperf.common.messages.command_messages import CommandMessage
    from aiperf.common.mixins.aiperf_lifecycle_mixin import AIPerfLifecycleMixin
    from aiperf.common.mixins.hooks_mixin import HooksMixin
    from aiperf.common.models.base_models import AIPerfBaseModel
    from aiperf.common.models.dataset_models import Media
    from aiperf.common.protocols import ServiceProtocol


AnyT: TypeAlias = Any
AnyClassT: TypeAlias = type | UnionType
AIPerfBaseModelT = TypeVar("AIPerfBaseModelT", bound="AIPerfBaseModel")
BaseModelT = TypeVar("BaseModelT", bound="BaseModel")
ClassEnumT = TypeVar("ClassEnumT", bound="CaseInsensitiveStrEnum")
ClassProtocolT = TypeVar("ClassProtocolT", bound=Any)
CommAddressType: TypeAlias = Union["CommAddress", str]
CommandCallbackMapT: TypeAlias = dict[
    "CommandType", Callable[["CommandMessage"], Awaitable[Any]]
]
CommandTypeT: TypeAlias = CommandType | str
ConfigT = TypeVar("ConfigT", bound=Any, covariant=True)
HooksMixinT = TypeVar("HooksMixinT", bound="HooksMixin")
HookParamsT = TypeVar("HookParamsT", bound=Any)
HookCallableParamsT = HookParamsT | Callable[["SelfT"], HookParamsT]
InputT = TypeVar("InputT", bound=Any)
LifecycleMixinT = TypeVar("LifecycleMixinT", bound="AIPerfLifecycleMixin")
MediaT = TypeVar("MediaT", bound="Media")
MediaTypeT = MediaType | str
MessageT = TypeVar("MessageT", bound="Message")
MessageCallbackMapT: TypeAlias = dict["MessageTypeT", Callable[["Message"], Any] | list[Callable[["Message"], Any]]]  # fmt: skip
MessageOutputT = TypeVar("MessageOutputT", bound="Message")
MetricTypeT: TypeAlias = MetricType | str
MessageTypeT: TypeAlias = MessageType | str
MetricTagT: TypeAlias = str
ModelEndpointInfoT = TypeVar("ModelEndpointInfoT", bound="ModelEndpointInfo")
OutputT = TypeVar("OutputT", bound=Any)
ProtocolT = TypeVar("ProtocolT", bound=Any)
RawRequestT = TypeVar("RawRequestT", bound=Any, contravariant=True)
RawResponseT = TypeVar("RawResponseT", bound=Any, contravariant=True)
RequestInputT = TypeVar("RequestInputT", bound=Any, contravariant=True)
RequestOutputT = TypeVar("RequestOutputT", bound=Any, covariant=True)
ResponseT = TypeVar("ResponseT", bound=Any, covariant=True)
SelfT = TypeVar("SelfT", bound=Any)
ServiceProtocolT = TypeVar("ServiceProtocolT", bound="ServiceProtocol")
ServiceTypeT: TypeAlias = ServiceType | str
