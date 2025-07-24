# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
This module defines common used alias types for AIPerf. This both helps prevent circular imports and
helps with type hinting.
"""

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, TypeVar, Union

from aiperf.common.enums import (
    MessageType,  # NOTE: Required for pydantic models to work
    ServiceType,  # NOTE: Required for pydantic models to work
)

if TYPE_CHECKING:
    from aiperf.clients.model_endpoint_info import ModelEndpointInfo
    from aiperf.common.enums import (
        CaseInsensitiveStrEnum,
        CommAddress,
        CommandType,
    )
    from aiperf.common.messages.base_messages import Message
    from aiperf.common.messages.command_messages import CommandMessage
    from aiperf.common.mixins.aiperf_lifecycle_mixin import AIPerfLifecycleMixin
    from aiperf.common.mixins.hooks_mixin import HooksMixin
    from aiperf.common.models import AIPerfBaseModel, Turn
    from aiperf.common.models.record_models import (
        ParsedResponseRecord,
        RequestRecord,
        ResponseData,
    )
    from aiperf.common.models.service_models import ServiceRunInfo
    from aiperf.common.protocols import ServiceProtocol, TaskManagerProtocol
    from aiperf.common.tokenizer import Tokenizer


BaseModelT = TypeVar("BaseModelT", bound="AIPerfBaseModel")
ClassEnumT = TypeVar("ClassEnumT", bound="CaseInsensitiveStrEnum")
ClassProtocolT = TypeVar("ClassProtocolT", bound=Any)
CommAddressType = Union["CommAddress", str]
CommandCallbackMapT = dict["CommandType", Callable[["CommandMessage"], Awaitable[Any]]]
ConfigT = TypeVar("ConfigT", bound=Any, covariant=True)
HooksMixinT = TypeVar("HooksMixinT", bound="HooksMixin")
InputT = TypeVar("InputT", bound=Any)
LifecycleMixinT = TypeVar("LifecycleMixinT", bound="AIPerfLifecycleMixin")
MessageCallbackMapT = dict["MessageTypeT", Callable[["MessageT"], Any] | list[Callable[["MessageT"], Any]]]  # fmt: skip
MessageOutputT = TypeVar("MessageOutputT", bound="Message")
MessageT = TypeVar("MessageT", bound="Message")
MessageTypeT = MessageType | str
ModelEndpointInfoT = TypeVar("ModelEndpointInfoT", bound="ModelEndpointInfo")
OutputT = TypeVar("OutputT", bound=Any)
ParsedResponseRecordT = TypeVar("ParsedResponseRecordT", bound="ParsedResponseRecord")
RawRequestT = TypeVar("RawRequestT", bound=Any, contravariant=True)
RawResponseT = TypeVar("RawResponseT", bound=Any, contravariant=True)
RequestInputT = TypeVar("RequestInputT", bound=Any, contravariant=True)
RequestOutputT = TypeVar("RequestOutputT", bound=Any, covariant=True)
RequestRecordT = TypeVar("RequestRecordT", bound="RequestRecord")
ResponseDataT = TypeVar("ResponseDataT", bound="ResponseData")
ResponseT = TypeVar("ResponseT", bound=Any, covariant=True)
ServiceProtocolT = TypeVar("ServiceProtocolT", bound="ServiceProtocol")
ServiceRunInfoT = TypeVar("ServiceRunInfoT", bound="ServiceRunInfo")
ServiceTypeT = ServiceType | str
TaskManagerProtocolT = TypeVar("TaskManagerProtocolT", bound="TaskManagerProtocol")
TokenizerT = TypeVar("TokenizerT", bound="Tokenizer")
TurnT = TypeVar("TurnT", bound="Turn")
