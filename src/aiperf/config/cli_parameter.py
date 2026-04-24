# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLI parameter helpers for cyclopts integration.

Provides ``CLIParameter``, ``DisableCLI``, ``Groups``, and the
``annotated_type()`` factory used by ``CLIModel`` to build
``Annotated[type, Field(...), CLIParameter(...)]`` field descriptors.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Annotated, Any

from cyclopts import Group, Parameter
from pydantic import BeforeValidator, Field


class CLIParameter(Parameter):
    """Configuration for a CLI parameter.

    This is a subclass of the cyclopts.Parameter class that includes the default configuration AIPerf uses
    for all of its CLI parameters. This is used to ensure that the CLI parameters are consistent across all
    of the AIPerf config.
    """

    def __init__(self, *args, negative: bool = False, **kwargs):
        super().__init__(*args, show_env_var=False, negative=negative, **kwargs)


class DisableCLI(CLIParameter):
    """Configuration for a CLI parameter that is disabled.

    This is a subclass of the CLIParameter class that is used to set a CLI parameter to disabled.
    """

    def __init__(self, reason: str = "Not supported via command line", *args, **kwargs):
        super().__init__(*args, parse=False, **kwargs)


@dataclass(frozen=True)
class Groups:
    """Cyclopts help groups controlling display order in --help."""

    ENDPOINT = Group.create_ordered("Endpoint")
    INPUT = Group.create_ordered("Input")
    FIXED_SCHEDULE = Group.create_ordered("Fixed Schedule")
    GOODPUT = Group.create_ordered("Goodput")
    OUTPUT = Group.create_ordered("Output")
    HTTP_TRACE = Group.create_ordered("HTTP Trace")
    TOKENIZER = Group.create_ordered("Tokenizer")
    LOAD_GENERATOR = Group.create_ordered("Load Generator")
    WARMUP = Group.create_ordered("Warmup")
    USER_CENTRIC = Group.create_ordered("User-Centric Rate")
    REQUEST_CANCELLATION = Group.create_ordered("Request Cancellation")
    CONVERSATION_INPUT = Group.create_ordered("Conversation Input")
    ISL = Group.create_ordered("Input Sequence Length (ISL)")
    OSL = Group.create_ordered("Output Sequence Length (OSL)")
    PROMPT = Group.create_ordered("Prompt")
    PREFIX_PROMPT = Group.create_ordered("Prefix Prompt")
    RANKINGS = Group.create_ordered("Rankings")
    SYNTHESIS = Group.create_ordered("Synthesis")
    AUDIO_INPUT = Group.create_ordered("Audio Input")
    IMAGE_INPUT = Group.create_ordered("Image Input")
    VIDEO_INPUT = Group.create_ordered("Video Input")
    SERVICE = Group.create_ordered("Service")
    SERVER_METRICS = Group.create_ordered("Server Metrics")
    GPU_TELEMETRY = Group.create_ordered("GPU Telemetry")
    UI = Group.create_ordered("UI")
    WORKERS = Group.create_ordered("Workers")
    ZMQ_COMMUNICATION = Group.create_ordered("ZMQ Communication")
    ACCURACY = Group.create_ordered("Accuracy")
    MULTI_RUN = Group.create_ordered("Multi-Run")


def annotated_type(
    tp: type,
    flags: str | tuple[str, ...] | None,
    group: Group | None,
    description: str,
    *,
    default: Any = None,
    validators: list[Callable] | None = None,
    parse: bool = True,
    consume_multiple: bool = False,
    negative: bool = False,
    show_choices: bool = True,
) -> type:
    """Build an Annotated type for a CLI field.

    Returns ``Annotated[tp, Field(...), BeforeValidator(...), CLIParameter(...)]``
    ready for Pydantic + cyclopts consumption.
    """
    metadata: list[Any] = [Field(default=default, description=description)]
    if validators:
        for v in validators:
            metadata.append(BeforeValidator(v))
    if not parse:
        metadata.append(DisableCLI())
    else:
        kw: dict[str, Any] = {"name": flags, "group": group}
        if consume_multiple:
            kw["consume_multiple"] = True
        if negative is None:
            kw["negative"] = None
        if not show_choices:
            kw["show_choices"] = False
        metadata.append(CLIParameter(**kw))
    return Annotated[tuple([tp, *metadata])]
