# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLI parameter helpers for cyclopts integration.

Provides ``CLIParameter`` and ``Groups`` used by ``CLIConfig`` fields to
build ``Annotated[type, Field(...), CLIParameter(...)]`` field descriptors.
"""

from dataclasses import dataclass

from cyclopts import Group, Parameter


class CLIParameter(Parameter):
    """Configuration for a CLI parameter.

    This is a subclass of the cyclopts.Parameter class that includes the default configuration AIPerf uses
    for all of its CLI parameters. This is used to ensure that the CLI parameters are consistent across all
    of the AIPerf config.
    """

    def __init__(self, *args, negative: bool | str = False, **kwargs):
        super().__init__(*args, show_env_var=False, negative=negative, **kwargs)


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
