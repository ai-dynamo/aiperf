# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from cyclopts import Group


class Groups:
    """Groups for the CLI.

    NOTE: The order of these groups are the order they will be displayed in the help text.
    """

    ENDPOINT = Group.create_ordered("Endpoint")
    INPUT = Group.create_ordered("Input")
    OUTPUT = Group.create_ordered("Output")
    TOKENIZER = Group.create_ordered("Tokenizer")
    LOAD_GENERATOR = Group.create_ordered("Load Generator")
    MULTI_RUN = Group.create_ordered("Multi-Run Confidence Reporting")
    CONVERSATION_INPUT = Group.create_ordered("Conversation Input")
    INPUT_SEQUENCE_LENGTH = Group.create_ordered("Input Sequence Length (ISL)")
    OUTPUT_SEQUENCE_LENGTH = Group.create_ordered("Output Sequence Length (OSL)")
    PROMPT = Group.create_ordered("Prompt")
    PREFIX_PROMPT = Group.create_ordered("Prefix Prompt")
    RANKINGS = Group.create_ordered("Rankings")
    SYNTHESIS = Group.create_ordered("Synthesis")
    AUDIO_INPUT = Group.create_ordered("Audio Input")
    IMAGE_INPUT = Group.create_ordered("Image Input")
    VIDEO_INPUT = Group.create_ordered("Video Input")
    SERVICE = Group.create_ordered("Service")
    SERVER_METRICS = Group.create_ordered("Server Metrics")
    TELEMETRY = Group.create_ordered("Telemetry")
    UI = Group.create_ordered("UI")
    WORKERS = Group.create_ordered("Workers")
    DEVELOPER = Group.create_ordered("Developer")
    ZMQ_COMMUNICATION = Group.create_ordered("ZMQ Communication")
