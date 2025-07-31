# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
    CONVERSATION_INPUT = Group.create_ordered("Conversation Input")
    INPUT_SEQUENCE_LENGTH = Group.create_ordered("Input Sequence Length (ISL)")
    OUTPUT_SEQUENCE_LENGTH = Group.create_ordered("Output Sequence Length (OSL)")
    PROMPT = Group.create_ordered("Prompt")
    PREFIX_PROMPT = Group.create_ordered("Prefix Prompt")
    AUDIO_INPUT = Group.create_ordered("Audio Input")
    IMAGE_INPUT = Group.create_ordered("Image Input")
    MEASUREMENT = Group.create_ordered("Measurement")
    SERVICE = Group.create_ordered("Service")
    WORKERS = Group.create_ordered("Workers")
    DEVELOPER = Group.create_ordered("Developer")
