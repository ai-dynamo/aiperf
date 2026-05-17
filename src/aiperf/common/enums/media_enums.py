# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.enums.base_enums import CaseInsensitiveStrEnum
from aiperf.common.enums.enums import AudioFormat as AudioFormat
from aiperf.common.enums.enums import ImageFormat as ImageFormat
from aiperf.common.enums.enums import MediaType as MediaType
from aiperf.common.enums.enums import VideoAudioCodec as VideoAudioCodec
from aiperf.common.enums.enums import VideoFormat as VideoFormat
from aiperf.common.enums.enums import VideoJobStatus as VideoJobStatus
from aiperf.common.enums.enums import VideoSynthType as VideoSynthType


class ContentType(CaseInsensitiveStrEnum):
    """Defines the semantic type for synthetic text content."""

    RANDOM_TOKENS = "random_tokens"
    """Generate random token sequences."""

    SYSTEM_PROMPT = "system_prompt"
    """Generate system prompt style content."""

    CONTEXT = "context"
    """Generate contextual information."""

    INSTRUCTION = "instruction"
    """Generate instruction-style content."""

    QUESTION = "question"
    """Generate question-style content."""
