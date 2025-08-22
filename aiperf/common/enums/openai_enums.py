# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.enums.base_enums import CaseInsensitiveStrEnum


class OpenAIObjectType(CaseInsensitiveStrEnum):
    """OpenAI object type definitions.
    See: https://platform.openai.com/docs/api-reference
    """

    CHAT_COMPLETION = "chat.completion"
    CHAT_COMPLETION_CHUNK = "chat.completion.chunk"
    COMPLETION = "completion"
    EMBEDDING = "embedding"
    LIST = "list"
    RESPONSE = "response"
    TEXT_COMPLETION = "text_completion"
