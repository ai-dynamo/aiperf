# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from aiperf.common.models.record_models import ToolCallResponseData


class TestToolCallResponseDataShape:
    def test_required_field_renamed(self):
        d = ToolCallResponseData(tool_call_text="get_weather('SF')")
        assert d.tool_call_text == "get_weather('SF')"
        assert d.content is None

    def test_old_name_rejected(self):
        with pytest.raises(TypeError):
            ToolCallResponseData(text="get_weather('SF')")

    def test_with_prose_content(self):
        d = ToolCallResponseData(
            tool_call_text="get_weather('SF')",
            content="Let me check the weather.",
        )
        assert d.tool_call_text == "get_weather('SF')"
        assert d.content == "Let me check the weather."

    def test_get_text_combines_content_then_tool_call(self):
        d = ToolCallResponseData(
            tool_call_text="get_weather('SF')",
            content="Let me check the weather. ",
        )
        assert d.get_text() == "Let me check the weather. get_weather('SF')"

    def test_get_text_pure_tool_call(self):
        d = ToolCallResponseData(tool_call_text="get_weather('SF')")
        assert d.get_text() == "get_weather('SF')"
