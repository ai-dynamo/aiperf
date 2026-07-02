# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pydantic import ConfigDict, Field

from aiperf.common.enums import CaseInsensitiveStrEnum
from aiperf.common.models import AIPerfBaseModel


class ExgenticHarness(CaseInsensitiveStrEnum):
    CLAUDE_CODE = "claude_code"
    OPENAI_SOLO = "openai_solo"
    SMOLAGENTS_CODE = "smolagents_code"
    TOOL_CALLING = "tool_calling"
    TOOL_CALLING_WITH_SHORTLISTING = "tool_calling_with_shortlisting"


class ExgenticSourceModel(CaseInsensitiveStrEnum):
    DEEPSEEK_V3_2 = "DeepSeek-V3.2"
    KIMI_K2_5 = "Kimi-K2.5"
    CLAUDE_OPUS_4_5 = "claude-opus-4-5"
    GEMINI_3_PRO_PREVIEW = "gemini-3-pro-preview"
    GPT_4_1 = "gpt-4.1"
    GPT_5_2 = "gpt-5.2-2025-12-11"


class ExgenticBenchmark(CaseInsensitiveStrEnum):
    APPWORLD = "appworld"
    BROWSECOMPPLUS = "browsecompplus"
    SWEBENCH = "swebench"
    TAU2_AIRLINE = "tau2_airline"
    TAU2_RETAIL = "tau2_retail"
    TAU2_TELECOM = "tau2_telecom"


class ExgenticDatasetFilters(AIPerfBaseModel):
    model_config = ConfigDict(extra="forbid")

    harness: ExgenticHarness | None = Field(
        default=None,
        description="Source agent harness to replay.",
    )
    source_model: ExgenticSourceModel | None = Field(
        default=None,
        description="Source model recorded by Exgentic, distinct from the target model.",
    )
    benchmark: ExgenticBenchmark | None = Field(
        default=None,
        description="Source benchmark. Available for Exgentic v2 traces.",
    )


ExgenticFilterPair = tuple[ExgenticHarness, ExgenticSourceModel]


def available_filter_values(
    unsupported_filter_pairs: frozenset[ExgenticFilterPair],
    *,
    supports_benchmark_filter: bool,
) -> str:
    """Format filter values that have at least one supported harness/model pair."""
    source_models = ", ".join(
        model.value
        for model in ExgenticSourceModel
        if any(
            (harness, model) not in unsupported_filter_pairs
            for harness in ExgenticHarness
        )
    )
    available = (
        f"harness=[{', '.join(item.value for item in ExgenticHarness)}], "
        f"source_model=[{source_models}]"
    )
    if supports_benchmark_filter:
        return f"{available}, benchmark=[{', '.join(item.value for item in ExgenticBenchmark)}]"
    return available


V1_UNSUPPORTED_FILTER_PAIRS: frozenset[ExgenticFilterPair] = frozenset(
    {
        (ExgenticHarness.OPENAI_SOLO, ExgenticSourceModel.GPT_5_2),
        (ExgenticHarness.SMOLAGENTS_CODE, ExgenticSourceModel.CLAUDE_OPUS_4_5),
        (ExgenticHarness.SMOLAGENTS_CODE, ExgenticSourceModel.GEMINI_3_PRO_PREVIEW),
        (ExgenticHarness.SMOLAGENTS_CODE, ExgenticSourceModel.GPT_5_2),
        (ExgenticHarness.TOOL_CALLING, ExgenticSourceModel.GPT_5_2),
        (
            ExgenticHarness.TOOL_CALLING_WITH_SHORTLISTING,
            ExgenticSourceModel.CLAUDE_OPUS_4_5,
        ),
        (
            ExgenticHarness.TOOL_CALLING_WITH_SHORTLISTING,
            ExgenticSourceModel.GPT_4_1,
        ),
        (
            ExgenticHarness.TOOL_CALLING_WITH_SHORTLISTING,
            ExgenticSourceModel.GPT_5_2,
        ),
    }
)

V2_UNSUPPORTED_FILTER_PAIRS: frozenset[ExgenticFilterPair] = frozenset(
    {
        (ExgenticHarness.CLAUDE_CODE, ExgenticSourceModel.GPT_4_1),
        (ExgenticHarness.OPENAI_SOLO, ExgenticSourceModel.GPT_4_1),
        (ExgenticHarness.SMOLAGENTS_CODE, ExgenticSourceModel.GPT_4_1),
        (ExgenticHarness.TOOL_CALLING, ExgenticSourceModel.GPT_4_1),
        (
            ExgenticHarness.TOOL_CALLING_WITH_SHORTLISTING,
            ExgenticSourceModel.CLAUDE_OPUS_4_5,
        ),
        (
            ExgenticHarness.TOOL_CALLING_WITH_SHORTLISTING,
            ExgenticSourceModel.GPT_4_1,
        ),
        (
            ExgenticHarness.TOOL_CALLING_WITH_SHORTLISTING,
            ExgenticSourceModel.GPT_5_2,
        ),
    }
)
