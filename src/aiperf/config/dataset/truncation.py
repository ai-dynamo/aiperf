# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Multi-turn truncation handling shared by every dataset variant.

``SyntheticDataset``, ``FileDataset``, and ``PublicDataset`` each inherit
``EndOnTruncationMixin`` so the option behaves the same regardless of where the
conversations came from. The ``None`` default is resolved once, before services
start, into ``ResolvedConfig.end_conversation_on_truncation``.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import Field

from aiperf.config.base import BaseConfig


class EndOnTruncationMixin(BaseConfig):
    """Adds ``end_conversation_on_truncation`` to a dataset variant."""

    end_conversation_on_truncation: Annotated[
        bool | None,
        Field(
            default=None,
            description="End a multi-turn conversation as soon as one of its turns "
            "is truncated by the output-token cap (the server reports "
            "finish_reason=length, stop_reason=max_tokens, or an incomplete "
            "Responses API reply), and skip that conversation's remaining turns. "
            "The truncated turn itself is still measured. Follow-up prompts assume "
            "a complete prior answer, so replaying a cut-off one sends later turns "
            "off-distribution. None uses the dataset's default: on for SPEED-Bench "
            "qualitative datasets and spec_al_mtbench, off otherwise. Rejected at "
            "startup for DAG datasets and agentic replay.",
        ),
    ]


def end_conversation_on_truncation_enabled(dataset: object) -> bool:
    """Resolve a dataset's effective ``end_conversation_on_truncation``.

    The explicit setting wins; ``None`` takes the public dataset loader's
    ``end_conversation_on_truncation`` metadata, else False. A pure function of
    the dataset config, so every service (including Kubernetes pods, which never
    run the CLI resolver chain) derives the same value from the serialized run.
    """
    explicit = getattr(dataset, "end_conversation_on_truncation", None)
    if explicit is not None:
        return explicit
    from aiperf.config.dataset.config import PublicDataset
    from aiperf.plugin import plugins

    if not isinstance(dataset, PublicDataset):
        return False
    return plugins.get_public_dataset_loader_metadata(
        dataset.dataset
    ).end_conversation_on_truncation
