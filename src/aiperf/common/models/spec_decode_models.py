# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Annotated

from pydantic import Field

from aiperf.common.finite import FiniteFloat
from aiperf.common.models.base_models import AIPerfBaseModel

# Counts (accepted-draft counts j, per-step values, step tallies) are never
# negative. A field-level ge bound cannot apply to a dict/list container, so the
# non-negativity is enforced on the element type instead.
NonNegativeInt = Annotated[int, Field(ge=0)]


class SpecDecodeAcceptanceRecord(AIPerfBaseModel):
    """Engine-neutral per-request speculative-decoding acceptance record.

    One record per request, produced by an engine-specific
    ``SpecDecodeAdapterProtocol`` (e.g. the vLLM adapter) from a raw response
    payload and consumed only by the metrics layer, which never learns which
    engine produced it. Kept tree-agnostic (an aggregate histogram, not
    per-draft-position data that would encode a tree's parent/child structure)
    and adaptive-safe (no fixed ``k`` assumption) so it survives variable-length
    drafting such as DSpark-style adaptive verification.

    The optional ``per_step_*`` arrays are ordered one-entry-per-verify-step
    sequences -- a temporal axis, still tree-agnostic. They are populated only
    when the source engine reports per-step data and stay ``None`` otherwise, so
    downstream metrics must treat them as best-effort.
    """

    engine: str = Field(
        description="Identifier of the serving engine that produced the stats "
        "(e.g. 'vllm'). Set by the adapter so provenance is explicit while the "
        "rest of the record stays engine-neutral.",
    )
    mean_acceptance_length: FiniteFloat = Field(
        description="Mean tokens emitted per verification step including the "
        "always-accepted bonus token: 1 + num_accepted_draft_tokens / "
        "num_spec_steps. Ranges from 1.0 (nothing accepted) to num_spec_tokens "
        "+ 1. The '(j + 1)' acceptance length.",
    )
    draft_acceptance_rate: FiniteFloat = Field(
        description="Fraction of proposed draft tokens that were accepted: "
        "num_accepted_draft_tokens / num_draft_tokens. Draft-only; excludes the "
        "bonus token.",
    )
    acceptance_histogram: dict[NonNegativeInt, NonNegativeInt] = Field(
        description="Sparse map from accepted draft count j to the number of "
        "verification steps that accepted exactly j draft tokens. Keys are "
        "integers (int-cast from the string JSON object keys on the wire); "
        "zero-count buckets are omitted. Excludes the bonus token.",
    )
    num_accepted_draft_tokens: int = Field(
        ge=0,
        description="Total accepted draft tokens for the request, excluding "
        "bonus tokens.",
    )
    num_draft_tokens: int = Field(
        ge=0,
        description="Total proposed draft tokens for the request counted toward "
        "acceptance; the denominator of draft_acceptance_rate. Engines that "
        "discard some proposals before counting report the post-adjustment "
        "total (e.g. vLLM subtracts structured-output-invalidated drafts).",
    )
    num_spec_steps: int = Field(
        ge=0,
        description="Number of verification steps for the request. Equals the "
        "sum of the histogram counts.",
    )
    num_spec_tokens: int | None = Field(
        default=None,
        ge=0,
        description="Maximum draft length per step (k) when the engine has a "
        "fixed per-step bound. None when the engine reports no fixed bound "
        "(e.g. fully variable-length drafting) or does not expose it.",
    )
    completion_tokens: int | None = Field(
        default=None,
        ge=0,
        description="Completion tokens for the request, copied from the "
        "response usage so the metrics layer can normalize acceptance per "
        "output token without re-reading usage. None when usage is absent.",
    )
    per_step_accepted: list[NonNegativeInt] | None = Field(
        default=None,
        description="Ordered accepted-draft count per verification step. "
        "Populated only when the engine reports per-step data; None otherwise.",
    )
    per_step_drafted: list[NonNegativeInt] | None = Field(
        default=None,
        description="Ordered proposed-draft count per verification step. "
        "Records the effective proposal length per step, representing "
        "variable-length drafting without a schema change. Populated only when "
        "the engine reports per-step data; None otherwise.",
    )
