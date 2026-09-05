# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Annotated, Self

from pydantic import Field, model_validator

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
        "verification steps that accepted exactly j draft tokens. Engine "
        "adapters populate it (e.g. vLLM inflates its dense on-the-wire "
        "list[int], dropping zero-count buckets); keys are integers and "
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

    @model_validator(mode="after")
    def _check_aggregate_invariants(self) -> Self:
        """Reject records whose aggregate counts contradict each other.

        Only the exact integer identities are enforced -- the histogram and the
        per-step arrays are just the per-step accepted/proposed tallies viewed
        another way, so their counts, lengths, and sums must reconcile with
        num_spec_steps / num_accepted_draft_tokens / num_draft_tokens, and one
        cannot accept more drafts than were proposed (per step or overall).
        These are definitional (not accounting choices), so a violation means a
        corrupt payload, not benign engine drift. Float relationships (mean/rate)
        are intentionally not re-derived here to avoid rounding false-positives.
        The vLLM adapter catches the resulting ValidationError and degrades to
        None.
        """
        step_count = sum(self.acceptance_histogram.values())
        if step_count != self.num_spec_steps:
            raise ValueError(
                f"acceptance_histogram counts sum to {step_count}, but "
                f"num_spec_steps is {self.num_spec_steps}"
            )
        accepted = sum(j * count for j, count in self.acceptance_histogram.items())
        if accepted != self.num_accepted_draft_tokens:
            raise ValueError(
                f"acceptance_histogram j-weighted sum is {accepted}, but "
                f"num_accepted_draft_tokens is {self.num_accepted_draft_tokens}"
            )
        if self.num_accepted_draft_tokens > self.num_draft_tokens:
            raise ValueError(
                f"num_accepted_draft_tokens ({self.num_accepted_draft_tokens}) "
                f"exceeds num_draft_tokens ({self.num_draft_tokens})"
            )
        self._check_per_step_invariants()
        return self

    def _check_per_step_invariants(self) -> None:
        """Detailed level: the ordered per-step arrays (each optional) must
        reconcile with the aggregates the same way the histogram does."""
        if self.per_step_accepted is not None:
            if len(self.per_step_accepted) != self.num_spec_steps:
                raise ValueError(
                    f"per_step_accepted has {len(self.per_step_accepted)} entries, "
                    f"but num_spec_steps is {self.num_spec_steps}"
                )
            if sum(self.per_step_accepted) != self.num_accepted_draft_tokens:
                raise ValueError(
                    f"per_step_accepted sums to {sum(self.per_step_accepted)}, but "
                    f"num_accepted_draft_tokens is {self.num_accepted_draft_tokens}"
                )
        if self.per_step_drafted is not None:
            if len(self.per_step_drafted) != self.num_spec_steps:
                raise ValueError(
                    f"per_step_drafted has {len(self.per_step_drafted)} entries, "
                    f"but num_spec_steps is {self.num_spec_steps}"
                )
            if sum(self.per_step_drafted) != self.num_draft_tokens:
                raise ValueError(
                    f"per_step_drafted sums to {sum(self.per_step_drafted)}, but "
                    f"num_draft_tokens is {self.num_draft_tokens}"
                )
        if (
            self.per_step_accepted is not None
            and self.per_step_drafted is not None
            and any(
                a > d
                for a, d in zip(
                    self.per_step_accepted, self.per_step_drafted, strict=True
                )
            )
        ):
            raise ValueError(
                "a verification step accepted more drafts than it proposed "
                "(per_step_accepted[i] > per_step_drafted[i])"
            )
