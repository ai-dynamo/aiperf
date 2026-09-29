# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from aiperf.common.aiperf_logger import AIPerfLogger
from aiperf.common.models import SpecDecodeAcceptanceRecord
from aiperf.common.models.record_models import find_last_non_empty_usage

if TYPE_CHECKING:
    from aiperf.common.models import ParsedResponse

_logger = AIPerfLogger(__name__)

ENGINE = "llamacpp"

# Keys that identify a payload as llama.cpp's timings-based spec-decode shape.
_LLAMACPP_SIGNATURE_KEYS = ("draft_n", "draft_n_accepted")


def _is_llamacpp_payload(payload: Any) -> bool:
    return isinstance(payload, dict) and all(
        key in payload for key in _LLAMACPP_SIGNATURE_KEYS
    )


def _find_spec_decode_payload(
    responses: list[ParsedResponse],
) -> dict[str, Any] | None:
    """Return the last spec_decode_stats payload that matches the llama.cpp shape."""
    for response in reversed(responses):
        stats = response.spec_decode_stats
        if stats and _is_llamacpp_payload(stats):
            return stats
    return None


class LlamaCppSpecDecodeAdapter:
    """Fills the acceptance record from llama.cpp's per-response ``timings`` object.

    llama.cpp includes spec-decode counters in the top-level ``timings`` field
    of every response when the server runs with a draft model::

        "timings": {
            "draft_n":          <total draft tokens proposed>,
            "draft_n_accepted": <total draft tokens accepted>,
            "predicted_n":      <total output tokens>,
            ...
        }

    ``num_spec_steps`` is derived from ``predicted_n - draft_n_accepted`` because
    each verification step emits ``accepted_j + 1`` tokens (accepted drafts plus
    the guaranteed bonus token), so ``predicted_n == num_accepted + num_steps``.

    The acceptance histogram is reconstructed from the aggregate totals via
    integer-division bucketing to satisfy ``SpecDecodeAcceptanceRecord`` invariants.
    """

    @classmethod
    def can_adapt(cls, responses: list[ParsedResponse]) -> bool:
        return _is_llamacpp_payload(_find_spec_decode_payload(responses))

    @classmethod
    def adapt(
        cls, responses: list[ParsedResponse]
    ) -> SpecDecodeAcceptanceRecord | None:
        payload = _find_spec_decode_payload(responses)
        if payload is None:
            return None

        try:
            num_draft_tokens: int = int(payload["draft_n"])
            num_accepted: int = int(payload["draft_n_accepted"])
            predicted_n: int = int(payload["predicted_n"])

            if num_accepted > num_draft_tokens:
                raise ValueError(
                    f"draft_n_accepted ({num_accepted}) > draft_n ({num_draft_tokens})"
                )

            num_spec_steps = predicted_n - num_accepted
            if num_spec_steps <= 0:
                raise ValueError(
                    f"derived num_spec_steps ({num_spec_steps}) is non-positive; "
                    f"predicted_n={predicted_n}, draft_n_accepted={num_accepted}"
                )

            # Distribute accepted tokens as evenly as possible across steps so
            # both SpecDecodeAcceptanceRecord histogram invariants hold.
            base, remainder = divmod(num_accepted, num_spec_steps)
            histogram: dict[int, int] = {}
            if remainder:
                histogram[base + 1] = remainder
            if num_spec_steps - remainder:
                histogram[base] = num_spec_steps - remainder

            usage = find_last_non_empty_usage(responses)
            return SpecDecodeAcceptanceRecord(
                engine=ENGINE,
                mean_acceptance_length=1.0 + num_accepted / num_spec_steps,
                draft_acceptance_rate=(
                    num_accepted / num_draft_tokens if num_draft_tokens > 0 else 0.0
                ),
                acceptance_histogram=histogram,
                num_accepted_draft_tokens=num_accepted,
                num_draft_tokens=num_draft_tokens,
                num_spec_steps=num_spec_steps,
                num_spec_tokens=None,
                completion_tokens=usage.completion_tokens if usage else None,
            )
        except (KeyError, TypeError, ValueError, AttributeError, ValidationError) as e:
            error = e
            _logger.warning(
                lambda: f"Ignoring malformed llama.cpp spec-decode timings {payload!r}: {error!r}"
            )
            return None
