# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from aiperf.common.aiperf_logger import AIPerfLogger
from aiperf.common.models import SpecDecodeAcceptanceRecord
from aiperf.common.models.record_models import find_last_non_empty_usage
from aiperf.spec_decode._shared import find_spec_decode_payload

if TYPE_CHECKING:
    from aiperf.common.models import ParsedResponse

_logger = AIPerfLogger(__name__)

ENGINE = "llamacpp"

# llama.cpp publishes only these two aggregate draft counters per request, and
# only when the request drafted at least one token. They are absent from every
# other engine's payload, so they discriminate on their own.
_LLAMACPP_SIGNATURE_KEYS = ("draft_n", "draft_n_accepted")


def is_llamacpp_payload(payload: Any) -> bool:
    """Return True when ``payload`` is llama.cpp's draft-counter ``timings`` shape."""
    return isinstance(payload, dict) and all(
        key in payload for key in _LLAMACPP_SIGNATURE_KEYS
    )


class LlamaCppSpecDecodeAdapter:
    """Fills the aggregate-only acceptance record from llama.cpp's ``timings``.

    Reads the top-level ``timings`` object on a completion response, which
    llama.cpp attaches when the server runs with a draft model. Only the draft
    counters ``draft_n`` and ``draft_n_accepted`` are used. llama.cpp does not
    publish a per-request verification-step count, so the step-derived fields
    (``num_spec_steps``, ``mean_acceptance_length``, ``acceptance_histogram``)
    stay None. Deriving steps from ``predicted_n`` would be wrong: accepted
    totals are updated before stop-string truncation, so the difference
    misstates the acceptance length.
    """

    @classmethod
    def can_adapt(cls, responses: list[ParsedResponse]) -> bool:
        return is_llamacpp_payload(find_spec_decode_payload(responses))

    @classmethod
    def adapt(
        cls, responses: list[ParsedResponse]
    ) -> SpecDecodeAcceptanceRecord | None:
        payload = find_spec_decode_payload(responses)
        if payload is None:
            return None

        try:
            drafted = payload["draft_n"]
            accepted = payload["draft_n_accepted"]
            if type(drafted) is not int or type(accepted) is not int:
                raise TypeError(
                    f"llama.cpp draft counters must be integers, got {drafted!r}, "
                    f"{accepted!r}"
                )
            if drafted <= 0:
                return None
            usage = find_last_non_empty_usage(responses)
            return SpecDecodeAcceptanceRecord(
                engine=ENGINE,
                draft_acceptance_rate=accepted / drafted,
                num_accepted_draft_tokens=accepted,
                num_draft_tokens=drafted,
                completion_tokens=usage.completion_tokens if usage else None,
            )
        except (TypeError, ValueError, KeyError, ValidationError) as e:
            error = e
            _logger.warning(
                lambda: f"Ignoring malformed llama.cpp spec-decode payload "
                f"{payload!r}: {error!r}"
            )
            return None
