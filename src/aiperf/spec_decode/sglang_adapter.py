# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import ValidationError

from aiperf.common.aiperf_logger import AIPerfLogger
from aiperf.common.models import SpecDecodeAcceptanceRecord
from aiperf.common.models.record_models import find_last_non_empty_usage
from aiperf.spec_decode._shared import find_spec_decode_payload, inflate_dense_histogram

if TYPE_CHECKING:
    from aiperf.common.models import ParsedResponse

_logger = AIPerfLogger(__name__)

# These SGLang counter names do not occur in vLLM or TensorRT-LLM payloads.
_SGLANG_SIGNATURE_KEYS = (
    "spec_num_correct_drafts",
    "spec_num_proposed_drafts",
    "spec_verify_ct",
)


class SGLangSpecDecodeAdapter:
    """Normalize SGLang's ``sglext.spec_tokens_details`` response extension.

    SGLang v0.5.19+ emits it when the request sets
    ``return_spec_tokens_details: true``. Multi-sequence lists are excluded by
    the endpoint. No fixed draft bound or ordered per-step arrays are reported.
    """

    @classmethod
    def can_adapt(cls, responses: list[ParsedResponse]) -> bool:
        payload = find_spec_decode_payload(responses)
        return isinstance(payload, dict) and all(
            key in payload for key in _SGLANG_SIGNATURE_KEYS
        )

    @classmethod
    def adapt(
        cls, responses: list[ParsedResponse]
    ) -> SpecDecodeAcceptanceRecord | None:
        payload = find_spec_decode_payload(responses)
        if payload is None:
            return None

        try:
            accepted, drafted, steps = (payload[key] for key in _SGLANG_SIGNATURE_KEYS)
            # Reject coercible counts before Pydantic can accept floats or booleans.
            if any(type(value) is not int for value in (accepted, drafted, steps)):
                raise ValueError(
                    "SGLang speculative decoding counters must be integers"
                )
            histogram = inflate_dense_histogram(
                payload["spec_correct_drafts_histogram"], None
            )
            usage = find_last_non_empty_usage(responses)
            return SpecDecodeAcceptanceRecord(
                engine="sglang",
                # SGLang's spec_accept_length includes the prefill output token.
                mean_acceptance_length=1 + accepted / steps if steps else 1.0,
                draft_acceptance_rate=accepted / drafted if drafted else 0.0,
                acceptance_histogram=histogram,
                num_accepted_draft_tokens=accepted,
                num_draft_tokens=drafted,
                num_spec_steps=steps,
                completion_tokens=usage.completion_tokens if usage else None,
            )
        except (KeyError, TypeError, ValueError, ValidationError) as e:
            # Exception variables are cleared before a deferred log call.
            error = e
            _logger.warning(
                lambda: f"Ignoring malformed SGLang spec-decode payload "
                f"{payload!r}: {error!r}"
            )
            return None
