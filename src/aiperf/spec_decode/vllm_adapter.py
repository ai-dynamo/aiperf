# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

ENGINE = "vllm"

# Keys that identify a payload as vLLM's ``speculative_decoding_stats`` shape.
# ``can_adapt`` checks for these so that, once other engines populate the same
# raw slot, this adapter claims only its own payloads (auto-detection) rather
# than greedily matching on mere presence. Both are always emitted by vLLM,
# including the zero-step case (``acceptance_histogram`` is ``{}`` but present).
_VLLM_SIGNATURE_KEYS = ("acceptance_histogram", "num_spec_steps")


def _is_vllm_payload(payload: Any) -> bool:
    return isinstance(payload, dict) and all(
        key in payload for key in _VLLM_SIGNATURE_KEYS
    )


def _find_spec_decode_payload(
    responses: list[ParsedResponse],
) -> dict[str, Any] | None:
    """Return the last non-empty ``spec_decode_stats`` payload across responses.

    vLLM attaches the payload to a single choice: the finish-reason chunk in
    streaming, or the sole choice non-streaming. Walking from the end mirrors
    ``find_last_non_empty_usage`` and tolerates either layout.
    """
    for response in reversed(responses):
        stats = response.spec_decode_stats
        if stats:
            return stats
    return None


class VLLMSpecDecodeAdapter:
    """Fills the acceptance record from vLLM's ``speculative_decoding_stats``.

    Reads the per-choice ``speculative_decoding_stats`` object emitted by vLLM
    when the server runs with ``--per-request-spec-decode-stats``. Present on
    chat and completions, streaming and non-streaming. The object's histogram
    keys are JSON strings, so they are int-cast into the neutral record.

    The field names and shape track vLLM PR
    https://github.com/vllm-project/vllm/pull/48915; its per-request
    acceptance-metrics feature doc is the authoritative wire-format reference.
    """

    @classmethod
    def can_adapt(cls, responses: list[ParsedResponse]) -> bool:
        return _is_vllm_payload(_find_spec_decode_payload(responses))

    @classmethod
    def adapt(
        cls, responses: list[ParsedResponse]
    ) -> SpecDecodeAcceptanceRecord | None:
        payload = _find_spec_decode_payload(responses)
        if payload is None:
            return None

        try:
            histogram = {
                int(j): count for j, count in payload["acceptance_histogram"].items()
            }
            usage = find_last_non_empty_usage(responses)
            return SpecDecodeAcceptanceRecord(
                engine=ENGINE,
                mean_acceptance_length=payload["mean_acceptance_length"],
                draft_acceptance_rate=payload["draft_acceptance_rate"],
                acceptance_histogram=histogram,
                num_accepted_draft_tokens=payload["num_accepted_draft_tokens"],
                num_draft_tokens=payload["num_draft_tokens"],
                num_spec_steps=payload["num_spec_steps"],
                num_spec_tokens=payload.get("num_spec_tokens"),
                completion_tokens=usage.completion_tokens if usage else None,
                per_step_accepted=payload.get("per_step_accepted"),
                per_step_drafted=payload.get("per_step_drafted"),
            )
        except (KeyError, TypeError, ValueError, AttributeError, ValidationError) as e:
            # Degrade to None on an unexpected shape: a single malformed
            # payload must not abort a run. can_adapt already matched the vLLM
            # signature, so this only fires when a signature-matching body still
            # has a broken value (a non-dict histogram, a negative count, ...).
            _logger.warning(
                f"Ignoring malformed vLLM spec-decode payload {payload!r}: {e!r}"
            )
            return None
