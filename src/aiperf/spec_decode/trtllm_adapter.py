# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from aiperf.common.aiperf_logger import AIPerfLogger
from aiperf.common.models import SpecDecodeAcceptanceRecord
from aiperf.common.models.record_models import find_last_non_empty_usage
from aiperf.spec_decode._shared import (
    find_spec_decode_payload,
    inflate_dense_histogram,
)

if TYPE_CHECKING:
    from aiperf.common.models import ParsedResponse

_logger = AIPerfLogger(__name__)

ENGINE = "tensorrt_llm"

# Keys that identify a payload as TensorRT-LLM's ``speculative_decoding`` shape.
# Both are TRT-LLM's own names for these quantities and neither appears in
# vLLM's payload, which uses ``num_accepted_draft_tokens`` / ``num_draft_tokens``
# -- so the two signatures are disjoint in both directions and each adapter
# claims only its own payloads. The keys the two engines *do* share
# (``acceptance_histogram``, ``num_spec_steps``, ``num_spec_tokens``) are
# deliberately not used to discriminate.
_TRTLLM_SIGNATURE_KEYS = ("total_accepted_draft_tokens", "total_draft_tokens")


def _is_trtllm_payload(payload: Any) -> bool:
    return isinstance(payload, dict) and all(
        key in payload for key in _TRTLLM_SIGNATURE_KEYS
    )


class TRTLLMSpecDecodeAdapter:
    """Fills the acceptance record from TensorRT-LLM's ``speculative_decoding``.

    Reads the per-choice ``speculative_decoding`` object emitted when the server
    sets ``per_request_spec_decode_stats`` in the YAML passed to
    ``--extra_llm_api_options``. That is the whole opt-in -- the client sends
    nothing extra. Present on chat and completions; in streaming
    it rides the terminal chunk's choice (the one carrying ``finish_reason``),
    alongside TRT-LLM's existing ``avg_decoded_tokens_per_iter``.

    Two differences from vLLM's payload, both deliberate upstream:

    - **No mean acceptance length.** TRT-LLM already reports it per choice as
      ``avg_decoded_tokens_per_iter``; duplicating it inside the block would let
      the two drift. This adapter derives it from the counts instead, so the
      value can never disagree with the histogram it is reported beside.
    - **Its own field names** -- ``acceptance_rate`` and ``total_*`` rather than
      ``draft_acceptance_rate`` and ``num_*`` -- matching the vocabulary TRT-LLM
      already uses internally for these counters.

    ``per_step_accepted`` / ``per_step_drafted`` are never populated: TRT-LLM
    keeps per-*position* vectors, not per-step sequences, so it has no per-step
    data to report. Support is PyTorch-backend only; the C++/TRT path has no
    per-position vectors and emits nothing.
    """

    @classmethod
    def can_adapt(cls, responses: list[ParsedResponse]) -> bool:
        return _is_trtllm_payload(find_spec_decode_payload(responses))

    @classmethod
    def adapt(
        cls, responses: list[ParsedResponse]
    ) -> SpecDecodeAcceptanceRecord | None:
        payload = find_spec_decode_payload(responses)
        if payload is None:
            return None

        try:
            num_spec_tokens = payload.get("num_spec_tokens")
            histogram = inflate_dense_histogram(
                payload["acceptance_histogram"], num_spec_tokens
            )
            num_spec_steps = payload["num_spec_steps"]
            num_accepted = payload["total_accepted_draft_tokens"]
            if num_spec_steps <= 0:
                # The record's own definition of mean acceptance length divides
                # by this. TRT-LLM omits the block entirely for a request that
                # never drafted, so reaching here means a malformed payload
                # rather than a legitimately step-free request.
                raise ValueError(
                    f"num_spec_steps must be positive, got {num_spec_steps}"
                )
            usage = find_last_non_empty_usage(responses)
            return SpecDecodeAcceptanceRecord(
                engine=ENGINE,
                # Derived rather than read: both operands come from this same
                # payload, so the reported length cannot contradict the counts.
                mean_acceptance_length=1 + num_accepted / num_spec_steps,
                draft_acceptance_rate=payload["acceptance_rate"],
                acceptance_histogram=histogram,
                num_accepted_draft_tokens=num_accepted,
                num_draft_tokens=payload["total_draft_tokens"],
                num_spec_steps=num_spec_steps,
                num_spec_tokens=num_spec_tokens,
                completion_tokens=usage.completion_tokens if usage else None,
            )
        except (
            KeyError,
            TypeError,
            ValueError,
            ZeroDivisionError,
            AttributeError,
            ValidationError,
        ) as e:
            # Degrade to None on an unexpected shape: a single malformed payload
            # must not abort a run. can_adapt already matched the TRT-LLM
            # signature, so this only fires when a signature-matching body still
            # has a broken value -- including one whose counts violate the
            # record's aggregate identities, which is the case worth dropping
            # rather than trusting.
            error = e
            _logger.warning(
                lambda: f"Ignoring malformed TensorRT-LLM spec-decode payload "
                f"{payload!r}: {error!r}"
            )
            return None
