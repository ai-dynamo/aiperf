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

ENGINE = "vllm"

# Keys that identify a payload as vLLM's ``speculative_decoding`` shape.
# ``can_adapt`` checks for these so that, once other engines populate the same
# raw slot, this adapter claims only its own payloads (auto-detection) rather
# than greedily matching on mere presence. All three are always emitted by vLLM,
# including the zero-step case (``acceptance_histogram`` is all-zero but present).
#
# ``mean_acceptance_length`` is the discriminating key and must stay here.
# TensorRT-LLM emits ``acceptance_histogram`` and ``num_spec_steps`` too, so a
# signature without a vLLM-unique key matches a TRT-LLM payload -- and since the
# shared field names parse, this adapter would build a record and stamp
# ``engine="vllm"`` on another engine's numbers, with no error to notice.
# TRT-LLM omits mean acceptance length by design, reporting it per choice as
# ``avg_decoded_tokens_per_iter``. Requiring it costs nothing: ``adapt`` already
# reads it unguarded, so a payload lacking it could never produce a record.
_VLLM_SIGNATURE_KEYS = (
    "acceptance_histogram",
    "num_spec_steps",
    "mean_acceptance_length",
)


def _is_vllm_payload(payload: Any) -> bool:
    return isinstance(payload, dict) and all(
        key in payload for key in _VLLM_SIGNATURE_KEYS
    )


class VLLMSpecDecodeAdapter:
    """Fills the acceptance record from vLLM's ``metrics.speculative_decoding``.

    Reads the response-root ``metrics.speculative_decoding`` object emitted by
    vLLM when the server runs with ``--per-request-spec-decode-metrics``
    (``summary`` or ``detailed``). Present on chat and completions, streaming and
    non-streaming. Its ``acceptance_histogram`` is a dense ``list[int]`` (index j
    holds the number of steps that accepted exactly j draft tokens), inflated
    into the neutral record's sparse ``{j: count}`` map with zero-count buckets
    dropped.

    The field names and shape track vLLM PR
    https://github.com/vllm-project/vllm/pull/48915; its per-request
    acceptance-metrics feature doc is the authoritative wire-format reference.
    """

    @classmethod
    def can_adapt(cls, responses: list[ParsedResponse]) -> bool:
        return _is_vllm_payload(find_spec_decode_payload(responses))

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
            usage = find_last_non_empty_usage(responses)
            return SpecDecodeAcceptanceRecord(
                engine=ENGINE,
                mean_acceptance_length=payload["mean_acceptance_length"],
                draft_acceptance_rate=payload["draft_acceptance_rate"],
                acceptance_histogram=histogram,
                num_accepted_draft_tokens=payload["num_accepted_draft_tokens"],
                num_draft_tokens=payload["num_draft_tokens"],
                num_spec_steps=payload["num_spec_steps"],
                num_spec_tokens=num_spec_tokens,
                completion_tokens=usage.completion_tokens if usage else None,
                per_step_accepted=payload.get("per_step_accepted"),
                per_step_drafted=payload.get("per_step_drafted"),
            )
        except (KeyError, TypeError, ValueError, AttributeError, ValidationError) as e:
            # Degrade to None on an unexpected shape: a single malformed
            # payload must not abort a run. can_adapt already matched the vLLM
            # signature, so this only fires when a signature-matching body still
            # has a broken value (a non-dict histogram, a negative count, ...).
            # Rebind to a normal local so the lazy lambda can reference it (the
            # ``except ... as e`` name is cleared at block exit).
            error = e
            _logger.warning(
                lambda: f"Ignoring malformed vLLM spec-decode payload {payload!r}: {error!r}"
            )
            return None
