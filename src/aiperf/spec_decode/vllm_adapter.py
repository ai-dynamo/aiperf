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

# Keys that identify a payload as vLLM's ``speculative_decoding`` shape.
# ``can_adapt`` checks for these so that, once other engines populate the same
# raw slot, this adapter claims only its own payloads (auto-detection) rather
# than greedily matching on mere presence. Both are always emitted by vLLM,
# including the zero-step case (``acceptance_histogram`` is all-zero but present).
_VLLM_SIGNATURE_KEYS = ("acceptance_histogram", "num_spec_steps")


def _is_vllm_payload(payload: Any) -> bool:
    return isinstance(payload, dict) and all(
        key in payload for key in _VLLM_SIGNATURE_KEYS
    )


def _find_spec_decode_payload(
    responses: list[ParsedResponse],
) -> dict[str, Any] | None:
    """Return the last non-empty ``spec_decode_stats`` payload across responses.

    vLLM attaches the payload once per request at the response root: on the body
    non-streaming, or on the trailing usage chunk streaming. Walking from the end
    mirrors ``find_last_non_empty_usage`` and tolerates either layout.
    """
    for response in reversed(responses):
        stats = response.spec_decode_stats
        if stats:
            return stats
    return None


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
        return _is_vllm_payload(_find_spec_decode_payload(responses))

    @classmethod
    def adapt(
        cls, responses: list[ParsedResponse]
    ) -> SpecDecodeAcceptanceRecord | None:
        payload = _find_spec_decode_payload(responses)
        if payload is None:
            return None

        try:
            # vLLM sends a dense ``list[int]`` (index j -> step count); inflate
            # into the neutral record's sparse map, dropping zero-count buckets.
            # Validate the shape AND every element before filtering: a str/dict
            # is iterable (enumerate would build a bucket per character/key),
            # and filtering on truthiness first would silently drop a falsey
            # malformed entry (None/False/0.0) while keeping a truthy one that
            # coerces to zero ("0"), which would violate the record's
            # zero-buckets-omitted invariant. ``type(...) is int`` rather than
            # isinstance so bools -- an int subclass -- are rejected too.
            raw_histogram = payload["acceptance_histogram"]
            if not isinstance(raw_histogram, list) or not all(
                type(count) is int for count in raw_histogram
            ):
                raise TypeError(
                    "acceptance_histogram must be a list of ints, got "
                    f"{raw_histogram!r}"
                )
            # Length is num_spec_tokens + 1 (one bucket per accepted count from
            # 0..k). Enforcing it rejects a payload whose bucket indices exceed
            # the draft budget -- j > k is physically impossible, yet satisfies
            # the record's arithmetic validators, so it would otherwise be
            # reported as a real acceptance length. Doubles as a tripwire if the
            # still-unmerged upstream PR changes the histogram shape.
            # Optional on the record, so only enforce when the server sent it.
            num_spec_tokens = payload.get("num_spec_tokens")
            if (
                num_spec_tokens is not None
                and len(raw_histogram) != num_spec_tokens + 1
            ):
                raise ValueError(
                    f"acceptance_histogram has {len(raw_histogram)} buckets, "
                    f"expected num_spec_tokens + 1 = {num_spec_tokens + 1}"
                )
            histogram = {
                j: count for j, count in enumerate(raw_histogram) if count != 0
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
            # Rebind to a normal local so the lazy lambda can reference it (the
            # ``except ... as e`` name is cleared at block exit).
            error = e
            _logger.warning(
                lambda: (
                    f"Ignoring malformed vLLM spec-decode payload {payload!r}: {error!r}"
                )
            )
            return None
