# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The engine-neutral parts of adapting a spec-decode payload.

An adapter exists to absorb one engine's on-the-wire shape, but two pieces of
that work are identical for every engine: locating the captured payload among a
request's responses, and converting the dense acceptance histogram every engine
sends into the sparse map the neutral record stores. Both are subtle enough that
duplicating them across adapters would let the copies drift, so they live here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from aiperf.common.models import ParsedResponse


def find_spec_decode_payload(
    responses: list[ParsedResponse],
) -> dict[str, Any] | None:
    """Return the last non-empty ``spec_decode_stats`` payload across responses.

    An engine attaches the payload once per request: on the body non-streaming,
    or on a trailing chunk streaming. Walking from the end mirrors
    ``find_last_non_empty_usage`` and tolerates either layout.
    """
    for response in reversed(responses):
        stats = response.spec_decode_stats
        if stats:
            return stats
    return None


def inflate_dense_histogram(
    raw_histogram: Any, num_spec_tokens: int | None
) -> dict[int, int]:
    """Validate a dense on-the-wire histogram and inflate it to the sparse map.

    Raises ``TypeError`` or ``ValueError`` on a malformed histogram; callers
    catch and degrade to ``None`` so one bad payload cannot abort a run.
    """
    # Validate the shape AND every element before filtering: a str/dict is
    # iterable (enumerate would build a bucket per character/key), and filtering
    # on truthiness first would silently drop a falsey malformed entry
    # (None/False/0.0) while keeping a truthy one that coerces to zero ("0"),
    # which would violate the record's zero-buckets-omitted invariant.
    # ``type(...) is int`` rather than isinstance so bools -- an int subclass --
    # are rejected too.
    if not isinstance(raw_histogram, list) or not all(
        type(count) is int for count in raw_histogram
    ):
        raise TypeError(
            f"acceptance_histogram must be a list of ints, got {raw_histogram!r}"
        )
    # Length is num_spec_tokens + 1 (one bucket per accepted count from 0..k).
    # Enforcing it rejects a payload whose bucket indices exceed the draft
    # budget -- j > k is physically impossible, yet satisfies the record's
    # arithmetic validators, so it would otherwise be reported as a real
    # acceptance length. Optional on the record, so only enforce it when the
    # server sent a fixed bound; engines that draft variable-length report None.
    if num_spec_tokens is not None and len(raw_histogram) != num_spec_tokens + 1:
        raise ValueError(
            f"acceptance_histogram has {len(raw_histogram)} buckets, "
            f"expected num_spec_tokens + 1 = {num_spec_tokens + 1}"
        )
    return {j: count for j, count in enumerate(raw_histogram) if count != 0}
